"""Table-plane calibration from the printed ArUco sheet.

Table frame: origin at the top-left corner of the A4 page (as printed, landscape),
x along the long edge to the right, y down the page, units = cm.
This module maps image pixels <-> table frame, per video frame.

Typical use:
    layout = PageLayout.scaled(measured_marker_cm=5.0)
    calib = calibrate_video(frames, layout)          # auto-detects fixed vs moving camera
    xy_cm = calib.to_table(pixel_track)              # (N,2) pixels -> (N,2) cm
    preview = draw_overlay(frames[0], calib.H[0], layout)
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

aruco = cv2.aruco


# --------------------------------------------------------------------------- layout
@dataclass(frozen=True)
class PageLayout:
    marker_size: float = 5.0      # cm
    margin: float = 1.5           # cm, page edge -> marker edge
    page_w: float = 29.7          # cm, landscape A4
    page_h: float = 21.0
    dictionary: int = aruco.DICT_4X4_50

    @classmethod
    def scaled(cls, measured_marker_cm: float, **kw) -> "PageLayout":
        """If the printer rescaled the page, everything scales by the same factor.
        Measure one marker with a ruler and pass it here."""
        base = cls(**kw)
        k = measured_marker_cm / base.marker_size
        return replace(base, marker_size=base.marker_size * k, margin=base.margin * k,
                       page_w=base.page_w * k, page_h=base.page_h * k)

    @classmethod
    def from_dict(cls, d: dict) -> "PageLayout":
        d = dict(d)
        measured = d.pop("measured_marker_cm", None)
        return cls.scaled(measured, **d) if measured else cls(**d)

    def marker_origins(self) -> dict[int, tuple[float, float]]:
        """Top-left corner of each marker in the table frame (cm)."""
        s, m, W, H = self.marker_size, self.margin, self.page_w, self.page_h
        return {0: (m, m), 1: (W - m - s, m), 2: (W - m - s, H - m - s), 3: (m, H - m - s)}

    def marker_corners(self, marker_id: int) -> np.ndarray:
        """4 corners in ArUco order (TL, TR, BR, BL), table frame, shape (4,2)."""
        x, y = self.marker_origins()[marker_id]
        s = self.marker_size
        return np.float32([(x, y), (x + s, y), (x + s, y + s), (x, y + s)])

    def page_outline(self) -> np.ndarray:
        return np.float32([(0, 0), (self.page_w, 0), (self.page_w, self.page_h), (0, self.page_h)])


def make_marker_page(layout: PageLayout = PageLayout(), dpi: int = 300) -> np.ndarray:
    """Greyscale image of the printable sheet. Save with PIL at `dpi` and print at 100%."""
    px = dpi / 2.54
    page = np.full((round(layout.page_h * px), round(layout.page_w * px)), 255, np.uint8)
    s = round(layout.marker_size * px)
    d = aruco.getPredefinedDictionary(layout.dictionary)
    for i, (x, y) in layout.marker_origins().items():
        X, Y = round(x * px), round(y * px)
        page[Y:Y + s, X:X + s] = aruco.generateImageMarker(d, i, s)
    return page


# --------------------------------------------------------------------------- detection
def make_detector(layout: PageLayout) -> aruco.ArucoDetector:
    params = aruco.DetectorParameters()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX   # sub-pixel corners
    return aruco.ArucoDetector(aruco.getPredefinedDictionary(layout.dictionary), params)


def detect(frame: np.ndarray, detector: aruco.ArucoDetector, layout: PageLayout) -> dict[int, np.ndarray]:
    """Marker id -> (4,2) pixel corners, only for ids that belong to the sheet."""
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return {}
    valid = layout.marker_origins()
    return {int(i): c.reshape(4, 2) for c, i in zip(corners, ids.ravel()) if int(i) in valid}


def fit_homography(dets: dict[int, np.ndarray], layout: PageLayout,
                   min_markers: int = 2) -> tuple[np.ndarray | None, float]:
    """Image -> table homography from detected markers. Returns (H, mean reprojection error in cm).
    One marker is technically enough (4 points) but extrapolates badly across the page, hence min 2."""
    if len(dets) < min_markers:
        return None, np.nan
    img = np.concatenate([dets[i] for i in dets]).astype(np.float32)
    world = np.concatenate([layout.marker_corners(i) for i in dets])
    H, _ = cv2.findHomography(img, world, cv2.RANSAC, ransacReprojThreshold=0.5)
    if H is None:
        return None, np.nan
    err = np.linalg.norm(apply_h(H, img) - world, axis=1).mean()
    return H / H[2, 2], float(err)


# --------------------------------------------------------------------------- geometry
def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply homography to points. NaN points stay NaN.
    H (3,3)   with pts (...,2)          -> same H for all points
    H (N,3,3) with pts (N,2) or (N,K,2) -> one H per frame"""
    pts = np.asarray(pts, dtype=float)
    ph = np.concatenate([pts, np.ones(pts.shape[:-1] + (1,))], axis=-1)
    if H.ndim == 2:
        out = ph @ H.T
    elif pts.ndim == 2:
        out = np.einsum("nij,nj->ni", H, ph)
    else:
        out = np.einsum("nij,nkj->nki", H, ph)
    return out[..., :2] / out[..., 2:3]


@dataclass
class Calibration:
    H: np.ndarray          # (N,3,3) image -> table (cm), one per video frame
    fitted: np.ndarray     # (N,) True where H came from that frame's own detection
    err_cm: np.ndarray     # (N,) reprojection error of the fit (NaN where interpolated)
    static: bool           # True if camera judged fixed and a single pooled H is used
    motion_px: float       # how much marker corners moved over the clip (95th pct, px)

    def to_table(self, pts_px: np.ndarray) -> np.ndarray:
        """(N,2) or (N,K,2) pixel tracks -> table cm, using each frame's own H."""
        return apply_h(self.H, pts_px)

    def to_image(self, pts_cm: np.ndarray) -> np.ndarray:
        return apply_h(np.linalg.inv(self.H), pts_cm)

    def summary(self) -> str:
        mode = "static (pooled)" if self.static else "per-frame"
        return (f"{mode} | marker motion {self.motion_px:.1f}px | "
                f"fitted {self.fitted.mean() * 100:.0f}% of frames | "
                f"reproj err median {np.nanmedian(self.err_cm):.2f}cm")


def calibrate_video(frames, layout: PageLayout, step: int = 1, static: bool | None = None,
                    static_px: float = 3.0, min_markers: int = 2) -> Calibration:
    """Homography for every frame of a clip.

    static=None  -> decide automatically: if every marker corner stays within `static_px`
                    of its median position, the camera is fixed and we pool all detections
                    into one H (robust to the hand occluding markers).
    static=False -> fit per sampled frame, linearly interpolate H between fits.
    """
    n = len(frames)
    detector = make_detector(layout)
    sampled = list(range(0, n, step))
    dets = {i: detect(frames[i], detector, layout) for i in sampled}

    # how much did each marker move across the clip?
    per_marker = {}
    for d in dets.values():
        for mid, c in d.items():
            per_marker.setdefault(mid, []).append(c)
    if not per_marker:
        raise RuntimeError("No ArUco markers detected in any frame.")
    motion = 0.0
    for obs in per_marker.values():
        obs = np.stack(obs)                                         # (T,4,2)
        dev = np.linalg.norm(obs - np.median(obs, axis=0), axis=-1)
        motion = max(motion, float(np.percentile(dev, 95)))
    if static is None:
        static = motion < static_px

    if static:
        pooled = {mid: np.median(np.stack(obs), axis=0) for mid, obs in per_marker.items()}
        H, err = fit_homography(pooled, layout, min_markers)
        if H is None:
            raise RuntimeError(f"Only {len(pooled)} marker(s) ever seen; need {min_markers}.")
        return Calibration(H=np.repeat(H[None], n, axis=0), fitted=np.ones(n, bool),
                           err_cm=np.full(n, err), static=True, motion_px=motion)

    Hs = np.full((n, 3, 3), np.nan)
    err = np.full(n, np.nan)
    for i in sampled:
        H, e = fit_homography(dets[i], layout, min_markers)
        if H is not None:
            Hs[i], err[i] = H, e
    fitted = ~np.isnan(Hs[:, 0, 0])
    if not fitted.any():
        raise RuntimeError(f"No frame had >= {min_markers} markers visible.")
    # Fill gaps: element-wise linear interpolation of normalised H (fine for smooth hand-held motion);
    # before the first / after the last fit, hold the nearest one.
    idx = np.flatnonzero(fitted)
    flat = Hs.reshape(n, 9)
    for k in range(9):
        flat[:, k] = np.interp(np.arange(n), idx, flat[idx, k])
    return Calibration(H=flat.reshape(n, 3, 3), fitted=fitted, err_cm=err,
                       static=False, motion_px=motion)


# --------------------------------------------------------------------------- visual check
def draw_overlay(frame: np.ndarray, H: np.ndarray, layout: PageLayout,
                 grid_cm: float = 5.0) -> np.ndarray:
    """Project the page outline, marker squares and a cm grid back onto the image.
    If calibration is right, the green lines sit exactly on the printed markers."""
    out = frame.copy()
    Hinv = np.linalg.inv(H)

    def poly(pts_cm, colour, t):
        p = apply_h(Hinv, pts_cm).round().astype(np.int32)
        cv2.polylines(out, [p], True, colour, t, cv2.LINE_AA)

    for x in np.arange(0, layout.page_w + 1e-6, grid_cm):
        poly(np.float32([(x, 0), (x, layout.page_h)]), (180, 180, 255), 1)
    for y in np.arange(0, layout.page_h + 1e-6, grid_cm):
        poly(np.float32([(0, y), (layout.page_w, y)]), (180, 180, 255), 1)
    poly(layout.page_outline(), (0, 200, 0), 2)
    for mid in layout.marker_origins():
        poly(layout.marker_corners(mid), (0, 255, 0), 2)
    return out