"""Table-plane calibration from the printed ArUco sheet.

Maps image pixels <-> "table frame" coordinates, separately for every video frame.

Table frame: origin at the top-left corner of the A4 page (as printed, landscape),
x along the long edge to the right, y down the page, units = cm.

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
    """Geometry of the printed marker sheet. All lengths in cm.

    Fields:
        marker_size: float -- side length of each square marker
        margin:      float -- gap between the page edge and the nearest marker edge
        page_w:      float -- page width (long edge, landscape A4 = 29.7)
        page_h:      float -- page height (short edge, landscape A4 = 21.0)
        dictionary:  int   -- OpenCV ArUco dictionary id the markers were generated from
    """
    marker_size: float = 5.0
    margin: float = 1.5
    page_w: float = 29.7
    page_h: float = 21.0
    dictionary: int = aruco.DICT_4X4_50

    @classmethod
    def scaled(cls, measured_marker_cm: float, **kw) -> "PageLayout":
        """Layout rescaled to match a ruler measurement of one printed marker.
        Why: printers often shrink pages slightly ("fit to page"); every length scales by the same factor.

        Inputs:
            measured_marker_cm: float -- measured side of one marker, cm
            **kw: optional overrides of the default fields before scaling
        Returns:
            PageLayout with all lengths multiplied by measured_marker_cm / marker_size
        """
        base = cls(**kw)
        k = measured_marker_cm / base.marker_size
        return replace(base, marker_size=base.marker_size * k, margin=base.margin * k,
                       page_w=base.page_w * k, page_h=base.page_h * k)

    @classmethod
    def from_dict(cls, d: dict) -> "PageLayout":
        """Layout from a config dictionary (e.g. loaded from configs/default.yaml).

        Inputs:
            d: dict -- any of the field names above, plus optional "measured_marker_cm"
        Returns:
            PageLayout (scaled if "measured_marker_cm" is present)
        """
        d = dict(d)
        measured = d.pop("measured_marker_cm", None)
        return cls.scaled(measured, **d) if measured else cls(**d)

    def marker_origins(self) -> dict[int, tuple[float, float]]:
        """Top-left corner of each marker in the table frame.

        Returns:
            dict {marker id (int): (x_cm, y_cm)} for ids 0 (top-left), 1 (top-right),
            2 (bottom-right), 3 (bottom-left)
        """
        s, m, W, H = self.marker_size, self.margin, self.page_w, self.page_h
        return {0: (m, m), 1: (W - m - s, m), 2: (W - m - s, H - m - s), 3: (m, H - m - s)}

    def marker_corners(self, marker_id: int) -> np.ndarray:
        """The 4 corners of one marker in the table frame, in the order ArUco reports them.

        Inputs:
            marker_id: int -- 0..3
        Returns:
            np.ndarray float32, shape (4,2), cm, order top-left, top-right, bottom-right, bottom-left
        """
        x, y = self.marker_origins()[marker_id]
        s = self.marker_size
        return np.float32([(x, y), (x + s, y), (x + s, y + s), (x, y + s)])

    def page_outline(self) -> np.ndarray:
        """The 4 page corners in the table frame.

        Returns:
            np.ndarray float32, shape (4,2), cm, clockwise from top-left
        """
        return np.float32([(0, 0), (self.page_w, 0), (self.page_w, self.page_h), (0, self.page_h)])


def make_marker_page(layout: PageLayout = PageLayout(), dpi: int = 300) -> np.ndarray:
    """Image of the printable marker sheet. Save it with PIL at the same dpi and print at 100%.

    Inputs:
        layout: PageLayout -- sheet geometry
        dpi:    int        -- print resolution, dots per inch
    Returns:
        np.ndarray uint8, shape (page_h_px, page_w_px), greyscale (255 = white)
    """
    px = dpi / 2.54                                   # pixels per cm
    page = np.full((round(layout.page_h * px), round(layout.page_w * px)), 255, np.uint8)
    s = round(layout.marker_size * px)
    d = aruco.getPredefinedDictionary(layout.dictionary)
    for i, (x, y) in layout.marker_origins().items():
        X, Y = round(x * px), round(y * px)
        page[Y:Y + s, X:X + s] = aruco.generateImageMarker(d, i, s)
    return page


# --------------------------------------------------------------------------- detection
def make_detector(layout: PageLayout) -> aruco.ArucoDetector:
    """ArUco detector for the sheet's dictionary.
    Why sub-pixel refinement: corner accuracy directly sets the accuracy of every cm value downstream.

    Inputs:
        layout: PageLayout
    Returns:
        cv2.aruco.ArucoDetector
    """
    params = aruco.DetectorParameters()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    return aruco.ArucoDetector(aruco.getPredefinedDictionary(layout.dictionary), params)


def detect(frame: np.ndarray, detector: aruco.ArucoDetector, layout: PageLayout) -> dict[int, np.ndarray]:
    """Find the sheet's markers in one image.

    Inputs:
        frame:    np.ndarray uint8, shape (H,W,3) RGB or (H,W) greyscale
        detector: cv2.aruco.ArucoDetector from make_detector
        layout:   PageLayout -- used to ignore markers that are not part of the sheet
    Returns:
        dict {marker id (int): np.ndarray float32 (4,2) pixel corners}; empty dict if none found
    """
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return {}
    valid = layout.marker_origins()
    return {int(i): c.reshape(4, 2) for c, i in zip(corners, ids.ravel()) if int(i) in valid}


def fit_homography(dets: dict[int, np.ndarray], layout: PageLayout,
                   min_markers: int = 2) -> tuple[np.ndarray | None, float]:
    """Homography (3x3 perspective mapping) from image pixels to table cm, fitted to detected markers.
    Why min 2 markers: one marker gives the 4 points needed, but its corners are close together,
    so small errors get magnified across the rest of the page.

    Inputs:
        dets:        dict {marker id: (4,2) pixel corners}, as returned by detect
        layout:      PageLayout
        min_markers: int -- refuse to fit with fewer markers than this
    Returns:
        (H, err_cm):
            H:      np.ndarray float64 (3,3) image -> table, normalised so H[2,2] = 1; None if fit failed
            err_cm: float -- mean distance between detected corners mapped to cm and their true
                    positions (reprojection error); NaN if fit failed
    """
    if len(dets) < min_markers:
        return None, np.nan
    img = np.concatenate([dets[i] for i in dets]).astype(np.float32)
    world = np.concatenate([layout.marker_corners(i) for i in dets])
    # RANSAC ignores the odd badly-detected corner instead of letting it skew the fit
    H, _ = cv2.findHomography(img, world, cv2.RANSAC, ransacReprojThreshold=0.5)
    if H is None:
        return None, np.nan
    err = np.linalg.norm(apply_h(H, img) - world, axis=1).mean()
    return H / H[2, 2], float(err)


# --------------------------------------------------------------------------- geometry
def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Map points through a homography. NaN points stay NaN.

    Inputs (two modes):
        H (3,3)   with pts (...,2)          -> the same H for every point
        H (N,3,3) with pts (N,2) or (N,K,2) -> H[i] applied to the point(s) of frame i
    Returns:
        np.ndarray float64, same shape as pts
    """
    pts = np.asarray(pts, dtype=float)
    # append a 1 to each (x,y) -> (x,y,1), multiply by H, then divide by the 3rd value
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
    """Calibration of one video clip with N frames.

    Fields:
        H:         np.ndarray (N,3,3) -- image -> table (cm) homography for every frame
        fitted:    np.ndarray bool (N,) -- True where H came from that frame's own markers,
                   False where it was interpolated from neighbouring frames
        err_cm:    np.ndarray float (N,) -- reprojection error of each fit, cm (NaN if interpolated)
        static:    bool -- True if the camera was judged fixed and one pooled H is used for all frames
        motion_px: float -- how far marker corners moved during the clip (95th percentile), pixels
    """
    H: np.ndarray
    fitted: np.ndarray
    err_cm: np.ndarray
    static: bool
    motion_px: float

    def to_table(self, pts_px: np.ndarray) -> np.ndarray:
        """Pixel track -> table cm, using each frame's own H.

        Inputs:
            pts_px: np.ndarray (N,2) one point per frame, or (N,K,2) K points per frame, pixels
        Returns:
            np.ndarray float64, same shape, cm
        """
        return apply_h(self.H, pts_px)

    def to_image(self, pts_cm: np.ndarray) -> np.ndarray:
        """Table cm -> pixels, per frame (inverse of to_table).

        Inputs:
            pts_cm: np.ndarray (N,2) or (N,K,2), cm
        Returns:
            np.ndarray float64, same shape, pixels
        """
        return apply_h(np.linalg.inv(self.H), pts_cm)

    def summary(self) -> str:
        """One-line health check: mode, camera motion, share of frames fitted, median error."""
        mode = "static (pooled)" if self.static else "per-frame"
        return (f"{mode} | marker motion {self.motion_px:.1f}px | "
                f"fitted {self.fitted.mean() * 100:.0f}% of frames | "
                f"reproj err median {np.nanmedian(self.err_cm):.2f}cm")


def calibrate_video(frames, layout: PageLayout, step: int = 1, static: bool | None = None,
                    static_px: float = 3.0, min_markers: int = 2) -> Calibration:
    """Homography for every frame of a clip.

    Inputs:
        frames:      sequence of N images, each np.ndarray uint8 (H,W,3) RGB
        layout:      PageLayout
        step:        int   -- detect markers on every step-th frame only (speed); others are interpolated
        static:      None  -> decide automatically from how much the markers move
                     True  -> fixed camera: pool all detections into one H
                              (why: robust when the hand covers markers in some frames)
                     False -> moving camera: fit each sampled frame, interpolate H in between
        static_px:   float -- auto mode calls the camera fixed if marker corners move less than this, pixels
        min_markers: int   -- minimum markers needed to fit a frame
    Returns:
        Calibration with H of shape (N,3,3)
    Raises:
        RuntimeError if no usable markers are found anywhere in the clip
    """
    n = len(frames)
    detector = make_detector(layout)
    sampled = list(range(0, n, step))
    dets = {i: detect(frames[i], detector, layout) for i in sampled}

    # How much did each marker move across the clip? Collect every sighting per marker id.
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
        # 95th percentile, not max: one misdetection shouldn't flip the decision
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
    # Fill gaps: element-wise linear interpolation of the normalised H (fine for smooth hand-held motion);
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
    """Draw the page outline, marker squares and a cm grid back onto the image.
    Why: if calibration is right, the green lines sit exactly on the printed markers.

    Inputs:
        frame:   np.ndarray uint8 (H,W,3) RGB
        H:       np.ndarray (3,3) image -> table homography for this frame
        layout:  PageLayout
        grid_cm: float -- grid spacing, cm
    Returns:
        np.ndarray uint8 (H,W,3) RGB, a copy of frame with the overlay drawn
    """
    out = frame.copy()
    Hinv = np.linalg.inv(H)

    def poly(pts_cm, colour, t):
        # draw a closed polygon given in cm, converted to this frame's pixels
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