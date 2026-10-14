"""Hand + object tracking: phone video -> where the "gripper" is on the table, and when it is closed.

Pipeline for one clip:
    1. MediaPipe finds 21 hand landmarks (pixels) in every frame.
    2. Short gaps are filled in, jitter is smoothed out.
    3. The pinch point (midway between thumb tip and index tip) is our stand-in for the robot gripper.
    4. A coloured object is found by colour thresholding.
    5. Everything is converted from pixels to table cm with the calibration from calib.py.
    6. Grasp timing comes from the OBJECT: the hand is "closed" while the object is being carried.

Typical use:
    calib = calibrate_video(frames, layout)
    track = process_video(frames, fps, calib, colour="blue", layout=layout)
    track.save("data/processed/clip01.npz")
    plot_track(track)
"""
from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass, fields

import cv2
import numpy as np

from .calib import Calibration, PageLayout

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")

# MediaPipe landmark numbers we use (full map: 0 wrist, 1-4 thumb, 5-8 index, 9-12 middle, ...)
WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP, MIDDLE_MCP = 0, 4, 5, 8, 9

# Colour ranges in OpenCV HSV (hue 0-179, saturation 0-255, value 0-255), as a list of (low, high) pairs.
# Red wraps around hue 0, so it needs two ranges. Skin is reddish too, so blue/green objects are safer.
COLOURS = {
    "red":    [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (179, 255, 255))],
    "orange": [((10, 120, 100), (22, 255, 255))],
    "yellow": [((22, 100, 100), (35, 255, 255))],
    "green":  [((40, 80, 50), (85, 255, 255))],
    "blue":   [((95, 120, 50), (130, 255, 255))],
}


# --------------------------------------------------------------------------- 1. hand landmarks
def ensure_model(path: str = "hand_landmarker.task") -> str:
    """Download the MediaPipe hand model the first time it is needed.

    Inputs:
        path: str -- where the model file should live
    Returns:
        str -- the same path, now guaranteed to exist
    """
    if not os.path.exists(path):
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


def run_hand_landmarker(frames, fps: float, model_path: str | None = None,
                        min_conf: float = 0.3) -> np.ndarray:
    """Run MediaPipe on every frame and return the 21 hand landmarks.

    Inputs:
        frames:     sequence of N images, each np.ndarray uint8 (H,W,3) RGB
        fps:        float -- video frame rate, used to give MediaPipe a timestamp per frame
        model_path: str or None -- MediaPipe .task file (downloaded if missing)
        min_conf:   float in [0,1] -- detection/tracking confidence threshold
                    (why 0.3, below the 0.5 default: recovers borderline frames, e.g. hand at the edge)
    Returns:
        np.ndarray float64 (N,21,2) -- landmark (x,y) in pixels; all NaN in frames with no hand
    """
    import mediapipe as mp                       # imported here so the sim runtime doesn't need it
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision

    opts = vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=ensure_model(model_path or "hand_landmarker.task")),
        running_mode=vision.RunningMode.VIDEO,   # uses the previous frame to track, smoother than per-image
        num_hands=1,
        min_hand_detection_confidence=min_conf,
        min_hand_presence_confidence=min_conf,
        min_tracking_confidence=min_conf)

    n = len(frames)
    h, w = frames[0].shape[:2]
    lm = np.full((n, 21, 2), np.nan)
    with vision.HandLandmarker.create_from_options(opts) as detector:
        for i, f in enumerate(frames):
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(f))
            res = detector.detect_for_video(img, int(round(i * 1000 / fps)))   # timestamp in ms
            if res.hand_landmarks:
                lm[i] = [[p.x * w, p.y * h] for p in res.hand_landmarks[0]]    # normalised 0-1 -> pixels
    return lm


# --------------------------------------------------------------------------- 2. clean-up
def fill_gaps(x: np.ndarray, max_gap: int) -> np.ndarray:
    """Draw straight lines across short runs of missing (NaN) frames.
    Why only short ones: across a long gap we would be inventing motion. Missing frames at the very
    start/end also stay NaN, since there is nothing on one side to interpolate from.

    Inputs:
        x:       np.ndarray (N, ...) -- a value per frame, any trailing shape (e.g. (N,2) or (N,21,2));
                 a frame counts as missing if any of its values is NaN
        max_gap: int -- longest gap to fill, in frames
    Returns:
        np.ndarray float64, same shape as x
    """
    x = np.array(x, dtype=float)
    flat = x.reshape(len(x), -1)
    ok = ~np.isnan(flat).any(axis=1)
    good = np.flatnonzero(ok)
    if len(good) < 2:
        return x
    filled = np.empty_like(flat)
    for k in range(flat.shape[1]):
        filled[:, k] = np.interp(np.arange(len(x)), good, flat[good, k])
    # np.interp filled every gap; keep only the short interior ones
    keep = ok.copy()
    for a, b in zip(good[:-1], good[1:]):
        if 1 < b - a <= max_gap + 1:
            keep[a + 1:b] = True
    filled[~keep] = np.nan
    return filled.reshape(x.shape)


def smooth(x: np.ndarray, win: int) -> np.ndarray:
    """Centred moving average to remove frame-to-frame jitter. NaNs are skipped, not spread.

    Inputs:
        x:   np.ndarray (N, ...) -- a value per frame, any trailing shape
        win: int -- window length in frames (e.g. 5 = 2 before, the frame itself, 2 after)
    Returns:
        np.ndarray float64, same shape as x; frames that were NaN stay NaN
    """
    if win <= 1:
        return np.array(x, dtype=float)
    x = np.array(x, dtype=float)
    flat = x.reshape(len(x), -1)
    out = np.empty_like(flat)
    kernel = np.ones(win)
    for k in range(flat.shape[1]):
        v = flat[:, k]
        m = ~np.isnan(v)
        # sum of valid values in each window / number of valid values in each window
        num = np.convolve(np.where(m, v, 0.0), kernel, mode="same")
        den = np.convolve(m.astype(float), kernel, mode="same")
        out[:, k] = np.where(m, num / np.maximum(den, 1e-9), np.nan)
    return out.reshape(x.shape)


# --------------------------------------------------------------------------- 4. object
def track_object(frames, hsv_ranges, min_area_px: int = 150) -> np.ndarray:
    """Centre of the largest blob of the given colour in each frame.

    Inputs:
        frames:      sequence of N images, each np.ndarray uint8 (H,W,3) RGB
        hsv_ranges:  list of ((h,s,v) low, (h,s,v) high) tuples, OpenCV HSV scale (see COLOURS)
        min_area_px: int -- blobs smaller than this many pixels are ignored as noise
    Returns:
        np.ndarray float64 (N,2) -- blob centre (x,y) in pixels; NaN where not found
    """
    out = np.full((len(frames), 2), np.nan)
    kernel = np.ones((5, 5), np.uint8)
    for i, f in enumerate(frames):
        hsv = cv2.cvtColor(f, cv2.COLOR_RGB2HSV)
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in hsv_ranges:
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)       # remove speckle noise
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < min_area_px:
            continue
        m = cv2.moments(c)
        out[i] = (m["m10"] / m["m00"], m["m01"] / m["m00"])         # area-weighted centre
    return out


def hsv_at(frame: np.ndarray, x: int, y: int, r: int = 5) -> np.ndarray:
    """Typical colour around one pixel. Use it to check/pick an HSV range for your object.

    Inputs:
        frame: np.ndarray uint8 (H,W,3) RGB
        x, y:  int -- pixel position (x right, y down)
        r:     int -- half-size of the square patch sampled, pixels
    Returns:
        np.ndarray (3,) -- median (h,s,v) of the patch, OpenCV HSV scale
    """
    patch = cv2.cvtColor(frame[y - r:y + r + 1, x - r:x + r + 1], cv2.COLOR_RGB2HSV)
    return np.median(patch.reshape(-1, 3), axis=0)


# --------------------------------------------------------------------------- 6. grasp timing
def detect_grasp(obj_cm: np.ndarray, pinch_cm: np.ndarray, fps: float, move_cm: float = 2.0,
                 near_cm: float = 6.0, pad_s: float = 0.25):
    """Decide when the gripper is closed, from the object's motion. Assumes ONE pick-and-place per clip.

    The object rests at a start position, gets carried, then rests at an end position.
    A frame is 'carried' if the object is away from both rest positions AND close to the pinch point.
    Why the padding: fingers close slightly before the object starts moving, and open slightly after it stops.

    Inputs:
        obj_cm:   np.ndarray (N,2) -- object position per frame, table cm (NaN where not seen)
        pinch_cm: np.ndarray (N,2) -- pinch point per frame, table cm (NaN where not seen)
        fps:      float -- frame rate
        move_cm:  float -- distance from a rest position that counts as "moved", cm
        near_cm:  float -- max object-to-pinch distance that counts as "in the hand", cm
        pad_s:    float -- close this long before the first carried frame / open this long after the last, s
    Returns:
        (grasp, t_close, t_open):
            grasp:   np.ndarray bool (N,) -- True = gripper closed
            t_close: int or None -- first closed frame (None if nothing was carried)
            t_open:  int or None -- last closed frame
    """
    n = len(obj_cm)
    seen = np.flatnonzero(~np.isnan(obj_cm).any(axis=1))
    none = (np.zeros(n, bool), None, None)
    if len(seen) < 2:
        return none
    half_s = max(int(fps * 0.5), 1)                          # frames in half a second
    # median, not mean: one bad detection can't drag the rest position off
    start = np.nanmedian(obj_cm[seen[:half_s]], axis=0)      # where it sat at the beginning
    end = np.nanmedian(obj_cm[seen[-half_s:]], axis=0)       # where it sat at the end
    with np.errstate(invalid="ignore"):                      # NaN comparisons are simply False
        away = (np.linalg.norm(obj_cm - start, axis=1) > move_cm) & \
               (np.linalg.norm(obj_cm - end, axis=1) > move_cm)
        near = np.linalg.norm(obj_cm - pinch_cm, axis=1) < near_cm
    carried = np.flatnonzero(away & near)
    if len(carried) == 0:
        return none
    pad = int(round(pad_s * fps))
    t_close, t_open = max(carried[0] - pad, 0), min(carried[-1] + pad, n - 1)
    grasp = np.zeros(n, bool)
    grasp[t_close:t_open + 1] = True
    return grasp, int(t_close), int(t_open)


# --------------------------------------------------------------------------- result container
@dataclass
class HandTrack:
    """Everything tracked in one clip of N frames.

    Fields:
        fps:          float
        landmarks_px: np.ndarray (N,21,2) -- cleaned hand landmarks, pixels
        detected:     np.ndarray bool (N,) -- True where MediaPipe actually saw the hand (before gap filling)
        pinch_px:     np.ndarray (N,2) -- pinch point, pixels
        pinch_cm:     np.ndarray (N,2) -- pinch point, table cm (the gripper stand-in)
        aperture_cm:  np.ndarray (N,) -- thumb-index distance, cm (cross-check for the grasp)
        obj_px:       np.ndarray (N,2) -- object centre, pixels
        obj_cm:       np.ndarray (N,2) -- object centre, table cm
        grasp:        np.ndarray bool (N,) -- True = gripper should be closed
        t_close:      int -- first closed frame, -1 if no grasp found
                      (why -1 and not None: .npz files can't store None)
        t_open:       int -- last closed frame, -1 if no grasp found
    """
    fps: float
    landmarks_px: np.ndarray
    detected: np.ndarray
    pinch_px: np.ndarray
    pinch_cm: np.ndarray
    aperture_cm: np.ndarray
    obj_px: np.ndarray
    obj_cm: np.ndarray
    grasp: np.ndarray
    t_close: int
    t_open: int

    def save(self, path: str) -> None:
        """Write all fields to one compressed .npz file.

        Inputs:
            path: str -- e.g. "data/processed/clip01.npz"
        """
        np.savez_compressed(path, **{f.name: getattr(self, f.name) for f in fields(self)})

    @classmethod
    def load(cls, path: str) -> "HandTrack":
        """Read a HandTrack back from a .npz file written by save.

        Inputs:
            path: str
        Returns:
            HandTrack
        """
        d = np.load(path)
        kw = {f.name: d[f.name] for f in fields(cls)}
        # .npz stores single numbers as 0-d arrays; turn them back into plain float/int
        kw["fps"] = float(kw["fps"]); kw["t_close"] = int(kw["t_close"]); kw["t_open"] = int(kw["t_open"])
        return cls(**kw)

    def summary(self) -> str:
        """One-line health check: frames, % hand seen, % object seen, grasp frames."""
        ok = ~np.isnan(self.pinch_cm[:, 0])
        g = f"close@{self.t_close} open@{self.t_open}" if self.t_close >= 0 else "NO GRASP FOUND"
        return (f"{len(self.detected)} frames | hand seen {self.detected.mean() * 100:.0f}% "
                f"(usable after gap-fill {ok.mean() * 100:.0f}%) | "
                f"object seen {(~np.isnan(self.obj_cm[:, 0])).mean() * 100:.0f}% | {g}")


# --------------------------------------------------------------------------- whole clip
def process_video(frames, fps: float, calib: Calibration, colour: str = "blue", hsv_ranges=None,
                  layout: PageLayout | None = None, max_gap_s: float = 0.3, smooth_win: int = 5,
                  model_path: str | None = None) -> HandTrack:
    """Run the whole tracking pipeline (stages 1-6) on one clip.

    Inputs:
        frames:     sequence of N images, each np.ndarray uint8 (H,W,3) RGB
        fps:        float -- frame rate
        calib:      Calibration of this clip (from calib.calibrate_video)
        colour:     str -- a key of COLOURS; ignored if hsv_ranges is given
        hsv_ranges: optional custom list of ((h,s,v) low, (h,s,v) high) tuples
        layout:     PageLayout or None -- if given, colour blobs more than 5 cm off the sheet are discarded
        max_gap_s:  float -- longest hand-tracking gap to fill, seconds
        smooth_win: int -- smoothing window for the landmarks, frames
        model_path: str or None -- MediaPipe model file
    Returns:
        HandTrack
    """
    lm_raw = run_hand_landmarker(frames, fps, model_path)
    detected = ~np.isnan(lm_raw[:, 0, 0])
    lm = smooth(fill_gaps(lm_raw, int(max_gap_s * fps)), smooth_win)

    # pinch point = midway between thumb tip and index tip, where a held object sits
    pinch_px = 0.5 * (lm[:, THUMB_TIP] + lm[:, INDEX_TIP])
    lm_cm = calib.to_table(lm)
    pinch_cm = calib.to_table(pinch_px)
    aperture_cm = np.linalg.norm(lm_cm[:, THUMB_TIP] - lm_cm[:, INDEX_TIP], axis=1)

    obj_px = smooth(track_object(frames, hsv_ranges or COLOURS[colour]), 3)
    obj_cm = calib.to_table(obj_px)
    if layout is not None:                     # ignore colour blobs far off the sheet
        x, y = obj_cm[:, 0], obj_cm[:, 1]
        with np.errstate(invalid="ignore"):
            off = (x < -5) | (x > layout.page_w + 5) | (y < -5) | (y > layout.page_h + 5)
        obj_cm[off] = np.nan
        obj_px[off] = np.nan

    grasp, t_close, t_open = detect_grasp(obj_cm, pinch_cm, fps)
    return HandTrack(fps=float(fps), landmarks_px=lm, detected=detected, pinch_px=pinch_px,
                     pinch_cm=pinch_cm, aperture_cm=aperture_cm, obj_px=obj_px, obj_cm=obj_cm,
                     grasp=grasp, t_close=-1 if t_close is None else t_close,
                     t_open=-1 if t_open is None else t_open)


# --------------------------------------------------------------------------- visual checks
def draw_track(frame: np.ndarray, track: HandTrack, i: int) -> np.ndarray:
    """Draw the tracking result for frame i onto the image.
    White dots = thumb/index tips; filled dot = pinch point (red = closed, green = open);
    blue ring = object centre.

    Inputs:
        frame: np.ndarray uint8 (H,W,3) RGB -- frame i of the clip
        track: HandTrack of that clip
        i:     int -- frame index
    Returns:
        np.ndarray uint8 (H,W,3) RGB, a copy of frame with the overlay drawn
    """
    out = frame.copy()
    for j in (THUMB_TIP, INDEX_TIP):
        p = track.landmarks_px[i, j]
        if not np.isnan(p).any():
            cv2.circle(out, tuple(p.round().astype(int)), 5, (255, 255, 255), -1)
    p = track.pinch_px[i]
    if not np.isnan(p).any():
        cv2.circle(out, tuple(p.round().astype(int)), 10, (255, 0, 0) if track.grasp[i] else (0, 200, 0), -1)
    o = track.obj_px[i]
    if not np.isnan(o).any():
        cv2.circle(out, tuple(o.round().astype(int)), 14, (0, 0, 255), 3)
    return out


def plot_track(track: HandTrack):
    """Four stacked time plots: x and y (pinch vs object, cm), thumb-index aperture (cm),
    and the hand-seen / grasp flags. The grasp period is shaded red.

    Inputs:
        track: HandTrack
    Returns:
        matplotlib Figure
    """
    import matplotlib.pyplot as plt
    t = np.arange(len(track.grasp)) / track.fps
    fig, ax = plt.subplots(4, 1, figsize=(9, 7), sharex=True)
    ax[0].plot(t, track.pinch_cm[:, 0], label="pinch"); ax[0].plot(t, track.obj_cm[:, 0], "--", label="object")
    ax[0].set_ylabel("x (cm)"); ax[0].legend(loc="upper right")
    ax[1].plot(t, track.pinch_cm[:, 1]); ax[1].plot(t, track.obj_cm[:, 1], "--"); ax[1].set_ylabel("y (cm)")
    ax[2].plot(t, track.aperture_cm); ax[2].set_ylabel("thumb-index\n(cm)")
    ax[3].plot(t, track.detected, label="hand seen"); ax[3].plot(t, track.grasp, label="grasp")
    ax[3].set_ylabel("flags"); ax[3].set_xlabel("time (s)"); ax[3].legend(loc="upper right")
    for a in ax[:3]:
        a.axvspan(*(t[[track.t_close, track.t_open]] if track.t_close >= 0 else (0, 0)), color="red", alpha=0.1)
    plt.tight_layout()
    return fig