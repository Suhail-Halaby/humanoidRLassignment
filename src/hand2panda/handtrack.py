"""Hand + object tracking: phone video -> where the "gripper" is on the table, and when it is closed.

Pipeline for one clip:
    1. MediaPipe finds 21 hand landmarks (pixels) in every frame.
    2. Short gaps are filled in, jitter is smoothed out.
    3. The pinch point (midway between thumb tip and index tip) is our stand-in for the robot gripper.
    4. A coloured object is found by colour thresholding.
    5. Everything is converted from pixels to table cm with the calibration from calib.py.
    6. Grasp timing from two signals: the OBJECT moving with the hand says whether and roughly when;
       the FINGERS closing (thumb-index gap / hand size) pin down the exact close and open moments.

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
# Two independent signals, each used for what it is good at:
#   object carried  -> WHETHER a grasp happened and roughly when (hard to fool)
#   fingers closed  -> EXACTLY when the fingers closed / opened (precise but noisy)

def grip_ratio(lm_px: np.ndarray) -> np.ndarray:
    """Thumb-index gap relative to hand size (the method from the original HandTrack notebook).
    Why divide by hand size: the raw gap in pixels or cm changes with distance from the camera and
    with hand height above the table; a ratio of two lengths on the same hand does not.

    Inputs:
        lm_px: np.ndarray (N,21,2) -- hand landmarks, pixels
    Returns:
        np.ndarray float64 (N,) -- |thumb tip - index tip| / |wrist - middle knuckle|; NaN where no hand
    """
    gap_px = np.linalg.norm(lm_px[:, THUMB_TIP] - lm_px[:, INDEX_TIP], axis=1)
    hand_px = np.linalg.norm(lm_px[:, WRIST] - lm_px[:, MIDDLE_MCP], axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return gap_px / hand_px


def object_carry(obj_cm: np.ndarray, pinch_cm: np.ndarray, fps: float, move_cm: float = 2.0,
                 near_cm: float = 6.0) -> np.ndarray:
    """Frames in which the object is being carried by the hand. Assumes ONE pick-and-place per clip:
    the object rests at a start position, gets carried, then rests at an end position.

    Inputs:
        obj_cm:   np.ndarray (N,2) -- object position per frame, table cm (NaN where not seen)
        pinch_cm: np.ndarray (N,2) -- pinch point per frame, table cm (NaN where not seen)
        fps:      float -- frame rate
        move_cm:  float -- distance from a rest position that counts as "moved", cm
        near_cm:  float -- max object-to-pinch distance that counts as "in the hand", cm
    Returns:
        np.ndarray bool (N,) -- True where the object is away from both rest positions AND near the
        pinch point; all False if the object was seen in fewer than 2 frames
    """
    n = len(obj_cm)
    seen = np.flatnonzero(~np.isnan(obj_cm).any(axis=1))
    if len(seen) < 2:
        return np.zeros(n, bool)
    half_s = max(int(fps * 0.5), 1)                          # frames in half a second
    # median, not mean: one bad detection can't drag the rest position off
    start = np.nanmedian(obj_cm[seen[:half_s]], axis=0)      # where it sat at the beginning
    end = np.nanmedian(obj_cm[seen[-half_s:]], axis=0)       # where it sat at the end
    with np.errstate(invalid="ignore"):                      # NaN comparisons are simply False
        away = (np.linalg.norm(obj_cm - start, axis=1) > move_cm) & \
               (np.linalg.norm(obj_cm - end, axis=1) > move_cm)
        near = np.linalg.norm(obj_cm - pinch_cm, axis=1) < near_cm
    return away & near


def finger_threshold(ratio: np.ndarray, carry: np.ndarray, min_contrast: float = 0.15) -> float:
    """Grip-ratio value that separates "open" from "closed", set automatically for this clip.
    Why automatic: "closed" around a can is a much bigger ratio than "closed" around a pen,
    so a fixed number (like the old 0.5) can't work for every object.
    Closed level = typical ratio while carrying; open level = 75th percentile outside the carry
    (75th, not median: the hand is half-closed for part of the approach). Threshold = halfway.

    Inputs:
        ratio:        np.ndarray (N,) -- grip_ratio per frame
        carry:        np.ndarray bool (N,) -- from object_carry
        min_contrast: float -- if open and closed levels differ by less than this, the fingers
                      don't show a clear grasp (e.g. thin object) and the signal is not trusted
    Returns:
        float -- threshold; NaN if the finger signal is unusable for this clip
    """
    closed = ratio[carry & ~np.isnan(ratio)]
    opened = ratio[~carry & ~np.isnan(ratio)]
    if len(closed) == 0 or len(opened) == 0:
        return np.nan
    closed_level = np.median(closed)
    open_level = np.percentile(opened, 75)
    if open_level - closed_level < min_contrast:
        return np.nan
    return float(0.5 * (open_level + closed_level))


def _first_open(ratio: np.ndarray, thr: float, start: int, step: int, max_frames: int):
    """Walk from frame `start` in direction `step` (-1 = back in time, +1 = forward) and return the
    first frame where the fingers are open (ratio > thr).

    Inputs:
        ratio: np.ndarray (N,); thr: float; start: int frame; step: -1 or +1; max_frames: int
    Returns:
        int frame index, or None if not found within max_frames, or the hand is not visible on the way
        (why give up on NaN: we would otherwise be guessing the timing)
    """
    for k in range(max_frames + 1):
        j = start + step * k
        if j < 0 or j >= len(ratio) or np.isnan(ratio[j]):
            return None
        if ratio[j] > thr:
            return j
    return None


def detect_grasp(carry: np.ndarray, ratio: np.ndarray, fps: float, pad_s: float = 0.25,
                 search_s: float = 1.0, min_contrast: float = 0.15) -> dict:
    """Combine both signals into when the gripper is closed.

    1. No carried frames -> no grasp (the finger signal alone is never enough: pinching air is not a grasp).
    2. Close moment: search back from the first carried frame (up to search_s) for the last time the
       fingers were open; the gripper closes on the frame after that.
       Open moment: search forward from the last carried frame for the first time the fingers open.
    3. Fallback for either moment, if the finger signal is unusable or the hand isn't visible there:
       a fixed pad_s before / after the carry.

    Inputs:
        carry:        np.ndarray bool (N,) -- from object_carry
        ratio:        np.ndarray (N,) -- from grip_ratio
        fps:          float -- frame rate
        pad_s:        float -- fallback margin, s
        search_s:     float -- how far before / after the carry to look for the finger movement, s
        min_contrast: float -- passed to finger_threshold
    Returns:
        dict with keys
            "grasp":     np.ndarray bool (N,) -- True = gripper closed
            "t_close":   int -- first closed frame (-1 if no grasp)
            "t_open":    int -- last closed frame (-1 if no grasp)
            "close_src": str -- "fingers", "padding" or "none": which signal set t_close
            "open_src":  str -- same, for t_open
            "threshold": float -- grip-ratio threshold (NaN if the finger signal was unusable)
    """
    n = len(carry)
    out = {"grasp": np.zeros(n, bool), "t_close": -1, "t_open": -1,
           "close_src": "none", "open_src": "none", "threshold": np.nan}
    carried = np.flatnonzero(carry)
    if len(carried) == 0:
        return out
    first, last = carried[0], carried[-1]
    thr = finger_threshold(ratio, carry, min_contrast)
    out["threshold"] = thr
    search = int(round(search_s * fps))
    pad = int(round(pad_s * fps))

    j = None if np.isnan(thr) else _first_open(ratio, thr, first, -1, search)
    if j is not None:
        # min(): if the fingers already look open at the first carried frame, don't close after the carry starts
        out["t_close"], out["close_src"] = min(j + 1, first), "fingers"
    else:
        out["t_close"], out["close_src"] = max(first - pad, 0), "padding"

    j = None if np.isnan(thr) else _first_open(ratio, thr, last, +1, search)
    if j is not None:
        out["t_open"], out["open_src"] = max(j - 1, last), "fingers"
    else:
        out["t_open"], out["open_src"] = min(last + pad, n - 1), "padding"

    out["grasp"][out["t_close"]:out["t_open"] + 1] = True
    return out


# --------------------------------------------------------------------------- result container
@dataclass
class HandTrack:
    """Everything tracked in one clip of N frames.

    Fields:
        fps:           float
        landmarks_px:  np.ndarray (N,21,2) -- cleaned hand landmarks, pixels
        detected:      np.ndarray bool (N,) -- True where MediaPipe actually saw the hand (before gap filling)
        pinch_px:      np.ndarray (N,2) -- pinch point, pixels
        pinch_cm:      np.ndarray (N,2) -- pinch point, table cm (the gripper stand-in)
        grip_ratio:    np.ndarray (N,) -- thumb-index gap / hand size (also usable as a gripper width later)
        grip_thr:      float -- open/closed threshold for grip_ratio, NaN if the finger signal was unusable
        finger_closed: np.ndarray bool (N,) -- grip_ratio below grip_thr (all False if grip_thr is NaN)
        obj_px:        np.ndarray (N,2) -- object centre, pixels
        obj_cm:        np.ndarray (N,2) -- object centre, table cm
        carry:         np.ndarray bool (N,) -- object judged to be carried by the hand
        grasp:         np.ndarray bool (N,) -- final answer: True = gripper should be closed
        t_close:       int -- first closed frame, -1 if no grasp found
                       (why -1 and not None: .npz files can't store None)
        t_open:        int -- last closed frame, -1 if no grasp found
        close_src:     str -- "fingers" / "padding" / "none": which signal decided t_close
        open_src:      str -- same, for t_open
    """
    fps: float
    landmarks_px: np.ndarray
    detected: np.ndarray
    pinch_px: np.ndarray
    pinch_cm: np.ndarray
    grip_ratio: np.ndarray
    grip_thr: float
    finger_closed: np.ndarray
    obj_px: np.ndarray
    obj_cm: np.ndarray
    carry: np.ndarray
    grasp: np.ndarray
    t_close: int
    t_open: int
    close_src: str
    open_src: str

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
        # .npz stores single values as 0-d arrays; turn them back into plain float/int/str
        for k in ("fps", "grip_thr"):
            kw[k] = float(kw[k])
        for k in ("t_close", "t_open"):
            kw[k] = int(kw[k])
        for k in ("close_src", "open_src"):
            kw[k] = str(kw[k])
        return cls(**kw)

    def summary(self) -> str:
        """One-line health check: frames, % hand / object seen, grasp frames, which signal set them,
        and how far the measured finger movement was from the object's movement (ms)."""
        ok = ~np.isnan(self.pinch_cm[:, 0])
        line = (f"{len(self.detected)} frames | hand seen {self.detected.mean() * 100:.0f}% "
                f"(usable after gap-fill {ok.mean() * 100:.0f}%) | "
                f"object seen {(~np.isnan(self.obj_cm[:, 0])).mean() * 100:.0f}% | ")
        if self.t_close < 0:
            return line + "NO GRASP FOUND"
        carried = np.flatnonzero(self.carry)
        lead_ms = (carried[0] - self.t_close) / self.fps * 1000     # fingers closed this long before object moved
        lag_ms = (self.t_open - carried[-1]) / self.fps * 1000      # fingers opened this long after it stopped
        return line + (f"close@{self.t_close} ({self.close_src}, {lead_ms:.0f}ms before carry) "
                       f"open@{self.t_open} ({self.open_src}, {lag_ms:.0f}ms after) | "
                       f"grip threshold {self.grip_thr:.2f}")


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
    pinch_cm = calib.to_table(pinch_px)
    ratio = grip_ratio(lm)                     # in pixels on purpose: it's a ratio, so units cancel

    obj_px = smooth(track_object(frames, hsv_ranges or COLOURS[colour]), 3)
    obj_cm = calib.to_table(obj_px)
    if layout is not None:                     # ignore colour blobs far off the sheet
        x, y = obj_cm[:, 0], obj_cm[:, 1]
        with np.errstate(invalid="ignore"):
            off = (x < -5) | (x > layout.page_w + 5) | (y < -5) | (y > layout.page_h + 5)
        obj_cm[off] = np.nan
        obj_px[off] = np.nan

    carry = object_carry(obj_cm, pinch_cm, fps)
    g = detect_grasp(carry, ratio, fps)
    with np.errstate(invalid="ignore"):
        finger_closed = ratio < g["threshold"]              # all False if threshold is NaN
    return HandTrack(fps=float(fps), landmarks_px=lm, detected=detected, pinch_px=pinch_px,
                     pinch_cm=pinch_cm, grip_ratio=ratio, grip_thr=g["threshold"],
                     finger_closed=finger_closed, obj_px=obj_px, obj_cm=obj_cm, carry=carry,
                     grasp=g["grasp"], t_close=g["t_close"], t_open=g["t_open"],
                     close_src=g["close_src"], open_src=g["open_src"])


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
    """Four stacked time plots, grasp period shaded red on all of them:
    x and y (pinch vs object, cm); grip ratio with its threshold; and the four on/off signals
    (hand seen, object carried, fingers closed, final grasp) drawn as separate lanes.

    Inputs:
        track: HandTrack
    Returns:
        matplotlib Figure
    """
    import matplotlib.pyplot as plt
    t = np.arange(len(track.grasp)) / track.fps
    fig, ax = plt.subplots(4, 1, figsize=(9, 8), sharex=True)
    ax[0].plot(t, track.pinch_cm[:, 0], label="pinch"); ax[0].plot(t, track.obj_cm[:, 0], "--", label="object")
    ax[0].set_ylabel("x (cm)"); ax[0].legend(loc="upper right")
    ax[1].plot(t, track.pinch_cm[:, 1]); ax[1].plot(t, track.obj_cm[:, 1], "--"); ax[1].set_ylabel("y (cm)")
    ax[2].plot(t, track.grip_ratio, label="grip ratio")
    if not np.isnan(track.grip_thr):
        ax[2].axhline(track.grip_thr, ls="--", color="k", label="threshold")
    ax[2].set_ylabel("thumb-index /\nhand size"); ax[2].legend(loc="upper right")
    # each on/off signal gets its own lane (offset by 1.5) so they don't overlap
    lanes = [("hand seen", track.detected), ("object carried", track.carry),
             ("fingers closed", track.finger_closed), ("GRASP", track.grasp)]
    for k, (name, sig) in enumerate(lanes):
        ax[3].step(t, sig.astype(float) + 1.5 * k, where="post")
    ax[3].set_yticks([1.5 * k + 0.5 for k in range(len(lanes))], [name for name, _ in lanes])
    ax[3].set_xlabel("time (s)")
    if track.t_close >= 0:
        for a in ax:
            a.axvspan(t[track.t_close], t[track.t_open], color="red", alpha=0.1)
    plt.tight_layout()
    return fig