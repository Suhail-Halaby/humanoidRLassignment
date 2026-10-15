"""Track every phone clip in a folder: video -> hand/object path in table cm + grasp timing, saved as .npz.

For each clip:
    1. Load the frames.
    2. Calibrate the table plane from the ArUco sheet (calib.py) and print a health check.
    3. Track the hand, the coloured object and the grasp moments (handtrack.py) and print a health check.
    4. Save the result to <out>/<clip name>.npz, plus a plot (.png) and an overlay video (.mp4) to check by eye.

The printed health checks are the point: a clip with poor calibration or no grasp found should be
fixed or left out before it reaches the robot, not discovered later as a failed replay.

Run (Colab):
    !python scripts/track.py --videos /content/drive/MyDrive/humanoid/videos \
        --out /content/drive/MyDrive/humanoid/processed --colour blue --marker-cm 5.0
"""
import argparse
import glob
import os

import numpy as np


def load_video(path: str):
    """Read every frame of a video file.

    Inputs:
        path: str -- video file (.mp4 / .mov)
    Returns:
        (frames, fps): np.ndarray uint8 (N,H,W,3) RGB; float frames per second
    """
    import cv2
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))   # OpenCV reads colours as BGR
    cap.release()
    return np.stack(frames), fps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", required=True, help="folder with the phone clips")
    p.add_argument("--out", required=True, help="folder for the .npz / .png / .mp4 results")
    p.add_argument("--colour", default="blue", help="object colour, a key of handtrack.COLOURS")
    p.add_argument("--marker-cm", type=float, default=5.0, help="ruler-measured side of one printed marker")
    p.add_argument("--no-video", action="store_true", help="skip the overlay videos (faster)")
    a = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")                  # why: draw plots straight to files; a script has no screen to show them on
    import matplotlib.pyplot as plt
    import mediapy
    from hand2panda.calib import PageLayout, calibrate_video
    from hand2panda.handtrack import draw_track, plot_track, process_video

    os.makedirs(a.out, exist_ok=True)
    layout = PageLayout.scaled(measured_marker_cm=a.marker_cm)
    clips = sorted(glob.glob(os.path.join(a.videos, "*.mp4")) + glob.glob(os.path.join(a.videos, "*.mov")))
    print(f"{len(clips)} clips found in {a.videos}\n")

    for path in clips:
        name = os.path.splitext(os.path.basename(path))[0]
        print(f"=== {name}")
        try:
            frames, fps = load_video(path)
            calib = calibrate_video(frames, layout)
            print("  calibration:", calib.summary())
            track = process_video(frames, fps, calib, colour=a.colour, layout=layout)
            print("  tracking:   ", track.summary())
        except RuntimeError as e:          # e.g. no markers found: report it and move on to the next clip
            print("  FAILED:", e)
            continue

        track.save(os.path.join(a.out, f"{name}.npz"))
        fig = plot_track(track)
        fig.savefig(os.path.join(a.out, f"{name}.png"), dpi=80)
        plt.close(fig)                     # why: otherwise every clip's figure stays in memory
        if not a.no_video:
            overlay = [draw_track(f, track, i) for i, f in enumerate(frames)]
            mediapy.write_video(os.path.join(a.out, f"{name}_overlay.mp4"), overlay[::2], fps=fps / 2)
        del frames                         # why: each clip is ~1 GB in memory; free it before the next one


if __name__ == "__main__":
    main()