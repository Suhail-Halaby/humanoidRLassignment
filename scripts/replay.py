"""Replay every tracked hand demo on the simulated Panda and count how many complete the LIBERO task.

For each clip (.npz from track.py) and each chosen start layout:
    1. Reset the scene to that layout and read where the object and the target are.
    2. Retarget the hand path onto that layout (retarget.py).
    3. Drive the robot through it and ask LIBERO whether the task was completed.
Saves one video per clip (first layout) and a results table (replay_results.csv).

Run (Colab; PYTHONPATH because LIBERO's own install doesn't make it importable):
    !PYTHONPATH=/content/LIBERO python scripts/replay.py \
        --tracks /content/drive/MyDrive/humanoid/processed --out /content/drive/MyDrive/humanoid/replay \
        --object alphabet_soup_1 --place basket_1 --inits 0,1,2,3,4
"""
import argparse
import glob
import os

import numpy as np

from hand2panda import sim
from hand2panda.handtrack import HandTrack
from hand2panda.retarget import execute, retarget


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tracks", required=True, help="folder with the .npz files from track.py")
    p.add_argument("--out", required=True, help="folder for videos and the results table")
    p.add_argument("--suite", default="libero_object")
    p.add_argument("--task", type=int, default=0)
    p.add_argument("--object", default="alphabet_soup_1", help="sim object your hand demo picks up")
    p.add_argument("--place", default="basket_1", help="sim object it is put into / onto")
    p.add_argument("--inits", default="0", help="comma-separated start layouts, e.g. 0,1,2,3,4")
    p.add_argument("--speed", type=float, default=1.0, help="1.0 = your speed, 0.5 = half speed")
    p.add_argument("--settle", type=int, default=15, help="steps to wait for the fingers at each grasp / release")
    a = p.parse_args()

    import mediapy
    os.makedirs(a.out, exist_ok=True)
    env, info = sim.make_env(a.suite, a.task)
    inits = [int(s) for s in a.inits.split(",")]
    print("task:", info["language"])

    rows = []
    for path in sorted(glob.glob(os.path.join(a.tracks, "*.npz"))):
        name = os.path.splitext(os.path.basename(path))[0]
        track = HandTrack.load(path)
        if track.t_close < 0:
            print(f"=== {name}: skipped (no grasp in the tracking)")
            continue
        print(f"=== {name}")
        for n, init in enumerate(inits):
            obs = sim.reset(env, info["init_states"][init])
            try:
                plan = retarget(track, sim.object_m(obs, a.object), sim.object_m(obs, a.place), speed=a.speed)
            except ValueError as e:
                print(f"  layout {init}: retarget FAILED: {e}")
                rows.append((name, init, "retarget_failed"))
                continue
            obs, frames, ok = execute(env, plan, obs, settle_steps=a.settle)
            print(f"  layout {init}: {'SUCCESS' if ok else 'fail'} | {plan.summary()}")
            rows.append((name, init, "success" if ok else "fail"))
            if n == 0:                       # one video per clip is enough to check by eye
                mediapy.write_video(os.path.join(a.out, f"{name}_layout{init}.mp4"), frames, fps=sim.CONTROL_HZ)

    with open(os.path.join(a.out, "replay_results.csv"), "w") as f:
        f.write("clip,layout,result\n")
        f.writelines(f"{c},{i},{r}\n" for c, i, r in rows)
    wins = sum(r == "success" for _, _, r in rows)
    print(f"\nTOTAL: {wins}/{len(rows)} replays succeeded "
          f"({len({c for c, _, _ in rows})} clips x {len(inits)} layouts)")


if __name__ == "__main__":
    main()