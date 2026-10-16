"""Turn the successful replay demos (.npz from replay.py --record) into a LeRobot dataset for SmolVLA.

The dataset uses the same names and layout as SmolVLA's LIBERO training data (HuggingFaceVLA/libero):
    observation.images.image   scene camera, 256x256, rotated 180 degrees (done in sim.policy_inputs)
    observation.images.image2  wrist camera, 256x256, rotated 180 degrees
    observation.state          8 numbers: gripper position (3), axis-angle (3), finger joints (2)
    action                     7 numbers: the exact actions sent to the simulator
so HuggingFaceVLA/smolvla_libero can be post-trained on it with no renaming.

Run in the GPU notebook (needs LeRobot installed, not LIBERO):
    !python scripts/to_lerobot.py --demos /content/drive/MyDrive/humanoid/demos \
        --out /content/drive/MyDrive/humanoid/dataset_hand2panda
"""
import argparse
import glob
import os

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--demos", required=True, help="folder with the .npz demos from replay.py --record")
    p.add_argument("--out", required=True, help="dataset folder to create (must not exist yet)")
    p.add_argument("--episodes", type=int, default=25, help="test episodes = layouts 0..episodes-1")
    p.add_argument("--repo-id", default="local/hand2panda", help="dataset name (only used as a label)")
    a = p.parse_args()

    from lerobot.datasets import LeRobotDataset   # heavy import: only in the GPU notebook

    paths = sorted(glob.glob(os.path.join(a.demos, "*.npz")))
    if not paths:
        raise SystemExit(f"no .npz demos in {a.demos}")
    h, w = np.load(paths[0])["image"].shape[1:3]

    def img(): return {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]}
    features = {
        "observation.images.image": img(),
        "observation.images.image2": img(),
        "observation.state": {"dtype": "float32", "shape": (8,), "names": None},
        "action": {"dtype": "float32", "shape": (7,), "names": None},
    }
    ds = LeRobotDataset.create(repo_id=a.repo_id, fps=20, features=features, root=a.out,
                               robot_type="panda", use_videos=True)

    n_frames = 0
    for path in paths:
        d = np.load(path)
        task = str(d["task"])
        for t in range(len(d["action"])):
            ds.add_frame({"observation.images.image": d["image"][t],
                          "observation.images.image2": d["image2"][t],
                          "observation.state": d["state"][t],
                          "action": d["action"][t],
                          "task": task})
        ds.save_episode()
        n_frames += len(d["action"])
        print(f"  {os.path.basename(path)}: {len(d['action'])} steps")
    ds.finalize()        # writes the closing metadata; without it the dataset is unreadable
    print(f"\n{len(paths)} episodes, {n_frames} frames -> {a.out}")


if __name__ == "__main__":
    main()