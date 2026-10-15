"""Check that the LIBERO setup works end to end, before any hand data is involved.

Steps:
    1. Create the task and print its instruction, observation keys and object positions.
    2. Probe the controller: how far the gripper moves per action unit, and which gripper command closes.
    3. Hand-coded pick: go above the chosen object, descend, close, lift. Report whether the object rose.
    4. Save a video of step 3.

If step 3 lifts the object, the robot side is ready: retargeting only has to produce target points.

Run (Colab):
    !python scripts/check_sim.py --suite libero_object --task 0 --object alphabet_soup_1 --video sim_check.mp4
Leave out --object the first time to just see the object names.
"""
import argparse

import numpy as np

from hand2panda import sim


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="libero_object")
    p.add_argument("--task", type=int, default=0)
    p.add_argument("--init", type=int, default=0, help="which fixed start state (0-49)")
    p.add_argument("--object", default=None, help="object to pick, e.g. alphabet_soup_1")
    p.add_argument("--video", default="sim_check.mp4")
    a = p.parse_args()

    env, info = sim.make_env(a.suite, a.task)
    init = info["init_states"][a.init]
    print("task:", info["language"])
    print(f"{len(info['init_states'])} fixed start states available")

    obs = sim.reset(env, init)
    print("gripper starts at (m):", np.round(sim.eef_m(obs), 3))
    sim.print_obs(obs)

    pr = sim.probe_controller(env, init)
    print("\ncontroller probe -- movement per step at action=+1 (rows: commanded x,y,z; cols: measured x,y,z, mm):")
    print(np.round(pr["move_m_per_step"] * 1000, 1))
    print(f"finger gap after -1: {pr['width_after_minus1_m'] * 1000:.1f} mm, "
          f"after +1: {pr['width_after_plus1_m'] * 1000:.1f} mm -> close command is {pr['close_cmd']:+.0f}")
    if pr["close_cmd"] != sim.CLOSE:
        print("WARNING: gripper sign is the opposite of sim.CLOSE; swap OPEN/CLOSE in sim.py")

    if a.object is None:
        print("\nNo --object given; pick one of the names above and run again.")
        return

    # hand-coded pick: above -> down -> close -> up
    obs = sim.reset(env, init)
    start_m = sim.object_m(obs, a.object)
    frames = []
    for target, grip, what in [(start_m + [0, 0, 0.10], sim.OPEN, "above"),
                               (start_m, sim.OPEN, "down")]:
        obs, f, ok = sim.go_to(env, target, grip, obs, max_steps=150)
        frames += f
        print(f"{what:6s}: reached={ok}, gripper at {np.round(sim.eef_m(obs), 3)}")
    obs, f = sim.wait(env, 15, sim.CLOSE)
    frames += f
    print(f"closed: finger gap {sim.gripper_width_m(obs) * 1000:.1f} mm")
    obs, f, ok = sim.go_to(env, start_m + [0, 0, 0.15], sim.CLOSE, obs, max_steps=150)
    frames += f
    rise_m = sim.object_m(obs, a.object)[2] - start_m[2]
    print(f"lift  : reached={ok}, object rose {rise_m * 100:.1f} cm -> {'PICKED' if rise_m > 0.05 else 'NOT picked'}")

    import mediapy
    mediapy.write_video(a.video, frames, fps=sim.CONTROL_HZ)
    print("video saved to", a.video)


if __name__ == "__main__":
    main()