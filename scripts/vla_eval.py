"""SmolVLA benchmark: success rate on one LIBERO task, with fixed evaluation settings.

By default it evaluates the ready-made pretrained model HuggingFaceVLA/smolvla_libero (the baseline),
with the same settings as the notebook run that gave 7/10:

    task            libero_object, task 0: "pick up the alphabet soup and place it in the basket"
    episodes        10, from LIBERO's fixed start layouts 0-9
    n_action_steps  10   -- SmolVLA plans 50 actions per call; it carries out 10 before planning again.
                            Why: ~4x faster per episode than the default of 1 (82 s vs 328 s in our runs).
    async envs      off  -- the background simulator process stalled on Colab

The same settings must be used for any later model (e.g. the post-trained one, via --policy),
otherwise the comparison is not fair. That is why they live here once instead of being retyped.

Run in the GPU notebook (needs LeRobot installed):
    !python scripts/vla_benchmark.py --out /content/drive/MyDrive/humanoid/eval_smolvla_baseline
Prints one RESULT line at the end; LeRobot also saves eval_info.json and a video per episode in --out.
"""
import argparse
import json
import os
import subprocess
import sys

DEFAULT_POLICY = "HuggingFaceVLA/smolvla_libero"
EVAL_ARGS = [
    "--env.type=libero", "--env.task=libero_object", "--env.task_ids=[0]",
    "--policy.n_action_steps=10",
    "--eval.batch_size=1", "--eval.n_episodes=10", "--eval.use_async_envs=false",
]


def run(cmd: list[str], log_path: str) -> None:
    """Run a command, showing its output live and copying it into a log file.
    Why the log: if Colab disconnects, the log on Drive still shows how far it got.

    Inputs:  cmd: list of str -- the command and its arguments; log_path: str -- text file to write
    Returns: nothing; stops the script if the command fails
    """
    print("running:", " ".join(cmd), "\n")
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        if proc.wait() != 0:
            sys.exit(f"benchmark failed (exit code {proc.returncode}); see {log_path}")


def summarise(out_dir: str) -> None:
    """Print the headline numbers from LeRobot's eval_info.json.

    Inputs:  out_dir: str -- the evaluation output folder
    Returns: nothing (prints successes, 95% interval and seconds per episode)
    """
    path = os.path.join(out_dir, "eval_info.json")
    if not os.path.exists(path):
        print("no eval_info.json found in", out_dir)
        return
    o = json.load(open(path))["overall"]
    lo, hi = o["pc_success_ci95"]
    print(f"\nRESULT: {o['n_success']}/{o['n_episodes']} episodes succeeded "
          f"({o['pc_success']:.0f}%, 95% interval {lo:.0f}-{hi:.0f}%), {o['eval_ep_s']:.0f} s per episode")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default=DEFAULT_POLICY,
                   help="model to test: hub id (default: the pretrained baseline) or a local pretrained_model folder")
    p.add_argument("--out", required=True, help="output folder on Drive (must not exist yet)")
    a = p.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")      # draw simulator images without a screen
    out = a.out.rstrip("/")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    run(["lerobot-eval", f"--policy.path={a.policy}", *EVAL_ARGS, f"--output_dir={out}"], out + "_log.txt")
    summarise(out)


if __name__ == "__main__":
    main()