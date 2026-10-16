"""SmolVLA benchmark: success rate on one LIBERO task, with fixed evaluation settings.

By default it evaluates the ready-made pretrained model HuggingFaceVLA/smolvla_libero (the baseline).
Any other model (e.g. the post-trained one) is passed with --policy and tested in exactly the same way,
otherwise the comparison would not be fair. That is why the settings live here once instead of being retyped.

    task            libero_object, task 0: "pick up the alphabet soup and place it in the basket"
    episodes        25 by default, from LIBERO's fixed start layouts 0-24 (episode i uses layout i).
                    Why 25: with 10 episodes the 95% interval of a 7/10 result is 40-89%, too wide to show
                    an improvement. The hand demos are recorded on layouts 25-49 so the test stays unseen.
    n_action_steps  10   -- SmolVLA plans 50 actions per call; it carries out 10 before planning again.
                            Why: ~4x faster per episode than the default of 1 (82 s vs 328 s in our runs).
    async envs      off  -- one simulator only, so a background process adds nothing but stall risk on Colab

First measurement (10 episodes, same settings otherwise): 7/10, 95% interval 40-89%, 82 s per episode.

Run in the GPU notebook (needs LeRobot installed):
    !python scripts/vla_benchmark.py --out /content/drive/MyDrive/humanoid/eval_baseline_25
    !python scripts/vla_benchmark.py --policy <train out>/checkpoints/last/pretrained_model \
        --out /content/drive/MyDrive/humanoid/eval_posttrained_25
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
    "--eval.batch_size=1", "--eval.use_async_envs=false",
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
            sys.exit(f"command failed (exit code {proc.returncode}); see {log_path}")


def summarise(out_dir: str) -> None:
    """Print the headline numbers from LeRobot's eval_info.json.

    Inputs:  out_dir: str -- the evaluation output folder
    Returns: nothing (prints successes, 95% interval, seconds per episode, and which episodes failed)
    """
    path = os.path.join(out_dir, "eval_info.json")
    if not os.path.exists(path):
        print("no eval_info.json found in", out_dir)
        return
    info = json.load(open(path))
    o = info["overall"]
    lo, hi = o["pc_success_ci95"]
    print(f"\nRESULT: {o['n_success']}/{o['n_episodes']} episodes succeeded "
          f"({o['pc_success']:.0f}%, 95% interval {lo:.0f}-{hi:.0f}%), {o['eval_ep_s']:.0f} s per episode")
    try:   # which layouts failed: useful for comparing baseline vs post-trained episode by episode
        ok = info["per_task"][0]["metrics"]["successes"]
        print("failed layouts:", [i for i, s in enumerate(ok) if not s])
    except (KeyError, IndexError, TypeError):
        pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default=DEFAULT_POLICY,
                   help="model to test: hub id (default: the pretrained baseline) or a local pretrained_model folder")
    p.add_argument("--out", required=True, help="output folder on Drive (must not exist yet)")
    p.add_argument("--episodes", type=int, default=25, help="test episodes = layouts 0 .. episodes-1")
    a = p.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")      # draw simulator images without a screen
    out = a.out.rstrip("/")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    run(["lerobot-eval", f"--policy.path={a.policy}", *EVAL_ARGS, f"--eval.n_episodes={a.episodes}",
         f"--output_dir={out}"], out + "_log.txt")
    summarise(out)


if __name__ == "__main__":
    main()
