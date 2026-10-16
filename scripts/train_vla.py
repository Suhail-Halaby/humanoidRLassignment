"""Post-train SmolVLA on the demos made from my hand videos (the learning step of the project).

What happens: LeRobot's trainer (lerobot-train) starts from HuggingFaceVLA/smolvla_libero, a SmolVLA that is
already trained on LIBERO, and keeps training it on our dataset (built by to_lerobot.py). Each update it
shows the model a batch of recorded moments (camera images + robot state), compares the model's predicted
actions with the actions in the demo (the loss), and nudges the weights to reduce that difference.
This is imitation learning: the model learns to copy the demos, and the demos come from my hand.

My code produces the data; LeRobot's code (lerobot/scripts/lerobot_train.py) does the training.

Settings and why:
    steps 2000, batch 8   few demos: much longer mostly memorises them
    lr 1e-4               SmolVLA's own default learning rate (size of each nudge)
    warmup 100 steps      SmolVLA's default (1000) is meant for 30000-step runs; it would be half of ours
    decay over all steps  so the nudges actually shrink by the end of our short run
    augmentation on       small random brightness / contrast / shift changes, so it can't memorise exact images
    vision encoder frozen, only the "action expert" trained (SmolVLA defaults, kept): with little data we
                          change how it moves, not how it sees
    checkpoints at 1000 and 2000 steps (each > 1 GB on Drive)

Run in the GPU notebook (needs LeRobot installed):
    !python scripts/train_vla.py --dataset /content/drive/MyDrive/humanoid/dataset_hand2panda \
        --out /content/drive/MyDrive/humanoid/train_hand2panda
The trained model ends up in <out>/checkpoints/last/pretrained_model; test it with vla_benchmark.py --policy.
Watch the "loss" lines: they should fall quickly, then flatten.
"""
import argparse
import os
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="dataset folder from to_lerobot.py")
    p.add_argument("--out", required=True, help="training output folder on Drive (must not exist yet)")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    a = p.parse_args()

    out = a.out.rstrip("/")
    cmd = ["lerobot-train",
           "--policy.path=HuggingFaceVLA/smolvla_libero",
           "--dataset.repo_id=local/hand2panda", f"--dataset.root={a.dataset}",
           "--dataset.image_transforms.enable=true",
           f"--steps={a.steps}", f"--batch_size={a.batch_size}",
           f"--policy.optimizer_lr={a.lr}",
           "--policy.scheduler_warmup_steps=100", f"--policy.scheduler_decay_steps={a.steps}",
           f"--save_freq={max(a.steps // 2, 1)}", "--log_freq=50", "--seed=1000",
           "--policy.push_to_hub=false",
           f"--output_dir={out}"]
    print("running:", " ".join(cmd), "\n")

    # show the output live and keep a copy on Drive, in case Colab disconnects
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + "_log.txt", "w") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        if proc.wait() != 0:
            sys.exit(f"training failed (exit code {proc.returncode}); see {out}_log.txt")
    print(f"\ndone. model: {out}/checkpoints/last/pretrained_model")


if __name__ == "__main__":
    main()
