# humanoidRLassignment — phone-recorded hand demos → Panda in LIBERO → post-trained SmolVLA

I recorded myself picking up a small object and dropping it into a container with a phone.
This repo turns those videos into robot demonstrations for the LIBERO task
*"pick up the alphabet soup and place it in the basket"* (`libero_object`, task 0),
and prepares them as training data for SmolVLA. Retargeted hand demos complete the task in simulation;
the post-training run itself was **not completed before the deadline** (code and data are ready, see step 6).

```
phone video -> 1 calibrate table -> 2 track hand + object -> 3 retarget to robot table
           -> 4 replay in LIBERO -> 5 record successful replays -> 6 post-train SmolVLA ─► 7 benchmark
```

| | result |
|---|---|
| hand demos replayed successfully on the robot | FILL / FILL replays (26 clips × 5 layouts) → FILL training demos |
| SmolVLA baseline (pretrained `smolvla_libero`) | 7/10 episodes (95% CI 40–89%) — FILL: 25-episode result if it finished |
| SmolVLA post-trained on my demos | **not run** (time) — dataset and training script ready |
| evaluation speed-up from `n_action_steps=10` | 82 s vs 328 s per episode (~4×) |

---

## 1 · Calibrating the table — `src/hand2panda/calib.py`, `scripts/make_markers.py`

**What:** an A4 sheet with four ArUco markers (printable square codes) lies under the scene. In every
frame the markers are detected and a *homography* (a perspective mapping) is fitted that converts
pixels into centimetres on the sheet.

**Key decisions**
- **Markers instead of camera calibration.** A phone's lens settings are unknown; four markers at known
  positions give metric table coordinates directly, with no calibration step.
- **Measure the printed marker with a ruler** (`--marker-cm`). Printers often shrink pages slightly; every
  length scales by the same factor, so one measurement corrects everything.
- **Handle a moving phone.** If the markers move more than 3 px during a clip, a mapping is fitted for every
  frame; otherwise all frames are pooled into one mapping (robust when my hand covers a marker).
- **Check it.** The reprojection error (detected marker corners vs. their true positions) was
  0.02–0.08 cm on my clips — sub-millimetre.

## 2 · Tracking the hand and the object — `src/hand2panda/handtrack.py`, `scripts/track.py`

**What:** MediaPipe finds 21 hand landmarks per frame. The **pinch point** (midway between thumb tip and
index tip) stands in for the robot gripper. The yellow object is found by colour. Both are converted to
table cm with step 1.

**Key decisions**
- **Pinch point as the gripper.** It is where a held object actually sits, unlike the wrist.
- **Missing data stays missing.** Gaps of up to 0.3 s are interpolated; longer ones stay empty rather
  than being invented.
- **Grasp timing from two signals.** *Whether* a grasp happened comes from the object moving together
  with the hand (hard to fool); *exactly when* the fingers closed and opened comes from the thumb–index gap
  divided by hand size (precise but noisy). Pinching air is never counted as a grasp.
- **A per-clip open/closed threshold**, because "closed" around a can and around a pen are very different
  finger gaps.

## 3 · Retargeting onto the robot's table — `src/hand2panda/retarget.py` (`retarget`)

**What:** the hand path (2D, cm, on the sheet) becomes a gripper path (3D, metres, robot frame) for the
simulated scene.

**Key decisions**
- **Two anchors.** My pinch point at the grasp is mapped onto the soup can, and at the release onto the
  basket. Two point pairs fix exactly one rotation, one uniform scale and one shift, applied to the whole
  path, so the shape, curvature and timing of my motion are kept. (With the re-filmed clips the scale came
  out at ~0.99: the robot path is the size of my hand path.)
- **The sheet's y axis is flipped first**, because it runs down the page; this way the mapping never
  mirrors my motion.
- **Height is rebuilt by rule.** The phone looks down, so it cannot see height. The gripper hovers on
  approach, descends as my hand converges on the object, carries at a safe height and lowers over the
  basket. Descents are triggered by *distance* to the grasp/release point, so they happen where my hand
  actually homed in, at my speed.
- **Time is resampled** from the phone's 30 fps to LIBERO's 20 steps/s. Steps where the hand was not seen
  are skipped, not filled in.

**Retargeting options** — the parts of the motion that are rules rather than my hand, set from the
command line of `replay.py`. Defaults were first guesses; final values came from watching replays fail:

| option | default | final | what it controls | why it changed |
|---|---|---|---|---|
| `--clear` | 0.15 m | 0.25 m | carry height above the higher of can / basket | the can clipped the basket on the way over |
| `--release-above` | 0.10 m | 0.20 m | release height above the basket's position (near its base) | at 0.10 m the gripper opened at rim height and the can caught the rim |
| `--ramp` | 0.05 m | 0.03 m | horizontal distance over which the gripper descends to the grasp / release point | with 5 cm it sank while the can was still over the rim |
| `--lift` | 0 (off) | 0.10 m | rise straight up after the grasp; never dip below this while carrying | moving sideways while still low dragged or tipped the can |
| `--speed` | 1.0 | 1.0 | playback speed of my motion | not needed: the robot kept up at my speed |

Only the horizontal path's shape and timing and the grasp / release moments come from my video.
Moving `--release-above` and `--ramp` turned clip 20 from a near miss into a success.

## 4 · Replaying on the simulated Panda — `retarget.py` (`execute`), `src/hand2panda/sim.py`, `scripts/replay.py`

**What:** the robot follows the path one target per step (each action is the remaining distance, scaled,
so it slows down as it arrives). LIBERO reports whether the soup ended up in the basket.

**Key decisions — adapting to the robot's body.**
- **`--settle 30`** (1.5 s): at the grasp and release the robot first reaches the exact anchor point, then
  waits for its fingers — the Panda gripper is far slower than my fingers, and moving off early lost the can.
- The `--lift` from step 3 is executed here as a straight-up move after the grasp.

- **Check the robot before the data.** `scripts/check_sim.py` measures the controller (mm per action
  unit, which gripper command closes) and does a hand-coded pick — it lifted the can, so failures later
  could be attributed to the data, not the simulator.

## 5 · Recording training demos — `replay.py --record`, `sim.policy_inputs`, `scripts/to_lerobot.py`

**What:** during each replay, every step's camera images, robot state and action are saved; successful
replays become a LeRobot dataset.

**Key decisions**
- **Exactly SmolVLA's input format.** Two 256×256 cameras rotated 180° and an 8-number state
  (gripper position, axis-angle orientation, finger joints) — the same layout as SmolVLA's LIBERO training
  data and LeRobot's evaluation, so no renaming or conversion at training time.
- **Only successes are kept**; failed demos would teach the policy to fail.
- **One hand clip → several demos.** Each clip is replayed on 5 different table layouts (the anchors adapt
  the path to where the can and basket are), giving variety from few recordings.
- **No overlap with the test.** LIBERO has 50 fixed layouts per task; demos use layouts 25–29, the
  benchmark uses 0–24.

## 6 · Post-training SmolVLA — `scripts/train_vla.py`

**Status: implemented, not run before the deadline.** The dataset builder (`to_lerobot.py`), the training
script and the benchmark for the trained model are written; FILL: dataset built? (N demos, M frames).

**What:** starting from `HuggingFaceVLA/smolvla_libero` (already trained on LIBERO), LeRobot's trainer
continues training on my demos: imitation learning — predict the demo's action from what the robot saw,
measure the error, adjust the weights. My code produces the data; LeRobot does the training.

**Key decisions**
- **Train only the action expert**, keep the vision-language part frozen (SmolVLA's defaults): with little
  data, change how it moves, not how it sees.
- **Short run: 2000 steps, batch 8**, learning-rate warm-up 100 steps and decay over the whole run
  (SmolVLA's defaults are for 30 000-step runs).
- **Image augmentation on**, so it cannot memorise exact pictures.

## 7 · Benchmark — `scripts/vla_benchmark.py`

**Key decisions**
- **One fixed protocol** for the baseline and the post-trained model; only the model path changes.
- **25 episodes, not 10.** With 10, a 7/10 result has a 95% interval of 40–89% — too wide to show an
  improvement.
- **`n_action_steps=10`:** SmolVLA plans 50 actions per call; executing 10 before re-planning cut
  evaluation from ~328 s to ~82 s per episode. (Same setting for both models.)

---

## What didn't work

- **Clips 01–04** were recorded without the marker sheet — no metric scale, dropped.
- **First tracking run found no object**: it searched for blue; the object is yellow.
- **The first 19 clips did not match the task** (different motion and distances), so retargeted replays
  failed. I re-filmed clips 20–45 as a lift-carry-drop into a container, which brought the scale to ~1.
- **Robot-body gaps** (gripper speed, dragging the can, catching the basket rim) — each needed a setting
  in step 4.
- **LIBERO is not importable after `pip install -e`** in Colab; scripts are run with
  `PYTHONPATH=/content/LIBERO`.
- **Height is a rule, not data.** It is the main part of the motion that does not come from my videos.
- **Post-training was not completed in time.** Next step: `train_vla.py` (~1 h on a T4), then
  `vla_benchmark.py --policy …` on the same 25 layouts, compared with the baseline.

## Limitations

- The replays use the simulator's true object positions to place the anchors (privileged information).
  The trained SmolVLA does not: it only sees camera images, the robot state and the instruction.
- One task and few test episodes: the baseline number is indicative, see its confidence interval.

## Run it yourself

| notebook | runtime | what it shows |
|---|---|---|
| [01 · track and replay](notebooks/01_track_and_replay.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Suhail-Halaby/humanoidRLassignment/blob/main/notebooks/01_track_and_replay.ipynb) | CPU | one clip: tracking → robot replay (~10 min); all clips → training demos |
| [02 · SmolVLA](notebooks/02_smolvla.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Suhail-Halaby/humanoidRLassignment/blob/main/notebooks/02_smolvla.ipynb) | GPU | baseline benchmark; dataset → post-training → benchmark (not run, see step 6) |

**Data access.** My clips, demos and results are in a shared Google Drive folder: FILL-LINK
1. Open the link → click the folder name at the top → **Organize → Add shortcut → My Drive**.
   The shortcut must be called `humanoid` (the default) and sit directly in My Drive.
2. Open a notebook with the Colab badge and run it top to bottom; allow Drive access when asked.
   Nothing is written to the shared folder; outputs go to `/content/out`.

Scripts, in order: `make_markers.py` → `track.py` → `replay.py --record` → `to_lerobot.py` → `train_vla.py`
→ `vla_benchmark.py`. Final replay settings:
`--settle 30 --lift 0.10 --clear 0.25 --release-above 0.20 --ramp 0.03`. Untested: `train_vla.py` and the
post-trained benchmark.
