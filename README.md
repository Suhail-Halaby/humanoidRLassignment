# humanoidRLassignment — phone-recorded hand demos, Panda in LIBERO, post-trained SmolVLA

I recorded myself picking up a yellow shuttlecock and dropping it into a container with a phone.
This repo turns those videos into robot demonstrations for the LIBERO task
*"pick up the alphabet soup and place it in the basket"* (`libero_object`, task 0),
and uses them to post-train SmolVLA. 

Objective question: **does a VLA get better at the task after
training on demonstrations that came from my hand?**

```
phone video -> 1 calibrate table -> 2 track hand + object -> 3 retarget to robot table
           -> 4 replay in LIBERO -> 5 record successful replays -> 6 post-train SmolVLA -> 7 benchmark
```


## Run it yourself

| notebook | runtime | what it shows |
|---|---|---|
| [01 · track and replay](notebooks/01_track_and_replay.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Suhail-Halaby/humanoidRLassignment/blob/main/notebooks/01_track_and_replay.ipynb) | CPU | one clip: tracking → robot replay (~10 min) |
| [02 · SmolVLA](notebooks/02_smolvla.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Suhail-Halaby/humanoidRLassignment/blob/main/notebooks/02_smolvla.ipynb) | GPU | baseline vs post-trained results (saved results shown; rerun optional) |

**Data access.** My clips, demos and results are in a shared Google Drive folder:
<https://drive.google.com/drive/folders/15pZnqmGep8M9QO_Sku8k7tmXop1WN00A?usp=drive_link>
1. Open the link → click the folder name at the top → **Organize → Add shortcut → My Drive**.
   The shortcut must be called `humanoid` (the default) and sit directly in My Drive.
2. Open a notebook with the Colab badge and run it top to bottom; allow Drive access when asked.
   Nothing is written to the shared folder; outputs go to `/content/out`.

## 1 · Calibrating the table — `src/hand2panda/calib.py`, `scripts/make_markers.py`

**What:** an A4 sheet with four ArUco markers lies under the scene. In every
frame the markers are detected and a homography is fitted that converts
pixels into centimetres on the sheet.

**Key decisions**
- **Markers instead of camera calibration.** A phone's lens settings are unknown, no lidar; four markers at known
  positions give metric table coordinates directly, with no calibration step.
- **Measure the printed marker with a ruler** (`--marker-cm`). Printers often shrink pages slightly; every
  length scales by the same factor, so one measurement corrects everything.
- **Handle a moving phone.** If the markers move more than 3 px during a clip, a mapping is fitted for every
  frame; otherwise all frames are pooled into one mapping (robust when my hand covers a marker).
- **Check it.** The reprojection error (detected marker corners vs. their true positions) was
  0.02–0.08 cm on my clips.

## 2 · Tracking the hand and the object — `src/hand2panda/handtrack.py`, `scripts/track.py`

**What:** MediaPipe finds 21 hand landmarks per frame. The pinch point (midway between thumb tip and
index tip) stands in for the robot gripper. The bright yellow object is found by colour (albeit unreliably, scope for future improvement). Both are converted to
table cm with step 1.

**Key decisions**
- **Pinch point as the gripper.** It is where a held object sits, like on the Panda robot with symmetric gripping.
- **Missing data is handled.** Gaps of up to 0.3 s are interpolated. Longer ones stay empty, clips with long gaps would be flagged and discarded by the user.
- **Grasp timing from two signals.** Deter if an object is held by assesing if the object is moving together
  with the hand (unreliable, but shows promise, especially with larger brightly coloured objects). The shuttlecock was an attempt at getting this to work. Analyzing when the fingers are closed and opened comes from the thumb–index gap divided by hand size (precise but noisy). 
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
- **Height is rebuilt by rule.** The phone's perspective means that seeing height is inaccurate. The gripper
  hovers on approach, descends as my hand converges on the object, carries at a safe height and lowers over
  the basket. Descents are triggered by distance to the grasp/release point, so they happen where my hand
  actually homed in, at my speed.
- **Time is resampled** from the phone's 30 fps to LIBERO's 20 steps/s..

**Retargeting options** — the parts of the motion that are rules rather than my hand, all set from the
command line of `replay.py`. The defaults were my first guesses; the final values came from watching
replays fail:

| option | default | final | what it controls | why it changed |
|---|---|---|---|---|
| `--clear` | 0.15 m | 0.25 m | carry height above the higher of can / basket | the can clipped the basket on the way over |
| `--release-above` | 0.10 m | 0.20 m | release height above the basket's position (near its base) | at 0.10 m the gripper opened at rim height and the can caught the rim |
| `--ramp` | 0.05 m | 0.03 m | horizontal distance over which the gripper descends towards the grasp / release point | with 5 cm the gripper started sinking while the can was still over the rim; 3 cm keeps it high until it is over the opening |
| `--lift` | 0 (off) | 0.10 m | rise straight up after the grasp, and never dip below this height while carrying | moving sideways while still low dragged or tipped the can |
| `--speed` | 1.0 | 1.0 | playback speed of my motion (0.5 = half speed) | not needed: the robot kept up at my speed |
| hover height | 0.10 m | 0.10 m | approach height above the can (fixed in code) | — |

Only the shape and timing of the horizontal path and the grasp / release moments come from my video;
everything in this table is a rule. Moving `--release-above` and `--ramp` was what turned clip 20 from a
near miss into a success.

## 4 · Replaying on the simulated Panda — `retarget.py` (`execute`), `src/hand2panda/sim.py`, `scripts/replay.py`

**What:** the robot follows the path one target per step (each action is the remaining distance, scaled,
so it slows down as it arrives). LIBERO reports whether the soup ended up in the basket.

**Key decisions — adapting to the robot's body.** My hand is not a Panda; these were added because
replays failed without them:

| setting | value | why |
|---|---|---|
| `--settle` | 30 steps (1.5 s) | the Panda gripper closes/opens far slower than my fingers |
| `--lift` | 0.10 m | lift straight up after grasping; moving sideways while still low dragged the can |
| `--clear` | 0.25 m | carry height above can/basket |
| `--release-above` | 0.20 m | the default 0.10 m released at rim height and the can caught the rim |
| `--ramp` | 0.03 m | only descend once over the basket opening |

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
- **The first 19 clips did not match the task** (different motion and distances, table perspective/lighting caused issues), so retargeted replays
  failed. I re-filmed clips 20–45 as a lift-carry-drop into a container.
- **Robot-body gaps** (gripper speed, dragging the can, catching the basket rim) — each needed a setting
  in step 4.
- **LIBERO is not importable after `pip install -e`** in Colab; scripts are run with
  `PYTHONPATH=/content/LIBERO`.
- **Height is a rule, not data.** It is the main part of the motion that does not come from my videos.
- TODO: anything from training / evaluation.

## Limitations

- The replays use the simulator's true object positions to place the anchors (privileged information).
  The trained SmolVLA does not: it only sees camera images, the robot state and the instruction.
- One task, 25 test episodes: results are indicative, see the confidence intervals.

## How to run

Two Colab notebooks (TODO: add to `notebooks/`):
1. **`01_track_and_replay.ipynb`** (CPU): tracking, replay, recording demos.
2. **`02_smolvla.ipynb`** (GPU): baseline benchmark, dataset, post-training, benchmark.

```
# notebook 1 (CPU)
python scripts/make_markers.py --out aruco_a4.pdf          # print at 100 %
python scripts/track.py --videos <clips> --out <processed> --colour yellow --marker-cm 5.0
PYTHONPATH=<LIBERO> python scripts/replay.py --tracks <processed> --out <replay> \
    --inits 25,26,27,28,29 --settle 30 --lift 0.10 --clear 0.25 --release-above 0.20 --ramp 0.03 \
    --record <demos>
# notebook 2 (GPU)
python scripts/vla_benchmark.py --out <eval_baseline>
python scripts/to_lerobot.py --demos <demos> --out <dataset>
python scripts/train_vla.py --dataset <dataset> --out <train>
python scripts/vla_benchmark.py --policy <train>/checkpoints/last/pretrained_model --out <eval_post>
```