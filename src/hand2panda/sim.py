"""LIBERO simulator setup: create a task, reset it, read the robot's state, and move the gripper to a point.

This is the robot-side counterpart of calib.py. calib.py tells us where things are on the paper sheet (cm);
this module tells us where things are in the simulator (metres) and how to make the gripper go there.

What it provides:
    make_env()          -- one LIBERO task, ready to step, plus its instruction and its fixed start states
    reset()             -- put the scene into one of those start states and let the objects settle
    eef_m(), gripper_width_m(), object_m(), camera()  -- read what we need out of an observation
    action_toward()     -- "I want the gripper at point P" -> one 7-number LIBERO action
    go_to(), wait()     -- repeat actions until the gripper gets there / hold still
    probe_controller()  -- measure how the robot actually responds to actions (step size, gripper sign)

Units: LIBERO works in metres, in a fixed "world frame" (z points up). Names ending in _m are metres.

An action is 7 numbers, each in [-1, 1]:
    [dx, dy, dz, droll, dpitch, dyaw, gripper]
The first three ask the gripper to move by up to a few cm per step (the controller scales them),
the next three ask it to rotate (we always send 0 = keep pointing down), the last one opens/closes the fingers.
The simulator takes 20 steps per second.

Typical use:
    env, info = make_env("libero_object", task_id=0)
    obs = reset(env, info["init_states"][0])
    obs, frames, ok = go_to(env, eef_m(obs) + [0, 0, -0.10], grip=OPEN)
"""
from __future__ import annotations

import os

import numpy as np

# Gripper commands. In robosuite (the library under LIBERO) -1 opens and +1 closes.
# probe_controller() checks this in the real simulator rather than trusting it.
OPEN, CLOSE = -1.0, 1.0

CONTROL_HZ = 20          # LIBERO's default: 20 actions per simulated second


# --------------------------------------------------------------------------- creating the environment
def _ensure_libero_config() -> None:
    """Write LIBERO's config file if it doesn't exist yet.
    Why: on first import LIBERO stops and waits for keyboard answers (Y/N questions about a dataset folder).
    That hangs a script, and on Colab it happens again after every runtime reset because the file is lost.
    We write the same default paths LIBERO would write itself, so the questions never come up.

    Inputs:  none (reads the LIBERO_CONFIG_PATH environment variable if set, otherwise uses ~/.libero)
    Returns: nothing; creates <config dir>/config.yaml if missing
    """
    import importlib.util
    import yaml

    cfg_dir = os.environ.get("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero"))
    cfg_file = os.path.join(cfg_dir, "config.yaml")
    if os.path.exists(cfg_file):
        return
    # find where the libero.libero package is installed, without running its import-time prompt
    root = os.path.dirname(importlib.util.find_spec("libero.libero").origin)
    os.makedirs(cfg_dir, exist_ok=True)
    with open(cfg_file, "w") as f:
        yaml.safe_dump({"benchmark_root": root,
                        "bddl_files": os.path.join(root, "bddl_files"),
                        "init_states": os.path.join(root, "init_files"),
                        "datasets": os.path.join(root, "../datasets"),
                        "assets": os.path.join(root, "assets")}, f)


def make_env(suite: str = "libero_object", task_id: int = 0, img_px: int = 256, seed: int = 0):
    """Create one LIBERO task.

    Inputs:
        suite:   str -- benchmark suite: "libero_object", "libero_spatial", "libero_goal", "libero_10", ...
        task_id: int -- which task in the suite (0-9)
        img_px:  int -- camera image height = width, pixels
        seed:    int -- random seed for the simulator
    Returns:
        (env, info):
            env:  LIBERO OffScreenRenderEnv -- call env.step(action) on it
            info: dict with
                "language":    str -- the task instruction, e.g. "pick up the alphabet soup and place it in the basket"
                "name":        str -- the task's short name
                "init_states": np.ndarray (K, S) -- K fixed start states the benchmark evaluates from
                               (each is the full simulator state flattened to S numbers)
    """
    # MuJoCo must be told to draw images without a screen BEFORE it is imported (Colab has no screen)
    os.environ.setdefault("MUJOCO_GL", "egl")
    _ensure_libero_config()
    import torch
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task = benchmark.get_benchmark_dict()[suite]().get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=img_px, camera_widths=img_px)
    env.seed(seed)

    # The start states are stored with torch.save. Recent torch versions refuse to load plain numpy
    # arrays unless weights_only=False, which breaks LIBERO's own loader, so we load the file directly.
    path = os.path.join(get_libero_path("init_states"), task.problem_folder, task.init_states_file)
    init_states = np.asarray(torch.load(path, weights_only=False))
    return env, {"language": task.language, "name": task.name, "init_states": init_states}


def reset(env, init_state: np.ndarray | None = None, settle_steps: int = 10) -> dict:
    """Start a new episode, optionally from one of the benchmark's fixed start states.
    Why settle: objects are placed slightly above the table and drop into place during the first steps;
    if the robot moves while they are still falling, their positions are not yet the real ones.

    Inputs:
        env:          LIBERO env from make_env
        init_state:   np.ndarray (S,) or None -- one row of info["init_states"]; None = random layout
        settle_steps: int -- steps to wait (robot still, gripper open) before returning
    Returns:
        dict -- the observation after settling (keys listed by print_obs)
    """
    obs = env.reset()
    if init_state is not None:
        obs = env.set_init_state(init_state)
    obs, _ = wait(env, settle_steps, OPEN, record=False)
    return obs


# --------------------------------------------------------------------------- reading observations
def eef_m(obs: dict) -> np.ndarray:
    """Position of the gripper ("end effector"), the point between the two fingertips.

    Inputs:  obs: dict -- LIBERO observation
    Returns: np.ndarray float64 (3,) -- (x, y, z) world frame, metres
    """
    return np.asarray(obs["robot0_eef_pos"], dtype=float)


def gripper_width_m(obs: dict) -> float:
    """Approximate gap between the two fingers.
    Why it's a difference: the two finger joints move in opposite directions, so one reads +, the other -.

    Inputs:  obs: dict -- LIBERO observation
    Returns: float -- finger gap, metres (about 0.08 fully open, near 0 fully closed on nothing)
    """
    q = obs["robot0_gripper_qpos"]
    return float(abs(q[0] - q[1]))


def object_m(obs: dict, name: str) -> np.ndarray:
    """Position of a named object, e.g. "alphabet_soup_1" or "basket_1" (see print_obs for the names).

    Inputs:  obs: dict; name: str -- object name without the "_pos" suffix
    Returns: np.ndarray float64 (3,) -- object centre, world frame, metres
    """
    return np.asarray(obs[f"{name}_pos"], dtype=float)


def camera(obs: dict, which: str = "agentview") -> np.ndarray:
    """Camera image the right way up.
    Why the flip: MuJoCo's renderer returns images upside down.

    Inputs:  obs: dict; which: "agentview" (fixed camera facing the robot) or "eye_in_hand" (wrist camera)
    Returns: np.ndarray uint8 (H, W, 3) RGB
    """
    key = "agentview_image" if which == "agentview" else "robot0_eye_in_hand_image"
    return np.ascontiguousarray(obs[key][::-1])


def print_obs(obs: dict) -> None:
    """Print every observation key with its shape, and the objects in the scene with their positions.
    Why: object names differ per task; this is how you find out what to pass to object_m.

    Inputs:  obs: dict -- LIBERO observation
    Returns: nothing (prints)
    """
    for k, v in obs.items():
        print(f"  {k:40s} {np.shape(v)}")
    names = [k[:-4] for k in obs if k.endswith("_pos") and "_to_" not in k and not k.startswith("robot0")]
    print("objects (world frame, m):")
    for n in names:
        print(f"  {n:30s} {np.round(object_m(obs, n), 3)}")


# --------------------------------------------------------------------------- moving the robot
def action_toward(eef_now_m: np.ndarray, target_m: np.ndarray, grip: float,
                  m_per_unit: float = 0.05, gain: float = 1.0) -> np.ndarray:
    """One action that pushes the gripper from where it is toward a target point.

    How: the controller reads an action value of 1.0 as "move m_per_unit metres this step" (0.05 m by default
    in robosuite; probe_controller measures what really happens). So the action is the remaining distance
    divided by m_per_unit. Far away -> large action; close -> small action, so the gripper slows down as it
    arrives instead of overshooting. Because the gap is re-measured every step, small errors correct themselves.

    Why we shrink all three numbers together instead of clipping each to [-1,1] separately: clipping each axis
    on its own would change the direction of travel (the gripper would first slide along one axis, then another).

    Inputs:
        eef_now_m:  np.ndarray (3,) -- current gripper position, m
        target_m:   np.ndarray (3,) -- where we want it, m
        grip:       float in [-1,1] -- gripper command (OPEN or CLOSE)
        m_per_unit: float -- metres per action unit (controller scale)
        gain:       float -- below 1 = gentler approach, above 1 = more aggressive
    Returns:
        np.ndarray float64 (7,) -- [dx, dy, dz, 0, 0, 0, grip], each in [-1, 1]
    """
    d = gain * (np.asarray(target_m, float) - np.asarray(eef_now_m, float)) / m_per_unit
    biggest = np.abs(d).max()
    if biggest > 1.0:
        d = d / biggest
    return np.concatenate([d, [0.0, 0.0, 0.0, grip]])


def wait(env, steps: int, grip: float, record: bool = True):
    """Keep the arm still for a number of steps, e.g. to let the fingers finish closing.

    Inputs:
        env: LIBERO env; steps: int; grip: float in [-1,1]; record: bool -- keep camera frames
    Returns:
        (obs, frames): last observation dict, list of uint8 (H,W,3) images (empty if record=False)
    """
    frames, obs = [], None
    for _ in range(steps):
        obs, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, grip], float))
        if record:
            frames.append(camera(obs))
    return obs, frames


def go_to(env, target_m: np.ndarray, grip: float, obs: dict | None = None, tol_m: float = 0.005,
          max_steps: int = 100, m_per_unit: float = 0.05, record: bool = True):
    """Move the gripper to a point, step by step, until it is within tol_m of it.

    Inputs:
        env:        LIBERO env
        target_m:   np.ndarray (3,) -- target, world frame, m
        grip:       float -- gripper command held during the move
        obs:        dict or None -- current observation (None = take one zero-motion step to get it)
        tol_m:      float -- "arrived" distance, m (5 mm)
        max_steps:  int -- give up after this many steps (why: a target outside the robot's reach, or
                    behind an object, would otherwise loop forever)
        m_per_unit: float -- passed to action_toward
        record:     bool -- keep camera frames
    Returns:
        (obs, frames, reached): last observation; list of uint8 (H,W,3) images; bool, True if within tol_m
    """
    frames = []
    if obs is None:
        obs, frames = wait(env, 1, grip, record)
    for _ in range(max_steps):
        if np.linalg.norm(eef_m(obs) - target_m) < tol_m:
            return obs, frames, True
        obs, _, _, _ = env.step(action_toward(eef_m(obs), target_m, grip, m_per_unit))
        if record:
            frames.append(camera(obs))
    return obs, frames, bool(np.linalg.norm(eef_m(obs) - target_m) < tol_m)


# --------------------------------------------------------------------------- checking the controller
def probe_controller(env, init_state: np.ndarray | None = None, steps: int = 10) -> dict:
    """Measure how the robot really responds to actions, instead of trusting documentation.
    This plays the same role for the robot that the ArUco sheet plays for the camera.

    Test 1 (movement): from the same start, push action = +1 along x, then y, then z, for `steps` steps each,
    and record how far the gripper moves per step. Only the second half of the steps is used,
    because the arm needs a few steps to speed up.
    Test 2 (gripper): send -1 for 15 steps and read the finger gap, then +1 for 15 steps and read it again.

    Inputs:
        env:        LIBERO env
        init_state: np.ndarray (S,) or None -- start state used for every test
        steps:      int -- steps per movement test
    Returns:
        dict with
            "move_m_per_step": np.ndarray (3,3) -- row i = commanded axis (x,y,z), column j = measured movement
                               along x,y,z per step, m. Ideally large on the diagonal, ~0 elsewhere.
            "m_per_unit":      float -- mean of the diagonal: metres moved per step per action unit at full speed
            "width_after_minus1_m", "width_after_plus1_m": float -- finger gap after each gripper command, m
            "close_cmd":       float -- whichever of -1 / +1 gave the smaller gap (should equal CLOSE)
    """
    move = np.zeros((3, 3))
    for axis in range(3):
        obs = reset(env, init_state)
        a = np.zeros(7)
        a[axis], a[6] = 1.0, OPEN
        track = [eef_m(obs)]
        for _ in range(steps):
            obs, _, _, _ = env.step(a)
            track.append(eef_m(obs))
        track = np.array(track)
        half = steps // 2
        move[axis] = (track[-1] - track[half]) / (steps - half)

    obs = reset(env, init_state)
    obs, _ = wait(env, 15, -1.0, record=False)
    w_minus = gripper_width_m(obs)
    obs, _ = wait(env, 15, +1.0, record=False)
    w_plus = gripper_width_m(obs)
    return {"move_m_per_step": move, "m_per_unit": float(np.mean(np.diag(move))),
            "width_after_minus1_m": w_minus, "width_after_plus1_m": w_plus,
            "close_cmd": 1.0 if w_plus < w_minus else -1.0}