"""Retargeting: a tracked hand path on the paper sheet -> a gripper path on the robot's table.

The phone gives, per video frame, where the pinch point was on the sheet (cm, 2D) and whether the
fingers were closed on the object. The robot needs targets in its own frame (m, 3D) at 20 steps/s.

How (retarget):
    1. Two anchors place the sheet on the robot's table:
           where your pinch was when you GRASPED  -> where the sim object is
           where your pinch was when you RELEASED -> above the sim target (e.g. the basket)
       Two point pairs fix exactly one rotation + shift + uniform scale. Every other point of your path
       goes through the same mapping, so the shape, curvature and timing of your motion are kept.
    2. Height is not in the video (camera looks down), so it is rebuilt from the grasp timing:
       hover on approach -> down to the object as you get close -> up to a safe carry height ->
       down over the target as you get close -> up again after release.
    3. Time: video frames (e.g. 30 fps) are resampled to the robot's 20 steps per second.

How (execute): one robot step per path point; at each grasp/release moment the robot first gets
exactly to the anchor point, then waits for its fingers to finish moving (a Panda gripper takes
~0.5 s, a human hand much less -- this wait is the one place where we adapt to the robot's body).

Typical use:
    track = HandTrack.load("data/processed/clip07.npz")
    obs = sim.reset(env, init_state)
    plan = retarget(track, sim.object_m(obs, "alphabet_soup_1"), sim.object_m(obs, "basket_1"))
    obs, frames, success = execute(env, plan, obs)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .handtrack import HandTrack


# --------------------------------------------------------------------------- result container
@dataclass
class RobotPlan:
    """A gripper path for the robot, one row per simulator step.

    Fields:
        target_m:     np.ndarray (T,3) -- gripper target per step, robot world frame, m
        grip:         np.ndarray (T,)  -- gripper command per step (-1 open, +1 closed)
        frame:        np.ndarray int (T,) -- video frame each row came from (to compare with the clip)
        grasp_m:      np.ndarray (3,) -- exact point where the gripper closes (on the object)
        release_m:    np.ndarray (3,) -- exact point where the gripper opens (above the target)
        scale:        float -- robot metres per sheet metre (1.0 = same size as your motion)
        rotation_deg: float -- how far the sheet was turned to line up with the robot's table
        dropped:      int   -- resampled steps skipped because the hand wasn't visible at that moment
        lift_m:       float -- straight-up lift right after the grasp, before any sideways motion, m (0 = none)
    """
    target_m: np.ndarray
    grip: np.ndarray
    frame: np.ndarray
    grasp_m: np.ndarray
    release_m: np.ndarray
    scale: float
    rotation_deg: float
    dropped: int
    lift_m: float = 0.0

    def summary(self) -> str:
        """One-line health check of the plan."""
        return (f"{len(self.target_m)} steps ({len(self.target_m) / 20:.1f}s) | scale {self.scale:.2f} | "
                f"rotation {self.rotation_deg:.0f} deg | {self.dropped} steps dropped (hand not visible)")


# --------------------------------------------------------------------------- 1. placing the sheet
def sheet_cm_to_m(xy_cm: np.ndarray) -> np.ndarray:
    """Sheet coordinates in cm -> metres, with the y axis flipped.
    Why the flip: the sheet's y runs DOWN the page (like an image), so seen from above the table its axes
    are mirrored compared with the robot's (x, y, z-up) frame. Flipping y makes them match, so the mapping
    below only needs to turn, shift and scale -- never mirror your motion.

    Inputs:  xy_cm: np.ndarray (...,2) -- sheet coordinates, cm
    Returns: np.ndarray (...,2) -- metres, y flipped
    """
    xy = np.asarray(xy_cm, float) / 100.0
    return np.stack([xy[..., 0], -xy[..., 1]], axis=-1)


def fit_anchor_map(src_a: np.ndarray, src_b: np.ndarray, dst_a: np.ndarray, dst_b: np.ndarray):
    """The rotation + uniform scale + shift that sends src_a -> dst_a and src_b -> dst_b (2D).
    Why complex numbers: a 2D point (x, y) can be written as x + iy; then "rotate by angle t and scale by s"
    is just multiplying by one complex number s*e^(it). So the whole mapping is  z_robot = k * z_sheet + c,
    and two point pairs give two equations for the two unknowns k and c.

    Inputs:
        src_a, src_b: np.ndarray (2,) -- two points on the sheet, m (after sheet_cm_to_m)
        dst_a, dst_b: np.ndarray (2,) -- where they must land on the robot's table, m (x, y)
    Returns:
        (k, c): complex, complex -- mapping parameters for apply_anchor_map
    Raises:
        ValueError if the two sheet points are (almost) the same point (the direction would be undefined)
    """
    za, zb = complex(*src_a), complex(*src_b)
    wa, wb = complex(*dst_a), complex(*dst_b)
    if abs(zb - za) < 1e-3:                      # less than 1 mm between grasp and release on the sheet
        raise ValueError("grasp and release points are too close together to define a direction")
    k = (wb - wa) / (zb - za)
    return k, wa - k * za


def apply_anchor_map(k: complex, c: complex, xy_m: np.ndarray) -> np.ndarray:
    """Apply the mapping from fit_anchor_map to many points.

    Inputs:  k, c: complex; xy_m: np.ndarray (N,2) -- sheet points, m (after sheet_cm_to_m)
    Returns: np.ndarray (N,2) -- robot table points (x, y), m
    """
    z = k * (xy_m[:, 0] + 1j * xy_m[:, 1]) + c
    return np.stack([z.real, z.imag], axis=1)


# --------------------------------------------------------------------------- 2. height
def _ramp(dist_m: np.ndarray, low: float, high: float, ramp_m: float) -> np.ndarray:
    """Height that is `low` at distance 0 and rises linearly to `high` at distance ramp_m (and stays there).

    Inputs:  dist_m: np.ndarray (N,) horizontal distances, m; low, high: float heights, m; ramp_m: float, m
    Returns: np.ndarray (N,) heights, m
    """
    return low + (high - low) * np.clip(dist_m / ramp_m, 0.0, 1.0)


def height_profile(xy_m: np.ndarray, phase: np.ndarray, grasp_xy: np.ndarray, release_xy: np.ndarray,
                   z_grasp: float, z_hover: float, z_carry: float, z_release: float,
                   ramp_m: float, lift_m: float = 0.0) -> np.ndarray:
    """Rebuild the gripper height from where the hand is relative to the grasp and release points.
    Why distance-based and not time-based: the descent then happens exactly where your hand converged on
    the object, at your speed -- not at a fixed moment we would have to guess.

    Inputs:
        xy_m:       np.ndarray (N,2) -- gripper path on the robot table, m
        phase:      np.ndarray int (N,) -- 0 before the grasp, 1 while carrying, 2 after the release
        grasp_xy, release_xy: np.ndarray (2,) -- anchor points on the robot table, m
        z_grasp:    float -- height at which the fingers close (object centre), m
        z_hover:    float -- approach height far from the object, m
        z_carry:    float -- carrying height, high enough to clear the target, m
        z_release:  float -- height at which the fingers open over the target, m
        ramp_m:     float -- horizontal distance over which the gripper goes down / up, m
        lift_m:     float -- while carrying, never go lower than this far above the grasp height, m
                    (why: after a straight-up lift the path must not dip back down towards the table)
    Returns:
        np.ndarray (N,) -- height per step, m
    """
    d_grasp = np.linalg.norm(xy_m - grasp_xy, axis=1)
    d_release = np.linalg.norm(xy_m - release_xy, axis=1)
    z = np.empty(len(xy_m))
    before, carry, after = phase == 0, phase == 1, phase == 2
    z[before] = _ramp(d_grasp[before], z_grasp, z_hover, ramp_m)
    # while carrying: rise when leaving the object, sink when arriving at the target; lower of the two
    z[carry] = np.minimum(_ramp(d_grasp[carry], min(z_grasp + lift_m, z_carry), z_carry, ramp_m),
                          _ramp(d_release[carry], z_release, z_carry, ramp_m))
    z[after] = _ramp(d_release[after], z_release, z_carry, ramp_m)
    return z


# --------------------------------------------------------------------------- whole retarget
def _nearest_seen(xy: np.ndarray, i: int) -> int:
    """Index of the frame closest to i where xy is not NaN.

    Inputs:  xy: np.ndarray (N,2); i: int
    Returns: int frame index
    """
    seen = np.flatnonzero(~np.isnan(xy).any(axis=1))
    return int(seen[np.argmin(np.abs(seen - i))])


def retarget(track: HandTrack, obj_m: np.ndarray, place_m: np.ndarray, hz: float = 20.0,
             speed: float = 1.0, hover_m: float = 0.10, clear_m: float = 0.15,
             release_above_m: float = 0.10, ramp_m: float = 0.05, lift_m: float = 0.0) -> RobotPlan:
    """Turn one tracked clip into a robot gripper path for one simulator layout.

    Inputs:
        track:           HandTrack -- must contain a grasp (track.t_close >= 0)
        obj_m:           np.ndarray (3,) -- sim object position (what you picked up), m
        place_m:         np.ndarray (3,) -- sim target position (where it goes, e.g. basket), m
        hz:              float -- simulator steps per second
        speed:           float -- 1.0 = your speed; 0.5 = half speed (twice as many steps, robot keeps up better)
        hover_m:         float -- approach height above the object, m
        clear_m:         float -- carry height above the higher of object / target, m
        release_above_m: float -- release height above the target position, m
        ramp_m:          float -- horizontal distance over which the gripper descends / rises, m
        lift_m:          float -- lift straight up this far right after the grasp, before following your
                         path sideways, m (0 = off). Why: a slow robot that moves sideways while still
                         low drags the object along the table or knocks it over.
    Returns:
        RobotPlan
    Raises:
        ValueError if the clip has no grasp, or grasp and release are at the same place
    """
    if track.t_close < 0:
        raise ValueError("clip has no grasp")
    pinch = track.pinch_cm

    # 1. anchors: your pinch point at the grasp / release moments (nearest frame where the hand was seen)
    i_close = _nearest_seen(pinch, track.t_close)
    i_open = _nearest_seen(pinch, track.t_open)
    k, c = fit_anchor_map(sheet_cm_to_m(pinch[i_close]), sheet_cm_to_m(pinch[i_open]),
                          obj_m[:2], place_m[:2])

    # 3. resample time: one row per robot step, taken from the nearest video frame
    seen = np.flatnonzero(~np.isnan(pinch).any(axis=1))
    t0_s, t1_s = seen[0] / track.fps, seen[-1] / track.fps
    frames = np.round(np.arange(t0_s, t1_s, speed / hz) * track.fps).astype(int)
    ok = ~np.isnan(pinch[frames]).any(axis=1)
    # Missing data is NOT filled in: the robot simply heads straight for the next frame where your hand was seen.
    dropped = int((~ok).sum())
    frames = frames[ok]

    xy = apply_anchor_map(k, c, sheet_cm_to_m(pinch[frames]))
    phase = np.where(frames < track.t_close, 0, np.where(frames <= track.t_open, 1, 2))

    # 2. heights
    z_grasp = float(obj_m[2])
    z_carry = max(obj_m[2], place_m[2]) + clear_m
    z_release = min(place_m[2] + release_above_m, z_carry)
    z = height_profile(xy, phase, obj_m[:2], place_m[:2], z_grasp, z_grasp + hover_m,
                       z_carry, z_release, ramp_m, lift_m)

    return RobotPlan(target_m=np.column_stack([xy, z]),
                     grip=np.where(phase == 1, 1.0, -1.0),
                     frame=frames,
                     grasp_m=np.array([obj_m[0], obj_m[1], z_grasp]),
                     release_m=np.array([place_m[0], place_m[1], z_release]),
                     scale=float(abs(k)), rotation_deg=float(np.degrees(np.angle(k))),
                     dropped=dropped, lift_m=lift_m)


# --------------------------------------------------------------------------- running it on the robot
def execute(env, plan: RobotPlan, obs: dict, settle_steps: int = 15, tail_steps: int = 20):
    """Drive the simulated robot through a plan and report whether LIBERO counted the task as done.

    Inputs:
        env:          LIBERO env (from sim.make_env), already reset
        plan:         RobotPlan
        obs:          dict -- current observation
        settle_steps: int -- steps to wait for the fingers at each grasp / release
        tail_steps:   int -- steps to keep still at the end (lets a dropped object land)
    Returns:
        (obs, frames, success): last observation; list of uint8 (H,W,3) camera images;
        bool -- True if LIBERO reported the task complete at any step
    """
    from . import sim

    frames, success = [], False

    def step(action):
        # one simulator step; keep the image and remember if the task was ever completed
        nonlocal obs, success
        obs, _, done, _ = env.step(action)
        frames.append(sim.camera(obs))
        success = success or bool(done)

    for i in range(len(plan.target_m)):
        if i > 0 and plan.grip[i] != plan.grip[i - 1]:
            # grasp or release moment: reach the exact anchor point first, then let the fingers move
            anchor = plan.grasp_m if plan.grip[i] > 0 else plan.release_m
            obs, f, _ = sim.go_to(env, anchor, plan.grip[i - 1], obs, tol_m=0.01, max_steps=60)
            frames += f
            for _ in range(settle_steps):
                step(np.array([0, 0, 0, 0, 0, 0, plan.grip[i]]))
            if plan.grip[i] > 0 and plan.lift_m > 0:
                # just grasped: lift straight up before following the path sideways
                obs, f, _ = sim.go_to(env, plan.grasp_m + [0, 0, plan.lift_m], plan.grip[i], obs,
                                      tol_m=0.01, max_steps=60)
                frames += f
        step(sim.action_toward(sim.eef_m(obs), plan.target_m[i], plan.grip[i]))
    for _ in range(tail_steps):
        step(np.array([0, 0, 0, 0, 0, 0, plan.grip[-1]]))
    return obs, frames, success