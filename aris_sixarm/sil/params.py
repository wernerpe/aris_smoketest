"""Parameter objects for the software-in-the-loop simulator.

Every dataclass here is frozen and carries its units in the field docstring.
Nothing in this module reads the environment or mutates process globals: the
ACTIVE rig (`fleet.ACTIVE_RIG`) and the ACTIVE tool (`frames.ACTIVE_TOOL`) are
read once, by the CLI, as DEFAULTS for its flags — the same door `ARIS_RIG` /
`ARIS_TOOL` already open in `aris_sixarm/__init__.py`, and no new one.

FRAMES USED THROUGHOUT THE PACKAGE
  world   the canvas frame of `aris_sixarm.fleet`: z up, z = 0 the paper's
          top surface, origin at the paper corner.  The Drake plant lives here.
  base    the arm's own `fr3_link0`.  The pathway CSV, the equilibrium-pose
          topic, `o_t_ee` and the impedance law all live here (contract §1/§3).
  hand    Drake's `panda_hand` == `frames.link_frames_many(...)[:, 9]`.
  EE      the NOMINAL pen tip, which is what the robot believes its end
          effector is (`setEE`, contract §4).  Its origin is
          hand + R_hand @ (pen_lat, 0, D_HAND_TCP + pen_ext) and its
          orientation is the hand's; the pen axis is EE +Z.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Sequence

import numpy as np

from .. import fleet, frames

CONFIG_DIR = Path(__file__).resolve().parent / "config"
DEFAULT_CONTROLLER_YAML = CONFIG_DIR / "cartesian_impedance_controller.yaml"

# The tool offsets `frames` knows, resolved WITHOUT touching `frames.PEN_LAT` /
# `frames.PEN_EXT_ACTIVE` (process globals that a SIL run has no business
# rewriting).  Values and provenance: `aris_sixarm/frames.py`.
_TOOL_OFFSETS = {
    "inline": (0.0, frames.PEN_EXT),                       # gate-B measured
    "lateral": (frames.PEN_LAT_HOLDER, frames.PEN_EXT_HOLDER),  # USER-SPECIFIED
}


def _arr(x, n: int, name: str) -> np.ndarray:
    a = np.asarray(x, float).reshape(-1)
    if a.size != n:
        raise ValueError(f"{name} must have {n} entries, got {a.size}")
    return a


@dataclass(frozen=True)
class ArmMount:
    """Where the arm is bolted. `T_world_base` is (4,4), metres, world frame."""

    arm_id: int
    rig: str
    mount: str                      # "floor" | "inv" | "wall" | "synthetic"
    T_world_base: np.ndarray        # (4,4) homogeneous, world <- base

    @staticmethod
    def from_fleet(rig: str, arm_id: int) -> "ArmMount":
        """The mount of one arm of a named rig (`fleet.rig`, nothing mutated)."""
        fl, _ = fleet.rig(rig)
        if arm_id not in fl:
            raise ValueError(f"rig {rig!r} has no arm {arm_id}; "
                             f"it has {sorted(fl)}")
        spec = fl[arm_id]
        return ArmMount(arm_id, rig, spec.mount,
                        np.asarray(spec.T_world_base(), float))

    @staticmethod
    def synthetic(kind: str, h: float, arm_id: int = 0) -> "ArmMount":
        """A single arm on no rig: `floor` upright at z = h, `inverted` hanging
        at z = h (base z axis pointing at the paper).  `h` in metres above the
        paper plane."""
        T = np.eye(4)
        if kind == "floor":
            T[2, 3] = h
        elif kind == "inverted":
            T[:3, :3] = np.diag([1.0, -1.0, -1.0])   # roty(pi) . rotz(pi)
            T[2, 3] = h
        else:
            raise ValueError(f"mount kind must be floor|inverted, got {kind!r}")
        return ArmMount(arm_id, "synthetic",
                        "floor" if kind == "floor" else "inv", T)

    @property
    def T_base_world(self) -> np.ndarray:
        """(4,4) base <- world."""
        R = self.T_world_base[:3, :3]
        T = np.eye(4)
        T[:3, :3] = R.T
        T[:3, 3] = -R.T @ self.T_world_base[:3, 3]
        return T


@dataclass(frozen=True)
class Tool:
    """The pen. All offsets in metres, in the hand-TCP frame of `frames`.

    `offset_tcp` is `frames.tool_offset()` for this tool — the NOMINAL tip, the
    one `setEE` is set to and the one the controller's Jacobian belongs to.
    `tip_error_m` is signed along the pen axis (EE +Z): negative means the
    PHYSICAL tip is closer to the hand than the robot believes (contract §4).
    """

    name: str
    offset_tcp: np.ndarray          # (3,) hand-TCP frame -> nominal tip
    tip_error_m: float = 0.0

    @staticmethod
    def from_frames(name: str, tip_error_m: float = 0.0) -> "Tool":
        if name not in _TOOL_OFFSETS:
            raise ValueError(f"unknown tool {name!r}; "
                             f"want one of {tuple(_TOOL_OFFSETS)}")
        lat, ext = _TOOL_OFFSETS[name]
        return Tool(name, np.array([lat, 0.0, ext]), float(tip_error_m))

    @property
    def nominal_tip_hand(self) -> np.ndarray:
        """(3,) the NOMINAL tip in the `panda_hand` frame."""
        return self.offset_tcp + np.array([0.0, 0.0, frames.D_HAND_TCP])

    @property
    def actual_tip_hand(self) -> np.ndarray:
        """(3,) the PHYSICAL tip in the `panda_hand` frame."""
        return self.nominal_tip_hand + np.array([0.0, 0.0, self.tip_error_m])


@dataclass(frozen=True)
class Paper:
    """The sheet: a thin box welded to the world, top surface at `z_world`.

    `size` (m) is the sheet in the world x/y; `center_xy` (m) its centre.
    Contact is COMPLIANT POINT contact (see `plant.py` for why):
    `point_stiffness` N/m, `dissipation` s/m (Hunt-Crossley), Coulomb friction
    dimensionless.

    `thickness` is the box BELOW the drawing surface and stands for the table
    as much as the sheet.  0.10 m, not 0.002 m, because a thin box is a
    trapdoor: a run that loses the arm drives the 1 mm tip sphere through it
    within one 1 ms step, the contact silently disappears, and
    `max_penetration_m` reads a quarter of a metre (measured on the arm-31 v18
    pathway: 255-341 mm against a 10 mm box, i.e. unmodelled in exactly the
    runs that were failing).  The top face does not move.
    """

    z_world: float = 0.0
    size: tuple = (1.8034, 3.63064)
    center_xy: tuple = (0.9017, 1.81532)
    point_stiffness: float = 1.0e5
    dissipation: float = 20.0
    friction_static: float = 0.6
    friction_dynamic: float = 0.5
    thickness: float = 0.10

    @staticmethod
    def for_rig(rig: str, z_world: float = 0.0, **kw) -> "Paper":
        """The rig's own canvas -- USE THIS FOR `--rig/--arm`.

        `fleet.rig(...)[1]` with the corner at the world origin, which is the
        canvas convention (`fleet` module docstring) and where the rig's base
        poses put the arms.  The class defaults above ARE this sheet for the
        six-arm canvas, so a `Paper()` with no arguments is a rig sheet.
        """
        _, sheet = fleet.rig(rig)
        sx, sy = float(sheet[0]), float(sheet[1])
        return Paper(z_world=z_world, size=(sx, sy),
                     center_xy=(0.5 * sx, 0.5 * sy), **kw)

    @staticmethod
    def under(mount: ArmMount, z_world: float = 0.0, span: float = 4.0,
              **kw) -> "Paper":
        """A `span` x `span` sheet CENTRED ON THE BASE -- use this for
        `--mount`.

        A synthetic mount has no rig and therefore no canvas: its base sits at
        the world origin, so the rig sheet (corner at the origin, extending
        +x/+y) covers only the quadrant the arm happens to reach into -- on the
        real arm-31 pathway that left 61 % of the setpoints over nothing.  4 m
        squared centred on the base covers the FR3's whole 1.07 m tip reach;
        override with `--paper-size` / `--paper-center`.
        """
        p = mount.T_world_base[:3, 3]
        return Paper(z_world=z_world, size=(span, span),
                     center_xy=(float(p[0]), float(p[1])), **kw)

    @property
    def normal_world(self) -> np.ndarray:
        """(3,) the outward normal of the drawing surface. The paper is a
        horizontal table in every rig this package models, so it is +z."""
        return np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class Joints:
    """Joint-level dissipation applied OUTSIDE the impedance law, as the
    hardware would: tau_friction = -viscous * dq - coulomb * sign(dq).

    BOTH ARE UNCALIBRATED AND BOTH DEFAULT TO ZERO.  Nothing in this repo or
    the briefing measures FR3 joint friction; what the briefing gives is its
    CONSEQUENCE at the tip ("~5 N equivalent", 5-20 mm of free-air lag
    collapsing in stick-slip steps, §8.4/§15.1), a tip-space number that
    cannot become seven joint numbers without the identification experiment of
    §15.3 stage 2.  A zero here means "not modelled", not "negligible": the
    simulated arm lands sooner and holds contact better than the real one, and
    every touch fraction this package reports is optimistic for that reason.
    Units: viscous Nm.s/rad, coulomb Nm.
    """

    viscous: np.ndarray = field(
        default_factory=lambda: np.zeros(7))
    coulomb: np.ndarray = field(
        default_factory=lambda: np.zeros(7))

    def torque(self, dq: np.ndarray) -> np.ndarray:
        """-> (7,) the friction torque at joint velocity `dq` (rad/s)."""
        dq = np.asarray(dq, float)
        return -self.viscous * dq - self.coulomb * np.sign(dq)


@dataclass(frozen=True)
class Controller:
    """The C++ controller's own parameters, one field per ROS parameter.

    Units: `k_cartesian` N/m (xyz) and Nm/rad (rxryrz); `d_cartesian` Ns/m and
    Nms/rad; `k_nullspace` Nm/rad, `d_nullspace` Nms/rad; `q_nullspace` rad;
    `tool_tip_offset` m in the EE frame; `max_torques` Nm; `max_torque_rate`
    Nm/s; `max_pose_error_pos` m; `max_pose_error_rot` rad; `filter_alpha`
    dimensionless (per 1 kHz tick); `damping_pinv` the damped-least-squares
    lambda of the nullspace projector.

    `joint_reference_timeout_s` is contract §2's controller-side parameter: a
    joint reference older than this is ignored and the nullspace target falls
    back to the posture latched at activation, i.e. today's behaviour.

    THE NULLSPACE REFERENCE AND ITS CONTINUITY GUARD.
      `nullspace_mode` "latched"   = today's DEPLOYED behaviour (redesign stage
                                     0): `q_nullspace_` is the posture latched
                                     at activation and the CSV's q columns are
                                     ignored entirely.
                       "reference" = contract §2 / redesign stage 1: a fresh
                                     joint reference replaces the target.
      `qref_rate_limit_rad_s` bounds how fast the EFFECTIVE target may move
      toward the received one.  It is a guard this simulator adds, not a
      deployed parameter, and it exists because a plan whose transits are not
      in the file hands the controller a 4.5 rad step between strokes; the
      posture spring then swings the arm into its joint stops at the torque
      clamp -- the 2026-06-12 `joint_velocity_violation` failure mode, in
      software.  <= 0 disables the limit (raw contract §2 behaviour).
      `qref_divergence_rad` is only a REPORTING threshold: every tick where
      the raw reference is further than this from the effective target is
      counted, so a plan that asks for a posture the controller cannot follow
      shows up as a number instead of as a mystery.

      THE GUARD IS NOT FREE, AND IT WAS MEASURED.  On the arm-31 v18 pathway
      BEFORE the exporter emitted its transit rows the reference stepped
      4.7 rad between strokes; at 1.0 rad/s the effective target then lagged
      for 3.8 % of the run and the arm was still pinned when the pen was next
      commanded down.  Once the transits ARE in the file the step is 0.045 rad
      and the guard binds on nothing.  The step is the symptom, the missing
      transit is the disease, and the cure is the file.
    """

    k_cartesian: np.ndarray
    d_cartesian: np.ndarray
    k_nullspace: np.ndarray
    d_nullspace: np.ndarray
    q_nullspace: np.ndarray
    tool_tip_offset: np.ndarray
    max_torques: np.ndarray
    max_torque_rate: float
    max_pose_error_pos: float
    max_pose_error_rot: float
    filter_alpha: float
    damping_pinv: float
    joint_reference_timeout_s: float = 0.5
    nullspace_mode: str = "reference"        # "latched" | "reference"
    qref_rate_limit_rad_s: float = 1.0       # <= 0 disables the guard
    qref_divergence_rad: float = 0.5         # reporting threshold only
    source: str = "<defaults>"

    def __post_init__(self):
        if self.nullspace_mode not in ("latched", "reference"):
            raise ValueError(f"nullspace_mode must be latched|reference, "
                             f"got {self.nullspace_mode!r}")

    @staticmethod
    def from_yaml(path=None) -> "Controller":
        """Load the controller's own YAML (the deployed schema).

        Accepts the deployed nesting
        `/**: cartesian_impedance_controller: ros__parameters: {...}` and a
        flat mapping alike, so `--controller-yaml` can point straight at the
        operator's live file when it is copied off the machine (§2.3 says it
        is unversioned; this package ships a transcription, not a copy).
        `max_torques` empty falls back to the scalar `max_torque`, exactly as
        `on_configure` does.
        """
        import yaml                          # ships with the `sil` extra

        path = Path(DEFAULT_CONTROLLER_YAML if path is None else path)
        doc = yaml.safe_load(path.read_text())
        p = _find_ros_params(doc)
        n_max = float(p.get("max_torque", 5.0))
        mt = p.get("max_torques") or []
        max_torques = _arr(mt, 7, "max_torques") if len(mt) == 7 \
            else np.full(7, n_max)
        return Controller(
            k_cartesian=_arr(p["k_cartesian"], 6, "k_cartesian"),
            d_cartesian=_arr(p["d_cartesian"], 6, "d_cartesian"),
            k_nullspace=_arr(p["k_nullspace"], 7, "k_nullspace"),
            d_nullspace=_arr(p["d_nullspace"], 7, "d_nullspace"),
            q_nullspace=_arr(p["q_nullspace"], 7, "q_nullspace"),
            tool_tip_offset=_arr(p.get("tool_tip_offset", [0, 0, 0]), 3,
                                 "tool_tip_offset"),
            max_torques=max_torques,
            max_torque_rate=float(p["max_torque_rate"]),
            max_pose_error_pos=float(p["max_pose_error_pos"]),
            max_pose_error_rot=float(p["max_pose_error_rot"]),
            filter_alpha=float(p["filter_alpha"]),
            damping_pinv=float(p["damping_pinv"]),
            joint_reference_timeout_s=float(
                p.get("joint_reference_timeout_s", 0.5)),
            nullspace_mode=str(p.get("nullspace_mode", "reference")),
            qref_rate_limit_rad_s=float(p.get("qref_rate_limit_rad_s", 1.0)),
            qref_divergence_rad=float(p.get("qref_divergence_rad", 0.5)),
            source=str(path))

    def with_k(self, k: Sequence[float]) -> "Controller":
        """-> a copy with a different `k_cartesian` (the per-phase K of
        briefing §8.4, which the executor sets over the parameter service)."""
        return replace(self, k_cartesian=_arr(k, 6, "k_cartesian"))


def _find_ros_params(doc, top: bool = True) -> dict:
    """Dig the `ros__parameters` block out of a controller YAML.

    The deployed nesting is `/**: <controller>: ros__parameters: {...}`; a flat
    mapping of the same keys is accepted too, so a hand-written override does
    not need the ROS scaffolding.  Identified by `k_cartesian`, the one key no
    other block in that file has.
    """
    if isinstance(doc, dict):
        if "k_cartesian" in doc:
            return doc
        for value in doc.values():
            found = _find_ros_params(value, top=False)
            if found is not None:
                return found
    if top:
        raise ValueError("no ros__parameters block with k_cartesian in this yaml")
    return None


@dataclass(frozen=True)
class Walker:
    """The executor's press-into-paper strategy, as an OPEN LOOP.

    Each field is named after the `RTFF_*` environment variable or CLI flag it
    mimics (operator briefing §8.5, live settings §9).  Metres and m/s.

      hover_m           RTFF_HOVER          pen-up height above the plane
      touchdown_speed   RTFF_TDOWN_SPEED    descent in the slow zone
      touchdown_fast    RTFF_TDOWN_FAST     descent above the slow zone
      touchdown_slowzone RTFF_TDOWN_SLOWZONE height at which it slows
      press_m           --press             commanded depth past the plane
      dmax_m            RTFF_DMAX / --d-max the PAPER's protection: press cap
      draw_speed        --draw-speed
      travel_speed      --travel-speed
      lift_max_m        RTFF_LIFT_MAX       minimum lift between strokes; the
                        briefing's rule is lift > press or the pen drags, and
                        `__post_init__` refuses the pair that breaks it.
    """

    hover_m: float = 0.030
    touchdown_speed: float = 0.002
    touchdown_fast: float = 0.020
    touchdown_slowzone: float = 0.002
    press_m: float = 0.010
    dmax_m: float = 0.012
    draw_speed: float = 0.02
    travel_speed: float = 0.04
    lift_max_m: float = 0.015

    def __post_init__(self):
        if self.lift_max_m <= self.press_m:
            raise ValueError(
                f"lift_max_m ({self.lift_max_m}) must exceed press_m "
                f"({self.press_m}) or the pen drags between strokes "
                "(briefing §9)")
        if self.press_m > self.dmax_m:
            raise ValueError(
                f"press_m ({self.press_m}) exceeds dmax_m ({self.dmax_m}); "
                "the depth cap is the paper's protection (briefing §12.4)")

    @property
    def press_applied_m(self) -> float:
        """The press the walker actually commands, after the DMAX cap."""
        return min(self.press_m, self.dmax_m)


@dataclass(frozen=True)
class SilParams:
    """Everything one SIL run needs."""

    mount: ArmMount
    tool: Tool
    paper: Paper
    controller: Controller
    walker: Walker = Walker()
    joints: Joints = field(default_factory=Joints)
    dt: float = 1.0e-3          # s, plant time step AND controller tick
    settle_s: float = 0.5       # s, activation hold before the stream starts
    contact_force_eps: float = 0.01   # N, the "carrying real force" threshold
