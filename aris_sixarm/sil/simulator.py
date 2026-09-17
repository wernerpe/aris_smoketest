"""Wiring: plant + law + setpoint stream + activation + recording.

A MANUAL 1 kHz LOOP, not a Diagram of LeafSystems, and deliberately.  The
deployed controller is a single `update()` with its own state, called once per
millisecond by ros2_control between reading the state interfaces and writing
the effort ones; a manual loop is that sequence written out -- read state,
compute tau, add what the hardware layer adds, step the plant -- so the
replica can be read against the C++ line by line and the law stays testable
with no Drake in the room.  A LeafSystem would buy an update-order guarantee
this loop already has explicitly, at the cost of hiding it.

Gravity: `libfranka` compensates it inside the robot, so the law never sees it
and this loop adds `-tau_g` on top of the law's output (briefing §3, §7.2).
Joint friction, if `params.joints` carries any, is added in the same place, for
the same reason: it belongs to the machine, not to the controller.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .. import fleet, frames, ik
from ..metrics import sigma_min, tip_jacobian
from .impedance import CartesianImpedanceLaw
from .params import SilParams
from .plant import SilPlant, build_simulator
from .rotations import matrix_from_quat, slerp
from .setpoints import PHASES, Setpoint, SetpointSource
from .trace import Recorder, Trace


@dataclass(frozen=True)
class Activation:
    """What the controller latched when it went active."""

    q0: np.ndarray               # (7,) rad
    source: str                  # "q_ref" | "ik"
    margin_rad: float
    sigma_min: float


def seed_configuration(params: SilParams, setpoint: Setpoint) -> Activation:
    """The joint configuration the run starts from.

    The first setpoint's `q_ref` when the CSV carries joint columns (contract
    §1) -- the planner's own branch, which is the whole point of shipping q
    with the pose.  Otherwise the analytic IK of `aris_sixarm.ik`, scanned over
    `ik.Q7_GRID` and every FR3-valid branch, best joint margin wins.  The
    solver is FK-verified inside `ik.solve`, so a workspace-boundary clamp
    cannot come back as a solution.
    """
    lat, ext = params.tool.offset_tcp[0], params.tool.offset_tcp[2]
    if setpoint.q_ref is not None:
        q0 = np.asarray(setpoint.q_ref, float)
        return Activation(q0, "q_ref", frames.joint_margin(q0),
                          sigma_min(tip_jacobian(q0, pen_ext=ext, pen_lat=lat)))
    # tip pose -> hand-TCP pose: the tool offset is a pure translation in the
    # TCP frame, so the TCP sits `R @ offset` back from the tip.
    R = matrix_from_quat(setpoint.quat_xyzw)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(setpoint.p_xyz, float) - R @ params.tool.offset_tcp
    seed = _q_seed(params)
    best = None
    for q7 in ik.Q7_GRID:
        for q in ik.solve(T, q7, seed):
            m = frames.joint_margin(q)
            if best is None or m > best[0]:
                best = (m, q)
    if best is None:
        raise RuntimeError(
            "no FR3 IK solution for the first setpoint "
            f"{np.round(setpoint.p_xyz, 4)} in the base frame of arm "
            f"{params.mount.arm_id}; the CSV and the rig disagree")
    m, q0 = best
    return Activation(q0, "ik", m,
                      sigma_min(tip_jacobian(q0, pen_ext=ext, pen_lat=lat)))


def _q_seed(params: SilParams) -> np.ndarray:
    """The IK seed: the arm's own ready pose where a rig defines one."""
    if params.mount.rig != "synthetic":
        fl, _ = fleet.rig(params.mount.rig)
        return np.asarray(fl[params.mount.arm_id].q_seed, float)
    return (frames.Q_READY_INV if params.mount.mount == "inv"
            else frames.Q_READY_FLOOR)


def run(params: SilParams, source: SetpointSource, label: str = "",
        plant: SilPlant | None = None) -> Trace:
    """Simulate `source` against `params`. -> the recorded `Trace`.

    Sequence:
      1. build the plant (or reuse one passed in),
      2. place the arm at the activation configuration and let it stand,
      3. `on_activate`: latch the tip pose as the desired and the posture as
         the nullspace target, zero the torque history,
      4. hold for `params.settle_s` with NO setpoint -- the RT buffer is empty
         and the controller holds what it latched, which is exactly what the
         real one does between `pen_switch.sh down` and the executor's first
         publish,
      5. GLIDE from there to the stream's first setpoint at `travel_speed`,
         because the executor never publishes a step (briefing §7.1: the node
         glides, the controller's filter is not a rate limiter) and a 30 mm
         jump from the activation pose to the first hover point would be a
         30 mm step,
      6. stream the setpoints, one per tick.
    """
    plant = build_simulator(params) if plant is None else plant
    stream = iter(source)
    first = next(stream)
    act = seed_configuration(params, first)
    plant.set_state(act.q0)

    law = CartesianImpedanceLaw(params.controller)
    p_tip, quat_tip = plant.tip_pose_base()
    law.activate(act.q0, p_tip, quat_tip)

    rec = Recorder(params.paper.z_world, params.tool.tip_error_m)
    n_settle = int(round(params.settle_s / params.dt))
    settle = (Setpoint(0.0, p_tip, quat_tip, None, "settle", 0.0)
              for _ in range(n_settle))
    approach = _approach(p_tip, quat_tip, first, params)
    for sp in itertools.chain(settle, approach, [first], stream):
        _tick(plant, law, params, rec, sp, hold=sp.phase == "settle")

    meta = {
        "label": label,
        "rig": params.mount.rig, "arm": params.mount.arm_id,
        "mount": params.mount.mount, "tool": params.tool.name,
        "tip_error_m": params.tool.tip_error_m,
        "paper_z_world": params.paper.z_world,
        "press_m": params.walker.press_applied_m,
        "dmax_m": params.walker.dmax_m,
        "k_cartesian": params.controller.k_cartesian.tolist(),
        "d_cartesian": params.controller.d_cartesian.tolist(),
        "max_torques": params.controller.max_torques.tolist(),
        "controller_yaml": params.controller.source,
        "activation": {"source": act.source, "q0": act.q0.tolist(),
                       "margin_rad": act.margin_rad,
                       "sigma_min": act.sigma_min},
        "point_stiffness_n_per_m": params.paper.point_stiffness,
        "dissipation_s_per_m": params.paper.dissipation,
        "joint_viscous": params.joints.viscous.tolist(),
        "joint_coulomb": params.joints.coulomb.tolist(),
        "nullspace_mode": params.controller.nullspace_mode,
        "qref_rate_limit_rad_s": params.controller.qref_rate_limit_rad_s,
        "qref_divergence_rad": params.controller.qref_divergence_rad,
        "qref_ticks": law.qref_ticks,
        "qref_divergence_ticks": law.qref_divergence_ticks,
        "paper_size_m": list(params.paper.size),
        "paper_center_xy_m": list(params.paper.center_xy),
    }
    return rec.freeze(meta)


def _approach(p_from: np.ndarray, quat_from: np.ndarray, first: Setpoint,
              params: SilParams):
    """Yield a `travel`-speed glide from the activation pose to `first`.

    The executor's own first move: `travel` at `travel_speed` to the hover
    point above the first stroke.  Without it the controller's first input is
    a step of whatever the activation posture happens to leave -- 30 mm when
    the CSV's joint columns put the arm on the paper and the walker's first
    setpoint is a hover -- and the pose-error clamp then dumps 0.05 m of
    spring into the arm at once.
    """
    dist = float(np.linalg.norm(np.asarray(first.p_xyz, float) - p_from))
    n = int(round(dist / params.walker.travel_speed / params.dt))
    for k in range(n):
        u = (k + 1) / n
        yield Setpoint(0.0, p_from + u * (np.asarray(first.p_xyz, float) - p_from),
                       slerp(quat_from, first.quat_xyzw, u), first.q_ref,
                       "travel", first.press_cmd_m, first.intensity,
                       first.stroke_idx)


def _tick(plant: SilPlant, law: CartesianImpedanceLaw, params: SilParams,
          rec: Recorder, sp: Setpoint, hold: bool) -> None:
    """One 1 kHz iteration: read, control, record, step."""
    q, dq = plant.q, plant.dq
    p_tip, quat_tip = plant.tip_pose_base()
    J = plant.tip_jacobian_base()
    tau, dbg = law.step(q, dq, p_tip, quat_tip, J, plant.coriolis(),
                        params.dt, None if hold else sp)
    tau_applied = (tau + plant.gravity_compensation()
                   + params.joints.torque(dq))

    contact = plant.contact()
    T_wb = params.mount.T_world_base
    sp_world = T_wb[:3, :3] @ np.asarray(sp.p_xyz, float) + T_wb[:3, 3]
    rec.add(t=plant.t, q=q, dq=dq, tau=tau, q_null=law.q_nullspace,
            q_null_raw=law.q_reference_raw,
            sp_pos_base=np.asarray(sp.p_xyz, float),
            sp_quat=np.asarray(sp.quat_xyzw, float), sp_pos_world=sp_world,
            filt_pos_base=dbg.pos_d, filt_quat=dbg.quat_d,
            tip_pos_base=p_tip, tip_quat=quat_tip,
            tip_nom_world=plant.tip_world(False),
            tip_act_world=plant.tip_world(True),
            f_normal=contact.normal_force_n, penetration=contact.penetration_m,
            in_contact=abs(contact.normal_force_n) > params.contact_force_eps,
            phase_idx=PHASES.index(sp.phase), stroke_idx=sp.stroke_idx,
            press_cmd_m=sp.press_cmd_m,
            track_err_m=float(np.linalg.norm(
                np.asarray(sp.p_xyz, float) - p_tip)))
    plant.step(tau_applied)
