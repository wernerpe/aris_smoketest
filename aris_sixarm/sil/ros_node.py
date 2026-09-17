"""ROS 2 wrapper: the SIL plant presented as a franka robot-state broadcaster.

WHAT THIS CLOSES.  `simulator.run` drives the plant from
`setpoints.PathwayWalker`, an OPEN-LOOP replica of the executor's geometry.
This module drives the SAME plant from the REAL `rtff_pathway_exec.py` over
DDS, so the executor's
closed-loop half -- the lag latch, air-trim, `_servo_press`, the float and
over-press sentinels, the landing rate limiter -- runs against a pencil whose
physical tip is `--tip-error` metres from where the robot believes it is
(contract §4).  Everything the walker deliberately omits is exercised here.

    python3 -m aris_sixarm.sil.ros_node \\
        --csv aris_sixarm/sil/examples/line10cm_arm31.csv \\
        --rig proposed --arm 31 --tool lateral --tip-error -0.010 \\
        --press 0.010 --out /out/t8/run --no-plot

INTERFACE (contract §3), all in the ARM's own base frame, EE = the NOMINAL tip:

  publishes at `--state-rate` (100 Hz):
    /franka_robot_state_broadcaster/robot_state   franka_msgs/FrankaRobotState
        o_t_ee               the NOMINAL tip pose (`SilPlant.tip_pose_base`),
                             never the physical one -- that is the point.
        o_t_ee_d / o_t_ee_c  the law's FILTERED setpoint (what the controller
                             is actually regulating to).
        o_f_ext_hat_k        the force the ENVIRONMENT exerts on the tool,
                             expressed in base (contract §3 sign note).
        measured_joint_state q (rad), dq (rad/s), effort = the LAW's torque
                             (Nm, no gravity, as the controller writes it)
        time                 SIMULATED seconds; robot_mode = MOVE.
    /franka_robot_state_broadcaster/current_pose                  PoseStamped
    /franka_robot_state_broadcaster/external_wrench_in_base_frame WrenchStamped
        the fallback pair, same content, always published.
  subscribes:
    /cartesian_impedance/equilibrium_pose  PoseStamped -> the law's setpoint.
        Held between messages exactly as the controller's RT buffer holds it,
        and low-passed by the law's own `filter_alpha`.
    /cartesian_impedance/joint_reference   JointState (contract §2) -> matched
        BY INDEX, 7 positions; a reference is handed to the law on the tick
        after it arrives and then AGES, so `joint_reference_timeout_s` decides
        whether it replaces `q_null` or the activation posture stands.

TIME.  The executor is WALL-CLOCK paced (50 Hz `time.sleep`) and calls
robot_state older than 0.5 s stale, 2 s a reflex.  So the node is wall-clock
paced too: one 100 Hz timer advances the 1 kHz physics toward
`(wall - t0) * --rt-factor`, at most `--physics-substep` ticks per slot, and
THEN publishes -- the state stream keeps its 100 Hz even when the physics
falls behind, and the achieved real-time factor is logged and recorded in
`summary.json`.  If the plant cannot keep up on a slower machine, drop
`--rt-factor` (the executor's own pacing is unchanged; only the simulated
clock slows, so its speeds and rates become that factor faster relative to the
plant) or raise `--physics-substep`.

PHASES.  The node does NOT know the executor's phases -- there is no phase on
the wire.  `draw` ticks for `trace.metrics` are derived from the commanded
stream: a tick is `draw` when the commanded tip sits no more than
`--draw-tol` (2 mm) ABOVE the paper plane, i.e. within press + 2 mm of the
plane or below it.  Everything else is labelled `travel`.  Read every
draw-phase metric with that definition in mind: a touchdown that descends
through the plane is counted from the moment it crosses, not from the
executor's own stroke start.

NOT MODELLED, and it flatters the executor: the ~2 N pose-dependent wrench
bias and the drifting baseline of briefing §10, and joint friction
(`params.Joints`, zero by default), whose tip-space consequence -- 5-20 mm of
free-air lag -- is exactly what the deployed press floors of 9-10 mm exist to
cross.  In this plant a 9 mm floor is 9 mm of real steel-on-paper press.

`rclpy` is imported INSIDE `main()` and the node class is built by a factory,
so `import aris_sixarm.sil.ros_node` stays ROS-free on the host (contract §0).
"""
from __future__ import annotations

import argparse
import json
import signal
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import __main__ as cli
from .impedance import CartesianImpedanceLaw
from .params import SilParams
from .plant import build_simulator
from .setpoints import PHASES, Setpoint, read_csv_v2
from .simulator import _tick, seed_configuration
from .trace import Recorder, metrics

EQ_POSE_TOPIC = "/cartesian_impedance/equilibrium_pose"
JOINT_REF_TOPIC = "/cartesian_impedance/joint_reference"
STATE_NS = "/franka_robot_state_broadcaster"
JOINT_NAMES = tuple(f"fr3_joint{i}" for i in range(1, 8))

_PHASE_DRAW = PHASES.index("draw")
_PHASE_TRAVEL = PHASES.index("travel")


# ---------------------------------------------------------------------------
# the physics half: no ROS in this class
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RobotState:
    """One 100 Hz sample of what the robot would report. SI, base frame."""

    t_s: float                  # SIMULATED seconds since activation
    tip_pos: np.ndarray         # (3,) m, the NOMINAL tip = o_t_ee
    tip_quat: np.ndarray        # (4,) xyzw
    des_pos: np.ndarray         # (3,) m, the law's filtered setpoint
    des_quat: np.ndarray        # (4,) xyzw
    force_base: np.ndarray      # (3,) N, environment -> tool, base frame
    q: np.ndarray               # (7,) rad
    dq: np.ndarray              # (7,) rad/s
    tau: np.ndarray             # (7,) Nm, the LAW's output (no gravity)


class SilBackend:
    """Plant + law + recorder, stepped one 1 kHz tick at a time.

    The tick is `simulator._tick` with the setpoint coming off the wire
    instead of out of a `SetpointSource`: read the state, run the law, add
    what the hardware layer adds (gravity, joint friction), record, step.
    """

    def __init__(self, params: SilParams, q0: np.ndarray,
                 draw_tol_m: float = 0.002):
        self.params = params
        self.draw_tol_m = float(draw_tol_m)
        self.q0 = np.asarray(q0, float)
        self.plant = build_simulator(params)
        self.plant.set_state(self.q0)
        self.law = CartesianImpedanceLaw(params.controller)
        p_tip, quat_tip = self.plant.tip_pose_base()
        self.law.activate(self.q0, p_tip, quat_tip)
        self.rec = Recorder(params.paper.z_world, params.tool.tip_error_m)

        self._sp_pos: np.ndarray | None = None
        self._sp_quat: np.ndarray | None = None
        self._pending_qref: np.ndarray | None = None
        self.last_tau = np.zeros(7)
        self.n_pose_msgs = 0
        self.n_qref_msgs = 0
        self.n_ticks = 0
        self._hand_index = self.plant.hand_body.index()
        self._contact_port = \
            self.plant.plant.get_contact_results_output_port()

    # -- the wire, in ----------------------------------------------------
    def submit_pose(self, p_xyz, quat_xyzw) -> None:
        """Latch an equilibrium pose (m, xyzw, base frame). Held until the
        next one arrives -- the controller's RT buffer, exactly."""
        self._sp_pos = np.asarray(p_xyz, float)
        self._sp_quat = np.asarray(quat_xyzw, float)
        self.n_pose_msgs += 1

    def submit_joint_reference(self, positions) -> None:
        """Latch a contract §2 joint reference (7, rad, matched BY INDEX).

        It is handed to the law ONCE, on the next tick; after that the law
        ages it, so `joint_reference_timeout_s` -- not this node -- decides
        when the activation posture takes the nullspace back.
        """
        q = np.asarray(positions, float).reshape(-1)
        if q.size != 7:
            raise ValueError(
                f"joint reference needs 7 positions, got {q.size}")
        self._pending_qref = q
        self.n_qref_msgs += 1

    # -- state, out ------------------------------------------------------
    @property
    def t(self) -> float:
        """Simulated seconds since activation."""
        return self.plant.t

    def state(self) -> RobotState:
        """-> the `RobotState` to publish this cycle."""
        p_tip, quat_tip = self.plant.tip_pose_base()
        pos_d, quat_d = self.law.filtered_setpoint
        return RobotState(self.plant.t, p_tip, quat_tip, pos_d, quat_d,
                          self.contact_force_base(), self.plant.q,
                          self.plant.dq, self.last_tau.copy())

    def contact_force_base(self) -> np.ndarray:
        """(3,) N, the force the ENVIRONMENT exerts on the tool, base frame.

        Summed over the point pairs of `SilPlant`'s only proximity pair (the
        physical tip sphere against the sheet); `contact_force()` is the force
        on body B, so the sign flips when the hand is body A.  Contract §3:
        the sim reports the environment's force on the tool and the executor's
        `--force-sign` absorbs the robot's own convention.
        """
        results = self._contact_port.Eval(self.plant.plant_context)
        f_world = np.zeros(3)
        for i in range(results.num_point_pair_contacts()):
            info = results.point_pair_contact_info(i)
            sign = 1.0 if info.bodyB_index() == self._hand_index else -1.0
            f_world += sign * np.asarray(info.contact_force(), float)
        return self.params.mount.T_world_base[:3, :3].T @ f_world

    # -- one 1 kHz iteration ---------------------------------------------
    def tick(self) -> None:
        """One 1 kHz iteration, off the wire.

        The iteration itself is `simulator._tick` -- the same read / law /
        hardware-layer / record / step sequence the open-loop runs use, called
        rather than copied so the two paths can never drift apart.  All this
        method does is turn the latched message into the `Setpoint` that
        function expects:

          * no pose received yet -> the law's own filtered target, `hold=True`.
            That IS the empty RT buffer, and it is exactly what `run` does for
            its settle phase, so the trace labels those ticks `settle` too.
          * a joint reference is attached ONCE, on the tick after it arrives;
            after that the law ages it and `joint_reference_timeout_s` decides.
        """
        if self._sp_pos is None:
            pos_d, quat_d = self.law.filtered_setpoint
            _tick(self.plant, self.law, self.params, self.rec,
                  Setpoint(self.plant.t, pos_d, quat_d, None, "settle",
                           self.commanded_depth(pos_d), 0.0, -1), hold=True)
        else:
            depth = self.commanded_depth(self._sp_pos)
            sp = Setpoint(self.plant.t, self._sp_pos, self._sp_quat,
                          self._pending_qref,
                          "draw" if depth >= -self.draw_tol_m else "travel",
                          float(depth), 0.0, -1)
            self._pending_qref = None
            _tick(self.plant, self.law, self.params, self.rec, sp, hold=False)
        # `_last_tau` is the law's own `last_tau_` member (see its class
        # docstring): the torque the controller last commanded, which is what
        # `measured_joint_state.effort` carries on the real broadcaster.
        self.last_tau = np.asarray(self.law._last_tau, float)
        self.n_ticks += 1

    def commanded_depth(self, p_base) -> float:
        """m, how far a base-frame command sits BELOW the paper plane.

        Positive = pressed into the sheet, negative = above it.  The paper is
        a horizontal table in every rig this package models, so the normal is
        world +z (`Paper.normal_world`).
        """
        T_wb = self.params.mount.T_world_base
        z_world = float(T_wb[2, :3] @ np.asarray(p_base, float) + T_wb[2, 3])
        return self.params.paper.z_world - z_world


# ---------------------------------------------------------------------------
# the recorded setpoint stream
# ---------------------------------------------------------------------------
class SetpointRecorder:
    """Every equilibrium pose the node received, as a replayable npz.

    Written in the `setpoints.SetpointLog` npz format (`t_s`, `pos`, `quat`,
    `q`, `press_cmd_m`, `phase_idx`, `phase_names`), so a stream captured off
    the REAL executor replays through the same plant on the host with
    `python -m aris_sixarm.sil --setpoint-log <file>` -- no ROS required.
    """

    def __init__(self) -> None:
        self.t: list[float] = []
        self.pos: list[np.ndarray] = []
        self.quat: list[np.ndarray] = []
        self.q: list[np.ndarray | None] = []
        self.press: list[float] = []
        self.phase: list[int] = []

    def add(self, t_s: float, pos, quat, press_cmd_m: float,
            phase_idx: int) -> None:
        self.t.append(float(t_s))
        self.pos.append(np.asarray(pos, float))
        self.quat.append(np.asarray(quat, float))
        self.q.append(None)
        self.press.append(float(press_cmd_m))
        self.phase.append(int(phase_idx))

    def attach_joint_reference(self, q_ref) -> None:
        """Fill in the newest row's `q`.

        The executor publishes the pose and then the JointState of the SAME
        interpolation fraction, back to back in `_pub`, so the reference that
        arrives next belongs to the pose that arrived last.  A row whose
        reference never arrives (v1 segment, flag off) keeps `None` and the
        whole `q` block is then omitted from the npz.
        """
        if self.q and self.q[-1] is None:
            self.q[-1] = np.asarray(q_ref, float)

    def save(self, path) -> Path:
        """Write the npz. -> the path written."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        kw = {}
        if self.q and all(x is not None for x in self.q):
            kw["q"] = np.asarray(self.q, float)
        np.savez_compressed(
            path, t_s=np.asarray(self.t, float),
            pos=np.asarray(self.pos, float).reshape(-1, 3),
            quat=np.asarray(self.quat, float).reshape(-1, 4),
            press_cmd_m=np.asarray(self.press, float),
            phase_idx=np.asarray(self.phase, int),
            phase_names=np.array(PHASES, dtype=object).astype(str), **kw)
        return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """The SIL CLI's parser (`__main__.build_parser`) plus the node's flags.

    Arm, tool, paper, contact and controller flags are the CLI's own, so a
    node run and a `python -m aris_sixarm.sil` run of the same geometry are
    the same `SilParams` by construction.  `--compare` and `--setpoint-log`
    make no sense here and are refused.
    """
    ap = cli.build_parser()
    ap.prog = "python -m aris_sixarm.sil.ros_node"
    ap.description = __doc__.splitlines()[0]
    g = ap.add_argument_group("ros node")
    g.add_argument("--start-q", type=float, nargs=7, metavar="Q",
                   help="rad, the configuration the sim holds at start")
    g.add_argument("--start-pose", type=float, nargs=7,
                   metavar=("X", "Y", "Z", "QX", "QY", "QZ", "QW"),
                   help="m + xyzw in the base frame; IK'd for the start "
                        "configuration when --start-q is absent")
    g.add_argument("--start-hover", type=float, default=0.030,
                   help="m above the CSV's first draw point to start at when "
                        "neither --start-q nor --start-pose is given")
    g.add_argument("--rt-factor", type=float, default=1.0,
                   help="simulated seconds per wall second")
    g.add_argument("--physics-substep", type=int, default=40,
                   help="max 1 kHz physics ticks per state-publish slot; the "
                        "cap is what keeps the 100 Hz state stream alive when "
                        "the plant falls behind")
    g.add_argument("--state-rate", type=float, default=100.0,
                   help="Hz, robot-state publish rate (contract §3)")
    g.add_argument("--base-frame", default="fr3_link0",
                   help="header.frame_id of every published pose/wrench")
    g.add_argument("--draw-tol", type=float, default=0.002,
                   help="m: a tick is a DRAW tick when the commanded tip is "
                        "no higher than this above the paper plane")
    g.add_argument("--max-wall-s", type=float, default=0.0,
                   help="s, stop and write the outputs after this much wall "
                        "time (0 = run until SIGINT/SIGTERM)")
    g.add_argument("--ready-file", default="",
                   help="path touched once the plant is up and the first "
                        "state has been published (default <out>/ready)")
    return ap


def _csv_has_joint_columns(path) -> bool:
    """Does this pathway CSV carry contract §1's q1..q7 on every row?

    It decides the SIL controller's nullspace default exactly as the CLI's
    does (`__main__.params_for`): a file with q runs contract §2's reference
    mode, a file without it runs the deployed stage-0 latched posture.  The
    executor makes the same call from the same columns, so the two agree
    without either asking the other.
    """
    if not path:
        return False
    return all(row.q is not None for row in read_csv_v2(path))


def start_configuration(params: SilParams, args) -> tuple[np.ndarray, str]:
    """-> (q0 (7,) rad, how it was obtained).

    `--start-q` wins; then the IK of `--start-pose`; otherwise the IK of a
    pose `--start-hover` metres back along the pen axis from the CSV's first
    `draw` row.  The executor's approach ramp starts from the pose it MEASURES,
    so the sim must start somewhere it would plausibly have been parked --
    hovering over the first stroke, which is where `pen_switch.sh down` and
    the preposition leave it.
    """
    if args.start_q:
        return np.asarray(args.start_q, float), "start-q"
    if args.start_pose:
        p = np.asarray(args.start_pose[:3], float)
        quat = np.asarray(args.start_pose[3:], float)
        return seed_configuration(
            params, Setpoint(0.0, p, quat, None, "travel", 0.0)).q0, \
            "ik(start-pose)"
    if not args.csv:
        raise SystemExit(
            "one of --start-q / --start-pose / --csv is required to place the "
            "arm at start")
    rows = read_csv_v2(args.csv)
    first = next((r for r in rows if r.is_draw), None)
    if first is None:
        raise SystemExit(f"{args.csv}: no rows with kind == 'draw'")
    hover = first.p - float(args.start_hover) * first.pen_axis
    act = seed_configuration(
        params, Setpoint(0.0, hover, first.quat, None, "travel",
                         -float(args.start_hover)))
    return act.q0, f"ik(csv hover {1e3 * args.start_hover:.0f} mm)"


# ---------------------------------------------------------------------------
# the ROS half: built lazily so this module imports without rclpy
# ---------------------------------------------------------------------------
def _node_class():
    """-> the `Node` subclass. Built here so `rclpy` stays out of import."""
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

    from franka_msgs.msg import FrankaRobotState
    from geometry_msgs.msg import PoseStamped, WrenchStamped
    from sensor_msgs.msg import JointState

    class SilRobotStateNode(Node):
        """Thin adapter: messages and the clock in, the plant's state out.

        No physics lives here.  The node owns the wall-clock pacing, the
        message translation and the recording of what it received; every
        newton and every millimetre comes from `SilBackend`.
        """

        def __init__(self, backend: SilBackend, args):
            super().__init__("sil_robot_state")
            self.backend = backend
            self.args = args
            self.frame = args.base_frame
            self.setpoints = SetpointRecorder()
            self.stop = False
            self._warned_names = False
            self._t0_wall = time.monotonic()
            self._last_report = 0.0
            self._ready = False
            self.rt_factor_achieved = float("nan")

            qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST, depth=10)
            self.pub_state = self.create_publisher(
                FrankaRobotState, STATE_NS + "/robot_state", qos)
            self.pub_pose = self.create_publisher(
                PoseStamped, STATE_NS + "/current_pose", qos)
            self.pub_wrench = self.create_publisher(
                WrenchStamped,
                STATE_NS + "/external_wrench_in_base_frame", qos)
            # The executor publishes both inputs RELIABLE / KEEP_LAST / 10.
            self.create_subscription(PoseStamped, EQ_POSE_TOPIC,
                                     self._on_eq_pose, 10)
            self.create_subscription(JointState, JOINT_REF_TOPIC,
                                     self._on_joint_ref, 10)
            self.create_timer(1.0 / float(args.state_rate), self._on_cycle)
            self.get_logger().info(
                "SIL up: %s arm %d (%s), tool %s, tip_error %+.1f mm, paper "
                "z_world %.3f m, rt-factor %.2f, state %.0f Hz"
                % (backend.params.mount.rig, backend.params.mount.arm_id,
                   backend.params.mount.mount, backend.params.tool.name,
                   1e3 * backend.params.tool.tip_error_m,
                   backend.params.paper.z_world, args.rt_factor,
                   args.state_rate))

        # -- in ----------------------------------------------------------
        def _on_eq_pose(self, msg) -> None:
            p, o = msg.pose.position, msg.pose.orientation
            pos = np.array([p.x, p.y, p.z])
            quat = np.array([o.x, o.y, o.z, o.w])
            self.backend.submit_pose(pos, quat)
            depth = self.backend.commanded_depth(pos)
            self.setpoints.add(
                self.backend.t, pos, quat, depth,
                _PHASE_DRAW if depth >= -self.args.draw_tol else _PHASE_TRAVEL)

        def _on_joint_ref(self, msg) -> None:
            if len(msg.position) != 7:
                self.get_logger().warn(
                    "joint reference has %d positions, want 7 -- ignored"
                    % len(msg.position))
                return
            if msg.name and list(msg.name) != list(JOINT_NAMES) \
                    and not self._warned_names:
                self._warned_names = True
                self.get_logger().warn(
                    "joint reference names %s != %s -- matching BY INDEX "
                    "(contract §2)" % (list(msg.name), list(JOINT_NAMES)))
            self.backend.submit_joint_reference(msg.position)
            self.setpoints.attach_joint_reference(msg.position)

        # -- the cycle: physics to wall clock, then publish ---------------
        def _on_cycle(self) -> None:
            wall = time.monotonic() - self._t0_wall
            target = wall * float(self.args.rt_factor)
            n = 0
            while (self.backend.t + 0.5 * self.backend.params.dt < target
                   and n < self.args.physics_substep):
                self.backend.tick()
                n += 1
            self._publish(self.backend.state())
            self.rt_factor_achieved = (self.backend.t / wall if wall > 0
                                       else float("nan"))
            if not self._ready:
                self._ready = True
                Path(self.args.ready_file).write_text("ready\n")
                self.get_logger().info("READY: first robot_state published")
            if wall - self._last_report >= 5.0:
                self._last_report = wall
                self.get_logger().info(
                    "sim %.2f s / wall %.2f s (rt %.2f), %d poses, %d joint "
                    "refs, %d ticks"
                    % (self.backend.t, wall, self.rt_factor_achieved,
                       self.backend.n_pose_msgs, self.backend.n_qref_msgs,
                       self.backend.n_ticks))
            if self.args.max_wall_s and wall >= self.args.max_wall_s:
                self.get_logger().warn(
                    "--max-wall-s %.0f s reached -- stopping"
                    % self.args.max_wall_s)
                self.stop = True

        # -- out ----------------------------------------------------------
        def _stamped(self, msg):
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.frame
            return msg

        def _pose_msg(self, pos, quat):
            m = self._stamped(PoseStamped())
            m.pose.position.x = float(pos[0])
            m.pose.position.y = float(pos[1])
            m.pose.position.z = float(pos[2])
            m.pose.orientation.x = float(quat[0])
            m.pose.orientation.y = float(quat[1])
            m.pose.orientation.z = float(quat[2])
            m.pose.orientation.w = float(quat[3])
            return m

        def _wrench_msg(self, force):
            m = self._stamped(WrenchStamped())
            m.wrench.force.x = float(force[0])
            m.wrench.force.y = float(force[1])
            m.wrench.force.z = float(force[2])
            return m

        def _publish(self, st: RobotState) -> None:
            pose = self._pose_msg(st.tip_pos, st.tip_quat)
            desired = self._pose_msg(st.des_pos, st.des_quat)
            wrench = self._wrench_msg(st.force_base)

            msg = self._stamped(FrankaRobotState())
            msg.o_t_ee = pose
            msg.o_t_ee_d = desired
            msg.o_t_ee_c = desired
            msg.o_f_ext_hat_k = wrench
            js = msg.measured_joint_state
            js.header.stamp = msg.header.stamp
            js.name = list(JOINT_NAMES)
            js.position = [float(v) for v in st.q]
            js.velocity = [float(v) for v in st.dq]
            js.effort = [float(v) for v in st.tau]
            msg.robot_mode = FrankaRobotState.ROBOT_MODE_MOVE
            msg.control_command_success_rate = 1.0
            msg.time = float(st.t_s)

            self.pub_state.publish(msg)
            self.pub_pose.publish(pose)
            self.pub_wrench.publish(wrench)

    return SilRobotStateNode


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def _write_outputs(node, params: SilParams, out: Path, args,
                   start_source: str) -> dict:
    """Freeze the trace, write trace/setpoints/summary. -> the metrics dict."""
    backend = node.backend
    trace = backend.rec.freeze({
        "label": "ros_node",
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
        "activation": {"source": start_source,
                       "q0": backend.q0.tolist()},
        "point_stiffness_n_per_m": params.paper.point_stiffness,
        "dissipation_s_per_m": params.paper.dissipation,
        "joint_viscous": params.joints.viscous.tolist(),
        "joint_coulomb": params.joints.coulomb.tolist(),
        "nullspace_mode": params.controller.nullspace_mode,
        "qref_rate_limit_rad_s": params.controller.qref_rate_limit_rad_s,
        "qref_divergence_rad": params.controller.qref_divergence_rad,
        "qref_ticks": backend.law.qref_ticks,
        "qref_divergence_ticks": backend.law.qref_divergence_ticks,
        "paper_size_m": list(params.paper.size),
        "paper_center_xy_m": list(params.paper.center_xy),
        "draw_tol_m": args.draw_tol,
        "phase_rule": ("draw = commanded tip <= paper plane + draw_tol "
                       "(the node cannot see the executor's phases)"),
    })
    out.mkdir(parents=True, exist_ok=True)
    trace.save(out / "trace.npz")
    node.setpoints.save(out / "setpoints_recv.npz")
    m = metrics(trace)
    summary = {
        "metrics": m,
        "meta": trace.meta,
        "ros": {
            "eq_pose_msgs": backend.n_pose_msgs,
            "joint_ref_msgs": backend.n_qref_msgs,
            "physics_ticks": backend.n_ticks,
            "sim_duration_s": backend.t,
            "rt_factor_requested": args.rt_factor,
            "rt_factor_achieved": node.rt_factor_achieved,
            "state_rate_hz": args.state_rate,
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not args.no_plot:
        try:
            from .trace import plot
            plot(trace, out / "plot.png")
        except Exception as exc:                     # matplotlib is optional
            print(f"[sil-node] plot skipped: {exc!r}")
    return m


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.compare:
        raise SystemExit("--compare is a batch flag; the node runs one arm")
    if args.setpoint_log:
        raise SystemExit("--setpoint-log replays offline; the node takes its "
                         "stream from the equilibrium-pose topic")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if not args.ready_file:
        args.ready_file = str(out / "ready")
    Path(args.ready_file).unlink(missing_ok=True)

    has_q = _csv_has_joint_columns(args.csv)
    params = cli.params_for(args, args.tip_error, has_q)
    q0, start_source = start_configuration(params, args)
    print(f"[sil-node] {params.mount.rig} arm {params.mount.arm_id} "
          f"({params.mount.mount}), tool {params.tool.name}, tip_error "
          f"{1e3 * params.tool.tip_error_m:+.1f} mm, start {start_source}: "
          f"q0 = {np.round(q0, 4).tolist()}", flush=True)
    backend = SilBackend(params, q0, draw_tol_m=args.draw_tol)

    import rclpy
    rclpy.init(args=None)
    node = _node_class()(backend, args)
    signal.signal(signal.SIGTERM, lambda *_: setattr(node, "stop", True))
    try:
        while rclpy.ok() and not node.stop:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    except Exception as exc:                          # ExternalShutdown etc.
        print(f"[sil-node] spin ended: {exc!r}", flush=True)

    m = _write_outputs(node, params, out, args, start_source)
    print(f"[sil-node] {backend.n_ticks} ticks, {backend.t:.2f} s simulated, "
          f"rt {node.rt_factor_achieved:.2f}, {backend.n_pose_msgs} poses, "
          f"{backend.n_qref_msgs} joint refs", flush=True)
    print("[sil-node] SUMMARY " + json.dumps(
        {k: m[k] for k in ("n_draw_ticks", "touch_frac", "f_mean_n",
                           "f_mean_contact_n", "f_max_n", "max_penetration_m",
                           "mean_actual_tip_height_m")}), flush=True)
    print(f"[sil-node] written to {out}", flush=True)
    try:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
