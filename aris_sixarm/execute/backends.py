"""Where a `FleetProgram` goes.  One backend that draws, one that records,
one that would move metal and does not.

THE PROTOCOL IS SEVEN METHODS AND THE ORDER THEY ARE CALLED IN IS THE SAFETY
STORY:

    connect(arms)            open whatever talks to the arms
    read_state()             -> {arm: q}; the ONLY input from the world
    goto(arm, q, duration_s) a slow, supervised, point-to-point move.  Used
                             ONCE per run, to reach the start barrier's pose,
                             and never inside a programme
    start_stream(arms, hz)   begin the real-time loop
    send(t_s, q_by_arm)      one commanded fleet configuration
    barrier(barrier, measured)  the fleet is stationary; verify and hold
    stop(reason)             end the stream, leave the arms held and safe

`send` is the only call in the hot loop and it takes the WHOLE fleet at one
program time, so a backend physically cannot advance one arm without the
others — the constraint `trajectory.Governor` argues for, made structural.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np


class Backend:
    """The interface.  Subclass it; the base class refuses to move anything."""

    name = "base"
    #: True only for a backend that commands real hardware.  Every gate in
    #: `runner.py` that could hurt somebody keys off this and not off the class.
    is_hardware = False

    def connect(self, arms):
        raise NotImplementedError

    def read_state(self):
        """-> {arm: (7,) q}.  A backend with no sensing returns its last command."""
        raise NotImplementedError

    def goto(self, arm, q, duration_s):
        raise NotImplementedError

    def start_stream(self, arms, hz):
        pass

    def send(self, t_s, q_by_arm):
        raise NotImplementedError

    def barrier(self, barrier, measured):
        """Called once the clock has stopped at `barrier`. -> [] or problems."""
        return barrier.mismatch(measured)

    def stop(self, reason=""):
        pass

    # a backend may refuse a programme before anything moves
    def preflight(self, program):
        return []


# ---------------------------------------------------------------------------
@dataclass
class RecordingBackend(Backend):
    """Swallows a programme and remembers it.  The tests' backend.

    It also models the one thing a real arm does that an animation does not:
    it starts wherever `initial` says, so a test can drive the "the arm is not
    where the programme thinks it is" path without a robot.
    """
    initial: dict = field(default_factory=dict)
    name: str = "recording"
    is_hardware: bool = False
    sent: list = field(default_factory=list)         # [(t, {arm: q})]
    barriers: list = field(default_factory=list)     # [(barrier, problems)]
    gotos: list = field(default_factory=list)        # [(arm, q, duration)]
    stopped: str | None = None
    _state: dict = field(default_factory=dict)

    def connect(self, arms):
        self._state = {int(a): np.asarray(self.initial.get(int(a),
                                                           np.zeros(7)), float)
                       for a in arms}

    def read_state(self):
        return {a: q.copy() for a, q in self._state.items()}

    def goto(self, arm, q, duration_s):
        self.gotos.append((int(arm), np.asarray(q, float).copy(),
                           float(duration_s)))
        self._state[int(arm)] = np.asarray(q, float).copy()

    def send(self, t_s, q_by_arm):
        self.sent.append((float(t_s), {int(a): np.asarray(q, float).copy()
                                       for a, q in q_by_arm.items()}))
        self._state.update({int(a): np.asarray(q, float).copy()
                            for a, q in q_by_arm.items()})

    def barrier(self, barrier, measured):
        problems = barrier.mismatch(measured)
        self.barriers.append((barrier, problems))
        return problems

    def stop(self, reason=""):
        self.stopped = reason


# ---------------------------------------------------------------------------
class MeshcatDryRun(Backend):
    """Plays a programme into the meshcat scene `viz/` already builds.

    THE POINT OF THIS BACKEND IS THE REHEARSAL, not the picture.  The web
    viewer (`web/viewer`, `program_schema.export_bundle`) is a scrubber: it
    shows the whole timeline and lets a person hunt for the worst frame.  This
    one runs the SAME numbers through the SAME executor, at wall-clock speed,
    with the barriers stopping it and a person acknowledging the pen swaps —
    so the thing that gets rehearsed before hardware is the executor and the
    procedure, not just the geometry.

    It reads its arm bases from `fleet.FLEET`, so it inherits `ARIS_RIG` /
    `ARIS_TOOL` exactly like `export_viewer_bundle.py` does, and shows a
    different room if those are wrong.
    """

    name = "meshcat"
    is_hardware = False

    def __init__(self, url=None, pen_lat=None, show_paper=True):
        self.url, self._pen_lat, self._show_paper = url, pen_lat, show_paper
        self.vis = None
        self._links = self._joints = None
        self._bases = {}
        self._state = {}
        self.n_sent = 0

    def connect(self, arms):
        import meshcat
        import meshcat.geometry as g
        import meshcat.transformations as tf

        from .. import frames
        from ..fleet import FLEET, SHEET
        from ..viz import robot_model

        self.vis = (meshcat.Visualizer(zmq_url=self.url) if self.url
                    else meshcat.Visualizer())
        self.vis["/Background"].set_property("top_color", [0.95, 0.95, 0.97])
        self.vis["/Background"].set_property("bottom_color", [0.85, 0.85, 0.90])
        if self._show_paper:
            self.vis["paper"].set_object(
                g.Box([SHEET[0], SHEET[1], 0.004]),
                g.MeshLambertMaterial(color=0xFAFAF5))
            self.vis["paper"].set_transform(
                tf.translation_matrix([SHEET[0] / 2, SHEET[1] / 2, -0.002]))
        self._links, self._joints = robot_model.load_model()
        lat = frames.lat_of(self._pen_lat)
        for a in arms:
            spec = FLEET[int(a)]
            T = spec.T_world_base()
            self._bases[int(a)] = T
            robot_model.add_robot(
                self.vis, f"robots/arm{a}", self._links, self._joints,
                spec.q_seed, T,
                pen_color=int(0x10000 * int(255 * spec.color[0])
                              + 0x100 * int(255 * spec.color[1])
                              + int(255 * spec.color[2])),
                pen_len=frames.ext_of(spec.pen_ext), pen_lat=lat)
            self._state[int(a)] = np.asarray(spec.q_seed, float)
        print(f"meshcat dry run at {self.vis.url()}")
        return self.vis.url()

    def read_state(self):
        return {a: q.copy() for a, q in self._state.items()}

    def goto(self, arm, q, duration_s):
        # A dry run has nothing to protect, so the supervised move is a jump —
        # and it is LOGGED as one, because on hardware this is the call a human
        # watches with a hand on the stop.
        self._pose(int(arm), np.asarray(q, float))
        print(f"  [dry run] goto arm {arm} over {duration_s:.1f} s")

    def send(self, t_s, q_by_arm):
        for a, q in q_by_arm.items():
            self._pose(int(a), np.asarray(q, float))
        self.n_sent += 1

    def barrier(self, barrier, measured):
        print(f"  [dry run] barrier {barrier.name} at t = {barrier.t_s:.3f} s"
              + ("  (ACK)" if barrier.requires_ack else ""))
        return barrier.mismatch(measured)

    def stop(self, reason=""):
        print(f"  [dry run] stop: {reason or 'end of programme'} "
              f"({self.n_sent} frames sent)")

    def _pose(self, arm, q):
        from ..viz import robot_model
        self._state[arm] = q.copy()
        T = self._bases[arm]
        poses = robot_model.link_poses(self._joints, q)
        for name in self._links:
            if name in poses:
                self.vis[f"robots/arm{arm}/{name}"].set_transform(T @ poses[name])


# ---------------------------------------------------------------------------
#: The installation's own arm -> IP map, copied from
#: `Aris_Kindt/aris_orchestrator/arm_registry.py` (the declared source of
#: truth) so that a preflight can say "you have no IP for arm 71" without this
#: package importing the installation's code.  Arm 71 has none yet.
#: NOT 172.16.0.2 — that is the Franka factory default and the operator's own
#: notes call every occurrence of it wrong.
INSTALLATION_IPS = {13: "192.168.50.11", 17: "192.168.50.16",
                    31: "192.168.50.12", 71: None,
                    2: "192.168.50.13", 97: "192.168.50.15"}


class Fr3BundleBackend(Backend):
    """SKELETON.  The offline half is real; every call that moves metal raises.

    ================  NOTHING IN THIS CLASS COMMANDS A ROBOT  =================
    `connect`, `read_state`, `goto`, `send` and `stop` all raise.  Only
    `preflight` and `export` run, and neither leaves the filesystem.
    ===========================================================================

    WHY THIS SHAPE AND NOT A libfranka CALLBACK.  There is already a driver on
    this machine with a real safety architecture — `~/git/fr3drivers`,
    `franka_driver_v5` — and it does not take a callback.  It takes a
    PIECEWISE-BEZIER BUNDLE (`tools/fr3_bundle.py`: `control_points`
    `(n_seg, degree+1, ndof)`, `t_start`, `t_end`, physical seconds) which
    `tools/fr3_sender.py` evaluates in closed form and publishes as
    `fr3::fr3_command {utime, mode, q[7], dq[7], ddq[7]}` at 400–500 Hz into a
    1 kHz control loop.  Its gate refuses non-finite commands, velocity over
    `QD_MAX`, acceleration over 10 rad/s², a position step over 0.15 rad and a
    stale command over 10 ms, and it latches a tracking fault at 20 mrad.

    So the adapter this project needs is a FILE FORMAT CONVERSION, and that is
    the half implemented here: `export` turns a `JointTrajectory` into a bundle
    npz that `fr3_sender.py --dry-run` can read.  Streaming is the driver's
    job, not this package's, and the process that runs the sender is the
    process that must hold the fleet clock.

    **AND THE CONVERSION IS NOT FREE — READ `export`'s DOCSTRING.**  A degree-1
    bundle reproduces the conducted timeline exactly (the chords ARE what the
    checker graded) and has a discontinuous velocity at every knot, which
    `position_velocity_accel` mode cannot use.  Smoothing it is a planning
    change, not a formatting one: it must stay inside the certified tube.  That
    is the same gap `pacing.py` names as its own "TODO (v2)".

    THE OTHER STACK, FOR THE RECORD.  What actually drew on paper in this
    installation is not this driver: it is ROS 2 MoveIt (Cartesian waypoints,
    `computeCartesianPath` + TOTG) for transit and a 50 Hz Cartesian-impedance
    equilibrium-pose stream for the strokes, launched over SSH onto the
    operator box.  That stack takes a CARTESIAN pathway CSV, not a joint
    trajectory, and adapting to it would throw away the redundancy resolution
    this whole package exists to compute.  Named here so the choice is a
    choice.
    """

    name = "fr3_bundle"
    is_hardware = True

    def __init__(self, ips=None, control_hz=1000.0, send_hz=500.0,
                 speed_scale=1.0):
        self.ips = dict(INSTALLATION_IPS if ips is None else ips)
        self.control_hz = float(control_hz)
        self.send_hz = float(send_hz)
        #: fr3drivers ships a conservative 0.37 hardware scale.  A scale is a
        #: reparametrisation: the control points are untouched, so the image of
        #: the curve — and every clearance certificate over it — is unchanged.
        #: The same argument `trajectory.Governor` makes, from the other side.
        self.speed_scale = float(speed_scale)

    def _stub(self, what):
        raise NotImplementedError(
            f"Fr3BundleBackend.{what} is a STUB. Live control is deliberately "
            "unimplemented in this package: the driver at ~/git/fr3drivers "
            "owns it. See docs/HARDWARE_LADDER.md; do not wire this up before "
            "rung 1 has produced a MEASURED pen-tip transform.")

    # -- the half that is real ---------------------------------------------
    def preflight(self, program):
        """Everything checkable without a robot.  Real, not a stub."""
        bad = []
        missing = [a for a in program.arms if not self.ips.get(a)]
        if missing:
            bad.append(f"no IP configured for arm(s) {missing} "
                       "(arm 71 has never had one)")
        if program.stride > 1:
            bad.append(
                f"the programme is decimated (stride {program.stride}); the "
                "driver runs at 1 kHz and would fly the chords between "
                "certified frames. Re-conduct with --substeps 1.")
        bad += program.check()
        for a in program.arms:
            tr = program.track(a).resample(self.control_hz)
            bad += [f"at {self.control_hz:g} Hz: {m}" for m in tr.check()]
        return bad

    @staticmethod
    def export(traj, path=None, speed_scale=1.0, source=""):
        """A `JointTrajectory` -> an `fr3_bundle` v1 npz (degree 1).

        DEGREE 1 IS THE HONEST ENCODING AND IT IS NOT YET FLYABLE.  One Bezier
        segment per sample interval with control points `[q_k, q_{k+1}]` is
        EXACTLY the straight joint-space line between two conducted frames —
        the same chord `scene_check` graded, reproduced bit for bit rather than
        approximated.  Its velocity is piecewise constant and therefore
        discontinuous at every knot, and its acceleration is zero on every
        segment and undefined at every knot, so `position_velocity_accel` mode
        would be handed a `ddq` that is a lie.  The driver's own gate does not
        catch that (a zero ddq is inside any box); the ARM does, as tracking
        error.

        What ships here is therefore a FAITHFUL bundle for `--dry-run`,
        inspection and the sender's own preflight — not a flyable one.  Making
        it flyable means re-parameterising with bounded acceleration while
        staying inside the certified tube, which is a planning change.

        Returns the dict that was (or would be) saved.
        """
        q, t = np.asarray(traj.q, float), np.asarray(traj.t, float)
        if len(q) < 2:
            raise ValueError("a bundle needs at least two samples")
        cp = np.stack([q[:-1], q[1:]], axis=1)          # (n_seg, 2, 7)
        d = dict(format="fr3_bundle", version=1,
                 control_points=cp, t_start=t[:-1] * 1.0, t_end=t[1:] * 1.0,
                 source=source or f"aris_sixarm.execute {traj.source}",
                 speed_scale=float(speed_scale),
                 meta_json=json.dumps(dict(
                     arm=int(traj.arm_id), degree=1,
                     samples=int(len(q)), duration_s=float(traj.duration_s),
                     warning="degree 1: velocity is discontinuous at every "
                             "knot; NOT flyable in position_velocity_accel "
                             "mode. See Fr3BundleBackend.export.")))
        if speed_scale != 1.0:
            d["t_start"] = d["t_start"] / speed_scale
            d["t_end"] = d["t_end"] / speed_scale
        if path is not None:
            np.savez(path, **d)
        return d

    # -- the half that raises ----------------------------------------------
    def connect(self, arms):
        self._stub("connect")

    def read_state(self):
        self._stub("read_state")

    def goto(self, arm, q, duration_s):
        self._stub("goto")

    def start_stream(self, arms, hz):
        self._stub("start_stream")

    def send(self, t_s, q_by_arm):
        self._stub("send")

    def stop(self, reason=""):
        self._stub("stop")


#: The old name, kept pointing at the new class for one release: nothing in the
#: repo imports it, and the driver this package would talk to is not raw
#: libfranka.
LibfrankaBackend = Fr3BundleBackend


# ---------------------------------------------------------------------------
class RateKeeper:
    """Sleeps just enough to hold a loop at `hz`. -> the wall dt it observed.

    Reports the shortfall rather than hiding it: a dry run that cannot keep up
    with 24 Hz is telling you something about the executor, and a run that
    silently drifts is telling you nothing.
    """

    def __init__(self, hz, realtime=True):
        self.dt = 1.0 / float(hz)
        self.realtime = bool(realtime)
        self.t_last = None
        self.late_n = 0
        self.late_worst = 0.0
        self.n = 0

    def tick(self):
        self.n += 1
        if not self.realtime:
            return self.dt
        now = time.perf_counter()
        if self.t_last is None:
            self.t_last = now
            return self.dt
        target = self.t_last + self.dt
        slack = target - now
        if slack > 0:
            time.sleep(slack)
        else:
            self.late_n += 1
            self.late_worst = max(self.late_worst, -slack)
        now2 = time.perf_counter()
        dt, self.t_last = now2 - self.t_last, now2
        return dt

    def report(self):
        return (f"{self.n} ticks at {1 / self.dt:g} Hz, {self.late_n} late "
                f"({100 * self.late_n / max(self.n, 1):.1f} %), worst overrun "
                f"{1000 * self.late_worst:.1f} ms")
