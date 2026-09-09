"""The unit of execution: one arm's joint trajectory, with a clock.

Everything upstream of `aris_sixarm.execute` speaks in geometry and in a
CONDUCTED timeline; everything downstream speaks in joint angles at wall-clock
instants.  This module is the boundary object, and it is deliberately dumb: an
array of times, an array of configurations, and the checks that say whether a
robot could follow it.

WHAT A `JointTrajectory` IS AND IS NOT.  It is a *sampled path with a clock*,
not a controller input.  The samples come off the conductor at its own step
(`1 / (fps * substeps)`, 1/48 s on every programme shipped so far) and the
straight line between two of them is what a robot commanded at those instants
would actually fly.  That is the same chord `writing.densify` bounds by
`MAX_DQ_FRAME` and the same chord `csail_drawing_demo.py` measures at 0.762 mm
of tip error, so the trajectory's fidelity is a property of the SAMPLING and
resampling it does not improve it — `resample` interpolates the chords, it does
not re-solve the IK.

THE CLOCK IS THE FLEET'S, AND THAT IS THE SAFETY ARGUMENT.  `scene_check`
certifies inter-arm clearance frame by frame on ONE clock shared by all six
arms.  Re-timing one arm — pausing it, speeding it, smoothing it — moves it
against the other five at instants nobody certified.  So the only legal
re-timing is a scalar applied to every arm at once, which is what `Governor`
is, and every per-arm helper here is written to make an accidental per-arm
re-time hard rather than convenient.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..frames import FR3_MAX, FR3_MIN, QD_MAX

# --- the envelopes execution is held to --------------------------------------
# Position margin off the hard joint stop.  `frames.py` records that the repo
# keeps >= 0.15 rad everywhere and paces at safety 0.8 to stay inside
# libfranka's position-dependent braking curve; this is that number, restated
# where the executor can check it.
JOINT_MARGIN = 0.15          # rad
# Fraction of QD_MAX an executed trajectory may demand.  `pacing.SAFETY` is the
# planner's own 0.8; execution re-checks against the same number so a
# trajectory that was paced correctly cannot fail here, and one that was paced
# by something else cannot pass.
SPEED_SAFETY = 0.8
# Joint acceleration ceiling used only as a SCREEN.  10 rad/s^2 flat is not a
# datasheet figure: it is `~/git/fr3drivers`'s own command gate
# (`--fr3_max_joint_acceleration=10.0`, whose flag help calls ddq "the single
# most dangerous field in this message"), i.e. the box the one driver on this
# machine with a safety architecture would REFUSE a command outside of.  The
# planner does not model acceleration at all (`pacing.py`'s "TODO (v2)"), so
# this is the executor noticing that a path the planner called feasible asks
# for a step that driver would reject — not a certificate that a smaller one
# is fine.
QDD_SCREEN = np.full(7, 10.0)                                      # rad/s^2


@dataclass
class JointTrajectory:
    """One arm, `t` seconds against `q` radians.  `t[0]` is 0 by construction.

    `dt_nominal` is the uniform step the samples were taken at, or 0.0 when the
    samples are not uniform.  It is carried rather than re-derived because the
    thing downstream needs to know is what the CONDUCTOR's step was, and a
    resampled copy has to be able to say it is no longer that.
    """
    arm_id: int
    t: np.ndarray                  # (N,) s, strictly increasing, t[0] == 0
    q: np.ndarray                  # (N, 7) rad
    dt_nominal: float = 0.0
    source: str = ""               # provenance, printed by every report

    def __post_init__(self):
        self.t = np.ascontiguousarray(np.asarray(self.t, float).reshape(-1))
        self.q = np.ascontiguousarray(np.asarray(self.q, float).reshape(len(self.t), 7))

    # -- shape -------------------------------------------------------------
    def __len__(self):
        return len(self.t)

    @property
    def duration_s(self):
        return float(self.t[-1]) if len(self.t) else 0.0

    # -- reading -----------------------------------------------------------
    def sample(self, t):
        """q at time `t` (scalar or array), LINEARLY, clamped at both ends.

        Linear and not spline on purpose: the chord between two conductor
        samples is what `scene_check` certified, and a spline would fly a
        different curve — smoother, prettier, and not the one that was proved
        clear of the neighbours.
        """
        ts = np.atleast_1d(np.asarray(t, float))
        out = np.empty((len(ts), 7))
        for j in range(7):
            out[:, j] = np.interp(ts, self.t, self.q[:, j])
        return out[0] if np.ndim(t) == 0 else out

    def resample(self, hz):
        """A copy sampled uniformly at `hz`.  Chords unchanged (see `sample`).

        The end point is always included: a controller that stops one step
        short of the last commanded configuration leaves the pen down.
        """
        n = max(int(round(self.duration_s * float(hz))) + 1, 2)
        t = np.linspace(0.0, self.duration_s, n)
        return JointTrajectory(self.arm_id, t, self.sample(t),
                               dt_nominal=float(t[1] - t[0]),
                               source=f"{self.source} @resample({hz:g} Hz)")

    def slice_time(self, t0, t1):
        """The part of this trajectory in [t0, t1], re-zeroed at t0."""
        keep = (self.t >= t0 - 1e-12) & (self.t <= t1 + 1e-12)
        t = np.concatenate([[t0], self.t[keep], [t1]])
        t = np.unique(np.clip(t, t0, t1))
        return JointTrajectory(self.arm_id, t - t0, self.sample(t),
                               dt_nominal=self.dt_nominal,
                               source=f"{self.source}[{t0:.3f}:{t1:.3f}]")

    # -- what a robot would be asked for ----------------------------------
    def joint_speed(self):
        """(N-1, 7) |dq/dt| across every step. Empty for a 1-sample path."""
        if len(self) < 2:
            return np.zeros((0, 7))
        return np.abs(np.diff(self.q, axis=0)) / np.diff(self.t)[:, None]

    def joint_accel(self):
        """(N-2, 7) |d2q/dt2|, one-sided across the sample steps."""
        v = self.joint_speed()
        if len(v) < 2:
            return np.zeros((0, 7))
        dt = 0.5 * (np.diff(self.t)[:-1] + np.diff(self.t)[1:])
        return np.abs(np.diff(v, axis=0)) / dt[:, None]

    def check(self, speed_safety=SPEED_SAFETY, joint_margin=JOINT_MARGIN,
              qdd_screen=QDD_SCREEN):
        """-> [] when a robot could fly this.  Otherwise, what is wrong.

        Every entry is one line naming the joint, the sample and both numbers,
        because the only useful form of "this trajectory is unsafe" is one a
        person can go and look at.
        """
        bad = []
        if len(self) < 1:
            return [f"arm {self.arm_id}: empty trajectory"]
        if not np.isfinite(self.q).all():
            bad.append(f"arm {self.arm_id}: q has non-finite entries")
        if not np.isfinite(self.t).all():
            bad.append(f"arm {self.arm_id}: t has non-finite entries")
        if len(self) > 1 and not (np.diff(self.t) > 0).all():
            k = int(np.argmin(np.diff(self.t)))
            bad.append(f"arm {self.arm_id}: t is not strictly increasing at "
                       f"sample {k} (dt = {np.diff(self.t)[k]:.6g} s)")
        if abs(float(self.t[0])) > 1e-9:
            bad.append(f"arm {self.arm_id}: t[0] = {self.t[0]:.6g}, want 0")

        lo, hi = FR3_MIN + joint_margin, FR3_MAX - joint_margin
        under = self.q < lo
        over = self.q > hi
        for j in range(7):
            for name, mask, lim in (("under", under[:, j], lo[j]),
                                    ("over", over[:, j], hi[j])):
                if mask.any():
                    k = int(np.argmax(mask))
                    bad.append(
                        f"arm {self.arm_id}: joint {j + 1} {name} the "
                        f"{joint_margin:g} rad margin at sample {k}: "
                        f"{self.q[k, j]:.4f} vs {lim:.4f}")
        v = self.joint_speed()
        if len(v):
            cap = speed_safety * QD_MAX
            hit = v > cap[None, :]
            for j in np.flatnonzero(hit.any(axis=0)):
                k = int(np.argmax(hit[:, j]))
                bad.append(f"arm {self.arm_id}: joint {j + 1} speed "
                           f"{v[k, j]:.3f} > {cap[j]:.3f} rad/s "
                           f"({speed_safety:g} x QD_MAX) at sample {k}")
        a = self.joint_accel()
        if len(a):
            hit = a > np.asarray(qdd_screen)[None, :]
            for j in np.flatnonzero(hit.any(axis=0)):
                k = int(np.argmax(hit[:, j]))
                bad.append(f"arm {self.arm_id}: joint {j + 1} accel SCREEN "
                           f"{a[k, j]:.1f} > {qdd_screen[j]:.1f} rad/s^2 at "
                           f"sample {k} (a screen, not a certificate)")
        return bad

    def report(self):
        """Two lines: what it is, and how close to the envelope it runs."""
        v = self.joint_speed()
        peak = (v / QD_MAX[None, :]).max() if len(v) else 0.0
        j = int(np.unravel_index(np.argmax(v / QD_MAX[None, :]), v.shape)[1]) \
            if len(v) else 0
        return [f"arm {self.arm_id}: {len(self)} samples, {self.duration_s:.3f} s"
                + (f", dt {1000 * self.dt_nominal:.2f} ms" if self.dt_nominal
                   else ", non-uniform")
                + (f"  [{self.source}]" if self.source else ""),
                f"  peak {100 * peak:.1f} % of QD_MAX (joint {j + 1}); "
                f"budget is {100 * SPEED_SAFETY:.0f} %"]


@dataclass
class Governor:
    """ONE clock rate for the WHOLE fleet.  The only legal re-timing.

    `scene_check` certified the six arms against each other frame by frame on a
    shared clock.  Scaling that clock by a scalar moves every arm the same way
    along the same geometry, so every certified pairwise distance is attained at
    a different WALL time and at no different CONFIGURATION — the proof carries.
    Slowing one arm does not: the pair (a, b) then meets at configurations the
    checker never evaluated.  Hence one rate, applied to the program clock, and
    no per-arm entry point anywhere in this package.

    `rate` ramps toward `target` at `accel` per second so a hold and a resume
    are not step changes in commanded velocity.  A hold is therefore not
    instantaneous: `stop_distance_s` is how much program time still passes.
    """
    max_rate: float = 1.0
    accel: float = 2.0        # rate units per second (0 -> 1 in 0.5 s)
    rate: float = 0.0
    target: float = 0.0
    t_program: float = 0.0
    history: list = field(default_factory=list, repr=False)

    def resume(self, rate=None):
        self.target = float(self.max_rate if rate is None else
                            min(rate, self.max_rate))
        return self.target

    def hold(self):
        self.target = 0.0

    @property
    def holding(self):
        return self.target == 0.0 and self.rate == 0.0

    @property
    def stop_distance_s(self):
        """Program seconds still to elapse if `hold()` were called now."""
        return 0.5 * self.rate * self.rate / self.accel if self.accel > 0 else 0.0

    def step(self, wall_dt):
        """Advance by `wall_dt` seconds of REAL time -> the new program time."""
        wall_dt = float(wall_dt)
        r0 = self.rate
        dr = self.accel * wall_dt
        self.rate = (min(r0 + dr, self.target) if self.target > r0
                     else max(r0 - dr, self.target))
        self.t_program += 0.5 * (r0 + self.rate) * wall_dt     # trapezoid
        return self.t_program

    def seek(self, t_program):
        """Jump the clock (a barrier release, a rung that starts mid-phase)."""
        self.t_program = float(t_program)
        return self.t_program
