"""Setpoint streams: what the executor would publish on
`/cartesian_impedance/equilibrium_pose` (+ the §2 joint reference).

Two sources, one interface.  Both yield `Setpoint`s at a FIXED rate (1 kHz by
default, the controller's own), because that is what the controller sees: the
RT buffer holds the last message until the next arrives, so a source that
publishes slower simply repeats itself.

  PathwayWalker  an OPEN-LOOP replica of `rtff_pathway_exec.py`'s phase walk
                 over a pathway CSV v2 (contract §1).
  SetpointLog    replays a recorded stream, so a run can be reproduced or a
                 stream captured from the real executor can be pushed through
                 the same plant.

Frame: the arm's own base frame, metres, quaternion (x, y, z, w); the pen axis
is EE +Z (contract §1).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from .params import Walker
from .rotations import quat_align, quat_normalize, slerp

PHASES = ("settle", "travel", "touchdown", "draw", "lift")


@dataclass(frozen=True)
class Setpoint:
    """One equilibrium pose. `p_xyz` m and `quat_xyzw` in the arm base frame.

    `q_ref` is contract §2's joint reference (7, rad) or None when the source
    has no joint columns for this segment -- in which case nothing is
    published and the controller keeps the posture latched at activation.
    `press_cmd_m` is the signed offset of this commanded pose from the CSV row
    along the pen axis: negative above the paper (hover), positive into it
    (press).  `intensity` is the row's tone, 0..1, carried because it is the
    only pressure channel the artwork has (contract §1, briefing §12.3).
    """

    t_s: float
    p_xyz: np.ndarray
    quat_xyzw: np.ndarray
    q_ref: np.ndarray | None = None
    phase: str = "draw"
    press_cmd_m: float = 0.0
    intensity: float = 0.0
    stroke_idx: int = -1


@dataclass
class Row:
    """One pathway CSV v2 row, parsed."""

    stroke_idx: int
    wp_idx: int
    kind: str
    p: np.ndarray                # (3,) m, base frame
    quat: np.ndarray             # (4,) xyzw
    intensity: float
    q: np.ndarray | None = None  # (7,) rad, contract §1's optional columns

    @property
    def pen_axis(self) -> np.ndarray:
        """(3,) unit, the pen axis in the base frame = R(quat) @ [0,0,1]."""
        x, y, z, w = self.quat
        return np.array([2 * (x * z + y * w), 2 * (y * z - x * w),
                         1 - 2 * (x * x + y * y)])

    @property
    def is_draw(self) -> bool:
        """The executor tests `kind == "draw"` and nothing else."""
        return self.kind == "draw"


def read_csv_v2(path) -> list[Row]:
    """Parse a pathway CSV v2 (contract §1). -> rows in file order."""
    rows: list[Row] = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            qcols = [r.get(f"q{i}", "") for i in range(1, 8)]
            has_q = all(c not in (None, "") for c in qcols)
            rows.append(Row(
                stroke_idx=int(r["stroke_idx"]), wp_idx=int(r["wp_idx"]),
                kind=r["kind"].strip(),
                p=np.array([float(r["x_m"]), float(r["y_m"]), float(r["z_m"])]),
                quat=quat_normalize([float(r["qx"]), float(r["qy"]),
                                     float(r["qz"]), float(r["qw"])]),
                intensity=float(r.get("intensity", 0.0) or 0.0),
                q=np.array([float(c) for c in qcols]) if has_q else None))
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def runs_of(rows: Sequence[Row]) -> list[tuple[bool, list[Row]]]:
    """-> [(is_draw, rows)] for each maximal run of consecutive rows of the
    same class, IN FILE ORDER.

    The executor walks the CSV in file order and decides only one thing per
    row: `spd = draw_speed if kind == "draw" else travel_speed`.  So does the
    walker, and the run is the unit that carries that decision.
    """
    out: list[tuple[bool, list[Row]]] = []
    for row in rows:
        if out and out[-1][0] == row.is_draw:
            out[-1][1].append(row)
        else:
            out.append((row.is_draw, [row]))
    if not any(is_draw for is_draw, _ in out):
        raise ValueError("no rows with kind == 'draw'")
    return out


# ---------------------------------------------------------------------------
# legs: a piece of motion with a duration and a pose at each fraction of it
# ---------------------------------------------------------------------------
@dataclass
class _Leg:
    """One piece of motion: a knot list walked by arc length over a duration.

    A knot is `(position, quaternion, press, q_ref | None, intensity)`.  A
    straight leg is two knots and a polyline leg is one per CSV row; the same
    interpolation serves both, so there is one code path and not two.
    """

    phase: str
    duration_s: float
    stroke_idx: int
    knots: list                       # [(p, quat, press, q|None, tone)]
    s: np.ndarray                     # cumulative arc length of `knots`

    def at(self, u: float):
        """-> (p, quat, press, q_ref, intensity) at fraction u in [0, 1]."""
        dist = float(np.clip(u, 0.0, 1.0)) * self.s[-1]
        i = int(np.clip(np.searchsorted(self.s, dist, side="right") - 1,
                        0, len(self.knots) - 2))
        span = self.s[i + 1] - self.s[i]
        f = 0.0 if span <= 0 else (dist - self.s[i]) / span
        a, b = self.knots[i], self.knots[i + 1]
        q = None if (a[3] is None or b[3] is None) else a[3] + f * (b[3] - a[3])
        return (a[0] + f * (b[0] - a[0]), slerp(a[1], b[1], f),
                a[2] + f * (b[2] - a[2]), q, a[4] + f * (b[4] - a[4]))


def _leg(phase: str, speed: float, stroke_idx: int, knots: list,
         floor_s: float = 0.0) -> _Leg:
    """Build a leg from knots, paced at `speed` along its own arc length.

    `floor_s` is a minimum duration.  A leg of zero length is dropped by the
    caller, which is right for a connect leg that has nowhere to go and wrong
    for a one-row DRAW run: that row is a mark on the paper and must be
    commanded for at least a tick.
    """
    if len(knots) == 1:
        knots = [knots[0], knots[0]]
    pts = np.array([k[0] for k in knots], float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    return _Leg(phase, max(float(s[-1]) / speed, floor_s), stroke_idx, knots, s)


class SetpointSource:
    """Fixed-rate iterable of `Setpoint`.

    Subclasses implement `__iter__` (one setpoint per `dt`, in order) and
    `duration_s`.  `simulator.run` needs nothing else, so a caller with its own
    stream -- a captured executor log, a hand-built hold -- can be one too.
    """

    def __init__(self, rate_hz: float = 1000.0):
        self.rate_hz = float(rate_hz)
        self.dt = 1.0 / self.rate_hz

    def __iter__(self) -> Iterator[Setpoint]:
        raise NotImplementedError

    @property
    def duration_s(self) -> float:
        raise NotImplementedError

    @property
    def has_joint_columns(self) -> bool:
        """Does this stream publish a contract §2 joint reference at all?"""
        return False


class PathwayWalker(SetpointSource):
    """Open-loop replica of the executor's walk over a pathway CSV v2.

    THE RULES, IN ONE PLACE.  The file is walked IN FILE ORDER and split into
    RUNS of consecutive rows of the same class (`runs_of`).  Nothing is
    reordered and no row is skipped; the only thing the row's `kind` decides is
    its speed and whether the press is applied, which is the executor's own
    single test (`spd = draw_speed if is_draw else travel_speed`).

      PEN-UP RUN (`travel`, `lift`, `lift_start`, `lift_end`, anything not
        `draw`) -> phase `travel`.  A CONNECT leg from wherever the pen is to
        the run's first row at `travel_speed`, then the run's rows walked as a
        Cartesian polyline at `travel_speed`.  The poses are commanded AS
        WRITTEN -- the generator already lifted the pen, so no hover is added
        and no press applied.  `q_ref` is interpolated along the polyline, and
        absent for any leg an endpoint of which carries no joint columns
        (contract §2: no q on a waypoint, nothing published for that segment).

      DRAW RUN -> the landing sequence, then phase `draw`:
        `travel`     CONNECT at `travel_speed` to the APPROACH POINT, `hover_m`
                     above the run's first row along that row's own pen axis.
                     Synthesised whether or not the file carries pen-up rows,
                     because `hover_m` (RTFF_HOVER) is the executor's parameter
                     and not the generator's: a file whose lift rows sit 50 mm
                     up still lands from 30 mm.  Dropped when zero-length.
        `touchdown`  the executor's two phases along the pen axis:
                     `touchdown_fast` to `touchdown_slowzone` above the plane,
                     `touchdown_speed` through the zone and the press.
        `draw`       the run's rows at `draw_speed`, each pushed
                     `press_applied_m` along ITS OWN pen axis into the paper.
                     The spring turns that overshoot into contact force; the
                     press is capped at `dmax_m`, the paper's own protection
                     (briefing §10, §12.4).
        `lift`       back up the pen axis to `hover_m` above the last row, at
                     `touchdown_fast`; `lift_max_m` floors it (briefing §9:
                     lift must exceed press or the pen drags).

    WHAT THIS DELIBERATELY DOES NOT MODEL, because none of it is open loop:
    the lag latch, the memory/table gate, air-trim, the landing rate limiter,
    the force servo `_servo_press`, the tone -> force / tone -> depth laws,
    tilt compensation, the ladder gate.  All of those need the robot-state
    feedback of contract §3 and belong to `ros_node.py`.  What is left is the
    geometry the executor commands when every measurement agrees with the plan.

    AND ONE THING THE WALKER CANNOT INVENT: a TRANSIT.  If the plan
    reconfigures the arm between two strokes and the file carries no rows for
    the motion that realises it, the walker flies a straight Cartesian line
    while the joint reference steps by whole radians.  That is a hole in the
    FILE, not in the walk; the controller-side guard is
    `Controller.qref_rate_limit_rad_s`.
    """

    def __init__(self, csv_path, walker: Walker = Walker(),
                 rate_hz: float = 1000.0):
        super().__init__(rate_hz)
        self.path = Path(csv_path)
        self.walker = walker
        self.rows = read_csv_v2(self.path)
        self.runs = runs_of(self.rows)
        self._legs_cache = self._build_legs()

    @property
    def duration_s(self) -> float:
        return float(sum(leg.duration_s for leg in self._legs_cache))

    @property
    def has_joint_columns(self) -> bool:
        """True when every DRAW row carries q1..q7 (contract §1)."""
        return all(row.q is not None for row in self.rows if row.is_draw)

    def _build_legs(self) -> list[_Leg]:
        w = self.walker
        press = w.press_applied_m
        legs: list[_Leg] = []
        cursor = None                       # (p, quat, q) currently commanded

        def emit(leg: _Leg) -> None:
            if leg.duration_s > 0.0:
                legs.append(leg)

        for is_draw, rows in self.runs:
            first, last = rows[0], rows[-1]
            knots = self._knots(rows, press if is_draw else 0.0)
            if not is_draw:
                cursor = cursor or knots[0]
                emit(_leg("travel", w.travel_speed, first.stroke_idx,
                          [cursor, self._align(knots[0], cursor)]))
                emit(_leg("travel", w.travel_speed, first.stroke_idx, knots))
                cursor = knots[-1]
                continue

            n0, n1 = first.pen_axis, last.pen_axis
            zone = w.touchdown_slowzone
            approach = (first.p - w.hover_m * n0, first.quat, -w.hover_m,
                        first.q, first.intensity)
            slow_top = (first.p - zone * n0, first.quat, -zone, first.q,
                        first.intensity)
            cursor = cursor or approach
            emit(_leg("travel", w.travel_speed, first.stroke_idx,
                      [cursor, self._align(approach, cursor)]))
            emit(_leg("touchdown", w.touchdown_fast, first.stroke_idx,
                      [approach, slow_top]))
            emit(_leg("touchdown", w.touchdown_speed, first.stroke_idx,
                      [slow_top, knots[0]]))
            emit(_leg("draw", w.draw_speed, first.stroke_idx, knots,
                      floor_s=self.dt))
            hover1 = (last.p - w.hover_m * n1, knots[-1][1], -w.hover_m,
                      last.q, last.intensity)
            emit(_leg("lift", w.touchdown_fast, last.stroke_idx,
                      [knots[-1], hover1]))
            cursor = hover1
        if not legs:
            raise ValueError(f"{self.path}: nothing to walk")
        return legs

    @staticmethod
    def _knots(rows: list[Row], press: float) -> list:
        """The run's rows as knots, each offset `press` along its own pen axis
        and its quaternion made sign-continuous with the one before it."""
        knots, prev = [], rows[0].quat
        for row in rows:
            prev = quat_align(row.quat, prev)
            knots.append((row.p + press * row.pen_axis, prev, press, row.q,
                          row.intensity))
        return knots

    @staticmethod
    def _align(knot, ref) -> tuple:
        """`knot` with its quaternion on `ref`'s hemisphere."""
        return (knot[0], quat_align(knot[1], ref[1]), *knot[2:])

    def __iter__(self) -> Iterator[Setpoint]:
        t_idx = 0
        for leg in self._legs_cache:
            n = max(1, int(round(leg.duration_s / self.dt)))
            for k in range(n):
                p, quat, press, q, tone = leg.at(k / n)
                yield Setpoint(t_idx * self.dt, p, quat, q, leg.phase,
                               float(press), float(tone), leg.stroke_idx)
                t_idx += 1
        last = self._legs_cache[-1]
        p, quat, press, q, tone = last.at(1.0)
        yield Setpoint(t_idx * self.dt, p, quat, q, last.phase, float(press),
                       float(tone), last.stroke_idx)


class SetpointLog(SetpointSource):
    """Replay a recorded (t, pose, q_ref) stream, holding the last sample.

    Accepts an `.npz` written by `trace.Trace.save_setpoint_log` (keys `t_s`,
    `pos` (N,3), `quat` (N,4 xyzw), optional `q` (N,7), optional `phase_idx`
    and `press_cmd_m`) or a CSV with the header
    `t_s,x,y,z,qx,qy,qz,qw[,q1..q7]`.
    """

    def __init__(self, path, rate_hz: float = 1000.0):
        super().__init__(rate_hz)
        self.path = Path(path)
        if self.path.suffix == ".npz":
            self._load_npz()
        else:
            self._load_csv()
        if len(self.t) == 0:
            raise ValueError(f"{path}: empty setpoint log")

    def _load_npz(self) -> None:
        d = np.load(self.path, allow_pickle=False)
        self.t = np.asarray(d["t_s"], float)
        self.pos = np.asarray(d["pos"], float)
        self.quat = np.asarray(d["quat"], float)
        self.q = np.asarray(d["q"], float) if "q" in d.files else None
        self.press = (np.asarray(d["press_cmd_m"], float)
                      if "press_cmd_m" in d.files else np.zeros(len(self.t)))
        names = ([str(s) for s in d["phase_names"]]
                 if "phase_names" in d.files else list(PHASES))
        self.phase = ([names[int(i)] for i in d["phase_idx"]]
                      if "phase_idx" in d.files else ["draw"] * len(self.t))

    def _load_csv(self) -> None:
        t, pos, quat, q = [], [], [], []
        with open(self.path, newline="") as fh:
            for r in csv.DictReader(fh):
                t.append(float(r["t_s"]))
                pos.append([float(r["x"]), float(r["y"]), float(r["z"])])
                quat.append([float(r["qx"]), float(r["qy"]), float(r["qz"]),
                             float(r["qw"])])
                cols = [r.get(f"q{i}", "") for i in range(1, 8)]
                q.append([float(c) for c in cols]
                         if all(c not in (None, "") for c in cols) else None)
        self.t = np.asarray(t, float)
        self.pos = np.asarray(pos, float)
        self.quat = np.asarray(quat, float)
        self.q = None if any(x is None for x in q) else np.asarray(q, float)
        self.press = np.zeros(len(self.t))
        self.phase = ["draw"] * len(self.t)

    @property
    def duration_s(self) -> float:
        return float(self.t[-1] - self.t[0])

    @property
    def has_joint_columns(self) -> bool:
        return self.q is not None

    def __iter__(self) -> Iterator[Setpoint]:
        n = int(round(self.duration_s * self.rate_hz)) + 1
        j = 0
        for k in range(n):
            t = self.t[0] + k * self.dt
            while j + 1 < len(self.t) and self.t[j + 1] <= t + 1e-12:
                j += 1                       # hold-last, never interpolate
            yield Setpoint(k * self.dt, self.pos[j], self.quat[j],
                           None if self.q is None else self.q[j],
                           self.phase[j], float(self.press[j]))
