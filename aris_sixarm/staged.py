"""STAGE ASSIGNMENT, WIRED END TO END: pieces -> plans -> legs -> a stage programme.

Build item 4 of docs/ARCHITECTURE_V2.md.  `traces.py` answers "how few pieces
can this picture be drawn in, and who draws which, in which stage".  This module
takes that answer and turns it into something an arm could actually be handed: a
certified joint trajectory per (stage, arm), the pen-up legs between the pieces,
the order they are drawn in, and the two independent checks a stage has to pass
before anybody believes it.

THERE ARE TWO SAFETY QUESTIONS IN A STAGE AND THEY HAVE TWO DIFFERENT TOOLS.
This is the correction docs/V2_WORKCELLS.md section 5 needs and it is the whole
design of this module:

  BETWEEN TWO ACTIVE ARMS, whose poses are not yet chosen when the stage is
  designed, the criterion is the ENVELOPE one: the two work cells' envelopes --
  the union over every pose either arm could hold anywhere inside its cell --
  must be at least `PAIR_MARGIN` apart.  The zigzag's 0.40 m dead band gives
  +85.8 mm and that is the stage design; it is settled and this module does not
  re-open it.  What this module DOES do is re-measure it from the trajectories
  the planner actually produced, which is a strictly weaker claim than the
  envelope and therefore a check rather than a certificate: `active_pair_gap`.

  BETWEEN AN ACTIVE ARM AND A PARKED PARTNER, the partner is not an envelope at
  all.  It is one known, measured, barrier-verified pose, and the right tool is
  the per-stroke SOLO certification this repo already ships: `frozen.freeze`
  puts the partner's REAL capsules into the static set (aris_sixarm/frozen.py),
  the planner chooses poses and legs that clear 50 mm against them, and
  `scene_check.check_timeline` re-derives the result with no planner state.
  The certified 3-layer map at h = 0.970 with the shipped parks frozen is
  already 100 % of the block at the 50 mm gate, so NO park search, NO cell
  erosion and NO gate change is needed for the parked half of the problem.

  THE "+4.7 mm ENVELOPE VS A PARKED PARTNER" OF V2_WORKCELLS section 5 IS THE
  WRONG TEST FOR A PARKED PARTNER.  It measures an active arm's whole work-cell
  envelope against a parked arm's whole envelope -- a union over poses the
  parked arm will not hold, because it is holding one.  A parked arm is a
  constant, and a constant is certified cell by cell and leg by leg, not as a
  union.  See docs/DECISIONS.md, the 2026-09-11 correction entry.

WHAT IS NOT HERE, ON PURPOSE.  No balancer, no split and no merge: the DP in
`traces.py` already minimised the piece count exactly, and a second optimiser
over the same pieces would be optimising against a cost the first one did not
pay.  No conductor for the main stages either -- three actives in a stage are
static keep-outs for each other, so there is no clock to share (the seam stages
are two-active and `idle.conduct` is still the tool for them; this module does
not invoke it, and says so in its output rather than pretending).

    ARIS_RIG=proposed ARIS_TOOL=lateral python3 -m aris_sixarm.staged --help
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from . import allocate, coordination, frozen, paper, scene_check, stroke_api
from . import traces as traces_mod
from . import writing
from .fleet import FLEET, H_INV_DEFAULT

# The programme this module writes is NOT `program_schema.Bundle`: that schema's
# `_from_dict` refuses unknown AND missing keys, so a stage id and a per-piece
# hover are a `SCHEMA_VERSION` bump rather than an extension, and folding the
# two together is build item 5.  Until then the staged programme carries its own
# version, and it is a superset of what item 5 has to absorb: stage id, barrier
# list, per-piece entry/exit hover configurations and per-piece q_first/q_last.
STAGED_SCHEMA_VERSION = 1

ATLAS_DEFAULT = "out/atlas_proposed_h0970_lat0860_gated63"

PAIR_MARGIN = coordination.PAIR_MARGIN      # 50 mm, the arm-to-arm gate
SWEEP_FRAC = 0.55                           # `scene_check.check_timeline`'s own
CHECK_DT = 0.05                             # s, the clock the checks sample on
MAX_CHECK_POSES = 900                       # per arm, per stage, for check (a)


# ---------------------------------------------------------------------------
# 1.  THE PIECES, OUT OF THE DP
# ---------------------------------------------------------------------------
@dataclass
class Piece:
    """One maximal run of a line, with the (stage, arm) the DP gave it."""
    stage: int
    arm: int
    line: int
    k: int                              # which piece of that line
    pts: np.ndarray                     # (N, 2) paper metres, already lapped
    length_m: float

    @property
    def key(self) -> tuple[int, int, int, int]:
        return (self.stage, self.arm, self.line, self.k)


def pieces_of(plan: "traces_mod.Plan") -> list[Piece]:
    """Every drawn piece of every line, in line order. -> [Piece].

    `LinePlan.piece_points` is what the arm is actually told to draw -- the run
    lapped by `OVERDRAW_M` at an overlap seam and abutting exactly at a hard
    edge -- so this is the geometry, not the run.
    """
    out: list[Piece] = []
    for lp in plan.lines:
        for k, p in enumerate(lp.pieces):
            pts = lp.piece_points(k)
            if len(pts) < 2:
                continue
            out.append(Piece(int(p.stage), int(p.arm), int(lp.index), int(k),
                             np.asarray(pts, float),
                             float(traces_mod.cumlen(pts)[-1])))
    return out


def bucket(pieces: Sequence[Piece]) -> dict[tuple[int, int], list[Piece]]:
    """-> {(stage, arm): [Piece]}, the unit this module plans and orders."""
    out: dict[tuple[int, int], list[Piece]] = {}
    for p in pieces:
        out.setdefault((p.stage, p.arm), []).append(p)
    return out


def stage_actives(pattern: "traces_mod.Pattern", stage: int) -> tuple[int, ...]:
    """The arms a stage names.  Everybody else is parked, by definition."""
    return tuple(sorted({int(c.arm) for c in pattern.stage(stage)}))


def same_row_pairs(arms: Iterable[int]) -> list[tuple[int, int]]:
    """Pairs of arms on the same ROW of the base lattice. -> [(a, b)].

    THE TRANSVERSE PAIR IS UNSEPARABLE ON THIS RIG, on either axis
    (docs/V2_WORKCELLS.md section 4b): 0.80 m of x between two certified
    drawing poses is -160.8 mm and 0.81 m of y is -163.1 mm, because both
    elbows stand in the same column about the mid-line whatever the pens do.
    No stage may ever put one in the air together, so this is checked rather
    than assumed.
    """
    a = sorted(int(x) for x in arms)
    return [(i, j) for n, i in enumerate(a) for j in a[n + 1:]
            if traces_mod.ROW_OF.get(i) is not None
            and traces_mod.ROW_OF.get(i) == traces_mod.ROW_OF.get(j)]


# ---------------------------------------------------------------------------
# 2.  THE PARKED PARTNERS, AS REAL CAPSULES
# ---------------------------------------------------------------------------
def shipped_parks(specs=None) -> dict[int, np.ndarray]:
    """{arm: q} of the pose each arm holds when it is not the one drawing.

    `spec.q_seed` IS the shipped park on the proposed rig -- `layout.build_fleet`
    is called with `q_park=Q_PARK_PROPOSED` and stamps it in as `q_ready` -- and
    it is what `writing.arm_program` flies home to, so taking it from the spec
    is what keeps the pose this module freezes and the pose the timeline ends at
    the same pose by construction rather than by agreement.
    """
    fl = FLEET if specs is None else specs
    return {int(a): np.asarray(fl[a].q_seed, float).reshape(7) for a in fl}


def freeze_partners(arm: int, parks: dict[int, np.ndarray], specs=None,
                    pens=None, h_inv=H_INV_DEFAULT, leg_cache=True,
                    leg_cache_root=None) -> tuple[int, ...]:
    """Model every arm but `arm` by its REAL capsules at its park. -> the ids.

    This is the parked half of the safety argument and it is the whole of it:
    the five partners stop being 0.32 m pose-invariant bounding bands and become
    the capsules of the pose they are actually holding, checked at the same
    floors the bands were checked at (`frozen.py`).  Everything downstream that
    reaches `paper.static_boxes` -- the router, the hover solver, the screen --
    then sees the room the arm is really in.

    THE LEG STORE IS RE-NAMESPACED ON THE FROZEN SET, and it has to be.
    `paper.route_key` does not contain the frozen set (the set changes
    `static_boxes` without changing any memo key), so the store refuses to
    answer at all while `frozen` is in a state it was not opened under
    (`paper._dyn_state`).  Opening it under a signature that CONTAINS the frozen
    poses gives each (stage, arm) its own directory, which is both correct --
    arm 13's routes past a parked 17 are not arm 17's routes past a parked 13 --
    and warm across runs, which is the whole point of build item 1.
    """
    fl = FLEET if specs is None else specs
    pens = {a: fl[a].pen for a in fl} if pens is None else pens
    others = {int(a): np.asarray(q, float).reshape(7)
              for a, q in parks.items() if int(a) != int(arm) and a in fl}
    frozen.freeze(others, fl, pens, h_inv)
    frozen.observe(int(arm))
    paper.clear_cache()                 # the memos are not keyed on the set
    if leg_cache:
        paper.disk_cache_open(leg_cache_root,
                              signature=leg_cache_signature(others))
    else:
        paper.disk_cache_close()
    return tuple(sorted(others))


def leg_cache_signature(frozen_poses: dict[int, np.ndarray]) -> str:
    """`paper.cache_signature()` plus the frozen set it is valid under."""
    blob = json.dumps(dict(
        base=paper.cache_signature(),
        frozen={str(int(a)): [round(float(x), 9) for x in np.asarray(q, float).ravel()]
                for a, q in sorted(frozen_poses.items())}), sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def thaw():
    """Back to the shipped pose-invariant partner model."""
    frozen.thaw()
    paper.clear_cache()


# ---------------------------------------------------------------------------
# 3.  ONE (STAGE, ARM) BUCKET: PLAN, ORDER, FLY
# ---------------------------------------------------------------------------
@dataclass
class PiecePlan:
    """A piece and what `plan_stroke` made of it."""
    piece: Piece
    status: str                         # "ok" | "split" | "degenerate" | "bug"
    reason: str = ""
    plan: dict | None = None
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == "ok"


@dataclass
class ArmStage:
    """One arm's whole job in one stage: plans, order, legs, trajectory."""
    stage: int
    arm: int
    q_park: np.ndarray
    planned: list[PiecePlan] = field(default_factory=list)
    programme: list[dict] = field(default_factory=list)   # ordered seg dicts
    order: list[int] = field(default_factory=list)
    timeline: dict | None = None
    hovers: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    frozen_partners: tuple[int, ...] = ()
    frozen_poses: dict[int, list[float]] = field(default_factory=dict)
    plan_s: float = 0.0
    seq_s: float = 0.0
    prog_s: float = 0.0
    wall_s: float = 0.0
    note: str = ""

    @property
    def refused(self) -> list[PiecePlan]:
        return [p for p in self.planned if not p.ok]

    @property
    def accepted(self) -> list[PiecePlan]:
        return [p for p in self.planned if p.ok]

    @property
    def duration(self) -> float:
        return 0.0 if self.timeline is None else float(self.timeline["duration"])

    @property
    def ink_m(self) -> float:
        return float(sum(p.piece.length_m for p in self.accepted))


def _seg_of(pp: PiecePlan, index: int) -> dict:
    """A `PiecePlan` in the `segs` contract `sequence` and `writing` already take."""
    return dict(stroke_id=int(pp.piece.line), seg=int(index),
                color="", kind="piece", s_range=(0.0, 1.0), direction=1,
                flipped=False, length=float(pp.plan["arc_len"]),
                pts=np.asarray(pp.plan["pts"], float),
                stage=int(pp.piece.stage), piece=int(pp.piece.k),
                plan=pp.plan)


def plan_bucket(stage: int, arm: int, pieces: Sequence[Piece], specs=None,
                pens=None, parks=None, h_inv=H_INV_DEFAULT, opts=None,
                sequencer="opt", seq_opts=None, leg_cache=True,
                leg_cache_root=None, fly=True, on_piece: Callable | None = None,
                verbose=False) -> ArmStage:
    """Plan, order and fly one (stage, arm) bucket. -> ArmStage.

    The four steps, and none of them is new machinery:

      1. the FIVE PARTNERS ARE FROZEN at their parks before a single call is
         made, so every leg this bucket certifies is certified against the real
         capsules of the arms that are really standing there;
      2. every piece goes through `stroke_api.plan_stroke` -- the same funnel
         `scripts/csail_allocate.py` uses -- and a piece it refuses is RECORDED
         AND DROPPED rather than split: the DP's atlas-permitted piece count is
         a prefilter, and what fraction of it the local planner refuses is a
         measurement this module exists to take (docs/V2_STAGED.md);
      3. the accepted pieces are ordered by `allocate.sequence_arm`, which is
         `sequence.cost_matrix` + `sequence.solve` -- one tour per bucket, the
         same `segs` contract the allocator hands it today;
      4. `writing.arm_program` lays the timeline down: park -> entry hover ->
         draw -> exit hover -> ... -> park, with every pen-up leg a
         `paper.route` against the frozen room and the persistent store.

    `park=PARK_HOME` is not the idle policy this repo usually runs, and it is
    not a preference here either: a stage BARRIER is defined as every arm
    pen-up, stopped, at the park the NEXT stage's envelopes were certified
    against (docs/ARCHITECTURE_V2.md section 2e), so a stage programme that did
    not end there would not be a stage programme.
    """
    fl = FLEET if specs is None else specs
    pens = {a: fl[a].pen for a in fl} if pens is None else pens
    parks = shipped_parks(fl) if parks is None else parks
    spec = fl[arm]
    q_park = np.asarray(parks[arm], float).reshape(7)
    st = ArmStage(int(stage), int(arm), q_park)
    t_all = time.perf_counter()

    st.frozen_partners = freeze_partners(arm, parks, fl, pens, h_inv,
                                         leg_cache, leg_cache_root)
    st.frozen_poses = {int(a): [float(x) for x in q]
                       for a, q in frozen.poses().items()}

    o = dict(opts or {})
    o.setdefault("h_inv", h_inv)
    o.setdefault("pen_ext", pens.get(arm))
    t0 = time.perf_counter()
    for pc in pieces:
        if on_piece is not None:
            on_piece(stage, arm, pc)
        t1 = time.perf_counter()
        r = stroke_api.plan_stroke(pc.pts, spec, o)
        st.planned.append(PiecePlan(pc, str(r.get("status")),
                                    str(r.get("reason") or ""),
                                    r if r.get("status") == "ok" else None,
                                    time.perf_counter() - t1))
    st.plan_s = time.perf_counter() - t0
    good = st.accepted
    if verbose:
        print(f"  stage {stage} arm {arm}: {len(good)}/{len(pieces)} pieces "
              f"planned in {st.plan_s:.1f} s")
    if not good or not fly:
        st.programme = [_seg_of(p, i) for i, p in enumerate(good)]
        st.order = list(range(len(good)))
        return _finish(st, t_all)

    segs = [_seg_of(p, i) for i, p in enumerate(good)]
    so = dict(h_inv=h_inv, pen_ext=pens.get(arm), q_start=q_park,
              return_home=True)
    so.update(seq_opts or {})
    t0 = time.perf_counter()
    seq = allocate.sequence_arm(segs, spec, sequencer=sequencer, opts=o,
                                seq_opts=so)
    st.seq_s = time.perf_counter() - t0
    st.programme = list(seq["programme"])
    st.order = [int(i) for i in seq["order"]]

    t0 = time.perf_counter()
    try:
        st.timeline = writing.arm_program(
            spec, st.programme, h_inv=h_inv, pen_ext=pens.get(arm),
            q_start=q_park, park=writing.PARK_HOME,
            transit_speed=so.get("transit_speed", writing.TRANSIT_SPEED),
            qd_frac=so.get("qd_frac", writing.QD_FRAC))
    except writing.PaperRefused as exc:
        st.note = f"arm_program refused: {exc}"
    st.prog_s = time.perf_counter() - t0
    st.hovers = [(writing.lifted_or_lower(spec, np.asarray(s["plan"]["qs"])[0],
                                          np.asarray(s["plan"]["pts"])[0],
                                          h_inv=h_inv,
                                          pen_ext=pens.get(arm))[0],
                  writing.lifted_or_lower(spec, np.asarray(s["plan"]["qs"])[-1],
                                          np.asarray(s["plan"]["pts"])[-1],
                                          h_inv=h_inv,
                                          pen_ext=pens.get(arm))[0])
                 for s in st.programme]
    return _finish(st, t_all)


def _finish(st: ArmStage, t_all: float) -> ArmStage:
    st.wall_s = time.perf_counter() - t_all
    return st


# ---------------------------------------------------------------------------
# 4.  THE TWO CHECKS A STAGE HAS TO PASS
# ---------------------------------------------------------------------------
def _samples(st: ArmStage, dt: float, max_n: int, with_pen=False):
    """One arm's stage trajectory on a uniform clock, decimated. -> (N, 7).

    `with_pen` also returns the pen-down mask: `seg >= 0` is the sample the
    timeline is DRAWING at, which is the half of a trajectory the work-cell
    envelope is a claim about.
    """
    if st.timeline is None:
        Q = np.asarray(st.q_park, float).reshape(1, 7)
        return (Q, np.zeros(1, bool)) if with_pen else Q
    u = writing.uniform_samples(st.timeline, dt)
    Q, seg = np.asarray(u["q"], float), np.asarray(u["seg"], int)
    if len(Q) > max_n:
        keep = np.unique(np.linspace(0, len(Q) - 1, max_n).astype(int))
        Q, seg = Q[keep], seg[keep]
    return (Q, seg >= 0) if with_pen else Q


def _chains(Q: np.ndarray, spec, h_inv: float, pen: float) -> np.ndarray:
    return np.array([scene_check._chain(q, spec, h_inv, pen) for q in Q])


def active_pair_gap(stages: dict[int, ArmStage], specs=None, pens=None,
                    h_inv=H_INV_DEFAULT, dt=CHECK_DT, max_n=MAX_CHECK_POSES
                    ) -> dict:
    """Two ACTIVE arms' realised clearance, with NO assumption about timing.

    -> dict(min_m, per_pair, n_samples).

    WHY THIS IS NOT `check_timeline` ON A MERGED CLOCK.  Inside a stage the
    actives are asynchronous by construction -- that is the whole reason the
    stage needs no conductor -- so there is no single alignment of their two
    timelines to check, and checking one would certify a schedule nobody is
    running.  The claim that has to hold is the ENVELOPE claim restricted to
    the poses the planner actually produced: for EVERY pair of instants, one on
    each arm's timeline, the two arms clear the gate.  That is the minimum over
    the CROSS PRODUCT of the two pose sets, and it is what this computes.

    The between-sample bound is `check_timeline`'s own, applied on both axes:
    a chain point moves at most `step[i]` between grid samples, so
    `d(A(t), B(t')) >= d(A_i, B_j) - 0.55 (step^A_i + step^B_j)` for the grid
    cell containing (t, t').  Decimating makes the steps larger and the bound
    more conservative, never wrong.

    INK-VS-INK IS REPORTED SEPARATELY, and it is the number that answers the
    envelope's own question.  A work-cell envelope is a union over POSES -- the
    drawing pose at each certified cell, the hover above it, the park -- and the
    pen-up LEG between two hovers is a PATH whose interior is in none of them
    (docs/ARCHITECTURE_V2.md section 2f names this gap in as many words).  So
    `min_ink_m` is the two arms measured only where BOTH pens are down, which is
    the claim `scripts/workcell_envelopes.py` made and the one the 0.40 m dead
    band was chosen for; `min_m` is the whole trajectory, legs included, which
    is a strictly stronger statement and the one a stage has to pass.
    """
    fl = FLEET if specs is None else specs
    pens = {a: fl[a].pen for a in fl} if pens is None else pens
    arms = sorted(stages)
    rr = scene_check._radii_for(fl, arms)
    Q, DOWN = {}, {}
    for a in arms:
        Q[a], DOWN[a] = _samples(stages[a], dt, max_n, with_pen=True)
    P, SW = {}, {}
    for a in arms:
        P[a] = _chains(Q[a], fl[a], h_inv, scene_check.pen_len(pens.get(a), a))
        step = np.concatenate([[0.0], np.linalg.norm(np.diff(P[a], axis=0),
                                                     axis=2).max(1)])
        SW[a] = SWEEP_FRAC * np.maximum(step, np.concatenate([step[1:], [0.0]]))
    per: dict[str, float] = {}
    per_ink: dict[str, float] = {}
    worst = worst_ink = np.inf
    at = None
    for n, ai in enumerate(arms):
        for aj in arms[n + 1:]:
            Pi, Pj = P[ai], P[aj]
            m, mi = np.inf, np.inf
            for j in range(len(Pj)):
                d = scene_check.pair_clearance(
                    Pi, np.repeat(Pj[j][None], len(Pi), axis=0), rr) \
                    - SW[ai] - SW[aj][j]
                k = int(np.argmin(d))
                if float(d[k]) < m:
                    m = float(d[k])
                    if m < worst:
                        at = (int(ai), int(aj),
                              "ink" if DOWN[ai][k] else "leg",
                              "ink" if DOWN[aj][j] else "leg")
                if DOWN[aj][j] and DOWN[ai].any():
                    mi = min(mi, float(np.min(d[DOWN[ai]])))
            per[f"{ai}-{aj}"] = m
            per_ink[f"{ai}-{aj}"] = mi
            worst = min(worst, m)
            worst_ink = min(worst_ink, mi)
    return dict(min_m=float(worst), min_ink_m=float(worst_ink),
                worst_at=at, per_pair=per, per_pair_ink=per_ink,
                n_samples={int(a): int(len(Q[a])) for a in arms},
                n_ink={int(a): int(DOWN[a].sum()) for a in arms})


def drop_bands(spec, owners: Iterable[int]):
    """`spec` with the pose-invariant BODY BANDS of `owners` removed.

    A `body:<aid>_column<k>` box is a 0.32 m AABB standing in for a neighbour's
    shoulder and upper links AT ANY POSE, written in by
    `mounts.attach_body_columns` because at atlas-sweep time nobody has decided
    what pose the neighbour will hold (`frozen.py`).  In a stage timeline
    EVERYBODY'S POSE IS DECIDED -- the actives are in the trajectory and the
    parked partners are at an identified park -- so every one of those arms is
    already in the check as its real capsules, and its band is the same arm
    counted a second time in its most conservative form.

    This is `frozen.filter_boxes` applied to the CHECK instead of to the
    planner, and it is the same convention `scene_check.neighbour_columns`
    already uses for the cylinder version of the same object ("in the room" =
    not in this timeline).  TRUE STRUCTURE IS NEVER DROPPED: mounts, plates,
    the drop cluster and the runway all stay, and so does the band of any arm
    NOT named in `owners`.
    """
    import dataclasses
    own = {int(x) for x in owners}
    kw = {}
    for fname in ("mount_boxes", "column_boxes"):
        cur = getattr(spec, fname, None)
        if not cur:
            continue
        new = tuple(b for b in cur
                    if frozen.band_owner(b.get("name")
                                         if isinstance(b, dict) else None)
                    not in own)
        if len(new) != len(cur):
            kw[fname] = new
    return dataclasses.replace(spec, **kw) if kw else spec


def solo_check(st: ArmStage, parks: dict[int, np.ndarray], specs=None,
               pens=None, h_inv=H_INV_DEFAULT, margin=PAIR_MARGIN,
               dt=CHECK_DT, sub=2, bands=False) -> dict:
    """One active arm's whole stage timeline against the PARKED fleet and the steel.

    `scene_check.check_timeline` with the five partners held at their parks for
    every frame.  This is the independent re-derivation of the claim the planner
    made with `frozen` switched on: the same poses, no planner state, and the
    parked partners measured as the capsules they ARE rather than as the
    envelope they are not.

    `bands=False` (the default) drops the partners' pose-invariant body bands
    from the STATIC set, because every arm those bands stand for is in this
    timeline as its own capsules -- see `drop_bands`.  `bands=True` is the
    shipped pose-invariant reading, kept because the difference between the two
    is exactly what the frozen model buys and it should be reported rather than
    assumed: the band refuses what the arm does not, by up to 127 mm
    (`frozen.py`, measured 2026-09-09).
    """
    fl = FLEET if specs is None else specs
    pens = {a: fl[a].pen for a in fl} if pens is None else pens
    Q = _samples(st, dt, 10 ** 9)
    traj = {int(st.arm): Q}
    for a in sorted(fl):
        if int(a) == int(st.arm):
            continue
        traj[int(a)] = np.repeat(
            np.asarray(parks[a], float).reshape(1, 7), len(Q), axis=0)
    room = ({a: drop_bands(s, traj) for a, s in fl.items()} if not bands
            else fl)
    rep = scene_check.check_timeline(
        traj, dt, margin, programs=None, h_inv=h_inv,
        pen_ext={int(a): scene_check.pen_len(pens.get(a), a) for a in fl},
        sub=sub, verbose=False, fleet=room)
    # THE COLUMN CYLINDER IS THE SAME OBJECT AS THE BAND and it is reported the
    # same way: `neighbour_columns` hands out a cylinder for every other arm
    # whether or not it is in the timeline, so under the frozen model it is
    # informational and not a gate.  Everything else in the report -- inter-arm
    # (real capsules), steel, paper, self, joint limits -- is the verdict.
    rep["column_is_gate"] = bool(bands)
    rep["full_ok"] = bool(rep["ok"])   # check_timeline's own verdict, which
    #                                     also gates the column CYLINDER
    if not bands:
        rep["ok"] = bool(
            float(rep["min_clearance"]) >= float(margin)
            and not rep.get("frame_failed") and not rep.get("paper_failed")
            and not rep.get("self_failed") and int(rep.get("frozen_failed", 0)) == 0
            and min(rep["joint_margin"].values()) > 0.0
            and bool(rep.get("monotone", True)))
    return rep


# ---------------------------------------------------------------------------
# 5.  A WHOLE STAGE, AND THE WHOLE PROGRAMME
# ---------------------------------------------------------------------------
@dataclass
class StageResult:
    stage: int
    actives: tuple[int, ...]
    arms: dict[int, ArmStage]
    pair: dict = field(default_factory=dict)
    solo: dict = field(default_factory=dict)
    wall_s: float = 0.0

    @property
    def duration(self) -> float:
        """The stage costs its BUSIEST arm: the actives never wait for each other."""
        return max((a.duration for a in self.arms.values()), default=0.0)

    @property
    def n_pieces(self) -> int:
        return sum(len(a.accepted) for a in self.arms.values())

    @property
    def ink_m(self) -> float:
        return float(sum(a.ink_m for a in self.arms.values()))

    @property
    def ok(self) -> bool:
        pair_ok = self.pair.get("min_m", np.inf) >= PAIR_MARGIN
        solo_ok = all(r.get("ok", False) for r in self.solo.values())
        return bool(pair_ok and solo_ok)


@dataclass
class StagedResult:
    pattern: str
    stages: list[StageResult]
    pieces: list[Piece]
    dp_s: float = 0.0
    plan_s: float = 0.0
    check_s: float = 0.0
    wall_s: float = 0.0
    ttfm_s: float | None = None
    parks: dict[int, list[float]] = field(default_factory=dict)

    @property
    def makespan(self) -> float:
        """The barrier serialises the stages, so the sequence costs the SUM."""
        return float(sum(s.duration for s in self.stages))

    @property
    def parallel_plan_s(self) -> float:
        """Planning wall clock if each stage's actives plan on their own worker."""
        return float(sum(max((a.wall_s for a in s.arms.values()), default=0.0)
                         for s in self.stages))

    def barriers(self) -> list[dict]:
        """The rendezvous between consecutive stages. -> [dict].

        A barrier is not a time, it is a STATE: every arm pen-up, stopped, at
        the specific park the next stage's envelope set was certified against,
        with its queue drained and no un-cleared fault (docs/ARCHITECTURE_V2.md
        section 2e).  Recorded here as the park identity per arm, because park
        IDENTITY is what the guarantee is indexed by.
        """
        out = []
        for k, s in enumerate(self.stages):
            out.append(dict(kind="stage", index=k, before_stage=s.stage,
                            actives=[int(a) for a in s.actives],
                            parks={str(a): q for a, q in self.parks.items()}))
        out.append(dict(kind="end", index=len(self.stages), before_stage=None,
                        actives=[], parks={str(a): q
                                           for a, q in self.parks.items()}))
        return out


def run(lines, pattern=None, coverage=None, specs=None, pens=None, parks=None,
        h_inv=H_INV_DEFAULT, opts=None, trace_opts=None, stages=None,
        leg_cache=True, leg_cache_root=None, check=True, fly=True,
        measure_ttfm=True, dt=CHECK_DT, max_check_poses=MAX_CHECK_POSES,
        route_jobs=None, verbose=True, on_piece=None) -> StagedResult:
    """The whole of build item 4: lines -> pieces -> plans -> legs -> checks.

    `lines` are polylines in paper metres, `pattern` a `traces.Pattern` (the
    eight-stage zigzag by default) and `coverage` a `traces.Coverage` (the
    shipped atlas by default).  Everything else is the fleet's own.
    """
    t_all = time.perf_counter()
    fl = FLEET if specs is None else specs
    pens = {a: fl[a].pen for a in fl} if pens is None else pens
    parks = shipped_parks(fl) if parks is None else parks
    pattern = traces_mod.zigzag_pattern() if pattern is None else pattern
    # THE ROUTE SCREEN'S FORK COUNT IS A CALLER'S BUSINESS.  `sequence.route_jobs`
    # defaults to `min(cpu_count, 24)`, which is the right answer for a machine
    # running one allocation and the wrong one for a box running several stages
    # (or several of these) at once; `sequence.ROUTE_JOBS` is the knob that is
    # already there for it and `tests/test_balance.py` already sets it the same
    # way.  `None` leaves the module exactly as shipped.
    from . import sequence as _sequence
    jobs0 = _sequence.ROUTE_JOBS
    if route_jobs is not None:
        _sequence.ROUTE_JOBS = int(route_jobs)
    if coverage is None:
        coverage = traces_mod.coverage_from_atlas(ATLAS_DEFAULT,
                                                  arms=tuple(sorted(fl)))
    cap = traces_mod.capability(coverage, pattern)

    t0 = time.perf_counter()
    tplan = traces_mod.plan_lines(lines, cap, trace_opts)
    pcs = pieces_of(tplan)
    dp_s = time.perf_counter() - t0
    buckets = bucket(pcs)
    want = (range(pattern.n_stages) if stages is None else list(stages))

    ttfm = None
    if measure_ttfm:
        ttfm = time_to_first_motion(buckets, want, fl, pens, parks, h_inv, opts,
                                    leg_cache, leg_cache_root, dp_s)

    out: list[StageResult] = []
    plan_s = check_s = 0.0
    for s in want:
        acts = stage_actives(pattern, s)
        bad = same_row_pairs(acts)
        if bad:
            raise ValueError(f"stage {s} puts a same-ROW pair in the air "
                             f"together: {bad} -- see traces.zigzag_pattern")
        t1 = time.perf_counter()
        arms = {}
        for a in acts:
            arms[a] = plan_bucket(s, a, buckets.get((s, a), []), fl, pens,
                                  parks, h_inv, opts, leg_cache=leg_cache,
                                  leg_cache_root=leg_cache_root, fly=fly,
                                  on_piece=on_piece, verbose=verbose)
        plan_s += time.perf_counter() - t1
        sr = StageResult(int(s), acts, arms,
                         wall_s=time.perf_counter() - t1)
        if check and fly:
            t2 = time.perf_counter()
            thaw()          # the CHECK is not allowed to inherit the planner's
            sr.pair = active_pair_gap(arms, fl, pens, h_inv, dt,
                                      max_check_poses) if len(arms) > 1 else \
                dict(min_m=float("inf"), per_pair={}, n_samples={})
            sr.solo = {int(a): solo_check(arms[a], parks, fl, pens, h_inv,
                                          PAIR_MARGIN, dt)
                       for a in acts}
            check_s += time.perf_counter() - t2
        out.append(sr)
        if verbose:
            _report_stage(sr)
    thaw()
    _sequence.close_pool()
    _sequence.ROUTE_JOBS = jobs0
    res = StagedResult(pattern.name, out, pcs, dp_s, plan_s, check_s,
                       time.perf_counter() - t_all, ttfm,
                       {int(a): [float(x) for x in np.asarray(q).ravel()]
                        for a, q in parks.items()})
    return res


def time_to_first_motion(buckets, want, specs, pens, parks, h_inv, opts,
                         leg_cache, leg_cache_root, dp_s: float) -> float:
    """Seconds from "the picture is in hand" to "the first arm may move".

    The DP, the first piece of the first stage's first arm, its entry hover and
    the park -> hover leg.  Nothing else gates the first motion: the rest of
    that arm's bucket, and every other arm, is planned behind the pens
    (docs/ARCHITECTURE_V2.md section 1.2).
    """
    first = None
    for s in want:
        cand = sorted(k for k in buckets if k[0] == s and buckets[k])
        if cand:
            first = cand[0]
            break
    if first is None:
        return float("nan")
    s, a = first
    spec = specs[a]
    t0 = time.perf_counter()
    freeze_partners(a, parks, specs, pens, h_inv, leg_cache, leg_cache_root)
    o = dict(opts or {})
    o.setdefault("h_inv", h_inv)
    o.setdefault("pen_ext", pens.get(a))
    pc = buckets[first][0]
    r = stroke_api.plan_stroke(pc.pts, spec, o)
    if r.get("status") != "ok":
        return float("nan")
    qs, pts = np.asarray(r["qs"], float), np.asarray(r["pts"], float)
    hov, z = writing.lifted_or_lower(spec, qs[0], pts[0], h_inv=h_inv,
                                     pen_ext=pens.get(a))
    leg = paper.route(spec, np.asarray(parks[a], float).reshape(7), hov,
                      pen_ext=pens.get(a), h_inv=h_inv,
                      tip_floor=paper.travel_floor(z, z))
    if leg is None:
        return float("nan")
    return float(dp_s + time.perf_counter() - t0)


def _report_stage(sr: StageResult) -> None:
    pm = sr.pair.get("min_m")
    sm = min((r.get("min_clearance", np.inf) for r in sr.solo.values()),
             default=np.inf)
    print(f"stage {sr.stage}: arms {list(sr.actives)}  "
          f"{sr.n_pieces} pieces  {sr.ink_m:.3f} m  "
          f"{sr.duration:.1f} s"
          + ("" if pm is None else
             f"  active-pair {1000 * pm:+.1f} mm "
             f"(ink {1000 * sr.pair['min_ink_m']:+.1f})")
          + (f"  solo {1000 * sm:+.1f} mm" if np.isfinite(sm) else "")
          + ("" if not sr.solo else
             f"  [{'PASS' if sr.ok else 'FAIL'}]"))


# ---------------------------------------------------------------------------
# 6.  THE TYPED PROGRAMME
# ---------------------------------------------------------------------------
def programme(res: StagedResult, trajectories: bool = True) -> dict:
    """The staged programme, as the document item 5 has to absorb. -> dict.

    Per stage: the actives, and per arm its ordered pieces (each with the stage
    it belongs to, the piece's own polyline, the first and last joint vectors of
    its certified plan, and the ENTRY and EXIT HOVER CONFIGURATIONS that the
    pen-up legs fly between), the pen-up legs as the waypoint blocks
    `writing.arm_program` laid down, and the joint trajectory.  Plus the
    barrier list.

    `Segment` in `program_schema` carries no joint vector at all and
    `export_bundle` drops `q_first`/`q_last`, so this is a superset of that
    schema rather than an instance of it; folding the two together is build
    item 5 and it is a `SCHEMA_VERSION` bump, not a silent extension.
    """
    doc = dict(schema=int(STAGED_SCHEMA_VERSION), pattern=res.pattern,
               n_stages=len(res.stages), parks=res.parks,
               pair_margin_m=float(PAIR_MARGIN),
               makespan_s=float(res.makespan),
               ttfm_s=(None if res.ttfm_s is None else float(res.ttfm_s)),
               timing=dict(dp_s=float(res.dp_s), plan_s=float(res.plan_s),
                           check_s=float(res.check_s),
                           parallel_plan_s=float(res.parallel_plan_s),
                           wall_s=float(res.wall_s)),
               barriers=res.barriers(), stages=[])
    for sr in res.stages:
        one = dict(stage=int(sr.stage), actives=[int(a) for a in sr.actives],
                   duration_s=float(sr.duration), n_pieces=int(sr.n_pieces),
                   ink_m=float(sr.ink_m),
                   checks=dict(active_pair=_jsonable(sr.pair),
                               solo={str(a): dict(
                                   ok=bool(r.get("ok")),
                                   min_clearance_m=float(r.get("min_clearance",
                                                               np.nan)))
                                   for a, r in sr.solo.items()}),
                   arms={})
        for a, st in sr.arms.items():
            hov = st.hovers or [(None, None)] * len(st.programme)
            pieces = []
            for i, sg in enumerate(st.programme):
                qs = np.asarray(sg["plan"]["qs"], float)
                pieces.append(dict(
                    stage=int(sr.stage), arm=int(a), line=int(sg["stroke_id"]),
                    piece=int(sg.get("piece", i)), order=int(i),
                    flipped=bool(sg.get("flipped", False)),
                    home_before=bool(sg.get("home_before", False)),
                    length_m=float(sg["length"]),
                    q_first=[float(x) for x in qs[0]],
                    q_last=[float(x) for x in qs[-1]],
                    hover_in=(None if hov[i][0] is None
                              else [float(x) for x in np.asarray(hov[i][0]).ravel()]),
                    hover_out=(None if hov[i][1] is None
                               else [float(x) for x in np.asarray(hov[i][1]).ravel()]),
                    pts=np.round(np.asarray(sg["pts"], float), 5).tolist()))
            tl = st.timeline
            one["arms"][str(a)] = dict(
                arm=int(a), q_park=[float(x) for x in np.asarray(st.q_park).ravel()],
                frozen_partners=[int(x) for x in st.frozen_partners],
                n_pieces=len(pieces), ink_m=float(st.ink_m),
                duration_s=float(st.duration),
                refused=[dict(line=int(p.piece.line), piece=int(p.piece.k),
                              length_m=float(p.piece.length_m),
                              status=p.status, reason=p.reason)
                         for p in st.refused],
                wall=dict(plan_s=float(st.plan_s), seq_s=float(st.seq_s),
                          prog_s=float(st.prog_s), total_s=float(st.wall_s)),
                legs=([dict(kind=ph["kind"], seg=int(ph["seg"]),
                            t0=float(ph["t0"]), t1=float(ph["t1"]))
                       for ph in tl["phases"] if ph["kind"] != "stroke"]
                      if tl else []),
                pieces=pieces,
                trajectory=(dict(t=np.round(tl["t"], 6).tolist(),
                                 q=np.round(tl["q"], 6).tolist(),
                                 seg=[int(x) for x in tl["seg"]])
                            if (tl and trajectories) else None))
        doc["stages"].append(one)
    return doc


def _jsonable(d: dict) -> dict:
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            out[k] = {str(kk): (float(vv) if isinstance(vv, (int, float))
                                else vv) for kk, vv in v.items()}
        elif isinstance(v, (int, float, np.floating)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def summary(res: StagedResult) -> dict:
    """The table docs/V2_STAGED.md quotes."""
    rows = []
    for sr in res.stages:
        rows.append(dict(
            stage=int(sr.stage), actives=[int(a) for a in sr.actives],
            pieces=int(sr.n_pieces), ink_m=round(sr.ink_m, 4),
            duration_s=round(sr.duration, 3),
            plan_s_per_arm={str(a): round(st.wall_s, 2)
                            for a, st in sr.arms.items()},
            plan_s_parallel=round(max((st.wall_s for st in sr.arms.values()),
                                      default=0.0), 2),
            plan_s_serial=round(sum(st.wall_s for st in sr.arms.values()), 2),
            refused=int(sum(len(st.refused) for st in sr.arms.values())),
            active_pair_mm=(None if not sr.pair else
                            round(1000 * sr.pair["min_m"], 2)),
            active_pair_ink_mm=(None if not sr.pair else
                                round(1000 * sr.pair["min_ink_m"], 2)),
            active_pair_at=(None if not sr.pair else sr.pair.get("worst_at")),
            solo_min_mm=(None if not sr.solo else round(1000 * min(
                r.get("min_clearance", np.nan) for r in sr.solo.values()), 2)),
            ok=bool(sr.ok)))
    n_pieces = len(res.pieces)
    refused = sum(len(st.refused) for sr in res.stages
                  for st in sr.arms.values())
    planned = sum(len(st.planned) for sr in res.stages
                  for st in sr.arms.values())
    return dict(pattern=res.pattern, stages=rows,
                n_pieces_dp=int(n_pieces), n_pieces_planned=int(planned),
                n_refused=int(refused),
                refused_frac=(refused / planned if planned else 0.0),
                makespan_s=round(res.makespan, 3),
                dp_s=round(res.dp_s, 4), plan_s=round(res.plan_s, 2),
                check_s=round(res.check_s, 2), wall_s=round(res.wall_s, 2),
                parallel_plan_s=round(res.parallel_plan_s, 2),
                ttfm_s=(None if res.ttfm_s is None else round(res.ttfm_s, 3)),
                all_ok=bool(all(s.ok for s in res.stages)))


def refusal_table(res: StagedResult) -> dict:
    """What refuses a piece the 2 cm atlas permitted. -> {reason: count}."""
    out: dict[str, int] = {}
    for sr in res.stages:
        for st in sr.arms.values():
            for p in st.refused:
                k = f"{p.status}:{p.reason}" if p.reason else p.status
                out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


# ---------------------------------------------------------------------------
# 7.  CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--atlas", default=ATLAS_DEFAULT)
    ap.add_argument("--lines", default=None,
                    help="a JSON file of polylines in paper metres")
    ap.add_argument("--synthetic", default=None, help="kind[:n]")
    ap.add_argument("--stages", default=None, help="e.g. '0,1' (default: all)")
    ap.add_argument("--no-fly", action="store_true",
                    help="plan the pieces and stop: no ordering, no legs, no "
                         "timeline and no checks (the refusal measurement)")
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--no-leg-cache", action="store_true")
    ap.add_argument("--leg-cache", default=None)
    ap.add_argument("--tilt-max-deg", type=float, default=0.0)
    ap.add_argument("--gate", choices=("strict", "flat"), default="strict")
    ap.add_argument("--dt", type=float, default=CHECK_DT)
    ap.add_argument("--route-jobs", type=int, default=6,
                    help="processes the pen-up route screen may fork "
                         "(sequence.ROUTE_JOBS; 0 = the machine decides)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--programme", default=None)
    a = ap.parse_args(argv)

    if a.lines:
        lines = traces_mod.load_lines(a.lines)
        name = Path(a.lines).name
    elif a.synthetic:
        kind, _, n = a.synthetic.partition(":")
        lines = traces_mod.synthetic(kind, int(n or 1000))
        name = f"synthetic {kind} x{n or 1000}"
    else:
        raise SystemExit("one of --lines / --synthetic is required")
    cov = traces_mod.coverage_from_atlas(a.atlas, arms=tuple(sorted(FLEET)),
                                         gate=a.gate,
                                         tilt_max_deg=a.tilt_max_deg)
    stages = None if not a.stages else [int(x) for x in a.stages.split(",")]
    print(f"=== {name}: {len(lines)} lines, "
          f"{sum(float(traces_mod.cumlen(np.asarray(p, float))[-1]) for p in lines):.2f} m ===")
    res = run(lines, coverage=cov, opts=dict(tilt_max_deg=a.tilt_max_deg),
              stages=stages, leg_cache=not a.no_leg_cache,
              leg_cache_root=a.leg_cache, check=not a.no_check,
              fly=not a.no_fly, dt=a.dt, route_jobs=a.route_jobs)
    d = summary(res)
    print(json.dumps(d, indent=1))
    print("refusals:", json.dumps(refusal_table(res)))
    if a.json:
        Path(a.json).write_text(json.dumps(
            dict(d, refusals=refusal_table(res), case=name), indent=1))
        print("wrote", a.json)
    if a.programme:
        Path(a.programme).write_text(json.dumps(programme(res)))
        print("wrote", a.programme)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
