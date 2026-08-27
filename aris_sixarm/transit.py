"""PEN-UP TRANSIT AS A SEARCH IN CONFIGURATION SPACE.

THE HOLE THIS FILLS.  `paper.route` is a SHAPE LADDER: forty-odd fixed
shapes — the direct move, a lift over either end, retract-then-go, an arc, a
cartesian traverse in three step sizes, a walk, a fold through the depot, a
skirt round a column's footprint — every one of them a chain of hover poses
solved by `writing.hover_solve` and certified leg by leg.  It is a good
ladder: measured on the CSAIL logo at the v8 placement it settles 480 of 520
span-to-span crossings, most of them on the first or second rung, and it does
it for a few milliseconds each.

What it cannot do is REconfigure.  `paper.py` says it outright, and it says it
from a measurement: adding 8 cm and 5 cm rungs to `TRAVERSE_STEPS` recovers 2
of 520 blocked crossings for 1.9x the clock, "because a fold is not a long hop
on the hover plane, it is a change of IK branch that no walk through hover
poses avoids".  Every shape on the ladder is a walk on a two-dimensional
surface — a tip position on a hover plane, with the elbow following whatever
the analytic solver hands back — and the transits that remain blocked are the
ones whose two endpoints are in different components of that surface while
being perfectly well connected in the seven-dimensional space the arm actually
moves in.

So this module searches the seven-dimensional space.  Bidirectional
RRT-Connect between two configurations that are ALREADY CERTIFIED, with every
edge held to the same gate stack a ladder leg pays, and the assembled path
handed back to `paper.route`'s own `legs_ok` to be certified whole.  It is the
LAST tier and never the first: the ladder is cheaper by two orders of
magnitude and wins the overwhelming majority of crossings, so this runs only
where the ladder is exhausted.

WHY THE EDGES ARE CERTIFIED THE WAY THEY ARE (and what was checked first)
------------------------------------------------------------------------
`~/git/cc_experiment` holds `certified_ccd`, a genuinely certified continuous
collision checker for Drake: it converts a trajectory to piecewise Bezier,
bounds robot-point motion over each interval with per-joint motion constants,
and returns `kCertifiedFree` as a statement about the CONTINUUM rather than
about samples — 0.251 ms for a 7-DOF straight joint-space edge, faster than
Drake's own sampled checker at 0.05 rad and 9x faster at 0.01.  That is
strictly stronger than what this module does and it was the first thing
looked at.  It is not usable from here today, for reasons that are about
plumbing and not about the mathematics:

  * IT HAS NO PYTHON SEAM.  There are no bindings of any kind — `docs/SPEC.md`
    lists a `bindings/` directory that does not exist, the CMake target is a
    STATIC archive with no `find_package(Python)`, and the only executable is
    a hardcoded iiwa14 benchmark driver.  The public API takes
    `std::shared_ptr<const drake::planning::RobotDiagram<double>>`,
    `Eigen::MatrixXd` and `std::optional<Options>`; nothing about it is
    reachable through ctypes.
  * AND THE DRAKE IT LINKS IS NOT THE DRAKE THIS PACKAGE HAS.  It is built
    against a fork install (0.0.20251016, ~v1.45) whose pydrake is py3.12;
    the drake this repo's demos run under is a pip wheel, 1.47.0, py3.10.  A
    pybind11 module built against one cannot accept a `RobotDiagram` built by
    the other — different `libdrake.so`, different type-caster registry —
    which `cc_experiment/docs/UPSTREAMING.md` names as the reason it defers
    bindings in the first place.  Its own author's plan is to write them
    inside Drake during upstreaming.
  * AND IT CARRIES NO LICENCE FILE, which is fine for a local experiment and
    is not a thing to take a hard dependency on without asking.

So: days of ABI-shaped work, on the far side of a Drake migration, for a
checker this package would still have to restate independently — because the
discipline here is that PRODUCERS AND CHECKERS ARE SEPARATE DERIVATIONS and
`scene_check` would have to grow its own copy anyway.  The reuse is worth
doing later, as an A/B ORACLE against these bounds, and it is written down in
the final report rather than half-done here.

What ships instead is the bound this repo already lives by, applied to a new
kind of path: sample the edge, measure the gate at the samples, and SUBTRACT
the 1-Lipschitz residual for what the quantity can lose between two of them.
`paper.leg_bounds` and `paper.leg_self_lb` are those bounds and they are
refined to `scene_check`'s own density, so an edge this module accepts is an
edge the independent checkers pass.  The ordering the whole repo rests on is
preserved and then tightened: this module holds an edge to the ladder's floors
PLUS `PAD` (2 mm), so the producer is strictly tighter than the certifier that
grades it, and the certifier that grades it (`paper.route`'s `legs_ok`) is
itself strictly tighter than `scene_check`.

DETERMINISM.  A randomised planner in a pipeline with bit-identical replay is
only acceptable if its randomness is a pure function of its question.  The
seed here is `blake2b` over the scene signature — arm id, tool, pen, mount
height, every static box, the four floors — and both endpoints, so the same
question always produces the same tree, in any process, in any order, with or
without a memo, under `multiprocessing` or not.  `hash()` is deliberately not
used: it is salted per process.

WHAT IT COSTS.  One certified edge is a forward-kinematics pass over
`paper.SAMPLES` configurations against the near static boxes plus a screened
165-pair self test — 3 to 12 ms alone, and about 0.5 ms when the sub-edges of
one straight CONNECT are checked as a block, which is how this does it.  A
solved plan is typically a few hundred edges; a refused one spends its whole
budget.  Both numbers are reported by `stats()` and neither is hidden.
"""
import hashlib

import numpy as np

from . import paper, selfcoll
from .fleet import H_INV_DEFAULT
from .frames import (FR3_MAX, FR3_MIN, PEN_EXT, QD_MAX, joint_margin_many)
from . import frames as _frames

# --------------------------------------------------------------------------
# the knobs, and what each one is for
# --------------------------------------------------------------------------
RRT_SAFE = True         # is the C-space tier available at all
#   ...and it is part of `paper._key`, because a run with it off must not read
#   a route it bought with it on.  `--no-rrt` reproduces a pre-2026-08-26
#   number exactly.

STEP = 0.30             # seconds of `_dq_time` per tree extension
#   THE STEP IS MEASURED IN THE COST THE SEQUENCER PAYS, not in radians.  The
#   metric here is `max_i |dq_i| / QD_MAX_i`, which is `writing._dq_time`
#   without its speed fraction — so a step is a fixed slice of wall clock, the
#   nearest-neighbour query ranks candidates by what they will actually cost,
#   and shortcutting optimises the number the tour is priced on.  0.30 is about
#   0.9 rad on the slow joints and 1.6 on the wrist: long enough that a metre
#   of reconfiguration is a dozen edges, short enough that the sampled bound on
#   an edge is not mostly residual.

MAX_NODES = 900         # nodes per tree, per attempt
TIME_BUDGET = 2.5       # seconds per attempt
ATTEMPTS = 2            # restarts, each with its own derived seed
#   TWO BUDGETS, AND ONLY ONE OF THEM IS REPRODUCIBLE.  `MAX_NODES` is a
#   property of the search and gives the same answer on a loaded box as on an
#   idle one; `TIME_BUDGET` does not, and it is here anyway because a tier that
#   can hang is worse than a tier that can be non-deterministic on the clock.
#   The honest statement is the ordering: a plan that finishes inside its time
#   is reproducible, and a plan that runs out of clock is reported as
#   "the budget was spent", which is not a claim about the geometry.  Size
#   `MAX_NODES` so the node bound is the one that usually bites — measured on
#   the proposed rig, a solved transit uses 39 nodes and 598 certified edges,
#   two orders of magnitude inside it.
GOAL_BIAS = 0.10        # probability a sample is the other tree's root
BATCH = 96              # candidate configurations validated in one FK pass

PAD = 0.002             # m of floor a producer adds over its certifier's
#   `paper.route`'s `legs_ok` is what certifies the assembled path, and it is
#   the thing this must not disagree with.  Both sides compute lower bounds on
#   the same quantities at the same density; holding this side to 2 mm more
#   makes the disagreement one-directional, which is the property the whole
#   producer/checker discipline in this package rests on.  The direction is the
#   safe one: this can only refuse a path `legs_ok` would have taken.

LIMIT_MARGIN = 0.15     # rad from every joint stop a SAMPLED via must keep
VIA_MARGIN = 0.10       # ...and the floor EVERY via is held to, endpoints aside
#   `frames` says why the sampler keeps 0.15: libfranka enforces a
#   position-dependent velocity envelope near each stop and "we keep margin
#   >= 0.15 rad everywhere".
#
#   `VIA_MARGIN` is the harder constraint and it is not a preference, it is a
#   PINNED INVARIANT of this package: `tests/test_paper.py`'s
#   `test_every_via_is_a_pose_the_arm_may_stand_in` asserts
#   `joint_margin(v) >= writing.HOVER_MARGIN` for every via any route returns,
#   and it has always held because every via was a `hover_solve` IK solution
#   that was gated on exactly that.  A via this module invents is not an IK
#   solution and nothing would enforce it, so it is enforced here, on every
#   node the search keeps.
#
#   The GIVEN endpoints are exempt and have to be: they are certified poses
#   handed in by the caller and this module does not get to re-gate them.  The
#   joint box is convex and `joint_margin` concave on it, so a segment between
#   two configurations never holds less margin than the worse of its ends —
#   which is why enforcing the floor at the NODES is enough.

SHORTCUT_ROUNDS = 160   # random shortcut attempts on a found path
SHORTCUT_TIME = 1.0     # seconds, whichever comes first

EDGE_N = paper.SAMPLES  # configurations sampled along one candidate edge
#   THE SAME DENSITY `legs_ok` USES, on purpose.  A coarser screen would be a
#   second opinion about the same edge computed a different way, and the two
#   would eventually disagree; at the same density with a wider floor they
#   cannot.

_STATS = dict(calls=0, solved=0, failed=0, nodes=0, edges=0, seconds=0.0,
              shortcut_from=0, shortcut_to=0, recert_failed=0, deadline=0)
#   `deadline` counts the searches that stopped on the CLOCK rather than on
#   `max_nodes`.  It is the only thing between this module and a reproducible
#   answer, so it is counted rather than assumed away: a run that reports
#   `deadline == 0` gave the same answer it would give on any other box, and a
#   run that does not has to say so.  `scripts/feasible_workspace.py` prints it
#   for exactly that reason.


def stats():
    """What this module has done since the last `reset_stats`. -> dict."""
    return dict(_STATS)


def reset_stats():
    for k in _STATS:
        _STATS[k] = 0 if k != "seconds" else 0.0


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------
def scene_signature(spec, boxes, pen_ext, h_inv, floors):
    """Everything about the QUESTION that is not the two endpoints. -> bytes.

    Stable across processes and across runs — `arm_id` and the box geometry,
    never `id(spec)`.  A rig whose boxes moved is a different question and gets
    a different tree.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update(np.asarray([int(getattr(spec, "arm_id", -1)), float(pen_ext),
                         float(_frames.PEN_LAT), float(h_inv)],
                        float).tobytes())
    h.update(np.asarray([float(f) for f in floors], float).tobytes())
    if boxes:
        B = np.stack([np.concatenate([np.asarray(b["lo"], float),
                                      np.asarray(b["hi"], float)])
                      for b in boxes])
        h.update(np.round(B[np.lexsort(B.T[::-1])], 9).tobytes())
    return h.digest()


def seed_for(sig, q0, q1, attempt=0):
    """A reproducible 64-bit seed for one (scene, pair, attempt). -> int."""
    h = hashlib.blake2b(sig, digest_size=8)
    h.update(np.round(np.asarray(q0, float).reshape(7), 9).tobytes())
    h.update(np.round(np.asarray(q1, float).reshape(7), 9).tobytes())
    h.update(int(attempt).to_bytes(4, "little"))
    return int.from_bytes(h.digest(), "little")


# --------------------------------------------------------------------------
# the metric
# --------------------------------------------------------------------------
def dist(a, b):
    """`writing._dq_time`'s shape, without its speed fraction. -> seconds."""
    return float(np.max(np.abs(np.asarray(b, float) - np.asarray(a, float))
                        / QD_MAX))


def dist_many(Q, b):
    """`dist` from every row of `Q` to `b`. -> (N,)."""
    return np.max(np.abs(Q - np.asarray(b, float).reshape(1, 7)) / QD_MAX,
                  axis=1)


def steer(a, b, step=STEP):
    """`b`, pulled back to at most `step` of metric from `a`. -> (7,)."""
    a = np.asarray(a, float).reshape(7)
    b = np.asarray(b, float).reshape(7)
    d = dist(a, b)
    return b if d <= step else a + (b - a) * (step / d)


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------
class Gate:
    """The gate stack one pen-up leg pays, asked about edges instead of legs.

    Three obstacles and one budget, exactly the set `paper.route`'s `legs_ok`
    charges — the paper (`chain_floor`, `tip_floor`), the neighbours' steel and
    base columns (`static_floor`), the arm's own metal (`self_floor`) — plus
    the optional fourth this module is allowed to know about and the ladder is
    not: the PARKED PARTNERS, handed in as a probe by a caller that has them
    (`allocate.ParkProbe`, which is what `feasible_workspace` prices a cell
    with).  A route that a park probe will refuse three stages later is a route
    worth not finding.

    Every floor is raised by `PAD` here.  See the module docstring: this is the
    producer and `legs_ok` is the certifier, and the two are only allowed to
    disagree in one direction.
    """

    def __init__(self, spec, pen_ext=PEN_EXT, h_inv=H_INV_DEFAULT, boxes=None,
                 chain_floor=paper.CHAIN_CLEAR, tip_floor=paper.TIP_CLEAR,
                 static_floor=-np.inf, self_floor=-np.inf, probe=None,
                 probe_margin=0.0, pad=PAD, n=EDGE_N):
        self.spec = spec
        self.pen = float(pen_ext)
        self.h_inv = float(h_inv)
        self.boxes = paper.static_boxes(spec) if boxes is None else boxes
        self.chain_floor = float(chain_floor) + pad
        self.tip_floor = float(tip_floor) + pad
        self.static_floor = float(static_floor) + (pad if
                                                   np.isfinite(static_floor)
                                                   else 0.0)
        self.self_floor = float(self_floor) + (pad if np.isfinite(self_floor)
                                               else 0.0)
        self.probe = probe
        self.probe_margin = float(probe_margin) + pad
        self.n = int(n)
        self.edges = 0

    # ---- poses ----------------------------------------------------------
    def pose_ok(self, Q):
        """Which of these configurations may the arm STAND in? -> (N,) bool.

        Batched, because the sampler throws configurations away by the
        hundred and a forward-kinematics pass over a block of them is 70 us
        each against 700 one at a time.  A pose is not an edge and carries no
        residual: the same three obstacles, measured exactly.
        """
        Q = np.asarray(Q, float).reshape(-1, 7)
        if not len(Q):
            return np.zeros(0, bool)
        cz, tz, sc = paper.chain_screen(Q, self.spec, self.pen, self.h_inv,
                                        self.boxes)
        ok = (cz >= self.chain_floor) & (tz >= self.tip_floor)
        if self.boxes and np.isfinite(self.static_floor):
            ok &= sc >= self.static_floor
        if np.isfinite(self.self_floor) and ok.any():
            sv = np.zeros(len(Q), bool)
            idx = np.nonzero(ok)[0]
            A, B, R = selfcoll.capsule_ends(Q[idx], self.pen)
            sv[idx] = selfcoll.clearance_screened(A, B, R, self.self_floor) \
                >= self.self_floor
            ok &= sv
        if self.probe is not None and ok.any():
            idx = np.nonzero(ok)[0]
            keep = np.zeros(len(Q), bool)
            # `sweep=False`: a block of candidate poses is a SET, not a path,
            # and charging the residual between two unrelated rows would refuse
            # them for being far apart (`ParkProbe.clearance`'s own argument).
            for i in idx:
                keep[i] = float(self.probe(Q[i:i + 1], False)) \
                    >= self.probe_margin
            ok &= keep
        return ok

    # ---- edges ----------------------------------------------------------
    def edges_ok(self, pairs):
        """Certify a BLOCK of straight joint-space edges. -> (R,) bool.

        The whole point of the block form: a CONNECT toward a target is a
        straight line cut into sub-edges, so every sub-edge of one connect is
        checked in a single forward-kinematics pass.  Measured on this rig that
        is 0.46 ms an edge against 3.5 unbatched.

        The bound is `paper`'s, restated on a block: `block_screen` for the
        paper and the steel with its 1-Lipschitz residual returned alongside,
        `block_self_lb` for the arm's own metal with its own.  No refinement —
        an edge in the ambiguous band is REFUSED rather than looked at, which
        costs this module a few edges it could have had and cannot cost it an
        edge it should not have had.
        """
        if not len(pairs):
            return np.zeros(0, bool)
        A = np.stack([np.asarray(a, float).reshape(7) for a, _ in pairs])
        B = np.stack([np.asarray(b, float).reshape(7) for _, b in pairs])
        f = np.linspace(0.0, 1.0, self.n)[None, :, None]
        L = (A[:, None, :] * (1.0 - f) + B[:, None, :] * f)[:, None]  # (R,1,n,7)
        self.edges += len(pairs)
        _STATS["edges"] += len(pairs)
        cz, tz, sc, res = paper.block_screen(L, self.spec, self.pen,
                                             self.h_inv, self.boxes)
        ok = (cz[:, 0] >= self.chain_floor) & (tz[:, 0] >= self.tip_floor)
        if self.boxes and np.isfinite(self.static_floor):
            ok &= (sc[:, 0] - res[:, 0]) >= self.static_floor
        if np.isfinite(self.self_floor) and ok.any():
            sm, sres = paper.block_self_lb(L, self.pen, self.self_floor)
            ok &= (sm[:, 0] - sres[:, 0]) >= self.self_floor
        if self.probe is not None and ok.any():
            for i in np.nonzero(ok)[0]:
                ok[i] = float(self.probe(L[i, 0], True)) >= self.probe_margin
        return ok

    def edge_ok(self, a, b):
        return bool(self.edges_ok([(a, b)])[0])

    def line_prefix(self, a, b, step=STEP, via_margin=None):
        """How far along the straight line `a -> b` may the arm actually go?

        -> (q_reached, reached_b).  The line is cut into sub-edges of at most
        `step` of metric and ALL of them are certified in one pass; the answer
        is the last node of the longest certified prefix.  This is `CONNECT`,
        and doing it as a block instead of a loop is most of what makes the
        planner affordable.

        The prefix stops at the first refused SUB-EDGE and also at the first
        interior node that does not hold `via_margin` from its joint stops —
        an interior node of a prefix is a node the tree keeps and therefore a
        via somebody's timeline will stand in, and `VIA_MARGIN` explains why
        that is not optional.  The FINAL node is the target itself and is
        exempt: it is a sample the gate already passed, or another tree's node,
        or one of the two given endpoints, none of which this may re-gate.
        """
        a = np.asarray(a, float).reshape(7)
        b = np.asarray(b, float).reshape(7)
        d = dist(a, b)
        if d <= 1e-12:
            return a, True
        k = int(np.ceil(d / max(step, 1e-9)))
        f = np.linspace(0.0, 1.0, k + 1)[:, None]
        P = a[None] * (1.0 - f) + b[None] * f
        ok = self.edges_ok(list(zip(P[:-1], P[1:])))
        vm = VIA_MARGIN if via_margin is None else float(via_margin)
        if k > 1 and vm > -np.inf:
            thin = joint_margin_many(P[1:k]) < vm       # interior nodes only
            if thin.any():
                ok[int(np.nonzero(thin)[0][0]):] = False
        bad = np.nonzero(~ok)[0]
        j = int(bad[0]) if len(bad) else k
        return P[j], j == k


# --------------------------------------------------------------------------
# the planner
# --------------------------------------------------------------------------
class _Tree:
    __slots__ = ("Q", "parent", "n")

    def __init__(self, root, cap):
        self.Q = np.empty((cap, 7))
        self.parent = np.empty(cap, np.int32)
        self.Q[0] = np.asarray(root, float).reshape(7)
        self.parent[0] = -1
        self.n = 1

    def add(self, q, parent):
        if self.n >= len(self.Q):
            return -1
        self.Q[self.n] = q
        self.parent[self.n] = parent
        self.n += 1
        return self.n - 1

    def nearest(self, q):
        return int(np.argmin(dist_many(self.Q[:self.n], q)))

    def path_to(self, i):
        out = []
        while i >= 0:
            out.append(self.Q[i].copy())
            i = int(self.parent[i])
        return out[::-1]


class _Sampler:
    """A supply of configurations the arm may STAND in.

    Rejection sampling out of the joint box, validated a BLOCK at a time.
    Measured on the proposed rig at h = 0.940, 47 % of the box is standable
    for a mid-canvas arm, so a block of 96 yields about 45 usable draws for
    one forward-kinematics pass — which is why this buffers blocks instead of
    validating one draw at a time (70 us against 700).

    `next()` returns `None` rather than looping forever when a block yields
    nothing and the deadline has passed: a scene where almost nothing is
    standable is a scene this must refuse on the clock, not hang in.
    """

    def __init__(self, rng, gate, lo, hi, batch=BATCH):
        self.rng, self.gate, self.lo, self.hi = rng, gate, lo, hi
        self.batch, self.buf, self.i = int(batch), None, 0

    def next(self, deadline=None):
        import time
        while self.buf is None or self.i >= len(self.buf):
            if deadline is not None and time.perf_counter() >= deadline:
                return None
            Q = self.lo + self.rng.random((self.batch, 7)) * (self.hi - self.lo)
            self.buf, self.i = Q[self.gate.pose_ok(Q)], 0
        self.i += 1
        return self.buf[self.i - 1]


def plan(spec, q0, q1, pen_ext=PEN_EXT, h_inv=H_INV_DEFAULT, boxes=None,
         chain_floor=paper.CHAIN_CLEAR, tip_floor=paper.TIP_CLEAR,
         static_floor=-np.inf, self_floor=-np.inf, probe=None,
         probe_margin=0.0, step=None, max_nodes=None,
         time_budget=None, attempts=None, seeds=(), gate=None,
         shortcut=True, extra_roots=()):
    """A certified pen-up path from `q0` to `q1` in the joint space. -> [q]|None.

    -> the INTERIOR configurations only (`route`'s `vias` convention): the
    returned list `v` is such that every straight joint-space move in
    `[q0, *v, q1]` holds the four floors this was given, each of them raised by
    `PAD` so the certifier that grades the result cannot disagree.  `None` is
    "the budget was spent and no path was found", which is not the same claim
    as "no path exists" and is reported as what it is.

    Bidirectional RRT-Connect.  Two trees, one rooted at each endpoint; on each
    iteration one tree extends one step toward a sample and the other tries to
    CONNECT all the way to the new node, which is a single straight line and
    therefore a single blocked check.  Swap, repeat.  The classical algorithm,
    with the two things this application needs bolted on: a rejection sampler
    that only ever offers configurations the arm may stand in, and an edge
    oracle that is this repo's own residual-corrected bound rather than a
    sample sweep.

    `extra_roots` seeds the start tree with configurations that are certified
    by construction and known to be well connected — in practice the arm's
    park pose, which is the one reconfiguration the ladder already trusts.  A
    root that does not certify against `q0` is simply dropped.
    """
    import time
    # RESOLVED HERE, NOT IN THE SIGNATURE — the lesson `_shortcut` already
    # carries, and this function did not.  A default argument is bound at `def`
    # time, so `transit.MAX_NODES = 500` from a caller never reached the search:
    # every budget `scripts/feasible_workspace.py` and `scripts/csail_allocate.py
    # --transit-budget` set was inert, and the runs they logged were the module
    # defaults wearing the caller's numbers.  An explicit argument still wins;
    # `None` now means "whatever the module says at the moment of the call",
    # which is what a knob has to mean for a rebinding to be a knob at all.
    step = STEP if step is None else float(step)
    max_nodes = MAX_NODES if max_nodes is None else int(max_nodes)
    time_budget = TIME_BUDGET if time_budget is None else float(time_budget)
    attempts = ATTEMPTS if attempts is None else int(attempts)
    t0 = time.perf_counter()
    _STATS["calls"] += 1
    q0 = np.asarray(q0, float).reshape(7)
    q1 = np.asarray(q1, float).reshape(7)
    if gate is None:
        gate = Gate(spec, pen_ext, h_inv, boxes, chain_floor, tip_floor,
                    static_floor, self_floor, probe, probe_margin)
    lo, hi = FR3_MIN + LIMIT_MARGIN, FR3_MAX - LIMIT_MARGIN
    sig = scene_signature(spec, gate.boxes, gate.pen, gate.h_inv,
                          (gate.chain_floor, gate.tip_floor, gate.static_floor,
                           gate.self_floor, gate.probe_margin))
    if not seeds:
        seeds = [seed_for(sig, q0, q1, a) for a in range(int(attempts))]

    out = None
    for k, sd in enumerate(seeds):
        rng = np.random.default_rng(int(sd) & ((1 << 63) - 1))
        raw = _rrt_connect(gate, q0, q1, rng, lo, hi, step, max_nodes,
                           time.perf_counter() + time_budget,
                           extra_roots if k == 0 else ())
        if raw is not None:
            _STATS["shortcut_from"] += len(raw)
            path = _shortcut(gate, raw, rng, step) if shortcut else raw
            _STATS["shortcut_to"] += len(path)
            out = [np.asarray(q, float).reshape(7) for q in path[1:-1]]
            break
    _STATS["seconds"] += time.perf_counter() - t0
    _STATS["solved" if out is not None else "failed"] += 1
    return out


def _rrt_connect(gate, q0, q1, rng, lo, hi, step, max_nodes, deadline,
                 extra_roots=()):
    """The raw tree search. -> [q0, ..., q1] | None (endpoints included)."""
    import time
    if gate.line_prefix(q0, q1, step)[1]:
        return [q0, q1]
    ta, tb = _Tree(q0, max_nodes), _Tree(q1, max_nodes)
    for r in extra_roots:
        r = np.asarray(r, float).reshape(7)
        if not gate.pose_ok(r[None])[0]:
            continue
        q, hit = gate.line_prefix(q0, r, step)
        if hit:
            ta.add(r, 0)
        elif dist(q, q0) > 1e-9:
            ta.add(q, 0)
    sample = _Sampler(rng, gate, lo, hi)
    a, b = ta, tb
    while ta.n + tb.n < 2 * max_nodes:
        if time.perf_counter() >= deadline:
            _STATS["deadline"] += 1
            break
        q_rand = b.Q[0] if rng.random() < GOAL_BIAS else sample.next(deadline)
        if q_rand is None:
            _STATS["deadline"] += 1
            break
        i = a.nearest(q_rand)
        q_new, _ = gate.line_prefix(a.Q[i], steer(a.Q[i], q_rand, step), step)
        if dist(q_new, a.Q[i]) <= 1e-9:
            a, b = b, a
            continue
        ia = a.add(q_new, i)
        if ia < 0:
            break
        j = b.nearest(q_new)
        q_hit, done = gate.line_prefix(b.Q[j], q_new, step)
        jb = j
        if dist(q_hit, b.Q[j]) > 1e-9:
            jb = b.add(q_hit, j)
            if jb < 0:
                break
        if done:
            _STATS["nodes"] += ta.n + tb.n
            pa, pb = a.path_to(ia), b.path_to(jb)
            path = pa + pb[::-1] if a is ta else pb + pa[::-1]
            return _dedupe(path)
        a, b = b, a
    _STATS["nodes"] += ta.n + tb.n
    return None


def _dedupe(path):
    out = [np.asarray(path[0], float).reshape(7)]
    for q in path[1:]:
        q = np.asarray(q, float).reshape(7)
        if dist(out[-1], q) > 1e-9:
            out.append(q)
    return out


def _shortcut(gate, path, rng, step=STEP, rounds=None, seconds=None):
    """Random shortcutting, and then a greedy pass. -> [q] (endpoints kept).

    THE TOUR PAYS FOR EVERY VIA IN SECONDS.  `writing._beat` prices a routed
    transit hop by hop at `_dq_time`, and `sequence.cost_matrix` prices the
    tour on exactly that number, so a raw RRT path with thirty jagged nodes is
    a real cost to the drawing and not a cosmetic one.  Shortcutting is
    therefore not a nicety here; measured on this rig it takes a solved
    transit from twenty-odd nodes to three or four and its cost from four
    seconds to under one.

    Every accepted shortcut is CERTIFIED at the same gate as the edges it
    replaces, so smoothing cannot smuggle in a leg the search would have
    refused.
    """
    import time
    # RESOLVED HERE, NOT IN THE SIGNATURE.  A default argument is bound at
    # `def` time and no rebinding of the module constant reaches it — the
    # lesson `fleet._SHEET_BINDERS` already carries — and a caller with a
    # budget (`scripts/feasible_workspace.py`) has to be able to turn these
    # down.
    rounds = SHORTCUT_ROUNDS if rounds is None else int(rounds)
    seconds = SHORTCUT_TIME if seconds is None else float(seconds)
    t0 = time.perf_counter()
    P = [np.asarray(q, float).reshape(7) for q in path]
    for _ in range(int(rounds)):
        if len(P) <= 2 or time.perf_counter() - t0 >= seconds:
            break
        i = int(rng.integers(0, len(P) - 2))
        j = int(rng.integers(i + 2, len(P)))
        if gate.line_prefix(P[i], P[j], step)[1]:
            P = P[:i + 1] + P[j:]
    # ...and then the greedy sweep, because random shortcutting leaves a tail
    # of adjacent nodes that a single pass removes for free.
    i = 0
    while i < len(P) - 2 and time.perf_counter() - t0 < 2 * seconds:
        j = len(P) - 1
        while j > i + 1:
            if gate.line_prefix(P[i], P[j], step)[1]:
                P = P[:i + 1] + P[j:]
                break
            j -= 1
        i += 1
    return _dedupe(P)
