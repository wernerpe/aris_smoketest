"""Tracer + allocator regressions (fast: no IK, no image files).

Run: python3 tests/test_csail.py     (or pytest tests/)
"""
import sys
import time
from pathlib import Path

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from aris_sixarm import allocate, coordination, scene_check, trace   # noqa: E402
from aris_sixarm.allocate import Interval            # noqa: E402
from aris_sixarm.fleet import (FLEET as FLEET_ACTIVE,        # noqa: E402
                               FLEET_SIXARM as FLEET, SHEET_SIXARM as SHEET)


def ArmPath6(arm_id, *a, **kw):
    """coordination.ArmPath pinned to the LEGACY fleet these tests were
    written on (the active registry is the 3-arm final rig)."""
    return coordination.ArmPath(arm_id, *a, spec=FLEET[arm_id], **kw)
from aris_sixarm.frames import FR3_MAX, FR3_MIN   # noqa: E402
from aris_sixarm.validate import validate_plan       # noqa: E402


def _bar(mask, r0, r1, c0, c1):
    mask[r0:r1, c0:c1] = True
    return mask


def _dense(p, ds=0.5):
    p = np.asarray(p, float)
    t = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    s = np.arange(0, t[-1], ds)
    return np.column_stack([np.interp(s, t, p[:, 0]), np.interp(s, t, p[:, 1])])


def _plus(n=81, w=5):
    m = np.zeros((n, n), bool)
    mid = n // 2
    _bar(m, mid - w // 2, mid + w // 2 + 1, 6, n - 6)
    _bar(m, 6, n - 6, mid - w // 2, mid + w // 2 + 1)
    return m


def _ex(n=81, w=5):
    m = np.zeros((n, n), bool)
    for i in range(6, n - 6):
        for d in range(-(w // 2), w // 2 + 1):
            m[i, min(max(i + d, 0), n - 1)] = True
            m[i, min(max(n - 1 - i + d, 0), n - 1)] = True
    return m


def test_crossing_stays_two_strokes():
    """The whole point of the junction pairing: an X leaves as 2 strokes, not 4.

    A skeleton graph cut at every junction would return four ~35 px stubs from
    each of these; the straightest-continuation pairing has to carry both
    branches through the crossing.
    """
    for name, m in (("plus", _plus()), ("ex", _ex())):
        strokes, skel = trace.trace_lines(m, bridge=False)
        assert len(strokes) == 2, f"{name}: {len(strokes)} strokes, want 2"
        for s in strokes:
            assert trace.plen(s) > 0.8 * (m.shape[0] - 12), \
                f"{name}: stroke only {trace.plen(s):.0f} px long"
        # and the two strokes really do cross (RDP leaves the crossing point
        # implicit — it is collinear — so compare the densified paths)
        a, b = (_dense(s) for s in strokes)
        d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
        assert d.min() < 2.0, f"{name}: the two strokes never meet ({d.min():.1f} px)"


def test_thinning_and_thick_split():
    """Zhang-Suen returns a 1 px skeleton; solid glyphs split off from line art.

    `split_thick` judges whole CONNECTED COMPONENTS, which is what the logo
    needs (the wordmark does not touch the buildings) and is why the two
    shapes here are disjoint.
    """
    m = np.zeros((60, 130), bool)
    _bar(m, 28, 33, 5, 60)               # a 5 px line
    _bar(m, 5, 40, 80, 115)              # a 35 px solid glyph, not touching it
    sk = trace.thin(m)
    assert sk.sum() < 0.2 * m.sum(), "skeleton is not thin"
    assert sk[28:33, 20:40].sum() == 20, "the line should thin to one row"
    assert trace.label_components(m)[1] == 2
    thick, thin_ = trace.split_thick(m, n_erode=4)
    assert thick[20, 90] and not thick[30, 10], "the glyph is not the thick part"
    assert thin_[30, 10] and not thin_[20, 90], "the line is not the thin part"
    assert thick.sum() + thin_.sum() == m.sum(), "the split must partition"


def test_rdp_and_sheet_mapping():
    """RDP keeps corners and drops collinear points; the sheet map keeps aspect."""
    line = np.column_stack([np.arange(0, 50, 1.0), np.zeros(50)])
    line[25:, 1] = np.arange(25) * 1.0
    assert len(trace.rdp(line, 1.2)) == 3
    px = [dict(pts=np.array([[0.0, 0.0], [200.0, 0.0], [200.0, 100.0],
                             [0.0, 100.0], [0.0, 0.0]]), color="grey",
               kind="outline")]
    out, info = trace.to_sheet(px, SHEET, margin=0.06)
    p = out[0]["pts"]
    assert abs((p[:, 0].max() - p[:, 0].min()) /
               (p[:, 1].max() - p[:, 1].min()) - 2.0) < 1e-9   # aspect kept
    assert p[:, 0].min() > 0.059 and p[:, 0].max() < SHEET[0] - 0.059
    assert p[:, 1].min() > 0.059 and p[:, 1].max() < SHEET[1] - 0.059
    assert abs(p[:, 0].mean() * 0 + (p[:, 0].max() + p[:, 0].min()) / 2
               - SHEET[0] / 2) < 1e-9                          # centred


def test_greedy_cover_is_minimal_and_finds_gaps():
    """Fewest pieces, and everything nobody covers comes back as a gap."""
    ivs = [Interval(0.0, 0.4, 13), Interval(0.3, 0.5, 17), Interval(0.35, 0.8, 31),
           Interval(0.9, 1.0, 97)]
    chosen, gaps = allocate.greedy_cover(ivs)
    assert [(v.s0, v.s1) for v in chosen] == [(0.0, 0.4), (0.35, 0.8), (0.9, 1.0)]
    assert gaps == [(0.8, 0.9)]
    # a single interval that spans everything must win outright
    chosen, gaps = allocate.greedy_cover(ivs + [Interval(0.0, 1.0, 97)])
    assert len(chosen) == 1 and not gaps
    # ties go to the arm with less work so far
    tie = [Interval(0.0, 1.0, 13), Interval(0.0, 1.0, 31)]
    assert allocate.greedy_cover(tie, load={13: 5.0, 31: 1.0})[0][0].arm == 31
    assert allocate.uncovered([Interval(0.2, 0.6, 13)]) == [(0.0, 0.2), (0.6, 1.0)]


def test_handoff_cut_lands_mid_overlap():
    """Seams go to the middle of the overlap, with both sides grown to meet."""
    chosen = [Interval(0.0, 0.6, 13), Interval(0.4, 1.0, 31)]
    spans = allocate.place_cuts(chosen, L=1.0, overlap=0.01)
    assert abs(spans[0]["s1"] - 0.51) < 1e-9      # mid-overlap 0.5, +1 cm
    assert abs(spans[1]["s0"] - 0.49) < 1e-9      # mid-overlap 0.5, -1 cm
    assert spans[0]["s1"] > spans[1]["s0"], "the two segments must overlap"
    for sp, iv in zip(spans, chosen):             # never outside what was certified
        assert iv.s0 - 1e-12 <= sp["s0"] and sp["s1"] <= iv.s1 + 1e-12


def test_colour_partitions():
    """14 non-trivial pen assignments for four arms, 16 with the degenerate two."""
    arms = [13, 17, 31, 97]
    p = allocate.partitions(arms)
    assert len(p) == 14 and len(allocate.partitions(arms, nontrivial=False)) == 16
    assert all(set(d) == set(arms) for d in p)
    assert all(len(set(d.values())) == 2 for d in p)
    assert len({tuple(sorted(d.items())) for d in p}) == 14      # all distinct
    assert len(allocate.partitions([13, 17])) == 2


def test_best_partition_follows_the_drops():
    """The enumeration must pick the pen split that leaves least paper empty."""
    strokes = [dict(id=0, color="grey", kind="outline",
                    pts=np.array([[0.0, 0.0], [1.0, 0.0]])),
               dict(id=1, color="orange", kind="outline",
                    pts=np.array([[0.0, 1.0], [3.0, 1.0]]))]
    ivmap = {0: [Interval(0.0, 1.0, 13)], 1: [Interval(0.0, 1.0, 31)]}
    colors, cover, table = allocate.best_partition(strokes, ivmap,
                                                   [13, 17, 31, 97])
    assert colors[13] == "grey" and colors[31] == "orange"
    assert cover["dropped_len"] < 1e-9
    assert table[0]["dropped"] <= table[-1]["dropped"]


def test_active_override_does_not_touch_the_registry():
    """A what-if fleet is an ARGUMENT, never an edit to fleet.FLEET.

    The registry's `active` flags are the record of which arms are up today;
    an "all six arms" study that flipped them would silently change what every
    other caller in the process means by "the fleet".
    """
    before = {a: s.active for a, s in FLEET.items()}
    # the ACTIVE registry is the FINAL RIG (3 arms, all up)
    assert allocate.active_arms() == allocate.ACTIVE == [13, 31, 2]
    assert allocate.active_arms("all") == [13, 31, 2]
    # the LEGACY fleet is an argument away, bit for bit
    assert allocate.active_arms(fleet=FLEET) == [13, 17, 31, 97]
    assert allocate.active_arms("all", FLEET) == [13, 17, 31, 2, 71, 97]
    assert allocate.active_arms([97, 13], FLEET) == [13, 97]          # order normalised
    assert allocate.active_arms({2: True}, FLEET) == [13, 17, 31, 2, 97]
    assert allocate.active_arms({13: False, 71: True}, FLEET) == [17, 31, 71, 97]
    assert {a: s.active for a, s in FLEET.items()} == before, "the registry moved"
    assert {a: s.active for a, s in FLEET_ACTIVE.items()} == \
        {13: True, 31: True, 2: True}, "the final registry moved"

    for bad in ("some", [13, 999], {42: True}):
        try:
            allocate.active_arms(bad, FLEET)
        except ValueError:
            continue
        raise AssertionError(f"active_override={bad!r} should have been rejected")

    # the entry point refuses the ambiguous call rather than picking a winner
    try:
        allocate.allocate([], arms=[13], active_override="all", fleet=FLEET)
    except ValueError:
        pass
    else:
        raise AssertionError("arms= and active_override= together must raise")


def test_active_override_reaches_the_partition_enumeration():
    """Six arms means 62 pen partitions, and the allocation must use all six."""
    assert len(allocate.partitions(allocate.active_arms("all", FLEET))) == 62
    strokes = [dict(id=0, color="grey", kind="outline",
                    pts=np.array([[0.0, 0.0], [1.0, 0.0]])),
               dict(id=1, color="orange", kind="outline",
                    pts=np.array([[0.0, 1.0], [1.0, 1.0]]))]
    ivmap = {0: [Interval(0.0, 1.0, 71)], 1: [Interval(0.0, 1.0, 2)]}
    arms = allocate.active_arms("all", FLEET)
    colors, cover, table = allocate.best_partition(strokes, ivmap, arms)
    # only the two parked arms can cover anything, so the partition has to give
    # them the two different pens — which the four-arm fleet cannot do at all
    assert colors[71] == "grey" and colors[2] == "orange"
    assert cover["dropped_len"] < 1e-9
    assert allocate.best_partition(strokes, ivmap,
                                   allocate.active_arms(fleet=FLEET)
                                   )[1]["dropped_len"] > 1.9


def test_capsule_distance_agrees_with_the_independent_one():
    """`coordination` and `scene_check` derive segment distance differently.

    That is the entire value of the second implementation, so it is worth a test
    that they cannot silently drift apart — including the degenerate cases
    (parallel, touching, one segment a point) each derivation handles in its own
    branch.
    """
    rng = np.random.default_rng(7)
    P = rng.normal(size=(400, 4, 3))
    P[:50, 1] = P[:50, 0]                       # first segment degenerate
    P[50:100, 3] = P[50:100, 2]                 # second segment degenerate
    P[100:150, 2:] = P[100:150, :2] + np.array([0.3, 0.0, 0.0])   # parallel
    a = coordination.seg_seg_dist(P[:, 0], P[:, 1], P[:, 2], P[:, 3])
    b = scene_check.segment_distance(P[:, 0], P[:, 1], P[:, 2], P[:, 3])
    assert np.max(np.abs(a - b)) < 1e-9, f"max disagreement {np.max(np.abs(a - b)):.2e}"
    # and both are right on a case with a known answer: two skew unit segments
    # 0.5 apart along z
    d = coordination.seg_seg_dist(np.array([0.0, 0, 0]), np.array([1.0, 0, 0]),
                                  np.array([0.5, -1, 0.5]), np.array([0.5, 1, 0.5]))
    assert abs(float(d) - 0.5) < 1e-12


def test_a_parked_arm_is_an_obstacle_for_everyone():
    """An arm that draws nothing still stands in the workspace.

    Priority used to be "most motion first", and an arm is only ever checked
    against the arms scheduled BEFORE it — so an idle arm, having zero motion,
    sorted last and appeared in nobody's collision image.  Harmless while the
    idle arms are at the edge of the rig; wrong the moment a two-pass run parks
    a middle arm through a whole phase.  Parked arms must therefore be
    scheduled first and appear in every moving arm's image.
    """
    q0 = np.asarray(FLEET[31].q_seed, float)
    moving = np.linspace(q0, q0 + 0.25, 40)
    paths = {31: ArmPath6(31, moving, 0.02),
             97: ArmPath6(97, q0[None, :], 0.02),
             71: ArmPath6(71, np.linspace(q0, q0 + 0.1, 20), 0.02)}
    res = coordination.coordinate(paths, verbose=False)
    assert not paths[97].moves and paths[31].moves
    assert res["order"][0] == 97, \
        f"parked arm should be scheduled first, order is {res['order']}"
    for mover in (31, 71):
        assert (mover, 97) in res["free"], \
            f"arm {mover} was never checked against the parked arm 97"
    # the parked arm's own schedule is the constant it has to be
    assert np.all(res["progress"][97] == 0)


def test_capsule_length_follows_the_pen():
    """The conductor's last capsule IS the pen, so its length has to be the
    arm's real one.

    An arm holding 300 mm reaches 190 mm further along the tool axis than the
    same arm holding 110 mm, and every clearance the conductor computes is
    about that endpoint.  Building the collision model at a default length
    while the planner used another is the one way a per-arm pen can pass every
    plan-level check and still schedule a collision.
    """
    q = np.repeat(np.asarray(FLEET[31].q_seed, float)[None, :], 3, axis=0)
    short = ArmPath6(31, q, 0.01, pen_ext=0.110)
    long_ = ArmPath6(31, q, 0.01, pen_ext=0.300)
    # capsule index -1 is (chain point 8 -> pen tip); its far end is B[-1]
    tip_s, tip_l = short.B[0][-1], long_.B[0][-1]
    assert abs(float(np.linalg.norm(tip_l - tip_s)) - 0.190) < 1e-6, \
        f"pen tip moved {float(np.linalg.norm(tip_l - tip_s)):.4f} m, want 0.190"
    # the rest of the chain is untouched: only the pen changed
    assert np.allclose(short.A, long_.A) and np.allclose(short.B[:, :-1],
                                                         long_.B[:, :-1])
    # and `arm_paths` hands each arm its own, defaulting the ones it is not told
    # (arm 97 sits at a different base, so it is compared against its OWN
    # default-pen path, not against arm 31's)
    paths = coordination.arm_paths({31: q, 97: q}, 0.01, pens={31: 0.300}, fleet=FLEET)
    assert np.allclose(paths[31].B[0][-1], tip_l)
    assert np.allclose(paths[97].B[0][-1],
                       ArmPath6(97, q, 0.01, pen_ext=0.110).B[0][-1])
    # scene_check reads the same mapping, scalar or per-arm
    assert scene_check.pen_len(0.2, 31) == 0.2
    assert scene_check.pen_len({31: 0.300}, 31) == 0.300
    assert scene_check.pen_len({31: 0.300}, 97) == 0.110


def test_allocation_plans_and_validates_with_the_arm_s_own_pen():
    """A segment certified with a 300 mm pen is not a segment for a 110 mm one.

    `allocate` is asked for a fleet in which arm 31 holds 300 mm; the shipped
    plan has to carry that pen all the way to the independent validator, so the
    same joints re-validated against the DEFAULT pen must fail — if they passed,
    `pen_ext` would not actually be reaching `validate_plan` and the certificate
    would be about the wrong tool.
    """
    pts = np.column_stack([np.linspace(1.20, 1.34, 15), np.full(15, 1.12)])
    strokes = [dict(pts=pts, color="grey", kind="outline", id=0)]
    res = allocate.allocate(strokes, arms=[31], pens={31: 0.300}, fleet=FLEET,
                            colors={31: "grey"}, verbose=False)
    assert res["pens"][31] == 0.300
    segs = res["programs"][31]
    assert segs, "arm 31 should reach this stroke"
    plan = segs[0]["plan"]
    right = validate_plan(np.asarray(plan["pts"], float), FLEET[31],
                          np.asarray(plan["qs"], float), pen_ext=0.300)
    wrong = validate_plan(np.asarray(plan["pts"], float), FLEET[31],
                          np.asarray(plan["qs"], float), pen_ext=0.110)
    assert right["ok"], "the plan does not validate at the pen it was planned with"
    assert not wrong["ok"], "pen_ext is not reaching the validator"
    # and the per-arm option dict is what carries it
    assert allocate.pen_opts(None, {31: 0.300}, 31)["pen_ext"] == 0.300
    assert "pen_ext" not in allocate.pen_opts(None, {31: 0.300}, 97)
    assert allocate.pen_of({31: 0.300}, 97) == 0.110


def test_gap_tolerance_is_the_one_leftover_uses():
    """Covering with a 25 mm tolerance and reporting with a 2 mm one is how a
    run claims 100 % with a centimetre of bare paper in it.

    `cover_all`'s gaps and `leftover`'s must be the same question asked twice,
    so a cover that leaves a 10 mm hole has to SAY it leaves a 10 mm hole.
    """
    pts = np.column_stack([np.linspace(0.0, 1.0, 101), np.zeros(101)])
    strokes = [dict(pts=pts, color="grey", kind="outline", id=0)]
    ivmap = {0: [Interval(0.0, 0.40, 13), Interval(0.41, 1.0, 17)]}   # 10 mm hole
    colors = {13: "grey", 17: "grey"}
    cov = allocate.cover_all(strokes, ivmap, colors)
    assert abs(cov["dropped_len"] - 0.01) < 1e-9, \
        f"a 10 mm hole was reported as {cov['dropped_len']:.4f} m"
    # the old conflation: tolerate MIN_SEG and the same hole vanishes
    loose = allocate.cover_all(strokes, ivmap, colors,
                              gap_tol=allocate.MIN_SEG_M)
    assert loose["dropped_len"] == 0.0
    # and `leftover`, which works off the SHIPPED programmes, agrees with the
    # strict one rather than the loose one
    progs = {13: [dict(stroke_id=0, s_range=(0.0, 0.40))],
             17: [dict(stroke_id=0, s_range=(0.41, 1.0))]}
    assert abs(sum(d["length"] for d in allocate.leftover(strokes, progs))
               - 0.01) < 1e-9


def test_pause_schedule_is_monotone_and_avoids_the_blocked_cells():
    """The DP must wait rather than walk through an unsafe cell, and never
    reverse.  A hand-built image: arm B is a wall across the middle of A's
    progress until B has gone past its own halfway point."""
    nA, nB = 40, 30
    free = np.ones((nA - 1, nB - 1), bool)
    free[10:20, :15] = False                   # A may not be at 10..19 while B < 15
    prog_b = np.minimum(np.arange(200), nB - 1)
    P, m_end = coordination._dp({"B": free}, {"B": prog_b}, nA, 200)
    assert P is not None, "a feasible wait-then-go schedule was not found"
    d = np.diff(P)
    assert np.all(d >= 0) and np.all(d <= 1), "progress must advance by 0 or 1"
    assert P[0] == 0 and P[m_end] == nA - 1
    for m in range(m_end):
        assert free[min(P[m], nA - 2), min(prog_b[m], nB - 2)], \
            f"scheduled into a blocked cell at step {m}"
    assert m_end > nA - 1, "the blockage should have cost at least one pause"


class _StubPath:
    """The three attributes `_search_priority` and `_schedule` read off a path."""

    def __init__(self, n, dt):
        self.n, self.dt, self.moves, self.motion = int(n), float(dt), True, float(n)


def _random_images(arms, ns, rng, block):
    """Synthetic collision images, TRANSPOSE-CONSISTENT the way real ones are.

    `free_cells(a, b)` and `free_cells(b, a)` are the same predicate read the
    two ways round, so a fabricated pair that is not a transpose of itself is
    not a rig any geometry could produce.  Both ends of every path are kept
    clear so that every order is feasible and the brute force below has
    something to compare against at all.
    """
    F = {}
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            M = rng.random((ns[a] - 1, ns[b] - 1)) > block
            M[:2, :] = M[-2:, :] = True
            M[:, :2] = M[:, -2:] = True
            F[(a, b)], F[(b, a)] = M, M.T.copy()
    return F


def test_priority_search_returns_the_minimum_makespan_order():
    """The conductor's only free variable, searched instead of guessed.

    Priority decides both feasibility and duration, and "busiest first, promote
    whoever deadlocks" stops at the first order that works — which on the CSAIL
    logo's phase 1 was the WORST of the three feasible ones (63.5 s against
    48.3 s).  Two things are checked here, on synthetic collision images so it
    is exact and fast: the search returns the true minimum over every
    permutation (brute-forced with the same `_schedule` the old path used), and
    it never returns worse than the busiest-first order the heuristic starts
    from.  A third case does the same on real fleet geometry.
    """
    import itertools
    rng = np.random.default_rng(20260819)
    strictly_better = 0
    for trial in range(12):
        arms = [11, 22, 33]
        ns = {a: int(rng.integers(14, 34)) for a in arms}
        dt = 0.05
        paths = {a: _StubPath(ns[a], dt) for a in arms}
        F = _random_images(arms, ns, rng, block=rng.uniform(0.15, 0.5))
        M = int(np.ceil(max(ns.values()) * 3.0)) + 64

        def cells(a, b, F=F):
            return F[(a, b)]

        brute = {}
        for order in itertools.permutations(arms):
            res, _ = coordination._schedule(paths, list(order), [], cells, {},
                                            M, dt)
            if res is not None:
                brute[order] = max(res[1].values())
        assert brute, f"trial {trial}: no order is feasible, nothing to compare"
        best, stats = coordination._search_priority(paths, list(arms), [], cells,
                                                    {}, M, dt)
        assert best is not None, f"trial {trial}: the search found nothing"
        want = min(brute.values())
        assert abs(best["makespan"] - want) < 1e-9, \
            (f"trial {trial}: search says {best['makespan']:.3f} s, the "
             f"minimum over {len(brute)} feasible orders is {want:.3f} s")
        assert tuple(best["order"]) in brute
        # ...and it is never worse than what busiest-first would have taken
        heur = tuple(sorted(arms, key=lambda a: (-paths[a].motion, a)))
        if heur in brute:
            assert best["makespan"] <= brute[heur] + 1e-9
            strictly_better += brute[heur] > want + 1e-9
        assert stats["n_dp"] <= sum(len(list(itertools.permutations(arms, k)))
                                    for k in range(1, len(arms) + 1))
    assert strictly_better >= 1, \
        "no trial where the busiest-first order was beaten: the test is vacuous"

    # real geometry, real ArmPath, both code paths through `coordinate`
    q0 = np.asarray(FLEET[31].q_seed, float)
    paths = {31: ArmPath6(31, np.linspace(q0, q0 + 0.25, 40), 0.02),
             97: ArmPath6(97, q0[None, :], 0.02),
             71: ArmPath6(71, np.linspace(q0, q0 + 0.1, 20), 0.02)}
    searched = coordination.coordinate(paths, verbose=False)
    guessed = coordination.coordinate(paths, priority_search=False, verbose=False)
    assert searched["duration"] <= guessed["duration"] + 1e-9, \
        (f"the search made it worse: {searched['duration']:.3f} s vs "
         f"{guessed['duration']:.3f} s")
    assert searched["search"]["n_permutations"] == 2 and guessed["search"] is None
    # the deadlock-recovery path is still there and still reachable
    assert coordination.coordinate(paths, priority_search=False,
                                   retry_orders=True, verbose=False)["attempts"] >= 1


def test_a_refused_priority_search_reports_what_it_searched():
    """The refusal may not say "0 priority orders were tried" after trying all.

    `attempts` is `stats["n_orders"]` — COMPLETE orders costed — and a search
    that fails completes none of them, so the refusal reported zero for a walk
    that had just been through every permutation.  It read as "the DFS cut
    every prefix", which sent a whole investigation of the all-ceiling rig
    looking for a pruning bug that was not there.  The bound only ever prunes
    against a complete order already in hand, so a FAILED search has pruned
    nothing and `n_dp` is exactly how many prefixes it paid for.
    """
    q0 = np.asarray(FLEET[31].q_seed, float)
    paths = {31: ArmPath6(31, np.linspace(q0, q0 + 0.25, 40), 0.02),
             71: ArmPath6(71, np.linspace(q0, q0 + 0.10, 20), 0.02)}
    stats = dict(n_dp=57, n_orders=0, n_pruned=0, fail_depth=0, failed=71)
    old = coordination._search_priority
    coordination._search_priority = lambda *a, **k: (None, stats)
    try:
        with pytest.raises(RuntimeError) as got:
            coordination.coordinate(paths, verbose=False)
    finally:
        coordination._search_priority = old
    msg = str(got.value)
    assert "all 2 priority orders of the 2 moving arms were searched" in msg, msg
    assert "57 prefix DP solves" in msg, msg
    assert "0 priority order" not in msg, msg


def _busy_scene(n=130):
    """Three moving arms and one parked, close enough to interfere."""
    out = {}
    for arm, lo, hi in ((31, 0.0, 0.30), (71, 0.0, 0.22), (2, 0.0, 0.26)):
        q0 = np.asarray(FLEET[arm].q_seed, float)
        out[arm] = ArmPath6(arm, np.linspace(q0, q0 + hi, n), 0.02)
    q0 = np.asarray(FLEET[97].q_seed, float)
    out[97] = ArmPath6(97, q0[None, :], 0.02)
    return out


def _brute_clearance(pi, pj, cap=coordination.BROAD_CAP):
    """Every one of the 49 capsule pairs, no broad phase at all.

    The independent reference for the two broad phases: whatever they reject,
    the answer has to be the answer this returns.
    """
    d = coordination.seg_seg_dist(pi.A[:, :, None, None, :],
                                  pi.B[:, :, None, None, :],
                                  pj.A[None, None, :, :, :],
                                  pj.B[None, None, :, :, :])
    d = np.transpose(d, (0, 2, 1, 3)) - (pi.r[:, None] + pj.r[None, :])
    return np.minimum(d.reshape(pi.n, pj.n, -1).min(2), cap).astype(np.float32)


def test_the_broad_phase_drops_only_what_could_not_have_mattered():
    """Rejecting a capsule pair must not change one cell of the matrix.

    `clearance_matrix` clips at `cap`, so a capsule pair whose boxes are `cap`
    apart cannot lower any cell below what is already recorded — that is the
    whole argument, and it holds however the paths are tiled, so the same
    matrix must come out at every tile size and from a reference that does no
    rejecting whatever.  It is worth a test because the filter is the
    difference between 49 exact distances per cell and about 13.
    """
    paths = _busy_scene(60)
    for a, b in ((31, 71), (2, 31), (31, 97)):
        pi, pj = paths[a], paths[b]
        want = _brute_clearance(pi, pj)
        for tile in (8, 32, 4096):
            got = coordination.clearance_matrix(pi, pj, tile=tile)
            assert np.array_equal(got, want), \
                (f"({a},{b}) at tile {tile}: {np.count_nonzero(got != want)} "
                 "cells differ from the unfiltered reference")
        # and a block of rows is exactly those rows of the whole image
        whole = coordination.free_cells(pi, pj, 0.08)
        r0, r1 = 7, min(23, pi.n - 1)
        assert np.array_equal(coordination.free_cells(pi, pj, 0.08,
                                                      rows=(r0, r1)),
                              whole[r0:r1]), f"({a},{b}): row block drifted"
        # the picture is the same read the other way round, to the bit
        assert np.array_equal(coordination.free_cells(pj, pi, 0.08), whole.T), \
            f"({a},{b}): the image is not its own transpose"


def _plain_dp(free_ab, prog_hi, n, horizon, deadline=None, rest=True):
    """`_dp` with the whole reachable table built up front, and no blocking.

    The straightforward reading of the same recurrence, kept as an independent
    reference: `_dp` builds its free-cell table a block of time steps at a time
    and stops when the arm arrives, which is worth a third of a solve and is
    exactly the kind of change that can be right on every case a fixture
    happens to contain.
    """
    M = horizon
    D = M if deadline is None else int(min(max(deadline, 0) + 1, M))
    rest_row = np.ones(M, bool)
    if rest:
        for b, F in free_ab.items():
            rest_row &= F[n - 2, np.clip(prog_hi[b], 0, F.shape[1] - 1)]
    rest_ok = np.logical_and.accumulate(rest_row[::-1])[::-1][:D]
    ok = np.ones((n - 1, D), bool)
    for b, F in free_ab.items():
        ok &= F[:, np.clip(prog_hi[b][:D], 0, F.shape[1] - 1)]
    okf = np.vstack([ok, ok[-1:]])
    reach = np.zeros((n, D), bool)
    reach[0, 0] = True
    for m in range(1, D):
        av = reach[:, m - 1] & okf[:, m - 1]
        col = av.copy()
        col[1:] |= av[:-1]
        reach[:, m] = col
        if reach[n - 1, m] and rest_ok[m]:
            break
    hits = np.flatnonzero(reach[n - 1] & rest_ok)
    if not len(hits):
        return None, None
    m_end = int(hits[0])
    prog = np.full(M, n - 1, int)
    p, m = n - 1, m_end
    while m > 0:
        if p > 0 and reach[p - 1, m - 1] and okf[p - 1, m - 1]:
            p -= 1
        elif reach[p, m - 1] and okf[p, m - 1]:
            pass
        else:
            raise RuntimeError("backtrack fell off the reachable set")
        m -= 1
        prog[m] = p
    return prog, m_end


def test_the_schedule_dp_builds_its_table_lazily_and_gets_the_same_answer():
    """Same arrival, same progress, same refusals — deadline and rest included.

    The randomised instances are deliberately mixed: some arms arrive early
    (where the laziness bites), some are blocked outright (where it cannot),
    and the deadlines straddle the arrival so both the "beaten the bound" and
    "made it" branches are exercised.
    """
    rng = np.random.default_rng(20260825)
    seen = dict(arrived=0, refused=0, bounded=0)
    for _ in range(60):
        n = int(rng.integers(3, 40))
        nb = int(rng.integers(3, 40))
        M = int(rng.integers(n + 2, 4 * n + 30))
        blockers = {}
        for b in range(int(rng.integers(0, 4))):
            F = rng.random((n - 1, nb - 1)) > rng.uniform(0.05, 0.6)
            blockers[b] = F
        prog = {b: np.clip(np.cumsum(rng.integers(0, 2, M)), 0, nb - 2)
                for b in blockers}
        for deadline in (None, int(rng.integers(0, M + 5))):
            for rest in (True, False):
                got = coordination._dp(blockers, prog, n, M, deadline, rest)
                want = _plain_dp(blockers, prog, n, M, deadline, rest)
                assert (got[1] is None) == (want[1] is None), \
                    f"one refused and the other did not (n={n}, M={M})"
                if want[1] is None:
                    seen["refused"] += 1
                    continue
                seen["arrived"] += 1
                seen["bounded"] += deadline is not None
                assert got[1] == want[1], \
                    f"arrival {got[1]} vs {want[1]} (n={n}, M={M})"
                assert np.array_equal(got[0], want[0]), \
                    f"the progress traces differ (n={n}, M={M})"
    assert seen["arrived"] > 20 and seen["refused"] > 5 and seen["bounded"] > 10, \
        f"the instances did not cover both outcomes: {seen}"


def test_the_image_bag_is_keyed_on_what_the_image_is_made_of():
    """Built once per unordered pair, and kept for the pass that follows.

    The conductor runs three or four times over one fleet and each pass
    re-programmes one or two arms, so most pairs are bit-identical to the pass
    before — but every pass builds fresh `ArmPath` objects, so a memo keyed on
    identity would never hit.  What is pinned here: the same content hits
    whatever object carries it, a changed path misses, a changed margin misses,
    and the two directions of a pair cost ONE build between them.
    """
    coordination.clear_images()
    paths = _busy_scene(80)
    pairs = [(a, b) for a in (31, 71, 2) for b in paths if b != a]
    before = dict(coordination._IMAGE_STATS)
    bag = coordination.build_images(paths, pairs, 0.08, coordination.SWEEP_K,
                                    jobs=1)
    built = coordination._IMAGE_STATS["built"] - before["built"]
    # 3 moving x 3 others = 9 ordered pairs, but only 6 unordered ones
    assert len(bag) == 9 and built == 6, \
        f"{len(bag)} images read, {built} built; want 9 read and 6 built"
    for a, b in pairs:
        assert np.array_equal(bag[(a, b)], bag[(b, a)].T if (b, a) in bag
                              else bag[(a, b)])
        assert np.array_equal(bag[(a, b)],
                              coordination.free_cells(paths[a], paths[b], 0.08))

    # SAME CONTENT, DIFFERENT OBJECTS: every one of them is already in hand
    again = {a: ArmPath6(a, p.q, p.dt) for a, p in paths.items()}
    assert all(again[a].key == paths[a].key for a in paths)
    mark = coordination._IMAGE_STATS["built"]
    bag2 = coordination.build_images(again, pairs, 0.08, coordination.SWEEP_K,
                                     jobs=1)
    assert coordination._IMAGE_STATS["built"] == mark, \
        "a rebuilt path with identical content should not rebuild its images"
    for k in bag:
        assert np.array_equal(bag[k], bag2[k])

    # a path that MOVED, and a margin that changed, are both misses
    q = np.asarray(paths[71].q, float).copy()
    q[:, 0] += 0.05
    again[71] = ArmPath6(71, q, paths[71].dt)
    assert again[71].key != paths[71].key
    coordination.build_images(again, pairs, 0.08, coordination.SWEEP_K, jobs=1)
    assert coordination._IMAGE_STATS["built"] == mark + 3, \
        "only the three pairs that touch the arm that moved should rebuild"
    coordination.build_images(paths, pairs, 0.09, coordination.SWEEP_K, jobs=1)
    assert coordination._IMAGE_STATS["built"] == mark + 9, \
        "a different margin is a different image and must be rebuilt"
    coordination.clear_images()


def test_conducting_on_a_pool_picks_the_same_order_as_conducting_serially():
    """The priority search must not depend on how many cores built its images.

    The images are the only thing the pool touches — the search itself is
    serial and exact — but "the only thing" is worth pinning, because a row
    block assembled out of order or an image read as its own transpose would
    change the schedule that SHIPS while leaving every test about makespan
    happy.  So: same winner, same makespan, same total pause, same images, and
    the same again with everything already memoised.
    """
    want, saw = None, {}
    for tag, jobs, wipe in (("serial", 1, True), ("pooled", 4, True),
                            ("pooled again", 4, True), ("memoised", 4, False)):
        if wipe:
            coordination.clear_images()
        res = coordination.coordinate(_busy_scene(), jobs=jobs, verbose=False)
        saw[tag] = res["images"]
        got = (round(float(res["duration"]), 12), tuple(res["order"]),
               round(float(res["pause_total"]), 12),
               tuple(res["search"]["order"]), res["search"]["n_dp"],
               {k: int(v.sum()) for k, v in sorted(res["free"].items())})
        if want is None:
            want, first = got, tag
        assert got == want, f"{tag} disagrees with {first}"
    assert len(want[5]) == 9, "three moving arms should read nine images"
    # the runs above have to have been different runs, or this pins nothing
    assert saw["serial"]["jobs"] == 1 and saw["pooled"]["jobs"] == 4, \
        f"the pool was never used: {saw}"
    assert saw["serial"]["built"] == 6 and saw["memoised"]["built"] == 0, \
        f"the memoised run rebuilt images: {saw}"


def test_balancing_lowers_the_busiest_arm():
    """A cover optimal for pen-ups can be terrible for the clock.

    The synthetic instance is the pathological one the CSAIL orange pass is a
    mild version of: one arm holds every segment and a second arm could draw
    most of them.  What is asserted is that the pass lowers the maximum load,
    that it stays inside `options` (segment 0 is pinned and must not move), and
    that it lands within a stated factor of the true optimum — which is
    brute-forced here over all 2^n assignments, because a greedy for an NP-hard
    objective should be measured, not assumed.
    """
    import itertools
    size = [5.0, 4.0, 3.0, 3.0, 2.0, 2.0, 1.0, 1.0]
    options = [{0}] + [{0, 1} for _ in size[1:]]      # segment 0 is arm 0's alone
    owner = [0] * len(size)

    def load_fn(arm, idx):
        return sum(size[i] for i in idx)

    out, info = allocate.balance_loads(owner, options, load_fn)
    assert out[0] == 0, "a segment only one arm certifies was moved anyway"
    for i, (a, o) in enumerate(zip(out, options)):
        assert a in o, f"segment {i} was given to arm {a}, not in {sorted(o)}"
    best = min(max(load_fn(0, [i for i, a in enumerate(m) if a == 0]),
                   load_fn(1, [i for i, a in enumerate(m) if a == 1]))
               for m in itertools.product(*[sorted(o) for o in options]))
    assert info["max_before"] == sum(size)
    assert info["max_after"] < info["max_before"], "the pass did nothing"
    assert info["max_after"] <= best * 1.2 + 1e-9, \
        (f"greedy landed at {info['max_after']} against an optimum of {best}")
    assert info["rounds"] == len(info["moves"]) >= 1
    # idempotent: a second pass over its own answer finds nothing left to do
    again, info2 = allocate.balance_loads(out, options, load_fn)
    assert again == out and info2["rounds"] == 0


def load_score_bad(load_fn, owner):
    """How many arms hold a bag they cannot fly, under this assignment."""
    return allocate.load_score(
        {a: load_fn(a, tuple(i for i, x in enumerate(owner) if x == a))
         for a in set(owner) | {0, 1}})[0]


def test_balancing_escapes_a_bag_with_no_flyable_tour():
    """An arm that cannot FLY its bag must lose the span, not stall the pass.

    `sequence.cost_matrix` prices a pen-up `paper.route` refuses as `inf`, so a
    bag of spans an arm certifies AS INK can still have no order the arm can fly
    (`allocate.prune_unflyable`).  `arm_load` prices such a bag `UNFLYABLE`, and
    the question this pins is what the balancer then does with it.

    The trap is that `inf` is not an ordering.  Scored on the old
    `(max, sum of squares)` ruler every assignment containing an unflyable arm
    is `(inf, inf)` — all equal, none strictly better — and the balancer's
    strict `<` finds no move at all.  That is not hypothetical: the first
    two-pass allocation on `final6_opt` reported "0 moves and 0 splits taken,
    busiest arm inf s -> inf s" and shipped nothing.  Counting the unflyable
    arms FIRST is what gives that plateau a gradient.

    Unflyability is COMBINATIONAL, which is the shape the real defect has: arm 0
    can fly segment 0 and can fly segment 1, and cannot fly a bag holding both —
    exactly arm 97, which certifies its piece of stroke 26 and cannot cross to
    it from any of its other four orange spans.  So the escape exists (hand one
    of the two to arm 1) but it is the WRONG move by seconds alone: it takes the
    busiest FINITE load from 0.0 s to 11.0 s.  A balancer ranking on seconds
    first would sit on the plateau for ever, which is what this pins.
    """
    size = [1.0, 1.0, 10.0]
    options = [{0, 1}, {0, 1}, {0, 1}]
    owner = [0, 0, 0]

    def load_fn(arm, idx):
        if arm == 0 and 0 in idx and 1 in idx:
            return allocate.UNFLYABLE      # arm 0 cannot fly these two together
        return sum(size[i] for i in idx)

    assert load_score_bad(load_fn, owner) == 1, "the instance starts unflyable"
    out, info = allocate.balance_loads(owner, options, load_fn)
    assert load_score_bad(load_fn, out) == 0, \
        "the balancer stalled on the unflyable plateau instead of escaping it"
    assert np.isfinite(info["max_after"]) and info["rounds"] >= 1
    # it really did have to give up seconds to do it: the score it escaped from
    # had a busiest FINITE arm of 0.0 s and the one it landed on cannot
    assert info["max_after"] > info["max_before"] - 1e-9

    # and the ruler is unchanged where every bag IS flyable: same arity-3 tuple,
    # but the two terms that used to be the whole score still decide it
    a = allocate.load_score({0: 3.0, 1: 5.0})
    b = allocate.load_score({0: 4.0, 1: 4.0})
    assert a[0] == b[0] == 0 and a[1] == 5.0 and b[1] == 4.0 and b < a
    # an unflyable assignment loses to every feasible one, however lopsided
    assert allocate.load_score({0: 1e9, 1: 0.0}) \
        < allocate.load_score({0: allocate.UNFLYABLE, 1: 0.0})


def test_prune_unflyable_drops_only_the_unreachable_span():
    """The bag-level test, on a matrix whose one bad span is known by name.

    `prune_unflyable` asks the TOUR rather than a per-span predicate, and this
    pins why: segment 2 below is reachable from nowhere (in-degree 0, the shape
    arm 97's piece of stroke 26 has on the real rig) and must go, while segment
    1 is unreachable from the DEPOT only and must stay — a cheap "can the arm
    get there from its ready pose" gate would throw it away, and on the real
    orange phase that gate would have cost arm 71 two spans it draws perfectly
    well from its other work.
    """
    n = 3
    N = 2 * n
    C = np.full((N + 1, N + 1), 1.0)
    for i in range(n):                     # a segment cannot follow itself
        C[2 * i:2 * i + 2, 2 * i:2 * i + 2] = np.inf
    C[N, 2:4] = np.inf                     # depot cannot reach segment 1 ...
    C[:, 4:6] = np.inf                     # ... and NOTHING can reach segment 2

    segs = [dict(length=1.0), dict(length=1.0), dict(length=0.5)]
    calls = {}

    def fake_cost_matrix(spec, s, **kw):
        calls["n"] = len(s)
        return C

    real = allocate.sequence.cost_matrix
    allocate.sequence.cost_matrix = fake_cost_matrix
    try:
        keep, drop = allocate.prune_unflyable(None, segs, {})
    finally:
        allocate.sequence.cost_matrix = real

    assert drop == [2], f"pruned {drop}, want only the unreachable segment 2"
    assert keep == [0, 1], f"kept {keep}; segment 1 is reachable from segment 0"
    # and what survives really does have a tour
    sub = allocate._sub_matrix(C, keep, n)
    assert np.isfinite(allocate.sequence.solve(sub, len(keep))["cost"])


def _covered(r):
    """{stroke id: merged covered spans} of a shipped allocation.

    WHAT IS ON THE PAPER, with who drew it and in how many pieces projected
    away.  That is the quantity the balancer must not change, and it is not the
    same as the list of spans: allocation v2 may CUT a span in two, which
    changes the pieces and not one millimetre of the ink.
    """
    return allocate.merged_spans(
        [dict(stroke=dict(id=s["stroke_id"]),
              sp=dict(s0=min(s["s_range"]), s1=max(s["s_range"])))
         for a in r["arms"] for s in r["programs"][a]])


def test_balancing_cannot_change_what_is_drawn():
    """Coverage is invariant under the balancer BY CONSTRUCTION, and stays so.

    A move only ever hands a span to an arm that has certified the SAME span at
    the same endpoints (`replan_same_span` refuses a re-plan that gives back so
    much as a millimetre), and a SPLIT only ever hands over a piece whose
    complement its owner has re-planned — the union of the two is the input span
    for every cut position.  So the covered ink cannot change, which is what
    makes "does load balancing cost coverage" a question with a structural
    answer rather than a measured one.

    Checked three ways: the pure pass refuses an assignment it was not offered;
    a real two-arm allocation covers exactly the same ink with the pass off,
    with whole-segment moves only, and with cutting enabled; and v1's stronger
    claim — that the SPANS THEMSELVES are untouched — is still pinned for v1.
    """
    try:
        allocate.balance_loads([7], [{3, 5}], lambda a, i: 0.0)
        raise AssertionError("an owner outside its own options was accepted")
    except ValueError:
        pass

    ys = (1.631, 1.700, 1.560)          # three strokes arms 31 and 71 both reach
    strokes = [dict(pts=np.column_stack([np.linspace(1.74, 1.87, 14),
                                         np.full(14, y)]),
                    color="grey", kind="outline", id=i) for i, y in enumerate(ys)]
    kw = dict(arms=[31, 71], fleet=FLEET, pens={31: 0.200, 71: 0.200},
              colors={31: "grey", 71: "grey"}, verbose=False)
    raw = allocate.allocate(strokes, balance=False, **kw)
    v1 = allocate.allocate(strokes, balance=True, split=False, **kw)
    v2 = allocate.allocate(strokes, balance=True, split=True, **kw)

    def spans(r):
        return sorted((s["stroke_id"], round(min(s["s_range"]), 9),
                       round(max(s["s_range"]), 9))
                      for a in r["arms"] for s in r["programs"][a])

    assert v1["balance"]["n_movable"] >= 2, \
        "no segment had an alternative arm: the test proves nothing"
    assert v1["balance"]["rounds"] >= 1, "the balancer never moved anything"
    assert spans(raw) == spans(v1), "a whole-segment move changed the SPANS"
    assert v1["balance"]["n_splits"] == 0, "split=False cut something anyway"
    for r, tag in ((v1, "v1"), (v2, "v2")):
        assert _covered(raw) == _covered(r), f"{tag} changed WHICH ink is drawn"
        assert abs(raw["dropped_len"] - r["dropped_len"]) < 1e-12, tag
        assert r["drawn_len"] >= raw["drawn_len"] - 1e-9, \
            f"{tag} ships less ink than the cover it started from"
        assert r["balance"]["coverage_lost_m"] == 0.0, tag
        assert r["balance"]["max_after"] < r["balance"]["max_before"], \
            f"{tag}: the busiest arm did not get lighter"
        # ...and every segment is still drawn by an arm that certified it:
        # re-plan the shipped geometry for its new owner and it must be "ok"
        from aris_sixarm.stroke_api import plan_stroke
        for a in r["arms"]:
            for s in r["programs"][a]:
                p = plan_stroke(np.asarray(s["pts"], float), FLEET[a],
                                {"pen_ext": r["pens"][a]})
                assert p["status"] == "ok", \
                    (f"{tag}: arm {a} cannot certify the segment it was given: "
                     f"{p['status']}")


# ==========================================================================
# allocation v2: stroke splitting as a balancing move
# ==========================================================================
def test_a_cut_covers_the_span_it_cut_and_makes_no_confetti():
    """The geometry of a split, before any arm is asked to certify it.

    `split_span`'s contract is that the union of the two pieces IS the input
    span, for every cut position and both sides — that is the whole coverage
    guarantee, and it should hold at the ends of the interval and at silly cut
    positions outside it, not merely in the middle where it is obvious.
    `split_candidates`'s contract is the other half: never offer a cut that
    makes a piece shorter than the floor.
    """
    L, sp = 2.0, dict(s0=0.20, s1=0.80, direction=1, source="cover")
    splice = allocate.SPLIT_OVERLAP_M
    for side in ("head", "tail"):
        for c in (0.0, 0.2, 0.35, 0.5, 0.799, 0.8, 1.0):
            keep, give = allocate.split_span(sp, c, L, side, splice)
            # the union of the two pieces IS the input span: it reaches both of
            # its ends and there is no gap in the middle, at any cut position
            assert abs(min(keep["s0"], give["s0"]) - sp["s0"]) < 1e-12 and \
                abs(max(keep["s1"], give["s1"]) - sp["s1"]) < 1e-12, \
                f"{side} at {c}: the pieces do not reach the span's own ends"
            lo, hi = sorted((keep, give), key=lambda s: (s["s0"], s["s1"]))
            assert lo["s1"] >= hi["s0"] - 1e-12, \
                f"{side} at {c}: a hole of {hi['s0'] - lo['s1']:.2e} between them"
            # away from the ends the seam is ink drawn twice, 5 mm of it; a cut
            # within half a splice of an end has nowhere to put the other half
            e = 0.5 * splice / L
            if sp["s0"] + e < c < sp["s1"] - e:
                assert abs((lo["s1"] - hi["s0"]) * L - splice) < 1e-9, \
                    (f"{side} at {c}: the seam is "
                     f"{1000 * (lo['s1'] - hi['s0']) * L:.2f} mm, not "
                     f"{1000 * splice:.0f}")

    # and no candidate cut may leave a piece under the floor
    mins = 0.05
    for pre, suf in ((0.80, 0.20), (0.50, 0.60), (0.25, 0.75), (0.20, 0.80)):
        for c, side in allocate.split_candidates(0.20, 0.80, L, pre, suf, 0.3,
                                                 mins, splice):
            keep, give = allocate.split_span(sp, c, L, side, splice)
            for piece in (keep, give):
                assert (piece["s1"] - piece["s0"]) * L >= mins - 1e-9, \
                    (f"candidate {c:.4f} ({side}) makes a "
                     f"{(piece['s1'] - piece['s0']) * L * 1000:.1f} mm piece")
            # the receiver's half must lie inside what it certified
            assert (give["s1"] <= pre + 1e-9 if side == "head"
                    else give["s0"] >= suf - 1e-9), \
                f"candidate {c:.4f} ({side}) reaches past the certified run"


def test_a_split_certifies_both_halves_and_keeps_the_coverage():
    """One stroke, one arm holding all of it, and a cut that both arms certify.

    The instance is the smallest one that a whole-segment balancer provably
    cannot improve: a SINGLE segment.  Handing it over whole only makes the
    receiver the new busiest arm, so v1 has no move at all and must leave the
    load exactly where it found it; v2 cuts it.  What is asserted is not the
    speed (that is the next test) but the two guarantees: both halves come back
    from an INDEPENDENT `plan_stroke` as "ok" for the arm that was given them,
    and the ink on the paper is the same ink.
    """
    from aris_sixarm.stroke_api import plan_stroke
    strokes = [dict(pts=np.column_stack([np.linspace(1.55, 2.05, 60),
                                         np.full(60, 1.64)]),
                    color="grey", kind="line", id=0)]
    kw = dict(arms=[31, 71], fleet=FLEET, pens={31: 0.200, 71: 0.200},
              colors={31: "grey", 71: "grey"}, verbose=False)
    v1 = allocate.allocate(strokes, split=False, **kw)
    v2 = allocate.allocate(strokes, split=True, **kw)

    assert v1["balance"]["rounds"] == 0 and v1["balance"]["n_splits"] == 0, \
        "a one-segment instance should offer a whole-segment balancer nothing"
    n = v2["balance"]["n_splits"]
    assert n >= 1, "the splitter left a single 0.5 m segment on one arm"
    assert v2["balance"]["n_segments_after"] == \
        v2["balance"]["n_segments_before"] + n
    assert v2["balance"]["coverage_lost_m"] == 0.0
    assert _covered(v1) == _covered(v2), "the cut changed WHICH ink is drawn"
    assert v2["dropped_len"] <= v1["dropped_len"] + 1e-12

    used, pieces = set(), 0
    for a in v2["arms"]:
        for s in v2["programs"][a]:
            pieces += 1
            used.add(a)
            assert s["length"] >= allocate.MIN_SPLIT_M - 1e-9, \
                f"arm {a} was handed {1000 * s['length']:.1f} mm of confetti"
            r = plan_stroke(np.asarray(s["pts"], float), FLEET[a],
                            {"pen_ext": v2["pens"][a]})
            assert r["status"] == "ok", \
                f"arm {a} cannot certify its half of the cut: {r['status']}"
    assert len(used) == 2 and pieces == 1 + n, \
        f"{pieces} pieces over {len(used)} arms; wanted {1 + n} over 2"
    # the seam is ink drawn twice, and only that much of it
    extra = v2["drawn_len"] - v1["drawn_len"]
    assert 0 < extra <= n * allocate.SPLIT_OVERLAP_M + 1e-6, \
        f"the cut added {1000 * extra:.2f} mm of ink, not one 5 mm splice"


def test_splitting_beats_not_splitting_on_a_constructed_instance():
    """The constructed case the whole feature exists for, and it must win.

    Three long strokes in the band arms 31 and 71 share.  Whole-segment moves
    can only deal them out 2-1, which leaves one arm carrying half again what
    the other does; cutting can do better, and the assertion is that it DOES —
    a strictly lower busiest arm, out of the same ink, with the coverage
    unmoved.  The run is also required to be reproducible, because a balancer
    whose answer depends on the wall clock cannot be regression-tested.

    THE STROKES ARE 0.70 m, NOT 0.40 m, AND THAT IS THE POINT OF THIS NOTE.
    At 0.40 m this instance is only imbalanced under ONE of the two band
    objectives: `min_travel` balances it to 13.85 s against 13.74 s with no cut
    at all — better than the 14.58 s `maximin_sigma` needs a cut to reach — so
    the splitter examines 20 candidates and correctly refuses every one, and
    the test fails for a reason that has nothing to do with splitting.  At
    0.70 m the whole-segment deal is 38.7 s against 26.1 s under BOTH
    objectives and cutting is needed under both, so what this pins is the
    SPLITTER and not whichever objective happens to be `pwl.OBJECTIVE` today.
    The assertions are the ones it has always had.
    """
    ys = (1.560, 1.640, 1.720)
    strokes = [dict(pts=np.column_stack([np.linspace(1.50, 2.20, 48),
                                         np.full(48, y)]),
                    color="grey", kind="line", id=i) for i, y in enumerate(ys)]
    kw = dict(arms=[31, 71], fleet=FLEET, pens={31: 0.200, 71: 0.200},
              colors={31: "grey", 71: "grey"}, verbose=False)
    v1 = allocate.allocate(strokes, split=False, **kw)
    v2 = allocate.allocate(strokes, split=True, **kw)

    assert v2["balance"]["n_splits"] >= 1, "no cut was taken at all"
    assert v2["balance"]["max_after"] < v1["balance"]["max_after"] - 1e-6, \
        (f"splitting did not beat not splitting: {v2['balance']['max_after']:.2f} "
         f"s vs {v1['balance']['max_after']:.2f} s")
    assert _covered(v1) == _covered(v2), "the extra move cost coverage"
    assert v2["balance"]["coverage_lost_m"] == 0.0

    again = allocate.allocate(strokes, split=True, **kw)
    assert again["balance"]["n_splits"] == v2["balance"]["n_splits"]
    assert abs(again["balance"]["max_after"] - v2["balance"]["max_after"]) < 1e-9, \
        "two runs of the same allocation disagreed"
    assert [round(m["s_cut"], 12) for m in again["balance"]["splits"]] == \
        [round(m["s_cut"], 12) for m in v2["balance"]["splits"]], \
        "the cut positions are not a function of the input"


def test_the_bench_corpus_is_deterministic_and_is_not_the_logo():
    """The anti-overfitting corpus has to be reproducible to be a regression.

    Two calls must give bit-identical geometry, the seeded generators must
    actually depend on their seed (a generator that ignores it is deterministic
    for the wrong reason), and every drawing must land on the sheet — a corpus
    half of which is off the paper would measure the clipper, not the allocator.
    """
    from aris_sixarm import bench
    from aris_sixarm.fleet import SHEET as SH
    assert len(bench.ORDER) == 5 and set(bench.ORDER) == set(bench.BENCH)
    for name in bench.ORDER:
        a, ma = bench.make(name)
        b, mb = bench.make(name)
        assert len(a) == len(b) == ma["n_strokes"], name
        for x, y in zip(a, b):
            assert x["color"] == y["color"] and x["id"] == y["id"], name
            assert np.array_equal(np.asarray(x["pts"]), np.asarray(y["pts"])), \
                f"{name} is not reproducible"
        assert abs(ma["total_m"] - mb["total_m"]) < 1e-12
        pts = np.vstack([np.asarray(s["pts"], float) for s in a])
        assert pts[:, 0].min() > 0 and pts[:, 0].max() < SH[0], f"{name} off sheet"
        assert pts[:, 1].min() > 0 and pts[:, 1].max() < SH[1], f"{name} off sheet"
        assert ma["total_m"] > 5.0, f"{name} is only {ma['total_m']:.2f} m"

    # the two that draw randomness must actually use it
    for name in ("scatter",):
        a, _ = bench.make(name)
        c, _ = bench.make(name, seed=999)
        same = all(np.array_equal(np.asarray(x["pts"]), np.asarray(y["pts"]))
                   for x, y in zip(a, c))
        assert not same, f"{name} ignores its seed"
    # and the corpus is the regime table it claims to be
    assert len(bench.make("spiral")[0]) == 1, "the spiral is not one stroke"
    assert {s["color"] for s in bench.make("duotone")[0]} == {"grey", "orange"}
    assert bench.make("hatch")[1]["total_m"] > 25.0


# ==========================================================================
# the idle policy (aris_sixarm/idle.py)
# ==========================================================================
def _hover(arm, xy):
    """The pose `writing.arm_program` would freeze this arm in over `xy`."""
    from aris_sixarm import writing
    q, _ = writing.lifted_or_lower(FLEET[arm], FLEET[arm].q_seed, np.asarray(xy, float))
    return np.asarray(q, float)


def _short_segment(arm, cx, cy, length=0.20, pen=0.110, n=21):
    """One certified programme entry: a straight `length` m stroke at (cx, cy)."""
    from aris_sixarm.stroke_api import plan_stroke
    pts = np.column_stack([np.linspace(cx - length / 2, cx + length / 2, n),
                           np.full(n, cy)])
    r = plan_stroke(pts, FLEET[arm], {"pen_ext": pen})
    assert r["status"] == "ok", f"arm {arm} at ({cx}, {cy}): {r['status']}"
    return dict(plan=r, pts=np.asarray(r["pts"], float), length=float(length),
                stroke_id=0, color="grey", kind="", s_range=(0.0, 1.0),
                direction=1, flipped=False)


def test_a_frozen_arm_is_an_obstacle_all_the_way_to_the_horizon():
    """Freezing in place moves WHERE an arm stands, not whether it is in the way.

    Two claims, because the policy rests on both.  First the semantics: the
    reachability DP validates an arm's final pose over the whole SUFFIX of the
    run, so an arm whose frozen pose sits in a corridor another arm has not
    reached yet is refused permission to arrive — and `rest_delays` recovers
    exactly that cost by re-running the same DP with the requirement dropped.
    Second the geometry: the clearance `idle` reads off a frozen pose is the
    same number the conductor's own collision image carries in its last row, so
    "static obstacle" means the image, not a separate model that could drift
    from it.
    """
    from aris_sixarm import idle
    # --- semantics, on an image built by hand so the answer is arithmetic ---
    nA, nB, dt, M = 6, 40, 0.5, 200
    F = np.ones((nA - 1, nB - 1), bool)
    F[nA - 2, 10:20] = False          # A's REST pose blocks B's cells 10..19
    prog_b = np.clip(np.arange(M), 0, nB - 1)
    with_rest, m_rest = coordination._dp({"B": F}, {"B": prog_b}, nA, M)
    free, m_free = coordination._dp({"B": F}, {"B": prog_b}, nA, M, rest=False)
    assert m_free == nA - 1, \
        f"nothing blocks A on its way, it should arrive at {nA - 1}, got {m_free}"
    assert m_rest == 21, \
        (f"A may neither move into nor stand in its last cell while B is in "
         f"cells 10..19, so it arrives one step after B leaves them (21), "
         f"not {m_rest}")
    assert m_rest > m_free, "the rest requirement cost nothing: nothing is tested"
    assert np.all(np.diff(with_rest) >= 0) and np.all(np.diff(with_rest) <= 1)
    assert with_rest[m_rest] == nA - 1 and free[m_free] == nA - 1

    # --- geometry: `idle`'s frozen-pose clearance vs the conductor's image ---
    q0 = np.asarray(FLEET[31].q_seed, float)
    qa = np.linspace(q0, q0 + np.array([0.4, 0.1, 0, 0.1, 0, 0, 0]), 12)
    qb = np.linspace(q0, q0 + np.array([0.0, 0.2, 0, 0.2, 0, 0, 0]), 30)
    pa = ArmPath6(31, qa, 0.02)
    pb = ArmPath6(71, qb, 0.02)
    D = coordination.clearance_matrix(pa, pb)
    row = np.minimum(D[-1, :-1], D[-1, 1:]) - coordination.SWEEP_K * pb.step
    got = idle.tube_clearance(pa.q[-1], 31, 0.110, pb, 0)
    assert abs(got - float(row.min())) < 1e-6, \
        (f"the frozen pose reads {1000 * got:.2f} mm, the conductor's own last "
         f"image row says {1000 * float(row.min()):.2f} mm")
    # ...and it is CONSERVATIVE against the row the DP actually consults, which
    # still carries the sweep of A moving into that pose
    margin = 0.08
    img = coordination.free_cells(pa, pb, margin)
    assert np.all(row[img[-1]] >= margin - 1e-6), \
        "the image called a cell free that the frozen pose is not clear of"
    for j0 in (0, 10, 25):
        assert idle.tube_clearance(pa.q[-1], 31, 0.110, pb, j0) >= got - 1e-9, \
            "a shorter remaining tube cannot be tighter than the whole of it"


def test_a_retreat_is_offered_only_on_genuine_interference():
    """The retreat is a repair, and a repair nobody needs is a regression.

    The same frozen pose is put next to two different neighbours: one whose
    swept tube it sits inside (4 mm of clearance against an 80 mm margin) and
    one 153 mm clear.  The first must be offered a way out; the second must be
    left exactly where it is, because moving an arm that is not in the way
    costs seconds, invites a new conflict, and is the sort of thing a policy
    does when it is triggering on geometry instead of on interference.

    What the repair has to be is also pinned: certified as a pose in its own
    right, clear of the whole remaining tube afterwards, and SMALL — the point
    of the policy is that it is not the trip home, and here it is 8.9x less
    joint travel than `q_seed` would have been.
    """
    from aris_sixarm import idle, validate, writing
    spec = FLEET[31]
    bx, by = spec.xy
    q_frozen = _hover(31, (bx + 0.30, by - 0.30))
    home_dq = float(np.max(np.abs(np.asarray(spec.q_seed, float) - q_frozen)))

    def neighbour(cx):
        s = _short_segment(71, cx, 1.331)
        return ArmPath6(71, np.asarray(s["plan"]["qs"], float), 1 / 48.)

    tight, clear = neighbour(1.85), neighbour(2.00)
    gap_t = idle.tube_clearance(q_frozen, 31, 0.110, tight, 0, spec=FLEET[31])
    gap_c = idle.tube_clearance(q_frozen, 31, 0.110, clear, 0, spec=FLEET[31])
    assert gap_t < 0.08 <= gap_c, \
        (f"the fixture is not what the test needs: {1000 * gap_t:.1f} mm and "
         f"{1000 * gap_c:.1f} mm against an 80 mm margin")

    assert idle.frozen_interference(q_frozen, 31, 0.110, {71: clear}, {71: 0},
                                    0.08, spec=FLEET[31]) == {}
    assert idle.plan_retreat(spec, q_frozen, 31, 0.110, {71: clear}, {71: 0},
                             0.08) is None, \
        "a retreat was offered to an arm that is 153 mm clear of everything"

    hit = idle.frozen_interference(q_frozen, 31, 0.110, {71: tight}, {71: 0}, 0.08,
                                   spec=FLEET[31])
    assert set(hit) == {71}
    got = idle.plan_retreat(spec, q_frozen, 31, 0.110, {71: tight}, {71: 0}, 0.08)
    assert got is not None, "no retreat found for a pose 4 mm from another arm"
    assert validate.check_pose(got["q"], spec, None, 0.110)["ok"], \
        f"the retreat pose does not certify: {got['tag']}"
    assert got["clearance"] >= 0.08, \
        f"the retreat is still inside the tube ({1000 * got['clearance']:.1f} mm)"
    assert idle.frozen_interference(got["q"], 31, 0.110, {71: tight}, {71: 0},
                                    0.08, spec=FLEET[31]) == {}
    assert got["dq"] < 0.5 * home_dq, \
        (f"the 'minimal' retreat moves {got['dq']:.2f} rad where going home "
         f"would move {home_dq:.2f} rad")
    # every candidate is a pose the arm may legally stand in, in order of cost
    cands = idle.retreat_candidates(spec, q_frozen, pen_ext=0.110)
    assert cands and all(np.min(np.minimum(c["q"] - FR3_MIN, FR3_MAX - c["q"]))
                         >= validate.MARGIN_GATE - 1e-9 for c in cands)
    assert all(a["dq"] <= b["dq"] for a, b in zip(cands, cands[1:]))
    assert writing.lifted_config(spec, q_frozen, (bx + 0.30, by - 0.30),
                                 margin_min=5.0)[0] is None, \
        "the hover IK ignored the joint-margin floor it was given"


def test_the_slow_taxi_stretches_the_pen_ups_and_never_the_ink():
    """"Drawing has priority over taxiing" is an arithmetic property here.

    The just-in-time policy spends an arm's slack by slowing the moves BETWEEN
    strokes.  If a single one of those seconds ever landed on a stroke instead,
    the certified velocity profile of that stroke would be a different profile
    and the pen would be somewhere else on the paper.  So: the whole stretch
    lands in the pen-up blocks, every stroke keeps its exact duration, the
    drawn joint samples are identical, and the number the sequencer minimised
    (`transit_s`) is untouched — the stretch is reported separately as `taxi_s`
    precisely so that cross-check can keep working.
    """
    from aris_sixarm import idle, writing
    spec = FLEET[31]
    bx, by = spec.xy
    segs = [_short_segment(31, bx + 0.30, by - 0.30),
            _short_segment(31, bx - 0.30, by - 0.30)]
    base = writing.arm_program(spec, segs, pen_ext=0.110)
    S = 7.5
    slow = writing.arm_program(spec, segs, pen_ext=0.110, taxi_stretch=S)

    assert abs(slow["duration"] - (base["duration"] + S)) < 1e-9, \
        f"asked for +{S} s, got +{slow['duration'] - base['duration']:.4f} s"
    assert abs(slow["taxi_s"] - S) < 1e-9
    assert abs(slow["transit_s"] - base["transit_s"]) < 1e-12, \
        "the stretch was folded into the price the sequencer optimised"
    assert abs(slow["draw_s"] - base["draw_s"]) < 1e-9, "the ink got slower"
    sb = [p for p in base["phases"] if p["kind"] == "stroke"]
    ss = [p for p in slow["phases"] if p["kind"] == "stroke"]
    assert len(sb) == len(ss) == len(segs)
    for x, y in zip(sb, ss):
        assert abs((x["t1"] - x["t0"]) - (y["t1"] - y["t0"])) < 1e-9, \
            "a stroke changed duration when only the transits were stretched"
    assert np.allclose(base["q"], slow["q"]), "the stretch moved the PATH"
    assert np.array_equal(base["seg"], slow["seg"])
    # every drawing sample still moves at the same joint speed it was paced at
    for k in range(len(segs)):
        m = base["seg"] == k
        db = np.diff(base["t"][m]), np.diff(slow["t"][m])
        assert np.allclose(db[0], db[1], atol=1e-9)
    # and the guard that throws a stretch away is a strict makespan-then-pause
    # comparison, so a slower fleet can never be kept
    assert not idle._better(dict(duration=10.0, pause_total=0.0),
                            dict(duration=9.0, pause_total=99.0))
    assert idle._better(dict(duration=9.0, pause_total=1.0),
                        dict(duration=9.0, pause_total=2.0))
    assert not idle._better(dict(duration=9.0, pause_total=3.0),
                            dict(duration=9.0, pause_total=2.0)), \
        "a tie on the clock must be broken on pause, not waved through"
    assert idle._better(dict(duration=8.0, pause_total=99.0),
                        dict(duration=9.0, pause_total=0.0)), \
        "makespan is the objective; pause is only the tie-break"


def test_freeze_in_place_beats_going_home_on_a_two_arm_scene():
    """The A/B the whole change rests on, small enough to read.

    Two arms, one 0.20 m stroke each, 0.4 m apart on the paper; the only
    difference between the two runs is what the arms do when they have finished
    drawing.  Going home is not free — it is a metre of joint travel through
    the middle of the rig that the conductor then has to keep everybody else
    out of, and that the arm may not complete until the pose is clear to the
    horizon.  Freezing costs a pen lift.

    The ink is asserted identical in both, because a policy that bought its
    seconds by drawing less would be no policy at all.
    """
    from aris_sixarm import idle, scene_check
    b31, b71 = FLEET[31].xy, FLEET[71].xy
    segs = {a: [] for a in FLEET}
    segs[31] = [_short_segment(31, b31[0] + 0.30, b31[1] - 0.30)]
    segs[71] = [_short_segment(71, b71[0] - 0.30, b71[1] - 0.30)]
    pens = {a: 0.110 for a in FLEET}
    dt = 1 / 48.

    runs = {p: idle.conduct(segs, pens, dt, policy=p, verbose=False, specs=FLEET)
            for p in (idle.POLICY_FREEZE, idle.POLICY_HOME)}
    fre, hom = runs[idle.POLICY_FREEZE], runs[idle.POLICY_HOME]

    assert fre["sch"]["duration"] < hom["sch"]["duration"], \
        (f"freeze {fre['sch']['duration']:.2f} s is not faster than home "
         f"{hom['sch']['duration']:.2f} s")
    for a in (31, 71):
        assert np.allclose(hom["q_end"][a], FLEET[a].q_seed), \
            f"'home' left arm {a} somewhere other than its ready pose"
        assert not np.allclose(fre["q_end"][a], FLEET[a].q_seed), \
            f"'freeze' sent arm {a} home anyway"
        assert abs(fre["progs"][a]["draw_s"] - hom["progs"][a]["draw_s"]) < 1e-6
        assert abs(fre["progs"][a]["draw_len"] - hom["progs"][a]["draw_len"]) < 1e-12
    # both timelines are ones the independent checker signs off, frozen poses
    # and all — the policy may not buy time out of the margin
    for name, r in runs.items():
        M = r["sch"]["M"]
        qtraj = {a: r["samp"][a]["q"][np.clip(r["sch"]["progress"][a][:M], 0,
                                             r["samp"][a]["n"] - 1)]
                 for a in FLEET}
        rep = scene_check.check_timeline(qtraj, dt, r["sch"]["margin"], fleet=FLEET,
                                         pen_ext=pens, verbose=False)
        assert rep["ok"], (f"{name}: scene_check refused (clearance "
                           f"{1000 * rep['min_clearance']:.1f} mm, "
                           f"{rep['frozen_failed']} bad frozen poses)")
        assert rep["frozen_failed"] == 0


if __name__ == "__main__":
    t0 = time.time()
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            t1 = time.time()
            try:
                fn()
                print(f"{name} PASS ({time.time() - t1:.2f} s)")
            except AssertionError as e:
                fails += 1
                print(f"{name} FAIL: {e}")
    print(f"\n{'ALL PASS' if not fails else f'{fails} FAILURES'} "
          f"in {time.time() - t0:.1f} s")
    sys.exit(1 if fails else 0)


# ==========================================================================
# the refusal path is a path, and it has to survive its own edge case
# ==========================================================================
def test_a_refusal_during_the_entry_taxi_reports_instead_of_raising():
    """`unrunnable` must sort the edge that starts from nowhere.

    A blocked progress index BEFORE the first stroke is the entry taxi — the
    move from wherever the arm stands to its first segment — and it has no
    source segment, so `unrunnable` emits `(None, j)` for it.  Mixed with the
    ordinary `(i, j)` edges from later blocks, `sorted` compared `None` with an
    int and raised `TypeError` *inside the refusal path*: a six-arm conduct that
    had correctly diagnosed which transits were impossible died with a type
    error instead of handing the sequencer the blacklist it had just computed.
    `sequence_arm` already understood `i is None` (it forbids the START row of
    the cost matrix); only the sort did not.
    """
    from aris_sixarm import idle
    dt = 0.1
    progs = {7: dict(phases=[
        dict(kind="stroke", seg=0, t0=1.0, t1=2.0),      # first ink at t=1.0
        dict(kind="transit", seg=0, t0=2.0, t1=3.0),
        dict(kind="stroke", seg=1, t0=3.0, t1=4.0),
        dict(kind="transit", seg=1, t0=4.0, t1=5.0),
        dict(kind="stroke", seg=2, t0=5.0, t1=6.0)])}
    orders = {7: [5, 2, 9]}
    blocks = {(7, 3): [2,        # t = 0.2 s: before any ink -> the entry taxi
                       45,       # t = 4.5 s: the transit out of segment 1
                       ]}
    tr, dr = idle.unrunnable(progs, blocks, dt, orders)
    assert tr == {7: [(None, 5), (2, 9)]}                # sorted, None first
    assert dr == {}
    # and the two halves of the fix agree: the key orders, `sequence_arm`
    # consumes.  A `(None, j)` edge forbids the START row, an `(i, j)` edge the
    # (i, j) block — neither is dropped.
    assert idle.edge_key((None, 5)) < idle.edge_key((0, 0))
    assert sorted({(None, 3), (2, 3), (3, 2)}, key=idle.edge_key) \
        == [(None, 3), (2, 3), (3, 2)]
    # ink is still ink: a block inside a stroke is not a tour edge
    tr2, dr2 = idle.unrunnable(progs, {(7, 3): [35]}, dt, orders)
    assert tr2 == {} and dr2 == {7: [1]}


def test_a_refusal_that_is_both_ink_and_a_tour_edge_reports_both():
    """`_refusal` may not drop the INK because a transit was blocked too.

    The note used to be `if tr: ... elif dr: ...`, so an arm that could neither
    fly its tour NOR draw its ink was reported as a tour problem and nothing
    else — and the caller ACTS on that: `csail_schedule.build_phase` blacklists
    the named edges and re-sequences, up to `--reseq-tries` times, chasing an
    ordering that cannot exist because the ALLOCATION is what is impossible.
    Measured on the all-ceiling rig's three-arm CSAIL run, where the three arms
    that draw nothing are still parked in the scene: 9 blocked tour edges
    reported, and arm 71 segments 10/13/18/19 plus arm 2 segments 7/8/9 —
    1598 of the 1910 blocked indices — silently dropped.
    """
    from aris_sixarm import idle

    class _Body:                      # all `hard_blocks` reads off a path
        def __init__(self, moves):
            self.moves = moves

    dt = 0.1
    progs = {7: dict(phases=[
        dict(kind="stroke", seg=0, t0=0.0, t1=2.0),
        dict(kind="transit", seg=0, t0=2.0, t1=3.0),
        dict(kind="stroke", seg=1, t0=3.0, t1=4.0)], duration=4.0)}
    F = np.ones((40, 4), bool)
    F[5] = False                      # t = 0.5 s: inside stroke 0  -> INK
    F[25] = False                     # t = 2.5 s: the transit out of segment 0
    exc = RuntimeError("arm 7 has no monotone pause schedule inside 4 s")
    exc.paths = {7: _Body(True), 3: _Body(False)}
    exc.free = {(7, 3): F}
    exc.margin, exc.sweep = 0.08, 0.55
    out = idle._refusal(exc, progs, dt, {7: [5, 2]}, None)
    assert isinstance(out, idle.Unconductable)
    assert out.draws == {7: [0]} and out.transits == {7: [(5, 2)]}
    msg = str(out)
    assert "INK: arm 7 segment(s) [0]" in msg, msg
    assert "only a different allocation" in msg, msg
    assert "pen-up transits: arm 7 5->2" in msg, msg


def test_a_retreat_that_cannot_be_flown_costs_the_retreat_and_not_the_run():
    """`_programs_per_arm` drops the arm, never the schedule.

    `plan_retreat` certifies a POSE; the move that reaches it is a separate
    question and `writing.arm_program` answers it with `PaperRefused` — which
    the docstring of that exception says in as many words about the entry, the
    go-home and the retreat.  Raised out of the whole-fleet `_programs` call it
    killed the conduct, and it killed it in passes that are only ever OPTIONAL:
    JIT and retreat are both kept only if they beat the baseline.  Measured on
    the all-ceiling rig: a fleet with a conducted 172.7 s schedule in hand lost
    it to arm 2's un-flyable retreat.
    """
    from aris_sixarm import idle, writing
    seen = []

    def fake(spec, segs, **kw):
        seen.append((spec, kw.get("retreat") is not None))
        if kw.get("retreat") is not None and spec == 2:
            raise writing.PaperRefused("arm 2: retreat at segment 10 cannot "
                                       "clear the paper plane")
        return dict(tag="new", arm=spec)

    old, writing.arm_program = writing.arm_program, fake
    try:
        prev = {2: dict(tag="old", arm=2), 71: dict(tag="old", arm=71)}
        progs, refused = idle._programs_per_arm(
            {2: 2, 71: 71}, {2: [], 71: []}, {}, None,
            {2: idle.POLICY_FREEZE, 71: idle.POLICY_FREEZE}, None,
            {2: np.zeros(7), 71: np.zeros(7)}, {2, 71}, prev,
            draw_speed=0.15, transit_speed=0.30, qd_frac=0.6, h_inv=0.85)
    finally:
        writing.arm_program = old
    assert refused == [2]                      # named, so the caller can say so
    assert progs[2] is prev[2]                 # and it KEPT the one it had
    assert progs[71]["tag"] == "new"           # while 71 got its retreat


def test_rotating_the_logo_is_a_placement_and_not_a_distortion():
    """`to_sheet(rotate_deg=90)` turns the logo; it never reshapes it.

    The canvas stopped being the logo's shape when the two units were merged:
    the CSAIL logo is 1.31x wider than tall and the merged canvas is 2.01x
    taller than wide, so upright the logo's WIDTH binds and two thirds of the
    paper is unusable at any size.  Turned 90 degrees the same width limit buys
    a logo 1.31x longer — 1.71x the area — and the aspect ratio must survive
    that untouched, because it is a logo.
    """
    px = [dict(pts=np.array([[0.0, 0.0], [200.0, 0.0], [200.0, 100.0],
                             [0.0, 100.0], [0.0, 0.0]]), color="grey",
               kind="outline")]
    tall = (1.0, 3.0)                                  # a canvas 3x taller
    up, i0 = trace.to_sheet(px, tall, margin=0.05)
    turned, i90 = trace.to_sheet(px, tall, margin=0.05, rotate_deg=90.0)

    # the bounding box swaps, the aspect ratio does not
    assert i0["rotate_deg"] == 0.0 and i90["rotate_deg"] == 90.0
    assert i0["logo_w"] / i0["logo_h"] == pytest.approx(2.0)
    assert i90["logo_w"] / i90["logo_h"] == pytest.approx(0.5)
    # width binds either way on this canvas, so turning it buys area
    # width binds both ways here, so the linear scale grows by the aspect ratio
    # (2x) and the AREA by its square (4x) — the same lever that takes the CSAIL
    # logo from 2.16 m2 upright to 3.71 m2 on its side (aspect 1.31, area 1.71x)
    assert i0["logo_w"] == pytest.approx(i90["logo_w"]) == pytest.approx(0.9)
    assert i0["logo_h"] == pytest.approx(0.45)
    assert i90["logo_h"] == pytest.approx(4.0 * i0["logo_h"]) == pytest.approx(1.8)
    assert (i90["logo_w"] * i90["logo_h"]) == pytest.approx(
        4.0 * i0["logo_w"] * i0["logo_h"])
    # still centred, still inside the margin, still one closed loop of 5 points
    for p, sheet in ((turned[0]["pts"], tall),):
        assert (p[:, 0].max() + p[:, 0].min()) / 2 == pytest.approx(sheet[0] / 2)
        assert (p[:, 1].max() + p[:, 1].min()) / 2 == pytest.approx(sheet[1] / 2)
        assert p[:, 0].min() > 0.049 and p[:, 0].max() < sheet[0] - 0.049
        assert p[:, 1].min() > 0.049 and p[:, 1].max() < sheet[1] - 0.049
    assert len(turned) == len(up) == 1
    assert len(turned[0]["pts"]) == len(up[0]["pts"])
    # 360 degrees is the identity, and the path LENGTH is scale-only
    _, i360 = trace.to_sheet(px, tall, margin=0.05, rotate_deg=360.0)
    assert i360["logo_w"] == pytest.approx(i0["logo_w"])
    assert trace.plen(turned[0]["pts"]) == pytest.approx(
        2.0 * trace.plen(up[0]["pts"]))
