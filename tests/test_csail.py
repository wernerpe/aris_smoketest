"""Tracer + allocator regressions (fast: no IK, no image files).

Run: python3 tests/test_csail.py     (or pytest tests/)
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from aris_sixarm import allocate, coordination, scene_check, trace   # noqa: E402
from aris_sixarm.allocate import Interval            # noqa: E402
from aris_sixarm.fleet import FLEET, SHEET           # noqa: E402
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
    assert allocate.active_arms() == allocate.ACTIVE == [13, 17, 31, 97]
    assert allocate.active_arms("all") == [13, 17, 31, 2, 71, 97]     # registry order
    assert allocate.active_arms([97, 13]) == [13, 97]                 # order normalised
    assert allocate.active_arms({2: True}) == [13, 17, 31, 2, 97]     # patch one flag
    assert allocate.active_arms({13: False, 71: True}) == [17, 31, 71, 97]
    assert {a: s.active for a, s in FLEET.items()} == before, "the registry moved"

    for bad in ("some", [13, 999], {42: True}):
        try:
            allocate.active_arms(bad)
        except ValueError:
            continue
        raise AssertionError(f"active_override={bad!r} should have been rejected")

    # the entry point refuses the ambiguous call rather than picking a winner
    try:
        allocate.allocate([], arms=[13], active_override="all")
    except ValueError:
        pass
    else:
        raise AssertionError("arms= and active_override= together must raise")


def test_active_override_reaches_the_partition_enumeration():
    """Six arms means 62 pen partitions, and the allocation must use all six."""
    assert len(allocate.partitions(allocate.active_arms("all"))) == 62
    strokes = [dict(id=0, color="grey", kind="outline",
                    pts=np.array([[0.0, 0.0], [1.0, 0.0]])),
               dict(id=1, color="orange", kind="outline",
                    pts=np.array([[0.0, 1.0], [1.0, 1.0]]))]
    ivmap = {0: [Interval(0.0, 1.0, 71)], 1: [Interval(0.0, 1.0, 2)]}
    arms = allocate.active_arms("all")
    colors, cover, table = allocate.best_partition(strokes, ivmap, arms)
    # only the two parked arms can cover anything, so the partition has to give
    # them the two different pens — which the four-arm fleet cannot do at all
    assert colors[71] == "grey" and colors[2] == "orange"
    assert cover["dropped_len"] < 1e-9
    assert allocate.best_partition(strokes, ivmap, allocate.ACTIVE)[1]["dropped_len"] > 1.9


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
    paths = {31: coordination.ArmPath(31, moving, 0.02),
             97: coordination.ArmPath(97, q0[None, :], 0.02),
             71: coordination.ArmPath(71, np.linspace(q0, q0 + 0.1, 20), 0.02)}
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
    short = coordination.ArmPath(31, q, 0.01, pen_ext=0.110)
    long_ = coordination.ArmPath(31, q, 0.01, pen_ext=0.300)
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
    paths = coordination.arm_paths({31: q, 97: q}, 0.01, pens={31: 0.300})
    assert np.allclose(paths[31].B[0][-1], tip_l)
    assert np.allclose(paths[97].B[0][-1],
                       coordination.ArmPath(97, q, 0.01, pen_ext=0.110).B[0][-1])
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
    res = allocate.allocate(strokes, arms=[31], pens={31: 0.300},
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
    paths = {31: coordination.ArmPath(31, np.linspace(q0, q0 + 0.25, 40), 0.02),
             97: coordination.ArmPath(97, q0[None, :], 0.02),
             71: coordination.ArmPath(71, np.linspace(q0, q0 + 0.1, 20), 0.02)}
    searched = coordination.coordinate(paths, verbose=False)
    guessed = coordination.coordinate(paths, priority_search=False, verbose=False)
    assert searched["duration"] <= guessed["duration"] + 1e-9, \
        (f"the search made it worse: {searched['duration']:.3f} s vs "
         f"{guessed['duration']:.3f} s")
    assert searched["search"]["n_permutations"] == 2 and guessed["search"] is None
    # the deadlock-recovery path is still there and still reachable
    assert coordination.coordinate(paths, priority_search=False,
                                   retry_orders=True, verbose=False)["attempts"] >= 1


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


def test_balancing_cannot_change_what_is_drawn():
    """Coverage is invariant under the balancer BY CONSTRUCTION, and stays so.

    A move only ever hands a span to an arm that has certified the SAME span at
    the same endpoints (`replan_same_span` refuses a re-plan that gives back so
    much as a millimetre), so the set of drawn spans cannot change — which is
    what makes "does load balancing cost coverage" a question with a
    structural answer rather than a measured one.  Checked twice: the pure pass
    refuses an assignment it was not offered, and a real two-arm allocation
    draws exactly the same ink with the pass on and off.
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
    kw = dict(arms=[31, 71], pens={31: 0.200, 71: 0.200},
              colors={31: "grey", 71: "grey"}, verbose=False)
    raw = allocate.allocate(strokes, balance=False, **kw)
    bal = allocate.allocate(strokes, balance=True, **kw)

    def spans(r):
        return sorted((s["stroke_id"], round(min(s["s_range"]), 9),
                       round(max(s["s_range"]), 9))
                      for a in r["arms"] for s in r["programs"][a])

    assert bal["balance"]["n_movable"] >= 2, \
        "no segment had an alternative arm: the test proves nothing"
    assert bal["balance"]["rounds"] >= 1, "the balancer never moved anything"
    assert spans(raw) == spans(bal), "the balancer changed WHICH ink is drawn"
    assert abs(raw["dropped_len"] - bal["dropped_len"]) < 1e-12
    assert abs(raw["drawn_len"] - bal["drawn_len"]) < 1e-9
    assert bal["balance"]["max_after"] < bal["balance"]["max_before"], \
        "the busiest arm did not get lighter"
    # ...and every segment is still drawn by an arm that certified it: re-plan
    # the shipped geometry for its new owner and it must come back "ok"
    from aris_sixarm.stroke_api import plan_stroke
    for a in bal["arms"]:
        for s in bal["programs"][a]:
            r = plan_stroke(np.asarray(s["pts"], float), FLEET[a],
                            {"pen_ext": bal["pens"][a]})
            assert r["status"] == "ok", \
                f"arm {a} cannot certify the segment it was given: {r['status']}"


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
    pa = coordination.ArmPath(31, qa, 0.02)
    pb = coordination.ArmPath(71, qb, 0.02)
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
        return coordination.ArmPath(71, np.asarray(s["plan"]["qs"], float), 1 / 48.)

    tight, clear = neighbour(1.85), neighbour(2.00)
    gap_t = idle.tube_clearance(q_frozen, 31, 0.110, tight, 0)
    gap_c = idle.tube_clearance(q_frozen, 31, 0.110, clear, 0)
    assert gap_t < 0.08 <= gap_c, \
        (f"the fixture is not what the test needs: {1000 * gap_t:.1f} mm and "
         f"{1000 * gap_c:.1f} mm against an 80 mm margin")

    assert idle.frozen_interference(q_frozen, 31, 0.110, {71: clear}, {71: 0},
                                    0.08) == {}
    assert idle.plan_retreat(spec, q_frozen, 31, 0.110, {71: clear}, {71: 0},
                             0.08) is None, \
        "a retreat was offered to an arm that is 153 mm clear of everything"

    hit = idle.frozen_interference(q_frozen, 31, 0.110, {71: tight}, {71: 0}, 0.08)
    assert set(hit) == {71}
    got = idle.plan_retreat(spec, q_frozen, 31, 0.110, {71: tight}, {71: 0}, 0.08)
    assert got is not None, "no retreat found for a pose 4 mm from another arm"
    assert validate.check_pose(got["q"], spec, None, 0.110)["ok"], \
        f"the retreat pose does not certify: {got['tag']}"
    assert got["clearance"] >= 0.08, \
        f"the retreat is still inside the tube ({1000 * got['clearance']:.1f} mm)"
    assert idle.frozen_interference(got["q"], 31, 0.110, {71: tight}, {71: 0},
                                    0.08) == {}
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

    runs = {p: idle.conduct(segs, pens, dt, policy=p, verbose=False)
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
        rep = scene_check.check_timeline(qtraj, dt, r["sch"]["margin"],
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
