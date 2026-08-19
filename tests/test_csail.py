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
