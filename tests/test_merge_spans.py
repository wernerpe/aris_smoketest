"""Drawing the whole logo: the segment merge, and the two spans that were left.

The shipped two-pass programme drew 99.3907 % of the CSAIL logo and left two
holes, and they failed for two unrelated reasons:

  stroke 26, s[0.6213, 0.7571], 49.7 mm — arm 2's certified interval stops at
      0.6213 and arm 71's starts at 0.7571.  Both refusals are FOLD-limited,
      not reach-limited, which is precisely what a pen tilt unfolds; arm 97
      certifies the whole span flat and cannot fly to it (docs/TWO_PASS.md 1.2).
  stroke 30, s[0.6488, 0.7012], 10.2 mm — under every minimum in the allocator,
      so no arm was ever asked, while arm 71 held the ink either side of it.

Both are pinned here as CERTIFIED, at the level the allocator actually works
at: an interval a `plan_stroke` call returned as planned, validated
independently, with the lean re-derived from the kinematics.

The geometry is a literal rather than a trace of the source image, so these run
without reading `assets/` and so that a change to the tracer cannot silently
re-aim the regression at a different piece of paper.
"""
import numpy as np
import pytest

from aris_sixarm import allocate, fleet as fleet_mod
from aris_sixarm import stroke_api, tilt, validate
from aris_sixarm.stroke_api import polyline_length, truncate_polyline

# out/csail_program_sub8.json's placement: rotate 90, target width 0.8417 m,
# offset (0, 0), margin 0.06 -> logo centred at (0.9017, 1.81532).
STROKE_26 = np.array([
    [0.8311091232365184, 1.622019921806364],
    [0.8402826862694368, 1.6613351919474426],
    [0.8402826862694368, 1.6809928270179817],
    [0.8442142132835446, 1.6849243540320895],
    [0.8442142132835446, 1.7438972592437074],
    [0.8402826862694368, 1.7491392952625178],
    [0.8402826862694367, 1.7674864213283545],
    [0.8311091232365183, 1.8120437274882435],
    [0.8166935241847896, 1.85529052464343],
    [0.7891728350860346, 1.9077108848315345],
    [0.7524785829543613, 1.9457156459679106],
    [0.7393734929073351, 1.947026154972613],
])
GAP_26 = (0.6213070993040252, 0.7570712790743537)

STROKE_30 = np.array([
    [0.7380629839026325, 1.9457156459679106],
    [0.7380629839026326, 1.7504498042672205],
])
GAP_30 = (0.648787765884172, 0.7012122341158282)


@pytest.fixture
def final6():
    prev = fleet_mod.ACTIVE_RIG
    fleet_mod.activate("final6_opt")
    yield fleet_mod.FLEET
    fleet_mod.activate(prev)


def _certified(plan, spec, cone):
    """The claim, re-derived: certified end to end, inside the cone it names."""
    assert plan["status"] == "ok"
    assert plan["validation"]["ok"]
    rep = validate.validate_plan(plan["pts"], spec, plan["qs"],
                                 times=plan["times"],
                                 tilt_max_deg=float(plan.get("tilt_max_deg",
                                                             0.0) or 0.0))
    assert rep["ok"], [v["kind"] for v in rep["violations"]]
    assert rep["worst"]["max_lean_deg"] <= cone + validate.CONE_EPS_DEG
    assert plan["tip_err"] < 2e-3
    assert plan["coverage_gap"] == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------
# 1. the geometry these tests are about
# --------------------------------------------------------------------------
def test_the_two_spans_are_the_ones_that_were_undrawn():
    """Pin the arithmetic the rest of the file rests on."""
    L26 = polyline_length(STROKE_26)
    L30 = polyline_length(STROKE_30)
    assert L26 == pytest.approx(0.3657, abs=5e-4)
    assert L30 == pytest.approx(0.1953, abs=5e-4)
    assert (GAP_26[1] - GAP_26[0]) * L26 == pytest.approx(0.0497, abs=5e-5)
    assert (GAP_30[1] - GAP_30[0]) * L30 == pytest.approx(0.0102, abs=5e-5)
    # stroke 30's remainder is under every minimum in the allocator, which is
    # why nothing ever asked an arm about it
    assert (GAP_30[1] - GAP_30[0]) * L30 < stroke_api.DEFAULTS["min_length"]
    assert (GAP_30[1] - GAP_30[0]) * L30 < allocate.MIN_SEG_M


# --------------------------------------------------------------------------
# 2. span 26 — the tilt rescue
# --------------------------------------------------------------------------
def test_span_26_is_refused_flat_by_both_neighbours(final6):
    """The negative half, without which the rescue proves nothing.

    Arm 2 walks up to s = 0.6213 and stops; arm 71 walks down to 0.7571 and
    stops.  Neither can be argued into the 49.7 mm between them with the pen
    held perpendicular, in either direction.
    """
    gap = truncate_polyline(STROKE_26, *GAP_26)
    for arm in (2, 71):
        for pts in (gap, gap[::-1]):
            r = stroke_api.plan_stroke(pts, final6[arm], None)
            assert r["status"] == "split" and r["s_star"] == 0.0


def test_span_26_certifies_with_a_15_degree_pen(final6):
    """Both neighbours draw it once the pen may lean, and the extension —
    the whole span from the arm's existing segment through the hole — is what
    ships, so no pen-up is added to close it."""
    for arm, span in ((2, (0.0, GAP_26[1])), (71, (GAP_26[0], 1.0))):
        spec = final6[arm]
        sub = truncate_polyline(STROKE_26, *span)
        flat = stroke_api.plan_stroke(sub, spec, None)
        assert flat["status"] == "split"          # flat cannot reach across

        r = stroke_api.plan_stroke(sub, spec, dict(tilt_max_deg=15.0))
        _certified(r, spec, 15.0)
        assert r["tilt_used"] is True
        assert r["max_lean_deg"] > 1.0
        assert r["arc_len"] == pytest.approx(polyline_length(sub), rel=1e-6)


def test_span_26_the_lean_stays_inside_fifteen_degrees(final6):
    """The cap is a material limit, so it is checked against the KINEMATICS
    and not against the planner's own bookkeeping."""
    spec = final6[2]
    sub = truncate_polyline(STROKE_26, 0.0, GAP_26[1])
    r = stroke_api.plan_stroke(sub, spec, dict(tilt_max_deg=15.0))
    assert r["status"] == "ok"
    _, worst, inside = tilt.cone_check(r["qs"], spec, 15.0)
    assert inside and worst <= 15.0 + 1e-6


def test_span_26_flat_ink_is_untouched_by_the_flag(final6):
    """Turning the rescue on may not re-plan a stroke that already worked.

    `plan_adaptive` plans flat first and returns that result unless tilt beat
    it, so the arm's existing 227 mm of stroke 26 comes back BIT for bit.
    """
    spec = final6[2]
    sub = truncate_polyline(STROKE_26, 0.0, GAP_26[0])
    a = stroke_api.plan_stroke(sub, spec, None)
    b = stroke_api.plan_stroke(sub, spec, dict(tilt_max_deg=15.0))
    assert a["status"] == b["status"] == "ok"
    assert np.array_equal(a["qs"], b["qs"])
    assert b["tilt_used"] is False
    assert b["tilt_max_deg"] == 0.0            # a flat plan, flatly certified
    assert b["tilt_allowance"] == 15.0         # ... under a 15 deg allowance


# --------------------------------------------------------------------------
# 3. span 30 — the merge, and it needs no tilt at all
# --------------------------------------------------------------------------
def test_span_30_merges_flat(final6):
    """10.2 mm that was never a reach problem.

    Arm 71 certifies s[0, 0.6488] and arm 31 s[0.7012, 1]; the hole between
    them is smaller than any span either could be OFFERED, because
    `plan_stroke` calls anything under 20 mm degenerate and the allocator will
    not spend a pen-up under 25 mm.  Asked for the UNION instead — the span it
    already draws plus the remainder — each arm certifies it with the pen
    perpendicular.  No tilt, no extra segment, no extra pen-up.
    """
    micro = truncate_polyline(STROKE_30, *GAP_30)
    assert stroke_api.plan_stroke(micro, final6[71], None)["status"] \
        == "degenerate"

    for arm, span in ((71, (0.0, GAP_30[1])), (31, (GAP_30[0], 1.0))):
        spec = final6[arm]
        r = stroke_api.plan_stroke(truncate_polyline(STROKE_30, *span), spec,
                                   None)
        _certified(r, spec, 0.0)
        assert r.get("max_lean_deg", 0.0) in (0.0, None) or \
            r["max_lean_deg"] < 1e-6


# --------------------------------------------------------------------------
# 4. the merge itself
# --------------------------------------------------------------------------
def _stroke_30_programs(final6):
    """The shipped allocation of stroke 30: arm 71 below the hole, arm 31
    above it, and 10.2 mm of paper between them that nobody was ever asked
    about."""
    st = dict(id=30, color="orange", kind="outline", pts=STROKE_30)
    programs, specs = {}, {}
    for arm, s0, s1, d in ((71, 0.0, GAP_30[0], 1), (31, GAP_30[1], 1.0, -1)):
        spec = final6[arm]
        pts = truncate_polyline(STROKE_30, s0, s1)
        plan = stroke_api.plan_stroke(pts[::-1] if d < 0 else pts, spec, None)
        assert plan["status"] == "ok", (arm, plan["status"])
        sp = dict(s0=s0, s1=s1, arm=arm, direction=d, source="test")
        programs[arm] = [allocate._entry(st, sp, plan)]
        specs[arm] = spec
    return st, programs, specs


def test_merge_absorbs_a_sub_minimum_remainder(final6):
    """The named feature: a hole no arm would take, drawn by its neighbour."""
    st, programs, specs = _stroke_30_programs(final6)
    hole = allocate.leftover([st], programs, allocate.GAP_TOL_M)
    assert len(hole) == 1
    assert hole[0]["length"] == pytest.approx(0.0102, abs=5e-5)

    n, recs = allocate.merge_remainders([st], programs, specs,
                                        {a: None for a in specs})
    assert n == 1 and len(recs) == 1
    assert recs[0]["kind"] == "hole"
    # no segment was added anywhere — one of them simply got longer
    assert sum(len(v) for v in programs.values()) == 2
    grown = programs[recs[0]["arm"]][0]
    assert grown["s_range"][0] <= GAP_30[0] + 1e-9
    assert grown["s_range"][1] >= GAP_30[1] - 1e-9
    _certified(grown["plan"], specs[recs[0]["arm"]], 0.0)
    # and the paper is now completely covered
    assert allocate.leftover([st], programs, allocate.GAP_TOL_M) == []


def test_merge_keeps_the_minimum_for_standalone_segments(final6):
    """The 25 mm floor is a rule about PEN-UPS and it still holds.

    The merge does not create a segment; it lengthens one.  A remainder with no
    segment against it is left alone, however short — there is nothing to
    absorb it into, and a standalone piece of it is exactly what the minimum
    exists to refuse.
    """
    st, programs, specs = _stroke_30_programs(final6)
    # pull both neighbours away so the hole touches nothing
    programs[71][0]["s_range"] = (0.0, 0.40)
    programs[31][0]["s_range"] = (0.90, 1.0)
    n, recs = allocate.merge_remainders([st], programs, specs,
                                        {a: None for a in specs})
    assert n == 0 and recs == []
    assert programs[71][0]["s_range"] == (0.0, 0.40)
    assert programs[31][0]["s_range"] == (0.90, 1.0)


def test_merge_coalesces_two_touching_segments_of_one_arm(final6):
    """The second pass: one arm, one stroke, two segments end to end.

    This is what stroke 26 looks like after the tilt rescue closes its hole —
    arm 2 holding s[0, 0.6213] and s[0.6213, 0.7571] — and the pen-up between
    them buys nothing, so the two are re-planned as one segment.
    """
    arm, spec = 2, final6[2]
    o = dict(tilt_max_deg=15.0)
    st = dict(id=26, color="orange", kind="outline", pts=STROKE_26)
    segs = []
    for s0, s1 in ((0.0, GAP_26[0]), GAP_26):
        plan = stroke_api.plan_stroke(truncate_polyline(STROKE_26, s0, s1),
                                      spec, o)
        assert plan["status"] == "ok"
        segs.append(allocate._entry(st, dict(s0=s0, s1=s1, arm=arm,
                                             direction=1, source="test"), plan))
    programs = {arm: segs}
    n, recs = allocate.merge_remainders([st], programs, {arm: spec}, {arm: o})
    assert n == 1 and recs[0]["kind"] == "adjacent"
    assert len(programs[arm]) == 1               # one pen-up saved
    s0, s1 = programs[arm][0]["s_range"]
    assert s0 == pytest.approx(0.0) and s1 == pytest.approx(GAP_26[1])
    _certified(programs[arm][0]["plan"], spec, 15.0)


def test_merge_rolls_back_what_the_arm_cannot_fly(final6):
    """A longer segment is a different segment to fly to.

    The endpoint moves, so the hover moves, so the tour that was proved
    flyable is not automatically the tour the arm now has.  With a transit
    model that refuses everything, every merge must be given back.
    """
    st, programs, specs = _stroke_30_programs(final6)
    before = {a: v[0]["s_range"] for a, v in programs.items()}

    real = allocate.prune_unflyable
    try:
        allocate.prune_unflyable = lambda *a, **k: ([], [0])
        n, recs = allocate.merge_remainders(
            [st], programs, specs, {a: None for a in specs},
            mat_of=lambda _a: dict(transit_speed=0.3))
    finally:
        allocate.prune_unflyable = real
    assert n == 0 and recs == []
    assert {a: v[0]["s_range"] for a, v in programs.items()} == before
