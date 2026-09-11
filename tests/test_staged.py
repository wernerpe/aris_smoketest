"""STAGE ASSIGNMENT, WIRED END TO END: the pieces, the freeze, and the checks.

NO ENVIRONMENT VARIABLES.  The rig and the tool are switched IN PROCESS by the
`rig` fixture and put back afterwards, exactly as `tests/test_merge_spans.py`
switches to `final6_opt` -- `ARIS_RIG` / `ARIS_TOOL` are the launcher's door and
a test that set them would be changing the run it is part of.

NO ATLAS EITHER, except where the docstring says so.  The capability map the DP
reads is rasterised from rectangles through `traces.coverage_from_rects`, the
same `Coverage` object a swept atlas produces, so a re-sweep can never turn one
of these red for a reason that is not a bug.  What the rectangles CANNOT promise
is that `plan_stroke` will accept a piece -- that is the whole point of the
refusal loop -- so the end-to-end test asserts about the pieces that WERE
accepted and about the checks they passed, and reports the refusals rather than
requiring there to be none.
"""
import numpy as np
import pytest

from aris_sixarm import fleet as fleet_mod
from aris_sixarm import frames, frozen, paper
from aris_sixarm import staged
from aris_sixarm import traces as T


# ---------------------------------------------------------------------------
# 1.  THE PATTERN ITSELF -- no rig, no planner
# ---------------------------------------------------------------------------
def test_no_same_row_pair_is_active_in_any_stage():
    """THE TRANSVERSE PAIR IS UNSEPARABLE ON THIS RIG, ON EITHER AXIS.

    docs/V2_WORKCELLS.md section 4b: two arms of a same-ROW pair are at
    -160.8 mm with 0.80 m of x between their pens and -163.1 mm with 0.81 m of
    y, because both elbows stand in the same column about the mid-line whatever
    the pens do.  No stage of any pattern this module will plan may put one in
    the air together, and the eight-stage zigzag does not.
    """
    pat = T.zigzag_pattern()
    assert pat.n_stages == 8
    for s in range(pat.n_stages):
        acts = staged.stage_actives(pat, s)
        assert acts, f"stage {s} has no active arm"
        assert staged.same_row_pairs(acts) == [], \
            f"stage {s} is {acts}, which contains a same-row pair"
    # ...and the rows are the ones `traces` says they are, so the test is not
    # trivially true for want of a row map
    assert T.ROW_OF == {13: 0, 17: 0, 31: 1, 71: 1, 2: 2, 97: 2}
    assert staged.same_row_pairs((13, 17)) == [(13, 17)]
    assert staged.same_row_pairs((13, 31, 2)) == []


def test_run_refuses_a_pattern_with_a_same_row_stage():
    """The assertion is a REFUSAL, not a report: a stage like this is unrunnable."""
    bad = T.Pattern("bad", (T.StageCell(0, 13, ((0.0, 0.0, 1.0, 1.0),)),
                            T.StageCell(0, 17, ((1.0, 0.0, 2.0, 1.0),))))
    cov = T.coverage_from_rects({13: [(0.0, 0.0, 1.0, 1.0)],
                                 17: [(1.0, 0.0, 2.0, 1.0)]},
                                extent=(0.0, 0.0, 2.0, 1.0))
    line = np.column_stack([np.linspace(0.1, 0.9, 4), np.full(4, 0.5)])
    with pytest.raises(ValueError, match="same-ROW"):
        staged.run([line], pattern=bad, coverage=cov, check=False, fly=False,
                   measure_ttfm=False, verbose=False)


def test_pieces_of_is_exactly_the_dp_s_answer():
    """`pieces_of` adds nothing and loses nothing: one Piece per drawn piece."""
    cov = T.coverage_from_rects({13: [(0.0, 0.0, 1.2, 1.0)],
                                 17: [(0.8, 0.0, 2.0, 1.0)]},
                                extent=(0.0, 0.0, 2.0, 1.0))
    pat = T.Pattern("two", (T.StageCell(0, 13, ((0.0, 0.0, 1.0, 1.0),)),
                            T.StageCell(1, 17, ((1.0, 0.0, 2.0, 1.0),))))
    cap = T.capability(cov, pat)
    line = np.column_stack([np.linspace(0.1, 1.9, 40), np.full(40, 0.5)])
    plan = T.plan_lines([line], cap)
    pcs = staged.pieces_of(plan)
    assert len(pcs) == plan.n_pieces == 2
    assert [p.stage for p in pcs] == [0, 1]
    assert [p.arm for p in pcs] == [13, 17]
    for i, p in enumerate(pcs):
        assert np.allclose(p.pts, plan.lines[0].piece_points(i))
    b = staged.bucket(pcs)
    assert sorted(b) == [(0, 13), (1, 17)]


# ---------------------------------------------------------------------------
# 2.  THE PROPOSED RIG, AND A THREE-STROKE PICTURE
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def rig():
    rig0, tool0 = fleet_mod.ACTIVE_RIG, frames.ACTIVE_TOOL
    fleet_mod.activate("proposed")
    frames.activate_tool("lateral")
    paper.clear_cache()
    yield fleet_mod.FLEET
    staged.thaw()
    paper.disk_cache_close()
    frames.activate_tool(tool0)
    fleet_mod.activate(rig0)
    paper.clear_cache()


def toy_coverage():
    """A hand-written capability map over the certified block.

    Each arm reaches a band around its own row, generously overlapping its
    neighbours' -- which is the shape the real atlas has (13/17 reach
    y in [0, 1.32], 31/71 [1.08, 2.56], 2/97 [2.28, 3.60]) without pretending
    to BE it.
    """
    x0, x1 = T.BLOCK[0], T.BLOCK[2]
    return T.coverage_from_rects(
        {13: [(x0, 0.30, x1, 1.32)], 17: [(x0, 0.30, x1, 1.32)],
         31: [(x0, 1.08, x1, 2.56)], 71: [(x0, 1.08, x1, 2.56)],
         2: [(x0, 2.28, x1, 3.40)], 97: [(x0, 2.28, x1, 3.40)]},
        extent=(0.0, 0.0, T.BLOCK[2] + 0.2, T.BLOCK[3]))


# One stroke inside row band 1, one inside the dead band SEAM0, and one that
# CROSSES from the row band into the dead band.  R1 = y in [1.41, 2.22] and
# SEAM0 = y in [1.01, 1.41] (traces.row_band / traces.seam_band).
STROKE_IN_ROW = np.column_stack([np.linspace(0.55, 0.85, 8), np.full(8, 1.70)])
STROKE_IN_BAND = np.column_stack([np.linspace(0.55, 0.85, 8), np.full(8, 1.22)])
STROKE_CROSSES = np.column_stack([np.full(8, 0.70), np.linspace(1.66, 1.16, 8)])


@pytest.fixture(scope="module")
def toy(rig):
    lines = [STROKE_IN_ROW, STROKE_IN_BAND, STROKE_CROSSES]
    cap = T.capability(toy_coverage(), T.zigzag_pattern())
    plan = T.plan_lines(lines, cap)
    return plan, staged.pieces_of(plan)


def test_a_stroke_crossing_the_dead_band_is_deferred_to_a_seam_stage(toy):
    """Every dead-band metre belongs to a SEAM stage, and none is lost.

    The dead band is the paper the three-active stages give back so that two
    adjacent rows' elbow sweeps stop overlapping (docs/V2_WORKCELLS.md section
    4: -127.2 mm at 0.20 m of band, +85.8 mm at 0.40 m).  It is not a hole: six
    2-active seam stages come back for it, and `traces.plan_lines` cuts the
    stroke at the boundary to 0.01 mm rather than dropping the crossing.
    """
    plan, pcs = toy
    band = T.seam_band(0)
    main = {0, 1}
    for p in pcs:
        mid = p.pts[len(p.pts) // 2]
        inside = band[1] <= mid[1] <= band[3]
        assert (p.stage not in main) == inside, \
            f"piece at y={mid[1]:.3f} is in stage {p.stage}"
    # the crossing stroke is cut into exactly two pieces, one each side, and
    # the cut lands on the boundary
    crossing = [p for p in pcs if p.line == 2]
    assert len(crossing) == 2
    ys = sorted(float(p.pts[-1][1]) for p in crossing) + \
        sorted(float(p.pts[0][1]) for p in crossing)
    assert min(abs(y - band[3]) for y in ys) < 1e-4
    # ...and nothing was dropped: every metre of every line has a drawer
    assert plan.summary()["gaps"] == 0
    assert plan.summary()["covered_frac"] == pytest.approx(1.0)


def test_the_frozen_partner_model_is_what_is_installed_while_planning(rig, toy):
    """THE PARKED PARTNERS ARE REAL CAPSULES, NOT BANDS, FOR EVERY PLAN CALL.

    This is the whole of the parked half of the safety argument: a parked arm
    is ONE known, barrier-verified pose, so it is certified against as that
    pose (`frozen.freeze`), and not as the pose-invariant 0.32 m band the
    atlas sweep has to assume for an arm whose pose nobody has chosen yet.
    The hook fires inside `plan_bucket`, once per piece, and asserts the model
    that is live at that instant.
    """
    _, pcs = toy
    b = staged.bucket(pcs)
    key = (0, 71)
    assert b.get(key), "the toy picture must give arm 71 a stage-0 bucket"
    seen = []

    def spy(stage, arm, piece):
        seen.append((frozen.active(), tuple(frozen.frozen_ids()),
                     frozen.observer()))

    st = staged.plan_bucket(*key, b[key], fly=False, on_piece=spy,
                            leg_cache=False)
    assert seen, "the hook never fired"
    for active, ids, obs in seen:
        assert active is True
        assert ids == (2, 13, 17, 31, 97)      # the five partners, not the mover
        assert obs == 71                       # ...and never itself
    assert st.frozen_partners == (2, 13, 17, 31, 97)
    assert set(st.frozen_poses) == {2, 13, 17, 31, 97}
    for a, q in st.frozen_poses.items():
        assert np.allclose(q, rig[a].q_seed), \
            f"arm {a} is frozen somewhere other than its shipped park"
    staged.thaw()
    assert not frozen.active()


def test_drop_bands_drops_only_the_bands(rig):
    """The CHECK may drop a band whose arm is in the timeline -- and nothing else."""
    spec = rig[71]
    before = spec.static_obstacles()
    names = [b["name"] for b in before]
    assert any(n.startswith("body:13_column") for n in names)
    after = staged.drop_bands(spec, (2, 13, 17, 31, 97)).static_obstacles()
    left = [b["name"] for b in after]
    assert not [n for n in left if n.startswith("body:")]
    # TRUE STRUCTURE IS NEVER DROPPED
    assert [n for n in names if not n.startswith("body:")] == left
    # ...and an arm the caller does not name keeps its band
    some = [b["name"] for b in staged.drop_bands(spec, (13,)).static_obstacles()]
    assert "body:13_column0" not in some
    assert "body:17_column0" in some


def test_staged_end_to_end_on_a_three_stroke_picture(rig, toy):
    """Pieces -> plans -> legs -> a timeline, and both checks pass per stage.

    The two checks are two different questions (see `staged`'s docstring):
    `active_pair_gap` is the ENVELOPE question re-measured from the
    trajectories the planner actually produced, with no assumption about how
    the asynchronous actives line up in time; `solo_check` is the PARKED
    question, one `scene_check.check_timeline` per active arm against the five
    partners held at the parks the freeze was installed at.
    """
    _, pcs = toy
    want = sorted({p.stage for p in pcs})
    res = staged.run([STROKE_IN_ROW, STROKE_IN_BAND, STROKE_CROSSES],
                     coverage=toy_coverage(), stages=want, route_jobs=2,
                     leg_cache=False, verbose=False)
    drawn = 0
    for sr in res.stages:
        for a, st in sr.arms.items():
            drawn += len(st.accepted)
            if st.accepted:
                assert st.timeline is not None, f"stage {sr.stage} arm {a}: {st.note}"
                # THE BARRIER IS WHERE THE STAGE ENDS: pen up, at the park the
                # next stage's envelopes were certified against.
                assert np.allclose(st.timeline["q"][0], st.q_park, atol=1e-9)
                assert np.allclose(st.timeline["q_end"], st.q_park, atol=1e-6)
                assert len(st.programme) == len(st.accepted)
                assert len(st.hovers) == len(st.programme)
        if sr.pair:
            assert sr.pair["min_m"] >= staged.PAIR_MARGIN, sr.pair
        for a, rep in sr.solo.items():
            assert rep["min_clearance"] >= staged.PAIR_MARGIN, (a, rep["min_clearance"])
            assert not rep["frame_failed"] and not rep["paper_failed"]
            assert not rep["self_failed"]
            assert rep["ok"], (a, {k: v for k, v in rep.items()
                                   if k.endswith("failed") and v})
    assert drawn >= 3, "nothing was drawn at all"
    assert res.makespan > 0.0
    assert res.makespan == pytest.approx(sum(s.duration for s in res.stages))
    # the barrier list is one per stage plus the final one
    assert len(res.barriers()) == len(res.stages) + 1
    doc = staged.programme(res, trajectories=False)
    assert doc["schema"] == staged.STAGED_SCHEMA_VERSION
    for one in doc["stages"]:
        for arm in one["arms"].values():
            for pc in arm["pieces"]:
                assert pc["stage"] == one["stage"]
                assert len(pc["q_first"]) == len(pc["q_last"]) == 7
                assert pc["hover_in"] is not None and pc["hover_out"] is not None
