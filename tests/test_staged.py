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
import pathlib

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
    # ENVELOPES OFF HERE, on purpose.  The toy map is rasterised from
    # rectangles and is far more generous than the atlas the ENVELOPES are
    # built from, so it hands arm 71 ink that the real arm 31's envelope stands
    # over; with the envelope room installed that leg is (correctly) refused,
    # which is a statement about the toy map and not about the pipeline.  The
    # envelope room has its own tests above; this one is the item-4 pipeline.
    res = staged.run([STROKE_IN_ROW, STROKE_IN_BAND, STROKE_CROSSES],
                     coverage=toy_coverage(), stages=want, route_jobs=2,
                     leg_cache=False, envelopes=False, refusal_rounds=0,
                     verbose=False)
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


# ---------------------------------------------------------------------------
# 3.  THE OTHER ACTIVE ARMS, AS AN ENVELOPE  (the leg gap, closed)
# ---------------------------------------------------------------------------
def test_cluster_capsules_contains_every_capsule_it_replaces():
    """THE REDUCTION IS CONSERVATIVE OR IT IS NOTHING.

    A row band's envelope is some 9 000 capsules and
    `frozen.partner_clearance` is linear in them, so they are bounded by a few
    hundred grid-local spheres before they go into the room.  The claim that
    makes that legal is containment: every capsule the reduction replaced is
    inside one of the spheres, so a query that clears the spheres clears the
    capsules.  Checked here on a cloud that spans several grid cells.
    """
    rng = np.random.default_rng(7)
    A = rng.uniform(-0.5, 0.5, (400, 3))
    B = A + rng.normal(0, 0.05, (400, 3))
    R = rng.uniform(0.02, 0.06, 400)
    c, r = staged.cluster_capsules(A, B, R, cell=0.15, pad=0.0)
    assert 1 < len(c) < 400
    for i in range(len(A)):
        okA = np.min(np.linalg.norm(c - A[i], axis=1) + R[i] - r)
        okB = np.min(np.linalg.norm(c - B[i], axis=1) + R[i] - r)
        assert okA <= 1e-9 and okB <= 1e-9, i
    # ...and the pad only ever grows them
    _, r2 = staged.cluster_capsules(A, B, R, cell=0.15, pad=0.03)
    assert np.allclose(r2, r + 0.03)


def test_an_active_partner_is_in_the_room_as_its_envelope(rig):
    """`freeze_stage` puts a SET of poses into the static model, not one.

    A parked partner is one pose and an active partner is an envelope; both
    arrive through `frozen.freeze_sets` and everything downstream —
    `filter_boxes`, `partner_clearance`, `paper.static_boxes` — takes them the
    same way.  The test is the one that matters: a point inside the envelope is
    refused, a point outside it is not, and the parked partners are still there.
    """
    parks = staged.shipped_parks(rig)
    # a hand-made "envelope" for arm 71: one sphere, 0.20 m, right in front of
    # arm 13's base, where nothing of arm 71's park is
    centre = np.array([0.60, 1.00, 0.50])
    env = {71: (centre[None, :].copy(), np.array([0.20]))}
    got = staged.freeze_stage(13, parks, env, rig, leg_cache=False)
    assert got == (2, 17, 31, 71, 97)
    assert frozen.observer() == 13
    assert 71 in frozen.frozen_ids()
    P = np.repeat(centre[None, :], 10, axis=0)[None]   # (1 sample, 10 links, 3)
    d_in = float(frozen.partner_clearance(P)[0])
    assert d_in < 0.0, d_in
    far = centre + np.array([0.0, 0.0, 4.0])
    P2 = np.repeat(far[None, :], 10, axis=0)[None]
    assert float(frozen.partner_clearance(P2)[0]) > 0.20
    staged.thaw()


def test_the_envelope_changes_the_leg_store_namespace(rig):
    """A leg bought in one room may never be served in another.

    `paper.route_key` contains neither the frozen poses nor the envelopes —
    both change `static_boxes` without changing any memo key — so the store is
    namespaced on them instead (`staged.leg_cache_signature`).
    """
    parks = staged.shipped_parks(rig)
    base = staged.leg_cache_signature(parks, None)
    env = {71: (np.zeros((1, 3)), np.array([0.2]))}
    assert staged.leg_cache_signature(parks, env) != base
    env2 = {71: (np.zeros((1, 3)), np.array([0.3]))}
    assert staged.leg_cache_signature(parks, env2) != \
        staged.leg_cache_signature(parks, env)


def test_ink_that_crosses_an_active_envelope_is_a_refusal(rig, toy):
    """`plan_stroke` NEVER CONSULTS THE STATIC SET, so the ink needs its own check.

    A piece can be certified end to end — tip on the curve, margin, sigma, the
    arm's own metal — and still be drawn straight through a neighbour's
    envelope.  `ink_vs_envelope` is the missing half, and a piece that fails it
    is a refusal like any other so that the DP can give the ink to somebody
    else.
    """
    _, pcs = toy
    b = staged.bucket(pcs)
    key = (0, 71)
    plain = staged.plan_bucket(*key, b[key], fly=False, leg_cache=False)
    good = plain.accepted
    assert good, "the toy picture must give arm 71 something to draw"
    # an envelope placed exactly on the ink it just certified
    qs = np.asarray(good[0].plan["qs"], float)
    from aris_sixarm import coordination
    P = coordination.chain_world(qs, rig[71], 1.0, rig[71].pen)
    tip = np.asarray(P)[len(P) // 2, 9]
    env = {13: (tip[None, :].copy(), np.array([0.25]))}
    staged.plan_memo_clear()
    hit = staged.plan_bucket(*key, b[key], fly=False, leg_cache=False,
                             envelopes=env, ink_gate=staged.PAIR_MARGIN)
    assert any(p.reason == "ink_vs_active_envelope" for p in hit.refused), \
        [(p.status, p.reason) for p in hit.planned]
    assert hit.ink_clearance and min(hit.ink_clearance) < staged.PAIR_MARGIN
    # ...and WITHOUT the gate it is a measurement and not a refusal, because the
    # envelope is a conservative bound and ink is too expensive to throw at one
    staged.plan_memo_clear()
    soft = staged.plan_bucket(*key, b[key], fly=False, leg_cache=False,
                              envelopes=env)
    assert not [p for p in soft.refused
                if p.reason == "ink_vs_active_envelope"]
    assert soft.ink_clearance == hit.ink_clearance
    staged.thaw()
    staged.plan_memo_clear()


def test_row_lift_ladder_separates_the_bands_and_keeps_the_shipped_rungs():
    """The z lever: one height per ROW, with the shipped ladder behind it."""
    l0, l1, l2 = (staged.row_lift_ladder(a) for a in (13, 71, 2))
    assert l0[0] < l1[0] < l2[0]
    assert l1[0] - l0[0] == pytest.approx(staged.ROW_LIFT_STEP)
    assert l2[0] - l1[0] == pytest.approx(staged.ROW_LIFT_STEP)
    for lad in (l0, l1, l2):
        assert lad[-len(staged.SHIPPED_LADDER):] == staged.SHIPPED_LADDER
    # arms in the same row fly at the same height; 13 and 17 are row 0
    assert staged.row_lift_ladder(17)[0] == l0[0]


# ---------------------------------------------------------------------------
# 4.  THE REFUSAL LOOP
# ---------------------------------------------------------------------------
def test_mask_atoms_cuts_the_atom_rather_than_banning_all_of_it():
    """A REFUSAL IS A NEW TRANSITION, so the atom is cut at it.

    Clearing the bit on every atom a ban merely touches throws the state out of
    the part of the atom the ban does not cover — measured, that is what took
    the CSAIL refusal loop from 100 % coverage to 80.6 %.
    """
    atoms = [T.Atom(0.0, 1.0, 0b111), T.Atom(1.0, 2.0, 0b111),
             T.Atom(2.0, 3.0, 0b111)]
    out = staged.mask_atoms(atoms, [(1, 0.9, 1.5)])
    assert [(round(a.s0, 6), round(a.s1, 6), a.bits) for a in out] == [
        (0.0, 0.9, 0b111), (0.9, 1.0, 0b101),
        (1.0, 1.5, 0b101), (1.5, 2.0, 0b111), (2.0, 3.0, 0b111)]
    # ...and the ink is conserved: the cuts tile the original span exactly
    assert sum(a.s1 - a.s0 for a in out) == pytest.approx(3.0)
    whole = staged.mask_atoms(atoms, [(0, 0.0, 3.0)])
    assert all(a.bits == 0b110 for a in whole)
    assert len(whole) == 3


def test_a_masked_stretch_goes_to_the_neighbour_and_the_ink_survives():
    """Striking one (stage, arm) out of a stretch re-enters it into the DP.

    The whole point of the loop: a piece `plan_stroke` refuses is not ink
    nobody can draw, it is ink THAT ARM cannot draw IN THAT STAGE.  With an
    overlapping map the neighbour absorbs it and the coverage does not move.
    """
    cov = T.coverage_from_rects({13: [(0.0, 0.0, 2.0, 1.0)],
                                 17: [(0.0, 0.0, 2.0, 1.0)]},
                                extent=(0.0, 0.0, 2.0, 1.0))
    pat = T.Pattern("two", (T.StageCell(0, 13, ((0.0, 0.0, 2.0, 1.0),)),
                            T.StageCell(1, 17, ((0.0, 0.0, 2.0, 1.0),))))
    cap = T.capability(cov, pat)
    line = np.column_stack([np.linspace(0.1, 1.9, 40), np.full(40, 0.5)])
    free = staged.plan_lines_masked([line], cap)
    assert free.n_pieces == 1
    k = free.lines[0].pieces[0].state
    banned = staged.plan_lines_masked([line], cap, masks={0: [(k, 0.0, 9.9)]})
    assert banned.n_pieces == 1
    assert banned.lines[0].pieces[0].state != k
    assert banned.summary()["covered_frac"] == pytest.approx(1.0)
    assert banned.summary()["gaps"] == 0
    # ...and with BOTH struck out there is genuinely nobody, which the DP says
    # out loud as a gap rather than by dropping the line
    none = staged.plan_lines_masked(
        [line], cap, masks={0: [(0, 0.0, 9.9), (1, 0.0, 9.9)]})
    assert none.n_pieces == 0
    assert none.summary()["covered_frac"] == pytest.approx(0.0)


def test_resolve_refusals_moves_a_refused_arms_ink_to_its_neighbour(rig, monkeypatch):
    """The loop, end to end, with the planner's verdict forced.

    `plan_stroke` is made to refuse everything arm 71 is offered.  The loop
    must strike (stage 0, arm 71) out of those stretches, hand the ink to a
    stage-compatible neighbour, and come back to full coverage — and it must
    terminate rather than ban the same span for ever.
    """
    real = staged.stroke_api.plan_stroke

    def refuse_71(pts, spec, opts=None, **kw):
        if int(getattr(spec, "arm_id", -1)) == 71:
            return dict(status="split", reason="empty_fiber", s_star=0.0)
        return real(pts, spec, opts, **kw)

    monkeypatch.setattr(staged.stroke_api, "plan_stroke", refuse_71)
    staged.plan_memo_clear()
    cap = T.capability(toy_coverage(), T.zigzag_pattern())
    plan, masks, log = staged.resolve_refusals(
        [STROKE_IN_ROW], cap, rig, stages=[0, 1], rounds=3,
        leg_cache=False, verbose=False)
    assert log[0]["refused"] > 0, "arm 71 was never offered the stroke"
    assert log[-1]["refused"] == 0, log
    assert len(log) <= 4
    assert masks, "nothing was struck out"
    assert all(p.arm != 71 or p.stage != 0 for p in staged.pieces_of(plan))
    assert plan.summary()["covered_frac"] == pytest.approx(1.0)
    staged.plan_memo_clear()
    staged.thaw()


def test_run_installs_the_envelope_room_for_every_active_arm(rig, toy):
    """With `envelopes=True` every bucket is planned in the STAGE's room.

    The pieces and the legs are not the claim here — the toy map is not the
    atlas — only that the other actives of the stage are in the room as
    envelopes when the arm plans, which is what `envelope_partners` records.
    """
    res = staged.run([STROKE_IN_ROW], coverage=toy_coverage(), stages=[0],
                     route_jobs=1, leg_cache=False, fly=False, check=False,
                     refusal_rounds=0, measure_ttfm=False, verbose=False)
    sr = res.stages[0]
    assert sr.actives == (2, 13, 71)
    for a, st in sr.arms.items():
        assert st.envelope_partners == tuple(x for x in (2, 13, 71) if x != a)
    assert res.envelope_s >= 0.0
    staged.plan_memo_clear()
    staged.thaw()


# ---------------------------------------------------------------------------
# 5.  HOW TIGHT THE ENVELOPE HAS TO BE, AND WHY THE CERTIFICATE SURVIVES IT
# ---------------------------------------------------------------------------
ATLAS = pathlib.Path(staged.ATLAS_DEFAULT)
needs_atlas = pytest.mark.skipif(
    not (ATLAS / "atlas_arm13.npz").exists(),
    reason=f"no swept atlas at {ATLAS}")


def test_the_sphere_radius_is_the_whole_conservatism_and_it_is_the_cell():
    """WHY THE CELL IS THE LEVER AND THE PAD IS NOT.

    A cluster sphere's radius is (half-diagonal of the cell it bounds) +
    (the capsule radius) + (the pad).  At the 0.15 m cell the first term alone
    is 0.13 m against a real capsule radius of about 0.06 — the bound is three
    times the thing it bounds — and at 0.05 m it is 0.043.  So shrinking the
    cell buys an order more than zeroing the pad, which is what the sweep in
    docs/V2_STAGED.md section 13 measured and what this pins.
    """
    rng = np.random.default_rng(3)
    A = rng.uniform(-0.4, 0.4, (600, 3))
    B = A + rng.normal(0, 0.02, (600, 3))
    R = np.full(600, 0.06)
    big = staged.cluster_capsules(A, B, R, cell=0.15, pad=0.0)[1].max()
    small = staged.cluster_capsules(A, B, R, cell=0.05, pad=0.0)[1].max()
    assert small < big
    # The bound, stated exactly: a sphere is drawn round the ENDPOINTS of every
    # capsule whose MIDPOINT fell in the cell, so its radius is at most the
    # cell's half-diagonal plus half the longest capsule plus the capsule
    # radius.  The first term is the only one the caller controls, and it is
    # 0.13 m at the 0.15 m cell against 0.043 m at 0.05 m.
    half = 0.5 * float(np.max(np.linalg.norm(B - A, axis=1)))
    for cell in (0.15, 0.10, 0.05):
        r = staged.cluster_capsules(A, B, R, cell=cell, pad=0.0)[1]
        assert r.max() <= cell * np.sqrt(3) / 2 + half + 0.06 + 1e-9
    assert (staged.cluster_capsules(A, B, R, 0.05, 0.0)[1].max()
            < staged.cluster_capsules(A, B, R, 0.15, 0.0)[1].max() - 0.05)
    # ...and zeroing the pad can only shrink, never grow
    assert staged.cluster_capsules(A, B, R, 0.05, 0.0)[1].max() < \
        staged.cluster_capsules(A, B, R, 0.05, 0.04)[1].max()


@needs_atlas
def test_stride_one_needs_no_pad_because_it_skips_no_pose(rig):
    """`ENVELOPE_PAD` EXISTS ONLY TO COVER THE CELLS A STRIDE SKIPS.

    At stride 1 the envelope reads EVERY strict-GO cell of the region, so there
    is no skipped pose for a pad to stand in for and `pad = 0` is not an
    optimism — it is the exact object `scripts/workcell_envelopes.py` measures.
    At stride 2 three cells in four are skipped and the pad is the only thing
    covering them, which is why the two settings are swept together.
    """
    from aris_sixarm import atlas as atlas_mod
    region = (traces_mod_row := T.row_band(1),)
    arr, meta = atlas_mod.load(ATLAS, 71)
    arr = arr[atlas_mod.strict_go(arr)]
    inside = T.rect_contains(region, arr[:, 0], arr[:, 1])
    n_cells = int(inside.sum())
    Q1 = staged.envelope_poses(71, region, str(ATLAS), rig[71], 1.0,
                               staged.shipped_parks(rig)[71], stride=1)
    assert len(Q1) == 2 * n_cells + 1        # a draw, a hover, and the park
    Q2 = staged.envelope_poses(71, region, str(ATLAS), rig[71], 1.0,
                               staged.shipped_parks(rig)[71], stride=2)
    assert len(Q2) < len(Q1)                 # ...three cells in four skipped
    assert np.allclose(Q1[-1], Q2[-1])       # both end at the same park


@needs_atlas
def test_stage_envelope_is_deterministic_and_cached(rig, tmp_path):
    """The same setting gives the same spheres, from cache or from scratch."""
    region = (T.seam_band(0),)
    kw = dict(stride=2, cluster=0.10, pad=0.0, cache_dir=str(tmp_path))
    parks = staged.shipped_parks(rig)
    c1, r1 = staged.stage_envelope(13, region, str(ATLAS), rig[13], 1.0,
                                   parks[13], {13: rig[13].pen}, **kw)
    assert list(tmp_path.glob("*.npz")), "nothing was cached"
    c2, r2 = staged.stage_envelope(13, region, str(ATLAS), rig[13], 1.0,
                                   parks[13], {13: rig[13].pen}, **kw)
    assert np.array_equal(c1, c2) and np.array_equal(r1, r2)
    # a different setting is a different file and a different answer
    c3, r3 = staged.stage_envelope(13, region, str(ATLAS), rig[13], 1.0,
                                   parks[13], {13: rig[13].pen},
                                   **dict(kw, cluster=0.05))
    assert len(c3) > len(c1) and r3.max() < r1.max()


@needs_atlas
def test_the_shipped_envelope_contains_the_capsules_of_every_pose_in_it(rig):
    """THE CERTIFICATE, ON A REAL ENVELOPE AT THE SHIPPED SETTING.

    `test_cluster_capsules_contains_every_capsule_it_replaces` pins the
    reduction on a synthetic cloud.  This pins the thing that actually goes into
    the room: the spheres `stage_envelope` builds at the module's own defaults
    must contain the link capsules of EVERY pose of the envelope — every
    certified drawing pose in the cell, every hover, and the park — or a leg
    certified against them is certified against the wrong object.

    Checked capsule by capsule, both endpoints, radius included.
    """
    from aris_sixarm import coordination as co
    region = (T.seam_band(0),)
    parks = staged.shipped_parks(rig)
    Q = staged.cached_envelope_poses(13, region, str(ATLAS), rig[13], 1.0,
                                     parks[13], stride=staged.ENVELOPE_STRIDE)
    c, r = staged.stage_envelope(13, region, str(ATLAS), rig[13], 1.0,
                                 parks[13], {13: rig[13].pen})
    assert len(Q) > 10 and len(c) > 1
    path = co.ArmPath(13, Q, 0.01, 1.0, rig[13].pen, rig[13])
    keep = [k for k in range(len(path.r))
            if k not in co.FROZEN_SWEEP_BANDS]
    A = np.asarray(path.A, float)[:, keep].reshape(-1, 3)
    B = np.asarray(path.B, float)[:, keep].reshape(-1, 3)
    R = np.tile(np.asarray(path.r, float)[keep], len(Q))
    # a stride > 1 reads a SUBSET of the cells, so the containment claim is
    # about the poses the envelope was built from — which is exactly what the
    # pad exists to widen beyond (see `test_stride_one_needs_no_pad...`)
    for i in range(0, len(A), max(1, len(A) // 400)):
        dA = np.min(np.linalg.norm(c - A[i], axis=1) + R[i] - r)
        dB = np.min(np.linalg.norm(c - B[i], axis=1) + R[i] - r)
        assert dA <= 1e-9 and dB <= 1e-9, (i, dA, dB)


def test_the_adopted_envelope_setting_is_the_one_the_docs_quote():
    """The defaults are a measurement's conclusion, so they are pinned.

    If these move, `docs/V2_STAGED.md` section 13's sweep table and the
    `docs/DECISIONS.md` entry that adopts them are describing a different room.
    """
    assert staged.ENVELOPE_STRIDE == 1
    assert staged.ENVELOPE_CLUSTER == 0.075
    assert staged.ENVELOPE_PAD == 0.0
    # ...and pad 0 is only honest at stride 1, which is the pairing adopted
    assert not (staged.ENVELOPE_PAD == 0.0 and staged.ENVELOPE_STRIDE > 1)
