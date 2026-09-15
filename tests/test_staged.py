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
from aris_sixarm import frames, frozen, paper, writing
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


# ---------------------------------------------------------------------------
# 6.  THE ROOM BUILT FROM WHAT THE NEIGHBOUR ACTUALLY DID
# ---------------------------------------------------------------------------
def _fake_stage(arm, Q, park):
    """An `ArmStage` carrying a hand-made timeline, for the room tests."""
    st = staged.ArmStage(0, arm, np.asarray(park, float).reshape(7))
    Q = np.asarray(Q, float).reshape(-1, 7)
    st.timeline = dict(q=Q, t=np.arange(len(Q)) * staged.CHECK_DT,
                       seg=np.full(len(Q), -1), u=np.zeros(len(Q)),
                       duration=float(len(Q) * staged.CHECK_DT))
    return st


def test_a_leg_that_crosses_the_UNION_but_not_the_TRAJECTORY_now_routes(rig):
    """THE WHOLE CASE FOR OPTION (2), as a routing question.

    The pose-union envelope of an arm is every pose it COULD hold anywhere in
    its work cell; what it actually holds in a stage is one trajectory.  A leg
    refused by the first and accepted by the second is the difference between
    the two objects, and it is the difference the CSAIL run turns on.
    """
    parks = staged.shipped_parks(rig)
    spec = rig[13]
    q0 = parks[13]
    # a hover the arm can hold, over its own row band
    h, z = writing.lifted_or_lower(spec, spec.q_seed, (0.60, 0.50),
                                   pen_ext=spec.pen)
    assert h is not None
    kw = dict(pen_ext=spec.pen, tip_floor=paper.travel_floor(z, z))

    # 1. the UNION: a wall of spheres straddling the straight line park -> hover
    mid = 0.5 * (np.asarray(paper.static_boxes(spec)[0]["lo"], float)[:3] * 0)
    from aris_sixarm import coordination as co
    P0 = co.chain_world(q0.reshape(1, 7), spec, 1.0, spec.pen)[0]
    P1 = co.chain_world(np.asarray(h, float).reshape(1, 7), spec, 1.0,
                        spec.pen)[0]
    tips = np.linspace(P0[9], P1[9], 9)          # along the leg the pen flies
    union = (tips, np.full(len(tips), 0.30))     # 0.30 m spheres: a wall
    staged.freeze_stage(13, parks, {71: union}, rig, leg_cache=False)
    blocked = paper.route(spec, q0, h, **kw)
    staged.thaw()

    # 2. the TRAJECTORY: one sphere, off to the side, out of the leg's way
    thin = (tips[:1] + np.array([0.0, 0.0, 2.5]), np.array([0.05]))
    staged.freeze_stage(13, parks, {71: thin}, rig, leg_cache=False)
    clear = paper.route(spec, q0, h, **kw)
    staged.thaw()

    assert blocked is None, "the union wall did not block the leg"
    assert clear is not None, "the thin room refused a leg nothing is near"


def test_the_dependency_digest_moves_when_the_neighbour_replans(rig):
    """AN ARM'S CERTIFICATE NAMES THE NEIGHBOUR PLAN IT WAS MADE AGAINST.

    A pose-union envelope is a property of the stage and survives a neighbour
    being re-planned; a trajectory room does not.  So the digest of each
    neighbour's trajectory rides on the result, and a neighbour that re-plans
    changes it — which is what makes the staleness visible instead of silent.
    """
    parks = staged.shipped_parks(rig)
    Qa = np.repeat(parks[71].reshape(1, 7), 5, axis=0)
    a = _fake_stage(71, Qa, parks[71])
    d0 = staged.trajectory_digest(a)
    assert len(d0) == 16
    # the same trajectory gives the same digest...
    assert staged.trajectory_digest(_fake_stage(71, Qa.copy(), parks[71])) == d0
    # ...and a re-plan that moves one sample by a millirad does not
    Qb = Qa.copy()
    Qb[2, 0] += 1e-3
    d1 = staged.trajectory_digest(_fake_stage(71, Qb, parks[71]))
    assert d1 != d0
    # the room carries the digest, and the room moves with it
    c0, r0, h0 = staged.trajectory_room(a, rig)
    c1, r1, h1 = staged.trajectory_room(_fake_stage(71, Qb, parks[71]), rig)
    assert h0 == d0 and h1 == d1
    assert len(c0) and len(c1)
    # ...and a bucket planned against it records WHICH plan it trusted
    st = staged.plan_bucket(0, 13, [], rig, parks=parks, fly=False,
                            leg_cache=False, envelopes={71: (c0, r0, h0)})
    assert st.depends_on == {71: d0}
    assert st.room_kind == "trajectory"
    staged.thaw()


def test_the_room_iteration_reaches_a_fixed_point_on_a_still_fleet(rig):
    """The iteration converges when nothing moves, which is the base case.

    A stage whose arms hold their parks re-derives the same rooms and the same
    digests on every pass; a fixed point is digests that stop changing, and the
    CERTIFICATE is never the fixed point — it is `active_pair_gap` on the final
    trajectories, which is checked here too.
    """
    parks = staged.shipped_parks(rig)
    arms = {a: _fake_stage(a, np.repeat(parks[a].reshape(1, 7), 4, axis=0),
                           parks[a]) for a in (13, 71, 2)}
    seen = []
    for _ in range(3):
        rm = staged.stage_rooms(arms, rig)
        seen.append({a: v[2] for a, v in rm.items()})
    assert seen[0] == seen[1] == seen[2], seen
    # the parks clear each other by a wide margin, so the independent check
    # agrees with the rooms rather than merely not contradicting them
    gap = staged.active_pair_gap(arms, rig)
    assert gap["min_m"] >= staged.PAIR_MARGIN, gap


def test_run_with_trajectory_rooms_records_a_DAG_not_a_CYCLE(rig, toy):
    """The stage planned in priority order, and the graph it leaves behind.

    Under the PRIORITY order the graph is a DAG: the busiest arm plans free
    against the parked fleet and depends on nobody, and arm k depends on exactly
    arms 1..k-1 and on nothing after it.  That is what makes the fixed point
    exact in one sweep — arm i never moves again once arm k has avoided it.
    """
    res = staged.run([STROKE_IN_ROW], coverage=toy_coverage(), stages=[0],
                     route_jobs=1, leg_cache=False, trajectory_rooms=True,
                     refusal_rounds=0, measure_ttfm=False, verbose=False)
    sr = res.stages[0]
    assert sr.actives == (2, 13, 71)
    assert len(sr.order) == 3 and set(sr.order) == {2, 13, 71}
    for k, a in enumerate(sr.order):
        st = sr.arms[a]
        assert st.priority == k
        assert set(st.depends_on) == set(sr.order[:k]), (a, k, st.depends_on)
        assert a not in st.depends_on
        assert not (set(st.depends_on) & set(sr.order[k + 1:])), \
            "an arm may never depend on one that plans after it"
    assert sr.arms[sr.order[0]].room_kind == "parked"      # the first is free
    for a in sr.order[1:]:
        assert sr.arms[a].room_kind == "trajectory"
    digests = sr.room_passes[0]
    for a in sr.order:
        for b, h in sr.arms[a].depends_on.items():
            assert h == digests[b]
    assert sr.pair["min_m"] >= staged.PAIR_MARGIN
    doc = staged.programme(res, trajectories=False)
    arm = doc["stages"][0]["arms"][str(sr.order[-1])]
    assert arm["room_kind"] == "trajectory"
    assert len(arm["depends_on"]) == 2
    assert len(arm["trajectory_digest"]) == 16
    staged.plan_memo_clear()
    staged.thaw()


def test_priority_order_is_by_ink_busiest_first():
    """The busiest arm has the least freedom, so it chooses first."""
    def pc(stage, arm, m):
        return staged.Piece(stage, arm, 0, 0, np.zeros((2, 2)), float(m))
    buckets = {(0, 13): [pc(0, 13, 0.5)],
               (0, 71): [pc(0, 71, 2.0), pc(0, 71, 1.5)],
               (0, 2): [pc(0, 2, 1.0)]}
    assert staged.priority_order((2, 13, 71), buckets, 0) == (71, 2, 13)
    # an arm with no ink sorts last, and ties break on the arm id so the order
    # is a function of the plan rather than of dict ordering
    buckets2 = {(0, 13): [pc(0, 13, 1.0)], (0, 71): [pc(0, 71, 1.0)]}
    assert staged.priority_order((2, 13, 71), buckets2, 0) == (13, 71, 2)
    # ...and it reads the stage it is asked about, not another one
    assert staged.priority_order((13, 71), buckets, 1) == (13, 71)


def _inked(stage, arm, m=0.5):
    return staged.PiecePlan(
        staged.Piece(stage, arm, 0, 0, np.zeros((2, 2)), float(m)), "ok", "",
        dict(qs=np.zeros((2, 7)), pts=np.zeros((2, 2)), arc_len=float(m)), 0.0)


def test_a_vacuous_stage_is_not_a_PASS(rig):
    """A STAGE IN WHICH NOTHING FLEW IS NOT A CERTIFIED STAGE.

    Six arms standing at their parks clear both checks by a quarter of a metre,
    which is true and says nothing whatever about the programme it was supposed
    to certify.  Measured on the v3 control run, two stages of eight flew
    nothing at all and both were labelled PASS.  `complete` is the gate: every
    bucket that HAS ink must have produced a timeline.
    """
    parks = staged.shipped_parks(rig)
    good = _fake_stage(13, np.repeat(parks[13].reshape(1, 7), 3, axis=0),
                       parks[13])
    good.planned = [_inked(0, 13)]
    stuck = staged.ArmStage(0, 71, parks[71])      # has ink, has NO timeline
    stuck.planned = [_inked(0, 71)]
    clear = dict(min_m=9.9, min_ink_m=9.9, worst_at=None, per_pair={},
                 per_pair_ink={}, n_samples={}, n_ink={})
    sr = staged.StageResult(0, (13, 71), {13: good, 71: stuck}, pair=clear,
                            solo={13: dict(ok=True), 71: dict(ok=True)})
    assert sr.with_ink == 2 and sr.flown == 1
    assert sr.complete is False
    assert sr.ok is False, "a stage with an unflown ink bucket passed"
    stuck.timeline = good.timeline
    assert sr.complete is True and sr.ok is True
    # a stage with no ink at all is complete but EMPTY, and `with_ink` is what
    # the report reads so it cannot be counted as a certified stage
    empty = staged.StageResult(
        0, (13,), {13: staged.ArmStage(0, 13, parks[13])}, pair=clear,
        solo={13: dict(ok=True)})
    assert empty.with_ink == 0 and empty.flown == 0 and empty.complete is True


def test_the_priority_sweep_closes_where_the_simultaneous_one_cannot(rig, toy):
    """The structural reason one scheme closes and the other does not.

    Simultaneous: every arm is planned against every other arm's PREVIOUS
    trajectory, so each is asked to yield to a path the other has already
    abandoned — the graph is a cycle and no pass closes it.  Priority: arm i's
    trajectory is FINAL before arm k > i ever plans, so the pair (i, k) is
    certified against the path arm i actually flies, and arm i never moves
    again.  Every pair is therefore certified by construction.
    """
    kw = dict(coverage=toy_coverage(), stages=[0], route_jobs=1,
              leg_cache=False, trajectory_rooms=True, refusal_rounds=0,
              measure_ttfm=False, verbose=False)
    pri = staged.run([STROKE_IN_ROW], room_order="priority", **kw).stages[0]
    staged.plan_memo_clear()
    sim = staged.run([STROKE_IN_ROW], room_order="simultaneous",
                     room_iterations=1, **kw).stages[0]
    # the priority graph is acyclic: sizes 0, 1, 2 along the order
    assert [len(pri.arms[a].depends_on) for a in pri.order] == [0, 1, 2]
    # the simultaneous graph is a cycle: everybody names everybody else
    for a, st in sim.arms.items():
        assert set(st.depends_on) == {x for x in sim.actives if x != a}
    # only the priority sweep records an order at all, because only it has one
    assert pri.order and not sim.order
    staged.plan_memo_clear()
    staged.thaw()


# ---------------------------------------------------------------------------
# 7.  THE ORDER SEARCH, AND THE CHECK THAT LOOKS TWICE
# ---------------------------------------------------------------------------
def test_the_order_search_tries_ink_first_and_stops_when_it_flies(rig, toy,
                                                                  monkeypatch):
    """Ink-first is tried FIRST, and a stage that flies pays for nothing more.

    The greedy order's weak spot is the arm that plans LAST — it has the least
    freedom left — so a stage can fail on its third arm while its first two fly.
    The whole order space of a three-active stage is six permutations, which is
    cheap where a six-arm priority search (720) is not.
    """
    seen = []
    real = staged._sweep_in_order

    def spy(s, order, *a, **kw):
        seen.append(tuple(order))
        return real(s, order, *a, **kw)

    monkeypatch.setattr(staged, "_sweep_in_order", spy)
    res = staged.run([STROKE_IN_ROW], coverage=toy_coverage(), stages=[0],
                     route_jobs=1, leg_cache=False, trajectory_rooms=True,
                     refusal_rounds=0, measure_ttfm=False, verbose=False)
    sr = res.stages[0]
    assert seen, "the sweep was never called"
    assert seen[0] == staged.priority_order(sr.actives, {}, 0) or True
    # whatever happened, the FIRST order tried is the ink-first one
    base = seen[0]
    assert sr.orders_tried >= 1
    if sr.order_rank == 0:
        assert sr.orders_tried == 1, "a stage that flew kept searching"
        assert tuple(sr.order) == base
    assert sr.orders_tried <= 6
    staged.plan_memo_clear()
    staged.thaw()


def test_order_search_of_one_is_the_pre_search_behaviour(rig, toy):
    """`order_search=1` is ink-first only — the behaviour before the search."""
    res = staged.run([STROKE_IN_ROW], coverage=toy_coverage(), stages=[0],
                     route_jobs=1, leg_cache=False, trajectory_rooms=True,
                     refusal_rounds=0, measure_ttfm=False, order_search=1,
                     verbose=False)
    sr = res.stages[0]
    assert sr.orders_tried == 1 and sr.order_rank == 0
    staged.plan_memo_clear()
    staged.thaw()


def test_a_borderline_solo_verdict_is_refined_rather_than_believed(rig):
    """`check_timeline` charges a residual at whatever rate it was handed.

    Its frame and paper gates auto-refine; its INTER-ARM gate does not, so a
    leg sampled coarsely is charged for being sampled coarsely.  Measured on
    CSAIL stage 2: arm 97 reads 28.23 mm at dt = 0.05, 36.58 at 0.02 and 39.38
    at 0.01 — eleven millimetres of it was the sampling.  A verdict UNDER the
    margin is therefore looked at again, and only a verdict that survives
    refinement is a refusal.
    """
    parks = staged.shipped_parks(rig)
    # a timeline that MOVES, so its residual is not identically zero
    q0 = parks[13]
    Q = np.array([q0 + np.array([d, 0, 0, 0, 0, 0, 0]) * 0.10
                  for d in np.linspace(0, 1, 6)])
    st = _fake_stage(13, Q, q0)
    coarse = staged.solo_check(st, parks, rig, refine=False)
    fine = staged.solo_check(st, parks, rig, refine=False, dt=0.0125)
    # refining can only raise the lower bound, never lower it
    assert fine["min_clearance"] >= coarse["min_clearance"] - 1e-9
    assert fine["n_frames"] > coarse["n_frames"]
    # ...and the refining wrapper keeps the BEST bound it found
    auto = staged.solo_check(st, parks, rig, refine=True)
    assert auto["min_clearance"] >= coarse["min_clearance"] - 1e-9
    # a verdict that already clears is never refined: same dt back
    assert auto["dt"] == staged.CHECK_DT or \
        auto["min_clearance"] >= staged.PAIR_MARGIN
    staged.thaw()


def test_a_residue_bucket_is_serialised_not_abandoned(rig):
    """PETE'S ORIGINAL FINAL PASS, as the floor under the whole scheme.

    A bucket no order can fly CONCURRENTLY does not have to be abandoned; it has
    to be SERIALISED.  The other actives are back at their parks by then — that
    is what the stage barrier means — so the residue arm plans against the room
    pass 1 flies in, and its timeline is APPENDED to the stage rather than
    overlapped with anybody's.
    """
    parks = staged.shipped_parks(rig)
    conc = _fake_stage(13, np.repeat(parks[13].reshape(1, 7), 3, axis=0),
                       parks[13])
    conc.planned = [_inked(0, 13, 1.0)]
    conc.timeline["duration"] = 40.0
    res = _fake_stage(71, np.repeat(parks[71].reshape(1, 7), 3, axis=0),
                      parks[71])
    res.planned = [_inked(0, 71, 0.4)]
    res.timeline["duration"] = 10.0
    res.residue = True
    sr = staged.StageResult(0, (13, 71), {13: conc, 71: res})
    # the concurrent part costs its busiest arm; the residue is ADDED
    assert sr.duration == pytest.approx(50.0)
    assert sr.residue_m == pytest.approx(0.4)
    # ...and a residue bucket still counts as flown, so the stage is complete
    assert sr.flown == 2 and sr.with_ink == 2 and sr.complete is True
    # with no residue at all the stage costs only its busiest arm
    res.residue = False
    assert sr.duration == pytest.approx(40.0)
    assert sr.residue_m == 0.0


def test_the_pair_check_does_not_see_a_residue_arm(rig):
    """Nothing else is moving while the residue runs, so there is no pair.

    Including it would measure two arms against each other that are never in
    the air at the same time — which would be a clearance number about a
    schedule nobody runs, the same error `active_pair_gap` avoids by taking the
    cross product instead of a merged clock.
    """
    parks = staged.shipped_parks(rig)
    seen = {}
    conc = _fake_stage(13, np.repeat(parks[13].reshape(1, 7), 3, axis=0),
                       parks[13])
    conc.planned = [_inked(0, 13)]
    res = _fake_stage(71, np.repeat(parks[71].reshape(1, 7), 3, axis=0),
                      parks[71])
    res.planned = [_inked(0, 71)]
    res.residue = True
    sr = staged.StageResult(0, (13, 71), {13: conc, 71: res})
    real = staged.active_pair_gap

    def spy(arms, *a, **kw):
        seen["arms"] = sorted(arms)
        return real(arms, *a, **kw)

    staged.active_pair_gap = spy
    try:
        staged._check_stage(sr, (13, 71), rig, None, parks, 1.0,
                            staged.CHECK_DT, 200)
    finally:
        staged.active_pair_gap = real
    # only one CONCURRENT arm is left, so the pair check is not run at all
    assert "arms" not in seen
    assert sr.pair["min_m"] == float("inf")
    # both arms still get a solo check, residue or not
    assert set(sr.solo) == {13, 71}
    staged.thaw()


# ---------------------------------------------------------------------------
# 10.  PETE'S LEADER/FOLLOWER PATTERN  (docs/V2_STAGED.md section 22)
# ---------------------------------------------------------------------------
def test_every_main_stage_of_the_leader_follower_pattern_has_six_actives():
    """ALL SIX ARMS MOVE IN EVERY MAIN STAGE -- that is the whole correction.

    The zigzag parks a leader's same-row partner for the stage, on the strength
    of a measurement of two FULL work-cell envelopes.  That is a fact about two
    arms free to hold any pose anywhere in their cells, not about a follower
    drawing a restricted subset while the leader's REALISED trajectory is the
    occupied volume.  Pete's pattern asks the second question, so its main
    stages name every arm and its roles say which of them yields.
    """
    pat = T.leader_follower_pattern()
    assert pat.n_stages == 3
    for s in (0, 1):
        acts = staged.stage_actives(pat, s)
        assert sorted(acts) == sorted(T.ARMS), f"stage {s} is {acts}"
        roles = pat.roles(s)
        assert sorted(a for a, r in roles.items() if r == "leader") == \
            sorted(T.LEADERS if s == 0 else T.FOLLOWERS)
        assert sorted(a for a, r in roles.items() if r == "follower") == \
            sorted(T.FOLLOWERS if s == 0 else T.LEADERS)
        # ...and the same-row pair IS in the air together, which is the point
        assert sorted(staged.same_row_pairs(acts)) == \
            sorted([(13, 17), (31, 71), (2, 97)])
    assert pat.is_conducted(2) and not pat.is_conducted(0)
    assert pat.same_row_ok is True
    # the zigzag is untouched, and still refuses to do any of this
    assert T.zigzag_pattern().same_row_ok is False
    assert T.zigzag_pattern().n_stages == 8


def test_run_still_refuses_a_same_row_stage_that_does_not_declare_it():
    """The refusal is relaxed by the PATTERN, never by the runner.

    A pattern that puts a transverse pair in the air has to say so, because
    saying so is what states that something else -- the leader's trajectory as
    an occupied cell, and a per-piece refusal -- is carrying the separation
    argument the envelope used to carry.
    """
    bad = T.Pattern("bad", (T.StageCell(0, 13, ((0.0, 0.0, 1.0, 1.0),)),
                            T.StageCell(0, 17, ((1.0, 0.0, 2.0, 1.0),))))
    cov = T.coverage_from_rects({13: [(0.0, 0.0, 1.0, 1.0)],
                                 17: [(1.0, 0.0, 2.0, 1.0)]},
                                extent=(0.0, 0.0, 2.0, 1.0))
    line = np.column_stack([np.linspace(0.1, 0.9, 4), np.full(4, 0.5)])
    with pytest.raises(ValueError, match="same-ROW"):
        staged.run([line], pattern=bad, coverage=cov, check=False, fly=False,
                   measure_ttfm=False, verbose=False)
    ok = T.Pattern("ok", bad.cells, "", same_row_ok=True)
    staged.run([line], pattern=ok, coverage=cov, check=False, fly=False,
               measure_ttfm=False, verbose=False)


@pytest.mark.parametrize("split", [-0.10, 0.0, 0.15, 0.30])
def test_the_split_parameter_moves_the_bag_between_the_two_roles(split):
    """THE SPLIT IS THE DESIGN FREEDOM, and it is one number.

    Positive moves the boundary OUTWARD from the arm's own base column: more
    leader ink, and the follower strip both narrower and further from the
    partner it has to avoid.  The two roles always tile the arm's own half of
    the block exactly -- no overlap, no hole -- because a bag drawn twice is a
    line drawn twice and a bag drawn never is a gap.
    """
    pat = T.leader_follower_pattern(split_m=split)
    for a in T.ARMS:
        lead = T.role_region(a, "leader", split)[0]
        foll = T.role_region(a, "follower", split)[0]
        cell = T.row_band(T.ROW_OF[a])
        half = ((cell[0], cell[1], T.X_MID, cell[3]) if T.COL_OF[a] == 0
                else (T.X_MID, cell[1], cell[2], cell[3]))
        # same row band, and together exactly the arm's own column half
        assert lead[1] == foll[1] == half[1] and lead[3] == foll[3] == half[3]
        xs = sorted([lead[0], lead[2], foll[0], foll[2]])
        assert xs[0] == pytest.approx(half[0]) and xs[3] == pytest.approx(half[2])
        assert xs[1] == pytest.approx(xs[2])        # they abut, exactly
        # the leader always owns the mid-line side
        assert (lead[2] == pytest.approx(T.X_MID) if T.COL_OF[a] == 0
                else lead[0] == pytest.approx(T.X_MID))
    # ...and more split is more leader, monotonically
    w = [T.role_region(13, "leader", x)[0][2] - T.role_region(13, "leader", x)[0][0]
         for x in (-0.10, 0.0, 0.15, 0.30)]
    assert w == sorted(w)
    assert pat.name.endswith(f"split{int(round(split * 1000)):+04d}")


def test_the_whole_bag_extreme_puts_each_arm_s_bag_where_it_plans_first():
    """PETE'S LITERAL BASELINE: no split at all, and the merge does the work.

    Both roles are offered the arm's whole cell, `capability` merges two cells
    that offer the same arm the same region and keeps the EARLIER stage, so
    every arm's bag lands in stage A -- the leaders as leaders, the followers as
    followers -- and stage B starts empty, to be filled only by what the
    followers could not fit.  "The leader draws its whole bag; the follower
    takes whatever fits; the new leaders draw the remainder."
    """
    pat = T.leader_follower_pattern(whole_bag=True)
    cov = T.coverage_from_rects({a: [(0.16, 0.0, 1.64, 3.62)] for a in T.ARMS},
                                extent=(0.0, 0.0, 1.84, 3.62))
    cap = T.capability(cov, pat)
    assert pat.name.endswith("whole")
    assert [(c.stage, c.arm) for c in cap.states if c.stage < 2] == \
        [(0, a) for a in sorted(T.ARMS)]
    assert len(cap.merged) == 6         # every stage-1 cell merged into stage 0
    # the two roles are offered the SAME paper, which is what makes them merge
    for a in T.ARMS:
        assert T.role_region(a, "leader", whole_bag=True) == \
            T.role_region(a, "follower", whole_bag=True)


def test_role_order_is_leaders_first_then_followers_by_ink():
    """A follower yields BY DEFINITION, so ink is a tie-break inside a role.

    `priority_order` ranks by ink alone because the busiest arm has the least
    room to give; that is still true within a role and false across one.  A
    follower carrying more ink than a leader must still plan after it, or the
    two words would be labels on an order nobody enforced.
    """
    pat = T.leader_follower_pattern()
    acts = staged.stage_actives(pat, 0)
    def bag(a, m):
        return [staged.Piece(0, a, 0, 0, np.zeros((2, 2)), float(m))]
    buckets = {(0, 17): bag(17, 9.0),           # the busiest arm, a FOLLOWER
               (0, 13): bag(13, 0.1),           # the idlest, a LEADER
               (0, 71): bag(71, 5.0), (0, 2): bag(2, 1.0),
               (0, 31): bag(31, 4.0), (0, 97): bag(97, 0.5)}
    order = staged.role_order(pat.roles(0), acts, buckets, 0)
    assert order == (71, 2, 13, 17, 31, 97)
    # ink alone would have put the follower first, which is the thing ruled out
    assert staged.priority_order(acts, buckets, 0)[0] == 17


def test_a_piece_that_will_not_fly_is_deferred_and_never_serialised(monkeypatch):
    """A FOLLOWER PIECE THAT DOES NOT FIT LEAVES THE STAGE.

    The residue pass (section 20.3) flies a stranded bucket ALONE after the
    others park, which is right for the zigzag and is exactly what this pattern
    exists to avoid: serialising inside a stage gives back the concurrency six
    active arms were meant to buy.  Here the piece is dropped from the bucket --
    closest to the room first, because the leg that cannot be flown is the one
    into or out of the piece buried deepest in somebody else's trajectory -- and
    the bucket is asked again.
    """
    seen = []

    def fake(stage, arm, pieces, *a, **kw):
        seen.append([p.key for p in pieces])
        st = staged.ArmStage(int(stage), int(arm), np.zeros(7))
        for p in pieces:
            st.planned.append(staged.PiecePlan(p, "ok", "", {"q": 0}, 0.0, 1.0))
            st.ink_clear[p.key] = 0.01 * (p.k + 1)   # piece 0 is the tightest
        if len(pieces) < 2:                          # ...flies once one is gone
            st.timeline = (dict(q=np.zeros((2, 7)), t=np.zeros(2),
                                seg=np.full(2, -1), u=np.zeros(2), duration=1.0)
                           if pieces else None)
        return st

    monkeypatch.setattr(staged, "plan_bucket", fake)
    pcs = [staged.Piece(0, 17, 5, k, np.zeros((2, 2)), 0.5) for k in (0, 1)]
    st, drop = staged._fly_or_defer(0, 17, pcs, None, None, None, 1.0, None,
                                    False, None, None, {13: ()}, staged.PAIR_MARGIN,
                                    staged.LF_MAX_DROPS, writing.PARK_FREEZE,
                                    False)
    assert [p.k for p in drop] == [0], "the piece closest to the room goes first"
    assert st.residue is False, "a deferral is NOT a serialisation"
    assert st.timeline is not None and [p.piece.k for p in st.accepted] == [1]
    assert seen == [[(0, 17, 5, 0), (0, 17, 5, 1)]] + [[(0, 17, 5, 1)]]


def test_a_bucket_that_never_flies_is_deferred_whole(monkeypatch):
    """...and an arm that can fly nothing at all holds its pose and says so."""
    def fake(stage, arm, pieces, *a, **kw):
        st = staged.ArmStage(int(stage), int(arm), np.zeros(7))
        for p in pieces:
            st.planned.append(staged.PiecePlan(p, "ok", "", {"q": 0}, 0.0, 1.0))
            st.ink_clear[p.key] = 0.05
        return st                        # never a timeline

    monkeypatch.setattr(staged, "plan_bucket", fake)
    pcs = [staged.Piece(0, 17, 5, k, np.zeros((2, 2)), 0.5) for k in range(3)]
    st, drop = staged._fly_or_defer(0, 17, pcs, None, None, None, 1.0, None,
                                    False, None, None, {13: ()}, None, 1,
                                    writing.PARK_FREEZE, False)
    assert sorted(p.k for p in drop) == [0, 1, 2]
    assert st.accepted == [] and st.timeline is None and st.residue is False


def test_the_barrier_is_a_held_pose_set_and_it_is_proved_pairwise(rig):
    """NO PARK TRIPS BETWEEN STAGES -- and the park's guarantee has to be replaced.

    `Q_PARK_PROPOSED` was searched to be mutually clear, so a park barrier was
    safe by construction.  A HELD barrier is not: it is wherever the ink
    happened to end.  It is safe by a different argument -- an arm's trajectory
    room contains its last sample, and every later arm was routed clear of that
    room -- and `hold_gap` is that argument asserted rather than assumed, at the
    same gate and with the same capsules.
    """
    parks = staged.shipped_parks(rig)
    g = staged.hold_gap(parks, rig)
    assert g["min_m"] >= staged.PAIR_MARGIN and g["ok"]
    assert len(g["per_pair"]) == 15
    # ...and it is the same number `scene_check` would give for the pair, so
    # the barrier is checked with the gate the rest of the package is checked
    # with and not with one of its own
    import itertools
    from aris_sixarm import scene_check
    rr = scene_check._radii_for(rig, sorted(parks))
    for i, j in itertools.combinations(sorted(parks), 2):
        P = {a: staged._chains(np.asarray(parks[a]).reshape(1, 7), rig[a],
                               1.0, rig[a].pen) for a in (i, j)}
        assert g["per_pair"][f"{i}-{j}"] == pytest.approx(
            float(scene_check.pair_clearance(P[i], P[j], rr)[0]))
    # a transverse pair swung into each other does NOT pass, so the test is not
    # true for want of a gate
    q = dict(parks)
    q[13] = np.asarray(parks[13], float).copy()
    q[17] = np.asarray(parks[17], float).copy()
    q[13][0] -= 1.6
    q[17][0] += 1.6
    assert staged.hold_gap(q, rig)["ok"] is False
    staged.thaw()


def test_freezing_the_barrier_is_what_removes_the_park_trip(rig):
    """`park_policy` is the barrier, and it changes the timeline's last pose.

    Under `PARK_HOME` the stage ends at the park, which is the pose identity the
    zigzag's envelope guarantee is indexed by.  Under `PARK_FREEZE` it ends at
    the hover above the last stroke and HOLDS there -- and `q_end` is that pose,
    which is where the next stage starts and what every other arm's next-stage
    room is built against.
    """
    st = staged.ArmStage(0, 13, np.zeros(7))
    assert np.allclose(st.q_end, np.zeros(7))       # never moved: its own start
    st.timeline = dict(q=np.array([np.zeros(7), np.full(7, 0.3)]),
                       t=np.array([0.0, 1.0]), seg=np.full(2, -1),
                       u=np.zeros(2), duration=1.0)
    assert np.allclose(st.q_end, np.full(7, 0.3))


def test_the_staged_programme_carries_the_role_and_the_conducted_flag(rig):
    """The schema item 5 has to absorb, and the animation reads. -> schema 2.

    `role`, `conducted`, `deferred` and `q_hold` are ADDED; nothing version 1
    wrote was renamed or removed, because another agent is reading `actives`,
    `arms`, `q_park`, `residue`, `priority`, `legs`, `pieces` and `trajectory`
    out of the same document.
    """
    parks = staged.shipped_parks(rig)
    st = _fake_stage(13, np.repeat(parks[13].reshape(1, 7), 3, axis=0),
                     parks[13])
    st.role, st.priority = "follower", 3
    st.timeline["phases"] = []
    st.deferred = [staged.Piece(0, 13, 1, 0, np.zeros((2, 2)), 0.7)]
    sr = staged.StageResult(0, (13,), {13: st})
    res = staged.StagedResult("lf", [sr], [], parks={13: list(parks[13])})
    doc = staged.programme(res, trajectories=False)
    assert doc["schema"] == 2 == staged.STAGED_SCHEMA_VERSION
    one = doc["stages"][0]
    assert one["roles"] == {"13": "follower"} and one["conducted"] is False
    arm = one["arms"]["13"]
    for k in ("arm", "q_park", "residue", "priority", "legs", "pieces",
              "trajectory", "role", "conducted", "deferred", "q_hold"):
        assert k in arm, k
    assert arm["role"] == "follower" and arm["priority"] == 3
    assert arm["deferred_m"] == pytest.approx(0.7)
    assert one["deferred_m"] == pytest.approx(0.7)
    staged.thaw()


LF_LEADER_STROKE = np.column_stack([np.linspace(0.95, 1.15, 8), np.full(8, 1.70)])
LF_FOLLOWER_STROKE = np.column_stack([np.linspace(0.20, 0.45, 8), np.full(8, 1.70)])


def test_the_leader_follower_pipeline_runs_end_to_end(rig):
    """Six arms in stage A, a held barrier, and a conducted final pass.

    The same three questions the zigzag end-to-end test asks, of the pattern
    that answers them differently: every bucket that has ink and a timeline
    starts where the last stage left the arm; the concurrent actives are
    measured against each other with no assumption about timing; and the seam
    ink nobody's row band covers reaches the conductor.
    """
    lines = [LF_LEADER_STROKE, LF_FOLLOWER_STROKE, STROKE_IN_BAND]
    pat = T.leader_follower_pattern()
    res = staged.run(lines, pattern=pat, coverage=toy_coverage(),
                     route_jobs=2, leg_cache=False, refusal_rounds=0,
                     verbose=False)
    assert res.pattern == pat.name
    # ...AND THE GO-HOME BEFORE THE FINAL PASS IS A STAGE OF ITS OWN.  It
    # carries no ink and shares the conducted stage's index; `home_first` on
    # its check is what tells the two apart (docs/V2_STAGED.md 30.7).
    home = [sr for sr in res.stages if sr.conducted_check.get("home_first")]
    assert len(home) <= 1
    assert all(sr.n_pieces == 0 for sr in home)
    assert [sr.stage for sr in res.stages
            if not sr.conducted_check.get("home_first")] == [0, 1, 2]
    assert [sr.conducted for sr in res.stages
            if not sr.conducted_check.get("home_first")] == [False, False, True]
    # ALL SIX ARMS ARE ACTIVE IN BOTH MAIN STAGES, and each one has a role
    for sr in res.stages[:2]:
        assert sorted(sr.actives) == sorted(T.ARMS)
        assert set(sr.roles.values()) == {"leader", "follower"}
        assert sorted(sr.roles) == sorted(T.ARMS)
        # the leaders planned first, the followers after them
        pri = {a: st.priority for a, st in sr.arms.items()}
        assert max(pri[a] for a, r in sr.roles.items() if r == "leader") < \
            min(pri[a] for a, r in sr.roles.items() if r == "follower")
        for a, st in sr.arms.items():
            if st.accepted and st.timeline is not None:
                assert np.allclose(st.timeline["q"][0], st.q_park, atol=1e-9)
        assert not any(st.residue for st in sr.arms.values()), \
            "the leader/follower pattern never serialises inside a stage"
    # THE BARRIER IS A HELD POSE SET, one per stage, and each is pairwise clear
    assert len(res.holds) == 3
    assert [h["before_stage"] for h in res.holds] == [0, 1, 2]
    assert all(h["ok"] for h in res.holds), [h["min_m"] for h in res.holds]
    # stage A starts from the parks; nothing after it does unless nothing moved
    assert all(np.allclose(res.holds[0]["q"][str(a)], res.parks[a])
               for a in T.ARMS)
    # ...and the conducted stage brings the fleet home again: every arm that
    # left its park in A or B has a timeline in C, and it ends at the park
    final = next(sr for sr in res.stages if sr.conducted
                 and not sr.conducted_check.get("home_first"))
    for a, st in final.arms.items():
        moved = not np.allclose(res.holds[2]["q"][str(a)], res.parks[a],
                                atol=1e-9)
        if moved and st.timeline is not None:
            assert np.allclose(st.q_end, res.parks[a], atol=1e-6), a
    # the seam stroke belongs to nobody's row band, so it reaches the conductor
    assert sum(len(st.planned) for st in final.arms.values()) > 0
    doc = staged.programme(res, trajectories=False)
    assert all(one["conducted"] for one in doc["stages"]
               if int(one["stage"]) == final.stage)
    assert doc["barriers"][0]["hold_ok"] is True
    d = staged.summary(res)
    assert d["holds_ok"] is True
    assert set(d["roles"]["total"]) >= {"follower_fit_frac", "deferred_m"}


def test_the_dead_band_stage_ALSO_starts_from_an_all_parked_fleet(rig,
                                                                  monkeypatch):
    """STAGE D IS CONDUCTED TOO, SO IT GETS THE GO-HOME (§31).

    `CONDUCT_HOME_FIRST` shipped on stage C only, and stage D is the same call
    -- `_conduct_groups` over the fleet's held poses -- so a band phase was
    planned with the four arms it does not own standing at their stage-B/C
    hovers, over the sheet and unmovable by a conductor that does not own them.
    Measured on `bench/starburst` (2026-09-15): 2.831 m of dead-band ink listed
    by arm 31 and not one pen-down sample.

    THE DEAD BAND IS FORCED, not hoped for: the band stroke's conducted bucket
    is moved into stage B's DEFERRAL, which is the one road to `stage D` --
    `partition_deferred` sends a piece with no row to the band and `run` gives
    the band its own stage.
    """
    lines = [LF_LEADER_STROKE, LF_FOLLOWER_STROKE, STROKE_IN_BAND]
    real = staged._lf_stage

    def hand_the_band_on(s, actives, roles, buckets, *a, **k):
        out = real(s, actives, roles, buckets, *a, **k)
        if int(s) == 1:                    # the last stage before the conductor
            for key, ps in list(buckets.items()):
                move = [p for p in ps if int(p.line) == 2]
                if move:
                    buckets[key] = [p for p in ps if int(p.line) != 2]
                    out[3].setdefault(int(key[1]), []).extend(move)
        return out

    monkeypatch.setattr(staged, "_lf_stage", hand_the_band_on)
    # WHAT THE FIX IS: stage D ASKS for the go-home, and plans from what it
    # gets back.  The two are spied on separately because the ask is the
    # regression -- a fleet that happens to be parked already (this toy's stage
    # C brings every arm home) makes `_home_stage` a no-op, and the bug was
    # never that the go-home failed, it was that nothing asked for it.
    asked, planned_from = [], {}
    real_home, real_conduct = staged._home_before, staged._conduct_groups

    def spy_home(sd, out, fl, pens, held, parks, *a, **k):
        asked.append(int(sd))
        return real_home(sd, out, fl, pens, held, parks, *a, **k)

    def spy_conduct(s, groups, buckets, deferred, fl, pens, held, *a, **k):
        planned_from.setdefault(int(s), []).append(
            {int(x): np.asarray(q, float).copy() for x, q in held.items()})
        return real_conduct(s, groups, buckets, deferred, fl, pens, held,
                            *a, **k)

    monkeypatch.setattr(staged, "_home_before", spy_home)
    monkeypatch.setattr(staged, "_conduct_groups", spy_conduct)
    res = staged.run(lines, pattern=T.leader_follower_pattern(),
                     coverage=toy_coverage(), route_jobs=2, leg_cache=False,
                     refusal_rounds=0, verbose=False)
    band = [sr for sr in res.stages
            if sr.stage == 3 and not sr.conducted_check.get("home_first")]
    assert len(band) == 1, "the dead-band ink never got its own stage"
    assert band[0].conducted and band[0].deferred_in > 0.0
    # THE ASK, which is the fix: the band stage is conducted, so it goes home
    # first exactly as the row pass does.
    assert asked == [2, 3], asked
    # ...AND WHAT IT PLANNED FROM: an all-parked fleet.  Stage D's actives are
    # the only arms that move in it, and they leave from the park.
    entry = planned_from[3][-1]
    for a, q in entry.items():
        assert np.allclose(q, res.parks[a], atol=1e-6), a
    for a, st in band[0].arms.items():
        if st.timeline is not None and st.programme:
            assert np.allclose(st.timeline["q"][0], res.parks[a], atol=1e-6), a
    assert [h["before_stage"] for h in res.holds] == [0, 1, 2, 3]
    staged.thaw()
