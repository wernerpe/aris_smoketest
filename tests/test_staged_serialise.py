"""THE LAST TWO DEFECTS between the staged programme and a certificate.

Both are about a pose somebody STANDS IN while somebody else moves, and both
were measured on the lf6 programmes of 2026-09-14 (docs/V2_STAGED.md §29).

  1. A ROW GROUP THE CONDUCTOR REFUSED FLEW ONE ARM AT A TIME AND NOBODY TOLD
     THE MOVER ABOUT THE ARM STANDING NEXT TO IT.  The routes came off
     `plan_bucket`, which runs BEFORE `freeze_conduct` with `partners=outside`
     only, so a row's own two arms were never in each other's room: 2 <-> 97 at
     -161.3 mm, 31 <-> 71 at -21.0 mm, in both cases against an arm whose
     per-frame travel is 0.00 mm.  `_serialise_group` re-plans the group IN
     PRIORITY ORDER -- busiest arm first against its partner's HELD POSE, then
     the second against the first's REALISED TRAJECTORY as an exact swept room,
     with the partners' finishing poses as the fallback for a piece the room
     costs -- and `_merge_conducts` lays the slots down in that same order.
  2. A HELD HOVER THAT KEPT NEITHER THE JOINT MARGIN NOR THE ROOM WAS HELD
     ANYWAY.  `hover_solve`'s `hold` ask SETTLES back to `HOVER_MARGIN` where
     the fiber has nothing stricter, so `arm_program` was stopping the stage on
     a pose at 0.1064 rad against `validate.MARGIN_GATE`'s 0.15 (`lf6_s150`
     stage B, arm 31) and the whole stage was refused on `frozen_failed`.  The
     ask is now VERIFIED, and where it fails the arm ends its stage on a
     CERTIFIED RETREAT instead: more air over the same stroke end, a hover over
     an earlier one, or the park -- which is always valid.

NO ENVIRONMENT VARIABLES.  The rig and the tool are switched in process by the
`rig` fixture and put back, exactly as `tests/test_staged.py` does it.
"""
import numpy as np
import pytest

from aris_sixarm import fleet as fleet_mod
from aris_sixarm import frames, paper, staged, writing


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


def _piece(arm, m, stage=2, line=0, k=0):
    pts = np.array([[0.0, 0.0], [float(m), 0.0]])
    return staged.Piece(stage=stage, arm=arm, line=line, k=k, pts=pts,
                        length_m=float(m))


def _held_stage(arm, q, n=8, dt=0.05, pieces=0):
    """An `ArmStage` whose timeline holds one pose for `n` frames."""
    q = np.asarray(q, float).reshape(7)
    st = staged.ArmStage(2, int(arm), q)
    st.planned = [staged.PiecePlan(_piece(arm, 0.1, k=i), "ok",
                                   plan=dict(arc_len=0.1))
                  for i in range(int(pieces))]
    Q = np.repeat(q.reshape(1, 7), n, axis=0)
    st.timeline = dict(t=dt * np.arange(n), q=Q, seg=-np.ones(n, int),
                       u=np.zeros(n), phases=[], duration=dt * (n - 1),
                       draw_s=0.0, transit_s=0.0)
    return st


# ---------------------------------------------------------------------------
# 1.  THE MERGE LAYS THE SLOTS DOWN IN THE ORDER THEY WERE PLANNED IN
# ---------------------------------------------------------------------------
def test_a_refused_group_is_merged_in_its_planned_order_not_by_arm_id(rig):
    """THE CERTIFICATE IS ORDERED AND SO IS THE PROGRAMME.

    `_serialise_group` certifies the SECOND arm against the FIRST arm's realised
    trajectory, so laying the slots down in ascending arm id -- which is what the
    merge used to do -- would ship a programme in which the certified-against arm
    flies second.  Arm 71 is the busier one here, so it must own frame 0 and
    arm 31 must start where 71 stops.
    """
    parks = staged.shipped_parks(rig)
    n71, n31 = 12, 7
    part = ({71: _held_stage(71, parks[71], n71),
             31: _held_stage(31, parks[31], n31)},
            dict(ok=True, serialised=True, serial_order=[71, 31],
                 min_clearance=1.0), 1.0)
    # the slot the composer reserves is the SUM, which is what makes the two
    # disjoint (docs/V2_STAGED.md §28.7) -- asked BEFORE the merge, which pads
    # every timeline it is given to the merged length.
    assert staged._slot_frames(part, 0.05) == (n71 - 1) + (n31 - 1) + 1
    _, rep = staged._merge_conducts(2, [part], rig, {a: rig[a].pen for a in rig},
                                    parks, parks, staged.H_INV_DEFAULT,
                                    0.05, 2)
    off = rep["offsets_s"]
    assert "71" not in off                      # frame 0
    assert off["31"] == pytest.approx(0.05 * (n71 - 1))


def test_the_merge_still_falls_back_to_the_arm_ids_without_a_planned_order(rig):
    """An older part carries no `serial_order`; it is laid down as before."""
    parks = staged.shipped_parks(rig)
    part = ({71: _held_stage(71, parks[71], 12),
             31: _held_stage(31, parks[31], 7)},
            dict(ok=True, serialised=True, min_clearance=1.0), 1.0)
    _, rep = staged._merge_conducts(2, [part], rig, {a: rig[a].pen for a in rig},
                                    parks, parks, staged.H_INV_DEFAULT,
                                    0.05, 2)
    assert "31" not in rep["offsets_s"]         # arm 31 first, by id
    assert rep["offsets_s"]["71"] == pytest.approx(0.05 * 6)


# ---------------------------------------------------------------------------
# 2.  THE PRIORITY ORDER INSIDE THE GROUP
# ---------------------------------------------------------------------------
def test_the_busiest_arm_of_a_refused_group_plans_first_against_its_partner(rig):
    """BUSIEST FIRST, AND THE PARTNER IS IN THE ROOM AS THE POSE IT HOLDS.

    The whole defect is that the arm standing still was in nobody's room.  This
    pins the two halves of the fix that are decisions rather than geometry: the
    order, and WHAT each arm was frozen against -- the outside four, plus the
    other arm of its own row.
    """
    parks = staged.shipped_parks(rig)
    pens = {a: rig[a].pen for a in rig}
    calls = []

    def fake_plan(stage, arm, pieces, specs=None, pens=None, parks=None,
                  h_inv=None, opts=None, envelopes=None, partners=None,
                  **kw):
        calls.append(dict(arm=int(arm), partners=tuple(sorted(partners or ())),
                          envelopes=tuple(sorted(envelopes or ())),
                          q_start=np.asarray(parks[int(arm)], float).copy()))
        return _held_stage(int(arm), parks[int(arm)], 6 + len(calls),
                           pieces=len(pieces))

    real = staged.plan_bucket
    staged.plan_bucket = fake_plan
    try:
        arms, rep = staged._serialise_group(
            2, (31, 71), {(2, 31): [_piece(31, 0.2)],
                          (2, 71): [_piece(71, 2.0), _piece(71, 1.0, k=1)]},
            {}, rig, pens, dict(parks), parks, staged.H_INV_DEFAULT, None,
            False, None, 0.05, 2, False, [2, 13, 17, 97], {}, {}, "drop",
            "the conductor refused")
    finally:
        staged.plan_bucket = real

    assert rep["serial_order"] == [71, 31]       # 3.0 m against 0.2 m
    assert [c["arm"] for c in calls] == [71, 31]
    # the busiest arm sees the four outside arms AND its own row partner
    assert calls[0]["partners"] == (2, 13, 17, 31, 97)
    assert calls[0]["envelopes"] == ()           # ...as a pose, not a room
    # ...and the second sees the first as an EXACT SWEPT ROOM
    assert calls[1]["partners"] == (2, 13, 17, 71, 97)
    assert calls[1]["envelopes"] == (71,)
    assert set(arms) == {31, 71}
    assert rep["serialised"] and rep["in_group_priority"]


def test_the_second_arm_is_replanned_against_the_finishing_poses_if_the_room_costs_ink(rig):
    """THE PIECE THE SWEPT ROOM COSTS IS OFFERED THE SLOT INSTEAD.

    The slots do not overlap -- `_merge_conducts` lays a refused group end to
    end -- so while the second arm flies, the first has finished and is standing
    at its park.  Certifying against its whole swept room is therefore strictly
    stronger than the programme needs, and where that strength costs a piece the
    arm is re-planned against the finishing POSE, which is the honest statement
    of the slot it actually flies in.
    """
    parks = staged.shipped_parks(rig)
    pens = {a: rig[a].pen for a in rig}
    calls = []

    def fake_plan(stage, arm, pieces, specs=None, pens=None, parks=None,
                  h_inv=None, opts=None, envelopes=None, partners=None, **kw):
        calls.append((int(arm), tuple(sorted(envelopes or ()))))
        st = _held_stage(int(arm), parks[int(arm)], 6, pieces=len(pieces))
        if int(arm) == 31 and envelopes:
            st.planned = []                      # the room refused both pieces
            st.timeline = None
        return st

    real = staged.plan_bucket
    staged.plan_bucket = fake_plan
    try:
        arms, rep = staged._serialise_group(
            2, (31, 71), {(2, 31): [_piece(31, 0.2), _piece(31, 0.2, k=1)],
                          (2, 71): [_piece(71, 2.0)]},
            {}, rig, pens, dict(parks), parks, staged.H_INV_DEFAULT, None,
            False, None, 0.05, 2, False, [2, 13, 17, 97], {}, {}, "drop", "x")
    finally:
        staged.plan_bucket = real

    assert calls == [(71, ()), (31, (71,)), (31, ())]
    assert len(arms[31].accepted) == 2           # the post-slot plan shipped
    assert arms[31].timeline is not None
    assert "finishing poses" in arms[31].note


def test_every_slot_boundary_of_a_refused_group_carries_a_hold_proof(rig):
    """A SLOT BOUNDARY IS A BARRIER: six arms in one scene, nothing moving."""
    parks = staged.shipped_parks(rig)
    pens = {a: rig[a].pen for a in rig}

    def fake_plan(stage, arm, pieces, specs=None, pens=None, parks=None,
                  h_inv=None, opts=None, **kw):
        return _held_stage(int(arm), parks[int(arm)], 6, pieces=len(pieces))

    real = staged.plan_bucket
    staged.plan_bucket = fake_plan
    try:
        _, rep = staged._serialise_group(
            2, (31, 71), {(2, 31): [_piece(31, 0.2)], (2, 71): [_piece(71, 2.0)]},
            {}, rig, pens, dict(parks), parks, staged.H_INV_DEFAULT, None,
            False, None, 0.05, 2, False, [2, 13, 17, 97], {}, {}, "drop", "x")
    finally:
        staged.plan_bucket = real
    # one per slot, plus the state the group leaves behind
    assert [h["before"] for h in rep["slot_holds"]] == [71, 31, None]
    assert all(h["ok"] for h in rep["slot_holds"])


# ---------------------------------------------------------------------------
# 3.  A BARRIER MAY NOT HOLD AN UNCERTIFIABLE POSE
# ---------------------------------------------------------------------------
def test_hold_gap_refuses_a_pose_the_judge_would_refuse(rig):
    """PAIRWISE CLEARANCE IS HALF THE QUESTION.

    The other half is about each pose ALONE, and it is the half that failed:
    `scene_check` judges a held pose with `validate_pose`, so the barrier asks
    the same question where it is declared rather than finding out afterwards.
    """
    parks = staged.shipped_parks(rig)
    pens = {a: rig[a].pen for a in rig}
    good = staged.hold_gap(parks, rig, pens)
    assert good["ok"] and good["poses_ok"] and good["bad_poses"] == []
    # ...a pose at a joint limit is not one a barrier may hold, however far it
    # stands from everybody else.
    bad = dict(parks)
    q = np.asarray(parks[31], float).copy()
    q[3] = frames.FR3_MIN[3] + 1e-4
    bad[31] = q
    rep = staged.hold_gap(bad, rig, pens)
    assert rep["bad_poses"] == [31]
    assert not rep["poses_ok"] and not rep["ok"]


# ---------------------------------------------------------------------------
# 4.  THE CERTIFIED RETREAT
# ---------------------------------------------------------------------------
def test_held_pose_ok_refuses_the_hover_the_settle_hands_back(rig, monkeypatch):
    """`hover_solve` SETTLES; the verification is what makes it a hold.

    The park is a pose a barrier holds every day.  A pose 0.11 rad off a joint
    limit is legal at `HOVER_MARGIN` -- which is what the settle hands back --
    and is exactly the pose `validate.MARGIN_GATE` refused on `lf6_s150` stage
    B.  The room half of the gate is neutralised for the second half of this so
    that what is measured is the MARGIN and not this rig's geometry.
    """
    spec = rig[31]
    park = np.asarray(spec.q_seed, float)
    assert writing.held_pose_ok(spec, park, spec.pen,
                                hold=writing.HOVER_HOLD_MARGIN)
    q = park.copy()
    q[3] = frames.FR3_MIN[3] + 0.11             # legal at 0.10, not at 0.15
    assert 0.10 <= frames.joint_margin(q) < 0.15
    monkeypatch.setattr(writing, "static_gate", lambda *a, **k: None)
    assert writing.held_pose_ok(spec, q, spec.pen, hold=writing.HOVER_MARGIN)
    assert not writing.held_pose_ok(spec, q, spec.pen,
                                    hold=writing.HOVER_HOLD_MARGIN)


def test_the_park_is_always_the_last_rung_and_is_never_filtered(rig):
    """(c) ALWAYS EXISTS.  An arm with nowhere else to stand goes home."""
    spec = rig[71]
    cands = writing.hold_candidates(spec, [], np.asarray(spec.q_seed, float),
                                    spec.pen)
    assert [k for _, k in cands] == ["park"]
    assert np.allclose(cands[0][0], spec.q_seed)


def test_the_three_rungs_are_walked_cheapest_first(rig, monkeypatch):
    """(a) MORE AIR, (b) AN EARLIER STROKE END, (c) THE PARK -- in that order.

    The rungs are exercised by controlling what the fiber answers, because what
    is under test is WHICH QUESTION IS ASKED NEXT and not whether this rig's
    geometry happens to answer the first one.
    """
    spec = rig[71]
    q0 = np.asarray(spec.q_seed, float)
    dense = [dict(pts=np.array([[0.40, 1.00], [0.45, 1.00]])),
             dict(pts=np.array([[0.60, 1.20], [0.65, 1.20]]))]
    last, prev = (0.65, 1.20), (0.45, 1.00)
    monkeypatch.setattr(writing, "held_pose_ok",
                        lambda *a, **k: True)

    def fiber(allow):
        def fake(spec_, q_ref, xy, heights=None, **kw):
            xy = tuple(round(float(v), 6) for v in np.asarray(xy).ravel())
            if xy not in allow:
                return None
            return np.asarray(spec_.q_seed, float) + 0.01, float(heights[0])
        return fake

    # (a) the same tip, higher -- and the ladder it is asked of is the EXTRA one
    monkeypatch.setattr(writing, "lifted_or_lower", fiber({last, prev}))
    kinds = [k for _, k in writing.hold_candidates(spec, dense, q0, spec.pen)]
    assert kinds[0] == "hover_z" and kinds[-1] == "park"
    # (b) nothing over the last end; an earlier one answers
    monkeypatch.setattr(writing, "lifted_or_lower", fiber({prev}))
    assert [k for _, k in
            writing.hold_candidates(spec, dense, q0, spec.pen)] \
        == ["hover_prev", "park"]
    # (c) nothing on the fiber at all
    monkeypatch.setattr(writing, "lifted_or_lower", fiber(set()))
    assert [k for _, k in
            writing.hold_candidates(spec, dense, q0, spec.pen)] == ["park"]


def test_a_pose_the_barrier_may_not_hold_is_never_offered_as_a_rung(rig,
                                                                    monkeypatch):
    """Every non-park rung passes exactly the gate the barrier will apply."""
    spec = rig[71]
    q0 = np.asarray(spec.q_seed, float)
    dense = [dict(pts=np.array([[0.60, 1.20], [0.65, 1.20]]))]
    monkeypatch.setattr(writing, "lifted_or_lower",
                        lambda spec_, q_ref, xy, heights=None, **kw:
                        (np.asarray(spec_.q_seed, float) + 0.01,
                         float(heights[0])))
    monkeypatch.setattr(writing, "held_pose_ok", lambda *a, **k: False)
    assert [k for _, k in
            writing.hold_candidates(spec, dense, q0, spec.pen)] == ["park"]


def test_an_empty_bucket_still_reports_what_it_is_holding(rig):
    """`hold_kind` is on every timeline, so a barrier can say what it holds."""
    spec = rig[71]
    tl = writing.arm_program(spec, [], h_inv=staged.H_INV_DEFAULT,
                             pen_ext=spec.pen,
                             q_start=np.asarray(spec.q_seed, float))
    assert tl["hold_kind"] == "still"


# ---------------------------------------------------------------------------
# 2b.  THE POST-SLOT ORDER -- A BUCKET IS NEVER DROPPED FOR WANT OF A SLOT
# ---------------------------------------------------------------------------
def test_a_bucket_that_will_not_fly_is_given_the_post_slot_not_dropped(rig):
    """§29.5's named next step, measured on `lf6b_s150` and closed here.

    Busiest-first makes the busiest arm plan against its row partner AT THE
    PARTNER'S ENTRY POSE and the second arm against that partner AT ITS PARK --
    two different questions, and a group of two has only the two.  On
    `lf6b_s150` stage C arm 31's 8 pieces (1.914 m) were PLANNED and never
    flown: the bucket produced no timeline in either reading, the go-home
    `aside` gave the arm a timeline with no ink in it, and the metre and a half
    is the largest hole in the picture.  So when a bucket does not fly, the arm
    goes to the BACK of the order -- the slot in which every partner has
    finished and gone home -- and the order that flies more ink ships.
    """
    parks = staged.shipped_parks(rig)
    pens = {a: rig[a].pen for a in rig}
    # THE ENTRY POSE AND THE PARK HAVE TO DIFFER or the two orders ask the same
    # question: an arm that has flown is at its PARK (`PARK_HOME`) and one that
    # has not started is at the pose the last stage left it holding.
    held = dict(parks)
    held[71] = np.asarray(parks[71], float) + 0.02
    calls = []

    def fake_plan(stage, arm, pieces, specs=None, pens=None, parks=None,
                  h_inv=None, opts=None, envelopes=None, partners=None, **kw):
        # arm 31 flies only while arm 71 is still standing at its ENTRY pose,
        # i.e. only when arm 31 goes FIRST.  Planned after 71 has flown and
        # gone home, its bucket refuses -- which is `lf6b_s150` stage C.
        home71 = bool(np.allclose(np.asarray(parks[71], float),
                                  np.asarray(shipped[71], float)))
        calls.append(int(arm))
        st = _held_stage(int(arm), shipped[int(arm)], 6, pieces=len(pieces))
        if int(arm) == 31 and home71:
            st.planned, st.timeline = [], None
        return st

    shipped = parks
    real = staged.plan_bucket
    staged.plan_bucket = fake_plan
    try:
        arms, rep = staged._serialise_group(
            2, (31, 71), {(2, 31): [_piece(31, 0.9)],
                          (2, 71): [_piece(71, 2.0)]},
            {}, rig, pens, held, parks, staged.H_INV_DEFAULT, None,
            False, None, 0.05, 2, False, [2, 13, 17, 97], {}, {}, "drop", "x")
    finally:
        staged.plan_bucket = real

    # the ink-first order is tried first and LOSES arm 31's bucket...
    assert rep["orders_tried"][0]["order"] == [71, 31]
    assert rep["orders_tried"][0]["lost_m"] == pytest.approx(0.9)
    # ...so the post-slot order is tried, loses nothing, and ships
    assert rep["orders_tried"][1]["order"] == [31, 71]
    assert rep["orders_tried"][1]["lost_m"] == pytest.approx(0.0)
    assert rep["serial_order"] == [31, 71]
    assert rep["lost_m"] == pytest.approx(0.0)
    assert arms[31].timeline is not None and arms[31].accepted


def test_the_second_order_is_not_planned_when_the_first_one_loses_nothing(rig):
    """THE SEARCH IS PAID FOR ONLY WHERE IT BUYS SOMETHING.

    A refused group is already planned twice -- once by `plan_bucket` against
    the outside fleet and once per arm under the in-group room -- and stage C's
    wall went from 93 s to 1 428 s for it (§29.3).  A second ORDER doubles that
    again, so it is tried only where the first order lost a whole bucket.
    """
    parks = staged.shipped_parks(rig)
    pens = {a: rig[a].pen for a in rig}
    calls = []

    def fake_plan(stage, arm, pieces, specs=None, pens=None, parks=None,
                  h_inv=None, opts=None, **kw):
        calls.append(int(arm))
        return _held_stage(int(arm), parks[int(arm)], 6, pieces=len(pieces))

    real = staged.plan_bucket
    staged.plan_bucket = fake_plan
    try:
        _, rep = staged._serialise_group(
            2, (31, 71), {(2, 31): [_piece(31, 0.2)], (2, 71): [_piece(71, 2.0)]},
            {}, rig, pens, dict(parks), parks, staged.H_INV_DEFAULT, None,
            False, None, 0.05, 2, False, [2, 13, 17, 97], {}, {}, "drop", "x")
    finally:
        staged.plan_bucket = real
    assert calls == [71, 31]                    # one order, two plans
    assert len(rep["orders_tried"]) == 1
    assert rep["lost_m"] == pytest.approx(0.0)
