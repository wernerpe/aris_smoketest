"""THE ROOM BOUNDARY AS A CUT LINE, and the final pass as three row conductors.

Two builds, two files' worth of claims, one place:

  1. `staged.split_at_room` -- a piece the room ink gate refuses WHOLE is re-cut
     where it enters the room, and the certified clear stretches come back.  The
     claims are that the clear half really is the clear half, that a stretch too
     short to be a piece is not kept, and that a new piece end whose hover the
     router could not hold is not offered.
  2. `staged.partition_deferred` -- the final pass runs as one conductor per ROW
     because rows are separated by the 0.40 m dead band, so a piece that does
     NOT stay inside a row band may never reach a row conductor.

NO ENVIRONMENT VARIABLES.  The rig and the tool are switched in process by the
`rig` fixture and put back, exactly as `tests/test_staged.py` does it.
"""
import numpy as np
import pytest

from aris_sixarm import coordination, fleet as fleet_mod
from aris_sixarm import frames, paper, staged
from aris_sixarm import traces as T


# ---------------------------------------------------------------------------
# 1.  `clear_runs` -- the cut itself, with no rig and no planner
# ---------------------------------------------------------------------------
def test_clear_runs_keeps_the_clear_stretches_and_nothing_else():
    cum = np.linspace(0.0, 1.0, 11)          # 11 samples, 100 mm apart
    d = np.array([0.2, 0.2, 0.2, -0.1, -0.1, -0.1, 0.2, 0.2, 0.2, 0.2, 0.2])
    got = staged.clear_runs(d, cum, gate=0.05, min_len=0.01)
    assert got == [(0, 2), (6, 10)]


def test_a_stretch_shorter_than_the_minimum_piece_is_not_a_piece():
    """`traces.absorb_short` makes the same judgement about the DP's own runs.

    A 3 mm stroke is a pen-down, a pen-up and no picture, so a clear stretch
    below the minimum piece length is dropped rather than kept.
    """
    cum = np.linspace(0.0, 1.0, 11)
    d = np.array([0.2, 0.2, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, 0.2, 0.2])
    # the two clear stretches are 100 mm each
    assert staged.clear_runs(d, cum, 0.05, 0.05) == [(0, 1), (9, 10)]
    assert staged.clear_runs(d, cum, 0.05, 0.15) == []
    # ...and a single clear SAMPLE is never a stretch, whatever the minimum
    d2 = np.array([-0.1, 0.2, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1,
                   -0.1])
    assert staged.clear_runs(d2, cum, 0.05, 0.0) == []


def test_the_whole_piece_clear_is_one_run_and_the_whole_piece_buried_is_none():
    cum = np.linspace(0.0, 1.0, 11)
    assert staged.clear_runs(np.full(11, 0.2), cum, 0.05, 0.01) == [(0, 10)]
    assert staged.clear_runs(np.full(11, -0.2), cum, 0.05, 0.01) == []


def test_the_part_ids_are_unique_and_never_collide_with_a_dp_piece_id():
    """`(line, piece)` identifies ink everywhere downstream; parts need their own."""
    staged.split_ids_reset()
    ids = [staged._next_split_id() for _ in range(50)]
    assert len(set(ids)) == 50
    assert min(ids) > staged.SPLIT_ID0 - 1
    staged.split_ids_reset()
    assert staged._next_split_id() == ids[0]


# ---------------------------------------------------------------------------
# 2.  THE PROPOSED RIG: a real piece, half of it inside a room
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


@pytest.fixture(scope="module")
def half_buried(rig):
    """One certified piece of arm 71, with a room over its second half.

    -> (piece, plan, room-as-envelope, spec).  The room is a sphere centred on
    the pen tip three quarters of the way along the stroke, big enough to
    swallow that end of the ink and to leave the other end alone -- which is
    the shape §23.3 measured on the real follower and the shape the cut exists
    for.
    """
    arm = 71
    spec = rig[arm]
    # a 0.60 m stroke down the middle of arm 71's own row band (R1 is
    # y in [1.41, 2.22]), which the planner certifies end to end
    pts = np.column_stack([np.full(60, 0.90), np.linspace(2.10, 1.50, 60)])
    pc = staged.Piece(0, arm, 0, 0, pts,
                      float(T.cumlen(pts)[-1]), state=0, s0=0.0, s1=1.0)
    staged.thaw()
    paper.clear_cache()
    plan = staged.stroke_api.plan_stroke(
        pts, spec, dict(h_inv=1.0, pen_ext=spec.pen))
    assert plan.get("status") == "ok", plan.get("reason")
    P = coordination.chain_world(np.asarray(plan["qs"], float), spec, 1.0,
                                 float(spec.pen))
    tip = np.asarray(P)[int(0.90 * (len(P) - 1)), 9]
    env = {13: (tip[None, :].copy(), np.array([0.10]))}
    return pc, plan, env, spec


def _install(env, rig, arm=71):
    parks = staged.shipped_parks(rig)
    staged.freeze_stage(arm, parks, env, rig, {a: rig[a].pen for a in rig},
                        1.0, leg_cache=False)


def test_the_gate_refuses_the_whole_piece_and_the_cut_keeps_the_clear_half(
        rig, half_buried):
    """THE FINDING THE BUILD EXISTS FOR (docs/V2_STAGED.md section 23.3).

    `ink_vs_envelope` is a MINIMUM over the piece's poses, so a piece most of
    which is clear is still refused entire.  The cut turns the pose-wise
    fraction into flown ink: the stretches it keeps every pose of which clears
    the gate, the rest deferred.
    """
    pc, plan, env, spec = half_buried
    _install(env, rig)
    whole = staged.ink_vs_envelope(plan, spec, 1.0, spec.pen)
    assert whole < staged.PAIR_MARGIN, "the fixture must build a REFUSED piece"

    prof = staged.ink_clearance_profile(plan, spec, 1.0, spec.pen)
    frac = float(np.mean(prof >= staged.PAIR_MARGIN))
    assert 0.15 < frac < 0.95, f"half buried, not {100 * frac:.0f} % buried"

    staged.split_ids_reset()
    keep, drop, info = staged.split_at_room(pc, plan, spec, 1.0, spec.pen,
                                            staged.PAIR_MARGIN)
    assert keep, "the clear stretch must survive the cut"
    assert drop, "and the buried stretch must not"
    # nothing is invented and nothing is lost: the parts tile the parent
    assert info["keep_m"] + info["drop_m"] == pytest.approx(
        float(T.cumlen(np.asarray(plan["pts"], float))[-1]), rel=1e-6)
    assert all(p.length_m >= staged.SPLIT_MIN_M for p in keep)

    # EVERY POSE OF EVERY KEPT PART CLEARS THE GATE.  The parts are contiguous
    # slices of the parent's own dense points, so the claim is checkable
    # against the parent's profile without re-planning anything.
    dense = np.asarray(plan["pts"], float)
    for p in keep:
        i0 = int(np.argmin(np.linalg.norm(dense - p.pts[0], axis=1)))
        i1 = int(np.argmin(np.linalg.norm(dense - p.pts[-1], axis=1)))
        assert np.allclose(dense[i0:i1 + 1], p.pts)
        assert float(prof[i0:i1 + 1].min()) >= staged.PAIR_MARGIN
    # ...and the parts carry ids of their own, and their parent's line
    assert {p.line for p in keep + drop} == {pc.line}
    assert len({p.k for p in keep + drop}) == len(keep) + len(drop)
    assert all(p.k >= staged.SPLIT_ID0 for p in keep + drop)
    staged.thaw()


def test_the_minimum_length_rule_is_honoured_by_the_cut(rig, half_buried):
    """A minimum longer than the clear stretch keeps nothing and defers all."""
    pc, plan, env, spec = half_buried
    _install(env, rig)
    keep, drop, _ = staged.split_at_room(pc, plan, spec, 1.0, spec.pen,
                                         staged.PAIR_MARGIN, min_len=99.0)
    assert keep == []
    assert len(drop) == 1 and drop[0] is pc, \
        "nothing kept means the piece is deferred WHOLE, not re-cut"
    staged.thaw()


def test_a_new_end_without_a_certified_hover_is_not_offered(rig, half_buried,
                                                            monkeypatch):
    """THE PIECE-END RULE.  A cut makes two new pen-ups.

    `writing.arm_program` lifts to `lifted_or_lower`'s hover at every piece end
    and `paper.route` flies the leg from there; an end whose hover stands
    inside the leader's room is a leg the router refuses, and that costs the
    WHOLE bucket its timeline rather than the piece.  So the ends are gated
    before the part is ever offered.
    """
    pc, plan, env, spec = half_buried
    _install(env, rig)
    keep, _, info = staged.split_at_room(pc, plan, spec, 1.0, spec.pen,
                                         staged.PAIR_MARGIN)
    assert keep and info["hover_refused"] == 0
    monkeypatch.setattr(staged, "hover_clears", lambda *a, **k: False)
    keep2, drop2, info2 = staged.split_at_room(pc, plan, spec, 1.0, spec.pen,
                                               staged.PAIR_MARGIN)
    assert keep2 == []
    assert info2["hover_refused"] == len(keep)
    assert len(drop2) == 1 and drop2[0] is pc
    # ...and with the gate switched off the same stretches come back, which is
    # what says the refusal above was the hover and not the geometry
    keep3, _, _ = staged.split_at_room(pc, plan, spec, 1.0, spec.pen,
                                       staged.PAIR_MARGIN, hover_gate=False)
    assert len(keep3) == len(keep)
    staged.thaw()


def test_the_sweep_residual_makes_the_profile_stricter_than_the_raw_gate(
        rig, half_buried):
    """A clear stretch is a claim about the motion BETWEEN the ink samples.

    The room already carries the leader's own residual
    (`exact_room.from_samples`); this is the follower's half, and it can only
    make the profile smaller -- never larger, or the cut would be keeping ink
    the gate would refuse.
    """
    pc, plan, env, spec = half_buried
    _install(env, rig)
    P = coordination.chain_world(np.asarray(plan["qs"], float), spec, 1.0,
                                 float(spec.pen))
    from aris_sixarm import frozen
    raw = frozen.partner_clearance(P, floor=staged.INK_CAP)
    prof = staged.ink_clearance_profile(plan, spec, 1.0, spec.pen)
    assert np.all(prof <= raw + 1e-12)
    assert float(np.max(raw - prof)) > 0.0, "a moving arm has a residual"
    staged.thaw()


# ---------------------------------------------------------------------------
# 3.  THE ROW PARTITION -- which pieces a row conductor may take
# ---------------------------------------------------------------------------
def _piece(arm, y0, y1, line=0, k=0, x=0.9):
    pts = np.column_stack([np.full(8, x), np.linspace(y0, y1, 8)])
    return staged.Piece(2, arm, line, k, pts, float(T.cumlen(pts)[-1]),
                        state=0, s0=0.0, s1=1.0)


def test_row_arms_are_the_row_map_and_nothing_else():
    assert staged.ROW_ARMS == {0: (13, 17), 1: (31, 71), 2: (2, 97)}


def test_a_piece_inside_its_row_band_goes_to_that_row():
    for j, arms in staged.ROW_ARMS.items():
        _, y0, _, y1 = T.row_band(j)
        for a in arms:
            assert staged.piece_row(_piece(a, y0 + 0.02, y1 - 0.02)) == j


def test_a_piece_that_leaves_its_row_band_goes_to_no_row_conductor():
    """THE DEAD BAND IS THE WHOLE LICENCE FOR THREE CONDUCTORS.

    docs/V2_WORKCELLS.md section 4b measures cross-row pairs at +194.3 and
    +201.1 mm -- with each row drawing INSIDE ITS OWN BAND.  A piece that
    straddles the dead band is not that measurement's case, so it may never
    ride with a row conductor.
    """
    _, y0, _, y1 = T.row_band(1)
    # arm 31 reaching up out of row 1 into the dead band above it
    assert staged.piece_row(_piece(31, y1 - 0.05, y1 + 0.05)) is None
    assert staged.piece_row(_piece(31, y0 - 0.05, y0 + 0.05)) is None
    # ...and the same for an outer row, at either end
    _, z0, _, z1 = T.row_band(0)
    assert staged.piece_row(_piece(13, z1 - 0.05, z1 + 0.05)) is None
    assert staged.piece_row(_piece(2, 2.5, 2.7)) is None


def test_partition_sends_every_dead_band_piece_to_the_band_and_no_row():
    _, y0, _, y1 = T.row_band(1)
    _, z0, _, z1 = T.row_band(0)
    good1 = _piece(31, y0 + 0.02, y0 + 0.30, line=1, k=0)
    good2 = _piece(71, y1 - 0.30, y1 - 0.02, line=2, k=0)
    good0 = _piece(13, z0 + 0.02, z0 + 0.30, line=3, k=0)
    bad = _piece(31, y1 - 0.05, y1 + 0.10, line=4, k=0)
    rows, band, stats = staged.partition_deferred(
        {31: [good1, bad], 71: [good2], 13: [good0]})
    assert sorted(rows) == [0, 1]
    assert [p.line for p in rows[1][31]] == [1]
    assert [p.line for p in rows[1][71]] == [2]
    assert [p.line for p in rows[0][13]] == [3]
    assert [p.line for p in band[31]] == [4]
    # NO DEAD-BAND PIECE IS ANYWHERE IN A ROW GROUP -- the claim, stated once
    assert all(staged.piece_row(p) is not None
               for r in rows.values() for v in r.values() for p in v)
    assert stats["n_band"] == 1 and stats["n_row"] == 3
    assert stats["band_frac"] == pytest.approx(
        bad.length_m / (bad.length_m + good1.length_m + good2.length_m
                        + good0.length_m))


def test_an_arm_in_no_group_is_still_in_the_merged_check(rig):
    """A row conduct sees two arms; the whole-timeline check sees six.

    The merge is where that is re-established, and an arm in no group holds
    its pose for the length of the stage rather than vanishing from the scene.
    """
    parks = staged.shipped_parks(rig)
    st = staged.ArmStage(2, 71, np.asarray(parks[71], float).reshape(7))
    q = np.repeat(np.asarray(parks[71], float).reshape(1, 7), 12, axis=0)
    st.timeline = dict(t=0.05 * np.arange(12), q=q, seg=np.zeros(12, int),
                       u=np.zeros(12), phases=[], duration=0.55, draw_s=0.0,
                       transit_s=0.0)
    arms, rep = staged._merge_conducts(
        2, [({71: st}, dict(ok=True, min_clearance=1.0), 1.0)], rig,
        {a: rig[a].pen for a in rig}, parks, parks, 1.0, 0.05, 1)
    assert sorted(arms) == sorted(int(a) for a in rig)
    assert rep["n_frames"] == 12
    assert set(rep["frame_clearance"]) == {int(a) for a in rig}
    staged.thaw()
