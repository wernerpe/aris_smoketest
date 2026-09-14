"""HOW THE ROW CONDUCTORS COMPOSE, and the self gate the judge could not read.

Three claims, measured on the lf3 s150 programme and pinned here
(docs/V2_STAGED.md section 26):

  1. THE CONDUCT'S ROOM SURVIVES THE HANDOVER.  `idle.conduct` does not re-time
     the timelines `plan_bucket` built, it calls `writing.arm_program` again and
     ROUTES EVERY PEN-UP LEG AGAIN.  `_conduct_stage` used to thaw the frozen
     set before handing over, so a two-arm row conductor emitted legs that were
     routed against nothing but the pose-invariant base columns -- and arm 31
     flew through arm 2's standing chain at -204.6 mm.  The arms a conduct does
     NOT move are now in the room for the length of it.
  2. THE GROUPS COMPOSE IN A PRIORITY ORDER.  Busiest group first, and every
     later group sees the earlier groups' REALISED trajectories as exact swept
     rooms -- the same argument `_priority_stage` makes between arms, one level
     up.  `serial` is the control: the groups run one after another in time.
  3. THE PRODUCER PAYS THE JUDGE'S PLAYBACK RESIDUAL.  A pen-up leg whose true
     self-clearance is 26.1 mm reads +16.2 mm to `scene_check`, which samples
     the timeline that is written down and charges 0.55 x its frame step.  The
     leg is not close to itself; it is fast.  `writing.self_pace_beat` stretches
     exactly the beats that owe the residual and nothing else.

NO ENVIRONMENT VARIABLES.  The rig and the tool are switched in process by the
`rig` fixture and put back, exactly as `tests/test_staged.py` does it.
"""
import numpy as np
import pytest

from aris_sixarm import fleet as fleet_mod
from aris_sixarm import frames, frozen, paper, scene_check, selfcoll
from aris_sixarm import staged, writing
from aris_sixarm import exact_room


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


def _stage(arm, q, n=8, dt=0.05):
    """An `ArmStage` holding one pose for `n` frames. -> ArmStage."""
    st = staged.ArmStage(2, int(arm), np.asarray(q, float).reshape(7))
    Q = np.repeat(np.asarray(q, float).reshape(1, 7), n, axis=0)
    st.role, st.conducted = "conductor", True
    st.timeline = dict(t=dt * np.arange(n), q=Q, seg=-np.ones(n, int),
                       u=np.zeros(n), phases=[], duration=dt * (n - 1),
                       draw_s=0.0, transit_s=0.0)
    return st


# ---------------------------------------------------------------------------
# 1.  THE ORDER
# ---------------------------------------------------------------------------
def test_group_order_is_busiest_first_and_counts_both_sources():
    """A group's ink is its arms' deferred ink AND its own stage bucket."""
    def pc(arm, m):
        pts = np.array([[0.0, 0.0], [float(m), 0.0]])
        return staged.Piece(stage=2, arm=arm, line=0, k=0, pts=pts,
                            length_m=float(m))

    groups = [(13, 17), (31, 71), (2, 97)]
    deferred = {31: [pc(31, 5.0)], 17: [pc(17, 1.0)]}
    buckets = {(2, 97): [pc(97, 2.0)]}
    assert staged.group_order(groups, deferred, buckets, 2) == [
        (31, 71), (2, 97), (13, 17)]


def test_group_order_breaks_ties_on_the_arm_ids_not_on_dict_order():
    groups = [(31, 71), (13, 17)]
    assert staged.group_order(groups, {}, {}, 2) == [(13, 17), (31, 71)]
    assert staged.group_order(list(reversed(groups)), {}, {}, 2) == [
        (13, 17), (31, 71)]


def test_an_unknown_compose_mode_is_refused_rather_than_guessed(rig):
    with pytest.raises(ValueError):
        staged._conduct_groups(2, [(13, 17)], {}, {}, rig, None, {}, {}, 0.97,
                               None, False, None, 0.05, 2, 1, 1.0, False,
                               mode="whatever")


# ---------------------------------------------------------------------------
# 2.  THE ROOM A CONDUCT FLIES IN
# ---------------------------------------------------------------------------
def test_freeze_conduct_holds_the_arms_outside_the_group_and_nobody_else(rig):
    """The movers are absent from the room; the four others are in it."""
    parks = staged.shipped_parks(rig)
    got = staged.freeze_conduct([2, 13, 17, 97], parks, rig,
                                {a: rig[a].pen for a in rig}, 0.97,
                                leg_cache=False)
    assert got == (2, 13, 17, 97)
    assert frozen.frozen_ids() == [2, 13, 17, 97]
    assert frozen.observer() is None          # no mover to exclude
    assert frozen.room_kinds() == {2: "pose", 13: "pose", 17: "pose",
                                   97: "pose"}
    staged.thaw()


def test_an_earlier_groups_trajectory_enters_the_conduct_as_an_exact_room(rig):
    """A mover of an earlier priority group is its SWEPT VOLUME, not a pose."""
    parks = staged.shipped_parks(rig)
    pens = {a: rig[a].pen for a in rig}
    st = _stage(71, parks[71], n=6)
    room = staged.trajectory_room(st, rig, pens, 0.97)
    assert isinstance(room[0], exact_room.ExactRoom)
    staged.freeze_conduct([2, 71, 97], parks, rig, pens, 0.97,
                          rooms={71: room}, leg_cache=False)
    kinds = frozen.room_kinds()
    assert kinds[71] == "capsules" and kinds[2] == "pose"
    staged.thaw()


def test_the_conduct_keeps_its_room_installed_while_idle_conducts(rig,
                                                                 monkeypatch):
    """THE DEFECT, PINNED.  `idle.conduct` routes; it must see the room.

    `_conduct_stage` used to `thaw()` before handing over, so every leg the
    conductor emitted was routed against the base columns alone -- which is how
    a certified stage-C trajectory came to pass through a standing arm at
    -204.6 mm.  The room the four outside arms make has to still be installed
    when the conductor asks for a route.
    """
    from aris_sixarm import idle as idle_mod
    parks = staged.shipped_parks(rig)
    pens = {a: rig[a].pen for a in rig}
    held = {int(a): np.asarray(q, float).reshape(7) for a, q in parks.items()}
    held[31] = held[31].copy()
    held[31][3] += 2e-3                    # so the stage has something to do
    seen = {}

    def spy(*args, **kw):
        seen["ids"] = frozen.frozen_ids()
        seen["observer"] = frozen.observer()
        raise idle_mod.Unconductable("stop here")

    monkeypatch.setattr(idle_mod, "conduct", spy)
    arms, rep, _ = staged._conduct_stage(
        2, {}, {}, rig, pens, held, parks, 0.97, None, False, None, None,
        0.05, 2, False, only=(31, 71))
    assert seen["ids"] == [2, 13, 17, 97], seen
    assert seen["observer"] is None
    # ...and the room is given back before the check is asked anything
    assert frozen.frozen_ids() == []
    assert set(arms) == {31, 71}
    assert rep.get("serialised") is True
    staged.thaw()


# ---------------------------------------------------------------------------
# 3.  THE MERGE: t = 0, AND THE SERIAL CLOCK
# ---------------------------------------------------------------------------
def test_the_merge_proves_the_six_entry_poses_pairwise_at_t_zero(rig):
    """`hold_gap` over the merged timeline's first frame, before anything else."""
    parks = staged.shipped_parks(rig)
    pens = {a: rig[a].pen for a in rig}
    part = ({71: _stage(71, parks[71])}, dict(ok=True, min_clearance=1.0), 1.0)
    _, rep = staged._merge_conducts(2, [part], rig, pens, parks, parks, 0.97,
                                    0.05, 1)
    assert rep["t0_holds"]["ok"] is True
    assert rep["t0_holds"]["min_m"] > staged.PAIR_MARGIN
    staged.thaw()


def test_a_merge_whose_entry_poses_already_touch_is_refused_at_t_zero(rig):
    """The proof is a gate, not a print: a bad t = 0 fails the merge."""
    parks = staged.shipped_parks(rig)
    pens = {a: rig[a].pen for a in rig}
    bad = dict(parks)
    bad[31] = np.asarray(parks[71], float).reshape(7)   # two arms, one pose
    part = ({71: _stage(71, bad[71])}, dict(ok=True, min_clearance=1.0), 1.0)
    _, rep = staged._merge_conducts(2, [part], rig, pens, bad, parks, 0.97,
                                    0.05, 1)
    assert rep["t0_holds"]["ok"] is False
    assert "t0_holds" in rep["failed"] and rep["ok"] is False
    staged.thaw()


def test_serial_offsets_put_the_second_group_after_the_first_in_time(rig):
    """`serial` is the control: the rows' MOTION is the sum, not the max."""
    parks = staged.shipped_parks(rig)
    pens = {a: rig[a].pen for a in rig}
    q71 = np.asarray(parks[71], float).reshape(7)
    q13 = np.asarray(parks[13], float).reshape(7)
    moved = q13.copy()
    moved[3] += 0.05
    g1 = ({71: _stage(71, q71, n=8)}, dict(ok=True, min_clearance=1.0), 1.0)
    st13 = _stage(13, moved, n=6)
    st13.timeline["q"][0] = q13             # it starts where it was standing
    g2 = ({13: st13}, dict(ok=True, min_clearance=1.0), 1.0)
    arms, rep = staged._merge_conducts(
        2, [g1, g2], rig, pens, parks, parks, 0.97, 0.05, 1,
        offsets={71: 0, 13: 7})
    q = np.asarray(arms[13].timeline["q"], float)
    assert len(q) == 13                       # 7 held frames, then its 6
    assert np.allclose(q[:7], q13)            # holds its entry pose until then
    assert np.allclose(q[-1], moved)          # and its finishing pose after
    assert rep["makespan_s"] == pytest.approx(0.05 * 12)
    staged.thaw()


def test_a_group_the_conductor_refused_is_laid_down_one_arm_at_a_time(rig):
    """`serialised` means each arm has its OWN clock, and the merge must say so.

    Measured 2026-09-14: group [2, 97] refuses `idle.conduct` -- a same-row pair
    whose nominal timelines cannot share a clock -- and the fallback flies them
    in turn.  Merged from frame zero they read **-208.3 mm** against each other,
    which is a programme nobody was ever going to run.
    """
    parks = staged.shipped_parks(rig)
    pens = {a: rig[a].pen for a in rig}
    q2 = np.asarray(parks[2], float).reshape(7)
    q97 = np.asarray(parks[97], float).reshape(7)
    a2, a97 = _stage(2, q2, n=5), _stage(97, q97, n=4)
    for st in (a2, a97):
        st.residue, st.conducted = True, False
    part = ({2: a2, 97: a97}, dict(ok=False, serialised=True,
                                   min_clearance=0.02), 1.0)
    arms, rep = staged._merge_conducts(2, [part], rig, pens, parks, parks, 0.97,
                                       0.05, 1)
    assert rep["serialised_groups"] == [[2, 97]]
    # arm 2 flies frames 0..4, arm 97 flies 4..7, and neither moves in the
    # other's window
    assert len(np.asarray(arms[2].timeline["q"], float)) == 8
    assert rep["offsets_s"] == {"97": 0.2}
    assert rep["makespan_s"] == pytest.approx(0.05 * 7)
    staged.thaw()


def test_priority_hands_every_earlier_groups_room_to_every_later_one(rig,
                                                                    monkeypatch):
    """Group k's payload carries a room for every arm groups 1..k-1 moved."""
    parks = staged.shipped_parks(rig)
    pens = {a: rig[a].pen for a in rig}
    order = [(31, 71), (2, 97), (13, 17)]
    saw = []

    def fake(pay, cap_s):
        who = pay[1]
        saw.append((tuple(who), sorted((pay[15] or {}))))
        return ({int(a): _stage(a, parks[a]) for a in who},
                dict(ok=True, min_clearance=1.0), 1.0), None

    monkeypatch.setattr(staged, "_one_group", fake)
    pcs = {31: [staged.Piece(2, 31, 0, 0, np.array([[0.0, 0.0], [9.0, 0.0]]),
                             9.0)],
           2: [staged.Piece(2, 2, 1, 0, np.array([[0.0, 0.0], [5.0, 0.0]]),
                            5.0)]}
    _, rep, _ = staged._conduct_groups(
        2, [(13, 17), (31, 71), (2, 97)], {}, pcs, rig, pens, parks, parks,
        0.97, None, False, None, 0.05, 2, 3, 60.0, False, mode="priority")
    assert [w for w, _ in saw] == order
    assert saw[0][1] == []                      # the busiest group plans free
    assert saw[1][1] == [31, 71]                # ...then against what it flew
    assert saw[2][1] == [2, 31, 71, 97]         # ...and so on, cumulatively
    assert rep["compose"] == "priority"
    assert rep["group_order"] == [[31, 71], [2, 97], [13, 17]]
    staged.thaw()


# ---------------------------------------------------------------------------
# 4.  THE SELF GATE THE JUDGE COULD NOT READ
# ---------------------------------------------------------------------------
# The leg, verbatim: arm 31's stage-C transit `seg 13` of
# `out/staged_csail_h097_lf3_s150_program.json`, the two conducted frames that
# bracket its worst self-clearance (t = 86.35 s and t = 86.40 s at dt = 0.05).
LEG_31 = (
    [0.55679, 1.018816, -2.081408, -2.344109, -1.464539, 2.091105, -1.343707],
    [0.572427, 0.979516, -2.055671, -2.34825, -1.401158, 2.090409, -1.308204],
)


def _judge(spec, q0, q1, dt, pen, dt_play=writing.SELF_PLAY_DT):
    """`scene_check`'s reading of one straight move, restated here.

    The playback frames of a move of duration `dt` are `dt_play` apart, and the
    judge charges `0.55 x` the capsule-endpoint travel between two of them.
    """
    n = max(2, int(round(float(dt) / float(dt_play))) + 1)
    Q = np.asarray(q0, float) + np.linspace(0.0, 1.0, n)[:, None] * (
        np.asarray(q1, float) - np.asarray(q0, float))
    A, B, R = selfcoll.capsule_ends(Q, pen)
    c = selfcoll.clearance_screened(A, B, R, 0.30)
    X = np.concatenate([A, B], axis=1)
    step = float(np.linalg.norm(np.diff(X, axis=0), axis=2).max())
    return float(c.min() - writing.SELF_PLAY_K * step)


def test_the_leg_that_failed_is_not_close_to_itself_it_is_fast(rig):
    """26.1 mm of real clearance, refused at +16.2 mm by the residual alone."""
    pen = scene_check.pen_len(None, 31)
    q0, q1 = (np.asarray(q, float) for q in LEG_31)
    Q = q0 + np.linspace(0.0, 1.0, 401)[:, None] * (q1 - q0)
    A, B, R = selfcoll.capsule_ends(Q, pen)
    true_min = float(selfcoll.clearance_screened(A, B, R, 0.30).min())
    assert true_min == pytest.approx(0.0262, abs=5e-4)
    assert true_min > selfcoll.SELF_PLAN_MARGIN      # the ROUTER was right
    assert _judge(rig[31], q0, q1, 0.05, pen) < scene_check.SELF_MARGIN


def test_pacing_that_leg_buys_it_the_judges_margin_and_nothing_else(rig):
    """The geometry does not move; the beat gets longer until the bound fits."""
    pen = scene_check.pen_len(None, 31)
    q0, q1 = LEG_31
    dt2, why = writing.self_pace_beat(rig[31], q0, q1, 0.05, pen)
    assert why == "paced" and dt2 > 0.05
    assert dt2 < writing.SELF_PACE_MAX * 0.05
    assert _judge(rig[31], q0, q1, dt2, pen) >= scene_check.SELF_MARGIN
    # ...and once it fits, asking again buys nothing more
    assert writing.self_pace_beat(rig[31], q0, q1, dt2, pen)[1] is None


def test_a_leg_with_room_to_spare_is_not_slowed_down(rig):
    """Only the beats that owe the residual pay it."""
    parks = staged.shipped_parks(rig)
    q0 = np.asarray(parks[31], float).reshape(7)
    q1 = q0.copy()
    q1[3] += 0.05
    dt, why = writing.self_pace_beat(rig[31], q0, q1, 0.4,
                                     scene_check.pen_len(None, 31))
    assert why is None and dt == pytest.approx(0.4)


def test_self_pace_block_never_shortens_a_beat(rig):
    """A pacing pass is monotone in time, so a priced tour is never undersold."""
    parks = staged.shipped_parks(rig)
    q0 = np.asarray(parks[31], float).reshape(7)
    steps = [(0.05, np.asarray(LEG_31[0], float)),
             (0.05, np.asarray(LEG_31[1], float)),
             (0.30, q0)]
    out, n, added = writing.self_pace_block(rig[31], q0, steps,
                                            scene_check.pen_len(None, 31))
    assert len(out) == len(steps)
    assert all(b[0] >= a[0] - 1e-12 for a, b in zip(steps, out))
    assert added == pytest.approx(sum(b[0] - a[0]
                                      for a, b in zip(steps, out)), abs=1e-9)
    assert n >= 1
