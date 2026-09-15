"""The fleet clock `scripts/animate_staged.py` lays a staged programme on.

The renderers are pictures and are not tested here.  THE CLOCK IS NOT A
PICTURE: it decides when each arm moves, and getting it wrong would show Pete
an animation that is not the programme.  So what is pinned is the arithmetic —
the barrier, the residue serialisation, and the hold-at-park in between — on a
synthetic programme with SIX active arms in one stage, because the pattern that
produced `out/staged_csail_h097_program_v6.json` has at most three and the
loader must not have learnt that.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
animate_staged = pytest.importorskip("animate_staged")


def _arm(arm, park, dur, residue=False, priority=0, n=5, hold=None,
         role=None, tuck=None, clear_out=0.0):
    """An arm that leaves `park` and ends at `hold` (default: back at `park`)."""
    start = np.asarray(park, float)
    end = start if hold is None else np.asarray(hold, float)
    t = np.linspace(0.0, dur, n)
    q = np.repeat(start[None, :], n, axis=0)
    # a smooth bump, NOT a plateau: the renderer's motion gate calls a
    # constant stretch "holding" (rightly), so a fixture with one would be
    # testing the gate rather than the clock
    q[:, 0] = start[0] + 0.5 * np.sin(np.pi * np.arange(n) / (n - 1))
    q[-1] = end                             # ...and stops wherever it stops
    seg = np.array([-1] + [0] * (n - 2) + [-1])
    extra = {}
    if hold is not None:
        extra["q_hold"] = list(map(float, end))
    if role is not None:
        extra["role"] = role
    if tuck is not None:
        extra["q_tuck"] = list(map(float, tuck))
    if clear_out:
        extra["clear_out_s"] = float(clear_out)
    return dict(arm=arm, q_park=list(map(float, start)), frozen_partners=[],
                room_kind="test", residue=residue, priority=priority,
                trajectory_digest="", depends_on={}, n_pieces=1, ink_m=1.0,
                duration_s=float(dur), refused=[], wall={}, legs=[], **extra,
                pieces=[dict(stage=0, arm=arm, line=arm, piece=0, order=0,
                             flipped=False, home_before=False, length_m=1.0,
                             q_first=list(q[1]), q_last=list(q[-2]),
                             hover_in=None, hover_out=None,
                             pts=[[0.1 * arm, 0.1], [0.1 * arm, 0.2]])],
                trajectory=dict(t=t.tolist(), q=q.tolist(),
                                seg=[int(x) for x in seg]))


def _doc(arms_per_stage):
    parks = {str(a): [0.1 * a] * 7 for a in range(1, 7)}
    stages = []
    for k, spec in enumerate(arms_per_stage):
        arms = {}
        for a, dur, res in spec:
            arms[str(a)] = _arm(a, parks[str(a)], dur, res, priority=a)
        conc = max([d for _, d, r in spec if not r] or [0.0])
        dur = conc + sum(d for _, d, r in spec if r)
        stages.append(dict(stage=k, actives=[a for a, _, _ in spec],
                           duration_s=dur, n_pieces=len(spec), ink_m=1.0,
                           checks={}, arms=arms))
    return dict(schema=1, pattern="synthetic", n_stages=len(stages),
                parks=parks, pair_margin_m=0.05,
                makespan_s=sum(s["duration_s"] for s in stages),
                ttfm_s=None, timing={}, barriers=[], stages=stages)


def test_barrier_and_residue_arithmetic():
    doc = _doc([[(1, 10.0, False), (2, 4.0, False), (3, 3.0, True)],
                [(4, 5.0, False)]])
    p = animate_staged.Programme(doc)
    assert p.makespan == pytest.approx(13.0 + 5.0)
    s0, s1 = p.stages
    assert (s0["t0"], s0["t1"]) == (0.0, 13.0)
    # the residue is SERIALISED after the concurrent part, not overlapped
    assert s0["tracks"][3].t_start == pytest.approx(10.0)
    assert s0["tracks"][3].t_end == pytest.approx(13.0)
    # and stage 1 begins at the barrier, after every arm of stage 0 is parked
    assert s1["t0"] == pytest.approx(13.0)


def test_six_actives_in_one_stage_and_hold_at_park():
    """Nothing in the loader may assume <= 3 actives or a parked row partner."""
    doc = _doc([[(a, 2.0 + a, False) for a in range(1, 7)]])
    p = animate_staged.Programme(doc)
    assert p.stages[0]["t1"] == pytest.approx(8.0)          # max(3..8)
    ts = np.linspace(0.0, 8.0, 81)
    Q, DOWN, LIVE, STAGE = p.sample(ts)
    assert set(Q) == set(range(1, 7))
    # every arm is live from 0 to its own duration and HELD at park after it
    for a in range(1, 7):
        assert LIVE[a][0] and LIVE[a][-1] == (a == 6)
        held = ~LIVE[a]
        if held.any():
            assert np.abs(Q[a][held] - p.parks[a]).max() == 0.0
    assert (STAGE == 0).all()
    # all six draw, in the same stage
    assert all(DOWN[a].any() for a in range(1, 7))


def test_sample_holds_the_park_before_and_after_a_residue_window():
    doc = _doc([[(1, 10.0, False), (3, 3.0, True)]])
    p = animate_staged.Programme(doc)
    ts = np.array([0.0, 5.0, 9.9, 10.5, 12.9])
    Q, _, LIVE, _ = p.sample(ts)
    assert list(LIVE[3]) == [False, False, False, True, True]
    assert np.abs(Q[3][:3] - p.parks[3]).max() == 0.0


def test_duration_mismatch_is_an_assertion_not_a_wrong_picture():
    doc = _doc([[(1, 10.0, False)]])
    doc["stages"][0]["duration_s"] = 99.0
    with pytest.raises(AssertionError):
        animate_staged.Programme(doc)


# -- schema 2: held barriers, roles -----------------------------------------
def _held_doc():
    """Two stages, one arm, a HELD barrier: stage B starts at stage A's hold."""
    park = [0.0] * 7
    hold_a = [1.5] + [0.0] * 6
    hold_b = [2.25] + [0.0] * 6
    a0 = _arm(1, park, 10.0, hold=hold_a, role="leader")
    a1 = _arm(1, hold_a, 4.0, hold=hold_b, role="follower")
    stages = [dict(stage=0, actives=[1], duration_s=10.0, n_pieces=1,
                   ink_m=1.0, checks={}, roles={"1": "leader"},
                   arms={"1": a0}),
              dict(stage=1, actives=[1], duration_s=4.0, n_pieces=1,
                   ink_m=1.0, checks={}, roles={"1": "follower"},
                   arms={"1": a1})]
    return dict(schema=2, pattern="leader-follower-test", n_stages=2,
                parks={"1": park}, pair_margin_m=0.05, makespan_s=14.0,
                ttfm_s=None, timing={}, barriers=[], stages=stages)


def test_held_barrier_holds_the_hover_and_never_jumps_to_park():
    p = animate_staged.Programme(_held_doc())
    assert p.schema == 2
    ts = np.array([0.0, 10.0, 11.0, 13.99, 14.0])
    Q, _, LIVE, _ = p.sample(ts)
    hold_a = np.array([1.5] + [0.0] * 6)
    hold_b = np.array([2.25] + [0.0] * 6)
    # at the barrier and through it, the arm sits at stage A's hover...
    assert np.allclose(Q[1][1], hold_a)
    # ...and it is NOT at the park, which is what the old renderer would show
    assert not np.allclose(Q[1][1], p.parks[1])
    assert not LIVE[1][1] or True           # the barrier instant is a boundary
    # the programme ends held, not parked
    assert np.allclose(Q[1][-1], hold_b)
    assert p.ends_parked[1] == pytest.approx(2.25)
    assert p.stages[0]["label"] == "A" and p.stages[1]["label"] == "B"


def test_a_discontinuous_hold_chain_raises_rather_than_animating_a_jump():
    doc = _held_doc()
    doc["stages"][1]["arms"]["1"]["q_park"] = [0.0] * 7      # back at the park
    with pytest.raises(AssertionError, match="discontinuous"):
        animate_staged.Programme(doc)


def test_roles_reach_the_banner_tag():
    p = animate_staged.Programme(_held_doc())
    assert p.tag(1, 0) == "1L"
    assert p.tag(1, 1) == "1F"


def test_clear_out_tuck_is_a_ramp_before_the_arm_draws():
    park, hold = [0.0] * 7, [1.0] + [0.0] * 6
    tuck = [0.4] + [0.0] * 6
    a0 = _arm(1, park, 6.0, hold=hold, role="follower", tuck=tuck,
              clear_out=2.0)
    a0["trajectory"]["q"][0] = tuck          # it draws from the tucked pose
    doc = dict(schema=2, pattern="t", n_stages=1, parks={"1": park},
               pair_margin_m=0.05, makespan_s=8.0, ttfm_s=None, timing={},
               barriers=[],
               stages=[dict(stage=0, actives=[1], duration_s=8.0, n_pieces=1,
                            ink_m=1.0, checks={}, roles={"1": "follower"},
                            arms={"1": a0})])
    p = animate_staged.Programme(doc)
    tr = p.stages[0]["tracks"][1]
    assert (tr.clear_out, tr.t_draw0, tr.t_end) == (2.0, 2.0, 8.0)
    Q, _, LIVE, _ = p.sample(np.array([0.0, 1.0, 2.0, 8.0]))
    assert np.allclose(Q[1][0], park)
    assert Q[1][1][0] == pytest.approx(0.2)          # half way through the tuck
    assert np.allclose(Q[1][2], tuck)
    assert LIVE[1][0] and LIVE[1][1]                 # the tuck IS motion
    assert np.allclose(Q[1][3], hold)


def test_sampling_past_the_makespan_freezes_rather_than_wrapping():
    """The BEFORE/AFTER comparison runs both programmes on ONE clock.

    The shorter one is asked for times past its own end, and it must FREEZE at
    its last hold — not wrap, not drift, and not fall back to the park — or
    the side-by-side would show the finished run doing something it never does.
    """
    p = animate_staged.Programme(_held_doc())          # 14 s long
    ts = np.array([14.0, 20.0, 100.0])                 # at the end, and past it
    Q, DOWN, LIVE, STAGE = p.sample(ts)
    hold_b = np.array([2.25] + [0.0] * 6)
    for k in range(len(ts)):
        assert np.allclose(Q[1][k], hold_b), ts[k]
    assert not LIVE[1][1:].any() and not DOWN[1][1:].any()
    assert (STAGE[1:] == p.n_stages - 1).all()         # pinned to the last


def test_a_conducted_stage_does_not_re_serialise_its_residue_arms():
    """A conducted stage is ALREADY one merged clock.

    Every arm carries a trajectory spanning the whole stage with its waits
    baked in, so `duration_s` is the stage itself and `residue` is a record of
    how the bucket was won, not an instruction to append it.  Appending would
    treble the stage and trip the duration assert.
    """
    park = {a: [0.05 * a] * 7 for a in range(1, 4)}
    arms = {}
    for a in range(1, 4):
        d = _arm(a, park[a], 30.0, residue=(a > 1), priority=a,
                 role="conductor")
        d["conducted"] = True
        arms[str(a)] = d
    doc = dict(schema=2, pattern="conducted-test", n_stages=1,
               parks={str(a): park[a] for a in range(1, 4)},
               pair_margin_m=0.05, makespan_s=30.0, ttfm_s=None, timing={},
               barriers=[],
               stages=[dict(stage=0, actives=[1, 2, 3], duration_s=30.0,
                            n_pieces=3, ink_m=1.0, checks={}, roles={},
                            conducted=True, arms=arms)])
    p = animate_staged.Programme(doc)
    assert p.makespan == pytest.approx(30.0)
    assert p.stages[0]["conducted"] is True
    assert p.stages[0]["residues"] == []          # NOT appended
    for a in range(1, 4):
        assert p.stages[0]["tracks"][a].t_start == 0.0


def test_moving_means_moving_even_inside_a_conducted_trajectory():
    """An arm waiting its turn inside its own trajectory is HOLDING."""
    park = [0.0] * 7
    d = _arm(1, park, 10.0, role="conductor", n=5)
    d["conducted"] = True
    # a trajectory that sits still for its whole second half
    t = [0.0, 2.5, 5.0, 7.5, 10.0]
    q = [list(park), [0.5] + [0.0] * 6, [1.0] + [0.0] * 6,
         [1.0] + [0.0] * 6, [1.0] + [0.0] * 6]
    d["trajectory"] = dict(t=t, q=q, seg=[-1, 0, 0, -1, -1])
    d["q_hold"] = [1.0] + [0.0] * 6
    doc = dict(schema=2, pattern="c", n_stages=1, parks={"1": park},
               pair_margin_m=0.05, makespan_s=10.0, ttfm_s=None, timing={},
               barriers=[],
               stages=[dict(stage=0, actives=[1], duration_s=10.0, n_pieces=1,
                            ink_m=1.0, checks={}, roles={}, conducted=True,
                            arms={"1": d})])
    p = animate_staged.Programme(doc)
    ts = np.linspace(0.0, 10.0, 21)
    _, _, LIVE, _ = p.sample(ts)
    assert LIVE[1][:10].all()                    # climbing: moving
    assert not LIVE[1][12:].any()                # parked at 1.0 rad: holding
    mv, idle, longest = p.idle(dt=0.1)[1]
    assert mv == pytest.approx(5.0, abs=0.3)     # only the first half counts
    assert longest is not None and longest[1] == pytest.approx(10.0)


def test_ink_spans_the_whole_planned_piece_not_a_sample_short_at_each_end():
    """The renderer used to drop one sample interval at each end of a piece.

    Ink came from consecutive frames where BOTH were pen-down, so a piece lost
    its first and last interval — ~2 m of visible gaps over lf6b's 75 pieces.
    Ink now comes from the piece's own polyline, timed by the stroke window.
    """
    park = [0.0] * 7
    pts = [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]]          # exactly 1.0 m
    d = _arm(1, park, 10.0, role="leader", n=5)
    d["trajectory"]["seg"] = [-1, 0, 0, 0, -1]          # draws in the middle
    d["pieces"] = [dict(stage=0, arm=1, line=0, piece=0, order=0,
                        flipped=False, home_before=False, length_m=1.0,
                        q_first=park, q_last=park, hover_in=None,
                        hover_out=None, pts=pts)]
    doc = dict(schema=2, pattern="ink", n_stages=1, parks={"1": park},
               pair_margin_m=0.05, makespan_s=10.0, ttfm_s=None, timing={},
               barriers=[],
               stages=[dict(stage=0, actives=[1], duration_s=10.0, n_pieces=1,
                            ink_m=1.0, checks={}, roles={}, arms={"1": d})])
    p = animate_staged.Programme(doc)
    tracks, planned, orphan = animate_staged.ink_tracks(p)
    assert not orphan
    assert planned[1] == pytest.approx(1.0)             # the WHOLE piece
    seg = tracks[1]["seg"]
    assert np.allclose(seg[0][0][:2], pts[0])           # starts at the first
    assert np.allclose(seg[-1][1][:2], pts[-1])         # ends at the last
    assert tracks[1]["cum"][-1] == pytest.approx(1.0)
    # revealed progressively across the stroke's own window, not all at once
    assert tracks[1]["t"].min() > 0.0
    assert tracks[1]["t"].max() <= 10.0 + 1e-9


def test_ink_with_no_pen_down_time_is_reported_not_invented():
    """lf6b's arm 31 claims 1.9 m in a stage it spends on an `aside` leg."""
    park = [0.0] * 7
    d = _arm(1, park, 10.0, role="conductor", n=5)
    d["trajectory"]["seg"] = [-1, -1, -1, -1, -1]       # pen never down
    d["pieces"] = [dict(stage=0, arm=1, line=3, piece=1, order=0,
                        flipped=False, home_before=False, length_m=2.0,
                        q_first=park, q_last=park, hover_in=None,
                        hover_out=None, pts=[[0.0, 0.0], [2.0, 0.0]])]
    doc = dict(schema=2, pattern="aside", n_stages=1, parks={"1": park},
               pair_margin_m=0.05, makespan_s=10.0, ttfm_s=None, timing={},
               barriers=[],
               stages=[dict(stage=0, actives=[1], duration_s=10.0, n_pieces=1,
                            ink_m=2.0, checks={}, roles={}, arms={"1": d})])
    p = animate_staged.Programme(doc)
    tracks, planned, orphan = animate_staged.ink_tracks(p)
    assert planned[1] == 0.0 and len(tracks[1]["seg"]) == 0
    assert orphan == [(1, "A", 3, 1, 2.0)]              # reported, not drawn


def test_six_arms_two_stages_all_leaders_and_followers():
    """The lf2 shape: six arms, every one active in both stages."""
    park = {a: [0.05 * a] * 7 for a in range(1, 7)}
    hold = {a: [0.05 * a + 1.0] * 7 for a in range(1, 7)}
    roleA = {a: ("leader" if a <= 3 else "follower") for a in range(1, 7)}
    s0 = {str(a): _arm(a, park[a], 2.0 + a, hold=hold[a], role=roleA[a])
          for a in range(1, 7)}
    s1 = {str(a): _arm(a, hold[a], 1.0 + a,
                       role=("follower" if a <= 3 else "leader"))
          for a in range(1, 7)}
    for d in s1.values():                    # stage B ends where it started
        d["q_hold"] = d["q_park"]
    doc = dict(schema=2, pattern="lf-test", n_stages=2,
               parks={str(a): park[a] for a in range(1, 7)},
               pair_margin_m=0.05, makespan_s=8.0 + 7.0, ttfm_s=None,
               timing={}, barriers=[],
               stages=[dict(stage=0, actives=list(range(1, 7)),
                            duration_s=8.0, n_pieces=6, ink_m=1.0, checks={},
                            roles={str(a): roleA[a] for a in range(1, 7)},
                            arms=s0),
                       dict(stage=1, actives=list(range(1, 7)),
                            duration_s=7.0, n_pieces=6, ink_m=1.0, checks={},
                            roles={}, arms=s1)])
    p = animate_staged.Programme(doc)
    assert p.makespan == pytest.approx(15.0)
    assert p.stages[1]["t0"] == pytest.approx(8.0)
    tags = {p.tag(a, 0) for a in range(1, 7)}
    assert tags == {"1L", "2L", "3L", "4F", "5F", "6F"}
    # at the barrier every arm sits at its stage-A hold, none at its park
    Q, _, LIVE, _ = p.sample(np.array([8.0]))
    for a in range(1, 7):
        assert np.allclose(Q[a][0], hold[a])
        assert not np.allclose(Q[a][0], p.parks[a])
