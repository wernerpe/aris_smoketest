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


def _arm(arm, park, dur, residue=False, priority=0, n=5):
    """A straight-line-in-joint-space arm that leaves its park and returns."""
    t = np.linspace(0.0, dur, n)
    q = np.repeat(np.asarray(park, float)[None, :], n, axis=0)
    q[1:-1, 0] += 0.5                       # it moves, and it comes back
    seg = np.array([-1] + [0] * (n - 2) + [-1])
    return dict(arm=arm, q_park=list(map(float, park)), frozen_partners=[],
                room_kind="test", residue=residue, priority=priority,
                trajectory_digest="", depends_on={}, n_pieces=1, ink_m=1.0,
                duration_s=float(dur), refused=[], wall={}, legs=[],
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
