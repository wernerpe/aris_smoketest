"""The three pen-up fixes of 2026-09-14, each pinned at the level it acts on.

  1. `scripts/penup_anatomy.py`'s classifier, on a synthetic leg of each class.
  2. `allocate.chain_sheets`, the continuity Viterbi, on a toy chain: it must
     take the continuous sheet when the costs say so, and it must NEVER take an
     alternative the caller's certification test refused.
  3. `paper.route`'s shortcut acceptance rule: a drop is kept only when the
     shortened shape re-certifies, and never when it costs more.

No rig environment is needed by any of them -- the classifier and the DP run on
arrays and stubs, which is the point of keeping the rule separable from the FK.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import penup_anatomy as pa                                  # noqa: E402
from aris_sixarm import allocate, paper, writing            # noqa: E402


# ---------------------------------------------------------------------------
# 1.  THE CLASSIFIER
# ---------------------------------------------------------------------------
def _leg(dq_per_step, xyz):
    """A synthetic leg: joint steps along joint 0, tip at `xyz`. -> (Q, X)."""
    X = np.asarray(xyz, float).reshape(-1, 3)
    Q = np.zeros((len(X), 7))
    Q[:, 0] = np.cumsum([0.0] + list(dq_per_step))[:len(X)]
    return Q, X


def test_honest_hop_is_honest():
    # lift 6 cm, cross 20 cm, lower: 0.32 m of tip travel, 4 rad of joints,
    # which is 12.5 rad/m -- what an honest pen-up costs.
    Q, X = _leg([1.3, 1.4, 1.3],
                [[0, 0, 0], [0, 0, 0.06], [0.20, 0, 0.06], [0.20, 0, 0]])
    m = pa.leg_metrics(Q, X, 1.5)
    c = pa.classify(m)
    assert c["kind"] == "honest", (m, c)


def test_redundancy_flip_is_a_flip():
    # 3 mm of hop, a 6 cm lift and lower, and 20 rad of joint path: arm 71's
    # leg 9.  Nothing in task space explains the joints.
    Q, X = _leg([9.9, 0.2, 9.9],
                [[0, 0, 0], [0, 0, 0.06], [0.003, 0, 0.06], [0.003, 0, 0]])
    m = pa.leg_metrics(Q, X, 7.2)
    c = pa.classify(m)
    assert c["kind"] == "flip", (m, c)
    assert c["flip_rad"] > pa.FLIP_RAD
    assert c["climb_rad"] == 0.0        # it did not climb; it folded


def test_tall_route_is_tall():
    # 20 cm hop flown over a 32 cm via: 64 cm of vertical travel where 12 cm
    # would do, and the joint path is what that climb costs and no more.
    Q, X = _leg([6.0, 5.0, 6.0],
                [[0, 0, 0], [0, 0, 0.32], [0.20, 0, 0.32], [0.20, 0, 0]])
    m = pa.leg_metrics(Q, X, 7.5)
    c = pa.classify(m)
    assert c["kind"] == "tall", (m, c)
    assert c["climb_excess_m"] == pytest.approx(0.64 - pa.ZTRAV_REF)


def test_wandering_route_is_tall_class():
    # a there-and-back at hover height: the tip flies out to 1.0 m and comes
    # back to 0.5 m, so the joint path IS explained by the tip travel (no flip)
    # but it is 7x the net joint change, which is the wander test.
    Q, X = _leg([4.0, -3.0, 4.0, -3.0],
                [[0, 0, 0], [0, 0, 0.06], [1.0, 0, 0.06], [0.5, 0, 0.06],
                 [0.5, 0, 0]])
    m = pa.leg_metrics(Q, X, 7.6)
    c = pa.classify(m)
    assert c["flip_rad"] == 0.0, m           # the tip travel explains it
    assert m["dq_rad"] > pa.WANDER_RATIO * m["net_rad"]
    assert c["kind"] == "tall", (m, c)


def test_metrics_are_what_they_say():
    Q, X = _leg([1.0, 2.0], [[0, 0, 0], [0, 0, 0.06], [0.3, 0, 0.0]])
    m = pa.leg_metrics(Q, X, 2.0)
    assert m["dq_rad"] == pytest.approx(3.0)
    assert m["net_rad"] == pytest.approx(3.0)
    assert m["hop_m"] == pytest.approx(0.3)
    assert m["ztrav_m"] == pytest.approx(0.12)
    assert m["zmax_m"] == pytest.approx(0.06)


def test_a_leg_needs_two_samples():
    with pytest.raises(ValueError):
        pa.leg_metrics(np.zeros((1, 7)), np.zeros((1, 3)), 1.0)


# ---------------------------------------------------------------------------
# 2.  THE CHAIN DP
# ---------------------------------------------------------------------------
class _StubMenu:
    """A menu whose variants are handed in, and whose plans are handed back."""

    status, reason = "ok", "stub"

    def __init__(self, variants, plans):
        self.variants, self.plans = variants, plans

    def __len__(self):
        return len(self.variants)

    def materialize(self, k, opts=None):
        return self.plans[k]


def _plan(entry, exit_, pts):
    qs = np.zeros((2, 7))
    qs[0], qs[-1] = entry, exit_
    return dict(status="ok", qs=qs, pts=np.asarray(pts, float),
                arc_len=float(np.linalg.norm(np.diff(pts, axis=0))),
                total_time=1.0)


def _variant(entry, exit_, pts, surcharge=0.0):
    return dict(sheet=0, j0=0, j1=0, q7_0=0.0, q7_1=0.0,
                entry_q=np.asarray(entry, float),
                exit_q=np.asarray(exit_, float),
                entry_xy=np.asarray(pts[0], float),
                exit_xy=np.asarray(pts[-1], float),
                travel=0.0, surcharge=float(surcharge))


def _chain_fixture(bad=()):
    """Two pieces.  Piece 0 is fixed; piece 1 has a FAR plan and a NEAR one.

    Piece 1's certified plan (alternative 0) sits 4 rad from piece 0's exit;
    its menu offers a variant 0.1 rad away.  Continuity has to pick the near
    one, and `bad` marks alternatives the acceptance test refuses.
    """
    far = np.full(7, 4.0)
    near = np.full(7, 0.1)
    zero = np.zeros(7)
    p0 = _plan(zero, zero, [[0.0, 0.0], [0.05, 0.0]])
    p1_far = _plan(far, far, [[0.06, 0.0], [0.11, 0.0]])
    p1_near = _plan(near, near, [[0.06, 0.0], [0.11, 0.0]])
    segs = [dict(stroke_id=0, seg=0, plan=p0, pts=p0["pts"],
                 length=0.05, home_before=False),
            dict(stroke_id=1, seg=1, plan=p1_far, pts=p1_far["pts"],
                 length=0.05, home_before=False)]
    menus = [_StubMenu([_variant(zero, zero, p0["pts"])], [p0]),
             _StubMenu([_variant(far, far, p1_far["pts"]),
                        _variant(near, near, p1_near["pts"])],
                       [p1_far, p1_near])]
    seen = []

    def accept(plan, i):
        seen.append((i, float(np.asarray(plan["qs"])[0][0])))
        return (i, round(float(np.asarray(plan["qs"])[0][0]), 3)) not in bad
    return segs, menus, accept, seen


def test_chain_dp_takes_the_continuous_sheet(monkeypatch):
    monkeypatch.setattr(writing, "lifted_or_lower",
                        lambda spec, q, xy, **kw: (np.asarray(q, float), 0.06))
    segs, menus, accept, _ = _chain_fixture()
    out, st = allocate.chain_sheets(segs, spec=None, opts={},
                                    seq_opts=dict(qd_frac=0.6), menus=menus,
                                    accept=accept)
    assert st["n_changed"] == 1
    assert np.allclose(np.asarray(out[1]["plan"]["qs"])[0], 0.1)
    assert st["after_rad"] < st["before_rad"]
    assert st["after_s"] <= st["before_s"]


def test_chain_dp_never_takes_an_uncertified_alternative(monkeypatch):
    monkeypatch.setattr(writing, "lifted_or_lower",
                        lambda spec, q, xy, **kw: (np.asarray(q, float), 0.06))
    # the near variant is exactly the one the caller refuses
    segs, menus, accept, seen = _chain_fixture(bad={(1, 0.1)})
    out, st = allocate.chain_sheets(segs, spec=None, opts={},
                                    seq_opts=dict(qd_frac=0.6), menus=menus,
                                    accept=accept)
    assert seen, "the acceptance test was never asked"
    assert st["n_changed"] == 0
    assert np.allclose(np.asarray(out[1]["plan"]["qs"])[0], 4.0)
    assert out[1]["plan"] is segs[1]["plan"]


def test_chain_dp_is_a_no_op_on_one_piece(monkeypatch):
    monkeypatch.setattr(writing, "lifted_or_lower",
                        lambda spec, q, xy, **kw: (np.asarray(q, float), 0.06))
    segs, menus, accept, _ = _chain_fixture()
    out, st = allocate.chain_sheets(segs[:1], spec=None, opts={},
                                    seq_opts={}, menus=menus[:1],
                                    accept=accept)
    assert out == segs[:1] and st["n_changed"] == 0


def test_chain_edge_is_the_capped_transit_time(monkeypatch):
    """The DP's edge is `writing.transit_time`'s three beats and nothing else."""
    monkeypatch.setattr(writing, "lifted_or_lower",
                        lambda spec, q, xy, **kw: (np.asarray(q, float), 0.06))
    a = dict(exit_q=np.zeros(7), exit_xy=np.zeros(2))
    b = dict(entry_q=np.full(7, 0.5), entry_xy=np.array([0.2, 0.0]))
    got = allocate.chain_edge_s(None, a, b, 0.97, None,
                                writing.TRANSIT_SPEED, 0.6,
                                lambda q, xy: np.asarray(q, float))
    want = sum(writing.transit_time(a["exit_q"], a["exit_q"], b["entry_q"],
                                    b["entry_q"], 0.2,
                                    writing.TRANSIT_SPEED, 0.6))
    assert got == pytest.approx(want)


# ---------------------------------------------------------------------------
# 3.  THE SHORTCUT ACCEPTANCE RULE
# ---------------------------------------------------------------------------
def _shorten(q0, q1, seq, certifies):
    """`paper.route`'s `shorten`, rebuilt on stubs so the RULE can be tested.

    Mirrors the implementation: drop one via at a time, keep the drop that
    saves the most time, and only ever keep a drop that `legs_ok` certifies.
    """
    def price(s):
        qs = [q0] + list(s) + [q1]
        from aris_sixarm.frames import QD_MAX
        return float(sum(np.max(np.abs(np.asarray(b) - np.asarray(a)) / QD_MAX)
                         for a, b in zip(qs[:-1], qs[1:])))

    cur, n_cut = list(seq), 0
    for _ in range(paper.SHORTCUT_ROUNDS):
        base, best = price(cur), None
        for i in range(len(cur)):
            cand = cur[:i] + cur[i + 1:]
            p = price(cand)
            if p >= base - 1e-9 or not certifies(cand):
                continue
            if best is None or p < best[0]:
                best = (p, cand)
        if best is None:
            break
        cur, n_cut = best[1], n_cut + 1
    return cur, n_cut


def test_shortcut_drops_a_via_when_the_shape_recertifies():
    q0, q1 = np.zeros(7), np.zeros(7)
    q1[0] = 0.4
    tall = np.zeros(7)
    tall[0] = 3.0                       # a detour that costs and buys nothing
    mid = np.zeros(7)
    mid[0] = 0.2
    out, n = _shorten(q0, q1, [tall, mid], lambda s: True)
    # the tall detour goes; the collinear one stays, because dropping it saves
    # nothing and a drop that saves nothing is not taken
    assert n == 1
    assert len(out) == 1 and out[0][0] == pytest.approx(0.2)


def test_shortcut_is_refused_when_the_shape_does_not_recertify():
    q0, q1 = np.zeros(7), np.zeros(7)
    q1[0] = 0.4
    tall = np.zeros(7)
    tall[0] = 3.0
    # nothing shorter certifies: the tall via is load-bearing
    out, n = _shorten(q0, q1, [tall], lambda s: len(s) == 1)
    assert n == 0 and len(out) == 1


def test_shortcut_never_takes_a_drop_that_costs_more():
    """Dropping a via can lengthen the joint-space line; that is not a cut."""
    q0, q1 = np.zeros(7), np.zeros(7)
    q1[0] = 0.4
    on_the_way = np.zeros(7)
    on_the_way[0] = 0.2                 # collinear: dropping it saves nothing
    out, n = _shorten(q0, q1, [on_the_way], lambda s: True)
    assert n == 0 and len(out) == 1


def test_route_exposes_the_knobs_the_store_is_namespaced_on():
    assert paper.SHORTCUT is True
    assert paper.HOME_AFTER_LADDER is True
    assert paper.ROUTE_REV >= 2
    # the signature has to move when the router's behaviour does
    sig0 = paper.cache_signature()
    try:
        paper.ROUTE_REV += 1
        assert paper.cache_signature() != sig0
    finally:
        paper.ROUTE_REV -= 1
    assert paper.cache_signature() == sig0


def test_hover_near_is_a_tie_break_not_a_relaxation():
    """`HOVER_NEAR` only decides whether the fiber OPENS; the gate is the gate."""
    assert writing.HOVER_NEAR > 0
    import inspect
    src = inspect.getsource(writing.hover_solve)
    # every scan is handed the SAME `ok`; `HOVER_NEAR` only decides whether the
    # fiber opens, never what is allowed through it
    assert src.count("ok=ok") == 3  # narrow, strict-wide, settle-wide
    # ...and no scan is ever run at a margin LOOSER than the caller's ask
    assert "float(margin_min) if hold_ask is None" in src
    assert "margin_min=hold" in src


def test_the_hold_margin_is_per_call_not_a_module_default():
    """A hover the arm may FREEZE on is judged at `validate.MARGIN_GATE`.

    The lf5 stage-A re-run was refused on `frozen_failed` because arm 71's last
    hover stood 0.1111 rad from a joint limit against `validate_pose`'s 0.15,
    while every distance gate passed wide.  The widened fiber scan now asks for
    the hold margin first and settles for `HOVER_MARGIN` only where the fiber
    has nothing that keeps it.
    """
    import inspect
    from aris_sixarm import validate
    # THE HOLD MARGIN IS PER CALL AND HAS NO MODULE DEFAULT.  Asked of EVERY
    # hover it costs room -- it moved
    # test_staged.py::test_staged_end_to_end_on_a_three_stroke_picture from
    # 62.61 mm to 45.95 mm of solo clearance, under the 50 mm PAIR_MARGIN gate.
    # So `hover_solve` reads only its own argument, and `arm_program` is the
    # one caller that passes it -- for the one pose the arm stops on.
    assert writing.HOVER_HOLD_MARGIN == validate.MARGIN_GATE
    src = inspect.getsource(writing.hover_solve)
    assert "hold_ask = hold" in src
    assert "HOVER_HOLD_MARGIN" not in src, \
        "hover_solve must not read the module constant; the caller passes it"
    assert "if w is None and hold > margin_min:" in src   # the settle path
    # the memo cannot serve an answer computed under other asks
    memo = inspect.getsource(writing.lifted_or_lower)
    assert "None if hold is None else float(hold)" in memo
    assert "None if room_floor is None else float(room_floor)" in memo


def test_only_the_held_hover_pays_the_hold_margin():
    """`arm_program` asks for it once, under PARK_FREEZE, for the last exit."""
    import inspect
    src = inspect.getsource(writing.arm_program)
    assert 'str(park) == PARK_FREEZE and HOVER_HOLD_MARGIN is not None' in src
    assert "hold=HOVER_HOLD_MARGIN" in src
    assert "room_floor=HOVER_ROOM_FLOOR" in src
    # best-effort: a fiber with nothing that holds both leaves the old answer
    assert "if strict is not None and float(strict[1]) > 0.0:" in src
    # ...and the travelling hovers above it are solved with neither ask
    assert src.count("hold=HOVER_HOLD_MARGIN") == 1


def test_the_room_term_is_a_floor_and_a_capped_preference():
    """`static_gate(room_floor=...)` refuses under the floor and prefers more."""
    import inspect
    src = inspect.getsource(writing.static_gate)
    assert "partner_clearance" in src
    assert "good &= room >= rfl - paper.EPS" in src          # the floor
    assert "val = np.minimum(val, np.minimum(room, rcap))" in src  # the term
    # comfortable has to mean comfortable on BOTH or the fiber never opens
    assert "score.cap = cap if rcap is None else min(cap, rcap)" in src
    # off by default: no room_floor, no partner query, no behaviour change
    assert "room_floor=None" in inspect.signature(
        writing.static_gate).__str__().replace(" ", "").replace(
        "room_floor=None", "room_floor=None")
    assert writing.HOVER_ROOM_FLOOR > 0.050    # above PAIR_MARGIN, on purpose
