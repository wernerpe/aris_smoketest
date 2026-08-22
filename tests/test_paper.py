"""The paper plane as an obstacle: the gate, the routing, and the dead tiers.

THE REGRESSION THIS FILE EXISTS FOR.  `out/csail_schedule_final6.npz` shipped
on 2026-08-21 with three pen-up blocks that go through the table — pen tip
253.6 mm under the canvas, a chain point 156.1 mm under it — and every check in
the repo passed it, because the paper was the one piece of the scene nothing
modelled between poses.  A trimmed copy of that exact timeline is checked in as
`tests/data/csail_final6_prefix_timeline.npz` and the first test here asserts
that `scene_check` now REFUSES it.  If that test ever goes green-by-passing,
the gate has been weakened back to where it was.
"""
from pathlib import Path

import numpy as np
import pytest

from aris_sixarm import dead as dead_mod
from aris_sixarm import fleet as fleet_mod
from aris_sixarm import paper, scene_check, sequence, writing
from aris_sixarm.fleet import FLEET

ROOT = Path(__file__).parents[1]
PREFIX = ROOT / "tests/data/csail_final6_prefix_timeline.npz"
ATLAS6 = ROOT / "out/atlas_final6_opt"

# the two arms whose transits dive, and how far down they go (metres)
OFFENDERS = [2, 97]

# Hover-reachable canvas points for arm 2, spanning the mirror plane at
# y = 1.81532 — which is where its crossings get into trouble.  20 of the 42
# ordered pairs between them violate the paper gate on a straight joint-space
# line, so a sweep over this set exercises the router and not the happy path.
PROBE_XY = [(0.80, 1.60), (0.90, 1.70), (0.95, 1.55), (0.95, 1.93),
            (1.00, 1.80), (1.10, 1.65), (0.85, 1.90)]


@pytest.fixture()
def six():
    """The six-arm extended-pole rig, restored afterwards.

    The corpus is pinned to the 3-arm default (`docs/MERGED_CANVAS.md` §1), so
    this is switched on per test and switched back — `paper`'s memo is keyed on
    `id(spec)` and the fleet dict is mutated in place, so the cache is dropped
    on the way in and on the way out.
    """
    before = fleet_mod.ACTIVE_RIG
    paper.clear_cache()
    fleet_mod.activate("final6_opt")
    try:
        yield fleet_mod.FLEET
    finally:
        fleet_mod.activate(before)
        paper.clear_cache()


@pytest.fixture()
def shipped():
    if not PREFIX.exists():
        pytest.skip("pre-fix timeline fixture not checked in")
    d = np.load(PREFIX, allow_pickle=True)
    arms = [int(a) for a in d["arms"]]
    return dict(q={a: np.asarray(d[f"q_{a}"], float) for a in arms},
                seg={a: np.asarray(d[f"seg_{a}"], int) for a in arms},
                pens={a: float(p) for a, p in zip(arms, d["pen_ext"])},
                fps=float(d["fps"]), margin=float(d["margin"]), arms=arms)


# ==========================================================================
# 1. the gate refuses what shipped
# ==========================================================================
def test_shipped_timeline_fails_the_new_paper_gate(six, shipped):
    """THE POINT OF THE WHOLE CHANGE.  Pre-fix timeline in, FAIL out."""
    rep = scene_check.check_timeline(
        shipped["q"], 1.0 / shipped["fps"], shipped["margin"],
        pen_ext=shipped["pens"], sub=4, verbose=False)
    assert rep["paper_failed"] == OFFENDERS
    assert rep["ok"] is False
    for a in OFFENDERS:                       # and it fails by a country mile
        cl = rep["paper_clearance"][a]
        assert cl["chain"] < -0.05, f"arm {a} chain {cl['chain']}"
        assert cl["tip"] < -0.15, f"arm {a} tip {cl['tip']}"


def test_the_gate_is_not_tripped_by_the_arms_that_are_fine(six, shipped):
    """Arms 31 and 71 fly the same kind of transit and never break the plane.

    They sit inside the contact band by a few mm because their pen is ON the
    paper while drawing; a gate that called that a violation would be useless.
    """
    rep = scene_check.check_timeline(
        shipped["q"], 1.0 / shipped["fps"], shipped["margin"],
        pen_ext=shipped["pens"], sub=4, verbose=False)
    for a in (13, 17, 31, 71):
        assert a not in rep["paper_failed"]
        cl = rep["paper_clearance"][a]
        assert cl["tip"] >= -scene_check.PAPER_TIP
        assert cl["chain"] >= scene_check.PAPER_CHAIN


@pytest.mark.parametrize("sub", [1, 2, 4, 8])
def test_the_verdict_does_not_depend_on_the_sampling_rate(six, shipped, sub):
    """A gate whose answer moved with `sub` would be a coin toss.

    The residual is a 1-Lipschitz bound on 3D displacement, which over-charges
    a tip SLIDING along the paper; `check_timeline` auto-refines until the
    per-point motion is under `PAPER_STEP`, so the verdict is a property of the
    trajectory.  Without that refinement this test fails at sub=1 and 2.
    """
    rep = scene_check.check_timeline(
        shipped["q"], 1.0 / shipped["fps"], shipped["margin"],
        pen_ext=shipped["pens"], sub=sub, verbose=False)
    assert rep["paper_failed"] == OFFENDERS


def test_the_chain_gate_alone_catches_all_of_it(six, shipped):
    """The tip gate is a second net; the chain gate does the precise work."""
    rep = scene_check.check_timeline(
        shipped["q"], 1.0 / shipped["fps"], shipped["margin"],
        pen_ext=shipped["pens"], sub=4, verbose=False)
    by_chain = [a for a, v in rep["paper_clearance"].items()
                if v["chain"] < scene_check.PAPER_CHAIN]
    assert sorted(by_chain) == OFFENDERS


# ==========================================================================
# 2. routing: a route is certified, or there is no route
# ==========================================================================
def _hover(spec, xy, z, pen=0.110):
    q, _ = writing.lifted_config(spec, np.asarray(spec.q_seed, float), xy, z=z,
                                 pen_ext=pen)
    return q


def test_route_never_returns_a_path_that_touches_the_paper(six):
    """Every leg of whatever `route` hands back clears both floors.

    Swept over a grid of real hover pairs on the merged canvas rather than one
    hand-picked case, so this is a property of the router and not of an example.
    """
    spec = FLEET[2]
    pen = 0.110
    hs = [(p, _hover(spec, p, writing.LIFT_Z, pen)) for p in PROBE_XY]
    hs = [(p, q) for p, q in hs if q is not None]
    assert len(hs) == len(PROBE_XY), "every probe point should hover on this rig"
    n_direct = n_routed = n_refused = 0
    for _, q0 in hs:
        for _, q1 in hs:
            if q0 is q1:
                continue
            r = paper.route(spec, q0, q1, pen_ext=pen, q_home=spec.q_seed)
            if r is None:
                n_refused += 1
                ok, _, _ = paper.move_ok(spec, q0, q1, pen)
                assert not ok, "refused a move that was fine all along"
                continue
            n_direct += r["mode"] == "direct"
            n_routed += r["mode"] != "direct"
            cz, tz = paper.path_clearance(
                spec, [q0] + list(r["vias"]) + [q1], pen)
            assert cz >= paper.CHAIN_CLEAR - 1e-9
            assert tz >= paper.TIP_CLEAR - 1e-9
            # and the report agrees with an independent re-measurement
            assert cz == pytest.approx(r["chain_z"], abs=1e-9)
            assert tz == pytest.approx(r["tip_z"], abs=1e-9)
    assert n_direct + n_routed + n_refused == len(hs) * (len(hs) - 1)
    assert n_routed + n_refused >= 10, ("this probe set is supposed to be hard; "
                                        "a sweep in which nothing violates "
                                        "proves nothing about the router")


def test_a_route_never_trades_the_paper_for_the_frame(six):
    """The way out of the paper is UP, and up is where the steel is.

    An inserted route is certified against `rig_final.chain_static_clearance`
    as well as the paper, because the first routed conduct of this logo was
    refused at 49.9 mm of frame clearance against a 50 mm margin while the
    paper gate passed comfortably — the exact trade this forbids.
    """
    from aris_sixarm import rig_final
    spec = FLEET[2]
    pen = 0.110
    hs = [_hover(spec, p, writing.LIFT_Z, pen) for p in PROBE_XY]
    hs = [q for q in hs if q is not None]
    n_checked = 0
    for q0 in hs:
        for q1 in hs:
            if q0 is q1:
                continue
            r = paper.route(spec, q0, q1, pen_ext=pen, q_home=spec.q_seed)
            if r is None or not r["vias"]:
                continue
            n_checked += 1
            fc = paper.path_frame_clearance(
                spec, [q0] + list(r["vias"]) + [q1], pen)
            assert fc >= rig_final.STATIC_MARGIN - 1e-9, (
                f"routed path clears the paper but comes {1000 * fc:.1f} mm "
                f"from the frame")
    assert n_checked > 0, "no route was inserted, so nothing was checked"


def test_every_via_is_a_pose_the_arm_may_stand_in(six):
    """Vias are gated IK solutions, not interpolation artefacts."""
    from aris_sixarm.frames import joint_margin
    spec = FLEET[2]
    pen = 0.110
    a = _hover(spec, (0.95, 1.55), writing.LIFT_Z, pen)
    b = _hover(spec, (0.95, 1.93), writing.LIFT_Z, pen)
    if a is None or b is None:
        pytest.skip("no hover IK for the probe pair on this rig")
    r = paper.route(spec, a, b, pen_ext=pen, q_home=spec.q_seed)
    if r is None:
        pytest.skip("this pair is unroutable — covered by the inf-edge test")
    for v in r["vias"]:
        assert joint_margin(np.asarray(v, float)) >= writing.HOVER_MARGIN - 1e-9


def test_a_beat_with_no_vias_is_the_old_arithmetic_exactly(six):
    """Backward compatibility, pinned.

    39 of the shipped run's 42 pen-up blocks never needed a via, and every
    pinned number in the corpus was earned on that arithmetic.  `_beat` must
    reduce to `_dq_time` when it inserts nothing.
    """
    rng = np.random.default_rng(0)
    for _ in range(50):
        q0 = rng.uniform(-1.0, 1.0, 7)
        q1 = rng.uniform(-1.0, 1.0, 7)
        for floor in (0.0, 0.2, 1.2):
            got = writing._beat([q0, q1], writing.QD_FRAC, floor)
            assert len(got) == 1
            assert got[0][0] == pytest.approx(
                writing._dq_time(q0, q1, writing.QD_FRAC, floor), rel=1e-12)


def test_a_routed_beat_is_never_faster_than_the_move_it_replaced(six):
    """A via buys clearance with seconds; it must not buy it with speed."""
    rng = np.random.default_rng(1)
    q = [rng.uniform(-0.6, 0.6, 7) for _ in range(4)]
    steps = writing._beat(q, writing.QD_FRAC, floor=0.0)
    for (dt, b), a in zip(steps, q[:-1]):
        assert dt >= writing._dq_time(a, b, writing.QD_FRAC) - 1e-12


def test_cost_matrix_prices_an_unroutable_crossing_as_infinite(six,
                                                              monkeypatch):
    """An unflyable transit must leave the search space, not raise later.

    `paper.route` is forced to refuse everything, so every off-diagonal cell
    that violates becomes `inf` and the sequencer simply cannot propose it.
    """
    spec = FLEET[2]
    pen = 0.110
    segs = []
    for p in PROBE_XY[:5]:
        q = _hover(spec, p, writing.LIFT_Z, pen)
        assert q is not None, f"no hover IK at {p}"
        segs.append(dict(length=0.05,
                         plan=dict(qs=np.array([q, q]),
                                   pts=np.array([p, (p[0] + 0.02, p[1])]))))
    N = 2 * len(segs)
    base = sequence.cost_matrix(spec, segs, pen_ext=pen, paper_safe=False)
    monkeypatch.setattr(paper, "route", lambda *a, **k: None)
    paper.clear_cache()
    hard = sequence.cost_matrix(spec, segs, pen_ext=pen, paper_safe=True)
    assert hard.shape == base.shape
    # with routing forced to fail, every crossing that violates leaves the
    # search space entirely rather than surviving as a flyable-looking cost
    lost = np.isfinite(base[:N, :N]) & ~np.isfinite(hard[:N, :N])
    assert lost.any(), "the probe set must contain a violating crossing"
    assert np.isfinite(hard[:N, :N]).sum() < np.isfinite(base[:N, :N]).sum()
    # and nothing that was already impossible became possible
    assert not (np.isfinite(hard[:N, :N]) & ~np.isfinite(base[:N, :N])).any()


# ==========================================================================
# 3. the dead tiers
# ==========================================================================
@pytest.fixture()
def tiers(six):
    if not (ATLAS6 / "atlas_arm2.npz").exists():
        pytest.skip("out/atlas_final6_opt not swept in this checkout")
    return dead_mod.tiers(ATLAS6, sorted(FLEET), fleet_mod.SHEET, 0.02)


def test_dead_layer_generation_is_deterministic(tiers):
    """Two runs over one atlas produce bit-identical masks."""
    again = dead_mod.tiers(ATLAS6, sorted(FLEET), fleet_mod.SHEET, 0.02)
    for k in ("dead_permissive", "strict_only_dead", "dead_strict", "strict",
              "permissive", "per_arm_strict", "per_arm_permissive"):
        assert np.array_equal(tiers[k], again[k]), k
    assert tiers["counts"] == again["counts"]


def test_the_two_dead_tiers_nest(tiers):
    """Permissive-dead is a strict subset of atlas-strict-dead, by definition.

    The gates are nested (0.15 <= 0.30, 0.10 <= 0.14) and both are read off the
    same atlas columns, so a cell no arm reaches at the loose gate cannot be
    reached at the tight one.  If this ever fails the two masks were built from
    different data.
    """
    assert not (tiers["dead_permissive"] & ~tiers["dead_strict"]).any()
    assert np.array_equal(tiers["strict_only_dead"],
                          tiers["dead_strict"] & ~tiers["dead_permissive"])
    c = tiers["counts"]
    assert c["dead_strict"] == c["dead_permissive"] + c["strict_only_dead"]


def test_the_strict_tier_reproduces_the_published_atlas_numbers(tiers):
    """75.93 % strict-GO / 24.07 % dead, from docs/MERGED_CANVAS.md §2.

    This module recomputes strict-GO from the raw atlas columns rather than
    calling `atlas.strict_go`, so agreeing with the published sweep is a real
    cross-check of the recomputation.
    """
    c = tiers["counts"]
    assert c["cells"] == 16562
    assert c["strict_go_pct"] == pytest.approx(75.93, abs=0.01)
    assert c["dead_strict_pct"] == pytest.approx(24.07, abs=0.01)
    # ...and the whole reason for the second tier
    assert c["dead_permissive_pct"] < c["dead_strict_pct"] - 3.0


def test_attribution_conserves_metres(tiers):
    """Every undrawn metre lands in exactly one cause."""
    prog = ROOT / "out/csail_program_final6.json"
    if not prog.exists():
        pytest.skip("no shipped programme in this checkout")
    spans, p = dead_mod.load_dropped(prog)
    rows, att = dead_mod.attribute(spans, tiers, dead_mod.arms_with_ink(p))
    assert att["n_spans"] == len(spans)
    assert sum(att["by_cause_m"].values()) == pytest.approx(
        sum(s["length_m"] for s in spans), rel=1e-9)
    # the per-sample mixture is a partition too
    for r in rows:
        if r["n"]:
            assert sum(r["frac"].values()) == pytest.approx(1.0, abs=1e-9)
    assert sum(att["by_cause_weighted_m"].values()) == pytest.approx(
        att["total_m"], rel=1e-6)
