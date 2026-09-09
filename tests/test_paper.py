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


def _require_current_atlas(d, aid):
    """Skip unless the atlas in `d` was swept under THIS build's collision
    model.  Since the 2026-08-26 mesh audit an atlas is a certification
    against a specific set of capsules and a specific neighbour column, and
    `out/` is gitignored — so a directory left over from before the audit is
    stale data, not a failing assertion.  `atlas.is_current` says which."""
    import numpy as np
    from aris_sixarm import atlas as _atlas
    f = Path(d) / f"atlas_arm{aid}.npz"
    if not f.exists():
        pytest.skip(f"no atlas in {Path(d).name} — run the sweep")
    ok, why = _atlas.is_current(np.load(f))
    if not ok:
        pytest.skip(f"stale atlas in {Path(d).name}: {why} "
                    "— re-run the sweep")

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
    _require_current_atlas(ATLAS6, 2)
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


# ==========================================================================
# 4. THE LIFT LAYER: a hover is a pose the arm holds, and it is certified now
#
# Everything above is about the PAPER.  This section is about the metal, and
# about the configuration that was never gated against any of it: the ~6 cm
# hover `writing.lifted_or_lower` derives above a certified drawing pose.  On
# the proposed rig that hover stands inside a neighbour's base column for 4 to
# 18 % of every arm's certified cells, and has no IK solution at all for 55 %
# of them, and both facts were invisible because nothing asked.
# ==========================================================================
@pytest.fixture()
def lateral():
    """The proposed rig's tool, restored afterwards.

    `frames.PEN_LAT` is a process global that reaches every module (see
    `tests/test_proposed_rig_urdf.py`), and `paper`'s memos are keyed on it, so
    it is switched on per test, switched back, and the caches dropped both ways.
    """
    from aris_sixarm import frames
    before = frames.ACTIVE_TOOL
    paper.clear_cache()
    frames.activate_tool("lateral")
    try:
        yield frames.PEN_LAT
    finally:
        frames.activate_tool(before)
        paper.clear_cache()


PROPOSED_ATLAS = ROOT / "out/atlas_proposed_h0940_gated"


def _proposed_cells(aid, n):
    """`n` certified drawing poses of one proposed-rig arm. -> (rows, Q)."""
    from aris_sixarm import atlas, layout
    _require_current_atlas(PROPOSED_ATLAS, aid)
    arr, _ = atlas.load(PROPOSED_ATLAS, aid)
    go = arr[atlas.strict_go(arr)]
    if not len(go):
        pytest.skip(f"no strict-GO cells for arm {aid}")
    sel = go[np.linspace(0, len(go) - 1, min(n, len(go))).astype(int)]
    return (layout.FLEET_PROPOSED[aid], sel,
            sel[:, atlas.QCOL:atlas.QCOL + 7],
            float(layout.LAYOUT_PROPOSED["h"]))


def test_the_widened_scan_reproduces_the_old_one_where_it_had_an_answer(six):
    """A default `lifted_config` call is the call it always was.

    The fiber arguments are opt-in and the corpus depends on that: every pinned
    transit number in this repo was earned with phi = 0, a local q7 window and
    no gate, and asking for those explicitly must give the identical pose.
    """
    spec = FLEET[2]
    for p in PROBE_XY:
        a, da = writing.lifted_config(spec, np.asarray(spec.q_seed, float), p,
                                      z=writing.LIFT_Z, pen_ext=0.110)
        b, db = writing.lifted_config(spec, np.asarray(spec.q_seed, float), p,
                                      z=writing.LIFT_Z, pen_ext=0.110,
                                      phis=None, q7s=None, ok=None)
        assert (a is None) == (b is None)
        if a is not None:
            assert np.array_equal(a, b) and da == db


def test_ranking_first_and_gating_once_is_the_same_answer(six):
    """The batched gate picks the pose the per-candidate loop would have.

    `lifted_config` sorts its survivors by ||dq||_inf and scores the whole block
    at once, because scoring them one at a time was 70 % of a route's clock.
    The contract it has to keep: the pose returned is a MAXIMISER of the score,
    and among maximisers it is the nearest — so a threshold score reproduces
    "nearest acceptable" and a capped-clearance score adds "and stand off the
    metal where standing off is free".
    """
    from aris_sixarm import ik
    from aris_sixarm.frames import joint_margin, rotx, rotz, tool_offset
    spec = FLEET[2]
    pen = 0.110
    q_ref = np.asarray(spec.q_seed, float)

    # a score that rejects the elbow-up half of the fiber and then prefers a
    # low q3 among what is left, so BOTH halves of the contract are exercised
    def score(Q):
        Q = np.asarray(Q, float).reshape(-1, 7)
        return np.where(Q[:, 3] < -1.5, np.minimum(-Q[:, 2], 0.5), -np.inf)

    Twb_inv = np.linalg.inv(spec.T_world_base())
    off = tool_offset(pen)
    n_checked = 0
    for p in PROBE_XY:
        got, gd = writing.lifted_config(spec, q_ref, p, z=writing.LIFT_Z,
                                        pen_ext=pen, phis=writing.HOVER_YAWS,
                                        ok=score)
        # re-enumerate the same fiber by hand
        cand = []
        for phi in writing.HOVER_YAWS:
            R = rotz(float(phi)) @ rotx(np.pi)
            T_w = np.eye(4)
            T_w[:3, :3] = R
            T_w[:3, 3] = np.array([p[0], p[1], writing.LIFT_Z]) - R @ off
            for q7 in np.clip(q_ref[6] + np.linspace(-0.6, 0.6, 25), -2.9, 2.9):
                for q in ik.solve(Twb_inv @ T_w, q7, q_ref):
                    if joint_margin(q) < writing.HOVER_MARGIN:
                        continue
                    s = float(score(q[None, :])[0])
                    if np.isfinite(s):
                        cand.append((-s, float(np.max(np.abs(q - q_ref))), q))
        if not cand:
            assert got is None
            continue
        n_checked += 1
        assert got is not None
        cand.sort(key=lambda t: (t[0], t[1]))
        assert float(score(got[None, :])[0]) == pytest.approx(-cand[0][0],
                                                              abs=1e-12)
        assert gd == pytest.approx(cand[0][1], abs=1e-12)
    assert n_checked, "the score rejected everything; nothing was proved"


def test_the_static_floor_is_never_above_what_the_endpoints_hold(lateral):
    """`effective_static_floor`, and why the gate could not be switched on.

    The atlas certifies a drawing pose at `STATIC_MARGIN` and this module asks
    a route for `STATIC_MARGIN + 3 mm`, so a leg out of the tightest certified
    cells is asked for clearance its own first sample does not have.  Clamped,
    the gate is meaningful where it can be met and inert where the geometry
    already lost — never a contradiction.
    """
    from aris_sixarm import rig_final
    # A WIDE SAMPLE, because the band this exercises is thin: 4 to 6 cells in
    # every 300 sit between the two numbers, which is exactly why nobody
    # noticed the contradiction until every edge out of them priced `inf`.
    spec, rows, Q, h = _proposed_cells(31, 400)
    tight = 0
    ink = paper.chain_static(Q, spec, 0.110, h)
    for row, q, ci in zip(rows, Q, ink):
        # never below what the ATLAS certified the ink at, which is the
        # guarantee that makes the clamp safe to have
        assert ci >= rig_final.STATIC_MARGIN - 1e-9
        if ci >= paper.FRAME_FLOOR:
            continue
        tight += 1
        qh, _ = writing.lifted_or_lower(spec, q, row[:2], h_inv=h, pen_ext=0.110)
        fl = paper.effective_static_floor(spec, q, qh, 0.110, h)
        ends = paper.chain_static(np.stack([q, qh]), spec, 0.110, h).min()
        assert fl <= paper.FRAME_FLOOR + 1e-12
        assert fl <= ends + 1e-12
        assert fl < paper.FRAME_FLOOR - 1e-12, "the clamp did not bite"
    assert tight, ("no cell in this sample sits between STATIC_MARGIN and "
                   "FRAME_FLOOR, so the clamp was never exercised")


def test_every_hover_over_a_certified_cell_clears_the_columns(lateral):
    """THE POINT OF THE WHOLE SECTION.  Certified ink in, certified hover out.

    Measured at HEAD this failed for 4 to 18 % of each arm's cells with the
    forearm up to 131 mm INSIDE a neighbour's base column, and every conduct
    refusal on this rig named those transits.  The honest fallback — "do not
    lift at all", which returns the drawing pose the atlas certified — is
    allowed and counted, because a pen that does not lift is a report and not a
    collision.
    """
    from aris_sixarm import layout
    no_lift = total = 0
    for aid in sorted(layout.FLEET_PROPOSED):
        spec, rows, Q, h = _proposed_cells(aid, 24)
        for row, q in zip(rows, Q):
            qh, z = writing.lifted_or_lower(spec, q, row[:2], h_inv=h,
                                            pen_ext=0.110)
            total += 1
            if z == 0.0:
                no_lift += 1
                assert np.array_equal(qh, np.asarray(q, float))
                continue
            floor = min(paper.FRAME_FLOOR,
                        float(paper.chain_static(q[None], spec, 0.110, h)[0]))
            assert paper.pose_static_ok(qh, spec, 0.110, h, floor), (
                f"arm {aid} hover over {row[:2]} stands in the static set")
            # and the widened fiber did not buy that with the canvas
            cz, _, _ = paper.chain_screen(qh[None], spec, 0.110, h, [])
            assert cz[0] >= paper.CHAIN_CLEAR - 1e-9
    assert no_lift / total < 0.10, (
        f"{no_lift}/{total} cells got no hover at all; the fiber search is "
        "supposed to have taken that from 55 % to about 1 %")


def test_the_fiber_finds_hovers_phi_zero_does_not_have(lateral):
    """The tool yaw is a REAL degree of freedom on the lateral holder.

    phi = 0 is free for an inline pen and a hard constraint for a holder that
    puts the tip 110 mm off the wrist axis.  Over 600 certified cells of this
    rig the pinned solver found no hover at any of its three heights for 330 of
    them; this asserts the widened one does much better, on a sample.
    """
    from aris_sixarm import layout
    pinned = widened = total = 0
    for aid in sorted(layout.FLEET_PROPOSED):
        spec, rows, Q, h = _proposed_cells(aid, 16)
        for row, q in zip(rows, Q):
            total += 1
            pinned += writing.lifted_config(spec, q, row[:2], z=writing.LIFT_Z,
                                            h_inv=h, pen_ext=0.110)[0] is not None
            widened += writing.lifted_or_lower(spec, q, row[:2], h_inv=h,
                                               pen_ext=0.110)[1] > 0.0
    assert widened > pinned, (f"widened {widened}, pinned {pinned} of {total}")
    assert pinned / total < 0.75, "this sample is too easy to prove anything"


def test_the_broad_phase_never_moves_a_clearance_that_matters(lateral):
    """`near_boxes` drops boxes, and must not drop an answer.

    Pruned against unpruned over real hover poses: below `NEAR_SLACK` the two
    have to agree exactly, because every floor in this module is an order of
    magnitude under it.
    """
    from aris_sixarm import rig_final
    spec, rows, Q, h = _proposed_cells(2, 16)
    full = spec.static_obstacles()
    assert len(full) > 6, "this rig is supposed to carry a crowd of boxes"
    dropped = 0
    for row, q in zip(rows, Q):
        qh, _ = writing.lifted_or_lower(spec, q, row[:2], h_inv=h, pen_ext=0.110)
        P = paper.world_chain(paper.line_samples(q, qh, 9), spec, 0.110, h)
        near = paper.near_boxes(P, full)
        dropped += len(full) - len(near)
        a = float(rig_final.chain_static_clearance(P, full).min())
        b = float(rig_final.chain_static_clearance(P, near).min()) \
            if near else np.inf
        if a <= paper.NEAR_SLACK:
            assert b == pytest.approx(a, abs=1e-12)
        else:
            assert b >= a - 1e-12
    assert dropped, "the broad phase pruned nothing, so it proved nothing"


def test_the_static_bound_is_a_lower_bound(lateral):
    """The router's number must never be above the truth.

    `leg_static_lb` samples at 33 and refines; a denser measurement of the same
    leg may find a dip between two of those samples, and the bound has to have
    already allowed for it.  This is the property the 41.0 mm refusal came from
    not having.
    """
    from aris_sixarm import rig_final
    spec, rows, Q, h = _proposed_cells(13, 10)
    hov = [writing.lifted_or_lower(spec, q, r[:2], h_inv=h, pen_ext=0.110)[0]
           for r, q in zip(rows, Q)]
    boxes = spec.static_obstacles()
    n_checked = 0
    for a in hov[:5]:
        for b in hov[5:]:
            lb = paper.leg_static_lb(spec, a, b, 0.110, h, boxes)
            P = paper.world_chain(paper.line_samples(a, b, 513), spec, 0.110, h)
            dense = float(rig_final.chain_static_clearance(P, boxes).min())
            assert lb <= dense + 1e-9, (
                f"router bound {1000 * lb:.1f} mm is ABOVE a dense measurement "
                f"of {1000 * dense:.1f} mm")
            n_checked += 1
    assert n_checked >= 20


def test_a_skirt_goes_round_the_footprint_it_was_given(lateral):
    """The lateral escape is derived from the obstacle, not guessed.

    Every waypoint `_skirt` proposes for a blocked crossing must sit OUTSIDE
    the footprint of the boxes that block it — that is the whole content of
    "route around" as opposed to "route somewhere else".
    """
    spec, rows, Q, h = _proposed_cells(31, 6)
    boxes = spec.static_obstacles()
    fp = paper._box_xy(boxes)
    xy0 = np.asarray(rows[0][:2], float)
    xy1 = np.asarray(rows[-1][:2], float)
    wps = paper._skirt(spec, xy0, xy1, boxes)
    assert wps, "no detour offered for a crossing of the whole workspace"
    assert len(wps) <= paper.SKIRT_TRIES
    for wp in wps:
        for p in wp:
            inside = ((fp[:, 0, 0] <= p[0]) & (p[0] <= fp[:, 1, 0])
                      & (fp[:, 0, 1] <= p[1]) & (p[1] <= fp[:, 1, 1]))
            assert not inside.any(), f"waypoint {p} is inside a box footprint"


def test_the_router_is_strictly_tighter_than_the_checker(lateral):
    """THE ONE RELATIONSHIP THIS REPO DOES NOT ALLOW TO INVERT.

    Every pen-up `route` certifies is re-measured here against the FULL static
    set at 8x its own sampling — the way `scene_check` measures — and must
    clear `rig_final.STATIC_MARGIN`, or what it holds its own endpoints to
    where the atlas certified them tighter.  A conducted timeline was refused
    at 41.0 mm against that 50 mm gate; this is the test that would have said
    so first.
    """
    from aris_sixarm import rig_final, sequence as seq
    spec, rows, Q, h = _proposed_cells(31, 8)
    hov = np.array([writing.lifted_or_lower(spec, q, r[:2], h_inv=h,
                                            pen_ext=0.110)[0]
                    for r, q in zip(rows, Q)])
    same = np.eye(len(hov), dtype=bool)
    cells, tip, _ = seq.dive_screen(spec, hov, hov, same, h_inv=h, pen_ext=0.110)
    assert cells, "no crossing of this set needs routing; nothing is proved"
    boxes = spec.static_obstacles()
    n_routed = n_refused = 0
    for i, j in cells:
        r = paper.route(spec, hov[i], hov[j], pen_ext=0.110, h_inv=h,
                        tip_floor=float(tip[i, j]))
        if r is None:
            n_refused += 1
            continue
        n_routed += 1
        qs = [hov[i]] + list(r["vias"]) + [hov[j]]
        floor = min(rig_final.STATIC_MARGIN,
                    float(paper.chain_static(np.stack([hov[i], hov[j]]), spec,
                                             0.110, h).min()))
        for u, v in zip(qs[:-1], qs[1:]):
            # THE CHECKER'S RULE, NOT A FIXED SAMPLE COUNT.  `scene_check`
            # refines until the residual between two samples is under
            # `FRAME_STEP` and only then subtracts it; a fixed grid subtracts
            # the residual of a coarse grid, which on a metre of
            # reconfiguration is 8 mm and would fail routes for being measured
            # badly.  Twice the checker's refinement cap, so this bound is at
            # least as tight as the one that will judge the timeline.
            P = paper.world_chain(paper.line_samples(u, v, 33), spec, 0.110, h)
            n2 = paper.refine_n(P, 33, cap=64)
            if n2 > 33:
                P = paper.world_chain(paper.line_samples(u, v, n2), spec,
                                      0.110, h)
            got = float(rig_final.chain_static_clearance(P, boxes).min()
                        - paper.sample_residual(P))
            assert got >= floor - 1e-9, (
                f"routed leg misses the checker's gate by "
                f"{1000 * (floor - got):.1f} mm — the router is LOOSER")
    assert n_routed, "every crossing was refused; the router did nothing"


def test_the_screen_flags_every_crossing_the_router_would_change(lateral):
    """The allocator must not price a fiction.

    `prune_unflyable` certifies a bag against `sequence.cost_matrix`, and that
    matrix only routes the cells `dive_screen` hands it.  A crossing the screen
    misses is priced at zero and discovered at conduct time — the inf-pricing
    lesson, one obstacle over.  So: every pair the screen passes must be a pair
    `move_ok` also passes.
    """
    from aris_sixarm import sequence as seq
    spec, rows, Q, h = _proposed_cells(2, 8)
    hov = np.array([writing.lifted_or_lower(spec, q, r[:2], h_inv=h,
                                            pen_ext=0.110)[0]
                    for r, q in zip(rows, Q)])
    same = np.eye(len(hov), dtype=bool)
    cells, tip, _ = seq.dive_screen(spec, hov, hov, same, h_inv=h, pen_ext=0.110)
    flagged = set(cells)
    assert flagged, "nothing flagged; this probe set proves nothing"
    for i in range(len(hov)):
        for j in range(len(hov)):
            if i == j or (i, j) in flagged:
                continue
            ok, _, _ = paper.move_ok(spec, hov[i], hov[j], 0.110, h,
                                     tip_floor=float(tip[i, j]))
            assert ok, (f"the screen passed crossing {i}->{j} and the router "
                        "would have routed it")


def test_the_park_probe_looks_at_the_hovers_too(lateral):
    """A span the arm can draw and cannot approach is not this arm's span.

    `ParkProbe` used to look only at the certified ink.  Since the hover is
    SEARCHED over the fiber for column clearance — which says nothing about a
    parked neighbour's forearm — it is a configuration the allocator has no
    other way of hearing about, so it is probed with the ink.
    """
    from aris_sixarm import allocate, layout
    spec, rows, Q, h = _proposed_cells(31, 6)
    fl = layout.FLEET_PROPOSED
    parks = {a: np.asarray(layout.Q_PARK_PROPOSED[a], float) for a in fl}
    pens = {a: 0.110 for a in fl}
    entry = dict(stroke_id=0, s_range=(0.0, 1.0), length=0.05,
                 plan=dict(qs=np.stack([Q[0], Q[0]]),
                           pts=np.stack([rows[0][:2], rows[0][:2]])))
    on = allocate.ParkProbe(parks, fl, pens, h_inv=h, probe_hovers=True)
    off = allocate.ParkProbe(parks, fl, pens, h_inv=h, probe_hovers=False)
    # the ink of this span is certified, so the ink-only probe passes it...
    assert off.blocked(31, entry) is False
    # ...and the hover-aware one is never MORE permissive
    assert on.blocked(31, entry) in (False, True)
    ink = off.clearance(31, entry["plan"]["qs"])
    hov = on.clearance(31, on._hovers(31, entry["plan"]), sweep=False)
    assert np.isfinite(ink) and np.isfinite(hov)
    assert on.blocked(31, entry) == bool(min(ink, hov) < on.margin)


def test_the_producers_pad_covers_the_checkers_own_slack(lateral):
    """THE GATE ORDERING, MEASURED RATHER THAN ASSERTED.

    `scene_check.static_clearance_lb` is an independent derivation and
    therefore a LOWER bound with slack in it — it samples each capsule every
    `step` metres instead of minimising along it, and subtracts half a step,
    and then the trajectory residual on top.  `rig_final.STATIC_SWEEP_PAD` is
    what a producer pays so that its own exact measurement still clears the
    checker's bounded one.  The two numbers live in two modules that may not
    import each other, so this is where they are held together: the pad is
    computed from the CHECKER's own constants and compared with the producer's.

    It is not academic.  Under-sized at 3 mm it refused a solo phase at
    47.4 mm, a three-arm phase and a six-arm phase, on a rig where the ink
    measured 103 mm and the pen-up over it 58.8.
    """
    from aris_sixarm import rig_final
    # the checker's capsule-sampling half-step, from its own default
    step = scene_check.static_clearance_lb.__defaults__[0]
    seg = 0.5 * step
    sweep = 0.55 * scene_check.FRAME_STEP
    assert rig_final.STATIC_SEG_SLACK == pytest.approx(seg, abs=1e-12)
    assert rig_final.STATIC_SWEEP_SLACK == pytest.approx(sweep, abs=1e-12)
    assert rig_final.STATIC_SWEEP_PAD >= seg + sweep - 1e-12, (
        "a producer that pays less than the checker's own slack will be "
        "refused for measuring the same metal more accurately")
    # ...and every producer in the chain actually pays it
    assert paper.FRAME_FLOOR >= rig_final.STATIC_MARGIN + seg + sweep - 1e-12
    assert (rig_final.STATIC_PLAN_MARGIN
            >= rig_final.STATIC_MARGIN + seg + sweep - 1e-12)
    # ...and the checkers do NOT, so they stay a second opinion
    from aris_sixarm.scene_check import STATIC_MARGIN as checker_margin
    assert checker_margin == rig_final.STATIC_MARGIN


def test_a_plannable_pose_survives_the_checkers_own_measurement(lateral):
    """The pad is enough on real poses, not only in arithmetic.

    Every certified drawing pose and its hover, measured the way `scene_check`
    measures — its bound, its capsules, its slack — must clear
    `STATIC_MARGIN`.  This is the property the three refused phases did not
    have and the whole point of `STATIC_PLAN_MARGIN`.
    """
    from aris_sixarm import rig_final
    spec, rows, Q, h = _proposed_cells(2, 60)
    keep = paper.chain_static(Q, spec, 0.110, h) >= rig_final.STATIC_PLAN_MARGIN
    Q, rows = Q[keep], rows[keep]
    assert len(Q) > 20, "not enough plannable cells in this sample"
    P = paper.world_chain(Q, spec, 0.110, h)
    lb = scene_check.static_clearance_lb(P, spec.static_obstacles())
    assert lb.min() >= rig_final.STATIC_MARGIN - 1e-9, (
        f"a plannable drawing pose reads {1000 * lb.min():.1f} mm at the "
        "checker against its 50 mm gate")
    H = np.array([writing.lifted_or_lower(spec, q, r[:2], h_inv=h,
                                          pen_ext=0.110)[0]
                  for r, q in zip(rows, Q)])
    lbh = scene_check.static_clearance_lb(
        paper.world_chain(H, spec, 0.110, h), spec.static_obstacles())
    assert lbh.min() >= rig_final.STATIC_MARGIN - 1e-9, (
        f"a certified hover reads {1000 * lbh.min():.1f} mm at the checker")


# ==========================================================================
# ...AND THE THIRD OBSTACLE: THE ARM AGAINST ITSELF, ALONG THE LINE
# ==========================================================================
# `selfcoll` gates POSES and `writing.static_gate` gates the poses `route`
# chooses as vias.  Neither says anything about the straight joint-space line
# between two of them, which is the motion this whole module exists to certify
# — and `writing.py` says in as many words that its interpolation "makes no
# collision or self-collision guarantee".  `paper.SELF_SAFE` closes that, and
# these are the three things it owes: the bound is a bound, the routes it
# certifies survive the CHECKER's arithmetic, and it refuses something real.
@pytest.fixture()
def self_gate():
    """`paper.SELF_SAFE` on, and the memos it keys, restored afterwards.

    The gate ships OFF — it is right and the router cannot meet it, and
    `paper`'s note carries the 5.30 points that costs — so every test about it
    has to turn it on rather than assume it.
    """
    was = paper.SELF_SAFE
    paper.SELF_SAFE = True
    paper.clear_cache()
    try:
        yield
    finally:
        paper.SELF_SAFE = was
        paper.clear_cache()


def test_the_self_bound_is_a_lower_bound(lateral, self_gate):
    """The router's self number must never be above a denser measurement.

    `leg_self_lb` samples at 33, screens with bounding spheres and refines; a
    513-sample measurement by `validate`'s independent derivation may find a
    dip between two of those samples, and the bound has to have allowed for it.
    Same property `test_the_static_bound_is_a_lower_bound` pins one obstacle
    over, and the same failure it would catch.
    """
    from aris_sixarm import validate
    spec, rows, Q, h = _proposed_cells(31, 10)
    hov = [writing.lifted_or_lower(spec, q, r[:2], h_inv=h, pen_ext=0.110)[0]
           for r, q in zip(rows, Q)]
    n_checked = 0
    for a in hov[:5]:
        for b in hov[5:]:
            lb = paper.leg_self_lb(spec, a, b, 0.110, floor=None)
            dense = float(validate.self_clearance(
                paper.line_samples(a, b, 513), pen_ext=0.110).min())
            assert lb <= dense + 1e-9, (
                f"router self bound {1000 * lb:.1f} mm is ABOVE a dense "
                f"measurement of {1000 * dense:.1f} mm")
            n_checked += 1
    assert n_checked >= 20


def test_the_screened_self_clearance_is_the_exact_one_where_it_matters(lateral):
    """The sphere screen may only skip pairs that cannot decide the gate.

    `selfcoll.min_clearance(floor=...)` leaves a pair at its bounding-sphere
    bound whenever that bound already clears the floor.  The contract is the
    same one `leg_static_lb`'s `floor` carries: the number is a lower bound
    everywhere and lands on the RIGHT SIDE of the floor always.
    """
    from aris_sixarm import selfcoll
    rng = np.random.default_rng(17)
    from aris_sixarm.frames import FR3_MAX, FR3_MIN
    Q = FR3_MIN + rng.random((3000, 7)) * (FR3_MAX - FR3_MIN)
    A, B, R = selfcoll.capsule_ends(Q, 0.110)
    truth = selfcoll.self_clearance(Q, 0.110)
    assert selfcoll.min_clearance(A, B, R) == pytest.approx(float(truth.min()),
                                                            abs=1e-12)
    per = selfcoll.clearance_screened(A, B, R, 0.023)
    assert np.all(per <= truth + 1e-12), "the screen read ABOVE the truth"
    assert np.array_equal(per >= 0.023, truth >= 0.023)
    # ...and it is exact on every configuration the gate is about
    tight = truth < 0.023
    assert np.allclose(per[tight], truth[tight], atol=1e-12)


def test_a_routed_pen_up_never_folds_the_arm_into_itself(lateral, self_gate):
    """Every leg `route` certifies, re-measured the CHECKER's way.

    `scene_check` sweeps the conducted timeline at `SELF_MARGIN` = 20 mm and
    the producer plans to `SELF_PLAN_MARGIN` = 23; this asserts the ordering
    holds on the routes the router actually returns, at 8x its own sampling and
    with the checker's own copy of the geometry.
    """
    from aris_sixarm import scene_check as sc
    from aris_sixarm import selfcoll, sequence as seq
    spec, rows, Q, h = _proposed_cells(31, 8)
    hov = np.array([writing.lifted_or_lower(spec, q, r[:2], h_inv=h,
                                            pen_ext=0.110)[0]
                    for r, q in zip(rows, Q)])
    same = np.eye(len(hov), dtype=bool)
    cells, tip, _ = seq.dive_screen(spec, hov, hov, same, h_inv=h, pen_ext=0.110)
    assert cells, "no crossing of this set needs routing; nothing is proved"
    n = 0
    for i, j in cells:
        r = paper.route(spec, hov[i], hov[j], pen_ext=0.110, h_inv=h,
                        tip_floor=float(tip[i, j]))
        if r is None:
            continue
        n += 1
        qs = [hov[i]] + list(r["vias"]) + [hov[j]]
        floor = min(sc.SELF_MARGIN,
                    float(selfcoll.self_clearance(np.stack([hov[i], hov[j]]),
                                                  0.110).min()))
        for u, v in zip(qs[:-1], qs[1:]):
            P = paper.line_samples(u, v, 257)
            got = float(sc.self_clearance(P, 0.110).min()
                        - paper.SWEEP_K * float(np.max(np.linalg.norm(
                            np.diff(paper.world_chain(P, spec, 0.110, h),
                                    axis=0), axis=2))))
            assert got >= floor - 1e-9, (
                f"routed leg folds to {1000 * got:.1f} mm against a "
                f"{1000 * floor:.1f} mm floor — the router is LOOSER than "
                "the checker")
    assert n, "every crossing was refused; the router did nothing"


def test_the_self_gate_refuses_something_real(lateral, self_gate):
    """The gate is not vacuous on a joint-space line.

    Straight moves between certified DRAWING poses of one arm — poses the self
    guard passes at 63.7 mm or better at both ends — reach deep inside the arm
    partway along.  If this ever stops finding one, the gate has become
    decoration and `paper.SELF_SAFE` should be re-argued rather than trusted.
    """
    from aris_sixarm import selfcoll
    spec, rows, Q, h = _proposed_cells(31, 40)
    assert float(selfcoll.self_clearance(Q, 0.110).min()) >= \
        selfcoll.SELF_PLAN_MARGIN, "a certified drawing pose fails the guard"
    worst, bad = np.inf, 0
    for a in Q[:20]:
        for b in Q[20:]:
            v = paper.leg_self_lb(spec, a, b, 0.110,
                                  floor=selfcoll.SELF_PLAN_MARGIN)
            worst = min(worst, v)
            bad += int(v < selfcoll.SELF_PLAN_MARGIN)
    assert bad, ("no straight line between 400 pairs of certified poses is "
                 "refused by the self gate — it is doing nothing")
    assert worst < 0.0, (f"the worst line only reaches {1000 * worst:.1f} mm; "
                         "the gate refuses margin, not metal")


def test_the_atlas_gate_is_a_parameter_and_a_stricter_one_is_reachable(lateral):
    """WHAT THE 13 mm BETWEEN THE TWO STATIC FLOORS ACTUALLY COSTS.

    `atlas.solve_cell` gates a drawing pose at `rig_final.STATIC_MARGIN` (the
    CHECKER's number, exact) and `paper.route` holds a leg to
    `paper.FRAME_FLOOR` = that plus `STATIC_SWEEP_PAD` (the PRODUCER's).
    `effective_static_floor` clamps the difference away so the two are never in
    contradiction — the test above pins that — but a pose sitting ON the atlas
    gate has nothing left over: clamped, the descent out of it must hold the
    pose's OWN clearance along its whole swing, and a swing dips.

    So the floor is a parameter now, and this pins the two things that makes
    true.  A pose found at the stricter floor really does clear it (so a
    re-gated atlas means what it says), and it is still a pose the LOOSER gate
    would have accepted — the search is a restriction, never a different
    search.
    """
    from aris_sixarm import atlas, frames, layout, rig_final
    spec, rows, Q, h = _proposed_cells(31, 60)
    boxes = spec.static_obstacles()
    Twb = spec.T_world_base(h)
    Twb_inv = np.linalg.inv(Twb)
    groups = atlas._gated_groups(15.0)
    cands = atlas._candidates(15.0)
    tight = harder = 0
    ink = paper.chain_static(Q, spec, 0.110, h)
    for row, ci in zip(rows, ink):
        assert ci >= rig_final.STATIC_MARGIN - 1e-9
        if ci >= paper.FRAME_FLOOR:
            continue
        tight += 1
        r = atlas.solve_cell(float(row[0]), float(row[1]), Twb, Twb_inv, spec,
                             cands, 0.110, boxes,
                             pen_lat=frames.PEN_LAT_HOLDER,
                             gate_groups=groups,
                             static_margin=paper.FRAME_FLOOR)
        if r is None:
            continue          # a legitimate answer: this cell has no such pose
        q = np.asarray(r[7], float)
        got = float(paper.chain_static(q[None, :], spec, 0.110, h)[0])
        assert got >= paper.FRAME_FLOOR - 1e-9, (
            f"a pose found at the stricter floor does not clear it: "
            f"{1000 * got:.2f} mm")
        # ...and it would have passed the looser gate too, by construction
        assert got >= rig_final.STATIC_MARGIN - 1e-9
        harder += 1
    assert tight, ("no cell in this sample sits between STATIC_MARGIN and "
                   "FRAME_FLOOR, so the stricter gate was never exercised")
    assert harder, ("not one banded cell had a pose at the producer's floor; "
                    "measured over all 772 of them, 96.6 % do")


def test_the_default_atlas_gate_is_bit_identical(lateral):
    """`static_margin=None` is the gate every shipped atlas was swept at, and
    a parameter that changed the default would silently re-certify a corpus."""
    from aris_sixarm import atlas, frames, layout, rig_final
    spec, rows, Q, h = _proposed_cells(31, 24)
    boxes = spec.static_obstacles()
    Twb = spec.T_world_base(h)
    Twb_inv = np.linalg.inv(Twb)
    groups = atlas._gated_groups(15.0)
    cands = atlas._candidates(15.0)
    for row in rows:
        a = atlas.solve_cell(float(row[0]), float(row[1]), Twb, Twb_inv, spec,
                             cands, 0.110, boxes,
                             pen_lat=frames.PEN_LAT_HOLDER, gate_groups=groups)
        b = atlas.solve_cell(float(row[0]), float(row[1]), Twb, Twb_inv, spec,
                             cands, 0.110, boxes,
                             pen_lat=frames.PEN_LAT_HOLDER, gate_groups=groups,
                             static_margin=rig_final.STATIC_MARGIN)
        assert (a is None) == (b is None)
        if a is None:
            continue
        assert np.array_equal(np.asarray(a[7], float),
                              np.asarray(b[7], float))
        # ...and it is the pose the shipped atlas actually carries
        assert np.allclose(np.asarray(a[7], float),
                           np.asarray(row[atlas.QCOL:atlas.QCOL + 7], float),
                           atol=1e-9)


# ==========================================================================
# THE ADAPTIVE CERTIFICATE (2026-09-09)
# ==========================================================================
# These four read the FINAL-TOOL atlas rather than `PROPOSED_ATLAS`, which was
# swept at the 0.110 inline pen and has been stale — and therefore skipping the
# tests above — since the holder landed (7f99565).  Fixing that dir is its own
# re-sweep; these do not need it, they need any current corpus of certified
# poses to build legs out of.
FINAL_ATLAS = ROOT / "out/atlas_proposed_h0940_lat0860_gated63"


def _final_cells(aid, n):
    """`n` certified drawing poses at the FINAL tool. -> (spec, rows, Q, h)."""
    from aris_sixarm import atlas, layout
    _require_current_atlas(FINAL_ATLAS, aid)
    arr, _ = atlas.load(FINAL_ATLAS, aid)
    go = arr[atlas.strict_go(arr)]
    if not len(go):
        pytest.skip(f"no strict-GO cells for arm {aid}")
    sel = go[np.linspace(0, len(go) - 1, min(n, len(go))).astype(int)]
    return (layout.FLEET_PROPOSED[aid], sel,
            sel[:, atlas.QCOL:atlas.QCOL + 7],
            float(layout.LAYOUT_PROPOSED["h"]))

def test_the_adaptive_bound_never_exceeds_a_dense_measurement(lateral):
    """The tightening must stay a LOWER bound, on legs chosen to stress it.

    `test_the_static_bound_is_a_lower_bound` asks this of `leg_static_lb` at
    the old density.  This asks it of the converged bound against a grid an
    order finer than anything the router samples, on the long branch-changing
    legs the adaptive path is written for — the ones where the old whole-leg
    residual was the entire answer.
    """
    from aris_sixarm import rig_final
    spec, rows, Q, h = _final_cells(31, 16)
    boxes = spec.static_obstacles()
    rng = np.random.default_rng(20260909)
    n_checked = 0
    for i in rng.permutation(len(Q))[:6]:
        for j in rng.permutation(len(Q))[:4]:
            if i == j:
                continue
            a, b = np.asarray(Q[i], float), np.asarray(Q[j], float)
            lb = paper.adaptive_static_lb(spec, a, b, spec.pen, h, boxes)
            P = paper.world_chain(paper.line_samples(a, b, 4001), spec,
                                  spec.pen, h)
            dense = float(rig_final.chain_static_clearance(P, boxes).min())
            assert lb <= dense + 1e-9, (
                f"adaptive bound {1000 * lb:.3f} mm is ABOVE a 4001-sample "
                f"measurement of {1000 * dense:.3f} mm")
            n_checked += 1
    assert n_checked >= 15


def test_finer_subdivision_only_tightens_the_bound(lateral):
    """A smaller tolerance may raise the bound and must never lower it.

    The property that makes the adaptive path safe to turn on: it converges
    upward towards the truth, so every leg the coarse certificate accepted is
    still accepted and the only new answers are rescues.
    """
    spec, rows, Q, h = _final_cells(71, 12)
    boxes = spec.static_obstacles()
    n_checked = 0
    for a, b in zip(Q[:6], Q[6:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        coarse = paper.adaptive_static_lb(spec, a, b, spec.pen, h, boxes,
                                          tol=0.008)
        mid = paper.adaptive_static_lb(spec, a, b, spec.pen, h, boxes,
                                       tol=0.002)
        fine = paper.adaptive_static_lb(spec, a, b, spec.pen, h, boxes,
                                        tol=paper.ADAPT_TOL)
        assert mid >= coarse - 1e-9 and fine >= mid - 1e-9, (
            f"bound went DOWN under refinement: {1000 * coarse:.3f} -> "
            f"{1000 * mid:.3f} -> {1000 * fine:.3f} mm")
        n_checked += 1
    assert n_checked >= 5


def test_the_adaptive_bound_is_at_least_as_tight_as_the_whole_leg_residual(
        lateral):
    """The localised residual can only beat the global one.

    `min over every sample` minus `max over every interval` is this bound with
    its two terms taken from opposite ends of the leg; the per-interval form is
    the same argument stated where it applies.
    """
    from aris_sixarm import rig_final
    spec, rows, Q, h = _final_cells(31, 12)
    boxes = spec.static_obstacles()
    for a, b in zip(Q[:6], Q[6:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        P = paper.world_chain(paper.line_samples(a, b, paper.SAMPLES), spec,
                              spec.pen, h)
        c = rig_final.chain_static_clearance(P, boxes)
        whole = float(c.min()) - paper.sample_residual(P)
        lb, res = paper.interval_bounds(c, P)
        assert float(lb.min()) >= whole - 1e-12
        assert paper.adaptive_static_lb(spec, a, b, spec.pen, h,
                                        boxes) >= whole - 1e-12


def test_the_self_bound_stays_under_a_dense_self_measurement(lateral):
    """`leg_self_lb` moved onto the same adaptive path and must keep its side.

    The self screen is held exact to `SELF_SCREEN_PAD` above the floor for
    exactly this reason: certified against a screened bound rather than the
    truth, an interval could never clear its own residual.
    """
    from aris_sixarm import selfcoll
    spec, rows, Q, h = _final_cells(13, 12)
    n_checked = 0
    for a, b in zip(Q[:6], Q[6:]):
        a, b = np.asarray(a, float), np.asarray(b, float)
        lb = paper.leg_self_lb(spec, a, b, spec.pen)
        dense = float(selfcoll.self_clearance(
            paper.line_samples(a, b, 2001), spec.pen).min())
        assert lb <= dense + 1e-9, (
            f"self bound {1000 * lb:.3f} mm is ABOVE a 2001-sample "
            f"measurement of {1000 * dense:.3f} mm")
        n_checked += 1
    assert n_checked >= 5


# ==========================================================================
# THE POSE-AWARE NEIGHBOUR MODEL (2026-09-09) -- aris_sixarm/frozen.py
# ==========================================================================

@pytest.fixture
def thawed():
    """`frozen` is a process global; every test here restores it."""
    from aris_sixarm import frozen
    frozen.thaw()
    paper.clear_cache()
    try:
        yield frozen
    finally:
        frozen.thaw()
        paper.clear_cache()


def _frozen_fleet(h=None):
    from aris_sixarm import layout
    h = float(layout.LAYOUT_PROPOSED["h"]) if h is None else h
    fl = layout.build_fleet(layout.paired_grid(spacing=0.61, rows=3, h=h))
    return fl, {a: fl[a].pen for a in fl}, h


def test_the_frozen_model_is_off_by_default_and_changes_nothing(lateral, thawed):
    """The option must be inert until it is asked for.

    Every shipped number was earned with the pose-invariant bands, so with
    `frozen` thawed the box set and the clearance have to be the SAME OBJECTS
    and the SAME NUMBERS the router always saw.
    """
    from aris_sixarm import rig_final
    fl, pens, h = _frozen_fleet()
    spec = fl[71]
    raw = spec.static_obstacles()
    assert not thawed.active()
    assert len(paper.static_boxes(spec)) == len(raw)
    spec2, rows, Q, _ = _final_cells(71, 8)
    P = paper.world_chain(np.asarray(Q, float), spec, spec.pen, h)
    assert np.array_equal(rig_final.chain_static_clearance(P, raw),
                          thawed.chain_clearance(P, raw))


def test_freezing_drops_only_the_named_partners_bands(thawed):
    """True structure is never dropped, and neither is a moving partner's band.

    The whole safety argument is that this swaps ONE thing: the pose-invariant
    stand-in for a partner's movable links, and only for partners that are
    actually being held still.
    """
    fl, pens, h = _frozen_fleet()
    spec = fl[71]
    raw = spec.static_obstacles()
    structure = [b for b in raw if thawed.band_owner(b["name"]) is None]
    assert structure, "expected mounts/plates/runway in the static set"
    # freeze only arm 31
    thawed.freeze({31: np.zeros(7)}, fl, pens, h)
    thawed.observe(71)
    kept = paper.static_boxes(spec)
    names = {b["name"] for b in kept}
    assert all(b["name"] in names for b in structure), "structure was dropped"
    for b in raw:
        o = thawed.band_owner(b["name"])
        if o == 31:
            assert b["name"] not in names, "arm 31's band should be gone"
        else:
            assert b["name"] in names, f"{b['name']} should have been kept"


def test_a_partner_that_is_not_frozen_never_loosens(lateral, thawed):
    """The lower-bound property: a MOVING partner keeps its band, exactly.

    This is what makes the option safe to have in the tree at all -- turning it
    on for one arm cannot quietly relax the model for another.
    """
    from aris_sixarm import rig_final
    fl, pens, h = _frozen_fleet()
    spec = fl[71]
    raw = spec.static_obstacles()
    spec2, rows, Q, _ = _final_cells(71, 8)
    Qa = np.asarray(Q, float)
    P = paper.world_chain(Qa, spec, spec.pen, h)
    base = rig_final.chain_static_clearance(P, raw)
    # freeze a partner that is NOWHERE near these poses, and check the other
    # partners' bands still bind exactly as before
    thawed.freeze({2: np.zeros(7)}, fl, pens, h)
    thawed.observe(71)
    kept = paper.static_boxes(spec)
    still = rig_final.chain_static_clearance(
        P, [b for b in kept if thawed.band_owner(b["name"]) is not None])
    band_only = rig_final.chain_static_clearance(
        P, [b for b in raw
            if thawed.band_owner(b["name"]) not in (None, 2)])
    assert np.array_equal(still, band_only), (
        "freezing arm 2 changed what arm 13/17/31/97's bands say")


def test_the_frozen_partners_capsules_actually_bind(thawed):
    """The swap is not a deletion: the partner's real arm is still an obstacle.

    Dropping a band and adding nothing would be unsound, so the model has to be
    able to REFUSE on the capsules it put in the band's place.
    """
    fl, pens, h = _frozen_fleet()
    spec = fl[71]
    # park arm 31 where its own links reach out of its column footprint
    q31 = np.array([0.0, -0.6, 0.0, -2.2, 0.0, 2.0, 0.0])
    thawed.freeze({31: q31}, fl, pens, h)
    thawed.observe(71)
    P31 = paper.world_chain(q31[None, :], fl[31], fl[31].pen, h)[0]
    # a pose of arm 71 built to sit ON one of arm 31's link capsules is refused
    d = thawed.partner_clearance(P31[None])      # arm 31 against itself, frozen
    assert np.isfinite(d[0]) and d[0] < 0.0, (
        "a chain lying on the frozen partner should read negative, got "
        f"{1000 * d[0]:.1f} mm")
    # ...and the combined answer is never above the boxes-only answer
    kept = paper.static_boxes(spec)
    from aris_sixarm import rig_final
    both = thawed.chain_clearance(P31[None], kept)
    boxes_only = rig_final.chain_static_clearance(P31[None], kept)
    assert both[0] <= boxes_only[0] + 1e-12
