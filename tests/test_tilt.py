"""Pen-tilt planning: the equivalences that make it safe to ship flag-gated.

The point of nearly every test here is that tilt is INERT until asked for.
`aris_sixarm/tilt.py` re-implements the lattice build and the DP so it can
carry a tilt axis, and the only way that is allowed to reach the shipping
pipeline is if the tilt-0 case is the flat case node for node — otherwise
"default 0 = today's planner" is a hope rather than a property.
"""
import numpy as np
import pytest

from aris_sixarm import fleet as fleet_mod
from aris_sixarm import ik, planner, pwl, rig_final, stroke_api, tilt
from aris_sixarm.frames import FR3_MAX, FR3_MIN, rotx


# THE FIXTURES ARE FUNCTION-SCOPED ON PURPOSE.  `fleet.activate` mutates
# module-level global state, and a module-scoped fixture only runs its setup
# ONCE — so interleaving a `final6` test between two `sixarm` tests left
# final6_opt active for the second one, which then planned the legacy rim arc
# on a rig that cannot reach it and failed for a reason that had nothing to do
# with tilt.  Re-activating per test is microseconds and removes the ordering
# dependency entirely.
@pytest.fixture
def sixarm():
    prev = fleet_mod.ACTIVE_RIG
    fleet_mod.activate("sixarm")
    yield fleet_mod.FLEET
    fleet_mod.activate(prev)


@pytest.fixture
def final6():
    prev = fleet_mod.ACTIVE_RIG
    fleet_mod.activate("final6_opt")
    yield fleet_mod.FLEET
    fleet_mod.activate(prev)


def rim_arc(spec, n=400):
    bx, by = spec.xy
    th = np.linspace(-0.6 * np.pi, 1.05 * np.pi, n)
    return planner.clip_to_sheet(
        np.column_stack([bx + 0.66 * np.cos(th), by + 0.66 * np.sin(th)]),
        verbose=False)


# --------------------------------------------------------------------------
# 1. the chart
# --------------------------------------------------------------------------
def test_disc_is_a_hex_lattice_inside_the_cone():
    d = tilt.hex_disc(15.0, n_ring=2)
    assert len(d["tilt"]) == 19                      # 1 + 6 + 6 + 6
    r = np.rad2deg(np.linalg.norm(d["tilt"], axis=1))
    assert r.max() <= 15.0 + 1e-9                    # nothing outside the cone
    assert r[0] == pytest.approx(0.0)                # index 0 is perpendicular
    assert np.all(np.diff(r) >= -1e-12)              # radius-ordered
    # six neighbours everywhere they exist, and adjacency is symmetric
    for k, row in enumerate(d["nbr"]):
        assert row[0] == k
        for j in row[1:]:
            if j >= 0:
                assert k in d["nbr"][j][1:]


def test_degenerate_disc_keeps_its_self_edge():
    """tilt_max = 0 is one point that must still be adjacent to ITSELF.

    Without it the DP has no edges at all and every stroke cuts at step 1 —
    which is exactly the bug this test was written after.
    """
    d = tilt.hex_disc(0.0)
    assert len(d["tilt"]) == 1
    assert d["nbr"][0, 0] == 0
    assert np.all(d["nbr"][0, 1:] == -1)


def test_fixed_pitch_discs_are_nested():
    """tilt <= 30 must OFFER every pose tilt <= 15 offers, or a comparison
    between them measures the sampling and not the freedom."""
    a = tilt.hex_disc(15.0, pitch_deg=7.5)["tilt"]
    b = tilt.hex_disc(30.0, pitch_deg=7.5)["tilt"]
    for t in a:
        assert np.min(np.linalg.norm(b - t, axis=1)) < 1e-12


def test_tilt_rotation_realises_the_commanded_lean():
    d = tilt.hex_disc(30.0, n_ring=2)
    R = tilt.pen_rot(d["tilt"])
    axis = R[:, :, 2]                                # tool z = the pen axis
    lean = np.rad2deg(np.arctan2(np.linalg.norm(axis[:, :2], axis=1), -axis[:, 2]))
    assert np.allclose(lean, np.rad2deg(np.linalg.norm(d["tilt"], axis=1)),
                       atol=1e-12)
    az = np.arctan2(axis[:, 1], axis[:, 0])
    want = np.arctan2(d["tilt"][:, 1], d["tilt"][:, 0])
    m = np.linalg.norm(d["tilt"], axis=1) > 1e-9
    assert np.abs(np.angle(np.exp(1j * (az[m] - want[m])))).max() < 1e-9


def test_zero_tilt_pose_is_the_shipping_pose(sixarm):
    """`tilt.pen_poses(..., None)` IS `pwl.pen_down_poses`, bit for bit."""
    spec = sixarm[31]
    Twb_inv = np.linalg.inv(spec.T_world_base())
    pts = np.column_stack([np.linspace(1.0, 1.4, 25), np.full(25, 1.5)])
    a = tilt.pen_poses(pts, None, Twb_inv, 0.110)
    b = pwl.pen_down_poses(pts, Twb_inv, 0.110)
    assert np.array_equal(a, b)
    assert np.array_equal(tilt.pen_poses(pts, np.zeros((len(pts), 2)),
                                         Twb_inv, 0.110), b)


def test_tip_lands_on_the_curve_at_every_tilt(sixarm):
    """The whole premise: the pen leans, the TIP does not move."""
    spec = sixarm[31]
    Twb = spec.T_world_base()
    d = tilt.hex_disc(30.0, n_ring=2)
    pts = np.tile(np.array([[1.30, 1.55]]), (len(d["tilt"]), 1))
    Tw = Twb @ tilt.pen_poses(pts, d["tilt"], np.linalg.inv(Twb), 0.110)
    tip = Tw[:, :3, 3] + Tw[:, :3, :3] @ np.array([0.0, 0.0, 0.110])
    assert np.abs(tip[:, :2] - pts).max() < 1e-12
    assert np.abs(tip[:, 2]).max() < 1e-12


# --------------------------------------------------------------------------
# 2. tilt 0 is the flat planner
# --------------------------------------------------------------------------
@pytest.mark.parametrize("rig", ["sixarm", "final6_opt"])
def test_tilt0_lattice_matches_planner(rig):
    prev = fleet_mod.ACTIVE_RIG
    fleet_mod.activate(rig)
    try:
        spec = fleet_mod.FLEET[31 if rig == "sixarm" else 2]
        if rig == "sixarm":
            pts, _ = planner.resample(rim_arc(spec), 0.02)
            pts = pts[:40]
        else:
            pts = np.column_stack([np.linspace(1.0, 1.3, 30),
                                   np.full(30, 1.25)])
        flat = planner.build_lattice(pts, spec)
        lat = tilt.build_lattice(pts, spec, tilt_max_deg=0.0)
        assert np.array_equal(flat["valid"], lat["valid"][:, :, 0, :])
        assert np.array_equal(np.nan_to_num(flat["Q"], nan=0.0),
                              np.nan_to_num(lat["Q"][:, :, 0, :, :], nan=0.0))
        assert np.array_equal(flat["margin"], lat["margin"][:, :, 0, :])
        assert np.array_equal(flat["sigma"], lat["sigma"][:, :, 0, :])
    finally:
        fleet_mod.activate(prev)


def test_tilt0_dp_matches_planner_plan(sixarm):
    """One-point disc, full continuity budget -> `planner.plan`'s answer."""
    spec = sixarm[31]
    pts, _ = planner.resample(rim_arc(spec), 0.012)
    lat = tilt.build_lattice(pts, spec, tilt_max_deg=0.0)
    mine = tilt.plan_lattice(lat, free=lat["valid"], objective="maximin_sigma",
                             jump=planner.JUMP_THRESH)
    ref = planner.plan(planner.build_lattice(pts, spec))
    assert mine["ok"] is ref["ok"] is True
    # `planner.plan` reports the DP's QUANTISED bottleneck (1e-4 buckets, the
    # unit it compares in); `tilt.plan_lattice` reports the path's true min
    # sigma, so that the number means the same thing under both objectives.
    # Same path, two readouts — compare them in the coarser one.
    assert (np.round(mine["bottleneck"] * planner._SIGMA_Q)
            / planner._SIGMA_Q) == pytest.approx(ref["bottleneck"], abs=1e-12)
    assert np.allclose(mine["q7"], ref["q7s"], atol=0, rtol=0)
    assert np.allclose(mine["qs"], ref["qs"], atol=0, rtol=0)


def test_from_flat_reuses_the_flat_lattice(sixarm):
    spec = sixarm[31]
    pts, _ = planner.resample(rim_arc(spec), 0.02)
    flat = planner.build_lattice(pts, spec)
    lat = tilt.from_flat(flat, tilt.hex_disc(15.0))
    assert np.array_equal(lat["valid"][:, :, 0, :], flat["valid"])
    assert lat["n_ik"] == 0                       # nothing was re-solved
    assert lat["cells"][:, 0].all()
    assert not lat["cells"][:, 1:].any()


# --------------------------------------------------------------------------
# 3. the clearance screen is a speed change, not a semantic one
# --------------------------------------------------------------------------
def test_frame_clear_matches_exact(final6):
    spec = final6[2]
    boxes = spec.static_obstacles()
    assert boxes
    rng = np.random.default_rng(7)
    q = rng.uniform(FR3_MIN, FR3_MAX, size=(2500, 7))
    Twb = spec.T_world_base()
    T, p = ik.fk_batch(q)
    pw = p @ Twb[:3, :3].T + Twb[:3, 3]
    tip = T[:, :3, 3] + T[:, :3, :3] @ np.array([0.0, 0.0, 0.110])
    P10 = np.concatenate([pw, (tip @ Twb[:3, :3].T + Twb[:3, 3])[:, None, :]],
                         axis=1)
    exact = rig_final.chain_static_clearance(P10, boxes) >= rig_final.STATIC_MARGIN
    assert np.array_equal(tilt._frame_clear(P10, boxes), exact)


# --------------------------------------------------------------------------
# 4. the adaptive planner: inert when it should be, monotone when it is not
# --------------------------------------------------------------------------
def test_adaptive_returns_the_shipping_plan_when_flat_succeeds(sixarm):
    """Tilt enabled must not perturb a stroke the flat planner already draws."""
    spec = sixarm[31]
    poly = rim_arc(spec)
    o = {"ds_lattice": 0.012}
    base = stroke_api.plan_stroke(poly, spec, o)
    assert base["status"] == "ok"
    for tm in (0.0, 15.0, 30.0):
        r = tilt.plan_adaptive(poly, spec, tm, pitch_deg=7.5, opts=o)
        assert r["status"] == "ok"
        assert r["tilt_used"] is False
        assert r["n_knots"] == base["n_knots"]
        assert np.array_equal(r["qs"], base["qs"])


def test_adaptive_never_loses_a_stroke(final6):
    """A tilt attempt that does not certify returns the FLAT result, so the
    feature can add certified strokes and can never remove one."""
    spec = final6[2]
    poly = np.column_stack([np.linspace(1.05, 1.35, 40), np.full(40, 1.257)])
    flat = tilt.plan_adaptive(poly, spec, 0.0, opts={"ds_lattice": 0.008})
    for tm in (15.0, 30.0):
        r = tilt.plan_adaptive(poly, spec, tm, pitch_deg=7.5,
                               opts={"ds_lattice": 0.008})
        assert r["status"] in ("ok", flat["status"])


def test_certified_tilt_plan_stays_inside_its_cone(final6):
    """The lean is re-derived from the KINEMATICS, not read off the plan."""
    spec = final6[2]
    from aris_sixarm.metrics import GATE_MARGIN, GATE_SIGMA
    cx, cy = 1.2768, 1.2572
    th = np.linspace(np.deg2rad(170), np.deg2rad(240), 70)
    poly = np.column_stack([cx + 0.13 * np.cos(th), cy + 0.13 * np.sin(th)])
    o = dict(ds_lattice=0.008, margin_gate=GATE_MARGIN, sigma_gate=GATE_SIGMA)
    r = tilt.plan_adaptive(poly, spec, 15.0, pitch_deg=7.5, opts=o)
    assert r["status"] == "ok" and r["tilt_used"] is True
    lean, worst, inside = tilt.cone_check(r["qs"], spec, 15.0)
    assert inside and worst <= 15.0 + 1e-6
    assert r["validation"]["ok"]
    assert r["min_margin"] >= GATE_MARGIN - 1e-9
    assert r["tip_err"] < 2e-3


def test_flat_plan_reports_zero_lean(sixarm):
    """A perpendicular plan must measure as perpendicular — `cone_check` uses
    arctan2 rather than arccos so it does not report 1e-8 rad of phantom lean
    and fail its own cone."""
    spec = sixarm[31]
    r = stroke_api.plan_stroke(rim_arc(spec), spec, {"ds_lattice": 0.012})
    lean, worst, inside = tilt.cone_check(r["qs"], spec, 0.0)
    assert inside and worst < 1e-9


# --------------------------------------------------------------------------
# 5. the flag
# --------------------------------------------------------------------------
def test_flag_defaults_to_off_and_is_bit_identical(sixarm):
    """`tilt_max_deg = 0` must not merely behave like today — it must BE today."""
    spec = sixarm[31]
    poly = rim_arc(spec)
    a = stroke_api.plan_stroke(poly, spec, {"ds_lattice": 0.012})
    b = stroke_api.plan_stroke(poly, spec,
                               {"ds_lattice": 0.012, "tilt_max_deg": 0.0})
    assert a["status"] == b["status"] == "ok"
    assert np.array_equal(a["qs"], b["qs"])
    assert a["n_knots"] == b["n_knots"]
    assert stroke_api.DEFAULTS["tilt_max_deg"] == 0.0


def test_flag_on_rescues_a_donut_stroke(final6):
    """The flag is the whole feature, exercised through the shipping entry.

    A RADIAL stroke through arm 2's comfort donut is the case that needs no
    special gates to make the point: it splits at the pipeline's OWN gates
    (margin 0.15, sigma 0.10) because the fiber is empty at s = 0, and a
    15-degree cone draws it end to end.
    """
    spec = final6[2]
    t = np.linspace(0.06, 0.30, 60)[:, None]
    poly = np.array([1.2768, 1.2572]) + t * np.array([-1.0, 0.0])
    o = dict(ds_lattice=0.008)
    flat = stroke_api.plan_stroke(poly, spec, o)
    tilted = stroke_api.plan_stroke(poly, spec, dict(o, tilt_max_deg=15.0))
    assert flat["status"] == "split"
    assert tilted["status"] == "ok"
    assert tilted["tilt_used"] is True
    assert tilted["validation"]["ok"]
    assert tilted["max_lean_deg"] <= 15.0 + 1e-6
    assert tilted["tip_err"] < 2e-3


def test_strict_gates_reach_plan_stroke(final6):
    """`margin_gate` in `opts` used to be accepted and SILENTLY IGNORED — the
    band search and the certification chase both read `pwl.MARGIN_GATE` off the
    module — so "plan this stroke at the strict comfort gate" was a question
    the shipping entry point could not be asked, and the version of this test
    that shipped pinned the bug rather than the behaviour.  It now reaches all
    three places a gate has to arrive: the band DP, the 5 mm chase, and the
    independent validator.

    The arm-2 donut arc is the witness.  At the permissive gate it certifies
    flat with 0.158 rad of margin; asked for 0.30 it must SPLIT rather than
    hand back the 0.158 plan with a strict label on it."""
    from aris_sixarm.metrics import GATE_MARGIN
    spec = final6[2]
    th = np.linspace(np.deg2rad(170), np.deg2rad(240), 70)
    poly = np.column_stack([1.2768 + 0.13 * np.cos(th),
                            1.2572 + 0.13 * np.sin(th)])
    loose = stroke_api.plan_stroke(poly, spec, dict(ds_lattice=0.008))
    assert loose["status"] == "ok"
    assert loose["min_margin"] < GATE_MARGIN      # the stroke IS the witness

    strict = stroke_api.plan_stroke(
        poly, spec, dict(ds_lattice=0.008, margin_gate=GATE_MARGIN))
    assert strict["status"] == "split"            # honoured, not ignored
    if strict.get("head") is not None:
        assert strict["head"]["min_margin"] >= GATE_MARGIN - 1e-9

    # the same question through the tilt planner, which always could ask it
    tilted = tilt.plan_adaptive(poly, spec, 0.0,
                                opts=dict(ds_lattice=0.008,
                                          margin_gate=GATE_MARGIN))
    assert tilted["status"] == "split"

    # and a gate the stroke DOES meet still certifies, so the plumbing is a
    # gate and not a blanket refusal
    ok = stroke_api.plan_stroke(
        poly, spec, dict(ds_lattice=0.008, margin_gate=0.15))
    assert ok["status"] == "ok" and ok["min_margin"] >= 0.15
