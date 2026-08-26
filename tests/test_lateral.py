"""The LATERAL pen holder: tool math, broken degeneracy, certification.

The holder puts the tip at TCP + R @ (0.110, 0, 0.110) — 11 cm off the wrist
axis along hand x (USER-SPECIFIED, 2026-08-25), the axial 0.110 a
user-confirmed estimate carried from the inline pen's gate-B touchdown.  The
INLINE pen stays the repo default (frames.PEN_LAT == 0.0), so every pinned
number elsewhere is untouched; these tests cover the new tool model and the
switch itself.
"""
import numpy as np
import pytest

from aris_sixarm import frames, lateral, metrics, planner, stroke_api, validate
from aris_sixarm import coordination, rig_final, scene_check
from aris_sixarm.fleet import FLEET_FINAL
from aris_sixarm.pwl import pen_down_poses

LAT = frames.PEN_LAT_HOLDER
SPEC = FLEET_FINAL[13]
LINE = np.column_stack([np.linspace(0.65, 0.95, 20), np.full(20, 0.30)])


@pytest.fixture
def lateral_active():
    """Activate the lateral tool for one test and ALWAYS restore inline."""
    frames.activate_tool("lateral")
    try:
        yield
    finally:
        frames.activate_tool("inline")


# --------------------------------------------------------------------------
# the tool transform
# --------------------------------------------------------------------------
def test_default_is_inline():
    assert frames.PEN_LAT == 0.0 and frames.ACTIVE_TOOL == "inline"
    assert np.allclose(frames.tool_offset(), [0.0, 0.0, frames.PEN_EXT])


def test_tip_pos_lateral_matches_manual():
    q = frames.Q_READY_INV_FINAL
    T, _ = frames.fk(q)
    want = T[:3, 3] + T[:3, :3] @ np.array([LAT, 0.0, frames.PEN_EXT])
    assert np.allclose(frames.tip_pos(q, pen_lat=LAT), want, atol=1e-15)
    assert np.allclose(frames.tip_pos_many(q[None], pen_lat=LAT)[0], want,
                       atol=1e-15)


def test_inline_tip_unchanged():
    q = frames.Q_READY_FLOOR
    T, _ = frames.fk(q)
    legacy = T[:3, 3] + T[:3, :3] @ np.array([0.0, 0.0, frames.PEN_EXT])
    assert np.allclose(frames.tip_pos(q), legacy, atol=0)


def test_lateral_jacobian_matches_fd():
    qs = np.array([frames.Q_READY_FLOOR, frames.Q_READY_INV_FINAL,
                   frames.Q_READY_WALL])
    Ja = metrics.tip_jacobian_many(qs, pen_lat=LAT)
    Jf = np.array([metrics.tip_jacobian(q, pen_ext=frames.PEN_EXT, pen_lat=LAT)
                   for q in qs])
    assert np.abs(Ja - Jf).max() < 1e-9


def test_tool_points_many_widths():
    T, _ = frames.fk_many(frames.Q_READY_FLOOR[None])
    assert len(frames.tool_points_many(T, frames.PEN_EXT, 0.0)) == 1
    tool = frames.tool_points_many(T, frames.PEN_EXT, LAT)
    assert len(tool) == 2
    # corner sits pen_ext ABOVE the tip when the pen is vertical is not true
    # in the base frame generally — but tip - corner must be pen_ext * z_tool
    z_tool = T[0, :3, 2]
    assert np.allclose(tool[0][0] - tool[1][0], frames.PEN_EXT * z_tool,
                       atol=1e-12)


# --------------------------------------------------------------------------
# the degeneracy is BROKEN
# --------------------------------------------------------------------------
def test_phi_moves_tcp_on_circle():
    """Same tip, phi rotated: the TCP travels on an 11 cm circle."""
    Twb_inv = np.linalg.inv(SPEC.T_world_base())
    pt = np.array([[0.9, 0.4]])
    p0 = pen_down_poses(pt, Twb_inv, 0.110, phi=0.0, pen_lat=LAT)[0, :3, 3]
    p9 = pen_down_poses(pt, Twb_inv, 0.110, phi=np.pi / 2,
                        pen_lat=LAT)[0, :3, 3]
    p18 = pen_down_poses(pt, Twb_inv, 0.110, phi=np.pi, pen_lat=LAT)[0, :3, 3]
    assert np.linalg.norm(p9 - p0) == pytest.approx(LAT * np.sqrt(2), abs=1e-9)
    assert np.linalg.norm(p18 - p0) == pytest.approx(2 * LAT, abs=1e-9)


def test_inline_phi_is_noop_on_tcp_position():
    """With the inline pen phi only spins the pose in place (the q7 alias)."""
    Twb_inv = np.linalg.inv(SPEC.T_world_base())
    pt = np.array([[0.9, 0.4]])
    p0 = pen_down_poses(pt, Twb_inv, 0.110, phi=0.0, pen_lat=0.0)[0, :3, 3]
    p9 = pen_down_poses(pt, Twb_inv, 0.110, phi=1.3, pen_lat=0.0)[0, :3, 3]
    assert np.allclose(p0, p9, atol=1e-15)


# --------------------------------------------------------------------------
# certification with the lateral tool
# --------------------------------------------------------------------------
def test_lateral_stroke_certifies():
    r = stroke_api.plan_stroke(LINE, SPEC, dict(pen_lat=LAT))
    assert r["status"] == "ok"
    assert r["validation"]["ok"]
    assert r["pen_lat"] == pytest.approx(LAT)
    assert "phi" in r and np.isfinite(r["phi"])
    assert r["tip_err"] < 1e-6
    assert r["lateral"]["rescue_used"] is False


def test_default_plan_has_no_lateral_detour():
    r = stroke_api.plan_stroke(LINE, SPEC)
    assert r["status"] == "ok"
    assert r["pen_lat"] == 0.0 and r["phi"] == 0.0
    assert "lateral" not in r


def test_validator_catches_wrong_tool():
    """An inline plan re-validated as lateral misses the curve by pen_lat."""
    r = stroke_api.plan_stroke(LINE, SPEC)
    rep = validate.validate_plan(r["pts"], SPEC, r["qs"],
                                 pen_ext=frames.PEN_EXT, pen_lat=LAT)
    assert not rep["ok"]
    assert any(v["kind"] == "tip_off_curve" for v in rep["violations"])
    assert rep["worst"]["tip_err"] == pytest.approx(LAT, abs=1e-3)


def test_reverse_lateral_plan():
    r = stroke_api.plan_stroke(LINE, SPEC, dict(pen_lat=LAT))
    rv = stroke_api.reverse_plan(r, SPEC, dict(pen_lat=LAT))
    assert rv["status"] == "ok" and rv["validation"]["ok"]
    assert rv["phi"] == pytest.approx(r["phi"])
    assert np.allclose(rv["qs"], r["qs"][::-1])


def test_coupled_phi_lattice_plans():
    """The (s x phi x q7 x branch) DP spans a short stroke."""
    pts = np.column_stack([np.linspace(0.65, 0.80, 16), np.full(16, 0.30)])
    lat = lateral.build_lattice_phi(pts, SPEC, lateral.phi_ring(pts, SPEC),
                                    pen_ext=frames.PEN_EXT, pen_lat=LAT)
    res = lateral.plan_phi(lat)
    assert res["ok"]
    assert res["bottleneck"] >= planner.HARD_SIGMA
    # the path is continuous in q
    assert np.max(np.abs(np.diff(res["qs"], axis=0))) <= planner.JUMP_THRESH


# --------------------------------------------------------------------------
# the capsule model and the tool switch
# --------------------------------------------------------------------------
def test_lateral_capsule_tables_pinned_together():
    assert scene_check.RADII_LAT[-2:] == ((8, 10, rig_final.BRACKET_R_LAT),
                                          (10, 9, rig_final.PEN_R_LAT))
    assert coordination.CAPSULES_LAT[-2:] == (
        (8, 10, rig_final.PEN_R_FINAL), (10, 9, rig_final.PEN_R_FINAL))
    assert rig_final.STATIC_CAPSULES_LAT[-2:] == (
        (8, 10, rig_final.BRACKET_R_LAT), (10, 9, rig_final.PEN_R_LAT))
    # the arm segments stay identical to the inline tables
    assert scene_check.RADII_LAT[:-2] == scene_check.RADII[:-1]
    assert coordination.CAPSULES_LAT[:-2] == coordination.CAPSULES[:-1]


def test_chain_width_follows_active_tool(lateral_active):
    P = scene_check._chain(frames.Q_READY_FLOOR, SPEC, None, frames.PEN_EXT)
    assert P.shape == (11, 3)
    Pc = coordination.chain_world(frames.Q_READY_FLOOR[None], SPEC)
    assert Pc.shape == (1, 11, 3)


def test_chain_width_inline():
    P = scene_check._chain(frames.Q_READY_FLOOR, SPEC, None, frames.PEN_EXT)
    assert P.shape == (10, 3)


def test_chain_static_clearance_selects_by_width():
    boxes = [dict(name="b", lo=np.array([-1.0, -1, -1]),
                  hi=np.array([1.0, 1, 1]), source="", tag="")]
    P10 = np.zeros((1, 10, 3)) + 2.0
    P11 = np.zeros((1, 11, 3)) + 2.0
    d10 = rig_final.chain_static_clearance(P10, boxes)
    d11 = rig_final.chain_static_clearance(P11, boxes)
    # same geometry, both finite; the 11-wide one used the LAT table without
    # raising on index 10
    assert np.isfinite(d10[0]) and np.isfinite(d11[0])


def test_activate_tool_roundtrip():
    assert frames.activate_tool("lateral") == pytest.approx(LAT)
    assert frames.PEN_LAT == pytest.approx(LAT)
    assert frames.activate_tool("inline") == 0.0
    assert frames.PEN_LAT == 0.0
    with pytest.raises(ValueError):
        frames.activate_tool("banana")


def test_check_pose_pen_below_paper_sees_lateral_tip(lateral_active):
    """The ready-pose pen-below-paper check must measure the LATERAL tip."""
    rep = validate.check_pose(frames.Q_READY_FLOOR, SPEC)
    T, _ = frames.fk(frames.Q_READY_FLOOR)
    Twb = SPEC.T_world_base()
    tip = Twb[:3, :3] @ (T[:3, 3] + T[:3, :3]
                         @ np.array([LAT, 0.0, frames.PEN_EXT])) + Twb[:3, 3]
    assert rep["worst"]["tip_z"] == pytest.approx(float(tip[2]), abs=1e-12)


# --------------------------------------------------------------------------
# reach: the lateral tool extends it
# --------------------------------------------------------------------------
def test_lateral_extends_reach():
    """Along a ray from the base, the lateral tool certifies farther cells."""
    from aris_sixarm import atlas
    spec = SPEC
    Twb = spec.T_world_base()
    Twb_inv = np.linalg.inv(Twb)
    cand = atlas._candidates(0.0)
    bx, by = spec.xy

    def reach(pen_lat):
        far = 0.0
        for r in np.arange(0.55, 1.00, 0.03):
            x, y = bx, by + r          # straight up the sheet from the base
            got = atlas.solve_cell(x, y, Twb, Twb_inv, spec, cand,
                                   pen_ext=frames.PEN_EXT, boxes=(),
                                   pen_lat=pen_lat)
            if got is not None:
                far = r
        return far

    r_in, r_lat = reach(0.0), reach(LAT)
    assert r_lat >= r_in + 0.05, (r_in, r_lat)


# ==========================================================================
# THE PEN LEAN, THROUGH THE LATERAL PLANNER (2026-08-26)
# ==========================================================================
# `plan_adaptive` used to drop `tilt_max_deg` with a note.  What these pin is
# that restoring it is additive: flat still plans flat, byte for byte; a lean
# is only ever reached when flat has already failed; the lean the plan uses is
# the lean the validator is told about and no more; and the least lean that
# certifies is the one that wins.
def test_a_cone_does_not_change_a_stroke_that_certifies_flat():
    """The fa0544d discipline: an opt-in must not move a shipped number."""
    spec, poly = SPEC, LINE
    flat = lateral.plan_adaptive(poly, spec,
                                 opts=dict(pen_lat=LAT, validate=True))
    coned = lateral.plan_adaptive(poly, spec,
                                  opts=dict(pen_lat=LAT, tilt_max_deg=15.0,
                                            validate=True))
    assert flat["status"] == "ok"
    assert coned["status"] == "ok"
    assert np.array_equal(flat["qs"], coned["qs"])
    assert coned["lean_deg"] == 0.0
    assert coned["lateral"]["lean_used"] is False
    assert coned["lateral"]["cone_deg"] == 15.0


def test_the_lean_ring_is_the_atlas_cone():
    from aris_sixarm import atlas
    assert lateral.LEAN_GRID_DEG == atlas.GATE_CONE_DEG
    for deg in (2.5, 15.0):
        ring = lateral.lean_ring(deg)
        assert len(ring) == lateral.LEAN_DIRS
        for t in ring:
            assert planner.lean_deg(t) == pytest.approx(deg, abs=1e-9)


def test_a_pinned_lean_is_exactly_the_lean_the_pen_gets():
    """|tilt| is the angle from vertical, by construction and by measurement."""
    spec, poly = SPEC, LINE
    for deg in (2.5, 7.5, 15.0):
        for tilt in lateral.lean_ring(deg):
            r = stroke_api.plan_stroke(poly, spec,
                                       dict(pen_lat=LAT, phi=0.0, tilt=tilt,
                                            validate=True))
            if r["status"] != "ok":
                continue
            assert r["lean_deg"] == pytest.approx(deg, abs=1e-9)
            # the INDEPENDENT validator re-derives it from the FK
            got = r["validation"]["worst"]["max_lean_deg"]
            assert got == pytest.approx(deg, abs=1e-6)
            return
    pytest.skip("no leaned plan certified on this stroke")


def test_the_validator_refuses_a_lean_it_was_not_told_about():
    spec, poly = SPEC, LINE
    tilt = lateral.lean_ring(10.0)[0]
    r = stroke_api.plan_stroke(poly, spec, dict(pen_lat=LAT, phi=0.0,
                                                tilt=tilt, validate=True))
    if r["status"] != "ok":
        pytest.skip("stroke does not certify at this lean")
    from aris_sixarm.validate import validate_plan
    strict = validate_plan(r["pts"], spec, r["qs"], pen_lat=r["pen_lat"],
                           tilt_max_deg=0.0)
    assert not strict["ok"]
    assert any(v["kind"] == "pen_cone" for v in strict["violations"])
    loose = validate_plan(r["pts"], spec, r["qs"], pen_lat=r["pen_lat"],
                          tilt_max_deg=r["lean_deg"])
    assert loose["ok"]


def test_the_ladder_climbs_and_stops_at_the_first_lean_that_works():
    """Least lean that certifies, the same rule `atlas.solve_cell` follows.

    Sourced from wherever this arm's own fiber runs out rather than from an
    atlas file, so the test needs no artefact and no rig environment.
    """
    spec = SPEC
    leaned = tried_flat = 0
    grid = [(x, y) for y in np.linspace(0.10, 0.60, 6)
            for x in np.linspace(0.20, 1.40, 25)]
    for x, y in grid:
        poly = np.column_stack([[x - 0.015, x + 0.015], [y, y]])
        flat = lateral.plan_adaptive(poly, spec,
                                     opts=dict(pen_lat=LAT, validate=True))
        if flat["status"] == "ok":
            continue
        tried_flat += 1
        r = lateral.plan_adaptive(poly, spec,
                                  opts=dict(pen_lat=LAT, tilt_max_deg=15.0,
                                            validate=True))
        if r["status"] == "ok" and r["lateral"].get("lean_used"):
            leaned += 1
            assert r["lean_deg"] in lateral.LEAN_GRID_DEG
            assert r["validation"]["ok"]
            # THE LADDER CLIMBS AND STOPS.  Every full plan it attempted was
            # at a lean no larger than the one that won, the attempts are in
            # ascending order, and the LAST one is the winner — smaller leans
            # were either screened out with an empty fiber (so no plan is
            # recorded for them) or planned and refused, never skipped over.
            tried = [a[0] for a in r["lateral"]["lean_tried"]]
            assert tried == sorted(tried)
            assert max(tried) == r["lean_deg"]
            assert all(d <= r["lean_deg"] + 1e-9 for d in tried)
        if leaned >= 3:
            break
    if not tried_flat:
        pytest.skip("no stroke on this grid fails flat")
    assert leaned > 0, "the ladder recovered nothing where flat failed"
