"""The layout study's registered artefacts stay sane.

The PROPOSED rig is a study output, not a build: these tests pin that it
stays env-selectable, constraint-clean, and plannable with the lateral tool
— so a later edit cannot silently break the recommendation the redesign is
based on (docs/LAYOUT_STUDY.md).

v2 (2026-08-25) added the MOUNT HARDWARE, so they also pin that the proposed
fleet carries its neighbours' steel and that every arm's certified ready pose
clears it.
"""
import numpy as np
import pytest

from aris_sixarm import fleet, frames, layout, mounts, rig_final, stroke_api
from aris_sixarm.rig_final6 import SHEET_FINAL6


def test_proposed_rig_is_selectable_and_not_default():
    fl, sheet = fleet.rig("proposed")
    assert sorted(fl) == [2, 13, 17, 31, 71, 97]
    assert sheet == SHEET_FINAL6
    assert fleet.ACTIVE_RIG != "proposed"      # never the default
    assert "proposed" in fleet.RIG_NAMES


def test_proposed_layout_constraints_clean():
    assert layout.check_spacing(layout.LAYOUT_PROPOSED) == []
    W, H = SHEET_FINAL6
    for x, y in layout.LAYOUT_PROPOSED["floor"]:
        d = np.hypot(max(0.0 - x, x - W, 0.0), max(0.0 - y, y - H, 0.0))
        assert d >= layout.FLOOR_SETBACK[0] - 1e-9
    assert layout.LAYOUT_PROPOSED["h"] in (0.85, 0.922, 1.00)
    assert len(layout.LAYOUT_PROPOSED["floor"]) \
        + len(layout.LAYOUT_PROPOSED["inv"]) == 6


def test_v1_layout_is_kept_for_the_honest_comparison():
    """docs/LAYOUT_STUDY.md quotes v1's 99.38 % and its re-score under
    mounts; the coordinates behind both must not drift."""
    assert layout.check_spacing(layout.LAYOUT_V1) == []
    assert len(layout.LAYOUT_V1["floor"]) == 2
    assert len(layout.LAYOUT_V1["inv"]) == 4
    assert layout.LAYOUT_V1["h"] == 0.850


def test_study_specs_carry_the_other_arms_mounts_on_the_merged_canvas():
    fl, _ = fleet.rig("proposed")
    for aid, spec in fl.items():
        boxes = spec.static_obstacles()
        assert boxes, "v2 specs are NOT green field any more"
        assert {b["tag"] for b in boxes} == {f"mount:{o}" for o in fl
                                             if o != aid}
        assert fleet.sheet_for(spec) == SHEET_FINAL6
        assert spec.mount in ("floor", "inv")


def test_both_families_build_six_arms_with_distinct_ids():
    for lay in (layout.LAYOUT_V1,
                dict(floor=[], inv=[(0.6, 0.5 + 0.6 * k) for k in range(6)],
                     h=0.85)):
        fl = layout.build_fleet(lay)
        assert len(fl) == 6 and len(set(fl)) == 6
        assert sorted(fl) == [2, 13, 17, 31, 71, 97]
        fids, iids = layout.arm_ids(lay)
        assert len(fids) == len(lay["floor"])
        assert len(iids) == len(lay["inv"])
        assert all(fl[a].mount == "floor" for a in fids)
        assert all(fl[a].mount == "inv" for a in iids)


def test_paired_grid_is_the_all_ceiling_optimum_in_round_numbers():
    W, Hs = SHEET_FINAL6
    lay = layout.paired_grid(h=0.85)
    assert lay["floor"] == [] and len(lay["inv"]) == 6
    assert layout.check_spacing(lay) == []
    xs = sorted({round(x, 6) for x, _ in lay["inv"]})
    ys = sorted({round(y, 6) for _, y in lay["inv"]})
    assert len(xs) == 2 and len(ys) == 3
    # two columns straddling the centre line at the pair spacing
    assert xs[1] - xs[0] == pytest.approx(layout.PAIR_SPACING)
    assert (xs[0] + xs[1]) / 2 == pytest.approx(W / 2)
    # three rows at the centres of an even tiling of the canvas length
    assert ys == pytest.approx([(2 * j + 1) * Hs / 6 for j in range(3)])
    assert layout.pair_spacing_of(lay) == pytest.approx(layout.PAIR_SPACING)


def test_pair_spacing_window_covers_the_under_base_hole():
    """The window is geometry, not a preference: a partner must cover the
    whole of an arm's r < r0 under-base hole without falling outside its own
    outer radius."""
    r0, r1 = layout.PROFILES_LAT[("inv", 0.850)]
    lo, hi = layout.PAIR_WINDOW
    assert lo == pytest.approx(2 * r0)          # far lip clears r0
    assert hi == pytest.approx(r1 - r0)         # near lip stays inside r1
    assert lo <= layout.PAIR_SPACING <= hi


def test_proposed_inverted_arm_plans_lateral_stroke():
    fl, _ = fleet.rig("proposed")
    aid = next(a for a, s in fl.items() if s.mount == "inv")
    spec = fl[aid]
    bx, by = spec.xy
    h = layout.LAYOUT_PROPOSED["h"]
    line = np.column_stack([np.linspace(bx + 0.30, bx + 0.60, 16),
                            np.full(16, by)])
    r = stroke_api.plan_stroke(line, spec,
                               dict(pen_lat=frames.PEN_LAT_HOLDER, h_inv=h))
    assert r["status"] == "ok", (r["status"], r.get("reason"))
    assert r["validation"]["ok"]
    assert r["min_sigma"] >= 0.10 and r["min_margin"] >= 0.15


def test_every_proposed_ready_pose_clears_every_other_mount():
    """The v1 study could not make this check — there was no hardware to
    check against.  The pose is the one the scene draws."""
    # NOTE: no `activate_tool` here.  `certified_ready_pose` takes the tool
    # explicitly, so this test cannot leak `frames.PEN_LAT` into every module
    # that runs after it — which is exactly what it used to do.
    assert frames.ACTIVE_TOOL == "inline", "the global tool must be untouched"
    fl, _ = fleet.rig("proposed")
    h = layout.LAYOUT_PROPOSED["h"]
    for aid, spec in sorted(fl.items()):
        q, xy, rep = layout.certified_ready_pose(spec, h)
        assert rep["ok"] and rep["worst"]["tip_z"] > 0.0
        T, pts = frames.fk(q)
        tool = frames.tool_points_many(T[None], pen_lat=frames.PEN_LAT_HOLDER)
        P = np.vstack([pts] + list(tool))
        Twb = spec.T_world_base(h)
        Pw = (Twb[:3, :3] @ P.T).T + Twb[:3, 3]
        cl = rig_final.chain_static_clearance(Pw, spec.static_obstacles())[0]
        assert cl >= rig_final.STATIC_MARGIN, (aid, float(cl))
    assert frames.ACTIVE_TOOL == "inline" and frames.PEN_LAT == 0.0


def test_the_proposed_layout_leaves_the_booms_out_of_reach():
    """The study's central physical claim, restated as a gate: at the
    proposed height no certified drawing pose can touch a neighbour's boom
    or plate, so the only hardware that costs coverage is at the pen's
    level."""
    h = layout.LAYOUT_PROPOSED["h"]
    for spec in layout.build_fleet(layout.LAYOUT_PROPOSED).values():
        base_z = mounts.MOUNTS.z_floor if spec.mount == "floor" else h
        active, headroom = mounts.boom_shadow_active(spec.mount, base_z, h)
        assert not active and headroom > 0, (spec.arm_id, headroom)


def test_certified_ready_pose_is_gated_not_merely_solved():
    fl, _ = fleet.rig("proposed")
    spec = next(s for s in fl.values() if s.mount == "inv")
    q, xy, rep = layout.certified_ready_pose(spec,
                                             layout.LAYOUT_PROPOSED["h"])
    assert q.shape == (7,)
    assert frames.joint_margin(q) >= 0.30
    assert rep["worst"]["min_chain_z"] >= 0.02
    # an inverted arm hung 4 m up cannot certify anything: the solver RAISES
    # rather than handing back an ungated pose
    with pytest.raises(RuntimeError):
        layout.certified_ready_pose(spec, 4.0)
