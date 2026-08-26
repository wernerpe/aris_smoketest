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

from aris_sixarm import (fleet, frames, layout, mounts, rig_final, stroke_api,
                         validate)
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
    # rather than handing back an ungated pose.  The height is the SPEC's now
    # (`study_spec` writes the base pose in), so asking about 4 m means
    # building the arm at 4 m — and asking the 0.85 m arm about 4 m is a
    # contradiction the function refuses instead of quietly ignoring.
    high = layout.study_spec(spec.arm_id, "inv", spec.xy, h=4.0)
    assert high.T_world_base()[2, 3] == 4.0
    with pytest.raises(RuntimeError):
        layout.certified_ready_pose(high)
    with pytest.raises(ValueError):
        layout.certified_ready_pose(spec, 4.0)


# ---------------------------------------------------------------------------
# THE h_inv TRAP (2026-08-25)
# ---------------------------------------------------------------------------
# `ArmSpec.T_world_base(h_inv=H_INV_DEFAULT)` defaults to 1.00 m, and v2 built
# the study specs with `z=None, R=None`.  Every layout-study call site passes
# `h_inv=LAYOUT_PROPOSED["h"]`, so the study was right — but the GENERIC draw
# pipeline (planner, sequence, coordination, idle, scene_check, viz) never
# passes it, so `ARIS_RIG=proposed scripts/draw.py` planned the whole fleet
# 15 cm above where it is bolted, with its own mount boxes still at 0.850.
# `study_spec` now writes the base pose into the spec.  These pin BOTH halves:
# the bare call is right, and the explicit call did not move.
def test_proposed_specs_know_their_own_height_without_being_told():
    h = layout.LAYOUT_PROPOSED["h"]
    assert h == 0.850
    for aid, spec in sorted(layout.FLEET_PROPOSED.items()):
        T = spec.T_world_base()                     # NO ARGUMENT — the trap
        assert T[2, 3] == h, (aid, T[2, 3])
        # and hanging, not standing: the legacy inverted seat's rotation
        assert np.allclose(T[:3, :3], frames.roty(np.pi) @ frames.rotz(spec.yaw),
                           atol=0), aid
        assert np.array_equal(T, spec.T_world_base(h))
        assert np.array_equal(T, spec.T_world_base(1.00))   # explicit-R branch


def test_the_pipeline_default_h_inv_now_reaches_the_right_base():
    """The bug was never in `T_world_base` — it was in every caller that had
    no height to pass.  `coordination.chain_world` is one of them, and it is
    the one the conductor measures every inter-arm clearance with."""
    from aris_sixarm import coordination
    h = layout.LAYOUT_PROPOSED["h"]
    spec = layout.FLEET_PROPOSED[31]
    q = np.asarray(spec.q_seed, float)[None, :]
    bare = coordination.chain_world(q, spec)         # h_inv=H_INV_DEFAULT=1.00
    told = coordination.chain_world(q, spec, h_inv=h)
    assert np.array_equal(bare, told)
    # the shoulder hangs BELOW the 0.850 mount plane, not below a 1.00 one
    assert bare[0, :, 2].max() <= h + 1e-9


def test_every_swept_height_reproduces_the_legacy_transform_exactly():
    """Nothing in the certified study shifts: for every h the study swept and
    both families, the explicit-R branch returns bit-for-bit what the legacy
    branch computed from the same inputs."""
    def legacy(spec, h_inv):
        T = np.eye(4)
        if spec.mount == "floor":
            T[:3, :3] = frames.rotz(spec.yaw)
            T[:3, 3] = [*spec.xy, fleet.Z_FLOOR_BASE]
        else:
            T[:3, :3] = frames.roty(np.pi) @ frames.rotz(spec.yaw)
            T[:3, 3] = [*spec.xy, h_inv]
        return T

    for h in (0.850, 0.922, 1.000):
        for lay in (layout.paired_grid(h=h), dict(layout.LAYOUT_V1, h=h)):
            for aid, spec in layout.build_fleet(lay).items():
                assert np.array_equal(spec.T_world_base(h), legacy(spec, h)), \
                    (h, aid)


# ---------------------------------------------------------------------------
# THE PARKED FLEET (2026-08-25)
# ---------------------------------------------------------------------------
# `spec.q_seed` is the home the sequencer flies back to and the configuration
# every arm HOLDS for every phase it is not drawing in, so it is in the
# conductor's collision images from t = 0.  The all-ceiling rig cannot inherit
# the mount default and cannot aim all six at the canvas centre; these pin both
# refusals with the numbers that motivate them.
LAT = frames.PEN_LAT_HOLDER


def _park_clearance(poses, fl):
    """Min capsule clearance between any two arms holding `poses`.  Clipped
    at `coordination.BROAD_CAP` — a value AT the cap means "at least"."""
    from aris_sixarm import coordination
    paths = {aid: coordination.ArmPath(aid, np.asarray(q, float)[None, :],
                                       0.05, spec=fl[aid])
             for aid, q in poses.items()}
    ids = sorted(poses)
    return min(float(np.min(coordination.clearance_matrix(paths[a], paths[b])))
               for i, a in enumerate(ids) for b in ids[i + 1:])


@pytest.fixture
def lateral():
    """The proposed rig's tool, for the duration of one test only."""
    frames.activate_tool("lateral")
    try:
        yield frames.PEN_LAT
    finally:
        frames.activate_tool("inline")


def test_the_parked_fleet_does_not_park_inside_the_table():
    """The legacy inverted seed is NOT valid at h = 0.850 with the lateral
    holder — it is 61.6 mm under the paper and 0.184 of joint margin, under
    the 0.30 gate.  Six arms would park inside the table.  This pins the
    refusal AND the replacement."""
    bad = frames.Q_READY_INV
    spec = layout.FLEET_PROPOSED[31]
    rep = validate.check_pose(bad, spec, pen_lat=LAT)
    assert not rep["ok"]
    assert rep["worst"]["tip_z"] < -0.05          # under the paper
    assert frames.joint_margin(bad) < 0.30

    for aid, spec in sorted(layout.FLEET_PROPOSED.items()):
        q = spec.q_seed                            # what the pipeline reads
        assert np.array_equal(q, np.asarray(layout.Q_PARK_PROPOSED[aid], float))
        rep = validate.check_pose(q, spec, pen_lat=LAT)
        assert rep["ok"], (aid, rep)
        assert rep["worst"]["tip_z"] >= 0.09, aid          # hovering, not down
        assert frames.joint_margin(q) >= 0.30, aid
        assert rep["worst"]["min_frame_clearance"] >= 0.30, aid


def test_the_parked_fleet_does_not_park_inside_itself(lateral):
    """WHY THE BEARING IS OUTWARD.  Aimed at the canvas centre — which is what
    one arm alone wants — the six park in a huddle and the closest pair
    overlaps by 95.6 mm.  Away from the fleet centroid they stand off towards
    their own rims and hold 181 mm, against the 80 mm the conductor asks of
    every pair while they MOVE."""
    fl = layout.FLEET_PROPOSED
    from aris_sixarm.coordination import SAFETY_M, CALIB_M
    assert _park_clearance(layout.Q_PARK_PROPOSED, fl) >= SAFETY_M + CALIB_M

    inward = {aid: layout.certified_ready_pose(s, pen_lat=LAT)[0]
              for aid, s in sorted(fl.items())}
    assert _park_clearance(inward, fl) < 0.0       # INTERPENETRATING

    # and the function will not hand back a fleet that does that.  0.20 m of
    # hover is the near miss it was written for: six gated poses, 4 mm apart.
    bare = layout.build_fleet(layout.LAYOUT_PROPOSED)
    with pytest.raises(RuntimeError, match="park .* mm apart"):
        layout.certified_park_poses(bare, hover=0.20, pen_lat=LAT)


def test_baked_park_poses_are_that_functions_own_output():
    """The literals in `layout.py` are `certified_park_poses`' output on
    `LAYOUT_PROPOSED` at `PARK_GRID_PROPOSED`, so they cannot drift from the
    recipe that made them.  Derived from the BARE fleet: the park pose is also
    the IK seed, and a fleet already carrying one would be seeded by its own
    answer.

    WHAT THIS DOES NOT RE-RUN is the (radius, hover) SEARCH behind the grid —
    6 radii x 3 hovers x 24 cells x two directions of `paper.route` per arm is
    a quarter of an hour, and it needs an atlas out of gitignored `out/`.  The
    grid's scores are recorded where it is defined; what is pinned here is
    that the six poses are what that grid produces.
    """
    bare = layout.build_fleet(layout.LAYOUT_PROPOSED)
    assert all(np.array_equal(s.q_seed, frames.Q_READY_INV)
               for s in bare.values()), "derive from the LEGACY seed"
    made = layout.certified_park_poses(bare, layout.PARK_GRID_PROPOSED,
                                       pen_lat=LAT)
    assert sorted(made) == sorted(layout.Q_PARK_PROPOSED)
    for aid, q in made.items():
        assert np.allclose(q, layout.Q_PARK_PROPOSED[aid], atol=5e-5), aid

    # each arm stands off along its OWN outward bearing at its own radius, and
    # holds the pen at its own hover — the three numbers the grid records
    cent = np.mean([s.xy for s in bare.values()], axis=0)
    for aid, q in layout.Q_PARK_PROPOSED.items():
        r, hv = layout.PARK_GRID_PROPOSED[aid]
        T = frames.fk(np.asarray(q, float))[0]
        tip = T[:3, 3] + T[:3, :3] @ frames.tool_offset(pen_lat=LAT)
        Twb = layout.FLEET_PROPOSED[aid].T_world_base()
        w = Twb[:3, :3] @ tip + Twb[:3, 3]
        assert np.allclose(w[:2], layout.PARK_HOVER_PROPOSED[aid], atol=1e-3)
        assert abs(w[2] - hv) < 1e-4, aid      # the literals are 4 decimals
        b = np.asarray(bare[aid].xy, float)
        u = (b - cent) / np.linalg.norm(b - cent)
        # on the bearing, at the radius (or clipped to the sheet edge, which
        # is why this is a bound and not an equality)
        assert float(np.dot(w[:2] - b, u)) > 0.5 * r, aid
        assert float(np.linalg.norm(w[:2] - b)) <= r + 1e-4, aid
