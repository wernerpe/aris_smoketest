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
    assert layout.LAYOUT_PROPOSED["h"] in (0.85, 0.922, 0.940, 1.00)
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
        assert {b["tag"] for b in boxes} == \
            {f"mount:{o}" for o in fl if o != aid} \
            | {f"body:{o}" for o in fl if o != aid}
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
    r0, r1 = layout.PROFILES_LAT[("inv", layout.LAYOUT_PROPOSED["h"])]
    lo, hi = layout.PAIR_WINDOW
    assert lo == pytest.approx(2 * r0)          # far lip clears r0
    assert hi == pytest.approx(r1 - r0)         # near lip stays inside r1
    assert lo <= layout.PAIR_SPACING <= hi


def test_proposed_inverted_arm_plans_lateral_stroke():
    """A stroke out along the arm's own annulus, AWAY from its partner.

    The direction matters now and it did not before.  This test used to run
    the line towards +x, which on the paired grid is straight at the
    transverse partner 0.61 m away: with the partner's base column in the
    obstacle set (`mounts.arm_column_box`) that line ends INSIDE another
    robot, and the planner says so — see the second half.
    """
    fl, _ = fleet.rig("proposed")
    aid = next(a for a, s in fl.items() if s.mount == "inv")
    spec = fl[aid]
    bx, by = spec.xy
    h = layout.LAYOUT_PROPOSED["h"]
    opts = dict(pen_lat=frames.PEN_LAT_HOLDER, h_inv=h)
    line = np.column_stack([np.full(16, bx),
                            np.linspace(by + 0.30, by + 0.60, 16)])
    r = stroke_api.plan_stroke(line, spec, opts)
    assert r["status"] == "ok", (r["status"], r.get("reason"))
    assert r["validation"]["ok"]
    assert r["min_sigma"] >= 0.10 and r["min_margin"] >= 0.15

    # ...and the partner-ward line, which the empty-air planner used to
    # certify, is refused: the far end is under the partner's shoulder
    at_partner = np.column_stack([np.linspace(bx + 0.30, bx + 0.60, 16),
                                  np.full(16, by)])
    other = min((a for a in fl if a != aid),
                key=lambda a: np.hypot(*(np.asarray(fl[a].xy) - [bx, by])))
    assert abs(fl[other].xy[0] - (bx + 0.61)) < 1e-6
    bad = stroke_api.plan_stroke(at_partner, spec, opts)
    assert bad["status"] != "ok", bad["status"]


def test_the_inward_ready_pose_is_gone_for_the_middle_row():
    """WHAT THE CORRECTED CAPSULES TOOK AWAY (2026-08-26).

    `certified_ready_pose` aims at the canvas CENTRE, and on this grid that
    walks arm 31 straight at arm 71's base column.  Under the old 0.09 m
    capsules it still certified — at the margin itself, 0.0500 m, which the
    docstring of this test used to call "float noise on a gate the pose now
    sits exactly on".  It was not noise.  The mesh audit says that column is
    0.155 m and the arm reaching at it is 0.130 m, and with the true widths
    there is NO radius on the ladder and no hover between 0.10 and 0.25 m at
    which either middle-row arm can stand facing the middle.

    That is the same finding `certified_park_poses` exists for, arrived at
    twice: a pose aimed at the middle of a canvas with six arms over it is a
    pose aimed at somebody.  The outward bearing is not a preference any more,
    it is the only thing that certifies.
    """
    assert frames.ACTIVE_TOOL == "inline", "the global tool must be untouched"
    fl, _ = fleet.rig("proposed")
    for aid in (31, 71):
        with pytest.raises(RuntimeError, match="no certified ready pose"):
            layout.certified_ready_pose(fl[aid], pen_lat=LAT, pen_ext=EXT)
    for aid in (13, 17, 2, 97):                 # the outer rows still can
        layout.certified_ready_pose(fl[aid], pen_lat=LAT, pen_ext=EXT)
    assert frames.ACTIVE_TOOL == "inline" and frames.PEN_LAT == 0.0


def test_every_proposed_park_pose_clears_every_other_mount():
    """The v1 study could not make this check — there was no hardware to
    check against.  The pose is the one the scene draws.

    It is the PARK pose now, not the inward ready pose: since the capsules
    were corrected the inward one does not exist for the middle row at all
    (see the test above), and the park poses are what every arm actually
    holds — `spec.q_seed` is `Q_PARK_PROPOSED`.
    """
    # NOTE: no `activate_tool` here.  `certified_ready_pose` takes the tool
    # explicitly, so this test cannot leak `frames.PEN_LAT` into every module
    # that runs after it — which is exactly what it used to do.
    assert frames.ACTIVE_TOOL == "inline", "the global tool must be untouched"
    fl, _ = fleet.rig("proposed")
    h = layout.LAYOUT_PROPOSED["h"]
    for aid, spec in sorted(fl.items()):
        q = np.asarray(layout.Q_PARK_PROPOSED[aid], float)
        rep = validate.check_pose(q, spec, pen_ext=EXT, pen_lat=LAT)
        assert rep["ok"] and rep["worst"]["tip_z"] > 0.0
        T, pts = frames.fk(q)
        tool = frames.tool_points_many(T[None], pen_lat=frames.PEN_LAT_HOLDER)
        P = np.vstack([pts] + list(tool))
        Twb = spec.T_world_base(h)
        Pw = (Twb[:3, :3] @ P.T).T + Twb[:3, 3]
        cl = rig_final.chain_static_clearance(Pw, spec.static_obstacles())[0]
        assert cl >= rig_final.STATIC_MARGIN - 1e-9, (aid, float(cl))
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
    assert h == 0.940
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
# BOTH HALVES OF THE HOLDER'S TOOL.  Passing only `pen_lat` used to be
# enough because the axial half was the inline pen's 0.110 too; since
# 2026-09-03 it is `PEN_EXT_HOLDER` (frames.py), and a bare `pen_lat=LAT`
# asks about a tool that exists nowhere.
EXT = frames.PEN_EXT_HOLDER


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
    """The legacy inverted seed is not a park pose, and this pins BOTH halves
    of why — the half that is about the pose and the half that was only ever
    about the height.

    At h = 0.850 it hung 61.6 mm UNDER the paper.  Raising the rig to 0.940
    lifts the same joints 90 mm and the tip now clears by 70 mm, so that
    particular refusal is gone; what is left is the reason it was never a
    depot anyway, which no height fixes: 0.184 of joint margin against the
    0.30 gate.  The tip clearance is asserted as a height-derived quantity
    rather than a constant, so this test says something true at any h.

    THE DEPTH BELOW THE BASE MOVED ON 2026-09-03, from 0.9116 m to 0.8698,
    because the holder's tool did: `PEN_EXT_HOLDER` is 0.0588421 m where the
    pair used to be the inline pen's 0.110 (frames.py), so the same joints
    hold the tip 41.8 mm higher.  The pose is no better a depot for it.
    """
    bad = frames.Q_READY_INV
    spec = layout.FLEET_PROPOSED[31]
    h = float(layout.LAYOUT_PROPOSED["h"])
    rep = validate.check_pose(bad, spec, pen_ext=EXT, pen_lat=LAT)
    # DERIVED, NOT A LITERAL.  The depth below the base is a property of the
    # pose and the ACTIVE tool, and it has moved every time the tool has:
    # 0.9116 m at the inline pen, 0.8698 at the 0.0588421 pair, 0.8532 at the
    # 2026-09-07 pair.  A baked number here just goes red on the next holder
    # revision and says nothing; this says the thing the test is actually
    # about — where that pose puts the tip, at whatever tool is fitted.
    T = frames.fk(np.asarray(bad, float))[0]
    depth = float((T[:3, 3] + T[:3, :3] @ frames.tool_offset(EXT, LAT))[2])
    assert rep["worst"]["tip_z"] == pytest.approx(h - depth, abs=2e-3)
    assert 0.80 < depth < 0.95, "the seed still hangs most of a metre down"
    assert frames.joint_margin(bad) < 0.30, "and THAT is why it is not a depot"

    for aid, spec in sorted(layout.FLEET_PROPOSED.items()):
        q = spec.q_seed                            # what the pipeline reads
        assert np.array_equal(q, np.asarray(layout.Q_PARK_PROPOSED[aid], float))
        rep = validate.check_pose(q, spec, pen_ext=EXT, pen_lat=LAT)
        assert rep["ok"], (aid, rep)
        assert rep["worst"]["tip_z"] >= 0.09, aid          # hovering, not down
        assert frames.joint_margin(q) >= 0.30, aid
        # STEEL is far away (>= 0.466 m); the nearest obstacle a parked arm
        # has is now a NEIGHBOUR — arm 17 holds 0.097 m to arm 13's base
        # column boxes, which is 0.169 m to the arm inside them, against the
        # 0.080 m the conductor asks of every moving pair.  Both numbers
        # shrank when the mesh audit widened the column from 0.12 to 0.185
        # and the mover's own capsules with it.
        assert rep["worst"]["min_frame_clearance"] >= 0.09, aid
        steel = [b for b in spec.static_obstacles()
                 if b["tag"].startswith("mount")]
        T, pts = frames.fk(np.asarray(q, float))
        P = np.vstack([pts] + list(frames.tool_points_many(T[None],
                                                           pen_lat=LAT)))
        Twb = spec.T_world_base()
        Pw = (Twb[:3, :3] @ P.T).T + Twb[:3, 3]
        assert rig_final.chain_static_clearance(Pw, steel)[0] >= 0.46, aid


def coordination_PAIR():
    from aris_sixarm.coordination import PAIR_MARGIN
    return PAIR_MARGIN


def test_the_parked_fleet_does_not_park_inside_itself(lateral):
    """WHY THE BEARING IS OUTWARD.  Aimed at the canvas centre — which is what
    one arm alone wants — the middle row cannot certify a pose at all and the
    four that can stand too close.  Away from the fleet centroid they stand
    off towards their own rims and hold 195 mm, against the 80 mm the
    conductor asks of every pair while they MOVE.  (181 mm before the mesh
    audit; the number moved because BOTH the poses and the capsules did.)"""
    fl = layout.FLEET_PROPOSED
    from aris_sixarm.coordination import SAFETY_M, CALIB_M
    assert _park_clearance(layout.Q_PARK_PROPOSED, fl) >= SAFETY_M + CALIB_M

    # THE INWARD CONTROL IS NOW A REFUSAL.  Before the capsules were
    # corrected, aiming every arm at the canvas centre certified six poses
    # that then overlapped by 95.6 mm.  At the true widths the middle row
    # cannot certify an inward pose AT ALL, so the control is a refusal
    # rather than an interpenetration — and THAT half is the substantive one.
    # The four arms that can still certify inward used to stand 4 mm under
    # the conductor's 80 mm; with the holder's own shorter tool
    # (frames.PEN_EXT_HOLDER, 2026-09-03) they land 0.4 mm OVER it, which is
    # a near miss and not a clearance, so the bar is stated as such.
    with pytest.raises(RuntimeError, match="no certified ready pose"):
        {aid: layout.certified_ready_pose(s, pen_lat=LAT, pen_ext=EXT)[0]
         for aid, s in sorted(fl.items())}
    outer = [13, 17, 2, 97]
    inward = {aid: layout.certified_ready_pose(fl[aid], pen_lat=LAT, pen_ext=EXT)[0]
              for aid in outer}
    # THE ABSOLUTE NUMBER, because the gate it is compared against moved.
    # The inward four stand 61.3 mm apart.  Against the OLD 80 mm pair margin
    # that was a refusal, which is what this control was written to show;
    # against Pete's 2026-09-09 50 mm gate the same poses now CLEAR, by
    # 11.3 mm.  The geometry did not move and the test says so in millimetres
    # rather than in a constant that has.
    inward_clear = _park_clearance(inward, {a: fl[a] for a in outer})
    assert inward_clear == pytest.approx(0.0613, abs=5e-4)
    assert inward_clear < 0.080                    # refused by the old gate
    assert inward_clear >= coordination_PAIR()     # cleared by the new one

    # ...and the function still refuses a fleet that parks inside itself.
    # THE GATE IS WHAT THIS PINS, AND ONLY THE GATE.  Which INPUT trips it has
    # changed with every tool: the 0.20 m hover was a 4 mm near miss, then the
    # corrected capsules cleared it, then the 0.0588421 pair made it a refusal
    # again (arms 2 and 97 overlapping by 77.8 mm), and at the 2026-09-07 pair
    # NO hover from 0.05 to 0.50 refuses — a tip 86 mm off the hand's axis
    # parks the outward poses further apart than a 59 mm one did.  So the trip
    # is asked for directly, by demanding more clearance than the layout can
    # give, which is a property of the function rather than of this month's
    # holder.
    bare = layout.build_fleet(layout.LAYOUT_PROPOSED)
    layout.certified_park_poses(bare, hover=0.20, pen_lat=LAT, pen_ext=EXT)
    for clear in (0.30, 0.60):
        with pytest.raises(RuntimeError, match="park .* mm apart"):
            layout.certified_park_poses(bare, hover=0.20, pen_lat=LAT,
                                        pen_ext=EXT, clear=clear)


def test_no_shipped_park_pose_stands_in_another_arms_certified_ink():
    """THE NUMBER THAT DECIDES WHETHER A PHASE CAN BE CONDUCTED.

    A park pose is held for the whole of every phase its arm is not drawing
    in, so a parked arm is an obstacle whose schedule is a CONSTANT: a mover
    whose certified ink runs through it has no monotone schedule and no
    ordering fixes it (`coordination.hard_blocks`).  This walks every
    certified pose of every arm in the shipped atlas against every OTHER
    arm's parked chain, with the conductor's own capsules and margin.

    Skipped without an atlas: `out/` is gitignored, so this is a check the
    rig's own re-certification runs, not something a clean checkout can do.

    AND THE ATLAS HAS TO BE THE ONE THIS TOOL SWEPT (2026-09-03).  The
    selection used to ask `atlas.is_current`, which is answered against the
    PROCESS-GLOBAL tool — and a bare test run is the INLINE pen, so every
    lateral atlas in `out/` read as stale and this test SKIPPED for as long as
    the directory has existed, while its body measured with the holder's own
    `EXT`/`LAT`.  It now asks for the tool the atlas records and answers
    `is_current` with that tool active, so the check either runs against an
    atlas swept at the geometry it is measuring or says why it did not.
    """
    from pathlib import Path
    from aris_sixarm import atlas as atlas_mod, coordination
    fl = layout.FLEET_PROPOSED
    h = float(layout.LAYOUT_PROPOSED["h"])
    out = Path(__file__).resolve().parents[1] / "out"
    d = None
    was = frames.ACTIVE_TOOL
    frames.activate_tool("lateral")
    try:
        for cand in sorted(out.glob("atlas_proposed*")):
            f = cand / "atlas_arm31.npz"
            if not f.is_file():
                continue
            meta = np.load(f)
            if abs(float(meta["pen_ext"]) - EXT) > 1e-9 \
                    or abs(float(meta["pen_lat"]) - LAT) > 1e-9:
                continue
            if atlas_mod.is_current(meta)[0] \
                    and abs(float(meta["base"][2, 3]) - h) <= 1e-9:
                d = cand
                break
    finally:
        frames.activate_tool(was)
    if d is None:
        pytest.skip(f"no current proposed atlas at h = {h:.3f} and the "
                    f"holder's tool ({EXT}, {LAT}) in out/ — run "
                    "scripts/run_atlas6.py --rig proposed --pen "
                    f"{EXT} --pen-lat {LAT}")
    margin = coordination.SAFETY_M + coordination.CALIB_M

    def chain(Q, spec):
        """World chain with the LATERAL tool, whatever the process global is
        (`frames.PEN_LAT` is not set in a bare test run)."""
        T, P = frames.fk_many(np.asarray(Q, float).reshape(-1, 7))
        tl = frames.tool_points_many(T, EXT, LAT)
        P = np.concatenate([P] + [t[:, None, :] for t in tl], axis=1)
        Twb = np.asarray(spec.T_world_base(), float)
        return P @ Twb[:3, :3].T + Twb[:3, 3]

    parks = {a: chain(np.asarray(layout.Q_PARK_PROPOSED[a], float)[None, :],
                      fl[a]) for a in fl}
    tab = coordination.CAPSULES_LAT
    rr = np.array([c[2] for c in tab], float)
    worst, who, n = np.inf, None, 0
    for aid, spec in sorted(fl.items()):
        arr, _ = atlas_mod.load(d, aid)
        Q = arr[atlas_mod.strict_go(arr)][:,
                                          atlas_mod.QCOL:atlas_mod.QCOL + 7]
        if not len(Q):
            continue
        P = chain(Q, spec)
        A, B = coordination.cap_endpoints(P, tab)
        for other in sorted(fl):
            if other == aid:
                continue
            Ab, Bb = coordination.cap_endpoints(parks[other], tab)
            for s in range(0, len(A), 512):
                d2 = coordination.seg_seg_dist(
                    A[s:s + 512, :, None, :], B[s:s + 512, :, None, :],
                    Ab[:, None, :, :], Bb[:, None, :, :])
                g = float((d2 - rr[None, :, None] - rr[None, None, :]).min())
                if g < worst:
                    worst, who = g, (aid, other)
            n += len(A)
    assert n > 1000, "the atlas must actually carry certified poses"
    assert worst >= margin, (
        f"arm {who[1]}'s park pose is {1000 * worst:.1f} mm from arm "
        f"{who[0]}'s certified ink, under the {1000 * margin:.0f} mm the "
        "conductor holds every pair to — the fleet cannot conduct a phase "
        "in which it is parked")


def test_baked_park_poses_are_that_functions_own_output():
    """The literals in `layout.py` are `certified_park_poses`' output on
    `LAYOUT_PROPOSED` at `PARK_GRID_PROPOSED`, so they cannot drift from the
    recipe that made them.  Derived from the BARE fleet: the park pose is also
    the IK seed, and a fleet already carrying one would be seeded by its own
    answer.

    WHAT THIS DOES NOT RE-RUN is the (radius, hover, bearing) SEARCH behind
    the grid — 6 radii x 4 hovers x 24 bearings, each gated, then ranked on
    park-vs-ink and on 24 cells x two directions of `paper.route` per arm — a
    quarter of an hour that needs an atlas out of gitignored `out/`.  The
    grid's scores are recorded where it is defined; what is pinned here is
    that the six poses are what that grid produces.

    AND IT IS RE-RUN AT THE HOLDER'S OWN TOOL AGAIN, which is where it
    started.  `PEN_LAT_HOLDER` / `PEN_EXT_HOLDER` moved from 0.110 / 0.110 to
    0.0588421 / 0.0588421 on 2026-09-03 (frames.py, docs/SYSTEM_MODEL.md 7e)
    and the poses were STALE for one afternoon: run at the new pair the OLD
    grid parked arms 13 and 17 53.7 mm apart and `certified_park_poses`
    refused the fleet.  The grid was re-searched at the new pair the same day
    and both the grid and the six literals it makes moved (see `layout`'s own
    note), so this re-derives at `PEN_*_HOLDER` — the module constants — and
    no longer at a frozen 0.110.
    """
    bare = layout.build_fleet(layout.LAYOUT_PROPOSED)
    assert all(np.array_equal(s.q_seed, frames.Q_READY_INV)
               for s in bare.values()), "derive from the LEGACY seed"
    made = layout.certified_park_poses(bare, layout.PARK_GRID_PROPOSED,
                                       pen_lat=LAT, pen_ext=EXT)
    assert sorted(made) == sorted(layout.Q_PARK_PROPOSED)
    for aid, q in made.items():
        assert np.allclose(q, layout.Q_PARK_PROPOSED[aid], atol=5e-5), aid

    # each arm stands off along its own SEARCHED bearing at its own radius,
    # and holds the pen at its own hover — the three numbers the grid records.
    # The bearing is a grid entry now, not the outward ray: held outward the
    # middle row's ray runs off the short edge of the canvas and the park set
    # saturates 5 mm under the conductor's margin (see `layout`'s note).
    cent = np.mean([s.xy for s in bare.values()], axis=0)
    for aid, q in layout.Q_PARK_PROPOSED.items():
        r, hv, bdeg = layout.PARK_GRID_PROPOSED[aid]
        T = frames.fk(np.asarray(q, float))[0]
        # the HOLDER's own pair, which is what the 2026-09-03 re-search used
        # and what `PARK_HOVER_PROPOSED` was recorded with
        tip = T[:3, 3] + T[:3, :3] @ frames.tool_offset(EXT, LAT)
        Twb = layout.FLEET_PROPOSED[aid].T_world_base()
        w = Twb[:3, :3] @ tip + Twb[:3, 3]
        assert np.allclose(w[:2], layout.PARK_HOVER_PROPOSED[aid], atol=1e-3)
        assert abs(w[2] - hv) < 1e-4, aid      # the literals are 4 decimals
        b = np.asarray(bare[aid].xy, float)
        u = np.array([np.cos(np.deg2rad(bdeg)), np.sin(np.deg2rad(bdeg))])
        # on the bearing, at the radius (or clipped to the sheet edge, which
        # is why this is a bound and not an equality)
        assert float(np.dot(w[:2] - b, u)) > 0.5 * r, aid
        assert float(np.linalg.norm(w[:2] - b)) <= r + 1e-4, aid
        # ...and NOT INWARD.  Outwardness was the 2026-08-26 rule and it was
        # always a means: what keeps the six apart is `fleet_park_clearance`,
        # asserted directly above.  So the BAR is "not inward" and that is
        # what is pinned; how many of the six are strictly outward is an
        # instance, and it has already changed twice (five of six at the
        # 2026-09-03 tool, with arm 2's tangential at dot -0.015; six of six
        # at the 2026-09-07 tool).  Pinning the count made this test fail for
        # getting BETTER, which is not a property worth guarding.
        out = (b - cent) / np.linalg.norm(b - cent)
        assert float(np.dot(u, out)) > -0.05, aid
    assert sum(float(np.dot(
        np.array([np.cos(np.deg2rad(layout.PARK_GRID_PROPOSED[a][2])),
                  np.sin(np.deg2rad(layout.PARK_GRID_PROPOSED[a][2]))]),
        (np.asarray(bare[a].xy, float) - cent)
        / np.linalg.norm(np.asarray(bare[a].xy, float) - cent))) > 0.4
        for a in layout.Q_PARK_PROPOSED) >= 5, \
        "at least five of the six stand off along their own outward ray"
    # A TWO-NUMBER GRID ENTRY IS STILL THE OLD OUTWARD RECIPE, AND IT IS A
    # DIFFERENT ANSWER — which is the property worth pinning, because it is
    # the one that has held at every tool.  WHAT THE RECIPE COSTS HAS NOT:
    # at the 0.110 pen it merely differed; at 0.0588421 it REFUSED outright
    # (arms 2 and 97 came out 111.0 mm inside each other); at the 2026-09-07
    # pair it certifies again, because a tip 86 mm off the hand's axis parks
    # the outward poses further apart than a 59 mm one did.  So the assertion
    # is "the bearing changes the set", not "the bearing rescues the set" —
    # the latter was true of one tool and this test outlived it twice.
    outward_only = layout.certified_park_poses(
        bare, {a: v[:2] for a, v in layout.PARK_GRID_PROPOSED.items()},
        pen_lat=LAT, pen_ext=EXT)
    assert sorted(outward_only) == sorted(layout.Q_PARK_PROPOSED)
    assert any(not np.allclose(outward_only[a], layout.Q_PARK_PROPOSED[a],
                               atol=1e-6) for a in outward_only), \
        "the third number has to change something or it is not a variable"


# ===========================================================================
# REGION-AWARE PARKING
# ===========================================================================
def _an_arm_that_gets_stood_over(fl, base):
    """An arm the feature itself reports as moved. -> (aid, target, drawing).

    DERIVED, NOT NAMED, and that is a lesson from 2026-09-07.  These tests used
    to name arm 71 because at the 0.0588421 tool its park held the pen 0.30 m
    out on a -60 degree bearing and its own chain sat 85 mm off its column —
    the tightest in the fleet.  The park re-search at the 2026-09-07 tool moved
    it to 208 mm, so arm 71 became the ONE arm of six that no longer has to
    swing aside, and four tests went red for a rig that had got better.  What
    they are about is the FEATURE, so they ask the feature which arm to use.
    """
    for aid in sorted(fl):
        drawing = next(x for x in sorted(fl) if x != aid)
        target = tuple(float(v) for v in fl[aid].xy)
        _parks, info = layout.region_aware_parks(fl, base, target,
                                                 drawing=drawing)
        if info.get(aid, {}).get("moved"):
            return aid, target, drawing
    raise AssertionError("no arm in the shipped set is stood over by a target "
                         "on its own base; region-aware parking is untestable")


def test_region_aware_parking_is_the_shipped_set_when_nothing_fires(lateral):
    """A TARGET NOBODY IS STANDING OVER GETS THE BAKED LITERALS, unchanged.

    The whole guarantee of this feature is that it is inert where it does not
    apply: a map or a phase that never draws under anybody has to be
    bit-identical to one built before it existed, and "bit-identical" means the
    same numbers, not a re-derivation that agrees to five decimals.
    """
    fl = layout.FLEET_PROPOSED
    base = layout.Q_PARK_PROPOSED
    # a corner of the canvas, further than ASIDE_DISC_R from every base
    far = (0.03, 1.80)
    assert min(float(np.hypot(*(np.asarray(s.xy, float) - np.array(far))))
               for s in fl.values()) > layout.ASIDE_DISC_R
    parks, info = layout.region_aware_parks(fl, base, far, drawing=31)
    assert info == {}
    assert sorted(parks) == sorted(base)
    for a in base:
        assert parks[a] is base[a], f"arm {a} was re-derived, not reused"


def test_region_aware_parking_moves_the_arm_that_is_stood_over(lateral):
    """A cell under arm 71's boom, drawn by somebody else.

    Arm 71's shipped park holds its pen 0.30 m out on a -60 degree bearing,
    which puts its own chain 85 mm from the column over its own base — the
    tightest in the fleet, and 5 mm inside what the conductor asks of a pair
    that MOVES.  Swung aside it has to do better, and the set it lands in has
    to survive the same fleet-pairwise proof the shipped literals did.
    """
    from aris_sixarm.coordination import SAFETY_M, CALIB_M
    fl = layout.FLEET_PROPOSED
    base = layout.Q_PARK_PROPOSED
    aid, target, drawing = _an_arm_that_gets_stood_over(fl, base)
    was = layout.corridor_clearance(base[aid], fl[aid], target)

    parks, info = layout.region_aware_parks(fl, base, target, drawing=drawing)
    assert aid in info, "the arm being drawn under must be considered"
    assert info[aid]["moved"], info
    assert info[aid]["clear"] > was, "an aside park that is not further is not a move"
    assert not np.allclose(parks[aid], base[aid], atol=1e-6)
    # ...and every arm nobody is standing over keeps its literal, by identity
    for a in base:
        if a == aid or info.get(a, {}).get("moved"):
            continue
        assert parks[a] is base[a], a
    # the FLEET is proved, not the pose: the same check that made the literals
    worst, _pair = layout.fleet_park_clearance(parks, fl)
    assert worst >= SAFETY_M + CALIB_M
    # and the pose itself is gated exactly as a shipped park is
    rep = validate.check_pose(parks[aid], fl[aid], pen_ext=EXT, pen_lat=LAT)
    assert rep["ok"], rep
    assert rep["worst"]["tip_z"] > 0.0


def test_the_drawing_arm_is_never_asked_to_move(lateral):
    """`drawing` is not parked, so its own park is not a variable."""
    fl = layout.FLEET_PROPOSED
    base = layout.Q_PARK_PROPOSED
    target = tuple(float(v) for v in fl[31].xy)
    parks, info = layout.region_aware_parks(fl, base, target, drawing=31)
    assert 31 not in info
    assert parks[31] is base[31]


def test_an_aside_park_is_one_the_arm_can_fly_to(lateral):
    """A DEPOT AN ARM CANNOT FLY TO IS NOT A DEPOT — the lesson
    `PARK_GRID_PROPOSED` already carries, applied to the pose that replaces
    it.  One certified `paper.route` from the park it leaves to the one it
    takes, at the flying floor."""
    fl = layout.FLEET_PROPOSED
    base = layout.Q_PARK_PROPOSED
    aid, target, drawing = _an_arm_that_gets_stood_over(fl, base)
    parks, info = layout.region_aware_parks(fl, base, target, drawing=drawing)
    assert info[aid]["moved"]
    r = layout.repark_route(fl[aid], base[aid], parks[aid],
                            h_inv=float(layout.LAYOUT_PROPOSED["h"]))
    assert r is not None, f"arm {aid} cannot fly from its park to its aside park"


def test_the_aside_ranking_is_a_total_order_and_replays(lateral):
    """Two runs of the same question give the same poses in the same order.

    The proxy ORDERS and the caller's real check DECIDES, so the order has to
    be reproducible or the caller's answer is not.  Ties are broken by the
    order the candidates were generated in, which is fixed by the three tuples.
    """
    fl = layout.FLEET_PROPOSED
    cs = layout.aside_candidates(fl[2], pen_lat=LAT,
                                 extra=(layout.PARK_GRID_PROPOSED[2],))
    assert len(cs) > 10
    assert tuple(cs[0][1]) == tuple(layout.PARK_GRID_PROPOSED[2]), \
        "the incumbent is first in the candidate list"
    t = (0.60, 3.00)
    one = layout.aside_park_ranking(fl[2], t, cs)
    two = layout.aside_park_ranking(fl[2], t, cs)
    assert [r[1] for r in one] == [r[1] for r in two]
    assert all(one[i][0] >= one[i + 1][0] - 1e-12 for i in range(len(one) - 1))
    # rank walks that same list rather than re-scoring it
    seen = set()
    for k in (1, 2, 3):
        p, i = layout.region_aware_parks(fl, layout.Q_PARK_PROPOSED, t,
                                         drawing=13, rank=k)
        if i and any(v["moved"] for v in i.values()):
            key = tuple(np.round(p[a], 6).tobytes() for a in sorted(p))
            assert key not in seen, f"rank {k} repeated an earlier set"
            seen.add(key)


def test_fleet_park_clearance_is_the_check_that_made_the_literals(lateral):
    """One implementation, two callers: `certified_park_poses` and
    `region_aware_parks` are held to the same number by the same function."""
    from aris_sixarm.coordination import SAFETY_M, CALIB_M
    fl = layout.FLEET_PROPOSED
    worst, pair = layout.fleet_park_clearance(layout.Q_PARK_PROPOSED, fl)
    assert worst >= SAFETY_M + CALIB_M
    assert worst == pytest.approx(_park_clearance(layout.Q_PARK_PROPOSED, fl))
    assert pair[0] in fl and pair[1] in fl and pair[0] != pair[1]


def test_a_parked_arm_flies_to_its_aside_park_rather_than_appearing_there(lateral):
    """THE MOVE IS THE POINT.  A per-phase park that the arm teleports into is
    a pose nothing checked the way in to; `writing.arm_program` lays a
    certified `paper.route` and the waypoints go on the timeline, so the
    conductor schedules the motion and `scene_check` plays it back.

    And it is priced where the idle policy's other additions are priced:
    `transit_s` is the number `csail_schedule.cross_check` holds to the
    sequencer's own, to the float, so the repark must not be in it.
    """
    from aris_sixarm import writing
    fl = layout.FLEET_PROPOSED
    h = float(layout.LAYOUT_PROPOSED["h"])
    aid, target, drawing = _an_arm_that_gets_stood_over(
        fl, layout.Q_PARK_PROPOSED)
    spec, base = fl[aid], np.asarray(layout.Q_PARK_PROPOSED[aid], float)

    still = writing.arm_program(spec, [], h_inv=h, pen_ext=spec.pen,
                                q_start=base)
    assert still["duration"] == 0.0 and still["aside_s"] == 0.0
    assert np.array_equal(still["q_end"], base)

    parks, info = layout.region_aware_parks(fl, layout.Q_PARK_PROPOSED,
                                            target, drawing=drawing)
    assert info[aid]["moved"]
    moved = writing.arm_program(spec, [], h_inv=h, pen_ext=spec.pen,
                                q_start=base, aside=parks[aid])
    assert moved["aside_s"] > 0.0
    assert moved["transit_s"] == 0.0, "a repark is not a re-pricing of the tour"
    assert moved["draw_s"] == 0.0 and moved["draw_len"] == 0.0
    assert np.allclose(moved["q_end"], parks[aid])
    assert len(moved["q"]) >= 2 and moved["duration"] == pytest.approx(
        moved["t"][-1])
    assert [p["kind"] for p in moved["phases"]] == ["aside"]
    # every waypoint is a pose the arm may stand in, and the LAST one is where
    # the phase's collision images will find it
    for q in moved["q"]:
        assert frames.joint_margin(q) >= 0.0
    assert validate.check_pose(moved["q"][-1], spec, pen_ext=EXT, pen_lat=LAT)["ok"]

    # an aside pose the arm cannot fly to is REFUSED, not flown
    with pytest.raises(writing.PaperRefused):
        writing.arm_program(spec, [], h_inv=h, pen_ext=spec.pen, q_start=base,
                            aside=np.zeros(7))


def test_conduct_reports_the_parks_it_actually_flew(lateral):
    """`idle.conduct` is where a phase's geometry is decided, so the aside
    parks have to reach it and come back out — a caller that asked for one and
    got a frozen arm must be able to tell."""
    from aris_sixarm import idle, writing
    fl = layout.FLEET_PROPOSED
    h = float(layout.LAYOUT_PROPOSED["h"])
    pens = {a: fl[a].pen for a in fl}
    segs = {a: [] for a in fl}
    q0 = {a: np.asarray(layout.Q_PARK_PROPOSED[a], float) for a in fl}

    off = idle.conduct(segs, pens, 0.05, q_start=q0, specs=fl, h_inv=h,
                       jit=False, retreat=False, verbose=False)
    assert off["aside"] == {}
    assert all(np.allclose(off["q_end"][a], q0[a]) for a in fl)

    # WHICHEVER ASIDE THE CONDUCTOR CAN ACTUALLY FLY, and at the 2026-09-07
    # park set that is not all of them: every moved arm has a certified
    # `repark_route`, but with the other five standing still arms 2 and 17
    # have no monotone pause schedule for theirs and `idle.conduct` raises
    # `Unconductable`.  That is a real property of the aside feature on this
    # rig and it is recorded in docs/DECISIONS.md rather than papered over;
    # what THIS test is about is that an aside reaches the conductor and comes
    # back out, which any conductable one demonstrates.
    on = aid = parks = None
    for cand in sorted(fl):
        drawing = next(x for x in sorted(fl) if x != cand)
        target = tuple(float(v) for v in fl[cand].xy)
        cparks, cinfo = layout.region_aware_parks(fl, layout.Q_PARK_PROPOSED,
                                                  target, drawing=drawing)
        if not cinfo.get(cand, {}).get("moved"):
            continue
        try:
            on = idle.conduct(segs, pens, 0.05, q_start=q0, specs=fl, h_inv=h,
                              jit=False, retreat=False, verbose=False,
                              aside={cand: cparks[cand]})
        except idle.Unconductable:
            continue
        aid, parks = cand, cparks
        break
    assert on is not None, "no arm's aside park can be conducted at all"
    assert sorted(on["aside"]) == [aid] and on["aside"][aid] > 0.0
    assert np.allclose(on["q_end"][aid], parks[aid])
    for a in fl:
        if a != aid:
            assert np.allclose(on["q_end"][a], q0[a])
    assert "ASIDE" in "\n".join(idle.report(on))
