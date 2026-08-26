"""MOUNT HARDWARE is a first-class obstacle (layout study v2).

v1 of the layout study modelled no steel at all, so its coverage numbers were
a kinematic ceiling.  These tests pin the machinery that closed that gap:

  * the schematic boxes have the documented dimensions and z bands;
  * an arm sees every OTHER arm's hardware and NOT its own;
  * a mount box standing in a reach annulus really does remove cells from the
    REAL atlas (not just from the coarse proxy), and removing the box brings
    them back;
  * the coarse proxy's wedge subtraction removes what is behind a column and
    keeps what is beside it;
  * the z-band gate that decides shadow-vs-keep-out is the physical one.

...and, since 2026-08-25, the same again for the OTHER ARMS THEMSELVES: the
neighbour base column (`mounts.arm_column_box`), whose radius is derived from
the conductor's own capsule table, which removes from the real atlas the cells
a 0.61 m transverse partner is standing on, and which `scene_check` re-derives
from the base transform without reading a line of this module.
"""
import numpy as np
import pytest

from aris_sixarm import atlas, layout, mounts, rig_final
from aris_sixarm.frames import PEN_LAT_HOLDER

H = 0.850
LAT = PEN_LAT_HOLDER


# ---------------------------------------------------------------------------
# the schematic hardware
# ---------------------------------------------------------------------------
def test_inverted_hardware_is_a_plate_then_a_boom_to_the_ceiling():
    m = mounts.MOUNTS
    plate, boom = mounts.arm_mount_boxes("inv", (0.9, 1.4), 0.0, H)
    assert plate["name"].endswith("_plate") and boom["name"].endswith("_boom")
    # the plate sits ON TOP of the base flange, the boom on top of the plate
    assert plate["lo"][2] == pytest.approx(H)
    assert plate["hi"][2] == pytest.approx(H + m.plate_t)
    assert boom["lo"][2] == pytest.approx(H + m.plate_t)
    assert boom["hi"][2] == pytest.approx(m.ceiling_z)
    # footprints: the plate as drawn, the boom as the CIRCUMSCRIBED square of
    # the r = 0.10 cylinder (conservative — the box machinery takes boxes)
    assert (plate["hi"][:2] - plate["lo"][:2]) == pytest.approx(m.plate_xy)
    assert (boom["hi"][:2] - boom["lo"][:2]) == pytest.approx(
        [2 * m.boom_r, 2 * m.boom_r])


def test_floor_hardware_is_a_pedestal_under_the_paper_plane():
    m = mounts.MOUNTS
    (ped,) = mounts.arm_mount_boxes("floor", (0.6, -0.15), 0.0, H)
    assert ped["hi"][2] == pytest.approx(m.z_floor)
    assert ped["lo"][2] == pytest.approx(m.z_floor - m.ped_drop)
    assert (ped["hi"][:2] - ped["lo"][:2]) == pytest.approx(m.ped_xy)


def test_a_yawed_plate_keeps_a_conservative_bounding_box():
    """A yawed footprint is carried as its AABB, which can only grow."""
    m = mounts.MOUNTS
    flat = mounts.arm_mount_boxes("inv", (0.0, 0.0), 0.0, H)[0]
    yawed = mounts.arm_mount_boxes("inv", (0.0, 0.0), np.pi / 4, H)[0]
    assert np.all(yawed["hi"][:2] >= flat["hi"][:2] - 1e-12)
    assert np.all(yawed["lo"][:2] <= flat["lo"][:2] + 1e-12)
    assert (yawed["hi"][0] - yawed["lo"][0]) == pytest.approx(
        (m.plate_xy[0] + m.plate_xy[1]) / np.sqrt(2))


def test_mount_dimensions_are_parameterised():
    big = mounts.MOUNTS.scaled(boom_r=0.20, ceiling_z=3.0)
    _, boom = mounts.arm_mount_boxes("inv", (0.5, 0.5), 0.0, H, m=big)
    assert (boom["hi"][:2] - boom["lo"][:2]) == pytest.approx([0.4, 0.4])
    assert boom["hi"][2] == pytest.approx(3.0)
    assert mounts.min_column_spacing(big) > mounts.min_column_spacing()


# ---------------------------------------------------------------------------
# own-mount exclusion
# ---------------------------------------------------------------------------
def test_every_arm_sees_all_other_mounts_and_never_its_own():
    fl = layout.build_fleet(layout.LAYOUT_V1)
    assert len(fl) == 6
    for aid, spec in fl.items():
        tags = {b["tag"] for b in spec.static_obstacles()}
        assert f"mount:{aid}" not in tags, "an arm is bolted to its own mount"
        assert f"body:{aid}" not in tags, "an arm is not its own obstacle"
        assert tags == {f"mount:{o}" for o in fl if o != aid} \
            | {f"body:{o}" for o in fl if o != aid}
    # ... and the count matches the hardware PLUS one body column each: 2
    # boxes per inverted arm, 1 per floor arm, 1 column per arm, minus own
    inv_ids = [a for a, s in fl.items() if s.mount == "inv"]
    assert len(fl[inv_ids[0]].static_obstacles()) == \
        2 * (len(inv_ids) - 1) + 2 + (len(fl) - 1)


def test_obstacles_for_matches_the_fleet_minus_the_owner():
    fl = layout.build_fleet(layout.LAYOUT_V1)
    bare = layout.build_fleet(layout.LAYOUT_V1, with_mounts=False)
    for aid in fl:
        want = mounts.obstacles_for(aid, bare, H)
        got = fl[aid].static_obstacles()
        assert [b["name"] for b in got] == [b["name"] for b in want]


def test_green_field_specs_still_available_for_the_v1_comparison():
    """v1's numbers must stay reproducible: `with_mounts=False` is v1."""
    fl = layout.build_fleet(layout.LAYOUT_V1, with_mounts=False)
    assert all(s.static_obstacles() == [] for s in fl.values())


# ---------------------------------------------------------------------------
# the REAL atlas: a mount box inside a reach annulus removes cells
# ---------------------------------------------------------------------------
def _solve(spec, x, y, boxes):
    Twb = spec.T_world_base(H)
    return atlas.solve_cell(x, y, Twb, np.linalg.inv(Twb), spec,
                            atlas._candidates(0.0), boxes=boxes, pen_lat=LAT)


def test_a_boom_inside_the_annulus_removes_the_cell_from_the_real_atlas():
    """The plumbing that matters: `atlas.solve_cell` consumes mount boxes
    through the SAME lattice clearance check the final rig uses."""
    spec = layout.study_spec(31, "inv", (0.9, 1.20), h=H)
    x, y = 0.9, 1.20 + 0.55           # a cell well inside the GO annulus
    assert _solve(spec, x, y, ()) is not None, "green field must reach it"
    # a neighbour's hardware standing ON that cell, brought down to the pen's
    # level so the z bands really overlap (a boom at z >= h cannot reach it —
    # that is the physical finding this study reports, not an oversight)
    blocker = dict(name="blocker", tag="mount:99", source="test",
                   lo=np.array([x - 0.12, y - 0.12, -0.10]),
                   hi=np.array([x + 0.12, y + 0.12, 0.02]))
    assert _solve(spec, x, y, [blocker]) is None
    # and a cell on the far side of the base is untouched
    assert _solve(spec, x, 1.20 - 0.55, [blocker]) is not None


def test_a_neighbour_pedestal_eats_the_web_edge_next_to_it():
    """The floor pedestals are the hardware that genuinely bit v1: their top
    is at the PEN's level, just outside the web."""
    spec = layout.study_spec(31, "inv", (0.90, 0.60), h=H)
    px = 0.90
    boxes = mounts.arm_mount_boxes("floor", (px, -0.145), 0.0, H,
                                   tag="mount:13")
    x, y = px, 0.02                   # right at the south edge, over the plate
    assert _solve(spec, x, y, ()) is not None
    assert _solve(spec, x, y, boxes) is None
    # 0.30 m along the edge, clear of the 0.30 m pedestal, it survives
    assert _solve(spec, px + 0.32, y, boxes) is not None


def test_static_margin_is_the_gate_the_boxes_are_judged_by():
    """Not a private threshold: the mount model reuses rig_final's policy."""
    assert mounts.MOUNTS.margin == rig_final.STATIC_MARGIN


# ---------------------------------------------------------------------------
# the coarse proxy
# ---------------------------------------------------------------------------
def _grid(step=0.02, r=1.2):
    a = np.arange(-r, r + 1e-9, step)
    X, Y = np.meshgrid(a, a)
    return np.column_stack([X.ravel(), Y.ravel()])


def test_the_wedge_subtraction_removes_what_is_behind_a_column():
    P = _grid()
    base = np.array([0.0, 0.0])
    d = np.linalg.norm(P - base, axis=1)
    mask = (d >= 0.20) & (d <= 0.84)
    col = np.array([0.40, 0.0])
    shadows = [(col, 0.24)]
    keep = mounts.apply_keepouts(mask, P, base, shadows, [])
    assert keep.sum() < mask.sum()

    def at(x, y):
        return int(np.argmin(np.linalg.norm(P - [x, y], axis=1)))
    behind, beside = at(0.75, 0.0), at(0.0, 0.75)
    assert mask[behind] and mask[beside]
    assert not keep[behind], "a cell directly behind the column is shadowed"
    assert keep[beside], "a cell 90 degrees away is untouched"
    # the disc around the column itself goes too
    assert not keep[at(0.40, 0.10)]
    # and the wedge is a WEDGE: it widens with distance
    assert not keep[at(0.80, 0.15)]
    assert keep[at(0.30, 0.25)]


def test_a_pedestal_keepout_is_a_footprint_not_a_shadow():
    """Links pass OVER a pedestal, so it may not shadow the cells behind."""
    P = _grid()
    base = np.array([0.0, 0.0])
    d = np.linalg.norm(P - base, axis=1)
    mask = (d >= 0.20) & (d <= 0.84)
    blocks = [(np.array([0.25, -0.15]), np.array([0.55, 0.15]), 0.10)]
    keep = mounts.apply_keepouts(mask, P, base, [], blocks)

    def at(x, y):
        return int(np.argmin(np.linalg.norm(P - [x, y], axis=1)))
    assert not keep[at(0.40, 0.0)], "inside the inflated footprint"
    assert keep[at(0.80, 0.0)], "behind it, but reachable from above"


def test_z_bands_decide_shadow_versus_keepout():
    """The physical gate, not a tuning knob: an inverted arm's own chain
    hangs 0.30 m below its base plane, so a neighbour's plate at the SAME
    height is unreachable and earns no correction at all."""
    others = [("inv", (0.60, 0.0))]
    sh, bl = mounts.keepouts("inv", (0.0, 0.0), H, others, H)
    assert sh == [] and bl == []
    # drop the mount plane into the arm's chain envelope and it bites
    sh, bl = mounts.keepouts("inv", (0.0, 0.0), H, others, H - 0.40)
    assert len(sh) == 1 and bl == []
    # a floor arm's elbow climbs to base + 0.649, so a LOW ceiling bites it
    sh, _ = mounts.keepouts("floor", (0.0, 0.0), 0.0127, others, 0.75)
    assert len(sh) == 1
    sh, _ = mounts.keepouts("floor", (0.0, 0.0), 0.0127, others, 0.922)
    assert sh == []
    # a floor pedestal is at the PEN's level and always earns a keep-out
    _, bl = mounts.keepouts("inv", (0.0, 0.0), H, [("floor", (0.6, 0.0))], H)
    assert len(bl) == 1


def test_boom_shadow_active_reports_the_headroom():
    ok, head = mounts.boom_shadow_active("inv", H, H)
    assert not ok and head > 0
    ok, head = mounts.boom_shadow_active("floor", 0.0127, 0.60)
    assert ok and head < 0


# ---------------------------------------------------------------------------
# spacing minima implied by the hardware
# ---------------------------------------------------------------------------
def test_hardware_minima_are_derived_from_the_collision_margin():
    m = mounts.MOUNTS
    assert mounts.min_column_spacing(m) == pytest.approx(
        float(np.hypot(*m.plate_xy)) + m.margin)
    assert mounts.min_pedestal_spacing(m) == pytest.approx(
        float(np.hypot(*m.ped_xy)) + m.margin)
    # the base-spacing rule the search already carried is the STRICTER one,
    # so the hardware minima never get to be the active constraint
    assert mounts.min_column_spacing(m) < layout.MIN_BASE_DIST


def test_check_spacing_catches_interpenetrating_hardware():
    lay = dict(layout.LAYOUT_V1)
    bad = layout.check_spacing(dict(lay, inv=[(0.5, 1.0), (0.6, 1.0),
                                              (0.5, 2.8), (1.2, 2.8)]))
    assert any("hardware" in b for b in bad)
    assert layout.check_spacing(layout.LAYOUT_V1) == []


# ---------------------------------------------------------------------------
# THE NEIGHBOUR'S OWN BODY COLUMN (2026-08-25)
# ---------------------------------------------------------------------------
def test_the_column_radius_is_the_conductors_capsule_plus_its_calibration():
    """The number is DERIVED, not chosen: a box gate is compared against
    STATIC_MARGIN and the conductor's pair gate against SAFETY + CALIB, so the
    obstacle carries the difference and the two gates become one statement."""
    from aris_sixarm import coordination, frames
    m = mounts.MOUNTS
    assert m.link_r == coordination.LINK_R
    assert m.calib == coordination.CALIB_M
    assert m.d1 == frames.DH[0][2]
    assert m.column_r == pytest.approx(m.link_r + m.calib)
    # the capsule (0, 1) of the conductor's own table is what is being modelled
    assert coordination.CAPSULES[0] == (0, 1, coordination.LINK_R)
    # and the equality that makes the box gate mean the conductor's margin
    assert m.column_r + rig_final.STATIC_MARGIN == pytest.approx(
        coordination.LINK_R + coordination.SAFETY_M + coordination.CALIB_M)


def test_the_column_box_is_the_capsule_aabb_at_the_base():
    spec = layout.study_spec(31, "inv", (0.9, 1.20), h=H)
    b = mounts.arm_column_box(spec)
    m = mounts.MOUNTS
    assert b["tag"] == "body:31" and b["name"] == "body:31_column"
    # an inverted arm's column hangs DOWN from the flange, d1 long
    assert b["hi"][2] == pytest.approx(H + m.column_r)
    assert b["lo"][2] == pytest.approx(H - m.d1 - m.column_r)
    assert (b["hi"][:2] - b["lo"][:2]) == pytest.approx(
        [2 * m.column_r, 2 * m.column_r])
    # a floor arm's runs UP from its plate instead
    up = mounts.arm_column_box(layout.study_spec(13, "floor", (0.6, -0.2), h=H))
    assert up["hi"][2] == pytest.approx(mounts.Z_FLOOR + m.d1 + m.column_r)


def test_a_pose_that_clears_the_column_box_clears_the_real_arm():
    """The box is conservative against the thing it stands for: clearing it by
    STATIC_MARGIN means clearing the neighbour's real capsule by the margin
    the conductor will ask for."""
    from aris_sixarm import coordination, frames
    rng = np.random.default_rng(11)
    spec = layout.study_spec(31, "inv", (0.9, 1.20), h=H)
    nb = layout.study_spec(71, "inv", (0.9 + 0.61, 1.20), h=H)
    box = mounts.arm_column_box(nb)
    Twb = spec.T_world_base()
    p0, p1 = np.array(nb.T_world_base()[:3, 3]), None
    p1 = p0 + mounts.MOUNTS.d1 * np.asarray(nb.T_world_base()[:3, 2], float)
    n_checked = 0
    for _ in range(400):
        q = rng.uniform(frames.FR3_MIN, frames.FR3_MAX)
        T, pts = frames.fk(q)
        P = np.vstack([pts, (T[:3, 3] + T[:3, :3] @ [0, 0, 0.11])[None]])
        Pw = (Twb[:3, :3] @ P.T).T + Twb[:3, 3]
        cl = float(rig_final.chain_static_clearance(Pw, [box])[0])
        if cl < rig_final.STATIC_MARGIN:
            continue
        n_checked += 1
        # the same chain against the REAL capsule, the conductor's way
        worst = np.inf
        for (i, j, r) in rig_final.STATIC_CAPSULES:
            d = coordination.seg_seg_dist(Pw[i], Pw[j], p0, p1)
            worst = min(worst, float(d) - r - coordination.LINK_R)
        assert worst >= coordination.SAFETY_M + coordination.CALIB_M - 1e-9
    assert n_checked > 50, "the sample must actually exercise the gate"


def test_the_partners_column_removes_cells_from_the_real_atlas():
    """The finding this obstacle exists for: on a 0.61 m transverse pair the
    empty-air atlas certifies cells that the PARTNER is standing on."""
    lay = layout.paired_grid(rows=1, h=H)          # one pair, 0.61 m apart
    fl = layout.build_fleet(dict(lay, floor=[], h=H))
    aid, other = sorted(fl)[0], sorted(fl)[1]
    spec, nb = fl[aid], fl[other]
    x, y = nb.xy                                   # dead under the partner
    green = layout.study_spec(spec.arm_id, "inv", spec.xy, h=H)
    assert _solve(green, x, y, ()) is not None, "empty air reaches it"
    assert _solve(spec, x, y, spec.static_obstacles()) is None
    # ...and the column is what did it, not the plate or the boom (which live
    # at z >= h and no drawing pose can reach — the study's own claim)
    hw = [b for b in spec.static_obstacles() if b["tag"].startswith("mount")]
    assert _solve(spec, x, y, hw) is not None


def test_scene_check_catches_a_column_graze_on_its_own():
    """The checker re-derives the column from the base transform and its own
    capsule radius — it never reads `mounts`, and it gates arms that are not
    in the timeline at all."""
    from aris_sixarm import frames, scene_check
    fl = layout.FLEET_PROPOSED
    aid, nb = 13, 17                               # a transverse pair
    q = np.asarray(layout.Q_PARK_PROPOSED[aid], float)
    P = scene_check._chain(q, fl[aid], None, 0.110)
    cols = scene_check.neighbour_columns(fl, aid, {aid})
    rr = scene_check.RADII_LAT if P.shape[0] >= 11 else scene_check.RADII_FINAL
    assert len(cols) == 5, "five arms are still in the room"
    good = float(scene_check.column_clearance(P[None], cols, rr)[0])
    assert good >= 0.08, "the parked fleet already clears every column"
    # a pose whose wrist is inside the partner's column is refused, and the
    # refusal survives the partner being absent from the timeline
    p0, p1 = scene_check.base_column(fl[nb])
    bad = P.copy()
    bad[5:9] = 0.5 * (p0 + p1)
    assert float(scene_check.column_clearance(bad[None], cols, rr)[0]) < 0.0
    assert frames.DH[0][2] == scene_check.COLUMN_D1
