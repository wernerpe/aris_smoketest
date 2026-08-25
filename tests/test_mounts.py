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
        assert tags == {f"mount:{o}" for o in fl if o != aid}
    # ... and the count matches the hardware: 2 boxes per inverted arm, 1 per
    # floor arm, minus the arm's own
    inv_ids = [a for a, s in fl.items() if s.mount == "inv"]
    assert len(fl[inv_ids[0]].static_obstacles()) == 2 * (len(inv_ids) - 1) + 2


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
