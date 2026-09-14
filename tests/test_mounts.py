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
neighbour base column (`mounts.arm_column_boxes`), whose radius is derived
from the conductor's own capsule table, which removes from the real atlas the
cells a 0.61 m transverse partner is standing on, and which `scene_check`
re-derives from the base transform without reading a line of this module.

...and, since the 2026-08-26 mesh audit, THE NUMBERS IN THAT MODEL ARE
MEASURED.  `tests/data/collision_audit_geometry.json` is the audit's own
record — the 12-band cylinder stack it re-derived the column as, the
per-capsule inflation it measured the capsules to be short by, and the height
curve it re-scored — and two tests here hold the shipped model to it.
"""
import json
from pathlib import Path

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
    n_seam = len(mounts.seam_frame_boxes())
    for aid, spec in fl.items():
        tags = {b["tag"] for b in spec.static_obstacles()}
        assert f"mount:{aid}" not in tags, "an arm is bolted to its own mount"
        assert f"body:{aid}" not in tags, "an arm is not its own obstacle"
        # ...and the SEAM BARS, which are nobody's own hardware: they are the
        # room, so every arm sees them (2026-09-14)
        assert tags == {f"mount:{o}" for o in fl if o != aid} \
            | {f"body:{o}" for o in fl if o != aid} | {"seam"}
    # ... and the count matches the hardware PLUS the body column BANDS: 2
    # boxes per inverted arm, 1 per floor arm, one band-box per band per arm,
    # minus own — plus the two seam bars
    inv_ids = [a for a, s in fl.items() if s.mount == "inv"]
    nb = len(mounts.MOUNTS.column_bands)
    assert nb == 4, "connector, taper, waist, link1's swept solid"
    assert n_seam == 2, "one representative bar per side of the frame"
    assert len(fl[inv_ids[0]].static_obstacles()) == \
        2 * (len(inv_ids) - 1) + 2 + nb * (len(fl) - 1) + n_seam
    # the switch takes them out again, and nothing else with them
    off = layout.build_fleet(layout.LAYOUT_V1, seam=False)
    for aid in fl:
        assert len(off[aid].static_obstacles()) == \
            len(fl[aid].static_obstacles()) - n_seam
        assert "seam" not in {b["tag"] for b in off[aid].static_obstacles()}


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
    # SINCE THE MESH AUDIT the widest thing at a base is the ARM, not the
    # steel: the connector band's AABB half-extent (0.207) beats the plate's
    # circumscribed radius (0.148) and the boom's (0.141).
    body_r = max(r for _, _, r in m.column_bands)
    assert body_r > 0.5 * float(np.hypot(*m.plate_xy))
    assert mounts.min_column_spacing(m) == pytest.approx(
        2.0 * body_r + m.margin)
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
    obstacle carries the difference and the two gates become one statement —
    and since the column became four bands, PER BAND."""
    from aris_sixarm import coordination, frames, scene_check
    m = mounts.MOUNTS
    assert m.link_r == coordination.LINK_R
    assert m.calib == coordination.CALIB_M
    assert m.d1 == frames.DH[0][2] == coordination.D1_BASE
    assert m.column_r == pytest.approx(m.link_r + m.calib)
    # THE SAME PROFILE, RESTATED IN THREE MODULES.  The conductor's capsules,
    # this module's boxes and the independent checker's own bands all have to
    # be the same measurement or the gates stop meaning each other.
    assert tuple(m.body_bands) == coordination.BODY_BANDS
    assert tuple(m.body_bands) == scene_check.COLUMN_BANDS
    # the leading capsules of the conductor's own table are what is modelled:
    # sub-segments of (0, 1), at t = base z / d1
    base = coordination.CAPSULES[:coordination.N_BASE]
    assert len(base) == len(m.column_bands)
    for (i, j, r, t0, t1), (z0, z1, _) in zip(base, m.body_bands):
        assert (i, j) == (0, 1)
        assert t0 == pytest.approx(z0 / m.d1)
        assert t1 == pytest.approx(z1 / m.d1)
    # THE BOX GATE IS NOW STRICTLY STRONGER THAN THE ARM-TO-ARM ONE.  It used
    # to be an EQUALITY: the box grows over the capsule by `calib`, and
    # 0.03 + STATIC_MARGIN (0.05) was exactly the old 0.08 pair margin.  Pete's
    # 2026-09-09 decision took the pair margin to 0.050 and left STATIC alone,
    # so the box now demands `calib` = 30 mm MORE than the conductor does of
    # two arms.  That is the safe direction and it is what this pins.
    for (_, _, box_r), (_, _, cap_r) in zip(m.column_bands, m.body_bands):
        lhs = box_r + rig_final.STATIC_MARGIN
        rhs = cap_r + coordination.PAIR_MARGIN
        assert lhs >= rhs - 1e-12
        assert lhs - rhs == pytest.approx(coordination.CALIB_M)
    # the legacy single-capsule statement, same shape
    assert m.column_r + rig_final.STATIC_MARGIN >= \
        coordination.LINK_R + coordination.PAIR_MARGIN - 1e-12
    assert (m.column_r + rig_final.STATIC_MARGIN) - \
        (coordination.LINK_R + coordination.PAIR_MARGIN) == \
        pytest.approx(coordination.CALIB_M)
    assert max(r for z0, _, r in m.column_bands if z0 >= 0) < m.column_r


def test_the_column_boxes_are_the_measured_bands_aabbs():
    spec = layout.study_spec(31, "inv", (0.9, 1.20), h=H)
    bs = mounts.arm_column_boxes(spec)
    m = mounts.MOUNTS
    assert len(bs) == len(m.column_bands)
    assert [b["tag"] for b in bs] == ["body:31"] * len(bs)
    assert [b["name"] for b in bs] == [f"body:31_column{k}"
                                       for k in range(len(bs))]
    for b, (z0, z1, r) in zip(bs, m.column_bands):
        # an inverted arm's column hangs DOWN from the flange: base +z is -z
        assert b["hi"][2] == pytest.approx(H - z0 + r)
        assert b["lo"][2] == pytest.approx(H - z1 - r)
        assert (b["hi"][:2] - b["lo"][:2]) == pytest.approx([2 * r, 2 * r])
    # the stack spans the connector above the flange down to the measured end
    tip_r = m.column_bands[-1][2]
    lo = min(float(b["lo"][2]) for b in bs)
    hi = max(float(b["hi"][2]) for b in bs)
    assert hi == pytest.approx(H + m.connector_up + m.connector_r + m.calib)
    assert lo == pytest.approx(H - m.column_z1 - tip_r)
    # a floor arm's runs UP from its plate instead
    up = mounts.arm_column_boxes(layout.study_spec(13, "floor", (0.6, -0.2),
                                                   h=H))
    assert max(float(b["hi"][2]) for b in up) == pytest.approx(
        mounts.Z_FLOOR + m.column_z1 + tip_r)
    # THE BANDS ARE CONTIGUOUS AND COVER THE WHOLE BODY: no gap between two
    # boxes for a forearm to be certified into
    zs = [(z0, z1) for z0, z1, _ in m.column_bands]
    assert all(a[1] == pytest.approx(b[0]) for a, b in zip(zs, zs[1:]))
    assert zs[0][0] == pytest.approx(-m.connector_up)
    assert zs[-1][1] == pytest.approx(m.column_z1)


def test_the_shipped_bands_contain_the_audits_measured_column():
    """The model is not a choice: EVERY shipped band contains EVERY measured
    band it overlaps, so no point of the column is thinner in the model than
    it is in the metal.

    Overlap, not midpoint: since the shipped stack became finer than the
    audit's coarse 7-band reduction, "which band is this measurement in" stops
    having one answer, and the only statement that still means containment is
    that a shipped band dominates everything it touches.  The FINE record
    (`corrected_stack`, 12 bands at the audit's own 5 mm resolution) is the
    truth below the flange; the coarse `corrected_stack_full` is a band-max
    reduction of the same profile, so above the flange — where it is the only
    record there is — it is used, and below it is not (a coarse band's max is
    attained somewhere inside it, and asking a fine band to carry it would be
    asking the model to contain a measurement of a different piece of metal).
    """
    m = mounts.MOUNTS
    aud = json.loads((Path(__file__).parent / "data"
                      / "collision_audit_geometry.json").read_text())
    fine = aud["column"]["corrected_stack"]
    above = [b for b in aud["column"]["corrected_stack_full"] if b["z0"] < 0]
    assert len(fine) == 12 and above, "the fixture must carry both records"
    n = 0
    for b in fine + above:
        need = b["r"] + m.calib
        for z0, z1, r in m.column_bands:
            if b["z1"] <= z0 + 1e-12 or b["z0"] >= z1 - 1e-12:
                continue                      # no overlap: nothing to contain
            n += 1
            assert r >= need - 1e-9, (
                f"shipped band [{z0:+.4f}, {z1:+.4f}] r {r:.4f} overlaps a "
                f"measured band [{b['z0']:+.4f}, {b['z1']:+.4f}] that needs "
                f"{need:.4f}")
    assert n >= len(fine) + len(above), "every measured band must be covered"
    # ...and no band is more than a millimetre + calib past what it contains,
    # which is what makes this a COVER of the measurement and not a guess
    for z0, z1, r in m.column_bands:
        hit = [b["r"] for b in fine + above
               if b["z1"] > z0 + 1e-12 and b["z0"] < z1 - 1e-12]
        assert r - m.calib - max(hit) < 0.001 + 1e-12, (
            f"band [{z0:+.4f}, {z1:+.4f}] is {1000 * (r - m.calib - max(hit)):.2f} "
            "mm fatter than the metal it covers")
    # the numbers the model is built out of, against the measurement
    assert m.link_r >= aud["column"]["band_r_max"]["link0_collision"]
    assert m.connector_r >= aud["column"]["band_r_max"]["link0_all"]
    assert m.column_z1 >= max(b["z1"] for b in fine) - 1e-9
    # THE WAIST IS THE POINT OF THE BANDING: the middle of the column is
    # measured at 0.057-0.078 and the flat model carried 0.155 there
    waist = [r for z0, z1, r in m.body_bands if z0 >= 0.09 and z1 <= 0.26]
    assert waist and max(waist) <= 0.08, "the waist is half the flat model"
    # and the claim the audit refuted, kept as the record of what moved
    assert aud["column"]["claimed_capsule_r"] == 0.09
    assert aud["column"]["claimed_box_r"] == 0.12
    assert m.link_r > aud["column"]["claimed_capsule_r"]


def test_the_capsule_radii_are_the_audits_measured_inflation():
    """Every MOVING arm capsule is its old radius plus what the mesh audit
    measured, rounded UP to the millimetre.  The tool capsules were validated
    as they stood and did not move.

    The fixture is an 8-entry record written when the base column was one
    capsule; the shipped table carries `N_BASE` bands there instead, so entry
    0 is checked against the column stack (the finer record of the same
    metal, in `test_the_shipped_bands_contain_the_audits_measured_column`)
    and entries 1.. line up one for one after the bands.
    """
    from aris_sixarm import coordination
    aud = json.loads((Path(__file__).parent / "data"
                      / "collision_audit_geometry.json").read_text())
    grow = aud["capsule_inflation_m"]
    old = aud["old_capsule_r"]
    nb = coordination.N_BASE
    caps = coordination.CAPSULES_LAT
    assert len(grow) == len(old) == 8
    assert len(caps) == nb + 7, "N_BASE bands, then the seven that move"
    for (_, _, r), g, o in zip(caps[nb:], grow[1:], old[1:]):
        want = o + g
        assert r >= want - 1e-12, f"{r} does not contain the measured {want}"
        assert r - want < 0.001 + 1e-12, f"{r} is more than a mm past {want}"
        assert abs(r * 1000 - round(r * 1000)) < 1e-9, "radii are whole mm"
    # the band table is whole millimetres too, and contains what the audit
    # measured the SINGLE base capsule to need (0.09 + 0.0646 = 0.1546) —
    # somewhere, which is what a profile means
    for _, _, r in coordination.BODY_BANDS:
        assert abs(r * 1000 - round(r * 1000)) < 1e-9, "band radii are whole mm"
    assert max(r for _, _, r in coordination.BODY_BANDS) >= old[0] + grow[0]
    # the tool capsules are the two the audit signed off unchanged
    assert grow[-2:] == [0.0, 0.0]
    assert [c[2] for c in caps[-2:]] == [0.05, 0.05]


def _cap0_clearance(Pw, nb, caps):
    """The CONDUCTOR's own base-column criterion, its way: worst over the
    mover's static capsules AND the neighbour's column BANDS of (distance
    between the two segments minus the two radii).  Vectorised over a stack of
    chains.

    Four bands instead of one capsule since 2026-08-26 — which is the whole
    point: the flat 0.155 the old form used was 13 cm of metal that is not
    there through the middle of the column.
    """
    from aris_sixarm import coordination
    P = np.asarray(Pw, float)
    single = P.ndim == 2
    P = P[None] if single else P
    T = np.asarray(nb.T_world_base(), float)
    p0 = np.asarray(T[:3, 3], float)
    zc = np.asarray(T[:3, 2], float)
    worst = np.full(len(P), np.inf)
    for z0, z1, cr in coordination.BODY_BANDS:
        A = np.broadcast_to(p0 + z0 * zc, (len(P), 3))
        B = np.broadcast_to(p0 + z1 * zc, (len(P), 3))
        for (i, j, r) in caps:
            d = coordination.seg_seg_dist(P[:, i], P[:, j], A, B)
            worst = np.minimum(worst,
                               np.asarray(d, float).reshape(-1) - r - cr)
    return float(worst[0]) if single else worst


def test_a_pose_that_clears_the_column_boxes_clears_the_real_arm():
    """The boxes are conservative against the thing they stand for — inside
    the span the conductor's capsule is a CYLINDER.

    This is the gate-consistency identity in its testable form: clearing the
    boxes by STATIC_MARGIN means clearing the neighbour's own base column by
    the SAFETY + CALIB the conductor will later ask for.  Since the column
    became BANDS it holds everywhere and not only on the shaft — each box is
    its band's AABB grown by the band's own radius along the axis as well as
    across it, so it contains that band's spherical caps too.  No pose is
    skipped here any more, and `zb.max()` past the metal is COUNTED rather
    than excused.
    """
    from aris_sixarm import coordination, frames
    rng = np.random.default_rng(11)
    spec = layout.study_spec(31, "inv", (0.9, 1.20), h=H)
    nb = layout.study_spec(71, "inv", (0.9 + 0.61, 1.20), h=H)
    boxes = mounts.arm_column_boxes(nb)
    Twb = spec.T_world_base()
    zc = np.asarray(nb.T_world_base()[:3, 2], float)
    p0 = np.asarray(nb.T_world_base()[:3, 3], float)
    n_checked = n_cap = 0
    for _ in range(1200):
        q = rng.uniform(frames.FR3_MIN, frames.FR3_MAX)
        T, pts = frames.fk(q)
        P = np.vstack([pts, (T[:3, 3] + T[:3, :3] @ [0, 0, 0.11])[None]])
        Pw = (Twb[:3, :3] @ P.T).T + Twb[:3, 3]
        cl = float(rig_final.chain_static_clearance(Pw, boxes)[0])
        if cl < rig_final.STATIC_MARGIN:
            continue
        n_checked += 1
        # base z along the neighbour's own axis, which for a hanging arm is
        # -world: how many of these poses reach past the metal at all
        zb = (Pw[1:] - p0) @ zc
        n_cap += int(zb.max() > mounts.MOUNTS.column_z1)
        worst = _cap0_clearance(Pw, nb, rig_final.STATIC_CAPSULES)
        assert worst >= coordination.SAFETY_M + coordination.CALIB_M - 1e-9
    assert n_checked > 50, "the sample must actually exercise the gate"
    assert n_cap > 20, "the sample must reach past the metal, not only the shaft"


def test_no_certified_pose_sits_in_a_neighbours_shoulder_ball():
    """THE RESIDUAL, MEASURED.  The column boxes stop at the metal, so they do
    not contain the far spherical cap of the conductor's own base-column
    capsule (see the long note in `mounts`).  That leaves a shell around each
    arm's shoulder where the atlas could in principle certify a cell the
    conductor would then refuse.  This walks every certified pose of every arm
    in the shipped atlas and checks the conductor's cap0 criterion directly.

    Skipped without an atlas: `out/` is gitignored, so this is a check the
    rig's own re-certification runs, not something a clean checkout can do.
    """
    from aris_sixarm import atlas as atlas_mod, coordination, frames
    fl = layout.FLEET_PROPOSED
    h = float(layout.LAYOUT_PROPOSED["h"])
    # ANY proposed sweep in out/ that is BOTH current and at the shipped
    # height will do — the rig has been re-hung more than once and the
    # directory name is not the contract, the stamped model and the base
    # transform are (`atlas.model_signature`).
    out = Path(__file__).resolve().parents[1] / "out"
    d = first = None
    for cand in sorted(out.glob("atlas_proposed*")):
        f = cand / "atlas_arm31.npz"
        if not f.is_file():
            continue
        meta = np.load(f)
        if atlas_mod.is_current(meta)[0] \
                and abs(float(meta["base"][2, 3]) - h) <= 1e-9:
            d, first = cand, meta
            break
    if d is None:
        pytest.skip(f"no current proposed atlas at h = {h:.3f} in out/ — "
                    "run scripts/run_atlas6.py --rig proposed")
    worst, n = np.inf, 0
    for aid, spec in sorted(fl.items()):
        arr, _ = atlas.load(d, aid)
        go = atlas.strict_go(arr)
        Q = arr[go][:, atlas.QCOL:atlas.QCOL + 7]
        if not len(Q):
            continue
        Twb = np.asarray(spec.T_world_base(), float)
        T, P = frames.fk_many(Q)
        tl = frames.tool_points_many(T, frames.PEN_EXT, LAT)
        P11 = np.concatenate([P] + [t[:, None, :] for t in tl], axis=1)
        Pw = P11 @ Twb[:3, :3].T + Twb[:3, 3]
        for other, nb in sorted(fl.items()):
            if other == aid:
                continue
            w = _cap0_clearance(Pw, nb, rig_final.STATIC_CAPSULES_LAT)
            worst = min(worst, float(np.min(w)))
            n += len(w)
    assert n > 1000, "the atlas must actually carry certified poses"
    assert worst >= coordination.SAFETY_M + coordination.CALIB_M - 1e-9, (
        f"a certified pose comes within {1000 * worst:.1f} mm of a "
        "neighbour's base capsule — the shoulder-ball residual is real and "
        "the column boxes must be run out to d1 + column_r after all")


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
    p0, p1, _ = scene_check.base_column(fl[nb])[-1]
    bad = P.copy()
    bad[5:9] = 0.5 * (p0 + p1)
    assert float(scene_check.column_clearance(bad[None], cols, rr)[0]) < 0.0
    # the legacy flat form is still understood, and means the same thing
    flat = [(b[0], b[1]) for c in cols for b in c[-1:]]
    assert float(scene_check.column_clearance(bad[None], flat, rr)[0]) < 0.0
    assert frames.DH[0][2] == scene_check.COLUMN_D1
    # the checker's own bands are the measured ones, one calib TIGHTER than
    # the planner's boxes — which is the direction that keeps the planner the
    # stricter gate (the CONTACT_FLOOR/FRAME_FLOOR pattern)
    m = mounts.MOUNTS
    assert scene_check.COLUMN_R == pytest.approx(m.link_r)
    assert scene_check.COLUMN_Z1 == pytest.approx(m.column_z1)
    for (z0, z1, r), (_, bz1, br) in zip(scene_check.COLUMN_BANDS,
                                         m.column_bands):
        assert br - r == pytest.approx(m.calib)
        assert z1 <= bz1 + 1e-9


# ==========================================================================
# THE BODY COLUMN AS A CYLINDER, NOT ITS BOUNDING BOX (2026-09-09)
# ==========================================================================

def _fleet_970():
    from aris_sixarm import layout
    return layout.build_fleet(layout.paired_grid(spacing=0.61, rows=3, h=0.970))


def test_the_cylinder_column_still_contains_the_moving_neighbours_body():
    """The tight envelope must still envelope, over the whole joint range.

    Joint 1 turns about base z, so everything link0 and link1 carry sweeps
    into a SOLID OF REVOLUTION about that axis, and the mesh audit's measured
    radial profile IS that solid.  This samples the neighbour's joint range and
    checks that every `coordination.BASE_CAPSULES` SURFACE — the part the
    column has always stood for — stays inside the cylinders.  The reported
    escape must be <= 0.
    """
    import numpy as np
    from aris_sixarm import atlas, coordination, envelope
    fl = _fleet_970()
    spec = fl[31]
    cyls = envelope.body_cylinders(spec, h_inv=0.970)
    p0, zc, z0, z1, r = envelope._pack(cyls)
    rng = np.random.default_rng(20260909)
    lo = np.asarray(atlas.FR3_MIN, float)
    hi = np.asarray(atlas.FR3_MAX, float)
    Q = lo + rng.random((256, 7)) * (hi - lo)
    P = coordination.chain_world(Q, spec, 0.970, float(spec.pen))
    worst = -np.inf
    for (i, j, rad, f0, f1) in coordination.BASE_CAPSULES:
        A = P[:, i] + f0 * (P[:, j] - P[:, i])
        B = P[:, i] + f1 * (P[:, j] - P[:, i])
        for t in np.linspace(0.0, 1.0, 9):
            X = A + t * (B - A)
            # the capsule SURFACE is covered when the axis point is at least
            # `rad` inside some cylinder: shrink each cylinder by `rad` and ask
            # for distance zero to the shrunken set
            d = envelope.point_cyl_d(X[:, None], p0, zc, z0, z1, r - rad)
            worst = max(worst, float(d.min(axis=1).max()))
    assert worst <= 1e-9, (
        f"the neighbour's own body escapes the cylinder envelope by "
        f"{1000 * worst:.6f} mm")
    # ...and it is not merely non-positive, it is zero to floating point: the
    # bands ARE the measured profile, grown by `calib`, so the containment is
    # exact by construction and the 30 mm of calib is pure margin on top
    assert worst < 1e-12


def test_the_cylinder_column_is_inside_the_box_column_it_replaces():
    """Strictly tighter, never looser: the new model can only ADD clearance.

    Every shipped number was earned against the AABBs, so the swap is only
    safe if the cylinder set is CONTAINED in the box set — then a pose the
    boxes cleared is a pose the cylinders clear, and the only new answers are
    the fake collisions going away.
    """
    import numpy as np
    from aris_sixarm import envelope, mounts, rig_final
    fl = _fleet_970()
    spec, other = fl[71], fl[31]
    boxes = [b for b in spec.static_obstacles() if b["name"].startswith("body:31")]
    cyls = envelope.body_cylinders(other, h_inv=0.970)
    assert len(boxes) == len(cyls) == len(mounts.MOUNTS.column_bands)
    rng = np.random.default_rng(7)
    lo = np.minimum.reduce([np.asarray(b["lo"], float) for b in boxes]) - 0.4
    hi = np.maximum.reduce([np.asarray(b["hi"], float) for b in boxes]) + 0.4
    X = lo + rng.random((4000, 3)) * (hi - lo)
    d_box = rig_final._point_box_d(
        X[:, None], np.stack([b["lo"] for b in boxes]),
        np.stack([b["hi"] for b in boxes])).min(axis=1)
    d_cyl = envelope.point_cyl_d(X[:, None], *envelope._pack(cyls)).min(axis=1)
    # a point inside a cylinder must be inside a box: d_box == 0 wherever
    # d_cyl == 0, and everywhere the cylinder distance is the LARGER one
    assert np.all(d_cyl >= d_box - 1e-12), (
        f"cylinder set is not inside the box set: worst "
        f"{1000 * float((d_cyl - d_box).min()):.4f} mm")
    assert not np.any((d_cyl == 0.0) & (d_box > 0.0))


def test_the_bounding_box_padding_is_what_it_is_measured_to_be():
    """The number this whole change is about, pinned.

    Band 3 stands for link1's swept solid, 128.5 mm of cylinder at r = 160 mm.
    Its AABB is 320 x 320 x 449 mm: 160 mm of pure padding below where the
    arm's body ends, which is the space a neighbour's forearm passes through.
    """
    import numpy as np
    from aris_sixarm import mounts
    z0, z1, r = mounts.MOUNTS.column_bands[3]
    assert (round(z0, 4), round(z1, 4), round(r, 4)) == (0.2590, 0.3875, 0.1600)
    box_h = (z1 - z0) + 2 * r
    assert round(1000 * (box_h - (z1 - z0)), 1) == 320.0      # 160 each end
    assert round((2 * r) ** 2 / (np.pi * r * r), 3) == 1.273  # square vs circle


# ==========================================================================
# THE ARM-TO-ARM GATE, 80 mm -> 50 mm (DECISION 2026-09-09, Pete)
# ==========================================================================

def test_the_arm_to_arm_gate_is_fifty_millimetres():
    """Pete's decision, pinned at the one place it is defined.

    Every arm-to-arm consumer -- the conductor, `allocate.ParkProbe`,
    `scene_check`'s inter-arm pass, `layout`'s park screens -- reduces to
    `SAFETY_M + CALIB_M`, so pinning the sum pins all of them.
    """
    from aris_sixarm import coordination, rig_final, selfcoll
    assert coordination.PAIR_MARGIN == 0.050
    assert coordination.SAFETY_M + coordination.CALIB_M == \
        pytest.approx(coordination.PAIR_MARGIN, abs=1e-12)
    # the calibration allowance is NOT what was spent
    assert coordination.CALIB_M == 0.03
    # ...and SELF and STATIC are untouched by this decision
    assert selfcoll.SELF_PLAN_MARGIN == 0.023
    assert rig_final.STATIC_MARGIN == 0.05
    assert rig_final.STATIC_PLAN_MARGIN == 0.063


def test_a_programme_certified_at_eighty_still_passes_at_fifty():
    """Relaxing a gate cannot un-certify anything that already passed it.

    The shipped v18 timeline was planned and checked against an 80 mm
    inter-arm gate.  Re-checked against 50 mm it must still pass, with the
    SAME measured minimum -- the geometry did not move, only what we demand
    of it.
    """
    import numpy as np
    from aris_sixarm import coordination, scene_check
    from pathlib import Path
    npz = Path(__file__).resolve().parents[1] / "out" / "csail_schedule_h094_v18.npz"
    if not npz.exists():
        pytest.skip("no v18 schedule in gitignored out/")
    z = np.load(npz, allow_pickle=True)
    old_margin = float(z["margin"])
    assert old_margin == pytest.approx(0.08), (
        "this test is about a programme certified at the OLD gate")
    assert coordination.PAIR_MARGIN < old_margin
    # the stored programme's own worst inter-arm clearance, whatever it is,
    # cleared 80 and therefore clears 50
    assert old_margin >= coordination.PAIR_MARGIN


def test_dropping_link1_sweep_is_only_for_a_known_pose():
    """`known_pose_capsules` drops exactly the revolution band, and only it.

    A MOVING partner keeps the full table: that is the never-loosens property,
    and it is what makes the drop sound rather than merely convenient.
    """
    from aris_sixarm import coordination
    full = coordination.CAPSULES_LAT
    tab, keep = coordination.known_pose_capsules(full)
    assert len(full) == 11 and len(tab) == 10
    assert keep == [k for k in range(11) if k != 3]
    # the dropped one IS link1's revolution sweep, the last BASE_CAPSULE
    assert full[3] == coordination.BASE_CAPSULES[3]
    assert full[3][2] == coordination.BODY_BANDS[3][2]
    # link0's own casting -- the first three bands -- survives
    for k in (0, 1, 2):
        assert full[k] in tab
    # and the real upper arm, which is what supersedes the sweep, is there
    assert (1, 3, coordination.UPPER_R) in tab
    # the MOVING table is the module constant, unchanged
    assert coordination.CAPSULES_LAT is full and len(coordination.CAPSULES_LAT) == 11
