"""The system model: does the committed asset still say what the code says?

Three things are pinned here and they are different questions.

  * THE TRUTH MODULE reproduces the original drawing.  `aris_sixarm/
    system_model.py` derives the cage from `rig_final`'s extraction of the
    drawing; if that extraction moves, or if somebody edits a dimension in
    place, these fail.  The load-bearing one is the DATUM: the cage's total
    height above the floor must still come out at the drawing's own 233,7 cm.

  * THE GENERATOR IS DETERMINISTIC and the committed URDFs are what it makes.
    No clock, no rng, no dict ordering.

  * THE ASSET AGREES WITH THE PACKAGE.  Every cage body, every base pose,
    every pen tip — the last one re-derived from `frames.fk` by walking the
    URDF's own joint tree in plain XML, so neither side borrows the other's
    forward kinematics.

The drake-level checks (parse, collision roles, geometry provenance) live in
`scripts/check_system_model.py`, which needs the station venv.  Nothing here
imports drake.
"""
import hashlib
import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from aris_sixarm import frames, layout, mounts, rig_final, selfcoll
from aris_sixarm import system_model as SM
from aris_sixarm.frames import D_HAND_TCP, PEN_EXT_HOLDER, PEN_LAT_HOLDER, fk

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "assets/system_model"
INSTALL = DIR / "installation.urdf"
CAPSULES = DIR / "installation_capsules.urdf"
ENV = DIR / "environment.urdf"
MANIFEST = DIR / "model_manifest.json"
H_INV = float(layout.LAYOUT_PROPOSED["h"])
MM = SM.MM


def _gen():
    spec = importlib.util.spec_from_file_location(
        "gen_system_model", ROOT / "scripts/gen_system_model.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def root():
    return ET.parse(INSTALL).getroot()


@pytest.fixture(scope="module")
def links(root):
    return {el.get("name"): el for el in root.findall("link")}


@pytest.fixture(scope="module")
def joints(root):
    return {el.get("name"): el for el in root.findall("joint")}


@pytest.fixture(scope="module")
def manifest():
    return json.loads(MANIFEST.read_text())


# ---------------------------------------------------------------------------
# 1.  the truth module vs the drawing
# ---------------------------------------------------------------------------
def test_the_cage_reproduces_the_drawings_own_overall_height():
    """The one number the whole datum correction turns on.

    The drawing carries a single overall dimension, 233,7 cm, and it is FLOOR
    to top-of-construction.  The model puts the paper 636.68 mm above the
    floor and the cage top 1699.82 mm above the paper; those must add back to
    the drawing's number, or the correction is not a correction.
    """
    assert SM.CAGE_TOTAL_H == pytest.approx(2336.5, abs=1e-6)
    assert SM.FLOOR_Z == pytest.approx(-636.68, abs=1e-6)
    assert SM.O_TOP - SM.FLOOR_Z == pytest.approx(SM.CAGE_TOTAL_H, abs=1e-9)
    # and the top slab is exactly one profile thick, as a 3 in member must be
    assert SM.GRID_T - SM.GRID_U == pytest.approx(SM.PROFILE, abs=1e-9)


def test_the_model_reproduces_the_drawings_own_TEXT_block():
    """A second provenance path: the drawing's words, not its solids.

    Everything else here descends from the DXF through
    `rig_final.FRAME_BOXES_W_CM`.  The PDF also carries a human-readable text
    block, written by the person who drew it, and it is an independent
    statement of the same geometry.  If the extraction ever silently changed
    which solid it read, these would part company with it.

    Quoted verbatim from "Drawing installation, 3 arms, 1 up, 1 side, 1 down,
    sizes cm", page 1:

        "Table height 63,5 cm, Height construction  233,7 cm"
        "Entire cage construction built with T-slotted aluminum profile:
         3" x 3" (7,62 x 7,62 cm) or 1,5" x 3""
        "Base plate robot arms: 22,6 x 19,0 cm"
        "Position of center of rotation of the arm: 13,8 x 9,5 cm on the
         base plate"
        "Note: the axis of rotation of the robot arms is not in the center
         of the base plates!"

    The tolerances are the TEXT's own rounding, not the model's: the block is
    quoted to the millimetre and the solids carry more digits.
    """
    # "Height construction 233,7 cm" — and the front view dimensions it from
    # the FLOOR.  This is the whole datum correction, in the author's words.
    assert SM.CAGE_TOTAL_H == pytest.approx(2337.0, abs=1.0)
    # "Table height 63,5 cm"
    assert -SM.FLOOR_Z + SM.TABLE_TOP_Z == pytest.approx(635.0, abs=2.0)
    # '3" x 3" (7,62 x 7,62 cm)'
    assert SM.PROFILE == pytest.approx(76.2, abs=1e-9)
    # "Base plate robot arms: 22,6 x 19,0 cm"
    assert SM.PLATE[0] == pytest.approx(226.0, abs=0.5)
    assert SM.PLATE[1] == pytest.approx(190.0, abs=0.5)
    # "the axis of rotation ... 13,8 ... on the base plate", i.e. the centre of
    # a 22.6 plate is 11.3 from the edge and the axis is 13.8, so the plate
    # centre sits 25 mm off the axis — which is the offset whose DIRECTION is
    # OPEN_QUESTIONS["plate_offset_direction"], and the drawing itself calls
    # the asymmetry out: "Note: the axis of rotation of the robot arms is not
    # in the center of the base plates!"
    assert SM.PLATE_OFF == pytest.approx((22.6 / 2 - 13.8) * 10, abs=0.5)
    # "218,4" on both views — and the derived frame lands on it by choosing
    # the 190.5 margin, which is the one place the new frame quotes the old
    assert SM.FR_W == pytest.approx(2184.0, abs=1.0)


def test_the_drawings_text_and_its_solids_disagree_about_the_mount_plane():
    """91,6 cm in the text, 92.2 in the solid.  Noted, and it changes nothing.

    "Distance backside base plate to surface ( drawing paper: 91,6 cm" says
    916 mm; `down_plate` lo sits at 922.0 above the paper.  The 6 mm is a
    text-versus-solid disagreement in the ORIGINAL drawing, and this model is
    unaffected by it because the mount plane in force is
    `layout.LAYOUT_PROPOSED["h"] = 940`, not the drawing's.  It is pinned so
    that nobody re-derives 922 from the text and thinks the model is wrong.
    """
    assert SM.O_MOUNT == pytest.approx(922.0, abs=0.05)
    assert abs(SM.O_MOUNT - 916.0) == pytest.approx(6.0, abs=0.1)
    # 970.0 since 2026-09-10 (was 940.0, and 850.0 before that).  The point
    # of this line is that `H_MOUNT` is READ FROM `layout.LAYOUT_PROPOSED`
    # and is NOT the drawing's own 916/922 — so it is pinned as a literal,
    # because reading the layout on both sides would pass at any height.
    assert SM.H_MOUNT == 970.0, "the rig's own height, not the drawing's"


def test_the_paper_datum_is_corroborated_by_two_independent_numbers():
    """The whole correction turns on where the paper's top surface is.

    `system_model` derives it from the drawing's `table_block`: the paper's
    TOP is `PAPER_ORIGIN_W_CM[2]` and the table's top is `table_block` hi, so
    the gap between them is the paper's thickness — 2.00 mm.  `rig_final`
    carries that thickness as its own constant, from the drawing's text rather
    than from the solid.  They agree exactly, which is what says the paper
    origin is on the right face.
    """
    assert SM.PAPER_T == pytest.approx(rig_final.PAPER_THICK_CM * 10.0,
                                       abs=1e-9)
    assert SM.PAPER_T == pytest.approx(2.0, abs=1e-9)
    paper = next(b for b in SM.bodies() if b.name == "paper")
    assert paper.hi[2] == 0.0, "z = 0 is the paper's TOP surface"
    assert paper.lo[2] == pytest.approx(SM.TABLE_TOP_Z, abs=1e-9)


def test_the_datum_bug_is_modelled_as_a_difference_not_silently_adopted():
    """`mounts.ceiling_z` stays where it is; the model records the gap."""
    assert mounts.MOUNTS.ceiling_z == 2.34, "mounts.py must not be edited here"
    assert SM.GRID_U_CODE == pytest.approx(2340.0)
    assert SM.GRID_U == pytest.approx(1623.62, abs=1e-6)
    item = [r for r in SM.reconciliation() if r["item"] == "ceiling datum"]
    assert len(item) == 1
    assert item[0]["delta_mm"] == pytest.approx(716.38, abs=1e-6)
    assert "CONSERVATIVE" in item[0]["direction"]


def test_the_post_length_is_the_originals_again_at_the_corrected_datum():
    """688.6 against the original's 736.9 — the code datum asks for 1405.0.

    THE HEADLINE IS THE ORDER OF MAGNITUDE, NOT THE 48 MM.  At h = 0.940 the
    corrected post came out 718.60 and landed within 18 mm of the post the
    original rig actually has, which was the coincidence that made the datum
    bug obvious.  At the h = 0.970 adopted 2026-09-10 it is 688.60 — 48.3 mm
    shorter than the built one, and still nowhere near the 1404.98 the buggy
    paper-referenced ceiling asks for.  The three numbers are pinned together
    because what this test is about is that they are three DIFFERENT numbers
    from three different datums, and which of them the fabricator cuts to.
    """
    z = SM.z_ladder()
    assert z["post_length"] == pytest.approx(688.60, abs=1e-6)
    assert SM.O_POST_L == pytest.approx(736.9, abs=0.05)
    assert SM.POST_L_CODE == pytest.approx(1404.98, abs=1e-6)
    # the corrected post is the one that is plausible; the code datum's is
    # 716.4 mm too long, and that gap does not move with h
    assert SM.POST_L_CODE - z["post_length"] == pytest.approx(716.38, abs=0.02)


def test_cage_members_come_from_the_drawing_not_from_here():
    """Every section and pitch must still equal the extracted drawing box."""
    lo, hi = [np.asarray(v) for v in
              (next(b["lo"] for b in rig_final.FRAME_BOXES_W_CM
                    if b["name"] == "down_plate"),
               next(b["hi"] for b in rig_final.FRAME_BOXES_W_CM
                    if b["name"] == "down_plate"))]
    assert SM.PLATE == pytest.approx(tuple(np.round((hi - lo) * 10.0, 2)))
    assert SM.PLATE[2] == pytest.approx(12.7, abs=1e-9)      # 0.5 in
    assert SM.POST_PITCH_X == pytest.approx(317.6, abs=1e-9)
    assert SM.POST_Y_BAND == pytest.approx(152.4, abs=1e-9)  # 2 x 3 in
    assert SM.POST_OVER == pytest.approx(34.98, abs=1e-9)
    assert SM.PROFILE == pytest.approx(3 * 25.4, abs=1e-9)
    assert SM.RAIL_LEN_X == pytest.approx(80.0 * 25.4, abs=1e-9)
    assert SM.FR_W == pytest.approx(86.0 * 25.4, abs=1e-9)


def test_the_plate_nests_between_the_posts_only_just():
    """7.79 mm each side — which is why the offset DIRECTION is load-bearing.

    MEASURED OFF THE BODIES THE MODEL EMITS, not off `POST_GAP`.  `POST_GAP`
    is the slot the DRAWING's slightly-fat 77.2 mm booms leave (240.5); this
    model builds nominal 76.2 posts, so its own slot is 241.4.  A test that
    reads the constant cannot see the pitch or the section move.
    """
    by = {b.name: b for b in SM.bodies()}
    for aid in layout.FLEET_PROPOSED:
        west, east = by[f"post{aid}_00"], by[f"post{aid}_10"]
        plate = by[f"plate{aid}"]
        slot = east.lo[0] - west.hi[0]
        assert slot == pytest.approx(SM.POST_SLOT, abs=1e-9)
        assert plate.lo[0] > west.hi[0], f"arm {aid}: plate fouls the west post"
        assert plate.hi[0] < east.lo[0], f"arm {aid}: plate fouls the east post"
        clear = min(plate.lo[0] - west.hi[0], east.lo[0] - plate.hi[0])
        assert clear == pytest.approx(SM.PLATE_SIDE_CLEAR, abs=0.01)
    clear = SM.PLATE_SIDE_CLEAR
    assert clear == pytest.approx(7.79, abs=0.01)
    assert "plate_offset_direction" in SM.OPEN_QUESTIONS
    q = SM.OPEN_QUESTIONS["plate_offset_direction"]
    assert "7.34" in q["rides_on"]
    # and the offset really does flip: the drawing has it in -x, the rig +x
    assert SM.PLATE_OFF < 0
    assert SM.plate_centre_x(0.0) == pytest.approx(25.15, abs=1e-9)


def test_the_rotated_gussets_actually_clear_each_other():
    """The reason they were rotated: 406.4 needed, 216.2 available."""
    assert SM.GUSSET_NEED > SM.GUSSET_GAP, "the conflict must still be real"
    assert SM.GUSSET_GAP == pytest.approx(216.20, abs=0.01)
    assert SM.GUSSET_PAIR_CLEAR > 0, "the fix must actually fix it"
    assert SM.GUSSET_PAIR_CLEAR == pytest.approx(89.20, abs=0.01)


def test_no_two_static_bodies_interpenetrate():
    """Steel inside steel is a length nobody can cut.

    This caught the corner legs: drawn to the perimeter rail's TOP they put
    76.2 mm of themselves inside the rail they carry, and `reconciliation()`
    published the resulting 1727.2 mm as a leg length.  They stop at the
    rail's underside now, the way the drop posts stop under the runway, and
    the cut length is 1651.0.
    """
    bs = SM.bodies()
    bad = []
    for i, a in enumerate(bs):
        for b in bs[i + 1:]:
            ov = [min(a.hi[k], b.hi[k]) - max(a.lo[k], b.lo[k])
                  for k in range(3)]
            if all(o > 1e-9 for o in ov):
                bad.append(f"{a.name} n {b.name} by "
                           f"{tuple(round(o, 2) for o in ov)}")
    assert not bad, "; ".join(bad)


def test_the_legs_are_a_length_somebody_can_cut():
    by = {b.name: b for b in SM.bodies()}
    leg = by["leg_FL"]
    rail = by["frame_side_W"]
    assert leg.hi[2] == pytest.approx(SM.GRID_U, abs=1e-9)
    assert rail.lo[2] == pytest.approx(SM.GRID_U, abs=1e-9), \
        "the leg must stop where the rail it carries begins"
    assert leg.hi[2] - leg.lo[2] == pytest.approx(1651.0, abs=0.01)
    item = next(r for r in SM.reconciliation() if r["item"] == "cage legs")
    assert item["delta_mm"] == pytest.approx(1651.0, abs=0.01)


def test_every_body_carries_a_provenance_class_and_assumed_ones_a_note():
    for b in SM.bodies():
        assert b.provenance in SM.PROVENANCE_CLASSES, b.name
        assert b.source, b.name
        if b.provenance == "ASSUMED":
            assert b.note or "ASSUMED" in b.source, b.name


def test_the_inverted_base_cable_is_carried_as_a_finding():
    """The model found a hole that has to be cut and nothing else had seen it.

    On an inverted arm the manufacturer's own base cable stub points UP
    through the mount plate and the clamp stack.  The collision shell stops at
    the flange, so no clearance check in this repo covers it.
    """
    trimesh = pytest.importorskip("trimesh")
    m = trimesh.load(DIR / "meshes/fr3/link0.gltf", force="mesh")
    R = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0.]])   # the visual's Rx(+90)
    v = (R @ np.asarray(m.vertices, float).T).T
    assert v[:, 2].min() == pytest.approx(-0.2307, abs=5e-4)
    shell = trimesh.load(DIR / "meshes/collision/link0.obj", force="mesh")
    # the shell stops at the flange to within 0.033 mm; the visual runs 230 mm
    # past it.  That gap IS the finding — if the shell ever grows a cable, the
    # hole stops being invisible and this test should be revisited.
    assert shell.bounds[0][2] > -1e-3, shell.bounds[0][2]
    assert shell.bounds[0][2] - v[:, 2].min() > 0.22
    q = SM.OPEN_QUESTIONS["base_cable_passthrough"]
    assert "230.7" in q["why"]
    assert [r for r in SM.reconciliation()
            if r["item"] == "base cable pass-through"]


def test_the_open_questions_are_the_ones_that_block_fabrication():
    want = {"plate_offset_direction", "ceiling_survey", "penholder_cradle",
            "cable_dress", "cage_legs", "gusset_attachment",
            "base_cable_passthrough"}
    assert set(SM.OPEN_QUESTIONS) == want
    for k, q in SM.OPEN_QUESTIONS.items():
        assert set(q) == {"what", "why", "rides_on", "answer_by", "blocking"}, k


def test_the_loader_finds_all_three_files_and_refuses_a_fourth():
    assert SM.urdf_path("mesh") == INSTALL
    assert SM.urdf_path("capsule") == CAPSULES
    assert SM.urdf_path(with_arms=False) == ENV
    assert SM.manifest_path() == MANIFEST
    for p in (INSTALL, CAPSULES, ENV, MANIFEST):
        assert p.is_file()
    with pytest.raises(ValueError):
        SM.urdf_path("spheres")      # the model this one exists to replace


def test_the_truth_module_does_not_touch_the_tool_global():
    before = frames.PEN_LAT
    SM.report()
    SM.bodies()
    SM.arm_collision_capsules()
    assert frames.PEN_LAT == before == 0.0


# ---------------------------------------------------------------------------
# 2.  the generator
# ---------------------------------------------------------------------------
def test_regeneration_is_byte_stable(tmp_path):
    """Twice, and both times equal to what is committed.  No clock, no rng.

    GREEN AGAIN SINCE 2026-09-03, and worth saying why.  This was red from
    2026-09-02: the committed `installation.urdf` and `installation_capsules.
    urdf` differed from what this station writes on 48 and 150 lines, every one
    of them a rotation out of `_rpy_checked`, worst 4.44e-16 — two ULP on pi,
    written by a third environment.  They were left alone rather than churned
    for two ULP.  The 7c housing correction re-wrote both files anyway, so the
    two ULP went with it; the 48/150 lines were re-measured against this
    environment first (they reproduce exactly), so nothing was hidden inside
    the bigger diff.  See docs/SYSTEM_MODEL.md 10.
    """
    gen = _gen()
    gen.OUT_DIR = tmp_path
    # the manifest reads the mesh record the mesh stage writes; copy the
    # committed one across rather than re-vendoring 12 MB in a unit test
    (tmp_path / "meshes").mkdir()
    (tmp_path / "meshes/MESH_SOURCES.json").write_bytes(
        (DIR / "meshes/MESH_SOURCES.json").read_bytes())
    first = {}
    for _ in range(2):
        gen.write_urdfs(tmp_path)
        for name in ("installation.urdf", "installation_capsules.urdf",
                     "environment.urdf", "model_manifest.json"):
            got = (tmp_path / name).read_bytes()
            if name in first:
                assert got == first[name], f"{name} is not deterministic"
            first[name] = got
            assert got == (DIR / name).read_bytes(), \
                f"{name} is stale — re-run scripts/gen_system_model.py urdf"


def test_the_generator_does_not_touch_the_tool_global():
    before = frames.PEN_LAT
    _gen()
    assert frames.PEN_LAT == before == 0.0


def test_the_urdf_numbers_round_trip_exactly():
    """`repr` is the point: `float(text) == value` for every number written.

    A writer that quietly went back to `%.9f` would still pass every geometric
    test at 1e-9 while costing a nanometre at the far row, so the property is
    pinned directly.
    """
    gen = _gen()
    for v in (3.0255333333333336, 1.8034 / 2, np.pi, -0.0, 1e-17, 0.1 + 0.2):
        assert float(gen._fmt(v)) == float(v), v


def test_the_drake_namespace_is_declared():
    """Without it ElementTree emits ns0: and every filter group dies silently."""
    assert b"xmlns:drake=" in INSTALL.read_bytes()
    assert b"xmlns:drake=" in CAPSULES.read_bytes()


# ---------------------------------------------------------------------------
# 3.  the asset vs the package
# ---------------------------------------------------------------------------
def _origin_of(el):
    o = el.find("origin")
    xyz = [0.0, 0.0, 0.0]
    rpy = [0.0, 0.0, 0.0]
    if o is not None:
        xyz = [float(v) for v in o.get("xyz", "0 0 0").split()]
        rpy = [float(v) for v in o.get("rpy", "0 0 0").split()]
    return np.asarray(xyz), np.asarray(rpy)


def _rpy(r, p, y):
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)],
                   [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0],
                   [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx


def _T(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = _rpy(*rpy)
    T[:3, 3] = xyz
    return T


def urdf_link_poses(root, q_by_joint):
    """FK over the URDF's own tree, in plain XML -> {link: 4x4}."""
    T = {"world": np.eye(4)}
    pending = [j for j in root.findall("joint")]
    while pending:
        moved = False
        for j in list(pending):
            parent = j.find("parent").get("link")
            if parent not in T:
                continue
            xyz, rpy = _origin_of(j)
            X = _T(xyz, rpy)
            if j.get("type") == "revolute":
                a = np.asarray([float(v) for v in
                                j.find("axis").get("xyz").split()], float)
                th = float(q_by_joint.get(j.get("name"), 0.0))
                K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]],
                              [-a[1], a[0], 0]])
                R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
                Rot = np.eye(4)
                Rot[:3, :3] = R
                X = X @ Rot
            T[j.find("child").get("link")] = T[parent] @ X
            pending.remove(j)
            moved = True
        assert moved, "the URDF joint tree does not close"
    return T


def test_base_poses_match_fleet_proposed(joints):
    for aid, spec in layout.FLEET_PROPOSED.items():
        j = joints[f"arm{aid}_mount_weld"]
        assert j.get("type") == "fixed"
        assert j.find("parent").get("link") == "world"
        assert j.find("child").get("link") == f"arm{aid}_panda_link0"
        xyz, rpy = _origin_of(j)
        assert np.allclose(_T(xyz, rpy), spec.T_world_base(H_INV), atol=1e-12)


def test_every_arm_hangs(joints):
    for aid in layout.FLEET_PROPOSED:
        xyz, rpy = _origin_of(joints[f"arm{aid}_mount_weld"])
        assert np.allclose(_rpy(*rpy)[:, 2], [0, 0, -1], atol=1e-12)


def test_pen_tip_fk_matches_frames(root):
    """The independent check: URDF tree in XML vs `frames.fk`, no drake.

    The generator writes exact float64 round trips, so the only error left is
    accumulation through two different orderings of the same chain.
    """
    rng = np.random.default_rng(20260901)
    qs = [np.asarray(next(iter(layout.FLEET_PROPOSED.values())).q_seed, float)]
    qs += list(rng.uniform(frames.FR3_MIN + 0.1, frames.FR3_MAX - 0.1,
                           size=(8, 7)))
    off = frames.tool_offset(PEN_EXT_HOLDER, PEN_LAT_HOLDER)   # explicit, never global
    worst = 0.0
    for q in qs:
        qmap = {}
        for aid in layout.FLEET_PROPOSED:
            qmap.update({f"arm{aid}_panda_joint{i + 1}": q[i]
                         for i in range(7)})
        T = urdf_link_poses(root, qmap)
        T_b, _ = fk(q)
        for aid, spec in layout.FLEET_PROPOSED.items():
            Twb = spec.T_world_base(H_INV)
            want = Twb[:3, :3] @ (T_b[:3, 3] + T_b[:3, :3] @ off) + Twb[:3, 3]
            got = T[f"arm{aid}_pen_tip"][:3, 3]
            worst = max(worst, float(np.max(np.abs(got - want))))
    assert worst < 1e-11, worst


def test_every_cage_body_matches_the_truth_module(links, joints):
    for body in SM.bodies():
        xyz, rpy = _origin_of(joints[f"{body.name}_weld"])
        assert np.allclose(xyz, np.asarray(body.centre) / MM, atol=1e-12), \
            body.name
        assert np.allclose(rpy, 0.0), body.name
        size = [float(v) for v in
                links[body.name].find("visual/geometry/box").get("size").split()]
        assert np.allclose(size, np.asarray(body.size) / MM, atol=1e-12), \
            body.name
        has_col = links[body.name].find("collision") is not None
        assert has_col == body.collision, body.name


def test_the_arm_collision_is_never_the_unaudited_spheres(links):
    """The single biggest reason this asset exists."""
    for name, el in links.items():
        for c in el.findall("collision"):
            assert c.find("geometry/sphere") is None, \
                f"{name} carries a collision sphere"
    root_c = ET.parse(CAPSULES).getroot()
    for el in root_c.findall("link"):
        for c in el.findall("collision"):
            assert c.find("geometry/sphere") is None, el.get("name")


def test_the_mesh_variant_uses_the_manufacturers_own_shells(links):
    want = {f"link{i}" for i in range(8)} | {"hand", "finger"}
    seen = set()
    for name, el in links.items():
        if not name.startswith("arm13_"):
            continue
        for c in el.findall("collision"):
            m = c.find("geometry/mesh")
            if m is not None:
                p = Path(m.get("filename"))
                assert p.parts[:2] == ("meshes", "collision"), m.get("filename")
                assert (DIR / p).is_file(), p
                seen.add(p.stem)
    assert seen == want, seen
    # a SET would not notice one finger losing its shell, since both fingers
    # name the same mesh; count the collision elements too
    n = sum(len(el.findall("collision/geometry/mesh"))
            for name, el in links.items() if name.startswith("arm13_"))
    assert n == 11, f"{n} shells on arm 13; want link0..7 + hand + 2 fingers"


def test_every_collision_shell_lands_on_its_own_link():
    """The shell and the manufacturer's own visual must occupy the same box.

    This is the invariant that catches a mis-placed shell, and it caught one:
    `finger.obj` lies entirely on one side (y in [-0.0001, 0.0264]) and is used
    for BOTH fingers, so the right one has to carry the same Rz(180) mirror its
    visual carries or its collision geometry sits 52.8 mm from the finger.

    link0 is exempt and its exemption is the finding in
    `test_the_inverted_base_cable_is_carried_as_a_finding`: the visual has a
    base cable the shell does not.
    """
    trimesh = pytest.importorskip("trimesh")
    root = ET.parse(INSTALL).getroot()
    links = {el.get("name"): el for el in root.findall("link")}
    # drake's glTF import lands a mesh in the frame trimesh gives + Rx(90);
    # measured, not assumed — it is what makes the shells and the visuals
    # coincide for the nine links whose visual origin is identity
    RX90 = _rpy(np.pi / 2, 0, 0)
    worst = 0.0
    for bare in [f"panda_link{i}" for i in range(8)] + [
            "panda_hand", "panda_leftfinger", "panda_rightfinger"]:
        el = links[f"arm13_{bare}"]
        vis = el.find("visual")
        v = np.asarray(trimesh.load(
            DIR / Path(vis.find("geometry/mesh").get("filename")),
            force="mesh").vertices, float)
        v = (_rpy(*_origin_of(vis)[1]) @ RX90 @ v.T).T
        col = el.find("collision")
        c = trimesh.load(DIR / Path(col.find("geometry/mesh").get("filename")),
                         force="mesh")
        cv = (_rpy(*_origin_of(col)[1]) @ np.asarray(c.vertices, float).T).T
        err = float(max(np.abs(cv.min(0) - v.min(0)).max(),
                        np.abs(cv.max(0) - v.max(0)).max()))
        if bare == "panda_link0":
            assert err == pytest.approx(0.2307, abs=5e-4), \
                "link0's only disagreement should be the base cable"
            continue
        worst = max(worst, err)
    # link5's visual overhangs its shell by 5.6 mm; everything else is under 1
    assert worst < 0.006, worst


def test_the_capsule_variant_is_the_audited_set():
    """Every capsule's ENDPOINTS, not just its radius.

    `_add_capsule` writes a centre, an rpy and a length; whether that says
    what `selfcoll.BODY_CAPSULES` says is a question about frame construction
    and it is worth asking directly.  Reconstruct a and b from what was
    written and compare.
    """
    root_c = ET.parse(CAPSULES).getroot()
    ns = "{http://drake.mit.edu}capsule"
    want = {}
    for lnk, a, b, r, _tag in SM.arm_collision_capsules():
        want.setdefault(lnk, []).append((a, b, r))
    n = 0
    worst = 0.0
    for el in root_c.findall("link"):
        if not el.get("name").startswith("arm13_"):
            continue
        bare = el.get("name")[len("arm13_"):]
        got = []
        for c in el.findall("collision"):
            g = c.find(f"geometry/{ns}")
            if g is None:
                continue
            xyz, r_ = _origin_of(c)
            z = _rpy(*r_)[:, 2]
            L = float(g.get("length"))
            got.append((xyz - 0.5 * L * z, xyz + 0.5 * L * z,
                        float(g.get("radius"))))
        assert len(got) == len(want.get(bare, [])), bare
        for (ga, gb, gr), (wa, wb, wr) in zip(got, want.get(bare, [])):
            assert gr == pytest.approx(wr, abs=1e-12)
            # a capsule is symmetric, so either endpoint pairing is correct
            worst = max(worst, min(
                max(np.abs(ga - wa).max(), np.abs(gb - wb).max()),
                max(np.abs(ga - wb).max(), np.abs(gb - wa).max())))
            n += 1
    assert n == len(selfcoll.BODY_CAPSULES), n
    assert worst < 1e-12, worst


def test_the_two_collision_variants_differ_only_where_selfcoll_is_silent():
    """And say where: the capsule set has no row for link8 or either finger."""
    caps = {lnk for lnk, *_ in SM.arm_collision_capsules()}
    assert caps == {f"panda_link{i}" for i in range(8)} | {"panda_hand"}
    root_c = ET.parse(CAPSULES).getroot()
    bare = {}
    for el in root_c.findall("link"):
        n = el.get("name")
        if n.startswith("arm13_panda"):
            bare[n[len("arm13_"):]] = len(el.findall("collision"))
    for lnk in ("panda_link8", "panda_leftfinger", "panda_rightfinger"):
        assert bare[lnk] == 0, f"{lnk} should be bare in the capsule variant"
    # and the mesh variant does clothe the fingers
    root_m = ET.parse(INSTALL).getroot()
    m = {el.get("name"): el for el in root_m.findall("link")}
    for lnk in ("panda_leftfinger", "panda_rightfinger"):
        assert m[f"arm13_{lnk}"].findall("collision"), lnk


def test_the_holder_is_the_inferred_placement_and_says_so(links, joints,
                                                          manifest):
    j = joints["arm13_pen_holder_weld"]
    assert j.find("parent").get("link") == "arm13_panda_hand"
    xyz, rpy = _origin_of(j)
    assert np.allclose(xyz, 0) and np.allclose(rpy, 0)
    want = rig_final.penholder22_collision(PEN_EXT_HOLDER, PEN_LAT_HOLDER, D_HAND_TCP)
    cols = links["arm13_pen_holder"].findall("collision")
    # FOUR since 2026-09-03: the three the housing and cap were fitted to,
    # plus the pencil tail's own (docs/SYSTEM_MODEL.md 7c)
    assert len(cols) == len(want) == 4
    for c, (_, T, (r, L)) in zip(cols, want):
        g = c.find("geometry/cylinder")
        assert float(g.get("radius")) == pytest.approx(r, abs=1e-12)
        assert float(g.get("length")) == pytest.approx(L, abs=1e-12)
        xyz, rpy = _origin_of(c)
        assert np.allclose(xyz, T[:3, 3], atol=1e-12)
        # the AXIS too: "coaxial with the bore" is the whole claim, and a
        # cylinder written with an identity rotation would pass on radius,
        # length and centre alone
        assert np.allclose(_rpy(*rpy), T[:3, :3], atol=1e-12), \
            "the envelope cylinder is not on the bore axis"
    # the red flag must be in the machine-readable manifest, not only in prose
    assert manifest["tool"]["provenance"] == "ASSUMED"
    assert "INFERRED" in manifest["tool"]["red_flag"]
    assert "23.0" in manifest["tool"]["red_flag"]
    assert "45.0" in manifest["tool"]["red_flag"]


def test_the_internal_stack_fills_the_bore_it_measured_exactly():
    """The stack is a CHAIN, and a chain that does not close is not a stack.

    Tail shoulder to the cap's own 16 mm shoulder is one measured span, and
    spring + shim + sleeve must add back to it for every shim in the set —
    otherwise the parts are floating and the preload is a story.
    """
    P = rig_final.PENHOLDER22
    span = P["stack_back_x"] - P["stack_front_x"]
    for sp in (0.0,) + tuple(P["spacers"]):
        rows = {b["name"]: b for b in rig_final.penholder22_stack(sp)}
        chain = sum(abs(rows[n]["x1"] - rows[n]["x0"])
                    for n in ("spring", "sleeve") + (("spacer",) if sp else ()))
        assert chain == pytest.approx(span, abs=1e-12), sp
        # and the spring is really preloaded, never past coil bind
        squeezed = abs(rows["spring"]["x1"] - rows["spring"]["x0"])
        assert P["spring"]["solid"] < squeezed < P["spring"]["free"], sp
    # both shims at once is the one combination that must be refused
    with pytest.raises(ValueError):
        rig_final.penholder22_stack(sum(P["spacers"]))


def test_the_complete_assembly_still_fits_the_hull_the_shell_was_fitted_to():
    """Modelling the internals is only free if they are inside the envelope.

    The 3-cylinder hull was fitted to the housing and cap alone, and the
    fourth cylinder to the pencil tail.  Adding six more bodies is allowed to
    change nothing about collision ONLY while every one of them stays inside
    that hull — for every shim setting, not just the one that ships.
    """
    for sp in (0.0,) + tuple(rig_final.PENHOLDER22["spacers"]):
        assert rig_final.penholder22_internals_escape(sp) == 0.0, sp


# ---------------------------------------------------------------------------
# 3a.  which way round the housing is — docs/SYSTEM_MODEL.md 7c
# ---------------------------------------------------------------------------
def test_the_housing_is_not_mounted_end_for_end():
    """The pen leaves through the CAP, and these are the assembly's numbers.

    Until 2026-09-03 `penholder22_T_hand` pointed the housing's +X AWAY from
    the tip: the 17.00 mm tail land 55.099 mm toward the paper and the cap
    30.001 mm back toward the wrist.  Four things off
    `Natural hold assembly - closed.SLDASM` itself say otherwise, and each of
    them is re-derivable with `scripts/read_solidworks.py asm`:

      * along that assembly's bore the parts run cap 1017.121..1028.558,
        sleeve 1020.121..1060.221, SPRING 1060.121..1098.919, housing tail
        face 1102.221 mm — so the spring is at the end away from the cap and
        pushes the pen assembly toward it.  Mounted the other way round the
        tool has no compliance at all;
      * its pencil spans 1000.121..1174.735, i.e. 17.000 mm of sharpened point
        past the cap's outer face and 72.514 mm of blunt tail past the tail
        face — which is what the assembly's own preview draws;
      * the 17.00 mm land will not pass the 19.05 mm spring and does not touch
        a 7 mm stick: it is the spring's stop, not the pen's;
      * the grip is 25.001 mm from the housing's cap end and 55.099 mm from
        its tail, and docs/FINAL_RIG.md's independent extraction read 25 mm.

    The numbers pinned here are the two the old model had exactly backwards.

    THE BORE IS THE GRIP -> TIP RAY.  It was the TCP -> tip ray while the grip
    sat on the TCP; on 2026-09-04 the photo of the real gripper put the holder
    at the FAR END of the Fat finger plates ("the tip of the finger
    extension"), so the grip is 66.5 mm out along hand x
    (`PENHOLDER22["grip_hand_x"]`) and the two rays are 20 deg apart — 45.00
    for the bore, 64.85 for the TCP ray.  docs/SYSTEM_MODEL.md 7e.
    """
    P = rig_final.PENHOLDER22
    GX = P["grip_hand_x"]
    T_h, T_c, exit_x, reach = rig_final.penholder22_T_hand(
        PEN_EXT_HOLDER, PEN_LAT_HOLDER, D_HAND_TCP)
    u = np.array([PEN_LAT_HOLDER - GX, 0.0, PEN_EXT_HOLDER])
    u = u / np.linalg.norm(u)
    # the SENSE: +X_housing points AT the tip, not away from it
    assert np.allclose(T_h[:3, :3] @ [1, 0, 0], u, atol=1e-12)
    # ...and it is a rotation about the POST axis and nothing else
    assert np.allclose(T_h[:3, :3] @ [0, 0, 1], [0, 1, 0], atol=1e-12)
    grip = np.array([P["post_xy"][0], P["post_xy"][1], P["bore_yz"][1]])
    assert np.allclose(T_h[:3, :3] @ grip + T_h[:3, 3], [GX, 0, D_HAND_TCP],
                       atol=1e-12)
    # ...and the grip is where a 26 mm post's OUTER face is flush with the
    # plate's own far edge, which is the whole of the 2026-09-04 placement
    assert GX == pytest.approx(rig_final.FATFINGER["plate_y"][1]
                               - P["post_side"] / 2, abs=1e-9)
    # THE BORE LEANS 23 DEG AT THE GRIP, AND THAT IS THE HOUSING'S OWN ANGLE
    # (2026-09-07).  The post is a 26 mm square whose four flats are clocked
    # `post_clock` = 23.00 deg about the post axis relative to the bore, so
    # the block sits SQUARE to the hand at this lean and nowhere else — which
    # is what the photograph shows ("the square part of the holder is flush
    # with the metal part").  Measured off the STL: the flats' normals are at
    # -113.00 / -23.00 / +67.00 / +157.00 deg off the housing's +X, and with
    # the bore at 23 deg they land at -180 / -90 / 0 / +90 off hand z.
    assert np.degrees(np.arctan2(u[0], u[2])) == pytest.approx(23.0, abs=1e-3)   # the constants are 7 dp
    assert np.degrees(P["post_clock"]) == pytest.approx(23.0, abs=1e-4)
    # the TCP -> tip ray is a different line, and always will be now
    assert np.degrees(np.arctan2(PEN_LAT_HOLDER, PEN_EXT_HOLDER)) \
        == pytest.approx(61.8551, abs=1e-3)
    # grip -> where the pen leaves: the CAP's outer face, 30.001 mm IN FRONT
    assert exit_x == pytest.approx(0.030001, abs=1e-6)
    assert exit_x == pytest.approx(P["cap_end_x"] - P["post_xy"][0], abs=1e-15)
    # grip -> the housing's own end face, the 25 mm docs/FINAL_RIG.md read
    assert P["thread_x"][1] - P["post_xy"][0] == pytest.approx(0.025001,
                                                               abs=1e-6)
    # grip -> the tail face, which is now BEHIND the grip
    assert P["post_xy"][0] == pytest.approx(0.055099, abs=1e-6)
    # the cap sits in FRONT of the TCP along the bore (it used to be behind)
    assert (T_c[:3, 3] - np.array([0, 0, D_HAND_TCP])) @ u > 0
    # what the fixed tip therefore asks of the graphite.  20.000 mm since
    # 2026-09-07, and it is now an INPUT rather than a consequence: Pete read
    # it off the side view ("it only juts out 3-4 cm max", then "about 2 cm"
    # once the orientation was right) and the tip is derived from it,
    # `frames.PEN_GRAPHITE_HOLDER`.  It was 53.2 mm while the tip came off the
    # blades, and 125.562 at the old 0.110 / 0.110 pair.
    assert (reach - exit_x) == pytest.approx(frames.PEN_GRAPHITE_HOLDER,
                                             abs=1e-6)
    assert frames.PEN_GRAPHITE_HOLDER == 0.020


def test_the_pencil_tail_is_the_assemblys_own_overhang():
    """72.514 mm, MEASURED — and its envelope contains it by construction."""
    P = rig_final.PENHOLDER22
    b = rig_final.penholder22_tail()
    assert b["x0"] == pytest.approx(-0.072514, abs=1e-9)
    assert b["x1"] == P["tail_x"] == 0.0
    assert b["r"] == P["lead_r"]
    x0, x1, r = P["tail_cylinder"]
    assert x0 <= b["x0"] and x1 >= b["x1"] and r > b["r"]
    # and it is the LAST primitive of the hull, which callers rely on
    hull = rig_final.penholder22_hull()
    assert len(hull) == 4 and hull[:3] == tuple(P["env_cylinders"])
    assert hull[3] == P["tail_cylinder"]


def test_the_urdf_tip_is_the_tool_transform_and_nothing_else(root):
    """THE PEN TIP IS AN INPUT TO THE HOLDER PLACEMENT, NEVER AN OUTPUT.

    The URDF's own `pen_tip` link, walked through its joint tree, must land on
    TCP + R @ (PEN_LAT_HOLDER, 0, PEN_EXT_HOLDER) exactly — whatever those two
    numbers currently are — and must not depend on where the housing sits.

    THE PAIR MOVED ON 2026-09-03 and this test moved with it.  It used to pin
    0.110 / 0.110 and call them gate-validated; only the INLINE pen's
    `PEN_EXT = 0.110` ever was (gate B, MZ 0.924).  The holder's pair was
    USER-SPECIFIED — an estimate on 2026-08-25, and now the photo of the real
    gripper: the tip sits ~5 cm below the bottom edge of the Fat blades'
    contact plates.  It has moved twice more since: the grip to the FAR END
    of those plates (2026-09-04) and the bore to the HOUSING's own 23 deg with
    20 mm of graphite past the cap (2026-09-07), which is where it stands —
    `PEN_EXT_HOLDER` 0.0460262, `PEN_LAT_HOLDER` 0.0860369.  See frames.py and
    docs/SYSTEM_MODEL.md 7e.
    """
    qmap = {f"arm{aid}_panda_joint{i + 1}": 0.0
            for aid in layout.FLEET_PROPOSED for i in range(7)}
    T = urdf_link_poses(root, qmap)
    for aid in layout.FLEET_PROPOSED:
        hand = T[f"arm{aid}_panda_hand"]
        want = hand[:3, :3] @ (np.array([0.0, 0.0, D_HAND_TCP])
                               + frames.tool_offset(PEN_EXT_HOLDER,
                                                    PEN_LAT_HOLDER)) \
            + hand[:3, 3]
        got = T[f"arm{aid}_pen_tip"][:3, 3]
        assert np.max(np.abs(got - want)) < 1e-12, aid
    # the tip is an INPUT to the placement, never an output of it
    _, _, _, reach = rig_final.penholder22_T_hand(PEN_EXT_HOLDER, PEN_LAT_HOLDER,
                                                  D_HAND_TCP)
    # `reach` is measured from the GRIP, which is 66.5 mm out along hand x
    # since 2026-09-04 — not from the TCP
    gx = rig_final.PENHOLDER22["grip_hand_x"]
    assert reach == pytest.approx(np.hypot(PEN_LAT_HOLDER - gx,
                                           PEN_EXT_HOLDER), abs=1e-15)
    # THE DERIVATION, in one line: the tip is 30.001 mm of holder plus
    # 35.000 mm of graphite along the bore, from the grip.
    gx = rig_final.PENHOLDER22["grip_hand_x"]
    grip = np.array([gx, 0.0, D_HAND_TCP])
    lean = frames.PEN_LEAN_HOLDER
    u = np.array([np.sin(lean), 0.0, np.cos(lean)])
    exit_x = (rig_final.PENHOLDER22["cap_end_x"]
              - rig_final.PENHOLDER22["post_xy"][0])
    want = grip + (exit_x + frames.PEN_GRAPHITE_HOLDER) * u
    got = np.array([PEN_LAT_HOLDER, 0.0, D_HAND_TCP + PEN_EXT_HOLDER])
    assert np.max(np.abs(got - want)) < 1e-7
    # ...and the "tip 50 mm below the blades" rule it SUPERSEDED is now also
    # CONTRADICTED, which is worth pinning rather than glossing: the derived
    # tip sits 37.2 mm below the plate edge.  The blades' edge was a plausible
    # datum for a tip nobody had measured; the holder is a better one, because
    # the protrusion is what a person can see and adjust.
    plate_bottom = rig_final.FATFINGER["plate_link_z"][1] + 0.0584
    assert D_HAND_TCP + PEN_EXT_HOLDER - plate_bottom == \
        pytest.approx(0.0372, abs=5e-4)
    # and the INLINE pen, the one that IS gate-validated, did not move
    assert frames.PEN_EXT == 0.110


def test_the_urdf_draws_the_stack_and_keeps_it_out_of_collision(links,
                                                                manifest):
    """The whole stack as visual bodies, and not one collision.

    A cylinder that quietly became a collision element would change every
    certified clearance in the repo without changing a single number in it.
    """
    want = rig_final.penholder22_internals(PEN_EXT_HOLDER, PEN_LAT_HOLDER, D_HAND_TCP)
    # spring, sleeve, clutch, buried graphite, the pencil TAIL — plus a shim
    # if one is fitted
    assert len(want) == 5 + bool(rig_final.PENHOLDER22["spacer_fitted"])
    assert [w[0] for w in want][-1] == "graphite_tail"
    vis = links["arm13_pen_holder"].findall("visual")
    cyl = [v for v in vis if v.find("geometry/cylinder") is not None]
    assert len(vis) == 2 + len(want) and len(cyl) == len(want)
    for v, (name, T, (r, L), _) in zip(cyl, want):
        g = v.find("geometry/cylinder")
        assert float(g.get("radius")) == pytest.approx(r, abs=1e-12), name
        assert float(g.get("length")) == pytest.approx(L, abs=1e-12), name
        xyz, rpy = _origin_of(v)
        assert np.allclose(xyz, T[:3, 3], atol=1e-12), name
        assert np.allclose(_rpy(*rpy), T[:3, :3], atol=1e-12), name
    # still exactly the four envelope cylinders, and nothing else
    assert len(links["arm13_pen_holder"].findall("collision")) == 4
    ins = manifest["tool"]["internals"]
    assert ins["provenance"] == "AUDIT" and ins["collision"] is False
    assert ins["envelope_escape_m"] == 0.0
    assert len(ins["bodies"]) == len(want)


def test_the_fingers_close_on_the_socket_floors_not_the_post_ends(joints,
                                                                  manifest):
    """18.0 mm, and it is the post's own arithmetic that says so.

    The mount post is 50 mm long with a 7 mm socket in each end, so the faces
    the fingertips seat on are 50 - 2 x 7 = 36 mm apart.  The 28.5 this used
    to be is the fingertip's BACK face: 57 mm, which is 7 mm wider than the
    post is long and would hold nothing at all.
    """
    gen = _gen()
    P = rig_final.PENHOLDER22
    post_len = P["post_z"][1] - P["post_z"][0]
    socket = 0.007                       # measured: floors at post z 7 and 43
    assert gen.FINGER_FIX == pytest.approx((post_len - 2 * socket) / 2,
                                           abs=1e-12)
    assert manifest["tool"]["finger_half_width_m"] == gen.FINGER_FIX
    for k, s in ((1, +1), (2, -1)):
        xyz, _ = _origin_of(joints[f"arm13_panda_finger_joint{k}"])
        assert xyz[1] == pytest.approx(s * gen.FINGER_FIX, abs=1e-12)


def test_the_holder_meshes_exist_and_stay_within_budget():
    for p in sorted((DIR / "meshes/penholder").glob("*.obj")):
        n = sum(1 for ln in p.read_text().splitlines() if ln.startswith("f "))
        assert 500 < n <= 12000, (p.name, n)
        assert p.stat().st_size < 2_000_000, (p.name, p.stat().st_size)


def test_the_holder_envelope_still_encloses_the_committed_meshes():
    trimesh = pytest.importorskip("trimesh")
    P = rig_final.PENHOLDER22
    T_h, _, _, _ = rig_final.penholder22_T_hand(PEN_EXT_HOLDER, PEN_LAT_HOLDER,
                                                D_HAND_TCP)
    Ti = np.linalg.inv(T_h)
    by, bz = P["bore_yz"]
    worst = -np.inf
    for p in sorted((DIR / "meshes/penholder").glob("*.obj")):
        v = np.asarray(trimesh.load(p, force="mesh").vertices, float)
        vh = (Ti[:3, :3] @ v.T).T + Ti[:3, 3]
        out = np.full(len(vh), np.inf)
        for x0, x1, r in P["env_cylinders"]:
            out = np.minimum(out, np.maximum(
                np.hypot(vh[:, 1] - by, vh[:, 2] - bz) - r,
                np.maximum(x0 - vh[:, 0], vh[:, 0] - x1)))
        worst = max(worst, float(out.max()))
    # 10 um, and the manifest records the measured value — see the note in
    # gen_system_model.decimate_holder for why this is not zero
    assert worst <= 1e-5, worst


# ---------------------------------------------------------------------------
# 3b.  the Fat Franka Finger — docs/SYSTEM_MODEL.md 7d
# ---------------------------------------------------------------------------
FAT_MESH = DIR / "meshes/fatfinger/fatfinger_leftfinger.obj"
FATURDF = DIR / "installation_fatfingers.urdf"


def test_the_fat_finger_mesh_is_the_part_that_arrived():
    """Extents, in the LEFT finger's link frame, and the budget.

    The raw STL is 18.4339 x 90.0003 x 50.000 mm; the placement is a 90-degree
    rotation about the finger's own z, so the link-frame extents come out
    permuted.  Decimation costs 0.0000 mm of it — 8234 faces welds to 8234 and
    the target is 8000.
    """
    trimesh = pytest.importorskip("trimesh")
    m = trimesh.load(FAT_MESH, force="mesh")
    n = sum(1 for ln in FAT_MESH.read_text().splitlines() if ln.startswith("f "))
    assert n == 8000, n
    assert FAT_MESH.stat().st_size < 2_000_000
    assert np.asarray(m.extents) * 1000 == pytest.approx(
        [90.0, 18.4338, 50.0], abs=0.002)
    lo, hi = np.asarray(m.bounds) * 1000
    assert lo == pytest.approx([-10.500, 8.066, 3.842], abs=0.002)
    assert hi == pytest.approx([79.500, 26.500, 53.842], abs=0.002)


def test_the_fat_finger_lands_on_the_fingertips_own_frame():
    """The transform is a proper rotation, and its residual is 0.155 mm.

    Nothing is fitted here.  The part is drawn in the same CAD frame as
    `Franka_Finger_FR3 Fingertip only.SLDPRT`, so the map into the finger is
    fixed by placing the FINGERTIP — and what that costs is measured against
    the manufacturer's own finger, whose two meshes disagree with each other
    by 0.051 mm.
    """
    trimesh = pytest.importorskip("trimesh")
    F = rig_final.FATFINGER
    T = rig_final.fatfinger_T_finger()
    assert np.linalg.det(T[:3, :3]) == pytest.approx(1.0, abs=1e-12)
    assert T[:3, :3].T @ T[:3, :3] == pytest.approx(np.eye(3), abs=1e-12)
    # the right finger's placement is the OTHER foot, and also proper
    Tm = rig_final.fatfinger_T_finger(mirrored=True)
    assert np.linalg.det(Tm[:3, :3]) == pytest.approx(1.0, abs=1e-12)

    vis = trimesh.load(DIR / "meshes/fr3/finger.gltf", force="mesh")
    vis.apply_transform(trimesh.transformations.rotation_matrix(np.pi,
                                                                [1, 0, 0]))
    col = trimesh.load(DIR / "meshes/collision/finger.obj", force="mesh")
    back = max(float(vis.bounds[1][1]), float(col.bounds[1][1]))
    # the foot's outer face is the carriage side, and it lands on the stock
    # finger's own back face
    assert F["foot_link_y"][1] - back == pytest.approx(0.0, abs=1.6e-4)
    # the plate's contact face is the fingertip's back face, 0.1502 mm out
    assert F["plate_offset"] - 0.0105 == pytest.approx(1.502e-4, abs=1e-6)
    # and the fingertip's distal face IS the finger mesh's own tip
    tip = F["plate_z"][1] + F["z_offset"]
    assert tip - float(vis.bounds[1][2]) == pytest.approx(0.0, abs=6e-5)
    # the grip centre falls on the 10-deg assembly's own 103.26 mm
    z_mid = 0.5 * sum(F["plate_link_z"]) + 0.0584
    assert z_mid == pytest.approx(0.10326, abs=2e-5)


def test_the_blades_converge_toward_the_paper_and_the_mirror_cannot_grip():
    """THE MOUNTING SENSE, AGAINST THE PHOTOGRAPH OF THE REAL GRIPPER.

    Pete photographed the mounted gripper on 2026-09-03.  What it shows: each
    blade's mounting FOOT outboard, up against the hand at the carriage; the
    slanted WEB running down and inward from it; the flat contact PLATE
    inboard of the foot; and the two blades forming a V that converges toward
    the paper.  No fingertips, and the bare plates clamping the holder post's
    end faces.

    This pins that the model says the same thing, in `panda_hand` and at the
    joint value the URDF draws — feet 70.867 mm apart up at the carriages,
    plates 55.168 mm apart down at the paper — and, separately, that there was
    never another option: the only other way to bolt a foot to the carriage
    flat is on the foot's INNER face, which runs the web outward and puts the
    contact plate 34.350 mm out instead of 10.650.  Two of those are 68.700 mm
    apart at q = 0, so closing them on a 50 mm post needs a NEGATIVE joint
    value.  The blade cannot grip this holder mounted the other way round.

    (The 10.650 mm is "outboard" only of the STOCK GRIP PLANE, which is a
    different datum from the foot: the blade is 15.850 mm thick from carriage
    face to contact face where the stock finger is 26.4 mm from carriage face
    to grip plane, and that difference is the whole of it.)
    """
    F = rig_final.FATFINGER
    gen = _gen()
    q = gen.FAT_FINGER_FIX
    JZ = 0.0584                       # panda_finger_joint origin, panda_hand z

    # the V: outboard and proximal at the foot, inboard and distal at the plate
    foot_y0, foot_y1 = F["foot_link_y"]
    plate_y0 = F["plate_offset"]
    assert foot_y0 > plate_y0                      # plate INBOARD of the foot
    foot_z = (0.003842, 0.017842)
    plate_z = F["plate_link_z"]
    assert foot_z[1] < plate_z[0]                  # foot PROXIMAL of the plate
    assert 2 * (q + foot_y0) == pytest.approx(0.070867, abs=1e-6)
    assert 2 * (q + plate_y0) == pytest.approx(0.055168, abs=1e-6)
    # ...and in panda_hand the bands sit where the photo puts them
    assert (foot_z[0] + JZ, foot_z[1] + JZ) == pytest.approx(
        (0.062242, 0.076242), abs=1e-6)
    assert (plate_z[0] + JZ, plate_z[1] + JZ) == pytest.approx(
        (0.094242, 0.112242), abs=1e-6)
    # the rib is what actually lands, 2.5839 mm short of the plate
    assert 2 * (q + F["rib_offset"]) == pytest.approx(0.050, abs=1e-9)

    # THE MIRROR: foot's inner face on the carriage instead of its outer one
    carriage = F["foot_link_y"][1]                 # 0.0265, the carriage flat
    mirrored = F["plate_x"][1] - F["foot_x"][1] + carriage
    assert mirrored == pytest.approx(0.034350, abs=1e-6)
    post = (rig_final.PENHOLDER22["post_z"][1]
            - rig_final.PENHOLDER22["post_z"][0])
    assert 2 * mirrored > post                     # already wider than the post
    assert 0.5 * post - mirrored < 0.0             # so q would have to be < 0


def test_the_fat_fingers_box_envelope_still_contains_it():
    """Four boxes, measured off this very file, and nothing escapes them."""
    trimesh = pytest.importorskip("trimesh")
    gen = _gen()
    m = trimesh.load(FAT_MESH, force="mesh")
    boxes = [(tuple(lo), tuple(hi)) for lo, hi in
             json.loads((DIR / "meshes/MESH_SOURCES.json").read_text())
             ["fat_finger"]["collision_boxes_m"]]
    assert len(boxes) == 4
    assert gen._fat_escape(m, boxes) <= 0.0
    # and the committed boxes are the ones the generator measures today
    assert gen._fat_boxes(m) == pytest.approx(boxes, abs=1e-9)


def test_the_fat_finger_is_recorded_with_its_provenance(manifest):
    """Vendored, hashed, and the six grasp hypotheses are in the manifest.

    The 2026-09-03 photo cuts the six to TWO — no fingertip is fitted, so
    every row that seats one is out — and both survivors are under the running
    GUI's 0.0432.  That is the open item, not a contradiction: the GUI's menu
    path falls back to a move-close when a grasp reports failure.
    """
    rec = [f for f in manifest["meshes"]["files"]
           if f["file"].startswith("meshes/fatfinger/")]
    assert len(rec) == 1, rec
    f = rec[0]
    assert f["source"].endswith("Fat Franka Finger v250904.STL")
    assert (DIR / f["file"]).stat().st_size == f["bytes"]
    assert hashlib.sha256((DIR / f["file"]).read_bytes()).hexdigest() \
        == f["sha256"]
    ff = manifest["tool"]["fat_finger"]
    assert ff["locates_post"] is False
    assert ff["collision"]["worst_escape_m"] == 0.0
    assert ff["stl_y_shift_m"] == rig_final.FATFINGER["stl_y_shift"]
    # ONE number off the robot picks one of these, and 0.0432 picks neither of
    # the two the CAD would have predicted
    w = ff["grasp_width_hypotheses_m"]
    assert w == rig_final.fatfinger_widths()
    assert sorted(round(v, 4) for v in w.values()) == \
        [0.0287, 0.0339, 0.0357, 0.036, 0.0497, 0.05]
    assert sum(1 for v in w.values() if v > 0.0432) == 2
    # the two the photo leaves standing, and neither of them clears 0.0432
    bare = [v for k, v in w.items() if "fingertip" not in k and "stock" not in k]
    assert sorted(round(v, 4) for v in bare) == [0.0287, 0.0339]
    assert max(bare) < 0.0432


def test_the_fat_variant_urdf_carries_the_blade_on_both_fingers():
    """One mesh, two fingers — and the right one is the OTHER foot.

    The part is its own mirror image, so `Rz(pi)` about x = mirror_pitch / 2
    puts the far foot on the carriage.  Any other rigid motion that swaps the
    feet is improper, which is exactly why the part has two of them.
    """
    gen = _gen()
    root = ET.parse(FATURDF).getroot()
    lk = {e.get("name"): e for e in root.findall("link")}
    jt = {e.get("name"): e for e in root.findall("joint")}
    F = rig_final.FATFINGER
    for aid in layout.FLEET_PROPOSED:
        for k, s in ((1, +1), (2, -1)):
            xyz, _ = _origin_of(jt[f"arm{aid}_panda_finger_joint{k}"])
            assert xyz[1] == pytest.approx(s * gen.FAT_FINGER_FIX, abs=1e-12)
        for side, right in (("left", False), ("right", True)):
            el = lk[f"arm{aid}_panda_{side}finger"]
            assert [Path(m.get("filename")).name
                    for m in el.findall("visual/geometry/mesh")] \
                == ["fatfinger_leftfinger.obj"]
            assert not el.findall("collision/geometry/mesh")
            assert len(el.findall("collision/geometry/box")) == 4
            xyz, rpy = _origin_of(el.find("visual"))
            want_xyz = (F["mirror_pitch"], 0, 0) if right else (0, 0, 0)
            want_rpy = (0, 0, np.pi) if right else (0, 0, 0)
            assert xyz == pytest.approx(want_xyz, abs=1e-12)
            assert rpy == pytest.approx(want_rpy, abs=1e-12)
    # the joint value is the post's own arithmetic, not a number somebody liked
    post = (rig_final.PENHOLDER22["post_z"][1]
            - rig_final.PENHOLDER22["post_z"][0])
    assert gen.FAT_FINGER_FIX == pytest.approx(post / 2 - F["rib_offset"],
                                               abs=5e-8)


def test_the_blade_escapes_the_capsule_set_and_it_is_only_reported():
    """49.93 mm out of selfcoll, 0.51 mm inside HAND_R — measured, not fixed.

    The stock finger is CONTAINED by the same three hand capsules over the
    same joint range, so this is a statement about the blade and not about the
    capsules.  Nothing here changes a radius: see docs/SYSTEM_MODEL.md 7d.
    """
    trimesh = pytest.importorskip("trimesh")
    from aris_sixarm import coordination
    F = rig_final.FATFINGER
    caps = [(np.asarray(a, float), np.asarray(b, float), r)
            for n, _, _, a, b, r in selfcoll.BODY_CAPSULES if n == "hand"]
    p7 = np.array([0.0, 0.0, -(frames.TCP_D - D_HAND_TCP)])
    p8 = np.array([0.0, 0.0, D_HAND_TCP])

    def seg(P, a, b):
        d = b - a
        t = np.clip(((P - a) @ d) / float(d @ d), 0.0, 1.0)
        return np.linalg.norm(P - (a + t[:, None] * d), axis=1)

    V = np.asarray(trimesh.load(FAT_MESH, force="mesh").vertices, float)
    S = np.asarray(trimesh.load(DIR / "meshes/collision/finger.obj",
                                force="mesh").vertices, float)
    worst_sc = worst_hr = worst_stock = -np.inf
    for q in np.linspace(0.0, 0.04, 41):
        for right in (False, True):
            P = V.copy()
            if right:
                P[:, 0] = F["mirror_pitch"] - P[:, 0]
                P[:, 1] = -P[:, 1]
            P = P + np.array([0.0, -q if right else q, 0.0584])
            out = np.full(len(P), np.inf)
            for a, b, r in caps:
                out = np.minimum(out, seg(P, a, b) - r)
            worst_sc = max(worst_sc, float(out.max()))
            worst_hr = max(worst_hr,
                           float((seg(P, p7, p8) - coordination.HAND_R).max()))
        Q = S + np.array([0.0, q, 0.0584])
        o = np.full(len(Q), np.inf)
        for a, b, r in caps:
            o = np.minimum(o, seg(Q, a, b) - r)
        worst_stock = max(worst_stock, float(o.max()))
    assert worst_sc * 1000 == pytest.approx(49.93, abs=0.15)
    assert worst_stock < 0.0, worst_stock          # the stock finger fits
    assert -0.001 < worst_hr < 0.0, worst_hr      # in HAND_R, but by 0.5 mm
    assert coordination.HAND_R == 0.104            # unchanged, and pinned


def test_the_loader_finds_the_fat_variant_and_refuses_a_capsule_one():
    assert SM.urdf_path(fingers="fat") == FATURDF
    assert FATURDF.is_file()
    assert SM.urdf_path(fingers="stock") == INSTALL
    with pytest.raises(ValueError):
        SM.urdf_path(fingers="thin")
    with pytest.raises(ValueError):
        SM.urdf_path("capsule", fingers="fat")


def test_the_recert_label_is_measured_not_asserted(manifest):
    """Every piece of mount hardware really does escape the modelled keep-out.

    The re-issued layout sheet reads the PLATE as inside on thickness alone
    (12.7 real against 50 modelled).  In plan it escapes by 25.06 mm, because
    the modelled plate is centred on the J1 axis and the real one is offset
    25.15 mm off it — which is why the label is computed from the geometry
    rather than assigned by hand.
    """
    hw = [b for b in SM.bodies() if SM.recert_escape_mm(b) is not None]
    assert len(hw) == 10 * len(layout.FLEET_PROPOSED)
    by = {b["name"]: b for b in manifest["bodies"]}
    for b in hw:
        e = SM.recert_escape_mm(b)
        assert e > 0.0, f"{b.name} is inside the envelope — has mounts.py moved?"
        assert by[b.name]["status"] == "RE-CERT-PENDING", b.name
        assert by[b.name]["escapes_modelled_envelope_mm"] == pytest.approx(
            e, abs=1e-9)
    assert manifest["recert"]["bodies_pending"] == len(hw)
    assert manifest["recert"]["worst_escape_mm"] == pytest.approx(185.55,
                                                                  abs=0.01)
    # AND INDEPENDENTLY, because re-deriving through recert_escape_mm cannot
    # see a bug in recert_escape_mm.  A gusset sits 1.5 m up, where the only
    # envelope box is the 200-wide COLUMN; measuring it against the union's
    # bounding box (226 wide, from the plate below) reported 13 mm short.
    g = next(b for b in SM.bodies() if b.name == "gusset13_10")
    col = SM.modelled_envelope(13)[1]                       # the boom column
    assert col[0][2] > SM.z_ladder()["plate_top"], "boxes[1] must be the column"
    assert g.lo[2] > col[0][2], "the gusset must sit in the column's band only"
    assert g.hi[0] - col[1][0] == pytest.approx(185.55, abs=0.01)
    assert SM.recert_escape_mm(g) == pytest.approx(185.55, abs=0.01)
    assert SM.recert_escape_mm(
        next(b for b in SM.bodies() if b.name == "plate13")) == pytest.approx(
            25.06, abs=0.01)
    # a body that is not mount hardware is not labelled
    assert SM.recert_escape_mm(
        next(b for b in SM.bodies() if b.name == "paper")) is None
    assert not [b for b in manifest["bodies"]
                if b.get("status") == "RE-CERT-PENDING"
                and b["name"] not in {x.name for x in hw}]


def test_the_modelled_envelope_is_read_from_mounts_not_restated():
    """`modelled_envelope` must be `mounts.arm_mount_boxes`, not a copy."""
    for aid, spec in layout.FLEET_PROPOSED.items():
        want = mounts.arm_mount_boxes(spec.mount, spec.xy, spec.yaw, H_INV,
                                      tag=f"mount{aid}")
        got = SM.modelled_envelope(aid)
        assert len(got) == len(want) == 2
        for (lo, hi), w in zip(got, want):
            assert np.allclose(lo, np.asarray(w["lo"]) * MM, atol=1e-9)
            assert np.allclose(hi, np.asarray(w["hi"]) * MM, atol=1e-9)


def test_the_cable_dress_is_visual_only_and_marked_estimated(links, manifest):
    found = [n for n in links if "cable_dress" in n]
    assert len(found) == len(SM.CABLE_DRESS) * len(layout.FLEET_PROPOSED)
    for n in found:
        assert not links[n].findall("collision"), n
        assert links[n].findall("visual"), n
    assert manifest["cable_dress"]["provenance"] == "ASSUMED"
    assert manifest["cable_dress"]["collision"] is False
    assert "ESTIMATED" in manifest["cable_dress"]["note"]


def test_environment_is_the_installation_without_the_arms(links):
    env = {el.get("name") for el in ET.parse(ENV).getroot().findall("link")}
    static = {n for n in links if not n.startswith("arm")}
    assert env == static
    assert not [j for j in ET.parse(ENV).getroot().findall("joint")
                if j.get("type") != "fixed"]


# ---------------------------------------------------------------------------
# 4.  the manifest
# ---------------------------------------------------------------------------
def test_manifest_matches_the_code_it_describes(manifest):
    assert manifest["z_ladder_mm"] == {k: pytest.approx(v, abs=1e-9)
                                       for k, v in SM.z_ladder().items()}
    assert manifest["canvas"]["size_mm"] == pytest.approx(
        [SM.CANVAS_W, SM.CANVAS_L])
    assert manifest["mount_plane_mm"]["value"] == pytest.approx(H_INV * MM)
    assert len(manifest["bodies"]) == len(SM.bodies())
    by_name = {b["name"]: b for b in manifest["bodies"]}
    for b in SM.bodies():
        m = by_name[b.name]
        assert m["provenance"] == b.provenance
        assert m["size_mm"] == pytest.approx(list(b.size), abs=1e-9)
        assert m["lo_mm"] == pytest.approx(list(b.lo), abs=1e-9)
    assert set(manifest["open_questions"]) == set(SM.OPEN_QUESTIONS)
    assert len(manifest["reconciliation"]) == len(SM.reconciliation())
    assert set(manifest["arms"]) == {str(a) for a in layout.FLEET_PROPOSED}


def test_manifest_provenance_mix_adds_up(manifest):
    mix = manifest["provenance_mix"]
    assert sum(v["count"] for v in mix.values()) == len(SM.bodies())
    assert sum(v["percent"] for v in mix.values()) == pytest.approx(100.0,
                                                                    abs=0.3)
    assert mix["DRAWING"]["count"] > mix["ASSUMED"]["count"], \
        "more of this model should come off the drawing than out of my head"


def test_manifest_records_where_every_vendored_file_came_from(manifest):
    files = manifest["meshes"]["files"]
    # 27 textures (3 maps x link0..7 + hand) + 10 gltf + 10 .bin
    # + 10 collision shells + 2 holder parts + the Fat finger (docs 7d)
    assert len(files) == 27 + 10 + 10 + 10 + 2 + 1, len(files)
    for f in files:
        p = DIR / f["file"]
        assert p.is_file(), f["file"]
        assert f["source"], f["file"]
        assert p.stat().st_size == f["bytes"], f["file"]
        # the hash, not just the size: a re-vendored texture or a re-decimated
        # holder that drifted from its record would otherwise pass
        assert hashlib.sha256(p.read_bytes()).hexdigest() == f["sha256"], \
            f"{f['file']} does not match its manifest sha256"


def test_no_gltf_asks_for_a_texture_that_is_not_there():
    """The whole reason the visuals were flat white before."""
    n = 0
    for g in sorted((DIR / "meshes/fr3").glob("*.gltf")):
        d = json.loads(g.read_text())
        assert "KHR_texture_basisu" not in json.dumps(d), g.name
        for im in d.get("images", []):
            assert (g.parent / im["uri"]).is_file(), f"{g.name}: {im['uri']}"
            n += 1
        for buf in d.get("buffers", []):
            assert (g.parent / buf["uri"]).is_file(), f"{g.name}: {buf['uri']}"
    assert n == 27, n      # 3 maps x (link0..7 + hand); finger has none
