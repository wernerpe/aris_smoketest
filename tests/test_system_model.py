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
import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from aris_sixarm import frames, layout, mounts, rig_final, selfcoll
from aris_sixarm import system_model as SM
from aris_sixarm.frames import D_HAND_TCP, PEN_EXT, PEN_LAT_HOLDER, fk

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
    """718.6 against the original's 736.9 — the code datum asks for 1435.0."""
    z = SM.z_ladder()
    assert z["post_length"] == pytest.approx(718.60, abs=1e-6)
    assert SM.O_POST_L == pytest.approx(736.9, abs=0.05)
    assert SM.POST_L_CODE == pytest.approx(1434.98, abs=1e-6)


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
    """7.34 mm each side — which is why the offset DIRECTION is load-bearing."""
    clear = (SM.POST_GAP - SM.PLATE[0]) / 2
    assert clear == pytest.approx(7.34, abs=0.01)
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
    """Twice, and both times equal to what is committed.  No clock, no rng."""
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
    off = frames.tool_offset(PEN_EXT, PEN_LAT_HOLDER)   # explicit, never global
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


def test_the_capsule_variant_is_the_audited_set():
    root_c = ET.parse(CAPSULES).getroot()
    ns = "{http://drake.mit.edu}capsule"
    got = []
    for el in root_c.findall("link"):
        if not el.get("name").startswith("arm13_"):
            continue
        for c in el.findall("collision"):
            g = c.find(f"geometry/{ns}")
            if g is not None:
                got.append(float(g.get("radius")))
    want = sorted(r for *_, r in
                  ((c[0], c[5]) for c in selfcoll.BODY_CAPSULES))
    assert len(got) == len(selfcoll.BODY_CAPSULES)
    assert sorted(got) == pytest.approx(want, abs=1e-12)


def test_the_holder_is_the_inferred_placement_and_says_so(links, joints,
                                                          manifest):
    j = joints["arm13_pen_holder_weld"]
    assert j.find("parent").get("link") == "arm13_panda_hand"
    xyz, rpy = _origin_of(j)
    assert np.allclose(xyz, 0) and np.allclose(rpy, 0)
    want = rig_final.penholder22_collision(PEN_EXT, PEN_LAT_HOLDER, D_HAND_TCP)
    cols = links["arm13_pen_holder"].findall("collision")
    assert len(cols) == len(want) == 3
    for c, (_, T, (r, L)) in zip(cols, want):
        g = c.find("geometry/cylinder")
        assert float(g.get("radius")) == pytest.approx(r, abs=1e-12)
        assert float(g.get("length")) == pytest.approx(L, abs=1e-12)
        xyz, rpy = _origin_of(c)
        assert np.allclose(xyz, T[:3, 3], atol=1e-12)
    # the red flag must be in the machine-readable manifest, not only in prose
    assert manifest["tool"]["provenance"] == "ASSUMED"
    assert "INFERRED" in manifest["tool"]["red_flag"]
    assert "23.0" in manifest["tool"]["red_flag"]
    assert "45.0" in manifest["tool"]["red_flag"]


def test_the_holder_meshes_exist_and_stay_within_budget():
    for p in sorted((DIR / "meshes/penholder").glob("*.obj")):
        n = sum(1 for ln in p.read_text().splitlines() if ln.startswith("f "))
        assert 500 < n <= 12000, (p.name, n)
        assert p.stat().st_size < 2_000_000, (p.name, p.stat().st_size)


def test_the_holder_envelope_still_encloses_the_committed_meshes():
    trimesh = pytest.importorskip("trimesh")
    P = rig_final.PENHOLDER22
    T_h, _, _, _ = rig_final.penholder22_T_hand(PEN_EXT, PEN_LAT_HOLDER,
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
    # + 10 collision shells + 2 holder parts
    assert len(files) == 27 + 10 + 10 + 10 + 2, len(files)
    for f in files:
        assert (DIR / f["file"]).is_file(), f["file"]
        assert len(f["sha256"]) == 64
        assert f["source"], f["file"]
        assert (DIR / f["file"]).stat().st_size == f["bytes"], f["file"]


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
