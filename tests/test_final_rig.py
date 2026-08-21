"""The FINAL RIG: geometry, obstacles, tool, and the legacy regression.

Everything here is about the 3-arm installation extracted from the
authoritative drawing (docs/FINAL_RIG.md, aris_sixarm/rig_final.py) — and
about the six-arm layout STAYING exactly what it was, because every
historical number in this repo was earned on it.

Run: pytest tests/test_final_rig.py   (~10 s; the pydrake load test lives in
scripts/check_final_rig_urdf.py, station venv, because the system python has
no drake.)
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from aris_sixarm import coordination, rig_final, scene_check, validate  # noqa: E402
from aris_sixarm.fleet import (FLEET, FLEET_FINAL, FLEET_SIXARM,  # noqa: E402
                               SHEET, SHEET_FINAL, SHEET_SIXARM, sheet_for)
from aris_sixarm.frames import PEN_EXT, Q_READY_INV  # noqa: E402
from aris_sixarm.stroke_api import plan_stroke  # noqa: E402

ROOT = Path(__file__).parents[1]


# ==========================================================================
# the registry
# ==========================================================================
def test_the_active_rig_is_the_final_rig():
    # `FLEET` is a COPY of the registry, not the registry: `fleet.activate`
    # mutates it in place (it is the object every `from .fleet import FLEET`
    # holds), so the four registries have to survive being switched between.
    assert FLEET == FLEET_FINAL and FLEET is not FLEET_FINAL
    assert SHEET == SHEET_FINAL == (1.8034, 1.700)
    assert sorted(FLEET) == [2, 13, 31]                  # the drawing's ids
    assert FLEET[13].mount == "floor" and FLEET[31].mount == "inv" \
        and FLEET[2].mount == "wall"
    assert all(s.rig == "final" and s.active for s in FLEET.values())


def test_final_base_poses_match_the_drawing():
    """One dimension-anchored number per arm, in the drawing's own cm."""
    T13 = FLEET[13].T_world_base()
    T31 = FLEET[31].T_world_base()
    T2 = FLEET[2].T_world_base()
    # arm 13: plate top 64.738 above paper 63.668
    assert abs(T13[2, 3] - (64.738 - 63.668) / 100) < 1e-12
    # arm 31: mounting plane 155.868, paper 63.668 => 92.20 cm, axis DOWN
    assert abs(T31[2, 3] - 0.92200) < 1e-12
    assert np.allclose(T31[:3, 2], [0, 0, -1])
    # arm 2: plate face 182.230 from wall, paper corner at 21.246 => 160.984
    assert abs(T2[0, 3] - 1.60984) < 1e-12
    assert np.allclose(T2[:3, 2], [-1, 0, 0])            # J1 horizontal
    # h_inv MUST NOT move a final-rig base (it is the legacy knob)
    assert np.allclose(FLEET[31].T_world_base(0.5), T31)


def test_legacy_sixarm_layout_is_bit_identical():
    """The regression: FLEET_SIXARM is the layout every published number
    used, transform for transform."""
    assert sorted(FLEET_SIXARM) == [2, 13, 17, 31, 71, 97]
    assert SHEET_SIXARM == (3.607, 1.961)
    want_xy = {13: (-0.1118, 1.0008), 17: (3.7186, 1.0008),
               31: (1.3030, 1.6307), 2: (1.3030, 0.3505),
               71: (2.3038, 1.6307), 97: (2.3038, 0.3505)}
    want_active = {13: True, 17: True, 31: True, 2: False, 71: False, 97: True}
    for a, s in FLEET_SIXARM.items():
        assert s.rig == "sixarm" and s.xy == want_xy[a]
        assert s.active is want_active[a]
        assert s.static_obstacles() == []                # legacy: proxies only
    # floor seat: z = one base plate; inverted seat: roty(pi) at h_inv
    T = FLEET_SIXARM[13].T_world_base()
    assert np.allclose(T[:3, 3], [-0.1118, 1.0008, 0.0127])
    T = FLEET_SIXARM[31].T_world_base(0.924)
    assert np.allclose(T[:3, 3], [1.3030, 1.6307, 0.924])
    assert np.allclose(T[:3, :3], np.diag([-1.0, 1.0, -1.0]))
    assert sheet_for(FLEET_SIXARM[13]) == SHEET_SIXARM
    assert sheet_for(FLEET[13]) == SHEET_FINAL


# ==========================================================================
# the frame obstacle model
# ==========================================================================
def test_frame_boxes_are_well_formed_and_exclude_own_mounts():
    all_boxes = rig_final.frame_boxes_canvas(zmin=-10)
    plan_boxes = rig_final.frame_boxes_canvas()
    assert len(all_boxes) == 35
    assert all(np.all(np.asarray(b["hi"]) >= np.asarray(b["lo"]))
               for b in all_boxes)
    # planning drops exactly the below-paper structure
    dropped = {b["name"] for b in all_boxes} - {b["name"] for b in plan_boxes}
    assert dropped == {"table_block"}
    # each arm is excused from ITS OWN mount hardware, nobody else's
    for aid, key in ((13, "up"), (31, "down"), (2, "side")):
        mine = {b["name"] for b in plan_boxes} \
            - {b["name"] for b in FLEET[aid].static_obstacles()}
        assert mine == {b["name"] for b in plan_boxes
                        if b["tag"] == f"mount:{key}"}, (aid, mine)


def test_segment_box_clearance_is_exact():
    """The ternary search against a dense brute force, on seeded segments."""
    rng = np.random.default_rng(7)
    boxes = rig_final.frame_boxes_canvas()
    A = rng.uniform([-0.3, -0.3, 0.0], [2.1, 2.0, 1.8], size=(60, 3))
    B = A + rng.uniform(-0.5, 0.5, size=(60, 3))
    got = rig_final.segment_box_clearance(A, B, boxes)
    ts = np.linspace(0, 1, 2001)[:, None]
    for i in range(len(A)):
        pts = A[i] * (1 - ts) + B[i] * ts
        brute = rig_final.box_clearance(pts, boxes).min()
        step = np.linalg.norm(B[i] - A[i]) / 2000 / 2   # Lipschitz residual
        assert got[i] <= brute + 1e-9, i                # never optimistic
        assert got[i] >= brute - step - 1e-9, i         # and actually exact


def test_frame_obstacle_blocks_a_known_colliding_pose():
    """A real IK solution for arm 31 whose chain enters the feed-roll/boom
    boxes (found by sweeping the left paper edge) must be refused by the
    independent pose validator with a frame_keepout violation — and by the
    planner's own clearance chain."""
    q = np.array([-0.7117, 1.3805, -2.7614, -2.7747, -2.7640, 1.7219, -1.7795])
    spec = FLEET[31]
    rep = validate.check_pose(q, spec, pen_ext=0.110)
    kinds = [v["kind"] for v in rep["violations"]]
    assert "frame_keepout" in kinds, kinds
    # the same pose passes with the obstacles taken away (it is the frame
    # that kills it, not the joint geometry) — via the legacy twin
    legacy_rep = validate.check_pose(q, FLEET_SIXARM[31], h_inv=0.922,
                                     pen_ext=0.110)
    assert "frame_keepout" not in [v["kind"] for v in legacy_rep["violations"]]


def test_planner_certifies_on_all_three_final_arms():
    """One comfortable stroke per arm, planned end to end with the frame
    active, independently validated — including the wall mount, which never
    existed before this rig."""
    cases = {13: [[0.75, 0.32], [1.02, 0.32]],
             31: [[0.35, 1.00], [0.60, 1.05]],
             2:  [[1.40, 1.10], [1.55, 1.25]]}
    for aid, pts in cases.items():
        r = plan_stroke(np.asarray(pts, float), FLEET[aid])
        assert r["status"] == "ok", (aid, r["status"], r.get("reason"))
        w = r["validation"]["worst"]
        assert w["min_frame_clearance"] >= rig_final.STATIC_MARGIN, aid


# ==========================================================================
# the tool
# ==========================================================================
def test_pen_above_surface_gate_with_the_real_rig():
    """The historic bug: park poses put pens 16-113 mm below the paper.

    On the final rig this bit AGAIN, immediately: the LEGACY inverted ready
    pose at this rig's h = 0.922 has the tip 6.6 mm below the paper and the
    elbow inside the boom margin — so each final arm carries its own ready
    pose, and all three must pass the pose validator with the default pen."""
    for aid, spec in FLEET.items():
        rep = validate.check_pose(spec.q_seed, spec, pen_ext=PEN_EXT)
        assert not rep["violations"], (aid, rep["violations"])
    # the legacy pose is refused, and for the right reasons
    bad = validate.check_pose(Q_READY_INV, FLEET[31], pen_ext=PEN_EXT)
    kinds = {v["kind"] for v in bad["violations"]}
    assert scene_check.PEN_PAPER in kinds and "frame_keepout" in kinds, kinds
    # and a 300 mm pen dips even the final ready pose: the gate must SAY so
    worse = validate.check_pose(FLEET[31].q_seed, FLEET[31], pen_ext=0.300)
    assert any(v["kind"] == scene_check.PEN_PAPER
               for v in worse["violations"])


def test_capsule_models_agree_between_modules():
    """The conductor, the veto and the planner restate the same envelope on
    purpose; these pins are what 'restate' means."""
    assert scene_check.RADII == coordination.CAPSULES
    assert scene_check.RADII_FINAL[:-1] == coordination.CAPSULES[:-1]
    assert scene_check.RADII_FINAL[-1] == (8, 9, rig_final.PEN_R_FINAL)
    # static capsules = the conductor's chain minus the bolted base column
    assert rig_final.STATIC_CAPSULES == \
        tuple(c[:2] + (c[2],) for c in scene_check.RADII_FINAL[1:])
    # per-rig pen capsule radius in the conductor
    q = np.asarray(FLEET[31].q_seed, float)[None, :]
    pf = coordination.ArmPath(31, q, 0.02, spec=FLEET[31])
    pl = coordination.ArmPath(31, q, 0.02, spec=FLEET_SIXARM[31])
    assert pf.r[-1] == np.float32(rig_final.PEN_R_FINAL)
    assert pl.r[-1] == np.float32(coordination.PEN_R)


# ==========================================================================
# the generated URDF stays the geometry source's shadow
# ==========================================================================
def test_installation_urdf_matches_rig_final():
    urdf = ROOT / "assets/final_rig/installation.urdf"
    assert urdf.exists(), "run scripts/gen_final_rig_urdf.py"
    root = ET.parse(urdf).getroot()
    # every frame box, center and size, to the writer's 1e-6
    boxes = {f"frame_{b['name']}": b
             for b in rig_final.frame_boxes_canvas(zmin=-10)}
    seen = set()
    for link in root.findall("link"):
        name = link.get("name")
        if name not in boxes:
            continue
        seen.add(name)
        b = boxes[name]
        col = link.find("collision")
        xyz = np.array([float(v) for v in
                        col.find("origin").get("xyz").split()])
        size = np.array([float(v) for v in
                         col.find("geometry/box").get("size").split()])
        lo, hi = np.asarray(b["lo"]), np.asarray(b["hi"])
        assert np.allclose(xyz, (lo + hi) / 2, atol=2e-6), name
        assert np.allclose(size, hi - lo, atol=2e-6), name
    assert seen == set(boxes), set(boxes) - seen
    # the three mount welds
    for key, aid in rig_final.ARM_IDS.items():
        j = [x for x in root.findall("joint")
             if x.get("name") == f"arm{aid}_mount_weld"]
        assert len(j) == 1, aid
        xyz = np.array([float(v) for v in
                        j[0].find("origin").get("xyz").split()])
        p, _ = rig_final.arm_base_canvas(key)
        assert np.allclose(xyz, p, atol=2e-6), aid
    # one pen holder per arm, cylinder envelope as configured
    holders = [x for x in root.findall("link")
               if x.get("name", "").endswith("pen_holder")]
    assert len(holders) == 3
    cc = rig_final.TOOL["collision"]
    for h in holders:
        cyl = h.find("collision/geometry/cylinder")
        assert abs(float(cyl.get("radius")) - cc["radius"]) < 1e-9
        assert abs(float(cyl.get("length")) - (cc["z1"] - cc["z0"])) < 1e-9
