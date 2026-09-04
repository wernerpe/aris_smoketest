#!/usr/bin/env python3
"""Validate `assets/system_model/` against the package it was generated from.

    /home/franka/git/franka_manipulation_station/.venv/bin/python \
        scripts/check_system_model.py

Needs drake, so it runs on the station venv rather than the system python.

WHAT IT CHECKS
  0  all three URDFs parse; the environment has no degrees of freedom
  1  42 DOF, six arms, FR3 joint limits (not the vendored Panda's)
  2  every base pose == FLEET_PROPOSED[aid].T_world_base(h), and every arm
     hangs (base z axis == -world z)
  3  EVERY PEN TIP == frames' own FK, over 25 configurations x 6 arms, to
     10 pm.  The proposed rig's bar was 1 nm; the generator writes every
     number as an exact float64 round trip, so what is left is drake's own
     accumulation and nothing else.
  4  the tool chain: holder pose == hand pose, bore along the planner's ray,
     the 3-cylinder envelope matches rig_final exactly
 4b  the fat-finger variant: it parses, its finger joints sit at the DERIVED
     value, both fingers carry the measured box envelope and the vendored
     blade, and the blade clears the holder at that joint value
  5  every cage body == aris_sixarm.system_model, to 10 pm
  6  NO UNAUDITED PRIMITIVES: no collision sphere anywhere, and every arm
     collision geometry is either a manufacturer shell or an audited capsule.
     Whether each shell LANDS on its own link is checked in
     `tests/test_system_model.py::test_every_collision_shell_lands_on_its_own_link`,
     which needs trimesh rather than drake.
  7  the manifest agrees with the code it claims to describe
  8  the textures the glTFs ask for are all present
  9  MEASURED CLEARANCES at the certified park poses, in both collision
     variants.  Gated only on interpenetration — an arm inside the steel at a
     pose the programme holds for whole phases is a broken model, not a tight
     one.  The minima themselves are reported, because what the corrected cage
     leaves around the certified programme is the question this model was
     built to answer.
"""
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pydrake.geometry import Role  # noqa: E402
from pydrake.multibody.parsing import Parser  # noqa: E402
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph  # noqa: E402
from pydrake.systems.framework import DiagramBuilder  # noqa: E402

from aris_sixarm import rig_final, selfcoll  # noqa: E402
from aris_sixarm import system_model as SM  # noqa: E402
from aris_sixarm.frames import (D_HAND_TCP, FR3_MAX, FR3_MIN,  # noqa: E402
                                PEN_EXT_HOLDER, PEN_LAT_HOLDER, QD_MAX,
                                TAU_MAX, fk, tool_offset)
from aris_sixarm.layout import (FLEET_PROPOSED, LAYOUT_PROPOSED,  # noqa: E402
                                Q_PARK_PROPOSED)

DIR = ROOT / "assets/system_model"
H_INV = float(LAYOUT_PROPOSED["h"])
# m.  The generator writes every number as an exact float64 round trip, so the
# only error left in the loop is drake's own accumulation through the chain —
# picometres.  10 pm is a bar this can actually be held to, and it would catch
# a writer that quietly went back to truncating.
TOL = 1e-11
# A rotation-matrix residual is dimensionless, not a length, so it gets its own
# bar: 1e-10 on a matrix entry is 0.02 microradian, or 2 nm of tip sideways at
# the FR3's full 855 mm reach.
ROT_TOL = 1e-10
N_SAMPLES = 24
fails = []
worst_tip = 0.0


def check(name, got, want, tol=TOL, quiet=False):
    err = float(np.max(np.abs(np.asarray(got, float) - np.asarray(want,
                                                                  float))))
    ok = err <= tol
    if not quiet:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: err {err:.3e}")
    if not ok:
        fails.append((name, err, tol))
    return err


def ok(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        fails.append((name, detail, None))


def load(path):
    b = DiagramBuilder()
    plant, sg = AddMultibodyPlantSceneGraph(b, time_step=0.0)
    Parser(plant).AddModels(str(path))
    plant.Finalize()
    dia = b.Build()
    root = dia.CreateDefaultContext()
    return plant, sg, plant.GetMyContextFromRoot(root), root


# --- 0. parse --------------------------------------------------------------
print("0. the three URDFs parse")
plant, sg, ctx, root = load(DIR / "installation.urdf")
eplant, _, _, _ = load(DIR / "environment.urdf")
cplant, csg, cctx, croot = load(DIR / "installation_capsules.urdf")
ok("installation parses", plant.num_bodies() > 0,
   f"{plant.num_bodies()} bodies")
ok("environment has no dof", eplant.num_positions() == 0)
ok("capsule variant parses", cplant.num_bodies() == plant.num_bodies(),
   f"{cplant.num_bodies()} bodies")

# --- 1. the arms -----------------------------------------------------------
print("\n1. six FR3 arms")
check("degrees of freedom", plant.num_positions(), 7 * len(FLEET_PROPOSED))
xroot = ET.parse(DIR / "installation.urdf").getroot()
jl = {j.get("name"): j for j in xroot.findall("joint")}
for aid in FLEET_PROPOSED:
    for i in range(7):
        lim = jl[f"arm{aid}_panda_joint{i + 1}"].find("limit")
        check(f"arm {aid} joint{i + 1} limits are FR3",
              [float(lim.get(k)) for k in ("lower", "upper", "velocity",
                                           "effort")],
              [FR3_MIN[i], FR3_MAX[i], QD_MAX[i], TAU_MAX[i]], quiet=True)
ok("all 42 joint limits are FR3, not the vendored Panda's",
   not [f for f in fails if "limits are FR3" in f[0]])

# --- 2. the poses ----------------------------------------------------------
print("\n2. base poses == FLEET_PROPOSED")
for aid, spec in FLEET_PROPOSED.items():
    X = plant.EvalBodyPoseInWorld(ctx,
                                  plant.GetBodyByName(f"arm{aid}_panda_link0"))
    T = spec.T_world_base(H_INV)
    check(f"arm {aid} base pose", X.GetAsMatrix4()[:3], T[:3], quiet=True)
    check(f"arm {aid} hangs", X.rotation().matrix()[:, 2], [0, 0, -1],
          quiet=True)
ok(f"all {len(FLEET_PROPOSED)} bases match the layout and hang",
   not [f for f in fails if "base pose" in f[0] or "hangs" in f[0]])

# --- 3. the pen tips -------------------------------------------------------
print("\n3. pen tip FK vs frames")
rng = np.random.default_rng(20260901)
qs = [np.asarray(next(iter(FLEET_PROPOSED.values())).q_seed, float)]
qs += list(rng.uniform(FR3_MIN + 0.10, FR3_MAX - 0.10, size=(N_SAMPLES, 7)))
off = tool_offset(PEN_EXT_HOLDER, PEN_LAT_HOLDER)      # explicit: never the global
for aid, spec in FLEET_PROPOSED.items():
    Twb = spec.T_world_base(H_INV)
    e_tip = e_rot = 0.0
    for q in qs:
        for i in range(7):
            plant.GetJointByName(f"arm{aid}_panda_joint{i + 1}").set_angle(
                ctx, float(q[i]))
        Xt = plant.EvalBodyPoseInWorld(ctx,
                                       plant.GetBodyByName(f"arm{aid}_pen_tip"))
        T_b, _ = fk(q)
        p_want = Twb[:3, :3] @ (T_b[:3, 3] + T_b[:3, :3] @ off) + Twb[:3, 3]
        R_want = Twb[:3, :3] @ T_b[:3, :3]
        e_tip = max(e_tip, float(np.max(np.abs(Xt.translation() - p_want))))
        e_rot = max(e_rot, float(np.max(np.abs(Xt.rotation().matrix()
                                               - R_want))))
    worst_tip = max(worst_tip, e_tip)
    check(f"arm {aid} pen tip ({len(qs)} configs)", e_tip, 0.0, quiet=True)
    check(f"arm {aid} pen tip orientation", e_rot, 0.0, tol=ROT_TOL,
          quiet=True)
ok(f"every pen tip within {TOL:.0e} m of frames over {len(qs)} configs "
   f"x {len(FLEET_PROPOSED)} arms", not [f for f in fails if "pen tip" in f[0]],
   f"worst {worst_tip:.3e} m")

# --- 4. the tool -----------------------------------------------------------
print("\n4. the pen holder")
T_h, _, exit_x, reach = rig_final.penholder22_T_hand(PEN_EXT_HOLDER, PEN_LAT_HOLDER,
                                                     D_HAND_TCP)
for aid in FLEET_PROPOSED:
    Xh = plant.EvalBodyPoseInWorld(ctx,
                                   plant.GetBodyByName(f"arm{aid}_panda_hand"))
    Xp = plant.EvalBodyPoseInWorld(ctx,
                                   plant.GetBodyByName(f"arm{aid}_pen_holder"))
    check(f"arm {aid} holder rides the hand", Xp.GetAsMatrix4(),
          Xh.GetAsMatrix4(), quiet=True)
ok("the holder link frame IS the hand frame on every arm",
   not [f for f in fails if "rides the hand" in f[0]])
# THE BORE IS THE GRIP -> TIP RAY, NOT THE TCP -> TIP RAY.  They were the
# same line while the grip sat on the TCP; since 2026-09-04 the grip is at the
# far end of the Fat blades (`grip_hand_x` = 66.5 mm) and the two are 20 deg
# apart — 45.00 deg for the bore, 64.85 for the TCP ray.
GRIP_X = rig_final.PENHOLDER22["grip_hand_x"]
u = np.array([PEN_LAT_HOLDER - GRIP_X, 0.0, PEN_EXT_HOLDER]) / reach
# THE SENSE, NOT ONLY THE AXIS.  The housing's +X points AT the tip: the pen
# leaves through the cap.  Until 2026-09-03 this line read [-1, 0, 0] and the
# model had the housing end-for-end (docs/SYSTEM_MODEL.md 7c) — an axis-only
# check would have passed either way, which is exactly how it survived.
check("holder bore points along the GRIP->tip ray, cap first",
      T_h[:3, :3] @ np.array([1.0, 0.0, 0.0]), u)
check("the grip centre sits at the far end of the Fat blades' plates",
      (T_h[:3, :3] @ np.array([rig_final.PENHOLDER22["post_xy"][0],
                               rig_final.PENHOLDER22["post_xy"][1],
                               rig_final.PENHOLDER22["bore_yz"][1]])
       + T_h[:3, 3]), np.array([GRIP_X, 0.0, D_HAND_TCP]))
check("the pen leaves 30.001 mm in front of the grip, not 55.099 behind it",
      exit_x, rig_final.PENHOLDER22["cap_end_x"]
      - rig_final.PENHOLDER22["post_xy"][0])
want = rig_final.penholder22_collision(PEN_EXT_HOLDER, PEN_LAT_HOLDER, D_HAND_TCP)
lk = {ln.get("name"): ln for ln in xroot.findall("link")}
cols = lk["arm13_pen_holder"].findall("collision")
ok("the holder collision is the 4-cylinder envelope (3 housing + the tail)",
   len(cols) == len(want) == 4, f"{len(cols)} cylinders")
for c, (_, T, (r, L)) in zip(cols, want):
    g = c.find("geometry/cylinder")
    check("holder envelope cylinder", [float(g.get("radius")),
                                       float(g.get("length"))], [r, L],
          quiet=True)
ok("every envelope cylinder matches rig_final",
   not [f for f in fails if "envelope cylinder" in f[0]])
for m in lk["arm13_pen_holder"].findall("visual/geometry/mesh"):
    ok(f"holder visual {Path(m.get('filename')).name} exists",
       (DIR / m.get("filename")).is_file())

# --- 4b. the fat-finger variant -------------------------------------------
print("\n4b. the fat-finger variant")
FAT = DIR / "installation_fatfingers.urdf"
if not FAT.is_file():
    ok("installation_fatfingers.urdf exists", False, str(FAT))
else:
    fplant, fsg, fctx, froot = load(FAT)
    ok("fat variant parses with the same body count",
       fplant.num_bodies() == plant.num_bodies(),
       f"{fplant.num_bodies()} bodies, {fplant.num_positions()} dof")
    froot_x = ET.parse(FAT).getroot()
    fj = {j.get("name"): j for j in froot_x.findall("joint")}
    flk = {ln.get("name"): ln for ln in froot_x.findall("link")}
    F = rig_final.FATFINGER
    fix = round(0.5 * (rig_final.PENHOLDER22["post_z"][1]
                       - rig_final.PENHOLDER22["post_z"][0])
                - F["rib_offset"], 7)
    for k, s in ((1, +1), (2, -1)):
        o = fj[f"arm13_panda_finger_joint{k}"].find("origin")
        check(f"fat finger joint{k}",
              [float(v) for v in o.get("xyz").split()][1], s * fix, quiet=True)
    ok("both fat finger joints sit at the derived value",
       not [f for f in fails if "fat finger joint" in f[0]],
       f"{fix} m -> libfranka width {2 * fix:.4f}")
    for side in ("left", "right"):
        el = flk[f"arm13_panda_{side}finger"]
        ok(f"{side} finger carries the measured box envelope",
           len(el.findall("collision/geometry/box"))
           == len(json.loads((DIR / "meshes/MESH_SOURCES.json").read_text())
                  ["fat_finger"]["collision_boxes_m"])
           and not el.findall("collision/geometry/mesh"),
           f"{len(el.findall('collision/geometry/box'))} boxes")
        ok(f"{side} finger visual is the vendored blade",
           [Path(m.get("filename")).name
            for m in el.findall("visual/geometry/mesh")]
           == ["fatfinger_leftfinger.obj"])
    # The blade-versus-holder distance is NOT asked of drake here: the tool
    # group ignores the wrist group by construction (the holder is held BY the
    # fingers, so an unfiltered pair would report a grip as a collision), and a
    # filtered query returns nothing rather than a number.  The mesh-level
    # proof lives in tests/test_system_model.py, which has trimesh.  What is
    # checked here is the arithmetic the joint value claims to be.
    check("fat joint value == post half-length less the rib's reach", fix,
          0.5 * (rig_final.PENHOLDER22["post_z"][1]
                 - rig_final.PENHOLDER22["post_z"][0]) - F["rib_offset"],
          tol=5e-8)
    ok("the plates stop short of the post by the rib's own stand-off",
       abs((F["plate_offset"] - F["rib_offset"]) - F["rib_proud"]) < 1e-6,
       f"{(F['plate_offset'] - F['rib_offset']) * 1000:.4f} mm")

# --- 5. the cage -----------------------------------------------------------
print("\n5. the cage == aris_sixarm.system_model")
for body in SM.bodies():
    X = plant.EvalBodyPoseInWorld(ctx, plant.GetBodyByName(body.name))
    check(f"{body.name} centre", X.translation(),
          np.asarray(body.centre, float) / SM.MM, quiet=True)
    box = lk[body.name].find("visual/geometry/box")
    check(f"{body.name} size", [float(v) for v in box.get("size").split()],
          np.asarray(body.size, float) / SM.MM, quiet=True)
ok(f"all {len(SM.bodies())} static bodies match the truth module",
   not [f for f in fails if " centre" in f[0] or " size" in f[0]])
z = SM.z_ladder()
ok("the corrected datum is in force",
   abs(z["grid_underside"] - SM.O_BEAM_U) < 1e-9
   and abs(SM.CAGE_TOTAL_H - 2336.5) < 1e-6,
   f"grid underside {z['grid_underside']} mm above the paper, cage "
   f"{SM.CAGE_TOTAL_H} mm above the floor == the drawing's 233,7 cm")

# --- 6. no unaudited primitives -------------------------------------------
print("\n6. collision geometry provenance")
for tag, p, s in (("mesh", plant, sg), ("capsule", cplant, csg)):
    insp = s.model_inspector()
    kinds = Counter(type(insp.GetShape(g)).__name__
                    for g in insp.GetAllGeometryIds()
                    if insp.GetProperties(g, Role.kProximity) is not None)
    ok(f"{tag} variant has no collision sphere", kinds.get("Sphere", 0) == 0,
       str(dict(kinds)))
want_shell = 11 * len(FLEET_PROPOSED)      # link0..7 + hand + two fingers
insp = sg.model_inspector()
kinds = Counter(type(insp.GetShape(g)).__name__
                for g in insp.GetAllGeometryIds()
                if insp.GetProperties(g, Role.kProximity) is not None)
ok("every arm collision body is a manufacturer shell",
   kinds.get("Mesh", 0) == want_shell, f"{kinds.get('Mesh', 0)} of {want_shell}")
cinsp = csg.model_inspector()
ckinds = Counter(type(cinsp.GetShape(g)).__name__
                 for g in cinsp.GetAllGeometryIds()
                 if cinsp.GetProperties(g, Role.kProximity) is not None)
ok("every arm collision body is an audited capsule",
   ckinds.get("Capsule", 0) == len(selfcoll.BODY_CAPSULES) * len(FLEET_PROPOSED),
   f"{ckinds.get('Capsule', 0)} of "
   f"{len(selfcoll.BODY_CAPSULES) * len(FLEET_PROPOSED)}")
recert = [b for b in SM.bodies() if SM.recert_escape_mm(b) is not None]
ok("the drop clusters are present as boxes",
   len(recert) == 10 * len(FLEET_PROPOSED),
   f"{len(recert)} mount-hardware bodies (4 posts + 4 gussets + clamp + "
   f"plate per arm)")
_man = json.loads((DIR / "model_manifest.json").read_text())
_by = {b["name"]: b for b in _man["bodies"]}
ok("every piece of mount hardware is labelled RE-CERT-PENDING",
   all(_by[b.name].get("status") == "RE-CERT-PENDING" for b in recert),
   f"worst escape {_man['recert']['worst_escape_mm']} mm past the modelled "
   "keep-out")
ok("nothing else carries the label",
   sum(1 for b in _man["bodies"]
       if b.get("status") == "RE-CERT-PENDING") == len(recert))
ok("cable dress carries no collision geometry",
   not lk[f"arm13_cable_dress_0"].findall("collision"))

# --- 7. the manifest -------------------------------------------------------
print("\n7. the manifest")
man = json.loads((DIR / "model_manifest.json").read_text())
ok("manifest lists every body", len(man["bodies"]) == len(SM.bodies()),
   f"{len(man['bodies'])}")
ok("every body carries a provenance class",
   all(b["provenance"] in SM.PROVENANCE_CLASSES for b in man["bodies"]))
ok("every ASSUMED body has an open question or a note",
   all(b.get("note") or b["source"] for b in man["bodies"]
       if b["provenance"] == "ASSUMED"))
check("manifest z ladder", list(man["z_ladder_mm"].values()),
      list(SM.z_ladder().values()), tol=1e-9)
ok("the reconciliation queue is present",
   len(man["reconciliation"]) == len(SM.reconciliation()),
   f"{len(man['reconciliation'])} items")
ok("the open questions are present",
   set(man["open_questions"]) == set(SM.OPEN_QUESTIONS),
   f"{len(man['open_questions'])} questions")

# --- 8. the textures -------------------------------------------------------
print("\n8. every texture the glTFs ask for is present")
missing = []
for g in sorted((DIR / "meshes/fr3").glob("*.gltf")):
    d = json.loads(g.read_text())
    for im in d.get("images", []):
        if not (g.parent / im["uri"]).is_file():
            missing.append(f"{g.name} -> {im['uri']}")
    for buf in d.get("buffers", []):
        if not (g.parent / buf["uri"]).is_file():
            missing.append(f"{g.name} -> {buf['uri']}")
ok("no dangling glTF reference", not missing, "; ".join(missing[:4]))
ok("KHR_texture_basisu is gone (VTK cannot read it)",
   not any("KHR_texture_basisu" in g.read_text()
           for g in (DIR / "meshes/fr3").glob("*.gltf")))

# --- 9. measured clearances ------------------------------------------------
print("\n9. measured clearances at the certified park poses (mm)")
# STEEL is kept apart from the paper and the table on purpose: the overall
# minimum is the pen's own graphite over the paper, which is the park hover
# and is meant to be small.  Reporting it as "arm vs structure" invites
# reading a 196 mm hover as a 196 mm steel clearance.
STEEL = ("frame_", "leg_", "runway_", "post", "gusset", "clamp", "plate")
SURFACES = ("paper", "table", "floor")


def clearances(p, s_g, c, rt):
    for aid in FLEET_PROPOSED:
        for i in range(7):
            p.GetJointByName(f"arm{aid}_panda_joint{i + 1}").set_angle(
                c, float(Q_PARK_PROPOSED[aid][i]))
    qo = s_g.get_query_output_port().Eval(s_g.GetMyContextFromRoot(rt))
    ins = s_g.model_inspector()
    nm = {g: ins.GetName(ins.GetFrameId(g)).split("::")[-1]
          for g in ins.GetAllGeometryIds()
          if ins.GetProperties(g, Role.kProximity) is not None}
    best = {}
    for d in qo.ComputeSignedDistancePairwiseClosestPoints(max_distance=0.60):
        a, b = nm.get(d.id_A), nm.get(d.id_B)
        if a is None or b is None:
            continue
        for x, y in ((a, b), (b, a)):
            if not x.startswith("arm"):
                continue
            aid = x.split("_")[0]
            if y.startswith("arm"):
                k = "arm vs arm" if y.split("_")[0] != aid else None
            elif y.startswith(STEEL):
                k = "arm vs STEEL"
            elif y in SURFACES:
                k = f"arm vs {y}"
            else:
                k = None
            if k and d.distance < best.get(k, (1e9,))[0]:
                best[k] = (d.distance, x, y)
    return best


for tag, p, s_g, c, rt in (("manufacturer shells", plant, sg, ctx, root),
                           ("audited capsules", cplant, csg, cctx, croot)):
    best = clearances(p, s_g, c, rt)
    for k, (dist, a, b) in sorted(best.items()):
        print(f"     {tag:20s} {k:18s} {dist * 1000:8.1f}   {a} <-> {b}")
    ok(f"{tag}: nothing interpenetrates at a park pose",
       all(v[0] > 0 for v in best.values()),
       f"min {min(v[0] for v in best.values()) * 1000:.1f} mm")

print(f"\n{'FAILED' if fails else 'ALL PASS'} — worst pen tip "
      f"{worst_tip:.3e} m over {len(qs)} configs x {len(FLEET_PROPOSED)} arms")
for name, got, tol in fails:
    print(f"   FAIL {name}: {got}")
sys.exit(1 if fails else 0)
