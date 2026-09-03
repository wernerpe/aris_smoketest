#!/usr/bin/env python3
"""Load assets/proposed_rig/*.urdf in pydrake and FK-verify against the code.

Run with the station venv (the system python has no drake):
  /home/franka/git/franka_manipulation_station/.venv/bin/python \
      scripts/check_proposed_rig_urdf.py

The proposed rig has no drawing to anchor to — it is a LAYOUT, and the layout
lives in `aris_sixarm/layout.py`.  So every check here anchors to the code
that `ARIS_RIG=proposed` actually activates:

  1. both URDFs parse; installation has 6 arms x 7 revolute DOF (fingers fixed)
  2. every base pose == fleet.FLEET_PROPOSED[aid].T_world_base(h)
  3. every revolute limit == frames.FR3_MIN/MAX/QD_MAX/TAU_MAX (NOT Panda)
  4. drake FK of each arm's PEN TIP == frames FK + the base transform, over
     sampled joint configurations — the cross-check that proves the URDF's
     tool chain is `frames.tool_offset(PEN_EXT, PEN_LAT_HOLDER)`
  4b. the 22-deg CAD holder's visual meshes resolve and ride the hand frame,
     and its INFERRED placement aims the bore along the planner's own ray
  5. paper == the merged canvas; mount plates/booms == mounts.arm_mount_boxes
  6. the ceiling reference plate carries NO collision geometry (the grid is
     not designed yet and must not pretend to be an obstacle)

`frames.PEN_LAT` is a process GLOBAL; nothing here mutates it — the lateral
offset is passed explicitly, everywhere.
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pydrake.geometry import Role                                # noqa: E402
from pydrake.multibody.parsing import Parser                     # noqa: E402
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph  # noqa: E402
from pydrake.systems.framework import DiagramBuilder             # noqa: E402

from aris_sixarm import mounts, rig_final                        # noqa: E402
from aris_sixarm.frames import (D_HAND_TCP, FR3_MAX, FR3_MIN,    # noqa: E402
                                PEN_EXT, PEN_LAT_HOLDER, QD_MAX, TAU_MAX,
                                fk, tool_offset)
from aris_sixarm.layout import FLEET_PROPOSED, LAYOUT_PROPOSED   # noqa: E402
from aris_sixarm.rig_final6 import SHEET_FINAL6                  # noqa: E402

URDF_DIR = ROOT / "assets/proposed_rig"
H_INV = float(LAYOUT_PROPOSED["h"])
TOL = 1e-6         # m; URDF numbers are written at 1e-6 resolution
N_SAMPLES = 24     # random joint configurations per arm
fails = []
worst_tip = 0.0


def check(name, got, want, tol=TOL, quiet=False):
    err = float(np.max(np.abs(np.asarray(got) - np.asarray(want))))
    ok = err <= tol
    if not quiet:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}: err {err:.2e}")
    if not ok:
        fails.append((name, got, want))
    return err


def load(path):
    builder = DiagramBuilder()
    plant, sg = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    Parser(plant).AddModels(str(path))
    plant.Finalize()
    diagram = builder.Build()
    ctx = diagram.CreateDefaultContext()
    return plant, plant.GetMyContextFromRoot(ctx), sg


# ---------------------------------------------------------------------------
print("environment.urdf ...")
plant, ctx, sg = load(URDF_DIR / "environment.urdf")
assert plant.num_positions() == 0
print(f"  PASS  parses; {plant.num_bodies()} bodies, 0 DOF")

print("installation.urdf ...")
plant, ctx, sg = load(URDF_DIR / "installation.urdf")
n = plant.num_positions()
want_dof = 7 * len(FLEET_PROPOSED)
print(f"  {'PASS' if n == want_dof else 'FAIL'}  DOF: {n} (want {want_dof})")
if n != want_dof:
    fails.append(("dof", n, want_dof))

# --- 2. base poses ---------------------------------------------------------
X = {}
for aid, spec in FLEET_PROPOSED.items():
    X[aid] = plant.EvalBodyPoseInWorld(
        ctx, plant.GetBodyByName(f"arm{aid}_panda_link0"))
    T = spec.T_world_base(H_INV)
    check(f"arm {aid} base position", X[aid].translation(), T[:3, 3])
    check(f"arm {aid} base rotation", X[aid].rotation().matrix(), T[:3, :3])
    check(f"arm {aid} z-axis straight down (inverted)",
          X[aid].rotation().matrix()[:, 2], [0, 0, -1])
check("every base at the layout height",
      [X[a].translation()[2] for a in X], [H_INV] * len(X))
xs = sorted({round(float(X[a].translation()[0]), 6) for a in X})
ys = sorted({round(float(X[a].translation()[1]), 6) for a in X})
print(f"  INFO  2 x 3 grid: x = {xs}, y = {ys}")
if len(xs) != 2 or len(ys) != 3:
    fails.append(("grid shape", (len(xs), len(ys)), (2, 3)))

# --- 3. FR3, not Panda -----------------------------------------------------
# the clone drops the ROS <transmission> blocks (the final-rig generator does
# too), so drake builds no JointActuators and the effort limit is only in the
# XML — read it there rather than not check it at all.
root = ET.parse(URDF_DIR / "installation.urdf").getroot()
xml_joints = {j.get("name"): j for j in root.findall("joint")}
for aid in FLEET_PROPOSED:
    lo, hi, vel, eff = [], [], [], []
    for i in range(7):
        j = plant.GetJointByName(f"arm{aid}_panda_joint{i + 1}")
        lo.append(j.position_lower_limit())
        hi.append(j.position_upper_limit())
        vel.append(j.velocity_upper_limit())
        eff.append(float(xml_joints[f"arm{aid}_panda_joint{i + 1}"]
                         .find("limit").get("effort")))
    check(f"arm {aid} FR3 position limits", lo + hi,
          list(FR3_MIN) + list(FR3_MAX))
    check(f"arm {aid} FR3 velocity limits", vel, QD_MAX)
    check(f"arm {aid} FR3 torque limits", eff, TAU_MAX)
    assert not any("acceleration" in k for k in
                   xml_joints[f"arm{aid}_panda_joint1"].find("limit").attrib), \
        "the Panda drake:acceleration hint survived the FR3 rewrite"

# --- 4. pen-tip FK cross-check --------------------------------------------
rng = np.random.default_rng(20260825)
qs = [np.asarray(next(iter(FLEET_PROPOSED.values())).q_seed, float)]
qs += list(rng.uniform(FR3_MIN + 0.10, FR3_MAX - 0.10, size=(N_SAMPLES, 7)))
off = tool_offset(PEN_EXT, PEN_LAT_HOLDER)     # explicit: never the global
for aid, spec in FLEET_PROPOSED.items():
    Twb = spec.T_world_base(H_INV)
    e_tip = e_rot = 0.0
    for q in qs:
        for i in range(7):
            plant.GetJointByName(f"arm{aid}_panda_joint{i + 1}").set_angle(
                ctx, q[i])
        Xt = plant.EvalBodyPoseInWorld(
            ctx, plant.GetBodyByName(f"arm{aid}_pen_tip"))
        T_b, _ = fk(q)                          # hand TCP in link0 (frames DH)
        p_want = Twb[:3, :3] @ (T_b[:3, 3] + T_b[:3, :3] @ off) + Twb[:3, 3]
        R_want = Twb[:3, :3] @ T_b[:3, :3]      # tip frame == TCP frame
        e_tip = max(e_tip, float(np.max(np.abs(Xt.translation() - p_want))))
        e_rot = max(e_rot, float(np.max(np.abs(
            Xt.rotation().matrix() - R_want))))
    worst_tip = max(worst_tip, e_tip)
    check(f"arm {aid} pen_tip world position vs frames ({len(qs)} configs)",
          e_tip, 0.0)
    check(f"arm {aid} pen_tip orientation vs frames", e_rot, 0.0)

# --- 4b. the CAD pen holder ------------------------------------------------
# The meshes are baked in the panda_hand frame, so drake's own pose for the
# pen_holder body must BE the hand's, and every visual file must resolve.
T_h, _, exit_x, reach = rig_final.penholder22_T_hand(PEN_EXT, PEN_LAT_HOLDER,
                                                     D_HAND_TCP)
for mesh in rig_final.PENHOLDER22["visual_meshes"]:
    ok = (URDF_DIR / mesh).is_file()
    print(f"  {'PASS' if ok else 'FAIL'}  holder visual {mesh} resolves")
    if not ok:
        fails.append((mesh, "missing", "present"))
for aid in FLEET_PROPOSED:
    Xh = plant.EvalBodyPoseInWorld(
        ctx, plant.GetBodyByName(f"arm{aid}_panda_hand"))
    Xp = plant.EvalBodyPoseInWorld(
        ctx, plant.GetBodyByName(f"arm{aid}_pen_holder"))
    check(f"arm {aid} pen_holder frame == panda_hand (meshes are baked)",
          Xp.GetAsMatrix4(), Xh.GetAsMatrix4(), quiet=True)
print(f"  PASS  the holder rides the hand frame on all "
      f"{len(FLEET_PROPOSED)} arms" if not fails else "  see failures above")
# the inferred placement must aim the bore at the planning tip
lean = np.degrees(np.arctan2(PEN_LAT_HOLDER, PEN_EXT))
u = np.array([PEN_LAT_HOLDER, 0.0, PEN_EXT])
u = u / np.linalg.norm(u)
# the SENSE as well as the axis: the housing's +X points AT the tip, because
# the pen leaves through the cap.  Until 2026-09-03 this read [-1, 0, 0] and
# the housing was mounted end-for-end (docs/SYSTEM_MODEL.md 7c).
bore = T_h[:3, :3] @ np.array([1.0, 0.0, 0.0])
check("holder bore points along the planner's TCP->tip ray, cap first", bore, u)
print(f"  INFO  planner lean {lean:.2f} deg / reach {reach * 1000:.1f} mm; "
      f"housing clocking {np.degrees(rig_final.PENHOLDER22['post_clock']):.2f}"
      f" deg, grip->exit {exit_x * 1000:.1f} mm (the cap's outer face; "
      f"{rig_final.PENHOLDER22['post_xy'][0] * 1000:.1f} mm of barrel behind "
      f"the grip)  =>  {(reach - exit_x) * 1000:.1f} mm of graphite past the "
      f"cap, {lean - np.degrees(rig_final.PENHOLDER22['post_clock']):.2f} deg "
      f"owed by the fingertip cradle (FLAGGED, see rig_final.PENHOLDER22)")

# --- 5. static geometry ----------------------------------------------------
W, H = SHEET_FINAL6
t = rig_final.PAPER_THICK_CM / 100.0
check("paper link welded at the canvas origin",
      plant.EvalBodyPoseInWorld(ctx, plant.GetBodyByName("paper")
                                ).translation(), [0, 0, 0])
links = {lk.get("name"): lk for lk in root.findall("link")}
pc = links["paper"].find("collision")
check("paper == the merged canvas web",
      [float(v) for v in pc.find("origin").get("xyz").split()]
      + [float(v) for v in pc.find("geometry/box").get("size").split()],
      [W / 2, H / 2, -t / 2, W, H, t])

for aid, spec in FLEET_PROPOSED.items():
    for b in mounts.arm_mount_boxes(spec.mount, spec.xy, spec.yaw, H_INV,
                                    tag=f"mount{aid}"):
        col = links[b["name"]].find("collision")
        xyz = np.array([float(v) for v in col.find("origin").get("xyz").split()])
        size = np.array([float(v) for v in
                         col.find("geometry/box").get("size").split()])
        lo, hi = np.asarray(b["lo"]), np.asarray(b["hi"])
        check(f"{b['name']} collision box", np.concatenate([xyz, size]),
              np.concatenate([(lo + hi) / 2, hi - lo]), tol=2e-6, quiet=True)
print(f"  PASS  {2 * len(FLEET_PROPOSED)} mount boxes == mounts.arm_mount_boxes"
      if not fails else "  see failures above")

# --- 6. the ceiling grid is a LEVEL, not a structure -----------------------
ceil = links["ceiling_grid_ref"]
ok = ceil.find("collision") is None and ceil.find("visual") is not None
print(f"  {'PASS' if ok else 'FAIL'}  ceiling_grid_ref is visual-only "
      f"(the grid's steel is not designed; nothing was gated against it)")
if not ok:
    fails.append(("ceiling_grid_ref collision", "present", "absent"))

# --- 7. the own-mount exclusion is the planning model's ---------------------
# `mounts.obstacles_for` gives an arm every OTHER arm's hardware and never its
# own.  The URDF has to say the same thing or the two models disagree the
# first time anyone simulates this.  Probed on link1, not link0: link0 is
# WELDED to the world, so drake filters it against every world-welded box on
# its own and would hide the question.
insp = sg.model_inspector()
ids = {b: insp.GetGeometries(plant.GetBodyFrameIdOrThrow(
    plant.GetBodyByName(b).index()), Role.kProximity)
    for b in [f"arm{a}_panda_link1" for a in FLEET_PROPOSED]
    + [f"mount{a}_boom" for a in FLEET_PROPOSED]}
aids = list(FLEET_PROPOSED)
own = [insp.CollisionFiltered(g, h) for a in aids
       for g in ids[f"arm{a}_panda_link1"] for h in ids[f"mount{a}_boom"]]
other = [insp.CollisionFiltered(g, h)
         for a, b in zip(aids, aids[1:] + aids[:1])
         for g in ids[f"arm{a}_panda_link1"] for h in ids[f"mount{b}_boom"]]
ok = own and all(own) and other and not any(other)
print(f"  {'PASS' if ok else 'FAIL'}  own boom filtered out of every arm "
      f"({sum(own)}/{len(own)} pairs), neighbours' booms kept "
      f"({len(other) - sum(other)}/{len(other)} pairs)")
if not ok:
    fails.append(("own-mount collision filter", (sum(own), sum(other)),
                  (len(own), 0)))

print(f"\nworst pen_tip world error over {len(qs)} configs x "
      f"{len(FLEET_PROPOSED)} arms: {worst_tip:.3e} m")
print(f"{'ALL CHECKS PASS' if not fails else f'{len(fails)} FAILURES'}")
sys.exit(1 if fails else 0)
