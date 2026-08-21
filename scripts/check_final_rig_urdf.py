#!/usr/bin/env python3
"""Load assets/final_rig/*.urdf in pydrake and FK-verify against the drawing.

Run with the station venv:
  /home/franka/git/franka_manipulation_station/.venv/bin/python \
      scripts/check_final_rig_urdf.py

Checks (each anchored to a DIMENSION IN THE DRAWING, cm, via rig_final):
  1. both URDFs parse; installation has 3 arms x 7 revolute DOF (fingers fixed)
  2. arm 13 link0 height above paper = plate top 64.738 - paper 63.668 = 1.07 cm
  3. arm 31 link0 height above paper = 155.868 - 63.668 = 92.20 cm, z-axis DOWN
  4. arm  2 link0: x = 182.230 - 21.246 = 160.984 cm from the paper corner,
     base z-axis = world -X (J1 axis horizontal)
  5. arm 31 <-> arm 2 base origin distance = |(136.493, 0.045, -14.600)| cm
     (both ends dimension-anchored: 45,7 / 55,8 / 34,9 / 91,6 chains)
  6. drake FK of each arm's hand-TCP == frames.fk in that arm's base frame
     (cross-validates the vendored URDF against the project's DH model)
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pydrake.multibody.parsing import Parser                     # noqa: E402
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph  # noqa: E402
from pydrake.systems.framework import DiagramBuilder             # noqa: E402

from aris_sixarm import rig_final                                # noqa: E402
from aris_sixarm.frames import fk, TCP_D, Q_READY_FLOOR          # noqa: E402

TOL = 1e-6         # m; URDF numbers are written at 1e-6 resolution
fails = []


def check(name, got, want, tol=TOL):
    err = float(np.max(np.abs(np.asarray(got) - np.asarray(want))))
    ok = err <= tol
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: err {err:.2e}")
    if not ok:
        fails.append((name, got, want))


def load(path):
    builder = DiagramBuilder()
    plant, sg = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    Parser(plant).AddModels(str(path))
    plant.Finalize()
    diagram = builder.Build()
    ctx = diagram.CreateDefaultContext()
    return plant, plant.GetMyContextFromRoot(ctx)


print("environment.urdf ...")
plant, ctx = load(ROOT / "assets/final_rig/environment.urdf")
assert plant.num_positions() == 0
print(f"  PASS  parses; {plant.num_bodies()} bodies, 0 DOF")

print("installation.urdf ...")
plant, ctx = load(ROOT / "assets/final_rig/installation.urdf")
n = plant.num_positions()
print(f"  {'PASS' if n == 21 else 'FAIL'}  DOF: {n} (want 21)")
if n != 21:
    fails.append(("dof", n, 21))

X = {}
for key, aid in rig_final.ARM_IDS.items():
    body = plant.GetBodyByName(f"arm{aid}_panda_link0")
    X[aid] = plant.EvalBodyPoseInWorld(ctx, body)
    p_want, R_want = rig_final.arm_base_canvas(key)
    check(f"arm {aid} base position", X[aid].translation(), p_want)
    check(f"arm {aid} base rotation", X[aid].rotation().matrix(), R_want)

# drawing-anchored spot checks (independent arithmetic, not via arm_base_canvas)
check("arm 13 base height above paper [1.07 cm]",
      X[13].translation()[2], (64.738 - 63.668) / 100.0)
check("arm 31 base height above paper [92.20 cm]",
      X[31].translation()[2], (155.868 - 63.668) / 100.0)
check("arm 31 z-axis straight down", X[31].rotation().matrix()[:, 2], [0, 0, -1])
check("arm 2 base x from paper corner [160.984 cm]",
      X[2].translation()[0], (182.230 - 21.246) / 100.0)
check("arm 2 z-axis = world -X (J1 horizontal)",
      X[2].rotation().matrix()[:, 2], [-1, 0, 0])
d_drawing = np.linalg.norm([182.230 - 45.737, 152.471 - 152.426,
                            141.268 - 155.868]) / 100.0
check("arm 31 <-> arm 2 base distance [137.27 cm]",
      np.linalg.norm(X[31].translation() - X[2].translation()), d_drawing)

# 6. drake FK vs frames.fk at the ready pose, per arm, in the BASE frame
q = Q_READY_FLOOR
T_frames, _ = fk(q)                       # hand TCP in link0 (frames.py DH)
for aid in rig_final.ARM_IDS.values():
    for i in range(7):
        j = plant.GetJointByName(f"arm{aid}_panda_joint{i + 1}")
        j.set_angle(ctx, q[i])
    Xh = plant.EvalBodyPoseInWorld(ctx, plant.GetBodyByName(f"arm{aid}_panda_hand"))
    T_drake = X[aid].inverse().multiply(Xh)            # hand in base frame
    # panda_hand frame: TCP is +0.1034 along hand z; hand == flange + Rz(-pi/4)
    p_tcp_drake = (T_drake.translation()
                   + T_drake.rotation().matrix()[:, 2] * 0.1034)
    check(f"arm {aid} FK hand-TCP position vs frames.fk",
          p_tcp_drake, T_frames[:3, 3], tol=1e-9)
    check(f"arm {aid} FK hand rotation vs frames.fk",
          T_drake.rotation().matrix(), T_frames[:3, :3], tol=1e-9)

# paper pose
paper = plant.GetBodyByName("paper")
check("paper welded at canvas origin",
      plant.EvalBodyPoseInWorld(ctx, paper).translation(), [0, 0, 0])

print(f"\n{'ALL CHECKS PASS' if not fails else f'{len(fails)} FAILURES'}")
sys.exit(1 if fails else 0)
