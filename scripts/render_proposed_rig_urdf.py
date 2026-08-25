#!/usr/bin/env python3
"""Render assets/proposed_rig/installation.urdf itself -> out/proposed_rig_urdf.html

Not a scene built from the registries (that is `scripts/make_proposed_scene.py`)
— this loads THE GENERATED URDF in drake and draws what drake sees, so the
picture is evidence about the file: six arms hanging inverted on the 2 x 3
grid, their booms rising to the ceiling reference level, and every pen tip
down on the paper.

Each arm is posed at `layout.certified_ready_pose` with a small hover, so the
pens are AT the web rather than 10 cm over it; the printed table gives each
arm's pen-tip height straight out of drake's own FK of the URDF.

Run with the station venv (the system python has no drake):
  /home/franka/git/franka_manipulation_station/.venv/bin/python \
      scripts/render_proposed_rig_urdf.py [--hover M]

out/ is gitignored: this file is a look, not an artefact.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pydrake.geometry import Meshcat, MeshcatVisualizer  # noqa: E402
from pydrake.multibody.parsing import Parser             # noqa: E402
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph  # noqa: E402
from pydrake.systems.framework import DiagramBuilder     # noqa: E402

from aris_sixarm.layout import (FLEET_PROPOSED, LAYOUT_PROPOSED,  # noqa: E402
                                certified_ready_pose)
from aris_sixarm.rig_final6 import SHEET_FINAL6          # noqa: E402

URDF = ROOT / "assets/proposed_rig/installation.urdf"
OUT = ROOT / "out/proposed_rig_urdf.html"
H_INV = float(LAYOUT_PROPOSED["h"])


def ready_pose(spec, hover):
    """The certified ready pose at `hover`, falling back to the study's own
    0.10 m if nothing certifies that low (the pose must stay gated, so the
    fallback is a report line, never a silent relaxation)."""
    for h in (hover, 0.10):
        try:
            q, xy, rep = certified_ready_pose(spec, H_INV, hover=h)
            return q, xy, rep, h
        except RuntimeError:
            continue
    raise RuntimeError(f"no certified pose for arm {spec.arm_id}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hover", type=float, default=0.01,
                    help="pen-tip height above the paper, m (default 0.01)")
    args = ap.parse_args()

    builder = DiagramBuilder()
    plant, sg = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    Parser(plant).AddModels(str(URDF))
    plant.Finalize()
    meshcat = Meshcat()
    MeshcatVisualizer.AddToBuilder(builder, sg, meshcat)
    diagram = builder.Build()
    root = diagram.CreateDefaultContext()
    ctx = plant.GetMyContextFromRoot(root)

    W, H = SHEET_FINAL6
    print(f"{URDF.relative_to(ROOT)}: {plant.num_bodies()} bodies, "
          f"{plant.num_positions()} DOF, canvas {W} x {H} m, h = {H_INV} m")
    print(" arm   base (x, y)      hover over      pen tip (drake FK)   z")
    for aid, spec in FLEET_PROPOSED.items():
        q, xy, rep, h = ready_pose(spec, args.hover)
        for i in range(7):
            plant.GetJointByName(f"arm{aid}_panda_joint{i + 1}").set_angle(
                ctx, q[i])
        tip = plant.EvalBodyPoseInWorld(
            ctx, plant.GetBodyByName(f"arm{aid}_pen_tip")).translation()
        flag = "" if abs(h - args.hover) < 1e-12 else f"  (hover {h})"
        print(f" {aid:>3}   ({spec.xy[0]:.4f}, {spec.xy[1]:.4f})  "
              f"({xy[0]:.2f}, {xy[1]:.2f})   "
              f"({tip[0]:+.4f}, {tip[1]:+.4f}, {tip[2]:+.4f})  "
              f"{tip[2] * 1000:+6.1f} mm{flag}")
        on_paper = 0.0 <= tip[0] <= W and 0.0 <= tip[1] <= H
        if not on_paper:
            print(f"       WARNING: arm {aid} pen tip is off the web")

    diagram.ForcedPublish(root)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    html = meshcat.StaticHtml()
    OUT.write_text(html)
    print(f"\nwrote {OUT} ({len(html) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
