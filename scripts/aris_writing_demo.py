#!/usr/bin/env python3
"""Drake + meshcat animation: the fleet writes "ARIS" on the paper.

    /home/franka/git/franka_manipulation_station/.venv/bin/python \
        scripts/aris_writing_demo.py

Needs pydrake, so it runs under the station venv's python3.10 (the system
python3 has no drake).  `aris_sixarm.ik` falls back to the installed
`franka_analytical_ik` wheel there, so the package imports unchanged.

What this is:  KINEMATIC PLAYBACK ONLY.  There is no Simulator, no controller
and no dynamics — `plant.SetPositions` writes the planned joint vector for
every arm into a diagram context, the context time is set to the frame time
and `diagram.ForcedPublish` hands the pose to MeshcatVisualizer, which records
it.  MultibodyPlant is built with time_step 0 purely as a kinematics engine.

Ink:  each stroke is pre-loaded into the scene as ~15 short polyline chunks,
all `visible = False`, BEFORE StartRecording.  During playback each chunk is
switched visible with `SetProperty(..., time_in_recording=t)` at the frame the
pen tip reaches its far end, so the recorded animation grows the word as the
arms write it.  (The `time_in_recording` argument is what makes the call land
in the animation; without it, SetProperty is a live-only call.)

Outputs: out/aris_writing.html (self-contained, with drake's playback controls)
         out/aris_plan.json    (planned strokes; reused on the next run)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from aris_sixarm import letters, writing  # noqa: E402
from aris_sixarm.fleet import FLEET, SHEET  # noqa: E402
from aris_sixarm.frames import fk, Q_READY_FLOOR  # noqa: E402

from pydrake.geometry import (Box, Cylinder, Meshcat, MeshcatVisualizer,  # noqa: E402
                              Rgba, Sphere)
from pydrake.math import RigidTransform, RotationMatrix  # noqa: E402
from pydrake.multibody.parsing import Parser  # noqa: E402
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph  # noqa: E402
from pydrake.systems.framework import DiagramBuilder  # noqa: E402

PANDA_URL = "package://drake_models/franka_description/urdf/panda_arm_hand.urdf"
D_HAND_TCP = 0.1034      # panda_hand frame -> hand TCP, along hand z
PEN_LEN = 0.16           # pen cylinder length (visual)
PEN_R = 0.0045
PEN_TIP_Z = D_HAND_TCP + writing.PEN_EXT      # 0.2134 in the hand frame
FINGER_OPEN = 0.005      # each finger, so the jaws close on the pen
TABLE_T = 0.05
PAPER_T = 0.004


def _resolve_panda(parser):
    """Load the panda URDF by drake url; fall back to the on-disk cache copy.

    The url route reads ~/.cache/drake/package_map, so it works offline once
    the package has been fetched.  If it ever tries to go to the network we
    re-point the `drake_models` package at the cached directory instead.
    """
    try:
        return parser.AddModels(url=PANDA_URL)[0]
    except Exception as exc:                                   # pragma: no cover
        print(f"  url load failed ({exc}); falling back to the drake cache")
        hits = sorted(Path.home().glob(
            ".cache/drake/package_map/*/franka_description/urdf/panda_arm_hand.urdf"))
        if not hits:
            raise
        pkg_root = hits[0].parents[2]
        pm = parser.package_map()
        if pm.Contains("drake_models"):
            pm.Remove("drake_models")
        pm.Add("drake_models", str(pkg_root))
        return parser.AddModels(str(hits[0]))[0]


def build_scene(active_ids):
    """6 welded pandas + pen visuals + paper/table/base markers."""
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    arms = {}
    for aid, spec in FLEET.items():
        parser = Parser(plant, f"arm{aid}")
        mi = _resolve_panda(parser)
        # inverted arms carry the Ry(pi) of the hanging seat inside T_world_base
        X = RigidTransform(spec.T_world_base())
        plant.WeldFrames(plant.world_frame(),
                         plant.GetFrameByName("panda_link0", mi), X)
        if aid in active_ids:
            plant.RegisterVisualGeometry(
                plant.GetBodyByName("panda_hand", mi),
                RigidTransform([0.0, 0.0, PEN_TIP_Z - PEN_LEN / 2.0]),
                Cylinder(PEN_R, PEN_LEN), f"pen{aid}", [0.06, 0.06, 0.06, 1.0])
        arms[aid] = mi

    world = plant.world_body()
    plant.RegisterVisualGeometry(
        world, RigidTransform([SHEET[0] / 2, SHEET[1] / 2, -PAPER_T / 2]),
        Box(SHEET[0], SHEET[1], PAPER_T), "paper", [0.99, 0.99, 0.96, 1.0])
    plant.RegisterVisualGeometry(
        world, RigidTransform([SHEET[0] / 2, SHEET[1] / 2,
                               -PAPER_T - TABLE_T / 2]),
        Box(SHEET[0] + 0.30, SHEET[1] + 0.30, TABLE_T), "table",
        [0.45, 0.31, 0.19, 1.0])
    for aid, spec in FLEET.items():
        T = spec.T_world_base()
        plant.RegisterVisualGeometry(
            world, RigidTransform(T[:3, 3]), Sphere(0.045), f"base{aid}",
            [*spec.color, 1.0 if spec.active else 0.35])
    plant.Finalize()
    return builder, plant, scene_graph, arms


def set_arm(plant, context, mi, q7):
    """Kinematic pose: 7 arm joints + the two fingers, no dynamics."""
    plant.SetPositions(context, mi,
                       np.concatenate([q7, [FINGER_OPEN, FINGER_OPEN]]))


def fk_cross_check(plant, context, arms):
    """drake hand-TCP vs frames.fk, in world, for a floor and an inverted arm."""
    worst = 0.0
    for aid in (13, 31):
        spec = FLEET[aid]
        q = spec.q_seed if aid != 13 else Q_READY_FLOOR
        set_arm(plant, context, arms[aid], q)
        X_WH = plant.EvalBodyPoseInWorld(
            context, plant.GetBodyByName("panda_hand", arms[aid]))
        tcp_drake = X_WH.translation() + X_WH.rotation().matrix() @ [0, 0, D_HAND_TCP]
        T_b, _ = fk(q)
        Twb = spec.T_world_base()
        tcp_ours = Twb[:3, :3] @ T_b[:3, 3] + Twb[:3, 3]
        e = float(np.linalg.norm(tcp_drake - tcp_ours))
        print(f"  FK cross-check arm {aid} ({spec.mount}): |drake - ours| = {e:.3e} m")
        worst = max(worst, e)
    assert worst < 1e-6, f"drake/DH mismatch {worst:.3e} m"
    return worst


def pen_tip_world(plant, context, mi):
    X_WH = plant.EvalBodyPoseInWorld(context, plant.GetBodyByName("panda_hand", mi))
    return X_WH.translation() + X_WH.rotation().matrix() @ [0, 0, PEN_TIP_Z]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replan", action="store_true", help="ignore the cached plan")
    ap.add_argument("--fps", type=float, default=writing.FPS)
    ap.add_argument("--out", default=str(ROOT / "out/aris_writing.html"))
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "out/aris_plan.json"

    if cache.exists() and not args.replan:
        plan = writing.load_plan(cache)
        print(f"loaded cached plan {cache}")
    else:
        t0 = time.time()
        plan = writing.plan_word()
        print(f"planned the word in {time.time() - t0:.1f} s")
        writing.save_plan(plan, cache)
    for line in writing.report(plan):
        print("  " + line)

    sch = writing.build_schedule(plan, fps=args.fps)
    ts, qtraj, duration = sch["t"], sch["q"], sch["duration"]
    active = {L["arm_id"] for L in plan}

    print("building the drake scene...")
    builder, plant, scene_graph, arms = build_scene(active)
    meshcat = Meshcat()
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    pctx = plant.GetMyContextFromRoot(context)

    worst_fk = fk_cross_check(plant, pctx, arms)

    meshcat.SetProperty("/Grid", "visible", False)
    meshcat.SetProperty("/Axes", "visible", False)
    meshcat.SetProperty("/Background", "top_color", [0.93, 0.94, 0.97])
    meshcat.SetProperty("/Background", "bottom_color", [0.78, 0.80, 0.85])
    meshcat.SetCameraPose([SHEET[0] / 2, -2.45, 2.35],
                          [SHEET[0] / 2, SHEET[1] / 2, 0.15])

    # --- ink: every chunk in the scene up front, invisible ---
    reveal = {}
    for t_vis, li, si, ci, xyz in sch["ink"]:
        path = f"/ink/L{li}_S{si}_c{ci:03d}"
        meshcat.SetLine(path, np.asfortranarray(xyz.T), 2.0, Rgba(0.05, 0.05, 0.05, 1))
        meshcat.SetProperty(path, "visible", False)
        reveal.setdefault(int(round(t_vis * args.fps)), []).append(path)
    print(f"  ink: {len(sch['ink'])} chunks pre-loaded (hidden)")

    # --- kinematic playback ---
    print(f"recording {len(ts)} frames ({duration:.1f} s)...")
    t0 = time.time()
    meshcat.StartRecording(frames_per_second=args.fps,
                           set_visualizations_while_recording=False)
    # A boolean track with a single `true` key is CONSTANT true in three.js
    # (keyframe interpolation extrapolates backwards from the first key), which
    # would show the whole word from frame 0.  Anchoring every chunk `false` at
    # t = 0 gives each track a real false -> true transition.
    for paths in reveal.values():
        for path in paths:
            meshcat.SetProperty(path, "visible", False, time_in_recording=0.0)
    checks = []
    stroke_spans = [p for p in sch["phases"] if p["kind"] == "stroke"]
    for k, t in enumerate(ts):
        context.SetTime(float(t))
        for aid, mi in arms.items():
            set_arm(plant, pctx, mi, qtraj[aid][k])
        diagram.ForcedPublish(context)
        for path in reveal.get(k, ()):
            meshcat.SetProperty(path, "visible", True, time_in_recording=float(t))
        # end-to-end check, every drawing frame: does the pen tip that drake
        # renders actually sit on the commanded curve, at z = 0?
        for ph in stroke_spans:
            if not (ph["t0"] + 1e-9 < t < ph["t1"] - 1e-9):
                continue
            pts = plan[ph["letter"]]["strokes"][ph["stroke"]]["pts"]
            # the reference is interpolated exactly the way the trajectory is,
            # else the check measures its own quantisation of the step grid
            fi = (t - ph["t0"]) / (ph["t1"] - ph["t0"]) * (len(pts) - 1)
            i0 = min(int(fi), len(pts) - 2)
            ref = pts[i0] + (fi - i0) * (pts[i0 + 1] - pts[i0])
            tip = pen_tip_world(plant, pctx, arms[ph["arm"]])
            checks.append(np.linalg.norm(tip - [ref[0], ref[1], 0.0]))
    meshcat.StopRecording()
    meshcat.PublishRecording()
    print(f"  recorded in {time.time() - t0:.1f} s")
    worst_tip = float(np.max(checks)) if checks else float("nan")
    print(f"  drake pen-tip on paper: max {worst_tip:.2e} m over {len(checks)} "
          f"sampled stroke frames")
    assert worst_tip < 1e-3, f"drake pen tip off the stroke by {worst_tip:.2e} m"

    html = meshcat.StaticHtml()
    out.write_text(html)
    print(f"wrote {out} ({len(html) / 1e6:.1f} MB, {duration:.1f} s of animation, "
          f"{len(ts)} frames @ {args.fps:g} fps)")
    print(f"summary: FK cross-check {worst_fk:.1e} m, pen-tip {worst_tip:.1e} m, "
          f"ink chunks {len(sch['ink'])}")


if __name__ == "__main__":
    main()
