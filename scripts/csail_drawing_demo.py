#!/usr/bin/env python3
"""Drake + meshcat: six arms draw the CSAIL logo at once, on a conducted clock.

    /home/franka/git/franka_manipulation_station/.venv/bin/python \
        scripts/csail_drawing_demo.py

Needs pydrake, so it runs under the station venv's python3.10.  It PLANS
NOTHING: `scripts/csail_schedule.py` (system python3.12, where the batch IK
entry points live) has already allocated, frozen the per-arm paths, conducted
them into one collision-checked timeline and had `scene_check` sign it off.
All that arrives here is out/csail_schedule_6arm.npz — joints per frame, ink
chunks with the instant each becomes visible, and the reference curve every
drawing frame is checked against.

KINEMATIC PLAYBACK ONLY.  No Simulator, no controller, no dynamics:
`plant.SetPositions` writes the scheduled joint vector for all six arms into a
diagram context, the context time is set to the frame time, and
`diagram.ForcedPublish` hands the pose to MeshcatVisualizer, which records it.
A conductor PAUSE is therefore literally an arm holding still for some frames.

Ink: every chunk is pre-loaded invisible before StartRecording and switched on
with `SetProperty(..., time_in_recording=t)` at the frame the pen reaches its
far end, in the pen colour of the arm that drew it.  The explicit `visible =
False` key at t = 0 is load-bearing: a boolean track with a single `true` key
reads as constant-true in three.js and would show the finished logo from frame
zero.

Outputs: out/csail_drawing.html (self-contained) and, with --zip, the -9 zip
the size budget is measured on.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from aris_sixarm.fleet import FLEET, SHEET                      # noqa: E402
from pydrake.geometry import (Box, Cylinder, Meshcat,           # noqa: E402
                              MeshcatVisualizer, Rgba, Sphere)
from pydrake.math import RigidTransform                          # noqa: E402
from pydrake.multibody.parsing import Parser                     # noqa: E402
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph  # noqa: E402
from pydrake.systems.framework import DiagramBuilder             # noqa: E402

PANDA_URL = "package://drake_models/franka_description/urdf/panda_arm_hand.urdf"
D_HAND_TCP = 0.1034
PEN_EXT = 0.110
PEN_LEN = 0.16
PEN_R = 0.0045
PEN_TIP_Z = D_HAND_TCP + PEN_EXT
FINGER_OPEN = 0.005
TABLE_T = 0.05
PAPER_T = 0.004
TIP_TOL = 5e-4           # m, the brief's per-frame fidelity gate


def _rgba(hexstr, alpha=1.0):
    h = hexstr.lstrip("#")
    return [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)] + [alpha]


def _resolve_panda(parser):
    try:
        return parser.AddModels(url=PANDA_URL)[0]
    except Exception as exc:                                   # pragma: no cover
        print(f"  url load failed ({exc}); falling back to the drake cache")
        hits = sorted(Path.home().glob(
            ".cache/drake/package_map/*/franka_description/urdf/panda_arm_hand.urdf"))
        if not hits:
            raise
        pm = parser.package_map()
        if pm.Contains("drake_models"):
            pm.Remove("drake_models")
        pm.Add("drake_models", str(hits[0].parents[2]))
        return parser.AddModels(str(hits[0]))[0]


def build_scene(pen_color):
    """Six welded pandas, each with a pen in its own ink colour, + paper/table."""
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    arms = {}
    for aid, spec in FLEET.items():
        mi = _resolve_panda(Parser(plant, f"arm{aid}"))
        plant.WeldFrames(plant.world_frame(),
                         plant.GetFrameByName("panda_link0", mi),
                         RigidTransform(spec.T_world_base()))
        plant.RegisterVisualGeometry(
            plant.GetBodyByName("panda_hand", mi),
            RigidTransform([0.0, 0.0, PEN_TIP_Z - PEN_LEN / 2.0]),
            Cylinder(PEN_R, PEN_LEN), f"pen{aid}", _rgba(pen_color[aid]))
        arms[aid] = mi
    world = plant.world_body()
    plant.RegisterVisualGeometry(
        world, RigidTransform([SHEET[0] / 2, SHEET[1] / 2, -PAPER_T / 2]),
        Box(SHEET[0], SHEET[1], PAPER_T), "paper", [0.99, 0.99, 0.96, 1.0])
    plant.RegisterVisualGeometry(
        world, RigidTransform([SHEET[0] / 2, SHEET[1] / 2, -PAPER_T - TABLE_T / 2]),
        Box(SHEET[0] + 0.30, SHEET[1] + 0.30, TABLE_T), "table",
        [0.45, 0.31, 0.19, 1.0])
    for aid, spec in FLEET.items():
        plant.RegisterVisualGeometry(
            world, RigidTransform(spec.T_world_base()[:3, 3]), Sphere(0.045),
            f"base{aid}", [*spec.color, 1.0])
    plant.Finalize()
    return builder, plant, scene_graph, arms


LEGEND = """
<div style="position:fixed;top:12px;left:12px;z-index:1000;background:rgba(255,255,255,0.94);
border:1px solid #bbb;border-radius:8px;padding:10px 14px;font:12px/1.55 sans-serif;color:#222;max-width:430px">
<b>Aris Kindt &mdash; six arms draw the CSAIL logo</b><br>
%(rows)s
<hr style="margin:6px 0">
<span style="color:#c9c9c9">&#9632;</span> left empty &mdash; %(drop).2f m,
%(dropf).1f %% of the %(tot).2f m traced: no arm reaches it, at any placement.<br>
Logo %(lw).2f x %(lh).2f m (%(scale)d %% of the margin-limited size), the
placement search's pick: largest size within 1 pp of the best coverage found.<br>
<hr style="margin:6px 0">
<b>Conducted, not merely concurrent.</b>  Every arm's path and stroke order is
frozen; conductor v1 only stretches the clock, inserting %(pause).1f s of pauses
(priority = busiest arm first) so that all 15 arm pairs keep at least
%(margin)d mm of capsule clearance for the whole run.  An independent validator
(<code>scene_check</code>) re-derived the merged timeline and measured a minimum
clearance of <b>%(clear).1f mm</b>; the animation is only rendered from a
timeline that passes it.<br>
<i>v1 does not re-order strokes, re-route a transit or model dynamics, and this
is kinematic playback &mdash; a pause is instantaneous here, but a real run needs
the acceleration-limited version of the same schedule.</i><br>
%(nframe)d frames @ %(fps)g fps = %(dur).1f s.  Pen tip on the commanded curve to
%(tip).2f mm on every drawing frame.
</div>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", default=str(ROOT / "out/csail_schedule_6arm.npz"))
    ap.add_argument("--summary", default=str(ROOT / "out/csail_schedule_6arm.json"))
    ap.add_argument("--out", default=str(ROOT / "out/csail_drawing.html"))
    ap.add_argument("--zip", default=str(ROOT / "out/csail_drawing.zip"))
    ap.add_argument("--budget", type=float, default=28.0, help="MiB, zip cap")
    args = ap.parse_args()

    d = np.load(args.schedule, allow_pickle=False)
    summary = json.loads(Path(args.summary).read_text())
    fps = float(d["fps"])
    nF = int(d["n_frames"])
    ts = np.arange(nF) / fps
    fleet = [int(x) for x in d["arms"]]
    drawing = [int(x) for x in d["drawing_arms"]]
    pen = {a: str(d[f"pen_{a}"]) for a in fleet}
    print(f"schedule: {nF} frames @ {fps:g} fps = {(nF - 1) / fps:.1f} s, "
          f"{len(d['ink_t'])} ink chunks, min clearance "
          f"{float(d['min_clearance']) * 1000:.1f} mm")

    print("building the drake scene...")
    builder, plant, scene_graph, arms = build_scene(pen)
    meshcat = Meshcat()
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    pctx = plant.GetMyContextFromRoot(context)

    meshcat.SetProperty("/Grid", "visible", False)
    meshcat.SetProperty("/Axes", "visible", False)
    meshcat.SetProperty("/Background", "top_color", [0.93, 0.94, 0.97])
    meshcat.SetProperty("/Background", "bottom_color", [0.78, 0.80, 0.85])
    meshcat.SetCameraPose([SHEET[0] / 2, -2.30, 2.55],
                          [SHEET[0] / 2, SHEET[1] / 2, 0.10])

    # --- ink: every chunk in the scene up front, invisible, in its pen colour ---
    reveal = {}
    off, tvis, iarm = d["ink_off"], d["ink_t"], d["ink_arm"]
    for k in range(len(tvis)):
        xyz = d["ink_xyz"][off[k]:off[k + 1]]
        aid = int(iarm[k])
        path = f"/ink/a{aid}_c{k:04d}"
        meshcat.SetLine(path, np.asfortranarray(xyz.T), 2.4,
                        Rgba(*_rgba(pen[aid])))
        meshcat.SetProperty(path, "visible", False)
        reveal.setdefault(int(round(float(tvis[k]) * fps)), []).append(path)
    print(f"  ink: {len(tvis)} chunks pre-loaded (hidden)")

    # --- kinematic playback ---
    print(f"recording {nF} frames...")
    t0 = time.time()
    meshcat.StartRecording(frames_per_second=fps,
                           set_visualizations_while_recording=False)
    for paths in reveal.values():
        for p in paths:
            meshcat.SetProperty(p, "visible", False, time_in_recording=0.0)

    q = {a: d[f"q_{a}"] for a in fleet}
    seg = {a: d[f"seg_{a}"] for a in fleet}
    uu = {a: d[f"u_{a}"] for a in fleet}
    segpts = {a: d[f"segpts_{a}"] for a in fleet}
    segoff = {a: d[f"segoff_{a}"] for a in fleet}
    checks, n_draw = [], 0
    for k, t in enumerate(ts):
        context.SetTime(float(t))
        for aid, mi in arms.items():
            plant.SetPositions(pctx, mi, np.concatenate(
                [q[aid][k], [FINGER_OPEN, FINGER_OPEN]]))
        diagram.ForcedPublish(context)
        for p in reveal.get(k, ()):
            meshcat.SetProperty(p, "visible", True, time_in_recording=float(t))
        # every drawing frame: is the pen tip drake renders on the commanded
        # curve, at z = 0?  This is the end-to-end check — schedule, densified
        # joints, drake's own kinematics and the pen offset, all at once.
        for aid in drawing:
            s = int(seg[aid][k])
            if s < 0:
                continue
            P = segpts[aid][segoff[aid][s]:segoff[aid][s + 1]]
            fi = float(np.clip(uu[aid][k], 0.0, 1.0)) * (len(P) - 1)
            i0 = min(int(fi), len(P) - 2)
            ref = P[i0] + (fi - i0) * (P[i0 + 1] - P[i0])
            X = plant.EvalBodyPoseInWorld(pctx, plant.GetBodyByName("panda_hand",
                                                                    arms[aid]))
            tip = X.translation() + X.rotation().matrix() @ [0, 0, PEN_TIP_Z]
            checks.append(np.linalg.norm(tip - [ref[0], ref[1], 0.0]))
            n_draw += 1
    meshcat.StopRecording()
    meshcat.PublishRecording()
    worst = float(np.max(checks)) if checks else float("nan")
    print(f"  recorded in {time.time() - t0:.1f} s")
    print(f"  pen tip on the commanded curve: max {worst * 1000:.3f} mm over "
          f"{n_draw} drawing-frame samples")
    assert worst < TIP_TOL, f"pen tip off the stroke by {worst * 1000:.2f} mm"

    rows = []
    for aid in fleet:
        m = float(summary["arm_metres"].get(str(aid), 0.0))
        n = int(summary["arm_segments"].get(str(aid), 0))
        p = summary["pauses"].get(str(aid))
        rows.append(
            f'<span style="color:{pen[aid]}">&#9632;</span> arm {aid} '
            f"{FLEET[aid].name} &mdash; <b>{summary['colors'][str(aid)]}</b> pen "
            f"&mdash; " + (f"{m:.2f} m in {n} segments"
                           + (f", {p:.1f} s paused" if p else ", no pauses")
                           if n else "reaches none of the logo (idle)"))
    tot = summary["traced_m"]
    html = meshcat.StaticHtml().replace("</body>", LEGEND % dict(
        rows="<br>".join(rows), drop=summary["dropped_m"], tot=tot,
        dropf=100 * summary["dropped_m"] / tot, lw=summary["logo"]["w"],
        lh=summary["logo"]["h"], scale=round(100 * summary["logo"]["w"] / 2.4106),
        pause=summary["pause_total"], margin=round(1000 * summary["margin"]),
        clear=1000 * summary["min_clearance"], nframe=nF, fps=fps,
        dur=(nF - 1) / fps, tip=1000 * worst) + "</body>")
    out = Path(args.out)
    out.write_text(html)
    mb = len(html) / 2**20
    print(f"wrote {out} ({mb:.1f} MiB)")

    if args.zip:
        z = Path(args.zip)
        if z.exists():
            z.unlink()
        subprocess.run(["zip", "-9", "-j", str(z), str(out)], check=True,
                       stdout=subprocess.DEVNULL)
        zmb = z.stat().st_size / 2**20
        print(f"wrote {z} ({zmb:.1f} MiB, budget {args.budget:g} MiB) "
              f"-> {'OK' if zmb <= args.budget else 'OVER BUDGET'}")
    print(f"summary: {nF} frames @ {fps:g} fps = {(nF - 1) / fps:.1f} s, "
          f"tip {worst * 1000:.3f} mm, min clearance "
          f"{1000 * summary['min_clearance']:.1f} mm, "
          f"{summary['pause_total']:.1f} s of conducted pauses")


if __name__ == "__main__":
    main()
