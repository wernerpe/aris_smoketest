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

# the logo's own two inks (trace.GREY_RGB / trace.ORANGE_RGB), inlined so
# this file imports nothing that needs the system python's numpy stack
INK_HEX = {"grey": "#666665", "orange": "#cb6608"}
from pydrake.geometry import (Box, Cylinder, Meshcat,           # noqa: E402
                              MeshcatVisualizer, Rgba, Sphere)
from pydrake.math import RigidTransform                          # noqa: E402
from pydrake.multibody.parsing import Parser                     # noqa: E402
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph  # noqa: E402
from pydrake.systems.framework import DiagramBuilder             # noqa: E402

PANDA_URL = "package://drake_models/franka_description/urdf/panda_arm_hand.urdf"
D_HAND_TCP = 0.1034
PEN_EXT = 0.110          # the default pen; the schedule carries the real ones
PEN_MARGIN = 0.05        # m of pen body above the tip's own length
PEN_R = 0.0045
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


def build_scene(pen_ext, inks):
    """Six welded pandas, each with ITS OWN pen, + paper/table.

    The pen is not decoration: its length is the `pen_ext` the segment was
    planned and validated with, so drawing it at a fixed 110 mm while arm 2
    holds 300 mm would show a robot reaching 19 cm short of the ink it is
    laying.  One cylinder per (arm, ink) is registered at the same pose, and
    the animation shows exactly one of them at a time — that is the pen swap.
    """
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    arms, pen_names = {}, {}
    for aid, spec in FLEET.items():
        mi = _resolve_panda(Parser(plant, f"arm{aid}"))
        plant.WeldFrames(plant.world_frame(),
                         plant.GetFrameByName("panda_link0", mi),
                         RigidTransform(spec.T_world_base()))
        tip_z = D_HAND_TCP + float(pen_ext[aid])
        length = float(pen_ext[aid]) + PEN_MARGIN
        for ink, hexcol in inks.items():
            name = f"pen{aid}_{ink}"
            plant.RegisterVisualGeometry(
                plant.GetBodyByName("panda_hand", mi),
                RigidTransform([0.0, 0.0, tip_z - length / 2.0]),
                Cylinder(PEN_R, length), name, _rgba(hexcol))
            pen_names.setdefault(aid, {})[ink] = name
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
    return builder, plant, scene_graph, arms, pen_names


LEGEND = """
<div style="position:fixed;top:12px;left:12px;z-index:1000;background:rgba(255,255,255,0.94);
border:1px solid #bbb;border-radius:8px;padding:10px 14px;font:12px/1.55 sans-serif;color:#222;max-width:470px">
<b>Aris Kindt &mdash; six arms draw the whole CSAIL logo</b><br>
%(rows)s
<hr style="margin:6px 0">
<b>%(cov).2f %% of the %(tot).2f m traced is drawn</b> &mdash; every metre of it
returned by <code>plan_stroke</code> with an independent validator's certificate.
Logo %(lw).2f x %(lh).2f m at offset (%(ox)+.3f, %(oy)+.3f) m: the largest size
at which the fleet certifies ALL of it.<br>
%(swap)s
<hr style="margin:6px 0">
<b>Conducted, not merely concurrent.</b>  Every arm's path and stroke order is
frozen; conductor v1 only stretches the clock, inserting %(pause).1f s of pauses
(priority = busiest arm first) so that all 15 arm pairs keep at least
%(margin)d mm of capsule clearance for the whole run.  An independent validator
(<code>scene_check</code>) re-derived each phase's merged timeline and measured a
minimum clearance of <b>%(clear).1f mm</b>; the animation is only rendered from
timelines that pass it.<br>
<i>v1 does not re-order strokes, re-route a transit or model dynamics, and this
is kinematic playback &mdash; a pause is instantaneous here, but a real run needs
the acceleration-limited version of the same schedule.</i><br>
%(nframe)d frames @ %(fps)g fps = %(dur).1f s.  Pen tip on the commanded curve to
%(tip).2f mm on every drawing frame.
</div>
"""

SWAP_NOTE = """<b>Two passes, one piece.</b>  Phase 1 lays every grey line; all
six arms then park for %(swapdur).1f s while a human swaps grey pens for orange;
phase 2 lays every orange line.  One pen per arm per phase &mdash; the constraint
that is lifted is one pen per arm for the WHOLE piece, and lifting it is what
takes the coverage to 100 %%.  Pen LENGTHS do not change at the swap: they are
fixtures (%(pens)s).<br>"""


# MeshcatVisualizer builds a geometry's path from its SCOPED name with `::`
# split into path segments, so a cylinder registered as `pen31_grey` on model
# `arm31::panda` lands at `.../arm31/panda/panda_hand/arm31/panda/pen31_grey`:
# the model scope appears twice, once for the body and once inside the
# geometry's own scoped name.  That is drake's business and has changed before,
# so the path is PROBED with `HasPath` and a miss is reported rather than
# silently leaving both pens visible (which would read as an arm holding two).
def pen_paths(meshcat, plant, arms, pen_names, prefix="/drake/visualizer"):
    """-> {(arm, ink): meshcat path or None} for every registered pen."""
    out = {}
    for aid, mi in arms.items():
        model = plant.GetModelInstanceName(mi)          # e.g. "arm31::panda"
        scope = model.replace("::", "/")
        for ink, name in pen_names[aid].items():
            cands = (f"{prefix}/{scope}/panda_hand/{scope}/{name}",
                     f"{prefix}/{scope}/panda_hand/{model}::{name}",
                     f"{prefix}/{scope}/panda_hand/{name}",
                     f"{prefix}/arm{aid}/panda_hand/{name}")
            out[(aid, ink)] = next((c for c in cands if meshcat.HasPath(c)), None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", default=str(ROOT / "out/csail_schedule_full.npz"))
    ap.add_argument("--summary", default=str(ROOT / "out/csail_schedule_full.json"))
    ap.add_argument("--out", default=str(ROOT / "out/csail_full.html"))
    ap.add_argument("--zip", default=str(ROOT / "out/csail_full.zip"))
    ap.add_argument("--budget", type=float, default=28.0, help="MiB, zip cap")
    args = ap.parse_args()

    d = np.load(args.schedule, allow_pickle=False)
    summary = json.loads(Path(args.summary).read_text())
    fps = float(d["fps"])
    nF = int(d["n_frames"])
    ts = np.arange(nF) / fps
    fleet = [int(x) for x in d["arms"]]
    drawing = [int(x) for x in d["drawing_arms"]]
    pen_ext = {a: float(v) for a, v in zip(sorted(FLEET), d["pen_ext"])}
    phase = d["phase"] if "phase" in d else np.zeros(nF, np.int64)
    phase_ink = [str(x) for x in d["phase_ink"]] if "phase_ink" in d else ["grey"]
    inks = {"grey": INK_HEX["grey"], "orange": INK_HEX["orange"]}
    print(f"schedule: {nF} frames @ {fps:g} fps = {(nF - 1) / fps:.1f} s, "
          f"{len(d['ink_t'])} ink chunks, {len(phase_ink)} phase(s), min clearance "
          f"{float(d['min_clearance']) * 1000:.1f} mm")
    print("  pens: " + ", ".join(f"arm {a} = {1000 * pen_ext[a]:.0f} mm"
                                 for a in fleet))

    print("building the drake scene...")
    builder, plant, scene_graph, arms, pen_names = build_scene(pen_ext, inks)
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
    diagram.ForcedPublish(context)          # so the geometry paths exist

    # --- pens: one per (arm, ink), exactly one visible at a time ---------
    pen_path = pen_paths(meshcat, plant, arms, pen_names)
    missing = [k for k, v in pen_path.items() if v is None]
    if missing:
        print(f"  WARNING: {len(missing)} pen geometries not found in meshcat; "
              "the pen swap will not be shown")

    # --- ink: every chunk in the scene up front, invisible, in its own ink ---
    reveal = {}
    off, tvis, iarm = d["ink_off"], d["ink_t"], d["ink_arm"]
    ihex = ([str(x) for x in d["ink_hex"]] if "ink_hex" in d
            else [INK_HEX["grey"]] * len(tvis))
    for k in range(len(tvis)):
        xyz = d["ink_xyz"][off[k]:off[k + 1]]
        aid = int(iarm[k])
        path = f"/ink/a{aid}_c{k:04d}"
        meshcat.SetLine(path, np.asfortranarray(xyz.T), 2.4, Rgba(*_rgba(ihex[k])))
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
    for (a, ink), p in pen_path.items():        # start every arm holding phase 1's
        if p is not None:
            meshcat.SetProperty(p, "visible", ink == phase_ink[0],
                                time_in_recording=0.0)

    q = {a: d[f"q_{a}"] for a in fleet}
    seg = {a: d[f"seg_{a}"] for a in fleet}
    uu = {a: d[f"u_{a}"] for a in fleet}
    segpts = {a: d[f"segpts_{a}"] for a in fleet}
    segoff = {a: d[f"segoff_{a}"] for a in fleet}
    checks, n_draw, shown = [], 0, phase_ink[0]
    for k, t in enumerate(ts):
        context.SetTime(float(t))
        for aid, mi in arms.items():
            plant.SetPositions(pctx, mi, np.concatenate(
                [q[aid][k], [FINGER_OPEN, FINGER_OPEN]]))
        diagram.ForcedPublish(context)
        for p in reveal.get(k, ()):
            meshcat.SetProperty(p, "visible", True, time_in_recording=float(t))
        # the swap happens in the middle of the parked pause, which is where a
        # human would actually be doing it
        want = phase_ink[int(phase[k])] if phase[k] >= 0 else shown
        if want != shown:
            for aid in fleet:
                for ink in inks:
                    if pen_path[(aid, ink)] is not None:
                        meshcat.SetProperty(pen_path[(aid, ink)], "visible",
                                            ink == want, time_in_recording=float(t))
            shown = want
        # every drawing frame: is the pen tip drake renders on the commanded
        # curve, at z = 0?  This is the end-to-end check — schedule, densified
        # joints, drake's own kinematics and THIS ARM's pen offset, all at once.
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
            tip = X.translation() + X.rotation().matrix() @ [
                0, 0, D_HAND_TCP + pen_ext[aid]]
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
    per_arm = {}
    for ph in summary["phases"]:
        for a, m in ph["arm_metres"].items():
            per_arm.setdefault(int(a), []).append(
                (ph["ink"] or "grey", float(m), int(ph["arm_segments"][a])))
    for aid in fleet:
        bits = [f"{ink} {m:.2f} m in {n} seg" for ink, m, n in per_arm.get(aid, [])
                if n]
        rows.append(
            f'<span style="color:{INK_HEX["grey"]}">&#9632;</span> arm {aid} '
            f"{FLEET[aid].name} &mdash; <b>{1000 * pen_ext[aid]:.0f} mm</b> pen "
            "&mdash; " + (" then ".join(bits) if bits
                          else "reaches none of the logo (idle)"))
    swap = ""
    if summary.get("two_pass"):
        swap = SWAP_NOTE % dict(
            swapdur=summary["pen_swap_pause_s"],
            pens=", ".join(f"arm {a} {v:.0f} mm"
                           for a, v in sorted(summary["pens_mm"].items(),
                                              key=lambda kv: int(kv[0]))))
    html = meshcat.StaticHtml().replace("</body>", LEGEND % dict(
        rows="<br>".join(rows), tot=summary["traced_m"],
        cov=100 * summary["coverage"], lw=summary["logo"]["w"],
        lh=summary["logo"]["h"], ox=summary["logo"]["offset"][0],
        oy=summary["logo"]["offset"][1], swap=swap,
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
          f"{summary['pause_total']:.1f} s of conducted pauses, coverage "
          f"{100 * summary['coverage']:.2f} %")


if __name__ == "__main__":
    main()
