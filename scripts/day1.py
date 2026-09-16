#!/usr/bin/env python3
"""HARDWARE DAY 1 — one front door, two subcommands.  Read this at the rig.

    scripts/day1.py line --arm 31 --from 0.50,1.70 --to 0.65,1.70 --name probe
    scripts/day1.py word --variant alt

`line` is rung B of `docs/HARDWARE_DAY1.md` cut down to ONE straight line for
ONE arm: it plans it with the certified single-arm stroke planner
(`aris_sixarm.stroke_api.plan_stroke`), lays the solo programme down with
`aris_sixarm.writing.arm_program` (park -> hover -> down -> line -> up -> park),
grades the whole thing with the INDEPENDENT checker (`scene_check.check_timeline`,
every gate, the seam bars in, the other five arms standing at their parks), and
only then writes the timeline, the impedance pathway CSV and a summary.  An
uncertifiable line is a non-zero exit and no CSV.

`word` is rungs C/D and it PLANS NOTHING.  It takes yesterday's certified
h = 0.970 assets, re-derives the certificate on site by re-running the
independent whole-timeline check on them today, copies the per-arm CSVs into
`out/day1/`, and prints the run order and the pass criteria.  `--replan` is the
escape hatch: it re-runs the conductor with yesterday's exact job parameters and
REFUSES to call the result a success on anything but the conducted `coverage`
in the schedule JSON (never the log's COVERAGE line — see HARDWARE_DAY1 §3).

WHAT IS NEW HERE AND WHAT IS NOT.  No planning machinery is new.  Three small
adapters live in this file and nowhere else, and they are named so nobody has to
go looking:

  `_segment`      one certified `plan_stroke` result -> the one-entry programme
                  list `writing.arm_program` takes (the shape
                  `allocate._entry` builds; this file builds one by hand
                  because there is no allocator in a one-line run).
  `_payload`      the six-arm npz `execute.program.from_schedule`,
                  `recheck_timeline` and `export.pathway` all read, with the
                  mover's frames in it and the other five arms tiled at their
                  parks.  It is `csail_schedule.payload` minus the animation
                  arrays, which describe ink on a clock a one-line run has not
                  got.
  `_program_json` the label file the exporter reads (rig, h_inv, phases,
                  draw_speed).  Nothing load-bearing is in it.

THE HOVER IS A LONGER PEN, NOT A LOWER PLATE.  `--hover 0.03` plans the line
with `pen_ext + 0.030` and then GRADES it with the real pen at the real height.
That is the same trajectory `docs/HARDWARE_DAY1.md` §4.1 gets by planning at
h = 0.940 and flying at 0.970 — a base 30 mm lower and a pen 30 mm longer put
the hand in the identical place relative to the base — and it needs no second
fleet, no second atlas and no second park set.  The CSV for a hover line is
written with `--z-mode fk`, so its `z_m` is the tip's real height above the
paper and not the paper plane it is deliberately not touching.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# The rig and the tool are process-global state read at import time, and the
# tests run without env vars — so default them here, exactly the way
# `scripts/recheck_timeline.py` does, and let a real shell override.
os.environ.setdefault("ARIS_RIG", "proposed")
os.environ.setdefault("ARIS_TOOL", "lateral")
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from aris_sixarm import (coordination, fleet, frames, layout,  # noqa: E402
                         mounts, pwl, scene_check, stroke_api, writing)
from aris_sixarm.export import pathway                          # noqa: E402

import recheck_timeline as rt                                   # noqa: E402

# --- the numbers a one-line run is conducted at ----------------------------
# The fleet clock of every programme in `out/`: `--fps 48 --substeps 1`, so the
# npz is UN-DECIMATED and `scene_check` graded the frames that are in the file.
FPS = 48.0
STRIDE = 1
DT = 1.0 / FPS
SUB = 2                                   # scene_check interpolation, as shipped
DRAW_SPEED = 0.12                         # m/s asked for; measured ~0.0867
TRANSIT_SPEED = 0.8
TILT_MAX_DEG = 15.0                       # v19's cone; changes the atlas rights
HOVER_DEFAULT = 0.030                     # m, the day plan's hover height
OUT_DIR = ROOT / "out" / "day1"
ARMS = (31, 71)

# --- where yesterday's assets live on a machine with no out/ ----------------
# `out/` is gitignored, so a fresh clone on the robot PC has none of it.  The
# handful of 0.970 artefacts this file READS — three schedule npz, the per-arm
# pathway CSVs it copies, the strokes file `--replan` needs and the atlas
# `--replan` plans against — are TRACKED under `assets/site/h0970/`, laid out
# with the same names they have under `out/`.  `out/` still wins when it is
# there, so a workstation that has just re-planned sees its own new file and
# not yesterday's copy; `ARIS_ASSETS=DIR` moves the fallback.
ASSETS = Path(os.environ.get("ARIS_ASSETS")
              or ROOT / "assets" / "site" / "h0970")


def asset(rel):
    """`out/<rel>` if it exists, else the tracked site copy. -> Path.

    Returns the `out/` path when NEITHER exists, so a missing-file message
    names the place the file is normally written rather than the fallback.
    """
    p = ROOT / "out" / rel
    if p.exists():
        return p
    alt = ASSETS / rel
    return alt if alt.exists() else p


# --- yesterday's certified assets, and what each is for --------------------
# `program`/`summary` are not read by `word` itself — they are the two files
# `aris_sixarm.gui.day1_bundle` needs to turn the same asset into a viewer
# bundle, and they belong next to the npz they describe.
VARIANTS = {
    "alt": dict(
        npz=asset("unknown_h0970_home_alt.npz"),
        csv_stem="unknown_h0970_home_alt",
        program=asset("unknown_h0970_home_program.json"),
        summary=asset("unknown_h0970_home_schedule.json"),
        what="the ALTERNATING word — the deliverable (rung C)"),
    "concurrent": dict(
        npz=asset("unknown_h0970_home_schedule.npz"),
        csv_stem="unknown_h0970_home",
        program=asset("unknown_h0970_home_program.json"),
        summary=asset("unknown_h0970_home_schedule.json"),
        what="the CONCURRENT word — livelier, and the only collision risk (rung D)"),
    "hover": dict(
        npz=asset("unknown_h0970_hover_schedule.npz"),
        csv_stem="unknown_h0970_hover",
        program=asset("unknown_h0970_hover_program.json"),
        summary=asset("unknown_h0970_hover_schedule.json"),
        what="the 30 mm HOVER pass — never touches the paper (rung A)"),
}

# the conductor's job parameters from 2026-09-15, verbatim.  They are not
# tidyable: HARDWARE_DAY1 §3 measures what each one is holding up.
REPLAN_FLAGS = [
    "--arms", "31,71",
    "--atlas", "out/atlas_proposed_h0970_lat0860",
    "--tilt-max-deg", "15",
    "--min-len", "0.02",
    "--fps", "48", "--substeps", "1", "--subcheck", "2",
    "--idle-policy", "home",
    "--skip-unconductable", "--freeze-refused-phase", "--depot-hover-selective",
    "--residual-passes", "6", "--residual-min-gain", "0.0005",
    "--jobs", "6",
    "--program", "--no-anim",
]
COVERAGE_GATE = 0.99


# ===========================================================================
# what both subcommands say before they do anything
# ===========================================================================
def assumptions(arms=ARMS):
    """Every number this run stands on, printed first. -> list[str]."""
    fl = layout.FLEET_PROPOSED
    h = float(layout.LAYOUT_PROPOSED["h"])
    lat, ext = float(frames.PEN_LAT_HOLDER), float(frames.PEN_EXT_HOLDER)
    out = [
        "ASSUMPTIONS (every one of these is a thing that can be wrong):",
        f"  mounting height   h = {h:.3f} m, inverted, plate underside above "
        f"the paper surface",
    ]
    for a in arms:
        if a not in fl:
            continue
        T = np.asarray(fl[a].T_world_base(), float)
        yaw = float(np.degrees(np.arctan2(T[1, 0], T[0, 0])))
        out.append(f"  arm {a:<3d} base      ({T[0, 3]:.4f}, {T[1, 3]:.5f}, "
                   f"{T[2, 3]:.3f}) m, mount {fl[a].mount}, yaw {yaw:+.1f} deg")
    out += [
        f"  tool tip          ({lat:.4f}, 0, {frames.D_HAND_TCP + ext:.4f}) m in "
        f"the hand (flange) frame",
        f"                    = ({lat:.4f}, 0, {ext:.4f}) m from the hand TCP — "
        f"READ OFF A PHOTOGRAPH, not a touchdown",
        f"  seam bars         {mounts.SEAM_SOURCE}",
        f"                    modelled: {'ON' if mounts.SEAM_POSTS_ON else 'OFF'}"
        f" (ARIS_SEAM_POSTS); they are the gate closest to arms 31 and 71",
        f"  PAIR_MARGIN       {1000 * coordination.PAIR_MARGIN:.0f} mm — the "
        f"inter-arm, frame and neighbour-column gate",
        f"  rig / tool        {os.environ['ARIS_RIG']} / {os.environ['ARIS_TOOL']}"
        f", six arms in the scene always (--arms restricts who DRAWS, not who "
        f"is there)",
        "",
    ]
    return out


# ===========================================================================
# the three adapters
# ===========================================================================
def _segment(plan, stroke_id=0, kind="line"):
    """A certified `plan_stroke` result -> one `arm_program` programme entry.

    The shape is `allocate._entry`'s; a one-line run has no allocator to build
    it, and `writing.arm_program` reads exactly `plan`, `length` and
    `home_before` off it.
    """
    pts = np.asarray(plan["stroke"], float)
    return dict(stroke_id=int(stroke_id), color="black", kind=kind,
                s_range=(0.0, 1.0), direction=1, pts=pts,
                length=float(plan["arc_len"]), plan=plan)


def _payload(qtraj, segtraj, pens, margin, min_clearance, name):
    """The six-arm execution npz, without the animation arrays.

    `csail_schedule.payload` also writes `segpts_`/`segoff_`/`ink_*`, which say
    where ink was laid on a clock; a one-line programme has no tracer behind it
    and a fabricated copy of those arrays would be a lie a viewer would believe
    (`serialise_timeline.py` declines to rebuild them for the same reason).
    Everything `execute.program.from_schedule`, `recheck_timeline.recheck` and
    `export.pathway` read IS here.
    """
    arms = sorted(qtraj)
    M = len(qtraj[arms[0]])
    d = dict(fps=np.float64(FPS), dt=np.float64(DT), stride=np.int64(STRIDE),
             n_phases=np.int64(1), pause_s=np.float64(0.0),
             margin=np.float64(margin),
             min_clearance=np.float64(min_clearance),
             pause_total=np.float64(0.0),
             arms=np.array(arms, np.int64),
             drawing_arms=np.array(sorted(a for a in arms
                                          if (segtraj[a] >= 0).any()), np.int64),
             pen_ext=np.array([pens[a] for a in arms], float),
             sheet=np.array(fleet.SHEET, float),
             n_frames=np.int64(M), duration=np.float64((M - 1) / FPS),
             phase=np.zeros(M, np.int64),
             phase_start_s=np.array([0.0], float),
             phase_ink=np.array(["black"]),
             ink_names=np.array(["black"]),
             ink_palette=np.array(["#111111"]),
             ink_t=np.zeros(0, float), ink_arm=np.zeros(0, np.int64),
             ink_off=np.zeros(1, np.int64), ink_xyz=np.zeros((0, 3)),
             ink_hex=np.array([], dtype="<U7"),
             name=np.array([str(name)]))
    for a in arms:
        d[f"q_{a}"] = np.asarray(qtraj[a], float).astype(np.float32)
        d[f"seg_{a}"] = np.asarray(segtraj[a], np.int64)
        d[f"u_{a}"] = np.zeros(M, float)
    return d


def _plan_segment_json(plan, draw_s):
    """The certified plan as ONE phase-segment record. -> dict.

    The shape is `csail_allocate._phase_json`'s, which is the shape the pathway
    exporter and `program_schema.export_bundle` both read, and every number in
    it is the plan's own — nothing is invented here.  A one-line run has no
    allocator to write it, in the same way it has no allocator to write
    `_segment`'s programme entry.
    """
    return dict(
        seg=0, stroke_id=0, s_range=[0.0, 1.0], direction=1, flipped=False,
        length_m=float(plan["arc_len"]), color="black", kind="line",
        min_sigma=float(plan["min_sigma"]),
        min_margin=float(plan["min_margin"]),
        tip_err_m=float(plan["tip_err"]),
        max_lean_deg=float(plan.get("lean_deg", 0.0) or 0.0),
        tilt_cone_deg=float(TILT_MAX_DEG),
        draw_time_s=float(draw_s),
        plan_ok=(plan["status"] == "ok"),
        validated=bool((plan.get("validation") or {}).get("ok", False)),
        home_before=False,
        n_dense=int(plan.get("n_dense", 0)),
        n_knots=int(plan.get("n_knots", 0)),
        pts=[[float(x), float(y)] for x, y in np.asarray(plan["stroke"], float)])


def _program_json(arm, name, source, plan, pen_plan, pen_real, hover,
                  draw_s=0.0):
    """The label file the exporter reads.  Nothing load-bearing is in it."""
    seg = _plan_segment_json(plan, draw_s)
    return dict(
        rig=os.environ["ARIS_RIG"], tool=os.environ["ARIS_TOOL"],
        h_inv=float(fleet.H_INV_DEFAULT),
        arms_in_fleet=sorted(layout.FLEET_PROPOSED),
        sheet=list(fleet.SHEET), n_phases=1, two_pass=False,
        name=name, source=source,
        inks=["black"], palette=dict(black="#111111"),
        pens_mm={str(arm): 1000.0 * pen_real},
        plan_pen_ext_m=float(pen_plan), hover_m=float(hover),
        strokes=[dict(id=0, color="black", kind="line",
                      length=float(plan["arc_len"]))],
        totals=dict(traced_m=float(plan["arc_len"]),
                    drawn_m=float(plan["arc_len"]), dropped_m=0.0,
                    coverage_pct=100.0, n_strokes=1, n_segments=1, wall_s=0.0),
        phases=[dict(name="single line", ink="black", draw_speed=DRAW_SPEED,
                     arms={str(arm): [seg]}, dropped=[])],
        arms={str(arm): [seg]},
        colors={str(arm): "black"})


# ===========================================================================
# `line`
# ===========================================================================
class Refused(SystemExit):
    """A plain message and a non-zero exit.  Never a traceback at the rig."""

    def __init__(self, msg):
        super().__init__(f"REFUSED: {msg}")


def _xy(text, what):
    parts = [p for p in str(text).replace(",", " ").split() if p]
    if len(parts) != 2:
        raise Refused(f"--{what} wants X,Y in metres (canvas datum); got {text!r}")
    try:
        return np.array([float(parts[0]), float(parts[1])], float)
    except ValueError:
        raise Refused(f"--{what} wants two numbers; got {text!r}")


def _gate_numbers(rep, margin):
    """The gates as plain numbers, for the summary json and the one-liner."""
    pc = rep.get("paper_clearance") or {}
    fc = rep.get("frame_clearance") or {}
    cc = rep.get("column_clearance") or {}
    sc = rep.get("self_clearance") or {}
    jm = rep.get("joint_margin") or {}
    return dict(
        pair_margin_m=float(margin),
        min_inter_arm_m=float(rep["min_clearance"]),
        worst_pair=list(rep.get("worst_pair") or []),
        min_frame_m=(min(v[0] for v in fc.values()) if fc else None),
        min_column_m=(min(v[0] for v in cc.values()) if cc else None),
        min_self_m=(min(sc.values()) if sc else None),
        self_margin_m=float(rep.get("self_margin", 0.0)),
        frame_margin_m=float(rep.get("frame_margin", 0.0)),
        min_paper_chain_m=(min(v["chain"] for v in pc.values()) if pc else None),
        paper_chain_margin_m=float(rep.get("paper_chain_margin", 0.0)),
        min_paper_tip_m=(min(v["tip"] for v in pc.values()) if pc else None),
        paper_tip_margin_m=float(rep.get("paper_tip_margin", 0.0)),
        min_joint_margin_rad=(min(jm.values()) if jm else None),
        frame_failed=rep.get("frame_failed"), paper_failed=rep.get("paper_failed"),
        column_failed=rep.get("column_failed"), self_failed=rep.get("self_failed"),
        ok=bool(rep["ok"]))


def _one_liner(name, arm, g, dur, certified):
    def mm(v):
        return "  n/a" if v is None else f"{1000 * v:6.1f}"
    return (f"{'PASS' if certified else 'FAIL'}  {name}_{arm}  "
            f"inter-arm {mm(g['min_inter_arm_m'])} mm (gate "
            f"{1000 * g['pair_margin_m']:.0f})  "
            f"frame {mm(g['min_frame_m'])}  column {mm(g['min_column_m'])}  "
            f"self {mm(g['min_self_m'])}  "
            f"paper chain {mm(g['min_paper_chain_m'])} tip "
            f"{mm(g['min_paper_tip_m'])}  "
            f"joint {g['min_joint_margin_rad']:.4f}  {dur:.2f} s")


def _write_csv(pw, out_dir, stem):
    """`export.pathway.write_pathway` with this day's file names.

    The rows and the manifest are the exporter's own — `build_arm_pathway` has
    already run the contract's FK gate over the formatted text — and the only
    thing that differs is `<stem>_<arm>.csv` against the exporter's
    `<name>_arm<arm>.csv`, which is what the day asks for.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{stem}_{pw.arm_id}.csv"
    man_path = out_dir / f"{stem}_{pw.arm_id}.manifest.json"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(pathway.CSV_COLUMNS)
        w.writerows(pw.rows)
    man = dict(pw.manifest)
    man["csv"] = csv_path.name
    man_path.write_text(json.dumps(man, indent=1, default=str) + "\n")
    return csv_path, man_path


def plan_line(arm, p0, p1, name="line", hover=0.0, out_dir=OUT_DIR,
              write=True, verbose=True):
    """Plan, certify and (if it certifies) write ONE line.  -> dict.

    Raises `Refused` — a plain message and a non-zero exit — when the line is
    unreachable, uncertifiable by the stroke planner, unflyable from or to the
    park, or when the independent scene check fails.  Nothing is written in any
    of those cases: an uncertified line never gets a CSV.
    """
    arm = int(arm)
    fl, h = rt.fleet_for(None, None, "uniform")
    if arm not in fl:
        raise Refused(f"arm {arm} is not in rig {os.environ['ARIS_RIG']!r} "
                      f"(has {sorted(fl)})")
    spec = fl[arm]
    pen_real = float(frames.ext_of(None))
    pen_plan = pen_real + float(hover)
    pts = np.array([p0, p1], float)

    # ---- 1. the certified single-arm stroke planner -----------------------
    opts = dict(objective=pwl.OBJECTIVE, tilt_max_deg=TILT_MAX_DEG,
                pen_ext=pen_plan)
    plan = stroke_api.plan_stroke(pts, spec, opts)
    if verbose:
        print(f"  plan_stroke: {stroke_api.plan_summary(plan)}")
    if plan["status"] != "ok":
        raise Refused(
            f"arm {arm} cannot draw ({p0[0]:.3f}, {p0[1]:.3f}) -> "
            f"({p1[0]:.3f}, {p1[1]:.3f}): plan_stroke says "
            f"{plan['status']} / {plan.get('reason')!r}"
            + (f"; it certifies {plan['s_star']:.3f} of the line "
               f"(reach {plan.get('s_reach', 0.0):.3f})"
               if plan["status"] == "split" else "")
            + ".  Move the line, or give it to the other arm.")

    # ---- 2. the solo programme: park -> hover -> down -> line -> up -> park
    try:
        prog = writing.arm_program(
            spec, [_segment(plan)], draw_speed=DRAW_SPEED,
            transit_speed=TRANSIT_SPEED, h_inv=fleet.H_INV_DEFAULT,
            pen_ext=pen_plan, q_start=spec.q_seed, park=writing.PARK_HOME)
    except writing.PaperRefused as e:
        raise Refused(
            f"{e}.  The LINE certifies but the pen-up leg to or from the park "
            "does not clear the paper plane — this is the entry/go-home, which "
            "no tour chose.  Move the line.")
    samp = writing.uniform_samples(prog, DT)
    M = int(samp["n"])

    # ---- 3. the six-arm scene: the mover, and five arms at their parks -----
    q = {a: (np.asarray(samp["q"], float) if a == arm
             else np.tile(np.asarray(fl[a].q_seed, float), (M, 1)))
         for a in sorted(fl)}
    seg = {a: (np.asarray(samp["seg"], np.int64) if a == arm
               else np.full(M, -1, np.int64)) for a in sorted(fl)}
    # graded with the REAL pen at the REAL height — for a hover line that is the
    # whole point: the plan used a 30 mm longer pen, the certificate does not.
    pens = {a: pen_real for a in sorted(fl)}
    margin = float(coordination.PAIR_MARGIN)
    rep = scene_check.check_timeline(
        q, DT, margin, programs=None, h_inv=h, pen_ext=pens, sub=SUB,
        verbose=False, fleet=fl, drawing={a: seg[a] >= 0 for a in sorted(fl)})
    g = _gate_numbers(rep, margin)
    dur = float(prog["duration"])
    if verbose:
        for ln in rt.summarise(rep, margin):
            print("  " + ln)
    if not rep["ok"]:
        raise Refused(
            f"the independent scene check FAILS for this line "
            f"(frame {rep.get('frame_failed')}, paper {rep.get('paper_failed')}, "
            f"column {rep.get('column_failed')}, self {rep.get('self_failed')}, "
            f"min inter-arm {1000 * rep['min_clearance']:.1f} mm against a "
            f"{1000 * margin:.0f} mm gate).  Nothing written.")

    T = np.asarray(spec.T_world_base(), float)
    summary = dict(
        name=name, arm=arm, certified=True,
        from_xy=[float(p0[0]), float(p0[1])], to_xy=[float(p1[0]), float(p1[1])],
        hover_m=float(hover),
        rig=os.environ["ARIS_RIG"], tool=os.environ["ARIS_TOOL"],
        h_m=float(h), duration_s=dur,
        draw_speed_m_s=DRAW_SPEED, draw_s=float(prog["draw_s"]),
        transit_speed_m_s=TRANSIT_SPEED, fps=FPS, stride=STRIDE, sub=SUB,
        arc_len_m=float(plan["arc_len"]),
        plan=dict(min_sigma=float(plan["min_sigma"]),
                  min_margin=float(plan["min_margin"]),
                  tip_err_m=float(plan["tip_err"]),
                  sheet=int(plan["sheet"]), n_dense=int(plan["n_dense"]),
                  lean_deg=float(plan.get("lean_deg", 0.0)),
                  phi_rad=(float(plan["phi"]) if np.ndim(plan.get("phi")) == 0
                           else None),
                  tilt_max_deg=TILT_MAX_DEG, objective=pwl.OBJECTIVE),
        tool_tip=dict(
            pen_lat_m=float(frames.lat_of(None)),
            pen_ext_m=pen_real, plan_pen_ext_m=pen_plan,
            hand_tcp_offset_m=[float(frames.lat_of(None)), 0.0, pen_real],
            flange_frame_m=[float(frames.lat_of(None)), 0.0,
                            float(frames.D_HAND_TCP) + pen_real],
            source="USER-SPECIFIED, read off a photograph; no touchdown"),
        base=dict(T_world_base=[float(v) for v in T.reshape(-1)],
                  translation_m=[float(v) for v in T[:3, 3]],
                  yaw_deg=float(np.degrees(np.arctan2(T[1, 0], T[0, 0]))),
                  mount=str(spec.mount)),
        seam_bars=dict(on=bool(mounts.SEAM_POSTS_ON), source=mounts.SEAM_SOURCE),
        frozen_arms={str(a): [float(v) for v in fl[a].q_seed]
                     for a in sorted(fl) if a != arm},
        gates=g)

    if not write:
        return dict(summary=summary, report=rep, payload=None, csv=None)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{name}_{arm}"
    npz_path = out_dir / f"{stem}.npz"
    pj = _program_json(arm, f"day1 line, arm {arm}", "scripts/day1.py line",
                       plan, pen_plan, pen_real, hover,
                       draw_s=float(prog["draw_s"]))
    pj_path = out_dir / f"{stem}_program.json"
    pj_path.write_text(json.dumps(pj, indent=1) + "\n")
    # The TARGET polyline, under the name `program_schema._stroke_polylines`
    # looks for.  It is the two points the line was asked for and nothing else;
    # without it the 3D viewer can animate the arm but cannot draw the line it
    # is drawing.
    (out_dir / f"{stem}_strokes.json").write_text(json.dumps(dict(
        strokes=[dict(id=0, color="black", kind="line",
                      length=float(plan["arc_len"]),
                      pts=[[float(p0[0]), float(p0[1])],
                           [float(p1[0]), float(p1[1])]])]), indent=1) + "\n")
    np.savez_compressed(npz_path, **_payload(q, seg, pens, margin,
                                             float(rep["min_clearance"]), stem))

    # ---- 4. the impedance pathway CSV, through the shipped exporter --------
    from aris_sixarm.execute.program import from_schedule
    fp = from_schedule(npz_path, pj_path)
    z = np.load(npz_path, allow_pickle=False)
    pw = pathway.build_arm_pathway(
        fp, z, arm, spec, tool=os.environ["ARIS_TOOL"],
        rig=os.environ["ARIS_RIG"], intensity=1.0,
        # a hover line is deliberately NOT on the paper, so its draw rows carry
        # the certified pose's own z rather than the plane it does not touch
        z_mode=("fk" if hover > 0 else "plane"),
        h_inv=float(pj["h_inv"]), program_json=pj,
        source_paths=dict(schedule=str(npz_path), program=str(pj_path)))
    csv_path, man_path = _write_csv(pw, out_dir, name)
    summary["files"] = dict(npz=str(npz_path), csv=str(csv_path),
                            manifest=str(man_path), program=str(pj_path))
    summary["csv"] = dict(n_rows=pw.stats["n_rows"],
                          n_draw_rows=pw.stats["n_draw_rows"],
                          n_travel_rows=pw.stats["n_travel_rows"],
                          draw_length_m=pw.stats["draw_length_m"],
                          paper_z_base_m=pw.stats["paper_z_base_m"],
                          max_row_dq_rad=pw.stats["max_row_dq_rad"],
                          joint_columns=True)
    (out_dir / f"{stem}.json").write_text(
        json.dumps(summary, indent=1, default=str) + "\n")
    summary["files"]["summary"] = str(out_dir / f"{stem}.json")
    return dict(summary=summary, report=rep, payload=npz_path, csv=csv_path,
                pathway=pw)


def cmd_line(a):
    p0, p1 = _xy(a.frm, "from"), _xy(a.to, "to")
    for ln in assumptions([int(a.arm)]):
        print(ln)
    hov = float(a.hover or 0.0)
    print(f"arm {a.arm}: ({p0[0]:.4f}, {p0[1]:.4f}) -> ({p1[0]:.4f}, {p1[1]:.4f}) m"
          f", {1000 * float(np.linalg.norm(p1 - p0)):.1f} mm"
          + (f", HOVER {1000 * hov:.0f} mm above the paper — no ink, no contact"
             if hov > 0 else ", ON THE PAPER"))
    print(f"  the other five arms stand at their parks for the whole programme; "
          f"every gate is graded against all six.")
    r = plan_line(int(a.arm), p0, p1, name=a.name, hover=hov,
                  out_dir=Path(a.out))
    s = r["summary"]
    print()
    print(_one_liner(a.name, int(a.arm), s["gates"], s["duration_s"], True))
    print(f"  wrote {s['files']['npz']}")
    print(f"        {s['files']['csv']}  ({s['csv']['n_rows']} rows, "
          f"{s['csv']['n_draw_rows']} draw + {s['csv']['n_travel_rows']} travel, "
          f"q1..q7 on every row)")
    print(f"        {s['files']['summary']}")
    if hov > 0:
        print(f"  the pen rides {1000 * hov:.0f} mm off the paper: measure it "
              f"with a ruler at three points before ANY pen-down run.")
    return 0


# ===========================================================================
# `word`
# ===========================================================================
ALT_INSTRUCTIONS = """RUN ORDER — ALTERNATING (rung C, the deliverable):
  1. start arm 31's block.  Arm 71 stands at its park and does not move.
  2. WAIT for arm 31 to finish AND to be back at its park.  Look at it.
  3. only then start arm 71's block.
  The two arms are NEVER both in motion, so the certificate does not depend on
  a fleet clock — and there is no cross-process fleet clock.
  The seam between the blocks has no step in either arm: nothing to reposition."""

CONCURRENT_INSTRUCTIONS = """RUN ORDER — CONCURRENT (rung D, the only collision risk of the day):
  1. start arm 31 FIRST and watch it actually draw.
  2. start arm 71 NO SOONER THAN 2 s later.  Never earlier, and NEVER at 1 s.
  ARM 71 MUST NEVER START EARLY.  Measured: at -10 s the arms overlap by
  119 mm; at -0.5 s by 25 mm.  Every negative skew is a COLLISION.
  AND "a second later" IS THE WRONG INSTRUCTION: the notch at +1.00/+1.25 s
  bottoms at 35.8 mm against a 50 mm gate.  +1.75 s to +10 s clears by 61 mm
  or better at all 34 sampled points.  Two seconds, minimum.
  One person on the e-stop, watching the MIDDLE of the paper, doing nothing else.
  Run rung C cleanly first, or skip this."""

HOVER_INSTRUCTIONS = """RUN ORDER — HOVER (rung A, the first powered motion):
  This programme NEVER TOUCHES THE PAPER, and it NEVER MOVES ARM 31 — measured,
  arm 31 draws 0 segments and 0.00 m in the main pass and in all six residual
  passes.  Run it as ARM 71's dry run (`--solo 71`).
  It has SIX phases, so five barriers to acknowledge, and its frame gate sits
  at 50.9 mm against a 50 mm gate on arm 31 standing at its park at t = 0.
  Arm 31 gets NO hover pass from this file.  Decide deliberately, before the
  day starts, whether arm 31's first powered motion is a supervised jog or the
  drawing programme itself."""

PASS_CRITERIA = {
    "alt": """PASSES WHEN:
  the word is legible; both hand-overs happened where the plan says (x ~ 0.90);
  no pair got closer than the re-checked minimum minus the calibration
  allowance; no reflex, no tracking fault, no gate rejection.
LOG: where 31 stops and where 71 starts, and the realised gap between the pens
  at that moment; the 'k'/'n' junction across x ~ 0.90; total ink against plan;
  makespan against the file.""",
    "concurrent": """PASSES WHEN:
  both arms complete; the realised inter-arm clearance stays above the
  re-checked minimum minus the calibration allowance; the measured skew stayed
  inside the passing band for the WHOLE run.
LOG: both start timestamps to the second, and the realised clearance.""",
    "hover": """PASSES WHEN:
  the arm completes with no reflex, no tracking fault, no gate rejection, and
  the measured tip height is 28 +/- 5 mm everywhere.  The planner's own minimum
  over this programme is 27.6 mm — expect the ruler to say 28, not 30.
  IF THE TIP HEIGHT IS NOT RIGHT, STOP AND DO NOT DRAW: the discrepancy is the
  tool transform, the mounting height or the paper plane, and you now have a
  number for it.
LOG: joint tracking error per arm against the 20 mrad latching watchdog; the
  joint-5 static offset (the lab measured ~9 mrad on 2026-09-10 — if joint 5 is
  quiet today, say so); the tip height at three points along the word.""",
}


def check_replan(schedule_json, arms=ARMS, coverage_gate=COVERAGE_GATE):
    """Is a re-planned word a success?  -> (ok, lines).

    THE CONDUCTED COVERAGE, OUT OF THE SCHEDULE JSON, AND NOTHING ELSE.  The
    log prints the ALLOCATOR's coverage — what the planner believes it could
    cover — and `--skip-unconductable` is precisely the flag that lets the two
    diverge quietly: the h = 0.850 run logged `COVERAGE 84.2074 %` over a file
    whose conducted `coverage` is 0.0123 (HARDWARE_DAY1 §3).  And a 100 %
    coverage drawn entirely by one arm is not the demonstration either, so both
    arms must have metres.
    """
    d = json.loads(Path(schedule_json).read_text())
    cov = float(d.get("coverage", 0.0))
    metres = {int(k): float(v) for k, v in (d.get("arm_metres") or {}).items()}
    lines = [f"  conducted coverage {100 * cov:.4f} % "
             f"(gate {100 * coverage_gate:.2f} %) — read out of "
             f"{schedule_json}, NOT out of the log"]
    for a in arms:
        lines.append(f"  arm {a}: {metres.get(a, 0.0):.4f} m drawn, "
                     f"{int((d.get('arm_segments') or {}).get(str(a), 0))} segments")
    if d.get("skipped_phases"):
        lines.append(f"  ! skipped phases {d['skipped_phases']} "
                     f"({float(d.get('skipped_m', 0.0)):.3f} m left undrawn)")
    ok = cov >= coverage_gate and all(metres.get(a, 0.0) > 0.0 for a in arms)
    if not ok:
        why = []
        if cov < coverage_gate:
            why.append(f"conducted coverage {100 * cov:.4f} % is below "
                       f"{100 * coverage_gate:.2f} %")
        dead = [a for a in arms if metres.get(a, 0.0) <= 0.0]
        if dead:
            why.append(f"arm(s) {dead} draw NOTHING — this is no longer a "
                       "two-arm demonstration")
        lines.append("  REFUSED: " + "; and ".join(why))
    return ok, lines


def replan(strokes, out_name, arms=ARMS, outdir=ROOT / "out", verbose=True):
    """Re-plan the word with yesterday's exact job params. -> (ok, lines)."""
    strokes = Path(strokes)
    if not strokes.exists():
        # `out/unknown_strokes.json` is the name the day plan uses; on a clone
        # with no out/ the same file is the tracked one.  See `asset`.
        cand = asset(strokes.name)
        if not cand.exists():
            raise Refused(f"no strokes file at {strokes}")
        strokes = cand
    flags = list(REPLAN_FLAGS)
    # the ONE substitution: the atlas is a directory, and on a clone with no
    # out/ it is the tracked copy.  Every other flag is verbatim.
    flags[flags.index("--atlas") + 1] = str(
        asset("atlas_proposed_h0970_lat0860"))
    cmd = [sys.executable, str(ROOT / "scripts" / "draw.py"), str(strokes),
           "--out", out_name, "--outdir", str(outdir)] + flags
    env = dict(os.environ, ARIS_RIG="proposed", ARIS_TOOL="lateral")
    print("  " + " ".join(cmd))
    print("  (the 0.970 conduct took ~700 s; a cold cache takes ~3700 s)")
    r = subprocess.run(cmd, cwd=str(ROOT), env=env)
    sched = Path(outdir) / f"{out_name}_schedule.json"
    if not sched.exists():
        raise Refused(f"the conductor exited {r.returncode} and wrote no "
                      f"{sched} — there is no schedule to believe or disbelieve")
    return check_replan(sched, arms)


def cmd_word(a):
    arms = [int(x) for x in str(a.arms).replace(",", " ").split()]
    for ln in assumptions(arms):
        print(ln)

    if a.replan:
        if not a.strokes:
            raise Refused("--replan needs --strokes FILE (the word, as a stroke "
                          "case file; scripts/text_strokes.py writes one)")
        print("RE-PLANNING with the conductor's job parameters from 2026-09-15:")
        ok, lines = replan(a.strokes, a.name or "unknown_day1_replan", arms)
        print()
        for ln in lines:
            print(ln)
        if not ok:
            raise SystemExit(2)
        print("  re-plan ACCEPTED: full coverage and both arms draw.")
        return 0

    v = VARIANTS[a.variant]
    npz = Path(v["npz"])
    if not npz.exists():
        raise Refused(f"no asset at {npz}; this subcommand does not plan — "
                      f"it re-checks what was conducted yesterday")
    print(f"VARIANT {a.variant}: {v['what']}")
    print(f"  asset {npz}")
    print(f"  re-deriving the certificate ON SITE, today, with the independent "
          f"whole-timeline checker:")
    print()
    rep, margin = rt.recheck(str(npz), sub=SUB)
    print()
    for ln in rt.summarise(rep, margin):
        print(ln)

    OUT = Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"unknown_{a.variant}_recheck.json").write_text(json.dumps(
        {k: val for k, val in rep.items() if k != "segments"},
        default=str, indent=1) + "\n")
    print(f"\n  wrote {OUT / f'unknown_{a.variant}_recheck.json'}")

    if not rep["ok"]:
        raise Refused(f"{npz} does NOT re-certify today.  Do not fly it.  "
                      f"No CSV copied.")

    copied, missing = [], []
    for arm in arms:
        src = asset(f"pathways/{v['csv_stem']}_arm{arm}.csv")
        man = src.parent / f"{src.stem}.manifest.json"
        if not src.exists():
            missing.append(str(src))
            continue
        dst = OUT / f"unknown_{a.variant}_{arm}.csv"
        shutil.copyfile(src, dst)
        copied.append(dst)
        if man.exists():
            shutil.copyfile(man, OUT / f"unknown_{a.variant}_{arm}.manifest.json")
    for d in copied:
        n = sum(1 for _ in open(d)) - 1
        print(f"  copied {d}  ({n} rows, q1..q7 on every row)"
              + ("   ! HEADER ONLY — this arm draws NOTHING in this variant"
                 if n == 0 else ""))
    for m in missing:
        print(f"  ! NO CSV at {m} — that arm was never exported for this "
              f"variant.  This file does not make one (no new formats); it is "
              f"one command:\n"
              f"      ARIS_RIG=proposed ARIS_TOOL=lateral python3 -m "
              f"aris_sixarm.export.pathway --schedule {v['npz']} "
              f"--program out/{v['csv_stem']}_program.json --arms <id> "
              f"--out out/pathways --name {v['csv_stem']}"
              + ("  --z-mode fk" if a.variant == "hover" else ""))

    print()
    print({"alt": ALT_INSTRUCTIONS, "concurrent": CONCURRENT_INSTRUCTIONS,
           "hover": HOVER_INSTRUCTIONS}[a.variant])
    print()
    print(PASS_CRITERIA[a.variant])
    print()
    print("ABORT RULE, EVERY RUNG: the physical e-stop.  The software gate "
          "brakes at 2 rad/s^2 and the watchdog latches at 20 mrad, and neither "
          "of those is the abort path.")
    return 0


# ===========================================================================
def build_parser():
    ap = argparse.ArgumentParser(
        prog="scripts/day1.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Hardware day 1: draw one line with one arm, then run the "
                    "word with two.  Nothing here plans the word — that was "
                    "done and certified yesterday.",
        epilog="""examples
  # one arm, one straight line, on the paper, certified end to end
  scripts/day1.py line --arm 31 --from 0.50,1.70 --to 0.65,1.70 --name probe

  # the same line 30 mm ABOVE the paper — the first powered motion
  scripts/day1.py line --arm 31 --from 0.50,1.70 --to 0.65,1.70 --hover 0.03 \\
      --name probe_hover

  # the word, alternating: re-check yesterday's certified file, today
  scripts/day1.py word --variant alt

coordinates are METRES in the canvas datum: x across the 1.8034 m sheet, y
along the 3.63064 m one.  The seam / the middle row's own line is y = 1.8153.
Arm 31's J1 axis is at x = 0.5967, arm 71's at x = 1.2067.

everything is written to out/day1/.  An uncertified line writes NOTHING.""")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser(
        "line", formatter_class=argparse.RawDescriptionHelpFormatter,
        help="plan + certify + export ONE straight line for ONE arm",
        description="One straight line, one arm, the other five frozen at "
                    "their parks.  Plans with the certified stroke planner, "
                    "builds the solo programme park -> hover -> down -> line "
                    "-> up -> park, grades it with the independent checker, "
                    "and writes out/day1/<name>_<arm>.{npz,csv,json}.  Refuses "
                    "(non-zero, no CSV) if the line will not certify.")
    p.add_argument("--arm", type=int, required=True, choices=sorted(ARMS))
    p.add_argument("--from", dest="frm", required=True, metavar="X,Y",
                   help="start of the line, metres, canvas datum")
    p.add_argument("--to", dest="to", required=True, metavar="X,Y",
                   help="end of the line, metres, canvas datum")
    p.add_argument("--name", default="line", help="file stem (default: line)")
    p.add_argument("--hover", type=float, nargs="?", const=HOVER_DEFAULT,
                   default=0.0, metavar="M",
                   help=f"fly the line this far ABOVE the paper instead of on "
                        f"it (default when the flag is given: "
                        f"{HOVER_DEFAULT:g} m).  Plans with a pen that much "
                        f"longer and grades with the real one.")
    p.add_argument("--out", default=str(OUT_DIR))
    p.set_defaults(func=cmd_line)

    w = sub.add_parser(
        "word", formatter_class=argparse.RawDescriptionHelpFormatter,
        help="re-check yesterday's certified word and print how to run it",
        description="Does NOT plan.  Re-runs the independent whole-timeline "
                    "check on a certified 2026-09-15 asset, copies the per-arm "
                    "pathway CSVs into out/day1/, and prints the run order and "
                    "the pass criteria.  --replan re-plans from a strokes file "
                    "with the conductor's exact job parameters and refuses to "
                    "report success on anything but the conducted coverage.")
    w.add_argument("--variant", default="alt", choices=sorted(VARIANTS),
                   help="alt (default, the deliverable) | concurrent | hover")
    w.add_argument("--arms", default="31,71")
    w.add_argument("--out", default=str(OUT_DIR))
    w.add_argument("--replan", action="store_true",
                   help="re-plan the word from --strokes with yesterday's job "
                        "parameters (~700 s warm, ~3700 s cold)")
    w.add_argument("--strokes", default=None,
                   help="stroke case file for --replan, e.g. "
                        "out/unknown_strokes.json")
    w.add_argument("--name", default=None, help="--replan output stem")
    w.set_defaults(func=cmd_word)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
