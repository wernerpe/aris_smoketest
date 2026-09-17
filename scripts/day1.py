#!/usr/bin/env python3
"""HARDWARE DAY 1 — one front door, two subcommands.  Read this at the rig.

    scripts/day1.py line --arm 31 --from 0.50,1.70 --to 0.65,1.70 --name probe
    scripts/day1.py word --arm 31 --hover
    scripts/day1.py word --arm 31
    scripts/day1.py word --variant alt

`line` is rung B of `docs/HARDWARE_DAY1.md` cut down to ONE straight line for
ONE arm: it plans it with the certified single-arm stroke planner
(`aris_sixarm.stroke_api.plan_stroke`), lays the solo programme down with
`aris_sixarm.writing.arm_program` (park -> hover -> down -> line -> up -> park),
grades the whole thing with the INDEPENDENT checker (`scene_check.check_timeline`,
every gate, the seam bars in, the other five arms standing at their parks), and
only then writes the timeline, the impedance pathway CSV and a summary.  An
uncertifiable line is a non-zero exit and no CSV.

`word --arm 31` (or 71) is the SOLO WORD — the rung between the two.  It is
`line` with thirteen strokes instead of one and an ordering step in the middle,
and every other thing about it is `line`'s: the same certified single-arm
planner per stroke, the same `arm_program`, the same independent whole-timeline
check with the other five arms frozen at their parks, the same three files.
The word comes from `scripts/text_strokes.py` — the vendored single-stroke
Hershey font — sized to `--width` and centred on THAT ARM's own J1 axis, with
its baseline `--dy` off the seam line (default -0.10 m, because arm 31 refuses
lines exactly on the seam: the go-home leg does not clear the paper plane
there).  The strokes are ordered by `aris_sixarm.sequence` over the REAL
transit cost — the same Held-Karp the conductor runs, not a left-to-right
sweep — so the pen-up legs between letters are the ones the timeline pays.
If any stroke will not certify, NOTHING is written and the refusals are named;
`--allow-partial` writes the certified subset instead and says so.

`word` WITHOUT `--arm` is rungs C/D and it PLANS NOTHING.  It takes yesterday's
certified
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
import re
import shutil
import subprocess
import sys
import time
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

from aris_sixarm import (allocate, coordination, fleet, frames,  # noqa: E402
                         layout, mounts, pwl, scene_check, sequence,
                         stroke_api, writing)
from aris_sixarm.export import pathway                          # noqa: E402

import recheck_timeline as rt                                   # noqa: E402
import text_strokes                                             # noqa: E402

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

# --- THE ONLY PLACE THIS FILE KNOWS ABOUT ANOTHER MACHINE ------------------
# `send` delivers a pathway CSV to the box that runs the arms and prints the
# one command that draws it.  Everything machine-specific is here, it is all
# strings, and every one of them is overridable from the command line or the
# environment — so a clone on the robot PC edits this block and nothing else.
#
# The values are `briefings/CONTROL_STACK_line_to_joint_torques_2026-09-09.md`
# §1/§8: the OPERATOR is the Dell Precision 7960 on the direct NIC, the CSV
# lands under `~/RTff/pathway_persist/<folder>/`, and the supervisor
# `~/RTff/draw_rtff_supervised.sh <csv> <fmin> <fmax> <levels> <fresh|resume>`
# is what turns it into motion (it runs the ladder gate, switches to impedance,
# and calls `rtff_pathway_exec.py --csv ...`).  The arm is chosen by the
# `ARM_ID` environment variable — it is the DDS domain, not a flag and not an
# IP.  CHECK THE ADDRESS BEFORE THE FIRST SEND; the briefing also names a
# RETIRED box at 192.168.50.4 whose `~/RTff` is a stale copy, and this file
# must never point there.
OPERATOR = dict(
    host=os.environ.get("ARIS_OPERATOR", "diemut@192.168.50.2"),
    store="~/RTff/pathway_persist",          # where the CSV lands
    supervisor="~/RTff/draw_rtff_supervised.sh",
    stack_check="~/RTff/aris_hold.sh",       # `aris_hold.sh stack <N>`
    force=("1.0", "2.5", "5"),               # <fmin> <fmax> <levels>, the
    #   executor's own defaults (--force-min/--force-max/--force-levels)
    mode="fresh",                            # NOT `resume`: the runner decides
    #   fresh vs resume from a progress file, and a new CSV dropped next to an
    #   old checkpoint is picked up as a resume of the old one
)

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


def _plan_segment_json(plan, draw_s, seg=0, stroke_id=0, kind="line",
                       direction=1, flipped=False):
    """The certified plan as ONE phase-segment record. -> dict.

    The shape is `csail_allocate._phase_json`'s, which is the shape the pathway
    exporter and `program_schema.export_bundle` both read, and every number in
    it is the plan's own — nothing is invented here.  A one-line run has no
    allocator to write it, in the same way it has no allocator to write
    `_segment`'s programme entry.

    `seg` is the index of this segment in the arm's DRAW ORDER, which is what
    `export.pathway._program_stroke_labels` and `program_schema` both key on;
    `stroke_id` names the artwork stroke it is a piece of, and for a solo word
    those two differ the moment the sequencer reorders anything.
    """
    return dict(
        seg=int(seg), stroke_id=int(stroke_id), s_range=[0.0, 1.0],
        direction=int(direction), flipped=bool(flipped),
        length_m=float(plan["arc_len"]), color="black", kind=kind,
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


def _program_json(arm, name, source, segs_json, strokes_json, pen_plan,
                  pen_real, hover, phase_name="single line", wall_s=0.0):
    """The label file the exporter reads.  Nothing load-bearing is in it.

    `segs_json` are `_plan_segment_json` records IN DRAW ORDER; `strokes_json`
    are the artwork strokes they are pieces of (`id`, `color`, `kind`,
    `length`).  One line is the one-segment, one-stroke case of both.
    """
    drawn = float(sum(s["length_m"] for s in segs_json))
    traced = float(sum(s["length"] for s in strokes_json))
    return dict(
        rig=os.environ["ARIS_RIG"], tool=os.environ["ARIS_TOOL"],
        h_inv=float(fleet.H_INV_DEFAULT),
        arms_in_fleet=sorted(layout.FLEET_PROPOSED),
        sheet=list(fleet.SHEET), n_phases=1, two_pass=False,
        name=name, source=source,
        inks=["black"], palette=dict(black="#111111"),
        pens_mm={str(arm): 1000.0 * pen_real},
        plan_pen_ext_m=float(pen_plan), hover_m=float(hover),
        strokes=list(strokes_json),
        totals=dict(traced_m=traced, drawn_m=drawn,
                    dropped_m=max(0.0, traced - drawn),
                    coverage_pct=(100.0 * drawn / traced if traced > 0 else 0.0),
                    n_strokes=len(strokes_json), n_segments=len(segs_json),
                    wall_s=float(wall_s)),
        phases=[dict(name=phase_name, ink="black", draw_speed=DRAW_SPEED,
                     arms={str(arm): list(segs_json)}, dropped=[])],
        arms={str(arm): list(segs_json)},
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


def _rates(Q, dt):
    """Peak |dq/dt| and |d2q/dt2| per joint. -> (7,), (7,)."""
    Q = np.asarray(Q, float).reshape(-1, 7)
    if len(Q) < 2:
        return np.zeros(7), np.zeros(7)
    v = np.diff(Q, axis=0) / float(dt)
    a = (np.diff(v, axis=0) / float(dt)) if len(v) > 1 else np.zeros((1, 7))
    return np.abs(v).max(axis=0), np.abs(a).max(axis=0)


def speed_audit(prog, q_rows, fps=FPS, khz_dt=0.001):
    """Is the joint reference inside the FR3's velocity limits? -> dict.

    THE CSV IS A JOINT REFERENCE AND SOMEBODY IS GOING TO STREAM IT.  Every
    other gate in this file is about where the arm is; this one is about how
    fast it gets there, and it is the one a stiff controller turns into torque.
    `frames.QD_MAX` is the FR3's own per-joint limit and `writing.QD_FRAC`
    (0.30) is the fraction the pacer aims at, so a healthy programme reads
    about a third of the limit and anything near 1.0 is a bug upstream, not a
    tight day.

    TWO READINGS, because they answer different questions:
      `at_csv_rows`   consecutive CSV rows one frame period apart.  This is
                      what a controller that consumes the file row by row at
                      the planning rate sees.  The synthesized lift ramps at
                      the ends of a stroke carry no time of their own and are
                      charged the same period, which OVER-states their speed —
                      conservative on purpose.
      `at_1khz`       the programme's own waypoints linearly interpolated onto
                      a 1 ms grid, which is the stream rate of the deployed
                      stack.  Linear interpolation makes the velocity
                      piecewise-constant, so the peak speed matches the
                      waypoint intervals and the ACCELERATION is the step
                      between two of them divided by a millisecond — an
                      impulse, reported for information and not gated.
    """
    lim = np.asarray(frames.QD_MAX, float).reshape(7)
    t = np.asarray(prog["t"], float)
    Q = np.asarray(prog["q"], float).reshape(-1, 7)
    grid = np.arange(0.0, float(t[-1]) + khz_dt, khz_dt) if len(t) > 1 \
        else np.zeros(1)
    Qi = np.column_stack([np.interp(grid, t, Q[:, j]) for j in range(7)])
    out = {}
    for key, (QQ, dt) in dict(
            at_csv_rows=(np.asarray(q_rows, float).reshape(-1, 7), 1.0 / fps),
            at_1khz=(Qi, khz_dt)).items():
        v, a = _rates(QQ, dt)
        frac = v / lim
        out[key] = dict(
            dt_s=float(dt), n_samples=int(len(QQ)),
            peak_qd_rad_s=[float(x) for x in v],
            frac_of_limit=[float(x) for x in frac],
            worst_joint=int(np.argmax(frac)) + 1,
            worst_frac=float(frac.max()),
            peak_qdd_rad_s2=[float(x) for x in a],
            ok=bool((frac <= 1.0).all()))
    out.update(qd_max_rad_s=[float(x) for x in lim],
               qd_frac_target=float(writing.QD_FRAC),
               ok=bool(out["at_csv_rows"]["ok"] and out["at_1khz"]["ok"]),
               note="q1..q7 on every CSV row are THE PLANNER'S OWN redundancy "
                    "resolution for that waypoint — the configuration the "
                    "certified plan chose, carried alongside the Cartesian "
                    "pose, not re-solved by the controller.")
    return out


def _speed_line(sa):
    """The joint-speed verdict, as one line. -> str."""
    c, k = sa["at_csv_rows"], sa["at_1khz"]
    return (f"  joint speed {'OK ' if sa['ok'] else 'OVER'}  "
            f"rows {100 * c['worst_frac']:5.1f} % of limit (worst j"
            f"{c['worst_joint']})  1 kHz {100 * k['worst_frac']:5.1f} % (worst j"
            f"{k['worst_joint']})  peak qdd {max(k['peak_qdd_rad_s2']):.0f} "
            f"rad/s^2 @1 kHz, {max(c['peak_qdd_rad_s2']):.0f} @rows")


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
    pj = _program_json(
        arm, f"day1 line, arm {arm}", "scripts/day1.py line",
        [_plan_segment_json(plan, float(prog["draw_s"]))],
        [dict(id=0, color="black", kind="line",
              length=float(plan["arc_len"]))],
        pen_plan, pen_real, hover)
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
# `word --arm N` — the SOLO word.  `line` with thirteen strokes.
# ===========================================================================
WORD = "unknown"
WORD_WIDTH = 0.55          # m of ink across, centred on the arm's own J1 axis
WORD_HEIGHT = 0.10         # m x-height: how tall 'u', 'n', 'o', 'w' come out
WORD_DY = -0.10            # m off the seam line.  NOT zero, and not a taste:
#   arm 31 refuses a line whose baseline is exactly on the seam (y = 1.8153) —
#   the pen-up leg to or from its park does not clear the paper plane there.
#   `tests/test_day1.py` pins that refusal, `docs/HARDWARE_DAY1.md` measures it,
#   and this default is the one number that keeps a person from meeting it.


def _word_geometry(spec, width=WORD_WIDTH, height=WORD_HEIGHT, dy=WORD_DY,
                   word=WORD):
    """The word as polylines in the canvas datum. -> (polys, info, x0, x1, y).

    Centred in x on THIS ARM's J1 axis and baselined `dy` off the seam line,
    both read off the arm's own base transform rather than restated here, so a
    layout change moves the word instead of leaving it behind.
    `text_strokes.layout` is the single-stroke Hershey font and the only place
    a letter shape is defined.
    """
    T = np.asarray(spec.T_world_base(), float)
    cx, seam = float(T[0, 3]), float(T[1, 3])
    x0, x1 = cx - 0.5 * float(width), cx + 0.5 * float(width)
    y = seam + float(dy)
    polys, info = text_strokes.layout(word, float(height), x0, x1, y)
    return polys, info, x0, x1, y


def _draw_times(prog, n):
    """Seconds the timeline spends inside each segment. -> list[float]."""
    t = np.asarray(prog["t"], float)
    s = np.asarray(prog["seg"], int)
    out = []
    for k in range(n):
        m = s == k
        out.append(float(t[m].max() - t[m].min()) if m.any() else 0.0)
    return out


SEQ_RETRIES = 8            # orders to try against the cheap matrix before the
#   expensive one.  Each retry is one Held-Karp solve and one `arm_program`
#   (~0.8 s together); building the routed matrix is ~100 s, so the retries are
#   free by comparison and the fallback is still there behind them.

# `writing.arm_program` refuses a pen-up leg it cannot fly with
# "arm 31: transit at segment 2 cannot clear the paper plane" — the `what` and
# the `k` in `writing.arm_program.need`.  That names ONE edge of the tour, and
# marking exactly that edge infinite is what the routed matrix would have done
# for it, at the cost of one solve instead of a hundred seconds of routing.
_REFUSAL = re.compile(
    r"(entry|transit|go-home|final lift|retreat) at segment (\d+)")


def _refused_edge(msg, tour, n):
    """The cost-matrix cell a `PaperRefused` names. -> (a, b) or None.

    `tour` is `sequence.tour_of`'s: the depot at index 0, then one node per
    segment IN DRAW ORDER, each node already carrying the direction the tour
    chose.  A `what` this does not recognise (a retreat, a final lift — neither
    of which a `park="home"` programme lays down) returns None, and the caller
    falls back to the expensive matrix rather than guessing.
    """
    m = _REFUSAL.search(str(msg))
    if m is None:
        return None
    what, k, depot = m.group(1), int(m.group(2)), 2 * n

    def node(j):
        return tour[1 + j] if 0 <= j < n else None

    if what == "entry":
        a, b = (depot, node(k)) if k == 0 else (node(k - 1), node(k))
    elif what == "transit":
        a, b = node(k), node(k + 1)
    elif what == "go-home":
        a, b = node(k), depot
    else:
        return None
    return None if a is None or b is None else (int(a), int(b))


def _apply_order(spec, segs, order, dirs, opts):
    """Segments in the tour's order, reversed where it asked. -> (segs, n_flip).

    A segment the tour wants drawn backwards is re-certified by
    `allocate.reverse_segment`, which re-runs the independent validator rather
    than inheriting a certificate; one that will not re-certify keeps its
    forward orientation and pays the extra transit.
    """
    out, flipped = [], 0
    for i, d in zip(order, dirs):
        seg = segs[int(i)]
        if int(d) < 0:
            rev = allocate.reverse_segment(seg, spec, opts)
            if rev is not None:
                seg, flipped = rev, flipped + 1
        out.append(seg)
    return out, flipped


def _plan_programme(spec, unordered, h_inv, pen_ext, opts, verbose=True,
                    retries=SEQ_RETRIES):
    """An order and the programme that flies it.  -> (segs, prog, info, 3 walls).

    THE MATRIX IS BUILT CHEAP, AND THAT COSTS SECONDS OF CLOCK, NOT SAFETY.
    With `paper_safe=True` `sequence.cost_matrix` ROUTES every crossing that
    dives through the canvas, grazes a base column or folds the arm through its
    own shoulder, so the tour is chosen against the detours it causes.
    Measured on this word: 410 of the 676 crossings dive, routing them costs
    ~100 s on 24 workers, and the tour it buys is 32.3 s of transit against the
    cheap matrix's 35.2 s — three seconds of programme for a hundred of
    planning, on the one run whose whole point is to iterate.  So the cheap
    matrix goes first.

    THE ORDER AND THE PROGRAMME ARE THEN ONE QUESTION, because the only thing
    that can go wrong with a cheap order is that a leg of it will not fly.  The cheap
    matrix does not know that; `writing.arm_program` does, and it NAMES the leg
    when it refuses.  So the loop is: solve, build, and on a refusal mark that
    one edge infinite and solve again.  Each round is 0.07 s of Held-Karp and
    under a second of programme building, against ~100 s to route all 676
    crossings up front, and it converges in one or two rounds because a word's
    unflyable crossings are a handful of long reaches across the arm's own base.

    If the retries run out — or the refusal names something that is not a tour
    edge, or the marked matrix has no finite tour left — the expensive matrix is
    built after all and asked once.  That is the answer the sequencer would have
    given from the start, so the fallback costs time and nothing else.

    NO GATE IS INVOLVED IN ANY OF THIS.  The matrix chooses an ORDER; every
    certificate is `plan_stroke`'s per stroke, `arm_program`'s per leg and
    `scene_check`'s over the finished timeline, and all three are unchanged.
    """
    n = len(unordered)
    t_mat = t_solve = t_prog = 0.0
    last = None
    for paper_safe in (False, True):
        t0 = time.perf_counter()
        C = sequence.cost_matrix(spec, unordered, TRANSIT_SPEED,
                                 writing.QD_FRAC, h_inv, pen_ext=pen_ext,
                                 q_start=spec.q_seed, return_home=True,
                                 paper_safe=paper_safe)
        t_mat += time.perf_counter() - t0
        for attempt in range(1 if paper_safe else max(1, int(retries))):
            t0 = time.perf_counter()
            try:
                res = sequence.solve(C, n)
            except RuntimeError as e:          # nothing finite left to try
                last = e
                break
            t_solve += time.perf_counter() - t0
            segs, flipped = _apply_order(spec, unordered, res["order"],
                                         res["dirs"], opts)
            info = dict(cost_s=float(res["cost"]), method=str(res["method"]),
                        matrix_wall_s=t_mat, solve_wall_s=t_solve,
                        reversed_n=int(flipped), paper_safe=bool(paper_safe),
                        retries=int(attempt), marked_edges=[],
                        order=[int(i) for i in res["order"]],
                        dirs=[int(d) for d in res["dirs"]])
            if verbose:
                print(f"  sequence: {n} strokes, {info['method']}, transit "
                      f"{info['cost_s']:.2f} s, {flipped} drawn backwards, "
                      f"{'routed' if paper_safe else 'cheap'} matrix"
                      + (f", attempt {attempt + 1}" if attempt else "")
                      + f" ({t_mat:.2f} s matrix + {t_solve:.2f} s solve)")
            t0 = time.perf_counter()
            try:
                prog = writing.arm_program(
                    spec, segs, draw_speed=DRAW_SPEED,
                    transit_speed=TRANSIT_SPEED, h_inv=fleet.H_INV_DEFAULT,
                    pen_ext=pen_ext, q_start=spec.q_seed,
                    park=writing.PARK_HOME)
                t_prog += time.perf_counter() - t0
                return segs, prog, info, t_mat, t_solve, t_prog
            except writing.PaperRefused as e:
                t_prog += time.perf_counter() - t0
                last = e
                edge = _refused_edge(
                    e, sequence.tour_of(res["order"], res["dirs"], n), n)
                if paper_safe or edge is None:
                    break
                if verbose:
                    print(f"    ! {e} — marking that one crossing unflyable "
                          f"and re-ordering")
                C[edge] = np.inf
        if verbose and not paper_safe:
            print(f"  ! the cheap matrix ran out of orders; building the "
                  f"routed one (slow, ~100 s)")
    raise Refused(
        f"{last}.  Every stroke certifies but a pen-up leg does not clear the "
        "paper plane, and re-ordering with the full routing screen did not "
        "find a way round it — the entry, a crossing between two letters, or "
        "the go-home.  Move the word (--dy, --width).  Nothing written.")


def plan_word(arm, width=WORD_WIDTH, height=WORD_HEIGHT, dy=WORD_DY,
              word=WORD, name=None, hover=0.0, out_dir=OUT_DIR, write=True,
              verbose=True, allow_partial=False):
    """Plan, certify and (if it certifies) write the SOLO word.  -> dict.

    Everything `plan_line` does, thirteen times, with `sequence` choosing the
    order in the middle.  The refusal rule is `plan_line`'s and one step
    stronger: a stroke the certified planner will not certify stops the whole
    run and NOTHING is written, because a word with a letter missing is not the
    word.  `allow_partial` trades that for the certified subset and says so in
    the one-liner and in the json.
    """
    arm = int(arm)
    name = name or (f"{word}_hover" if hover > 0 else word)
    fl, h = rt.fleet_for(None, None, "uniform")
    if arm not in fl:
        raise Refused(f"arm {arm} is not in rig {os.environ['ARIS_RIG']!r} "
                      f"(has {sorted(fl)})")
    spec = fl[arm]
    pen_real = float(frames.ext_of(None))
    pen_plan = pen_real + float(hover)
    polys, info, x0, x1, y = _word_geometry(spec, width, height, dy, word)
    if verbose:
        print(f"  {word!r}: {len(polys)} single-stroke Hershey polylines, "
              f"x {x0:.4f} .. {x1:.4f} m (centred on arm {arm}'s J1 axis), "
              f"baseline y = {y:.4f} m = seam {dy:+.3f} m")
        print(f"  x-height {1000 * height:.0f} mm, ascender "
              f"{1000 * info['ascender_m']:.0f} mm, tracking "
              f"{1000 * info['tracking_m']:+.1f} mm per gap")
        if info["tight"]:
            print(f"  !! NEGATIVE TRACKING: at this height the word's natural "
                  f"width is {info['natural_width']:.3f} m and --width is "
                  f"{width:.3f} m, so the letters are being pushed into each "
                  f"other.  Raise --width or lower --height "
                  f"(~{width / 9.0:.3f} m x-height fits {width:.2f} m).")

    # ---- 1. the certified single-arm stroke planner, once per stroke -------
    opts = dict(objective=pwl.OBJECTIVE, tilt_max_deg=TILT_MAX_DEG,
                pen_ext=pen_plan)
    t0 = time.perf_counter()
    segs, refused, strokes_json = [], [], []
    for i, p in enumerate(polys):
        pts = np.asarray(p, float)
        L = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        strokes_json.append(dict(id=i, color="black", kind="outline", length=L))
        plan = stroke_api.plan_stroke(pts, spec, opts)
        if plan["status"] != "ok":
            refused.append(dict(stroke=i, length_m=L,
                                status=str(plan["status"]),
                                reason=str(plan.get("reason")),
                                s_star=float(plan.get("s_star", 0.0) or 0.0),
                                x0=float(pts[0, 0]), y0=float(pts[0, 1])))
            continue
        segs.append(_segment(plan, stroke_id=i, kind="outline"))
    t_plan = time.perf_counter() - t0
    if verbose:
        print(f"  plan_stroke: {len(segs)}/{len(polys)} certified in "
              f"{t_plan:.2f} s")
    if refused and not allow_partial:
        raise Refused(
            f"arm {arm} cannot draw {len(refused)} of {len(polys)} strokes of "
            f"{word!r} at width {width:.3f} m, baseline y = {y:.4f}:\n"
            + "\n".join(f"    stroke {r['stroke']:2d} at "
                        f"({r['x0']:.3f}, {r['y0']:.3f}) m, "
                        f"{1000 * r['length_m']:.0f} mm: {r['status']} / "
                        f"{r['reason']!r}" for r in refused)
            + f"\n  Move the word (--dy, --width) or pass --allow-partial to "
              f"write the {len(segs)} that certify.  Nothing written.")
    if not segs:
        raise Refused(f"arm {arm} certifies NONE of {word!r} at width "
                      f"{width:.3f} m, baseline y = {y:.4f}.  Nothing written.")

    # ---- 2. the order, and 3. the solo programme ---------------------------
    segs, prog, seq_info, t_mat, t_solve, t_prog = _plan_programme(
        spec, segs, h, pen_plan, opts, verbose)
    samp = writing.uniform_samples(prog, DT)
    M = int(samp["n"])

    # ---- 4. the six-arm scene: the mover, and five arms at their parks -----
    t0 = time.perf_counter()
    q = {a: (np.asarray(samp["q"], float) if a == arm
             else np.tile(np.asarray(fl[a].q_seed, float), (M, 1)))
         for a in sorted(fl)}
    seg = {a: (np.asarray(samp["seg"], np.int64) if a == arm
               else np.full(M, -1, np.int64)) for a in sorted(fl)}
    pens = {a: pen_real for a in sorted(fl)}
    margin = float(coordination.PAIR_MARGIN)
    rep = scene_check.check_timeline(
        q, DT, margin, programs=None, h_inv=h, pen_ext=pens, sub=SUB,
        verbose=False, fleet=fl, drawing={a: seg[a] >= 0 for a in sorted(fl)})
    t_check = time.perf_counter() - t0
    g = _gate_numbers(rep, margin)
    dur = float(prog["duration"])
    if verbose:
        for ln in rt.summarise(rep, margin):
            print("  " + ln)
    if not rep["ok"]:
        raise Refused(
            f"the independent scene check FAILS for this word "
            f"(frame {rep.get('frame_failed')}, paper {rep.get('paper_failed')}, "
            f"column {rep.get('column_failed')}, self {rep.get('self_failed')}, "
            f"min inter-arm {1000 * rep['min_clearance']:.1f} mm against a "
            f"{1000 * margin:.0f} mm gate).  Nothing written.")

    T = np.asarray(spec.T_world_base(), float)
    draw_times = _draw_times(prog, len(segs))
    timing = dict(plan_strokes_s=t_plan, cost_matrix_s=t_mat, solve_s=t_solve,
                  arm_program_s=t_prog, scene_check_s=t_check, export_s=0.0)
    summary = dict(
        name=name, arm=arm, certified=True, word=word,
        placement=dict(width_m=float(width), height_m=float(height),
                       dy_m=float(dy), x0=float(x0), x1=float(x1),
                       baseline_y=float(y), seam_y=float(T[1, 3]),
                       centre_x=float(T[0, 3]),
                       tracking_m=float(info["tracking_m"]),
                       tight=bool(info["tight"]),
                       natural_width_m=float(info["natural_width"]),
                       ascender_m=float(info["ascender_m"]),
                       bbox=[float(v) for v in info["bbox"]]),
        hover_m=float(hover),
        rig=os.environ["ARIS_RIG"], tool=os.environ["ARIS_TOOL"],
        h_m=float(h), duration_s=dur,
        draw_speed_m_s=DRAW_SPEED, draw_s=float(prog["draw_s"]),
        transit_speed_m_s=TRANSIT_SPEED, fps=FPS, stride=STRIDE, sub=SUB,
        strokes=dict(asked=len(polys), planned=len(segs),
                     refused=len(refused), partial=bool(refused),
                     allow_partial=bool(allow_partial),
                     refusals=refused,
                     draw_order=[int(s["stroke_id"]) for s in segs],
                     directions=[int(s["direction"]) for s in segs]),
        sequence=dict(transit_s=seq_info["cost_s"], method=seq_info["method"],
                      reversed_n=seq_info["reversed_n"],
                      paper_routed_matrix=seq_info["paper_safe"],
                      retries=seq_info.get("retries", 0),
                      order=seq_info["order"], dirs=seq_info["dirs"]),
        arc_len_m=float(sum(s["length"] for s in segs)),
        traced_len_m=float(sum(s["length"] for s in strokes_json)),
        plan=dict(min_sigma=float(min(s["plan"]["min_sigma"] for s in segs)),
                  min_margin=float(min(s["plan"]["min_margin"] for s in segs)),
                  tip_err_m=float(max(s["plan"]["tip_err"] for s in segs)),
                  n_dense=int(sum(s["plan"]["n_dense"] for s in segs)),
                  lean_deg=float(max(s["plan"].get("lean_deg", 0.0) or 0.0
                                     for s in segs)),
                  tilt_max_deg=TILT_MAX_DEG, objective=pwl.OBJECTIVE),
        timing_s=timing,
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
    # THE HOVER'S WHOLE POINT, AS A NUMBER.  The plan used a 30 mm longer pen;
    # the certificate above did not, so this is the tip's real height above the
    # paper over the whole programme, measured by the independent checker.
    if hover > 0:
        summary["hover"] = dict(
            asked_m=float(hover),
            measured_tip_above_paper_m=g["min_paper_tip_m"],
            note="the plan used a pen `asked_m` longer and the certificate was "
                 "re-derived with the REAL pen; `measured_tip_above_paper_m` is "
                 "scene_check's own minimum tip clearance over the programme. "
                 "Measure it with a ruler at three points before any pen-down "
                 "run.")

    if not write:
        return dict(summary=summary, report=rep, payload=None, csv=None,
                    segs=segs)

    t0 = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{name}_{arm}"
    npz_path = out_dir / f"{stem}.npz"
    pj = _program_json(
        arm, f"day1 {word}, arm {arm}", "scripts/day1.py word --arm",
        [_plan_segment_json(s["plan"], dt, seg=k, stroke_id=s["stroke_id"],
                            kind="outline", direction=int(s["direction"]),
                            flipped=bool(s.get("flipped", False)))
         for k, (s, dt) in enumerate(zip(segs, draw_times))],
        strokes_json, pen_plan, pen_real, hover,
        phase_name=f"{word}, arm {arm}",
        wall_s=sum(timing.values()))
    pj_path = out_dir / f"{stem}_program.json"
    pj_path.write_text(json.dumps(pj, indent=1) + "\n")
    # The TARGET polylines, under the name `program_schema._stroke_polylines`
    # looks for: without them the 3D viewer can animate the arm but cannot draw
    # the word it is drawing.
    (out_dir / f"{stem}_strokes.json").write_text(json.dumps(dict(
        strokes=[dict(sj, pts=[[float(x), float(y_)] for x, y_ in
                               np.asarray(p, float)])
                 for sj, p in zip(strokes_json, polys)]), indent=1) + "\n")
    np.savez_compressed(npz_path, **_payload(q, seg, pens, margin,
                                             float(rep["min_clearance"]), stem))

    from aris_sixarm.execute.program import from_schedule
    fp = from_schedule(npz_path, pj_path)
    z = np.load(npz_path, allow_pickle=False)
    pw = pathway.build_arm_pathway(
        fp, z, arm, spec, tool=os.environ["ARIS_TOOL"],
        rig=os.environ["ARIS_RIG"], intensity=1.0,
        z_mode=("fk" if hover > 0 else "plane"),
        h_inv=float(pj["h_inv"]), program_json=pj,
        source_paths=dict(schedule=str(npz_path), program=str(pj_path)))

    # ---- 5. THE JOINT REFERENCE'S OWN SPEED, before the CSV exists ---------
    # The last gate, and the only one about the file rather than the scene.  A
    # reference that asks a joint for more than the FR3 will give is not a file
    # anybody should be able to stream, so it is refused here and the three
    # files already written are taken back with it.
    sa = speed_audit(prog, [[float(v) for v in row[11:18]] for row in pw.rows])
    summary["joint_speed"] = sa
    if verbose:
        print(_speed_line(sa))
    if not sa["ok"]:
        for p in (npz_path, pj_path, out_dir / f"{stem}_strokes.json"):
            try:
                p.unlink()
            except OSError:
                pass
        c, k = sa["at_csv_rows"], sa["at_1khz"]
        raise Refused(
            f"the joint reference EXCEEDS the FR3 velocity limit: "
            f"joint {c['worst_joint']} reaches "
            f"{100 * c['worst_frac']:.1f} % of its limit between CSV rows and "
            f"joint {k['worst_joint']} {100 * k['worst_frac']:.1f} % at 1 kHz "
            f"(limits {sa['qd_max_rad_s']} rad/s, pacer target "
            f"{100 * sa['qd_frac_target']:.0f} %).  Nothing written.")

    csv_path, man_path = _write_csv(pw, out_dir, name)
    timing["export_s"] = time.perf_counter() - t0
    summary["files"] = dict(npz=str(npz_path), csv=str(csv_path),
                            manifest=str(man_path), program=str(pj_path))
    summary["csv"] = dict(n_rows=pw.stats["n_rows"],
                          n_draw_rows=pw.stats["n_draw_rows"],
                          n_travel_rows=pw.stats["n_travel_rows"],
                          draw_length_m=pw.stats["draw_length_m"],
                          paper_z_base_m=pw.stats["paper_z_base_m"],
                          max_row_dq_rad=pw.stats["max_row_dq_rad"],
                          n_strokes=len(pw.stroke_meta)
                          if hasattr(pw, "stroke_meta") else len(segs),
                          joint_columns=True,
                          # THE HALF OF PETE'S CRITERION THE INK DOES NOT TEST,
                          # said in the file that gets streamed.
                          q_source="q1..q7 on every row are the certified "
                                   "plan's own joint solution for that "
                                   "waypoint — THE PLANNER'S REDUNDANCY "
                                   "RESOLUTION, carried through to the "
                                   "controller rather than thrown away and "
                                   "re-solved there",
                          q_limits_rad_s=[float(v) for v in frames.QD_MAX],
                          q_peak_frac_of_limit=sa["at_csv_rows"]["worst_frac"])
    (out_dir / f"{stem}.json").write_text(
        json.dumps(summary, indent=1, default=str) + "\n")
    summary["files"]["summary"] = str(out_dir / f"{stem}.json")
    return dict(summary=summary, report=rep, payload=npz_path, csv=csv_path,
                pathway=pw, segs=segs)


def _timing_line(t):
    """The planning breakdown, as one line of the one-liner. -> str."""
    return ("  wall " + " ".join(
        f"{k}={t[k]:.2f}" for k in ("plan_strokes_s", "cost_matrix_s",
                                    "solve_s", "arm_program_s",
                                    "scene_check_s", "export_s")
        if k in t).replace("_s=", " ")
        + f"  total {sum(t.values()):.2f} s")


def _word_one_liner(s, certified):
    """PASS/FAIL for a solo word: `line`'s gates plus what it drew. -> str."""
    st = s["strokes"]
    head = _one_liner(s["name"], s["arm"], s["gates"], s["duration_s"],
                      certified)
    tail = (f"  strokes {st['planned']}/{st['asked']}"
            + (f" (PARTIAL, {st['refused']} refused)" if st["refused"] else "")
            + f"  ink {s['arc_len_m']:.3f} m")
    return head + tail


def cmd_word_arm(a):
    """`word --arm N`: the solo word, planned and certified here and now."""
    arm = int(a.arm)
    hov = float(a.hover or 0.0)
    for ln in assumptions([arm]):
        print(ln)
    print(f"arm {arm}: the word {WORD!r} alone, "
          + (f"HOVER {1000 * hov:.0f} mm above the paper — no ink, no contact"
             if hov > 0 else "ON THE PAPER"))
    print(f"  the other five arms stand at their parks for the whole programme; "
          f"every gate is graded against all six.")
    t0 = time.perf_counter()
    r = plan_word(arm, width=float(a.width), height=float(a.height),
                  dy=float(a.dy), name=a.name, hover=hov,
                  out_dir=Path(a.out), allow_partial=bool(a.allow_partial))
    s = r["summary"]
    print()
    print(_word_one_liner(s, True))
    print(_speed_line(s["joint_speed"]))
    print(_timing_line(s["timing_s"]))
    print(f"  wall total (this command) {time.perf_counter() - t0:.2f} s")
    if s["strokes"]["refused"]:
        print(f"  ! PARTIAL: {s['strokes']['refused']} strokes were refused and "
              f"are NOT in this file:")
        for rf in s["strokes"]["refusals"]:
            print(f"      stroke {rf['stroke']:2d} at ({rf['x0']:.3f}, "
                  f"{rf['y0']:.3f}) m: {rf['status']} / {rf['reason']!r}")
    print(f"  wrote {s['files']['npz']}")
    print(f"        {s['files']['csv']}  ({s['csv']['n_rows']} rows, "
          f"{s['csv']['n_draw_rows']} draw + {s['csv']['n_travel_rows']} travel, "
          f"q1..q7 on every row)")
    print(f"        {s['files']['summary']}")
    if hov > 0:
        print(f"  the pen rides "
              f"{1000 * s['hover']['measured_tip_above_paper_m']:.1f} mm off "
              f"the paper (asked {1000 * hov:.0f}; scene_check's own minimum "
              f"over the programme): measure it with a ruler at three points "
              f"before ANY pen-down run.")
    print(f"  send it:  scripts/day1.py send --arm {arm} "
          f"--file {s['files']['csv']}")
    return 0


# ===========================================================================
# `word` — yesterday's two-arm assets
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
    # `--arm N` is the SOLO word and it PLANS; everything below it is the
    # two-arm asset re-check and plans nothing.  One subcommand, because at the
    # rig "run the word on arm 31" and "run the word" are the same sentence.
    if getattr(a, "arm", None):
        return cmd_word_arm(a)
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
# `send` — the interface to the arms.  File in, one command out.
# ===========================================================================
SEND_DOC = """Deliver ONE pathway CSV to the operator box and print the ONE
command that draws it.  This machine never moves an arm: `send` copies the file
with scp and then PRINTS the supervisor command; `--live` is the only way it is
ever run over ssh, and `--dry-run` copies nothing at all.

The chain, from `briefings/CONTROL_STACK_line_to_joint_torques_2026-09-09.md`:

    out/day1/<name>_<N>.csv
      -- scp -->  OPERATOR  ~/RTff/pathway_persist/<folder>/<name>_<N>.csv
      -- ARM_ID=<N> ~/RTff/draw_rtff_supervised.sh <csv> <fmin> <fmax>
                    <levels> fresh
            -> ladder gate (position control)   [SKIPPED on inverted arms]
            -> MoveIt to the start (position control)
            -> pen_switch down  (cartesian_impedance_controller)
            -> rtff_pathway_exec.py --csv <csv>   ... the drawing
            -> pen_switch up    (fr3_arm_controller)

WHAT THE BRIEFING DOES NOT SAY, AND THIS FILE THEREFORE DOES NOT INVENT.  It
names `run_forever_arm13.sh` / `run_forever_arm17.sh` and a
`dispatch_arm17_mine4H.sh`, and it names NO runner, NO dispatch script and NO
`pathway_persist` folder for arms 31 or 71 — only their control-box IPs
(192.168.50.12 / .14), their DDS domains (31 / 71) and that they are INVERTED.
So `send` targets the supervisor directly, which is the one entry point the
briefing gives a signature for, and leaves the keeper loop alone.  If the day
has a `run_forever_arm31.sh` by then, hold it first
(`bash ~/RTff/aris_hold.sh hold 31`) or the keeper will start its own pass on
top of this one.

TWO THINGS THE BRIEFING IS EMPHATIC ABOUT AND `send` PRINTS EVERY TIME.  The
stack must be HEALTHY before a pass (hardware active, three controllers,
robot_mode 2 — never heal on a user stop) and the ladder gate, which measures
the paper plane before every descend, is SKIPPED FOR INVERTED ARMS.  Arms 31
and 71 are inverted.  Nothing downstream will measure the plane for them, so
the `z_m` baked into this CSV is the plane they will draw at: a hover pass with
a ruler first is not optional on these two."""


def _csv_rows(path):
    with open(path) as f:
        return sum(1 for ln in f if ln.strip()) - 1


def cmd_send(a):
    """scp the CSV, then print the command that draws it.  -> 0."""
    src = Path(a.file)
    if not src.exists():
        raise Refused(f"no CSV at {src}.  `word --arm {a.arm}` writes one into "
                      f"{OUT_DIR}.")
    if src.suffix.lower() != ".csv":
        raise Refused(f"{src} is not a .csv — `send` delivers the impedance "
                      f"pathway CSV and nothing else.")
    hdr = open(src).readline().rstrip("\n").split(",")
    if hdr[:len(pathway.CSV_COLUMNS)] != pathway.CSV_COLUMNS:
        raise Refused(
            f"{src} does not carry this repo's pathway columns.\n"
            f"    want {','.join(pathway.CSV_COLUMNS)}\n"
            f"    got  {','.join(hdr)}")
    arm = int(a.arm)
    host = a.host or OPERATOR["host"]
    folder = a.folder or f"day1_arm{arm}"
    remote_dir = f"{OPERATOR['store']}/{folder}"
    remote_csv = f"{remote_dir}/{src.name}"
    fmin, fmax, levels = OPERATOR["force"]
    mkdir = f"ssh {host} 'mkdir -p {remote_dir}'"
    scp = f"scp {src} {host}:{remote_dir}/"
    draw = (f"ARM_ID={arm} bash {OPERATOR['supervisor']} {remote_csv} "
            f"{fmin} {fmax} {levels} {OPERATOR['mode']}")
    stack = f"ssh {host} 'bash {OPERATOR['stack_check']} stack {arm}'"

    print(f"ARM {arm}   {src}  ({_csv_rows(src)} rows, q1..q7 on every row)")
    print(f"  operator  {host}")
    print(f"  lands at  {remote_csv}")
    print()
    if a.dry_run:
        print("DRY RUN — nothing copied.  The two commands are:")
        print(f"  {mkdir}")
        print(f"  {scp}")
    else:
        for cmd in (mkdir, scp):
            print(f"  $ {cmd}")
            r = subprocess.run(cmd, shell=True)
            if r.returncode != 0:
                raise Refused(f"`{cmd}` exited {r.returncode}.  The file is NOT "
                              f"on the operator box.  Check the address at the "
                              f"top of scripts/day1.py (or --host), and that "
                              f"the key is loaded — the briefing says password "
                              f"auth is disabled.")
        print("  copied.")
    print()
    print("BEFORE YOU RUN IT — the stack must be healthy (hardware active, "
          "three controllers,")
    print("robot_mode 2).  Never heal on a user stop (mode 5) or guiding "
          "(mode 3):")
    print(f"  {stack}")
    print()
    print("THEN, ON THE OPERATOR BOX, exactly this one command:")
    print(f"  {draw}")
    print()
    print(f"ARM {arm} IS INVERTED, so the supervisor SKIPS the ladder gate: "
          f"nothing downstream")
    print(f"will measure the paper plane for it.  The z in this file is the "
          f"plane it will draw")
    print(f"at.  Fly the hover pass and measure it with a ruler at three "
          f"points first.")
    print("ABORT: the physical e-stop.  The software gate and the 20 mrad "
          "watchdog are not the abort path.")
    if a.live:
        print()
        print(f"--live: running it over ssh NOW.")
        cmd = f"ssh {host} '{draw}'"
        print(f"  $ {cmd}")
        r = subprocess.run(cmd, shell=True)
        if r.returncode != 0:
            raise Refused(f"the draw command exited {r.returncode}.")
    return 0


# ===========================================================================
def build_parser():
    ap = argparse.ArgumentParser(
        prog="scripts/day1.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Hardware day 1: one line with one arm, then the word with "
                    "one arm, then the word with two.  `send` hands the "
                    "resulting CSV to the box that runs the arms.",
        epilog="""examples — the ladder, in order
  # one arm, one straight line, on the paper, certified end to end
  scripts/day1.py line --arm 31 --from 0.50,1.70 --to 0.65,1.70 --name probe

  # the same line 30 mm ABOVE the paper — the first powered motion
  scripts/day1.py line --arm 31 --from 0.50,1.70 --to 0.65,1.70 --hover 0.03 \\
      --name probe_hover

  # the SOLO word, 30 mm above the paper, then on it.  One arm, then the other
  scripts/day1.py word --arm 31 --hover
  scripts/day1.py word --arm 31
  scripts/day1.py word --arm 71 --hover
  scripts/day1.py word --arm 71

  # the two-arm word, alternating: re-check yesterday's certified file, today
  scripts/day1.py word --variant alt

  # deliver a CSV to the operator box and print the command that draws it
  scripts/day1.py send --arm 31 --file out/day1/unknown_31.csv --dry-run

coordinates are METRES in the canvas datum: x across the 1.8034 m sheet, y
along the 3.63064 m one.  The seam / the middle row's own line is y = 1.8153.
Arm 31's J1 axis is at x = 0.5967, arm 71's at x = 1.2067.

ARM 31 REFUSES LINES EXACTLY ON THE SEAM LINE: the pen-up leg to or from its
park does not clear the paper plane there.  `word --arm` therefore baselines
the word at --dy = -0.10 m by default; try -0.15 or -0.20 if that refuses.

everything is written to out/day1/.  An uncertified line, word or joint
reference writes NOTHING.""")
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
        help="the word: --arm N plans a SOLO one; without it, re-check "
             "yesterday's two-arm asset",
        description=f"WITH --arm N: plans, certifies and exports the word "
                    f"{WORD!r} for ONE arm, the other five frozen at their "
                    f"parks.  Same planner, same programme builder, same "
                    f"independent check and same three files as `line`, with "
                    f"`sequence` choosing the stroke order over the real "
                    f"transit cost in the middle.  A stroke that will not "
                    f"certify stops the run and writes nothing unless "
                    f"--allow-partial.\n\n"
                    f"WITHOUT --arm: does NOT plan.  Re-runs the independent "
                    f"whole-timeline check on a certified 2026-09-15 asset, "
                    f"copies the per-arm pathway CSVs into out/day1/, and "
                    f"prints the run order and the pass criteria.  --replan "
                    f"re-plans from a strokes file with the conductor's exact "
                    f"job parameters and refuses to report success on anything "
                    f"but the conducted coverage.")
    w.add_argument("--arm", type=int, default=None, choices=sorted(ARMS),
                   help="plan the SOLO word for this arm (31 or 71).  Without "
                        "it this subcommand re-checks the two-arm asset.")
    w.add_argument("--width", type=float, default=WORD_WIDTH, metavar="M",
                   help=f"metres of ink across, centred on the arm's own J1 "
                        f"axis (default {WORD_WIDTH:g}; arm 31's axis is at "
                        f"x = 0.5967, arm 71's at x = 1.2067)")
    w.add_argument("--height", type=float, default=WORD_HEIGHT, metavar="M",
                   help=f"x-HEIGHT: how tall 'u', 'n', 'o', 'w' come out "
                        f"(default {WORD_HEIGHT:g}; 'k' reaches 1.5x this).  "
                        f"The word's NATURAL width is about 9x this, so a "
                        f"--width below that squeezes the letters together.")
    w.add_argument("--dy", type=float, default=WORD_DY, metavar="M",
                   help=f"baseline offset from the seam line y = 1.8153 "
                        f"(default {WORD_DY:g}).  NOT zero: arm 31 refuses "
                        f"lines exactly on the seam line — the pen-up leg to "
                        f"or from its park does not clear the paper plane "
                        f"there.  Try -0.15 or -0.20 if -0.10 refuses.")
    w.add_argument("--hover", type=float, nargs="?", const=HOVER_DEFAULT,
                   default=0.0, metavar="M",
                   help=f"fly the word this far ABOVE the paper instead of on "
                        f"it (default when the flag is given: "
                        f"{HOVER_DEFAULT:g} m).  Plans with a pen that much "
                        f"longer and grades with the real one; the json states "
                        f"the measured tip height above the paper.")
    w.add_argument("--allow-partial", action="store_true",
                   help="write the certified subset instead of refusing when "
                        "some strokes will not certify (and say so)")
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
    w.add_argument("--name", default=None,
                   help=f"file stem (--arm: default {WORD!r}, or "
                        f"{WORD + '_hover'!r} with --hover; --replan: the "
                        f"output stem)")
    w.set_defaults(func=cmd_word)

    s = sub.add_parser(
        "send", formatter_class=argparse.RawDescriptionHelpFormatter,
        help="deliver a pathway CSV to the operator box and print the one "
             "command that runs it",
        description=SEND_DOC)
    s.add_argument("--arm", type=int, required=True,
                   help="the arm this file is for (its DDS domain id)")
    s.add_argument("--file", required=True, metavar="CSV",
                   help="the pathway CSV, e.g. out/day1/unknown_31.csv")
    s.add_argument("--host", default=None, metavar="USER@HOST",
                   help=f"the operator box (default {OPERATOR['host']!r} — "
                        f"fill it in at the top of this file)")
    s.add_argument("--folder", default=None, metavar="NAME",
                   help="subfolder under the operator's pathway store "
                        "(default: day1_arm<N>)")
    s.add_argument("--dry-run", action="store_true",
                   help="print the scp and the run command, copy nothing")
    s.add_argument("--live", action="store_true",
                   help="RUN the draw command over ssh instead of printing it. "
                        "Without this the arm never moves from this machine.")
    s.set_defaults(func=cmd_send)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
