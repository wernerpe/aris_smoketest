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
                         layout, mounts, paper, pwl, scene_check, selfcoll,
                         sequence, stroke_api, transit, writing)
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

# --- EVERYTHING SITE-SPECIFIC IS IN ONE TRACKED FILE --------------------
# `config/site.json`, and nothing in this script hard-codes an address, an arm
# id, a remote path or a dispatch flag.  It is printed at the top of every run,
# `day1.py site` shows it, `day1.py site --set slot31.arm=97` edits it, and the
# GUI reads and writes the same file — so reconfiguring the rig is editing one
# JSON file and not hunting through Python.
SITE_PATH = ROOT / "config" / "site.json"
_SITE = {}


def site(path=None, reload=False):
    """The site configuration. -> dict.

    `--site FILE` (or `$ARIS_SITE`) moves it; the default is
    `config/site.json`, which is tracked, so a fresh clone on the robot PC has
    today's facts in it and edits them in place.
    """
    global _SITE
    if _SITE and not path and not reload:
        return _SITE
    p = Path(path or os.environ.get("ARIS_SITE") or SITE_PATH)
    if not p.exists():
        raise Refused(f"no site configuration at {p}.  It is tracked in this "
                      f"repository; pass --site FILE if it lives elsewhere.")
    try:
        _SITE = json.loads(p.read_text())
    except Exception as exc:
        raise Refused(f"{p} is not readable JSON: {exc}")
    _SITE["_path"] = str(p)
    return _SITE


def slot_cfg(slot):
    """One slot's block, by the PLAN's slot id. -> dict."""
    sl = (site().get("slots") or {}).get(str(int(slot)))
    if sl is None:
        raise Refused(f"slot {slot} is not in {site()['_path']} "
                      f"(has {sorted((site().get('slots') or {}))})")
    return sl


def site_lines():
    """The site configuration, as the lines every run prints. -> list[str]."""
    st = site()
    out = [f"SITE ({st['_path']}) — edit this file, not the script:",
           f"  operator          {st['operator']['host']}"]
    for k in sorted(st.get("slots") or {}, key=int):
        sl = st["slots"][k]
        m = sl.get("mounted")
        out.append(
            f"  slot {k:<3} ({sl.get('position', '?'):<12}) -> arm "
            f"{sl.get('arm')}  ip {sl.get('ip')}  domain {sl.get('domain')}  "
            f"paper_z {1000 * float(sl.get('paper_z') or 0.0):+.1f} mm  "
            f"mounted {'UNCONFIRMED' if m is None else ('yes' if m else 'NO')}")
    out.append("  dispatch env      "
               + " ".join(f"{k}={v}" for k, v in
                          (st.get("rtff_env") or {}).items()))
    if st.get("notes"):
        out += _wrap(st["notes"], "  ! ")
    out.append("")
    return out


# WHAT THE DEPLOYED EXECUTOR ACTUALLY READS, AND IT IS NOT EVERYTHING WE WRITE.
# Measured on the deployed branch in the 2026-09-17 session.  This sentence is
# printed on the PASS line and written into the summary json and the manifest,
# because the most expensive mistake available here is to believe the CSV's
# joint columns are a joint reference the robot will follow.
EXECUTOR_NOTE = (
    "THE DEPLOYED IMPEDANCE EXECUTOR IGNORES q1..q7 AND t_s.  It follows the "
    "TIP POSE of each row, paces by Cartesian arc length at RTFF_TRAVEL_SPEED "
    "(0.02 m/s), holds the row's orientation without slerping between rows, "
    "and latches its own nullspace — so the joints are NOT followed by the "
    "deployed executor and the redundancy is resolved by the controller, not "
    "by this file.  The joint columns and t_s are carried for a joint-capable "
    "executor, for the velocity certificate, and as the record of what the "
    "planner chose; on the deployed stack they are a CHECK, not a command.")

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
    out = site_lines()
    fl = layout.FLEET_PROPOSED
    h = float(layout.LAYOUT_PROPOSED["h"])
    lat, ext = float(frames.PEN_LAT_HOLDER), float(frames.PEN_EXT_HOLDER)
    out += [
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


def _rates(Q, t):
    """Peak |dq/dt| and |d2q/dt2| per joint over samples at times `t`.

    -> (7,), (7,).  `t` need not be uniform — between CSV rows it is not: the
    exporter collapses a run of IDENTICAL poses to one row, so a barrier hold
    is one row and a real elapsed gap.  Charging that gap the nominal frame
    period (which is what a uniform reading does) would invent a speed nothing
    moves at.
    """
    Q = np.asarray(Q, float).reshape(-1, 7)
    t = np.asarray(t, float).reshape(-1)
    if len(Q) < 2:
        return np.zeros(7), np.zeros(7)
    dt = np.diff(t)
    ok = dt > 0
    v = np.zeros((len(dt), 7))
    v[ok] = np.diff(Q, axis=0)[ok] / dt[ok][:, None]
    if len(v) > 1:
        dtm = 0.5 * (dt[:-1] + dt[1:])
        okm = dtm > 0
        a = np.zeros((len(v) - 1, 7))
        a[okm] = np.diff(v, axis=0)[okm] / dtm[okm][:, None]
    else:
        a = np.zeros((1, 7))
    return np.abs(v).max(axis=0), np.abs(a).max(axis=0)


def speed_audit(t_s, q_rows, khz_dt=0.001):
    """Is the joint reference inside the FR3's velocity limits? -> dict.

    THE CSV IS A JOINT REFERENCE AND SOMEBODY IS GOING TO STREAM IT.  Every
    other gate in this file is about where the arm is; this one is about how
    fast it gets there, and it is the one a stiff controller turns into torque.
    `frames.QD_MAX` is the FR3's own per-joint limit and `writing.QD_FRAC`
    (0.30) is the fraction the pacer aims at, so a healthy programme reads
    about a third of the limit and anything near 1.0 is a bug upstream, not a
    tight day.

    BOTH READINGS ARE TAKEN FROM THE FILE'S OWN `t_s` COLUMN, not from the
    programme it came out of, because the file is the thing being certified:
      `at_csv_rows`   consecutive rows, at the elapsed time between their own
                      timestamps.  This is a controller that consumes the file
                      row by row at the pacing it carries.
      `at_1khz`       those same rows linearly interpolated onto a 1 ms grid,
                      which is the stream rate of the deployed stack.  Linear
                      interpolation makes the velocity piecewise-constant, so
                      the peak speed matches the row intervals and the
                      ACCELERATION is the step between two of them over a
                      millisecond — an impulse, reported for information and
                      not gated.

    THE CERTIFICATE HOLDS AT THIS TIMING AND NO OTHER.  `rtff_pathway_exec`
    paces by Cartesian arc length and can traverse the same path faster or
    slower than the planner did; run it quicker and every number here scales
    with it.  That sentence is in the json and in the manifest as well as here,
    because the file outlives this process.
    """
    lim = np.asarray(frames.QD_MAX, float).reshape(7)
    t = np.asarray(t_s, float).reshape(-1)
    Q = np.asarray(q_rows, float).reshape(-1, 7)
    n_bad = int((np.diff(t) <= 0).sum()) if len(t) > 1 else 0
    if len(t) > 1:
        grid = np.arange(float(t[0]), float(t[-1]) + khz_dt, khz_dt)
        Qi = np.column_stack([np.interp(grid, t, Q[:, j]) for j in range(7)])
    else:
        grid, Qi = t, Q
    out = {}
    for key, (QQ, tt) in dict(at_csv_rows=(Q, t), at_1khz=(Qi, grid)).items():
        v, a = _rates(QQ, tt)
        frac = v / lim
        dts = np.diff(np.asarray(tt, float)) if len(tt) > 1 else np.zeros(1)
        out[key] = dict(
            n_samples=int(len(QQ)),
            dt_min_s=float(dts.min()), dt_max_s=float(dts.max()),
            dt_median_s=float(np.median(dts)),
            span_s=float(tt[-1] - tt[0]) if len(tt) > 1 else 0.0,
            peak_qd_rad_s=[float(x) for x in v],
            frac_of_limit=[float(x) for x in frac],
            worst_joint=int(np.argmax(frac)) + 1,
            worst_frac=float(frac.max()),
            peak_qdd_rad_s2=[float(x) for x in a],
            ok=bool((frac <= 1.0).all()))
    out.update(qd_max_rad_s=[float(x) for x in lim],
               qd_frac_target=float(writing.QD_FRAC),
               source="the CSV's own t_s column",
               n_nonmonotonic_t=n_bad,
               ok=bool(out["at_csv_rows"]["ok"] and out["at_1khz"]["ok"]
                       and n_bad == 0),
               note="q1..q7 on every CSV row are THE PLANNER'S OWN redundancy "
                    "resolution for that waypoint — the configuration the "
                    "certified plan chose, carried alongside the Cartesian "
                    "pose, not re-solved by the controller.",
               validity="THIS CERTIFICATE HOLDS AT THE FILE'S OWN t_s TIMING "
                        "AND NO OTHER.  The executor paces by Cartesian arc "
                        "length and may re-pace the path; run it faster than "
                        "t_s says and every speed here scales with it.")
    return out


#: the largest joint step the exporter may legitimately put between two rows.
#: DERIVED, not chosen, and it is `tests/test_export_pathway.py`'s own
#: constant: the conductor bounds its sub-step at `writing.MAX_DQ_FRAME` and
#: the npz keeps every `stride`-th one.  Anything above it is a branch change.
ROW_DQ_BOUND = 2 * writing.MAX_DQ_FRAME


def _row_step_audit(rows, h_inv, spec, tool, hover=0.0):
    """Is every pen-up run Cartesian-followable, and is the hover where it
    should be?  -> dict.

    TWO QUESTIONS THE DEPLOYED EXECUTOR MAKES INTO ONE.  It walks consecutive
    non-draw rows as a Cartesian path at its own travel speed and latches its
    own nullspace (ARIS2_CONTRACTS §1), so (a) a joint step between two travel
    rows that no continuous IK branch could produce is a path it cannot follow,
    and (b) the height it flies those rows at is whatever the rows say, since
    with `RTFF_CONTACT_DESCEND=0` nothing measures the paper.

    THE HOVER SIGN, FOR AN INVERTED ARM.  `fr3_link0`'s +z points DOWN in the
    world on these two, so "above the paper" is a SMALLER base z: a 30 mm hover
    is `z_paper - 0.030`, not plus.  The check is stated as
    `z_paper - z_row`, which must come out POSITIVE and equal to the hover.
    """
    Q = np.asarray([[float(v) for v in r[11:18]] for r in rows], float)
    kind = [r[2] for r in rows]
    z = np.asarray([float(r[5]) for r in rows], float)
    z_paper, _ = pathway.paper_z_base(pathway.base_transform(spec, h_inv))
    draw = np.array([k == pathway.KIND_DRAW for k in kind])
    trav = np.array([k != pathway.KIND_DRAW for k in kind])
    if len(Q) < 2:
        return dict(n_rows=len(Q), max_dq_rad=0.0, max_travel_dq_rad=0.0,
                    bound_rad=float(ROW_DQ_BOUND), reconfigures=False)
    dq = np.abs(np.diff(Q, axis=0)).max(axis=1)
    pair = trav[:-1] & trav[1:]
    tmax = float(dq[pair].max()) if pair.any() else 0.0
    above = z_paper - z
    out = dict(
        n_rows=int(len(Q)), n_draw=int(draw.sum()), n_travel=int(trav.sum()),
        max_dq_rad=float(dq.max()), max_travel_dq_rad=tmax,
        bound_rad=float(ROW_DQ_BOUND),
        n_over_bound=int((dq > ROW_DQ_BOUND).sum()),
        reconfigures=bool(dq.max() > ROW_DQ_BOUND),
        note="the executor walks consecutive non-draw rows as a CARTESIAN "
             "path and latches its own nullspace, so a joint step no "
             "continuous IK branch could produce is a path it cannot follow "
             "(ARIS2_CONTRACTS §1)",
        paper_z_base_m=float(z_paper),
        tip_above_paper_base_m=dict(
            draw_min=float(above[draw].min()) if draw.any() else None,
            draw_max=float(above[draw].max()) if draw.any() else None,
            travel_max=float(above[trav].max()) if trav.any() else None),
        hover_sign_note="`z_paper - z_row` is the height ABOVE the paper: "
                        "fr3_link0's +z points DOWN on an inverted arm, so a "
                        "30 mm hover is z_paper - 0.030 in the base frame")
    if hover > 0 and draw.any():
        lo, hi = float(above[draw].min()), float(above[draw].max())
        out["hover_check"] = dict(
            asked_m=float(hover), row_min_m=lo, row_max_m=hi,
            ok=bool(abs(lo - hover) < 0.002 and abs(hi - hover) < 0.002))
    return out


def _rows_line(r):
    """The row-step verdict, as one line. -> str."""
    h = r.get("hover_check")
    return (f"  rows        {'OK  ' if not r['reconfigures'] else 'RECONFIG'} "
            f"max dq/row {r['max_dq_rad']:.4f} rad (bound "
            f"{r['bound_rad']:.4f}), across pen-up pairs "
            f"{r['max_travel_dq_rad']:.4f}"
            + (f"  hover rows {1000 * h['row_min_m']:.1f}.."
               f"{1000 * h['row_max_m']:.1f} mm above paper "
               f"({'OK' if h['ok'] else 'WRONG'})" if h else ""))


def _speed_line(sa):
    """The joint-speed verdict, as one line. -> str.

    It says "at t_s" because that is the whole qualification: the executor
    paces by Cartesian arc length and may traverse the same path faster.
    """
    c, k = sa["at_csv_rows"], sa["at_1khz"]
    return (f"  joint speed {'OK ' if sa['ok'] else 'OVER'} at t_s  "
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


# --- the t_s column -------------------------------------------------------
# THE EXPORTER'S CSV CARRIES NO CLOCK.  `CSV_COLUMNS` is stroke_idx, wp_idx,
# kind, the pose, intensity and q1..q7 — where the arm is and how it is folded,
# and nothing about when.  Pete asked for the planner's own pacing in the file,
# so `t_s` is APPENDED AS THE LAST COLUMN: a reader that indexes by position
# sees exactly the file it saw before, and one that reads the header gains a
# clock.  `aris_sixarm/export/pathway.py` is NOT edited for this — the times
# are reconstructed here from the exporter's own functions, and the
# reconstruction is checked against the row count before anything is written.
CSV_COLUMNS_T = list(pathway.CSV_COLUMNS) + ["t_s"]
T_S_NOTE = (
    "t_s is seconds from the start of the arm's programme on the SAME CLOCK as "
    "the schedule npz (frame k of the timeline is k/fps), i.e. the planner's "
    "own pacing. It is the LAST column, appended after the v1 contract's "
    "columns, so a positional reader of the first 18 is unaffected. Rows the "
    "exporter SYNTHESIZES (the lift_start / lift_end ramps it solves when the "
    "timeline gives no lift of its own) are not in that clock and are "
    "extrapolated at the nominal frame period; `n_synth_rows` says how many "
    "there are, and it is 0 for a programme that parks at both ends. THE "
    "EXECUTOR MAY RE-PACE THE PATH: rtff_pathway_exec paces by Cartesian arc "
    "length, so t_s is what the planner intended and not a promise about the "
    "wall clock — the joint-velocity certificate in the summary json holds at "
    "THIS timing and scales with any other.")


def _row_times(prog, z, arm, spec, tool, h_inv, n_rows, decimate_m=0.0):
    """Seconds from the start of the programme, per CSV row. -> (N,) or None.

    `build_arm_pathway` emits, in this order: the synthesized approach ramp,
    one row per frame `_emit_frames` kept, and the synthesized retract ramp.
    Calling `_emit_frames` again with the same arguments reproduces that list
    exactly — it is a deterministic function of the frames — so the k-th
    middle row is timeline frame `frame_idx[k]` and its time is `k/fps`.

    -> (times, n_synthesized_rows), or None if the reconstruction does not
    account for every row — the only way this can be wrong, and therefore
    checked rather than trusted: the caller refuses to write a file it cannot
    timestamp.
    """
    pen_lat, pen_ext = pathway.tool_offsets(tool)
    z_paper, tilt_deg = pathway.paper_z_base(
        pathway.base_transform(spec, h_inv))
    Q, SEG, _ = pathway._arm_frames(prog, z, arm)
    frame_idx, _drw, ends, _notes = pathway._emit_frames(
        Q, SEG, pen_ext, pen_lat, z_paper, pathway.LIFT_M,
        window=int(round(pathway.END_WINDOW_S * prog.fps)),
        decimate_m=decimate_m)
    pre, post = len(ends["pre"]), len(ends["post"])
    if pre + len(frame_idx) + post != int(n_rows):
        return None
    if not len(frame_idx):
        return np.zeros(0), 0
    dt = 1.0 / float(prog.fps)
    mid = np.asarray(frame_idx, float) * dt
    t_pre = mid[0] - dt * np.arange(pre, 0, -1.0)
    t_post = mid[-1] + dt * np.arange(1.0, post + 1)
    return np.concatenate([t_pre, mid, t_post]), pre + post


def _write_csv(pw, out_dir, stem, times=None, t_note=None):
    """`export.pathway.write_pathway` with this day's file names.

    The rows and the manifest are the exporter's own — `build_arm_pathway` has
    already run the contract's FK gate over the formatted text — and the only
    things that differ are `<stem>_<arm>.csv` against the exporter's
    `<name>_arm<arm>.csv`, which is what the day asks for, and the appended
    `t_s` column (see `T_S_NOTE`).  The manifest is deep-copied before the
    extra keys go in, so the exporter's own record is not mutated.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{stem}_{pw.arm_id}.csv"
    man_path = out_dir / f"{stem}_{pw.arm_id}.manifest.json"
    cols = list(pathway.CSV_COLUMNS) if times is None else list(CSV_COLUMNS_T)
    rows = (list(pw.rows) if times is None
            else [list(r) + [f"{float(t):.6f}"]
                  for r, t in zip(pw.rows, times)])
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)
    man = json.loads(json.dumps(pw.manifest, default=str))
    man["csv"] = csv_path.name
    if times is not None:
        man.setdefault("format", {})["columns"] = cols
        man["t_s"] = dict(t_note or {})
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
    # the planner's own pacing, as the file's last column — see `_row_times`
    got = _row_times(fp, z, arm, spec, os.environ["ARIS_TOOL"],
                     float(pj["h_inv"]), len(pw.rows))
    if got is None:
        for p in (npz_path, pj_path, out_dir / f"{stem}_strokes.json"):
            try:
                p.unlink()
            except OSError:
                pass
        raise Refused(f"the {len(pw.rows)} CSV rows could not be matched to "
                      f"the timeline's frames, so the t_s column would be a "
                      f"guess.  Nothing written.")
    t_s, n_synth = got
    sa = speed_audit(t_s, [[float(v) for v in row[11:18]] for row in pw.rows])
    csv_path, man_path = _write_csv(
        pw, out_dir, name, times=t_s,
        t_note=dict(note=T_S_NOTE, fps=float(FPS), n_rows=int(len(t_s)),
                    n_synth_rows=int(n_synth), t_first_s=float(t_s[0]),
                    t_last_s=float(t_s[-1]), programme_duration_s=dur,
                    joint_speed=sa))
    summary["joint_speed"] = sa
    summary["files"] = dict(npz=str(npz_path), csv=str(csv_path),
                            manifest=str(man_path), program=str(pj_path))
    summary["csv"] = dict(n_rows=pw.stats["n_rows"],
                          n_draw_rows=pw.stats["n_draw_rows"],
                          n_travel_rows=pw.stats["n_travel_rows"],
                          draw_length_m=pw.stats["draw_length_m"],
                          paper_z_base_m=pw.stats["paper_z_base_m"],
                          max_row_dq_rad=pw.stats["max_row_dq_rad"],
                          joint_columns=True,
                          columns=CSV_COLUMNS_T,
                          t_s=dict(first_s=float(t_s[0]),
                                   last_s=float(t_s[-1]),
                                   n_synth_rows=int(n_synth), note=T_S_NOTE))
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
    print()
    for ln in _file_pose_lines(s["files"]["csv"], int(a.arm)):
        print(ln)
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
              verbose=True, allow_partial=False, paper_z=0.0):
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
    # THE HOVER AND THE PAPER CORRECTION ARE THE SAME LEVER, and it is the pen.
    # A pen `d` LONGER in the plan puts the real tip `d` ABOVE the modelled
    # paper plane (measured: --hover 0.030 reads +27 to +29 mm).  So a paper
    # surface that really sits `paper_z` ABOVE where the model puts it is
    # corrected by exactly the same `paper_z` of extra planned pen.
    pen_plan = pen_real + float(hover) + float(paper_z)
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
    summary["paper_z_m"] = float(paper_z)
    summary["executor"] = EXECUTOR_NOTE
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

    # ---- 5. THE CLOCK, and THE JOINT SPEED ON IT, before the CSV exists ----
    # The last gate, and the only one about the file rather than the scene.  A
    # reference that asks a joint for more than the FR3 will give is not a file
    # anybody should be able to stream, so it is refused here and the three
    # files already written are taken back with it.  The times come first
    # because the speed is measured ON them.
    def _undo():
        for p in (npz_path, pj_path, out_dir / f"{stem}_strokes.json"):
            try:
                p.unlink()
            except OSError:
                pass

    got = _row_times(fp, z, arm, spec, os.environ["ARIS_TOOL"],
                     float(pj["h_inv"]), len(pw.rows))
    if got is None:
        _undo()
        raise Refused(
            f"the {len(pw.rows)} CSV rows could not be matched to the "
            f"timeline's frames, so the t_s column would be a guess.  Nothing "
            f"written.")
    t_s, n_synth = got
    q_rows = [[float(v) for v in row[11:18]] for row in pw.rows]
    sa = speed_audit(t_s, q_rows)
    summary["joint_speed"] = sa
    if verbose:
        print(_speed_line(sa))
    if not sa["ok"]:
        _undo()
        c, k = sa["at_csv_rows"], sa["at_1khz"]
        raise Refused(
            f"the joint reference EXCEEDS the FR3 velocity limit at its own "
            f"t_s pacing: joint {c['worst_joint']} reaches "
            f"{100 * c['worst_frac']:.1f} % of its limit between CSV rows and "
            f"joint {k['worst_joint']} {100 * k['worst_frac']:.1f} % at 1 kHz "
            f"(limits {sa['qd_max_rad_s']} rad/s, pacer target "
            f"{100 * sa['qd_frac_target']:.0f} %"
            + (f"; {sa['n_nonmonotonic_t']} row times do not increase"
               if sa["n_nonmonotonic_t"] else "")
            + ").  Nothing written.")

    # ---- 6. IS EVERY PEN-UP RUN A CARTESIAN-FOLLOWABLE PATH? --------------
    # The deployed executor walks consecutive non-draw rows as a CARTESIAN path
    # (ARIS2_CONTRACTS §1) and latches its own nullspace, so a transit that
    # reconfigures the arm between two rows is one it cannot follow: the tip
    # would have to jump branches with nothing telling it to.  This measures
    # the joint step the file actually asks for, per row and across pen-up
    # pairs, against the conductor's own per-frame bound.
    summary["rows"] = _row_step_audit(pw.rows, float(pj["h_inv"]),
                                      spec, os.environ["ARIS_TOOL"],
                                      hover=float(hover))
    if verbose:
        print(_rows_line(summary["rows"]))
    if summary["rows"]["reconfigures"]:
        _undo()
        raise Refused(
            f"a pen-up run in this file RECONFIGURES the arm: "
            f"{summary['rows']['max_travel_dq_rad']:.3f} rad between two "
            f"consecutive travel rows (bound "
            f"{summary['rows']['bound_rad']:.3f}).  The deployed executor "
            f"walks those rows as a Cartesian path and latches its own "
            f"nullspace, so it cannot follow a branch change.  Nothing "
            f"written.")

    t_note = dict(
        note=T_S_NOTE, fps=float(FPS), n_rows=int(len(t_s)),
        n_synth_rows=int(n_synth),
        t_first_s=float(t_s[0]), t_last_s=float(t_s[-1]),
        programme_duration_s=float(dur), joint_speed=sa,
        executor=EXECUTOR_NOTE, rows=summary["rows"],
        park=park_pose(arm))
    csv_path, man_path = _write_csv(pw, out_dir, name, times=t_s,
                                    t_note=t_note)
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
                          q_peak_frac_of_limit=sa["at_csv_rows"]["worst_frac"],
                          columns=CSV_COLUMNS_T,
                          t_s=dict(first_s=float(t_s[0]),
                                   last_s=float(t_s[-1]),
                                   n_synth_rows=int(n_synth), note=T_S_NOTE))
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
    # THE MEASURED PAPER OFFSET LIVES IN THE SITE FILE, so a run that does not
    # say otherwise uses the number somebody measured with a ruler and wrote
    # down, rather than silently planning against the nominal plane.
    # --measured-float H IS --paper-z (0.030 - H) AND NOTHING ELSE.  The ruler
    # reads the gap; the plan needs the offset; doing that subtraction in a
    # person's head at the rig is how a sign error gets into the paper.
    if a.measured_float is not None:
        if a.paper_z:
            raise Refused("--measured-float and --paper-z say the same thing "
                          "two ways; give one.")
        pz = HOVER_DEFAULT - float(a.measured_float)
    elif a.paper_z:
        pz = float(a.paper_z)
    else:
        pz = float(slot_cfg(arm).get("paper_z") or 0.0)
    for ln in assumptions([arm]):
        print(ln)
    if a.measured_float is not None:
        print(f"  --measured-float {1000 * a.measured_float:.1f} mm: the "
              f"floating pass asked for {1000 * HOVER_DEFAULT:.0f} and the "
              f"ruler read {1000 * a.measured_float:.1f}, so the paper sits "
              f"{1000 * pz:+.1f} mm from where the model puts it")
    if pz:
        print(f"  --paper-z {1000 * pz:+.1f} mm: the real paper sits that far "
              f"ABOVE the modelled plane, and the whole plan is lifted by it")
    print(f"arm {arm}: the word {WORD!r} alone, "
          + (f"HOVER {1000 * hov:.0f} mm above the paper — no ink, no contact"
             if hov > 0 else "ON THE PAPER"))
    print(f"  the other five arms stand at their parks for the whole programme; "
          f"every gate is graded against all six.")
    t0 = time.perf_counter()
    r = plan_word(arm, width=float(a.width), height=float(a.height),
                  dy=float(a.dy), name=a.name, hover=hov,
                  out_dir=Path(a.out), allow_partial=bool(a.allow_partial),
                  paper_z=pz)
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
    print()
    for ln in _file_pose_lines(s["files"]["csv"], arm):
        print(ln)
    print()
    print(f"  send it:  scripts/day1.py send --arm {arm} "
          f"--file {s['files']['csv']} --from-q <measured joints>")
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
# `site` — show and edit the one file everything site-specific lives in
# ===========================================================================
def _set_site(st, dotted):
    """`slot31.arm=97` -> the edited dict. -> (path list, old, new).

    `slot31` and `slot71` are spelled the way a person types them; everything
    else is the literal JSON path.  The value is parsed as JSON when it can be
    (so `97`, `true`, `null`, `0.003` keep their types) and kept as a string
    when it cannot.
    """
    key, _, raw = str(dotted).partition("=")
    if not _:
        raise Refused(f"--set wants KEY=VALUE; got {dotted!r}")
    parts = []
    for k in key.strip().split("."):
        if k.startswith("slot") and k[4:].isdigit():
            parts += ["slots", k[4:]]
        else:
            parts.append(k)
    try:
        val = json.loads(raw)
    except json.JSONDecodeError:
        val = raw
    node = st
    for k in parts[:-1]:
        if k not in node or not isinstance(node[k], dict):
            raise Refused(f"--set {key}: {'.'.join(parts[:parts.index(k)+1])} "
                          f"is not in the site file")
        node = node[k]
    old = node.get(parts[-1], "<absent>")
    node[parts[-1]] = val
    return parts, old, val


def cmd_site(a):
    st = site(a.site) if getattr(a, "site", None) else site()
    path = Path(st["_path"])
    if a.set:
        changed = []
        for one in a.set:
            parts, old, val = _set_site(st, one)
            changed.append(f"  {'.'.join(parts)}: {old!r} -> {val!r}")
        doc = {k: v for k, v in st.items() if k != "_path"}
        doc["_updated"] = __import__("datetime").date.today().isoformat()
        path.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"edited {path}")
        for ln in changed:
            print(ln)
        print()
        site(reload=True)
    for ln in site_lines():
        print(ln)
    if not a.set:
        print("edit it:  day1.py site --set slot31.arm=97 "
              "--set slot31.mounted=true")
        print("          day1.py site --set slot71.paper_z=0.003")
        print("          day1.py site --set operator.host=USER@HOST")
    return 0


# ===========================================================================
# `park` — the pose every file starts at, and a certified way back to it
# ===========================================================================
PARK_DOC = """The park pose, and a certified joint path to it.

WHY THIS EXISTS.  The deployed impedance executor's FIRST move is an
UNCERTIFIED straight ramp from wherever the arm is standing to row 0 of the
file, and no gate in this repository has anything to say about it.  `park`
answers the two questions that ramp raises:

  day1.py park --arm 31
      prints the park joint vector and the park tip pose in fr3_link0 — the
      pose to drive the arm to, under position control, before streaming
      anything (the operator's go_start_pos.py).

  day1.py park --arm 31 --from-q <the arm's MEASURED joints>
      plans a collision-free joint path from that configuration to the park
      with the repo's own C-space RRT (`aris_sixarm.transit.plan` — the same
      tier `paper.route` falls back to), against the static scene, the seam
      bars and the other five arms standing at their parks, certifies it with
      `scene_check` (self, frame, paper, inter-arm) and writes
      out/day1/park_<arm>.{npz,csv,json}.

READ THIS BEFORE BELIEVING THE CSV.  The deployed impedance executor FOLLOWS
TIP POSES ONLY and latches its own nullspace, so this joint path is certified
FOR A JOINT-CAPABLE EXECUTOR and is, on the deployed stack, a CHECK: it says a
collision-free way from here to the park exists and what it costs, and it is
the thing to compare the Cartesian ramp against.  THE WAY TO ACTUALLY GET THERE
TODAY IS THE POSITION-CONTROL MOVE (go_start_pos.py), not this file."""

PARK_DENSE = 65          # configurations per RRT leg handed to scene_check


def _travel_rows(spec, Q, tool, h_inv, intensity=1.0):
    """A joint path -> pen-up `travel` CSV rows + their clock. -> (rows, t_s).

    Built with the exporter's OWN row formatter and FK so the file is the same
    text the pathway exporter would write, and paced the way `writing` paces a
    pen-up move — `QD_FRAC` of the joint-velocity limit — so `t_s` means the
    same thing here as it does in a drawing file.
    """
    pen_lat, pen_ext = pathway.tool_offsets(tool)
    Q = np.asarray(Q, float).reshape(-1, 7)
    T, tip = pathway._tip_of(Q, pen_ext, pen_lat)
    quats = pathway.quats_from_R(T[:, :3, :3])
    rows = [pathway._fmt_row(0, i, pathway.KIND_TRAVEL, tip[i], quats[i],
                             intensity, Q[i]) for i in range(len(Q))]
    step = np.abs(np.diff(Q, axis=0)) / (np.asarray(frames.QD_MAX, float)
                                         * writing.QD_FRAC)
    dt = np.maximum(step.max(axis=1), 1e-4) if len(Q) > 1 else np.zeros(0)
    return rows, np.concatenate([[0.0], np.cumsum(dt)])


def plan_park(arm, q_from=None, out_dir=OUT_DIR, write=True, verbose=True,
              dense=PARK_DENSE):
    """The park pose, or a certified path to it from `q_from`. -> dict."""
    arm = int(arm)
    fl, h = rt.fleet_for(None, None, "uniform")
    if arm not in fl:
        raise Refused(f"arm {arm} is not in rig {os.environ['ARIS_RIG']!r} "
                      f"(has {sorted(fl)})")
    spec = fl[arm]
    pose = park_pose(arm)
    if q_from is None:
        return dict(summary=dict(arm=arm, park=pose, path=None), report=None)

    q0 = np.asarray(q_from, float).reshape(7)
    q1 = np.asarray(spec.q_seed, float).reshape(7)
    pen = float(frames.ext_of(None))
    # THE SAME TIER `paper.route` FALLS BACK TO, with the same floors it hands
    # in: the static scene (which is where the seam bars and the neighbours'
    # base columns come from — `paper.static_boxes`), the paper, and the arm's
    # own metal.  Nothing about the search is new here.
    boxes = paper.static_boxes(spec)
    # THE FLOORS ARE CLAMPED TO THE ENDPOINTS, exactly as `paper.route` clamps
    # them, and that is not a detail: the park itself stands closer to the
    # frame than `FRAME_FLOOR`, so a search held to the nominal floor has an
    # infeasible GOAL and returns None for every start.  `effective_*` lower
    # the floor to whatever the two ends already achieve, which is the only
    # question a transit can be asked (`paper.effective_static_floor` explains
    # why the clamp is not optional).
    tip_fl, chain_fl = paper.effective_floors(spec, q0, q1, pen, h,
                                              paper.TIP_CLEAR,
                                              paper.CHAIN_CLEAR)
    static_fl = (paper.effective_static_floor(spec, q0, q1, pen, h, boxes=boxes)
                 if boxes and paper.STATIC_SAFE else -np.inf)
    self_fl = (paper.self_floor(spec, q0, q1, pen) if paper.SELF_SAFE
               else -np.inf)
    t0 = time.perf_counter()
    vias = transit.plan(spec, q0, q1, pen_ext=pen, h_inv=h, boxes=boxes,
                        chain_floor=chain_fl, tip_floor=tip_fl,
                        static_floor=static_fl, self_floor=self_fl)
    t_rrt = time.perf_counter() - t0
    if vias is None:
        raise Refused(
            f"no collision-free joint path from that configuration to arm "
            f"{arm}'s park: the C-space RRT spent its budget "
            f"({transit.ATTEMPTS} attempts x {transit.TIME_BUDGET:g} s, "
            f"{transit.MAX_NODES} nodes) and found none.  Either the start "
            f"pose is itself in collision — check it against the robot state "
            f"you read it from — or the way out is narrow.  Nothing written.")
    knots = [q0] + [np.asarray(v, float).reshape(7) for v in vias] + [q1]
    Q = np.vstack([paper.line_samples(u, v, dense)
                   for u, v in zip(knots[:-1], knots[1:])])
    # drop the duplicated join between legs
    keep = np.concatenate([[True], (np.abs(np.diff(Q, axis=0)).max(axis=1)
                                    > 1e-12)])
    Q = Q[keep]
    M = len(Q)

    rows, t_s = _travel_rows(spec, Q, os.environ["ARIS_TOOL"], h)
    dq = np.abs(np.diff(Q, axis=0)).max(axis=1) if M > 1 else np.zeros(1)

    # ---- the independent check, the other five arms at their parks --------
    qt = {a: (Q if a == arm else np.tile(np.asarray(fl[a].q_seed, float),
                                         (M, 1))) for a in sorted(fl)}
    margin = float(coordination.PAIR_MARGIN)
    dt = float(t_s[-1]) / max(M - 1, 1) if M > 1 else DT
    rep = scene_check.check_timeline(
        qt, dt, margin, programs=None, h_inv=h,
        pen_ext={a: pen for a in sorted(fl)}, sub=SUB, verbose=False,
        fleet=fl, drawing={a: np.zeros(M, bool) for a in sorted(fl)})
    g = _gate_numbers(rep, margin)
    if verbose:
        print(f"  RRT: {len(vias)} vias, {M} configurations, "
              f"{float(t_s[-1]):.2f} s at {100 * writing.QD_FRAC:.0f} % of the "
              f"joint limit ({t_rrt:.2f} s of search)")
        for ln in rt.summarise(rep, margin):
            print("  " + ln)
    if not rep["ok"]:
        raise Refused(
            f"the path to the park does NOT certify (frame "
            f"{rep.get('frame_failed')}, paper {rep.get('paper_failed')}, "
            f"column {rep.get('column_failed')}, self {rep.get('self_failed')},"
            f" min inter-arm {1000 * rep['min_clearance']:.1f} mm against a "
            f"{1000 * margin:.0f} mm gate).  Nothing written.")

    sa = speed_audit(t_s, Q)
    summary = dict(
        arm=arm, certified=True, kind="park",
        park=pose, from_q=[float(v) for v in q0],
        rrt=dict(n_vias=len(vias), n_configs=int(M), wall_s=float(t_rrt),
                 step_s=float(transit.STEP), attempts=int(transit.ATTEMPTS),
                 max_nodes=int(transit.MAX_NODES),
                 static_floor_m=float(static_fl),
                 self_floor_m=float(self_fl),
                 chain_floor_m=float(chain_fl),
                 tip_floor_m=float(tip_fl),
                 floors_note="clamped to the two endpoints the way "
                             "`paper.route` clamps them; the nominal floors "
                             "are TIP_CLEAR/CHAIN_CLEAR/FRAME_FLOOR/"
                             "SELF_PLAN_MARGIN"),
        duration_s=float(t_s[-1]),
        rows=dict(n_rows=int(M), max_dq_rad=float(dq.max()),
                  bound_rad=float(ROW_DQ_BOUND),
                  reconfigures=bool(dq.max() > ROW_DQ_BOUND),
                  n_over_bound=int((dq > ROW_DQ_BOUND).sum())),
        joint_speed=sa,
        rig=os.environ["ARIS_RIG"], tool=os.environ["ARIS_TOOL"], h_m=float(h),
        seam_bars=dict(on=bool(mounts.SEAM_POSTS_ON), source=mounts.SEAM_SOURCE),
        frozen_arms={str(a): [float(v) for v in fl[a].q_seed]
                     for a in sorted(fl) if a != arm},
        gates=g, executor=EXECUTOR_NOTE,
        what_this_is=(
            "A JOINT path, certified for a JOINT-CAPABLE executor.  The "
            "deployed impedance executor follows tip poses only and latches "
            "its own nullspace, so on that stack this file is a CHECK — it "
            "says a collision-free way from here to the park exists and what "
            "it costs — and the way to actually get there today is the "
            "position-control move (go_start_pos.py)."))
    if not write:
        return dict(summary=summary, report=rep, rows=rows, t_s=t_s, Q=Q)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"park_{arm}"
    seg = {a: np.full(M, -1, np.int64) for a in sorted(fl)}
    npz_path = out_dir / f"{stem}.npz"
    np.savez_compressed(npz_path, **_payload(qt, seg, {a: pen for a in fl},
                                             margin,
                                             float(rep["min_clearance"]), stem))
    csv_path = out_dir / f"{stem}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS_T)
        w.writerows([list(r) + [f"{float(t):.6f}"] for r, t in zip(rows, t_s)])
    summary["files"] = dict(npz=str(npz_path), csv=str(csv_path),
                            summary=str(out_dir / f"{stem}.json"))
    (out_dir / f"{stem}.json").write_text(
        json.dumps(summary, indent=1, default=str) + "\n")
    return dict(summary=summary, report=rep, rows=rows, t_s=t_s, Q=Q,
                csv=csv_path)


def _q7(text, what="--from-q"):
    parts = [p for p in str(text).replace(",", " ").split() if p]
    if len(parts) != 7:
        raise Refused(f"{what} wants SEVEN joint values in radians "
                      f"(q1,...,q7); got {len(parts)}")
    try:
        return np.array([float(v) for v in parts], float)
    except ValueError:
        raise Refused(f"{what} wants seven numbers; got {text!r}")


def cmd_park(a):
    arm = int(a.arm)
    pose = park_pose(arm)
    print(f"ARM {arm} PARK POSE — where the certified PROGRAMME starts and "
          f"ends, and where to")
    print(f"put the arm AFTER a run.  IT IS NOT WHERE A RUN STARTS: the "
          f"exporter begins the")
    print(f"CSV at the first frame whose tip is 50 mm clear of the paper, so "
          f"row 0 is about")
    print(f"0.6 m and 4 rad from here on the day-1 word files.  THE START POSE "
          f"IS ROW 0 —")
    print(f"`day1.py send` prints it, and that is the pose to drive to under "
          f"position control.")
    print(f"This pose is what `--from-q <end joints>` plans a certified path "
          f"BACK to.")
    for ln in _park_lines(arm):
        print(ln)
    if not a.from_q:
        print()
        print("  --from-q q1,...,q7  (the arm's MEASURED joints, read off the "
              "robot state)")
        print("  plans and certifies a collision-free joint path from there "
              "to this pose.")
        return 0
    q0 = _q7(a.from_q)
    print()
    print(f"  from q  = " + ", ".join(f"{v:+.6f}" for v in q0))
    print(f"  max joint difference from the park: "
          f"{float(np.abs(q0 - np.asarray(pose['q'], float)).max()):.4f} rad")
    r = plan_park(arm, q0, out_dir=Path(a.out))
    s = r["summary"]
    print()
    print(_one_liner("park", arm, s["gates"], s["duration_s"], True)
          + f"  {s['rrt']['n_configs']} rows  {s['rrt']['n_vias']} vias")
    print(_speed_line(s["joint_speed"]))
    print(f"  max dq/row {s['rows']['max_dq_rad']:.4f} rad (bound "
          f"{s['rows']['bound_rad']:.4f})"
          + ("  ! RECONFIGURES" if s["rows"]["reconfigures"] else ""))
    print(f"  wrote {s['files']['csv']}")
    print(f"        {s['files']['summary']}")
    print()
    for ln in _wrap(s["what_this_is"], "  "):
        print(ln)
    return 0


# ===========================================================================
# `send` — the interface to the arms.  File in, one command out.
# ===========================================================================
SEND_DOC = """Deliver ONE pathway CSV to the operator box and print the ONE
command that draws it.  This machine never moves an arm: `send` copies the file
with scp and then PRINTS the supervisor command; `--live` is the only way it is
ever run over ssh, and `--dry-run` copies nothing at all.

THE CHAIN, as confirmed on the deployed branches in the 2026-09-17 hardware
session.  THERE IS NO RUNNER FOR ARM 31 OR 71: the supervisor takes the CSV as
its first positional argument and that is the whole interface.

    out/day1/<name>_<N>.csv
      -- scp -->  <operator.host>:<remote_csv>, both from config/site.json
      -- ssh host 'RTFF_CONTACT_DESCEND=0 RTFF_FORCE_SIGN=1
                   RTFF_TRAVEL_SPEED=0.02 RTFF_MODE=observe
                   RTFF_DEPART_LIFT=0 ARM_ID=<N>
                   bash ~/RTff/draw_rtff_supervised.sh <csv> 1.0 2.5 5 fresh'
            -> ladder gate (position control)   [SKIPPED on inverted arms]
            -> MoveIt to the start (position control)
            -> pen_switch down  (cartesian_impedance_controller)
            -> rtff_pathway_exec.py --csv <csv>   ... the drawing
            -> pen_switch up    (fr3_arm_controller)

THE FIVE RTFF_* VARIABLES ARE NOT DECORATION.  Each is off by default and each
default is wrong for this file: CONTACT_DESCEND=0 flies the planned z instead
of feeling for the paper, MODE=observe keeps depth open-loop for the first
passes, TRAVEL_SPEED=0.02 is 20 mm/s, and DEPART_LIFT=0 is because our
programmes already END AT THE PARK POSE — a depart lift on top of that is
motion past the end of the certificate.

WHAT THE EXECUTOR READS.  The tip pose of each row, and nothing else: it paces
by CARTESIAN ARC LENGTH, holds each row's orientation without slerping, and
latches its own nullspace.  IT IGNORES q1..q7 AND t_s.  Those columns are the
record of what the planner chose and the basis of the velocity certificate;
on the deployed stack they are a check, not a command.

THE START POSE IS ROW 0, NOT THE PARK.  The executor ramps in an uncertified
straight line from the arm's measured configuration to row 0, and it cannot
execute a joint transit, so the arm is driven to ROW 0's JOINTS under POSITION
control (the operator's go_start_pos.py) and the ramp is then zero.  `send`
prints those joints as START POSE, and `--from-q <measured joints>` REFUSES to
dispatch unless the arm is already within 0.05 rad and 10 mm of them
(`--allow-ramp` overrides, deliberately).  The arm then STOPS at the last row
(RTFF_DEPART_LIFT=0), which `send` prints as END POSE; return it to the park
under position control afterwards, and `day1.py park --arm N --from-q <end
joints>` is the certified check that the way back is clear.

Plain `ssh host 'cmd'`, key-only — not `bash -lc`, which would source a
profile an ssh session has not got."""


def _csv_rows(path):
    with open(path) as f:
        return sum(1 for ln in f if ln.strip()) - 1


def _wrap(text, indent="  ", width=78):
    import textwrap
    return textwrap.wrap(" ".join(str(text).split()), width=width,
                         initial_indent=indent, subsequent_indent=indent)


def park_pose(arm, tool=None):
    """The arm's park: its joints, and its tip pose in fr3_link0. -> dict.

    THE POSE ROW 0 OF EVERY FILE THIS SCRIPT WRITES STARTS AT, and therefore
    the pose the arm has to be standing in before anything is streamed: the
    deployed executor's FIRST move is an uncertified straight ramp from the
    arm's measured configuration to row 0, and no gate in this repository has
    anything to say about that ramp.  `fr3_link0` is the arm's own base frame,
    which is the frame the CSV's xyz/quaternion are already in.
    """
    arm = int(arm)
    fl, _ = rt.fleet_for(None, None, "uniform")
    if arm not in fl:
        raise Refused(f"arm {arm} is not in rig {os.environ['ARIS_RIG']!r}")
    spec = fl[arm]
    q = np.asarray(spec.q_seed, float).reshape(7)
    pen_lat, pen_ext = pathway.tool_offsets(tool or os.environ["ARIS_TOOL"])
    T, tip = pathway._tip_of(q[None, :], pen_ext, pen_lat)
    return dict(arm=arm, q=[float(v) for v in q],
                tip_xyz_base_m=[float(v) for v in tip[0]],
                tip_quat_xyzw=[float(v) for v in
                               pathway.quat_xyzw(T[0, :3, :3])],
                frame="fr3_link0", tool=tool or os.environ["ARIS_TOOL"])


# --- THE START POSE IS ROW 0, AND IT IS NOT THE PARK -----------------------
# MEASURED, and it decided the interface: the certified PROGRAMME parks at both
# ends, but the exporter begins the CSV at the first frame whose tip is 50 mm
# clear of the paper — 641.7 mm and 4.16 rad from the park on the arm-31 word.
# The deployed executor ramps from wherever the arm is standing to row 0 in a
# straight line that nothing in this repository certifies, AND it cannot
# execute our park->hover joint transit (it walks rows as a Cartesian path with
# its own nullspace).  So the arm is driven to ROW 0 under position control and
# the executor's ramp is then zero.  Every command that writes or sends a file
# prints the row-0 joints for exactly that purpose.
START_GATE_RAD = 0.05     # per joint, between the measured pose and row 0
START_GATE_M = 0.010      # at the tip


def _row_pose(row):
    """One CSV row -> dict(q, tip_xyz, tip_quat).  Raises on a malformed row."""
    return dict(q=[float(v) for v in row[11:18]],
                tip_xyz_base_m=[float(row[3]), float(row[4]), float(row[5])],
                tip_quat_xyzw=[float(row[6]), float(row[7]), float(row[8]),
                               float(row[9])])


def _csv_ends(path):
    """The first and last data rows of a pathway CSV. -> (row0, rowN)."""
    with open(path) as f:
        rd = csv.reader(f)
        next(rd, None)
        rows = [r for r in rd if r]
    if not rows:
        raise Refused(f"{path} has a header and no rows.")
    return rows[0], rows[-1]


def _pose_lines(tag, pose, indent="     "):
    return [indent + f"{tag} q1..q7 = "
            + ", ".join(f"{v:+.6f}" for v in pose["q"]),
            indent + f"{' ' * len(tag)} tip (fr3_link0) xyz = "
            + ", ".join(f"{v:+.6f}" for v in pose["tip_xyz_base_m"])
            + "  quat xyzw = "
            + ", ".join(f"{v:+.6f}" for v in pose["tip_quat_xyzw"])]


def _start_end_lines(row0, rowN, arm, indent="     "):
    """What to do before and after the run, as the poses to do it with."""
    out = [indent + "START POSE = ROW 0 of this file.  Move the arm HERE under "
                    "POSITION control first"]
    out += [indent + "(the operator's go_start_pos.py, with these joints); the "
                     "executor's ramp is then zero."]
    out += _pose_lines("START", _row_pose(row0), indent)
    out += ["", indent + "END POSE = the LAST row.  RTFF_DEPART_LIFT=0, so the "
                         "executor stops here.",
            indent + f"Return to park under POSITION control afterwards; "
                     f"`day1.py park --arm {arm} --from-q <end joints>`",
            indent + "gives the certified path and the check that it is clear."]
    out += _pose_lines("END  ", _row_pose(rowN), indent)
    return out


def _tip_of_q(q):
    """FK tip of one configuration, in the arm's own base frame. -> (3,)."""
    pen_lat, pen_ext = pathway.tool_offsets(os.environ["ARIS_TOOL"])
    _T, tip = pathway._tip_of(np.asarray(q, float).reshape(1, 7), pen_ext,
                              pen_lat)
    return np.asarray(tip[0], float)


def _ramp_check(row0, q_from):
    """How far the measured pose is from row 0. -> dict.

    Both distances, because either one alone can be small while the other is
    not: a wrist roll moves 2 rad and no millimetres, and a shoulder nudge
    moves 10 mm and almost no radians.  Both are the ramp the executor flies.
    """
    p0 = _row_pose(row0)
    q0 = np.asarray(p0["q"], float)
    qm = np.asarray(q_from, float).reshape(7)
    dq = np.abs(qm - q0)
    d_tip = float(np.linalg.norm(_tip_of_q(qm)
                                 - np.asarray(p0["tip_xyz_base_m"], float)))
    return dict(max_dq_rad=float(dq.max()),
                worst_joint=int(np.argmax(dq)) + 1,
                per_joint_rad=[float(v) for v in dq],
                tip_m=d_tip,
                gate_rad=float(START_GATE_RAD), gate_m=float(START_GATE_M),
                ok=bool(dq.max() <= START_GATE_RAD and d_tip <= START_GATE_M))


def _ramp_lines(row0, q_from, allowed, indent="     "):
    c = _ramp_check(row0, q_from)
    out = [indent + f"measured pose -> row 0: {c['max_dq_rad']:.4f} rad "
                    f"(worst j{c['worst_joint']}, gate "
                    f"{c['gate_rad']:.2f}), {1000 * c['tip_m']:.1f} mm at the "
                    f"tip (gate {1000 * c['gate_m']:.0f})"]
    if c["ok"]:
        out.append(indent + "the executor's ramp is effectively zero.  Good.")
    elif allowed:
        out.append(indent + "! --allow-ramp: dispatching anyway.  That gap is "
                            "flown as an UNCERTIFIED straight line.")
    return out


def _file_pose_lines(csv_path, arm, indent="  "):
    """The two poses a person needs before and after streaming a file."""
    try:
        row0, rowN = _csv_ends(csv_path)
    except (Refused, OSError):
        return []
    return _start_end_lines(row0, rowN, arm, indent)


def _gate_ramp(row0, q_from, allowed):
    """Refuse to dispatch a file whose row 0 is far from the measured pose."""
    c = _ramp_check(row0, q_from)
    if c["ok"] or allowed:
        return c
    raise Refused(
        f"the arm is NOT at this file's start pose: "
        f"{c['max_dq_rad']:.4f} rad on joint {c['worst_joint']} (gate "
        f"{c['gate_rad']:.2f}) and {1000 * c['tip_m']:.1f} mm at the tip "
        f"(gate {1000 * c['gate_m']:.0f}).\n"
        f"  The executor would fly that gap as an UNCERTIFIED straight ramp.  "
        f"Move the arm to row 0\n"
        f"  under POSITION control first (go_start_pos.py with the START "
        f"joints below), then send again.\n"
        + "\n".join(_pose_lines("START", _row_pose(row0), "  "))
        + "\n  --allow-ramp dispatches anyway, deliberately.  Nothing copied.")


def _summary_beside(csv_path):
    """The `<stem>.json` this CSV was written with, if it is there. -> dict|None."""
    p = Path(csv_path)
    cand = p.with_suffix(".json")
    if cand.exists():
        try:
            return json.loads(cand.read_text())
        except Exception:
            return None
    return None


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
    slot = int(a.arm)
    # THE SLOT IS THE PLAN, THE ARM IS THE ROBOT, AND THEY NEED NOT MATCH.
    # `--arm` names the POSITION the word was planned for (31 = left-middle,
    # 71 = right-middle); `--as-arm` names the physical arm bolted into it.
    # The CSV's poses are in that POSITION's base frame, which is the physical
    # arm's base frame for exactly as long as it is mounted there — so the one
    # thing that has to be true is the mounting, and the printed line says so.
    arm = int(a.as_arm) if a.as_arm else int(slot_cfg(slot).get("arm", slot))
    st = site()
    host = a.host or st["operator"]["host"]
    remote_csv = (a.remote or st["remote_csv"]).format(arm=arm)
    log = st["log"].format(arm=arm)
    fmin, fmax, levels = st["force"]
    env = " ".join(f"{k}={v}" for k, v in (st.get("rtff_env") or {}).items())
    scp = f"scp {src} {host}:{remote_csv}"
    # THE COMMAND, VERBATIM.  Plain `ssh host 'cmd'` — not `bash -lc`, which
    # would source a profile the session does not have.  The supervisor takes
    # the CSV as $1; ARM_ID is the DDS domain and the only thing that says
    # which robot.
    inner = (f"{env} ARM_ID={arm} bash {st['supervisor']} {remote_csv} "
             f"{fmin} {fmax} {levels} {st['mode']}")
    draw = f"ssh {host} '{inner}'"
    stack = f"ssh {host} 'bash {st['stack_check']} stack {arm}'"
    watch = f"ssh {host} 'tail -f {log}'"

    for ln in site_lines():
        print(ln)
    row0, rowN = _csv_ends(src)
    # THE GATE ON THE RAMP, AND IT REFUSES BEFORE IT COPIES.  Handed the
    # measured joints, `send` will not dispatch a file whose row 0 is far from
    # where the arm is actually standing: that gap is flown as an uncertified
    # straight line and it is the one motion of the day nothing has graded.
    if a.from_q:
        _gate_ramp(row0, _q7(a.from_q, "--from-q"), bool(a.allow_ramp))
    where = slot_cfg(slot).get("position", f"slot {slot}")
    ip = ((st.get("arms") or {}).get(str(arm)) or {}).get("ip")
    if arm == slot:
        print(f"ARM {arm}   {src}  ({_csv_rows(src)} rows)")
    else:
        print(f"slot {slot} -> arm {arm}  (base frame of the {where} "
              f"position)")
        print(f"  {src}  ({_csv_rows(src)} rows)")
        print(f"  the poses in this file are the {where} POSITION's base "
              f"frame.  They are")
        print(f"  arm {arm}'s base frame for exactly as long as arm {arm} is "
              f"bolted into that")
        print(f"  position.  CONFIRM THE MOUNTING BEFORE YOU RUN IT — which "
              f"arm ids are in")
        print(f"  the two middle positions is NOT known from this repository.")
    print(f"  operator  {host}")
    print(f"  ARM_ID    {arm}   (the DDS domain; control box "
          f"{ip or 'UNKNOWN — not in the site file'})")
    print(f"  lands at  {remote_csv}")
    print()
    if a.dry_run:
        print(f"DRY RUN — nothing copied.  The copy would be:\n  {scp}")
    else:
        print(f"  $ {scp}")
        r = subprocess.run(scp, shell=True)
        if r.returncode != 0:
            raise Refused(f"`{scp}` exited {r.returncode}.  The file is NOT on "
                          f"the operator box.  Check the address at the top of "
                          f"scripts/day1.py (or --host), and that the key is "
                          f"loaded — password auth is disabled.")
        print("  copied.")
    print()
    print("1. WHERE THE ARM MUST BE STANDING, AND WHERE IT WILL STOP.  The "
          "executor's FIRST")
    print("   move is an UNCERTIFIED straight ramp from the arm's measured "
          "configuration to")
    print("   row 0, and it cannot execute a joint transit — it walks rows as "
          "a Cartesian")
    print("   path with its own nullspace.  So the start pose IS ROW 0, not "
          "the park:")
    for ln in _start_end_lines(row0, rowN, slot):
        print(ln)
    if a.from_q:
        for ln in _ramp_lines(row0, _q7(a.from_q, "--from-q"),
                              bool(a.allow_ramp)):
            print(ln)
    print()
    print("2. THE STACK MUST BE HEALTHY — hardware active, three controllers, "
          "robot_mode 2.")
    print("   Never heal on a user stop (mode 5) or guiding (mode 3):")
    print(f"     {stack}")
    print()
    print("3. THEN, exactly this one command:")
    print(f"     {draw}")
    print()
    print("4. WATCH IT:")
    print(f"     {watch}")
    print()
    for ln in _wrap(EXECUTOR_NOTE, "   "):
        print(ln)
    print()
    print(f"ARM {arm} IS INVERTED, so the supervisor SKIPS the ladder gate: "
          f"nothing downstream")
    print(f"will measure the paper plane for it, and RTFF_CONTACT_DESCEND=0 "
          f"means it will not")
    print(f"feel for it either.  The z_m baked into this file IS the plane it "
          f"will draw at.")
    print(f"Fly the hover pass, measure the tip with a ruler at three points, "
          f"and re-plan with")
    print(f"`word --arm {arm} --paper-z <measured>` if it is not where the "
          f"file says.")
    print("ABORT: the physical e-stop.  The software gate and the 20 mrad "
          "watchdog are not the abort path.")
    if a.live:
        print()
        print("--live: running it over ssh NOW.")
        print(f"  $ {draw}")
        r = subprocess.run(draw, shell=True)
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
    ap.add_argument("--site", default=None, metavar="FILE",
                    help="the site configuration (default config/site.json, "
                         "or $ARIS_SITE).  Everything machine-specific is in "
                         "it; `day1.py site` shows and edits it.")
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
    w.add_argument("--measured-float", type=float, default=None, metavar="M",
                   dest="measured_float",
                   help=f"what the RULER READ during the floating pass, in "
                        f"metres.  The human-facing alias for --paper-z: it "
                        f"computes --paper-z = {HOVER_DEFAULT:g} - H, so "
                        f"nobody does arithmetic at the rig.  A float that "
                        f"measured 27 mm means the paper is 3 mm higher than "
                        f"the model: --measured-float 0.027.")
    w.add_argument("--paper-z", type=float, default=0.0, metavar="M",
                   dest="paper_z",
                   help="metres the REAL paper surface sits ABOVE the modelled "
                        "plane; positive lifts the whole plan by that much "
                        "(default 0).  After a hover pass measuring H metres "
                        "of tip gap where 0.030 was asked for, pass "
                        "--paper-z (0.030 - H).  Same lever as --hover: it is "
                        "a longer pen in the plan, graded with the real one.")
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

    st = sub.add_parser(
        "site", formatter_class=argparse.RawDescriptionHelpFormatter,
        help="show (and edit) the one file everything site-specific lives in",
        description="EVERYTHING SITE-SPECIFIC IS IN `config/site.json` — the "
                    "operator host, each slot's physical arm id, its control "
                    "box IP, its DDS domain, its measured paper z and whether "
                    "the mounting is confirmed, plus the RTFF_* environment "
                    "the dispatch line carries.  Nothing in this script "
                    "hard-codes any of it, every run prints it, and the GUI "
                    "reads and writes the same file.")
    st.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="edit one field and write the file back, e.g. "
                         "`--set slot31.arm=97`, `--set slot31.mounted=true`, "
                         "`--set slot71.paper_z=0.003`.  Repeatable.")
    st.set_defaults(func=cmd_site)

    k = sub.add_parser(
        "park", formatter_class=argparse.RawDescriptionHelpFormatter,
        help="the pose every file starts at, and a certified joint path to it",
        description=PARK_DOC)
    k.add_argument("--arm", type=int, required=True, choices=sorted(ARMS))
    k.add_argument("--from-q", dest="from_q", default=None,
                   metavar="q1,...,q7",
                   help="the arm's MEASURED joints in radians, read off the "
                        "robot state.  Given one, a collision-free joint path "
                        "from there to the park is planned and certified.")
    k.add_argument("--out", default=str(OUT_DIR))
    k.set_defaults(func=cmd_park)

    s = sub.add_parser(
        "send", formatter_class=argparse.RawDescriptionHelpFormatter,
        help="deliver a pathway CSV to the operator box and print the one "
             "command that runs it",
        description=SEND_DOC)
    s.add_argument("--arm", type=int, required=True,
                   help="the SLOT this file was planned for: 31 = left-middle "
                        "position, 71 = right-middle.  It names the base frame "
                        "the poses are in, not necessarily the robot.")
    s.add_argument("--as-arm", dest="as_arm", type=int, default=None,
                   metavar="ID",
                   help="the PHYSICAL arm id to dispatch to (its DDS domain, "
                        "its ARM_ID and its /tmp file name).  Default: the "
                        "same id as --arm.  Use it when the arm bolted into "
                        "that position is not the one the plan is named after "
                        "— which arms are mounted is NOT known from this "
                        "repository, so identify the arm first.")
    s.add_argument("--file", required=True, metavar="CSV",
                   help="the pathway CSV, e.g. out/day1/unknown_31.csv")
    s.add_argument("--host", default=None, metavar="USER@HOST",
                   help="the operator box (default: the site file's "
                        "`operator.host`)")
    s.add_argument("--remote", default=None, metavar="PATH",
                   help="where the CSV lands on the operator box (default: "
                        "the site file's `remote_csv`)")
    s.add_argument("--from-q", dest="from_q", default=None,
                   metavar="q1,...,q7",
                   help="the arm's MEASURED joints (the GUI's Identify view "
                        "shows them).  Given these, `send` REFUSES to "
                        "dispatch unless the arm is already at row 0 — within "
                        f"{START_GATE_RAD:g} rad on every joint and "
                        f"{1000 * START_GATE_M:.0f} mm at the tip.")
    s.add_argument("--allow-ramp", action="store_true",
                   help="dispatch even though the arm is not at row 0.  The "
                        "executor then flies that gap as an UNCERTIFIED "
                        "straight ramp.  Deliberate only.")
    s.add_argument("--dry-run", action="store_true",
                   help="print the scp and the run command, copy nothing")
    s.add_argument("--live", action="store_true",
                   help="RUN the draw command over ssh instead of printing it. "
                        "Without this the arm never moves from this machine.")
    s.set_defaults(func=cmd_send)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    if getattr(a, "site", None):
        site(a.site)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
