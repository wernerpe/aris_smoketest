#!/usr/bin/env python3
"""T8 report: the four SIL runs side by side, plant truth vs executor belief.

    python3 t8_report.py <out>/t8 [--json <file>]

Left half of every column is what the PLANT did (`sil/summary.json`, i.e.
`aris_sixarm.sil.trace.metrics` over the Drake trace): touch fraction, contact
force, where the PHYSICAL tip actually was.  Right half is what the EXECUTOR
believed (`exec.log` + its own per-tick `force.csv`): the press it settled at,
the air-trim it accumulated, whether any guard fired, and how it exited.

Reads only files; asserts nothing.  The PASS/FAIL verdict is run_tests.sh's.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

# The four runs the task asks for, then the two EXTRA lag-latch probes (see
# t8_sil_run.sh: the live inverted env cannot reach that code path at all).
ORDER = ["observe_tip0", "observe_tip-10mm", "closed_tip0", "closed_tip-10mm",
         "latch_tip0", "latch_tip-10mm"]

# executor log lines that matter, by the mechanism they belong to
MARKERS = {
    "lag_latch": re.compile(r"ENCODER LAG LATCH|LAG LATCH overrides"),
    "contact_declared": re.compile(r"CONTACT|contact at |surface|SURFACE|"
                                   r"shift|latch"),
    "air_trim": re.compile(r"AIR-TRIM: under-resisted"),
    "float_guard": re.compile(r"^.*FLOATING:"),
    "overpress_guard": re.compile(r"OVER-PRESS:"),
    "over_ceiling": re.compile(r"F over ceiling"),
    "lost_touch": re.compile(r"LOST TOUCH:"),
    "stale": re.compile(r"robot_state STALE"),
    "complete": re.compile(r"RTff pathway complete"),
}


def _exec_facts(run: Path) -> dict:
    """-> what the executor's own log and force log say about this run."""
    log = run / "exec.log"
    text = log.read_text(errors="replace") if log.exists() else ""
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    hits = {name: [ln for ln in lines if rx.search(ln)]
            for name, rx in MARKERS.items()}
    facts = {
        "exit_code": int((run / "exec_rc.txt").read_text().strip())
        if (run / "exec_rc.txt").exists() else None,
        "final_line": lines[-1] if lines else "",
        "n_log_lines": len(lines),
    }
    for name, found in hits.items():
        facts["n_" + name] = len(found)
        if found:
            facts["first_" + name] = found[0][-160:]

    # the executor's own per-tick record: press it settled at, force it saw
    fcsv = run / "force.csv"
    draw = []
    if fcsv.exists():
        with fcsv.open(newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("phase") == "draw":
                    draw.append(row)
    facts["n_draw_rows"] = len(draw)
    if draw:
        def col(name):
            out = []
            for r in draw:
                try:
                    out.append(float(r[name]))
                except (TypeError, ValueError, KeyError):
                    pass
            return out
        press = col("press_cmd_m")
        fmeas = col("f_meas_N")
        tail = press[max(0, len(press) - 50):]
        facts["press_first_mm"] = round(1e3 * press[0], 2) if press else None
        facts["press_last_mm"] = round(1e3 * press[-1], 2) if press else None
        facts["press_settled_mm"] = (round(1e3 * sum(tail) / len(tail), 2)
                                     if tail else None)
        facts["press_max_mm"] = round(1e3 * max(press), 2) if press else None
        facts["f_meas_mean_N"] = (round(sum(fmeas) / len(fmeas), 3)
                                  if fmeas else None)
        facts["f_meas_absmax_N"] = (round(max(abs(v) for v in fmeas), 3)
                                    if fmeas else None)
        tgt = col("f_target_N")
        facts["f_target_N"] = round(tgt[-1], 3) if tgt else None
    return facts


PRESSED_M = 0.002


def _pressed_window(run: Path) -> dict:
    """The plant's numbers over the ticks the executor was really PRESSING.

    The node's own phase rule (contract-derived: commanded tip at or below the
    paper plane) counts the executor's touchdown DWELL as draw -- on this CSV
    the pen-up rows sit at the plane, so the executor holds the tip exactly at
    z_paper for five seconds before it starts pressing, and those ticks are
    air by construction.  This window is the honest comparison against the
    open-loop walker: ticks whose COMMANDED depth is at least `PRESSED_M`
    (2 mm) below the plane, i.e. the executor asked for a press.
    """
    path = run / "sil" / "trace.npz"
    if not path.exists():
        return {}
    try:
        import numpy as np
    except ImportError:
        return {}
    d = np.load(path, allow_pickle=False)
    mask = np.asarray(d["press_cmd_m"], float) >= PRESSED_M
    n = int(mask.sum())
    if not n:
        return {"n_pressed_ticks": 0}
    tip = np.asarray(d["tip_act_world"], float)[:, 2] - float(d["paper_z"])
    f = np.asarray(d["f_normal"], float)
    return {
        "n_pressed_ticks": n,
        "touch_frac_pressed": float(
            np.asarray(d["in_contact"]).astype(bool)[mask].mean()),
        "f_mean_pressed_n": float(f[mask].mean()),
        "f_max_pressed_n": float(f[mask].max()),
        "tip_height_pressed_mm": float(1e3 * tip[mask].mean()),
        "press_cmd_mean_mm": float(
            1e3 * np.asarray(d["press_cmd_m"], float)[mask].mean()),
        "press_cmd_max_mm": float(1e3 * np.asarray(d["press_cmd_m"],
                                                   float).max()),
    }


def _plant_facts(run: Path) -> dict:
    """-> `trace.metrics` + the node's own bookkeeping for this run."""
    path = run / "sil" / "summary.json"
    if not path.exists():
        return {}
    doc = json.loads(path.read_text())
    m, ros = doc.get("metrics", {}), doc.get("ros", {})
    return {
        "n_draw_ticks": m.get("n_draw_ticks"),
        "touch_frac": m.get("touch_frac"),
        "f_mean_n": m.get("f_mean_n"),
        "f_mean_contact_n": m.get("f_mean_contact_n"),
        "f_max_n": m.get("f_max_n"),
        "tip_height_mm": (None if m.get("mean_actual_tip_height_m") is None
                          else 1e3 * m["mean_actual_tip_height_m"]),
        "penetration_mm": (None if m.get("max_penetration_m") is None
                           else 1e3 * m["max_penetration_m"]),
        "eq_pose_msgs": ros.get("eq_pose_msgs"),
        "joint_ref_msgs": ros.get("joint_ref_msgs"),
        "sim_duration_s": ros.get("sim_duration_s"),
        "rt_factor": ros.get("rt_factor_achieved"),
        "tip_error_mm": 1e3 * doc.get("meta", {}).get("tip_error_m", 0.0),
    }


ROWS = [
    ("tip error", "mm", "plant", "tip_error_mm", 1),
    ("draw ticks (sim)", "", "plant", "n_draw_ticks", 0),
    ("touch fraction (draw)", "", "plant", "touch_frac", 3),
    ("mean force (draw)", "N", "plant", "f_mean_n", 3),
    ("mean force while touching", "N", "plant", "f_mean_contact_n", 3),
    ("peak force (draw)", "N", "plant", "f_max_n", 3),
    ("physical tip height (draw)", "mm", "plant", "tip_height_mm", 3),
    ("max tip depth in paper", "mm", "plant", "penetration_mm", 3),
    ("-- pressed >= 2 mm window --", "", "plant", "n_pressed_ticks", 0),
    ("touch fraction (pressed)", "", "plant", "touch_frac_pressed", 3),
    ("mean force (pressed)", "N", "plant", "f_mean_pressed_n", 3),
    ("peak force (pressed)", "N", "plant", "f_max_pressed_n", 3),
    ("physical tip height (pressed)", "mm", "plant",
     "tip_height_pressed_mm", 3),
    ("commanded press (pressed)", "mm", "plant", "press_cmd_mean_mm", 2),
    ("commanded press, deepest", "mm", "plant", "press_cmd_max_mm", 2),
    ("eq poses received", "", "plant", "eq_pose_msgs", 0),
    ("joint refs received", "", "plant", "joint_ref_msgs", 0),
    ("simulated", "s", "plant", "sim_duration_s", 2),
    ("real-time factor", "", "plant", "rt_factor", 3),
    ("exec draw rows", "", "exec", "n_draw_rows", 0),
    ("press at stroke start", "mm", "exec", "press_first_mm", 2),
    ("press settled (last 50)", "mm", "exec", "press_settled_mm", 2),
    ("press max", "mm", "exec", "press_max_mm", 2),
    ("f_meas mean (exec)", "N", "exec", "f_meas_mean_N", 3),
    ("f_meas |max| (exec)", "N", "exec", "f_meas_absmax_N", 3),
    ("f_target (exec)", "N", "exec", "f_target_N", 3),
    ("lag-latch lines", "", "exec", "n_lag_latch", 0),
    ("air-trim engagements", "", "exec", "n_air_trim", 0),
    ("over-ceiling retreats", "", "exec", "n_over_ceiling", 0),
    ("float-guard trips", "", "exec", "n_float_guard", 0),
    ("over-press trips", "", "exec", "n_overpress_guard", 0),
    ("lost-touch exits", "", "exec", "n_lost_touch", 0),
    ("stale/reflex events", "", "exec", "n_stale", 0),
    ("pathway complete", "", "exec", "n_complete", 0),
    ("exit code", "", "exec", "exit_code", 0),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    runs = {}
    for tag in ORDER:
        d = args.root / tag
        if d.is_dir():
            facts = _plant_facts(d)
            facts.update(_pressed_window(d))
            runs[tag] = {"plant": facts, "exec": _exec_facts(d)}
    if not runs:
        print("T8: no run directories under %s" % args.root)
        return 1

    names = list(runs)
    width = max(len(r[0]) for r in ROWS)
    head = f"{'':<{width}}  {'unit':<4}" + "".join(f"  {n:>16}" for n in names)
    print(head)
    print("-" * len(head))
    for label, unit, side, key, nd in ROWS:
        cells = []
        for n in names:
            v = runs[n][side].get(key)
            cells.append("  " + ("%16s" % "-" if v is None
                                 else f"{float(v):>16.{nd}f}"))
        print(f"{label:<{width}}  {unit:<4}" + "".join(cells))
    print()
    for n in names:
        e = runs[n]["exec"]
        print(f"[{n}] final: {e.get('final_line', '')[-150:]}")
        for k in ("first_air_trim", "first_float_guard",
                  "first_overpress_guard", "first_lost_touch",
                  "first_lag_latch", "first_over_ceiling"):
            if k in e:
                print(f"      {k[6:]:<18} {e[k]}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(runs, indent=2) + "\n")
        print(f"\nwritten to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
