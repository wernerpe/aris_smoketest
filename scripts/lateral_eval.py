#!/usr/bin/env python3
"""Phase-A evaluation of the LATERAL pen holder (tip = TCP + R@(0.110,0,0.110)).

1. E2E: the rim arc around arm 31's base on the final6_opt rig, planned inline
   and lateral — status, min sigma, min margin, joint travel, plan time, phi.
2. Atlas patch: one arm (31, inverted, h = 0.922), same grid and gates, inline
   vs lateral — reachable and strict-GO cell counts, radial extent.

Run:  python3 scripts/lateral_eval.py [--grid 0.04] [--arm 31]
(no ARIS_RIG needed; the script activates final6_opt itself.)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from aris_sixarm import atlas, fleet, frames, planner, stroke_api  # noqa: E402

ROOT = Path(__file__).parents[1]


def rim_arc(spec, sheet, r=0.66, th0=-0.6, th1=1.05):
    """A rim arc around the base, clipped to the sheet.  The full README arc
    (th -0.6..1.05 pi) certifies on the LEGACY rig; on final6_opt its start
    lies in a frame-box dead zone for BOTH tools, so the E2E there uses the
    (-0.3, 0.5) pi sub-arc, which certifies inline and lateral alike."""
    th = np.linspace(th0 * np.pi, th1 * np.pi, 400)
    bx, by = spec.xy
    return planner.clip_to_sheet(
        np.column_stack([bx + r * np.cos(th), by + r * np.sin(th)]),
        verbose=False, sheet=sheet)


def plan_line(tag, r, dt):
    if r["status"] != "ok":
        return (f"  {tag:8s} {r['status']:5s} reason={r.get('reason')} "
                f"s*={r.get('s_star', 0):.3f} t={1000 * dt:.0f} ms")
    return (f"  {tag:8s} ok    sigma>={r['min_sigma']:.4f} "
            f"margin>={r['min_margin']:.3f} travel={r['sum_travel']:.2f} rad "
            f"knots={r['n_knots']} tip={r['tip_err']:.1e} m "
            f"phi={r.get('phi', 0.0):+.3f} t={1000 * dt:.0f} ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", type=int, default=31)
    ap.add_argument("--grid", type=float, default=0.04)
    ap.add_argument("--tilt", type=float, default=0.0,
                    help="atlas tilt cone (0 = perpendicular only)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    fl = fleet.activate("final6_opt")
    sheet = fleet.SHEET
    spec = fl[a.arm]
    print(f"rig final6_opt, arm {a.arm} ({spec.name}, {spec.mount}, "
          f"z={spec.z:.3f}), canvas {sheet[0]:.4f} x {sheet[1]:.5f} m")

    # ---- 1. rim arcs, end to end ------------------------------------------
    out = dict(arm=a.arm, strokes={})

    def e2e(name, arc, sp):
        L = stroke_api.polyline_length(arc)
        print(f"\nE2E {name}: {L:.3f} m")
        out["strokes"][name] = dict(arc_len=L)
        for tag, opts in (("inline", {}),
                          ("lateral", dict(pen_lat=frames.PEN_LAT_HOLDER))):
            t0 = time.perf_counter()
            r = stroke_api.plan_stroke(arc, sp, opts)
            dt = time.perf_counter() - t0
            print(plan_line(tag, r, dt))
            out["strokes"][name][tag] = dict(
                status=r["status"], t_ms=1000 * dt,
                min_sigma=float(r.get("min_sigma", np.nan)),
                min_margin=float(r.get("min_margin", np.nan)),
                travel=float(r.get("sum_travel", np.nan)),
                phi=float(r.get("phi", 0.0)) if np.ndim(r.get("phi", 0.0)) == 0
                else "varying",
                tip_err=float(r.get("tip_err", np.nan)),
                valid=bool(r.get("validation", {}).get("ok", False)))

    # the README rim arc, on the rig it was earned on
    fl_leg, sheet_leg = fleet.rig("sixarm")
    e2e("rim_arc_legacy", rim_arc(fl_leg[31], sheet_leg), fl_leg[31])
    # the final6_opt sub-arc that certifies for both tools
    e2e("rim_arc_final6_opt", rim_arc(spec, sheet, th0=-0.3, th1=0.5), spec)

    # a handful of shorter probes for the plan-time distribution
    print("\nper-stroke plan time, lateral (8 probe strokes):")
    bx, by = spec.xy
    rng = np.random.default_rng(7)
    times, statuses = [], []
    for k in range(8):
        ang = rng.uniform(0, 2 * np.pi)
        r0 = rng.uniform(0.35, 0.62)
        c = np.array([bx + r0 * np.cos(ang), by + r0 * np.sin(ang)])
        d = rng.uniform(0, 2 * np.pi)
        seg = np.array([c - 0.12 * np.array([np.cos(d), np.sin(d)]),
                        c + 0.12 * np.array([np.cos(d), np.sin(d)])])
        seg = planner.clip_to_sheet(seg, verbose=False, sheet=sheet)
        if len(seg) < 2:
            continue
        t0 = time.perf_counter()
        r = stroke_api.plan_stroke(seg, spec,
                                   dict(pen_lat=frames.PEN_LAT_HOLDER))
        times.append(1000 * (time.perf_counter() - t0))
        statuses.append(r["status"])
    print(f"  {len(times)} strokes: median {np.median(times):.0f} ms, "
          f"min {min(times):.0f}, max {max(times):.0f}; statuses: "
          f"{dict((s, statuses.count(s)) for s in set(statuses))}")
    out["probe_times_ms"] = times
    out["probe_statuses"] = statuses

    # ---- 2. the atlas patch ----------------------------------------------
    print(f"\natlas patch, arm {a.arm}, grid {a.grid} m, tilt {a.tilt} deg:")
    res = {}
    for tag, lat in (("inline", 0.0), ("lateral", frames.PEN_LAT_HOLDER)):
        d = ROOT / "out" / f"lateral_eval_{tag}"
        t0 = time.time()
        arr = atlas.sweep_arm(a.arm, d, grid=a.grid, tilt_max_deg=a.tilt,
                              fleet=fl, sheet=sheet, pen_lat=lat)
        go = atlas.strict_go(arr)
        r_cells = ((arr[:, 0] - bx) ** 2 + (arr[:, 1] - by) ** 2) ** 0.5
        res[tag] = dict(
            reach=int(len(arr)), go=int(go.sum()),
            r_max=float(r_cells.max()) if len(arr) else 0.0,
            r_max_go=float(r_cells[go].max()) if go.any() else 0.0,
            sigma_med=float(np.median(arr[go, 3])) if go.any() else 0.0,
            margin_med=float(np.median(arr[go, 2])) if go.any() else 0.0,
            t=time.time() - t0)
        print(f"  {tag:8s} reach={res[tag]['reach']:5d} cells  "
              f"strict-GO={res[tag]['go']:5d}  r_max={res[tag]['r_max']:.2f} "
              f"r_max_GO={res[tag]['r_max_go']:.2f} m  "
              f"median sigma={res[tag]['sigma_med']:.3f} "
              f"margin={res[tag]['margin_med']:.3f}  ({res[tag]['t']:.0f} s)")
    gi, gl = res["inline"]["go"], res["lateral"]["go"]
    print(f"\n  strict-GO delta: {gi} -> {gl} cells "
          f"({100 * (gl - gi) / max(gi, 1):+.1f} %)")
    out["atlas"] = res

    path = Path(a.out) if a.out else ROOT / "out" / "lateral_eval.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
