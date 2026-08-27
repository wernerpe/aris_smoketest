"""WHAT THE CELLS UNDER THE BOOMS ARE ASKING FOR: the lean, cell by cell.

59 canvas cells of the shipped map have no certified drawing pose at all, and
every one of them is directly under a base.  That is not a claim that no arm
can reach them — it is a claim about a CONE.  `atlas.sweep_arm` is swept at
`tilt_max_deg = 15`, the pen lean the run permits, and a cell whose only gated
pose leans 17.5 degrees is recorded as unreachable by a search that was never
allowed to look there.

This asks the same question the atlas asks, with the cone as a variable:

    for each cell, for each arm whose annulus covers it, the SMALLEST lean at
    which `atlas.solve_cell` returns a strict-GO row — same gates, same
    obstacle set, same margin and sigma floors, one parameter changed.

It changes NOTHING.  The shipped cone stays 15 degrees, the shipped atlas stays
what it is, and the map keeps its numbers; what comes back is a histogram and a
per-cell table saying what a wider cone would buy, so the decision that owns it
— how far this pen may lean and still lay a line — can be made on a number.

    ARIS_RIG=proposed ARIS_TOOL=lateral python3 scripts/lean_study.py \
        --raw out/feasible_workspace_v11_raw.npz --out out/lean_study.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("ARIS_RIG", "proposed")
os.environ.setdefault("ARIS_TOOL", "lateral")

from aris_sixarm import atlas, frames, layout          # noqa: E402

GRID = 0.02
SHEET = (1.8034, 3.63064)
# the shipped ladder, and then the rungs above it.  ASCENDING, because
# `solve_cell` stops at the first lean that certifies and the answer is
# therefore the MINIMUM lean the cell needs.
CONE = tuple(atlas.GATE_CONE_DEG) + (17.5, 20.0, 22.5, 25.0, 30.0)


def _cells(raw, arms):
    """The canvas cells no arm has a certified drawing pose for. -> [(x, y)]."""
    xs = np.arange(0.0, SHEET[0] + 1e-9, GRID)
    ys = np.arange(0.0, SHEET[1] + 1e-9, GRID)
    have = np.zeros((len(ys), len(xs)), bool)
    for a in arms:
        d = raw[f"arm{a}"]
        if not len(d):
            continue
        have[np.round(d[:, 1] / GRID).astype(int),
             np.round(d[:, 0] / GRID).astype(int)] = True
    ii, jj = np.nonzero(~have)
    return [(float(xs[j]), float(ys[i])) for i, j in zip(ii, jj)]


def probe(job):
    """(arm, x, y) -> (arm, x, y, min_lean_deg | -1, margin, sigma)."""
    a, x, y = job
    fl = layout.FLEET_PROPOSED
    spec = fl[a]
    h = layout.LAYOUT_PROPOSED["h"]
    boxes = spec.static_obstacles()
    Twb = spec.T_world_base(h)
    r = atlas.solve_cell(x, y, Twb, np.linalg.inv(Twb), spec,
                         atlas._candidates(max(CONE)), spec.pen, boxes,
                         pen_lat=frames.PEN_LAT_HOLDER,
                         gate_groups=atlas._gated_groups(max(CONE), CONE))
    if r is None:
        return (a, x, y, -1.0, float("nan"), float("nan"))
    m, sig = float(r[0]), float(r[1])
    lean = float(r[8])
    if not (m >= atlas.GATE_MARGIN and sig >= atlas.GATE_SIGMA):
        lean = -1.0
    return (a, x, y, lean, m, sig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "out" /
                                         "feasible_workspace_v11_raw.npz"))
    ap.add_argument("--out", default=str(ROOT / "out" / "lean_study.json"))
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 6))
    ap.add_argument("--rmax", type=float, default=1.16,
                    help="only ask arms whose base is within this of the cell")
    a = ap.parse_args()

    fl = layout.FLEET_PROPOSED
    arms = sorted(fl)
    raw = np.load(a.raw)
    cells = _cells(raw, arms)
    print(f"rig=proposed tool=lateral h={layout.LAYOUT_PROPOSED['h']}  "
          f"cone {CONE}")
    print(f"{len(cells)} cells with no certified drawing pose at 15 deg")

    jobs = [(int(x), cx, cy) for cx, cy in cells for x in arms
            if float(np.hypot(*(np.asarray(fl[x].xy, float)
                                - np.array([cx, cy])))) <= a.rmax]
    print(f"{len(jobs)} (arm, cell) probes, {a.workers} workers", flush=True)

    import multiprocessing as mp
    t0 = time.time()
    with mp.get_context("fork").Pool(a.workers) as pool:
        got = pool.map(probe, jobs, chunksize=1)
    print(f"{time.time() - t0:.0f}s", flush=True)

    per = {}
    for arm, x, y, lean, m, sig in got:
        k = (round(x, 4), round(y, 4))
        if lean < 0:
            continue
        if k not in per or lean < per[k][1]:
            per[k] = (int(arm), float(lean), float(m), float(sig))

    hist = {}
    for _k, (_arm, lean, _m, _s) in per.items():
        hist[f"{lean:g}"] = hist.get(f"{lean:g}", 0) + 1
    cum = {}
    for c in CONE:
        cum[f"{c:g}"] = sum(1 for v in per.values() if v[1] <= c + 1e-9)

    out = dict(
        cone=list(CONE), shipped_cone_deg=15.0,
        cells_no_pose_at_15=len(cells),
        cells_with_a_pose_in_the_wider_cone=len(per),
        cells_with_no_pose_at_30=len(cells) - len(per),
        histogram_min_lean=dict(sorted(hist.items(),
                                       key=lambda kv: float(kv[0]))),
        cumulative_by_cone=cum,
        per_cell=[dict(x=k[0], y=k[1], arm=v[0], min_lean_deg=v[1],
                       margin=round(v[2], 4), sigma=round(v[3], 4))
                  for k, v in sorted(per.items())],
        note=("Same gates, same obstacles, same GATE_MARGIN/GATE_SIGMA as the "
              "shipped atlas; the ONLY change is the cone the gated search may "
              "climb.  Nothing here is shipped: the run's cone is still 15 deg "
              "and the map's numbers are measured at it."),
    )
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n{len(per)} of {len(cells)} cells have a certified pose inside "
          f"{max(CONE):g} deg")
    print("  minimum lean, cell count:")
    for k, v in out["histogram_min_lean"].items():
        print(f"    {float(k):5.1f} deg : {v}")
    print("  cells recovered by raising the cone to:")
    for k, v in cum.items():
        if float(k) >= 15.0:
            print(f"    {float(k):5.1f} deg : {v} of {len(cells)}")
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
