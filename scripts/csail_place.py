#!/usr/bin/env python3
"""Where on the paper should the logo go, and how big?

    python3 scripts/csail_place.py [--arms all] [--jobs 6]

Two stages, because the two questions cost three orders of magnitude apart:

  PROXY   `allocate.reach_fraction` scores a placement from the atlas alone —
          the length-weighted fraction of the traced logo that lands within
          `--radius` of a cell some arm's pen can stand on.  10 ms per
          candidate, so the whole scale x translation grid is affordable.  It
          is an UPPER BOUND on coverage (a reachable cell is not a plannable
          stroke), used only to RANK.
  REAL    the top `--top` translations of every scale are then allocated for
          real (probe -> colour partition -> cover -> clean re-plan), which is
          the only number that means anything, and the one the choice is made
          on.

CHOICE RULE (the user's): take the LARGEST size whose real coverage is within
`--slack` of the best coverage anyone achieved, then that size's best
translation.  Shrinking is allowed, but only bought when it pays.

Writes out/csail_placement_<tag>.json (the chosen placement + both curves) and
out/csail_placement_<tag>.png (coverage vs scale).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                  # noqa: E402

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from aris_sixarm import allocate, trace          # noqa: E402
from aris_sixarm.fleet import SHEET              # noqa: E402

BASE_WIDTH = 2.4106405991564377      # m, the margin-limited width (scale 1.0)


def _sheet(px, f, dx, dy, margin):
    return trace.to_sheet(px, SHEET, margin=margin, target_width=f * BASE_WIDTH,
                          offset=(dx, dy))


def proxy_grid(px, grids, scales, offs, margin, radius):
    """-> list of dicts, one per (scale, dx, dy) that fits on the paper."""
    out = []
    for f in scales:
        for dx in offs:
            for dy in offs:
                st, info = _sheet(px, f, dx, dy, margin)
                if not info["fits"]:
                    continue
                out.append(dict(f=float(f), dx=float(dx), dy=float(dy),
                                proxy=allocate.reach_fraction(st, grids, radius),
                                logo_w=info["logo_w"], logo_h=info["logo_h"]))
    return out


def _real_one(args):
    px, f, dx, dy, margin, arms, atlas_dir = args
    st, info = _sheet(px, f, dx, dy, margin)
    res = allocate.allocate(st, arms=arms, atlas_dir=atlas_dir, verbose=False)
    return dict(f=float(f), dx=float(dx), dy=float(dy),
                drawn=res["drawn_len"], traced=res["total_len"],
                cov=res["drawn_len"] / max(res["total_len"], 1e-9),
                logo_w=info["logo_w"], logo_h=info["logo_h"],
                loads={int(a): float(sum(s["length"] for s in res["programs"][a]))
                       for a in res["arms"]})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=str(ROOT / "assets/csail/csail_old_med.gif"))
    ap.add_argument("--out", default=str(ROOT / "out"))
    ap.add_argument("--tag", default="6arm")
    ap.add_argument("--arms", default="all")
    ap.add_argument("--margin", type=float, default=0.06)
    ap.add_argument("--radius", type=float, default=0.03, help="proxy reach radius, m")
    ap.add_argument("--scales", type=float, nargs=3, default=(0.7, 1.0, 7),
                    metavar=("LO", "HI", "N"))
    ap.add_argument("--offset", type=float, default=0.30, help="+-metres searched")
    ap.add_argument("--offset-step", type=float, default=0.10)
    ap.add_argument("--top", type=int, default=3, help="translations re-run for real per scale")
    ap.add_argument("--slack", type=float, default=0.01,
                    help="coverage a bigger logo may give up (fraction)")
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args(argv)
    out = Path(a.out)

    arms = allocate.active_arms(a.arms if a.arms != "all" else "all")
    px, _ = trace.trace_logo(a.image)
    grids = allocate.atlas_cells(arms, str(out))
    scales = np.linspace(a.scales[0], a.scales[1], int(a.scales[2]))
    n_off = int(round(a.offset / a.offset_step))
    offs = np.arange(-n_off, n_off + 1) * a.offset_step

    t0 = time.time()
    grid = proxy_grid(px, grids, scales, offs, a.margin, a.radius)
    print(f"proxy: {len(grid)} placements fit on the sheet "
          f"({len(scales)} scales x {len(offs)}^2 offsets) in {time.time() - t0:.1f} s")

    jobs = []
    for f in scales:
        cand = sorted([g for g in grid if abs(g["f"] - f) < 1e-9],
                      key=lambda g: -g["proxy"])[:a.top]
        for g in cand:
            jobs.append((px, g["f"], g["dx"], g["dy"], a.margin, arms, str(out)))
    print(f"real: allocating {len(jobs)} placements ({a.top} per scale)...")

    t1 = time.time()
    if a.jobs > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(a.jobs) as pool:
            real = pool.map(_real_one, jobs)
    else:
        real = [_real_one(j) for j in jobs]
    print(f"  {len(real)} allocations in {time.time() - t1:.1f} s")

    per_scale = {}
    for r in real:
        k = round(r["f"], 6)
        if k not in per_scale or r["cov"] > per_scale[k]["cov"]:
            per_scale[k] = r
    best = max(r["cov"] for r in real)
    ok = [f for f, r in per_scale.items() if r["cov"] >= best - a.slack]
    pick = per_scale[max(ok)]

    print(f"\n{'scale':>6} {'width':>7} {'best offset':>16} {'proxy':>7} {'REAL cov':>9}")
    for f in sorted(per_scale):
        r = per_scale[f]
        p = max(g["proxy"] for g in grid if abs(g["f"] - f) < 1e-9)
        print(f"{f:>6.3f} {r['logo_w']:>6.2f}m ({r['dx']:>+5.2f},{r['dy']:>+5.2f}) m "
              f"{p:>7.3f} {100 * r['cov']:>8.1f} %"
              + ("   <- chosen" if abs(f - pick["f"]) < 1e-9 else ""))
    print(f"\nbest coverage anywhere {100 * best:.1f} %; largest scale within "
          f"{100 * a.slack:.0f} pp of it = {pick['f']:.3f} "
          f"({pick['logo_w']:.3f} x {pick['logo_h']:.3f} m at offset "
          f"({pick['dx']:+.2f}, {pick['dy']:+.2f}) m, {100 * pick['cov']:.1f} % drawn)")

    doc = dict(chosen=dict(scale=pick["f"], target_width=pick["f"] * BASE_WIDTH,
                           offset=[pick["dx"], pick["dy"]], margin=a.margin,
                           logo_w=pick["logo_w"], logo_h=pick["logo_h"],
                           coverage=pick["cov"]),
               arms=[int(x) for x in arms], base_width=BASE_WIDTH,
               rule=f"largest scale within {a.slack} of the best real coverage",
               proxy_radius=a.radius, proxy=grid, real=real,
               per_scale={str(f): per_scale[f] for f in sorted(per_scale)})
    (out / f"csail_placement_{a.tag}.json").write_text(json.dumps(doc))

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    fs = sorted(per_scale)
    ax.plot([f for f in fs], [100 * max(g["proxy"] for g in grid
                                        if abs(g["f"] - f) < 1e-9) for f in fs],
            "o--", color="#9aa0a6", label=f"atlas proxy (upper bound, r={a.radius} m)")
    ax.plot(fs, [100 * per_scale[f]["cov"] for f in fs], "o-", color="#cb6608",
            lw=2.2, label="real allocation (certified + validated)")
    ax.axhline(100 * (best - a.slack), color="#666665", lw=1.0, ls=":",
               label=f"best - {100 * a.slack:.0f} pp")
    ax.plot([pick["f"]], [100 * pick["cov"]], "*", ms=18, color="#cb6608",
            mec="black", zorder=5, label="chosen")
    ax.set_xlabel("logo scale (x %.3f m width)" % BASE_WIDTH)
    ax.set_ylabel("% of traced length drawn")
    ax.set_title("CSAIL logo placement: coverage vs size, best translation per size\n"
                 f"{len(arms)} arms, offsets searched +-{a.offset:.2f} m", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / f"csail_placement_{a.tag}.png", dpi=130)
    print(f"wrote {out}/csail_placement_{a.tag}.json and .png")
    return doc


if __name__ == "__main__":
    main()
