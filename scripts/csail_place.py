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

# m, the margin-limited width at scale 1.0 on the LEGACY 3.607 m sheet.  It is
# kept only as the documented default for reproducing the runs that used it —
# and it is exactly why those runs' size sweeps were flat: on the 1.8034 m final
# canvas 0.7 x 2.4106 = 1.687 m is still wider than the 1.6834 m the margin
# allows, so every "scale" collapsed onto the same logo (see
# out/csail_placement_final.json: seven scales, one width).  `--base-width auto`
# (the default now) measures the width the ACTIVE canvas actually permits FOR
# THIS ROTATION, so scale 1.0 means "as big as it goes" and 0.7 means 70 % of
# that, on any paper.
BASE_WIDTH = 2.4106405991564377


def base_width(px, margin, rotate_deg=0.0):
    """The widest this logo goes on the active sheet at this rotation. -> m."""
    return trace.to_sheet(px, SHEET, margin=margin,
                          rotate_deg=rotate_deg)[1]["logo_w"]


def _sheet(px, f, dx, dy, margin, rot=0.0, base=BASE_WIDTH):
    return trace.to_sheet(px, SHEET, margin=margin, target_width=f * base,
                          offset=(dx, dy), rotate_deg=rot)


def proxy_grid(px, grids, scales, offs, margin, radius, rots, bases):
    """-> list of dicts, one per (rotation, scale, dx, dy) that fits."""
    out = []
    for rot in rots:
        for f in scales:
            for dx in offs:
                for dy in offs:
                    st, info = _sheet(px, f, dx, dy, margin, rot, bases[rot])
                    if not info["fits"]:
                        continue
                    out.append(dict(rot=float(rot), f=float(f), dx=float(dx),
                                    dy=float(dy),
                                    proxy=allocate.reach_fraction(st, grids, radius),
                                    logo_w=info["logo_w"], logo_h=info["logo_h"]))
    return out


def _real_one(args):
    px, rot, f, dx, dy, margin, arms, atlas_dir, base = args
    st, info = _sheet(px, f, dx, dy, margin, rot, base)
    res = allocate.allocate(st, arms=arms, atlas_dir=atlas_dir, verbose=False)
    return dict(rot=float(rot), f=float(f), dx=float(dx), dy=float(dy),
                drawn=res["drawn_len"], traced=res["total_len"],
                cov=res["drawn_len"] / max(res["total_len"], 1e-9),
                logo_w=info["logo_w"], logo_h=info["logo_h"],
                target_width=float(f * base),
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
    ap.add_argument("--rotate", default="0",
                    help="comma-separated logo rotations in degrees to search "
                         "as PLACEMENT VARIANTS, e.g. '0,90'.  The aspect ratio "
                         "is never touched; only which way round the logo sits "
                         "on the paper.  Worth searching whenever the canvas "
                         "and the logo disagree about which axis is long")
    ap.add_argument("--base-width", default="auto",
                    help="metres at scale 1.0, or 'auto' (default) = the widest "
                         "the ACTIVE canvas allows at each rotation")
    ap.add_argument("--atlas", default=None,
                    help="atlas directory for the prefilter and the proxy "
                         "(default: --out)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    atlas_dir = a.atlas or str(out)

    arms = allocate.active_arms(a.arms if a.arms != "all" else "all")
    px, _ = trace.trace_logo(a.image)
    grids = allocate.atlas_cells(arms, atlas_dir)
    if grids is None:
        raise SystemExit(f"no atlas for arms {arms} under {atlas_dir}")
    rots = [float(x) for x in str(a.rotate).split(",") if x.strip() != ""]
    bases = {r: (base_width(px, a.margin, r) if a.base_width == "auto"
                 else float(a.base_width)) for r in rots}
    scales = np.linspace(a.scales[0], a.scales[1], int(a.scales[2]))
    n_off = int(round(a.offset / a.offset_step))
    offs = np.arange(-n_off, n_off + 1) * a.offset_step
    print(f"canvas {SHEET[0]:.4f} x {SHEET[1]:.5f} m, margin {a.margin} m, "
          f"{len(arms)} arms {arms}")
    for r in rots:
        _, i0 = _sheet(px, 1.0, 0.0, 0.0, a.margin, r, bases[r])
        print(f"  rotation {r:>5.1f} deg: scale 1.0 = {i0['logo_w']:.4f} x "
              f"{i0['logo_h']:.4f} m  (base width {bases[r]:.4f} m)")

    t0 = time.time()
    grid = proxy_grid(px, grids, scales, offs, a.margin, a.radius, rots, bases)
    print(f"proxy: {len(grid)} placements fit on the sheet "
          f"({len(rots)} rotations x {len(scales)} scales x {len(offs)}^2 "
          f"offsets) in {time.time() - t0:.1f} s")

    jobs = []
    for rot in rots:
        for f in scales:
            cand = sorted([g for g in grid if abs(g["f"] - f) < 1e-9
                           and abs(g["rot"] - rot) < 1e-9],
                          key=lambda g: -g["proxy"])[:a.top]
            for g in cand:
                jobs.append((px, rot, g["f"], g["dx"], g["dy"], a.margin, arms,
                             atlas_dir, bases[rot]))
    print(f"real: allocating {len(jobs)} placements ({a.top} per rotation x "
          "scale)...")

    t1 = time.time()
    if a.jobs > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(a.jobs) as pool:
            real = pool.map(_real_one, jobs)
    else:
        real = [_real_one(j) for j in jobs]
    print(f"  {len(real)} allocations in {time.time() - t1:.1f} s")

    # THE SIZE CHOICE IS MADE OVER (rotation, scale) TOGETHER.  A rotation is
    # not a tie-break applied after a size is picked: turning the logo changes
    # what "largest" MEANS (the same scale 1.0 is a 1.68 x 1.29 m logo upright
    # and a 1.68 x 2.20 m one on its side), so the two knobs are one grid and
    # the rule reads "the largest logo whose real coverage is within `slack` of
    # the best anyone achieved", with SIZE MEASURED IN METRES OF WIDTH, not in
    # units of a per-rotation scale that means different things.
    per_cell = {}
    for r in real:
        k = (round(r["rot"], 6), round(r["f"], 6))
        if k not in per_cell or r["cov"] > per_cell[k]["cov"]:
            per_cell[k] = r
    best = max(r["cov"] for r in real)
    ok = [r for r in per_cell.values() if r["cov"] >= best - a.slack]
    pick = max(ok, key=lambda r: (round(r["logo_w"] * r["logo_h"], 9), r["cov"]))

    print(f"\n{'rot':>5} {'scale':>6} {'logo w x h':>17} {'best offset':>16} "
          f"{'proxy':>7} {'REAL cov':>9}")
    for k in sorted(per_cell):
        r = per_cell[k]
        p = max(g["proxy"] for g in grid
                if abs(g["f"] - r["f"]) < 1e-9 and abs(g["rot"] - r["rot"]) < 1e-9)
        print(f"{r['rot']:>5.0f} {r['f']:>6.3f} {r['logo_w']:>7.3f} x "
              f"{r['logo_h']:<7.3f} ({r['dx']:>+5.2f},{r['dy']:>+5.2f}) m "
              f"{p:>7.3f} {100 * r['cov']:>8.1f} %"
              + ("   <- chosen" if r is pick else ""))
    print(f"\nbest coverage anywhere {100 * best:.1f} %; largest logo within "
          f"{100 * a.slack:.0f} pp of it = {pick['logo_w']:.3f} x "
          f"{pick['logo_h']:.3f} m ({pick['logo_w'] * pick['logo_h']:.3f} m², "
          f"rotation {pick['rot']:.0f} deg, scale {pick['f']:.3f}) at offset "
          f"({pick['dx']:+.2f}, {pick['dy']:+.2f}) m, {100 * pick['cov']:.1f} % drawn")

    doc = dict(chosen=dict(scale=pick["f"], target_width=pick["target_width"],
                           rotate_deg=pick["rot"],
                           offset=[pick["dx"], pick["dy"]], margin=a.margin,
                           logo_w=pick["logo_w"], logo_h=pick["logo_h"],
                           coverage=pick["cov"]),
               arms=[int(x) for x in arms], base_width=bases,
               sheet=[float(SHEET[0]), float(SHEET[1])],
               rotations=rots, atlas=atlas_dir,
               rule=f"largest logo AREA within {a.slack} of the best real "
                    "coverage, over rotations x scales",
               proxy_radius=a.radius, proxy=grid, real=real,
               per_scale={f"{k[0]:.0f}deg@{k[1]:.3f}": per_cell[k]
                          for k in sorted(per_cell)})
    (out / f"csail_placement_{a.tag}.json").write_text(json.dumps(doc))

    # coverage against SIZE IN SQUARE METRES, one pair of curves per rotation:
    # the whole point of searching rotations is that the same "scale" is a
    # different logo in each, so a scale axis would compare unlike things.
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    cols = {rots[i]: c for i, c in
            zip(range(len(rots)), ["#cb6608", "#1f77b4", "#2ca02c", "#9467bd"])}
    for rot in rots:
        ks = sorted([k for k in per_cell if abs(k[0] - rot) < 1e-9])
        ar = [per_cell[k]["logo_w"] * per_cell[k]["logo_h"] for k in ks]
        ax.plot(ar, [100 * max(g["proxy"] for g in grid
                               if abs(g["f"] - k[1]) < 1e-9
                               and abs(g["rot"] - rot) < 1e-9) for k in ks],
                "o--", ms=4, color=cols[rot], alpha=0.45,
                label=f"{rot:.0f}$\\degree$ atlas proxy (upper bound, "
                      f"r={a.radius} m)")
        ax.plot(ar, [100 * per_cell[k]["cov"] for k in ks], "o-", lw=2.2,
                color=cols[rot], label=f"{rot:.0f}$\\degree$ real allocation "
                                       "(certified + validated)")
    ax.axhline(100 * (best - a.slack), color="#666665", lw=1.0, ls=":",
               label=f"best - {100 * a.slack:.0f} pp")
    ax.plot([pick["logo_w"] * pick["logo_h"]], [100 * pick["cov"]], "*", ms=18,
            color=cols[pick["rot"]], mec="black", zorder=5, label="chosen")
    ax.set_xlabel("logo area on the paper  [m$^2$]")
    ax.set_ylabel("% of traced length drawn")
    ax.set_title("CSAIL logo placement: coverage vs size and rotation, best "
                 "translation per cell\n"
                 f"{len(arms)} arms, canvas {SHEET[0]:.3f} x {SHEET[1]:.3f} m, "
                 f"offsets searched +-{a.offset:.2f} m", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / f"csail_placement_{a.tag}.png", dpi=130)
    print(f"wrote {out}/csail_placement_{a.tag}.json and .png")
    return doc


if __name__ == "__main__":
    main()
