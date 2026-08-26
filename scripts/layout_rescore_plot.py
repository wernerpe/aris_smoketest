#!/usr/bin/env python3
"""Plan views of the re-scored layout family — out/layout_rescore.png.

Reads what `scripts/layout_rescore.py` wrote (`out/layout_rescore.json` for the
scorecard, `out/layout_rescore.npz` for the per-scenario coverage-count grids)
and draws, for each candidate asked for, the OCCUPANCY-CORRECTED coverage map:
how many arms can certifiably draw each 2 cm cell with the other five reduced
to their pose-invariant base cylinders.  Dead cells are marked, the six bases
and the 0.35 m handoff annulus around each are drawn on top, and the empty-air
map is drawn beside it so the two objectives can be read against each other.

    ARIS_TOOL=lateral python3 scripts/layout_rescore_plot.py \
        --only adopted,stag061_dy605,pitch080
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402
from matplotlib.colors import ListedColormap, BoundaryNorm  # noqa: E402
from matplotlib.patches import Circle                      # noqa: E402

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from aris_sixarm.rig_final6 import SHEET_FINAL6            # noqa: E402

W, H = SHEET_FINAL6
NEAR_BASE = 0.35
CNT_COLORS = ["#7f1d1d", "#fde0dd", "#fbb4b9", "#9ecae1", "#4292c6",
              "#08519c", "#08306b"]
CMAP = ListedColormap(CNT_COLORS)
NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5], CMAP.N)


def panel(ax, cnt, lay, title, sub, annuli=True):
    ax.imshow(cnt, origin="lower", extent=[0, W, 0, H], cmap=CMAP, norm=NORM,
              interpolation="nearest", aspect="equal")
    xs = np.linspace(0, W, cnt.shape[1])
    ys = np.linspace(0, H, cnt.shape[0])
    dy, dx = np.nonzero(cnt == 0)
    if len(dy):
        ax.plot(xs[dx], ys[dy], ".", ms=2.2, color="#ffef00", mew=0)
    for (bx, by) in lay["inv"]:
        if annuli:
            ax.add_patch(Circle((bx, by), NEAR_BASE, fill=False, lw=0.8,
                                ls="--", ec="#111"))
        ax.add_patch(Circle((bx, by), 0.06, fc="#111", ec="white", lw=0.8,
                            zorder=5))
    ax.set_xlim(-0.05, W + 0.05)
    ax.set_ylim(-0.05, H + 0.05)
    ax.set_title(title, fontsize=9.5, loc="left")
    ax.set_xlabel(sub, fontsize=7.6)
    ax.tick_params(labelsize=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--json", default=str(ROOT / "out/layout_rescore.json"))
    ap.add_argument("--npz", default=str(ROOT / "out/layout_rescore.npz"))
    ap.add_argument("--png", default=str(ROOT / "out/layout_rescore.png"))
    a = ap.parse_args()
    rec = json.loads(Path(a.json).read_text())
    Z = np.load(a.npz)
    names = [n.strip() for n in a.only.split(",") if n.strip()]
    if not names:
        names = sorted(rec["rows"],
                       key=lambda n: -rec["rows"][n]["union_mounts"])[:4]

    n = len(names)
    fig, axes = plt.subplots(3, n, figsize=(3.35 * n + 1.6, 15.6),
                             squeeze=False)
    fig.subplots_adjust(left=0.075, right=0.90, top=0.905, bottom=0.045,
                        hspace=0.30, wspace=0.14)
    rows_lab = ["1.  EMPTY AIR\nthe objective the grid was chosen on",
                "2.  OCCUPANCY-CORRECTED\nneighbours as their base cylinders,"
                " 80 mm",
                "3.  THE PACKAGE'S OWN BOX GATE\nr 0.12 AABB, 50 mm "
                "(commit 14b01cd)"]
    for k, nm in enumerate(names):
        row = rec["rows"][nm]
        lay = row["layout"]
        pr = rec.get("pairs", {}).get(nm)
        for r, (scen, lab) in enumerate((("air", "dead_air"),
                                         ("mounts", "dead_mounts"),
                                         ("box", "dead_box"))):
            d = row[lab]
            head = (f"{nm}   pitch {row['spacing']:.3f},  h {lay['h']:.3f} m\n"
                    if r == 0 else "")
            sub = (f"{d['dead']} dead cells"
                   + (f", {d['dead_near_base']} within {NEAR_BASE} m of a base"
                      if r else "")
                   + (f"\n>=2 arms {row['ge2_' + scen]:.1f} %"
                      f",  handoff annuli {d['near_covered_pct']:.1f} % "
                      "covered"
                      if r else ""))
            if r == 1 and pr is not None:
                sub += (f"\nworst pair {pr['worst_pair_pct']:.0f} % clear, "
                        f"{pr['n_below_90']}/15 below 90 %")
            panel(axes[r][k], np.asarray(Z[f"{nm}__{scen}"], int), lay,
                  head + f"union {row['union_' + scen]:.2f} %", sub,
                  annuli=bool(r))
            if k:
                axes[r][k].set_yticklabels([])
    for r, lab in enumerate(rows_lab):
        axes[r][0].set_ylabel(lab, fontsize=9.2, labelpad=10)

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=NORM)
    cb = fig.colorbar(sm, ax=axes, fraction=0.018, pad=0.015, ticks=range(7))
    cb.set_label("arms that can certifiably draw the cell", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    fig.suptitle("Six inverted arms over the 1.80 x 3.63 m canvas, re-scored "
                 "with the fleet IN THE ROOM\n"
                 "yellow dots are dead cells; dashed circles are the 0.35 m "
                 "handoff annulus round each base", fontsize=12.5,
                 x=0.012, ha="left", y=0.985)
    fig.savefig(a.png, dpi=118)
    print(f"wrote {a.png}")


if __name__ == "__main__":
    main()
