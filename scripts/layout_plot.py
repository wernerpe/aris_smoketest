#!/usr/bin/env python3
"""out/layout_study.png — coverage maps for the layout study (v2).

Panels: the CURRENT final6_opt rig swept with the lateral tool, v1's winner
BOTH ways (green field, then re-scored with the mounts active), and the v2
finalists of each family.  Colour = how many arms hold a strict-GO solution
over the cell.  The SCHEMATIC MOUNT HARDWARE is drawn on every v2 panel —
boom footprints as circles, base plates and floor pedestals as rectangles —
because it is the whole point of v2 that the steel is on the map.

A final panel plots coverage against inverted-pair spacing, so the build can
trade steel convenience against coverage by reading it off.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from aris_sixarm import atlas, mounts  # noqa: E402
from aris_sixarm.fleet import rig  # noqa: E402
from aris_sixarm.rig_final6 import SHEET_FINAL6  # noqa: E402

W, H = SHEET_FINAL6
CMAP = ListedColormap(["#f3f0ec", "#c6dbef", "#6baed6", "#2171b5", "#08306b"])
NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 6.5], CMAP.N)
M = mounts.MOUNTS


def load_fine(d):
    z = np.load(Path(d) / "coverage.npz")
    return z["count_go"], z["xs"], z["ys"]


def panel(ax, cnt, xs, ys, title, bases_floor, bases_inv, hardware=True):
    # canvas y on the horizontal axis, x vertical — run_atlas6's convention
    ax.pcolormesh(ys, xs, cnt.T, cmap=CMAP, norm=NORM, shading="nearest")
    ax.add_patch(plt.Rectangle((0, 0), H, W, fill=False, ec="k", lw=1.2))
    for x, y in bases_floor:
        if hardware:                       # pedestal footprint, at the pen's
            ax.add_patch(plt.Rectangle(    # level: the hardware that BITES
                (y - M.ped_xy[1] / 2, x - M.ped_xy[0] / 2),
                M.ped_xy[1], M.ped_xy[0], fill=True, fc="#d95f02",
                alpha=0.30, ec="#d95f02", lw=1.0, zorder=4))
        ax.plot(y, x, "s", ms=8, mfc="#d95f02", mec="k", mew=1.2, zorder=5)
    for x, y in bases_inv:
        if hardware:                       # plate + boom, both above z = h
            ax.add_patch(plt.Rectangle(
                (y - M.plate_xy[1] / 2, x - M.plate_xy[0] / 2),
                M.plate_xy[1], M.plate_xy[0], fill=False, ec="#1b9e77",
                lw=0.9, ls=":", zorder=4))
            ax.add_patch(plt.Circle((y, x), M.boom_r, fill=True,
                                    fc="#1b9e77", alpha=0.25, ec="#1b9e77",
                                    lw=1.0, zorder=4))
        ax.plot(y, x, "o", ms=8, mfc="#1b9e77", mec="k", mew=1.2, zorder=5)
    un = float(np.mean(cnt >= 1))
    ge2 = float(np.mean(cnt >= 2))
    ax.set_title(f"{title}\nunion {100 * un:.2f} %   >=2 arms "
                 f"{100 * ge2:.2f} %", fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlim(-0.45, H + 0.45)
    ax.set_ylim(-0.45, W + 0.45)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_yticks([0, 1, 1.8])
    ax.tick_params(labelsize=8)


def main():
    rec = json.loads((ROOT / "out" / "layout_candidates.json").read_text())
    panels = []

    cur = np.load(ROOT / "out" / "atlas_final6_opt_lat" / "coverage.npz")
    fl6, _ = rig("final6_opt")
    panels.append((cur["count_go"], cur["xs"], cur["ys"],
                   "CURRENT final6_opt + lateral tool (frame boxes active)",
                   [(fl6[13].xy[0], fl6[13].xy[1]),
                    (fl6[17].xy[0], fl6[17].xy[1])],
                   [(fl6[a].xy[0], fl6[a].xy[1]) for a in (31, 71, 2, 97)],
                   False))

    v1 = rec.get("v1_rescore")
    if v1:
        lay = v1["layout"]
        for tag, label in (("green_field", "v1 winner, NO mounts modelled "
                            "(the v1 number)"),
                           ("with_mounts", "v1 winner, MOUNTS ACTIVE "
                            "(re-scored)")):
            d = Path(v1[tag]["dir"])
            if (d / "coverage.npz").exists():
                cnt, xs, ys = load_fine(d)
                panels.append((cnt, xs, ys,
                               f"{label}  h={lay['h']:.3f}",
                               lay["floor"], lay["inv"],
                               tag == "with_mounts"))

    # the RECOMMENDED layout: the all-ceiling optimum in round numbers
    grid = sorted(rec.get("grid", []),
                  key=lambda g: -(g["union"] + 0.08 * g["ge2"]))
    if grid and (Path(grid[0]["dir"]) / "coverage.npz").exists():
        g = grid[0]
        cnt, xs, ys = load_fine(Path(g["dir"]))
        panels.append((cnt, xs, ys,
                       "*** PROPOSED: 0+6 regular 2x3 ceiling grid, mounts "
                       f"active, h={g['h']:.3f} ***",
                       g["layout"]["floor"], g["layout"]["inv"], True))

    # and the best layout the search found for the OTHER family
    for f in rec.get("fine", []):
        if f.get("n_floor", 2) != 2:
            continue
        d = Path(f["dir"])
        if not (d / "coverage.npz").exists():
            continue
        cnt, xs, ys = load_fine(d)
        lay = f["layout"]
        panels.append((cnt, xs, ys,
                       "v2 2+4 BEST (mounts active, "
                       f"h={lay['h']:.3f}"
                       + (", symmetric)" if f.get("sym") else ")"),
                       lay["floor"], lay["inv"], True))
        break

    sp = rec.get("spacing")
    n = len(panels) + (1 if sp else 0)
    fig = plt.figure(figsize=(11, 3.1 * n))
    gs = fig.add_gridspec(n, 1)
    for i, p in enumerate(panels):
        panel(fig.add_subplot(gs[i]), *p)
        if i == len(panels) - 1:
            fig.axes[i].set_xlabel("canvas y (m)", fontsize=9)
        fig.axes[i].set_ylabel("canvas x (m)", fontsize=9)

    if sp:
        ax = fig.add_subplot(gs[len(panels)])
        rows = sorted(sp["rows"], key=lambda r: r["spacing"])
        x = [r["spacing"] for r in rows]
        # two scales: union lives in the last 8 %, overlap in the 50s — one
        # axis would flatten the very cliff this panel exists to show
        ax.plot(x, [100 * r["union"] for r in rows], "o-", color="#2171b5",
                label="union strict-GO (left)")
        ax.set_ylabel("union (% of canvas)", color="#2171b5", fontsize=9)
        ax.tick_params(axis="y", labelcolor="#2171b5", labelsize=8)
        ax2 = ax.twinx()
        ax2.plot(x, [100 * r["ge2"] for r in rows], "s-", color="#d95f02",
                 label=">= 2 arms (right)")
        ax2.set_ylabel(">= 2 arms (%)", color="#d95f02", fontsize=9)
        ax2.tick_params(axis="y", labelcolor="#d95f02", labelsize=8)
        bad = [r["spacing"] for r in rows if r["violations"]]
        if bad:
            ax.axvspan(min(x) - 0.02, max(bad) + 0.005, color="#cccccc",
                       alpha=0.6, zorder=0)
            ax.text(min(x), 93.2, " bases < 0.50 m:\n forbidden",
                    fontsize=7, va="bottom")
        ax.axvspan(0.50, 0.65, color="#b7e4c7", alpha=0.45, zorder=0)
        ax.text(0.575, 93.2, "coverage-neutral\nwindow", fontsize=7,
                ha="center", va="bottom")
        ax.axvline(sp["current"], color="k", ls="--", lw=1)
        ax.set_xlabel("inverted-pair spacing (m)", fontsize=9)
        ax.set_title("Sensitivity: coverage vs pair spacing (2 cm certified, "
                     f"mounts active; proposed {sp['current']:.2f} m)",
                     fontsize=10)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center left")

    fig.suptitle(
        "Layout study v2 — strict-GO arm count per cell, MOUNT HARDWARE "
        "MODELLED\n"
        "squares = floor bases (pedestal footprint shaded), circles = "
        "inverted bases (boom footprint shaded, plate outline dotted)",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out = ROOT / "out" / "layout_study.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out} ({n} panels)")


if __name__ == "__main__":
    main()
