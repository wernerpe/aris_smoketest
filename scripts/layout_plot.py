#!/usr/bin/env python3
"""out/layout_study.png — coverage maps: current rig vs the top proposals.

Left panel: the CURRENT final6_opt rig swept with the lateral tool
(out/atlas_final6_opt_lat/coverage.npz).  Right panels: the layout study's
finalists (out/layout_study/fine_*/coverage.npz, falling back to the medium
4 cm sweeps).  Colour = how many arms hold a strict-GO solution over the
cell; bases drawn as squares (floor) and circles (inverted).
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
from aris_sixarm import atlas  # noqa: E402
from aris_sixarm.fleet import rig  # noqa: E402
from aris_sixarm.layout import FLOOR_IDS, INV_IDS  # noqa: E402
from aris_sixarm.rig_final6 import SHEET_FINAL6  # noqa: E402

W, H = SHEET_FINAL6
CMAP = ListedColormap(["#f3f0ec", "#c6dbef", "#6baed6", "#2171b5", "#08306b"])
NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 6.5], CMAP.N)


def load_fine(d):
    z = np.load(Path(d) / "coverage.npz")
    return z["count_go"], z["xs"], z["ys"]


def load_medium(d, grid=0.04):
    xs = np.arange(0.0, W + 1e-9, grid)
    ys = np.arange(0.0, H + 1e-9, grid)
    cnt = np.zeros((len(ys), len(xs)), np.int16)
    for f in sorted(Path(d).glob("atlas_arm*.npz")):
        arr = np.load(f)["data"]
        if not len(arr):
            continue
        go = atlas.strict_go(arr)
        ix = np.rint(arr[go, 0] / grid).astype(int)
        iy = np.rint(arr[go, 1] / grid).astype(int)
        k = (ix >= 0) & (ix < len(xs)) & (iy >= 0) & (iy < len(ys))
        cnt[iy[k], ix[k]] += 1
    return cnt, xs, ys


def panel(ax, cnt, xs, ys, title, bases_floor, bases_inv):
    # canvas y on the horizontal axis, x vertical — run_atlas6's convention
    ax.pcolormesh(ys, xs, cnt.T, cmap=CMAP, norm=NORM, shading="nearest")
    ax.add_patch(plt.Rectangle((0, 0), H, W, fill=False, ec="k", lw=1.2))
    for x, y in bases_floor:
        ax.plot(y, x, "s", ms=9, mfc="#d95f02", mec="k", mew=1.2, zorder=5)
    for x, y in bases_inv:
        ax.plot(y, x, "o", ms=9, mfc="#1b9e77", mec="k", mew=1.2, zorder=5)
    un = float(np.mean(cnt >= 1))
    ge2 = float(np.mean(cnt >= 2))
    ax.set_title(f"{title}\nunion {100 * un:.1f} %   >=2 arms "
                 f"{100 * ge2:.1f} %", fontsize=10)
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
                   "CURRENT final6_opt + lateral tool\n(frame boxes active)",
                   [(fl6[13].xy[0], fl6[13].xy[1]),
                    (fl6[17].xy[0], fl6[17].xy[1])],
                   [(fl6[a].xy[0], fl6[a].xy[1]) for a in (31, 71, 2, 97)]))

    for rank, f in enumerate(rec.get("fine", [])[:3]):
        d = Path(f["dir"])
        if not (d / "coverage.npz").exists():
            continue
        cnt, xs, ys = load_fine(d)
        lay = f["layout"]
        panels.append((cnt, xs, ys,
                       f"PROPOSED rank {rank + 1} (green field, "
                       f"h={lay['h']:.3f}"
                       + (", symmetric)" if f.get("sym") else ")"),
                       lay["floor"], lay["inv"]))

    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(11, 3.1 * n))
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, panels):
        panel(ax, *p)
    axes[-1].set_xlabel("canvas y (m)", fontsize=9)
    for ax in axes:
        ax.set_ylabel("canvas x (m)", fontsize=9)
    fig.suptitle("Layout study: strict-GO arm count per cell "
                 "(2 floor + 4 ceiling-inverted, lateral pen; squares = "
                 "floor bases, circles = inverted)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = ROOT / "out" / "layout_study.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out} ({n} panels)")


if __name__ == "__main__":
    main()
