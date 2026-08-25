#!/usr/bin/env python3
"""Where on the paper should the logo go, and how big?

    python3 scripts/csail_place.py [--arms all] [--jobs 6]

A thin CLI over `aris_sixarm.artwork.search_placement`, which is this script's
own two-stage search (atlas proxy to RANK, real allocation to DECIDE) and its
own choice rule (the LARGEST placement whose real coverage is within `--slack`
of the best anyone achieved), lifted into the package so that
`scripts/draw.py` runs the same search on an arbitrary picture rather than a
second copy of it.  Behaviour, flags and both outputs are unchanged; see
`aris_sixarm/artwork.py` for why each half is the way it is.

Writes out/csail_placement_<tag>.json (the chosen placement + both curves) and
out/csail_placement_<tag>.png (coverage vs size and rotation).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                  # noqa: E402

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from aris_sixarm import allocate, artwork, trace  # noqa: E402
from aris_sixarm.fleet import SHEET              # noqa: E402

# m, the margin-limited width at scale 1.0 on the LEGACY 3.607 m sheet.  Kept
# only as the documented default for reproducing the runs that used it — and it
# is exactly why those runs' size sweeps were flat (see `artwork.base_width`).
BASE_WIDTH = 2.4106405991564377


def coverage_png(doc, grid, path, arms, margin, offset, radius, slack):
    """Coverage against SIZE IN SQUARE METRES, one pair of curves per rotation.

    A scale axis would compare unlike things: the same "scale" is a different
    logo in each rotation, which is the whole reason rotations are searched.
    """
    rots = doc["rotations"]
    per_cell = {}
    for key, r in doc["per_scale"].items():
        per_cell[(round(r["rot"], 6), round(r["f"], 6))] = r
    pick = doc["chosen"]
    best = max(r["cov"] for r in doc["real"])
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    cols = {rots[i]: c for i, c in
            zip(range(len(rots)), ["#cb6608", "#1f77b4", "#2ca02c", "#9467bd"])}
    # MATCH ON THE CELL'S OWN `f`, NEVER ON THE ROUNDED KEY.  The key is
    # `round(f, 6)` and the grid rows carry the raw `np.linspace` value, so any
    # `--scales` whose step is not exact to six decimals (0.30 to 0.85 in 8, for
    # one) matches nothing and `max()` is handed an empty iterable.  Every
    # published sweep used a round step and never hit it.
    def proxy_at(cell, rot):
        hits = [g["proxy"] for g in grid
                if abs(g["f"] - cell["f"]) < 1e-9 and abs(g["rot"] - rot) < 1e-9]
        return 100 * max(hits) if hits else float("nan")

    for rot in rots:
        ks = sorted([k for k in per_cell if abs(k[0] - rot) < 1e-9])
        ar = [per_cell[k]["logo_w"] * per_cell[k]["logo_h"] for k in ks]
        ax.plot(ar, [proxy_at(per_cell[k], rot) for k in ks],
                "o--", ms=4, color=cols[rot], alpha=0.45,
                label=f"{rot:.0f}$\\degree$ atlas proxy (upper bound, "
                      f"r={radius} m)")
        ax.plot(ar, [100 * per_cell[k]["cov"] for k in ks], "o-", lw=2.2,
                color=cols[rot], label=f"{rot:.0f}$\\degree$ real allocation "
                                       "(certified + validated)")
    ax.axhline(100 * (best - slack), color="#666665", lw=1.0, ls=":",
               label=f"best - {100 * slack:.0f} pp")
    ax.plot([pick["logo_w"] * pick["logo_h"]], [100 * pick["coverage"]], "*",
            ms=18, color=cols[pick["rotate_deg"]], mec="black", zorder=5,
            label="chosen")
    ax.set_xlabel("logo area on the paper  [m$^2$]")
    ax.set_ylabel("% of traced length drawn")
    ax.set_title("CSAIL logo placement: coverage vs size and rotation, best "
                 "translation per cell\n"
                 f"{len(arms)} arms, canvas {SHEET[0]:.3f} x {SHEET[1]:.3f} m, "
                 f"offsets searched +-{offset:.2f} m", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


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
    rots = [float(x) for x in str(a.rotate).split(",") if x.strip() != ""]

    t0 = time.time()
    doc = artwork.search_placement(
        px, arms, atlas_dir, margin=a.margin, radius=a.radius,
        scales=tuple(a.scales), offset=a.offset, offset_step=a.offset_step,
        rotations=rots, top=a.top, slack=a.slack, jobs=a.jobs,
        base=(None if a.base_width == "auto" else float(a.base_width)))
    print(f"  searched in {time.time() - t0:.1f} s")

    (out / f"csail_placement_{a.tag}.json").write_text(json.dumps(doc))
    coverage_png(doc, doc["proxy"], out / f"csail_placement_{a.tag}.png",
                 arms, a.margin, a.offset, a.radius, a.slack)
    print(f"wrote {out}/csail_placement_{a.tag}.json and .png")
    return doc


if __name__ == "__main__":
    main()
