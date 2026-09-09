#!/usr/bin/env python3
"""THE BIGGEST PIECE OF PAPER WITH NO HOLES IN IT.

    ARIS_RIG=proposed ARIS_TOOL=lateral python3 scripts/certified_area.py \
        --atlas out/atlas_proposed_h0940_lat0860 --h 0.940 \
        --json out/certified_area_h0940.json --png out/certified_area_h0940.png

    ...or on the full three-layer map, which is the honest one:
    scripts/certified_area.py --map out/feasible_workspace_v14_map.npz --h 0.940

WHY THIS IS A DIFFERENT QUESTION FROM COVERAGE, AND A BETTER ONE FOR A DRAWING.
`feasible_workspace` answers "what fraction of the canvas can be drawn", and a
rig can score 98 % with a dead cell in the middle of every place anyone would
put a picture.  Pete's ask is the other question: *"we just don't want any
holes if we place a drawing in the certified area"*.  So the figure of merit
here is the largest region a drawing can be placed ANYWHERE inside with zero
dead cells — and by that measure a hole under an arm is not 0.02 % of the
canvas, it is a knife through every rectangle that contains it.

WHAT IS REPORTED, AND WHY EACH ONE.

  largest_rect          the biggest axis-aligned all-live rectangle.  This is
                        the number: a box that fits inside it is certified
                        wherever it sits.  `feasible_workspace.all_rects`
                        enumerates maximal all-True rectangles, so "no hole
                        inside" is true by construction and not by a check.
  by aspect class       landscape (w/h >= 1.5), near-square, portrait — Pete
                        places drawings of different shapes and the biggest
                        box is only useful to the shape it suits.
  centred               the biggest all-live rectangle that CONTAINS the
                        paper's centre, because "put it in the middle" is the
                        default and it is not the same rectangle.
  largest_component     the biggest connected live region and the area of the
                        holes enclosed INSIDE it.  A component with holes is
                        why the rectangle is smaller than the component.
  dead_under_base       per arm, r < UNDER_BASE_R of that arm's own base.
  dead_rim              r >= RIM_R from every base — the outer annulus edge,
                        which no park set or router can buy back.

The rectangle is emitted in PAPER COORDINATES (metres, the canvas frame) with
the atlas, tool and gates it came from, so a placement can be checked against
it later without re-deriving anything — see `scripts/placement_proxy.py
--certified-area`.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("ARIS_RIG", "proposed")
os.environ.setdefault("ARIS_TOOL", "lateral")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import feasible_workspace as fw                                   # noqa: E402
from aris_sixarm import atlas as atlas_mod                        # noqa: E402
from aris_sixarm import frames, layout                            # noqa: E402

GRID = fw.GRID
SHEET = fw.SHEET
UNDER_BASE_R = 0.30      # the base disc, as the height study has always drawn it
RIM_R = 0.60             # beyond this a dead cell is the annulus's outer edge


def live_from_atlas(atlas_dir, fleet):
    """Union strict-GO — the DRAWING-POSE layer only. -> (H, W) bool."""
    xs = np.arange(0.0, SHEET[0] + 1e-9, GRID)
    ys = np.arange(0.0, SHEET[1] + 1e-9, GRID)
    u = np.zeros((len(ys), len(xs)), bool)
    for a in sorted(fleet):
        arr, _ = atlas_mod.load(Path(atlas_dir), a)
        if len(arr):
            ii = np.rint(arr[:, 1] / GRID).astype(int)
            jj = np.rint(arr[:, 0] / GRID).astype(int)
            u[ii, jj] |= atlas_mod.strict_go(arr)
    return u, xs, ys


def live_from_map(map_npz):
    """All three layers — `feasible_workspace`'s own FEASIBLE. -> (H, W) bool."""
    z = np.load(map_npz)
    return z["cause"] == fw.FEASIBLE, z["xs"], z["ys"]


def rect_m(r, xs, ys):
    i0, j0, hh, ww = r
    return dict(cells=int(hh * ww),
                x0=round(float(xs[j0]), 4), y0=round(float(ys[i0]), 4),
                x1=round(float(xs[j0] + (ww - 1) * GRID), 4),
                y1=round(float(ys[i0] + (hh - 1) * GRID), 4),
                w=round(float(ww * GRID), 4), h=round(float(hh * GRID), 4),
                area_m2=round(float(ww * hh * GRID * GRID), 4),
                aspect=round(float(ww / hh), 3))


def best_rects(live, xs, ys):
    """Largest all-live rectangle overall, per aspect class, and centred."""
    R = fw.all_rects(live)
    if not R:
        return {}
    area = lambda r: r[2] * r[3]                              # noqa: E731

    def pick(pred):
        c = [r for r in R if pred(r)]
        return rect_m(max(c, key=area), xs, ys) if c else None

    cx, cy = SHEET[0] / 2.0, SHEET[1] / 2.0
    ci, cj = int(round(cy / GRID)), int(round(cx / GRID))

    def holds_centre(r):
        i0, j0, hh, ww = r
        return i0 <= ci < i0 + hh and j0 <= cj < j0 + ww

    return dict(
        largest=pick(lambda r: True),
        landscape=pick(lambda r: r[3] >= 1.5 * r[2]),
        near_square=pick(lambda r: r[3] < 1.5 * r[2] and r[2] < 1.5 * r[3]),
        portrait=pick(lambda r: r[2] >= 1.5 * r[3]),
        centred=pick(holds_centre),
    )


def _frozen_dependency_of(src):
    """The `frozen_dependency` block of the map this area came from. -> dict|None.

    `src` is a `*_map.npz`; its sibling `*.json` is what
    `scripts/feasible_workspace.py` writes.  Absent for every map built with
    the shipped pose-invariant bands, which is the default.
    """
    import json as _json
    from pathlib import Path as _Path
    p = _Path(str(src))
    j = p.with_name(p.name[:-len("_map.npz")] + ".json") \
        if p.name.endswith("_map.npz") else None
    if j is None or not j.is_file():
        return None
    try:
        return _json.loads(j.read_text()).get("frozen_dependency")
    except Exception:
        return None


def components_and_holes(live):
    """Largest connected live component and the holes sealed inside it."""
    from scipy import ndimage
    lab, n = ndimage.label(live)
    if n == 0:
        return dict(n_components=0)
    sizes = ndimage.sum(live, lab, range(1, n + 1))
    k = int(np.argmax(sizes)) + 1
    comp = lab == k
    filled = ndimage.binary_fill_holes(comp)
    holes = filled & ~comp
    hl, hn = ndimage.label(holes)
    hsz = sorted((int(v) for v in ndimage.sum(holes, hl, range(1, hn + 1))),
                 reverse=True) if hn else []
    return dict(n_components=int(n),
                largest_component_cells=int(comp.sum()),
                largest_component_m2=round(float(comp.sum()) * GRID * GRID, 4),
                enclosed_holes=int(hn),
                enclosed_hole_cells=int(holes.sum()),
                enclosed_hole_m2=round(float(holes.sum()) * GRID * GRID, 5),
                biggest_holes_cells=hsz[:8])


def dead_breakdown(live, xs, ys, fleet):
    """Dead cells by where they are: under whose base, or out on the rim."""
    X, Y = np.meshgrid(xs, ys)
    dead = ~live
    per_arm, nearest, rmin = {}, None, np.full(X.shape, np.inf)
    for a in sorted(fleet):
        bx, by = fleet[a].xy
        d = np.hypot(X - bx, Y - by)
        per_arm[a] = int((dead & (d < UNDER_BASE_R)).sum())
        nearest = d if nearest is None else np.minimum(nearest, d)
        rmin = np.minimum(rmin, d)
    r = rmin[dead]
    return dict(
        dead_cells=int(dead.sum()),
        dead_pct=round(100.0 * float(dead.mean()), 3),
        under_base_per_arm={int(a): v for a, v in per_arm.items()},
        under_base_total=int(sum(per_arm.values())),
        under_base_m2=round(sum(per_arm.values()) * GRID * GRID, 5),
        rim_cells=int((r >= RIM_R).sum()),
        rim_m2=round(float((r >= RIM_R).sum()) * GRID * GRID, 5),
        between_cells=int(((r >= UNDER_BASE_R) & (r < RIM_R)).sum()),
        r_median=round(float(np.median(r)), 4) if r.size else None,
        r_max=round(float(r.max()), 4) if r.size else None)


def draw(live, xs, ys, rect, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots(figsize=(6.0, 10.4))
    ax.imshow(live, origin="lower", cmap="Greys_r", vmin=0, vmax=1,
              extent=[xs[0] - GRID / 2, xs[-1] + GRID / 2,
                      ys[0] - GRID / 2, ys[-1] + GRID / 2],
              interpolation="nearest")
    dy, dx = np.where(~live)
    ax.plot(xs[dx], ys[dy], ".", ms=2.0, color="#d62728", zorder=3,
            label=f"{len(dx)} dead cells")
    if rect:
        ax.add_patch(Rectangle((rect["x0"] - GRID / 2, rect["y0"] - GRID / 2),
                               rect["w"], rect["h"], fill=False, lw=2.4,
                               edgecolor="#1f9e3a", zorder=5,
                               label=(f"certified {rect['w']:.2f} x "
                                      f"{rect['h']:.2f} m = "
                                      f"{rect['area_m2']:.3f} m²")))
    for a, s in sorted(layout.FLEET_PROPOSED.items()):
        ax.plot(*s.xy, "s", ms=8, mfc="#1f77b4", mec="white", zorder=6)
        ax.annotate(str(a), s.xy, color="white", fontsize=6,
                    ha="center", va="center", zorder=7)
    ax.set_xlim(0, SHEET[0])
    ax.set_ylim(0, SHEET[1])
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--atlas", help="atlas dir: the DRAWING-POSE layer only")
    g.add_argument("--map", help="feasible_workspace _map.npz: all three layers")
    ap.add_argument("--h", type=float, required=True)
    ap.add_argument("--json", default=None)
    ap.add_argument("--png", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--recipe", default=None,
                    help="the pen-up recipe this rectangle DEPENDS ON, e.g. "
                         "'rescue rungs 0,1,2,3'.  A certified area is only "
                         "reproducible if the escalation that produced it is "
                         "named: at h = 0.970 rungs 2-3 are worth 0.082 m² of "
                         "it, so a map rebuilt at rungs 0,1 does not have this "
                         "rectangle.")
    a = ap.parse_args(argv)

    # THE BARE FLEET, because only the base xy is read here (to say which dead
    # cell sits under whose base) and `rig`'s park derivation raises at heights
    # the 0.940 grid does not transfer to — which is a fact about that grid and
    # not a reason to be unable to measure the canvas.
    fl = layout.build_fleet(layout.paired_grid(
        spacing=fw.SHIPPED_PITCH, rows=3, h=float(a.h)))
    if a.atlas:
        live, xs, ys = live_from_atlas(a.atlas, fl)
        src, layers = a.atlas, "draw-pose only (atlas union strict-GO)"
    else:
        live, xs, ys = live_from_map(a.map)
        src, layers = a.map, "draw-pose + hover + reachability"

    rects = best_rects(live, xs, ys)
    doc = dict(h=float(a.h), source=src, layers=layers,
               tool=dict(pen_ext=frames.PEN_EXT_HOLDER,
                         pen_lat=frames.PEN_LAT_HOLDER,
                         lean_deg=round(float(np.rad2deg(
                             frames.PEN_LEAN_HOLDER)), 3)),
               recipe=a.recipe,
               # THE DEPENDENCY TRAVELS WITH THE AREA.  A map built with
               # `--frozen-partners` is certified only while the named arms
               # hold the named poses; that block lives in the map's own JSON
               # and a certified area computed FROM that map inherits it, so a
               # caller reading only this file still learns what it owes.
               frozen_dependency=_frozen_dependency_of(src),
               grid=GRID, sheet=[float(SHEET[0]), float(SHEET[1])],
               cells=int(live.size),
               live_cells=int(live.sum()),
               live_pct=round(100.0 * float(live.mean()), 3),
               rect=rects, **components_and_holes(live),
               **dead_breakdown(live, xs, ys, fl))
    big = rects.get("largest")
    print(f"h = {a.h:.3f}   {layers}")
    print(f"  live {doc['live_pct']:.2f} %   dead {doc['dead_cells']} cells "
          f"({doc['dead_pct']:.2f} %)")
    print(f"  under-base {doc['under_base_total']} "
          f"({doc['under_base_m2']:.4f} m²) per arm {doc['under_base_per_arm']}"
          f";  rim(>= {RIM_R} m) {doc['rim_cells']}; between "
          f"{doc['between_cells']}")
    print(f"  largest component {doc['largest_component_m2']:.3f} m² with "
          f"{doc['enclosed_holes']} enclosed hole(s) = "
          f"{doc['enclosed_hole_m2']:.4f} m²")
    for k in ("largest", "landscape", "near_square", "portrait", "centred"):
        r = rects.get(k)
        print(f"  {k:<12} " + ("-" if r is None else
              f"{r['w']:.2f} x {r['h']:.2f} m = {r['area_m2']:.3f} m² "
              f"at ({r['x0']:.2f}, {r['y0']:.2f})"))
    if a.json:
        Path(a.json).write_text(json.dumps(doc, indent=1))
        print("wrote", a.json)
    if a.png:
        draw(live, xs, ys, big, a.png,
             a.label or f"certified hole-free area, h = {a.h:.3f} "
                        f"({layers})")
        print("wrote", a.png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
