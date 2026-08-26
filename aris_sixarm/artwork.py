"""Where on the paper a traced picture goes, and how big — for ANY picture.

`scripts/csail_place.py` answered that question for the CSAIL mark and answered
it well; every line of the reasoning below is that script's, lifted out of it so
that `scripts/draw.py` runs the SAME search on an arbitrary image instead of a
second implementation of it that has to be kept in step.  The CSAIL script is
now a thin CLI over `search_placement` and its outputs are unchanged.

Two stages, because the two questions cost three orders of magnitude apart:

  PROXY   `allocate.reach_fraction` scores a placement from the atlas alone —
          the length-weighted fraction of the traced path that lands within
          `radius` of a cell some arm's pen can stand on.  ~10 ms, so the whole
          rotation x scale x translation grid is affordable.  It is an UPPER
          BOUND on coverage (a reachable cell is not a plannable stroke) and is
          used only to RANK.

          THE PROXY IS EXACTLY AS HONEST AS THE ATLAS IT READS, and on a rig
          whose arms stand OVER the paper that is the whole question.  Swept
          in empty air, the atlas says the proposed rig covers 99.98 % of the
          canvas and every placement scores ~1.00, so the search ranks on
          nothing; swept with the neighbours' base columns in the obstacle set
          (`mounts.arm_column_boxes`) it says 96.69 %, the dead cells cluster
          under the six bases, and "where the picture goes" becomes a real
          question with a real answer.  A placement whose ink avoids those
          blobs is what `probe_place` measured by hand and what this scores
          for free — the two are the same rasterisation, and the atlas is the
          only thing that had to change.
  REAL    the top `top` translations of every (rotation, scale) are allocated
          for real — probe, colour partition, cover, clean re-plan — which is
          the only number that means anything and the one the choice is made on.

CHOICE RULE (the user's): take the LARGEST placement whose real coverage is
within `slack` of the best coverage anyone achieved, then that cell's best
translation.  Shrinking is allowed, but only bought when it pays.

SIZE IS MEASURED IN SQUARE METRES ON THE PAPER, NOT IN UNITS OF "SCALE".
Rotation is a placement variant (`trace.to_sheet(rotate_deg=…)` turns the pixel
polylines about their own bbox centre before the aspect-preserving fit, so it is
never a distortion), and the same scale 1.0 is a different picture in each
rotation — 1.68 x 1.29 m upright and 1.68 x 2.20 m on its side for the CSAIL
mark.  So the two knobs are ONE grid and "largest" is an area.
"""
import numpy as np

from . import allocate, trace
from .fleet import SHEET


# ---------------------------------------------------------------------------
# Ink names -> a colour to DRAW them in.  The tracer measures each ink's own RGB
# off the picture, so the honest default is the picture's own colour and this
# table is only the fallback for a name that arrived without one (and the two
# CSAIL inks, whose published hexes every existing figure and animation uses).
INK_HEX = {"grey": "#%02x%02x%02x" % trace.GREY_RGB,
           "orange": "#%02x%02x%02x" % trace.ORANGE_RGB,
           "black": "#111111", "white": "#dddddd", "red": "#cc2222",
           "yellow": "#d8c020", "green": "#2f9e44", "cyan": "#22a8bd",
           "blue": "#2455c8", "purple": "#7a3fb4", "magenta": "#c0329c",
           "brown": "#8a5628"}
FALLBACK_HEX = "#333333"


def hex_of(name, palette=None):
    """Ink name -> '#rrggbb'.  `palette` (from the trace) wins over the table."""
    if palette and name in palette:
        v = palette[name]
        if isinstance(v, str):
            return v
        return "#%02x%02x%02x" % tuple(int(np.clip(round(x), 0, 255)) for x in v)
    return INK_HEX.get(name, FALLBACK_HEX)


def palette_of(dbg):
    """A tracer debug dict -> {ink name: '#rrggbb'}.

    An ink drawn at its own measured colour is the one choice that cannot be
    wrong: the final still and the animation then show the paper as the picture
    describes it rather than as a palette in this module imagines it.
    """
    names = list(dbg.get("names") or [])
    pal = list(dbg.get("palette") or [])
    out = {}
    for i, nm in enumerate(names):
        out[nm] = ("#%02x%02x%02x" % tuple(int(np.clip(v, 0, 255)) for v in pal[i])
                   if i < len(pal) else INK_HEX.get(nm, FALLBACK_HEX))
    return out


def inks_of(strokes):
    """The inks a stroke set actually uses, darkest-first order preserved."""
    seen = []
    for s in strokes:
        if s["color"] not in seen:
            seen.append(s["color"])
    return seen


# ---------------------------------------------------------------------------
def base_width(px, margin, rotate_deg=0.0, sheet=None):
    """The widest this picture goes on the active sheet at this rotation. -> m.

    `--base-width auto` in the CSAIL script, and the default here, because a
    constant is only right for the sheet it was measured on: on the 1.8034 m
    canvas the legacy 2.4106 m base made every scale below 0.70 collapse onto
    the same picture (`out/csail_placement_final.json`: seven scales, one
    width).  Measured per rotation, scale 1.0 means "as big as it goes".
    """
    return trace.to_sheet(px, sheet or SHEET, margin=margin,
                          rotate_deg=rotate_deg)[1]["logo_w"]


def place(px, f, dx, dy, margin, rot=0.0, base=None, sheet=None, min_len=0.025):
    """One candidate -> `trace.to_sheet`'s (strokes, info)."""
    sh = sheet or SHEET
    b = base_width(px, margin, rot, sh) if base is None else base
    return trace.to_sheet(px, sh, margin=margin, target_width=f * b,
                          offset=(dx, dy), rotate_deg=rot, min_len=min_len)


def proxy_grid(px, grids, scales, offs, margin, radius, rots, bases, sheet=None,
               min_len=0.025):
    """-> list of dicts, one per (rotation, scale, dx, dy) that fits the margin."""
    out = []
    for rot in rots:
        for f in scales:
            for dx in offs:
                for dy in offs:
                    st, info = place(px, f, dx, dy, margin, rot, bases[rot],
                                     sheet, min_len)
                    if not info["fits"]:
                        continue
                    out.append(dict(rot=float(rot), f=float(f), dx=float(dx),
                                    dy=float(dy),
                                    proxy=allocate.reach_fraction(st, grids, radius),
                                    logo_w=info["logo_w"], logo_h=info["logo_h"]))
    return out


def _real_one(args):
    """One placement, allocated for real.  Module level so `mp.Pool` can send it."""
    px, rot, f, dx, dy, margin, arms, atlas_dir, base, colors, kw, min_len = args
    st, info = place(px, f, dx, dy, margin, rot, base, None, min_len)
    res = allocate.allocate(st, arms=arms, atlas_dir=atlas_dir, verbose=False,
                            colors=(None if colors is None
                                    else {a: colors for a in arms}),
                            **(kw or {}))
    return dict(rot=float(rot), f=float(f), dx=float(dx), dy=float(dy),
                drawn=res["drawn_len"], traced=res["total_len"],
                cov=res["drawn_len"] / max(res["total_len"], 1e-9),
                logo_w=info["logo_w"], logo_h=info["logo_h"],
                target_width=float(f * base),
                loads={int(a): float(sum(s["length"] for s in res["programs"][a]))
                       for a in res["arms"]})


def search_placement(px, arms, atlas_dir, margin=0.06, radius=0.03,
                     scales=(0.7, 1.0, 7), offset=0.30, offset_step=0.10,
                     rotations=(0.0,), top=3, slack=0.01, jobs=6,
                     base=None, single_ink=None, alloc_kw=None, sheet=None,
                     min_len=0.025, verbose=True):
    """Proxy-rank, allocate for real, apply the choice rule. -> dict.

    `single_ink` is the ink name to fix every arm's pen to.  A ONE-INK picture
    must say so: left None the allocator enumerates the 2^n - 2 grey/orange
    partitions (`allocate.best_partition`), which on a single-ink drawing gives
    half the fleet the wrong pen and halves the coverage the search is ranking.

    `alloc_kw` is handed to every real allocation.  Left None the allocator runs
    with all of its defaults, which is what `scripts/csail_place.py` has always
    done and what its published placements were measured with.  A caller that
    is only RANKING can afford to switch off the two passes that provably cannot
    move the number it is ranking on — `balance` re-assigns spans a second arm
    already certified at the same endpoints, and `split` re-measures coverage
    and is invariant by construction — and to use the cheap sequencer, since no
    ordering changes what is drawn.  On a picture with 16 m of path that is the
    difference between a search that takes half an hour and one that takes
    three; see `scripts/draw.py`, which passes exactly that set.
    """
    sh = sheet or SHEET
    grids = allocate.atlas_cells(arms, atlas_dir)
    if grids is None:
        raise SystemExit(f"no atlas for arms {arms} under {atlas_dir}")
    rots = [float(r) for r in rotations]
    bases = {r: (base_width(px, margin, r, sh) if base is None else float(base))
             for r in rots}
    sc = np.linspace(scales[0], scales[1], int(scales[2]))
    n_off = int(round(offset / offset_step))
    offs = np.arange(-n_off, n_off + 1) * offset_step

    if verbose:
        print(f"canvas {sh[0]:.4f} x {sh[1]:.5f} m, margin {margin} m, "
              f"{len(arms)} arms {arms}")
        for r in rots:
            _, i0 = place(px, 1.0, 0.0, 0.0, margin, r, bases[r], sh, min_len)
            print(f"  rotation {r:>5.1f} deg: scale 1.0 = {i0['logo_w']:.4f} x "
                  f"{i0['logo_h']:.4f} m  (base width {bases[r]:.4f} m)")

    grid = proxy_grid(px, grids, sc, offs, margin, radius, rots, bases, sh,
                      min_len)
    if verbose:
        print(f"proxy: {len(grid)} placements fit the sheet ({len(rots)} "
              f"rotations x {len(sc)} scales x {len(offs)}^2 offsets)")
    jobs_list = []
    for rot in rots:
        for f in sc:
            cand = sorted([g for g in grid if abs(g["f"] - f) < 1e-9
                           and abs(g["rot"] - rot) < 1e-9],
                          key=lambda g: -g["proxy"])[:top]
            for g in cand:
                jobs_list.append((px, rot, g["f"], g["dx"], g["dy"], margin,
                                  arms, atlas_dir, bases[rot], single_ink,
                                  alloc_kw, min_len))
    if verbose:
        print(f"real: allocating {len(jobs_list)} placements "
              f"({top} per rotation x scale)...")
    if jobs > 1 and len(jobs_list) > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(jobs) as pool:
            real = pool.map(_real_one, jobs_list)
    else:
        real = [_real_one(j) for j in jobs_list]

    def proxy_of(r):
        """The LIVE-INK fraction of one placement: how much of the path lands
        on a cell some arm can actually stand a pen on."""
        for g in grid:
            if all(abs(g[k] - r[k]) < 1e-9 for k in ("rot", "f", "dx", "dy")):
                return float(g["proxy"])
        return float("nan")

    per_cell = {}
    for r in real:
        k = (round(r["rot"], 6), round(r["f"], 6))
        if k not in per_cell or r["cov"] > per_cell[k]["cov"]:
            per_cell[k] = r
    best = max(r["cov"] for r in real)
    ok = [r for r in per_cell.values() if r["cov"] >= best - slack]
    pick = max(ok, key=lambda r: (round(r["logo_w"] * r["logo_h"], 9), r["cov"]))

    if verbose:
        print(f"\n{'rot':>5} {'scale':>6} {'w x h':>17} {'best offset':>16} "
              f"{'proxy':>7} {'REAL cov':>9}")
        for k in sorted(per_cell):
            r = per_cell[k]
            p = max(g["proxy"] for g in grid
                    if abs(g["f"] - r["f"]) < 1e-9 and abs(g["rot"] - r["rot"]) < 1e-9)
            print(f"{r['rot']:>5.0f} {r['f']:>6.3f} {r['logo_w']:>7.3f} x "
                  f"{r['logo_h']:<7.3f} ({r['dx']:>+5.2f},{r['dy']:>+5.2f}) m "
                  f"{p:>7.3f} {100 * r['cov']:>8.1f} %"
                  + ("   <- chosen" if r is pick else ""))
        print(f"\nbest coverage anywhere {100 * best:.1f} %; largest within "
              f"{100 * slack:.0f} pp of it = {pick['logo_w']:.3f} x "
              f"{pick['logo_h']:.3f} m ({pick['logo_w'] * pick['logo_h']:.3f} m2, "
              f"rotation {pick['rot']:.0f} deg, scale {pick['f']:.3f}) at offset "
              f"({pick['dx']:+.2f}, {pick['dy']:+.2f}) m, "
              f"{100 * pick['cov']:.1f} % drawn, "
              f"{100 * proxy_of(pick):.2f} % of its ink on LIVE cells")

    return dict(chosen=dict(scale=pick["f"], target_width=pick["target_width"],
                            rotate_deg=pick["rot"],
                            offset=[pick["dx"], pick["dy"]], margin=margin,
                            logo_w=pick["logo_w"], logo_h=pick["logo_h"],
                            coverage=pick["cov"], live=proxy_of(pick)),
                arms=[int(x) for x in arms], base_width=bases,
                sheet=[float(sh[0]), float(sh[1])], rotations=rots,
                atlas=atlas_dir, proxy_radius=radius,
                rule=f"largest AREA within {slack} of the best real coverage, "
                     "over rotations x scales",
                proxy=grid, real=real,
                per_scale={f"{k[0]:.0f}deg@{k[1]:.3f}": per_cell[k]
                           for k in sorted(per_cell)})
