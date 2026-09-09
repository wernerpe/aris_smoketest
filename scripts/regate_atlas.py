"""THE ATLAS CERTIFIES INK AT 50 mm AND THE ROUTER MAY NOT FLY TO IT UNDER 63.

`atlas.solve_cell` gates a drawing pose against the neighbours' body-column
boxes at `rig_final.STATIC_MARGIN` = 50 mm.  `paper.route` holds every pen-up
LEG to `paper.FRAME_FLOOR` = `rig_final.STATIC_PLAN_MARGIN` = 63 mm — 50 plus
the 13 mm of slack `scene_check`'s own independent lower bound carries, which
the producer must pay or the checker is a lottery rather than a second opinion.
`rig_final` says so in as many words: "an atlas is now an OPTIMISTIC prefilter
by up to 13 mm".

MEASURED, IT IS NOT A ROUNDING ERROR — IT IS THE DEAD SET.  On the v11 map, of
the 1 112 refused arm-cells that sit on a dead canvas cell, 772 have a drawing
pose whose own clearance to the boxes lies in [50, 63) mm — and ZERO of the
22 437 FEASIBLE arm-cells do.  Not a sample: all of them.  The band and the
refused set coincide, which is also why re-gating there cannot cost a cell.

THAT IS NOT A PROOF OF INFEASIBILITY AND MUST NOT BE READ AS ONE.
`paper.effective_static_floor` clamps a leg's static floor down to what its own
ENDPOINTS hold, and it exists to stop precisely that contradiction: a pose in
the band is asked for its own clearance, not for 63 mm.  What such a pose HAS
is nothing left over.  Clamped, the descent has to hold the pose's own
clearance along its whole swing, and a swing dips.  Measured over 36 route-dead
cells and all 48 hovers on each one's fiber: 27 are walled at that clamped
static floor, nothing else binds on any of them — not the paper, not the tip,
not the arm against itself — and the best hover misses BY A MEDIAN OF 1.4 mm.
The drawing poses under them sit at a median 58.5 mm, on the gate.

...AND THE POSE IS NOT THE CELL.  `solve_cell` returns the FIRST gated pose
that clears 50 mm, in descending joint-margin order, and it never asks whether
another one clears more.  Probed on all 772: 746 of them (96.6 %) have a
different pose at the same cell — another tool yaw, another q7, another IK
branch, or a lean inside the same 15-degree cone — that clears 63 mm, with a
median of 120 mm, and 446 of those need no lean at all.  A descent out of 120
mm of air is not threading a 1.4 mm gap.

So this re-solves them.  Every strict-GO row whose pose is under `--floor` is
re-searched at `--floor` by the SAME `solve_cell` under the SAME gates, and:

  * a row that finds a pose gets that pose (the search is strictly harder, so
    the answer is a pose the old gate would also have accepted);
  * a row that finds none is DROPPED, because a certified cell nobody can fly
    to is not a certified cell — it is the thing this file exists to stop the
    atlas from claiming;
  * every other row is copied byte for byte.

The result is stamped with `static_margin` so a reader can tell which floor it
was gated at.  It is written to a NEW directory: the shipped
`out/atlas_proposed_h0940_gated` is not touched, and neither is any constant.

    ARIS_RIG=proposed ARIS_TOOL=lateral python3 scripts/regate_atlas.py \
        --in out/atlas_proposed_h0940_gated \
        --out out/atlas_proposed_h0940_gated63
"""
import argparse
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

from aris_sixarm import atlas, frames, layout, paper, rig_final   # noqa: E402
from aris_sixarm import envelope, frozen

_W = {}


def _init(floor, cone, h=None):
    _W["floor"] = float(floor)
    _W["cone"] = tuple(cone)
    _W["fleet"] = _fleet_at(h)
    _W["h"] = float(layout.LAYOUT_PROPOSED["h"]) if h is None else float(h)


def _fleet_at(h=None):
    """The rig to re-solve against. -> fleet.

    THE BASE POSE IS THE WHOLE POINT OF A RE-GATE.  `solve_cell` re-searches
    each row against the neighbours' steel and body columns at a stricter
    floor, and every one of those clearances is measured from the arm's base.
    Re-gating an atlas swept at one height against a fleet built at another
    silently answers a question about a rig that does not exist -- and this
    script bound `layout.FLEET_PROPOSED` unconditionally, so a height study
    that re-gated its own atlas got the shipped 0.940 bases.  The parks are
    NOT read here (no routing happens), so only the geometry is rebuilt.
    """
    if h is None:
        return layout.FLEET_PROPOSED
    return layout.build_fleet(layout.paired_grid(
        spacing=layout.PAIR_SPACING, rows=3, h=float(h)))


def _resolve(job):
    """(arm, x, y) -> the row `solve_cell` returns at the stricter floor."""
    a, x, y = job
    spec = _W["fleet"][a]
    h = _W["h"]
    # THE SAME ROOM THE ROUTER USES, when one is installed: body columns as
    # cylinders and parked partners as their real capsules.  Both are inert
    # unless `--parks`/the model was switched on, so a plain re-gate is the
    # bit-identical one it always was (aris_sixarm/envelope.py, frozen.py).
    frozen.observe(a)
    boxes = envelope.swap(frozen.filter_boxes(spec.static_obstacles()))
    Twb = spec.T_world_base(h)
    # the HOLDER's OWN axial depth, not the spec's (which is the inline pen's
    # 0.110 unless ARIS_TOOL says otherwise): naming pen_lat and letting
    # pen_ext fall through asks about a tool that exists nowhere.  frames.py.
    r = atlas.solve_cell(x, y, Twb, np.linalg.inv(Twb), spec,
                         atlas._candidates(15.0), frames.PEN_EXT_HOLDER, boxes,
                         pen_lat=frames.PEN_LAT_HOLDER,
                         gate_groups=atlas._gated_groups(15.0, _W["cone"]),
                         static_margin=_W["floor"])
    if r is None:
        return a, x, y, None
    return a, x, y, [x, y, *r[:7], *r[7], r[8], r[9]]


def regate(arm, src, dst, floor, cone, pool, log=print, h_over=None):
    arr, meta = atlas.load(src, arm)
    ok, why = atlas.is_current(meta)
    if not ok:
        raise SystemExit(f"atlas for arm {arm} is STALE: {why}")
    fl = _fleet_at(h_over)
    spec = fl[arm]
    h = float(layout.LAYOUT_PROPOSED["h"]) if h_over is None else float(h_over)
    go = atlas.strict_go(arr)
    boxes = paper.static_boxes(spec)
    sc = np.full(len(arr), np.inf)
    if go.any():
        sc[go] = np.asarray(paper.chain_static(
            np.asarray(arr[go][:, atlas.QCOL:atlas.QCOL + 7], float),
            spec, spec.pen, h, boxes), float)
    need = go & (sc < floor)
    log(f"  arm {arm}: {len(arr)} reachable, {int(go.sum())} strict-GO, "
        f"{int(need.sum())} under {1000 * floor:.0f} mm "
        f"({100.0 * need.sum() / max(1, go.sum()):.1f} % of them)", flush=True)
    if not need.any():
        return arr, meta, 0, 0

    idx = np.flatnonzero(need)
    jobs = [(int(arm), float(arr[i, 0]), float(arr[i, 1])) for i in idx]
    got = pool.map(_resolve, jobs, chunksize=4)
    keep = np.ones(len(arr), bool)
    fixed = dropped = 0
    for i, (_a, _x, _y, row) in zip(idx, got):
        if row is None:
            keep[i] = False
            dropped += 1
        else:
            arr[i] = np.asarray(row, float)
            fixed += 1
    return arr[keep], meta, fixed, dropped


def _cells(d, arm):
    """{(x, y): row index} for one atlas npz."""
    z = np.load(Path(d) / f"atlas_arm{arm}.npz")
    C = list(z["columns"])
    ix, iy = C.index("x"), C.index("y")
    return {(round(float(r[ix]), 4), round(float(r[iy]), 4)): i
            for i, r in enumerate(z["data"])}


def compare(src, dst, arms, fl, band=None):
    """What the re-gate took, per arm and as REDUNDANCY. -> None.

    THE MAP CANNOT SEE THIS AND THE CONDUCTOR LIVES ON IT.  The feasibility
    map is a UNION over arms — a cell is alive if ANY arm can draw it — so a
    re-gate that takes a cell away from the SECOND arm that could reach it
    costs the map nothing at all.  The allocator's fallback is exactly that
    second arm, and when a phase is refused, having another arm that can draw
    the same ink is the difference between a re-offer and a skip.  So the
    honest question about a re-gate is not "how many cells died" (none) but
    "how many cells lost an arm", and this asks it.
    """
    print(f"\n=== what {dst} drops against {src} ===")
    tot = 0
    per_cell = {}
    for arm in arms:
        A, B = _cells(src, arm), _cells(dst, arm)
        lost = sorted(set(A) - set(B))
        tot += len(lost)
        b = fl[arm].T_world_base(1.0)[:2, 3]
        line = (f"  arm {arm:>3}: {len(A):5d} -> {len(B):5d} rows, "
                f"dropped {len(lost):4d} ({100.0 * len(lost) / max(len(A), 1):4.1f} %)")
        if lost:
            L = np.array(lost, float)
            c = L.mean(axis=0)
            line += (f"   band x[{L[:, 0].min():.2f},{L[:, 0].max():.2f}] "
                     f"y[{L[:, 1].min():.2f},{L[:, 1].max():.2f}], "
                     f"{np.linalg.norm(c - b):.2f} m from its own base "
                     f"({b[0]:.2f}, {b[1]:.2f})")
            if band:
                inb = int(((L[:, 0] >= band[0]) & (L[:, 0] <= band[1])
                           & (L[:, 1] >= band[2]) & (L[:, 1] <= band[3])).sum())
                line += f", {inb} of them inside --band"
        print(line)
        for k in A:
            per_cell.setdefault(k, [0, 0])[0] += 1
        for k in B:
            per_cell.setdefault(k, [0, 0])[1] += 1
    print(f"  {tot} rows dropped in total")

    sel = [v for k, v in per_cell.items()
           if band is None or (band[0] <= k[0] <= band[1]
                               and band[2] <= k[1] <= band[3])]
    if not sel:
        return
    n = len(sel)
    ha, hb = {}, {}
    for va, vb in sel:
        ha[va] = ha.get(va, 0) + 1
        hb[vb] = hb.get(vb, 0) + 1
    where = "the whole canvas" if band is None else \
        f"x[{band[0]:.3f},{band[1]:.3f}] y[{band[2]:.3f},{band[3]:.3f}]"
    print(f"\n  REDUNDANCY over {n} cells of {where} "
          "(how many arms have a drawing pose there)")
    for h, tag in ((ha, str(src)), (hb, str(dst))):
        tot_a = sum(k * v for k, v in h.items())
        print("    " + "  ".join(f"{k} arm(s): {h.get(k, 0):4d}"
                                 for k in sorted(set(ha) | set(hb)) if k)
              + f"   mean {tot_a / n:.3f} arms/cell   {tag}")
    moved = sum(1 for va, vb in sel if vb < va)
    print(f"    {moved} cells lost an arm; "
          f"{sum(1 for va, vb in sel if vb == 0 < va)} lost their last one")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true",
                    help="do not re-gate: report what --out already drops "
                         "against --in, per arm and as REDUNDANCY over "
                         "--band.  This is the A/B a union-valued map cannot "
                         "show (see `compare`).")
    ap.add_argument("--band", default="",
                    help="x0,x1,y0,y1 in canvas metres to report redundancy "
                         "over; empty is the whole canvas")
    ap.add_argument("--in", dest="src",
                    default=str(ROOT / "out" / "atlas_proposed_h0940_gated"))
    ap.add_argument("--out", dest="dst",
                    default=str(ROOT / "out" / "atlas_proposed_h0940_gated63"))
    ap.add_argument("--floor", type=float, default=float(paper.FRAME_FLOOR),
                    help="the static floor the drawing pose must keep to the "
                         "neighbours' boxes.  The default is the ROUTER's own "
                         "(paper.FRAME_FLOOR = rig_final.STATIC_PLAN_MARGIN), "
                         "which is the whole point: an atlas whose cells the "
                         "router may not fly to is an atlas that overstates "
                         "the canvas.")
    ap.add_argument("--h", type=float, default=None, metavar="M",
                    help="REPORT-ONLY.  Re-gate an atlas swept at a mounting "
                         "height other than the shipped one.  Every clearance "
                         "re-solved here is measured from the arm's BASE, so "
                         "re-gating a 0.970 atlas against the 0.940 fleet "
                         "answers a question about a rig that does not exist.")
    ap.add_argument("--arms", default="")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 8))
    ap.add_argument("--model-parks", default=None, metavar="JSON",
                    help="switch the 2026-09-09 neighbour model ON for this "
                         "re-gate: body columns as cylinders and these parks' "
                         "poses as the partners' real capsules.  Omit to "
                         "re-gate against the shipped bounding boxes.")
    a = ap.parse_args()

    fl = _fleet_at(a.h)
    if a.model_parks:
        import json as _json
        if a.model_parks == "shipped":
            _pk = {int(k): np.asarray(v, float)
                   for k, v in layout.Q_PARK_PROPOSED.items()}
        else:
            _doc = _json.load(open(a.model_parks))
            _pk = {int(k): np.asarray(v["q"], float)
                   for k, v in _doc["best"].items()}
        _fl2 = _fleet_at(a.h)
        envelope.install(_fl2, float(a.h if a.h else layout.LAYOUT_PROPOSED["h"]))
        frozen.freeze(_pk, _fl2, {x: _fl2[x].pen for x in _fl2},
                      float(a.h if a.h else layout.LAYOUT_PROPOSED["h"]))
        print("neighbour model ON: cylinders + frozen partners from %s"
              % a.model_parks)

    arms = [int(v) for v in a.arms.split(",")] if a.arms else sorted(fl)
    if a.compare:
        band = tuple(float(v) for v in a.band.split(",")) if a.band else None
        if band is not None and len(band) != 4:
            raise SystemExit("--band wants x0,x1,y0,y1")
        return compare(a.src, a.dst, arms, fl, band)
    dst = Path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)
    _h_shown = a.h if a.h is not None else layout.LAYOUT_PROPOSED["h"]
    print(f"rig=proposed tool=lateral h={_h_shown}")
    print(f"atlas gate  STATIC_MARGIN      = "
          f"{1000 * rig_final.STATIC_MARGIN:.0f} mm  (what {a.src} was swept at)")
    print(f"router floor FRAME_FLOOR       = "
          f"{1000 * paper.FRAME_FLOOR:.0f} mm")
    print(f"re-gating at {1000 * a.floor:.0f} mm into {dst}\n", flush=True)

    import multiprocessing as mp
    t0 = time.time()
    tot_fixed = tot_dropped = 0
    with mp.get_context("fork").Pool(a.workers, initializer=_init,
                                     initargs=(a.floor,
                                               atlas.GATE_CONE_DEG,
                                               a.h)) as pool:
        for arm in arms:
            arr, meta, fixed, dropped = regate(arm, Path(a.src), dst, a.floor,
                                               atlas.GATE_CONE_DEG, pool,
                                               h_over=a.h)
            tot_fixed += fixed
            tot_dropped += dropped
            out = dst / f"atlas_arm{arm}.npz"
            kw = {k: meta[k] for k in meta.files if k not in ("data",)}
            kw["data"] = arr
            kw["static_margin"] = float(a.floor)
            np.savez_compressed(out, **kw)
            go = atlas.strict_go(arr)
            print(f"    -> {len(arr)} reachable, {int(go.sum())} strict-GO   "
                  f"({fixed} re-posed, {dropped} dropped)", flush=True)
    print(f"\n{tot_fixed} rows re-posed, {tot_dropped} dropped, "
          f"{time.time() - t0:.0f}s")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
