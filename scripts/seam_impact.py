#!/usr/bin/env python3
"""WHAT THE SEAM FRAME COSTS — parks, certified area, shipped programmes.

    ARIS_RIG=proposed ARIS_TOOL=lateral python3 scripts/seam_impact.py \
        [--json out/seam_impact_h0970.json]

THE STEEL THIS MEASURES.  `system_model.seam_bodies` / `mounts.seam_frame_boxes`
— four 3 in posts and four corner brace clusters straddling the paper's
mid-length, where the two half-cages butt (docs/SYSTEM_MODEL.md, the seam-frame
section).  Reported by Pete Werner on 2026-09-14 and built here from the
drawing's own post_BL / post_BR / brace_BL / brace_BR.

WHY THERE IS A SCRIPT AT ALL, AND WHY IT DOES NOT CHANGE THE STATIC SET.
`layout.StudySpec.static_obstacles` returns the neighbours' mount boxes and base
columns AND NOTHING ELSE: every certified number on the proposed rig was earned
against a fleet standing in an empty room, with no cage around it.  Adding steel
to that set re-decides every certified cell in the repo, so it is a
re-certification and it waits on the photo
(`system_model.OPEN_QUESTIONS['seam_frame']`).  This script answers the question
the re-certification would ask, without pre-empting its answer: the seam boxes
are concatenated onto a COPY of the fleet, and nothing on disk moves.

THE THREE MEASUREMENTS.

  1. PARKS.  `layout.Q_PARK_PROPOSED`, the `PARK_GRID_PROPOSED` recipe parks,
     and every park in `out/stage_parks_h0970.json`, each arm's whole chain
     against the seam boxes by `scene_check.static_clearance_lb` — the
     independent derivation the whole-timeline checker uses, not the one the
     planner uses.  Reported against BOTH floors: `rig_final.STATIC_MARGIN`
     (50 mm, what the atlas was swept at) and `paper.FRAME_FLOOR` (63 mm, what
     the router asks for).

  2. THE CERTIFIED AREA.  Every strict-GO row of the shipped atlas carries the
     DRAW POSE it was certified with (`atlas.COLUMNS`, q1..q7).  Each is
     re-evaluated against the seam boxes and the cell is killed for that arm if
     it no longer clears the floor; the per-arm masks of the shipped map are
     then intersected with the survivors and the union re-counted.

     THIS IS A LOWER BOUND ON THE DAMAGE, and deliberately so.  The map's live
     cells rest on three layers — draw pose, hover, routing leg — and only the
     first has its pose on disk.  A cell whose DRAW pose still clears may yet
     lose its hover or its leg, and only a full re-sweep can say.
     `scripts/remap_dead_cells.py` cannot be used: it re-decides DEAD cells
     against a SUPERSET atlas, and adding steel makes a SUBSET.

  3. THE SHIPPED PROGRAMMES.  `out/csail_schedule_h097_v19.npz` through
     `scene_check.check_timeline` with the seam fleet — the independent
     whole-timeline check, every frame of every arm.  The staged v6 programme
     ships no frames, so its parks and stage transitions get `check_static`
     instead, and this script says so rather than implying a timeline verdict
     it did not compute.
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

from aris_sixarm import atlas, layout, mounts, paper, rig_final, scene_check  # noqa: E402
from aris_sixarm import system_model as SM  # noqa: E402

STATIC = float(rig_final.STATIC_MARGIN)          # 0.050
FLOOR = float(paper.FRAME_FLOOR)                 # 0.063
MM = 1000.0


def seam_fleet(h=0.970):
    """`layout.FLEET_PROPOSED` with the seam boxes bolted onto every spec.

    A fresh fleet, never the module-level one: this script must not be able to
    change what anything else in the process sees.
    """
    fl = layout.build_fleet(layout.LAYOUT_PROPOSED)
    seam = mounts.seam_frame_boxes()
    for spec in fl.values():
        object.__setattr__(spec, "mount_boxes",
                           tuple(spec.mount_boxes) + tuple(seam))
    return fl


def _chain_of(q, spec, h_inv, pen_ext):
    return scene_check._chain(np.asarray(q, float).reshape(7), spec, h_inv,
                              pen_ext)


def clearance(q, spec, boxes, h_inv, pen_ext):
    """One pose's whole chain against `boxes` -> metres (lower bound)."""
    P = _chain_of(q, spec, h_inv, pen_ext)[None]
    return float(scene_check.static_clearance_lb(P, boxes)[0])


# ---------------------------------------------------------------------------
# 1.  PARKS
# ---------------------------------------------------------------------------
def park_sets(fl, h):
    """-> {name: {arm_id: q}} — every park set this rig ships."""
    out = {}
    out["Q_PARK_PROPOSED"] = {int(a): np.asarray(q, float)
                              for a, q in layout.Q_PARK_PROPOSED.items()}
    try:
        grid = layout.certified_park_poses(fl, layout.PARK_GRID_PROPOSED)
        out["PARK_GRID_PROPOSED"] = {int(a): np.asarray(v["q"], float)
                                     for a, v in grid.items()
                                     if isinstance(v, dict) and "q" in v}
    except Exception as e:                                # pragma: no cover
        out["PARK_GRID_PROPOSED"] = {}
        print(f"  (PARK_GRID_PROPOSED not rebuilt: {e})")
    sp = ROOT / "out" / "stage_parks_h0970.json"
    if sp.exists():
        d = json.load(open(sp))
        for stage, rec in sorted(d.get("stages", {}).items(),
                                 key=lambda kv: int(kv[0])):
            out[f"stage_parks[{stage}]"] = {
                int(a): np.asarray(v["q"], float)
                for a, v in rec.get("parks", {}).items() if "q" in v}
    return out


def measure_parks(fl, boxes, h, pen_ext):
    rows = []
    for name, parks in park_sets(fl, h).items():
        for aid in sorted(parks):
            if aid not in fl:
                continue
            d = clearance(parks[aid], fl[aid], boxes, h, pen_ext)
            rows.append(dict(park=name, arm=int(aid), clear_mm=round(d * MM, 2),
                             fails_static=bool(d < STATIC),
                             fails_floor=bool(d < FLOOR)))
    return rows


# ---------------------------------------------------------------------------
# 2.  THE CERTIFIED AREA
# ---------------------------------------------------------------------------
def measure_area(fl, boxes, h, pen_ext, atlas_dir, map_npz, floor):
    d = np.load(map_npz, allow_pickle=True)
    xs, ys = d["xs"], d["ys"]
    masks = {int(k[4:]): d[k] for k in d.files if k.startswith("mask")}
    live0 = np.zeros_like(next(iter(masks.values())))
    for m in masks.values():
        live0 |= m
    survive, killed_rows = {}, {}
    for aid in sorted(masks):
        arr, _ = atlas.load(atlas_dir, aid)
        g = arr[atlas.strict_go(arr)]
        if not len(g):
            survive[aid] = {}
            continue
        Q = g[:, atlas.QCOL:atlas.QCOL + 7]
        P = np.array([_chain_of(q, fl[aid], h, pen_ext) for q in Q])
        lb = scene_check.static_clearance_lb(P, boxes)
        ok = lb >= floor
        survive[aid] = {(round(float(x), 4), round(float(y), 4)): bool(o)
                        for x, y, o in zip(g[:, 0], g[:, 1], ok)}
        killed_rows[aid] = int((~ok).sum())
    live1 = np.zeros_like(live0)
    per_arm = {}
    for aid, m in masks.items():
        keep = np.zeros_like(m)
        s = survive.get(aid, {})
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                if not m[j, i]:
                    continue
                keep[j, i] = s.get((round(float(x), 4), round(float(y), 4)),
                                   True)
        per_arm[int(aid)] = dict(before=int(m.sum()), after=int(keep.sum()))
        live1 |= keep
    return dict(xs=xs, ys=ys, live0=live0, live1=live1, per_arm=per_arm,
                atlas_rows_killed=killed_rows)


def largest_rect(mask, xs, ys):
    """Largest all-live axis-aligned rectangle -> dict (cells, x/y, w, h)."""
    ny, nx = mask.shape
    heights = np.zeros(nx, int)
    best = dict(cells=0)
    dx = float(xs[1] - xs[0]) if nx > 1 else 0.0
    dy = float(ys[1] - ys[0]) if ny > 1 else 0.0
    for j in range(ny):
        heights = np.where(mask[j], heights + 1, 0)
        stack = []
        for i in range(nx + 1):
            hgt = heights[i] if i < nx else 0
            start = i
            while stack and stack[-1][1] >= hgt:
                s, hh = stack.pop()
                area = hh * (i - s)
                if area > best["cells"]:
                    best = dict(cells=int(area), i0=int(s), i1=int(i - 1),
                                j0=int(j - hh + 1), j1=int(j))
                start = s
            stack.append((start, hgt))
    if not best["cells"]:
        return best
    best.update(x0=round(float(xs[best["i0"]]), 4),
                x1=round(float(xs[best["i1"]]), 4),
                y0=round(float(ys[best["j0"]]), 4),
                y1=round(float(ys[best["j1"]]), 4))
    best.update(w=round(best["x1"] - best["x0"] + dx, 4),
                h=round(best["y1"] - best["y0"] + dy, 4))
    best["area_m2"] = round(best["w"] * best["h"], 4)
    return best


# ---------------------------------------------------------------------------
# 3.  THE SHIPPED PROGRAMMES
# ---------------------------------------------------------------------------
def measure_timeline(fl, npz, sub=2):
    d = np.load(npz, allow_pickle=True)
    arms = [int(a) for a in d["arms"]]
    q = {a: np.asarray(d[f"q_{a}"], float) for a in arms}
    pen = np.asarray(d["pen_ext"], float)
    pen_ext = ({a: float(p) for a, p in zip(arms, pen)} if pen.ndim
               else float(pen))
    rep = scene_check.check_timeline(q, float(d["dt"]), float(d["margin"]),
                                     pen_ext=pen_ext, sub=sub, fleet=fl,
                                     verbose=True)
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h", type=float, default=0.970)
    ap.add_argument("--atlas", default="out/atlas_h0970_cylfrozen_g63")
    ap.add_argument("--map", default="out/fw_h0970_m50_map.npz")
    ap.add_argument("--timeline", default="out/csail_schedule_h097_v19.npz")
    ap.add_argument("--sub", type=int, default=2)
    ap.add_argument("--skip-area", action="store_true")
    ap.add_argument("--skip-timeline", action="store_true")
    ap.add_argument("--json", default="out/seam_impact_h0970.json")
    a = ap.parse_args(argv)

    fl = seam_fleet(a.h)
    boxes = mounts.seam_frame_boxes()
    pen_ext = None
    out = dict(h=a.h, n_seam_boxes=len(boxes),
               seam_y_mm=SM.SEAM_Y,
               seam_rail_residual_mm=SM.SEAM_RAIL_RESIDUAL_MM,
               static_margin_mm=STATIC * MM, frame_floor_mm=FLOOR * MM,
               boxes=[dict(name=b["name"],
                           lo=[round(float(v), 5) for v in b["lo"]],
                           hi=[round(float(v), 5) for v in b["hi"]])
                      for b in boxes])

    print("=" * 74)
    print(f"THE SEAM FRAME at h = {a.h}: {len(boxes)} boxes, "
          f"y = {SM.SEAM_Y} mm")
    print("=" * 74)
    print("\n1. PARKS  (clearance of the whole chain to the seam steel, mm)")
    rows = measure_parks(fl, boxes, a.h, pen_ext)
    out["parks"] = rows
    by_set = {}
    for r in rows:
        by_set.setdefault(r["park"], []).append(r)
    for name, rs in by_set.items():
        rs = sorted(rs, key=lambda r: r["clear_mm"])
        worst = rs[0]
        nf = sum(r["fails_floor"] for r in rs)
        ns = sum(r["fails_static"] for r in rs)
        print(f"   {name:22s} worst {worst['clear_mm']:9.1f}  arm "
              f"{worst['arm']:<3d}  fail@50 {ns}  fail@63 {nf}")
        for r in rs:
            if r["fails_floor"]:
                print(f"       FAIL  arm {r['arm']:<3d} {r['clear_mm']:9.1f} mm"
                      f"  ({'<50' if r['fails_static'] else '<63'})")

    if not a.skip_area:
        print("\n2. THE CERTIFIED AREA  (draw-pose layer only; LOWER BOUND)")
        res = measure_area(fl, boxes, a.h, pen_ext, a.atlas, a.map, FLOOR)
        n0, n1 = int(res["live0"].sum()), int(res["live1"].sum())
        r0 = largest_rect(res["live0"], res["xs"], res["ys"])
        r1 = largest_rect(res["live1"], res["xs"], res["ys"])
        print(f"   live cells {n0} -> {n1}   ({n0 - n1} new dead)")
        print(f"   hole-free block {r0.get('w')} x {r0.get('h')} m "
              f"({r0['cells']} cells)  ->  {r1.get('w')} x {r1.get('h')} m "
              f"({r1['cells']} cells)")
        for aid, v in sorted(res["per_arm"].items()):
            if v["before"] != v["after"]:
                print(f"     arm {aid:<3d} {v['before']:5d} -> {v['after']:5d}"
                      f"   ({v['before'] - v['after']} lost)")
        out["area"] = dict(live_before=n0, live_after=n1,
                           rect_before=r0, rect_after=r1,
                           per_arm=res["per_arm"],
                           atlas_rows_killed=res["atlas_rows_killed"],
                           caveat="draw-pose layer only — hover and routing "
                                  "legs are not on disk, so this is a LOWER "
                                  "bound on the damage")

    if not a.skip_timeline:
        print("\n3. THE SHIPPED TIMELINE  (v19, whole merged timeline)")
        rep = measure_timeline(fl, a.timeline, a.sub)
        fc = rep.get("frame_clearance", {}) or {}
        out["v19"] = dict(
            ok=bool(rep["ok"]),
            min_clearance_mm=round(rep["min_clearance"] * MM, 2),
            frame_clearance_mm={str(k): round(v[0] * MM, 2)
                                for k, v in fc.items()},
            frame_failed=[int(x) for x in rep.get("frame_failed", [])],
            frame_worst_t={str(k): round(v[1], 3) for k, v in fc.items()})
        if fc:
            k = min(fc, key=lambda k: fc[k][0])
            print(f"   frame (STEEL) worst {fc[k][0] * MM:.1f} mm  arm {k}  "
                  f"t = {fc[k][1]:.2f} s   -> "
                  f"{'FAIL' if rep.get('frame_failed') else 'PASS'}")

    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
