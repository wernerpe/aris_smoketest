"""Re-decide only the DEAD cells of an existing map, against a new atlas.

WHY THIS INSTEAD OF A FULL SWEEP.  Re-sweeping the three-layer map costs hours
once the neighbour model is honest, because far more cells now reach the
ROUTING layer instead of failing fast at the hover gate — the map got expensive
precisely because the fix worked.  But when the new atlas is a strict SUPERSET
of the old one (checked here, and refused if it is not), every cell the old map
called live is still live: nothing was taken away.  Only the DEAD cells can
change, and there are a few hundred of them rather than sixteen thousand.

So this re-offers the escalation ladder to exactly those, with the same rungs
and budgets the map's own recipe uses, and merges the answers back.  The
result is the map the full sweep would have produced, restricted to the cells
that could possibly differ.

    remap_dead_cells.py --map OLD_map.npz --atlas NEW_ATLAS --h H --out DIR
                        [--parks JSON] [--rungs 0,1] [--workers 6]
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import feasible_workspace as fw  # noqa: E402
from aris_sixarm import atlas  # noqa: E402

_J = {}


def _boot(atlas_dir, h, parks_json, tries, rrt, nodes):
    if parks_json:
        fw.set_park_override({int(k): v["q"] for k, v in
                              json.load(open(parks_json))["best"].items()})
    fw._init(atlas_dir, h, True, None, True, 15.0, tries, rrt, nodes)
    _J["rows"] = {}
    for a in sorted(fw._W["fleet"]):
        arr, _ = atlas.load(atlas_dir, a)
        g = arr[atlas.strict_go(arr)]
        _J["rows"][a] = {(round(float(r[0]), 4), round(float(r[1]), 4)): r
                         for r in g}


def _one(job):
    """(arm, x, y, rungs) -> (x, y, arm, code, z, park_clear)."""
    a, x, y, rungs = job
    row = _J["rows"][a].get((x, y))
    if row is None:
        return (x, y, a, fw.NO_DRAW, 0.0, float("nan"))
    best = (fw.NO_DRAW, 0.0, float("nan"))
    for ri in rungs:
        r = fw.rung_settings(ri)
        fw._apply_rung(r)
        code, z, pc = fw._cell_escalated(a, row, atlas.QCOL, r)
        if code == fw.FEASIBLE:
            return (x, y, a, code, z, pc)
        if code > best[0] or best[0] == fw.NO_DRAW:
            best = (code, z, pc)
    return (x, y, a, best[0], best[1], best[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--atlas", required=True)
    ap.add_argument("--h", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parks", default=None)
    ap.add_argument("--old-atlas", default=None,
                    help="the atlas the OLD map was built on; when given, the "
                         "superset property is checked and a violation is "
                         "refused rather than silently merged")
    ap.add_argument("--rungs", default="0,1")
    ap.add_argument("--tries", type=int, default=48)
    ap.add_argument("--rrt", type=float, default=60.0)
    ap.add_argument("--nodes", type=int, default=300)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    rungs = tuple(int(v) for v in a.rungs.split(","))

    z = np.load(a.map, allow_pickle=True)
    cause = np.asarray(z["cause"], int).copy()
    n_arms = np.asarray(z["n_arms"], int).copy()
    xs, ys = np.asarray(z["xs"], float), np.asarray(z["ys"], float)
    masks = {int(k[4:]): np.asarray(z[k], bool).copy()
             for k in z.files if k.startswith("mask")}
    live0 = n_arms > 0

    arms = sorted(masks)
    new_rows = {}
    for arm in arms:
        arr, _ = atlas.load(a.atlas, arm)
        g = arr[atlas.strict_go(arr)]
        new_rows[arm] = {(round(float(r[0]), 4), round(float(r[1]), 4))
                         for r in g}
    if a.old_atlas:
        for arm in arms:
            arr, _ = atlas.load(a.old_atlas, arm)
            g = arr[atlas.strict_go(arr)]
            old = {(round(float(r[0]), 4), round(float(r[1]), 4)) for r in g}
            gone = old - new_rows[arm]
            if gone:
                raise SystemExit(
                    f"arm {arm}: the new atlas DROPS {len(gone)} strict-GO "
                    "cells, so the old map's live cells are not carried "
                    "forward by construction and this shortcut is invalid — "
                    "run a full sweep instead")
        print("superset check: the new atlas keeps every old strict-GO cell")

    iy, ix = np.where(~live0)
    jobs = []
    for i, j in zip(iy, ix):
        xy = (round(float(xs[j]), 4), round(float(ys[i]), 4))
        for arm in arms:
            if xy in new_rows[arm]:
                jobs.append((arm, xy[0], xy[1], rungs))
    print(f"{int((~live0).sum())} dead cells; {len(jobs)} (arm, cell) pairs "
          f"have a drawing pose in the new atlas", flush=True)

    t0 = time.time()
    ctx = mp.get_context("fork")
    got = {}
    with ctx.Pool(a.workers, initializer=_boot,
                  initargs=(a.atlas, a.h, a.parks, a.tries, a.rrt,
                            a.nodes)) as pool:
        for n, (x, y, arm, code, zh, pc) in enumerate(
                pool.imap_unordered(_one, jobs, chunksize=1), 1):
            key = (x, y)
            prev = got.get(key)
            if prev is None or code < prev[0]:
                got[key] = (code, arm, zh, pc)
            if n % 25 == 0:
                el = time.time() - t0
                print(f"  {n}/{len(jobs)} pairs  {el:.0f}s  "
                      f"eta {el / n * (len(jobs) - n):.0f}s", flush=True)

    turned = 0
    for (x, y), (code, arm, zh, pc) in got.items():
        j = int(np.argmin(np.abs(xs - x)))
        i = int(np.argmin(np.abs(ys - y)))
        cause[i, j] = code
        if code == fw.FEASIBLE:
            n_arms[i, j] = max(1, int(n_arms[i, j]))
            masks[arm][i, j] = True
            turned += 1
    live = n_arms > 0
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out) + "_map.npz", cause=cause, n_arms=n_arms,
                        xs=xs, ys=ys, draw_any=np.asarray(z["draw_any"]),
                        hover_any=np.asarray(z["hover_any"]),
                        **{f"mask{k}": v for k, v in masks.items()})
    print(f"\n{turned} cells turned in {time.time() - t0:.0f}s")
    print(f"live {live.sum()} / {live.size} = {100 * live.mean():.3f} %  "
          f"| NO_DRAW {(cause == 1).sum()} NO_HOVER {(cause == 2).sum()} "
          f"NO_ROUTE {(cause == 3).sum()}")
    print(f"wrote {out}_map.npz")


if __name__ == "__main__":
    main()
