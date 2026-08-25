#!/usr/bin/env python3
"""GREEN-FIELD layout study: place 2 floor + 4 ceiling-inverted arms.

Three stages, coarse to fine (docs/LAYOUT_STUDY.md):

  coarse   disc-cover search: every arm is its measured strict-GO annulus
           (aris_sixarm/layout.py PROFILES_LAT, lateral tool), and a
           multi-start pattern search places the six bases to maximise the
           union area on a 4 cm grid (tie-breaks: >=2-arm overlap, then
           180-degree rotational symmetry about the canvas centre —
           buildability as two identical units).  Heights {0.85, 0.922, 1.00}
           swept; FREE and SYMMETRIC parameterisations both run.
  medium   the top layouts re-scored by REAL per-arm atlas sweeps (4 cm grid,
           perpendicular pen, lateral tool, green field) and re-ranked.
  fine     the top 2 at the full 2 cm atlas with the 15-degree tilt cone —
           the same sweep the 75.93 % / 84.91 % current-rig numbers come
           from (minus the frame boxes, which do not exist here).

Outputs:  out/layout_candidates.json   every stage's ranking
          out/layout_study/            per-arm atlases of the finalists
          out/layout_study.png         coverage maps, current rig vs top 3

Run:      python3 scripts/layout_study.py            # all stages
          python3 scripts/layout_study.py --coarse-only
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from aris_sixarm import atlas, layout  # noqa: E402
from aris_sixarm.layout import (FLOOR_IDS, INV_IDS, MIN_BASE_DIST,  # noqa: E402
                                MIN_FLOOR_INV_DIST, PROFILES_LAT, build_fleet)
from aris_sixarm.rig_final6 import SHEET_FINAL6  # noqa: E402

W, H = SHEET_FINAL6
GRID_COARSE = 0.04
GRID_FINE = 0.02
HEIGHTS = (0.85, 0.922, 1.00)
MARGIN_OUT = 0.35          # how far outside the canvas a base may sit
RNG = np.random.default_rng(20260825)


# ==========================================================================
# stage 1: the disc model
# ==========================================================================
def grid_pts(grid):
    xs = np.arange(0.0, W + 1e-9, grid)
    ys = np.arange(0.0, H + 1e-9, grid)
    X, Y = np.meshgrid(xs, ys)               # (ny, nx), y-major like run_atlas6
    return xs, ys, np.column_stack([X.ravel(), Y.ravel()])


def annuli(layout_d):
    """-> [(base (2,), r0, r1)] for the six arms of a layout dict."""
    h = layout_d.get("h", 0.922)
    fr = PROFILES_LAT[("floor", None)]
    ir = PROFILES_LAT[("inv", round(float(h), 3))]
    out = [(np.asarray(p, float), fr[0], fr[1]) for p in layout_d["floor"]]
    out += [(np.asarray(p, float), ir[0], ir[1]) for p in layout_d["inv"]]
    return out


def score(layout_d, P, w_ge2=0.08):
    """Disc-model score of one layout: union + w*ge2.  -> (score, un, ge2)."""
    cnt = np.zeros(len(P), np.int8)
    for b, r0, r1 in annuli(layout_d):
        d2 = np.einsum("ij,ij->i", P - b, P - b)
        cnt += ((d2 >= r0 * r0) & (d2 <= r1 * r1))
    un = float(np.mean(cnt >= 1))
    ge2 = float(np.mean(cnt >= 2))
    return un + w_ge2 * ge2, un, ge2


def _ok(layout_d):
    if not (len(layout_d["floor"]) == 2 and len(layout_d["inv"]) == 4):
        return False
    for x, y in layout_d["floor"]:
        if 0.0 < x < W and 0.0 < y < H:      # floor bases stay OFF the web
            return False
        if not (-MARGIN_OUT <= x <= W + MARGIN_OUT
                and -MARGIN_OUT <= y <= H + MARGIN_OUT):
            return False
    for x, y in layout_d["inv"]:
        if not (-0.25 <= x <= W + 0.25 and -0.25 <= y <= H + 0.25):
            return False
    return not layout.check_spacing(layout_d)


def _rand_layout(h, symmetric):
    """A random constraint-satisfying start."""
    for _ in range(400):
        edge = RNG.integers(0, 4, size=2)
        floor = []
        for e in (edge if not symmetric else edge[:1]):
            t = RNG.uniform(0.1, 0.9)
            d = RNG.uniform(0.14, 0.32)
            floor.append({0: (t * W, -d), 1: (t * W, H + d),
                          2: (-d, t * H), 3: (W + d, t * H)}[int(e)])
        inv = [(RNG.uniform(0.2, W - 0.2), RNG.uniform(0.2, H - 0.2))
               for _ in range(4 if not symmetric else 2)]
        d = unfold(dict(floor=floor, inv=inv, h=h)) if symmetric \
            else dict(floor=floor, inv=inv, h=h)
        if _ok(d):
            return dict(floor=floor, inv=inv, h=h)
    return None


def unfold(half):
    """Symmetric mode: 1 floor + 2 inv, mirrored by a 180-degree rotation
    about the canvas centre -> the full 2 + 4 layout."""
    c = np.array([W / 2, H / 2])
    rot = lambda p: tuple((2 * c - np.asarray(p, float)))  # noqa: E731
    return dict(floor=[tuple(half["floor"][0]), rot(half["floor"][0])],
                inv=[tuple(p) for p in half["inv"]]
                + [rot(p) for p in half["inv"]],
                h=half["h"])


def _search_one(h, symmetric, P, iters=4):
    half = _rand_layout(h, symmetric)
    if half is None:
        return None
    full = unfold(half) if symmetric else half
    best = score(full, P)[0]
    for step in (0.24, 0.12, 0.06, 0.03):
        improved = True
        rounds = 0
        while improved and rounds < iters:
            improved = False
            rounds += 1
            groups = (("floor", len(half["floor"])), ("inv", len(half["inv"])))
            for kind, n in groups:
                for i in range(n):
                    for dx, dy in ((step, 0), (-step, 0), (0, step), (0, -step)):
                        cand = dict(floor=[list(p) for p in half["floor"]],
                                    inv=[list(p) for p in half["inv"]], h=h)
                        cand[kind][i][0] += dx
                        cand[kind][i][1] += dy
                        cf = unfold(cand) if symmetric else cand
                        if not _ok(cf):
                            continue
                        s = score(cf, P)[0]
                        if s > best + 1e-9:
                            best, half, full = s, cand, cf
                            improved = True
    s, un, ge2 = score(full, P)
    return dict(layout=dict(floor=[tuple(p) for p in full["floor"]],
                            inv=[tuple(p) for p in full["inv"]], h=h),
                score=s, union=un, ge2=ge2, symmetric=bool(symmetric))


def stage_coarse(restarts=40):
    _, _, P = grid_pts(GRID_COARSE)
    t0 = time.time()
    cands, n_evals = [], 0
    for h in HEIGHTS:
        for symmetric in (True, False):
            for _ in range(restarts):
                r = _search_one(h, symmetric, P)
                if r is not None:
                    cands.append(r)
    # dedupe near-identical layouts (sorted base sets within 6 cm)
    uniq = []
    for c in sorted(cands, key=lambda c: -c["score"]):
        key = np.sort(np.round(np.array(c["layout"]["floor"]
                                        + c["layout"]["inv"]) / 0.06), axis=0)
        if not any(np.array_equal(key, u[0]) and c["layout"]["h"] == u[1]
                   for u in [(k, hh) for k, hh, _ in uniq]):
            uniq.append((key, c["layout"]["h"], c))
    top = [c for _, _, c in uniq]
    print(f"coarse: {len(cands)} local optima ({time.time() - t0:.0f} s), "
          f"{len(top)} distinct; best per height:")
    for h in HEIGHTS:
        hh = [c for c in top if c["layout"]["h"] == h]
        if hh:
            b = hh[0]
            print(f"  h={h:.3f}: union {100 * b['union']:.2f} % "
                  f"ge2 {100 * b['ge2']:.2f} % sym={b['symmetric']}")
    return top


# ==========================================================================
# stages 2/3: real sweeps
# ==========================================================================
def _sweep(job):
    aid, spec_args, h, grid, tilt, out_dir = job
    fl = build_fleet(spec_args)
    arr = atlas.sweep_arm(aid, out_dir, grid=grid, rmax=1.05, h_inv=h,
                          tilt_max_deg=tilt, fleet=fl, sheet=SHEET_FINAL6,
                          pen_lat=0.110)
    return aid, arr


def masks(arr, xs, ys, grid):
    reach = np.zeros((len(ys), len(xs)), bool)
    go = np.zeros_like(reach)
    if len(arr):
        ix = np.rint(arr[:, 0] / grid).astype(int)
        iy = np.rint(arr[:, 1] / grid).astype(int)
        keep = (ix >= 0) & (ix < len(xs)) & (iy >= 0) & (iy < len(ys))
        reach[iy[keep], ix[keep]] = True
        g = atlas.strict_go(arr) & keep
        go[iy[g], ix[g]] = True
    return reach, go


def sweep_layout(layout_d, grid, tilt, out_dir, jobs=6):
    xs, ys = (np.arange(0.0, W + 1e-9, grid), np.arange(0.0, H + 1e-9, grid))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    jobs_list = [(aid, layout_d, layout_d.get("h", 0.922), grid, tilt,
                  str(out_dir)) for aid in list(FLOOR_IDS) + list(INV_IDS)]
    with mp.get_context("fork").Pool(min(jobs, len(jobs_list))) as pool:
        res = pool.map(_sweep, jobs_list)
    per_go, per_reach, arms = [], [], []
    for aid, arr in res:
        r, g = masks(arr, xs, ys, grid)
        arms.append(aid)
        per_reach.append(r)
        per_go.append(g)
    per_go = np.array(per_go)
    cnt = per_go.sum(axis=0)
    return dict(arms=arms, xs=xs, ys=ys, per_go=per_go,
                per_reach=np.array(per_reach), cnt=cnt,
                union=float(np.mean(cnt >= 1)), ge2=float(np.mean(cnt >= 2)),
                ge3=float(np.mean(cnt >= 3)),
                per_arm={a: float(g.mean()) for a, g in zip(arms, per_go)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restarts", type=int, default=40)
    ap.add_argument("--medium", type=int, default=5)
    ap.add_argument("--fine", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--coarse-only", action="store_true")
    a = ap.parse_args()

    out_json = ROOT / "out" / "layout_candidates.json"
    record = dict(sheet=[W, H], grid_coarse=GRID_COARSE, heights=HEIGHTS,
                  profiles={f"{k[0]}@{k[1]}": v
                            for k, v in PROFILES_LAT.items()},
                  tool="lateral (pen_lat 0.110)")

    top = stage_coarse(a.restarts)
    record["coarse"] = [dict(c, layout=c["layout"]) for c in top[:20]]
    out_json.write_text(json.dumps(record, indent=1, default=float))
    if a.coarse_only:
        print(f"wrote {out_json}")
        return

    # ---- medium: real 4 cm sweeps of the top layouts ----------------------
    # the pool guarantees the best SYMMETRIC candidates a seat: symmetry is
    # the buildability tie-break and the medium stage is what should judge it
    pool = list(top[:max(1, a.medium - 2)])
    for c in top:
        if c["symmetric"] and c not in pool:
            pool.append(c)
        if len(pool) >= a.medium:
            break
    med = []
    print(f"\nmedium: real 4 cm atlases for the top {len(pool)}:")
    for k, c in enumerate(pool):
        d = ROOT / "out" / "layout_study" / f"med_{k}"
        t0 = time.time()
        r = sweep_layout(c["layout"], GRID_COARSE, 0.0, d, jobs=a.jobs)
        med.append(dict(idx=k, layout=c["layout"], sym=c["symmetric"],
                        disc_union=c["union"], union=r["union"],
                        ge2=r["ge2"], ge3=r["ge3"], per_arm=r["per_arm"]))
        print(f"  #{k} h={c['layout']['h']:.3f} disc {100 * c['union']:.2f} % "
              f"-> real {100 * r['union']:.2f} % (ge2 {100 * r['ge2']:.2f} %) "
              f"[{time.time() - t0:.0f} s]")
    med.sort(key=lambda m: -(m["union"] + 0.08 * m["ge2"]))
    record["medium"] = med
    out_json.write_text(json.dumps(record, indent=1, default=float))

    # ---- fine: 2 cm + 15-degree tilt cone ---------------------------------
    # pool: the top `a.fine` overall, plus the best h = 0.922 layout and the
    # best SYMMETRIC one if they are not already in — so the height call and
    # the buildability call are both made at full resolution.
    fpool = list(med[:a.fine])
    for pick in ([m for m in med if abs(m["layout"]["h"] - 0.922) < 1e-6],
                 [m for m in med if m["sym"]]):
        if pick and pick[0] not in fpool:
            fpool.append(pick[0])
    fine = []
    print(f"\nfine: 2 cm certified atlases (tilt <= 15 deg) for "
          f"{len(fpool)} finalists:")
    for k, m in enumerate(fpool):
        d = ROOT / "out" / "layout_study" / f"fine_{k}"
        t0 = time.time()
        r = sweep_layout(m["layout"], GRID_FINE, 15.0, d, jobs=a.jobs)
        np.savez_compressed(Path(d) / "coverage.npz", xs=r["xs"], ys=r["ys"],
                            per_arm_go=r["per_go"],
                            per_arm_reach=r["per_reach"],
                            union_go=r["cnt"] >= 1, count_go=r["cnt"],
                            arms=np.array(r["arms"], np.int64),
                            rig=np.array("layout_study"))
        fine.append(dict(idx=m["idx"], layout=m["layout"], sym=m["sym"],
                         union=r["union"], ge2=r["ge2"], ge3=r["ge3"],
                         per_arm=r["per_arm"], dir=str(d)))
        print(f"  #{m['idx']} h={m['layout']['h']:.3f} "
              f"union {100 * r['union']:.2f} % ge2 {100 * r['ge2']:.2f} % "
              f"ge3 {100 * r['ge3']:.2f} % [{time.time() - t0:.0f} s]")
    fine.sort(key=lambda f: -(f["union"] + 0.08 * f["ge2"]))
    record["fine"] = fine
    out_json.write_text(json.dumps(record, indent=1, default=float))
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
