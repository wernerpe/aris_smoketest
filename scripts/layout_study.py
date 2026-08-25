#!/usr/bin/env python3
"""LAYOUT STUDY v2 — place six arms WITH THEIR MOUNTING HARDWARE.

v1 (commit 83ad415) modelled NO steel: neighbouring booms and base plates
were not obstacles, so its 99.38 % winner was an optimistic kinematic
ceiling.  v2 makes the schematic mounts (`aris_sixarm/mounts.py`) first-class
static obstacles at every stage, and studies TWO FAMILIES:

    2 + 4    two floor arms + four ceiling-inverted arms
    0 + 6    all six ceiling-inverted — uniform hardware, one grid

Three stages, coarse to fine (docs/LAYOUT_STUDY.md):

  coarse   disc-cover search with an OBSTACLE-AWARE proxy: every arm is its
           measured strict-GO annulus (layout.PROFILES_LAT, lateral tool)
           MINUS the shadow of every other arm's column (a wedge/capsule
           subtraction along the base->cell ray) and MINUS the footprint
           keep-out of every other arm's pedestal.  Which correction a piece
           of hardware earns is decided by z bands, not by taste — see
           `mounts.keepouts`.  Multi-start pattern search on a 4 cm grid,
           heights swept, FREE and SYMMETRIC parameterisations both run.
  medium   the top layouts re-scored by REAL per-arm atlas sweeps (4 cm grid,
           perpendicular pen, lateral tool) WITH the mount obstacle boxes
           active in the lattice clearance check.
  fine     the finalists at the full 2 cm atlas with the 15-degree tilt cone,
           mounts active — plus the certified ready-pose check against every
           other arm's hardware.

Extra stages this driver also runs (`--stages`):
  v1       v1's winner re-scored with and without mounts (how optimistic?)
  spacing  coverage vs inverted-pair spacing around the optimum
  profiles re-measure the radial GO annuli for a height

Outputs:  out/layout_candidates.json   every stage's ranking
          out/layout_study/            per-arm atlases of the finalists
          out/layout_study.png         coverage maps

Run:      ARIS_TOOL=lateral python3 scripts/layout_study.py --restarts 60 \
              --medium 8 --fine 4 --jobs 12
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

from aris_sixarm import atlas, layout, mounts  # noqa: E402
from aris_sixarm.layout import (MIN_BASE_DIST, MIN_FLOOR_INV_DIST,  # noqa: E402
                                PROFILES_LAT, Z_FLOOR, arm_ids, build_fleet)
from aris_sixarm.rig_final6 import SHEET_FINAL6  # noqa: E402

W, H = SHEET_FINAL6
GRID_COARSE = 0.04
GRID_FINE = 0.02
HEIGHTS = (0.85, 0.922, 1.00)
FAMILIES = (2, 0)          # number of FLOOR arms; the rest hang
MARGIN_OUT = 0.35          # how far outside the canvas a floor base may sit
PEN_LAT = 0.110
RNG = np.random.default_rng(20260825)


# ==========================================================================
# stage 1: the OBSTACLE-AWARE disc model
# ==========================================================================
def grid_pts(grid):
    xs = np.arange(0.0, W + 1e-9, grid)
    ys = np.arange(0.0, H + 1e-9, grid)
    X, Y = np.meshgrid(xs, ys)               # (ny, nx), y-major like run_atlas6
    return xs, ys, np.column_stack([X.ravel(), Y.ravel()])


def arms_of(layout_d):
    """-> [(mount, base (2,), (r0, r1))] for the six arms of a layout dict."""
    h = round(float(layout_d.get("h", 0.922)), 3)
    fr, ir = PROFILES_LAT[("floor", None)], PROFILES_LAT[("inv", h)]
    return [("floor", np.asarray(p, float), fr) for p in layout_d["floor"]] \
        + [("inv", np.asarray(p, float), ir) for p in layout_d["inv"]]


def arm_masks(layout_d, P, m=mounts.MOUNTS, with_mounts=True):
    """Per-arm coarse coverage masks -> (n_arms, N) bool.

    The annulus is the measured strict-GO profile; the corrections are the
    mount hardware of every OTHER arm (own mount excluded by construction).
    """
    h = float(layout_d.get("h", 0.922))
    arms = arms_of(layout_d)
    out = np.empty((len(arms), len(P)), bool)
    for i, (mnt, b, (r0, r1)) in enumerate(arms):
        d2 = np.einsum("ij,ij->i", P - b, P - b)
        mask = (d2 >= r0 * r0) & (d2 <= r1 * r1)
        if with_mounts:
            others = [(mo, bo) for j, (mo, bo, _) in enumerate(arms) if j != i]
            sh, bl = mounts.keepouts(mnt, b, Z_FLOOR if mnt == "floor" else h,
                                     others, h, m)
            mask = mounts.apply_keepouts(mask, P, b, sh, bl)
        out[i] = mask
    return out


def score(layout_d, P, w_ge2=0.08, with_mounts=True):
    """Disc-model score of one layout: union + w*ge2.  -> (score, un, ge2)."""
    cnt = arm_masks(layout_d, P, with_mounts=with_mounts).sum(axis=0)
    un = float(np.mean(cnt >= 1))
    ge2 = float(np.mean(cnt >= 2))
    return un + w_ge2 * ge2, un, ge2


def _ok(layout_d, n_floor):
    if not (len(layout_d["floor"]) == n_floor
            and len(layout_d["inv"]) == 6 - n_floor):
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


def _rand_layout(h, symmetric, n_floor):
    """A random constraint-satisfying start (half a layout if symmetric)."""
    n_inv = 6 - n_floor
    for _ in range(600):
        floor = []
        for _ in range(n_floor if not symmetric else n_floor // 2):
            e = int(RNG.integers(0, 4))
            t, d = RNG.uniform(0.1, 0.9), RNG.uniform(0.14, 0.32)
            floor.append({0: (t * W, -d), 1: (t * W, H + d),
                          2: (-d, t * H), 3: (W + d, t * H)}[e])
        inv = [(RNG.uniform(0.2, W - 0.2), RNG.uniform(0.2, H - 0.2))
               for _ in range(n_inv if not symmetric else n_inv // 2)]
        half = dict(floor=floor, inv=inv, h=h)
        if _ok(unfold(half) if symmetric else half, n_floor):
            return half
    return None


def unfold(half):
    """Symmetric mode: half the arms, mirrored by a 180-degree rotation about
    the canvas centre -> the full six."""
    c = np.array([W / 2, H / 2])
    rot = lambda p: tuple((2 * c - np.asarray(p, float)))  # noqa: E731
    return dict(floor=[tuple(p) for p in half["floor"]]
                + [rot(p) for p in half["floor"]],
                inv=[tuple(p) for p in half["inv"]]
                + [rot(p) for p in half["inv"]],
                h=half["h"])


def _search_one(h, symmetric, n_floor, P, iters=4):
    half = _rand_layout(h, symmetric, n_floor)
    if half is None:
        return None
    full = unfold(half) if symmetric else half
    best = score(full, P)[0]
    for step in (0.24, 0.12, 0.06, 0.03):
        improved, rounds = True, 0
        while improved and rounds < iters:
            improved = False
            rounds += 1
            for kind in ("floor", "inv"):
                for i in range(len(half[kind])):
                    for dx, dy in ((step, 0), (-step, 0), (0, step),
                                   (0, -step)):
                        cand = dict(floor=[list(p) for p in half["floor"]],
                                    inv=[list(p) for p in half["inv"]], h=h)
                        cand[kind][i][0] += dx
                        cand[kind][i][1] += dy
                        cf = unfold(cand) if symmetric else cand
                        if not _ok(cf, n_floor):
                            continue
                        s = score(cf, P)[0]
                        if s > best + 1e-9:
                            best, half, full = s, cand, cf
                            improved = True
    s, un, ge2 = score(full, P)
    return dict(layout=dict(floor=[tuple(p) for p in full["floor"]],
                            inv=[tuple(p) for p in full["inv"]], h=h),
                score=s, union=un, ge2=ge2, symmetric=bool(symmetric),
                n_floor=n_floor)


def _coarse_job(args):
    h, symmetric, n_floor, seed, restarts = args
    global RNG
    RNG = np.random.default_rng(seed)
    _, _, P = grid_pts(GRID_COARSE)
    out = []
    for _ in range(restarts):
        r = _search_one(h, symmetric, n_floor, P)
        if r is not None:
            out.append(r)
    return out


def stage_coarse(restarts=40, jobs=6, heights=HEIGHTS, families=FAMILIES):
    t0 = time.time()
    jobs_list = [(h, sym, nf, 20260825 + 7919 * k, restarts)
                 for k, (h, sym, nf) in enumerate(
                     [(h, s, n) for h in heights for s in (True, False)
                      for n in families])]
    with mp.get_context("fork").Pool(min(jobs, len(jobs_list))) as pool:
        cands = [c for chunk in pool.map(_coarse_job, jobs_list) for c in chunk]
    uniq, keys = [], []
    for c in sorted(cands, key=lambda c: -c["score"]):
        key = np.sort(np.round(np.array(c["layout"]["floor"]
                                        + c["layout"]["inv"]) / 0.06), axis=0)
        sig = (c["layout"]["h"], c["n_floor"])
        if not any(np.array_equal(key, k) and sig == s for k, s in keys):
            keys.append((key, sig))
            uniq.append(c)
    # STRATIFY: one family scoring 2 pp better than the other would otherwise
    # take every seat at the medium stage and the family comparison the user
    # asked for would never be measured at full resolution.  Keep the best
    # `per_bucket` of every (family, height, symmetry) cell, then sort.
    per_bucket, buckets = 6, {}
    for c in uniq:
        b = (c["n_floor"], c["layout"]["h"], c["symmetric"])
        buckets.setdefault(b, []).append(c)
    strat = [c for b in buckets.values() for c in b[:per_bucket]]
    strat.sort(key=lambda c: -c["score"])
    print(f"coarse: {len(cands)} local optima ({time.time() - t0:.0f} s), "
          f"{len(uniq)} distinct, {len(strat)} stratified; "
          f"best per (family, height):")
    for nf in families:
        for h in heights:
            hh = [c for c in strat
                  if c["layout"]["h"] == h and c["n_floor"] == nf]
            if hh:
                b = hh[0]
                print(f"  {nf}+{6 - nf} h={h:.3f}: union {100 * b['union']:.2f}"
                      f" % ge2 {100 * b['ge2']:.2f} % sym={b['symmetric']}")
    return strat


# ==========================================================================
# stages 2/3: real sweeps, mounts active
# ==========================================================================
def _sweep(job):
    aid, spec_args, h, grid, tilt, out_dir, with_mounts = job
    fl = build_fleet(spec_args, with_mounts=with_mounts)
    arr = atlas.sweep_arm(aid, out_dir, grid=grid, rmax=1.05, h_inv=h,
                          tilt_max_deg=tilt, fleet=fl, sheet=SHEET_FINAL6,
                          pen_lat=PEN_LAT)
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


def sweep_layout(layout_d, grid, tilt, out_dir, jobs=6, with_mounts=True):
    xs, ys = (np.arange(0.0, W + 1e-9, grid), np.arange(0.0, H + 1e-9, grid))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fids, iids = arm_ids(layout_d)
    jobs_list = [(aid, layout_d, layout_d.get("h", 0.922), grid, tilt,
                  str(out_dir), with_mounts) for aid in list(fids) + list(iids)]
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


def save_fine(r, d, layout_d):
    np.savez_compressed(Path(d) / "coverage.npz", xs=r["xs"], ys=r["ys"],
                        per_arm_go=r["per_go"], per_arm_reach=r["per_reach"],
                        union_go=r["cnt"] >= 1, count_go=r["cnt"],
                        arms=np.array(r["arms"], np.int64),
                        rig=np.array("layout_study_v2"),
                        floor=np.array(layout_d["floor"], float).reshape(-1, 2),
                        inv=np.array(layout_d["inv"], float),
                        h=float(layout_d["h"]))


# ==========================================================================
# the extra stages
# ==========================================================================
def ready_pose_check(layout_d, m=mounts.MOUNTS):
    """Every arm's CERTIFIED ready pose vs EVERY OTHER arm's mount hardware.

    The pose is `layout.certified_ready_pose` — the same one the scene draws,
    gated by `validate.check_pose` (joint margins, chain above the paper, pen
    tip above the paper).  This adds the check v1 could not make: the chain
    (frames.fk's 9 points plus both lateral-tool points, exactly what
    `atlas.solve_cell` clearance-checks) against the neighbours' booms,
    plates and pedestals, at `rig_final.STATIC_MARGIN`.
    """
    from aris_sixarm import frames, rig_final
    fl = build_fleet(layout_d)
    h = float(layout_d["h"])
    out = []
    for aid, spec in sorted(fl.items()):
        Twb = spec.T_world_base(h)
        try:
            q, xy, rep = layout.certified_ready_pose(spec, h,
                                                     pen_lat=PEN_LAT)
        except RuntimeError as e:
            out.append(dict(arm=int(aid), mount=spec.mount, ok=False,
                            reason=str(e)))
            continue
        T, pts = frames.fk(q)
        tool = frames.tool_points_many(T[None], pen_lat=PEN_LAT)
        P = np.vstack([pts] + [t for t in tool])
        Pw = (Twb[:3, :3] @ P.T).T + Twb[:3, 3]
        cl = float(rig_final.chain_static_clearance(
            Pw, spec.static_obstacles())[0])
        worst = rep["worst"]
        out.append(dict(arm=int(aid), mount=spec.mount,
                        hover=[float(xy[0]), float(xy[1])],
                        clearance=cl,
                        margin=float(worst["joint_margin"]),
                        tip_z=float(worst["tip_z"]),
                        min_chain_z=float(worst["min_chain_z"]),
                        ok=bool(cl >= rig_final.STATIC_MARGIN and rep["ok"])))
    return out


def spacing_sweep(layout_d, spacings, grid, tilt, out_root, jobs=6):
    """Coverage vs INVERTED-PAIR spacing: each transverse pair is pushed
    symmetrically apart about its own midpoint, everything else held."""
    inv = np.array(layout_d["inv"], float)
    base_pairs = _pairs(inv)
    rows = []
    for s in spacings:
        new = inv.copy()
        for (i, j) in base_pairs:
            mid = 0.5 * (inv[i] + inv[j])
            d = inv[j] - inv[i]
            n = d / max(np.linalg.norm(d), 1e-9)
            new[i], new[j] = mid - 0.5 * s * n, mid + 0.5 * s * n
        cand = dict(layout_d, inv=[tuple(p) for p in new])
        bad = layout.check_spacing(cand)
        d = Path(out_root) / f"sp_{int(round(1000 * s))}"
        r = sweep_layout(cand, grid, tilt, d, jobs=jobs)
        rows.append(dict(spacing=float(s), union=r["union"], ge2=r["ge2"],
                         ge3=r["ge3"], violations=bad,
                         layout=dict(floor=[tuple(p) for p in cand["floor"]],
                                     inv=[tuple(p) for p in new],
                                     h=cand["h"])))
        print(f"  pair spacing {s:.2f} m -> union {100 * r['union']:.2f} % "
              f"ge2 {100 * r['ge2']:.2f} %"
              + ("  [VIOLATES: " + "; ".join(bad) + "]" if bad else ""))
    return rows


def _pairs(inv):
    """Group inverted bases into transverse pairs by nearest neighbour."""
    left, out = list(range(len(inv))), []
    while len(left) >= 2:
        i = left.pop(0)
        d = [(float(np.linalg.norm(inv[i] - inv[j])), j) for j in left]
        _, j = min(d)
        left.remove(j)
        out.append((i, j))
    return out


def dead_zones(cnt, xs, ys):
    """Connected components of the uncovered cells -> [(size, bbox)]."""
    dead = cnt < 1
    seen = np.zeros_like(dead)
    comps = []
    ny, nx = dead.shape
    for iy in range(ny):
        for ix in range(nx):
            if not dead[iy, ix] or seen[iy, ix]:
                continue
            stack, cells = [(iy, ix)], []
            seen[iy, ix] = True
            while stack:
                y0, x0 = stack.pop()
                cells.append((y0, x0))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    y1, x1 = y0 + dy, x0 + dx
                    if 0 <= y1 < ny and 0 <= x1 < nx and dead[y1, x1] \
                            and not seen[y1, x1]:
                        seen[y1, x1] = True
                        stack.append((y1, x1))
            ys_, xs_ = zip(*cells)
            comps.append(dict(cells=len(cells),
                              x=[float(xs[min(xs_)]), float(xs[max(xs_)])],
                              y=[float(ys[min(ys_)]), float(ys[max(ys_)])]))
    comps.sort(key=lambda c: -c["cells"])
    return comps


def measure_profile(mount, h, grid=0.02, bearings=5):
    """Re-measure a radial strict-GO annulus (module docstring of layout.py).

    Rays from the base at `bearings` bearings, `grid` step, perpendicular pen,
    lateral tool, NO mount boxes (a profile is a single-arm property).
    """
    c = np.array([W / 2, H / 2])
    lay = (dict(floor=[(c[0], -0.20), (c[0] + 0.60, -0.20)],
                inv=[(c[0], y) for y in (0.9, 1.6, 2.3, 3.0)], h=h)
           if mount == "floor" else
           dict(floor=[], inv=[(c[0], 0.5 + 0.6 * k) for k in range(6)], h=h))
    fl = build_fleet(lay, with_mounts=False)
    aid = next(a for a, s in fl.items() if s.mount == mount)
    spec = fl[aid]
    Twb = spec.T_world_base(h)
    Twb_inv = np.linalg.inv(Twb)
    cands = atlas._candidates(0.0)
    bx, by = spec.xy
    rs = np.arange(0.10, 1.06, grid)
    ok = np.zeros(len(rs), bool)
    for k, r in enumerate(rs):
        hits = 0
        for th in np.linspace(0, 2 * np.pi, bearings, endpoint=False):
            x, y = bx + r * np.cos(th), by + r * np.sin(th)
            res = atlas.solve_cell(x, y, Twb, Twb_inv, spec, cands,
                                   pen_lat=PEN_LAT)
            if res is not None and res[0] >= 0.30 and res[1] >= 0.14:
                hits += 1
        ok[k] = hits == bearings
    idx = np.nonzero(ok)[0]
    if not len(idx):
        return None
    # longest contiguous run
    runs, start = [], idx[0]
    for a, b in zip(idx, idx[1:]):
        if b != a + 1:
            runs.append((start, a))
            start = b
    runs.append((start, idx[-1]))
    i0, i1 = max(runs, key=lambda t: t[1] - t[0])
    return float(rs[i0]), float(rs[i1])


# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restarts", type=int, default=40)
    ap.add_argument("--medium", type=int, default=6)
    ap.add_argument("--fine", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--stages", default="coarse,medium,fine,v1,ready,spacing")
    ap.add_argument("--heights", default=",".join(str(h) for h in HEIGHTS))
    a = ap.parse_args()
    stages = set(a.stages.split(","))
    heights = tuple(float(x) for x in a.heights.split(","))

    out_json = ROOT / "out" / "layout_candidates.json"
    record = json.loads(out_json.read_text()) if out_json.exists() else {}
    record.update(dict(
        version=2, sheet=[W, H], grid_coarse=GRID_COARSE, heights=list(heights),
        families=[f"{n}+{6 - n}" for n in FAMILIES],
        profiles={f"{k[0]}@{k[1]}": v for k, v in PROFILES_LAT.items()},
        tool=f"lateral (pen_lat {PEN_LAT})",
        mounts=dict(boom_r=mounts.MOUNTS.boom_r,
                    ceiling_z=mounts.MOUNTS.ceiling_z,
                    plate_xy=list(mounts.MOUNTS.plate_xy),
                    plate_t=mounts.MOUNTS.plate_t,
                    ped_xy=list(mounts.MOUNTS.ped_xy),
                    ped_drop=mounts.MOUNTS.ped_drop,
                    static_margin=mounts.MOUNTS.margin,
                    min_column_spacing=mounts.min_column_spacing(),
                    min_pedestal_spacing=mounts.min_pedestal_spacing()),
        constraints=dict(min_base_dist=MIN_BASE_DIST,
                         min_floor_inv_dist=MIN_FLOOR_INV_DIST)))

    def dump():
        out_json.write_text(json.dumps(record, indent=1, default=float))

    # ---- v1's winner, re-scored with and without mounts -------------------
    if "v1" in stages:
        print("v1 winner re-scored (real 2 cm atlas, tilt <= 15):")
        v1 = {}
        for tag, wm in (("green_field", False), ("with_mounts", True)):
            d = ROOT / "out" / "layout_study" / f"v1_{tag}"
            r = sweep_layout(layout.LAYOUT_V1, GRID_FINE, 15.0, d,
                             jobs=a.jobs, with_mounts=wm)
            save_fine(r, d, layout.LAYOUT_V1)
            v1[tag] = dict(union=r["union"], ge2=r["ge2"], ge3=r["ge3"],
                           per_arm=r["per_arm"], dir=str(d),
                           dead=dead_zones(r["cnt"], r["xs"], r["ys"])[:12])
            print(f"  {tag:12s} union {100 * r['union']:.2f} % "
                  f"ge2 {100 * r['ge2']:.2f} % ge3 {100 * r['ge3']:.2f} %")
        v1["cost_pp"] = 100 * (v1["green_field"]["union"]
                               - v1["with_mounts"]["union"])
        v1["layout"] = layout.LAYOUT_V1
        record["v1_rescore"] = v1
        dump()

    # ---- coarse -----------------------------------------------------------
    if "coarse" in stages:
        top = stage_coarse(a.restarts, a.jobs, heights)
        record["coarse"] = top
        dump()
    else:
        top = record.get("coarse", [])

    # ---- medium: real 4 cm sweeps, mounts active --------------------------
    if "medium" in stages:
        # RESERVED SEATS: the best of each family, and the best SYMMETRIC of
        # each family (symmetry is the buildability tie-break).  Without them
        # the stronger family takes every seat and the family comparison the
        # user asked for is never measured at full resolution.
        pool, seen = [], set()

        def seat(pred):
            for c in top:
                if id(c) not in seen and pred(c):
                    pool.append(c)
                    seen.add(id(c))
                    return
        for nf in FAMILIES:
            seat(lambda c, nf=nf: c["n_floor"] == nf)
            seat(lambda c, nf=nf: c["n_floor"] == nf and c["symmetric"])
        for c in top:
            if len(pool) >= a.medium:
                break
            if id(c) not in seen:
                pool.append(c)
                seen.add(id(c))
        med = []
        print(f"\nmedium: real 4 cm atlases (mounts ACTIVE) for {len(pool)}:")
        for k, c in enumerate(pool):
            d = ROOT / "out" / "layout_study" / f"med_{k}"
            t0 = time.time()
            r = sweep_layout(c["layout"], GRID_COARSE, 0.0, d, jobs=a.jobs)
            med.append(dict(idx=k, layout=c["layout"], sym=c["symmetric"],
                            n_floor=c["n_floor"], disc_union=c["union"],
                            union=r["union"], ge2=r["ge2"], ge3=r["ge3"],
                            per_arm=r["per_arm"]))
            print(f"  #{k} {c['n_floor']}+{6 - c['n_floor']} "
                  f"h={c['layout']['h']:.3f} disc {100 * c['union']:.2f} % "
                  f"-> real {100 * r['union']:.2f} % "
                  f"(ge2 {100 * r['ge2']:.2f} %) [{time.time() - t0:.0f} s]")
        med.sort(key=lambda m: -(m["union"] + 0.08 * m["ge2"]))
        record["medium"] = med
        dump()
    else:
        med = record.get("medium", [])

    # ---- fine: 2 cm + 15-degree tilt cone, mounts active ------------------
    if "fine" in stages:
        fpool, fseen = [], set()

        def fseat(pred):
            for m in med:
                if m["idx"] not in fseen and pred(m):
                    fpool.append(m)
                    fseen.add(m["idx"])
                    return
        for nf in FAMILIES:                     # reserved seats, as at medium
            fseat(lambda m, nf=nf: m["n_floor"] == nf)
            fseat(lambda m, nf=nf: m["n_floor"] == nf and m["sym"])
        for m in med:
            if len(fpool) >= a.fine:
                break
            if m["idx"] not in fseen:
                fpool.append(m)
                fseen.add(m["idx"])
        fine = []
        print(f"\nfine: 2 cm certified atlases (tilt <= 15 deg, mounts "
              f"ACTIVE) for {len(fpool)} finalists:")
        for m in fpool:
            d = ROOT / "out" / "layout_study" / f"fine_{m['idx']}"
            t0 = time.time()
            r = sweep_layout(m["layout"], GRID_FINE, 15.0, d, jobs=a.jobs)
            save_fine(r, d, m["layout"])
            fine.append(dict(idx=m["idx"], layout=m["layout"], sym=m["sym"],
                             n_floor=m["n_floor"], union=r["union"],
                             ge2=r["ge2"], ge3=r["ge3"],
                             per_arm=r["per_arm"], dir=str(d),
                             dead=dead_zones(r["cnt"], r["xs"], r["ys"])[:12],
                             ready=ready_pose_check(m["layout"])))
            print(f"  #{m['idx']} {m['n_floor']}+{6 - m['n_floor']} "
                  f"h={m['layout']['h']:.3f} union {100 * r['union']:.2f} % "
                  f"ge2 {100 * r['ge2']:.2f} % ge3 {100 * r['ge3']:.2f} % "
                  f"[{time.time() - t0:.0f} s]")
        fine.sort(key=lambda f: -(f["union"] + 0.08 * f["ge2"]))
        record["fine"] = fine
        dump()
    else:
        fine = record.get("fine", [])

    # ---- the all-ceiling optimum in ROUND NUMBERS -------------------------
    # Every 0 + 6 restart converges on the same regular figure, so the build
    # should not have to copy six survey numbers off a search.  This scores
    # `layout.paired_grid` — the same figure set out from the canvas
    # dimensions — at full resolution, and the study recommends it only if it
    # matches the search winner.
    if "grid" in stages:
        print(f"\ngrid: the all-ceiling optimum in round numbers "
              f"(paired_grid, spacing {layout.PAIR_SPACING} m):")
        rows = []
        for h in heights:
            lay = layout.paired_grid(h=h)
            d = ROOT / "out" / "layout_study" / f"grid_h{int(round(1000 * h))}"
            r = sweep_layout(lay, GRID_FINE, 15.0, d, jobs=a.jobs)
            save_fine(r, d, lay)
            rows.append(dict(h=h, layout=lay, union=r["union"], ge2=r["ge2"],
                             ge3=r["ge3"], per_arm=r["per_arm"], dir=str(d),
                             violations=layout.check_spacing(lay),
                             dead=dead_zones(r["cnt"], r["xs"], r["ys"])[:12],
                             ready=ready_pose_check(lay)))
            print(f"  h={h:.3f} union {100 * r['union']:.2f} % "
                  f"ge2 {100 * r['ge2']:.2f} % ge3 {100 * r['ge3']:.2f} % "
                  f"ready {sum(x['ok'] for x in rows[-1]['ready'])}/6 clear")
        record["grid"] = rows
        dump()

    # ---- ready poses ------------------------------------------------------
    if "ready" in stages and fine:
        rows = ready_pose_check(fine[0]["layout"])
        record["ready_poses"] = rows
        print("\nready poses vs every other arm's mount hardware:")
        for r in rows:
            print(f"  arm {r['arm']:3d} {r['mount']:5s} mount clearance "
                  f"{r.get('clearance', float('nan')):+.3f} m  tip z "
                  f"{1000 * r.get('tip_z', 0):+.0f} mm  "
                  f"{'OK' if r['ok'] else 'VIOLATION'}")
        dump()

    # ---- spacing sensitivity ---------------------------------------------
    if "spacing" in stages and (fine or record.get("grid")):
        # sweep around whatever the study RECOMMENDS: the round-number grid
        # when it holds up, else the search winner
        gr = sorted(record.get("grid", []),
                    key=lambda g: -(g["union"] + 0.08 * g["ge2"]))
        win = gr[0]["layout"] if gr else fine[0]["layout"]
        inv = np.array(win["inv"], float)
        cur = float(np.mean([np.linalg.norm(inv[i] - inv[j])
                             for i, j in _pairs(inv)]))
        # the interesting range is the pair window (layout.PAIR_WINDOW) plus
        # enough either side to see both walls: below it a partner stops
        # covering the under-base hole, above it the pair pulls apart and the
        # hole opens.  Spacings the constraints forbid are swept anyway and
        # reported as violations — the user asked what the trade costs.
        sp = sorted({round(x, 3) for x in
                     list(np.arange(0.40, 0.96, 0.05)) + [cur]})
        print(f"\nspacing sensitivity around the optimum (current pair "
              f"spacing {cur:.3f} m):")
        record["spacing"] = dict(
            current=cur,
            rows=spacing_sweep(win, sp, GRID_FINE, 15.0,
                               ROOT / "out" / "layout_study" / "spacing",
                               jobs=a.jobs))
        dump()

    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
