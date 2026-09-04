#!/usr/bin/env python3
"""WHAT THE MOUNTING HEIGHT BUYS, at whatever tool is in the gripper today.

    ARIS_RIG=proposed ARIS_TOOL=lateral python3 scripts/height_sweep.py sweep \
        --h 0.850 --out out/atlas_proposed_h0850_lat0588 --jobs 6
    ARIS_RIG=proposed ARIS_TOOL=lateral python3 scripts/height_sweep.py report \
        --case 0.850=out/atlas_proposed_h0850_lat0588 ... \
        --json out/h_sweep_lat0588.json
    ARIS_RIG=proposed ARIS_TOOL=lateral python3 scripts/height_sweep.py pilot \
        --h 0.850 --atlas out/atlas_proposed_h0850_lat0588 --every 25

WHY IT EXISTS.  `h` is a LAYOUT constant, and every instrument that reads it —
`scripts/run_atlas6.py`, `scripts/regate_atlas.py`,
`scripts/feasible_workspace.py` — takes it from `layout.LAYOUT_PROPOSED` and
offers no way to ask about another one.  That is right for a shipped number and
wrong for a decision: the height was chosen on 2026-08-26 against a 110 mm tool
that no longer exists, and re-asking it must not mean editing the constant that
the answer is supposed to justify.  So this builds the layout LOCALLY, with
`feasible_workspace.rig(None, h, None)` — the same call the map already makes,
with the `h` its own CLI never passes — and nothing under `aris_sixarm/` is
touched or read as anything but a library.  A run at h = 0.940 reproduces the
shipped rig exactly, which is what makes the other rows believable.

THREE MODES, AND ONE OF THEM IS NOT A MAP.

  sweep   the 2 cm reachability atlas at that height, `atlas.sweep_arm` per arm,
          exactly as `feasible_workspace --sweep-atlas` does it.  ~8-25 min.
  report  everything the atlas alone can answer, at FULL resolution: union
          strict-GO, redundancy, the largest inscribed live rectangle, where
          the dead cells sit relative to the nearest base — and the number that
          decided the height last time, how far a parked arm stands from every
          OTHER arm's certified ink (`coordination`'s own capsules and margin).
  pilot   `feasible_workspace.sweep` at v13's ladder settings over every Nth
          certified cell.  A PILOT IS NOT A MAP: it reports what fraction of an
          arm's CERTIFIED cells survive the hover and route layers, never what
          fraction of the canvas does, because the cells it skipped are
          indistinguishable from cells no arm can draw.  It is here because the
          canvas-level correction between the two (measured: 97.08 % union ->
          96.84 % solo-drawable at h = 0.940, v13) is the one thing an atlas
          comparison cannot supply, and a height that changed it would not
          announce itself any other way.

THE PARK SET AT ANY OTHER HEIGHT IS THE SHIPPED RECIPE, NOT A NEW SEARCH.
`rig` re-derives it with `certified_park_poses(PARK_GRID_PROPOSED)`, and that
(radius, hover, bearing) grid was searched at h = 0.940.  So a comparison row
answers "the shipped park recipe at this height", not "the best park set at
this height", and the park-vs-ink column is a LOWER bound on what a re-search
would find there.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# read at import time, before the package is touched (see feasible_workspace)
os.environ.setdefault("ARIS_RIG", "proposed")
os.environ.setdefault("ARIS_TOOL", "lateral")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import feasible_workspace as fw                                   # noqa: E402
from aris_sixarm import atlas as atlas_mod                        # noqa: E402
from aris_sixarm import coordination, frames                      # noqa: E402

GRID = fw.GRID
SHEET = fw.SHEET
# The HOLDER's own tool, named rather than inherited: `frames.PEN_LAT` is the
# process-global and a bare run is the inline pen, which is how a whole test
# spent a year measuring one tool and gating on another (DECISIONS 2026-09-03).
EXT, LAT = frames.PEN_EXT_HOLDER, frames.PEN_LAT_HOLDER


# ---------------------------------------------------------------------------
def do_sweep(a):
    fl, parks, h, pitch = fw.rig(None, a.h, None)
    arms = sorted(fl)
    d = Path(a.out)
    d.mkdir(parents=True, exist_ok=True)
    print(f"h={h} pitch={pitch} grid={a.grid} PEN_LAT={frames.PEN_LAT} "
          f"pen={fl[arms[0]].pen}", flush=True)
    t0 = time.time()
    with mp.get_context("fork").Pool(min(a.jobs, len(arms))) as pool:
        pool.starmap(atlas_mod.sweep_arm,
                     [(x, str(d), a.grid, 1.05, h, 15.0, fl[x].pen, fl,
                       SHEET, frames.PEN_LAT) for x in arms])
    print(f"atlas h={h} -> {d}  {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------
def canvas_masks(d, arms):
    """Per-arm reach/strict-GO canvas rasters, as `run_atlas6.masks` builds them."""
    xs = np.arange(0.0, SHEET[0] + 1e-9, GRID)
    ys = np.arange(0.0, SHEET[1] + 1e-9, GRID)
    per_go, per_reach = [], []
    for a in arms:
        arr, _ = atlas_mod.load(Path(d), a)
        reach = np.zeros((len(ys), len(xs)), bool)
        go = np.zeros_like(reach)
        if len(arr):
            ix = np.rint(arr[:, 0] / GRID).astype(int)
            iy = np.rint(arr[:, 1] / GRID).astype(int)
            reach[iy, ix] = True
            go[iy, ix] = atlas_mod.strict_go(arr)
        per_go.append(go)
        per_reach.append(reach)
    return xs, ys, np.array(per_go), np.array(per_reach)


def rim(dead, xs, ys, fl, arms):
    """Where the dead cells sit, in metres from the NEAREST base. -> bands."""
    X, Y = np.meshgrid(xs, ys)
    R = np.full(X.shape, np.inf)
    for a in arms:
        bx, by = fl[a].xy
        R = np.minimum(R, np.hypot(X - bx, Y - by))
    r = R[dead]
    edges = [0.0, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 9.9]
    counts = [int(((r >= edges[i]) & (r < edges[i + 1])).sum())
              for i in range(len(edges) - 1)]
    stat = (float(r.min()), float(np.median(r)), float(r.max())) if len(r) \
        else (None, None, None)
    return edges, counts, stat


def park_vs_ink(d, fl, parks, arms):
    """The 2026-08-26 conduction criterion, at this h. -> (m, (mover, parked), n).

    Every certified drawing pose of every arm against every OTHER arm's PARKED
    chain, with `coordination`'s capsules — the same walk
    `test_no_shipped_park_pose_stands_in_another_arms_certified_ink` makes, at
    a height that test cannot ask about.
    """
    def chain(Q, spec):
        T, P = frames.fk_many(np.asarray(Q, float).reshape(-1, 7))
        tl = frames.tool_points_many(T, EXT, LAT)
        P = np.concatenate([P] + [t[:, None, :] for t in tl], axis=1)
        Twb = np.asarray(spec.T_world_base(), float)
        return P @ Twb[:3, :3].T + Twb[:3, 3]

    pk = {a: chain(np.asarray(parks[a], float)[None, :], fl[a]) for a in arms}
    tab = coordination.CAPSULES_LAT
    rr = np.array([c[2] for c in tab], float)
    worst, who, n = np.inf, None, 0
    for aid in arms:
        arr, _ = atlas_mod.load(Path(d), aid)
        Q = arr[atlas_mod.strict_go(arr)][:, atlas_mod.QCOL:atlas_mod.QCOL + 7]
        if not len(Q):
            continue
        A, B = coordination.cap_endpoints(chain(Q, fl[aid]), tab)
        for other in arms:
            if other == aid:
                continue
            Ab, Bb = coordination.cap_endpoints(pk[other], tab)
            for s in range(0, len(A), 512):
                dd = coordination.seg_seg_dist(
                    A[s:s + 512, :, None, :], B[s:s + 512, :, None, :],
                    Ab[:, None, :, :], Bb[:, None, :, :])
                g = float((dd - rr[None, :, None] - rr[None, None, :]).min())
                if g < worst:
                    worst, who = g, (aid, other)
        n += len(A)
    return worst, who, n


def column_ceiling(d, fl, arms):
    """The BEST park-vs-ink any park set at this height could reach. -> (m, pair).

    A parked arm carries one segment that no pose can move: the flange-to-
    shoulder column at its own base xy, which `scene_check.base_column` models
    and `column_clearance` measures against.  So the clearance from a mover's
    certified ink to a neighbour's COLUMN is an upper bound on park-vs-ink for
    that ordered pair, whatever park pose the neighbour is given — which is the
    same fact DECISIONS records as "a parked arm's own base column is
    pose-invariant, so no bearing buys past it", measured instead of inferred.
    """
    from aris_sixarm import scene_check
    # `column_clearance` takes the CAPSULE TABLE, not a radius vector: it walks
    # (i, j, r) triples and skips capsule 0, the mover's own bolted-down column.
    rr = scene_check._radii_for(fl, arms)
    best, who = np.inf, None
    for aid in arms:
        arr, _ = atlas_mod.load(Path(d), aid)
        sel = atlas_mod.strict_go(arr)
        Q = arr[sel][:, atlas_mod.QCOL:atlas_mod.QCOL + 7]
        if not len(Q):
            continue
        T, P = frames.fk_many(np.asarray(Q, float))
        tl = frames.tool_points_many(T, EXT, LAT)
        P = np.concatenate([P] + [t[:, None, :] for t in tl], axis=1)
        Twb = np.asarray(fl[aid].T_world_base(), float)
        P = P @ Twb[:3, :3].T + Twb[:3, 3]
        for other in arms:
            if other == aid:
                continue
            col = [scene_check.base_column(fl[other])]
            g = float(np.min(scene_check.column_clearance(P, col, rr)))
            if g < best:
                best, who = g, (aid, other)
    return best, who


def do_report(a):
    cases = []
    for c in a.case:
        hs, _, d = c.partition("=")
        cases.append((float(hs), str(ROOT / d) if not d.startswith("/") else d))
    margin = coordination.SAFETY_M + coordination.CALIB_M
    out = {}
    for h, d in cases:
        if len(list(Path(d).glob("atlas_arm*.npz"))) < 6:
            print(f"h={h}: {d} INCOMPLETE, skipped")
            continue
        fl, parks, hh, pitch = fw.rig(None, h, None)
        arms = sorted(fl)
        xs, ys, per_go, per_reach = canvas_masks(d, arms)
        cnt = per_go.sum(axis=0)
        union, reach = cnt >= 1, per_reach.any(axis=0)
        n = union.size
        big, port, land = fw.rect_report(union, GRID)
        dead = ~union
        edges, counts, rstat = rim(dead, xs, ys, fl, arms)
        pk, who, npose = park_vs_ink(d, fl, parks, arms)
        cc, ccwho = column_ceiling(d, fl, arms)
        rec = dict(
            h=h, atlas=d, cells=int(n), area_m2=round(n * GRID * GRID, 4),
            union_go_pct=round(100.0 * union.mean(), 3),
            union_go_cells=int(union.sum()),
            reach_pct=round(100.0 * reach.mean(), 3),
            dead_pct=round(100.0 * dead.mean(), 3), dead_cells=int(dead.sum()),
            ge2_pct=round(100.0 * (cnt >= 2).mean(), 3),
            ge3_pct=round(100.0 * (cnt >= 3).mean(), 3),
            max_arms=int(cnt.max()),
            per_arm={int(x): int(per_go[i].sum()) for i, x in enumerate(arms)},
            rect_largest=big, rect_portrait=port, rect_landscape=land,
            rim_edges=edges, rim_counts=counts, dead_r_min=rstat[0],
            dead_r_med=rstat[1], dead_r_max=rstat[2],
            park_vs_ink_mm=round(1000.0 * pk, 1), park_worst_pair=who,
            park_gate_mm=round(1000.0 * margin, 1), n_certified_poses=npose,
            park_ceiling_mm=round(1000.0 * cc, 1), park_ceiling_pair=ccwho,
            park_hover=fw.park_hovers(fl, parks, hh))
        out[f"{h:.3f}"] = rec
        print(f"\n=== h = {h:.3f}  ({Path(d).name}) ===")
        print(f"  union strict-GO {rec['union_go_pct']:.2f} % of {n} cells "
              f"({rec['union_go_cells']} live, {rec['dead_cells']} dead = "
              f"{rec['dead_pct']:.2f} %); reachable {rec['reach_pct']:.2f} %")
        print(f"  >=2 arms {rec['ge2_pct']:.2f} %   >=3 {rec['ge3_pct']:.2f} % "
              f"  max {rec['max_arms']}")
        print(f"  per-arm strict-GO cells: {rec['per_arm']}")
        print(f"  largest live rect {big['w']:.2f} x {big['h']:.2f} m = "
              f"{big['area_m2']:.3f} m2 at ({big['x0']:.2f}, {big['y0']:.2f});"
              f" portrait {port['area_m2']:.3f} m2, landscape "
              f"{land['area_m2']:.3f} m2")
        print("  dead cells by distance to the nearest base:")
        for i, c in enumerate(counts):
            if c:
                print(f"    {edges[i]:.2f}-{edges[i+1]:.2f} m: {c}")
        print(f"    r min/median/max {rstat[0]:.3f} / {rstat[1]:.3f} / "
              f"{rstat[2]:.3f} m")
        print(f"  park-vs-ink {rec['park_vs_ink_mm']:.1f} mm (gate "
              f"{rec['park_gate_mm']:.0f} mm), worst mover/parked {who}, "
              f"{npose} certified poses walked")
        print(f"  park-vs-ink CEILING (base columns, pose-invariant) "
              f"{rec['park_ceiling_mm']:.1f} mm, binding pair {ccwho} — no "
              f"park set at this height can beat it")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
        print(f"\nwrote {a.json}")


# ---------------------------------------------------------------------------
def do_pilot(a):
    fl, parks, h, pitch = fw.rig(None, a.h, None)
    arms = sorted(fl)
    print(f"PILOT h={h} atlas={a.atlas} every={a.every} workers={a.workers} "
          f"(v13 ladder: fiber-tries 48, hover-lean 15 deg, RRT 60 s x 300 "
          f"nodes)", flush=True)
    t0 = time.time()
    per_arm = fw.sweep(arms, Path(a.atlas), h, a.workers, chunk=24,
                       every=a.every, redundant=True,
                       ckpt=a.out + "_ckpt.npz" if a.out else None,
                       pitch=None, fiber=True, lean=15.0, tries=48,
                       rrt=60.0, rrt_nodes=300)
    el = time.time() - t0
    allc = np.vstack([v for v in per_arm.values() if len(v)])
    c = allc[:, 2].astype(int)
    rec = dict(h=h, atlas=a.atlas, every=a.every, wall_s=round(el, 1),
               n_arm_cells=int(len(allc)),
               feasible_pct=round(100.0 * (c == fw.FEASIBLE).mean(), 3),
               no_hover_pct=round(100.0 * (c == fw.NO_HOVER).mean(), 3),
               no_route_pct=round(100.0 * (c == fw.NO_ROUTE).mean(), 3),
               per_arm={})
    for arm in arms:
        ca = per_arm[arm][:, 2].astype(int)
        rec["per_arm"][int(arm)] = dict(
            n=int(len(ca)),
            feasible_pct=round(100.0 * (ca == fw.FEASIBLE).mean(), 2),
            no_hover_pct=round(100.0 * (ca == fw.NO_HOVER).mean(), 2),
            no_route_pct=round(100.0 * (ca == fw.NO_ROUTE).mean(), 2))
        p = rec["per_arm"][int(arm)]
        print(f"  arm {arm}: {p['n']} cells  feasible {p['feasible_pct']:5.2f}%"
              f"  no-hover {p['no_hover_pct']:5.2f}%  no-route "
              f"{p['no_route_pct']:5.2f}%")
    print(f"ALL {len(allc)} arm-cells: feasible {rec['feasible_pct']:.2f} %, "
          f"no-hover {rec['no_hover_pct']:.2f} %, no-route "
          f"{rec['no_route_pct']:.2f} %   [{el:.0f}s]", flush=True)
    if a.out:
        np.savez_compressed(a.out + "_raw.npz",
                            **{f"arm{k}": v for k, v in per_arm.items()})
        Path(a.out + ".json").write_text(json.dumps(rec, indent=1))
        print("wrote", a.out + ".json")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    s = sub.add_parser("sweep", help="the 2 cm atlas at one height")
    s.add_argument("--h", type=float, required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--jobs", type=int, default=6)
    s.add_argument("--grid", type=float, default=GRID)
    s.set_defaults(fn=do_sweep)

    r = sub.add_parser("report", help="what the atlases say, per height")
    r.add_argument("--case", action="append", required=True,
                   metavar="H=ATLAS_DIR")
    r.add_argument("--json", default=None)
    r.set_defaults(fn=do_report)

    p = sub.add_parser("pilot", help="the hover and route layers, sampled")
    p.add_argument("--h", type=float, required=True)
    p.add_argument("--atlas", required=True)
    p.add_argument("--every", type=int, default=25)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out", default=None)
    p.set_defaults(fn=do_pilot)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
