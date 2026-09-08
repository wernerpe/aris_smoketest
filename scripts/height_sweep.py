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
from aris_sixarm import coordination, frames, layout              # noqa: E402

GRID = fw.GRID
SHEET = fw.SHEET
# The HOLDER's own tool, named rather than inherited: `frames.PEN_LAT` is the
# process-global and a bare run is the inline pen, which is how a whole test
# spent a year measuring one tool and gating on another (DECISIONS 2026-09-03).
EXT, LAT = frames.PEN_EXT_HOLDER, frames.PEN_LAT_HOLDER


# ---------------------------------------------------------------------------
def do_sweep(a):
    # THE ATLAS DOES NOT DEPEND ON THE PARKS, so a park recipe that does not
    # transfer to this height must not block the sweep.  `atlas.sweep_arm`
    # reads the spec's base pose, its pen and its static obstacles and nothing
    # else; `rig` computes depots on the way past and at h = 0.850 the 0.940
    # grid has no certified ready pose for arm 97 at the final tool.  That is
    # a fact about the GRID (and the reason `park` is re-run per height), not
    # a reason to have no atlas.
    try:
        fl, parks, h, pitch = fw.rig(None, a.h, None)
    except RuntimeError as exc:
        h, pitch = float(a.h), fw.SHIPPED_PITCH
        fl = layout.build_fleet(layout.paired_grid(spacing=pitch, rows=3, h=h))
        print(f"  parks refused at this height ({exc}); sweeping the BARE "
              "fleet — the atlas does not read them", flush=True)
    arms = sorted(fl)
    d = Path(a.out)
    d.mkdir(parents=True, exist_ok=True)
    print(f"h={h} pitch={pitch} grid={a.grid} tilt<={a.tilt} deg "
          f"PEN_LAT={frames.PEN_LAT} pen={fl[arms[0]].pen}", flush=True)
    t0 = time.time()
    with mp.get_context("fork").Pool(min(a.jobs, len(arms))) as pool:
        pool.starmap(atlas_mod.sweep_arm,
                     [(x, str(d), a.grid, 1.05, h, a.tilt, fl[x].pen, fl,
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
# the park (radius, hover, bearing) search, at a chosen height
# ---------------------------------------------------------------------------
# The grid the 2026-09-03 re-search used, read back off the literals it
# produced: `PARK_GRID_PROPOSED`'s radii live in {0.40, 0.48, 0.55}, its hovers
# in {0.10, 0.20, 0.30, 0.35}, and every one of its bearings is a multiple of
# 15 degrees in the CANVAS frame.  6 x 4 x 24 = 576 candidates per arm.
P_RADII = (0.30, 0.40, 0.48, 0.55, 0.62, 0.70)
P_HOVERS = (0.10, 0.20, 0.30, 0.35)
P_NBEAR = 24
P_NCAND = 24          # candidates carried into the depot-flyability stage
P_NCELL = 24          # certified cells sampled per arm for that stage
_LAYERS = {}          # {arm: (ink (N,11,3), lift (M,11,3))}, filled before fork
_PARK = {}            # per-worker: fleet, h, atlas


def _pk_world(Q, spec):
    T, P = frames.fk_many(np.asarray(Q, float).reshape(-1, 7))
    tl = frames.tool_points_many(T, EXT, LAT)
    P = np.concatenate([P] + [t[:, None, :] for t in tl], axis=1)
    Twb = np.asarray(spec.T_world_base(), float)
    return P @ Twb[:3, :3].T + Twb[:3, 3]


def _pk_layers(job):
    """One arm's two occupied layers: its certified ink and the hover over it.

    BOTH, because an arm occupies both: `out/park_search3.py` found that a
    depot clearing the ink by 97 mm could still sit 51 mm inside the 6 cm LIFT
    layer above it, and the conductor named those transits.
    """
    from aris_sixarm import writing
    aid, atlas_dir, h, fl = job
    spec = fl[aid]
    arr, _ = atlas_mod.load(Path(atlas_dir), aid)
    go = arr[atlas_mod.strict_go(arr)]
    Q = go[:, atlas_mod.QCOL:atlas_mod.QCOL + 7]
    hov = []
    for row, q in zip(go, Q):
        try:
            qh, _ = writing.lifted_or_lower(spec, q, row[:2], h_inv=h,
                                            pen_ext=EXT)
        except Exception:
            qh = None
        if qh is not None:
            hov.append(qh)
    lift = _pk_world(np.array(hov), spec) if hov else np.zeros((0, 11, 3))
    return aid, (_pk_world(Q, spec), lift)


def _pk_caps(P):
    tab = coordination.CAPSULES_LAT if P.shape[1] >= 11 else coordination.CAPSULES
    A, B = coordination.cap_endpoints(P, tab)
    return A, B, np.array([c[2] for c in tab], float)


def _pk_clear_vs(park_P, layer_P, margin, chunk=512):
    """One parked chain (11,3) vs a stack of chains (N,11,3). -> min m.

    Copied verbatim from `out/park_search.py`'s `clear_vs`, which is the
    prototype the shipped park grid was searched with.  It is copied and not
    imported because `out/` is gitignored: a committed script whose numbers a
    clean checkout cannot reproduce is the failure mode this file exists to
    avoid.  The broad phase is the bounding sphere of the parked chain against
    each pose in the layer, and it is what makes 576 candidates x ~24 000 poses
    tractable at all.
    """
    if not len(layer_P):
        return float("inf")
    Aa, Ba, ra = _pk_caps(park_P[None])
    Aa, Ba = Aa[0], Ba[0]
    ca = park_P.mean(0)
    Ra = float(np.linalg.norm(park_P - ca, axis=1).max()) + float(ra.max())
    cb = layer_P.mean(1)
    Rb = np.linalg.norm(layer_P - cb[:, None], axis=2).max(1) + float(ra.max())
    live = np.linalg.norm(cb - ca, axis=1) - Ra - Rb < margin + 0.05
    if not live.any():
        return float("inf")
    P = layer_P[live]
    worst = np.inf
    for s in range(0, len(P), chunk):
        Ab, Bb, rb = _pk_caps(P[s:s + chunk])
        d = coordination.seg_seg_dist(Aa[None, :, None, :],
                                      Ba[None, :, None, :],
                                      Ab[:, None, :, :], Bb[:, None, :, :])
        worst = min(worst, float((d - ra[None, :, None]
                                  - rb[None, None, :]).min()))
    return worst


def _pk_score(P, aid, margin):
    """One parked chain (11,3) against every OTHER arm's two layers. -> m."""
    out = np.inf
    for b, (ink, lift) in _LAYERS.items():
        if b == aid:
            continue
        out = min(out, _pk_clear_vs(P, ink, margin),
                  _pk_clear_vs(P, lift, margin))
    return float(out)


def _pk_cells(atlas_dir, arm, n=P_NCELL):
    """`n` certified cells spread over one arm's GO set (farthest-point)."""
    arr, _ = atlas_mod.load(Path(atlas_dir), arm)
    go = arr[atlas_mod.strict_go(arr)]
    if not len(go):
        return []
    P = go[:, :2]
    idx = [int(np.argmin(P[:, 0] + P[:, 1]))]
    d = np.linalg.norm(P - P[idx[0]], axis=1)
    while len(idx) < min(n, len(P)):
        k = int(np.argmax(d))
        idx.append(k)
        d = np.minimum(d, np.linalg.norm(P - P[k], axis=1))
    return [(go[i, :2], go[i, atlas_mod.QCOL:atlas_mod.QCOL + 7]) for i in idx]


def _pk_flyable(spec, q_home, sample, h):
    """How many sampled cells this depot can fly to and back. -> (in, out)."""
    from aris_sixarm import writing
    n_in = n_out = 0
    for xy, q_entry in sample:
        try:
            q_hov, _ = writing.lifted_or_lower(spec, q_entry, xy, h_inv=h,
                                               pen_ext=EXT)
        except Exception:
            continue
        if q_hov is None:
            continue
        if writing.enter_beats(spec, q_home, q_hov, q_entry, pen_ext=EXT,
                               h_inv=h) is not None:
            n_in += 1
        if writing.exit_beats(spec, q_entry, q_hov, q_home, pen_ext=EXT,
                              h_inv=h) is not None:
            n_out += 1
    return n_in, n_out


def _pk_one_arm(job):
    from aris_sixarm import metrics, rig_final
    from aris_sixarm.frames import joint_margin
    aid, atlas_dir, h, fl = job
    margin = coordination.SAFETY_M + coordination.CALIB_M
    spec = fl[aid]
    rows, n_try = [], 0
    for k in range(P_NBEAR):
        bdeg = -180.0 + 360.0 * k / P_NBEAR      # absolute, canvas frame
        br = np.deg2rad(bdeg)
        bvec = np.array([np.cos(br), np.sin(br)])
        for r in P_RADII:
            for hv in P_HOVERS:
                n_try += 1
                try:
                    q, xy, _ = layout.certified_ready_pose(
                        spec, hover=hv, pen_lat=LAT, pen_ext=EXT,
                        radii=(r,), bearing=bvec)
                except RuntimeError:
                    continue
                P = coordination.chain_world(np.asarray(q)[None, :], spec,
                                             None, EXT)[0]
                steel = float(rig_final.chain_static_clearance(
                    P, spec.static_obstacles())[0])
                if steel < rig_final.STATIC_MARGIN - 1e-9:
                    continue
                rows.append(dict(
                    r=r, hover=hv, bearing=round(bdeg, 1), steel=steel,
                    q=[float(v) for v in q], xy=[float(v) for v in xy],
                    worst=_pk_score(P, aid, margin),
                    key=float(min(joint_margin(q), 2.5 * metrics.sigma_min(
                        metrics.tip_jacobian(q, pen_ext=EXT, pen_lat=LAT))))))
    top = sorted(rows, key=lambda x: (-round(x["worst"] / 0.005), -x["key"]))
    sample = _pk_cells(atlas_dir, aid)
    out = []
    for x in top[:P_NCAND]:
        n_in, n_out = _pk_flyable(spec, np.asarray(x["q"], float), sample, h)
        out.append(dict(x, n_in=n_in, n_out=n_out, n_cells=len(sample)))
    return aid, out, len(rows), n_try


def do_park(a):
    """The (radius, hover, bearing) search, at `--h`, against `--atlas`."""
    import multiprocessing as mp
    fl, parks, h, pitch = fw.rig(None, a.h, None)
    arms = sorted(fl)
    margin = coordination.SAFETY_M + coordination.CALIB_M
    print(f"PARK SEARCH h={h} atlas={a.atlas} tool=({EXT}, {LAT})")
    print(f"  {P_NBEAR} bearings x {len(P_RADII)} radii x {len(P_HOVERS)} "
          f"hovers = {P_NBEAR * len(P_RADII) * len(P_HOVERS)} candidates/arm, "
          f"gate {1000 * margin:.0f} mm", flush=True)
    t0 = time.time()
    with mp.get_context("fork").Pool(min(a.jobs, len(arms))) as pool:
        for aid, L in pool.map(_pk_layers,
                               [(x, a.atlas, h, fl) for x in arms]):
            _LAYERS[aid] = L
    print("  both layers for 6 arms in "
          f"{time.time() - t0:.0f} s: "
          + " ".join(f"{k}:{len(v[0])}/{len(v[1])}"
                     for k, v in sorted(_LAYERS.items())), flush=True)
    t0 = time.time()
    with mp.get_context("fork").Pool(min(a.jobs, len(arms))) as pool:
        res = pool.map(_pk_one_arm, [(x, a.atlas, h, fl) for x in arms])
    print(f"  searched in {time.time() - t0:.0f} s\n", flush=True)

    best, allrows, cert = {}, {}, {}
    for aid, rows, n_ok, n_try in res:
        allrows[str(aid)] = rows
        cert[aid] = (n_ok, n_try)
        print(f"arm {aid}: {n_ok} of {n_try} candidates certify and clear "
              "the steel")
        rank = sorted(rows, key=lambda x: (-(x["n_in"] + x["n_out"]),
                                           -round(x["worst"] / 0.005),
                                           -x["key"]))
        for x in rank[:3]:
            print(f"    r {x['r']:.2f} hover {x['hover']:.2f} bearing "
                  f"{x['bearing']:+7.1f}  both layers "
                  f"{1000 * x['worst']:7.1f} mm  entry {x['n_in']:2d}/"
                  f"{x['n_cells']} home {x['n_out']:2d}  xy "
                  f"({x['xy'][0]:.2f}, {x['xy'][1]:.2f})")
        pick = next((x for x in rank if x["worst"] >= margin - 1e-9), None)
        best[aid] = pick
        if pick is None:
            bw = max((x["worst"] for x in rows), default=float("-inf"))
            print(f"    !! NO depot clears both layers by "
                  f"{1000 * margin:.0f} mm; best {1000 * bw:.1f} mm")

    ok = all(v is not None for v in best.values())
    doc = dict(h=h, atlas=a.atlas, margin=margin, certifies=bool(ok),
               grid=dict(radii=list(P_RADII), hovers=list(P_HOVERS),
                         n_bearings=P_NBEAR),
               certified_counts={str(k): list(v) for k, v in cert.items()},
               best={str(k): v for k, v in best.items()}, all=allrows)
    if ok:
        worst = min(best[x]["worst"] for x in arms)
        paths = {x: coordination.ArmPath(x, np.asarray(best[x]["q"], float)
                                         [None, :], 0.05, h_inv=None,
                                         spec=fl[x]) for x in arms}
        pair = np.inf
        for i, x in enumerate(arms):
            for y in arms[i + 1:]:
                pair = min(pair, float(np.min(
                    coordination.clearance_matrix(paths[x], paths[y]))))
        n_in = sum(best[x]["n_in"] for x in arms)
        n_out = sum(best[x]["n_out"] for x in arms)
        n_cell = sum(best[x]["n_cells"] for x in arms)
        doc.update(fleet_worst_mm=round(1000 * worst, 1),
                   fleet_pair_mm=round(1000 * pair, 1),
                   entries=[n_in, n_cell], go_homes=[n_out, n_cell])
        print(f"\nFLEET worst park-vs-(ink AND lift) {1000 * worst:.1f} mm "
              f"against the {1000 * margin:.0f} mm the conductor asks — "
              f"{'CERTIFIES' if worst >= margin else 'REFUSES'}")
        print(f"FLEET worst park-vs-park {1000 * pair:.1f} mm")
        print(f"entries flyable {n_in}/{n_cell} "
              f"({100.0 * n_in / n_cell:.1f} %), go-homes {n_out}/{n_cell} "
              f"({100.0 * n_out / n_cell:.1f} %)")
        print(f"\nPARK_GRID at h = {h:.3f}  (REPORT ONLY, not committed) = {{")
        for aid in arms:
            b = best[aid]
            print(f"    {aid}: ({b['r']:.2f}, {b['hover']:.2f}, "
                  f"{b['bearing']:.1f}),")
        print("}")
    else:
        miss = [k for k, v in best.items() if v is None]
        print(f"\nNO FLEET PARK SET at h = {h:.3f}: arms {miss} have no depot "
              f"clearing {1000 * margin:.0f} mm")
    if a.out:
        Path(a.out).write_text(json.dumps(doc, indent=1))
        print("\nwrote", a.out)


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
    s.add_argument("--tilt", type=float, default=15.0,
                   help="the pen-lean cone the gated search may climb, in "
                        "degrees.  15 is the shipped allowance; 20 is Pete's "
                        "pending decision and a wider cone certifies cells a "
                        "perpendicular pen cannot reach.")
    s.set_defaults(fn=do_sweep)

    r = sub.add_parser("report", help="what the atlases say, per height")
    r.add_argument("--case", action="append", required=True,
                   metavar="H=ATLAS_DIR")
    r.add_argument("--json", default=None)
    r.set_defaults(fn=do_report)

    k = sub.add_parser("park", help="the (radius, hover, bearing) search")
    k.add_argument("--h", type=float, required=True)
    k.add_argument("--atlas", required=True)
    k.add_argument("--jobs", type=int, default=6)
    k.add_argument("--out", default=None)
    k.set_defaults(fn=do_park)

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
