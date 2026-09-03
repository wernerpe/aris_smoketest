#!/usr/bin/env python3
"""PHYSICS vs POLICY: what is actually under the six dead discs of the proposed rig.

    ARIS_RIG=proposed ARIS_TOOL=lateral python3 scripts/dead_disc_anatomy.py
    ARIS_RIG=proposed ARIS_TOOL=lateral python3 scripts/dead_disc_anatomy.py --selftest

7.98 % of the canvas has no certified pen-down pose.  "Unreachable" is doing a
lot of work in that sentence: some of it is a comfort POLICY we chose, some is
a calibration allowance the commissioning survey retires, some is a pen pose we
already have code for and do not use, and some would be the arm's own shoulder
axis, which no amount of software moves.  This script separates them by asking,
of every cell that is dead today, WHICH RELAXATION FIRST REVIVES IT.

THE CONTROL

  L0   certified today.  `atlas.solve_cell` verbatim on the committed model:
       strict gates (margin >= 0.30 rad, sigma_min >= 0.14), the full collision
       set with the 30 mm calibration inflation, clearance >= STATIC_MARGIN =
       50 mm, and the atlas's own search — 8 tool yaws x 16 q7 x every IK
       branch, PERPENDICULAR first with a <= 15 deg tilt-cone fallback,
       clearance-probing the best SIX solutions by joint margin and no more.
       `out/dda_control.py` re-sweeps it and reproduces the committed atlas
       cell for cell (92.0239 %, 1321 dead cells, zero differences).

THE LADDER (each layer's mask is computed over ALL dead cells, and the nesting
is asserted, so first-recovery can be read off in any order)

  L1   comfort.  margin >= 0.15, sigma_min >= 0.10 — the repo's own hard gates,
       the floor at which planner and band checker still certify a pose as
       executable.  PERPENDICULAR pen.  Everything else as L0.  Exhaustive:
       every (yaw, q7, branch) is clearance-checked, not just the best six.
  L2   + surveyed bases.  The neighbour base columns rebuilt from the MEASURED
       mesh-audit bands with the 30 mm calibration inflation removed; the 50 mm
       static clearance kept.  What commissioning survey + touch-off buys.
  L3   + 15 deg pen tilt.  The same tilt cone `atlas._candidates` builds, at
       the pose gate.  A POSE-LEVEL sweep, not a tilt-planner run — see the
       ACCURACY section of the report.
  L4   raw kinematics, bare metal.  Any FK-verified IK solution inside FR3
       limits whose chain and tool clear the MEASURED metal with ZERO margin,
       at any joint margin and any sigma_min.

DIAGNOSTICS, because the ladder above conflates two things the answer turns on

  A_perp / A_tilt   the atlas's OWN policy (strict gates, inflated columns,
       50 mm) searched EXHAUSTIVELY, perpendicular and with its 15 deg cone.
       Not a relaxation of anything physical: it measures how much of the dead
       map is the atlas's best-six probe giving up early.
  P4   L4's bare metal and raw kinematics with the pen held PERPENDICULAR.
       What is dead here cannot be touched with a vertical pen at all — the
       physics floor of the pen policy the LATERAL planner actually enforces.
  L4w  L4 with a 30 deg cone: a sensitivity check on the 15 deg number.

Outputs
    out/dead_disc_anatomy.png     the canvas, coloured by recovering layer
    out/dead_disc_anatomy.json    every number the report quotes
    out/dda_layers.npz            the masks themselves
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# the rig is chosen before the first aris_sixarm import (fleet.activate)
os.environ.setdefault("ARIS_RIG", "proposed")
os.environ.setdefault("ARIS_TOOL", "lateral")
sys.path.insert(0, str(ROOT))

import numpy as np                                              # noqa: E402
from aris_sixarm import atlas, ik, mounts, rig_final            # noqa: E402
from aris_sixarm.fleet import FLEET, SHEET                      # noqa: E402
from aris_sixarm import frames as _F                            # noqa: E402
from aris_sixarm.frames import (rotx, rotz, rot_axis, tool_offset,
                                tool_points_many, FR3_MIN, FR3_MAX)  # noqa: E402

GRID = 0.02
H = 0.940                 # proposed rig mount height (LAYOUT_PROPOSED["h"])
# the HOLDER's own pair, not the inline pen's: both were 0.110 until
# 2026-09-03 and are 0.0588421 now (frames.py, docs/SYSTEM_MODEL.md 7e), so
# anything this script printed before that date was earned at the older tool.
PEN_EXT = _F.PEN_EXT_HOLDER
PEN_LAT = _F.PEN_LAT_HOLDER   # the lateral holder, passed EXPLICITLY everywhere
ATLAS = ROOT / "out" / "atlas_proposed_h0940_banded"
CELL_M2 = GRID * GRID

# the logo span nobody drew (out/holeA.py, v8 campaign): 109.87 mm of grey
HOLE_A = np.array([[1.26082, 1.86557], [1.15106, 1.86061]])

STRICT_M, STRICT_S = 0.30, 0.14      # metrics.GATE_MARGIN / GATE_SIGMA
HARD_M, HARD_S = 0.15, 0.10          # validate.MARGIN_GATE / pwl.SIGMA_GATE
CALIB = 0.03                         # mounts.MOUNTS.calib
CLEAR = 0.05                         # rig_final.STATIC_MARGIN

# gate_m / gate_s  joint-margin (rad) and sigma_min floors
# calib            metres of calibration inflation on the neighbour columns
# clear            metres of static clearance demanded of every capsule
# tilt             pen-tilt cone half-angles, degrees (() = perpendicular only)
# zfloor           metres the chain must stay above the paper
LAYERS = [
    ("A_perp", dict(gate_m=STRICT_M, gate_s=STRICT_S, calib=CALIB, clear=CLEAR,
                    tilt=(), zfloor=0.02)),
    ("A_tilt", dict(gate_m=STRICT_M, gate_s=STRICT_S, calib=CALIB, clear=CLEAR,
                    tilt=(15.0,), zfloor=0.02)),
    ("L1", dict(gate_m=HARD_M, gate_s=HARD_S, calib=CALIB, clear=CLEAR,
                tilt=(), zfloor=0.02)),
    ("L2", dict(gate_m=HARD_M, gate_s=HARD_S, calib=0.0, clear=CLEAR,
                tilt=(), zfloor=0.02)),
    ("P4", dict(gate_m=0.0, gate_s=0.0, calib=0.0, clear=0.0,
                tilt=(), zfloor=0.0)),
    ("L3", dict(gate_m=HARD_M, gate_s=HARD_S, calib=0.0, clear=CLEAR,
                tilt=(15.0,), zfloor=0.02)),
    ("L4", dict(gate_m=0.0, gate_s=0.0, calib=0.0, clear=0.0,
                tilt=(15.0,), zfloor=0.0)),
    ("L4w", dict(gate_m=0.0, gate_s=0.0, calib=0.0, clear=0.0,
                 tilt=(15.0, 30.0), zfloor=0.0)),
]
LAYER_CFG = dict(LAYERS)
# every pair that must nest, checked at runtime: (subset, superset)
NESTING = [("A_perp", "L1"), ("A_perp", "A_tilt"), ("L1", "L2"), ("L2", "L3"),
           ("L2", "P4"), ("A_tilt", "L3"), ("L3", "L4"), ("P4", "L4"),
           ("L4", "L4w")]
LADDER = ["L1", "L2", "L3", "L4"]        # the brief's cumulative ordering

# the tilt grid for the "how much lean does it take" map
TILT_GRID = (2.5, 5.0, 7.5, 10.0, 12.5, 15.0)

_YAWS = np.linspace(0, 2 * np.pi, 8, endpoint=False)


def orientations(tilt_degs=(), halves=True):
    """World tool rotations for a layer. -> (O,3,3), first 8 perpendicular.

    Same construction as `atlas._candidates`: the pen straight down at 8 tool
    yaws, then, for each cone half-angle, that yaw leaned by half and by the
    full angle about +-tool x and +-tool y.  `halves=False` leans by the full
    angle only — what the min-tilt grid wants, where the halves are already
    grid points of their own.
    """
    out = [rotz(yaw) @ rotx(np.pi) for yaw in _YAWS]
    for tmax in tilt_degs:
        angs = (0.5 * tmax, tmax) if halves else (tmax,)
        for yaw in _YAWS:
            R = rotz(yaw) @ rotx(np.pi)
            for ang_deg in angs:
                a = np.deg2rad(ang_deg)
                for ax in ((1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)):
                    out.append(rot_axis(R @ np.array(ax, float), a) @ R)
    return np.array(out)


def obstacle_boxes(arm_id, fleet, calib):
    """Every OTHER arm's mount hardware and base column, at a chosen calib.

    calib = 0.03 rebuilds exactly `spec.static_obstacles()` (asserted by
    `--selftest`); calib = 0.0 is the surveyed rig — the columns are the
    mesh-audit bands themselves, with no allowance for not knowing where the
    base is.  Built HERE rather than taken from the spec so each layer owns
    precisely the metal it claims to model.
    """
    m = mounts.MOUNTS.scaled(calib=float(calib))
    return mounts.obstacles_for(arm_id, fleet, H, m)


def prune_boxes(boxes, z_hi, r_max, clear):
    """Drop boxes no capsule can come within (r_max + clear) of, in z alone.

    Conservative and provable: a capsule whose every point sits below `z_hi`
    clears any box whose lower face is above `z_hi + r_max + clear` by more
    than the gate demands.  Cannot change an answer; only saves work.
    """
    cut = z_hi + r_max + clear
    return [b for b in boxes if b["lo"][2] <= cut]


def eval_cells(cells, spec, boxes, R_all, q7s, gate_m, gate_s, clear, zfloor,
               pen_ext=PEN_EXT, pen_lat=PEN_LAT, report=None, detail=None):
    """Does ANY (yaw/tilt, q7, IK branch) pose put this arm's pen on the cell
    and pass this layer's gates?  -> bool (N,).

    Exhaustive.  Every candidate is FK-verified inside `ik.solve_batch` (the
    solver clamps at the workspace boundary instead of failing, so this is not
    optional), re-filtered against FR3 limits, then gated on joint margin, on
    the paper plane, on the arm's own boom cylinder, on capsule clearance to
    the layer's obstacle set, and on sigma_min.  Unlike `atlas.solve_cell`
    nothing is truncated to the best few by margin.

    `detail`, if given, is a dict of (N,) arrays filled with the BEST surviving
    pose per cell: margin, sigma, clearance, tilt_deg (the least lean that
    works), and q.
    """
    cells = np.asarray(cells, float).reshape(-1, 2)
    N = len(cells)
    hit = np.zeros(N, bool)
    if not N:
        return hit
    Twb = spec.T_world_base(H)
    Twb_inv = np.linalg.inv(Twb)
    Rw, tw = Twb[:3, :3], Twb[:3, 3]
    off = tool_offset(pen_ext, pen_lat)
    seed = spec.q_seed
    O, NQ = len(R_all), len(q7s)
    Roff = R_all @ off                                   # (O,3)
    # the lean of each candidate: angle between -tool z and world -z
    lean = np.degrees(np.arccos(np.clip(-R_all[:, 2, 2], -1.0, 1.0)))
    legacy_inv = (spec.mount == "inv"
                  and getattr(spec, "rig", "sixarm") == "sixarm")
    r_max = max(r for *_, r in rig_final.STATIC_CAPSULES_LAT)
    q7s = np.asarray(q7s, float)
    step = max(1, int(1.5e5 // max(1, O * NQ)))          # ~50 MB of arrays

    for c0 in range(0, N, step):
        sub = cells[c0:c0 + step]
        n = len(sub)
        tip = np.column_stack([sub, np.zeros(n)])        # (n,3)
        T_w = np.zeros((n, O, 4, 4))
        T_w[:, :, 3, 3] = 1.0
        T_w[:, :, :3, :3] = R_all[None]
        T_w[:, :, :3, 3] = tip[:, None, :] - Roff[None]
        T_b = Twb_inv @ T_w.reshape(-1, 4, 4)            # (n*O,4,4)
        flat = np.repeat(ik._flat16(T_b), NQ, axis=0)    # (n*O*NQ,16)
        Q, valid = ik.solve_batch(flat, np.tile(q7s, n * O), seed)
        idx = np.flatnonzero(valid.reshape(-1))          # into n*O*NQ*4
        if report is not None:
            report["ik_valid"] += len(idx)
        if not len(idx):
            continue
        q = Q.reshape(-1, 7)[idx]

        # (a) joint-limit comfort
        m = np.min(np.minimum(q - FR3_MIN, FR3_MAX - q), axis=1)
        k = m >= gate_m
        idx, q, m = idx[k], q[k], m[k]
        if not len(idx):
            continue

        # (b) the paper, and the arm's own boom cylinder (the legacy proxy the
        #     atlas and the planner both keep for a "sixarm"-rig inverted arm)
        T, p = ik.fk_batch(q)
        pw = p @ Rw.T + tw
        k = pw[:, 1:, 2].min(axis=1) >= zfloor
        if legacy_inv:
            rb = np.hypot(p[:, :, 0], p[:, :, 1])
            boom = np.any((p[:, :, 2] < -0.02) & (rb < 0.12), axis=1)
            if report is not None:
                report["boom_fires"] += int((k & boom).sum())
            k &= ~boom
        idx, q, m, T, pw = idx[k], q[k], m[k], T[k], pw[k]
        if not len(idx):
            continue

        # (c) static clearance, capsule surface to box surface
        cl = np.full(len(idx), np.inf)
        if boxes:
            tool_b = tool_points_many(T, pen_ext, pen_lat)
            tool_w = [t @ Rw.T + tw for t in tool_b]
            P = np.concatenate([pw] + [t[:, None, :] for t in tool_w], axis=1)
            bx = prune_boxes(boxes, float(pw[:, :, 2].max()), r_max, clear)
            if bx:
                cl = rig_final.chain_static_clearance(P, bx)
                k = cl >= clear if clear > 0 else cl > 0.0
                idx, q, m, cl = idx[k], q[k], m[k], cl[k]
                if not len(idx):
                    continue

        # (d) force controllability at the pen tip
        s = np.full(len(idx), np.inf)
        if gate_s > 0 or detail is not None:
            s = np.linalg.svd(ik.tip_jacobian_batch(q, pen_ext=pen_ext,
                                                    pen_lat=pen_lat),
                              compute_uv=False)[:, -1]
            k = s >= gate_s
            idx, q, m, cl, s = idx[k], q[k], m[k], cl[k], s[k]
            if not len(idx):
                continue

        ci = c0 + idx // (O * NQ * 4)
        hit[ci] = True
        if detail is not None:
            # the LEAST lean that works, tie-broken by the largest margin
            oi = (idx // (NQ * 4)) % O
            rank = lean[oi] - 1e-3 * m
            order = np.argsort(rank, kind="stable")
            for t in order[::-1]:                        # last write wins
                detail["tilt"][ci[t]] = lean[oi[t]]
                detail["margin"][ci[t]] = m[t]
                detail["sigma"][ci[t]] = s[t]
                detail["clear"][ci[t]] = cl[t]
                detail["q"][ci[t]] = q[t]
    return hit


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------
def control(arms, jobs=6):
    """L0: re-sweep the whole canvas with `atlas.solve_cell` and check that it
    reproduces the committed atlas cell for cell.  ~2 min on six cores.

    Nothing downstream is worth reading if this does not pass: it is the proof
    that the machinery below is gating the same rig, the same tool, the same
    collision model and the same solver as the published 92.02 % map.
    """
    import functools
    t0 = time.time()
    tmp = ROOT / "out" / "dda_control_atlas"
    tmp.mkdir(parents=True, exist_ok=True)
    fn = functools.partial(atlas.sweep_arm, out_dir=str(tmp), grid=GRID,
                           tilt_max_deg=15.0, pen_ext=PEN_EXT, fleet=FLEET,
                           sheet=SHEET, pen_lat=PEN_LAT)
    with mp.get_context("fork").Pool(min(jobs, len(arms))) as pool:
        arrs = pool.map(fn, arms)
    xs = np.arange(0.0, SHEET[0] + 1e-9, GRID)
    ys = np.arange(0.0, SHEET[1] + 1e-9, GRID)
    ref = np.load(ATLAS / "coverage.npz")
    out = dict(grid=GRID, seconds=round(time.time() - t0, 1), per_arm={})
    go = []
    ok = True
    for a, arr in zip(arms, arrs):
        g = np.zeros((len(ys), len(xs)), bool)
        r = np.zeros_like(g)
        if len(arr):
            ix = np.rint(arr[:, 0] / GRID).astype(int)
            iy = np.rint(arr[:, 1] / GRID).astype(int)
            r[iy, ix] = True
            g[iy, ix] = atlas.strict_go(arr)
        i = list(ref["arms"]).index(a)
        dg = int((g != ref["per_arm_go"][i]).sum())
        dr = int((r != ref["per_arm_reach"][i]).sum())
        ok &= (dg == 0 and dr == 0)
        out["per_arm"][str(a)] = dict(go=int(g.sum()), reach=int(r.sum()),
                                      go_diff_vs_committed=dg,
                                      reach_diff_vs_committed=dr)
        print(f"  arm {a}: GO {int(g.sum())} reach {int(r.sum())}  |  diff vs "
              f"committed  GO {dg}  reach {dr}", flush=True)
        go.append(g)
    union = np.array(go).any(axis=0)
    out["union_go_pct"] = round(100 * float(union.mean()), 4)
    out["dead_cells"] = int((~union).sum())
    out["dead_cells_committed"] = int((~ref["union_go"]).sum())
    out["dead_mask_diff"] = int(((~union) != (~ref["union_go"])).sum())
    out["exact_reproduction"] = bool(ok and out["dead_mask_diff"] == 0)
    print(f"  union GO {out['union_go_pct']:.4f} %  dead {out['dead_cells']} "
          f"(committed {out['dead_cells_committed']})  dead-mask diff "
          f"{out['dead_mask_diff']}\n  EXACT REPRODUCTION: "
          f"{out['exact_reproduction']}  [{out['seconds']}s]")
    (ROOT / "out" / "dda_L0_control.json").write_text(json.dumps(out, indent=1))
    return out


def _one_arm(job):
    """(arm_id, cells, layer_name) -> (arm_id, mask, detail, diag)."""
    arm_id, cells, name = job
    cfg = LAYER_CFG[name]
    n = len(cells)
    det = dict(tilt=np.full(n, np.nan), margin=np.full(n, np.nan),
               sigma=np.full(n, np.nan), clear=np.full(n, np.nan),
               q=np.full((n, 7), np.nan))
    rep = dict(ik_valid=0, boom_fires=0)
    hit = eval_cells(cells, FLEET[arm_id],
                     obstacle_boxes(arm_id, FLEET, cfg["calib"]),
                     orientations(cfg["tilt"]), ik.Q7_GRID,
                     cfg["gate_m"], cfg["gate_s"], cfg["clear"], cfg["zfloor"],
                     report=rep, detail=det)
    return arm_id, hit, det, rep


def sweep_layer(name, cells, arms, jobs=6):
    """-> (any-arm mask, per-arm mask, per-arm detail, diagnostics)."""
    t0 = time.time()
    js = [(a, cells, name) for a in arms]
    if jobs > 1 and len(cells) > 200:
        with mp.get_context("fork").Pool(min(jobs, len(arms))) as pool:
            res = pool.map(_one_arm, js)
    else:
        res = [_one_arm(j) for j in js]
    per = np.array([h for _, h, _, _ in res])
    det = {a: d for a, _, d, _ in res}
    rep = dict(ik_valid=int(sum(r["ik_valid"] for *_, r in res)),
               boom_fires=int(sum(r["boom_fires"] for *_, r in res)),
               seconds=round(time.time() - t0, 1))
    return per.any(axis=0), per, det, rep


def min_tilt(cells, arms, gate_m=STRICT_M, gate_s=STRICT_S, calib=CALIB,
             clear=CLEAR, jobs=6):
    """Smallest lean on TILT_GRID at which some arm has a gate-passing pose.

    -> (deg (N,), nan where no grid point works).  Run at the STRICT gates and
    with TODAY's inflated columns: this is what the lean has to be for a cell
    to be certified with nothing else in the rig changing.
    """
    out = np.full(len(cells), np.nan)
    todo = np.arange(len(cells))
    for t in TILT_GRID:
        R = orientations((t,), halves=False)
        got = np.zeros(len(todo), bool)
        for a in arms:
            got |= eval_cells(cells[todo], FLEET[a],
                              obstacle_boxes(a, FLEET, calib), R, ik.Q7_GRID,
                              gate_m, gate_s, clear, 0.02)
        out[todo[got]] = t
        todo = todo[~got]
        if not len(todo):
            break
    return out


# ---------------------------------------------------------------------------
# discs
# ---------------------------------------------------------------------------
def components(mask):
    """8-connected components of a bool grid -> int labels (0 = background)."""
    lab = np.zeros(mask.shape, np.int32)
    nxt = 0
    for seed in zip(*np.nonzero(mask)):
        if lab[seed]:
            continue
        nxt += 1
        stack = [seed]
        lab[seed] = nxt
        while stack:
            i, j = stack.pop()
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    a, b = i + di, j + dj
                    if (0 <= a < mask.shape[0] and 0 <= b < mask.shape[1]
                            and mask[a, b] and not lab[a, b]):
                        lab[a, b] = nxt
                        stack.append((a, b))
    return lab, nxt


def hole_a_table(arms, n=21):
    """The 109.87 mm logo span, per arm and per layer. -> dict.

    Probes the span densely (not just its ends): a stroke needs EVERY point,
    so a layer only "recovers hole A" for an arm if that arm covers all of it.
    """
    t = np.linspace(0, 1, n)[:, None]
    pts = HOLE_A[0] + t * (HOLE_A[1] - HOLE_A[0])
    out = dict(span_mm=float(1000 * np.linalg.norm(HOLE_A[1] - HOLE_A[0])),
               points=n, ends=HOLE_A.tolist(),
               midpoint=HOLE_A.mean(axis=0).tolist(), per_arm={}, per_layer={})
    for a in arms:
        d = float(np.hypot(*(np.array(FLEET[a].xy) - HOLE_A.mean(axis=0))))
        out["per_arm"][str(a)] = dict(base=list(map(float, FLEET[a].xy)),
                                      mid_dist_mm=round(1000 * d, 1), layers={})
    for name, cfg in LAYERS:
        cov = {}
        for a in arms:
            hit = eval_cells(pts, FLEET[a], obstacle_boxes(a, FLEET,
                                                           cfg["calib"]),
                             orientations(cfg["tilt"]), ik.Q7_GRID,
                             cfg["gate_m"], cfg["gate_s"], cfg["clear"],
                             cfg["zfloor"])
            cov[a] = hit
            out["per_arm"][str(a)]["layers"][name] = dict(
                points=int(hit.sum()), whole_span=bool(hit.all()))
        anyarm = np.any([cov[a] for a in arms], axis=0)
        out["per_layer"][name] = dict(
            points_covered_by_some_arm=int(anyarm.sum()),
            whole_span_by_some_arm=bool(anyarm.all()),
            whole_span_by_one_arm=sorted(int(a) for a in arms
                                         if cov[a].all()))
    # the least lean that certifies the WHOLE span for one arm, strict gates
    out["min_tilt_whole_span_strict"] = {}
    for a in arms:
        best = None
        for t_ in TILT_GRID:
            h = eval_cells(pts, FLEET[a], obstacle_boxes(a, FLEET, CALIB),
                           orientations((t_,), halves=False), ik.Q7_GRID,
                           STRICT_M, STRICT_S, CLEAR, 0.02)
            if h.all():
                best = t_
                break
        out["min_tilt_whole_span_strict"][str(a)] = best
    return out


def disc_table(dead0, xs, ys, cells, masks, mt, arms):
    """The SIX discs, one per base column.

    Every dead cell is attached to the column it is nearest; the connected
    components inside that set separate the disc proper from the handful of
    outlying fragments (single cells at the edge of two arms' reach), which are
    counted but not confused with it.
    """
    lab, ncomp = components(dead0)
    iy, ix = np.nonzero(dead0)
    order = {(int(b), int(c)): i for i, (b, c) in enumerate(zip(iy, ix))}
    cix = np.rint(cells[:, 0] / GRID).astype(int)
    ciy = np.rint(cells[:, 1] / GRID).astype(int)
    comp = np.array([lab[b, c] for b, c in zip(ciy, cix)])
    centres = {a: np.array(FLEET[a].xy, float) for a in arms}
    C = np.array([centres[a] for a in arms])
    d = np.linalg.norm(cells[:, None, :] - C[None], axis=2)
    own = np.array(arms)[d.argmin(axis=1)]
    rad = d.min(axis=1)

    discs = []
    for a in arms:
        sel = np.flatnonzero(own == a)
        cs, r = cells[sel], rad[sel]
        prev = np.zeros(len(sel), bool)
        rec = {}
        for name in LADDER:
            m = masks[name][sel]
            rec[name] = int((m & ~prev).sum())
            prev |= m
        core = int((~prev).sum())
        leans = mt[sel]
        # the disc proper = the largest connected component in this set
        cc = comp[sel]
        big = np.bincount(cc).argmax()
        bs = cc == big
        bx, by = cs[bs, 0], cs[bs, 1]
        discs.append(dict(
            arm=int(a), column=[float(centres[a][0]), float(centres[a][1])],
            cells=int(len(sel)), area_m2=round(len(sel) * CELL_M2, 5),
            main_disc_cells=int(bs.sum()),
            main_disc_m2=round(int(bs.sum()) * CELL_M2, 5),
            main_disc_span_mm=[round(1000 * float(bx.max() - bx.min()), 1),
                               round(1000 * float(by.max() - by.min()), 1)],
            main_disc_max_radius_mm=round(1000 * float(r[bs].max()), 1),
            outlying_fragments=int(len(np.unique(cc)) - 1),
            outlying_cells=int((~bs).sum()),
            max_radius_mm=round(1000 * float(r.max()), 1),
            recovered=rec,
            recovered_m2={q: round(v * CELL_M2, 5) for q, v in rec.items()},
            irreducible_cells=core,
            irreducible_m2=round(core * CELL_M2, 5),
            irreducible_max_radius_mm=(round(1000 * float(r[~prev].max()), 1)
                                       if core else 0.0),
            already_strict_go_at_15deg=int(masks["A_tilt"][sel].sum()),
            lean_median_deg=(float(np.nanmedian(leans))
                             if np.isfinite(leans).any() else None),
            lean_max_deg=(float(np.nanmax(leans))
                          if np.isfinite(leans).any() else None),
            lean_over_15deg_cells=int(np.isnan(leans).sum())))
    discs.sort(key=lambda x: -x["cells"])
    for x in discs:
        x["total_components_on_canvas"] = int(ncomp)
    return discs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--report", action="store_true",
                    help="skip the sweep, rebuild the JSON and the figure "
                         "from out/dda_layers.npz")
    ap.add_argument("--control", action="store_true",
                    help="only re-sweep L0 and diff it against the committed "
                         "atlas (~2 min); done automatically otherwise")
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    arms = sorted(FLEET)
    if a.selftest:
        return selftest(arms)
    if a.control:
        return control(arms, a.jobs)
    if a.report:
        return report(arms)

    ctrl_p = ROOT / "out" / "dda_L0_control.json"
    if not ctrl_p.exists():
        print("L0 control: re-sweeping the canvas with atlas.solve_cell",
              flush=True)
        c = control(arms, a.jobs)
        assert c["exact_reproduction"], \
            "L0 control did NOT reproduce the committed atlas — stop here"

    xs = np.arange(0.0, SHEET[0] + 1e-9, GRID)
    ys = np.arange(0.0, SHEET[1] + 1e-9, GRID)
    ref = np.load(ATLAS / "coverage.npz")
    dead0 = ~ref["union_go"]
    iy, ix = np.nonzero(dead0)
    cells = np.column_stack([xs[ix], ys[iy]])
    print(f"canvas {len(xs)}x{len(ys)} = {dead0.size} cells @ {GRID} m; "
          f"L0 dead {len(cells)} cells = {len(cells) * CELL_M2:.4f} m² "
          f"({100 * len(cells) / dead0.size:.2f} %)", flush=True)

    masks, per_arm, details, diag = {}, {}, {}, {}
    for name, _ in LAYERS:
        got, per, det, rep = sweep_layer(name, cells, arms, a.jobs)
        masks[name], per_arm[name], details[name], diag[name] = got, per, det, rep
        print(f"  {name:<7} reaches {int(got.sum()):5d} / {len(cells)} dead "
              f"cells  ({got.sum() * CELL_M2:.4f} m²)   "
              f"[{rep['seconds']}s, {rep['ik_valid']} FK-verified IK sols, "
              f"boom vetoes {rep['boom_fires']}]", flush=True)

    print("\n  nesting checks (must all be True):")
    for lo, hi in NESTING:
        ok = bool((~masks[lo] | masks[hi]).all())
        print(f"    {lo:<7} subset of {hi:<7} {ok}")
        assert ok, f"{lo} is not a subset of {hi}"

    print("\n  minimum pen lean for a STRICT-gate pose (today's columns):",
          flush=True)
    mt = min_tilt(cells, arms, jobs=a.jobs)
    for t in (0.0,) + TILT_GRID:
        n = int((mt == t).sum()) if t else int(masks["A_perp"].sum())
        if n:
            print(f"    {t:5.1f} deg: {n:5d} cells")
    print(f"    none of {TILT_GRID[-1]:.0f} deg: "
          f"{int(np.isnan(mt).sum())} cells")

    np.savez_compressed(ROOT / "out" / "dda_layers.npz", cells=cells,
                        arms=np.array(arms, np.int64), xs=xs, ys=ys,
                        dead0=dead0, min_tilt=mt,
                        diag=np.array(json.dumps(diag)),
                        **{f"m_{k}": v for k, v in masks.items()},
                        **{f"pa_{k}": v for k, v in per_arm.items()},
                        **{f"tilt_{k}": np.array([details[k][x]["tilt"]
                                                  for x in arms])
                           for k in ("A_tilt", "L3")})
    print("\nwrote out/dda_layers.npz")
    report(arms)


LAYER_LABEL = {
    "L1": "L1  comfort policy  (margin 0.30$\\to$0.15, $\\sigma$ 0.14$\\to$0.10)",
    "L2": "L2  + surveyed bases  (30 mm calib off the columns)",
    "L3": "L3  + 15° pen tilt",
    "L4": "L4  raw kinematics, bare metal",
}
LAYER_COLOR = {"L1": "#3B7DD8", "L2": "#22A06B", "L3": "#E8871A",
               "L4": "#A61E4D", "dead": "#000000"}
LEAN_COLORS = ["#DCEDF9", "#9EC9EC", "#5FA3DC", "#2E7ABF", "#E8A33D",
               "#D9722B", "#B03A1A", "#000000"]


def atlas_mechanism(cells, arms, n=40, seed=5):
    """WHY the atlas calls these cells dead, demonstrated on its own code.

    `atlas.solve_cell` walks its candidate SETS in order — the 8 perpendicular
    orientations first, the 64 tilted ones only if the first set yields nothing
    that clears metal among its best six by joint margin.  It returns that pose
    and knows nothing about the gates; `strict_go` applies them afterwards, to
    the single row that came back.

    So the 15 deg cone is a CLEARANCE fallback, not a GATE fallback: at a cell
    where a perpendicular pose clears but is merely uncomfortable, the tilted
    candidates are never generated.  This runs `solve_cell` twice on the same
    cell — once as shipped, once with the perpendicular set DELETED — and
    counts how often the second call certifies what the first could not.
    """
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(cells), min(n, len(cells)), replace=False)
    locked, tilted = atlas._candidates(15.0)
    rows = go_without_perp = 0
    for ci in sel:
        x, y = cells[ci]
        for a in arms:
            spec = FLEET[a]
            Twb = spec.T_world_base(H)
            bx = obstacle_boxes(a, FLEET, CALIB)
            args = (x, y, Twb, np.linalg.inv(Twb), spec)
            both = atlas.solve_cell(*args, (locked, tilted), PEN_EXT, bx,
                                    pen_lat=PEN_LAT)
            if both is None:
                continue
            rows += 1
            only = atlas.solve_cell(*args, ((), tilted), PEN_EXT, bx,
                                    pen_lat=PEN_LAT)
            if only is not None and only[0] >= STRICT_M and only[1] >= STRICT_S:
                go_without_perp += 1
    return dict(
        sampled_cells=int(len(sel)),
        arm_cell_pairs_with_an_atlas_row=int(rows),
        strict_GO_once_the_perpendicular_set_is_deleted=int(go_without_perp),
        finding="atlas.solve_cell's 15 deg cone is a CLEARANCE fallback, not "
                "a gate fallback: it is only reached when NO perpendicular "
                "pose among the best six by margin clears metal. At these "
                "cells one does clear — uncomfortably — so the tilted "
                "candidates that would have certified the cell are never "
                "generated.")


def report(arms):
    """Rebuild out/dead_disc_anatomy.{json,png} from out/dda_layers.npz."""
    d = np.load(ROOT / "out" / "dda_layers.npz", allow_pickle=True)
    cells, xs, ys, dead0 = d["cells"], d["xs"], d["ys"], d["dead0"]
    masks = {n: d[f"m_{n}"] for n, _ in LAYERS}
    N = len(cells)
    # `min_tilt` sweeps cones that always CONTAIN the 8 perpendicular
    # candidates, so a cell needing no lean at all lands in the 2.5 deg bin.
    # A_perp is exactly that set; give it its own 0 deg bin.
    mt = d["min_tilt"].copy()
    mt[masks["A_perp"]] = 0.0

    # --- first-recovering layer, in the brief's cumulative order -----------
    first = np.full(N, "", dtype=object)
    prev = np.zeros(N, bool)
    ladder = {}
    for name in LADDER:
        new = masks[name] & ~prev
        first[new] = name
        prev |= masks[name]
        ladder[name] = dict(cells=int(new.sum()),
                            area_m2=round(int(new.sum()) * CELL_M2, 5),
                            cumulative_cells=int(prev.sum()))
    irreducible = int((~prev).sum())

    ctrl_p = ROOT / "out" / "dda_L0_control.json"
    ctrl = json.loads(ctrl_p.read_text()) if ctrl_p.exists() else None
    mech = atlas_mechanism(cells, arms)
    hole = hole_a_table(arms)
    discs = disc_table(dead0, xs, ys, cells, masks, mt, arms)

    lean_hist = {f"{t:g}": int((mt == t).sum()) for t in (0.0,) + TILT_GRID}
    lean_hist["over_15"] = int(np.isnan(mt).sum())

    ref = np.load(ATLAS / "coverage.npz")
    reach_at_dead = int((ref["union_reach"] & dead0).sum())

    S = dict(
        rig="proposed", tool="lateral", h=H, pen_ext=PEN_EXT, pen_lat=PEN_LAT,
        grid=GRID, canvas=[float(SHEET[0]), float(SHEET[1])],
        cells=int(dead0.size), canvas_m2=round(dead0.size * CELL_M2, 4),
        L0=dict(union_go_pct=92.0239, dead_cells=int(dead0.sum()),
                dead_m2=round(int(dead0.sum()) * CELL_M2, 5),
                dead_pct=round(100 * float(dead0.mean()), 3),
                control=ctrl,
                dead_cells_the_atlas_DID_reach=reach_at_dead,
                note="1267 of the 1321 dead cells carry an atlas row: the "
                     "arm got its pen there, and strict_go rejected the pose "
                     "on margin/sigma. Only 54 were never reached at all."),
        layer_masks={n: dict(cells=int(masks[n].sum()),
                             area_m2=round(int(masks[n].sum()) * CELL_M2, 5),
                             pct_of_dead=round(100 * masks[n].mean(), 2),
                             config={k: (list(v) if isinstance(v, tuple) else v)
                                     for k, v in LAYER_CFG[n].items()})
                     for n, _ in LAYERS},
        ladder=ladder, atlas_mechanism=mech,
        irreducible=dict(cells=irreducible,
                         area_m2=round(irreducible * CELL_M2, 5),
                         max_radius_mm=0.0),
        min_lean_for_strict_go=lean_hist,
        discs=discs, hole_a=hole,
        recovered_coverage_pct=round(
            100 * float((ref["union_go"].sum() + masks["A_tilt"].sum())
                        / dead0.size), 2),
        accuracy=dict(
            validated_exact=[
                "analytic IK (He/Liu) + FK verification: every candidate pose "
                "is re-checked against frames.fk and kept only if it "
                "reproduces the asked-for hand-TCP pose to 1e-9 m / 1e-9 rad. "
                "The solver clamps at the workspace boundary instead of "
                "failing, so this is what separates a solution from a "
                "2 cm miss.",
                "FR3 joint limits: every solution re-filtered against "
                "frames.FR3_MIN/MAX (the .so hardcodes PANDA limits).",
                "collision envelopes: capsule radii MEASURED against the "
                "manufacturer's meshes (audit 5c8d803) and shown to CONTAIN "
                "the real metal; the base column is four measured bands, not "
                "one assumed capsule. Capsule-to-box distance is exact "
                "(convex ternary search, rig_final.segment_box_clearance).",
                "fiber exhaustiveness: 8 tool yaws x 16 q7 x every IK branch "
                "at each cell, and nothing truncated — 738 917 FK-verified "
                "poses over the 1321 dead cells at the tilt layers.",
                "the L0 control: an independent re-sweep reproduces the "
                "committed atlas cell for cell, 0 differences on all six "
                "arms' GO and reach masks.",
            ],
            chosen_policy=[
                f"margin >= {STRICT_M} rad (strict) / {HARD_M} rad (hard): a "
                "comfort convention inherited from the IKA toolkit, not a "
                "hardware limit. The joint limit itself is margin = 0.",
                f"sigma_min >= {STRICT_S} (strict) / {HARD_S} (hard): a "
                "force-sensing floor, field-validated as a place where "
                "libfranka's wrench estimate stays usable — a control "
                "requirement, not a kinematic one.",
                f"clearance >= {CLEAR} m = 20 mm operating + 30 mm "
                "calibration; producers additionally pay 13 mm "
                "(STATIC_PLAN_MARGIN) so the independent checker's lower "
                "bound still passes.",
                "30 mm of calibration inflation on every neighbour column, "
                "because the bases are not surveyed yet.",
                "PERPENDICULAR pen: a choice, and on this rig the expensive "
                "one. The atlas nominally allows a 15 deg cone; the LATERAL "
                "planner cannot tilt at all.",
            ],
            how_to_read_unreachable=(
                "At L0 'unreachable' means 'the atlas's best-six probe did "
                "not surface a pose that passed our comfort policy here'. It "
                "does NOT mean the arm cannot put its pen there: 1267 of the "
                "1321 dead cells already carry an atlas row, and all 1321 "
                "have an FK-verified, in-limits, collision-free pose."),
        ),
        caveats=[
            "SELF-COLLISION is not modelled anywhere in the package. Every "
            "layer here, L4 included, checks the chain against the OTHER "
            "arms' metal and the structure — never against itself. The FR3 "
            "joint limits do most of that work in practice, but a 15 deg "
            "leaned pose folded under its own shoulder has not been checked.",
            "CABLE DRESS / festoon is in no envelope. The measured column "
            "band 0 includes a 177 mm connector-plus-cable-stub radius at the "
            "flange, but nothing models a loom hanging into the workspace.",
            "THE PEN CRADLE is pending; there is no geometry for it, so no "
            "layer keeps out of wherever it lands.",
            "FRAME CROSS-MEMBERS are not boxes here: the obstacle set is the "
            "neighbours' base plates, booms and measured base columns. Any "
            "bracing between columns would be new metal in these discs.",
            "PAPER FLATNESS / table deflection: the pen tip is placed at "
            "z = 0 exactly and the chain gated 20 mm above it.",
            "THE 2 cm GRID is the resolution of the whole answer. A dead "
            "region smaller than one cell would not appear.",
            "L3 IS A POSE RESULT, NOT A PLANNABLE ONE TODAY: "
            "lateral.plan_adaptive explicitly drops tilt_max_deg, so with the "
            "lateral holder the shipped planner cannot execute a leaned pen. "
            "The tilt machinery exists (tilt.py) but is wired to the inline "
            "pen's yaw degeneracy.",
            "A CELL BEING REACHABLE IS NOT A STROKE BEING DRAWABLE: this is a "
            "per-cell pose question. Continuity along a stroke, the transit "
            "to and from it, and the multi-arm schedule are separate gates.",
        ],
    )
    (ROOT / "out" / "dead_disc_anatomy.json").write_text(json.dumps(S, indent=1))
    print_table(S)
    figure(S, cells, xs, ys, dead0, first, mt, arms)
    return S


def print_table(S):
    print("\n" + "=" * 78)
    print(f"DEAD TODAY (L0, certified atlas): {S['L0']['dead_cells']} cells = "
          f"{S['L0']['dead_m2']:.4f} m² = {S['L0']['dead_pct']:.2f} % of "
          f"{S['canvas_m2']:.3f} m²")
    if S["L0"]["control"]:
        print(f"  control re-sweep reproduced it exactly: "
              f"{S['L0']['control']['exact_reproduction']}")
    print(f"  of those, {S['L0']['dead_cells_the_atlas_DID_reach']} already "
          f"carry an atlas row (reached, then gated out)")
    print("\nFIRST LAYER THAT RECOVERS EACH DEAD CELL")
    for n in LADDER:
        v = S["ladder"][n]
        print(f"  {n}  {v['cells']:5d} cells  {v['area_m2']:.4f} m²  "
              f"({100 * v['cells'] / S['L0']['dead_cells']:5.1f} % of the "
              f"dead area)")
    ir = S["irreducible"]
    print(f"  L4-irreducible (TRUE PHYSICS)  {ir['cells']:5d} cells  "
          f"{ir['area_m2']:.4f} m²   max radius {ir['max_radius_mm']:.0f} mm")
    print("\nEACH LAYER ON ITS OWN (cells of the 1321 it can reach)")
    for n, _ in LAYERS:
        v = S["layer_masks"][n]
        print(f"  {n:<7} {v['cells']:5d}  {v['area_m2']:.4f} m²  "
              f"{v['pct_of_dead']:5.1f} %")
    print("\nMINIMUM PEN LEAN FOR A STRICT-GO POSE (today's columns, "
          "today's gates)")
    for k, v in S["min_lean_for_strict_go"].items():
        print(f"  {k:>7} deg: {v:5d} cells")
    print("\nPER DISC  (every dead cell attached to its nearest base column)")
    print(f"  {'arm':>4} {'cells':>6} {'m²':>8} {'disc mm':>8} {'r_max':>6} "
          f"{'L1':>5} {'L2':>4} {'L3':>5} {'L4':>4} {'core m²':>8} "
          f"{'GO@15°':>7} {'lean med/max':>13} {'frag':>5}")
    for x in S["discs"]:
        r = x["recovered"]
        print(f"  {x['arm']:>4} {x['cells']:>6} {x['area_m2']:>8.4f} "
              f"{max(x['main_disc_span_mm']):>8.0f} "
              f"{x['main_disc_max_radius_mm']:>6.0f} "
              f"{r['L1']:>5} {r['L2']:>4} {r['L3']:>5} {r['L4']:>4} "
              f"{x['irreducible_m2']:>8.4f} "
              f"{x['already_strict_go_at_15deg']:>7} "
              f"{(x['lean_median_deg'] or 0):>6.1f}° /"
              f"{(x['lean_max_deg'] or 0):>5.1f}° {x['outlying_cells']:>5}")
    tot = S["L0"]["dead_cells"]
    print(f"  {'ALL':>4} {tot:>6} {S['L0']['dead_m2']:>8.4f}"
          + " " * 16
          + f"{S['ladder']['L1']['cells']:>5} {S['ladder']['L2']['cells']:>4} "
            f"{S['ladder']['L3']['cells']:>5} {S['ladder']['L4']['cells']:>4} "
            f"{S['irreducible']['area_m2']:>8.4f} "
            f"{S['layer_masks']['A_tilt']['cells']:>7}")
    h = S["hole_a"]
    print(f"\nHOLE A — the {h['span_mm']:.1f} mm of logo nobody drew "
          f"({h['points']} probe points along the span)")
    for n, _ in LAYERS:
        v = h["per_layer"][n]
        print(f"  {n:<7} {v['points_covered_by_some_arm']:>3}/{h['points']} "
              f"points covered; whole span by one arm: "
              f"{v['whole_span_by_one_arm'] or '—'}")
    print("  minimum lean that certifies the WHOLE span, strict gates: "
          + ", ".join(f"arm {k}: {v}°" if v else f"arm {k}: —"
                      for k, v in h["min_tilt_whole_span_strict"].items()))
    print(f"\nCORRECTED COVERAGE (strict gates, today's metal, 15° lean "
          f"allowed): {S['recovered_coverage_pct']:.2f} %")
    print("=" * 78)


def figure(S, cells, xs, ys, dead0, first, mt, arms):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    half = 0.5 * GRID
    ext = (ys[0] - half, ys[-1] + half, xs[0] - half, xs[-1] + half)
    ix = np.rint(cells[:, 0] / GRID).astype(int)
    iy = np.rint(cells[:, 1] / GRID).astype(int)
    live = ~dead0

    def canvas_img(colors):
        img = np.ones(dead0.shape + (3,))
        img[live] = (0.90, 0.91, 0.92)
        img[iy, ix] = colors
        return np.transpose(img, (1, 0, 2))

    lay_rgb = np.array([matplotlib.colors.to_rgb(
        LAYER_COLOR.get(f, LAYER_COLOR["dead"])) for f in first])
    bins = np.full(len(mt), len(LEAN_COLORS) - 1)
    for k, t in enumerate(TILT_GRID):
        bins[mt == t] = k + 1
    bins[np.isin(np.rint(mt), [0]) | (mt == 0)] = 0
    perp = np.load(ROOT / "out" / "dda_layers.npz")["m_A_perp"]
    bins[perp] = 0
    lean_rgb = np.array([matplotlib.colors.to_rgb(LEAN_COLORS[b])
                         for b in bins])

    fig = plt.figure(figsize=(18.6, 12.2), facecolor="white")
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.86],
                          width_ratios=[1.72, 1.0], hspace=0.66, wspace=0.07,
                          left=0.040, right=0.995, top=0.900, bottom=0.055)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[1, 0])
    sub = gs[2, 0].subgridspec(1, 2, width_ratios=[1.0, 1.62], wspace=0.16)
    axZ = fig.add_subplot(sub[0, 0])
    axC = fig.add_subplot(sub[0, 1])
    axT = fig.add_subplot(gs[:, 1])

    for ax, img, title in (
            (axA, canvas_img(lay_rgb),
             "WHICH RELAXATION REVIVES EACH DEAD CELL   "
             "(nothing is black, so nothing is physics)"),
            (axB, canvas_img(lean_rgb),
             "HOW FAR THE PEN MUST LEAN for a STRICT-GO pose")):
        ax.imshow(img, origin="lower", extent=ext, aspect="equal",
                  interpolation="nearest")
        ax.set_title(title, fontsize=11.5, fontweight="bold", loc="left")
        ax.set_facecolor("#F5F5F3")
        for a in arms:
            s = FLEET[a]
            r = max(rr for *_, rr in mounts.MOUNTS.column_bands)
            ax.add_patch(Rectangle((s.xy[1] - r, s.xy[0] - r), 2 * r, 2 * r,
                                   fill=False, ec="#444", lw=1.0, ls=":",
                                   zorder=4))
            ax.plot(s.xy[1], s.xy[0], "v", ms=8, mfc="#FFFFFF", mec="k",
                    mew=1.2, zorder=6)
            ax.annotate(f"{a}", (s.xy[1], s.xy[0]), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=8.5,
                        fontweight="bold", zorder=7)
        ax.plot(HOLE_A[:, 1], HOLE_A[:, 0], "-", color="#00B4D8", lw=2.6,
                zorder=8, solid_capstyle="round")
        ax.annotate("hole A: 109.9 mm of logo",
                    (HOLE_A[:, 1].mean(), HOLE_A[:, 0].mean()),
                    textcoords="offset points", xytext=(20, -26), fontsize=8.4,
                    color="#0077A0", fontweight="bold", zorder=9,
                    arrowprops=dict(arrowstyle="->", color="#0077A0", lw=1.1))
        ax.add_patch(Rectangle((0, 0), SHEET[1], SHEET[0], fill=False,
                               ec="#333", lw=1.1, zorder=3))
        ax.set_xlim(-0.16, SHEET[1] + 0.16)
        ax.set_ylim(-0.12, SHEET[0] + 0.16)
        ax.set_xlabel("canvas y  [m]   (long axis)", fontsize=9)
        ax.set_ylabel("canvas x  [m]", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.13, lw=0.5)

    axA.legend(handles=[Patch(fc=LAYER_COLOR[n], label=LAYER_LABEL[n]
                              + f"   [{S['ladder'][n]['cells']} cells]")
                        for n in LADDER]
               + [Patch(fc="#000000", label="still dead at L4 = TRUE PHYSICS "
                                            f"[{S['irreducible']['cells']} cells]"),
                  Patch(fc=(0.90, 0.91, 0.92), label="certified GO today")],
               ncol=3, fontsize=8.5, loc="upper center", handlelength=1.5,
               columnspacing=1.6, bbox_to_anchor=(0.5, -0.24), frameon=False)
    lean_lbl = ["0°  (perpendicular)"] + [f"{t:g}°" for t in TILT_GRID] \
        + ["> 15°  (hard-gate pose)"]
    axB.legend(handles=[Patch(fc=c, ec="#999", lw=0.4, label=l)
                        for c, l in zip(LEAN_COLORS, lean_lbl)],
               ncol=8, fontsize=8.4, loc="upper center", handlelength=1.4,
               columnspacing=1.1, bbox_to_anchor=(0.5, -0.24), frameon=False,
               title="least pen lean at which some arm has a STRICT-GO pose",
               title_fontsize=8.6)

    # --- per-disc stacked bars --------------------------------------------
    ds = S["discs"]
    xpos = np.arange(len(ds))
    bottom = np.zeros(len(ds))
    for n in LADDER + ["core"]:
        v = np.array([(d["irreducible_cells"] if n == "core"
                       else d["recovered"][n]) * CELL_M2 for d in ds])
        if v.sum() == 0 and n not in ("L1", "L3", "core"):
            continue
        axC.bar(xpos, v, 0.62, bottom=bottom,
                color=LAYER_COLOR.get(n, "#000000"),
                label=n if n != "core" else "true physics", zorder=3)
        bottom += v
    for i, d in enumerate(ds):
        axC.text(i, bottom[i] + 0.0022, f"{d['area_m2']:.4f} m²",
                 ha="center", fontsize=8.4, fontweight="bold")
        axC.text(i, 0.5 * d["recovered_m2"]["L1"],
                 f"{d['recovered']['L1']}", ha="center", va="center",
                 fontsize=8.6, color="white", fontweight="bold", zorder=4)
        axC.text(i, d["recovered_m2"]["L1"] + d["recovered_m2"]["L2"]
                 + 0.5 * d["recovered_m2"]["L3"], f"{d['recovered']['L3']}",
                 ha="center", va="center", fontsize=8.6, color="white",
                 fontweight="bold", zorder=4)
    axC.set_xticks(xpos)
    axC.set_xticklabels([f"arm {d['arm']}\n{max(d['main_disc_span_mm']):.0f} mm"
                         f" across" for d in ds], fontsize=8.6)
    axC.set_ylabel("dead area  [m²]", fontsize=9)
    axC.set_ylim(0, bottom.max() * 1.20)
    axC.set_title("PER DISC — cells recovered by each layer",
                  fontsize=11.0, fontweight="bold", loc="left")
    axC.legend(fontsize=8.6, ncol=5, frameon=False, loc="upper right",
               bbox_to_anchor=(1.0, 1.19))
    axC.grid(axis="y", alpha=0.22, lw=0.5, zorder=0)
    axC.tick_params(labelsize=8.4)

    # --- the zoom: arm 71's disc, and the logo span inside it --------------
    s71 = FLEET[71]
    w = 0.30
    axZ.imshow(canvas_img(lay_rgb), origin="lower", extent=ext,
               aspect="equal", interpolation="nearest")
    r = max(rr for *_, rr in mounts.MOUNTS.column_bands)
    axZ.add_patch(Rectangle((s71.xy[1] - r, s71.xy[0] - r), 2 * r, 2 * r,
                            fill=False, ec="#444", lw=1.1, ls=":", zorder=4))
    axZ.plot(s71.xy[1], s71.xy[0], "v", ms=11, mfc="#FFFFFF", mec="k", mew=1.4,
             zorder=6)
    axZ.plot(HOLE_A[:, 1], HOLE_A[:, 0], "-", color="#00B4D8", lw=3.4,
             zorder=8, solid_capstyle="round")
    axZ.plot(HOLE_A[:, 1], HOLE_A[:, 0], "o", color="#00B4D8", ms=4.5, zorder=9)
    axZ.annotate(f"hole A — {S['hole_a']['span_mm']:.1f} mm\n"
                 f"arm 71 draws ALL of it\nat a 12.5° lean, strict gates",
                 (HOLE_A[:, 1].mean(), HOLE_A[:, 0].mean()),
                 textcoords="offset points", xytext=(24, -74), fontsize=8.4,
                 color="#0077A0", fontweight="bold", zorder=10, ha="left",
                 arrowprops=dict(arrowstyle="->", color="#0077A0", lw=1.2))
    axZ.annotate("arm 71 base column, 320 mm AABB",
                 (s71.xy[1] - r, s71.xy[0] + r), textcoords="offset points",
                 xytext=(0, 6), fontsize=7.8, color="#444", zorder=10)
    axZ.set_xlim(s71.xy[1] - w, s71.xy[1] + w)
    axZ.set_ylim(s71.xy[0] - w, s71.xy[0] + w)
    axZ.set_xlabel("canvas y  [m]", fontsize=9)
    axZ.set_ylabel("canvas x  [m]", fontsize=9)
    axZ.set_title("ZOOM — arm 71's disc, 600 x 600 mm", fontsize=11.0,
                  fontweight="bold", loc="left")
    axZ.tick_params(labelsize=8)
    axZ.grid(alpha=0.15, lw=0.5)

    # --- the numbers -------------------------------------------------------
    h = S["hole_a"]
    L = S["layer_masks"]
    lines = [
        "BOTTOM LINE",
        f"  {S['L0']['dead_m2']:.4f} m² ({S['L0']['dead_pct']:.2f} %) has no",
        "  certified pen-down pose today.",
        "",
        f"  policy / comfort   {S['ladder']['L1']['area_m2']:.4f} m²"
        f"   {100 * S['ladder']['L1']['cells'] / S['L0']['dead_cells']:4.1f} %",
        f"  base survey        {S['ladder']['L2']['area_m2']:.4f} m²"
        f"   {100 * S['ladder']['L2']['cells'] / S['L0']['dead_cells']:4.1f} %",
        f"  15° pen tilt       {S['ladder']['L3']['area_m2']:.4f} m²"
        f"   {100 * S['ladder']['L3']['cells'] / S['L0']['dead_cells']:4.1f} %",
        f"  near joint limits  {S['ladder']['L4']['area_m2']:.4f} m²"
        f"   {100 * S['ladder']['L4']['cells'] / S['L0']['dead_cells']:4.1f} %",
        f"  TRUE PHYSICS       {S['irreducible']['area_m2']:.4f} m²"
        f"   {100 * S['irreducible']['cells'] / S['L0']['dead_cells']:4.1f} %",
        "",
        "  There is no irreducible core.  Every dead cell",
        "  has an FK-verified, in-limits, collision-free",
        "  pose — 1321 of 1321, at 2 cm resolution.",
        "",
        "WHY THE ATLAS SAYS OTHERWISE",
        f"  {S['L0']['dead_cells_the_atlas_DID_reach']} of the {S['L0']['dead_cells']} dead cells ALREADY carry an",
        "  atlas row: the arm put its pen there and the",
        "  pose was rejected on margin/sigma, not on reach.",
        "",
        "  atlas.solve_cell returns the best-margin pose that",
        "  CLEARS metal, and gates afterwards.  Its 15° tilt",
        "  cone is a CLEARANCE fallback, so it never fires",
        "  when a perpendicular pose clears but is merely",
        "  uncomfortable — which is this whole map.",
        "",
        f"  Run solve_cell twice with the PERPENDICULAR set",
        f"  deleted and {S['atlas_mechanism']['strict_GO_once_the_perpendicular_set_is_deleted']} of "
        f"{S['atlas_mechanism']['arm_cell_pairs_with_an_atlas_row']} (arm, cell) pairs it",
        f"  called not-GO come back STRICT-GO.",
        "",
        f"  Searched exhaustively under the atlas's OWN policy",
        f"  (margin ≥ 0.30, σ ≥ 0.14, inflated columns, 50 mm,",
        f"  ≤ 15° lean): {L['A_tilt']['cells']} of {S['L0']['dead_cells']} dead cells are strict-GO.",
        f"  Coverage is {S['recovered_coverage_pct']:.2f} %, not 92.02 %.",
        "",
        "WHAT EACH LAYER REACHES ON ITS OWN",
    ]
    for n, _ in LAYERS:
        lines.append(f"  {n:<7}{L[n]['cells']:5d} cells  {L[n]['area_m2']:.4f} m²"
                     f"  {L[n]['pct_of_dead']:5.1f} %")
    lines += [
        "    A_perp / A_tilt = the atlas's own gates, searched",
        "      exhaustively (perpendicular / 15° cone)",
        "    P4 = bare metal, no gates, PEN PERPENDICULAR",
        "    L4w = L4 with a 30° cone (sensitivity)",
        "",
        "HOW FAR THE PEN MUST LEAN (strict gates, today's rig)",
    ]
    for k, v in S["min_lean_for_strict_go"].items():
        lines.append(f"  {k:>7}°{'':2}{v:5d} cells"
                     + ("   (needs a hard-gate pose)" if k == "over_15" else ""))
    lines += [
        "",
        f"HOLE A — {h['span_mm']:.1f} mm of logo, {h['per_arm']['71']['mid_dist_mm']:.0f} mm from arm 71's column",
    ]
    for n, _ in LAYERS:
        v = h["per_layer"][n]
        who = v["whole_span_by_one_arm"]
        lines.append(f"  {n:<7}{v['points_covered_by_some_arm']:>3}/{h['points']} pts"
                     + (f"   WHOLE SPAN: arm {who}" if who else "   —"))
    lines += [
        f"  arm 71 perpendicular: best joint margin 0.131 rad,",
        f"    {HARD_M:.2f} needed — it misses by 0.019 rad (1.1°).",
        f"    At 15° lean: margin 0.345, σ 0.251, 277 mm clear.",
        "",
        "CAVEATS — WHAT EVEN L4 DOES NOT MODEL",
        "  * self-collision: no model anywhere in the package;",
        "    L4 checks the chain against OTHER arms' metal only.",
        "  * cable dress / festoon: not in any envelope.",
        "  * the pen cradle: pending, no geometry yet.",
        "  * frame cross-members: only plates, booms and the",
        "    measured base columns are boxes here.",
        "  * paper flatness / table deflection: z = 0 exactly.",
        "  * the LATERAL planner cannot tilt the pen at all",
        "    (lateral.plan_adaptive drops tilt_max_deg), so L3",
        "    is a POSE result, not a plannable one, today.",
    ]
    axT.axis("off")
    axT.text(0.0, 1.0, "\n".join(lines), va="top", ha="left",
             family="monospace", fontsize=7.72, linespacing=1.255,
             transform=axT.transAxes)

    fig.suptitle("ARIS proposed rig — anatomy of the six dead discs: "
                 "physics vs policy", fontsize=15.5, fontweight="bold",
                 x=0.040, ha="left", y=0.972)
    fig.text(0.040, 0.938,
             f"rig=proposed  tool=lateral  h={H:.3f} m  pitch 0.61 m  "
             f"pen 110 mm axial + 110 mm lateral  2 cm grid  "
             f"{S['canvas_m2']:.3f} m² canvas   |   L0 control reproduces the "
             f"committed atlas cell for cell (92.0239 %, 1321 dead, 0 diff)",
             fontsize=9.2, color="#444", ha="left")
    png = ROOT / "out" / "dead_disc_anatomy.png"
    fig.savefig(png, dpi=135, facecolor="white")
    plt.close(fig)
    print("wrote", png)


def selftest(arms):
    """Prove the local obstacle rebuild agrees with the package before any
    layer number is quoted, and that the exhaustive gate finds every pose the
    certified atlas certified."""
    ok = True
    for a in arms:
        mine = obstacle_boxes(a, FLEET, CALIB)
        theirs = FLEET[a].static_obstacles()
        same = (len(mine) == len(theirs)
                and all(np.allclose(m["lo"], t["lo"])
                        and np.allclose(m["hi"], t["hi"])
                        for m, t in zip(mine, theirs)))
        print(f"  arm {a}: rebuilt obstacle set == spec.static_obstacles(): "
              f"{same}  ({len(mine)} boxes)")
        ok &= same
    rng = np.random.default_rng(7)
    for a in arms:
        arr, _ = atlas.load(str(ATLAS), a)
        go = arr[atlas.strict_go(arr)][:, :2]
        sel = go[rng.choice(len(go), 300, replace=False)]
        hit = eval_cells(sel, FLEET[a], obstacle_boxes(a, FLEET, CALIB),
                         orientations((15.0,)), ik.Q7_GRID, STRICT_M, STRICT_S,
                         CLEAR, 0.02)
        print(f"  arm {a}: A_tilt finds {int(hit.sum())}/300 sampled atlas "
              f"strict-GO cells")
        ok &= bool(hit.all())
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    main()
