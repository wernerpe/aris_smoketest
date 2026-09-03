#!/usr/bin/env python3
"""LAYOUT RE-SCORE — the same six arms, judged with the fleet IN THE ROOM.

WHY THIS EXISTS.  `scripts/layout_study.py` chose `layout.LAYOUT_PROPOSED` (six
inverted arms on a 2 x 3 grid, transverse pitch 0.61 m, h = 0.85) by maximising
the union of the six arms' certified strict-GO sets — measured ONE ARM AT A
TIME, in empty air.  That objective is wrong in a way the study could not see:
`atlas.sweep_arm` gates a cell on joint margin, sigma, the paper plane and the
static boxes ABOVE the mount plate, and on nothing that hangs BELOW it.  Every
neighbour has something hanging below it in every configuration it will ever
hold — the flange-to-shoulder column, a capsule from z = h - 0.333 to z = h at
its base XY, which is pose-INVARIANT.  An arm reaching under its transverse
partner has to get its own shoulder past that column, and at a 0.61 m pitch it
often cannot.

So this script re-scores the layout FAMILY under the corrected objective:

  air       nothing in the room but the arm itself      (the published number)
  mounts    every OTHER arm reduced to its base CYLINDER — the CEILING on what
            any park policy whatever can buy back, because that capsule is
            there whatever the neighbour is doing
  parked    every other arm at a certified park set, if one is asked for

plus a CONCURRENCY proxy — for each of the fifteen arm pairs, the fraction of
(certified pose a, certified pose b) combinations that clear the 80 mm the
conductor holds every pair to.  Coverage says whether the canvas can be drawn;
concurrency says how much of it can be drawn AT THE SAME TIME.

and a third scenario, `box`, which is not a model of the physics but a model of
the PACKAGE: commit 14b01cd put the same column into `mounts.arm_column_boxes`
an axis-aligned box at `column_r` = 0.12 judged against `STATIC_MARGIN` = 0.05.
That box is a conservative envelope of the capsule (square where the capsule is
round, and its caps stand 0.12 m proud of both ends), so it refuses strictly
more — and it is what the atlas, the planner and the allocator will now certify
against, so a layout has to be good under both.

THE SCORING IS SELF-CONTAINED AND FULL-FIBER.  There is no coarse disc proxy
and no cached atlas: every candidate pays the whole 8 tool-yaws x 16 q7 x every
IK branch fiber at every 2 cm cell in reach, gated exactly as
`atlas.solve_cell` gates it (joint margin, the 9-point paper plane, the legacy
r = 0.12 own-boom proxy, the mount steel, sigma) and then, per scenario,
against the neighbours' occupancy.  ~90 s an arm, ~10 min for 29 layouts on 24
workers —
cheap enough that a coarse stage would only add a way to be wrong.  A locked
(perpendicular) pen fiber is used, no tilt-cone rescue: that is what
`out/probe_cover.py` used to reproduce the published 99.98 % union exactly, and
`--check` re-verifies BOTH pins before anything is swept.

SELF-CONTAINED MEANS THE OBSTACLES TOO, and that is not fastidiousness.  This
file first ran against `spec.static_obstacles()`; commit 14b01cd landed the
body columns inside it mid-sweep, and every "empty air" number silently became
an occupancy-corrected one — air and mounts came out identical and the whole
comparison was meaningless.  So the steel (`_steel`) and the column (`_column`,
`_column_box`) are rebuilt here from `mounts.arm_mount_boxes` and from the base
pose, and a scenario is a set this file chose.  `--check` is what caught it.

READ-ONLY on the package.  Everything here is new; nothing in `aris_sixarm/` is
touched.

WHAT IT FOUND (2026-08-25, 54 candidates).  The grid is not the problem and no
grid variant fixes it: the adopted figure ranks 6th of 54 on empty air and 23rd
on the corrected union, and every re-arrangement at h = 0.85 — tighter pitch,
wider pitch, staggered columns, 3 x 2, a single centre line — trades one metric
for another.  HEIGHT does not.  At h = 0.85 an arm must FOLD to reach the
paper, and folding is what lifts its elbow back into the [h - 0.333, h] band
its neighbour's base column occupies; the same grid at h = 0.90-0.95 makes the
same reach with a straighter arm whose chain hangs below that band.  Same six
base positions, one number changed, and the occupancy penalty goes to zero.

Run:
    ARIS_TOOL=lateral python3 scripts/layout_rescore.py --check
    ARIS_TOOL=lateral python3 scripts/layout_rescore.py --jobs 30
    ARIS_TOOL=lateral python3 scripts/layout_rescore.py --jobs 30 --pairs \
        --only adopted,zig061_dy605
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("ARIS_TOOL", "lateral")     # the study's settled tool

from aris_sixarm import ik, layout, metrics, mounts, rig_final  # noqa: E402
from aris_sixarm.coordination import (ArmPath, CAPSULES_LAT,    # noqa: E402
                                      LINK_R, cap_endpoints,
                                      clearance_matrix, seg_seg_dist)
from aris_sixarm.frames import (PEN_EXT_HOLDER, PEN_LAT_HOLDER,  # noqa: E402
                                fk_many, joint_margin, rotx, rotz,
                                tool_offset, tool_points_many)
from aris_sixarm.metrics import GATE_MARGIN, GATE_SIGMA         # noqa: E402
from aris_sixarm.rig_final6 import SHEET_FINAL6                 # noqa: E402

W, H = SHEET_FINAL6
GRID = 0.02
PEN = PEN_EXT_HOLDER             # the HOLDER's axial pen extension.  It was
                                 # the literal 0.110 — the inline pen's — while
                                 # the two agreed; the holder's own is
                                 # 0.0588421 since 2026-09-03 (frames.py).
RMAX = 1.05 + PEN_LAT_HOLDER     # the reach the atlas sweeps, lateral tool
MARGIN = 0.08                    # coordination.SAFETY_M + CALIB_M
NEAR_BASE = 0.35                 # m, the handoff annulus the report reads
YAWS = np.linspace(0, 2 * np.pi, 8, endpoint=False)
CAP_R = np.array([c[2] for c in CAPSULES_LAT], float)
NCAP = len(CAPSULES_LAT)         # 11 since the base column became 4 bands
XS = np.arange(0.0, W + 1e-9, GRID)
YS = np.arange(0.0, H + 1e-9, GRID)
NCELL = len(XS) * len(YS)


# ==========================================================================
# the candidate families
# ==========================================================================
def paired(spacing, rows=3, h=0.850, dy=0.0, sheet=SHEET_FINAL6):
    """`layout.paired_grid` with a LONGITUDINAL STAGGER between the columns.

    `dy = 0` is exactly `paired_grid`.  `dy > 0` slides the left column back
    and the right column forward by dy/2 each, so the figure stays symmetric
    about the canvas centre and every arm's transverse partner moves off its
    own row.  At `dy = H / rows / 2` the two columns interleave into ONE
    evenly-spaced zig-zag of 2*rows arms — which is the point: an arm's
    nearest neighbour goes from `spacing` to hypot(spacing, dy) without the
    columns leaving the centre line.
    """
    Wd, Hs = sheet
    xs = (Wd / 2 - spacing / 2, Wd / 2 + spacing / 2)
    ys = [(2 * j + 1) * Hs / (2 * rows) for j in range(rows)]
    inv = []
    for j, y in enumerate(ys):
        inv.append((xs[0], y - dy / 2))
        inv.append((xs[1], y + dy / 2))
    return dict(floor=[], inv=inv, h=float(h))


def cols3(spacing, h=0.850, dy=0.0, sheet=SHEET_FINAL6):
    """THREE columns x TWO rows — the other way to tile six over the web."""
    Wd, Hs = sheet
    xs = (Wd / 2 - spacing, Wd / 2, Wd / 2 + spacing)
    ys = (Hs / 4, 3 * Hs / 4)
    inv = []
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            inv.append((x, y + (dy / 2 if i == 1 else -dy / 2)))
    return dict(floor=[], inv=inv, h=float(h))


def rowpitch(spacing, pitch, h=0.850, dy=0.0, sheet=SHEET_FINAL6):
    """2 x 3, rows at an EXPLICIT longitudinal pitch about the centre."""
    Wd, Hs = sheet
    xs = (Wd / 2 - spacing / 2, Wd / 2 + spacing / 2)
    ys = [Hs / 2 + (j - 1) * pitch for j in range(3)]
    inv = []
    for y in ys:
        inv.append((xs[0], y - dy / 2))
        inv.append((xs[1], y + dy / 2))
    return dict(floor=[], inv=inv, h=float(h))


DY_ZIG = H / 6.0                 # the half-row stagger: 0.6051 m


def candidates():
    """-> {name: (layout dict, family, note)} — every geometry this scores."""
    C = {}

    def add(name, lay, fam, note):
        C[name] = (lay, fam, note)

    # --- 1. the ADOPTED layout, scored under the corrected objective -------
    add("adopted", layout.paired_grid(spacing=0.61, rows=3, h=0.850),
        "adopted", "LAYOUT_PROPOSED: 2x3, pitch 0.61, h 0.85")

    # --- 2. transverse pitch sweep, no stagger ----------------------------
    # 0.40 and 0.45 are BELOW `layout.MIN_BASE_DIST` and are swept anyway,
    # reported as violations: the corrected objective turned out to want a
    # TIGHTER pitch than the air one, so where it stops mattering is a
    # question the 0.50 m workspace floor should be asked, not assumed.  The
    # HARDWARE floor is `mounts.min_column_spacing()` = 0.345 m.
    for s in (0.40, 0.45, 0.50, 0.55, 0.62, 0.65, 0.68, 0.70, 0.75, 0.80,
              0.85, 0.90, 0.95):
        add(f"pitch{int(round(100 * s)):03d}", paired(s), "pitch",
            f"2x3 grid, pitch {s:.2f}")

    # --- 3. staggered rows: the partner leaves the annulus, the columns stay
    for s in (0.61, 0.70, 0.80, 0.90):
        for dy in (0.30, 0.45, DY_ZIG):
            if dy == 0.45 and s != 0.61:
                continue            # 0.45 is a resolution point, not a family
            add(f"stag{int(round(100 * s)):03d}_dy{int(round(1000 * dy)):03d}",
                paired(s, dy=dy), "stagger",
                f"2x3, pitch {s:.2f}, column stagger {dy:.3f}")
    add("stag061_dy150", paired(0.61, dy=0.15), "stagger",
        "2x3, pitch 0.61, column stagger 0.150")

    # --- 4. row count / aspect --------------------------------------------
    for s in (0.50, 0.55, 0.62):
        add(f"cols3_{int(round(100 * s)):03d}", cols3(s), "aspect",
            f"3 columns x 2 rows, column pitch {s:.2f}")
    add("cols3_055_dy300", cols3(0.55, dy=0.30), "aspect",
        "3 columns x 2 rows, middle column staggered 0.300")
    add("line6", paired(0.0, dy=DY_ZIG), "aspect",
        "one centre-line column of six, 0.605 m apart")
    # rows pushed apart / pulled together, at the adopted pitch and at a zigzag
    for p in (0.95, 1.10, 1.35):
        add(f"rowp{int(round(100 * p)):03d}", rowpitch(0.61, p), "aspect",
            f"2x3, pitch 0.61, ROW pitch {p:.2f}")
    add("rowp110_dy605", rowpitch(0.61, 1.10, dy=DY_ZIG), "aspect",
        "2x3 zigzag, pitch 0.61, row pitch 1.10, stagger 0.605")

    # --- 5. HEIGHT, which is where the corrected objective actually bites --
    # The air objective picked h = 0.85 because the lateral tool's GO annulus
    # is widest there.  Occupancy asks a different question: at a LOW mount
    # plane an arm has to fold to reach the paper, and folding is what lifts
    # its elbow back into the band its neighbour's base column occupies
    # ([h - 0.333, h]).  Hang the same grid higher and the same reach is made
    # with a straighter arm, whose chain hangs BELOW the column band.
    for h in (0.875, 0.900, 0.925, 0.950, 1.000):
        add(f"h{int(round(1000 * h)):04d}", paired(0.61, h=h), "height",
            f"2x3 grid, pitch 0.61, h = {h:.3f}")

    # --- 6. pitch sweep AT the corrected height optimum --------------------
    for s in (0.50, 0.55, 0.65, 0.70, 0.80):
        add(f"p{int(round(100 * s)):03d}_h900", paired(s, h=0.900), "pitch@h",
            f"2x3 grid, pitch {s:.2f}, h = 0.900")

    # --- 7. the (pitch, height) plateau around the corrected optimum -------
    for s, h in ((0.62, 0.900), (0.65, 0.875), (0.65, 0.925), (0.65, 0.950),
                 (0.70, 0.925), (0.75, 0.900), (0.61, 0.922), (0.70, 0.950),
                 (0.80, 0.950)):
        add(f"p{int(round(100 * s)):03d}_h{int(round(1000 * h)):04d}",
            paired(s, h=h), "plateau",
            f"2x3 grid, pitch {s:.2f}, h = {h:.3f}")
    return C


def h_variants(base_lay, hs=(0.80, 0.90)):
    return {f"h{int(round(1000 * h)):03d}": dict(base_lay, h=float(h))
            for h in hs}


# ==========================================================================
# the gated fiber — one arm, one cell
# ==========================================================================
D1 = 0.333                       # base flange -> shoulder, modified DH
COLUMN_R = getattr(mounts.MOUNTS, "column_r", LINK_R + 0.03)


def _column(xy, h):
    """A base's pose-INVARIANT capsule: flange -> shoulder. -> (A, B, r).

    The CAPSULE model of the neighbour's body, at the conductor's own radius
    (`coordination.LINK_R`), to be compared against `MARGIN` = SAFETY + CALIB.
    This is the model `out/probe_cover.py` measured the 98.10 % ceiling with.
    """
    p0 = np.array([xy[0], xy[1], h])
    p1 = np.array([xy[0], xy[1], h - D1])
    return p0[None, :], p1[None, :], np.array([LINK_R])


def _column_box(xy, h, aid=0):
    """The same column as the BOXES the package now gates on -> box dicts.

    Built here rather than taken from `mounts.arm_column_boxes` so this study
    keeps measuring the same thing while the package's obstacle policy moves
    under it (commit 14b01cd added exactly this box to `StudySpec`).  Each
    band's AABB at its own radius, compared against `rig_final.STATIC_MARGIN`
    = 0.05 — the same statement as the capsule at `MARGIN`, but square where
    the capsule is round (up to 41 % conservative on the diagonals).

    TWO BOXES SINCE THE MESH AUDIT (2026-08-26): the connector band above the
    plate and the column proper down to the measured end of the metal.  The
    numbers come from `mounts.MOUNTS.column_bands` because they are
    MEASUREMENTS now, not a policy this study should be free to disagree with.
    """
    out = []
    for k, (z0, z1, r) in enumerate(mounts.MOUNTS.column_bands):
        p0 = np.array([xy[0], xy[1], h - z0])
        p1 = np.array([xy[0], xy[1], h - z1])
        out.append(dict(name=f"body:{aid}_column{k}", tag=f"body:{aid}",
                        lo=np.minimum(p0, p1) - r,
                        hi=np.maximum(p0, p1) + r,
                        source=f"neighbour base column band {k}, AABB at "
                               f"r = {r}"))
    return out


def _steel(fleet, aid, h):
    """Every OTHER arm's MOUNT HARDWARE only — plates, booms, pedestals.

    `spec.static_obstacles()` is no longer only steel (14b01cd folds the
    neighbours' body columns into it), and this study has to be able to ask
    the empty-air question the layout was chosen on.  So the steel is rebuilt
    here from `mounts.arm_mount_boxes`, which is still only steel.
    """
    return [b for o, s in fleet.items() if o != aid
            for b in mounts.arm_mount_boxes(s.mount, s.xy, s.yaw, h,
                                            tag=f"mount:{o}")]


def _chain_caps(q, spec, h):
    """A pose -> its capsules in world. -> (A (C,3), B (C,3), r (C,))."""
    Pw = _world_chain(np.asarray(q, float).reshape(1, 7), spec, h)
    A, B = cap_endpoints(Pw, CAPSULES_LAT)
    return A[0], B[0], CAP_R


def _world_chain(Q, spec, h):
    T, P = fk_many(Q)
    tool = tool_points_many(T, PEN, PEN_LAT_HOLDER)
    P = np.concatenate([P] + [t[:, None, :] for t in tool], axis=1)
    Twb = spec.T_world_base(h)
    return P @ Twb[:3, :3].T + Twb[:3, 3]


def _clear(A, B, others):
    """(K,C,3) capsules vs [(Ab,Bb,rb)] -> (K,) worst clearance."""
    worst = np.full(len(A), np.inf)
    for Ab, Bb, rb in others:
        d = seg_seg_dist(A[:, :, None, :], B[:, :, None, :],
                         Ab[None, None, :, :], Bb[None, None, :, :])
        d = d - CAP_R[None, :, None] - rb[None, None, :]
        worst = np.minimum(worst, d.min(axis=(1, 2)))
    return worst


def score_arm(job):
    """One arm of one layout, every scenario. -> (arm, ix, iy, ok, qbest).

    `ok` is {scenario: (n,) bool} over the cells the arm can reach at all;
    `qbest` is the max-margin certified pose per cell (NaN where none), which
    is what the atlas would have stored and what the pair-concurrency proxy
    reads.
    """
    name, lay, aid, scen_names, parks = job
    h = float(lay["h"])
    fleet = layout.build_fleet(lay)
    spec = fleet[aid]
    Twb = spec.T_world_base(h)
    Twb_inv = np.linalg.inv(Twb)
    steel = _steel(fleet, aid, h)
    body_boxes = [x for b in fleet if b != aid
                  for x in _column_box(fleet[b].xy, h, b)]
    off = tool_offset(PEN, PEN_LAT_HOLDER)

    # scenarios: `caps` are capsule obstacles judged at MARGIN, `bxs` are box
    # obstacles judged at STATIC_MARGIN.  `air` carries neither, and is the
    # objective the adopted layout was chosen on.
    scen = {}
    for nm in scen_names:
        if nm == "air":
            scen[nm] = ([], [])
        elif nm == "mounts":
            scen[nm] = ([_column(fleet[b].xy, h) for b in fleet if b != aid],
                        [])
        elif nm == "box":
            scen[nm] = ([], body_boxes)
        else:
            qp = parks[nm]
            scen[nm] = ([_chain_caps(qp[b], fleet[b], h) for b in fleet
                         if b != aid], [])

    bx, by = spec.xy
    cells, ix, iy = [], [], []
    for j, y in enumerate(YS):
        for i, x in enumerate(XS):
            if (x - bx) ** 2 + (y - by) ** 2 <= RMAX ** 2:
                cells.append((x, y))
                ix.append(i)
                iy.append(j)
    cells = np.asarray(cells, float)
    ok = {nm: np.zeros(len(cells), bool) for nm in scen_names}
    qbest = np.full((len(cells), 7), np.nan)
    t0 = time.time()
    for k, (x, y) in enumerate(cells):
        # ---- the locked fiber -------------------------------------------
        sols = []
        tip = np.array([x, y, 0.0])
        for yaw in YAWS:
            R = rotz(yaw) @ rotx(np.pi)
            T_w = np.eye(4)
            T_w[:3, :3] = R
            T_w[:3, 3] = tip - R @ off
            Tb = Twb_inv @ T_w
            for q7 in ik.Q7_GRID:
                sols.extend(ik.solve(Tb, q7, spec.q_seed))
        if not sols:
            continue
        Q = np.asarray(sols, float).reshape(-1, 7)
        # ---- the gates `atlas.solve_cell` applies, in its own order ------
        jm = np.array([joint_margin(q) for q in Q])
        keep = jm >= GATE_MARGIN
        Q, jm = Q[keep], jm[keep]
        if not len(Q):
            continue
        T, P = fk_many(Q)
        tool = tool_points_many(T, PEN, PEN_LAT_HOLDER)
        P11 = np.concatenate([P] + [t[:, None, :] for t in tool], axis=1)
        Pw = P11 @ Twb[:3, :3].T + Twb[:3, 3]
        keep = Pw[:, 1:9, 2].min(1) >= 0.02                  # the paper plane
        rb = np.hypot(P[:, :, 0], P[:, :, 1])                # own-boom proxy
        keep &= ~np.any((P[:, :, 2] < -0.02) & (rb < 0.12), axis=1)
        Q, jm, Pw = Q[keep], jm[keep], Pw[keep]
        if not len(Q):
            continue
        if steel:                                           # neighbours' steel
            keep = (rig_final.chain_static_clearance(Pw, steel)
                    >= rig_final.STATIC_MARGIN)
            Q, jm, Pw = Q[keep], jm[keep], Pw[keep]
            if not len(Q):
                continue
        A, B = cap_endpoints(Pw, CAPSULES_LAT)
        order = np.argsort(-jm)
        sig = {}
        for nm in scen_names:
            caps, bxs = scen[nm]
            gap = np.full(len(Q), np.inf) if not caps else _clear(A, B, caps)
            if bxs:
                gap = np.minimum(gap, rig_final.chain_static_clearance(Pw, bxs)
                                 - rig_final.STATIC_MARGIN + MARGIN)
            for t in order:                                 # best margin first
                if gap[t] < MARGIN:
                    continue
                if t not in sig:
                    sig[t] = metrics.sigma_min(metrics.tip_jacobian(
                        Q[t], pen_ext=PEN, pen_lat=PEN_LAT_HOLDER))
                if sig[t] >= GATE_SIGMA:
                    ok[nm][k] = True
                    if nm == "air":
                        qbest[k] = Q[t]
                    break
    return (name, aid, np.array(ix), np.array(iy),
            {nm: ok[nm] for nm in scen_names}, qbest,
            float(time.time() - t0), int(len(cells)))


# ==========================================================================
# aggregation
# ==========================================================================
def dead_stats(grid_ok, lay):
    """Where the uncovered cells are, relative to the six bases."""
    dead = ~grid_ok
    bx = np.array([p[0] for p in lay["inv"]])
    by = np.array([p[1] for p in lay["inv"]])
    X, Y = np.meshgrid(XS, YS)
    d = np.min(np.hypot(X[..., None] - bx, Y[..., None] - by), axis=2)
    near = d <= NEAR_BASE
    return dict(dead=int(dead.sum()),
                dead_near_base=int((dead & near).sum()),
                near_cells=int(near.sum()),
                near_covered_pct=float(100 * (grid_ok & near).sum()
                                       / max(near.sum(), 1)),
                dead_max_dist_to_base=float(d[dead].max()) if dead.any()
                else 0.0)


def assemble(name, lay, res, scen_names):
    """Per-arm results -> the scorecard row for one layout."""
    grids = {nm: np.zeros((len(YS), len(XS)), np.int16) for nm in scen_names}
    per_arm = {}
    for (_, aid, ix, iy, ok, _q, _t, ncell) in res:
        per_arm[aid] = {nm: int(ok[nm].sum()) for nm in scen_names}
        for nm in scen_names:
            grids[nm][iy[ok[nm]], ix[ok[nm]]] += 1
    row = dict(name=name, layout=dict(floor=[], inv=[list(map(float, p))
                                                    for p in lay["inv"]],
                                      h=float(lay["h"])),
               spacing=float(layout.pair_spacing_of(lay)),
               violations=layout.check_spacing(lay),
               n_cells=int(NCELL),
               per_arm={str(a): per_arm[a] for a in per_arm})
    for nm in scen_names:
        c = grids[nm]
        row[f"union_{nm}"] = float(100 * (c >= 1).mean())
        row[f"ge2_{nm}"] = float(100 * (c >= 2).mean())
        row[f"ge3_{nm}"] = float(100 * (c >= 3).mean())
        row[f"dead_{nm}"] = dead_stats(c >= 1, lay)
    return row, grids


# ==========================================================================
# the concurrency proxy
# ==========================================================================
def pair_clearance(name, lay, qmaps, stride=4, margin=MARGIN):
    """Per PAIR: how much of the certified work can be done AT THE SAME TIME.

    Both arms' certified (max-margin, air-gated) poses are sampled on a
    `stride * GRID` lattice of their own covered cells and every combination is
    priced with `coordination.clearance_matrix` — the very matrix the conductor
    thresholds.  `pct` is the fraction of pose pairs at or above `margin`; the
    conductor cannot schedule the rest at all, whatever the timing.
    """
    h = float(lay["h"])
    fleet = layout.build_fleet(lay)
    paths, npose = {}, {}
    for aid, (ix, iy, q) in qmaps.items():
        good = ~np.isnan(q[:, 0])
        sub = good & (ix % stride == 0) & (iy % stride == 0)
        Q = q[sub]
        npose[aid] = int(len(Q))
        paths[aid] = ArmPath(aid, Q, 0.05, h_inv=h, pen_ext=PEN,
                             spec=fleet[aid])
    ids = sorted(paths)
    rows = []
    for a, i in enumerate(ids):
        for j in ids[a + 1:]:
            D = clearance_matrix(paths[i], paths[j])
            n = D.size
            rows.append(dict(a=int(i), b=int(j), n_a=npose[i], n_b=npose[j],
                             pairs=int(n),
                             pct=float(100 * (D >= margin).sum() / max(n, 1)),
                             worst_mm=float(1000 * D.min()),
                             median_mm=float(1000 * np.median(D)),
                             base_dist=float(np.linalg.norm(
                                 np.subtract(fleet[i].xy, fleet[j].xy)))))
    pcts = np.array([r["pct"] for r in rows])
    return dict(name=name, stride=stride, margin=margin, pairs=rows,
                worst_pair_pct=float(pcts.min()),
                mean_pair_pct=float(pcts.mean()),
                n_below_90=int((pcts < 90).sum()),
                n_below_75=int((pcts < 75).sum()))


# ==========================================================================
def run(names, C, jobs, scen_names, want_pairs, out_json, out_npz, stride):
    record = json.loads(out_json.read_text()) if out_json.exists() else {}
    record.setdefault("rows", {})
    record.setdefault("pairs", {})
    record["meta"] = dict(grid=GRID, margin=MARGIN, rmax=RMAX, pen=PEN,
                          pen_lat=PEN_LAT_HOLDER, near_base=NEAR_BASE,
                          gate_margin=GATE_MARGIN, gate_sigma=GATE_SIGMA,
                          sheet=[W, H], n_cells=int(NCELL),
                          scenarios=scen_names, fiber="locked (no tilt cone)")
    todo = [(n, C[n][0]) for n in names]
    print(f"scoring {len(todo)} layouts x 6 arms on {jobs} workers "
          f"({NCELL} canvas cells, {GRID * 100:.0f} cm)", flush=True)
    jl = [(n, lay, aid, scen_names, {})
          for n, lay in todo for aid in sorted(layout.build_fleet(lay))]
    t0 = time.time()
    by_layout = {n: [] for n, _ in todo}
    with mp.get_context("fork").Pool(min(jobs, len(jl))) as pool:
        for k, r in enumerate(pool.imap_unordered(score_arm, jl)):
            by_layout[r[0]].append(r)
            print(f"  [{k + 1}/{len(jl)}] {r[0]} arm {r[1]}: {r[7]} cells "
                  f"in {r[6]:.0f} s -> "
                  + " ".join(f"{nm} {int(r[4][nm].sum())}"
                             for nm in scen_names), flush=True)
    print(f"all sweeps done in {time.time() - t0:.0f} s", flush=True)

    saved = {}
    for n, lay in todo:
        row, grids = assemble(n, lay, by_layout[n], scen_names)
        row["family"], row["note"] = C[n][1], C[n][2]
        record["rows"][n] = row
        for nm in scen_names:
            saved[f"{n}__{nm}"] = grids[nm].astype(np.int8)
        print(f"{n:>18}: air {row['union_air']:7.3f} %  mounts "
              f"{row['union_mounts']:7.3f} %  ge2m {row['ge2_mounts']:6.2f}%  "
              f"dead {row['dead_mounts']['dead']:5d} "
              f"({row['dead_mounts']['dead_near_base']} within "
              f"{NEAR_BASE} m of a base)", flush=True)
        if want_pairs:
            qmaps = {r[1]: (r[2], r[3], r[5]) for r in by_layout[n]}
            t1 = time.time()
            pr = pair_clearance(n, lay, qmaps, stride=stride)
            record["pairs"][n] = pr
            print(f"{'':>18}  pairs: worst {pr['worst_pair_pct']:.1f} % mean "
                  f"{pr['mean_pair_pct']:.1f} % ({pr['n_below_90']} of 15 "
                  f"below 90 %) [{time.time() - t1:.0f} s]", flush=True)
        out_json.write_text(json.dumps(record, indent=1, default=float))
    if saved:
        old = dict(np.load(out_npz)) if out_npz.exists() else {}
        old.update(saved)
        old["xs"], old["ys"] = XS, YS
        np.savez_compressed(out_npz, **old)
    print(f"\nwrote {out_json} and {out_npz}")
    return record


def check():
    """Reproduce the published numbers before trusting anything else.

    Two independent pins, one per scenario, and they are what makes the rest
    of this file believable:

      air     16559 / 99.982 %   the atlas union docs/LAYOUT_STUDY.md §0 gives
      mounts  16248 / 98.104 %   out/cover_occupancy.json, the capsule ceiling

    The `box` column has no pin — it is the model commit 14b01cd put into the
    package, and it is STRICTER than the capsule by construction, so it is
    expected to sit below `mounts`.
    """
    lay = layout.LAYOUT_PROPOSED
    scens = ["air", "mounts", "box"]
    jl = [("adopted", lay, aid, scens, {})
          for aid in sorted(layout.build_fleet(lay))]
    with mp.get_context("fork").Pool(6) as pool:
        res = pool.map(score_arm, jl)
    row, _ = assemble("adopted", lay, res, scens)
    pins = dict(air="16559 / 99.982 % (docs/LAYOUT_STUDY.md)",
                mounts="16248 / 98.104 % (out/cover_occupancy.json)",
                box="no pin — the package's own box model, stricter")
    for nm in scens:
        print(f"  {nm:>6} union {row['union_' + nm]:7.3f} %  "
              f"({int(round(row['union_' + nm] * NCELL / 100))} cells)"
              f"   pin: {pins[nm]}")
    for aid in sorted(row["per_arm"], key=int):
        print(f"    arm {aid:>3}: "
              + "  ".join(f"{nm} {row['per_arm'][aid][nm]:5d}"
                          for nm in scens))
    return row


def park_scores(names, C, jobs, out_json, use_baked=False):
    """The PARKED union of each candidate, on ONE park recipe for all of them.

    The `mounts` scenario is the ceiling every park policy is measured against;
    this is what a park policy actually delivers.  Comparing candidates means
    giving them the SAME recipe, so every layout gets
    `layout.certified_park_poses(fleet)` — the plain outward-bearing ladder at
    hover 0.10 — rather than a per-layout tuned depot grid.  `use_baked` scores
    the adopted layout at its own baked `Q_PARK_PROPOSED` as well, which is the
    only park set that has actually been chosen for a build.
    """
    rec = json.loads(Path(out_json).read_text())
    rec.setdefault("parked", {})
    for n in names:
        lay = C[n][0]
        fleet = layout.build_fleet(lay)
        try:
            qp = layout.certified_park_poses(fleet, pen_lat=PEN_LAT_HOLDER)
            recipe = "ladder"
        except RuntimeError as e:
            print(f"{n:>18}: NO certified park set on the plain ladder — {e}")
            rec["parked"][n] = dict(ok=False, reason=str(e))
            Path(out_json).write_text(json.dumps(rec, indent=1, default=float))
            continue
        parks = {"parked": {a: np.asarray(q, float) for a, q in qp.items()}}
        if use_baked and n == "adopted":
            parks["baked"] = {a: np.asarray(q, float)
                              for a, q in layout.Q_PARK_PROPOSED.items()}
        scens = ["air", "mounts"] + list(parks)
        jl = [(n, lay, aid, scens, parks) for aid in sorted(fleet)]
        with mp.get_context("fork").Pool(min(jobs, len(jl))) as pool:
            res = pool.map(score_arm, jl)
        row, _ = assemble(n, lay, res, scens)
        rec["parked"][n] = dict(ok=True, recipe=recipe,
                                q={str(a): [float(v) for v in q]
                                   for a, q in qp.items()},
                                **{k: v for k, v in row.items()
                                   if k.startswith(("union_", "ge2_",
                                                    "dead_"))})
        print(f"{n:>18}: parked union {row['union_parked']:7.3f} %  "
              f"(air {row['union_air']:.3f} %, ceiling in the main table)"
              + (f"   baked {row['union_baked']:.3f} %"
                 if "union_baked" in row else ""))
        Path(out_json).write_text(json.dumps(rec, indent=1, default=float))
    return rec


def report(out_json, sort="union_mounts"):
    """The scorecard, ranked -> stdout.  Reads only what a run wrote."""
    rec = json.loads(Path(out_json).read_text())
    rows = list(rec["rows"].values())
    rows.sort(key=lambda r: -r[sort])
    hdr = (f"{'#':>3} {'candidate':>16} {'family':>8} {'pitch':>6} {'h':>6} "
           f"{'CORRECTED':>10} {'box':>8} {'air':>8} {'parked':>8} "
           f"{'>=2 corr':>9} {'dead':>6} {'<35cm':>6} {'annuli':>7} "
           f"{'worst':>7} {'mean':>6} {'<90':>4}")
    print(hdr)
    print("-" * len(hdr))
    for k, r in enumerate(rows):
        p = rec.get("pairs", {}).get(r["name"])
        pk = rec.get("parked", {}).get(r["name"], {})
        d = r["dead_mounts"]
        print(f"{k + 1:>3} {r['name']:>16} {r['family']:>8} "
              f"{r['spacing']:>6.3f} {r['layout']['h']:>6.3f} "
              f"{r['union_mounts']:>9.3f}% "
              f"{r.get('union_box', float('nan')):>7.3f}% "
              f"{r['union_air']:>7.3f}% "
              + (f"{pk['union_parked']:>7.3f}% " if pk.get("ok") else
                 f"{'-':>8} ")
              + f"{r['ge2_mounts']:>8.2f}% {d['dead']:>6d} "
              f"{d['dead_near_base']:>6d} {d['near_covered_pct']:>6.2f}% "
              + ("      -      -    -" if p is None else
                 f"{p['worst_pair_pct']:>6.1f}% {p['mean_pair_pct']:>5.1f}% "
                 f"{p['n_below_90']:>4d}"))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--parks", action="store_true")
    ap.add_argument("--sort", default="union_mounts")
    ap.add_argument("--only", default="", help="comma list of candidate names")
    ap.add_argument("--family", default="", help="comma list of families")
    ap.add_argument("--scen", default="air,mounts,box")
    ap.add_argument("--pairs", action="store_true")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--json", default=str(ROOT / "out/layout_rescore.json"))
    ap.add_argument("--npz", default=str(ROOT / "out/layout_rescore.npz"))
    ap.add_argument("--hsweep", default="", help="candidate name to h-sweep")
    a = ap.parse_args()

    C = candidates()
    if a.hsweep:
        base = C[a.hsweep][0]
        for k, v in h_variants(base).items():
            C[f"{a.hsweep}_{k}"] = (v, "height",
                                    f"{C[a.hsweep][2]} at h = {v['h']:.3f}")
    if a.list:
        for n, (lay, fam, note) in C.items():
            bad = layout.check_spacing(lay)
            print(f"{n:>18} [{fam:>8}] {note}"
                  + ("   VIOLATES: " + "; ".join(bad) if bad else ""))
        return
    if a.report:
        report(a.json, a.sort)
        return
    if a.check:
        check()
        return
    names = [n.strip() for n in a.only.split(",") if n.strip()] or list(C)
    if a.family:
        fams = {f.strip() for f in a.family.split(",")}
        names = [n for n in names if C[n][1] in fams]
    bad = [n for n in names if n not in C]
    if bad:
        raise SystemExit(f"unknown candidates: {bad}")
    if a.parks:
        park_scores(names, C, a.jobs, Path(a.json), use_baked=True)
        return
    run(names, C, a.jobs, [s.strip() for s in a.scen.split(",")], a.pairs,
        Path(a.json), Path(a.npz), a.stride)


if __name__ == "__main__":
    main()
