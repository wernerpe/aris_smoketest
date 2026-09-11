#!/usr/bin/env python3
"""STAGED WORK CELLS: which arms can draw WHERE at the same time, measured.

    ARIS_RIG=proposed ARIS_TOOL=lateral python3 scripts/workcell_envelopes.py \
        --atlas out/atlas_proposed_h0970_lat0860_gated63 \
        --json out/workcell_envelopes.json --jobs 6

Pete, 2026-09-11: *"draw things by the one arm per column in a 121 pattern and
make those arms the leaders and then have the remaining arms only plan to fill
in safe spots they can handle while the leader arms are doing their thing and
then afterwards reverse it.  If there are ways to restrict the work cell of the
arms temporarily that would allow us to do some asynchronous planning."*

WHAT A STAGE IS, AS A MEASURABLE OBJECT.  A stage assigns each arm a REGION of
the paper it may draw in.  The arm's ENVELOPE for that stage is the union of
its own metal over every certified pose it could hold while working there — the
atlas's drawing pose at every certified cell of the region, the pen-up hover
above each of those cells, and the park it starts and ends the stage at.  Two
arms whose envelopes are `coordination.PAIR_MARGIN` apart CANNOT TOUCH
whatever either of them does inside its region, in whatever order, at whatever
speed: each is a static keep-out for the other and the pair needs no conductor,
no shared clock and no re-planning when one of them is late.  That is the whole
of the asynchrony Pete is asking for, and it is a geometric question with a
number for an answer.

THE MEASUREMENT IS ONE CELL-PAIR MATRIX PER ARM PAIR, AND EVERYTHING ELSE IS A
MIN OVER A SUBMATRIX.  An envelope is a union over cells, and the clearance
between two unions is the minimum over the cell pairs:

    d(E(a, Ra), E(b, Rb)) = min over ca in Ra, cb in Rb of d(pose_a(ca), pose_b(cb))

so the expensive part is computed ONCE per arm pair over all certified cells,
and then every region, every pattern and every stage sequence below is a numpy
reduction over a slice of it.  The distances are `coordination`'s own —
`ArmPath` + `clearance_matrix`, the shipped measured capsules, exact
segment-to-segment, clipped at `BROAD_CAP` (0.25 m) because nothing above that
is a question anyone is asking.

WHAT IS APPROXIMATE, SAID OUT LOUD.

  CELL STRIDE.  The atlas is a 2 cm grid and the matrices are n^2, so the
  default `--stride 2` reads every second cell in each axis: a 4 cm lattice,
  ~960 cells an arm instead of ~3 830, and a worst-pair matrix that costs 45 s
  instead of 12 minutes.  The envelope is therefore sampled, not swept: a pose
  at a cell the stride skipped could stick out further than its neighbours.
  `--stride 1` is the whole atlas and is the number to quote before anything is
  built.

  THE HOVER IS `writing.lifted_config`, NOT `hover_solve`.  The shipped hover
  solver scores candidates against the static scene; this one takes the narrow
  local scan where it exists and the whole fiber (8 tool yaws x the q7 grid)
  where it does not, which is `hover_solve`'s own escalation without the
  static score.  It is the same pose on the great majority of cells and a
  neighbour of it elsewhere.

  THE PEN-UP LEG BETWEEN TWO HOVERS IS NOT SAMPLED.  A leg is a joint-space
  line between two hovers of the SAME region, and both ends are in the
  envelope; the line between them is not, and a joint-space line can bulge
  outside the convex hull of its ends.  Everything reported here is therefore
  an envelope of the ENDPOINTS, and a stage that clears by only a few
  millimetres is not certified by this script — `scene_check.check_timeline`
  is what certifies a leg.  The patterns recommended below clear by 10 cm and
  more, which is the margin that makes the gap safe to have.
"""
import argparse
import itertools
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("ARIS_RIG", "proposed")
os.environ.setdefault("ARIS_TOOL", "lateral")
sys.path.insert(0, str(ROOT))

from aris_sixarm import atlas as atlas_mod                        # noqa: E402
from aris_sixarm import coordination as co                        # noqa: E402
from aris_sixarm import ik, layout, writing                       # noqa: E402

ARMS = (2, 13, 17, 31, 71, 97)
GRID = 0.02
# THE CERTIFIED BLOCK, from `out/certified_area_h0970.json` — the largest
# all-live axis-aligned rectangle of the three-layer map at the shipped height.
# Every area fraction below is a fraction of THIS, not of the paper: a stage
# pattern that covers the rim is covering cells no arm can certify anyway.
BLOCK = (0.16, 0.00, 1.64, 3.62)
# the base lattice: two columns, three rows (layout.paired_grid at 0.61 m)
COL_X = (0.5967, 1.2067)
ROW_Y = (0.6051, 1.8153, 3.0255)
X_MID = float(np.mean(COL_X))                       # 0.9017
Y_CUT = (float(np.mean(ROW_Y[:2])), float(np.mean(ROW_Y[1:])))   # 1.210, 2.421
ARM_AT = {(0, 0): 13, (1, 0): 17, (0, 1): 31, (1, 1): 71, (0, 2): 2, (1, 2): 97}
COL_OF = {a: i for (i, j), a in ARM_AT.items()}
ROW_OF = {a: j for (i, j), a in ARM_AT.items()}

GATE = co.PAIR_MARGIN          # 0.050 m, the arm-to-arm gate in force
GATE_OLD = 0.080               # what it was before 2026-09-09, for reference


# ==========================================================================
# the cells, the poses, the envelopes
# ==========================================================================
def lattice(stride):
    """The common cell lattice over the certified block. -> (xs, ys)."""
    step = GRID * stride
    xs = np.arange(BLOCK[0], BLOCK[2] + 1e-9, step)
    ys = np.arange(BLOCK[1], BLOCK[3] + 1e-9, step)
    return xs, ys


def arm_cells(arm, atlas_dir, xs, ys, spec, h_inv, verbose=True):
    """One arm's certified cells on the lattice, with a drawing and a hover pose.

    -> dict(ix, iy, x, y, q_draw (n,7), q_hover (n,7), has_hover (n,) bool)

    `q_hover` falls back to the drawing pose where no hover exists at all, which
    is the honest thing to put in an envelope: the arm can still draw the cell,
    it just cannot lift off it, and `writing`'s own `lifts` list would read zero.
    """
    arr, meta = atlas_mod.load(Path(atlas_dir), arm)
    cur, why = atlas_mod.is_current(meta)
    if not cur:
        raise SystemExit(f"atlas for arm {arm} is stale: {why}")
    arr = arr[atlas_mod.strict_go(arr)]
    ix = np.rint((arr[:, 0] - xs[0]) / (xs[1] - xs[0])).astype(int)
    iy = np.rint((arr[:, 1] - ys[0]) / (ys[1] - ys[0])).astype(int)
    on = (np.abs(xs[0] + ix * (xs[1] - xs[0]) - arr[:, 0]) < 1e-6) & \
         (np.abs(ys[0] + iy * (ys[1] - ys[0]) - arr[:, 1]) < 1e-6) & \
         (ix >= 0) & (ix < len(xs)) & (iy >= 0) & (iy < len(ys))
    arr, ix, iy = arr[on], ix[on], iy[on]
    qd = np.ascontiguousarray(arr[:, atlas_mod.QCOL:atlas_mod.QCOL + 7])
    qh = qd.copy()
    have = np.zeros(len(arr), bool)
    pen = spec.pen
    for k in range(len(arr)):
        xy = (arr[k, 0], arr[k, 1])
        q, _ = writing.lifted_config(spec, qd[k], xy, z=writing.LIFT_Z,
                                     h_inv=h_inv, pen_ext=pen)
        if q is None:       # the whole fiber, which is `hover_solve`'s own
            q, _ = writing.lifted_config(spec, qd[k], xy, z=writing.LIFT_Z,
                                         h_inv=h_inv, pen_ext=pen,
                                         phis=writing.HOVER_YAWS,
                                         q7s=ik.Q7_GRID)
        if q is not None:
            qh[k], have[k] = q, True
    if verbose:
        print(f"  arm {arm:>2}: {len(arr):>5} certified cells on the lattice, "
              f"{int(have.sum())} with a hover ({100 * have.mean():.1f} %)")
    return dict(ix=ix, iy=iy, x=arr[:, 0], y=arr[:, 1], q_draw=qd, q_hover=qh,
                has_hover=have)


def drop_sweep_band(path, on=True):
    """`coordination.FROZEN_SWEEP_BANDS`, applied to a whole envelope.

    EVERY POSE IN AN ENVELOPE IS A KNOWN POSE.  The last base band is link1's
    REVOLUTION about joint 1 — the envelope of the upper arm over an unknown
    q1 — and every pose here has a q1.  Its real upper arm is already carried
    by the `(1, 3, UPPER_R)` capsule, so keeping the band is the double count
    DECISION 2026-09-09 ("a known pose stops paying for a sweep") took out of
    the frozen-partner check, and it is the same double count here: measured
    below, it is worth 8 to 60 mm of pair clearance on this rig.
    """
    if not on:
        return path
    keep = [k for k in range(len(path.r)) if k not in co.FROZEN_SWEEP_BANDS]
    path.A = np.ascontiguousarray(path.A[:, keep])
    path.B = np.ascontiguousarray(path.B[:, keep])
    path.r = np.ascontiguousarray(path.r[keep])
    path._box, path._tiles, path._key = None, {}, None
    return path


_PATHS = None      # inherited by the fork pool; see `pair_job`


def pair_job(task):
    """One arm pair's cell-to-cell clearance, in a forked worker.

    -> (a, b, C (na, nb), pa_vs_b (nb,), pb_vs_a (na,), park_park)
    """
    a, b, na, nb = task
    M = co.clearance_matrix(_PATHS[a], _PATHS[b])
    C = np.minimum(np.minimum(M[:na, :nb], M[na:2 * na, :nb]),
                   np.minimum(M[:na, nb:2 * nb], M[na:2 * na, nb:2 * nb]))
    pa = np.minimum(M[2 * na, :nb], M[2 * na, nb:2 * nb])
    pb = np.minimum(M[:na, 2 * nb], M[na:2 * na, 2 * nb])
    return a, b, C.astype(np.float32), pa.astype(np.float32), \
        pb.astype(np.float32), float(M[2 * na, 2 * nb])


def build(atlas_dir, stride, jobs, verbose=True, known_pose=True):
    """Cells, poses and the per-arm-pair cell clearance matrices. -> dict."""
    global _PATHS
    fleet = layout.FLEET_PROPOSED
    h_inv = layout.LAYOUT_PROPOSED["h"]
    xs, ys = lattice(stride)
    if verbose:
        print(f"lattice {len(xs)} x {len(ys)} cells of {GRID * stride:.2f} m "
              f"over the certified block {BLOCK}")
    cells = {a: arm_cells(a, atlas_dir, xs, ys, fleet[a], h_inv, verbose)
             for a in ARMS}
    # the envelope's pose stack: every drawing pose, every hover, then the park
    paths = {}
    for a in ARMS:
        c = cells[a]
        Q = np.vstack([c["q_draw"], c["q_hover"],
                       np.asarray(layout.Q_PARK_PROPOSED[a], float)[None, :]])
        paths[a] = drop_sweep_band(co.ArmPath(a, Q, 0.01, h_inv, fleet[a].pen,
                                              fleet[a]), known_pose)
    tasks = [(a, b, len(cells[a]["ix"]), len(cells[b]["ix"]))
             for a, b in itertools.combinations(ARMS, 2)]
    t0 = time.time()
    _PATHS = paths
    try:
        if jobs > 1:
            with mp.get_context("fork").Pool(jobs) as pool:
                out = list(pool.imap_unordered(pair_job, tasks))
        else:
            out = [pair_job(t) for t in tasks]
    finally:
        _PATHS = None
    C, PA, PP = {}, {}, {}
    for a, b, m, pa, pb, pp in out:
        C[(a, b)] = m
        PA[(a, b)] = pa          # a's PARK against b's cells
        PA[(b, a)] = pb          # b's park against a's cells
        PP[(a, b)] = PP[(b, a)] = pp
    if verbose:
        print(f"{len(tasks)} cell-pair matrices on {jobs} process(es) in "
              f"{time.time() - t0:.1f} s")
    return dict(xs=xs, ys=ys, cells=cells, C=C, PA=PA, PP=PP, stride=stride)


# ==========================================================================
# regions
# ==========================================================================
def rect_mask(xs, ys, r):
    """(x0, y0, x1, y1) -> (H, W) boolean over the lattice (half-open at hi)."""
    x0, y0, x1, y1 = r
    return ((ys[:, None] >= y0 - 1e-9) & (ys[:, None] < y1 - 1e-9) &
            (xs[None, :] >= x0 - 1e-9) & (xs[None, :] < x1 - 1e-9))


def split_frontier(d, a, b, axis):
    """How far apart two work cells must be SPLIT before the arms clear.

    Arm `a` is confined to `u <= t_a` and arm `b` to `u >= t_b` along axis
    `u` (0 = paper x, 1 = paper y); everything else about both regions is left
    open, so this is the weakest possible separation hypothesis and the
    strongest possible statement of what it costs.

    -> dict(half_gap_m, gap_m, best_gap_m, at, curve)

    `half_gap_m` is the symmetric answer — the dead band centred on the
    midline between the two bases, which is where a work-cell partition would
    naturally be drawn — and `best_gap_m` is the narrowest band ANY placement
    of the two thresholds achieves, which says whether the symmetric cut is the
    right one.
    """
    key = (a, b) if (a, b) in d["C"] else (b, a)
    M = d["C"][key]
    if key != (a, b):
        M = M.T
    ua = d["cells"][a]["x" if axis == 0 else "y"]
    ub = d["cells"][b]["x" if axis == 0 else "y"]
    oa, ob = np.argsort(ua, kind="stable"), np.argsort(ub, kind="stable")
    S = np.asarray(M, np.float32)[np.ix_(oa, ob)]
    # Q[i, j] = min clearance with a's cells at u <= ua[i] and b's at u >= ub[j]
    Q = np.minimum.accumulate(S, axis=0)
    Q = np.minimum.accumulate(Q[:, ::-1], axis=1)[:, ::-1]
    sa, sb = ua[oa], ub[ob]
    # the symmetric cut, about the midline between the two BASES
    base_u = ((COL_X if axis == 0 else ROW_Y)[COL_OF[a] if axis == 0
                                              else ROW_OF[a]],
              (COL_X if axis == 0 else ROW_Y)[COL_OF[b] if axis == 0
                                              else ROW_OF[b]])
    c = float(np.mean(base_u))
    curve, half = [], None
    for h in np.arange(0.0, 1.01, 0.02):
        i = np.searchsorted(sa, c - h, "right") - 1
        j = np.searchsorted(sb, c + h, "left")
        v = float(Q[i, j]) if (i >= 0 and j < len(sb)) else float("inf")
        curve.append((round(float(h), 3), round(1000 * v, 1)))
        if half is None and v >= GATE:
            half = float(h)
    # the narrowest band any placement achieves
    ok = Q >= GATE
    best, at = float("inf"), None
    ii, jj = np.nonzero(ok)
    if len(ii):
        g = sb[jj] - sa[ii]
        k = int(np.argmin(g))
        best, at = float(g[k]), (float(sa[ii[k]]), float(sb[jj[k]]))
    return dict(half_gap_m=half, gap_m=None if half is None else 2 * half,
                best_gap_m=best, at=at, curve=curve, midline=c)


def elbow_span(d, arm):
    """Where this arm's ELBOW goes while it draws. -> dict of world extents.

    The frontier numbers below are not about the pen: they are about the joint
    the atlas puts a metre away from it.  This is the one-line explanation, per
    arm: the x and y range of chain point 3 (the elbow) and of point 5 (the
    wrist) over every certified drawing pose, against the base's own xy.
    """
    from aris_sixarm import layout as L
    spec = L.FLEET_PROPOSED[arm]
    W = co.chain_world(d["cells"][arm]["q_draw"], spec, L.LAYOUT_PROPOSED["h"],
                       spec.pen)
    out = dict(base_xy=[float(spec.xy[0]), float(spec.xy[1])])
    for name, k in (("elbow", 3), ("wrist", 5), ("tip", 9)):
        out[name] = dict(x=[float(W[:, k, 0].min()), float(W[:, k, 0].max())],
                         y=[float(W[:, k, 1].min()), float(W[:, k, 1].max())],
                         z=[float(W[:, k, 2].min()), float(W[:, k, 2].max())])
    return out


def block_rect(i, j, shrink=0.0, shrink_y=None):
    """The Voronoi cell of base (col i, row j), as a rectangle.

    THE VORONOI CELLS OF A REGULAR 2 x 3 GRID *ARE* THE COLUMN-BY-ROW BLOCKS,
    so "each arm's own base zone", "the thirds of a column band" and "the halves
    of a row band" are three names for the same six rectangles, and the region
    vocabulary below only has to speak rectangles.  `shrink` erodes ONLY the
    internal boundaries — the ones shared with another arm — because giving back
    paper at the rim buys no clearance from anybody.
    """
    sy = shrink if shrink_y is None else shrink_y
    x0 = BLOCK[0] if i == 0 else X_MID + shrink
    x1 = X_MID - shrink if i == 0 else BLOCK[2] + 1e-6
    y0 = BLOCK[1] if j == 0 else Y_CUT[j - 1] + sy
    y1 = (Y_CUT[j] - sy) if j < 2 else BLOCK[3] + 1e-6
    return (x0, y0, x1, y1)


def named_regions(xs, ys):
    """The region vocabulary. -> {name: (H, W) bool}."""
    R = {}
    R["ALL"] = rect_mask(xs, ys, (BLOCK[0], BLOCK[1], BLOCK[2] + 1e-6,
                                  BLOCK[3] + 1e-6))
    for i in (0, 1):
        for j in (0, 1, 2):
            R[f"B{i}{j}"] = rect_mask(xs, ys, block_rect(i, j))
            for s in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60):
                R[f"B{i}{j}-e{int(s * 100):02d}"] = rect_mask(
                    xs, ys, block_rect(i, j, s))
                # eroded in y ONLY: the partition a 3-active stage needs, where
                # the two arms of a column never draw in the same stage and the
                # x seam is therefore not a seam
                R[f"B{i}{j}-y{int(s * 100):02d}"] = rect_mask(
                    xs, ys, block_rect(i, j, 0.0, s))
    for j in (0, 1, 2):                                   # full-width row bands
        R[f"R{j}"] = R[f"B0{j}"] | R[f"B1{j}"]
        for s in (0.05, 0.10, 0.15, 0.20, 0.30):
            R[f"R{j}-y{int(s * 100):02d}"] = (
                R[f"B0{j}-y{int(s * 100):02d}"] | R[f"B1{j}-y{int(s * 100):02d}"])
    # THE SEAMS: the y dead band a 3-active stage gives up between row bands,
    # which some later stage has to come back for.
    for k in (0, 1):
        for s in (0.05, 0.10, 0.15, 0.20, 0.30):
            R[f"SEAM{k}-y{int(s * 100):02d}"] = rect_mask(
                xs, ys, (BLOCK[0], Y_CUT[k] - s, BLOCK[2] + 1e-6, Y_CUT[k] + s))
    for i in (0, 1):                                      # full-length columns
        R[f"C{i}"] = R[f"B{i}0"] | R[f"B{i}1"] | R[f"B{i}2"]
        R[f"C{i}-lo"] = rect_mask(xs, ys, (block_rect(i, 0)[0], BLOCK[1],
                                           block_rect(i, 0)[2], ROW_Y[1]))
        R[f"C{i}-hi"] = rect_mask(xs, ys, (block_rect(i, 0)[0], ROW_Y[1],
                                           block_rect(i, 0)[2], BLOCK[3] + 1e-6))
    # THE PAPER ENDS, which is what Pete's followers get in reading (a): the
    # strip of each end block beyond its own base row, at a ladder of depths.
    for i in (0, 1):
        bx0, _, bx1, _ = block_rect(i, 0)
        for y in (0.40, 0.605, 0.80, 1.00, 1.2103):
            R[f"S{i}-{int(y * 100):03d}"] = rect_mask(
                xs, ys, (bx0, BLOCK[1], bx1, y))
        for y in (3.22, 3.0255, 2.82, 2.62, 2.4204):
            R[f"N{i}-{int(y * 100):03d}"] = rect_mask(
                xs, ys, (bx0, y, bx1, BLOCK[3] + 1e-6))
    # ...and the middle band at a ladder of widths, for reading (b)
    for h in (0.20, 0.40, 0.60, 0.80, 1.00, 1.2103):
        for i in (0, 1):
            bx0, _, bx1, _ = block_rect(i, 0)
            R[f"M{i}-{int(h * 100):03d}"] = rect_mask(
                xs, ys, (bx0, ROW_Y[1] - h / 2, bx1, ROW_Y[1] + h / 2))
    return R


def cell_sel(d, arm, mask):
    """Indices into arm `arm`'s cell list that fall inside `mask`. -> (k,) int."""
    c = d["cells"][arm]
    return np.flatnonzero(mask[c["iy"], c["ix"]])


def pair_clear(d, a, ia, b, ib, with_park=True):
    """Envelope-to-envelope clearance for two (arm, index set) work cells. -> m.

    `+inf` when either side has no certified cell in its region — an arm with
    nothing to draw is not a participant, and its park is handled by the caller.
    """
    if not len(ia) or not len(ib):
        return float("inf")
    key = (a, b) if (a, b) in d["C"] else (b, a)
    M = d["C"][key]
    v = float(M[np.ix_(ia, ib)].min() if key == (a, b)
              else M[np.ix_(ib, ia)].min())
    if with_park:
        v = min(v, float(d["PA"][(a, b)][ib].min()),
                float(d["PA"][(b, a)][ia].min()), float(d["PP"][(a, b)]))
    return v


def park_vs(d, a, b, ib):
    """Arm a's PARK against arm b's work cell (b's cells `ib` plus b's park)."""
    if not len(ib):
        return float(d["PP"][(a, b)])
    return min(float(d["PA"][(a, b)][ib].min()), float(d["PP"][(a, b)]))


# ==========================================================================
# stage patterns
# ==========================================================================
def stage_clearances(d, stage):
    """One stage {arm: mask|None} -> dict(ink, env, worst_ink, worst_env, ...).

    TWO NUMBERS, BECAUSE THEY ARE TWO DIFFERENT DECISIONS.

      `ink`  the work cells against each other: active arm a's envelope over its
             own region against active arm b's over its own, and nothing else.
             This is the question "can these two arms be told to go away and
             draw asynchronously" — the one a stage pattern is FOR.
      `env`  the same, plus the SHIPPED park (`layout.Q_PARK_PROPOSED`) on both
             sides, and every inactive arm's park against every active arm's
             envelope.  An inactive arm holds its park for the whole stage and
             is therefore a static keep-out for the whole stage.

    They come apart hard on this rig and the gap is the finding: a park chosen
    to be clear of the ALLOCATED ink of a particular programme is not clear of
    everything an arm could be told to draw.
    """
    sel = {a: (cell_sel(d, a, m) if m is not None else None)
           for a, m in stage.items()}
    active = [a for a in ARMS if sel.get(a) is not None and len(sel[a])]
    ink, env = {}, {}
    for a, b in itertools.combinations(ARMS, 2):
        ia, ib = sel.get(a), sel.get(b)
        both = (ia is not None and len(ia) and ib is not None and len(ib))
        if both:
            ink[(a, b)] = pair_clear(d, a, ia, b, ib, with_park=False)
            env[(a, b)] = pair_clear(d, a, ia, b, ib, with_park=True)
        elif ia is not None and len(ia):
            env[(a, b)] = park_vs(d, b, a, ia)
        elif ib is not None and len(ib):
            env[(a, b)] = park_vs(d, a, b, ib)
        else:
            env[(a, b)] = float(d["PP"][(a, b)])
    return dict(ink=ink, env=env, active=active,
                worst_ink=float(min(ink.values())) if ink else float("inf"),
                worst_env=float(min(env.values())) if env else float("inf"),
                n_cells={a: int(len(sel[a])) for a in active})


def evaluate(d, name, stages, note=""):
    """A whole stage SEQUENCE, priced. -> dict.

    Each certified block cell is charged to the FIRST (stage, arm) that can
    draw it; a stage costs its busiest arm, the sequence costs the sum, and the
    serial baseline is the total.  Uniform ink density over the block is the
    only assumption, and it is the same one on both sides of the ratio.
    """
    H, W = len(d["ys"]), len(d["xs"])
    done = np.zeros((H, W), bool)
    rows, total_cost = [], 0.0
    for si, stage in enumerate(stages):
        cl = stage_clearances(d, stage)
        load, drew = {}, np.zeros((H, W), bool)
        for a in ARMS:
            m = stage.get(a)
            if m is None:
                continue
            c = d["cells"][a]
            mine = np.zeros((H, W), bool)
            mine[c["iy"], c["ix"]] = True
            mine &= m & ~done & ~drew
            load[a] = int(mine.sum())
            drew |= mine
        done |= drew
        cost = max(load.values()) if load else 0
        total_cost += cost
        rows.append(dict(stage=si, active=sorted(load), load=load,
                         cells=int(drew.sum()), cost=int(cost),
                         worst_ink_mm=round(1000 * cl["worst_ink"], 1),
                         worst_env_mm=round(1000 * cl["worst_env"], 1),
                         ink={f"{a}-{b}": round(1000 * v, 1)
                              for (a, b), v in cl["ink"].items()},
                         env={f"{a}-{b}": round(1000 * v, 1)
                              for (a, b), v in cl["env"].items()}))
    blk = d["block_mask"]
    reach = d["reach_any"] & blk
    covered = int((done & blk).sum())
    n_block = int(blk.sum())
    n_reach = int(reach.sum())
    serial = int((d["reach_any"] & blk).sum())
    out = dict(name=name, note=note, n_stages=len(stages), stages=rows,
               covered=covered, block_cells=n_block, reach_cells=n_reach,
               coverage=covered / max(n_reach, 1),
               cost=float(total_cost), serial=float(serial),
               speedup=serial / total_cost if total_cost else float("nan"),
               worst_ink_mm=min(r["worst_ink_mm"] for r in rows) if rows else None,
               worst_env_mm=min(r["worst_env_mm"] for r in rows) if rows else None,
               uncovered=int(reach.sum() - (done & reach).sum()))
    # WHERE the holes are, not just how many: a hole is only ever a band of y
    # on this rig (the partition is by row), so the y values of the uncovered
    # cells are the whole diagnosis.
    miss = reach & ~done
    out["uncovered_y"] = ([round(float(v), 3) for v in
                           np.unique(d["ys"][np.nonzero(miss)[0]])]
                          if miss.any() else [])
    out["_masks"] = stages          # kept for the redundancy pass; not JSON
    return out


def blocks(R):
    """{arm: its own Voronoi block mask}, optionally eroded."""
    return {a: R[f"B{COL_OF[a]}{ROW_OF[a]}"] for a in ARMS}


def eroded(R, s):
    tag = "" if s == 0 else f"-e{int(s * 100):02d}"
    return {a: R[f"B{COL_OF[a]}{ROW_OF[a]}{tag}"] for a in ARMS}


def ladder(d, R, make, values, label):
    """Walk a one-parameter family and report the first value that clears GATE."""
    out = []
    for v in values:
        cl = stage_clearances(d, make(v))
        worst_pair = min(cl["ink"], key=cl["ink"].get) if cl["ink"] else None
        out.append(dict(value=v, worst_ink_mm=round(1000 * cl["worst_ink"], 1),
                        worst_env_mm=round(1000 * cl["worst_env"], 1),
                        binding=f"{worst_pair[0]}-{worst_pair[1]}"
                        if worst_pair else None,
                        active=cl["active"],
                        cells=int(sum(cl["n_cells"].values()))))
    print(f"\n{label}")
    print(f"   {'value':>16}  {'ink-vs-ink':>11}  {'+parks':>9}  "
          f"{'binds':>6}  {'cells':>6}")
    for r in out:
        ok = "GO" if r["worst_ink_mm"] >= 1000 * GATE else "--"
        print(f"  {ok} {r['value']!s:>16}  {r['worst_ink_mm']:>8.1f} mm  "
              f"{r['worst_env_mm']:>6.1f} mm  {str(r['binding']):>6}  "
              f"{r['cells']:>6}")
    return out


# ==========================================================================
# main
# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas",
                    default="out/atlas_proposed_h0970_lat0860_gated63")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--json", default="out/workcell_envelopes.json")
    ap.add_argument("--keep-sweep-band", action="store_true",
                    help="keep link1's revolution sweep even though every pose "
                         "in an envelope is a known pose (see drop_sweep_band)")
    a = ap.parse_args()

    d = build(a.atlas, a.stride, a.jobs, known_pose=not a.keep_sweep_band)
    d["known_pose"] = not a.keep_sweep_band
    xs, ys = d["xs"], d["ys"]
    R = named_regions(xs, ys)
    d["block_mask"] = R["ALL"]
    reach = np.zeros((len(ys), len(xs)), bool)
    n_arms = np.zeros((len(ys), len(xs)), int)
    for arm in ARMS:
        c = d["cells"][arm]
        m = np.zeros_like(reach)
        m[c["iy"], c["ix"]] = True
        reach |= m
        n_arms += m
    d["reach_any"], d["n_arms"] = reach, n_arms
    blk = R["ALL"]
    print(f"\ncertified block: {int(blk.sum())} lattice cells, "
          f"{int((reach & blk).sum())} reachable by some arm "
          f"({100 * (reach & blk).sum() / blk.sum():.1f} %); "
          f"{int(((n_arms >= 2) & blk).sum())} by two or more "
          f"({100 * ((n_arms >= 2) & blk).sum() / max((reach & blk).sum(), 1):.1f} % "
          "of the reachable block)")

    report = dict(gate_m=GATE, gate_old_m=GATE_OLD, stride=a.stride,
                  atlas=a.atlas, block=BLOCK, known_pose=d["known_pose"],
                  lattice=[len(xs), len(ys)], grid=GRID * a.stride)

    # ---- 1. the compatibility table -------------------------------------
    print("\n" + "=" * 74)
    print("1. ENVELOPE CLEARANCE, mm — every arm pair, own block vs own block")
    print("=" * 74)
    own = blocks(R)
    tab = {}
    for x, y in itertools.combinations(ARMS, 2):
        ix, iy = cell_sel(d, x, own[x]), cell_sel(d, y, own[y])
        v_ink = pair_clear(d, x, ix, y, iy, with_park=False)
        v_env = pair_clear(d, x, ix, y, iy, with_park=True)
        tab[f"{x}-{y}"] = dict(ink_mm=round(1000 * v_ink, 1),
                               env_mm=round(1000 * v_env, 1),
                               go50=v_env >= GATE, go80=v_env >= GATE_OLD)
        print(f"  {x:>2} vs {y:>2}   ink {1000 * v_ink:>8.1f}   "
              f"+park {1000 * v_env:>8.1f}   "
              f"{'GO' if v_env >= GATE else 'NO':>2}@50  "
              f"{'GO' if v_env >= GATE_OLD else 'NO':>2}@80")
    report["own_block_pairs"] = tab

    # the FULL region-pair table, for the arms that share a row or a column
    print("\n  region-pair minimum clearance (mm), selected regions:")
    sel_regions = ["ALL", "B00", "B10", "B01", "B11", "B02", "B12",
                   "R0", "R1", "R2", "C0", "C1"]
    full = {}
    for x, y in itertools.combinations(ARMS, 2):
        for rx in sel_regions:
            for ry in sel_regions:
                ix, iy = cell_sel(d, x, R[rx]), cell_sel(d, y, R[ry])
                if not len(ix) or not len(iy):
                    continue
                full[f"{x}:{rx}|{y}:{ry}"] = round(
                    1000 * pair_clear(d, x, ix, y, iy), 1)
    report["region_pairs"] = full
    print(f"    {len(full)} region pairs measured "
          f"({len(sel_regions)} regions x 15 arm pairs)")

    # ---- 1b. WHY: where the elbow goes -----------------------------------
    print("\n  the elbow is the problem, not the pen (world m, over every "
          "certified drawing pose):")
    print(f"    {'arm':>4} {'base x':>8} {'elbow x':>16} {'elbow y':>16} "
          f"{'tip x':>16}")
    els = {}
    for arm in ARMS:
        e = els[arm] = elbow_span(d, arm)
        print(f"    {arm:>4} {e['base_xy'][0]:>8.3f} "
              f"[{e['elbow']['x'][0]:>6.3f},{e['elbow']['x'][1]:>6.3f}] "
              f"[{e['elbow']['y'][0]:>6.3f},{e['elbow']['y'][1]:>6.3f}] "
              f"[{e['tip']['x'][0]:>6.3f},{e['tip']['x'][1]:>6.3f}]")
    report["elbow_span"] = els

    # ---- 1c. the separation frontier --------------------------------------
    print("\n  SEPARATION FRONTIER — the dead band two work cells need, per "
          "pair and axis (m; '-' = no split on that axis ever clears 50 mm):")
    print(f"    {'pair':>7} {'axis':>5} {'half gap':>9} {'dead band':>10} "
          f"{'narrowest band, any placement':>32}")
    fro = {}
    for x, y in itertools.combinations(ARMS, 2):
        for axis, nm in ((0, "x"), (1, "y")):
            lo, hi = (x, y) if (COL_OF if axis == 0 else ROW_OF)[x] <= \
                (COL_OF if axis == 0 else ROW_OF)[y] else (y, x)
            if (COL_OF if axis == 0 else ROW_OF)[lo] == \
               (COL_OF if axis == 0 else ROW_OF)[hi]:
                continue                 # same column/row: no split on this axis
            f = split_frontier(d, lo, hi, axis)
            fro[f"{lo}-{hi}:{nm}"] = f
            hg = "-" if f["half_gap_m"] is None else f"{f['half_gap_m']:.2f}"
            bg = ("-" if not np.isfinite(f["best_gap_m"])
                  else f"{f['best_gap_m']:.2f} m at "
                       f"{f['at'][0]:.2f} / {f['at'][1]:.2f}")
            dg = "-" if f["gap_m"] is None else f"{f['gap_m']:.2f}"
            print(f"    {lo:>3}-{hi:<3} {nm:>5} {hg:>9} {dg:>10} {bg:>32}")
    report["frontier"] = fro

    # ---- 1d. who can reach which seam ------------------------------------
    print("\n  REACH, over the certified block (tip, m) — the seam bands are "
          "[1.010, 1.410] and [2.220, 2.620] at a 0.40 m dead band:")
    print(f"    {'arm':>4} {'tip y':>16} {'tip x':>16} {'in SEAM0':>18} "
          f"{'in SEAM1':>18}")
    reachtab = {}
    for arm in ARMS:
        c = d["cells"][arm]
        rr = dict(y=[float(c["y"].min()), float(c["y"].max())],
                  x=[float(c["x"].min()), float(c["x"].max())])
        for k, nm in ((0, "SEAM0-y20"), (1, "SEAM1-y20")):
            s = cell_sel(d, arm, R[nm])
            rr[f"seam{k}"] = ([float(c["y"][s].min()), float(c["y"][s].max())]
                              if len(s) else None)
        reachtab[arm] = rr
        f0 = ("-" if rr["seam0"] is None
              else f"[{rr['seam0'][0]:.3f},{rr['seam0'][1]:.3f}]")
        f1 = ("-" if rr["seam1"] is None
              else f"[{rr['seam1'][0]:.3f},{rr['seam1'][1]:.3f}]")
        print(f"    {arm:>4} [{rr['y'][0]:>6.3f},{rr['y'][1]:>6.3f}] "
              f"[{rr['x'][0]:>6.3f},{rr['x'][1]:>6.3f}] {f0:>18} {f1:>18}")
    report["reach"] = reachtab

    # ---- 2b. the seam-stage pairing matrix --------------------------------
    print("\n  SEAM-STAGE PAIRINGS — arm a working SEAM0 against arm b working "
          "SEAM1, ink-vs-ink (mm); the two bands are 0.81 m apart in y:")
    s0 = [a for a in ARMS if len(cell_sel(d, a, R["SEAM0-y20"]))]
    s1 = [b for b in ARMS if len(cell_sel(d, b, R["SEAM1-y20"]))]
    print("      SEAM0\\SEAM1  " + "".join(f"{q:>9}" for q in s1))
    seamtab = {}
    for p0 in s0:
        i0 = cell_sel(d, p0, R["SEAM0-y20"])
        row = []
        for p1 in s1:
            if p0 == p1:
                row.append("     self")
                continue
            v = pair_clear(d, p0, i0, p1, cell_sel(d, p1, R["SEAM1-y20"]),
                           with_park=False)
            seamtab[f"{p0}|{p1}"] = round(1000 * v, 1)
            row.append(f"{1000 * v:>9.1f}")
        print(f"      {p0:>11}  " + "".join(row))
    report["seam_pairings"] = seamtab

    # ---- 2. the patterns -------------------------------------------------
    print("\n" + "=" * 74)
    print("2. STAGE PATTERNS")
    print("=" * 74)
    pats = []
    lad = {}

    # 6-active, shrunken cells: the erosion ladder
    lad["6-active-erosion"] = ladder(
        d, R, lambda s: eroded(R, s),
        (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60),
        "6-ACTIVE, all six in their own (eroded) Voronoi block "
        "(value = erosion of the internal boundaries, m):")

    # 3-active, one arm per row, eroded in Y ONLY (the x seam is not a seam
    # when the two arms of a column never draw in the same stage)
    lad["3-active-y-erosion"] = ladder(
        d, R, lambda s: {13: R[f"B00-y{int(s * 100):02d}"] if s else R["B00"],
                         71: R[f"B11-y{int(s * 100):02d}"] if s else R["B11"],
                         2: R[f"B02-y{int(s * 100):02d}"] if s else R["B02"]},
        (0.0, 0.05, 0.10, 0.15, 0.20, 0.30),
        "3-ACTIVE (13 / 71 / 2, one per row, alternating columns), own block "
        "eroded in y only:")
    lad["3-active-y-erosion-B"] = ladder(
        d, R, lambda s: {17: R[f"B10-y{int(s * 100):02d}"] if s else R["B10"],
                         31: R[f"B01-y{int(s * 100):02d}"] if s else R["B01"],
                         97: R[f"B12-y{int(s * 100):02d}"] if s else R["B12"]},
        (0.0, 0.05, 0.10, 0.15, 0.20, 0.30),
        "3-ACTIVE, the other three (17 / 31 / 97), own block eroded in y only:")
    lad["3-active-rowband-y"] = ladder(
        d, R, lambda s: {13: R[f"B00-y{int(s*100):02d}"] | R[f"B10-y{int(s*100):02d}"] if s else R["R0"],
                         71: R[f"B01-y{int(s*100):02d}"] | R[f"B11-y{int(s*100):02d}"] if s else R["R1"],
                         2: R[f"B02-y{int(s*100):02d}"] | R[f"B12-y{int(s*100):02d}"] if s else R["R2"]},
        (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40),
        "3-ACTIVE on FULL-WIDTH row bands (13 / 71 / 2), eroded in y:")

    # 1-2-1 (a): middle row leads on full blocks, ends work the paper ends
    lad["121a-end-depth"] = ladder(
        d, R, lambda t: {31: R["B01"], 71: R["B11"],
                         13: R[f"S0-{t[0]}"], 17: R[f"S1-{t[0]}"],
                         2: R[f"N0-{t[1]}"], 97: R[f"N1-{t[1]}"]},
        (("040", "322"), ("060", "302"), ("080", "282"), ("100", "262")),
        "1-2-1 (a) leaders 31/71 on FULL middle blocks, the four end arms "
        "confined to (south y <= a, north y >= b) in cm:")

    # 1-2-1 (b): end rows lead on full blocks, middle fills a band
    lad["121b-middle-band"] = ladder(
        d, R, lambda h: {13: R["B00"], 17: R["B10"], 2: R["B02"],
                         97: R["B12"], 31: R[f"M0-{h}"], 71: R[f"M1-{h}"]},
        ("020", "040", "060", "080", "100", "121"),
        "1-2-1 (b) leaders 13/17/2/97 on FULL end blocks, 31/71 confined to a "
        "middle band of this height (cm):")
    report["ladders"] = lad

    # PATTERN A — 3-active, one per row, alternating columns, on FULL row bands
    pats.append(evaluate(d, "3-active-rowband", [
        {13: R["R0"], 71: R["R1"], 2: R["R2"]},
        {17: R["R0"], 31: R["R1"], 97: R["R2"]}],
        "one arm per row band, whole width; columns alternate between stages"))

    # PATTERN A' — the same with the measured 0.15 m y dead band, and two
    # further stages that come back for the seams it gave up.  THE ONE THAT
    # WORKS; everything above it is here to show what it is better than.
    for s in (0.10, 0.15, 0.20):
        t = f"y{int(s * 100):02d}"
        pats.append(evaluate(d, f"3-active-rowband-{t}+seams", [
            {13: R[f"R0-{t}"], 71: R[f"R1-{t}"], 2: R[f"R2-{t}"]},
            {17: R[f"R0-{t}"], 31: R[f"R1-{t}"], 97: R[f"R2-{t}"]},
            {13: R[f"SEAM0-{t}"], 97: R[f"SEAM1-{t}"]},
            {71: R[f"SEAM0-{t}"], 2: R[f"SEAM1-{t}"]}],
            f"one arm per row band, whole width, {s:.2f} m dead band in y "
            "between bands; columns alternate; two seam stages"))

    # ...and the same with FOUR seam stages, so every arm that can reach a seam
    # is offered it.  A stage that draws nothing costs nothing here (each cell
    # is charged to the first stage that can take it), so the extra two are
    # free and they are what closes the last few per cent of the block.
    #
    # WHY THIS ONE IS NOT THE ANSWER — it offers SEAM1 only to arms 2 and 97,
    # whose certified cells stop at y = 2.280, and SEAM1 starts at y = 2.220.
    # `scripts/traces.py` (docs/V2_TRACES.md) found the hole from the other end:
    # CSAIL stroke 17 lies entirely inside it.  Kept in the table because it is
    # the version the 2026-09-11 recommendation shipped with, and the version
    # the corrected one has to be compared against.
    pats.append(evaluate(d, "3-active-rowband-y20+4seams", [
        {13: R["R0-y20"], 71: R["R1-y20"], 2: R["R2-y20"]},
        {17: R["R0-y20"], 31: R["R1-y20"], 97: R["R2-y20"]},
        {13: R["SEAM0-y20"], 97: R["SEAM1-y20"]},
        {17: R["SEAM0-y20"], 2: R["SEAM1-y20"]},
        {31: R["SEAM0-y20"], 97: R["SEAM1-y20"]},
        {71: R["SEAM0-y20"], 2: R["SEAM1-y20"]}],
        "one arm per full-width row band with a 0.40 m dead band in y, columns "
        "alternating, then four 2-active seam stages -- SEAM1 offered only to "
        "2 and 97, which is the hole"))

    # A SEAM NEEDS BOTH AN OUTER ARM AND A MIDDLE ARM, and that is arithmetic,
    # not taste.  Over the certified block the six arms' tip-y reach is
    #   13, 17: [0.000, 1.340]   31, 71: [1.080, 2.560]   2, 97: [2.280, 3.620]
    # and the two seams at a 0.40 m dead band are [1.010, 1.410] and
    # [2.220, 2.620].  So SEAM0's top 70 mm is reachable ONLY by 31/71 and its
    # bottom 70 mm only by 13/17; SEAM1's bottom 60 mm only by 31/71 and its
    # top 60 mm only by 2/97.  Each seam therefore has to be offered to one of
    # each kind, and since 31 and 71 are a transverse pair they have to take
    # opposite seams in the same stage -- which is the pairing measured in
    # section 2b.
    pats.append(evaluate(d, "3-active-rowband-y20+4seams-crossed", [
        {13: R["R0-y20"], 71: R["R1-y20"], 2: R["R2-y20"]},
        {17: R["R0-y20"], 31: R["R1-y20"], 97: R["R2-y20"]},
        {13: R["SEAM0-y20"], 97: R["SEAM1-y20"]},
        {17: R["SEAM0-y20"], 2: R["SEAM1-y20"]},
        {31: R["SEAM0-y20"], 71: R["SEAM1-y20"]},
        {71: R["SEAM0-y20"], 31: R["SEAM1-y20"]}],
        "the same six stages, but the last two CROSS the middle pair over the "
        "two seams instead of giving both of them SEAM0"))

    # ...and the conservative fallback, if the crossed transverse pair does not
    # clear: leave the four stages alone and add two more that pair a row-0 arm
    # on SEAM0 with a middle arm on SEAM1.
    pats.append(evaluate(d, "3-active-rowband-y20+6seams", [
        {13: R["R0-y20"], 71: R["R1-y20"], 2: R["R2-y20"]},
        {17: R["R0-y20"], 31: R["R1-y20"], 97: R["R2-y20"]},
        {13: R["SEAM0-y20"], 97: R["SEAM1-y20"]},
        {17: R["SEAM0-y20"], 2: R["SEAM1-y20"]},
        {31: R["SEAM0-y20"], 97: R["SEAM1-y20"]},
        {71: R["SEAM0-y20"], 2: R["SEAM1-y20"]},
        {13: R["SEAM0-y20"], 31: R["SEAM1-y20"]},
        {17: R["SEAM0-y20"], 71: R["SEAM1-y20"]}],
        "the four seam stages plus two more that put the MIDDLE pair on SEAM1, "
        "each against a row-0 arm on SEAM0"))

    # PATTERN B — 3-active, one per row, own COLUMN half only (2 stages)
    pats.append(evaluate(d, "3-active-block", [
        {13: R["B00"], 71: R["B11"], 2: R["B02"]},
        {17: R["B10"], 31: R["B01"], 97: R["B12"]}],
        "one arm per row, own Voronoi block only; the other column next stage"))
    pats.append(evaluate(d, "3-active-block-y15+seams", [
        {13: R["B00-y15"], 71: R["B11-y15"], 2: R["B02-y15"]},
        {17: R["B10-y15"], 31: R["B01-y15"], 97: R["B12-y15"]},
        {13: R["SEAM0-y15"], 97: R["SEAM1-y15"]},
        {71: R["SEAM0-y15"], 2: R["SEAM1-y15"]}],
        "one arm per row on its own Voronoi block, 0.15 m dead band in y"))

    # PATTERN C — 2-active, one per column, three stages
    pats.append(evaluate(d, "2-active-column", [
        {13: R["B00"], 97: R["B12"]},
        {31: R["B01"], 71: R["B11"]},
        {2: R["B02"], 17: R["B10"]}],
        "two arms a stage, one in each column, on their own blocks"))

    # PATTERN D — 6-active on eroded blocks, one stage, then a seam stage
    for s in (0.10, 0.15, 0.20, 0.30):
        tag = f"-e{int(s * 100):02d}"
        seam0 = R["ALL"] & ~np.logical_or.reduce(
            [R[f"B{COL_OF[x]}{ROW_OF[x]}{tag}"] for x in ARMS])
        pats.append(evaluate(d, f"6-active{tag}+seams", [
            eroded(R, s),
            {13: seam0 & R["R0"], 71: seam0 & R["R1"], 2: seam0 & R["R2"]},
            {17: seam0 & R["R0"], 31: seam0 & R["R1"], 97: seam0 & R["R2"]}],
            f"all six at once inside blocks eroded {s:.2f} m, then the seams "
            "in two 3-active stages"))

    # PATTERN E — Pete's 1-2-1, both readings, at the depth the ladder allows
    pats.append(evaluate(d, "121-middle-leads", [
        {31: R["B01"], 71: R["B11"], 13: R["S0-060"], 17: R["S1-060"],
         2: R["N0-302"], 97: R["N1-302"]},
        {13: R["B00"] & ~R["S0-060"], 17: R["B10"] & ~R["S1-060"],
         2: R["B02"] & ~R["N0-302"], 97: R["B12"] & ~R["N1-302"]}],
        "Pete (a): 31/71 lead on full middle blocks while the four end arms "
        "work only beyond their own base row; then the ends finish"))
    pats.append(evaluate(d, "121-ends-lead", [
        {13: R["B00"], 17: R["B10"], 2: R["B02"], 97: R["B12"],
         31: R["M0-040"], 71: R["M1-040"]},
        {31: R["B01"] & ~R["M0-040"], 71: R["B11"] & ~R["M1-040"]}],
        "Pete (b): the four end arms lead on full blocks while 31/71 fill a "
        "narrow middle band; then the middle finishes"))

    # PATTERN F — the serial control: one arm at a time, six stages
    pats.append(evaluate(d, "serial-6", [{a: R["ALL"]} for a in
                                         (13, 17, 31, 71, 2, 97)],
                         "the baseline: one arm active at a time, whole block"))

    print(f"\n  {'pattern':<26}{'stages':>7}{'cover':>8}{'ink':>10}{'+parks':>10}"
          f"{'cost':>8}{'speedup':>9}")
    for p in pats:
        print(f"  {p['name']:<26}{p['n_stages']:>7}"
              f"{100 * p['coverage']:>7.1f}%{p['worst_ink_mm']:>7.1f}mm"
              f"{p['worst_env_mm']:>8.1f}mm"
              f"{p['cost']:>8.0f}{p['speedup']:>8.2f}x")
    print("\n  where the uncovered cells are (y, m):")
    for p in pats:
        if p["uncovered"]:
            uy = p["uncovered_y"]
            print(f"    {p['name']:<28}{p['uncovered']:>5} cells at y "
                  f"{uy[0]:.2f}..{uy[-1]:.2f} ({len(uy)} rows)")
        else:
            print(f"    {p['name']:<28}    - none, 100 % of the block")
    report["patterns"] = pats

    # ---- 3. transitions --------------------------------------------------
    print("\n" + "=" * 74)
    print("3. STAGE TRANSITIONS — is the shipped park inside the next cell?")
    print("=" * 74)
    park_in = {}
    for arm in ARMS:
        hx, hy = layout.PARK_HOVER_PROPOSED[arm]
        names = ("ALL", f"B{COL_OF[arm]}{ROW_OF[arm]}", f"R{ROW_OF[arm]}",
                 f"C{COL_OF[arm]}")
        inside = {n: region_holds(R[n], xs, ys, hx, hy) for n in names}
        park_in[arm] = dict(hover_xy=[hx, hy], inside=inside)
        blk_s = "yes" if inside["ALL"] else "NO (off the certified block)"
        own_s = "yes" if inside[names[1]] else "no"
        row_s = "yes" if inside[names[2]] else "no"
        print(f"  arm {arm:>2}: park hover at ({hx:.3f}, {hy:.3f})  "
              f"in block: {blk_s:<28} own cell: {own_s:<4} "
              f"own row band: {row_s}")
    report["park_in_cell"] = park_in

    print("\n  park vs every OTHER arm's whole-block envelope (mm):")
    pv = {}
    for arm in ARMS:
        worst, who = float("inf"), None
        for b in ARMS:
            if b == arm:
                continue
            v = park_vs(d, arm, b, cell_sel(d, b, R["ALL"]))
            if v < worst:
                worst, who = v, b
        pv[arm] = dict(worst_mm=round(1000 * worst, 1), against=who)
        print(f"    arm {arm:>2}: {1000 * worst:>7.1f} mm, worst against arm {who}")
    report["park_vs_all_ink"] = pv

    # ---- 4. redundancy ----------------------------------------------------
    print("\n" + "=" * 74)
    print("4. REDUNDANCY — cells with a second arm that could take them over")
    print("=" * 74)
    reach_blk = reach & blk
    red = {}
    for k in (1, 2, 3, 4, 5, 6):
        red[k] = float(((n_arms >= k) & blk).sum() / max(reach_blk.sum(), 1))
        print(f"  >= {k} certified drawers: {100 * red[k]:>5.1f} % of the "
              f"reachable block")
    report["redundancy_any_stage"] = {str(k): red[k] for k in red}

    # ...and the STAGE-COMPATIBLE version, per pattern
    print("\n  and with a second drawer the PATTERN can actually use:")
    report["stage_redundancy"] = redundancy_by_stage(d, pats)
    for name, r in report["stage_redundancy"].items():
        print(f"    {name:<26} {100 * r['frac_ge2']:>5.1f} % of the reachable "
              f"block has >= 2 stage-compatible drawers "
              f"(mean {r['mean_drawers']:.2f})")

    for p in pats:
        p.pop("_masks", None)
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(report, indent=1, default=float))
    print(f"\nwrote {a.json}")


def region_holds(mask, xs, ys, x, y):
    """Does `mask` contain the paper point (x, y)? -> bool (False if off grid)."""
    if not (xs[0] - 1e-9 <= x <= xs[-1] + 1e-9
            and ys[0] - 1e-9 <= y <= ys[-1] + 1e-9):
        return False
    j = int(round((x - xs[0]) / (xs[1] - xs[0])))
    i = int(round((y - ys[0]) / (ys[1] - ys[0])))
    return bool(mask[min(max(i, 0), len(ys) - 1), min(max(j, 0), len(xs) - 1)])


def redundancy_by_stage(d, pats):
    """Which cells a pattern could hand to a SECOND arm. -> {name: dict}.

    The fault-tolerance question: arm `a` dies, and the cells it had left are
    re-queued.  A cell is re-queueable only to an arm that both CERTIFIES it
    and is OFFERED it by some stage of the pattern — a stage assignment that
    was never measured is not a fallback, it is a new coordination problem.
    So the count is: how many distinct arms `a` have a stage `s` with the cell
    inside `region_s(a)` and the cell in `a`'s certified set.
    """
    out = {}
    blk, reach = d["block_mask"], d["reach_any"] & d["block_mask"]
    H, W = blk.shape
    cert = {}
    for arm in ARMS:
        c = d["cells"][arm]
        m = np.zeros((H, W), bool)
        m[c["iy"], c["ix"]] = True
        cert[arm] = m
    for p in pats:
        n = np.zeros((H, W), int)
        for arm in ARMS:
            offered = np.zeros((H, W), bool)
            for stage in p.get("_masks", []):
                msk = stage.get(arm)
                if msk is not None:
                    offered |= msk
            n += (offered & cert[arm]).astype(int)
        tot = max(int(reach.sum()), 1)
        out[p["name"]] = dict(
            frac_ge2=float(((n >= 2) & reach).sum() / tot),
            frac_ge3=float(((n >= 3) & reach).sum() / tot),
            mean_drawers=float(n[reach].mean()),
            orphans=int(((n == 0) & reach).sum()))
    return out


if __name__ == "__main__":
    main()
