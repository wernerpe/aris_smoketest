"""Piecewise-linear planning in the (s, q7) redundancy band.

The lattice planner in `planner.py` answers "which joint configuration at every
one of the 131 arc-length steps?" — a 131-node schedule that happens to live on
a grid.  This module asks the *representation* question first: what is the
smallest description of the redundancy decision that is still feasible?

THE PICTURE.  Pin the pen to the paper along the stroke and the only freedom
left is q7 (yaw is degenerate with it, see planner.py) plus a discrete IK
branch.  Plot q7 against arc length s and the feasible set is a 2-D *band* — a
terrain whose height is sigma_min (controllability) and whose holes are the
postures that fail the gates.  A plan is then simply a curve q7(s) crossing
that terrain from s = 0 to s = 1, and:

  * **sheets** are the connected components of the band.  Two grid nodes belong
    to the same sheet iff they are neighbours in (s, q7) *and* their
    configurations are continuous (||dq||_inf <= JUMP_THRESH).  That is exactly
    the DP's edge test, so a sheet is "the set of postures the arm can reach
    from here without a branch flip".  A sheet has a single-valued
    representative q per cell, which turns the (s x q7 x branch) lattice into a
    plain 2-D field per sheet.
  * **clearance** is the distance (in grid units) from a point of the band to
    the nearest hole or joint-limit wall.  Planning down the middle of the
    corridor is singularity avoidance stated in the redundancy space itself,
    not as a per-node penalty: it is what buys the freedom to *straighten* the
    path afterwards.
  * **the plan is a polyline**: q7(s) monotone in s (a function, by
    construction — one q7 per arc length) simplified by Ramer-Douglas-Peucker
    down to a handful of knots, with every candidate segment corridor-checked
    against the same sheet's free region.  A stroke's redundancy schedule stops
    being 131 numbers and becomes ~5 knots, which is what one would actually
    hand to a controller, log, or re-time.

Then `backout()` turns knots back into joint configurations by re-solving the
case-consistent IK at every dense sample.  It never interpolates q — see its
docstring for why that is not a stylistic choice.
"""
import numpy as np

from . import ik, planner
from .frames import (PEN_EXT, QD_MAX, joint_margin, rotx, rotz, tip_pos_many,
                     lat_of, tool_offset)
from .metrics import sigma_min as _sigma_min, tip_jacobian_many
from .pacing import dq_ds

SIGMA_GATE = 0.10        # band gate, sigma_min (above planner.HARD_SIGMA = 0.08)
MARGIN_GATE = 0.15       # band gate, joint margin (rad)
W_CLEARANCE = 1.0        # tie-break pull toward the middle of the corridor
W_TRAVEL = 0.25          # tie-break pull toward fewer q7 index steps (maximin only)
RDP_EPS = 1.0            # simplification tolerance, q7 GRID INDICES
_SIGMA_Q = planner._SIGMA_Q      # bottleneck quantum, shared with the lattice DP
_TRAVEL_Q = 1e6          # travel quantum (1e-6 rad), for an EXACT lexicographic DP
# THE DEFAULT IS THE BOTTLENECK OBJECTIVE, AND THAT IS A MEASUREMENT, NOT A
# PREFERENCE.  "min_travel" is implemented, tested and available, and on the
# band it does exactly what it claims: -68 % of q7 wander, -32 % of lattice
# travel, -13 % of knots, both gates untouched.  It also costs the CLOCK, which
# is the one thing this project is allowed to optimise.  Constraining the band
# path raises |dq/ds|; `writing.draw_duration` then stretches the ink until no
# joint exceeds `qd_frac` of its velocity limit; and drawing is the dominant
# term in the makespan floor.  On the shipped CSAIL run the busiest grey arm's
# draw time went 20.2 -> 30.9 s over the SAME 2.00 m of ink, and the two-pass
# floor went 84.3 -> 93.2 s.  docs/REDUNDANCY.md has the measurement.
#
# AND THAT MEASUREMENT IS AS MUCH ABOUT `writing.QD_FRAC` = 0.30 AS ABOUT THE
# BAND.  The stretch is what the cap scales: the speed below which no plan is
# joint-limited anywhere is v* = length / need, and it is LINEAR in `qd_frac`,
# so the cap sets the range of draw speeds over which this argument bites at
# all.  At the rig's own 0.02 m/s it barely bites even at 0.30 (docs/BENCH.md's
# speed sweep: 1 to 2 of ~57 segments capped, ink 1.006x the material's time),
# which is why the verdict above is a verdict at 0.12 m/s and says so.
OBJECTIVE = "maximin_sigma"      # default band objective; see `_dense_dp`
TRAVEL_MODE = "time"             # how an edge is charged; see `_edge_travel`
OBJECTIVES = ("min_travel", "maximin_sigma")


# --------------------------------------------------------------------------
# 0. the certification chase — one implementation, three callers
# --------------------------------------------------------------------------
def arc_length(points):
    """Total polyline length in METRES.

    Normalised s is the right coordinate to plan in and the wrong one to move
    in: every velocity statement downstream (pacing.py) needs the stroke's real
    scale back, so it is carried alongside s rather than recovered later.
    """
    p = np.asarray(points, float)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def pen_down_poses(pts_xy, Twb_inv, pen_ext, phi=0.0, pen_lat=None):
    """(M,4,4) base-frame hand-TCP poses for a pen-down stroke.

    The pen convention: R = rotz(phi) @ rotx(pi) points tool z straight down
    (planner.py explains why phi is pinned to 0 for the INLINE pen, where it
    aliases with q7), and the tip sits at TCP + R @ (pen_lat, 0, pen_ext) —
    so the TCP is `pen_ext` ABOVE and `pen_lat` BESIDE the paper point it is
    drawing, the lateral direction chosen by phi.
    """
    p = np.asarray(pts_xy, float)
    R = rotz(phi) @ rotx(np.pi) if phi else rotx(np.pi)
    T = np.tile(np.eye(4), (len(p), 1, 1))
    T[:, :3, :3] = R
    T[:, :3, 3] = np.column_stack([p[:, 0], p[:, 1], np.zeros(len(p))]) \
        - R @ tool_offset(pen_ext, pen_lat)
    return np.asarray(Twb_inv, float) @ T


def chase_cc(poses, q7, q_seed, pen_ext=PEN_EXT, margin_gate=None,
             sigma_gate=None, jump_gate=None, fallback=False, pen_lat=None):
    """Walk a stroke with case-consistent IK.  THE feasibility test of the
    project: search on the grid, certify against the real kinematics.

    Given the stroke's poses (M,4,4), a commanded q7 per sample and a seed
    configuration, `ik.solve_cc` is chased sample to sample — the same branch
    throughout, each solve seeded with the previous configuration.  sigma_min
    and the joint margin are computed for every sample that is accepted.

    Each of the three gates is enforced only when its value is not None;
    otherwise it is merely measured.  That single switch is the whole
    difference between the module's callers:

      * `Corridor._chase_ok` (segment screening) passes all three and wants the
        early exit — an infeasible candidate segment should cost as little as
        possible;
      * `backout()` passes none and never gives up early: it reports what the
        plan actually does, including a fallback to a full `ik.solve` when the
        case-consistent solve dies (`fallback=True`);
      * `smooth.certify` passes all three, so a completed walk IS the
        certificate and its `qs` are the deliverable — no second pass.

    Returns dict: ok (walked every sample), n, qs (n,7), sigmas, margins,
    max_step (rad, ||dq||_inf), fallbacks, fails, stop ("ok"/"seed"/"no_ik"/
    "jump"/"margin"/"sigma"), stop_index.
    """
    poses = np.asarray(poses, float)
    q7 = np.atleast_1d(np.asarray(q7, float))
    q_prev = np.asarray(q_seed, float)
    M = len(poses)
    out = dict(ok=False, n=0, qs=np.zeros((0, 7)), sigmas=np.zeros(0),
               margins=np.zeros(0), max_step=0.0, fallbacks=0, fails=0,
               stop="seed", stop_index=0)
    if q_prev.shape != (7,) or not np.all(np.isfinite(q_prev)):
        return out

    qs, sig, mar = [], [], []
    fallbacks = fails = 0
    stop, stop_i, max_step = "ok", M, 0.0
    for n in range(M):
        q = ik.solve_cc(poses[n], float(q7[n]), q_prev)
        if q is None and fallback:
            cand = ik.solve(poses[n], float(q7[n]), q_prev)
            if cand:
                q = min(cand, key=lambda c: np.max(np.abs(c - q_prev)))
                fallbacks += 1
        if q is None:
            fails, stop, stop_i = fails + 1, "no_ik", n
            break
        if n:
            step = float(np.max(np.abs(q - q_prev)))
            if jump_gate is not None and step > jump_gate:
                stop, stop_i = "jump", n
                break
            max_step = max(max_step, step)
        m = joint_margin(q)
        if margin_gate is not None and m < margin_gate:
            stop, stop_i = "margin", n
            break
        # analytic Jacobian: the chase is sequential (each solve is seeded by
        # the last), so this is the one place a per-sample call still pays —
        # but it need not be 14 forward kinematics.
        s = _sigma_min(tip_jacobian_many(q[None], pen_ext=pen_ext,
                                         pen_lat=pen_lat)[0])
        if sigma_gate is not None and s < sigma_gate:
            stop, stop_i = "sigma", n
            break
        qs.append(q)
        mar.append(m)
        sig.append(s)
        q_prev = q
    out.update(ok=bool(len(qs) == M), n=len(qs),
               qs=np.array(qs) if qs else np.zeros((0, 7)),
               sigmas=np.array(sig), margins=np.array(mar), max_step=max_step,
               fallbacks=fallbacks, fails=fails, stop=stop, stop_index=stop_i)
    return out


# --------------------------------------------------------------------------
# 1. sheets: connected components of the (s x q7 x branch) lattice
# --------------------------------------------------------------------------
def _uf_find(parent, a):
    r = a
    while parent[r] != r:
        r = parent[r]
    while parent[a] != r:            # path compression
        parent[a], a = r, parent[a]
    return r


def sheet_fields(lat, jump=planner.JUMP_THRESH):
    """Label IK sheets by continuity and flatten each one to 2-D fields.

    Nodes are the valid lattice entries (s_i, q7_j, branch_k).  Two nodes are
    joined iff they are grid neighbours — (ds, dq7) in {(0,+-1), (+1,-1),
    (+1,0), (+1,+1)} — and ||dq||_inf <= `jump`.  This is the DP's own edge
    test, so components are "reachable without a branch flip"; branch *labels*
    are never used (the analytic solver's branch order is not stable along a
    stroke, its continuity is).

    Returns a list of sheet dicts, largest first:
        id       index in this list (0 = dominant)
        sel      (Ns, Nq, Nb) bool, the component's lattice nodes
        mask     (Ns, Nq) bool, cells the sheet occupies
        sigma    (Ns, Nq) sigma_min of the representative node, nan off-mask
        margin   (Ns, Nq) likewise
        branch   (Ns, Nq) int32 representative branch index, -1 off-mask
        Q        (Ns, Nq, 7) representative configuration, nan off-mask
        nodes    number of lattice nodes in the component
        i0, i1   s-index extent, spans_s = (i0 == 0 and i1 == Ns - 1)
    The representative at a cell is the sheet's highest-sigma branch there
    (cells with two branches of one sheet are rare; they are self-motion
    duplicates, and the DP would pick the better one anyway).
    """
    V, Q, S, M = lat["valid"], lat["Q"], lat["sigma"], lat["margin"]
    Ns, Nq, Nb = V.shape
    Qf = planner._fill(lat)
    nid = np.full((Ns, Nq, Nb), -1, np.int64)
    n_nodes = int(V.sum())
    nid[V] = np.arange(n_nodes)
    parent = np.arange(n_nodes)

    for di, dj in ((0, 1), (1, -1), (1, 0), (1, 1)):
        jlo, jhi = max(0, -dj), min(Nq, Nq - dj)
        if jlo >= jhi or Ns - di <= 0:
            continue
        src = (slice(0, Ns - di), slice(jlo, jhi))
        dst = (slice(di, Ns), slice(jlo + dj, jhi + dj))
        d = np.max(np.abs(Qf[src][:, :, :, None, :] - Qf[dst][:, :, None, :, :]),
                   axis=-1)
        good = (V[src][:, :, :, None] & V[dst][:, :, None, :] & (d <= jump))
        a, b = np.broadcast_arrays(nid[src][:, :, :, None], nid[dst][:, :, None, :])
        for u, v in zip(a[good], b[good]):
            ru, rv = _uf_find(parent, u), _uf_find(parent, v)
            if ru != rv:
                parent[max(ru, rv)] = min(ru, rv)

    root = np.array([_uf_find(parent, i) for i in range(n_nodes)])
    labels, counts = np.unique(root, return_counts=True)
    order = np.argsort(-counts)
    lab3 = np.full((Ns, Nq, Nb), -1, np.int64)
    lab3[V] = root

    sheets = []
    for sid, li in enumerate(order):
        lab = labels[li]
        sel = lab3 == lab                                   # (Ns, Nq, Nb)
        mask = sel.any(axis=2)
        rank = np.where(sel, S, -np.inf)
        kb = np.argmax(rank, axis=2)
        take = np.take_along_axis
        kb3 = kb[:, :, None]
        sig = np.where(mask, take(S, kb3, 2)[:, :, 0], np.nan)
        mar = np.where(mask, take(M, kb3, 2)[:, :, 0], np.nan)
        qrep = np.where(mask[:, :, None], take(Q, kb3[:, :, None], 2)[:, :, 0, :], np.nan)
        rows = np.flatnonzero(mask.any(axis=1))
        sheets.append(dict(id=sid, mask=mask, sel=sel, sigma=sig, margin=mar,
                           branch=np.where(mask, kb, -1).astype(np.int32), Q=qrep,
                           nodes=int(counts[li]), i0=int(rows[0]), i1=int(rows[-1]),
                           spans_s=bool(rows[0] == 0 and rows[-1] == Ns - 1)))
    return sheets


def dominant_sheet(sheets):
    """Largest sheet that spans the whole stroke (else simply the largest)."""
    for sh in sheets:
        if sh["spans_s"]:
            return sh
    return sheets[0]


def sheet_of_path(sheets, path):
    """Which sheet a lattice-DP path lies on: (sheet id, fraction of its steps).

    `path` is planner.plan()'s list of (q7 index, branch index) per s step.
    """
    best = (-1, 0.0)
    for sh in sheets:
        f = float(np.mean([sh["sel"][i, j, kb] for i, (j, kb) in enumerate(path)]))
        if f > best[1]:
            best = (int(sh["id"]), f)
    return best


def sheet_report(sheets, Ns, top=6):
    """Terse per-sheet lines: node count and s-extent."""
    out = [f"{len(sheets)} sheets"]
    for sh in sheets[:top]:
        out.append(f"  sheet {sh['id']}: {sh['nodes']:5d} nodes, "
                   f"s in [{sh['i0'] / (Ns - 1):.3f}, {sh['i1'] / (Ns - 1):.3f}]"
                   + ("  <- spans s" if sh["spans_s"] else ""))
    if len(sheets) > top:
        rest = sum(sh["nodes"] for sh in sheets[top:])
        out.append(f"  ... {len(sheets) - top} more, {rest} nodes total")
    return out


# --------------------------------------------------------------------------
# 2. clearance + the dense path + PWL simplification
# --------------------------------------------------------------------------
def clearance_map(free):
    """Chamfer (1, sqrt2) distance transform of `free`, in grid units.

    scipy.ndimage.distance_transform_edt would do this in one call; scipy is
    not installed under the system python3.12 this project runs on, so here is
    the classic two-pass version.  Within a row the recurrence
    d[j] = min(d[j], d[j-1] + 1) is done with the running-min identity
    d[j] = j + min_{k<=j}(d[k] - k), i.e. one np.minimum.accumulate.

    WALLS.  The two q7 rows just outside the grid count as obstacles — they are
    joint limits, real walls.  The s ends do NOT: the stroke has to reach s = 0
    and s = 1, so clearance there must not be eroded for being an endpoint.
    """
    free = np.asarray(free, bool)
    Ns, Nq = free.shape
    pad = np.zeros((Ns, Nq + 2), bool)          # obstacle columns = q7 limits
    pad[:, 1:-1] = free
    C1, C2 = 1.0, np.sqrt(2.0)
    d = np.where(pad, np.inf, 0.0)
    jj = np.arange(Nq + 2, dtype=float)

    def _row(prev, cur):
        if prev is not None:
            up = prev + C1
            ul = np.concatenate([[np.inf], prev[:-1] + C2])
            ur = np.concatenate([prev[1:] + C2, [np.inf]])
            cur = np.minimum(cur, np.minimum(up, np.minimum(ul, ur)))
        return cur

    for i in range(Ns):                          # forward pass
        cur = _row(d[i - 1] if i else None, d[i])
        d[i] = np.minimum.accumulate(cur - jj) + jj
    for i in range(Ns - 1, -1, -1):              # backward pass
        cur = _row(d[i + 1] if i < Ns - 1 else None, d[i])
        rev = cur[::-1]
        d[i] = (np.minimum.accumulate(rev - jj) + jj)[::-1]
    return d[:, 1:-1]


def _edge_ok(sheet, jump):
    """(Ns-1, Nq, 3) bool: representative configs of (i, j) and (i+1, j+dj)
    are within the continuity budget.  Same-sheet membership only promises a
    path exists *somewhere* in the component; the plan needs the direct edge."""
    Q, mask = sheet["Q"], sheet["mask"]
    Ns, Nq = mask.shape
    out = np.zeros((Ns - 1, Nq, 3), bool)
    for t, dj in enumerate((-1, 0, 1)):
        lo, hi = max(0, -dj), min(Nq, Nq - dj)
        if lo >= hi:
            continue
        a, b = Q[:-1, lo:hi], Q[1:, lo + dj:hi + dj]
        d = np.max(np.abs(a - b), axis=-1)
        out[:, lo:hi, t] = (mask[:-1, lo:hi] & mask[1:, lo + dj:hi + dj]
                            & (np.nan_to_num(d, nan=np.inf) <= jump))
    return out


def _edge_travel(sheet, mode="separable"):
    """(Ns-1, Nq, 3) joint travel of every lattice edge, sum_j |dq_j|.

    INDEXED BY SOURCE, exactly like `_edge_ok`: entry [i, j, t] is the edge
    LEAVING (i, j) with dj = t - 1.  The two arrays are consumed together and a
    mismatched convention between them would be silent, so they are built the
    same way on purpose.  The norm is L1 over the seven joints because that is
    the project's own `sum_travel` (`chase_report`, `planner.path_report`,
    `smooth_report`): the quantity the DP minimises is then the quantity every
    report downstream prints, rather than a proxy that correlates with it.

    TWO WAYS TO PRICE AN EDGE, AND THE DIFFERENCE IS THE STAIRCASE.

    `mode="chord"` is the obvious one: the straight-line joint distance
    ||Q[i+1, j+dj] - Q[i, j]||_1 between the two lattice nodes.  It prices the
    path the DP literally walks — and that path is NOT the one that gets
    executed.  A dense DP crossing the band at half an index per step has to
    alternate dj = 0, 1, 0, 1; RDP then straightens that staircase into a
    ramp, and the ramp's travel is what the arm actually pays.  Charging the
    staircase makes the DP optimise a quantity the simplification is about to
    throw away, which is measurable: over the 35 certified CSAIL stroke/arm
    pairs the chord cost cut the LATTICE travel 31.8 % and moved the certified
    dense travel the wrong way by 2.0 %.

    `mode="separable"` (the default) prices the ramp instead, by splitting the
    edge into the two motions it is made of:

        travel = ||Q[i, j+dj] - Q[i, j]||_1        the null-space step sideways
               + ||Q[i+1, j+dj] - Q[i, j+dj]||_1   following the stroke there

    The first term is charged per index of q7 actually crossed, so a staircase
    and the ramp it approximates cost the SAME (both cross the same indices) —
    the DP stops caring about quantisation noise and starts caring about the
    net excursion.  The second term is the cost of advancing the stroke at the
    column the edge lands in, which is what makes the objective prefer the part
    of the band where dragging the pen forward is cheap in joint space.  That
    second term is the one the chord cost hides, and it is where the dense
    travel was going.

    Infeasible, off-sheet or off-mask edges are +inf so they can never be
    chosen; the caller masks with `_edge_ok` as well, which is belt and braces
    and costs nothing.
    """
    if mode not in ("separable", "chord", "time"):
        raise ValueError(f"unknown edge-travel mode {mode!r}")
    Q, mask = sheet["Q"], sheet["mask"]
    Ns, Nq = mask.shape
    out = np.full((Ns - 1, Nq, 3), np.inf)

    def norm(d):
        """L1 over the joints, or SECONDS at full joint speed for mode=time."""
        return (np.max(np.abs(d) / QD_MAX, axis=-1) if mode == "time"
                else np.sum(np.abs(d), axis=-1))

    for t, dj in enumerate((-1, 0, 1)):
        lo, hi = max(0, -dj), min(Nq, Nq - dj)
        if lo >= hi:
            continue
        src, dst = slice(lo, hi), slice(lo + dj, hi + dj)
        ok = mask[:-1, src] & mask[1:, dst]
        chord = norm(Q[:-1, src] - Q[1:, dst])
        if mode == "chord":
            d = chord
        else:
            side = norm(Q[:-1, dst] - Q[:-1, src]) if dj else 0.0
            fwd = norm(Q[1:, dst] - Q[:-1, dst])
            d = side + fwd
            if dj:
                # AN UNDEFINED PRICE MUST NOT BECOME AN INFINITE ONE.  The
                # separable price routes the edge through the intermediate node
                # (i, j+dj), and where the sheet has no node there the two
                # terms read NaN.  Letting that become +inf would DELETE an
                # edge `_edge_ok` has already certified — a cost model acting
                # as a feasibility gate — and the DP cannot tell "expensive"
                # from "forbidden", so the band would be reported as
                # disconnected at a step where it is not.  Measured on the
                # CSAIL bands: 21 of 1452 certified edges (1.45 %) priced
                # +inf this way, which is enough to concede false splits and
                # to shrink the fiber menus.  The chord is always defined for
                # an edge whose two ENDPOINTS are on the sheet, so it is what
                # those edges are charged.
                d = np.where(mask[:-1, dst], d, chord)
        out[:, src, t] = np.where(ok, np.nan_to_num(d, nan=np.inf,
                                                    posinf=np.inf), np.inf)
    return out


def _start_mask(free0, j_start):
    """Row of admissible s = 0 cells, optionally pinned to one q7 index."""
    m = np.asarray(free0, bool).copy()
    if j_start is not None:
        keep = np.zeros(m.shape, bool)
        j = int(j_start)
        if 0 <= j < len(m):
            keep[j] = True
        m &= keep
    return m


def _backtrack(parent, j_end, Ns):
    """Walk the dq7-index backpointers from a terminal q7 index to s = 0."""
    js = [int(j_end)]
    for i in range(Ns - 1, 0, -1):
        k = int(parent[i][js[-1]])
        assert k >= 0, f"missing parent at step {i}"
        js.append(js[-1] - (k - 1))
    return np.array(js[::-1])


def travel_forward(free, clear, edge_ok, edge_travel, w_clearance=W_CLEARANCE,
                   j_start=None):
    """One forward sweep of the GATED MIN-TRAVEL DP. -> (A, C, parent, last).

    `A[j]` is the least total joint travel (QUANTISED — see below) with which
    s = Ns-1 can be reached at q7 index j, `C[j]` the accompanying clearance
    tie-break cost, both +inf where j is not reachable.  `parent` holds the
    dq7-index backpointers and `last` the farthest s reached (< Ns-1 means the
    gated band disconnected, which is the caller's split signal).

    THE OBJECTIVE, AND WHY IT IS THE WHOLE POINT.  The gates are not in the
    cost: sigma_min >= sigma_gate and margin >= margin_gate have already carved
    the free region this DP runs on, so they are HARD and a path either respects
    them or does not exist.  What is left to choose among the admissible paths
    is then a plain shortest-path question — minimise the joint travel the arm
    actually spends — and not a bottleneck question.  Maximising min-sigma over
    a region where every cell already clears the gate buys controllability the
    gate has already bought, and pays for it in q7 wander: the maximin path will
    climb the band to sit on a ridge and climb back down, and every radian of
    that climb is a null-space self-motion the pen does not need.

    EXACTLY LEXICOGRAPHIC, NOT NEARLY.  Travel is quantised to 1e-6 rad and
    accumulated as an integer in a float64 (a 15 m stroke tops out around 1e10,
    against the 9e15 where float64 stops counting exactly), so `A == A.min()`
    is a true equality test and the clearance tie-break is applied to exactly
    the set of optimal paths.  Both terms are additive, so unlike the maximin
    DP — whose tie-break is only greedy-lexicographic, as `planner.plan` says —
    this one is exact in both components.  Clearance stays the tie-break for
    the reason `plan_pwl` wants it: among equally short paths, the one down the
    middle of the corridor is the one RDP can straighten.
    """
    Ns, Nq = free.shape
    INF = np.inf
    Tq = np.where(edge_ok, np.round(np.where(np.isfinite(edge_travel),
                                             edge_travel, 0.0) * _TRAVEL_Q), INF)
    Tq = np.where(np.isfinite(edge_travel), Tq, INF)
    node_c = np.where(free, -w_clearance * clear, INF)
    parent = np.full((Ns, Nq), -1, np.int8)
    start = _start_mask(free[0], j_start)
    A = np.where(start, 0.0, INF)
    C = np.where(start, node_c[0], INF)
    last = 0 if np.isfinite(A).any() else -1
    for i in range(1, Ns):
        ca = np.full((Nq, 3), INF)
        cc = np.full((Nq, 3), INF)
        for t, dj in enumerate((-1, 0, 1)):
            lo, hi = max(0, dj), min(Nq, Nq + dj)
            if lo >= hi:
                continue
            tgt, src = slice(lo, hi), slice(lo - dj, hi - dj)
            feas = free[i][tgt] & np.isfinite(A[src]) & edge_ok[i - 1][src, t]
            ca[tgt, t] = np.where(feas, A[src] + Tq[i - 1][src, t], INF)
            cc[tgt, t] = np.where(feas, C[src] + node_c[i][tgt], INF)
        best = ca.min(axis=1)
        k = np.argmin(np.where(ca == best[:, None], cc, INF), axis=1)
        An = np.take_along_axis(ca, k[:, None], 1)[:, 0]
        if not np.isfinite(An).any():
            break
        A = An
        C = np.take_along_axis(cc, k[:, None], 1)[:, 0]
        parent[i], last = np.where(np.isfinite(An), k, -1), i
    return A, C, parent, last


def _dense_dp_travel(free, clear, edge_ok, edge_travel, w_clearance,
                     j_start=None, j_end=None):
    """Gated min-travel path across the band. -> (js | None, last, travel rad).

    `j_start` / `j_end` pin the entry / exit q7 index (the fiber-menu case);
    left None they are free and the DP picks the cheapest pair.  A None path
    with `last == Ns - 1` means the band spans the stroke but not to the exit
    that was asked for — which is exactly the reachability question the menu
    prunes on, answered by the band DP itself rather than guessed.
    """
    Ns = free.shape[0]
    A, C, parent, last = travel_forward(free, clear, edge_ok, edge_travel,
                                        w_clearance, j_start)
    if last < Ns - 1:
        return None, last, np.inf
    fin = np.isfinite(A)
    if j_end is not None:
        pick = np.zeros(fin.shape, bool)
        j = int(j_end)
        if 0 <= j < len(fin):
            pick[j] = True
        fin &= pick
    if not fin.any():
        return None, last, np.inf
    lo = A[fin].min()
    tie = fin & (A == lo)
    j = int(np.argmin(np.where(tie, C, np.inf)))
    return _backtrack(parent, j, Ns), last, float(lo) / _TRAVEL_Q


def _dense_dp(free, sigma, clear, edge_ok, w_clearance, w_travel,
              j_start=None, j_end=None):
    """Monotone-in-s DP over the free cells, transitions dq7 index in {-1,0,+1}.

    OBJECTIVE (same shape as planner.plan, deliberately): lexicographic
    (maximise the bottleneck sigma_min along the path, then minimise
    sum(w_travel*|dj| - w_clearance*clearance)).  The project's whole stance is
    that a stroke is only as controllable as its worst step, so sigma stays the
    primary and cannot be averaged away; clearance enters as the tie-break,
    where it does the job the plain DP has no notion of — among the many paths
    that share the optimal bottleneck it picks the one running down the middle
    of the corridor, which is what survives being straightened into segments.
    (Exact for the primary bottleneck; the tie-break is greedy-lexicographic,
    exactly as in planner.plan.)

    SINCE THE MIN-TRAVEL OBJECTIVE BECAME THE DEFAULT this is the FALLBACK
    rather than the usual path: `stroke_api` reaches for it when the gated
    shortest path fails to certify on every sheet.  A min-travel path is
    entitled to run along the gate boundary — every cell it uses clears
    sigma_gate, but only just — and the 5 mm chase that has the last word
    samples BETWEEN the lattice's 10 mm nodes, where "only just" can become
    "not quite".  Maximin buys the margin back by construction, so it is the
    right thing to fall back to and the wrong thing to start from.
    `j_start` / `j_end` pin the entry / exit q7 index, so a fiber-menu variant
    can be re-planned here on exactly the endpoints it was costed on.
    """
    Ns, Nq = free.shape
    NEG, INF = -np.inf, np.inf
    Sq = np.where(free, np.round(np.nan_to_num(sigma) * _SIGMA_Q), NEG)
    node_c = np.where(free, -w_clearance * clear, INF)
    parent = np.full((Ns, Nq), -1, np.int8)
    start = _start_mask(free[0], j_start)
    B = np.where(start, Sq[0], NEG)
    C = np.where(start, node_c[0], INF)
    last = 0 if np.isfinite(C).any() else -1
    for i in range(1, Ns):
        cb = np.full((Nq, 3), NEG)
        cc = np.full((Nq, 3), INF)
        for t, dj in enumerate((-1, 0, 1)):
            lo, hi = max(0, dj), min(Nq, Nq + dj)
            if lo >= hi:
                continue
            tgt, src = slice(lo, hi), slice(lo - dj, hi - dj)
            feas = free[i][tgt] & (B[src] > NEG) & edge_ok[i - 1][src, t]
            cb[tgt, t] = np.where(feas, np.minimum(B[src], Sq[i][tgt]), NEG)
            cc[tgt, t] = np.where(feas, C[src] + w_travel * abs(dj) + node_c[i][tgt], INF)
        best = cb.max(axis=1)
        k = np.argmin(np.where(cb == best[:, None], cc, INF), axis=1)
        Cn = np.take_along_axis(cc, k[:, None], 1)[:, 0]
        if not np.isfinite(Cn).any():
            break
        B = np.where(np.isfinite(Cn), np.take_along_axis(cb, k[:, None], 1)[:, 0], NEG)
        C, parent[i], last = Cn, np.where(np.isfinite(Cn), k, -1), i
    if last < Ns - 1:
        return None, last
    fin = np.isfinite(C)
    if j_end is not None:
        pick = np.zeros(fin.shape, bool)
        j = int(j_end)
        if 0 <= j < len(fin):
            pick[j] = True
        fin &= pick
    if not fin.any():
        return None, last
    tie = fin & (B == B[fin].max())
    j = int(np.argmin(np.where(tie, C, INF)))
    return _backtrack(parent, j, Ns), last


def _interp_j(f, ii, jf):
    """Interpolate a hole-filled field (holes = 0) in q7 at fractional index jf.

    A bracketing cell that is a hole takes the free side's value, so the result
    is the field where the lattice defines it and never a fabricated dip: with
    plain zero-filling, a point half a cell from a hole reads half the true
    sigma, which is an artefact of the 48-sample q7 grid, not a posture.
    Both sides holes -> 0, which is the honest "outside the band".
    """
    Nq = f.shape[1]
    lo = np.clip(np.floor(jf + 1e-9).astype(int), 0, Nq - 1)
    hi = np.clip(lo + 1, 0, Nq - 1)
    w = np.clip(jf - lo, 0.0, 1.0)
    a, b = f[ii, lo], f[ii, hi]
    return (1 - w) * np.where(a > 0, a, b) + w * np.where(b > 0, b, a)


class Corridor:
    """Feasibility test for one straight segment of the PWL path.

    A segment is sampled at the lattice's own s resolution (every integer i
    between its knots — s therefore needs no interpolation, only q7 does) and
    screened in two stages:

    1. GRID PROXY (cheap, vectorised).  The nearest grid cell must be free on
       this sheet, and the fields interpolated in q7 between the bracketing
       cells must clear the gates — where a bracketing cell that is a hole
       contributes the free side's value instead of a zero.  Zero-filling
       holes is the tempting alternative and it is wrong here: it demands half
       a cell of clearance from every hole, and these bands are 2-3 cells
       wide, so it rejects everything off the grid nodes and hands back the
       staircase the PWL exists to remove (measured: 36 knots instead of 2 on
       the rim arc).

    2. EXACT CHASE (only for segments the proxy accepts).  The grid is a
       SAMPLING of the redundancy space at 48 q7 values; a PWL segment
       commands q7 BETWEEN those samples, which no lattice node certifies.  So
       the segment is finally checked the same way it will be executed:
       `ik.solve_cc` chased along it from the sheet's own node at the first
       knot, requiring at every sample a case-consistent solution, continuity
       (||dq||_inf <= jump), and both gates on the freshly computed margin and
       sigma_min.  Search on the grid, certify against the real kinematics.

    `calls`/`samples` count the exact-stage work for reporting.
    """

    def __init__(self, lat, sheet, free, sigma_gate, margin_gate,
                 jump=planner.JUMP_THRESH, exact=True):
        self.free, self.sheet, self.q7s = free, sheet, lat["q7s"]
        self.sig = np.where(free, np.nan_to_num(sheet["sigma"]), 0.0)
        self.mar = np.where(free, np.nan_to_num(sheet["margin"]), 0.0)
        self.sigma_gate, self.margin_gate, self.jump = sigma_gate, margin_gate, jump
        self.exact, self.calls, self.samples = exact, 0, 0
        self.pen_ext, self.pts = lat["pen_ext"], lat["pts"]
        self.pen_lat = float(lat.get("pen_lat", 0.0))
        self.phi = float(lat.get("phi", 0.0))
        self._poses = pen_down_poses(lat["pts"], np.linalg.inv(lat["Twb"]),
                                     lat["pen_ext"], phi=self.phi,
                                     pen_lat=self.pen_lat)
        self._jidx = np.arange(free.shape[1])

    def _grid_ok(self, ii, jf):
        Nq = self.free.shape[1]
        jn = np.clip(np.rint(jf).astype(int), 0, Nq - 1)
        if not self.free[ii, jn].all():
            return False
        for f, gate in ((self.sig, self.sigma_gate), (self.mar, self.margin_gate)):
            if np.any(_interp_j(f, ii, jf) < gate):
                return False
        return True

    def _chase_ok(self, ii, jf, j_start):
        q = self.sheet["Q"][ii[0], int(round(j_start))]
        if not np.all(np.isfinite(q)):
            return False
        self.calls += 1
        self.samples += len(ii)
        return chase_cc(self._poses[ii], np.interp(jf, self._jidx, self.q7s), q,
                        pen_ext=self.pen_ext, pen_lat=self.pen_lat,
                        margin_gate=self.margin_gate,
                        sigma_gate=self.sigma_gate, jump_gate=self.jump)["ok"]

    def __call__(self, i0, j0, i1, j1):
        if i1 <= i0:
            return True
        ii = np.arange(i0, i1 + 1)
        jf = j0 + (j1 - j0) * (ii - i0) / (i1 - i0)
        if not self._grid_ok(ii, jf):
            return False
        return self._chase_ok(ii, jf, j0) if self.exact else True


def _rdp_corridor(js, eps, ok):
    """RDP on the dense path j(i), with a feasibility veto on every chord.

    Deviation is measured VERTICALLY, in q7 grid indices: the path is a
    function of s, and a perpendicular distance would mix arc length with
    radians.  A chord is accepted only if it is both within `eps` and
    corridor-feasible; otherwise the interval splits at the worst point (or at
    its midpoint when the chord is exact but infeasible, which happens when a
    hole sits beside a locally straight run).  Returns knot indices.

    Every adjacent pair of returned knots has therefore been through `ok`,
    except pairs one lattice step apart, which are edges of the dense DP path
    and were certified when it was built.
    """
    keep = [0, len(js) - 1]

    def rec(i0, i1):
        if i1 <= i0 + 1:
            return
        ii = np.arange(i0, i1 + 1)
        chord = js[i0] + (js[i1] - js[i0]) * (ii - i0) / (i1 - i0)
        dev = np.abs(js[ii] - chord)
        k = int(np.argmax(dev))
        if dev[k] <= eps and ok(i0, js[i0], i1, js[i1]):
            return
        cut = i0 + k if 0 < k < i1 - i0 else (i0 + i1) // 2
        keep.append(cut)
        rec(i0, cut)
        rec(cut, i1)

    rec(0, len(js) - 1)
    return np.array(sorted(set(keep)))


def _sample_pwl(knot_i, knot_j, Ns):
    """Dense j(i) implied by the PWL knots (float, one value per lattice s)."""
    return np.interp(np.arange(Ns), knot_i, knot_j)


def band_free(sheet, sigma_gate=SIGMA_GATE, margin_gate=MARGIN_GATE):
    """The gated free region of a sheet: where BOTH hard gates hold.

    Named because it is now the interface between the gates and every objective
    that runs on them — `plan_pwl`, the fiber menus in `menu.py`, and the tests
    that pin that a gate is never traded away for a shorter path.
    """
    sig, mar = sheet["sigma"], sheet["margin"]
    return (sheet["mask"] & (np.nan_to_num(sig) >= sigma_gate)
            & (np.nan_to_num(mar) >= margin_gate))


def plan_pwl(lat, sheet, sigma_gate=SIGMA_GATE, margin_gate=MARGIN_GATE,
             w_clearance=W_CLEARANCE, w_travel=W_TRAVEL, eps_idx=RDP_EPS,
             jump=planner.JUMP_THRESH, exact=True, objective=OBJECTIVE,
             j_start=None, j_end=None, travel_mode=TRAVEL_MODE):
    """Plan one stroke's redundancy as a piecewise-linear q7(s) on `sheet`.

    `sheet` is a dict from `sheet_fields` (e.g. `dominant_sheet(sheets)`).
    Steps: gate the sheet to a free region -> chamfer clearance map -> dense
    monotone DP -> RDP simplification with a per-segment corridor check
    (`Corridor`: grid proxy, then — with `exact=True` — a case-consistent IK
    chase along the candidate segment, so the knots that come out are certified
    against the real kinematics and not merely against the 48-sample q7 grid).

    OBJECTIVE.  "min_travel" (the default) minimises total joint travel over
    the gated free region, with clearance as an exact tie-break; the gates are
    hard and are not in the cost.  "maximin_sigma" is the older bottleneck
    objective, kept as the automatic fallback for strokes the shortest path
    cannot get certified (see `_dense_dp`).  Both walk the SAME free region, so
    switching between them can never trade a gate for anything.

    `j_start` / `j_end` pin the entry / exit q7 index for the fiber-menu path
    in `menu.py`; left None the DP chooses both ends itself, which is the
    behaviour every existing caller gets.

    Returns dict:
        ok          the PWL spans the whole stroke
        sheet       sheet id it was planned on
        objective   which DP produced it
        knots       (K, 2) [s in 0..1, q7 in rad] — THE PLAN
        knot_idx    (K, 2) the same knots as (i, j) lattice indices
        dense       (Ns, 2) the DP path in the same units, before simplification
        sigma, margin, clearance   (Ns,) fields sampled along the PWL
        dense_sigma, dense_margin  (Ns,) fields along the dense path
        bottleneck  min sigma along the dense path (lattice values)
        dp_travel   total lattice joint travel of the dense path (rad)
        n_dense, n_knots, cut_index, cut_s
        exact_calls, exact_samples   work done by the exact corridor stage
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}, want one of {OBJECTIVES}")
    Ns, Nq = sheet["mask"].shape
    q7s = lat["q7s"]
    sig, mar = sheet["sigma"], sheet["margin"]
    free = band_free(sheet, sigma_gate, margin_gate)
    clear = clearance_map(free)
    edges = _edge_ok(sheet, jump) & free[:-1][:, :, None]
    if objective == "min_travel":
        js, last, dp_cost = _dense_dp_travel(free, clear, edges,
                                             _edge_travel(sheet, travel_mode),
                                             w_clearance, j_start, j_end)
    else:
        js, last = _dense_dp(free, sig, clear, edges, w_clearance, w_travel,
                             j_start, j_end)
        dp_cost = np.inf
    # `dp_travel` is always the CHORD travel of the chosen lattice path, so the
    # two objectives are described in the same units whatever they optimise.
    dp_travel = (float(_edge_travel(sheet, "chord")[
        np.arange(Ns - 1), js[:-1], js[1:] - js[:-1] + 1].sum())
        if js is not None else np.inf)
    out = dict(sheet=int(sheet["id"]), ok=js is not None, free=free,
               objective=objective, dp_travel=float(dp_travel),
               dp_cost=float(dp_cost), travel_mode=travel_mode,
               clearance_map=clear, cut_index=int(last),
               cut_s=float(last / (Ns - 1)))
    if js is None:
        return out

    corr = Corridor(lat, sheet, free, sigma_gate, margin_gate, jump, exact=exact)
    ki = _rdp_corridor(js, eps_idx, corr)
    kj = js[ki].astype(float)
    jf = _sample_pwl(ki, kj, Ns)
    ii = np.arange(Ns)
    s = ii / (Ns - 1)

    def lerp(f):
        return _interp_j(f, ii, jf)

    out.update(knots=np.column_stack([ki / (Ns - 1), np.interp(kj, np.arange(Nq), q7s)]),
               knot_idx=np.column_stack([ki, kj]),
               dense=np.column_stack([s, q7s[js]]), dense_idx=js,
               pwl_idx=jf, sigma=lerp(corr.sig), margin=lerp(corr.mar),
               clearance=lerp(np.where(free, clear, 0.0)),
               dense_sigma=sig[ii, js], dense_margin=mar[ii, js],
               dense_clearance=clear[ii, js],
               bottleneck=float(np.min(sig[ii, js])),
               n_dense=int(Ns), n_knots=int(len(ki)),
               exact_calls=corr.calls, exact_samples=corr.samples)
    return out


# --------------------------------------------------------------------------
# 3. back out joint configurations from the PWL path
# --------------------------------------------------------------------------
def backout(stroke_pts, spec, knots, lat=None, ds=0.005, sheet=None, q_seed=None,
            h_inv=None, pen_ext=None, jump=planner.JUMP_THRESH):
    """PWL redundancy plan -> dense joint trajectory, by RE-SOLVING IK.

    q7(s) is read off the polyline by linear interpolation at every sample of
    the stroke resampled at `ds`; the configuration at each sample is then
    obtained from `ik.solve_cc` (case-consistent, i.e. same IK branch) seeded
    with the previous sample, starting from the sheet's node at s = 0.

    WHY NOT INTERPOLATE q.  This is the pitfall `writing.densify` was written
    to fix, and it is the reason backout re-solves instead of resampling the
    lattice DP's output.  Advancing one q7 index is a null-space self-motion:
    q1 and q3 counter-rotate while the tip barely moves.  Two such
    configurations are not collinear in joint space, so a straight line between
    them leaves the constraint manifold — in the ARIS demo, linear
    interpolation between planned steps threw the pen tip 7.7 mm off the paper,
    while re-solving the case-consistent IK put it back to 0.19 mm.  The PWL is
    a plan in (s, q7), NOT in q: only the redundancy parameter may be
    interpolated, never the configuration it indexes.

    If a case-consistent solve fails at a sample (the branch genuinely ends
    there, or the analytic solver refuses), it falls back to a full `ik.solve`
    at that (pose, q7) and takes the branch nearest the previous q; those are
    counted, because a nonzero count means the PWL grazed a sheet boundary.

    Returns dict: ok, qs (M,7), pts (M,2), s (M,), q7 (M,), sigmas, margins,
    tip_err (max, m), max_step (rad), fallbacks, fails, min_sigma, mean_sigma,
    min_margin, arc_len (m) and ds (m) — the last two are what turns the
    normalised plan back into something that can be timed (pacing.py).
    """
    setup = stroke_setup(stroke_pts, spec,
                         lambda t: np.interp(t, np.asarray(knots, float)[:, 0],
                                             np.asarray(knots, float)[:, 1]),
                         lat=lat, ds=ds, sheet=sheet, q_seed=q_seed,
                         h_inv=h_inv, pen_ext=pen_ext)
    if setup["q_seed"] is None:
        return dict(ok=False, fails=len(setup["pts"]), fallbacks=0, cut_index=-1)
    ch = chase_cc(setup["poses"], setup["q7"], setup["q_seed"],
                  pen_ext=setup["pen_ext"], pen_lat=setup["pen_lat"],
                  fallback=True)
    if not ch["n"]:
        return dict(ok=False, fails=ch["fails"], fallbacks=ch["fallbacks"],
                    cut_index=-1)
    return chase_report(ch, setup, jump=jump)


def stroke_setup(stroke_pts, spec, q7_of_s, lat=None, ds=0.005, sheet=None,
                 q_seed=None, h_inv=None, pen_ext=None, phi=None,
                 pen_lat=None):
    """Everything a chase needs before it can take its first step.

    Resamples the stroke at `ds` METRES, reads the redundancy plan off
    `q7_of_s` (anything callable on the normalised arc-length array — a raw
    polyline via np.interp, or smooth.py's corner-rounded curve), builds the
    base-frame poses, and picks the seed configuration at s = 0: the sheet's
    own node nearest the commanded q7 if a sheet was handed in, otherwise the
    most comfortable branch the full IK offers there.

    The seed is the one genuinely stateful choice in the whole pipeline — a
    case-consistent chase never changes branch, so whichever branch this picks
    is the branch the entire stroke is drawn on.

    Returns dict: pts, s, q7, poses, Twb, pen_ext, ds, arc_len, q_seed
    (q_seed None when no branch exists at s = 0).
    """
    from .fleet import H_INV_DEFAULT
    Twb = lat["Twb"] if lat is not None else spec.T_world_base(
        H_INV_DEFAULT if h_inv is None else h_inv)
    pen_ext = (lat["pen_ext"] if lat is not None else PEN_EXT) if pen_ext is None else pen_ext
    if phi is None:
        phi = float(lat.get("phi", 0.0)) if lat is not None else 0.0
    if pen_lat is None:
        pen_lat = float(lat.get("pen_lat", 0.0)) if lat is not None \
            else lat_of(None)
    pts, s = planner.resample(stroke_pts, ds)
    q7 = np.atleast_1d(np.asarray(q7_of_s(s), float))
    poses = pen_down_poses(pts, np.linalg.inv(Twb), pen_ext, phi=phi,
                           pen_lat=pen_lat)

    if q_seed is None and sheet is not None and lat is not None:
        j0 = int(np.argmin(np.abs(lat["q7s"] - q7[0])))
        row = np.flatnonzero(sheet["mask"][0])
        if len(row):
            j0 = int(row[np.argmin(np.abs(row - j0))])
            q_seed = sheet["Q"][0, j0]
    if q_seed is None:                      # no sheet handed in: best branch at s=0
        cand = ik.solve(poses[0], q7[0], spec.q_seed)
        q_seed = max(cand, key=joint_margin) if cand else None
    return dict(pts=pts, s=s, q7=q7, poses=poses, Twb=Twb, pen_ext=pen_ext,
                pen_lat=float(pen_lat), phi=float(phi),
                ds=ds, arc_len=arc_length(stroke_pts), q_seed=q_seed)


def chase_report(ch, setup, jump=planner.JUMP_THRESH):
    """Package a finished chase the way the rest of the project consumes it.

    Tip error is measured against the COMMANDED stroke points in world, which
    is the only check that catches a plan that quietly left the paper; the
    gates come straight off the chase, and dq/ds restores the metre scale so
    the result can be handed to pacing.py.  `backout` and `smooth.certify`
    share this so a smoothed path and the polyline it was rounded from are
    measured by literally the same code, never by two similar-looking ones.
    """
    qs, n = ch["qs"], ch["n"]
    pts, Twb, pen_ext = setup["pts"], setup["Twb"], setup["pen_ext"]
    tip = tip_pos_many(qs, pen_ext, setup.get("pen_lat", 0.0)) \
        @ Twb[:3, :3].T + Twb[:3, 3]
    err = np.hypot(np.linalg.norm(tip[:, :2] - pts[:n], axis=1), tip[:, 2])
    sig, mar = ch["sigmas"], ch["margins"]
    dq = np.abs(np.diff(qs, axis=0))
    dqds = dq_ds(qs, setup["arc_len"], ds_m=setup["ds"])
    return dict(ok=bool(n == len(pts)), qs=qs, pts=pts[:n], s=setup["s"][:n],
                q7=setup["q7"][:n], sigmas=sig, margins=mar, tip_errs=err,
                tip_err=float(err.max()),
                max_step=float(dq.max()) if len(dq) else 0.0,
                sum_travel=float(dq.sum()), fallbacks=ch["fallbacks"],
                fails=ch["fails"], arc_len=setup["arc_len"], ds=setup["ds"],
                dqds=dqds, max_dqds=float(np.abs(dqds).max()),
                max_dqds_jump=float(np.abs(np.diff(dqds, axis=0)).max())
                if len(dqds) > 1 else 0.0,
                min_sigma=float(sig.min()), mean_sigma=float(sig.mean()),
                min_margin=float(mar.min()), cut_index=n - 1,
                continuous=bool(len(dq) == 0 or dq.max() <= jump))
