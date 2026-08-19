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
from .frames import PEN_EXT, joint_margin, rotx, tip_pos
from .metrics import sigma_min as _sigma_min, tip_jacobian
from .pacing import dq_ds

SIGMA_GATE = 0.10        # band gate, sigma_min (above planner.HARD_SIGMA = 0.08)
MARGIN_GATE = 0.15       # band gate, joint margin (rad)
W_CLEARANCE = 1.0        # tie-break pull toward the middle of the corridor
W_TRAVEL = 0.25          # tie-break pull toward fewer q7 index steps
RDP_EPS = 1.0            # simplification tolerance, q7 GRID INDICES
_SIGMA_Q = planner._SIGMA_Q      # bottleneck quantum, shared with the lattice DP


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


def pen_down_poses(pts_xy, Twb_inv, pen_ext):
    """(M,4,4) base-frame hand-TCP poses for a pen-down stroke.

    The pen convention lives here and nowhere else: R = rotx(pi) points tool z
    straight down with yaw 0 (planner.py explains why yaw is not a free axis),
    and the tip sits `pen_ext` beyond the TCP along that z — so the TCP is
    `pen_ext` ABOVE the paper point it is drawing.
    """
    p = np.asarray(pts_xy, float)
    R = rotx(np.pi)
    T = np.tile(np.eye(4), (len(p), 1, 1))
    T[:, :3, :3] = R
    T[:, :3, 3] = np.column_stack([p[:, 0], p[:, 1], np.zeros(len(p))]) \
        - pen_ext * R[:, 2]
    return np.asarray(Twb_inv, float) @ T


def chase_cc(poses, q7, q_seed, pen_ext=PEN_EXT, margin_gate=None,
             sigma_gate=None, jump_gate=None, fallback=False):
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
        s = _sigma_min(tip_jacobian(q, pen_ext=pen_ext))
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


def _dense_dp(free, sigma, clear, edge_ok, w_clearance, w_travel):
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
    """
    Ns, Nq = free.shape
    NEG, INF = -np.inf, np.inf
    Sq = np.where(free, np.round(np.nan_to_num(sigma) * _SIGMA_Q), NEG)
    node_c = np.where(free, -w_clearance * clear, INF)
    parent = np.full((Ns, Nq), -1, np.int8)
    B, C = Sq[0].copy(), np.where(free[0], node_c[0], INF)
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
    tie = B == B.max()
    j = int(np.argmin(np.where(tie, C, INF)))
    js = [j]
    for i in range(Ns - 1, 0, -1):
        k = int(parent[i][js[-1]])
        assert k >= 0, f"missing parent at step {i}"
        js.append(js[-1] - (k - 1))
    return np.array(js[::-1]), last


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
        self._poses = pen_down_poses(lat["pts"], np.linalg.inv(lat["Twb"]),
                                     lat["pen_ext"])
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
                        pen_ext=self.pen_ext, margin_gate=self.margin_gate,
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


def plan_pwl(lat, sheet, sigma_gate=SIGMA_GATE, margin_gate=MARGIN_GATE,
             w_clearance=W_CLEARANCE, w_travel=W_TRAVEL, eps_idx=RDP_EPS,
             jump=planner.JUMP_THRESH, exact=True):
    """Plan one stroke's redundancy as a piecewise-linear q7(s) on `sheet`.

    `sheet` is a dict from `sheet_fields` (e.g. `dominant_sheet(sheets)`).
    Steps: gate the sheet to a free region -> chamfer clearance map -> dense
    monotone DP (maximin sigma, clearance tie-break) -> RDP simplification with
    a per-segment corridor check (`Corridor`: grid proxy, then — with
    `exact=True` — a case-consistent IK chase along the candidate segment, so
    the knots that come out are certified against the real kinematics and not
    merely against the 48-sample q7 grid).

    Returns dict:
        ok          the PWL spans the whole stroke
        sheet       sheet id it was planned on
        knots       (K, 2) [s in 0..1, q7 in rad] — THE PLAN
        knot_idx    (K, 2) the same knots as (i, j) lattice indices
        dense       (Ns, 2) the DP path in the same units, before simplification
        sigma, margin, clearance   (Ns,) fields sampled along the PWL
        dense_sigma, dense_margin  (Ns,) fields along the dense path
        bottleneck  min sigma along the dense path (lattice values)
        n_dense, n_knots, cut_index, cut_s
        exact_calls, exact_samples   work done by the exact corridor stage
    """
    Ns, Nq = sheet["mask"].shape
    q7s = lat["q7s"]
    sig, mar = sheet["sigma"], sheet["margin"]
    free = (sheet["mask"] & (np.nan_to_num(sig) >= sigma_gate)
            & (np.nan_to_num(mar) >= margin_gate))
    clear = clearance_map(free)
    edges = _edge_ok(sheet, jump) & free[:-1][:, :, None]
    js, last = _dense_dp(free, sig, clear, edges, w_clearance, w_travel)
    out = dict(sheet=int(sheet["id"]), ok=js is not None, free=free,
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
                  pen_ext=setup["pen_ext"], fallback=True)
    if not ch["n"]:
        return dict(ok=False, fails=ch["fails"], fallbacks=ch["fallbacks"],
                    cut_index=-1)
    return chase_report(ch, setup, jump=jump)


def stroke_setup(stroke_pts, spec, q7_of_s, lat=None, ds=0.005, sheet=None,
                 q_seed=None, h_inv=None, pen_ext=None):
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
    pts, s = planner.resample(stroke_pts, ds)
    q7 = np.atleast_1d(np.asarray(q7_of_s(s), float))
    poses = pen_down_poses(pts, np.linalg.inv(Twb), pen_ext)

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
    tip = np.array([Twb[:3, :3] @ tip_pos(q, pen_ext) + Twb[:3, 3] for q in qs])
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
