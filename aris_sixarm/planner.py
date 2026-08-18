"""Single-arm stroke planner: global redundancy resolution on a ladder graph.

Given a curve r(s) on the paper allocated to one arm, we search the fiber
bundle over s: at each arc-length step the fiber is (q7 x IK-branch) — every
analytic-IK solution that passes the comfort gates. Edges only connect
consecutive steps, within a +-1 q7 index window and with configuration
continuity ||dq||_inf <= jump_thresh (which resolves branch identity without
labels). The graph is a DAG in s, so one DP sweep yields the globally optimal
joint path — the lookahead that diffIK lacks.

WHY NO YAW AXIS.  For a pen pointing straight down the tool orientation is
R = rotz(yaw) @ rotx(pi); joint 7's axis is then collinear with the pen axis,
so a yaw change of d is EXACTLY a q7 change of d: the solution at
(yaw + d, q7 + d) has identical q1..q6 and q7 shifted.  A (yaw x q7) grid
therefore aliases into diagonal bands (with the old 8 x 24 grid: valid q7
indices formed a comb of period 3 = 0.785/0.256) and a +-1-index DP window
cannot travel along that diagonal, producing artificial disconnects.  We fix
yaw = 0 (R = rotx(pi)) and let q7 carry the whole 1-D self-motion freedom.
Nothing wraps in this lattice.

Objectives
  "maximin_sigma" (default): maximize the SMALLEST sigma_min along the whole
      path (bottleneck / max-min DP, exact on the DAG), ties broken by total
      ||dq||^2.  Keeps the stroke as far from force-blind postures as the
      geometry allows, instead of averaging a deep dip away.
  "additive": weighted sum of smoothness + margin + sigma shortfalls.

If the band disconnects, the forward pass reports the farthest reachable
s* — the natural split point for multi-arm allocation.
"""
import numpy as np

from . import ik
from .frames import fk, rotx, PEN_EXT, joint_margin, FR3_MIN, FR3_MAX
from .metrics import tip_jacobian, sigma_min as _sigma_min

HARD_MARGIN = 0.15        # node gate, rad (permissive tier)
HARD_SIGMA = 0.08         # node gate, sigma_min of the pen-tip Jacobian
JUMP_THRESH = 0.35        # rad, max per-step joint motion (continuity)
W_SMOOTH = 4.0            # cost weight on ||dq||^2
W_MARGIN = 1.0            # cost weight pulling toward comfortable margins
W_SIGMA = 3.0             # cost weight pulling away from singularities
MARGIN_REF = 0.45
SIGMA_REF = 0.20
N_Q7 = 48                 # default q7 samples across the FR3 range
N_BRANCH = 4              # analytic IK returns at most 4 branches
_SIGMA_Q = 1e4            # bottleneck quantum (1e-4) for lexicographic DP


def resample(points, ds=0.01):
    """Polyline (N,2) -> arc-length resampled (M,2)."""
    p = np.asarray(points, float)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    t = np.concatenate([[0], np.cumsum(seg)])
    L = t[-1]
    s = np.arange(0, L + 1e-9, ds)
    return np.column_stack([np.interp(s, t, p[:, 0]), np.interp(s, t, p[:, 1])]), s / L


def build_lattice(pts_xy, spec, h_inv=None, pen_ext=PEN_EXT, n_q7=N_Q7,
                  clearance=True):
    """IK + gate the whole (s x q7 x branch) lattice for a vertical pen.

    Returns dict of arrays: Q (Ns,Nq,4,7), valid/margin/sigma (Ns,Nq,4).
    Nodes are kept only if margin >= HARD_MARGIN, sigma_min >= HARD_SIGMA and
    (optionally) the arm clears the paper and its own boom.
    """
    from .fleet import H_INV_DEFAULT
    Twb = spec.T_world_base(H_INV_DEFAULT if h_inv is None else h_inv)
    Twb_inv = np.linalg.inv(Twb)
    q7s = np.linspace(FR3_MIN[6] + 0.05, FR3_MAX[6] - 0.05, n_q7)
    R_w = rotx(np.pi)                     # pen straight down, yaw fixed (see docstring)
    Ns = len(pts_xy)
    Q = np.full((Ns, n_q7, N_BRANCH, 7), np.nan)
    valid = np.zeros((Ns, n_q7, N_BRANCH), bool)
    marg = np.full((Ns, n_q7, N_BRANCH), -1.0)
    sig = np.full((Ns, n_q7, N_BRANCH), -1.0)
    seed = spec.q_seed
    T_w = np.eye(4)
    T_w[:3, :3] = R_w
    for i, (x, y) in enumerate(pts_xy):
        T_w[:3, 3] = np.array([x, y, 0.0]) - pen_ext * R_w[:, 2]
        T_b = Twb_inv @ T_w
        for jq, q7 in enumerate(q7s):
            for kb, q in enumerate(ik.solve(T_b, q7, seed)):
                if kb >= N_BRANCH:
                    break
                m = joint_margin(q)
                if m < HARD_MARGIN:
                    continue
                if clearance:
                    _, p = fk(q)
                    pw = (Twb[:3, :3] @ p.T).T + Twb[:3, 3]
                    if np.any(pw[1:, 2] < 0.02):        # 2 cm above the paper
                        continue
                    if spec.mount == "inv":             # own boom cylinder
                        rb = np.hypot(p[:, 0], p[:, 1])
                        if np.any((p[:, 2] < -0.02) & (rb < 0.12)):
                            continue
                s = _sigma_min(tip_jacobian(q, pen_ext=pen_ext))
                if s < HARD_SIGMA:
                    continue
                Q[i, jq, kb] = q
                valid[i, jq, kb] = True
                marg[i, jq, kb] = m
                sig[i, jq, kb] = s
    return dict(Q=Q, valid=valid, margin=marg, sigma=sig, q7s=q7s, yaw=0.0,
                Twb=Twb, pts=np.asarray(pts_xy, float), pen_ext=pen_ext, spec=spec)


def _fill(lat):
    """NaN-free copy of Q so vectorised diffs never touch NaN (invalid nodes
    are masked out explicitly, so their values are irrelevant)."""
    return np.where(lat["valid"][..., None], lat["Q"], 0.0)


def plan(lat, objective="maximin_sigma", w_smooth=W_SMOOTH, w_margin=W_MARGIN,
         w_sigma=W_SIGMA, jump=JUMP_THRESH):
    """DP over the ladder graph.

    objective="maximin_sigma": lexicographic (max bottleneck sigma_min,
        then min sum ||dq||^2).  Exact on the DAG:
        V[n] = min(sigma[n], max_{p -> n} V[p]).
    objective="additive": min sum of w_smooth*||dq||^2 + margin/sigma shortfalls.

    Returns dict: ok, cut_index, cut_s and (if any path exists) qs, path,
    q7s, margins, sigmas, bottleneck.
    """
    if objective not in ("maximin_sigma", "additive"):
        raise ValueError(f"unknown objective {objective!r}")
    maximin = objective == "maximin_sigma"
    Q, V, M, S = lat["Q"], lat["valid"], lat["margin"], lat["sigma"]
    Ns, Nq, Nb, _ = Q.shape
    Qf = _fill(lat)
    NEG, INF = -np.inf, np.inf

    node_cost = np.where(V, w_margin * np.clip(MARGIN_REF - M, 0, None)
                         + w_sigma * np.clip(SIGMA_REF - S, 0, None), INF)
    Sq = np.where(V, np.round(S * _SIGMA_Q), NEG)   # quantised sigma (exact min)

    parent = np.full((Ns, Nq, Nb), -1, np.int32)    # encoded (dj+1)*Nb + kb_prev
    if maximin:
        B = Sq[0].copy()                            # bottleneck-to-come (quantised)
        C = np.where(V[0], 0.0, INF)                # tie-break smoothness cost
    else:
        B = None
        C = node_cost[0].copy()
    last_ok = 0 if np.isfinite(C).any() else -1
    if last_ok < 0:
        return dict(ok=False, cut_index=-1, cut_s=0.0, objective=objective,
                    reach_counts=np.zeros(Ns, int))
    reach_counts = np.zeros(Ns, int)
    reach_counts[0] = int(np.isfinite(C).sum())

    ncand = 3 * Nb
    for i in range(1, Ns):
        cb = np.full((Nq, Nb, ncand), NEG)          # candidate bottleneck
        cc = np.full((Nq, Nb, ncand), INF)          # candidate cost
        for t, dj in enumerate((-1, 0, 1)):         # q7 index step; no wrap
            lo, hi = max(0, dj), min(Nq, Nq + dj)
            if lo >= hi:
                continue
            tgt, src = slice(lo, hi), slice(lo - dj, hi - dj)
            d = Qf[i][tgt][:, :, None, :] - Qf[i - 1][src][:, None, :, :]
            dinf = np.max(np.abs(d), axis=-1)
            step = np.sum(d * d, axis=-1)
            feas = (V[i][tgt][:, :, None] & V[i - 1][src][:, None, :]
                    & (dinf <= jump))
            sl = slice(t * Nb, (t + 1) * Nb)
            if maximin:
                feas &= (B[src] > NEG)[:, None, :]
                cb[tgt, :, sl] = np.where(
                    feas, np.minimum(B[src][:, None, :], Sq[i][tgt][:, :, None]), NEG)
                cc[tgt, :, sl] = np.where(
                    feas, C[src][:, None, :] + w_smooth * step, INF)
            else:
                cc[tgt, :, sl] = np.where(
                    feas, C[src][:, None, :] + w_smooth * step
                    + node_cost[i][tgt][:, :, None], INF)
        if maximin:
            best_b = cb.max(axis=-1)                                  # lexicographic:
            k = np.argmin(np.where(cb == best_b[..., None], cc, INF), -1)  # then cost
        else:
            k = np.argmin(cc, axis=-1)
        Cn = np.take_along_axis(cc, k[..., None], -1)[..., 0]
        if not np.isfinite(Cn).any():
            break            # band disconnected: keep the last finite C/B/parent
        if maximin:
            Bn = np.take_along_axis(cb, k[..., None], -1)[..., 0]
            B = np.where(np.isfinite(Cn), Bn, NEG)
        C = Cn
        parent[i] = np.where(np.isfinite(Cn), k, -1)
        last_ok = i
        reach_counts[i] = int(np.isfinite(Cn).sum())

    ok = last_ok == Ns - 1
    out = dict(ok=ok, cut_index=last_ok, cut_s=last_ok / (Ns - 1),
               objective=objective, reach_counts=reach_counts)
    if last_ok < 1:
        return out
    if maximin:                                   # best terminal node, lexicographic
        tie = B == B.max()
        idx = np.unravel_index(np.argmin(np.where(tie, C, INF)), C.shape)
        out["bottleneck"] = float(B[idx]) / _SIGMA_Q
    else:
        idx = np.unravel_index(np.argmin(C), C.shape)
    assert np.isfinite(C[idx]), "backtrack from an infeasible terminal node"
    path = [(int(idx[0]), int(idx[1]))]
    for i in range(last_ok, 0, -1):
        k = int(parent[i][path[-1]])
        assert k >= 0, f"missing parent at step {i}"
        dj, kb = k // Nb - 1, k % Nb
        path.append((path[-1][0] - dj, kb))
    path = path[::-1]
    qs = np.array([Q[i][p] for i, p in enumerate(path)])
    assert np.all(np.isfinite(qs)), "planned path visits an invalid node"
    out.update(qs=qs, path=path, cost=float(C[idx]),
               q7s=np.array([lat["q7s"][p[0]] for p in path]),
               margins=np.array([M[i][p] for i, p in enumerate(path)]),
               sigmas=np.array([S[i][p] for i, p in enumerate(path)]))
    return out


def greedy(lat, jump=JUMP_THRESH):
    """diffIK-style baseline: at each step take the valid node nearest the
    previous q (no lookahead). Returns the same fields as plan()."""
    Q, V, M, S = lat["Q"], lat["valid"], lat["margin"], lat["sigma"]
    Ns = Q.shape[0]
    Qf = _fill(lat)
    if not V[0].any():
        return dict(ok=False, cut_index=-1, cut_s=0.0)

    def _pack(path, ok, cut):
        return dict(ok=ok, cut_index=cut, cut_s=cut / (Ns - 1),
                    qs=np.array([Q[i][p] for i, p in enumerate(path)]), path=path,
                    q7s=np.array([lat["q7s"][p[0]] for p in path]),
                    margins=np.array([M[i][p] for i, p in enumerate(path)]),
                    sigmas=np.array([S[i][p] for i, p in enumerate(path)]))

    idx = np.unravel_index(np.argmax(np.where(V[0], M[0], -1.0)), V[0].shape)
    path = [(int(idx[0]), int(idx[1]))]
    for i in range(1, Ns):
        d = np.max(np.abs(Qf[i] - Q[i - 1][path[-1]]), axis=-1)
        d = np.where(V[i], d, np.inf)
        j = np.unravel_index(np.argmin(d), d.shape)
        if not np.isfinite(d[j]) or d[j] > jump:
            return _pack(path, False, i - 1)
        path.append((int(j[0]), int(j[1])))
    return _pack(path, True, Ns - 1)


def path_report(lat, res, pen_ext=None):
    """Summary along a planned path (sigma comes from the lattice, no re-FD)."""
    if "qs" not in res:
        return dict(feasible=False, cut_s=res.get("cut_s", 0.0))
    if "sigmas" in res:
        sig = np.asarray(res["sigmas"], float)
    else:                                        # fallback: recompute
        pe = lat["pen_ext"] if pen_ext is None else pen_ext
        sig = np.array([_sigma_min(tip_jacobian(q, pen_ext=pe)) for q in res["qs"]])
    dq = np.abs(np.diff(res["qs"], axis=0))
    return dict(feasible=bool(res["ok"]), cut_s=res["cut_s"],
                cut_index=res.get("cut_index", len(res["qs"]) - 1),
                min_margin=float(np.min(res["margins"])),
                min_sigma=float(sig.min()), mean_sigma=float(sig.mean()), sigma=sig,
                max_step=float(dq.max()) if len(dq) else 0.0,
                sum_travel=float(dq.sum()),
                q7=np.asarray(res.get("q7s", []), float))
