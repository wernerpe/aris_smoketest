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
from . import rig_final
from .frames import (fk, rotx, rotz, rot_axis, PEN_EXT, joint_margin, FR3_MIN,
                     FR3_MAX, lat_of, tool_offset)
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


def clip_to_sheet(pts, border=0.02, verbose=True, return_slice=False,
                  sheet=None):
    """Keep the LONGEST CONTIGUOUS in-sheet run of a polyline.

    A plain boolean mask would concatenate the disjoint in-sheet pieces of a
    curve into one polyline, silently inserting a straight chord across the
    off-sheet gap. For a rim arc around an inverted arm that phantom chord
    dives from the r = 0.66 rim to r = 0.30 straight under the base — a
    genuinely unplannable dead zone that has nothing to do with the stroke
    being asked for.

    `return_slice` additionally hands back the `slice` that was kept, so a
    caller can say WHERE in the original stroke its plan starts and ends
    (stroke_api reports it as `clip_s`).
    """
    from .fleet import SHEET
    sheet = SHEET if sheet is None else sheet
    pts = np.asarray(pts, float)
    m = ((pts[:, 0] > border) & (pts[:, 0] < sheet[0] - border)
         & (pts[:, 1] > border) & (pts[:, 1] < sheet[1] - border))
    runs, i = [], 0
    while i < len(m):
        if m[i]:
            j = i
            while j < len(m) and m[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    if not runs:
        return (pts[:0], slice(0, 0)) if return_slice else pts[:0]
    i0, i1 = max(runs, key=lambda r: r[1] - r[0])
    if len(runs) > 1 and verbose:
        print(f"  clip: {len(runs)} in-sheet runs {[b - a for a, b in runs]}, "
              f"keeping the longest ({i1 - i0} pts)")
    return (pts[i0:i1], slice(i0, i1)) if return_slice else pts[i0:i1]


# THE PEN'S LEAN, AS A TOOL-FRAME ROTATION (2026-08-26)
# ------------------------------------------------------
# `tilt` is a 2-vector in the TOOL's own xy plane: the rotation axis is
# (tx, ty, 0) in tool coordinates and the lean is its magnitude in radians, so
# `R_tilted = R @ rot_axis(...)`.  POST-multiplication, deliberately, and not
# the world-frame pre-multiplication `tilt.py` uses: with the inline pen the
# two span the same set of pen directions because tool yaw is a q7 shift and
# free, and with the LATERAL holder they do not — a world-frame lean would
# swing the 11 cm bracket somewhere the atlas never looked.  `atlas._candidates`
# leans the same way (`rot_axis(R @ ax, a) @ R` is exactly `R @ rot_axis(ax,
# a)`), so a lean the atlas certified is a lean this lattice can reproduce.
#
# The pen axis is the tool's z, and a rotation about an axis in the xy plane
# moves it by exactly |tilt| — so |tilt| IS the lean from vertical that
# `validate._pen_lean_deg` will measure back off the FK.
def tool_lean(R, tilt=None):
    """Tool-frame lean of a tool rotation. `tilt` = (tx, ty) rad -> (3,3)."""
    if tilt is None:
        return R
    t = np.asarray(tilt, float).reshape(2)
    a = float(np.hypot(t[0], t[1]))
    if a < 1e-12:
        return R
    return R @ rot_axis(np.array([t[0] / a, t[1] / a, 0.0]), a)


def lean_deg(tilt=None):
    """The lean `tilt` commands, in degrees."""
    if tilt is None:
        return 0.0
    t = np.asarray(tilt, float).reshape(2)
    return float(np.rad2deg(np.hypot(t[0], t[1])))


Z_PAPER = 0.02            # m, chain points must stay this far above the paper
BOOM_R = 0.12             # m, inverted-mount boom cylinder radius (base frame)
BOOM_Z = -0.02            # m, below which the boom cylinder is an obstacle


def _lattice_setup(spec, h_inv, n_q7):
    from .fleet import H_INV_DEFAULT
    Twb = spec.T_world_base(H_INV_DEFAULT if h_inv is None else h_inv)
    return (Twb, np.linalg.inv(Twb),
            np.linspace(FR3_MIN[6] + 0.05, FR3_MAX[6] - 0.05, n_q7))


def build_lattice(pts_xy, spec, h_inv=None, pen_ext=PEN_EXT, n_q7=N_Q7,
                  clearance=True, pen_lat=None, phi=0.0, tilt=None):
    """IK + gate the whole (s x q7 x branch) lattice for a vertical pen.

    Returns dict of arrays: Q (Ns,Nq,4,7), valid/margin/sigma (Ns,Nq,4).
    Nodes are kept only if margin >= HARD_MARGIN, sigma_min >= HARD_SIGMA and
    (optionally) the arm clears the paper and its own boom.

    THE LATERAL TOOL AND THE phi AXIS.  `pen_lat` (None -> the ACTIVE tool,
    see frames.PEN_LAT) is the pen's lateral offset along hand x:
    tip = TCP + R @ (pen_lat, 0, pen_ext).  With it nonzero the WHY-NO-YAW
    argument below breaks — rotating the tool about the pen's vertical axis
    swings the TCP on a `pen_lat` circle around the tip, so the tool yaw
    `phi` becomes a REAL second redundancy DOF.  This function builds the
    FIXED-phi slice (R = rotz(phi) @ rotx(pi)); `lateral.py` owns the coarse
    phi search and the coupled (s x phi x q7 x branch) rescue lattice, the
    same adaptive shape as the tilt work.  With pen_lat == 0, phi is exactly
    the q7 degeneracy and must stay 0.

    This is THE hot loop of the project — a 1.5 m stroke is ~7500 IK calls, and
    for years every one of them crossed the pybind boundary on its own, checked
    its own limits in python, verified its own FK in python, and paid 14 more
    forward kinematics for a finite-difference Jacobian.  The C++ solver is
    2-5 us; the scaffolding around it was fifty.  `_build_lattice_batch` does
    the identical arithmetic in five array operations instead, and
    `_build_lattice_scalar` below is kept verbatim for extensions that predate
    the batch entry points (the station's cp310 wheel) — and as the reference
    the equality test measures the fast path against.
    """
    impl = _build_lattice_batch if ik.has_batch() else _build_lattice_scalar
    return impl(pts_xy, spec, h_inv, pen_ext, n_q7, clearance,
                lat_of(pen_lat), float(phi), tilt)


def _build_lattice_batch(pts_xy, spec, h_inv, pen_ext, n_q7, clearance,
                         pen_lat=0.0, phi=0.0, tilt=None):
    """Vectorised `build_lattice`.  Numerically identical to the scalar path.

    The gates are ANDs over independent per-node quantities, so applying them
    in stages to a shrinking index set gives exactly the mask the nested loop
    gives — and each stage only pays for the nodes that survived the last one,
    which is why the clearance FK and the Jacobians are cheap here.
    """
    Twb, Twb_inv, q7s = _lattice_setup(spec, h_inv, n_q7)
    pts = np.asarray(pts_xy, float)
    Ns = len(pts)
    off = tool_offset(pen_ext, pen_lat)   # tip = TCP + R @ off
    R_w = tool_lean(rotz(phi) @ rotx(np.pi) if phi else rotx(np.pi), tilt)

    # (a) every (step, q7) target pose, in one (Ns*Nq, 16) array.  Row order is
    #     i*Nq + j so a reshape recovers the lattice axes.
    T_w = np.tile(np.eye(4), (Ns, 1, 1))
    T_w[:, :3, :3] = R_w
    T_w[:, :3, 3] = np.column_stack([pts[:, 0], pts[:, 1], np.zeros(Ns)]) \
        - R_w @ off
    T_b = Twb_inv @ T_w
    flat = np.repeat(ik._flat16(T_b), n_q7, axis=0)

    # (b) one solve, FK-verified in C++ and re-filtered against FR3 limits
    Q, valid = ik.solve_batch(flat, np.tile(q7s, Ns), spec.q_seed)
    Q = Q.reshape(Ns, n_q7, N_BRANCH, 7)
    valid = valid.reshape(Ns, n_q7, N_BRANCH)

    shape = (Ns, n_q7, N_BRANCH)
    idx = np.flatnonzero(valid.reshape(-1))
    q = Q.reshape(-1, 7)[idx]

    # (c) joint-limit comfort margin
    m = np.min(np.minimum(q - FR3_MIN, FR3_MAX - q), axis=1)
    keep = m >= HARD_MARGIN
    idx, q, m = idx[keep], q[keep], m[keep]

    # (d) clearance: the paper, the legacy arms' own boom cylinder, and the
    #     FINAL RIG's frame structure (rig_final boxes, exact capsule dist)
    if clearance and len(idx):
        T, p = ik.fk_batch(q)                      # (K,9,3) chain points
        pw = p @ Twb[:3, :3].T + Twb[:3, 3]
        keep = pw[:, 1:, 2].min(axis=1) >= Z_PAPER
        if spec.mount == "inv" and getattr(spec, "rig", "sixarm") == "sixarm":
            rb = np.hypot(p[:, :, 0], p[:, :, 1])
            keep &= ~np.any((p[:, :, 2] < BOOM_Z) & (rb < BOOM_R), axis=1)
        boxes = spec.static_obstacles() if hasattr(spec, "static_obstacles") \
            else []
        if boxes:
            from .frames import tool_points_many
            tool_b = tool_points_many(T, pen_ext, pen_lat)   # [tip(,corner)]
            tool_w = [t @ Twb[:3, :3].T + Twb[:3, 3] for t in tool_b]
            P10 = np.concatenate([pw] + [t[:, None, :] for t in tool_w], axis=1)
            # STRICTLY TIGHTER THAN THE CHECKER, see rig_final.STATIC_PLAN_MARGIN
            keep &= (rig_final.chain_static_clearance(P10, boxes)
                     >= rig_final.STATIC_PLAN_MARGIN)
        # ...AND THE ARM AGAINST ITSELF (2026-08-26).  Nothing in this package
        # checked that until the day the pen was allowed to lean; see
        # `aris_sixarm/selfcoll.py`.  It costs the certified maps nothing (no
        # pose of the shipped atlas fails it, the tightest holds 63.7 mm) and
        # it is the only gate here that a leaning lattice node can newly need.
        from . import selfcoll
        keep &= selfcoll.self_ok(q, margin=selfcoll.SELF_PLAN_MARGIN,
                                 pen_ext=pen_ext, pen_lat=pen_lat)
        idx, q, m = idx[keep], q[keep], m[keep]

    # (e) controllability: analytic tip Jacobians, one batched SVD
    if len(idx):
        s = np.linalg.svd(ik.tip_jacobian_batch(q, pen_ext=pen_ext,
                                                pen_lat=pen_lat),
                          compute_uv=False)[:, -1]
        keep = s >= HARD_SIGMA
        idx, q, m, s = idx[keep], q[keep], m[keep], s[keep]
    else:
        s = np.zeros(0)

    Qo = np.full((Ns * n_q7 * N_BRANCH, 7), np.nan)
    vo = np.zeros(Ns * n_q7 * N_BRANCH, bool)
    mo = np.full(Ns * n_q7 * N_BRANCH, -1.0)
    so = np.full(Ns * n_q7 * N_BRANCH, -1.0)
    Qo[idx], vo[idx], mo[idx], so[idx] = q, True, m, s
    return dict(Q=Qo.reshape(*shape, 7), valid=vo.reshape(shape),
                margin=mo.reshape(shape), sigma=so.reshape(shape), q7s=q7s,
                yaw=float(phi), phi=float(phi), pen_lat=float(pen_lat),
                tilt=None if tilt is None else tuple(map(float, tilt)),
                Twb=Twb, pts=pts, pen_ext=pen_ext, spec=spec)


def _build_lattice_scalar(pts_xy, spec, h_inv, pen_ext, n_q7, clearance,
                          pen_lat=0.0, phi=0.0, tilt=None):
    """The original nested-loop lattice: one pybind crossing and one
    finite-difference Jacobian per node.  Still the fallback wherever the
    extension has no batch entry points, and the reference the batched path is
    tested against (tests/test_planner_robustness.py)."""
    Twb, Twb_inv, q7s = _lattice_setup(spec, h_inv, n_q7)
    Ns = len(pts_xy)
    off = tool_offset(pen_ext, pen_lat)
    R_w = tool_lean(rotz(phi) @ rotx(np.pi) if phi else rotx(np.pi), tilt)
    Q = np.full((Ns, n_q7, N_BRANCH, 7), np.nan)
    valid = np.zeros((Ns, n_q7, N_BRANCH), bool)
    marg = np.full((Ns, n_q7, N_BRANCH), -1.0)
    sig = np.full((Ns, n_q7, N_BRANCH), -1.0)
    seed = spec.q_seed
    boxes = spec.static_obstacles() if hasattr(spec, "static_obstacles") \
        else []
    T_w = np.eye(4)
    T_w[:3, :3] = R_w
    for i, (x, y) in enumerate(pts_xy):
        T_w[:3, 3] = np.array([x, y, 0.0]) - R_w @ off
        T_b = Twb_inv @ T_w
        for jq, q7 in enumerate(q7s):
            for kb, q in enumerate(ik.solve(T_b, q7, seed)):
                if kb >= N_BRANCH:
                    break
                m = joint_margin(q)
                if m < HARD_MARGIN:
                    continue
                if clearance:
                    T, p = fk(q)
                    pw = (Twb[:3, :3] @ p.T).T + Twb[:3, 3]
                    if np.any(pw[1:, 2] < Z_PAPER):     # 2 cm above the paper
                        continue
                    if spec.mount == "inv" and \
                            getattr(spec, "rig", "sixarm") == "sixarm":
                        rb = np.hypot(p[:, 0], p[:, 1])  # own boom cylinder
                        if np.any((p[:, 2] < BOOM_Z) & (rb < BOOM_R)):
                            continue
                    if boxes:
                        from .frames import tool_points_many
                        tool_b = tool_points_many(T[None], pen_ext, pen_lat)
                        tool_w = [Twb[:3, :3] @ t[0] + Twb[:3, 3]
                                  for t in tool_b]
                        P10 = np.vstack([pw] + [t[None] for t in tool_w])
                        # ...and the same floor here as in the batched gate
                        if (rig_final.chain_static_clearance(P10, boxes)[0]
                                < rig_final.STATIC_PLAN_MARGIN):
                            continue
                    from . import selfcoll          # the arm against itself
                    if not selfcoll.self_ok(
                            q[None], margin=selfcoll.SELF_PLAN_MARGIN,
                            pen_ext=pen_ext, pen_lat=pen_lat)[0]:
                        continue
                s = _sigma_min(tip_jacobian(q, pen_ext=pen_ext,
                                            pen_lat=pen_lat))
                if s < HARD_SIGMA:
                    continue
                Q[i, jq, kb] = q
                valid[i, jq, kb] = True
                marg[i, jq, kb] = m
                sig[i, jq, kb] = s
    return dict(Q=Q, valid=valid, margin=marg, sigma=sig, q7s=q7s,
                yaw=float(phi), phi=float(phi), pen_lat=float(pen_lat),
                tilt=None if tilt is None else tuple(map(float, tilt)),
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
