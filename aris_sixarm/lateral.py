"""The LATERAL pen holder as a planning axis: tool yaw phi is REAL redundancy.

THE GEOMETRY.  The real holder offsets the pen 11 cm from the wrist axis along
hand x: tip = TCP + R @ (PEN_LAT, 0, PEN_EXT) (frames.py).  With the inline
pen, rotating the tool about the vertical pen axis is EXACTLY a q7 shift
(planner.py, "WHY NO YAW AXIS"), so yaw was pinned to 0 and q7 carried the
whole 1-D self-motion.  With the lateral offset that degeneracy is BROKEN:
rotating the tool by phi about the pen's vertical axis swings the TCP on an
11 cm circle around the tip — same tip, genuinely different arm.  The fiber
over a stroke point is therefore 2-D, (phi x q7), plus the discrete IK branch,
and the full lattice is (s x phi x q7 x branch).

THE SHAPE OF THE SEARCH (the tilt work's shape, deliberately).  Planning the
full coupled lattice for every stroke is 8x the IK and ~10x the DP of the flat
pipeline, and almost no stroke needs it: phi is a slowly-varying preference
(point the wrist AWAY from the work), not something a pen stroke has to spin
through.  So:

  1. FIXED-phi FIRST.  A coarse ring of N_PHI = 8 tool yaws is screened with a
     3-point mini-lattice (a few ms), ordered by how much of the stroke's
     fiber each phi opens (ties broken toward the reach heuristic below), and
     each surviving candidate is planned by the UNCHANGED certified pipeline —
     `stroke_api.plan_stroke` with phi pinned, which is the same lattice ->
     sheets -> PWL -> smooth -> validate machinery every shipped stroke went
     through.  First certified phi wins; typical cost is one or two flat-plan
     equivalents (tens of ms).
  2. COUPLED RESCUE.  Only when NO fixed phi certifies end-to-end and the ring
     disagrees about where the stroke dies, the (s x phi x q7 x branch) DP is
     run: maximin-sigma bottleneck objective, +-1 windows in phi (WRAPPING —
     phi is a circle) and q7, ||dq||_inf <= JUMP_THRESH continuity.  Its path
     is certified by the standard dense chase (per-sample poses at the
     interpolated phi) and the independent validator; a rescue that does not
     beat the best fixed-phi split is discarded, so the phi axis can add
     certified strokes and never remove one.

THE HEURISTIC, AND WHY IT POINTS AWAY.  TCP = tip - PEN_LAT * (cos phi,
sin phi, 0) + (0, 0, PEN_EXT): choosing phi along the base->stroke direction
pulls the WRIST toward the base by a full pen_lat, so strokes at the edge of
reach become reachable — the wrist stands off from the work.  That is the
reach-extending orientation the screen tries first; well inside reach other
phis may score better and the screen is what decides.

Pen TILT is NOT combined with the lateral holder: the tilt disc machinery
leans on the inline pen's yaw degeneracy.  `tilt_max_deg` > 0 is noted and
ignored here.
"""
import time

import numpy as np

from . import pacing, planner, pwl, stroke_api
from .frames import lat_of, rotx, rotz, tool_offset
from .validate import validate_plan

N_PHI = 8                 # coarse tool-yaw ring (the task's 8-12 band)
SCREEN_PTS = 3            # mini-lattice probes per phi (ends + middle)
RESCUE_N_Q7 = 24          # q7 samples of the coupled rescue lattice: half the
#   flat 48 to keep (s x 8 x 24 x 4) affordable.  The q7 step is then 0.24 rad
#   < JUMP_THRESH, so +-1-index edges stay traversable.


def phi_ring(pts_xy, spec, n_phi=N_PHI):
    """The coarse phi candidates, anchored on the reach heuristic. -> (n,)"""
    p = np.asarray(pts_xy, float)
    c = p.mean(axis=0)
    bx, by = spec.xy
    phi0 = float(np.arctan2(c[1] - by, c[0] - bx))
    step = 2 * np.pi / n_phi
    offs = [0.0]
    for k in range(1, n_phi // 2 + 1):
        offs.append(k * step)
        if len(offs) < n_phi:
            offs.append(-k * step)
    return np.array([phi0 + o for o in offs[:n_phi]])


def screen_phis(pts_xy, spec, phis, o):
    """Order the ring by how much fiber each phi opens on a 3-point probe.

    A tiny lattice (ends + middle, full n_q7) per phi: score = number of probe
    points with a non-empty fiber (endpoints weighted double — a stroke that
    cannot start is dead however healthy its middle), tie-break = the ring's
    own heuristic order.  -> (ordered phis, scores)
    """
    p = np.asarray(pts_xy, float)
    probes = p[np.unique(np.linspace(0, len(p) - 1, SCREEN_PTS).astype(int))]
    scores = []
    for phi in phis:
        lat = planner.build_lattice(probes, spec, h_inv=o["h_inv"],
                                    pen_ext=o["pen_ext"], n_q7=o["n_q7"],
                                    pen_lat=o["pen_lat"], phi=float(phi))
        fib = lat["valid"].any(axis=(1, 2))
        scores.append(int(fib[0]) * 2 + int(fib[-1]) * 2 + int(fib[1:-1].sum()))
    order = np.argsort(-np.asarray(scores), kind="stable")
    return phis[order], [scores[i] for i in order]


# --------------------------------------------------------------------------
# the coupled (s x phi x q7 x branch) lattice and its DP
# --------------------------------------------------------------------------
def build_lattice_phi(pts_xy, spec, phis, h_inv=None, pen_ext=None,
                      pen_lat=None, n_q7=RESCUE_N_Q7):
    """The full coupled lattice: one fixed-phi lattice per ring point, stacked.

    -> dict with Q (Ns,Np,Nq,Nb,7), valid/margin/sigma (Ns,Np,Nq,Nb), phis,
    q7s, Twb, pts, pen_ext, pen_lat, spec.  Each slice is exactly
    `planner.build_lattice` at that phi, so every node passed the same gates
    (margin, sigma, paper, boom, frame boxes) as a flat lattice node.
    """
    o_ext = stroke_api.DEFAULTS["pen_ext"] if pen_ext is None else pen_ext
    lats = [planner.build_lattice(pts_xy, spec, h_inv=h_inv, pen_ext=o_ext,
                                  n_q7=n_q7, pen_lat=pen_lat, phi=float(ph))
            for ph in phis]
    return dict(Q=np.stack([l["Q"] for l in lats], axis=1),
                valid=np.stack([l["valid"] for l in lats], axis=1),
                margin=np.stack([l["margin"] for l in lats], axis=1),
                sigma=np.stack([l["sigma"] for l in lats], axis=1),
                phis=np.asarray(phis, float), q7s=lats[0]["q7s"],
                Twb=lats[0]["Twb"], pts=lats[0]["pts"],
                pen_ext=lats[0]["pen_ext"], pen_lat=lats[0]["pen_lat"],
                spec=spec)


def plan_phi(lat, jump=planner.JUMP_THRESH, w_smooth=planner.W_SMOOTH):
    """Maximin-sigma DP over the coupled lattice.

    Transitions per s step: dphi in {-1, 0, +1} WITH WRAP (phi is a circle),
    dq7 in {-1, 0, +1} without, any branch, ||dq||_inf <= jump.  Objective is
    `planner.plan`'s lexicographic (max bottleneck sigma, then min sum
    ||dq||^2), quantised the same way.  Returns the same fields, with `path`
    a list of (phi index, q7 index, branch index).
    """
    Q, V, S = lat["Q"], lat["valid"], lat["sigma"]
    M = lat["margin"]
    Ns, Np, Nq, Nb, _ = Q.shape
    Qf = np.where(V[..., None], Q, 0.0)
    NEG, INF = -np.inf, np.inf
    Sq = np.where(V, np.round(S * planner._SIGMA_Q), NEG)

    ncand = 9 * Nb
    parent = np.full((Ns, Np, Nq, Nb), -1, np.int32)   # (t*Nb + kb_prev)
    B = Sq[0].copy()
    C = np.where(V[0], 0.0, INF)
    last_ok = 0 if np.isfinite(C).any() else -1
    if last_ok < 0:
        return dict(ok=False, cut_index=-1, cut_s=0.0)
    for i in range(1, Ns):
        cb = np.full((Np, Nq, Nb, ncand), NEG)
        cc = np.full((Np, Nq, Nb, ncand), INF)
        t = 0
        for dp in (-1, 0, 1):
            Qp = np.roll(Qf[i - 1], dp, axis=0)
            Vp = np.roll(V[i - 1], dp, axis=0)
            Bp = np.roll(B, dp, axis=0)
            Cp = np.roll(C, dp, axis=0)
            for dj in (-1, 0, 1):
                lo, hi = max(0, dj), min(Nq, Nq + dj)
                if lo >= hi:
                    t += 1
                    continue
                tgt, src = slice(lo, hi), slice(lo - dj, hi - dj)
                d = Qf[i][:, tgt][:, :, :, None, :] - Qp[:, src][:, :, None, :, :]
                dinf = np.max(np.abs(d), axis=-1)
                step = np.sum(d * d, axis=-1)
                feas = (V[i][:, tgt][:, :, :, None] & Vp[:, src][:, :, None, :]
                        & (dinf <= jump) & (Bp[:, src] > NEG)[:, :, None, :])
                sl = slice(t * Nb, (t + 1) * Nb)
                cb[:, tgt, :, sl] = np.where(
                    feas, np.minimum(Bp[:, src][:, :, None, :],
                                     Sq[i][:, tgt][:, :, :, None]), NEG)
                cc[:, tgt, :, sl] = np.where(
                    feas, Cp[:, src][:, :, None, :] + w_smooth * step, INF)
                t += 1
        best_b = cb.max(axis=-1)
        k = np.argmin(np.where(cb == best_b[..., None], cc, INF), -1)
        Cn = np.take_along_axis(cc, k[..., None], -1)[..., 0]
        if not np.isfinite(Cn).any():
            break
        Bn = np.take_along_axis(cb, k[..., None], -1)[..., 0]
        B = np.where(np.isfinite(Cn), Bn, NEG)
        C = Cn
        parent[i] = np.where(np.isfinite(Cn), k, -1)
        last_ok = i

    ok = last_ok == Ns - 1
    out = dict(ok=ok, cut_index=last_ok, cut_s=last_ok / max(Ns - 1, 1))
    if last_ok < 1:
        return out
    tie = B == B.max()
    idx = np.unravel_index(np.argmin(np.where(tie, C, INF)), C.shape)
    out["bottleneck"] = float(B[idx]) / planner._SIGMA_Q
    path = [(int(idx[0]), int(idx[1]), int(idx[2]))]
    for i in range(last_ok, 0, -1):
        k = int(parent[i][path[-1]])
        assert k >= 0, f"missing parent at step {i}"
        tt, kb = divmod(k, Nb)
        dp, dj = divmod(tt, 3)
        dp, dj = dp - 1, dj - 1
        p, j, _ = path[-1]
        path.append(((p - dp) % lat["Q"].shape[1], j - dj, kb))
    path = path[::-1]
    qs = np.array([Q[i][p] for i, p in enumerate(path)])
    assert np.all(np.isfinite(qs)), "planned path visits an invalid node"
    out.update(qs=qs, path=path,
               phi_idx=np.array([p[0] for p in path]),
               q7s=np.array([lat["q7s"][p[1]] for p in path]),
               margins=np.array([M[i][p] for i, p in enumerate(path)]),
               sigmas=np.array([S[i][p] for i, p in enumerate(path)]))
    return out


def _phi_of_path(lat, res):
    """UNWRAPPED per-step phi values of a DP path. -> (Ns,)"""
    phis = lat["phis"]
    Np = len(phis)
    step = 2 * np.pi / Np
    idx = res["phi_idx"]
    d = np.diff(idx)
    d = np.where(d > Np // 2, d - Np, np.where(d < -Np // 2, d + Np, d))
    return phis[idx[0]] + np.concatenate([[0.0], np.cumsum(d)]) * step


def poses_phi(pts_xy, phi, Twb_inv, pen_ext, pen_lat):
    """Base-frame TCP poses for a stroke with PER-SAMPLE tool yaw. -> (M,4,4)"""
    p = np.asarray(pts_xy, float)
    phi = np.broadcast_to(np.asarray(phi, float), (len(p),))
    off = tool_offset(pen_ext, pen_lat)
    c, s = np.cos(phi), np.sin(phi)
    T = np.tile(np.eye(4), (len(p), 1, 1))
    # R = rotz(phi) @ rotx(pi): columns (cos, sin, 0), (sin, -cos, 0), (0,0,-1)
    T[:, 0, 0], T[:, 0, 1] = c, s
    T[:, 1, 0], T[:, 1, 1] = s, -c
    T[:, 2, 2] = -1.0
    tip = np.column_stack([p[:, 0], p[:, 1], np.zeros(len(p))])
    T[:, :3, 3] = tip - np.einsum("nij,j->ni", T[:, :3, :3], off)
    return np.asarray(Twb_inv, float) @ T


def _rescue(poly, spec, o, phis, notes):
    """The coupled-lattice rescue: DP -> dense chase -> independent validate.

    -> a plan dict (status "ok") or None.  The chase is `pwl.chase_cc` with
    per-sample poses at the interpolated phi, all three gates enforced; the
    result is validated by `validate.validate_plan` exactly as a fixed-phi
    plan would be.
    """
    L = stroke_api.polyline_length(poly)
    ds_lat = stroke_api._fit_ds(L, o["ds_lattice"])
    pts, _ = planner.resample(poly, ds_lat)
    lat = build_lattice_phi(pts, spec, phis, h_inv=o["h_inv"],
                            pen_ext=o["pen_ext"], pen_lat=o["pen_lat"],
                            n_q7=RESCUE_N_Q7)
    res = plan_phi(lat)
    if not res.get("ok"):
        return None
    phi_path = _phi_of_path(lat, res)
    s_lat = np.linspace(0.0, 1.0, len(pts))
    ds_dense = stroke_api._fit_ds(L, o["ds_dense"])
    dense_pts, s = planner.resample(poly, ds_dense)
    q7_d = np.interp(s, s_lat, res["q7s"])
    phi_d = np.interp(s, s_lat, phi_path)
    poses = poses_phi(dense_pts, phi_d, np.linalg.inv(lat["Twb"]),
                      lat["pen_ext"], lat["pen_lat"])
    ch = pwl.chase_cc(poses, q7_d, res["qs"][0], pen_ext=lat["pen_ext"],
                      pen_lat=lat["pen_lat"], margin_gate=o["margin_gate"],
                      sigma_gate=o["sigma_gate"], jump_gate=planner.JUMP_THRESH)
    if not ch["ok"]:
        return None
    qs = ch["qs"]
    dq = np.abs(np.diff(qs, axis=0))
    pc = pacing.pace(qs, L, v_draw=o["v_draw"], safety=o["safety"],
                     ds_m=ds_dense)
    rep = validate_plan(dense_pts, spec, qs, times=pc["t"], h_inv=o["h_inv"],
                        pen_ext=o["pen_ext"], pen_lat=lat["pen_lat"],
                        margin_gate=o["margin_gate"],
                        sigma_gate=o["sigma_gate"]) \
        if o["validate"] else dict(ok=True)
    if not rep["ok"]:
        return None
    knots = np.column_stack([s, q7_d])
    return dict(status="ok", reason="", arm=getattr(spec, "arm_id", None),
                depth=0, arc_len=L, arc_len_input=L, clip_s=(0.0, 1.0),
                n_lattice=len(pts), notes=notes + ["phi-varying rescue"],
                stroke=poly, phi=phi_d, phi_mode="varying",
                pen_lat=float(lat["pen_lat"]),
                knots=knots, qs=qs, pts=dense_pts, times=pc["t"], s=s,
                q7=q7_d, sigmas=ch["sigmas"], margins=ch["margins"],
                min_sigma=float(ch["sigmas"].min()),
                min_margin=float(ch["margins"].min()),
                tip_err=float(rep["worst"]["tip_err"]),
                max_step=float(dq.max()) if len(dq) else 0.0,
                sum_travel=float(dq.sum()), n_knots=int(len(knots)),
                n_dense=int(len(qs)), sheet=-1, n_sheets=0,
                objective="maximin_sigma_phi", dp_travel=float(dq.sum()),
                q7_span=float(np.ptp(q7_d)), fallbacks=int(ch["fallbacks"]),
                windows=np.zeros(0), n_bisect=0,
                total_time=float(pc["total_time"]),
                frac_slowed=float(pc["frac_slowed"]),
                headroom=float(pc["headroom"]), validation=rep,
                coverage_gap=0.0)


# --------------------------------------------------------------------------
# the adaptive entry point (reached from stroke_api.plan_stroke)
# --------------------------------------------------------------------------
def plan_adaptive(pts_xy, spec, opts=None, n_phi=N_PHI, rescue=True):
    """Plan one stroke with the lateral tool: coarse phi ring, then rescue.

    Same contract as `stroke_api.plan_stroke` (ok | split | degenerate | bug,
    never raises); every returned plan went through the standard certification
    and carries `phi` (the tool yaw it was planned at), `pen_lat`, and a
    `lateral` metadata dict (ring, screen scores, per-phi outcomes, timings).
    """
    t0 = time.perf_counter()
    o = dict(stroke_api.DEFAULTS)
    o.update(opts or {})
    o["pen_lat"] = lat_of(o.get("pen_lat"))
    notes = []
    if float(o.get("tilt_max_deg", 0.0)) > 0:
        notes.append("pen tilt is not supported with the lateral holder; "
                     "planned with the pen vertical")
        o["tilt_max_deg"] = 0.0
    ring = phi_ring(pts_xy, spec, n_phi)
    try:
        phis, scores = screen_phis(pts_xy, spec, ring, o)
    except Exception:
        phis, scores = ring, [None] * len(ring)
    t_screen = time.perf_counter() - t0

    tried, best_split, best_key = [], None, (-1.0, -1.0)
    for phi in phis:
        r = stroke_api.plan_stroke(pts_xy, spec, dict(o, phi=float(phi)))
        st = r.get("status")
        tried.append((float(phi), st,
                      float(r.get("s_star", 1.0 if st == "ok" else 0.0))))
        if st == "ok":
            r["lateral"] = dict(n_phi=int(n_phi), phi=float(phi),
                                screen_scores=scores, tried=tried,
                                t_screen=t_screen,
                                t_total=time.perf_counter() - t0,
                                rescue_used=False)
            return r
        if st == "degenerate":
            return r                      # input hygiene is phi-independent
        if st == "split":
            key = (float(r.get("s_star", 0.0)), float(r.get("s_reach", 0.0)))
            if key > best_key:
                best_key, best_split = key, r
        # a "bug" at one phi does not condemn the others; keep going

    if rescue and best_split is not None:
        rr = _rescue(np.asarray(best_split.get("stroke", pts_xy), float),
                     spec, o, ring, notes)
        if rr is not None:
            rr["lateral"] = dict(n_phi=int(n_phi), screen_scores=scores,
                                 tried=tried, t_screen=t_screen,
                                 t_total=time.perf_counter() - t0,
                                 rescue_used=True)
            return rr

    out = best_split if best_split is not None else dict(
        status="split", reason="empty_fiber", s_star=0.0, s_reach=0.0,
        head=None, arm=getattr(spec, "arm_id", None), depth=0)
    out["lateral"] = dict(n_phi=int(n_phi), screen_scores=scores, tried=tried,
                          t_screen=t_screen,
                          t_total=time.perf_counter() - t0, rescue_used=False)
    out.setdefault("notes", [])
    out["notes"] = list(out["notes"]) + notes
    return out
