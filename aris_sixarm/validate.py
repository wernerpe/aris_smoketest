"""Independent validation of a planned stroke — the certificate, not the plan.

Everything upstream reports on itself: `pwl.chase_report` measures the chase it
just ran, `smooth.certify` believes the gates it enforced, `pacing.pace` times
the path it was handed.  Each of those numbers is only as trustworthy as the
bookkeeping that produced it, and the end goal — six arms, hundreds of strokes —
multiplies every per-stroke failure probability together.  So this module takes
the RAW outputs (the stroke asked for, the arm, the joint samples, optionally
the clock) and re-derives every invariant from scratch:

  * FK the pen tip through `frames.fk` + the arm's own base transform and
    compare it with the stroke that was requested (< 2 mm);
  * re-derive the pen's LEAN from the same FK and check it against the cone the
    material allows (`tilt_max_deg`, default 0 = perpendicular);
  * recompute the joint margin and sigma_min from the configurations, never
    reading a planner field;
  * check continuity, the FR3 joint limits, paper clearance and the inverted
    arms' boom keep-out with the same geometry the lattice gates with, computed
    here a second time;
  * with a clock: strictly increasing times and |dq/dt| inside the FR3
    velocity limits.

It shares only `frames`/`metrics` with the planner — no planner state, no
lattice, no sheet.  A plan that passes here is a plan whose invariants hold in
the kinematics, whatever the pipeline believed about itself.  The checks are
run as whole-array operations (`fk_many`, `tip_jacobian_many`) rather than
sample by sample, which is a change of speed and not of substance: the batched
FK is bit-identical to `frames.fk`, and the analytic Jacobian agrees with the
finite-difference reference to ~3e-10 — both pinned by tests.

`validate_plan` NEVER raises: a malformed input or an internal error is a
violation record (`kind="validator_error"`), because a validator that throws is
just another way for a campaign to lose a result.
"""
import traceback

import numpy as np

from . import rig_final
from .frames import (FR3_MAX, FR3_MIN, PEN_EXT, QD_MAX, fk_many,
                     joint_margin_many, tip_pos_many, tool_offset,
                     tool_points_many, lat_of)
from .metrics import sigma_min_many as _sigma_min_many, tip_jacobian_many

TIP_TOL = 2e-3           # m, pen tip must stay on the commanded curve
CONE_GATE = 0.0          # deg, pen lean from vertical the plan may use
CONE_EPS_DEG = 1e-6      # deg, float slack on the cone.  A PERPENDICULAR plan
# measures 6e-11 deg of lean over the whole CSAIL corpus (arctan2, not arccos —
# see `_pen_lean_deg`), so 1e-6 is five orders of magnitude of headroom above
# the noise and still refuses a lean no artist would call perpendicular.
MARGIN_GATE = 0.15       # rad, joint-limit comfort (planner.HARD_MARGIN)
SIGMA_GATE = 0.10        # pen-tip Jacobian sigma_min (pwl.SIGMA_GATE)
JUMP_GATE = 0.35         # rad, ||dq||_inf between dense samples
Z_CLEAR = 0.02           # m, chain points above the paper
BOOM_R = 0.12            # m, inverted-mount boom cylinder radius (base frame)
BOOM_Z = -0.02           # m, below which the boom cylinder is an obstacle
EPS = 1e-6               # float slack on the gates

# --- the arm against itself, re-derived here (2026-08-26) -----------------
# `selfcoll` is the producers' implementation; this is the certificate's, and
# the two share nothing but the MEASURED table and `frames`.  The table is a
# mesh measurement (`scripts/self_collision_audit.py`, pinned against
# out/self_collision_audit.json) and restating 29 fitted capsules by hand would
# add typos, not independence — what has to be independent is the DERIVATION,
# and all of it is below: this module builds its own capsule endpoints, its own
# segment distance, its own pair list from its own chain-distance rule, and
# holds them to its own copy of the margin.  `tests/test_selfcoll.py` runs the
# two against each other over thousands of configurations.
SELF_MARGIN = 0.020      # m, restated from selfcoll.SELF_MARGIN on purpose
SELF_CHAIN_D = 4         # joints of separation below which the mechanism owns
#                          the pair; restated from selfcoll.WATCH_CHAIN_D


def _self_pairs(caps):
    pos = dict(link0=0, link1=1, link2=2, link3=3, link4=4, link5=5, link6=6,
               link7=7, hand=8, tool=8)
    body = [c[0] for c in caps] + ["tool", "tool"]
    n = len(body)
    return [(i, j) for i in range(n) for j in range(i + 1, n)
            if abs(pos[body[j]] - pos[body[i]]) >= SELF_CHAIN_D]


def self_clearance(qs, pen_ext=None, pen_lat=None):
    """Worst gap between two bodies of one arm, own derivation. (N,7) -> (N,)"""
    from .frames import D_HAND_TCP, ext_of, link_frames_many
    from .selfcoll import BODY_CAPSULES, TOOL_R_INLINE, TOOL_R_LAT
    qs = np.asarray(qs, float).reshape(-1, 7)
    if not len(qs):
        return np.zeros(0)
    lat = lat_of(pen_lat)
    T = link_frames_many(qs)
    A, B, R = [], [], []
    for _, _, f, a, b, r in BODY_CAPSULES:
        Rf, tf = T[:, f, :3, :3], T[:, f, :3, 3]
        A.append(Rf @ np.asarray(a, float) + tf)
        B.append(Rf @ np.asarray(b, float) + tf)
        R.append(r)
    Rf, tf = T[:, 9, :3, :3], T[:, 9, :3, 3]
    rt = TOOL_R_LAT if lat != 0.0 else TOOL_R_INLINE
    tcp = Rf @ np.array([0.0, 0.0, D_HAND_TCP]) + tf
    cor = Rf @ np.array([lat, 0.0, D_HAND_TCP]) + tf
    tip = Rf @ np.array([lat, 0.0, D_HAND_TCP + ext_of(pen_ext)]) + tf
    A += [tcp, cor]
    B += [cor, tip]
    R += [rt, rt]
    out = np.full(len(qs), np.inf)
    for i, j in _self_pairs(BODY_CAPSULES):
        out = np.minimum(out, _seg_seg(A[i], B[i], A[j], B[j]) - R[i] - R[j])
    return out


def _seg_seg(p0, p1, q0, q1):
    """Segment-to-segment distance, by DENSE SAMPLING of one segment against
    the other's exact point-to-segment distance, plus the sampling residual.

    Own derivation on purpose, and deliberately not the closed form the
    producers use: a bound that samples every <= 5 mm and subtracts half a step
    cannot share an algebra bug with a stationary-point solve, and the capsules
    here are 0.04-0.19 m long so 64 samples is well under that.
    """
    K = 64
    ts = np.linspace(0.0, 1.0, K)
    P = p0[:, None, :] + ts[:, None] * (p1 - p0)[:, None, :]
    ab = q1 - q0
    den = np.sum(ab * ab, -1)[:, None]
    t = np.where(den > 1e-15,
                 np.einsum("nkj,nj->nk", P - q0[:, None, :], ab)
                 / np.where(den > 1e-15, den, 1.0), 0.0)
    t = np.clip(t, 0.0, 1.0)
    d = P - (q0[:, None, :] + t[..., None] * ab[:, None, :])
    lo = np.sqrt(np.sum(d * d, -1)).min(axis=1)
    return lo - np.linalg.norm(p1 - p0, axis=-1) / (2 * (K - 1))


def _dist_to_polyline(P, poly):
    """Per-point distance from (M,2) points to a (N,2) polyline (segments)."""
    a, b = poly[:-1], poly[1:]
    if not len(a):
        return np.linalg.norm(P - poly[0], axis=1)
    d = b - a                                       # (S,2)
    L2 = np.einsum("ij,ij->i", d, d)
    L2 = np.where(L2 > 0, L2, 1.0)
    w = P[:, None, :] - a[None, :, :]               # (M,S,2)
    t = np.clip(np.einsum("msj,sj->ms", w, d) / L2, 0.0, 1.0)
    proj = a[None, :, :] + t[..., None] * d[None, :, :]
    return np.linalg.norm(P[:, None, :] - proj, axis=2).min(axis=1)


def _pen_lean_deg(T, Rwb):
    """Per-sample pen lean from vertical IN WORLD (deg), from FK'd tool frames.

    ARCTAN2, NEVER ARCCOS.  `arccos` of the down-component is ill-conditioned
    exactly where a perpendicular plan lives — it reports ~1e-8 rad of phantom
    lean for a pen that is vertical to the last bit of double precision, which
    is enough to fail a zero-degree cone.  The two-argument arctangent of
    (sideways, down) is conditioned the other way round and measures the same
    angle.  Re-derived here from `frames.fk` and the arm's own base transform:
    the plan's own `tilt`/`max_lean_deg` fields are never read.
    """
    axis_w = (Rwb @ np.asarray(T, float)[:, :3, 2].T).T     # pen axis in world
    return np.rad2deg(np.arctan2(np.linalg.norm(axis_w[:, :2], axis=1),
                                 -axis_w[:, 2]))


def validate_plan(pts_xy, spec, qs, times=None, h_inv=None, pen_ext=None,
                  tip_tol=TIP_TOL, margin_gate=MARGIN_GATE,
                  sigma_gate=SIGMA_GATE, jump=JUMP_GATE, qd_max=QD_MAX,
                  clearance=True, eps=EPS, tilt_max_deg=CONE_GATE,
                  pen_lat=None):
    """Re-derive every invariant of a planned stroke.  Never raises.

    Args:
      pts_xy   the stroke as requested, (M,2) in world/paper xy.  When it has
               exactly one point per joint sample the comparison is pointwise
               (the strong check); otherwise the tip is measured against the
               polyline itself (correspondence-free, noted in the report).
      spec     the owning `fleet.ArmSpec` (its `T_world_base` puts the FK in
               world, its `mount` selects the boom keep-out).
      qs       (M,7) dense joint samples — the plan.
      times    optional (M,) clock; enables the velocity and monotonicity
               checks.
      h_inv    inverted-mount height, passed to `spec.T_world_base`.
      pen_ext  pen length used by the plan (tip = TCP + pen_ext along tool z).
      tilt_max_deg
               the cone the MATERIAL allows the pen to lean inside, in degrees.
               0 (the default) is the perpendicular pen the planner has always
               commanded, so every plan written before pen tilt existed is
               checked against the pose it was actually asked for.  A tilted
               plan (`aris_sixarm/tilt.py`) is validated against the cone it was
               granted and against nothing else: the tip check above is already
               orientation-aware — `tip_pos_many` steps `pen_ext` along the tool
               z of the FK'd pose — so a leaning pen that draws the curve passes
               it, and WITHOUT this gate nothing downstream would ever notice a
               plan that leaned 40 degrees to get there.

    Returns dict:
      ok           no violations
      n            number of samples checked
      violations   [{kind, index, value, limit}, ...] — index is the sample
                   (or the first of the pair, for per-step checks), value what
                   was measured, limit what was required
      worst        the extremes: tip_err, min_margin, min_sigma, max_step,
                   max_qd_frac (peak |dq/dt| / QD_MAX), min_chain_z
      notes        human-readable remarks (e.g. a non-pointwise comparison)
    """
    V, notes = [], []

    def add(kind, index, value, limit):
        V.append(dict(kind=kind, index=int(index), value=float(value),
                      limit=float(limit)))

    worst = dict(tip_err=float("nan"), min_margin=float("nan"),
                 min_sigma=float("nan"), max_step=0.0, max_qd_frac=0.0,
                 min_chain_z=float("nan"), max_lean_deg=float("nan"))
    try:
        qs = np.asarray(qs, float)
        if qs.ndim != 2 or qs.shape[1] != 7 or len(qs) == 0:
            add("shape", 0, qs.size, 7)
            return dict(ok=False, n=0, violations=V, worst=worst,
                        notes=["qs is not a non-empty (M,7) array"])
        if not np.all(np.isfinite(qs)):
            for i in np.flatnonzero(~np.isfinite(qs).all(axis=1)):
                add("nonfinite_q", i, 0.0, 0.0)
            return dict(ok=False, n=len(qs), violations=V, worst=worst,
                        notes=["qs contains non-finite entries"])
        M = len(qs)
        pts = np.asarray(pts_xy, float)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) == 0:
            add("shape", 0, pts.size, 2)
            return dict(ok=False, n=M, violations=V, worst=worst,
                        notes=["pts_xy is not a non-empty (N,2) array"])

        Twb = spec.T_world_base() if h_inv is None else spec.T_world_base(h_inv)
        Rwb, twb = Twb[:3, :3], Twb[:3, 3]

        # ONE FK FOR THE WHOLE CERTIFICATE.  The tip, the pen axis and the
        # chain points are three readings of the same forward kinematics, and
        # this module used to call `fk_many` three times to get them (once
        # inside `tip_pos_many`, once for the chain, once more for the frame
        # gate's tip).  `tip_b` below is `tip_pos_many`'s body verbatim, so the
        # numbers are bit-identical to the three-call version — pinned by
        # tests/test_validate_cone.py::test_tip_matches_tip_pos_many.
        T_fk, P_fk = fk_many(qs)
        # `pen_lat=None` resolves to the ACTIVE tool (frames.PEN_LAT): the
        # lateral holder's tip is TCP + R @ (pen_lat, 0, pen_ext), re-derived
        # here from the same FK — the plan's own fields are never read.
        pen_lat = lat_of(pen_lat)
        tip_b = T_fk[:, :3, 3] + T_fk[:, :3, :3] @ tool_offset(pen_ext, pen_lat)

        # ---- 1. the pen drew the stroke -----------------------------------
        tip_w = tip_b @ Rwb.T + twb
        if len(pts) == M:
            err = np.hypot(np.linalg.norm(tip_w[:, :2] - pts, axis=1), tip_w[:, 2])
        else:
            notes.append(f"pts_xy has {len(pts)} points for {M} samples — tip "
                         "measured against the polyline, not pointwise")
            planar = _dist_to_polyline(tip_w[:, :2], pts)
            err = np.hypot(planar, tip_w[:, 2])
        worst["tip_err"] = float(err.max())
        for i in np.flatnonzero(err > tip_tol):
            add("tip_off_curve", i, err[i], tip_tol)

        # ---- 1b. the pen stayed inside the cone the material allows -------
        # THE TIP CHECK CANNOT SEE THIS.  `tip_pos_many` steps `pen_ext` along
        # the tool z of the FK'd pose, so it is orientation-agnostic by
        # construction: a pen leaning 40 degrees that still puts its tip on the
        # curve passes section 1 exactly as a perpendicular one does.  Every
        # gate downstream is likewise about the CHAIN, not the tool.  So until
        # this gate existed, "the plan kept the lean the artist allowed" was the
        # one invariant of a tilted plan that nothing re-derived — and it is the
        # invariant the whole feature rests on.
        lean = _pen_lean_deg(T_fk, Rwb)
        worst["max_lean_deg"] = float(lean.max())
        cone = float(tilt_max_deg)
        for i in np.flatnonzero(lean > cone + CONE_EPS_DEG):
            add("pen_cone", i, lean[i], cone)

        # ---- 2. joint limits, margin, controllability ---------------------
        lo = FR3_MIN - qs                       # > 0 => below the lower limit
        hi = qs - FR3_MAX
        for i, j in zip(*np.where(lo > 0)):
            add("joint_limit_low", i, qs[i, j], FR3_MIN[j])
        for i, j in zip(*np.where(hi > 0)):
            add("joint_limit_high", i, qs[i, j], FR3_MAX[j])

        marg = joint_margin_many(qs)
        sig = _sigma_min_many(tip_jacobian_many(qs, pen_ext=pen_ext,
                                                pen_lat=pen_lat))
        worst["min_margin"] = float(marg.min())
        worst["min_sigma"] = float(sig.min())
        for i in np.flatnonzero(marg < margin_gate - eps):
            add("margin", i, marg[i], margin_gate)
        for i in np.flatnonzero(sig < sigma_gate - eps):
            add("sigma", i, sig[i], sigma_gate)

        # ---- 3. continuity -------------------------------------------------
        if M > 1:
            step = np.max(np.abs(np.diff(qs, axis=0)), axis=1)
            worst["max_step"] = float(step.max())
            for i in np.flatnonzero(step > jump + eps):
                add("continuity", i, step[i], jump)

        # ---- 4. clearance: paper, and the inverted arms' own boom ----------
        if clearance:
            p = P_fk                                 # (M,9,3) chain points
            pw = p @ Rwb.T + twb
            z = pw[:, 1:, 2].min(axis=1)             # lowest link, per sample
            worst["min_chain_z"] = float(z.min())
            for i in np.flatnonzero(z < Z_CLEAR - eps):
                add("paper_clearance", i, z[i], Z_CLEAR)
            if spec.mount == "inv" and getattr(spec, "rig", "sixarm") == "sixarm":
                rb = np.hypot(p[:, :, 0], p[:, :, 1])
                hit = (p[:, :, 2] < BOOM_Z) & (rb < BOOM_R)
                for i in np.flatnonzero(hit.any(axis=1)):
                    add("boom_keepout", i, float(rb[i][hit[i]].min()), BOOM_R)
            boxes = spec.static_obstacles() \
                if hasattr(spec, "static_obstacles") else []
            if boxes:
                tool_b = tool_points_many(T_fk, pen_ext, pen_lat)
                tool_w = [t @ Rwb.T + twb for t in tool_b]
                P10 = np.concatenate([pw] + [t[:, None, :] for t in tool_w],
                                     axis=1)
                c = rig_final.chain_static_clearance(P10, boxes)
                worst["min_frame_clearance"] = float(c.min())
                for i in np.flatnonzero(c < rig_final.STATIC_MARGIN - eps):
                    add("frame_keepout", i, float(c[i]), rig_final.STATIC_MARGIN)
            # ---- 4d. and the arm against itself ------------------------
            sc = self_clearance(qs, pen_ext=pen_ext, pen_lat=pen_lat)
            worst["min_self_clearance"] = float(sc.min())
            for i in np.flatnonzero(sc < SELF_MARGIN - eps):
                add("self_collision", i, float(sc[i]), SELF_MARGIN)

        # ---- 5. the clock --------------------------------------------------
        if times is not None:
            t = np.atleast_1d(np.asarray(times, float))
            if t.shape != (M,):
                add("time_shape", 0, t.size, M)
            elif not np.all(np.isfinite(t)):
                add("nonfinite_time", int(np.flatnonzero(~np.isfinite(t))[0]),
                    0.0, 0.0)
            else:
                dt = np.diff(t)
                for i in np.flatnonzero(dt <= 0):
                    add("time_monotonic", i, dt[i], 0.0)
                good = dt > 0
                if M > 1 and good.any():
                    qd = np.abs(np.diff(qs, axis=0)[good]) / dt[good, None]
                    frac = qd / np.asarray(qd_max, float)[None, :]
                    worst["max_qd_frac"] = float(frac.max())
                    idx = np.flatnonzero(good)
                    for a, j in zip(*np.where(frac > 1.0 + eps)):
                        add("velocity", idx[a], qd[a, j], float(qd_max[j]))
    except Exception:                                  # never raise (see module doc)
        V.append(dict(kind="validator_error", index=-1, value=0.0, limit=0.0,
                      traceback=traceback.format_exc()))
        return dict(ok=False, n=int(len(np.atleast_2d(qs))), violations=V,
                    worst=worst, notes=notes)
    return dict(ok=not V, n=int(len(qs)), violations=V, worst=worst, notes=notes)


def check_pose(q, spec, h_inv=None, pen_ext=None, margin_gate=MARGIN_GATE,
               z_clear=Z_CLEAR, eps=EPS, pen_lat=None):
    """The single-configuration half of `validate_plan`. -> dict(ok, ...).

    A POSE AN ARM STANDS IN IS NOT A PLAN, AND IS STILL A CLAIM.  When an arm
    stops where it finished rather than going home (`idle.py`), the pose it
    holds for the rest of the run was never handed to the stroke validator —
    nobody drew with it — and yet the fleet has to live next to it for a minute.
    This re-derives the three gates that are about the configuration alone and
    not about a curve: the FR3 joint limits with the same comfort margin, the
    chain's height above the paper, and the inverted arms' boom keep-out.

    Deliberately NOT checked here: tip-on-curve, sigma, continuity, velocity —
    a static pose has no curve, no motion and no clock.  Inter-arm clearance is
    somebody else's job (`coordination`/`scene_check`), because it is a property
    of the fleet and not of the pose.
    """
    q = np.asarray(q, float).reshape(1, 7)
    V, worst = [], {}
    try:
        Twb = spec.T_world_base() if h_inv is None else spec.T_world_base(h_inv)
        marg = float(joint_margin_many(q)[0])
        worst["joint_margin"] = marg
        if marg < margin_gate - eps:
            V.append(dict(kind="margin", index=0, value=marg, limit=margin_gate))
        T1, p = fk_many(q)                               # (1,9,3) base frame
        pw = p @ Twb[:3, :3].T + Twb[:3, 3]
        z = float(pw[0, 1:, 2].min())
        worst["min_chain_z"] = z
        if z < z_clear - eps:
            V.append(dict(kind="paper_clearance", index=0, value=z, limit=z_clear))
        pen_lat = lat_of(pen_lat)
        tip = Twb[:3, :3] @ tip_pos_many(q, pen_ext, pen_lat)[0] + Twb[:3, 3]
        worst["tip_z"] = float(tip[2])
        if tip[2] < -eps:
            V.append(dict(kind="pen_below_paper", index=0, value=float(tip[2]),
                          limit=0.0))
        if spec.mount == "inv" and getattr(spec, "rig", "sixarm") == "sixarm":
            rb = np.hypot(p[0, :, 0], p[0, :, 1])
            hit = (p[0, :, 2] < BOOM_Z) & (rb < BOOM_R)
            worst["boom_r"] = float(rb[hit].min()) if hit.any() else float("inf")
            if hit.any():
                V.append(dict(kind="boom_keepout", index=0,
                              value=worst["boom_r"], limit=BOOM_R))
        boxes = spec.static_obstacles() \
            if hasattr(spec, "static_obstacles") else []
        if boxes:
            tool_b = tool_points_many(T1, pen_ext, pen_lat)
            tool_w = [t @ Twb[:3, :3].T + Twb[:3, 3] for t in tool_b]
            P10 = np.concatenate([pw] + [t[:, None, :] for t in tool_w],
                                 axis=1)
            c = float(rig_final.chain_static_clearance(P10, boxes)[0])
            worst["min_frame_clearance"] = c
            if c < rig_final.STATIC_MARGIN - eps:
                V.append(dict(kind="frame_keepout", index=0, value=c,
                              limit=rig_final.STATIC_MARGIN))
        sc = float(self_clearance(q, pen_ext=pen_ext, pen_lat=pen_lat)[0])
        worst["min_self_clearance"] = sc
        if sc < SELF_MARGIN - eps:
            V.append(dict(kind="self_collision", index=0, value=sc,
                          limit=SELF_MARGIN))
    except Exception:                                  # never raise (see module doc)
        V.append(dict(kind="validator_error", index=-1, value=0.0, limit=0.0,
                      traceback=traceback.format_exc()))
    return dict(ok=not V, n=1, violations=V, worst=worst, notes=[])


def violation_summary(rep, top=6):
    """One line per distinct violation kind, worst first."""
    kinds = {}
    for v in rep["violations"]:
        k = kinds.setdefault(v["kind"], [0, v])
        k[0] += 1
    out = []
    for kind, (n, v) in sorted(kinds.items(), key=lambda kv: -kv[1][0])[:top]:
        out.append(f"{kind} x{n} (first at sample {v['index']}: "
                   f"{v['value']:.4g} vs {v['limit']:.4g})")
    return out
