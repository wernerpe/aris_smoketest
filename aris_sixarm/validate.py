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

from .frames import (FR3_MAX, FR3_MIN, PEN_EXT, QD_MAX, fk_many,
                     joint_margin_many, tip_pos_many)
from .metrics import sigma_min_many as _sigma_min_many, tip_jacobian_many

TIP_TOL = 2e-3           # m, pen tip must stay on the commanded curve
MARGIN_GATE = 0.15       # rad, joint-limit comfort (planner.HARD_MARGIN)
SIGMA_GATE = 0.10        # pen-tip Jacobian sigma_min (pwl.SIGMA_GATE)
JUMP_GATE = 0.35         # rad, ||dq||_inf between dense samples
Z_CLEAR = 0.02           # m, chain points above the paper
BOOM_R = 0.12            # m, inverted-mount boom cylinder radius (base frame)
BOOM_Z = -0.02           # m, below which the boom cylinder is an obstacle
EPS = 1e-6               # float slack on the gates


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


def validate_plan(pts_xy, spec, qs, times=None, h_inv=None, pen_ext=PEN_EXT,
                  tip_tol=TIP_TOL, margin_gate=MARGIN_GATE,
                  sigma_gate=SIGMA_GATE, jump=JUMP_GATE, qd_max=QD_MAX,
                  clearance=True, eps=EPS):
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
                 min_chain_z=float("nan"))
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

        # ---- 1. the pen drew the stroke -----------------------------------
        tip_w = tip_pos_many(qs, pen_ext) @ Rwb.T + twb
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

        # ---- 2. joint limits, margin, controllability ---------------------
        lo = FR3_MIN - qs                       # > 0 => below the lower limit
        hi = qs - FR3_MAX
        for i, j in zip(*np.where(lo > 0)):
            add("joint_limit_low", i, qs[i, j], FR3_MIN[j])
        for i, j in zip(*np.where(hi > 0)):
            add("joint_limit_high", i, qs[i, j], FR3_MAX[j])

        marg = joint_margin_many(qs)
        sig = _sigma_min_many(tip_jacobian_many(qs, pen_ext=pen_ext))
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
            _, p = fk_many(qs)                       # (M,9,3) chain points
            pw = p @ Rwb.T + twb
            z = pw[:, 1:, 2].min(axis=1)             # lowest link, per sample
            worst["min_chain_z"] = float(z.min())
            for i in np.flatnonzero(z < Z_CLEAR - eps):
                add("paper_clearance", i, z[i], Z_CLEAR)
            if spec.mount == "inv":
                rb = np.hypot(p[:, :, 0], p[:, :, 1])
                hit = (p[:, :, 2] < BOOM_Z) & (rb < BOOM_R)
                for i in np.flatnonzero(hit.any(axis=1)):
                    add("boom_keepout", i, float(rb[i][hit[i]].min()), BOOM_R)

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
