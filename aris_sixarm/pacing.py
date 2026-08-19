"""TOPP-lite: give the geometric plan a clock.

Everything upstream of here is timeless.  `planner.plan`, `pwl.plan_pwl` and
`smooth.certify` all answer "which configuration at which arc length?" and stop
there — the output is a curve in joint space parameterised by s, with no
statement about when the arm is at any point of it.  That is the right place to
stop, because the redundancy decision and the timing decision have nothing to
say to each other: s is monotone, so a path that is feasible at all is feasible
at *some* speed, and choosing the speed cannot invalidate the geometry.

WHAT SETS THE SPEED.  Drawing wants a constant tip speed `v_draw` — ink flow
and line weight depend on it, and the whole point of a pen plotter is that the
line looks the same everywhere.  The arm disagrees in exactly one place: where
the stroke demands a lot of joint motion per metre of paper.  dq/ds (rad per
METRE) is a property of the geometric path alone, and the joint velocity that
results from tracking it at tip speed v is simply

    qdot_j = (dq_j/ds) * v

which is linear in v.  So the per-sample speed ceiling is a division, not a
search:

    v_i = safety * min_j ( qdot_max_j / |dq_j/ds|_i )

and the commanded speed is min(v_draw, v_i) — constant where the arm is
comfortable, backing off only through the pinches.  This is the degenerate case
of TOPP (time-optimal path parameterisation): with velocity limits only, the
feasible set at each s is an interval [0, v_i] whose upper bound is independent
of everything else, so the time-optimal profile is pointwise maximal and one
pass computes it.  No integration of forward/backward acceleration arcs, no
switching-point search.

TODO (v2): acceleration and torque limits.  Those genuinely couple s to its
neighbours — qddot = (d2q/ds2) v^2 + (dq/ds) a — and the pointwise-maximal
profile above is then no longer reachable, because slowing into a pinch takes
distance.  That is the real TOPP, and it needs the forward/backward integration
this module deliberately does not have.  Until it exists, the v profile here is
a ceiling, not a trajectory: a downstream controller still has to ramp into it.
"""
import numpy as np

from .frames import QD_MAX

V_DRAW = 0.02        # m/s, nominal pen speed on paper
SAFETY = 0.8         # fraction of the joint velocity limit we allow ourselves
V_FLOOR = 1e-4       # m/s, refuse to divide by zero when a path is impossible


def dq_ds(qs, arc_len_m, ds_m=None):
    """(M,7) dq/ds in rad per METRE of stroke.

    The plan lives in normalised s and the velocity limits live in seconds, so
    something has to restore the stroke's physical scale; this is that place.
    A 1.56 m rim arc and a 0.02 m serif with the same normalised q7(s) make
    completely different demands on the arm, and only the metre-denominated
    derivative can tell them apart.

    `qs` is assumed uniformly sampled in arc length (which is what
    `planner.resample` produces).  ds_m overrides the spacing when the caller
    knows it exactly — resample's last sample can fall short of the full length
    by up to one step, so arc_len_m / (M-1) is a slight over-estimate of ds and
    therefore a slight UNDER-estimate of dq/ds, i.e. optimistic.  Pass ds_m.
    Central differences inside, one-sided at the ends (np.gradient).
    """
    qs = np.asarray(qs, float)
    if len(qs) < 2:
        return np.zeros_like(qs)
    ds = float(arc_len_m) / (len(qs) - 1) if ds_m is None else float(ds_m)
    return np.gradient(qs, ds, axis=0)


def pace(qs, arc_len_m, v_draw=V_DRAW, qd_max=QD_MAX, safety=SAFETY,
         ds_m=None, dqds=None):
    """Time-parameterise a dense joint path under joint velocity limits.

    Returns dict:
        t           (M,) time at each sample, t[0] = 0
        v           (M,) commanded tip speed, m/s = min(v_draw, v_limit)
        v_limit     (M,) what the joint limits alone allow (may exceed v_draw)
        slow        (M,) bool, where the arm forced us below v_draw
        qd          (M,7) joint velocity actually commanded, rad/s
        qd_unpaced  (M,7) what constant v_draw WOULD have demanded
        max_qd, max_qd_unpaced   (7,) per-joint peaks of the two above
        viol        (7,) bool, joints whose unpaced peak exceeds qd_max
        total_time, naive_time   s, paced vs constant-v_draw over the same span
        arc_len, span            m, the polyline's length and what the samples
                                 actually cover (span = ds * (M-1) <= arc_len)
        frac_slowed, worst_factor (= v_draw / min(v)), headroom
        dqds, ds
    The path is integrated as dt = ds / v with the trapezoid rule on 1/v, which
    is the exact integral of ds/v(s) for v piecewise-linear in s and errs on the
    slow side elsewhere.
    """
    qs = np.asarray(qs, float)
    qd_max = np.asarray(qd_max, float)
    M = len(qs)
    ds = float(arc_len_m) / max(M - 1, 1) if ds_m is None else float(ds_m)
    if dqds is None:
        dqds = dq_ds(qs, arc_len_m, ds_m=ds)
    dqds = np.asarray(dqds, float)

    with np.errstate(divide="ignore", invalid="ignore"):
        per_joint = qd_max[None, :] / np.abs(dqds)      # m/s allowed by joint j
    per_joint = np.where(np.isfinite(per_joint), per_joint, np.inf)
    v_limit = safety * per_joint.min(axis=1)
    v = np.maximum(np.minimum(v_draw, v_limit), V_FLOOR)

    dt = ds * 0.5 * (1.0 / v[:-1] + 1.0 / v[1:])
    t = np.concatenate([[0.0], np.cumsum(dt)]) if M > 1 else np.zeros(1)
    # The naive baseline must cover the SAME span the samples do.  resample's
    # last sample can fall up to one ds short of the full polyline length, and
    # billing the paced time for (M-1)*ds against a naive time for arc_len_m
    # makes an unslowed stroke look faster than constant speed, which is
    # nonsense.  arc_len is reported separately for the record.
    span = ds * (M - 1)

    qd = dqds * v[:, None]
    qd_unpaced = dqds * v_draw
    max_qd = np.abs(qd).max(axis=0)
    max_qd_unpaced = np.abs(qd_unpaced).max(axis=0)
    slow = v < v_draw * (1 - 1e-9)
    return dict(t=t, v=v, v_limit=v_limit, slow=slow, qd=qd,
                qd_unpaced=qd_unpaced, max_qd=max_qd,
                max_qd_unpaced=max_qd_unpaced, viol=max_qd_unpaced > qd_max,
                total_time=float(t[-1]), naive_time=float(span / v_draw),
                arc_len=float(arc_len_m), span=float(span),
                frac_slowed=float(slow.mean()),
                worst_factor=float(v_draw / v.min()),
                headroom=float((max_qd / qd_max).max()),
                unpaced_headroom=float((max_qd_unpaced / qd_max).max()),
                dqds=dqds, ds=ds, v_draw=v_draw, qd_max=qd_max, safety=safety)


def pace_report(name, pc):
    """Two terse lines per stroke, the ones worth reading out loud."""
    j = int(np.argmax(pc["max_qd_unpaced"] / pc["qd_max"]))
    return [f"  pace : unpaced at {pc['v_draw']:.3f} m/s max|qd| = "
            f"{pc['max_qd_unpaced'].max():.3f} rad/s (joint {j + 1}, "
            f"{100 * pc['unpaced_headroom']:.1f} % of its limit) -> "
            + ("VIOLATES" if pc["viol"].any() else "within limits"),
            f"         paced {pc['total_time']:.2f} s vs {pc['naive_time']:.2f} s "
            f"naive ({100 * pc['frac_slowed']:.1f} % of the stroke slowed, worst "
            f"factor {pc['worst_factor']:.2f}x), peak {100 * pc['headroom']:.1f} % "
            f"of limit"]
