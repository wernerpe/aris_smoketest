"""Per-configuration quality metrics: controllability and force capacity.

sigma_min: smallest singular value of the 3x7 pen-tip position Jacobian —
force controllability in the worst direction. libfranka CLAMPS the external
wrench estimate to zero near singularities, so the force loop is blind
exactly where this collapses. Field-validated strict gate: >= 0.14.

f_max: largest normal force (N) the arm can press with before any joint
torque limit saturates: tau = J^T (n * f)  =>  f_max = min_j tau_max_j / |(J^T n)_j|.
Gravity is not included — treat as a comparative capacity map, not a
guarantee at the last Nm.
"""
import numpy as np

from .frames import TAU_MAX, tip_pos, PEN_EXT

GATE_MARGIN = 0.30   # rad, strict joint-limit comfort (IKA)
GATE_SIGMA = 0.14    # sigma_min force-sensing floor (ika_plan dual mask)


def tip_jacobian(q, eps=1e-5, pen_ext=PEN_EXT):
    """3x7 position Jacobian of the pen tip (central finite differences)."""
    J = np.zeros((3, 7))
    for j in range(7):
        dq = np.zeros(7)
        dq[j] = eps
        J[:, j] = (tip_pos(q + dq, pen_ext) - tip_pos(q - dq, pen_ext)) / (2 * eps)
    return J


def tip_jacobian_many(qs, pen_ext=PEN_EXT):
    """`tip_jacobian` for a whole array. (N,7) -> (N,3,7).

    Where the extension has the batch entry points this is the ANALYTIC
    geometric Jacobian, z_i x (p_tip - p_i), not a finite difference — exact,
    and 14 forward kinematics per configuration cheaper.  The two agree to
    ~3e-10, i.e. to the central differences' own truncation error, and
    `tip_jacobian` above stays the reference the fast path is tested against
    (tests/test_planner_robustness.py::test_analytic_tip_jacobian_matches_fd).
    """
    qs = np.ascontiguousarray(np.asarray(qs, float).reshape(-1, 7))
    from . import ik                       # lazy: ik imports frames, not metrics
    if ik.has_batch():
        return ik._IK.tip_jacobian_batch(qs, float(pen_ext))
    if not len(qs):
        return np.zeros((0, 3, 7))
    return np.array([tip_jacobian(q, pen_ext=pen_ext) for q in qs])


def sigma_min(J):
    return float(np.linalg.svd(J, compute_uv=False)[-1])


def sigma_min_many(Js):
    """`sigma_min` for a stack of Jacobians. (N,3,7) -> (N,)."""
    Js = np.asarray(Js, float)
    if not len(Js):
        return np.zeros(0)
    return np.linalg.svd(Js, compute_uv=False)[:, -1]


def f_max(J, press_dir_base, cap=100.0):
    """Max sustainable force along press_dir_base (unit, in link0 frame)."""
    tau_per_N = np.abs(J.T @ np.asarray(press_dir_base, float))
    return float(min(cap, np.min(TAU_MAX / np.maximum(tau_per_N, 1e-9))))
