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

from .frames import TAU_MAX, tip_pos

GATE_MARGIN = 0.30   # rad, strict joint-limit comfort (IKA)
GATE_SIGMA = 0.14    # sigma_min force-sensing floor (ika_plan dual mask)


def tip_jacobian(q, eps=1e-5):
    """3x7 position Jacobian of the pen tip (central finite differences)."""
    J = np.zeros((3, 7))
    for j in range(7):
        dq = np.zeros(7)
        dq[j] = eps
        J[:, j] = (tip_pos(q + dq) - tip_pos(q - dq)) / (2 * eps)
    return J


def sigma_min(J):
    return float(np.linalg.svd(J, compute_uv=False)[-1])


def f_max(J, press_dir_base, cap=100.0):
    """Max sustainable force along press_dir_base (unit, in link0 frame)."""
    tau_per_N = np.abs(J.T @ np.asarray(press_dir_base, float))
    return float(min(cap, np.min(TAU_MAX / np.maximum(tau_per_N, 1e-9))))
