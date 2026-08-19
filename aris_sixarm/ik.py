"""Analytic IK wrapper (He/Liu solver, wernerpe python bindings).

IMPORTANT conventions:
  - We call the RAW `_franka_ik.solve_ik(T16, q7, seed)`: T16 is the
    hand-TCP pose (Rz(-pi/4) twist included), flattened COLUMN-major.
    The higher-level `franka_analytical_ik.SolveIK` python wrapper instead
    expects the FLANGE pose and adds the hand offset internally — do not mix.
  - The .so hardcodes PANDA limits; every solution is re-filtered against
    FR3 limits here (frames.FR3_MIN/MAX).
  - The solver CLAMPS at the workspace boundary instead of failing.  Near the
    inner boundary (elbow folded, q4 close to its stop) it returns q2 = 0.0
    exactly with no NaN, inside every joint limit, for a pose it misses by up
    to ~2 cm.  Nothing downstream can tell that apart from a real solution —
    the lattice would gate it on margin and sigma_min and happily plan through
    it — so every solution is FK-verified here against the pose that was asked
    for.  Search on the solver, certify against `frames.fk`.
"""
import os
import sys

import numpy as np

from .frames import FR3_MIN, FR3_MAX, fk, joint_margin

POSE_TOL = 1e-6      # m / dimensionless: real solutions land within ~2e-12

_IK_PATH = os.environ.get(
    "ARIS_FRANKA_IK_PATH",
    "/home/franka/aris_project/franka_analytical_ik/franka_analytical_ik")
sys.path.insert(0, _IK_PATH)
try:
    # local build tree (system python3.12: _franka_ik.cpython-312-*.so)
    import _franka_ik as _IK  # noqa: E402
except ImportError:
    # installed wheel (e.g. the pydrake venv on python3.10, where the cp312
    # extension above is invisible). Same C++ solver, same raw entry points.
    from franka_analytical_ik import _franka_ik as _IK  # noqa: E402

Q7_GRID = np.linspace(FR3_MIN[6] + 0.05, FR3_MAX[6] - 0.05, 16)


def _as_T(T_base_tcp):
    """(4,4) hand-TCP pose, from a 4x4 or a column-major flat 16."""
    T = np.asarray(T_base_tcp, float)
    return T if T.shape == (4, 4) else T.reshape((4, 4), order="F")


def reaches(T, q, tol=POSE_TOL):
    """Does FK(q) reproduce the requested hand-TCP pose?  See module docstring:
    a False here is the solver clamping at the workspace boundary, not a near
    miss — genuine solutions reproduce the pose to ~1e-12."""
    Tf, _ = fk(q)
    return bool(np.linalg.norm(Tf[:3, 3] - T[:3, 3]) <= tol
                and np.max(np.abs(Tf[:3, :3] - T[:3, :3])) <= tol)


def solve(T_base_tcp, q7, seed, tol=POSE_TOL):
    """All FR3-valid branches for one (pose, q7). -> list of q (7,)."""
    out = []
    T = _as_T(T_base_tcp)
    T16 = T.flatten(order="F")
    for q in _IK.solve_ik(T16, float(q7), np.asarray(seed, float)):
        if np.any(np.isnan(q)):
            continue
        if np.any(q < FR3_MIN) or np.any(q > FR3_MAX):
            continue
        q = np.asarray(q, float)
        if not reaches(T, q, tol):
            continue
        out.append(q)
    return out


def solve_cc(T_base_tcp, q7, q_prev, tol=POSE_TOL):
    """Case-consistent single-branch solve (for chasing along strokes)."""
    T = _as_T(T_base_tcp)
    T16 = T.flatten(order="F")
    q = np.asarray(_IK.solve_ik_cc(T16, float(q7), np.asarray(q_prev, float)), float)
    if q.shape != (7,) or np.any(np.isnan(q)) or np.any(q < FR3_MIN) or np.any(q > FR3_MAX):
        return None
    return q if reaches(T, q, tol) else None


def scan(T_base_tcp, seed, q7_grid=Q7_GRID):
    """Scan the q7 redundancy: yields (q7, q, margin) for every valid branch."""
    for q7 in q7_grid:
        for q in solve(T_base_tcp, q7, seed):
            yield q7, q, joint_margin(q)
