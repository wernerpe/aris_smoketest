"""Analytic IK wrapper (He/Liu solver, wernerpe python bindings).

IMPORTANT conventions:
  - We call the RAW `_franka_ik.solve_ik(T16, q7, seed)`: T16 is the
    hand-TCP pose (Rz(-pi/4) twist included), flattened COLUMN-major.
    The higher-level `franka_analytical_ik.SolveIK` python wrapper instead
    expects the FLANGE pose and adds the hand offset internally — do not mix.
  - The .so hardcodes PANDA limits; every solution is re-filtered against
    FR3 limits here (frames.FR3_MIN/MAX).
"""
import os
import sys

import numpy as np

from .frames import FR3_MIN, FR3_MAX, joint_margin

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


def solve(T_base_tcp, q7, seed):
    """All FR3-valid branches for one (pose, q7). -> list of q (7,)."""
    out = []
    T16 = np.asarray(T_base_tcp, float).flatten(order="F")
    for q in _IK.solve_ik(T16, float(q7), np.asarray(seed, float)):
        if np.any(np.isnan(q)):
            continue
        if np.any(q < FR3_MIN) or np.any(q > FR3_MAX):
            continue
        out.append(np.asarray(q, float))
    return out


def solve_cc(T_base_tcp, q7, q_prev):
    """Case-consistent single-branch solve (for chasing along strokes)."""
    T16 = np.asarray(T_base_tcp, float).flatten(order="F")
    q = np.asarray(_IK.solve_ik_cc(T16, float(q7), np.asarray(q_prev, float)), float)
    if q.shape != (7,) or np.any(np.isnan(q)) or np.any(q < FR3_MIN) or np.any(q > FR3_MAX):
        return None
    return q


def scan(T_base_tcp, seed, q7_grid=Q7_GRID):
    """Scan the q7 redundancy: yields (q7, q, margin) for every valid branch."""
    for q7 in q7_grid:
        for q in solve(T_base_tcp, q7, seed):
            yield q7, q, joint_margin(q)
