"""Analytic IK wrapper (He/Liu solver, wernerpe python bindings).

IMPORTANT conventions:
  - We call the RAW `_franka_ik.solve_ik(T16, q7, seed)`: T16 is the
    hand-TCP pose (Rz(-pi/4) twist included), flattened COLUMN-major.
    The higher-level `franka_analytical_ik.SolveIK` python wrapper instead
    expects the FLANGE pose and adds the hand offset internally — do not mix.
  - The .so hardcodes PANDA limits; every solution is re-filtered against
    FR3 limits here (frames.FR3_MIN/MAX).
  - BATCH vs SCALAR.  `solve_batch` / `fk_batch` / `tip_jacobian_batch` do the
    same work as `solve` / `frames.fk` / `metrics.tip_jacobian` for a whole
    array at once, inside the C++ extension.  They are OPTIONAL: a station venv
    on an older wheel (the cp310 one the drake demo imports) carries only the
    scalar entry points, so each wrapper looks for its C++ counterpart at CALL
    time and otherwise runs the python loop.  Detection is per call rather than
    at import so a test can take the extension away and exercise the fallback.
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

from .frames import FR3_MIN, FR3_MAX, PEN_EXT, TCP_D, fk_many, joint_margin

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
    miss — genuine solutions reproduce the pose to ~1e-12.

    `solve_cc` calls this once per dense sample of every chase, so it is worth
    the batched FK even for a single configuration: `fk_many` is bit-identical
    to `fk` and about six times cheaper to reach.
    """
    Tf = fk_many(q)[0][0]
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


# --------------------------------------------------------------------------
# batch entry points (see the module docstring on detection and fallback)
# --------------------------------------------------------------------------
BATCH_POS_TOL = 1e-9      # C++ FK verification: real solutions land at ~1e-12,
BATCH_ROT_TOL = 1e-9      # boundary-clamped ones miss by millimetres


def has_batch():
    """Does the loaded extension carry the batch entry points?"""
    return all(hasattr(_IK, n)
               for n in ("solve_batch", "fk_batch", "tip_jacobian_batch"))


def _flat16(T):
    """(N,4,4) or (N,16) -> (N,16) column-major flattened poses."""
    T = np.asarray(T, float)
    if T.ndim == 3:
        return np.ascontiguousarray(T.transpose(0, 2, 1)).reshape(len(T), 16)
    return np.ascontiguousarray(T).reshape(-1, 16)


def solve_batch(T_base_tcp, q7, seed, tol=POSE_TOL):
    """`solve` for a whole array of (pose, q7) pairs.

    Args:
      T_base_tcp  (N,4,4) hand-TCP poses, or (N,16) already column-major flat.
      q7          (N,) redundancy parameter per pose.
      seed        (7,) or (N,7) solver seed.

    Returns (Q, valid): Q is (N,4,7) and valid is (N,4).  Branches are
    COMPACTED to the front in solver order, so `Q[i][:valid[i].sum()]` is
    exactly `solve(T_base_tcp[i], q7[i], seed)` — same solutions, same order,
    same slots.  Invalid slots are NaN.  Callers index by slot (the lattice's
    branch axis), which is why compaction is part of the contract and not an
    afterthought.
    """
    flat = _flat16(T_base_tcp)
    q7 = np.ascontiguousarray(np.atleast_1d(np.asarray(q7, float)))
    seed = np.asarray(seed, float)
    N = len(flat)

    if has_batch():
        raw = _IK.solve_batch(flat, q7, seed, BATCH_POS_TOL, BATCH_ROT_TOL)
        # NaN (solver refusal, or FK verification) => False in both comparisons
        ok = (np.all(raw >= FR3_MIN, axis=-1) & np.all(raw <= FR3_MAX, axis=-1))
    else:                                        # older wheel: the scalar path
        raw = np.full((N, 4, 7), np.nan)
        ok = np.zeros((N, 4), bool)
        for i in range(N):
            s = seed if seed.ndim == 1 else seed[i]
            sols = solve(flat[i].reshape((4, 4), order="F"), q7[i], s, tol)
            for k, q in enumerate(sols[:4]):
                raw[i, k] = q
                ok[i, k] = True
        return raw, ok                           # `solve` already compacted

    # stable partition: kept branches to the front, solver order preserved
    order = np.argsort(~ok, axis=1, kind="stable")
    Q = np.take_along_axis(raw, order[..., None], axis=1)
    valid = np.take_along_axis(ok, order, axis=1)
    return np.where(valid[..., None], Q, np.nan), valid


def fk_batch(qs, tcp=TCP_D):
    """`frames.fk` for a whole array. (N,7) -> (T (N,4,4), pts (N,9,3)).

    Kinematics belong to `frames`; this is the name the IK-side callers reach
    for, and it is the same function.
    """
    return fk_many(qs, tcp)


def tip_jacobian_batch(qs, pen_ext=None, pen_lat=None):
    """Pen-tip position Jacobians for a whole array. (N,7) -> (N,3,7).

    The C++ path is the ANALYTIC geometric Jacobian, z_i x (p_tip - p_i); the
    fallback is `metrics.tip_jacobian`'s central finite differences, which cost
    14 forward kinematics apiece.  They agree to ~3e-10, i.e. to the finite
    differences' own truncation error, so a caller gating on sigma_min cannot
    tell them apart.
    """
    from .metrics import tip_jacobian_many     # lazy: metrics is not an import
    return tip_jacobian_many(qs, pen_ext, pen_lat)
