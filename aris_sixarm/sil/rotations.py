"""Quaternion algebra in the DEPLOYED convention: unit quaternions stored
(x, y, z, w), rotations acting on column vectors, all angles in radians.

This is the convention of the pathway CSV (contract §1) and of
`geometry_msgs/Quaternion`, so nothing in this package ever has to reorder a
quaternion except where it talks to Eigen's (w, x, y, z) STORAGE order — which
it never does, because the C++ controller is replicated in numpy here.

Pure numpy, no Drake: `impedance.py` must stay importable without a plant.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """-> the unit quaternion parallel to `q` (xyzw)."""
    q = np.asarray(q, float)
    n = float(np.linalg.norm(q))
    if n < _EPS:
        raise ValueError("cannot normalize a zero quaternion")
    return q / n


def quat_align(q: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """-> `q` or `-q`, whichever lies on `ref`'s hemisphere.

    The double cover made visible: the controller does exactly this before it
    stores a new target and before it takes an orientation error, because a
    sign flip across a waypoint would otherwise command the long way round.
    """
    q = np.asarray(q, float)
    return -q if float(np.dot(q, np.asarray(ref, float))) < 0.0 else q


def quat_conj(q: np.ndarray) -> np.ndarray:
    """-> the conjugate (= the inverse for a unit quaternion)."""
    q = np.asarray(q, float)
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """-> the Hamilton product a*b (apply b first, then a)."""
    ax, ay, az, aw = np.asarray(a, float)
    bx, by, bz, bw = np.asarray(b, float)
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def matrix_from_quat(q: np.ndarray) -> np.ndarray:
    """-> the (3,3) rotation matrix of a unit quaternion (xyzw)."""
    x, y, z, w = quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quat_from_matrix(R: np.ndarray) -> np.ndarray:
    """-> the unit quaternion (xyzw) of a (3,3) rotation matrix.

    Shepperd's branch-on-the-largest-component method: numerically safe at
    every 180-degree case, which is where the drawing poses actually live
    (a floor arm's pen-down orientation IS a pi rotation about x).
    """
    R = np.asarray(R, float)
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        q = np.array([(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
                      (R[1, 0] - R[0, 1]) / s, 0.25 * s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = np.array([0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s,
                      (R[2, 1] - R[1, 2]) / s])
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = np.array([(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s,
                      (R[0, 2] - R[2, 0]) / s])
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = np.array([(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s,
                      (R[1, 0] - R[0, 1]) / s])
    return quat_normalize(q)


def rotvec_from_quat(q: np.ndarray) -> np.ndarray:
    """-> axis * angle (3,), the rotation vector of a unit quaternion.

    Transcribes Eigen's `AngleAxis(Quaternion)`, which the C++ controller uses
    for the orientation half of its pose error: angle = 2*atan2(|vec|, |w|) so
    it lands in [0, pi], and the axis is negated when w < 0 so that the short
    way round is always the answer.
    """
    q = np.asarray(q, float)
    v, w = q[:3], float(q[3])
    n = float(np.linalg.norm(v))
    if n < _EPS:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(n, abs(w))
    axis = v / (-n if w < 0.0 else n)
    return axis * angle


def slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """-> the unit quaternion `t` of the way from `a` to `b` (xyzw).

    Transcribes `Eigen::Quaternion::slerp`: it takes the SHORT arc (the sign of
    `b` is flipped when the dot product is negative) and degrades to linear
    interpolation as the arc closes, which is the regime the 1 kHz setpoint
    filter runs in permanently (alpha = 0.05 on a per-tick orientation step of
    microradians).
    """
    a = quat_normalize(a)
    b = quat_normalize(b)
    d = float(np.dot(a, b))
    if d < 0.0:
        b, d = -b, -d
    d = min(1.0, max(-1.0, d))
    if d > 1.0 - 1e-9:                      # arcs this short: lerp is exact
        return quat_normalize(a + t * (b - a))
    theta = np.arccos(d)
    s = np.sin(theta)
    return (np.sin((1.0 - t) * theta) / s) * a + (np.sin(t * theta) / s) * b


def rotx(a: float) -> np.ndarray:
    """-> the (3,3) rotation of `a` radians about x."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def rotz(a: float) -> np.ndarray:
    """-> the (3,3) rotation of `a` radians about z."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
