"""FR3 kinematic model & constants — single source of truth for the project.

Conventions (gate-validated against the real arm-31 touchdown, 2026-07-12,
see Aris_Kindt branch diemut-operator-ika:frame_arm31.py):
  - Craig modified DH, frames identical to the franka URDF link frames.
  - hand TCP = J7 origin + TCP_D along tool z with the Rz(-pi/4) flange twist.
  - pen tip = hand TCP + PEN_EXT along tool z (no CAD pen model exists;
    the pen is gripped by the stock Franka Hand).
  - FR3 joint limits are applied HERE, in python: the analytic IK .so
    hardcodes Panda limits and must not be trusted for limit checking.
"""
import numpy as np

# --- joint space ---
FR3_MIN = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
FR3_MAX = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])
TAU_MAX = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])  # Nm
# rad/s. SOURCE: the FR3 URDF (my_ros2_ws/src/fr3/fr3.urdf, expanded from
# franka_description via operator_franka_patches/fr3.urdf.xacro) — <limit
# velocity=...> on fr3_joint1..7.  Independently confirmed by libfranka's own
# rate limiter (franka_ros2_ws/src/libfranka/include/franka/rate_limiting.h:122
# saturates at exactly these values), so it is the firmware's number too.
# The URDF wins over the datasheet figures this was first drafted with
# ([2.0, 1.0, 1.5, 1.25, 3.0, 1.5, 3.0]), which were uniformly stricter.
# NOT the MoveIt config (fr3_moveit_config/config/joint_limits.yaml: 2.175 x4,
# 2.61 x3) — those are Panda values copied wholesale, a different robot.
# CAVEAT: libfranka enforces a POSITION-dependent envelope (min of this flat cap
# and a sqrt braking curve near each joint stop), so the flat cap is only
# available away from the limits.  We keep margin >= 0.15 rad everywhere and
# pace at safety = 0.8, which stays inside the braking curve by a wide margin.
QD_MAX = np.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26])

Q_READY_FLOOR = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
# measured on arm 31, Desk fine-adjust 2026-05-29 — standard inverted seed
Q_READY_INV = np.array([-2.2876, -1.60, -0.8564, -2.0905, 1.6853, 2.3160, 1.0468])
# FINAL-RIG ready poses, derived 2026-08-21 (docs/FINAL_RIG.md): analytic-IK
# hover 0.10 m above the paper in each arm's comfortable patch, best
# min(margin, 2.5 sigma) among solutions that PASS validate.check_pose with
# the frame boxes active (margin >= 0.30, sigma >= 0.14, frame clearance >=
# STATIC_MARGIN + 0.02, pen above paper).  SEEDS, not measurements; replace
# with Desk fine-adjusts once the rig stands.  The legacy Q_READY_INV is NOT
# valid on the final rig: at h = 0.922 its tip is 6.6 mm BELOW the paper and
# its elbow 18 mm from the boom (tests/test_final_rig.py pins the refusal).
# Arm 13 keeps Q_READY_FLOOR (checked clean on the final rig: tip +0.39 m).
Q_READY_INV_FINAL = np.array(   # arm 31, hover over canvas (0.55, 1.05)
    [-0.6858, -1.2181, 1.0783, -2.5881, -2.1652, 1.5101, 0.9886])
Q_READY_WALL = np.array(        # arm 2, hover over canvas (1.50, 1.20)
    [1.5988, 1.4251, -1.0822, -2.3822, -2.4899, 2.2478, 1.7795])

# --- tool chain (metres, along tool z) ---
D_FLANGE = 0.107      # J7 -> flange
D_HAND_TCP = 0.1034   # flange -> hand TCP
TCP_D = D_FLANGE + D_HAND_TCP   # = 0.2104, the solver's d7e
PEN_EXT = 0.110       # hand TCP -> pen tip (gate-B validated at MZ=0.924)
# --- FINAL RIG tool (pen holder CAD; docs/FINAL_RIG.md "Pen holder") ------
# The holder is CLAMPED BY THE HAND'S FINGERS (custom fingertips, half-width
# 28.5 mm); the flange->hand chain is stock, so TCP_D stays the solver
# convention.  The CAD does NOT reproduce the scalar pen model: the complete
# 10-deg "natural hold" build puts the tip at (-8.0, 0, +45.3) mm FROM THE
# TCP in the hand frame with the pen axis tilted 10 deg about y_hand; the
# newer 23-deg clutch build has ADJUSTABLE protrusion (tip not determined by
# CAD; the upstream 0.209 m flange->tip needs ~90 mm protrusion = 45 mm off
# axis).  PEN_EXT = 0.110 above is a REAL touchdown measurement and remains
# the planning default until the deployed build+protrusion is confirmed;
# these constants are the CAD's own numbers, ready for that day.
TIP_HAND_HOLDER10 = np.array([-0.00804, 0.0, 0.14866])  # tip, panda_hand frame
PEN_TILT_HOLDER10 = np.deg2rad(10.0)   # pen axis about y_hand (23.0 for clutch)
PEN_EXT_HOLDER10 = 0.14866 - 0.1034    # = 0.0453: the along-z part, from TCP

# --- modified DH: (alpha_{i-1}, a_{i-1}, d_i) ---
DH = [(0, 0, 0.333), (-np.pi / 2, 0, 0), (np.pi / 2, 0, 0.316),
      (np.pi / 2, 0.0825, 0), (-np.pi / 2, -0.0825, 0.384),
      (np.pi / 2, 0, 0), (np.pi / 2, 0.088, 0)]


def rotx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def roty(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rot_axis(axis, ang):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def fk(q, tcp=TCP_D):
    """Hand-TCP pose in link0 + the 9 chain points (base, J1..J7, TCP)
    used for clearance checks."""
    T = np.eye(4)
    pts = [T[:3, 3].copy()]
    for (al, a, d), th in zip(DH, q):
        ca, sa, ct, st = np.cos(al), np.sin(al), np.cos(th), np.sin(th)
        T = T @ np.array([[ct, -st, 0, a], [st * ca, ct * ca, -sa, -sa * d],
                          [st * sa, ct * sa, ca, ca * d], [0, 0, 0, 1]])
        pts.append(T[:3, 3].copy())
    c, s = np.cos(-np.pi / 4), np.sin(-np.pi / 4)
    T = T @ np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, tcp], [0, 0, 0, 1.0]])
    pts.append(T[:3, 3].copy())
    return T, np.array(pts)


def _ext():
    """The C++ extension, if it carries the batch entry points, else None.

    Imported lazily: `ik` imports THIS module at load time, so the dependency
    can only ever run the other way at call time.  A station venv on an older
    wheel simply gets None and the python loops below.
    """
    from . import ik                      # lazy: see docstring
    return ik._IK if ik.has_batch() else None


def fk_many(qs, tcp=TCP_D):
    """`fk` for a whole array. (N,7) -> (T (N,4,4), pts (N,9,3)).

    The C++ chain is a literal transcription of `fk` above and reproduces it
    BIT for bit (tests/test_planner_robustness.py pins that), so a caller
    gating on a geometric threshold cannot tell the two apart.
    """
    qs = np.ascontiguousarray(np.asarray(qs, float).reshape(-1, 7))
    ext = _ext()
    if ext is not None:
        d = ext.fk_batch(qs, tcp)
        return d["T"], d["pts"]
    T = np.empty((len(qs), 4, 4))
    P = np.empty((len(qs), 9, 3))
    for i, q in enumerate(qs):
        T[i], P[i] = fk(q, tcp)
    return T, P


def tip_pos(q, pen_ext=PEN_EXT):
    """Pen tip position in link0."""
    T, _ = fk(q)
    return T[:3, 3] + T[:3, :3] @ np.array([0.0, 0.0, pen_ext])


def tip_pos_many(qs, pen_ext=PEN_EXT):
    """`tip_pos` for a whole array. (N,7) -> (N,3)."""
    T, _ = fk_many(qs)
    return T[:, :3, 3] + T[:, :3, :3] @ np.array([0.0, 0.0, pen_ext])


def joint_margin(q):
    """Worst distance to a joint limit (rad). Strict comfort gate: >= 0.30."""
    return float(np.min(np.minimum(q - FR3_MIN, FR3_MAX - q)))


def joint_margin_many(qs):
    """`joint_margin` for a whole array. (N,7) -> (N,)."""
    qs = np.asarray(qs, float).reshape(-1, 7)
    return np.min(np.minimum(qs - FR3_MIN, FR3_MAX - qs), axis=1)
