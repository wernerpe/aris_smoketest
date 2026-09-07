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
# EXTENDED-POLE BUILD (rig_final6.FLEET_FINAL6_OPT): the side arm's base drops
# 0.20 m to canvas z 0.576, which drags Q_READY_WALL's pen 0.100 m BELOW the
# paper — the same failure mode as the legacy Q_READY_INV on the final rig, and
# the reason that pose exists.  Re-derived 2026-08-21 by the identical recipe
# (hover 0.10 m over the paper, 16 tool yaws x ik.Q7_GRID x 4 branches, gated by
# validate.check_pose with the EXTENDED-pole frame boxes active, best
# min(margin, 2.5 sigma)) over the arm's own half of the canvas, y <= 1.35 so
# that the park pose stays clear of the mirror plane where the other three
# hanging arms live: hover canvas (1.00, 0.99), margin 0.680, sigma 0.272,
# min chain z 0.21 m, min frame clearance 0.557 m.
Q_READY_WALL_LOW = np.array(    # arm 2 / 97 at canvas z 0.576
    [-0.3733, -0.8839, 1.0723, -2.2148, 2.1269, 3.5589, 2.1750])

# --- tool chain (metres, along tool z) ---
D_FLANGE = 0.107      # J7 -> flange
D_HAND_TCP = 0.1034   # flange -> hand TCP
TCP_D = D_FLANGE + D_HAND_TCP   # = 0.2104, the solver's d7e
PEN_EXT = 0.110       # hand TCP -> pen tip (gate-B validated at MZ=0.924)

# --- LATERAL PEN HOLDER (2026-08-25; RE-DERIVED FROM THE PHOTO 2026-09-03) --
# The real pen holder offsets the pen LATERALLY from the wrist axis, along the
# hand's x-axis (perpendicular to the finger-travel direction):
#
#     tip = TCP + R_tcp @ (PEN_LAT, 0, PEN_EXT_ACTIVE)
#
# WHAT THESE TWO NUMBERS ARE, AND WHAT THEY ARE NOT.  The INLINE pen's
# `PEN_EXT = 0.110` above IS a real touchdown measurement (gate B, MZ 0.924)
# and NOTHING here touches it.  The holder's own pair never was one:
# `PEN_LAT_HOLDER = 0.110` was USER-SPECIFIED on 2026-08-25 from an estimate
# ("tip ~15 cm below the bottom of the gripper's white housing"), and the
# holder simply borrowed the inline pen's 0.110 for its axial part because
# nobody had measured the holder's own.  Several places in the repo said that
# pair was "gate-validated"; it never was, and they now say what this comment
# says (docs/SYSTEM_MODEL.md 7a/7c/7e).
#
# USER-SPECIFIED 2026-09-03, from the photo of the real gripper: the pen tip
# sits ~5 cm below the bottom edge of the Fat Franka Finger blades' contact
# plates.  That edge is at panda_hand z = 0.1122421 m
# (rig_final.FATFINGER["plate_link_z"][1] + the finger joint's own 0.0584), so
# the tip is at z = 0.1622421 and, from the hand TCP at D_HAND_TCP = 0.1034,
#
#     PEN_EXT_HOLDER = 0.1622421 - 0.1034 = 0.0588421 m       (AXIAL, fixed)
#
# ...AND BOTH HALVES WERE RE-DERIVED ON 2026-09-07, FROM THE HOLDER RATHER
# THAN FROM THE BLADES.  Pete, reviewing the side view along the jaw axis:
# "shorten the shaft of the pen and bring the tip closer to the cylindrical
# pen holder, I think it only juts out 3-4 cm max", and "in the picture the
# square part of the holder is flush with the metal part, so the entire
# pen-holder plastic bit that is being pinched is about 20 degrees more
# upright".  Two statements, and the second one is a MEASUREMENT in disguise.
#
#   THE BLOCK SQUARE TO THE HAND *IS* THE BORE AT 23 DEG, and the housing's
#   own STL says so to 0.00 deg.  The mount post is a 26 mm square section
#   whose four flats are clocked `PENHOLDER22["post_clock"]` = 23.00 deg about
#   the post axis relative to the bore (measured: the four large flats' face
#   normals sit at -113.00 / -23.00 / +67.00 / +157.00 deg off the housing's
#   own +X, which is the bore).  The post axis is hand y, so those normals
#   live in the hand's x-z plane.  Place the bore at 23 deg and they land at
#   -180 / -90 / 0 / +90 deg off hand z — flush with the blades' plate faces
#   and edges, which is exactly what the photograph shows.  Place it at 45 and
#   they land at -158 / -68 / +22 / +112: the block sits askew, 22 deg off,
#   which is the "about 20 degrees more upright" Pete is asking for.
#
#   So the housing's file-name angle IS the mounted lean after all.  That is
#   what 7a concluded on 2026-09-02 and what the casing argument of 2026-09-03
#   briefly overturned; the casing argument was itself withdrawn on 2026-09-04
#   (it was measured with the grip on the finger centreline), and nothing now
#   stands against 23 deg.
#
#   AND THE TIP COMES OFF THE HOLDER NOW, NOT OFF THE BLADES.  Pete first
#   said "3-4 cm max" past the cylindrical body and then, confirming the
#   corrected orientation ("looks great now"), settled it at about 2 cm ->
#   20 mm of graphite past the cap's outer face, which is 30.001 mm from the
#   grip along the bore:
#
#       tip = grip + (30.001 + 20.000) mm * u,   u = (sin 23, 0, cos 23)
#
#   THIS REPLACES the "tip 50 mm below the bottom edge of the blades" rule of
#   2026-09-03.  That rule is SUPERSEDED, and this time it is also
#   CONTRADICTED: the derived tip sits 37.184 mm below the plate edge, not 50.
#   The blades' edge was a plausible datum for a tip nobody had measured; the
#   holder is a better one, because the protrusion is the thing a person can
#   actually see and adjust.  One corroboration that is not an argument but is
#   worth writing down: 20 mm of graphite + the 85.100 mm barrel + the 72.514
#   assembly tail is a 177.6 mm stick, and a Cretacolor Monolith is 175 mm.
#
# ...AND THE LATERAL HALF MOVED ON 2026-09-04, for a reason that has
# nothing to do with the pen and everything to do with WHERE THE HOLDER IS
# CLAMPED.  USER-SPECIFIED, from the same photograph read a second time: the
# holder sits at the FAR END of the Fat finger plates — "the tip of the finger
# extension" — not on the finger centreline where the plate's own 6 mm hole
# is.  The blade's contact plate runs link x -9.000 .. +79.500 mm, so a 26 mm
# post with its outer face flush with that far edge has its axis at
# 79.500 - 13.000 = 66.500 mm: `rig_final.PENHOLDER22["grip_hand_x"]`.  The
# whole holder slides 66.5 mm along hand x with it, and so does the tip:
#
#     PEN_LAT_HOLDER = grip_hand_x + PEN_EXT_HOLDER * tan(PEN_LEAN_HOLDER)
#                    = 0.066500 + 0.0195369 = 0.0860369 m
#
# NOTE WHAT `PEN_LEAN_HOLDER` NOW MEANS.  It is the BORE's lean off the hand's
# approach axis, measured at the GRIP — which is what a protractor on the real
# tool would read.  It is no longer the angle of the TCP -> tip ray, because
# the grip is no longer on the TCP: that ray leans atan2(0.1253421, 0.0588421)
# = 64.85 deg.  `penholder22_T_hand` knows the difference and aims the bore at
# the grip -> tip ray.
#
# WHAT DOES NOT ENTER THE CHOICE, ON THE USER'S INSTRUCTION (2026-09-04): the
# housing's clearance from the gripper casing.  The pencil's modelled length
# is arbitrary until somebody measures the real protrusion, so a barrel or a
# tail that intersects the hand shell is a drawing artefact and not a fact
# about the build.  The numbers are still MEASURED and REPORTED, at this
# placement, against the manufacturer's own hand collision shell:
#
#     housing +13.84 mm    cap  +59.90 mm    the 72.5 mm pencil tail  -10.61
#
# (the housing and the tail do not move with the graphite; only the tip does)
#
# — the barrel is clear and the TAIL runs 11 mm INTO the casing, which is the
# artefact Pete named.  AND THE ONE ARGUMENT THIS WITHDREW: on 2026-09-03,
# with the grip on the finger centreline, the casing ruled out any lean under
# 35.17 deg and that was the reason given here for 45 deg.  At the far-end
# placement it does not — 23 deg clears the shell BETTER than 45 did — and the
# lean is now 23 deg on the housing's own geometry anyway.  docs/SYSTEM_MODEL.md 7e.
#
# Refine by touchdown calibration once the holder is mounted
# (docs/DECISIONS.md).  What one ruler reading would settle is in
# docs/SYSTEM_MODEL.md 7e.
#
# `PEN_LAT` and `PEN_EXT_ACTIVE` are the ACTIVE pair and they DEFAULT TO the
# inline pen (0.0, 0.110) — the tool every published number and every pinned
# test was earned with.  Switch the whole stack to the lateral holder the same
# way rigs are switched:
#
#     ARIS_TOOL=lateral python3 scripts/whatever.py     (read by __init__.py)
#     frames.activate_tool("lateral")                    (in process, tests)
#
# Every function that takes `pen_lat=None` or `pen_ext=None` resolves None to
# the ACTIVE value at call time, so a single switch reaches the planner, the
# atlas, the validator, the capsule models and the transit router
# consistently.  BOTH halves switch together: before 2026-09-03 only the
# lateral offset did, and a lateral run silently kept the inline pen's axial
# 0.110.
PEN_LEAN_HOLDER = np.deg2rad(23.0)   # rad, the BORE's lean off the approach
                                     # axis, measured AT THE GRIP.  It is the
                                     # housing's own clocking: the block sits
                                     # SQUARE to the hand at this angle and
                                     # nowhere else (see above).
PEN_GRAPHITE_HOLDER = 0.020  # m, graphite standing past the CAP's outer face
                             # ("about 2 cm", USER-SPECIFIED 2026-09-07 with
                             # the orientation confirmed against the real
                             # gripper).  The tip derives from this and the
                             # holder; it is no longer set by the blades.
PEN_EXT_HOLDER = 0.0460262   # m, the holder's AXIAL tip depth below the TCP
                             # = (0.030001 + PEN_GRAPHITE_HOLDER) * cos(lean)
PEN_LAT_HOLDER = 0.0860369   # m, = grip_hand_x
                             #     + (0.030001 + PEN_GRAPHITE_HOLDER)*sin(lean)
PEN_LAT = 0.0            # m, ACTIVE lateral offset (0.0 = legacy inline pen)
PEN_EXT_ACTIVE = PEN_EXT  # m, ACTIVE axial depth (PEN_EXT = the inline pen)
TOOL_NAMES = ("inline", "lateral")
ACTIVE_TOOL = "inline"


def activate_tool(name):
    """Select the ACTIVE tool model ("inline" | "lateral"), in this process.

    Sets BOTH halves of the tool offset — the lateral `PEN_LAT` and the axial
    `PEN_EXT_ACTIVE`.  Until 2026-09-03 it set only the first, so a lateral run
    drew the holder's 45 deg ray out to the INLINE pen's 0.110 m of axial
    depth; the holder's own is `PEN_EXT_HOLDER`.
    """
    global PEN_LAT, PEN_EXT_ACTIVE, ACTIVE_TOOL
    if name not in TOOL_NAMES:
        raise ValueError(f"unknown tool {name!r}; want one of {TOOL_NAMES}")
    lateral = name == "lateral"
    PEN_LAT = PEN_LAT_HOLDER if lateral else 0.0
    PEN_EXT_ACTIVE = PEN_EXT_HOLDER if lateral else PEN_EXT
    ACTIVE_TOOL = name
    return PEN_LAT


def lat_of(pen_lat=None):
    """Resolve a `pen_lat` argument: None means the ACTIVE tool's offset."""
    return PEN_LAT if pen_lat is None else float(pen_lat)


def ext_of(pen_ext=None):
    """Resolve a `pen_ext` argument: None means the ACTIVE tool's depth.

    The twin of `lat_of`, and it exists for the same reason: `PEN_EXT` is a
    module constant, so a signature written `pen_ext=PEN_EXT` binds the INLINE
    pen's 0.110 AT IMPORT and no later `activate_tool` can reach it.  Every
    such default in this package is `None` instead, and resolves here.
    """
    return PEN_EXT_ACTIVE if pen_ext is None else float(pen_ext)


def tool_offset(pen_ext=None, pen_lat=None):
    """The tip offset in the hand-TCP frame -> (3,).  tip = TCP + R @ this."""
    return np.array([lat_of(pen_lat), 0.0, ext_of(pen_ext)])
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
#
# CONFIRMED 2026-09-02, independently, off the assembly itself.  Re-deriving
# the 10-deg build from `Natural hold assembly - closed.SLDASM`'s own component
# transforms (scripts/read_solidworks.py) gives: bore 10.0000 deg off the
# hand's approach axis and 90.0000 deg off finger travel; grip centre 103.2 mm
# from panda_hand against this file's 0.1034; and the pencil's SHARP point
# 46.096 mm from that grip centre at (7.93, 0, 45.41) mm — which is this
# constant, TCP + (-8.0, 0, +45.3), to a tenth of a millimetre.  The sign of x
# is a mounting choice (square post, seats either way up), not a disagreement.
# The 46.1 is the cap end at 25.361 mm plus 20.7 mm of pencil past it, i.e.
# docs/FINAL_RIG.md's "21 mm protrusion" measured rather than assumed.
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


def tip_pos(q, pen_ext=None, pen_lat=None):
    """Pen tip position in link0.  `pen_lat=None` -> the ACTIVE tool."""
    T, _ = fk(q)
    return T[:3, 3] + T[:3, :3] @ tool_offset(pen_ext, pen_lat)


def tip_pos_many(qs, pen_ext=None, pen_lat=None):
    """`tip_pos` for a whole array. (N,7) -> (N,3)."""
    T, _ = fk_many(qs)
    return T[:, :3, 3] + T[:, :3, :3] @ tool_offset(pen_ext, pen_lat)


def tool_points_many(T, pen_ext=None, pen_lat=None):
    """Chain points of the TOOL beyond the TCP, from (N,4,4) TCP poses.

    -> list of (N,3) arrays: [tip] for the inline pen; [tip, corner] for the
    lateral holder, where `corner` is the bracket elbow TCP + R @ (lat, 0, 0)
    — the point the two-capsule tool model (bracket TCP->corner, pen
    corner->tip) hangs on.  Callers append these to `fk`'s 9 chain points, so
    the chain is 10 points inline and 11 lateral, and every capsule table
    selects on that width.
    """
    T = np.asarray(T, float)
    lat = lat_of(pen_lat)
    tip = T[:, :3, 3] + T[:, :3, :3] @ tool_offset(pen_ext, lat)
    if lat == 0.0:
        return [tip]
    corner = T[:, :3, 3] + T[:, :3, :3] @ np.array([lat, 0.0, 0.0])
    return [tip, corner]


def joint_axes_many(qs):
    """Joint axes and origins for a whole array, plus the hand-TCP pose.

    (N,7) -> (z (N,7,3), p (N,7,3), T_tcp (N,4,4)).  In this modified-DH
    chain joint i rotates about frame i's own z axis, so `z[:, i]` / `p[:, i]`
    are the axis and origin the geometric Jacobian z_i x (p_tool - p_i) needs.
    Vectorised transcription of `fk`'s loop; agrees with it bit for bit on the
    origins (same arithmetic, same order).
    """
    qs = np.asarray(qs, float).reshape(-1, 7)
    N = len(qs)
    T = np.tile(np.eye(4), (N, 1, 1))
    zs = np.empty((N, 7, 3))
    ps = np.empty((N, 7, 3))
    for i, (al, a, d) in enumerate(DH):
        ca, sa = np.cos(al), np.sin(al)
        ct, st = np.cos(qs[:, i]), np.sin(qs[:, i])
        A = np.zeros((N, 4, 4))
        A[:, 0, 0], A[:, 0, 1], A[:, 0, 3] = ct, -st, a
        A[:, 1, 0], A[:, 1, 1], A[:, 1, 2], A[:, 1, 3] = st * ca, ct * ca, -sa, -sa * d
        A[:, 2, 0], A[:, 2, 1], A[:, 2, 2], A[:, 2, 3] = st * sa, ct * sa, ca, ca * d
        A[:, 3, 3] = 1.0
        T = T @ A
        zs[:, i] = T[:, :3, 2]
        ps[:, i] = T[:, :3, 3]
    c, s = np.cos(-np.pi / 4), np.sin(-np.pi / 4)
    F = np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1.0, TCP_D], [0, 0, 0, 1.0]])
    return zs, ps, T @ F


# --- the LINK frames, not just the chain points ---------------------------
# `fk` walks the modified-DH chain and keeps each step's ORIGIN, which is all a
# capsule drawn about the chain's own segments ever needed.  A capsule drawn
# about a LINK's own metal — which is what `selfcoll` measures, because the
# metal's principal axis is nothing like the segment between two joint origins
# — needs the whole transform, and so does anything that wants to pose the
# manufacturer's meshes.  Index i is link i for i in 0..7, 8 is the flange
# (link8) and 9 is the hand frame: the same order and the same frames
# `scripts/collision_audit.py` names in `FRAMES`, which is the order the
# meshes are delivered in.  `out[:, 9] @ trans(0, 0, D_HAND_TCP)` is `fk`'s
# `T`, and `out[:, i, :3, 3]` is `fk`'s point i, both bit for bit
# (tests/test_selfcoll.py pins them).
LINK_FRAMES = ("link0", "link1", "link2", "link3", "link4", "link5", "link6",
               "link7", "link8", "hand")


def link_frames_many(qs, tcp=TCP_D):
    """Per-link frames of the FR3 chain in link0. (N,7) -> (N,10,4,4)."""
    qs = np.asarray(qs, float).reshape(-1, 7)
    N = len(qs)
    out = np.empty((N, 10, 4, 4))
    T = np.tile(np.eye(4), (N, 1, 1))
    out[:, 0] = T
    for i, (al, a, d) in enumerate(DH):
        ca, sa = np.cos(al), np.sin(al)
        ct, st = np.cos(qs[:, i]), np.sin(qs[:, i])
        A = np.zeros((N, 4, 4))
        A[:, 0, 0], A[:, 0, 1], A[:, 0, 3] = ct, -st, a
        A[:, 1, 0], A[:, 1, 1], A[:, 1, 2], A[:, 1, 3] = st * ca, ct * ca, -sa, -sa * d
        A[:, 2, 0], A[:, 2, 1], A[:, 2, 2], A[:, 2, 3] = st * sa, ct * sa, ca, ca * d
        A[:, 3, 3] = 1.0
        T = T @ A
        out[:, i + 1] = T
    F = np.eye(4)
    F[2, 3] = tcp - D_HAND_TCP                 # link7 -> flange (link8)
    out[:, 8] = out[:, 7] @ F
    c, s = np.cos(-np.pi / 4), np.sin(-np.pi / 4)
    R = np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]])
    out[:, 9] = out[:, 8] @ R                  # the flange twist -> hand frame
    return out


def joint_margin(q):
    """Worst distance to a joint limit (rad). Strict comfort gate: >= 0.30."""
    return float(np.min(np.minimum(q - FR3_MIN, FR3_MAX - q)))


def joint_margin_many(qs):
    """`joint_margin` for a whole array. (N,7) -> (N,)."""
    qs = np.asarray(qs, float).reshape(-1, 7)
    return np.min(np.minimum(qs - FR3_MIN, FR3_MAX - qs), axis=1)
