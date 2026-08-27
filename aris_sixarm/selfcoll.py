"""THE ARM AGAINST ITSELF — the check nothing in this package ever made.

Until 2026-08-26 every collision gate here asked the same two questions: is
this arm clear of the OTHER arms, and is it clear of the STRUCTURE.  Nobody
asked whether it was clear of itself.  `scripts/dead_disc_anatomy.py` wrote it
down rather than fixing it — "SELF-COLLISION is not modelled anywhere in the
package ... the FR3 joint limits do most of that work in practice, but a 15 deg
leaned pose folded under its own shoulder has not been checked" — and that was
tolerable only for as long as the comfort margins stayed fat.  They are about
to stop being fat: the atlas is about to start searching a tilt cone for poses
that PASS the gates rather than poses that merely clear metal, and the drawing
planner is about to be allowed to lean the pen.  Both of those go looking in
exactly the part of the configuration space the joint limits were quietly
covering.

THE LIMITS ARE NOT ENOUGH, AND THAT IS MEASURED, NOT ASSUMED.  Over 1 200
configurations drawn uniformly from the joint box AND held to the STRICT
comfort margin (0.30 rad from every limit), the manufacturer's meshes put the
arm's own metal within 10 mm of itself on 8 of them and in contact on at least
one.  Over the whole shipped certified atlas — 23 376 strict-GO cells at
h = 0.940 with the lateral holder — the same measurement never reads below
114.3 mm.  So the gate this module adds costs the shipped map nothing and
refuses something real (`scripts/self_collision_audit.py`, `--part truth`).

WHY THIS IS NOT `coordination.CAPSULES_LAT` AGAINST ITSELF
----------------------------------------------------------
That is the first thing to try and it does not work.  The audited inter-arm
table is drawn about the KINEMATIC CHAIN's segments — the straight lines
between joint origins — and its radius is the largest distance any point of a
link reaches from that line.  Turned on itself it produces two failures:

  1. THREE PAIRS ARE NEGATIVE IN EVERY CONFIGURATION.  (1,3)-(4,5) sits at
     -178.5 mm because the elbow offset is 0.0825 m and the two radii are
     0.130 + 0.131; (4,5)-(7,8) at -147.0 across the 0.088 m wrist offset;
     (7,8)-(10,9) at -44.0 across the 0.110 m bracket.  A pair that is
     negative everywhere cannot tell a safe pose from an unsafe one.
  2. AND THE BASE COLUMN IS THE WRONG BODY.  Its band 3 is link1 swept about
     the base axis, which is precisely right for a NEIGHBOUR (which meets
     link1 at every q1) and precisely wrong for the arm itself (which meets it
     at one).  Measured on the certified atlas that band against the forearm
     reads -22.5 mm at its worst WHILE THE METAL IS 161 mm APART.  A guard
     built on it would have refused 1 021 certified cells for phantom metal.

So the self model is measured in each link's OWN frame, where a capsule can be
tight, by `scripts/self_collision_audit.py --part capsules`: the smallest
enclosing capsule of that body's mesh (the manufacturer's collision meshes
UNION the full-resolution visual meshes, link0's cable included), found by
searching segment directions and shrinking the segment.  The radii come out
roughly half the inter-arm table's — link1 0.1295 -> 0.081, link2 0.2017 ->
0.082, link4 0.1852 -> 0.086 — because these are drawn about the metal's own
principal axis instead of a line between two joint origins.  Every radius is
the exact maximum over EVERY vertex of every mesh, rounded UP to the
millimetre: the capsule contains the body by construction, so a clearance this
module reports is a lower bound on the gap in the metal.

THE PAIRS IT WATCHES, AND THE ONES IT DOES NOT
----------------------------------------------
Four joints of separation, and that number is measured.  `--part touching`
puts the largest separation each body pair ever reaches over the joint box;
the consecutive pairs never reach 1 mm (they meet at their joint), link5-link7
reaches 22 mm and link6-hand 29 mm across two short offsets, and hand-tool
3 mm because it is a grip.  Against the shipped certified map:

    chain distance >= 2   refuses ALL 23 376 strict-GO cells
    chain distance >= 3   refuses 17 807 of them
    chain distance >= 4   refuses NONE, and the tightest certified pose in the
                          whole map still holds 63.7 mm

— while the METAL at those same poses is never closer than 114.3 mm.  So
everything the closer rules refuse is capsule fat around a joint, and what is
left is the fold: the wrist, the hand and the pen coming back at the base, the
shoulder and the upper arm.

AND HERE IS WHAT THAT LEAVES UNCHECKED, said out loud because a guard that
does not say what it does not cover is worse than no guard.  Over 12 000
uniform configurations, exact mesh distance per body pair
(`--part near`, out/self_collision_audit.json):

    WATCHED    link1-link5   0.00 mm    link2-link6   0.00 mm    <- real folds,
               link0-link4  35.34 mm    link0-link5  83.52 mm       and caught
    NOT WATCHED
               link2-link5   0.00 mm    link5-hand    0.00 mm    <- REAL, and
               link5-link7   0.00 mm       (contact through one short offset)      NOT caught
               link2-link4  49.47 mm    link3-link5  52.66 mm
               link6-hand   29.04 mm    link4-link6 121.18 mm

The two four-joint folds this guard exists for are both live and both caught.
Three closer pairs can also reach contact, and this guard does not see them:
what stands between the arm and those is the FR3's joint limits and the
mechanical design, which is the same thing that stood there before — no
protection has been removed, and now the residual has a number.  Closing it
would need a decomposition several times finer than three bands per link,
because at three joints of separation the capsules overlap around the
intervening joint in every configuration an arm actually draws in.

THE TOOL IS THE PACKAGE'S OWN ENVELOPE, NOT THE CAD.  The mesh fit finds a
much tighter holder — one 0.031 m capsule along the 45-degree ray — but it
finds it by placing a CAD delivery that arrived with no assembly file, and
`rig_final` is explicit that "the inference in that placement is the residual
risk, not the radius".  A safety gate does not get to depend on an inference,
so the tool here is the same L that `rig_final.STATIC_CAPSULES_LAT` ships
(TCP -> bracket corner -> tip, both at 0.05, both CAD-validated as CONTAINING
the holder), and the inline pen is its 0.03.

WHAT THE MARGIN IS.  `SELF_MARGIN` is 0.02 m — the operating clearance this
repo demands against a static surface it knows exactly (`rig_final.Z_STATIC`,
`planner.Z_PAPER`), and deliberately WITHOUT the 0.03 calibration term the
static and inter-arm gates carry.  That term exists because base positions are
unsurveyed and two arms do not share a clock; an arm's own links share its own
encoders and its own kinematic chain, and there is nothing to calibrate
between them.

PRODUCERS USE THIS MODULE; CHECKERS RESTATE IT.  `validate` and `scene_check`
carry their own copies of the table and their own segment arithmetic, and
`tests/test_selfcoll.py` pins all three together — the same discipline the
column bands already live under.
"""
import numpy as np

from .frames import D_HAND_TCP, PEN_EXT, lat_of, link_frames_many

SELF_MARGIN = 0.020      # m of metal-to-metal clearance an arm owes itself

# ...AND WHAT A PRODUCER PAYS, WHICH IS MORE — the same argument, and the same
# arithmetic, as `rig_final.STATIC_PLAN_MARGIN`.  `validate.self_clearance` is
# a deliberately independent derivation and therefore a LOWER BOUND with slack
# in it: it samples one capsule every 1/63 of its length and subtracts half a
# step, which on the longest capsule in the table (link0 band 1, 0.264 m) is
# 2.1 mm.  Measured over 4 000 configurations the checker reads 0.7-2.1 mm
# below the exact value here and never above it.  So a producer that plans to
# exactly 20.0 mm would be refused by a checker that is right, and the
# producers pay the difference instead.  It costs nothing today: the tightest
# certified pose in the shipped map holds 63.7 mm.
SELF_SEG_SLACK = 0.0022  # measured worst residual of the checker's sampling
SELF_PLAN_PAD = 0.003    # >= that, rounded up
SELF_PLAN_MARGIN = SELF_MARGIN + SELF_PLAN_PAD

# THE TABLE.  (body, band, link-frame index, a, b, radius), with a/b in that
# frame's own coordinates.  Frames are `frames.LINK_FRAMES`: 0..7 are link0..
# link7 and 9 is the hand.  MEASURED by `scripts/self_collision_audit.py
# --part capsules` — see out/self_collision_audit.json ["capsules"]; the
# comment on each row is the exact mesh maximum the shipped radius rounds up
# from, and every radius is that maximum over EVERY vertex of the band.
#
# THREE BANDS PER BODY, BECAUSE ONE IS NOT ENOUGH.  A single fitted capsule per
# link is already half the inter-arm table's radius (link4 0.185 -> 0.086), and
# it is still mostly air near the joints — a sausage drawn round an L-shaped
# casting has to be.  Splitting each body into three bands along its own fitted
# axis takes the radii to 0.038-0.077 and puts the metal where the metal is.
# Every vertex falls in exactly one band, so the union still contains the body
# by construction.
#
# ...EXCEPT LINK0, WHICH IS THE ONE BODY THAT NEVER MOVES, and is therefore
# modelled the way the column audit already models it: seven z-bands about the
# base axis, each at the largest radius the metal reaches in that slice.  A
# principal-axis fit draws chords ACROSS the base disc, and a chord through a
# disc is mostly air: on arm 71's certified park pose the chord fit reads
# -120.4 mm against the wrist where the metal is 26.9 mm, and this stack reads
# -9.2.  The bands stop at z = 0.1440, where the collision mesh does — above
# that the arm's own metal is link1's and belongs to link1's capsules.
BODY_CAPSULES = (
    ("link0", 0, 0, (0.0000, 0.0000, -0.2380), (0.0000, 0.0000, -0.0750), 0.177),   # mesh 0.1769
    ("link0", 1, 0, (0.0000, 0.0000, -0.0750), (0.0000, 0.0000, 0.0000), 0.176),   # mesh 0.1757
    ("link0", 2, 0, (0.0000, 0.0000, 0.0000), (0.0000, 0.0000, 0.0350), 0.171),   # mesh 0.1709
    ("link0", 3, 0, (0.0000, 0.0000, 0.0350), (0.0000, 0.0000, 0.0700), 0.160),   # mesh 0.1600
    ("link0", 4, 0, (0.0000, 0.0000, 0.0700), (0.0000, 0.0000, 0.1000), 0.112),   # mesh 0.1119
    ("link0", 5, 0, (0.0000, 0.0000, 0.1000), (0.0000, 0.0000, 0.1200), 0.076),   # mesh 0.0755
    ("link0", 6, 0, (0.0000, 0.0000, 0.1200), (0.0000, 0.0000, 0.1440), 0.078),   # mesh 0.0776
    ("link1", 0, 1, (0.0011, 0.0018, -0.1069), (-0.0018, 0.0070, -0.1962), 0.063),   # mesh 0.0620
    ("link1", 1, 1, (-0.0029, -0.0576, -0.0136), (0.0148, 0.0074, -0.1251), 0.068),   # mesh 0.0678
    ("link1", 2, 1, (-0.0127, -0.0145, -0.0386), (0.0121, -0.0976, 0.0535), 0.076),   # mesh 0.0752
    ("link2", 0, 2, (-0.0008, -0.1986, -0.0081), (0.0014, -0.1105, 0.0001), 0.063),   # mesh 0.0626
    ("link2", 1, 2, (0.0103, -0.1237, 0.0012), (0.0027, -0.0163, 0.0476), 0.069),   # mesh 0.0683
    ("link2", 2, 2, (0.0093, 0.0542, 0.0977), (-0.0072, -0.0389, 0.0146), 0.075),   # mesh 0.0747
    ("link3", 0, 3, (0.0732, -0.0047, -0.0006), (0.1003, 0.1079, 0.0090), 0.062),   # mesh 0.0616
    ("link3", 1, 3, (0.0361, -0.0447, -0.0482), (0.0396, 0.0952, -0.0350), 0.075),   # mesh 0.0741
    ("link3", 2, 3, (0.0184, 0.0076, -0.1417), (-0.0404, -0.0208, -0.0299), 0.059),   # mesh 0.0588
    ("link4", 0, 4, (0.0107, -0.0025, 0.1107), (0.0043, -0.0086, -0.0034), 0.062),   # mesh 0.0615
    ("link4", 1, 4, (-0.0991, -0.0014, 0.0151), (-0.0049, 0.0772, 0.0528), 0.077),   # mesh 0.0762
    ("link4", 2, 4, (-0.0675, 0.1436, 0.0275), (-0.1157, 0.0410, -0.0440), 0.064),   # mesh 0.0636
    ("link5", 0, 5, (-0.0131, 0.0089, -0.0072), (0.0096, 0.1292, -0.0110), 0.063),   # mesh 0.0626
    ("link5", 1, 5, (0.0041, 0.1051, -0.1199), (-0.0022, 0.0072, -0.1083), 0.067),   # mesh 0.0666
    ("link5", 2, 5, (0.0141, 0.0790, -0.1657), (-0.0073, -0.0417, -0.2735), 0.061),   # mesh 0.0610
    ("link6", 0, 6, (-0.0196, -0.0460, 0.0285), (-0.0129, 0.0561, 0.0223), 0.051),   # mesh 0.0508
    ("link6", 1, 6, (0.0658, 0.0834, 0.0020), (0.0341, -0.0526, 0.0140), 0.056),   # mesh 0.0550
    ("link6", 2, 6, (0.0839, -0.0541, 0.0004), (0.1083, 0.0748, -0.0036), 0.049),   # mesh 0.0485
    ("link7", 0, 7, (0.0076, 0.0044, 0.0522), (-0.0360, -0.0316, 0.0967), 0.047),   # mesh 0.0460
    ("link7", 1, 7, (-0.0169, 0.0449, 0.0861), (0.0444, -0.0183, 0.0861), 0.044),   # mesh 0.0440
    ("link7", 2, 7, (0.0233, 0.0236, 0.0810), (0.0641, 0.0603, 0.0886), 0.038),   # mesh 0.0379
    ("hand", 0, 9, (-0.0039, 0.1035, 0.0073), (-0.0142, 0.0053, 0.1370), 0.050),   # mesh 0.0499
    ("hand", 1, 9, (-0.0025, 0.0563, -0.0294), (0.0031, -0.0700, 0.0845), 0.040),   # mesh 0.0397
    ("hand", 2, 9, (-0.0088, -0.0505, -0.0313), (0.0087, -0.1125, 0.0519), 0.045),   # mesh 0.0444
)
N_BODY = len(BODY_CAPSULES)
BRACKET, PEN = N_BODY, N_BODY + 1          # the two tool capsules, appended
N_CAP = N_BODY + 2
NAMES = tuple(f"{c[0]}.{c[1]}" for c in BODY_CAPSULES) + ("bracket", "pen")
BODY_OF = tuple(c[0] for c in BODY_CAPSULES) + ("tool", "tool")

# Tool radii, restated from `rig_final` on purpose (a test pins them):
# the lateral holder's CAD-validated 0.05 envelope, and the legacy inline
# pen's unaudited 0.03.
TOOL_R_LAT = 0.05
TOOL_R_INLINE = 0.03

# THE PAIRS THE GUARD WATCHES, and the one rule that produces them.
#
# Two capsules whose BODIES are fewer than four joints apart in the chain are
# not watched, and that is a statement about the mechanism rather than a
# convenience.  Bodies that near each other are held near each other BY the
# mechanism: consecutive links meet at their joint, link1/link2 and link5/link6
# share an origin, the hand is a fixed flange on link7, the holder is gripped
# by the fingers.  What keeps them out of each other is the FR3's own joint
# limits and the casting geometry the manufacturer designed around them, not
# anything a planner chooses — and a capsule model cannot say otherwise,
# because two capsules that meet at a joint overlap there in every
# configuration.  Measured on the shipped certified map, the three-joint rule
# refuses 17 807 of 23 376 strict-GO cells and the two-joint rule refuses all
# of them, for metal that `scripts/self_collision_audit.py --part truth`
# measures at 114.3 mm or more.  The four-joint rule refuses NONE of them, and
# the tightest certified pose in the whole map still holds 63.7 mm.
#
# What is left is the fold: the wrist, the hand and the pen coming back at the
# base, the shoulder and the upper arm — which is the self-collision an arm
# drawing on a table can actually commit, and the one a leaning pen makes
# reachable.  165 capsule pairs.
CHAIN_POS = dict(link0=0, link1=1, link2=2, link3=3, link4=4, link5=5,
                 link6=6, link7=7, hand=8, tool=8)
WATCH_CHAIN_D = 4        # joints apart, below which the mechanism owns the pair


def _pairs(d_min=WATCH_CHAIN_D):
    return tuple((i, j) for i in range(N_CAP) for j in range(i + 1, N_CAP)
                 if abs(CHAIN_POS[BODY_OF[j]] - CHAIN_POS[BODY_OF[i]]) >= d_min)


SELF_PAIRS = _pairs()
_PI = np.array([p[0] for p in SELF_PAIRS], int)
_PJ = np.array([p[1] for p in SELF_PAIRS], int)


def capsule_ends(qs, pen_ext=PEN_EXT, pen_lat=None):
    """(N,7) joints -> (A (N,C,3), B (N,C,3), R (C,)) in the arm's BASE frame.

    Self-collision is a question about one arm and nothing else, so this stays
    in link0 and never touches `spec.T_world_base`: a rigid motion of the whole
    arm cannot change the answer, and not asking for one keeps the gate usable
    from the places that have joints and no fleet — the atlas fiber, the
    planner's lattice, a hover scan.
    """
    lat = lat_of(pen_lat)
    T = link_frames_many(qs)
    n = len(T)
    A = np.empty((n, N_CAP, 3))
    B = np.empty((n, N_CAP, 3))
    R = np.empty(N_CAP)
    for k, (_, _, f, a, b, r) in enumerate(BODY_CAPSULES):
        Rf, tf = T[:, f, :3, :3], T[:, f, :3, 3]
        A[:, k] = Rf @ np.asarray(a, float) + tf
        B[:, k] = Rf @ np.asarray(b, float) + tf
        R[k] = r
    # THE TOOL IS THE PACKAGE'S OWN ENVELOPE, NOT THE CAD.  The mesh fit finds
    # a much tighter holder (three bands, 0.004-0.030, along the 45-degree
    # ray), but it finds it by placing a CAD delivery that arrived with no
    # assembly file, and `rig_final` is explicit that "the inference in that
    # placement is the residual risk, not the radius".  A safety gate does not
    # get to rest on an inference, so the tool here is the same L the package
    # already ships: TCP -> bracket corner -> tip, both at 0.05 and both
    # CAD-validated as CONTAINING the holder.  With the inline pen the corner
    # IS the TCP, so the bracket degenerates to a point inside the pen capsule
    # and changes no answer.
    Rf, tf = T[:, 9, :3, :3], T[:, 9, :3, 3]
    tcp = Rf @ np.array([0.0, 0.0, D_HAND_TCP]) + tf
    corner = Rf @ np.array([lat, 0.0, D_HAND_TCP]) + tf
    tip = Rf @ np.array([lat, 0.0, D_HAND_TCP + float(pen_ext)]) + tf
    A[:, BRACKET], B[:, BRACKET] = tcp, corner
    A[:, PEN], B[:, PEN] = corner, tip
    R[BRACKET] = R[PEN] = TOOL_R_LAT if lat != 0.0 else TOOL_R_INLINE
    return A, B, R


def _pt_seg(p, a, b):
    ab = b - a
    den = np.sum(ab * ab, -1)
    t = np.where(den > 1e-15,
                 np.sum((p - a) * ab, -1) / np.where(den > 1e-15, den, 1.0), 0.0)
    t = np.clip(t, 0.0, 1.0)
    d = p - (a + t[..., None] * ab)
    return np.sqrt(np.sum(d * d, -1))


def segment_distance(p0, p1, q0, q1):
    """Distance between two segments, broadcasting over any leading axes.

    The minimum of a convex quadratic over the unit square is attained at the
    interior stationary point when that exists and lies inside, and otherwise
    on the boundary, where it is one of the four point-to-segment distances.
    """
    d1, d2, r = p1 - p0, q1 - q0, p0 - q0
    a = np.sum(d1 * d1, -1)
    e = np.sum(d2 * d2, -1)
    b = np.sum(d1 * d2, -1)
    c = np.sum(d1 * r, -1)
    f = np.sum(d2 * r, -1)
    den = a * e - b * b
    inside = den > 1e-12
    s = np.where(inside, (b * f - c * e) / np.where(inside, den, 1.0), -1.0)
    t = np.where(inside, (a * f - b * c) / np.where(inside, den, 1.0), -1.0)
    good = inside & (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)
    w = r + s[..., None] * d1 - t[..., None] * d2
    interior = np.where(good, np.sqrt(np.maximum(np.sum(w * w, -1), 0.0)), np.inf)
    edge = np.minimum(np.minimum(_pt_seg(p0, q0, q1), _pt_seg(p1, q0, q1)),
                      np.minimum(_pt_seg(q0, p0, p1), _pt_seg(q1, p0, p1)))
    return np.minimum(interior, edge)


CHUNK = 2048          # configurations per broadcast; the (n, P, 3) temporaries
#   are what decides this, and P is ~300 pairs


def pair_clearance(qs, pen_ext=PEN_EXT, pen_lat=None, pairs=None):
    """Per-pair surface gap. (N,7) -> (N, len(pairs))."""
    qs = np.asarray(qs, float).reshape(-1, 7)
    I, J = (_PI, _PJ) if pairs is None else (
        np.array([p[0] for p in pairs], int), np.array([p[1] for p in pairs], int))
    out = np.empty((len(qs), len(I)))
    for s in range(0, len(qs), CHUNK):
        A, B, R = capsule_ends(qs[s:s + CHUNK], pen_ext, pen_lat)
        out[s:s + CHUNK] = segment_distance(A[:, I], B[:, I], A[:, J], B[:, J]) \
            - (R[I] + R[J])[None, :]
    return out


def sphere_bounds(A, B, R):
    """Each capsule's bounding sphere. -> (centres (N,C,3), radii (N,C)).

    THE SCREEN THAT MAKES THIS GATE AFFORDABLE ON A PATH.  A capsule is inside
    the ball on its own midpoint of radius `|B - A| / 2 + r`, so
    `|c_i - c_j| - Rad_i - Rad_j` is a LOWER BOUND on the surface gap between
    two of them: a pair that clears a floor by this bound clears it in the
    metal, and no segment arithmetic has to be done for it.

    It is worth having because the arm is mostly not near itself.  Measured
    over 792 configurations sampled along straight joint-space moves between
    certified cells of arm 31, the bound settles 99.876 % of the 165 watched
    pairs at a 23 mm floor, and 98.0 % of the configurations outright — the
    exact `segment_distance` then runs on the remaining 0.124 %.  End to end
    that is 12.6 us per configuration against 95.
    """
    C = 0.5 * (A + B)
    return C, 0.5 * np.linalg.norm(B - A, axis=-1) + R[None, :]


def min_clearance(A, B, R, floor=None):
    """Worst gap between two watched bodies, over EVERY configuration. -> float.

    Takes `capsule_ends`' output so a caller that already has it (a sampled
    path, say) pays for one forward-kinematics pass and not two.

    `floor` IS A CONTRACT, and it is `paper.leg_static_lb`'s: given one, the
    number that comes back is only guaranteed to be on the RIGHT SIDE of it.
    Every (configuration, pair) the sphere screen puts above the floor is left
    at its bound instead of being measured exactly, which is what makes the
    screen worth having; nothing that decides a gate is decided by a bound.
    """
    return float(clearance_screened(A, B, R, floor).min()) if len(A) else np.inf


def clearance_screened(A, B, R, floor=None):
    """`self_clearance` per configuration, off precomputed ends. -> (N,).

    Same `floor` contract as `min_clearance`, per row: a configuration whose
    sphere bound already clears the floor keeps that bound instead of being
    measured, so the number is a LOWER bound everywhere and exact wherever it
    matters to a gate at `floor`.
    """
    n = len(A)
    out = np.empty(n)
    for s in range(0, n, CHUNK):
        a, b = A[s:s + CHUNK], B[s:s + CHUNK]
        if floor is None:
            out[s:s + CHUNK] = (segment_distance(a[:, _PI], b[:, _PI],
                                                 a[:, _PJ], b[:, _PJ])
                                - (R[_PI] + R[_PJ])[None, :]).min(axis=1)
            continue
        C, Rad = sphere_bounds(a, b, R)
        g = np.linalg.norm(C[:, _PI] - C[:, _PJ], axis=-1) \
            - Rad[:, _PI] - Rad[:, _PJ]
        sel = g < float(floor)
        if sel.any():
            ns, ps = np.where(sel)
            i, j = _PI[ps], _PJ[ps]
            g[ns, ps] = segment_distance(a[ns, i], b[ns, i], a[ns, j], b[ns, j]) \
                - (R[i] + R[j])
        out[s:s + CHUNK] = g.min(axis=1)
    return out


def path_clearance_lb(qs, floor=None, pen_ext=PEN_EXT, pen_lat=None, k=0.0):
    """A LOWER BOUND on one arm's self-clearance ALONG a sampled path. -> (m, res).

    `qs` is (N,7) read as consecutive samples of one motion.  `m` is
    `min_clearance` at the samples; `res` is `k` times the worst distance any
    CAPSULE ENDPOINT travels between two of them, which is the 1-Lipschitz
    residual for exactly this quantity — `scene_check` charges the same
    coefficient against the chain points, and the capsule ends are what this
    gate is actually drawn about, so charging their own travel is the honest
    version of the same arithmetic and never the looser one.
    """
    A, B, R = capsule_ends(qs, pen_ext, pen_lat)
    res = 0.0
    if k and len(A) > 1:
        d = np.concatenate([np.diff(A, axis=0), np.diff(B, axis=0)], axis=1)
        res = float(k) * float(np.max(np.linalg.norm(d, axis=2)))
    lo = None if floor is None else float(floor) + res
    return min_clearance(A, B, R, lo), res


def self_clearance(qs, pen_ext=PEN_EXT, pen_lat=None):
    """Worst surface gap between two non-adjacent bodies of one arm. (N,7)->(N,)

    A LOWER BOUND on the gap in the metal, because every capsule contains its
    band's mesh.  Negative means the model says the arm is inside itself.
    """
    qs = np.asarray(qs, float).reshape(-1, 7)
    if not len(qs):
        return np.zeros(0)
    return pair_clearance(qs, pen_ext, pen_lat).min(axis=1)


def self_ok(qs, margin=SELF_MARGIN, pen_ext=PEN_EXT, pen_lat=None, eps=0.0):
    """Per-configuration pass/fail at `margin`. (N,7) -> (N,) bool."""
    return self_clearance(qs, pen_ext, pen_lat) >= float(margin) - float(eps)


def pair_reach(n=200000, seed=11, pen_ext=PEN_EXT, pen_lat=None,
               chunk=20000, d_min=1):
    """Largest clearance each candidate pair ever reaches. -> {pair: metres}.

    The instrument behind `WATCH_CHAIN_D`: a pair whose clearance never
    reaches `SELF_MARGIN` anywhere in the joint box cannot inform a gate at
    that margin, and every pair below four joints of separation is in that
    state or close to it.  `tests/test_selfcoll.py` runs it.
    """
    from .frames import FR3_MAX, FR3_MIN
    cand = _pairs(d_min)
    rng = np.random.default_rng(seed)
    hi = np.full(len(cand), -np.inf)
    for _ in range(0, n, chunk):
        Q = FR3_MIN + rng.random((chunk, 7)) * (FR3_MAX - FR3_MIN)
        hi = np.maximum(hi, pair_clearance(Q, pen_ext, pen_lat, cand).max(axis=0))
    return {cand[k]: float(hi[k]) for k in range(len(cand))}


def signature(pen_ext=PEN_EXT, pen_lat=None):
    """The whole self model as one flat array, for `atlas.model_signature`."""
    flat = [v for (_, band, f, a, b, r) in BODY_CAPSULES
            for v in (float(band), float(f), *a, *b, r)]
    lat = lat_of(pen_lat)
    flat += [lat, float(pen_ext), TOOL_R_LAT if lat != 0.0 else TOOL_R_INLINE]
    flat += [float(v) for p in SELF_PAIRS for v in p]
    flat.append(SELF_MARGIN)
    return np.asarray(flat, float)
