"""THE MOVING LINKS AS SPHERES — the accuracy Pete asked for, behind a flag.

Pete, 2026-09-09, looking at the :7005 scene: "for the rest of the robot why
not just use the standard collision geometry models that are shipped with the
frankas, or you can scavenge the collision geoms from the other repo where we
use a bunch of spheres?  that will be more accurate for the robots.  the static
orange cylinders for the pivoting bases are great though."

WHAT THIS REPLACES, AND WHAT IT DOES NOT.  `coordination.CAPSULES` is two
different models bolted together.  Its first four entries are `BASE_CAPSULES` —
the measured body column, four bands of a solid of revolution about the base
axis — and those are the orange cylinders, they are EXACT for what they model
(`envelope.py`), and they do not move.  Its last six are the arm: five sausages
drawn about the lines between JOINT ORIGINS (shoulder->elbow at 0.130,
elbow at 0.117, forearm at 0.131, wrist at 0.091, hand at 0.104), plus the
tool.  Those five are what this module replaces.  The base column and the tool
capsules are untouched.

WHY THOSE FIVE ARE THE PROBLEM.  A capsule about a line between two joint
origins has to be as wide as the farthest metal reaches from that line, and an
FR3's castings are L-shaped: the line runs through the inside of the bend and
the radius is set by the outside of it.  Measured against the manufacturer's
own meshes over six random configurations, the five moving capsules enclose
58.0 litres where the metal's own convex hulls are 16.9, and the mean point of
real metal sits 55.7 mm inside the model with the worst at 130.6 mm.  That
130 mm of fictitious steel is charged to EVERY arm-to-arm clearance twice over,
against a `PAIR_MARGIN` of 50 mm.

WHAT THE SHIPPED SPHERE SETS ACTUALLY ARE, MEASURED (2026-09-09).  Pete's
"scavenge the spheres from the other repo" was tried first and it does not
work, because none of the three sphere models on this machine CONTAINS the
robot.  Escape is the largest distance any mesh point of a body lies OUTSIDE
the spheres attached to that body; the meshes are the manufacturer's collision
shells UNION the full-resolution visuals, the same ground truth
`scripts/self_collision_audit.py` fits against:

    model                                          spheres   worst escape
    assets/franka_description/urdf/                   66      +235.2 mm (link0)
        panda_arm_hand.urdf  (Drake's stock Panda)             +66.5 (link5)
    ~/git/vamp/resources/panda/panda_spherized.urdf   59      +225.7 mm (link0)
                                                              +62.1 (link5)
    ~/git/mmt_gcs/assets/                             35      +146.9 mm (link0)
        fr3_franka_hand_sphere_collisions.urdf                 +56.8 (link5)

Every one of them is OPTIMISTIC on every body, by 4 mm at best and 235 mm at
worst.  They are planner models, tuned so a planner is fast and its paths look
right; they were never envelopes and nothing in them claims to be.  Dropping
one into a gate that certifies a programme would put a hole in the certificate
exactly where the fingers, the wrist bulge and the base connector are.
(`~/git/cc_experiment` has no Franka sphere set at all — it benchmarks an
iiwa14 and computes bounding spheres at runtime; `/home/franka/aris_project/
reachability` has none either.  Both were checked.)

SO THE SPHERES ARE FITTED HERE, AND CONTAINMENT IS BY CONSTRUCTION.  Eight
spheres per moving body — the same budget Drake's stock Panda spends, so the
comparison is like for like — placed by farthest-point seeding, Lloyd descent
and a Badoiu-Clarkson minimum-enclosing-ball polish on the body's own point
cloud, and then each radius set to the EXACT maximum distance from its centre
to any point assigned to it, rounded UP to the millimetre.  Every point of the
cloud is inside its own sphere before rounding and further inside after, so the
union contains the cloud with nothing to prove; `tests/test_link_spheres.py`
re-measures it anyway and the worst body still clears by 0.21 mm.

WHAT IT BUYS, AND IT IS A NUMBER.  Same six configurations, same meshes:

    the five moving capsules      58.02 L    mean offset 55.7 mm   max 130.6 mm
    these 64 spheres              32.81 L    mean offset 24.7 mm   max  75.2 mm
                                  -43.5 %                -31.0 mm       -55.4 mm

"Offset" is how far a point of real metal lies inside the model — the fictitious
steel the gate is charged for.  Halving it on both arms of a pair is why the
flag exists.

AND WHAT IT DOES NOT BUY: THE SELF MODEL.  `selfcoll.BODY_CAPSULES` is already
a per-link, per-link-FRAME, mesh-fitted table (three bands a body, radii
0.038-0.078), not a chain model, and it does not have the L-shape problem.
Measured the same way, 64 spheres are 38.5 L against its 43.6 — 12 % smaller
overall, but LARGER on link1 (7.11 vs 6.31), link2 (7.08 vs 6.12) and link5
(8.27 vs 7.33), which are three of the four bodies the fold gate is FOR.  So
`selfcoll` keeps its capsules under this flag and the module says so out loud
rather than switching a safety gate for a 12 % that is negative where it
matters.  See docs/DECISIONS.md, 2026-09-09.

HOW IT IS SWITCHED.  `ARIS_COLLISION_MODEL=spheres` in the environment, or
`install()` / `uninstall()` at runtime — the same shape `envelope.install` and
`frozen.freeze` use, and read at the same point in `feasible_workspace.py`.
DEFAULT IS OFF.  Turning it on changes what every certified number in this repo
was earned against, so flipping the default is a re-certification, not a commit
(docs/DECISIONS.md says what that costs).

A SPHERE IS A CAPSULE WITH A ZERO-LENGTH SEGMENT, and that is the whole of the
wiring.  Every funnel in this package — `coordination.ArmPath`'s boxes and
tiles and exact distances, `frozen`'s partner capsules, `paper`'s 1-Lipschitz
residual on capsule ENDPOINT travel, `scene_check`'s own segment arithmetic —
takes `(A, B, R)` and never asks whether `A == B`.  So the sphere block is
handed over as degenerate capsules and nothing downstream changes.  The
Lipschitz bound is then exactly right rather than merely valid: a sphere's
centre IS the per-body point whose motion bounds the body's, so charging its
travel is charging the true bound.
"""
import os

import numpy as np

from .frames import link_frames_many

# (body, LINK_FRAMES index, centre in that frame, radius).  Frames are
# `frames.LINK_FRAMES`: 0..7 are link0..link7 and 9 is the hand.  FITTED by
# `scripts/link_sphere_fit.py`; the comment on each row is the exact cloud
# maximum the shipped radius rounds UP from.  link0 is NOT here — it is the
# base column, and the base column stays cylinders (see the module docstring).
SPHERES = (
    ("link1", 1, (+0.0213, -0.0438, -0.0554), 0.075),   # mesh 0.0742
    ("link1", 1, (+0.0024, -0.0414, -0.1589), 0.074),   # mesh 0.0738
    ("link1", 1, (-0.0259, -0.0399, -0.0486), 0.074),   # mesh 0.0733
    ("link1", 1, (+0.0293, +0.0145, -0.1047), 0.073),   # mesh 0.0722
    ("link1", 1, (+0.0005, -0.0282, +0.0068), 0.072),   # mesh 0.0711
    ("link1", 1, (-0.0415, +0.0066, -0.1268), 0.071),   # mesh 0.0704
    ("link1", 1, (-0.0004, -0.0819, -0.0048), 0.069),   # mesh 0.0683
    ("link1", 1, (+0.0008, +0.0150, -0.1751), 0.066),   # mesh 0.0652
    ("link2", 2, (+0.0200, +0.0130, +0.0278), 0.075),   # mesh 0.0745
    ("link2", 2, (+0.0138, -0.0682, +0.0466), 0.075),   # mesh 0.0743
    ("link2", 2, (-0.0131, -0.0749, +0.0158), 0.075),   # mesh 0.0742
    ("link2", 2, (+0.0411, -0.1391, -0.0142), 0.071),   # mesh 0.0705
    ("link2", 2, (-0.0086, -0.0045, +0.0846), 0.070),   # mesh 0.0696
    ("link2", 2, (-0.0056, -0.1580, +0.0453), 0.069),   # mesh 0.0682
    ("link2", 2, (-0.0199, -0.1599, -0.0222), 0.066),   # mesh 0.0656
    ("link2", 2, (-0.0306, +0.0002, +0.0372), 0.066),   # mesh 0.0651
    ("link3", 3, (-0.0066, +0.0338, -0.0957), 0.067),   # mesh 0.0662
    ("link3", 3, (+0.0455, +0.0593, -0.0270), 0.066),   # mesh 0.0651
    ("link3", 3, (+0.0697, +0.0004, -0.0709), 0.066),   # mesh 0.0657
    ("link3", 3, (-0.0037, -0.0149, -0.0955), 0.062),   # mesh 0.0615
    ("link3", 3, (+0.0835, +0.0199, +0.0028), 0.062),   # mesh 0.0612
    ("link3", 3, (-0.0003, +0.0002, -0.0225), 0.062),   # mesh 0.0613
    ("link3", 3, (+0.0823, +0.0687, +0.0211), 0.060),   # mesh 0.0592
    ("link3", 3, (+0.1003, +0.0612, -0.0254), 0.060),   # mesh 0.0595
    ("link4", 4, (-0.0901, +0.0974, +0.0229), 0.067),   # mesh 0.0666
    ("link4", 4, (-0.0154, +0.0666, +0.0037), 0.065),   # mesh 0.0644
    ("link4", 4, (-0.0389, +0.0361, +0.0618), 0.065),   # mesh 0.0642
    ("link4", 4, (-0.0070, -0.0227, +0.0682), 0.062),   # mesh 0.0616
    ("link4", 4, (-0.0847, +0.0988, -0.0162), 0.062),   # mesh 0.0612
    ("link4", 4, (+0.0040, -0.0081, +0.0160), 0.062),   # mesh 0.0617
    ("link4", 4, (-0.0864, +0.0206, +0.0013), 0.061),   # mesh 0.0610
    ("link4", 4, (+0.0189, +0.0202, +0.0613), 0.059),   # mesh 0.0587
    ("link5", 5, (-0.0055, +0.0646, -0.0772), 0.080),   # mesh 0.0797
    ("link5", 5, (-0.0018, -0.0087, -0.1521), 0.080),   # mesh 0.0797
    ("link5", 5, (+0.0074, +0.0206, -0.0225), 0.080),   # mesh 0.0793
    ("link5", 5, (+0.0003, +0.0537, -0.1720), 0.074),   # mesh 0.0731
    ("link5", 5, (-0.0323, +0.0577, -0.0064), 0.072),   # mesh 0.0714
    ("link5", 5, (+0.0097, +0.0876, +0.0024), 0.071),   # mesh 0.0700
    ("link5", 5, (+0.0181, +0.0005, -0.2383), 0.067),   # mesh 0.0664
    ("link5", 5, (-0.0198, +0.0002, -0.2332), 0.066),   # mesh 0.0658
    ("link6", 6, (+0.0991, -0.0210, +0.0226), 0.054),   # mesh 0.0532
    ("link6", 6, (+0.0969, -0.0145, -0.0221), 0.053),   # mesh 0.0524
    ("link6", 6, (+0.0407, -0.0148, -0.0058), 0.053),   # mesh 0.0528
    ("link6", 6, (+0.0719, +0.0485, +0.0022), 0.051),   # mesh 0.0501
    ("link6", 6, (+0.0419, +0.0006, +0.0423), 0.051),   # mesh 0.0507
    ("link6", 6, (-0.0160, -0.0089, +0.0121), 0.051),   # mesh 0.0507
    ("link6", 6, (+0.0039, +0.0241, +0.0068), 0.050),   # mesh 0.0498
    ("link6", 6, (+0.0973, +0.0412, -0.0032), 0.050),   # mesh 0.0497
    ("link7", 7, (-0.0051, -0.0296, +0.0938), 0.037),   # mesh 0.0368
    ("link7", 7, (+0.0017, +0.0290, +0.0568), 0.037),   # mesh 0.0362
    ("link7", 7, (+0.0043, -0.0274, +0.0594), 0.037),   # mesh 0.0363
    ("link7", 7, (-0.0346, -0.0012, +0.0743), 0.037),   # mesh 0.0365
    ("link7", 7, (-0.0012, +0.0283, +0.0968), 0.036),   # mesh 0.0355
    ("link7", 7, (+0.0340, -0.0006, +0.0787), 0.036),   # mesh 0.0358
    ("link7", 7, (+0.0532, +0.0369, +0.0832), 0.035),   # mesh 0.0345
    ("link7", 7, (+0.0315, +0.0571, +0.0831), 0.032),   # mesh 0.0318
    ("hand", 9, (-0.0004, -0.0377, +0.0628), 0.051),   # mesh 0.0504
    ("hand", 9, (+0.0005, +0.0439, +0.0688), 0.047),   # mesh 0.0464
    ("hand", 9, (-0.0019, -0.0069, +0.0060), 0.046),   # mesh 0.0458
    ("hand", 9, (-0.0000, -0.0682, -0.0012), 0.046),   # mesh 0.0453
    ("hand", 9, (+0.0005, +0.0102, +0.0427), 0.044),   # mesh 0.0430
    ("hand", 9, (-0.0013, +0.0554, +0.0017), 0.044),   # mesh 0.0439
    ("hand", 9, (+0.0005, +0.0838, +0.0321), 0.042),   # mesh 0.0412
    ("hand", 9, (+0.0008, -0.0778, +0.0412), 0.042),   # mesh 0.0420
)
N_SPHERE = len(SPHERES)
BODY_OF = tuple(s[0] for s in SPHERES)
_FRAME = np.array([s[1] for s in SPHERES], int)
_CENTRE = np.array([s[2] for s in SPHERES], float)
RADII = np.array([s[3] for s in SPHERES], float)

# WHICH `coordination.CAPSULES` ENTRIES THESE STAND IN FOR.  The five moving
# arm capsules, by their chain endpoints — everything that is neither the base
# column (chain point 0) nor the tool (chain point 8 onwards).  Named by their
# endpoints rather than by position so a table edit cannot silently re-point
# this at the wrong rows; `tests/test_link_spheres.py` pins the set.
REPLACES = ((1, 3), (3, 4), (4, 5), (5, 7), (7, 8))

ENV_VAR = "ARIS_COLLISION_MODEL"
_ON = None          # None: read the environment.  True/False: installed.


def enabled():
    """Is the sphere model the active one? -> bool.  Default False."""
    if _ON is not None:
        return _ON
    return os.environ.get(ENV_VAR, "capsules").strip().lower() in (
        "sphere", "spheres")


def install():
    """Make the sphere model active for this process."""
    global _ON
    _ON = True


def uninstall():
    """Back to the shipped capsule model."""
    global _ON
    _ON = False


def reset():
    """Forget the runtime override; go back to reading the environment."""
    global _ON
    _ON = None


def centres_base(qs):
    """(N,7) joints -> sphere centres in the arm's BASE frame. (N,S,3).

    The arm against itself, or an arm whose fleet placement nobody has yet
    asked for.  A rigid motion of the whole arm cannot change a self answer.
    """
    qs = np.asarray(qs, float).reshape(-1, 7)
    Tf = link_frames_many(qs)[:, _FRAME]           # (N, S, 4, 4)
    return np.einsum("nsij,sj->nsi", Tf[..., :3, :3], _CENTRE) + Tf[..., :3, 3]


def centres_world(qs, Twb):
    """(N,7) joints + the arm's base transform -> centres in WORLD. (N,S,3)."""
    C = centres_base(qs)
    Twb = np.asarray(Twb, float)
    return C @ Twb[:3, :3].T + Twb[:3, 3]


def as_capsules(C):
    """Centres (N,S,3) -> (A, B, R) with A == B: spheres as degenerate capsules.

    Nothing downstream of this package's collision funnels asks whether a
    capsule has length, so this is the entire adapter (see the docstring).
    """
    return C, C, RADII.copy()


def moving_capsules(caps):
    """`caps` minus the rows these spheres stand in for. -> (tab, idx).

    `idx` is the surviving columns, so a caller holding `cap_endpoints` output
    can slice it instead of recomputing — the same shape and the same contract
    as `coordination.known_pose_capsules`.
    """
    keep = [k for k, c in enumerate(caps) if (c[0], c[1]) not in REPLACES]
    return tuple(caps[k] for k in keep), keep


def signature():
    """The whole sphere model as one flat array, for `atlas.model_signature`."""
    return np.concatenate([
        _FRAME.astype(float), _CENTRE.reshape(-1), RADII,
        np.array([float(v) for p in REPLACES for v in p])])
