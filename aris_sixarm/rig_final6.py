"""THE REAL INSTALLATION: two FINAL-RIG units, mirrored back-to-back.

`rig_final.py` holds ONE extracted 3-arm unit (docs/FINAL_RIG.md).  The user
states the real installation is **two of those units, mirrored across the side
where the hanging arms live**, so that:

    floor 13 ........ [ web A ] .. 31 2 | 71 97 .. [ web B ] ........ floor 17
                                        ^ mirror plane

the two floor arms end up as individuals at the two OUTER ends, and the four
hanging arms (2 inverted + 2 side-mounted) form a cluster in the middle band.

This module NEVER modifies `rig_final`: it reads that geometry and reflects it.

--------------------------------------------------------------------------
THE MIRROR PLANE  (task 1 — derived, not assumed)
--------------------------------------------------------------------------
The hanging arms (31 inverted, 2 side) both hang off the central double top
beam at W-frame Y = 152.45 cm, i.e. in the BACK half of the 208.28 cm depth
(mid-depth is 104.14).  The floor arm 13 sits at Y = 14.07 cm, at the FRONT.
So "the side where the hanging arms live" is the BACK face of the frame, and
the two units abut back-to-back there:

    MIRROR_PLANE_W_CM      = 208.28  cm   (W frame, Y)
    MIRROR_PLANE_CANVAS_Y  = 1.815320 m   (canvas frame, y)

Provenance of 208.28:
  * PDF dimension annotation "208,3" labelled *entire table*, measured in the
    DXF as exactly 208.280 (docs/FINAL_RIG.md, dimension inventory).
  * the installation envelope quoted from the same extraction: Y 0 -> 208.28.
  * the structural boxes that actually SIT on that face all stop there:
    post_BL / post_BR hi-Y = 208.28, top_slab hi-Y = 208.28
    (rig_final.FRAME_BOXES_W_CM).  `outer_face_provenance()` re-derives it
    from those boxes at import-check time rather than trusting this comment.
  * canvas value = (208.28 - PAPER_ORIGIN_W_CM.y) / 100
                 = (208.28 - 26.748) / 100 = 1.815320 m.

ASSUMPTION — FLAGGED, NEEDS USER CONFIRMATION
  The two frames are assumed to **abut exactly, zero gap**, outer face on
  outer face.  Two things argue with that and are flagged rather than
  silently modelled:
    (a) the levelling-foot pads overhang the leg lines by 0.33 cm
        (rig_final "table_block" note), so two frames touching at the PADS
        would hold their structural faces 2 x 0.33 = 0.66 cm apart;
    (b) three unit-A collision boxes already poke PAST the plane by design
        conservatism (table_block +0.40, device_B1/B2 +1.00 cm) — see
        `boxes_crossing_the_plane()`.  Their mirrors poke back.  Neither is
        real steel, and both sit below the paper or above z = 1.53 m, so
        nothing reachable is affected — but a real gap G would shift every
        unit-B number by G, so it must be confirmed before anything is
        re-derived deeply.
  `GAP_CM` is the single knob: set it to the surveyed gap and every unit-B
  pose, box and atlas mirror moves consistently.

--------------------------------------------------------------------------
UNIT B ARM IDS — FLAGGED, NEEDS USER CONFIRMATION
--------------------------------------------------------------------------
The drawing names only unit A's arms ("13 floor, 31 left - upside down,
2 right side position").  Unit B is given the OTHER three physical arms from
the legacy six-arm registry (fleet.FLEET_SIXARM): **17** (the other floor
arm), **71** and **97** (the other two hanging arms), keeping their legacy
identity colours.  This is an ASSUMPTION about which hardware goes where, not
a fact from any drawing.

--------------------------------------------------------------------------
MIRRORING HARDWARE THAT CANNOT BE MIRRORED
--------------------------------------------------------------------------
A reflection is improper (det = -1); no FR3 can be built left-handed.  What
IS available is the CONJUGATION  R' = S R S  (S = diag(1,-1,1)), which has
det = +1 and is therefore a real, buildable base orientation.  Per mount it
lands exactly on the physically natural realisation:

  floor 13  rotz(+pi/2), front +Y  ->  17  rotz(-pi/2), front -Y
            (the same upright arm at the far end, turned to face its own web)
  inv   31  rotx(pi), front +X     ->  71  rotx(pi), front +X
            (yaw mirrored; at yaw 0 the mirror is the identity — it still
             hangs from the beam and still faces +X across its own web)
  side  2   roty(-pi/2) rotz(pi)   ->  97  roty(-pi/2) rotz(pi)
            (J1 axis stays horizontal along -X, front still straight down;
             the mount reflects to unit B's boom, pointing into ITS half)

and in JOINT space the mirror is exact and limit-preserving:

    MIRROR_Q_SIGN = (-1, +1, -1, +1, -1, +1, -1)

    fk(q * sign).pts  ==  S @ fk(q).pts                 (verified to 3e-16)
    tip_pos(q * sign) ==  S @ tip_pos(q)                (4e-16)
    joint_margin, sigma_min, f_max are IDENTICAL                (exact / 1e-11)

because exactly the four sign-flipped joints (1,3,5,7) are the ones whose FR3
limits are antisymmetric (+-2.7437, +-2.9007, +-2.8065, +-3.0159), while the
asymmetric ones (q4 in [-3.0421,-0.1518], q6 in [0.5445,4.5169]) are the ones
left alone.  That is what makes the coverage-by-symmetry preview in
scripts/make_atlas6_preview.py EXACT rather than approximate.

(The TCP *orientation* picks up a fixed 90 deg tool yaw,
 R_tcp(q*sign) = S R_tcp(q) S @ rotz(-pi/2), from fk's Rz(-pi/4) flange
 twist.  It changes nothing scored here: the pen axis is the tool z, which is
 unaffected, and the atlas's 8-yaw candidate set is closed under
 psi -> pi/2 - psi.)
"""
import numpy as np

from . import mounts
from . import rig_final
from .fleet import ArmSpec
from .frames import (Q_READY_FLOOR, Q_READY_INV_FINAL, Q_READY_WALL,
                     Q_READY_WALL_LOW)

# --- the plane -----------------------------------------------------------
MIRROR_PLANE_W_CM = 208.28          # W frame, Y: outer BACK face of the frame
GAP_CM = 0.0                        # ASSUMED zero gap between the two frames
MIRROR_PLANE_CANVAS_Y = float(
    (MIRROR_PLANE_W_CM + GAP_CM / 2.0 - rig_final.PAPER_ORIGIN_W_CM[1]) / 100.0)

S_MIRROR = np.diag([1.0, -1.0, 1.0])            # the improper reflection
MIRROR_Q_SIGN = np.array([-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])

# --- the combined drawing surface ----------------------------------------
# Canvas origin stays unit A's web corner, so every existing unit-A number is
# untouched.  Unit B's web is the reflection of unit A's.
MERGE_WEBS = True    # CONFIG FLAG — DECIDED BY THE USER 2026-08-21.
# True (the decision): ONE continuous drawable surface, y 0 -> 3.63064,
#   spanning the two units AND the 23.064 cm strip between their webs.  The
#   seam band is no longer unscored — `scripts/run_atlas6.py` sweeps the whole
#   1.8034 x 3.63064 m canvas cell by cell, seam included, and
#   `out/atlas_final6_opt/coverage.npz` is that sweep.  What the flag still
#   ASSUMES is physical: that a single web is actually fed across both tables
#   and is FLAT and drawable over the seam.  The frames' own back faces are
#   11.532 cm inboard of each paper edge, so the strip is over structure, not
#   over a hole — but it is over the two frames' top rails, not over a
#   tabletop, and nothing in the drawing shows a bridging surface there.
# False: the as-drawn reading — TWO webs, unit A y 0 -> 1.700, unit B
#   y 1.93064 -> 3.63064, and a SEAM_M = 0.23064 m gap between them (the two
#   11.532 cm back margins of paper-edge-to-frame-face, back to back).
WEB_A = ((0.0, 0.0), rig_final.SHEET_FINAL)                  # (origin, size)
_WEB_B_Y0 = 2 * MIRROR_PLANE_CANVAS_Y - rig_final.SHEET_FINAL[1]
WEB_B = ((0.0, _WEB_B_Y0), rig_final.SHEET_FINAL)
SEAM_M = float(_WEB_B_Y0 - rig_final.SHEET_FINAL[1])         # 0.23064 m
SHEET_FINAL6 = (rig_final.SHEET_FINAL[0],
                float(2 * MIRROR_PLANE_CANVAS_Y))            # bounding extent


def webs():
    """-> [((x0,y0),(w,h)), ...] the drawable surface(s), canvas m."""
    if MERGE_WEBS:
        return [((0.0, 0.0), SHEET_FINAL6)]
    return [WEB_A, WEB_B]


# --- the mirror ----------------------------------------------------------
def mirror_point(p):
    """Canvas point(s) (...,3) -> reflected across the plane."""
    p = np.asarray(p, float)
    out = p.copy()
    out[..., 1] = 2 * MIRROR_PLANE_CANVAS_Y - p[..., 1]
    return out


def mirror_y(y):
    """Scalar/array canvas y -> reflected y."""
    return 2 * MIRROR_PLANE_CANVAS_Y - np.asarray(y, float)


def mirror_rotation(R):
    """Base rotation -> the PROPER realisation on the other side, S R S.

    A true reflection S@R is improper (det -1) and unbuildable; conjugating
    keeps det = +1 while sending every axis to its mirrored direction up to
    the body-y sign, which is the handedness the real hardware supplies.
    """
    R = np.asarray(R, float)
    return S_MIRROR @ R @ S_MIRROR


def mirror_q(q):
    """Joint vector -> the configuration whose chain is the exact mirror."""
    return np.asarray(q, float) * MIRROR_Q_SIGN


def mirror_box(lo, hi):
    """AABB (canvas) -> reflected AABB (y bounds swap and reflect)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    nlo = np.array([lo[0], 2 * MIRROR_PLANE_CANVAS_Y - hi[1], lo[2]])
    nhi = np.array([hi[0], 2 * MIRROR_PLANE_CANVAS_Y - lo[1], hi[2]])
    return nlo, nhi


def mirror_T(T):
    """4x4 canvas pose -> mirrored pose (proper rotation, reflected origin)."""
    T = np.asarray(T, float)
    out = np.eye(4)
    out[:3, :3] = mirror_rotation(T[:3, :3])
    out[:3, 3] = mirror_point(T[:3, 3])
    return out


# --- the extended-pole variant -------------------------------------------
# docs/ARM2_HEIGHT.md, section 4: the side arm's ONE free mounting parameter is
# how far down its 2 x 3" pole the plate is clamped, and the study's answer is
# "not by sliding — the plate is already at the bottom stop; LENGTHEN THE POLE
# 20 cm and re-clamp at canvas z 0.576".  Adopted by the user 2026-08-21 for
# BOTH side arms (2 in unit A, 97 in unit B by mirror symmetry).
#
# WHAT MOVES, AND WHAT IS NEW STEEL
#   `side_plate`, `side_clamps`, `side_bracket` are bolted to the arm and slide
#   with it — the same three boxes `scripts/arm2_height_sweep.py` slides.
#   `side_boom` (the pole) is NOT slid: it is EXTENDED downward, because the
#   whole point is that there is no pole down there today.  That is the one
#   thing the sweep did not model (it kept the drawn 71/73.66 cm pole and let
#   the bracket hang past its end, which is exactly what cannot be built), so
#   the numbers here are NOT the sweep's numbers re-quoted: the lengthened pole
#   is 20 cm of extra obstacle for every OTHER arm and has to be re-swept.
#   `side_gusset` is the pole's TOP brace and stays where it is.
SIDE_DZ_OPT_CM = -20.0        # slide of the clamped stack, cm (negative = down)
POLE_EXT_OPT_CM = 20.0        # extra profile welded on the pole's BOTTOM end
SIDE_SLIDING = ("side_plate", "side_clamps", "side_bracket")
SIDE_POLE = "side_boom"


def boxes_with_extended_pole(dz_cm=SIDE_DZ_OPT_CM, pole_ext_cm=POLE_EXT_OPT_CM):
    """`rig_final.FRAME_BOXES_W_CM` for the extended-pole build. -> list.

    Same 35 boxes, same W frame, same conservatism; two edits:
      * the three CLAMPED boxes slide `dz_cm` along z with the arm;
      * `side_boom`'s bottom drops `pole_ext_cm`, so the pole still carries the
        bracket (which now sits at z_W 135.07-140.83) instead of ending above
        it.  20 cm is the doc's recommendation and 17.3 cm the minimum; the
        volume it grows into (X 187.3-195.0, Y 144.8-160.1, z 63.7-155.1) was
        swept in the DXF and is empty all the way to the tabletop.
    `rig_final.py` is not modified: it is the drawing, and this is a build.
    """
    out = []
    for b in rig_final.FRAME_BOXES_W_CM:
        if b["name"] in SIDE_SLIDING:
            b = dict(b, lo=(b["lo"][0], b["lo"][1], b["lo"][2] + dz_cm),
                     hi=(b["hi"][0], b["hi"][1], b["hi"][2] + dz_cm),
                     source=b["source"] + f" [re-clamped {dz_cm:+.1f} cm]")
        elif b["name"] == SIDE_POLE:
            b = dict(b, lo=(b["lo"][0], b["lo"][1], b["lo"][2] - pole_ext_cm),
                     source=b["source"] + f" [pole extended {pole_ext_cm:.1f} cm "
                                          "DOWNWARD - new steel, docs/ARM2_HEIGHT.md]")
        out.append(b)
    return out


FRAME_BOXES6_OPT_W_CM = boxes_with_extended_pole()


# --- the combined structure ----------------------------------------------
def frame_boxes6_canvas(exclude_tag=None, zmin=0.0, boxes_w=None):
    """Both units' structure in the CANVAS frame, m.

    Unit A's boxes verbatim from `rig_final.frame_boxes_canvas`; unit B's are
    their exact reflections.  Names are suffixed "@A" / "@B" and each box
    carries `unit`.  `exclude_tag` follows rig_final's convention but is
    matched against the UNIT-QUALIFIED tag ("mount:down@B"), so an arm is
    excused from its own mount hardware and nobody else's — including its
    mirror twin's.

    `boxes_w` selects the BUILD: None is the drawing (`rig_final`'s own list),
    `FRAME_BOXES6_OPT_W_CM` the extended-pole build.  Both units are always
    built from the SAME list, because the two frames are the same frame.
    """
    out = []
    for b in rig_final.frame_boxes_canvas(zmin=zmin, boxes=boxes_w):
        a = dict(b)
        a["name"] = b["name"] + "@A"
        a["tag"] = (b["tag"] + "@A") if b["tag"] else ""
        a["unit"] = "A"
        lo, hi = mirror_box(b["lo"], b["hi"])
        m = dict(name=b["name"] + "@B", lo=lo, hi=hi, source=b["source"],
                 tag=(b["tag"] + "@B") if b["tag"] else "", unit="B")
        out.extend([a, m])
    if exclude_tag is not None:
        out = [b for b in out if b["tag"] != exclude_tag]
    return out


def boxes_crossing_the_plane(tol=0.0):
    """Unit-A boxes that extend past the mirror plane -> [(name, overshoot_m)].

    Evidence for the abutment flag: these are the only places where the two
    units' CONSERVATIVE collision models overlap each other.  All of them are
    padding, not steel.
    """
    out = []
    for b in rig_final.frame_boxes_canvas(zmin=-10):
        over = float(b["hi"][1] - MIRROR_PLANE_CANVAS_Y)
        if over > tol:
            out.append((b["name"], over, float(b["lo"][2]), float(b["hi"][2])))
    return sorted(out, key=lambda t: -t[1])


def outer_face_provenance():
    """Re-derive the plane from the frame geometry instead of trusting a
    constant -> (y_w_cm, [box names that define that face]).

    The outer face on the hanging-arm side = the largest hi-Y shared by the
    STRUCTURAL members that stand on it (corner posts + top slab).  The
    padded/conservative boxes (table_block, device_B*) are excluded by name
    because their source strings say the padding is deliberate.
    """
    structural = [b for b in rig_final.FRAME_BOXES_W_CM
                  if b["name"].startswith("post_")
                  or b["name"] == "top_slab"]
    y = max(b["hi"][1] for b in structural)
    return y, sorted(b["name"] for b in structural
                     if abs(b["hi"][1] - y) < 1e-9)


def which_side_do_the_hanging_arms_live_on():
    """-> ("back"|"front", mean hanging-arm Y in cm, mid-depth Y in cm).

    The derivation the plane choice rests on: both hanging mounts sit in the
    back half of the depth, the floor mount in the front half.
    """
    ys = [rig_final.ARM_MOUNTS_W[k]["p_w_cm"][1] for k in ("down", "side")]
    mid = MIRROR_PLANE_W_CM / 2.0
    m = float(np.mean(ys))
    return ("back" if m > mid else "front"), m, mid


# --- the six-arm fleet ---------------------------------------------------
# unit B ids: the legacy registry's OTHER three physical arms (ASSUMED).
UNIT_B_IDS = {"up": 17, "down": 71, "side": 97}
UNIT_B_COLORS = {17: (0.09, 0.75, 0.81),      # cyan   (legacy "back")
                 71: (1.00, 0.50, 0.05),      # orange (legacy "L-inv-back")
                 97: (0.58, 0.40, 0.74)}      # purple (legacy "R-inv-back")
UNIT_B_NAMES = {17: "up-back", 71: "down-left-B", 97: "side-right-B"}


class Arm6Spec(ArmSpec):
    """An `ArmSpec` that knows which unit it belongs to.

    Only `static_obstacles` differs: on the combined rig an arm must clear
    BOTH frames, minus its own mount hardware.  (`fleet.ArmSpec` maps
    arm_id -> mount key through the 3-arm table, which has no entry for
    17/71/97 — this override replaces that lookup with the stored key and is
    the whole reason for the subclass.  `fleet.py` is untouched.)
    """
    __slots__ = ()

    @property
    def unit(self):
        return getattr(self, "_unit", "A")

    @property
    def boxes_w(self):
        """The W-frame box list this arm's BUILD stands in (None = as drawn)."""
        return getattr(self, "_boxes_w", None)

    def static_obstacles(self):
        return frame_boxes6_canvas(
            exclude_tag=f"mount:{self._key}@{self._unit}",
            boxes_w=getattr(self, "_boxes_w", None)) + list(self.column_boxes)


UNIT_A_IDS = {"up": 13, "down": 31, "side": 2}
UNIT_A_NAMES = {13: "up-front", 31: "down-left", 2: "side-right"}
UNIT_A_COLORS = {13: (0.12, 0.47, 0.71),      # blue
                 31: (0.84, 0.15, 0.16),      # red
                 2: (0.17, 0.63, 0.17)}       # green


def _spec(key, unit, arm_id, name, color, dz_cm=0.0, boxes_w=None):
    p, R = rig_final.arm_base_canvas(key)
    ready = {"up": Q_READY_FLOOR, "down": Q_READY_INV_FINAL,
             "side": Q_READY_WALL}[key]
    if key == "side" and dz_cm:
        p = p + np.array([0.0, 0.0, dz_cm / 100.0])   # down the (longer) pole
        # THE READY POSE MOVES WITH THE BASE OR IT IS NOT A READY POSE.  The
        # drawn one hovers 0.10 m over the paper from z 0.776; from z 0.576 the
        # same joints put the pen 0.100 m UNDER it.  See frames.Q_READY_WALL_LOW.
        ready = Q_READY_WALL_LOW
    if unit == "B":
        p, R, ready = mirror_point(p), mirror_rotation(R), mirror_q(ready)
    mount = {"up": "floor", "down": "inv", "side": "wall"}[key]
    s = Arm6Spec(arm_id, name, mount, (float(p[0]), float(p[1])), 0.0, True,
                 color, rig="final", z=float(p[2]),
                 R=tuple(np.asarray(R).flatten()), q_ready=tuple(ready))
    object.__setattr__(s, "_key", key)      # frozen dataclass
    object.__setattr__(s, "_unit", unit)
    object.__setattr__(s, "_boxes_w", boxes_w)
    return s


def _build_fleet6(dz_cm=0.0, boxes_w=None):
    """The six specs of one BUILD, in unit-A-then-unit-B registry order."""
    out = {}
    for unit, ids, names, cols in (("A", UNIT_A_IDS, UNIT_A_NAMES, UNIT_A_COLORS),
                                   ("B", UNIT_B_IDS, UNIT_B_NAMES, UNIT_B_COLORS)):
        for key in ("up", "down", "side"):
            aid = ids[key]
            out[aid] = _spec(key, unit, aid, names[aid], cols[aid],
                             dz_cm=dz_cm, boxes_w=boxes_w)
    # six arms in one room: each one's static obstacles include the other
    # five's pose-invariant base columns (mounts.arm_column_box)
    return mounts.attach_body_columns(out)


# AS DRAWN: both side arms hang off the ends of their poles at canvas z 0.776.
FLEET_FINAL6 = _build_fleet6()

# THE OPTIMAL CONFIGURATION (user decision, 2026-08-21): both poles lengthened
# 20 cm downward and both side arms re-clamped at canvas z 0.576.  This is the
# fleet `fleet.activate("final6_opt")` installs, and it ASSUMES the two ~20 cm
# pole extensions are physically fitted — see `ASSUMPTIONS` and every output.
FLEET_FINAL6_OPT = _build_fleet6(dz_cm=SIDE_DZ_OPT_CM,
                                 boxes_w=FRAME_BOXES6_OPT_W_CM)
SIDE_Z_OPT = float(FLEET_FINAL6_OPT[2].z)          # 0.576 m above the paper

UNIT_OF = {13: "A", 31: "A", 2: "A", 17: "B", 71: "B", 97: "B"}
TWIN = {13: 17, 31: 71, 2: 97, 17: 13, 71: 31, 97: 2}   # mirror partners


def web_of(arm_id):
    """The web an arm draws on -> ((x0,y0),(w,h))."""
    if MERGE_WEBS:
        return ((0.0, 0.0), SHEET_FINAL6)
    return WEB_A if UNIT_OF[arm_id] == "A" else WEB_B


ASSUMPTIONS = [
    ("mirror plane",
     f"the two frames abut EXACTLY, zero gap, at the hanging-arm-side outer "
     f"face W Y = {MIRROR_PLANE_W_CM} cm (canvas y = "
     f"{MIRROR_PLANE_CANVAS_Y:.6f} m).  Levelling-foot pads overhang the leg "
     f"lines by 0.33 cm, so touching pads would hold the structural faces "
     f"0.66 cm apart — confirm the intended gap (knob: GAP_CM)."),
    ("unit B arm ids",
     "17 (floor), 71 (inverted), 97 (side) — the legacy six-arm registry's "
     "other three physical arms.  No drawing names them; confirm which arm "
     "goes to which mount."),
    ("web seam",
     f"MERGE_WEBS = {MERGE_WEBS}: ONE continuous drawable surface "
     f"{rig_final.SHEET_FINAL[0]:.4f} x {SHEET_FINAL6[1]:.5f} m spanning both "
     f"units and the {SEAM_M*100:.2f} cm strip between the two webs.  The "
     "strip IS swept now (scripts/run_atlas6.py), but that the web is "
     "physically flat and drawable across it is a build assumption: the strip "
     "lies over the two frames' abutting top rails, not over a tabletop, and "
     "no drawing shows a bridging surface."),
    ("extended poles",
     f"FLEET_FINAL6_OPT re-clamps BOTH side arms (2 and 97) at canvas z "
     f"{SIDE_Z_OPT:.3f} m ({SIDE_DZ_OPT_CM:+.0f} cm), which requires "
     f"{POLE_EXT_OPT_CM:.0f} cm of EXTRA POLE on each unit "
     "(docs/ARM2_HEIGHT.md section 4).  Nothing in the drawing has this steel; "
     "every number computed on this fleet assumes it is fitted.  The same "
     "change is what makes the joint stiff (~28 cm of bracket engagement "
     "instead of 5.8) and answers the drawing's own 'not stiff and stable' "
     "warning — but until it is built AND surveyed under load, keep "
     "rig_final.CALIB_STATIC at 0.03 for both side arms."),
    ("unit B orientations",
     "realised as proper rotations S R S (a true reflection is improper and "
     "unbuildable).  Confirm each real base plate's yaw — rig_final Flags #6 "
     "already flags a possible 180 deg yaw error per plate, which this "
     "mirroring propagates to unit B unchanged."),
    ("inherited",
     "every flag in docs/FINAL_RIG.md applies to BOTH units (arm-2 boom "
     "bottom ambiguity, the drawing's own 'not stiff and stable' warning on "
     "the side mount, arm 13 having no dimension chain)."),
]
