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

from . import rig_final
from .fleet import ArmSpec
from .frames import Q_READY_FLOOR, Q_READY_INV_FINAL, Q_READY_WALL

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
MERGE_WEBS = False   # CONFIG FLAG — user decides.
# False (default): TWO webs with a seam.  Unit A y 0 -> 1.700, unit B
#   y 1.93064 -> 3.63064, and a SEAM_M = 0.23064 m gap between them (the two
#   11.532 cm back margins of paper-edge-to-frame-face, back to back).
# True: one continuous drawable surface y 0 -> 3.63064 spanning the seam.
#   NOT scored anywhere yet — no atlas covers the seam strip, so turning this
#   on changes the geometry but leaves the seam band UNSCORED until a real
#   sweep runs.  Flagged, deliberately not faked.
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


# --- the combined structure ----------------------------------------------
def frame_boxes6_canvas(exclude_tag=None, zmin=0.0):
    """Both units' structure in the CANVAS frame, m.

    Unit A's boxes verbatim from `rig_final.frame_boxes_canvas`; unit B's are
    their exact reflections.  Names are suffixed "@A" / "@B" and each box
    carries `unit`.  `exclude_tag` follows rig_final's convention but is
    matched against the UNIT-QUALIFIED tag ("mount:down@B"), so an arm is
    excused from its own mount hardware and nobody else's — including its
    mirror twin's.
    """
    out = []
    for b in rig_final.frame_boxes_canvas(zmin=zmin):
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

    def static_obstacles(self):
        return frame_boxes6_canvas(
            exclude_tag=f"mount:{self._key}@{self._unit}")


def _spec(key, unit, arm_id, name, color):
    p, R = rig_final.arm_base_canvas(key)
    ready = {"up": Q_READY_FLOOR, "down": Q_READY_INV_FINAL,
             "side": Q_READY_WALL}[key]
    if unit == "B":
        p, R, ready = mirror_point(p), mirror_rotation(R), mirror_q(ready)
    mount = {"up": "floor", "down": "inv", "side": "wall"}[key]
    s = Arm6Spec(arm_id, name, mount, (float(p[0]), float(p[1])), 0.0, True,
                 color, rig="final", z=float(p[2]),
                 R=tuple(np.asarray(R).flatten()), q_ready=tuple(ready))
    object.__setattr__(s, "_key", key)      # frozen dataclass
    object.__setattr__(s, "_unit", unit)
    return s


FLEET_FINAL6 = {}
for _k, _aid, _nm, _c in (("up", 13, "up-front", (0.12, 0.47, 0.71)),
                          ("down", 31, "down-left", (0.84, 0.15, 0.16)),
                          ("side", 2, "side-right", (0.17, 0.63, 0.17))):
    FLEET_FINAL6[_aid] = _spec(_k, "A", _aid, _nm, _c)
for _k in ("up", "down", "side"):
    _aid = UNIT_B_IDS[_k]
    FLEET_FINAL6[_aid] = _spec(_k, "B", _aid, UNIT_B_NAMES[_aid],
                               UNIT_B_COLORS[_aid])
del _k, _aid, _nm, _c

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
     f"MERGE_WEBS = {MERGE_WEBS}: two separate webs with a {SEAM_M*100:.2f} cm "
     "seam between them.  If one continuous web is intended, the seam strip "
     "is currently UNSCORED (no atlas covers it) and needs a real sweep."),
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
