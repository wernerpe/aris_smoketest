"""SCHEMATIC MOUNT HARDWARE — and the arms' own base columns — as first-class
static obstacles (layout study v2; body columns 2026-08-25).

THREE things live here now.  The first is the STEEL (below).  The second,
added after the proposed rig refused to conduct anything at all, is the part
of a NEIGHBOUR ARM that is an obstacle in every configuration it can hold: its
base column, `arm_column_boxes`, whose long note is at the bottom of this
module.  The third, added 2026-09-14, is the first piece of the CAGE that any
certified number has ever been earned against: the two SEAM BARS where the
installation's two half-cages butt, `SEAM_BARS_MM` / `seam_frame_boxes`, which
stand on the middle arm row's own line.  `obstacles_for` hands all three out
together and `ARIS_SEAM_POSTS=0` removes the third.


v1 of the layout study (docs/LAYOUT_STUDY.md v1, commit 83ad415) modelled NO
mounting hardware: neighbouring booms and base plates were not obstacles, so
its 99.38 % winner was an optimistic kinematic ceiling.  This module supplies
the missing steel — deliberately SCHEMATIC and conservative, derived from the
known rig hardware rather than from a CAD model that does not exist yet for
arbitrary base positions:

  INVERTED arm  base plate box  PLATE_XY (0.226 x 0.190) x PLATE_T (0.05),
                sitting ON TOP of the arm's base flange, i.e. z in [h, h+T];
                vertical boom, a cylinder of radius BOOM_R = 0.10 m running
                from the plate top up to the nominal ceiling grid CEILING_Z
                = 2.34 m.  The cylinder is carried as the CIRCUMSCRIBED
                square column (2*BOOM_R wide) so the box machinery in
                `rig_final` can consume it and stays conservative.
  FLOOR arm     base pedestal box PED_XY (0.30 x 0.25) footprint, from the
                base plate top (Z_FLOOR = 0.0127, the surveyed legacy value)
                down PED_DROP = 0.40 m — the stand the arm bolts to.

Every arm sees ALL OTHER arms' hardware.  It does NOT see its own (it is
bolted there by construction) — the legacy own-boom proxy (r = 0.12 cylinder
above its own plate, `atlas.solve_cell`) still gates its own column, and is
the MORE conservative of the two radii.  That is exactly the final rig's
convention (`fleet.ArmSpec.static_obstacles` with `exclude_tag`).

WHY THE Z BANDS DECIDE EVERYTHING
---------------------------------
Measured from the v1 winner's own certified 2 cm atlas (every strict-GO
pose of all six arms, chain points + both tool points, world frame):

    floor arm, base z 0.0127 : highest chain point 0.650  (the elbow, pt 3)
    inv   arm, base z 0.850  : highest chain point 0.520  (pt 1/2, the
                               shoulder) — the BASE point itself is at 0.850
                               but capsule tables start at index 1, because
                               the base is bolted to the very plate it would
                               otherwise collide with.

A drawing pose has the pen on the paper, so the whole chain hangs LOW.  An
inverted arm's own mount plane is 0.33 m above anything it can put a link
into; a floor arm's elbow tops out 0.20 m below an h = 0.85 mount plane.
The consequence is not a modelling convenience, it is the physical answer:

  * hardware ABOVE the chain (plates, booms) can only bite when
    `plate_bottom < chain_top + CHAIN_R + MARGIN`.  For floor arms that
    threshold is h < 0.813 m (0.790 m on the measured elbow rather than the
    kinematic bound); for inverted arms at a COMMON height it can never be
    met, at any height, because the gap is a constant — 0.095 m since the
    mesh audit widened `chain_r` from 0.09 to 0.155, and it was 0.16 m
    before.  Hence `boom_shadow_active`.
  * THE NEIGHBOUR'S OWN BODY is not in this coarse proxy at all, and at these
    heights it is the only thing an inverted arm's neighbour has that CAN
    bite: `column_bands` hangs to 0.3875 m below the mount plane, well inside
    the chain envelope, while the steel above it never reaches.  `keepouts`
    is a ranking tool for the layout search and the REAL atlas is the
    arbiter (`arm_column_boxes` is what the atlas gates on), so the proxy is
    left optimistic on purpose rather than made half-right — but read a
    coarse number for an all-ceiling rig as an upper bound, not an answer.
  * hardware AT THE PEN'S LEVEL (the floor pedestals, top 0.0127 m) bites
    every OTHER arm that tries to draw next to it, because the pen capsule
    (TOOL_R) plus STATIC_MARGIN must clear the box.  This is where the v1
    study was genuinely optimistic.

MIXED-HEIGHT LAYOUTS ARE OUT OF SCOPE: every layout here puts all inverted
arms on ONE ceiling grid at one h.  A taller arm's boom descending past a
shorter arm's mount plane WOULD bite, and `boom_shadow_active` reports it.
"""
import os
from dataclasses import dataclass, replace

import numpy as np

from . import rig_final

# ---------------------------------------------------------------------------
# the schematic hardware — every dimension parameterised
# ---------------------------------------------------------------------------
Z_FLOOR = 0.0127          # m, surveyed legacy floor base-plate top


@dataclass(frozen=True)
class MountModel:
    """The schematic mount hardware.  All lengths in metres, canvas frame."""
    boom_r: float = 0.10              # ceiling boom cylinder radius
    ceiling_z: float = 2.34           # nominal ceiling grid height
    plate_xy: tuple = (0.226, 0.190)  # inverted base plate footprint
    plate_t: float = 0.05             # inverted base plate thickness
    ped_xy: tuple = (0.30, 0.25)      # floor pedestal footprint
    ped_drop: float = 0.40            # pedestal depth below the plate top
    z_floor: float = Z_FLOOR          # floor base-plate top

    # --- clearance policy (the chain's own widths; see rig_final) ----------
    chain_r: float = 0.155            # thickest link capsule radius
    tool_r: float = 0.05              # pen / bracket capsule radius
    margin: float = rig_final.STATIC_MARGIN   # 0.05 = Z_STATIC + CALIB_STATIC

    # --- chain envelopes (see module docstring) ----------------------------
    # floor_chain_rise is the KINEMATIC bound on the elbow (chain point 3):
    # d1 + d3 = 0.333 + 0.316 = 0.649 above the base, whatever the pose.  The
    # v1 winner's 5 526 certified floor poses measure 0.637.  (Point 4 can in
    # principle swing 0.0825 higher, but only in poses whose pen is nowhere
    # near the paper; the REAL atlas with the real boxes is the arbiter and
    # this constant only steers the coarse ranking.)
    floor_chain_rise: float = 0.66
    inv_chain_drop: float = 0.30      # <= measured 0.330, rounded down
    tool_top: float = 0.12            # highest z of a tool capsule point

    # --- THE NEIGHBOUR'S OWN BODY (see `arm_column_boxes`) -----------------
    # `body_bands` restates coordination.BODY_BANDS, link_r/calib restate
    # coordination.LINK_R / coordination.CALIB_M and d1 restates
    # frames.DH[0][2]; tests/test_mounts.py pins all four, and the same four
    # bands again against the mesh audit's own measurement.
    #
    # MEASURED, and none of it derivable from the DH table (mesh audit
    # 2026-08-26; the numbers are `out/collision_audit.json` ["column"] and
    # tests/data/collision_audit_geometry.json).  `d1` is still the DH
    # constant; where the BODY ends is a different question and a bigger
    # number.  See the long note at the bottom of this module.
    body_bands: tuple = ((-0.2325, 0.0667, 0.177),   # connector + cable stub
                         (0.0667, 0.0988, 0.118),    # the shoulder-ward taper
                         (0.0988, 0.2590, 0.078),    # THE WAIST
                         (0.2590, 0.3875, 0.130))    # link1's swept solid
    link_r: float = 0.155             # widest radius of the casting proper
    calib: float = 0.03               # unsurveyed-base allowance (inter-arm)
    d1: float = 0.333                 # base flange -> shoulder, modified DH

    @property
    def column_z1(self):
        """Far end of the body, base z (not d1)."""
        return self.body_bands[-1][1]

    @property
    def connector_r(self):
        """Base connector + cable stub, radial."""
        return self.body_bands[0][2]

    @property
    def connector_z1(self):
        """...how far below the flange the connector band reaches."""
        return self.body_bands[0][1]

    @property
    def connector_up(self):
        """...and how far ABOVE the flange it runs, up the back."""
        return -self.body_bands[0][0]

    @property
    def column_r(self):
        """Radius the LEGACY single-capsule column obstacle wore — link_r one
        `calib` wider.  No band is this fat below the connector any more (the
        widest is 0.160); it is kept because it is the number the
        gate-consistency identity is stated in.  See `arm_column_boxes`."""
        return self.link_r + self.calib

    @property
    def column_bands(self):
        """The body column as (z0, z1, r) bands in BASE z, flange downwards.

        Four bands, each the measured profile of `body_bands` grown by
        `calib` — see the long note at the bottom of this module for why that
        is the whole derivation and where the four edges come from.
        """
        return tuple((z0, z1, r + self.calib) for z0, z1, r in self.body_bands)

    def scaled(self, **kw):
        """A variant of this model — `MOUNTS.scaled(boom_r=0.12)`."""
        return replace(self, **kw)


MOUNTS = MountModel()


# ---------------------------------------------------------------------------
# hard spacing minima implied by the hardware + the collision margin
# ---------------------------------------------------------------------------
def min_column_spacing(m=MOUNTS):
    """Hard minimum horizontal distance between two inverted arms' COLUMNS.

    Plate and boom share a z band with every other arm's plate and boom, so
    two columns must not interpenetrate: twice the larger circumscribed
    radius plus the static margin.  (The base-spacing constraint the layout
    search already carries, MIN_BASE_DIST = 0.50 m, is stricter — this is the
    HARDWARE floor beneath it, and the study reports which one binds.)

    The ARM's own body joined this list on 2026-08-26: the connector band of
    `column_bands` is 0.207 m of half-extent, wider than either piece of
    steel, and it shares its z band with the neighbour's connector.
    """
    plate_r = 0.5 * float(np.hypot(*m.plate_xy))
    boom_r = float(np.hypot(m.boom_r, m.boom_r))     # circumscribed square
    body_r = max(r for _, _, r in m.column_bands)    # the fattest band's AABB
    return 2.0 * max(plate_r, boom_r, body_r) + m.margin


def min_pedestal_spacing(m=MOUNTS):
    """Hard minimum between two floor pedestals (both at the pen's level)."""
    return float(np.hypot(*m.ped_xy)) + m.margin


def boom_shadow_active(observer_mount, observer_base_z, h, m=MOUNTS):
    """Can a certified DRAWING pose of `observer` reach another arm's boom or
    plate?  -> (bool, headroom_m).  Positive headroom = clear.

    The mount hardware of an inverted arm starts at z = h (plate bottom).
    The observer's chain, plus its thickest capsule radius and the static
    margin, tops out at `chain_top`.  If that is below h the hardware is
    unreachable and the shadow subtraction is geometrically inactive.
    """
    top = chain_top(observer_mount, observer_base_z, m) + m.chain_r + m.margin
    return bool(top > h), float(h - top)


def chain_top(mount, base_z, m=MOUNTS):
    """Highest chain-point z (capsule CENTRES) a certified drawing pose of an
    arm with this mount and base height reaches.  Excludes the base point,
    which every capsule table excludes too."""
    if mount == "floor":
        return base_z + m.floor_chain_rise
    return base_z - m.inv_chain_drop


# ---------------------------------------------------------------------------
# the boxes
# ---------------------------------------------------------------------------
def _aabb_half(w, d, yaw):
    """Half-extents of the axis-aligned bounding box of a w x d rectangle
    rotated by `yaw` — so a yawed plate stays conservative."""
    c, s = abs(np.cos(yaw)), abs(np.sin(yaw))
    return 0.5 * (w * c + d * s), 0.5 * (w * s + d * c)


def _box(name, cx, cy, hx, hy, z0, z1, source, tag):
    return dict(name=name, lo=np.array([cx - hx, cy - hy, z0]),
                hi=np.array([cx + hx, cy + hy, z1]), source=source, tag=tag)


def arm_mount_boxes(mount, xy, yaw=0.0, h=0.850, tag="mount", m=MOUNTS):
    """The schematic hardware of ONE arm -> list of boxes (canvas frame, m).

    inverted: [plate, boom]   floor: [pedestal]
    """
    cx, cy = float(xy[0]), float(xy[1])
    if mount == "inv":
        phx, phy = _aabb_half(*m.plate_xy, yaw)
        return [
            _box(f"{tag}_plate", cx, cy, phx, phy, h, h + m.plate_t,
                 f"schematic inverted base plate {m.plate_xy[0]}x"
                 f"{m.plate_xy[1]}x{m.plate_t} on top of the base flange",
                 tag),
            _box(f"{tag}_boom", cx, cy, m.boom_r, m.boom_r, h + m.plate_t,
                 m.ceiling_z,
                 f"schematic ceiling boom, cylinder r={m.boom_r} carried as "
                 f"its circumscribed square column up to the {m.ceiling_z} m "
                 "grid", tag),
        ]
    if mount == "floor":
        phx, phy = _aabb_half(*m.ped_xy, yaw)
        return [
            _box(f"{tag}_pedestal", cx, cy, phx, phy,
                 m.z_floor - m.ped_drop, m.z_floor,
                 f"schematic floor pedestal {m.ped_xy[0]}x{m.ped_xy[1]} from "
                 f"the base plate top down {m.ped_drop}", tag),
        ]
    raise ValueError(f"no schematic mount for mount={mount!r}")


# ---------------------------------------------------------------------------
# THE SEAM SUPPORT — steel this module never had, because nobody had seen it
# ---------------------------------------------------------------------------
# `StudySpec.static_obstacles` (layout.py) returned the neighbours' mount boxes
# and base columns AND NOTHING ELSE.  Every certified number on the proposed
# rig — the atlas, the certified area, v19, the staged programme — was earned
# against a fleet standing in an EMPTY ROOM with no cage around it at all.
# That was defensible while the cage was a 4.01 m frame on four corner legs
# whose steel all sat outboard of the canvas or 1.6 m above it.
#
# It stopped being defensible on 2026-09-14, when Pete Werner reported that
# the real installation is two half-cages side by side and that the half-cage
# END FRAME — which stands at the paper's MID-LENGTH, on the middle arm row —
# is real steel we do not model.  `system_model.seam_bodies` builds it: one
# REPRESENTATIVE BAR PER SIDE, "as wide as two of the corner struts" (Pete
# Werner, same day), 76.2 x 152.4 in plan, tabletop to runway underside.  This
# function is those bars in the box shape the gates here consume.
#
# IT IS WIRED IN NOW (2026-09-14, the re-certification).  `obstacles_for` adds
# it to every arm's static set, so `validate.check_pose`, `atlas.solve_cell`,
# `paper.route`, `planner` and `scene_check` all gate against it and every
# number earned after this date has the seam in it.  What that cost is in
# docs/DECISIONS.md: the middle row's parks moved, and the certified area lost
# 165 of 16 184 cells with the hole-free block untouched.
#
# THE SWITCH.  `ARIS_SEAM_POSTS=0` in the environment takes the bars back out
# and reproduces the pre-seam numbers exactly — for a regression run against a
# number earned before today, and for nothing else.  It is read ONCE, here, at
# import: a static set that could change under a running process is not a
# static set.  Per-call, pass `seam=False`.
def _seam_default():
    # an EMPTY value is "not set", not "off": the only way to lose the bars is
    # to say so
    v = os.environ.get("ARIS_SEAM_POSTS", "").strip().lower()
    return v not in ("0", "off", "no", "false")


SEAM_POSTS_ON = _seam_default()

# ---------------------------------------------------------------------------
# THE TABLE.  One bar per side, MILLIMETRES, canvas frame, as the build sheet
# speaks.  A hardware change is an edit HERE and nowhere else: another bar is
# one more row, a wider one is its y span, a bar standing on the floor instead
# of the tabletop is its z0.  `system_model.seam_bodies` reads this table and
# `tests/test_system_model.py` pins every number in it against the drawing's
# own derivation, so a hand edit that contradicts the drawing is a red test
# and not a silent disagreement.
#
# WHY THE TABLE IS HERE AND NOT IN `system_model`, where the rest of the
# installation lives: `system_model` reads `layout.FLEET_PROPOSED` at import,
# and `layout` builds that fleet through `obstacles_for` — so the module that
# owns the static set cannot import the model that describes it.  This module
# imports nothing but `rig_final`, so the table can live here and the system
# model can be downstream of it.  The derivation, from `system_model`'s own
# constants (mm):
#
#     x    FR_X0 = -190.5 and FR_X1 - PROFILE = 1917.7   the frame's corner
#          line, exactly the x band of leg_FL / leg_FR — PROFILE = 76.2 wide
#     y    SEAM_Y +/- PROFILE = 1815.32 +/- 76.2         "as wide as two of
#          the corner struts", centred on the seam plane
#     z    LEG_BOTTOM = -27.38 to GRID_U = 1623.62       tabletop to runway
#          underside; cut 1651.0, the same cut as a corner leg
SEAM_BARS_MM = (
    ("seam_bar_W", (-190.50, 1739.12, -27.38), (-114.30, 1891.52, 1623.62)),
    ("seam_bar_E", (1917.70, 1739.12, -27.38), (1993.90, 1891.52, 1623.62)),
)
SEAM_SOURCE = ("REPRESENTATIVE seam support, 76.2 x 152.4 in plan — 'a bar as "
               "wide as two of the corner struts' (Pete Werner, 2026-09-14), "
               "centred on the seam plane where the two half-cages butt.  "
               "Tabletop to runway underside, cut 1651.0 — the same cut as a "
               "corner leg.  IN THE CERTIFIED STATIC SET")


def seam_frame_boxes(tag="seam"):
    """The seam support bars -> boxes in the canvas frame, METRES.

    Same dict shape as `arm_mount_boxes`: name / lo / hi / source / tag, so a
    caller can concatenate it onto `spec.static_obstacles()` unchanged.
    """
    return [dict(name=n, lo=np.array(lo, float) / 1000.0,
                 hi=np.array(hi, float) / 1000.0, source=SEAM_SOURCE, tag=tag)
            for n, lo, hi in SEAM_BARS_MM]


def fleet_mount_boxes(fleet, h=0.850, m=MOUNTS):
    """Every arm's hardware -> {arm_id: [boxes]}, tagged mount:<arm_id>."""
    out = {}
    for aid, spec in fleet.items():
        out[aid] = arm_mount_boxes(spec.mount, spec.xy, spec.yaw, h,
                                   tag=f"mount:{aid}", m=m)
    return out


# ---------------------------------------------------------------------------
# THE NEIGHBOUR'S OWN BODY: the part of an arm that is an obstacle in EVERY
# configuration it can ever hold
# ---------------------------------------------------------------------------
# Steel was not the only thing missing from the model.  Until 2026-08-25 an
# arm's static obstacles were the rig's STRUCTURE and nothing else, so every
# certifying stage in this repo — the atlas sweep, the stroke planner's
# lattice gate, the transit router, the allocator's probes, the placement
# scorer — planned as if the other five arms were not in the room.  On the
# all-ceiling proposed rig that is not a small error: it is why the allocator
# handed each arm ink that its own transverse partner's SHOULDER is standing
# on, and why every conduct of the CSAIL logo refused with a monotone-schedule
# deadlock the conductor could not resolve by waiting.
#
# WHAT IS STATIC ABOUT AN ARM.  The leading `coordination.N_BASE` capsules of
# `coordination.CAPSULES` — bands of the base flange -> shoulder segment,
# `d1` = 0.333 m along the base z axis — do not move when the arm does: q1
# rotates ABOUT that axis and q2..q7 live beyond its far end.  So a
# neighbour's base column is true in every pose, including
# poses nobody has chosen yet, and it belongs with the booms and the plates.
# The REST of a parked neighbour (upper arm, forearm, wrist, pen) is pose
# DEPENDENT and stays the conductor's job, where a schedule can still move it:
# baking a park pose into the static model would be a promise about a pose the
# rig has not committed to.
#
# THE RADIUS, AND WHY IT IS THE MEASURED BAND + calib.  A box gate asks for
# `rig_final.STATIC_MARGIN` = 0.05 m of clearance between the mover's capsule
# SURFACE and the box; the conductor asks every pair of arms for
# `SAFETY_M + CALIB_M` = 0.08 m between two capsule surfaces.  Inflating the
# neighbour's own capsule by exactly that difference — PER BAND,
#
#     box_r(band k) = coordination.BODY_BANDS[k].r + calib,
#
# makes the two gates the SAME statement: a pose that clears the box by
# STATIC_MARGIN clears the neighbour's column by the 0.08 m the conductor will
# later demand of it, so a cell the atlas certifies is a cell the conductor
# can still be handed.  A larger number would refuse ink that runs fine; a
# smaller one would certify ink that deadlocks, which is exactly the failure
# this obstacle exists to end.  That identity is the invariant here; the
# NUMBERS on both sides of it moved on 2026-08-26, twice, and it still holds.
#
# AND IT IS AIRTIGHT, WHICH IT WAS NOT BEFORE.  Each box is the band's AABB
# grown by the band's own radius in EVERY direction (`arm_column_boxes`), so
# it contains that band's capsule inflated by `calib` — the segment's AABB
# contains the segment, and an L-inf ball of radius R contains an L-2 one.
# Hence `d(p, box) >= mover_r + 0.05` implies `d(p, band axis) >= mover_r +
# band_r + 0.08`, which is the conductor's own criterion, at every point and
# with no residual left over for the far spherical caps to hide in.
#
# WHAT MOVED (mesh audit, commit 5c8d803, `scripts/collision_audit.py`).  The
# whole model used to be one box: r = 0.09 + 0.03 = 0.12 over `d1` = 0.333 m.
# Exact triangle-mesh distance against the manufacturer's link0 collision mesh
# and the full-resolution visual mesh says all three numbers were wrong:
#
#   * the body is 0.1546 m at its widest, not 0.09 — so `link_r` is 0.155 and
#     the legacy one-capsule `column_r` was 0.185 (`coordination.LINK_R`
#     carries the same 0.155);
#   * it does not stop at the shoulder.  link1's swept volume about q1 is a
#     solid of revolution and therefore pose-invariant too, and it carries the
#     body to `column_z1` = 0.3875 m — 54.5 mm PAST `d1`.  `d1` is untouched:
#     it is the DH constant, and where the metal ends is a separate fact;
#   * the base CONNECTOR and its cable stub reach `connector_r` = 0.177 m
#     radially over the first 67 mm below the flange and 0.2325 m back UP it —
#     outside the plate box (0.113 x 0.095 half-extent, 50 mm thick), outside
#     the boom box (0.1), and outside every model this package had.
#
# FOUR BANDS, AND WHY NOT TWO (2026-08-26, the second pass).  The audit
# re-derived the column as a 12-band cylinder stack: fat 0.171 at the plate,
# WAISTED to 0.057 through the middle third, and back out to 0.129 where
# link1's swept solid carries it past the shoulder.  The first version of this
# module shipped TWO bands, flat at `column_r` = 0.185 below the connector,
# and gave the honest reason: the conductor's own base column was ONE capsule
# at `link_r` over the whole span, so a box thinner than `link_r + calib`
# anywhere inside it would certify cells the conductor then refuses — the
# gate-consistency identity above, read backwards.
#
# That reason was about the CONDUCTOR'S model, not about the arm.  So the
# conductor's model moved too: `coordination.BODY_BANDS` is now the same four
# bands, as sub-segment capsules of the same pose-invariant axis, and the
# identity holds band for band.  What the flat 0.185 was costing is 13 cm of
# fictitious metal across exactly the slab two transverse arms' forearms have
# to cross in:
#
#     base z          box half-extent   was    is     measured band max
#     [-0.2325, 0.0667]  connector      0.207  0.207  0.1769  (unchanged)
#     [ 0.0667, 0.0988]  taper          0.185  0.148  0.1172   -37 mm
#     [ 0.0988, 0.2590]  THE WAIST      0.185  0.108  0.0779   -77 mm
#     [ 0.2590, 0.3875]  link1 sweep    0.185  0.160  0.1295   -25 mm
#
# WHERE THE FOUR EDGES COME FROM.  They are the audit's own 12-band edges,
# chosen by exhaustive search over that edge set: for each band count, the
# cover of the measurement that minimises the obstacle's cross-section
# integrated over the slab a drawing arm's links can reach (base z >= 0.095 at
# these heights: `inv_chain_drop` 0.30 less `chain_r` + `margin`).  The
# search's answer, in that proxy's units — 1 band 0.0118, 2 bands 0.0075,
# 3 bands 0.0053, 4 bands 0.0051, 5 bands 0.0050, against 0.0100 for the flat
# pair this replaces.  Three bands below the connector take 96 % of everything
# a twelve-band stack could take, and the fourth and fifth take 0.4 % between
# them, which is not worth eight more boxes through every gate in the package.
# Each edge is rounded off the audit's grid in the direction that makes the
# FATTER of its two neighbours grow, so no shipped band is ever thinner than a
# measured band it touches; `tests/test_mounts.py` checks that overlap by
# overlap against `tests/data/collision_audit_geometry.json`.
#
# THE FAR END IS THE METAL, AND THE CAPS ARE INSIDE THE BOXES.  A capsule is a
# segment thickened, so each band carries a SPHERICAL CAP past its own ends —
# the last one to base z 0.5175, while the metal stops at `column_z1` =
# 0.3875.  Nothing has to be done about that: the box is the band's AABB grown
# by the band's radius along the axis as well as across it, so it contains the
# cap it belongs to (see the identity above).  What the boxes must NOT do is
# run out to the cap as a BAND — the atlas says what a fat column standing in
# the forearms' slab costs, measured on the real 2 cm sweep of all six arms:
#
#     h = 0.940   bands to 0.3875 (the metal)   union 99.70 %   >=2 arms 48.8 %
#                 bands to 0.518  (the cap)     union 92.13 %   >=2 arms 36.9 %
#
# 7.6 points of canvas and 12 points of concurrency, to model a shape that is
# not there.  `tests/test_mounts.py` still walks the certified atlas poses of
# every arm against the conductor's own base-column criterion directly, band
# for band, because a measured discharge of the identity is worth more than
# an argument about it.
#
# Band 0 runs `connector_up` above the flange because the connector is really
# there — 0.177 m of it, outside the plate box and outside the boom box, and
# outside every model this package had before the audit.  Nothing reaches it:
# a hanging arm's chain never rises above `d1` below its own mount plane, so
# the headroom is a constant 0.333 m at every height (measured, all six arms,
# every certified drawing pose).  It is carried anyway, because the day
# something DOES reach up there the model should already know about it.
#
# THE BOXES ARE THE BANDS' AABBs, which for a column parallel to a canvas axis
# — every mount in every rig here — is the circumscribed square column the
# booms already use (`arm_mount_boxes`), tight on the four faces and up to
# 41 % conservative on the diagonals.  A base whose axis is NOT axis-aligned
# (none today) would get fatter, still conservative, boxes.
def arm_column_boxes(spec, tag=None, m=MOUNTS, h_inv=None):
    """The pose-INVARIANT body column of ONE arm -> boxes (canvas frame).

    `spec` supplies the base pose; each band of `m.column_bands` runs along
    the base z axis and becomes the AABB of that band.  See the note above
    for every number in that sentence.
    """
    T = spec.T_world_base() if h_inv is None else spec.T_world_base(h_inv)
    p0 = np.asarray(T[:3, 3], float)
    zc = np.asarray(T[:3, 2], float)
    aid = spec.arm_id
    out = []
    for k, (z0, z1, r) in enumerate(m.column_bands):
        a, b = p0 + z0 * zc, p0 + z1 * zc
        out.append(dict(
            name=f"body:{aid}_column{k}",
            lo=np.minimum(a, b) - r, hi=np.maximum(a, b) + r,
            source=f"arm {aid} base column band {k}: r = {r:.3f} m over base "
                   f"z [{z0:+.4f}, {z1:+.4f}] — MEASURED body (mesh audit) "
                   f"inflated by calib {m.calib}, pose-invariant, carried as "
                   "the band's AABB",
            tag=f"body:{aid}" if tag is None else tag))
    return out


def fleet_body_boxes(fleet, m=MOUNTS, h_inv=None):
    """Every arm's own body column -> {arm_id: [boxes]}, tagged body:<id>."""
    return {aid: arm_column_boxes(spec, m=m, h_inv=h_inv)
            for aid, spec in fleet.items()}


def obstacles_for(arm_id, fleet, h=0.850, m=MOUNTS, seam=None):
    """Every OTHER arm's hardware AND body column, and the SEAM BARS -> boxes.

    The own-arm exclusion is the point: an arm is bolted to its own plate and
    boom (and the legacy r = 0.12 own-boom proxy still gates its own column),
    and its own base column is the one capsule every static check skips
    (`rig_final.STATIC_CAPSULES` starts at index 1).

    The seam bars are NOT any arm's own hardware — they are the room — so
    every arm sees them, including the two whose J1 axes stand on the seam
    plane.  `seam=False` (or `ARIS_SEAM_POSTS=0`) leaves them out and gives
    back the set every pre-2026-09-14 number was earned against.
    """
    hw = fleet_mount_boxes(fleet, h, m)
    body = fleet_body_boxes(fleet, m)
    out = [b for aid in fleet if aid != arm_id
           for b in hw[aid] + body[aid]]
    if SEAM_POSTS_ON if seam is None else bool(seam):
        out += seam_frame_boxes()
    return out


def attach_body_columns(fleet, m=MOUNTS, h_inv=None):
    """Give every spec in `fleet` the OTHER arms' body columns. -> the fleet.

    Written onto the frozen specs after construction (the `rig_final6._spec`
    pattern), because a spec cannot know its neighbours until the registry it
    belongs to exists.  A fleet that never gets this call keeps
    `column_boxes = ()` and behaves exactly as it did before — which is what
    the legacy six-arm registry wants (`fleet.FLEET_SIXARM` is kept verbatim
    for regression and models no structure at all).
    """
    per = fleet_body_boxes(fleet, m, h_inv)
    for aid, spec in fleet.items():
        object.__setattr__(spec, "column_boxes",
                           tuple(b for o in fleet if o != aid
                                 for b in per[o]))
    return fleet


# ---------------------------------------------------------------------------
# the CHEAP coarse proxy: what one arm's disc loses to another arm's hardware
# ---------------------------------------------------------------------------
def keepouts(observer_mount, observer_xy, observer_base_z, others, h,
             m=MOUNTS):
    """The obstacle-aware corrections to ONE arm's reach annulus.

    `others` = [(mount, (x, y))] of every OTHER arm.  -> (shadows, blocks):

      shadows  [(centre (2,), r)]   a COLUMN the observer's links cannot pass:
               hardware whose bottom is inside the observer's chain envelope.
               The caller removes every cell whose base->cell SEGMENT comes
               within r of the centre — a wedge/capsule subtraction that is
               conservative by construction (it also removes the disc around
               the column itself).
      blocks   [(lo (2,), hi (2,), r)]  hardware at the PEN's level: the
               observer may not put its tool inside the inflated footprint,
               but its links pass freely OVER it, so this is a footprint
               keep-out and NOT a shadow.

    Which of the two a piece of hardware becomes is decided by z bands, not
    by taste — see the module docstring.
    """
    top = chain_top(observer_mount, observer_base_z, m)
    link_reach = top + m.chain_r + m.margin      # highest a link capsule goes
    tool_reach = m.tool_top + m.tool_r + m.margin
    shadows, blocks = [], []
    for mount, xy in others:
        c = np.asarray(xy, float)
        if mount == "inv":
            # plate [h, h+t] and boom [h+t, ceiling] — one column from h up
            if h < link_reach:
                r = max(0.5 * float(np.hypot(*m.plate_xy)), m.boom_r)
                shadows.append((c, r + m.chain_r + m.margin))
            elif h < tool_reach:                 # unreachable in practice
                r = max(0.5 * float(np.hypot(*m.plate_xy)), m.boom_r)
                blocks.append((c - r, c + r, m.tool_r + m.margin))
        elif mount == "floor":
            # pedestal top 0.0127: under every link, but right at the pen
            hx, hy = _aabb_half(*m.ped_xy, 0.0)
            blocks.append((c - [hx, hy], c + [hx, hy], m.tool_r + m.margin))
    return shadows, blocks


def apply_keepouts(mask, P, base, shadows, blocks):
    """Remove the shadowed/blocked cells from one arm's annulus mask.

    `mask` (N,) bool over cells `P` (N,2); `base` (2,) the arm's base.
    Returns a NEW mask.  Vectorised: a shadow is a point-to-segment distance
    (base -> cell) against the column axis; a block is a point-to-rectangle
    distance.  ~4 numpy ops per obstacle over the whole grid.
    """
    keep = mask.copy()
    if not keep.any():
        return keep
    a = np.asarray(base, float)
    for c, r in shadows:
        u = P - a                                   # (N,2) base -> cell
        w = c - a                                   # (2,)
        uu = np.einsum("ij,ij->i", u, u)
        t = np.divide(u @ w, uu, out=np.zeros(len(P)), where=uu > 1e-12)
        t = np.clip(t, 0.0, 1.0)
        d = np.linalg.norm(w[None] - t[:, None] * u, axis=1)
        keep &= d >= r
    for lo, hi, r in blocks:
        d = np.linalg.norm(np.maximum(np.maximum(lo - P, P - hi), 0.0), axis=1)
        keep &= d >= r
    return keep
