"""SCHEMATIC MOUNT HARDWARE — and the arms' own base columns — as first-class
static obstacles (layout study v2; body columns 2026-08-25).

Two things live here.  The first is the STEEL (below).  The second, added
after the proposed rig refused to conduct anything at all, is the part of a
NEIGHBOUR ARM that is an obstacle in every configuration it can hold: its base
column, `arm_column_boxes`, whose long note is at the bottom of this module.


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
    # link_r/calib restate coordination.LINK_R / coordination.CALIB_M and
    # d1 restates frames.DH[0][2]; tests/test_mounts.py pins all three.
    link_r: float = 0.155             # capsule radius of the base column
    calib: float = 0.03               # unsurveyed-base allowance (inter-arm)
    d1: float = 0.333                 # base flange -> shoulder, modified DH
    # MEASURED, and none of it derivable from the DH table (mesh audit
    # 2026-08-26; the numbers are `out/collision_audit.json` ["column"] and
    # tests/data/collision_audit_geometry.json).  `d1` above is still the DH
    # constant; where the BODY ends is a different question and a bigger
    # number.  See the long note at the bottom of this module.
    column_z1: float = 0.3875         # far end of the body, base z (not d1)
    connector_r: float = 0.177        # base connector + cable stub, radial
    connector_z1: float = 0.0667      # ...how far below the flange it reaches
    connector_up: float = 0.2325      # ...and how far ABOVE it, up the back

    @property
    def column_r(self):
        """Radius of the neighbour-body-column obstacle.  See
        `arm_column_boxes` for why it is link_r + calib and not link_r."""
        return self.link_r + self.calib

    @property
    def column_bands(self):
        """The body column as (z0, z1, r) bands in BASE z, flange downwards.

        Two bands, and every number in them is forced — see the long note at
        the bottom of this module for the derivation.
        """
        return ((-self.connector_up, self.connector_z1,
                 self.connector_r + self.calib),
                (self.connector_z1, self.column_z1, self.column_r))

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
# WHAT IS STATIC ABOUT AN ARM.  Capsule (0, 1) of `coordination.CAPSULES` —
# base flange to shoulder, `d1` = 0.333 m along the base z axis — does not
# move when the arm does: q1 rotates ABOUT that axis and q2..q7 live beyond
# its far end.  So a neighbour's base column is true in every pose, including
# poses nobody has chosen yet, and it belongs with the booms and the plates.
# The REST of a parked neighbour (upper arm, forearm, wrist, pen) is pose
# DEPENDENT and stays the conductor's job, where a schedule can still move it:
# baking a park pose into the static model would be a promise about a pose the
# rig has not committed to.
#
# THE RADIUS, AND WHY IT IS link_r + calib.  A box gate asks for
# `rig_final.STATIC_MARGIN` = 0.05 m of clearance between the mover's capsule
# SURFACE and the box; the conductor asks every pair of arms for
# `SAFETY_M + CALIB_M` = 0.08 m between two capsule surfaces.  Inflating the
# neighbour's own capsule by exactly that difference,
#
#     column_r = link_r + calib,
#
# makes the two gates the SAME statement: a pose that clears the box by
# STATIC_MARGIN clears the neighbour's column by the 0.08 m the conductor will
# later demand of it, so a cell the atlas certifies is a cell the conductor
# can still be handed.  A larger number would refuse ink that runs fine; a
# smaller one would certify ink that deadlocks, which is exactly the failure
# this obstacle exists to end.  That identity is the invariant here; the
# NUMBERS on both sides of it moved on 2026-08-26 and it still holds.
#
# WHAT MOVED (mesh audit, commit 5c8d803, `scripts/collision_audit.py`).  The
# whole model used to be one box: r = 0.09 + 0.03 = 0.12 over `d1` = 0.333 m.
# Exact triangle-mesh distance against the manufacturer's link0 collision mesh
# and the full-resolution visual mesh says all three numbers were wrong:
#
#   * the body is 0.1546 m at its widest, not 0.09 — so `link_r` is 0.155 and
#     `column_r` is 0.185 (`coordination.LINK_R` carries the same 0.155);
#   * it does not stop at the shoulder.  link1's swept volume about q1 is a
#     solid of revolution and therefore pose-invariant too, and it carries the
#     body to `column_z1` = 0.3875 m — 54.5 mm PAST `d1`.  `d1` is untouched:
#     it is the DH constant, and where the metal ends is a separate fact;
#   * the base CONNECTOR and its cable stub reach `connector_r` = 0.177 m
#     radially over the first 67 mm below the flange and 0.2325 m back UP it —
#     outside the plate box (0.113 x 0.095 half-extent, 50 mm thick), outside
#     the boom box (0.1), and outside every model this package had.
#
# TWO BANDS, NOT TWELVE.  The audit re-derived the column as a 12-band
# cylinder stack (fat 0.171 at the plate, waisted to 0.057 in the middle, 0.129
# at the far end).  This ships TWO because the middle bands cannot be used:
# the conductor's own base-column capsule is ONE capsule at `link_r` over the
# whole span, so a box that is thinner than `link_r + calib` anywhere inside
# that span certifies cells the conductor then refuses — the gate-consistency
# identity above, read backwards.  Band 2 is therefore flat at `column_r`, and
# it dominates every measured band below the connector (worst 0.1295 + 0.03 =
# 0.1595 < 0.185).  Band 1 is the connector, where the measurement is FATTER
# than the capsule and the capsule does not dominate.  Measured against the
# real atlas the 12-band stack buys 0.02-0.04 pp of union coverage at the
# heights under consideration, which is not a reason to carry ten more boxes
# through every gate in the package.
#
# THE FAR END IS THE METAL, AND THE IDENTITY IS NOT AIRTIGHT THERE — THE ONE
# PLACE THIS MODULE IS WEAKER THAN THE CONDUCTOR, STATED PLAINLY.  A capsule
# is a segment thickened, so the conductor's cap0 carries a SPHERICAL CAP that
# bulges `link_r` past the shoulder: its far surface is at base z 0.518, while
# the metal stops at `column_z1` = 0.3875.  Containing that cap would mean
# running these boxes to 0.518, and the atlas says what that costs — measured
# on the real 2 cm sweep of all six arms, the corrected capsules included:
#
#     h = 0.940   boxes to 0.3875 (the metal)   union 99.70 %   >=2 arms 48.8 %
#                 boxes to 0.518  (the cap)     union 92.13 %   >=2 arms 36.9 %
#
# 7.6 points of canvas and 12 points of concurrency, to model a shape that is
# not there.  Tapering the box down the cap's own profile recovers almost none
# of it (the ball is still 0.178 m wide 50 mm past the shoulder), because what
# costs the coverage is any fat column standing in the slab the forearms work
# in.  So the boxes stop at the metal, and the residual is DISCHARGED BY
# MEASUREMENT rather than by conservatism: `tests/test_mounts.py` walks the
# certified atlas poses of every arm and checks the conductor's own cap0
# criterion — `seg_seg_dist(mover capsule, neighbour base->shoulder) >=
# mover_r + link_r + 0.08` — directly.  If a rig ever puts a certified pose
# inside a neighbour's shoulder ball, that test fails and says so, which is
# worth more than a box nobody can reach past.  (The cap is not pure fiction:
# it is what makes cap0 CONTAIN link1's swept volume out to 0.3875, which is
# why the capsule cannot simply be shortened.)
#
# Band 1 runs `connector_up` above the flange because the connector is really
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


def obstacles_for(arm_id, fleet, h=0.850, m=MOUNTS):
    """Every OTHER arm's hardware AND body column -> flat box list.

    The own-arm exclusion is the point: an arm is bolted to its own plate and
    boom (and the legacy r = 0.12 own-boom proxy still gates its own column),
    and its own base column is the one capsule every static check skips
    (`rig_final.STATIC_CAPSULES` starts at index 1).
    """
    hw = fleet_mount_boxes(fleet, h, m)
    body = fleet_body_boxes(fleet, m)
    return [b for aid in fleet if aid != arm_id
            for b in hw[aid] + body[aid]]


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
