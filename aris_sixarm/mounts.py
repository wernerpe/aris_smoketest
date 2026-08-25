"""SCHEMATIC MOUNT HARDWARE as first-class static obstacles (layout study v2).

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
    met, at any height, because the gap is a constant 0.16 m.  Hence
    `boom_shadow_active`.
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
    chain_r: float = 0.09             # thickest link capsule radius
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
    """
    plate_r = 0.5 * float(np.hypot(*m.plate_xy))
    boom_r = float(np.hypot(m.boom_r, m.boom_r))     # circumscribed square
    return 2.0 * max(plate_r, boom_r) + m.margin


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


def obstacles_for(arm_id, fleet, h=0.850, m=MOUNTS):
    """Every OTHER arm's hardware -> flat box list.  The own-mount exclusion
    is the point: an arm is bolted to its own plate and boom (and the legacy
    r = 0.12 own-boom proxy still gates its own column)."""
    per = fleet_mount_boxes(fleet, h, m)
    return [b for aid, boxes in per.items() if aid != arm_id for b in boxes]


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
