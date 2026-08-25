"""GREEN-FIELD layout study: 2 floor + 4 ceiling-inverted arms, lateral tool.

USER MANDATE (2026-08-25): re-evaluate the arm positions.  Configuration is
fixed at 2 FLOOR arms + 4 CEILING-INVERTED arms, NO wall mounts, drawing the
merged 1.8034 x 3.63064 m canvas with the LATERAL pen holder (docs/DECISIONS
"LATERAL PEN HOLDER").  THIS IS A GREEN-FIELD STUDY TO INFORM THE PHYSICAL
REDESIGN: no frame model exists for arbitrary base positions, so the study
specs carry NO static obstacle boxes — structure and collision modelling
follow once a layout is picked.  What IS modelled conservatively:

  * the paper-plane clearance every certified pose keeps (Z_PAPER);
  * each inverted arm's own mounting boom (the legacy r = 0.12 cylinder above
    the plate — ceiling arms will hang from SOMETHING);
  * base-spacing sanity: bases >= 0.5 m apart, and every floor base at least
    0.6 m horizontally from every inverted base so no ceiling boom descends
    into a floor arm's workspace envelope.

MOUNT CONVENTIONS (all gate-validated legacy machinery, reused verbatim):
  floor:  upright, base plate top at canvas z = 0.0127 (the SURVEYED legacy
          floor value, one 12.7 mm plate above the paper; the final drawing's
          arm 13 sits at 0.0107 — the 2 mm difference is noise at this
          grid).  Bases sit OUTSIDE the drawable web, off any edge.
  inv:    hanging, base plate at canvas z = h (0.85 / 0.922 / 1.00 swept),
          R = roty(pi) @ rotz(yaw) — the legacy inverted seat.

RADIAL GO PROFILES (measured 2026-08-25, `scripts/layout_study.py --profiles`;
strict gates margin >= 0.30 & sigma >= 0.14, perpendicular pen, LATERAL tool,
no boxes; yaw-invariant to the 2 cm probe across 5 bearings):

    mount     h      inline GO annulus   LATERAL GO annulus
    floor    0.0127     [0.46, 0.80]        [0.34, 0.90]
    inv      0.850      [0.32, 0.68]        [0.20, 0.84]
    inv      0.922      [0.38, 0.66]        [0.16, 0.82]
    inv      1.000      [0.26, 0.64]        [0.16, 0.78]

The disc model the coarse search covers the canvas with is exactly these
annuli.  `FLEET_PROPOSED` at the bottom is the study's winner, env-selectable
(`ARIS_RIG=proposed`), and NOT the default.
"""
import numpy as np

from .fleet import ArmSpec
from .rig_final6 import SHEET_FINAL6

# the measured strict-GO annuli, LATERAL tool (see module docstring)
PROFILES_LAT = {
    ("floor", None): (0.34, 0.90),
    ("inv", 0.850): (0.20, 0.84),
    ("inv", 0.922): (0.16, 0.82),
    ("inv", 1.000): (0.16, 0.78),
}
PROFILES_INLINE = {
    ("floor", None): (0.46, 0.80),
    ("inv", 0.850): (0.32, 0.68),
    ("inv", 0.922): (0.38, 0.66),
    ("inv", 1.000): (0.26, 0.64),
}

Z_FLOOR = 0.0127          # canvas m, surveyed legacy floor base plate top
MIN_BASE_DIST = 0.50      # m, any two bases (horizontal)
MIN_FLOOR_INV_DIST = 0.60 # m, floor base to inverted base (boom vs workspace)
# m, a floor base's J1 axis must sit at least this far outside the web: the
# 22.582 x 19.0 cm base plate would otherwise lie ON the paper (the final
# rig's arm 13 sits 12.7 cm out).  Upper bound = how far out the search roams.
FLOOR_SETBACK = (0.13, 0.35)

# arm ids: the six physical arms.  13/17 stay on the floor; 31/71 stay
# inverted; 2/97 are RE-MOUNTED inverted (no wall mounts in this layout).
FLOOR_IDS = (13, 17)
INV_IDS = (31, 71, 2, 97)
COLORS = {13: (0.12, 0.47, 0.71), 17: (0.09, 0.75, 0.81),
          31: (0.84, 0.15, 0.16), 71: (1.00, 0.50, 0.05),
          2: (0.17, 0.63, 0.17), 97: (0.58, 0.40, 0.74)}


class StudySpec(ArmSpec):
    """An ArmSpec whose sheet is the MERGED canvas and whose world has no
    frame boxes (green field).  The legacy proxies stay: paper plane, and the
    inverted arms' own boom cylinder (rig stays "sixarm" so `planner`,
    `validate` and `atlas` keep gating it)."""
    __slots__ = ()

    @property
    def unit(self):
        return "study"        # fleet.sheet_for -> SHEET_FINAL6

    def static_obstacles(self):
        return []


def study_spec(arm_id, mount, xy, h=0.922, name=None):
    """One green-field arm.  Floor arms get z through T_world_base's legacy
    rule (Z_FLOOR_BASE); inverted arms take `h` through the h_inv argument,
    so callers pass h_inv=h to atlas/planner for those."""
    yaw = 0.0
    if mount == "floor":
        # face the canvas centre: reach is yaw-invariant, but the seed pose
        # and q1 limits prefer the work in front of the arm
        c = np.array([SHEET_FINAL6[0] / 2, SHEET_FINAL6[1] / 2])
        yaw = float(np.arctan2(c[1] - xy[1], c[0] - xy[0]))
    return StudySpec(arm_id, name or f"{mount}{arm_id}", mount,
                     (float(xy[0]), float(xy[1])), yaw, True,
                     COLORS.get(arm_id, (0.5, 0.5, 0.5)))


def build_fleet(layout):
    """A layout dict -> {arm_id: StudySpec}.

    layout = dict(floor=[(x, y), (x, y)], inv=[(x, y) x4], h=0.922)
    """
    fl = {}
    for aid, xy in zip(FLOOR_IDS, layout["floor"]):
        fl[aid] = study_spec(aid, "floor", xy)
    for aid, xy in zip(INV_IDS, layout["inv"]):
        fl[aid] = study_spec(aid, "inv", xy, h=layout.get("h", 0.922))
    return fl


def check_spacing(layout):
    """-> list of violated constraints (empty = sane)."""
    bad = []
    pts = [("floor", np.asarray(p, float)) for p in layout["floor"]] \
        + [("inv", np.asarray(p, float)) for p in layout["inv"]]
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = float(np.linalg.norm(pts[i][1] - pts[j][1]))
            lim = MIN_FLOOR_INV_DIST if {pts[i][0], pts[j][0]} == \
                {"floor", "inv"} else MIN_BASE_DIST
            if d < lim:
                bad.append(f"{pts[i][0]}{i}-{pts[j][0]}{j}: {d:.2f} < {lim}")
    W, H = SHEET_FINAL6
    for k, p in enumerate(layout["floor"]):
        x, y = p
        d = float(np.hypot(max(0.0 - x, x - W, 0.0), max(0.0 - y, y - H, 0.0)))
        if d < FLOOR_SETBACK[0]:
            bad.append(f"floor{k} base ({x:.2f},{y:.2f}) is {d:.3f} m from "
                       f"the web edge (< {FLOOR_SETBACK[0]} setback: the "
                       "22.6 cm base plate would lie on the paper)")
    return bad


# ===========================================================================
# THE PROPOSED LAYOUT — the study's winner (scripts/layout_study.py)
# ===========================================================================
# Filled by the 2026-08-25 study (docs/LAYOUT_STUDY.md): the fine-stage
# winner — 99.38 % union strict-GO at 2 cm with the 15-degree cone, 50.31 %
# >= 2-arm overlap.  The two floor arms stand side by side off the SOUTH
# short edge; the four inverted arms form two transverse pairs at y ~ 1.40
# and y ~ 2.87, each pair splitting the width so its partner covers its
# under-base hole.  GREEN FIELD: no structure exists for these positions;
# every number assumes a redesign puts steel where these bases need it and
# NOTHING ELSE inside the envelope the atlas swept.
LAYOUT_PROPOSED = dict(
    floor=[(1.15672, -0.14343), (0.55872, -0.13209)],
    inv=[(0.56596, 1.41294), (1.23186, 1.39667),      # south transverse pair
         (0.54082, 2.85289), (1.19379, 2.88897)],     # north transverse pair
    h=0.850,
)


def _proposed():
    return build_fleet(LAYOUT_PROPOSED)


FLEET_PROPOSED = _proposed()
