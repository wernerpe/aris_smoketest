"""LAYOUT STUDY: 2 floor + 4 ceiling-inverted arms (and the all-ceiling
variant), lateral tool.

USER MANDATE (2026-08-25): re-evaluate the arm positions.  NO wall mounts;
the canvas is the merged 1.8034 x 3.63064 m web and the tool is the LATERAL
pen holder (docs/DECISIONS "LATERAL PEN HOLDER").  Two FAMILIES are studied:

    2 + 4   two floor arms + four ceiling-inverted arms
    0 + 6   all six ceiling-inverted (uniform hardware, one grid)

**v2 (2026-08-25) — MOUNTING HARDWARE IS NOW A FIRST-CLASS OBSTACLE.**  v1 of
this study modelled NO steel, so its 99.38 % winner was an optimistic
kinematic ceiling: neighbouring booms and base plates were not obstacles.
Every arm now sees ALL OTHER arms' schematic hardware — inverted booms
(r = 0.10 to a 2.34 m ceiling grid) and base plates (0.226 x 0.190 x 0.05),
floor pedestals (0.30 x 0.25 from the plate top down) — through
`aris_sixarm/mounts.py`, which the specs hand to the atlas as static
obstacle boxes.  What is modelled, conservatively:

  * the paper-plane clearance every certified pose keeps (Z_PAPER);
  * EVERY OTHER arm's mount hardware, as boxes, in the real lattice
    clearance check (`mounts.obstacles_for`, own mount excluded);
  * each inverted arm's own mounting boom (the legacy r = 0.12 cylinder above
    its own plate — MORE conservative than the schematic r = 0.10);
  * base-spacing sanity: bases >= 0.5 m apart, every floor base at least
    0.6 m horizontally from every inverted base, and the HARDWARE minima
    (`mounts.min_column_spacing`, `mounts.min_pedestal_spacing`).

Still NOT modelled: the ceiling grid's own cross-members, paper transport,
cable routing, and inter-arm (moving-arm-vs-moving-arm) collision.

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
from dataclasses import dataclass, field

import numpy as np

from . import mounts
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
# m, a floor base's J1 axis must sit at least this far outside the web.  v1
# set this from the 22.582 x 19.0 cm base PLATE (the final rig's arm 13 sits
# 12.7 cm out); v2 must also keep the 0.30 x 0.25 m PEDESTAL off the paper,
# which needs 0.125 — so 0.13 still binds, by 5 mm.  Upper bound = how far
# out the search roams.
FLOOR_SETBACK = (0.13, 0.35)

# HARDWARE minima implied by the schematic mounts + the static collision
# margin (mounts.py).  Reported alongside the base-spacing constraints so the
# study can say WHICH one binds — at these dimensions the base-spacing rules
# are stricter, and the hardware never gets to be the active constraint.
MIN_COLUMN_DIST = mounts.min_column_spacing()      # inverted column vs column
MIN_PEDESTAL_DIST = mounts.min_pedestal_spacing()  # pedestal vs pedestal

# arm ids: the six physical arms.  In the 2 + 4 family 13/17 stay on the
# floor, 31/71 stay inverted and 2/97 are RE-MOUNTED inverted (no wall mounts
# in this layout).  In the 0 + 6 family all six hang from the ceiling grid.
FLOOR_IDS = (13, 17)
INV_IDS = (31, 71, 2, 97)
ALL_IDS = FLOOR_IDS + INV_IDS
COLORS = {13: (0.12, 0.47, 0.71), 17: (0.09, 0.75, 0.81),
          31: (0.84, 0.15, 0.16), 71: (1.00, 0.50, 0.05),
          2: (0.17, 0.63, 0.17), 97: (0.58, 0.40, 0.74)}


@dataclass(frozen=True, eq=False)
class StudySpec(ArmSpec):
    """An ArmSpec whose sheet is the MERGED canvas and whose static obstacles
    are the SCHEMATIC MOUNT HARDWARE of every OTHER arm (`mounts.py`).

    v1 carried no boxes at all.  v2 carries the neighbours' booms, plates and
    pedestals, so `atlas.solve_cell` runs its real lattice clearance check
    against them.  The legacy proxies stay on top: paper plane, and the
    inverted arms' OWN boom cylinder (rig stays "sixarm", so `planner`,
    `validate` and `atlas` keep gating it at the more conservative r = 0.12).
    """
    mount_boxes: tuple = field(default=(), compare=False, repr=False)

    @property
    def unit(self):
        return "study"        # fleet.sheet_for -> SHEET_FINAL6

    def static_obstacles(self):
        """Every OTHER arm's mount hardware.  Own mount excluded by
        construction — the arm is bolted to it (fleet.ArmSpec convention)."""
        return list(self.mount_boxes)


def study_spec(arm_id, mount, xy, h=0.922, name=None, mount_boxes=()):
    """One study arm.  Floor arms get z through T_world_base's legacy rule
    (Z_FLOOR_BASE); inverted arms take `h` through the h_inv argument, so
    callers pass h_inv=h to atlas/planner for those."""
    yaw = 0.0
    if mount == "floor":
        # face the canvas centre: reach is yaw-invariant, but the seed pose
        # and q1 limits prefer the work in front of the arm
        c = np.array([SHEET_FINAL6[0] / 2, SHEET_FINAL6[1] / 2])
        yaw = float(np.arctan2(c[1] - xy[1], c[0] - xy[0]))
    return StudySpec(arm_id, name or f"{mount}{arm_id}", mount,
                     (float(xy[0]), float(xy[1])), yaw, True,
                     COLORS.get(arm_id, (0.5, 0.5, 0.5)),
                     mount_boxes=tuple(mount_boxes))


def arm_ids(layout):
    """-> (floor ids, inverted ids) for a layout of either family.  The six
    physical arms in a fixed order, floor arms taking 13/17 first."""
    nf = len(layout.get("floor", ()))
    fids = FLOOR_IDS[:nf]
    iids = tuple(a for a in ALL_IDS if a not in fids)
    return fids, iids


def build_fleet(layout, mount_model=mounts.MOUNTS, with_mounts=True):
    """A layout dict -> {arm_id: StudySpec}, each carrying the OTHER arms'
    schematic mount hardware as static obstacle boxes.

    layout = dict(floor=[(x, y) x nf], inv=[(x, y) x (6 - nf)], h=0.922)
    `with_mounts=False` reproduces the v1 GREEN-FIELD specs exactly (no
    boxes) — that is how the study re-scores v1's winner honestly.
    """
    h = float(layout.get("h", 0.922))
    fids, iids = arm_ids(layout)
    bare = {}
    for aid, xy in zip(fids, layout["floor"]):
        bare[aid] = study_spec(aid, "floor", xy, h=h)
    for aid, xy in zip(iids, layout["inv"]):
        bare[aid] = study_spec(aid, "inv", xy, h=h)
    if not with_mounts:
        return bare
    per = mounts.fleet_mount_boxes(bare, h, mount_model)
    return {aid: study_spec(s.arm_id, s.mount, s.xy, h=h,
                            mount_boxes=[b for o, bs in per.items()
                                         if o != aid for b in bs])
            for aid, s in bare.items()}


def check_spacing(layout, m=mounts.MOUNTS):
    """-> list of violated constraints (empty = sane).

    Both the WORKSPACE rules (v1) and the HARDWARE minima the schematic
    mounts imply (v2).  A layout that passes is one whose steel fits and
    whose arms are not sitting in each other's laps.
    """
    bad = []
    pts = [("floor", np.asarray(p, float)) for p in layout["floor"]] \
        + [("inv", np.asarray(p, float)) for p in layout["inv"]]
    col = mounts.min_column_spacing(m)
    ped = mounts.min_pedestal_spacing(m)
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = float(np.linalg.norm(pts[i][1] - pts[j][1]))
            kinds = {pts[i][0], pts[j][0]}
            lim = MIN_FLOOR_INV_DIST if kinds == {"floor", "inv"} \
                else MIN_BASE_DIST
            if d < lim:
                bad.append(f"{pts[i][0]}{i}-{pts[j][0]}{j}: {d:.2f} < {lim}")
            # the HARDWARE floor beneath the base-spacing rule
            hw = col if kinds == {"inv"} else (ped if kinds == {"floor"}
                                               else 0.0)
            if hw and d < hw:
                bad.append(f"{pts[i][0]}{i}-{pts[j][0]}{j}: mount hardware "
                           f"interpenetrates ({d:.2f} < {hw:.3f})")
    W, H = SHEET_FINAL6
    for k, p in enumerate(layout["floor"]):
        x, y = p
        d = float(np.hypot(max(0.0 - x, x - W, 0.0), max(0.0 - y, y - H, 0.0)))
        if d < FLOOR_SETBACK[0]:
            bad.append(f"floor{k} base ({x:.2f},{y:.2f}) is {d:.3f} m from "
                       f"the web edge (< {FLOOR_SETBACK[0]} setback: the "
                       "22.6 cm base plate would lie on the paper)")
        if d < 0.5 * m.ped_xy[1]:
            bad.append(f"floor{k} pedestal ({m.ped_xy[0]}x{m.ped_xy[1]}) "
                       f"overhangs the web ({d:.3f} < {0.5 * m.ped_xy[1]})")
    return bad


# ===========================================================================
# THE BUILDABLE FORM OF THE ALL-CEILING OPTIMUM
# ===========================================================================
# The 0 + 6 search optimum is not a scatter of survey numbers: every restart
# converges on the SAME regular figure — three transverse PAIRS, evenly
# spaced along the canvas, each pair straddling the centre line.  The pair
# spacing is set by one piece of geometry: an inverted arm's annulus is
# [0.20, 0.84] at h = 0.85, so for a partner to cover the whole of an arm's
# r < 0.20 under-base hole the two bases must be at least 0.40 apart (the far
# lip of the hole must clear the partner's inner radius) and at most 0.64
# (the near lip must stay inside the partner's outer radius).  The search
# lands at ~0.61 — near the top of that window, which is also where the pair
# reaches furthest sideways.  Rows sit at the centres of an even tiling of
# the canvas length, which is what puts them 1.21 m apart here.
#
# So the build does not need the search's coordinates; it needs a spacing and
# a row count.  `paired_grid` is that layout in round numbers.
PAIR_SPACING = 0.61       # m, transverse pair separation (see above)
PAIR_WINDOW = (0.40, 0.64)  # m, spacings that keep a partner over the hole


def paired_grid(spacing=PAIR_SPACING, rows=3, h=0.850, sheet=SHEET_FINAL6):
    """The all-ceiling layout as a REGULAR GRID -> a layout dict.

    `rows` transverse pairs at the centres of an even tiling of the canvas
    length, each pair straddling the centre line at `spacing`.  Six arms for
    rows = 3.  Derived from the canvas dimensions, not from search output —
    the figure a fabricator can set out with a tape measure.
    """
    W, Hs = sheet
    xs = (W / 2 - spacing / 2, W / 2 + spacing / 2)
    ys = [(2 * j + 1) * Hs / (2 * rows) for j in range(rows)]
    return dict(floor=[], inv=[(x, y) for y in ys for x in xs], h=float(h))


def pair_spacing_of(layout_d):
    """Mean nearest-neighbour spacing of the inverted bases (the transverse
    pair spacing for a paired grid) -> m."""
    inv = np.asarray(layout_d["inv"], float)
    if len(inv) < 2:
        return float("nan")
    left, ds = list(range(len(inv))), []
    while len(left) >= 2:
        i = left.pop(0)
        d, j = min((float(np.linalg.norm(inv[i] - inv[k])), k) for k in left)
        left.remove(j)
        ds.append(d)
    return float(np.mean(ds))


# ===========================================================================
# THE CERTIFIED READY POSE
# ===========================================================================
def certified_ready_pose(spec, h_inv, hover=0.10, sheet=SHEET_FINAL6,
                         pen_lat=None):
    """A gated READY pose for one study arm with the LATERAL tool.

    `pen_lat` defaults to the lateral holder EXPLICITLY rather than to the
    process-global ACTIVE tool: this is a study function, the study's tool is
    settled, and a caller must not have to mutate `frames.PEN_LAT` (a global
    that leaks into every other module) just to ask for a ready pose.

    Hover `hover` m over a comfortable point of the arm's own annulus, on the
    ray towards the canvas centre.  Scans 8 tool yaws x the q7 grid x all IK
    branches — with the lateral holder phi is a REAL DOF, so a hover pinned to
    phi = 0 can be unreachable at a spot the arm covers at another phi.  Keeps
    the best min(margin, 2.5 sigma) among poses that pass `validate.check_pose`
    with the PEN TIP above the paper — the check that caught the legacy
    inverted ready pose dipping 16 mm under at this h.

    -> (q (7,), hover xy (2,), validate report).  Raises if none is certified.

    Lives here rather than in the scene script because the layout study's fine
    stage must certify the SAME pose it draws (docs/LAYOUT_STUDY.md v2 §ready
    poses): one implementation, two callers.
    """
    from . import ik, metrics
    from .frames import (joint_margin, PEN_LAT_HOLDER, rotx, rotz,
                         tool_offset)
    from .validate import check_pose

    lat = PEN_LAT_HOLDER if pen_lat is None else float(pen_lat)
    W, H = sheet
    b = np.asarray(spec.xy, float)
    u = np.array([W / 2, H / 2]) - b
    u = u / max(float(np.linalg.norm(u)), 1e-9)
    Twb_inv = np.linalg.inv(spec.T_world_base(h_inv))
    off = tool_offset(pen_lat=lat)
    for r in (0.55, 0.62, 0.48, 0.70, 0.40):
        xy = np.clip(b + r * u, [0.05, 0.05], [W - 0.05, H - 0.05])
        best = None
        for phi in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            R = rotz(phi) @ rotx(np.pi)
            T_w = np.eye(4)
            T_w[:3, :3] = R
            T_w[:3, 3] = np.array([xy[0], xy[1], hover]) - R @ off
            for q7 in ik.Q7_GRID:
                for q in ik.solve(Twb_inv @ T_w, q7, spec.q_seed):
                    m = joint_margin(q)
                    if m < 0.30:
                        continue
                    rep = check_pose(q, spec, h_inv=h_inv, pen_lat=lat)
                    if not rep["ok"] or rep["worst"]["tip_z"] <= 0.0:
                        continue
                    key = min(m, 2.5 * metrics.sigma_min(
                        metrics.tip_jacobian(q)))
                    if best is None or key > best[0]:
                        best = (key, q, xy, rep)
        if best is not None:
            return best[1], best[2], best[3]
    raise RuntimeError(f"no certified ready pose for arm {spec.arm_id}")


# ===========================================================================
# THE v1 LAYOUT — kept for the honest comparison, SUPERSEDED
# ===========================================================================
# The 2026-08-25 v1 study's fine-stage winner: 99.38 % union strict-GO at
# 2 cm with the 15-degree cone, 50.31 % >= 2-arm — WITH NO MOUNTING HARDWARE
# MODELLED.  Two floor arms side by side off the SOUTH short edge, four
# inverted arms in two transverse pairs at y ~ 1.40 and y ~ 2.87.  Re-scored
# with the mounts active (docs/LAYOUT_STUDY.md v2 §1) it is 99.28 % / 48.04 %
# — the union barely moves, the REDUNDANCY loses 2.27 pp, and every cell of
# the loss falls on the two floor arms, whose neighbour's pedestal sits at
# the pen's own working level just off the web edge.
LAYOUT_V1 = dict(
    floor=[(1.15672, -0.14343), (0.55872, -0.13209)],
    inv=[(0.56596, 1.41294), (1.23186, 1.39667),      # south transverse pair
         (0.54082, 2.85289), (1.19379, 2.88897)],     # north transverse pair
    h=0.850,
)

# ===========================================================================
# THE PROPOSED LAYOUT — the v2 winner (scripts/layout_study.py)
# ===========================================================================
# ALL SIX ARMS HANG.  The 0 + 6 family beat 2 + 4 at every stage once the
# mounts were real, and it is not close: 99.98 % vs 99.37 % union, 55.98 % vs
# 47.82 % >= 2-arm, 13.95 % vs 1.94 % >= 3-arm, 3 dead cells vs 89 (2 cm,
# tilt <= 15, mounts ACTIVE).  The reason is structural, not incidental: a
# FLOOR arm must stand outside the web, so a third of its annulus lands off
# the canvas and its pedestal — the only mount hardware at the pen's own
# working height — eats the web edge its neighbours want to draw.  A hanging
# arm sits OVER the canvas and spends its whole annulus on paper, and its
# hardware lives at z >= h where no drawing pose can reach it.
#
# The figure is a regular 2 x 3 grid and the build should set it out with a
# tape measure, not copy six survey numbers: `paired_grid` scores 99.98 % —
# indistinguishable from the search's own best (99.99 %) — with all six
# certified ready poses clear of every neighbour's steel by >= 0.34 m.
#
#   h = 0.850 m       (beats 0.922 by 0.12 pp union / 1.8 pp overlap, 1.00
#                      by 0.73 pp / 11.2 pp — the lateral tool's annulus is
#                      widest here)
#   x = 0.5967, 1.2067    (canvas centre line +/- PAIR_SPACING/2)
#   y = 0.6051, 1.8153, 3.0255    (H/6, H/2, 5H/6)
#
# NOT modelled, and the redesign must still answer them: the ceiling grid's
# own cross-members, paper transport, cable routing, and inter-arm collision
# between six arms whose workspaces now overlap on 55.98 % of the canvas.
LAYOUT_PROPOSED = paired_grid(spacing=PAIR_SPACING, rows=3, h=0.850)


def _proposed():
    return build_fleet(LAYOUT_PROPOSED)


FLEET_PROPOSED = _proposed()
