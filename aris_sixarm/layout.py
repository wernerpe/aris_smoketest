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

The `inv` rows above are the 2026-08-25 probe, taken with NO boxes and the
pre-audit capsules.  The shipped height's row is re-measured off the real
six-arm atlas under the corrected collision model (`out/atlas_proposed_h0940`,
every arm's certified strict-GO cell, radius from its own base):

    inv      0.940      -                   [0.13, 0.86]

The disc model the coarse search covers the canvas with is exactly these
annuli.  `FLEET_PROPOSED` at the bottom is the study's winner, env-selectable
(`ARIS_RIG=proposed`), and NOT the default.
"""
from dataclasses import dataclass, field

import numpy as np

from . import mounts
from .fleet import ArmSpec, Z_FLOOR_BASE
from .frames import roty, rotz
from .rig_final6 import SHEET_FINAL6

# the measured strict-GO annuli, LATERAL tool (see module docstring)
PROFILES_LAT = {
    ("floor", None): (0.34, 0.90),
    ("inv", 0.850): (0.20, 0.84),
    ("inv", 0.922): (0.16, 0.82),
    # MEASURED off the shipped atlas, not off the box-free probe the other
    # rows come from — the tightest arm's inner lip and outer lip over all six
    # (per-arm inner lips run 0.057-0.130; the common annulus is what a
    # pair-spacing argument may use).
    ("inv", 0.940): (0.13, 0.86),
    ("inv", 1.000): (0.16, 0.78),
}
PROFILES_INLINE = {
    ("floor", None): (0.46, 0.80),
    ("inv", 0.850): (0.32, 0.68),
    ("inv", 0.922): (0.38, 0.66),
    ("inv", 1.000): (0.26, 0.64),
}

# canvas m, surveyed legacy floor base plate top.  THE SAME OBJECT as the
# legacy rule `ArmSpec.T_world_base` applies to a floor mount, so a study
# spec's explicit z cannot drift from the height the legacy branch gave it.
Z_FLOOR = Z_FLOOR_BASE
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

    Since 2026-08-25 `mount_boxes` also carries the neighbours' BASE COLUMNS
    (`mounts.arm_column_boxes`) — the arms themselves, in the only 0.333 m of
    them that is the same in every pose.  On this rig that is not a detail:
    a transverse pair sits 0.61 m apart, so each arm's partner stands inside
    the ink the empty-air atlas was handing it.
    """
    mount_boxes: tuple = field(default=(), compare=False, repr=False)

    @property
    def unit(self):
        return "study"        # fleet.sheet_for -> SHEET_FINAL6

    def static_obstacles(self):
        """Every OTHER arm's mount hardware and base column.  Own mount and
        own column excluded by construction — the arm is bolted to the one
        and IS the other (fleet.ArmSpec convention)."""
        return list(self.mount_boxes)


def study_spec(arm_id, mount, xy, h=0.922, name=None, mount_boxes=(),
               q_ready=None):
    """One study arm, carrying its base pose EXPLICITLY (z and R).

    WHY EXPLICIT, AND NOT THROUGH `h_inv` (2026-08-25).  v2 built these specs
    with `z=None, R=None` and let `ArmSpec.T_world_base`'s legacy branch put
    an inverted base at whatever `h_inv` the caller passed.  Every layout-study
    call site passes `h_inv=LAYOUT_PROPOSED["h"]`, so the study was right — but
    that argument DEFAULTS to `H_INV_DEFAULT = 1.00`, and the generic draw
    pipeline (planner, sequence, coordination, idle, scene_check, viz) never
    passes it.  A bare `T_world_base()` therefore hung the proposed fleet 15 cm
    above where it is bolted, silently: the whole rig planned at h = 1.00 while
    its own mount boxes stayed at 0.850.

    So the pose lives in the spec, where the rest of the package can read it
    without being told a height it has no way to know:

        floor   z = Z_FLOOR_BASE (the legacy surveyed plate top), R = rotz(yaw)
        inv     z = h,                                R = roty(pi) @ rotz(yaw)

    These are EXACTLY the transforms the legacy branch computed for the same
    inputs, so every certified study number reproduces; `T_world_base`'s
    explicit-R branch simply ignores `h_inv` from now on — which is the same
    contract the final rig has always had (`tests/test_final_rig.py`).  The
    consequence a caller must know: an arm's height is now decided when the
    SPEC is built, so asking about another height means building another spec.

    `q_ready` is the pose the arm PARKS in — `ArmSpec.q_seed`, which is both
    the IK seed and the home the sequencer flies back to.  `None` keeps the
    mount's legacy default, which is right for the study's coverage numbers
    (the seed does not move an atlas cell: the analytic solver enumerates all
    four branches independently of it, only `ik.solve_cc` reads it) and WRONG
    for anything that flies the arm — see `certified_park_poses`.
    """
    if mount == "floor":
        # face the canvas centre: reach is yaw-invariant, but the seed pose
        # and q1 limits prefer the work in front of the arm
        c = np.array([SHEET_FINAL6[0] / 2, SHEET_FINAL6[1] / 2])
        yaw = float(np.arctan2(c[1] - xy[1], c[0] - xy[0]))
        z, R = Z_FLOOR_BASE, rotz(yaw)
    else:
        yaw = 0.0
        z, R = float(h), roty(np.pi) @ rotz(yaw)
    return StudySpec(arm_id, name or f"{mount}{arm_id}", mount,
                     (float(xy[0]), float(xy[1])), yaw, True,
                     COLORS.get(arm_id, (0.5, 0.5, 0.5)),
                     z=float(z), R=tuple(np.asarray(R, float).flatten()),
                     q_ready=None if q_ready is None else tuple(
                         np.asarray(q_ready, float).reshape(7)),
                     mount_boxes=tuple(mount_boxes))


def arm_ids(layout):
    """-> (floor ids, inverted ids) for a layout of either family.  The six
    physical arms in a fixed order, floor arms taking 13/17 first."""
    nf = len(layout.get("floor", ()))
    fids = FLOOR_IDS[:nf]
    iids = tuple(a for a in ALL_IDS if a not in fids)
    return fids, iids


def build_fleet(layout, mount_model=mounts.MOUNTS, with_mounts=True,
                q_park=None):
    """A layout dict -> {arm_id: StudySpec}, each carrying the OTHER arms'
    schematic mount hardware as static obstacle boxes.

    layout = dict(floor=[(x, y) x nf], inv=[(x, y) x (6 - nf)], h=0.922)
    `with_mounts=False` reproduces the v1 GREEN-FIELD specs exactly (no
    boxes AT ALL — no hardware and no neighbour bodies) — that is how the
    study re-scores v1's winner honestly.
    `q_park` is {arm_id: q} of park poses (`certified_park_poses`); `None`
    leaves every arm on its mount's legacy default seed, which is what the
    COVERAGE study wants and what nothing that FLIES the arm can use.
    """
    h = float(layout.get("h", 0.922))
    fids, iids = arm_ids(layout)
    q_park = {} if q_park is None else dict(q_park)
    bare = {}
    for aid, xy in zip(fids, layout["floor"]):
        bare[aid] = study_spec(aid, "floor", xy, h=h, q_ready=q_park.get(aid))
    for aid, xy in zip(iids, layout["inv"]):
        bare[aid] = study_spec(aid, "inv", xy, h=h, q_ready=q_park.get(aid))
    if not with_mounts:
        return bare
    return {aid: study_spec(s.arm_id, s.mount, s.xy, h=h,
                            q_ready=q_park.get(aid),
                            mount_boxes=mounts.obstacles_for(aid, bare, h,
                                                             mount_model))
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
# [0.13, 0.86] at the shipped h, so for a partner to cover the whole of an
# arm's r < 0.13 under-base hole the two bases must be at least 0.26 apart
# (the far lip of the hole must clear the partner's inner radius) and at most
# 0.73 (the near lip must stay inside the partner's outer radius).  The
# search landed at ~0.61, which was near the TOP of the window the 0.850
# annulus implied (0.40-0.64); raising the rig opens the window at both ends
# and 0.61 is now comfortably inside it.  Rows sit at the centres of an even
# tiling of the canvas length, which is what puts them 1.21 m apart here.
#
# WHICH MATTERS MORE THAN IT USED TO (2026-08-26).  The window is now wide
# enough to hold 0.65 and 0.70, and the re-certification measured what the
# extra 4 cm buys: at h = 0.940 a pitch of 0.65 lifts union strict-GO from
# 90.70 % to 91.95 %, takes the best CSAIL placement from 1.18 x 0.90 m at
# 97.6 % allocated to 1.35 x 1.03 m at 100 %, and lifts the shipped
# programme's coverage from 22.6 % to 26.1 %.  It does not change the verdict
# below — nothing conducts more than one arm at a time at any pitch tested —
# but it is the cheapest thing on the table and it is a BUILD SHEET number,
# so it is measured here and decided by the rig owner, not changed here.
#
# So the build does not need the search's coordinates; it needs a spacing and
# a row count.  `paired_grid` is that layout in round numbers.
PAIR_SPACING = 0.61       # m, transverse pair separation (see above)
PAIR_WINDOW = (0.26, 0.73)  # m, spacings that keep a partner over the hole


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
# metres out along the bearing, IN PREFERENCE ORDER: the first radius that
# certifies anything is the one used, so this ladder is "how far out a ready
# pose would like to stand", not a search.  `certified_park_poses` overrides
# it per arm, because a DEPOT wants a different radius than a ready pose does.
READY_RADII = (0.55, 0.62, 0.48, 0.70, 0.40)


def certified_ready_pose(spec, h_inv=None, hover=0.10, sheet=SHEET_FINAL6,
                         pen_lat=None, bearing=None, radii=None):
    """A gated READY pose for one study arm with the LATERAL tool.

    `h_inv` IS NO LONGER WHAT DECIDES THE HEIGHT — `study_spec` writes the base
    pose into the spec, so the spec is the truth and `None` is the honest
    default.  A height passed here must AGREE with the spec's own, and a
    disagreement raises rather than being silently ignored: this function used
    to be the one place a caller could ask about a different height, and it
    stops being so quietly enough to hide a fifteen-centimetre error.  To ask
    about another height, build the spec at that height.

    `pen_lat` defaults to the lateral holder EXPLICITLY rather than to the
    process-global ACTIVE tool: this is a study function, the study's tool is
    settled, and a caller must not have to mutate `frames.PEN_LAT` (a global
    that leaks into every other module) just to ask for a ready pose.

    Hover `hover` m over a comfortable point of the arm's own annulus, on the
    ray towards the canvas centre — or along `bearing`, a fixed (dx, dy)
    direction in the canvas frame, which is what `certified_park_poses` hands
    in: six arms that all reach for the middle of the canvas end up INSIDE
    each other (see there).  Scans 8 tool yaws x the q7 grid x all IK branches —
    with the lateral holder phi is a REAL DOF, so a hover pinned to phi = 0 can
    be unreachable at a spot the arm covers at another phi.  Keeps the best
    min(margin, 2.5 sigma) among poses that pass `validate.check_pose` with the
    PEN TIP above the paper — the check that caught the legacy inverted ready
    pose dipping 16 mm under at this h.

    -> (q (7,), hover xy (2,), validate report).  Raises if none is certified.

    Lives here rather than in the scene script because the layout study's fine
    stage must certify the SAME pose it draws (docs/LAYOUT_STUDY.md v2 §ready
    poses): one implementation, two callers.
    """
    from . import ik, metrics
    from .frames import (joint_margin, PEN_LAT_HOLDER, rotx, rotz,
                         tool_offset)
    from .validate import check_pose

    if h_inv is not None and abs(float(h_inv) - float(spec.z)) > 1e-12:
        raise ValueError(
            f"arm {spec.arm_id} is built at base z = {spec.z}, not "
            f"{float(h_inv)}: `study_spec` carries the pose now, so a "
            "different height means a different spec, not a different "
            "argument")
    lat = PEN_LAT_HOLDER if pen_lat is None else float(pen_lat)
    W, H = sheet
    b = np.asarray(spec.xy, float)
    u = (np.array([W / 2, H / 2]) - b if bearing is None
         else np.asarray(bearing, float).reshape(2))
    u = u / max(float(np.linalg.norm(u)), 1e-9)
    Twb_inv = np.linalg.inv(spec.T_world_base(h_inv))
    off = tool_offset(pen_lat=lat)
    for r in READY_RADII if radii is None else tuple(np.atleast_1d(radii)):
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
                    # `pen_lat=lat`, NOT the process global.  This is the
                    # ranking key, and a study function that took its tool
                    # explicitly everywhere except here would rank the
                    # candidates by the ACTIVE tool's wrist — which silently
                    # picked a different pose for arm 31 depending on whether
                    # ARIS_TOOL happened to be set.
                    key = min(m, 2.5 * metrics.sigma_min(
                        metrics.tip_jacobian(q, pen_lat=lat)))
                    if best is None or key > best[0]:
                        best = (key, q, xy, rep)
        if best is not None:
            return best[1], best[2], best[3]
    raise RuntimeError(f"no certified ready pose for arm {spec.arm_id}")


def certified_park_poses(fleet, grid=None, hover=0.10, sheet=SHEET_FINAL6,
                         pen_lat=None, clear=None):
    """Where the six arms WAIT. -> {arm_id: q (7,)}, one certified pose each.

    THE PARK POSE IS NOT DECORATION AND IT IS NOT INHERITED.  `spec.q_seed` is
    the home the sequencer flies back to, the pose `paper.route` folds through
    on a long transit, and the configuration every arm holds for the whole of
    every phase it is not drawing in — so it is in the conductor's collision
    images from t = 0 to the end.  The all-ceiling rig cannot use the mount
    default: at h = 0.850 with the lateral holder the legacy `Q_READY_INV`
    puts the pen tip 61.6 mm BELOW the paper and its joint margin at 0.184,
    under the 0.30 gate — SIX arms parked inside the table.

    BEARING, AND WHY IT WAS OUTWARD.  `certified_ready_pose` aims each arm at
    the canvas centre, which is right for one arm and catastrophic for six
    hung over the same canvas: on this layout it parks the fleet in a huddle
    and the closest pair (13, 17) INTERPENETRATES by 95.6 mm.  So each arm was
    sent along its own base's bearing AWAY from the fleet centroid — the one
    direction that is different for every arm and that no two of them share.
    For `paired_grid` the centroid is the canvas centre, so that is exactly
    "each arm stands off towards its own nearest rim".

    ...AND WHY IT IS SEARCHED NOW.  Outward is a rule that keeps the fleet off
    ITSELF, and the criterion that decides whether a phase can be conducted is
    a different one: how far the parked chain stands from every OTHER arm's
    certified INK.  On the proposed rig the outward rule saturates against it —
    the middle row's outward ray runs off the short edge of a 1.80 m canvas, so
    every radius past ~0.55 m lands on the same clipped point and the best that
    set can do is 75 mm against the 80 mm the conductor asks.  A bearing is a
    free variable, so `grid` may now name one: `(radius, hover, bearing_deg)`,
    an ABSOLUTE direction in the canvas frame.  Given three numbers the arm is
    sent that way; given two it still goes outward, and `grid=None` is the old
    recipe exactly.

    HOW FAR OUT, AND HOW HIGH, IS MEASURED — because a DEPOT IS NOT A READY
    POSE.  `certified_ready_pose` ranks on min(margin, 2.5 sigma), which is
    what a pose held under load wants; a park pose is held under no load at
    all and its whole job is to be the node every tour starts and ends at.  So
    `grid` is {arm_id: (radius, hover)} chosen on THAT: the fraction of the
    arm's own certified drawing cells it can fly to (`writing.enter_beats`)
    and back from (`exit_beats`), over 6 radii x 3 hovers x 24 cells per arm.
    Ranking on the static key instead put every arm at r = 0.55 / 0.10 and
    left 62 % of entries flyable; ranking on the job gives 92 %, and the two
    arms it helps most are the two that were nearly stranded — 71 goes from
    33 % to 92 %, 17 from 55 % to 96 %.  That is not a tidiness argument: on
    the first CSAIL run every execution profile died at "go-home ... cannot
    clear the paper plane", which is the conductor's own escape hatch (fall
    back to conductor v1 and send everybody home) failing because home was
    somewhere the arm could not fly to.

    `grid=None` falls back to the `READY_RADII` ladder at `hover` for every
    arm, which is what the ready-pose recipe does.

    AND THEN THE FLEET IS CHECKED AGAINST ITSELF, because six individually
    certified poses are not a certified fleet.  `certified_ready_pose` knows
    about one arm; the bearing keeps the six apart by construction and NOT by
    proof, and the proof is cheap.  It is also not academic: the ladder at
    `hover = 0.20` picks a set that leaves 4 mm between two arms — every pose
    gated, every pose fine, the fleet unflyable.  `clear` m is the floor
    (default `coordination.SAFETY_M + CALIB_M`, the margin the conductor holds
    every pair to); a fleet under it RAISES rather than being handed back.
    """
    from .coordination import ArmPath, clearance_matrix, SAFETY_M, CALIB_M
    clear = SAFETY_M + CALIB_M if clear is None else float(clear)
    grid = {} if grid is None else dict(grid)
    cent = np.mean([np.asarray(s.xy, float) for s in fleet.values()], axis=0)
    out = {}
    for aid, spec in sorted(fleet.items()):
        got = tuple(grid.get(aid, (None, hover)))
        r, hv = got[0], got[1]
        bear = (np.asarray(spec.xy, float) - cent if len(got) < 3 else
                np.array([np.cos(np.deg2rad(got[2])),
                          np.sin(np.deg2rad(got[2]))]))
        out[aid] = certified_ready_pose(
            spec, hover=hv, sheet=sheet, pen_lat=pen_lat,
            radii=None if r is None else (float(r),), bearing=bear)[0]
    paths = {aid: ArmPath(aid, q[None, :], 0.05, spec=fleet[aid])
             for aid, q in out.items()}
    ids = sorted(out)
    for x, i in enumerate(ids):
        for j in ids[x + 1:]:
            d = float(np.min(clearance_matrix(paths[i], paths[j])))
            if d < clear:
                raise RuntimeError(
                    f"arms {i} and {j} park {1000 * d:.1f} mm apart, under the "
                    f"{1000 * clear:.0f} mm the conductor holds every pair to: "
                    f"the six poses are individually certified and the FLEET "
                    f"is not (hover {hover})")
    return out


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
#   h = 0.940 m       (was 0.850; re-certified 2026-08-26 under the audited
#                      capsules — see the note below, and README)
#   x = 0.5967, 1.2067    (canvas centre line +/- PAIR_SPACING/2)
#   y = 0.6051, 1.8153, 3.0255    (H/6, H/2, 5H/6)
#
# NOT modelled, and the redesign must still answer them: the ceiling grid's
# own cross-members, paper transport, cable routing, and inter-arm collision
# between six arms whose workspaces now overlap on 55.98 % of the canvas.
# THE HEIGHT WAS RE-DECIDED ON MEASURED GEOMETRY (2026-08-26).  0.850 came
# out of a study whose collision model the mesh audit then disproved; every
# number that chose it moved.  Re-swept under the corrected capsules and the
# measured column, at 2 cm over all six arms:
#
#     h        union strict-GO   >=2 arms   sigma p1   park-vs-ink (best)
#     0.850        87.00 %        37.14 %    0.1513         10 mm
#     0.925        91.10 %        34.43 %    0.1516         61 mm
#     0.940        90.70 %        33.85 %    0.1468         72 mm
#
# 0.940 is not the union-coverage optimum — 0.925 beats it by 0.4 pp — and it
# is here anyway, because it is the only one of the three at which the CSAIL
# logo produces a certified programme at all.  "Park-vs-ink" is why: the best
# park set the (radius, hover) search can find, measured against every other
# arm's certified drawing poses at the conductor's own 0.08 m.  It rises
# monotonically with height and reaches 0.072 m at 0.940, and a phase whose
# parked partners are inside its ink is a phase no schedule can fix.  At
# 0.850 and 0.925 nothing conducts; at 0.940 one solo phase does.
#
# READ THAT AS A FLOOR, NOT A RESULT.  22.6 % of the logo, one arm, is worse
# than the pre-audit programme claimed and better than the corrected 0.850
# rig's nothing.  The blocking constraint is not the height and not the
# margin (dropping calib to zero changes nothing): it is that the two
# MIDDLE-ROW arms have nowhere over a 1.80 m canvas to stand that is not in
# somebody's ink.  See README for what would actually move it.
LAYOUT_PROPOSED = paired_grid(spacing=PAIR_SPACING, rows=3, h=0.940)

# RE-SEARCHED AT THE SHIPPED HEIGHT, ON THE OTHER CRITERION, AND OVER THE
# BEARING TOO (2026-08-26, after the base column became four measured bands).
# A depot's own flyability is what the 0.850 rig tuned for; under the audited
# capsules the criterion that decides whether a phase can be CONDUCTED at all
# is the other one — how far the parked chain stands from every OTHER arm's
# certified ink, at the conductor's own 0.08 m.  The grid in force maximises
# that, searched over 7 radii x 10 hovers x 12 bearings per arm against the
# banded-column atlas, with the neighbours' steel and body columns as a hard
# gate (`rig_final.chain_static_clearance >= STATIC_MARGIN`) and the pose's
# own min(margin, 2.5 sigma) breaking ties inside each 5 mm bucket:
#
#      arm   (r, hover, bearing)     ink    entry   home   was (0.940 ship)
#        2   (0.55, 0.35, +134.1)   97.3 mm  20/24  20/24   19/24  97.3 mm
#       13   (0.62, 0.20, -134.1)   98.4 mm  18/24  18/24   17/24  98.4 mm
#       17   (0.62, 0.20,  -45.9)   97.6 mm  14/24  14/24   13/24  97.6 mm
#       31   (0.62, 0.30, +150.0)   98.0 mm  18/24  18/24   18/24  80.9 mm
#       71   (0.30, 0.55,  -60.0)   97.7 mm  18/24  18/24   10/24  61.3 mm
#       97   (0.62, 0.20,  +45.9)   97.7 mm  16/24  16/24    9/24  97.7 mm
#
# 97.3 mm against the 80 mm the conductor asks, park-vs-park at the clearance
# matrix's 250 mm clip, and 277 mm to the nearest neighbour's steel or body
# column on every one of them.  The set DOMINATES the one it replaces on both
# criteria at once, arm for arm.
#
# TWO CRITERIA, AND THE SECOND ONE IS THE TIE-BREAK BECAUSE THE FIRST IS A
# PLATEAU.  Every candidate in the top ink bucket clears by 97-98 mm — a parked
# arm's own base column is pose-invariant, so the layout, not the pose, is what
# sets that ceiling.  On a plateau the depot's own job decides: `entry` and
# `home` above are how many of 24 farthest-point-sampled certified cells the
# arm can `writing.enter_beats` its way into and `exit_beats` its way out of.
# The 0.940 ship set was chosen before the bearing was a variable and pays for
# it twice — arm 71 could reach 10 of its 24 cells and stood 61 mm from arm
# 31's ink.  THE BEARING IS WHAT BOUGHT IT.  Held to the
# outward ray the same search tops out at 75 mm — 3 mm better than the 72 mm
# the flat column gave and still under the gate — because the middle row's
# outward ray runs off the short edge of a 1.80 m canvas: `certified_ready_pose`
# clips the hover point into the sheet, so past ~0.55 m every candidate lands
# on the same clamped xy and the radius saturates.  A bearing does not
# saturate.  Arms 31 and 71 now stand at (0.06, 2.13) and (1.74, 1.51),
# diagonally off their own bases instead of straight out into the rim.
#
# AND THERE IS A CEILING, WHICH THIS IS AT.  Every winner clears by 97-98 mm
# and so do the five candidates behind it: a parked arm's own BASE COLUMN is
# pose-invariant, so no park pose can put it further from a neighbour's ink
# than the layout already does.  The atlas gates every cell against those
# column boxes at `STATIC_MARGIN`, which by the gate-consistency identity is
# exactly the conductor's 80 mm — and the measured slack on top of it is
# 17-18 mm.  Park-vs-ink has stopped being the binding constraint on this rig.
#
# The historical sweep, kept because its numbers are still the reason the
# GRID has the shape it has:
#
# WHERE EACH ARM WAITS, AND WHY THERE.  `(radius, hover)` per arm, on the
# outward bearing, MEASURED as the depot it has to be rather than picked as
# the ready pose it looks like: for each of 6 radii x 3 hovers, how many of
# that arm's own 24 sampled certified drawing cells it can fly to
# (`writing.enter_beats`) and back from (`exit_beats`).  The winners, entry
# and go-home out of 24, against what the plain ready-pose ladder
# (r = 0.55, hover = 0.10) scored:
#
#      arm   r     hover   entry   home        ladder entry
#       13   0.40  0.10    20/24   20/24        78 %
#       17   0.30  0.10    23/24   23/24        55 %
#       31   0.30  0.20    23/24   23/24        75 %
#       71   0.48  0.20    22/24   22/24        33 %   <- nearly stranded
#        2   0.48  0.10    24/24   24/24        90 %
#       97   0.48  0.20    20/24   20/24        43 %
#
# 92 % of entries flyable against the ladder's 62 %.  The two arms it rescues
# are the two the first CSAIL run could not get home: every execution profile
# refused at "go-home at segment N cannot clear the paper plane".  The shipped
# grid sits in the same radius band (0.55-0.62) and the same hover band
# (0.20-0.30) as those winners, which is not a coincidence — it is why the
# conditioning tie-break was worth applying.
#
# ...AND A DEPOT MUST ALSO NOT STAND ON SOMEBODY ELSE'S INK (2026-08-26).  A
# park pose is held for the whole of every phase its arm is not drawing in, so
# it is also an OBSTACLE for the whole of that phase, and on a rig whose
# transverse pairs are 0.61 m apart that is the tighter of the two
# constraints.  On the flat-column model the fleet's worst was 20 mm (arm 97
# lying across the north half with arm 2's ink under it) and then 72 mm after
# the height went up; the whole fleet refused to conduct on it.  It is 97.3 mm
# now, and `allocate.ParkProbe` prunes against these poses at allocation time
# so a span inside one of them never reaches the conductor at all.
PARK_GRID_PROPOSED = {2: (0.55, 0.35, 134.1), 13: (0.62, 0.20, -134.1),
                      17: (0.62, 0.20, -45.9), 31: (0.62, 0.30, 150.0),
                      71: (0.30, 0.55, -60.0), 97: (0.62, 0.20, 45.9)}

# THE PARKED FLEET: `certified_park_poses(build_fleet(LAYOUT_PROPOSED),
# PARK_GRID_PROPOSED)`, baked the way `frames.Q_READY_*` are baked and for the
# same two reasons — an operator types these into Desk, and importing a rig
# should not re-solve six IK searches.  `tests/test_layout.py` re-derives them
# and compares, so the literals cannot drift from the recipe that made them.
#
# RE-DERIVED 2026-08-26 at h = 0.940 from the grid above, against the
# BANDED-column atlas.  Every one of them passes `validate.check_pose` with the
# corrected capsules and the measured column, clears the neighbours' steel and
# body columns by 277 mm, and stands 97-98 mm off every other arm's certified
# ink.  They hover 0.20-0.30 m over the paper — back down where the 0.850 rig
# parked, because the bearing search no longer has to buy clearance with
# height.
#
# SEEDS, NOT MEASUREMENTS, like every other pose in this repo that no arm has
# yet held: re-derive by Desk fine-adjust once the ceiling grid exists.
Q_PARK_PROPOSED = {
    2:  (-0.1933, 1.2642, 1.2767, -2.0492, 1.7237, 1.1764, -1.7795),
    13: (0.6776, 1.0744, -1.5633, -2.1141, -2.0017, 1.3162, 0.5932),
    17: (-0.6777, -1.0400, -1.5916, -2.1168, 2.0272, 1.2889, 0.9886),
    31: (-0.5776, 1.1469, 1.2019, -1.7054, 1.9749, 1.1800, 2.1750),
    71: (-0.1637, -1.4052, -2.4461, -2.5861, 0.8489, 1.0011, -1.7795),
    97: (0.6776, -1.0744, 1.5783, -2.1141, -2.0017, 1.3162, 0.5932),
}
# where each of them holds the pen (canvas m), for the log and the scene
PARK_HOVER_PROPOSED = {
    13: (0.165, 0.160), 17: (1.638, 0.160),     # hover 0.20 m
    97: (1.638, 3.471),                         # hover 0.20 m
    31: (0.060, 2.125),                         # hover 0.30 m
    2:  (0.214, 3.421),                         # hover 0.35 m
    71: (1.357, 1.556),                         # hover 0.55 m
}


def _proposed():
    return build_fleet(LAYOUT_PROPOSED, q_park=Q_PARK_PROPOSED)


FLEET_PROPOSED = _proposed()
