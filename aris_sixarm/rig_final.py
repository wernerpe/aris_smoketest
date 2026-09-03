"""FINAL RIG geometry — extracted from the authoritative drawing (2026-08).

Sources (raw_slack_file_dump/):
  PDF  "Drawing installation, 3 arms, 1 up, 1 side, 1 down, sizes cm.pdf"
       (1 page, Vectorworks, front + top views, dimensions in cm)
  DXF  "D.I., 3 arms, 1 up, 1 side, 1 down, sizes cm.dxf"
       (AC1032, $INSUNITS=5 => centimeters, two complete 3D model copies;
       the FRONT copy is authoritative, the top copy cross-checks it)
Extraction + cross-check: docs/FINAL_RIG.md.  Every AABB below is an exact
8-vertex box from the DXF unless its source string says otherwise.

FRAMES
  W ("world", drawing): origin = outer front-left corner of the installation
     at floor level, +X right, +Y back, +Z up.  Units here: cm (as drawn).
  C ("canvas", planning): origin = front-left corner of the PAPER top surface,
     axes parallel to W, units m.  z=0 is the paper plane — the same
     convention every planner module already uses.
       C = (W - PAPER_ORIGIN_W_CM) / 100

CONSERVATISM
  Collision boxes ENCLOSE the drawn member (padding never shrinks).  Where the
  two DXF model copies disagree (arm-2 boom bottom: z 155.07 vs 152.37) the
  union is used.  Curved bodies (paper feed roll) have point-sampled extents
  padded outward and are flagged in docs/FINAL_RIG.md.
"""
import numpy as np

# --- the paper (drawing surface) -----------------------------------------
# solid 40 (front copy): 180.34 x 170.00 x 0.2 cm web on the tabletop
PAPER_ORIGIN_W_CM = (21.246, 26.748, 63.668)   # W, cm: paper corner, TOP face
SHEET_FINAL = (1.8034, 1.700)                  # m, drawable web on the table
PAPER_THICK_CM = 0.2                           # top 63.668, tabletop 63.468
TABLETOP_TOP_W_CM = 63.468

# --- arm mounts (J1 axis poses, W frame, cm / rad) -----------------------
# The drawing NAMES the arms (MTEXT): "Three robot arms: 13 floor,
# 31 left - upside down, 2 right side position" — the same physical arms as
# the legacy six-arm registry.
# Each entry: J1-axis intersection with the mounting plane + base orientation.
# "front" of a Franka base plate = the 8.78 cm-offset side of the 22.58 cm
# plate edge (inference from base asymmetry + the three depicted poses;
# FLAGGED in docs/FINAL_RIG.md — check against the real plate before ship).
ARM_MOUNTS_W = {
    # upright on the tabletop, front strip, in front of the paper.
    # plate X 99.789-118.789, Y 0.268-22.868, z 63.468-64.738 (1.27 plate).
    # J1 axis: plate center X, 13.8 from plate front edge.  NO red dimension
    # references this arm (model geometry only; both DXF copies agree).
    "up": dict(
        p_w_cm=(109.289, 14.068, 64.738),      # mounting plane = plate top
        axis="+Z", front="+Y",                 # front faces the paper
        source="DXF plate solid Groep-66/137 + 13.8 marker; no dimension"),
    # upside-down under the central double top beam.
    # plate X 31.931-54.513, Y 142.926-161.926, z 155.868-157.138;
    # the arm hangs from the plate BOTTOM face z=155.868.
    "down": dict(
        p_w_cm=(45.737, 152.426, 155.868),
        axis="-Z", front="+X",                 # front toward table center
        source="dim 45,7 (X); dim/text 55,8 (Y); plate solid 83 + marker 86"),
    # side-mounted on a vertical plate clamped to the right boom.
    # plate X 182.230-183.500 (1.27 thick), Y 142.971-161.971,
    # z 132.486-155.068; the arm mounts on the LEFT face x=182.230.
    "side": dict(
        p_w_cm=(182.230, 152.471, 141.268),    # J1 axis horizontal, along -X
        axis="-X", front="-Z",                 # front faces straight down
        source="dims 34,9 / 55,8 / 91,6; plate solid Groep-72/143; "
               "z = plate top 155.068 - 13.8 (ring solid says 141.279)"),
}

# base rotations (W axes).  Verified against the DXF quats in the extraction:
#   up:   quat (0,0,0.70711,0.70711)  = rotz(pi/2)
#   down: quat (1,0,0,0)              = rotx(pi)   (= roty(pi) @ rotz(pi))
#   side: quat (0.70711,0,-0.70711,0) = roty(-pi/2) @ rotz(pi)
def _rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _roty(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


ARM_IDS = {"up": 13, "down": 31, "side": 2}     # drawing MTEXT labels

ARM_R_W = {
    "up": _rotz(np.pi / 2),
    "down": np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]]),  # rotx(pi)
    "side": _roty(-np.pi / 2) @ _rotz(np.pi),
}


def arm_base_canvas(key):
    """-> (p_m (3,), R (3,3)): base pose in the CANVAS frame, meters."""
    m = ARM_MOUNTS_W[key]
    p = (np.array(m["p_w_cm"]) - np.array(PAPER_ORIGIN_W_CM)) / 100.0
    return p, ARM_R_W[key].copy()


# --- structure: conservative AABBs, W frame, cm --------------------------
# (name, (x0,y0,z0), (x1,y1,z1), source, tag)
#   tag: "" plain structure | "mount:<key>" = that arm's own mount hardware
#        (excluded from ITS OWN base-column check, nothing else)
_B = lambda name, lo, hi, source, tag="": dict(  # noqa: E731
    name=name, lo=tuple(lo), hi=tuple(hi), source=source, tag=tag)

FRAME_BOXES_W_CM = [
    # everything below the tabletop as ONE enclosing block: legs, feet, rails
    # (top z 53.31-60.93 + bottom z 5.05-12.67), 45-deg corner braces, and the
    # tabletop plate itself.  Arms may never be below the paper plane anyway.
    _B("table_block", (-0.4, -0.4, 0), (218.84, 208.68, 63.468),
       "encloses legs/feet/rails/braces/top plate + levelling-foot pads "
       "(pads overhang the leg lines by 0.33; verified)"),
    # cage corner posts, tabletop to the very top
    _B("post_FL", (0, 0, 60.93), (7.62, 7.62, 233.65), "7.62 sq posts"),
    _B("post_FR", (210.82, 0, 60.93), (218.44, 7.62, 233.65), "7.62 sq posts"),
    _B("post_BL", (0, 200.66, 60.93), (7.62, 208.28, 233.65), "7.62 sq posts"),
    _B("post_BR", (210.82, 200.66, 60.93), (218.44, 208.28, 233.65),
       "7.62 sq posts"),
    # the whole top band as one slab: perimeter beams + central double beam
    # (Y 144.831-160.071) + 4 connector plates (z 229.84-233.65)
    _B("top_slab", (0, 0, 226.03), (218.50, 208.28, 233.65),
       "encloses all top beams + connector plates + device-R mount rails "
       "(Groep-49 drawn 0.04 outboard; verified)"),
    # central double beam hangs BELOW nothing (it is inside the slab) but the
    # arm booms and gussets do hang below it:
    # corner brace plates at each post, z 205.71-226.03 (8 plates, 2 per post)
    _B("brace_FL", (0, 0, 205.71), (27.94, 27.94, 226.03),
       "20.32x20.32x3.81 plates, enclosed with post corner"),
    _B("brace_FR", (190.50, 0, 205.71), (218.44, 27.94, 226.03),
       "20.32x20.32x3.81 plates, enclosed with post corner"),
    _B("brace_BL", (0, 180.34, 205.71), (27.94, 208.28, 226.03),
       "20.32x20.32x3.81 plates, enclosed with post corner"),
    _B("brace_BR", (190.50, 180.34, 205.71), (218.44, 208.28, 226.03),
       "20.32x20.32x3.81 plates, enclosed with post corner"),
    # --- arm-31-style boom (the "down" arm) --------------------------------
    _B("down_boom_W", (23.48, 144.83, 152.37), (31.20, 160.07, 226.06),
       "west vertical beam pair straddling the central beam", "mount:down"),
    _B("down_boom_E", (55.25, 144.83, 152.37), (62.95, 160.07, 226.06),
       "east vertical beam pair straddling the central beam", "mount:down"),
    _B("down_gusset_W", (3.16, 144.83, 205.71), (23.56, 160.07, 226.06),
       "vertical gusset 20.32x3.81x20.32 (Y band taken conservative)",
       "mount:down"),
    _B("down_gusset_E", (62.83, 144.83, 205.71), (83.27, 160.07, 226.06),
       "vertical gusset 20.32x3.81x20.32 (Y band taken conservative)",
       "mount:down"),
    _B("down_clamps", (31.93, 144.83, 157.14), (54.53, 160.07, 166.71),
       "clamp stack above the plate (two levels, union)", "mount:down"),
    _B("down_plate", (31.931, 142.926, 155.868), (54.513, 161.926, 157.138),
       "base plate 22.582x19.0x1.27", "mount:down"),
    # --- arm-2-style boom (the "side" arm) ---------------------------------
    _B("side_boom", (187.28, 144.83, 152.37), (194.95, 160.07, 226.07),
       "vertical beam pair; bottom z is the UNION of the two DXF copies "
       "(155.07 front / 152.37 top copy) - FLAGGED", "mount:side"),
    _B("side_gusset", (194.92, 144.83, 205.71), (215.32, 160.07, 226.06),
       "vertical gussets 20.32x3.81x20.32 (Y band taken conservative)",
       "mount:side"),
    _B("side_clamps", (183.50, 144.87, 132.49), (187.31, 160.07, 155.07),
       "clamp blocks (two z levels 132.49-139.62 / 147.94-155.07, union)",
       "mount:side"),
    _B("side_bracket", (181.93, 144.87, 155.07), (187.33, 160.07, 160.83),
       "bracket above the plate top", "mount:side"),
    _B("side_plate", (182.230, 142.971, 132.486), (183.500, 161.971, 155.068),
       "vertical base plate 1.27 thick", "mount:side"),
    # --- the "up" arm's own plate ------------------------------------------
    _B("up_plate", (99.789, 0.268, 63.468), (118.789, 22.868, 64.738),
       "base plate 22.582x19.0x1.27 on the tabletop", "mount:up"),
    # --- paper transport ---------------------------------------------------
    _B("feed_roll", (-0.15, 23.5, 62.5), (21.35, 200.0, 83.1),
       "feed roll ~O20.2 + end flanges R7.0 reaching X=0.0 (verified) + "
       "brackets; +X face kept at the measured 21.2+0.15"),
    _B("guide_rods", (210.3, 23.8, 63.4), (215.6, 197.7, 67.35),
       "3 guide rods R~0.76 + O2 couplers (X-lo 210.37, z-hi 67.30, "
       "coupler runs to the crank and winder; verified)"),
    _B("winder_box", (209.57, 197.61, 63.47), (216.57, 200.61, 68.47),
       "winder box 7x3x5"),
    _B("crank_motor", (209.38, 9.38, 63.47), (216.38, 23.82, 71.34),
       "crank/motor block"),
    # the paper web itself, rising off the table over the guide rods to the
    # winder (solid 5D2, X 201.586-213.994, up to z 66.33): a physical surface
    # a pen holder must not plough through
    _B("paper_curl", (201.5, 26.7, 63.4), (214.0, 196.8, 66.4),
       "paper web rising over the guide rods (verified, solid 5D2)"),
    # --- 8 hung devices (cameras or lights - purpose not stated) -----------
    _B("device_F1", (61.0, -1.0, 216.8), (85.2, 13.7, 226.58),
       "body + plate + knob/lens details protruding 4.92 inboard "
       "(verified; box extended inboard to Y 13.7)"),
    _B("device_F2", (133.4, -1.0, 216.8), (157.6, 13.7, 226.58), "as F1"),
    _B("device_B1", (61.0, 194.5, 216.8), (85.2, 209.28, 226.58), "as F1"),
    _B("device_B2", (133.4, 194.5, 216.8), (157.6, 209.28, 226.58), "as F1"),
    _B("device_L1", (-1.0, 27.9, 216.8), (13.7, 52.1, 226.58), "as F1"),
    _B("device_L2", (-1.0, 100.3, 216.8), (13.7, 124.5, 226.58), "as F1"),
    _B("device_R1", (204.7, 27.9, 216.8), (219.5, 52.1, 226.58), "as F1"),
    _B("device_R2", (204.7, 100.3, 216.8), (219.5, 124.5, 226.58), "as F1"),
]


def frame_boxes_canvas(exclude_tag=None, zmin=0.0, boxes=None):
    """Structure boxes in the CANVAS frame, meters.

    -> list of dict(name, lo (3,), hi (3,), source, tag).
    exclude_tag: drop boxes whose tag equals it ("mount:down" etc.) — used for
    an arm's own base-column check only, never for other arms.
    zmin: boxes that never rise above this canvas z are dropped.  The default
    0.0 drops exactly the below-paper structure (table_block, whose top face
    is the tabletop 2 mm under the paper): the z >= Z_PAPER plane gate already
    forbids that half-space, and checking margins against a slab 2 mm under
    the pen would veto every legitimate drawing pose.  URDF export passes -10
    to keep everything.
    `boxes`: a W-frame box list to convert INSTEAD of this module's own.  The
    hook a rig VARIANT hangs on — `rig_final6.boxes_with_extended_pole()`
    hands in the same 35 boxes with the side arm's clamped stack slid down a
    lengthened pole — so a variant never has to monkey-patch
    `FRAME_BOXES_W_CM`, which is the drawing and stays the drawing.
    """
    o = np.array(PAPER_ORIGIN_W_CM)
    out = []
    for b in (FRAME_BOXES_W_CM if boxes is None else boxes):
        if exclude_tag is not None and b["tag"] == exclude_tag:
            continue
        lo = (np.array(b["lo"]) - o) / 100.0
        hi = (np.array(b["hi"]) - o) / 100.0
        if hi[2] < zmin:
            continue
        out.append(dict(name=b["name"], lo=lo, hi=hi,
                        source=b["source"], tag=b["tag"]))
    return out


# --- static clearance policy (docs/DECISIONS.md, FINAL RIG table) --------
# Distinct from the INTER-ARM margin (two moving arms + schedule slop,
# safety 0.05 + calib 0.03): against STATIC steel the operating term follows
# the established static-surface convention (Z_PAPER = 0.02 against the
# tabletop), plus the same 0.03 calibration term — the drawing is a plan, not
# an as-built survey, and the drawing itself warns arm 2's mount is "not
# stiff and stable".  Shrink CALIB_STATIC per-rig the day a survey lands.
Z_STATIC = 0.02
CALIB_STATIC = 0.03
STATIC_MARGIN = Z_STATIC + CALIB_STATIC

# ...AND WHAT A PRODUCER MUST KEEP, WHICH IS MORE (2026-08-26).
#
# `STATIC_MARGIN` is what the CHECKER compares against — and what it compares
# is not this quantity.  `scene_check.static_clearance_lb` is a deliberately
# independent derivation (see its docstring: "the two code paths share nothing
# but the geometry itself"), and an independent derivation of a minimum is a
# LOWER BOUND with slack in it.  Two pieces of slack, both of them explicit
# there and neither of them optional:
#
#   0.010 m  it does not minimise along a capsule analytically the way
#            `segment_box_clearance` does — it SAMPLES the segment every
#            <= 0.02 m and subtracts `|b - a| / (2 (K - 1))`, which is half a
#            step: up to 10 mm on a 0.4 m link.
#   0.00275  and then the 1-Lipschitz residual between two samples of the
#            TRAJECTORY, 0.55 * FRAME_STEP, after its own refinement.
#
# So a configuration whose exact clearance is 50.0 mm reads 37 to 47 mm at the
# checker, and a timeline that met every gate it was given is refused by a
# checker that is right.  THE PRODUCERS HAVE TO PAY BOTH, or the checker is not
# a second opinion but a lottery.
#
# That is not a hypothesis.  On 2026-08-26 it refused, on one number each, a
# solo phase (47.4 mm), a THREE-arm phase and a SIX-arm phase — every one of
# them a monotone schedule the conductor had already found, with 83-88 mm of
# inter-arm clearance and every other gate passing by tens of millimetres.  The
# ink under them measures 103 mm exactly; the pen-up over them measured 58.8.
# It was never a collision.  It was a gate ordering that had been inverted
# since the frame boxes were introduced.
#
# 13 mm, restated here rather than imported: `scene_check` may not depend on
# this module and this module may not depend on it, so the number is written
# twice and `tests/test_paper.py` pins it against the checker's own constants.
# It costs the tightest ~2 % of certified cells and every pen-up that only just
# cleared, which is the difference between a certified programme and a
# programme that is certified until somebody measures it.
#
# The CHECKERS keep `STATIC_MARGIN`: `validate.validate_plan` and `scene_check`
# are second opinions and must stay independent of what a planner chose to
# spend.  So does `atlas.solve_cell`, whose gate is baked into every swept
# atlas and into `atlas.model_signature`; an atlas is now an OPTIMISTIC
# prefilter by up to 13 mm, which costs probe time and not safety —
# `probe_stroke` plans for real and the planner refuses.
STATIC_SEG_SLACK = 0.010     # scene_check's capsule-sampling half-step
STATIC_SWEEP_SLACK = 0.00275  # ...and its trajectory 1-Lipschitz residual
STATIC_SWEEP_PAD = 0.013     # >= the sum, rounded up
STATIC_PLAN_MARGIN = STATIC_MARGIN + STATIC_SWEEP_PAD

# capsules for static checks: the conductor's chain topology (a test pins
# them against coordination.CAPSULES) MINUS the base column's bands — the base
# is bolted to its mount by construction, and its capsule radius would
# false-positive against the very plate it is bolted to.  Radii mirror
# coordination.UPPER_R / ELBOW_R / FORE_R / WRIST_R / HAND_R, which since
# 2026-08-26 are MEASURED against the manufacturer's meshes rather than the
# three round numbers this table used to restate — see the long note by
# `coordination.LINK_R` for what moved and by how much.  Restated as literals
# here on purpose: if someone widens a capsule there and forgets this table,
# the pinning test is supposed to notice.
PEN_R_FINAL = 0.05    # pen capsule radius, FINAL rig: the UNION envelope of
                      # both holder builds (10-deg: tip 8 mm off axis; 23-deg
                      # clutch at 0.209 flange->tip: 45 mm off axis + pencil)
STATIC_CAPSULES = ((1, 3, 0.130), (3, 4, 0.117), (4, 5, 0.131),
                   (5, 7, 0.091), (7, 8, 0.104), (8, 9, PEN_R_FINAL))

# LATERAL HOLDER (2026-08-25): the tool is an L — an 11 cm bracket along hand
# x, then the pen down to the tip.  A single TCP->tip capsule would need
# r ~ 0.078 + pen radius to cover the L's corner (the corner sits lat/sqrt(2)
# off the diagonal), so the tool is TWO capsules through the corner point
# instead: TCP->corner (the bracket) and corner->tip (the pen), both at the
# same conservative 0.05 envelope — the holder has no CAD in this
# configuration, so the radius is deliberately generous.  Chain layout:
# frames.fk's 9 points + tip (index 9) + bracket corner (index 10) — see
# frames.tool_points_many.  `chain_static_clearance` selects the table by the
# chain's own width, so a caller cannot pair the wrong tool with its points.
# AUDITED AND KEPT (2026-08-26).  The 22-deg CAD landed after these two were
# chosen, and `scripts/collision_audit.py` measured the assembled holder
# (housing + cap + clutch, placed by `penholder22_T_hand`) against them: both
# capsules CONTAINED it by 1.720 mm, so unlike every arm capsule they did not
# have to move.  The inference in that placement is the residual risk, not the
# radius.
#
# ...AND THAT AUDIT WAS RUN ON A HOUSING MOUNTED END-FOR-END (2026-09-03).
# `penholder22_T_hand` had the housing's +X pointing away from the tip; with it
# the right way round the 55.099 mm of barrel that used to reach toward the
# paper reaches BEHIND the grip instead, and the same measurement now says:
#
#     housing + cap                +6.546 mm out  (bracket r 0.0565 needed)
#     + the pencil tail (7c)      +77.661 mm out  (bracket r 0.1277 needed)
#
# SO THESE TWO CAPSULES DO NOT CONTAIN THE HOLDER ANY MORE.  They are kept at
# 0.05 all the same: they are what every certified number in this repo was
# earned against, and re-deriving them is a re-certification with its own
# gate, not a constant edit.  What it would cost is measured rather than
# guessed — at r = 0.0565 the 100 % programme's inter-arm minimum falls from
# 80.73 mm to 79.12 and fails the 80 mm gate; at 0.1277 it falls to -18.21.
# docs/SYSTEM_MODEL.md 7c.  THIS IS THE REPO'S LARGEST OPEN RE-CERT ITEM ON
# THE TOOL, and it is the pencil tail that makes it expensive.
BRACKET_R_LAT = 0.05  # bracket capsule radius (see the note above: it no
                      # longer contains the holder, and it has not moved)
PEN_R_LAT = 0.05      # pen capsule radius on the lateral holder
STATIC_CAPSULES_LAT = ((1, 3, 0.130), (3, 4, 0.117), (4, 5, 0.131),
                       (5, 7, 0.091), (7, 8, 0.104),
                       (8, 10, BRACKET_R_LAT), (10, 9, PEN_R_LAT))

# the holder as URDF tool geometry (visual mesh extracted from the SolidWorks
# CAD, panda_hand frame; collision cylinder = the same union envelope)
TOOL = dict(
    parent="panda_hand",
    visual_mesh="meshes/penholder_rig10_panda_hand_frame.stl",
    collision=dict(type="cylinder", radius=PEN_R_FINAL, z0=-0.033, z1=0.210),
    source="Pen holder cad(1).zip: 10-deg natural-hold assembly (complete); "
           "collision is the union with the 23-deg clutch build extended to "
           "the 0.209 m flange->tip reading",
)


# ===========================================================================
# THE 22-DEG CLUTCH HOLDER AS CAD (2026-08-25 delivery)
# ===========================================================================
# Source: raw_slack_file_dump/"Pen holder all parts 2026.08.19"/ — eight
# printed parts as STL + SLDPRT pairs and NO ASSEMBLY FILE FOR THIS BUILD.
# Units in the STLs are MILLIMETRES (housing 80.1 x 33.78 x 50.0 mm);
# everything below is metres.
#
# THE OTHER DELIVERY DOES HAVE AN ASSEMBLY, and it is what the internal stack
# below is read off.  raw_slack_file_dump/"Pen holder cad(1).zip" carries a
# nested "Natural hold assembly - closed.zip" holding
# "Natural hold assembly - closed.SLDASM": the COMPLETE 10-deg build, seven
# components, mated, with two FR3 fingertips in it.  Its architecture is this
# build's architecture part-for-part (same 40.1 x 21.0 mm sleeve, same cap
# thread, same 50 mm mount post), so it fixes the axial ORDER that eight loose
# parts cannot.  See `penholder22_internals` for the order and its proof, and
# docs/SYSTEM_MODEL.md 7 for the tip-transform verdict it also settles.
#
# WHAT THE HOUSING ACTUALLY IS (measured, not guessed — the numbers come out
# of `scripts/extract_penholder22_meshes.py`, which prints every one of them):
#
#   * a BARREL along its own +X with a through bore: OD 27.2 mm, bore
#     21.148 mm the whole way to the threaded end, necking to a 17.00 mm land
#     over x in [0.00055, 0.0027] at the TAIL.  The step between them is a
#     chamfer bottoming at x = 0.00315 and reaching full bore at x = 0.0036; a
#     flat 21.0 mm face lands on it at x = 0.003340, which is
#     `stack_front_x` and the TAIL STOP of everything inside.  x = 0 is the
#     TAIL, not the nose: the 17.00 land is the SPRING's stop (17.00 will not
#     pass a 19.05 spring, which is what it is cut for) and THE PEN LEAVES
#     THROUGH THE CAP, at x = `cap_end_x` = 0.0851.  See `penholder22_stack`
#     and docs/SYSTEM_MODEL.md 7c.  (An earlier reading called that 17.0
#     "exactly the clutch's 17.07 OD, so the clutch seats in the nose".  It
#     does not: 17.07 does not enter 17.00, the assembly puts the clutch 45 mm
#     further back, and the pen leaves at the OTHER end.  A second earlier
#     reading kept the right verdict in prose and still mounted the housing
#     end-for-end; that is FIXED, 2026-09-03 — see `penholder22_T_hand`.)
#   * an external THREAD at the far end, x in [0.072, 0.0801], OD 28.5 mm,
#     onto which "pen holder cap v20250903" screws (its threaded recess is
#     6 mm deep, matching);
#   * a MOUNT POST across the barrel: a 26 x 26 mm square section, 50 mm long
#     along the housing's own +Z, centred on the bore at x = 0.05513, with an
#     18 x 18 x 7 mm square socket in each end.
#
# THE POST IS THE GRIP, AND THE ASSEMBLY NOW PROVES IT.  In the 10-deg
# assembly the two FR3 fingertips are MATED to the housing (four Coincident
# and two Parallel mates name both), the post axis lands on the finger-travel
# axis to 0 deg, and each fingertip's 18.116 x 18.116 mm block sits 7.000 mm
# inside a post socket.  So the post axis IS y_hand and the grip centre is
# where the post axis crosses the bore.
#
# THE JAW GAP IS 36.0 mm, NOT 57.  The seating faces are the socket FLOORS:
# 50 mm of post MINUS 2 x 7 mm of socket is 36.000, and the assembly puts the
# two fingertip grip faces 36.0008 mm apart.  This housing's own sockets
# measure the same (floors at post z = 7.00 and 43.00).  The 57 mm this
# comment used to carry added the 3.5 mm of fingertip left OUTSIDE each socket
# instead of subtracting the 7 mm inside it; 57 mm would hold nothing, because
# it is 7 mm wider than the post is long.  Both planes are real — grip faces
# at +/-18.000, fingertip BACK faces at +/-28.500 — and `FINGER_FIX` wants the
# first, because the URDF's finger mesh includes its own tip.
#
# "22 DEG" IS A CLOCKING, NOT A TILT — AND IT MEASURES 23.00.  The post's four
# flats (and its sockets) are rotated 23.00 +/- 0.00 deg about the POST axis
# relative to the bore direction; the post axis itself is exactly perpendicular
# to the bore.  Mounted on the hand, that clocking is a rotation about y_hand,
# i.e. the pen leans 23 deg out of tool-z — which is the same 23.0 deg
# docs/FINAL_RIG.md already recorded for this build and already flagged against
# the file's "22 deg" name.  (Independent check off this STL: the socket's
# half-width along the bore direction is 9.78 mm = 9.0 / cos 23.03 deg.)
#
# AND THE 10-DEG ASSEMBLY SAYS THAT CLOCKING *IS* THE TILT.  Same naming
# convention ("10 d natural" / "22 deg"), same post-perpendicular-to-bore, same
# fingertips-in-sockets — and when the assembly is resolved the bore comes out
# 10.0000 deg off the hand's approach axis and 90.0000 deg off finger travel,
# with the grip centre 103.26 mm from panda_hand against the stock TCP's 103.4.
# The file's number is the lean, the grip centre is the TCP, and neither is a
# coincidence at four decimal places.
#
# ...AND 23 DEG IS NOT WHAT THE PLANNER USES.  `frames`' lateral tool puts the
# tip at TCP + R @ (0.110, 0, 0.110): a lean of 45 deg, 0.15556 m from the TCP.
# The planning transform is GATE-VALIDATED and stays truth (`PEN_LAT_HOLDER`,
# docs/DECISIONS.md), so the holder is still DRAWN along the planner's ray —
# but the assembly has removed the place the 22 deg of difference used to be
# parked.  There is no cradle: the fingertips seat square in the post's own
# sockets, so the clocking reaches the hand undivided.  Consequences, all in
# docs/SYSTEM_MODEL.md 7:
#   1. at 23 deg and the gate-validated 0.110 m of axial depth the tip is
#      0.0467 m lateral, not 0.110 — 63.3 mm of tip position;
#   2. the grip-to-exit length is 30.001 mm — grip to the CAP'S OUTER FACE,
#      which is where the pen leaves (25.001 mm to the housing's own end face,
#      plus the 5.000 mm the cap stands proud of it).  Reaching 155.563 mm
#      therefore needs 125.562 mm of graphite past the cap, against 89.499 mm
#      at 23 deg and the 10-deg assembly's own measured 17.000;
#   3. the SIGN of the lateral offset is a mounting choice, not a CAD fact:
#      the post is square, so the holder seats in the sockets either way up
#      and the lean is +/-23 deg.
#
# THE HOUSING USED TO BE MOUNTED END-FOR-END, AND IS NOT ANY MORE (2026-09-03).
# `penholder22_T_hand` pointed the housing's +X AWAY from the tip, which put
# the tail land 55.099 mm toward the paper and the cap 30.001 mm back toward
# the wrist.  It is the other way round: the 55.099 mm of barrel is BEHIND the
# grip and the pen leaves through the cap 30.001 mm in FRONT of it.  What that
# costs is a re-certification and it is written down rather than absorbed —
# docs/SYSTEM_MODEL.md 7c.

# the pencil tail's own length, named because `tail_cylinder` is derived from
# it and a dict literal cannot refer to itself.  See PENHOLDER22["tail_len"].
_TAIL_LEN = 0.072514

PENHOLDER22 = dict(
    parent="panda_hand",
    source='raw_slack_file_dump/"Pen holder all parts 2026.08.19"/ '
           '(STL, mm, no SLDASM — placement inferred; see docstring above)',
    # --- housing, in its own frame (metres) ---
    bore_yz=(0.016900, 0.025000),     # bore axis, housing (y, z)
    tail_x=0.0,                       # the housing's TAIL face: the 17.0 mm
                                      # land that stops the spring, and where
                                      # the pencil's tail leaves (7c).  The
                                      # pen leaves at the OTHER end, cap_end_x
    thread_x=(0.072000, 0.080100),    # external thread for the cap
    barrel_r=0.013585,                # measured max OD/2 over x in [0, 0.036]
    thread_r=0.014272,                # measured max OD/2 over the thread
    cap_end_x=0.085100,               # the cap's outer face, along the bore —
                                      # AND THIS IS WHERE THE PEN LEAVES (7c)
    post_xy=(0.055099, 0.016894),     # post axis, housing (x, y)
    post_z=(0.0, 0.050000),           # post extent along the housing's +Z
    post_side=0.026000,               # square section
    post_clock=0.401426,              # 23.00 deg, MEASURED (file says "22")
    lead_r=0.003500,                  # clutch bore/2 = the graphite stick
    lead_r_coll=0.005000,             # conservative envelope for the stick
    # --- the internal stack, all MEASURED off the same STL delivery ---
    stack_front_x=0.003340,           # where a flat 21.0 face lands on the
                                      # tail chamfer: the stack's TAIL stop
                                      # (x = 0 is the tail; see 7c)
    stack_back_x=0.083115,            # the cap's 16.0 mm shoulder, from the
                                      # cap's own z = 0.001985 at cap_end_x:
                                      # the stack's BACK stop
    bore_r=0.010574,                  # the 21.148 mm through bore
    spring=dict(part="9657K26_Compression Spring", od=0.019050, id=0.014970,
                wire=0.002040, free=0.050810, coils=14, solid=0.028500),
    sleeve=dict(part="pen holder for clutches v1.00", length=0.040100,
                od=0.021000, bore_small=0.015030, bore_large=0.016290,
                pusher_land=(0.013000, 0.003000)),
    clutch=dict(part="pen clutch - Creatcolor monolith graphite v1.01",
                length=0.035000, od=0.017066, bore=0.007000),
    spacers=(0.005100, 0.010100),     # the shim set: 5 mm and 10 mm, 21.0 OD
    spacer_fitted=0.0,                # WHICH ONE IS IN.  ASSUMED: none.
    # --- THE PENCIL TAIL.  The pen does not stop at the tail land: it goes
    # STRAIGHT THROUGH and out the back, and the 10-deg assembly measures how
    # far.  Resolved (scripts/read_solidworks.py asm), its Conte pencil spans
    # assembly z 1000.121 .. 1174.735 mm, the cap's outer face is at 1017.121
    # and the housing's tail face at 1102.221 — so 17.000 mm of sharpened
    # point stands proud of the cap and 72.514 mm of blunt, flat-cut tail
    # stands proud of the tail face.  (Housing + cap measure 85.100 mm on that
    # build and 85.100 mm on this one, so the two barrels are the same length
    # and the overhang transfers part-for-part.)  MEASURED off the assembly;
    # what is ASSUMED is that this build's stick is long enough to show the
    # same overhang — see `penholder22_tail` for the arithmetic that implies.
    tail_len=_TAIL_LEN,               # how far the tail stands proud of x = 0
    # its own primitive envelope, fitted the way `env_cylinders` are: an
    # (x0, x1, r) coaxial with the bore whose radius is the largest distance
    # the body reaches off that axis, with the same 1 mm of axial lead-in the
    # first band carries at this end.  The body is itself a cylinder of radius
    # `lead_r`, so `lead_r_coll` is the same conservative envelope `pen_lead`
    # already carries and the escape is negative by construction — proved, not
    # asserted, in `penholder22_internals_escape`.
    tail_cylinder=(-_TAIL_LEN - 0.001, 0.0, 0.005),
    # --- the cap, in its own frame ---
    cap_xy=(0.018000, 0.018000),      # bore axis in the cap's own (x, y)
    cap_seat_z=0.005000,              # recess bottom: meets the housing end
    # --- the decimated, hand-frame visual meshes (see extract script) ---
    visual_meshes=("meshes/penholder22_housing_hand.obj",
                   "meshes/penholder22_cap_hand.obj"),
    # --- THE COLLISION ENVELOPE: three cylinders COAXIAL WITH THE BORE, as
    # (x0, x1, radius) in the housing's own frame.  Each radius is the largest
    # distance any housing OR cap vertex reaches from the bore axis inside
    # that band — raw CAD and decimated mesh both — so the union encloses the
    # visual by construction, and the extract script re-proves it every run.
    # A tighter box for the mount post was tried and REJECTED: a rotated
    # square's x-extent grows with its side, so enlarging it to swallow the
    # reinforcing gussets at x ~ 0.036 only drags in more bare barrel.  The
    # middle cylinder is fat (r 0.030) because the 50 mm post genuinely
    # reaches that far off the bore at its ends.
    env_cylinders=((-0.001000, 0.034000, 0.013600),
                   (0.034000, 0.074000, 0.030100),
                   (0.074000, 0.085500, 0.018100)),
    # --- parts DELIBERATELY not placed ---
    omitted=("pen holder spacer 5 mm", "pen holder spacer 10 mm",
             "pen clutch - extractor"),
    omitted_why="the two spacers are a SHIM SET, not a stack: the bore holds "
                "the sleeve plus exactly one of {none, 5 mm, 10 mm} and the "
                "third choice sets the spring preload (11.1 / 16.2 / 21.2 mm "
                "of deflection against 22.3 mm to solid — fitting both "
                "spacers would ask 26.3 mm and coil-bind).  `spacer_fitted` "
                "picks one; the model draws none, which is ASSUMED.  The "
                "extractor is a bench tool: a 30 mm head on a 12.0 x 40 mm "
                "pusher rod that goes down the sleeve's 13.0 mm land to push "
                "the clutch out of its taper, which is what that land is FOR.",
)


def penholder22_T_hand(pen_ext, pen_lat, d_hand_tcp):
    """(4,4) panda_hand <- housing placement, and the same for the cap.

    -> (T_hand_housing, T_hand_cap, exit_along_bore, tip_along_bore).

    THE PLACEMENT IS INFERRED (no assembly file for THIS build).  Three
    assumptions, each of them the only one the parts support:

      1. the mount post's axis is y_hand — the fingers plug into its two end
         sockets, and 50 mm of post + 2 x 3.5 mm engagement is exactly the
         57 mm jaw gap the 28.5 mm finger half-width gives;
      2. the grip centre — where the post axis crosses the bore — sits at the
         hand TCP, the stock grasp point;
      3. the bore points along the PLANNER's ray from the TCP to the pen tip,
         normalize(pen_lat, 0, pen_ext), rather than along the housing's own
         23 deg clocking.  Assumption 3 is what makes the drawing consistent
         with the gate-validated tool transform; see PENHOLDER22's docstring
         for what it costs.

    ...AND THE HOUSING'S +X POINTS AT THE TIP, WHICH IT DID NOT UNTIL
    2026-09-03.  This function used to set `Xh = -u`, i.e. it mounted the
    housing END-FOR-END: the 17.00 mm tail land 55.099 mm toward the paper
    and the cap 30.001 mm back toward the wrist.  Four things say the pen
    leaves through the CAP and the model is now the way they say (each one
    re-derived from `Natural hold assembly - closed.SLDASM` itself, not from
    prose — docs/SYSTEM_MODEL.md 7c):

      * the assembly's own preview shows the sharpened point out of the cap
        and 72.514 mm of blunt tail out the far end (`tail_len`);
      * resolved, the assembly's parts run cap -> sleeve -> SPRING -> tail
        face along its bore, so the spring pushes the pen assembly TOWARD the
        cap and paper force compresses it.  Mounted the other way round the
        tool has no compliance at all;
      * the 17.00 mm land will not pass the 19.05 mm spring and passes a 7 mm
        stick without touching it: it is the spring's stop, not the pen's;
      * the grip sits 25.001 mm from the housing's cap end and 55.099 mm from
        its tail, and `docs/FINAL_RIG.md`'s independent extraction read the
        same 25 mm.

    It is a 180 deg rotation about the post axis and NOTHING ELSE: the post
    axis is still y_hand, the grip centre is still the TCP, the bore is still
    the planner's ray and the pen tip does not move by a picometre.  What
    moves is the housing BODY, from 30.001 mm behind the grip to 55.099 mm
    behind it — see `penholder22_collision` for what that costs the envelopes.
    """
    d = np.array([float(pen_lat), 0.0, float(pen_ext)])
    reach = float(np.linalg.norm(d))
    u = d / reach                                 # TCP -> tip, unit
    P = PENHOLDER22
    Xh = u                                        # housing +X points AT the
                                                  # tip: the pen leaves the CAP
    Zh = np.array([0.0, 1.0, 0.0])                # post axis == finger travel
    Yh = np.cross(Zh, Xh)
    R = np.column_stack([Xh, Yh, Zh])
    grip = np.array([P["post_xy"][0], P["post_xy"][1], P["bore_yz"][1]])
    tcp = np.array([0.0, 0.0, float(d_hand_tcp)])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tcp - R @ grip
    # the cap: its +Z runs back along the housing's -X, its recess bottom
    # (cap z = cap_seat_z) seated on the housing's threaded end face
    Rc = np.array([[0.0, 0.0, -1.0],              # housing <- cap
                   [1.0, 0.0, 0.0],
                   [0.0, -1.0, 0.0]])
    Tc = np.eye(4)
    Tc[:3, :3] = Rc
    Tc[:3, 3] = np.array([P["cap_end_x"], P["bore_yz"][0], P["bore_yz"][1]]) \
        - Rc @ np.array([P["cap_xy"][0], P["cap_xy"][1], 0.0])
    return T, T @ Tc, P["cap_end_x"] - P["post_xy"][0], reach


def penholder22_hull():
    """Every (x0, x1, r) the tool's collision model is made of, housing frame.

    THE ORDER IS PART OF THE CONTRACT: the three `env_cylinders` that the
    housing and cap were fitted to come first, and the pencil tail's own
    envelope last.  A caller that wants to prove something about the HOUSING
    — `scripts/extract_penholder22_meshes.py` does — takes the first three and
    says so; a caller that wants the tool's collision geometry takes them all.
    """
    P = PENHOLDER22
    return tuple(P["env_cylinders"]) + (P["tail_cylinder"],)


def penholder22_collision(pen_ext, pen_lat, d_hand_tcp):
    """The holder's CONSERVATIVE primitive envelope, in the panda_hand frame.

    -> [("cylinder", T (4,4), (radius, length))], the cylinder's axis being
    its own frame's z, as URDF wants it.

    A 7 000-triangle concave printed part is not a collision geometry, and the
    final rig's convention for exactly this problem is a primitive envelope
    (`TOOL["collision"]`, one cylinder).  This one is FOUR coaxial cylinders
    (`penholder22_hull`): the three `PENHOLDER22["env_cylinders"]` the housing
    and cap were fitted to, whose radii are measured maxima rather than
    guesses and which `scripts/extract_penholder22_meshes.py` re-proves on
    every run, plus `tail_cylinder` for the length of pencil that stands out
    of the tail face (7c).

    WHAT THE 2026-09-03 FLIP COSTS HERE, because it is a re-certification and
    not a render.  These cylinders used to reach 30.001 mm behind the grip and
    now reach 55.099 mm behind it, and the tail reaches 127.613 mm behind it.
    Measured against the two lateral tool capsules `STATIC_CAPSULES_LAT`
    ships — the envelope every gate in this repo actually plans against — the
    raw CAD escapes them:

        housing + cap, as mounted before   -1.720 mm   (contained)
        housing + cap, mounted correctly   +6.546 mm  (bracket r 0.0565 needed)
        + the pencil tail                 +77.661 mm  (bracket r 0.1277 needed)

    NOTHING HERE WIDENS A CAPSULE.  `BRACKET_R_LAT` / `PEN_R_LAT` are the
    radii the certified programme was gated at and moving them re-opens every
    number that was earned with them; the escape is REPORTED — in
    docs/SYSTEM_MODEL.md 7c, in the manifest, and by
    `scripts/collision_audit.py --part tool` on every run — and it is the
    re-certification item, not this function's to absorb.
    """
    T_h, _, _, _ = penholder22_T_hand(pen_ext, pen_lat, d_hand_tcp)
    by, bz = PENHOLDER22["bore_yz"]
    out = []
    for x0, x1, r in penholder22_hull():
        T = np.eye(4)
        T[:3, 3] = (0.5 * (x0 + x1), by, bz)
        # the cylinder's own z must run along the housing's x
        T[:3, :3] = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])
        out.append(("cylinder", T_h @ T, (r, x1 - x0)))
    return out


def penholder22_stack(spacer=None):
    """The internal stack along the bore -> [dict], housing frame, metres.

    Each row is `dict(name, x0, x1, r, note)` with x measured from the TAIL
    face (x = 0) along the housing's own +X, i.e. TOWARD the cap the pen
    leaves by.  This is the ORDER, and the order is the part of the holder
    eight loose STLs cannot give you.

    WHERE THE ORDER COMES FROM.  The 10-deg "natural hold" assembly in
    raw_slack_file_dump/"Pen holder cad(1).zip" -> "Natural hold assembly -
    closed.zip" is the same architecture built out of the same shapes, and
    resolving its component transforms puts, along its bore from the tail:

        tail shoulder  ->  SPRING  ->  SLEEVE  ->  CAP -> the pen

    with the spring's end coil landing 3.302 mm from the tail face (this
    housing's own shoulder is at 3.34) and the sleeve's face on the spring's
    other coil to 0.1 mm.  In the assembly's own coordinates that reads, along
    its bore: cap 1017.121..1028.558, sleeve 1020.121..1060.221, spring
    1060.121..1098.919, housing 1020.855..1102.221 mm — the spring is at the
    end AWAY from the cap, which is the whole argument in 7c.  Every interface
    below is then a measured fit on THIS delivery's parts:

      * TAIL STOP, AND IT IS THE SPRING'S.  The bore necks to a 17.00 mm land
        at x = 0, and 17.00 will not pass the 19.05 mm spring — that land
        exists to stop it.  Nothing 21 mm gets past `stack_front_x` either.
        The spring clears the 21.148 bore by 1.05 mm and bears on the 21 mm
        parts' end annulus, which the 15.0 mm sleeve bore is cut to match.
        THE PEN LEAVES AT THE OTHER END, through the cap; see 7c.
      * BACK STOP.  The cap is a THREADED COLLAR, not a lid — 36 mm flange,
        11.4 mm long, open right through, with a 21.51 mm counterbore and a
        16.00 mm shoulder.  16.00 is under 21.0, so the shoulder is what
        retains the stack; `stack_back_x` is where it sits once the cap is
        home on the 5.5 mm of external thread.
      * PRELOAD.  stack_back_x - stack_front_x = 79.775 mm and the sleeve is
        40.100, so the spring is squeezed from 50.810 to 39.675 — 11.135 mm
        of preload, and the tip can retract 11.175 mm more before coil bind.
        A spacer takes 5.1 or 10.1 mm of that travel and adds it to preload.
      * THE CLUTCH IS A SPLIT COLLET AND THE SLEEVE IS ITS TAPER.  The
        sleeve's bore is a true cone, 15.030 mm growing to 16.290 over its
        length (0.01747 mm/mm, a surface of revolution to 1 um); the clutch's
        nose cone grows at 0.01750 mm/mm.  Two matched tapers wedge.  The
        clutch's free 17.066 mm does not enter a 16.290 mm hole, which is the
        point: it is slit, and going in closes it onto the 7.0 mm graphite.
        It goes in from the sleeve's LARGE end, so its collet points at the
        13.0 mm land — the same land the 12.0 mm extractor rod comes down —
        and drawing load pushes it deeper, i.e. tighter.

    WHAT IS STILL ASSUMED: which shim is fitted (`spacer_fitted`, drawn as
    none), and that the clutch is seated to the back of the sleeve rather
    than part-way.  Neither moves anything outside the barrel.
    """
    P = PENHOLDER22
    sp = P["spacer_fitted"] if spacer is None else float(spacer)
    if sp and not any(abs(sp - s) < 1e-9 for s in P["spacers"]):
        raise ValueError(f"spacer {sp} is not one of {P['spacers']} (or 0)")
    x0, x1 = P["stack_front_x"], P["stack_back_x"]
    sleeve, spring, clutch = P["sleeve"], P["spring"], P["clutch"]
    sleeve_x0 = x1 - sleeve["length"]           # the sleeve's front face
    spring_x1 = sleeve_x0 - sp                  # the spring's back coil
    free_gap = spring_x1 - x0                   # what the spring squeezes to
    if free_gap <= spring["solid"]:
        raise ValueError(f"spacer {sp} coil-binds the spring: {free_gap:.5f} "
                         f"m against a {spring['solid']} m solid height")
    out = [dict(name="spring", x0=x0, x1=spring_x1, r=spring["od"] / 2,
                note=f"{spring['part']}, free {spring['free']} m, squeezed to "
                     f"{free_gap:.6f} (preload {spring['free'] - free_gap:.6f}"
                     f" m, {free_gap - spring['solid']:.6f} m left to solid). "
                     "DRAWN AS ITS OUTER ENVELOPE, not as a coil.")]
    if sp:
        out.append(dict(name="spacer", x0=spring_x1, x1=sleeve_x0,
                        r=sleeve["od"] / 2,
                        note=f"shim {sp} m of the set {P['spacers']}"))
    out.append(dict(name="sleeve", x0=sleeve_x0, x1=x1, r=sleeve["od"] / 2,
                    note=f"{sleeve['part']}, {sleeve['length']} m, seated on "
                         "the cap's 16.0 mm shoulder"))
    out.append(dict(name="clutch", x0=x1 - clutch["length"], x1=x1,
                    r=clutch["od"] / 2,
                    note=f"{clutch['part']}, inside the sleeve, collet at the "
                         "sleeve's 13.0 mm land"))
    out.append(dict(name="graphite_buried", x0=P["tail_x"], x1=P["cap_end_x"],
                    r=P["lead_r"],
                    note="the 7.0 mm stick inside the barrel, tail face to "
                         "cap face: it runs right THROUGH the holder — the "
                         "clutch, the sleeve's 13.0 mm land, the spring's "
                         "14.97 mm bore and the 17.00 mm tail land, none of "
                         "which a 7 mm stick touches.  What shows past the "
                         "cap is `pen_lead`; what shows past the tail face "
                         "is `penholder22_tail`"))
    return out


def penholder22_tail():
    """The pencil TAIL — the stick where it stands out of the tail face.

    -> `dict(name, x0, x1, r, note)` in the housing frame, the same shape
    `penholder22_stack` returns, spanning x in [-`tail_len`, `tail_x`].

    WHY THERE IS ONE AT ALL.  The 10-deg assembly's pencil is 174.614 mm long
    in an 85.100 mm barrel: it goes STRAIGHT THROUGH and out the back, and its
    own preview draws it that way — the sharpened point 17.000 mm past the cap
    and 72.514 mm of blunt, flat-cut tail past the housing's tail face.  The
    tail is a real body of the assembled tool, it points at the WRIST, and
    until 2026-09-03 nothing in this model had it anywhere.

    MEASURED vs ASSUMED, precisely.  72.514 mm is MEASURED, off the assembly.
    That this build shows the SAME overhang is ASSUMED — and the assumption
    has a price worth writing down: with the gate-validated tip 155.563 mm
    from the TCP, a stick that still shows 72.514 mm of tail is
    72.514 + 55.099 + 155.563 = 283.176 mm long, which no 7 mm graphite stick
    is (a Cretacolor Monolith is 175 mm).  Take the assembly's PENCIL LENGTH
    instead of its overhang and the arithmetic runs the other way: 174.614 mm
    pushed out to that tip ends 36.049 mm INSIDE the barrel and there is no
    tail at all.  The model draws the tail because a body that might be there
    and reaches at the wrist is the conservative half of that pair — and
    because the two readings together are one more way of saying what 7c
    already says, that the planner's 155.563 mm tip is not this housing with a
    short stick in it.
    """
    P = PENHOLDER22
    return dict(name="graphite_tail", x0=-P["tail_len"], x1=P["tail_x"],
                r=P["lead_r"],
                note=f"{P['tail_len'] * 1000:.3f} mm of stick past the tail "
                     "face, MEASURED off the 10-deg assembly (its pencil "
                     "stands 72.514 mm proud of the housing's tail face and "
                     "17.000 mm proud of the cap).  Points at the WRIST; see "
                     "`penholder22_tail` for what its length assumes")


def penholder22_bodies(spacer=None):
    """Every drawn body of the assembled pen — the bore stack AND the tail."""
    return list(penholder22_stack(spacer)) + [penholder22_tail()]


def penholder22_internals(pen_ext, pen_lat, d_hand_tcp, spacer=None):
    """`penholder22_bodies` placed in the panda_hand frame.

    -> [(name, T (4,4), (radius, length), note)], each cylinder's axis being
    its own frame's z, the way URDF wants it — the same convention
    `penholder22_collision` uses, and for the same reason.

    VISUAL ONLY, and that is a claim with a proof rather than a convenience.
    Every bore body lives inside the 21.148 mm bore and so inside
    `env_cylinders`; the tail lives inside `tail_cylinder`, which is in the
    collision model precisely because it is the one body that does NOT.
    Either way, drawing these as collision geometry would add faces and not
    volume — `penholder22_internals_escape` re-proves that on every generator
    run, over the whole hull and every shim setting.
    """
    T_h, _, _, _ = penholder22_T_hand(pen_ext, pen_lat, d_hand_tcp)
    by, bz = PENHOLDER22["bore_yz"]
    out = []
    for b in penholder22_bodies(spacer):
        lo, hi = sorted((b["x0"], b["x1"]))
        T = np.eye(4)
        T[:3, 3] = (0.5 * (lo + hi), by, bz)
        T[:3, :3] = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])
        out.append((b["name"], T_h @ T, (b["r"], hi - lo), b["note"]))
    return out


def penholder22_internals_escape(spacer=None):
    """How far the drawn bodies reach outside the collision hull -> metres.

    0.0 when the complete assembly still fits `penholder22_hull` — the three
    cylinders the housing and cap were fitted to plus the tail's own — which
    is the whole question modelling the internals raises.  Measured on the
    bodies' own corner rings, so a body that pokes out radially OR axially is
    caught.
    """
    hull = penholder22_hull()
    worst = -np.inf
    for b in penholder22_bodies(spacer):
        lo, hi = sorted((b["x0"], b["x1"]))
        for x in (lo, hi):
            d = np.inf
            for c0, c1, r in hull:
                d = min(d, max(b["r"] - r, c0 - x, x - c1))
            worst = max(worst, d)
    return float(max(0.0, worst))


# ===========================================================================
# THE "FAT FRANKA FINGER" — a whole-finger replacement, as CAD (2026-09-02)
# ===========================================================================
# Source: raw_slack_file_dump/"Pen holder all parts 2026.08.19"/
# "Fat Franka Finger v250904.STL" — millimetres, watertight, 8234 faces.  The
# same solid ships as an SLDPRT two levels inside "Pen holder cad(1).zip", and
# THE TWO DO NOT SHARE A DATUM: the STL is exported 10.5000 mm along +y off the
# SLDPRT's own origin (x and z agree to 0.0002 mm; only y moves), and
# `stl_y_shift` undoes that.  The shift is not a guess: see WHICH FRAME,
# below.
#
# WHAT THE PART IS.  A Z-section bracket, 18.4339 x 90.0003 x 50.000 mm:
#
#   * a MOUNTING FOOT at each end of the 90 mm — 8.000 mm thick (x 52.0578 ..
#     60.0578), 20 mm of y, 14 mm of z (14.000 .. 28.000) — each carrying TWO
#     M4 CLEARANCE holes on axes along x at z = 22.000, 12.000 mm apart: a
#     4.296 mm waist over x 55.06 .. 57.06 with a 7.293 mm counterbore 3.0 mm
#     deep from BOTH faces, so the foot bolts up either way round;
#   * a SLANTED WEB, 8 mm thick, face normals +/-(0.8412, 0, -0.5407), rising
#     over z 28 .. 46 and carrying the section from x 52..60 out to x 64..68;
#   * a CONTACT PLATE, 3.8498 mm thick (x 64.0578 .. 67.9076), z 46.000 ..
#     64.000, spanning the full length, with TWO 6.000 mm through-holes at
#     z = 55.000, 69.000 mm apart, and R5 corners;
#   * a RIB along the plate's PROXIMAL edge — the web's own outer face, run
#     past the plate plane and then flat-topped at x = 70.4915 over z 45.509 ..
#     46.000.  It stands 2.5839 mm proud of the contact plane, for the full
#     90 mm, at every station sampled.  It is the innermost thing on the part,
#     and (see below) it is what a BARE blade would touch first.
#
# EVERY FEATURE IS DOUBLED, and that is the design.  The part is mirror-
# symmetric about y = 34.5 to 0.44 mm over everything that mates — both feet,
# all four foot holes, both plate holes, both plate faces — and 99.05 % by
# volume.  The two ends differ ONLY in the plate's outer edge (y = +79.500 at
# one end, -9.000 at the other) and one R5 corner: 1.5045 mm, worst case.  Bolt
# the near foot down and the blade reaches +69 mm along the finger's x; bolt
# the far foot down and it reaches -69 mm.  Those two placements are the mirror
# pair a LEFT and a RIGHT finger need if both blades are to reach THE SAME WAY
# in the hand — which is why one printed part serves both, and why there are
# two of everything.  (A 180-degree rotation about any one axis will not do
# it: the map that swaps the feet is improper, and only the part's own mirror
# symmetry makes it realisable.)
#
# WHICH FRAME — AND THIS IS THE LOAD-BEARING FINDING.  The SLDPRT is drawn in
# the SAME coordinate system as "Franka_Finger_FR3 Fingertip only.SLDPRT", the
# stock tip the 10-deg assembly seats in the post sockets.  Three independent
# things say so, and none of them is a fit:
#
#   1. that fingertip is an 18.1156 mm square block on x 68.0578 .. 78.5578,
#      centred on (y, z) = (0.0004, 55.000);
#   2. the "Rodgers fingertip" sub-assembly presses its one 93514A130 brass
#      insert on the axis (y, z) = (0.0004, 55.000) — and THIS part's plate
#      hole is on (0.0002, 54.9998).  0.0002 mm apart;
#   3. this part's plate occupies exactly the fingertip's own z band and lands
#      0.1502 mm outboard of the fingertip's back face.
#
# So the map into the finger link frame is fixed by placing the FINGERTIP, and
# the fingertip's place is already known (docs/SYSTEM_MODEL.md 7a):
#
#     link x =  cad y                (across the hand; the tip is centred)
#     link y =  0.0785578 - cad x    (jaw axis; cad x 78.5578 IS the grip
#                                     plane, where libfranka's width/2 is)
#     link z =  cad z - 0.0101579    (finger length; the tip's distal face IS
#                                     the finger mesh's own tip)
#
# a proper rotation (det +1, Rz(-90 deg)) plus a translation, no scaling and no
# free parameter.  Residuals against the manufacturer's own finger: foot outer
# face vs the finger's back face +0.155 mm (visual) / +0.097 (collision); plate
# contact face vs the fingertip's back face +0.150; grip plane vs the finger's
# own inner face +0.084 / +0.133; tip z 0.000 / +0.051.  WORST 0.155 mm, and
# the two manufacturer meshes disagree with each other by 0.051.  Independent
# check: the plate's z centre lands at panda_hand z = 103.242 mm against the
# 10-deg assembly's own grip centre of 103.26 and the stock TCP's 103.4.
#
# WHAT IT MEANS FOR THE GRIP — see docs/SYSTEM_MODEL.md 7d.  The contact face
# sits 10.6502 mm OUTBOARD of the stock grip plane and the rib 8.0663, so the
# gap between two of these is `2 q + 21.3004` at the plates and `2 q + 16.1326`
# at the ribs, where q is the finger joint and 2 q is what libfranka reports as
# `width`.  TWO READINGS OF THE PLATE, and this module does not choose:
#
#   A  THE PLATE GRIPS.  Then it never reaches the post: closing on the 50 mm
#      post ends, the RIB lands first, at q = 16.9337 mm (25.000 post half-
#      length less the rib's own 8.0663), leaving the plates 2.5839 mm off.
#      Measured, not argued: bisecting q against the committed holder meshes
#      gives 16.9337 mm and names the rib crest as the touching vertex.
#      libfranka width 0.0339.
#   B  THE PLATE CARRIES THE STOCK FINGERTIP, and four measurements point at
#      it.  The flat band between the rib and the plate's far edge is 18.000
#      mm and the FR3 fingertip is an 18.1156 mm square.  The 6.000 mm hole is
#      centred in that band on the fingertip's own brass-insert axis, 0.0002
#      mm out.  The plate face is 0.1502 mm outboard of the fingertip's back
#      face, i.e. exactly a seat.  And with the tips seated in the post's own
#      sockets (the 10-deg assembly's 36.0008 mm), the rib clears the post by
#      0.916 mm — which is a tight fit, not a coincidence.  Grip face at link
#      y = 0.1502, gap `2 q + 0.3004`.
#
# NOTHING ON THIS FINGER LOCATES THE POST EITHER WAY.  The contact face is one
# flat plane, the two holes are 69 mm apart where the post is 26 mm square, and
# the post's end faces have no bore for a pin (rays down the post axis of the
# 22-deg housing hit solid material at z = 5.426 and 44.574 — the socket floor
# is a chamfered cone).  The only thing in the whole system that fixes the
# clocking is the housing's own 18 x 18 socket closing on a FINGERTIP, and
# whether a fingertip is fitted and seated is a decision made by hand.  Read
# against the running GUI's `width = 0.0432` with `epsilon_inner = 0.0`, the
# only configurations that clear 43.2 mm are grips on the post's BARE 50 mm
# ends (0.0497 with a tip on the plate, 0.0500 with a stock finger) — and a
# bare post end is a flat 26 mm square with nothing to key into.  So the pen's
# lean out of the approach axis is set at grasp time, not by the CAD.
FATFINGER = dict(
    parent="panda_leftfinger / panda_rightfinger",
    source='raw_slack_file_dump/"Pen holder all parts 2026.08.19"/'
           '"Fat Franka Finger v250904.STL" (mm, watertight, 8234 faces); '
           'same solid as "Fat Franka Finger v250904.SLDPRT" in '
           '"Pen holder cad(1).zip"',
    # --- the STL's own frame ---
    stl_y_shift=-0.010500,        # STL -> SLDPRT datum, y only (x, z agree)
    extent=(0.0184339, 0.0900003, 0.050000),
    # --- CAD (SLDPRT) frame, metres ---
    plate_x=(0.0640578, 0.0679076),   # (back face, CONTACT face)
    plate_z=(0.046000, 0.064000),
    plate_y=(-0.009000, 0.079500),    # the 1.5 mm the two ends differ by
    plate_hole_d=0.006000,            # 5.994..6.000 over the bore, chamfered
    plate_hole_y=(0.0, 0.069000),
    plate_hole_z=0.055000,            # 18.000 of flat, rib to far edge —
                                      # and the FR3 fingertip is 18.1156 square
    rib_x=0.0704915,                  # the rib crest, cad x
    rib_z=(0.045509, 0.046000),       # the flat top of it
    rib_proud=0.0025839,              # how far past the contact plane
    foot_x=(0.0520578, 0.0600578),    # (OUTER face -> carriage, inner face)
    foot_z=(0.014000, 0.028000),
    foot_y=((-0.010500, 0.009500), (0.059500, 0.079500)),
    foot_hole_d=0.004296,             # M4 clearance waist, x 55.06..57.06
    foot_hole_cbore_d=0.007293,       # 3.0 mm deep from BOTH faces
    foot_hole_y=(-0.006000, 0.006000, 0.063000, 0.075000),
    foot_hole_z=0.022000,
    web_normal=(0.8412, 0.0, -0.5407),
    mirror_y=0.034500,                # the part's own mirror plane
    mirror_worst=0.0015045,           # and how far it misses being one
    mirror_volume_fraction=0.99049,
    # --- the placement, CAD -> panda_leftfinger ---
    grip_face_x=0.0785578,            # cad x of the FR3 fingertip's grip plane
    z_offset=-0.0101579,              # cad z -> link z
    mirror_pitch=0.069000,            # left <-> right, about link z at x/2
    # --- what falls out, in the finger link frame ---
    plate_offset=0.0106502,           # contact face, link y  (the whole story)
    rib_offset=0.0080663,             # the rib crest, link y — the innermost
    fingertip_offset=0.0001502,       # a tip ON the plate would grip here
    plate_link_z=(0.0358421, 0.0538421),
    foot_link_y=(0.018500, 0.026500),
    foot_hole_link=((-0.006, 0.0118421), (0.006, 0.0118421)),
    # --- COLLISION.  A printed Z-bracket is not a collision geometry and the
    # convex hull of one swallows the concavity whole, so this follows
    # PENHOLDER22's own convention: a PRIMITIVE ENVELOPE, MEASURED off the
    # vendored mesh rather than typed in, and re-proved on every generator run
    # (`gen_system_model.vendor_fat_finger`).  The part cuts naturally into
    # three z bands at the CAD's own steps — foot, web, plate — and the foot
    # band into two boxes at the part's mirror plane, because the two feet have
    # 48.6 mm of air between them.  Each box is that cell's AABB, so the web's
    # fills the Z's concave side: conservative, which is the direction an
    # envelope is allowed to be wrong in.  The two foot boxes ABUT on the split
    # rather than stopping at their own geometry, so a triangle drawn across
    # that air has its middle in a box rather than in neither.
    collision_bands=(0.0178421, 0.0358421),   # link z cuts = cad z 28 and 46
    collision_foot_split=0.034500,            # the foot band's own x cut
    collision_overlap=0.000500,               # z slop when fitting the AABBs
    insert="93514A130 flanged barbed insert, flange 7.9248 mm — the same one "
           "the Rodgers fingertip carries, on the same axis to 0.0002 mm",
    locates_post=False,
    locates_post_why="the contact face is one flat plane (1516 mm^2, the "
                     "part's largest) broken only by the two 6.000 mm holes "
                     "and the R5 corners; the holes are 69 mm apart and the "
                     "post is 26 mm square, so at most one can face it; and "
                     "the post's end faces have no bore for a pin to enter "
                     "(measured on the 22-deg housing STL: rays down the post "
                     "axis hit solid material at z = 5.426 and 44.574).  The "
                     "clocking about the jaw axis is FREE unless a FINGERTIP "
                     "is fitted and seated in the housing's own 18 x 18 mm "
                     "socket, which is a decision made by hand.",
)


def fatfinger_widths():
    """The `width` libfranka would report, per grasp hypothesis -> dict.

    One number off the running robot settles which build is on the arms, and
    these are the five it has to choose between.  `width` is 2 q, the STOCK
    grip plane's opening; each row adds back how far that build's real contact
    face sits outboard of it.
    """
    F = FATFINGER
    post = PENHOLDER22["post_z"][1] - PENHOLDER22["post_z"][0]      # 0.050
    socket = 0.007
    seated = post - 2 * socket                                      # 0.036
    return {
        "fat plates on the bare post ends":
            round(post - 2 * F["plate_offset"], 6),
        "fat RIBS on the bare post ends (the plates cannot reach)":
            round(post - 2 * F["rib_offset"], 6),
        "fat plate + fingertip, seated in the sockets":
            round(seated - 2 * F["fingertip_offset"], 6),
        "fat plate + fingertip, flat on the bare post ends":
            round(post - 2 * F["fingertip_offset"], 6),
        "stock finger, tips seated in the sockets": round(seated, 6),
        "stock finger, faces flat on the bare post ends": round(post, 6),
    }


def fatfinger_T_finger(mirrored=False):
    """(4,4) panda_leftfinger <- the Fat finger STL (metres).

    `mirrored=True` gives the placement on the OTHER finger — the far foot
    bolted down instead of the near one — which is a proper rotation only
    because the part is its own mirror image (see FATFINGER's docstring).
    In the right finger's own link frame it is `Rz(pi)` about x = 34.5 mm.
    """
    F = FATFINGER
    T = np.eye(4)
    T[:3, :3] = np.array([[0.0, 1.0, 0.0],
                          [-1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0]])
    T[:3, 3] = (F["stl_y_shift"], F["grip_face_x"], F["z_offset"])
    if not mirrored:
        return T
    M = np.diag([-1.0, -1.0, 1.0, 1.0])
    M[0, 3] = F["mirror_pitch"]
    return M @ T


def fatfinger_jaw_gap(q):
    """Finger joint value `q` (m) -> the gap between the two contact plates.

    libfranka reports `2 q` as the gripper `width`; the plates add
    `2 * plate_offset` because they sit that far outboard of the stock grip
    plane.  The inverse is `fatfinger_width_for_gap`.
    """
    return 2.0 * (float(q) + FATFINGER["plate_offset"])


def fatfinger_width_for_gap(gap):
    """Gap between the plates (m) -> the `width` libfranka would report."""
    return float(gap) - 2.0 * FATFINGER["plate_offset"]


def _point_box_d(P, lo, hi):
    """(...,3) points vs one box -> (...,) distance (0 inside)."""
    d = np.maximum(np.maximum(lo - P, P - hi), 0.0)
    return np.linalg.norm(d, axis=-1)


def segment_box_clearance(A, B, boxes, iters=36):
    """(N,3),(N,3) segments -> (N,) exact min distance to the box set.

    Per box, d(t) = dist(A + t(B-A), box) is convex in t (each coordinate
    deficit is a max of affines; a norm of nonnegative convex components is
    convex), so ternary search finds the true minimum.  A Lipschitz bound
    (d >= min(d(A), d(B)) - |B-A|/2) prunes pairs that cannot come close, so
    the search only runs where it matters.
    """
    A = np.asarray(A, float).reshape(-1, 3)
    B = np.asarray(B, float).reshape(-1, 3)
    if not boxes:
        return np.full(len(A), np.inf)
    lo = np.stack([b["lo"] for b in boxes])
    hi = np.stack([b["hi"] for b in boxes])
    dA = _point_box_d(A[:, None], lo, hi)          # (N, M)
    dB = _point_box_d(B[:, None], lo, hi)
    L = np.linalg.norm(B - A, axis=1)              # (N,)
    ends = np.minimum(dA, dB)
    out = ends.copy()
    # pairs whose Lipschitz bound could beat the endpoint distance
    ni, mi = np.nonzero(ends - 0.5 * L[:, None] < out.min(axis=1)[:, None])
    if len(ni):
        a, d = A[ni], (B - A)[ni]
        blo, bhi = lo[mi], hi[mi]
        t0 = np.zeros(len(ni))
        t1 = np.ones(len(ni))
        for _ in range(iters):
            m1 = t0 + (t1 - t0) / 3.0
            m2 = t1 - (t1 - t0) / 3.0
            f1 = _point_box_d(a + m1[:, None] * d, blo, bhi)
            f2 = _point_box_d(a + m2[:, None] * d, blo, bhi)
            take1 = f1 <= f2
            t1 = np.where(take1, m2, t1)
            t0 = np.where(take1, t0, m1)
        dmin = _point_box_d(a + (0.5 * (t0 + t1))[:, None] * d, blo, bhi)
        np.minimum.at(out, (ni, mi), dmin)
    return out.min(axis=1)


def chain_static_clearance(P, boxes, capsules=None):
    """(N,10,3) or (N,11,3) world chain points (frames.fk's 9 + tool points)
    -> (N,) min over capsules of (segment-to-box-set distance minus capsule
    radius).  Compare against STATIC_MARGIN.  With `capsules=None` the table
    is selected by the chain's width: 10 points = inline pen
    (STATIC_CAPSULES), 11 = lateral holder (STATIC_CAPSULES_LAT)."""
    P = np.asarray(P, float)
    if P.ndim == 2:
        P = P[None]
    if capsules is None:
        capsules = STATIC_CAPSULES_LAT if P.shape[1] >= 11 else STATIC_CAPSULES
    if not boxes:
        return np.full(len(P), np.inf)
    worst = np.full(len(P), np.inf)
    for i, j, r in capsules:
        d = segment_box_clearance(P[:, i], P[:, j], boxes) - r
        worst = np.minimum(worst, d)
    return worst


def box_clearance(points, boxes):
    """(N,3) canvas points -> (N,) distance to the nearest box surface.

    Exact for points outside; 0.0 inside (the conservative sign).  Vectorised
    over points x boxes.
    """
    P = np.asarray(points, float).reshape(-1, 3)
    if not boxes:
        return np.full(len(P), np.inf)
    lo = np.stack([b["lo"] for b in boxes])           # (B,3)
    hi = np.stack([b["hi"] for b in boxes])
    d = np.maximum(lo[None] - P[:, None], P[:, None] - hi[None])  # (N,B,3)
    outside = np.linalg.norm(np.maximum(d, 0.0), axis=2)
    return outside.min(axis=1)
