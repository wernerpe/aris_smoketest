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
# capsules CONTAIN it, so unlike every arm capsule they did not have to move.
# The inference in that placement is the residual risk, not the radius.
BRACKET_R_LAT = 0.05  # bracket capsule radius (CAD-validated envelope)
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
#     over x in [0.00055, 0.0027] at the nose, and THE PEN LEAVES AT x = 0.
#     The step between them is a chamfer bottoming at x = 0.00315 and reaching
#     full bore at x = 0.0036; a flat 21.0 mm face lands on it at
#     x = 0.003340, which is `stack_front_x` and the TAIL STOP of everything
#     inside.  x = 0 is the TAIL, not the nose: the 17.00 land is the SPRING's
#     stop (17.00 will not pass a 19.05 spring, which is what it is cut for)
#     and the pen leaves through the CAP at the other end.  See
#     `penholder22_stack` and docs/SYSTEM_MODEL.md 7c.  (An earlier reading
#     called that 17.0 "exactly the clutch's 17.07 OD, so the clutch seats in
#     the nose".  It does not: 17.07 does not enter 17.00, the assembly puts
#     the clutch 45 mm further back, and the pen leaves at the OTHER end.)
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
#   2. the grip-to-exit length is 25.00 mm — NOT the 55.10 this module places
#      the housing on, which is the tail; see docs/SYSTEM_MODEL.md 7c — so
#      reaching 155.56 mm needs 130.5 mm of graphite past the cap, against
#      94.5 mm at 23 deg and the 10-deg build's own measured 20.7;
#   3. the SIGN of the lateral offset is a mounting choice, not a CAD fact:
#      the post is square, so the holder seats in the sockets either way up
#      and the lean is +/-23 deg.
PENHOLDER22 = dict(
    parent="panda_hand",
    source='raw_slack_file_dump/"Pen holder all parts 2026.08.19"/ '
           '(STL, mm, no SLDASM — placement inferred; see docstring above)',
    # --- housing, in its own frame (metres) ---
    bore_yz=(0.016900, 0.025000),     # bore axis, housing (y, z)
    nose_x=0.0,                       # the pen leaves the housing here
    thread_x=(0.072000, 0.080100),    # external thread for the cap
    barrel_r=0.013585,                # measured max OD/2 over x in [0, 0.036]
    thread_r=0.014272,                # measured max OD/2 over the thread
    cap_end_x=0.085100,               # the cap's closed face, along the bore
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

    -> (T_hand_housing, T_hand_cap, nose_along_bore, tip_along_bore).

    THE PLACEMENT IS INFERRED (no assembly file).  Three assumptions, each of
    them the only one the parts support:

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
    """
    d = np.array([float(pen_lat), 0.0, float(pen_ext)])
    reach = float(np.linalg.norm(d))
    u = d / reach                                 # TCP -> tip, unit
    P = PENHOLDER22
    Xh = -u                                       # housing +X points AWAY
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
    return T, T @ Tc, P["post_xy"][0] - P["nose_x"], reach


def penholder22_collision(pen_ext, pen_lat, d_hand_tcp):
    """The holder's CONSERVATIVE primitive envelope, in the panda_hand frame.

    -> [("cylinder", T (4,4), (radius, length))], the cylinder's axis being
    its own frame's z, as URDF wants it.

    A 7 000-triangle concave printed part is not a collision geometry, and the
    final rig's convention for exactly this problem is a primitive envelope
    (`TOOL["collision"]`, one cylinder).  This one is three coaxial cylinders
    — `PENHOLDER22["env_cylinders"]`, whose radii are measured maxima, not
    guesses — and `scripts/extract_penholder22_meshes.py` re-proves on every
    run that no visual vertex escapes them.
    """
    T_h, _, _, _ = penholder22_T_hand(pen_ext, pen_lat, d_hand_tcp)
    by, bz = PENHOLDER22["bore_yz"]
    out = []
    for x0, x1, r in PENHOLDER22["env_cylinders"]:
        T = np.eye(4)
        T[:3, 3] = (0.5 * (x0 + x1), by, bz)
        # the cylinder's own z must run along the housing's x
        T[:3, :3] = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])
        out.append(("cylinder", T_h @ T, (r, x1 - x0)))
    return out


def penholder22_stack(spacer=None):
    """The internal stack along the bore -> [dict], housing frame, metres.

    Each row is `dict(name, x0, x1, r, note)` with x measured from the nose
    (x = 0) along the housing's own +X.  This is the ORDER, and the order is
    the part of the holder eight loose STLs cannot give you.

    WHERE THE ORDER COMES FROM.  The 10-deg "natural hold" assembly in
    raw_slack_file_dump/"Pen holder cad(1).zip" -> "Natural hold assembly -
    closed.zip" is the same architecture built out of the same shapes, and
    resolving its component transforms puts, along its bore from the nose:

        tail shoulder  ->  SPRING  ->  SLEEVE  ->  CAP -> the pen

    with the spring's front coil landing 3.302 mm behind the nose (this
    housing's own shoulder is at 3.34) and the sleeve's front face on the
    spring's back coil to 0.1 mm.  Every interface below is then a measured
    fit on THIS delivery's parts:

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
    out.append(dict(name="graphite_buried", x0=x1 - clutch["length"], x1=0.0,
                    r=P["lead_r"],
                    note="the 7.0 mm stick from the clutch's grip out through "
                         "the nose; what shows PAST the nose is `pen_lead`"))
    return out


def penholder22_internals(pen_ext, pen_lat, d_hand_tcp, spacer=None):
    """`penholder22_stack` placed in the panda_hand frame.

    -> [(name, T (4,4), (radius, length), note)], each cylinder's axis being
    its own frame's z, the way URDF wants it — the same convention
    `penholder22_collision` uses, and for the same reason.

    VISUAL ONLY.  Every one of these lives inside the 21.148 mm bore and is
    already inside `env_cylinders`; adding them to the collision model would
    add nothing but faces.  `penholder22_internals_escape` proves the first
    half of that claim on every run.
    """
    T_h, _, _, _ = penholder22_T_hand(pen_ext, pen_lat, d_hand_tcp)
    by, bz = PENHOLDER22["bore_yz"]
    out = []
    for b in penholder22_stack(spacer):
        lo, hi = sorted((b["x0"], b["x1"]))
        T = np.eye(4)
        T[:3, 3] = (0.5 * (lo + hi), by, bz)
        T[:3, :3] = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])
        out.append((b["name"], T_h @ T, (b["r"], hi - lo), b["note"]))
    return out


def penholder22_internals_escape(spacer=None):
    """How far the stack reaches outside `env_cylinders` -> metres.

    0.0 when the complete assembly still fits the collision hull the housing
    and cap were fitted to, which is the whole question modelling the
    internals raises.  Measured on the bodies' own corner rings, so a body
    that pokes out radially OR axially is caught.
    """
    P = PENHOLDER22
    worst = -np.inf
    for b in penholder22_stack(spacer):
        lo, hi = sorted((b["x0"], b["x1"]))
        for x in (lo, hi):
            d = np.inf
            for c0, c1, r in P["env_cylinders"]:
                d = min(d, max(b["r"] - r, c0 - x, x - c1))
            worst = max(worst, d)
    return float(max(0.0, worst))


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
