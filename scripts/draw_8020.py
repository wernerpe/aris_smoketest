#!/usr/bin/env python3
"""FABRICATION DRAWINGS for the ARIS 80/20 cage — plan, elevation, cut list.

Three deliverables, all parametrised by the mount plane `h`:

    out/drawings/arm_spacing_topdown.pdf / .png    plan view, A3 landscape
    out/drawings/cage_side_view.pdf / .png         elevations, A3 landscape
    out/drawings/8020_cut_list.md                  the cut list

READ-ONLY against the package.  Every number below is read at run time from
`aris_sixarm.system_model` (the dimensioned physical model), `aris_sixarm.
layout` (the certified base positions), `aris_sixarm.mounts` (the schematic
keep-out the certified numbers were earned against) and
`out/certified_area_h####.json` (the certified drawing area).  Nothing here
defines a dimension of its own; if a number moves in the package it moves
here.

THE DATUM, AND WHY THIS SHEET SUPERSEDES THE EARLIER ONE
--------------------------------------------------------
`out/ceiling_8020_layout.*` drew the grid underside at z = 2340 mm above the
paper, because `mounts.MOUNTS.ceiling_z = 2.34 m` reads the original drawing's
233,7 cm as if it were measured from the paper.  It is measured FROM THE
FLOOR, to the top of a self-supporting cage, and the paper sits 636.68 mm
above that floor.  The runway underside is therefore 1623.62 mm above the
paper, not 2340.  That earlier sheet's drop posts — 1435.0 at h = 940 and
1525.0 at h = 850 — are 716.4 mm too long and MUST NOT BE CUT.  See
`docs/SYSTEM_MODEL.md` §1 and `system_model.reconciliation()`.

    drop post cut length = (grid underside - h) + POST_OVER
                         = (1623.62 - h) + 34.98

`POST_OVER` is the drawing's own over-run of the post past the plate
underside (`down_plate` lo z minus `down_boom_W` lo z), not a round 35.

    h = 970  ->  688.6 mm     layout.LAYOUT_PROPOSED, ADOPTED 2026-09-10, and
                              the height this sheet is issued at
    h = 940  ->  718.6 mm     what shipped 2026-08-26 .. 2026-09-09
    h = 850  ->  808.6 mm     what the hardware is built at now

NOTHING HERE IS A SURVEY, and nothing here changes a layout or gate constant.

    python3 scripts/draw_8020.py                 # h = 0.970 m
    python3 scripts/draw_8020.py --h 0.940
    python3 scripts/draw_8020.py --h 850 --out out/drawings_850
"""
import argparse
import datetime as _dt
import json
import os
import sys
import textwrap

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon, FancyArrow
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aris_sixarm import frames, layout, mounts, rig_final       # noqa: E402
from aris_sixarm import system_model as SM                      # noqa: E402

# ===========================================================================
# 1.  THE TRUTH, READ FROM THE PACKAGE
# ===========================================================================
IN = 25.4                       # mm per inch
MM = 1000.0                     # m -> mm

H_DESIGN = 970.0                # mm — the height the certified-workspace work
                                #      recommended (docs/DECISIONS.md,
                                #      "OPEN — FOR PETE: NO HOLES UNDER THE
                                #      ARMS", 2026-09-08) and that Pete
                                #      ADOPTED on 2026-09-10.  It is now also
                                #      `layout.LAYOUT_PROPOSED["h"]`, so
                                #      H_DESIGN == H_SHIPPED and the sheet's
                                #      "design" and "shipped" rows coincide.
H_TABLE = (970.0, 940.0, 850.0)   # the three heights the sheet tabulates
H_SHIPPED = SM.H_MOUNT            # 970.0, layout.LAYOUT_PROPOSED["h"]
H_ASBUILT = 850.0                 # docs/BUILD_SHEET.md §1, the hardware today

# --- canvas and base positions --------------------------------------------
CW, CL = SM.CANVAS_W, SM.CANVAS_L                 # 1803.40 x 3630.64
ARMS = dict(SM.ARM_XY)                            # {arm_id: (x, y)} mm
COL_X, ROW_Y = list(SM.COL_X), list(SM.ROW_Y)
COL_SP = round(COL_X[1] - COL_X[0], 2)            # 610.0
ROW_SP = round(ROW_Y[1] - ROW_Y[0], 2)            # 1210.21

# --- the steel -------------------------------------------------------------
P = SM.PROFILE                                    # 76.2 = 3 in square T-slot
PT = SM.PROFILE_THIN                              # 38.1 = 1.5 in
PLATE = tuple(float(v) for v in SM.PLATE)         # 225.82 x 190 x 12.7
CLAMP = tuple(float(v) for v in SM.CLAMP)         # 226 x 152.4 x 95.7
GUSSET = SM.GUSSET                                # 203.2 x 38.1 x 203.2
FR_X0, FR_X1, FR_Y0, FR_Y1 = SM.FR_X0, SM.FR_X1, SM.FR_Y0, SM.FR_Y1
IN_X0, IN_X1 = SM.IN_X0, SM.IN_X1
GRID_U, GRID_T = SM.GRID_U, SM.GRID_T             # 1623.62 / 1699.82
LEG_BOTTOM = SM.LEG_BOTTOM                        # -27.38
LEG_LEN = round(GRID_U - LEG_BOTTOM, 2)           # 1651.00
FLOOR_Z, TABLE_TOP_Z = SM.FLOOR_Z, SM.TABLE_TOP_Z  # -636.68 / -2.00
PAPER_ABOVE_FLOOR = round(-FLOOR_Z, 2)            # 636.68

# --- the seam frame, where the two half-cages butt -------------------------
# `system_model` section 3b, added 2026-09-14: the real rig is TWO copies of
# the original three-arm half-cage (218.44 x 208.28 cm) butted along the
# paper's long axis, and the steel that holds the two butted END RAILS up was
# missing from every sheet this script has ever issued.  Read at run time like
# everything else here — the eight boxes come straight out of
# `system_model.seam_bodies()` and NOT ONE coordinate is restated below.
SEAM_Y = SM.SEAM_Y                            # 1815.32, and it IS ROW_Y[1]
HALF_CAGE_L = SM.HALF_CAGE_L                  # 2082.80, one half-cage in y
SEAM_RAIL_RESIDUAL = SM.SEAM_RAIL_RESIDUAL_MM  # 0.78 — the whole argument
SEAM_OVERLAP = SM.SEAM_OVERLAP_MM             # 153.96 = one doubled end frame
SEAM_BRACE = tuple(float(v) for v in SM.SEAM_BRACE)   # 203.2 x 38.1 x 203.2
SEAM_BRACE_RUN = SM.SEAM_BRACE_RUN            # 203.2 inboard off the post
SEAM_BRACE_H = SM.SEAM_BRACE_H                # 203.2 of it under the rail

# THE FLAG.  Verbatim, everywhere the seam frame appears — plan, both
# elevations, the title-block banner, the cut list and the open items.  The
# seam frame is REPORTED steel (Pete Werner, 2026-09-14) that nobody has
# photographed; it must never read as surveyed.
SEAM_FLAG = "AS DIRECTED 2026-09-14 — one representative bar per side"

# --- the arm-31 mount, lifted from the drawing -----------------------------
POST_PITCH_X = SM.POST_PITCH_X                    # 317.6
POST_SLOT = SM.POST_SLOT                          # 241.4 between the pairs
PLATE_CLR = SM.PLATE_SIDE_CLEAR                   # 7.79 each side
PLATE_OFF = abs(SM.PLATE_OFF)                     # 25.15, DIRECTION INFERRED

# ---------------------------------------------------------------------------
# CLOCKING — which way each column's front, connector and plate offset point
# ---------------------------------------------------------------------------
# `aris_sixarm.system_model` is written for the SHIPPED uniform clocking and
# hard-codes the flip once (`plate_centre_x`): every front faces canvas -x, so
# every connector and every plate offset go to canvas +x.  A mirrored sheet
# needs the same arithmetic PER COLUMN, and this is the only place in this
# script that knows which.
#
# `SIGN[side]` is the direction the connector and the plate offset point, in
# canvas x.  Uniform: +1 for both columns.  Mirrored: the LEFT column turns to
# face +x, so its connector and plate go to -1, and the right column does not
# move.  Everything else on the sheet is clocking-invariant, including the
# certified keep-out — the neighbour column obstacle is built from the base
# origin and z axis and a turn about that axis cannot move it.
CLOCKINGS = {
    "uniform":  {"left": +1.0, "right": +1.0},
    "mirrored": {"left": -1.0, "right": +1.0},
}
CLOCKING = "uniform"          # module state, set once by main()


def _side(x_axis):
    """'left' or 'right' column for a J1 axis at `x_axis` mm."""
    return "left" if float(x_axis) < 0.5 * (COL_X[0] + COL_X[1]) else "right"


def clock_sign(x_axis):
    """+1 if this arm's connector and plate offset run canvas +x, else -1."""
    return CLOCKINGS[CLOCKING][_side(x_axis)]


def plate_cx(x_axis):
    """Plate centre x under the clocking in force. -> mm."""
    return float(x_axis) + clock_sign(x_axis) * PLATE_OFF


def post_xs(x_axis):
    """The two drop-post x centres under the clocking in force."""
    c = plate_cx(x_axis)
    return c - POST_PITCH_X / 2, c + POST_PITCH_X / 2


def cluster_gap():
    """Clear x between a transverse pair's two drop clusters. -> mm."""
    return ((plate_cx(COL_X[1]) - POST_PITCH_X / 2 - P / 2)
            - (plate_cx(COL_X[0]) + POST_PITCH_X / 2 + P / 2))


def gusset_pair_clear():
    """Clear x between a transverse pair's facing gussets. -> mm."""
    return ((plate_cx(COL_X[1]) - POST_PITCH_X / 2 - GUSSET[0] / 2)
            - (plate_cx(COL_X[0]) + POST_PITCH_X / 2 + GUSSET[0] / 2))
POST_OVER = SM.POST_OVER                          # 34.98 past the plate
CLUSTER_W, CLUSTER_D = SM.CLUSTER_W, SM.CLUSTER_D  # 393.8 x 152.4

# --- the base cable pass-through the model found ---------------------------
CABLE_UP = 230.7          # mm the link0 connector runs past the flange
CABLE_R = mounts.MOUNTS.connector_r * MM          # 177.0 radial

# --- the modelled keep-out, for the re-cert flag ---------------------------
BOOM_D = 2 * mounts.MOUNTS.boom_r * MM            # 200.0 column diameter
CEIL_CODE = mounts.MOUNTS.ceiling_z * MM          # 2340.0, the buggy datum

REV_NOTE = ("REV A — SUPERSEDES THE 1435 / 1525 DROP POSTS of "
            "out/ceiling_8020_layout.*")
# the banner line the seam frame adds to BOTH sheets' title blocks
SEAM_BANNER = (f"  SEAM FRAME items I + J — the two half-cages butt at "
               f"y = {SEAM_Y:.2f} and the steel that holds their two butted "
               f"end rails up is NEW ON THIS REVISION:  {SEAM_FLAG}.")
TODAY = _dt.date.today().isoformat()


def post_length(h):
    """Drop-post cut length (mm) for mount plane `h` (mm) — from the model."""
    return SM.z_ladder(float(h))["post_length"]


def zl(h):
    """The z ladder at `h`, straight out of `system_model.z_ladder`."""
    return SM.z_ladder(float(h))


# ---------------------------------------------------------------------------
# THE SEAM FRAME — enumerated from the model, never from a literal
# ---------------------------------------------------------------------------
def seam_members():
    """Every seam member -> [Body], straight out of `system_model`."""
    return list(SM.seam_bodies())


def seam_posts():
    """The seam's VERTICAL members -> [Body].

    Two representative bars since 2026-09-14 (`seam_bar_W` / `seam_bar_E`,
    3 in x 6 in, one per side, centred on the seam); the four inferred
    end-frame corner posts before that, whose union they are.  Either way
    they are read off `system_model`, never off a literal here.
    """
    return [b for b in seam_members()
            if b.name.startswith(("seam_bar", "seam_post"))]


def seam_braces_are_bodies():
    """True if the model carries the corner braces as collision bodies."""
    return any(b.name.startswith("seam_brace") for b in seam_members())


def seam_braces():
    """The four corner brace clusters -> [(lo, hi)] boxes in mm.

    `system_model.seam_bodies()` deliberately does NOT carry these as bodies:
    a brace is bolted to its post's FACES, so any axis-aligned box that
    contains the brace also contains the post, and the model's own
    interpenetration test rightly refuses that geometry (see the comment at
    the end of `system_model.seam_bodies`).  They are still steel somebody has
    to buy and hang, so this sheet draws them — built from the model's own
    post boxes and `SEAM_BRACE_RUN` / `SEAM_BRACE_H`, never from a literal,
    and drawn as a dashed outline that says it is not in the collision model.
    If a later model does carry them as bodies, those are used instead.
    """
    named = [b for b in seam_members() if b.name.startswith("seam_brace")]
    if named:
        return [(b.lo, b.hi) for b in named]
    out = []
    for b in seam_posts():
        # the brace runs inboard in x, toward the canvas, and inboard in y,
        # into the half-cage its own post belongs to — the south post band is
        # the NORTH half-cage's end frame, so its brace runs north
        west = b.lo[0] < 0.5 * (FR_X0 + FR_X1)
        south = b.lo[1] < SEAM_Y
        out.append((
            (b.lo[0] if west else b.lo[0] - SEAM_BRACE_RUN,
             b.lo[1] if south else b.lo[1] - SEAM_BRACE_RUN,
             GRID_U - SEAM_BRACE_H),
            (b.hi[0] + SEAM_BRACE_RUN if west else b.hi[0],
             b.hi[1] + SEAM_BRACE_RUN if south else b.hi[1],
             GRID_U)))
    return out


def seam_post_length():
    """Seam corner-post cut length (mm), measured off the model's own box."""
    ps = seam_posts()
    if not ps:
        return round(GRID_U - LEG_BOTTOM, 2)
    return round(ps[0].hi[2] - ps[0].lo[2], 2)


def seam_clear_span():
    """Clear runway span (mm) from a corner leg to the seam post beside it."""
    return round((SEAM_Y - P) - (FR_Y0 + P), 2)


def certified_area(h):
    """The certified drawing-area rectangle at `h` -> dict, or None.

    Read from `out/certified_area_h####.json` — the block the workspace study
    emits with the atlas, tool and gates it came from.  Not recomputed here.
    """
    tag = f"h{int(round(h)):04d}"
    p = os.path.join(_repo(), "out", f"certified_area_{tag}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        d = json.load(f)
    r = d["rect"].get("largest") or d["rect"].get("centred")
    return dict(x0=r["x0"] * MM, y0=r["y0"] * MM, x1=r["x1"] * MM,
                y1=r["y1"] * MM, w=r["w"] * MM, h=r["h"] * MM,
                area=r["area_m2"], live_pct=d.get("live_pct"),
                source=os.path.relpath(p, _repo()),
                tool_lean=d.get("tool", {}).get("lean_deg"))


def _repo():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ===========================================================================
# 2.  CUT LIST
# ===========================================================================
def cut_list(h):
    """Every member -> (item, name, profile, length_mm, qty, what)."""
    pl = post_length(h)
    return [
        ("A", "perimeter side rail", '3" x 3" T-slot', SM.RAIL_LEN_Y, 2,
         "runs in y, full length. The frame outside sits 190.5 (7.5 in) "
         "clear of the canvas on all four sides."),
        ("B", "perimeter end rail", '3" x 3" T-slot', SM.RAIL_LEN_X, 2,
         "runs in x, butts between the side rails. 2032.0 = 80.00 in, the "
         "original frame's own cut."),
        ("C", "runway beam", '3" x 3" T-slot', SM.RAIL_LEN_X, 6,
         "two laid side by side per arm row (152.4 overall) = the drawing's "
         "CENTRAL DOUBLE BEAM. Seam ON the row line, so each arm's J1 axis "
         "lies on it. 3 rows."),
        ("D", "drop post", '3" x 3" T-slot', pl, 24,
         f"four per arm in a 2x2 cluster, pitch {POST_PITCH_X} (x) x {P} (y). "
         f"THE HEIGHT-DEPENDENT CUT — see the table. (The original rig's own "
         f"is {SM.O_POST_L} = 29.00 in.)"),
        ("E", "corner leg", '3" x 3" T-slot', LEG_LEN, 4,
         "floor to the perimeter rail's UNDERSIDE. ASSUMED — the cage is "
         "self-supporting and floor-standing, and a 4.01 m frame on four "
         "legs has no precedent (the original spans 2.08 m). MID-SPAN LEGS "
         "ARE ALMOST CERTAINLY REQUIRED: see open item 3."),
        ("F", "top gusset", '8" x 8" x 1.5" gusset', GUSSET[0], 24,
         f"{GUSSET[0]} x {GUSSET[2]} x {GUSSET[1]}, four per arm, top flush "
         "with the grid, ROTATED onto the runway's outboard y faces. The "
         f"drawing's inboard orientation needs {SM.GUSSET_NEED} mm across a "
         f"transverse pair and only {SM.GUSSET_GAP} exists; rotated, the "
         f"pair clears by {SM.GUSSET_PAIR_CLEAR}. No part number: see open "
         "item 4."),
        ("I", "seam support bar", '3" (x) x 6" (y)', seam_post_length(),
         len(seam_posts()),
         f"{SEAM_FLAG}. Pete Werner: \"just put a representative bar in the "
         f"middle that is as wide as two of the corner struts.\" ONE PER "
         f"SIDE, tabletop ({LEG_BOTTOM}) to runway underside ({GRID_U}), "
         f"standing in the same x bands as the corner legs but at the seam, "
         f"y = {SEAM_Y:.2f}, {2 * P} wide in y and centred on it. Two 3 in "
         f"posts side by side build the same thing — the bar is exactly their "
         f"union — and that is what the original drawing's own "
         f"post_BL / post_BR do. These are the mid-span legs open item 3 said "
         f"were almost certainly required. Same cut as item E."),
        ("J", "seam corner brace", '8" x 8" x 1.5" gusset', SEAM_BRACE[0],
         2 * len(seam_braces()),
         f"{SEAM_FLAG}. {SEAM_BRACE[0]} x {SEAM_BRACE[2]} x {SEAM_BRACE[1]}, "
         f"the drawing's brace_BL / brace_BR — two plates on each seam post's "
         f"two INBOARD faces, running {SEAM_BRACE_RUN} into that post's OWN "
         f"half-cage and {SEAM_BRACE_H} down from the runway underside. Four "
         f"clusters, eight plates. Same envelope as item F and no part number "
         f"either. DRAWN DASHED — these are NOT in the collision model: a box "
         "that contains a face-bolted brace also contains its post, and "
         "system_model refuses geometry that interpenetrates. Nothing rides "
         f"on that, they sit {GRID_U - SEAM_BRACE_H - H_DESIGN:.0f} mm above "
         f"the mount plane."),
    ]


def plate_list():
    """The non-extrusion parts -> (item, name, size, qty, what)."""
    return [
        ("G", "robot base plate",
         f"{PLATE[0]} x {PLATE[1]} x {PLATE[2]}", 6,
         "the drawing's own 22,6 x 19,0 plate at 12.7 (0.5 in). The arm "
         "bolts to its UNDERSIDE, and that underside IS the mount plane h. "
         f"Nests in the {POST_SLOT} mm slot between the post pairs with "
         f"{PLATE_CLR} mm clear each side. NEEDS A HOLE for the base cable "
         "— open item 2."),
        ("H", "plate clamp stack",
         f"{CLAMP[0]} x {CLAMP[1]} x {CLAMP[2]}", 6,
         "clamps the plate between the post pairs, sitting on top of it. "
         "ENVELOPE ONLY — the drawing carries no part number. NEEDS A HOLE "
         "for the base cable — open item 2."),
    ]


def _plate_size(item):
    """Three-figure size string for a cut-list row that is NOT cut to length.

    Items F and J are bought plates, so the table shows an envelope, not a
    length.  Both come from the package; neither is a literal here.
    """
    box = {"J": SEAM_BRACE}.get(item, GUSSET)
    return f"{box[0]} x {box[2]} x {box[1]}"


def extrusion_m(h):
    """Total 3-in T-slot extrusion (m) at mount plane `h`."""
    tot = 0.0
    for it, _n, prof, ln, qty, _w in cut_list(h):
        if "T-slot" in prof:
            tot += ln * qty
    return tot / 1000.0


OPEN_ITEMS = [
    ("1", "PLATE OFFSET DIRECTION",
     f"Does the base plate's {PLATE_OFF} mm offset from the J1 axis run in "
     f"canvas +x or -x?  The drawing gives it for an arm whose front faces "
     f"+X; every arm here is clocked the other way (R_world_base = Ry(180)), "
     f"so the offset flips — and that flip is an INFERENCE.  {PLATE_CLR} mm "
     f"rides on it: the plate nests in a {POST_SLOT} mm slot, and offset the "
     f"wrong way it does not fit at all.",
     f"Measure the real plate, or open the post pitch to 381.0 (posts at the "
     f"axis +/- 190.5), which fits either way with 14.4 mm each side.",
     "BLOCKS the drop clusters — the post x positions depend on it."),
    ("2", "BASE CABLE PASS-THROUGH",
     f"The plate and the clamp stack are solid across the arm's own base "
     f"cable.  The manufacturer's link0 visual carries the connector and "
     f"cable stub {CABLE_UP} mm past the base flange and {CABLE_R:.0f} mm "
     f"radially; every arm here is INVERTED, so that {CABLE_UP} mm points "
     f"straight UP — through the {PLATE[2]} mm plate, through the "
     f"{CLAMP[2]} mm clamp stack, and on into the cluster slot.  THE MODEL "
     f"FOUND THIS, not a person: the collision shell stops dead at the "
     f"flange, so no clearance check in this repo has ever seen it.",
     "Measure the connector envelope on a real FR3 base and cut the plate "
     "and the clamp stack around it.",
     "BLOCKS the plate and the clamp stack."),
    ("3", "CAGE LEGS",
     f"How many legs does a {SM.FR_L / 1000:.2f} m frame need, and where?  "
     f"Correcting the ceiling datum turned this cage from something hanging "
     f"off a room ceiling into something standing on the floor.  Four corner "
     f"legs at {LEG_LEN} mm are drawn.  PARTLY ANSWERED 2026-09-14: the seam "
     f"frame (item I, open item 7) puts FOUR more posts at the paper's "
     f"mid-length, so the clear runway span is now "
     f"{seam_clear_span():.2f} mm corner-to-seam, not the full frame — but "
     f"those posts are REPORTED, NOT PHOTOGRAPHED, and nothing else has "
     f"changed.",
     "A structural check of the unsupported spans, then a leg count and "
     "position — and a check that no mid-span leg lands in a certified "
     "flight path.  The seam posts do land in one: see open item 7.",
     "BLOCKS the leg cut (item E qty) and any FURTHER mid-span member."),
    ("4", "GUSSET PART AND ATTACHMENT",
     f"The drawing has no part number for a gusset, and the {GUSSET[0]} x "
     f"{GUSSET[2]} x {GUSSET[1]} figure is its envelope, not a detail.  The "
     f"rotation onto the runway's outboard y faces resolves a hard "
     f"interference and leaves {SM.GUSSET_PAIR_CLEAR} mm inside a transverse "
     f"pair.",
     "The fabricator's own bracket selection, against the 76.2 post face and "
     "the runway's y face.",
     "BLOCKS item F."),
    ("5", "ROOM SURVEY",
     f"The real room has NEVER been measured.  Every ceiling number in this "
     f"repo descends from the original drawing's 233,7 cm, which is "
     f"floor-to-top-of-cage — the drawing defines no room ceiling at all.  "
     f"The cage as drawn stands {SM.CAGE_TOTAL_H} mm tall.",
     "Survey the room: floor flatness, clear height, and door/route width "
     "for a 4.01 m frame.",
     "BLOCKS nothing that is cut to the corrected datum, but it is the "
     "reason the datum has to be stated on every sheet."),
    ("6", "RE-CERTIFICATION OF THE DROP CLUSTER",
     f"The certified obstacle model is a single {BOOM_D:.0f} x {BOOM_D:.0f} "
     f"mm column on the base axis.  The real steel is a "
     f"{CLUSTER_W} x {CLUSTER_D} mm cluster offset {PLATE_OFF} mm off it, "
     f"plus {GUSSET[0]} mm gussets.  Neither contains the other: all 60 "
     f"pieces of mount hardware fall outside the certified envelope, worst "
     f"185.55 mm.",
     "Send the steel design back for re-certification against the certified "
     "poses (minutes, not days) — system_model.reconciliation().",
     "BLOCKS final fabrication of the clusters and gussets."),
    ("7", f"SEAM FRAME — {SEAM_FLAG}",
     f"Pete Werner, 2026-09-14, on the real hardware: \"there are a few bars "
     f"on the real hardware that are not in our model.  they are supports in "
     f"the middle ... the real thing is essentially the two halves next to "
     f"each other.\"  The rig is TWO of the original "
     f"{SM.FR_W:.1f} x {HALF_CAGE_L:.1f} half-cages butted along the paper, "
     f"and two butted halves are {2 * HALF_CAGE_L:.1f} against this frame's "
     f"{SM.FR_L:.2f} — a difference of {SEAM_OVERLAP} mm, one doubled 3 in "
     f"end frame to within {abs(SEAM_OVERLAP - 2 * P):.2f} mm.  Laid flush "
     f"with this frame's own ends, each half's seam-side END RAIL lands "
     f"within {SEAM_RAIL_RESIDUAL} mm of the MIDDLE RUNWAY already drawn: the "
     f"runway IS the two butted end rails and is NOT cut twice.  What was "
     f"missing is what holds them up — items I and J, {len(seam_posts())} "
     f"bars and {len(seam_braces())} brace clusters straddling "
     f"y = {SEAM_Y:.2f}.  The bars are model bodies; the braces are drawn "
     f"DASHED because a box containing a face-bolted brace also contains its "
     f"post and system_model will not carry interpenetrating geometry.  "
     f"WHAT TO BUILD THERE IS SETTLED, not photographed: Pete Werner, the "
     f"same day — \"just put a representative bar in the middle that is as "
     f"wide as two of the corner struts\" — so this sheet asks for one "
     f"{P} x {2 * P} bar per side instead of a pair of inferred posts, and "
     f"nothing about the seam is waiting on a photograph.",
     "NOTHING OUTSTANDING.  The bar is what was asked for, and it is in the "
     "planner's certified static set (mounts.obstacles_for) as of "
     "2026-09-14 — so the parks and the certified area already account for "
     "it.  A builder who finds something else at the seam edits one table, "
     "mounts.SEAM_BARS_MM.",
     "CLOSED 2026-09-14.  It blocked the middle row's parks, which were "
     "re-searched against the bar and moved — a seam bar stands 114.3 mm "
     "outboard of the canvas edge over the full "
     f"{seam_post_length():.0f} mm, straight through the band a middle-row "
     "arm's links sweep.  docs/DECISIONS.md."),
]


def write_cut_list(path, h):
    """Write out/drawings/8020_cut_list.md at mount plane `h` (mm)."""
    pl = post_length(h)
    z = zl(h)
    ca = certified_area(h)
    L = []
    a = L.append
    a("# 80/20 cut list — ARIS six-arm cage")
    a("")
    a(f"**Issued {TODAY} · design mount plane h = {h:.1f} mm above the paper "
      f"· all dimensions mm.**")
    a("")
    a(f"> **{REV_NOTE}.**  Those posts came from reading the original "
      f"drawing's 233,7 cm as a paper-referenced dimension when it is "
      f"floor-referenced.  They are {CEIL_CODE - GRID_U:.1f} mm too long.  "
      f"Do not cut them.")
    a("")
    a("## Open items — READ BEFORE CUTTING")
    a("")
    n_block = sum(1 for _n, _t, _w, _h, b in OPEN_ITEMS
                  if b.startswith("BLOCKS") and "nothing" not in b)
    a(f"{n_block} of these {len(OPEN_ITEMS)} block a cut on this sheet.  "
      f"Nothing below is a guess about them; each says what it blocks.")
    a("")
    a("| # | item | blocks | how it closes |")
    a("|---|---|---|---|")
    for n, title, _why, how, blocks in OPEN_ITEMS:
        a(f"| {n} | **{title}** | {blocks} | {how} |")
    a("")
    for n, title, why, how, blocks in OPEN_ITEMS:
        a(f"**{n}. {title}** — {why}  *Closes by:* {how}  *{blocks}*")
        a("")
    a("## The datum, in one sentence")
    a("")
    a(f"**z = 0 is the top surface of the paper as laid on the table; the "
      f"floor is {FLOOR_Z} mm and the runway beams' underside — what a drop "
      f"post hangs from — is {GRID_U} mm, so the cage stands "
      f"{SM.CAGE_TOTAL_H} mm floor to top of steel, which is the original "
      f"drawing's own 233,7 cm.**")
    a("")
    a(f"x runs across the short side of the canvas (0 -> {CW:.1f}), y along "
      f"the long side (0 -> {CL:.2f}), z up.  The canvas reference corner is "
      f"(0, 0) and every plan dimension on the drawings is measured from it.")
    a("")
    a("## Drop post — the one height-dependent cut")
    a("")
    a("```")
    a("post cut length = (runway underside - h) + post over-run")
    a(f"                = ({GRID_U} - h) + {POST_OVER}")
    a("```")
    a("")
    a(f"`{POST_OVER}` is the drawing's own over-run of the post past the "
      f"plate underside (`down_plate` lo z minus `down_boom_W` lo z), not a "
      f"round 35.  24 off, whichever height is chosen.")
    a("")
    a("| mount plane h | post cut length | total 3-in extrusion | what h is |")
    a("|---:|---:|---:|---|")
    for hh in H_TABLE:
        what = {970.0: "**the design height of this sheet AND "
                       "`layout.LAYOUT_PROPOSED['h']`** — adopted 2026-09-10; "
                       "what the software plans against today, and the only "
                       "height at which the canvas has no enclosed dead cells",
                940.0: "what the software planned against 2026-08-26 .. "
                       "2026-09-09; superseded",
                850.0: "what the hardware is built at now (verticals "
                       "trimmable)"}[hh]
        b = "**" if abs(hh - h) < 1e-6 else ""
        a(f"| {b}{hh:.0f}{b} | {b}{post_length(hh):.1f}{b} | "
          f"{extrusion_m(hh):.2f} m | {what} |")
    a("")
    a(f"At the sheet's own h = {h:.0f} the post is **{pl:.1f} mm**.  Cut all "
      f"24 to one length and keep the six mount planes mutually coplanar "
      f"within +/-3 mm.")
    a("")
    a("## Members")
    a("")
    a("| item | qty | profile | cut length | what it is |")
    a("|---|---:|---|---:|---|")
    for it, name, prof, ln, qty, what in cut_list(h):
        disp = f"{ln:.2f}" if "T-slot" in prof else _plate_size(it)
        a(f"| **{it}** {name} | {qty} | {prof} | **{disp}** | {what} |")
    a("")
    a(f"**Total 3-in T-slot extrusion at h = {h:.0f}: {extrusion_m(h):.2f} "
      f"m.**  (Items A-E and I; the gussets F and the seam braces J are "
      f"bought brackets.)")
    a("")
    a(f"> **Items I and J are the SEAM FRAME — {SEAM_FLAG}.**  They are the "
      f"steel that holds up the two butted half-cage end rails at "
      f"y = {SEAM_Y:.2f}, read at run time from "
      f"`system_model.seam_bodies()` ({len(seam_members())} boxes).  The "
      f"END RAILS THEMSELVES ARE NOT A NEW CUT: they are item C's middle "
      f"runway, which this model already builds within "
      f"{SEAM_RAIL_RESIDUAL} mm of where the two halves put them.  Open "
      f"item 7.")
    a("")
    a("## Plate and fabricated parts")
    a("")
    a("| item | qty | size | what it is |")
    a("|---|---:|---|---|")
    for it, name, size, qty, what in plate_list():
        a(f"| **{it}** {name} | {qty} | {size} | {what} |")
    a("")
    a("## Fasteners")
    a("")
    a("**Per the original drawing** — T-slot corner brackets, end fasteners "
      "and gusset hardware to the fabricator's own standard for 3-in "
      "profile.  The drawing carries no fastener schedule and neither does "
      "this sheet; nothing in the certified model depends on one.")
    a("")
    a("## The z ladder at this height")
    a("")
    a("| level | mm above the paper |")
    a("|---|---:|")
    for k in ("grid_top", "grid_underside", "gusset_bottom", "clamp_top",
              "plate_top", "mount_plane", "post_bottom", "paper_top",
              "table_top", "leg_bottom", "floor"):
        a(f"| {k.replace('_', ' ')} | {z[k]:.2f} |")
    a("")
    a("## Plan set-out")
    a("")
    a(f"Frame outside **{SM.FR_W:.1f} x {SM.FR_L:.2f}** "
      f"({SM.FR_W / IN:.2f} x {SM.FR_L / IN:.2f} in), {SM.MARGIN} mm clear "
      f"of the canvas on all four sides.  Columns x = {COL_X[0]:.1f} / "
      f"{COL_X[1]:.1f} (pitch {COL_SP:.1f}); rows y = {ROW_Y[0]:.2f} / "
      f"{ROW_Y[1]:.2f} / {ROW_Y[2]:.2f} (pitch {ROW_SP:.2f}).  Base "
      f"positions are to the **J1 axis** (the centre of the base bolt "
      f"circle), not to a plate edge; tolerance +/-10 mm per base.")
    a("")
    if CLOCKING == "uniform":
        a(f"Every arm is clocked identically — `R_world_base = Ry(180)`, the "
          f"arm's front toward the canvas x = 0 edge, so **all six connector "
          f"panels face the x = {CW:.1f} edge**.  The plate centre sits "
          f"{PLATE_OFF} mm from the J1 axis toward that same edge — DIRECTION "
          f"INFERRED, open item 1.")
    else:
        a(f"**MIRRORED CLOCKING — A VARIANT SHEET, NOT THE SHIPPED ONE.**  "
          f"The LEFT column (13, 31, 2) is clocked `Ry(180) @ Rz(180)` and "
          f"the RIGHT column (17, 71, 97) `Ry(180)`, so the two columns FACE "
          f"EACH OTHER across the centre line at x = "
          f"{0.5 * (COL_X[0] + COL_X[1]):.1f}.  Each column's connector panel "
          f"and its plate's {PLATE_OFF} mm offset run toward its OWN nearest "
          f"long edge, so **no cable is dressed across the paper** — and the "
          f"plate flip opens the steel: cluster-to-cluster gap across a "
          f"transverse pair **{cluster_gap():.2f} mm** (uniform 216.20) and "
          f"gusset pair clearance **{gusset_pair_clear():.2f} mm** (uniform "
          f"89.20), +50.30 on both.  The offset DIRECTION is still inferred "
          f"(open item 1) and it now matters twice, once per column.  "
          f"**The certified workspace is NOT re-earned by this sheet**: "
          f"mirrored measures +4 live cells in 16562 and the SAME certified "
          f"rectangle, and it needs its own park set — "
          f"docs/DECISIONS.md 2026-09-10.")
    if ca:
        a("")
        a(f"The certified drawing area at this height is "
          f"**{ca['w']:.0f} x {ca['h']:.0f} mm** "
          f"(x {ca['x0']:.0f}..{ca['x1']:.0f}, y {ca['y0']:.0f}.."
          f"{ca['y1']:.0f}), {ca['area']:.3f} m², from `{ca['source']}` — "
          f"drawn dashed on the plan.  It is what this spacing buys, and it "
          f"is not the whole canvas.")
    a("")
    a("## Provenance")
    a("")
    a("Generated by `scripts/draw_8020.py`, which reads "
      "`aris_sixarm.system_model` (dimensions and provenance), "
      "`aris_sixarm.layout` (base positions and h), `aris_sixarm.mounts` "
      "(the certified keep-out) and `out/certified_area_h####.json`.  It "
      "defines no dimension of its own.  **Nothing here is a survey.**")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    return path


# ===========================================================================
# 3.  DRAWING PRIMITIVES  (the vocabulary of out/arm_spacing.*)
# ===========================================================================
INK = "#141414"
DIMC = "#8a2846"
NOTEC = "#33445a"
GRN = "#146b4e"
WARN = "#a8430f"
ORIGC = "#1c6f8c"
THIN, MED, HEAVY = 0.55, 0.85, 1.5
FS_DIM, FS_LBL, FS_NOTE, FS_HEAD = 6.2, 7.2, 6.0, 9.4

STEEL = "#9aa2ad"
STEEL2 = "#6f7885"
POSTC = "#4e5763"
PLATEC = "#c8b99f"
ARMC = "#b9bec6"

# the seam frame gets its own ink, and it is deliberately NOT a steel grey:
# it is reported hardware nobody has photographed, and it must not read like
# the rest of the cage does
SEAMC = "#8c4a17"
SEAM_FC = "#e6c49f"

A3 = (16.5354, 11.6929)         # A3 landscape, inches

TBOX = dict(fc="white", ec="none", pad=0.7, alpha=0.95)


class Sheet:
    """One A3 landscape sheet with an exact-scale panel helper."""

    def __init__(self):
        self.fig = plt.figure(figsize=A3, facecolor="white")
        self.W, self.H = A3
        self.arrow = dict(
            arrowstyle="-|>,head_width=0.10,head_length=0.30", color=DIMC,
            lw=THIN, shrinkA=0, shrinkB=0, mutation_scale=6.0)

    def fx(self, v):
        return v / self.W

    def fy(self, v):
        return v / self.H

    def panel(self, x0, y0, xlim, ylim, den, label=None, sub=None):
        w = (xlim[1] - xlim[0]) / den / IN
        h = (ylim[1] - ylim[0]) / den / IN
        ax = self.fig.add_axes([self.fx(x0), self.fy(y0), self.fx(w),
                                self.fy(h)])
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.axis("off")
        if label:
            self.fig.text(self.fx(x0), self.fy(y0 + h + 0.09), label,
                          ha="left", va="bottom", fontsize=FS_HEAD,
                          fontweight="bold", color=INK)
            if sub:
                self.fig.text(self.fx(x0 + w), self.fy(y0 + h + 0.11), sub,
                              ha="right", va="bottom", fontsize=FS_NOTE,
                              color=NOTEC)
        return ax

    def textbox(self, x0, y0, w, h):
        """A blank axes in inches, 0..1 in both directions, for text."""
        ax = self.fig.add_axes([self.fx(x0), self.fy(y0), self.fx(w),
                                self.fy(h)])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        return ax

    # -- dimension primitives ------------------------------------------
    def ext(self, ax, x0, y0, x1, y1):
        ax.add_line(Line2D([x0, x1], [y0, y1], color=DIMC, lw=0.35,
                           ls=(0, (3, 2)), zorder=8.5))

    def dim_h(self, ax, x0, x1, y, text, ext_y=None, over=40, txt_off=26,
              fs=FS_DIM, outside=False, txt_x=None, out_len=None):
        if ext_y is not None:
            for x, yf in ((x0, ext_y[0]), (x1, ext_y[1])):
                self.ext(ax, x, yf, x, y + np.sign(y - yf) * over)
        if outside:
            d = out_len or (abs(x1 - x0) * 0.6 + 170)
            ax.annotate("", xy=(x0, y), xytext=(x0 - d, y),
                        arrowprops=self.arrow)
            ax.annotate("", xy=(x1, y), xytext=(x1 + d, y),
                        arrowprops=self.arrow)
            ax.add_line(Line2D([x0, x1], [y, y], color=DIMC, lw=THIN,
                               zorder=8.5))
        else:
            ax.annotate("", xy=(x0, y), xytext=(x1, y), arrowprops=self.arrow)
            ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=self.arrow)
        ax.text((x0 + x1) / 2 if txt_x is None else txt_x, y + txt_off, text,
                ha="center", va="bottom" if txt_off >= 0 else "top",
                fontsize=fs, color=DIMC, zorder=9, bbox=TBOX, linespacing=1.2)

    def dim_v(self, ax, y0, y1, x, text, ext_x=None, over=40, txt_off=26,
              fs=FS_DIM, outside=False, rot=90, txt_y=None):
        if ext_x is not None:
            for y, xf in ((y0, ext_x[0]), (y1, ext_x[1])):
                self.ext(ax, xf, y, x + np.sign(x - xf) * over, y)
        if outside:
            d = abs(y1 - y0) * 0.6 + 170
            ax.annotate("", xy=(x, y0), xytext=(x, y0 - d),
                        arrowprops=self.arrow)
            ax.annotate("", xy=(x, y1), xytext=(x, y1 + d),
                        arrowprops=self.arrow)
            ax.add_line(Line2D([x, x], [y0, y1], color=DIMC, lw=THIN,
                               zorder=8.5))
        else:
            ax.annotate("", xy=(x, y0), xytext=(x, y1), arrowprops=self.arrow)
            ax.annotate("", xy=(x, y1), xytext=(x, y0), arrowprops=self.arrow)
        ax.text(x + txt_off, (y0 + y1) / 2 if txt_y is None else txt_y, text,
                ha="center", va="center", rotation=rot, fontsize=fs,
                color=DIMC, zorder=9, bbox=TBOX)

    def leader(self, ax, xy, xytext, text, ha="left", va="center",
               fs=FS_NOTE, c=NOTEC, rad=0.0):
        ax.annotate(text, xy=xy, xytext=xytext, ha=ha, va=va, fontsize=fs,
                    color=c, zorder=9.5, linespacing=1.35,
                    bbox=dict(fc="white", ec=c, lw=0.4, pad=2.0, alpha=0.96),
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.5,
                                    shrinkA=2, shrinkB=2,
                                    connectionstyle=f"arc3,rad={rad}"))

    # -- title block ----------------------------------------------------
    def title_block(self, h, sheet_no, title, extra="", of=4, datum=None,
                    datum_sub=None):
        x0, y0, w, hh = 0.42, 0.30, self.W - 0.84, 0.92
        ax = self.textbox(x0, y0, w, hh)
        ax.add_patch(Rectangle((0, 0), 1, 1, fc="white", ec=INK, lw=1.2))
        cols = (0.0, 0.315, 0.475, 0.635, 0.795, 1.0)
        for c in cols[1:-1]:
            ax.add_line(Line2D([c, c], [0, 1], color=INK, lw=0.6))

        def cell(i, head, body, bold=True, c=INK, fs=8.4):
            ax.text(cols[i] + 0.010, 0.74, head, ha="left", va="center",
                    fontsize=5.6, color="#6a7280",
                    fontweight="bold")
            ax.text(cols[i] + 0.010, 0.34, body, ha="left", va="center",
                    fontsize=fs, color=c,
                    fontweight="bold" if bold else "normal",
                    linespacing=1.35)

        ax.text(0.010, 0.80, "ARIS  ·  six inverted FR3 over a horizontal "
                             "paper table", ha="left", va="center",
                fontsize=6.0, color="#6a7280", fontweight="bold")
        ax.text(0.010, 0.46, title, ha="left", va="center", fontsize=11.4,
                color=INK, fontweight="bold")
        ax.text(0.010, 0.14, f"sheet {sheet_no} of {of}  ·  A3  ·  all "
                             f"dimensions mm  ·  scales as noted",
                ha="left", va="center", fontsize=6.0, color=NOTEC)
        cell(1, "DESIGN MOUNT PLANE  h", f"{h:.1f} mm", c=GRN, fs=10.0)
        ax.text(cols[1] + 0.010, 0.13, "above the paper top surface",
                ha="left", va="center", fontsize=5.6, color=NOTEC)
        cell(2, "DROP POST  item D", f"{post_length(h):.1f} mm", c=DIMC,
             fs=10.0)
        ax.text(cols[2] + 0.010, 0.13,
                f"= ({GRID_U} - h) + {POST_OVER},  24 off",
                ha="left", va="center", fontsize=5.6, color=NOTEC)
        cell(3, "DATUM", datum or "z = 0  paper top", fs=8.2)
        ax.text(cols[3] + 0.010, 0.13,
                datum_sub or f"floor {FLOOR_Z}  ·  runway underside {GRID_U}",
                ha="left", va="center", fontsize=5.6, color=NOTEC)
        cell(4, "DATE", TODAY, bold=False, fs=8.2)
        ax.text(cols[4] + 0.010, 0.13, "scripts/draw_8020.py  ·  read-only "
                                       "against the package",
                ha="left", va="center", fontsize=5.6, color=NOTEC)
        # revision banner.  `extra` gets a LINE OF ITS OWN — one long line
        # runs off the A3 sheet, and a flag that is clipped is not a flag.
        rb = self.textbox(x0, y0 + hh + 0.04, w, 0.20)
        rb.add_patch(Rectangle((0, 0), 1, 1, fc="#fdf1ea", ec=WARN, lw=0.9))
        rev = ("  " + REV_NOTE + "  —  that grid datum read the drawing's "
               "233,7 cm (FLOOR to top of cage) as if it were measured from "
               "the paper.  DO NOT CUT FROM THE OLD SHEET.")
        if extra:
            rb.text(0.008, 0.72, rev, ha="left", va="center", fontsize=5.8,
                    color=WARN, fontweight="bold")
            rb.text(0.008, 0.27, extra, ha="left", va="center", fontsize=5.8,
                    color=SEAMC, fontweight="bold")
        else:
            rb.text(0.008, 0.5, rev, ha="left", va="center", fontsize=6.2,
                    color=WARN, fontweight="bold")


def member(ax, x0, y0, x1, y1, fc=STEEL, ec=INK, lw=0.7, z=4, alpha=1.0,
           hatch=None):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=fc, ec=ec, lw=lw,
                           zorder=z, alpha=alpha, hatch=hatch))


def draw_seam(ax, ax0, ax1, lw=0.7, z=5.9, alpha=0.88):
    """Project the SEAM FRAME onto axes (`ax0`, `ax1`) of the canvas frame.

    (0, 1) is the plan, (0, 2) an x-z elevation, (1, 2) a y-z one — the same
    axis-pair convention `draw_park` uses.  Every box is
    `system_model`'s own: this function draws, it does not define.  Posts are
    HATCHED rather than filled because in two of the three views they project
    onto steel that is already there (the corner legs in x-z, the middle
    row's drop posts in y-z) and a solid fill would simply hide it.  Braces
    are a DASHED outline whenever the model does not carry them as bodies —
    see `seam_braces`.
    """
    solid = seam_braces_are_bodies()
    for lo, hi in seam_braces():
        ax.add_patch(Rectangle((lo[ax0], lo[ax1]), hi[ax0] - lo[ax0],
                               hi[ax1] - lo[ax1],
                               fc=SEAM_FC if solid else "none", ec=SEAMC,
                               lw=lw * 0.8, zorder=z,
                               alpha=alpha * 0.62 if solid else 0.9,
                               hatch="////" if solid else None,
                               ls="solid" if solid else (0, (4, 2))))
    for b in seam_posts():
        member(ax, b.lo[ax0], b.lo[ax1], b.hi[ax0], b.hi[ax1], fc=SEAM_FC,
               ec=SEAMC, lw=lw, z=z + 0.25, alpha=alpha, hatch="\\\\\\\\")
    return len(seam_posts()), len(seam_braces())


def seam_centre_line(ax, a0, a1, horizontal=True):
    """The seam plane itself, as a long chain-dot line through a view."""
    if horizontal:
        ax.add_line(Line2D([a0, a1], [SEAM_Y, SEAM_Y], color=SEAMC, lw=0.7,
                           ls=(0, (10, 3, 1.6, 3)), zorder=7.9))
    else:
        ax.add_line(Line2D([SEAM_Y, SEAM_Y], [a0, a1], color=SEAMC, lw=0.7,
                           ls=(0, (10, 3, 1.6, 3)), zorder=7.9))


def cmark(ax, x, y, r, c="#93a2b2", lw=0.5, z=7):
    for a, b in (((x - r, x + r), (y, y)), ((x, x), (y - r, y + r))):
        ax.add_line(Line2D(a, b, color=c, lw=lw, zorder=z,
                           ls=(0, (9, 2.5, 1.4, 2.5))))


def capsule(ax, a, b, r, **kw):
    """A 2-D capsule (segment `a`->`b` thickened by `r`) as a Polygon."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    n = float(np.hypot(*d))
    if n < 1e-9:
        ax.add_patch(Circle(a, r, **kw))
        return
    u = d / n
    p = np.array([-u[1], u[0]])
    th = np.linspace(0.0, np.pi, 20)
    # round cap past b (from +p through +u to -p), then round cap past a
    capb = [b + r * (np.cos(t) * p + np.sin(t) * u) for t in th]
    capa = [a + r * (-np.cos(t) * p - np.sin(t) * u) for t in th]
    ax.add_patch(Polygon(np.array(capb + capa), closed=True, **kw))


def _flow(ax, lines, x, y0, width=88, fs=6.3, dy=0.027):
    """Lay out (colour, bold, text) rows in a 0..1 axes, wrapping at `width`.

    matplotlib has no text wrapping worth the name, so the wrap is done here
    and every physical line gets its own `dy`.
    """
    import textwrap
    yy = y0
    for c, bold, t in lines:
        if not t:
            yy -= dy * 0.55
            continue
        for k, seg in enumerate(textwrap.wrap(t, width) or [""]):
            ax.text(x, yy, seg, ha="left", va="top", fontsize=fs,
                    color=c or INK, fontweight="bold" if bold else "normal")
            yy -= dy
    return yy


def ground(ax, x0, x1, z, depth=90, c="#7f7f7f"):
    """Floor line with hatch ticks below it."""
    ax.add_line(Line2D([x0, x1], [z, z], color=INK, lw=1.3, zorder=3))
    n = max(6, int((x1 - x0) / 90))
    for i in range(n + 1):
        xq = x0 + (x1 - x0) * i / n
        ax.add_line(Line2D([xq, xq - depth * 0.55], [z, z - depth],
                           color=c, lw=0.45, zorder=2.6))


# ===========================================================================
# 4.  PARK-POSE ENVELOPE  (a silhouette, not a certified pose)
# ===========================================================================
def park_capsules(aid, h):
    """The park-pose capsule envelope of one arm -> [(A, B, r)] in mm, canvas.

    `layout.Q_PARK_PROPOSED` is the certified park set, searched at
    h = 940.  Drawn at any other h it is the same arm translated, which is
    all a fabrication silhouette needs — it is NOT a certification at this
    height, and the sheet says so.
    """
    frames.activate_tool("lateral")
    q = np.asarray(layout.Q_PARK_PROPOSED[aid], float)
    T_tcp, P9 = frames.fk(q)
    pts = [np.asarray(P9, float)]
    pts += [np.asarray(t, float) for t in
            frames.tool_points_many(np.asarray(T_tcp)[None])]
    chain = np.vstack([pts[0]] + [p.reshape(1, 3) for p in pts[1:]])
    xa, ya = ARMS[aid]
    R = frames.roty(np.pi)                   # every arm: R_world_base = Ry(180)
    W = (R @ chain.T).T + np.array([xa / MM, ya / MM, float(h) / MM])
    W = W * MM
    # the base column first — `STATIC_CAPSULES_LAT` starts at index 1 because
    # an arm is bolted to the very plate its base would otherwise collide
    # with, so without it the silhouette floats free of the plate
    caps = [(W[0], W[1], mounts.MOUNTS.link_r * MM)]
    return caps + [(W[i], W[j], r * MM)
                   for i, j, r in rig_final.STATIC_CAPSULES_LAT]


def draw_park(ax, aid, h, ax0, ax1, fc=ARMC, ec="#7f8792", alpha=0.85):
    """Project one arm's park envelope onto axes (`ax0`, `ax1`) of the canvas
    frame — (0, 2) for an x-z elevation, (1, 2) for a y-z one."""
    lo = np.inf
    for A, B, r in park_capsules(aid, h):
        capsule(ax, (A[ax0], A[ax1]), (B[ax0], B[ax1]), r, fc=fc, ec=ec,
                lw=0.4, zorder=3.4, alpha=alpha)
        lo = min(lo, A[2] - r, B[2] - r)
    return lo


# ===========================================================================
# 5.  SHEET 1 — PLAN
# ===========================================================================
def sheet_topdown(h, out_dir):
    s = Sheet()
    fig = s.fig
    z = zl(h)
    ca = certified_area(h)

    # ---------------- panel A: the plan ---------------------------------
    DEN = 20.0
    ax = s.panel(0.50, 1.52, (-1150, 2450), (-700, 4150), DEN,
                 "A   PLAN — BASE SPACING AND THE STEEL OVER IT",
                 f"viewed from below (arms hang toward the reader) · "
                 f"scale 1 : {DEN:.0f}")

    ax.add_patch(Rectangle((0, 0), CW, CL, fc="#fbf8f2", ec=INK, lw=HEAVY,
                           zorder=2))
    ax.text(CW * 0.5, CL * 0.50, "CANVAS", ha="center", va="center",
            fontsize=FS_LBL + 4.0, color="#d8d0c0", zorder=2.2,
            fontweight="bold")
    ax.text(CW * 0.5, CL * 0.485, f"{CW:.1f} x {CL:.2f}   ·   z = 0",
            ha="center", va="top", fontsize=FS_NOTE, color="#d8d0c0",
            zorder=2.2)

    # certified drawing area
    if ca:
        ax.add_patch(Rectangle((ca["x0"], ca["y0"]), ca["w"], ca["h"],
                               fc="#eaf3ee", ec=GRN, lw=1.15,
                               ls=(0, (5, 3)), zorder=2.5, alpha=0.6))
        ax.text(ca["x0"] + ca["w"] / 2, ca["y1"] - 60,
                f"CERTIFIED DRAWING AREA  {ca['w']:.0f} x {ca['h']:.0f}\n"
                f"{ca['area']:.3f} m²  at  h = {h:.0f}",
                ha="center", va="top", fontsize=FS_NOTE, color=GRN,
                fontweight="bold", zorder=3.2, linespacing=1.35, bbox=TBOX)

    # perimeter frame
    for r in ((FR_X0, FR_Y0, IN_X0, FR_Y1), (IN_X1, FR_Y0, FR_X1, FR_Y1),
              (IN_X0, FR_Y0, IN_X1, FR_Y0 + P),
              (IN_X0, FR_Y1 - P, IN_X1, FR_Y1)):
        member(ax, *r, fc=STEEL2, z=5)
    # corner legs, seen end-on through the rails
    for cx, cy in ((FR_X0, FR_Y0), (FR_X1 - P, FR_Y0), (FR_X0, FR_Y1 - P),
                   (FR_X1 - P, FR_Y1 - P)):
        member(ax, cx, cy, cx + P, cy + P, fc="#3b424c", ec=INK, lw=0.8,
               z=5.4)
    # the seam frame — items I and J, straight out of system_model
    seam_centre_line(ax, FR_X0 - 150, FR_X1 + 150)
    draw_seam(ax, 0, 1)

    # runways + clusters
    for ry in ROW_Y:
        member(ax, IN_X0, ry - P, IN_X1, ry + P, fc=STEEL, z=5)
        ax.add_line(Line2D([IN_X0, IN_X1], [ry, ry], color="#4d5561",
                           lw=0.45, zorder=5.6))
    for aid, (xa, ya) in ARMS.items():
        for px in post_xs(xa):
            for py in (ya - P, ya):
                member(ax, px - P / 2, py, px + P / 2, py + P, fc=POSTC,
                       ec=INK, lw=0.6, z=6.5)
        # gussets, rotated onto the runway's outboard y faces
        for px in post_xs(xa):
            for y0 in (ya - P - GUSSET[1], ya + P):
                member(ax, px - GUSSET[0] / 2, y0, px + GUSSET[0] / 2,
                       y0 + GUSSET[1], fc="#c3cad3", ec=INK, lw=0.4, z=6.2,
                       alpha=0.9)
        cx = plate_cx(xa)
        member(ax, cx - PLATE[0] / 2, ya - PLATE[1] / 2, cx + PLATE[0] / 2,
               ya + PLATE[1] / 2, fc=PLATEC, ec=INK, lw=0.9, z=6.8)
        cmark(ax, xa, ya, 150, z=7.5)
        ax.add_patch(Circle((xa, ya), 17, fc=INK, ec="none", zorder=8))
        ax.text(xa - 195, ya + 120, f"{aid}", ha="center", va="bottom",
                fontsize=FS_LBL + 2.4, fontweight="bold", color=INK,
                zorder=8.5)
        # connector side: the connector faces AWAY from the front, so it runs
        # along `clock_sign` — the same direction the plate offset does
        sg = clock_sign(xa)
        ax.add_patch(FancyArrow(xa + sg * 60, ya, sg * 150, 0, width=12,
                                head_width=44, head_length=48,
                                length_includes_head=True, fc=ORIGC,
                                ec="none", zorder=8.4))

    # ---- dimension chains, x ------------------------------------------
    for xv in (0, COL_X[0], COL_X[1], CW):
        s.ext(ax, xv, 0, xv, -640)
    s.dim_h(ax, 0, COL_X[0], -300, f"{COL_X[0]:.1f}", over=0)
    s.dim_h(ax, COL_X[0], COL_X[1], -300, f"{COL_SP:.1f}\npitch", over=0)
    s.dim_h(ax, COL_X[1], CW, -300, f"{CW - COL_X[1]:.1f}", over=0)
    s.dim_h(ax, 0, CW, -540, f"{CW:.1f}   CANVAS", over=0)
    s.dim_h(ax, FR_X0, FR_X1, 3990, f"{SM.FR_W:.1f}   FRAME OUTSIDE   "
            f"({SM.FR_W / IN:.2f} in)", ext_y=(FR_Y1, FR_Y1), over=25)
    s.dim_h(ax, FR_X0, 0, 3838, f"{SM.MARGIN:.1f}", ext_y=(FR_Y1, CL),
            over=15, outside=True, txt_off=14)

    # ---- dimension chains, y ------------------------------------------
    for yv in (0, ROW_Y[0], ROW_Y[1], ROW_Y[2], CL):
        s.ext(ax, 0, yv, -800, yv)
    s.dim_v(ax, 0, ROW_Y[0], -320, f"{ROW_Y[0]:.2f}", over=0)
    s.dim_v(ax, ROW_Y[0], ROW_Y[1], -320, f"{ROW_SP:.2f}   pitch", over=0)
    s.dim_v(ax, ROW_Y[1], ROW_Y[2], -320, f"{ROW_SP:.2f}   pitch", over=0)
    s.dim_v(ax, ROW_Y[2], CL, -320, f"{CL - ROW_Y[2]:.2f}", over=0)
    s.dim_v(ax, 0, CL, -600, f"{CL:.2f}   CANVAS", over=0)
    s.dim_v(ax, FR_Y0, FR_Y1, 2270, f"{SM.FR_L:.2f}   FRAME OUTSIDE",
            ext_x=(FR_X1, FR_X1), over=25, txt_off=-24)

    # datum
    ax.add_patch(Circle((0, 0), 42, fc="white", ec=DIMC, lw=1.3, zorder=10))
    ax.plot([0], [0], marker="+", ms=9, mew=1.4, color=DIMC, zorder=10.5)
    ax.text(120, 90, "DATUM (0,0)\ncanvas reference corner", ha="left",
            va="bottom", fontsize=FS_NOTE, color=DIMC, zorder=10,
            fontweight="bold", linespacing=1.35)
    ax.annotate("", xy=(560, -110), xytext=(60, -110),
                arrowprops=dict(arrowstyle="-|>", color=DIMC, lw=0.8))
    ax.text(600, -110, "x", ha="left", va="center", fontsize=FS_LBL,
            color=DIMC, fontweight="bold")
    ax.annotate("", xy=(-110, 560), xytext=(-110, 60),
                arrowprops=dict(arrowstyle="-|>", color=DIMC, lw=0.8))
    ax.text(-110, 600, "y", ha="center", va="bottom", fontsize=FS_LBL,
            color=DIMC, fontweight="bold")

    s.leader(ax, (IN_X1 - 260, ROW_Y[2] + P / 2), (2410, ROW_Y[2] + 700),
             "RUNWAY  item C\ntwo 3-in beams side by side (152.4).\n"
             "Seam ON the row line, so every J1\naxis lies on it.",
             ha="right", c=ORIGC, rad=-0.15)
    s.leader(ax, (post_xs(COL_X[1])[1], ROW_Y[1] - P / 2),
             (2410, ROW_Y[1] - 700),
             f"DROP CLUSTER  item D\n4 posts, 2 x 2, pitch "
             f"{POST_PITCH_X} x {P}.\nCut length {post_length(h):.1f} at "
             f"h = {h:.0f}.\nSee detail B.", ha="right", c=ORIGC, rad=0.15)
    s.leader(ax, (FR_X0 + P / 2, FR_Y0 + P / 2), (-1140, -500),
             "CORNER LEG  item E\nto the floor — the cage is\n"
             "self-supporting.  THE MID-SPAN\nLEGS ARE THE SEAM FRAME "
             "(item I).", ha="left", c=WARN, rad=0.12)
    s.leader(ax, (FR_X1 - P / 2, SEAM_Y + P), (2410, 2530),
             f"SEAM FRAME  items I + J\n{SEAM_FLAG}\n"
             f"The rig is TWO half-cages "
             f"{HALF_CAGE_L:.1f} long\nbutted at y = {SEAM_Y:.2f}.  "
             f"{len(seam_posts())} bars + {len(seam_braces())} brace\n"
             f"clusters hold up the two butted END RAILS\n"
             f"— which ARE the middle runway already\n"
             f"drawn, to {SEAM_RAIL_RESIDUAL} mm.  Open item 7.",
             ha="right", c=SEAMC, rad=0.12)
    s.leader(ax, (FR_X0 + P + SEAM_BRACE_RUN / 2, SEAM_Y - P - 30),
             (-1140, 2530),
             f"SEAM BRACE  item J\n{SEAM_BRACE[0]} x {SEAM_BRACE[2]} x "
             f"{SEAM_BRACE[1]}, {SEAM_BRACE_RUN} inboard,\n"
             f"{SEAM_BRACE_H} under the rail.  Each pair\n"
             f"braces into its OWN half-cage.\n"
             f"DASHED — not in the collision model.\n{SEAM_FLAG}",
             ha="left", c=SEAMC, rad=-0.12)
    if CLOCKING == "uniform":
        s.leader(ax, (ARMS[17][0] + 210, ARMS[17][1]), (2410, 640),
                 "CONNECTOR SIDE\nAll six arms are clocked identically,\n"
                 f"yaw = 0, R_world_base = Ry(180): the front\nfaces x = 0 "
                 f"and every connector panel\nfaces the x = {CW:.1f} edge.",
                 ha="right", c=ORIGC, rad=-0.12)
    else:
        s.leader(ax, (ARMS[17][0] + 210, ARMS[17][1]), (2410, 700),
                 "CONNECTOR SIDE — MIRRORED, PER COLUMN\n"
                 "LEFT 13/31/2: yaw = 180, front faces +x,\n"
                 "connector and plate toward x = 0.\n"
                 f"RIGHT 17/71/97: yaw = 0, front faces -x,\n"
                 f"connector and plate toward x = {CW:.1f}.\n"
                 "The columns FACE EACH OTHER and every\n"
                 "cable leaves over its own nearest edge.",
                 ha="right", c=ORIGC, rad=-0.12)
        s.leader(ax, (0.5 * (COL_X[0] + COL_X[1]), ROW_Y[1] - 260),
                 (-1140, 1500),
                 "WHAT THE MIRROR BUYS, AND IT IS STEEL:\n"
                 f"cluster gap across a pair {cluster_gap():.2f}\n"
                 f"  (uniform 216.20)\n"
                 f"gusset pair clearance {gusset_pair_clear():.2f}\n"
                 f"  (uniform 89.20)\n"
                 "+50.30 on both — the plate's 25.15\n"
                 "offset flips with the front.\n"
                 "Workspace is unchanged (+4 cells in\n"
                 "16562, same certified rectangle).",
                 ha="left", c=WARN, rad=0.12)

    # ---------------- panel B: the cluster detail -----------------------
    DEN_B = 5.0
    axb = s.panel(8.02, 5.95, (-350, 350), (-300, 300), DEN_B,
                  "B   DETAIL — ONE DROP CLUSTER IN PLAN",
                  f"at the J1 axis · scale 1 : {DEN_B:.0f}")
    cxp = plate_cx(COL_X[0]) - COL_X[0]   # +/-25.15, this clocking
    member(axb, -350, -P, 350, P, fc=STEEL, z=4)
    axb.add_line(Line2D([-350, 350], [0, 0], color="#4d5561", lw=0.5,
                        zorder=4.4))
    for px in (v - COL_X[0] for v in post_xs(COL_X[0])):
        for py in (-P, 0):
            member(axb, px - P / 2, py, px + P / 2, py + P, fc=POSTC, ec=INK,
                   lw=0.9, z=6)
    for px in (v - COL_X[0] for v in post_xs(COL_X[0])):
        for y0 in (-P - GUSSET[1], P):
            member(axb, px - GUSSET[0] / 2, y0, px + GUSSET[0] / 2,
                   y0 + GUSSET[1], fc="#c3cad3", ec=INK, lw=0.5, z=5.4)
    member(axb, cxp - PLATE[0] / 2, -PLATE[1] / 2, cxp + PLATE[0] / 2,
           PLATE[1] / 2, fc=PLATEC, ec=INK, lw=1.1, z=6.6)
    axb.add_patch(Circle((0, 0), CABLE_R, fc="none", ec=WARN, lw=0.8,
                         ls=(0, (3, 2)), zorder=7))
    cmark(axb, 0, 0, 120, z=7.4)
    axb.add_patch(Circle((0, 0), 9, fc=INK, ec="none", zorder=8))
    axb.add_patch(FancyArrow(46, 0, 84, 0, width=8, head_width=26,
                             head_length=28, length_includes_head=True,
                             fc=ORIGC, ec="none", zorder=8.2))
    axb.text(150, 0, "connector", ha="left", va="center", fontsize=FS_NOTE,
             color=ORIGC, fontweight="bold")
    axb.add_patch(FancyArrow(-46, 0, -84, 0, width=8, head_width=26,
                             head_length=28, length_includes_head=True,
                             fc="#8d949c", ec="none", zorder=8.2))
    axb.text(-150, 0, "front", ha="right", va="center", fontsize=FS_NOTE,
             color="#6a7280")

    pxl, pxr = (v - COL_X[0] for v in post_xs(COL_X[0]))
    s.dim_h(s.__dict__ and axb, pxl, pxr, 232, f"{POST_PITCH_X}",
            ext_y=(P, P), over=10)
    s.dim_h(axb, pxl + P / 2, pxr - P / 2, 160, f"{POST_SLOT} slot",
            ext_y=(P, P), over=8)
    s.dim_h(axb, cxp - PLATE[0] / 2, cxp + PLATE[0] / 2, -158,
            f"{PLATE[0]}  PLATE", ext_y=(-PLATE[1] / 2, -PLATE[1] / 2),
            over=8, txt_off=-22)
    for xv, cc in ((0.0, DIMC), (cxp, "#7a6a52")):
        axb.add_line(Line2D([xv, xv], [-PLATE[1] / 2 - 20, -244], color=cc,
                            lw=0.45, ls=(0, (3, 2)), zorder=8.4))
    s.dim_h(axb, 0, cxp, -232, f"{PLATE_OFF}", over=8, txt_off=-24,
            outside=True, out_len=86)
    s.dim_v(axb, -PLATE[1] / 2, PLATE[1] / 2, -298, f"{PLATE[1]:.0f}",
            ext_x=(cxp - PLATE[0] / 2, cxp - PLATE[0] / 2), over=8,
            txt_off=-20)
    s.dim_v(axb, -P, P, 318, f"{2 * P}", ext_x=(pxr + P / 2, pxr + P / 2),
            over=8, txt_off=18)
    axb.text(0, 286, f"plate offset {PLATE_OFF} mm toward the connector "
                     f"edge — DIRECTION TBC (open item 1)", ha="center",
             va="bottom", fontsize=FS_NOTE, color=WARN, fontweight="bold")
    axb.text(0, -292, f"{PLATE_CLR} clear each side of the plate in the "
                      f"{POST_SLOT} slot — a plate offset the WRONG way "
                      f"does not fit", ha="center", va="top",
             fontsize=FS_NOTE, color=WARN)
    axb.text(CABLE_R * 0.72, CABLE_R * 0.72,
             f"base cable envelope\nr {CABLE_R:.0f} — needs a hole\n"
             f"in the plate AND the\nclamp stack (open item 2)",
             ha="left", va="bottom", fontsize=FS_NOTE - 0.4, color=WARN,
             linespacing=1.3, bbox=TBOX)

    # ---------------- panel C: what the plan is for ---------------------
    axc = s.textbox(8.02, 1.52, 5.30, 4.10)
    axc.add_patch(Rectangle((0, 0), 1, 1, fc="#f7f8fa", ec="#c3cad3",
                            lw=0.7))
    axc.text(0.022, 0.955, "C   SET-OUT AND WHAT THE SPACING BUYS",
             ha="left", va="top", fontsize=FS_HEAD, fontweight="bold",
             color=INK)
    lines = [
        (GRN, True, "Base positions are to the J1 AXIS — the centre of the "
                    "base bolt circle — not to a plate edge."),
        (INK, False, f"columns  x = {COL_X[0]:.1f} / {COL_X[1]:.1f}"
                     f"      pitch {COL_SP:.1f}"),
        (INK, False, f"rows     y = {ROW_Y[0]:.2f} / {ROW_Y[1]:.2f} / "
                     f"{ROW_Y[2]:.2f}      pitch {ROW_SP:.2f}"),
        (INK, False, "Tolerance +/-10 mm per base; record any as-built "
                     "offset rather than re-centring the others."),
        (None, False, ""),
    ]
    if CLOCKING == "uniform":
        lines += [
            (INK, True, "ORIENTATION — load-bearing, and identical for all "
                        "six."),
            (INK, False, "R_world_base = Ry(180), yaw = 0. The flipped base's "
                         "+x (the arm's front) points toward x = 0, so every "
                         f"connector panel faces the x = {CW:.1f} long edge. "
                         "Do not clock any arm differently: the certified "
                         "coverage and every program assume this exact "
                         "uniform orientation."),
            (None, False, ""),
        ]
    else:
        lines += [
            (WARN, True, "ORIENTATION — MIRRORED.  THIS IS A VARIANT SHEET, "
                         "NOT THE SHIPPED ONE."),
            (INK, False, "LEFT column 13/31/2: R_world_base = Ry(180) @ "
                         "Rz(180), front toward +x. RIGHT column 17/71/97: "
                         "Ry(180), front toward -x. The two columns face each "
                         f"other across x = {0.5 * (COL_X[0] + COL_X[1]):.1f}, "
                         "and each column's connector panel and plate offset "
                         "run toward its OWN nearest long edge — so no cable "
                         "is dressed across the paper."),
            (WARN, False, f"It opens the steel: cluster gap "
                          f"{cluster_gap():.2f} (uniform 216.20) and gusset "
                          f"pair clearance {gusset_pair_clear():.2f} (uniform "
                          f"89.20), +50.30 on both. The plate offset "
                          "DIRECTION is still inferred (open item 1) and now "
                          "matters twice, once per column — and the two "
                          "columns need HANDED cluster parts."),
            (WARN, False, "IT DOES NOT RE-EARN THE WORKSPACE. Mirrored "
                          "measures +4 live cells in 16562 and the SAME "
                          "certified rectangle, and it needs its own park "
                          "set: the shipped literals re-seated on mirrored "
                          "bases park two arms 100.1 mm inside each other. "
                          "See docs/DECISIONS.md 2026-09-10."),
            (None, False, ""),
        ]
    if ca:
        lines += [
            (GRN, True, f"CERTIFIED DRAWING AREA at h = {h:.0f}: "
                        f"{ca['w']:.0f} x {ca['h']:.0f} mm, "
                        f"{ca['area']:.3f} m²."),
            (INK, False, f"x {ca['x0']:.0f}..{ca['x1']:.0f}, "
                         f"y {ca['y0']:.0f}..{ca['y1']:.0f} from the datum "
                         f"corner. {ca['live_pct']:.2f} % of the canvas is "
                         f"live, but only this rectangle is hole-free. Read "
                         f"at run time from {ca['source']} and drawn dashed "
                         f"on panel A, so the spacing can be read against "
                         f"what it actually buys."),
            (None, False, ""),
        ]
    lines += [
        (WARN, True, "THE STEEL DOES NOT FIT THE CERTIFIED KEEP-OUT."),
        (WARN, False, f"Every certified number in this repo was earned "
                      f"against a single {BOOM_D:.0f} x {BOOM_D:.0f} mm "
                      f"column on the base axis. The real cluster is "
                      f"{CLUSTER_W} x {CLUSTER_D} offset {PLATE_OFF} mm off "
                      f"it, plus {GUSSET[0]} mm gussets: all 60 pieces of "
                      f"mount hardware fall outside it, worst 185.55 mm. "
                      f"Re-certification is minutes of compute and it has "
                      f"not been run — open item 6."),
    ]
    _flow(axc, lines, 0.022, 0.885, width=88, fs=6.35, dy=0.0272)

    # ---------------- notes column --------------------------------------
    axn = s.textbox(13.55, 1.52, 2.55, 8.85)
    axn.add_patch(Rectangle((0, 0), 1, 1, fc="white", ec="#c3cad3", lw=0.7))
    axn.text(0.04, 0.982, "NOTES", ha="left", va="top", fontsize=FS_HEAD,
             fontweight="bold", color=INK)
    notes = _plan_notes(h, z, ca)
    yy = 0.955
    for kind, t in notes:
        col = {"h": INK, "w": WARN, "n": NOTEC, "g": GRN}[kind]
        axn.text(0.04, yy, t, ha="left", va="top",
                 fontsize=6.6 if kind == "h" else 6.0, color=col,
                 fontweight="bold" if kind in "hw" else "normal",
                 linespacing=1.42)
        yy -= 0.0125 * (1 + t.count("\n")) + (0.011 if kind == "h" else 0.006)

    s.title_block(h, 1, "ARM SPACING — PLAN", extra=SEAM_BANNER)
    _save(fig, out_dir, "arm_spacing_topdown")


def _plan_notes(h, z, ca):
    n = []
    n.append(("h", "1  DATUM"))
    n.append(("n", f"z = 0 is the TOP SURFACE OF THE PAPER as laid\n"
                   f"on the table.  The floor is {FLOOR_Z} mm; the\n"
                   f"paper therefore sits {PAPER_ABOVE_FLOOR} mm above it.\n"
                   f"x across the short side (0 -> {CW:.1f}), y along\n"
                   f"the long side (0 -> {CL:.2f}), z up."))
    n.append(("h", "2  THE HEIGHT ON THIS SHEET"))
    n.append(("g", f"h = {h:.1f} mm.  Drop post {post_length(h):.1f} mm."))
    n.append(("n", f"Alternates, same sheet, same steel:\n"
                   f"    h = {H_TABLE[0]:.0f}   post {post_length(970):.1f}"
                   f"   (this sheet)\n"
                   f"    h = {H_TABLE[1]:.0f}   post {post_length(940):.1f}"
                   f"   (what ships)\n"
                   f"    h = {H_TABLE[2]:.0f}   post {post_length(850):.1f}"
                   f"   (built today)\n"
                   f"Only item D changes with h."))
    n.append(("h", "3  MEMBER SECTIONS"))
    n.append(("n", f"3\" x 3\" ({P} mm) square T-slot throughout,\n"
                   f"the original drawing's own MTEXT.  Gussets\n"
                   f"1.5\" x 3\" stock, {GUSSET[0]} x {GUSSET[2]}."))
    n.append(("h", "4  THE RUNWAY SEAM"))
    n.append(("n", f"Each row is TWO 3-in beams side by side\n"
                   f"({2 * P:.1f} overall).  The seam lands ON the row\n"
                   f"line, so each arm's J1 axis lies on it and\n"
                   f"the 2 x 2 cluster straddles it."))
    n.append(("h", f"4b  THE SEAM FRAME — items I + J"))
    n.append(("w", SEAM_FLAG))
    n.append(("n", f"The rig is TWO half-cages {HALF_CAGE_L:.1f} long\n"
                   f"butted at y = {SEAM_Y:.2f}.  Their two seam-side\n"
                   f"END RAILS *ARE* the middle runway already\n"
                   f"drawn ({SEAM_RAIL_RESIDUAL} mm) — do not cut them twice.\n"
                   f"{len(seam_posts())} bars + {len(seam_braces())} brace "
                   f"clusters hold them up."))
    n.append(("h", "5  GUSSETS ARE ROTATED"))
    n.append(("n", f"The drawing's inboard orientation needs\n"
                   f"{SM.GUSSET_NEED} mm across a transverse pair and\n"
                   f"only {SM.GUSSET_GAP} exists.  Rotated onto the\n"
                   f"runway's OUTBOARD y faces they clear by\n"
                   f"{SM.GUSSET_PAIR_CLEAR} mm and brace the cluster's\n"
                   f"weak axis (the posts are only {P} apart in y)."))
    n.append(("h", "6  KEEP-OUT BELOW THE MOUNT PLANE"))
    n.append(("n", f"Nothing but the six arms below z = {h:.0f} over the\n"
                   f"canvas plus 1 m all round.  No braces, no\n"
                   f"cable drops, no lights.  The posts already\n"
                   f"run {POST_OVER} mm past the plate underside."))
    n.append(("h", "7  OPEN ITEMS THAT BLOCK CUTTING"))
    for num, title, _w, _how, blocks in OPEN_ITEMS:
        if blocks.startswith("BLOCKS") and "nothing" not in blocks:
            # the notes column is 2.55 in wide: a title that does not wrap
            # runs off the sheet, and the seam item's title is the flag itself
            n.append(("w", f"  {num}  "
                           + "\n      ".join(textwrap.wrap(title, 36))))
    n.append(("n", "Full text in out/drawings/8020_cut_list.md."))
    n.append(("h", "8  WHAT THIS SHEET IS NOT"))
    n.append(("w", "NOTHING HERE IS A SURVEY."))
    n.append(("n", "The room has never been measured.  Every\n"
                   "ceiling number descends from the original\n"
                   "drawing's 233,7 cm, which is FLOOR to top of\n"
                   "a self-supporting cage."))
    return n


# ===========================================================================
# 6.  SHEET 2 — ELEVATIONS
# ===========================================================================
def sheet_side(h, out_dir):
    s = Sheet()
    fig = s.fig
    z = zl(h)
    pl = z["post_length"]
    CEIL_NOTE = 2050.0        # where the "survey required" ceiling line goes

    # ---------------- panel A: section in x-z ---------------------------
    DEN = 15.0
    ax = s.panel(0.50, 1.55, (-900, 2700), (-960, 2190), DEN,
                 "A   ELEVATION — SECTION AT AN ARM ROW",
                 f"looking along the paper's LONG axis (+y) · "
                 f"scale 1 : {DEN:.0f}")
    x0v, x1v = -880, 2680

    # room ceiling: unknown
    ax.add_line(Line2D([x0v, x1v], [CEIL_NOTE, CEIL_NOTE], color=WARN,
                       lw=1.0, ls=(0, (7, 4)), zorder=3))
    for i in range(30):
        xq = x0v + (x1v - x0v) * i / 29
        ax.add_line(Line2D([xq, xq - 60], [CEIL_NOTE, CEIL_NOTE + 62],
                           color=WARN, lw=0.4, alpha=0.55, zorder=2.8))
    ax.text((x0v + x1v) / 2, CEIL_NOTE + 82,
            "ROOM CEILING — SURVEY REQUIRED.  The cage is self-supporting; "
            "no room ceiling has ever been measured, and none is drawn.",
            ha="center", va="bottom", fontsize=FS_NOTE, color=WARN,
            fontweight="bold")

    # floor / table / paper
    ground(ax, x0v, x1v, FLOOR_Z, depth=110)
    member(ax, IN_X0, FLOOR_Z, IN_X1, TABLE_TOP_Z, fc="#8a6f56", ec=INK,
           lw=0.7, z=3.2, alpha=0.75)
    ax.text(IN_X1 - 70, (FLOOR_Z + TABLE_TOP_Z) / 2, "TABLE  (footprint "
            "ASSUMED)", ha="right", va="center", fontsize=FS_NOTE,
            color="white", fontweight="bold", zorder=3.4)
    member(ax, 0, TABLE_TOP_Z, CW, 0, fc="#fbf8f2", ec=INK, lw=1.4, z=3.6)
    ax.text(CW / 2, 44, f"PAPER  —  z = 0, and it sits "
                        f"{PAPER_ABOVE_FLOOR} mm above the floor",
            ha="center", va="bottom", fontsize=FS_NOTE, color=INK,
            fontweight="bold", zorder=8, bbox=TBOX)

    # perimeter rails, seen end-on, sitting on the legs
    for cx in (FR_X0, IN_X1):
        member(ax, cx, GRID_U, cx + P, GRID_T, fc=STEEL2, ec=INK, lw=0.8,
               z=5.2)
        member(ax, cx, LEG_BOTTOM, cx + P, GRID_U, fc="#3b424c", ec=INK,
               lw=0.8, z=5.0)
    # THE SEAM FRAME, items I + J.  This section is taken at an arm row, and
    # the MIDDLE arm row IS the seam plane, so at that row these four posts
    # stand exactly in the section — in this projection they land on the same
    # x bands as the corner legs, which is why they are hatched.  The braces
    # run inboard and are the one part of the seam that x-z shows on its own.
    draw_seam(ax, 0, 2, z=5.6, alpha=0.55)
    # runway, cut through
    member(ax, IN_X0, GRID_U, IN_X1, GRID_T, fc=STEEL, ec=INK, lw=0.8, z=4.6)
    for i in range(70):
        xq = IN_X0 + (IN_X1 - IN_X0) * i / 69
        ax.add_line(Line2D([xq, xq - 26], [GRID_U, GRID_T], color=INK,
                           lw=0.3, alpha=0.35, zorder=4.7))
    ax.text(IN_X1 - 60, (GRID_U + GRID_T) / 2, "RUNWAY  item C  (cut)",
            ha="right", va="center", fontsize=FS_NOTE, color="white",
            fontweight="bold", zorder=5)

    # the two arms of a transverse pair
    lowest = np.inf
    for xa, aid in ((COL_X[0], 31), (COL_X[1], 71)):
        lowest = min(lowest, draw_park(ax, aid, h, 0, 2))
    for xa in COL_X:
        pxl, pxr = post_xs(xa)
        cxp = plate_cx(xa)
        # gussets — BEYOND the section plane: they are rotated onto the
        # runway's outboard y faces, so in x-z they project as a square
        # centred on each post, top flush with the top of steel
        for px in (pxl, pxr):
            ax.add_patch(Rectangle((px - GUSSET[0] / 2, z["gusset_bottom"]),
                                   GUSSET[0], GRID_T - z["gusset_bottom"],
                                   fc="none", ec="#8d949c", lw=0.55,
                                   ls=(0, (4, 2)), zorder=5.4))
        for px in (pxl, pxr):
            member(ax, px - P / 2, z["post_bottom"], px + P / 2, GRID_U,
                   fc=POSTC, ec=INK, lw=0.85, z=6)
        member(ax, cxp - CLAMP[0] / 2, z["plate_top"], cxp + CLAMP[0] / 2,
               z["clamp_top"], fc="#c3cad3", ec=INK, lw=0.8, z=6.2)
        member(ax, cxp - PLATE[0] / 2, h, cxp + PLATE[0] / 2, z["plate_top"],
               fc=PLATEC, ec=INK, lw=1.1, z=6.5)
        # base cable stub
        ax.add_patch(Rectangle((xa - CABLE_R, h), 2 * CABLE_R, CABLE_UP,
                               fc="none", ec=WARN, lw=0.8, ls=(0, (3, 2)),
                               zorder=7))
        cmark(ax, xa, h, 240, z=7.2)

    ax.add_line(Line2D([-780, 1960], [h, h], color=GRN, lw=1.0,
                       ls=(0, (7, 3)), zorder=8))

    # dimensions
    s.dim_v(ax, z["post_bottom"], GRID_U, -300,
            f"{pl:.1f}\nDROP POST  item D  x24",
            ext_x=(post_xs(COL_X[0])[0] - P / 2, IN_X0), over=0,
            txt_off=-40)
    s.dim_v(ax, 0.0, h, -640, f"{h:.1f}   MOUNT PLANE h",
            ext_x=(0, post_xs(COL_X[0])[0]), over=0, txt_off=-40)
    s.dim_v(ax, FLOOR_Z, 0.0, -800, f"{PAPER_ABOVE_FLOOR}   paper "
            f"above the floor", ext_x=(0, 0), over=0, txt_off=-38)
    s.dim_v(ax, FLOOR_Z, GRID_T, 2600, f"{SM.CAGE_TOTAL_H}   floor to top "
            f"of steel  =  the drawing's 233,7 cm",
            ext_x=(FR_X1, FR_X1), over=0, txt_off=26)
    s.dim_v(ax, GRID_U, GRID_T, 2130, f"{P}", ext_x=(FR_X1, FR_X1), over=0,
            txt_off=20)

    # level ladder
    LX = 1990.0
    ax.add_line(Line2D([LX, LX], [z["post_bottom"] - 60, GRID_T + 60],
                       color=DIMC, lw=0.5, zorder=8))
    for zz, lbl, dy in ((GRID_T, f"{GRID_T}   TOP OF STEEL", 0),
                        (GRID_U, f"{GRID_U}   RUNWAY UNDERSIDE", 0),
                        (z["clamp_top"], f"{z['clamp_top']:.2f}   clamp top",
                         0),
                        (z["plate_top"], f"{z['plate_top']:.2f}   plate top",
                         66),
                        (h, f"{h:.2f}   MOUNT PLANE  h", 0),
                        (z["post_bottom"],
                         f"{z['post_bottom']:.2f}   post bottom", -66)):
        s.ext(ax, post_xs(COL_X[1])[1] + P / 2, zz, LX, zz)
        ax.add_line(Line2D([LX - 26, LX + 26], [zz, zz], color=DIMC, lw=0.9,
                           zorder=8.5))
        if dy:
            ax.add_line(Line2D([LX + 26, LX + 62], [zz, zz + dy], color=DIMC,
                               lw=0.5, zorder=8.5))
        bold = "MOUNT" in lbl or "UNDERSIDE" in lbl or "TOP OF STEEL" in lbl
        ax.text(LX + (62 if dy else 34), zz + dy, lbl, ha="left",
                va="center", fontsize=FS_NOTE - 0.2,
                color=GRN if "MOUNT" in lbl else DIMC, zorder=9, bbox=TBOX,
                fontweight="bold" if bold else "normal")

    s.leader(ax, (COL_X[0], h + CABLE_UP * 0.6), (-820, 1560),
             f"BASE CABLE STUB — CLEARANCE REQUIRED\n"
             f"{CABLE_UP} mm past the flange, r {CABLE_R:.0f}.  Every arm "
             f"is inverted,\nso it points UP through the {PLATE[2]} plate "
             f"and the {CLAMP[2]}\nclamp stack.  BOTH NEED CUTTING (open "
             f"item 2).\nVertically there is room: it tops out at "
             f"{h + CABLE_UP:.1f},\nwell under the runway at {GRID_U}.",
             ha="left", c=WARN, rad=-0.12)
    s.leader(ax, (COL_X[1] + 300, 430), (2660, 320),
             f"ARM AT PARK — ENVELOPE ONLY\nCapsule silhouette at "
             f"layout.Q_PARK_PROPOSED, which\nwas certified at h = "
             f"{H_SHIPPED:.0f} and is drawn here translated\nto this "
             f"sheet's height.  The envelope bottoms out\n"
             f"{lowest:.0f} mm above the paper.  NOT a certified pose\n"
             f"at h = {h:.0f}, and not a clearance claim.",
             ha="right", c=NOTEC, rad=0.14)
    s.leader(ax, (post_xs(COL_X[1])[1], z["gusset_bottom"] + 60),
             (2530, 1300),
             f"GUSSET  item F  —  BEYOND THE SECTION\n{GUSSET[0]} x "
             f"{GUSSET[2]} x {GUSSET[1]}, four per arm, rotated onto the\n"
             f"runway's OUTBOARD y faces, top flush with the steel.",
             ha="right", c=NOTEC, rad=-0.12)
    s.leader(ax, (FR_X0 + P + SEAM_BRACE_RUN * 0.6,
                  GRID_U - SEAM_BRACE_H / 2), (-880, 1985),
             f"SEAM FRAME  items I + J  —  {SEAM_FLAG}\n"
             f"AT THE MIDDLE ROW THIS SECTION IS THE SEAM: the two "
             f"half-cages\nbutt at y = {SEAM_Y:.2f}, and "
             f"{len(seam_posts())} representative bars (hatched, on the "
             f"corner-leg\nx bands) carry the butted end rails — item C's "
             f"middle runway.\nThe {len(seam_braces())} brace clusters run "
             f"{SEAM_BRACE_RUN} inboard, {SEAM_BRACE_H} under the rail — "
             f"DASHED, not in the collision model.",
             ha="left", c=SEAMC, rad=0.10)

    # ---------------- panel B: section in y-z ---------------------------
    DEN_B = 30.0
    # the lower limit carries THREE lines of seam note under the floor hatch
    axb = s.panel(9.95, 6.22, (-620, 4260), (-1310, 2050), DEN_B,
                  "B   ELEVATION — SECTION ALONG THE SHORT AXIS",
                  f"looking along +x · scale 1 : {DEN_B:.0f}")
    ground(axb, -600, 4240, FLOOR_Z, depth=150)
    member(axb, FR_Y0 + P, FLOOR_Z, FR_Y1 - P, TABLE_TOP_Z, fc="#8a6f56",
           ec=INK, lw=0.6, z=3.2, alpha=0.75)
    member(axb, 0, TABLE_TOP_Z, CL, 0, fc="#fbf8f2", ec=INK, lw=0.9, z=3.6)
    for cy in (FR_Y0, FR_Y1 - P):
        member(axb, cy, GRID_U, cy + P, GRID_T, fc=STEEL2, ec=INK, lw=0.7,
               z=5.2)
        member(axb, cy, LEG_BOTTOM, cy + P, GRID_U, fc="#3b424c", ec=INK,
               lw=0.7, z=5.0)
    for ry in ROW_Y:
        member(axb, ry - P, GRID_U, ry + P, GRID_T, fc=STEEL, ec=INK,
               lw=0.7, z=5.4)
    for aid in (13, 31, 2):
        draw_park(axb, aid, h, 1, 2)
    for ry in ROW_Y:
        for py in (ry - P, ry):
            member(axb, py, z["post_bottom"], py + P, GRID_U, fc=POSTC,
                   ec=INK, lw=0.7, z=6)
        member(axb, ry - CLAMP[1] / 2, z["plate_top"], ry + CLAMP[1] / 2,
               z["clamp_top"], fc="#c3cad3", ec=INK, lw=0.6, z=6.2)
        member(axb, ry - PLATE[1] / 2, h, ry + PLATE[1] / 2, z["plate_top"],
               fc=PLATEC, ec=INK, lw=0.9, z=6.5)
    # THE SEAM FRAME, items I + J — this is the view that shows it.  The four
    # posts run TABLETOP to runway underside in the middle row's own y band,
    # so from the tabletop up to the drop-post bottoms they stand alone: the
    # mid-span legs.  Above that they project onto the middle cluster, hence
    # the hatch.
    draw_seam(axb, 1, 2, lw=0.6, z=6.55, alpha=0.45)
    seam_centre_line(axb, LEG_BOTTOM - 260, GRID_T + 150, horizontal=False)
    axb.add_line(Line2D([-560, 4200], [h, h], color=GRN, lw=0.9,
                        ls=(0, (7, 3)), zorder=8))

    for yv in (FR_Y0 + P, ROW_Y[0], ROW_Y[1], ROW_Y[2], FR_Y1 - P):
        s.ext(axb, yv, GRID_U, yv, GRID_U + 620)
    s.dim_h(axb, FR_Y0 + P, ROW_Y[0], GRID_U + 300,
            f"{ROW_Y[0] - FR_Y0 - P:.1f}", over=0, txt_off=40)
    s.dim_h(axb, ROW_Y[0], ROW_Y[1], GRID_U + 300, f"{ROW_SP:.2f}", over=0,
            txt_off=40)
    s.dim_h(axb, ROW_Y[1], ROW_Y[2], GRID_U + 300, f"{ROW_SP:.2f}", over=0,
            txt_off=40)
    s.dim_h(axb, ROW_Y[2], FR_Y1 - P, GRID_U + 300,
            f"{FR_Y1 - P - ROW_Y[2]:.1f}", over=0, txt_off=40)
    s.dim_h(axb, FR_Y0, FR_Y1, GRID_U + 560, f"{SM.FR_L:.2f}  FRAME OUTSIDE",
            over=0, txt_off=40)
    axb.text((FR_Y0 + FR_Y1) / 2, -1280,
             f"SEAM FRAME  items I + J  —  {SEAM_FLAG}\n"
             f"{len(seam_posts())} bars tabletop to rail at y = "
             f"{SEAM_Y:.2f} — two per side, {P} apart, one per half-cage — "
             f"plus {len(seam_braces())} brace clusters (dashed)\n"
             f"THESE ARE THE MID-SPAN LEGS OF OPEN ITEM 3: clear runway span "
             f"corner to seam {seam_clear_span():.2f} mm",
             ha="center", va="bottom", fontsize=FS_NOTE, color=SEAMC,
             fontweight="bold", linespacing=1.5)
    s.dim_v(axb, LEG_BOTTOM, GRID_U, 4110, f"{LEG_LEN}   LEG  item E",
            ext_x=(FR_Y1, FR_Y1), over=0, txt_off=24)
    axb.text((FR_Y0 + FR_Y1) / 2, 790,
             f"Rows sit on the runway seam lines; each row is a double beam, "
             f"{2 * P:.1f} in y.  Arms shown at park — ENVELOPE ONLY.",
             ha="center", va="center", fontsize=FS_NOTE - 0.3, color=NOTEC,
             bbox=TBOX)

    # ---------------- panel C: the mount stack detail -------------------
    DEN_C = 6.0
    axc = s.panel(9.95, 1.95, (-300, 300), (h - 210, h + 400), DEN_C,
                  "C   DETAIL — THE MOUNT STACK",
                  f"scale 1 : {DEN_C:.0f}")
    pxl, pxr = (v - COL_X[0] for v in post_xs(COL_X[0]))
    cxp = plate_cx(COL_X[0]) - COL_X[0]
    for px in (pxl, pxr):
        member(axc, px - P / 2, z["post_bottom"], px + P / 2, h + 400,
               fc=POSTC, ec=INK, lw=0.9, z=6)
    # break symbol on the posts
    for px in (pxl, pxr):
        for dy in (0, 26):
            axc.add_line(Line2D([px - P / 2 - 8, px + P / 2 + 8],
                                [h + 320 + dy - 12, h + 320 + dy + 12],
                                color="white", lw=2.4, zorder=6.6))
            axc.add_line(Line2D([px - P / 2 - 8, px + P / 2 + 8],
                                [h + 320 + dy - 12, h + 320 + dy + 12],
                                color=INK, lw=0.6, zorder=6.7))
    member(axc, cxp - CLAMP[0] / 2, z["plate_top"], cxp + CLAMP[0] / 2,
           z["clamp_top"], fc="#c3cad3", ec=INK, lw=0.9, z=6.2)
    member(axc, cxp - PLATE[0] / 2, h, cxp + PLATE[0] / 2, z["plate_top"],
           fc=PLATEC, ec=INK, lw=1.2, z=6.5)
    axc.add_patch(Rectangle((-CABLE_R, h), 2 * CABLE_R, CABLE_UP, fc="none",
                            ec=WARN, lw=0.9, ls=(0, (3, 2)), zorder=7))
    # the arm's base flange, just below
    member(axc, -75, h - 150, 75, h, fc="#eae6df", ec=INK, lw=0.9, z=6.4)
    axc.text(0, h - 158, "arm hangs below", ha="center", va="top",
             fontsize=FS_NOTE, color="#6a7280")
    axc.add_line(Line2D([-300, 300], [h, h], color=GRN, lw=1.1,
                        ls=(0, (7, 3)), zorder=8))
    cmark(axc, 0, h + 90, 150, z=7.2)

    s.dim_v(axc, h, z["plate_top"], -252, f"{PLATE[2]}",
            ext_x=(cxp - PLATE[0] / 2, cxp - PLATE[0] / 2), over=8,
            txt_off=-20)
    s.dim_v(axc, z["plate_top"], z["clamp_top"], -178, f"{CLAMP[2]}",
            ext_x=(cxp - CLAMP[0] / 2, cxp - CLAMP[0] / 2), over=8,
            txt_off=-20)
    s.dim_v(axc, z["post_bottom"], h, 262, f"{POST_OVER}",
            ext_x=(pxr + P / 2, cxp + PLATE[0] / 2), over=8, txt_off=20)
    s.dim_v(axc, h, h + CABLE_UP, 150, f"{CABLE_UP}", ext_x=(0, 0), over=6,
            txt_off=18)
    s.dim_h(axc, pxl + P / 2, pxr - P / 2, h + 372, f"{POST_SLOT} slot",
            over=0, txt_off=12)
    axc.text(0, h - 196, f"PLATE {PLATE[0]} x {PLATE[1]} x {PLATE[2]}  "
                         f"item G          CLAMP {CLAMP[0]} x {CLAMP[1]} x "
                         f"{CLAMP[2]}  item H", ha="center", va="bottom",
             fontsize=FS_NOTE, color=INK)
    axc.text(-294, h - 120,
             f"BASE CABLE {CABLE_UP} — CLEARANCE\nREQUIRED through the "
             f"plate AND\nthe clamp stack (open item 2)",
             ha="left", va="top", fontsize=FS_NOTE - 0.3, color=WARN,
             fontweight="bold", linespacing=1.3, zorder=9, bbox=TBOX)

    # ---------------- the three-height table ----------------------------
    axt = s.textbox(14.15, 1.95, 1.95, 4.05)
    axt.add_patch(Rectangle((0, 0), 1, 1, fc="#f7f8fa", ec="#c3cad3",
                            lw=0.7))
    axt.text(0.06, 0.965, "DROP POST — item D", ha="left", va="top",
             fontsize=7.4, fontweight="bold", color=INK)
    axt.text(0.06, 0.912, "24 off.  The ONLY cut that\nchanges with the "
                          "mount height.", ha="left", va="top",
             fontsize=5.9, color=NOTEC, linespacing=1.4)
    axt.text(0.06, 0.836, f"length = ({GRID_U} - h) + {POST_OVER}",
             ha="left", va="top", fontsize=6.2, color=DIMC,
             fontweight="bold")
    axt.add_line(Line2D([0.06, 0.94], [0.812, 0.812], color="#c3cad3",
                        lw=0.6))
    axt.text(0.06, 0.795, "h  (mm)", ha="left", va="top", fontsize=6.0,
             color="#6a7280", fontweight="bold")
    axt.text(0.94, 0.795, "post  (mm)", ha="right", va="top", fontsize=6.0,
             color="#6a7280", fontweight="bold")
    yy = 0.748
    for hh in H_TABLE:
        this = abs(hh - h) < 1e-6
        axt.text(0.06, yy, f"{hh:.0f}", ha="left", va="top",
                 fontsize=8.2 if this else 7.2,
                 color=GRN if this else INK,
                 fontweight="bold" if this else "normal")
        axt.text(0.94, yy, f"{post_length(hh):.1f}", ha="right", va="top",
                 fontsize=8.2 if this else 7.2,
                 color=GRN if this else INK,
                 fontweight="bold" if this else "normal")
        tag = {970.0: "THIS SHEET — the certified-\nworkspace recommendation",
               940.0: "layout.LAYOUT_PROPOSED,\nwhat the software plans "
                      "against",
               850.0: "as built today\n(verticals trimmable)"}[hh]
        axt.text(0.06, yy - 0.030, tag, ha="left", va="top", fontsize=5.5,
                 color=NOTEC, linespacing=1.35)
        yy -= 0.098
    axt.add_line(Line2D([0.06, 0.94], [yy + 0.030, yy + 0.030],
                        color="#c3cad3", lw=0.6))
    axt.text(0.06, yy - 0.005,
             f"Everything else on the cut\nlist is height-independent.\n\n"
             f"SUPERSEDED, do not cut:\n"
             f"    1435.0  at h = 940\n"
             f"    1525.0  at h = 850\n"
             f"from the old 2340 grid datum,\n"
             f"{CEIL_CODE - GRID_U:.1f} mm too long.\n\n"
             f"The original rig's own post\nis {SM.O_POST_L} = 29.00 in — "
             f"the\ncorrected {post_length(940):.1f} at h = 940 lands\n"
             f"within {abs(post_length(940) - SM.O_POST_L):.1f} mm of it.",
             ha="left", va="top", fontsize=5.8, color=INK, linespacing=1.45)

    s.title_block(h, 2, "CAGE — ELEVATIONS AND DROP-POST CUT",
                  extra=SEAM_BANNER)
    _save(fig, out_dir, "cage_side_view")


# ===========================================================================
# 7.  THE CENTRE DATUM — SHEETS 3 AND 4, FOR A TAPE AT THE RIG
# ===========================================================================
# Pete Werner, standing at the rig with a tape, 2026-09-16: *"an updated top
# down drawing that has the measurements between the hanging struts and from
# the center of rotation of the arms to the outside of the hanging struts ...
# reference the measurements from the center because then it is unambiguous
# ... a top down one with all the measurements that is exact."*
#
# Sheets 1 and 2 dimension everything from the canvas reference CORNER.  That
# is the right datum for setting a frame out on an empty floor and the wrong
# one for CHECKING A RIG THAT IS ALREADY STANDING: the corner is buried under
# the steel, every number is a long add-up, and nothing on the sheet says
# which side of anything you are on.  Sheets 3 and 4 carry THE SAME MODEL
# NUMBERS re-datumed onto the TABLE CENTRE, and print every one of them as a
# SIGNED offset from one of two lines:
#
#     X = x_canvas - CENTRE_X    across the short side, + toward x = CANVAS_W
#     Y = y_canvas - CENTRE_Y    along the long side,   + toward y = CANVAS_L
#
# CENTRE_Y is THE SEAM LINE, and it is four things at once — the canvas's own
# mid-length, the middle arm row, the plane the two half-cages butt on, and
# the line the two seam bars straddle.  `system_model` makes all four the same
# number (SEAM_Y == 0.5*(FR_Y0+FR_Y1) == ROW_Y[1] == CANVAS_L/2) and
# `tests/test_draw_8020.py` pins that they stay the same number.
#
# NOT ONE NUMBER ON EITHER SHEET IS TYPED HERE.  Everything printed comes out
# of `centre_dims()` / `side_levels()` / `tape_checks()`, which are arithmetic
# on `system_model` constants, and the test re-derives each one independently.
# The single exception is the TAPE column of `hand_vs_model()`, which is typed
# because it is a measurement somebody made with a tape.
CENTRE_X = round(CW / 2.0, 2)                    # 901.70
CENTRE_Y = SEAM_Y                                # 1815.32

# Pete's tape, 2026-09-16: the table is 416.6 cm end to end.  That is the TWO
# BUTTED HALF-FRAMES — 2 x HALF_CAGE_L = 4165.60, the same arithmetic section
# 3b of `system_model` uses to find the seam — and it is NOT this model's own
# table footprint, which is ASSUMED to be the cage's inner span.  Both are
# drawn; `canvas_on_table()` reports the difference.
TABLE_L_TAPE = round(2 * HALF_CAGE_L, 2)         # 4165.60

CENTRE_BANNER = (
    "  CENTRE-DATUM SET — sheets 3 and 4 dimension the SAME model from the "
    "TABLE CENTRE, signed.  They do not supersede sheets 1 and 2; those are "
    "set-out from the canvas corner, these are for a tape at the built rig.")

# bigger ink than sheets 1 and 2: these get read off a phone at the rig
FS_C_DIM, FS_C_ORD, FS_C_NOTE, FS_C_HEAD = 7.4, 6.2, 7.0, 10.4


def off_x(x):
    """Canvas x (mm) -> SIGNED offset from the long centre line."""
    return round(float(x) - CENTRE_X, 2)


def off_y(y):
    """Canvas y (mm) -> SIGNED offset from the seam line."""
    return round(float(y) - CENTRE_Y, 2)


def sg(v, nd=2):
    """A signed offset, the way both centre sheets print one."""
    return f"{float(v):+.{nd}f}"


def model_table():
    """The model's own table body -> Body."""
    for b in SM.ground_bodies():
        if b.name == "table":
            return b
    raise KeyError("table")                               # pragma: no cover


def canvas_on_table():
    """How the canvas sits on the table, per `system_model` -> dict.

    The sheet has to say this out loud: a dimension "from the table centre"
    is only the same line as "from the canvas centre" if the canvas is
    centred on the table, and that is a MODEL CLAIM, not a measurement.  The
    model's `table` body carries its height from the drawing and its FOOTPRINT
    as an assumption (the cage's inner span), so this function reports the
    eccentricity both ways and the tape length separately.
    """
    t = model_table()
    x0, x1 = off_x(t.lo[0]), off_x(t.hi[0])
    y0, y1 = off_y(t.lo[1]), off_y(t.hi[1])
    ex, ey = round(0.5 * (x0 + x1), 2), round(0.5 * (y0 + y1), 2)
    return dict(
        x0=x0, x1=x1, y0=y0, y1=y1,
        w=round(x1 - x0, 2), l=round(y1 - y0, 2),
        ecc_x=ex, ecc_y=ey,
        centred=bool(abs(ex) < 0.005 and abs(ey) < 0.005),
        over_x=round(x1 - off_x(CW), 2),
        over_y=round(y1 - off_y(CL), 2),
        tape_y=round(TABLE_L_TAPE / 2.0, 2),
        tape_l=TABLE_L_TAPE,
        tape_over_y=round(TABLE_L_TAPE / 2.0 - off_y(CL), 2),
        tape_vs_model=round(TABLE_L_TAPE - (y1 - y0), 2),
        provenance=t.provenance)


def canvas_on_table_lines():
    """The sheet's own paragraph about it -> [(colour, bold, text)]."""
    t = canvas_on_table()
    out = []
    if t["centred"]:
        out.append((GRN, True,
                    f"THE MODEL CENTRES THE CANVAS ON THE TABLE, both ways. "
                    f"The table body's centre is ({sg(t['ecc_x'])}, "
                    f"{sg(t['ecc_y'])}) from the canvas centre — the same two "
                    f"lines — and the table overhangs the canvas by "
                    f"{t['over_x']} mm on all four sides."))
    else:
        out.append((WARN, True,
                    f"THE MODEL DOES NOT CENTRE THE CANVAS ON THE TABLE. The "
                    f"table body's centre is ({sg(t['ecc_x'])}, "
                    f"{sg(t['ecc_y'])}) from the canvas centre, so the TABLE "
                    f"CENTRE and the CANVAS CENTRE ARE DIFFERENT LINES and "
                    f"every dimension on this sheet is from the CANVAS one. "
                    f"Both tables are drawn."))
    out.append((INK, False,
                f"Model table {t['w']} x {t['l']}, X {sg(t['x0'], 1)} .. "
                f"{sg(t['x1'], 1)}, Y {sg(t['y0'], 1)} .. {sg(t['y1'], 1)}. "
                f"Its HEIGHT is the drawing's; its FOOTPRINT is "
                f"{t['provenance']} — the cage's own inner span, because the "
                f"original's 2.08 m table cannot carry this canvas."))
    out.append((WARN, True,
                f"AND THE TAPE DISAGREES ABOUT THE LENGTH: 416.6 cm measured "
                f"= {t['tape_l']} = two butted half-frames, against the "
                f"model's assumed {t['l']} — {t['tape_vs_model']:+.2f} mm. "
                f"THE TABLE ENDS ON THIS SHEET ARE THE TAPE'S, Y = "
                f"{sg(t['tape_y'], 1)}, drawn dashed; the model's assumed "
                f"footprint is drawn solid inside them."))
    return out


def centre_dims():
    """Every dimension sheets 3 and 4 print in plan -> {key: mm}, signed.

    Arithmetic on `system_model` only.  `tests/test_draw_8020.py` re-derives
    every entry from the package independently, so the sheet cannot carry a
    hand-typed plan dimension.
    """
    t = canvas_on_table()
    lo_l, hi_l = post_xs(COL_X[0])                 # left column post centres
    lo_r, hi_r = post_xs(COL_X[1])
    fl = (lo_l - P / 2, lo_l + P / 2, hi_l - P / 2, hi_l + P / 2)
    fr = (lo_r - P / 2, lo_r + P / 2, hi_r - P / 2, hi_r + P / 2)
    return dict(
        # --- the two datum lines, in canvas coordinates ------------------
        centre_x=CENTRE_X, centre_y=CENTRE_Y,
        # --- canvas ------------------------------------------------------
        canvas_edge=off_x(CW), canvas_end=off_y(CL),
        canvas_w=round(CW, 2), canvas_l=round(CL, 2),
        # --- table -------------------------------------------------------
        table_edge=t["x1"], table_end=t["y1"], table_w=t["w"], table_l=t["l"],
        table_end_tape=t["tape_y"], table_len_tape=t["tape_l"],
        table_over=t["over_x"], table_over_end=t["over_y"],
        # --- cage perimeter ----------------------------------------------
        rail_out=off_x(FR_X1), rail_in=off_x(IN_X1),
        rail_out_end=off_y(FR_Y1), rail_in_end=off_y(FR_Y1 - P),
        frame_w=round(SM.FR_W, 2), frame_l=round(SM.FR_L, 2),
        rail_len_x=round(SM.RAIL_LEN_X, 2), rail_len_y=round(SM.RAIL_LEN_Y, 2),
        profile=round(P, 2),
        # --- seam bars ----------------------------------------------------
        seam_bar_face=off_y(SEAM_Y + P), seam_bar_dy=round(2 * P, 2),
        # --- runways -------------------------------------------------------
        runway_mid_face=off_y(ROW_Y[1] + P),
        runway_row_face_in=off_y(ROW_Y[2] - P),
        runway_row_face_out=off_y(ROW_Y[2] + P),
        runway_w=round(2 * P, 2), runway_x=off_x(IN_X1),
        runway_len=round(SM.RAIL_LEN_X, 2),
        # --- arm axes -------------------------------------------------------
        axis_x=off_x(COL_X[1]), axis_pitch_x=round(COL_SP, 2),
        axis_y=off_y(ROW_Y[2]), axis_pitch_y=round(ROW_SP, 2),
        base_circle_d=round(BOOM_D, 2),
        # --- plate and drop posts ------------------------------------------
        plate_off=round(PLATE_OFF, 2),
        plate_cx_l=off_x(plate_cx(COL_X[0])),
        plate_cx_r=off_x(plate_cx(COL_X[1])),
        plate_w=round(PLATE[0], 2), plate_d=round(PLATE[1], 2),
        plate_f_l=[off_x(plate_cx(COL_X[0]) + k * PLATE[0] / 2)
                   for k in (-1, 1)],
        plate_f_r=[off_x(plate_cx(COL_X[1]) + k * PLATE[0] / 2)
                   for k in (-1, 1)],
        post_pitch=round(POST_PITCH_X, 2),
        post_c_l=[off_x(lo_l), off_x(hi_l)],
        post_c_r=[off_x(lo_r), off_x(hi_r)],
        post_f_l=[off_x(v) for v in fl],
        post_f_r=[off_x(v) for v in fr],
        # --- the five numbers Pete asked for --------------------------------
        pair_gap=round((hi_l - P / 2) - (lo_l + P / 2), 2),
        pair_outer_w=round(fl[3] - fl[0], 2),
        axis_face_short=round(min(abs(fl[0] - COL_X[0]),
                                  abs(fl[3] - COL_X[0])), 2),
        axis_face_long=round(max(abs(fl[0] - COL_X[0]),
                                 abs(fl[3] - COL_X[0])), 2),
        inner_pair_gap=round((fr[0] - fl[3]), 2),
    )


# ---------------------------------------------------------------------------
# WHAT A TAPE SAID, 2026-09-16 — the only hand-typed numbers on either sheet
# ---------------------------------------------------------------------------
HAND_TAPE_DATE = "2026-09-16"
_HAND = (
    ("clear gap, the two INNER posts of a row", 214.0, "inner_pair_gap"),
    ("outer width of ONE post pair", 396.0, "pair_outer_w"),
    ("J1 axis -> outside face, SHORT side", 156.0, "axis_face_short"),
    ("J1 axis -> outside face, LONG side", 240.0, "axis_face_long"),
    ("table length, end to end", 4166.0, "table_len_tape"),
)
HAND_FLAG = (
    "ONE ROW IS A REAL DISAGREEMENT AND IT IS THE PLATE OFFSET. Three of the "
    "five readings are within 2.5 mm of the model, which is a tape on 3-in "
    "extrusion. The two axis-to-face readings are not: they put the J1 axis "
    "42 mm off the centre of its own post pair where the model puts it "
    "25.15 mm off — and that 25.15 is the ONE number on the whole mount that "
    "was never measured (open item 1, PLATE OFFSET DIRECTION). RE-MEASURE IT "
    "with a straight edge laid across the base flange and the two post "
    "faces, not by eye off the plate edge.")


def hand_vs_model():
    """Pete's tape against the model -> [(what, tape, model, delta)].

    The derived row — the axis offset inside the pair — is half the difference
    of the two axis-to-face readings above it, computed here rather than typed,
    so it moves if either reading is corrected.
    """
    d = centre_dims()
    tape = {k: t for _w, t, k in _HAND}
    rows = [[w, t, d[k], round(t - d[k], 2)] for w, t, k in _HAND]
    off = round((tape["axis_face_long"] - tape["axis_face_short"]) / 2.0, 2)
    rows.insert(4, ["=> J1 axis offset inside the pair  (derived)", off,
                    d["plate_off"], round(off - d["plate_off"], 2)])
    return rows


# ---------------------------------------------------------------------------
# SHEET 4's HEIGHTS — the same z ladder, printed from BOTH datums
# ---------------------------------------------------------------------------
_LEVELS = (
    ("grid_top", "TOP OF STEEL", "TOP OF STEEL", True),
    ("grid_underside", "RUNWAY UNDERSIDE  (posts hang from here)",
     "RUNWAY UNDERSIDE", True),
    ("gusset_bottom", "gusset bottom", "gusset bottom", False),
    ("clamp_top", "clamp top", "clamp top", False),
    ("plate_top", "plate top", "plate top", False),
    ("mount_plane", "PLATE UNDERSIDE  =  mount plane h", "PLATE UNDERSIDE",
     True),
    ("post_bottom", "drop-post bottom", "post bottom", False),
    ("paper_top", "PAPER TOP  —  the sheet-1/2 datum, 0", "PAPER TOP", True),
    ("table_top", "table top", "table top", False),
    ("leg_bottom", "corner-leg and seam-bar foot", "leg / bar foot", False),
    ("floor", "FLOOR  —  the tape datum", "FLOOR", True),
)


# LABEL JOGS, mm — LAYOUT ONLY, never a dimension.  Four of the eleven levels
# are 2 to 35 mm apart (paper top and table top are 2.00), so on a 1 : 26
# elevation their labels print on top of each other.  Each label is jogged off
# its own tick and joined back to it by a short leader, exactly the way sheet
# 2's level ladder does it.  Moving a number here moves where it is PRINTED
# and nothing else; the tick stays on the level.
_JOG = {"clamp_top": 60.0, "plate_top": 80.0, "post_bottom": -75.0,
        "paper_top": 90.0, "table_top": 20.0, "leg_bottom": -55.0}


def side_levels(h):
    """Each height -> [(key, label, short, above_paper, above_floor, bold)]."""
    z = zl(h)
    return [(k, lbl, sh, round(float(z[k]), 2),
             round(float(z[k]) - FLOOR_Z, 2), b)
            for k, lbl, sh, b in _LEVELS]


def tape_checks(h):
    """The checks Pete can make with a tape and nothing else -> [(what, mm)].
    """
    z = zl(h)
    return [
        ("FLOOR  ->  PLATE UNDERSIDE", round(z["mount_plane"] - z["floor"], 2),
         "stand the tape on the floor under an arm and read the steel the "
         "arm bolts to.  The single most useful check on the rig."),
        ("TABLE TOP  ->  PLATE UNDERSIDE",
         round(z["mount_plane"] - z["table_top"], 2),
         "same reading from the table instead of the floor, for when the "
         "paper is down and the floor is not reachable."),
        ("FLOOR  ->  TABLE TOP", round(z["table_top"] - z["floor"], 2),
         "the table's own height — the drawing's 63,5 cm.  If this is not "
         "what your tape says, EVERY paper-referenced number on sheets 1 "
         "and 2 moves by the difference."),
        ("FLOOR  ->  TOP OF STEEL", round(z["grid_top"] - z["floor"], 2),
         "the whole cage, and it is the original drawing's own 233,7 cm."),
    ]


def side_cuts(h):
    """The three cut lengths sheet 4 carries -> [(item, what, mm, note)]."""
    return [
        ("D", "drop post", round(post_length(h), 2),
         f"({GRID_U} - h) + {POST_OVER},  24 off"),
        ("E", "corner leg", round(LEG_LEN, 2),
         f"leg foot {LEG_BOTTOM} to rail underside {GRID_U},  4 off"),
        ("I", "seam support bar", round(seam_post_length(), 2),
         f"the same cut as a corner leg,  {len(seam_posts())} off"),
    ]


# ---------------------------------------------------------------------------
# ORDINATE DIMENSIONING — the idiom the whole centre datum rests on
# ---------------------------------------------------------------------------
def ord_x(s, ax, y0, ticks, stem=160.0, gap=26.0, fs=FS_C_ORD, c=DIMC):
    """Ordinate dimensions off the X = 0 centre line, read along the bottom.

    ORDINATE, NOT CHAIN, and that is the whole point of these sheets: a chain
    of eight post faces overlaps itself at any plan scale that fits on A3, and
    a chain answers "how far from the last one" when the man with the tape is
    asking "how far from the middle".  Every tick carries its own SIGNED
    offset from the one line.  `ticks` is [(x, text, tier)]; tiers stagger the
    labels so that faces 76.2 mm apart do not collide.
    """
    xs = [t[0] for t in ticks]
    ax.add_line(Line2D([min(xs) - 60, max(xs) + 60], [y0, y0], color=c,
                       lw=THIN, zorder=8.6))
    for x, txt, tier in ticks:
        ye = y0 - stem * (1 + tier)
        ax.add_line(Line2D([x, x], [y0 + 34, ye], color=c, lw=0.45,
                           zorder=8.6))
        ax.add_patch(Circle((x, y0), 9, fc=c, ec="none", zorder=8.7))
        ax.text(x, ye - gap, txt, ha="center", va="top", rotation=90,
                fontsize=fs, color=c, zorder=9, bbox=TBOX)


def ord_y(s, ax, x0, ticks, stem=150.0, gap=30.0, fs=FS_C_ORD, c=DIMC):
    """Ordinate dimensions off the Y = 0 seam line, read up the right edge."""
    ys = [t[0] for t in ticks]
    ax.add_line(Line2D([x0, x0], [min(ys) - 60, max(ys) + 60], color=c,
                       lw=THIN, zorder=8.6))
    for y, txt, tier in ticks:
        xe = x0 + stem * (1 + tier)
        ax.add_line(Line2D([x0 - 34, xe], [y, y], color=c, lw=0.45,
                           zorder=8.6))
        ax.add_patch(Circle((x0, y), 9, fc=c, ec="none", zorder=8.7))
        ax.text(xe + gap, y, txt, ha="left", va="center", fontsize=fs,
                color=c, zorder=9, bbox=TBOX)


def centre_lines(ax, xlim, ylim):
    """The two datum lines, and the mark where they cross."""
    for a, b in (([xlim[0], xlim[1]], [0, 0]), ([0, 0], [ylim[0], ylim[1]])):
        ax.add_line(Line2D(a, b, color=ORIGC, lw=1.0,
                           ls=(0, (14, 4, 2.0, 4)), zorder=7.8))
    ax.add_patch(Circle((0, 0), 62, fc="white", ec=ORIGC, lw=1.5, zorder=10))
    for a, b in (((-96, 96), (0, 0)), ((0, 0), (-96, 96))):
        ax.add_line(Line2D(a, b, color=ORIGC, lw=1.5, zorder=10.4))
    ax.add_patch(Circle((0, 0), 22, fc=ORIGC, ec="none", zorder=10.5))


# ---------------------------------------------------------------------------
# SHEET 3 — THE PLAN, SIGNED OFF THE CENTRE
# ---------------------------------------------------------------------------
def sheet_centre_plan(h, out_dir):
    """out/drawings/centre/plan_centre_datum.pdf / .png — A3 landscape."""
    s = Sheet()
    fig = s.fig
    d = centre_dims()
    t = canvas_on_table()

    def X(v):
        return float(v) - CENTRE_X

    def Y(v):
        return float(v) - CENTRE_Y

    DEN = 22.0
    XL, YL = (-1700.0, 2050.0), (-2900.0, 2400.0)
    ax = s.panel(0.42, 1.58, XL, YL, DEN,
                 "A   PLAN — SIGNED FROM THE TABLE CENTRE",
                 f"viewed from below  ·  scale 1 : {DEN:.0f}")

    # --- the table, both of them ----------------------------------------
    ax.add_patch(Rectangle((t["x0"], -t["tape_y"]), t["w"], t["tape_l"],
                           fc="#f0e6da", ec="#8a6f56", lw=1.0,
                           ls=(0, (6, 3)), zorder=1.6))
    member(ax, t["x0"], t["y0"], t["x1"], t["y1"], fc="#e3d5c4", ec="#8a6f56",
           lw=0.9, z=1.8, alpha=0.9)
    ax.text(0, 1680,
            f"THE TABLE IS DRAWN TWICE\nMODEL footprint {t['w']} x {t['l']} "
            f"({t['provenance']})  —  solid\nTAPE ends Y "
            f"{sg(-t['tape_y'], 1)} / {sg(t['tape_y'], 1)}  "
            f"({t['tape_l']}, two butted half-frames)  —  dashed",
            ha="center", va="center", fontsize=FS_C_NOTE - 1.0,
            color="#7a5c42", fontweight="bold", zorder=2.4, linespacing=1.5,
            bbox=TBOX)

    # --- the canvas -------------------------------------------------------
    ax.add_patch(Rectangle((-d["canvas_edge"], -d["canvas_end"]),
                           d["canvas_w"], d["canvas_l"], fc="#fbf8f2", ec=INK,
                           lw=HEAVY, zorder=2))
    ax.text(-540, 480, f"CANVAS\n{d['canvas_w']} x {d['canvas_l']}",
            ha="center", va="center", fontsize=FS_C_NOTE + 2.2,
            color="#d8d0c0", fontweight="bold", zorder=2.2, linespacing=1.4)

    # --- the cage --------------------------------------------------------
    for r in ((X(FR_X0), Y(FR_Y0), X(IN_X0), Y(FR_Y1)),
              (X(IN_X1), Y(FR_Y0), X(FR_X1), Y(FR_Y1)),
              (X(IN_X0), Y(FR_Y0), X(IN_X1), Y(FR_Y0 + P)),
              (X(IN_X0), Y(FR_Y1 - P), X(IN_X1), Y(FR_Y1))):
        member(ax, *r, fc=STEEL2, z=5)
    for cx_, cy_ in ((X(FR_X0), Y(FR_Y0)), (X(FR_X1 - P), Y(FR_Y0)),
                     (X(FR_X0), Y(FR_Y1 - P)), (X(FR_X1 - P), Y(FR_Y1 - P))):
        member(ax, cx_, cy_, cx_ + P, cy_ + P, fc="#3b424c", ec=INK, lw=0.8,
               z=5.4)
    for b in seam_posts():
        member(ax, X(b.lo[0]), Y(b.lo[1]), X(b.hi[0]), Y(b.hi[1]), fc=SEAM_FC,
               ec=SEAMC, lw=0.9, z=5.6, hatch="\\\\\\\\")
    for ry in ROW_Y:
        member(ax, X(IN_X0), Y(ry - P), X(IN_X1), Y(ry + P), fc=STEEL, z=5)
        ax.add_line(Line2D([X(IN_X0), X(IN_X1)], [Y(ry), Y(ry)],
                           color="#4d5561", lw=0.45, zorder=5.6))

    # --- the six mounts ---------------------------------------------------
    for aid, (xa, ya) in ARMS.items():
        for px in post_xs(xa):
            for py in (ya - P, ya):
                member(ax, X(px - P / 2), Y(py), X(px + P / 2), Y(py + P),
                       fc=POSTC, ec=INK, lw=0.7, z=6.5)
        cxp = plate_cx(xa)
        member(ax, X(cxp - PLATE[0] / 2), Y(ya - PLATE[1] / 2),
               X(cxp + PLATE[0] / 2), Y(ya + PLATE[1] / 2), fc=PLATEC, ec=INK,
               lw=0.9, z=6.8)
        # the J1 axis: crosshair + the modelled base circle
        ax.add_patch(Circle((X(xa), Y(ya)), d["base_circle_d"] / 2, fc="none",
                            ec=ORIGC, lw=0.8, ls=(0, (4, 2)), zorder=7.6))
        cmark(ax, X(xa), Y(ya), 190, c=ORIGC, lw=0.7, z=7.7)
        ax.add_patch(Circle((X(xa), Y(ya)), 20, fc=INK, ec="none", zorder=8))
        ax.text(X(xa) + (-215 if X(xa) < 0 else 215), Y(ya) + 132, f"{aid}",
                ha="center", va="bottom", fontsize=FS_C_NOTE + 2.6,
                fontweight="bold", color=INK, zorder=8.5)

    centre_lines(ax, XL, YL)
    ax.text(-860, -290, f"(0, 0)  TABLE CENTRE\ncanvas x {CENTRE_X} · "
                        f"y {CENTRE_Y}\nthe seam line crossing the long "
                        f"centre line",
            ha="left", va="top", fontsize=FS_C_NOTE - 0.4, color=ORIGC,
            fontweight="bold", zorder=10.6, linespacing=1.4, bbox=TBOX)
    ax.annotate("", xy=(690, -170), xytext=(120, -170),
                arrowprops=dict(arrowstyle="-|>", color=ORIGC, lw=1.0))
    ax.text(730, -170, "+X", ha="left", va="center", fontsize=FS_C_DIM,
            color=ORIGC, fontweight="bold")
    ax.annotate("", xy=(-170, 690), xytext=(-170, 120),
                arrowprops=dict(arrowstyle="-|>", color=ORIGC, lw=1.0))
    ax.text(-170, 730, "+Y", ha="center", va="bottom", fontsize=FS_C_DIM,
            color=ORIGC, fontweight="bold")

    # --- x ordinate: every face, signed off the centre line --------------
    fl, fr = d["post_f_l"], d["post_f_r"]
    xticks = []
    for i, v in enumerate(fl + fr):
        xticks.append((v, sg(v), i % 2))
    for v in (-d["axis_x"], d["axis_x"]):
        xticks.append((v, f"{sg(v)} AXIS", 1))
    for v in (-d["rail_out"], d["rail_out"]):
        xticks.append((v, f"{sg(v)} frame", 0))
    for v in (-d["canvas_edge"], d["canvas_edge"]):
        xticks.append((v, f"{sg(v)} canvas", 0))
    for v in (-d["table_edge"], d["table_edge"]):
        xticks.append((v, f"{sg(v)} rail/table", 1))
    ord_x(s, ax, -2170.0, xticks, stem=150.0)
    ax.text(-1690, -2330, "ORDINATE\nevery X from the\ncentre line, signed",
            ha="left", va="top", fontsize=FS_C_NOTE - 0.2, color=DIMC,
            fontweight="bold", linespacing=1.4)

    # --- y ordinate -------------------------------------------------------
    yticks = [(0.0, "0.00   SEAM · CENTRE", 0)]
    for v in (-d["seam_bar_face"], d["seam_bar_face"]):
        yticks.append((v, f"{sg(v)}  seam bar", 0))
    for v in (-d["runway_row_face_in"], d["runway_row_face_in"],
              -d["runway_row_face_out"], d["runway_row_face_out"]):
        yticks.append((v, sg(v), 0))
    for v in (-d["axis_y"], d["axis_y"]):
        yticks.append((v, f"{sg(v)}  ARM ROW", 1))
    for v in (-d["canvas_end"], d["canvas_end"]):
        yticks.append((v, f"{sg(v)}  canvas", 0))
    for v in (-d["table_end"], d["table_end"]):
        yticks.append((v, f"{sg(v)}  rail · table", 0))
    for v in (-d["rail_out_end"], d["rail_out_end"]):
        yticks.append((v, f"{sg(v)}  frame", 1))
    for v in (-d["table_end_tape"], d["table_end_tape"]):
        yticks.append((v, f"{sg(v)}  TABLE (tape)", 0))
    ord_y(s, ax, 1160.0, yticks, stem=140.0)

    # --- the overall chains ----------------------------------------------
    s.dim_h(ax, -d["canvas_edge"], d["canvas_edge"], 2140,
            f"{d['canvas_w']}   CANVAS", ext_y=(d["canvas_end"],
                                                d["canvas_end"]),
            over=20, fs=FS_C_DIM, txt_off=22)
    s.dim_h(ax, -d["rail_out"], d["rail_out"], 2320,
            f"{d['frame_w']}   FRAME OUTSIDE", ext_y=(d["rail_out_end"],
                                                      d["rail_out_end"]),
            over=20, fs=FS_C_DIM, txt_off=22)
    s.dim_h(ax, -d["runway_x"], d["runway_x"], 1400,
            f"{d['runway_len']}   RUNWAY / END RAIL, x extent", over=0,
            fs=FS_C_DIM, txt_off=22)
    s.dim_h(ax, -d["axis_x"], d["axis_x"], -820,
            f"{d['axis_pitch_x']}   J1 AXIS PITCH", over=0, fs=FS_C_DIM,
            txt_off=-24)
    s.dim_v(ax, -d["canvas_end"], d["canvas_end"], -1250,
            f"{d['canvas_l']}   CANVAS", over=0, fs=FS_C_DIM, txt_off=-58)
    s.dim_v(ax, -d["table_end"], d["table_end"], -1430,
            f"{d['table_l']}   TABLE, model", over=0, fs=FS_C_DIM,
            txt_off=-58)
    s.dim_v(ax, -d["table_end_tape"], d["table_end_tape"], -1610,
            f"{d['table_len_tape']}   TABLE, TAPE 416.6 cm", over=0,
            fs=FS_C_DIM, txt_off=-58)
    s.dim_v(ax, 0, d["axis_y"], 640, f"{d['axis_pitch_y']}   ROW PITCH",
            over=0, fs=FS_C_DIM, txt_off=30)

    # --- the two members a tape has to find first ------------------------
    s.leader(ax, (X(FR_X0) + P / 2, 0), (-560, -620),
             f"SEAM BAR  item I — FIND THIS AND YOU HAVE FOUND Y = 0\n"
             f"{d['profile']} (X) x {d['seam_bar_dy']} (Y), centred on the "
             f"seam, one per side,\nstanding on the corner-leg line: "
             f"X {sg(d['table_edge'], 1)} .. {sg(d['rail_out'], 1)}, "
             f"Y {sg(-d['seam_bar_face'], 1)} .. "
             f"{sg(d['seam_bar_face'], 1)}.",
             ha="left", c=SEAMC, rad=0.12, fs=FS_C_NOTE - 0.4)
    s.leader(ax, (X(FR_X0) + P / 2, Y(FR_Y0) + P / 2), (-560, -1520),
             f"CORNER LEG  item E — {d['profile']} square, 4 off\n"
             f"X {sg(d['table_edge'], 1)} .. {sg(d['rail_out'], 1)},  "
             f"Y {sg(d['rail_in_end'], 1)} .. {sg(d['rail_out_end'], 1)} "
             f"(both signs).",
             ha="left", c=NOTEC, rad=-0.12, fs=FS_C_NOTE - 0.4)

    # ---------------- panel B: the zoom Pete asked for -------------------
    DEN_B = 4.6
    XB, YB = (-600.0, 30.0), (-320.0, 240.0)
    axb = s.panel(7.32, 5.85, XB, YB, DEN_B,
                  "B   DETAIL — ONE HANGING-STRUT PAIR",
                  f"arm 31  ·  scale 1 : {DEN_B:.1f}")
    member(axb, XB[0], -P, XB[1], P, fc=STEEL, z=4)
    axb.add_line(Line2D([XB[0], XB[1]], [0, 0], color="#4d5561", lw=0.5,
                        zorder=4.4))
    for px in d["post_c_l"]:
        for py in (-P, 0):
            member(axb, px - P / 2, py, px + P / 2, py + P, fc=POSTC, ec=INK,
                   lw=1.0, z=6)
    member(axb, d["plate_cx_l"] - d["plate_w"] / 2, -d["plate_d"] / 2,
           d["plate_cx_l"] + d["plate_w"] / 2, d["plate_d"] / 2, fc=PLATEC,
           ec=INK, lw=1.1, z=6.6)
    axl = -d["axis_x"]
    axb.add_patch(Circle((axl, 0), d["base_circle_d"] / 2, fc="none", ec=ORIGC,
                         lw=0.9, ls=(0, (4, 2)), zorder=7))
    cmark(axb, axl, 0, 130, c=ORIGC, lw=0.8, z=7.4)
    axb.add_patch(Circle((axl, 0), 8, fc=INK, ec="none", zorder=8))
    axb.add_line(Line2D([0, 0], [YB[0], YB[1]], color=ORIGC, lw=1.1,
                        ls=(0, (14, 4, 2.0, 4)), zorder=7.8))
    axb.text(XB[0] + 10, YB[1] - 10, "X = 0  is the CENTRE LINE", ha="left",
             va="top", fontsize=FS_C_NOTE - 0.4, color=ORIGC,
             fontweight="bold")

    s.dim_h(axb, d["post_f_l"][0], d["post_f_l"][3], 172,
            f"{d['pair_outer_w']}   OUTER WIDTH OF THE PAIR",
            ext_y=(P, P), over=8, fs=FS_C_DIM, txt_off=14)
    s.dim_h(axb, d["post_f_l"][1], d["post_f_l"][2], 110,
            f"{d['pair_gap']}   CLEAR BETWEEN THE TWO STRUTS", ext_y=(P, P),
            over=6, fs=FS_C_DIM, txt_off=12)
    s.dim_h(axb, d["post_f_l"][0], axl, -150,
            f"{d['axis_face_short']}\naxis -> outside face", ext_y=(-P, 0),
            over=6, fs=FS_C_DIM, txt_off=-16)
    s.dim_h(axb, axl, d["post_f_l"][3], -150,
            f"{d['axis_face_long']}\naxis -> outside face", ext_y=(0, -P),
            over=6, fs=FS_C_DIM, txt_off=-16)
    for xv, cc in ((axl, ORIGC), (d["plate_cx_l"], "#7a6a52")):
        axb.add_line(Line2D([xv, xv], [-d["plate_d"] / 2 - 10, -232],
                            color=cc, lw=0.5, ls=(0, (3, 2)), zorder=8.4))
    s.dim_h(axb, axl, d["plate_cx_l"], -224,
            f"{abs(d['plate_off'])}  plate/axis offset", over=6, txt_off=-16,
            outside=True, out_len=96, fs=FS_C_DIM)
    axb.annotate("", xy=(XB[1] - 6, 236), xytext=(d["post_f_l"][3], 236),
                 arrowprops=s.arrow)
    axb.add_line(Line2D([d["post_f_l"][3]] * 2, [P, 236], color=DIMC, lw=0.4,
                        ls=(0, (3, 2)), zorder=8.5))
    axb.text(XB[1] - 10, 224, f"{d['inner_pair_gap']}  clear to the NEXT "
                              f"pair's face at X = {sg(d['post_f_r'][0])}",
             ha="right", va="top", fontsize=FS_C_DIM, color=DIMC,
             linespacing=1.3, bbox=TBOX, zorder=9)
    ord_x(s, axb, -252.0,
          [(v, sg(v), 0) for v in d["post_f_l"]]
          + [(axl, f"{sg(axl)} axis", 0),
             (d["plate_cx_l"], f"{sg(d['plate_cx_l'])} plate", 0)],
          stem=14.0, gap=8.0, fs=FS_C_ORD)

    # ---------------- panel C: tape vs model -----------------------------
    axc = s.textbox(7.32, 1.58, 5.39, 3.95)
    axc.add_patch(Rectangle((0, 0), 1, 1, fc="#fdf6f2", ec=WARN, lw=0.9))
    axc.text(0.020, 0.962, f"C   MEASURED BY HAND {HAND_TAPE_DATE}  vs  THE "
                           f"MODEL", ha="left", va="top", fontsize=FS_C_HEAD,
             fontweight="bold", color=INK)
    C_TAPE, C_MODEL, C_DELTA = 0.690, 0.840, 0.980
    for lbl, cx_ in (("what a tape read", 0.020), ("tape", C_TAPE),
                     ("model", C_MODEL), ("tape - model", C_DELTA)):
        axc.text(cx_, 0.905, lbl, ha="left" if cx_ < 0.1 else "right",
                 va="top", fontsize=6.0, color="#6a7280", fontweight="bold")
    axc.add_line(Line2D([0.020, 0.980], [0.884, 0.884], color="#c3cad3",
                        lw=0.6))
    yy = 0.862
    for what, tp, md, dl in hand_vs_model():
        bad = abs(dl) > 5.0
        col = WARN if bad else INK
        axc.text(0.020, yy, what, ha="left", va="top", fontsize=7.0,
                 color=col, fontweight="bold" if bad else "normal")
        axc.text(C_TAPE, yy, f"{tp:.0f}", ha="right", va="top", fontsize=7.4,
                 color=col, fontweight="bold")
        axc.text(C_MODEL, yy, f"{md:g}", ha="right", va="top", fontsize=7.4,
                 color=col)
        axc.text(C_DELTA, yy, f"{dl:+.2f}", ha="right", va="top", fontsize=7.4,
                 color=WARN if bad else GRN,
                 fontweight="bold" if bad else "normal")
        yy -= 0.064
    axc.add_line(Line2D([0.020, 0.980], [yy + 0.018, yy + 0.018],
                        color="#c3cad3", lw=0.6))
    _flow(axc, [(WARN, True, HAND_FLAG)], 0.020, yy - 0.014, width=92,
          fs=6.6, dy=0.0300)

    # ---------------- panel D: the face schedule -------------------------
    axd = s.textbox(12.88, 1.58, 3.22, 3.95)
    axd.add_patch(Rectangle((0, 0), 1, 1, fc="#f7f8fa", ec="#c3cad3", lw=0.7))
    axd.text(0.040, 0.962, "D   STRUT FACE SCHEDULE", ha="left", va="top",
             fontsize=FS_C_HEAD, fontweight="bold", color=INK)
    axd.text(0.040, 0.912, "every face of all four struts in a row, as a "
                           "SIGNED X off the centre line", ha="left",
             va="top", fontsize=6.0, color=NOTEC)
    axd.add_line(Line2D([0.040, 0.960], [0.876, 0.876], color="#c3cad3",
                        lw=0.6))
    rows = []
    for tag, faces, ctrs, pf, pc, axv in (
            ("LEFT column  13 / 31 / 2", d["post_f_l"], d["post_c_l"],
             d["plate_f_l"], d["plate_cx_l"], -d["axis_x"]),
            ("RIGHT column  17 / 71 / 97", d["post_f_r"], d["post_c_r"],
             d["plate_f_r"], d["plate_cx_r"], d["axis_x"])):
        rows.append((True, tag, ""))
        rows.append((False, "J1 axis", sg(axv)))
        rows.append((False, "strut 1  faces",
                     f"{sg(faces[0])}   {sg(faces[1])}"))
        rows.append((False, "strut 1  centre", sg(ctrs[0])))
        rows.append((False, "strut 2  faces",
                     f"{sg(faces[2])}   {sg(faces[3])}"))
        rows.append((False, "strut 2  centre", sg(ctrs[1])))
        rows.append((False, "plate  edges", f"{sg(pf[0])}   {sg(pf[1])}"))
        rows.append((False, "plate  centre", sg(pc)))
    yy = 0.858
    for head, a, b in rows:
        axd.text(0.040, yy, a, ha="left", va="top",
                 fontsize=6.8 if head else 6.4,
                 color=INK if head else NOTEC,
                 fontweight="bold" if head else "normal")
        if b:
            axd.text(0.960, yy, b, ha="right", va="top", fontsize=6.4,
                     color=DIMC, fontweight="bold")
        yy -= 0.044 if head else 0.038
    axd.add_line(Line2D([0.040, 0.960], [yy + 0.012, yy + 0.012],
                        color="#c3cad3", lw=0.6))
    _flow(axd, [
        (INK, True, "THE TWO COLUMNS ARE NOT MIRRORED."),
        (INK, False, f"Every plate on the rig is offset {abs(d['plate_off'])} "
                     f"mm in +X from its own J1 axis (uniform clocking, "
                     f"BUILD_SHEET section 3), so the SHORT "
                     f"{d['axis_face_short']} face is on the -X side of BOTH "
                     f"columns: toward the centre on the right column and "
                     f"away from it on the left. Read the schedule, not the "
                     f"symmetry."),
    ], 0.040, yy - 0.008, width=54, fs=6.3, dy=0.0268)

    # ---------------- panel E: the datum, stated -------------------------
    axe = s.textbox(12.88, 5.85, 3.22, 4.79)
    axe.add_patch(Rectangle((0, 0), 1, 1, fc="#eef4f7", ec=ORIGC, lw=0.9))
    axe.text(0.040, 0.975, "E   THE DATUM, AND THE TABLE", ha="left",
             va="top", fontsize=FS_C_HEAD, fontweight="bold", color=INK)
    lines = [
        (ORIGC, True, "(0, 0) IS THE TABLE CENTRE: the SEAM LINE crossing "
                      "the LONG CENTRE LINE."),
        (INK, False, f"In sheet-1/2 canvas coordinates that is x = "
                     f"{CENTRE_X}, y = {CENTRE_Y}. Every number on this sheet "
                     f"is that point subtracted, and every number carries its "
                     f"sign: +X toward the x = {CW:.1f} long edge, +Y toward "
                     f"the y = {CL:.2f} end."),
        (ORIGC, True, "THE SEAM LINE IS FOUR LINES AT ONCE."),
        (INK, False, f"system_model makes SEAM_Y ({CENTRE_Y}) the canvas "
                     f"mid-length, the MIDDLE ARM ROW, the half-cage butt "
                     f"plane, and the line the seam bars straddle. Find the "
                     f"seam bars and you have found Y = 0 without a tape."),
        (None, False, ""),
    ] + canvas_on_table_lines() + [
        (None, False, ""),
        (ORIGC, True, "HOW TO USE IT AT THE RIG."),
        (INK, False, "Snap one line down the middle of the paper and one "
                     "across at the seam bars. Hook the tape on the crossing "
                     "and read outward; the sign says which way you went. "
                     "Panel A's ordinate ticks are all off those two lines "
                     "and nothing else."),
        (None, False, ""),
        (WARN, True, "NOTHING HERE IS A SURVEY."),
        (WARN, False, "It is the model re-datumed — the same steel sheets 1 "
                      "and 2 draw from the canvas corner. Where a tape and "
                      "the model disagree, panel C says so and the model is "
                      "not edited to match."),
    ]
    _flow(axe, lines, 0.040, 0.930, width=54, fs=6.3, dy=0.0229)

    s.title_block(h, 3, "PLAN — CENTRE DATUM, SIGNED", extra=CENTRE_BANNER,
                  datum="(0,0)  table centre",
                  datum_sub=f"seam y {CENTRE_Y}  x  centre line x {CENTRE_X}")
    _save(fig, out_dir, "plan_centre_datum")


# ---------------------------------------------------------------------------
# SHEET 4 — SIDE REFERENCE HEIGHTS, FROM THE PAPER AND FROM THE FLOOR
# ---------------------------------------------------------------------------
def _elev_frame(ax, axis, z, h, xlo, xhi):
    """Floor / table / paper / cage, projected on (`axis`, z).  axis 0 = x."""
    t = canvas_on_table()
    ground(ax, xlo, xhi, FLOOR_Z, depth=150)
    if axis == 1:
        ax.add_patch(Rectangle((-t["tape_y"], FLOOR_Z), t["tape_l"],
                               TABLE_TOP_Z - FLOOR_Z, fc="none", ec="#8a6f56",
                               lw=0.9, ls=(0, (6, 3)), zorder=3.1))
        member(ax, t["y0"], FLOOR_Z, t["y1"], TABLE_TOP_Z, fc="#8a6f56",
               ec=INK, lw=0.6, z=3.2, alpha=0.75)
        member(ax, -CL / 2, TABLE_TOP_Z, CL / 2, 0.0, fc="#fbf8f2", ec=INK,
               lw=1.0, z=3.6)
        ends = ((FR_Y0 - CENTRE_Y, FR_Y0 + P - CENTRE_Y),
                (FR_Y1 - P - CENTRE_Y, FR_Y1 - CENTRE_Y))
        for a, b in ends:
            member(ax, a, GRID_U, b, GRID_T, fc=STEEL2, ec=INK, lw=0.7, z=5.2)
            member(ax, a, LEG_BOTTOM, b, GRID_U, fc="#3b424c", ec=INK, lw=0.7,
                   z=5.0)
        for ry in ROW_Y:
            member(ax, ry - P - CENTRE_Y, GRID_U, ry + P - CENTRE_Y, GRID_T,
                   fc=STEEL, ec=INK, lw=0.7, z=5.4)
        for b in seam_posts():
            member(ax, b.lo[1] - CENTRE_Y, b.lo[2], b.hi[1] - CENTRE_Y,
                   b.hi[2], fc=SEAM_FC, ec=SEAMC, lw=0.7, z=5.1,
                   hatch="\\\\\\\\", alpha=0.75)
        for ry in ROW_Y:
            r0 = ry - CENTRE_Y
            for py in (r0 - P, r0):
                member(ax, py, z["post_bottom"], py + P, GRID_U, fc=POSTC,
                       ec=INK, lw=0.7, z=6)
            member(ax, r0 - CLAMP[1] / 2, z["plate_top"], r0 + CLAMP[1] / 2,
                   z["clamp_top"], fc="#c3cad3", ec=INK, lw=0.6, z=6.2)
            member(ax, r0 - PLATE[1] / 2, h, r0 + PLATE[1] / 2, z["plate_top"],
                   fc=PLATEC, ec=INK, lw=0.9, z=6.5)
            for px in (r0 - P - GUSSET[1] / 2, r0 + P + GUSSET[1] / 2):
                ax.add_patch(Rectangle((px - GUSSET[1] / 2,
                                        z["gusset_bottom"]), GUSSET[1],
                                       GRID_T - z["gusset_bottom"], fc="none",
                                       ec="#8d949c", lw=0.5, ls=(0, (4, 2)),
                                       zorder=5.6))
        return
    # --- axis 0: the section across, taken AT the seam -------------------
    member(ax, t["x0"], FLOOR_Z, t["x1"], TABLE_TOP_Z, fc="#8a6f56", ec=INK,
           lw=0.6, z=3.2, alpha=0.75)
    member(ax, -CW / 2, TABLE_TOP_Z, CW / 2, 0.0, fc="#fbf8f2", ec=INK, lw=1.0,
           z=3.6)
    for a in (FR_X0 - CENTRE_X, IN_X1 - CENTRE_X):
        member(ax, a, GRID_U, a + P, GRID_T, fc=STEEL2, ec=INK, lw=0.7, z=5.2)
        member(ax, a, LEG_BOTTOM, a + P, GRID_U, fc="#3b424c", ec=INK, lw=0.7,
               z=5.0)
    for b in seam_posts():
        member(ax, b.lo[0] - CENTRE_X, b.lo[2], b.hi[0] - CENTRE_X, b.hi[2],
               fc=SEAM_FC, ec=SEAMC, lw=0.7, z=5.3, hatch="\\\\\\\\",
               alpha=0.8)
    member(ax, IN_X0 - CENTRE_X, GRID_U, IN_X1 - CENTRE_X, GRID_T, fc=STEEL,
           ec=INK, lw=0.8, z=4.6)
    for xa in COL_X:
        for px in post_xs(xa):
            member(ax, px - P / 2 - CENTRE_X, z["post_bottom"],
                   px + P / 2 - CENTRE_X, GRID_U, fc=POSTC, ec=INK, lw=0.85,
                   z=6)
        cxp = plate_cx(xa) - CENTRE_X
        member(ax, cxp - CLAMP[0] / 2, z["plate_top"], cxp + CLAMP[0] / 2,
               z["clamp_top"], fc="#c3cad3", ec=INK, lw=0.8, z=6.2)
        member(ax, cxp - PLATE[0] / 2, h, cxp + PLATE[0] / 2, z["plate_top"],
               fc=PLATEC, ec=INK, lw=1.1, z=6.5)
        member(ax, xa - 75 - CENTRE_X, h - 150, xa + 75 - CENTRE_X, h,
               fc="#eae6df", ec=INK, lw=0.8, z=6.4)
        cmark(ax, xa - CENTRE_X, h, 260, c=ORIGC, lw=0.7, z=7.2)


def sheet_side_heights(h, out_dir):
    """out/drawings/centre/side_reference_heights.pdf / .png — A3 landscape."""
    s = Sheet()
    fig = s.fig
    z = zl(h)
    lv = side_levels(h)

    # ---------------- panel A: along the long side ------------------------
    DEN = 26.0
    ax = s.panel(0.42, 6.20, (-2700.0, 3450.0), (-900.0, 2280.0), DEN,
                 "A   ELEVATION ALONG THE LONG SIDE  (looking +X)",
                 f"the whole rig end to end  ·  scale 1 : {DEN:.0f}")
    _elev_frame(ax, 1, z, h, -2680, 2100)
    # the room ceiling nobody has measured — the same flag sheet 2 carries
    ax.add_line(Line2D([-2680, 2100], [2150, 2150], color=WARN, lw=1.0,
                       ls=(0, (7, 4)), zorder=3))
    for i in range(34):
        xq = -2680 + 4780 * i / 33.0
        ax.add_line(Line2D([xq, xq - 70], [2150, 2222], color=WARN, lw=0.4,
                           alpha=0.55, zorder=2.8))
    ax.text(-290, 2168, "ROOM CEILING — SURVEY REQUIRED.  The cage is "
                        "self-supporting; no room ceiling has ever been "
                        "measured, and none is drawn.",
            ha="center", va="bottom", fontsize=FS_C_NOTE - 0.6, color=WARN,
            fontweight="bold")
    for aid in (13, 31, 2):
        for A, B, r in park_capsules(aid, h):
            capsule(ax, (A[1] - CENTRE_Y, A[2]), (B[1] - CENTRE_Y, B[2]), r,
                    fc=ARMC, ec="#9aa2ad", lw=0.35, zorder=3.4, alpha=0.5)
    ax.add_line(Line2D([-2620, 2100], [h, h], color=GRN, lw=1.0,
                       ls=(0, (7, 3)), zorder=8))
    ax.add_line(Line2D([0, 0], [-900, 2050], color=ORIGC, lw=1.0,
                       ls=(0, (14, 4, 2.0, 4)), zorder=7.8))
    ax.text(34, 2030, "Y = 0   SEAM", ha="left", va="top",
            fontsize=FS_C_NOTE - 0.4, color=ORIGC, fontweight="bold")
    s.dim_v(ax, 0.0, h, -2280, f"{h:.1f}\nPAPER -> PLATE UNDERSIDE",
            ext_x=(0, 0), over=0, txt_off=-60, fs=FS_C_DIM)
    s.dim_v(ax, FLOOR_Z, 0.0, -2540, f"{PAPER_ABOVE_FLOOR}\nFLOOR -> PAPER",
            ext_x=(0, 0), over=0, txt_off=-60, fs=FS_C_DIM)

    # THE LADDER, AND IT IS THE POINT OF THE SHEET.  Every level twice: mm
    # above the paper (what the repo measures) and mm above the floor (what a
    # tape reads).  `_JOG` only moves the LABEL off its own tick — four of
    # these levels are 2 to 35 mm apart and would print on top of each other.
    LX = 2160.0
    ax.add_line(Line2D([LX, LX], [FLOOR_Z - 40, GRID_T + 40], color=DIMC,
                       lw=0.5, zorder=8))
    ax.text(LX + 200, GRID_T + 140, "mm above\nPAPER", ha="right", va="bottom",
            fontsize=5.8, color="#6a7280", fontweight="bold", linespacing=1.3)
    ax.text(LX + 580, GRID_T + 140, "mm above\nFLOOR", ha="right", va="bottom",
            fontsize=5.8, color="#6a7280", fontweight="bold", linespacing=1.3)
    for k, _lbl, short, pa, fo, bold in lv:
        dy = _JOG.get(k, 0.0)
        s.ext(ax, 2050, pa, LX, pa)
        ax.add_line(Line2D([LX - 26, LX + 26], [pa, pa], color=DIMC, lw=0.9,
                           zorder=8.5))
        if dy:
            ax.add_line(Line2D([LX + 26, LX + 76], [pa, pa + dy], color=DIMC,
                               lw=0.5, zorder=8.5))
        ax.text(LX + 200, pa + dy, f"{pa:+.2f}", ha="right", va="center",
                fontsize=7.0 if bold else 6.3, color=DIMC,
                fontweight="bold" if bold else "normal", zorder=9, bbox=TBOX)
        ax.text(LX + 580, pa + dy, f"{fo:.2f}", ha="right", va="center",
                fontsize=7.0 if bold else 6.3, color=GRN,
                fontweight="bold" if bold else "normal", zorder=9, bbox=TBOX)
        ax.text(LX + 640, pa + dy, short, ha="left", va="center",
                fontsize=6.4 if bold else 5.9, color=INK if bold else NOTEC,
                fontweight="bold" if bold else "normal", zorder=9)

    # ---------------- panel B: across, at the seam ------------------------
    DEN_B = 26.0
    axb = s.panel(10.20, 6.20, (-1800.0, 1800.0), (-900.0, 2280.0), DEN_B,
                  "B   ELEVATION ACROSS, AT THE SEAM  (looking +Y)",
                  f"scale 1 : {DEN_B:.0f}")
    _elev_frame(axb, 0, z, h, -1780, 1780)
    for aid in (31, 71):
        for A, B, r in park_capsules(aid, h):
            capsule(axb, (A[0] - CENTRE_X, A[2]), (B[0] - CENTRE_X, B[2]), r,
                    fc=ARMC, ec="#9aa2ad", lw=0.35, zorder=3.4, alpha=0.5)
    axb.add_line(Line2D([-1720, 1720], [h, h], color=GRN, lw=1.0,
                        ls=(0, (7, 3)), zorder=8))
    axb.add_line(Line2D([0, 0], [-900, 2000], color=ORIGC, lw=1.0,
                        ls=(0, (14, 4, 2.0, 4)), zorder=7.8))
    axb.text(34, -860, "X = 0", ha="left", va="bottom",
             fontsize=FS_C_NOTE - 0.4, color=ORIGC, fontweight="bold")
    s.dim_v(axb, TABLE_TOP_Z, h, -1330,
            f"{h - TABLE_TOP_Z:.1f}\nTABLE TOP -> PLATE UNDERSIDE",
            ext_x=(0, 0), over=0, txt_off=-60, fs=FS_C_DIM)
    s.dim_v(axb, z["post_bottom"], GRID_U, 1320,
            f"{z['post_length']:.1f}\nDROP POST  item D",
            ext_x=(post_xs(COL_X[1])[1] - CENTRE_X, IN_X1 - CENTRE_X), over=0,
            txt_off=62, fs=FS_C_DIM)
    s.dim_v(axb, LEG_BOTTOM, GRID_U, 1640, f"{LEG_LEN}\nLEG / SEAM BAR",
            ext_x=(FR_X1 - CENTRE_X, FR_X1 - CENTRE_X), over=0, txt_off=62,
            fs=FS_C_DIM)
    axb.text(0, 2010, f"SEAM BARS, hatched — they stand in the corner legs' "
                      f"own x bands\n{SEAM_FLAG}",
             ha="center", va="top", fontsize=FS_C_NOTE - 0.6, color=SEAMC,
             fontweight="bold", linespacing=1.4, zorder=9, bbox=TBOX)

    # ---------------- panel C: the two datums ----------------------------
    axc = s.textbox(14.02, 1.58, 2.08, 4.42)
    axc.add_patch(Rectangle((0, 0), 1, 1, fc="#eef4f7", ec=ORIGC, lw=0.9))
    axc.text(0.060, 0.972, "C   TWO DATUMS", ha="left", va="top",
             fontsize=FS_C_HEAD - 0.6, fontweight="bold", color=INK)
    _flow(axc, [
        (ORIGC, True, "PAPER TOP = 0"),
        (INK, False, "what sheets 1 and 2 and every module in the repo "
                     "measure from."),
        (None, False, ""),
        (ORIGC, True, f"FLOOR = 0, i.e. + {PAPER_ABOVE_FLOOR}"),
        (INK, False, "what a tape standing on the floor reads. The right-hand "
                     "column of panel D is the same ladder with the floor as "
                     "zero."),
        (None, False, ""),
        (WARN, True, "THE FLOOR COLUMN IS NOT A SURVEY."),
        (WARN, False, f"It is the paper column plus {PAPER_ABOVE_FLOOR}, and "
                      f"that {PAPER_ABOVE_FLOOR} is the drawing's table "
                      f"height, not a measured one. Check it first (panel E) "
                      f"— if it is not {abs(TABLE_TOP_Z - FLOOR_Z):.0f} "
                      f"the whole column shifts."),
    ], 0.060, 0.918, width=36, fs=6.4, dy=0.0278)

    # ---------------- panel D: the ladder, both ways ---------------------
    axd = s.textbox(0.42, 1.58, 7.30, 4.42)
    axd.add_patch(Rectangle((0, 0), 1, 1, fc="#f7f8fa", ec="#c3cad3", lw=0.7))
    axd.text(0.018, 0.968, "D   REFERENCE HEIGHTS — FROM THE PAPER AND FROM "
                           "THE FLOOR", ha="left", va="top",
             fontsize=FS_C_HEAD, fontweight="bold", color=INK)
    axd.text(0.700, 0.912, "above PAPER", ha="right", va="top",
             fontsize=6.2, color="#6a7280", fontweight="bold")
    axd.text(0.930, 0.912, "above FLOOR", ha="right", va="top",
             fontsize=6.2, color="#6a7280", fontweight="bold")
    axd.add_line(Line2D([0.018, 0.982], [0.890, 0.890], color="#c3cad3",
                        lw=0.6))
    yy = 0.866
    for _k, lbl, _short, pa, fo, bold in lv:
        axd.text(0.018, yy, lbl, ha="left", va="top",
                 fontsize=7.4 if bold else 6.8, color=INK if bold else NOTEC,
                 fontweight="bold" if bold else "normal")
        axd.text(0.700, yy, f"{pa:+.2f}", ha="right", va="top",
                 fontsize=7.8 if bold else 7.0, color=DIMC,
                 fontweight="bold" if bold else "normal")
        axd.text(0.930, yy, f"{fo:.2f}", ha="right", va="top",
                 fontsize=7.8 if bold else 7.0, color=GRN,
                 fontweight="bold" if bold else "normal")
        yy -= 0.0570
    axd.add_line(Line2D([0.018, 0.982], [yy + 0.016, yy + 0.016],
                        color="#c3cad3", lw=0.6))
    axd.text(0.018, yy - 0.010, "CUT LENGTHS ON THIS LADDER", ha="left",
             va="top", fontsize=7.2, fontweight="bold", color=INK)
    yy -= 0.060
    for it, what, ln, note in side_cuts(h):
        axd.text(0.018, yy, f"{it}   {what}", ha="left", va="top",
                 fontsize=7.0, color=INK)
        axd.text(0.320, yy, f"{ln:.1f}", ha="right", va="top", fontsize=7.4,
                 color=DIMC, fontweight="bold")
        axd.text(0.352, yy, note, ha="left", va="top", fontsize=6.4,
                 color=NOTEC)
        yy -= 0.0520

    # ---------------- panel E: the tape checks ---------------------------
    axe = s.textbox(7.92, 1.58, 5.90, 4.42)
    axe.add_patch(Rectangle((0, 0), 1, 1, fc="#eef7f1", ec=GRN, lw=0.9))
    axe.text(0.017, 0.965, "E   CHECKS YOU CAN MAKE WITH A TAPE AND NOTHING "
                           "ELSE", ha="left", va="top", fontsize=FS_C_HEAD,
             fontweight="bold", color=INK)
    yy = 0.890
    for what, mm, why in tape_checks(h):
        axe.text(0.017, yy, what, ha="left", va="top", fontsize=8.0,
                 fontweight="bold", color=INK)
        axe.text(0.983, yy, f"{mm:.2f}", ha="right", va="top", fontsize=11.0,
                 fontweight="bold", color=GRN)
        yy -= 0.054
        yy = _flow(axe, [(NOTEC, False, why)], 0.034, yy, width=74, fs=6.5,
                   dy=0.0272)
        yy -= 0.014
    axe.add_line(Line2D([0.017, 0.983], [yy + 0.010, yy + 0.010],
                        color="#c3cad3", lw=0.6))
    _flow(axe, [
        (WARN, True, "TOLERANCE, AND WHAT TO DO WITH A DISAGREEMENT."),
        (INK, False, "Build to +/-10 mm per arm and keep the six mount planes "
                     "mutually coplanar within +/-3 mm (BUILD_SHEET section "
                     "1). If a tape reading and this sheet disagree by more "
                     "than that, RECORD THE AS-BUILT NUMBER — do not "
                     "re-centre "
                     "the other five arms to it, and do not edit the model to "
                     "match a single reading."),
        (WARN, True, "NOTHING ON THIS SHEET IS A SURVEY."),
        (WARN, False, "Every height descends from the original drawing's "
                      "233,7 cm, read as FLOOR to top of a self-supporting "
                      "cage. The room has never been measured."),
    ], 0.017, yy - 0.006, width=76, fs=6.5, dy=0.0262)

    s.title_block(h, 4, "SIDE REFERENCE HEIGHTS", extra=CENTRE_BANNER,
                  datum="paper top 0  ·  floor " f"{PAPER_ABOVE_FLOOR}",
                  datum_sub=f"table top {TABLE_TOP_Z}  ·  top of steel "
                            f"{GRID_T}")
    _save(fig, out_dir, "side_reference_heights")


# ===========================================================================
# 8.  DRIVER
# ===========================================================================
def _save(fig, out_dir, stem):
    os.makedirs(out_dir, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", dict(dpi=220))):
        fig.savefig(os.path.join(out_dir, f"{stem}.{ext}"),
                    facecolor="white", **kw)
    plt.close(fig)
    print(f"  wrote {os.path.join(out_dir, stem)}.pdf / .png")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--h", type=float, default=H_DESIGN,
                    help="mount plane above the paper, m or mm "
                         f"(default {H_DESIGN:.0f} mm)")
    ap.add_argument("--out", default=None,
                    help="output directory (default out/drawings, or "
                         "out/drawings/<clocking> for a variant)")
    ap.add_argument("--clocking", default="uniform", choices=sorted(CLOCKINGS),
                    help="'uniform' is the SHIPPED clocking (BUILD_SHEET "
                         "section 3); 'mirrored' turns the LEFT column to "
                         "face the right.  A variant sheet — it writes to its "
                         "own directory and never overwrites the uniform set")
    a = ap.parse_args(argv)
    global CLOCKING
    CLOCKING = a.clocking
    if a.out is None:
        a.out = os.path.join(_repo(), "out", "drawings")
        if CLOCKING != "uniform":
            a.out = os.path.join(a.out, CLOCKING)
    h = a.h * MM if a.h < 10.0 else a.h        # accept 0.97 or 970
    os.makedirs(a.out, exist_ok=True)
    print(f"ARIS 80/20 fabrication drawings — h = {h:.1f} mm, "
          f"drop post {post_length(h):.1f} mm, clocking {CLOCKING}")
    if CLOCKING != "uniform":
        print(f"  cluster gap {cluster_gap():.2f} mm (uniform 216.20), "
              f"gusset pair clearance {gusset_pair_clear():.2f} mm "
              f"(uniform 89.20)")
    sheet_topdown(h, a.out)
    sheet_side(h, a.out)
    p = write_cut_list(os.path.join(a.out, "8020_cut_list.md"), h)
    print(f"  wrote {p}")
    # sheets 3 and 4 — the SAME model, re-datumed onto the table centre, for
    # a tape at the built rig.  Their own directory because they are a
    # different DATUM, not a different rig: nobody should ever be holding one
    # of each without noticing which is which.
    cdir = os.path.join(a.out, "centre")
    ct = canvas_on_table()
    print(f"  centre datum (0,0) = canvas ({CENTRE_X}, {CENTRE_Y})  ·  "
          f"canvas {'IS' if ct['centred'] else 'IS NOT'} centred on the "
          f"model table (eccentricity {ct['ecc_x']:+.2f}, {ct['ecc_y']:+.2f})")
    sheet_centre_plan(h, cdir)
    sheet_side_heights(h, cdir)
    return 0


if __name__ == "__main__":       # pragma: no cover
    sys.exit(main())
