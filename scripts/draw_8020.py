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

    h = 970  ->  688.6 mm     the design height this sheet is issued at
    h = 940  ->  718.6 mm     layout.LAYOUT_PROPOSED, what ships today
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
                                #      recommends (docs/DECISIONS.md,
                                #      "OPEN — FOR PETE: NO HOLES UNDER THE
                                #      ARMS", 2026-09-08)
H_TABLE = (970.0, 940.0, 850.0)   # the three heights the sheet tabulates
H_SHIPPED = SM.H_MOUNT            # 940.0, layout.LAYOUT_PROPOSED["h"]
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

# --- the arm-31 mount, lifted from the drawing -----------------------------
POST_PITCH_X = SM.POST_PITCH_X                    # 317.6
POST_SLOT = SM.POST_SLOT                          # 241.4 between the pairs
PLATE_CLR = SM.PLATE_SIDE_CLEAR                   # 7.79 each side
PLATE_OFF = abs(SM.PLATE_OFF)                     # 25.15, DIRECTION INFERRED
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
TODAY = _dt.date.today().isoformat()


def post_length(h):
    """Drop-post cut length (mm) for mount plane `h` (mm) — from the model."""
    return SM.z_ladder(float(h))["post_length"]


def zl(h):
    """The z ladder at `h`, straight out of `system_model.z_ladder`."""
    return SM.z_ladder(float(h))


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
     f"legs at {LEG_LEN} mm are drawn; the unsupported runway spans are "
     f"{ROW_SP:.0f} mm.",
     "A structural check of the unsupported spans, then a leg count and "
     "position — and a check that no mid-span leg lands in a certified "
     "flight path.",
     "BLOCKS the leg cut (item E qty) and any mid-span member."),
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
        what = {970.0: "**the design height of this sheet** — the "
                       "certified-workspace recommendation",
                940.0: "`layout.LAYOUT_PROPOSED['h']`, what the software "
                       "plans against today",
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
        disp = (f"{ln:.2f}" if "T-slot" in prof
                else f"{GUSSET[0]} x {GUSSET[2]} x {GUSSET[1]}")
        a(f"| **{it}** {name} | {qty} | {prof} | **{disp}** | {what} |")
    a("")
    a(f"**Total 3-in T-slot extrusion at h = {h:.0f}: {extrusion_m(h):.2f} "
      f"m.**  (Items A-E; the gussets are a bought bracket.)")
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
    a(f"Every arm is clocked identically — `R_world_base = Ry(180)`, the "
      f"arm's front toward the canvas x = 0 edge, so **all six connector "
      f"panels face the x = {CW:.1f} edge**.  The plate centre sits "
      f"{PLATE_OFF} mm from the J1 axis toward that same edge — DIRECTION "
      f"INFERRED, open item 1.")
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
    def title_block(self, h, sheet_no, title, extra=""):
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
        ax.text(0.010, 0.14, f"sheet {sheet_no} of 2  ·  A3  ·  all "
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
        cell(3, "DATUM", "z = 0  paper top", fs=8.2)
        ax.text(cols[3] + 0.010, 0.13,
                f"floor {FLOOR_Z}  ·  runway underside {GRID_U}",
                ha="left", va="center", fontsize=5.6, color=NOTEC)
        cell(4, "DATE", TODAY, bold=False, fs=8.2)
        ax.text(cols[4] + 0.010, 0.13, "scripts/draw_8020.py  ·  read-only "
                                       "against the package",
                ha="left", va="center", fontsize=5.6, color=NOTEC)
        # revision banner
        rb = self.textbox(x0, y0 + hh + 0.04, w, 0.20)
        rb.add_patch(Rectangle((0, 0), 1, 1, fc="#fdf1ea", ec=WARN, lw=0.9))
        rb.text(0.008, 0.5, "  " + REV_NOTE + "  —  that grid datum read the "
                            "drawing's 233,7 cm (FLOOR to top of cage) as if "
                            "it were measured from the paper.  DO NOT CUT "
                            "FROM THE OLD SHEET." + extra,
                ha="left", va="center", fontsize=6.2, color=WARN,
                fontweight="bold")


def member(ax, x0, y0, x1, y1, fc=STEEL, ec=INK, lw=0.7, z=4, alpha=1.0,
           hatch=None):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=fc, ec=ec, lw=lw,
                           zorder=z, alpha=alpha, hatch=hatch))


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

    # runways + clusters
    for ry in ROW_Y:
        member(ax, IN_X0, ry - P, IN_X1, ry + P, fc=STEEL, z=5)
        ax.add_line(Line2D([IN_X0, IN_X1], [ry, ry], color="#4d5561",
                           lw=0.45, zorder=5.6))
    for aid, (xa, ya) in ARMS.items():
        for px in SM.post_x(xa):
            for py in (ya - P, ya):
                member(ax, px - P / 2, py, px + P / 2, py + P, fc=POSTC,
                       ec=INK, lw=0.6, z=6.5)
        # gussets, rotated onto the runway's outboard y faces
        for px in SM.post_x(xa):
            for y0 in (ya - P - GUSSET[1], ya + P):
                member(ax, px - GUSSET[0] / 2, y0, px + GUSSET[0] / 2,
                       y0 + GUSSET[1], fc="#c3cad3", ec=INK, lw=0.4, z=6.2,
                       alpha=0.9)
        cx = SM.plate_centre_x(xa)
        member(ax, cx - PLATE[0] / 2, ya - PLATE[1] / 2, cx + PLATE[0] / 2,
               ya + PLATE[1] / 2, fc=PLATEC, ec=INK, lw=0.9, z=6.8)
        cmark(ax, xa, ya, 150, z=7.5)
        ax.add_patch(Circle((xa, ya), 17, fc=INK, ec="none", zorder=8))
        ax.text(xa - 195, ya + 120, f"{aid}", ha="center", va="bottom",
                fontsize=FS_LBL + 2.4, fontweight="bold", color=INK,
                zorder=8.5)
        # connector side: base +x is world -x, so the connector faces +x
        ax.add_patch(FancyArrow(xa + 60, ya, 150, 0, width=12,
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
    s.leader(ax, (SM.post_x(COL_X[1])[1], ROW_Y[1] - P / 2),
             (2410, ROW_Y[1] - 700),
             f"DROP CLUSTER  item D\n4 posts, 2 x 2, pitch "
             f"{POST_PITCH_X} x {P}.\nCut length {post_length(h):.1f} at "
             f"h = {h:.0f}.\nSee detail B.", ha="right", c=ORIGC, rad=0.15)
    s.leader(ax, (FR_X0 + P / 2, FR_Y0 + P / 2), (-1140, -500),
             "CORNER LEG  item E\nto the floor — the cage is\n"
             "self-supporting.  MID-SPAN LEGS\nARE PROBABLY REQUIRED "
             "(open item 3).", ha="left", c=WARN, rad=0.12)
    s.leader(ax, (ARMS[17][0] + 210, ARMS[17][1]), (2410, 640),
             "CONNECTOR SIDE\nAll six arms are clocked identically,\n"
             f"yaw = 0, R_world_base = Ry(180): the front\nfaces x = 0 and "
             f"every connector panel\nfaces the x = {CW:.1f} edge.",
             ha="right", c=ORIGC, rad=-0.12)

    # ---------------- panel B: the cluster detail -----------------------
    DEN_B = 5.0
    axb = s.panel(8.02, 5.95, (-350, 350), (-300, 300), DEN_B,
                  "B   DETAIL — ONE DROP CLUSTER IN PLAN",
                  f"at the J1 axis · scale 1 : {DEN_B:.0f}")
    cxp = SM.plate_centre_x(0.0)          # +25.15 with the inferred direction
    member(axb, -350, -P, 350, P, fc=STEEL, z=4)
    axb.add_line(Line2D([-350, 350], [0, 0], color="#4d5561", lw=0.5,
                        zorder=4.4))
    for px in SM.post_x(0.0):
        for py in (-P, 0):
            member(axb, px - P / 2, py, px + P / 2, py + P, fc=POSTC, ec=INK,
                   lw=0.9, z=6)
    for px in SM.post_x(0.0):
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

    pxl, pxr = SM.post_x(0.0)
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
        (INK, True, "ORIENTATION — load-bearing, and identical for all six."),
        (INK, False, "R_world_base = Ry(180), yaw = 0. The flipped base's +x "
                     "(the arm's front) points toward x = 0, so every "
                     f"connector panel faces the x = {CW:.1f} long edge. Do "
                     "not clock any arm differently: the certified coverage "
                     "and every program assume this exact uniform "
                     "orientation."),
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

    s.title_block(h, 1, "ARM SPACING — PLAN")
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
            n.append(("w", f"  {num}  {title}"))
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
        pxl, pxr = SM.post_x(xa)
        cxp = SM.plate_centre_x(xa)
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
            ext_x=(SM.post_x(COL_X[0])[0] - P / 2, IN_X0), over=0,
            txt_off=-40)
    s.dim_v(ax, 0.0, h, -640, f"{h:.1f}   MOUNT PLANE h",
            ext_x=(0, SM.post_x(COL_X[0])[0]), over=0, txt_off=-40)
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
        s.ext(ax, SM.post_x(COL_X[1])[1] + P / 2, zz, LX, zz)
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
    s.leader(ax, (SM.post_x(COL_X[1])[1], z["gusset_bottom"] + 60),
             (2530, 1300),
             f"GUSSET  item F  —  BEYOND THE SECTION\n{GUSSET[0]} x "
             f"{GUSSET[2]} x {GUSSET[1]}, four per arm, rotated onto the\n"
             f"runway's OUTBOARD y faces, top flush with the steel.",
             ha="right", c=NOTEC, rad=-0.12)

    # ---------------- panel B: section in y-z ---------------------------
    DEN_B = 30.0
    axb = s.panel(9.95, 6.55, (-620, 4260), (-960, 2050), DEN_B,
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
    axb.text((FR_Y0 + FR_Y1) / 2, -880,
             f"UNSUPPORTED RUNWAY SPANS OF {ROW_SP:.0f} mm between the "
             f"corner legs — mid-span legs are almost certainly required "
             f"(open item 3)", ha="center", va="bottom", fontsize=FS_NOTE,
             color=WARN, fontweight="bold")
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
    pxl, pxr = SM.post_x(0.0)
    cxp = SM.plate_centre_x(0.0)
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

    s.title_block(h, 2, "CAGE — ELEVATIONS AND DROP-POST CUT")
    _save(fig, out_dir, "cage_side_view")


# ===========================================================================
# 7.  DRIVER
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
    ap.add_argument("--out", default=os.path.join(_repo(), "out", "drawings"),
                    help="output directory (default out/drawings)")
    a = ap.parse_args(argv)
    h = a.h * MM if a.h < 10.0 else a.h        # accept 0.97 or 970
    os.makedirs(a.out, exist_ok=True)
    print(f"ARIS 80/20 fabrication drawings — h = {h:.1f} mm, "
          f"drop post {post_length(h):.1f} mm")
    sheet_topdown(h, a.out)
    sheet_side(h, a.out)
    p = write_cut_list(os.path.join(a.out, "8020_cut_list.md"), h)
    print(f"  wrote {p}")
    return 0


if __name__ == "__main__":       # pragma: no cover
    sys.exit(main())
