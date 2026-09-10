"""THE SYSTEM MODEL — the whole installation as dimensioned, sourced bodies.

This module is the single place where the *physical* installation is written
down: the 80/20 cage, the table and paper it stands over, the six arms' mount
plates, and the provenance of every number.  It is the truth that
`assets/system_model/` is generated from, and `tests/test_system_model.py`
holds the generated asset to it.

WHAT THIS MODULE IS NOT
-----------------------
It is NOT the obstacle model.  `aris_sixarm/mounts.py` carries the SCHEMATIC
keep-out envelope every certified number in this repo was earned against, and
nothing here changes it.  Where the two disagree — and they do, see
`reconciliation()` — the disagreement is reported, not resolved.  Resolving it
is a re-certification, which is a separate job with its own gate.

THE DATUM CORRECTION
--------------------
`mounts.MOUNTS.ceiling_z = 2.34` is a provenance bug, and this module models
the truth instead.

The original drawing ("Drawing installation, 3 arms, 1 up, 1 side, 1 down,
sizes cm", Stefan Strauss, Vectorworks + companion binary DXF) carries one
overall height dimension, 233,7 cm.  That dimension is FLOOR to TOP OF
CONSTRUCTION: the drawing describes a self-supporting cage standing on the
floor, and defines no room ceiling at all.  The paper it draws on sits
636.68 mm above that floor (`rig_final.PAPER_ORIGIN_W_CM[2]`), so above the
PAPER the same steel reaches only

    2336.5 - 636.68 = 1699.82 mm      (top of construction)
    2260.3 - 636.68 = 1623.62 mm      (beam underside — what a post hangs from)

`ceiling_z = 2.34` is that 233,7 re-datumed to the paper, i.e. the floor-
referenced number used as a paper-referenced one.  It puts the beam underside
at 2340.0 above the paper against the drawing's 1623.62, and the top of the
construction at 2416.2 against 1699.82 — 716.4 mm too high, both of them.
Measured from the FLOOR, where the drawing measures, the code datum stands the
cage 3052.88 mm tall against the 2336.5 that was drawn.

Being too tall is CONSERVATIVE for the collision model (the schematic booms
are modelled ~700 mm longer than the steel will be, and a longer obstacle
never certifies a pose a shorter one refuses), which is why nothing has
broken.  It is not conservative for a FABRICATOR: at h = 940 the buggy datum
asks for a 1435.0 mm drop post where the corrected one asks for 718.6 mm,
against the original's own 736.9 mm.  Both numbers are in `reconciliation()`.

NOTHING HERE IS A SURVEY.  The real room's ceiling has never been measured.

PROVENANCE CLASSES
------------------
Every body carries a `source` naming exactly one of:

    DRAWING   lifted from the original drawing, through the package's own
              gate-validated extraction (`rig_final.FRAME_BOXES_W_CM`,
              `rig_final.ARM_MOUNTS_W`).  The drawing entity is named.
    CODE      a constant this repo plans against (`layout`, `frames`,
              `mounts`, `rig_final`).  Changing it changes the programme.
    AUDIT     measured, by a script in this repo, off CAD or off the
              manufacturer's meshes.  The measurement is named.
    ASSUMED   chosen here.  Every one of them is also in `OPEN_QUESTIONS`.

`Body.provenance` is one of those four strings; `manifest()` reports the mix.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import layout, mounts, rig_final, selfcoll

MM = 1000.0                  # m -> mm
IN = 25.4                    # mm per inch

# ---------------------------------------------------------------------------
# 1.  THE ORIGINAL DRAWING, read through rig_final's extraction
# ---------------------------------------------------------------------------
_P0 = np.asarray(rig_final.PAPER_ORIGIN_W_CM)      # W-frame paper-top corner


def _dbox(name):
    """A named FRAME_BOXES_W_CM entry -> (lo, hi) in cm, drawing W frame."""
    for b in rig_final.FRAME_BOXES_W_CM:
        if b["name"] == name:
            return np.asarray(b["lo"], float), np.asarray(b["hi"], float)
    raise KeyError(name)


def _pz(z_cm):
    """Drawing W-frame z (cm) -> mm above the paper top surface."""
    return float((z_cm - _P0[2]) * 10.0)


_BW_LO, _BW_HI = _dbox("down_boom_W")      # arm-31 drop post, west
_BE_LO, _BE_HI = _dbox("down_boom_E")      # arm-31 drop post, east
_GW_LO, _GW_HI = _dbox("down_gusset_W")
_CL_LO, _CL_HI = _dbox("down_clamps")
_PL_LO, _PL_HI = _dbox("down_plate")
_TS_LO, _TS_HI = _dbox("top_slab")
_TB_LO, _TB_HI = _dbox("table_block")
_AX = np.asarray(rig_final.ARM_MOUNTS_W["down"]["p_w_cm"], float)   # J1 axis

# --- member sections (mm) --------------------------------------------------
PROFILE = 76.2                     # 3 in square T-slot, the drawing's MTEXT
PROFILE_THIN = 38.1                # 1.5 in, the permitted alternate

# --- the arm-31 ceiling mount, as built (mm) -------------------------------
PLATE = tuple(np.round((_PL_HI - _PL_LO) * 10.0, 2))          # 225.82x190x12.7
POST_PITCH_X = round(float((_BE_LO[0] + _BE_HI[0]
                            - _BW_LO[0] - _BW_HI[0]) / 2 * 10.0), 2)   # 317.6
# The drawing's own booms are 77.2 and 77.0 wide, so the slot they leave is
# 240.5.  This model builds nominal 76.2 posts at that pitch, so ITS slot is
# 241.4 and the plate has 7.79 mm each side rather than 7.34.  Both are
# carried: the fabricator gets the model's, and the 240.5 stays because it is
# what the drawing measures.
POST_GAP = round(float((_BE_LO[0] - _BW_HI[0]) * 10.0), 2)             # 240.5
POST_SLOT = round(POST_PITCH_X - PROFILE, 2)                           # 241.4
PLATE_SIDE_CLEAR = round((POST_SLOT - PLATE[0]) / 2, 2)                # 7.79
POST_PITCH_Y = PROFILE                                                 # 76.2
POST_Y_BAND = round(float((_BW_HI[1] - _BW_LO[1]) * 10.0), 2)          # 152.4
PLATE_OFF = round(float(((_PL_LO[0] + _PL_HI[0]) / 2 - _AX[0]) * 10.0), 2)
POST_OVER = round(float((_PL_LO[2] - _BW_LO[2]) * 10.0), 2)            # 34.98
CLAMP = tuple(np.round((_CL_HI - _CL_LO) * 10.0, 1))          # 226x152.4x95.7
GUSSET = (203.2, PROFILE_THIN, 203.2)          # nominal 8 x 1.5 x 8 in
RUNWAY_W = POST_Y_BAND                         # 152.4, the double beam

# --- the original's own z ladder, mm above the paper top -------------------
O_POST_L = round(float((_BW_HI[2] - _BW_LO[2]) * 10.0), 2)    # 736.9 = 29.0 in
O_MOUNT = round(_pz(_PL_LO[2]), 2)                            # 922.00
O_BEAM_U = round(_pz(_TS_LO[2]), 2)                           # 1623.62
O_TOP = round(_pz(_TS_HI[2]), 2)                              # 1699.82
O_POST_B = round(_pz(_BW_LO[2]), 2)                           # 887.02

# --- the floor, the table, the paper ---------------------------------------
FLOOR_Z = round(_pz(_TB_LO[2]), 2)              # -636.68, drawing W-frame z=0
TABLE_TOP_Z = round(_pz(_TB_HI[2]), 2)          # -2.00, paper underside
PAPER_T = round(-TABLE_TOP_Z, 2)                # 2.00 mm of paper
CAGE_TOTAL_H = round(O_TOP - FLOOR_Z, 2)        # 2336.50 == the drawing's 233,7

# ---------------------------------------------------------------------------
# 2.  THE UPDATED LAYOUT, read from the package
# ---------------------------------------------------------------------------
CANVAS_W = layout.SHEET_FINAL6[0] * MM          # 1803.40
CANVAS_L = layout.SHEET_FINAL6[1] * MM          # 3630.64
H_MOUNT = float(layout.LAYOUT_PROPOSED["h"]) * MM               # 940.0
ARM_XY = {aid: (s.xy[0] * MM, s.xy[1] * MM)
          for aid, s in layout.FLEET_PROPOSED.items()}
COL_X = sorted({x for x, _ in ARM_XY.values()})                 # 596.7, 1206.7
ROW_Y = sorted({y for _, y in ARM_XY.values()})   # 605.11, 1815.32, 3025.53

# ---------------------------------------------------------------------------
# 3.  THE CORRECTED CAGE
# ---------------------------------------------------------------------------
MARGIN = 190.5                     # 7.5 in, frame outside clear of the canvas
FR_X0, FR_X1 = -MARGIN, CANVAS_W + MARGIN          # -190.5 .. 1993.9
FR_Y0, FR_Y1 = -MARGIN, CANVAS_L + MARGIN          # -190.5 .. 3821.14
FR_W = FR_X1 - FR_X0                               # 2184.4 = 86.00 in
FR_L = FR_Y1 - FR_Y0                               # 4011.64
IN_X0, IN_X1 = FR_X0 + PROFILE, FR_X1 - PROFILE    # -114.3 .. 1917.7
RAIL_LEN_X = FR_W - 2 * PROFILE                    # 2032.0 = 80.00 in
RAIL_LEN_Y = FR_L                                  # 4011.64

# THE CORRECTED DATUM.  The grid sits where the drawing's own top slab sits.
GRID_U = O_BEAM_U                                  # 1623.62, beam underside
GRID_T = round(GRID_U + PROFILE, 2)                # 1699.82, top of cage
LEG_BOTTOM = round(_pz(_dbox("post_FL")[0][2]), 2)      # -27.38, as drawn

# THE BUGGY DATUM, kept so the two can be printed side by side.
GRID_U_CODE = mounts.MOUNTS.ceiling_z * MM              # 2340.0
GRID_T_CODE = round(GRID_U_CODE + PROFILE, 2)           # 2416.2

POST_B = round(H_MOUNT - POST_OVER, 2)                  # 905.02
POST_L = round(GRID_U - POST_B, 2)                      # 718.60 at h = 940
POST_L_CODE = round(GRID_U_CODE - POST_B, 2)            # 1434.98

# The gusset fix.  Four 8-in gussets per arm in the ORIGINAL orientation face
# each other across a transverse pair and need 406.4 mm where only 216.20 mm
# exists (the sheet's UNKNOWN 2).  Rotating them onto the runway's OUTBOARD y
# faces braces the cluster's weak axis — the posts are only 76.2 apart in y —
# and centres each 203.2 mm plate on its own post in x instead of cantilevering
# it inboard, which clears the conflict.  `GUSSET_PAIR_CLEAR` is what it buys.
GUSSET_GAP = round((COL_X[1] - PLATE_OFF - POST_PITCH_X / 2 - PROFILE / 2)
                   - (COL_X[0] - PLATE_OFF + POST_PITCH_X / 2 + PROFILE / 2), 2)
GUSSET_NEED = round(2 * GUSSET[0], 2)                   # 406.4
GUSSET_PAIR_CLEAR = round((COL_X[1] - PLATE_OFF - POST_PITCH_X / 2
                           - GUSSET[0] / 2)
                          - (COL_X[0] - PLATE_OFF + POST_PITCH_X / 2
                             + GUSSET[0] / 2), 2)

CLUSTER_W = round(POST_PITCH_X + PROFILE, 2)            # 393.8, plan x
CLUSTER_D = POST_Y_BAND                                 # 152.4, plan y


def plate_centre_x(x_axis):
    """Plate centre x for a J1 axis at `x_axis` (mm), canvas frame.

    In the drawing the plate centre sits 25.15 mm in -x from the J1 axis with
    the arm's front facing +X.  Every arm on this rig is clocked the other way
    (`R_world_base = Ry(180)`, front toward canvas -x, docs/BUILD_SHEET.md
    §3), so the offset flips: the plate centre is 25.15 mm in canvas +x.

    THE DIRECTION IS AN INFERENCE and 7.3 mm rides on it — the plate nests in
    a 241.4 mm slot with 7.79 mm clear each side, so a plate offset the wrong
    way does not fit.  See OPEN_QUESTIONS["plate_offset_direction"].
    """
    return float(x_axis) - PLATE_OFF


def post_x(x_axis):
    """The two drop-post x centres for a J1 axis at `x_axis` (mm)."""
    c = plate_centre_x(x_axis)
    return c - POST_PITCH_X / 2, c + POST_PITCH_X / 2


def z_ladder(h=None):
    """Every z datum of the corrected cage, mm above the paper top."""
    h = H_MOUNT if h is None else float(h)
    return {
        "floor": FLOOR_Z,
        "table_top": TABLE_TOP_Z,
        "paper_top": 0.0,
        "leg_bottom": LEG_BOTTOM,
        "post_bottom": round(h - POST_OVER, 2),
        "mount_plane": h,
        "plate_top": round(h + PLATE[2], 2),
        "clamp_top": round(h + PLATE[2] + CLAMP[2], 2),
        "gusset_bottom": round(GRID_T - GUSSET[2], 2),
        "grid_underside": GRID_U,
        "grid_top": GRID_T,
        "post_length": round(GRID_U - (h - POST_OVER), 2),
    }


# ---------------------------------------------------------------------------
# 4.  BODIES
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Body:
    """One rigid body of the model, as an axis-aligned box in the canvas frame.

    `lo`/`hi` are mm in the canvas frame (x across the short side of the paper,
    y along it, z up, z = 0 the paper's top surface).  `provenance` is one of
    DRAWING / CODE / AUDIT / ASSUMED; `source` says exactly which entity,
    constant or measurement.
    """
    name: str
    kind: str                    # cage | table | canvas | mount | tool | arm
    lo: tuple
    hi: tuple
    provenance: str
    source: str
    rgba: tuple = (0.62, 0.65, 0.69, 1.0)
    collision: bool = True
    note: str = ""

    @property
    def size(self):
        return tuple(round(b - a, 4) for a, b in zip(self.lo, self.hi))

    @property
    def centre(self):
        return tuple(round(0.5 * (a + b), 4) for a, b in zip(self.lo, self.hi))


PROVENANCE_CLASSES = ("DRAWING", "CODE", "AUDIT", "ASSUMED")

_STEEL = (0.66, 0.69, 0.73, 1.0)
_STEEL_DARK = (0.46, 0.49, 0.54, 1.0)
_PLATE_C = (0.78, 0.72, 0.62, 1.0)
_CLAMP_C = (0.34, 0.36, 0.40, 1.0)
_PAPER_C = (0.97, 0.96, 0.93, 1.0)
_TABLE_C = (0.42, 0.34, 0.26, 1.0)
_FLOOR_C = (0.30, 0.31, 0.33, 1.0)


def _b(name, kind, lo, hi, provenance, source, rgba=_STEEL, collision=True,
       note=""):
    if provenance not in PROVENANCE_CLASSES:
        raise ValueError(f"{name}: provenance {provenance!r} is not one of "
                         f"{PROVENANCE_CLASSES}")
    lo = tuple(round(float(v), 4) for v in lo)
    hi = tuple(round(float(v), 4) for v in hi)
    # raise rather than assert: a degenerate box writes a degenerate URDF, and
    # `python -O` must not be able to turn that check off
    if not all(b > a for a, b in zip(lo, hi)):
        raise ValueError(f"{name}: lo {lo} is not strictly below hi {hi}")
    return Body(name, kind, lo, hi, provenance, source, rgba, collision, note)


def ground_bodies():
    """Floor, table and paper -> [Body].  The datums the cage stands on."""
    pad = 600.0        # how far past the frame the floor/table are drawn
    return [
        _b("floor", "table",
           (FR_X0 - pad, FR_Y0 - pad, FLOOR_Z - 20.0),
           (FR_X1 + pad, FR_Y1 + pad, FLOOR_Z),
           "DRAWING",
           "drawing W-frame z = 0 is the floor; table_block sits on it. "
           "Extent is SCHEMATIC (the room is not surveyed).",
           _FLOOR_C, collision=False,
           note="visual datum only — the room has never been surveyed"),
        _b("table", "table",
           (IN_X0, FR_Y0 + PROFILE, FLOOR_Z),
           (IN_X1, FR_Y1 - PROFILE, TABLE_TOP_Z),
           "ASSUMED",
           f"top and floor from the drawing's table_block "
           f"({FLOOR_Z} .. {TABLE_TOP_Z} mm about the paper); the FOOTPRINT is "
           f"assumed to be the cage's INNER span, because the original's "
           f"2.08 m table cannot carry a {CANVAS_L / 1000:.2f} m canvas and a "
           "table drawn out to the frame's OUTSIDE would swallow the legs",
           _TABLE_C,
           note="height is DRAWING, footprint is ASSUMED"),
        _b("paper", "canvas",
           (0.0, 0.0, TABLE_TOP_Z), (CANVAS_W, CANVAS_L, 0.0),
           "CODE",
           f"layout.SHEET_FINAL6 = {layout.SHEET_FINAL6}; thickness "
           f"rig_final.PAPER_THICK_CM = {rig_final.PAPER_THICK_CM} cm",
           _PAPER_C),
    ]


def cage_bodies(h=None):
    """The 80/20 cage -> [Body], at mount plane `h` (mm, default H_MOUNT)."""
    h = H_MOUNT if h is None else float(h)
    z = z_ladder(h)
    out = []

    # --- perimeter, at the grid plane -------------------------------------
    for tag, x0, x1 in (("W", FR_X0, IN_X0), ("E", IN_X1, FR_X1)):
        out.append(_b(f"frame_side_{tag}", "cage",
                      (x0, FR_Y0, GRID_U), (x1, FR_Y1, GRID_T), "DRAWING",
                      f"3 in T-slot perimeter rail, full length {RAIL_LEN_Y} "
                      f"in y; section and 86.00 in frame width are the "
                      f"original frame's own cuts (post_FL..post_BR, "
                      f"top_slab)"))
    for tag, y0, y1 in (("S", FR_Y0, FR_Y0 + PROFILE),
                        ("N", FR_Y1 - PROFILE, FR_Y1)):
        out.append(_b(f"frame_end_{tag}", "cage",
                      (IN_X0, y0, GRID_U), (IN_X1, y1, GRID_T), "DRAWING",
                      f"3 in T-slot perimeter end rail {RAIL_LEN_X} = "
                      f"{RAIL_LEN_X / IN:.2f} in, butting between the side "
                      f"rails"))

    # --- the four corner legs ---------------------------------------------
    for tag, cx, cy in (("FL", FR_X0, FR_Y0), ("FR", FR_X1 - PROFILE, FR_Y0),
                        ("BL", FR_X0, FR_Y1 - PROFILE),
                        ("BR", FR_X1 - PROFILE, FR_Y1 - PROFILE)):
        out.append(_b(f"leg_{tag}", "cage",
                      (cx, cy, LEG_BOTTOM),
                      (cx + PROFILE, cy + PROFILE, GRID_U),
                      "ASSUMED",
                      "the cage is self-supporting and floor-standing, so it "
                      "needs legs; section and plan position mirror the "
                      f"drawing's post_{tag}.  It stops at the perimeter "
                      f"rail's UNDERSIDE ({GRID_U}), the way the drop posts "
                      "stop under the runway — a leg drawn to the rail's TOP "
                      "is 76.2 mm of steel inside the rail and a length no "
                      f"one can cut.  Cut length {round(GRID_U - LEG_BOTTOM, 2)}"
                      ".  But a "
                      f"{FR_L / 1000:.2f} m frame on four legs has NO "
                      "precedent — the original spans 2.08 m",
                      _STEEL_DARK,
                      note="mid-span legs are almost certainly required; "
                           "see OPEN_QUESTIONS['cage_legs']"))

    # --- three double-beam runways ----------------------------------------
    for r, ry in enumerate(ROW_Y):
        for half, y0 in (("S", ry - PROFILE), ("N", ry)):
            out.append(_b(f"runway_r{r}_{half}", "cage",
                          (IN_X0, y0, GRID_U), (IN_X1, y0 + PROFILE, GRID_T),
                          "DRAWING",
                          "one of the two 3 in beams of the drawing's CENTRAL "
                          f"DOUBLE BEAM ({RUNWAY_W} overall); the row line, "
                          "and so the arms' J1 axes, lie on the seam"))

    # --- per-arm drop cluster ---------------------------------------------
    for aid, (xa, ya) in ARM_XY.items():
        pxs = post_x(xa)
        for i, px in enumerate(pxs):
            for j, py in enumerate((ya - PROFILE, ya)):
                out.append(_b(f"post{aid}_{i}{j}", "cage",
                              (px - PROFILE / 2, py, z["post_bottom"]),
                              (px + PROFILE / 2, py + PROFILE, GRID_U),
                              "DRAWING",
                              "one of four 3 in drop posts in a 2x2 cluster, "
                              f"pitch {POST_PITCH_X} (x) x {POST_PITCH_Y} (y) "
                              f"= down_boom_W/E; length {z['post_length']} at "
                              f"h = {h:.0f} (the original's own is "
                              f"{O_POST_L})",
                              _STEEL_DARK))
        # gussets, rotated onto the runway's outboard y faces
        for i, px in enumerate(pxs):
            for j, (y0, y1) in enumerate(
                    ((ya - PROFILE - GUSSET[1], ya - PROFILE),
                     (ya + PROFILE, ya + PROFILE + GUSSET[1]))):
                out.append(_b(f"gusset{aid}_{i}{j}", "cage",
                              (px - GUSSET[0] / 2, y0, z["gusset_bottom"]),
                              (px + GUSSET[0] / 2, y1, GRID_T),
                              "ASSUMED",
                              f"an {GUSSET[0]} x {GUSSET[2]} x {GUSSET[1]} "
                              "gusset at the drawing's own size, but ROTATED "
                              "onto the runway's outboard y face — the "
                              "drawing's inboard orientation needs "
                              f"{GUSSET_NEED} mm across a transverse pair and "
                              f"only {GUSSET_GAP} exists",
                              _STEEL_DARK,
                              note=f"the rotation leaves {GUSSET_PAIR_CLEAR} "
                                   "mm clear between a pair's gussets; exact "
                                   "attachment is a fabrication detail"))
        # the clamp stack, on top of the plate, between the post pairs
        out.append(_b(f"clamp{aid}", "cage",
                      (plate_centre_x(xa) - CLAMP[0] / 2, ya - CLAMP[1] / 2,
                       z["plate_top"]),
                      (plate_centre_x(xa) + CLAMP[0] / 2, ya + CLAMP[1] / 2,
                       z["clamp_top"]),
                      "DRAWING",
                      f"clamp stack {CLAMP[0]} x {CLAMP[1]} x {CLAMP[2]} = "
                      "down_clamps, sitting on the plate between the post "
                      "pairs.  ENVELOPE ONLY — no part number in the drawing",
                      _CLAMP_C))
        # the robot mounting plate
        out.append(_b(f"plate{aid}", "mount",
                      (plate_centre_x(xa) - PLATE[0] / 2, ya - PLATE[1] / 2,
                       h),
                      (plate_centre_x(xa) + PLATE[0] / 2, ya + PLATE[1] / 2,
                       z["plate_top"]),
                      "DRAWING",
                      f"down_plate {PLATE[0]} x {PLATE[1]} x {PLATE[2]}; the "
                      "arm bolts to its UNDERSIDE, which IS the mount plane.  "
                      f"Centre offset {-PLATE_OFF} mm in canvas +x from the "
                      "J1 axis — DIRECTION INFERRED",
                      _PLATE_C,
                      note=f"nests in this model's {POST_SLOT} mm post slot "
                           f"with {PLATE_SIDE_CLEAR} mm each side (the "
                           f"drawing's own booms leave {POST_GAP}, i.e. "
                           f"{(POST_GAP - PLATE[0]) / 2:.2f} each side)"))
    return out


def modelled_envelope(aid, h=None):
    """`mounts.arm_mount_boxes` for one arm, in mm -> [(lo, hi)].

    The SCHEMATIC keep-out every certified number was earned against: a
    0.226 x 0.190 x 0.05 plate on the base axis and a 0.2 x 0.2 column above
    it to `ceiling_z`.  Read straight out of `mounts.py`, never restated.
    """
    spec = layout.FLEET_PROPOSED[aid]
    h_m = (H_MOUNT if h is None else float(h)) / MM
    return [(np.asarray(b["lo"], float) * MM, np.asarray(b["hi"], float) * MM)
            for b in mounts.arm_mount_boxes(spec.mount, spec.xy, spec.yaw,
                                            h_m, tag=f"mount{aid}")]


def recert_escape_mm(body, h=None):
    """How far `body` reaches outside the modelled keep-out -> mm, or None.

    None when the body is not a piece of one arm's mount hardware.  0.0 when
    it fits inside what was certified.  Anything positive is steel the
    certified envelope does not contain, and therefore a RE-CERTIFICATION
    item — the drop cluster is the big one, and it is 38.15 mm in x on the
    plate alone, purely because the plate is offset off the axis.
    """
    aid = None
    for a in layout.FLEET_PROPOSED:
        for pfx in (f"post{a}_", f"gusset{a}_", f"clamp{a}", f"plate{a}"):
            if body.name.startswith(pfx):
                aid = a
    if aid is None:
        return None
    lo = np.asarray(body.lo, float)
    hi = np.asarray(body.hi, float)
    boxes = modelled_envelope(aid, h)

    # THE UNION, NOT ITS BOUNDING BOX.  The two envelope boxes stack in z but
    # do NOT share a footprint: the plate is 226 x 190 over z in [h, h+50] and
    # the column is 200 x 200 above it.  Merging them into one AABB would
    # score a gusset 1.5 m up — where only the 200-wide column exists —
    # against a 226-wide envelope, and report every escape above the plate
    # band 13 mm SHORT.  Under-reporting is the wrong direction for a re-cert
    # queue, so the bands are walked separately.
    esc = 0.0
    # z first, against the union's own span (the boxes are contiguous in z)
    zlo = min(float(b[0][2]) for b in boxes)
    zhi = max(float(b[1][2]) for b in boxes)
    esc = max(esc, zlo - lo[2], hi[2] - zhi)
    # then x and y, against each band the body actually reaches into
    for blo, bhi in boxes:
        if hi[2] > blo[2] and lo[2] < bhi[2]:
            for ax in (0, 1):
                esc = max(esc, blo[ax] - lo[ax], hi[ax] - bhi[ax])
    return round(float(max(0.0, esc)), 2)


def bodies(h=None):
    """Every static body of the installation -> [Body]."""
    return ground_bodies() + cage_bodies(h)


# ---------------------------------------------------------------------------
# 5.  THE ARMS AND THE TOOL
# ---------------------------------------------------------------------------
def arm_collision_capsules():
    """The AUDITED per-link capsule set -> [(link, a, b, radius)], metres.

    `selfcoll.BODY_CAPSULES`, whose radii were fitted to the manufacturer's
    own collision meshes by `scripts/self_collision_audit.py` (each row's
    comment records the exact mesh maximum the shipped radius rounds up from).
    This is one of the two collision models the generator can emit; the other
    is the manufacturer's shells themselves.
    """
    link_of = {i: f"panda_link{i}" for i in range(8)}
    link_of[9] = "panda_hand"
    out = []
    for name, band, idx, a, b, r in selfcoll.BODY_CAPSULES:
        out.append((link_of[idx], np.asarray(a, float), np.asarray(b, float),
                    float(r), f"{name}.{band}"))
    return out


# The manufacturer's collision shells.  `scripts/collision_audit.py --validate`
# put these AABBs against the FR3's own collision boxes in the station's
# fr3_franka_hand.urdf: eight of the nine agree to <= 0.5 um and link6 to
# 0.344 mm, i.e. the Panda and FR3 collision shells are the same object.
# Drake reports their convex hulls as the same volume to 0.1 % on nine of ten
# (link6 4.2 %), so they are already convex and nothing is lost by using them.
COLLISION_MESH_SOURCE = "vamp/resources/panda/meshes/collision"
PANDA_VS_FR3_WORST_MM = 0.344      # out/collision_audit.json validate

# The cable dress.  NOT MEASURED — the collision audit's estimate is a 20-40 mm
# conduit dressed along the forearm and around the wrist.  Visual only.
CABLE_R = 30.0                     # mm, mid of the audit's 20-40 estimate
CABLE_DRESS = (
    # (parent link, a (m, link frame), b (m, link frame), what it is)
    ("panda_link4", (-0.075, 0.020, 0.030), (-0.020, 0.100, 0.055),
     "forearm service loop"),
    ("panda_link6", (0.020, -0.030, 0.020), (0.095, 0.045, 0.010),
     "wrist service loop"),
)


# ---------------------------------------------------------------------------
# 6.  RECONCILIATION AND OPEN QUESTIONS
# ---------------------------------------------------------------------------
def reconciliation(h=None):
    """Where this model and `mounts.py` disagree -> [dict], each a re-cert item.

    Nothing here changes `mounts.py`.  Every row is a number the certified
    obstacle model carries and the physical model contradicts, with which way
    the difference cuts.
    """
    h = H_MOUNT if h is None else float(h)
    m = mounts.MOUNTS
    z = z_ladder(h)
    return [
        dict(item="ceiling datum",
             code=f"mounts.MOUNTS.ceiling_z = {m.ceiling_z} m "
                  f"({GRID_U_CODE:.1f} mm above the paper)",
             model=f"grid underside {GRID_U} mm, top {GRID_T} mm above the "
                   f"paper ({CAGE_TOTAL_H} above the floor = the drawing's "
                   f"233,7 cm)",
             delta_mm=round(GRID_U_CODE - GRID_U, 2),
             direction="CODE IS CONSERVATIVE — the schematic boom is modelled "
                       f"{GRID_U_CODE - GRID_U:.1f} mm too long, and a longer "
                       "obstacle never certifies a pose a shorter one refuses",
             action="survey the real room, then re-cut the ladder and "
                    "re-certify; the fabricator's post length is "
                    f"{POST_L} not {POST_L_CODE}"),
        dict(item="drop cluster vs boom column",
             code=f"a single circumscribed square column {2 * m.boom_r * MM:.0f}"
                  f" x {2 * m.boom_r * MM:.0f} mm (a radius-{m.boom_r * MM:.0f}"
                  " cylinder) on the base axis",
             model=f"a 2x2 post cluster {CLUSTER_W} x {CLUSTER_D} mm in plan, "
                   f"offset {-PLATE_OFF} mm in +x, plus gussets "
                   f"{GUSSET[0]} mm wide",
             delta_mm=round(CLUSTER_W - 2 * m.boom_r * MM, 2),
             direction="MODEL IS WIDER IN X, NARROWER IN Y — the certified "
                       "column neither contains nor is contained by the real "
                       "steel.  MEASURED per body: all 60 pieces of mount "
                       "hardware escape it, worst 185.55 mm (a gusset), and "
                       "the PLATE escapes by 25.06 mm in plan even though the "
                       "sheet reads it as inside on thickness alone",
             action="RE-CERT REQUIRED before fabrication; send the steel "
                    "design back"),
        dict(item="mount plate thickness",
             code=f"{m.plate_t * MM:.0f} mm thick, "
                  f"{m.plate_xy[0] * MM:.0f} x {m.plate_xy[1] * MM:.0f}",
             model=f"{PLATE[2]} mm thick, {PLATE[0]} x {PLATE[1]}",
             delta_mm=round((m.plate_t * MM) - PLATE[2], 2),
             direction="CONSERVATIVE IN THICKNESS ONLY.  50 modelled against "
                       "12.7 real, and 226 against 225.82 — but the modelled "
                       "plate is centred on the J1 axis and the real one sits "
                       f"{-PLATE_OFF} mm off it, so IN PLAN it escapes by "
                       f"{25.06} mm.  The layout sheet reads this row as "
                       "'INSIDE' on the thickness alone; it is not",
             action="re-certify with the plate at its true offset — it is one "
                    "of the 60 bodies the queue already carries"),
        dict(item="steel below the mount plane",
             code="none modelled below z = h",
             model=f"the drop posts run {POST_OVER} mm past the plate "
                   "underside",
             delta_mm=POST_OVER,
             direction="MODEL HAS STEEL THE CODE DOES NOT — but the inverted "
                       f"chain gap is "
                       f"{(m.inv_chain_drop - m.chain_r - m.margin) * MM:.0f} "
                       f"mm, so "
                       f"{(m.inv_chain_drop - m.chain_r - m.margin) * MM - POST_OVER:.0f}"
                       " mm of it is still spare",
             action="confirm at re-certification rather than take it from "
                    "here"),
        # CLOSED 2026-09-10.  The sheet was re-issued at the height in force
        # (970.0) and at the corrected datum, so code, model and the paper the
        # fabricator holds now say the same number.  Kept in the queue as a
        # zero-delta row rather than deleted, because the item is "do these
        # three agree" and the answer has been no twice.
        dict(item="mount height",
             code=f"layout.LAYOUT_PROPOSED['h'] = {h / MM} m",
             model=f"the same {h:.0f} mm, and docs/BUILD_SHEET.md publishes "
                   f"{h:.1f} to the fabricator (re-issued 2026-09-10)",
             delta_mm=0.0,
             direction="agreed",
             action="none — re-issue the sheet again if h moves"),
        dict(item="base cable pass-through",
             code="not modelled at all — the collision shell for link0 stops "
                  "at the base flange (z = 0)",
             model="the manufacturer's link0 VISUAL carries the connector and "
                   "cable stub 230.7 mm past the flange, which on an inverted "
                   "arm is 230.7 mm up through the plate and the clamp stack",
             delta_mm=230.7,
             direction="NEITHER MODEL HAS IT AS AN OBSTACLE, and the steel "
                       "that has to be cut for it is not drawn",
             action="cut the plate and the clamp stack; see "
                    "OPEN_QUESTIONS['base_cable_passthrough']"),
        dict(item="cage legs",
             code="not modelled — no structure below z = h anywhere",
             model=f"four corner legs, {LEG_BOTTOM} .. {GRID_T} mm, at the "
                   "frame corners",
             delta_mm=round(GRID_U - LEG_BOTTOM, 2),
             direction="MODEL HAS STEEL THE CODE DOES NOT, but it stands "
                       f"{min(abs(FR_X0), abs(FR_Y0)):.0f} mm clear of the "
                       "canvas on every side",
             action="confirm the legs miss every certified pose once their "
                    "count and position are decided"),
    ]


OPEN_QUESTIONS = {
    "plate_offset_direction": dict(
        what="Does the mount plate's 25.15 mm offset run in canvas +x or -x?",
        why="The drawing gives the offset for an arm whose front faces +X; "
            "this rig clocks every arm the other way, so the offset flips. "
            "That flip is an INFERENCE about which edge of a Franka base "
            "plate is its front (rig_final flags it too).",
        rides_on="7.79 mm.  The plate nests in this model's 241.4 mm slot "
                 "between the post pairs with 7.79 mm clear each side (the "
                 "drawing's own slightly-fat booms leave 240.5 and 7.34); "
                 "offset the wrong way it does not fit at all.",
        answer_by="measure the real plate, or open the post gap to 304.8 mm "
                  "(posts at axis +/- 190.5), which fits either way with "
                  "14.4 mm each side",
        blocking="fabrication of the drop clusters"),
    "ceiling_survey": dict(
        what="How high is the real room's ceiling above the paper?",
        why="The drawing defines NO room ceiling — the cage is self-"
            "supporting. Every ceiling number in this repo descends from the "
            "drawing's 233,7 cm, which is floor-to-top-of-cage.",
        rides_on="716.4 mm of modelled boom length, and whether the cage can "
                 "stand up at all in the room it is going into.",
        answer_by="survey the room",
        blocking="cutting the grid"),
    "penholder_cradle": dict(
        what="Which fingers are on the arms, and how far in is the blade's "
             "foot bolted?  (THE LEAN IS SETTLED, 2026-09-07: 23 deg, the "
             "housing's own clocking, which is the only lean that puts the "
             "square block flush with the blades.  The graphite is "
             "USER-SPECIFIED at 20 mm past the cap.)",
        why="THERE IS NO CRADLE, AND THERE IS NO FEATURE ON THE FAT FINGER "
            "EITHER.  The 2026-09-02 mesh of 'Fat Franka Finger v250904' has "
            "been read (rig_final.FATFINGER, docs/SYSTEM_MODEL.md 7d).  It is "
            "drawn in the FR3 fingertip's own CAD frame — its plate hole and "
            "that tip's brass-insert axis agree to 0.0002 mm — so its place "
            "on the hand is fixed by placing the fingertip, with a worst "
            "residual of 0.155 mm against the manufacturer's own meshes.  Its "
            "contact face is ONE FLAT PLANE: two 6.000 mm holes 69 mm apart "
            "(mirror twins, because the part serves both fingers), R5 corners "
            "and nothing else.  The post has nothing to receive a pin either "
            "— rays down the 22-deg housing's post axis hit solid material at "
            "z = 5.426 and 44.574, so the socket floor is a chamfered cone "
            "and not a bore.  THE CLOCKING ABOUT THE JAW AXIS IS THEREFORE "
            "FREE unless a FINGERTIP is fitted AND seated in the housing's "
            "own 18 x 18 mm socket, which is a decision made by hand at grasp "
            "time.  45 deg and 23 deg can both be true statements about "
            "different builds.  WHAT THE ASSEMBLY ALREADY SAID, and still "
            "does: the fingertip cradle was the escape hatch, and the "
            "assembly "
            "closed it.  raw_slack_file_dump/'Pen holder cad(1).zip' nests "
            "'Natural hold assembly - closed.zip', which holds the COMPLETE "
            "10-deg build as an .SLDASM.  Resolved, it says: the two "
            "fingertips are stock FR3 tips (drilled for brass inserts, "
            "nothing more), they seat 7.000 mm into the mount post's own "
            "18 x 18 mm end sockets, the post axis is the finger-travel axis "
            "to 0 deg, the grip centre lands 103.26 mm from panda_hand "
            "against the stock TCP's 103.4 — and the bore comes out 10.0000 "
            "deg off the hand's approach axis, which is exactly the angle in "
            "that housing's file name.  This delivery's housing is named "
            "'22 deg' and its flats measure 23.00.  Same naming, same "
            "sockets, same square-seated tips: on THAT build the clocking "
            "reaches the hand undivided.  AND THE PHOTO OF THE REAL GRIPPER "
            "(2026-09-03) SAYS THAT BUILD IS NOT WHAT IS MOUNTED: no "
            "fingertip is fitted, and the Fat blades' bare plates clamp the "
            "post's END FACES.  So the clocking is free on the deployed "
            "build — and what then sets the lean is the HAND.  The post sits "
            "55.099 mm from the housing's tail face, so 55.1 mm of barrel "
            "stands behind the grip pointing at the wrist, and the grip is "
            "only 37.4 mm below the hand's underside.  Swept against the "
            "manufacturer's own hand collision shell, the raw housing STL was "
            "INSIDE the hand at every lean below 35.17 deg: -11.90 mm at 23, "
            "-0.16 at 35, +6.16 at 45.  THAT ARGUMENT IS WITHDRAWN "
            "(2026-09-04): it was measured with the grip on the finger "
            "centreline, and the photo read again puts the holder at the FAR "
            "END of the plates (grip_hand_x = 66.5 mm), where 23 deg clears "
            "the casing by 13.84 mm against 45 deg's 7.78.  The lean is the "
            "user's standing rule, not a forced choice — and per the user the "
            "casing does not constrain the model at all while the pencil's "
            "modelled length is arbitrary.",
        rides_on="HOW MUCH GRAPHITE, and it is an INPUT now.  The tool "
                 "transform moved three times on one photograph and settled "
                 "2026-09-07 at frames.PEN_EXT_HOLDER 0.0460262 / "
                 "PEN_LAT_HOLDER 0.0860369, derived in one line: a grip at "
                 "(0.066500, 0, 0.1034) — the far end of the Fat blades' "
                 "plates — plus 30.001 mm of holder and 20.000 mm of graphite "
                 "along a 23 deg bore.  The 20 mm is USER-SPECIFIED (\"about "
                 "2 cm\", orientation confirmed); it was 53.2 mm while the tip "
                 "came off the blades and 125.6 at the old 0.110 pair, which "
                 "asked for a 283 mm stick.  The protrusion is adjustable, so "
                 "one ruler reading confirms rather than sets it (see "
                 "docs/SYSTEM_MODEL.md 7e).  Neither number was ever "
                 "gate-validated — only the INLINE pen's axial 0.110 was "
                 "(gate B, MZ 0.924); frames.py says so.  RIDING ON "
                 "IT AS WELL, since the housing was found mounted END-FOR-END "
                 "(docs/SYSTEM_MODEL.md 7c, fixed 2026-09-03): with the "
                 "housing the right way round the two lateral tool capsules "
                 "rig_final.STATIC_CAPSULES_LAT ships no longer CONTAIN it.  "
                 "Housing + cap escape their r = 0.050 by 6.546 mm (bracket "
                 "r 0.0565 needed) where the end-for-end placement was "
                 "contained by 1.720 mm, and the pencil tail escapes by "
                 "77.661 mm (r 0.1277).  Re-run at those radii the 100 % "
                 "CSAIL programme's inter-arm minimum falls from 80.7 mm to "
                 "79.1 mm and then to -18.2 mm, i.e. the 80 mm gate FAILS in "
                 "both cases.  No capsule radius was widened; that is the "
                 "re-certification, and it is the tail that makes it "
                 "expensive.  RIDING ON "
                 "IT AS WELL, since the Fat finger was read: the self-"
                 "collision guard.  A 90 mm blade reaching 69 mm sideways out "
                 "of the hand escapes selfcoll.BODY_CAPSULES' hand rows by "
                 "49.93 mm over the finger-joint range, where the stock "
                 "finger is contained with 3.14 mm to spare, and it leaves "
                 "coordination.HAND_R = 0.104 just 0.51 mm of margin at full "
                 "open.  If the Fat fingers ship, the guard needs a finger "
                 "row.  Reported in docs/SYSTEM_MODEL.md 7d, changed nowhere.",
        answer_by="TWO measurements now, and the robot already knows one of "
                  "them.  (1) THE RULER: the perpendicular distance from the "
                  "mounted pen's tip to the gripper's approach axis.  At the "
                  "45 deg lean the model says 58.8 mm; docs/SYSTEM_MODEL.md "
                  "7e turns any reading into a tip.  Note the SIGN is a "
                  "mounting choice either way — the post is square, so the "
                  "holder seats both ways up and the lean is +/-45 deg.  "
                  "(2) THE GRIPPER'S OWN `width` "
                  "while the pen is held (franka::GripperState, or a caliper "
                  "across the jaw).  `width` is 2 q, the STOCK grip plane's "
                  "opening, and every build adds back how far its real "
                  "contact face sits outboard of it — so ONE number picks one "
                  "row out of six — and the photo already cuts it to the "
                  "first two, since no fingertip is fitted: 0.0287 fat "
                  "plates on the bare post ends, "
                  "0.0339 fat RIBS on them (the plates cannot reach; the rib "
                  "stands 2.5839 mm proud), 0.0357 fat plate + fingertip "
                  "seated in the sockets, 0.0360 stock tips seated, 0.0497 "
                  "fat plate + fingertip flat on the bare ends, 0.0500 stock "
                  "finger faces flat on the bare ends.  See "
                  "rig_final.fatfinger_widths().  WHAT THAT ALREADY RULES "
                  "OUT: Aris_Kindt franka_control_gui.py closes with width "
                  "0.0432 and epsilon_inner 0.0, and libfranka calls a grasp "
                  "successful only above width - epsilon_inner, so 43.2 mm is "
                  "a lower bound IF the grasp succeeds — and only the two "
                  "grips on the post's BARE ends clear it.  The earlier guess "
                  "here, 'Fat fingers flat on the post ends at ~50 mm', is "
                  "ARITHMETICALLY RULED OUT: the Fat finger's contact plate "
                  "sits 10.6502 mm outboard of the stock grip plane, so that "
                  "grasp reports 0.0287, not 0.0500.  Weakened, honestly: the "
                  "GUI's menu path falls back to a move-close on failure, so "
                  "a failing grasp still holds the pen and 0.0432 is evidence "
                  "rather than proof.",
        blocking="pen-tip calibration"),
    "cable_dress": dict(
        what="How is the arm cabling actually dressed?",
        why="Nothing has been measured.  The 20-40 mm conduit in this model "
            "is the collision audit's estimate.",
        rides_on="whether a dressed forearm still clears a neighbour at the "
                 "0.61 m transverse pair spacing.",
        answer_by="photograph and measure one dressed arm",
        blocking="nothing yet — the loops are visual-only and off in "
                 "collision"),
    "cage_legs": dict(
        what="How many legs does a 4.01 m frame need, and where?",
        why="Correcting the ceiling datum turns the cage from something "
            "hanging off a room ceiling into something standing on the "
            "floor. The re-issued cut list has no legs in it because it "
            "assumed the former.",
        rides_on="structure, and whether a mid-span leg lands in a certified "
                 "flight path.",
        answer_by="a structural check of the 1210.2 mm unsupported spans",
        blocking="fabrication"),
    "base_cable_passthrough": dict(
        what="The plate and the clamp stack both need a hole in them for the "
             "arm's own base cable.  How big, and where?",
        why="THE MODEL FOUND THIS, not a person.  The manufacturer's link0 "
            "visual includes the connector and cable stub at the base, and it "
            "reaches 230.7 mm below the base flange (677 of 40 142 vertices "
            "past z = -5 mm).  Every arm here is INVERTED, so that 230.7 mm "
            "points straight UP: through the 12.7 mm plate, through the "
            "95.7 mm clamp stack, and 122 mm on into the drop cluster's slot. "
            "The collision shell stops dead at the flange (z = 0), so no "
            "clearance check in this repo has ever seen it.",
        rides_on="whether the arm can be bolted down at all.  The sheet "
                 "already routes cabling up the 240.5 mm slot between the "
                 "post pairs, and vertically there is room — the posts run "
                 "from 905.0 to 1623.6 and the cable tops out at 1170.7 — but "
                 "the plate and the clamp stack are solid across it today.",
        answer_by="measure the connector envelope on a real FR3 base and cut "
                  "the plate and clamp stack around it",
        blocking="fabrication of the plate and the clamp stack"),
    "gusset_attachment": dict(
        what="How do the rotated gussets actually bolt up?",
        why="The rotation onto the runway's y faces resolves a hard "
            "interference, but the drawing has no part number for a gusset "
            "and the attachment is an envelope, not a detail.",
        rides_on=f"{GUSSET_PAIR_CLEAR} mm of clearance inside a transverse "
                 "pair.",
        answer_by="the fabricator's own bracket selection",
        blocking="fabrication"),
}


def provenance_mix(h=None):
    """{class: (count, percent)} over every static body."""
    bs = bodies(h)
    counts = {c: 0 for c in PROVENANCE_CLASSES}
    for b in bs:
        counts[b.provenance] += 1
    n = len(bs)
    return {c: (counts[c], round(100.0 * counts[c] / n, 1))
            for c in PROVENANCE_CLASSES}


# ---------------------------------------------------------------------------
# 7.  FINDING THE ASSET
# ---------------------------------------------------------------------------
ASSET_DIR = Path(__file__).resolve().parents[1] / "assets/system_model"
COLLISION_VARIANTS = ("mesh", "capsule")
FINGER_VARIANTS = ("stock", "fat")


def urdf_path(collision="mesh", with_arms=True, fingers="stock"):
    """The generated URDF -> Path.  So no consumer hardcodes the layout.

    `collision` picks which arm collision model the file carries:
      "mesh"     the manufacturer's own collision shells
      "capsule"  the audited capsule set (`selfcoll.BODY_CAPSULES`)
    `fingers` picks which finger is on the hand:
      "stock"    the manufacturer's `finger.gltf`, at gen's FINGER_FIX
      "fat"      the printed 90 mm "Fat Franka Finger" (docs 7d) — a VARIANT,
                 and only with the mesh collision model
    `with_arms=False` gives the static scene alone — cage, table, canvas.
    """
    if collision not in COLLISION_VARIANTS:
        raise ValueError(f"collision must be one of {COLLISION_VARIANTS}, "
                         f"not {collision!r}")
    if fingers not in FINGER_VARIANTS:
        raise ValueError(f"fingers must be one of {FINGER_VARIANTS}, "
                         f"not {fingers!r}")
    if not with_arms:
        return ASSET_DIR / "environment.urdf"
    if fingers == "fat":
        if collision != "mesh":
            raise ValueError("the fat finger is only built against the mesh "
                             "collision model — selfcoll.BODY_CAPSULES has no "
                             "finger row for it to replace")
        return ASSET_DIR / "installation_fatfingers.urdf"
    return ASSET_DIR / ("installation.urdf" if collision == "mesh"
                        else "installation_capsules.urdf")


def manifest_path():
    """The machine-readable provenance record -> Path."""
    return ASSET_DIR / "model_manifest.json"


def report(h=None):
    """Print the model's own numbers — the sheet, in text."""
    h = H_MOUNT if h is None else float(h)
    z = z_ladder(h)
    p = print
    p("=" * 76)
    p("ARIS SYSTEM MODEL — dimensions (mm above the paper top unless noted)")
    p("=" * 76)
    p(f"canvas            {CANVAS_W:.1f} x {CANVAS_L:.2f}")
    p(f"cage frame        {FR_W:.1f} x {FR_L:.2f}  "
      f"({FR_W / IN:.2f} x {FR_L / IN:.2f} in), margin {MARGIN} all round")
    p(f"profile           {PROFILE} sq T-slot")
    p("")
    p("Z LADDER (corrected datum)")
    for k in ("floor", "table_top", "paper_top", "leg_bottom", "post_bottom",
              "mount_plane", "plate_top", "clamp_top", "gusset_bottom",
              "grid_underside", "grid_top"):
        p(f"    {k:16s} {z[k]:10.2f}")
    p(f"    {'post length':16s} {z['post_length']:10.2f}"
      f"   (original {O_POST_L}, code datum {POST_L_CODE})")
    p(f"    cage total height above the floor  {CAGE_TOTAL_H}"
      f"  == the drawing's 233,7 cm")
    p("")
    p("PER ARM")
    for aid in sorted(ARM_XY, key=lambda a: (ARM_XY[a][1], ARM_XY[a][0])):
        xa, ya = ARM_XY[aid]
        a, b = post_x(xa)
        p(f"    arm {aid:>2}  axis ({xa:7.1f}, {ya:8.2f})  "
          f"plate centre x {plate_centre_x(xa):7.2f}  posts x {a:7.2f}/{b:7.2f}")
    p("")
    p("PROVENANCE")
    for c, (n, pc) in provenance_mix(h).items():
        p(f"    {c:9s} {n:3d}  {pc:5.1f} %")
    p("")
    p("RECONCILIATION WITH mounts.py  (nothing here changes it)")
    for r in reconciliation(h):
        p(f"    {r['item']:28s} delta {r['delta_mm']:9.2f} mm")
        p(f"        {r['direction']}")
    p("=" * 76)


if __name__ == "__main__":       # pragma: no cover
    report()
