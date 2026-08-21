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
    _B("table_block", (0, 0, 0), (218.44, 208.28, 63.468),
       "encloses legs/feet/rails/braces/top plate (solids, front copy)"),
    # cage corner posts, tabletop to the very top
    _B("post_FL", (0, 0, 60.93), (7.62, 7.62, 233.65), "7.62 sq posts"),
    _B("post_FR", (210.82, 0, 60.93), (218.44, 7.62, 233.65), "7.62 sq posts"),
    _B("post_BL", (0, 200.66, 60.93), (7.62, 208.28, 233.65), "7.62 sq posts"),
    _B("post_BR", (210.82, 200.66, 60.93), (218.44, 208.28, 233.65),
       "7.62 sq posts"),
    # the whole top band as one slab: perimeter beams + central double beam
    # (Y 144.831-160.071) + 4 connector plates (z 229.84-233.65)
    _B("top_slab", (0, 0, 226.03), (218.44, 208.28, 233.65),
       "encloses all top beams + connector plates"),
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
    _B("down_boom_W", (23.53, 144.83, 152.37), (31.15, 160.07, 226.03),
       "west vertical beam pair straddling the central beam", "mount:down"),
    _B("down_boom_E", (55.28, 144.83, 152.37), (62.90, 160.07, 226.03),
       "east vertical beam pair straddling the central beam", "mount:down"),
    _B("down_gusset_W", (3.21, 144.83, 205.71), (23.53, 160.07, 226.03),
       "vertical gusset 20.32x3.81x20.32 (Y band taken conservative)",
       "mount:down"),
    _B("down_gusset_E", (62.88, 144.83, 205.71), (83.22, 160.07, 226.03),
       "vertical gusset 20.32x3.81x20.32 (Y band taken conservative)",
       "mount:down"),
    _B("down_clamps", (31.93, 144.83, 157.14), (54.53, 160.07, 166.71),
       "clamp stack above the plate (two levels, union)", "mount:down"),
    _B("down_plate", (31.931, 142.926, 155.868), (54.513, 161.926, 157.138),
       "base plate 22.582x19.0x1.27", "mount:down"),
    # --- arm-2-style boom (the "side" arm) ---------------------------------
    _B("side_boom", (187.31, 144.83, 152.37), (194.93, 160.07, 226.07),
       "vertical beam pair; bottom z is the UNION of the two DXF copies "
       "(155.07 front / 152.37 top copy) - FLAGGED", "mount:side"),
    _B("side_gusset", (194.95, 144.83, 205.71), (215.27, 160.07, 226.03),
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
    _B("feed_roll", (0.8, 23.5, 62.5), (21.35, 200.0, 83.1),
       "feed roll ~O20.2, axis X~11.1 z~72.8 (curved solid, point-sampled; "
       "padded outward, +X face kept at the measured 21.2+0.15) + brackets"),
    _B("guide_rods", (211.6, 32.0, 63.4), (215.6, 192.1, 66.8),
       "3 thin guide rods X~212.1/213.6/215.1, z~64.7-66.3, padded 0.5"),
    _B("winder_box", (209.57, 197.61, 63.47), (216.57, 200.61, 68.47),
       "winder box 7x3x5"),
    _B("crank_motor", (209.38, 9.38, 63.47), (216.38, 23.82, 71.34),
       "crank/motor block"),
    # --- 8 hung devices (cameras or lights - purpose not stated) -----------
    _B("device_F1", (61.0, -1.0, 216.8), (85.2, 8.62, 226.58),
       "body 10.8x7.9x6.9 z 219.72-226.58 + plate + protrusions to 216.8"),
    _B("device_F2", (133.4, -1.0, 216.8), (157.6, 8.62, 226.58), "as F1"),
    _B("device_B1", (61.0, 199.66, 216.8), (85.2, 209.28, 226.58), "as F1"),
    _B("device_B2", (133.4, 199.66, 216.8), (157.6, 209.28, 226.58), "as F1"),
    _B("device_L1", (-1.0, 27.9, 216.8), (8.62, 52.1, 226.58), "as F1"),
    _B("device_L2", (-1.0, 100.3, 216.8), (8.62, 124.5, 226.58), "as F1"),
    _B("device_R1", (209.82, 27.9, 216.8), (219.44, 52.1, 226.58), "as F1"),
    _B("device_R2", (209.82, 100.3, 216.8), (219.44, 124.5, 226.58), "as F1"),
]


def frame_boxes_canvas(exclude_tag=None, zmin=0.0):
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
    """
    o = np.array(PAPER_ORIGIN_W_CM)
    out = []
    for b in FRAME_BOXES_W_CM:
        if exclude_tag is not None and b["tag"] == exclude_tag:
            continue
        lo = (np.array(b["lo"]) - o) / 100.0
        hi = (np.array(b["hi"]) - o) / 100.0
        if hi[2] < zmin:
            continue
        out.append(dict(name=b["name"], lo=lo, hi=hi,
                        source=b["source"], tag=b["tag"]))
    return out


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
