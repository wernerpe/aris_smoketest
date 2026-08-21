# FINAL RIG — the 3-arm installation, extracted from the authoritative drawing

Everything in this document is traceable to the two files the user named as THE
final setup (`raw_slack_file_dump/`):

- **PDF** `Drawing installation, 3 arms, 1 up, 1 side, 1 down, sizes cm.pdf` —
  1 page, Vectorworks export (author "Stefan Strauss"), front view + top view +
  text block. Dimensions in **cm**.
- **DXF** `D.I., 3 arms, 1 up, 1 side, 1 down, sizes cm.dxf` — AC1032,
  `$INSUNITS=5` (centimeters, verified against the 218,4 dimension),
  `$DIMLFAC=1`. The modelspace holds **two complete 3D copies** of the
  installation (a front copy and a rotated top copy); the front copy is the
  authoritative geometry, the top copy cross-checks it (agreement < 0.02 cm
  everywhere except one flagged item, see Flags #1).

Extraction: visual PDF read page-by-page + programmatic DXF parse (ezdxf;
layers `0` / `Design Layer-3` / `Defpoints`; 166 top-level 3DSOLID + 87 block
inserts with 175 more solids, 11 MTEXT, 11 DIMENSION — none text-overridden,
so displayed values ARE measured values rounded to one decimal). All 11
dimension annotations reconcile with the model geometry to ≤ 0.05 cm.

The machine-readable form of every number below is
**`aris_sixarm/rig_final.py`** — the single source the URDF generator, the
fleet registry and the collision model all read. This file is the provenance
record; that file is the data.

## The arms are named in the drawing

MTEXT: *"Three robot arms: 13 floor, 31 left - upside down, 2 right side
position"* — the same physical arms as the legacy six-arm registry. The final
fleet therefore reuses ids **13** (upright), **31** (inverted), **2** (side).
Notably, the legacy registry always listed arm 2 as a wall-plate mount ("
revisit if arm 2 goes active as a wall arm" — it has).

## Frames

- **W** ("world", the drawing): origin = outer front-left corner of the
  installation at floor level (underside of the leveling feet), +X right,
  +Y back, +Z up. Units cm. Envelope: X 0→218.44, Y 0→208.28, Z 0→233.65.
- **C** ("canvas", the planning frame): origin = front-left corner of the
  **paper top surface**, axes parallel to W, units m, z=0 the paper plane —
  the identical convention every planner module already used.
  `C = (W − (21.246, 26.748, 63.668)) / 100`.

## Dimension inventory (PDF annotation ↔ DXF measurement)

| Annotation | DXF measured | What it measures (W frame) |
|---|---|---|
| 218,4 (both views) | 218.440 | outer width, X 0→218.44 |
| 208,3 ("entire table") | 208.280 | outer depth, Y 0→208.28 |
| 233,7 height construction | 233.700 (dim) / 233.648 (structure) | floor→top beams; the dim's lower defpoint sits 0.052 below the floor |
| 63,5 table height | 63.468 | floor→tabletop top |
| 91,6 "backside base plate to surface (drawing paper)" | 91.600 | arm-2 plate TOP edge (z=155.068) → **tabletop** (63.468); to the paper top it is 91.40 — see Flags #3 |
| 45,7 | 45.737 | arm-31 J1 axis → left outer face |
| 55,8 (arm 31) | 55.854 | arm-31 J1 axis → back outer face |
| 55,8 (top view / arm 2) | 55.829/55.809 | back outer face → arm-2 boom centerline / plate center Y |
| 34,9 (both views) | 34.940 | arm-2 plate backside (x=183.500) → right outer face |
| 22,6 × 19,0 | 22.582 × 19.000 × 1.27 thick | all three base plates |
| 13,8 × 9,5 | 13.806 / 9.500 | J1 axis offset from plate edges (9.5 = centered on the 19 side; 13.8/8.78 = NOT centered on the 22.6 side) |
| 3″×3″ = 7,62 | 7.620 | every beam / post / leg cross-section |

Other exact plate thicknesses: tabletop 2.54 (1″), gusset/brace plates 3.81
(1.5″), gussets 20.32×20.32 (8″), device plates 24.13 (9.5″).

## Drawing surface

A continuous **paper web** feeding from a roll at the left edge, lying flat on
the tabletop, curling up at the right edge to guide rods and a winder:

- flat drawable region **180.34 × 170.00 cm**, W: X 21.246→201.586,
  Y 26.748→196.748, top face **z = 63.668** (paper drawn 0.2 thick on the
  63.468 tabletop; the 2 mm is modeler convenience — the top face is the datum)
- canvas frame: sheet `SHEET_FINAL = (1.8034, 1.700) m`, z = 0, paper center
  C (0.9017, 0.8500, 0), W (111.42, 111.75, 63.668) cm
- provenance: paper solid 40 (front copy) exact; roll/curl solids 41–47

## The three arm mounts

All three base plates are 22.582 × 19.0 × 1.27 cm; the J1 axis sits 13.8 cm
from one 19-edge (8.78 from the other) and centered (9.5/9.5) on the 22.6
sides. Base "front" (+X of link0) = the 8.78-offset side — an inference from
the Franka base asymmetry plus all three depicted poses (Flags #6).

| arm | mount | J1 axis, W cm | mounting plane | base orientation (quat xyzw, W) | canvas pose, m |
|---|---|---|---|---|---|
| **13** | upright on the tabletop, front strip, in front of the paper | (109.289, 14.068) | z = 64.738 (plate top), axis +Z | (0, 0, 0.70711, 0.70711) = rotz(π/2); front faces +Y, toward the paper | (0.88043, −0.12680, 0.01070) |
| **31** | inverted, hanging under the central double top beam | (45.737, 152.426) | z = 155.868 (plate BOTTOM), axis −Z | (1, 0, 0, 0) = rotx(π); front faces +X, toward table center | (0.24491, 1.25678, 0.92200) |
| **2** | side: vertical plate clamped to the right boom, J1 axis horizontal | Y 152.471, z 141.268 | x = 182.230 (plate left face), axis −X | (0.70711, 0, −0.70711, 0) = roty(−π/2)·rotz(π); front faces −Z, straight down | (1.60984, 1.25723, 0.77600) |

Derived quantities the planner cares about:

- arm 31 mounting plane is **92.20 cm above the paper** (92.40 above the
  tabletop) — within 2 mm of the legacy rig's measured `h_inv = 0.924`.
- arm 2 J1 axis is **77.60 cm above the paper**, pointing at the table center.
- arm 13's plate top is 1.07 cm above the paper plane; the paper's front edge
  is 12.68 cm in +Y from its J1 axis.
- arms 31 and 2 both hang on the central-double-beam centerline Y = 152.45
  (axes at Y 152.426 / 152.471, within 0.03).

Provenance per arm: arm 31 — dims 45,7 + 55,8, plate solid 83, axis marker
solid 86, ring solid Groep-57. Arm 2 — dims 34,9 (×2), 55,8, 91,6, plate solid
Groep-72/143. Arm 13 — **no dimension annotation references it**; position is
model geometry only (plate solid Groep-66/137 + 13.8 marker, identical in both
DXF copies) — Flags #5.

## Structure / collision model

`rig_final.FRAME_BOXES_W_CM` — 34 conservative AABBs that provably enclose
every drawn member. Key modeling choices:

- **table_block**: everything below the tabletop (legs, feet, top+bottom rail
  levels, 45° corner braces, tabletop plate) as ONE enclosing box
  (218.44 × 208.28 × 63.468). Planning drops it (`zmin=0.0`): the
  z ≥ `Z_PAPER` plane gate already forbids the whole below-paper half-space,
  and a slab 2 mm under the pen would veto every legitimate drawing pose.
- **top_slab**: all top beams (perimeter + central double beam at
  Y 144.83–160.07) + connector plates as one slab z 226.03→233.65
  (canvas z 1.624→1.700 — above any reachable pose, kept anyway).
- **arm-31 boom**: two vertical beam-pair boxes flanking the plate
  (X 23.53–31.15 and 55.28–62.90, canvas x 0.023–0.099 / 0.340–0.417,
  z down to 152.37 = 3.5 cm BELOW the mounting plane) + gussets + clamp
  stack. These replace the legacy `BOOM_R=0.12` cylinder proxy for this arm.
- **arm-2 boom**: vertical beam pair X 187.31–194.93 (canvas x 1.661–1.737,
  directly behind the mounting plate) + clamp blocks straddling the axis
  height + gussets. Bottom z taken as the **union** of the two DXF copies
  (Flags #1).
- **paper transport**: the feed roll (Ø≈20.2, canvas x −0.205→+0.001,
  z −0.012→0.194) sits **flush against the paper's left edge** — it, not
  reach, will bound the drawable strip there. Guide rods / winder / crank at
  the right edge stay ≥ 0.095 m outside the sheet.
- **8 hung devices** (cameras or lights — purpose not stated) under the top
  beams, canvas z ≥ 1.53.

Every box carries its source string in `rig_final.py`. Below-paper members
are enclosed but excluded from planning; everything above the paper plane is
in exactly one box (verified independently, see Verification).

## Pen holder / tool

*(pending — CAD extraction in progress; this section is filled by the
pen-holder campaign step. The legacy gate-validated tool chain — hand TCP =
flange + 0.1034, `PEN_EXT` = 0.110 below TCP — stays untouched as the
`sixarm` config; the final-rig tool is a NEW config alongside it.)*

## Verification

*(pending — independent re-derivation agent diffing the drawings against
`rig_final.py`; table lands here.)*

## Flags / ambiguities / risks

1. **Arm-2 boom bottom disagrees between the two DXF model copies**: front
   copy z 155.07→226.07, top copy z 152.37→226.03. Collision model uses the
   union (152.37→226.07). Ask which is as-built.
2. **The drawing itself warns about arm 2's mount**: MTEXT *"Arm 2 side
   position: conection base plate to boom pole is not stiff and stable."*
   A compliant mount means the surveyed base pose may not hold under load —
   the calibration margin term for arm 2 should NOT be reduced below 30 mm
   until the mount is stiffened or surveyed under load. Surface this to the
   fabricator.
3. The 91,6 annotation says "to surface (drawing paper)" but geometrically
   measures to the **tabletop** (91.40 to the paper top). We anchor arm 2 by
   the plate solids, not this dim, so nothing depends on the reading — but
   the wording should be clarified with the author.
4. Rounding: "63,5" is exactly 63.468; "233,7" structure is 233.648; "22,6"
   plate is 22.582. The model geometry (3 decimals) wins over the rounded
   annotations everywhere.
5. **Arm 13 has no dimension chain** — its pose rests on the plate solid
   alone (both copies agree exactly). It also has **no fasteners drawn**
   (arms 31/2 have clamp blocks; 13's plate just rests on the tabletop).
   Confirm how it will be fixed and whether its position is final.
6. Base **yaw convention**: "front = the 8.78-offset plate side" is inferred
   from the base asymmetry + the three depicted (decorative) poses. A 180°
   yaw error per arm is possible until checked against a real plate. The
   depicted poses all lean toward the inferred front, consistently.
7. Curved bodies (leveling feet, paper roll, decorative robot shells) have
   point-sampled extents; all structural box members are exact. The roll box
   is padded outward and flagged as approximate.
8. The DXF text offers an option: *"possibly 2 tables 208,3 × 109,2"* (split
   along X). Not modeled; ask if the split is happening.
9. Arm-31 axis X has three sources spanning 45.720–45.741 (marker / dim /
   ring); the dim value 45.737 is used. Arm-2 axis z: 141.268 (13.8 rule) vs
   141.279 (ring solid), 0.011 apart — 13.8 rule used.
10. The two DXF copies were cross-checked wherever both draw the same member;
    agreement < 0.02 cm except Flags #1.

## Open questions for the user

- Which reading of the arm-2 boom bottom is as-built (Flags #1)?
- Arm 13 fastening + final position (Flags #5)?
- Confirm each base plate's yaw (Flags #6) — a photo of each plate suffices.
- Is the two-table split (Flags #8) happening?
- What are the 8 hung devices (cameras/lights)? They are modeled as obstacles
  only.
