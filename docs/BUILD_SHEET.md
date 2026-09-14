# Build sheet — all-ceiling rig (6 inverted FR3, 2×3 grid)

**Re-issued 2026-09-10 at h = 970.0 mm and the CORRECTED datum. This sheet
supersedes the 2026-08-25 issue in two places** — the mounting height (was
850.0, then 940.0) and the ceiling datum (was a paper-referenced 2340, which
was a provenance bug). If you are holding a printout that says 850 or 2340,
throw it away.

The one number that changed most: **the drop posts are 688.6 mm, not 1435.0
and not 1525.0.** Those lengths are 716.4 mm too long and must not be cut.

Adopted 2026-09-10 (Pete: *"do you have the drawings for the 970 one? let's
just work with that one."*). Layout `layout.LAYOUT_PROPOSED`; certified
hole-free block 1.50 × 3.64 m at this height (`out/certified_area_h0970.json`).
All dimensions in **mm** unless noted. This sheet is the single source for the
fabricator; the software mirror is `aris_sixarm/layout.py` and
`aris_sixarm/system_model.py`, and the drawings that go with it are
`out/drawings/` (`scripts/draw_8020.py --h 0.970`) — plan, elevation and cut
list, generated from those same modules and agreeing with every number here.

## 0. Datum — set out first

Mark a rectangle **1803.4 × 3630.6 mm** on the table: this is the canvas
(drawing area). Everything below is measured from it:

- Origin `(0,0)` = one corner of that rectangle (pick it and mark it — it is
  the reference corner for every number here).
- **x** runs across the short side (0 → 1803.4), **y** along the long side
  (0 → 3630.6), **z** up, `z = 0` at the **top surface of the paper** as laid
  on the table.
- Arm positions are measured to the **vertical centreline of the robot base**
  (the joint-1 axis = centre of the base bolt circle), NOT to a plate edge.

**THE DATUM IS THE PAPER, AND THE CAGE IS FLOOR-STANDING.** The original
drawing's 233,7 cm is FLOOR to top-of-construction, not paper to ceiling, and
the paper sits 636.68 mm above that floor. There is no surveyed room ceiling
anywhere in this project; the cage supports itself and stands on the floor.
The whole z ladder at this height:

| level | mm above the paper |
|---|---:|
| grid top (top of steel) | 1699.82 |
| grid underside (what a drop post hangs from) | 1623.62 |
| gusset bottom | 1496.62 |
| clamp top | 1078.40 |
| plate top | 982.70 |
| **mount plane (plate underside) = h** | **970.00** |
| post bottom | 935.02 |
| paper top | **0.00** |
| table top | −2.00 |
| leg bottom | −27.38 |
| floor | −636.68 |

Cage height floor to top of steel: **2336.5 mm** — which is the original
drawing's own 233,7 cm, read the way it was written.

## 1. Mounting height — the one number that matters most

**The surface each robot bolts against (the underside of its mounting plate)
is at z = +970.0 mm above the paper surface. All six identical.**

The arms hang fully upside down below their plates. Build to **±10 mm** and
keep all six mutually coplanar within ±3 mm.

**WHY 970 AND NOT 940 OR 850.** 970 is not the coverage optimum — it is
0.6 pp *worse* than 940 on solo-drawable area (97.718 % against 98.587 %) and
has more unreachable cells at the rim. It is here because it is the only
height tested at which the canvas has **no enclosed dead cells at all**: every
dead cell is out-of-reach rim, where the red is honest, and there is no pocket
in the middle of the paper that an arm can neither hover over nor route to.
940 still has 9 such cells in 4 pockets. What that buys the drawing is the
certified block below.

**THE CERTIFIED DRAWING AREA AT THIS HEIGHT is 1500 × 3640 mm** (x 160…1640,
y 0…3620), **5.460 m²** — the full length of the canvas, stopping 160 mm short
of each long edge against the arms' reach. At 940 the same block is
1060 × 3640 = 3.86 m². It is drawn dashed on the plan. **It is not the whole
canvas, and artwork is placed inside it.**

## 2. Base positions (to the joint-1 axis)

Two columns × three rows, symmetric about both centrelines of the canvas:

| arm id | x (mm) | y (mm) |
|-------:|-------:|-------:|
| 13 | 596.7 | 605.1 |
| 17 | 1206.7 | 605.1 |
| 31 | 596.7 | 1815.3 |
| 71 | 1206.7 | 1815.3 |
| 2 | 596.7 | 3025.5 |
| 97 | 1206.7 | 3025.5 |

Tape-measure cross-checks (all must agree):
- Columns sit at the canvas centreline (901.7 from either long edge) ± 305.0
  → column spacing **610.0**.
- Rows sit at 1/6, 1/2, 5/6 of the length → row spacing **1210.2**; first and
  last rows are each 605.1 from their short edge.
- Position tolerance **±10 mm** per base. If a base ends up outside that,
  measure and report the as-built offset — do not silently re-centre others.

Column spacing may be anywhere in **500–650** if the steel prefers it
(coverage-neutral window), **but only after telling the planning side** —
610.0 is what is certified and programmed today.

## 3. Orientation — identical for all six, load-bearing

Every arm is mounted upside down in the **same** orientation: the flipped
base's **+x axis (the arm's "front") points in the canvas −x direction**,
i.e. toward the long edge that contains the reference corner. Formally:
R_world_base = Ry(180°) — the base frame is the upright factory frame rotated
half a turn about the canvas-y axis. No arm is clocked differently from the
others.

Acceptance check after bolting: command all joints to 0° — the folded arm
must lean out toward the x = 0 long edge. All six must lean the **same way**.
(Rule of thumb: the base connector panel faces away from the arm's front, so
all connector panels face the x = 1803.4 long edge.)

Do not improvise per-arm clocking "to point at the middle": the certified
coverage and every program assume this exact uniform orientation (joint-1
limits make clocking matter). A **mirrored** clocking — the two columns turned
to face each other — is an open question being measured, not a licence to
improvise; if it is adopted this section will be re-issued and the plate
offset in §7 flips with it.

## 4. Structure keep-out envelope

What the certification modelled — steel must stay inside it:

- **Below the mounting plane (z < 970): nothing but the six arms.** The whole
  volume between paper and plates, over the canvas plus 1 m margin around it,
  stays empty. No braces, no cable drops, no lights.
- **Mounting plates**: modelled 226 (x) × 190 (y) × 50 thick, occupying
  z 970→1020, centred on each base axis. Bigger/thicker plates → send
  dimensions for re-certification before fabricating.
- **Vertical supports (booms)**: from each plate up to the grid, everything
  within a **radius-100 column centred on the base axis**. Route arm cabling
  up inside/along this column.
- **Grid members**: the runway beams' underside is at **1623.62** and the top
  of steel at **1699.82**. The grid's own cross-members are assumed but NOT
  yet modelled — send the steel design (member sections + routing) before
  final fabrication and we re-certify (fast: minutes).

**THE MODELLED ENVELOPE IS NOT WHAT THE REAL STEEL DOES, AND THAT IS QUEUED.**
The certified obstacle is a single radius-100 column on the base axis; the
drawing's real drop cluster is a 393.8 × 152.4 mm 2 × 2 post group offset
25.15 mm off that axis, plus 203.2 mm gussets. Neither contains the other:
all 60 pieces of mount hardware fall outside the certified envelope, worst
**185.55 mm**. This blocks final fabrication of the clusters and gussets and
nothing else — see `system_model.reconciliation()` and open item 6 in
`out/drawings/8020_cut_list.md`.

Any deviation from this envelope is fine *if declared first* — re-checking a
proposed steel design against all certified poses is cheap; discovering a
brace with a moving arm is not.

## 5. Levelness, calibration, what absorbs error

- Aim: plates level and mutually coplanar within ±3 mm / ≤0.3° tilt.
- The planner carries a **50 mm arm-to-arm safety margin** (30 mm of it is
  calibration allowance — a real uncertainty about where the bases are, and
  the reason a base survey is worth doing), and commissioning includes a pen
  touch-off calibration per arm that absorbs residual height/level error.
- So: ±10 mm build accuracy is comfortable; just **record as-built numbers**
  if anything lands outside tolerance. `scripts/asbuilt_layout.py` is what
  reads that report back in.

## 6. Steel — cut lengths at this height

The full cut list, with quantities, sections and the open items that block
each one, is **`out/drawings/8020_cut_list.md`**, generated at h = 0.970 by
`scripts/draw_8020.py`. The height-dependent cut is the drop post, and it is
the only one:

```
drop post cut length = (grid underside − h) + post over-run
                     = (1623.62 − 970) + 34.98 = 688.6 mm     (24 off)
```

`34.98` is the drawing's own over-run of the post past the plate underside,
not a round 35. Cut all 24 to one length.

| member | qty | section | cut |
|---|---:|---|---:|
| A perimeter side rail | 2 | 3″ × 3″ T-slot | 4011.64 |
| B perimeter end rail | 2 | 3″ × 3″ T-slot | 2032.00 |
| C runway beam | 6 | 3″ × 3″ T-slot | 2032.00 |
| **D drop post** | **24** | 3″ × 3″ T-slot | **688.60** |
| E corner leg | 4 | 3″ × 3″ T-slot | 1651.00 |
| F top gusset | 24 | 8″ × 8″ × 1.5″ gusset | 203.2 × 203.2 × 38.1 |
| G robot base plate | 6 | — | 225.82 × 190.0 × 12.7 |
| H plate clamp stack | 6 | — | 226.0 × 152.4 × 95.7 |

Total 3-in T-slot extrusion at h = 970: **47.41 m**. Frame outside
**2184.4 × 4011.64** (86.00 × 157.94 in), 190.5 mm clear of the canvas on all
four sides.

Item E is ASSUMED and **mid-span legs are almost certainly required**: a
4.01 m frame on four corner legs has no precedent (the original spans 2.08 m),
and any mid-span leg must be checked against the certified flight paths before
it is welded in.

## 7. Plan view (not to scale)

```
 y=3630.6 ┌────────────────────────────┐
          │      2 ○      ○ 97         │   ← row 3, y=3025.5
          │                            │
          │                            │
          │     31 ○      ○ 71         │   ← row 2, y=1815.3
          │                            │
          │                            │
          │     13 ○      ○ 17         │   ← row 1, y=605.1
      y=0 └────────────────────────────┘
          x=0    596.7  1206.7    x=1803.4
        (reference corner at lower-left; all arms hang from above,
         fronts facing the x=0 edge, connector panels the x=1803.4 edge)
```

The plate centre sits **25.15 mm from the J1 axis toward the x = 1803.4 edge**
— i.e. the same way the connector panels face. **THAT DIRECTION IS AN
INFERENCE, not a measurement** (the drawing gives it for an arm whose front
faces +X and every arm here is clocked the other way), and 7.79 mm rides on
it: the plate nests in a 241.4 mm slot. Measure the real plate, or open the
post pitch to 381.0 (posts at the axis ± 190.5), which fits either way with
14.4 mm each side. This is open item 1 on the cut list.

## 8. Open items (tracked; 5 of 6 block a cut)

Full text with what each one blocks and how it closes is in
`out/drawings/8020_cut_list.md`.

1. **Plate offset direction** — blocks the drop clusters (post x positions).
2. **Base cable pass-through** — the plate and the 95.7 clamp stack are solid
   across the arm's own base cable, and on an INVERTED arm the connector and
   cable stub point straight UP through both. Blocks the plate and clamp.
3. **Cage legs** — count and position for a 4.01 m frame. Blocks item E.
4. **Gusset part and attachment** — no part number on the drawing. Blocks F.
5. **Room survey** — the room has never been measured. Blocks nothing cut to
   the corrected datum, and is the reason the datum is restated on every
   sheet.
6. **Re-certification of the drop cluster** against the certified poses
   (§4). Blocks final fabrication of the clusters and gussets.

Software-side, not builder-blocking: pen-holder fingertip-cradle geometry
(tool offset verification), and on-site base survey / touch-off calibration at
commissioning.


## 9. The seam frame — CONFIRM ON HARDWARE, photo requested

**Added 2026-09-14** after Pete reported bars at the middle of the real rig
that the model did not have. The real installation is **two copies of the
original 3-arm half-cage (218.44 × 208.28 cm) butted along the paper's long
axis**, and where they meet — at the paper's mid-length, **y = 1815.32 mm**,
which is the middle arm row — each half contributes its own end frame.

**Cut list (adds to §6):**

| qty | member | section | cut length | position |
|---:|---|---|---:|---|
| 4 | seam post | 3″ × 3″ T-slot | **1651.0 mm** | x = −190.5…−114.3 and 1917.7…1993.9; y = 1739.12…1815.32 and 1815.32…1891.52; z = −27.38 (tabletop) to 1623.62 (runway underside) |
| (8) | corner brace plate | 8″ × 1.5″ × 8″ | 203.2 mm | two per post, on its two inboard faces, top flush with the runway underside — **not in the collision model**, see docs/SYSTEM_MODEL.md §3b |

The post cut is **the same 1651.0 mm as the four corner legs** — one more line
on the same cut, not a new part.

**The seam's END RAILS are already in the build sheet**: they are
`runway_r1_S` / `runway_r1_N`, the middle double-beam runway. Two butted end
rails and this sheet's middle runway agree to **0.78 mm**. Do not order them
twice.

### CONFIRM ON HARDWARE — the photo, and the five questions it answers

**Two photographs: one from inside the cage looking along the paper at the
seam, one looking down the seam from an end.**

1. **How many posts at the seam, and at what x?** This sheet builds four — two
   per half-cage, at the frame's own x corners. Is there a **mid-width** post?
   The drawing says no (its front view looks along y and shows only the two
   corner verticals above the tabletop), but the drawing is a plan and the rig
   is built.
2. **Two bars or one?** Do the two half-cages' end rails sit side by side
   (152.4 mm of steel, which is what this sheet assumes and what the middle
   runway is), or has someone removed one and bolted the halves to a single
   rail?
3. **Is the gap really zero?** `rig_final6.GAP_CM` = 0 assumes the frames butt
   outer face to outer face. The levelling-foot pads overhang the leg lines by
   3.3 mm, so two frames touching **at the pads** stand 6.6 mm apart.
4. **Any diagonal at the seam?** The drawing puts X-bracing only in the 63.5 cm
   table frame, and corner gussets only above it. A diagonal across the seam
   bay would be new steel in the band the middle row's links sweep.
5. **Do the posts stand on the tabletop or on the floor beside the table?**
   This sheet stands them on the tabletop at z = −27.38, the way the drawing's
   own `post_FL` does.

### Why it matters — the middle row

Arms **31 and 71** have their J1 axes **on the seam plane**. The west and east
seam posts stand 114.3 mm outboard of the canvas edge over the whole 1651 mm
from the tabletop to the runway — through the mount plane, and through the band
a middle-row arm's links sweep when it reaches its own x extreme. Measured
(`scripts/seam_impact.py`, docs/DECISIONS.md): the shipped `Q_PARK_PROPOSED`
clears by **0.9 mm** against the 50 mm static margin, two staged parks are
**35.3 mm inside the steel**, and the v19 programme reads **−59.4 mm** and
fails. The certified area loses 165 cells of 16 184 and the 1.50 × 3.64 m
hole-free block is untouched.
