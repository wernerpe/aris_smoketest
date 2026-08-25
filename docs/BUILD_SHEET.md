# Build sheet — all-ceiling rig (6 inverted FR3, 2×3 grid)

Adopted 2026-08-25 (layout `LAYOUT_PROPOSED`, certified 99.98 % coverage at
commit 820dd3b; URDF `assets/proposed_rig/installation.urdf`, commit 58a7674).
All dimensions in **mm** unless noted. This sheet is the single source for
the fabricator; the software mirror is `aris_sixarm/layout.py`.

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

## 1. Mounting height — the one number that matters most

**The surface each robot bolts against (the underside of its mounting plate)
is at z = +850.0 mm above the paper surface. All six identical.**

The arms hang fully upside down below their plates. 850 is a certified
optimum, not a round-number suggestion — build to **±10 mm** and keep all six
mutually coplanar.

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
limits make clocking matter).

## 4. Structure keep-out envelope

What the certification modelled — steel must stay inside it:

- **Below the mounting plane (z < 850): nothing but the six arms.** The whole
  volume between paper and plates, over the canvas plus 1 m margin around it,
  stays empty. No braces, no cable drops, no lights.
- **Mounting plates**: modelled 226 (x) × 190 (y) × 50 thick, occupying
  z 850→900, centred on each base axis. Bigger/thicker plates → send
  dimensions for re-certification before fabricating.
- **Vertical supports (booms)**: from each plate up to the ceiling grid,
  everything within a **radius-100 column centred on the base axis**,
  z 900→2340. Route arm cabling up inside/along this column.
- **Ceiling grid**: cross-members at **z ≥ 2340** are assumed but NOT yet
  modelled — send the steel design (member sections + routing) before final
  fabrication and we re-certify (fast: minutes).

Any deviation from this envelope is fine *if declared first* — re-checking a
proposed steel design against all certified poses is cheap; discovering a
brace with a moving arm is not.

## 5. Levelness, calibration, what absorbs error

- Aim: plates level and mutually coplanar within ±3 mm / ≤0.3° tilt.
- The planner carries an 80 mm inter-arm safety margin (30 mm of it is
  calibration allowance), and commissioning includes a pen touch-off
  calibration per arm that absorbs residual height/level error.
- So: ±10 mm build accuracy is comfortable; just **record as-built numbers**
  if anything lands outside tolerance.

## 6. Plan view (not to scale)

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
         fronts facing the x=0 edge)
```

## 7. Open items (not builder-blocking, tracked)

- Ceiling grid steel design → re-certification (§4).
- Pen holder fingertip-cradle geometry (tool offset verification) — software
  side, does not affect this sheet.
- On-site base survey / touch-off calibration at commissioning.
