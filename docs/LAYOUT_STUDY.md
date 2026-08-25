# Layout study — six arms on the merged canvas, lateral pen

**v2 (2026-08-25) — MOUNTING HARDWARE IS MODELLED.**  v1 of this study (below,
§7) modelled NO steel: neighbouring booms and base plates were not obstacles,
so its 99.38 % winner was an optimistic kinematic ceiling.  The user caught
that gap.  v2 makes the schematic mounts first-class static obstacles at every
stage, and — on a second user mandate — studies TWO FAMILIES:

| family | | |
|---|---|---|
| **2 + 4** | two floor arms + four ceiling-inverted | v1's configuration |
| **0 + 6** | all six ceiling-inverted | uniform hardware, one grid |

## 0. The answer

**Hang all six.  Set them out on a regular 2 x 3 ceiling grid at h = 0.85 m.**

| | canvas x (m) | canvas y (m) | base z (m) |
|---|---:|---:|---:|
| pair S | **0.5967** / **1.2067** | **0.6051** | 0.850 |
| pair M | **0.5967** / **1.2067** | **1.8153** | 0.850 |
| pair N | **0.5967** / **1.2067** | **3.0255** | 0.850 |

Canvas frame: x across the 1.8034 m width, y along the 3.63064 m length,
z = 0 the paper.  The figure is derived from the canvas, not copied off a
search: the two columns straddle the centre line at the **pair spacing
0.61 m**, the three rows sit at **H/6, H/2, 5H/6**.
`layout.paired_grid(spacing=0.61, rows=3, h=0.850)` is exactly this, and it
is `LAYOUT_PROPOSED` / `ARIS_RIG=proposed`.

**Certified: 99.98 % union strict-GO** (2 cm grid, 15-degree tilt cone,
lateral pen, **mount obstacles active**), **55.98 % >= 2 arms**, 13.95 %
>= 3 arms, 5.39 % >= 4 arms.  **3 dead cells** out of 16 562 — three isolated
single cells at canvas corners.  All six certified ready poses clear every
other arm's steel by **>= 0.35 m**.

## 1. How optimistic was 99.38 %?

Barely, on union — and materially, on redundancy.  v1's winner re-scored on
the same 2 cm certified sweep, mounts off and on:

| v1 winner (2 + 4, h = 0.85) | union | >= 2 | >= 3 | dead cells |
|---|---:|---:|---:|---:|
| green field (the v1 number) | 99.384 % | 50.31 % | 4.70 % | 102 |
| **mounts active (re-scored)** | **99.281 %** | **48.04 %** | 4.70 % | 119 |
| cost of the steel | **−0.10 pp** | **−2.27 pp** | 0.00 | +17 |

Per arm, the whole cost falls on the two FLOOR arms, and the four inverted
arms lose **exactly nothing** — not approximately, identically:

| arm | mount | green field | with mounts |
|---|---|---:|---:|
| 13 | floor | 16.95 % | **15.70 %** |
| 17 | floor | 16.41 % | **15.29 %** |
| 31 / 71 / 2 / 97 | inv | 30.97 / 31.20 / 29.53 / 30.41 % | **identical** |

### Why — and it is physics, not a modelling convenience

Measured over all 5 526 certified floor poses and 20 224 inverted poses of
v1's own atlas (chain points + both tool points, world frame):

| arm | base z | highest chain point of a drawing pose |
|---|---:|---:|
| floor | 0.0127 | **0.650** (the elbow; kinematic bound d1 + d3 = 0.649) |
| inv | 0.850 | **0.520** (the shoulder) — 0.33 m BELOW its own base |

A drawing pose has the pen on the paper, so the whole chain hangs low.  Every
inverted arm's mount hardware — plate [h, h + 0.05], boom [h + 0.05, 2.34] —
sits at or above z = h.  So:

* **inverted vs inverted: unreachable.** An inverted arm's chain tops out
  0.33 m under its own mount plane; a neighbour's plate at the SAME height is
  0.16 m beyond its reach even after the 0.09 capsule and the 0.05 static
  margin.  Their base points touch z = h, which is why every capsule table in
  the repo starts at chain index 1 — the base is bolted to the very plate it
  would otherwise collide with.
* **floor vs inverted: clears, but only just.**  At h = 0.85 the measured
  elbow gives 0.650 + 0.09 + 0.05 = 0.790 against a plate bottom at 0.850 —
  **0.060 m of spare**.  The model uses the conservative kinematic bound
  (0.649 rise, so 0.813) and reports **0.037 m**.  Either way it clears, and
  `boom_shadow_active` flips NEGATIVE below **h = 0.813 m** — that is the
  height at which ceiling steel starts eating floor-arm coverage, and it is
  uncomfortably close to the 0.85 the study wants.  One more argument for
  not having floor arms at all.
* **the floor PEDESTALS bite.** Their tops are at z = 0.0127 — the pen's own
  working level — and they stand just outside the web, so a neighbour drawing
  along that edge must keep its pen capsule (0.05) plus the static margin
  (0.05) clear of a 0.30 x 0.25 m box. This is the entire cost above.

The gap the user caught was real; it just lands somewhere other than where it
was feared, and **it is an argument against floor arms**, not against tight
inverted pairs.

## 2. The family comparison

Best of each family, 2 cm certified, tilt <= 15, mounts ACTIVE:

| family | h | sym | union | >= 2 | >= 3 | >= 4 | dead cells | per-arm coverage |
|---|---:|---|---:|---:|---:|---:|---:|---|
| **0 + 6 (grid)** | 0.850 | yes | **99.982 %** | **55.98 %** | **13.95 %** | 5.39 % | **3** | 27.9–31.7 % each |
| 0 + 6 (search best) | 0.850 | no | 99.988 % | 55.87 % | 14.53 % | 4.97 % | 2 | 27.4–32.0 % |
| 0 + 6 (search, symmetric) | 0.850 | yes | 99.982 % | 55.72 % | 14.73 % | 5.66 % | 3 | 28.1–31.7 % |
| **2 + 4 (best)** | 0.850 | no | **99.366 %** | **47.82 %** | 1.94 % | 0.07 % | **105** | floor 11.2 / 14.3, inv 28.6–33.0 |
| 2 + 4 (symmetric) | 0.850 | yes | 99.330 % | 49.36 % | 5.04 % | 0.00 % | 111 | floor 14.6 / 14.6, inv 30.7–31.5 |
| v1 winner, re-scored | 0.850 | no | 99.281 % | 48.04 % | 4.70 % | 1.09 % | 119 | floor 15.7 / 15.3 |

**0 + 6 wins by 0.62 pp of union, 8.2 pp of >= 2-arm overlap, 12.0 pp of
>= 3-arm, and 102 dead cells** — this is not the 1–2 pp coin-flip the trade
was framed as, and the uniform-hardware argument comes free on top rather
than costing points.

The mechanism is in the per-arm column.  A floor arm must stand OUTSIDE the
web (0.13 m minimum setback, or its 0.30 x 0.25 pedestal lies on the paper),
so roughly a third of its 2.18 m² annulus falls off the canvas and it
contributes 11–16 % of the canvas.  A hanging arm sits OVER the canvas and
spends its whole 2.09 m² annulus on paper: 28–32 % each, nearly double.  Six
arms x 28 % is what buys 56 % double coverage.

**Buildability:** all-ceiling means one grid, one boom design, one plate, six
identical mounts — against v1's two pedestals plus four booms, and a table
that had to grow ~0.35 m at the edge the floor arms stood off.  It also
FREES both short edges, which is where paper transport wants to be: v1 put
its floor pair off the south short edge, exactly in the feed's way.

## 3. Height

The round-number grid at each swept height (2 cm certified, mounts active):

| h (m) | union | >= 2 | >= 3 | dead cells | ready poses |
|---|---:|---:|---:|---:|---|
| **0.850** | **99.982 %** | **55.98 %** | **13.95 %** | **3** | 6/6 clear |
| 0.922 | 99.861 % | 54.21 % | 10.58 % | 23 | 6/6 clear |
| 1.000 | 99.245 % | 44.81 % | 5.87 % | 125 | 6/6 clear |

0.85 wins on every metric — the lateral tool's GO annulus is widest there
([0.20, 0.84] vs [0.16, 0.82] at 0.922 and [0.16, 0.78] at 1.00).  Going
LOWER is untested here: no profile was measured below 0.85, and an
all-ceiling rig has no floor arm whose elbow could hit the steel, so the
0.813 m threshold of §1 does not bind this layout — what bounds it instead
is that the arm must still reach DOWN to the paper, which the annulus at
0.85 already does comfortably.

## 4. Sensitivity — coverage vs pair spacing

The one number a fabricator will want to round: how much does it cost to move
the two columns of the grid apart or together?  The window is set by the
annulus, not by preference — for a partner to cover the whole of an arm's
r < 0.20 under-base hole, the two bases must be **>= 0.40** apart (the far lip
of the hole must clear the partner's inner radius) and **<= 0.64** (the near
lip must stay inside its outer radius 0.84).  `layout.PAIR_WINDOW`.

Each of the three pairs pushed symmetrically about its own midpoint,
everything else held; 2 cm certified, tilt <= 15, mounts active:

| pair spacing (m) | union | >= 2 | >= 3 | vs proposed | |
|---:|---:|---:|---:|---:|---|
| 0.40 | 99.41 % | 62.08 % | 16.73 % | −0.57 | FORBIDDEN: bases < 0.50 |
| 0.45 | 99.75 % | 60.43 % | 16.16 % | −0.23 | FORBIDDEN: bases < 0.50 |
| 0.50 | 99.88 % | 58.63 % | 15.66 % | −0.10 | the tightest legal spacing |
| 0.55 | 99.84 % | 57.54 % | 14.95 % | −0.14 | |
| 0.60 | 99.96 % | 56.34 % | 14.26 % | −0.02 | |
| **0.61** | **99.98 %** | **55.98 %** | **13.95 %** | — | **proposed** |
| 0.65 | 99.99 % | 55.01 % | 13.20 % | +0.01 | top of the hole window (0.64) |
| 0.70 | 99.77 % | 53.49 % | 12.38 % | −0.21 | |
| 0.75 | 99.04 % | 53.19 % | 11.29 % | −0.94 | |
| 0.80 | 97.71 % | 53.27 % | 10.23 % | −2.27 | |
| 0.85 | 95.95 % | 53.43 % | 9.03 % | −4.03 | |
| 0.90 | 94.16 % | 53.49 % | 7.87 % | −5.82 | |
| 0.95 | 92.71 % | 53.59 % | 6.65 % | −7.27 | |

**Read this as a 15 cm window, not a knife edge.**  Anywhere in
**0.50–0.65 m** the union sits at 99.84–99.99 % — a 0.15 pp band, which is 25
cells, i.e. nothing.  The build should pick whatever spacing the steel wants
inside that window; 0.61 is merely where the search settled.

Outside it the two walls are different in character and both are visible in
the numbers:

* **Tighter** costs union slowly and BUYS redundancy fast: 0.50 m gives up
  0.10 pp of union for +2.65 pp of >= 2-arm and +1.7 pp of >= 3-arm.  If
  handoffs matter more than the last 25 cells, tighten — but 0.50 m is the
  hard floor, from the base-spacing rule (two arms 0.50 m apart with 0.84 m
  reach are already in each other's laps), not from the hardware, whose own
  minimum is 0.345 m.
* **Wider** costs union fast and buys nothing: the >= 2-arm column FLATTENS at
  ~53.4 % beyond 0.70 m while union falls off a cliff — the pairs have stopped
  covering each other's under-base holes (the window closes at 0.64 m) and the
  overlap that remains is between ROWS, not within pairs.  Past 0.80 m the
  holes are open and 2.3 pp of canvas is simply gone.

So: **do not go above 0.65 m.**  Below it, the trade is coverage-neutral and
the choice is yours.

## 5. Overlap structure and dead zones

**Overlap.** >= 2 arms on 55.98 % of the canvas (v1 green field: 50.31 %;
current rig with the lateral tool: 38.39 %; current rig inline: 25.47 %),
>= 3 arms 13.95 %, >= 4 arms 5.39 %.  Per arm: the MIDDLE row carries
31.6 % each, the two end rows 27.9–28.1 % each — the middle pair sees canvas
on both sides, the end pairs run out of paper.  Triple coverage rose from
4.70 % (v1) to 13.95 % because the three rows are 1.21 m apart while each
annulus is 0.84 m deep, so consecutive rows overlap rather than merely abut.

**Dead zones.** **3 cells of 16 562 (0.018 %)**, and they are three isolated
SINGLE cells at canvas corners — no slivers, no bands, no interior hole.
Compare: v1 green field 102 cells in 32 slivers, v1 re-scored 119 in 40, the
best 2 + 4 layout 105 cells in 27 (its largest a 31-cell patch in the far
corner, x 1.66–1.80, y 3.46–3.62).  The current rig runs 15.09 % dead with the
lateral tool.

**Ready poses.** All six certified ready poses (`layout.certified_ready_pose`
— hover 0.10 m, gated by `validate.check_pose` for joint margin, chain above
paper and PEN TIP above paper) clear every OTHER arm's plate, boom and
pedestal:

| arm | mount clearance | joint margin | tip z |
|---|---:|---:|---:|
| 31 / 71 (middle row) | +0.352 m | 0.703 | +100 mm |
| 13 / 97 | +0.457 m | 0.667 | +100 mm |
| 2 / 17 | +0.462 m | 0.644 | +100 mm |

against a `STATIC_MARGIN` of 0.05 — an order of magnitude of headroom, which
is the same finding as §1 seen from the park side.

## 6. Method, and the mount model

### The schematic mounts (`aris_sixarm/mounts.py`, all parameterised)

| piece | dimensions | z band |
|---|---|---|
| inverted base plate | 0.226 x 0.190 x 0.05 | [h, h + 0.05] |
| inverted boom | cylinder r = 0.10, carried as its circumscribed 0.20 square column | [h + 0.05, **2.34** ceiling grid] |
| floor pedestal | 0.30 x 0.25 | [z_floor − 0.40, **0.0127**] |

Clearance policy is `rig_final.STATIC_MARGIN` = 0.05 (Z_STATIC 0.02 + CALIB
0.03), the same gate the final rig's steel is judged by; capsule radii are
the chain's own (0.09 link, 0.05 pen/bracket).  **Every arm sees all other
arms' hardware and never its own** (`mounts.obstacles_for`, own-tag excluded
— it is bolted there by construction), exactly the final rig's convention;
its own column stays gated by the legacy r = 0.12 proxy, which is the more
conservative of the two radii.

Hardware spacing minima implied by the boxes plus the margin:
`min_column_spacing` = 0.345 m, `min_pedestal_spacing` = 0.441 m.  Both are
**looser than the workspace rule the search already carried** (bases >= 0.50 m
apart, floor-to-inverted >= 0.60 m), so hardware interpenetration is never the
active constraint — `check_spacing` reports both and the study says which
binds.  A yawed footprint is carried as its axis-aligned bounding box, which
can only grow.

### The pipeline

1. **Radial GO profiles** — strict-gate (margin >= 0.30, sigma >= 0.14) annuli
   per mount and height, lateral tool, 5 bearings, 2 cm step
   (`layout.PROFILES_LAT`; unchanged from v1, they are single-arm properties).

       floor 0.0127  [0.34, 0.90]   inv 0.850  [0.20, 0.84]
                                    inv 0.922  [0.16, 0.82]
                                    inv 1.000  [0.16, 0.78]

2. **COARSE — obstacle-aware disc cover.**  Each arm is its annulus MINUS
   corrections for every other arm's hardware, and which correction a piece
   earns is decided by **z bands, not by taste** (`mounts.keepouts`):
   * hardware whose bottom is inside the observer's chain envelope becomes a
     **shadow** — remove every cell whose base->cell SEGMENT passes within
     `r + 0.09 + 0.05` of the column axis.  A wedge/capsule subtraction,
     conservative by construction (it takes the disc around the column too),
     ~4 numpy ops per obstacle over the whole grid.
   * hardware at the PEN's level that the links pass OVER (the pedestals)
     becomes a **footprint keep-out**, not a shadow — the cells behind a
     pedestal are reachable from above and must not be thrown away.

   Multi-start pattern search (steps 0.24 -> 0.03 m), 60 restarts x 3 heights
   x {free, symmetric} x {2 + 4, 0 + 6} = 720 local optima in 28 s at
   ~1.1 ms per obstacle-aware evaluation.  Candidates are **stratified by
   (family, height, symmetry)** before the medium stage: without that the
   stronger family takes every seat and the comparison is never measured.

3. **MEDIUM** — top layouts re-scored by REAL per-arm atlas sweeps (4 cm,
   perpendicular pen) **with the mount boxes active in the lattice clearance
   check**.  The coarse proxy under-predicts by ~1 pp on 2 + 4 (its pedestal
   keep-out is deliberately blunt) and is within 0.05 pp on 0 + 6.

4. **FINE** — finalists at the full 2 cm atlas with the 15-degree tilt cone,
   mounts active, plus the certified ready-pose check of §5.

The obstacle boxes reach the solver through `layout.StudySpec.static_obstacles`
-> `atlas.solve_cell(boxes=...)` -> `rig_final.chain_static_clearance` — the
same plumbing the final rig's frame uses, not a parallel path.

## 6a. What this study still does NOT claim

- **Inter-arm collision is not modelled.**  Six arms whose workspaces overlap
  on 55.98 % of the canvas will contend; v1 already flagged this at 50 % and
  this layout raises it.  Handoffs get easier and traffic gets harder, and the
  conductor is untested at this density.
- The ceiling grid's own cross-members, hangers and services are not modelled
  — only the six booms.  A real grid adds structure the atlas must re-sweep.
- Paper transport, cable routing and the web's own curl are not modelled.  The
  all-ceiling layout leaves both short edges free, which HELPS, but the feed
  must still run under booms whose lowest steel is at z = 0.90 m.
- All six arms are at ONE height.  A mixed-height layout would put a taller
  arm's boom through a shorter arm's mount plane and the z-band reasoning of
  §1 would no longer hold; `boom_shadow_active` is what reports that.
- The bases sit OVER the paper.  Six arms hanging above the web is a
  drips/debris and a maintenance-access question the drawing must answer.

## 7. v1 (superseded) — the green-field study

The v1 mandate fixed the configuration at 2 floor + 4 ceiling-inverted arms
and modelled no steel.  Its winner (`layout.LAYOUT_V1`, kept for the
comparison in §1) put the two floor arms side by side off the south short
edge and the four inverted arms in two transverse pairs at y ~ 1.40 / 2.87,
h = 0.85, scoring **99.38 % union / 50.31 % >= 2-arm**.  Its ranking put four
layouts above 99.29 % and concluded "the placement problem is SOLVED by
geometry at this canvas size".  That conclusion survives v2; what does not
survive is the configuration, because v1 could not see that a floor arm pays
twice — once for standing off the web, once for the pedestal it leaves at its
neighbours' pen height.

### Baselines, unchanged

| configuration | union strict-GO | >= 2 arms | source |
|---|---:|---:|---|
| current `final6_opt`, INLINE pen, tilt <= 15 | 75.93 % | 25.47 % | docs/MERGED_CANVAS.md |
| current `final6_opt`, **LATERAL pen**, tilt <= 15 | 84.91 % | 38.39 % | `out/atlas_final6_opt_lat/` |
| v1 proposal, green field | 99.38 % | 50.31 % | `out/layout_study/v1_green_field/` |
| v1 proposal, mounts active | 99.28 % | 48.04 % | `out/layout_study/v1_with_mounts/` |
| **v2 proposal (0 + 6 grid), mounts active** | **99.98 %** | **55.98 %** | `out/layout_study/grid_h850/` |

The TOOL alone is worth +8.98 pp on the unchanged rig; the LAYOUT is worth
most of the rest.

## 7a. The URDF

`assets/proposed_rig/installation.urdf` is this layout as a robot model: six
namespaced Franka arms welded at `layout.FLEET_PROPOSED`'s base transforms
(inverted, h = 0.85), their base plates and booms, the paper web, and the
LATERAL pen holder on every hand as fixed links — `tcp` → `pen_bracket`
(0.110 along hand x) → `pen_body` (0.110 along tool z) → a `pen_tip` frame.
`environment.urdf` is the same file without the arms.

Both are GENERATED, by `scripts/gen_proposed_rig_urdf.py`, from `layout.py`,
`mounts.py`, `rig_final6.py` and `frames.py`; nothing geometric is typed into
the generator, and `tests/test_proposed_rig_urdf.py` fails if the committed
file is stale.  Three things are worth knowing about it:

- **FR3 limits, not Panda.**  The vendored `panda_arm_hand.urdf` carries Panda
  position/velocity limits — a different robot.  Every revolute limit is
  rewritten from `frames.FR3_MIN/MAX`, `QD_MAX` and `TAU_MAX`.  The Panda
  `drake:acceleration` hint is dropped rather than restated: this repo has no
  FR3 source for it.
- **The ceiling grid is a LEVEL, not a structure.**  §6a says the grid's
  cross-members are unmodelled and the study gated nothing against them, so
  `ceiling_grid_ref` carries visual geometry only and no collision geometry.
  The booms are the only ceiling steel in the file.
- **The boom is drawn round and gated square.**  `mounts.arm_mount_boxes`
  carries the r = 0.10 cylinder as its circumscribed square column so the box
  machinery stays conservative; the URDF writes that square as the boom's
  collision geometry (what was certified) and the cylinder as its visual
  (what gets built).  Each arm's own plate and boom are collision-filtered out
  of that arm, mirroring `mounts.obstacles_for`.

Verification, two independent routes, both reporting max pen-tip error ~1 nm
against `frames`' FK composed with the fleet base transform: the test file
walks the URDF's own chain in pure XML, and `scripts/check_proposed_rig_urdf.py`
loads it in pydrake (station venv).

### 7a.1 The real pen holder, and what it does not agree with

The 2026-08-19 delivery (`raw_slack_file_dump/"Pen holder all parts
2026.08.19"/`, 8 printed parts as STL + SLDPRT, **no assembly file**) is now
the URDF's tool VISUAL: `scripts/extract_penholder22_meshes.py` decimates the
housing and the cap, scales them from millimetres, bakes
`rig_final.penholder22_T_hand` into the vertices and writes
`assets/proposed_rig/meshes/penholder22_*_hand.obj`.  The internal stack (the
clutch, the clutch holder, both spacers, the spring) is omitted: it lives
inside the 21.1 mm bore, is invisible from outside, and its axial order is not
determined without an assembly.  The extractor **re-measures** every constant
in `rig_final.PENHOLDER22` from the CAD on each run and fails if the two ever
part company.

What the housing is, measured: a barrel with a 21.1 mm through bore (necking
to 17.0 mm at the nose, exactly the clutch's OD, so the pen leaves at the
nose), an external thread at the far end for the cap, and a **26 x 26 x 50 mm
square mount post across the barrel with an 18 x 18 x 7 mm socket in each
end**.  50 mm of post plus 2 x 3.5 mm of socket engagement is 57 mm — exactly
the jaw gap of the 28.5 mm finger half-width the 10-degree build's CAD gave —
so the post is what the fingers hold, and its axis is y_hand.

**The "22 deg" verdict.**  It is a CLOCKING, not a tilt, and it measures
**23.00 degrees**: the post's flats and sockets are rotated 23.00 deg about
the POST axis relative to the bore, while the post axis itself is exactly
perpendicular to the bore.  Mounted, that clocking is a lean about y_hand —
the same 23.0 deg docs/FINAL_RIG.md already recorded for this build, and the
same disagreement with the file's name.

**Two things Pete has to rule on, neither of which this URDF decides:**

1. **23 deg of CAD vs 45 deg of planner.**  `frames`' lateral tool puts the
   tip at TCP + R @ (0.110, 0, 0.110) — a 45-degree lean.  The transform is
   gate-validated and stays truth, so the meshes are drawn along the
   planner's ray and the missing 22 deg is parked in the fingertip cradle,
   whose geometry is not in the delivery.  If that cradle turns out to be
   square to the hand, the built tip lands at 23 deg — about **0.047 m
   lateral at this reach, not 0.110** — and either the constant or the
   housing has to move.
2. **100.5 mm of graphite.**  Grip-to-nose is 55.1 mm and the planning tip is
   155.6 mm from the TCP, so the stick has to protrude 100.5 mm past the
   nose.  (docs/FINAL_RIG.md already estimated ~90 mm for the older 0.209 m
   reading, so this is not new — but it is a lot of unsupported 7 mm
   graphite.)

Collision for the holder is NOT the mesh: three cylinders coaxial with the
bore (`rig_final.penholder22_collision`), whose radii are the largest distance
any vertex of either mesh reaches from that axis inside its band, so the union
encloses the visual by construction — 0 of 6 202 vertices outside, checked on
every extractor run and again in `tests/test_proposed_rig_urdf.py`.  A tighter
rotated box for the post was tried and rejected: a rotated square's x-extent
grows with its side, so enlarging it to swallow the reinforcing gussets only
drags in more bare barrel.  The study's own L-shaped two-capsule envelope
(`STATIC_CAPSULES_LAT`) stays in the file as well, collision-only, so anything
checking this URDF is checking at least what the planner certified — and the
union of the two is conservative for both models, which matters because **the
real holder is a straight tube and the planner's tool model is an L**: the
straight diagonal from TCP to tip runs up to 55 mm from either capsule axis,
5 mm outside their r = 0.05.

## 8. Reproduce

```
ARIS_TOOL=lateral python3 scripts/layout_study.py \
        --stages v1,coarse,medium,fine,grid,ready,spacing \
        --restarts 60 --medium 12 --fine 6 --jobs 12
ARIS_TOOL=lateral python3 scripts/layout_plot.py      # out/layout_study.png
ARIS_TOOL=lateral python3 scripts/make_proposed_scene.py  # proposed_scene.html
```

Artefacts: `out/layout_candidates.json` (every stage, the mount parameters and
the constraint set), `out/layout_study/{v1_*,med_*,fine_*,grid_h*,spacing/}`,
`out/layout_study.png`, `out/proposed_scene.html` (the schematic booms, plates
and pedestals are drawn at the dimensions the study treated as obstacles).
