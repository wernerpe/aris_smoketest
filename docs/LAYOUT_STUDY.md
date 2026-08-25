# Layout study — 2 floor + 4 ceiling-inverted arms, lateral pen (2026-08-25)

**USER MANDATE**: re-evaluate the arm positions.  Fixed configuration: **2
floor arms + 4 ceiling-inverted arms, NO wall mounts**, drawing the merged
**1.8034 x 3.63064 m** canvas with the **LATERAL pen holder** (tip = TCP +
R @ (0.110, 0, 0.110), docs/DECISIONS.md "LATERAL PEN HOLDER").

**THIS IS A GREEN-FIELD STUDY TO INFORM THE PHYSICAL REDESIGN.**  No frame
model exists for arbitrary base positions, so the study models NO steel: no
frame boxes, no paper-transport hardware.  What it does model conservatively:
the paper-plane clearance every certified pose keeps, each inverted arm's own
mounting-boom keep-out (r = 0.12 cylinder above the plate), and base-spacing
sanity (bases >= 0.5 m apart; every floor base >= 0.6 m horizontally from
every inverted base so no ceiling boom descends into a floor arm's
workspace envelope; floor bases >= 0.13 m outside the web so the 22.6 cm
base plate clears the paper).  Structure and collision modelling follow once
a layout is picked, and WILL cost coverage — the honest comparisons below
say how much that costs the current rig.

## 0. The baselines

| configuration | union strict-GO | >= 2 arms | source |
|---|---:|---:|---|
| current `final6_opt`, INLINE pen, tilt <= 15 | 75.93 % | 25.47 % | docs/MERGED_CANVAS.md |
| current `final6_opt`, **LATERAL pen**, tilt <= 15 | **84.91 %** | 38.39 % | `out/atlas_final6_opt_lat/` (this study) |
| proposed green-field layout, LATERAL pen, tilt <= 15 | **99.38 %** | 50.31 % | `out/layout_study/fine_0/` |

Two separate effects: the TOOL alone is worth **+8.98 pp on the unchanged
rig** (the wrist stands off from the work — both side arms jump 22.5 ->
26.3 %, and every mount gains), and the LAYOUT is worth most of the rest —
with the caveat that the green-field number excludes the steel and transport
clearances that account for a large share of the current rig's remaining
15.09 % dead.  Reserving the transport strips (see §4) still leaves the
proposal near-total over the drawable interior.

## 1. Method — rapid coarse-to-fine

1. **Radial GO profiles** (once per mount type and height): strict-gate
   (margin >= 0.30, sigma >= 0.14) annuli measured by `atlas.solve_cell` on
   rays from the base, 5 bearings, 2 cm step, perpendicular pen, lateral
   tool, green field.  Yaw-invariant to the probe — the disc model is honest:

   | mount | height | inline GO annulus | LATERAL GO annulus | area (lat) |
   |---|---|---|---|---:|
   | floor | 0.0127 | [0.46, 0.80] | **[0.34, 0.90]** | 2.18 m² |
   | inv | 0.850 | [0.32, 0.68] | **[0.20, 0.84]** | 2.09 m² |
   | inv | 0.922 | [0.38, 0.66] | **[0.16, 0.82]** | 2.03 m² |
   | inv | 1.000 | [0.26, 0.64] | **[0.16, 0.78]** | 1.83 m² |

2. **COARSE — disc cover** (`scripts/layout_study.py`): place the six annuli
   to maximise strict-GO union area on a 4 cm grid; tie-breaks >= 2-arm
   overlap (handoffs), then 180-degree rotational symmetry (buildability as
   two identical units).  Multi-start pattern search (steps 0.24 -> 0.03 m),
   60 restarts x 3 heights x {free, symmetric} = ~360 local optima in ~35 s
   — a disc evaluation is ~60 us, so the search visits a few hundred
   thousand layouts.
   Floor bases roam any edge at 0.13-0.35 m setback; inverted bases roam the
   canvas (+0.25 m).

3. **MEDIUM**: the top layouts re-scored with REAL per-arm atlas sweeps
   (4 cm, perpendicular pen, lateral tool) and re-ranked; the disc model
   under-predicts the real sweeps by ~0.5-1 pp (tilt-free profiles are
   conservative) and preserves their order.

4. **FINE**: the finalists at the full 2 cm atlas with the 15-degree tilt
   cone — the exact sweep the 75.93 % / 84.91 % baselines come from, minus
   the boxes that do not exist here.

## 2. The winning layout

**h = 0.85 m, 99.38 % union strict-GO, 50.31 % >= 2 arms** (2 cm,
tilt <= 15, lateral pen).  Exact base coordinates, canvas frame (x across
the 1.8034 m width, y along the 3.63064 m length, z = 0 the paper):

| arm | mount | x (m) | y (m) | base z (m) |
|---|---|---:|---:|---:|
| 13 | floor | 1.15672 | -0.14343 | 0.0127 |
| 17 | floor | 0.55872 | -0.13209 | 0.0127 |
| 31 | inv | 0.56596 | 1.41294 | 0.850 |
| 71 | inv | 1.23186 | 1.39667 | 0.850 |
| 2 | inv | 0.54082 | 2.85289 | 0.850 |
| 97 | inv | 1.19379 | 2.88897 | 0.850 |

The pattern is readable: the two floor arms stand side by side off ONE short
edge (y < 0, plates ~0.14 m outside the web) and own the first ~0.77 m of
the canvas; the four inverted arms form two transverse pairs at y ~ 1.40 and
y ~ 2.87, each pair splitting the width at x ~ 0.55 / 1.21 so each arm's
under-base hole (r < 0.20) falls inside its partner's annulus (outer 0.84).
Spacing: floor-floor 0.60 m, in-pair inverted 0.65-0.67 m, floor-inverted
>= 1.54 m — all constraints hold (`layout.check_spacing` -> clean).

Height: h = 0.85 beats 0.922 at every stage — at the fine stage by
**0.61 pp union and 3.5 pp overlap** (99.38/50.31 vs 98.77/46.85): the
lateral tool's annulus is widest at 0.85.  1.00 is behind both.

Registered as **`fleet.rig("proposed")` / `ARIS_RIG=proposed`**
(`aris_sixarm/layout.py`, `LAYOUT_PROPOSED`) — env-selectable, NOT the
default.

## 3. Ranking (fine stage, 2 cm, tilt <= 15)

| rank | layout | h | symmetric | union | >= 2 | >= 3 |
|---|---|---|---|---:|---:|---:|
| 1 | floor pair south + two transverse inv pairs | 0.85 | no | **99.38 %** | 50.31 % | 4.70 % |
| 2 | one floor arm per long edge, inverted zigzag | 0.85 | yes | 99.32 % | 50.04 % | 5.80 % |
| 3 | rank-1 family, inner pair pulled tighter | 0.85 | no | 99.29 % | 50.27 % | 4.81 % |
| 4 | best h = 0.922 (floor pair north, two pairs) | 0.922 | no | 98.77 % | 46.85 % | 2.55 % |

Everything at h = 0.85 lands above 99.29 % — the placement problem is
SOLVED by geometry at this canvas size; the discriminators are height,
overlap structure and buildability.  The SYMMETRIC runner-up (one floor arm
per long edge, 180-degree rotation symmetry — buildable as two identical
units) is only 0.06 pp behind with better triple coverage; it is the
recommended fallback if the build strongly prefers mirrored units.

## 4. Dead zones and the transport caveat

The winner's dead space is **0.62 % — 102 of 16 562 cells in 32 slivers**,
all on the canvas boundary: the largest is 23 cells (x 1.70-1.80,
y 2.08-2.22, the right edge between the two inverted pairs), the corners
carry ~14 cells each, and nothing else exceeds 4 cells.  **No interior hole
and no dead band.**  Compare the current rig: 24.07 % dead inline, 15.09 %
lateral, concentrated in two full-length edge strips and under-shoulder
holes.

**Transport caveat**: the feed roll, guide rods, winder and paper curl are
not modelled here.  On the current rig they pin the web's long edges;
reserving the same strips (keeping x in [0.06, 1.74]) from the proposal
still leaves **99.79 %** of the reduced canvas covered — the proposal's
coverage does not depend on the strips the transport needs.  The redesign
must still route the web feed PAST the short edge the floor arms occupy, or
feed along a long edge; either way the carve-outs land where the layout has
redundancy to spare.

## 5. Overlap structure

>= 2-arm coverage **50.31 %** (vs 38.39 % current-lateral, 25.47 %
current-inline), >= 3 arms 4.70 %, >= 4 arms 1.09 %.  Per arm (strict-GO,
% of the whole canvas): inverted 29.5-31.2 % EACH (vs 14.9-22.5 on the
current rig), floor 17.0/16.4 % (vs 13.7/13.6).  By band: the south band
(y < 0.8, floor pair + first inverted pair) runs at **67.3 %** >= 2-arm;
the two inverted bands at ~46 %; the old seam strip (y 1.70-1.93) is
**100 % covered, 49.4 %** >= 2-arm — the seam concept itself disappears
with the continuous canvas and no web boundary in the layout.

## 6. What this study does NOT claim

- No steel, no transport, no cable routing is modelled; every number is the
  kinematic ceiling of the layout.  The current rig loses ~9-14 pp to its
  own steel and transport — a redesigned frame will lose SOMETHING here too.
- The ceiling booms are modelled only as the legacy r = 0.12 keep-out above
  each plate; a real ceiling grid adds structure the atlas must re-sweep.
- Inter-arm coordination (the conductor) is untested on this density:
  51.6 % >= 2-arm overlap buys handoffs AND contention — docs/MERGED_CANVAS
  §middle-band shows both arrive together.  Conducts follow once a layout
  is picked and modelled.
- Floor-arm bases sit 0.13-0.14 m outside the web; the stands they need are
  ordinary base plates on the (extended) tabletop, but the table itself must
  grow ~0.35 m at that edge.

## 7. Reproduce

```
python3 scripts/lateral_eval.py                     # Phase A tool numbers
python3 scripts/run_atlas6.py --pen-lat 0.110 \
        --out out/atlas_final6_opt_lat --png out/atlas_final6_opt_lat.png
python3 scripts/layout_study.py --restarts 60 --medium 6 --fine 2 --jobs 12
python3 scripts/layout_plot.py                      # out/layout_study.png
python3 scripts/make_proposed_scene.py              # out/proposed_scene.html
```
