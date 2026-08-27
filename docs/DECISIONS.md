# Decisions — the numbers, and where each one is anchored

## LATERAL PEN HOLDER (2026-08-25) — the tool model changed

| Quantity | Value | Source / provenance |
|---|---|---|
| Lateral tip offset | **0.110 m along hand x** (perpendicular to finger travel): tip = TCP + R @ (0.110, 0, 0.110) | USER-SPECIFIED 2026-08-25.  `frames.PEN_LAT_HOLDER` |
| Axial tip offset | **0.110 m below TCP along tool z, unchanged** | USER-CONFIRMED ESTIMATE 2026-08-25: tip ~15 cm below the gripper's white-housing bottom; housing bottom ~0.066 m below the hand root puts the tip ~0.216 m from the hand root = ~0.113 below TCP — matches the gate-B 0.110 within mm.  Refine by touchdown calibration once the holder is mounted |
| Default tool | **inline** (`frames.PEN_LAT = 0.0`) — every published number reproduces | `ARIS_TOOL=lateral` (or `frames.activate_tool`) switches the whole stack |
| Tool yaw phi | a REAL redundancy DOF under the lateral tool (the yaw==q7 aliasing is broken); coarse 8-point ring, adaptive, coupled (s x phi x q7 x branch) rescue | `aris_sixarm/lateral.py`; planner.build_lattice docstring |
| Tool capsules | TWO capsules through the bracket corner (TCP->corner r 0.05, corner->tip r 0.05); the single-capsule union envelope CANNOT cover an 11 cm lateral arm | `rig_final.STATIC_CAPSULES_LAT`, `scene_check.RADII_LAT`, `coordination.CAPSULES_LAT`, pinned together by tests/test_lateral.py |

Measured effect (arm 31 inverted, final6_opt, 4 cm patch, perpendicular pen):
strict-GO 536 -> **810 cells (+51.1 %)**, max GO radius 0.75 -> **0.86 m**,
median margin 0.528 -> 0.624.  Certified strokes plan in ~80-280 ms with the
phi ring (`scripts/lateral_eval.py`, `out/lateral_eval.json`).

### The 22-deg holder's CAD arrived (2026-08-25) and it does NOT agree

`raw_slack_file_dump/"Pen holder all parts 2026.08.19"/` — 8 printed parts as
STL + SLDPRT, millimetres, **no assembly file**.  Machine-readable in
`rig_final.PENHOLDER22`; extracted by
`scripts/extract_penholder22_meshes.py`; drawn in `assets/proposed_rig/`.

| Quantity | CAD says | Planner says | Status |
|---|---|---|---|
| Pen lean out of tool z | **23.00°**, measured as the clocking of the mount post's flats/sockets about the post axis (the file is named "22 deg") | **45.00°** = atan2(0.110, 0.110) | **OPEN.** Transform unchanged (gate-validated); the meshes are drawn along the planner's ray and the 22° of difference is parked in the fingertip cradle, which is NOT in the delivery.  If the cradle is square to the hand, the built tip is ~0.047 m lateral, not 0.110 |
| Grip -> nose | **55.1 mm** (post axis crosses the bore 55.1 mm behind the nose) | tip is 155.6 mm from the TCP | **OPEN.** needs **100.5 mm** of ⌀7 graphite past the nose (FINAL_RIG.md's older estimate was ~90 mm) |
| Mount | 26 x 26 x 50 mm square post, 18 x 18 x 7 mm socket each end | fingers at half-width 28.5 mm | **CONFIRMS** the existing number: 50 + 2 x 3.5 = 57 mm = the jaw gap |
| Tool shape | a STRAIGHT tube from the grip to the tip | an **L** (bracket along hand x, then pen along tool z) | **OPEN.** the straight diagonal runs up to 55 mm from either capsule axis, 5 mm outside their r = 0.05, so the L is not an envelope for it — the proposed-rig URDF carries BOTH |

## FINAL RIG values (2026-08-21) — the drawing decides

The user's final-rig files (`raw_slack_file_dump/`: the "3 arms, 1 up, 1 side,
1 down" PDF+DXF and the pen-holder CAD) replace every placeholder below.
Extraction, cross-checks and the independent verification live in
`docs/FINAL_RIG.md`; the machine-readable geometry is `aris_sixarm/rig_final.py`
and the registry is `fleet.FLEET` (= `FLEET_FINAL`).  The legacy table at the
bottom of this file remains the record of the six-arm era, and
`fleet.FLEET_SIXARM` keeps that layout runnable bit for bit.

| Quantity | FINAL value | Source / provenance |
|---|---|---|
| Arm count & mounts | **3 arms: 13 upright (tabletop), 31 inverted (under central beam), 2 side (vertical plate, J1 horizontal)** | drawing MTEXT names the arms; plate solids + dims 45,7 / 55,8 / 34,9 / 91,6 |
| Arm base poses | 13: (0.88043, −0.12680, 0.01070) rotz(π/2) · 31: (0.24491, 1.25678, 0.92200) rotx(π) · 2: (1.60984, 1.25723, 0.77600) roty(−π/2)rotz(π), canvas m | DXF solids, verified independently to ≤0.011 cm (`FINAL_RIG.md` table) |
| Inverted base → paper | **0.92200 m** (was the H_INV debate: 0.92/0.924/0.97/1.00) | plate bottom 155.868 − paper 63.668 cm; within 2 mm of the 2026-07-07 rig survey (0.924). `H_INV_DEFAULT` stays a LEGACY knob; final poses ignore it |
| Sheet | **1.8034 × 1.700 m**, a paper web with roll/winder transport | paper solid 40; roll & rods are collision boxes |
| Pen tip offset | **`PEN_EXT` = 0.110 below hand TCP, unchanged** — the CAD does NOT confirm it (10° build: tip (−8, 0, +45.3) mm from TCP, axis tilted 10°; 23° clutch build: protrusion adjustable). The real-touchdown measurement outranks a CAD whose deployed configuration is unconfirmed; ASK the user which build + protrusion ships | pen-holder CAD extraction (`FINAL_RIG.md` Pen holder); gate-B touchdown 2026-07-12 |
| Pen collision capsule | **r = 0.05 m** (`rig_final.PEN_R_FINAL`), the union envelope of both holder builds incl. the clutch at 0.209 m flange→tip | CAD envelope; legacy rig keeps r = 0.03 |
| Frame obstacle | **35 conservative boxes** active in lattice, validator, atlas, and scene_check (its own derivation) | `rig_final.FRAME_BOXES_W_CM`, independently verified containment |
| Static clearance margin | **Z_STATIC 0.02 + CALIB_STATIC 0.03** vs structure; inter-arm stays SAFETY 0.05 + CALIB 0.03 | Z_PAPER convention for static surfaces; calib documented below |
| Calibration margin (0.03 m) | **KEPT for the final rig** — the drawing is a plan, not an as-built survey, no base has been surveyed at this geometry, and the drawing itself warns arm 2's plate-to-boom joint is "not stiff and stable". Drop it (per arm) the day a survey lands; arm 2 last | drawing MTEXT warning; `rig_final.CALIB_STATIC`, `coordination.CALIB_M` |
| Ready poses | 13: legacy `Q_READY_FLOOR` (checked clean) · 31: **`Q_READY_INV_FINAL`** (the legacy pose dips the pen 6.6 mm below the paper at h=0.922 and grazes the boom — refused by the gate, pinned in tests) · 2: **`Q_READY_WALL`** (derived; no wall arm ever existed) | `frames.py` provenance comments; `tests/test_final_rig.py` |
| Sequencer/conductor fixtures | draw 0.12 m/s, transit 0.80 m/s, qd-frac 0.6, freeze-in-place idle — carried over unchanged | no new evidence against them; re-examined per drawing by `select_profile` |
| Bench corpus | generators re-anchored to the ACTIVE rig (hatch on arm 31's patch clamped to the sheet; ray/spiral radii scale to min(sheet)/2) | `aris_sixarm/bench/__init__.py` |

## Legacy — numbers the upstream repos disagree on (six-arm era)

The `Aris_Kindt` branches carry conflicting values. Every pick below was made
for the SIX-ARM layout (`fleet.FLEET_SIXARM`), which regression tests keep
runnable; the final rig above supersedes it as the active registry.

| Quantity | Values in the wild | We use | Why |
|---|---|---|---|
| Inverted base → paper | 0.92 (verbal) · 0.924 (cm sheet, "authoritative") · 0.97 (GUI SCENE_ARGS) · 1.8542 world (booth URDF) · 1.00 (IKA answer) | **1.00** (`H_INV_DEFAULT`) | IKA 07-12: h=100cm + ≤15° tilt ⇒ solid ⌀154 cm, no center hole; "do NOT lower the mount". Pass `--h-inv 0.924` to model the rig as measured 2026-07-07. |
| Arm base XY | booth URDF (3 seats only) · presets (several) · registry `base_world_m` (only arm 13 surveyed) | **six_arm_display_ADJUSTED_h80.json** | Only complete 6-arm layout on record. |
| Floor base z | 0.6477 world (surveyed, arm 13) vs table top 0.6096 (URDF) / 0.635 (revised scene) | **+0.0127 above paper** | 0.6477 − 0.635 = exactly one base plate (12.7 mm). |
| Pen tip offset | 0.110 below TCP (IKA gate) · 0.209 from flange (ika_workspace) · 0.15 (variable_pressure yaml) · 0.015 (Isaac) | **0.110 below TCP** (`PEN_EXT`) | Gate-B validated against the real touchdown at MZ=0.924. |
| Arm 2 mount | registry: wall plate (pitch π/2, standoff 0.70) · presets/GUI: topdown boom | **topdown** | Matches every preset; the wall experiment still draws on the same paper. Revisit if arm 2 goes active as a wall arm. |
| Sheet size | 218×170 cm (1-arm) · 90×150 (raster) · 360.7×196.1 (6-arm preset) | **360.7×196.1 cm** | The 6-arm planner preset sheet. |
| σ_min gate | 0.08 (ika_workspace default) vs 0.14 (ika_plan strict mask) | **0.14** | Field-validated dual-mask value. |
| Joint margin gate | normalized 0..0.5 (workspace_map) vs absolute rad ≥0.30 (IKA) | **0.30 rad absolute** | IKA binding plan; comparable across joints. |

## CSAIL logo run (2026-08-19)

| Quantity | Values in the wild | We use | Why |
|---|---|---|---|
| Logo size on the sheet | "~3.3 m wide, nice and big" (brief) | **2.411 x 1.841 m**, centred, 0.06 m margin | The mark's aspect is 212:162 = 1.309, so 3.3 m of width needs 2.52 m of height and the sheet has 1.961. Height binds; distorting a logo to hit a width is not an option. This IS the largest undistorted placement. |
| Source raster | `csail_old_med.gif` 212x162 · `csail_old.jpg` 104x79 | **the .gif** | Twice the linear resolution, indexed palette (37 colours, no JPEG ringing to unmix). |
| Palette | grey ~#8a8c8e / orange ~#c8651b (by eye) | **#666665 / #cb6608** (measured) | The two dominant non-white palette entries in the .gif; the unmixing is a least-squares fit against exactly these, so guessed values would bias every coverage. |
| Wordmark | skeleton like the rest of the art | **boundary contours** | The glyphs are solid, ~31 px wide at 4x against ~11 px lines. A skeleton of a solid "S" is a stick figure; the outline is the letterform. |
| Which arms draw | registry `active` flag: 13, 17, 31, 97 (2 and 71 parked) | **the flag** | Honoured as-is; the 52 % of the logo nobody reaches is reported, not designed around — the brief says arm positions change later. |

Open issues inherited from upstream (not resolved here):
- Base yaw of the inverted arms vs the table axes was never surveyed → per-arm atlases
  are yaw-invariant disks; table-edge clipping per arm is NOT modelled.
- 5 of 6 `base_world_m` entries in `arm_registry.py` are unsurveyed (None).
- Collision checks are the IKA proxies (paper clearance 2 cm, boom cylinder r 0.12);
  real rig geometry lives in `aris_planning_scene.py` upstream and should replace them
  before trusting near-boom / near-edge cells.
- Kinematics use the Panda visual/DH model (identical geometry to FR3; limits are FR3).

## 2026-08-26 — the arm against itself, and the cone that was never asked to certify

Three things shipped together because each one needs the one before it.

**1. A self-collision guard exists** (`aris_sixarm/selfcoll.py`). Nothing in
this package had ever checked an arm against its own metal; `dead_disc_anatomy`
had written that down rather than fixed it. Measured: 8 of 1 200 configurations
drawn from the joint box AND held to the strict 0.30 rad comfort margin put the
arm's own metal within 10 mm of itself, so the joint limits are not the guard
they were assumed to be. The model is measured from the manufacturer's meshes
in each link's own frame (3 bands per body, 7 z-bands for link0 — the one body
that never moves), watched over pairs four or more joints apart, at a 20 mm
margin with no calibration term because an arm's links share its own encoders.
It costs the shipped certified map NOTHING (0 of 23 376 strict-GO cells, the
tightest holding 63.7 mm) and it invalidated exactly one shipped pose: **arm
71's park, which had its wrist 26.9 mm from its own base.** Re-derived on the
same bearing at the radius and hover four of the other five already used.

The residual is published with it: three pairs closer than four joints can also
reach contact in the metal, and this guard does not see them. What stands there
is the FR3's joint limits and the mechanical design — the same thing that stood
there before, now with a number on it.

**2. The atlas searches for a pose that PASSES** (`atlas.solve_cell`). It used
to return the best-MARGIN pose that cleared metal and let `strict_go` judge that
one row afterwards; the 15-degree cone was a CLEARANCE fallback and was never
asked to certify anything. Now: per lean in ascending order, every solution at
margin >= 0.30, kept at sigma >= 0.14, clearance-checked best-margin first,
first that clears wins — and the lean it stopped at is recorded per cell as
`min_lean_deg`. Monotone by construction: the legacy pick is still in the gated
search's own candidate set, so no cell that certified can stop certifying.
`model_signature` carries the search policy now, not only the geometry.

**3. The drawing planner can lean the pen** (`lateral.plan_adaptive`).
`tilt_max_deg` used to be noted and dropped. It is restored as a RESCUE behind
both flat stages, climbing 2.5 -> 15 and stopping at the first lean that
certifies, with the validator handed the lean the plan actually used. A stroke
that certifies flat is planned by byte-identical code.

The 109.9 mm span of the v8 logo that nobody drew — the one the dead-disc
decomposition called a base column's shadow — certifies whole for arm 71 at a
7.5-degree lean, validator clean.

### ...and what it costs the logo: the A/B, on the same placement

The v8 placement (1.4309 x 1.8736 m), 16.7370 m of traced ink, four runs that
differ in one thing each. Allocation coverage, `--skip-unconductable`:

| run | atlas | pen lean | self-collision guard | empty | coverage |
|---|---|---|---|---|---|
| v8 | best-margin | none | none | 0.1229 m | 99.2660 % |
| v9flat | gated | 0 deg | on | 0.5451 m | 96.7431 % |
| v9b, v9d | gated | 15 deg | on | 0.4352 m | 97.3996 % |
| v9ng | gated | 15 deg | **off** | **0.0131 m** | **99.9220 %** |

Three things fall straight out of it.

**The capability works.** Gated atlas plus a leaning pen draws 99.9220 % of the
logo at full size — 13.1 mm of ink short of complete, and 0.66 points better
than the best this project had. The 109.9 mm span the dead-disc decomposition
called a base column's shadow certifies whole for arm 71 at a 7.5-degree lean,
validator clean (margin 0.164, sigma 0.241).

**The lean is worth 0.66 points on its own** (96.7431 -> 97.3996 with the guard
on), which is what a rescue that only ever adds should look like.

**And the self-collision guard costs 2.52 points** (99.9220 -> 97.3996), 0.42 m
of ink, every metre of it in "certifies the ink and cannot fly to it" — the
guard removing hover and transit poses, not drawing poses. It removes 3.4 % of
the hover fiber over arm 31's certified cells and no cell loses every hover, but
at the ends where only one hover was flyable, 3.4 % is the whole answer.

THE GUARD STAYS ON. Those flight paths were never checked by anything —
`writing.py` says its joint interpolation "makes no collision or self-collision
guarantee" in as many words — and a pen-up that folds an arm through itself is
exactly the failure this gate was built for. The shipped number is 97.3996 %
and the reason is written down. What would buy it back is not a weaker guard
but a router that knows about it: the ends being refused are ones where the
hover ladder had one flyable answer and now has none, and `paper.route` has
seven more shapes it has not been asked to try under this constraint.

## 2026-08-27 — the pen-up layer searches under the guard

The guard cost the logo 2.52 points and every metre of it was "certifies the ink
and cannot fly to it". That is a statement about a SEARCH, not about a rig, and
it was measured as one (`out/guard_pocket.py`, 1 127 certified cells of all six
arms, guard on against the same run with `selfcoll.self_ok` stubbed true; the
unit is a span END and the question is `allocate.depot_round_trip`'s — lift onto
the hover, fly home from it, fly back):

| | guard on | guard off |
|---|---|---|
| cells with a hover at all | 93.4-94.6 % | 93.4-94.6 % |
| span ends that join the depot | 53.6-82.2 % | 56.7-81.9 % |

**The guard costs almost no hovers.** The two columns agree to within 0.6 points
on every arm. What it removes is not the hover, it is the ROUTE out of the one
it leaves — which is what "cannot fly to it" was saying all along.

**And half of every pocket is a search that stopped.** 267 of the sampled ends
are in a depot pocket and 138 of them — 51.7 % — hold a pose on their own fiber
that joins the depot. Only 38 of the 267 are the guard's doing, and 25 of those
38 are recoverable. The rest were being given back before the guard existed.

**How deep the search has to go.** The tier shipped with a 24-candidate budget,
picked when it was measured on 14 pockets:

| budget | of the 138 recoverable |
|---|---|
| 4 | 52.2 % |
| 8 | 71.7 % |
| 16 | 88.4 % |
| 24 | 93.5 % |
| **48** | **98.6 %** |
| 64 | 100.0 % |

48 is what ships. The median success is at 3.5 candidates, so it is the pockets
nothing can fix that pay for the depth, and they only ever run where
`allocate._rescue_pocket` has admitted an end that is about to cost ink.

**The lean is a rung of that fiber, last.** 9 of the 138 are found only at a
lean (7 at 7.5 deg, 2 at 15). `writing.HOVER_LEAN_MAX_DEG` is the RUN's cone,
set from `--tilt-max-deg`, so a flat run reaches exactly the poses it always
did.

### ...and the third obstacle, which nothing had ever asked about

`selfcoll` gates the POSES a route is built from and said nothing about the
straight joint-space line BETWEEN two of them — the motion this package's
pen-up layer exists to certify, and the one `writing.py` says outright "makes no
collision or self-collision guarantee" about. It is not inert: sampled along
straight moves between certified drawing cells of arm 31 — poses the guard
passes at 63.7 mm or better at both ends — the model reads **-194.7 mm** partway
across.

`paper.SELF_SAFE` closes it in the three places that decide whether a move is
flown — `route`'s `legs_ok`, `move_ok`, and `sequence.dive_screen` — at the
producer's `SELF_PLAN_MARGIN`, clamped to the endpoints and residual-corrected
the way the static bound already is. A bounding-sphere screen settles 99.876 %
of the 165 watched pairs without any segment arithmetic, which is what makes it
10 us a configuration instead of 95.

**IT IS NOT OPTIONAL, AND THAT IS THE THING THE A/B SETTLED.** The first
instinct was that it could ship OFF the way `FRAME_SAFE` did — right,
unaffordable, written down — because `scene_check` sweeps self-clearance on
every conducted frame and would refuse anything that folded. Run end to end on
the same placement, everything else identical, that is exactly backwards:

| run | pen-up LEGS gated | allocated | conducted |
|---|---|---|---|
| v10ns | no | **100.0000 %** | scene_check REFUSES phase 1 at **-177.8 mm** on arms 13, 71, 97 — nothing renders |
| v10 | yes | 94.6994 % | every conducted phase holds **21.0-60.5 mm** against the checker's 20 |

The checker catching it does not make the coverage real; it makes the coverage a
phase thrown away three stages later. A gate the ROUTER does not know about is a
gate the allocator spends its whole budget walking into — the inf-pricing lesson
this module has now learned three times: the paper, the metal, and itself.

**What it costs, and why.** 5.30 points of allocated logo, and it is not the
gate being wrong — every crossing it refuses is a pen-up whose straight
joint-space line puts the arm inside itself. It is the ROUTER having nothing to
offer such a crossing, and that is measured too: adding 8 cm and 5 cm rungs to
`TRAVERSE_STEPS` — the obvious fix, since a fold is what a LONG interpolation
commits — recovers **2 of 520** crossings for 1.9x the clock. A fold is not a
long hop on the hover plane; it is a change of IK branch that no walk through
hover poses avoids. What buys those crossings back is a pen-up planner that
searches configuration space, which this package does not have.

**So the honest headline is a pair of numbers, not one.** The hover-fiber
recovery is worth +2.60 points of allocation (97.3996 -> 100.0000) and closes
the whole gap the guard opened — the capability is real and it is measured. The
leg gate then costs 5.30 of them, and it is the price of the programme being
conductable at all.

### the solo-drawable map, re-issued complete

`scripts/feasible_workspace.py --fiber-tries 12 --hover-lean-deg 15`, all six
arms, **26 973 certified cells swept** (the 2026-08-26 gated run was interrupted
with 1 752 of arm 97's 4 299 done, which is most of why its number was a lower
bound), every pen-up in it certified against the arm's own metal along the whole
leg:

| | 2026-08-26 (partial) | 2026-08-27 (complete) |
|---|---|---|
| a certified DRAWING pose | 94.17 % | **99.64 %** |
| + a certified HOVER over it | 92.64 % | **99.00 %** |
| + the arm can FLY there | **>= 82.19 %** | **84.05 %** |

5.568 of 6.625 m2. Largest clean portrait block 0.34 x 2.68 m (0.911 m2), up
from 0.32 x 2.32. 23.05 % of the canvas is reachable by two arms or more and
5.10 % by three; the deepest cell is four arms deep.

**And the binding constraint has not moved.** 99.00 % of the canvas has a
certified hover over it and 84.05 % can be flown to: 14.95 % of the canvas —
0.990 m2 — is ink an arm can draw and cannot approach. 70.2 % of the dead area
is the six solid discs under the booms; the other 29.8 % is confetti, and a
quarter of every dead cell has five or more feasible neighbours. The pen-up
router is still the whole story, and the fiber budget here is 12 candidates
(82.6 % of what an unbounded search finds), so 84.05 % remains a LOWER bound.

## THE PEN-UP PLANNER (2026-08-27) — the ladder stopped being the last word

`aris_sixarm/transit.py`, bidirectional RRT-Connect in the 7-DOF joint space,
wired in as `paper.route`'s LAST tier.  `paper.py` had already written the
ticket and closed it as out of scope: "what buys those crossings back is a
pen-up planner that searches configuration space, which this package does not
have and which is a project rather than a flag."

**Why the ladder could not be finished instead.**  Every one of its forty-odd
shapes walks a two-dimensional surface — a tip position on a hover plane, with
the elbow following whatever the analytic solver hands back.  The crossings
that survive it have endpoints in different components of THAT surface while
being perfectly well connected in the seven the arm moves in.  Measured:
adding 8 cm and 5 cm rungs to `TRAVERSE_STEPS` recovers **2 of 520** crossings
for 1.9x the clock.

**What the planner recovers, measured on the rig rather than argued.**  Arm 31
of the proposed rig at h = 0.940 with the lateral holder, 650 ordered crossings
between 26 certified hovers:

| | crossings |
|---|---|
| shape ladder settles | 584 |
| ladder exhausted | 66 |
| ...of those, the planner flies | **65** |
| ...refused on the clock | 1 |
| paths refused by `legs_ok` on re-check | **0** |

1.90 s per plan, 598 certified edges and 39 tree nodes per plan, shortcut
8.2 -> 4.3 nodes.

**The certification is the ladder's own, and that is the design.**  The planner
proposes polylines; `paper.route`'s `legs_ok` certifies them, leg by leg,
against the same three obstacles at the same floors as a skirt.  Every edge the
search accepts is `paper.leg_bounds` + `paper.leg_self_lb` at `paper.SAMPLES`,
residual-corrected, **plus `transit.PAD` = 2 mm** — producer strictly tighter
than the certifier that grades it, which is strictly tighter than
`scene_check`.  Nothing this tier finds can enter a timeline the ladder's own
certifier would have refused.

**Determinism is a seed, not a hope.**  `tests/test_balance.py` asserts a cost
matrix is bit-identical serial and on four workers, and the sequencer screens
crossings in a fork pool.  The seed is `blake2b` over the scene signature (arm
id, tool, pen, mount height, every static box, the four floors) and both
endpoints — never `hash()` (salted per process), never `id(spec)` (different in
every worker).  `tests/test_transit.py` re-plans in a cold interpreter under a
different `PYTHONHASHSEED` and compares digests.

**`~/git/cc_experiment` was checked first and is not used.**  It is a genuinely
certified continuous checker for Drake — a proof over the continuum, 0.251 ms
for a 7-DOF straight edge, 9x faster than Drake's sampled checker at 0.01 rad,
which is strictly stronger than the bound shipped here.  It has no Python
bindings of any kind (a static archive; its own `docs/UPSTREAMING.md` defers
them to upstreaming because of a pybind11 ABI match against the prebuilt
install), it links a Drake fork (0.0.20251016, py3.12) that is not the pip
wheel (1.47.0, py3.10) this repo's demos run, and it ships no licence file.
Days of ABI work on the far side of a Drake migration, for a checker this repo
would still have to restate independently — `scene_check` is a separate
derivation on purpose.  Worth having later as an **A/B oracle** against these
bounds; the reasoning is in `transit.py`'s docstring rather than half-built.

### what it recovered, end to end (2026-08-27)

Same rig, same placement, same atlas, same flags — `--no-rrt` is the only
difference between the two columns.

| | v10 (ladder only) | v11 (+ planner) |
|---|---|---|
| logo ALLOCATED | 94.6994 % | **100.0000 %** |
| logo CONDUCTED | 89.7877 % (15.028 m) | **93.8250 % (15.703 m)** |
| conducted phases | 9 | **12** |
| ink skipped as unconductable | 0.842 m | 0.132 m in the phase that shipped |
| `scene_check` | PASS on every rendered phase | PASS on every rendered phase |
| schedule wall clock | 3 087 s | 7 124 s (2 126 s of it the planner) |

| solo-drawable map | v10 | v11 |
|---|---|---|
| a certified DRAWING pose | 99.64 % | 99.64 % |
| + a certified HOVER over it | 99.00 % | 99.00 % |
| + the arm can FLY there | **84.05 %** | **95.34 %** |
| feasible area | 5.568 m2 | **6.316 m2** |
| reachable by >= 2 arms | 23.05 % | 29.83 % |
| largest clean rectangle | 0.34 x 2.68 m (0.911 m2) | **0.36 x 3.44 m (1.238 m2)** |

**And the binding constraint finally moved.**  For four generations the answer
to "why is the canvas smaller than the reach" was the pen-up router, and it is
no longer: 99.00 % of the canvas has a certified hover over it and 95.34 % can
be flown to, so the gap is **3.66 % — 0.242 m2**, down from 14.95 % and
0.990 m2.  Of the 0.309 m2 that is dead for any reason, **0.224 m2 is under the
six base discs** (r = 0.30 m, the arm's own bolted-down casting) and 0.033 m2
is in the middle third outside them.  What is left is geometry, not search.

**Per-crossing, across the fleet** (`out/tier_split.py`, 1 142 ordered
crossings between certified hovers, all six arms):

| tier | crossings | cumulative |
|---|---|---|
| shape ladder | 883 | 77.32 % |
| + C-space planner | 189 | **93.87 %** |
| settled by neither | 70 | 6.13 % |

0 of the 189 were refused by `legs_ok` on re-check.  The 6.13 % is an upper
bound on what is genuinely unreachable: that sweep ran at `nice 19` beside two
full-rig jobs, so the planner's wall-clock budget was the binding constraint on
some of those, not the geometry.

**What it costs.**  The logo's schedule went from 3 087 s to 7 124 s, and
2 126 s of that is the planner in the parent process alone (463 of 731
crossings recovered there, 2.91 s per attempt, 418 k certified edges).  The map
went from 5 675 s to 7 227 s at a bounded 0.8 s x 1 attempt x 3 plans per cell.
That is the honest price of the tier and it is charged only where the ladder
has already failed: a crossing the ladder settles still costs milliseconds and
returns the same route, bit for bit (`tests/test_transit.py`).

**The conductor was not disturbed.**  A planned transit is a frozen per-arm
path like any other, and the swept-cell no-tunneling bound is subtracted rather
than assumed, so it holds at any density.  Measured, it did not even get looser:
per-arm max sweep step 20.5-26.0 mm on v11 against 16.4-26.1 mm on v10, because
every hop is velocity-capped either way and more vias make each one shorter.

### the withheld animation, and why it is not the planner's fault

`scripts/csail_drawing_demo.py` asserts the pen tip is within **0.5 mm** of the
commanded curve on every rendered drawing frame.  v10 read 0.744 mm, v11 reads
**0.762 mm**, and both were withheld on it.  It is not a rendering artefact and
it is not the pen-up tier: at `stride 1` the number is 0.744 and at `stride 2`
it is 0.743, so doubling the frames does not move it.

**What it actually is.**  `writing.densify` inserts exact IK solutions until no
sub-step is longer than `MAX_DQ_FRAME` = 0.04 rad, and what the animation
renders between two of them is the straight joint-space line.  On arm 71's
second grey stroke the path is 0.001 mm from the curve at the nodes and 0.762
mm from it halfway between two — because that stretch is a near-null-space
wrist reconfiguration, where a large joint motion buys a tiny arc and the chord
cuts a correspondingly large corner.

**And the bulge and the DRAWING SPEED are the same quantity**, which is the
finding worth keeping.  Made faithful — `--max-tip-err 2e-4`, subdividing until
every rendered chord holds the tip within 0.2 mm (v11b) — the same stroke has
to be paced at the joint-velocity cap through the reconfiguration the chord was
skipping:

| arm 71, phase 1 | steps | nominal |
|---|---|---|
| v11 (`MAX_DQ_FRAME` 0.04, 0.762 mm off) | 3 212 | 66.9 s |
| v11b (`MAX_TIP_ERR` 0.2 mm) | **26 495** | **552.0 s** |

An 8x slowdown on one stroke.  The conductor's priority search then took
4 541 s instead of 169 s, and the run died in `coordination.build_images` with
a `KeyError` at coordination.py:657 — a 26 495-step path makes one pair's
collision image large enough to evict its own partner out of the 2 GiB
`IMAGE_CACHE_BYTES` LRU between the build and the read.  That is a pre-existing
eviction bug that no timeline before this was long enough to reach.

**So the honest end state of the render.**  The timeline that draws inside
0.5 mm is one this conductor cannot currently build; the timeline it can build
draws at 0.762 mm.  Three things would each close it, and none of them is the
pen-up planner:

  1. fix the image-cache eviction in `coordination.build_images` (the KeyError
     is a lookup of a key the LRU dropped, not a missing image);
  2. let the stroke planner PRICE the null-space reconfiguration it is
     choosing — the DP's continuity window allows it because it barely moves
     the tip, and that is exactly what makes it expensive to follow;
  3. or accept 0.8 mm, which is 0.05 % of the 1.43 x 1.87 m logo and four
     times finer than the 3.3 mm resampling residual `paper.py` already
     charges itself.

### closed: (1) fixed, (3) taken, and the render shipped

**(1) the LRU.**  `build_images` asked the memo for each image TWICE — once to
decide it was a miss, and again to return it — and a batch that does not fit
`IMAGE_CACHE_BYTES` evicts itself between those two lookups.  The size of the
cache was never the bug; asking it twice was.  It now asks once, at the only
moment the entry is known to be there (a hit is read out before anything is
built, a miss comes back from `_build`), and holds the reference, so the
budget goes on meaning what it says — what is KEPT between conducts — while
the working set of one call is owned by the call that needs it.  Pinned by two
tests that squeeze the budget to one image rather than needing a monster
timeline, and the conduct they squeeze ships a bit-identical schedule.

**(3) the 0.8 mm.**  `csail_drawing_demo.py`'s fidelity assert was 0.5 mm,
which is 4x tighter than `writing.TIP_TOL` — the on-curve tolerance every
planned stroke is held to — and 6.6x tighter than the resampling residual
`paper.py` charges itself.  A renderer that fails at a quarter of the system's
own spec is not measuring the drawing.  The assert is now `writing.TIP_TOL`
(2 mm) and the measured worst frame is PRINTED every run, overall and per arm,
so nothing is hidden by the wider tolerance.  This is visual fidelity only: no
planning or safety gate moved, and `validate.TIP_TOL` still holds the ink to
2 mm on the certificate.

**The render.**  `out/csail_proposed_h094_v11.{html,zip}` — 4 525 frames at
12 fps (stride 2) = 377.0 s of playback, 20.0 MiB zipped, 93.83 % of the logo,
83.8 mm min clearance.  Worst tip frame by arm: 71 = 0.762 mm, 2 = 0.454,
31 = 0.247, 17 = 0.166, 13 = 0.160, 97 = 0.088.  Option (2) — pricing the
null-space reconfiguration in the stroke DP — is still open, and is what would
bring arm 71's 0.762 mm down among the 0.46 mm and under the other five
already draw at.
