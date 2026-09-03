# aris_sixarm

Motion analysis & planning for the **Aris Kindt** installation. This repo
consolidates the validated kinematic conventions, the fleet model, the
reachability/controllability atlas, and the multi-arm stroke planner —
extracted from the sprawling upstream `Aris_Kindt` branches into one clean,
tested codebase.

**THE DEFAULT RIG IS THE FINAL 3-ARM INSTALLATION** (2026-08): arm 13 upright
on the table, arm 31 hanging under the frame's central beam, arm 2
side-mounted with J1 horizontal, drawing a 1.80 × 1.70 m paper web inside a
steel cage that the planner now carries as verified collision geometry.
Every number and its provenance: `docs/FINAL_RIG.md`; machine-readable:
`aris_sixarm/rig_final.py`; URDFs: `assets/final_rig/`. The six-arm layout
this README's historical numbers were earned on lives on as
`fleet.FLEET_SIXARM` (regression-pinned); prose below that speaks of six
arms describes that era.

**THE REAL INSTALLATION IS TWO OF THOSE UNITS, MIRRORED** (2026-08-21) — six
arms, ONE continuous 1.8034 × 3.63064 m canvas across the seam, and both side
poles lengthened 20 cm so arms 2 and 97 hang at canvas z 0.576. It is a named
config, not the default, and one environment variable installs it everywhere:

```
ARIS_RIG=final6_opt python3 scripts/run_atlas6.py
ARIS_RIG=final6_opt python3 scripts/csail_schedule.py --arms all ...
```

`docs/MERGED_CANVAS.md` is that rig — the config, the six-arm atlas
(union strict-GO **75.93 %**, seam strip **88.00 %** with **74.91 %** of it
reachable from BOTH units), and what the CSAIL pipeline does on it. Every
output flags the two ~20 cm pole extensions, which are **not drawn steel**.

`docs/PAPER_PLANE.md` is the correction to it. The first `csail_final6` release
put three pen-up transits **through the canvas** — worst pen tip 253.6 mm under
the paper, worst wrist 156.1 mm, one of them held there for three quarters of a
second by a conductor pause — because the paper was certified against every
stroke and against no transit. It is now an obstacle at both layers: pen-up
moves are routed around it with certified via-configurations
(`aris_sixarm/paper.py`, priced through `sequence.cost_matrix`), and
`scene_check` gates every arm's tip and chain against it at every instant. The
same document has the dead-zone tiers (the atlas overstates death by **7.04 pp**)
and the attribution of the 86.97 % coverage.

**AND THE PHYSICAL MODEL OF ALL OF IT IS `assets/system_model/`** (2026-09-01)
— the 80/20 cage off the original drawing, the table and paper under it, six
FR3 arms, a pen holder on every hand, visual and collision, with the drawing
entity or code constant behind every one of 77 static bodies in
`model_manifest.json`. `docs/SYSTEM_MODEL.md` is that model.

It carries a **datum correction**. The drawing's one overall height, 233,7 cm,
is FLOOR to top-of-construction — it draws a self-supporting cage and defines
no room ceiling anywhere — and the paper sits 636.68 mm above that floor. So
`mounts.MOUNTS.ceiling_z = 2.34 m` is that number re-datumed to the paper, and
it stands the cage **716.4 mm** too tall. Too tall is conservative for
collision, so no certified number is in question and **`mounts.py` is not
changed**; it is not conservative for a fabricator, who is being asked for a
1435.0 mm drop post where the corrected datum asks for 718.6 against the
736.9 that was actually built. All seven disagreements are in
`system_model.reconciliation()`, and the two that block fabrication are the
drop cluster (a 393.8 x 152.4 post cluster against a modelled 200 mm column)
and a hole nobody has cut: the manufacturer's link0 visual carries the base
connector **230.7 mm** past the flange, which on an inverted arm points
straight up through the 12.7 mm plate and the 95.7 mm clamp stack, and the
collision shell stops at the flange so nothing here has ever seen it.

The arms' **unaudited collision spheres are gone** — `installation.urdf` uses
the manufacturer's own shells and `installation_capsules.urdf` the audited
capsule set — and the **textures are really there**, vendored byte-identically
from where the meshes came from, so the thing renders like the robot it is.

Since 2026-09-02 there is a fourth URDF, `installation_fatfingers.urdf`
(`gen_system_model.py urdf --fingers stock|fat|both`,
`SM.urdf_path(fingers="fat")`): the printed 90 mm **"Fat Franka Finger"**,
whose mesh finally arrived, in place of the stock one. It is drawn in the FR3
fingertip's own CAD frame — its plate hole and that tip's brass-insert axis
agree to **0.0002 mm** — so its place on the hand is fixed with a 0.155 mm
residual and no free parameter. Two things fall out and both are in
`docs/SYSTEM_MODEL.md` §7d: **nothing on it locates the pen holder's post**, so
the pen's lean is set by hand at grasp time and 45° and 23° can both be true;
and a 90 mm blade reaching 69 mm sideways out of the hand **escapes
`selfcoll.BODY_CAPSULES` by 49.93 mm** where the stock finger is contained.
Reported, not fixed — no capsule radius, no gate constant and no tool
transform is changed.

## Setup — the repo's own venv

Everything in this README that says `python3` wants numpy, scipy, trimesh and
pytest. The repo carries no vendored environment; make one:

```
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
```

`.venv/` is gitignored. The drake-only scripts (`check_system_model.py`,
`render_system_model.py`, `scene_check`) still want the station venv at
`/home/franka/git/franka_manipulation_station/.venv`, which is why they name it
explicitly.

## Layout

```
aris_sixarm/
  frames.py        FR3 DH model, FK, joint/torque limits, pen transform  (single source of truth)
  fleet.py         the 6-arm world layout (mounts, base transforms, per-arm colors)
  system_model.py  THE PHYSICAL INSTALLATION: the 80/20 cage, table and canvas
                   as dimensioned, provenance-classed bodies; the corrected
                   ceiling datum; the reconciliation queue against mounts.py
                   and the open physical questions  (docs/SYSTEM_MODEL.md)
  ik.py            analytic IK wrapper (He/Liu solver via wernerpe bindings; FR3-filtered)
  metrics.py       joint margin, pen-tip Jacobian, sigma_min, f_max
  atlas.py         reachability sweep over the paper plane
  planner.py       single-arm stroke planner: ladder-graph DP over (s x q7 x branch)
  pwl.py           the same strokes as a piecewise-linear q7(s): IK sheets,
                   clearance in the (s, q7) band, RDP knots, IK back-out
  smooth.py        C1 corner-rounding of that polyline (Bézier blends on the graph
                   q7(s)), certified by IK chase with window bisection
  pacing.py        TOPP-lite: constant tip speed, slowed only where a joint
                   velocity limit would be hit  (frames.QD_MAX, from the FR3 URDF)
  stroke_api.py    THE entry point: plan_stroke() -> ok | split | degenerate | bug
  validate.py      independent re-derivation of every invariant from raw outputs
  trace.py         raster art -> pen strokes: colour unmixing, Zhang-Suen thinning,
                   skeleton-graph tracing that continues STRAIGHT THROUGH crossings,
                   boundary contours for solid glyphs  (numpy/PIL only).
                   §7 `trace_art` does the same for ANY picture — the ink count
                   is measured (`detect_inks`), the working resolution is fixed
                   rather than upsampled, and the line/fill threshold is
                   measured off the picture's own skeleton (`auto_fill_erode`);
                   §8 `trace_svg` flattens a vector source into the same shape
  artwork.py       where a traced picture goes on the paper and how big: the
                   atlas-proxy + real-allocation search and the "largest within
                   1 pp of the best coverage" rule, shared by csail_place.py
                   and draw.py; plus the ink-name -> colour palette
  allocate.py      strokes x arms -> certified per-arm programs: interval probing
                   via plan_stroke, pen-colour partition enumeration, greedy
                   interval cover, mid-overlap handoff cuts, dropped-span report;
                   v2 balances on SECONDS by move | swap | SPLIT, cutting a span
                   on the busiest arm and handing one piece to an arm that
                   certifies it; active_override = a hypothetical fleet, never
                   a registry edit; and `merge_remainders`, which lets an arm
                   draw a span AND the ink lying against it as one segment —
                   the move that took the logo to 100 % (docs/FULL_COVERAGE.md)
  bench/           five generated drawings the pipeline is regression-tested on,
                   none of them the CSAIL logo  (docs/BENCH.md)
  coordination.py  CONDUCTOR v1: capsule model, pairwise collision images over
                   progress indices (swept cells, no tunnelling), priority
                   pause-scheduling by exact reachability DP.  Only the clock moves.
  idle.py          what an arm does when it stops drawing: freeze in place (the
                   default) instead of going home, a certified minimal retreat
                   for a frozen pose that is genuinely in the way, and slow
                   just-in-time taxiing out of measured slack.  docs/IDLE.md
  scene_check.py   independent re-derivation of a MERGED multi-arm timeline:
                   all-pairs clearance incl. between-sample sweeps, monotone
                   progress, validate_plan per segment.  It has the veto.
  viz/             drake-mesh robot model + static meshcat scene builder
scripts/
  run_atlas.py     sweep all/selected arms  (~35 s for all six)
  make_scene.py    build out/reach_atlas.html (static meshcat + legend)
  fuzz_planner.py  seeded parallel fuzz campaign + atlas cross-check
  csail_trace.py   trace the CSAIL logo -> out/csail_trace.png + masks + strokes.json
  csail_allocate.py  trace + allocate + all three outputs (~3 s end to end)
  csail_place.py   placement search: scale x translation, atlas proxy then real
                   allocations, "largest size within 1 pp of the best coverage"
  csail_schedule.py  allocate -> freeze timelines -> conduct -> validate -> npz
  csail_drawing_demo.py  drake/meshcat playback of that npz (venv python3.10).
                   Reads the ink names and colours out of the payload, so it
                   replays a one-ink drawing or a three-ink one unchanged
  draw.py          THE GENERIC FRONT DOOR: any raster or SVG -> trace -> place
                   -> allocate -> conduct -> scene_check -> animation, every
                   output named by --out.  Plans nothing of its own; every
                   stage is the call the csail_* scripts make  (docs/ANY_PICTURE.md)
  solo_time.py     read-only: how much of the run is one arm drawing alone, and
                   whether anybody else certified the span (docs/SOLO_TIME.md)
  bench.py         the standing regression: the whole pipeline over five
                   generated drawings that are NOT the logo (docs/BENCH.md)
tests/
  test_gates.py    real-touchdown validation gates (run these after ANY kinematics change)
  test_planner_robustness.py   regressions distilled from the fuzz campaign
  test_csail.py    tracer / allocator / conductor / idle-policy regressions
  test_sequence.py per-arm sequencing: reversal, cost model, exact + fallback
  test_tilt.py     pen tilt: the equivalences that make it safe to ship flag-gated
  test_validate_cone.py  the validator's pen-cone gate, and that its DEFAULT is
                   the perpendicular pen every pre-tilt plan was asked for
  test_merge_spans.py  the segment merge, and the two spans that used to be
                   the last 59.9 mm of the logo
  test_draw.py     the generic front door: ink counting, the line/fill split,
                   the vector branch, and a whole picture end to end
  test_balance.py  the incremental improvement loop: the sliced cost matrix,
                   first-improvement against the exhaustive scan, the phase
                   floor, what the four profiles may share, and which one is
                   certified first  (docs/FAST_PLANNING.md)
docs/
  DECISIONS.md     every number the upstream repos disagree on, and what we picked
  CONCURRENCY.md   where the 148 s of pause came from, and what each lever bought
  IDLE.md          the idle policy: freeze / retreat / just-in-time taxi
  SOLO_TIME.md     how much of the run is one arm alone, and why
  BENCH.md         the standing regression table over five non-CSAIL drawings
  TWO_PASS.md      one pen swap, both inks for every arm: 86.97 % -> 99.3907 %
  TILT_EXPLORATION.md  pen tilt as a planning axis: what it buys, what it costs
  FULL_COVERAGE.md the last 59.9 mm -> 100.0000 %, and why the fix was the
                   allocator rather than the pen tilt everyone expected
  ANY_PICTURE.md   the generic front door: the three CSAIL-specific assumptions
                   in the tracer and what measured them instead, and the
                   Trollface run end to end
  FAST_PLANNING.md why a picture used to take an hour to become a certified
                   programme and now does not: the balancer priced every
                   candidate from scratch, and 94 % of a cost matrix is
                   `paper.route`.  Time to the FIRST certified programme
```

## Key facts

- **Pen**: no CAD model exists anywhere. tip = hand-TCP + **0.110 m** tool-z
  (gate-validated on the rig). IK targets the tip, not the flange.
- **LATERAL HOLDER (2026-08-25, env-selectable, NOT default; RE-SPECIFIED
  2026-09-03)**: the real holder offsets the pen along hand x — tip =
  TCP + R @ (`PEN_LAT_HOLDER`, 0, `PEN_EXT_HOLDER`) — which BREAKS the
  yaw==q7 degeneracy: tool yaw phi becomes a real redundancy DOF
  (`aris_sixarm/lateral.py`, coarse 8-phi ring + coupled rescue lattice).
  `ARIS_TOOL=lateral` switches planner, atlas, validator, capsules and router
  together, and since 2026-09-03 it switches **both halves** of the tool
  rather than only the lateral one. Both constants are **0.0588421 m** — the
  45° lean the hand forces, out to the tip a photo of the real gripper shows
  (~5 cm below the Fat blades' plates). They were 0.110 / 0.110 and were
  **never gate-validated**; only the inline pen's axial 0.110 is
  (docs/DECISIONS.md, docs/SYSTEM_MODEL.md §7e). The reach numbers below were
  measured at the OLD pair: inverted-arm strict-GO +51 % on its patch, GO
  radius 0.75 -> 0.86 m, strokes certify in ~80-280 ms
  (`scripts/lateral_eval.py`) — re-run pending.
- **IK**: raw `_franka_ik.solve_ik(T16_column_major, q7, seed)` where T16 is the
  **hand-TCP** pose. The `.so` hardcodes Panda limits → FR3 limits re-filtered in python.
  Build lives in `../franka_analytical_ik` (env `ARIS_FRANKA_IK_PATH` to relocate).
- **Strict-GO gates** (field-validated): joint margin ≥ 0.30 rad AND σ_min(pen-tip
  position Jacobian) ≥ 0.14. σ_min is the binding constraint — libfranka zeroes the
  external-wrench estimate near singularities, so low σ_min = force-blind. Torque
  capacity (`f_max`) is ≥ 72 N everywhere GO; materials need ≤ 14 N.

## Planner

- **Lattice**: over a resampled stroke the fiber at each arc-length step is
  (q7 × IK branch) — every analytic-IK solution with margin ≥ 0.15 and
  σ_min ≥ 0.08 that clears paper and boom. Edges join consecutive steps within
  ±1 q7 index and ‖Δq‖∞ ≤ 0.35 rad, so the graph is a DAG in s and one DP sweep
  is globally optimal (the lookahead diffIK lacks).
- **No yaw axis**: with the pen vertical, R = rotz(yaw)·rotx(π) puts joint 7 on
  the pen axis, so (yaw+δ, q7+δ) is the *same* arm — a (yaw × q7) grid aliases
  into diagonal bands a ±1-index window cannot follow. yaw is pinned to 0.
- **Objective** `maximin_sigma` (default) maximizes the *worst* σ_min along the
  whole path (ties → least ‖Δq‖²): one force-blind step ruins a stroke, so the
  bottleneck matters, not the average. `additive` keeps the weighted-sum variant.
- **Splits**: when the band disconnects the forward pass reports the farthest
  reachable s* — the natural cut for reallocation / pen-up transit.
  `scripts/demo_stroke.py` shows both (rim arc planned end to end where greedy
  dies at s=0.015; under-base line split at s*=0.370).
- **PWL in (s, q7)** (`pwl.py`, `scripts/demo_pwl.py`): plan the redundancy as a
  polyline q7(s) *before* solving for joints — label IK sheets by continuity,
  maximin-σ with a clearance tie-break down the middle of the band, simplify to
  knots that a case-consistent IK chase certifies, then back out joints by
  re-solving IK at 5 mm (never interpolating q). The 1.56 m rim arc becomes
  **2 knots** instead of 131 steps, at higher σ_min and 7× less joint travel;
  see `docs/REDUNDANCY.md`.

## Reliability

- **Certified-or-split contract** (`stroke_api.plan_stroke`, the single entry
  point): every call returns `ok` (dense joints + clock + a passed validation
  report), `split` (`s_star` = the arc length actually *certified*, `head` = the
  certified plan for it, `s_reach` = where the band gives out), `degenerate`, or
  `bug` — it never raises, and a plan that fails its own validator is a `bug`.
- **Independent validator** (`validate.py`): re-derives tip-on-curve < 2 mm,
  margin ≥ 0.15, σ_min ≥ 0.10, ‖Δq‖∞ ≤ 0.35, paper/boom clearance and
  |dq/dt| ≤ `QD_MAX` from `frames.fk` and the raw joint samples — no planner
  bookkeeping is trusted, and it returns violations rather than raising.
- **Fuzz** (`scripts/fuzz_planner.py`, seeded, multiprocessing): 900 strokes
  (6 arms × 6 generators × inside/straddle/wild) in ~60 s. Two clean runs:
  591/190/119 and 563/213/124 ok/split/degenerate, **0 bugs**, 0 validator
  rejections, 0 nondeterminism (10 % replanned and compared bitwise), every
  split's head re-planned from scratch, atlas strict-GO plan rate 100 %.
- **Bugs it found**: the analytic IK *clamps* at the workspace boundary —
  q2 = 0.0 exactly, NaN-free, inside all limits, for a pose it misses by 2 cm —
  so every solution is now FK-verified in `ik.py`; and the fixed-step resample
  left the last < 1 lattice step of every stroke outside the lattice, planned on
  faith (`stroke_api` now fits a whole number of steps into the length).
- Regressions live in `tests/test_planner_robustness.py` (16 tests, ~13 s).

## Writing demo — "ARIS"

`letters.py` holds polyline letterforms (A R I S, 6 strokes total, single
width); `writing.py` plans every stroke with the DP planner, adds the pen-up
transits and time-indexes the whole word; `scripts/aris_writing_demo.py`
renders it in drake + meshcat to `out/aris_writing.html` (self-contained, with
drake's playback controls) and `scripts/aris_letters_png.py` draws the layout.

- **Placement is reach-limited.** The inverted arms only reach r ≈ 0.72 m on
  the paper, so R and I had to be nudged 0.10 m back toward their bases; the
  fleet's front/back base rows then put those letters on different baselines.
  The nudge search (±0.15 m along the base→centre ray, then a 0.30 m fallback
  height) is automatic and reported.
- **Transits are a PLACEHOLDER, not a plan**: lift 0.06 m, linear joint
  interpolation, descend. No collision reasoning — roadmap item 3 still wants
  the RRT.
- **Animation ≠ planned steps.** The DP spends its ±0.35 rad continuity budget
  on q7 self-motion (q1/q3 counter-rotate, the tip barely moves). Those
  configurations are not collinear in joint space, so interpolating them for
  frames threw the tip 7.7 mm off the paper. `writing.densify()` re-solves the
  case-consistent IK along each segment, which puts it back to 0.19 mm.
- Run the demo with the pydrake venv python (the system python3 has no drake);
  `ik.py` falls back to the installed `franka_analytical_ik` wheel there.

## CSAIL logo — image to fleet (`trace.py` + `allocate.py`)

`python3 scripts/csail_allocate.py` takes `assets/csail/csail_old_med.gif`
(212x162) to certified per-arm programs in **3 s wall clock**.

- **Tracing.** Colours are *unmixed*, not thresholded: every pixel is a
  mixture `white + a_grey*(grey-white) + a_orange*(orange-white)` and the 3x2
  least-squares solve gives both coverages, so an antialiased orange pixel and
  a solid light-grey one are separated by hue. Masks upsample 4x, thin by
  Zhang-Suen, and the skeleton graph is traced with **straight-through
  junction pairing** — at each node the incident branches are matched by turn
  angle, so an X crossing leaves as two strokes that pass through each other
  instead of four stubs. Result: **39 strokes, 21.53 m**, median 0.54 m; the
  same skeletons cut at every junction give 80 fragments instead of 33 outline
  strokes. Endpoints facing each other across a gap are re-joined too — the
  orange art is painted over the grey, so every grey line passing under an
  orange one is interrupted by exactly one stroke width of white.
- **The wordmark is not skeletonised.** Solid glyphs are separated from line
  art per connected component (how much survives 6 erosions: >45 % for a 31 px
  letterform, 0 % for an 11 px line) and traced as boundary contours on the
  sub-pixel coverage field, so C S A I L read as letterforms with the A's
  counter intact.
- **Size is height-bound.** The mark's 1.309 aspect makes the requested 3.3 m
  width need 2.52 m of paper height; the sheet has 1.961. Keeping the aspect,
  the biggest honest logo is **2.411 x 1.841 m**, centred, 0.06 m margin.
- **Allocation.** Each (stroke, arm) is probed with up to three `plan_stroke`
  calls — forward, reversed, and one on the largest remaining gap — which buys
  certified s-intervals; per stroke a greedy interval cover takes the fewest
  pieces; the pen-colour split of the four arms is chosen by enumerating all
  **14 non-trivial partitions**. The atlas is a lossless pre-filter here: 72
  probes instead of 308, same allocation.

**52 % of the logo is left empty, and that is the fleet, not the logo.**
With arms 2 and 71 parked the four active arms reach only ~59 % of the traced
path (floor arms 13/17 are mounted off the long edges and reach r <= 0.81 m;
the inverted pair are annuli of r 0.17-0.72 m around (1.30, 1.63) and
(2.30, 0.35), leaving a hole through the middle of the sheet). Scaling the
logo down does not fix it — at 0.55x the reachable fraction is 64 %, barely
better than 59 % at full size, and shifting it +-0.3 m changes it by <3 points.
Arm 17 reaches none of it and draws nothing.

| arm | pen | strokes | metres | pieces | transit |
|---|---|---|---|---|---|
| 13 front (floor) | grey | 2 | 0.58 | 2 | 1.30 |
| 17 back (floor) | grey | 0 | 0.00 | 0 | — |
| 31 L-inv-front | grey | 8 | 3.64 | 9 | 2.61 |
| 97 R-inv-back | orange | 11 | 6.11 | 13 | 4.14 |

10.33 m drawn of 21.53 m; 32 dropped spans (18 whole strokes). Every shipped
segment is a re-planned, independently validated `ok` plan (min sigma 0.167,
tip error ~1e-11 m). Handoff cuts are placed mid-overlap and unit-tested, but
**no stroke on this input is actually shared between two arms** — with two
arms parked, no two same-colour arms overlap on any one stroke.

Outputs: `out/csail_trace.png` (traced strokes at sheet scale),
`out/csail_masks.png` (masks / skeletons / trace over source),
`out/csail_allocation.png` (coloured by arm, dropped dashed),
`out/csail_scene.html` (static meshcat, arms posed mid-stroke),
`out/csail_program.json` (per-arm segments with plan metrics + dropped list).

Single-arm-sequential throughout: this is allocation plus a certified plan per
segment. No multi-arm timing, no inter-arm collision reasoning, and the pen-up
transits are measured but not planned — their ORDER and their direction are
chosen for minimum transit time (`sequence.py`, below); the motion between two
strokes is still a straight joint-space line.

## All six arms, conducted (`csail_place` → `csail_schedule` → `csail_drawing_demo`)

The run above is today's rig. This one is the hypothetical **"all six arms up"**:
`allocate(..., active_override="all")` — a parameter, never an edit to
`fleet.FLEET`, whose `active` flags stay the record of which arms are actually
up (`test_active_override_does_not_touch_the_registry` pins that).

**Placement is worth more than any planner change here.** `csail_place.py`
scores 7 scales × a ±0.3 m translation grid, first with an atlas proxy
(`allocate.reach_fraction`, 10 ms) and then — because a real allocation is only
3 s — with the real thing, 133 of them in 71 s. Best translation per scale:

| scale | width | best offset | drawn |
|---|---|---|---|
| 1.00 | 2.41 m | (−0.20, 0.00) | 74.9 % |
| 0.95 | 2.29 m | (0.00, 0.00) | 76.6 % |
| 0.90 | 2.17 m | (+0.10, 0.00) | 75.1 % |
| 0.85 | 2.05 m | (+0.10, −0.10) | 81.9 % |
| 0.80 | 1.93 m | (+0.30, −0.10) | 81.7 % |
| **0.75** | **1.81 m** | **(0.00, −0.20)** | **86.3 %** |
| 0.70 | 1.69 m | (0.00, −0.20) | 85.6 % |

Rule: the largest size within 1 pp of the best coverage anyone achieved → **0.75×,
1.808 × 1.381 m centred at (1.804, 0.780)**. Six arms at full size already lift
coverage from 47.9 % to 74.9 %; shrinking to 0.75× buys the rest.

**86.2 % drawn (13.94 m of 16.15 m), 2.22 m left empty in 13 spans.** All of the
residue is in the lower-middle of the sheet — the bottom two thirds of the
centre column — where the four inverted annuli still do not meet; the top third
is now completely covered.

| arm | pen | segments | metres | pen-up | pause |
|---|---|---|---|---|---|
| 2 R-wall-front | orange | 12 | 4.53 | 16.5 s | 0.0 s |
| 97 R-inv-back | orange | 11 | 4.83 | 14.6 s | 17.5 s |
| 31 L-inv-front | grey | 10 | 2.76 | 11.4 s | 2.1 s |
| 71 L-inv-back | grey | 5 | 1.82 | 9.6 s | 13.8 s |
| 13 front (floor) | grey | 0 | 0.00 | — | idle |
| 17 back (floor) | grey | 0 | 0.00 | — | idle |

**Four arms draw it, not six, and that is geometry.** Arms 13 and 17 are on
opposite long edges 3.83 m apart with r ≤ 0.81 m; no placement of a 1.8 m logo
is in reach of both. Pushing the logo +0.7 m to put arm 17 to work costs arm 13
and 7 points of coverage (78.9 % with five arms drawing). The best coverage wins
per the search rule, and the two floor arms stand in the scene holding their
ready pose.

### Which segment next, and which way round (`sequence.py`)

An arm's segments used to be chained nearest-neighbour in **paper distance**,
each drawn the way it happened to be certified. The arm pays neither of those.
It pays a hover transit — lift, joint-interpolate between two hover poses at
≤ 60 % of the FR3 velocity limits, lower — and that is now computed in exactly
one place (`writing.transit_time`) which both the frozen timeline and the
sequencer read. `csail_schedule.py` re-checks the two against each other every
run and refuses to render if they disagree by more than 1 µs (measured
disagreement: **1e-14 s**).

- **A certified segment is direction-agnostic.** Every gate `stroke_api`
  enforces — tip on curve, joint margin, sigma_min, continuity, paper and boom
  clearance — is a property of a configuration or of an unordered pair of
  neighbours, so none of them can tell which way s runs.
  `stroke_api.reverse_plan` is therefore array flipping (`qs`, `pts`, `q7`,
  sigmas, margins, knots reflected to 1 − s) with the clock re-integrated by
  `pacing.pace` from the reversed samples — and the reversed plan goes back
  through the independent validator, so it carries its own certificate rather
  than inheriting one. **18 of the 38 segments come out drawn backwards**, and
  no reversal was refused.
- **Exact, not greedy.** Each segment offers two nodes (forward, backward) of
  which exactly one must be visited; Held-Karp over (subset × last segment ×
  last direction) solves it outright — 2¹² × 24 = **98 304 states in 0.04 s**
  for the busiest arm. The exact ceiling is 16 segments (2.1 M states, 0.76 s).
- **Above 16, a heuristic on the same matrix** — the full (2n+1)² transit-time
  matrix is built as array operations, so every improvement move is an O(1)
  lookup: nearest-neighbour seed, then 2-opt and Or-opt passes whose move set
  includes direction flips, then double-bridge kicks until a 2 s budget runs
  out. The 2-opt deltas are written for an ASYMMETRIC cost (reversing a block
  turns every segment in it round, which need not be free) and carry a prefix
  sum of the per-edge reversal penalty; a test checks each delta against a full
  re-evaluation. A synthetic **300-segment** arm: 15 % better than its greedy
  seed, inside the budget.
- **Per-arm pen-up time 78.4 s → 52.1 s (−33.6 %)**, and with it the makespan
  **83.1 s → 64.8 s (−22 %)**. Nothing about the certified plans changed: the
  same 38 segments, the same 13.94 m, the same validator.

### Conductor v1 (`coordination.py`) — the clock is the only thing that moves

Each arm's timeline is frozen first (`writing.arm_program`): its segments in the
allocator's order, hover transits between them, entry and exit lifts, densified
so frame interpolation stays on the curve. The conductor may then only stretch
that clock.

- **Capsules.** 7 per arm off `frames.fk`'s own chain points, r = 0.09 (links) /
  0.07 (wrist, hand) / 0.03 (pen). Required clearance = **safety 0.05 m +
  calib 0.03 m**; the second is not about the arm but about *us* — four base XYs
  come from a preset file and have never been surveyed. Drop it the day one does.
- **Swept cells, not sampled points.** Cell (a, b) of a pair's collision image
  is free iff the *whole* cell is: the smallest of its four corner clearances
  minus both half-step motions (clearance is 1-Lipschitz in body displacement).
  Nothing can tunnel between two samples.
- **Pacing is a clearance budget, not decoration.** The first cut of this ran
  transits as 1.2 s joint-space lines and moved the elbow **295 mm between two
  frames** — the sweep slack alone then exceeded the margin four times over and
  the schedule was infeasible. Capping every move at 60 % of the FR3 joint
  velocity limit brings it to ≤ 32 mm, and the problem becomes easy.
- **Schedule.** Priority = busiest arm first; each next arm gets the
  earliest-arrival monotone schedule (advance or wait) by an exact reachability
  DP over progress × time. **33.4 s of pauses** in total, arm 2 — now the
  busiest — none.
- **Pauses went UP when the transits came down, and that is not a regression.**
  Shorter programmes put the arms in the same place at the same time more
  often, so the conductor has more waiting to do (23.2 s → 33.4 s); the number
  that matters, the makespan, still fell 83.1 s → 64.8 s. The priority order
  changed hands with it (arm 97 → arm 2).

### The validator has the veto (`scene_check.py`)

Written as a separate code path on purpose — its own kinematics call, its own
capsule assembly, and a segment-distance routine derived differently from the
conductor's (interior critical point + four endpoint distances, vs the clamped
parametrisation). A unit test holds the two to 1e-9 across the degenerate cases
so they cannot drift apart silently. It re-samples the merged timeline 2× finer,
bounds the gaps between samples, and re-runs `validate_plan` on all 38 segments.

**Minimum inter-arm clearance over the whole 64.8 s: 83.7 mm** (margin 80 mm),
between arms 2 and 97 at t = 28.4 s. 38/38 segments valid, progress monotone,
**PASS** — and the animation is only rendered from a timeline that passes.
(The re-sequenced run re-validates every segment from scratch, reversed ones
included: the 18 backwards plans are checked here a second time, by a code path
that has never heard of `sequence.py`.)

### The animation

`out/csail_drawing.html` — **1556 frames @ 24 fps = 64.8 s**, kinematic playback
(`SetPositions` + `ForcedPublish`, no simulator, no controller), 567 ink chunks
revealed progressively in each arm's own pen colour (#666665 / #cb6608, the
logo's own inks), pen cylinders coloured to match. Pen tip on the commanded
curve to **0.173 mm** on every drawing frame, checked against drake's own
kinematics. 37.2 MiB of HTML, **17.8 MiB zipped** (budget 28). Draw speed
0.15 m/s, transits ≤ 0.8 m/s and joint-velocity-capped.

Reproduce (the last two flags are not defaults — `--qd-frac` is the conductor's
60 % cap, and it is the sequencer's cost model as well as the timeline's):

```
python3 scripts/csail_allocate.py --arms all --tag _6arm \
    --placement out/csail_placement_6arm.json --qd-frac 0.6 --no-scene
python3 scripts/csail_schedule.py --arms all --tag _6arm \
    --placement out/csail_placement_6arm.json --draw-speed 0.15 --qd-frac 0.6 \
    --final out/csail_final.png
/home/franka/git/franka_manipulation_station/.venv/bin/python \
    scripts/csail_drawing_demo.py --schedule out/csail_schedule_6arm.npz \
    --summary out/csail_schedule_6arm.json --out out/csail_drawing.html \
    --zip out/csail_drawing.zip
```

Add `--sequencer nn` to either of the first two for the old paper-distance
chain with every segment drawn forward — the same pipeline, the same
validator, 83.1 s instead of 64.8 s.

**The whole logo** (`out/csail_full.html`, `out/csail_full.zip`,
`out/csail_full_final.png`, and the `_full` schedule the demo reads by default)
is the two-pass run, and every flag it needs is on one line:

```
python3 scripts/csail_schedule.py --arms all --two-pass \
    --pens 2:300,31:200,71:200,97:200 --max-probes 5 --min-coverage 0.99 \
    --target-width 1.2969246423461636 --offset 0.1 0.05 \
    --tag _full --fps 12 --substeps 4 --final out/csail_full_final.png \
    --select-profile --program
python3 scripts/solo_time.py --arms all --two-pass \
    --pens 2:300,31:200,71:200,97:200 --max-probes 5 \
    --target-width 1.2969246423461636 --offset 0.1 0.05
/home/franka/git/franka_manipulation_station/.venv/bin/python \
    scripts/csail_drawing_demo.py
```

`--select-profile` is the last two flags' worth of the argument that used to be
made by hand: it allocates all four EXECUTION PROFILES — `qd_frac` 0.30 or
0.60, fiber menus off or on — conducts them cheapest-floor-first, and ships the
fastest one `scene_check` certifies, recording all four outcomes in the
schedule JSON.  On this logo it ships **qd0.60+cluster at 77.792 s** (82.4 mm
clearance, 99.2139 % coverage) after conducting two cells and proving the other
two could not win; see `docs/BENCH.md`.  `--program` writes
`out/csail_program_full.json` from the allocation that shipped, so the
programme and the schedule cannot describe different runs.  Drop both flags to
conduct exactly the `--qd-frac` / `--cluster` you pass.

`--idle-policy home` on the first of those puts conductor v1's go-home
behaviour back for comparison (`docs/IDLE.md`); `--no-jit` and `--no-retreat`
switch off one piece of the policy each.

`out/csail_final.png` is the end state: ink in pen colours, the 13 unreachable
spans dashed.

## The real installation: six arms, one continuous canvas (2026-08-21)

Two user decisions turned the mirrored layout from a preview into a rig the
pipeline runs end to end — `docs/MERGED_CANVAS.md` is the whole story, this is
the summary.

1. **The canvas is CONTINUOUS across the middle**: `rig_final6.MERGE_WEBS =
   True`, one drawable surface **1.8034 × 3.63064 m** spanning both units and
   the 23.064 cm strip between their webs.
2. **Both side arms are re-clamped 20 cm lower on 20 cm of new pole** (canvas
   z 0.576) — `docs/ARM2_HEIGHT.md`'s optimum, applied to arms 2 and 97.
   **That steel is not in the drawing; every number below assumes it is fitted.**

`fleet.py` now selects among four rigs, and the 3-arm one stays the DEFAULT so
every published number above still reproduces:

```
ARIS_RIG=final6_opt python3 scripts/run_atlas6.py     # or final6 / final / sixarm
```

### The atlas — a real sweep, not a mirror (`scripts/run_atlas6.py`)

All six arms over the whole canvas, 2 cm grid, the standing gates, the 15° tilt
cone, both frames' 62–67 boxes per arm. 16 562 cells. The old symmetry preview
could not answer this: it mirrored one unit's web and reported the seam and the
cross-web region as *reach-bound only, never scored*.

| | as drawn | **extended poles** |
|---|---:|---:|
| union strict-GO | 69.13 % | **75.93 %** |
| ≥ 2 arms | 7.86 % | **25.47 %** |
| cross-unit (an A arm and a B arm) | 3.99 % | **8.46 %** |
| most arms over one cell | 2 | **4** |
| arm 2 / arm 97 | 9.89 % each | **22.47 / 22.39 %** |
| **seam strip** GO | 77.47 % | **88.00 %** |
| **seam strip** ≥ 2 arms | 41.67 % | **77.20 %** |
| **seam strip** cross-unit | 41.30 % | **74.91 %** |

The seam strip is now the best-covered band on the canvas, and three quarters
of it is reachable from *both* units — against the preview's headline of
**0.00 %** cross-unit overlap. Arms 13, 31, 17 and 71 are unaffected by the
longer poles **to the cell**.

### Placement — and the logo turns 90° (`scripts/csail_place.py --rotate 0,90`)

`trace.to_sheet(rotate_deg=…)` turns the logo before fitting it, which matters
because the logo is 1.31× wider than tall and the canvas is 2.01× taller than
wide: the same width limit buys a 90° logo **1.71× the area**. `--base-width
auto` also fixed a silently dead knob — the old constant is the *legacy* sheet's
margin-limited width, so on this canvas every "scale" collapsed onto one logo
(visible in `out/csail_placement_final.json`: seven scales, one width).

The standing rule (largest within 1 pp of the best real coverage) picks **90°,
0.842 × 1.102 m at (−0.20, +0.20), 98.9 % drawn** — 2.1× the area of the best
0° candidate at the same coverage. What binds is the canvas's **short** axis:
0.26 m of the 1.8034 m width is dead at every y (feed roll left, guide rods and
paper curl right), leaving ≈ 1.54 m usable against 3.63 m of length.

### The middle band: redundancy is not concurrency (`scripts/middle_band_diag.py`)

The seam's ≥2-arm coverage is owned by exactly the two pairs that cannot be in
it at the same time. Capsule clearance between the atlas's own certified poses,
margin 80 mm:

| pair | | pose pairs ≥ 80 mm | median |
|---|---|---:|---:|
| **31 vs 71** | cross-unit, both inverted | **33.9 %** | **21 mm** |
| **2 vs 97** | cross-unit, both side | **45.9 %** | 63 mm |
| the four other pairs | | 93.7–95.1 % | 250 mm |

and in each unit's own half the side/inverted pair drops to **83.6 %**, because
the 20 cm drop puts the side arm 20 cm closer to its own neighbour as well as
20 cm further across the seam. **The coverage the extended poles buy and the
contention they create are the same fact.**

It shows: **twelve** pipeline configurations were conducted, four execution
profiles each, and every seam-centred two-pass run was refused — always on
`phase 1`, the pass that must GO HOME so the next can start from the ready
pose. A single pass freezes in place and conducts. Then `scene_check` vetoed
even that, on a 40.9 mm frame clearance during a pen-up: the two frames' back-
left corner posts stand together at the left end of the seam, and an inverted
elbow finds them. Moving the logo 0.20 m right fixed it (56.1 mm).

### The shipped run — six arms, one canvas, ink across the seam

```
ARIS_RIG=final6_opt python3 scripts/csail_schedule.py --arms all --max-probes 5 \
    --rotate 90 --target-width 0.8417 --offset 0.0 0.0 \
    --atlas out/atlas_final6_opt --tag _final6 \
    --draw-speed 0.15 --transit-speed 0.30 --fps 12 --substeps 4 \
    --final out/csail_final6_final.png --select-profile --program

ARIS_RIG=final6_opt /home/franka/git/franka_manipulation_station/.venv/bin/python \
    scripts/csail_drawing_demo.py --schedule out/csail_schedule_final6.npz \
    --summary out/csail_schedule_final6.json --out out/csail_final6.html \
    --zip out/csail_final6.zip
```

0.842 × 1.102 m turned 90°, centred **on the mirror plane**. 38 certified
segments, **86.97 %** of 9.83 m drawn. Of the four execution profiles, three
certified and the fourth was pruned on its own floor; **qd0.60 ships at
50.125 s**. `scene_check` **PASS** — 81.8 mm minimum inter-arm clearance
(margin 80), 65.6 mm frame clearance (margin 50), 38/38 segments re-validated.
Animation: 602 frames @ 12 fps = 50.1 s, pen tip on the commanded curve to
**0.169 mm**, 16.2 MiB zipped (budget 28).

| arm | unit | ink | segments | metres | in the seam strip |
|---|---|---|---:|---:|---:|
| 97 | B | grey | 13 | 2.738 | 0.822 |
| 71 | B | orange | 11 | 2.292 | 0.385 |
| 31 | A | orange | 7 | 2.241 | 0.683 |
| 2 | A | orange | 7 | 1.278 | 0.605 |
| 13 / 17 | A / B | — | 0 | 0.000 | — |

**2.4946 m — 29.18 % of all the ink — lies inside the old 23.064 cm seam
strip, and 6 of the 38 segments (21.44 % of the ink) cross a former web edge.**
On the two-web layout that paper does not exist and neither does that ink; all
four hanging arms contribute, two from each unit. The two floor arms draw
nothing, which is geometry: they sit 1.5 m from any placement worth having.

## The layout study — hang all six, mounts modelled (2026-08-25, v2)

`docs/LAYOUT_STUDY.md`: v1 of this study modelled NO mounting hardware, so its
99.38 % winner was a kinematic ceiling.  v2 makes the schematic mounts
(`aris_sixarm/mounts.py` — booms r = 0.10 up to a 2.34 m grid, 0.226×0.190×0.05
base plates, 0.30×0.25 floor pedestals) **first-class static obstacles at every
stage**, through the same `static_obstacles → atlas.solve_cell(boxes=…) →
chain_static_clearance` plumbing the final rig's frame uses, and adds an
all-ceiling family on a second mandate.

**Re-scored under mounts, v1's winner is 99.28 %** — union −0.10 pp, ≥2-arm
−2.27 pp (50.31 → 48.04 %).  Every cell of that loss falls on the two FLOOR
arms; the four inverted arms lose *identically nothing*.  That is physics, not
modelling slack: a drawing pose hangs its whole chain low (an inverted arm's
tops out 0.33 m *below* its own base plane, a floor arm's elbow at 0.650 m), so
all inverted mount steel at z ≥ h is unreachable — while the floor PEDESTALS
sit at the pen's own working height just outside the web and eat the edge their
neighbours want to draw.

So the configuration changed: **hang all six, on a regular 2×3 ceiling grid at
h = 0.85** — x = 0.5967 / 1.2067 (centre line ± 0.305), y = 0.6051 / 1.8153 /
3.0255 (H/6, H/2, 5H/6), i.e. `layout.paired_grid(0.61, 3, 0.850)`.  Certified
**99.98 %** union strict-GO with mounts active, **55.98 %** ≥2-arm, 13.95 %
≥3-arm, **3 dead cells out of 16 562**, all six certified ready poses clear of
every neighbour's steel by ≥ 0.35 m — against 99.37 % / 47.82 % / 105 dead for
the best 2+4 layout, and 84.91 % for the current rig with the lateral tool
alone (`out/atlas_final6_opt_lat/`).  The mechanism is per-arm coverage: a
hanging arm sits OVER the canvas and spends its whole annulus on paper (28–32 %
each), where a floor arm must stand outside the web and manages 11–16 %.  All
six mounts are then identical and both short edges are free for transport.
Not modelled: inter-arm collision at 56 % overlap, the grid's own members,
transport.  `ARIS_RIG=proposed` selects it (`aris_sixarm/layout.py`);
`out/layout_study.png`, `out/proposed_scene.html`.

**URDFs: `assets/proposed_rig/`** — `installation.urdf` (six inverted arms on
the grid, FR3 joint limits, plates + booms, the lateral pen holder as fixed
links ending in a `pen_tip` frame) and `environment.urdf` (the static geometry
alone).  Generated by `scripts/gen_proposed_rig_urdf.py` from `layout.py` /
`mounts.py` / `frames.py` — no number in the file is typed there.  Verified two
ways: `tests/test_proposed_rig_urdf.py` walks the file's own kinematic chain
and `scripts/check_proposed_rig_urdf.py` (station venv) loads it in pydrake;
both put every pen tip within **1 nm** of `frames`' FK.  The ceiling grid
itself is NOT in the file — its steel is not designed, so `ceiling_grid_ref`
is a visual-only plate marking the level the booms stop at.
`scripts/render_proposed_rig_urdf.py` → `out/proposed_rig_urdf.html`.

**The real pen holder is in it** (2026-08-19 CAD delivery, 8 printed parts,
**no assembly file**): `scripts/extract_penholder22_meshes.py` measures the
housing, decimates it to ~5 k faces, and bakes the inferred hand-frame
placement into `assets/proposed_rig/meshes/penholder22_*_hand.obj`.  The pen
tip moved on 2026-09-03 — `TCP + R @ (0.0588421, 0, 0.0588421)`, from the
photo of the real gripper — and the CAD's one remaining disagreement is
flagged rather than silently reconciled: the housing's **"22 deg" is a
clocking about the mount post and measures 23.00°**, where the transform
implies a 45° lean. That one is now settled *against* the CAD: at 23° the
housing's own barrel would be 11.9 mm inside the manufacturer's hand shell.
§7a.1 of `docs/LAYOUT_STUDY.md` is the full account.
(That account used to say "a grip 55.1 mm behind the nose … 100.5 mm of
graphite", which is the housing mounted **end-for-end**.  Fixed 2026-09-03:
the grip is 30.001 mm behind the CAP, where the pen actually leaves, and the
graphite is **125.6 mm**.  `docs/SYSTEM_MODEL.md` §7c is the correction and
its re-certification — the pen tip did not move, no gate on the certified
programme flipped, and the r = 0.050 tool capsules no longer contain the
holder.)

**Update 2026-09-02 — the assembly exists, and it settles the 23-vs-45.**
`Pen holder cad(1).zip` nests `Natural hold assembly - closed.zip`, which
holds the complete 10° build as an `.SLDASM`.  Resolved, it says there is **no
fingertip cradle** — stock FR3 tips seat 7.000 mm square inside the mount
post's own 18 × 18 mm sockets — so the housing's clocking reaches the hand
undivided, and on that build the angle in the housing's file name **is** the
lean (10.0000°, with the grip centre on the TCP to 0.14 mm).  That verdict —
"the pen leans 23°" — was overturned on 2026-09-03 by a photo of the real
gripper and by the hand itself: **no fingertip is fitted**, so nothing
transmits the clocking, and 55.1 mm of barrel behind the grip does not fit in
the 37.4 mm under the hand at any lean below **35.17°**.  The transform moved
instead — to `TCP + R @ (0.0588421, 0, 0.0588421)`, the tip ~5 cm below the
Fat blades' plates — and it was **never** gate-validated, contrary to what
this file and several others used to say.  `docs/SYSTEM_MODEL.md` §7e.  The same
assembly gave the internal stack, which `assets/system_model/` now draws in
full.  `docs/SYSTEM_MODEL.md` §7a–7b is the account.

## The capsules were wrong, and the package now says so (2026-08-26)

`scripts/collision_audit.py` (commit 5c8d803) put every schematic collision
model in this repo on an instrument — exact triangle-mesh distance (FCL BVH)
between franka's own collision shells UNION the full-resolution visual shells,
posed by `frames.fk`, and the capsules that claim to contain them — and found
that **none of the arm capsules did**.  This commit applies the corrections.

**The seven capsules are measured now, not assumed.**
`coordination.CAPSULES` was three round numbers (0.09 below the wrist, 0.07 at
the wrist and hand, 0.05 for the tool).  Each arm radius is now the old one
plus the inflation the audit measured, rounded UP to the millimetre:

| capsule | was | is | what it is |
|---|---|---|---|
| base column (0,1) | 0.090 | **0.155** | link0's casting |
| shoulder→elbow (1,3) | 0.090 | **0.130** | link1 + link2 |
| elbow offset (3,4) | 0.090 | **0.117** | link3 — the only one nearly right |
| forearm (4,5) | 0.090 | **0.131** | link4 |
| wrist (5,7) | 0.070 | **0.091** | link5 |
| hand (7,8) | 0.070 | **0.104** | flange, gripper body, fingers |
| tool (8,10),(10,9) | 0.050 | **0.050** | validated against the 22° CAD, kept |

The same numbers are restated, on purpose and pinned by tests, in
`rig_final.STATIC_CAPSULES(_LAT)` and `scene_check.RADII(_LAT)`.

**The neighbour base column is two boxes, and neither of them is 0.12.**  The
audit re-derived it as a 12-band cylinder stack: 0.171 m at the plate, waisted
to 0.057 in the middle, 0.129 at the end, running to base z **0.3875** — 54.5
mm past the `d1` the model stopped at — with a **connector and cable stub**
reaching 0.177 m radially and 0.2325 m back UP the base, outside the plate box,
the boom box and every model this package had.  `mounts.column_bands` ships two
bands, and the reason it is two and not twelve is the gate-consistency identity
this obstacle exists for: the conductor's own base capsule is ONE capsule at
`link_r` over the whole span, so a box thinner than `link_r + calib` anywhere
inside that span certifies cells the conductor then refuses.  `column_r` is
still `link_r + calib`; both sides of that identity moved and it still holds.

The boxes stop at the metal, not at the far spherical cap of that capsule.
Containing the cap would mean running them to base z 0.518, and on the real
2 cm sweep of all six arms that costs **7.6 points of union coverage and 12
points of ≥2-arm concurrency at h = 0.940** to model a shape that is not there.
The residual is discharged by measurement instead:
`tests/test_mounts.py::test_no_certified_pose_sits_in_a_neighbours_shoulder_ball`
walks every certified pose in the shipped atlas and checks the conductor's own
criterion directly.

**An atlas is now stamped with the model it was swept under.**
`atlas.model_signature()` goes into every `atlas_arm*.npz`, and
`atlas.is_current()` tells a caller whether the certifications it is holding
were made against the geometry it is running.  Nothing recorded that before,
which is exactly how a directory of pre-audit `.npz` files stayed in `out/`
looking authoritative.  Tests skip a stale atlas loudly rather than asserting
against it.

**What the corrections cost at the adopted h = 0.850**: union strict-GO
**96.69 % → 87.00 %**, ≥2-arm 50.33 → 37.14 %, dead 3.31 → 13.00 %
(`out/atlas_proposed_occ/`).  Arm 17's park pose had to move — at the true
widths the one the old model certified stands 15 mm inside arm 13's corrected
column box — and the **inward** ready pose stopped existing for the middle row
at any radius or hover, which is the "why the bearing is outward" finding
arrived at a second time and much harder.  The URDF's own arm collision
spheres are a third, still-unaudited schematic; `gen_proposed_rig_urdf.py`
now says so at the top of the file.

## Re-certifying the rig at height: the decision package (2026-08-26)

With the capsules corrected, the height question the audit reopened was put
end to end at **h = 0.925 and h = 0.940** against the adopted **0.850** —
parks re-derived, atlas re-swept, URDF regenerated, placement re-searched, the
CSAIL logo re-allocated and re-conducted in every mode the conductor has.

| | h = 0.850 | h = 0.925 | **h = 0.940** | 0.940, pitch 0.65 |
|---|---|---|---|---|
| union strict-GO | 87.00 % | **91.10 %** | 90.70 % | **91.95 %** |
| ≥2-arm | **37.14 %** | 34.43 % | 33.85 % | 34.61 % |
| dead | 13.00 % | 8.90 % | 9.30 % | **8.05 %** |
| σ p1 / median | 0.1513 / 0.2633 | 0.1516 / 0.2511 | 0.1468 / 0.2483 | — |
| σ within 1.25× gate | **2.7 %** | 6.0 % | 7.1 % | — |
| pair clearance, worst / mean | 80.3 / 92.3 % | 82.8 / 93.7 % | **83.6 / 94.0 %** | — |
| best park-vs-ink | 10 mm | 61 mm | **72 mm** | 68 mm |
| placement (100 % live) | 1.263 × 0.966 m, 97.3 % | 1.094 × 0.837 m, 97.9 % | 1.178 × 0.901 m, 97.6 % | **1.347 × 1.030 m, 100 %** |
| 6-mover conduct | refused | refused | refused | refused |
| partner-disjoint | refused | refused | refused | refused |
| solo | **refused** | **refused** | 1 phase, **22.6 %** | 1 phase, **26.1 %** |

**Height helps, and it does not solve it.**  Union coverage gains 4 pp, the
dead area falls by a third, pair clearance improves at every pitch, and the
one number that decides whether anything runs at all — how far a parked arm
can get from every other arm's certified ink — climbs monotonically from
**10 mm at 0.850 to 72 mm at 0.940**.  The conductor asks 80 mm.  At 0.850 and
0.925 the shortfall is total and the rig draws *nothing*; at 0.940 one solo
phase clears and the rig draws 22.6 % of the logo with one arm.

**It is not the margin.**  Re-conducting with the calibration term cut to zero
(margin 0.08 → 0.05, i.e. the day a base survey lands) returns the *same*
programme: same phase, same 22.6 %.  The blocker is geometric.

**It is the middle row.**  A park's radius saturates — `certified_ready_pose`
clips the hover point into the sheet, so past ~0.55 m every candidate lands on
the same clamped xy — and its hover saturates too, because the middle-row arms
certify nothing above 0.40–0.50 m.  Arms 31 and 71 sit over the middle of a
1.80 m-wide canvas and have nowhere to stand that is not inside somebody's
drawing.  The conductor says so in as many words: *the impossible indices are
INK: arm 13 segment(s) [11] — no order fixes that, only a different
allocation.*

**Shipped**: h = 0.940, pitch 0.61 unchanged, `ARIS_RIG=proposed`.  One
certified phase (arm 31, 10 segments, 2.388 m), makespan 37.6 s, **zero
conducted pause**, scene_check PASS, min inter-arm 146.2 mm, column 144.1 mm,
frame 51.2 mm, paper chain 108.4 mm.  `out/csail_proposed_h094.{html,zip}`.

**What would actually move it** — item 1 is DONE and the section below is what
it found:
1. ~~**Park-aware allocation.**~~  Shipped (`allocate.ParkProbe`), together
   with the banded base column that made the parks clear the ink at all.  It
   did not unlock a second arm: the refusal moved from the ink to the pen-up
   transits, one layer down.  See the next section.
2. **Pitch 0.65** — inside the window the raised annulus opens (0.26–0.73 m),
   worth +1.25 pp union, a 31 % bigger logo at 100 % allocation, and 26.1 %
   conducted.  A build-sheet number, so the rig owner's call.
3. **Fewer arms over the middle**, or a wider canvas.  Two rows of two would
   give every arm an edge to park over.

## The column has a waist, and the refusal moved one layer down (2026-08-26)

Two evidence-backed changes, then the whole pipeline again at h = 0.940.

### 1. The base column is four measured bands, not one flat capsule

The mesh audit's own 12-band stack says the body is 0.171 at the plate,
**waisted to 0.057** through the middle third and 0.129 where link1's swept
solid runs past the shoulder.  The package shipped two bands flat at 0.185
because the *conductor's* base column was one capsule at `link_r`, and a box
thinner than that anywhere inside the span would certify cells the conductor
then refuses.  Both ends moved together: `coordination.BODY_BANDS` is the
profile, the base capsule became four **sub-segment** capsules of the same
pose-invariant axis, `mounts.column_bands` is the same four one `calib` wider,
`scene_check.COLUMN_BANDS` restates them a third time — and the
gate-consistency identity holds *per band* and is now airtight (each box is its
band's AABB grown by the band's radius along the axis too, so it contains that
band's capsule caps included).

| base z | | box was | is | measured max |
|---|---|---|---|---|
| −0.2325 … 0.0667 | connector | 0.207 | 0.207 | 0.1769 |
| 0.0667 … 0.0988 | taper | 0.185 | **0.148** | 0.1172 |
| 0.0988 … 0.2590 | **the waist** | 0.185 | **0.108** | 0.0779 |
| 0.2590 … 0.3875 | link1 sweep | 0.185 | **0.160** | 0.1295 |

Edges chosen by exhaustive search over the audit's own band edges, minimising
the obstacle's cross-section over the slab a drawing arm's links reach: 1 band
0.0118, 2 bands 0.0075, **3 bands 0.0053**, 4 bands 0.0051, 5 bands 0.0050,
against 0.0100 for the flat pair.  On the real 2 cm six-arm sweep:

| h = 0.940, pitch 0.61 | flat 2-band | **banded 4-band** |
|---|---|---|
| union strict-GO | 90.70 % | **92.02 %** |
| ≥2-arm | 33.85 % | **36.35 %** |
| dead | 9.30 % | **7.98 %** |
| cross-unit | 23.16 % | **26.10 %** |
| seam strip, ≥2 arms | 0.00 % | **1.83 %** |
| best park-vs-ink | 72 mm | **97.3 mm** |

The park set was re-derived against it over 7 radii × 10 hovers × **12
bearings** per arm.  The bearing is what bought the last 22 mm: held to the
outward ray the same search tops out at 75 mm, because the middle row's
outward ray runs off the short edge of a 1.80 m canvas and the hover point is
clipped into the sheet, so every radius past ~0.55 lands on the same xy.  Every
candidate in the top bucket clears by 97–98 mm — a parked arm's own base column
is pose-invariant, so the *layout* sets that ceiling — and the tie-break inside
the plateau is the depot's own job (certified cells the arm can enter and leave).
The shipped set dominates the one it replaces arm for arm; arm 71 goes from
10/24 entries at 61 mm to **18/24 at 97.7 mm**.

### 2. Park-aware allocation (`allocate.ParkProbe`)

An arm outside the drawing group stands at its depot for the whole phase, and a
parked arm's schedule is a *constant* — the conductor cannot wait it out.  The
allocator now checks each certified span's own joint path against those parked
chains at the conductor's margin (with the 1-Lipschitz residual between path
samples subtracted) and bans a blocked span for that arm through the machinery
`prune_unflyable` already uses, so the ink goes to a different arm instead of to
a refusal three stages later.  `--arm-phases` decides who is parked and moved
to the allocator's own argument set; allocation and conduct read one grouping.

Measured on the logo, solo phases, same placement, same strokes: **51.9 s with
the probe against 51.6 s without** (59 probes + 91 cached, 0.2 s), same cover to
the digit.  It refuses **0 spans** on the shipped rig — because §1's park set
already stands 97.3 mm off every arm's ink.  On the set this rig shipped
yesterday it is what would have caught arm 71's depot at 61 mm.

### The run: still one arm at a time, and the reason changed  (SUPERSEDED — see "SIX ARMS AT ONCE" below)

Full pipeline at h = 0.940, pitch 0.61: placement re-searched against the banded
atlas (**1.094 × 1.431 m, turned 90°, 97.19 % allocated, 100 % live** — 47 %
more logo area than the 1.178 × 0.901 m the flat model chose), 12.851 m traced,
**12.490 m allocated over 43 certified segments** in 60 s.

| conduct mode | verdict |
|---|---|
| 6-mover (`--arm-phases off`) | refused — 4 movers, no monotone schedule for arms 31/71 |
| partner-disjoint `{2,13,31}` / `{17,71,97}` | both phases refused |
| **solo** | 4 phases, **2 certified** (arms 2 and 13), 2 refused (arms 31, 71) |

Shipped: makespan **54.333 s**, **0.0 s of conducted pause**, min inter-arm
**124.3 mm**, frame 190.8 mm, neighbour column 261.7 mm, paper chain 109.2 mm,
`scene_check` **PASS** on both phases, tip error 0.226 mm.  Effective
parallelism **0.96 arm-seconds per second** — one arm, always.  Drawn 1.749 m of
12.851 m = **13.61 %**.  `out/csail_proposed_h094_v2.{html,zip}` (34.6 / 16.3 MiB).

**THE NEXT BOTTLENECK IS THE LIFT LAYER, AND IT IS NOT THE PARKS.**  The
conductor names *pen-up transits*, and names the same ones whether the phase has
one moving arm or three — which can only be a stationary obstacle.  It is not a
park pose.  `atlas.solve_cell` certifies a cell on its **drawing** pose; the
6 cm hover over that same cell is a different configuration that no stage
certifies against the neighbours' pose-invariant base columns:

| arm | certified cells | drawing pose vs columns | 6 cm hover vs columns |
|---|---|---|---|
| 2 | 3719 | 50.0 mm, 0 fail | **−131.0 mm, 564 fail (15.2 %)** |
| 13 | 3757 | 50.0 mm, 0 fail | −131.0 mm, 558 fail (14.9 %) |
| 17 | 3763 | 50.0 mm, 0 fail | −91.9 mm, 165 fail (4.4 %) |
| 31 | 4211 | 50.1 mm, 0 fail | −131.0 mm, 560 fail (13.3 %) |
| 71 | 4203 | 50.0 mm, 0 fail | −91.9 mm, 174 fail (4.1 %) |
| 97 | 3723 | 50.0 mm, 0 fail | −91.9 mm, 164 fail (4.4 %) |

Every drawing pose clears the gate exactly; the hover over 4–15 % of the same
cells is up to 131 mm *inside* a neighbour's column.  The parked chains add
essentially nothing on top — 0 to 4 cells per arm out of ~4200 are blocked by a
park and not already by a column — and no park pose can fix it: a search over
593 candidates for arm 2 that scores **both** layers tops out at −3.9 mm.

**Pitch 0.65 does not answer it either**, and it was measured rather than
assumed: re-basing the same hover configurations on a 0.65 m pitch cuts the
blocked fraction (15.2 → 11.4 % for arm 2, 4.4 → 2.0 % for arm 17) and leaves
the worst case exactly where it was, at −131.0 mm.  That number is saturated —
`chain_static_clearance` returns 0 minus the capsule radius when a segment is
*inside* a box, and 0.131 is the forearm — so the deepest hovers have the
forearm fully through a neighbour's column and 4 cm of pitch is not the scale
of that problem.  The full 0.65 run was not spent on that evidence.

**What would actually move it now**, in the order the measurements rank them —
items 1 and 2 are DONE and the section below is what they found:
1. ~~**Certify the lift layer.**~~  Shipped.
2. ~~**Route pen-ups around the arms**, not only around the paper.~~  Shipped
   (`paper._skirt`).
3. **Fewer arms over the middle**, or a wider canvas — still there, and now
   measured at the lift layer rather than inferred.

# THE COMPOSED PROGRAMME (2026-08-26, h = 0.940, pitch 0.61)

**91.92 % of the logo, seven phases, `scene_check` PASS on every one of them**,
against the 62.19 % the single six-mover phase drew this morning — and on a
BIGGER mark: 13.55 m of certified ink against 7.95, at 1.263 x 1.653 m against
1.094 x 1.433.  205.2 s of makespan against 92.4.
`out/csail_proposed_h094_v4.{html,zip}` (45.5 / 18.6 MiB at `--stride 2`).

Two thirds of that came from the composition and one third from the PLACEMENT,
and the placement was the bigger half of what was left.  The order the
measurements landed in is the order they are written up in below, because the
last one invalidates the emphasis of the first three.

## The 37.9 % was never the metal

This morning's post mortem said the undrawn third was physics: middle-row arms
losing 17–18 points of their workspace to a transverse partner's base column,
and a base column is pose-invariant metal no search moves.  Every measurement
in it is still true.  The conclusion was wrong, because nobody had asked the
question the other way round.

`out/residual_anatomy.py` asks it — route every stroke from every arm's own
DEPOT with a bag of ONE segment and nobody parked, which is the most permissive
scene this rig has (a bigger bag only adds predecessors; a parked partner only
takes them away):

| | metres | what it would take |
|---|---|---|
| traced | 12.7783 | |
| no arm certifies the INK | **0.0000** | the layout, and nothing else |
| ink certified, no arm can FLY to it alone | **0.0000** | an RRT, or the layout |
| some arm certifies it AND can fly to it | **12.7783** | a programme |

All of it.  Every stroke, both inks, and **arm 31 alone certifies and can fly to
all thirty-eight**.  Nothing at this placement is out of reach.  Two SOFTWARE
constraints held the 4.84 m, and both are properties of a PASS:

**One pen per arm is a constraint within a phase.**  17 of the 18 strokes the
single pass gives back are orange, and one partition puts orange on arms 13 and
31 only.  `--two-pass` gives every arm every colour, one phase at a time:
62.15 % → **70.09 %** allocated, nothing else changed.

**A bag is flown in one tour.**  `prune_unflyable` wants a Hamiltonian path
through the arm's whole bag; finding no isolated node it falls through to its
shortest-segment fallback and drops eighteen strokes one at a time.  Every one
is reachable from the depot.  They are not reachable in the SAME TOUR.

## What each lever is actually worth, conducted

| composition | allocated | **conducted** | makespan | phases | par |
|---|---|---|---|---|---|
| single pass, 6-mover | 62.15 % | **62.19 %** | 92.4 s | 1 | 2.70 |
| two-pass, no rescue | 70.09 % | **32.43 %** | 55.1 s | 1 of 2 | 2.68 |
| two-pass + rescue | 70.09 % | **69.81 %** | 189.4 s | 6 | 1.42 |
| **two-pass + rescue + residual** | 73.35 % | **70.82 %** | **194.8 s** | **7** | **1.44** |

(all five rows at the SHIPPED placement, so that the levers are compared
against each other and not against a different picture; the placement itself is
worth more than any of them and has its own section below)

**The colour lever ALONE is a coverage regression.**  Two-pass moves ink onto
arms that then cannot share the paper: the orange 6-mover phase comes back
"arm 13 has no monotone pause schedule inside 136 s; all 24 priority orders of
the 4 moving arms were searched and none completed", and `--skip-unconductable`
threw all 4.83 m of it away.  32.43 %, from a change that allocated more.

A refusal is the conductor saying those arms cannot be on the paper together,
and the answer is fewer of them there — a SCHEDULE, not an allocation.  The
rescue ladder re-offers a refused phase as its arm columns, then as solos, then
cuts the one remaining arm's TOUR in half with a trip home between: 32.43 % →
**69.81 %**, 0.05 m lost, every phase signed off.

## What did NOT work, measured

**The residual chain plateaus.**  Re-allocating the holes as a further pass
gives back 0.4166 m in round 1 and **0.0000 m in round 2** — the same arm gets
the same untourable bag and nothing between rounds changes it.  `--residual-
passes 10` buys exactly one useful round.

**The routed depot fold is worth 5.5 points, not the 27 I first claimed.**
`paper.route` offered the depot only as a STRAIGHT joint-space line in and out;
routing each half instead and certifying the concatenation is a real
improvement — arm 31's crossings go 70.9 % → **76.4 %** — but `fold_home` is
chosen on four of sixty flagged crossings, 26 still have no route by any shape,
and the allocator bans **exactly the same eighteen** (stroke, arm) pairs with it
as without.  The first number I published for it, 98.1 %, was my own sampling
error: `--cells 12` against a `--cells 10` baseline.  See the correction commit.

## ...and then the placement, which was worth more than all of it

**The shipped placement was chosen by a search that could not see the columns.**
`out/csail_proposed_h094_v2_placement.json` was written at 04:50 on 2026-08-26,
at commit 8c1e7b1; `paper.STATIC_SAFE` landed at 07:24 in 6c1c864.  The search's
REAL stage is a full `allocate.allocate`, so its coverage number is a
flyability-aware number NOW and was not one then — which is why it scored this
placement 97.19 % and the honest allocator draws 62 % of it.

Re-run under the honest gate (78 real allocations, 1 h 24 m at `--jobs 10`,
`out/csail_place_v3_placement.json`), the answer moves — and it moves UP in
size, which nobody expected:

| rot | scale | w x h (m) | offset | REAL cov | area |
|---|---|---|---|---|---|
| 90 | **0.750** | **1.263 x 1.651** | (−0.20, −0.10) | **96.16 %** | 2.084 |
| 90 | 0.550 | 0.926 x 1.211 | (−0.30, −0.20) | 95.54 % | 1.121 |
| 90 | 0.700 | 1.178 x 1.541 | (−0.20, −0.10) | 95.22 % | 1.816 |
| 90 | 0.600 | 1.010 x 1.321 | (−0.30, −0.10) | 91.55 % | 1.334 |
| 90 | 0.650 | 1.094 x 1.431 | (−0.20, **−0.20**) | 89.48 % | 1.566 |

The shipped 0.65 is fifth, and note its offset: even at its own scale the
shipped placement is not the best one — it uses (−0.20, −0.10) and the honest
search wants (−0.20, −0.20).  **A bigger logo is a better logo here**, because
scale 0.75 puts more of the mark into the end-row arms' reach instead of
concentrating it in the middle band where only arms 31 and 71 can work.

Conducted at 0.75, the same composition draws **91.92 %**:

| | shipped placement | honest placement |
|---|---|---|
| logo | 1.094 x 1.433 m | **1.263 x 1.653 m** |
| traced | 12.7783 m | 14.7442 m |
| allocated | 73.35 % | **92.73 %** |
| **conducted** | 70.82 % | **91.92 %** |
| ink drawn | 9.050 m | **13.554 m** |
| makespan | 194.8 s | 205.2 s |
| phases | 7, all PASS | 7, all PASS |
| min clearance | 82.6 mm | 83.9 mm |
| nobody draws | 0.338 m | **0.170 m** |

## Where the last 8 % is

1.19 m, and it is the same shape as before, one lever down: ink no arm can
thread into a tour, on a placement where far less of the mark is in that
position.  It is still not unreachable.  The remaining levers:

1. **A real pen-up planner.**  26 of arm 31's 60 flagged crossings have no
   route by any shape on the ladder, and the ladder is now nine families deep.
2. **Fewer arms over the middle**, which is the physical fork and still open —
   but it is now worth 8 points of one drawing, not 38.

**And one thing that is NOT a lever, measured:** the balancer.
`allocate.balance_loads` piles 6.94 m of the 12.78 onto arm 31 and a big bag is
exactly what `prune_unflyable` cannot thread, so it looked like the obvious
culprit.  `--no-balance` at the shipped placement draws **7.94 m against
7.95 m**.  It is not the balancer.

# SIX ARMS AT ONCE (2026-08-26, h = 0.940, pitch 0.61)

**Yes.**  All six arms draw in one conducted phase, `scene_check` PASS,
**62.15 % of the logo** against the 13.61 % this rig shipped yesterday.
`out/csail_proposed_h094_v3.{html,zip}` (42.2 / 19.2 MiB).

Nothing about the rig changed.  Three gates did.

### 1. A hover is a pose the arm holds (`writing.lifted_or_lower`)

`atlas.solve_cell` gates a **drawing** pose against the static set.  The 6 cm
hover over that same cell is a **different configuration** — same tip,
different elbow — and nothing gated it against anything.  Two findings, both
measured over the six arms' certified cells:

* 4–18 % of hovers stood **inside** a neighbour's base column, up to 131 mm
  deep.  That is what every conduct refusal on this rig was naming, identically
  with one moving arm as with three, which can only be a stationary obstacle.
* worse, **55 %** of certified cells (330 of 600 sampled) had **no hover at
  all**: `lifted_config` pinned the tool yaw to phi = 0, which is free for an
  inline pen and a hard constraint for a holder with the tip 110 mm off the
  wrist axis.  `lifted_or_lower` silently returned the drawing pose with a lift
  of zero and the pen was dragged across the paper to the next stroke.

The fix is the fiber the drawing pose already spent: a hover has to hold a tip
position and nothing else, so phi, q7, the branch and the height are all free.
`hover_solve` tries the old narrow slice first and opens the whole fiber — 8
tool yaws x the q7 grid x every branch — when that answer is missing or sits on
the gate, ranked by a **capped-clearance score** rather than a threshold (a
threshold lands every hover a millimetre over the gate and makes every transit
between two of them marginal).  15 of 1200 cells now get no lift.

### 2. The way past a column is around it (`paper.STATIC_SAFE`, `_skirt`)

`paper.route` frame-checked only the detours it inserted, never the direct
move, so a transit that grazed a column was priced at zero and discovered by
`scene_check`.  It is on now, and two things had to be fixed before it would
converge — both bugs in the router, not facts about the rig:

* the floor was a **contradiction at the endpoints**.  The atlas certifies ink
  at `STATIC_MARGIN` and the router asked every leg for more, so a lift out of
  the tightest certified cells could never be satisfied.
  `effective_static_floor` clamps it, the same argument `effective_floors`
  already made one obstacle over.
* the only escape was **up**, on a rig whose blocker is a column and whose
  ceiling is steel.  `_skirt` reads the blocking boxes' own footprints and
  walks the hover plane around them — first when the metal is the only
  complaint, last when it is not.

### 3. The checker measures the same metal more roughly (`STATIC_PLAN_MARGIN`)

Three conducted phases — a solo arm, a **three**-arm phase and a **six**-arm
phase, each with a monotone schedule the conductor had already found and 83–88
mm of inter-arm clearance — were refused on one number: *min frame clearance
47.4 mm against a 50 mm margin*.  The ink under them measures 103 mm and the
pen-up over it 58.8.  It was never a collision.

`scene_check.static_clearance_lb` is an independent derivation, which is the
whole point of it, and an independent derivation of a minimum is a **lower
bound with slack in it**: it samples each capsule every 2 cm instead of
minimising along it and subtracts half a step (10 mm), then the trajectory
residual (2.75 mm) on top.  Every producer was gating at exactly
`STATIC_MARGIN`, which inverts the one ordering this repo runs on.
`rig_final.STATIC_PLAN_MARGIN` = 50 + 13 mm is what a producer pays now, ink
and pen-up alike; the checkers keep 50 and stay independent.  A test computes
the 13 from `scene_check`'s own constants, and a second one pushes real
certified poses and their hovers **through** the checker's bound.

### The run

Same placement, same atlas, same h and pitch as the 13.61 % baseline
(1.094 x 1.433 m at (0.702, 1.715), turned 90°, 100 % live).

| conduct mode | phases certified | drawn | makespan | arm-s / s |
|---|---|---|---|---|
| **6-mover (`--arm-phases off`)** | **1 of 1, all six arms** | **7.95 m = 62.15 %** | **92.4 s** | **2.69** |
| partner-disjoint | 1 of 2 (`{17,71,97}`) | 3.69 m = 28.9 % | 75.8 s | 1.5 |
| solo | 4 of 5 (13 refused) | 5.89 m = 46.1 % | 191.2 s | 1.0 |
| *baseline, 2026-08-26 am* | *2 of 4 solo* | *1.749 m = 13.61 %* | *54.3 s* | *0.96* |

**Shipped: the 6-mover.**  Makespan **92.4 s** with **75.7 s** of conducted
pause (all of it draw-vs-draw — no arm waits at a park), min inter-arm
**84.0 mm** against the 80 mm margin, frame **60.9 mm** against 50, neighbour
base column **150.6 mm** against 80, paper chain **42.2 mm** against 20, pen tip
−2.7 mm against the −10 mm contact floor, tip error 0.234 mm.  Effective
parallelism **2.69 arm-seconds per second** against the baseline's 0.96.

**Allocation is 5.8x slower and that is the honest price**: 357 s against 61 s,
almost all of it in `balance` (24 → 191 s), because a candidate bag is now
priced against a router that certifies every pen-up against thirty boxes
instead of against the canvas alone.  The screen was cut from flagging half of
an arm's crossings to a fifth (`dive_screen` settles the undecidable band with
`paper.leg_bounds` at 7 ms rather than handing it to `route` at 400), and the
route memo is global and parallel; what is left is real work.

### The next bottleneck is the middle row, and it is the layout

37.9 % of the logo is still undrawn and **every metre of it is in the middle
third of the sheet** (2.08 + 2.71 + 0.05 m by third).  Measured, per arm, over
certified cells under the logo — every ordered crossing routed twice, once with
the static set in play and once with it removed (`out/flyability.py`):

| arm | row | crossings that route | with the metal removed | refused by metal alone |
|---|---|---|---|---|
| 13 | end | 99.1 % | 99.1 % | **0** |
| 71 | middle | 80.0 % | 97.3 % | 19 |
| 31 | middle | 70.9 % | 89.1 % | 20 |

An **end-row** arm is untouched.  The two **middle-row** arms lose 17–18 points
of their own workspace to their transverse partners' base columns, and 39 of
their 54 refused crossings are refused by the column alone.  A base column is
pose-invariant metal 0.6 m long standing 0.61 m from its neighbour: no search
moves it, and the router has now exhausted every family it has (higher hovers,
Cartesian traverses, folding through the depot, and lateral skirts around the
footprints).

**So the fork Pete needs is a physical one.**  Either fewer arms over the middle
of a 1.80 m canvas (a wider pitch, a wider web, or a 2+4 row count), or the
pen-up layer stops being a shape library and becomes a real motion planner —
an RRT over the pen-up configuration space, which is roadmap item 3 and a
serious piece of work with no guarantee the corridor exists.  The measurement
says the first one is the cheap answer.

## Results snapshot (2026-08-17, h_inv = 1.00)

75.9 % of the 3.6×2.0 m sheet is strict-GO; ≥2-arm overlap only 10.2 %, no 3-arm
overlap. Inverted arms are annular at strict level; 15° tilt rescues the centers.
Gaps: the band between the four inverted arms, and the sheet corners.

**Whole-logo run (2026-08-19).** A 1.2969 × 0.9905 m CSAIL at offset
(+0.100, +0.050) m, drawn in TWO PASSES with a human pen swap between them and
one pen length per arm (arm 2 = 300 mm, arms 31/71/97 = 200 mm): **99.2139 % of
11.567 traced metres certified**, one 0.091 m span left empty at the centre of
the four inverted bases. 49 segments, both phases signed off by `scene_check`
(82.0 mm minimum clearance against an 80 mm margin), 1304 frames at 12 fps —
**108.6 s of wall clock**, down from 153.5 for the same ink after the makespan
pass below. `docs/DEAD_SPANS.md` has the recipe and what each knob was worth;
the pipeline is `csail_schedule.py --two-pass --pens 2:300,31:200,71:200,97:200
--min-coverage 0.99`.

**Makespan pass (2026-08-19).** Two changes, no safety spent and no metre
given up: the conductor now SEARCHES its priority order (every permutation of
up to six moving arms, ranked on makespan, branch-and-bounded through a shared
set of collision images) instead of taking the first that works, and the
allocator now balances the fleet on SECONDS instead of metres — a greedy
min-max pass that re-assigns spans a second arm has already certified at
identical endpoints, so coverage cannot change. Phase 1 63.5 → **38.4 s**,
phase 2 88.0 → **68.3 s**, total 153.5 → **108.6 s** (−29.3 %), pause 148.0 →
50.5 s (−66 %). Single pass with the arms permanently split grey/orange is
faster still and tops out at 91.1 % of the logo, so it is refused: coverage is
a constraint, not a term in the objective. `docs/CONCURRENCY.md`.

**Idle policy (2026-08-19).** What an arm does when it stops drawing, which was
"go home" and is now "stop where you are" (`aris_sixarm/idle.py`,
`--idle-policy freeze|home`, default freeze). Three quarters of every pause the
fleet paid was an arm waiting for permission to reach its PARK pose, so this
attacks where an arm STOPS rather than where it goes: freeze in place, a
certified minimal retreat for a frozen pose that is measurably in the way, and
slow just-in-time taxiing out of measured slack. **108.6 → 105.4 s (−2.95 %)**
at unchanged coverage (99.2139 %), margin and certificates, both phases signed
off by `scene_check` at 83.8 mm against the 80 mm margin.

The win is all in the orange pass: **68.3 → 65.1 s, and it now finishes exactly
at its floor** — dropping the trip home lowers the floor and the conductor lands
on it, so there is no concurrency loss left to remove. Freezing only pays for
the pass nobody follows, so the grey pass still goes home (the next pass has to
start somewhere, and walking home *between* passes is strictly worse than
walking home *during* one). Arm 31 had nowhere clear to stop and went home
alone; arms 2 and 71 took retreats of 0.4 and 2.3 rad against the 3.4–3.7 a trip
home would have cost. Pause goes UP, 50.5 → 101.4 s, which is the objective
hierarchy working as written: makespan first, pause only as the tie-break.

Two things fell out of it. At the inverted ready pose a 200 mm pen sits 16 mm
BELOW the paper and a 300 mm pen 113 mm below — v1 parked four arms there three
times a run and no gate looked, because paper clearance is checked on the nine
FK chain points and the pen is not one of them (reported, deliberately not
gated). And a pen-up the conductor cannot run is an EDGE of a tour, so the
refusal is now handed back to the sequencer, which is asked for the best order
that avoids it. `docs/IDLE.md`; `docs/SOLO_TIME.md` for what is left — a third
of the run is one arm drawing alone, and two thirds of that is spans another arm
certifies only PART of, which whole-segment moves cannot touch.

## Allocation v2: stroke splitting, and what the conductor did with it (2026-08-19)

`allocate.rebalance` now iterates **move | swap | SPLIT**. A span on the busiest
arm is cut at a chosen `s` and one piece handed to a lighter arm that re-plans
it from scratch; the pieces are `[s0, cut+e]` and `[cut-e, s1]`, so **their
union is the input span for every cut position** and coverage is invariant by
construction, not by measurement. Both pieces are held to 5 cm, the 5 mm they
share is ink laid down twice so the pens meet, and the receiver's reach inside
the span is measured by asking `plan_stroke` about that sub-span and reading
`s_star` off both ends rather than guessing from whole-stroke intervals. The cut
is then placed by bisecting to where the two arms' loads cross. `coverage_lost`
re-derives the merged cover afterwards and refuses rather than return a
programme that gave ink back: **0.000 mm** on every run in this repository.

**It lowers the floor everywhere, and the conductor cannot always keep it.**

| | floor v1 | floor v2 | cuts |
|---|---|---|---|
| CSAIL phase 1 grey | 32.1 s | **30.7 s** | 1 |
| CSAIL phase 2 orange | 65.0 s | **51.1 s** | 2 |
| bench spiral (ONE 14 m stroke) | 79.7 s | **56.7 s** | 9 |
| bench duotone | 282.6 s | 251.1 s | 5 |
| bench starburst | 160.8 s | 150.2 s | 2 |
| bench scatter | 40.4 s | 37.8 s | 1 |
| bench hatch | 351.5 s | 335.7 s | 9 |

Conducted, the CSAIL orange pass goes the OTHER way: **65.06 → 95.3 s**, floor
down 14 s and makespan up 30. `balance_loads` prices every arm as though it were
alone on the paper — exactly right for the arm, blind to the other five — and
under v1 arm 97 drew most of that pass by itself, which is 65 s of *solo* time
and solo time never waits. Balancing it away put four arms into the middle of
the sheet at once; the conductor refused the first schedule outright and its
rescue spends 110 s of pause. So the objective hierarchy decides it where the
information is: `csail_schedule.build_phases` conducts the split allocation
**and** the unsplit one and ships the faster, with `nominal_floor` as an exact
lower bound so the second conduct is skipped whenever it could not win.

**The whole piece runs in 104.02 s against 105.42**, at the same 99.2139 % of
11.567 traced metres, 82.1 mm of clearance against the 80 mm margin and
`scene_check` PASS on both phases — grey keeps its cut (38.35 → 36.96 s), orange
hands both of its own back. 1.3 %, where the floors said 21 %.

```
python3 scripts/csail_schedule.py --arms all --two-pass \
    --pens 2:300,31:200,71:200,97:200 --max-probes 5 --min-coverage 0.99 \
    --target-width 1.2969246423461636 --offset 0.1 0.05 --tag _v2 \
    --fps 12 --substeps 4
```

`--no-split` recovers allocation v1 exactly (105.42 s, verified); `--no-verify`
ships the split allocation without asking the conductor, which is how the 134 s
run above was measured. `docs/SOLO_TIME.md` has the post mortem; `docs/BENCH.md`
has the corpus.

**That 104.02 s is one EXECUTION PROFILE of four, and it is no longer the one
that ships (2026-08-20).** The same A/B machinery now chooses `qd_frac` and the
fiber menus per programme instead of per repository, and on this logo it ships
**77.792 s** — 82.4 mm of clearance, the same 99.2139 % coverage, both phases
`scene_check` PASS. The shipped `_full` artefacts are that run. The two 0.30
cells were never conducted: their floors (83.789 s and 92.716 s) are already
worse than the certified 77.792 s, which is a proof and not a guess.
`docs/BENCH.md` has the grid, and the corpus is why it is a choice — the same
0.60 cap that wins here is REFUSED outright on `bench`'s spiral.

## The bench — five drawings that are not the logo (`scripts/bench.py`)

Every headline above is one picture at one placement, which is the measurement a
balancer tuned to that picture passes. `aris_sixarm/bench` generates five
seeded, analytic drawings placed in sheet coordinates that owe nothing to the
logo — a dense hatching patch on one arm, sparse scatter over the whole sheet, a
starburst through the waist, one continuous 14 m spiral, and ten interleaved
two-colour bands — and `scripts/bench.py` runs the FULL pipeline on each.

| drawing | ink | coverage | makespan | floor | eff | cuts kept | solo |
|---|---|---|---|---|---|---|---|
| hatch | 30.1 m | 73.55 % | 355.1 s | 355.1 s | 1.00 | 0 (+9 back) | 82 % |
| scatter | 8.6 m | 84.72 % | 41.4 s | 41.4 s | 1.00 | 1 | 26 % |
| starburst | 20.2 m | 87.70 % | 162.3 s | 153.7 s | 0.95 | 2 | 6 % |
| spiral | 14.4 m | 85.57 % | 89.4 s | 83.4 s | 0.93 | 0 (+9 back) | 12 % |
| duotone | 29.8 m | 82.35 % | 268.5 s | 260.5 s | 0.97 | 5 | 41 % |

Splitting lowers the allocation floor on **all five** (4.5 % to 28.9 %) at
coverage identical to the digit, and the conductor keeps the cuts on three of
them and hands them back on two — **8 kept, 18 reverted**. The two it refuses
are the two where the ink is packed into one region, which is the same shape as
the CSAIL orange pass, found twice more on geometry that owes it nothing.

It also immediately found three things the logo could not, none of them about
splitting:

- **`plan_stroke` refuses a single stroke over 15.00 m outright**, as
  `too_long` rather than as a split — 1500 lattice steps at the default 10 mm.
  A 15.1 m spiral is not a hard allocation problem, it is one nobody is allowed
  to attempt.
- **The probe budget is per STROKE and reach is per METRE.** Worse, the gap walk
  dead-ends: a window that certifies nothing is marked tried and never
  subdivided, so it stops after four probes however large the budget is. The
  14 m spiral certified **0.00 %** of itself at a budget of 5 *and* at a budget
  of 40, while the atlas said 96 % of it was reachable. Scaling the budget and
  bisecting barren windows (`allocate.stroke_probes`,
  `probe_stroke(bisect=...)`, both behind one `probe_ref_m` argument and both
  off by default) take it to **85.6 %**.
- **Freeze-in-place can park an arm in a pose with no joint-limit margin left.**
  The planner certified the *stroke*; the hover above its end is a separate IK
  solve, and `idle.plan_retreat` only offers a retreat to a pose that is in
  somebody's WAY — so a pose that is merely bad goes unnoticed until
  `scene_check` refuses the phase. A phase that is clear and certified all the
  way through and fails only on frozen poses is now re-conducted with conductor
  v1's go-home: the same last resort, for the same reason, as the
  `Unconductable` fallback beside it.

# 100 % IS NOT AVAILABLE AT THIS GEOMETRY, AND HERE IS THE PROOF (2026-08-26)

The mandate was **"make it draw 100 % of the logo."**  It cannot, and the
reason is 110 mm long and sits 48 mm from arm 71's mount.  What the schedule
CAN do it now does: **99.3018 % of the paper covered against 99.2573 %** — the
certified maximum here is 99.3435 % — with no ink allocated that no phase
draws, and a makespan 8.8 s shorter.
`out/csail_proposed_h094_v6.{html,zip}` (43.9 / 18.7 MiB).

## The ceiling, measured before any allocator gets a vote

For every stroke and every arm, `probe_stroke` returns the sub-intervals that
arm can certify a drawing plan for.  The UNION of those over the six arms is an
exact upper bound on any programme at that placement: the cover, the balancer,
the router and the conductor can each only take ink away.  It costs probes and
nothing else — 30 s a placement against 16 minutes for an allocation — so it
can be asked of a whole neighbourhood (`out/ceiling.py`).

At the shipped placement it is **99.3435 %**: 16.6271 m of 16.7370 m traced, and
the 109.9 mm it leaves out is ONE contiguous span of grey.  The number does not
move with effort — identical to four decimals at a probe budget of 3 and of 12,
with the same single hole.

**That span lies under arm 71's own base column.**  Its midpoint is 47.8 mm from
the column centre; the nearest other arm is 611 mm away.  `plan_stroke` is
categorical rather than marginal about it: at both endpoints, in both
directions, arm 71 reports `empty_fiber` — no IK solution anywhere on the tool
fiber, because an inverted arm cannot fold back under its own mount — and the
other five report `start_infeasible`.  The atlas agrees independently: not one
of 201 samples along the span is within 15 mm of a strict-GO cell of ANY arm,
the nearest GO paper is 64–100 mm away, and the dead patch is 95 mm to its
nearest edge and about 300 mm across the midpoint (`out/holeA.py`).

## ...and the placement cannot dodge it

A nudge translates the logo rigidly, so the ceiling can be re-measured at every
neighbouring offset for one probe sweep each.  Fifteen of the 25 offsets on the
±0.10 m grid fit the sheet at this scale:

| dx, dy | ceiling | holes | worst hole |
|---|---:|---:|---:|
| −0.10, +0.05 | **99.4587 %** | 1 | 90.6 mm |
| **−0.10, 0.00** (shipped) | **99.3435 %** | 1 | 109.9 mm |
| 0.00, +0.10 | 98.7975 % | 3 | 100.8 mm |
| 0.00, +0.05 | 98.7383 % | 2 | 130.7 mm |
| −0.05, 0.00 | 98.4423 % | 2 | 160.8 mm |
| ...the other ten | 96.84 – 98.44 % | 2–4 | 140–248 mm |

**No offset in the neighbourhood has a ceiling of 100 %**, and the shipped one is
within 0.115 points of the best.  Moving the logo does not remove a column
shadow, it trades one for another — which is what the table IS: every arm in
this fleet casts one, and the sheet has six.  The one candidate with a higher
ceiling, (−0.10, +0.05), `csail_place` had already allocated in full and it came
out 0.56 points WORSE in real coverage than the shipped offset, so it is
measured, named and not adopted.

## The two levers, both fired SELECTIVELY

That word is the whole of the difference between them and the global switches
that were measured as a wash.

**1. The depot-aware hover, one span end at a time (`--depot-hover-selective`).**
`writing.HOVER_DEPOT_AWARE` re-searches the hover fiber for a pose that joins
the depot, and switched on globally it is +57.5 mm of grey against −81.5 mm of
orange.  Read again, that is an argument against the SWITCH: the tier fires at
every pocket and only some pockets are about to be paid for, and at the others
it swaps a 6 cm lift for a pose most of a radian away and re-prices the whole
bag around ink that was never at risk.  So the tier takes an ALLOW-SET of
span-end identities.  `lifted_or_lower`'s branch **and its memo key** both read
`depot_hover_selected(spec, q_ref, xy)` rather than the module flag, so an end
outside the set runs the same code and fills the same memo slot a tier-off run
does — "every other hover is untouched" is a property of the key, not a
measurement.  `allocate.fly_shrink` is the only thing allowed to grow the set,
because it is the only place that knows a pocket is about to cost something: it
admits the offending ends, asks `depot_round_trip` again, and keeps the
admission ONLY IF THE ANSWER CHANGED.  A rescue that buys nothing is rolled
back exactly.

On this logo it fires **four times and moves eight hovers**, and the proof it
moved nothing else is in the log: every other give-back comes back to the
millimetre and to the same s-range (stroke 7: 9 mm, stroke 11: 18 mm →
[0.30, 1.00], stroke 13: 25 mm).

| | v7 | v8 |
|---|---:|---:|
| grey drawn, primary pass | 5.700 m | **5.740 m** |
| grey given back at pocket ends | 93 mm | **53 mm** |
| orange given back at pocket ends | 433 mm | **216 mm** |
| whole strokes banned "cannot fly to it" | 1 | **0** |

**2. A refused pass is retried frozen (`--freeze-refused-phase`).**
`csail_schedule` allocates under the freeze policy — "can the arm fly OUT to
this span" — and then re-sequences every pass but the last WITH the trip home,
which is strictly harder; a phase that fails that is dropped whole.  The trip
home between passes is a CLOCK argument and the clock is only an objective,
while coverage is the constraint.  So such a pass is retried frozen before its
ink is dropped.  The allocation is untouched, the frozen pose still has to pass
`scene_check`'s own gate, and the inter-phase hold is still checked.

Lever 2 fires twice on this logo: on the 10.66 m orange pass, whose SPLIT
allocation has no ordering at all under `return_home=True` (v7 hit that too and
fell through to the unsplit one), and on the residual grey pass of one arm and
one segment that v7 dropped whole.

## The A/B, and the accounting it is measured in

`summary_json` reports `drawn_m / traced_m`, and `drawn_m` is a sum of SEGMENT
LENGTHS: it counts a handoff seam twice (`OVERLAP_M`, 4 mm each side of every
cut between two arms) and a split splice twice.  That ink is deliberately laid
twice so the pens meet, and counting it twice makes the ratio flatter than the
paper.  Worse, the `dropped` hole list rides on phase 0, and when the conductor
swaps phase 0 for its unsplit twin the composed list goes with it.  So the row
that matters is measured off the npz instead (`out/geocover.py`): every traced
stroke sampled at 1 mm, a sample DRAWN if some CONDUCTED segment of the same
colour passes within 1.5 mm.

Each lever was also run ALONE, so what each is worth is measured and not
apportioned:

| | v7 | v8b — lever 2 only | v8 — both, shipped |
|---|---:|---:|---:|
| **paper covered** (1 mm sampling) | 99.2573 % | **99.3018 %** | **99.3018 %** |
| empty | 124.3 mm | **116.9 mm** | **116.9 mm** |
| reported `drawn_m / traced_m` | 99.3649 % | 99.5516 % | 99.5217 % |
| ink allocated that no phase draws | 31.2 mm | **0** | **0** |
| makespan | 201.65 s | 195.62 s | **192.83 s** |
| pause | 88.40 s | **97.69 s** | 130.33 s |
| min clearance (margin 80 mm) | 83.2 mm | **83.2 mm** | 82.2 mm |
| conducted phases / segments | 5 / 46 | 4 / 47 | 4 / 45 |
| `scene_check` | PASS on all | **PASS on all** | **PASS on all** |

**Lever 2 buys all of the coverage and lever 1 buys none of it.**  Every one of
the 7.5 mm is the phase v7 dropped whole; the selective hover changes what the
ALLOCATOR keeps (40 mm more grey in the primary pass, stroke 26 no longer
banned, the pocket give-backs halved) and the residual passes had recovered all
of it anyway.  What it buys instead is 2.8 s of makespan, for 32.6 s more pause
and 1.0 mm of clearance.  The objective hierarchy says makespan first and pause
only as the tie-break, so **v8 ships** — but v8b is the same picture at the same
coverage with the v7 allocation untouched and a millimetre more clearance, and
on a rig somebody has to stand next to that is a real alternative rather than a
worse one.  `out/run_v8b.sh`.

And the refusal lever 2 catches on the orange pass is NOT lever 1's doing: v7
hit it too (`no feasible order over 7 segments`, log line 291) and fell through
to the unsplit allocation.  Lever 2 simply answers it one rung earlier.

**What is left is two holes and only one of them is geometry.**  106.9 mm of the
109.9 mm no arm can reach (the tolerance eats 1.5 mm at each end), and 13.0 mm
of orange at (0.657, 1.883).  That 13 mm is the head of a 216 mm span
`fly_shrink` gives back so arm 31 can fly to the other 203; it is under
`MIN_SEG_M`, so no cover, repair or residual pass will place it alone, the
fiber has no pose that joins the depot there, and — measured — **arm 31 is the
only arm that certifies any of that span at all**, so there is no second owner
for a pocket-aware assignment to prefer.  The one mechanism that would take it
is `merge_remainders`, which refuses on purpose: a merge may not hand back a
round trip, because the phase it is drawn in may have to go home.
`--freeze-refused-phase` makes that consequence recoverable and so makes the
refusal re-openable — 13 mm of ink against a certified programme, and not spent
here.

## The fork, for Pete

* **accept 99.30 %** — one 110 mm gap in a grey stroke, 48 mm from arm 71's
  mount, in the middle of the sheet.  Costs nothing.
* **widen the pitch, or go 2+4** — the shadow is the arm's own, and this is the
  same physical fork `docs/` has been naming since the middle-row study.
* **shrink the logo** — `csail_place` certified 98.18 % at f = 0.70 (1.178 m
  wide) against 97.93 % at the shipped 1.431 m, which is a smaller drawing and
  is the thing the v7 placement was chosen to avoid.

Reproduce: `out/ceiling.py` (the ceiling, one placement or a grid),
`out/holeA.py` (what `plan_stroke` says, and the dead patch),
`out/last106b.py` (per hole, per arm, tier off and on),
`out/geocover.py` (coverage off the conducted npz), `out/run_v8.sh`.

## Roadmap

1. ✅ reachability + controllability atlas
2. ✅ single-arm stroke planner: ladder-graph DP over (s × q7 × branch), maximin-σ
   objective, disconnect → split point (`planner.py`, `scripts/demo_stroke.py`);
   ✅ compact restatement of the same plan as a piecewise-linear q7(s) on one IK
   sheet, back-solved to joints (`pwl.py`, `scripts/demo_pwl.py`)
   ✅ C1 corner-rounded q7(s) + TOPP-lite pacing on top of it, both certified by
   the same IK chase (`smooth.py`, `pacing.py`, `scripts/demo_smooth.py`)
3. ▶ allocation: image plane → per-arm strokes, splitting at region boundaries /
   overlap handoff, RRT for the pen-up transits between planned strokes
   (`letters.py` + `writing.py` do the fixed-assignment case: one letter per
   arm, sequential, straight-line joint transits — no handoff, no RRT yet)
   ✅ image → strokes (`trace.py`) and strokes → certified per-arm programs
   with probe-measured reach intervals, pen-colour partitioning, greedy
   interval cover and mid-overlap handoff cuts (`allocate.py`, the CSAIL run)
   ✅ multi-arm timing: conductor v1 — capsule model, swept-cell collision
   images, priority pause-scheduling by exact reachability DP, and an
   independent merged-timeline validator with a veto (`coordination.py`,
   `scene_check.py`, `scripts/csail_schedule.py`; six arms, 83.7 mm minimum
   clearance over 65 s)
   ✅ per-arm sequencing: segments ordered AND turned round to minimise real
   transit time — exact Held-Karp to 16 segments, NN + 2-opt/Or-opt with
   direction flips above it (`sequence.py`, `stroke_api.reverse_plan`;
   −33.6 % pen-up time, −22 % makespan)
   ✅ makespan: exhaustive minimum-makespan priority search in the conductor
   (`coordination._search_priority`) and a min-max load-balancing pass over the
   interval cover (`allocate.balance_loads`, scored on the timeline's own
   clock) — 153.5 → 108.6 s on the whole logo at unchanged coverage, margin and
   certificates
   ✅ allocation v2: STROKE SPLITTING as a third balancing move — a span on the
   busiest arm is cut and one piece handed to an arm that certifies it, which
   is the only move that reaches the 48 % of solo time `docs/SOLO_TIME.md`
   measured as splittable (`allocate.rebalance`, `split_span`,
   `_Pricer.probe_span`).  Coverage invariant by construction and measured at
   0.000 mm.  It lowers the floor on all six drawings tested and the conductor
   can only keep some of it, so the conductor now rules on each phase
   (`csail_schedule.build_phases`) — see `docs/BENCH.md` and the honest post
   mortem at the end of `docs/SOLO_TIME.md`
   ✅ an anti-overfitting corpus: five generated drawings that are not the logo,
   run through the whole pipeline as a standing regression (`aris_sixarm/bench`,
   `scripts/bench.py`, `docs/BENCH.md`).  It found two things the logo never
   could: a single stroke over 15 m is refused outright by `plan_stroke`'s
   lattice ceiling, and the probe budget is per STROKE while reach is per METRE
   — a 14 m spiral certified 0.00 % of itself until both were fixed
   (`allocate.stroke_probes`, `probe_stroke(bisect=...)`)
   ▶ still open: the RRT transits (hover moves are still straight joint-space
   lines), re-ordering/re-routing in the conductor rather than pauses only (the
   orange pass's splitting win is sitting behind exactly this), and the
   acceleration-limited version of the schedule — playback is kinematic
