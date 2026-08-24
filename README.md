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

## Layout

```
aris_sixarm/
  frames.py        FR3 DH model, FK, joint/torque limits, pen transform  (single source of truth)
  fleet.py         the 6-arm world layout (mounts, base transforms, per-arm colors)
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
                   boundary contours for solid glyphs  (numpy/PIL only)
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
  csail_drawing_demo.py  drake/meshcat playback of that npz (venv python3.10)
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
```

## Key facts

- **Pen**: no CAD model exists anywhere. tip = hand-TCP + **0.110 m** tool-z
  (gate-validated on the rig). IK targets the tip, not the flange.
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
