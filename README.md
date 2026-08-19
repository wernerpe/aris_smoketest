# aris_sixarm

Motion analysis & planning for the **Aris Kindt** installation: six Franka FR3 arms
drawing on a shared horizontal paper plane. This repo consolidates the validated
kinematic conventions, the fleet model, the reachability/controllability atlas, and
(next) the multi-arm stroke planner — extracted from the sprawling upstream
`Aris_Kindt` branches into one clean, tested codebase.

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
                   active_override = a hypothetical fleet, never a registry edit
  coordination.py  CONDUCTOR v1: capsule model, pairwise collision images over
                   progress indices (swept cells, no tunnelling), priority
                   pause-scheduling by exact reachability DP.  Only the clock moves.
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
tests/
  test_gates.py    real-touchdown validation gates (run these after ANY kinematics change)
  test_planner_robustness.py   regressions distilled from the fuzz campaign
  test_csail.py    tracer / allocator / conductor regressions (11 tests, < 0.1 s)
docs/
  DECISIONS.md     every number the upstream repos disagree on, and what we picked
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
    scripts/csail_drawing_demo.py
```

Add `--sequencer nn` to either of the first two for the old paper-distance
chain with every segment drawn forward — the same pipeline, the same
validator, 83.1 s instead of 64.8 s.

`out/csail_final.png` is the end state: ink in pen colours, the 13 unreachable
spans dashed.

## Results snapshot (2026-08-17, h_inv = 1.00)

75.9 % of the 3.6×2.0 m sheet is strict-GO; ≥2-arm overlap only 10.2 %, no 3-arm
overlap. Inverted arms are annular at strict level; 15° tilt rescues the centers.
Gaps: the band between the four inverted arms, and the sheet corners.

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
   ▶ still open: the RRT transits (hover moves are still straight joint-space
   lines), re-ordering/re-routing in the conductor rather than pauses only, and
   the acceleration-limited version of the schedule — playback is kinematic
