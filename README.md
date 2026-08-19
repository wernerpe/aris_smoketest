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
  viz/             drake-mesh robot model + static meshcat scene builder
scripts/
  run_atlas.py     sweep all/selected arms  (~35 s for all six)
  make_scene.py    build out/reach_atlas.html (static meshcat + legend)
tests/
  test_gates.py    real-touchdown validation gates (run these after ANY kinematics change)
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
