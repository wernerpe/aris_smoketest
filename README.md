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

## Results snapshot (2026-08-17, h_inv = 1.00)

75.9 % of the 3.6×2.0 m sheet is strict-GO; ≥2-arm overlap only 10.2 %, no 3-arm
overlap. Inverted arms are annular at strict level; 15° tilt rescues the centers.
Gaps: the band between the four inverted arms, and the sheet corners.

## Roadmap

1. ✅ reachability + controllability atlas
2. ✅ single-arm stroke planner: ladder-graph DP over (s × q7 × branch), maximin-σ
   objective, disconnect → split point (`planner.py`, `scripts/demo_stroke.py`)
3. ▶ allocation: image plane → per-arm strokes, splitting at region boundaries /
   overlap handoff, RRT for the pen-up transits between planned strokes
