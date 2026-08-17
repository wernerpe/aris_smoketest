# Redundancy: how it's encoded and how we'll resolve it

## The redundancy space

With the pen tip pinned to the paper at (x, y), the free DOF per arm are:

1. **q7 self-motion** — the He/Liu solver's redundancy parameter (≈ wrist/elbow
   circle). Continuous, up to ~6 rad of travel, up to 4 discrete branches on top.
2. **tool yaw** — the pen is rotationally symmetric, so the full 2π of yaw is free.
3. **pen tilt** — a cone around perpendicular, bounded by drawing quality
   (`tilt_max_deg`, default 15°; see the tilt study).

So each drawable point sits on a ~3-D manifold of configurations (+ discrete
branches). Strokes are paths through this manifold; trouble happens where the
manifold pinches.

## Encoding: where are we rich vs pinched? (in the atlas today)

Per cell, `atlas.py` stores:

- **`valid_frac`** — fraction of the (orientation × q7) candidate grid with a
  comfortable solution (margin ≥ 0.15). Global "how many options" score.
- **`q7_window`** — the widest CONTIGUOUS q7 interval (rad) that stays
  comfortable at one fixed tool orientation. This is the corridor a stroke can
  glide through **without a branch jump**; it's the metric that predicts
  whether a line can be drawn in one smooth motion. (Inherited idea: IKA
  step4_corridor, which checked q7-set overlap between consecutive waypoints.)
- **`n_sol`** — raw count of limit-valid IK solutions (includes all branches).
- The best-margin `q` itself — a warm start / seed for the planner.

`redundancy` layer in the meshcat scene = `q7_window` (bright = many options).
Rule of thumb: cells with `q7_window` < ~0.5 rad are single-option pinch points —
strokes through them must enter with the right branch already selected.

## Resolution strategy (planner, next phase)

Per stroke, after allocation to an arm:

1. **Corridor pre-check** (atlas lookup): every waypoint's cell must have a
   non-empty comfortable q7 set; intersect windows along the stroke. Empty
   intersection somewhere → the stroke needs a branch change → split it at the
   pinch or reallocate.
2. **Global q7 schedule, not greedy**: dynamic programming over a discretized
   (waypoint × q7) lattice. Edge cost penalizes |Δq7| (smoothness), low margin,
   low σ_min, tilt use; forbidden where IK fails. This picks the q7 trajectory
   that keeps the arm centered in its corridors instead of drifting to a wall
   and dying mid-stroke.
3. **Branch chasing at execution resolution**: `ik.solve_cc` (case-consistent)
   follows the scheduled branch waypoint-to-waypoint; a failed chase = the DP
   grid was too coarse there → refine locally.
4. **Yaw/tilt as escape valves**: yaw re-aims the wrist without moving the tip
   (free); tilt buys reach at the rim (bounded by material quality). Both enter
   the DP as extra lattice dimensions only on strokes that need them (keeps the
   lattice small).
5. **Nullspace posture on the real robot**: the impedance controller's
   `k_nullspace` spring pulls toward the planned q — the plan IS the redundancy
   resolution; the controller just tracks it compliantly.

## Open questions

- Cross-arm coordination: two arms in the 10% overlap band must also resolve
  *each other's* swept volumes — corridor intersection becomes a joint problem.
  Plan: keep overlap handoffs time-separated first (no simultaneous strokes in
  the same overlap cell); true simultaneous coordination later.
- Elbow sweep vs. cage/boom: q7 travel moves the elbow a lot on inverted arms;
  the DP cost needs the real scene geometry (aris_planning_scene) before we
  trust large q7 excursions near the booms.
