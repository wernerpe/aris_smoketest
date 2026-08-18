# Redundancy: how it's encoded and how we'll resolve it

## The redundancy space

With the pen tip pinned to the paper at (x, y), the free DOF per arm are:

1. **q7 self-motion** — the He/Liu solver's redundancy parameter (≈ wrist/elbow
   circle). Continuous, up to ~6 rad of travel, up to 4 discrete branches on top.
2. **tool yaw** — the pen is rotationally symmetric, so the full 2π of yaw is free.
   **For a perpendicular pen this is not an independent DOF**: R = rotz(yaw)·rotx(π)
   puts joint 7's axis on the pen axis, so the solution at (yaw+δ, q7+δ) is the
   same arm with q1..q6 unchanged and q7 shifted. Yaw only becomes independent
   once the pen is tilted (see 3).
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

## Resolution strategy (implemented in `planner.py`)

Per stroke, after allocation to an arm:

1. **The fiber is q7 × branch — not yaw × q7 × branch.** For the perpendicular
   pen, yaw is degenerate with q7 (above), so a (yaw × q7) lattice is not a
   richer search space: it is the *same* space sampled twice, aliased into
   diagonal bands (with 8 yaws × 24 q7 the valid q7 indices formed a comb of
   period 3 = 0.785/0.256). A small ±1-index transition window cannot travel
   along that diagonal, so the DP reported artificial disconnects. Yaw is
   therefore pinned to 0 and q7 (48 samples over the FR3 range) carries the
   whole 1-D self-motion, × up to 4 analytic-IK branches. Nothing wraps.
2. **DP, not RRT, inside a stroke.** The pen must follow r(s) monotonically in
   s, so the search graph is *structurally* a DAG layered by s — every edge goes
   from step i to i+1. One forward sweep is therefore globally optimal, in
   Ns × (Nq × 4)² time; sampling-based search buys nothing where the layering is
   free. RRT is reserved for the **pen-up transits** between strokes, which are
   the only genuinely 7-D, non-monotone, obstacle-dominated motions.
3. **Maximin σ_min, not a weighted sum** (default objective). A stroke is only
   as controllable as its worst step — libfranka zeroes the external-wrench
   estimate near singularities — so the DP maximizes the *bottleneck*
   min_s σ_min(q(s)), with total ‖Δq‖² as the lexicographic tie-break:
   V[n] = min(σ[n], max_{p→n} V[p]). An additive cost averages a lethal dip
   away; the bottleneck cannot be traded off. The old weighted sum
   (smoothness + margin + σ shortfalls) is still available as `objective="additive"`.
4. **Nodes are gated, edges enforce continuity**: margin ≥ 0.15 rad and
   σ_min ≥ 0.08 per node (plus paper/boom clearance), ‖Δq‖∞ ≤ 0.35 rad per edge
   — which resolves branch *identity* without ever labelling branches.
5. **Disconnect = split, not failure.** If no node at step i+1 is reachable, the
   forward pass stops and reports the farthest reachable s*: the stroke's
   natural cut point for reallocation or a pen-up transit. Two mechanisms
   produce it — an empty fiber (the under-base dead zone: no IK at all) and a
   *reachable-set* collapse (the fiber is non-empty but only on an IK sheet a
   π-branch-flip away from the one the stroke is on). The second is the one
   worth reporting: the atlas says "reachable", the planner says "not from here".
6. **Branch chasing at execution resolution**: `ik.solve_cc` (case-consistent)
   follows the scheduled branch waypoint-to-waypoint; a failed chase = the DP
   grid was too coarse there → refine locally.
7. **Tilt as the escape valve**: tilt buys reach at the rim (bounded by material
   quality) and, unlike yaw, is a real extra dimension. It enters the lattice
   only on strokes that need it (keeps the lattice small).
8. **Nullspace posture on the real robot**: the impedance controller's
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
