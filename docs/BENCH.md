# The bench — five drawings that are not CSAIL

`scripts/bench.py`, numbers in `out/bench.json`, drawings in
`aris_sixarm/bench/`.  Every headline this repository has published is ONE
picture at ONE placement, which is exactly the measurement a balancer tuned to
that picture would pass.  These five are generated, seeded, analytic, and
placed in sheet coordinates that owe nothing to the logo; each one is written
to break a different part of the pipeline.

Regenerating them costs nothing and reproduces exactly: the two that use
randomness use `np.random.default_rng(seed)` with the seed pinned in
`bench.BENCH`, and the other three have no randomness at all.

| what it stresses | how |
|---|---|
| **hatch** | ~30 m of parallel lines packed into ONE arm's territory. Everything is reachable by the arm it sits on and almost nothing end to end by anybody else: the case a whole-segment balancer cannot touch. |
| **scatter** | ~7 m of short strokes over the whole sheet. Nothing is worth cutting and the clock is nearly all pen-up — the case that catches a splitter that cuts because it can. |
| **starburst** | 24 rays out of the sheet centre, which is the waist between the four inverted bases. Every ray runs from the hardest region on the paper to an easy one. |
| **spiral** | ONE continuous 15 m stroke. No whole-segment move exists, so whatever balance it reaches is the cutting move's alone. |
| **duotone** | Ten interleaved grey/orange bands spanning the sheet. Neither pass gets a tidy half of the paper. |

The fleet is the rig as it stands — all six arms, `2:300 31:200 71:200 97:200`
mm pens, 80 mm margin (50 safety + 30 calibration), freeze-in-place idle policy,
0.12 m/s draw and 0.80 m/s transit — because a pen is a fixture and not a knob.
`scene_check` has a veto on every row below; a drawing it refuses is printed as
REFUSED rather than quietly dropped.

**What this corpus found that the logo could not.** All three are about long
strokes or crowded arms, none is about splitting, and each is reported here
rather than tidied away:

1. `plan_stroke` refuses a single stroke over **15.00 m** outright, as
   `too_long` rather than as a split — 1500 lattice steps at the default 10 mm.
   The spiral is generated at 14.3 m so it measures the allocator instead.
2. The probe budget was per STROKE while reach is per METRE, and worse, the gap
   walk dead-ended: a window certifying nothing was marked tried and never
   subdivided, so it stopped after four probes however large the budget. The
   14 m spiral certified **0.00 %** of itself at a budget of 5 *and* of 40,
   while the atlas said 96 % of it was reachable. Fixed behind `probe_ref_m`
   (`allocate.stroke_probes`, `probe_stroke(bisect=...)`), off by default:
   **0.00 % → 85.6 %**.
3. Freeze-in-place can park an arm in a pose with no joint-limit margin left.
   The planner certified the STROKE; the hover above its end is a separate IK
   solve, and `idle.plan_retreat` only offers a retreat to a pose that is in
   somebody's WAY, so a pose that is merely bad goes unnoticed until
   `scene_check` looks at it. A phase that is clear and certified all the way
   through and fails only on frozen poses is now re-conducted with conductor
   v1's go-home (`csail_schedule.build_phase`), which is the same last resort,
   for the same reason, as the `Unconductable` fallback beside it.

**Where the determinism stops, stated rather than hoped.** The allocator's
choices — which spans, which arm, where to cut — are a function of the input
alone: no randomness, every iteration order sorted, every tie broken on values
the caller can see. So is the sequencer, up to `sequence.EXACT_MAX_N` = 16
segments per arm, where Held-Karp is exact. ABOVE 16 it is a local search under
a wall-clock budget, and on the two dense drawings (`hatch`, `duotone`) some
arms carry more than 16 pieces, so those two rows can move by a fraction of a
second between machines. The splits, the coverage and the floor do not.


## The 2026-08-20 refactor did not move a digit of this corpus, on purpose

The gated min-travel band objective (`pwl.OBJECTIVE`) and the entry/exit fiber
menus with the (segment, direction, variant) cluster DP (`allocate.CLUSTER`)
are both **off by default** — `docs/REDUNDANCY.md` and `docs/CONCURRENCY.md`
have the measurements that put them there, and the short version is that both
constrain the band, a constrained band has a larger |dq/ds|, and
`writing.draw_duration` stretches the ink until no joint exceeds `qd_frac`
= 0.30 of its limit. They buy pen-up time with ink time, and the ink is the
larger number.

So the claim this corpus has to support is a NEGATIVE one: with the shipped
defaults the pipeline is behaviour-identical to the one that produced the table
below. Re-run on that basis, `hatch` and `scatter` reproduce **every** column
to the digit — coverage, makespan, floor, efficiency, splits, pause, clearance,
`scene_check` — and differ only in wall clock (1854 → 1825 s and 2303 → 2207 s,
which is the machine and not the planner):

| drawing | coverage | makespan | floor | eff | splits | clearance | reconfig | verdict |
|---|---|---|---|---|---|---|---|---|
| hatch | 73.55 % | 355.1 s | 355.1 s | 1.00 | 0 | 138.4 mm | 43.7 rad | identical |
| scatter | 84.72 % | 41.4 s | 41.4 s | 1.00 | 1 | 85.0 mm | 77.4 rad | identical |
| starburst | 87.70 % | 162.3 s | 153.7 s | 0.95 | 2 | 81.4 mm | 32.7 rad | identical |
| spiral | 85.57 % | 89.4 s | 83.4 s | 0.93 | 0 | 82.1 mm | 49.9 rad | identical |
| duotone | 82.35 % | 268.5 s | 260.5 s | 0.97 | 5 | 85.1 mm | 98.5 rad | identical |

All five are re-run, and every one of `coverage`, `makespan_s`, `floor_s`,
`efficiency`, `splits`, `pause_total`, `min_clearance`, `n_segments` and
`solo_share` compares equal to 1e-6 against `out/bench.json` — column by
column, not eyeballed. Efficiency does not regress anywhere because nothing
moved anywhere. The only column that differs is the wall clock, which is the
machine.

**What the corpus gained is a column, not a number.** `bench.py` now records
`reconfig_rad` — Σ‖q_exit − q_entry_next‖∞ over consecutive segments, the
null-space swing between strokes that the floored transit beats hide and the
quantity the fiber menus exist to lower (`sequence.reconfiguration`). Turning
`--cluster` on cuts that quantity by 65 % on the CSAIL logo and costs 47 % of
the clock, which is why it is off.

## Results (2026-08-20, allocation v2 (splitting), 31706 s for all five)

| drawing | regime | ink m | cov % | makespan | floor | eff | splits | solo % | wall |
|---|---|---|---|---|---|---|---|---|---|
| hatch | dense parallel hatching in one arm's territory (arm 71) | 30.1 | 73.55 | **355.1 s** | 355.1 s | 1.00 | 0 (+9 reverted) | 82 | 1855 s |
| scatter | sparse short strokes over the whole sheet | 8.6 | 84.72 | **41.4 s** | 41.4 s | 1.00 | 1 | 26 | 2303 s |
| starburst | radial rays from the sheet centre (the waist between the inverted bases) | 20.2 | 87.70 | **162.3 s** | 153.7 s | 0.95 | 2 | 6 | 13353 s |
| spiral | one continuous spiral | 14.4 | 85.57 | **89.4 s** | 83.4 s | 0.93 | 0 (+9 reverted) | 12 | 3657 s |
| duotone | two-colour interleaved bands | 29.8 | 82.35 | **268.5 s** | 260.5 s | 0.97 | 5 | 41 | 10539 s |

`floor` is the busiest arm's own nominal programme summed over the phases — its ink plus its pen-ups, every other arm assumed free — and no schedule can beat it. It is what the ALLOCATOR moves. `eff` = floor / makespan is what the CONDUCTOR moves: 1.00 means nobody ever waited. `solo %` is the share of the run with exactly one pen on the paper, the quantity `docs/SOLO_TIME.md` diagnosed and the one stroke splitting exists to lower. `splits` counts the cuts the conductor KEPT; a bracketed number is cuts it handed back, because `csail_schedule.build_phases` conducts the unsplit allocation as well and ships the faster of the two.

**What splitting is worth before the conductor sees it**, measured on the same five drawings at the allocation stage alone (the busiest arm's nominal programme, v1 whole-segment moves only against v2 with cutting). Coverage is identical to the digit in every row, which is the invariant stated as a measurement rather than as an argument:

| drawing | coverage | floor v1 | floor v2 | change | cuts |
|---|---|---|---|---|---|
| hatch | 73.55 % | 351.5 s | 335.7 s | −4.5 % | 9 |
| scatter | 84.72 % | 40.4 s | 37.8 s | −6.4 % | 1 |
| starburst | 87.70 % | 160.8 s | 150.2 s | −6.6 % | 2 |
| spiral | 85.57 % | 79.7 s | **56.7 s** | **−28.9 %** | 9 |
| duotone | 82.35 % | 282.6 s | 251.1 s | −11.1 % | 5 |

The spiral is the one that matters most, because it is ONE stroke: there is no whole-segment move to make, so every second of that 28.9 % is the cutting move's and nothing else's.

| drawing | segments | phases | pause | clearance | scene_check |
|---|---|---|---|---|---|
| hatch | 79 | 1 | 0.0 s | 138.4 mm | PASS |
| scatter | 69 | 1 | 3.6 s | 85.0 mm | PASS |
| starburst | 28 | 1 | 55.1 s | 81.4 mm | PASS |
| spiral | 40 | 1 | 55.9 s | 82.1 mm | PASS |
| duotone | 65 | 2 | 34.4 s | 85.1 mm | PASS |


## Speed-dependent verdict: that was a verdict AT ONE DRAW SPEED (2026-08-20)

`scripts/speed_sweep.py`; floors in `out/floor_sp*.json`, conducted cells in
`out/csail_schedule_sp*.json`, the mechanism in `out/speed_sweep_ink.json`.

The section above judges the min-travel band and the fiber menus at
`writing.DRAW_SPEED_FLEET` = 0.12 m/s, the speed the ANIMATION runs at. The rig
draws at `pacing.V_DRAW` = **0.02 m/s**, six times slower, and the mechanism
that made the features expensive is speed-dependent in a way the verdict was
not. `writing._draw_time` is one `max`:

    draw_s(segment) = max( length / draw_speed ,  need )
    need            = max_i |dq_i| / (QD_MAX_i * qd_frac) / du_i

`need` is a property of the path through the band alone — radians per unit of
normalised arc over a velocity limit — and it never sees the draw speed. So
each segment has a critical speed **v\* = length / need**, above which the joint
cap binds and the ink is stretched and at or below which the ink is exactly the
material's `length / draw_speed`. Constraining the band raises `need`, and so
raises v\*; it cannot touch `length`. The premise the menus rest on — that
interior draw time is invariant across variants — is not false in general. It
is false **above** v\* and exactly true below it.

**Method, and what is measured versus estimated.** Conducting a cell costs ~45
min, so only four were conducted (0.08 and 0.12 m/s, both settings). Every cell
is ALLOCATED and FROZEN at its own draw speed — `allocate.rebalance` scores
assignments in seconds, so who draws what is itself a function of the speed —
which gives the floor exactly. Makespan is then floor + a per-setting offset
calibrated on the conducted cells (`speed_sweep.OFFSET_S`): 24.01 s for the
defaults, 31.34 s with the features. That offset absorbs the entry/exit lifts
and idle taxi that `floor_cell` omits, plus the conductor's pauses. **The
load-bearing assumption is that the offset does not grow as the draw speed
falls** — every term in it is a joint-space move paced by `qd_frac` and not by
`draw_speed`, and a slower programme spreads the same hover conflicts over more
seconds, so arms meet in time less often, not more. The two conducted speeds
per setting agree on it to ~3.5 s, which is the error bar on every estimated
row. Rebuilding the conducted cells from the model lands within **±1.8 s**:

| cell | floor (computed) | makespan (model) | makespan (conducted) | error |
|---|---|---|---|---|
| 0.08 defaults | 90.9 s | 114.9 s | **116.708 s** | −1.77 s |
| 0.08 min_travel+cluster | 107.1 s | 138.5 s | **140.083 s** | −1.63 s |
| 0.12 defaults | 81.8 s | 105.8 s | **104.021 s** | +1.78 s |
| 0.12 min_travel+cluster | 93.7 s | 125.0 s | **123.375 s** | +1.63 s |

No conducted confirmation at 0.05 m/s was attempted: a 10-minute budget buys
~350 frames there, i.e. fps ≈ 1.7, and the conductor's swept slack is
0.55 × the per-step motion, so a clock that coarse measures its own coarseness
and not the draw speed.

**The grid.** Coverage is **99.2139 %** in all eight cells and all four
conducted runs — asserted per cell from the allocator's own totals, not assumed.

| draw speed | features | floor | ink | stretch | capped | transit | makespan |
|---|---|---|---|---|---|---|---|
| 0.02 m/s | defaults | 254.4 s | 582.4 s | 1.006x | 2/58 | 89.0 s | 278.5 s *(est)* |
| 0.02 m/s | min_travel + cluster | 260.4 s | 579.6 s | **1.002x** | **1/57** | **81.1 s** | 291.8 s *(est)* |
| 0.05 m/s | defaults | 118.3 s | 241.2 s | 1.043x | 4/56 | 80.5 s | 142.3 s *(est)* |
| 0.05 m/s | min_travel + cluster | 132.6 s | 243.6 s | 1.053x | 6/55 | 82.1 s | 163.9 s *(est)* |
| 0.08 m/s | defaults | 90.9 s | 175.0 s | 1.210x | 18/57 | 85.7 s | **116.7 s** |
| 0.08 m/s | min_travel + cluster | 107.1 s | 182.1 s | 1.259x | 16/57 | 81.8 s | **140.1 s** |
| 0.12 m/s | defaults | 81.8 s | 140.2 s | 1.457x | 29/52 | 82.7 s | **104.0 s** |
| 0.12 m/s | min_travel + cluster | 93.7 s | 134.9 s | 1.401x | 24/55 | 77.9 s | **123.4 s** |

| draw speed | defaults | min_travel + cluster | change |
|---|---|---|---|
| 0.02 m/s | 278.5 s | 291.8 s | **+4.8 %** |
| 0.05 m/s | 142.3 s | 163.9 s | +15.2 % |
| 0.08 m/s | 116.7 s | 140.1 s | +20.0 % |
| 0.12 m/s | 104.0 s | 123.4 s | +18.6 % |

**Where the cap binds.** Under the shipped band the smallest v\* on this logo is
**0.0149 m/s** and the median is 0.109; under min-travel they are 0.0374 and
0.128. So below ≈ 0.015 m/s *no* plan is joint-limited anywhere, and at the
rig's 0.02 m/s exactly one or two segments of ~57 are — the ink comes out
**1.002x** the material's time with the features on, against 1.401x at 0.12.
**The hypothesis that the cap stops binding at the real draw speed is
confirmed.**

**But the features still lose at 0.02 m/s, and the reason has changed.** The
ink objection is gone and the transit win is real — 81.1 s against 89.0 s,
−8.9 % — and the cluster setting does **less total work**: draw + transit of
660.8 s against 671.4 s. What it does not do is spread that work evenly. The
floor is the busiest arm, and the whole residual penalty is one arm in one
phase:

| 0.02 m/s | phase 1 (grey) floor | phase 2 (orange) floor | imbalance ph2 |
|---|---|---|---|
| defaults | 102.1 s | 152.4 s | 1.40x |
| min_travel + cluster | **100.6 s** | 159.8 s | 1.49x |

That is precisely the open issue `docs/CONCURRENCY.md` already names:
`allocate.arm_load` prices a candidate assignment with the SINGLE-variant
sequencer, so the balancer balances a load the cluster DP then moves. At
0.12 m/s that defect was hidden behind a much larger ink-stretch penalty; at
0.02 m/s it is the only thing left, and it is now the one change that would
flip this table.

**The other half of the same knob is worth more than the features are.** v\* is
linear in `qd_frac`, so the 0.30 default is as much a part of the 2026-08-20
verdict as the draw speed. Re-conducting the 0.12 m/s pair at `--qd-frac 0.6` —
not a hypothetical, it is the cap `README.md` reproduces the six-arm animation
with — at identical coverage and clearance:

| 0.12 m/s | qd_frac 0.30 (default) | qd_frac 0.60 (README demo recipe) |
|---|---|---|
| defaults | 104.0 s (ink 1.49x) | **84.4 s** (ink 1.07x) |
| min_travel + cluster | 123.4 s (ink 1.48x) | **84.8 s** (ink 1.06x) |
| gap | +18.6 % | **+0.4 %, a tie** |

Doubling the cap is worth 18.9 % of the makespan on the shipped defaults alone,
and it closes the feature gap almost entirely.

**Reproduction note, reported rather than tidied away.** The 153.5 s and −65 %
reconfiguration quoted above and in `docs/CONCURRENCY.md` do NOT reproduce from
the committed code: `--cluster --band-objective min_travel` at 0.12 m/s now
gives **123.4 s** with reconfiguration 46.93 → 43.78 rad (−6.7 %, not −65 %).
The menus are the difference — mean 2.0–3.0 variants per segment now against
4.6–6.0 in the stored `out/csail_schedule_new.json` — because
`menu.MAX_SURCHARGE` = 0.05 s prunes variants by their interior draw-time cost,
a repair that landed after those numbers were taken. Note that the surcharge is
priced twice (`menu.py` prunes on it, `sequence.cost_matrix` charges
`w_surcharge / qd_frac` for it) and **neither place sees `draw_speed`**, so
below v\* both are paying for extra ink time that does not exist. That is
conservative rather than wrong — it forgoes good variants, it never ships a bad
plan — and nothing here changes it.

**Recommendation (report only; no default is flipped by this work).**
**REAL-RIG programs at 0.02 m/s:** keep `pwl.OBJECTIVE = "maximin_sigma"` and
`allocate.CLUSTER = False` — the features cost 4.8 % of makespan, but the ink
argument that put them there no longer applies and `allocate.arm_load` is the
thing to fix. **DEMO-ANIMATION programs at 0.12–0.15 m/s:** same two defaults,
and raise `--qd-frac` from 0.30 to 0.60, which is worth −18.9 % of makespan on
the defaults alone at unchanged coverage and clearance.
