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

