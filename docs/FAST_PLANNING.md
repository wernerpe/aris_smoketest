# THE ROBOTS START DRAWING SOONER, AND WHAT WAS ACTUALLY IN THE WAY

`scripts/draw.py` takes a picture and gives back a certified programme. On the
Trollface that took **4028.6 s**, and the number worth optimising is not the
whole grid but the **time to the FIRST certified programme** — the only one a
person standing in front of the fleet can feel.

It is now **1916.7 s**, and the interesting half of that sentence is what the
1916.7 s is made of: 178.7 s of planning and 1737 s of conducting. The stage
this pass went after — allocation — moved by **9.2x** and is no longer the run;
the conductor is, and §6 says exactly where.

Everything below is measured on the same input: the Trollface at the placement
it shipped (0.850 m wide, offset (0, −0.95), rotation 0, `final6_opt`, six
arms, `--max-probes 5`), with the before and after run on the same machine.

## 1. Where the clock was, and the surprise inside it

One profile's allocation, timed stage by stage:

| stage | seconds | share |
|---|---:|---:|
| prefilter | 0.2 | |
| probe | 8.7 | 0.9 % |
| repair | 0.2 | |
| clean re-plans | 3.3 | 0.3 % |
| **balance** | **931.1** | **93.4 %** |
| sequence | 13.7 | 1.4 % |
| **total** | **997.4** | |

and `--select-profile` runs four of those, one after another, for a grid
that took 1646.6 s before a single timeline was conducted.

The balancer's own record for that run says what it bought: **0 relocations, 0
swaps, 5 splits**, on a cover where only 7 of 57 segments have a second arm that
certifies them at all. Fifteen minutes to move nothing and cut five spans.

### 1.1 The four things that were wrong

**Pricing was from scratch.** `arm_load` rebuilt the arm's whole cost matrix for
every candidate bag and re-solved its tour from the greedy seed under the
sequencer's full 2 s budget. Every candidate the balancer considered differed
from the bag before it by ONE span.

**The scan was best-improvement.** Every relocation and every swap out of the
busiest arm was priced before one was taken — and the effect of a move on the
two loads it touches is bounded by the ink it hands over, which costs nothing to
compute.

**Nothing knew when to stop.** A phase whose busiest arm is already at the lower
bound its ink imposes cannot be improved by any assignment, and the pass had no
way to say so.

**The split search enumerated.** Eight segments x five receivers x two coarse
cuts, every one of them re-planning two halves and pricing two arms' tours.

### 1.2 And the thing nobody had measured

A profiler on the fixed balancer said something none of the above predicted:
**94 % of building a cost matrix is `paper.route`.** At this placement **57 % of
the crossings between one arm's spans dive through the canvas**, and each of
those walks the ladder — lift, retract-and-go, arc, Cartesian traverse,
fold-through-home — at about 36 ms. A whole allocation asks about **57 000
distinct crossings**: half an hour of ladder-walking, and everything else in
`sequence.py` is array arithmetic beside it.

That is the real cost model of this pipeline, and it had never been named.

## 2. What replaced them

### 2.1 The crossing is a property of two spans (`sequence.leg_costs`)

`C[a, b]` is the lift off node a, the hover-to-hover crossing and the lower onto
node b. Every term is a function of those two poses, the arm and the run's fixed
speeds — never of which OTHER spans happen to be in the bag. So the matrix over
the union of every span the balancer offers an arm CONTAINS the matrix over any
bag of them as a submatrix, and re-pricing is an index operation.

`allocate._ArmMatrix` keeps one such union per arm. It GROWS (a split makes new
spans): `ensure` appends the missing spans and computes only the new rows
against the old columns, the old rows against the new, and the new block against
itself. `matrix(keys)` slices a bag's `(C, T, e)` out of it. `tests/test_balance.py`
pins that the slice equals the freshly built matrix cell for cell under both cost
models, and that growing the union does not disturb what was in it.

`cost_matrix` and `cluster_cost_matrix` are now assembled from the same three
pieces (`node_lift`, `leg_costs`, `depot_legs`), so there is one definition of a
crossing and not two.

### 2.2 The tour search starts from the tour it already had (`sequence.seed_from`)

A candidate is the current bag plus or minus one span, and the arm's existing
tour already orders everything else in it. `seed_from` filters that tour down to
the segments this bag still has and drops the newcomers in at their cheapest
position; the descent does the rest, under `BALANCE_BUDGET` (0.15 s) instead of
`sequence.TIME_BUDGET` (2 s), and Held-Karp is used only while its state count
stays under `BALANCE_EXACT_STATES`.

The DP, the matrix and the objective are identical — only the effort spent
searching differs, and **the bag that finally ships is re-sequenced at full
budget by `allocate` regardless**. The price is memoised on the bag, which is
what keeps `load_score` a potential: a bag priced once keeps that price, every
accepted move strictly lowers the objective, and the loop still terminates.

### 2.3 First improvement, in an order the ink already knows

`balance_loads` sorts its candidates by the busiest arm the ink alone predicts —
`max(L[src] − d_i, L[b] + d_i)` for a relocation — and takes the best of the
first promising prefix rather than pricing all of them. `BALANCE_PATIENCE` is
why it is a prefix and not literally the first: a candidate that improves only
the sum-of-squares tie-break is an improvement, and taking it costs the round
that a real reduction of the maximum would have used.

Over 200 random synthetic instances (8-40 segments, 2-6 arms) against the
exhaustive scan: **worst case 1.5 % above best-improvement, mean 1.0000, and
5.4x fewer candidates priced.**

### 2.4 The floor is where improvement stops being possible (`_phase_floor`)

Every arm's load is at least its ink, and every segment costs at least what the
cheapest arm that certifies it would spend drawing it. Two exact bounds follow:
the longest single segment (no assignment of undivided segments beats it) and
the total ink spread over the arms that can hold any of it (which survives
splitting, because a cut adds the splice and the total only grows). The
improvement loop stops within `BALANCE_EPS` (2 %) of the first; the split search
never starts if the busiest arm is already at the second.
`tests/test_balance.py` enumerates small instances outright and checks nothing
gets under either.

### 2.5 The split search ranks before it prices

`_cut_gain` judges a cut on the ink and the seam alone — two multiplications —
and only `SPLIT_CAND_TRIES` of that ranking are ever priced; the winner is then
placed exactly by `_bisect_cut`, which is where the cut position was always
really decided. A receiver no more than `SPLIT_MIN_GAIN` lighter than the source
is not probed at all.

Two more things keep the union small. The shortlist is CERTIFIED in bulk and its
matrices grown in ONE step (`_Pricer.grow`), because a growth step's wall clock
is its slowest route and doing that eight times over is eight waits. And
`_Pricer.compact` drops the cuts a round rejected: pure index arithmetic, no
crossing re-screened, and without it the round after prices its candidates
against a union carrying every reject before it — on this picture that took an
arm's union from the 36 spans it can hold to 115.

### 2.6 The crossings are routed on every core there is

`sequence.prewarm` collects the diving crossings of a whole growth step and
routes them in one dispatch over a **persistent** fork pool
(`sequence.screen_pool`); `sequence._PRICED` memoises the PRICED detour, not just
the route, so a crossing is walked exactly once per run however many matrices
ask about it. The batch matters as much as the pool: a batch's wall clock is its
slowest member, and a crossing that CANNOT be routed walks the whole ladder.
Measured on the largest dispatch — 14 314 crossings in 21.6 s — the pool runs at
**23x**.

Three details are load-bearing. The key is computed in the PARENT and handed to
the worker, because `paper`'s memo keys on `id(spec)` and a worker's copy of the
arm is a different object. A crossing the parent has already routed is priced in
the parent, because a worker forked before it was bought would walk the ladder
again — and that is also what makes the other three profiles cheap. And a worker
never forks a pool of its own.

## 3. What the four profiles may share, and what they may not

Nothing the planner does depends on `qd_frac`. `plan_stroke` resolves the
redundancy against reach, clearance and the band objective; the joint-speed cap
enters only when `writing` turns a path into a clock and `sequence` prices a
transit. `allocate.plan_family` is that statement as a key — the strokes, the
arms, each arm's pen and its band objective — and the probe pass and the clean
re-plans are memoised under it in a `share` bag the caller keeps across
allocations. `paper`'s route memo and `sequence`'s priced-crossing memo need no
key at all: a pen-up route knows nothing about `qd_frac` either, and they are
process-global.

The four cells were always run in one process, so the route memo was already
shared; what `plan_family` adds is the PLANNER's half of it — the probes and the
clean re-plans, which are the two stages that still cost real seconds once the
crossings are memoised. Between them they take the second cell of a family from
70.9 s to 70.9 s of pure balancing with nothing bought at all (see §5.2).

That is also why `--profile-jobs` stays at 1. Forking the four allocations buys
cores and gives up both memos, and a worker may not fork the route screen's own
pool either, so each of the four would pay serially for geometry the shared run
buys once.

`tests/test_balance.py` pins that two profiles sharing a bag allocate exactly
what they allocate alone — on the spans themselves, not on a summary — and that
the second one re-probes nothing. The probe map is COPIED into each allocation
because `repair_gaps` appends to it.

The fiber menus are deliberately NOT shared: `menu.drop_lattice` releases the
lattice in place, and the second profile would be re-materialising what the first
one dropped.

## 4. The first programme, and then the rest

`select_profile` used to allocate all four cells, order them by floor, and
conduct them one at a time so each could prune the next. Nothing was written
until all four had been decided.

It now runs in two stages. `FIRST_PROFILE` names the cell this rig has always
shipped — `qd0.60+cluster`, what the CSAIL logo ships at (77.792 s against
104.021 s on the defaults, `docs/BENCH.md`), what the Trollface ships at, what
two-pass ships at — and it is allocated, conducted and certified BEFORE the other
three are allocated at all. The moment `scene_check` signs its timeline off,
`on_certified` fires and `draw.py` puts a runnable programme on disk marked
`provisional`, with `elapsed_s` counted from the picture.

The remaining three are then allocated, pruned against that certified makespan,
and conducted **in parallel** — they no longer prune each other, so conducting
them together costs the longest one instead of the sum. They replace the
provisional best only if one of them is faster, and the files say which is which.
`--conduct-jobs 1` restores the serial chain every published number was measured
on.

## 5. The numbers

### 5.1 One allocation, the profile that ships (`qd0.60+cluster`)

| | before | after |
|---|---:|---:|
| **balance** | **931.1 s** | **132.7 s** |
| probe | 8.7 s | 8.8 s |
| clean re-plans | 3.3 s | 3.4 s |
| sequence | 13.7 s | 4.5 s |
| **allocation total** | **997.4 s** | **154.9 s** |

and the allocation is the same allocation:

| | before | after | delta |
|---|---:|---:|---:|
| coverage | 98.7970 % | 98.7970 % | 0 |
| certified segments | 59 | 59 | 0 |
| splits taken | 5 | 5 | 0 |
| busiest arm, before balancing | 83.840 s | 84.017 s | +0.2 % |
| busiest arm, after balancing | 73.980 s | 74.038 s | **+0.08 %** |
| phase floor (`nominal_floor`) | 73.373 s | 73.417 s | **+0.06 %** |

Well inside the 2 % the mandate allowed, on both numbers it named.

### 5.2 All four, in one process

| profile | allocation | balance | probe | busiest arm | floor |
|---|---:|---:|---:|---:|---:|
| qd0.60+cluster (first) | 160.0 s | 137.3 s | 8.9 s | 84.02 -> 74.04 s | 73.42 s |
| qd0.30 | 63.1 s | 43.8 s | 9.5 s | 149.04 -> 136.71 s | 138.52 s |
| qd0.30+cluster | 56.0 s | 49.4 s | **0.0 s** | 112.96 -> 94.05 s | 99.87 s |
| qd0.60 | 21.8 s | 19.4 s | **0.0 s** | 99.03 -> 91.25 s | 91.76 s |
| **total** | **300.9 s** | | | | |

All four allocate the same 98.7970 % over 59 segments. The two zeroes are
`plan_family` doing its job: `qd0.30+cluster` asks the planner exactly what
`qd0.60+cluster` asked, and `qd0.60` exactly what `qd0.30` asked. The falling
balance times down the column are `paper`'s route memo doing its — by the fourth
cell every crossing on the picture has been walked already.

AND THE GRID IS NOT FOUR TIMES ONE ALLOCATION, WHICH IS WORTH SAYING BECAUSE IT
WAS NOT BEFORE EITHER.  `paper`'s memo has always been process-global, so the
old pipeline was already getting some of this for free and the honest comparison
is the pipeline's own log, not four cold runs:

| | before | after |
|---|---:|---:|
| qd0.30 | 202.8 s | 70.9 s |
| qd0.30+cluster | 1124.8 s | 70.9 s |
| qd0.60 | 87.7 s | 30.8 s |
| qd0.60+cluster | 231.3 s | 178.7 s |
| **four allocations** | **1646.6 s** | **351.3 s** |

(both measured inside `scripts/draw.py --select-profile`, which is why the
per-cell numbers differ from the standalone table above: the order is not the
same, and whichever cell goes first pays for the geometry.)  A COLD allocation —
one profile, nothing memoised, which is what the first cell of any run is —
goes 997.4 s to 154.9 s.

### 5.3 A second picture: the CSAIL logo, two passes

`docs/TWO_PASS.md`'s allocation — both inks, every arm available for both,
`--freeze-all-phases`, `qd0.60 --cluster` — run on the old code and the new,
same input, same rig:

| | before | after |
|---|---:|---:|
| grey pass, balance | 218.9 s | 26.5 s |
| orange pass, balance | 636.2 s | 74.4 s |
| **both passes, wall** | **879.8 s** | **113.9 s** |

and this one is not merely within tolerance, it is the same allocation:

| | before | after |
|---|---:|---:|
| coverage | 100.0000 % | 100.0000 % |
| certified segments | 46 | 46 |
| splits taken | 2 + 5 | 2 + 5 |
| busiest arm, grey | 18.847 s | 18.847 s |
| busiest arm, orange | 19.108 s | 19.108 s |
| **profile floor** | **41.2018 s** | **41.2018 s** |

That last row is what settled one design question. The split search first
stopped at the first candidate that improved, as `balance_loads` does — and on
the orange pass that took 5 splits down to 3 and left the busiest arm 8 %
heavier (20.641 s against 19.108 s). The two cases are not alike: in
`balance_loads` pricing a candidate IS the cost, while in the split search the
cost is the certifying and the matrix growth, which the whole shortlist has
already been charged for before any of it is priced. Pricing all of it costs
about two seconds a round and gets the splits back.

### 5.4 The Trollface, end to end

`scripts/draw.py` on the picture at the placement it shipped,
`--select-profile`, animation off (the render is a separate stage and needs the
drake venv).  Both runs on this machine, at the same time, so neither had it to
itself:

| | before | after |
|---|---:|---:|
| trace | 1.0 s | 1.0 s |
| four allocations | 1646.6 s | 351.3 s |
| — the winning cell alone | 231.3 s (fourth) | 178.7 s (first) |
| conducts | 1921 + 461 s, serial | 1737 s, then 2 in parallel |
| **time to a certified programme ON DISK** | **4028.6 s** | **1916.7 s** |

**2.1x on the headline, and that is the honest number.**  The old pipeline wrote
nothing until the whole grid had been decided, so its first programme and its
last are the same event; the new one writes the provisional best the moment
`scene_check` signs it, and 1737 s of the 1916.7 s that takes is ONE CONDUCT.

The stage this pass was aimed at moved by a lot more.  Time from the picture to
an allocation that could be conducted at all: **1646.6 s to 178.7 s, 9.2x** —
because the old run had to allocate all four cells before it ordered them by
floor, and the new one finishes the cell that has always won first.

and the programme that came out is not merely as good, it is the SAME
programme, to every digit the summary carries:

| | shipped | this run |
|---|---:|---:|
| execution profile | qd0.60+cluster | qd0.60+cluster |
| makespan | 108.375000 s | 108.375000 s |
| coverage | 98.7970 % | 98.7970 % |
| min inter-arm clearance | 82.918 mm | 82.918 mm |
| certified segments | 55 | 55 |
| drawn | 13.114555 m | 13.114555 m |
| conducted pause | 93.166667 s | 93.166667 s |

Which is the point worth making about all of §2: none of it is an
approximation of the old allocator. It is the same cost model, the same DP and
the same objective, asked fewer times and on more cores.

## 6. What is in the way now

The allocation is no longer the run, and the measurement says so plainly. On the
Trollface the first certified programme lands at 1916.7 s, of which the
allocation is 178.7 s — **the other 1737 s is one conduct**, and the conductor is
untouched by any of the above. Getting the next order of magnitude means going
after it, and this is what it is made of.

Its own log says where that goes:

```
priority search: 13 DP solves over the 6 orders of 3 moving arms in 148.7 s
priority search: 13 DP solves over the 6 orders of 3 moving arms in 219.7 s
priority search: 13 DP solves over the 6 orders of 3 moving arms in 153.4 s
```

Two levers are visible from here and neither was in this pass's scope:

**The 13 DP solves are independent.** `coordination._search_priority` walks
permutation PREFIXES depth-first, and the arms in one subtree are scheduled
against the arms before them and nothing else. The bound it prunes with is the
running incumbent, so a parallel version has to give something up or re-derive
it — but at 11 s a solve and three passes per conduct (baseline, jit, retreat),
that is where the next order of magnitude is.

**The split/unsplit A/B is two whole conducts, run one after the other.**
`build_phases` conducts the split allocation, and on this picture that
allocation cannot be conducted at all — freeze, retreat and v1's go-home are all
refused, at the cost of most of an hour — before the unsplit one is conducted
and ships. They are independent given the same starting pose, so the second
could be started speculatively beside the first and cancelled when the floor
says it cannot win. That is the same bargain `select_profile` now strikes with
the other three profiles, one level down.
