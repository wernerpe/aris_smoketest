# One arm, drawing alone — how much of the run, and whether it had to be

`scripts/solo_time.py` over the shipped schedule; numbers in `out/solo_time.json`.
Read-only: nothing is re-planned, and the allocation is re-run only to recover
the one thing the pipeline computes and then throws away — **for every drawn
span, the set of arms that could have drawn it**. `allocate.rebalance` builds
that set (probe coverage, then a clean re-plan that must give back not one
millimetre), uses it, and returns a count. This rebuilds it span by span, and
asks the finer question the balancer cannot: *which PART* of each span the other
arms reach.

## How many pens are down at once

| arms drawing | whole run (105.4 s) | phase 1 (grey) | phase 2 (orange) |
|---|---|---|---|
| 0 | 18.3 % (19.3 s) | 26.9 % | 10.8 % |
| **1** | **35.3 % (37.2 s)** | 31.5 % | 38.7 % |
| 2 | 39.1 % (41.2 s) | 40.6 % | 39.4 % |
| 3 | 5.3 % (5.6 s) | 1.1 % | 7.9 % |
| 4 | 2.0 % (2.1 s) | 0.0 % | 3.2 % |

**Four arms draw together for 2.1 seconds of a 105-second piece.** More than a
third of it is one arm working while three watch, and nearly a fifth is nobody
drawing at all.

## Why the solo stretches are solo

40 stretches, 37.2 s of them. Classified by what the probe data says about the
span being drawn:

| | seconds | share of solo | what it means |
|---|---|---|---|
| STRUCTURAL | 11.9 s | 32 % | no other arm of that colour certifies any part of the span. Only a different PLACEMENT changes this. |
| **SPLITTABLE** | **18.0 s** | **48 %** | another arm certifies PART of the span but not all of it — invisible to a balancer whose only move is the whole segment. |
| ALLOCATABLE | 7.3 s | 20 % | another arm certifies ALL of it and the balancer left it alone, because handing it over whole would only have made the receiver the new busiest arm. |

**Two thirds of the solo time is not structural.** Phase 2 is where it lives:
20 of its 32 spans (4.21 m of ink) are partly reachable by a second arm and only
2 are wholly reachable — exactly the shape of problem whole-segment moves cannot
touch. Phase 1 is the mirror image (5 spans wholly movable, 5 partly, 1.56 m)
and is already within 2.7 s of its floor.

## What a splitting balancer could reach

With spans divisible the min-max load problem has a closed form: for every
subset A of arms, work that can only be done inside A must fit inside A, so
`T* = max over A of (W(A) + O(A)) / |A|`. Cutting each span at its neighbours'
certified interval boundaries gives 22 atoms in phase 1 and 51 in phase 2.

| | floor now | makespan now | split-capable floor (ink only) | with today's pen-up tours |
|---|---|---|---|---|
| phase 1 | 35.7 s | 38.4 s | 20.2 s (binding {31, 71}) | **33.8 s** |
| phase 2 | 65.1 s | 65.1 s | 31.8 s (binding {2, 97}) | **46.7 s** |
| both + swap | — | 105.4 s | — | **≈ 82.5 s** |

**Phase 2 is the whole prize: 65.1 → 46.7 s.** Its floor today is arm 97's own
work — 49.6 s of ink that no other arm certifies *end to end*, plus its own
pen-ups — and it now finishes exactly AT that floor, so there is nothing left
for the conductor to win. Splitting is precisely what dissolves it: at sub-span
granularity only 31.8 s of ink is arm 97's alone.

Three things the estimate does not pay for, all in the optimistic direction:
sub-span eligibility comes from the probe intervals rather than from a clean
re-plan of the sub-span (the stricter test the balancer applies today, and the
one a real splitter would have to pass); each cut adds an entry and an exit to
the receiving arm's tour, which in this piece runs **1.4–1.9 s per pen-up**; and
a floor is a floor — the conductor's pauses sit on top of it. A splitting
allocator that reached 55–60 s in phase 2 would already be most of the win.

**Next milestone: allocation v2 with stroke splitting.** The machinery is nearly
all there — `stroke_api.plan_stroke` is certified-or-split and
`truncate_polyline` cuts at any arc position, `allocate.probe_stroke` already
records per-arm certified intervals, and `balance_loads` already scores on the
timeline's own clock. What is missing is a move that cuts a placed span at an
interval boundary, offers each piece to the arms that certify it, and re-plans
both halves — plus the seam handling `place_cuts` already does for the cover.
