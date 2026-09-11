# Stage assignment, wired end to end — and what the trajectories say that the envelopes did not

Build item 4 of `docs/ARCHITECTURE_V2.md`, landed as `aris_sixarm/staged.py`
with `tests/test_staged.py`. Everything below is one command:

```
ARIS_RIG=proposed ARIS_TOOL=lateral python3 -m aris_sixarm.staged \
    --lines out/csail_schedule_h097_v19_strokes.json --route-jobs 4 \
    --json out/staged_csail_h097.json --programme out/staged_csail_h097_program.json
```

against the shipped atlas `out/atlas_proposed_h0970_lat0860_gated63` at
h = 0.970, the shipped parks `layout.Q_PARK_PROPOSED`, the eight-stage
`traces.zigzag_pattern()`, and the persistent leg store of build item 1.

**The one-line answer.** The pipeline runs end to end and the first arm may
move **0.309 s** after the picture is in hand, against v19's 3 712.4 s. The
parked half of the safety argument holds: five of the six parked-partner
certifications clear 60–140 mm. **The active half does not, and not for the
reason anybody was worried about**: ink against ink clears by **+273.5 mm** at
worst, and the binding number in both three-active stages is **two PEN-UP LEGS
flying into each other** — +12.9 mm and +8.2 mm. That is the gap
`ARCHITECTURE_V2` §2(f) named in advance ("the pen-up leg between two hovers is
a *path*, whose interior is not in the envelope") and this is the run that
measures it.

---

## 1. What the module does, in four steps

Per `(stage, arm)` bucket, and nothing else touches it:

1. **The five partners are frozen at their parks** (`frozen.freeze`) before a
   single call is made, so every leg is certified against the partners' **real
   capsules** rather than against their 0.32 m pose-invariant bands. The
   persistent leg store is re-namespaced on the frozen set, because
   `paper.route_key` does not contain it — `staged.leg_cache_signature`.
2. **Every piece goes through `stroke_api.plan_stroke`**, the same funnel
   `scripts/csail_allocate.py` uses. A piece it refuses is recorded and
   dropped, not split: the DP's piece count is what the 2 cm atlas *permits*
   and the refusal fraction is a measurement this module exists to take.
3. **The accepted pieces are ordered by `allocate.sequence_arm`** — one tour
   per bucket, the same `segs` contract the allocator hands the sequencer
   today. No balancer, no split, no merge: the DP already minimised the piece
   count exactly.
4. **`writing.arm_program` lays the timeline down**: park → entry hover → draw
   → exit hover → … → park, every pen-up a `paper.route`. It ends at the park
   because a stage **barrier** is defined as every arm pen-up, stopped, at the
   park the next stage's envelopes were certified against.

No conductor anywhere. The three actives in a main stage are meant to be static
keep-outs for each other, and §3 is the measurement of whether they are.

## 2. The CSAIL logo, staged

39 strokes, 16.805 m of ink, at v19's placement. `traces.plan_lines` cuts it
into **54 pieces** across the eight stages in **30 ms**.

| stage | actives | pieces | ink (m) | stage time (s) | plan, parallel (s) | plan, serial (s) | refused | **ink vs ink (mm)** | **whole trajectory (mm)** | solo min (mm) | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 13, 71, 2 | 13 | 4.240 | 99.2 | 14.7 | 15.9 | 3 | **+273.5** | **+12.9** | +62.7 | **no** |
| 1 | 17, 31, 97 | 9 | 3.282 | 74.6 | 9.2 | 9.5 | 2 | **+338.8** | **+8.2** | +60.9 | **no** |
| 2 | 13, 97 | 6 | 1.461 | 48.0 | 2.2 | 3.3 | 0 | +1 055.3 | +977.2 | **+28.2** | **no** |
| 3 | 17, 2 | 5 | 1.145 | 29.4 | 1.3 | 2.2 | 0 | +1 151.2 | +870.9 | +96.2 | yes |
| 4 | 31, 97 | 5 | 1.083 | 31.8 | 2.4 | 2.4 | 0 | — | +897.0 | +140.2 | yes |
| 5 | 71, 2 | 4 | 0.923 | 58.6 | 2.1 | 2.1 | 0 | — | +623.8 | +72.5 | yes |
| 6 | 13, 31 | 4 | 0.889 | 26.5 | 1.9 | 1.9 | 0 | — | +263.7 | +71.2 | yes |
| 7 | 17, 71 | 2 | 0.959 | 23.6 | 1.1 | 1.1 | 1 | — | +525.3 | +80.9 | yes |

(The plan columns are the **warm** re-run; the cold numbers are in §4. "—" in
the ink-vs-ink column is a stage in which only one of the two actives was given
any ink by the DP, so there is no instant at which both pens are down: stages
4–7 of this logo are effectively solo.)

**Staged makespan 391.7 s against v19's conducted 209.9 s — 1.87× worse**, and
the reason is structural rather than a planner regression. A stage costs its
busiest arm and the barrier makes the sequence cost the **sum** of eight of
those, and each stage pays a full park → out → back trip for every active arm
whether it has 13 pieces to draw or 2. Stage 5 is the clearest case: 0.923 m of
ink for 58.6 s of clock. `traces.py` predicted this shape — the eight-stage
pattern is 2.35× the serial makespan in ink metres against a three-arm ceiling
of 3× — and the seam stages are where it is paid. It is the price of a
programme that can actually run at all, against a conducted one that needed
407.3 s of conducting to be allowed to.

**Time to first motion: 0.309 s.** The DP (30 ms), the first piece of stage 0's
first arm, its entry hover, and the park → hover leg. Pete's budget is 10 s and
`V2_SCALING_BASELINE` predicted 0.86 s at the median; this is under both,
because the binding term in that prediction — a cold `paper.route` that falls
through to the RRT — did not fire on this particular leg.

**Refusals: 6 of 54 pieces, 11.1 %.** What refuses them:

| reason | n | what it is |
|---|---|---|
| `split:empty_fiber` | 4 | the redundancy band has no q7 the arm can hold somewhere along the piece — the atlas cell is certified at 2 cm and the continuous piece is not |
| `split:sheet_collapse` | 1 | no single IK sheet spans the piece |
| `degenerate:too_short` | 1 | the piece is under `plan_stroke`'s 20 mm floor |

All six are `plan_stroke`'s honest "not end to end by this arm", and five of
the six carry a certified **head**. This module drops them rather than
re-entering the DP with a capability map re-derived from the refusals, which is
the refusal loop `ARCHITECTURE_V2` item 4 asks for and the one thing of item 4
that is **not** built (§6).

## 3. The two checks, and the one that fails

**They are two different questions and the module uses two different tools.**

**(a) Active against active — `staged.active_pair_gap`.** Not
`check_timeline` on a merged clock: inside a stage the actives are asynchronous
*by construction*, so there is no single alignment of their timelines to check
and checking one would certify a schedule nobody runs. The claim that has to
hold is the envelope claim restricted to the poses the planner actually
produced: for **every** pair of instants, one on each arm's timeline, the two
arms clear the gate. That is the minimum over the **cross product** of the two
pose sets, with `check_timeline`'s own 1-Lipschitz between-sample residual
applied on both axes.

**(b) Active against parked — `staged.solo_check`.** One
`scene_check.check_timeline` per active arm with the five partners held at
their parks for every frame, so the partners are measured as the capsules they
*are* rather than as the envelope they are not. The partners' pose-invariant
body bands are dropped from the static set (`staged.drop_bands`), because every
arm those bands stand for is in the timeline as its own capsules — the same
convention `scene_check.neighbour_columns` already uses for the cylinder
version of the same object. True structure is never dropped.

**The result.** Ink against ink is never the problem: **+273.5 mm** is the
worst it gets, against the +85.8 mm the 0.40 m dead band was chosen for. What
binds is the **pen-up legs**, and in both failing stages the binding instant has
*both* arms in a leg:

| stage | binding pair | both in | whole trajectory | ink only |
|---|---|---|---|---|
| 0 | 2 – 71 | leg / leg | **+12.9 mm** | +273.5 mm |
| 1 | 31 – 97 | leg / leg | **+8.2 mm** | +338.8 mm |

That is not a defect in the dead band and it is not a planner bug. It is
exactly the hole `ARCHITECTURE_V2` §2(f) identified: *"An envelope is a union
over poses — the drawing pose at each certified cell, the hover above it, and
the park — and the pen-up leg between two hovers is a path, whose interior is
not in the envelope."* Each arm's legs were certified against the **parked**
fleet, which is the right room at a barrier and the wrong one mid-stage, and
nothing in the pipeline has ever asked whether one active arm's leg crosses
another active arm's leg. §6 says what closes it.

**And one stage fails the parked check too.** Stage 2 reads **+28.2 mm**
against the 50 mm gate on the active-vs-parked side. The certified map at
h = 0.970 with the shipped parks frozen is 100 % of the block at 50 mm, so the
drawing poses are not the problem; this is a leg or a hover again, measured by
`check_timeline` with a sweep residual that `paper.route` does not charge
itself. Which of the two actives binds it was not attributed inside this pass
and is the first thing to run next.

## 4. What the planning cost, cold and warm

| | cold (first run) | warm (leg store hit) |
|---|---|---|
| lines → pieces (the DP) | 30 ms | 30 ms |
| plan + order + fly, all eight stages, serial | **221.2 s** | **38.4 s** |
| …the same charged per stage to its busiest arm | **213.2 s** | **34.8 s** |
| the two independent checks | 100.5 s | 99.9 s |
| wall clock, end to end | 322.0 s | 138.6 s |

**5.8× on the planning half, bought entirely by build item 1's persistent leg
store**, and the two runs produce a byte-identical makespan (391.732 s both
times). Per-arm planning is the row that matters for the architecture, because
the six arms plan independently: the worst single arm in a stage is 14.7 s warm
(arm 71 in stage 0, 13 pieces) against 4 240 mm of ink, so the planner stays
far ahead of the pens.

The check is now the expensive half — 100 s against 38 s of planning — and it
is the cross-product pair check that costs it, not `check_timeline`. That is a
fair trade for a claim with no timing assumption in it, and it is trivially
parallel per stage.

## 5. A thousand lines

Same module, `--synthetic strokes:1000 --no-fly --no-check`: the DP's pieces
run through `plan_stroke` and nothing else, because the question at 1 000 lines
is the **refusal fraction** — every number in `docs/V2_TRACES.md` is what the
2 cm atlas permits, and a piece the local planner refuses has to split and
re-enter the DP.

`traces.synthetic("strokes", 1000)`, the same deterministic set
`docs/V2_TRACES.md` §4 measures: 1 000 lines, 660.3 m of ink, **1 756 pieces**
— the figure that document reports, reproduced exactly — in **937 ms** of DP.

| | |
|---|---|
| pieces, from the DP | **1 756** |
| **refused by `plan_stroke`** | **318, 18.1 %** |
| `plan_stroke` over all 1 756, serial | **842.3 s** |
| …charged per stage to its busiest arm | **342.6 s** |
| time to first motion | **1.271 s** |

| reason | n | share of refusals |
|---|---|---|
| `split:empty_fiber` | 127 | 39.9 % |
| `split:start_infeasible` | 87 | 27.4 % |
| `split:sheet_collapse` | 58 | 18.2 % |
| `degenerate:too_short` | 33 | 10.4 % |
| `degenerate:off_sheet` | 11 | 3.5 % |
| `degenerate:too_short_after_clip` | 2 | 0.6 % |

**Two thirds of the refusals are the redundancy band, not the reach.**
`empty_fiber` and `start_infeasible` together are 214 of 318: the 2 cm atlas
certifies a *cell* and the continuous piece runs between cells, so the band
gives out somewhere along it or at its very first sample. `sheet_collapse` is
the next 58 — a piece no single IK sheet spans. Only the last 46 are the
piece itself being unplannable by anybody: 35 under `plan_stroke`'s 20 mm floor
(the DP's own `MIN_PIECE_M` is 10 mm, so it is entitled to hand out pieces the
planner calls degenerate) and 11 off the sheet.

**18.1 % against the logo's 11.1 %**, and the difference is where the ink is:
the synthetic set is spread over the whole certified block including its rim,
and the two three-active stages take almost all of the loss — 138 of 434 and
169 of 380, against 11 refusals in the six seam stages put together. A seam
stage's ink sits in a 0.40 m band under two arms that both reach comfortably
into it; a row band's ink runs out to the edge of one arm's reach.

**This is the number the refusal loop is worth, and it is not small.** 318
pieces have to split and re-enter the DP with a capability map re-derived from
the refusals, and the DP costs 10 ms, so the loop is cheap — but the re-planning
is not, and 18 % is too much ink to drop. It is also a direct measurement of
how optimistic the 2 cm prefilter is, which is a number nobody had.

**And the planning still keeps up.** 342.6 s of per-arm planning against a
draw of roughly 4 300 s (`V2_SCALING_BASELINE` §3.3) is an order of magnitude
of headroom, and that is cold — this run bought no legs at all, so a full run
adds the leg cost, which the logo measured at 5.8× cheaper warm.

## 6. What is not built, and what the numbers ask for next

**Not built, deliberately, inside this pass:**

- **The refusal loop.** A refused piece is dropped, not split and re-offered to
  the DP with a capability map re-derived from the refusal. It is the half of
  item 4 that needs the DP and the planner in a loop, and 11.1 % of the ink is
  what it is worth.
- **The seam stages are not conducted.** Stages 2–7 are two-active and
  `ARCHITECTURE_V2` §2(d) sends them through `idle.conduct`. This module plans
  them asynchronously like the others and reports the resulting clearance
  instead of scheduling it away. On this logo it did not matter — the seam
  stages are effectively solo — but on a picture that gives both seam arms ink
  it will.
- **The typed programme is not `program_schema.Bundle`.** `Segment` carries no
  joint vector and `_from_dict` refuses unknown *and* missing keys, so a stage
  id and a per-piece hover are a `SCHEMA_VERSION` bump rather than an
  extension. `staged.programme()` writes a superset of what item 5 has to
  absorb — stage id, barrier list, per-piece `q_first`/`q_last`,
  `hover_in`/`hover_out`, the pen-up leg blocks and the joint trajectory —
  under its own `STAGED_SCHEMA_VERSION`.

**What the numbers ask for next, in order:**

1. **Certify the legs against the other actives, not only against the parked
   fleet.** The cheapest form is the one `ARCHITECTURE_V2` §2(f) already names:
   one `check_timeline` per arm per stage against the other actives' work-cell
   **envelopes as static boxes**, and a leg that fails is re-routed rather than
   the stage being re-designed. The ink-vs-ink number says the cells are right;
   only the paths between them are unaccounted for.
2. **Or raise the legs.** Both failures are two arms in the air at once, and a
   hover plane is a free parameter the router already ladders over
   (`writing.HOVER_LADDER`). A per-stage flying height that separates the two
   row bands in *z* is a much cheaper lever than eroding the cells in *x*,
   which `DECISIONS.md` measured out on 2026-09-11.
3. **Make the barrier cheaper.** 391.7 s against 209.9 s is eight park → out →
   back trips, and four of the eight stages carry under a metre of ink. Merging
   the seam stages that do not actually share a horizon, or letting an arm stay
   out between two consecutive stages it is active in, is where the makespan
   is.

## 7. What is tested

`tests/test_staged.py`, 7 tests, **no environment variables** (the rig and tool
are switched in process and put back) and **no atlas** (every capability map is
rasterised from rectangles through the same `Coverage` object the atlas
produces):

- **No same-row pair is active in any stage** of `zigzag_pattern()`, and
  `staged.run` **refuses** a pattern that has one — the transverse pair is
  unseparable on this rig on either axis, so this is a refusal and not a
  report.
- `pieces_of` is exactly the DP's answer: one `Piece` per drawn piece, the same
  geometry as `LinePlan.piece_points`, bucketed by `(stage, arm)`.
- **A stroke crossing the dead band is deferred to a seam stage and drawn
  there**: every piece whose midpoint is inside SEAM0 carries a seam stage and
  every piece outside it carries a main stage, the crossing stroke is cut into
  exactly two pieces at the boundary to 0.1 mm, and nothing is dropped.
- **The frozen-partner model is what is installed while planning**: a hook
  fires inside `plan_bucket` once per piece and asserts, at that instant, that
  `frozen.active()`, that the frozen set is the five partners and not the
  mover, that `frozen.observer()` is the mover, and that every frozen pose is
  that arm's shipped park.
- `drop_bands` drops only `body:*_column*` boxes, keeps every mount and plate,
  and keeps the band of an arm the caller does not name.
- **End to end on a three-stroke picture**: pieces → plans → order → legs → a
  timeline that starts *and ends* at the park (the barrier), with the per-stage
  active-pair and solo checks both clearing `PAIR_MARGIN`, and a typed
  programme whose every piece carries its stage, its two joint endpoints and
  its two hover configurations.
