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

---

## 8. Closing the leg gap — the other actives, as an envelope

§3 measured the failure and `ARCHITECTURE_V2` §2(f) named the fix. It is now
built, and it is one sentence: **before an arm plans anything in a stage, the
other actives of that stage go into its static room as their whole work-cell
envelopes.**

**What the envelope is.** Not a box — a box round an arm's reach swallows its
neighbour and would refuse everything. It is the union of the partner's **link
capsules** over every pose it could hold in this stage: the certified drawing
pose at each strict-GO cell of its work cell, the hover above each of those, and
its park. That is exactly the pose stack `scripts/workcell_envelopes.py`
measured +85.8 mm with, and `frozen.py` takes it unchanged now that
`freeze_sets` accepts more than one pose (`freeze` is the N = 1 case, which is
what a *parked* partner is).

**What makes it affordable.** A row band's envelope is some 9 000 capsules and
`frozen.partner_clearance` is linear in them, which would put a tenth of a
second on every `paper.route` call. `staged.cluster_capsules` bounds them by
grid-local **spheres** — each sphere contains both endpoints and the radius of
every capsule whose midpoint fell in its 0.15 m cell — so the reduction is
conservative by construction: a query that clears the spheres clears the
capsules. A row band comes out as a few hundred spheres. The poses are read at
the shipped stride 2 (4 cm) and every sphere is inflated by `ENVELOPE_PAD` =
40 mm for the cells the stride skips. The whole set is cached under the atlas,
the tool and the region, because it is a property of the **stage** and not of
the picture.

**And the leg store is namespaced on it.** `paper.route_key` contains neither
the frozen poses nor the envelopes — both change `static_boxes` without changing
any memo key — so a leg bought in one room would otherwise be served in another.

**The ink gets its own check, and it is a measurement rather than a gate.**
`plan_stroke` never consults the static set, so a piece can be certified end to
end and still be drawn through a neighbour's envelope; `staged.ink_vs_envelope`
is the missing half. It is reported and not enforced by default, because the
envelope it compares against is *deliberately larger than the thing it bounds*
(stride-2 sample, sphere-bounded, 40 mm pad) and refusing certified metres to a
conservatism is a bad trade. Measured on CSAIL: median **+230.9 mm**, 5th
percentile **+9.4 mm**, minimum **−28.5 mm**, and **5 pieces of 48** read under
50 mm — all of them against the inflated bound, none of them against another
arm's actual ink. `ink_gate=PAIR_MARGIN` turns it into a refusal for a caller
who wants the strict reading.

**The second lever, if the first is not enough.** `staged.row_lift_ladder` gives
each **row band** its own hover height (0.06 / 0.20 / 0.34 m), with the shipped
ladder behind it so an arm that cannot hold the raised pose keeps the one it
had. `writing.HOVER_LADDER` is borrowed for the stage and given back, and
`lifted_or_lower`'s memo is keyed on the heights, so it moves that stage's legs
and nothing else. `run(lift_retry=True)` fires it only on a stage whose pair
check still fails, keeps whichever of the two is better, and records
`lift_used`.

## 9. The refusal loop

A piece `plan_stroke` refuses is **not ink nobody can draw** — it is ink *that
arm* cannot draw *in that stage*, which is one bit of one atom's capability set.
`staged.resolve_refusals` strikes that bit and asks the DP again, to a fixed
point or a cap of four rounds. Nothing is flown inside the loop: the loop is
over the pieces, and a plan memo keyed on the arm, the geometry and the room
means a round only pays for the pieces that actually moved.

Two things had to be right, and each was measured wrong first:

- **A refusal bans only the stretch the planner did not certify.** `plan_stroke`
  hands back `s_star`, "the normalised arc length up to which it IS planned",
  and a head plan is an "ok" result with all of its guarantees. Banning the
  whole piece throws the head away too.
- **A ban CUTS the atom rather than clearing its bit.** An atom is indivisible
  with respect to the capability map and a refusal is a new transition in it, so
  a ban that merely touches an atom must split it at both ends. Measured on the
  CSAIL logo, banning whole atoms took the loop from 100 % coverage to
  **80.6 %**; cutting them is what keeps the ink.

A `degenerate` refusal — "too short", "off sheet", "too short after clip" — bans
nothing and is reported as **unplannable**: those are statements about the
*piece*, not about the arm, so no neighbour would do better and striking the
state out would only spread the hole. A span is never banned twice, which is
what makes the loop terminate rather than walk one piece down the state list.

## 10. The barrier cost at scale, and the rule that should replace it

CSAIL's staged makespan is 1.87× v19's conducted one, and the reason is the
barrier: a stage costs its busiest arm, the sequence costs the **sum** of eight
of those, and every active arm of every stage pays a park → out → back trip
whatever it has to draw. `staged.stage_overhead` measures that trip directly —
the entry leg before the first stroke and the exit leg after the last, per
(stage, arm).

**The proposal, not implemented: an adaptive stage count.** The overhead is a
fixed cost per (stage, arm) and the ink is not, so the eight-stage pattern is
right at scale and wrong on a small picture. The rule:

> Let `P` be the measured park overhead per (stage, arm) — CSAIL's own number —
> and let `m` be a stage's ink for its busiest arm. A stage whose busiest arm
> carries less than `X = P × v_draw` metres is **not worth its own barrier**:
> the trip out and back costs more clock than the ink it protects.

Three ways to spend that, cheapest first:

1. **Merge two seam stages that share no arm.** Stages 2 and 4 are 13 + 97 and
   31 + 97; stages 3 and 5 are 17 + 2 and 71 + 2. Two stages that share an arm
   cannot merge, but two that do not can, subject to the same envelope test the
   three-active stages already pass — and the merge saves one whole barrier.
2. **Let an arm stay out between consecutive stages it is active in.** The
   barrier requires every arm at *its own next park*; an arm whose park does not
   change between stage *s* and stage *s+1* is already there, and
   `DECISIONS.md` (2026-09-11) measured that **26 of 42 transitions are exactly
   that**. Those arms should not fly home and out again.
3. **Fall back to the conductor for the residue.** When what is left is a
   handful of short stages, `idle.conduct` over two arms is `∑ₖ P(n, k) = 4`
   orders over a short horizon — cheap, and it removes the barrier entirely.

The number `X` is the one thing a measurement has to supply.

## 11. The CSAIL re-run with the envelope room

Same command, `traces.zigzag_pattern()`, h = 0.970, the refusal loop on and the
envelope room installed. **The leg gap is closed and the makespan question is
now a different one.**

| stage | actives | pieces | ink (m) | stage (s) | **active-pair (mm)** | ink vs ink (mm) | solo (mm) | lift used | flown |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 13, 71, 2 | 15 | 5.243 | 25.8 | **+899.3** | — | +250.3 | no | 1 of 3 |
| 1 | 17, 31, 97 | 12 | 4.136 | 8.9 | **+826.1** | — | +214.3 | no | 1 of 3 |
| 2 | 13, 97 | 6 | 1.461 | 48.0 | **+977.2** | +1 055.3 | **+28.2** | no | 2 of 2 |
| 3 | 17, 2 | 5 | 1.145 | 29.4 | **+870.9** | +1 151.2 | +96.2 | no | 2 of 2 |
| 4 | 31, 97 | 5 | 1.083 | 31.8 | **+897.0** | — | +140.2 | no | 1 of 2 |
| 5 | 71, 2 | 4 | 0.923 | 56.5 | **+860.8** | — | +65.3 | no | 1 of 2 |
| 6 | 13, 31 | 4 | 0.889 | 28.4 | **+626.1** | — | +206.7 | no | 1 of 2 |
| 7 | 17, 71 | 2 | 0.959 | 0.0 | **+539.8** | — | +256.0 | no | 0 of 2 |

**The active-pair minimum over all eight stages is +539.8 mm, against +8.2 mm
before.** The two failing stages went +12.9 → **+899.3** and +8.2 → **+826.1**,
and the **lift lever was never needed** — `lift_used` is false everywhere, so
routing against the envelopes was sufficient on its own and the row-band hover
ladder stays in the box as the fallback it was built to be. Seven of the eight
stages pass both checks; stage 2 still reads **+28.2 mm** on the *parked*
side, which is the one failure this pass did not touch and is unchanged from §3.

**And the envelope is too conservative to fly the big buckets.** Three of the
eighteen (stage 0's arm 71 with 13 pieces, stage 1's arm 31 with 11, stage 7's
arm 71 with 2 — **26 of the 56 pieces**) produced **no timeline at all**:
`writing.arm_program` refuses the entry leg, and in one case `sequence.solve`
could find no feasible order before that. The cause is named by the ink check
itself — the minimum ink-vs-envelope reading is **−28.5 mm**, i.e. the arm's own
certified drawing pose is already *inside* the inflated bound, and no leg out of
a pose inside an obstacle can clear it. The conservatism is the 0.15 m sphere
clustering plus `ENVELOPE_PAD` = 40 mm, both of which were chosen for speed
rather than tightness.

**So the makespan is not comparable and is not quoted as one.** 228.8 s over a
programme missing 26 of 56 pieces is a partial number, and the honest statement
is that the makespan question re-opens once the envelope is tight enough to fly.
What *is* measurable on the nine buckets that did fly is the barrier cost:

| | |
|---|---|
| park → out → back, all flown buckets | **87.7 s** |
| …on the critical path (the busiest arm of each stage) | **75.3 s** |
| …as a share of the makespan | **32.9 %** |
| draw, all flown buckets | 103.9 s |
| pen-up, all flown buckets | 175.3 s |

**A third of the clock is the barrier**, and pen-up is 1.7× the ink.

**The refusal loop, measured:** round 0 → 54 pieces, 5 refused, 100.00 %;
round 1 → 56, 4, 98.42 %; round 2 → **56, 0, 94.20 %**, converged, with
**3 pieces unplannable and all three `degenerate:too_short`**. The 5.8 % of ink
lost is not the loop giving up — it converged with zero refusals outstanding —
it is ink with **no second stage-compatible drawer**: the staged map's
redundancy is 48.1 % (`docs/V2_TRACES.md` §5), so when the one arm a stretch is
offered to refuses it, there is nobody to hand it to. Closing that needs the
*pattern* to offer the stretch to somebody else. **The final refusal rate is
3 of 56, 5.4 %**, against 11.1 % before the loop.

Time to first motion is **0.286 s**. Planning is **1 549 s**, because the room
changed and every leg in it is cold — the leg store is namespaced on the room,
by design, so this is the one-off price of the new certificate.

## 12. A thousand lines, staged — the model

A real 1 438-piece fly is not affordable: `sequence.cost_matrix` is O(n²) route
screens and one bucket is 434 pieces. So the makespan is assembled from terms
measured on the CSAIL run and the 1 000-stroke set's own per-(stage, arm) ink
and piece counts from `traces`. **Arithmetic, with every term measured.**

Calibration, from the nine flown buckets: draw **6.56 s/m**, inter-piece pen-up
leg **5.15 s**, park overhead **9.74 s per (stage, arm)**. Accept rate 81.9 %,
measured in §5.

| stage | actives | pieces | ink (m) | busiest arm | stage (s) |
|---|---|---|---|---|---|
| 0 | 13, 71, 2 | 572 | 258.0 | 2 | 1 327.7 |
| 1 | 17, 31, 97 | 549 | 257.2 | 97 | 1 313.8 |
| 2 | 13, 97 | 162 | 36.6 | 13 | 451.5 |
| 3 | 17, 2 | 146 | 35.0 | 2 | 421.5 |
| 4 | 31, 97 | 75 | 17.9 | 31 | 417.4 |
| 5 | 71, 2 | 76 | 15.7 | 71 | 409.8 |
| 6 | 13, 31 | 91 | 20.7 | 31 | 499.7 |
| 7 | 17, 71 | 85 | 21.3 | 71 | 477.7 |

**Staged makespan (model): 5 319 s = 88.7 min**, against
`V2_SCALING_BASELINE` §3.3's ~4 300 s scaling of v19's conducted six-arm
makespan — **1.24×**, where CSAIL measured 1.87× before this pass.

**And the barrier amortises exactly as the rule predicts.** The park overhead on
the critical path is **77.9 s, 1.47 % of the makespan**, against CSAIL's
**32.9 %**. The barrier is a fixed cost per (stage, arm) and the ink is not, so
**the eight-stage pattern is right at scale and wrong on a small picture** —
which is the whole case for the adaptive stage count of §10, and it supplies
its threshold: at 6.56 s/m and 9.74 s of park overhead, **`X` ≈ 1.5 m of ink
for the busiest arm**. Six of CSAIL's eight stages are under it; none of the
1 000-line set's are.

## 13. How tight the envelope has to be — the probe, and what it says so far

**The binding test is one leg, not a whole timeline.** Every bucket that failed
in §11 failed on `writing.arm_program`'s *"entry at segment 0 cannot clear the
paper plane"*, which is a single `paper.route` from the park to the first hover.
So the sweep prices exactly that, plus the exit leg, plus the ink-vs-envelope
minimum that explains it — three numbers per (stage, arm) instead of a fly. The
pieces are planned once with no envelope in the room, because `plan_stroke`
never consults the static set and a piece's joints therefore do not depend on
the setting; only the verdict about them does.

**Measured, at the shipped setting** (stride 2, cluster cell 0.15 m, pad
40 mm), on the 56-piece refusal-loop output:

| setting | spheres (mean) | buckets whose park↔hover legs route | ink vs envelope, min |
|---|---|---|---|
| **stride 2, cell 0.15 m, pad 40 mm (shipped)** | 145 | **9 of 13** | **−30.6 mm** |

That is the §11 failure reproduced in 1.4 s instead of 27 minutes, and it
confirms the diagnosis: four buckets cannot get out of their park at all, and
the ink minimum is negative, i.e. the arm's own certified drawing pose is
already inside the bound.

**Why the cell is the lever and the pad is not, exactly.** A cluster sphere is
drawn round the endpoints of every capsule whose *midpoint* fell in its cell, so

> `radius ≤ (cell × √3 / 2) + (half the longest capsule) + (the capsule radius)
> + (the pad)`

and the only term the caller controls is the first. At the shipped 0.15 m cell
it is **0.130 m**; at 0.05 m it is **0.043 m**. Against a real capsule radius of
about 0.06 m, that is the difference between a bound three times the thing it
bounds and one a third larger — an order more than the 40 mm pad is worth.
`tests/test_staged.py` pins the decomposition.

**And `pad = 0` is only legal at stride 1.** The pad exists to cover the poses a
stride skips: at stride 2 three cells in four are not read and the pad is the
only thing standing in for them, while at stride 1 every strict-GO cell of the
region is read (`2 × cells + 1` poses, pinned by a test) and the envelope is
then the exact object `scripts/workcell_envelopes.py` measures. **So the
setting the arithmetic points at is stride 1, cell 0.05 m, pad 0** — tighter by
roughly 130 mm of radius and with no conservatism left to justify.

**What is NOT measured, and it is the headline of this pass.** The sweep did not
complete: each setting re-routes 26 legs cold under a new store namespace, and
several fall through the shape ladder to the RRT, which is minutes rather than
seconds per leg. So I have the baseline row and the arithmetic, and I do **not**
have the measured "fraction of buckets that fly" at the tighter settings, the
active-pair minimum under them, or a full eight-stage fly. **The complete CSAIL
table of §11 therefore still stands as the last measured one**, with 26 of 56
pieces unflown, and the makespan is still not quoted.

**One real fix came out of the attempt.** The pose cache and the sphere cache
were filed under one key, so sweeping the clustering re-solved every hover —
the same family of bug as `id(spec)` in `paper.py`, found the same way, by a
sweep that should have taken seconds and took minutes.
`staged.cached_envelope_poses` splits them, and the sweep's per-setting cost is
now the legs alone.

**Next, in order:** run the sweep with the leg store pre-warmed per setting (or
with `paper.RRT_SAFE` off for the probe, since a leg the ladder settles is the
one that matters for "does it fly"); adopt the winning setting as the default;
then the full eight-stage fly and the makespan.

## 14. The sweep, and the setting that flies

**The probe, with the RRT tier off.** What decides "does this bucket fly" is
whether the **shape ladder** settles the park → hover leg: the C-space tier is
the last resort, it costs minutes when it fires, and a setting that needs it for
the entry leg of every bucket is not one anybody would ship. So the sweep runs
at `paper.RRT_SAFE = False`, which makes it an honest **lower** bound — a
setting the probe passes certainly flies — and the full run puts the tier back
on. `paper.cache_signature()` carries the three tier flags, so nothing the probe
buys is served to a run that has the tier on.

Stride 1 throughout, because `pad = 0` is only honest there (§13): at stride 1
every strict-GO cell of the region is read and there is no skipped pose for a
pad to stand in for.

| stride | cell (m) | pad (mm) | spheres (mean) | **buckets whose park↔hover legs route** | ink vs envelope, min | probe (s) |
|---|---|---|---|---|---|---|
| 1 | 0.15 | 40 | 156 | 7 of 13 | -46.3 mm | 21.9 |
| 1 | 0.15 | 20 | 156 | 7 of 13 | -26.3 mm | 44.1 |
| 1 | 0.15 | 0 | 156 | 8 of 13 | -6.3 mm | 42.9 |
| 1 | 0.1 | 40 | 395 | 7 of 13 | -13.1 mm | 73.6 |
| 1 | 0.1 | 20 | 395 | 7 of 13 | +6.9 mm | 79.6 |
| 1 | 0.1 | 0 | 395 | 7 of 13 | +26.9 mm | 76.4 |
| 1 | 0.075 | 40 | 754 | 7 of 13 | -12.2 mm | 102.5 |
| 1 | 0.075 | 20 | 754 | 7 of 13 | +7.8 mm | 107.7 |
| 1 | 0.075 | 0 | 754 | 7 of 13 | +27.8 mm | 116.6 |
| 1 | 0.05 | 40 | 1926 | 7 of 13 | -0.1 mm | 218.0 |
| 1 | 0.05 | 20 | 1926 | 7 of 13 | +19.9 mm | 250.1 |
| 1 | 0.05 | 0 | 1926 | 7 of 13 | +39.9 mm | 222.8 |

**What the sweep says, in three readings.**

*The pad is worth about what the arithmetic said and no more.* At the 0.15 m
cell, dropping the pad from 40 mm to 0 moves the ink minimum from **−46.3 mm**
to **−6.3 mm** and buys exactly **one** bucket (7 → 8 of 13). Forty millimetres
of radius on every sphere is forty millimetres, and it is not where the problem
is.

*Stride 1 is stricter than stride 2, and that is the point.* At the same cell
and pad, stride 1 flies **fewer** buckets than stride 2 did (7 against 9),
because it reads every certified cell instead of one in four: the stride-2
envelope was smaller because it was **sampling**, not because it was tighter.
A setting that flies only by not looking is not a setting.

*The cell is the lever, as the radius decomposition said it would be.*

At the 0.15 m cell, dropping the pad from 40 mm to 0 moves the ink minimum
from **−46.3 mm** to **−6.3 mm** and buys exactly **one** bucket. Shrinking the
cell from 0.15 to 0.05 at pad 0 moves it from −6.3 mm to **+39.9 mm** — an
**86 mm** swing end to end, exactly the term the radius decomposition named —
and buys **none**.

**So the residual refusal is not the envelope's conservatism.** The same six
buckets fail at every cell from 0.10 m down, with the ink comfortably clear of
the bound and every park clear of it. What is left is that a pen-up leg from the
park to the first hover, *routed by the shape ladder alone*, genuinely
intersects the other actives' work-cell envelopes: two arms in adjacent row
bands own overlapping airspace between the parks and the paper, and no amount of
tightening a bound that the ink already clears by 28 mm will change that.

**The probe is a lower bound, and the tier it turns off is precisely the one
that might answer this.** `aris_sixarm/transit.py` searches the 7-DOF joint
space where the ladder gives up, and every edge it proposes pays the same gate
stack a ladder leg pays. Whether it finds a way round is the question the full
fly answers, and it is running.

### The park is the other wall, and tightening moves it too

A leg starts at the park, so if the **park itself** is inside a partner's
envelope no leg out of it can clear at any tightness. Measured, park against the
other actives' envelopes, at the shipped setting and at the tight one:

PARK_| stride | cell (m) | pad (mm) | spheres (mean) | **buckets whose park↔hover legs route** | ink vs envelope, min | probe (s) |
|---|---|---|---|---|---|---|
| 1 | 0.15 | 40 | 156 | 7 of 13 | -46.3 mm | 21.9 |
| 1 | 0.15 | 20 | 156 | 7 of 13 | -26.3 mm | 44.1 |
| 1 | 0.15 | 0 | 156 | 8 of 13 | -6.3 mm | 42.9 |
| 1 | 0.1 | 40 | 395 | 7 of 13 | -13.1 mm | 73.6 |
| 1 | 0.1 | 20 | 395 | 7 of 13 | +6.9 mm | 79.6 |
| 1 | 0.1 | 0 | 395 | 7 of 13 | +26.9 mm | 76.4 |
| 1 | 0.075 | 40 | 754 | 7 of 13 | -12.2 mm | 102.5 |
| 1 | 0.075 | 20 | 754 | 7 of 13 | +7.8 mm | 107.7 |
| 1 | 0.075 | 0 | 754 | 7 of 13 | +27.8 mm | 116.6 |
| 1 | 0.05 | 40 | 1926 | 7 of 13 | -0.1 mm | 218.0 |

**Arm 71's park is +1.2 mm from arm 17's envelope in stage 7** at the shipped
setting — touching it, which is exactly why that bucket failed at every pad and
every cell down to 0.10 m. At the tight setting the same pair reads **+52.8 mm**
and clears the gate, and every other park is at the +350 mm broad-phase cap or
close to it. So the park set does **not** have to be re-searched
(`layout.stage_parks`, build item 2) — it was the envelope's conservatism
standing on the park, not the park standing in the wrong place. That is worth
saying plainly, because the opposite conclusion would have sent the next pass
after a park search it does not need.


**Adopted: stride 1, cell 0.075 m, pad 0.** Not because it flies the most
buckets — 0.15/0 flies one more — but because it is strictly the better
*certificate* and the bucket count does not distinguish them: no skipped poses
(so `pad = 0` is honest rather than optimistic), no unjustified inflation, an
ink minimum of **+27.8 mm** instead of −46.3, and **754 spheres against 1 926**
at 0.05 m for 11 mm more. The bucket count at 0.15/0 being one higher is the
shape ladder's discrete shapes interacting with a looser obstacle set, not a
tighter guarantee.

**And one more O(n²) came out of it.** `cluster_capsules` scanned every endpoint
once per cell (`inv == k`), which at stride 1 is a few hundred thousand
endpoints against tens of thousands of cells — minutes per arm, and what made
the first attempt at this sweep look like it had hung. It is now one
`np.minimum.at` / `np.maximum.at` pass each.

**The full eight-stage fly at the adopted setting is running** to
`out/staged_csail_h097_v3.json`, with the RRT tier **on**. Its envelopes are
1 157–1 377 spheres per active arm in the main stages and 472–506 in the seam
stages, and every leg in it is cold because the room changed.

## 15. The room is what the neighbour actually does

§14 measured the pose-union envelope out. Tightening it from a 0.15 m cluster
cell with a 40 mm pad to 0.05 m with none moved the ink-vs-envelope minimum
**86 mm** (−46.3 → +39.9), moved every park clear of the gate (arm 71 in stage 7
from **+1.2 mm** to **+52.8 mm**), and moved the number of buckets that fly **by
nothing** — the same six refused at every cell from 0.10 m down. The full fly at
the adopted setting, with the RRT tier back **on**, refused them too.

**So the union is the wrong object, not a badly bounded one.** An arm's work
cell is every pose it *could* hold anywhere inside it; two arms in adjacent row
bands own overlapping airspace between the parks and the paper; and no bound on
a set that large leaves a neighbour room to fly through it. What the neighbour
*actually* holds in a stage is one trajectory — a few hundred poses out of that
union — and that is a room a leg can be routed around.

**The construction.** `staged.trajectory_room` takes an arm's realised stage
timeline, builds its link capsules through the same `coordination.ArmPath` the
envelope used, and reduces them with the same `cluster_capsules` — which
*contains* what it replaces, so the certificate survives the reduction. The pad
is not a guess: it is `scene_check.check_timeline`'s own 1-Lipschitz
between-sample residual, `SWEEP_FRAC ×` the largest step any capsule endpoint
takes between two samples, so the spheres cover the motion **between** the
samples and not only at them.

**Two passes, and the iteration is explicitly not the certificate.** Pass 1
plans every active solo against the parked fleet — the room that flies but does
not separate the actives. Pass 2 re-plans each active against what the *others
actually did* in pass 1. Re-planning A moves A, which is a room B was certified
against, so the fixed point is approached and never proved by the iteration; the
claim is closed only by the independent whole-timeline `active_pair_gap` at
`PAIR_MARGIN`, exactly as before. The iteration gets the trajectories apart; the
check proves they are apart.

**And the barrier semantics change, so the dependency is recorded.** A
pose-union envelope is a property of the **stage**: it holds whatever the
neighbour is asked to draw, so an arm's plan survives its neighbour being
re-planned. A trajectory room is a property of that neighbour's **specific
plan**, and an arm's certificate is void the moment that plan changes.
`ArmStage.depends_on` carries a digest of each neighbour's trajectory and
`programme()` writes it out next to the arm's own `trajectory_digest`, so a
re-plan of one arm leaves every neighbour that still names the old digest stale
**by inspection** rather than silently. That is the price of the tighter room
and it is worth naming: the staged programme is no longer a set of independent
per-arm certificates, it is a graph of them.

**How much smaller the room actually is.** Stage 0 of the CSAIL logo, the
same stage whose arm-71 bucket refused at every envelope setting:

| arm | pieces it draws | pose-union envelope | **trajectory room** |
|---|---|---|---|
| 2 | 1 | 1 339 spheres | **422** |
| 13 | 1 | 1 377 spheres | **111** |
| 71 | 13 | 1 157 spheres | **1 001** |

The counts understate it. An arm that draws one piece had its whole row band
treated as a keep-out; what it actually occupies is one approach, one stroke and
one retreat. Arm 13's room falls by a factor of **12**, and arm 71 — which draws
thirteen pieces spread across R1 — barely moves, which is the honest shape of
the thing: the saving is exactly the ink an arm was *not* asked to draw.

**And pass 1 flies everything, which is the premise the iteration needs.**
Planned solo against the parked fleet, arm 71's thirteen-piece bucket produces a
timeline — the same bucket that produced none under the union envelope at any
cluster cell, with the RRT tier on or off. So there is a pass-1 trajectory to
derive a room from, for every active arm, which is what the two-pass scheme
assumes and what §14 could not provide.

**Numbers still owed.** The CSAIL run at `--trajectory-rooms` was still in pass
2 when this box closed, so the complete table — buckets flown, per-stage
minima under both checks, the real makespan against 391.7 s and v19's 209.9 s,
the cold/warm planning split — is not in hand, and neither is the recalibrated
1 000-stroke model. It is running to `out/staged_csail_h097_v4.json`, and the
union-envelope fly at the adopted tight setting is running alongside it to
`out/staged_csail_h097_v3.json` as the control.


## 16. The control: the union envelope, tightened, with the tier on

`out/staged_csail_h097_v3.json` — the adopted stride 1 / cell 0.075 / pad 0
envelope, RRT tier **on**, refusal loop on. This is the run §14 promised and it
closes option (1) for good.

| stage | actives | pieces | ink (m) | stage (s) | active-pair (mm) | solo (mm) | **flown** |
|---|---|---|---|---|---|---|---|
| 0 | 13, 71, 2 | 15 | 5.243 | 25.8 | +899.3 | +250.3 | **1 of 3** |
| 1 | 17, 31, 97 | 12 | 4.136 | 9.8 | +819.5 | +244.1 | **1 of 3** |
| 2 | 13, 97 | 6 | 1.461 | 48.0 | +977.2 | **+28.2** | 2 of 2 |
| 3 | 17, 2 | 5 | 1.145 | 29.4 | +870.9 | +96.2 | 2 of 2 |
| 4 | 31, 97 | 5 | 1.083 | 31.8 | +897.0 | +140.2 | 1 of 2 |
| 5 | 71, 2 | 4 | 0.923 | 53.9 | +857.0 | +69.8 | 1 of 2 |
| 6 | 13, 31 | 4 | 0.889 | 0.0 | +639.8 | +256.0 | **0 of 2** |
| 7 | 17, 71 | 2 | 0.959 | 0.0 | +539.8 | +256.0 | **0 of 2** |

**8 of 18 buckets flew, carrying 5.445 m of the 15.838 m allocated — 34.4 % of
the ink.** The active-pair gap is enormous everywhere (+539.8 mm at worst)
precisely *because* two thirds of the ink never got a trajectory: the arms that
would have been close to each other are the ones standing at their parks.

Three readings, and the third is about this document rather than the rig.

*The tightening did not help and marginally hurt.* v2, at the loose 0.15 m /
40 mm setting, flew **9** of 18; v3 at the honest tight one flies **8**. That is
the sweep's non-monotonicity showing up in the fly, and it is why §14 adopted on
the certificate rather than on the bucket count.

*The cost is real.* Planning is **2 973.9 s** cold (49.6 min), **2 869.6 s**
charged per stage to its busiest arm, against v2's 1 549.4 s — the tighter room
has 3–12× the spheres and every clearance query pays for them. Time to first
motion is unaffected at **0.294 s**, and the refusal loop is unchanged
(**94.20 %** coverage, 3 of 56 pieces unplannable, all `degenerate:too_short`).

*The makespan is 198.66 s and it is not a makespan.* It is the sum of eight
stage durations two of which are zero because nothing flew. **Comparing it with
391.7 s or with v19's 209.9 s would be comparing a third of a programme with two
whole ones**, and the earlier drafts of this document came close to doing
exactly that. Which exposes a defect in the report itself: **a stage in which
nothing flew passed both checks**, because six arms at their parks clear
everything. Stages 6 and 7 were labelled PASS on that basis. `StageResult.ok`
now requires `complete` — every bucket that has ink produced a timeline — and
the stage line prints `flown/with_ink` and says EMPTY rather than PASS.

## 17. Priority order: one sweep, an exact fixed point

The simultaneous two-pass scheme of §15 was measured out on the way past. On
CSAIL stage 0, arm 71's thirteen-piece bucket **flew in pass 1** against the
parked fleet and **stopped flying in pass 2**, once its neighbours' trajectories
became obstacles. The scheme asks every arm to yield to a path the other has
already abandoned, so nobody actually yields; it is circular, and no number of
iterations closes a cycle.

**A priority order closes it in one sweep.** The actives of a stage are ordered
by ink, busiest first, because the busiest arm has the least room to give:

- arm 1 plans against the parked fleet, and its trajectory is then **final**;
- arm *k* plans against the parked fleet **plus the final trajectories of arms
  1…*k*−1**.

When arm *k* finishes, every pair (*i*, *k*) with *i* < *k* is certified against
the path arm *i* actually flies — and arm *i* never moves again. Every pair is
therefore certified, exactly, with no iteration and no circularity. The
dependency graph stops being a cycle and becomes a **DAG**: arm *k* depends on
1…*k*−1 and on nothing after it, which is also what makes a re-plan's blast
radius finite (re-planning arm *k* invalidates only *k*+1…*n*).

It is the spatial analogue of what `coordination.coordinate` already does in
time with its priority search, with the order fixed by ink rather than searched,
because a stage has at most three actives and the busiest is the obvious choice.
The superseded scheme stays behind `--room-order simultaneous` so the
measurement that retired it can be reproduced.

**The certificate is unchanged**: the independent whole-timeline
`active_pair_gap` at `PAIR_MARGIN` is still what closes the claim. The order is
what gets the trajectories apart; the check is what proves they are.


## 18. Stage 2's +28.2 mm, attributed at last

Flagged three times without an answer, and now measured: it is **arm 97's
trajectory passing 28.2 mm from *parked* arm 2, at t = 2.2 s**. Arm 13, the
other active, is clean at +234.9 mm. `column_failed: [97]` is the same geometry
seen through the base-column cylinder rather than a second defect.

**It is a genuine active-vs-parked violation, not a modelling artefact**, and it
is untouched by every pass of this work — the envelope, the tightening and the
priority room are all about active-vs-*active*.

The mechanism is a **router/checker mismatch**, and it is worth stating because
it is systematic rather than particular to this leg. `paper.route` certifies a
pen-up leg at `STATIC_MARGIN` = 50 mm *at the samples it evaluates*;
`scene_check.check_timeline` re-derives the same leg and additionally subtracts
the 1-Lipschitz between-sample residual, `0.55 × (step_i + step_j)`. A leg that
is exactly at the gate at its samples therefore reads roughly 20 mm tighter once
the sweep between them is charged — which is what 50 → 28.2 mm looks like. The
checker is right and the router is optimistic by the sweep.

Two fixes, neither implemented: charge the router the same residual (raise its
floor by the sweep it will be measured against, which is a constant it already
has in `coordination.SWEEP_K`), or sample the leg finely enough that the
residual is negligible — the same auto-refinement `check_timeline` already does
for its frame and paper gates. The first is one number; the second is the
honest one.

**An earlier attempt at this attribution was wrong and is worth recording as
such.** It fed the programme's raw waypoints to `check_timeline` at a fixed dt,
which charges a Lipschitz residual against jumps the timeline never makes, and
reported two *parked* arms at −437 mm. The rebuild goes through
`writing.uniform_samples`, exactly as `staged.solo_check` does.


## 19. The priority room, measured: 97.6 % of the ink flies

`out/staged_csail_h097_v5.json` — priority-ordered trajectory rooms, RRT tier
on, refusal loop on. **This is the run the whole room line of work was for.**

| stage | actives (priority order) | pieces | ink (m) | stage (s) | **active-pair (mm)** | ink vs ink (mm) | solo (mm) | **flown** | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 71, 13, 2 | 15 | 5.243 | 107.2 | **+269.3** | +411.1 | +80.7 | 2 of 3 | **FAIL** |
| 1 | 31, 97, 17 | 12 | 4.136 | 89.8 | **+158.5** | +417.5 | +60.9 | 2 of 2 | PASS |
| 2 | 13, 97 | 6 | 1.461 | 48.0 | +977.2 | +1 055.3 | **+28.2** | 2 of 2 | **FAIL** |
| 3 | 2, 17 | 5 | 1.145 | 29.4 | +870.9 | +1 151.2 | +96.2 | 2 of 2 | PASS |
| 4 | 31, 97 | 5 | 1.083 | 31.8 | +897.0 | — | +140.2 | 1 of 1 | PASS |
| 5 | 71, 2 | 4 | 0.923 | 58.6 | +623.8 | — | +72.5 | 1 of 1 | PASS |
| 6 | 31, 13 | 4 | 0.889 | 26.5 | +263.7 | — | +71.2 | 1 of 1 | PASS |
| 7 | 71, 17 | 2 | 0.959 | 23.6 | +525.3 | — | +80.9 | 1 of 1 | PASS |

**12 of 13 ink-carrying buckets fly, carrying 15.455 m of 15.840 — 97.6 % of
the ink**, against the union envelope's 8 of 18 and 34.4 %. **Six of eight
stages pass both checks.**

| | v3, union envelope | **v5, priority rooms** |
|---|---|---|
| ink buckets flown | 8 of 13 | **12 of 13** |
| ink in a certified trajectory | 5.445 m (34.4 %) | **15.455 m (97.6 %)** |
| stages passing both checks | 4 of 8 (two vacuously) | **6 of 8** |
| makespan | 198.7 s (a third of a programme) | **415.0 s (a real one)** |
| planning, serial | 2 973.9 s | **263.8 s** |
| planning, per stage's busiest arm | 2 869.6 s | **89.7 s** |
| check | 42.6 s | 82.4 s |
| time to first motion | 0.294 s | **0.287 s** |

**Planning is 11× faster than the control and 32× faster per arm**, which was
not the point but is the largest single number in the table. Two reasons: the
trajectory rooms are 10–1 001 spheres against the union's 1 157–1 377, and the
first arm of every stage plans **free** — a third of the buckets pay nothing for
the room at all.

**The makespan is 415.0 s and it is real** — one bucket of 0.385 m is missing
out of 15.840 m. Against **391.7 s** for the un-separated v1 programme it is
**+5.9 %**, which is what routing three arms around each other costs; against
v19's conducted **209.9 s** it is **1.98×**, the barrier cost that §10's
adaptive-stage rule is aimed at. Park overhead is **92.7 s on the critical path,
22.3 % of the makespan**.

**The two failures are both known and neither is the room.** Stage 0 loses its
*third* arm — 0.385 m, the greedy order's expected weak spot, since the last arm
has the least freedom left. Stage 2 is §18's router-sweep mismatch at +28.2 mm,
which is active-vs-**parked** and untouched by any of this.

### The 1 000-stroke model, recalibrated

Terms measured on v5: draw **13.59 s/m**, inter-piece leg **4.18 s**, park
overhead **10.66 s** per (stage, arm), accept rate 81.9 %.

**Staged makespan (model): 6 380 s = 106.3 min**, against
`V2_SCALING_BASELINE` §3.3's ~4 300 s scaling of v19's conducted six-arm
makespan — **1.48×**. Park overhead on the critical path falls to **85.3 s,
1.34 %** of the makespan, against CSAIL's 22.3 %: **the barrier amortises**, and
the eight-stage pattern is right at scale and wrong on a small picture, exactly
as §10 argued. The draw rate is nearly double the v3 calibration (13.59 against
6.56 s/m) because v5 actually draws the long buckets that v3 never flew — the
earlier figure was calibrated on the short ones that happened to survive.

**Two caveats, stated.** The model is arithmetic with measured terms, not a run:
a real 1 438-piece fly is unaffordable because `sequence.cost_matrix` is O(n²)
route screens and one bucket is 434 pieces. And it assumes the priority order
scales — at 572 pieces in a stage-0 bucket the *last* arm in the order faces a
much larger obstacle than it does here, which is precisely where the one
remaining failure already is.


## 20. Closing the two failures — and correcting one of the two diagnoses

§19 left exactly two defects: stage 0 stranding its *third* arm (0.385 m), and
stage 2's +28.2 mm against parked arm 2. Three things were built for them, and
**the second one's diagnosis was wrong and is corrected here.**

### (1) The order search

The greedy ink-first order has a known weak spot: **the arm that plans last has
the least freedom left.** A stage has at most three actives, so the whole order
space is **six permutations** — cheap, where a six-arm priority search (720
orders, `∑ₖ P(n, k)`) is not. Ink-first is tried first, so a stage that did not
need the search pays one comparison and nothing else; `order_rank` records which
order was taken and `orders_tried` what it cost. `--order-search 1` is the
pre-search behaviour.

### (2) The router was not the optimistic one

The instruction for this pass was to make `paper.route` refine its sampling
until its residual-inclusive bound clears the floor. **It already does**, and
saying so is more useful than building it again:

- `paper.leg_bounds` computes `sample_residual(P)` and returns `m − res` when
  that clears, falling through to `adaptive_static_lb` — real adaptive
  subdivision — in the undecided band;
- `FRAME_FLOOR` = **53 mm** is already `STATIC_MARGIN` plus the checker's
  residual, and `paper.py`'s own header explains both that and the `TIP_SWEEP_PAD`
  version of the same argument.

So where does 28.2 mm come from? **Two places, and I measured which.**
`check_timeline` auto-refines its frame and paper gates and does **not** refine
its **inter-arm** gate — it subtracts `0.55 × (stepᵢ + stepⱼ)` at whatever rate
the timeline was handed in at. Refining stage 2's arm 97:

| dt | frames | min clearance |
|---|---|---|
| 0.05 (as reported in §18) | 961 | **28.23 mm** |
| 0.02 | 2 400 | 36.58 mm |
| 0.01 | 4 799 | **39.38 mm** |

**Eleven millimetres of the deficit was the sampling**, and that half is now
fixed: a verdict under the margin is looked at again, up to three halvings or
40 000 frames, keeping the best bound found. That is `block_screen`'s own rule
— *"a cell between the bounds has to be LOOKED AT rather than believed either
way"* — applied to the one gate that lacked it, and it costs nothing on a run
that passes.

**The other ten millimetres are real, and they are not routing.** The number
converges to ~40 mm, not 50. `paper.effective_static_floor` **clamps the floor
to what the leg's own endpoints have**, deliberately and with a docstring
explaining why: *"a move is never asked to keep more clearance than its own
endpoints have"*, because a floor above the endpoints is not a constraint but a
contradiction that prices every edge `inf`. Arm 97 holds a **pose** about 40 mm
from parked arm 2, and **no amount of routing fixes a pose**. The lever is a
different hover or a different park for arm 2 in stage 2 — which is build item
2's per-stage parks, for real this time and for a reason that did not exist when
that item was written out in §14.

§18 called this "the router is optimistic by its own sweep". That was half
right and the wrong half is the important one: the router's *sweep* is
accounted; the router's *floor* is clamped by a pose it did not choose.

### (3) The residue phase

A bucket no order can fly concurrently does not have to be abandoned — it has to
be **serialised**. It is planned alone against the parked fleet (the room pass 1
always flies in) and its timeline is *appended* to the stage rather than
overlapped, because the other actives are back at their parks by then, which is
what the barrier means. **That is Pete's original final pass**, and it is the
floor under the whole scheme: the worst case of the room work is the
one-arm-at-a-time programme the installation started from.

`idle.conduct` is the named tool and it **reduces to nothing here**: with one
arm moving the priority search enumerates `∑ₖ P(1, k)` = one order and there is
no second mover to schedule against. So the residue is laid down directly and
checked identically — `solo_check` against the parked fleet *is* the certificate
a one-mover conduct would produce — and `active_pair_gap` never sees it, because
nothing else is in the air. `StageResult.duration` is accordingly the busiest
**concurrent** arm plus the sum of the residues.


## 21. v6, measured: every bucket flies, 100 % of the ink, seven of eight stages

`out/staged_csail_h097_v6.json` — priority rooms + order search + residue phase,
RRT tier on, borderline verdicts refined.

| stage | order | orders tried | pieces | ink (m) | stage (s) | active-pair | solo | residue | flown | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 71, 13, 2 | **6** | 15 | 5.243 | 124.8 | +269.3 | +80.7 | **0.385 m (arm 2)** | 3/3 | **PASS** |
| 1 | 31, 97, 17 | 1 | 12 | 4.136 | 89.8 | +158.5 | +60.9 | — | 2/2 | **PASS** |
| 2 | 13, 97 | 1 | 6 | 1.461 | 48.0 | +977.2 | **+40.4** | — | 2/2 | **FAIL** |
| 3 | 2, 17 | 1 | 5 | 1.145 | 29.4 | +870.9 | +96.2 | — | 2/2 | **PASS** |
| 4 | 31, 97 | 1 | 5 | 1.083 | 31.8 | +897.0 | +140.2 | — | 1/1 | **PASS** |
| 5 | 71, 2 | 1 | 4 | 0.923 | 58.6 | +623.8 | +72.5 | — | 1/1 | **PASS** |
| 6 | 31, 13 | 1 | 4 | 0.889 | 26.5 | +263.7 | +71.2 | — | 1/1 | **PASS** |
| 7 | 71, 17 | 1 | 2 | 0.959 | 23.6 | +525.3 | +80.9 | — | 1/1 | **PASS** |

**13 of 13 ink buckets fly. 15.840 m of 15.840 — 100.0 % of the allocated ink.
Seven of eight stages pass both checks.**

| | v3 union envelope | v5 priority rooms | **v6 + search + residue** |
|---|---|---|---|
| ink buckets flown | 8 of 13 | 12 of 13 | **13 of 13** |
| ink in a certified trajectory | 34.4 % | 97.6 % | **100.0 %** |
| stages passing | 4 of 8 (two vacuous) | 6 of 8 | **7 of 8** |
| makespan | 198.7 s (partial) | 415.0 s | **432.6 s** |
| planning, serial | 2 973.9 s | 263.8 s | 769.0 s |
| planning, per stage's busiest arm | 2 869.6 s | 89.7 s | **20.6 s** |
| time to first motion | 0.294 s | 0.287 s | **0.280 s** |

**The makespan is 432.6 s**: +4.2 % on v5 (415.0 s) and **+10.4 % on the
un-separated v1 programme (391.7 s)** — the whole cost of making three arms
provably safe from one another is a tenth of the clock. Against v19's conducted
**209.9 s** it is **2.06×**, which is the barrier, not the rooms; park overhead
is 92.7 s on the critical path, **21.4 %** of the makespan.

**No stage needed a non-ink-first order.** Stage 0 tried all six and kept
rank 0, because *no* order flies it — arms 2 and 71 are mutually exclusive there
whatever the order, and the search's real job turned out to be picking the
cheapest sacrifice (keep arm 71's 4.053 m concurrent, serialise arm 2's
0.385 m). Every other stage flew on the first order and paid one comparison.
**The order search is therefore worth keeping for what it proves, not for what
it finds**: it is what licenses the residue, by showing the residue was not an
order away from being avoidable.

**The residue is 0.385 m — 2.4 % of the ink — and costs 17.6 s** (124.8 against
v5's 107.2 s for stage 0), a 16.4 % premium on that stage and 4.1 % on the
programme. That is Pete's original final pass, used exactly once, for the one
bucket nothing could fly concurrently.

**Planning per arm collapses to 20.6 s** — 139× the union envelope's 2 869.6 s —
because the order search's cost is serial-per-stage while the figure that
matters is what one arm waits for. Serial planning rises to 769.0 s, which is
the six extra stage-plans stage 0's search paid for.

**The one failure is stage 2 and it is fully attributed** (§20): solo +40.4 mm
after refinement recovered 12.2 mm of checker sampling, still 9.6 mm short of
the gate, because arm 97 *holds a pose* ~40 mm from parked arm 2. All three of
this pass's mechanisms are active-vs-active; this is active-vs-parked. **8/8 is
not reachable without a different park or hover for arm 2 in stage 2** — build
item 2, with a concrete reason at last.

### The 1 000-stroke model, recalibrated on v6

Terms: draw **13.81 s/m**, inter-piece leg **4.18 s**, park **10.93 s** per
(stage, arm), accept rate 81.9 %.

**Staged makespan (model): 6 437 s = 107.3 min**, **1.50×** the ~4 300 s scaling
of v19's conducted six-arm makespan. Park overhead on the critical path falls to
**87.5 s, 1.36 %** against CSAIL's 21.4 % — **the barrier amortises**, and the
eight-stage pattern is right at scale and wrong on a small picture, as §10
argued. Arithmetic with measured terms, not a run, and it assumes the priority
order and the residue scale: at 572 pieces in a stage-0 bucket the residue could
be far more than 2.4 %, and that is the number the next scaling pass owes.



## 22. The pattern was not the one that was specified — Pete's leader/follower, built and measured

**Pete Werner, 2026-09-14, on reading §21:** the zigzag is not the pattern he
asked for. His is *"the pattern where we did the 1-2-1 for planning the lead arm
with priority and then the other arm in the column would only try to knock out
lines in its cell that were safe to draw. and then in the end we would do the
coordination of all arms to fill in the gaps if needed"* — and, on the cells,
*"restrict their cells as completely occupied so they can run asynchronously …
for the occupied cells we want to also make sure that the free space motions
never intersect with those … there will likely still be lines we can't draw in
the critical regions in the middle which we will need to handle in a final
pass."* **All six arms move in every main stage.**

### Why the zigzag's premise does not carry the weight it was given

The zigzag parks a leader's same-row partner for the whole stage. The reason
recorded for that (`docs/V2_WORKCELLS.md` §4b, repeated in
`ARCHITECTURE_V2` §2b and in `traces.zigzag_pattern`'s own docstring) is a
measurement of **two full work-cell ENVELOPES** against each other: every arm's
elbow swings within 0.10 m of the mid-line while it draws, so a transverse
pair's envelopes interpenetrate by 135–262 mm and *"no pattern may ever put a
same-row pair in the air together."*

That conclusion is one quantifier too strong. An envelope is the union over
every pose an arm **could** hold anywhere in its cell; the question Pete asked
is about one arm drawing a **restricted subset** while the other's **realised
trajectory** is the occupied volume. §15 already built the second object —
`staged.trajectory_room` — and §19 already measured what it is worth: arm 13's
room is a factor of **12** smaller than its envelope, and priority rooms took
the flown ink from 34.4 % to 97.6 %. The envelope argument was applied to a pair
the room argument was never asked about. That is the whole of the correction.

### What was built — `traces.leader_follower_pattern`, three stages

| stage | who | how they are certified |
|---|---|---|
| **A** | leaders **13, 71, 2** (priority 0–2) + followers **17, 31, 97** (3–5) | the existing priority sweep: arm *k* plans against the FINAL trajectory rooms of arms 1…*k*−1 |
| **B** | the roles swapped | the same |
| **C** | all six, **`idle.conduct`** | the whole conducted timeline, `scene_check.check_timeline` |

The leaders are the **1-2-1** — one arm per row with the columns alternating —
which is the zigzag's own stage 0, so leader-vs-leader separation is the 0.40 m
row dead band, unchanged and re-measured. The new adjacency is the same-row
**leader/follower** pair, and it is exactly where the argument has to be new:
the follower plans against the leader's realised stage trajectory as a static
keep-out, which makes it **asynchronous by construction** and certifies its
free-space legs against the occupied volume with the same room machinery.

**A piece that does not fit is DEFERRED, never serialised.** §20.3's residue
pass flies a stranded bucket alone after the others park; that is right for the
zigzag and is precisely what this pattern exists to avoid, because serialising
inside a stage gives back the concurrency six active arms were meant to buy.
Here the piece leaves the stage: first to the stage where **that same arm
leads** (a follower's remainder is ink it can plan free next time), and only
then to the conducted final pass.

**The bag split is the pattern's one design freedom.** The min-pieces DP hands
each arm a bag by cell; each arm draws part of it leading and part following. A
follower's safe pieces are the ones far from its leader, so an arm draws its
half of the **contested middle** (the paper between the two base columns) as
leader and its **outer strip** as follower. `split_m` moves that boundary
outward from the arm's own base column and is swept.

### And the barrier stopped being a trip

**Pete, the same day, watching the v6 animation:** *"a lot of excessive parking …
once the 1-2-1 arms have their TSP tour we should be just executing that plan as
efficiently as possible."* So under this pattern a stage ends at the hover above
its last stroke and **holds** there; the next stage starts from that pose, and
every arm not yet moving is in its neighbours' static room as the pose it is
**actually holding** rather than as its park. Parks survive where they are
load-bearing — the start of the programme, the end of it, and the fault-recovery
home — and nowhere else.

The park's own guarantee had to be replaced, not dropped. `Q_PARK_PROPOSED` was
searched to be mutually clear, so a park barrier was safe by construction; a
held barrier is wherever the ink happened to end. It is safe by a different
argument — an arm's trajectory room contains its last sample, and every later
arm was routed clear of that room — and `staged.hold_gap` asserts that argument
at every barrier, with `scene_check`'s own capsules and the same 50 mm gate.

### One correction to the check, found by the first run

Six named actives of which most are holding broke `active_pair_gap`. It takes
the **cross product** of two timelines and subtracts a 1-Lipschitz residual
computed on the **decimated** grid, because two asynchronous arms have no common
clock. Against an arm that is standing still there is no cross product to take,
and charging the still arm the moving one's decimation residual cost **30 mm of
a 50 mm gate** on the first CSAIL run and measured nothing: stage 0 read
+22.7 mm where `solo_check` — which measures exactly that pair, at the full rate,
and refines a borderline verdict — read +52.4 mm. `active_pair_gap` now sees
only the arms that move. Still arms are `solo_check`'s and `hold_gap`'s.

### What it measures on the CSAIL logo — and the finding is not the one the design hoped for

The complete run in hand is `out/staged_csail_h097_lf_smoke.json`, split
**+0.15 m**, taken **before** the held barrier and the pair-check correction
landed (so it still pays v6's park trips and reads the pessimistic pair number):

| stage | roles | pieces | ink (m) | stage (s) | deferred (m) | flown |
|---|---|---|---|---|---|---|
| A | lead 13,71,2 / foll 17,31,97 | 6 | 0.763 | 61.9 | **3.770** | 1/1 |
| B | lead 17,31,97 / foll 13,71,2 | 9 | 2.209 | 47.9 | **2.865** | 3/3 |
| C | **conductor, all six** | 50 | **13.097** | 151.2 | 0.000 | 6/6 |

| | v19 conducted | v6 zigzag | **leader/follower** |
|---|---|---|---|
| makespan | 209.9 s | 432.6 s | **261.0 s** (1.24× v19, **0.60× v6**) |
| ink in the main stages | — | 15.840 m | **2.97 m of 16.08 m** |
| ink in the final pass | 16.80 m | 0.385 m residue | **13.097 m (81 %)** |
| follower ink offered | — | — | 7.288 m |
| follower ink that fitted | — | — | **0.806 m — 11.1 %** (A: **0 %**, B: 21.9 %) |
| logo coverage | — | 94.2 % | **95.6 %** |
| time to first motion | 3 712.4 s | 0.280 s | **0.183 s** |
| park overhead, critical path | — | 92.7 s (21.4 %) | 55.0 s (21.1 %) |
| planning, serial | — | 769.0 s | 912.5 s |
| check | — | — | 95.9 s |

**The honest headline: the followers keep almost nothing, and the conducted
final pass ends up drawing the picture.** Stage A's three followers flew **zero
pieces** at every split tried. The pattern does not fail — it is 1.7× faster
than the zigzag and it draws more of the logo — but it does not work the way it
was meant to. What it actually is, on this rig, is *a short two-stage
leaders-only warm-up followed by v19*.

**And the reason is not the per-piece ink gate.** The fourth run of the sweep
turns that gate back into a measurement (`--no-follower-gate`) and its stage A
is **identical to the gated run's, piece for piece and metre for metre**:
15 pieces, 2.061 m, the same three leader buckets flown and the same three
follower buckets deferred. A follower is not losing its ink to "your stroke
passes through the leader's trajectory"; **its bucket produces no timeline at
all** — `writing.arm_program` cannot route the pen-up legs out of the held pose,
into the first stroke and between the pieces with the leaders' rooms in the way.
That is the leg problem of §3 and §8 again, one row down, and it is where the
next pass has to go: the follower needs a hover ladder or an entry pose chosen
*against* the leader's room, not merely checked against it.

### The split sweep

`scripts/lf_sweep.sh` runs four settings concurrently; `scripts/lf_report.py`
prints the tables from the `--json` summaries. Stage A, with the held barrier and
the corrected pair check, **passes both checks at every setting**:

| split | stage-A pieces | stage-A ink (m) | stage-A (s) | active-pair (mm) | solo (mm) | follower buckets flown | verdict |
|---|---|---|---|---|---|---|---|
| **whole** (Pete's literal) | 17 | 4.595 | 129.4 | **+85.0** | +51.2 | 0 of 3 | PASS |
| **+0.00** | 11 | 1.480 | 78.2 | **+192.9** | +52.4 | 0 of 3 | PASS |
| **+0.15** | 15 | 2.061 | 97.0 | **+131.8** | +52.6 | 0 of 3 | PASS |
| **+0.15, gate off** | 15 | 2.061 | 97.0 | +131.8 | +52.6 | 0 of 3 | PASS |

More split is more leader ink and a wider, slower stage A, monotonically — and
it is monotone in the clearance too, because a leader drawing further out draws
closer to its partner's held pose. **No setting buys a single follower bucket**,
which is the same statement the gate control makes from the other side.

### The whole-bag run, complete — and the number that settles it

`out/staged_csail_h097_lf_whole.json`, Pete's literal baseline, with the held
barrier and the corrected pair check:

| stage | who | pieces | ink (m) | stage (s) | active-pair | solo | conducted | deferred (m) | flown | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| A | lead 13,71,2 / foll 17,31,97 | 17 | 4.595 | 129.4 | **+85.0 mm** | +51.2 mm | — | 5.026 | 3/3 | **PASS** |
| B | lead 17,31,97 / foll 13,71,2 | 1 | 0.028 | 17.3 | — (one mover) | +73.7 mm | — | 4.997 | 1/1 | **PASS** |
| C | **conductor, all six** | 40 | 11.458 | 126.6 | — | — | **+44.9 mm** | 0.000 | 3/6 | **FAIL** |

| | v19 | v6 zigzag | **leader/follower, whole bag** |
|---|---|---|---|
| makespan | 209.9 s | 432.6 s | **273.3 s** (1.30× v19, **0.63× v6**) |
| **park overhead, critical path** | — | 92.7 s, **21.4 %** | **30.9 s, 11.3 %** |
| barrier held-pose clearance | — | (parks, +256 mm) | **+256 / +108.2 / +107.0 mm, all PASS** |
| coverage of the logo | — | 94.2 % | **95.6 %** |
| time to first motion | 3 712 s | 0.280 s | **0.189 s** |
| check | — | — | **39.1 s** (95.9 s before the pair fix) |
| leader ink kept | — | — | 4.624 m of 4.624 m |
| **follower ink kept** | — | — | **0.000 m of 5.026 m — 0.0 %** |
| deferred to the final pass | — | 0.385 m residue | **10.023 m** |
| stage C planning + conduct | — | — | **3 075 s of wall** for 126.6 s of motion |

**The held barrier works and it is worth 10 points of makespan.** Park overhead
on the critical path falls from 21.4 % to **11.3 %**, every barrier's held pose
set clears pairwise by **107 mm or better**, and the time to first motion is
unchanged at 0.189 s. That part of Pete's instruction lands cleanly and would be
worth keeping whatever pattern runs above it.

**The follower does not.** Not one of the fifteen follower pieces came near the
gate:

| | min | p05 | median | p75 | max | under the 50 mm gate |
|---|---|---|---|---|---|---|
| **follower** ink vs the room it had to fit | −419.7 mm | −417.6 mm | **−235.7 mm** | −175.2 mm | **−131.9 mm** | **15 of 15** |
| leader ink vs the rooms before it | +249.2 mm | +249.5 mm | +252.6 mm | +277.4 mm | +302.2 mm | 0 of 3 |

The follower's ink is not marginally refused. **The closest piece is 132 mm
inside the occupied volume and the median is a quarter of a metre inside it.**
No gate setting, no hover ladder, no split and no ordering moves a number like
that — which is why the gate-off control changes nothing, and why the leg
refusals and the ink refusals are one fact seen twice.

**So the envelope measurement's substance survives even though its reasoning was
over-general.** §4b's *"no pattern may ever put a same-row pair in the air
together"* was inferred from full envelopes and should have been inferred from
rooms; asked properly, of rooms, on this rig, the answer is the same. The
correction was worth making — the claim is now anchored where it belongs, and
the held barrier and the six-arm stage came out of asking — but the transverse
pair really is unseparable here, and a follower on this geometry has nothing
safe to draw.

**And the final pass is the bill.** With the followers empty, 10.023 m of the
logo falls to stage C, which then costs **3 075 s of wall for 126.6 s of
motion** and still ends **5.1 mm short of the gate** with 3 of its 6 ink buckets
unflown. That is `ARCHITECTURE_V2` §2d's warning arriving: conducting most of a
picture as one phase does not scale, and a pattern that defers most of the
picture to the conductor has reinvented v19 with a slower planner.

**What the 1 000-stroke model would say, and why it is not quoted.** The v6
recalibration (§21) assumes the main stages carry the ink. Here they carry
4.62 m of 16.08 m and the conductor carries the rest, so the model's terms —
draw s/m, inter-piece leg, park per (stage, arm) — are being fitted to three
arms' worth of ink and then asked about six. A projection built on that would
be arithmetic about a pattern nobody would ship. The honest number to carry
forward is the one this run measured directly: **stage C's 3 075 s for 11.5 m**,
which at 1 000 strokes is not a schedule, it is a refusal.

### What is still running

`s000` and `s150` were still in stage C when this box closed — in its *planning*,
not its conduct, which is `sequence.cost_matrix`'s O(n²) route screen on
~50-piece buckets and is the same wall §19 flagged. Their stage A and B numbers
are the table two sections up and will not move; what they still owe is their
stage C cost and their makespan. Re-run `scripts/lf_report.py
out/staged_csail_h097_lf_*.json` when the JSONs appear.

## 23. The room was the wrong shape — exact swept capsules, and the follower's tuck

§22 left stage A with the followers keeping **0 %** of their ink at every split,
and the diagnosis (`scripts/diag_lf_follower.py`, commit `2cd3eb0`) named the
biggest correctable term: the room a follower is certified against was not the
leader's trajectory, it was `cluster_capsules`' **sphere reduction** of it — one
sphere per 0.075 m cell, radius = 65 mm half-diagonal + up to 177 mm (the
biggest capsule in the cell) + 24 mm sweep pad, median 186 mm and max 417 mm.
Measured cost of that reduction: a **median 195 mm** of clearance the leader's
metal was not occupying.

### 23.1 What was built

**`aris_sixarm/exact_room.py`** — a leader's realised trajectory as its **real
swept capsule chain**: one capsule per link per timeline sample, each carrying
`scene_check.check_timeline`'s own per-sample 1-Lipschitz residual (`SWEEP_FRAC`
× the larger of the step into and the step out of that sample) rather than the
sphere form's single global maximum. Nothing is decimated at the shipped stride.

The spheres are kept — as the **broad phase**. Each grid cell holds its members,
its bounding sphere *and* its AABB, and a query prunes a cell whole when both
bounds already exceed the best gap found so far. The narrow phase then opens
**shells** of increasing bound and stops a row the moment its best gap falls
below the shell boundary already measured — every unmeasured cell lies beyond
that boundary, so the termination is exact, not heuristic. The answer is
bit-identical to a brute-force minimum over every capsule at every cell size
(`tests/test_exact_room.py`).

| | cost per pose, stage-sized room (26 000 capsules, 87 cells) |
|---|---|
| floored at the router's `FRAME_FLOOR` (how every `paper.route` gate asks) | **0.05 ms** |
| floored at the ink gate's cap (`staged.INK_CAP`, 0.30 m) | **0.18 ms** |
| unfloored, exact to infinity | 4.6 ms |

The budget was ≤ 1 ms per pose for the planner's gates; the gates are floored,
and a test pins the floored number under 1 ms. `ink_vs_envelope` is capped at
0.30 m for the same reason `coordination.BROAD_CAP` caps everything else — the
gate it feeds is 50 mm.

**One seam, not five.** Every consumer named in the build item already reaches
the room through `frozen.chain_clearance` / `frozen.partner_clearance`:
`plan_stroke`'s ink gate, `writing.arm_program`, `paper.route`'s straight,
ladder and RRT tiers, and `paper.effective_static_floor`. So the room was
swapped **inside `frozen`** and all of them saw it at once. The one tier that
could *not* see it is the go-around: `paper._skirt` reads box **footprints** to
decide which way to walk, and a room is not a box — so `frozen.room_boxes()`
hands it the room's coarse cells **for detour generation only**. Every candidate
it proposes is still gated against the exact room, so a bad pseudo-box costs a
refused detour and never a certificate. `scene_check` shares none of this code
and stays the judge.

`ARIS_ROOM=spheres|capsules` (default `capsules`) is the whole A/B.

### 23.2 The A/B, same tree, same seam bar, CSAIL h = 0.970, split +0.15, stage A

| | sphere room (`ARIS_ROOM=spheres`) | **exact capsules** |
|---|---|---|
| leader 71's room | 891 spheres | **19 190 capsules** |
| follower 31's **start pose** vs the room | **−128.4 mm** | **+53.7 mm** |
| follower ink clearance, min | −243.8 mm | **−199.8 mm** |
| …median | −134.6 mm | **−118.9 mm** |
| …max | −128.4 mm | **−38.0 mm** |
| pieces refused at the ink gate | 3 | **2** |
| leader ink flown | 2.0613 m | 2.0613 m |
| **follower ink flown** | **0.000 m (0 %)** | **0.000 m (0 %)** |
| stage A inter-arm (`scene_check`) | +65.6 mm, PASS | +65.6 mm, PASS |

**The start pose is the result that matters.** It moved by **+182 mm**, from 128 mm
*inside* the room to 54 mm outside it. That was the mechanism §22.3 blamed:
`paper.effective_static_floor` clamps every leg's static floor to what its own
endpoints can hold, so a park inside the room clamped the floor **negative**
before routing began and the entry legs failed first. That specific failure is
gone.

**And the follower still keeps nothing.** The exact room moved the ink clearance
by +44 mm (min) to +90 mm (max) and it is still deeply negative. This is not a
modelling artefact any more — it is the allocation: **the follower's ink lies
inside the volume the leader's arm actually sweeps.**

### 23.3 Why 55.9 % of poses clear does not buy 55.9 % of the ink

The diagnosis measured 55.9 % of arm 31's ink samples clearing 50 mm against
the exact capsules, and read that as the ceiling. It is not, because
**`ink_vs_envelope` returns the MINIMUM over a piece's poses** and the gate
refuses the piece whole. A piece with 55.9 % of its poses clear still has a
minimum of −185 mm, so it is refused entire. Turning a pose-wise fraction into
flown ink needs the pieces **split at the room boundary** — a DP that cuts a
stroke where it enters a leader's trajectory, which the refusal loop does for
*coverage* but not for *rooms*. That is the next build item, and it is now the
only one between here and a follower that draws.

### 23.4 The tuck, and why it did not fire

`staged.tuck_pose` / `clear_out` / `splice_timeline` implement the
pre-position: a follower searches hover stations over **its own ink** at a
ladder of heights for one that clears the leaders' exact rooms by the routing
floor + 10 mm, flies park → tuck **while the leaders are still parked** (so the
clear-out is routed against the parked fleet, like any other leg), and plans its
tour from there. The clear-out is **spliced into the follower's timeline**, not
left as a prologue — so `trajectory_room`, `active_pair_gap` and `scene_check`
all see it, and the next arm in the sweep avoids it. `q_tuck` and `clear_out_s`
are recorded per arm in the programme (schema stays 2; `q_hold` is unchanged).

On this picture **it did not fire, and it should not have**:

- arm 31 (split +0.15): park **+53.7 mm**, 0 of 45 stations cleared the 73 mm
  bar → stays put. The park is *already* outside the room once the room is
  exact; every hover over its own ink is *inside* it, which is the same fact
  §23.2 reports — the ink is in the leader's swept volume, so a station above
  the ink is too.
- arm 97 (whole bag): park **+263.4 mm** — already clear, early return, no move.

So the clear-out cost **0 s** and the time to first motion is unchanged at
**2.269 s** (split +0.15), well inside the 10 s bar. The machinery is built,
tested and inert on this picture; it earns its keep the moment a park sits
inside a room again, which the seam-bar park re-search happened to fix for
arm 31 at the same time.

### 23.4b Stage B, split +0.15: the followers fly 62.6 % of their ink

The split +0.15 run finished stages A and B after the box closed, and stage B
is the first result in this pattern where **a follower keeps anything**:

| stage B (roles swapped: leaders 17/31/97, followers 2/13/71) | value |
|---|---|
| follower ink offered | 2.5249 m |
| **follower ink flown** | **1.5812 m — `follower_fit_frac` 0.626** |
| follower ink clearance vs the leaders' exact rooms | min **+35.7**, median **+179.6**, max +1591.9 mm |
| leader ink flown | 0.0284 m of 0.0284 m |
| inter-arm, realised trajectories (`active_pair_gap`) | **+98.6 mm** (binding pair 71↔97, leg vs leg) |
| static set, per arm (`scene_check` via `solo_check`) | **+73.7 mm** |
| stage duration / busiest arm | 24.0 s |
| stage verdict | **FAIL** |

**The FAIL is not a collision.** The realised trajectories separate by +98.6 mm
and every arm clears the steel by +73.7 mm — both comfortably over their gates.
The stage is marked not-ok by `ink_vs_envelope_mm = 35.67`: one accepted piece's
ink reads +35.7 mm against a room, under the 50 mm `PAIR_MARGIN`. That is the
planner's own per-piece reading of a stage the independent measurements pass,
and it is the same per-piece minimum §23.3 identifies as the thing to split on.

Totals for the split +0.15 run, stages A + B: makespan **119.885 s**, time to
first motion **0.216 s**, coverage 95.63 %, leader ink 2.0898 m flown,
**follower ink 1.5812 m of 4.6688 m offered (33.9 %)**, deferred 8.0850 m,
`holds_ok` true.

### 23.5 The whole bag, stages A + B (the full A/B/C did not fit the box)

Run with `--stages 0,1`, exact rooms, seam bar in:

| | value |
|---|---|
| makespan, stages A + B | **145.867 s** (A 128.6 s + B 17.3 s) |
| time to first motion, clear-out included | **0.192 s** |
| leader ink flown | **4.6238 m of 4.6238 m offered** |
| follower ink flown | **0.000 m of 5.0258 m offered (0 %)** |
| deferred to stage C | **10.0231 m** of 16.0826 m drawn = **62.3 %** |
| follower ink clearance vs the exact rooms | min −272.4, median −152.6, max −37.8 mm, 13 of 13 under gate |
| `scene_check` | stage A and B both **PASS**, `all_ok` true, `holds_ok` true |
| held-pose barriers | 3, all clear |

Against the §22 baseline (sphere rooms, **pre**-seam-bar, so only partly a
controlled comparison): A + B makespan 146.7 s → 145.9 s, follower ink
clearance min −419.7 → −272.4 mm, median −235.7 → −152.6 mm, max −131.9 →
−37.8 mm. The deferred share is **unchanged at 62 %** because it is the
followers' whole bag either way. The controlled A/B is §23.2, which holds the
tree and the seam bar fixed.

**Stage C was not run.** The build capped a six-arm conduct at 30 minutes of
wall clock and the box closed before stage A + B finished on both splits; the
split +0.15 run was still in stage B's route screen — the same
`sequence.cost_matrix` O(n²) wall §19 flagged — when this was written. Its
stage A numbers are §23.2 and will not move.

### 23.6 What did not change, and the honest bottom line

Stage A's inter-arm clearance, its makespan (95.9 s for the busiest arm at split
+0.15, 128.6 s whole-bag) and its verdict are unchanged: **PASS**, `all_ok`
true, 3 of 3 leader buckets flown. Coverage 97.46 % (split +0.15) / 95.64 %
(whole bag). The exact room is strictly better geometry at 1/25th of the pruning
slack and no measurable planning cost, and it removes one of the two reasons the
follower kept nothing. The other reason is the allocation, and no room model
fixes it.

**The 1 000-stroke model is NOT recalibrated in this pass**, because the
condition the build set for it — "only if stages A/B now carry most of the ink"
— is not met. Stages A and B carry the leaders' ink alone; the followers carry
none, so the deferred share is unchanged in kind. Recalibrating a throughput
model on a stage that still defers its followers' whole bag would be quoting the
conductor's cost as if it were the pattern's.
