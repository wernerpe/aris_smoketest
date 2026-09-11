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

