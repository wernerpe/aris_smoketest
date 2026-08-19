# The 148 s of pauses — where it comes from, and what would buy concurrency

Reconstructed from the shipped programme by `scripts/concurrency_diag.py` (the allocation is deterministic; every per-arm pause it produces reproduces `out/csail_schedule_full.json` to the float, which is asserted, not hoped). Figure `out/concurrency_diagnostic.png`, numbers `out/concurrency_diagnostic.json`. Nothing is re-planned anywhere below — every counterfactual is the same frozen paths read at a different threshold.

**148.0 s of pause is not 148.0 s of lost time.** Phase 1 runs 63.5 s against a floor of 41.0 s (its longest arm's nominal); phase 2 runs 88.0 s against a floor of 88.0 s. **The entire concurrency loss of the piece is 22.5 s, all of it in phase 1, all of it arm 31 waiting on arms 71 (19.1 s) and 97 (3.4 s).** The other 125.5 s is arms standing still inside slack they already had — real seconds of a stopped robot, and worth removing, but they cost the clock nothing. Phase 2 is already optimal: arm 97 draws its 4.52 m in 88.0 s and never waits once.

**Three quarters of the pause is arms held out of their own finished pose.** 116.6 s of the 148.0 (66.4 + 50.2) is the rest-suffix rule — an arm may not *arrive* until the pose it will then stand in is clear all the way to the horizon, so a short programme finishes its ink early and then waits for permission to stop. Adding that rule is what took phase 1 from 4 s of pauses to 90.6 s, and it is a cost of standing still in the wrong place, not of drawing. Only 31.4 s is an arm actually held up on its path.

**Two arms never fight over ink.** Of the 31.4 s spent held on a path: draw-vs-transit 19.4 s, transit-vs-transit 12.0 s, **draw-vs-draw 0.0 s**. The conflict is the hover moves between strokes, not the strokes. And the two floor arms parked at `q_seed` for both passes (13, 17) block **0.00 s** — deleting them from the world entirely changes no number in this document. Every second of contention is among the four inverted arms.

**The whole of phase 1 is six events, and two of them are refused by a fifth of a millimetre.** Every pause in the phase; "min gap" is the tightest swept clearance during the event:

| arm | from | for | blocked by | doing | blocker doing | min gap |
|---|---|---|---|---|---|---|
| 2 | 8.6 s | **44.0 s** | 31 | parking | transit | **79.8 mm** |
| 97 | 10.2 s | **22.5 s** | 71 | parking | transit | **79.3 mm** |
| 31 | 21.4 s | 10.1 s | 71 | transit | seg 3 | 59.0 mm |
| 31 | 9.0 s | 9.0 s | 71 | transit | seg 1 | 60.3 mm |
| 31 | 18.0 s | 3.4 s | 97 | transit | transit | 76.2 mm |
| 97 | 2.6 s | 1.7 s | 71 | transit | transit | 65.9 mm |

The top two are 66.4 s of a robot standing still because its *home* pose is 79.8 mm and 79.3 mm from another arm's hover corridor and the margin wants 80. Phase 2 repeats the pattern once: arm 2 waits 50.2 s to park, at 69.7 mm, against arm 97 transiting. Meanwhile the three events that actually cost clock (arm 31, 22.5 s) bottom out at 59.0–76.2 mm. Relaxing the margin to 65 mm takes phase 1 from 63.5 s to 47.7 s; below that the curve flattens — 47.3 s at 50 mm, 47.1 s at 35 mm — so what is left is no longer a margin problem.

**Nothing interpenetrates. Not once.** Across all 148.0 s of blocking cells the tightest *pose* clearance is 74.4 mm (phase 1) and 77.1 mm (phase 2), and it is never negative. 147.5 s of the 148.0 is blocked at cells whose four corner poses are all ≥ 80 mm apart, and it is only the swept-motion slack (0.55 × the two per-step motions — about 10 mm when one arm is transiting at 0.8 m/s on a 1/48 s clock) that drags the test under the margin. Weighted by the second, the histogram sits between 65 and 80 mm: 0.41 s of the 148.0 is below 65 mm and **0.00 s of it is below 50 mm**. The margin is the obstacle here, and the geometry is not.

| margin | ph1 pause | ph2 pause | total pause | ph1 makespan | total makespan | achieved clearance |
|---|---|---|---|---|---|---|
| **80 mm** (shipped: 50 safety + 30 calib) | 90.6 s | 57.4 s | **148.0 s** | 63.5 s | **151.5 s** | 80.0 mm |
| 65 mm | 70.4 s | 57.2 s | 127.6 s | 47.7 s | 135.7 s | 65.0 mm |
| 50 mm (calib surveyed away) | 12.6 s | 54.1 s | 66.6 s | 47.3 s | 135.4 s | 50.1 mm |
| 35 mm | 11.9 s | 52.6 s | 64.4 s | 47.1 s | 135.1 s | 35.2 mm |

**Priority order is worth as much as the margin, and it is free.** Phase 1 has only **3 of 24** orders feasible at all, and "busiest first, promote whoever deadlocks" landed on the *worst* of the three: `[31, 71, 97, 2]` finishes in **48.3 s** against the shipped 63.5 s, a 15.2 s win for a re-ordering that costs four DPs to evaluate. Phase 2 has 12/24 feasible and the shipped order is already optimal. The conductor stops at the first order that works; it should keep going and rank them.

**A 25 % crawl is worth 10.8 s, and cannot be worth much more.** Integrating the optimistic bound (a stop of L steps becomes a delay of 0.75 L) over every event gives 148.0 → 124.4 s of pause and 151.5 → 140.7 s of makespan. A rest hold is credited nothing on purpose: the arm is not being held up along its path, it is being held out of the pose it wants to finish in, and creeping the last centimetre more slowly still puts it there. Separately, the *geometric* half of the same idea — a creeping link sweeps a thinner tube, so re-conducting with each moving arm's own sweep slack cut to a quarter — recovers phase 1 to 48.3 s on its own, which is the same 15.2 s the re-ordering finds, and for the same reason.

**Tuck poses buy the most pause and the least clock.** Letting every arm retreat out of the way the moment it finishes (and dropping the rest-suffix requirement with it) takes the pause from 148.0 to **51.1 s (−65 %)** but the makespan only from 151.5 to 136.4 s. Removing the two parked floor arms buys exactly nothing (148.0 → 148.0).

**The levers are substitutes, not complements.** Best order alone → 136.4 s; a 50 mm margin alone → 135.4 s; both together → 135.4 s. Every one of them is attacking the same conflict (arm 31 against arm 71 in phase 1), and once it is cleared the phase floors at ≈ 47 s against its 41.0 s ideal. That residual ≈ 6 s survives a 35 mm margin, a tuck, and the best order simultaneously, and is the only part of the piece where the paths genuinely need to be somewhere else.

## Ranked, for what to build next

1. **Congestion-aware sequencing — build this.** −15.2 s of makespan (phase 1: 63.5 → 48.3 s) for a search the conductor is two lines from doing already: enumerate the priority orders instead of accepting the first feasible one, and rank on makespan rather than on "busiest first". It spends no safety, needs no hardware, re-uses the collision images it has already paid for, and 24 orders cost about a second.
2. **Margin reduction — do the survey, but not for the speed.** −16.1 s, marginally the largest single number, but it is a *substitute* for (1): stacked on top of the best order it adds 1.0 s. The 30 mm `calib` term is an admitted guess about where four preset bases are; surveying them is right regardless, and it would also make the 3-of-24 feasibility in phase 1 much less brittle.
3. **Velocity scheduling.** −10.8 s as a re-timing bound, and it is optimistic. Its real appeal is the geometric half — 99.7 % of the blocking is swept-tube slack over poses that are already 80 mm clear — but cashing that in means re-opening the per-segment velocity certificate `stroke_api` issues, which is the most invasive change on this list for a win (1) gets for free.
4. **Tuck poses.** Only −15.1 s of makespan, but −65 % of the pause and the right fix for a different problem: an arm's *parked* pose can make a schedule infeasible outright rather than slow, which is exactly how the 1.2656 m logo width failed (arm 71's 300 mm pen sweeping through arm 97 at its ready pose). Build it as robustness, and to stop the fleet standing frozen for two minutes of a two-and-a-half-minute run.
5. **Redundancy-space dodging — not yet.** Its entire headroom is the ≈ 6 s of phase 1 that survives every other lever. It is the most machinery on the list for the smallest measured win, and it should wait until (1) and (2) have been taken and re-measured.

**Caveats.** All of this is conductor v1's move set — advance or wait on a frozen path — so "makespan" means the kinematic playback, not an acceleration-limited run. The counterfactual margins are re-thresholds of the same collision images, checked only by the conductor's own test; a real margin change must go back through `scene_check`. Blame for a wait with no blocker at that instant is assigned to the first cell the arm would run into at full speed from there, which is a defensible reading of "what is in the way" and not the only one; 0.0 s of the 148.0 ended up unattributed.
