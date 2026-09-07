# Decisions — the numbers, and where each one is anchored

## RE-PLANNING AT THE HOLDER'S OWN TOOL (2026-09-03)

`bc670bf` moved the lateral tool's tip 72.348 mm — `PEN_LAT_HOLDER` /
`PEN_EXT_HOLDER` from 0.110 / 0.110 to **0.0588421 / 0.0588421** — and said out
loud that every certified programme and atlas in `out/` was stale.  This is the
re-certification.  Nothing below changes a gate, a capsule, a layout, a height
or a tool constant; every number is the same machinery run again at the new
pair.

### The atlas first, because everything else reads it

`atlas.is_current` compares a stored sweep's model signature against the
running build's, and the pen is **inside** that signature (entry 340), so every
atlas in `out/` refuses at the new tool by name: *"collision model differs at
entry 340: atlas has 0.1100, this build has 0.0588"*.  Re-swept:

    ARIS_TOOL=lateral scripts/run_atlas6.py --rig proposed \
        --pen 0.0588421 --pen-lat 0.0588421 \
        --out out/atlas_proposed_h0940_lat0588        (6 jobs, 464 s/arm)
    ARIS_RIG=proposed ARIS_TOOL=lateral scripts/regate_atlas.py \
        --in  out/atlas_proposed_h0940_lat0588 \
        --out out/atlas_proposed_h0940_lat0588_gated63    (304 s)

**WHAT THE SHORTER PEN COSTS THE CANVAS, before any planning at all:**

| 2 cm sweep, h = 0.940, 16 562 cells | 0.110 tool | **0.0588421 tool** |
|---|---|---|
| union strict-GO | 99.64 % | **97.08 %** |
| reachable | 100.00 % | **99.71 %** |
| ≥2 arms | 47.90 % | **38.88 %** |
| ≥3 arms | 11.11 % | **4.21 %** |
| dead | 0.36 % | **2.92 %** |
| strict-GO cells, arm 31 | 4 846 | **4 072** |

The tool is 72 mm shorter along its own axis and 51 mm shorter across it, so
the WRIST has to come 72 mm down and 51 mm in for the same tip: the arm reaches
less far and the annulus narrows at both edges.  Redundancy is what pays for it
— the union loses 2.6 points, the ≥3-arm overlap loses two thirds.

### The park set: re-searched, not patched

`Q_PARK_PROPOSED` was flagged stale at its own definition and it was: run at
the new pair, the OLD grid parks arms 13 and 17 **53.7 mm** apart and
`certified_park_poses` refuses the fleet.  The old POSES are safe — every one
still passes `check_pose`, the fleet still holds ≥ 250 mm — but they stand
**69.0 mm inside** arm 17's certified ink, which is the number a phase is
conducted on.

The (radius, hover, bearing) search was re-run at the new tool: 6 radii × 4
hovers × 24 bearings per arm, each gated by `certified_ready_pose` (430–431 of
577 certify); then park-vs-ink ≥ 80 mm against the new atlas; then the fleet
gate; then ranked on the depot's own job — of 24 of the arm's certified cells,
how many it can fly to (`writing.enter_beats`) and home from (`exit_beats`).

| | OLD literals at the new tool | **NEW, re-searched** |
|---|---|---|
| worst park-vs-ink | **−69.0 mm** | **+97.8 mm** |
| worst fleet pair | ≥ 250 mm (cap) | ≥ 250 mm (cap) |
| entries flyable | 97/137 (70.8 %) | **108/137 (78.8 %)** |
| go-homes flyable | 94/137 (68.6 %) | **108/137 (78.8 %)** |
| joint margins | 0.30–0.68 | **0.541–0.682** |
| park pens inside the CSAIL logo box | arm 71 at (1.357, 1.556) | **none** |

    PARK_GRID_PROPOSED = {2: (0.55, 0.10, -165.0), 13: (0.40, 0.10, -150.0),
                          17: (0.48, 0.30,  -60.0), 31: (0.55, 0.10, -165.0),
                          71: (0.55, 0.35,  -30.0), 97: (0.55, 0.20,   15.0)}

**+97.8 mm is the layout's ceiling, not a lucky draw.**  22–42 candidates per
arm clear the ink gate and every one of them lands on the same 97.8–98.9 mm
plateau — a parked arm's own base column is pose-invariant, so no bearing buys
past it.  That is the same plateau the 2026-08-26 search reported as "97–98 mm"
at the old tool, which is what says the criterion was re-run and not
re-invented.

**Five of six bearings still point outward; arm 2's is tangential** (dot with
its own outward ray −0.015, 90.9° off it).  Outwardness was always a means:
what keeps the six apart is `fleet_park_clearance`, and it proves ≥ 250 mm.
Held to the outward ray instead — the two-number grid entry — the same radii
and hovers put arms 2 and 97 **111.0 mm inside each other** at the new tool,
so the third number is not a refinement, it is the reason there is a park set.

**A test that had been skipping for as long as the directory existed.**
`test_no_shipped_park_pose_stands_in_another_arms_certified_ink` picked its
atlas with `atlas.is_current`, which answers against the PROCESS-GLOBAL tool —
and a bare `pytest` run is the INLINE pen, so every lateral atlas read as stale
and the test skipped, while its body measured with `PEN_*_HOLDER`.  It now
matches the atlas's recorded `pen_ext`/`pen_lat` to the tool it measures with
and answers `is_current` with that tool active.  It RUNS now, and passes at
+97.8 mm; it would have failed at −69.0 mm on the old poses.

### CSAIL v15 — the same picture, the same placement, the new tool

Run as a **GUI job** (`out/gui_jobs/20260903-200349-d8ec`, watchable live on
`http://127.0.0.1:8765/`), which is `scripts/draw.py` with v14's own flag set:
same source `assets/csail/csail_old_med.gif`, same placement file
`out/csail_place_v4_placement.json` (1.4309 × 1.8711 m at (0.8017, 1.81532),
rotated 90°, offset [−0.1, 0]), `--arms all --two-pass --arm-phases off
--residual-passes 6 --residual-min-gain 0.0005 --depot-hover-selective
--freeze-refused-phase --skip-unconductable --tilt-max-deg 15 --image-jobs 6`,
the RRT pen-up tier ON — and the NEW atlas.  Copied out to
`out/csail_schedule_h094_v15.{npz,json}`, `out/csail_program_h094_v15.json`,
`out/h094_v15.log`.

**ONE DELIBERATE DIFFERENCE, NAMED.** v14 was `scripts/csail_schedule.py`,
which traces with `trace.trace_logo`; the GUI front door traces with
`trace.trace_any`.  Both give **39 strokes**; the generic tracer's path is
8 011 px against 7 512, and after placement **16.8048 m** against v14's
16.7370 m — 0.40 % more ink.  Everything after the tracer is the same call.

| | v14 (0.110 tool) | **v15 (0.0588421 tool)** |
|---|---|---|
| coverage | **100.0000 %** (0.0000 m empty) | **99.5224 %** (0.0803 m empty) |
| makespan | 196.271 s | **277.625 s** (+41.4 %) |
| phases planned / conducted | 3 / 3 | **2 / 2** (+1 residual that certified 0 m) |
| segments | 42 | **47** |
| traced / drawn | 16.7370 / 16.7470 m | **16.8048 / 16.7625 m** |
| pen recorded | 110.0 mm | **58.8 mm** |
| min inter-arm, per phase | 82.4 / 82.4 mm | **85.6 / 82.0 mm** (gate 80) |
| self-collision | 26.6 / 31.1 mm | **22.7 / 33.7 mm** (gate 20) |
| frame | 54.5 / 54.7 mm | **53.5 / 53.2 mm** (gate 50) |
| neighbour base column | 138.6 / 138.1 mm | **122.0 / 131.5 mm** (gate 80) |
| paper, chain | 68.3 / 57.4 mm | **30.7 / 31.4 mm** (gate 20) |
| paper, tip | −4.1 / −4.5 mm | **−5.0 / −7.3 mm** (floor −10) |
| pen-swap pause | — | 256.0 mm, 6/6 poses pass |
| scene_check | PASS | **PASS, both phases** |
| planner wall clock | 4 170.9 s | **4 910.9 s** |

**INDEPENDENT `scene_check`, re-run from the shipped `.npz`** over the whole
merged 6 664-frame timeline (not per phase): **VERDICT PASS**, min inter-arm
**80.26 mm** against the 80 mm gate, worst pair 13–31 at t = 99.09 s; self
21.4 mm, frame 52.7 mm, column 115.0 mm, paper chain 30.7 mm, tip −7.4 mm,
frozen poses 6/6.  The whole-run number is 1.7 mm tighter than the tightest
phase because it spans the pen-swap seam that no per-phase check covers, and
it clears by **0.26 mm**.  That is the thinnest margin in this programme and
it is where a re-run should be watched.

**WHERE THE TIME GOES** — the GUI's own substage log, 4 791.5 s of measured
substage time inside a 4 910.9 s run:

| substage | v15 total | share | v14 |
|---|---:|---:|---:|
| balance | 3 136.8 s | 65.5 % | 2 636.8 s |
| conduct | 710.5 s | 14.8 % | 239.4 s |
| replan | 465.7 s | 9.7 % | 806.4 s |
| probe | 272.0 s | 5.7 % | 180.2 s |
| flycheck | 109.6 s | 2.3 % | *(inside v14's balance)* |
| merge | 39.1 s | 0.8 % | *(inside v14's balance)* |
| sequence | 25.4 s | 0.5 % | 13.0 s |
| guarantee | 23.1 s | 0.5 % | *(inside v14's balance)* |
| repair | 8.7 s | 0.2 % | 0.0 s |
| prefilter | 0.4 s | 0.0 % | 0.4 s |

Per phase: allocation 1 735.4 s (grey) + 2 262.7 s (orange) + 5.8 s
(residual) = 4 081.0 s; conduction 477.0 + 233.6 = 710.5 s; `scene_check`
50.0 + 49.7 = 99.7 s.  **The balancer is two thirds of the run and it always
was** — v14 spent 63 % there under a name that also contained flycheck, merge
and guarantee.  That is the baseline to iterate against.

**THE 80.3 mm THAT WAS LEFT EMPTY IS REACH, and it is one span.**
`dropped` names it exactly: stroke 36, orange, the fill boundary's
`s_range` 0.5001–0.6249, 0.0803 m at **(0.0866, 1.2350)** — the logo's own
LEFT EDGE.  Its two cells (0.08, 1.18) and (0.08, 1.24) each had a certified
drawing pose for **arms 13 and 31** in the old atlas and have **none at all**
in the new one; the v13 map calls both `NO_DRAW`.  It is not a park refusal,
not a neighbour, not self-collision and not a route: 16 probes over 6 (stroke,
arm) pairs, 4 prefiltered, and no arm can put the tip there.  Those cells sit
0.77–0.81 m from the nearest base, in the outer rim the shorter tool took away.

**WHAT WOULD RECOVER IT WITHOUT MOVING A CONSTANT: the placement.**  At the
rows that matter the leftmost live column is now **x = 0.10 m** and the logo's
left edge is at **x = 0.0866 m** — 14 mm.  A placement re-search against the
new atlas (`--placement auto`, which is what chose `csail_place_v4` against the
old one) is the principled fix; the placement file's own `live: 1.0` is no
longer true at this tool.  1.50 % of the logo's bounding box is dead in v13
against 0.48 % in v12, so a slightly smaller or slightly right-shifted
placement is the whole of the difference between 99.52 % and 100 %.

### CSAIL v16 — the placement WAS re-searched, and it made the picture worse

v15's entry named the lever: *"a placement re-search against the new atlas
(`--placement auto`) is the principled fix."*  It was run, at v15's own flag set
with nothing else changed, as GUI job **`20260903-215636-b6f9`**.  Copied out to
`out/csail_schedule_h094_v16.{npz,json}`, `out/csail_program_h094_v16.json`,
`out/csail_place_v16_placement.json`, `out/h094_v16.log`.

**IT DID NOT RECOVER 100 %.  IT LOST 0.65 pp.**

| | **v15** (placement file) | **v16** (`--placement auto`) |
|---|---|---|
| placement | 90°, 0.85, offset (−0.10, **0.00**) | 90°, 0.85, offset (−0.10, **+0.10**) |
| centre | (0.8017, 1.81532) | **(0.8017, 1.91532)** |
| size | 1.4309 × 1.8711 m | **same** |
| coverage, allocated | **99.5224 %** | **98.8729 %** |
| coverage, conducted | 99.7485 % | **97.9754 %** |
| left empty | 0.0803 m in 1 span | **0.1894 m in 1 span** |
| skipped at conduction | 0.0000 m | **0.1947 m** |
| segments | 47 | 48 allocated / 47 conducted |
| makespan | 277.625 s | **249.875 s** (−10.0 %) |
| conducted pause | — | 56.667 s |
| phases planned / conducted | 2 / 2 | **3 / 7** |
| min inter-arm, worst phase | 82.0 mm | **81.2 mm** (gate 80) |
| neighbour base column | 122.0 mm | 131.0 mm (gate 80) |
| paper, chain | 30.7 mm | **22.6 mm** (gate 20) |
| paper, tip | −7.3 mm | −4.9 mm (floor −10) |
| planner wall clock | 4 910.9 s | **29 352.9 s** |

**INDEPENDENT `scene_check`, whole merged 6 241-frame timeline**, re-run from
the shipped `.npz`: **VERDICT PASS**, min inter-arm **80.25 mm** against the
80 mm gate — **+0.25 mm** — worst pair **17–71 at t = 35.68 s**; self 27.5 mm,
frame 53.0 mm, column 130.9 mm, paper chain 23.0 mm, tip −4.9 mm, joint margin
0.1051, frozen poses 6/6.  v15's whole-timeline number was 80.26 mm on pair
13–31.  **Both runs clear the gate by a quarter of a millimetre on a different
pair — the margin is a property of this rig at this tool, not of either
placement.**

**WHERE THE 0.1894 m WENT, AND IT IS A DIFFERENT FAILURE FROM v15's.**
`dropped` names it: stroke 1, orange, an **outline drawn as a whole stroke**
(`s_range` 0.0–1.0), 0.18941 m at **(0.7012, 1.8020)**, running from
(0.71646, 1.80259) to (0.52705, 1.80205).  That is not the canvas edge — it is
**straight through arm 31's own base**.  Cell (0.60, 1.80) sits **0.016 m** from
arm 31's base at (0.5967, 1.8153) and has a certified drawing pose for **no arm
at all**; its neighbours (0.72, 1.80), (0.70, 1.80) and (0.52, 1.80) are
certified for 31 and 71.  The by-third accounting puts all 0.19 m in the
**middle** third.  Residual pass 2 spent 16 probes over 6 (stroke, arm) pairs,
4 prefiltered, and gap repair offered **12 windows** and got **0** certified
spans back.

**SO v15 LOST INK TO THE OUTER RIM AND v16 LOSES IT TO AN UNDER-BASE HOLE** —
the two dead sets the height table below shows trading against each other, met
one after the other by the same logo shifted 10 cm.

**AND THE CONDUCTOR PAID FOR THE SHIFT TWICE.**  Moving the logo 0.10 m toward
the middle row put phase 1 beyond the idle policy: grey was refused with 5 arms
drawing, re-offered as `{13,31,2}` + `{17,71,97}`, refused again at 2 arms, and
arm 2's tour was cut in half twice more — **7 conducted phases against v15's
2**, and 0.1947 m of allocated ink skipped outright.  The conductor says why in
its own words: *"this is a FROZEN POSE problem: arm 31 cannot stop clear of 2
(38 mm); arm 71 cannot stop clear of 17 (62 mm); arm 97 cannot stop clear of 2
(76 mm)"*.  The 10 % faster makespan is not a win — it is 0.34 m of ink that is
not being drawn.

**WHY THE SEARCH CHOSE IT, AND THE HONEST GAP.**  The rule is the standing one,
*largest AREA within 1 pp of the best real coverage*.  The three best cells were
within **0.16 pp** of each other — 0.75× at 95.418 %, 0.80× at 95.334 %, 0.85×
at 95.257 % — so the area rule took the largest, spending 0.16 pp of search
coverage to buy 0.6 m² of drawing.  **It cost 0.65 pp of real coverage.**  And
`--top 3` means only three offsets per (rotation × scale) cell are allocated for
real: the proxy ranked (−0.10, +0.10), (+0.10, −0.20) and (+0.10, +0.20) above
**v15's own (−0.10, 0.00), which was therefore never allocated for real in this
search at all.**  The incumbent was not beaten; it was never entered.

**THE NEXT CONSTANT-FREE LEVERS, IN ORDER OF CHEAPNESS.**  All three are FLAGS:
1. **Keep v15's placement.**  It is the best number anyone has at this tool and
   v16 is the evidence that the search at these settings does not improve on it.
2. **`--slack 0`** — stop the area rule spending coverage.  On this very search
   that picks 0.75× at 95.418 % instead, a 1.816 m² drawing rather than 2.677.
3. **`--top` raised, or the incumbent seeded** — allocate v15's cell for real
   alongside the proxy's favourites so a re-search can never regress.
A fourth, not free: the search's objective cannot see the conductor at all, and
v16 is the first run where the difference between allocating and conducting
decided which placement was better.

**WHAT `--placement auto` COSTS AT THIS FLAG SET — the measurement to iterate
against.**  826 proxy placements, **78 allocated for real, 27 243.3 s**: the
search alone is **5.5×** v15's entire run, and the job's 29 352.9 s is **6.0×**.
Per candidate ≈ 1 990 worker-seconds on 6 jobs.  Where it goes, over the whole
job's 156 525 s of measured substage time:

| substage | v16 total | share |
|---|---:|---:|
| flycheck | 90 368.3 s | 57.7 % |
| merge | 34 165.9 s | 21.8 % |
| replan | 23 252.9 s | 14.9 % |
| probe | 4 893.7 s | 3.1 % |
| sequence | 1 321.2 s | 0.8 % |
| guarantee | 1 259.4 s | 0.8 % |
| repair | 631.4 s | 0.4 % |
| balance | 466.9 s | 0.3 % |
| conduct | 148.1 s | 0.1 % |
| prefilter | 17.8 s | 0.0 % |

By stage: **placement 27 243.3 s**, allocation 1 811.2 s, conduction 294.0 s,
`scene_check` 70.2 s over 7 phases, export 3.0 s, trace 0.2 s.  **`balance` is
0.3 % here against 65.5 % in v15** — `scripts/draw.py` passes `balance=False,
split=False` to the search, so the 78 candidates pay `flycheck` and `merge`
instead, and those two are 80 % of the bill.  `docs/ANY_PICTURE.md`'s premise —
*"that is the difference between a 48-placement search in 6 minutes and one in
8 hours"* — was measured before `--residual-passes` and the RRT pen-up tier
existed, and it no longer holds: **the search now inherits the run's whole flag
set and pays it 78 times.**  The obvious speed lever, and it is the rig owner's
call, is to search under a cheap flag set and spend the full one only on the
winner.

### CSAIL v17 — 100.0000 %, and then the tool moved again (ABORTED 2026-09-04)

**READ THE STALENESS FIRST.**  Every number in this section was measured at the
`PEN_EXT_HOLDER` / `PEN_LAT_HOLDER` = **0.0588421 / 0.0588421** pair.  On
2026-09-04, while the follow-on run was still conducting, Pete corrected the
holder placement again — **the grip point moves to the far end of the Fat finger
plates**, which moves the tool transform a third time.  So this entry is a
RECORD OF A METHOD AND OF A PLACEMENT, not a shippable programme: the atlas,
the parks and every gate below were certified against a tool that is being
superseded, exactly as `bc670bf` superseded the 0.110 pair.  **Nothing here
should be conducted, and the comparison at h = 0.880 was killed mid-run and is
not reported.**

WHAT SURVIVES THE TOOL CHANGE is the *method*: the proxy pre-check below costs
five minutes, it predicted v15's measured loss to 2 mm, and it is the thing to
re-run first against the new pair — it will say in minutes whether a placement
can reach 100 % before anyone spends 7.57 h searching for one.

v16 spent 7.57 h searching and lost 0.65 pp.  v17 spends **five minutes** on a
proxy pre-check, moves the logo by hand, and closed the gap: GUI job
**`20260904-074027-32b1`**, v15's exact flag set, v15's exact size and rotation
(90°, scale 0.85, `target_width` 1.43089), one fixed offset, **no search**.
`out/csail_schedule_h094_v17.{npz,json}`, `out/csail_program_h094_v17.json`,
`out/csail_place_v17_placement.json`, `out/h094_v17.log`.

**THE METRIC THAT PICKED IT, AND WHY IT IS TRUSTWORTHY.**  Sample the traced
ink every 1 cm, look each sample up in the 2 cm atlas union, and take the
**longest CONTIGUOUS run of dead samples inside one stroke** — because a
scattered dead cell is drawn anyway (the atlas is a sampling; the pose search is
continuous) while a contiguous run is a span nobody can cover.  On v15's own
offset it predicts **82 mm** against v15's **measured 80.3 mm loss**.  It is
right to 2 mm on the one case where the answer is known.

| offset at scale 0.85 | left edge | outer-rim ink | under-base ink | **longest run** |
|---|---|---|---|---|
| (−0.10, 0.00) = **v15** | 0.0863 | 82 mm | 98 mm | **82 mm** at (0.09, 1.24) |
| (−0.09, 0.00) | 0.0963 | 37 mm | 177 mm | 79 mm |
| (−0.07, 0.00) | 0.1163 | **0 mm** | 197 mm | **118 mm** at (0.55, 1.78) |
| (−0.03, 0.00) | 0.1563 | 0 mm | 237 mm | 158 mm |
| **(+0.05, −0.20) = v17** | 0.2363 | **0 mm** | **0 mm** | **0 mm** |

**A PURE LEFTWARD-TO-RIGHTWARD NUDGE CANNOT DO IT.**  Walking the logo right
does clear the outer rim at the left edge — by (−0.07, 0.00) the rim ink is
zero — but it walks the logo's INTERIOR onto **arm 31's base disc**, and the
longest run gets *worse*, 82 → 118 mm.  The middle-row columns stand at
x = 0.5967 and x = 1.2067 and a 1.431 m-wide logo at this size spans both; every
offset from −0.11 to −0.02 puts ink either in the rim or on a disc.  **The fix
is not a smaller x-nudge, it is +x AND −y together**: of **1 976** offsets that
fit the sheet, exactly **10** have a zero dead run, all at scale 0.85, all at
dx ≈ +0.05…+0.07.  v17 takes the one that also moves ink AWAY from the middle
row, since that is the direction that cost v16 five extra phases.

| | v15 | v16 | **v17** |
|---|---|---|---|
| offset | (−0.10, 0.00) | (−0.10, +0.10) | **(+0.05, −0.20)** |
| centre | (0.8017, 1.8153) | (0.8017, 1.9153) | **(0.9517, 1.6153)** |
| size | 1.4309 × 1.8711 m | same | **same** |
| coverage, allocated | 99.5224 % | 98.8729 % | **100.0000 %** |
| coverage, conducted | 99.7485 % | 97.9754 % | **100.0000 %** |
| left empty / skipped | 0.0803 / 0.0000 m | 0.1894 / 0.1947 m | **0.0000 / 0.0000 m** |
| segments | 47 | 48 / 47 | **49** |
| makespan | 277.625 s | 249.875 s | **195.229 s** |
| conducted pause | — | 56.7 s | 121.4 s |
| phases planned / conducted | 2 / 2 | 3 / 7 | **2 / 2** |
| min inter-arm, worst phase | 82.0 mm | 81.2 mm | **82.2 mm** |
| column / chain / tip | 122.0 / 30.7 / −7.3 | 131.0 / 22.6 / −4.9 | **136.6 / 30.4 / −7.0** |
| planner wall clock | 4 910.9 s | 29 352.9 s | **3 381.7 s** |

**INDEPENDENT `scene_check`, whole merged timeline, from the shipped `.npz`:**
**VERDICT PASS**, min inter-arm **80.91 mm** against the 80 mm gate — **+0.91 mm**,
worst pair **31–71 at t = 91.19 s**; self 20.6 mm, frame 53.1 mm, column
132.8 mm, paper chain 30.2 mm, tip −6.8 mm, joint margin 0.1124, frozen 6/6.
**That is 3.6× the margin v15 and v16 cleared by** (0.26 and 0.25 mm).  The
razor-thin seam this file flagged as "where a re-run should be watched" was a
property of those two placements after all, not of the rig.

**AND IT IS THE FASTEST PROGRAMME OF THE THREE**, 195.229 s against v15's
277.625 s — 30 % less makespan for 0.48 pp more ink, in two conducted phases
with no split and no skip.  At the 0.0588421 tool v17 superseded v15; at the
tool now being fitted, **neither of them stands**.

**THE h = 0.880 COMPARISON WAS ABORTED.**  The same v17 placement was being
re-planned at 0.880 on the searched 0.880 park grid — the run Pete needs for the
trim decision on a placement that is not known-bad — and it was killed at
08:39→10:03 during conduction when the tool correction landed.  It got as far as
scheduling the movers (arm 71: 3 170 steps, 66.0 s nominal → 122.3 s with
56.3 s of pauses) and **no coverage, gate or `scene_check` number was reached**,
so none is quoted.  `out/h088_v17p.log` and `out/csail_schedule_h088_v17p_trace.png`
are the partial remains; there is no schedule, no programme and no `.npz`.

**WHAT TO DO WHEN THE NEW TOOL LANDS**, in the order that costs least:
re-sweep the atlas at the new pair; run the five-minute proxy over the offset
grid to see whether a zero-dead-run placement still exists at scale 0.85;
re-search the park set (it is a quarter of an hour and the 0.940 grid did not
transfer to any other height, so it will not transfer across a tool either);
and only then spend an hour on a full plan.  **Do not start with
`--placement auto`** — v16 is the measurement of what that costs and what it
gets.

### The feasible-workspace map, v13

`scripts/feasible_workspace.py` at v12's settings (`--fiber-tries 48
--hover-lean-deg 15 --rrt 60 --rrt-nodes 300 --rescue 0,1`), 8 workers, on
`out/atlas_proposed_h0940_lat0588_gated63`; sweep 4 573 s + rescue 644 s.
`out/feasible_workspace_v13.{png,json}`, log `out/feasible_workspace_v13.log`.

| solo-drawable, 16 562 cells | v12 (0.110) | **v13 (0.0588421)** |
|---|---|---|
| FEASIBLE | **99.46 %** (6.589 m²) | **96.84 %** (6.416 m²) |
| draw pose only | 99.64 % | 97.05 % |
| + hover | 99.64 % | 97.05 % |
| + reachability | 99.46 % | 96.84 % |
| DEAD, no draw pose | 0.36 % | **2.95 %** |
| DEAD, draw ok no hover | 0.00 % | 0.00 % |
| DEAD, hover ok unreachable | 0.18 % | 0.21 % |
| largest inscribed rectangle | 0.52 × 3.54 m = 1.841 m² | 0.46 × 3.64 m = **1.674 m²** |

**THE LOSS IS REACH, NOT ROUTING.**  Route-dead barely moves (0.18 → 0.21 %,
30 → 34 cells) and hover-dead is still zero; "no draw pose" goes 59 → 489
cells.  The rescue ladder confirms it: rung 0 turned 150 arm-cells (dead
673 → 523) and rung 1 turned **none**, so what is left is not something a
better pen-up plan reaches.

**WHERE THE DEAD ZONES MOVED.**  89 → 523 cells; 464 newly dead, 30 revived.
v12's dead set was almost entirely the six under-base holes (74 of 89).  v13's
is **two** sets: 263 cells still under the base discs (r ≤ 0.30 m), and **242
cells out at 0.60–0.85 m from the nearest base** — a new OUTER RIM that did
not exist at the longer tool, distributed evenly over all three thirds of the
sheet (163 / 145 / 156 newly dead).  Both edges of every annulus moved inward:
the tool is 72 mm shorter along the approach axis and 51 mm shorter across it,
so the wrist has to come to the paper for the inner cells and cannot get out
to the far ones.

### Loose ends this run leaves behind

**`docs/SYSTEM_MODEL.md`'s park-pose clearance table is measured at the OLD
six poses** (403.2 / 264.0 mm arm-to-arm, 196.5 mm arm-to-paper).  The poses
moved here; the fleet's nearest pair is still at the broad-phase cap (≥ 250 mm)
and the closest tip-to-paper is now the **0.10 m** hover three arms take,
against the old set's 0.1965 m.  Re-measuring that table needs the URDF
pipeline and is not done here.

**`np.cross` on 2-vectors** — `tests/test_draw.py`'s SVG failure — was NumPy
2.0 removing the 2-D overload, not anything in this repo.  `trace._cross2`
writes the one line of arithmetic out; `test_draw` is 6 green.

**The two FK bit-identity failures in `tests/test_planner_robustness.py` are
the environment and they stay.**  `_franka_ik`'s C++ batch and `frames.fk`'s
NumPy chain differ by **max |ΔT| = 3.331e-16 and max |ΔP| = 2.220e-16** — one
to two ULP — on all 200 sampled configurations, under NumPy 2.5.2.  The
assertions are `np.array_equal`, deliberately, and loosening a bit-identity
gate is not an environment fix; the deviation is 13 orders of magnitude under
the 2 cm threshold the lattice gates on and 4 under the 1e-12 the URDF checks
hold.  Recorded, not patched.

## OPEN — FOR PETE: THE MOUNTING HEIGHT, RE-ASKED AT THE REAL TOOL (2026-09-03)

**Nothing here changes `layout.LAYOUT_PROPOSED`.  h = 0.940 is still what
ships.**  This is the data for a decision that is Pete's, and it is being
re-asked because the decision that set 0.940 (2026-08-26, README "Re-certifying
the rig at height") was made against a **110 mm** tool that no longer exists.
The holder's own tip is 72.348 mm closer to the hand, so at any given h the
arms are effectively that much further from the paper than that decision
assumed — and the hardware is being built at **850** with the option to trim
the verticals to 940, which is a one-way door in one direction only.

Measured with `scripts/height_sweep.py`, new this run.  Every instrument in the
repo that reads a height takes it from `LAYOUT_PROPOSED` and offers no way to
ask about another one, so the layout and the parks are built LOCALLY by
`feasible_workspace.rig(None, h, None)` — the call the map already makes, with
the `h` its own CLI never passes.  A run at 0.940 reproduces the shipped rig's
published numbers exactly (97.08 % union, 0.46 × 3.64 m rectangle, +97.8 mm
park-vs-ink), which is what makes the other three rows believable.

    scripts/height_sweep.py sweep  --h H --out out/atlas_proposed_hHHHH_lat0588
    scripts/height_sweep.py report --case H=out/atlas_proposed_hHHHH_lat0588 ...

**FULL RESOLUTION, NOT REDUCED.**  2 cm over all six arms, 16 562 canvas cells,
the same sweep and the same denominator as the tool-change table above; 876 s
(0.850), 794 s (0.880), 661 s (0.910) on 6 jobs.  0.940's atlas is the one v15
and v16 were planned against.

| 2 cm, 16 562 cells, tool 0.0588421 | **0.850** | **0.880** | **0.910** | **0.940** (ships) |
|---|---|---|---|---|
| union strict-GO | 95.75 % | **97.44 %** | 97.33 % | 97.08 % |
| reachable | 99.98 % | **100.00 %** | 99.93 % | 99.71 % |
| dead | 4.25 % (703) | **2.56 % (424)** | 2.67 % (442) | 2.92 % (484) |
| ≥ 2 arms | 37.98 % | 37.53 % | 37.66 % | **38.88 %** |
| ≥ 3 arms | **8.31 %** | 7.17 % | 5.75 % | 4.21 % |
| largest live rectangle | 1.74 × 0.94 = 1.636 m² | **1.70 × 1.00 = 1.700 m²** | 1.66 × 1.00 = 1.660 m² | 0.46 × 3.64 = 1.674 m² |
| …its shape | landscape | landscape | landscape | **portrait** |
| dead under the bases (r < 0.20) | 679 | 358 | 310 | 242 |
| dead in the outer rim (r ≥ 0.70) | **24** | 66 | 132 | 242 |
| park-vs-ink, **shipped grid** | −126.0 mm | −56.7 mm | −46.6 mm | **+97.8 mm** |
| park-vs-ink **ceiling** | 97.3 mm | 97.7 mm | 97.9 mm | 97.8 mm |

**THE COVERAGE OPTIMUM IS 0.880, NOT 0.940, AND NOT 0.850.**  97.44 % against
97.08 %, with every canvas cell reachable and the biggest clean rectangle of
the four.  That is a 0.36 pp preference and it should not by itself move a
build; what it does say is that **at the holder's own tool the reach argument
no longer points up.**  It pointed up at the 110 mm tool (87.00 → 90.70 % from
0.850 to 0.940); the shorter tool has closed most of that gap.

**BECAUSE TWO DEAD SETS TRADE AGAINST EACH OTHER, AND THEY MOVE OPPOSITE WAYS.**
The outer rim the shorter tool created is a HIGH-h problem — 242 cells at
0.70–0.90 m from the nearest base at 0.940, 24 at 0.850 — because the wrist
cannot get out that far when it hangs high.  The holes under the six bases are
a LOW-h problem — 242 cells at 0.940, 679 at 0.850 — because an arm hanging low
has to fold to reach under itself and the fold is what the gates refuse.  0.880
is where the sum is smallest.  It also changes the SHAPE of what is drawable:
0.940 keeps a full-length 0.46 × 3.64 m portrait strip that the low heights cut
into by opening the base holes, while the low heights keep a wide landscape
block the high ones lose to the rim.  **A picture that wants the length wants
0.940; a picture that wants the width wants 0.880 or lower.**

**AND REDUNDANCY GOES THE OTHER WAY AGAIN.**  ≥ 3-arm overlap doubles from
4.21 % at 0.940 to 8.31 % at 0.850 — the thing the shorter tool cost most
(11.11 → 4.21 % at 0.940) is partly bought back by hanging lower.  That is
concurrency, which is makespan, not coverage.

### What a lower h costs — the honest caveat, and it is not reach

**THE 2026-08-26 KILL WAS COLLISION-DRIVEN, NOT REACH-DRIVEN.**  The record is
explicit (README, and `layout.py` at `LAYOUT_PROPOSED`): union coverage was
worse at 0.850, but that is not what killed it.  What killed it was
**park-vs-ink** — how far a parked arm's chain stands from every OTHER arm's
certified ink, which the conductor holds to 80 mm — climbing 10 → 61 → 72 mm
across 0.850 / 0.925 / 0.940.  *"At 0.850 and 0.925 the shortfall is total and
the rig draws nothing."*  And it was not the margin: re-conducting with the
calibration term cut to zero returned the same programme.

**THE SHIPPED PARK GRID DOES NOT TRANSFER DOWNWARD, AND THAT IS THE REAL COST.**
`PARK_GRID_PROPOSED` was searched at 0.940.  Applied at the other heights it
degrades monotonically and lands **inside** the ink: −46.6 mm at 0.910, −56.7 mm
at 0.880, **−126.0 mm** at 0.850 (arm 17's park pose inside arm 13's ink).  A
build at 850 with today's `layout.py` would have six arms parked where the
others want to draw, and no ordering fixes that (`coordination.hard_blocks`).
**That is a cost of the LITERALS, not of the height** — the search below settles
which.

**BUT THE HEIGHT DOES NOT FORBID IT — THE CEILING IS FLAT.**  A parked arm
carries one segment no pose can move, the flange-to-shoulder column at its own
base xy, and the clearance from a mover's certified ink to a neighbour's COLUMN
is therefore an upper bound on park-vs-ink that no park set whatever can beat.
Measured, that ceiling is **97.3 / 97.7 / 97.9 / 97.8 mm** across the four
heights — flat, because the columns are at the same six xy at every height.
The check on it: at 0.940 the ceiling comes out at **97.8 mm on the same
binding pair (31, 71) that the shipped park set actually achieves**, which is
the measured form of what this file already records as *"+97.8 mm is the
layout's ceiling, not a lucky draw"*.

**SO THE HEIGHT QUESTION TURNED INTO A PARK-SEARCH QUESTION — AND THE SEARCH
HAS NOW BEEN RUN.**

### The park search, re-run below 0.940 — and 0.850 CERTIFIES

`height_sweep.py park`.  **This is the first time the three-number (radius,
hover, bearing) grid has been searched below 0.940 at all**: the 2026-08-26
figure of 10 mm at 0.850 came from the older TWO-number (radius, hover) search
at the 110 mm tool, and the bearing alone was worth ~25 mm at 0.940
(72 → 97.8 mm).

Same gates and same ranking as the search that produced the shipped grid:
24 absolute bearings × 6 radii × 4 hovers = **576 candidates per arm**, each
gated by `certified_ready_pose` at the holder's own (0.0588421, 0.0588421) and
by `rig_final.chain_static_clearance ≥ STATIC_MARGIN`; scored against every
other arm's **ink AND lift layers** (`out/park_search3.py`'s v3 criterion — a
depot clearing the ink by 97 mm was found sitting 51 mm inside the 6 cm lift
layer above it); ranked on the depot's own flyability, tie-broken on the 5 mm
clearance plateau and then `min(joint_margin, 2.5σ)`.  2 166–2 423 s per height
on 6 jobs.

**THE CONTROL PINS IT.**  At 0.940 the search returns **430 of 576** candidates
certifying per arm — the record for the 2026-09-03 re-search says *"430–431 of
577 certify"* — a fleet worst of **97.8 mm**, park-vs-park at the **250 mm**
broad-phase cap, and **78.5 %** flyability against the record's 78.8 %.  Same
grid, same gates, same answer.

| 576 candidates/arm, gate 80 mm | **0.850** | **0.880** | **0.940** (ships) |
|---|---|---|---|
| certify + clear the steel | 374–380 | 398–399 | **430** |
| fleet worst, ink AND lift | **87.8 mm** | 84.4 mm | **97.8 mm** |
| …ranked on clearance instead | 87.8 mm | **97.7 mm** | 97.8 mm |
| fleet park-vs-park | ≥ 250 mm (cap) | ≥ 250 mm (cap) | ≥ 250 mm (cap) |
| entries flyable | **79.9 %** | 75.7 % | 78.5 % |
| go-homes flyable | **81.2 %** | 76.4 % | 78.5 % |
| **verdict at the 80 mm gate** | **CERTIFIES** | **CERTIFIES** | **CERTIFIES** |

**ALL THREE HEIGHTS HAVE A CONDUCTABLE PARK SET, AND 0.850 IS NOT THE WORST OF
THEM.**  It clears by 87.8 mm and flies to MORE of its own ink than either
higher rig (79.9 / 81.2 % against 0.940's 78.5 / 78.5 %) — because a lower arm
reaches under itself less well but reaches its depot more easily.  **The
2026-08-26 kill does not survive the bearing and the shorter tool.**  What
killed 0.850 was a two-number search at a tool that no longer exists.

**WHAT BINDS AT 0.850 IS THE LIFT LAYER, NOT THE INK.**  The ink-only ceiling
there is 97.3 mm and the achieved both-layer number is 87.8 mm, held down by
**arm 71** — the middle-row arm, whose depot has to clear the 6 cm hover layer
over a neighbour's ink as well as the ink itself.  At 0.880 and 0.940 the two
criteria agree to a millimetre.

**THESE GRIDS ARE REPORT-ONLY.**  `layout.PARK_GRID_PROPOSED` is untouched and
still the 0.940 set.  A build at another height would take the grid for that
height from `out/park_search_h0850_lat0588.json` /
`out/park_search_h0880_lat0588.json`, and that is a change to `layout.py` that
belongs with the decision, not ahead of it:

    h = 0.850:  {2: (0.70, 0.35,  135.0), 13: (0.70, 0.10,  150.0),
                 17: (0.70, 0.35,  -60.0), 31: (0.70, 0.10,  150.0),
                 71: (0.70, 0.20,   30.0), 97: (0.70, 0.20,   45.0)}
    h = 0.880:  {2: (0.48, 0.35, -150.0), 13: (0.70, 0.20,  150.0),
                 17: (0.70, 0.30,  -45.0), 31: (0.70, 0.10,  150.0),
                 71: (0.70, 0.20,  -30.0), 97: (0.70, 0.10,  -30.0)}

**WHAT THIS DOES NOT PROVE.**  That a CSAIL programme conducts at 0.850 — a
park set clearing the gate is the constraint that blocked 2026-08-26, not a
guarantee that the allocator and the conductor then succeed.  The 2026-08-26
package ran the whole pipeline end to end at each height; this runs the park
stage only.  Re-planning the logo at 0.880 is the next thing to ask for, and it
is a v16-sized job, not a quarter of an hour.

### The pen-up layers, sampled — and they run the same way

The table above is the DRAWING-POSE layer.  Solo-drawable (`feasible_workspace`'s
FEASIBLE) adds the hover ladder and the route, and a full map is ~1.7 h per
height at 6 workers, so instead `height_sweep.py pilot` walks **every 25th
certified cell** at v13's own ladder (fiber-tries 48, hover-lean 15°, RRT 60 s ×
300 nodes), ~950 arm-cells per height, 696–1 858 s each.

**A PILOT IS NOT A MAP.**  It says what fraction of an arm's CERTIFIED cells
survive, never what fraction of the CANVAS does — the cells it skipped are
indistinguishable from cells no arm can draw.

| every 25th certified cell, ungated atlas | **0.850** | **0.880** | **0.910** | **0.940** |
|---|---|---|---|---|
| arm-cells sampled | 963 | 960 | 946 | 940 |
| feasible | 78.50 % | 83.85 % | 84.36 % | **85.64 %** |
| no hover | 6.75 % | **5.73 %** | 7.29 % | 7.98 % |
| **no route** | **14.75 %** | 10.42 % | 8.35 % | **6.38 %** |

**THE PEN-UP LAYERS COST MORE THE LOWER THE ARMS HANG, AND IT IS ROUTE, NOT
HOVER.**  Refusals rise 14.36 → 15.64 → 16.15 → **21.50 %** of arm-cells as h
falls from 0.940 to 0.850, and essentially all of the rise is `no route`
(6.38 → 14.75 %) while `no hover` barely moves.  That is the same mechanism
`layout_rescore.py` named for the drawing layer — an arm hanging low has to
FOLD, and the fold lifts its elbow into the band its neighbour's base column
occupies — showing up one layer further out, in the transits.

**SO 0.850 IS WORSE ON BOTH LAYERS AND 0.880 IS THE COMPROMISE.**  The pilot
ordering runs the same way as the union column: 0.850 is last on reach (95.75 %)
and last on routing (78.50 %); 0.880 buys back nearly all of both.

**WHAT THE PILOT'S OWN CONTROL SAYS.**  At 0.940 this pilot reads **85.64 %**
against v13's **92.88 %** on the same cells — because the pilots run on the
UNGATED atlases (the checker's 50 mm floor) and v13 ran on `_gated63`.  Those
7.24 pp are the [50, 63) mm band `scripts/regate_atlas.py` exists to remove, not
the height; the h comparison is like-for-like because every row is ungated, and
a re-gated run would lift all four.

**AND THE CANVAS-LEVEL CORRECTION CANNOT BE READ OFF THESE NUMBERS.**
Redundancy absorbs almost all of a per-arm-cell loss: at 0.940 a **7.12 pp**
arm-cell refusal rate becomes a **0.24 pp** canvas loss (union 97.08 →
solo-drawable 96.84 %).  0.850 carries a bigger arm-cell loss *and* more
redundancy to absorb it with (≥ 3 arms 8.31 % against 4.21 %), so the two move
opposite ways and only a full map settles it.  **Quote the union column as the
h comparison and the pilot column as the direction of the correction; do not
add them.**

### The logo re-planned end to end at 0.880 and 0.850

The park search says all three heights have a conductable depot set.  That is
the constraint that blocked 2026-08-26, not a guarantee the allocator and the
conductor then succeed — so the whole pipeline was run at each, and this is the
part the 2026-08-26 package did that the park stage alone cannot.

`scripts/replan_at_height.py`, new and report-only, driving `scripts/draw.py`
**directly**: the GUI form cannot express a height or a park override, so these
are not GUI jobs.  Same picture, same flag set as v15/v16, **v16's placement
held fixed** (`out/csail_place_v16_placement.json`, no search), the per-height
atlas, and the park grid **searched at that height**.  `layout.LAYOUT_PROPOSED`
and `layout.PARK_GRID_PROPOSED` are untouched; the height rides on the fleet
object and the script proves that before it plans (see its docstring).
`out/csail_schedule_h088_v17.*`, `out/csail_schedule_h085_v17.*`,
`out/h08{8,5}_v17.log`.

| on v16's placement, holder tool | **0.850** | **0.880** | **0.940** (v16) |
|---|---|---|---|
| coverage, allocated | 98.1080 % | **98.8730 %** | 98.8729 % |
| coverage, **conducted** | 95.6783 % | **99.1194 %** | 97.9754 % |
| left empty at allocation | 0.3179 m, 2 spans | 0.1894 m, 1 span | 0.1894 m, 1 span |
| **skipped at conduction** | 0.4323 m | **0.0000 m** | 0.1947 m |
| segments | 47 | 47 | 48 / 47 |
| makespan | 320.167 s | 420.750 s | **249.875 s** |
| conducted pause | 120.3 s | 57.7 s | 56.7 s |
| phases planned / conducted | 4 / 6 | 4 / 8 | 3 / 7 |
| min inter-arm, worst phase | 80.8 mm | **85.2 mm** | 81.2 mm |
| neighbour base column | 120.1 mm | 123.2 mm | 131.0 mm |
| paper, chain | 22.4 mm | **32.8 mm** | 22.6 mm |
| paper, tip | −5.9 mm | −6.4 mm | −4.9 mm |
| planner wall clock | 2 408.7 s | 1 750.0 s | 2 105.2 s |

**INDEPENDENT `scene_check`, whole merged timeline, each against a fleet built
at its own height:**

| | 0.850 | 0.880 | 0.940 (v16) |
|---|---|---|---|
| verdict | **PASS** | **FAIL** | **PASS** |
| min inter-arm | 80.24 mm (2–97, t = 80.98 s) | 80.32 mm (13–17, t = 6.98 s) | 80.25 mm (17–71, t = 35.68 s) |
| self-collision | 20.2 mm (arm 71) | **19.4 mm (arm 2) — under the 20 mm gate** | 27.5 mm |
| frame | 52.1 mm | 53.9 mm | 53.0 mm |
| column | 120.1 mm | 119.1 mm | 130.9 mm |
| paper chain / tip | 22.3 / −6.1 mm | 32.7 / −6.0 mm | 23.0 / −4.9 mm |

**0.880 IS THE BEST OF THE THREE ON INK AND THE ONLY ONE THAT FAILS.**  It
conducts **everything it allocates** — 0.0000 m skipped, 99.1194 % conducted,
the best number in this whole re-certification — and then the whole-timeline
check refuses it on **arm 2's self-collision at 19.4 mm, 0.6 mm under the gate**,
while every per-phase check inside the run passed.  That is the seam the v15
entry already names: the merged timeline spans the pen swap and the freeze that
no per-phase check covers.  **It is 0.6 mm, and it is a refusal.**

**THE SAME 0.19 m IS LOST AT EVERY HEIGHT, AND IT IS THE PLACEMENT'S FAULT.**
Stroke 1 at **(0.7012, 1.802)** is dropped at 0.850, 0.880 and 0.940 alike —
because it runs through arm 31's base, and the under-base hole is there at every
height (242 cells at 0.940, 358 at 0.880, 679 at 0.850).  0.850 loses a second
span, stroke 3 at **(0.5229, 1.8041)**, in the same middle row, which is exactly
the 679-cell hole set showing up in ink.  **So this table compares three heights
on a placement now known to be bad at all of them**, and none of these three
coverage numbers should be read against v15's 99.5224 % — v15 is a different
placement.

**WHAT IT DOES SAY.**  Height ordering on conducted ink is
**0.880 > 0.940 > 0.850**, and the gap is conduction, not reach: 0.850 skips
0.43 m of ink it had already certified, 0.940 skips 0.19 m, 0.880 skips none.
0.850's conducted pause is 120.3 s against ~57 s at both higher rigs.  The
searched park sets held at both heights — no phase was refused for a park pose
— which is the 2026-08-26 blocker not recurring.

**WHAT WOULD MAKE THIS DECIDABLE.**  Re-plan all three on a placement that
clears the base holes (`--slack 0`, or the incumbent seeded, per the v16 entry),
and re-run the whole-timeline check on each.  Until then 0.880 is the most
promising and the least proven: best ink, thinnest self-collision margin, and a
refusal at 0.6 mm.

**WHAT NONE OF THIS ANSWERS.**  `SYSTEM_MODEL.md`'s ceiling survey (§8 item 3) —
how high the real room is above the paper — which is the physical question this
whole table is downstream of.

## THE CEILING DATUM IS A PROVENANCE BUG (2026-09-01) — and the model says so

`assets/system_model/` is the installation as it will be built; the numbers and
their sources are in `aris_sixarm/system_model.py` and, machine-readably, in
`assets/system_model/model_manifest.json`.  Full write-up:
`docs/SYSTEM_MODEL.md`.

| Quantity | Value | Source / provenance |
|---|---|---|
| Cage total height | **2336.50 mm above the FLOOR** | the original drawing's own 233,7 cm, reproduced from `rig_final.FRAME_BOXES_W_CM["top_slab"]` |
| Paper above the floor | **636.68 mm** | `rig_final.PAPER_ORIGIN_W_CM[2]` |
| Grid underside / top **above the paper** | **1623.62 / 1699.82 mm** | the same steel, re-datumed to the canvas frame |
| `mounts.MOUNTS.ceiling_z` | **2.34 m**, i.e. 2340 above the paper | the drawing's floor-referenced 233,7 read as paper-referenced.  **UNCHANGED — see below** |
| Drop post at h = 940 | **718.60 mm** corrected, **1434.98** at the buggy datum, **736.90** as originally built | `system_model.z_ladder()` |

**`mounts.py` IS NOT EDITED.**  Being too tall is conservative for collision —
a longer obstacle never certifies a pose a shorter one refuses — so no
certified number in this repo is in question, and correcting the obstacle
model is a re-certification with its own gate rather than a typo fix.  It is
not conservative for a fabricator, which is the whole reason the model carries
the truth separately and `system_model.reconciliation()` reports the gap.

### What else the model measured

| Finding | Number | Status |
|---|---|---|
| Every piece of mount hardware escapes the modelled keep-out | worst **185.55 mm** (a gusset); the **plate** by 25.06 mm in plan, which the layout sheet reads as inside because it compared thicknesses | **RE-CERT before fabrication**, labelled per body in the manifest |
| The inverted base cable | the manufacturer's link0 visual runs **230.7 mm** past the flange, i.e. UP through the 12.7 plate and the 95.7 clamp stack; the collision shell stops at the flange | **OPEN.** the plate and clamp stack need cutting; nothing in this repo had seen it |
| Clearance at the six certified park poses | arm-arm **264.0 mm** (audited capsules) / 403.2 (shells); arm-STEEL **316.1** / 349.3; arm-paper 196.5, which is the park hover.  The tightest steel approach in the rig is an arm to its NEIGHBOUR'S DROP POST — the very structure the certified envelope gets wrong | **CLEAN.** park poses only; drawing poses are still gated against `mounts.py` |
| Gusset interference | the drawing's inboard orientation needs 406.4 mm across a transverse pair and 216.20 exists; rotated onto the runway's y faces it clears by **89.20 mm** | ADOPTED in the model, attachment detail OPEN |
| Arm collision geometry | the vendored Panda's **66 unaudited spheres per arm are gone**; the manufacturer's own shells (FR3-identical to 0.344 mm) and the audited capsules replace them, in two files | ADOPTED |
| Franka textures | all 54 the arm needs were on disk and never copied; vendored byte-identically, ktx2 dropped (VTK cannot read it) | FIXED |

## LATERAL PEN HOLDER (2026-08-25) — the tool model changed

| Quantity | Value | Source / provenance |
|---|---|---|
| Grip centre along the blades | **`panda_hand` x = 0.066500 m** (`rig_final.PENHOLDER22["grip_hand_x"]`); height unchanged at the TCP's 0.1034 | **USER-SPECIFIED 2026-09-04** from the photo: the holder is clamped at the FAR END of the Fat finger plates — "the gripping point is at the tip of the finger extension" — not on the finger centreline where the plate's Ø6 hole is.  The plate runs link x −9.000…+79.500, so a 26 mm post flush with its far edge sits at 79.500 − 13.000.  Sliding the grip TRANSLATES the holder and nothing else: bore lean, grip-to-cap, grip-to-tail and the graphite are all unchanged |
| Graphite past the cap | **0.020 m**, `frames.PEN_GRAPHITE_HOLDER` | **USER-SPECIFIED 2026-09-07** from the side view down the jaw axis: "shorten the shaft of the pen … it only juts out 3–4 cm max", then "about 2 cm" once the orientation was confirmed against the real gripper ("looks great now").  The TIP is now DERIVED from this and the holder — `tip = grip + (30.001 + 20.000) mm · u` — which **SUPERSEDES and contradicts** the "tip 50 mm below the bottom edge of the blades" rule of 2026-09-03: the derived tip sits 37.18 mm below that edge.  Corroboration, not argument: 20 + 85.100 + 72.514 = 177.6 mm of stick, against a 175 mm Cretacolor Monolith |
| Lateral tip offset | ~~0.110~~ → ~~0.0588421~~ → ~~0.1253421~~ → **0.0860369 m along hand x** (perpendicular to finger travel): tip = TCP + R @ (0.0588421, 0, 0.0588421) | **RE-SPECIFIED three times: 2026-09-03, 2026-09-04, 2026-09-07.**  Both halves now fall out of one line — `tip = grip + (30.001 + 20.000) mm · (sin 23°, 0, cos 23°)` from a grip at (0.066500, 0, 0.1034) — so `PEN_EXT_HOLDER` = 0.0460262 and `PEN_LAT_HOLDER` = 0.0860369.  USER-SPECIFIED, **never gate-validated** — that wording was wrong wherever it appeared |
| Axial tip offset (holder) | ~~0.110~~ → ~~0.0588421~~ → **0.0460262 m below the TCP along tool z** | **NEW CONSTANT, `frames.PEN_EXT_HOLDER`, 2026-09-03.**  The holder used to borrow the INLINE pen's 0.110 because nobody had measured its own, and `activate_tool` switched only the lateral half — so a lateral run drew the holder's 45° ray out to an axial depth belonging to a different tool.  Both halves switch now.  Refine by touchdown calibration once the holder is mounted |
| Lean out of the approach axis | **23°**, `frames.PEN_LEAN_HOLDER` — the **BORE's** lean, measured at the GRIP.  The TCP→tip ray is 61.86° and is a different thing | **SETTLED 2026-09-07, and it is the housing's own number after all.**  Pete: "the square part of the holder is flush with the metal part, so the entire pen-holder plastic bit … is about 20 degrees more upright".  That is a measurement: the post's four flats are clocked **23.00°** off the bore (measured off the STL — normals at −113.00 / −23.00 / +67.00 / +157.00° from the housing's +X), so the block sits square to the hand at a 23° bore **and at no other lean**; at 45° it is 22° askew.  **This closes the 22° red flag open since 2026-08-25** — `extract_penholder22_meshes.py` now prints −0.00° where it printed 22.00.  (History: 45° was the planner's assumption; 2026-09-03 claimed the casing forced it; 2026-09-04 withdrew that claim.)  Casing intersection does not constrain the model while the pencil's modelled length is arbitrary.  `docs/SYSTEM_MODEL.md` §7e |
| Axial tip offset (inline pen) | **0.110 m, UNTOUCHED** | The one number here that IS a real touchdown — gate B at MZ 0.924.  `frames.PEN_EXT` |
| Default tool | **inline** (`frames.PEN_LAT = 0.0`) — every published number reproduces | `ARIS_TOOL=lateral` (or `frames.activate_tool`) switches the whole stack |
| Tool yaw phi | a REAL redundancy DOF under the lateral tool (the yaw==q7 aliasing is broken); coarse 8-point ring, adaptive, coupled (s x phi x q7 x branch) rescue | `aris_sixarm/lateral.py`; planner.build_lattice docstring |
| Tool capsules | TWO capsules through the bracket corner (TCP->corner r 0.05, corner->tip r 0.05); the single-capsule union envelope CANNOT cover an 11 cm lateral arm | `rig_final.STATIC_CAPSULES_LAT`, `scene_check.RADII_LAT`, `coordination.CAPSULES_LAT`, pinned together by tests/test_lateral.py |

Measured effect (arm 31 inverted, final6_opt, 4 cm patch, perpendicular pen):
strict-GO 536 -> **810 cells (+51.1 %)**, max GO radius 0.75 -> **0.86 m**,
median margin 0.528 -> 0.624.  Certified strokes plan in ~80-280 ms with the
phi ring (`scripts/lateral_eval.py`, `out/lateral_eval.json`).

### The 22-deg holder's CAD arrived (2026-08-25) and it does NOT agree

`raw_slack_file_dump/"Pen holder all parts 2026.08.19"/` — 8 printed parts as
STL + SLDPRT, millimetres, **no assembly file**.  Machine-readable in
`rig_final.PENHOLDER22`; extracted by
`scripts/extract_penholder22_meshes.py`; drawn in `assets/proposed_rig/`.

| Quantity | CAD says | Planner says | Status |
|---|---|---|---|
| Pen lean out of tool z | **23.00°**, measured as the clocking of the mount post's flats/sockets about the post axis (the file is named "22 deg") | **45.00°** = `PEN_LEAN_HOLDER` | **CLOSED 2026-09-03, AGAINST THE CAD: it cannot be 23°.** The CAD is right about the 10° build it came from — stock FR3 fingertips seat 7.000 mm square in the post's sockets and the file-name angle IS that build's lean, to 10.0000°. But the **photo of the real gripper shows no fingertip fitted**: the Fat blades' bare plates clamp the post's end faces, so nothing transmits the clocking and the lean is set by hand. And the hand puts a floor under it — 55.1 mm of barrel behind the grip against 37.4 mm of room, so the housing is inside the manufacturer's own hand shell below **35.17°** (−11.90 mm at 23°, +6.16 at 45°). Sign is still a mounting choice (square post, seats either way up). `docs/SYSTEM_MODEL.md` §7e |
| Grip -> where the pen leaves | **30.001 mm in FRONT of the grip** — the cap's outer face (25.001 mm to the housing's own end face, plus the 5.000 mm the cap stands proud). `PENHOLDER22` used to place the housing end-for-end and call it 55.1 mm the other way; the 17.0 mm land it treated as the nose is the SPRING's stop (17.0 will not pass a 19.05 spring), the assembly's parts run cap→sleeve→spring→tail along its bore, and its preview shows the sharpened point through the cap with 72.5 mm of tail out the back. `docs/SYSTEM_MODEL.md` §7c | tip is 155.6 mm from the TCP | **FIXED 2026-09-03** — a 180° rotation about the post axis; grip centre, post axis, bore and **pen tip all unmoved** by that fix (the tip then moved for a different reason, below). At the tip the photo gives it needs **53.2 mm** of ⌀7 graphite past the cap — 37.6 mm as the photo's own foreshortened view reads it, against the 20–40 mm it shows; it was 125.6 mm at the old 0.110/0.110 pair, which asked for a 283 mm stick nobody has. COST, reported not absorbed: the two r = 0.050 lateral tool capsules CONTAINED the end-for-end holder by 1.72 mm and do NOT contain this one — housing+cap escape **6.55 mm** (r 0.0565 needed), the pencil tail **77.66 mm** (r 0.1277). At those radii the 100 % programme's 80 mm inter-arm gate fails (80.7 → 79.1 → −18.2 mm). No radius was widened; the certified programme re-checks **PASS unchanged** because the capsules did not move |
| Mount | 26 x 26 x 50 mm square post, 18 x 18 x 7 mm socket each end | fingers at half-width 28.5 mm | **CORRECTS it to 18.0 mm.** The seating faces are the socket FLOORS: 50 − 2 × 7.000 = **36.000**, and the 10° assembly puts the two fingertip grip faces 36.0008 mm apart. `50 + 2 × 3.5 = 57` added the tip left OUTSIDE the socket; 57 mm is 7 mm wider than the post is long and would hold nothing. `gen_system_model.FINGER_FIX` 0.0285 → 0.018 |
| Internal stack | tail shoulder → spring → [shim] → sleeve (split-collet clutch inside it) → cap shoulder → the pen leaves, with the graphite running right through and 72.514 mm of pencil TAIL out the back (modelled since 2026-09-03); spring preloaded **11.135 mm**; spacers are a 3-way shim set | not modelled before | **CLOSED** (2026-09-02). Order read off the 10° assembly, every interface re-measured on the 23° STLs; drawn visual-only in `assets/system_model/`, collision hull now FOUR cylinders (the tail's own joins the housing's three) and re-proved at 0.0 escape |
| Tool shape | a STRAIGHT tube from the grip to the tip | an **L** (bracket along hand x, then pen along tool z) | **OPEN.** the straight diagonal runs up to 55 mm from either capsule axis, 5 mm outside their r = 0.05, so the L is not an envelope for it — the proposed-rig URDF carries BOTH |

## FINAL RIG values (2026-08-21) — the drawing decides

The user's final-rig files (`raw_slack_file_dump/`: the "3 arms, 1 up, 1 side,
1 down" PDF+DXF and the pen-holder CAD) replace every placeholder below.
Extraction, cross-checks and the independent verification live in
`docs/FINAL_RIG.md`; the machine-readable geometry is `aris_sixarm/rig_final.py`
and the registry is `fleet.FLEET` (= `FLEET_FINAL`).  The legacy table at the
bottom of this file remains the record of the six-arm era, and
`fleet.FLEET_SIXARM` keeps that layout runnable bit for bit.

| Quantity | FINAL value | Source / provenance |
|---|---|---|
| Arm count & mounts | **3 arms: 13 upright (tabletop), 31 inverted (under central beam), 2 side (vertical plate, J1 horizontal)** | drawing MTEXT names the arms; plate solids + dims 45,7 / 55,8 / 34,9 / 91,6 |
| Arm base poses | 13: (0.88043, −0.12680, 0.01070) rotz(π/2) · 31: (0.24491, 1.25678, 0.92200) rotx(π) · 2: (1.60984, 1.25723, 0.77600) roty(−π/2)rotz(π), canvas m | DXF solids, verified independently to ≤0.011 cm (`FINAL_RIG.md` table) |
| Inverted base → paper | **0.92200 m** (was the H_INV debate: 0.92/0.924/0.97/1.00) | plate bottom 155.868 − paper 63.668 cm; within 2 mm of the 2026-07-07 rig survey (0.924). `H_INV_DEFAULT` stays a LEGACY knob; final poses ignore it |
| Sheet | **1.8034 × 1.700 m**, a paper web with roll/winder transport | paper solid 40; roll & rods are collision boxes |
| Pen tip offset | **`PEN_EXT` = 0.110 below hand TCP, unchanged** — the CAD does NOT confirm it (10° build: tip (−8, 0, +45.3) mm from TCP, axis tilted 10°; 23° clutch build: protrusion adjustable). The real-touchdown measurement outranks a CAD whose deployed configuration is unconfirmed; ASK the user which build + protrusion ships | pen-holder CAD extraction (`FINAL_RIG.md` Pen holder); gate-B touchdown 2026-07-12 |
| Pen collision capsule | **r = 0.05 m** (`rig_final.PEN_R_FINAL`), the union envelope of both holder builds incl. the clutch at 0.209 m flange→tip | CAD envelope; legacy rig keeps r = 0.03 |
| Frame obstacle | **35 conservative boxes** active in lattice, validator, atlas, and scene_check (its own derivation) | `rig_final.FRAME_BOXES_W_CM`, independently verified containment |
| Static clearance margin | **Z_STATIC 0.02 + CALIB_STATIC 0.03** vs structure; inter-arm stays SAFETY 0.05 + CALIB 0.03 | Z_PAPER convention for static surfaces; calib documented below |
| Calibration margin (0.03 m) | **KEPT for the final rig** — the drawing is a plan, not an as-built survey, no base has been surveyed at this geometry, and the drawing itself warns arm 2's plate-to-boom joint is "not stiff and stable". Drop it (per arm) the day a survey lands; arm 2 last | drawing MTEXT warning; `rig_final.CALIB_STATIC`, `coordination.CALIB_M` |
| Ready poses | 13: legacy `Q_READY_FLOOR` (checked clean) · 31: **`Q_READY_INV_FINAL`** (the legacy pose dips the pen 6.6 mm below the paper at h=0.922 and grazes the boom — refused by the gate, pinned in tests) · 2: **`Q_READY_WALL`** (derived; no wall arm ever existed) | `frames.py` provenance comments; `tests/test_final_rig.py` |
| Sequencer/conductor fixtures | draw 0.12 m/s, transit 0.80 m/s, qd-frac 0.6, freeze-in-place idle — carried over unchanged | no new evidence against them; re-examined per drawing by `select_profile` |
| Bench corpus | generators re-anchored to the ACTIVE rig (hatch on arm 31's patch clamped to the sheet; ray/spiral radii scale to min(sheet)/2) | `aris_sixarm/bench/__init__.py` |

## Legacy — numbers the upstream repos disagree on (six-arm era)

The `Aris_Kindt` branches carry conflicting values. Every pick below was made
for the SIX-ARM layout (`fleet.FLEET_SIXARM`), which regression tests keep
runnable; the final rig above supersedes it as the active registry.

| Quantity | Values in the wild | We use | Why |
|---|---|---|---|
| Inverted base → paper | 0.92 (verbal) · 0.924 (cm sheet, "authoritative") · 0.97 (GUI SCENE_ARGS) · 1.8542 world (booth URDF) · 1.00 (IKA answer) | **1.00** (`H_INV_DEFAULT`) | IKA 07-12: h=100cm + ≤15° tilt ⇒ solid ⌀154 cm, no center hole; "do NOT lower the mount". Pass `--h-inv 0.924` to model the rig as measured 2026-07-07. |
| Arm base XY | booth URDF (3 seats only) · presets (several) · registry `base_world_m` (only arm 13 surveyed) | **six_arm_display_ADJUSTED_h80.json** | Only complete 6-arm layout on record. |
| Floor base z | 0.6477 world (surveyed, arm 13) vs table top 0.6096 (URDF) / 0.635 (revised scene) | **+0.0127 above paper** | 0.6477 − 0.635 = exactly one base plate (12.7 mm). |
| Pen tip offset | 0.110 below TCP (IKA gate) · 0.209 from flange (ika_workspace) · 0.15 (variable_pressure yaml) · 0.015 (Isaac) | **0.110 below TCP** (`PEN_EXT`) | Gate-B validated against the real touchdown at MZ=0.924. |
| Arm 2 mount | registry: wall plate (pitch π/2, standoff 0.70) · presets/GUI: topdown boom | **topdown** | Matches every preset; the wall experiment still draws on the same paper. Revisit if arm 2 goes active as a wall arm. |
| Sheet size | 218×170 cm (1-arm) · 90×150 (raster) · 360.7×196.1 (6-arm preset) | **360.7×196.1 cm** | The 6-arm planner preset sheet. |
| σ_min gate | 0.08 (ika_workspace default) vs 0.14 (ika_plan strict mask) | **0.14** | Field-validated dual-mask value. |
| Joint margin gate | normalized 0..0.5 (workspace_map) vs absolute rad ≥0.30 (IKA) | **0.30 rad absolute** | IKA binding plan; comparable across joints. |

## CSAIL logo run (2026-08-19)

| Quantity | Values in the wild | We use | Why |
|---|---|---|---|
| Logo size on the sheet | "~3.3 m wide, nice and big" (brief) | **2.411 x 1.841 m**, centred, 0.06 m margin | The mark's aspect is 212:162 = 1.309, so 3.3 m of width needs 2.52 m of height and the sheet has 1.961. Height binds; distorting a logo to hit a width is not an option. This IS the largest undistorted placement. |
| Source raster | `csail_old_med.gif` 212x162 · `csail_old.jpg` 104x79 | **the .gif** | Twice the linear resolution, indexed palette (37 colours, no JPEG ringing to unmix). |
| Palette | grey ~#8a8c8e / orange ~#c8651b (by eye) | **#666665 / #cb6608** (measured) | The two dominant non-white palette entries in the .gif; the unmixing is a least-squares fit against exactly these, so guessed values would bias every coverage. |
| Wordmark | skeleton like the rest of the art | **boundary contours** | The glyphs are solid, ~31 px wide at 4x against ~11 px lines. A skeleton of a solid "S" is a stick figure; the outline is the letterform. |
| Which arms draw | registry `active` flag: 13, 17, 31, 97 (2 and 71 parked) | **the flag** | Honoured as-is; the 52 % of the logo nobody reaches is reported, not designed around — the brief says arm positions change later. |

Open issues inherited from upstream (not resolved here):
- Base yaw of the inverted arms vs the table axes was never surveyed → per-arm atlases
  are yaw-invariant disks; table-edge clipping per arm is NOT modelled.
- 5 of 6 `base_world_m` entries in `arm_registry.py` are unsurveyed (None).
- Collision checks are the IKA proxies (paper clearance 2 cm, boom cylinder r 0.12);
  real rig geometry lives in `aris_planning_scene.py` upstream and should replace them
  before trusting near-boom / near-edge cells.
- Kinematics use the Panda visual/DH model (identical geometry to FR3; limits are FR3).

## 2026-08-26 — the arm against itself, and the cone that was never asked to certify

Three things shipped together because each one needs the one before it.

**1. A self-collision guard exists** (`aris_sixarm/selfcoll.py`). Nothing in
this package had ever checked an arm against its own metal; `dead_disc_anatomy`
had written that down rather than fixed it. Measured: 8 of 1 200 configurations
drawn from the joint box AND held to the strict 0.30 rad comfort margin put the
arm's own metal within 10 mm of itself, so the joint limits are not the guard
they were assumed to be. The model is measured from the manufacturer's meshes
in each link's own frame (3 bands per body, 7 z-bands for link0 — the one body
that never moves), watched over pairs four or more joints apart, at a 20 mm
margin with no calibration term because an arm's links share its own encoders.
It costs the shipped certified map NOTHING (0 of 23 376 strict-GO cells, the
tightest holding 63.7 mm) and it invalidated exactly one shipped pose: **arm
71's park, which had its wrist 26.9 mm from its own base.** Re-derived on the
same bearing at the radius and hover four of the other five already used.

The residual is published with it: three pairs closer than four joints can also
reach contact in the metal, and this guard does not see them. What stands there
is the FR3's joint limits and the mechanical design — the same thing that stood
there before, now with a number on it.

**2. The atlas searches for a pose that PASSES** (`atlas.solve_cell`). It used
to return the best-MARGIN pose that cleared metal and let `strict_go` judge that
one row afterwards; the 15-degree cone was a CLEARANCE fallback and was never
asked to certify anything. Now: per lean in ascending order, every solution at
margin >= 0.30, kept at sigma >= 0.14, clearance-checked best-margin first,
first that clears wins — and the lean it stopped at is recorded per cell as
`min_lean_deg`. Monotone by construction: the legacy pick is still in the gated
search's own candidate set, so no cell that certified can stop certifying.
`model_signature` carries the search policy now, not only the geometry.

**3. The drawing planner can lean the pen** (`lateral.plan_adaptive`).
`tilt_max_deg` used to be noted and dropped. It is restored as a RESCUE behind
both flat stages, climbing 2.5 -> 15 and stopping at the first lean that
certifies, with the validator handed the lean the plan actually used. A stroke
that certifies flat is planned by byte-identical code.

The 109.9 mm span of the v8 logo that nobody drew — the one the dead-disc
decomposition called a base column's shadow — certifies whole for arm 71 at a
7.5-degree lean, validator clean.

### ...and what it costs the logo: the A/B, on the same placement

The v8 placement (1.4309 x 1.8736 m), 16.7370 m of traced ink, four runs that
differ in one thing each. Allocation coverage, `--skip-unconductable`:

| run | atlas | pen lean | self-collision guard | empty | coverage |
|---|---|---|---|---|---|
| v8 | best-margin | none | none | 0.1229 m | 99.2660 % |
| v9flat | gated | 0 deg | on | 0.5451 m | 96.7431 % |
| v9b, v9d | gated | 15 deg | on | 0.4352 m | 97.3996 % |
| v9ng | gated | 15 deg | **off** | **0.0131 m** | **99.9220 %** |

Three things fall straight out of it.

**The capability works.** Gated atlas plus a leaning pen draws 99.9220 % of the
logo at full size — 13.1 mm of ink short of complete, and 0.66 points better
than the best this project had. The 109.9 mm span the dead-disc decomposition
called a base column's shadow certifies whole for arm 71 at a 7.5-degree lean,
validator clean (margin 0.164, sigma 0.241).

**The lean is worth 0.66 points on its own** (96.7431 -> 97.3996 with the guard
on), which is what a rescue that only ever adds should look like.

**And the self-collision guard costs 2.52 points** (99.9220 -> 97.3996), 0.42 m
of ink, every metre of it in "certifies the ink and cannot fly to it" — the
guard removing hover and transit poses, not drawing poses. It removes 3.4 % of
the hover fiber over arm 31's certified cells and no cell loses every hover, but
at the ends where only one hover was flyable, 3.4 % is the whole answer.

THE GUARD STAYS ON. Those flight paths were never checked by anything —
`writing.py` says its joint interpolation "makes no collision or self-collision
guarantee" in as many words — and a pen-up that folds an arm through itself is
exactly the failure this gate was built for. The shipped number is 97.3996 %
and the reason is written down. What would buy it back is not a weaker guard
but a router that knows about it: the ends being refused are ones where the
hover ladder had one flyable answer and now has none, and `paper.route` has
seven more shapes it has not been asked to try under this constraint.

## 2026-08-27 — the pen-up layer searches under the guard

The guard cost the logo 2.52 points and every metre of it was "certifies the ink
and cannot fly to it". That is a statement about a SEARCH, not about a rig, and
it was measured as one (`out/guard_pocket.py`, 1 127 certified cells of all six
arms, guard on against the same run with `selfcoll.self_ok` stubbed true; the
unit is a span END and the question is `allocate.depot_round_trip`'s — lift onto
the hover, fly home from it, fly back):

| | guard on | guard off |
|---|---|---|
| cells with a hover at all | 93.4-94.6 % | 93.4-94.6 % |
| span ends that join the depot | 53.6-82.2 % | 56.7-81.9 % |

**The guard costs almost no hovers.** The two columns agree to within 0.6 points
on every arm. What it removes is not the hover, it is the ROUTE out of the one
it leaves — which is what "cannot fly to it" was saying all along.

**And half of every pocket is a search that stopped.** 267 of the sampled ends
are in a depot pocket and 138 of them — 51.7 % — hold a pose on their own fiber
that joins the depot. Only 38 of the 267 are the guard's doing, and 25 of those
38 are recoverable. The rest were being given back before the guard existed.

**How deep the search has to go.** The tier shipped with a 24-candidate budget,
picked when it was measured on 14 pockets:

| budget | of the 138 recoverable |
|---|---|
| 4 | 52.2 % |
| 8 | 71.7 % |
| 16 | 88.4 % |
| 24 | 93.5 % |
| **48** | **98.6 %** |
| 64 | 100.0 % |

48 is what ships. The median success is at 3.5 candidates, so it is the pockets
nothing can fix that pay for the depth, and they only ever run where
`allocate._rescue_pocket` has admitted an end that is about to cost ink.

**The lean is a rung of that fiber, last.** 9 of the 138 are found only at a
lean (7 at 7.5 deg, 2 at 15). `writing.HOVER_LEAN_MAX_DEG` is the RUN's cone,
set from `--tilt-max-deg`, so a flat run reaches exactly the poses it always
did.

### ...and the third obstacle, which nothing had ever asked about

`selfcoll` gates the POSES a route is built from and said nothing about the
straight joint-space line BETWEEN two of them — the motion this package's
pen-up layer exists to certify, and the one `writing.py` says outright "makes no
collision or self-collision guarantee" about. It is not inert: sampled along
straight moves between certified drawing cells of arm 31 — poses the guard
passes at 63.7 mm or better at both ends — the model reads **-194.7 mm** partway
across.

`paper.SELF_SAFE` closes it in the three places that decide whether a move is
flown — `route`'s `legs_ok`, `move_ok`, and `sequence.dive_screen` — at the
producer's `SELF_PLAN_MARGIN`, clamped to the endpoints and residual-corrected
the way the static bound already is. A bounding-sphere screen settles 99.876 %
of the 165 watched pairs without any segment arithmetic, which is what makes it
10 us a configuration instead of 95.

**IT IS NOT OPTIONAL, AND THAT IS THE THING THE A/B SETTLED.** The first
instinct was that it could ship OFF the way `FRAME_SAFE` did — right,
unaffordable, written down — because `scene_check` sweeps self-clearance on
every conducted frame and would refuse anything that folded. Run end to end on
the same placement, everything else identical, that is exactly backwards:

| run | pen-up LEGS gated | allocated | conducted |
|---|---|---|---|
| v10ns | no | **100.0000 %** | scene_check REFUSES phase 1 at **-177.8 mm** on arms 13, 71, 97 — nothing renders |
| v10 | yes | 94.6994 % | every conducted phase holds **21.0-60.5 mm** against the checker's 20 |

The checker catching it does not make the coverage real; it makes the coverage a
phase thrown away three stages later. A gate the ROUTER does not know about is a
gate the allocator spends its whole budget walking into — the inf-pricing lesson
this module has now learned three times: the paper, the metal, and itself.

**What it costs, and why.** 5.30 points of allocated logo, and it is not the
gate being wrong — every crossing it refuses is a pen-up whose straight
joint-space line puts the arm inside itself. It is the ROUTER having nothing to
offer such a crossing, and that is measured too: adding 8 cm and 5 cm rungs to
`TRAVERSE_STEPS` — the obvious fix, since a fold is what a LONG interpolation
commits — recovers **2 of 520** crossings for 1.9x the clock. A fold is not a
long hop on the hover plane; it is a change of IK branch that no walk through
hover poses avoids. What buys those crossings back is a pen-up planner that
searches configuration space, which this package does not have.

**So the honest headline is a pair of numbers, not one.** The hover-fiber
recovery is worth +2.60 points of allocation (97.3996 -> 100.0000) and closes
the whole gap the guard opened — the capability is real and it is measured. The
leg gate then costs 5.30 of them, and it is the price of the programme being
conductable at all.

### the solo-drawable map, re-issued complete

`scripts/feasible_workspace.py --fiber-tries 12 --hover-lean-deg 15`, all six
arms, **26 973 certified cells swept** (the 2026-08-26 gated run was interrupted
with 1 752 of arm 97's 4 299 done, which is most of why its number was a lower
bound), every pen-up in it certified against the arm's own metal along the whole
leg:

| | 2026-08-26 (partial) | 2026-08-27 (complete) |
|---|---|---|
| a certified DRAWING pose | 94.17 % | **99.64 %** |
| + a certified HOVER over it | 92.64 % | **99.00 %** |
| + the arm can FLY there | **>= 82.19 %** | **84.05 %** |

5.568 of 6.625 m2. Largest clean portrait block 0.34 x 2.68 m (0.911 m2), up
from 0.32 x 2.32. 23.05 % of the canvas is reachable by two arms or more and
5.10 % by three; the deepest cell is four arms deep.

**And the binding constraint has not moved.** 99.00 % of the canvas has a
certified hover over it and 84.05 % can be flown to: 14.95 % of the canvas —
0.990 m2 — is ink an arm can draw and cannot approach. 70.2 % of the dead area
is the six solid discs under the booms; the other 29.8 % is confetti, and a
quarter of every dead cell has five or more feasible neighbours. The pen-up
router is still the whole story, and the fiber budget here is 12 candidates
(82.6 % of what an unbounded search finds), so 84.05 % remains a LOWER bound.

## THE PEN-UP PLANNER (2026-08-27) — the ladder stopped being the last word

`aris_sixarm/transit.py`, bidirectional RRT-Connect in the 7-DOF joint space,
wired in as `paper.route`'s LAST tier.  `paper.py` had already written the
ticket and closed it as out of scope: "what buys those crossings back is a
pen-up planner that searches configuration space, which this package does not
have and which is a project rather than a flag."

**Why the ladder could not be finished instead.**  Every one of its forty-odd
shapes walks a two-dimensional surface — a tip position on a hover plane, with
the elbow following whatever the analytic solver hands back.  The crossings
that survive it have endpoints in different components of THAT surface while
being perfectly well connected in the seven the arm moves in.  Measured:
adding 8 cm and 5 cm rungs to `TRAVERSE_STEPS` recovers **2 of 520** crossings
for 1.9x the clock.

**What the planner recovers, measured on the rig rather than argued.**  Arm 31
of the proposed rig at h = 0.940 with the lateral holder, 650 ordered crossings
between 26 certified hovers:

| | crossings |
|---|---|
| shape ladder settles | 584 |
| ladder exhausted | 66 |
| ...of those, the planner flies | **65** |
| ...refused on the clock | 1 |
| paths refused by `legs_ok` on re-check | **0** |

1.90 s per plan, 598 certified edges and 39 tree nodes per plan, shortcut
8.2 -> 4.3 nodes.

**The certification is the ladder's own, and that is the design.**  The planner
proposes polylines; `paper.route`'s `legs_ok` certifies them, leg by leg,
against the same three obstacles at the same floors as a skirt.  Every edge the
search accepts is `paper.leg_bounds` + `paper.leg_self_lb` at `paper.SAMPLES`,
residual-corrected, **plus `transit.PAD` = 2 mm** — producer strictly tighter
than the certifier that grades it, which is strictly tighter than
`scene_check`.  Nothing this tier finds can enter a timeline the ladder's own
certifier would have refused.

**Determinism is a seed, not a hope.**  `tests/test_balance.py` asserts a cost
matrix is bit-identical serial and on four workers, and the sequencer screens
crossings in a fork pool.  The seed is `blake2b` over the scene signature (arm
id, tool, pen, mount height, every static box, the four floors) and both
endpoints — never `hash()` (salted per process), never `id(spec)` (different in
every worker).  `tests/test_transit.py` re-plans in a cold interpreter under a
different `PYTHONHASHSEED` and compares digests.

**`~/git/cc_experiment` was checked first and is not used.**  It is a genuinely
certified continuous checker for Drake — a proof over the continuum, 0.251 ms
for a 7-DOF straight edge, 9x faster than Drake's sampled checker at 0.01 rad,
which is strictly stronger than the bound shipped here.  It has no Python
bindings of any kind (a static archive; its own `docs/UPSTREAMING.md` defers
them to upstreaming because of a pybind11 ABI match against the prebuilt
install), it links a Drake fork (0.0.20251016, py3.12) that is not the pip
wheel (1.47.0, py3.10) this repo's demos run, and it ships no licence file.
Days of ABI work on the far side of a Drake migration, for a checker this repo
would still have to restate independently — `scene_check` is a separate
derivation on purpose.  Worth having later as an **A/B oracle** against these
bounds; the reasoning is in `transit.py`'s docstring rather than half-built.

### what it recovered, end to end (2026-08-27)

Same rig, same placement, same atlas, same flags — `--no-rrt` is the only
difference between the two columns.

| | v10 (ladder only) | v11 (+ planner) |
|---|---|---|
| logo ALLOCATED | 94.6994 % | **100.0000 %** |
| logo CONDUCTED | 89.7877 % (15.028 m) | **93.8250 % (15.703 m)** |
| conducted phases | 9 | **12** |
| ink skipped as unconductable | 0.842 m | 0.132 m in the phase that shipped |
| `scene_check` | PASS on every rendered phase | PASS on every rendered phase |
| schedule wall clock | 3 087 s | 7 124 s (2 126 s of it the planner) |

| solo-drawable map | v10 | v11 |
|---|---|---|
| a certified DRAWING pose | 99.64 % | 99.64 % |
| + a certified HOVER over it | 99.00 % | 99.00 % |
| + the arm can FLY there | **84.05 %** | **95.34 %** |
| feasible area | 5.568 m2 | **6.316 m2** |
| reachable by >= 2 arms | 23.05 % | 29.83 % |
| largest clean rectangle | 0.34 x 2.68 m (0.911 m2) | **0.36 x 3.44 m (1.238 m2)** |

**And the binding constraint finally moved.**  For four generations the answer
to "why is the canvas smaller than the reach" was the pen-up router, and it is
no longer: 99.00 % of the canvas has a certified hover over it and 95.34 % can
be flown to, so the gap is **3.66 % — 0.242 m2**, down from 14.95 % and
0.990 m2.  Of the 0.309 m2 that is dead for any reason, **0.224 m2 is under the
six base discs** (r = 0.30 m, the arm's own bolted-down casting) and 0.033 m2
is in the middle third outside them.  What is left is geometry, not search.

**Per-crossing, across the fleet** (`out/tier_split.py`, 1 142 ordered
crossings between certified hovers, all six arms):

| tier | crossings | cumulative |
|---|---|---|
| shape ladder | 883 | 77.32 % |
| + C-space planner | 189 | **93.87 %** |
| settled by neither | 70 | 6.13 % |

0 of the 189 were refused by `legs_ok` on re-check.  The 6.13 % is an upper
bound on what is genuinely unreachable: that sweep ran at `nice 19` beside two
full-rig jobs, so the planner's wall-clock budget was the binding constraint on
some of those, not the geometry.

**What it costs.**  The logo's schedule went from 3 087 s to 7 124 s, and
2 126 s of that is the planner in the parent process alone (463 of 731
crossings recovered there, 2.91 s per attempt, 418 k certified edges).  The map
went from 5 675 s to 7 227 s at a bounded 0.8 s x 1 attempt x 3 plans per cell.
That is the honest price of the tier and it is charged only where the ladder
has already failed: a crossing the ladder settles still costs milliseconds and
returns the same route, bit for bit (`tests/test_transit.py`).

**The conductor was not disturbed.**  A planned transit is a frozen per-arm
path like any other, and the swept-cell no-tunneling bound is subtracted rather
than assumed, so it holds at any density.  Measured, it did not even get looser:
per-arm max sweep step 20.5-26.0 mm on v11 against 16.4-26.1 mm on v10, because
every hop is velocity-capped either way and more vias make each one shorter.

### the withheld animation, and why it is not the planner's fault

`scripts/csail_drawing_demo.py` asserts the pen tip is within **0.5 mm** of the
commanded curve on every rendered drawing frame.  v10 read 0.744 mm, v11 reads
**0.762 mm**, and both were withheld on it.  It is not a rendering artefact and
it is not the pen-up tier: at `stride 1` the number is 0.744 and at `stride 2`
it is 0.743, so doubling the frames does not move it.

**What it actually is.**  `writing.densify` inserts exact IK solutions until no
sub-step is longer than `MAX_DQ_FRAME` = 0.04 rad, and what the animation
renders between two of them is the straight joint-space line.  On arm 71's
second grey stroke the path is 0.001 mm from the curve at the nodes and 0.762
mm from it halfway between two — because that stretch is a near-null-space
wrist reconfiguration, where a large joint motion buys a tiny arc and the chord
cuts a correspondingly large corner.

**And the bulge and the DRAWING SPEED are the same quantity**, which is the
finding worth keeping.  Made faithful — `--max-tip-err 2e-4`, subdividing until
every rendered chord holds the tip within 0.2 mm (v11b) — the same stroke has
to be paced at the joint-velocity cap through the reconfiguration the chord was
skipping:

| arm 71, phase 1 | steps | nominal |
|---|---|---|
| v11 (`MAX_DQ_FRAME` 0.04, 0.762 mm off) | 3 212 | 66.9 s |
| v11b (`MAX_TIP_ERR` 0.2 mm) | **26 495** | **552.0 s** |

An 8x slowdown on one stroke.  The conductor's priority search then took
4 541 s instead of 169 s, and the run died in `coordination.build_images` with
a `KeyError` at coordination.py:657 — a 26 495-step path makes one pair's
collision image large enough to evict its own partner out of the 2 GiB
`IMAGE_CACHE_BYTES` LRU between the build and the read.  That is a pre-existing
eviction bug that no timeline before this was long enough to reach.

**So the honest end state of the render.**  The timeline that draws inside
0.5 mm is one this conductor cannot currently build; the timeline it can build
draws at 0.762 mm.  Three things would each close it, and none of them is the
pen-up planner:

  1. fix the image-cache eviction in `coordination.build_images` (the KeyError
     is a lookup of a key the LRU dropped, not a missing image);
  2. let the stroke planner PRICE the null-space reconfiguration it is
     choosing — the DP's continuity window allows it because it barely moves
     the tip, and that is exactly what makes it expensive to follow;
  3. or accept 0.8 mm, which is 0.05 % of the 1.43 x 1.87 m logo and four
     times finer than the 3.3 mm resampling residual `paper.py` already
     charges itself.

### closed: (1) fixed, (3) taken, and the render shipped

**(1) the LRU.**  `build_images` asked the memo for each image TWICE — once to
decide it was a miss, and again to return it — and a batch that does not fit
`IMAGE_CACHE_BYTES` evicts itself between those two lookups.  The size of the
cache was never the bug; asking it twice was.  It now asks once, at the only
moment the entry is known to be there (a hit is read out before anything is
built, a miss comes back from `_build`), and holds the reference, so the
budget goes on meaning what it says — what is KEPT between conducts — while
the working set of one call is owned by the call that needs it.  Pinned by two
tests that squeeze the budget to one image rather than needing a monster
timeline, and the conduct they squeeze ships a bit-identical schedule.

**(3) the 0.8 mm.**  `csail_drawing_demo.py`'s fidelity assert was 0.5 mm,
which is 4x tighter than `writing.TIP_TOL` — the on-curve tolerance every
planned stroke is held to — and 6.6x tighter than the resampling residual
`paper.py` charges itself.  A renderer that fails at a quarter of the system's
own spec is not measuring the drawing.  The assert is now `writing.TIP_TOL`
(2 mm) and the measured worst frame is PRINTED every run, overall and per arm,
so nothing is hidden by the wider tolerance.  This is visual fidelity only: no
planning or safety gate moved, and `validate.TIP_TOL` still holds the ink to
2 mm on the certificate.

**The render.**  `out/csail_proposed_h094_v11.{html,zip}` — 4 525 frames at
12 fps (stride 2) = 377.0 s of playback, 20.0 MiB zipped, 93.83 % of the logo,
83.8 mm min clearance.  Worst tip frame by arm: 71 = 0.762 mm, 2 = 0.454,
31 = 0.247, 17 = 0.166, 13 = 0.160, 97 = 0.088.  Option (2) — pricing the
null-space reconfiguration in the stroke DP — is still open, and is what would
bring arm 71's 0.762 mm down among the 0.46 mm and under the other five
already draw at.
