# The 2.22 m nobody drew — is it unreachable?

Mostly no. Of the 13 empty spans in `out/csail_program_6arm.json` (2.222 m, 13.8 % of the 16.15 m traced), **1.137 m in 7 spans cannot be certified by any arm at any pen length**, and 0.869 m of that is a single failure mode: the under-base holes of inverted arms 2 and 97. The other 1.085 m is recoverable, and 0.429 m of it needs no hardware at all.

Two verdicts, because neither dominates. **Atlas** = pointwise at the strict gates (margin ≥ 0.30, σ ≥ 0.14, 1-cell eroded per `fuzz_planner._go_mask`) — conservative. **Planner** = each span handed to `allocate.probe_stroke`, which certifies at 0.15 / 0.10 but demands a *continuous* walk of the redundancy band — harder in a different direction. Recovered = ≥ 95 % covered. Rebuild both with `python3 scripts/dead_spans.py`.

| # | ink | m | at (x, y) | atlas tier | planner tier | arm | why still dead |
|---|---|---|---|---|---|---|---|
| 0 | grey | 0.139 | 1.699, 0.979 | — | — (0.71 at pen300) | — | waist |
| 1 | grey | 0.109 | 1.766, 0.983 | pen300 | — (0.73 at pen300) | — | waist |
| 2 | grey | 0.020 | 0.990, 0.994 | — | — (0.00 at any pen) | — | waist |
| 3 | grey | 0.086 | 0.988, 0.991 | — | **pen200** | 31 | waist |
| 4 | grey | 0.408 | 0.921, 0.710 | pen300 | **pen200** | 2 (wrong ink) | — |
| 5 | grey | 0.360 | 2.404, 0.837 | — | **repartition** | 97 (wrong ink) | under-base, 97 |
| 6 | orange | 0.039 | 1.443, 1.052 | pen300 | **repartition** | 31 (wrong ink) | — |
| 7 | orange | 0.030 | 1.485, 1.024 | today | **repartition** | 31 (wrong ink) | — |
| 8 | orange | 0.161 | 1.792, 0.931 | — | **pen300** | 97 | waist |
| 9 | orange | 0.339 | 1.131, 0.410 | — | — | — | **under-base, arm 2** |
| 10 | orange | 0.210 | 2.173, 0.334 | — | — | — | **under-base, arm 97** |
| 11 | orange | 0.170 | 2.430, 0.334 | — | — | — | **under-base, arm 97** |
| 12 | orange | 0.150 | 2.362, 0.331 | — | — | — | **under-base, arm 97** |

**Per tier (planner).** re-ink 0.429 m (3 spans) · 200 mm pen 0.494 m (2) · 300 mm pen 0.161 m (1) · **dead 1.137 m (7)**. Length-weighted, ignoring the 95 % rule, some arm certifies 0.767 m today / 1.053 m at 200 mm / 1.233 m at 300 mm — but **0.000 m today by an arm holding the right ink**. The allocator already enumerated all 62 colour partitions, so that 0.767 m is a real two-colour conflict, not an oversight: spans 6–7 want arm 31 orange while spans 4–5 want arms 2 and 97 grey. It needs a mid-piece pen change, or a seventh arm.

**Tilt buys nothing, at any budget.** The inverted arms' eroded strict-GO annulus is 0.27–0.70 m today; ≤ 15° takes the outer edge to 0.72 m, ≤ 30°/45° only open the *inner* radius to 0.01 m. No tilt condition recovers a single span, and the best of them adds 0.15 m length-weighted. It is also not implementable: `planner.sheet_fields` hard-codes `R_w = rotx(π)`. Pen length is the lever that works (200 mm → 0.26–0.82 m, 300 mm → 0.34–0.88 m), but a longer pen *widens* the under-base hole, which is why span 5 falls from 1.00 to 0.92.

**What stays dead, and where.** Spans 9–12 (0.869 m) sit 0.07–0.18 m from the base axis of arms 2 (1.303, 0.351) and 97 (2.304, 0.351): an inverted FR3 cannot draw underneath itself (boom keep-out r < 0.12 m, plus joint limits). Spans 0–2 (0.269 m) sit in the **waist** — the y ≈ 0.99 band midway between the front row (y = 1.631) and back row (y = 0.351), 0.70–0.88 m from every base, where four 0.70 m annuli fail to meet. Arms 13 and 17 are floor-mounted at x = −0.112 / 3.719 with 0.78 m of reach and contribute 0.00 m of the logo; the whole piece is drawn by the four inverted arms.

**Gate caveat, checked.** Atlas-strict over-calls dead, and in both directions. Probing the worst point of each dead span with `plan_stroke` on a 5 cm stroke: span 5 at (2.391, 0.604) — dead under *every* atlas — returns **ok** for arm 97 at 110 mm (margin 0.193, σ 0.238), so the 0.30 gate condemned it, not reach; span 0 at (1.719, 0.979) returns **ok** for arm 2 at 300 mm (margin 0.360). But spans 9 and 10 return `split / start_infeasible, s* = 0` for **all 18 arm × pen combinations** — zero IK solutions even at permissive gates. Those are genuinely unreachable at this base layout. Conversely the atlas is *optimistic* on span 1 (96 % of points GO at 300 mm, only 73 % walkable), so the two verdicts are complements, not a bound and a refinement.

Figure `out/dead_spans_analysis.png`, numbers `out/dead_spans_analysis.json`. Inherited caveats: `h_inv = 1.00` (the rig measured 0.924 — a lower mount shrinks every annulus), yaw-invariant per-arm disks, no arm-vs-arm collision reasoning.
