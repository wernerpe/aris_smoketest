#!/usr/bin/env python3
"""The standing regression: the whole pipeline over five drawings that are not CSAIL.

    python3 scripts/bench.py                    # all five, the shipped settings
    python3 scripts/bench.py --only spiral      # one of them
    python3 scripts/bench.py --no-split         # allocation v1, for comparison

Every published number in this repository is one picture at one placement, and a
load balancer tuned until the CSAIL logo comes out fast is indistinguishable —
from inside that measurement — from one that has understood anything.  So this
runs the FULL pipeline (allocate v2 -> sequence -> conduct -> scene_check) over
`aris_sixarm/bench`'s five generated drawings, each of which stresses a
different part of the machinery and none of which was tuned to the logo, and
prints one row per drawing.

The columns, and what each is worth:

  coverage    certified metres over traced metres.  The constraint, not the
              objective: a run that goes faster by drawing less has not gone
              faster.  It is a property of the PLACEMENT as much as of the
              allocator — a stroke in a corner no arm reaches is dropped by any
              version of this code — so it is watched for regressions rather
              than maximised here.
  makespan    what the conductor's clock says, phases and pen swap included.
  floor       the busiest arm's own nominal programme, summed over phases.  No
              schedule can beat it: it is one arm's ink plus one arm's pen-ups,
              with every other arm assumed free.  It is what the ALLOCATOR
              controls, and it is the number stroke splitting moves.
  efficiency  floor / makespan.  What the CONDUCTOR controls: 1.00 means nobody
              ever waited.  Splitting the allocation and conducting it well are
              different jobs, and this table separates them on purpose — a
              drawing whose efficiency falls while its floor falls is telling
              you the allocator handed the conductor a harder problem.
  splits      cuts the balancer made that the conductor KEPT, summed over the
              phases, with the ones it handed back in brackets.  `build_phases`
              conducts the unsplit allocation too and ships the faster, so a
              phase whose cuts bought more contention than they saved floor
              comes back uncut — and that is a result, not a failure.
  solo        share of the run with EXACTLY ONE pen down.  `docs/SOLO_TIME.md`
              made this the diagnosis: 35 % of the shipped logo run was one arm
              working while three watched, and 48 % of that was splittable.  It
              is the metric the whole of allocation v2 exists to move.
  wall        wall clock of the whole pipeline for that drawing.

Results are written to docs/BENCH.md and out/bench.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from aris_sixarm import (allocate, bench, coordination, idle,  # noqa: E402
                         sequence, writing)
from aris_sixarm.fleet import FLEET, SHEET                            # noqa: E402
from csail_schedule import build_phases                               # noqa: E402

# THE RIG, NOT A TUNING KNOB.  A pen is a fixture bolted to an arm: the same
# lengths are on the fleet whatever it is asked to draw, so the corpus uses the
# ones the installation is standing with rather than a set chosen per drawing.
# Everything else here is a default from the module that owns it.
PENS = {2: 0.300, 31: 0.200, 71: 0.200, 97: 0.200}

# THE ONE SETTING THE CORPUS CHANGES, AND THE CORPUS IS WHY IT EXISTS.  The
# shipped `--max-probes 5` is a budget PER STROKE, and every CSAIL stroke is
# between 0.1 and 0.7 m, so five plan calls map one well.  Two of these five
# drawings have strokes an order of magnitude longer, and at a flat budget the
# spiral certifies 0.00 % of itself — not because no arm reaches it (the atlas
# says 96 % of it is reachable) but because nobody ever asked.  `probe_ref_m`
# turns on deep probing: the budget scales with length AND a barren gap is
# bisected instead of abandoned.  Both halves are needed — a flat budget of 40
# still certifies 0.00 % — and together they take the spiral to 85.6 %.  0.5 m
# is the stroke length the flat five was chosen for, so a CSAIL-shaped stroke
# gets exactly what it gets today and a 14 m one gets 144 calls.
PROBE_REF_M = 0.5


class Args:
    """The flags `build_phases` reads, with the shipped defaults."""

    def __init__(self, **kw):
        self.fps, self.substeps, self.subcheck = 12.0, 4, 2
        self.draw_speed = writing.DRAW_SPEED_FLEET
        self.transit_speed = writing.TRANSIT_SPEED
        self.qd_frac = 0.3
        self.safety, self.calib = coordination.SAFETY_M, coordination.CALIB_M
        self.idle_policy = idle.POLICY_FREEZE
        self.no_jit = self.no_retreat = False
        self.jit_frac = idle.JIT_FRAC
        self.freeze_all_phases = False
        self.reseq_tries = 3
        self.pause = 2.0
        self.__dict__.update(kw)


def solo_share(built):
    """Share of the conducted clock with EXACTLY ONE pen on the paper. -> float.

    Read off the schedule the conductor actually produced rather than modelled:
    at every clock step, how many arms' progress index lands on a sample whose
    segment id is not -1.  Weighted across phases by their step counts, and the
    pen-swap pause is excluded because nobody is drawing during it by
    construction and counting it would flatter every two-colour drawing.
    """
    steps = solo = duo = idlest = 0
    for B in built:
        M, samp, sch = B["M"], B["samp"], B["sch"]
        n = np.zeros(M, int)
        for aid in FLEET:
            if aid not in samp:
                continue
            P = np.clip(sch["progress"][aid][:M], 0, samp[aid]["n"] - 1)
            n += (samp[aid]["seg"][P] >= 0).astype(int)
        steps += M
        solo += int((n == 1).sum())
        duo += int((n >= 2).sum())
        idlest += int((n == 0).sum())
    if not steps:
        return 0.0, 0.0, 0.0
    return solo / steps, duo / steps, idlest / steps


def run_one(name, a, split=True, verbose=False, seq_opts=None):
    """One bench drawing, end to end. -> dict of metrics (or an `error`)."""
    t0 = time.time()
    strokes, meta = bench.make(name)
    arms = allocate.active_arms("all")
    inks = [c for c in allocate.COLORS if any(s["color"] == c for s in strokes)]
    kw = dict(opts=None, verbose=verbose, pens=PENS, active_override="all",
              sequencer="opt", max_probes=5, probe_ref_m=PROBE_REF_M,
              balance=True, split=split,
              draw_speed=a.draw_speed, seq_opts=dict(seq_opts or {}),
              return_home=a.idle_policy == idle.POLICY_HOME,
              atlas_dir=str(ROOT / "out"))
    def alloc(cut, tag=""):
        out = []
        for ink in inks:
            sub = [s for s in strokes if s["color"] == ink]
            r = allocate.allocate(sub, colors={x: ink for x in arms},
                                  **dict(kw, split=cut))
            r.update(name=f"phase {len(out) + 1}: {ink}{tag}", ink=ink,
                     strokes=sub)
            out.append(r)
        return out

    phases = alloc(split)
    # the same A/B the shipped pipeline runs (`csail_schedule.build_phases`):
    # the balancer prices every arm as though it were alone on the paper, so
    # only the conductor can say whether a cut paid for the contention it made
    alt = {}
    if split:
        cutk = [k for k, p in enumerate(phases)
                if (p.get("balance") or {}).get("n_splits", 0)]
        if cutk:
            base = alloc(False, " [unsplit]")
            alt = {k: base[k] for k in cutk}
    t_alloc = time.time() - t0

    traced = float(sum(p["total_len"] for p in phases))
    dropped = float(sum(p["dropped_len"] for p in phases))
    splits = int(sum((p.get("balance") or {}).get("n_splits", 0) for p in phases))
    row = dict(name=name, regime=meta["regime"], n_strokes=meta["n_strokes"],
               traced_m=traced, dropped_m=dropped,
               coverage=1.0 - dropped / max(traced, 1e-9),
               n_segments=int(sum(len(p["programs"][x]) for p in phases
                                  for x in p["arms"])),
               n_phases=len(phases), splits=splits, alloc_s=t_alloc,
               # the null-space swing between consecutive strokes that the
               # floored transit beats hide (`sequence.reconfiguration`)
               reconfig_rad=float(sum(sequence.reconfiguration(p["programs"][x])
                                      for p in phases for x in p["arms"])),
               transit_s=float(sum(p["transit_time"][x]
                                   for p in phases for x in p["arms"])),
               menu_variants=float(np.mean([v["mean_variants"]
                                            for p in phases
                                            for v in (p.get("menu_stats") or {}).values()
                                            if v.get("sizes")] or [0.0])),
               floor_alloc_s=float(sum(max((p.get("balance") or {})
                                           .get("loads_after", {0: 0.0}).values())
                                       for p in phases)))
    dt = 1.0 / (a.fps * a.substeps)
    pens = {aid: PENS.get(aid, 0.110) for aid in FLEET}
    try:
        built = build_phases(a, phases, dt, pens, alt=alt)
    except (SystemExit, idle.Unconductable, RuntimeError) as exc:
        row.update(error=f"{type(exc).__name__}: {exc}", wall_s=time.time() - t0)
        return row, phases, None
    pause = a.pause if len(built) > 1 else 0.0
    makespan = float(sum(B["sch"]["duration"] for B in built) + pause)
    floor = float(sum(max(B["sch"]["nominal"].values()) for B in built) + pause)
    solo, duo, none = solo_share(built)
    # SPLITS PROPOSED IS NOT SPLITS KEPT.  `build_phases` conducts the unsplit
    # allocation too and ships the faster, so a phase whose cuts cost more
    # contention than they saved floor comes back uncut — and the table has to
    # say which number it is quoting.
    kept = int(sum(int((B["res"].get("balance") or {}).get("n_splits", 0))
                   for B in built))
    row.update(
        splits_proposed=int(row["splits"]), splits=kept,
        splits_reverted=int(row["splits"]) - kept,
        makespan_s=makespan, floor_s=floor,
        efficiency=floor / max(makespan, 1e-9),
        solo_share=solo, multi_share=duo, idle_share=none,
        pause_total=float(sum(B["sch"]["pause_total"] for B in built)),
        min_clearance=float(min(B["rep"]["min_clearance"] for B in built)),
        margin=float(built[0]["sch"]["margin"]),
        scene_check=all(bool(B["rep"]["ok"]) for B in built),
        phases=[dict(ink=B["res"]["ink"], duration_s=float(B["sch"]["duration"]),
                     floor_s=float(max(B["sch"]["nominal"].values())),
                     splits=int((B["res"].get("balance") or {}).get("n_splits", 0)),
                     arm_nominal_s={str(x): float(B["progs"][x]["duration"])
                                    for x in B["res"]["arms"]},
                     arm_metres={str(x): float(sum(s["length"] for s in
                                                   B["res"]["programs"][x]))
                                 for x in B["res"]["arms"]})
                for B in built],
        wall_s=time.time() - t0)
    return row, phases, built


HEAD = ("drawing", "regime", "ink m", "cov %", "makespan", "floor", "eff",
        "splits", "solo %", "wall")


def _reconcile(rows):
    """Make `splits` mean SPLITS KEPT on rows from any version of this script.

    The per-phase records are written from the allocation the conductor
    actually shipped, so they are the authority on how many cuts survived the
    A/B; a top-level count written before that A/B ran is the number PROPOSED.
    Deriving one from the other here means a table rebuilt with `--from-json`
    from an older run still says the same thing as a fresh one.
    """
    for r in rows:
        if r.get("error") or "phases" not in r:
            continue
        kept = int(sum(int(p.get("splits", 0)) for p in r["phases"]))
        proposed = int(r.get("splits_proposed", r.get("splits", kept)))
        r["splits"], r["splits_proposed"] = kept, proposed
        r["splits_reverted"] = max(proposed - kept, 0)


def table(rows):
    """The rows as a markdown table. -> list of lines."""
    out = ["| " + " | ".join(HEAD) + " |",
           "|" + "|".join("---" for _ in HEAD) + "|"]
    for r in rows:
        if r.get("error"):
            out.append(f"| {r['name']} | {r['regime']} | {r['traced_m']:.1f} | "
                       f"{100 * r['coverage']:.2f} | REFUSED | — | — | "
                       f"{r['splits']} | — | {r['wall_s']:.0f} s |")
            continue
        sp = str(r["splits"])
        rev = int(r.get("splits_reverted", 0))
        if rev:
            sp += f" (+{rev} reverted)"
        out.append(
            f"| {r['name']} | {r['regime']} | {r['traced_m']:.1f} | "
            f"{100 * r['coverage']:.2f} | **{r['makespan_s']:.1f} s** | "
            f"{r['floor_s']:.1f} s | {r['efficiency']:.2f} | {sp} | "
            f"{100 * r['solo_share']:.0f} | {r['wall_s']:.0f} s |")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="one drawing by name")
    ap.add_argument("--no-split", action="store_true",
                    help="allocation v1: balance with whole-segment moves only")
    ap.add_argument("--budget", type=float, default=None,
                    help="seconds of local search per arm above 16 segments "
                         "(sequence.TIME_BUDGET); lower it to bound the wall "
                         "clock on the dense drawings")
    ap.add_argument("--out", default=str(ROOT / "out/bench.json"))
    ap.add_argument("--doc", default=str(ROOT / "docs/BENCH.md"))
    ap.add_argument("--no-doc", action="store_true")
    ap.add_argument("--from-json", default=None,
                    help="re-render docs/BENCH.md from a finished out/bench.json "
                         "instead of running anything; the numbers are the same "
                         "measurement, only the prose around them is rebuilt")
    ap.add_argument("--verbose", action="store_true")
    a0 = ap.parse_args(argv)
    if a0.from_json:
        doc = json.loads(Path(a0.from_json).read_text())
        _reconcile(doc["rows"])
        print("\n".join(table(doc["rows"])))
        write_doc(Path(a0.doc), doc["rows"], float(doc.get("wall_s", 0.0)),
                  split=bool(doc.get("split", True)))
        print(f"\nrewrote {a0.doc} from {a0.from_json}")
        return doc["rows"]
    names = [a0.only] if a0.only else list(bench.ORDER)
    seq = {} if a0.budget is None else dict(budget=float(a0.budget))

    rows, t0 = [], time.time()
    for name in names:
        print(f"\n{'=' * 74}\n=== {name}\n{'=' * 74}")
        r, _, _ = run_one(name, Args(), split=not a0.no_split,
                          verbose=a0.verbose, seq_opts=seq)
        rows.append(r)
        if r.get("error"):
            print(f"  !! {name}: {r['error']}")
        else:
            print(f"  {name}: coverage {100 * r['coverage']:.2f} %, makespan "
                  f"{r['makespan_s']:.1f} s against a floor of {r['floor_s']:.1f} s "
                  f"(efficiency {r['efficiency']:.2f}), {r['splits']} splits, "
                  f"solo {100 * r['solo_share']:.0f} %, "
                  f"clearance {1000 * r['min_clearance']:.1f} mm, "
                  f"{r['wall_s']:.0f} s wall")
    wall = time.time() - t0

    print("\n" + "\n".join(table(rows)))
    doc = dict(generated=time.strftime("%Y-%m-%d"), sheet=list(SHEET),
               pens_mm={str(k): round(1000 * v, 1) for k, v in sorted(PENS.items())},
               split=not a0.no_split, wall_s=wall, rows=rows)
    Path(a0.out).write_text(json.dumps(doc, indent=1))
    print(f"\nwrote {a0.out}  ({wall:.0f} s for {len(rows)} drawings)")
    if not a0.no_doc and not a0.only:
        write_doc(Path(a0.doc), rows, wall, split=not a0.no_split)
        print(f"wrote {a0.doc}")
    return rows


PREAMBLE = """# The bench — five drawings that are not CSAIL

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
"""


def write_doc(path, rows, wall, split=True):
    L = [PREAMBLE, "",
         f"## Results ({time.strftime('%Y-%m-%d')}, allocation "
         f"v{'2 (splitting)' if split else '1 (whole segments)'}, "
         f"{wall:.0f} s for all five)", ""]
    L += table(rows)
    L += ["",
          "`floor` is the busiest arm's own nominal programme summed over the "
          "phases — its ink plus its pen-ups, every other arm assumed free — "
          "and no schedule can beat it. It is what the ALLOCATOR moves. "
          "`eff` = floor / makespan is what the CONDUCTOR moves: 1.00 means "
          "nobody ever waited. `solo %` is the share of the run with exactly "
          "one pen on the paper, the quantity `docs/SOLO_TIME.md` diagnosed and "
          "the one stroke splitting exists to lower. `splits` counts the cuts "
          "the conductor KEPT; a bracketed number is cuts it handed back, "
          "because `csail_schedule.build_phases` conducts the unsplit "
          "allocation as well and ships the faster of the two.", "",
          "**What splitting is worth before the conductor sees it**, measured "
          "on the same five drawings at the allocation stage alone (the busiest "
          "arm's nominal programme, v1 whole-segment moves only against v2 with "
          "cutting). Coverage is identical to the digit in every row, which is "
          "the invariant stated as a measurement rather than as an argument:",
          "",
          "| drawing | coverage | floor v1 | floor v2 | change | cuts |",
          "|---|---|---|---|---|---|",
          "| hatch | 73.55 % | 351.5 s | 335.7 s | −4.5 % | 9 |",
          "| scatter | 84.72 % | 40.4 s | 37.8 s | −6.4 % | 1 |",
          "| starburst | 87.70 % | 160.8 s | 150.2 s | −6.6 % | 2 |",
          "| spiral | 85.57 % | 79.7 s | **56.7 s** | **−28.9 %** | 9 |",
          "| duotone | 82.35 % | 282.6 s | 251.1 s | −11.1 % | 5 |",
          "",
          "The spiral is the one that matters most, because it is ONE stroke: "
          "there is no whole-segment move to make, so every second of that "
          "28.9 % is the cutting move's and nothing else's.", ""]
    ok = [r for r in rows if not r.get("error")]
    if ok:
        L += ["| drawing | segments | phases | pause | clearance | scene_check |",
              "|---|---|---|---|---|---|"]
        for r in ok:
            L.append(f"| {r['name']} | {r['n_segments']} | {r['n_phases']} | "
                     f"{r['pause_total']:.1f} s | "
                     f"{1000 * r['min_clearance']:.1f} mm | "
                     f"{'PASS' if r['scene_check'] else 'FAIL'} |")
        L.append("")
    for r in rows:
        if r.get("error"):
            L.append(f"**{r['name']} was refused:** {r['error']}")
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
