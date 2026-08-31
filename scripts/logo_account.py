#!/usr/bin/env python3
"""TRACED = DRAWN + DROPPED, AND THE SEAMS ARE ONLY DRAWN ONCE.

Every A/B in this campaign is decided on "how much of the artwork does a run
actually put on the paper", and until now that number was computed by a script
in `out/` — which is gitignored, so the arithmetic behind a verdict could not
be re-run from a checkout.  It lives here instead.

`sequence.summary_json` reports `coverage = drawn_m / traced_m`, and `drawn_m`
is the sum of SEGMENT LENGTHS.  That counts a handoff seam twice (`OVERLAP_M`,
4 mm each side of every cut between two arms) and a split splice twice
(`SPLIT_OVERLAP_M`, 5 mm).  The ink is deliberately laid twice so the pens
meet, and counting it twice makes the ratio flatter than the sheet.  The UNION
figure is the one a photograph of the paper would agree with:

    empty  =  what the allocator left as holes  +  what no phase conducted
    drawn  =  traced - empty

Both are printed.  The reported figure is kept because it is the number the
run itself prints and the one earlier notes quote; the union figure is the one
to compare runs on, and the two move together.

Run:  python3 scripts/logo_account.py out/csail_schedule_h094_v11.json [more...]

With two or more runs it also prints an A/B table, first argument as baseline.
The programme JSON is found by name (`csail_schedule_X` -> `csail_program_X`)
and is what carries the allocator's hole list; without it only the skipped
phases are counted as empty, which UNDERSTATES the empty paper, so the script
says so rather than quietly reporting a better number.
"""
import json
import sys
from pathlib import Path


def account(sched_path):
    """The union coverage of one conducted run. -> dict."""
    sched_path = Path(sched_path)
    sch = json.loads(sched_path.read_text())
    prog_path = sched_path.with_name(
        sched_path.name.replace("csail_schedule_", "csail_program_"))
    prog = json.loads(prog_path.read_text()) if prog_path.exists() else None

    traced = float(sch["traced_m"])
    seg_sum = float(sch["drawn_m"])
    skipped = float(sch.get("skipped_m", 0.0))
    holes = []
    if prog:
        for ph in prog["phases"]:
            holes += [dict(h, phase=ph["name"]) for h in ph.get("dropped", [])]
        # `allocate_all` moves the FINAL hole list onto phase 0 and empties the
        # rest, so a non-empty list on a later phase is a stale intermediate
        if len(prog["phases"]) > 1:
            holes = [h for h in holes if h["phase"] == prog["phases"][0]["name"]]
    hole_m = sum(float(h["length_m"]) for h in holes)
    empty = hole_m + skipped
    return dict(name=sched_path.name, traced=traced, seg_sum=seg_sum,
                skipped=skipped, hole_m=hole_m, holes=holes, empty=empty,
                union=(traced - empty) / traced, reported=float(sch["coverage"]),
                makespan=float(sch["makespan_s"]),
                pause=float(sch["pause_total"]),
                clr=float(sch["min_clearance"]), segs=int(sch["n_segments"]),
                phases=int(sch["n_conducted"]),
                skipped_phases=list(sch.get("skipped_phases", [])),
                have_prog=prog is not None)


def show(g):
    print(f"\n=== {g['name']} ===")
    if not g["have_prog"]:
        print("  (no programme JSON beside it: the allocator's holes are NOT "
              "counted, so EMPTY PAPER below is a LOWER bound)")
    print(f"  traced                         {g['traced']:9.4f} m")
    print(f"  segment lengths (as reported)  {g['seg_sum']:9.4f} m   "
          f"-> coverage {100 * g['reported']:.4f} %   (seams counted twice)")
    print(f"  holes the allocator left       {g['hole_m']:9.4f} m   "
          f"({len(g['holes'])} span(s))")
    print(f"  ink no phase conducted         {g['skipped']:9.4f} m   "
          f"({len(g['skipped_phases'])} phase(s): "
          f"{', '.join(g['skipped_phases']) or '-'})")
    print(f"  EMPTY PAPER                    {g['empty']:9.4f} m")
    print(f"  UNION COVERAGE                 {100 * g['union']:9.4f} %")
    for h in sorted(g["holes"], key=lambda h: -float(h["length_m"])):
        print(f"      {1000 * float(h['length_m']):7.2f} mm  {h['color']:>6}  "
              f"stroke {h['stroke_id']:>3} s {h['s_range']}  at "
              f"({h['at'][0]:.3f}, {h['at'][1]:.3f})")
    print(f"  makespan {g['makespan']:.2f} s   pause {g['pause']:.2f} s   "
          f"min clearance {1000 * g['clr']:.1f} mm   "
          f"{g['phases']} phase(s) conducted, {g['segs']} segments")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    got = [account(p) for p in sys.argv[1:]]
    for g in got:
        show(g)
    if len(got) > 1:
        base = got[0]
        print(f"\n=== A/B against {base['name']} ===")
        print(f"  {'run':<40} {'union %':>9} {'delta':>8} {'reported %':>11} "
              f"{'empty m':>9} {'makespan':>9}")
        for g in got:
            print(f"  {g['name']:<40} {100 * g['union']:9.4f} "
                  f"{100 * (g['union'] - base['union']):+8.4f} "
                  f"{100 * g['reported']:11.4f} {g['empty']:9.4f} "
                  f"{g['makespan']:9.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
