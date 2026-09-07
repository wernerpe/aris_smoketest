#!/usr/bin/env python3
"""WHERE A GUI JOB'S TIME WENT, from its own append-only event log.

    python3 scripts/job_substages.py out/gui_jobs/<job_id>/events.jsonl
    python3 scripts/job_substages.py out/gui_jobs/<job_id>          # same

NOTHING IS RE-DERIVED AND NOTHING IS SUMMED THAT THE PLANNER DID NOT TIME.
`aris_sixarm.progress` brackets every stage and substage and writes one JSON
line per event; this reads those lines back and adds up `elapsed_s` per `sub`.
That is why the totals can EXCEED the wall clock — the allocator forks a pool
and six workers' substages run at once — and why the substage total is under
the stage total: whatever no substage bracketed is un-instrumented time, and
naming it is the point of the exercise.

WHAT IT IS FOR.  The published planner-cost tables in `docs/DECISIONS.md` are
this script's output (v15: balance 3 136.8 s, 65.5 % of 4 791.5 s; v16:
flycheck 90 368.3 s, 57.7 % of 156 525.5 s), and a number a document quotes
should come from something the repository can run again.
"""
import argparse
import collections
import json
import sys
from pathlib import Path


def read(path):
    """-> (per-substage totals, counts, per-stage totals, counts, job_end)."""
    p = Path(path)
    if p.is_dir():
        p = p / "events.jsonl"
    tot, n = collections.Counter(), collections.Counter()
    stage, stage_n = collections.Counter(), collections.Counter()
    job = {}
    with open(p) as f:
        for line in f:
            try:
                ev = json.loads(line)
            except Exception:
                continue                    # a corrupt line is not a stream
            kind = ev.get("kind")
            pl = ev.get("payload") or {}
            if kind == "substage_end":
                nm = pl.get("sub", "?")
                tot[nm] += float(pl.get("elapsed_s") or 0.0)
                n[nm] += 1
            elif kind == "stage_end":
                st = ev.get("stage") or "?"
                stage[st] += float(pl.get("elapsed_s") or 0.0)
                stage_n[st] += 1
            elif kind == "job_end":
                job = pl
    return tot, n, stage, stage_n, job


def report(tot, n, stage, stage_n, job):
    T = sum(tot.values())
    out = [f"{'substage':<12} {'total':>11} {'share':>8} {'calls':>6}"]
    for nm, v in tot.most_common():
        out.append(f"{nm:<12} {v:10.1f}s {100 * v / T if T else 0:7.1f} % "
                   f"{n[nm]:5d}")
    out.append(f"{'TOTAL':<12} {T:10.1f}s")
    out.append("")
    for st, v in stage.most_common():
        out.append(f"stage {st:<14} {v:10.1f}s  x{stage_n[st]}")
    if job:
        out.append("")
        out.append(f"job elapsed_s {job.get('elapsed_s')}  ok={job.get('ok')}"
                   + (f"  error={job['error']}" if job.get("error") else ""))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("events", help="events.jsonl, or the job directory")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    tot, n, stage, stage_n, job = read(a.events)
    for line in report(tot, n, stage, stage_n, job):
        print(line)
    if a.json:
        Path(a.json).write_text(json.dumps(
            dict(substage_s=dict(tot), substage_calls=dict(n),
                 stage_s=dict(stage), stage_calls=dict(stage_n),
                 total_substage_s=sum(tot.values()), job_end=job), indent=1))
        print("wrote", a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
