"""The leader/follower sweep, as the two tables docs/V2_STAGED.md section 22 quotes.

Reads the `--json` summaries `scripts/lf_sweep.sh` writes and prints (a) one
per-stage table per run and (b) the cross-run comparison the split is chosen on.
Nothing is computed here that `staged.summary` did not already measure; this is
formatting, so that the document and the file on disk cannot disagree.

    .venv/bin/python scripts/lf_report.py out/staged_csail_h097_lf_*.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

V19_MAKESPAN = 209.9        # docs/V2_STAGED.md section 19, the conducted six-arm
V6_MAKESPAN = 432.6         # section 21, the eight-stage zigzag
LOGO_INK = 16.805           # metres of CSAIL at v19's placement


def fmt(x, n=1, plus=False):
    if x is None:
        return "—"
    return f"{x:+.{n}f}" if plus else f"{x:.{n}f}"


def pct(x, n=1):
    return "—" if x is None else f"{100 * float(x):.{n}f} %"


def stage_table(d):
    out = ["| stage | roles | pieces | ink (m) | stage (s) | active-pair (mm) | "
           "solo (mm) | conducted (mm) | deferred (m) | flown | verdict |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in d["stages"]:
        if r.get("conducted"):
            who = "CONDUCTOR (6)"
        else:
            lead = [a for a, v in sorted(r.get("roles", {}).items())
                    if v == "leader"]
            foll = [a for a, v in sorted(r.get("roles", {}).items())
                    if v == "follower"]
            who = f"lead {','.join(lead)} / foll {','.join(foll)}"
        out.append(
            f"| {r['stage']} | {who} | {r['pieces']} | {r['ink_m']:.3f} | "
            f"{r['duration_s']:.1f} | {fmt(r['active_pair_mm'], 1, True)} | "
            f"{fmt(r['solo_min_mm'], 1, True)} | "
            f"{fmt(r.get('conducted_min_mm'), 1, True)} | "
            f"{fmt(r.get('deferred_m'), 3)} | "
            f"{r['buckets_flown']}/{r['buckets_with_ink']} | "
            f"{'PASS' if r['ok'] else 'FAIL'} |")
    return out


def row(d):
    ro = d.get("roles", {})
    t = ro.get("total", {})
    so = d.get("stage_overhead", {})
    fic = ro.get("follower_ink_clearance") or {}
    return dict(
        pattern=d["pattern"],
        makespan=d["makespan_s"],
        vs_v19=d["makespan_s"] / V19_MAKESPAN,
        vs_v6=d["makespan_s"] / V6_MAKESPAN,
        follower_offered=t.get("follower_m"),
        follower_flown=t.get("follower_flown_m"),
        fit=t.get("follower_fit_frac"),
        deferred=t.get("deferred_m"),
        conducted_m=t.get("conducted_m"),
        conducted_s=t.get("conducted_s"),
        coverage=d.get("coverage"),
        drawn_m=d.get("drawn_m"),
        park_crit=so.get("park_s_on_the_critical_path"),
        park_frac=so.get("park_frac_of_makespan"),
        ttfm=d.get("ttfm_s"),
        plan_s=d.get("plan_s"),
        par_plan_s=d.get("parallel_plan_s"),
        check_s=d.get("check_s"),
        holds_ok=d.get("holds_ok"),
        ink_med=fic.get("median_mm"), ink_p05=fic.get("p05_mm"),
        ink_min=fic.get("min_mm"), ink_n=fic.get("n"),
        ink_under=fic.get("under_gate"),
        all_ok=d.get("all_ok"))


def main(argv):
    paths = [Path(p) for p in argv[1:]]
    rows = []
    for p in paths:
        if not p.exists():
            print(f"(missing: {p})")
            continue
        d = json.loads(p.read_text())
        print(f"\n### {p.name} — {d['pattern']}\n")
        print("\n".join(stage_table(d)))
        rows.append((p.name, row(d)))
    if not rows:
        return 1
    print("\n\n### the sweep\n")
    hdr = ["split", "makespan (s)", "× v19", "× v6", "follower offered (m)",
           "follower flown (m)", "**fit**", "deferred (m)", "conducted (m)",
           "conducted (s)", "coverage", "park crit (s)", "park %", "ttfm (s)",
           "plan/arm (s)", "check (s)", "holds ok", "all ok"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for name, r in rows:
        tag = name.replace("staged_csail_h097_lf_", "").replace(".json", "")
        cells = [tag, fmt(r["makespan"]), fmt(r["vs_v19"], 2),
                 fmt(r["vs_v6"], 2), fmt(r["follower_offered"], 3),
                 fmt(r["follower_flown"], 3), pct(r["fit"]),
                 fmt(r["deferred"], 3), fmt(r["conducted_m"], 3),
                 fmt(r["conducted_s"]), pct(r["coverage"]),
                 fmt(r["park_crit"]), pct(r["park_frac"]),
                 fmt(r["ttfm"], 3), fmt(r["par_plan_s"]), fmt(r["check_s"]),
                 str(r["holds_ok"]), str(r["all_ok"])]
        print("| " + " | ".join(cells) + " |")
    print("\n\n### the follower's ink, against the room it had to fit\n")
    print("| split | pieces measured | min (mm) | p05 (mm) | median (mm) | "
          "under the 50 mm gate |")
    print("|---|---|---|---|---|---|")
    for name, r in rows:
        tag = name.replace("staged_csail_h097_lf_", "").replace(".json", "")
        print(f"| {tag} | {r['ink_n']} | {fmt(r['ink_min'], 1, True)} | "
              f"{fmt(r['ink_p05'], 1, True)} | {fmt(r['ink_med'], 1, True)} | "
              f"{r['ink_under']} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
