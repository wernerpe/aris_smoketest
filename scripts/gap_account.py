#!/usr/bin/env python
"""WHERE THE INK IS NOT, AND WHY -- the coverage account of a staged programme.

    ARIS_RIG=proposed ARIS_TOOL=lateral .venv/bin/python -m scripts.gap_account \
        --programme out/staged_csail_h097_lf6b_s150_program.json \
        --lines out/csail_schedule_h097_v19_strokes.json \
        --png out/lf6b_gaps.png --json out/lf6b_gaps.json

THE MEASUREMENT IS GEOMETRIC, NOT BOOK-KEEPING.  A piece that is LISTED in an
arm's programme is not ink: `staged._merge_conducts` pads an arm whose bucket
never flew to the stage's frame count, so the arm stands in the timeline with
its pieces still on its list and no pen-down sample anywhere.  So what counts as
drawn is a PEN-DOWN SAMPLE -- a frame whose `seg` is a real segment index -- and
the residual is what is left of the logo when every drawn piece's own polyline
is subtracted from it, at `TOL` metres.  That is the picture the animation
shows and it is the only number a gap argument may be made on.

THE REASON CHAIN IS RE-DERIVED, NOT GUESSED.  The DP and the refusal loop are
cheap (no legs, no routing, ~15 s on the CSAIL logo), so they are re-run here
against the same atlas and pattern the programme was built with, and every
missing stretch is asked:

  (a) which (stage, arm) cells the CAPABILITY MAP offers it to at all -- one
      cell is a single point of failure and the pattern's own doing;
  (b) which cells survive the refusal loop's bans, and what each ban was
      (`plan_stroke`'s own status and reason, and the `s_star` it certified);
  (c) which (stage, arm) the DP finally gave it to;
  (d) whether that (stage, arm) LISTED it and did not fly it, refused it, or
      never saw it.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from aris_sixarm import staged as S
from aris_sixarm import traces as T

# THE MEASUREMENT ITSELF LIVES IN `staged`, because `staged.run` ASSERTS it.
# Two implementations of "what is on the paper" is exactly one too many: this
# script is the programme-file front end and the reason chain, and the geometry
# is the module's.
DS = S.INK_DS           # m, the grid the logo is tested on
TOL = S.INK_TOL         # m, how near a drawn polyline a point must be to be INK
MIN_GAP = S.INK_MIN_GAP  # m, below which a gap is sampling dust
residual = S.residual_of


# ---------------------------------------------------------------------------
# 1.  WHAT IS ON THE PAPER
# ---------------------------------------------------------------------------
def drawn_pieces(doc: dict) -> tuple[dict, list]:
    """Every piece with PEN-DOWN SAMPLES, and the whole listed inventory.

    -> ({line: [pts]}, [piece records]).  `staged.flown_ink` is the same
    question asked of a live `StagedResult`; this asks it of a programme JSON,
    which is what a run that has already shipped leaves behind.
    """
    ink: dict[int, list] = {}
    listed: list[dict] = []
    for one in doc.get("stages", []):
        s = int(one["stage"])
        for a, arm in (one.get("arms") or {}).items():
            tl = arm.get("trajectory")
            seen = ({int(x) for x in tl["seg"] if int(x) >= 0}
                    if tl is not None else set())
            for pc in arm.get("pieces") or []:
                ok = int(pc["order"]) in seen
                listed.append(dict(stage=s, arm=int(a), line=int(pc["line"]),
                                   k=int(pc["piece"]), order=int(pc["order"]),
                                   m=float(pc["length_m"]), flown=bool(ok)))
                if ok:
                    ink.setdefault(int(pc["line"]), []).append(
                        np.asarray(pc["pts"], float).reshape(-1, 2))
    return ink, listed


# ---------------------------------------------------------------------------
# 2.  WHY IT IS NOT THERE
# ---------------------------------------------------------------------------
def rederive_dp(lines, atlas=S.ATLAS_DEFAULT, split_m=0.15, tilt_max_deg=0.0,
                verbose=True):
    """The DP and the refusal loop, again, with every refusal recorded.

    -> (plan, cap, raw_cap, rounds).  `plan_bucket` runs with `fly=False`, so
    this is `plan_stroke` and the DP and nothing else: no legs, no ordering, no
    timeline.  It is the same loop `staged.resolve_refusals` runs and it is
    re-run rather than read back because the masks it builds are the answer to
    "who was allowed to draw this after the refusals" and the programme JSON
    does not carry them.
    """
    cov = T.coverage_from_atlas(atlas, arms=tuple(sorted(S.FLEET)),
                                gate="strict", tilt_max_deg=tilt_max_deg)
    pat = T.leader_follower_pattern(split_m=float(split_m))
    cap = T.capability(cov, pat)
    fl = S.FLEET
    pens = {a: fl[a].pen for a in fl}
    parks = S.shipped_parks(fl)
    S.plan_memo_clear()
    S.split_ids_reset()
    masks: dict[int, list] = {}
    seen: set = set()
    rounds: list[dict] = []
    plan = None
    for rnd in range(S.REFUSAL_ROUNDS + 1):
        plan = S.plan_lines_masked(lines, cap, None, masks)
        buckets = S.bucket(S.pieces_of(plan))
        new, hits = 0, []
        for (st_, a_) in sorted(buckets):
            stg = S.plan_bucket(st_, a_, buckets[(st_, a_)], fl, pens, parks,
                                S.H_INV_DEFAULT, dict(tilt_max_deg=tilt_max_deg),
                                leg_cache=True, fly=False, verbose=False)
            for pp in stg.refused:
                pc = pp.piece
                span = S.ban_span(pp)
                rec = dict(round=int(rnd), stage=int(st_), arm=int(a_),
                           line=int(pc.line), k=int(pc.k), s0=float(pc.s0),
                           s1=float(pc.s1), m=float(pc.length_m),
                           status=str(pp.status), reason=str(pp.reason),
                           s_star=float(pp.s_star), banned=None)
                if pc.state >= 0 and span is not None:
                    b = (pc.state, round(float(span[0]), 6),
                         round(float(span[1]), 6))
                    if b not in seen:
                        seen.add(b)
                        masks.setdefault(pc.line, []).append(b)
                        rec["banned"] = [str(cap.label(pc.state)),
                                         float(span[0]), float(span[1])]
                        new += 1
                hits.append(rec)
        d = plan.summary()
        rounds.append(dict(round=int(rnd), new_bans=int(new), refusals=hits,
                           covered_frac=float(d["covered_frac"]),
                           uncovered_m=float(d["uncovered_m"])))
        if verbose:
            print(f"  DP round {rnd}: {d['n_pieces']} pieces, {new} new bans, "
                  f"coverage {100 * d['covered_frac']:.2f} %, uncovered "
                  f"{d['uncovered_m']:.4f} m")
        if not new:
            break
    return plan, cap, cov, rounds


def _bits_at(atoms, states, s):
    for at in atoms:
        if at.s0 - 1e-9 <= s <= at.s1 + 1e-9:
            return sorted({(states[k].stage, states[k].arm)
                           for k in range(len(states)) if at.bits >> k & 1})
    return []


def _overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def v19_cover(doc19: dict, lines) -> dict:
    """What v19 drew, per line. -> {line: [(s0, s1, arm, phase, seg, tilt)]}.

    THE INTERVAL COMES FROM `s_range`, NOT FROM A PROJECTION.  A v19 segment
    records the normalised span of its stroke it draws; multiplied by the
    line's own arc length that reproduces the segment's stored `length_m` to
    1e-5 m, while projecting the segment's endpoints onto the polyline
    mis-branches by up to a metre on the logo's self-approaching lines.
    """
    out: dict[int, list] = defaultdict(list)
    for ph, phase in enumerate(doc19.get("phases", [])):
        for a, segs in (phase.get("arms") or {}).items():
            for sg in segs:
                li = int(sg["stroke_id"])
                if li >= len(lines):
                    continue
                L = float(T.cumlen(lines[li])[-1])
                r = sorted(float(x) for x in sg["s_range"])
                out[li].append((r[0] * L, r[1] * L, int(a), int(ph),
                                int(sg["seg"]), float(sg.get("tilt_cone_deg",
                                                             0.0))))
    return out


REASONS = {
    "listed_not_flown": "PLANNED AND NEVER FLOWN -- the arm's bucket produced "
                        "no timeline; its pieces stay on the list with no "
                        "pen-down sample",
    "no_drawer": "NO DRAWER LEFT -- the only (stage, arm) cell the pattern "
                 "offers it to was banned by the refusal loop",
    "plan_refused": "plan_stroke REFUSED -- a stub under its minimum length",
    "deferred_never_taken": "DEFERRED AND NEVER RE-LISTED -- an arm handed it "
                            "on and no later stage put it on a bucket",
    "bucket_never_planned": "THE BUCKET WAS NEVER PLANNED -- the DP gave the "
                            "span to a (stage, arm) cell that listed nothing, "
                            "refused nothing and deferred nothing, so the ink "
                            "is in the DP and in no stage's book at all",
    "unattributed": "not attributed",
}


def account(lines, doc, plan, cap, cov, rounds, gaps, v19=None) -> list[dict]:
    """One reason chain per missing stretch. -> [dict]."""
    _, listed = drawn_pieces(doc)
    nodrawer: dict[int, list] = defaultdict(list)
    dp_at: dict[int, list] = defaultdict(list)
    for lp in plan.lines:
        for (k, i0, i1) in T.runs_of(lp.atoms, lp.assign):
            a0, a1 = float(lp.atoms[i0].s0), float(lp.atoms[i1 - 1].s1)
            if k == T.UNCOVERED:
                nodrawer[lp.index].append((a0, a1))
            else:
                dp_at[lp.index].append((a0, a1, int(cap.states[k].stage),
                                        int(cap.states[k].arm)))
    # THE RAW ATOMS ARE THE CAPABILITY MAP BEFORE ANY BAN.  `plan.lines[li]`
    # carries the MASKED ones -- `plan_lines_masked` cuts the atoms where the
    # refusal loop struck a state out -- so the two together say which cells
    # the pattern ever offered a stretch to and which survived.
    raw = {li: T.atoms_of(l, cap, T.SAMPLE_DS, T.REFINE_TOL)[0]
           for li, l in enumerate(lines)}
    hits = [h for r in rounds for h in r["refusals"]]
    # ...AND THE REFUSALS THE PROGRAMME ITSELF RECORDS.  A `split_at_room` part
    # is born inside a stage and the DP has never heard of it, so a stub the
    # stage refused (`degenerate:too_short` on a 5 mm remainder) is in the
    # programme's own per-arm `refused` list and nowhere else.
    shipped: dict[int, list] = defaultdict(list)
    handed_on: dict[int, list] = defaultdict(list)
    for one in doc.get("stages", []):
        for a, arm in (one.get("arms") or {}).items():
            for r in arm.get("refused") or []:
                shipped[int(r["line"])].append(
                    dict(stage=int(one["stage"]), arm=int(a), k=int(r["piece"]),
                         m=float(r["length_m"]), status=str(r["status"]),
                         reason=str(r["reason"])))
            for r in arm.get("deferred") or []:
                handed_on[int(r["line"])].append(
                    dict(stage=int(one["stage"]), arm=int(a), k=int(r["piece"]),
                         m=float(r["length_m"])))
    # A DP PIECE NO STAGE EVER OPENED ITS BOOK ON -- see
    # `staged.coverage_account`, which asks the same question of a live result.
    # `plan_bucket` records every piece of a bucket it is handed, so a piece in
    # none of the programme's three lists was never offered to it at all: its
    # whole bucket was skipped, which is what a conducted group that produces
    # no arms does.  Matched on (line, k) because a deferral changes the cell.
    drew = {(int(p["line"]), int(p["k"])) for p in listed}
    drew |= {(int(li), int(r["k"])) for li, v in shipped.items() for r in v}
    drew |= {(int(li), int(r["k"])) for li, v in handed_on.items() for r in v}
    unplanned: dict[int, list] = defaultdict(list)
    for pc in S.pieces_of(plan):
        if (int(pc.line), int(pc.k)) in drew:
            continue
        unplanned[int(pc.line)].append(
            dict(stage=int(pc.stage), arm=int(pc.arm), line=int(pc.line),
                 k=int(pc.k), s0=float(pc.s0), s1=float(pc.s1),
                 m=float(pc.length_m)))
    out = []
    for (li, a, b, m) in gaps:
        cum = T.cumlen(lines[li])
        mid = 0.5 * (a + b)
        xy = T.points_at(lines[li], cum, np.array([a, mid, b]))
        cells = _bits_at(raw[li], cap.states, mid)
        g = dict(line=int(li), s0=float(a), s1=float(b), m=float(m),
                 xy0=[round(float(x), 4) for x in xy[0]],
                 xy_mid=[round(float(x), 4) for x in xy[1]],
                 xy1=[round(float(x), 4) for x in xy[2]],
                 offered_to=cells,
                 offered_after_bans=_bits_at(plan.lines[li].atoms, cap.states,
                                             mid),
                 dp_gave_it_to=sorted({(p[2], p[3]) for p in dp_at.get(li, [])
                                       if _overlap(a, b, p[0], p[1]) > 1e-4}),
                 no_drawer_m=float(sum(_overlap(a, b, *iv)
                                       for iv in nodrawer.get(li, []))),
                 refusals=[h for h in hits if h["line"] == li
                           and _overlap(a, b, h["s0"], h["s1"]) > 1e-4],
                 not_flown=[p for p in listed if p["line"] == li
                            and not p["flown"]
                            and (p["stage"], p["arm"]) in
                            {(q[2], q[3]) for q in dp_at.get(li, [])
                             if _overlap(a, b, q[0], q[1]) > 1e-4}])
        g["stage_refused"] = [r for r in shipped.get(li, [])
                              if r["m"] + 2 * TOL >= m]
        g["handed_on"] = [r for r in handed_on.get(li, [])
                          if (r["stage"], r["arm"]) in
                          {(q[2], q[3]) for q in dp_at.get(li, [])
                           if _overlap(a, b, q[0], q[1]) > 1e-4}]
        g["never_planned"] = [u for u in unplanned.get(li, [])
                              if _overlap(a, b, u["s0"], u["s1"]) > 1e-4]
        if g["not_flown"]:
            g["reason"] = "listed_not_flown"
        elif g["no_drawer_m"] > 0.5 * m:
            g["reason"] = "no_drawer"
        elif any(h["status"] == "degenerate" for h in g["refusals"]) \
                or any(r["status"] == "degenerate" for r in g["stage_refused"]):
            g["reason"] = "plan_refused"
        elif g["handed_on"]:
            g["reason"] = "deferred_never_taken"
        elif g["never_planned"]:
            g["reason"] = "bucket_never_planned"
        else:
            g["reason"] = "unattributed"
        # ...AND WHO DREW IT IN v19, WHICH REACHED 100.0000 % ON THIS EXACT
        # PLACEMENT.  Every one of these stretches IS drawable at h = 0.970;
        # the contrast says with which arm, at what reach and under how much
        # tilt cone -- and the tilt cone is the one the staged pipeline runs
        # at zero.
        g["v19"] = []
        for (c0, c1, ar, ph, sg, tilt) in (v19 or {}).get(li, []):
            ov = _overlap(a, b, c0, c1)
            if ov <= 1e-4:
                continue
            xy = S.FLEET[int(ar)].xy
            g["v19"].append(dict(arm=int(ar), phase=int(ph), seg=int(sg),
                                 m=round(float(ov), 4), tilt_deg=float(tilt),
                                 reach_m=round(float(np.hypot(
                                     g["xy_mid"][0] - xy[0],
                                     g["xy_mid"][1] - xy[1])), 3)))
        out.append(g)
    return out


# ---------------------------------------------------------------------------
# 3.  THE PICTURE
# ---------------------------------------------------------------------------
COLOUR = {"listed_not_flown": "#d62728", "no_drawer": "#ff7f0e",
          "plan_refused": "#9467bd", "deferred_never_taken": "#2ca02c",
          "bucket_never_planned": "#17becf", "unattributed": "#7f7f7f"}


def picture(lines, doc, acc, path, total, inked, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, _ = drawn_pieces(doc)
    fig, (ax, tx) = plt.subplots(
        1, 2, figsize=(15.5, 12.0), gridspec_kw=dict(width_ratios=[1.0, 1.25]))
    for l in lines:
        ax.plot(l[:, 0], l[:, 1], color="#c8c8c8", lw=0.9, zorder=1)
    for v in ink.values():
        for p in v:
            ax.plot(p[:, 0], p[:, 1], color="#111111", lw=1.9, zorder=2)
    order = sorted(range(len(acc)), key=lambda i: -acc[i]["m"])
    for n, i in enumerate(order, 1):
        g = acc[i]
        cum = T.cumlen(lines[g["line"]])
        s = np.linspace(g["s0"], g["s1"], max(2, int(g["m"] / 0.002) + 2))
        P = T.points_at(lines[g["line"]], cum, s)
        ax.plot(P[:, 0], P[:, 1], color=COLOUR[g["reason"]], lw=4.2,
                solid_capstyle="round", zorder=3, alpha=0.95)
        mx, my = g["xy_mid"]
        ax.annotate(str(n), (mx, my), textcoords="offset points",
                    xytext=(9, 9), fontsize=9, fontweight="bold",
                    color=COLOUR[g["reason"]], zorder=4,
                    bbox=dict(boxstyle="circle,pad=0.16", fc="white",
                              ec=COLOUR[g["reason"]], lw=1.0))
        g["n"] = n
    for a, sp in sorted(S.FLEET.items()):
        ax.plot([sp.xy[0]], [sp.xy[1]], marker="s", ms=7, color="#2a6f97",
                zorder=5)
        ax.annotate(f"arm {a}", (sp.xy[0], sp.xy[1]),
                    textcoords="offset points", xytext=(8, -4), fontsize=8,
                    color="#2a6f97")
    ax.set_aspect("equal")
    ax.set_xlabel("paper x  [m]")
    ax.set_ylabel("paper y  [m]")
    ax.set_title(f"{title}\n{inked:.3f} m of {total:.3f} m drawn  "
                 f"= {100 * inked / total:.2f} %", fontsize=11)
    ax.grid(alpha=0.15, lw=0.5)
    from matplotlib.lines import Line2D
    by = defaultdict(float)
    for g in acc:
        by[g["reason"]] += g["m"]
    ax.legend(handles=[Line2D([], [], color="#c8c8c8", lw=1.2, label="logo"),
                       Line2D([], [], color="#111111", lw=2.0,
                              label="flown ink (pen-down samples)")]
              + [Line2D([], [], color=COLOUR[k], lw=4.0,
                        label=f"{k}  ({by[k]:.3f} m)")
                 for k in sorted(by, key=lambda k: -by[k])],
              loc="upper right", fontsize=8, framealpha=0.95)

    tx.axis("off")
    rows = ["  #   line    s0..s1  [m]      mm     reason / chain", ""]
    for n, i in enumerate(order, 1):
        g = acc[i]
        rows.append(f" {n:2d}   {g['line']:3d}   {g['s0']:.3f}..{g['s1']:.3f}"
                    f"  {1000 * g['m']:7.1f}   {g['reason']}")
        rows.append(f"        xy ({g['xy_mid'][0]:.3f}, {g['xy_mid'][1]:.3f})"
                    f"   DP offered -> {g['offered_to']}")
        rows.append(f"        after bans -> {g['offered_after_bans']}"
                    f"   DP gave it to {g['dp_gave_it_to']}")
        for h in (g["refusals"][:2]):
            rows.append(f"        plan_stroke s{h['stage']}/arm{h['arm']}: "
                        f"{h['status']}:{h['reason']} s*={h['s_star']:.3f}"
                        + (f" -> BANNED {h['banned'][0]}" if h["banned"]
                           else ""))
        for p in g["not_flown"][:2]:
            rows.append(f"        listed by s{p['stage']}/arm{p['arm']} "
                        f"(piece {p['k']}, {1000 * p['m']:.0f} mm) and never "
                        f"flown")
        for v in g["v19"][:2]:
            rows.append(f"        v19 DREW IT: arm {v['arm']} phase "
                        f"{v['phase']} seg {v['seg']}, reach {v['reach_m']} m,"
                        f" tilt cone {v['tilt_deg']:.1f} deg")
        rows.append("")
    tx.text(0.0, 1.0, "\n".join(rows), va="top", ha="left", family="monospace",
            fontsize=7.0, transform=tx.transAxes)
    fig.tight_layout()
    fig.subplots_adjust(top=0.94)
    fig.savefig(path, dpi=150)
    print("wrote", path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--programme", required=True)
    ap.add_argument("--lines", required=True)
    ap.add_argument("--split-m", type=float, default=0.15)
    ap.add_argument("--tilt-max-deg", type=float, default=0.0,
                    help="the cone the PROGRAMME was planned at; the DP is "
                         "re-derived with it, and a tilt-15 programme read at "
                         "0 invents refusals the run never had")
    ap.add_argument("--atlas", default=S.ATLAS_DEFAULT)
    ap.add_argument("--png", default=None)
    ap.add_argument("--v19", default=None,
                    help="the v19 programme JSON, for the "
                         "contrast column: which arm drew this "
                         "stretch in the run that reached 100 %%")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    lines = [np.asarray(p, float) for p in T.load_lines(a.lines)]
    doc = json.loads(Path(a.programme).read_text())
    ink, listed = drawn_pieces(doc)
    gaps, total, inked = residual(lines, ink)
    print(f"logo {total:.4f} m   flown {inked:.4f} m = "
          f"{100 * inked / total:.4f} %   missing {total - inked:.4f} m in "
          f"{len(gaps)} stretches")
    plan, cap, cov, rounds = rederive_dp(lines, a.atlas, a.split_m,
                                         tilt_max_deg=a.tilt_max_deg)
    v19 = (v19_cover(json.loads(Path(a.v19).read_text()), lines)
           if a.v19 else None)
    acc = account(lines, doc, plan, cap, cov, rounds, gaps, v19)
    by = defaultdict(lambda: [0.0, 0])
    for g in acc:
        by[g["reason"]][0] += g["m"]
        by[g["reason"]][1] += 1
    print("\nmetres missing, by reason:")
    for k, (m, n) in sorted(by.items(), key=lambda kv: -kv[1][0]):
        print(f"   {m:7.4f} m  {n:2d} stretch(es)  {k}: {REASONS[k]}")
    out = dict(programme=str(a.programme), lines=str(a.lines),
               total_m=float(total), flown_m=float(inked),
               flown_frac=float(inked / total),
               gaps_m=float(total - inked), n_gaps=len(gaps),
               by_reason={k: dict(m=float(v[0]), n=int(v[1])) for k, v in
                          by.items()},
               gap_list=acc,
               listed_not_flown=[p for p in listed if not p["flown"]])
    if a.png:
        picture(lines, doc, acc, a.png, total, inked,
                title=Path(a.programme).name)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=float))
        print("wrote", a.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
