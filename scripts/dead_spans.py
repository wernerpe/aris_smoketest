#!/usr/bin/env python3
"""Are the CSAIL drawing's 13 dead spans unreachable, and what would recover them?

    python3 scripts/dead_spans.py [--out out] [--no-probe]

The six-arm allocation left 2.22 m of the traced logo undrawn (13 spans, 13.8 %
of what was traced).  This asks the next question: is that geometry *out of
reach*, or merely out of reach for the pen the fleet is holding, held the way
the planner insists on holding it?  Two independent verdicts, because neither
dominates the other:

ATLAS VERDICT (conservative, pointwise).  Each span is resampled every 5 mm and
every sample tested against the strict-GO set (margin >= 0.30, sigma >= 0.14) of
each arm, under each condition swept:

    today   out_tilt0    pen 110 mm, perpendicular   <- the planner's own envelope
    tilt15  out_tilt15   pen 110 mm, tilt <= 15 deg  == out_pen110 == out
    tilt30  out_tilt30   pen 110 mm, tilt <= 30 deg
    pen200  out_pen200   pen 200 mm, tilt <= 15 deg
    tilt45  out_tilt45   pen 110 mm, tilt <= 45 deg
    pen300  out_pen300   pen 300 mm, tilt <= 15 deg
    combo   union of all of the above over all arms — a fleet with mixed pens and
            mixed tilt budgets, and then some (it lets one arm switch condition
            between adjacent samples), so it upper-bounds any real mixture.

Membership follows `fuzz_planner._go_mask`: the atlas has a 2 cm pixel and a
stroke sample only ever *rounds* to a cell, so the cell AND its eight neighbours
must be strict-GO before the sample counts as inside.  (The ALLOCATOR's
prefilter goes the other way — it dilates by PREFILTER_R = 5 cm — because it is
a cheap "worth probing?" screen, not a reachability claim.  The eroded
convention is the one a "would this actually help?" answer wants.)

PLANNER VERDICT (the real contract).  `stroke_api.plan_stroke` certifies at
margin 0.15 / sigma 0.10, not 0.30 / 0.14, so the strict atlas over-calls dead.
But it also demands a continuous certified walk of the redundancy band, which is
strictly harder than "every point is individually GO", so it under-calls elsewhere.
Each span is therefore also handed to `allocate.probe_stroke` — the allocator's
own prober, same three-probe budget — for every arm at every pen length, and the
certified fraction recorded.  This verdict knows about ink: each arm carries ONE
pen colour for the whole piece, so a span certifiable only by a wrong-ink arm is
not a reach problem at all.

Writes out/dead_spans_analysis.png and out/dead_spans_analysis.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
from matplotlib.lines import Line2D                          # noqa: E402
from matplotlib.colors import ListedColormap                 # noqa: E402

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from aris_sixarm import allocate, atlas                      # noqa: E402
from aris_sixarm.fleet import FLEET, SHEET                   # noqa: E402

# (key, atlas dir, human label, colour)
CONDITIONS = [
    ("today",  "out_tilt0",  "already GO today (110 mm pen, perpendicular)", "#1a9850"),
    ("tilt15", "out_tilt15", "tilt <= 15 deg",                               "#66bd63"),
    ("tilt30", "out_tilt30", "tilt <= 30 deg",                               "#fee08b"),
    ("pen200", "out_pen200", "200 mm pen (tilt <= 15)",                      "#fdae61"),
    ("tilt45", "out_tilt45", "tilt <= 45 deg",                               "#f46d43"),
    ("pen300", "out_pen300", "300 mm pen (tilt <= 15)",                      "#d73027"),
]
TIERS = [k for k, _, _, _ in CONDITIONS] + ["combo"]
TIER_COLOR = {k: c for k, _, _, c in CONDITIONS}
TIER_COLOR["combo"] = "#7a0177"
TIER_LABEL = {k: lab for k, _, lab, _ in CONDITIONS}
TIER_LABEL["combo"] = "a mixed-pen / mixed-tilt fleet (union of every atlas)"

# planner-verdict tiers, cheapest first
PENS = (0.110, 0.200, 0.300)
PTIERS = ["today", "repartition", "pen200", "pen300"]
PTIER_COLOR = {"today": "#1a9850", "repartition": "#4575b4",
               "pen200": "#fdae61", "pen300": "#d73027"}
PTIER_LABEL = {
    "today": "certified today, by an arm already holding the right ink",
    "repartition": "certified today by an arm holding the WRONG ink "
                   "(a pen problem, not a reach problem)",
    "pen200": "needs a 200 mm pen",
    "pen300": "needs a 300 mm pen",
}
DEAD_COLOR = "#252525"
RECOVER_FRAC = 0.95
DS = 0.005                 # m, span resampling step


# ---------------------------------------------------------------------------
# the atlas verdict
# ---------------------------------------------------------------------------
def go_cells(atlas_dir, arm_id, erode=True):
    """(grid, {(i, j)}) of strict-GO atlas cells, eroded by one cell.

    Lifted from scripts/fuzz_planner.py::_go_mask, minus its `tilt == 0` filter:
    that filter is what makes the mask "the planner's envelope", and here the
    whole question is what happens when the pen is allowed to lean.  Each atlas
    dir was swept with its own tilt cap, so the rows already respect it.
    """
    arr, meta = atlas.load(atlas_dir, arm_id)
    g = float(meta["grid"])
    sel = atlas.strict_go(arr)
    idx = {(int(round(x / g)), int(round(y / g))) for x, y in arr[sel][:, :2]}
    if not erode:
        return g, idx
    keep = {c for c in idx
            if all((c[0] + a, c[1] + b) in idx
                   for a in (-1, 0, 1) for b in (-1, 0, 1))}
    return g, keep


def densify(pts, ds=DS):
    """Resample a polyline at a fixed arc-length step, endpoints included."""
    p = np.asarray(pts, float)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    t = np.concatenate([[0.0], np.cumsum(seg)])
    if t[-1] <= 0:
        return p[:1]
    s = np.arange(0.0, t[-1], ds)
    d = np.column_stack([np.interp(s, t, p[:, 0]), np.interp(s, t, p[:, 1])])
    return np.vstack([d, p[-1]])


def covered_by(samples, grid, cells):
    """Boolean per sample: does its nearest atlas cell sit in `cells`?"""
    ij = np.rint(samples / grid).astype(int)
    return np.array([(int(a), int(b)) in cells for a, b in ij], bool)


def atlas_verdict(spans, root):
    dense = [densify(s["pts"]) for s in spans]
    masks = {}
    for key, sub, _, _ in CONDITIONS:
        for a in FLEET:
            g, cells = go_cells(root / sub, a)
            masks[(key, a)] = [covered_by(d, g, cells) for d in dense]

    out = []
    for i, sp in enumerate(spans):
        n = len(dense[i])
        rec, ever = {}, np.zeros(n, bool)
        for key, _, _, _ in CONDITIONS:
            any_arm = np.zeros(n, bool)
            for a in FLEET:
                any_arm |= masks[(key, a)][i]
            ever |= any_arm
            rec[key] = dict(frac=float(any_arm.mean()),
                            per_arm={a: float(masks[(key, a)][i].mean())
                                     for a in FLEET},
                            recovered=bool(any_arm.mean() >= RECOVER_FRAC))
        rec["combo"] = dict(frac=float(ever.mean()), per_arm={},
                            recovered=bool(ever.mean() >= RECOVER_FRAC))
        tier = next((k for k in TIERS if rec[k]["recovered"]), None)
        arm = None
        if tier and rec[tier]["per_arm"]:
            arm = max(rec[tier]["per_arm"], key=lambda a: rec[tier]["per_arm"][a])
        out.append(dict(tier=tier, arm=arm, per_cond=rec,
                        ever_frac=float(ever.mean()),
                        dense=dense[i], ever_mask=ever))
    return out


def failure_mode(row, bases, r_in=0.28, r_out=0.70):
    """Why the never-covered samples of a span are never covered.

    `r_in` / `r_out` are the inner and outer radius of an inverted arm's
    strict-GO annulus TODAY, measured off the eroded atlas.
    """
    d = row["dense"][~row["ever_mask"]]
    if not len(d):
        return "-"
    r = np.min([np.hypot(d[:, 0] - bx, d[:, 1] - by) for bx, by in bases], axis=0)
    under, past = float((r < r_in).mean()), float((r > r_out).mean())
    if under >= 0.6:
        return "under-base hole"
    if past >= 0.6:
        return "waist, past outer reach"
    if under + past < 0.4:
        return "margin trench"
    return "mixed"


# ---------------------------------------------------------------------------
# the planner verdict
# ---------------------------------------------------------------------------
def certified_frac(pts, spec, pen):
    """Fraction of one span `allocate.probe_stroke` can certify for one arm."""
    ivs, _ = allocate.probe_stroke(pts, spec, dict(pen_ext=pen))
    cov, last = 0.0, -1.0
    for s0, s1 in sorted((i.s0, i.s1) for i in ivs):
        s0 = max(s0, last)
        if s1 > s0:
            cov += s1 - s0
            last = s1
    return float(cov)


def planner_verdict(spans, colors):
    """Per span: what each arm can actually certify, at each pen, with ink."""
    out = []
    for sp in spans:
        pts = np.asarray(sp["pts"], float)
        same = [a for a in FLEET if colors[str(a)] == sp["color"]]
        per_pen = {}
        for pen in PENS:
            fr = {a: certified_frac(pts, FLEET[a], pen) for a in FLEET}
            best = max(fr, key=lambda a: fr[a])
            bestc = max(same, key=lambda a: fr[a]) if same else None
            per_pen[pen] = dict(per_arm=fr, best_arm=best, best=fr[best],
                                best_ink_arm=bestc,
                                best_ink=fr[bestc] if bestc else 0.0)
        # cheapest fix, in the order: nothing < re-ink the fleet < longer pen
        p110 = per_pen[0.110]
        if p110["best_ink"] >= RECOVER_FRAC:
            tier, arm = "today", p110["best_ink_arm"]
        elif p110["best"] >= RECOVER_FRAC:
            tier, arm = "repartition", p110["best_arm"]
        elif per_pen[0.200]["best"] >= RECOVER_FRAC:
            tier, arm = "pen200", per_pen[0.200]["best_arm"]
        elif per_pen[0.300]["best"] >= RECOVER_FRAC:
            tier, arm = "pen300", per_pen[0.300]["best_arm"]
        else:
            tier, arm = None, None
        out.append(dict(tier=tier, arm=arm, per_pen=per_pen))
    return out


def probe_dead(spans, av, want, stroke_len=0.05):
    """Hand the WORST point of a dead span to plan_stroke directly.

    The point chosen is the centre of the longest run of samples that no atlas
    covers under any condition — the hardest place on the span, not its
    midpoint.  Every arm, every pen.  If the planner certifies a 5 cm stroke
    there, "dead under every atlas" was the strict gates talking.
    """
    from aris_sixarm.stroke_api import plan_stroke
    out = []
    for i in want:
        if i >= len(spans):
            continue
        d, ev = av[i]["dense"], av[i]["ever_mask"]
        best, run = (0, len(d) // 2), 0
        for j, v in enumerate(~ev):
            run = run + 1 if v else 0
            if run > best[0]:
                best = (run, j - run // 2)
        mid = d[best[1]]
        v = d[-1] - d[0]
        v = v / max(np.linalg.norm(v), 1e-9)
        pts = np.array([mid - 0.5 * stroke_len * v, mid + 0.5 * stroke_len * v])
        res = {}
        for a, spec in FLEET.items():
            for pen in PENS:
                p = plan_stroke(pts, spec, dict(pen_ext=pen))
                res[f"arm{a}_pen{int(pen * 1000)}"] = dict(
                    status=p["status"], reason=p.get("reason", "") or "",
                    s_star=float(p.get("s_star", 0.0)),
                    min_margin=float(p.get("min_margin", float("nan"))),
                    min_sigma=float(p.get("min_sigma", float("nan"))))
        ok = [k for k, r in res.items() if r["status"] == "ok"]
        out.append(dict(span=i, stroke_id=spans[i]["stroke_id"],
                        at=[float(x) for x in mid],
                        dead_run_m=float(best[0] * DS), probes=res, ok=ok,
                        any_ok=bool(ok)))
    return out


# ---------------------------------------------------------------------------
def totals(spans, verdict, tiers):
    out = {k: 0.0 for k in tiers}
    out["dead"] = 0.0
    for sp, v in zip(spans, verdict):
        out[v["tier"] or "dead"] += sp["length_m"]
    return out


# ---------------------------------------------------------------------------
def sheet_panel(ax, spans, program, root, colour_of, title):
    ax.add_patch(plt.Rectangle((0, 0), *SHEET, facecolor="#fdfdf7",
                               edgecolor="#333333", lw=1.3, zorder=0))
    ax.set_aspect("equal")
    ax.set_xlim(-0.22, SHEET[0] + 0.22)
    ax.set_ylim(-0.24, SHEET[1] + 0.22)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # where NOBODY can stand today, as a pale wash — the holes are the point,
    # so shade the complement of the union rather than the union itself
    g, union = 0.02, set()
    for a in FLEET:
        g, cells = go_cells(root / "out_tilt0", a)
        union |= cells
    nx, ny = int(SHEET[0] / g) + 2, int(SHEET[1] / g) + 2
    ij = np.array(sorted(union))
    grid = np.ones((ny, nx), bool)
    ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < nx) & (ij[:, 1] < ny)
    grid[ij[ok, 1], ij[ok, 0]] = False
    ax.imshow(grid, origin="lower",
              extent=(-g / 2, (nx - .5) * g, -g / 2, (ny - .5) * g),
              cmap=ListedColormap([(0, 0, 0, 0), (0.86, 0.42, 0.35, 0.20)]),
              interpolation="nearest", zorder=1,
              clip_path=plt.Rectangle((0, 0), *SHEET, transform=ax.transData))

    # the logo the fleet actually drew, ghosted
    for segs in program["arms"].values():
        for s in segs:
            p = np.asarray(s["pts"], float)
            ax.plot(p[:, 0], p[:, 1], "-", color="#cccccc", lw=1.9, zorder=2,
                    solid_capstyle="round")

    # arm bases and their strict-GO annuli today
    for aid, spec in FLEET.items():
        _, cells = go_cells(root / "out_tilt0", aid)
        if cells:
            c = np.array(sorted(cells)) * g
            r = np.hypot(c[:, 0] - spec.xy[0], c[:, 1] - spec.xy[1])
            for rad, ls in ((r.max(), (0, (6, 4))), (r.min(), (0, (2, 3)))):
                ax.add_patch(plt.Circle(spec.xy, rad, fill=False, lw=1.0,
                                        ls=ls, ec=spec.color, alpha=0.5,
                                        zorder=3))
        ax.plot(*spec.xy, marker="o" if spec.mount == "floor" else "s", ms=12,
                mfc=spec.color, mec="white", mew=1.5, zorder=8, clip_on=False)
        ax.annotate(str(aid), spec.xy, color="white", fontsize=6.5,
                    fontweight="bold", ha="center", va="center", zorder=9,
                    clip_on=False)

    # label offsets, hand-set where spans crowd each other (6/7 and 10/11/12)
    off = {0: (-2, -15), 1: (9, 3), 2: (-14, 5), 3: (-17, -7), 6: (-4, 9),
           7: (9, 1), 8: (9, -9), 10: (-14, 7), 11: (11, 6), 12: (-3, 11)}
    for i, sp in enumerate(spans):
        col = colour_of(i)
        p = np.asarray(sp["pts"], float)
        ax.plot(p[:, 0], p[:, 1], "-", color="white", lw=6.4, zorder=5,
                solid_capstyle="round", alpha=0.9)
        ax.plot(p[:, 0], p[:, 1], "-", color=col, lw=4.2, zorder=6,
                solid_capstyle="round", solid_joinstyle="round")
        ax.annotate(str(i), sp["at"], xytext=off.get(i, (7, 6)),
                    textcoords="offset points", fontsize=7.5, color=col,
                    fontweight="bold", zorder=10)
    ax.set_title(title, fontsize=11, pad=8)


def figure(spans, program, root, av, pv, path):
    ta = totals(spans, av, TIERS)
    tp = totals(spans, pv, PTIERS)
    fig, axes = plt.subplots(2, 1, figsize=(13.2, 10.6),
                             gridspec_kw=dict(hspace=0.62))

    sheet_panel(axes[0], spans, program, root,
                lambda i: TIER_COLOR.get(av[i]["tier"], DEAD_COLOR),
                "ATLAS VERDICT — strict gates (margin ≥ 0.30, σ ≥ 0.14), "
                "pointwise, 1-cell eroded\n"
                "colour = cheapest swept condition whose strict-GO set covers "
                "≥ 95 % of the span")
    h, l = [], []
    for k in TIERS:
        n = sum(1 for v in av if v["tier"] == k)
        if n:
            h.append(Line2D([], [], color=TIER_COLOR[k], lw=4.2))
            l.append(f"{TIER_LABEL[k]} — {n} span{'' if n == 1 else 's'}, "
                     f"{ta[k]:.2f} m")
    n = sum(1 for v in av if v["tier"] is None)
    h.append(Line2D([], [], color=DEAD_COLOR, lw=4.2))
    l.append(f"no swept atlas covers 95 % of it — {n} spans, {ta['dead']:.2f} m")
    h += [Line2D([], [], color="#cccccc", lw=2.4),
          Line2D([], [], color="#dc9c92", lw=7),
          Line2D([], [], color="#888888", lw=1.0, ls=(0, (6, 4)))]
    l += [f"logo the six arms did draw ({program['totals']['drawn_m']:.2f} m)",
          "no arm is strict-GO here today (union of all six, 1-cell eroded)",
          "per-arm strict-GO annulus today (outer / inner radius)"]
    axes[0].legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.02),
                   ncol=2, fontsize=8.5, framealpha=0.95, borderpad=0.6,
                   labelspacing=0.4, columnspacing=1.6)

    sheet_panel(axes[1], spans, program, root,
                lambda i: PTIER_COLOR.get(pv[i]["tier"], DEAD_COLOR),
                "PLANNER VERDICT — plan_stroke's own gates (margin ≥ 0.15, "
                "σ ≥ 0.10), continuous certified walk\n"
                "colour = cheapest change that gets ≥ 95 % of the span "
                "certified by some arm")
    h, l = [], []
    for k in PTIERS:
        n = sum(1 for v in pv if v["tier"] == k)
        if n:
            h.append(Line2D([], [], color=PTIER_COLOR[k], lw=4.2))
            l.append(f"{PTIER_LABEL[k]} — {n} span{'' if n == 1 else 's'}, "
                     f"{tp[k]:.2f} m")
    n = sum(1 for v in pv if v["tier"] is None)
    h.append(Line2D([], [], color=DEAD_COLOR, lw=4.2))
    l.append(f"no arm certifies it at any pen — {n} spans, {tp['dead']:.2f} m")
    axes[1].legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.02),
                   ncol=1, fontsize=8.5, framealpha=0.95, borderpad=0.6,
                   labelspacing=0.4)

    fig.suptitle("CSAIL logo — the 2.22 m nobody drew: what it is, and what "
                 "would recover it", fontsize=13.5, y=0.996)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--out", default=str(ROOT / "out"))
    ap.add_argument("--no-probe", action="store_true")
    ap.add_argument("--probe-spans", default="5,9,10,0")
    args = ap.parse_args(argv)
    root, out = Path(args.root), Path(args.out)

    program = json.load(open(out / "csail_program_6arm.json"))
    spans = program["dropped"]
    av = atlas_verdict(spans, root)
    pv = planner_verdict(spans, program["colors"])
    ta, tp = totals(spans, av, TIERS), totals(spans, pv, PTIERS)
    bases = [FLEET[a].xy for a in FLEET if FLEET[a].mount == "inv"]

    hdr = (f"{'#':>2} {'sid':>4} {'ink':<6} {'len_m':>6} {'at (x, y)':>16}  "
           f"{'atlas':<7} {'arm':>3} {'ever%':>6}  {'planner':<11} {'arm':>3}  "
           f"{'why still dead':<24}")
    print(hdr)
    print("-" * len(hdr))
    for i, sp in enumerate(spans):
        print(f"{i:>2} {sp['stroke_id']:>4} {sp['color']:<6} "
              f"{sp['length_m']:>6.3f} "
              f"({sp['at'][0]:>6.3f},{sp['at'][1]:>6.3f})  "
              f"{(av[i]['tier'] or 'DEAD'):<7} {str(av[i]['arm'] or '-'):>3} "
              f"{100 * av[i]['ever_frac']:>5.1f}  "
              f"{(pv[i]['tier'] or 'DEAD'):<11} {str(pv[i]['arm'] or '-'):>3}  "
              f"{failure_mode(av[i], bases):<24}")

    print(f"\natlas verdict     {'spans':>6} {'whole':>8} {'in bits':>9}")
    for k in TIERS:
        n = sum(1 for v in av if v["tier"] == k)
        bits = sum(sp["length_m"] * v["per_cond"][k]["frac"]
                   for sp, v in zip(spans, av))
        print(f"  {k:<15} {n:>6} {ta[k]:>6.3f} m {bits:>7.3f} m   "
              f"{TIER_LABEL[k]}")
    never = sum(sp["length_m"] * (1 - v["ever_frac"]) for sp, v in zip(spans, av))
    print(f"  {'DEAD':<15} {sum(1 for v in av if v['tier'] is None):>6} "
          f"{ta['dead']:>6.3f} m {never:>7.3f} m   "
          "no atlas covers 95 % of the span / covers the point at all")

    print(f"\nplanner verdict   {'spans':>6} {'whole':>8} {'in bits':>9}")
    for k in PTIERS:
        n = sum(1 for v in pv if v["tier"] == k)
        print(f"  {k:<15} {n:>6} {tp[k]:>6.3f} m {'':>9}   {PTIER_LABEL[k]}")
    for pen in PENS:
        anyv = sum(sp["length_m"] * v["per_pen"][pen]["best"]
                   for sp, v in zip(spans, pv))
        inkv = sum(sp["length_m"] * v["per_pen"][pen]["best_ink"]
                   for sp, v in zip(spans, pv))
        print(f"  pen {int(pen * 1000)} mm {'':>6} {'':>8} {anyv:>7.3f} m   "
              f"certifiable by SOME arm ({inkv:.3f} m by a same-ink arm)")
    print(f"  {'DEAD':<15} {sum(1 for v in pv if v['tier'] is None):>6} "
          f"{tp['dead']:>6.3f} m")

    probes = [] if args.no_probe else probe_dead(
        spans, av, [int(s) for s in args.probe_spans.split(",") if s.strip()])
    for p in probes:
        print(f"\nprobe span {p['span']} (stroke {p['stroke_id']}) at "
              f"({p['at'][0]:.3f}, {p['at'][1]:.3f}) — centre of a "
              f"{p['dead_run_m']:.3f} m run no atlas covers anywhere: "
              f"any ok = {p['any_ok']}")
        bad = {}
        for k, v in p["probes"].items():
            if v["status"] == "ok" or v["s_star"] > 0:
                print(f"    {k:<14} {v['status']:<8} {v['reason']:<17} "
                      f"s*={v['s_star']:.2f} margin={v['min_margin']:.3f} "
                      f"sigma={v['min_sigma']:.3f}")
            else:
                bad.setdefault(v["reason"], []).append(k)
        for reason, ks in bad.items():
            print(f"    {'(none)':<14} {'split':<8} {reason:<17} "
                  f"x{len(ks)}: {', '.join(ks)}")

    figure(spans, program, root, av, pv, out / "dead_spans_analysis.png")
    json.dump(dict(
        recover_frac=RECOVER_FRAC, ds=DS, pens=list(PENS),
        conditions=[c[:3] for c in CONDITIONS],
        atlas_totals=ta, planner_totals=tp, never_covered_m=never,
        probes=probes,
        spans=[dict(i=i, stroke_id=sp["stroke_id"], color=sp["color"],
                    kind=sp["kind"], length_m=sp["length_m"], at=sp["at"],
                    atlas_tier=av[i]["tier"], atlas_arm=av[i]["arm"],
                    ever_frac=av[i]["ever_frac"],
                    why=failure_mode(av[i], bases),
                    per_cond={k: v["frac"] for k, v in av[i]["per_cond"].items()},
                    planner_tier=pv[i]["tier"], planner_arm=pv[i]["arm"],
                    per_pen={str(int(p * 1000)): dict(
                        best=pv[i]["per_pen"][p]["best"],
                        best_arm=pv[i]["per_pen"][p]["best_arm"],
                        best_ink=pv[i]["per_pen"][p]["best_ink"],
                        best_ink_arm=pv[i]["per_pen"][p]["best_ink_arm"],
                        per_arm=pv[i]["per_pen"][p]["per_arm"]) for p in PENS})
               for i, sp in enumerate(spans)],
    ), open(out / "dead_spans_analysis.json", "w"), indent=1, default=float)
    print(f"\nwrote {out / 'dead_spans_analysis.png'} and "
          f"{out / 'dead_spans_analysis.json'}")


if __name__ == "__main__":
    main()
