#!/usr/bin/env python3
"""ATOMIZE-THEN-MERGE: would a segmentation DP beat the current split-then-patch?

WHAT THE PIPELINE DOES TODAY.  `allocate.probe_stroke` maps each (stroke, arm)
into certified s-intervals, `allocate.greedy_cover` covers each stroke with the
fewest of them, and the seams land in the middle of the overlaps.  That cover
is FLYABILITY-BLIND: it knows the arm can ink the span, not that the arm can
get its pen there and away again.  Everything after it is repair —
`prune_unflyable` takes a whole stroke off an arm whose bag has no tour,
`fly_shrink` gives an unreachable END back a geometric ladder at a time,
`merge_remainders` glues the scraps that are left onto neighbouring ink, and
the residual passes re-offer whatever is still empty to the whole fleet.

PETE'S PROPOSAL, PROTOTYPED HERE AND NOWHERE NEAR THE PIPELINE.  Compute each
curve's certified interval cover per arm FIRST, with the flyability in it —
a piece counts only if the owning arm can ENTER at one end, draw the whole
piece on ONE certified corridor, and EXIT at the other, either way round —
and then choose the cuts with a DP along s that minimises the number of pieces
and, among the minimum-piece covers, maximises the SHORTEST piece.

THE VALIDITY TEST, AND WHY IT IS THE HONEST ONE.  For an arm `a`, a stroke and
two breakpoints i < j, the piece [s_i, s_j] is valid iff there is one certified
CORRIDOR of `a` containing it (a span some `plan_stroke` call returned as
planned, so the ink is certified end to end and by one continuous plan), and at
that corridor's own configurations

    (reach[i] and leave[j])   or   (reach[j] and leave[i])

where `reach[k]` is "`paper.route` joins the depot to the hover over s_k" and
`leave[k]` its reverse — the two halves of `allocate.depot_round_trip`, which
is exactly the predicate `fly_shrink` gives ink back to satisfy.  A span that
passes it can be flown AS ITS OWN TOUR in any bag in any order, so a DP whose
every piece passes it produces a partition `prune_unflyable` cannot touch.

  THE ONE APPROXIMATION, STATED.  The hover over a breakpoint is computed from
  the configuration the CORRIDOR's plan holds there, not from the piece's own
  re-plan — a piece cut at s_k may certify on a different q7 and hover
  somewhere else.  That makes the DP's answer slightly OPTIMISTIC and is
  flagged in the report; it is also the only way to keep the probe to one
  route call per breakpoint instead of one per candidate piece.

    ARIS_RIG=proposed ARIS_TOOL=lateral python3 scripts/segmentation_probe.py \
        --program out/csail_program_h094_v14.json --tag v14
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from aris_sixarm import allocate, sequence, trace, writing  # noqa: E402
from aris_sixarm.fleet import FLEET, SHEET  # noqa: E402
from aris_sixarm.stroke_api import polyline_length  # noqa: E402

INF = float("inf")


# ---------------------------------------------------------------------------
# 0. the artwork, re-traced at the shipped placement
# ---------------------------------------------------------------------------
def load_strokes(image, placement, margin=0.06, min_len=0.025):
    doc = json.loads(Path(placement).read_text())["chosen"]
    px, _ = trace.trace_logo(image)
    strokes, info = trace.to_sheet(px, SHEET, margin=margin,
                                   target_width=doc["target_width"],
                                   offset=tuple(doc["offset"]),
                                   rotate_deg=float(doc.get("rotate_deg", 0.0)),
                                   min_len=min_len)
    if not info["fits"]:
        raise SystemExit(f"placement does not fit: {info}")
    return strokes, info


def shipped_partition(path):
    """The v14 programme, as pieces per stroke. -> {stroke_id: [piece]}."""
    prog = json.load(open(path))
    out = {}
    for ph in prog.get("phases", []):
        for arm, segs in ph.get("arms", {}).items():
            for s in segs:
                out.setdefault(int(s["stroke_id"]), []).append(
                    dict(arm=int(arm), s0=float(s["s_range"][0]),
                         s1=float(s["s_range"][1]),
                         length=float(s["length_m"]),
                         direction=int(s["direction"])))
    for v in out.values():
        v.sort(key=lambda p: p["s0"])
    return out, prog


# ---------------------------------------------------------------------------
# 1. certified corridors, with the flyability in them
# ---------------------------------------------------------------------------
def merge_iv(ivs, tol=1e-9):
    """Union of (s0, s1) pairs -> disjoint sorted list."""
    out = []
    for a, b in sorted((float(x[0]), float(x[1])) for x in ivs):
        if out and a <= out[-1][1] + tol:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def corridors(st, spec, opts, max_probes, min_seg, bisect=True):
    """Certified corridors of one arm on one stroke. -> [(s0, s1, plan)].

    Each is a span `plan_stroke` returned as PLANNED and then re-planned end to
    end on its own, so the corridor carries a real dense joint path and not a
    promise inherited from a longer probe.  Direction-agnostic: forward is
    tried first and backward only if forward had to give ground, because a plan
    is a walk of the redundancy band and the band is not symmetric.  The
    corridor's `qs` are in the drawn order, which is all the hover probe needs.
    """
    pts = np.asarray(st["pts"], float)
    L = polyline_length(pts)
    ivs, stats = allocate.probe_stroke(pts, spec, opts, max_probes=int(max_probes),
                                       min_seg=min_seg, bisect=bisect)
    out, calls = [], int(stats["probes"])
    for a, b in merge_iv([(v.s0, v.s1) for v in ivs]):
        if (b - a) * L < min_seg:
            continue
        got = None
        for d in (+1, -1):
            plan, s2 = allocate.replan_segment(
                pts, dict(s0=a, s1=b, direction=d), spec, opts, min_seg=min_seg)
            calls += int(s2["replans"])
            if plan is not None:
                got = (float(s2["s0"]), float(s2["s1"]), plan, d)
                if s2["lost"] <= 1e-12:
                    break
        if got is not None and (got[1] - got[0]) * L >= min_seg:
            out.append(got)
    return out, calls


def breakpoints(L, step, s0=0.0, s1=1.0, cap=121):
    """Uniform cut grid on [s0, s1], in normalised arc length."""
    n = int(np.clip(round((s1 - s0) * L / max(step, 1e-9)), 1, cap - 1))
    return np.linspace(s0, s1, n + 1)


def hover_flags(spec, st, cor, bps, mat, min_seg):
    """`reach` / `leave` at each breakpoint inside one corridor. -> (r, l).

    Both come out of ONE `sequence.home_legs` call over the corridor's own
    consecutive breakpoint pieces: node `2i` of piece i is entered at its first
    sample and left at its last, so `into[2i]` is depot -> hover(bp i) and
    `outof[2i]` is hover(bp i+1) -> depot; the reversed node supplies the other
    two.  That is the same `endpoints` -> `depot_legs` chain the sequencer and
    the timeline both use, so a flag here is the router's real verdict.
    """
    s0, s1, plan, d = cor
    qs = np.asarray(plan["qs"], float)
    pts = np.asarray(plan["pts"], float)
    n = len(qs)
    # the plan runs s0 -> s1 when drawn forward and s1 -> s0 when reversed;
    # map every breakpoint to its sample index in the plan's own order
    u = (bps - s0) / max(s1 - s0, 1e-12)
    if d < 0:
        u = 1.0 - u
    k = np.clip(np.rint(u * (n - 1)).astype(int), 0, n - 1)
    segs = []
    for i in range(len(bps) - 1):
        lo, hi = sorted((int(k[i]), int(k[i + 1])))
        if hi - lo < 1:
            segs.append(None)
            continue
        sl = slice(lo, hi + 1)
        segs.append(dict(plan=dict(qs=qs[sl], pts=pts[sl])))
    live = [i for i, s in enumerate(segs) if s is not None]
    reach = np.zeros(len(bps), bool)
    leave = np.zeros(len(bps), bool)
    if not live:
        return reach, leave
    outof, into = sequence.home_legs(spec, [segs[i] for i in live], **mat)
    for m, i in enumerate(live):
        # node 2m enters at the piece's first sample, exits at its last
        a, b = (i, i + 1) if k[i] <= k[i + 1] else (i + 1, i)
        reach[a] |= bool(np.isfinite(into[2 * m]))
        leave[b] |= bool(np.isfinite(outof[2 * m]))
        reach[b] |= bool(np.isfinite(into[2 * m + 1]))
        leave[a] |= bool(np.isfinite(outof[2 * m + 1]))
    return reach, leave


# ---------------------------------------------------------------------------
# 2. the segmentation DP
# ---------------------------------------------------------------------------
def segment_dp(nb, valid, length, min_piece, max_pieces=8):
    """The segmentation DP: cover the most ink, in the fewest pieces, and make
    the SHORTEST piece as long as possible. -> (pieces, k, uncovered_m).

    `valid[(i, j)]` is the (non-empty) set of arms that can own the piece
    [i, j]; `length(i, j)` its metres.  A stretch no arm can own is left
    UNCOVERED rather than making the whole problem infeasible — the pipeline
    drops such ink too, and a DP that refused to would be answering a question
    nobody asked.  Coverage is therefore the primary objective, piece count the
    second and the max-min piece length the third.

    `val[k][j]` is the best (uncovered metres, largest minimum piece) over
    covers of [0, j] using exactly k owned pieces.  Coverage is additive and
    exact; the max-min tie-break is greedy-lexicographic behind it, the same
    discipline `planner.plan`'s bottleneck DP uses for its own tie-break.
    """
    K = int(max_pieces)
    val = [[None] * nb for _ in range(K + 1)]
    back = [[None] * nb for _ in range(K + 1)]     # (i, owned?)
    val[0][0] = (0.0, INF)

    def better(a, b):
        return b is None or (a[0], -a[1]) < (b[0], -b[1])

    for k in range(K + 1):
        if k:
            val[k][0] = None
        for j in range(1, nb):
            best, bk = None, None
            for i in range(j):
                if val[k][i] is not None:                       # a hole
                    c = (val[k][i][0] + length(i, j), val[k][i][1])
                    if better(c, best):
                        best, bk = c, (i, False)
                if k and val[k - 1][i] is not None:              # an owned piece
                    if valid.get((i, j)):
                        Lp = length(i, j)
                        if Lp >= min_piece - 1e-12:
                            c = (val[k - 1][i][0], min(val[k - 1][i][1], Lp))
                            if better(c, best):
                                best, bk = c, (i, True)
            val[k][j], back[k][j] = best, bk

    ends = [(k, val[k][nb - 1]) for k in range(K + 1) if val[k][nb - 1]]
    if not ends:
        return None, None, None
    lo = min(v[0] for _, v in ends)
    k = min(k for k, v in ends if v[0] <= lo + 1e-12)
    pieces, j, kk = [], nb - 1, k
    while j > 0:
        i, owned = back[kk][j]
        if owned:
            pieces.append((i, j, sorted(valid[(i, j)])))
            kk -= 1
        j = i
    return pieces[::-1], k, float(lo)


def piece_flyable(table, arm, s0, s1, bps):
    """Would THIS span pass the DP's own validity test? -> bool.

    The span's ends are snapped to the nearest candidate cut, which is what
    makes this a fair question to ask of a cover that never saw the grid: the
    seams `place_cuts` produces sit mid-overlap and land between breakpoints.
    """
    i = int(np.argmin(np.abs(bps - s0)))
    j = int(np.argmin(np.abs(bps - s1)))
    if i == j:
        return True                       # shorter than one cut step
    i, j = min(i, j), max(i, j)
    for t in table:
        if t["arm"] != arm:
            continue
        idx = list(t["idx"])
        if i not in idx or j not in idx:
            continue
        p, q = idx.index(i), idx.index(j)
        if (t["reach"][p] and t["leave"][q]) or (t["reach"][q] and t["leave"][p]):
            return True
    return False


def assign(pieces, load, lengths):
    """Owner per piece, least-loaded eligible arm first. -> [(piece, arm)]."""
    out = []
    for p, Lp in zip(pieces, lengths):
        arm = min(p[2], key=lambda a: (load.get(a, 0.0), a))
        load[arm] = load.get(arm, 0.0) + Lp
        out.append((p, arm))
    return out


# ---------------------------------------------------------------------------
def stats(lengths):
    if not lengths:
        return dict(n=0, min=0.0, p50=0.0, max=0.0, total=0.0)
    a = np.asarray(lengths, float)
    return dict(n=int(len(a)), min=float(a.min()), p50=float(np.median(a)),
                max=float(a.max()), total=float(a.sum()))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--program", default="out/csail_program_h094_v14.json")
    ap.add_argument("--bench", default=None,
                    help="a drawing from aris_sixarm/bench instead of the logo "
                         "(no shipped partition to compare against)")
    ap.add_argument("--image", default=str(ROOT / "assets/csail/csail_old_med.gif"))
    ap.add_argument("--placement", default="out/csail_place_v4_placement.json")
    ap.add_argument("--probe-ref-m", type=float, default=None,
                    help="stroke length --max-probes was chosen for; a stroke n "
                         "times longer then gets n times the calls "
                         "(allocate.stroke_probes)")
    ap.add_argument("--margin", type=float, default=0.06)
    ap.add_argument("--tilt-max-deg", type=float, default=15.0)
    ap.add_argument("--max-probes", type=int, default=6)
    ap.add_argument("--cut-step", type=float, default=0.05,
                    help="m of arc between candidate cut positions")
    ap.add_argument("--min-piece", type=float, default=allocate.MIN_SPLIT_M)
    ap.add_argument("--max-pieces", type=int, default=8)
    ap.add_argument("--transit-speed", type=float, default=writing.TRANSIT_SPEED)
    ap.add_argument("--qd-frac", type=float, default=0.30)
    ap.add_argument("--strokes", type=int, default=0, help="limit, for smoke runs")
    ap.add_argument("--tag", default="v14")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    t0 = time.time()
    if a.bench:
        from aris_sixarm import bench
        strokes, _ = bench.make(a.bench)
        ship = {}
    else:
        strokes, _ = load_strokes(a.image, a.placement, a.margin)
        ship, _ = shipped_partition(a.program)
    arms = sorted(FLEET)
    pens = {x: 0.110 for x in arms}
    opts = {x: dict(objective="maximin_sigma", tilt_max_deg=a.tilt_max_deg,
                    pen_ext=pens[x]) for x in arms}
    mat_of = {x: dict(transit_speed=a.transit_speed, qd_frac=a.qd_frac,
                      pen_ext=pens[x], paper_safe=True) for x in arms}
    if a.strokes:
        strokes = strokes[:a.strokes]
    print(f"{a.bench or 'CSAIL logo at the v14 placement'}: {len(strokes)} "
          f"strokes, {trace.total_length(strokes):.3f} m  "
          f"({time.time() - t0:.1f} s)")

    per_stroke, n_calls = [], 0
    for si, st in enumerate(strokes):
        ts = time.time()
        L = polyline_length(st["pts"])
        bps = breakpoints(L, a.cut_step)
        nb = len(bps)
        budget = allocate.stroke_probes(L, a.max_probes, a.probe_ref_m)
        cors = {}
        for x in arms:
            cs, c = corridors(st, FLEET[x], opts[x], budget, allocate.MIN_SEG_M)
            n_calls += c
            if cs:
                cors[x] = cs
        t_probe = time.time() - ts
        # per (arm, corridor): which breakpoints it contains, and their flags
        table = []
        for x, cs in cors.items():
            for cor in cs:
                s0, s1 = cor[0], cor[1]
                inside = np.flatnonzero((bps >= s0 - 1e-9) & (bps <= s1 + 1e-9))
                if len(inside) < 2:
                    continue
                r, l = hover_flags(FLEET[x], st, cor, bps[inside], mat_of[x],
                                   allocate.MIN_SEG_M)
                table.append(dict(arm=x, s0=s0, s1=s1, idx=inside,
                                  reach=r, leave=l))
        # validity of every candidate piece
        valid, ink_only = {}, {}
        for t in table:
            idx, r, l = t["idx"], t["reach"], t["leave"]
            for p in range(len(idx)):
                for q in range(p + 1, len(idx)):
                    key = (int(idx[p]), int(idx[q]))
                    ink_only.setdefault(key, set()).add(t["arm"])
                    if (r[p] and l[q]) or (r[q] and l[p]):
                        valid.setdefault(key, set()).add(t["arm"])

        def length(i, j):
            return float(bps[j] - bps[i]) * L

        floor = min(a.min_piece, L)
        pieces, k, unc = segment_dp(nb, valid, length, floor, a.max_pieces)
        pieces_ink, k_ink, unc_ink = segment_dp(nb, ink_only, length, floor,
                                                a.max_pieces)
        # the flyability-blind cover the pipeline actually builds today, on the
        # SAME corridors, so the comparison is about the choice and not the probe
        ivs = [allocate.Interval(c[0], c[1], x, +1, "corridor")
               for x, cs in cors.items() for c in cs]
        chosen, gaps = allocate.greedy_cover(ivs, {},
                                             min_gap=allocate.GAP_TOL_M / max(L, 1e-9))
        spans = allocate.place_cuts(chosen, L)
        rec = dict(stroke=int(st["id"]), L=float(L), nb=int(nb),
                   n_corridors=int(len(table)),
                   arms_with_ink=sorted(cors), bps=bps,
                   # the expensive half of the run, kept so the DP can be
                   # re-run over it without re-probing or re-routing anything
                   table=[dict(arm=int(t["arm"]), s0=float(t["s0"]),
                               s1=float(t["s1"]), idx=t["idx"],
                               reach=t["reach"], leave=t["leave"])
                          for t in table],
                   dp=dict(k=k, uncovered=unc,
                           pieces=[[int(i), int(j), arm]
                                   for i, j, arm in (pieces or [])],
                           lengths=[length(i, j) for i, j, _ in (pieces or [])]),
                   dp_ink=dict(k=k_ink, uncovered=unc_ink,
                               lengths=[length(i, j)
                                        for i, j, _ in (pieces_ink or [])]),
                   greedy=dict(k=len(spans),
                               lengths=[(s["s1"] - s["s0"]) * L for s in spans],
                               arms=[s["arm"] for s in spans],
                               unflyable=[not piece_flyable(table, s["arm"],
                                                            s["s0"], s["s1"], bps)
                                          for s in spans],
                               gap_m=float(sum(b - c for c, b in gaps) * L)),
                   shipped=dict(k=len(ship.get(int(st["id"]), [])),
                                lengths=[p["length"]
                                         for p in ship.get(int(st["id"]), [])],
                                arms=[p["arm"]
                                      for p in ship.get(int(st["id"]), [])]))
        rec["t_probe"], rec["t_total"] = t_probe, time.time() - ts
        per_stroke.append(rec)
        print(f"  [{si + 1}/{len(strokes)}] stroke {st['id']} L={L:.3f} m "
              f"bp={nb} corridors={len(table)} arms={sorted(cors)} "
              f"dp={k} dp_ink={k_ink} greedy={len(spans)} "
              f"shipped={rec['shipped']['k']}  "
              f"({t_probe:.1f}+{time.time() - ts - t_probe:.1f} s)", flush=True)

    # ---- the comparison table ---------------------------------------------
    def gather(key, field="lengths"):
        return [x for r in per_stroke for x in r[key][field]]

    rows = []
    for name, key in (("shipped v14", "shipped"), ("greedy on same corridors",
                                                   "greedy"),
                      ("segmentation DP (ink only)", "dp_ink"),
                      ("segmentation DP (flyable)", "dp")):
        Ls = gather(key)
        s = stats(Ls)
        n_solved = sum(1 for r in per_stroke
                       if (r[key].get("k") or 0) > 0)
        rows.append((name, s, n_solved))
    total_m = float(sum(r["L"] for r in per_stroke))
    print(f"\n=== {len(per_stroke)} strokes, {total_m:.3f} m: how the ink is "
          f"cut up, four ways ===")
    print(f"{'':32s} {'pieces':>7s} {'strokes':>8s} {'drawn m':>9s} "
          f"{'min m':>7s} {'med m':>7s} {'max m':>7s} {'seams':>6s} {'empty m':>8s}")
    for name, s, n_solved in rows:
        seams = max(s["n"] - n_solved, 0)
        print(f"{name:32s} {s['n']:7d} {n_solved:8d} {s['total']:9.3f} "
              f"{s['min']:7.3f} {s['p50']:7.3f} {s['max']:7.3f} {seams:6d} "
              f"{total_m - s['total']:8.3f}")
    nfly = sum(int(x) for r in per_stroke for x in r["greedy"]["unflyable"])
    mfly = sum(Lp for r in per_stroke
               for Lp, bad in zip(r["greedy"]["lengths"], r["greedy"]["unflyable"])
               if bad)
    print(f"\nof the greedy cover's {sum(r['greedy']['k'] for r in per_stroke)} "
          f"pieces, {nfly} ({mfly:.3f} m) fail the DP's own entry/exit test — "
          f"the ink split-then-patch has to repair afterwards")
    hole = sum(r["greedy"]["gap_m"] for r in per_stroke)
    print(f"the same corridors leave {hole:.4f} m no arm covers at all")

    # load sketch: DP pieces assigned to the least-loaded eligible arm
    load = {x: 0.0 for x in arms}
    for r in per_stroke:
        pcs = [(i, j, set(arm)) for i, j, arm in r["dp"]["pieces"]]
        assign(pcs, load, r["dp"]["lengths"])
    ship_load = {x: 0.0 for x in arms}
    for r in per_stroke:
        for arm, Lp in zip(r["shipped"]["arms"], r["shipped"]["lengths"]):
            ship_load[arm] = ship_load.get(arm, 0.0) + Lp
    print("\n=== metres per arm ===")
    print(f"{'arm':>5s} {'shipped v14':>12s} {'DP + greedy assign':>20s}")
    for x in arms:
        print(f"{x:5d} {ship_load.get(x, 0.0):12.3f} {load[x]:20.3f}")
    sv = [ship_load.get(x, 0.0) for x in arms]
    dv = [load[x] for x in arms]
    print(f"{'busiest':>5s} {max(sv):12.3f} {max(dv):20.3f}")
    print(f"{'spread':>5s} {max(sv) - min(sv):12.3f} {max(dv) - min(dv):20.3f}")

    out = a.out or f"out/segmentation_probe_{a.tag}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    def _j(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, set):
            return sorted(o)
        raise TypeError(type(o))

    with open(out, "w") as f:
        json.dump(dict(argv=vars(a), plan_calls=int(n_calls),
                       per_stroke=per_stroke,
                       load_dp=load, load_shipped=ship_load,
                       wall_s=time.time() - t0), f, default=_j)
    print(f"\n{n_calls} plan calls, {time.time() - t0:.1f} s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
