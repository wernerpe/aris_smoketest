"""Strokes + fleet -> per-arm programs of CERTIFIED segments (and what was dropped).

The planner's contract (`stroke_api.plan_stroke`) is certified-or-split: one
call either returns a plan an independent validator has accepted, or the arc
length `s_star` up to which it could.  That makes reach a *measured* quantity
rather than a modelled one, and this module is the consumer that contract was
written for.

  PROBE      For each (stroke, arm) up to three plan calls buy the arm's
             feasible s-intervals on that stroke: forward gives [0, s*],
             the reversed stroke gives [1-s*_rev, 1] (direction matters — the
             DP walks the redundancy band from wherever it starts), and one
             more call on the largest remaining gap finds an interval in the
             middle.  Everything a probe returns is certified; nothing here
             extrapolates a plan.
  PARTITION  Each arm carries ONE pen for the whole piece, so a grey stroke
             can only be covered by grey arms.  With four active arms there
             are 2^4 - 2 = 14 non-trivial colour partitions, so the choice is
             made by enumeration rather than by heuristic: minimise dropped
             length, then the spread of per-arm drawing length.
  COVER      Per stroke, greedy interval covering over the intervals of the
             arms with the right pen — which is optimal for "fewest pieces",
             i.e. fewest pen-up handoffs in the middle of a line.  Ties go to
             the least-loaded arm.  Whatever the intervals do not cover is
             DROPPED and reported; nothing is moved to make it fit.
  CUT        A handoff is placed in the MIDDLE of the overlap between the two
             intervals, not at either edge, and both segments are grown by
             `overlap` metres past it so the ink meets.  Cutting at an edge
             would put the seam exactly where one of the two arms is at the
             end of its certified reach.
  ORDER      Nearest-neighbour chaining of an arm's segments from its base.
             A heuristic, and labelled as one: transit motion is roadmap
             item 3's RRT, not this module's business.

Every chosen segment is re-planned from scratch at the end; a segment whose
clean re-plan is not "ok" is not shipped as one.
"""
import time
from dataclasses import dataclass

import numpy as np

from .fleet import FLEET
from .stroke_api import plan_stroke, polyline_length, truncate_polyline

ACTIVE = [aid for aid, s in FLEET.items() if s.active]
COLORS = ("grey", "orange")

EPS_S = 1e-6
OVERLAP_M = 0.004        # m of ink each side of a handoff cut
BACKOFF_M = 0.006        # m to give up per failed clean re-plan
MIN_SEG_M = 0.025        # m; a shorter piece is not worth a pen-up
PREFILTER_R = 0.05       # m; atlas cell distance that still counts as maybe


@dataclass
class Interval:
    """A certified span [s0, s1] of one stroke for one arm, in one direction."""
    s0: float
    s1: float
    arm: int
    direction: int = 1        # +1 = drawn with increasing s, -1 = reversed
    source: str = "fwd"


# ===========================================================================
# 1. probing: certified s-intervals per (stroke, arm)
# ===========================================================================
def _certified_span(res):
    """(fraction of the probed polyline that came back certified, status)."""
    st = res["status"]
    if st == "ok":
        return 1.0, st
    if st == "split":
        return float(max(res.get("s_star", 0.0), 0.0)), st
    return 0.0, st


def probe_stroke(pts, spec, opts=None, max_probes=3, min_seg=MIN_SEG_M):
    """Up to `max_probes` plan calls -> (intervals, stats) for one (stroke, arm).

    Intervals are in the stroke's own normalised arc length and carry the
    direction they were certified in, because a plan is a walk through the
    redundancy band and the band is not symmetric: an arm can often draw the
    last 60 % of a line end-to-start that it cannot reach start-to-end.
    """
    pts = np.asarray(pts, float)
    L = polyline_length(pts)
    iv, stats = [], dict(probes=0, statuses=[], arm=spec.arm_id)

    r = plan_stroke(pts, spec, opts)
    stats["probes"] += 1
    stats["statuses"].append(r["status"])
    s, st = _certified_span(r)
    if st == "ok":
        return [Interval(0.0, 1.0, spec.arm_id, +1, "fwd")], stats
    if st == "degenerate":
        return [], stats
    if s * L >= min_seg:
        iv.append(Interval(0.0, s, spec.arm_id, +1, "fwd"))
    # a stroke that starts off-reach reports where the arm could pick it up
    lo = float(r.get("s_resume") or 0.0)
    hi = 1.0

    if max_probes >= 2:
        rr = plan_stroke(pts[::-1], spec, opts)
        stats["probes"] += 1
        stats["statuses"].append(rr["status"])
        sr, st = _certified_span(rr)
        if st == "ok":
            return [Interval(0.0, 1.0, spec.arm_id, -1, "rev")], stats
        if sr * L >= min_seg:
            iv.append(Interval(1.0 - sr, 1.0, spec.arm_id, -1, "rev"))
        hi = 1.0 - float(rr.get("s_resume") or 0.0)

    gaps = [g for g in uncovered(iv, min_gap=min_seg / max(L, 1e-9))
            if min(g[1], hi) - max(g[0], lo) > min_seg / max(L, 1e-9)]
    if max_probes >= 3 and gaps and (lo > 0 or hi < 1 or iv):
        # (with no interval and no resume hint the third probe would just
        # repeat the first one on the same polyline)
        a, b = max(gaps, key=lambda g: min(g[1], hi) - max(g[0], lo))
        a, b = max(a, lo), min(b, hi)
        sub = truncate_polyline(pts, a, b)
        if len(sub) >= 2 and polyline_length(sub) >= min_seg:
            r3 = plan_stroke(sub, spec, opts)
            stats["probes"] += 1
            stats["statuses"].append(r3["status"])
            s3, st3 = _certified_span(r3)
            if st3 in ("ok", "split") and s3 * (b - a) * L >= min_seg:
                iv.append(Interval(a, a + s3 * (b - a), spec.arm_id, +1, "gap"))
    return iv, stats


def uncovered(intervals, lo=0.0, hi=1.0, min_gap=0.0):
    """The parts of [lo, hi] no interval covers, as a list of (a, b)."""
    if not intervals:
        return [(lo, hi)] if hi - lo > min_gap else []
    out, x = [], lo
    for i in sorted(intervals, key=lambda v: v.s0):
        if i.s0 > x + min_gap:
            out.append((x, min(i.s0, hi)))
        x = max(x, i.s1)
        if x >= hi:
            break
    if hi - x > min_gap:
        out.append((x, hi))
    return [(a, b) for a, b in out if b - a > min_gap]


# ===========================================================================
# 2. covering one stroke with the fewest pieces
# ===========================================================================
def greedy_cover(intervals, load=None, lo=0.0, hi=1.0, min_gap=0.0):
    """Fewest-intervals cover of [lo, hi] -> (chosen, gaps).

    The textbook greedy (repeatedly take the interval that starts at or before
    the frontier and reaches furthest) is optimal for the number of pieces,
    which is exactly the objective we want first: every extra piece is a
    pen-up, a handoff and a visible seam.  `load` (arm -> metres assigned so
    far) breaks ties toward the idler arm.
    """
    load = load or {}
    chosen, gaps, x = [], [], lo
    ivs = sorted(intervals, key=lambda v: (v.s0, -v.s1))
    while x < hi - min_gap:
        cand = [v for v in ivs if v.s0 <= x + EPS_S and v.s1 > x + EPS_S]
        if not cand:
            nxt = [v for v in ivs if v.s0 > x]
            if not nxt:
                gaps.append((x, hi))
                break
            nx = min(v.s0 for v in nxt)
            gaps.append((x, min(nx, hi)))
            x = nx
            continue
        best = max(v.s1 for v in cand)
        tied = [v for v in cand if v.s1 >= best - EPS_S]
        pick = min(tied, key=lambda v: (load.get(v.arm, 0.0), v.arm))
        chosen.append(pick)
        x = pick.s1
    return chosen, [(a, b) for a, b in gaps if b - a > min_gap]


def place_cuts(chosen, L, overlap=OVERLAP_M):
    """Chosen intervals -> drawn spans, handoffs moved to mid-overlap.

    Consecutive picks overlap (the greedy only advances the frontier); the
    seam goes at the middle of that overlap so neither arm is asked to draw to
    the very end of its certified reach, and each side is extended by
    `overlap` metres so the two pen strokes meet instead of leaving a gap.
    """
    if not chosen:
        return []
    d = overlap / max(L, 1e-9)
    spans = []
    for k, v in enumerate(chosen):
        a, b = v.s0, v.s1
        if k > 0:
            prev = chosen[k - 1]
            mid = 0.5 * (v.s0 + prev.s1)
            a = float(np.clip(mid - d, v.s0, v.s1))
        if k + 1 < len(chosen):
            nxt = chosen[k + 1]
            mid = 0.5 * (nxt.s0 + v.s1)
            b = float(np.clip(mid + d, v.s0, v.s1))
        spans.append(dict(s0=a, s1=b, arm=v.arm, direction=v.direction,
                          source=v.source, interval=(v.s0, v.s1)))
    return spans


# ===========================================================================
# 3. colour partitions
# ===========================================================================
def partitions(arms, nontrivial=True):
    """All ways to give each arm one pen colour -> list of {arm: "grey"|"orange"}.

    2^n assignments; with `nontrivial` the two that leave a colour with no arm
    (and therefore drop every stroke of that colour) are dropped, leaving 14
    for the four active arms.
    """
    out = []
    for bits in range(1 << len(arms)):
        cols = [COLORS[(bits >> i) & 1] for i in range(len(arms))]
        if nontrivial and len(set(cols)) < 2:
            continue
        out.append({a: c for a, c in zip(arms, cols)})
    return out


def cover_all(strokes, ivmap, colors, min_seg=MIN_SEG_M):
    """Cover every stroke using only arms whose pen matches -> dict of results."""
    load = {a: 0.0 for a in colors}
    per_stroke, dropped = [], []
    for st in strokes:
        L = polyline_length(st["pts"])
        ivs = [v for v in ivmap.get(st["id"], []) if colors.get(v.arm) == st["color"]]
        min_gap = min_seg / max(L, 1e-9)
        chosen, gaps = greedy_cover(ivs, load, min_gap=min_gap)
        for v in chosen:
            load[v.arm] += (v.s1 - v.s0) * L
        per_stroke.append(dict(stroke=st, chosen=chosen, gaps=gaps, L=L))
        dropped += [dict(stroke_id=st["id"], color=st["color"], s0=a, s1=b,
                         length=(b - a) * L) for a, b in gaps]
    return dict(per_stroke=per_stroke, dropped=dropped, load=load,
                dropped_len=float(sum(d["length"] for d in dropped)),
                cuts=int(sum(max(len(p["chosen"]) - 1, 0) for p in per_stroke)))


def best_partition(strokes, ivmap, arms=None, min_seg=MIN_SEG_M):
    """Enumerate the 14 partitions -> (best colors, best cover, ranking table)."""
    arms = arms or ACTIVE
    table = []
    for colors in partitions(arms):
        c = cover_all(strokes, ivmap, colors, min_seg)
        loads = [c["load"][a] for a in arms]
        spread = float(max(loads) - min(loads))
        table.append(dict(colors=colors, cover=c, dropped=c["dropped_len"],
                          spread=spread, cuts=c["cuts"], loads=dict(c["load"])))
    table.sort(key=lambda t: (round(t["dropped"], 4), round(t["spread"], 4),
                              t["cuts"]))
    return table[0]["colors"], table[0]["cover"], table


# ===========================================================================
# 4. clean re-plan of the chosen segments
# ===========================================================================
def _segment_points(pts, s0, s1, direction):
    sub = truncate_polyline(pts, s0, s1)
    return sub[::-1] if direction < 0 else sub


def replan_segment(pts, span, spec, opts=None, backoff=BACKOFF_M,
                   tries=3, min_seg=MIN_SEG_M):
    """Re-plan one chosen span from scratch, shrinking it if it does not certify.

    The span came out of a probe of a DIFFERENT polyline (the full stroke, or
    the reversed one, or a sub-range), so its endpoints are certified but its
    exact re-plan is a fresh question.  Rather than ship an uncertified
    segment we give the far end back, `backoff` metres at a time, and report
    whatever the last certified plan covers.
    """
    L = polyline_length(pts)
    s0, s1 = span["s0"], span["s1"]
    d = span["direction"]
    lost = 0.0
    for _ in range(tries):
        seg = _segment_points(pts, s0, s1, d)
        if len(seg) < 2 or polyline_length(seg) < min_seg:
            break
        r = plan_stroke(seg, spec, opts)
        if r["status"] == "ok":
            return r, dict(span, s0=s0, s1=s1, lost=lost, replans=1)
        if r["status"] == "split" and r.get("head") is not None:
            frac = float(np.clip(r["s_star"], 0.0, 1.0))
            keep = (s1 - s0) * frac
            lost += (s1 - s0) - keep
            if d > 0:
                s1 = s0 + keep
            else:
                s0 = s1 - keep
            continue
        back = backoff / max(L, 1e-9)
        lost += back
        if d > 0:
            s1 -= back
        else:
            s0 += back
    return None, dict(span, s0=s0, s1=s1, lost=lost, replans=tries)


# ===========================================================================
# 5. ordering
# ===========================================================================
def order_nearest(items, start_xy):
    """Nearest-neighbour chaining from `start_xy` -> (order, transit metres).

    Greedy, no 2-opt, no claim of optimality: pen-up transits are planned by
    roadmap item 3, and until they are, their length is only an indicator.
    """
    left = list(range(len(items)))
    pos = np.asarray(start_xy, float)
    order, transit = [], 0.0
    while left:
        k = min(left, key=lambda i: np.linalg.norm(items[i]["pts"][0] - pos))
        transit += float(np.linalg.norm(items[k]["pts"][0] - pos))
        pos = items[k]["pts"][-1]
        order.append(k)
        left.remove(k)
    return order, transit


# ===========================================================================
# 6. what nobody drew
# ===========================================================================
def leftover(strokes, programs, min_gap_m=0.002):
    """Complement of the finally-certified segments, WITH geometry.

    Derived from the shipped programs rather than accumulated along the way,
    so "traced = drawn + dropped" holds by construction and no bookkeeping
    slip can hide a piece of paper that stays empty.
    """
    drawn = {}
    for a, segs in programs.items():
        for s in segs:
            drawn.setdefault(s["stroke_id"], []).append(
                Interval(min(*s["s_range"]), max(*s["s_range"]), a))
    out = []
    for st in strokes:
        L = polyline_length(st["pts"])
        for a, b in uncovered(drawn.get(st["id"], []),
                              min_gap=min_gap_m / max(L, 1e-9)):
            pts = truncate_polyline(st["pts"], a, b)
            if len(pts) < 2:
                continue
            out.append(dict(stroke_id=st["id"], color=st["color"],
                            kind=st.get("kind", ""), s_range=(float(a), float(b)),
                            length=float((b - a) * L), pts=pts,
                            at=[float(x) for x in pts[len(pts) // 2]]))
    return out


def where(dropped, sheet, nx=3, ny=3):
    """Aggregate dropped length into a coarse grid of the sheet -> {(ix,iy): m}."""
    cell = {}
    for d in dropped:
        p = np.asarray(d["pts"], float)
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        mid = 0.5 * (p[1:] + p[:-1])
        for (x, y), w in zip(mid, seg):
            k = (min(int(x / sheet[0] * nx), nx - 1),
                 min(int(y / sheet[1] * ny), ny - 1))
            cell[k] = cell.get(k, 0.0) + float(w)
    return cell


# ===========================================================================
# 7. the whole allocation
# ===========================================================================
def prefilter(strokes, arms, atlas_dir=None, radius=PREFILTER_R):
    """-> {(stroke_id, arm): True} where the atlas says probing is worth it.

    The atlas is a 2 cm sweep of the paper; a cell counts if the pen reached
    it PERPENDICULAR (tilt 0 — the planner never leans the pen) with the
    planner's own permissive joint margin.  A stroke with no such cell near
    any of its points cannot be planned, so the probe is skipped.  Absent an
    atlas everything is probed.
    """
    ok = {}
    if atlas_dir is None:
        return None
    from .atlas import load
    grids = {}
    for a in arms:
        try:
            arr, meta = load(atlas_dir, a)
        except Exception:
            return None
        g = float(meta["grid"])
        rows = arr[(arr[:, 8] <= 0.0) & (arr[:, 2] >= 0.15)]
        grids[a] = (g, {(int(round(x / g)), int(round(y / g)))
                        for x, y in rows[:, :2]})
    k = int(np.ceil(radius / 0.02))
    for st in strokes:
        p = st["pts"]
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        t = np.concatenate([[0], np.cumsum(seg)])
        dense = np.column_stack([np.interp(np.arange(0, t[-1], 0.02), t, p[:, 0]),
                                 np.interp(np.arange(0, t[-1], 0.02), t, p[:, 1])])
        dense = np.vstack([p[0], dense, p[-1]])
        for a in arms:
            g, cells = grids[a]
            hit = any((int(round(x / g)) + dx, int(round(y / g)) + dy) in cells
                      for x, y in dense
                      for dx in range(-k, k + 1) for dy in range(-k, k + 1))
            ok[(st["id"], a)] = hit
    return ok


def allocate(strokes, arms=None, opts=None, atlas_dir=None, verbose=True,
             min_seg=MIN_SEG_M, overlap=OVERLAP_M):
    """Strokes -> per-arm certified programs + the dropped list.  See module docs."""
    arms = arms or ACTIVE
    specs = {a: FLEET[a] for a in arms}
    t0 = time.time()
    pre = prefilter(strokes, arms, atlas_dir)
    t_pre = time.time() - t0

    ivmap, probe_stats = {}, []
    t1 = time.time()
    for st in strokes:
        ivs = []
        for a in arms:
            if pre is not None and not pre.get((st["id"], a), True):
                probe_stats.append(dict(arm=a, probes=0, statuses=["prefiltered"],
                                        stroke=st["id"]))
                continue
            v, s = probe_stroke(st["pts"], specs[a], opts, min_seg=min_seg)
            s["stroke"] = st["id"]
            probe_stats.append(s)
            ivs += v
        ivmap[st["id"]] = ivs
        if verbose:
            print(f"  stroke {st['id']:3d} {st['color']:6s} "
                  f"L={polyline_length(st['pts']):.3f} m -> "
                  + (", ".join(f"{v.arm}[{v.s0:.2f},{v.s1:.2f}]" for v in ivs)
                     or "no arm"))
    t_probe = time.time() - t1

    colors, cover, table = best_partition(strokes, ivmap, arms, min_seg)

    # ---- clean re-plans -------------------------------------------------
    t2 = time.time()
    programs = {a: [] for a in arms}
    n_replan = 0
    for ps in cover["per_stroke"]:
        st, L = ps["stroke"], ps["L"]
        for span in place_cuts(ps["chosen"], L, overlap):
            plan, sp = replan_segment(st["pts"], span, specs[span["arm"]], opts,
                                      min_seg=min_seg)
            n_replan += sp["replans"]
            if plan is None:
                continue
            pts = _segment_points(st["pts"], sp["s0"], sp["s1"], sp["direction"])
            programs[span["arm"]].append(dict(
                stroke_id=st["id"], color=st["color"], kind=st.get("kind", ""),
                s_range=(float(sp["s0"]), float(sp["s1"])),
                direction=int(sp["direction"]), pts=pts,
                length=float(polyline_length(pts)), plan=plan))
    t_replan = time.time() - t2

    dropped = leftover(strokes, programs)
    out = dict(colors=colors, arms=arms, table=table, ivmap=ivmap,
               probe_stats=probe_stats, programs={}, dropped=dropped,
               timing=dict(prefilter=t_pre, probe=t_probe, replan=t_replan,
                           total=time.time() - t0))
    for a in arms:
        segs = programs[a]
        order, transit = order_nearest(segs, FLEET[a].xy) if segs else ([], 0.0)
        out["programs"][a] = [segs[i] for i in order]
        out.setdefault("transit", {})[a] = transit
    out["total_len"] = float(sum(polyline_length(s["pts"]) for s in strokes))
    out["drawn_len"] = float(sum(s["length"] for a in arms
                                 for s in out["programs"][a]))
    out["dropped_len"] = float(sum(d["length"] for d in dropped))
    out["n_probes"] = int(sum(p["probes"] for p in probe_stats))
    out["n_replans"] = n_replan
    return out


def report(res, strokes):
    """Terse allocation report -> list of printable lines."""
    lines = []
    tot = res["total_len"]
    lines.append(f"strokes {len(strokes)}   traced {tot:.2f} m   "
                 f"drawn {res['drawn_len']:.2f} m   "
                 f"dropped {res['dropped_len']:.2f} m "
                 f"({100 * res['dropped_len'] / max(tot, 1e-9):.1f} %)")
    lines.append(f"{'arm':>5} {'pen':>7} {'segs':>5} {'metres':>8} "
                 f"{'strokes':>8} {'cuts':>5} {'transit':>8}")
    for a in res["arms"]:
        segs = res["programs"][a]
        ids = {s["stroke_id"] for s in segs}
        cuts = len(segs) - len(ids)
        lines.append(f"{a:>5} {res['colors'][a]:>7} {len(segs):>5} "
                     f"{sum(s['length'] for s in segs):>8.2f} {len(ids):>8} "
                     f"{cuts:>5} {res['transit'][a]:>8.2f}")
    st = [p for p in res["probe_stats"]]
    lines.append(f"probes {res['n_probes']} over {len(st)} (stroke, arm) pairs, "
                 f"{sum(1 for p in st if p['probes'] == 0)} prefiltered; "
                 f"clean re-plans {res['n_replans']}")
    pieces = {}
    for a in res["arms"]:
        for s in res["programs"][a]:
            pieces.setdefault(s["stroke_id"], []).append(a)
    multi = {k: v for k, v in pieces.items() if len(v) > 1}
    handoff = sum(1 for v in multi.values() if len(set(v)) > 1)
    lines.append(f"{len(pieces)} strokes drawn, {len(multi)} of them in more than "
                 f"one piece ({handoff} handed between two arms); "
                 f"colour partition = "
                 + " ".join(f"{a}:{res['colors'][a]}" for a in res["arms"]))
    whole = sum(1 for d in res["dropped"] if d["s_range"] == (0.0, 1.0))
    lines.append(f"dropped {len(res['dropped'])} spans ({whole} whole strokes); "
                 "by third of the sheet, metres left empty:")
    from .fleet import SHEET
    cells = where(res["dropped"], SHEET)
    for iy in (2, 1, 0):
        lines.append("      " + "  ".join(
            f"{cells.get((ix, iy), 0.0):5.2f}" for ix in range(3))
            + ("   <- top" if iy == 2 else "   <- bottom" if iy == 0 else ""))
    t = res["timing"]
    lines.append(f"time  prefilter {t['prefilter']:.1f} s  probe {t['probe']:.1f} s"
                 f"  replan {t['replan']:.1f} s  total {t['total']:.1f} s")
    return lines
