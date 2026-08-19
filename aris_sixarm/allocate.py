"""Strokes + fleet -> per-arm programs of CERTIFIED segments (and what was dropped).

The planner's contract (`stroke_api.plan_stroke`) is certified-or-split: one
call either returns a plan an independent validator has accepted, or the arc
length `s_star` up to which it could.  That makes reach a *measured* quantity
rather than a modelled one, and this module is the consumer that contract was
written for.

  PROBE      For each (stroke, arm) a few plan calls buy the arm's feasible
             s-intervals on that stroke: forward gives [0, s*], the reversed
             stroke gives [1-s*_rev, 1] (direction matters — the DP walks the
             redundancy band from wherever it starts), and every probe after
             those two takes the largest remaining gap and plans it, each way
             round, until the budget runs out.  Everything a probe returns is
             certified; nothing here extrapolates a plan.  Each arm is probed
             with ITS OWN pen (`pens`, see `pen_opts`) — a fleet holding three
             different pen lengths is three different reach envelopes.
  PARTITION  Each arm carries ONE pen for the whole piece, so a grey stroke
             can only be covered by grey arms.  With four active arms there
             are 2^4 - 2 = 14 non-trivial colour partitions, so the choice is
             made by enumeration rather than by heuristic: minimise dropped
             length, then the spread of per-arm drawing length.  `colors=`
             fixes the partition instead, which is what a TWO-PASS piece needs:
             the constraint is one pen per arm per PHASE, and a run that stops
             for a human to swap the pens is two single-colour problems in
             which every arm is available for both.
  COVER      Per stroke, greedy interval covering over the intervals of the
             arms with the right pen — which is optimal for "fewest pieces",
             i.e. fewest pen-up handoffs in the middle of a line.  Ties go to
             the least-loaded arm.
  BALANCE    The cover is greedy per stroke and blind to the clock: on the
             CSAIL logo it hands arm 97 four and a half of the seven and a half
             orange metres and leaves three arms drawing 0.6 m each, so the
             phase takes as long as arm 97 does however well it is conducted.
             `balance_loads` is the repair — a greedy improvement pass that
             moves (and swaps) segments MORE THAN ONE ARM ALREADY CERTIFIES
             from the busiest arm to a less-loaded one, scored on the real
             per-arm nominal clock (`writing.segment_draw_time` for the ink,
             the sequencer's own tour cost for the pen-ups).  It never changes
             WHAT is drawn, only who draws it, so coverage is invariant.
  REPAIR     The cover's own gaps are the first statement of what is missing in
             the terms that matter — the UNION of the arms carrying the right
             ink — and `probe_stroke` never saw them, because it only ever knew
             one arm's coverage.  So each hole is offered back to every arm of
             that colour, both ways round, in a window widened to something
             `plan_stroke` will accept.  Whatever still will not certify is
             DROPPED and reported; nothing is moved to make it fit.
  CUT        A handoff is placed in the MIDDLE of the overlap between the two
             intervals, not at either edge, and both segments are grown by
             `overlap` metres past it so the ink meets.  Cutting at an edge
             would put the seam exactly where one of the two arms is at the
             end of its certified reach.
  ORDER      `sequence.py`: the order AND the direction of every segment,
             minimising the arm's total pen-up TIME (the real hover transit
             `writing` will execute, not paper distance).  Exact for up to 16
             segments.  The transit MOTION is still roadmap item 3's RRT; what
             is optimised here is the schedule that motion has to fill.

Every chosen segment is re-planned from scratch at the end; a segment whose
clean re-plan is not "ok" is not shipped as one.
"""
import time
from dataclasses import dataclass

import numpy as np

from . import sequence, writing
from .fleet import FLEET, H_INV_DEFAULT
from .stroke_api import (plan_stroke, polyline_length, reverse_plan,
                         truncate_polyline)

ACTIVE = [aid for aid, s in FLEET.items() if s.active]
COLORS = ("grey", "orange")
DRAW_SPEED = writing.DRAW_SPEED_FLEET    # m/s the material allows; a CAP, and
#   the load model's — see `writing.draw_duration`, which stretches it wherever
#   the redundancy resolution asks a joint to move faster than it may.
BALANCE_ROUNDS = 200                     # accepted moves the balancer may make


def active_arms(active_override=None):
    """Which arms this run may use -> list of arm ids, in registry order.

    `fleet.FLEET`'s `active` flags are a record of TODAY'S RIG (arms 2 and 71
    are parked), so a what-if run must not rewrite them.  `active_override` is
    that what-if, applied on top of the registry and nowhere else:

      None              the registry's own flags (the real fleet)
      "all"             every arm in the fleet, parked or not
      iterable of ids   exactly those arms
      {arm_id: bool}    the registry flags with those entries patched

    Raises on an id the fleet does not contain, because silently allocating to
    five arms when six were asked for is the kind of thing that only shows up
    in a coverage number nobody re-derives.
    """
    if active_override is None:
        return list(ACTIVE)
    if isinstance(active_override, str):
        if active_override != "all":
            raise ValueError(f"active_override={active_override!r}; want 'all', "
                             "a list of arm ids, or a {arm_id: bool} mapping")
        return list(FLEET)
    if isinstance(active_override, dict):
        flags = {aid: s.active for aid, s in FLEET.items()}
        unknown = set(active_override) - set(FLEET)
        if unknown:
            raise ValueError(f"active_override names arms not in the fleet: "
                             f"{sorted(unknown)}")
        flags.update({a: bool(v) for a, v in active_override.items()})
        return [aid for aid in FLEET if flags[aid]]
    want = list(active_override)
    unknown = set(want) - set(FLEET)
    if unknown:
        raise ValueError(f"active_override names arms not in the fleet: "
                         f"{sorted(unknown)}")
    return [aid for aid in FLEET if aid in set(want)]

def pen_opts(opts, pens, arm):
    """`opts` with THIS ARM's pen length substituted. -> a fresh dict.

    `pen_ext` is a plumbing parameter of `stroke_api.plan_stroke` all the way
    down (lattice, dense certification, independent validator), so a fleet in
    which arm 31 holds a 300 mm pen and arm 97 a 110 mm one is not a new
    planner — it is a different `opts` per arm.  `pens` is {arm_id: metres};
    an arm it does not name keeps whatever `opts` said (i.e. `frames.PEN_EXT`).

    Everything downstream of a probe MUST be given the same dict: a segment
    certified with a 300 mm pen and re-planned with a 110 mm one is a different
    stroke for a different tool, and the second plan's certificate would be
    about a robot that is not the one drawing.
    """
    o = dict(opts or {})
    if pens and arm in pens:
        o["pen_ext"] = float(pens[arm])
    return o


def pen_of(pens, arm, default=None):
    """The pen length arm `arm` is holding, in metres."""
    from .frames import PEN_EXT
    if pens and arm in pens:
        return float(pens[arm])
    return float(PEN_EXT if default is None else default)


EPS_S = 1e-6
SEQUENCER = "opt"        # "opt" = minimum transit time; "nn" = the old xy chain
OVERLAP_M = 0.004        # m of ink each side of a handoff cut
BACKOFF_M = 0.006        # m to give up per failed clean re-plan
MIN_SEG_M = 0.025        # m; a shorter piece is not worth a pen-up
GAP_TOL_M = 0.002        # m; below this a hole in the ink is not a hole
PREFILTER_R = 0.05       # m; atlas cell distance that still counts as maybe

# MIN_SEG and GAP_TOL are not the same number and used to be conflated.  What
# the fleet will be ASKED TO DRAW has a floor (a 5 mm pen-down between two
# pen-ups is not a segment); what counts as LEFT EMPTY does not get to inherit
# that floor, because `leftover` — the function that decides what nobody drew —
# has always reported every hole longer than 2 mm.  Covering with a 25 mm gap
# tolerance and reporting with a 2 mm one is how a run reaches "100 %" with
# 1 cm of bare paper in it.  So the two floors are separate: gaps are chased
# down to GAP_TOL, and a gap too short to plan is chased by WIDENING the probe
# window into ink its neighbours already cover, not by giving up on it.


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


def probe_stroke(pts, spec, opts=None, max_probes=3, min_seg=MIN_SEG_M,
                 gap_tol=GAP_TOL_M):
    """Up to `max_probes` plan calls -> (intervals, stats) for one (stroke, arm).

    Intervals are in the stroke's own normalised arc length and carry the
    direction they were certified in, because a plan is a walk through the
    redundancy band and the band is not symmetric: an arm can often draw the
    last 60 % of a line end-to-start that it cannot reach start-to-end.

    The first two probes are the whole stroke each way round; every probe after
    that takes the LARGEST REMAINING GAP and plans it — forward, and if that
    does not carry the gap to its far end, backward as well.  A partial result
    shrinks the gap rather than closing it, so the next probe picks up where
    this one stopped and the budget walks along a stroke that is reachable in
    pieces.  Three probes reproduce the original behaviour exactly; the budget
    only ever buys more certified interval, never a weaker certificate, because
    every interval here is a span some `plan_stroke` call returned as planned.
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

    if not (lo > 0 or hi < 1 or iv):
        # with no interval and no resume hint a gap probe would just repeat the
        # first one on the same polyline
        return iv, stats
    eps = min_seg / max(L, 1e-9)
    tried = set()
    while stats["probes"] < max_probes:
        gaps = [(max(g[0], lo), min(g[1], hi)) for g in uncovered(iv, min_gap=eps)]
        gaps = [g for g in gaps if (g[1] - g[0]) * L >= min_seg
                and (round(g[0], 6), round(g[1], 6)) not in tried]
        if not gaps:
            break
        a, b = max(gaps, key=lambda g: g[1] - g[0])
        tried.add((round(a, 6), round(b, 6)))
        sub = truncate_polyline(pts, a, b)
        if len(sub) < 2 or polyline_length(sub) < min_seg:
            continue
        w = b - a
        r3 = plan_stroke(sub, spec, opts)
        stats["probes"] += 1
        stats["statuses"].append(r3["status"])
        s3, st3 = _certified_span(r3)
        if st3 in ("ok", "split") and s3 * w * L >= min_seg:
            iv.append(Interval(a, a + s3 * w, spec.arm_id, +1, "gap"))
        if s3 >= 1.0 - EPS_S or stats["probes"] >= max_probes:
            continue
        # the gap's far end is still open: the band may be walkable from that
        # side even though it is not from this one
        r4 = plan_stroke(sub[::-1], spec, opts)
        stats["probes"] += 1
        stats["statuses"].append(r4["status"])
        s4, st4 = _certified_span(r4)
        if st4 in ("ok", "split") and s4 * w * L >= min_seg:
            iv.append(Interval(b - s4 * w, b, spec.arm_id, -1, "gap_rev"))
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
    (and therefore drop every stroke of that colour) are dropped, leaving
    2^n - 2: 14 for the four arms of today's rig, 62 for all six.
    """
    out = []
    for bits in range(1 << len(arms)):
        cols = [COLORS[(bits >> i) & 1] for i in range(len(arms))]
        if nontrivial and len(set(cols)) < 2:
            continue
        out.append({a: c for a, c in zip(arms, cols)})
    return out


def cover_all(strokes, ivmap, colors, min_seg=MIN_SEG_M, gap_tol=GAP_TOL_M):
    """Cover every stroke using only arms whose pen matches -> dict of results.

    The frontier is chased to `gap_tol`, not to `min_seg`: what is left empty is
    measured the way `leftover` measures it, so this function's `dropped_len`
    and the shipped programme's disagree only by what the clean re-plan gives
    back, and never by an accounting convention.
    """
    load = {a: 0.0 for a in colors}
    per_stroke, dropped = [], []
    for st in strokes:
        L = polyline_length(st["pts"])
        ivs = [v for v in ivmap.get(st["id"], []) if colors.get(v.arm) == st["color"]]
        min_gap = gap_tol / max(L, 1e-9)
        chosen, gaps = greedy_cover(ivs, load, min_gap=min_gap)
        for v in chosen:
            load[v.arm] += (v.s1 - v.s0) * L
        per_stroke.append(dict(stroke=st, chosen=chosen, gaps=gaps, L=L))
        dropped += [dict(stroke_id=st["id"], color=st["color"], s0=a, s1=b,
                         length=(b - a) * L) for a, b in gaps]
    return dict(per_stroke=per_stroke, dropped=dropped, load=load,
                dropped_len=float(sum(d["length"] for d in dropped)),
                cuts=int(sum(max(len(p["chosen"]) - 1, 0) for p in per_stroke)))


REPAIR_BUDGET = 600      # plan calls the repair pass may spend on one problem


def repair_gaps(strokes, ivmap, colors, cover, specs, aopts, min_seg=MIN_SEG_M,
                gap_tol=GAP_TOL_M, rounds=3, min_len=0.02,
                budget=REPAIR_BUDGET, verbose=False):
    """Ask every eligible arm about exactly the spans the cover could not fill.

    `probe_stroke` is myopic in the way a per-(stroke, arm) routine has to be:
    it walks the gaps in ONE ARM'S OWN coverage, and the gap that matters is a
    gap in the UNION of the arms that carry the right ink.  Those are different
    sets, and the second one is only known after `cover_all` has run.  So this
    is the pass that closes the loop: take the cover's own gap list, and for
    each hole offer it to every arm of that colour, both ways round.

    Two things make it find ink the first pass could not:

      WIDENING  a hole shorter than `plan_stroke`'s `min_length` is not a
                stroke anybody can be asked to plan, so the WINDOW is grown
                symmetrically into ink the neighbours already cover.  Refusing
                to grow it is why a 10 mm hole used to be uncloseable by an arm
                that reaches straight over it — the last 10 mm of a 16 m logo
                is exactly the distance between 99.9 % and 100 %.
      DIRECTION every window is offered forwards and backwards, because the DP
                walks the redundancy band from wherever it starts and the band
                is not symmetric.

    Everything appended to `ivmap` is a span some `plan_stroke` call returned as
    planned; this pass adds no metre the planner did not certify.  Iterated
    until a round adds nothing, `rounds` is spent, or `budget` plan calls are:
    a placement with fifty holes is a placement to reject, not one to spend an
    unbounded search on, and the budget keeps a sweep over thousands of
    candidates from being priced by its worst member.  Holes are taken LARGEST
    FIRST so a budget that runs out has been spent on the metres that matter.
    -> (cover, n_probes).
    """
    by_id = {st["id"]: st for st in strokes}
    n_probe, added_total = 0, 0
    for _ in range(max(int(rounds), 0)):
        holes = [(ps["stroke"], a, b) for ps in cover["per_stroke"]
                 for a, b in ps["gaps"]]
        holes.sort(key=lambda h: -(h[2] - h[1]) * polyline_length(h[0]["pts"]))
        if not holes:
            break
        added = 0
        for st, a, b in holes:
            if n_probe >= budget:
                break
            L = polyline_length(st["pts"])
            need = max(min_seg, 2.0 * min_len) / max(L, 1e-9)
            if b - a < need:
                pad = 0.5 * (need - (b - a))
                a, b = max(0.0, a - pad), min(1.0, b + pad)
                if b - a < need:                  # ran into an end of the stroke
                    a, b = ((0.0, min(1.0, need)) if a <= 0.0
                            else (max(0.0, 1.0 - need), 1.0))
            sub = truncate_polyline(st["pts"], a, b)
            if len(sub) < 2 or polyline_length(sub) < min_len:
                continue
            w, frac = b - a, 0.0
            for arm in specs:
                if colors.get(arm) != st["color"] or n_probe >= budget:
                    continue
                for pts_dir, sign in ((sub, +1), (sub[::-1], -1)):
                    r = plan_stroke(pts_dir, specs[arm], aopts[arm])
                    n_probe += 1
                    frac, status = _certified_span(r)
                    if status not in ("ok", "split") or frac * w * L < min_seg:
                        continue
                    iv = (Interval(a, a + frac * w, arm, +1, "repair") if sign > 0
                          else Interval(b - frac * w, b, arm, -1, "repair_rev"))
                    ivmap.setdefault(st["id"], []).append(iv)
                    added += 1
                    if frac >= 1.0 - EPS_S:
                        break
                if frac_covers(ivmap.get(st["id"], []), a, b):
                    break
        if verbose:
            print(f"  repair: {len(holes)} holes, {added} certified spans added")
        added_total += added
        if not added:
            break
        cover = cover_all(strokes, ivmap, colors, min_seg, gap_tol)
    cover["repair_probes"] = n_probe
    cover["repair_added"] = added_total
    return cover, n_probe


def frac_covers(intervals, a, b, tol=EPS_S):
    """Do these intervals already cover [a, b] with no hole?"""
    return not uncovered([v for v in intervals], lo=a, hi=b, min_gap=tol)


def best_partition(strokes, ivmap, arms=None, min_seg=MIN_SEG_M,
                   gap_tol=GAP_TOL_M):
    """Enumerate the 2^n - 2 partitions -> (best colors, best cover, table)."""
    arms = arms or ACTIVE
    table = []
    for colors in partitions(arms):
        c = cover_all(strokes, ivmap, colors, min_seg, gap_tol)
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


def _entry(st, sp, plan):
    """One programme entry from a re-planned span. -> dict."""
    pts = _segment_points(st["pts"], sp["s0"], sp["s1"], sp["direction"])
    return dict(stroke_id=st["id"], color=st["color"], kind=st.get("kind", ""),
                s_range=(float(sp["s0"]), float(sp["s1"])),
                direction=int(sp["direction"]), pts=pts,
                length=float(polyline_length(pts)), plan=plan)


def replan_same_span(st, sp, spec, opts=None, min_seg=MIN_SEG_M, tol=1e-12):
    """Re-plan EXACTLY this span for another arm. -> (entry, n_plan_calls).

    The entry is None unless the arm's clean re-plan certifies the span end to
    end — `replan_segment` is allowed to give ground and this is the one caller
    that refuses to take it, because the whole point of a balancing move is
    that the ink does not change.  Both directions are offered, because a plan
    is a walk of the redundancy band and the band is not symmetric.
    """
    n = 0
    for d in (int(sp["direction"]), -int(sp["direction"])):
        plan, s2 = replan_segment(st["pts"], dict(sp, direction=d), spec, opts,
                                  min_seg=min_seg)
        n += int(s2["replans"])
        if (plan is not None and s2["lost"] <= tol
                and abs(s2["s0"] - sp["s0"]) <= tol
                and abs(s2["s1"] - sp["s1"]) <= tol):
            return _entry(st, s2, plan), n
    return None, n


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
# 4b. load balancing: who draws it, when more than one arm may
# ===========================================================================
# THE COVER IS OPTIMAL FOR THE WRONG THING.  `greedy_cover` minimises pieces
# per stroke, which is the right first objective (a piece is a pen-up, a
# handoff and a visible seam) and says nothing at all about the clock.  A phase
# ends when its LAST arm stops, so what the makespan pays for is the maximum
# per-arm programme, and on the CSAIL logo's orange pass the greedy hands arm
# 97 4.52 of the 7.45 m — 88.0 s of ink and pen-up against 18.0, 27.7 and
# 34.4 s for the other three.  No conductor can recover that: 88.0 s is the
# phase's floor, and it was set by an allocation that never looked at a clock.
#
# What follows is the smallest honest repair.  Minimum-makespan scheduling with
# machine eligibility (R|M_j|C_max) is NP-hard, so this is not an optimiser: it
# is a first-order greedy that repeatedly takes the BEST single move or swap
# out of the busiest arm and stops when none improves.  Two properties are
# worth more here than optimality:
#
#   COVERAGE IS INVARIANT.  A move only ever re-assigns a span that the
#   receiving arm has ALREADY certified at exactly the same endpoints — the
#   candidate is re-planned from scratch for that arm and refused unless its
#   clean re-plan gives back not one millimetre (`lost == 0`).  The set of
#   drawn spans is therefore identical before and after, and "does the balancer
#   cost coverage" is not a measurement, it is a type.
#
#   THE SCORE IS THE TIMELINE'S.  `load_fn` prices an arm's programme as
#   `writing.segment_draw_time` per segment plus the sequencer's own optimal
#   tour cost over exactly those segments — which is `writing.arm_program`'s
#   `duration` to the float, not a proxy for it.  A balancer optimising a
#   different clock from the one the fleet runs on would be worse than none.
def _slice_ends(ends, k):
    """`sequence.endpoints` restricted to segment positions `k`, in that order."""
    k = list(k)
    return dict(q=ends["q"][k], xy=ends["xy"][k], hover=ends["hover"][k],
                z=ends["z"][k], n=len(k))


def arm_load(spec, segs, draw_s, transit_speed=writing.TRANSIT_SPEED,
             qd_frac=writing.QD_FRAC, h_inv=H_INV_DEFAULT,
             pen_ext=None, ends=None, exact_max_n=sequence.EXACT_MAX_N,
             budget=sequence.TIME_BUDGET):
    """Nominal seconds one arm needs for `segs`: its ink plus its best pen-up tour.

    `draw_s` is the per-segment ink time in the same order as `segs`.  The
    pen-up half is `sequence.solve` on the same cost matrix the real sequencing
    pass uses, so this is not an estimate of the arm's programme — it is the
    programme, costed before it is committed to.
    """
    if not len(segs):
        return 0.0
    pen = writing.PEN_EXT if pen_ext is None else float(pen_ext)
    C = sequence.cost_matrix(spec, segs, transit_speed, qd_frac, h_inv,
                             ends=ends, pen_ext=pen)
    r = sequence.solve(C, len(segs), exact_max_n, budget)
    return float(sum(draw_s) + r["cost"])


def balance_loads(owner, options, load_fn, max_rounds=BALANCE_ROUNDS,
                  verbose=False):
    """Greedy min-max load balancing over segments several arms can certify.

    `owner[i]` is the arm currently drawing segment i; `options[i]` is the set
    of arms that can draw it AT THE SAME SPAN (`owner[i]` included);
    `load_fn(arm, (i, j, ...)) -> seconds` prices one arm's whole programme.
    -> (owner, info) with `info` carrying the loads before and after and every
    move taken.

    THE OBJECTIVE IS THE MAXIMUM LOAD, and the tie-break is the sum of squares.
    The tie-break is not decoration: on a plateau — several moves that leave the
    busiest arm exactly where it is — flattening the rest is what makes the NEXT
    move able to lower the maximum, and a pure max objective stalls on the first
    one.  Both are compared lexicographically and strictly, so the potential
    falls at every accepted move; with finitely many assignments that is the
    termination proof, and `max_rounds` is a belt on top of it.

    Each round considers every single-segment RELOCATION out of the busiest arm
    and every SWAP of one of its segments against one held by another arm, and
    takes the best.  The busiest arm is the only source worth considering
    because it is the only arm whose load is the objective; a swap is worth
    considering separately from two moves because the pair can be admissible
    when neither half is.
    """
    owner = list(owner)
    options = [set(o) for o in options]
    for i, (a, o) in enumerate(zip(owner, options)):
        if a not in o:
            raise ValueError(f"segment {i} is drawn by arm {a}, which is not "
                             f"among the arms that certify it ({sorted(o)})")
    arms = sorted(set(owner) | {a for o in options for a in o})
    cache = {}

    def load(a, own):
        key = (a, tuple(i for i, x in enumerate(own) if x == a))
        if key not in cache:
            cache[key] = float(load_fn(a, key[1]))
        return cache[key]

    def score(own):
        L = {a: load(a, own) for a in arms}
        v = sorted(L.values(), reverse=True)
        return (round(v[0], 9) if v else 0.0,
                round(float(sum(x * x for x in v)), 6)), L

    key, L = score(owner)
    info = dict(loads_before=dict(L), max_before=key[0], moves=[], rounds=0,
                n_movable=int(sum(1 for o in options if len(o) > 1)))
    for _ in range(int(max_rounds)):
        src = max(arms, key=lambda a: (L[a], a))
        mine = [i for i, x in enumerate(owner) if x == src]
        best = None
        for i in mine:
            for b in sorted(options[i] - {src}):
                cand = list(owner)
                cand[i] = b
                k, _ = score(cand)
                if best is None or k < best[0]:
                    best = (k, cand, dict(kind="move", seg=i, frm=src, to=b))
        for i in mine:
            for j, b in enumerate(owner):
                if b == src or b not in options[i] or src not in options[j]:
                    continue
                cand = list(owner)
                cand[i], cand[j] = b, src
                k, _ = score(cand)
                if best is None or k < best[0]:
                    best = (k, cand, dict(kind="swap", seg=i, other=j,
                                          frm=src, to=b))
        if best is None or not best[0] < key:
            break
        owner, key = best[1], best[0]
        _, L = score(owner)
        best[2].update(max_after=key[0])
        info["moves"].append(best[2])
        info["rounds"] += 1
        if verbose:
            m = best[2]
            print(f"  balance {info['rounds']:>2}: {m['kind']} segment "
                  f"{m['seg']} arm {m['frm']} -> arm {m['to']}"
                  + (f" (against segment {m['other']})" if "other" in m else "")
                  + f"; busiest arm now {key[0]:.1f} s")
    info.update(loads_after=dict(L), max_after=key[0], n_loads=len(cache))
    return owner, info


def rebalance(placed, arms, colors, ivmap, specs, aopts, pens,
              draw_speed=DRAW_SPEED, seq_opts=None, min_seg=MIN_SEG_M,
              h_inv=H_INV_DEFAULT, verbose=False):
    """The probe data + the placed spans -> a re-assignment. -> (placed, info).

    Three steps, in the order that keeps the planner honest:

      1. ALTERNATIVES.  For every placed span, the arms of the right colour
         whose certified intervals already cover it end to end (`frac_covers`
         over `ivmap` — the probe data, not a new guess), each offered a clean
         re-plan of exactly that span and kept only if it certifies all of it.
      2. PRICE.  Per (arm, span) the ink seconds; per arm the hover poses of
         its whole candidate pool once, so a load is a cost-matrix slice and a
         Held-Karp rather than a fresh IK sweep.
      3. BALANCE.  `balance_loads` on the result.

    `placed` entries are mutated in place (`arm` and `entry`), which is what the
    caller then reads its programmes out of.
    """
    seq = dict(seq_opts or {})
    ts = float(seq.get("transit_speed", writing.TRANSIT_SPEED))
    qf = float(seq.get("qd_frac", writing.QD_FRAC))
    exact = int(seq.get("exact_max_n", sequence.EXACT_MAX_N))
    budget = float(seq.get("budget", sequence.TIME_BUDGET))

    options, entries, n_probe = [], [], 0
    for it in placed:
        st, sp, own = it["stroke"], it["sp"], it["arm"]
        opt, ent = {own}, {own: it["entry"]}
        for b in arms:
            if b == own or colors.get(b) != st["color"]:
                continue
            ivs = [v for v in ivmap.get(st["id"], []) if v.arm == b]
            if not frac_covers(ivs, sp["s0"], sp["s1"]):
                continue
            e, n = replan_same_span(st, sp, specs[b], aopts[b], min_seg)
            n_probe += n
            if e is not None:
                opt.add(b)
                ent[b] = e
        options.append(opt)
        entries.append(ent)

    pool = {a: [i for i, o in enumerate(options) if a in o] for a in arms}
    at = {a: {i: k for k, i in enumerate(pool[a])} for a in arms}
    ends, draw_s = {}, {}
    for a in arms:
        if not pool[a]:
            continue
        segs = [entries[i][a] for i in pool[a]]
        ends[a] = sequence.endpoints(specs[a], segs, h_inv, pens[a])
        for i, s in zip(pool[a], segs):
            draw_s[(a, i)] = writing.segment_draw_time(specs[a], s, draw_speed,
                                                       qf, h_inv, pens[a])

    def load_fn(a, idx):
        if not idx:
            return 0.0
        return arm_load(specs[a], [entries[i][a] for i in idx],
                        [draw_s[(a, i)] for i in idx], ts, qf, h_inv, pens[a],
                        ends=_slice_ends(ends[a], [at[a][i] for i in idx]),
                        exact_max_n=exact, budget=budget)

    owner0 = [it["arm"] for it in placed]
    owner, info = balance_loads(owner0, options, load_fn, verbose=verbose)
    for i, (it, a) in enumerate(zip(placed, owner)):
        it["arm"], it["entry"] = a, entries[i][a]
    info.update(n_replans=n_probe, draw_speed=float(draw_speed),
                metres_before={a: float(sum(entries[i][owner0[i]]["length"]
                                            for i in range(len(placed))
                                            if owner0[i] == a)) for a in arms},
                metres_after={a: float(sum(placed[i]["entry"]["length"]
                                           for i in range(len(placed))
                                           if owner[i] == a)) for a in arms})
    return placed, info


# ===========================================================================
# 5. ordering
# ===========================================================================
def order_nearest(items, start_xy):
    """Nearest-neighbour chaining from `start_xy` -> (order, transit metres).

    THE OLD ORDER, kept as the baseline the sequencer is measured against.
    Greedy in PAPER DISTANCE between one segment's last point and the next
    one's first, every segment drawn the way it was certified.  Both of those
    are wrong in the same direction: the arm pays joint-space seconds, not
    metres of paper, and it may draw a segment either way round (see
    `sequence.py`).  `--sequencer nn` still selects it, so old and new can be
    run through the identical downstream pipeline.
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


def transit_metres(items, start_xy):
    """Pen-tip paper distance of an ordered programme, base to last stroke end."""
    pos = np.asarray(start_xy, float)
    tot = 0.0
    for s in items:
        p = np.asarray(s["pts"], float)
        tot += float(np.linalg.norm(p[0] - pos))
        pos = p[-1]
    return tot


def sequence_arm(segs, spec, sequencer=SEQUENCER, opts=None, seq_opts=None):
    """One arm's bag of segments -> the programme in the order it will be drawn.

    -> dict(programme, order, dirs, method, cost, baseline_cost, n_reversed,
            n_refused, n, wall).  `cost` and `baseline_cost` are both transit
    SECONDS off the same matrix (`sequence.cost_matrix`), which is the point:
    "the new order saves X %" is then one subtraction inside one model, not a
    comparison of two different accountings.
    """
    n = len(segs)
    t0 = time.time()
    if n == 0:
        return dict(programme=[], order=[], dirs=[], method="empty", n=0,
                    cost=0.0, baseline_cost=0.0, n_reversed=0, n_refused=0,
                    wall=0.0)
    seq_opts = dict(seq_opts or {})
    mat = {k: seq_opts[k] for k in ("transit_speed", "qd_frac", "h_inv",
                                    "pen_ext") if k in seq_opts}
    exact = seq_opts.get("exact_max_n", sequence.EXACT_MAX_N)
    budget = seq_opts.get("budget", sequence.TIME_BUDGET)

    base_order, _ = order_nearest(segs, spec.xy)
    C = sequence.cost_matrix(spec, segs, **mat)
    base_cost = sequence.sequence_cost(C, n, base_order, [1] * n)
    if sequencer in ("nn", "nearest_xy"):
        r = dict(order=base_order, dirs=[1] * n, method="nearest_xy")
    elif sequencer in ("opt", "transit"):
        r = sequence.solve(C, n, exact, budget)
    else:
        raise ValueError(f"sequencer={sequencer!r}; want 'opt' or 'nn'")

    prog, dirs, n_rev, n_ref = [], [], 0, 0
    for k, d in zip(r["order"], r["dirs"]):
        rev = reverse_segment(segs[k], spec, opts) if d < 0 else None
        if d < 0 and rev is None:
            n_ref += 1
        if rev is None:
            prog.append(dict(segs[k], flipped=False))
            dirs.append(1)
        else:
            prog.append(rev)
            dirs.append(-1)
            n_rev += 1
    return dict(programme=prog, order=[int(i) for i in r["order"]], dirs=dirs,
                method=r["method"], n=n, n_reversed=n_rev, n_refused=n_ref,
                cost=float(sequence.sequence_cost(C, n, r["order"], dirs)),
                baseline_cost=float(base_cost),
                nn_cost=float(r.get("nn_cost", float("nan"))),
                wall=float(time.time() - t0))


def reverse_segment(seg, spec, opts=None):
    """A programme entry drawn the other way round, or None if it will not certify.

    The geometry is flipped (both the drawn polyline and the certified plan);
    `direction` is the direction the segment is EXECUTED in, so it changes
    sign, while `s_range` keeps naming the same span of the same stroke.
    `stroke_api.reverse_plan` re-runs the independent validator, and a plan
    that somehow does not pass it is refused here rather than shipped — the
    caller keeps the forward orientation and pays the extra transit.
    """
    rev = reverse_plan(seg["plan"], spec, opts)
    if rev.get("status") != "ok":
        return None
    return dict(seg, pts=np.asarray(seg["pts"], float)[::-1].copy(), plan=rev,
                direction=-int(seg["direction"]), flipped=True)


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
def atlas_cells(arms, atlas_dir):
    """-> {arm: (grid_m, {(ix, iy)})}, or None if any arm's atlas is missing.

    The atlas is a 2 cm sweep of the paper; a cell counts if the pen reached it
    PERPENDICULAR (tilt 0 — the planner never leans the pen) with the planner's
    own permissive joint margin.

    `atlas_dir` may be ONE directory (every arm read from it) or a mapping
    {arm_id: directory}, which is what a per-arm pen assignment needs: the
    atlas is swept for a particular pen length, so an arm holding a 300 mm pen
    must be prefiltered against the 300 mm sweep and not the 110 mm one.
    """
    if atlas_dir is None:
        return None
    from .atlas import load
    per_arm = atlas_dir if isinstance(atlas_dir, dict) else None
    grids = {}
    for a in arms:
        d = per_arm.get(a) if per_arm is not None else atlas_dir
        if d is None:
            return None
        try:
            arr, meta = load(d, a)
        except Exception:
            return None
        g = float(meta["grid"])
        rows = arr[(arr[:, 8] <= 0.0) & (arr[:, 2] >= 0.15)]
        grids[a] = (g, {(int(round(x / g)), int(round(y / g)))
                        for x, y in rows[:, :2]})
    return grids


def _densify_xy(pts, ds=0.02):
    p = np.asarray(pts, float)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    t = np.concatenate([[0], np.cumsum(seg)])
    if t[-1] <= 0:
        return p[:1], np.zeros(1)
    s = np.arange(0, t[-1], ds)
    d = np.column_stack([np.interp(s, t, p[:, 0]), np.interp(s, t, p[:, 1])])
    d = np.vstack([d, p[-1]])
    w = np.full(len(d), ds)
    w[-1] = t[-1] - (len(d) - 1) * ds + ds
    return d, np.maximum(w, 0.0)


def reach_fraction(strokes, grids, radius=PREFILTER_R, ds=0.02):
    """Length-weighted fraction of the strokes within `radius` of a reachable
    atlas cell of ANY arm in `grids`.

    A CHEAP UPPER BOUND on coverage, not a coverage: the atlas says the pen can
    stand on that cell, not that a whole stroke through it can be planned as one
    certified walk of the redundancy band.  Good enough to RANK placements
    (hundreds of them, in the time one real allocation takes), which is all the
    placement search asks of it — the winner is then allocated for real.
    """
    k = int(np.ceil(radius / 0.02))
    every = sorted(set().union(*(c for _, c in grids.values())) if grids else ())
    if not every:
        return 0.0
    ij = np.array(every)
    grown = np.zeros((ij[:, 0].max() + 2 * k + 2, ij[:, 1].max() + 2 * k + 2), bool)
    grown[ij[:, 0] + k, ij[:, 1] + k] = True
    for _ in range(k):                       # dilate by k cells (Chebyshev)
        grown[1:] |= grown[:-1].copy()
        grown[:-1] |= grown[1:].copy()
        grown[:, 1:] |= grown[:, :-1].copy()
        grown[:, :-1] |= grown[:, 1:].copy()
    g = next(iter(grids.values()))[0]
    tot = cov = 0.0
    for st in strokes:
        d, w = _densify_xy(st["pts"], ds)
        ii = np.round(d[:, 0] / g).astype(int) + k
        jj = np.round(d[:, 1] / g).astype(int) + k
        ok = ((ii >= 0) & (jj >= 0) & (ii < grown.shape[0]) & (jj < grown.shape[1]))
        ok[ok] &= grown[ii[ok], jj[ok]]
        tot += float(w.sum())
        cov += float(w[ok].sum())
    return cov / max(tot, 1e-9)


def _pad_to(m, shape):
    out = np.zeros(shape, bool)
    s = tuple(slice(0, min(a, b)) for a, b in zip(m.shape, shape))
    out[s] = m[s]
    return out


def prefilter(strokes, arms, atlas_dir=None, radius=PREFILTER_R):
    """-> {(stroke_id, arm): True} where the atlas says probing is worth it.

    A stroke with no reachable cell near any of its points cannot be planned by
    that arm, so the probe is skipped.  Absent an atlas everything is probed.
    """
    grids = atlas_cells(arms, atlas_dir)
    if grids is None:
        return None
    ok = {}
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
             min_seg=MIN_SEG_M, overlap=OVERLAP_M, active_override=None,
             sequencer=SEQUENCER, seq_opts=None, pens=None, colors=None,
             max_probes=3, gap_tol=GAP_TOL_M, repair_rounds=3,
             repair_budget=REPAIR_BUDGET, balance=True,
             draw_speed=DRAW_SPEED):
    """Strokes -> per-arm certified programs + the dropped list.  See module docs.

    `arms` names the arms outright; `active_override` (see `active_arms`) says
    which of the fleet's arms count as active for THIS run without touching the
    registry, so a hypothetical "all six arms up" run and the real four-arm rig
    come out of the same entry point.

    `pens` is {arm_id: pen length in metres}: every plan call this function
    makes for that arm — probe, clean re-plan, and the sequencer's hover poses —
    is made with that pen (see `pen_opts`).  It is recorded in the result so a
    downstream stage cannot freeze a timeline for the wrong tool.

    `colors` FIXES the colour partition instead of enumerating it.  One pen per
    arm is a constraint WITHIN a drawing phase, not across phases: if the piece
    is drawn grey first and orange after a human swaps the pens, each phase is
    an independent single-colour sub-problem in which every arm is available,
    and the caller expresses that by calling this twice with
    `colors={a: "grey"}` and `colors={a: "orange"}` over the two stroke
    subsets.  Left None, the 2^n - 2 partitions are enumerated as before, which
    is the single-pass answer.

    `sequencer` picks how each arm's segments are ordered: "opt" (the default;
    `sequence.solve` — minimum transit TIME over orders and directions,
    exact to 16 segments) or "nn" (the old paper-distance nearest neighbour,
    kept for comparison).  Either way the chosen order is costed on the same
    transit-time matrix, so `transit_time` is comparable across both.

    `balance` runs the min-max load pass over the chosen cover (see the
    section-4b commentary and `rebalance`): a phase ends when its slowest arm
    stops, and the interval cover alone will happily give one arm 60 % of the
    ink.  It only ever re-assigns spans a second arm has already certified at
    the same endpoints, so it cannot change the coverage; `draw_speed` is the
    material's limit, and it enters here because the load it balances is
    SECONDS and not metres.  `--no-balance` recovers the old allocation.
    """
    if arms is not None and active_override is not None:
        raise ValueError("pass arms= or active_override=, not both")
    arms = list(arms) if arms is not None else active_arms(active_override)
    specs = {a: FLEET[a] for a in arms}
    if pens is not None:
        unknown = set(pens) - set(FLEET)
        if unknown:
            raise ValueError(f"pens names arms not in the fleet: {sorted(unknown)}")
    aopts = {a: pen_opts(opts, pens, a) for a in arms}
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
            v, s = probe_stroke(st["pts"], specs[a], aopts[a],
                                max_probes=max_probes, min_seg=min_seg,
                                gap_tol=gap_tol)
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

    if colors is None:
        colors, cover, table = best_partition(strokes, ivmap, arms, min_seg,
                                              gap_tol)
    else:
        missing = [a for a in arms if a not in colors]
        if missing:
            raise ValueError(f"colors does not give arms {missing} a pen")
        colors = {a: colors[a] for a in arms}
        cover = cover_all(strokes, ivmap, colors, min_seg, gap_tol)
        table = []

    # the colours are fixed now, so the holes are finally known in the terms
    # that matter — the union of the arms carrying the right ink
    t_rep = time.time()
    cover, n_repair = repair_gaps(strokes, ivmap, colors, cover, specs, aopts,
                                  min_seg, gap_tol, rounds=repair_rounds,
                                  min_len=float((opts or {}).get("min_length", 0.02)),
                                  budget=repair_budget, verbose=verbose)
    t_repair = time.time() - t_rep

    # ---- clean re-plans -------------------------------------------------
    t2 = time.time()
    programs = {a: [] for a in arms}
    n_replan, placed = 0, []
    for ps in cover["per_stroke"]:
        st, L = ps["stroke"], ps["L"]
        for span in place_cuts(ps["chosen"], L, overlap):
            plan, sp = replan_segment(st["pts"], span, specs[span["arm"]],
                                      aopts[span["arm"]], min_seg=min_seg)
            n_replan += sp["replans"]
            if plan is None:
                continue
            placed.append(dict(stroke=st, sp=sp, arm=span["arm"],
                               entry=_entry(st, sp, plan)))
    t_replan = time.time() - t2

    # ---- load balancing -------------------------------------------------
    t2b = time.time()
    bal = None
    pen_m = {a: pen_of(pens, a) for a in arms}
    if balance and placed:
        placed, bal = rebalance(placed, arms, colors, ivmap, specs, aopts,
                                pen_m, draw_speed, seq_opts, min_seg,
                                verbose=verbose)
        n_replan += bal["n_replans"]
    for it in placed:
        programs[it["arm"]].append(it["entry"])
    t_balance = time.time() - t2b

    dropped = leftover(strokes, programs, gap_tol)
    out = dict(colors=colors, arms=arms, table=table, ivmap=ivmap,
               probe_stats=probe_stats, programs={}, dropped=dropped,
               sequencer=sequencer, pens=pen_m, balance=bal,
               draw_speed=float(draw_speed),
               timing=dict(prefilter=t_pre, probe=t_probe, repair=t_repair,
                           replan=t_replan, balance=t_balance))
    t3 = time.time()
    out["sequence"], out["transit"], out["transit_time"] = {}, {}, {}
    for a in arms:
        sq = dict(seq_opts or {})
        sq["pen_ext"] = out["pens"][a]
        seq = sequence_arm(programs[a], specs[a], sequencer, aopts[a], sq)
        out["programs"][a] = seq.pop("programme")
        out["sequence"][a] = seq
        out["transit"][a] = transit_metres(out["programs"][a], FLEET[a].xy)
        out["transit_time"][a] = seq["cost"]
    out["timing"]["sequence"] = time.time() - t3
    out["timing"]["total"] = time.time() - t0
    out["total_len"] = float(sum(polyline_length(s["pts"]) for s in strokes))
    out["drawn_len"] = float(sum(s["length"] for a in arms
                                 for s in out["programs"][a]))
    out["dropped_len"] = float(sum(d["length"] for d in dropped))
    out["n_probes"] = int(sum(p["probes"] for p in probe_stats)) + n_repair
    out["n_repair_probes"] = int(n_repair)
    out["n_repair_spans"] = int(cover.get("repair_added", 0))
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
    lines.append(f"{'arm':>5} {'pen':>7} {'mm':>4} {'segs':>5} {'metres':>8} "
                 f"{'strokes':>8} {'cuts':>5} {'transit':>8} {'transit_s':>10} "
                 f"{'was':>8} {'rev':>4}")
    for a in res["arms"]:
        segs = res["programs"][a]
        ids = {s["stroke_id"] for s in segs}
        cuts = len(segs) - len(ids)
        q = res["sequence"][a]
        lines.append(f"{a:>5} {res['colors'][a]:>7} "
                     f"{1000 * res['pens'][a]:>4.0f} {len(segs):>5} "
                     f"{sum(s['length'] for s in segs):>8.2f} {len(ids):>8} "
                     f"{cuts:>5} {res['transit'][a]:>8.2f} "
                     f"{res['transit_time'][a]:>9.1f}s {q['baseline_cost']:>7.1f}s "
                     f"{q['n_reversed']:>4}")
    saved = sum(q["baseline_cost"] for q in res["sequence"].values()) - \
        sum(res["transit_time"].values())
    base = sum(q["baseline_cost"] for q in res["sequence"].values())
    methods = sorted({q["method"] for q in res["sequence"].values() if q["n"]})
    lines.append(f"sequencer '{res['sequencer']}' [{', '.join(methods) or 'none'}]: "
                 f"{sum(res['transit_time'].values()):.1f} s of transit vs "
                 f"{base:.1f} s for the nearest-xy chain "
                 f"({100 * saved / max(base, 1e-9):.1f} % less), "
                 f"{sum(q['n_reversed'] for q in res['sequence'].values())} "
                 f"segments drawn backwards"
                 + (f", {sum(q['n_refused'] for q in res['sequence'].values())} "
                    "reversals REFUSED by the validator"
                    if any(q["n_refused"] for q in res["sequence"].values()) else ""))
    st = [p for p in res["probe_stats"]]
    lines.append(f"probes {res['n_probes']} over {len(st)} (stroke, arm) pairs, "
                 f"{sum(1 for p in st if p['probes'] == 0)} prefiltered; "
                 f"clean re-plans {res['n_replans']}; "
                 f"gap repair offered {res.get('n_repair_probes', 0)} windows to "
                 f"the arms and got {res.get('n_repair_spans', 0)} certified "
                 "spans back")
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
    b = res.get("balance")
    if b:
        n_seg = sum(len(res["programs"][a]) for a in res["arms"])
        lines.append(f"balance: {b['n_movable']} of {n_seg} segments are "
                     f"certified by more than one arm; {b['rounds']} "
                     f"move{'' if b['rounds'] == 1 else 's'} taken, busiest arm "
                     f"{b['max_before']:.1f} s -> {b['max_after']:.1f} s "
                     f"(the phase's floor), draw speed {b['draw_speed']:g} m/s")
        lines.append("      per-arm nominal s  " + "  ".join(
            f"{a}:{b['loads_before'].get(a, 0.0):.1f}->"
            f"{b['loads_after'].get(a, 0.0):.1f}" for a in res["arms"]))
    t = res["timing"]
    lines.append(f"time  prefilter {t['prefilter']:.1f} s  probe {t['probe']:.1f} s"
                 f"  repair {t.get('repair', 0.0):.1f} s"
                 f"  replan {t['replan']:.1f} s"
                 f"  balance {t.get('balance', 0.0):.1f} s"
                 f"  sequence {t['sequence']:.1f} s"
                 f"  total {t['total']:.1f} s")
    return lines
