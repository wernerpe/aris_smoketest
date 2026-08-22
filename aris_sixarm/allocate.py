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
  SPLIT      Allocation v2's third move, and the only one that reaches the
             48 % of solo drawing time `docs/SOLO_TIME.md` measured as
             SPLITTABLE — a span another arm certifies PART of, which a
             balancer whose moves are whole segments cannot see at all.  A span
             on the busiest arm is CUT at a chosen s and one piece handed to a
             lighter arm that re-plans it from scratch, with a 5 mm splice at
             the seam and a 5 cm floor on both pieces.  The two pieces' union is
             the input span for every cut position, so coverage is invariant
             here too — and `coverage_lost` measures it rather than assume it.
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
  PRICE      BALANCE and ORDER are the same cost model looked at twice — the
             balancer's price for a bag is the tour the sequencer will find in
             it — so they are no longer chosen separately.  `cost_model`
             returns ONE object with both halves on it (`load` and `sequence`),
             and it is the only thing in this module that knows whether the
             fiber menus are on.  Before it existed, the balancer priced every
             candidate with the single-variant DP while `--cluster` sequenced
             with the cluster DP, and the balancer balanced a load the
             sequencer then moved (see section 4b-i).

Every chosen segment is re-planned from scratch at the end; a segment whose
clean re-plan is not "ok" is not shipped as one.
"""
import time
from dataclasses import dataclass

import numpy as np

from . import menu, sequence, writing
from .fleet import FLEET, H_INV_DEFAULT
from .stroke_api import (plan_stroke, polyline_length, reverse_plan,
                         truncate_polyline)

ACTIVE = [aid for aid, s in FLEET.items() if s.active]
COLORS = ("grey", "orange")
DRAW_SPEED = writing.DRAW_SPEED_FLEET    # m/s the material allows; a CAP, and
#   the load model's — see `writing.draw_duration`, which stretches it wherever
#   the redundancy resolution asks a joint to move faster than it may.
BALANCE_ROUNDS = 200                     # accepted moves the balancer may make

# A BAG WITH NO PAPER-LEGAL TOUR IS INFINITELY EXPENSIVE, NOT AN EXCEPTION.
# `sequence.cost_matrix` prices a pen-up `paper.route` refuses as `inf`
# (`docs/PAPER_PLANE.md` §2.1), so a bag of spans each of which this arm can
# certify AS INK can still have no order in which the arm can FLY between them —
# `held_karp` then raises "no feasible order over n segments".  That is the
# right answer for the pass that has to commit to an order, and the wrong one
# for `balance_loads`, which is a PRICING function: it asks "what would this
# arm's programme cost if it held these spans", and the honest answer to an
# unflyable bag is "more than any alternative", not a traceback that takes the
# whole allocation down with it.  Priced as `inf`, the bag simply loses every
# comparison and the balancer moves the offending span to an arm that can fly
# to it — which is what `load_score` is ordered to let it do.
#
# It took two colours per arm to expose this.  With one pen per arm the bags
# were small and every arm's spans clustered in its own reach; giving every arm
# BOTH inks doubles the candidate bags, and the first one the balancer priced
# (arm 97's five orange spans) contained a span the arm can ink and cannot
# reach.  Not the lower onto it — that is `direct`, as is the lift off it — but
# every hover-to-hover CROSSING into it, from the depot and from all four of the
# arm's other spans, in both directions: in-degree 0 in its own cost matrix.
UNFLYABLE = float("inf")
UNFLYABLE_ROUNDS = 2      # times the cover may be re-run without a span an arm
#   certifies as ink and cannot fly to.  Each round costs a cover and its clean
#   re-plans; two is enough for every phase measured, and `prune_unflyable`
#   after the balancer is what makes the result correct rather than this cap.

# ---- allocation v2: splitting as a balancing move -------------------------
MIN_SPLIT_M = 0.05        # m; the shortest piece a split may create.  Not the
#   same floor as MIN_SEG_M (0.025): that one asks "is this worth a pen-up at
#   all", this one asks "is cutting a line here worth an ENTRY, an EXIT and a
#   visible seam to the arm receiving it", and the answer is no for confetti.
SPLIT_OVERLAP_M = 0.005   # m of ink drawn TWICE at a split seam, half either
#   side of the cut — the same idea as `place_cuts`'s handoff overlap, so the
#   two pens meet rather than leaving a hairline of bare paper at the join.
SPLIT_ROUNDS = 60         # accepted splits one rebalance may make
SPLIT_BUDGET = 3000       # clean re-plans the split search may spend
SPLIT_CAND_SEGS = 8       # segments off the busiest arm considered per round
SPLIT_CAND_RECV = 5       # ELIGIBLE receiving arms considered per segment
SPLIT_CAND_CUTS = 3       # cut positions `split_candidates` offers by default
SPLIT_COARSE_CUTS = 2     # of them tried while RANKING (segment, receiver, side)
SPLIT_REFINE = 5          # bisection steps that then place the winner's cut


def active_arms(active_override=None, fleet=None):
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
    fl = FLEET if fleet is None else fleet
    if active_override is None:
        return [aid for aid, sp in fl.items() if sp.active]
    if isinstance(active_override, str):
        if active_override != "all":
            raise ValueError(f"active_override={active_override!r}; want 'all', "
                             "a list of arm ids, or a {arm_id: bool} mapping")
        return list(fl)
    if isinstance(active_override, dict):
        flags = {aid: s.active for aid, s in fl.items()}
        unknown = set(active_override) - set(fl)
        if unknown:
            raise ValueError(f"active_override names arms not in the fleet: "
                             f"{sorted(unknown)}")
        flags.update({a: bool(v) for a, v in active_override.items()})
        return [aid for aid in fl if flags[aid]]
    want = list(active_override)
    unknown = set(want) - set(fl)
    if unknown:
        raise ValueError(f"active_override names arms not in the fleet: "
                         f"{sorted(unknown)}")
    return [aid for aid in fl if aid in set(want)]

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
# OFF BY DEFAULT, FOR THE SAME REASON THE BAND OBJECTIVE IS.  It wins the two
# things it was built to win — transit -14.8 %, reconfiguration -65.0 % on the
# shipped CSAIL run — and loses the one that outranks them: pinning a stroke's
# entry and exit fiber is a constraint on the band, and a constrained band
# draws slower (see `pwl.OBJECTIVE`).  Turn it on with `--cluster`.
CLUSTER = False          # final per-arm programmes go through the fiber menus
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
                 gap_tol=GAP_TOL_M, bisect=False):
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

    `bisect` fixes the walk's one dead end, and `aris_sixarm/bench`'s spiral is
    how it was found.  A gap that comes back certifying NOTHING is marked tried
    and never looked at again — so the walk stops, and it stops after four
    probes however large the budget is.  That is harmless while a stroke is
    short enough that "this arm cannot draw this gap" is the truth about the
    whole gap, and it is badly wrong for a long one, where it means "this arm
    cannot START this fourteen-metre gap" and nothing more.  With `bisect` a
    barren gap is halved and both halves queued instead, so the budget keeps
    buying information.  It is off by default because it changes which intervals
    a run finds, and the logo's published numbers were measured without it.
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
    tried, pending = set(), []
    while stats["probes"] < max_probes:
        gaps = [(max(g[0], lo), min(g[1], hi)) for g in uncovered(iv, min_gap=eps)]
        gaps += pending
        gaps = [g for g in gaps if (g[1] - g[0]) * L >= min_seg
                and (round(g[0], 6), round(g[1], 6)) not in tried]
        if not gaps:
            break
        a, b = max(gaps, key=lambda g: (g[1] - g[0], -g[0]))
        key = (round(a, 6), round(b, 6))
        tried.add(key)
        pending = [g for g in pending
                   if (round(g[0], 6), round(g[1], 6)) != key]
        sub = truncate_polyline(pts, a, b)
        if len(sub) < 2 or polyline_length(sub) < min_seg:
            continue
        w, got = b - a, False
        r3 = plan_stroke(sub, spec, opts)
        stats["probes"] += 1
        stats["statuses"].append(r3["status"])
        s3, st3 = _certified_span(r3)
        if st3 in ("ok", "split") and s3 * w * L >= min_seg:
            iv.append(Interval(a, a + s3 * w, spec.arm_id, +1, "gap"))
            got = True
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
            got = True
        if bisect and not got and w * L >= 2.0 * min_seg:
            # NEITHER END OF THIS WINDOW IS DRAWABLE, WHICH IS NOT THE SAME AS
            # THE WINDOW BEING OUT OF REACH.  Halve it and ask again: an arm may
            # well certify the middle of a span it can neither start nor finish,
            # and on a long stroke that is the usual case rather than the odd one.
            mid = 0.5 * (a + b)
            pending += [(a, mid), (mid, b)]
    return iv, stats


def stroke_probes(L, max_probes, ref_m=None):
    """How many plan calls ONE stroke's probe may spend. -> int.

    THE BUDGET IS PER STROKE AND THE REACH PATTERN IS PER METRE.  That is fine
    while every stroke is about the same length — the CSAIL logo's are 0.1 to
    0.7 m — and it stops being fine the moment one is not.  `probe_stroke`
    spends its budget walking the largest remaining gap, so five calls map a
    half-metre line well and a fourteen-metre spiral hardly at all; and a stroke
    the probe never mapped is a stroke the cover cannot place, which surfaces
    not as a slow allocation but as MISSING INK.

    A BIGGER BUDGET ON ITS OWN BUYS NOTHING, which is worth knowing before
    reaching for this: `aris_sixarm/bench`'s spiral certifies 0.00 % of itself
    at a flat budget of 5 AND at a flat budget of 40, because the walk dead-ends
    (see `probe_stroke`'s `bisect`) long before the budget runs out.  The two
    have to be turned on together, and `allocate` does exactly that from the one
    `probe_ref_m` argument.  Together they take that spiral to 85.6 %.

    `ref_m` is the stroke length the flat budget was chosen for: give it, and a
    stroke n times longer gets n times the calls.  Left None the budget is flat,
    which is what every run before this one did — so the logo's numbers are
    untouched unless a caller asks for the new behaviour.
    """
    if ref_m is None or float(ref_m) <= 0.0:
        return int(max_probes)
    return int(max(int(max_probes),
                   int(np.ceil(int(max_probes) * float(L) / float(ref_m)))))


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


def _sub_matrix(C, idx, n):
    """`sequence.cost_matrix` restricted to the segments `idx`, depot last."""
    nodes = [k for i in idx for k in (2 * i, 2 * i + 1)] + [2 * n]
    return C[np.ix_(nodes, nodes)]


def prune_unflyable(spec, segs, mat, exact_max_n=sequence.EXACT_MAX_N,
                    budget=sequence.TIME_BUDGET):
    """Drop the spans that make this arm's bag impossible to fly. -> (keep, drop).

    Both are lists of indices into `segs`.  `keep` is guaranteed to have a
    finite tour; `drop` is the ink this arm must give back.

    CERTIFYING THE INK IS NOT CERTIFYING THE APPROACH, and until the paper
    became an obstacle nothing here had to tell the two apart.  `plan_stroke`
    walks the redundancy band and proves every sample of the STROKE clears its
    gates.  The pen-up that flies to the span is a different motion, priced by
    `sequence.cost_matrix` and — since `docs/PAPER_PLANE.md` — routed around the
    paper by `paper.route`, which is allowed to REFUSE.  A span can pass the
    first test and fail the second: the arm can draw the line and cannot get
    its pen there without putting a link through the table.

    That is not hypothetical; it is what stopped the first two-pass allocation
    on this rig dead.  Arm 97's piece of stroke 26 is certified ink that
    `replan_segment` gives back end to end, and EVERY crossing into its hover is
    refused — from each of the arm's own four other spans, in both directions,
    and from its ready pose — because each one dives 137 to 189 mm of chain
    below the paper and no shape on the router's height ladder recovers it.  The
    span therefore has IN-DEGREE 0 in the arm's own transit matrix, Held-Karp
    reports "no feasible order over 5 segments", and one unreachable span makes
    every ordering of the other four unreachable with it.

    WHY THE TEST IS THE TOUR AND NOT A PER-SPAN PREDICATE.  The tempting cheap
    gate is "can the arm fly from its ready pose to this span", and it is wrong:
    arm 71's orange bag in this same phase has two spans the depot cannot reach
    and a perfectly good 14.5 s tour that reaches them from its other spans.
    Only the tour knows.  So the tour is what is asked, and the spans blamed are
    the ones with no finite predecessor or no finite successor — the ones no
    order could have used — with a shortest-span fallback for the rarer case
    where every node has a neighbour and there is still no Hamiltonian path.
    """
    n = len(segs)
    if n == 0:
        return [], []
    C = sequence.cost_matrix(spec, segs, **mat)
    idx, drop = list(range(n)), []
    while idx:
        S = _sub_matrix(C, idx, n)
        try:
            sequence.solve(S, len(idx), exact_max_n, budget)
            break
        except RuntimeError:
            fin, m = np.isfinite(S), len(idx)
            bad = [k for k in range(m)
                   if not (fin[:, 2 * k].any() or fin[:, 2 * k + 1].any())
                   or not (fin[2 * k, :].any() or fin[2 * k + 1, :].any())]
            if not bad:            # no isolated node, and still no tour
                bad = [min(range(m),
                           key=lambda k: (segs[idx[k]]["length"], k))]
            for k in sorted(bad, reverse=True):
                drop.append(idx.pop(k))
    return idx, sorted(drop)


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
             budget=sequence.TIME_BUDGET, q_start=None, return_home=True):
    """Nominal seconds one arm needs for `segs`: its ink plus its best pen-up tour.

    `draw_s` is the per-segment ink time in the same order as `segs`.  The
    pen-up half is `sequence.solve` on the same cost matrix the real sequencing
    pass uses, so this is not an estimate of the arm's programme — it is the
    programme, costed before it is committed to.

    THAT IDENTITY IS ONLY TRUE OF THE SEQUENCER THIS FUNCTION CALLS.  With the
    fiber menus on (`CLUSTER`) the pass that actually runs is
    `sequence_arm_cluster`, whose DP chooses variants as well as orders and
    therefore reaches a different tour and a different number of seconds; use
    `cluster_arm_load` for that, or better, let `cost_model` hand you the
    matching pair (see the section below).
    """
    if not len(segs):
        return 0.0
    pen = writing.PEN_EXT if pen_ext is None else float(pen_ext)
    C = sequence.cost_matrix(spec, segs, transit_speed, qd_frac, h_inv,
                             ends=ends, pen_ext=pen, q_start=q_start,
                             return_home=return_home)
    try:
        r = sequence.solve(C, len(segs), exact_max_n, budget)
    except RuntimeError:
        return UNFLYABLE
    return float(sum(draw_s) + r["cost"])


def _stack_cluster_ends(es):
    """One-segment `sequence.cluster_endpoints` dicts -> the many-segment one.

    The block for segment i is `base[i] : base[i+1]`, laid down in segment
    order and never re-ordered inside, so concatenating per-segment dicts and
    recomputing `base`/`seg` reproduces `cluster_endpoints` over the same
    menus exactly — the same reason `_stack_ends` works for the plain path,
    and the reason a candidate bag can be re-priced without re-solving one IK.
    """
    keys = ("ent_q", "exi_q", "ent_xy", "exi_xy", "ent_h", "exi_h",
            "surcharge", "var", "dirn")
    if not es:
        out = {k: np.zeros((0, 7)) if k.endswith("_q") or k.endswith("_h")
               else np.zeros((0, 2)) if k.endswith("_xy") else np.zeros(0)
               for k in keys}
        return dict(out, seg=np.zeros(0, int), base=np.zeros(1, int), nv=[],
                    n=0, N=0)
    out = {k: np.concatenate([e[k] for e in es]) for k in keys}
    nv = [int(e["nv"][0]) for e in es]
    base = np.concatenate([[0], np.cumsum([2 * k for k in nv])]).astype(int)
    seg = np.concatenate([np.full(int(e["N"]), i, int)
                          for i, e in enumerate(es)])
    return dict(out, seg=seg, base=base, nv=nv, n=len(es), N=int(base[-1]))


def cluster_arm_load(spec, segs, draw_s, menus,
                     transit_speed=writing.TRANSIT_SPEED,
                     qd_frac=writing.QD_FRAC, h_inv=H_INV_DEFAULT,
                     pen_ext=None, ends=None, exact_max_n=sequence.EXACT_MAX_N,
                     budget=sequence.TIME_BUDGET, q_start=None,
                     return_home=True, w_surcharge=sequence.W_SURCHARGE):
    """`arm_load`, priced with the DP that will actually sequence this bag. -> s

    Same contract, same units, and the same claim — it is the programme costed
    before it is committed to — but over the (segment, direction, VARIANT)
    state space, because that is what `sequence_arm_cluster` will search.

    THE NUMBER IS THE SEQUENCER'S OWN OBJECTIVE, DECOMPOSED.  `cluster_solve`
    minimises transit seconds plus the interior surcharge of the variant each
    segment is entered on (`sequence.cluster_cost_matrix`: the surcharge is
    extra DRAW time, charged at `w_surcharge / qd_frac`, exactly as
    `writing.draw_duration` will stretch the ink for it) plus an infinitesimal
    reconfiguration tie-break.  This returns the ink the allocator already
    measured, plus the transit the DP chose, plus the surcharge the DP agreed
    to pay for its choice; the 1e-7-per-radian tie-break is the one term left
    out, and it tops out around 1e-5 s over a whole tour.

    `draw_s` is the ink time of the plan the allocator certified, which is the
    band DP's own free-ended optimum, i.e. the menu variant with surcharge 0
    when the band is being scored on travel (`--band-objective min_travel`,
    which is what `--cluster` is measured with).  So the surcharge added here
    is a DELTA against the ink already counted rather than a second charge for
    it.  Under `maximin_sigma` that identification is approximate — the free
    band optimises clearance and takes travel as a tie-break — and it is
    reported here rather than hidden, because the sign of the error is
    conservative: a variant is priced at no less than it costs.
    """
    if not len(segs):
        return 0.0
    pen = writing.PEN_EXT if pen_ext is None else float(pen_ext)
    C, T, e = sequence.cluster_cost_matrix(spec, menus, transit_speed, qd_frac,
                                           h_inv, pen, q_start=q_start,
                                           return_home=return_home, ends=ends,
                                           w_surcharge=w_surcharge)
    try:
        r = sequence.cluster_solve(C, T, e, exact_max_n, budget)
    except RuntimeError:
        return UNFLYABLE
    base, sur = np.asarray(e["base"], int), np.asarray(e["surcharge"], float)
    nodes = [int(base[k]) + 2 * int(v) + (0 if d > 0 else 1)
             for k, v, d in zip(r["order"], r["variants"], r["dirs"])]
    extra = (w_surcharge / max(qd_frac, 1e-6)) * float(sur[nodes].sum())
    return float(sum(draw_s) + r["cost"] + extra)


# ===========================================================================
# 4b-i. the costing hook: one object prices a bag AND sequences it
# ===========================================================================
# THE DEFECT THIS EXISTS TO MAKE IMPOSSIBLE.  `rebalance` decides who draws
# what by pricing candidate bags in seconds, and `allocate` then hands each
# arm's bag to a sequencer.  Those were two independent choices of cost model:
# the balancer always priced with `sequence.solve` while the pass that ran
# could be `sequence_arm_cluster`, whose DP moves the tour the balancer had
# just balanced.  Measured on the CSAIL logo at the rig's own 0.02 m/s draw
# speed (`docs/BENCH.md`), that is not a rounding difference — the orange
# phase's imbalance goes 1.40x with the defaults to 1.49x with
# `min_travel + cluster`, and the whole residual penalty of the feature pair
# at rig speed is that one arm in that one phase.
#
# So the two are no longer allowed to be chosen separately.  A `CostModel` is
# the pair — `load` and `sequence` are the same DP over the same inputs — and
# `allocate` constructs exactly one and gives it to both stages.  Adding a
# third sequencing model means implementing both halves of one object, which
# is the property being bought: allocation and sequencing cannot disagree
# unless somebody writes two DPs into one class on purpose.
def _seg_key(seg, arm):
    """A menu's identity: which arm, and which polyline it will enumerate.

    `menu.stroke_menu` reads the segment's `pts`, and those are a function of
    (stroke, s-range, direction) alone (`_entry` -> `_segment_points`), so this
    is the whole of what a cached menu depends on besides the arm's own spec
    and planner options — which are fixed for the run.
    """
    s0, s1 = seg["s_range"]
    return (int(arm), int(seg["stroke_id"]), round(float(s0), 9),
            round(float(s1), 9), int(seg["direction"]))


class SegmentCost:
    """Price and sequence with the (segment, direction) DP.  The shipped model.

    Exactly what this pipeline did before the hook existed: `arm_load` for the
    price, `sequence_arm` for the order.  Both call `sequence.cost_matrix` and
    `sequence.solve` on the same segments, so the price is the tour, to the
    float, unless a reversal the DP asked for will not certify.
    """

    cluster = False
    name = "segment"

    def menus(self, arm, spec, segs, opts=None):
        return None

    def load(self, arm, spec, segs, draw_s, menus=None, **kw):
        kw.pop("opts", None)          # the same signature the cluster model
        return arm_load(spec, segs, draw_s, **kw)     # accepts; it needs opts

    def sequence(self, arm, segs, spec, sequencer=SEQUENCER, opts=None,
                 seq_opts=None, forbid=None, verbose=False):
        return sequence_arm(segs, spec, sequencer, opts, seq_opts, forbid)


class ClusterCost:
    """Price and sequence with the (segment, direction, variant) cluster DP.

    THE MENUS ARE BUILT ONCE AND SHARED.  A menu is a property of a span and an
    arm, not of the bag that span currently sits in, so the same `menu.Menu`
    object serves every candidate assignment the balancer prices AND the final
    sequencing pass — which is what makes the price and the programme the same
    computation rather than two computations that ought to agree.  It is also
    what makes the fix affordable: enumerating a menu costs about one plan
    call, and the balancer already pays one to certify the same span for the
    same arm.

    The lattices are kept while the allocation runs (they are what makes
    `materialize` cheap) and dropped by `allocate` before the result is
    returned, exactly as before.
    """

    cluster = True
    name = "cluster"

    def __init__(self, menu_opts=None, verbose=False):
        self.menu_opts = dict(menu_opts or {})
        # a menu is enumerated ONE span at a time here (that is what makes it
        # cacheable), so `build_menus`'s own per-call line would print once per
        # segment per arm; the summary is printed by `sequence` instead
        self.verbose = bool(self.menu_opts.pop("verbose", verbose))
        self.menu_opts["verbose"] = False
        self._menus = {}
        self.n_built = 0

    def menus(self, arm, spec, segs, opts=None):
        """One `menu.Menu` per segment, memoised on (arm, span). -> list."""
        out = []
        for s in segs:
            k = _seg_key(s, arm)
            m = self._menus.get(k)
            if m is None:
                m, _stat = build_menus([s], spec, opts, **self.menu_opts)
                m = m[0]
                self._menus[k] = m
                self.n_built += 1
            out.append(m)
        return out

    def load(self, arm, spec, segs, draw_s, menus=None, **kw):
        if menus is None:
            menus = self.menus(arm, spec, segs, kw.get("opts"))
        kw.pop("opts", None)
        return cluster_arm_load(spec, segs, draw_s, menus, **kw)

    def sequence(self, arm, segs, spec, sequencer=SEQUENCER, opts=None,
                 seq_opts=None, forbid=None, verbose=False):
        if sequencer not in ("opt", "transit") or not len(segs):
            return sequence_arm(segs, spec, sequencer, opts, seq_opts, forbid)
        mus = self.menus(arm, spec, segs, opts)
        if verbose or self.verbose:
            sizes = [len(m) for m in mus]
            n_plan = sum(1 for m in mus if isinstance(m, menu.PlanMenu))
            print(f"    menus: {len(mus) - n_plan} enumerated, {n_plan} fell "
                  f"back to the certified plan; sizes {sizes}")
        r = sequence_arm_cluster(segs, spec, mus, opts, seq_opts, forbid,
                                 verbose=verbose)
        r["menus"] = mus
        return r


def cost_model(cluster=None, menu_opts=None, verbose=False):
    """The costing hook for this run. -> `SegmentCost` or `ClusterCost`.

    One call site decides which sequencer a run uses, and the object it returns
    is the only thing that knows: `allocate` prices with it and sequences with
    it, so the two cannot be configured apart.
    """
    if isinstance(cluster, (SegmentCost, ClusterCost)):
        return cluster
    return (ClusterCost(menu_opts, verbose) if bool(CLUSTER if cluster is None
                                                    else cluster)
            else SegmentCost())


def load_score(loads):
    """The balancer's objective for one assignment.

    -> (arms with no flyable tour, max load, sum of squares).

    Compared lexicographically and strictly, so it is a POTENTIAL: every
    accepted move (relocation, swap or split) lowers it, which is what makes
    "iterate until nothing improves" terminate rather than cycle.  It lives at
    module scope because the split search in `_split_search` has to be scored on
    exactly the same ruler as `balance_loads`, and two rulers that disagree by a
    rounding convention would let the two moves undo each other for ever.

    FEASIBILITY IS THE FIRST TERM BECAUSE `inf` IS NOT AN ORDERING.  A bag with
    no paper-legal tour prices at `UNFLYABLE` (see the constant), and if that
    number went straight into the max the score of every assignment containing
    one would be `(inf, inf)` — all equal, all incomparable, and the balancer's
    strict `<` would find no move at all.  It is exactly what happened the first
    time the two-pass allocation was run on this rig: arm 97's orange bag was
    unflyable and the pass reported "0 moves and 0 splits taken, busiest arm
    inf s -> inf s".  Counting the unflyable arms FIRST gives that plateau a
    gradient — handing the offending span to an arm that can reach it takes the
    count from 1 to 0 and strictly wins, whatever it does to the seconds — and
    the remaining two terms are the old ruler, computed over the arms that have
    a tour, so they still break ties among assignments that are all feasible.
    THE OLD NUMBERS ARE UNCHANGED: when every bag is flyable the first term is
    0 for every candidate and the comparison falls through to the same
    `(max, sum of squares)` it always was, to the same rounding.
    """
    v = sorted(loads.values(), reverse=True)
    bad = int(sum(1 for x in v if not np.isfinite(x)))
    fin = [x for x in v if np.isfinite(x)]
    return (bad, round(fin[0], 9) if fin else 0.0,
            round(float(sum(x * x for x in fin)), 6))


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
        return load_score(L), L

    key, L = score(owner)
    info = dict(loads_before=dict(L), max_before=key[1], moves=[], rounds=0,
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
        best[2].update(max_after=key[1])
        info["moves"].append(best[2])
        info["rounds"] += 1
        if verbose:
            m = best[2]
            print(f"  balance {info['rounds']:>2}: {m['kind']} segment "
                  f"{m['seg']} arm {m['frm']} -> arm {m['to']}"
                  + (f" (against segment {m['other']})" if "other" in m else "")
                  + f"; busiest arm now {key[1]:.1f} s")
    info.update(loads_after=dict(L), max_after=key[1], n_loads=len(cache))
    return owner, info


# ===========================================================================
# 4c. allocation v2: SPLITTING a placed span is also a balancing move
# ===========================================================================
# THE BALANCER'S MOVE SET WAS THE WRONG SHAPE, AND `docs/SOLO_TIME.md` MEASURED
# BY HOW MUCH.  Over the shipped CSAIL schedule, 48 % of the time one arm spends
# drawing alone is SPLITTABLE: another arm of the right colour certifies PART of
# the span being drawn and not all of it.  A balancer whose only moves are
# "hand the whole segment over" and "trade two whole segments" cannot see any of
# it — the receiving arm refuses the span at its far end, so the move is never
# even offered — and the phase floors at whatever the busiest arm's own ink
# costs.  Phase 2 of the logo is exactly that shape: arm 97 finishes AT its
# floor, and its floor is 49.6 s of ink no other arm certifies end to end,
# against 31.8 s at sub-span granularity.
#
# So the move set grows a third member.  A split takes one placed span, cuts it
# at a chosen `s`, and hands one of the two pieces to a less-loaded arm.  Four
# things keep it honest, and they are the same four the whole-segment moves
# already obeyed:
#
#   COVERAGE CANNOT DROP.  The two pieces are [s0, cut + e] and [cut - e, s1]
#   with e half the splice, so their UNION is the input span for every cut
#   position — there is no `s` at which a piece of paper belongs to neither
#   half.  Each piece is then re-planned from scratch for the arm that will draw
#   it and refused unless the clean re-plan certifies it end to end
#   (`replan_same_span`, which gives back not one millimetre).  If either half
#   will not certify, the split is not taken; there is no partial acceptance
#   that could leave a hole.  `rebalance` re-derives the merged cover afterwards
#   and raises rather than return a programme that lost ink.
#
#   NO CONFETTI.  Both pieces are held to `min_split` (0.05 m), an order of
#   magnitude above the 5 mm splice and twice `MIN_SEG_M`.  A cut costs the
#   receiving arm an entry and an exit — 1.4-1.9 s of pen-up on this piece —
#   so a split that hands over 2 cm of ink is a loss dressed as a win, and the
#   floor says so before the clock has to.
#
#   THE SEAM IS A HANDOFF LIKE ANY OTHER.  `place_cuts` already puts a handoff
#   in the middle of the overlap and grows both sides past it so the ink meets;
#   a split has no overlap to sit in the middle of, so it MAKES one — `splice`
#   metres drawn twice, half either side of the cut.
#
#   THE COLOUR AND THE PHASE ARE CONSTRAINTS, NOT PREFERENCES.  A receiver is
#   only considered if `colors` says it is holding the right ink for this
#   stroke in this phase, which is the same test the cover and the whole-segment
#   moves apply.
#
# The search over cuts is not an optimisation, it is three candidates: the cut
# that would EQUALISE the two arms' loads, the cut that transfers as much as the
# receiver's reach allows, and the midpoint between them.  The equalising cut is
# the one that lowers the objective; the other two are what is left when the
# reach will not stretch that far.  Where the reach gives out is not guessed —
# it is `certified_prefix` / `certified_suffix` over the probe intervals
# `probe_stroke` already bought, which is the same data `docs/SOLO_TIME.md`
# counted its 48 % with.


def certified_prefix(ivs, s0, s1):
    """How far past `s0` the union of `ivs` runs unbroken. -> s in [s0, s1].

    The furthest a cut may be pushed if the receiver is to take the HEAD of the
    span: beyond it the arm's certified reach has a hole in it, and a piece with
    a hole is not a piece any clean re-plan will accept.
    """
    x = float(s0)
    for v in sorted(ivs, key=lambda v: (v.s0, -v.s1)):
        if v.s0 > x + EPS_S:
            break
        x = max(x, min(float(v.s1), float(s1)))
        if x >= s1 - EPS_S:
            return float(s1)
    return float(min(x, s1))


def certified_suffix(ivs, s0, s1):
    """How far back from `s1` the union of `ivs` runs unbroken. -> s in [s0, s1].

    The mirror of `certified_prefix`, for a receiver taking the TAIL.  Both ends
    are asked separately because a plan is a walk of the redundancy band and the
    band is not symmetric: an arm that cannot start a line can often finish it.
    """
    x = float(s1)
    for v in sorted(ivs, key=lambda v: (-v.s1, v.s0)):
        if v.s1 < x - EPS_S:
            break
        x = min(x, max(float(v.s0), float(s0)))
        if x <= s0 + EPS_S:
            return float(s0)
    return float(max(x, s0))


def split_candidates(s0, s1, L, prefix, suffix, want_m, min_split=MIN_SPLIT_M,
                     splice=SPLIT_OVERLAP_M, n=SPLIT_CAND_CUTS):
    """Where a span could be cut for one particular receiver. -> [(s_cut, side)].

    `side` names WHICH PIECE THE RECEIVER TAKES — "head" is [s0, cut + e] and
    "tail" is [cut - e, s1], e being half the splice.  `prefix` and `suffix` are
    that receiver's unbroken certified reach in from each end (see
    `certified_prefix`); `want_m` is the ink the balancer would like to hand
    over, i.e. the equalising cut.

    Candidates come back best-first and deduplicated, and every one of them is a
    pure function of the arguments: no state, no randomness, no tie broken on
    anything the caller cannot see.  Both pieces are held to `min_split` metres,
    which is what stops a balancer with a cutting move from turning a line into
    confetti one 5 cm win at a time.
    """
    e = 0.5 * float(splice) / max(L, 1e-9)
    mp = float(min_split) / max(L, 1e-9)
    w = float(want_m) / max(L, 1e-9)
    out, seen = [], set()
    for side in ("head", "tail"):
        if side == "head":                     # receiver draws [s0, cut + e]
            lo, hi = s0 + mp - e, min(float(prefix) - e, s1 - mp + e)
            picks = (s0 + w - e, hi, 0.5 * (lo + hi))
        else:                                  # receiver draws [cut - e, s1]
            lo, hi = max(float(suffix) + e, s0 + mp - e), s1 - mp + e
            picks = (s1 - w + e, lo, 0.5 * (lo + hi))
        if hi < lo - 1e-12:
            continue
        for c in picks[:max(int(n), 1)]:
            c = round(float(min(max(c, lo), hi)), 9)
            if (c, side) in seen:
                continue
            seen.add((c, side))
            out.append((c, side))
    return out


def split_span(sp, s_cut, L, side, splice=SPLIT_OVERLAP_M):
    """One placed span cut in two. -> (the span its owner keeps, the span given).

    `side` names which piece the RECEIVER takes.  Each piece is grown by half
    the splice past the cut, so `splice` metres of ink are laid down twice and
    the two pens meet instead of leaving a hairline of bare paper at the join —
    the same treatment `place_cuts` gives a handoff between two cover intervals.

    THE UNION OF THE TWO PIECES IS THE INPUT SPAN, for every `s_cut`.  That is
    not a property this function checks, it is the reason it is written this way:
    coverage invariance under splitting is a type, exactly as it is for a move.
    """
    e = 0.5 * float(splice) / max(L, 1e-9)
    s0, s1 = float(sp["s0"]), float(sp["s1"])
    c = float(np.clip(s_cut, s0, s1))
    lo = dict(sp, s0=s0, s1=min(c + e, s1), source="split")
    hi = dict(sp, s0=max(c - e, s0), s1=s1, source="split")
    return (hi, lo) if side == "head" else (lo, hi)


def merged_spans(items):
    """Placed spans -> {stroke id: merged, sorted, non-overlapping [(s0, s1)]}.

    What the fleet will have covered, with WHO draws it and IN HOW MANY PIECES
    projected away — which is exactly the quantity a split must not change.
    """
    by = {}
    for it in items:
        sp = it["sp"]
        by.setdefault(int(it["stroke"]["id"]), []).append(
            (float(min(sp["s0"], sp["s1"])), float(max(sp["s0"], sp["s1"]))))
    out = {}
    for k, v in by.items():
        v.sort()
        m = [list(v[0])]
        for a, b in v[1:]:
            if a <= m[-1][1] + EPS_S:
                m[-1][1] = max(m[-1][1], b)
            else:
                m.append([a, b])
        out[k] = [(float(a), float(b)) for a, b in m]
    return out


def coverage_lost(before, after, lengths, tol=EPS_S):
    """Metres in `before`'s cover that `after`'s does not cover. -> float.

    THE GUARD RAIL, stated in metres rather than in reasoning.  Splitting is
    coverage-invariant by construction, and this is the measurement that says so
    out loud once per allocation: an arithmetic slip in a cut position is a
    silent hole in the picture, and a silent hole is the one failure mode this
    pipeline has always refused to allow.
    """
    lost = 0.0
    for sid, spans in before.items():
        L = float(lengths.get(sid, 0.0))
        cover = [Interval(a, b, -1) for a, b in after.get(sid, [])]
        for a, b in spans:
            for x, y in uncovered(cover, lo=a, hi=b, min_gap=tol):
                lost += (y - x) * L
    return float(lost)


def _span_key(it, arm):
    """A placed span identified by what a plan call would actually be handed."""
    sp = it["sp"]
    return (int(it["stroke"]["id"]), round(float(sp["s0"]), 9),
            round(float(sp["s1"]), 9), int(sp["direction"]), int(arm))


def _stack_ends(es):
    """One-segment `sequence.endpoints` dicts -> the many-segment one."""
    if not es:
        return dict(q=np.zeros((0, 2, 7)), xy=np.zeros((0, 2, 2)),
                    hover=np.zeros((0, 2, 7)), z=np.zeros((0, 2)), n=0)
    return dict(q=np.concatenate([e["q"] for e in es]),
                xy=np.concatenate([e["xy"] for e in es]),
                hover=np.concatenate([e["hover"] for e in es]),
                z=np.concatenate([e["z"] for e in es]), n=len(es))


class _Pricer:
    """Certifies and prices per-arm programmes over a MUTABLE set of spans.

    `rebalance` used to build its cost model exactly once, and could, because
    its moves changed only WHO drew a segment.  Splitting changes the segments
    themselves, so the model has to be rebuildable — and rebuilding it must not
    re-buy a plan call for every alternative that was already certified two
    rounds ago.

    Everything expensive is therefore memoised on the SPAN (stroke, endpoints,
    direction, arm) rather than on its position in any list, which is precisely
    the granularity a split preserves: cutting segment i in two leaves every
    other segment's clean re-plan, hover poses and ink seconds untouched, and
    even the per-arm LOAD survives if that arm's bag did not change.  The load
    key is the sorted tuple of span keys for that reason, and the segments are
    priced in that same sorted order so the cache cannot return a number the
    recomputation would disagree with.
    """

    def __init__(self, arms, colors, ivmap, specs, aopts, pens, draw_speed,
                 seq, min_seg, h_inv, q_start, return_home, cost=None):
        self.cost = cost_model(cost)
        self.arms = sorted(arms)
        self.colors, self.ivmap = colors, ivmap
        self.specs, self.aopts, self.pens = specs, aopts, pens
        self.draw_speed, self.min_seg, self.h_inv = draw_speed, min_seg, h_inv
        self.ts = float(seq.get("transit_speed", writing.TRANSIT_SPEED))
        self.qf = float(seq.get("qd_frac", writing.QD_FRAC))
        self.exact = int(seq.get("exact_max_n", sequence.EXACT_MAX_N))
        self.budget = float(seq.get("budget", sequence.TIME_BUDGET))
        self.q_start = dict(q_start or {})
        self.return_home = bool(return_home)
        self._entry, self._ends, self._draw, self._load = {}, {}, {}, {}
        self._probe, self._cends = {}, {}
        self.n_replans = self.n_probes = 0
        self.reach = None      # {(stroke id, arm): bool} from the atlas prefilter

    # -- certification -----------------------------------------------------
    def seed(self, it):
        """Adopt the entry the cover already certified, at no plan-call cost."""
        self._entry.setdefault(_span_key(it, it["arm"]), it["entry"])

    def probe_span(self, it, arm):
        """Where this arm's reach gives out INSIDE this exact span. -> (pre, suf).

        Two `plan_stroke` calls on the sub-polyline the balancer is actually
        holding: forward, whose `s_star` is the certified head, and reversed,
        whose `s_star` is the certified tail.  Both come back in the stroke's own
        normalised arc length, and both are spans the certified-or-split contract
        has PLANNED — this reads no interval off a model and extrapolates none.

        `probe_stroke`'s intervals cannot answer this on their own.  Its budget
        is spent walking the whole STROKE's gaps, and a placed span is usually a
        fraction of one stroke; asking about the span directly is the difference
        between "this arm reaches somewhere on this line" and "this arm reaches
        the first 40 % of the piece arm 97 is holding".  It is also the cheap
        half of the work: two probes cost about what one clean re-plan does, and
        they replace a cut position that would otherwise have to be guessed.
        """
        k = _span_key(it, arm)
        if k in self._probe:
            return self._probe[k]
        st, sp = it["stroke"], it["sp"]
        s0, s1 = float(sp["s0"]), float(sp["s1"])
        pre, suf = s0, s1
        sub = truncate_polyline(st["pts"], s0, s1)
        if len(sub) >= 2:
            w = s1 - s0
            r = plan_stroke(sub, self.specs[arm], self.aopts[arm])
            self.n_probes += 1
            f, stt = _certified_span(r)
            if stt == "ok":
                pre = s1
            elif stt == "split":
                pre = s0 + f * w
            rr = plan_stroke(sub[::-1], self.specs[arm], self.aopts[arm])
            self.n_probes += 1
            g, stt = _certified_span(rr)
            if stt == "ok":
                suf = s0
            elif stt == "split":
                suf = s1 - g * w
        self._probe[k] = (float(pre), float(suf))
        return self._probe[k]

    def reaches(self, it, arm):
        """Is this (stroke, arm) pair worth a probe at all? -> bool.

        The atlas prefilter's answer, re-used: an arm with no reachable cell
        within 5 cm of any point of the stroke cannot certify a sub-span of it
        either, and asking costs two plan calls that will both come back
        `start_infeasible`.  Absent a prefilter everything is asked.
        """
        if self.reach is None:
            return True
        return bool(self.reach.get((it["stroke"]["id"], arm), True))

    def entry(self, it, arm):
        """This arm's clean re-plan of EXACTLY this span, or None. -> entry|None."""
        k = _span_key(it, arm)
        if k not in self._entry:
            e, n = replan_same_span(it["stroke"], it["sp"], self.specs[arm],
                                    self.aopts[arm], self.min_seg)
            self.n_replans += n
            self._entry[k] = e
        return self._entry[k]

    # -- pricing -----------------------------------------------------------
    def draw_s(self, arm, it, ent):
        k = _span_key(it, arm)
        if k not in self._draw:
            self._draw[k] = float(writing.segment_draw_time(
                self.specs[arm], ent, self.draw_speed, self.qf, self.h_inv,
                self.pens[arm]))
        return self._draw[k]

    def ends(self, arm, it, ent):
        k = _span_key(it, arm)
        if k not in self._ends:
            self._ends[k] = sequence.endpoints(self.specs[arm], [ent],
                                               self.h_inv, self.pens[arm])
        return self._ends[k]

    def menu(self, arm, it, ent):
        """This arm's fiber menu for exactly this span (cluster model only)."""
        return self.cost.menus(arm, self.specs[arm], [ent], self.aopts[arm])[0]

    def cluster_ends(self, arm, it, ent):
        k = _span_key(it, arm)
        if k not in self._cends:
            self._cends[k] = sequence.cluster_endpoints(
                self.specs[arm], [self.menu(arm, it, ent)], self.h_inv,
                self.pens[arm])
        return self._cends[k]

    def load(self, arm, pairs):
        """Nominal seconds arm `arm` needs for these (item, entry) pairs.

        Priced by THIS RUN'S cost model (`cost_model`), which is the same
        object that will sequence the bag once the balancer has stopped moving
        it.  Whichever model that is, the memoisation below is on the span and
        the bag, so switching models changes what is computed and not how often.
        """
        if not pairs:
            return 0.0
        pairs = sorted(pairs, key=lambda p: _span_key(p[0], arm))
        ck = (arm, tuple(_span_key(it, arm) for it, _ in pairs))
        if ck not in self._load:
            segs = [e for _, e in pairs]
            draw = [self.draw_s(arm, it, e) for it, e in pairs]
            kw = dict(transit_speed=self.ts, qd_frac=self.qf, h_inv=self.h_inv,
                      pen_ext=self.pens[arm], exact_max_n=self.exact,
                      budget=self.budget, q_start=self.q_start.get(arm),
                      return_home=self.return_home)
            if self.cost.cluster:
                kw["menus"] = [self.menu(arm, it, e) for it, e in pairs]
                kw["ends"] = _stack_cluster_ends(
                    [self.cluster_ends(arm, it, e) for it, e in pairs])
            else:
                kw["ends"] = _stack_ends([self.ends(arm, it, e)
                                          for it, e in pairs])
            self._load[ck] = self.cost.load(arm, self.specs[arm], segs, draw,
                                            **kw)
        return self._load[ck]

    def loads(self, items):
        """Per-arm seconds for the assignment `items` currently carries."""
        by = {}
        for it in items:
            by.setdefault(it["arm"], []).append((it, it["entry"]))
        return {a: self.load(a, by.get(a, [])) for a in self.arms}

    def load_fn(self, items, entries):
        """`balance_loads`'s pricing callback over this item list."""
        def f(a, idx):
            return self.load(a, [(items[i], entries[i][a]) for i in idx])
        return f

    # -- the alternatives a whole-segment move may choose from -------------
    def options(self, items):
        """(options, entries): the arms that certify each span AT ITS ENDPOINTS."""
        opts, ents = [], []
        for it in items:
            st, own = it["stroke"], it["arm"]
            self.seed(it)
            o, e = {own}, {own: it["entry"]}
            for b in self.arms:
                if b == own or self.colors.get(b) != st["color"]:
                    continue
                ivs = [v for v in self.ivmap.get(st["id"], []) if v.arm == b]
                if not frac_covers(ivs, it["sp"]["s0"], it["sp"]["s1"]):
                    continue
                x = self.entry(it, b)
                if x is not None:
                    o.add(b)
                    e[b] = x
            opts.append(o)
            ents.append(e)
        return opts, ents


def _try_cut(items, i, side, s_cut, L, src, recv, splice, pricer):
    """Cut span `i`, certify both halves, and price the whole fleet with them.

    -> (score, trial items, loads, record) or None if either half will not
    certify for the arm that would draw it.  BOTH halves are re-planned from
    scratch through `replan_same_span`, which refuses a plan that gives back so
    much as a millimetre — the donor's remainder is not grandfathered in just
    because it used to be part of a span the donor certified.
    """
    it = items[i]
    keep = dict(it, entry=None)
    give = dict(it, entry=None, arm=recv)
    keep["sp"], give["sp"] = split_span(it["sp"], s_cut, L, side, splice)
    e_give = pricer.entry(give, recv)
    if e_give is None:
        return None
    e_keep = pricer.entry(keep, src)
    if e_keep is None:
        return None
    keep["entry"], give["entry"] = e_keep, e_give
    trial = items[:i] + [keep, give] + items[i + 1:]
    loads = pricer.loads(trial)
    key = load_score(loads)
    return key, trial, loads, dict(
        kind="split", seg=int(i), frm=int(src), to=int(recv), side=side,
        stroke=int(it["stroke"]["id"]), s_cut=float(s_cut),
        keep_m=float(e_keep["length"]), give_m=float(e_give["length"]),
        max_after=float(key[1]))


def _bisect_cut(items, i, side, lo, hi, L, src, recv, splice, pricer, best,
                steps=SPLIT_REFINE):
    """Walk the cut to where the two arms' loads CROSS. -> (best, evaluations).

    Handing more ink over lowers the donor's load and raises the receiver's,
    both monotonically, so the maximum of the two is V-shaped in the cut
    position and its minimum is the crossing point.  That makes a bisection the
    right search and a fixed number of steps enough: five halvings place the cut
    to about 3 % of the span, which is finer than the 5 cm floor cares about.

    The coarse candidates in `_split_search` decide WHICH span to cut and WHO
    takes which end; this decides WHERE, and it is worth doing separately
    because the analytic guess has to price a pen-up tour it cannot see.  A cut
    the arms will not certify is treated as "too much ink handed over", which is
    the usual reason: the receiver's reach is what runs out.  `best` is threaded
    through so the answer is never worse than what was already found.
    """
    a, z, n = float(lo), float(hi), 0
    for _ in range(int(steps)):
        if z - a <= 1e-9:
            break
        c = round(0.5 * (a + z), 9)
        r = _try_cut(items, i, side, c, L, src, recv, splice, pricer)
        n += 1
        if r is None:                       # give less and try again
            a, z = (a, c) if side == "head" else (c, z)
            continue
        key, trial, loads, rec = r
        if best is None or key < best[0]:
            best = (key, trial, rec)
        more = loads[src] > loads.get(recv, 0.0)
        if side == "head":                  # a bigger cut hands over more
            a, z = (c, z) if more else (a, c)
        else:                               # a smaller cut hands over more
            a, z = (a, c) if more else (c, z)
    return best, n


def _split_search(items, loads, pricer, min_split, splice, budget):
    """The best IMPROVING split of one span off the busiest arm.

    -> ((score, new items, record), plan calls spent), the triple being None
    when nothing improves.  The busiest arm is the only source worth cutting for
    the same reason it is the only source worth moving from: it is the only arm
    whose load is the objective.  Segments are tried longest-first (the ink is
    where the seconds are) and receivers least-loaded-first.

    TWO STAGES, because the two questions have different prices.  WHICH span to
    cut and WHO takes which end is decided on two coarse candidates apiece — the
    load-equalising cut and the most the receiver's reach will take — and only
    the winner is then refined by `_bisect_cut`.  Every candidate is scored on
    `load_score`, the balancer's own ruler, so a split and a relocation can be
    compared without either being given a handicap; a split is accepted only if
    it beats the score the move/swap pass left behind.
    """
    key0 = load_score(loads)
    src = max(loads, key=lambda a: (loads[a], a))
    mine = sorted((i for i, it in enumerate(items) if it["arm"] == src),
                  key=lambda i: (-float(items[i]["entry"]["length"]), i))
    spent, best, where = 0, None, None
    for i in mine[:SPLIT_CAND_SEGS]:
        it = items[i]
        st, sp = it["stroke"], it["sp"]
        if float(it["entry"]["length"]) < 2.0 * min_split + splice:
            continue                       # cannot make two legal pieces
        L = polyline_length(st["pts"])
        rate = pricer.draw_s(src, it, it["entry"]) / max(it["entry"]["length"], 1e-9)
        # ELIGIBILITY BEFORE LOAD.  Ranking every arm by load and then keeping
        # the lightest few hands the shortlist to the arms that are idle BECAUSE
        # THEY REACH NOTHING — two parked floor arms at 0.0 s crowd out the arm
        # that certifies 93 % of the span and happens to be second busiest.  So
        # the colour and the atlas are applied first and the load only orders
        # what is left.
        recv = sorted((b for b in pricer.arms
                       if b != src and pricer.colors.get(b) == st["color"]
                       and pricer.reaches(it, b)),
                      key=lambda b: (loads.get(b, 0.0), b))
        for b in recv[:SPLIT_CAND_RECV]:
            ivs = [v for v in pricer.ivmap.get(st["id"], []) if v.arm == b]
            n0 = pricer.n_probes
            p2, s2 = pricer.probe_span(it, b)
            spent += pricer.n_probes - n0
            # the probe of THIS span and the probe pass's intervals over the
            # whole stroke are both certified runs in from the same end, so the
            # longer of the two is certified as well and neither is discarded
            pre = max(certified_prefix(ivs, sp["s0"], sp["s1"]), p2)
            suf = min(certified_suffix(ivs, sp["s0"], sp["s1"]), s2)
            if pre <= sp["s0"] + EPS_S and suf >= sp["s1"] - EPS_S:
                continue                   # this arm certifies neither end
            want = max(0.5 * (loads[src] - loads.get(b, 0.0)) / max(rate, 1e-9),
                       min_split)
            for c, side in split_candidates(sp["s0"], sp["s1"], L, pre, suf,
                                            want, min_split, splice,
                                            n=SPLIT_COARSE_CUTS):
                if spent >= budget:
                    return (best if best and best[0] < key0 else None), spent
                n0 = pricer.n_replans
                r = _try_cut(items, i, side, c, L, src, b, splice, pricer)
                spent += pricer.n_replans - n0
                if r is not None and (best is None or r[0] < best[0]):
                    best = (r[0], r[1], r[3])
                    where = (i, side, b, L, sp, pre, suf)
    if where is None:
        return None, spent
    # The winner is located; now place its cut exactly.  This runs even when no
    # COARSE candidate improved anything: the two coarse cuts are the equalising
    # guess (which has to price a pen-up tour it cannot see) and the receiver's
    # maximum reach (which usually overshoots), and the cut that actually helps
    # is routinely between them.
    i, side, b, L, sp, pre, suf = where
    e = 0.5 * splice / max(L, 1e-9)
    mp = min_split / max(L, 1e-9)
    if side == "head":
        lo, hi = sp["s0"] + mp - e, min(pre - e, sp["s1"] - mp + e)
    else:
        lo, hi = max(suf + e, sp["s0"] + mp - e), sp["s1"] - mp + e
    n0 = pricer.n_replans
    best, _ = _bisect_cut(items, i, side, lo, hi, L, src, b, splice, pricer, best)
    spent += pricer.n_replans - n0
    return (best if best[0] < key0 else None), spent


def rebalance(placed, arms, colors, ivmap, specs, aopts, pens,
              draw_speed=DRAW_SPEED, seq_opts=None, min_seg=MIN_SEG_M,
              h_inv=H_INV_DEFAULT, verbose=False, q_start=None,
              return_home=True, split=True, min_split=MIN_SPLIT_M,
              splice=SPLIT_OVERLAP_M, split_rounds=SPLIT_ROUNDS,
              split_budget=SPLIT_BUDGET, reach=None, cost=None):
    """The probe data + the placed spans -> a re-assignment. -> (placed, info).

    ALLOCATION v2.  The loop is (move | swap | split) until nothing improves:

      1. ALTERNATIVES.  For every placed span, the arms of the right colour
         whose certified intervals already cover it end to end (`frac_covers`
         over `ivmap` — the probe data, not a new guess), each offered a clean
         re-plan of exactly that span and kept only if it certifies all of it.
      2. BALANCE.  `balance_loads` takes relocations and swaps out of the
         busiest arm until neither lowers the objective.
      3. SPLIT.  `_split_search` then asks the finer question the whole-segment
         moves cannot: is there a span on the busiest arm that a lighter arm
         certifies PART of?  If cutting it there lowers the same objective, the
         cut is taken, both halves are re-planned from scratch for the arms that
         will draw them, and the loop goes back to (1) — a new piece is a new
         segment, and its neighbours' alternatives may have changed with it.

    Coverage is invariant through all three (see the section-4c commentary), and
    `coverage_lost` re-derives the merged cover at the end and REFUSES rather
    than return a programme that gave ink back.  `split=False` recovers
    allocation v1 exactly: one pass of steps 1-2 and no cutting.

    `cost` is the run's costing hook (`cost_model`) — the object that will also
    SEQUENCE the bags this function hands out.  Passing the same one to both is
    what stops the balancer from balancing a tour the sequencer then moves; a
    caller that leaves it None gets the module default, which is the model the
    module-level `CLUSTER` switch names.

    `placed` grows when a span is cut, so it is both mutated in place and
    returned; the caller reads its programmes out of the result either way.
    """
    pricer = _Pricer(arms, colors, ivmap, specs, aopts, pens, draw_speed,
                     dict(seq_opts or {}), min_seg, h_inv, q_start,
                     return_home, cost=cost)
    pricer.reach = reach
    items = [dict(it) for it in placed]
    lengths = {int(it["stroke"]["id"]): polyline_length(it["stroke"]["pts"])
               for it in items}
    cover0 = merged_spans(items)
    owner0 = [(it["arm"], float(it["entry"]["length"])) for it in items]
    moves, splits, loads_before, n_movable, spent = [], [], None, 0, 0

    for _ in range(int(split_rounds) + 1):
        options, entries = pricer.options(items)
        owner, info = balance_loads([it["arm"] for it in items], options,
                                    pricer.load_fn(items, entries),
                                    verbose=verbose)
        for i, (it, a) in enumerate(zip(items, owner)):
            it["arm"], it["entry"] = a, entries[i][a]
        if loads_before is None:
            loads_before, n_movable = dict(info["loads_before"]), info["n_movable"]
        moves += info["moves"]
        if not split:
            break
        best, used = _split_search(items, dict(info["loads_after"]), pricer,
                                   float(min_split), float(splice),
                                   max(int(split_budget) - spent, 0))
        spent += used
        if best is None:
            break
        key, items, rec = best
        splits.append(rec)
        if verbose:
            print(f"  split {len(splits):>2}: stroke {rec['stroke']} cut at "
                  f"s={rec['s_cut']:.4f}; arm {rec['frm']} keeps "
                  f"{rec['keep_m']:.3f} m, arm {rec['to']} takes "
                  f"{rec['give_m']:.3f} m; busiest arm now {key[1]:.1f} s")

    lost = coverage_lost(cover0, merged_spans(items), lengths)
    if lost > 1e-9:
        raise ValueError(
            f"the balancer gave back {lost:.6f} m of certified ink; a split may "
            "only ever hand over a piece an arm has re-planned at exactly the "
            "same endpoints (see allocate.split_span)")

    placed[:] = items
    loads_after = pricer.loads(items)
    info = dict(
        loads_before=loads_before or {}, loads_after=loads_after,
        max_before=load_score(loads_before or {})[1],
        max_after=load_score(loads_after)[1],
        moves=moves, rounds=len(moves), splits=splits, n_splits=len(splits),
        n_movable=int(n_movable), n_replans=int(pricer.n_replans),
        n_probes=int(pricer.n_probes), split_calls=int(spent),
        draw_speed=float(draw_speed), cost_model=pricer.cost.name,
        n_menus=int(getattr(pricer.cost, "n_built", 0)),
        min_split_m=float(min_split), splice_m=float(splice),
        coverage_lost_m=float(lost), split_enabled=bool(split),
        n_segments_before=len(owner0), n_segments_after=len(items),
        metres_before={a: float(sum(m for x, m in owner0 if x == a))
                       for a in pricer.arms},
        metres_after={a: float(sum(it["entry"]["length"] for it in items
                                   if it["arm"] == a)) for a in pricer.arms})
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


def sequence_arm(segs, spec, sequencer=SEQUENCER, opts=None, seq_opts=None,
                 forbid=None):
    """One arm's bag of segments -> the programme in the order it will be drawn.

    -> dict(programme, order, dirs, method, cost, baseline_cost, n_reversed,
            n_refused, n, wall).  `cost` and `baseline_cost` are both transit
    SECONDS off the same matrix (`sequence.cost_matrix`), which is the point:
    "the new order saves X %" is then one subtraction inside one model, not a
    comparison of two different accountings.

    `forbid` is a set of `(i, j)` bag-index pairs the tour may not contain —
    "do not draw segment j straight after segment i" — with `i = None` meaning
    "do not start with j".  It is how a conductor's refusal gets back to the
    component that made the choice: a pen-up transit that no schedule can run
    (`coordination.hard_blocks`) is an EDGE of this tour, and the exact solver
    is perfectly happy to give the best tour that avoids it.  Both directions of
    both segments are cut, because which hover pose the transit flies through
    depends on the directions and the refusal was about the geometry.
    """
    n = len(segs)
    t0 = time.time()
    if n == 0:
        return dict(programme=[], order=[], dirs=[], method="empty", n=0,
                    cost=0.0, baseline_cost=0.0, n_reversed=0, n_refused=0,
                    wall=0.0)
    seq_opts = dict(seq_opts or {})
    mat = {k: seq_opts[k] for k in ("transit_speed", "qd_frac", "h_inv",
                                    "pen_ext", "q_start", "return_home")
           if k in seq_opts}
    exact = seq_opts.get("exact_max_n", sequence.EXACT_MAX_N)
    budget = seq_opts.get("budget", sequence.TIME_BUDGET)

    base_order, _ = order_nearest(segs, spec.xy)
    C = sequence.cost_matrix(spec, segs, **mat)
    base_cost = sequence.sequence_cost(C, n, base_order, [1] * n)
    n_forbidden = 0
    for i, j in (forbid or ()):
        if not 0 <= int(j) < n:
            continue
        if i is None:
            C[2 * n, 2 * int(j):2 * int(j) + 2] = np.inf
        elif 0 <= int(i) < n:
            C[2 * int(i):2 * int(i) + 2, 2 * int(j):2 * int(j) + 2] = np.inf
        else:
            continue
        n_forbidden += 1
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
                n_forbidden=int(n_forbidden),
                cost=float(sequence.sequence_cost(C, n, r["order"], dirs)),
                baseline_cost=float(base_cost),
                nn_cost=float(r.get("nn_cost", float("nan"))),
                wall=float(time.time() - t0))


def build_menus(segs, spec, opts=None, n_cand=menu.N_CAND,
                max_sheets=menu.MAX_SHEETS, max_variants=menu.MAX_VARIANTS,
                max_surcharge=menu.MAX_SURCHARGE, verbose=False):
    """One entry/exit fiber menu per segment. -> (menus, stats).

    A segment whose menu comes back empty — or which the band will not span
    from any candidate fiber — keeps the plan the allocator already certified,
    wrapped as a one-variant `menu.PlanMenu`.  That is what makes turning the
    feature on incapable of losing a segment: the worst case is the menu the
    pipeline had before, and the DP that consumes it reduces to the DP that
    consumed that.
    """
    out, n_menu, n_plan, sizes = [], 0, 0, []
    for s in segs:
        m = None
        try:
            m = menu.stroke_menu(s["pts"], spec, opts, n_cand, max_sheets,
                                 max_variants, max_surcharge)
        except Exception:                      # a menu is an optimisation, not
            m = None                           # a promise: never lose a segment
        if m is not None and m.status == "ok" and len(m):
            out.append(m)
            n_menu += 1
            sizes.append(len(m))
        else:
            out.append(menu.PlanMenu(s["plan"]))
            n_plan += 1
            sizes.append(1)
    if verbose:
        print(f"    menus: {n_menu} enumerated, {n_plan} fell back to the "
              f"certified plan; sizes {sizes}")
    return out, dict(n_menu=n_menu, n_plan=n_plan, sizes=sizes,
                     n_variants=int(sum(sizes)),
                     mean_variants=float(np.mean(sizes)) if sizes else 0.0)


def sequence_arm_cluster(segs, spec, menus, opts=None, seq_opts=None,
                         forbid=None, verbose=False):
    """Order, direction AND entry/exit fiber, in one exact DP. -> dict.

    The same contract as `sequence_arm` — it returns a `programme` and the
    transit seconds it costs — with two differences.  The DP chooses among each
    segment's certified variants as well as its two directions, and the chosen
    variant is then MATERIALISED: planned for real through `stroke_api`, corner
    rounding, dense back-out and the independent validator included.

    `cost` is re-derived from the materialised programme with the ordinary
    `sequence.cost_matrix`, not read off the cluster matrix.  That is deliberate
    belt and braces: the advertised endpoints are exact (see `menu.py`), but
    the number this function reports is the one `writing.arm_program` will pay,
    measured on the plans that will actually be executed, and
    `csail_schedule.cross_check` compares the two to 1e-6.  It is also what
    makes a refused reversal or a variant that will not certify cost what it
    actually costs rather than what it was hoped to.

    `baseline_cost` is the nearest-xy chain priced on the ALLOCATOR's plans —
    the programme this arm would have had before any of this existed — so
    `baseline_cost - cost` is the whole L1->L2 saving (order, direction AND
    fiber), not just the ordering part.  The two therefore come off two
    matrices on purpose, which is the one place this differs from
    `sequence_arm`, where both come off one.
    """
    n = len(segs)
    t0 = time.time()
    if n == 0:
        return dict(programme=[], order=[], dirs=[], variants=[], method="empty",
                    n=0, cost=0.0, baseline_cost=0.0, n_reversed=0, n_refused=0,
                    n_rematerialised=0, wall=0.0)
    seq_opts = dict(seq_opts or {})
    mat = {k: seq_opts[k] for k in ("transit_speed", "qd_frac", "h_inv",
                                    "pen_ext", "q_start", "return_home")
           if k in seq_opts}
    exact = seq_opts.get("exact_max_n", sequence.EXACT_MAX_N)
    budget = seq_opts.get("budget", sequence.TIME_BUDGET)

    C, T, e = sequence.cluster_cost_matrix(spec, menus, **mat)
    base = sequence.cost_matrix(spec, segs, **mat)
    base_order, _ = order_nearest(segs, spec.xy)
    base_cost = sequence.sequence_cost(base, n, base_order, [1] * n)
    n_forbidden = 0
    for i, j in (forbid or ()):
        if not 0 <= int(j) < n:
            continue
        b1, b2 = int(e["base"][int(j)]), int(e["base"][int(j) + 1])
        if i is None:
            C[e["N"], b1:b2] = np.inf
        elif 0 <= int(i) < n:
            a1, a2 = int(e["base"][int(i)]), int(e["base"][int(i) + 1])
            C[a1:a2, b1:b2] = np.inf
        else:
            continue
        n_forbidden += 1
    T[~np.isfinite(C)] = 0.0
    r = sequence.cluster_solve(C, T, e, exact, budget)

    prog, dirs, n_rev, n_ref, n_mat = [], [], 0, 0, 0
    for k, d, v in zip(r["order"], r["dirs"], r["variants"]):
        seg, m = segs[k], menus[k]
        plan = seg["plan"]
        if not isinstance(m, menu.PlanMenu):
            cand = m.materialize(v, opts)
            if cand.get("status") == "ok":
                plan, n_mat = cand, n_mat + 1
            # a variant that will not certify is not an error: the segment
            # keeps the plan the allocator already had, and the re-costing
            # below prices whatever it ended up with rather than what was asked
        # `pts` stays the segment's own polyline, exactly as `sequence_arm`
        # leaves it: `endpoints` and `arm_program` read `plan["pts"]`, while
        # `transit_metres` and the JSON read `seg["pts"]`, and swapping in the
        # dense back-out here would make those two report a different quantity
        # in the cluster path than in the plain one for no gain.
        item = dict(seg, plan=plan, variant=int(v))
        rev = reverse_segment(item, spec, opts) if d < 0 else None
        if d < 0 and rev is None:
            n_ref += 1
        if rev is None:
            prog.append(dict(item, flipped=False))
            dirs.append(1)
        else:
            prog.append(rev)
            dirs.append(-1)
            n_rev += 1
    C2 = sequence.cost_matrix(spec, prog, **mat)
    cost = sequence.sequence_cost(C2, n, list(range(n)), [1] * n)
    if verbose:
        print(f"    cluster: {r['method']} {n} segs, {e['N']} nodes, "
              f"transit {cost:.2f} s (baseline {base_cost:.2f} s), "
              f"{n_mat} re-materialised")
    return dict(programme=prog, order=[int(i) for i in r["order"]], dirs=dirs,
                variants=[int(v) for v in r["variants"]], method=r["method"],
                n=n, n_reversed=n_rev, n_refused=n_ref, n_rematerialised=n_mat,
                n_forbidden=int(n_forbidden), n_nodes=int(e["N"]),
                cost=float(cost), cluster_cost=float(r["cost"]),
                baseline_cost=float(base_cost),
                nn_cost=float(r.get("nn_cost", float("nan"))),
                wall=float(time.time() - t0))


def resequence(res, q_start=None, return_home=None, specs=None, forbid=None):
    """Re-order every arm's segments from a different start pose. -> res.

    WHO DRAWS WHAT DOES NOT CHANGE — the same bag of certified spans stays with
    the same arm, so coverage is untouched by construction, exactly as it is for
    the balancer.  What changes is the order, the directions and the transit
    seconds, because both ends of the tour moved: the second pass of a two-pass
    run starts wherever the conductor's idle policy left the arm standing, and
    that is not known until the first pass has been conducted (a minimal retreat
    can move it).  Re-ordering from the BAG rather than from the already-ordered
    programme is what keeps a segment from being reversed twice.

    THE FIBER MENUS ARE RE-ORDERED WITH IT.  A re-sequence that fell back to
    the plain (segment, direction) DP would silently throw the variant choice
    away and restore the entry configurations the allocator happened to plan
    first — which is exactly the reconfiguration the menus exist to remove, and
    it would come back at the one moment it matters most, the start of the
    second pass.  So when `res` carries menus, this re-runs the CLUSTER DP over
    the same menus: a new start pose can make a different fiber the right one
    to enter on, and that is a decision worth re-taking rather than inheriting.

    Mutates and returns `res`.
    """
    specs = FLEET if specs is None else specs
    if "bag" not in res:
        raise KeyError("this allocation was not built with a bag to re-order; "
                       "run allocate() from this version")
    if return_home is None:
        return_home = res.get("return_home", True)
    res["return_home"] = bool(return_home)
    res["q_start"] = {a: np.asarray(v, float) for a, v in (q_start or {}).items()}
    for a in res["arms"]:
        sq = dict(res.get("seq_opts") or {})
        sq["pen_ext"] = res["pens"][a]
        sq["return_home"] = bool(return_home)
        if a in res["q_start"]:
            sq["q_start"] = res["q_start"][a]
        mus = (res.get("menus") or {}).get(a)
        if mus and len(mus) == len(res["bag"][a]) and \
                res.get("sequencer", SEQUENCER) in ("opt", "transit"):
            seq = sequence_arm_cluster(res["bag"][a], specs[a], mus,
                                       res["aopts"][a], sq,
                                       forbid=(forbid or {}).get(a))
        else:
            seq = sequence_arm(res["bag"][a], specs[a],
                               res.get("sequencer", SEQUENCER),
                               res["aopts"][a], sq, forbid=(forbid or {}).get(a))
        res["programs"][a] = seq.pop("programme")
        res["sequence"][a] = seq
        res["transit"][a] = transit_metres(res["programs"][a], specs[a].xy)
        res["transit_time"][a] = seq["cost"]
    return res


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
             repair_budget=REPAIR_BUDGET, balance=True, split=True,
             probe_ref_m=None,
             min_split=MIN_SPLIT_M, splice=SPLIT_OVERLAP_M,
             split_rounds=SPLIT_ROUNDS, split_budget=SPLIT_BUDGET,
             draw_speed=DRAW_SPEED, q_start=None, return_home=True,
             cluster=CLUSTER, menu_opts=None, fleet=None):
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

    `probe_ref_m` turns on DEEP PROBING for strokes much longer than the flat
    budget was chosen for: the per-stroke budget scales with length
    (`stroke_probes`) and a barren gap is bisected rather than abandoned
    (`probe_stroke`'s `bisect`).  They are one knob because they are one fix —
    a stroke twenty times the reference length needs the probe both to keep
    looking and to be allowed to look.  Left None both are off, which is what
    every run before this one used and what the logo's published numbers are.

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

    `split` is allocation v2 (see section 4c): the balancer may also CUT a span
    on the busiest arm and hand one piece to a less-loaded arm that certifies
    it, which is the only move that reaches the 48 % of solo drawing time
    `docs/SOLO_TIME.md` measured as splittable.  `min_split` is the shortest
    piece a cut may create and `splice` the ink drawn twice at the seam.
    Coverage is invariant under it by construction and re-measured before the
    result is returned; `split=False` recovers allocation v1 exactly.
    """
    if arms is not None and active_override is not None:
        raise ValueError("pass arms= or active_override=, not both")
    fl = FLEET if fleet is None else fleet
    arms = list(arms) if arms is not None else active_arms(active_override, fl)
    specs = {a: fl[a] for a in arms}
    if pens is not None:
        unknown = set(pens) - set(fl)
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
                                max_probes=stroke_probes(
                                    polyline_length(st["pts"]), max_probes,
                                    probe_ref_m),
                                min_seg=min_seg, gap_tol=gap_tol,
                                bisect=probe_ref_m is not None)
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

    pen_m = {a: pen_of(pens, a) for a in arms}

    def _mat(a):
        """`sequence.cost_matrix` kwargs for arm `a`, as this run will call it."""
        m = {k: (seq_opts or {})[k] for k in ("transit_speed", "qd_frac",
                                              "h_inv") if k in (seq_opts or {})}
        m.update(pen_ext=pen_m[a], q_start=(q_start or {}).get(a),
                 return_home=bool(return_home))
        return m

    # ---- cover, repair, re-plan — retried without the spans nobody can fly to
    # A SPAN AN ARM CANNOT REACH IS NOT THAT ARM'S SPAN, and the cover is where
    # that belongs: banning it there lets `greedy_cover` hand the paper to
    # somebody else instead of leaving a hole the size of the whole span.  On
    # this logo arm 97's piece of stroke 26 is 0.182 m; banned, arms 2 and 71
    # take all but the 0.050 m neither of them certifies, so the retry is worth
    # 0.132 m of ink over simply dropping what the arm cannot fly to.
    # `prune_unflyable` after the balancer is the guarantee; this is the
    # optimisation, and it is capped because each round re-plans the cover.
    banned, t_repair, t_replan, n_repair = set(), 0.0, 0.0, 0
    programs = {a: [] for a in arms}
    n_replan, placed = 0, []

    def _flyable_ivmap():
        """`ivmap` without the (arm, stroke) pairs the arm cannot fly to."""
        if not banned:
            return ivmap
        return {k: [v for v in vs if (v.arm, k) not in banned]
                for k, vs in ivmap.items()}

    for _round in range(UNFLYABLE_ROUNDS + 1):
        iv_ok = _flyable_ivmap()
        if banned:
            cover = cover_all(strokes, iv_ok, colors, min_seg, gap_tol)
        # the colours are fixed now, so the holes are finally known in the terms
        # that matter — the union of the arms carrying the right ink
        t_rep = time.time()
        cover, n_repair = repair_gaps(
            strokes, iv_ok, colors, cover, specs, aopts, min_seg, gap_tol,
            rounds=repair_rounds,
            min_len=float((opts or {}).get("min_length", 0.02)),
            budget=repair_budget, verbose=verbose)
        t_repair += time.time() - t_rep
        if banned:
            # GAP REPAIR PROBES, SO IT CAN INVENT A SPAN THE BAN NEVER SAW.  It
            # is handed the filtered `ivmap`, but it answers by asking arms
            # about the HOLES directly and certifying whatever comes back — and
            # the hole a banned arm just left is exactly the window it will be
            # offered.  Without this the arm is handed its own unreachable
            # paper straight back, one round after it was taken away.
            for ps in cover["per_stroke"]:
                sid = int(ps["stroke"]["id"])
                ps["chosen"] = [v for v in ps["chosen"]
                                if (v.arm, sid) not in banned]

        # ---- clean re-plans ---------------------------------------------
        t2 = time.time()
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
        t_replan += time.time() - t2

        fresh = set()
        for a in arms:
            mine = [i for i, it in enumerate(placed) if it["arm"] == a]
            if not mine:
                continue
            _, bad = prune_unflyable(specs[a], [placed[i]["entry"] for i in mine],
                                     _mat(a))
            for k in bad:
                it = placed[mine[k]]
                fresh.add((a, int(it["stroke"]["id"])))
        if not fresh or fresh <= banned:
            break
        print("  !! " + ", ".join(f"arm {a} certifies the ink of stroke {s} "
                                  "and cannot fly to it" for a, s in sorted(fresh))
              + "; re-covering without it")
        banned |= fresh

    # ---- load balancing -------------------------------------------------
    t2b = time.time()
    bal = None
    # ONE COST MODEL FOR THE WHOLE RUN.  The balancer below and the sequencing
    # pass at the bottom of this function are handed the same object, so the
    # seconds a bag is priced at are the seconds the DP that draws it will
    # reach.  See the section-4b-i commentary for what this closes.
    cost = cost_model(cluster and sequencer in ("opt", "transit"), menu_opts,
                      verbose)
    if balance and placed:
        # THE BALANCER MAY NOT OFFER BACK WHAT THE COVER JUST BANNED.  Its
        # alternatives come out of `ivmap`, so handed the raw one it would
        # happily relocate stroke 26 to the arm that cannot fly to it and undo
        # the retry above one move later.
        placed, bal = rebalance(placed, arms, colors, _flyable_ivmap(), specs, aopts,
                                pen_m, draw_speed, seq_opts, min_seg,
                                verbose=verbose, q_start=q_start,
                                return_home=return_home, split=split,
                                min_split=min_split, splice=splice,
                                split_rounds=split_rounds,
                                split_budget=split_budget, reach=pre,
                                cost=cost)
        n_replan += bal["n_replans"]
    for it in placed:
        programs[it["arm"]].append(it["entry"])

    # THE GUARANTEE, AFTER EVERY MOVE THE BALANCER MADE.  The retry above bans
    # a span from an arm before the cover so somebody else can have the paper;
    # this is what makes the bag the arm is finally handed one it can actually
    # fly, whatever the balancer's relocations, swaps and splits did to it.  A
    # span released here is not re-offered — the cover is behind us — so it
    # falls to `leftover` and is reported as dropped like any other hole.
    unflyable = []
    for a in arms:
        segs = programs[a]
        if not segs:
            continue
        keep, bad = prune_unflyable(specs[a], segs, _mat(a))
        if not bad:
            continue
        for k in bad:
            e = segs[k]
            unflyable.append(dict(arm=int(a), stroke_id=int(e["stroke_id"]),
                                  s_range=[float(x) for x in e["s_range"]],
                                  length_m=float(e["length"])))
            print(f"  !! arm {a} cannot fly to its own stroke {e['stroke_id']} "
                  f"span {np.round(e['s_range'], 4).tolist()} "
                  f"({e['length']:.4f} m); giving it back")
        programs[a] = [segs[i] for i in keep]
    t_balance = time.time() - t2b

    dropped = leftover(strokes, programs, gap_tol)
    out = dict(colors=colors, arms=arms, table=table, ivmap=ivmap,
               probe_stats=probe_stats, programs={}, dropped=dropped,
               unflyable=unflyable, banned=sorted(banned),
               sequencer=sequencer, pens=pen_m, balance=bal,
               draw_speed=float(draw_speed),
               timing=dict(prefilter=t_pre, probe=t_probe, repair=t_repair,
                           replan=t_replan, balance=t_balance))
    t3 = time.time()
    out["sequence"], out["transit"], out["transit_time"] = {}, {}, {}
    # the unsequenced bag and the per-arm planner options, kept so the pass can
    # be re-ordered from a different start pose without re-probing anything
    # (`resequence`); `programs[a]` below is the bag put in an order.
    out["bag"] = {a: list(programs[a]) for a in arms}
    out["aopts"], out["seq_opts"] = dict(aopts), dict(seq_opts or {})
    out["q_start"] = {a: np.asarray(v, float) for a, v in (q_start or {}).items()}
    out["return_home"] = bool(return_home)
    out["menus"], out["menu_stats"] = {}, {}
    for a in arms:
        sq = dict(seq_opts or {})
        sq["pen_ext"] = out["pens"][a]
        sq["return_home"] = bool(return_home)
        if q_start is not None and a in q_start:
            sq["q_start"] = np.asarray(q_start[a], float)
        seq = cost.sequence(a, programs[a], specs[a], sequencer, aopts[a], sq,
                            verbose=verbose)
        mus = seq.pop("menus", None)
        if mus is not None:
            # the lattices and their sheet decompositions are the memory-heavy
            # part (a 38-sheet band is tens of MB); the variant metadata that
            # `resequence` needs is not, so the menus are kept and the lattices
            # are not.  A later re-materialisation rebuilds, deterministically.
            out["menus"][a] = [m.drop_lattice() if hasattr(m, "drop_lattice")
                               else m for m in mus]
            out["menu_stats"][a] = dict(
                n_menu=sum(1 for m in mus if not isinstance(m, menu.PlanMenu)),
                n_plan=sum(1 for m in mus if isinstance(m, menu.PlanMenu)),
                sizes=[len(m) for m in mus],
                n_variants=int(sum(len(m) for m in mus)),
                mean_variants=float(np.mean([len(m) for m in mus])) if mus
                else 0.0)
        out["programs"][a] = seq.pop("programme")
        out["sequence"][a] = seq
        out["transit"][a] = transit_metres(out["programs"][a], fl[a].xy)
        out["transit_time"][a] = seq["cost"]
    out["timing"]["sequence"] = time.time() - t3
    out["timing"]["total"] = time.time() - t0
    out["total_len"] = float(sum(polyline_length(s["pts"]) for s in strokes))
    out["drawn_len"] = float(sum(s["length"] for a in arms
                                 for s in out["programs"][a]))
    out["dropped_len"] = float(sum(d["length"] for d in dropped))
    out["n_probes"] = (int(sum(p["probes"] for p in probe_stats)) + n_repair
                       + int((bal or {}).get("n_probes", 0)))
    out["n_split_probes"] = int((bal or {}).get("n_probes", 0))
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
        lines.append(f"balance: {b['n_movable']} of "
                     f"{b.get('n_segments_before', n_seg)} segments are "
                     f"certified by more than one arm; {b['rounds']} "
                     f"move{'' if b['rounds'] == 1 else 's'} and "
                     f"{b.get('n_splits', 0)} "
                     f"split{'' if b.get('n_splits', 0) == 1 else 's'} taken, "
                     f"busiest arm {b['max_before']:.1f} s -> "
                     f"{b['max_after']:.1f} s (the phase's floor), draw speed "
                     f"{b['draw_speed']:g} m/s")
        if b.get("n_splits"):
            lines.append(f"      {b['n_segments_before']} segments -> "
                         f"{b['n_segments_after']}; "
                         f"{1000 * b['splice_m']:.0f} mm splice per cut, "
                         f"shortest piece allowed {1000 * b['min_split_m']:.0f} "
                         f"mm, coverage given back "
                         f"{1000 * b['coverage_lost_m']:.3f} mm; the cut search "
                         f"spent {b.get('n_probes', 0)} sub-span probes")
            for m in b["splits"]:
                lines.append(f"        stroke {m['stroke']:>3} cut at "
                             f"s={m['s_cut']:.4f}: arm {m['frm']} keeps "
                             f"{m['keep_m']:.3f} m, arm {m['to']} takes the "
                             f"{m['side']} {m['give_m']:.3f} m -> "
                             f"{m['max_after']:.1f} s")
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
