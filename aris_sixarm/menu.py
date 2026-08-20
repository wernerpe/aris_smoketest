"""Entry/exit fiber menus: the interface between L1 (a stroke) and L2 (an order).

WHAT THE SEQUENCER USED TO BE TOLD, AND WHY IT WAS TOO LITTLE.  `stroke_api`
hands back ONE plan per segment, and with it exactly one entry configuration
and one exit configuration.  `sequence.py` then chooses an order and a
direction — but the two ends it is choosing between were fixed by a band DP
that never heard of the neighbouring strokes.  So the arm can be made to finish
a stroke at q7 = -1.2 and start the next one at q7 = +0.9, and no amount of
re-ordering can mend it, because BOTH numbers were decided before the order
existed.  That is the reconfiguration the timeline pays for twice: once in
transit seconds, and once in a metre of null-space swing that draws nothing.

WHAT IT IS TOLD NOW.  Every stroke end sits on a FIBER — the interval of q7 the
gates leave open there — and the band DP can start or finish anywhere on it.
So instead of one plan, L1 offers a small MENU: a few certified q7 candidates
spread across the feasible interval at each end, every surviving (entry, exit)
pair being one the band actually connects.  L2's state grows from
(segment, direction) to (segment, direction, VARIANT) and the transit cost
becomes a choice rather than a given.

THREE THINGS MAKE THIS AFFORDABLE.

  1. THE BAND DP ALREADY KNOWS REACHABILITY.  Pruning (entry, exit) pairs is
     not a search: one forward sweep of `pwl.travel_forward` from an entry
     leaves a finite cost at exactly the exits that entry can reach under the
     gates, and +inf at the others.  N entries cost N sweeps, not N x M.
  2. THE INTERIOR IS INVARIANT.  Every variant of a stroke draws the same
     polyline at the same speed, and at 0.12-0.15 m/s the joint-velocity limits
     never bind (`docs/REDUNDANCY.md`: peak qdot is 1.2 % of limit on the rim
     arc), so the DRAW time is identical across variants and cancels out of the
     comparison.  What differs is the interior joint travel, carried as a
     per-variant `surcharge`, and the two endpoint configurations.
  3. NOTHING IS PLANNED UNTIL IT IS CHOSEN.  A menu entry is a promise of
     endpoints and a cost, not a trajectory.  The dense IK back-out, the corner
     rounding and the independent validator run for the ONE variant per segment
     the sequencer picks, through the same `stroke_api` code path as before —
     so a materialised variant carries exactly the certificate an eagerly
     planned stroke carries, and `tests/test_menu.py` pins that it is the same
     plan to the float.

THE ENDPOINT CONFIGURATIONS ARE EXACT, NOT ESTIMATES.  A menu advertises the
sheet's representative configuration at (s = 0, j0) and (s = 1, j1).  The
materialised plan seeds its chase from that very node and commands q7 = q7s[j0]
at s = 0, so the first sample IS the advertised configuration; at the far end
the same pose and the same q7 are solved case-consistently on the same branch,
which reproduces the advertised node.  That is what lets the sequencer optimise
against numbers it has not computed yet — and it is asserted, not assumed.
"""
import numpy as np

from . import planner, pwl, stroke_api

N_CAND = 3           # q7 candidates per stroke end, per sheet
MAX_SHEETS = 2       # spanning sheets a menu may draw candidates from
MAX_VARIANTS = 6     # cap after pruning; see `_select`
MAX_SURCHARGE = 0.05 # s of extra interior draw time a variant may cost


def end_candidates(row, n=N_CAND):
    """~`n` q7 indices spread across the feasible fiber at one end. -> (k,) int.

    `row` is one s-row of the gated free region, so every index returned is a
    posture that already clears both gates.  The spread is taken over the
    SORTED feasible indices rather than over the index range, which matters
    because a fiber is often two runs with a hole between them (the two
    elbow-up/elbow-down lobes): picking evenly in index space would spend a
    candidate on the hole, while picking evenly in rank space always lands on
    something real and still reaches both lobes.

    Deterministic, and the ends are always included when there is room, because
    the extremes of the fiber are what give the sequencer somewhere to go.
    """
    idx = np.flatnonzero(np.asarray(row, bool))
    if len(idx) == 0 or n <= 0:
        return np.zeros(0, int)
    if len(idx) <= n:
        return idx.astype(int)
    pick = np.unique(np.round(np.linspace(0, len(idx) - 1, n)).astype(int))
    return idx[pick].astype(int)


def _select(variants, max_variants=MAX_VARIANTS):
    """Trim a menu to `max_variants`, keeping the SPREAD and not just the cheap.

    Sorting by travel and taking the head is the wrong trim: the cheapest six
    variants of a stroke are routinely six ways of entering at the same end of
    the fiber, which hands the sequencer six copies of one choice.  What L2 can
    actually use is one variant per distinct entry and one per distinct exit —
    those are the degrees of freedom the transit cost sees — so each distinct
    j0 contributes its best variant, then each distinct j1 does, and only then
    is the remainder filled by travel.  Ties break on (travel, j0, j1), so the
    menu is a function of the band and nothing else.
    """
    rank = lambda v: (v["travel"], v["j0"], v["j1"])      # noqa: E731
    order = sorted(variants, key=rank)
    if len(order) <= max_variants:
        return order
    keep, kept = [], set()

    def add(v):
        # identity, not equality: a variant dict holds numpy arrays, and `in`
        # would compare them elementwise and raise on the ambiguous truth value
        if id(v) not in kept and len(keep) < max_variants:
            kept.add(id(v))
            keep.append(v)

    for key in ("j0", "j1"):
        seen = set()
        for v in order:
            if v[key] not in seen:
                seen.add(v[key])
                add(v)
    for v in order:
        add(v)
    return sorted(keep, key=rank)


class Menu:
    """One stroke's certified variants, plus the lattice to materialise them.

    `variants` is a list of dicts, cheapest interior travel first:
        sheet          sheet id the variant lives on
        j0, j1         entry / exit q7 INDEX into `lat["q7s"]`
        q7_0, q7_1     the same, in radians
        entry_q (7,)   configuration at s = 0   — exact, see the module docstring
        exit_q  (7,)   configuration at s = 1
        entry_xy, exit_xy   the paper points those configurations draw
        travel         the band DP's interior joint travel (rad)
        surcharge      travel - the menu's cheapest travel (rad, >= 0)

    The object holds the built lattice, so `materialize` costs one certification
    and no IK re-build.  `drop_lattice()` frees it for callers that would rather
    pay the rebuild than carry ~2 MB per segment.
    """

    def __init__(self, ctx, spec, opts, variants, status="ok", reason=""):
        self.ctx, self.spec, self.opts = ctx, spec, dict(opts)
        self.variants, self.status, self.reason = variants, status, reason

    def __len__(self):
        return len(self.variants)

    @property
    def poly(self):
        return None if self.ctx is None else self.ctx["poly"]

    def drop_lattice(self):
        """Release the lattice; `materialize` will rebuild it on demand."""
        if self.ctx is not None:
            self.ctx = dict(self.ctx)
            for k in ("lat", "sheets", "order"):
                self.ctx.pop(k, None)
        return self

    def materialize(self, k, opts=None):
        """Plan variant `k` for real. -> a `stroke_api` result dict.

        Runs the ordinary `stroke_api` tail — band path pinned to this
        variant's entry and exit, corner rounding, the dense case-consistent
        back-out, pacing, and the independent validator — so what comes back is
        an "ok" plan with the same guarantees as any other, or a "split"/"bug"
        exactly as `plan_stroke` would report it.
        """
        if self.status != "ok" or not (0 <= k < len(self.variants)):
            return dict(status="degenerate", reason="no_such_variant")
        v = self.variants[k]
        o = dict(self.opts)
        o.update(opts or {})
        o.update(j_start=int(v["j0"]), j_end=int(v["j1"]),
                 sheet_id=int(v["sheet"]))
        if self.ctx is not None and "lat" in self.ctx:
            # THE EXCEPTION BARRIER IS PART OF THE CONTRACT, so the in-memory
            # path has to have one too: `plan_stroke` wraps `_plan`, and going
            # straight to `plan_from_ctx` would let a certification failure
            # raise out of the sequencer instead of coming back as "bug".
            try:
                r = stroke_api.plan_from_ctx(self.ctx, self.spec, o)
            except Exception as exc:
                return dict(status="bug", reason="exception",
                            error=f"{type(exc).__name__}: {exc}")
        else:
            r = stroke_api.plan_stroke(self.poly, self.spec, o)
        return self._verify(r, v)

    def _verify(self, r, v, tol=1e-9):
        """Refuse a plan that is not the variant that was advertised.

        A `sheet_id` is a RANK among connected components by size, not a stable
        name.  While the lattice is held that is exactly the sheet the menu
        enumerated; after `drop_lattice` the lattice is rebuilt, and if the
        component sizes reorder the same id can name a different sheet — which
        would hand back a plan on another IK branch, silently, after the
        sequencer had already costed the endpoints of this one.  Cheap to rule
        out: the advertised endpoints either are the plan's or they are not.
        """
        if r.get("status") != "ok":
            return r
        qs = np.asarray(r["qs"], float)
        if (np.max(np.abs(qs[0] - v["entry_q"])) > tol
                or np.max(np.abs(qs[-1] - v["exit_q"])) > tol):
            return dict(r, status="bug", reason="variant_drift",
                        notes=list(r.get("notes", []))
                        + ["the materialised plan does not start and end where "
                           "the menu said it would; the sheet numbering moved "
                           "under a lattice rebuild"])
        return r


class PlanMenu:
    """A one-variant menu wrapping a plan that is already certified.

    The degenerate menu — precisely what every segment offered before this
    module existed.  It earns its keep twice: it lets one cluster DP mix
    segments that have real menus with segments that do not, and it is how a
    stroke whose menu came back empty still gets SEQUENCED rather than dropped.
    `materialize` hands back the plan it was given, so nothing is re-planned
    and nothing is re-certified.
    """

    status, reason = "ok", "from_plan"

    def __init__(self, plan):
        self.plan = plan
        qs = np.asarray(plan["qs"], float)
        pts = np.asarray(plan["pts"], float)
        q7 = np.asarray(plan.get("q7", [0.0, 0.0]), float)
        self.variants = [dict(sheet=int(plan.get("sheet", -1)), j0=-1, j1=-1,
                              q7_0=float(q7[0]), q7_1=float(q7[-1]),
                              entry_q=qs[0].copy(), exit_q=qs[-1].copy(),
                              entry_xy=pts[0].copy(), exit_xy=pts[-1].copy(),
                              travel=float(plan.get("sum_travel", 0.0)),
                              surcharge=0.0)]

    def __len__(self):
        return 1

    def materialize(self, k, opts=None):
        return self.plan


def stroke_menu(pts_xy, spec, opts=None, n_cand=N_CAND, max_sheets=MAX_SHEETS,
                max_variants=MAX_VARIANTS, max_surcharge=MAX_SURCHARGE):
    """Certified entry/exit fiber menu for one stroke. -> `Menu`.

    `status` is "ok" with at least one variant, or mirrors whatever
    `stroke_api.prepare` refused the stroke for ("degenerate"/"split"), or
    "no_variant" when the band spans the stroke on no sheet the menu looked at.
    A caller that wants the old behaviour asks for `n_cand=1`, which yields the
    single cheapest variant and reduces L2 to exactly the DP it ran before.
    """
    o = dict(stroke_api.DEFAULTS)
    o.update(opts or {})
    ctx, early = stroke_api.prepare(pts_xy, spec, o)
    if early is not None:
        return Menu(None, spec, o, [], status=early["status"],
                    reason=early.get("reason", ""))

    lat, Ns = ctx["lat"], ctx["Ns"]
    q7s = lat["q7s"]
    variants = []
    spanning = [sh for sh in ctx["order"] if sh["spans_s"]][:max(1, max_sheets)]
    for sh in spanning:
        # THE GATES AND THE CONTINUITY BUDGET ARE READ FROM `pwl`, NOT FROM
        # `opts`.  `_sheet_pass` does not forward per-call gates to `plan_pwl`
        # either, so accepting them here would let a menu be ENUMERATED against
        # one band and MATERIALISED against another — the one way these two
        # halves can silently disagree about which fibers exist.  One source
        # for both, or the promise in this module's docstring is void.
        free = pwl.band_free(sh, pwl.SIGMA_GATE, pwl.MARGIN_GATE)
        if not (free[0].any() and free[-1].any()):
            continue
        clear = pwl.clearance_map(free)
        edges = pwl._edge_ok(sh, planner.JUMP_THRESH) & free[:-1][:, :, None]
        etr = pwl._edge_travel(sh, o.get("travel_mode", pwl.TRAVEL_MODE))
        exits = end_candidates(free[-1], n_cand)
        for j0 in end_candidates(free[0], n_cand):
            A, _C, _par, last = pwl.travel_forward(free, clear, edges, etr,
                                                   pwl.W_CLEARANCE,
                                                   j_start=int(j0))
            if last < Ns - 1:
                continue          # this entry cannot cross the band at all
            for j1 in exits:
                if not np.isfinite(A[j1]):
                    continue      # PRUNED: the band does not connect the pair
                variants.append(dict(
                    sheet=int(sh["id"]), j0=int(j0), j1=int(j1),
                    q7_0=float(q7s[j0]), q7_1=float(q7s[j1]),
                    entry_q=np.asarray(sh["Q"][0, j0], float).copy(),
                    exit_q=np.asarray(sh["Q"][Ns - 1, j1], float).copy(),
                    entry_xy=np.asarray(ctx["poly"][0], float).copy(),
                    exit_xy=np.asarray(ctx["poly"][-1], float).copy(),
                    travel=float(A[j1]) / pwl._TRAVEL_Q))
    if not variants:
        return Menu(ctx, spec, o, [], status="no_variant",
                    reason="no sheet spans the stroke under the gates")
    # THE PREMISE HAS TO BE MADE TRUE, NOT ASSUMED.  "Interior draw time is
    # invariant across variants" is what lets L2 choose a fiber on transit
    # alone — and it is FALSE in general: a variant is a different path through
    # the band, so it has a different |dq/ds|, and `writing.draw_duration`
    # stretches the ink until no joint exceeds `qd_frac` of its limit.  Left
    # uncapped, the sequencer will happily buy a second of transit with ten
    # seconds of drawing.  With `travel_mode="time"` the band DP costs an edge
    # in SECONDS at full joint speed, so the surcharge is exactly the extra
    # draw time the variant will cost, and variants that cost more than
    # `max_surcharge` of it are dropped here rather than priced downstream.
    # What is left is a menu the premise is true of, which is the only kind L2
    # can safely optimise over.
    lo = min(v["travel"] for v in variants)
    for v in variants:
        v["surcharge"] = float(v["travel"] - lo)
    if max_surcharge is not None:
        variants = [v for v in variants if v["surcharge"] <= max_surcharge]
    keep = _select(variants, max_variants)
    return Menu(ctx, spec, o, keep)


def menu_report(menus):
    """{key: Menu} -> terse lines."""
    out = [f"{'segment':>10} {'variants':>9} {'sheets':>7} {'entries':>8} "
           f"{'exits':>6} {'surcharge rad':>14}"]
    for k in sorted(menus, key=str):
        m = menus[k]
        if m.status != "ok":
            out.append(f"{str(k):>10} {m.status:>9}")
            continue
        v = m.variants
        out.append(f"{str(k):>10} {len(v):>9} {len({x['sheet'] for x in v}):>7} "
                   f"{len({x['j0'] for x in v}):>8} {len({x['j1'] for x in v}):>6} "
                   f"{max(x['surcharge'] for x in v):>14.3f}")
    return out
