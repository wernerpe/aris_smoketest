"""One entry point for one stroke: `plan_stroke(pts_xy, spec, opts) -> dict`.

THE CONTRACT IS CERTIFIED-OR-SPLIT.  Six arms drawing hundreds of strokes
multiply their per-stroke failure probabilities together, so the caller (the
allocator, roadmap item 3) cannot be handed a plan that is "probably fine", and
it cannot be handed an exception either.  Every call returns a dict whose
`status` is one of exactly four things:

  "ok"          a dense joint trajectory with a clock, and an INDEPENDENT
                validation report (`validate.validate_plan`) that passed.  The
                pen is on the curve to < 2 mm, margin >= 0.15, sigma_min >= 0.10,
                ||dq||_inf <= 0.35 between samples, the arm clears the paper and
                its own boom, and |dq/dt| is inside the FR3 velocity limits.
  "split"       the stroke is not plannable end to end by this arm.  `s_star` is
                the normalised arc length up to which it IS planned — the end of
                the certified `head`, obtained by calling this same entry point
                on the truncated stroke, so a head plan is an "ok" result with
                all of its guarantees.  s_star = 0 with `head` None means
                nothing at all was certified.  `reason` says what stops the
                stroke and `s_reach` is where the redundancy band itself gives
                out (>= s_star, and the number to reallocate against).  s_star
                is deliberately the *conservative* of the two: a caller that
                hands [0, s_star] to the arm and the rest to someone else is
                never trusting an uncertified metre of paper.
  "degenerate"  nothing to plan: empty, a single point, all-duplicate,
                shorter than `min_length`, or entirely off the sheet.
  "bug"         an exception escaped, or the plan failed its own validator.
                `traceback` and `validation` are attached.  A campaign's job is
                to drive this count to zero; it is never an expected outcome.

WHY THE VALIDATOR IS PART OF THE CONTRACT AND NOT A TEST.  `smooth.certify`
enforces the gates it knows about, along the samples it chose, using the numbers
it computed.  `validate.validate_plan` re-derives all of them from `frames.fk`
and the raw joint samples with no planner state.  Requiring the second to pass
before the first may be called "ok" is what makes a "bug" status possible at
all — without it a bookkeeping error is indistinguishable from a plan.

The pipeline itself is unchanged and un-special: resample -> `build_lattice` ->
`sheet_fields` -> `plan_pwl` -> `smooth_q7_of_s` -> `smooth.certify` (which IS
the dense IK back-out) -> `pacing.pace` -> `validate_plan`.  This module adds
the input hygiene, the sheet fallback, the split bookkeeping and the exception
barrier around it.
"""
import traceback

import numpy as np

from . import fleet, pacing, planner, pwl, smooth
from .frames import PEN_EXT
from .validate import validate_plan

DEFAULTS = dict(
    ds_lattice=0.010,      # m, arc-length step of the (s x q7 x branch) lattice
    ds_dense=0.005,        # m, execution/back-out resolution
    n_q7=planner.N_Q7,     # q7 samples across the FR3 range
    pen_ext=PEN_EXT,
    h_inv=None,            # inverted-mount height (None -> fleet default)
    clip_to_sheet=True,    # keep the longest in-sheet run before planning
    smooth_frac=smooth.ROUND_FRAC,
    v_draw=pacing.V_DRAW,
    safety=pacing.SAFETY,
    min_length=0.02,       # m, shorter strokes are "degenerate", not "split"
    max_steps=1500,        # lattice steps (15 m at ds_lattice) before we refuse
    dup_tol=1e-9,          # m, consecutive points closer than this are one point
    max_sheets=3,          # IK sheets to try before declaring a split
    max_split_depth=3,     # recursive head re-plans
    split_back=0.010,      # m pulled back from s* before re-planning the head
    validate=True,
    keep_debug=False,      # attach the lattice/sheet objects (memory-heavy)
    objective=pwl.OBJECTIVE,   # band objective; see `_sheet_pass`
    travel_mode=pwl.TRAVEL_MODE,          # how min_travel charges an edge
    fallback_objective="maximin_sigma",   # tried when the first certifies nothing
    j_start=None,          # pin the entry q7 index (fiber menus; see menu.py)
    j_end=None,            # pin the exit  q7 index
    sheet_id=None,         # pin the IK sheet; REQUIRED whenever j_start/j_end are
    tilt_max_deg=0.0,      # pen-tilt cone half-angle (deg); see below
)

# PEN TILT IS OPT-IN AND THE DEFAULT IS THE PEN POINTING STRAIGHT DOWN.
# `tilt_max_deg = 0` is not merely the recommended setting, it is a different
# code path: nothing below ever consults `aris_sixarm/tilt.py`, so a stroke
# planned with the default is planned by exactly the pipeline this module has
# always been.  Above zero, `tilt.plan_adaptive` plans the stroke FLAT FIRST
# through this same entry point's internals and only opens the tilt axis where
# the flat attempt was not enough, returning the flat result whenever tilt
# fails to improve on it — so the flag can add certified strokes and cannot
# remove one.  See docs/TILT_EXPLORATION.md for what it buys (arm 2's comfort
# donut: 0 of 4 test strokes certify at the strict gate without it, 4 of 4
# with a 15-degree cone) and what it costs (a rescue is 110-280 ms against a
# 13-57 ms flat plan; strokes that never need it pay nothing measurable).

SPLIT_REASONS = ("start_infeasible", "empty_fiber", "sheet_collapse",
                 "chase_failed")


# --------------------------------------------------------------------------
# polyline hygiene — the operations the pipeline assumes have already happened
# --------------------------------------------------------------------------
def polyline_length(pts):
    p = np.asarray(pts, float)
    if len(p) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def sanitize(pts, dup_tol=DEFAULTS["dup_tol"]):
    """Drop non-finite rows and zero-length segments. -> (pts, notes).

    Duplicate consecutive points are the classic degenerate input: they make
    `planner.resample`'s cumulative arc length non-strictly-increasing, and an
    all-duplicate polyline divides by a zero length and returns NaN parameters
    that then propagate silently through every downstream stage.
    """
    p = np.asarray(pts, float)
    notes = []
    good = np.isfinite(p).all(axis=1)
    if not good.all():
        notes.append(f"dropped {int((~good).sum())} non-finite points")
        p = p[good]
    if len(p) > 1:
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        keep = np.concatenate([[True], seg > dup_tol])
        if not keep.all():
            notes.append(f"dropped {int((~keep).sum())} duplicate points")
            p = p[keep]
    return p, notes


def truncate_polyline(pts, s0=0.0, s1=1.0):
    """Sub-polyline between two normalised arc lengths, endpoints interpolated.

    Used for the head re-plan ([0, s*)) and, by the fuzz campaign, for the tail
    ((s*, 1]).  Returns a (K,2) array with K >= 1; K < 2 means the requested
    span has no length and the caller should treat it as degenerate.
    """
    p = np.asarray(pts, float)
    if len(p) < 2:
        return p
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    t = np.concatenate([[0.0], np.cumsum(seg)])
    L = t[-1]
    if L <= 0:
        return p[:1]
    a, b = sorted((float(np.clip(s0, 0, 1)) * L, float(np.clip(s1, 0, 1)) * L))
    if b - a <= 0:
        return p[:1]
    inner = np.flatnonzero((t > a) & (t < b))
    xy = [np.array([np.interp(a, t, p[:, 0]), np.interp(a, t, p[:, 1])])]
    xy += [p[i] for i in inner]
    xy.append(np.array([np.interp(b, t, p[:, 0]), np.interp(b, t, p[:, 1])]))
    out, _ = sanitize(np.array(xy))
    return out


def _fit_ds(L, ds):
    """The step nearest `ds` that divides a length of `L` a whole number of
    times, so `planner.resample` lands exactly on both ends of the stroke."""
    n = max(int(round(L / ds)), 1)
    return float(L) / n


def _is_multi_stroke(obj):
    """A list of polylines (what `letters.place` returns) is not one stroke."""
    if isinstance(obj, np.ndarray):
        return obj.ndim == 3
    if isinstance(obj, (list, tuple)) and len(obj) > 0:
        try:
            return all(np.ndim(x) == 2 for x in obj)
        except Exception:
            return False
    return False


# --------------------------------------------------------------------------
# the entry point
# --------------------------------------------------------------------------
def plan_stroke(pts_xy, spec, opts=None, _depth=0):
    """Plan one stroke for one arm.  Never raises; see the module docstring."""
    o = dict(DEFAULTS)
    o.update(opts or {})
    try:
        if float(o.get("tilt_max_deg", 0.0)) > 0 and _depth == 0:
            # _depth guards the recursion: a split's head is re-planned through
            # this entry point, and `tilt.plan_adaptive` reaches `prepare` /
            # `plan_from_ctx` directly, so only the outermost call may branch.
            from . import tilt as _tilt
            return _tilt.plan_adaptive(pts_xy, spec, o["tilt_max_deg"], opts=o)
        return _plan(pts_xy, spec, o, _depth)
    except Exception as exc:                       # the exception barrier
        return dict(status="bug", reason="exception", error=f"{type(exc).__name__}: {exc}",
                    traceback=traceback.format_exc(), arm=getattr(spec, "arm_id", None),
                    depth=_depth)


def _degenerate(reason, spec, depth, **kw):
    return dict(status="degenerate", reason=reason,
                arm=getattr(spec, "arm_id", None), depth=depth, **kw)


def _split_from(ctx, spec, o, depth, s_reach, reason, **kw):
    """The "split" result, with its head certified through this same entry."""
    s_reach = float(np.clip(s_reach, 0.0, 1.0))
    head, s_cert = _head_plan(ctx["poly"], spec, o, depth, s_reach, ctx["L"])
    return dict(ctx["base"], status="split", s_star=s_cert, s_reach=s_reach,
                reason=reason, head=head, **kw)


def prepare(pts_xy, spec, o, depth=0):
    """Everything before an objective is chosen. -> (ctx, early).

    Hygiene -> resample at the lattice step -> sheet clip -> `build_lattice` ->
    `sheet_fields`.  `early` is a finished result dict (degenerate, or a split
    the band never got far enough to argue with) and `ctx` is None; otherwise
    `early` is None and `ctx` carries poly, pts, Ns, L, base, notes, lat,
    sheets, order, fiber_cut, ds_dense and md.

    WHY THIS IS A FUNCTION AND NOT A COMMENT.  `menu.py` needs exactly this
    prefix and nothing after it, and the q7 INDICES it hands to the sequencer
    have to mean the same thing they mean inside `_plan` — same resampling,
    same clip, same lattice, same sheet numbering.  Two implementations that
    "do the same preprocessing" would agree until the day one of them was
    edited, and the failure would be a silently mismatched entry
    configuration, which is the hardest kind of wrong number to see.  Sharing
    the code is the only way to promise it.
    """
    notes = []
    # ---- 0. shape / hygiene ------------------------------------------------
    if _is_multi_stroke(pts_xy):
        return None, _degenerate("not_a_single_stroke", spec, depth,
                                 notes=["input is a list of polylines (e.g. "
                                        "letters.place(...)); plan each separately"])
    try:
        arr = np.asarray(pts_xy, float)
    except Exception:
        return None, _degenerate("bad_shape", spec, depth,
                                 notes=["input is not numeric"])
    if arr.ndim != 2 or arr.size == 0 or arr.shape[1] != 2:
        return None, _degenerate("bad_shape" if arr.size else "empty", spec, depth,
                                 notes=[f"input shape {arr.shape}, want (N,2)"])
    poly, notes = sanitize(arr, o["dup_tol"])
    if len(poly) < 2:
        return None, _degenerate("too_few_points", spec, depth, notes=notes)
    L_in = polyline_length(poly)
    if L_in < o["min_length"]:
        return None, _degenerate("too_short", spec, depth, notes=notes, arc_len=L_in)

    # ---- 1. lattice sampling, then the sheet clip --------------------------
    # `planner.resample` steps by a fixed ds and stops at the last WHOLE step,
    # so a stroke whose length is not a multiple of ds loses its tail: the
    # lattice would cover [0, 0.160 m] of a 0.168 m stroke while the 5 mm
    # back-out covers [0, 0.165 m], and the difference is planned on faith at
    # exactly the place — the far end of a stroke — where the reach boundary
    # tends to be.  Fitting a whole number of steps into the length makes both
    # grids land on both ends of the stroke that was actually asked for.
    ds_lat = _fit_ds(L_in, o["ds_lattice"])
    if L_in / ds_lat > o["max_steps"]:
        # A caller can hand us anything, including a 200 m line through the
        # paper: resampling that at 10 mm would allocate 20 000 lattice steps
        # before anyone notices only 4 m of it is on the sheet.  Clip coarsely
        # first, at whatever step keeps the array bounded.
        coarse, _ = planner.resample(poly, L_in / o["max_steps"])
        kept = planner.clip_to_sheet(coarse, verbose=False,
                                     sheet=fleet.sheet_for(spec))
        if len(kept) < 2:
            return None, _degenerate("off_sheet", spec, depth, notes=notes,
                                     arc_len=L_in)
        notes.append(f"stroke is {L_in:.1f} m; coarse-clipped to the sheet first")
        poly, L_in = kept, polyline_length(kept)
        ds_lat = _fit_ds(L_in, o["ds_lattice"])
        if L_in / ds_lat > o["max_steps"]:
            return None, _degenerate("too_long", spec, depth, notes=notes,
                                     arc_len=L_in)
    pts, _ = planner.resample(poly, ds_lat)
    if len(pts) < 2:
        return None, _degenerate("too_short", spec, depth, notes=notes, arc_len=L_in)
    clip_s = (0.0, 1.0)
    if o["clip_to_sheet"]:
        kept, sl = planner.clip_to_sheet(pts, verbose=False, return_slice=True,
                                         sheet=fleet.sheet_for(spec))
        if len(kept) < 2:
            return None, _degenerate("off_sheet", spec, depth, notes=notes,
                                     arc_len=L_in)
        if len(kept) != len(pts):
            n = len(pts) - 1
            clip_s = (sl.start / n, (sl.stop - 1) / n)
            notes.append(f"clipped to the sheet: s in [{clip_s[0]:.3f}, "
                         f"{clip_s[1]:.3f}] of the input ({len(kept)}/{len(pts)} steps)")
            poly, pts = kept, kept
            if polyline_length(poly) < o["min_length"]:
                return None, _degenerate("too_short_after_clip", spec, depth,
                                         notes=notes,
                                         arc_len=polyline_length(poly))
    L = polyline_length(poly)
    Ns = len(pts)
    base = dict(arm=getattr(spec, "arm_id", None), depth=depth, arc_len=L,
                arc_len_input=L_in, clip_s=clip_s, n_lattice=Ns, notes=notes,
                stroke=poly)
    ctx = dict(poly=poly, pts=pts, Ns=Ns, L=L, base=base, notes=notes)

    # ---- 2. the lattice ----------------------------------------------------
    lat = planner.build_lattice(pts, spec, h_inv=o["h_inv"], pen_ext=o["pen_ext"],
                                n_q7=o["n_q7"])
    fiber = lat["valid"].any(axis=(1, 2))
    if not fiber[0]:
        # where the arm could pick the stroke up again — the tail the caller
        # should re-offer (to this arm after a pen-up, or to a neighbour).
        nxt = np.flatnonzero(fiber)
        return None, _split_from(
            ctx, spec, o, depth, 0.0, "start_infeasible", fiber_cut=0,
            s_resume=float(nxt[0] / (Ns - 1)) if len(nxt) else None)
    fiber_cut = int(np.flatnonzero(~fiber)[0] - 1) if not fiber.all() else Ns - 1
    base["fiber_cut_s"] = fiber_cut / (Ns - 1)

    # ---- 3. sheets ---------------------------------------------------------
    sheets = pwl.sheet_fields(lat)
    if not sheets:
        return None, _split_from(ctx, spec, o, depth, 0.0, "start_infeasible",
                                 fiber_cut=0)
    # A sheet is tried WHOLE — polyline and dense certification together —
    # before the next one is considered.  The grid says a corridor exists at
    # the lattice's 10 mm sampling; the 5 mm chase is the one that has to hold,
    # and a sheet can pass the first and fail the second.  Conceding a split
    # while another sheet would have carried the stroke end to end is the
    # difference between an honest split and a lazy one.
    ctx.update(lat=lat, sheets=sheets, fiber_cut=fiber_cut,
               order=sorted(sheets, key=lambda sh: (not sh["spans_s"], -sh["nodes"])),
               ds_dense=_fit_ds(L, o["ds_dense"]),
               md=max(int(round(L / _fit_ds(L, o["ds_dense"]))), 1))
    return ctx, None


def _plan(pts_xy, spec, o, depth):
    ctx, early = prepare(pts_xy, spec, o, depth)
    if early is not None:
        return early
    return plan_from_ctx(ctx, spec, o, depth)


def plan_from_ctx(ctx, spec, o, depth=0):
    """The half of `_plan` after `prepare`: choose a band path and certify it.

    Split out so a fiber-menu variant can be materialised from a lattice that
    has already been built (`menu.py`), instead of paying for the IK twice.
    With `o["j_start"]` / `o["j_end"]` set this plans exactly the variant the
    sequencer costed; with them None it is the ordinary whole-stroke plan.
    """
    poly, pts, Ns, L = ctx["poly"], ctx["pts"], ctx["Ns"], ctx["L"]
    # A ctx is planned from MORE THAN ONCE (one fiber-menu variant per call),
    # so the notes list has to be copied: shared, a note appended by the second
    # materialisation appears retroactively in the plan the first one returned.
    notes = list(ctx["notes"])
    base = dict(ctx["base"], notes=notes)
    lat, sheets = ctx["lat"], ctx["sheets"]
    order, fiber_cut = ctx["order"], ctx["fiber_cut"]
    ds_dense, md = ctx["ds_dense"], ctx["md"]

    def split(s_reach, reason, **kw):
        return _split_from(ctx, spec, o, depth, s_reach, reason, **kw)

    # ---- 4. PWL -> corner rounding -> dense certification -------------------
    res, sh, sm, best = _sheet_pass(poly, spec, o, lat, order, ds_dense, md, Ns,
                                    fiber_cut, o["objective"], notes)
    fb = o.get("fallback_objective")
    if sm is None and fb and fb != o["objective"]:
        # THE FALLBACK IS WHY THE OBJECTIVE CHANGE CANNOT COST COVERAGE.  The
        # gated shortest path is entitled to run along the gate boundary, and
        # the 5 mm chase samples between the lattice's 10 mm nodes; when that
        # loses a stroke on every sheet, the older bottleneck objective — which
        # buys margin by construction — is tried on exactly the same sheets and
        # the same gates before a split is conceded.  A split is therefore
        # never the new objective's doing: it is a statement about the band.
        r2, s2, m2, b2 = _sheet_pass(poly, spec, o, lat, order, ds_dense, md, Ns,
                                     fiber_cut, fb, notes)
        if m2 is not None:
            notes.append(f"the {o['objective']} band plan certified on no sheet; "
                         f"fell back to {fb}")
            res, sh, sm = r2, s2, m2
        elif b2[0] > best[0]:
            best = b2
    if sm is None:
        return split(best[0], best[1], fiber_cut=fiber_cut, n_sheets=len(sheets),
                     **best[2])

    knots = res["knots"]
    qs, dense_pts = sm["qs"], sm["pts"]

    # ---- 4. the clock ------------------------------------------------------
    pc = pacing.pace(qs, sm["arc_len"], v_draw=o["v_draw"], safety=o["safety"],
                     ds_m=sm["ds"], dqds=sm["dqds"])

    # ---- 5. the independent certificate ------------------------------------
    rep = validate_plan(dense_pts, spec, qs, times=pc["t"], h_inv=o["h_inv"],
                        pen_ext=o["pen_ext"]) if o["validate"] else dict(ok=True)
    out = dict(base, status="ok", reason="", knots=knots, qs=qs, pts=dense_pts,
               times=pc["t"], s=sm["s"], q7=sm["q7"], sigmas=sm["sigmas"],
               margins=sm["margins"], min_sigma=float(sm["min_sigma"]),
               min_margin=float(sm["min_margin"]), tip_err=float(sm["tip_err"]),
               max_step=float(sm["max_step"]), sum_travel=float(sm["sum_travel"]),
               n_knots=int(len(knots)), n_dense=int(len(qs)),
               sheet=int(sh["id"]), n_sheets=len(sheets),
               objective=res["objective"], dp_travel=float(res["dp_travel"]),
               q7_span=float(np.ptp(sm["q7"])),
               fallbacks=int(sm["fallbacks"]),
               windows=np.asarray(sm["windows"], float),
               n_bisect=int(sm["n_bisect"]),
               total_time=float(pc["total_time"]),
               frac_slowed=float(pc["frac_slowed"]),
               headroom=float(pc["headroom"]), validation=rep)
    if o["keep_debug"]:
        out.update(lat=lat, sheet_obj=sh, pwl=res, smooth=sm, pace=pc)
    # coverage: an "ok" plan draws the WHOLE stroke.  The tip-on-curve check is
    # pointwise against the dense samples, so it cannot see a plan that quietly
    # stopped short of the ends — this can.
    gap = max(float(np.linalg.norm(dense_pts[0] - poly[0])),
              float(np.linalg.norm(dense_pts[-1] - poly[-1])))
    out["coverage_gap"] = gap
    if gap > 1e-6:
        out.update(status="bug", reason="incomplete_coverage")
    if not rep["ok"]:
        out.update(status="bug", reason="validation_failed")
    return out


def _sheet_pass(poly, spec, o, lat, order, ds_dense, md, Ns, fiber_cut,
                objective, notes):
    """Try each candidate sheet WHOLE under one band objective.

    -> (pwl result, sheet, certified chase, best) — the first three None when
    no sheet carried the stroke end to end, and `best` the (reach, reason,
    extras) triple the caller turns into a split.  Factored out of `_plan` so
    the fallback objective is literally the same pass run again rather than a
    second implementation of it that could drift.
    """
    res = sh = sm = None
    best = (-1.0, "sheet_collapse", {})            # (s_reach, reason, extras)
    if o.get("sheet_id") is not None:
        # PINNING THE INDICES IS NOT ENOUGH — PIN THE SHEET.  A q7 index names
        # a column of the band, not a posture: two sheets can both be free at
        # (0, j0) and hold configurations a branch flip apart (measured: q7
        # exactly pi apart on a CSAIL segment).  A menu advertises the entry
        # configuration of ONE sheet, so materialising the variant on whichever
        # sheet happens to be tried first would hand the sequencer a plan that
        # starts somewhere other than where it was costed.  Silent, and worth
        # a whole schedule.
        order = [sh_ for sh_ in order if int(sh_["id"]) == int(o["sheet_id"])]
        if not order:
            return None, None, None, (-1.0, "sheet_collapse",
                                      dict(sheet=int(o["sheet_id"])))
    for cand in order[:max(1, o["max_sheets"])]:
        r = pwl.plan_pwl(lat, cand, jump=planner.JUMP_THRESH,
                         objective=objective, j_start=o.get("j_start"),
                         j_end=o.get("j_end"),
                         travel_mode=o.get("travel_mode", pwl.TRAVEL_MODE))
        if not r["ok"]:
            reach = int(r["cut_index"]) / (Ns - 1)
            reason = "empty_fiber" if int(r["cut_index"]) >= fiber_cut \
                and fiber_cut < Ns - 1 else "sheet_collapse"
            if reach > best[0]:
                best = (reach, reason, dict(sheet=int(cand["id"])))
            continue
        c, info = _certify_sheet(poly, spec, o, lat, cand, r, ds_dense)
        if c is not None:
            if info == "sharp":
                notes.append("corner rounding did not certify; kept the sharp "
                             "polyline")
            return r, cand, c, best
        reach = max(int(info.get("cut_index", -1)), 0) / md
        if reach > best[0]:
            best = (reach, "chase_failed",
                    dict(sheet=int(cand["id"]), n_knots=int(len(r["knots"])),
                         chase_stop=info["history"][-1][3] if info.get("history")
                         else ""))
    return res, sh, sm, best


def _certify_sheet(poly, spec, o, lat, sheet, res, ds_dense):
    """Round the PWL's corners and certify the result at `ds_dense`.

    -> (chase dict, "rounded"|"sharp") on success, (None, best failed chase) on
    failure.  The sharp polyline is the fallback because `plan_pwl` already
    certified it at the lattice's resolution: if rounding is what breaks the
    stroke, un-rounding is the repair, and `smooth.certify`'s own bisection has
    already tried the intermediate windows.
    """
    curve = smooth.smooth_q7_of_s(res["knots"], frac=o["smooth_frac"])
    sm = smooth.certify(poly, spec, curve, lat=lat, sheet=sheet, ds=ds_dense,
                        h_inv=o["h_inv"], pen_ext=o["pen_ext"])
    if sm.get("certified"):
        return sm, "rounded"
    sharp = curve.with_windows(np.zeros(len(curve.windows)))
    sm2 = smooth.certify(poly, spec, sharp, lat=lat, sheet=sheet, ds=ds_dense,
                         h_inv=o["h_inv"], pen_ext=o["pen_ext"])
    if sm2.get("certified"):
        return sm2, "sharp"
    far = sm2 if int(sm2.get("cut_index", -1)) > int(sm.get("cut_index", -1)) else sm
    return None, far


def _head_plan(poly, spec, o, depth, s_reach, L):
    """Certified plan for [0, s_reach), via this same entry point.

    Pulled back by `split_back` metres so the head does not end exactly on the
    step that failed, and flattened: if the truncated stroke splits again, its
    own head (already certified, or None) is what comes back.  Returns
    (head, s_star) where s_star is the head's extent in the PARENT's normalised
    arc length — 0.0 when nothing could be certified, which is what makes
    "everything below s_star is certified" true rather than hopeful.
    """
    none = (None, 0.0)
    if depth >= o["max_split_depth"] or s_reach <= 0.0:
        return none
    back = min(o["split_back"] / max(L, 1e-9), 0.5 * s_reach)
    s_head = s_reach - back
    if s_head * L < o["min_length"]:
        return none
    sub = truncate_polyline(poly, 0.0, s_head)
    if len(sub) < 2:
        return none
    # THE PINS DO NOT SURVIVE A TRUNCATION.  `j_start`, `j_end` and `sheet_id`
    # name a q7 column and a connected component of THIS stroke's lattice; the
    # head is a shorter stroke with its own lattice, its own sheet numbering
    # and its own fiber at the far end, so carrying them over constrains the
    # re-plan to an exit that has no meaning there.  Measured: a variant whose
    # exit was unreachable recursed to full depth still pinned to the parent's
    # j_end, so every head failed and s_star collapsed to 0.0 where a genuine
    # head would have certified.  A head is planned free.
    sub_o = dict(o)
    for k in ("j_start", "j_end", "sheet_id"):
        sub_o[k] = None
    r = plan_stroke(sub, spec, sub_o, _depth=depth + 1)
    head = r if r["status"] == "ok" else r.get("head")
    if head is None:
        return none
    return head, float(min(1.0, head["arc_len"] / max(L, 1e-9)))


# --------------------------------------------------------------------------
# direction agnosticism: a certified plan, executed the other way round
# --------------------------------------------------------------------------
def reverse_plan(plan, spec=None, opts=None, validate=True):
    """An "ok" plan, executed end-to-start. -> a new plan dict (never mutates).

    EVERY GATE THIS MODULE CERTIFIES IS SYMMETRIC IN s.  The tip-on-curve
    error, the joint margin, sigma_min and the paper/boom clearances are
    pointwise properties of a configuration; continuity (||dq||_inf between
    neighbours) is a property of an unordered pair.  None of them can tell
    which way the samples are walked, so reversing the sample order is not a
    new plan to be re-certified from scratch — it is the same certified set of
    configurations traversed the other way, and the whole operation is array
    flipping:

      qs, pts, q7, sigmas, margins, and the input polyline reverse;
      s and the PWL knots reflect to 1 - s;
      the corner windows reverse with the corners they belong to.

    THE CLOCK IS THE ONE THING THAT IS RE-DERIVED, not flipped, because it is
    not a per-sample field but the integral of one: `pacing.pace` divides the
    step by the speed the joint-velocity limits allow there, so the reversed
    path gets its times re-integrated from the reversed `qs` rather than
    inheriting t[-1] - t[::-1].  (The two agree to floating point here, since
    |dq/ds| is invariant under reversal — which is exactly why re-deriving is
    cheap insurance rather than a cost.)

    With `spec`, the reversed plan is handed to the INDEPENDENT validator
    again, so a reversed plan carries a certificate of its own and not an
    inherited one; the status goes to "bug" if it somehow does not pass.
    """
    if plan.get("status") != "ok":
        raise ValueError(f"only an 'ok' plan can be reversed; got "
                         f"{plan.get('status')!r}")
    o = dict(DEFAULTS)
    o.update(opts or {})
    out = dict(plan)
    for k in ("qs", "pts", "sigmas", "margins", "stroke", "q7"):
        if k in plan:
            out[k] = np.asarray(plan[k], float)[::-1].copy()
    out["s"] = 1.0 - np.asarray(plan["s"], float)[::-1]
    kn = np.asarray(plan["knots"], float)
    out["knots"] = np.column_stack([1.0 - kn[::-1, 0], kn[::-1, 1]])
    out["windows"] = np.asarray(plan["windows"], float)[::-1].copy()
    a, b = plan.get("clip_s", (0.0, 1.0))
    out["clip_s"] = (1.0 - float(b), 1.0 - float(a))

    qs = out["qs"]
    ds = float(plan["arc_len"]) / max(len(qs) - 1, 1)
    pc = pacing.pace(qs, plan["arc_len"], v_draw=o["v_draw"], safety=o["safety"],
                     ds_m=ds)
    out.update(times=pc["t"], total_time=float(pc["total_time"]),
               frac_slowed=float(pc["frac_slowed"]),
               headroom=float(pc["headroom"]))
    out["reversed"] = not bool(plan.get("reversed", False))
    out["notes"] = list(plan.get("notes", [])) + ["reversed (executed end-to-start)"]
    for k in ("lat", "sheet_obj", "pwl", "smooth", "pace"):
        out.pop(k, None)                     # debug objects belong to the original
    if validate and spec is not None:
        rep = validate_plan(out["pts"], spec, qs, times=out["times"],
                            h_inv=o["h_inv"], pen_ext=o["pen_ext"])
        out["validation"] = rep
        if not rep["ok"]:
            out.update(status="bug", reason="validation_failed")
    return out


# --------------------------------------------------------------------------
def plan_summary(r):
    """One terse line for logs."""
    st = r["status"]
    if st == "ok":
        v = r.get("validation", {})
        return (f"ok    L={r['arc_len']:.3f} m {r['n_knots']} knots "
                f"{r['n_dense']} samples sigma>={r['min_sigma']:.3f} "
                f"margin>={r['min_margin']:.3f} tip={r['tip_err']:.1e} m "
                f"t={r['total_time']:.1f} s valid={v.get('ok')}")
    if st == "split":
        h = r.get("head")
        return (f"split s*={r['s_star']:.3f} of {r.get('s_reach', 0.0):.3f} reach "
                f"({r['reason']}) head="
                + ("none" if h is None else f"{h['arc_len']:.3f} m certified"))
    if st == "degenerate":
        return f"degen {r['reason']}"
    return f"BUG   {r.get('reason')}: {r.get('error', '')}"
