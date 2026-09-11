#!/usr/bin/env python3
"""WHAT ONE STROKE COSTS, measured against the shipped atlas at h = 0.970.

    ARIS_RIG=proposed ARIS_TOOL=lateral \
      python3 scripts/profile_stroke_costs.py --json out/stroke_costs.json

`scripts/job_substages.py` says where a WHOLE RUN's clock went; it cannot say
what a SINGLE stroke costs, because no substage brackets one.  This does, by
calling the same functions the allocator calls, on the same inputs the v19 run
was given, one at a time:

  plan      `stroke_api.plan_stroke` — the ladder DP: lattice, DP over the
            redundancy band, dense certification, independent validator.  This
            is what `allocate.probe_stroke` and `allocate.replan_segment` call,
            and it is the only per-stroke work in the pipeline that no global
            pass can be made to skip.
  hover     `writing.hover_solve` at a planned stroke's two endpoints — the
            pose the pen-up layer starts and ends at.
  leg       `paper.route` park -> hover and hover -> hover: the pen-up leg the
            arm actually flies, certified against the steel, the neighbours'
            base columns and its own metal.  Measured COLD (`paper.clear_cache`
            first) and WARM (the memo answers), because the difference between
            those two numbers is the entire argument for a precomputed roadmap.
  rrt       `transit.plan` — the C-space tier the ladder falls through to.
  conduct   `idle`'s pairwise collision image for two arms' timelines, which
            `docs/FAST_PLANNING.md` §6 showed is ~99 % of a `coordinate` call.
  check     `scene_check` over a conducted phase, divided by its timeline
            seconds.

WHAT IS MEASURED AND WHAT IS NOT.  Every number here is one call, timed with
`time.perf_counter`, in ONE process with no pool.  The allocator runs probes
and route screens on a fork pool (`sequence.screen_pool`, 6 workers in v19), so
a substage's wall clock is NOT the sum of the calls below — the point of this
table is the per-call cost and how it scales with stroke length, which is what
a streaming architecture has to budget for.

COLD vs WARM.  `paper`'s memos (`_CACHE`, `_LIFTS`, `_LEGS`, `_SELF`, and
through `on_clear` also `writing._HOVERS` and `sequence._PRICED`) are
process-global and keyed on nothing the artwork knows: a leg between two poses
is the same leg whatever picture asked for it.  Every measurement below calls
`paper.clear_cache()` first and then calls again without clearing, so the two
columns are the same call under the two regimes.  The stroke planner has no
memo at all, which is why its two columns agree — see the note on `bench_plan`.

REPRODUCING THE TABLE IN `docs/V2_SCALING_BASELINE.md`:

    python3 scripts/profile_stroke_costs.py --only plan --max-pairs 0 --json A.json
    python3 scripts/profile_stroke_costs.py --only prefilter
    python3 scripts/profile_stroke_costs.py --only hover,leg,rrt \
        --max-legs 16 --max-rrt 3 --json B.json
    python3 scripts/profile_stroke_costs.py --only conduct,check \
        --conduct-s 10 --check-s 10 --arms 13,17,31,71 --json C.json

`--only conduct` is quadratic in `--conduct-s`: 30 s of timeline over six arms
is fifteen pairs of 8.3 M cells and takes about half an hour.  The per-cell rate
is what the table quotes and it does not depend on the window.
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# The rig and the tool are import-time choices (`aris_sixarm/fleet.py`), so they
# are set BEFORE the package is imported and not afterwards.  v19 ran
# `ARIS_RIG=proposed ARIS_TOOL=lateral`; this defaults to the same pair so the
# numbers are about the robot the 3 700 s were spent on.
os.environ.setdefault("ARIS_RIG", "proposed")
os.environ.setdefault("ARIS_TOOL", "lateral")

import numpy as np                                                  # noqa: E402

from aris_sixarm import (frames, layout, paper, pwl, stroke_api,  # noqa: E402
                         transit, writing)
from aris_sixarm.fleet import FLEET                                 # noqa: E402

STROKES = ROOT / ("out/gui_jobs/20260910-124546-ce72/"
                  "csail_schedule_h097_v19_strokes.json")
ATLAS = ROOT / "out/atlas_proposed_h0970_lat0860"
TILT_MAX_DEG = 15.0          # v19's `--tilt-max-deg`


# ----------------------------------------------------------------- timing ---
def clear():
    """Every process-global memo the pen-up layer keeps. -> None."""
    paper.clear_cache()


def timed(fn, *a, **kw):
    """-> (milliseconds, result).  One call, no repeats, no warm-up."""
    t = time.perf_counter()
    out = fn(*a, **kw)
    return (time.perf_counter() - t) * 1e3, out


def stats(xs):
    """-> {n, median, p95, mean, min, max} in the units given, or None."""
    xs = [float(x) for x in xs if x is not None]
    if not xs:
        return None
    xs_s = sorted(xs)
    k = min(len(xs_s) - 1, int(round(0.95 * (len(xs_s) - 1))))
    return dict(n=len(xs), median=statistics.median(xs_s), p95=xs_s[k],
                mean=statistics.fmean(xs_s), min=xs_s[0], max=xs_s[-1])


def row(name, s):
    if not s:
        return f"{name:<26} {'--':>9}"
    return (f"{name:<26} {s['median']:9.1f} {s['p95']:9.1f} {s['mean']:9.1f} "
            f"{s['min']:9.1f} {s['max']:9.1f} {s['n']:6d}")


HEAD = (f"{'measurement':<26} {'median':>9} {'p95':>9} {'mean':>9} "
        f"{'min':>9} {'max':>9} {'n':>6}    (ms)")


# ------------------------------------------------------------------ input ---
def load_strokes(path):
    """The v19 stroke set, already placed on the sheet. -> [(id, L, pts)]."""
    d = json.loads(Path(path).read_text())
    out = []
    for s in d["strokes"]:
        out.append((int(s["id"]), float(s["length"]),
                    np.asarray(s["pts"], float)))
    return out


def opts_for():
    """The planner options `csail_allocate.alloc_kwargs` builds for v19."""
    return dict(objective=pwl.OBJECTIVE, tilt_max_deg=TILT_MAX_DEG)


def parks():
    """{arm: q_park} for the active rig, from the certified park search."""
    return dict(getattr(layout, "Q_PARK_PROPOSED", {}) or {})


# ------------------------------------------------------- (a) the ladder DP ---
# THE PER-STROKE PLANNER DOES NOT CONSULT THE ATLAS AND DOES NOT MEMOISE.
# `stroke_api.py`, `planner.py`, `pwl.py`, `smooth.py` and `validate.py` never
# import `aris_sixarm.atlas`: the lattice is solved live by `ik.py`, and the
# atlas enters allocation only through `allocate.prefilter`, which is timed
# separately below.  There is no `lru_cache` anywhere in the package and
# nothing on this path writes a memo, so COLD AND WARM ARE THE SAME NUMBER —
# which is the finding, not an omission.  The split that matters instead is the
# one between `prepare` (resample, clip, build the lattice) and
# `plan_from_ctx` (the band DP, the dense certification, the validator).
def bench_plan(strokes, arms, opts, limit=None):
    """`plan_stroke` once per (stroke, arm), twice over. -> rows."""
    rows = []
    pairs = [(sid, L, pts, a) for (sid, L, pts) in strokes for a in arms]
    if limit:
        pairs = pairs[:limit]
    for sid, L, pts, a in pairs:
        spec = FLEET[a]
        clear()
        ms_cold, r = timed(stroke_api.plan_stroke, pts, spec, opts)
        ms_warm, _ = timed(stroke_api.plan_stroke, pts, spec, opts)
        ms_prep = ms_rest = ms_noval = None
        try:
            o = dict(stroke_api.DEFAULTS)
            o.update(opts)
            o["pen_ext"] = frames.ext_of(o.get("pen_ext"))
            ms_prep, (ctx, early) = timed(stroke_api.prepare, pts, spec, o)
            if early is None:
                ms_rest, _ = timed(stroke_api.plan_from_ctx, ctx, spec, o)
            o2 = dict(o)
            o2["validate"] = False
            ms_noval, _ = timed(stroke_api.plan_stroke, pts, spec, o2)
        except Exception:                       # a phase split is not a number
            pass
        rows.append(dict(stroke=sid, arm=a, length_m=L,
                         n_pts=int(len(pts)),
                         status=str(r.get("status")),
                         n_lattice=int(r.get("n_lattice") or 0),
                         n_dense=int(r.get("n_dense") or 0),
                         n_knots=int(r.get("n_knots") or 0),
                         ms_cold=ms_cold, ms_warm=ms_warm,
                         ms_prepare=ms_prep, ms_from_ctx=ms_rest,
                         ms_no_validate=ms_noval))
    return rows


def bench_prefilter(strokes, arms, atlas_dir, opts):
    """`allocate.prefilter` — the ONE place the atlas enters. -> (ms, n)."""
    from aris_sixarm import allocate
    st = [dict(id=sid, pts=pts, color="grey", kind="outline")
          for sid, _, pts in strokes]
    ms, pre = timed(allocate.prefilter, st, list(arms), str(atlas_dir),
                    tilt_max_deg=float(opts.get("tilt_max_deg", 0.0) or 0.0))
    ms2, _ = timed(allocate.prefilter, st, list(arms), str(atlas_dir),
                   tilt_max_deg=float(opts.get("tilt_max_deg", 0.0) or 0.0))
    return ms, ms2, len(st) * len(arms)


# ------------------------------------------------------- (b) the hover solve ---
# `lifted_or_lower` walks HOVER_LADDER = (0.06, 0.045, 0.03, 0.09, 0.12) and is
# memoised in `writing._HOVERS`; `hover_solve` is one rung of it and is memoised
# in `paper._LIFTS`.  Both memos are dropped by `paper.clear_cache()`.
def bench_hover(plans):
    """The hover at both ends of each certified plan. -> rows."""
    rows = []
    for p in plans:
        spec, qs, sid, a = p["spec"], p["qs"], p["stroke"], p["arm"]
        gate = writing.static_gate(spec, spec.pen)
        for end, (q, xy) in (("start", (qs[0], p["xy0"])),
                             ("end", (qs[-1], p["xy1"]))):
            clear()
            ms_c1, h = timed(writing.hover_solve, spec, q, xy,
                             pen_ext=spec.pen, ok=gate)
            ms_w1, _ = timed(writing.hover_solve, spec, q, xy,
                             pen_ext=spec.pen, ok=gate)
            clear()
            ms_c2, (hh, z) = timed(writing.lifted_or_lower, spec, q, xy,
                                   pen_ext=spec.pen)
            ms_w2, _ = timed(writing.lifted_or_lower, spec, q, xy,
                             pen_ext=spec.pen)
            rows.append(dict(stroke=sid, arm=a, end=end,
                             length_m=p["length_m"], ok=h is not None,
                             z=float(z), lifted=bool(z > 0),
                             ms_solve_cold=ms_c1, ms_solve_warm=ms_w1,
                             ms_ladder_cold=ms_c2, ms_ladder_warm=ms_w2))
    return rows


# ---------------------------------------------------- (c) the pen-up legs ---
# THREE REGIMES, and the spread between them is the whole roadmap argument:
#   cold      `paper.clear_cache()` first — the ladder walks VIA_HEIGHTS x the
#             shapes, and on a refusal recurses through `fold_home` and then
#             the RRT.  Seconds, not milliseconds.
#   route hit `_CACHE` answers — but `route` computes `effective_floors`,
#             `effective_static_floor` and `self_floor` BEFORE it looks, so a
#             hit is not free.
#   legs warm `cache=False` with `_LEGS`/`_LIFTS` still full: what a route costs
#             when every leg measurement in it is already certified.  This is
#             the number a persisted hover roadmap would buy.
def bench_legs(kind_pairs):
    """`paper.route` over the requested (from, to) pose pairs. -> rows."""
    rows = []
    for kind, a, q0, q1, floor, q_home, ctx in kind_pairs:
        spec = FLEET[a]
        kw = dict(pen_ext=spec.pen, tip_floor=floor, q_home=q_home)
        clear()
        ms_cold, r0 = timed(paper.route, spec, q0, q1, **kw)
        ms_hit, _ = timed(paper.route, spec, q0, q1, **kw)
        ms_legs, _ = timed(paper.route, spec, q0, q1, cache=False, **kw)
        ms_bounds, _ = timed(paper.leg_bounds, spec, q0, q1, spec.pen,
                             paper.H_INV_DEFAULT, paper.static_boxes(spec),
                             paper.SAMPLES, paper.FRAME_FLOOR)
        rows.append(dict(kind=kind, arm=a, ok=r0 is not None,
                         mode=(None if r0 is None else str(r0.get("mode"))),
                         tried=(0 if r0 is None else int(r0.get("tried") or 0)),
                         n_via=(0 if r0 is None else len(r0.get("vias") or [])),
                         ms_cold=ms_cold, ms_cache_hit=ms_hit,
                         ms_legs_warm=ms_legs, ms_leg_bounds=ms_bounds,
                         **ctx))
    return rows


# ------------------------------------------------------------- (d) the RRT ---
# The C-space tier is the LAST rung: `paper.route` reaches it only after the
# whole ladder and `fold_home` have refused (paper.py:1632).  Its budget is
# `ATTEMPTS x TIME_BUDGET + SHORTCUT_TIME`, and a refusal spends all of it.
def bench_rrt(pairs):
    """`transit.plan` on pose pairs the shape ladder refused. -> rows."""
    rows = []
    for a, q0, q1, ctx in pairs:
        spec = FLEET[a]
        clear()
        transit.reset_stats()
        ms, path = timed(transit.plan, spec, q0, q1, pen_ext=spec.pen,
                         boxes=paper.static_boxes(spec))
        st = transit.stats()
        rows.append(dict(arm=a, ok=path is not None, ms=ms,
                         n_via=(0 if not path else len(path)),
                         nodes=int(st.get("nodes", 0)),
                         edges=int(st.get("edges", 0)),
                         deadline=int(st.get("deadline", 0)), **ctx))
    return rows


def refused_pairs(hovers, q_park, limit):
    """Hover pairs the SHAPE ladder alone (rrt=False) cannot fly. -> pairs."""
    by_arm = {}
    for (sid, arm), (h0, h1) in hovers.items():
        for h in (h0, h1):
            if h is not None:
                by_arm.setdefault(arm, []).append((sid, np.asarray(h, float)))
    out = []
    for arm, hs in by_arm.items():
        spec = FLEET[arm]
        qh = q_park.get(arm)
        qh = None if qh is None else np.asarray(qh, float)
        for i in range(len(hs)):
            for j in range(len(hs)):
                if i == j or len(out) >= limit:
                    continue
                r = paper.route(spec, hs[i][1], hs[j][1], pen_ext=spec.pen,
                                q_home=qh, rrt=False)
                if r is None:
                    out.append((arm, hs[i][1], hs[j][1],
                                dict(a_stroke=hs[i][0], b_stroke=hs[j][0])))
    return out


# -------------------------------------------- (e) the pairwise conduct check ---
# There is no `can_these_two_run_concurrently(a, b)`.  The pairwise question is
# `coordination.free_cells(pi, pj, margin)`: the FREE-CELL IMAGE over the two
# arms' progress indices, which the priority DP then schedules against.  It is
# O(Ni x Nj) and it is ~99 % of a `coordinate` call (docs/FAST_PLANNING.md §6),
# so this is the number a streaming design has to budget per concurrent pair.
def bench_conduct(arms, secs, dt, cap=None):
    """One free-cell image per unordered pair of arms. -> rows."""
    from aris_sixarm import coordination
    n = int(round(secs / dt)) + 1
    paths = {}
    for a in arms:
        spec = FLEET[a]
        q0 = np.asarray(spec.q_seed, float)
        q = np.linspace(q0, q0 + 0.30, n)
        paths[a] = coordination.ArmPath(a, q, dt, spec=spec,
                                        pen_ext=spec.pen)
    rows = []
    for i, ai in enumerate(arms):
        for aj in arms[i + 1:]:
            coordination.clear_images()
            kw = {} if cap is None else dict(cap=float(cap))
            ms, F = timed(coordination.free_cells, paths[ai], paths[aj],
                          coordination.PAIR_MARGIN, **kw)
            ms_warm, _ = timed(coordination.free_cells, paths[ai], paths[aj],
                               coordination.PAIR_MARGIN, **kw)
            rows.append(dict(a=ai, b=aj, secs=secs, dt=dt, n=n,
                             cells=int(F.size), ms=ms, ms_repeat=ms_warm,
                             us_per_cell=1e3 * ms / max(1, F.size),
                             cap=("default" if cap is None else float(cap))))
    return rows


# ------------------------------------------- (f) scene_check per second ---
def bench_check(arms, secs, dt, sub=2):
    """`scene_check.check_timeline` over a synthetic fleet timeline. -> row."""
    from aris_sixarm import scene_check
    n = int(round(secs / dt)) + 1
    qt, pen = {}, {}
    for a in arms:
        spec = FLEET[a]
        q0 = np.asarray(spec.q_seed, float)
        qt[a] = np.linspace(q0, q0 + 0.20, n)
        pen[a] = float(spec.pen)
    ms, rep = timed(scene_check.check_timeline, qt, dt, 0.050,
                    pen_ext=pen, sub=sub, verbose=False, fleet=FLEET)
    return dict(arms=list(arms), n_pairs=len(arms) * (len(arms) - 1) // 2,
                secs=secs, dt=dt, sub=sub, n_steps=n,
                n_fine=int(rep.get("n_fine") or 0), ms=ms,
                s_per_s=1e-3 * ms / secs, ok=bool(rep.get("ok")),
                frame_refine=rep.get("frame_refine"))


# ------------------------------------------------------------------- main ---
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--strokes", default=str(STROKES))
    ap.add_argument("--atlas", default=str(ATLAS))
    ap.add_argument("--arms", default="all")
    ap.add_argument("--max-pairs", type=int, default=60,
                    help="(stroke, arm) pairs to plan; 0 = every one")
    ap.add_argument("--max-legs", type=int, default=40)
    ap.add_argument("--max-rrt", type=int, default=6)
    ap.add_argument("--only", default="",
                    help="comma list of prefilter,plan,hover,leg,rrt,"
                         "conduct,check")
    ap.add_argument("--dt", type=float, default=1.0 / 48.0,
                    help="the conducting clock; v19 ran 24 fps x 2 substeps")
    ap.add_argument("--conduct-s", type=float, default=30.0)
    ap.add_argument("--check-s", type=float, default=10.0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    want = set(x for x in a.only.split(",") if x) or {
        "plan", "hover", "leg", "rrt"}
    arms = (sorted(FLEET) if a.arms == "all"
            else [int(x) for x in a.arms.split(",")])
    strokes = load_strokes(a.strokes)
    opts = opts_for()
    q_park = parks()
    out = dict(rig=os.environ["ARIS_RIG"], tool=os.environ["ARIS_TOOL"],
               atlas=str(a.atlas), strokes=str(a.strokes),
               n_strokes=len(strokes), arms=arms,
               pen_ext_m={int(k): float(FLEET[k].pen) for k in arms},
               tilt_max_deg=TILT_MAX_DEG)

    print(f"rig {out['rig']}  tool {out['tool']}  arms {arms}")
    print(f"{len(strokes)} strokes, "
          f"{sum(L for _, L, _ in strokes):.3f} m of ink")
    print()

    plans = []
    if "plan" in want:
        rows = bench_plan(strokes, arms, opts,
                          limit=(a.max_pairs or None))
        out["plan"] = rows
        ok = [r for r in rows if r["status"] == "ok"]
        print(HEAD)
        print(row("plan_stroke cold", stats([r["ms_cold"] for r in rows])))
        print(row("plan_stroke warm", stats([r["ms_warm"] for r in rows])))
        print(row("  ...certified only", stats([r["ms_cold"] for r in ok])))
        print(f"{len(ok)}/{len(rows)} (stroke, arm) pairs certified whole; "
              f"the rest SPLIT — the arm cannot hold the whole span, "
              f"which is what `probe_stroke` walks the gaps for")
        for st in ("ok", "split", "degenerate", "bug"):
            sel = [r for r in rows if r["status"] == st]
            if sel:
                print(row(f"  status {st}",
                          stats([r["ms_cold"] for r in sel])))
        print(row("  prepare (lattice)",
                  stats([r["ms_prepare"] for r in ok])))
        print(row("  DP+certify+validate",
                  stats([r["ms_from_ctx"] for r in ok])))
        print(row("  whole call, validate off",
                  stats([r["ms_no_validate"] for r in ok])))
        print()

    # The poses every later measurement needs come from plans that certified.
    for sid, L, pts in strokes:
        for arm in arms:
            if len(plans) >= 12:
                break
            spec = FLEET[arm]
            r = stroke_api.plan_stroke(pts, spec, opts)
            if r.get("status") != "ok":
                continue
            qs = np.asarray(r["qs"], float)
            P = np.asarray(r["pts"], float)
            plans.append(dict(stroke=sid, arm=arm, spec=spec, qs=qs,
                              length_m=L, xy0=P[0, :2], xy1=P[-1, :2]))

    hovers, zs = {}, {}
    if plans and ("hover" in want or "leg" in want or "rrt" in want):
        for p in plans:
            h0, z0 = writing.lifted_or_lower(p["spec"], p["qs"][0], p["xy0"],
                                             pen_ext=p["spec"].pen)
            h1, z1 = writing.lifted_or_lower(p["spec"], p["qs"][-1], p["xy1"],
                                             pen_ext=p["spec"].pen)
            hovers[(p["stroke"], p["arm"])] = (h0, h1)
            zs[(p["stroke"], p["arm"])] = (float(z0), float(z1))

    if "hover" in want:
        rows = bench_hover(plans)
        out["hover"] = rows
        print(HEAD)
        print(row("hover_solve cold",
                  stats([r["ms_solve_cold"] for r in rows])))
        print(row("hover_solve memo",
                  stats([r["ms_solve_warm"] for r in rows])))
        print(row("lifted_or_lower cold",
                  stats([r["ms_ladder_cold"] for r in rows])))
        print(row("lifted_or_lower memo",
                  stats([r["ms_ladder_warm"] for r in rows])))
        print(f"HOVER_LADDER {writing.HOVER_LADDER}; "
              f"{sum(1 for r in rows if r['lifted'])}/{len(rows)} endpoints "
              f"got a hover above the paper")
        print()

    if "leg" in want:
        pairs = []
        for p in plans:
            key = (p["stroke"], p["arm"])
            h0 = hovers.get(key, (None, None))[0]
            z0 = zs.get(key, (0.0, 0.0))[0]
            qp = q_park.get(p["arm"])
            if h0 is None or qp is None:
                continue
            qp = np.asarray(qp, float).reshape(7)
            pairs.append(("park->hover", p["arm"], qp,
                          np.asarray(h0, float),
                          paper.travel_floor(writing.LIFT_Z, z0), qp,
                          dict(stroke=p["stroke"], length_m=p["length_m"])))
        by_arm = {}
        for p in plans:
            by_arm.setdefault(p["arm"], []).append(p)
        for arm, ps in by_arm.items():
            qp = q_park.get(arm)
            qp = None if qp is None else np.asarray(qp, float).reshape(7)
            for i in range(len(ps) - 1):
                ka, kb = (ps[i]["stroke"], arm), (ps[i + 1]["stroke"], arm)
                ha = hovers.get(ka, (None, None))[1]
                hb = hovers.get(kb, (None, None))[0]
                if ha is None or hb is None:
                    continue
                pairs.append(("hover->hover", arm, np.asarray(ha, float),
                              np.asarray(hb, float),
                              paper.travel_floor(zs[ka][1], zs[kb][0]), qp,
                              dict(stroke=ps[i]["stroke"],
                                   length_m=ps[i]["length_m"])))
        pairs = pairs[:a.max_legs] if a.max_legs else pairs
        rows = bench_legs(pairs)
        out["leg"] = rows
        print(HEAD)
        for kind in ("park->hover", "hover->hover"):
            sel = [r for r in rows if r["kind"] == kind]
            if not sel:
                continue
            print(row(f"route {kind} COLD",
                      stats([r["ms_cold"] for r in sel])))
            print(row(f"route {kind} cache hit",
                      stats([r["ms_cache_hit"] for r in sel])))
            print(row(f"route {kind} legs warm",
                      stats([r["ms_legs_warm"] for r in sel])))
            print(row(f"leg_bounds {kind}",
                      stats([r["ms_leg_bounds"] for r in sel])))
        print(f"{sum(1 for r in rows if r['ok'])}/{len(rows)} legs certified; "
              f"VIA_HEIGHTS {paper.VIA_HEIGHTS}, SAMPLES {paper.SAMPLES}")
        print()

    if "rrt" in want:
        pairs = refused_pairs(hovers, q_park, a.max_rrt)
        out["rrt_refused_pairs"] = len(pairs)
        rows = bench_rrt(pairs) if pairs else []
        out["rrt"] = rows
        print(HEAD)
        if rows:
            print(row("transit.plan (RRT)", stats([r["ms"] for r in rows])))
            print(f"{sum(1 for r in rows if r['ok'])}/{len(rows)} solved, "
                  f"{sum(r['deadline'] for r in rows)} hit the clock")
        else:
            print("the shape ladder refused none of the sampled hover pairs")
        print(f"budget: {transit.TIME_BUDGET} s x {transit.ATTEMPTS} attempts "
              f"+ {transit.SHORTCUT_TIME} s shortcut = "
              f"{transit.ATTEMPTS * transit.TIME_BUDGET + transit.SHORTCUT_TIME:.1f} s "
              f"worst case, {transit.MAX_NODES} nodes/tree")
        print()

    if "prefilter" in want:
        ms, ms2, npairs = bench_prefilter(strokes, arms, a.atlas, opts)
        out["prefilter"] = dict(ms_cold=ms, ms_warm=ms2, n_pairs=npairs,
                                ms_per_pair=ms / max(1, npairs))
        print(f"prefilter (the ONE atlas read): {ms:.1f} ms cold, "
              f"{ms2:.1f} ms again, over {npairs} (stroke, arm) pairs "
              f"= {ms / max(1, npairs):.3f} ms/pair")
        print()

    if "conduct" in want:
        rows = bench_conduct(arms, a.conduct_s, a.dt)
        rows += bench_conduct(arms, a.conduct_s, a.dt, cap=2.0)
        out["conduct"] = rows
        print(HEAD)
        for tag, sel in (("free_cells (broad hit)",
                          [r for r in rows if r["cap"] == "default"]),
                         ("free_cells (all pairs live)",
                          [r for r in rows if r["cap"] != "default"])):
            print(row(tag, stats([r["ms"] for r in sel])))
        if rows:
            print(f"{a.conduct_s:.0f} s of timeline at dt = {a.dt:.5f} s "
                  f"-> {rows[0]['n']} samples, {rows[0]['cells']} cells/pair, "
                  f"{len(arms) * (len(arms) - 1) // 2} unordered pairs")
        print()

    if "check" in want:
        r = bench_check(arms, a.check_s, a.dt)
        out["check"] = r
        print(f"scene_check: {r['ms'] / 1e3:.2f} s for {r['secs']:.0f} s of "
              f"timeline, {r['n_pairs']} pairs, sub={r['sub']} "
              f"-> {r['s_per_s']:.3f} s of check per second of timeline "
              f"({r['n_fine']} fine samples)")
        print()

    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=float))
        print("wrote", a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
