#!/usr/bin/env python3
"""Seeded parallel fuzz campaign against `stroke_api.plan_stroke`.

WHAT IT IS FOR.  The installation coordinates six arms over hundreds of
strokes, so a 1-in-200 per-stroke crash is a crash per picture.  The contract
`plan_stroke` promises — certified, or split, never garbage, never an
exception — is only worth what it survives, so this throws random geometry at
all six arms and checks the contract from the outside:

  * every "ok" is re-validated with `validate.validate_plan` (which shares
    nothing with the planner but `frames`), and a failure there is a BUG, not
    a rejection;
  * every "split" must certify its own head: re-planning [0, s* - eps] from
    scratch has to come back "ok" (or "degenerate" if nothing is left), and a
    random fifth of the tails ((s_reach + eps, 1]) are re-planned too, where
    only "no exception" is required — a tail may legitimately split again;
  * a tenth of all strokes are planned TWICE and compared bitwise, because a
    planner that is only usually deterministic cannot be regression-tested;
  * "degenerate" is a valid answer, "bug" never is.

Strokes are biased so that most of them are honest work (inside the arm's
plausible annulus), a quarter deliberately straddle a dead zone or the sheet
edge (those MUST split cleanly), and the rest are hostile: off-sheet, tiny,
zero-length, all-duplicate, closed loops, NaN-laced.

Also runs an ATLAS CONSISTENCY check: strokes drawn entirely inside strict-GO,
tilt-free atlas cells should plan at >= 95 %.  It cross-checks two independent
pieces of the stack — a miss is a finding about the atlas, the planner, or the
gap between "this cell has a solution" and "these cells have a common
corridor".

Outputs out/fuzz_report.json and out/fuzz_summary.png.  System python3 only.

    python3 scripts/fuzz_planner.py --n 150 --workers 8
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from aris_sixarm import atlas, letters, planner  # noqa: E402
from aris_sixarm.fleet import FLEET, SHEET  # noqa: E402
from aris_sixarm.stroke_api import (plan_stroke, polyline_length,  # noqa: E402
                                    truncate_polyline)
from aris_sixarm.validate import validate_plan, violation_summary  # noqa: E402

GENS = ("line", "arc", "bezier", "spiral", "corners", "letter")
CLASSES = ("inside", "straddle", "wild")
CLASS_P = (0.60, 0.25, 0.15)
LEN_RANGE = (0.15, 0.90)          # m — keeps the lattices fast
ANNULUS = {"floor": (0.32, 0.78), "inv": (0.32, 0.68)}
DEAD_R = {"floor": 0.20, "inv": 0.24}
BORDER = 0.05                      # m, sheet margin for "inside" placement
DET_FRAC = 0.10                    # strokes planned twice
TAIL_FRAC = 0.20                   # splits whose tail is also re-planned
OPTS = dict(keep_debug=False)


# --------------------------------------------------------------------------
# stroke generators — unit shapes, then placed
# --------------------------------------------------------------------------
def _unit_shape(rng, gen):
    """A shape spanning roughly [-0.5, 0.5]^2, to be scaled and placed."""
    if gen == "line":
        return np.array([[-0.5, 0.0], [0.5, 0.0]])
    if gen == "arc":
        sweep = rng.uniform(0.5, 4.5)
        a0 = rng.uniform(0, 2 * np.pi)
        t = np.linspace(a0, a0 + sweep * rng.choice([-1, 1]), 60)
        return np.column_stack([0.5 * np.cos(t), 0.5 * np.sin(t)])
    if gen == "bezier":
        P = np.array([[-0.5, 0.0], rng.uniform(-0.7, 0.7, 2),
                      rng.uniform(-0.7, 0.7, 2), [0.5, 0.0]])
        t = np.linspace(0, 1, 60)[:, None]
        return ((1 - t) ** 3 * P[0] + 3 * (1 - t) ** 2 * t * P[1]
                + 3 * (1 - t) * t ** 2 * P[2] + t ** 3 * P[3])
    if gen == "spiral":
        turns = rng.uniform(1.0, 2.5)
        th = np.linspace(0, turns * 2 * np.pi, 140)
        r = 0.08 + 0.42 * th / th[-1]
        return np.column_stack([r * np.cos(th), r * np.sin(th)])
    if gen == "corners":
        k = int(rng.integers(3, 6))
        return rng.uniform(-0.5, 0.5, (k, 2))
    if gen == "letter":
        name = str(rng.choice(list(letters.UNIT)))
        strokes = letters.UNIT[name]
        return np.asarray(strokes[int(rng.integers(len(strokes)))], float)
    raise ValueError(gen)


def _place(shape, rng, target_len, anchor, rot=None):
    """Scale a unit shape to `target_len` metres, rotate, centre on `anchor`."""
    p = np.asarray(shape, float)
    L = polyline_length(p)
    if L <= 0:
        return p + np.asarray(anchor, float)
    p = p * (target_len / L)
    a = rng.uniform(0, 2 * np.pi) if rot is None else rot
    c, s = np.cos(a), np.sin(a)
    p = p @ np.array([[c, s], [-s, c]])
    return p - p.mean(axis=0) + np.asarray(anchor, float)


def _on_sheet(p, border=BORDER):
    return bool(np.all((p[:, 0] > border) & (p[:, 0] < SHEET[0] - border)
                       & (p[:, 1] > border) & (p[:, 1] < SHEET[1] - border)))


def make_stroke(rng, spec, gen, cls):
    """-> (polyline (N,2), meta dict).  Deterministic in `rng`."""
    bx, by = spec.xy
    base = np.array([bx, by])
    rlo, rhi = ANNULUS[spec.mount]
    meta = dict(gen=gen, cls=cls)

    if cls == "inside":                      # bias: entirely in the good annulus
        shape = _unit_shape(rng, gen)
        for _ in range(14):
            Lt = rng.uniform(*LEN_RANGE)
            th = rng.uniform(0, 2 * np.pi)
            rr = rng.uniform(rlo + 0.12, rhi - 0.12)
            p = _place(shape, rng, Lt, base + rr * np.array([np.cos(th), np.sin(th)]))
            r = np.linalg.norm(p - base, axis=1)
            if _on_sheet(p) and r.min() > rlo and r.max() < rhi:
                meta.update(target_len=Lt, r=[float(r.min()), float(r.max())])
                return p, meta
            shape = _unit_shape(rng, gen)
        Lt = 0.15                            # fallback: a short radial line
        th = rng.uniform(0, 2 * np.pi)
        u = np.array([np.cos(th), np.sin(th)])
        p = base + np.outer([0.0, Lt], u) + (rlo + 0.15) * u
        meta.update(target_len=Lt, fallback=True)
        return p, meta

    if cls == "straddle":                    # must split cleanly
        kind = str(rng.choice(["through_base", "outward", "sheet_edge"]))
        meta["kind"] = kind
        th = rng.uniform(0, 2 * np.pi)
        u = np.array([np.cos(th), np.sin(th)])
        if kind == "through_base":           # straight through the dead zone
            d = rng.uniform(0.35, 0.65)
            p = np.array([base - d * u, base + d * u])
        elif kind == "outward":              # past the reach limit
            p = np.array([base + (rhi - 0.25) * u, base + (rhi + 0.45) * u])
        else:                                # across the paper's edge
            edge = np.array([rng.uniform(0, SHEET[0]), rng.choice([0.0, SHEET[1]])])
            if rng.random() < 0.5:
                edge = np.array([rng.choice([0.0, SHEET[0]]), rng.uniform(0, SHEET[1])])
            v = edge - base
            v = v / max(np.linalg.norm(v), 1e-9)
            p = np.array([base + (rlo + 0.1) * v, edge + 0.35 * v])
        if gen != "line":                    # bend it, keeping the same span
            shape = _unit_shape(rng, gen)
            span = float(np.linalg.norm(p[-1] - p[0]))
            ang = float(np.arctan2(p[-1, 1] - p[0, 1], p[-1, 0] - p[0, 0]))
            p = _place(shape, rng, max(span, 0.15), 0.5 * (p[0] + p[-1]), rot=ang)
        return p, meta

    kind = str(rng.choice(["off_sheet", "tiny", "zero_len", "duplicates",
                           "closed_loop", "single_point", "empty", "nan",
                           "huge", "spike", "gigantic"]))
    meta["kind"] = kind
    if kind == "off_sheet":
        c = np.array([rng.uniform(-2.0, SHEET[0] + 2.0), rng.uniform(-2.0, SHEET[1] + 2.0)])
        return _place(_unit_shape(rng, gen), rng, rng.uniform(0.2, 0.9), c), meta
    if kind == "tiny":
        c = base + rng.uniform(-0.5, 0.5, 2)
        return _place(_unit_shape(rng, gen), rng, 10.0 ** rng.uniform(-5, -2), c), meta
    if kind == "zero_len":
        c = base + rng.uniform(-0.6, 0.6, 2)
        return np.array([c, c]), meta
    if kind == "duplicates":
        c = base + rng.uniform(-0.6, 0.6, 2)
        p = np.repeat(np.array([c, c + [0.2, 0.0]]), 4, axis=0)   # 8 pts, 2 distinct
        return p, meta
    if kind == "closed_loop":
        c = base + rng.uniform(-0.4, 0.4, 2)
        t = np.linspace(0, 2 * np.pi, 80)
        rr = rng.uniform(0.05, 0.35)
        p = np.column_stack([c[0] + rr * np.cos(t), c[1] + rr * np.sin(t)])
        p[-1] = p[0]                                              # exactly closed
        return p, meta
    if kind == "single_point":
        return (base + rng.uniform(-0.6, 0.6, 2))[None, :], meta
    if kind == "empty":
        return np.zeros((0, 2)), meta
    if kind == "nan":
        p = _place(_unit_shape(rng, gen), rng, 0.4, base + rng.uniform(-0.4, 0.4, 2))
        p[int(rng.integers(len(p)))] = [np.nan, np.inf]
        return p, meta
    if kind == "huge":                       # several times the sheet
        c = base + rng.uniform(-0.3, 0.3, 2)
        return _place(_unit_shape(rng, gen), rng, rng.uniform(3.0, 8.0), c), meta
    if kind == "gigantic":                   # kilometres: must not be resampled
        c = base + rng.uniform(-0.3, 0.3, 2)
        return _place(_unit_shape(rng, gen), rng, 10.0 ** rng.uniform(3, 6), c), meta
    c = base + rng.uniform(-0.5, 0.5, 2)     # "spike": a doubled-back hairpin
    d = rng.uniform(0.05, 0.4)
    u = np.array([np.cos(rng.uniform(0, 2 * np.pi)), np.sin(rng.uniform(0, 2 * np.pi))])
    return np.array([c, c + d * u, c + 1e-6 * u, c + d * u]), meta


# --------------------------------------------------------------------------
# one fuzz task
# --------------------------------------------------------------------------
def _fail(rec, kind, **kw):
    rec.setdefault("failures", []).append(dict(kind=kind, **kw))


def run_task(task):
    """Plan one fuzz stroke and check the contract around it.  Never raises."""
    rec = dict(task)
    t0 = time.time()
    try:
        spec = FLEET[task["arm"]]
        rng = np.random.default_rng(task["seed"])
        poly, meta = make_stroke(rng, spec, task["gen"], task["cls"])
        rec["meta"] = meta
        rec["n_pts"] = int(len(poly))
        rec["len_in"] = polyline_length(poly) if len(poly) > 1 else 0.0

        r = plan_stroke(poly, spec, OPTS)
        st = rec["status"] = r["status"]
        rec["reason"] = r.get("reason", "")

        if st == "bug":
            _fail(rec, "bug", reason=r.get("reason"), error=r.get("error", ""),
                  tb=r.get("traceback", ""),
                  violations=violation_summary(r["validation"])
                  if r.get("validation") else [])
        elif st == "ok":
            rec.update(min_sigma=r["min_sigma"], min_margin=r["min_margin"],
                       tip_err=r["tip_err"], n_knots=r["n_knots"],
                       n_dense=r["n_dense"], arc_len=r["arc_len"],
                       total_time=r["total_time"])
            v = validate_plan(r["pts"], spec, r["qs"], times=r["times"])
            rec["valid_ok"] = bool(v["ok"])
            if not v["ok"]:
                _fail(rec, "validator_rejected_ok_plan",
                      violations=violation_summary(v), worst=v["worst"])
        elif st == "split":
            rec.update(s_star=r["s_star"], s_reach=r.get("s_reach", 0.0),
                       arc_len=r["arc_len"],
                       head=None if r.get("head") is None else r["head"]["arc_len"])
            stroke = np.asarray(r["stroke"], float)
            L = max(r["arc_len"], 1e-9)
            xy = truncate_polyline(stroke, 0.0, r["s_star"])
            rec["split_xy"] = [float(xy[-1, 0]), float(xy[-1, 1])]
            eps = min(0.5, 0.01 / L)
            # (a) the head must certify on its own
            s_head = max(r["s_star"] - eps, 0.0)
            sub = truncate_polyline(stroke, 0.0, s_head)
            hr = plan_stroke(sub, spec, OPTS) if len(sub) > 1 \
                else dict(status="degenerate", reason="empty_head")
            rec["head_status"] = hr["status"]
            if hr["status"] not in ("ok", "degenerate"):
                _fail(rec, "head_not_certifiable", head_status=hr["status"],
                      head_reason=hr.get("reason", ""), s_star=r["s_star"],
                      poly=np.round(sub, 6).tolist())
            elif hr["status"] == "ok":
                v = validate_plan(hr["pts"], spec, hr["qs"], times=hr["times"])
                if not v["ok"]:
                    _fail(rec, "head_validator_rejected",
                          violations=violation_summary(v))
            # (b) the tail must merely not crash
            if task["tail_check"]:
                s_t = min(r.get("s_reach", r["s_star"]) + eps, 1.0)
                tail = truncate_polyline(stroke, s_t, 1.0)
                tr = plan_stroke(tail, spec, OPTS) if len(tail) > 1 \
                    else dict(status="degenerate")
                rec["tail_status"] = tr["status"]
                if tr["status"] == "bug":
                    _fail(rec, "tail_bug", error=tr.get("error", ""),
                          tb=tr.get("traceback", ""))

        # (c) determinism
        if task["det_check"]:
            r2 = plan_stroke(poly, spec, OPTS)
            same = r2["status"] == st
            if same and st in ("ok", "split"):
                a, b = (r, r2) if st == "ok" else (r.get("head"), r2.get("head"))
                if a is None or b is None:
                    same = a is b
                else:
                    same = (np.array_equal(a["knots"], b["knots"])
                            and np.array_equal(a["qs"], b["qs"])
                            and np.array_equal(a["times"], b["times"]))
            rec["det_ok"] = bool(same)
            if not same:
                _fail(rec, "nondeterministic", status2=r2["status"])
    except Exception as exc:                      # the harness itself must not die
        rec["status"] = "harness_error"
        _fail(rec, "harness_error", error=f"{type(exc).__name__}: {exc}",
              tb=traceback.format_exc())
    rec["dt"] = time.time() - t0
    return rec


# --------------------------------------------------------------------------
# atlas consistency
# --------------------------------------------------------------------------
def _go_mask(arm_id, out_dir, erode=True):
    """(cell index set, grid, cell centres) of strict-GO tilt-free atlas cells.

    ERODED BY ONE CELL by default, and that is not a detail.  The atlas samples
    the sheet every 2 cm; a stroke sample only ever *rounds* to a cell, so on
    the raw mask it may sit up to 1.4 cm outside the GO region it was
    attributed to.  At the outer reach boundary feasibility falls off inside a
    centimetre — the first campaign's only atlas miss was exactly this: an
    endpoint 8 mm past the boundary, with 0 gated IK solutions, next to a cell
    the atlas (correctly) calls strict-GO.  Requiring the cell AND its eight
    neighbours to be GO is what "the stroke lies inside strict-GO" means when
    the map has a 2 cm pixel.
    """
    arr, d = atlas.load(out_dir, arm_id)
    grid = float(d["grid"])
    sel = atlas.strict_go(arr) & (arr[:, 8] == 0.0)
    idx = {(int(round(x / grid)), int(round(y / grid))) for x, y in arr[sel][:, :2]}
    if not erode:
        return idx, grid, arr[sel][:, :2]
    keep = {c for c in idx
            if all((c[0] + a, c[1] + b) in idx
                   for a in (-1, 0, 1) for b in (-1, 0, 1))}
    cells = np.array([[a * grid, b * grid] for a, b in sorted(keep)]) \
        if keep else np.zeros((0, 2))
    return keep, grid, cells


def _all_in_go(p, idx, grid):
    ii = np.rint(np.asarray(p, float) / grid).astype(int)
    return all((int(a), int(b)) in idx for a, b in ii)


def make_go_stroke(rng, cells, idx, grid, tries=40):
    """A stroke whose dense samples all sit on strict-GO tilt-free cells."""
    for _ in range(tries):
        c = cells[int(rng.integers(len(cells)))]
        th = rng.uniform(0, 2 * np.pi)
        Lt = rng.uniform(0.15, 0.50)
        u = np.array([np.cos(th), np.sin(th)])
        if rng.random() < 0.4:                      # a gentle arc, not a line
            R = rng.uniform(0.25, 1.2) * rng.choice([-1, 1])
            a = np.linspace(0, Lt / abs(R), 40)
            p = c + np.column_stack([R * np.sin(a), R * (1 - np.cos(a))]) @ \
                np.array([[u[0], u[1]], [-u[1], u[0]]])
        else:
            p = c + np.outer(np.linspace(0, Lt, 24), u)
        dense, _ = planner.resample(p, 0.01)
        if _all_in_go(dense, idx, grid):
            return p
    return None


def atlas_task(task):
    """Plan one atlas-GO stroke; report the miss in full if it is one."""
    rec = dict(task)
    try:
        spec = FLEET[task["arm"]]
        idx, grid, cells = _go_mask(task["arm"], task["out_dir"])
        rng = np.random.default_rng(task["seed"])
        p = make_go_stroke(rng, cells, idx, grid)
        if p is None:
            rec.update(status="no_stroke")
            return rec
        r = plan_stroke(p, spec, OPTS)
        rec.update(status=r["status"], reason=r.get("reason", ""),
                   xy0=[float(p[0, 0]), float(p[0, 1])],
                   xy1=[float(p[-1, 0]), float(p[-1, 1])],
                   arc_len=polyline_length(p))
        if r["status"] != "ok":
            rec["s_star"] = r.get("s_star")
            rec["s_reach"] = r.get("s_reach")
            rec["poly"] = np.round(p, 6).tolist()
            # where it stops, and what the atlas says about that spot
            dense, _ = planner.resample(p, 0.01)
            k = min(int(r.get("s_reach", 0.0) * (len(dense) - 1)), len(dense) - 1)
            rec["stop_xy"] = [float(dense[k, 0]), float(dense[k, 1])]
            rec["stop_r"] = float(np.hypot(dense[k, 0] - spec.xy[0],
                                           dense[k, 1] - spec.xy[1]))
    except Exception as exc:
        rec.update(status="harness_error", reason=f"{type(exc).__name__}: {exc}")
    return rec


# --------------------------------------------------------------------------
# campaign
# --------------------------------------------------------------------------
def build_tasks(n_per_arm, seed, arms):
    rng = np.random.default_rng(seed)
    tasks = []
    for arm in arms:
        for i in range(n_per_arm):
            tasks.append(dict(arm=int(arm), gen=str(rng.choice(GENS)),
                              cls=str(rng.choice(CLASSES, p=CLASS_P)), idx=i,
                              seed=int(rng.integers(1, 2 ** 31)),
                              det_check=bool(rng.random() < DET_FRAC),
                              tail_check=bool(rng.random() < TAIL_FRAC)))
    return tasks


def summarise(recs, atlas_recs, cfg, elapsed):
    tot = Counter(r["status"] for r in recs)
    per_arm = defaultdict(Counter)
    per_gen = defaultdict(Counter)
    per_cls = defaultdict(Counter)
    per_arm_gen = defaultdict(Counter)
    for r in recs:
        per_arm[r["arm"]][r["status"]] += 1
        per_gen[r["gen"]][r["status"]] += 1
        per_cls[r["cls"]][r["status"]] += 1
        per_arm_gen[f"{r['arm']}/{r['gen']}"][r["status"]] += 1
    fails = [dict(arm=r["arm"], gen=r["gen"], cls=r["cls"], seed=r["seed"],
                  status=r["status"], meta=r.get("meta"), **f)
             for r in recs for f in r.get("failures", [])]
    det = [r for r in recs if "det_ok" in r]
    heads = Counter(r["head_status"] for r in recs if "head_status" in r)
    tails = Counter(r["tail_status"] for r in recs if "tail_status" in r)
    splits = [r for r in recs if r["status"] == "split"]
    reasons = defaultdict(Counter)
    for r in splits:
        reasons[r["cls"]][r["reason"]] += 1
    sig = np.array([r["min_sigma"] for r in recs if r["status"] == "ok"])
    ss = np.array([r["s_star"] for r in splits]) if splits else np.zeros(0)
    degen = Counter(r["reason"] for r in recs if r["status"] == "degenerate")
    a_tot = Counter(r["status"] for r in atlas_recs)
    a_rate = a_tot["ok"] / max(sum(a_tot.values()), 1)
    return dict(
        config=cfg, elapsed_s=round(elapsed, 1), n=len(recs), totals=dict(tot),
        per_arm={str(k): dict(v) for k, v in sorted(per_arm.items())},
        per_generator={k: dict(v) for k, v in sorted(per_gen.items())},
        per_class={k: dict(v) for k, v in sorted(per_cls.items())},
        per_arm_generator={k: dict(v) for k, v in sorted(per_arm_gen.items())},
        split_head_status=dict(heads), split_tail_status=dict(tails),
        split_reasons={k: dict(v) for k, v in sorted(reasons.items())},
        degenerate_reasons=dict(degen),
        split_s_star=dict(
            n=len(ss), zero=int((ss <= 1e-9).sum()),
            quantiles={q: (round(float(np.quantile(ss, q)), 3) if len(ss) else None)
                       for q in (0.25, 0.5, 0.75, 0.95)}),
        ok_min_sigma=dict(
            n=len(sig), worst=round(float(sig.min()), 4) if len(sig) else None,
            quantiles={q: (round(float(np.quantile(sig, q)), 4) if len(sig) else None)
                       for q in (0.05, 0.25, 0.5, 0.95)}),
        splits=[dict(arm=r["arm"], gen=r["gen"], cls=r["cls"], seed=r["seed"],
                     reason=r["reason"], s_star=round(r["s_star"], 4),
                     s_reach=round(r.get("s_reach", 0.0), 4),
                     xy=[round(v, 4) for v in r.get("split_xy", [])])
                for r in splits],
        determinism=dict(checked=len(det),
                         mismatches=sum(1 for r in det if not r["det_ok"])),
        validator=dict(checked=sum(1 for r in recs if "valid_ok" in r),
                       rejected=sum(1 for r in recs
                                    if r.get("valid_ok") is False)),
        atlas=dict(rate=round(a_rate, 4), totals=dict(a_tot),
                   per_arm={str(a): dict(Counter(r["status"] for r in atlas_recs
                                                 if r["arm"] == a))
                            for a in sorted({r["arm"] for r in atlas_recs})},
                   misses=[r for r in atlas_recs if r["status"] != "ok"]),
        n_failures=len(fails), failures=fails,
        timing=dict(total_plan_s=round(sum(r.get("dt", 0.0) for r in recs), 1),
                    slowest=sorted((round(r.get("dt", 0.0), 2), r["arm"], r["gen"],
                                    r["cls"], r["seed"]) for r in recs)[-5:]))


def figure(recs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.0))
    order = ["ok", "split", "degenerate", "bug"]
    colors = {"ok": "#2ca02c", "split": "#ff9d00", "degenerate": "#7f7f7f",
              "bug": "#d62728"}
    arms = sorted({r["arm"] for r in recs})
    ax = axes[0]
    bottom = np.zeros(len(arms))
    for st in order:
        v = np.array([sum(1 for r in recs if r["arm"] == a and r["status"] == st)
                      for a in arms], float)
        ax.bar([str(a) for a in arms], v, bottom=bottom, color=colors[st],
               label=st, edgecolor="white", linewidth=0.6)
        bottom += v
    ax.set_xlabel("arm")
    ax.set_ylabel("strokes")
    ax.set_title(f"outcomes per arm ({len(recs)} strokes)")
    ax.legend(fontsize=8)

    ax = axes[1]
    sig = np.array([r["min_sigma"] for r in recs if r["status"] == "ok"])
    if len(sig):
        ax.hist(sig, bins=40, color="#1f77b4", edgecolor="white", linewidth=0.5)
        ax.axvline(0.10, color="#d62728", ls="--", lw=1.2, label="band gate 0.10")
        ax.axvline(0.14, color="0.35", ls="-.", lw=1.0, label="strict gate 0.14")
        ax.set_title(rf"min $\sigma_{{min}}$ of certified plans "
                     rf"(worst {sig.min():.3f}, median {np.median(sig):.3f})")
        ax.legend(fontsize=8)
    ax.set_xlabel(r"min $\sigma_{min}$ along the stroke")
    ax.set_ylabel("plans")

    ax = axes[2]
    ax.add_patch(plt.Rectangle((0, 0), *SHEET, fc="0.96", ec="0.5", lw=1.0))
    for aid, spec in FLEET.items():
        ax.plot(*spec.xy, "s", ms=7, color=spec.color, mec="black", mew=0.6)
        ax.annotate(str(aid), spec.xy, textcoords="offset points", xytext=(6, 4),
                    fontsize=8)
    for r in recs:
        if r["status"] == "split" and r.get("split_xy"):
            spec = FLEET[r["arm"]]
            ax.plot(*r["split_xy"], "x", ms=5, color=spec.color, mew=1.2)
    ax.set_aspect("equal")
    ax.set_xlim(-0.6, SHEET[0] + 0.6)
    ax.set_ylim(-0.6, SHEET[1] + 0.6)
    ax.set_title("where the certified head ends (split points)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=150, help="strokes per arm")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--arms", type=int, nargs="*", default=sorted(FLEET))
    ap.add_argument("--atlas-n", type=int, default=30, help="GO strokes per arm")
    ap.add_argument("--no-atlas", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "out"))
    ap.add_argument("--tag", default="", help="suffix for the output files")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(args.n, args.seed, args.arms)
    a_tasks = [] if args.no_atlas else [
        dict(arm=int(a), idx=i, out_dir=str(out_dir),
             seed=int(args.seed + 977 * a + i))
        for a in args.arms for i in range(args.atlas_n)]
    print(f"fuzz: {len(tasks)} strokes ({args.n}/arm x {len(args.arms)} arms) "
          f"+ {len(a_tasks)} atlas-GO strokes on {args.workers} workers")

    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers) as pool:
        recs = []
        for i, r in enumerate(pool.imap_unordered(run_task, tasks, chunksize=1), 1):
            recs.append(r)
            if i % 50 == 0 or i == len(tasks):
                c = Counter(x["status"] for x in recs)
                print(f"  {i:5d}/{len(tasks)}  ok={c['ok']} split={c['split']} "
                      f"degen={c['degenerate']} BUG={c['bug']} "
                      f"({time.time() - t0:.0f} s)")
        a_recs = list(pool.imap_unordered(atlas_task, a_tasks, chunksize=1)) \
            if a_tasks else []
    elapsed = time.time() - t0

    cfg = dict(n_per_arm=args.n, arms=list(args.arms), seed=args.seed,
               workers=args.workers, atlas_n=args.atlas_n, opts=OPTS,
               len_range=list(LEN_RANGE), class_p=list(CLASS_P))
    rep = summarise(recs, a_recs, cfg, elapsed)
    jpath = out_dir / f"fuzz_report{args.tag}.json"
    jpath.write_text(json.dumps(rep, indent=1, default=float))
    ppath = figure(recs, out_dir / f"fuzz_summary{args.tag}.png")

    t = rep["totals"]
    print(f"\n=== {rep['n']} strokes in {elapsed:.0f} s "
          f"({rep['timing']['total_plan_s']:.0f} s of planning) ===")
    print("  " + "  ".join(f"{k}={t.get(k, 0)}" for k in
                           ("ok", "split", "degenerate", "bug", "harness_error")))
    for k in ("per_class", "per_generator"):
        print(f"  {k}:")
        for name, c in rep[k].items():
            print(f"    {name:<10} " + "  ".join(f"{s}={c.get(s, 0)}" for s in
                                                 ("ok", "split", "degenerate", "bug")))
    print(f"  per arm: " + "  ".join(
        f"{a}:{c.get('ok', 0)}/{c.get('split', 0)}/{c.get('degenerate', 0)}/{c.get('bug', 0)}"
        for a, c in rep["per_arm"].items()) + "   (ok/split/degen/bug)")
    print("  split reasons: " + "  ".join(
        f"{c}:{dict(v)}" for c, v in rep["split_reasons"].items()))
    print(f"  s* of splits: {rep['split_s_star']['zero']} at zero, "
          f"quantiles {rep['split_s_star']['quantiles']}")
    print(f"  degenerate reasons: {rep['degenerate_reasons']}")
    print(f"  ok min_sigma: worst {rep['ok_min_sigma']['worst']}, "
          f"quantiles {rep['ok_min_sigma']['quantiles']}")
    d, v = rep["determinism"], rep["validator"]
    print(f"  determinism: {d['checked']} rechecked, {d['mismatches']} mismatches")
    print(f"  validator  : {v['checked']} ok-plans revalidated, {v['rejected']} rejected")
    print(f"  split heads: {rep['split_head_status']}   tails: {rep['split_tail_status']}")
    if not args.no_atlas:
        a = rep["atlas"]
        print(f"  atlas-GO   : {100 * a['rate']:.1f} % planned "
              f"({a['totals'].get('ok', 0)}/{sum(a['totals'].values())}), "
              f"{len(a['misses'])} misses")
        for m in a["misses"][:8]:
            print(f"    arm {m['arm']} {m['status']}/{m.get('reason', '')} "
                  f"{m.get('xy0')} -> {m.get('xy1')}")
    if rep["n_failures"]:
        print(f"  FAILURES: {rep['n_failures']}")
        for f in rep["failures"][:10]:
            print(f"    {f['kind']} arm {f['arm']} {f['gen']}/{f['cls']} "
                  f"seed {f['seed']}: {f.get('error', f.get('violations', ''))}")
    print(f"\nwrote {jpath}\nwrote {ppath}")
    return 0 if (t.get("bug", 0) == 0 and rep["n_failures"] == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
