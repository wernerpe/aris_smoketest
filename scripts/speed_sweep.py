#!/usr/bin/env python3
"""Does the min-travel band + fiber menus win or lose at the REAL draw speed?

    python3 scripts/speed_sweep.py                    # run the whole 2 x 4 grid
    python3 scripts/speed_sweep.py --collect-only     # just re-tabulate out/

The 2026-08-20 measurement (`docs/CONCURRENCY.md`, `docs/REDUNDANCY.md`) put
both halves of that refactor behind off-by-default switches, because on the
shipped two-pass logo they cut reconfiguration 65 % and cost 48 % of makespan.
That measurement was taken at ONE draw speed -- 0.12 m/s, the concurrent
animation default (`writing.DRAW_SPEED_FLEET`) -- and the mechanism that made
it expensive is speed-dependent in a way the verdict was not:

    writing._draw_time(qd, ud, dur) = max(dur, need_max)
        dur      = length / draw_speed          <- falls as the speed falls
        need_max = max_i |dq_i| / (QD_MAX * qd_frac) / du_i

`need_max` is a property of the PATH THROUGH THE BAND alone: rad per unit of
normalised arc, divided by a velocity limit.  It does not know what speed the
pen is moving at.  So each segment has a critical speed

    v* = length / need_max

above which the joint cap binds and the ink is stretched, and below which the
material's `length / draw_speed` wins and the ink time is EXACTLY invariant
across variants.  Constraining the band (either half of the refactor) raises
`need_max` and therefore raises v*; it cannot touch `length`.  The claim that
the menus rest on -- "interior draw time is invariant across variants" -- is
therefore not false in general, it is false ABOVE v* and true below it.

The rig's own material speed is `pacing.V_DRAW` = 0.02 m/s, six times slower
than the number the refactor was judged at.  This script measures the same A/B
across the speed range and reports where the verdict flips.  Nothing here
changes a default: `pwl.OBJECTIVE` and `allocate.CLUSTER` are read, not
written, and the "features" arm of every pair is `--cluster --band-objective
min_travel` on the command line exactly as a caller would pass it.

Writes out/speed_sweep.json and prints the markdown table docs/BENCH.md carries.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

# The shipped two-pass logo, flag for flag as README.md reproduces it.  These
# are the ONLY constants here that are not a default: the placement, the pens
# and the coverage floor are what make this the logo and not a bench drawing.
CANON = ["--arms", "all", "--two-pass",
         "--pens", "2:300,31:200,71:200,97:200",
         "--max-probes", "5", "--min-coverage", "0.99",
         "--target-width", "1.2969246423461636", "--offset", "0.1", "0.05",
         "--fps", "12", "--substeps", "4"]
# The same run without the conductor's own flags, for --ink-curve: `add_args`
# in csail_allocate.py knows nothing about frames or coverage floors.  The
# draw speed is PINNED at the shipped 0.12 here and not swept, on purpose:
# --ink-curve is ONE allocation re-priced under many clocks, so that what moves
# in its table is the timing rule alone.  The allocator balances on seconds
# (`allocate.rebalance`), so sweeping the speed here would sweep the assignment
# too and the two effects would be inseparable.  The grid above does sweep it,
# and its `stretch` column is the same quantity measured at the right
# allocation; these two disagreeing is informative, not a bug.
ALLOC_BASE = ["--arms", "all", "--two-pass",
              "--pens", "2:300,31:200,71:200,97:200", "--max-probes", "5",
              "--target-width", "1.2969246423461636",
              "--offset", "0.1", "0.05"]
ALLOC_CANON = [*ALLOC_BASE, "--draw-speed", "0.12"]
# `--cluster` needs `min_travel` with it: the menus are variants of the band
# the objective chooses, so measuring one without the other measures neither.
FEATURES = {"dflt": [],
            "clus": ["--cluster", "--band-objective", "min_travel"]}
SPEEDS = (0.02, 0.05, 0.08, 0.12)


def tag(speed, feat):
    """The `--tag` a grid cell's schedule is written under.

    Centimetres per second, zero-padded to three digits, so 0.02 m/s is
    `_sp002_dflt` and the files sort in speed order.  Two speeds that agree to
    the centimetre would collide; the grid is coarse on purpose and does not.
    """
    return f"_sp{round(100 * speed):03d}_{feat}"


def run_one(speed, feat, out_dir, log_dir):
    """One csail_schedule.py run. -> (tag, seconds of wall clock)."""
    t = tag(speed, feat)
    cmd = [sys.executable, "-u", str(ROOT / "scripts/csail_schedule.py"),
           *CANON, "--out", str(out_dir),
           "--draw-speed", str(speed), *FEATURES[feat], "--tag", t]
    log = Path(log_dir) / f"{t.lstrip('_')}.log"
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
    if rc != 0:
        raise SystemExit(f"{t} failed (rc {rc}); see {log}")
    return t, time.time() - t0


def read_one(out_dir, speed, feat):
    """The row this grid cell contributes. -> dict.

    Every field is read from the schedule's OWN summary, not recomputed here,
    so the table cannot drift from what `scene_check` signed off on.
    """
    p = Path(out_dir) / f"csail_schedule{tag(speed, feat)}.json"
    d = json.loads(p.read_text())
    phases = d["phases"]
    draw = sum(v for ph in phases for v in ph["arm_draw_s"].values())
    # what the material alone would ask for: the cap has not bound anywhere
    # when this equals `draw` to the float
    ink_m = sum(ph["drawn_m"] for ph in phases)
    return dict(
        speed=speed, features=feat,
        makespan_s=d["makespan_s"], transit_s=d["transit_s"],
        reconfig_rad=d["reconfig_rad"], pause_total=d["pause_total"],
        min_clearance_mm=1000 * d["min_clearance"],
        coverage_pct=100 * d["coverage"], n_segments=d["n_segments"],
        draw_s=draw, unstretched_draw_s=ink_m / speed,
        stretch=draw / max(ink_m / speed, 1e-9),
        floor_s=sum(ph["floor_s"] for ph in phases),
        drawn_m=ink_m, scene_check=all(ph["scene_check_ok"] for ph in phases))


def table(rows):
    """The 2 x N grid, markdown, one block per metric that moves."""
    by = {(r["speed"], r["features"]): r for r in rows}
    speeds = sorted({r["speed"] for r in rows})
    out = ["| draw speed | features | makespan | floor | draw (ink) | stretch |"
           " transit | reconfig | pause | clearance | coverage |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for v in speeds:
        for feat, name in (("dflt", "defaults"),
                           ("clus", "min_travel + cluster")):
            r = by.get((v, feat))
            if r is None:
                continue
            out.append(
                f"| {v:.2f} m/s | {name} | **{r['makespan_s']:.1f} s** | "
                f"{r['floor_s']:.1f} s | {r['draw_s']:.1f} s | "
                f"{r['stretch']:.2f}x | {r['transit_s']:.2f} s | "
                f"{r['reconfig_rad']:.2f} rad | {r['pause_total']:.1f} s | "
                f"{r['min_clearance_mm']:.1f} mm | "
                f"{r['coverage_pct']:.4f} % |")
    out.append("")
    out.append("| draw speed | makespan defaults | makespan features | change |"
               " verdict |")
    out.append("|---|---|---|---|---|")
    for v in speeds:
        a, b = by.get((v, "dflt")), by.get((v, "clus"))
        if not (a and b):
            continue
        d = b["makespan_s"] - a["makespan_s"]
        pct = 100 * d / a["makespan_s"]
        out.append(f"| {v:.2f} m/s | {a['makespan_s']:.1f} s | "
                   f"{b['makespan_s']:.1f} s | {d:+.1f} s ({pct:+.1f} %) | "
                   + ("**features win**" if d < 0 else "defaults win") + " |")
    return "\n".join(out)


def ink_curve(speeds):
    """Where the joint cap stops binding, from the segments themselves.

    The grid above measures the CONSEQUENCE (a makespan) through an allocator,
    a sequencer and a conductor.  This measures the MECHANISM directly and
    needs no conductor at all: allocate once per feature setting, densify every
    certified segment exactly as `writing.arm_program` will, and read off

        need_i  = `writing._draw_time(qd, ud, 0.0)`   seconds, speed-free
        v*_i    = length_i / need_i                   m/s, the critical speed

    A segment's ink is stretched iff the draw speed is above its own v*.  So
    min_i v*_i is the speed below which the ink time is EXACTLY the material's
    length / speed for every segment and every variant -- the regime the fiber
    menus were designed for -- and max_i v*_i is the speed above which every
    segment is joint-limited and the draw speed has stopped meaning anything.
    """
    from aris_sixarm import writing                             # noqa: E402
    from aris_sixarm.fleet import FLEET, H_INV_DEFAULT          # noqa: E402
    sys.path.insert(0, str(ROOT / "scripts"))
    import csail_allocate                                       # noqa: E402
    import numpy as np                                          # noqa: E402

    out = {}
    for feat, extra in FEATURES.items():
        ap = csail_allocate.add_args(argparse.ArgumentParser())
        a = ap.parse_args([*ALLOC_CANON, *extra])
        phases, _strokes, _info = csail_allocate.run_allocation(a)
        rows = []
        for ph in phases:
            for aid in ph["arms"]:
                spec = FLEET[aid]
                for seg in ph["programs"][aid]:
                    qs = np.asarray(seg["plan"]["qs"], float)
                    pts = np.asarray(seg["plan"]["pts"], float)
                    qd, ud, _ = writing.densify(qs, pts, spec, H_INV_DEFAULT,
                                                ph["pens"][aid])
                    # dur = 0 -> draw_duration returns need_max alone
                    need = writing._draw_time(qd, ud, 0.0, writing.QD_FRAC)
                    L = float(seg["length"])
                    rows.append(dict(arm=aid, phase=ph["name"], length=L,
                                     need_s=need,
                                     v_star=L / need if need > 0 else np.inf))
        vs = np.array([r["v_star"] for r in rows])
        Ls = np.array([r["length"] for r in rows])
        nd = np.array([r["need_s"] for r in rows])

        def floor_ink(v):
            """Σ over phases of the busiest arm's ink. -> seconds.

            The TOTAL ink is not what a makespan is made of: the arms draw at
            once, so each phase pays its busiest arm and the phases add.  A
            change can lower the total and raise this, which is exactly what
            the 2026-08-20 measurement caught.
            """
            per = {}
            for r, t in zip(rows, np.maximum(Ls / v, nd)):
                k = (r["phase"], r["arm"])
                per[k] = per.get(k, 0.0) + float(t)
            ph = {}
            for (name, _arm), t in per.items():
                ph[name] = max(ph.get(name, 0.0), t)
            return float(sum(ph.values()))

        out[feat] = dict(
            n=len(rows), v_star_min=float(vs.min()),
            v_star_med=float(np.median(vs)), v_star_max=float(vs.max()),
            total_m=float(Ls.sum()),
            ink_s={f"{v:g}": float(np.maximum(Ls / v, nd).sum())
                   for v in speeds},
            material_s={f"{v:g}": float(Ls.sum() / v) for v in speeds},
            floor_ink_s={f"{v:g}": floor_ink(v) for v in speeds},
            n_capped={f"{v:g}": int((vs < v).sum()) for v in speeds},
            rows=rows)
    return out


def floor_cell(speed, feat, out_dir, qd_frac=None):
    """One grid cell's FLOOR, allocated and frozen but never conducted.

    The floor -- Σ over phases of the busiest arm's own nominal programme --
    is what the allocator moves and what no schedule can beat, and on the six
    cells this repository has conducted it is `max(arm_nominal_s)` to 0.02 s
    and `draw + transit` for the arm that sets it.  Conducting is what costs
    the wall clock (45 min a cell); allocating and freezing costs about three,
    so the speed sweep is done here and the conductor is asked only for the
    OVERHEAD, which `--floor-grid` takes from the cells that were conducted.

    The allocation is redone AT `speed` on purpose: `allocate.rebalance` scores
    a candidate assignment in seconds, so who draws what is itself a function
    of the draw speed and freezing the 0.12 m/s assignment would measure the
    wrong thing.
    """
    from aris_sixarm import writing                             # noqa: E402
    from aris_sixarm.fleet import FLEET, H_INV_DEFAULT          # noqa: E402
    sys.path.insert(0, str(ROOT / "scripts"))
    import csail_allocate                                       # noqa: E402

    argv = [*ALLOC_BASE, "--draw-speed", str(speed), *FEATURES[feat]]
    if qd_frac is not None:
        argv += ["--qd-frac", str(qd_frac)]
    a = csail_allocate.add_args(argparse.ArgumentParser()).parse_args(argv)
    phases, _strokes, _info = csail_allocate.run_allocation(a)
    # COVERAGE IS THE INVARIANT THE WHOLE SWEEP RESTS ON, so it is recorded per
    # cell from the allocator's own totals rather than inferred from the sum of
    # segment lengths (which double-counts a split's shared endpoint).
    T = csail_allocate.totals(phases)

    cell = dict(speed=speed, features=feat, qd_frac=qd_frac or a.qd_frac,
                phases=[], floor_s=0.0, draw_s=0.0, transit_s=0.0,
                drawn_m=0.0, n_segments=0, n_capped=0, n_seg_total=0,
                coverage_pct=100 * T["covered"], traced_m=T["traced"],
                dropped_m=T["dropped"], n_dropped=T["n_dropped"])
    for ph in phases:
        per = {}
        for aid in ph["arms"]:
            segs = ph["programs"][aid]
            if not segs:
                continue
            spec, pen = FLEET[aid], ph["pens"][aid]
            draw = 0.0
            for s in segs:
                t = writing.segment_draw_time(spec, s, speed, a.qd_frac,
                                              H_INV_DEFAULT, pen)
                draw += t
                cell["n_seg_total"] += 1
                # capped == the joint limit, not the material, set this time
                if t > float(s["length"]) / speed + 1e-9:
                    cell["n_capped"] += 1
            tr = float(ph["sequence"][aid]["cost"])
            per[aid] = dict(draw_s=draw, transit_s=tr, nominal_s=draw + tr,
                            metres=sum(float(s["length"]) for s in segs))
            cell["draw_s"] += draw
            cell["transit_s"] += tr
            cell["drawn_m"] += per[aid]["metres"]
            cell["n_segments"] += len(segs)
        fl = max(v["nominal_s"] for v in per.values()) if per else 0.0
        cell["floor_s"] += fl
        cell["phases"].append(dict(name=ph["name"], floor_s=fl, arms=per))
    p = Path(out_dir) / f"floor{tag(speed, feat)}.json"
    p.write_text(json.dumps(cell, indent=1))
    return cell


# OFFSET = conducted makespan - the floor computed by `floor_cell`.  It absorbs
# BOTH things that function leaves out, and it is calibrated rather than
# derived because they are not separable from the outside:
#
#   * the parts of an arm's nominal programme that are not draw + inter-segment
#     transit -- the entry lift from the ready pose, the exit lift, and the
#     idle policy's taxi/retreat.  `floor_cell` reproduces the conducted floor
#     only to within -15 to -21 s for exactly this reason;
#   * the conductor's own pauses on top of the floor (4.5-4.7 s for the
#     defaults, 14.2-16.0 s with the features).
#
# Every term in it is speed-INDEPENDENT by construction: lifts and taxis are
# joint-space moves paced by `qd_frac`, not by `draw_speed`, and the pause is
# assumed not to grow as the draw speed falls because a slower programme
# spreads the same hover conflicts over more seconds -- arms meet in time LESS
# often, not more.  That assumption is the load-bearing one and it is stated in
# docs/BENCH.md, not buried here.
#
# Calibrated on the four cells that were conducted at qd_frac 0.30:
#   defaults           116.708 - 90.927 = 25.78 (0.08)   104.021 - 81.789 = 22.23 (0.12)
#   min_travel+cluster 140.083 - 107.111 = 32.97 (0.08)  123.375 - 93.663 = 29.71 (0.12)
# The two speeds agree to ~3.5 s, which is the honest error bar on every
# estimated cell below.
OFFSET_S = {"dflt": 24.01, "clus": 31.34}
OFFSET_SPREAD_S = 3.5


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--speeds", default=",".join(str(s) for s in SPEEDS))
    ap.add_argument("--out", default=str(ROOT / "out"))
    ap.add_argument("--logs", default=str(ROOT / "out"))
    ap.add_argument("--collect-only", action="store_true",
                    help="skip the runs and re-tabulate what is in --out")
    ap.add_argument("--ink-curve", action="store_true",
                    help="allocate only, and report the per-segment critical "
                         "speed v* = length / need instead of the grid")
    ap.add_argument("--floor-cell", nargs=2, metavar=("SPEED", "FEAT"),
                    default=None,
                    help="allocate + freeze ONE cell and write its floor; no "
                         "conductor, ~3 min instead of ~45")
    ap.add_argument("--floor-grid", action="store_true",
                    help="tabulate the floor cells into estimated makespans")
    ap.add_argument("--qd-frac", type=float, default=None)
    a = ap.parse_args(argv)
    speeds = [float(x) for x in a.speeds.split(",")]
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(a.logs).mkdir(parents=True, exist_ok=True)

    if a.floor_cell:
        v, feat = float(a.floor_cell[0]), a.floor_cell[1]
        c = floor_cell(v, feat, out_dir, a.qd_frac)
        print(f"{tag(v, feat)}: floor {c['floor_s']:.3f} s  draw {c['draw_s']:.2f}"
              f"  transit {c['transit_s']:.2f}  {c['n_capped']}/"
              f"{c['n_seg_total']} segments joint-capped  "
              f"coverage {c['coverage_pct']:.4f} %")
        return c

    if a.floor_grid:
        cells = {}
        for v in speeds:
            for feat in FEATURES:
                p = out_dir / f"floor{tag(v, feat)}.json"
                if p.exists():
                    cells[(v, feat)] = json.loads(p.read_text())
        # VALIDATE BEFORE EXTRAPOLATING: where a cell was also conducted, the
        # floor computed here must reproduce the conducted floor.  A model that
        # cannot rebuild the rows it was fitted on has no business predicting
        # the others, so the residual is printed, not assumed.
        print("model check -- computed floor vs conducted floor")
        for (v, feat), c in sorted(cells.items()):
            q = out_dir / f"csail_schedule{tag(v, feat)}.json"
            if not q.exists():
                continue
            d = json.loads(q.read_text())
            got = sum(ph["floor_s"] for ph in d["phases"])
            m_est = c["floor_s"] + OFFSET_S[feat]
            print(f"  {tag(v, feat)}: computed floor {c['floor_s']:8.3f} s vs "
                  f"conducted {got:8.3f} s | makespan estimated "
                  f"{m_est:8.3f} s vs measured {d['makespan_s']:8.3f} s  "
                  f"(error {m_est - d['makespan_s']:+.2f} s)")
        print("\n| draw speed | features | floor | draw (ink) | stretch | "
              "capped | transit | makespan (est) |")
        print("|---|---|---|---|---|---|---|---|")
        est = {}
        for v in speeds:
            for feat, name in (("dflt", "defaults"),
                               ("clus", "min_travel + cluster")):
                c = cells.get((v, feat))
                if not c:
                    continue
                mat = c["drawn_m"] / v
                m = c["floor_s"] + OFFSET_S[feat]
                est[(v, feat)] = m
                print(f"| {v:.2f} m/s | {name} | {c['floor_s']:.1f} s | "
                      f"{c['draw_s']:.1f} s | {c['draw_s'] / mat:.3f}x | "
                      f"{c['n_capped']}/{c['n_seg_total']} | "
                      f"{c['transit_s']:.1f} s | **{m:.1f} s** |")
        print("\n| draw speed | defaults | min_travel + cluster | change | verdict |")
        print("|---|---|---|---|---|")
        for v in speeds:
            if (v, "dflt") in est and (v, "clus") in est:
                x, y = est[(v, "dflt")], est[(v, "clus")]
                print(f"| {v:.2f} m/s | {x:.1f} s | {y:.1f} s | "
                      f"{y - x:+.1f} s ({100 * (y - x) / x:+.1f} %) | "
                      + ("**features win**" if y < x else "defaults win") + " |")
        (out_dir / "speed_sweep_floor.json").write_text(json.dumps(
            {f"{v}_{f}": c for (v, f), c in cells.items()}, indent=1))
        return cells

    if a.ink_curve:
        ic = ink_curve(speeds)
        (out_dir / "speed_sweep_ink.json").write_text(json.dumps(ic, indent=1))
        for feat, d in ic.items():
            print(f"\n{feat}: {d['n']} segments, {d['total_m']:.3f} m, "
                  f"v* min {d['v_star_min']:.4f} / median "
                  f"{d['v_star_med']:.4f} / max {d['v_star_max']:.4f} m/s")
            for v in speeds:
                k = f"{v:g}"
                print(f"   {v:.2f} m/s: ink {d['ink_s'][k]:8.2f} s vs material "
                      f"{d['material_s'][k]:8.2f} s "
                      f"({d['ink_s'][k] / d['material_s'][k]:.3f}x), "
                      f"{d['n_capped'][k]}/{d['n']} segments capped, "
                      f"busiest-arm ink floor {d['floor_ink_s'][k]:.2f} s")
        return ic

    if not a.collect_only:
        for v in speeds:
            for feat in FEATURES:
                t, wall = run_one(v, feat, out_dir, a.logs)
                print(f"{t}: {wall:.0f} s of wall clock")

    rows = []
    for v in speeds:
        for feat in FEATURES:
            try:
                rows.append(read_one(out_dir, v, feat))
            except FileNotFoundError:
                print(f"missing: {tag(v, feat)}", file=sys.stderr)
    (out_dir / "speed_sweep.json").write_text(json.dumps(rows, indent=1))

    cov = {round(r["coverage_pct"], 4) for r in rows}
    seg = {r["n_segments"] for r in rows}
    print(table(rows))
    print(f"\ncoverage across the grid: {sorted(cov)}  "
          f"segments: {sorted(seg)}  "
          f"scene_check: {'PASS' if all(r['scene_check'] for r in rows) else 'FAIL'}")
    if len(cov) != 1:
        print("COVERAGE MOVED ACROSS THE GRID -- the A/B is not an A/B",
              file=sys.stderr)
    return rows


if __name__ == "__main__":
    main()
