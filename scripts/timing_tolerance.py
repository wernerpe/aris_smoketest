#!/usr/bin/env python3
"""How far out of step may two independently started arms be? -> a number.

THE PROBLEM THIS EXISTS TO ANSWER.  `scene_check` certifies a conducted
timeline in which every arm advances on ONE clock: at program time t, arm 31 is
at q31(t) and arm 71 is at q71(t), and the 50 mm the checker reports is the
closest those two configurations ever come *at the same instant*.  On hardware
there is no such clock.  libfranka's loop owns its thread and its socket, so
two arms are two processes started by two hands, and
`docs/ARCHITECTURE_V2.md` §5 Q3 says in as many words that the cross-process
fleet clock is *"the biggest structural gap in the ladder"* and is not started.

So the certificate as written does not apply, and the honest question is not
"is it safe?" but "how much skew does it survive?".  This script measures that:
it re-checks the SAME timeline with one arm's clock displaced by Δt, for a grid
of Δt, and reports the largest displacement at which every gate still holds.

    ARIS_RIG=proposed ARIS_TOOL=lateral ARIS_ARMS=31,71 \\
        python3 scripts/timing_tolerance.py out/unknown_h0970_schedule.npz \\
            --shift-arm 71 --json out/unknown_h0970_timing.json

WHAT "SHIFTED" MEANS, PRECISELY.  A shift of +Δt says arm 71 started Δt LATE:
before its own t = 0 it stands at its first commanded pose, and after its own
end it stands at its last.  Both of those are poses the conductor already put
it in, so the shifted timeline is built entirely out of certified
configurations — what is NOT certified, and what this measures, is the PAIRS
those configurations now form.  The timeline is lengthened by |Δt| so the late
arm finishes rather than being truncated.  Nothing is interpolated and no
control point moves; this is a re-indexing, which is why the answer is a
property of the programme and not of a re-plan.

READ THE RESULT THE CONSERVATIVE WAY.  The grid is a grid.  A pass at ±2 s and
a pass at ±5 s is not a proof of every value between them — clearance is not
monotone in Δt, and two arms drawing adjacent letters can be further apart at
5 s of skew than at 3 s.  `certified_window_s` is therefore the largest tested
|Δt| such that EVERY tested shift of that magnitude or smaller passed, and
`--dense` is there for when that number has to carry weight.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_SHIFTS = (-10.0, -5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0)


# ---------------------------------------------------------------------------
# 1. The re-indexing.  Pure: no fleet, no model, no import of aris_sixarm.

def shift_timeline(q, drawing, dt, arm, dt_shift):
    """One arm's clock displaced by `dt_shift` seconds. -> (q, drawing, k).

    `q` is `{arm: (M, 7)}` and `drawing` is `{arm: (M,) bool}`, both on one
    clock of step `dt`.  Every arm's series is extended to `M + |k|` samples by
    HOLDING its end poses, and the named arm's is additionally displaced by
    `k = round(dt_shift / dt)` samples.  A held sample is not drawing: a pen
    that is standing still is not laying ink, and saying otherwise would hand
    the paper gate a stationary pen-down it would rightly refuse.

    Returns the integer `k` as well, because `k * dt` and not `dt_shift` is the
    shift that was actually applied, and a report that prints the request
    rather than the deed is a report that lies by rounding.
    """
    if arm not in q:
        raise KeyError(f"arm {arm} is not in this timeline ({sorted(q)})")
    M = len(next(iter(q.values())))
    for a, v in q.items():
        if len(v) != M:
            raise ValueError(f"arm {a} has {len(v)} samples, arm "
                             f"{next(iter(q))} has {M}: not one clock")
    k = int(round(float(dt_shift) / float(dt)))
    N = M + abs(k)
    idx = np.arange(N)
    base = np.clip(idx, 0, M - 1)                 # hold the last pose at the end
    moved = np.clip(idx - k, 0, M - 1)            # ...and the first at the start
    out_q, out_d = {}, {}
    for a in q:
        take = moved if a == arm else base
        out_q[a] = np.asarray(q[a], float)[take]
        # A HELD SAMPLE IS NOT DRAWING.  `take` repeats an index exactly where
        # the arm is standing still, so the mask is the conducted mask AND the
        # places where the index actually advanced.
        moving = np.ones(N, bool)
        moving[1:] = np.diff(take) > 0
        moving[0] = bool(np.asarray(drawing[a])[take[0]])
        out_d[a] = np.asarray(drawing[a], bool)[take] & moving
    return out_q, out_d, k


# ---------------------------------------------------------------------------
# 2. The sweep

def sweep(npz, shifts=DEFAULT_SHIFTS, shift_arm=None, gate=None, sub=2,
          h=None, parks=None, clocking="uniform", verbose=True):
    """Re-check the timeline at each Δt. -> dict, ready for `json.dumps`."""
    from recheck_timeline import fleet_for            # sets ARIS_* defaults
    from aris_sixarm import scene_check

    z = np.load(npz, allow_pickle=False)
    arms = [int(v) for v in z["arms"]]
    dt = float(z["dt"])
    margin = float(z["margin"]) if gate is None else float(gate)
    pens = {int(x): float(p) for x, p in zip(z["arms"], z["pen_ext"])}
    q = {x: np.asarray(z[f"q_{x}"], float) for x in arms}
    draw = {x: np.asarray(z[f"seg_{x}"]) >= 0 for x in arms}
    if len(arms) < 2:
        raise SystemExit(f"{npz} has {len(arms)} arm(s); a timing tolerance is "
                         "a statement about a PAIR")
    arm = int(shift_arm) if shift_arm is not None else max(arms)
    fl, hh = fleet_for(h, parks, clocking)

    M = len(next(iter(q.values())))
    if verbose:
        print(f"{npz}: {len(arms)} arms {arms}, {M} frames, dt={dt:.4f} s, "
              f"duration {(M - 1) * dt:.2f} s")
        print(f"  shifting arm {arm}'s clock; gate {1000 * margin:.0f} mm "
              f"inter-arm, h={hh}")
        print(f"  {len(shifts)} shifts: "
              + ", ".join(f"{v:+g}" for v in shifts) + " s\n")

    rows = []
    for ds in shifts:
        qs, dsg, k = shift_timeline(q, draw, dt, arm, ds)
        rep = scene_check.check_timeline(qs, dt, margin, programs=None,
                                         h_inv=hh, pen_ext=pens, sub=sub,
                                         verbose=False, fleet=fl, drawing=dsg)
        wp = rep.get("worst_pair")
        row = dict(
            dt_request_s=float(ds), dt_applied_s=float(k * dt), samples=int(k),
            ok=bool(rep["ok"]),
            min_clearance_m=float(rep["min_clearance"]),
            margin_m=float(rep["min_clearance"] - margin),
            worst_pair=(None if not wp else [int(wp[0]), int(wp[1]),
                                             float(wp[2])]),
            self_failed=bool(rep.get("self_failed")),
            frame_failed=bool(rep.get("frame_failed")),
            column_failed=bool(rep.get("column_failed")),
            paper_failed=bool(rep.get("paper_failed")))
        rows.append(row)
        if verbose:
            wtxt = (f"  pair {wp[0]}-{wp[1]} at t={wp[2]:7.2f} s" if wp else "")
            print(f"  dt {row['dt_applied_s']:+7.3f} s  "
                  f"min inter-arm {1000 * row['min_clearance_m']:7.2f} mm  "
                  f"({1000 * row['margin_m']:+7.2f} mm vs gate)  "
                  f"{'PASS' if row['ok'] else 'FAIL'}{wtxt}")

    # THE WINDOW IS THE LARGEST MAGNITUDE WITH NOTHING FAILING UNDER IT.
    # Not "the largest that passed": clearance is not monotone in the skew, so
    # a pass at 5 s with a failure at 2 s certifies 1 s, not 5.
    mags = sorted({abs(r["dt_applied_s"]) for r in rows})
    window, first_bad = 0.0, None
    for m in mags:
        under = [r for r in rows if abs(r["dt_applied_s"]) <= m + 1e-12]
        if all(r["ok"] for r in under):
            window = m
        else:
            first_bad = first_bad if first_bad is not None else m
            break
    worst = min(rows, key=lambda r: r["min_clearance_m"])
    doc = dict(source=str(npz), arms=arms, shift_arm=arm, dt_s=dt,
               n_frames=int(M), duration_s=float((M - 1) * dt),
               gate_m=float(margin), h=float(hh), sub=int(sub),
               certified_window_s=float(window),
               first_failing_s=(None if first_bad is None else float(first_bad)),
               tested_s=[float(r["dt_applied_s"]) for r in rows],
               worst=worst, rows=rows)
    if verbose:
        print(f"\nTIMING TOLERANCE  +/- {window:g} s at the {1000 * margin:.0f} "
              f"mm gate, over the tested grid")
        if first_bad is not None:
            print(f"  the first magnitude that fails is {first_bad:g} s")
        print(f"  worst tested case: {1000 * worst['min_clearance_m']:.2f} mm "
              f"at dt {worst['dt_applied_s']:+g} s")
        if window <= 0.0:
            print("  !! ZERO WINDOW: these two arms may not be run "
                  "concurrently by two independently started executors.  "
                  "Run the ALTERNATING variant.")
    return doc


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz")
    ap.add_argument("--shift-arm", type=int, default=None,
                    help="whose clock moves (default: the highest arm id)")
    ap.add_argument("--shifts", default=None, metavar="S,S,...",
                    help="seconds, comma separated "
                         f"(default {','.join(f'{v:g}' for v in DEFAULT_SHIFTS)})")
    ap.add_argument("--dense", action="store_true",
                    help="0.25 s steps out to +/-10 s instead of the coarse "
                         "grid: 81 re-checks, for when the number has to carry "
                         "weight")
    ap.add_argument("--gate", type=float, default=None, metavar="M",
                    help="inter-arm gate (default: the npz's own margin)")
    ap.add_argument("--sub", type=int, default=2)
    ap.add_argument("--h", type=float, default=None)
    ap.add_argument("--parks", default=None)
    ap.add_argument("--clocking", default="uniform")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    if a.dense:
        shifts = tuple(np.round(np.arange(-10.0, 10.0001, 0.25), 3).tolist())
    elif a.shifts:
        shifts = tuple(float(v) for v in a.shifts.replace(",", " ").split())
    else:
        shifts = DEFAULT_SHIFTS
    doc = sweep(a.npz, shifts=shifts, shift_arm=a.shift_arm, gate=a.gate,
                sub=a.sub, h=a.h, parks=a.parks, clocking=a.clocking)
    if a.json:
        Path(a.json).write_text(json.dumps(doc, indent=1))
        print(f"wrote {a.json}")
    return 0 if doc["certified_window_s"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
