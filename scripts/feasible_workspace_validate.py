"""Second opinion on the feasible-workspace map, on a stratified sample.

Three questions, on the same cells:

  REPRODUCE   recompute all three layers from scratch, in a fresh process with
              cold caches, and compare cell for cell with `_raw.npz`.  A memo
              keyed on `id(spec)` and a module-global tool are exactly the
              things that make a sweep non-reproducible; this is the check.

  PROXY       would the cheap screen have done?  `paper.move_ok` on the
              park -> hover leg is the straight-line test a hover-plane roadmap
              would reduce to.  Agreement with the real `paper.route` is
              reported here so the decision to route every cell for real is a
              measurement rather than a preference.

  CHECKER     the producer paid `STATIC_PLAN_MARGIN` (63 mm) and the parked
              partners were vetoed at the conductor's 80 mm.  Re-measure the
              certified path against the INDEPENDENT checkers — `validate`'s
              own pose gate on the draw and hover poses, and the parked-chain
              clearance at `coordination`'s margin — so nothing rests on the
              planner agreeing with itself.

    ARIS_RIG=proposed ARIS_TOOL=lateral \\
        python3 scripts/feasible_workspace_validate.py --n 180
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("ARIS_RIG", "proposed")
os.environ.setdefault("ARIS_TOOL", "lateral")

from aris_sixarm import allocate, atlas, layout, paper, validate, writing  # noqa
import feasible_workspace as FW                                            # noqa


def sample(raw, arms, n, seed=0):
    """Stratified over arm x canvas third x verdict. -> [(arm, row, code)].

    `row` is the ATLAS row index the sweep stored in column 5, not a position
    in the raw array: the pool returns strided chunks out of order, so the two
    only agree by accident.  Getting this wrong is the way a validator
    reproduces 100 % of a comparison it is not making.
    """
    rng = np.random.default_rng(seed)
    ymax = FW.SHEET[1]
    out = []
    per = max(1, n // (len(arms) * 3 * 2))
    for a in arms:
        d = raw[f"arm{a}"]
        if not len(d):
            continue
        third = np.clip((d[:, 1] / (ymax / 3)).astype(int), 0, 2)
        for t in range(3):
            for good in (True, False):
                m = (third == t) & ((d[:, 2] == FW.FEASIBLE) == good)
                idx = np.where(m)[0]
                if not len(idx):
                    continue
                k = min(per, len(idx))
                out += [(a, int(d[i, 5]), int(d[i, 2]))
                        for i in rng.choice(idx, k, replace=False)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "out" / "feasible_workspace_raw.npz"))
    ap.add_argument("--atlas", default=str(FW.ATLAS_DIR))
    ap.add_argument("--out", default=str(ROOT / "out" / "feasible_workspace_validation.json"))
    ap.add_argument("--n", type=int, default=180)
    a = ap.parse_args()

    raw = np.load(a.raw)
    fl = layout.FLEET_PROPOSED
    parks = layout.Q_PARK_PROPOSED
    arms = sorted(fl)
    probe = allocate.ParkProbe(parks, fl, {k: fl[k].pen for k in fl},
                              h_inv=layout.LAYOUT_PROPOSED["h"])
    h = layout.LAYOUT_PROPOSED["h"]

    # the atlas rows, in the same order the sweep walked them
    GO = {}
    for arm in arms:
        arr, _ = atlas.load(a.atlas, arm)
        GO[arm] = arr[atlas.strict_go(arr)]

    picks = sample(raw, arms, a.n)
    print(f"{len(picks)} stratified cells")
    parks_np = {k: np.asarray(v, float) for k, v in parks.items()}

    paper.clear_cache()                     # cold, on purpose
    agree = same = 0
    proxy_tp = proxy_fp = proxy_fn = proxy_tn = 0
    bad_pose = []
    soft = []
    park_clear = []
    rows = []
    for k, (arm, i, got_code) in enumerate(picks):
        spec = fl[arm]
        r = GO[arm][i]
        x, y = float(r[0]), float(r[1])
        q_draw = np.asarray(r[atlas.QCOL:atlas.QCOL + 7], float)
        q_park = parks_np[arm]

        # SAME REDUNDANCY THE MAP USED: any certified hover on the ladder
        hovers = FW._certified_hovers(spec, q_draw, (x, y), h)
        proxy = None
        if not hovers:
            code, clear, real = FW.NO_HOVER, float("nan"), False
        else:
            code, clear, real = FW.NO_ROUTE, float("nan"), False
            for q_hov, z in hovers:
                beats = writing.enter_beats(spec, q_park, q_hov, q_draw,
                                            pen_ext=spec.pen, h_inv=h)
                if beats is None:
                    continue
                real = True
                c = float(probe.clearance(arm, FW._dense(beats["steps"],
                                                         q_park)))
                if c >= probe.margin:
                    code, clear = FW.FEASIBLE, c
                    break
                clear = c if clear != clear else max(clear, c)
            # THE PROXY, measured: the straight line park -> the FIRST hover,
            # which is all a hover-plane roadmap could ever see
            ok, _, _ = paper.move_ok(spec, q_park, hovers[0][0],
                                     pen_ext=spec.pen, h_inv=h,
                                     tip_floor=paper.travel_floor(
                                         writing.LIFT_Z, writing.LIFT_Z))
            proxy = bool(ok)
            proxy_tp += proxy and real
            proxy_fp += proxy and not real
            proxy_fn += (not proxy) and real
            proxy_tn += (not proxy) and not real

        same += (code == got_code)
        agree += 1

        # THE CHECKERS, independent of the planner: `validate` keeps
        # STATIC_MARGIN and MARGIN_GATE and knows nothing about what the router
        # spent
        if code == FW.FEASIBLE:
            # EACH POSE AGAINST THE GATE IT IS ACTUALLY HELD TO.  A drawing
            # pose owes `validate.MARGIN_GATE` (0.15 rad); a HOVER owes
            # `writing.HOVER_MARGIN` (0.10), deliberately looser — see the note
            # by `writing.HOVER_MARGIN`.  Checking a hover at 0.15 measures the
            # difference between two of the package's own constants and calls
            # it a defect.  Both are reported: `bad_poses` is the real gate,
            # `hover_below_draw_gate` is the softer one, for information.
            for nm, q, g in (("draw", q_draw, validate.MARGIN_GATE),
                             ("hover", hovers[0][0], writing.HOVER_MARGIN)):
                v = validate.check_pose(q, spec, h_inv=h, pen_ext=spec.pen,
                                        margin_gate=g)
                if not v.get("ok", False):
                    bad_pose.append([int(arm), x, y, nm, g,
                                     v.get("violations", [])[:3]])
            vh = validate.check_pose(hovers[0][0], spec, h_inv=h,
                                     pen_ext=spec.pen)
            soft.append(not vh.get("ok", False))
            park_clear.append(clear)
        rows.append(dict(arm=int(arm), x=x, y=y, want=int(got_code),
                         got=int(code), proxy=proxy,
                         clear=None if clear != clear else round(clear, 5)))
        if (k + 1) % 25 == 0:
            print(f"  {k + 1}/{len(picks)}  reproduced {same}/{agree}",
                  flush=True)

    n_pr = proxy_tp + proxy_fp + proxy_fn + proxy_tn
    out = dict(
        n=len(picks),
        reproduced=same, reproduced_pct=round(100.0 * same / max(agree, 1), 2),
        proxy=dict(n=n_pr, agree=proxy_tp + proxy_tn,
                   agree_pct=round(100.0 * (proxy_tp + proxy_tn)
                                   / max(n_pr, 1), 2),
                   straightline_ok_and_flyable=proxy_tp,
                   straightline_ok_but_refused=proxy_fp,
                   straightline_blocked_but_router_found_a_way=proxy_fn,
                   straightline_blocked_and_refused=proxy_tn),
        park_clearance=dict(
            n=len(park_clear),
            min_m=round(float(np.min(park_clear)), 4) if park_clear else None,
            median_m=round(float(np.median(park_clear)), 4) if park_clear else None,
            gate_m=round(float(probe.margin), 4)),
        bad_poses=bad_pose,
        gates=dict(draw_pose=validate.MARGIN_GATE,
                   hover=writing.HOVER_MARGIN,
                   note="a hover is held to HOVER_MARGIN by the pipeline, not "
                        "to the drawing-pose MARGIN_GATE"),
        hover_below_draw_gate=int(sum(soft)),
        hover_below_draw_gate_pct=round(100.0 * sum(soft) / max(1, len(soft)), 1),
        rows=rows,
    )
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nreproduced {out['reproduced_pct']:.2f}% of {out['n']} cells")
    p = out["proxy"]
    print(f"straight-line proxy agrees with the router on "
          f"{p['agree_pct']:.2f}% of {p['n']} "
          f"(it would have WRONGLY PASSED {p['straightline_ok_but_refused']}, "
          f"wrongly failed {p['straightline_blocked_but_router_found_a_way']})")
    print(f"parked-partner clearance on feasible cells: "
          f"min {out['park_clearance']['min_m']} m, gate "
          f"{out['park_clearance']['gate_m']} m")
    print(f"poses failing the gate they are held to: {len(bad_pose)}")
    print(f"  (of the hovers, {sum(soft)} of {len(soft)} sit under the "
          f"STRICTER drawing-pose gate {validate.MARGIN_GATE} — expected: a "
          f"hover owes {writing.HOVER_MARGIN})")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
