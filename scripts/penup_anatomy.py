#!/usr/bin/env python3
"""Where a staged programme's PEN-UP time actually goes.

Pen-up is the staged pipeline's dominant cost: on
`out/staged_csail_h097_program_lf2_s150.json`, stage A, arm 71, the pen-up legs
are 70.5 s of a 95.9 s stage (74 %).  Knowing the total is not enough to fix it,
because three different faults all show up as "a slow pen-up":

  (a) REDUNDANCY FLIP  -- the joints do far more work than moving the tip from
      here to there needs, because the two ends sit on different IK sheets (a
      different tool yaw / q7 / elbow branch) and the leg has to fold the arm
      over to get between them.  The give-away is joint path with nothing in
      task space to show for it: arm 71's leg 9 moves 19.8 rad to hop the pen
      3 mm.
  (b) TALL / WANDERING ROUTE -- the tip climbs far above the hover ladder and
      comes back, or the joint path is several times the net joint change, i.e.
      the router went somewhere and returned.  The give-away is vertical travel
      well past the honest `lift + lower`, or a path/net ratio above 2.
  (c) HONEST HOP -- the leg lifts, crosses, and lowers, and the joints move
      about as much as that costs.

THE RULE, stated once (all thresholds are module constants below).  For every
pen-up leg, from the programme's own trajectory:

    dq     = sum over samples and joints of |dq_j|            (rad)
    net    = sum over joints of |q_end - q_start|             (rad)
    hop    = || xy_end - xy_start ||                          (m)
    ztrav  = sum over samples of |dz|                         (m)
    zmax   = highest tip z reached                            (m)

An honest leg lifts LIFT_Z, crosses `hop`, and lowers LIFT_Z, so its honest
vertical travel is ZTRAV_REF = 2 * LIFT_Z and its honest joint path is about
RATE_REF * (hop + ZTRAV_REF).  RATE_REF is calibrated on the legs that ARE
honest -- the short hops in this programme sit at 11-20 rad/m -- and is set
deliberately high (25 rad/m) so only clear outliers are named.  Then

    climb_excess = max(0, ztrav - ZTRAV_REF)         # m of pointless climbing
    climb_rad    = RATE_REF * climb_excess           # joint work that explains
    flip_rad     = max(0, dq - RATE_REF * (hop + ztrav))
                                                     # joint work NOTHING in
                                                     # task space explains

    flip   if flip_rad >= FLIP_RAD and flip_rad > climb_rad
    tall   elif climb_excess > TALL_M or zmax > HOVER_M + Z_EPS
           or (dq >= WANDER_RAD and dq > WANDER_RATIO * net)
    honest otherwise

`flip` is tested first on purpose: a leg whose tip barely moves while the joints
churn is a redundancy fault whatever its vertical profile, and `flip_rad`
already has the climbing subtracted out, so a merely tall leg cannot be called
a flip.  Legs that end at the park pose are flagged `home=True` and reported on
their own line, because the park pose is high by design and the climb is not
waste.

When the programme carries per-piece `hover_in` / `hover_out` (schema 2 does),
each leg is also split into LIFT (ink end -> hover out), TRAVEL (hover out ->
hover in) and LOWER (hover in -> ink start) so a flip can be attributed: a
lift/lower flip is a HOVER chosen off the ink's sheet, a travel flip is two
adjacent PIECES on different sheets.

Usage
-----
    scripts/penup_anatomy.py out/staged_csail_h097_program_lf2_s150.json
    scripts/penup_anatomy.py PROG.json --legs --json out/penup_anatomy.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------------
# the rule's constants -- see the module docstring
# ---------------------------------------------------------------------------
LIFT_Z = 0.06           # m, the pen-up lift height (writing.LIFT_Z)
ZTRAV_REF = 2 * LIFT_Z  # m, an honest leg's vertical travel: up and back down
HOVER_M = 0.12          # m, the hover ladder's reference rung
Z_EPS = 0.010           # m, tolerance on "above the ladder"
RATE_REF = 25.0         # rad/m, joint path an honest pen-up spends per metre
FLIP_RAD = 6.0          # rad, unexplained joint path that makes a leg a flip
TALL_M = 0.06           # m, vertical travel past ZTRAV_REF that makes it tall
WANDER_RAD = 6.0        # rad, floor under the path/net wander test
WANDER_RATIO = 2.0      # path > 2 x net is a there-and-back route

CLASSES = ("flip", "tall", "honest")


# ---------------------------------------------------------------------------
# 1.  the metrics, and the rule, on plain arrays (no rig import -- testable)
# ---------------------------------------------------------------------------
def leg_metrics(Q: np.ndarray, X: np.ndarray, dur_s: float) -> dict:
    """Raw anatomy of one pen-up leg. -> dict.

    `Q` is (N, 7) joint samples, `X` is (N, 3) WORLD tip positions, both in
    trajectory order and both covering exactly the leg (ink end .. ink start).
    """
    Q = np.asarray(Q, float).reshape(-1, Q.shape[-1] if Q.ndim > 1 else 7)
    X = np.asarray(X, float).reshape(-1, 3)
    if len(Q) < 2 or len(X) < 2:
        raise ValueError("a leg needs at least two samples")
    dq = float(np.abs(np.diff(Q, axis=0)).sum())
    net = float(np.abs(Q[-1] - Q[0]).sum())
    hop = float(np.linalg.norm(X[-1, :2] - X[0, :2]))
    ztrav = float(np.abs(np.diff(X[:, 2])).sum())
    tip_path = float(np.linalg.norm(np.diff(X, axis=0), axis=1).sum())
    return dict(dur_s=float(dur_s), dq_rad=dq, net_rad=net, hop_m=hop,
                ztrav_m=ztrav, zmax_m=float(X[:, 2].max()),
                tip_path_m=tip_path, n=int(len(Q)))


def classify(m: dict) -> dict:
    """The rule. -> dict with `kind` and the two excesses it was decided on."""
    dq, net = m["dq_rad"], m["net_rad"]
    hop, ztrav = m["hop_m"], m["ztrav_m"]
    climb_excess = max(0.0, ztrav - ZTRAV_REF)
    climb_rad = RATE_REF * climb_excess
    flip_rad = max(0.0, dq - RATE_REF * (hop + ztrav))
    if flip_rad >= FLIP_RAD and flip_rad > climb_rad:
        kind = "flip"
    elif (climb_excess > TALL_M or m["zmax_m"] > HOVER_M + Z_EPS
          or (dq >= WANDER_RAD and dq > WANDER_RATIO * max(net, 1e-9))):
        kind = "tall"
    else:
        kind = "honest"
    return dict(kind=kind, flip_rad=flip_rad, climb_rad=climb_rad,
                climb_excess_m=climb_excess)


# ---------------------------------------------------------------------------
# 2.  pulling the legs out of a staged programme
# ---------------------------------------------------------------------------
def _world_tip(q: np.ndarray, spec, h_inv: float) -> np.ndarray:
    from aris_sixarm import frames
    T = spec.T_world_base(h_inv)
    tip = frames.tip_pos_many(np.asarray(q, float), pen_ext=spec.pen_ext)
    return (T[:3, :3] @ tip.T).T + T[:3, 3]


def _window(t: np.ndarray, t0: float, t1: float) -> tuple[int, int]:
    """The sample indices that bracket [t0, t1].

    The programme's trajectory is DECIMATED -- a straight joint-space transit
    keeps only its knots -- so a leg can hold as few as three samples, and
    dropping an endpoint to a float comparison loses most of the leg.  Nearest
    index on each end, never an open comparison.
    """
    i0 = int(np.argmin(np.abs(t - t0)))
    i1 = int(np.argmin(np.abs(t - t1)))
    return i0, max(i1, i0 + 1)


def _sheet_split(pieces: list, seg: int) -> dict | None:
    """LIFT / TRAVEL / LOWER joint path around one leg, from the hovers."""
    if seg + 1 >= len(pieces):
        return None
    a, b = pieces[seg], pieces[seg + 1]
    for k, p in (("q_last", a), ("hover_out", a), ("hover_in", b),
                 ("q_first", b)):
        if p.get(k) is None:
            return None
    ql = np.asarray(a["q_last"], float)
    ho = np.asarray(a["hover_out"], float)
    hi = np.asarray(b["hover_in"], float)
    qf = np.asarray(b["q_first"], float)
    return dict(lift_rad=float(np.abs(ho - ql).sum()),
                travel_rad=float(np.abs(hi - ho).sum()),
                lower_rad=float(np.abs(qf - hi).sum()))


def arm_legs(arm_prog: dict, spec, h_inv: float, stage: int) -> list[dict]:
    """Every pen-up leg of one arm in one stage, measured and classified."""
    tr = arm_prog.get("trajectory")
    legs = arm_prog.get("legs") or []
    if not tr or not legs:
        return []
    t = np.asarray(tr["t"], float)
    q = np.asarray(tr["q"], float)
    X = _world_tip(q, spec, h_inv)
    pieces = arm_prog.get("pieces") or []
    n_last = len(pieces) - 1
    out = []
    for L in legs:
        if str(L.get("kind", "transit")) != "transit":
            continue
        i0, i1 = _window(t, float(L["t0"]), float(L["t1"]))
        try:
            m = leg_metrics(q[i0:i1 + 1], X[i0:i1 + 1],
                            float(L["t1"]) - float(L["t0"]))
        except ValueError:
            continue
        c = classify(m)
        seg = int(L.get("seg", -1))
        row = dict(stage=int(stage), arm=int(arm_prog["arm"]), seg=seg,
                   home=(seg == n_last), **m, **c)
        sp = _sheet_split(pieces, seg)
        if sp:
            row.update(sp)
        out.append(row)
    return out


def anatomy(prog: dict, h_inv: float, stages=None) -> dict:
    """Pen-up anatomy of a whole staged programme. -> dict."""
    from aris_sixarm.fleet import FLEET
    rows: list[dict] = []
    stage_s: dict[int, float] = {}
    for st in prog.get("stages", []):
        s = int(st.get("stage", 0))
        if stages is not None and s not in stages:
            continue
        stage_s[s] = float(st.get("duration_s", 0.0) or 0.0)
        for aid, a in (st.get("arms") or {}).items():
            rows.extend(arm_legs(a, FLEET[int(aid)], h_inv, s))
    return dict(legs=rows, stage_s=stage_s, h_inv=float(h_inv),
                rule=dict(RATE_REF=RATE_REF, FLIP_RAD=FLIP_RAD, TALL_M=TALL_M,
                          HOVER_M=HOVER_M, ZTRAV_REF=ZTRAV_REF,
                          WANDER_RAD=WANDER_RAD, WANDER_RATIO=WANDER_RATIO))


# ---------------------------------------------------------------------------
# 3.  the report
# ---------------------------------------------------------------------------
def _tally(rows: list[dict]) -> dict:
    d = {k: 0.0 for k in CLASSES}
    n = {k: 0 for k in CLASSES}
    for r in rows:
        d[r["kind"]] += r["dur_s"]
        n[r["kind"]] += 1
    d["total"] = sum(d[k] for k in CLASSES)
    d["n"] = n
    return d


def report(res: dict, prog: dict, show_legs=False) -> None:
    rows = res["legs"]
    if not rows:
        print("no pen-up legs found")
        return
    dur = {(int(st.get("stage", 0)), int(aid)): float(a.get("duration_s") or 0)
           for st in prog.get("stages", []) for aid, a in
           (st.get("arms") or {}).items()}
    ink = {(int(st.get("stage", 0)), int(aid)): float(a.get("ink_m") or 0)
           for st in prog.get("stages", []) for aid, a in
           (st.get("arms") or {}).items()}
    if show_legs:
        print(f"{'st':>2} {'arm':>3} {'seg':>3} {'kind':>6} {'s':>6} "
              f"{'dq':>7} {'net':>6} {'hop':>6} {'ztrav':>6} {'zmax':>6} "
              f"{'flip':>6} {'lift':>6} {'trav':>6} {'lower':>6}")
        for r in sorted(rows, key=lambda r: (r["stage"], r["arm"], r["seg"])):
            print(f"{r['stage']:2d} {r['arm']:3d} {r['seg']:3d} "
                  f"{r['kind']:>6} {r['dur_s']:6.2f} {r['dq_rad']:7.2f} "
                  f"{r['net_rad']:6.2f} {r['hop_m']:6.3f} {r['ztrav_m']:6.3f} "
                  f"{r['zmax_m']:6.3f} {r['flip_rad']:6.2f} "
                  f"{r.get('lift_rad', float('nan')):6.2f} "
                  f"{r.get('travel_rad', float('nan')):6.2f} "
                  f"{r.get('lower_rad', float('nan')):6.2f}"
                  + ("  [home]" if r["home"] else ""))
        print()
    keys = sorted({(r["stage"], r["arm"]) for r in rows})
    print(f"{'st':>2} {'arm':>3} {'ink_m':>6} {'stage_s':>8} {'penup_s':>8} "
          f"{'pu%':>5} | {'flip_s':>7} {'tall_s':>7} {'honest_s':>8} "
          f"| {'nflip':>5} {'ntall':>5} {'nhon':>5} {'home_s':>7}")
    for k in keys:
        sub = [r for r in rows if (r["stage"], r["arm"]) == k]
        t = _tally(sub)
        d = dur.get(k, 0.0)
        home_s = sum(r["dur_s"] for r in sub if r["home"])
        print(f"{k[0]:2d} {k[1]:3d} {ink.get(k, 0.0):6.3f} {d:8.2f} "
              f"{t['total']:8.2f} {100 * t['total'] / max(d, 1e-9):5.1f} | "
              f"{t['flip']:7.2f} {t['tall']:7.2f} {t['honest']:8.2f} | "
              f"{t['n']['flip']:5d} {t['n']['tall']:5d} "
              f"{t['n']['honest']:5d} {home_s:7.2f}")
    t = _tally(rows)
    tot = t["total"]
    print(f"{'ALL':>6} {'':>6} {'':>8} {tot:8.2f} {'':>5} | "
          f"{t['flip']:7.2f} {t['tall']:7.2f} {t['honest']:8.2f} | "
          f"{t['n']['flip']:5d} {t['n']['tall']:5d} {t['n']['honest']:5d}")
    if tot > 0:
        print(f"fleet pen-up {tot:.1f} s: flip {100 * t['flip'] / tot:.1f} %, "
              f"tall/wander {100 * t['tall'] / tot:.1f} %, "
              f"honest {100 * t['honest'] / tot:.1f} %")
    fl = [r for r in rows if r["kind"] == "flip" and "lift_rad" in r]
    if fl:
        lift = sum(r["lift_rad"] + r["lower_rad"] for r in fl)
        trav = sum(r["travel_rad"] for r in fl)
        print(f"flip cause: hover off-sheet (lift+lower) {lift:.1f} rad, "
              f"piece-to-piece (travel) {trav:.1f} rad")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("programme", help="a staged *_program.json")
    ap.add_argument("--h-inv", type=float, default=0.970)
    ap.add_argument("--stages", default=None, help="e.g. '0' or '0,1'")
    ap.add_argument("--legs", action="store_true", help="per-leg table")
    ap.add_argument("--json", default=None, help="write the rows here")
    a = ap.parse_args(argv)
    prog = json.loads(Path(a.programme).read_text())
    stages = (None if a.stages is None
              else {int(x) for x in a.stages.replace(" ", "").split(",") if x})
    res = anatomy(prog, a.h_inv, stages)
    report(res, prog, show_legs=a.legs)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
