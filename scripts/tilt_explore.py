#!/usr/bin/env python3
"""Evidence for docs/TILT_EXPLORATION.md: what pen tilt buys, and what it costs.

    ARIS_RIG=sixarm     python3 scripts/tilt_explore.py strokes
    ARIS_RIG=final6_opt python3 scripts/tilt_explore.py donut
    ARIS_RIG=final6_opt python3 scripts/tilt_explore.py atlas
    ARIS_RIG=final6_opt python3 scripts/tilt_explore.py runtime

Every part writes out/tilt_explore_<part>.json so the document quotes numbers
that were measured rather than remembered.

WHICH RIG.  The two standard test strokes (`scripts/demo_pwl.demo_strokes`)
are constructed around FLEET[31], and every published number for them — 1.561
m, 2 knots, sigma 0.196 — was earned on the LEGACY six-arm rig, where that
placement is on the paper.  On `final6_opt` the same construction runs off the
1.8034 m canvas and refuses before it plans, so `strokes` runs on `sixarm` to
stay comparable with the record, and the donut / atlas parts run on
`final6_opt`, which is where arm 2's comfort annulus lives.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from aris_sixarm import fleet, letters, planner, stroke_api, tilt, writing  # noqa: E402
from aris_sixarm.metrics import GATE_MARGIN, GATE_SIGMA  # noqa: E402

OUT = ROOT / "out"
TILTS = (0, 15, 30)
PITCH = 7.5          # deg: fixed hex pitch, so tilt<=30 is a SUPERSET of <=15


def _row(r, dt=None):
    v = r.get("validation") or {}
    return dict(status=r["status"], reason=r.get("reason", ""),
                ms=None if dt is None else round(dt * 1000, 1),
                n_knots=r.get("n_knots"), min_sigma=r.get("min_sigma"),
                min_margin=r.get("min_margin"), travel=r.get("sum_travel"),
                max_lean_deg=r.get("max_lean_deg"), n_ik=r.get("n_ik"),
                jump=r.get("jump_search"), tilt_used=r.get("tilt_used"),
                collar_steps=r.get("collar_steps"), stage=r.get("stage"),
                t_lattice=r.get("t_lattice"), t_dp=r.get("t_dp"),
                total_time=r.get("total_time"), valid=v.get("ok"),
                arc_len=r.get("arc_len"), s_reach=r.get("s_reach"))


# --------------------------------------------------------------------------
def demo_strokes():
    """The two standard test strokes, exactly as scripts/demo_pwl.py builds them."""
    spec = fleet.FLEET[31]
    bx, by = spec.xy
    th = np.linspace(-0.6 * np.pi, 1.05 * np.pi, 400)
    arc = planner.clip_to_sheet(np.column_stack([bx + 0.66 * np.cos(th),
                                                 by + 0.66 * np.sin(th)]),
                                verbose=False)
    bowl = letters.place("R", (1.30052, 1.14999), letters.DEFAULT_HEIGHT,
                         letters.DEFAULT_ASPECT)[1]
    return spec, [("A_rim_arc", arc, 0.012), ("R_bowl", bowl, writing.DS)]


def part_strokes():
    spec, strokes = demo_strokes()
    res = {}
    for name, poly, ds in strokes:
        for obj in ("maximin_sigma", "min_travel"):
            for tm in TILTS:
                o = dict(ds_lattice=ds, objective_tilt=obj)
                t0 = time.perf_counter()
                r = tilt.plan_oracle(poly, spec, tm, pitch_deg=PITCH, opts=o)
                res[f"{name}|{obj}|{tm}"] = _row(r, time.perf_counter() - t0)
                print(f"{name:10s} {obj:13s} tilt{tm:3d}: "
                      f"{res[f'{name}|{obj}|{tm}']}")
    return res


# --------------------------------------------------------------------------
def shoulder_xy(spec):
    """Where the arm's J1/J2 shoulder projects onto the paper."""
    from aris_sixarm.frames import fk
    Twb = spec.T_world_base()
    _, p = fk(spec.q_seed)
    pw = p @ Twb[:3, :3].T + Twb[:3, 3]
    return float(pw[1][0]), float(pw[1][1])


def donut_strokes(cx, cy):
    """Four strokes crossing r in [0.10, 0.24] of a shoulder projection.

    Two radial (in and across), one tangential arc that stays INSIDE the
    annulus for its whole length, one chord that clips it.  The tangential one
    is the hard case: a radial stroke only has to survive the donut for a few
    centimetres, the arc never leaves it.
    """
    out = []
    for name, ang in (("D1_radial_180", np.pi), ("D2_radial_225", 1.25 * np.pi)):
        u = np.array([np.cos(ang), np.sin(ang)])
        t = np.linspace(0.06, 0.30, 60)[:, None]
        out.append((name, np.array([cx, cy]) + t * u))
    th = np.linspace(np.deg2rad(150), np.deg2rad(260), 90)
    out.append(("D3_arc_r17", np.column_stack([cx + 0.17 * np.cos(th),
                                               cy + 0.17 * np.sin(th)])))
    th = np.linspace(np.deg2rad(170), np.deg2rad(240), 70)
    out.append(("D4_arc_r13", np.column_stack([cx + 0.13 * np.cos(th),
                                               cy + 0.13 * np.sin(th)])))
    return out


def part_donut():
    spec = fleet.FLEET[2]
    cx, cy = shoulder_xy(spec)
    print(f"arm 2 shoulder projection: ({cx:.4f}, {cy:.4f})")
    res = {"_shoulder": [cx, cy]}
    gates = (("permissive", 0.15, 0.10), ("strict", GATE_MARGIN, GATE_SIGMA))
    for name, poly in donut_strokes(cx, cy):
        r = np.hypot(poly[:, 0] - cx, poly[:, 1] - cy)
        print(f"\n=== {name}  L={stroke_api.polyline_length(poly):.3f} m  "
              f"r in [{r.min():.3f}, {r.max():.3f}] ===")
        for gname, mg, sg in gates:
            for tm in TILTS:
                o = dict(ds_lattice=0.008, margin_gate=mg, sigma_gate=sg,
                         objective_tilt="maximin_sigma")
                t0 = time.perf_counter()
                rr = tilt.plan_oracle(poly, spec, tm, pitch_deg=PITCH, opts=o)
                key = f"{name}|{gname}|{tm}"
                res[key] = _row(rr, time.perf_counter() - t0)
                res[key]["r_min"], res[key]["r_max"] = float(r.min()), float(r.max())
                d = res[key]
                print(f"  {gname:10s} tilt{tm:3d}: {d['status']:5s} "
                      f"knots={d['n_knots']} mar={d['min_margin']} "
                      f"sig={d['min_sigma']} lean={d['max_lean_deg']} "
                      f"s_reach={d['s_reach']} {d['ms']}ms")
    return res


# --------------------------------------------------------------------------
def part_runtime():
    """What the ADAPTIVE planner costs, on strokes that need it and that do not."""
    spec = fleet.FLEET[2]
    cx, cy = shoulder_xy(spec)
    res = {}
    cases = donut_strokes(cx, cy)
    for name, poly in cases:
        for tm in (0, 15):
            o = dict(ds_lattice=0.008, margin_gate=GATE_MARGIN,
                     sigma_gate=GATE_SIGMA)
            times = []
            for rep in range(3):
                t0 = time.perf_counter()
                r = tilt.plan_adaptive(poly, spec, tm, pitch_deg=PITCH, opts=o)
                times.append(time.perf_counter() - t0)
            row = _row(r, min(times))
            row["ms_median"] = round(float(np.median(times)) * 1000, 1)
            res[f"{name}|strict|{tm}"] = row
            print(f"{name:14s} tilt{tm:3d} strict: {row['status']:5s} "
                  f"used={row['tilt_used']} collar={row['collar_steps']} "
                  f"stage={row['stage']} nIK={row['n_ik']} "
                  f"{row['ms']}ms (med {row['ms_median']})")
    return res


# --------------------------------------------------------------------------
def part_atlas(grid=0.02, r_max=0.60):
    """Mini-atlas: the donut plus a rim band, with TILT IN THE FIBER.

    THIS IS NOT `atlas.py`'s TILT AND THE DIFFERENCE IS THE WHOLE POINT.
    `atlas.solve_cell` tries the perpendicular candidates first and RETURNS on
    the first set that yields any solution, so its cone is a reach rescue: a
    cell that already has a perpendicular solution never sees a tilted one, and
    its margin is whatever the perpendicular posture happened to give.  (Every
    donut cell in out/atlas_final6_opt is recorded at tilt_deg = 0 for exactly
    that reason.)  Here the cone is part of the search: every (tilt, q7,
    branch) is scored and the cell keeps the BEST margin, so the question asked
    is "how comfortable can this cell be made", not "is it reachable at all".
    """
    from aris_sixarm import ik, rig_final
    from aris_sixarm.frames import FR3_MAX, FR3_MIN, PEN_EXT
    spec = fleet.FLEET[2]
    cx, cy = shoulder_xy(spec)
    Twb = spec.T_world_base()
    Twb_inv = np.linalg.inv(Twb)
    boxes = spec.static_obstacles()
    sheet = fleet.sheet_for(spec)
    q7s = ik.Q7_GRID
    xs = np.arange(0.02, sheet[0] - 0.02, grid)
    ys = np.arange(0.02, sheet[1] - 0.02, grid)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    P = np.column_stack([X.ravel(), Y.ravel()])
    r = np.hypot(P[:, 0] - cx, P[:, 1] - cy)
    P = P[r <= r_max]
    r = r[r <= r_max]
    print(f"arm 2 shoulder ({cx:.4f}, {cy:.4f}); {len(P)} cells at {grid} m "
          f"within r <= {r_max}")

    res = {"_shoulder": [cx, cy], "_n_cells": int(len(P)), "_grid": grid}
    best = {}
    for tm in TILTS:
        disc = tilt.hex_disc(tm, pitch_deg=PITCH)
        T = disc["tilt"]
        bm = np.full(len(P), -1.0)
        bs = np.full(len(P), -1.0)
        bl = np.zeros(len(P))
        t0 = time.perf_counter()
        for ti, tv in enumerate(T):
            poses = tilt.pen_poses(P, tv, Twb_inv, PEN_EXT)
            flat = np.repeat(ik._flat16(poses), len(q7s), axis=0)
            Q, ok = ik.solve_batch(flat, np.tile(q7s, len(P)), spec.q_seed)
            cell = np.repeat(np.arange(len(P)), len(q7s) * 4)
            Qf = Q.reshape(-1, 7)
            okf = ok.reshape(-1)
            idx = np.flatnonzero(okf)
            if not len(idx):
                continue
            q = Qf[idx]
            c = cell[idx]
            m = np.min(np.minimum(q - FR3_MIN, FR3_MAX - q), axis=1)
            Tm, pts = ik.fk_batch(q)
            pw = pts @ Twb[:3, :3].T + Twb[:3, 3]
            k = pw[:, 1:, 2].min(axis=1) >= 0.02
            q, c, m, Tm, pw = q[k], c[k], m[k], Tm[k], pw[k]
            if not len(q):
                continue
            sg = np.linalg.svd(ik.tip_jacobian_batch(q, pen_ext=PEN_EXT),
                               compute_uv=False)[:, -1]
            tip = Tm[:, :3, 3] + Tm[:, :3, :3] @ np.array([0.0, 0.0, PEN_EXT])
            tw = tip @ Twb[:3, :3].T + Twb[:3, 3]
            P10 = np.concatenate([pw, tw[:, None, :]], axis=1)
            k = tilt._frame_clear(P10, boxes)
            q, c, m, sg = q[k], c[k], m[k], sg[k]
            # a cell keeps the posture with the best margin that also controls
            k = sg >= GATE_SIGMA
            c2, m2 = c[k], m[k]
            if len(c2):
                np.maximum.at(bm, c2, m2)
            np.maximum.at(bs, c, sg)
            lean = np.rad2deg(np.linalg.norm(tv))
            if len(c2):
                bl[c2] = np.where(m2 > bm[c2] - 1e-12, lean, bl[c2])
        best[tm] = (bm, bs, bl)
        print(f"  tilt<={tm:2d}: {time.perf_counter() - t0:.1f} s")

    ann = dict(donut=(0.10, 0.24), inner=(0.0, 0.10), band=(0.24, 0.45),
               rim=(0.45, r_max), all=(0.0, r_max))
    for gname, mg in (("permissive", 0.15), ("strict", GATE_MARGIN)):
        for aname, (lo, hi) in ann.items():
            sel = (r >= lo) & (r < hi)
            row = {}
            for tm in TILTS:
                bm, bs, bl = best[tm]
                go = sel & (bm >= mg)
                row[str(tm)] = dict(cells=int(sel.sum()), go=int(go.sum()),
                                    frac=float(go.sum() / max(sel.sum(), 1)),
                                    med_lean=float(np.median(bl[go]))
                                    if go.any() else 0.0)
            res[f"{gname}|{aname}"] = row
            g = " ".join(f"tilt{t}={row[str(t)]['go']:4d}"
                         f"({100 * row[str(t)]['frac']:5.1f}%)" for t in TILTS)
            print(f"  {gname:10s} r[{lo:.2f},{hi:.2f}) n={int(sel.sum()):4d}  {g}")
    return res


PARTS = dict(strokes=part_strokes, donut=part_donut, runtime=part_runtime,
             atlas=part_atlas)

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "strokes"
    out = PARTS[which]()
    OUT.mkdir(exist_ok=True)
    p = OUT / f"tilt_explore_{which}.json"
    p.write_text(json.dumps(out, indent=1, default=float))
    print(f"\nwrote {p}")
