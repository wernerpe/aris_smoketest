#!/usr/bin/env python3
"""Allocate -> freeze per-arm timelines -> conduct -> validate -> animation payload.

    python3 scripts/csail_schedule.py --arms all --tag _6arm \
        --placement out/csail_placement_6arm.json

Runs under the SYSTEM python (it needs the batch IK entry points the station
venv's older wheel does not carry).  Everything the drake demo needs comes out
as one npz, so `csail_drawing_demo.py` never has to plan anything.

The order of operations is the point:

  1. the allocator's certified per-arm programmes (same call as csail_allocate);
  2. `writing.arm_program` freezes each arm's path — segments in the
     allocator's order, hover transits between them, entry and exit lifts;
  3. `coordination.coordinate` leaves those paths alone and only stretches the
     clock: pauses, priority = busiest arm first;
  4. `scene_check.check_timeline` re-derives the whole merged timeline from
     scratch and has a veto.  The payload is not written unless it passes.

Writes out/csail_schedule<tag>.npz and (with --final) the end-state still.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from aris_sixarm import coordination, scene_check, trace, writing   # noqa: E402
from aris_sixarm.fleet import FLEET, SHEET, H_INV_DEFAULT           # noqa: E402
from csail_allocate import add_args, run_allocation, final_png      # noqa: E402

INK = {"grey": "#%02x%02x%02x" % trace.GREY_RGB,
       "orange": "#%02x%02x%02x" % trace.ORANGE_RGB}


def build(a):
    res, strokes, info = run_allocation(a, verbose=False)
    from aris_sixarm import allocate
    print()
    for line in allocate.report(res, strokes):
        print(line)

    dt = 1.0 / (a.fps * a.substeps)
    print(f"\nfreezing per-arm timelines (draw {a.draw_speed} m/s, "
          f"transit {a.transit_speed} m/s, clock {dt:.4f} s)...")
    progs, samp, worst_tip = {}, {}, 0.0
    for aid in res["arms"]:
        p = writing.arm_program(FLEET[aid], res["programs"][aid],
                                draw_speed=a.draw_speed,
                                transit_speed=a.transit_speed, qd_frac=a.qd_frac,
                                h_inv=H_INV_DEFAULT, verbose=a.verbose)
        progs[aid] = p
        samp[aid] = writing.uniform_samples(p, dt)
        worst_tip = max(worst_tip, p["dense_tip_err"])
        print(f"  arm {aid:>2}: {len(res['programs'][aid]):>2} segments, "
              f"{p['draw_len']:.2f} m drawn + {p['transit_len']:.2f} m transit, "
              f"{p['duration']:.1f} s nominal, {samp[aid]['n']} steps, "
              f"densify tip_err {p['dense_tip_err']:.2e} m")
    for aid in FLEET:                       # arms not in this run still take up room
        if aid not in samp:
            progs[aid] = writing.arm_program(FLEET[aid], [], verbose=False)
            samp[aid] = writing.uniform_samples(progs[aid], dt)

    print("\ncapsule model + collision images...")
    t0 = time.time()
    paths = {aid: coordination.ArmPath(aid, samp[aid]["q"], dt) for aid in FLEET}
    for aid, p in paths.items():
        print(f"  arm {aid:>2}: {p.n:>5} steps, {p.motion:6.2f} m of chain motion, "
              f"max step {float(p.step.max()) * 1000:5.1f} mm"
              + ("" if p.moves else "   (static)"))
    sch = coordination.coordinate(paths, safety=a.safety, calib=a.calib,
                                  verbose=True)
    print(f"  conducted in {time.time() - t0:.1f} s")
    for line in coordination.report(sch, paths):
        print("  " + line)

    M = sch["M"]
    qtraj = {aid: samp[aid]["q"][np.clip(sch["progress"][aid][:M], 0,
                                         samp[aid]["n"] - 1)] for aid in FLEET}

    print()
    t0 = time.time()
    rep = scene_check.check_timeline(
        qtraj, dt, sch["margin"], programs=res["programs"],
        progress={k: v[:M] for k, v in sch["progress"].items()}, sub=a.subcheck)
    print(f"  checked in {time.time() - t0:.1f} s")
    if not rep["ok"]:
        raise SystemExit("scene_check REFUSED the timeline; nothing rendered")
    return res, strokes, info, progs, samp, paths, sch, qtraj, rep, dt


def payload(res, progs, samp, sch, qtraj, rep, dt, a, out):
    """Everything the drake demo replays, as one npz."""
    stride = int(a.substeps)
    M = sch["M"]
    idx = np.arange(0, M, stride)
    nF = len(idx)
    d = dict(fps=np.float64(a.fps), dt=np.float64(dt), n_frames=np.int64(nF),
             stride=np.int64(stride),
             duration=np.float64((nF - 1) / a.fps),
             margin=np.float64(sch["margin"]),
             min_clearance=np.float64(rep["min_clearance"]),
             pause_total=np.float64(sch["pause_total"]),
             arms=np.array(sorted(FLEET), np.int64),
             drawing_arms=np.array(sorted(res["arms"]), np.int64),
             sheet=np.array(SHEET, float))
    ink_t, ink_arm, ink_xyz, ink_off = [], [], [], [0]
    for aid in FLEET:
        P = np.clip(sch["progress"][aid][:M], 0, samp[aid]["n"] - 1)
        d[f"q_{aid}"] = qtraj[aid][idx].astype(np.float32)
        d[f"prog_{aid}"] = P[idx].astype(np.int64)
        d[f"seg_{aid}"] = samp[aid]["seg"][P[idx]].astype(np.int64)
        d[f"u_{aid}"] = samp[aid]["u"][P[idx]].astype(np.float64)
        d[f"pen_{aid}"] = np.array(INK.get(res["colors"].get(aid, "grey"), "#666665"))
        segs = res["programs"].get(aid, [])
        pts = [np.asarray(s["plan"]["pts"], float) for s in segs]
        off = np.cumsum([0] + [len(p) for p in pts])
        d[f"segpts_{aid}"] = (np.vstack(pts) if pts else np.zeros((0, 2)))
        d[f"segoff_{aid}"] = off.astype(np.int64)
        # ink: nominal reveal time -> nominal index -> the scheduled instant the
        # arm's progress first reaches it
        for t_vis, xyz in progs[aid]["ink"]:
            p_vis = int(round(t_vis / dt))
            hit = np.flatnonzero(P >= min(p_vis, samp[aid]["n"] - 1))
            ink_t.append(float((hit[0] if len(hit) else M - 1) * dt))
            ink_arm.append(aid)
            ink_xyz.append(xyz)
            ink_off.append(ink_off[-1] + len(xyz))
    d["ink_t"] = np.array(ink_t, float)
    d["ink_arm"] = np.array(ink_arm, np.int64)
    d["ink_off"] = np.array(ink_off, np.int64)
    d["ink_xyz"] = (np.vstack(ink_xyz) if ink_xyz else np.zeros((0, 3)))
    np.savez_compressed(out, **d)
    return nF, len(ink_t)


def main(argv=None):
    ap = add_args(argparse.ArgumentParser())
    ap.add_argument("--tag", default="_6arm")
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--substeps", type=int, default=2,
                    help="coordination clock steps per animation frame")
    ap.add_argument("--subcheck", type=int, default=2,
                    help="scene_check re-sampling factor")
    ap.add_argument("--draw-speed", type=float, default=writing.DRAW_SPEED_FLEET)
    ap.add_argument("--transit-speed", type=float, default=writing.TRANSIT_SPEED)
    ap.add_argument("--qd-frac", type=float, default=writing.QD_FRAC,
                    help="fraction of the FR3 joint-velocity limit any move may use")
    ap.add_argument("--safety", type=float, default=coordination.SAFETY_M)
    ap.add_argument("--calib", type=float, default=coordination.CALIB_M)
    ap.add_argument("--final", default=None)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)

    t0 = time.time()
    res, strokes, info, progs, samp, paths, sch, qtraj, rep, dt = build(a)
    path = out / f"csail_schedule{a.tag}.npz"
    nF, nInk = payload(res, progs, samp, sch, qtraj, rep, dt, a, path)
    if a.final:
        final_png(res, strokes, a.final)
    print(f"\nwrote {path} ({path.stat().st_size / 1e6:.1f} MB): {nF} frames @ "
          f"{a.fps:g} fps = {(nF - 1) / a.fps:.1f} s, {nInk} ink chunks"
          + (f"; {a.final}" if a.final else ""))
    print(f"SCHEDULE WALL CLOCK {time.time() - t0:.1f} s")
    summary = dict(
        frames=nF, fps=a.fps, duration=(nF - 1) / a.fps, ink_chunks=nInk,
        draw_speed=a.draw_speed, transit_speed=a.transit_speed,
        drawn_m=res["drawn_len"], traced_m=res["total_len"],
        dropped_m=res["dropped_len"],
        coverage=res["drawn_len"] / max(res["total_len"], 1e-9),
        pause_total=sch["pause_total"],
        pauses={str(k): float(v) for k, v in sch["pauses"].items()},
        priority=[int(x) for x in sch["order"]],
        arm_metres={str(a): float(sum(s["length"] for s in res["programs"][a]))
                    for a in res["arms"]},
        arm_segments={str(a): len(res["programs"][a]) for a in res["arms"]},
        margin=sch["margin"], min_clearance=rep["min_clearance"],
        per_pair=rep["per_pair"], densify_tip_err=max(p["dense_tip_err"]
                                                      for p in progs.values()),
        logo=dict(w=info["logo_w"], h=info["logo_h"],
                  center=[float(x) for x in info["center"]]),
        colors={str(k): v for k, v in res["colors"].items()})
    (out / f"csail_schedule{a.tag}.json").write_text(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    main()
