#!/usr/bin/env python3
"""Allocate -> freeze per-arm timelines -> conduct -> validate -> animation payload.

    python3 scripts/csail_schedule.py --arms all --tag _6arm \
        --placement out/csail_placement_6arm.json

Runs under the SYSTEM python (it needs the batch IK entry points the station
venv's older wheel does not carry).  Everything the drake demo needs comes out
as one npz, so `csail_drawing_demo.py` never has to plan anything.

The order of operations is the point:

  1. the allocator's certified per-arm programmes (same call as csail_allocate),
     each already sequenced by `sequence.py`: the order AND the direction of
     every segment chosen to minimise the arm's real pen-up time;
  2. `writing.arm_program` freezes each arm's path — segments in that order,
     hover transits between them, entry and exit lifts — and the transit clock
     it lays down is checked against the cost the sequencer minimised;
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


def build_phase(a, res, dt, pens):
    """One drawing phase: freeze, conduct, and have the checker sign it off."""
    print(f"\nfreezing per-arm timelines for {res['name']} "
          f"(draw {a.draw_speed} m/s, transit {a.transit_speed} m/s, "
          f"clock {dt:.4f} s)...")
    progs, samp, worst_tip = {}, {}, 0.0
    for aid in res["arms"]:
        p = writing.arm_program(FLEET[aid], res["programs"][aid],
                                draw_speed=a.draw_speed,
                                transit_speed=a.transit_speed, qd_frac=a.qd_frac,
                                h_inv=H_INV_DEFAULT, pen_ext=pens[aid],
                                verbose=a.verbose)
        progs[aid] = p
        samp[aid] = writing.uniform_samples(p, dt)
        worst_tip = max(worst_tip, p["dense_tip_err"])
        q = res["sequence"][aid]
        print(f"  arm {aid:>2} ({1000 * pens[aid]:>3.0f} mm pen): "
              f"{len(res['programs'][aid]):>2} segments, "
              f"{p['draw_len']:.2f} m drawn + {p['transit_len']:.2f} m transit, "
              f"{p['duration']:.1f} s nominal "
              f"({p['draw_s']:.1f} s drawing + {p['transit_s']:.1f} s pen-up; "
              f"sequencer said {q['cost']:.1f} s, nearest-xy would be "
              f"{q['baseline_cost']:.1f} s), {samp[aid]['n']} steps, "
              f"densify tip_err {p['dense_tip_err']:.2e} m")
    # THE SEQUENCER'S MODEL IS THE TIMELINE'S CLOCK, or it optimised a fiction.
    # Both sides compute lift + travel + lower from the same hover poses with
    # the same joint-velocity cap AND THE SAME PEN, so they must agree to the
    # float — this is the check that keeps them agreeing when somebody edits one
    # of them, and the one that would catch a per-arm pen that only got as far
    # as the planner.
    drift = {aid: abs(progs[aid]["transit_s"] - res["sequence"][aid]["cost"])
             for aid in res["arms"]}
    worst = max(drift.values()) if drift else 0.0
    print(f"  sequencer cost model vs frozen timeline: worst disagreement "
          f"{worst:.2e} s over {len(drift)} arms")
    if worst > 1e-6:
        raise SystemExit(f"sequencer priced transits the timeline does not pay: "
                         + ", ".join(f"arm {a}: {d:.4f} s" for a, d in drift.items()
                                     if d > 1e-6))

    for aid in FLEET:                       # arms not in this run still take up room
        if aid not in samp:
            progs[aid] = writing.arm_program(FLEET[aid], [], verbose=False,
                                             pen_ext=pens.get(aid, 0.110))
            samp[aid] = writing.uniform_samples(progs[aid], dt)

    print("  capsule model + collision images...")
    t0 = time.time()
    paths = coordination.arm_paths({aid: samp[aid]["q"] for aid in FLEET}, dt,
                                   pens=pens)
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
    t0 = time.time()
    rep = scene_check.check_timeline(
        qtraj, dt, sch["margin"], programs=res["programs"], pen_ext=pens,
        progress={k: v[:M] for k, v in sch["progress"].items()}, sub=a.subcheck)
    print(f"  checked in {time.time() - t0:.1f} s")
    if not rep["ok"]:
        raise SystemExit(f"scene_check REFUSED {res['name']}; nothing rendered")
    return dict(res=res, progs=progs, samp=samp, paths=paths, sch=sch,
                qtraj=qtraj, rep=rep, M=M, worst_tip=worst_tip)


def check_parked(pens, dt, margin, sub=2):
    """The pen-swap pause is a POSE, and a pose can be unsafe too.

    Every arm holds `q_seed` while a human walks in and changes the pens, so the
    same independent checker is asked about that configuration rather than it
    being assumed clear because nothing is moving.
    """
    q = {aid: np.repeat(np.asarray(FLEET[aid].q_seed, float)[None, :], 2, axis=0)
         for aid in FLEET}
    return scene_check.check_timeline(q, dt, margin, pen_ext=pens, sub=sub,
                                      verbose=False)


def build(a):
    phases, strokes, info = run_allocation(a, verbose=False)
    from aris_sixarm import allocate
    from csail_allocate import totals
    for ph in phases:
        print(f"\n=== {ph['name']} ===")
        for line in allocate.report(ph, ph["strokes"]):
            print(line)
    T = totals(phases)
    print(f"\nALL PHASES: {T['traced']:.4f} m traced, {T['dropped']:.4f} m left "
          f"empty -> COVERAGE {100 * T['covered']:.4f} %")
    want = 1.0 if a.require_full else float(a.min_coverage)
    if T["covered"] < want - 1e-12:
        raise SystemExit(
            f"coverage is {100 * T['covered']:.4f} %, below the "
            f"{100 * want:.2f} % this run was asked for; {T['dropped']:.4f} m "
            f"in {T['n_dropped']} spans left empty (lower --min-coverage to "
            "render it anyway)")

    dt = 1.0 / (a.fps * a.substeps)
    pens = {aid: next((p["pens"][aid] for p in phases if aid in p["pens"]), 0.110)
            for aid in FLEET}
    built = [build_phase(a, ph, dt, pens) for ph in phases]

    if len(built) > 1:
        rep = check_parked(pens, dt, built[0]["sch"]["margin"], a.subcheck)
        print(f"\npen-swap pause: all six arms parked at q_seed, "
              f"min clearance {1000 * rep['min_clearance']:.1f} mm "
              f"(margin {1000 * built[0]['sch']['margin']:.0f} mm) -> "
              f"{'PASS' if rep['ok'] else 'FAIL'}")
        if not rep["ok"]:
            raise SystemExit("the parked pen-swap pose is not clear")
    return phases, strokes, info, built, dt, pens


def payload(built, dt, pens, a, out):
    """Everything the drake demo replays, as one npz.

    The phases are laid END TO END on one clock with `--pause` seconds of every
    arm parked at `q_seed` between them, because that is what happens: the
    fleet stops, a human swaps the pens, and the fleet starts again.  The pause
    is real time in the animation rather than a cut, so the run reads as one
    piece drawn in two passes and not as two videos.
    """
    stride = int(a.substeps)
    n_pause = int(round(a.pause * a.fps)) * stride if len(built) > 1 else 0
    starts, M_tot = [], 0
    for k, B in enumerate(built):
        starts.append(M_tot)
        M_tot += B["M"] + (n_pause if k + 1 < len(built) else 0)

    drawing = sorted({x for B in built for x in B["res"]["arms"]
                      if B["res"]["programs"][x]})
    d = dict(fps=np.float64(a.fps), dt=np.float64(dt),
             stride=np.int64(stride), n_phases=np.int64(len(built)),
             pause_s=np.float64(a.pause if len(built) > 1 else 0.0),
             margin=np.float64(built[0]["sch"]["margin"]),
             min_clearance=np.float64(min(B["rep"]["min_clearance"] for B in built)),
             pause_total=np.float64(sum(B["sch"]["pause_total"] for B in built)),
             arms=np.array(sorted(FLEET), np.int64),
             drawing_arms=np.array(drawing, np.int64),
             pen_ext=np.array([pens[x] for x in sorted(FLEET)], float),
             sheet=np.array(SHEET, float))

    idx = np.arange(0, M_tot, stride)
    nF = len(idx)
    d.update(n_frames=np.int64(nF), duration=np.float64((nF - 1) / a.fps))
    # which phase each ANIMATION FRAME belongs to (-1 during the pen swap)
    ph_of = np.full(M_tot, -1, np.int64)
    for k, B in enumerate(built):
        ph_of[starts[k]:starts[k] + B["M"]] = k
    d["phase"] = ph_of[idx]
    d["phase_start_s"] = np.array([s * dt for s in starts], float)
    d["phase_ink"] = np.array([B["res"]["ink"] or "grey" for B in built])

    ink_t, ink_arm, ink_xyz, ink_off, ink_col = [], [], [], [0], []
    for aid in sorted(FLEET):
        Q = np.zeros((M_tot, 7))
        SEG = np.full(M_tot, -1, np.int64)
        U = np.zeros(M_tot)
        allpts, alloff, base = [], [0], 0
        for k, B in enumerate(built):
            res, samp, sch = B["res"], B["samp"], B["sch"]
            M, s0 = B["M"], starts[k]
            P = np.clip(sch["progress"][aid][:M], 0, samp[aid]["n"] - 1)
            Q[s0:s0 + M] = B["qtraj"][aid]
            # segment indices are made global across phases, so `segpts_<arm>`
            # stays one flat array the demo can index without knowing the phase
            SEG[s0:s0 + M] = np.where(samp[aid]["seg"][P] < 0, -1,
                                      samp[aid]["seg"][P] + base)
            U[s0:s0 + M] = samp[aid]["u"][P]
            if k + 1 < len(built):            # hold the last pose through the swap
                Q[s0 + M:s0 + M + n_pause] = B["qtraj"][aid][-1]
            segs = res["programs"].get(aid, [])
            pts = [np.asarray(s["plan"]["pts"], float) for s in segs]
            allpts += pts
            for p in pts:
                alloff.append(alloff[-1] + len(p))
            base += len(segs)
            for t_vis, xyz in B["progs"][aid]["ink"]:
                p_vis = int(round(t_vis / dt))
                hit = np.flatnonzero(P >= min(p_vis, samp[aid]["n"] - 1))
                ink_t.append(float((s0 + (hit[0] if len(hit) else M - 1)) * dt))
                ink_arm.append(aid)
                ink_col.append(res["ink"] or res["colors"][aid])
                ink_xyz.append(xyz)
                ink_off.append(ink_off[-1] + len(xyz))
        d[f"q_{aid}"] = Q[idx].astype(np.float32)
        d[f"seg_{aid}"] = SEG[idx]
        d[f"u_{aid}"] = U[idx]
        d[f"segpts_{aid}"] = (np.vstack(allpts) if allpts else np.zeros((0, 2)))
        d[f"segoff_{aid}"] = np.array(alloff, np.int64)
    d["ink_t"] = np.array(ink_t, float)
    d["ink_arm"] = np.array(ink_arm, np.int64)
    d["ink_off"] = np.array(ink_off, np.int64)
    d["ink_xyz"] = (np.vstack(ink_xyz) if ink_xyz else np.zeros((0, 3)))
    # a chunk's colour is the colour of the PHASE it was laid in, not a property
    # of the arm: the same arm lays grey before the swap and orange after
    d["ink_hex"] = np.array([INK[c] for c in ink_col])
    np.savez_compressed(out, **d)
    return nF, len(ink_t), M_tot, n_pause


def main(argv=None):
    ap = add_args(argparse.ArgumentParser())
    ap.add_argument("--tag", default="_6arm")
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--substeps", type=int, default=2,
                    help="coordination clock steps per animation frame")
    ap.add_argument("--subcheck", type=int, default=2,
                    help="scene_check re-sampling factor")
    ap.add_argument("--draw-speed", type=float, default=writing.DRAW_SPEED_FLEET)
    # --transit-speed and --qd-frac come from add_args: the sequencer prices
    # its transits with them before this script ever freezes a timeline.
    ap.add_argument("--safety", type=float, default=coordination.SAFETY_M)
    ap.add_argument("--calib", type=float, default=coordination.CALIB_M)
    ap.add_argument("--pause", type=float, default=2.0,
                    help="seconds of every-arm-parked between two passes, "
                         "while a human swaps the pens")
    ap.add_argument("--min-coverage", type=float, default=0.0,
                    help="refuse to write a payload below this certified "
                         "coverage (fraction of traced metres)")
    ap.add_argument("--require-full", action="store_true",
                    help="shorthand for --min-coverage 1.0")
    ap.add_argument("--final", default=None)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    phases, strokes, info, built, dt, pens = build(a)
    path = out / f"csail_schedule{a.tag}.npz"
    nF, nInk, M_tot, n_pause = payload(built, dt, pens, a, path)
    if a.final:
        final_png(phases, strokes, a.final)
    print(f"\nwrote {path} ({path.stat().st_size / 1e6:.1f} MB): {nF} frames @ "
          f"{a.fps:g} fps = {(nF - 1) / a.fps:.1f} s, {nInk} ink chunks"
          + (f", including {n_pause * dt:.1f} s of pen-swap pause"
             if n_pause else "")
          + (f"; {a.final}" if a.final else ""))
    print(f"SCHEDULE WALL CLOCK {time.time() - t0:.1f} s")

    from csail_allocate import totals
    T = totals(phases)
    summary = dict(
        frames=nF, fps=a.fps, duration=(nF - 1) / a.fps, ink_chunks=nInk,
        n_phases=len(phases), two_pass=len(phases) > 1,
        pen_swap_pause_s=(n_pause * dt if n_pause else 0.0),
        draw_speed=a.draw_speed, transit_speed=a.transit_speed,
        pens_mm={str(k): round(1000 * v, 1) for k, v in sorted(pens.items())},
        traced_m=T["traced"], drawn_m=T["drawn"], dropped_m=T["dropped"],
        coverage=T["covered"], n_segments=T["n_segments"],
        margin=built[0]["sch"]["margin"],
        min_clearance=min(B["rep"]["min_clearance"] for B in built),
        densify_tip_err=max(B["worst_tip"] for B in built),
        pause_total=sum(B["sch"]["pause_total"] for B in built),
        logo=dict(w=info["logo_w"], h=info["logo_h"],
                  center=[float(x) for x in info["center"]],
                  offset=[float(x) for x in info["offset"]]),
        phases=[], sequencer=phases[0].get("sequencer", "opt"))
    for B in built:
        res, sch, rep, progs = B["res"], B["sch"], B["rep"], B["progs"]
        summary["phases"].append(dict(
            name=res["name"], ink=res["ink"],
            traced_m=res["total_len"], drawn_m=res["drawn_len"],
            dropped_m=res["dropped_len"],
            duration_s=float(sch["duration"]),
            pause_total=float(sch["pause_total"]),
            pauses={str(k): float(v) for k, v in sch["pauses"].items()},
            priority=[int(x) for x in sch["order"]],
            margin=float(sch["margin"]), min_clearance=float(rep["min_clearance"]),
            per_pair=rep["per_pair"], scene_check_ok=bool(rep["ok"]),
            n_segments_validated=int(rep["n_segments"]),
            segments_failed=int(rep["segments_failed"]),
            arm_metres={str(x): float(sum(s["length"] for s in res["programs"][x]))
                        for x in res["arms"]},
            arm_segments={str(x): len(res["programs"][x]) for x in res["arms"]},
            arm_transit_s={str(x): float(progs[x]["transit_s"]) for x in res["arms"]},
            arm_draw_s={str(x): float(progs[x]["draw_s"]) for x in res["arms"]},
            arm_nominal_s={str(x): float(progs[x]["duration"]) for x in res["arms"]},
            arm_transit_predicted_s={str(x): float(res["sequence"][x]["cost"])
                                     for x in res["arms"]},
            arm_transit_baseline_s={str(x): float(res["sequence"][x]["baseline_cost"])
                                    for x in res["arms"]},
            arm_reversed={str(x): int(res["sequence"][x]["n_reversed"])
                          for x in res["arms"]},
            arm_sequencer_method={str(x): res["sequence"][x]["method"]
                                  for x in res["arms"]},
            colors={str(k): v for k, v in res["colors"].items()}))
    # what the old single-phase readers look for, pointed at phase 1
    summary.update(pauses=summary["phases"][0]["pauses"],
                   priority=summary["phases"][0]["priority"],
                   arm_metres=summary["phases"][0]["arm_metres"],
                   arm_segments=summary["phases"][0]["arm_segments"],
                   per_pair=summary["phases"][0]["per_pair"],
                   colors=summary["phases"][0]["colors"])
    (out / f"csail_schedule{a.tag}.json").write_text(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    main()
