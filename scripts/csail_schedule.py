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
     hover transits between them, an entry lift and whatever the idle policy
     says to do at the end — and the transit clock it lays down is checked
     against the cost the sequencer minimised;
  3. `idle.conduct` chooses those paths and `coordination.coordinate` schedules
     them, leaving them alone and only stretching the clock: pauses, with the
     priority order searched for the minimum makespan.  The idle policy
     (`--idle-policy`, default `freeze`) decides where an arm STOPS, which is
     where most of the pause used to go; a refusal it cannot answer comes back
     as a tour edge and the phase is re-sequenced without it;
  4. `scene_check.check_timeline` re-derives the whole merged timeline from
     scratch and has a veto, and `scene_check.check_static` does the same for
     the pose the fleet holds while a human swaps the pens.  The payload is not
     written unless both pass.

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
from aris_sixarm import (allocate, coordination, idle, scene_check, trace,  # noqa: E402
                         writing)
from aris_sixarm.fleet import FLEET, SHEET, H_INV_DEFAULT           # noqa: E402
from csail_allocate import add_args, run_allocation, final_png      # noqa: E402

INK = {"grey": "#%02x%02x%02x" % trace.GREY_RGB,
       "orange": "#%02x%02x%02x" % trace.ORANGE_RGB}


def _balance_json(b):
    """The load-balancing pass's own record, JSON-shaped. -> dict or None.

    Kept per phase because "who drew what, and how long each arm therefore
    took" is the number the makespan is made of, and reading it back out of the
    per-arm metres afterwards loses the BEFORE half.
    """
    if not b:
        return None
    return dict(
        n_movable=int(b["n_movable"]), rounds=int(b["rounds"]),
        n_splits=int(b.get("n_splits", 0)),
        splits=[{k: (int(v) if isinstance(v, (int, np.integer)) else v)
                 for k, v in m.items()} for m in b.get("splits", [])],
        n_segments_before=int(b.get("n_segments_before", 0)),
        n_segments_after=int(b.get("n_segments_after", 0)),
        min_split_m=float(b.get("min_split_m", 0.0)),
        splice_m=float(b.get("splice_m", 0.0)),
        coverage_lost_m=float(b.get("coverage_lost_m", 0.0)),
        max_before_s=float(b["max_before"]), max_after_s=float(b["max_after"]),
        draw_speed=float(b["draw_speed"]), n_replans=int(b["n_replans"]),
        loads_before_s={str(k): float(v) for k, v in b["loads_before"].items()},
        loads_after_s={str(k): float(v) for k, v in b["loads_after"].items()},
        metres_before={str(k): float(v) for k, v in b["metres_before"].items()},
        metres_after={str(k): float(v) for k, v in b["metres_after"].items()},
        moves=[{k: (int(v) if isinstance(v, (int, np.integer)) else v)
                for k, v in m.items()} for m in b["moves"]])


def build_phase(a, res, dt, pens, q_start=None, policy=None):
    """One drawing phase: freeze, conduct under the idle policy, sign it off."""
    print(f"\nfreezing per-arm timelines for {res['name']} "
          f"(draw {a.draw_speed} m/s, transit {a.transit_speed} m/s, "
          f"clock {dt:.4f} s, idle policy {policy})...")

    def cross_check(progs):
        # THE SEQUENCER'S MODEL IS THE TIMELINE'S CLOCK, or it optimised a
        # fiction.  Both sides compute lift + travel + lower from the same hover
        # poses with the same joint-velocity cap, THE SAME PEN, THE SAME START
        # POSE and the same answer to "does this arm go home at the end" — so
        # they must agree to the float.  This is the check that keeps them
        # agreeing when somebody edits one of them; `transit_s` deliberately
        # excludes the two idle-policy additions (`taxi_s`, `retreat_s`), which
        # are seconds the conductor adds to a priced tour and not a re-pricing
        # of it.
        drift = {aid: abs(progs[aid]["transit_s"] - res["sequence"][aid]["cost"])
                 for aid in res["arms"]}
        worst = max(drift.values()) if drift else 0.0
        print(f"  sequencer cost model vs frozen timeline: worst disagreement "
              f"{worst:.2e} s over {len(drift)} arms")
        if worst > 1e-6:
            raise SystemExit("sequencer priced transits the timeline does not "
                             "pay: " + ", ".join(f"arm {x}: {d:.4f} s"
                                                 for x, d in drift.items()
                                                 if d > 1e-6))

    t0, forbid, out = time.time(), {}, None
    policy = a.idle_policy if policy is None else policy
    tries = int(a.reseq_tries)
    for attempt in range(tries + 2):
        try:
            out = idle.conduct(
                {aid: res["programs"].get(aid, []) for aid in FLEET}, pens, dt,
                q_start=q_start, policy=policy,
                retreat=not a.no_retreat, jit=not a.no_jit,
                jit_frac=a.jit_frac, draw_speed=a.draw_speed,
                transit_speed=a.transit_speed, qd_frac=a.qd_frac,
                h_inv=H_INV_DEFAULT, safety=a.safety, calib=a.calib,
                orders={x: res["sequence"][x]["order"] for x in res["arms"]},
                on_programs=cross_check, verbose=True)
            break
        except idle.Unconductable as exc:
            # A PEN-UP THE CONDUCTOR CANNOT RUN IS A TOUR EDGE, AND THE TOUR IS
            # SEARCHED.  The sequencer picks the cheapest order it can and knows
            # nothing about the other five arms; when the DP reports a progress
            # index that is impossible whatever anybody else does, the edge that
            # produced it goes on a blacklist and Held-Karp is asked again.  The
            # allocation does not move — the same arm draws the same ink, in a
            # different order.
            print(f"\n  {exc}")
            if exc.transits and attempt < tries:
                for x, edges in exc.transits.items():
                    forbid.setdefault(x, set()).update(edges)
                print(f"  re-sequencing without "
                      + ", ".join(f"arm {x}: {sorted(v)}"
                                  for x, v in forbid.items())
                      + f" (attempt {attempt + 2} of {tries + 1})")
            elif policy != idle.POLICY_HOME:
                # LAST RESORT, AND THE HIERARCHY SAYS SO.  The margin and the
                # gates are not negotiable and the makespan is only an
                # objective, so a phase that cannot be conducted at all under
                # the new policy gets conducted under the old one rather than
                # not at all.  It is a whole-phase step back — the per-arm one
                # inside `idle.conduct` has already been tried and was not
                # enough — and it is announced, not hidden.
                print(f"  !! {res['name']} cannot be conducted with arms frozen "
                      "in place; falling back to conductor v1's go-home for "
                      "this whole phase")
                policy, forbid = idle.POLICY_HOME, {}
            else:
                raise
            allocate.resequence(res, q_start=q_start,
                                return_home=policy == idle.POLICY_HOME,
                                forbid={x: sorted(v) for x, v in forbid.items()})
    progs, samp, paths, sch = out["progs"], out["samp"], out["paths"], out["sch"]
    worst_tip = max((progs[aid]["dense_tip_err"] for aid in res["arms"]),
                    default=0.0)
    for aid in res["arms"]:
        p, q = progs[aid], res["sequence"][aid]
        print(f"  arm {aid:>2} ({1000 * pens[aid]:>3.0f} mm pen): "
              f"{len(res['programs'][aid]):>2} segments, "
              f"{p['draw_len']:.2f} m drawn + {p['transit_len']:.2f} m transit, "
              f"{p['duration']:.1f} s nominal "
              f"({p['draw_s']:.1f} s drawing + {p['transit_s']:.1f} s pen-up"
              + (f" + {p['taxi_s']:.1f} s slow taxi" if p["taxi_s"] else "")
              + (f" + {p['retreat_s']:.1f} s retreat" if p["retreat_s"] else "")
              + f"; sequencer said {q['cost']:.1f} s, nearest-xy would be "
              f"{q['baseline_cost']:.1f} s), {samp[aid]['n']} steps, "
              f"densify tip_err {p['dense_tip_err']:.2e} m")
    for aid in sorted(paths):
        p = paths[aid]
        print(f"  arm {aid:>2}: {p.n:>5} steps, {p.motion:6.2f} m of chain motion, "
              f"max step {float(p.step.max()) * 1000:5.1f} mm"
              + ("" if p.moves else "   (static)"))
    if out is None:
        raise SystemExit(f"{res['name']} could not be conducted")
    print(f"  conducted in {time.time() - t0:.1f} s"
          + ("" if policy == a.idle_policy else
             f"   (idle policy fell back to {policy!r})"))
    for line in idle.report(out):
        print("  " + line)
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
        # THE POSE AN ARM STOPS IN IS A CHOICE, AND IT IS THE POLICY'S CHOICE.
        # Freeze-in-place parks an arm at the hover above its last stroke, and
        # nothing guarantees that pose has any joint-limit margin left — the
        # planner certified the STROKE, and the hover over its end is a separate
        # IK solve.  `idle.plan_retreat` only ever offers a retreat to a pose
        # that is in somebody's WAY, so a pose that is merely bad has no way of
        # being noticed until this gate looks at it.  When that is the only
        # thing wrong, the answer is conductor v1's: go home, where the pose is
        # `q_seed` and known good.  Same last resort, same reason, as the
        # `Unconductable` fallback above — the gates are not negotiable and the
        # makespan is only an objective — and it is announced, not hidden.
        frozen_only = (int(rep["frozen_failed"]) > 0
                       and float(rep["min_clearance"]) >= sch["margin"] - 1e-12
                       and bool(rep["monotone"])
                       and int(rep["segments_failed"]) == 0
                       and min(rep["joint_margin"].values()) > 0)
        if frozen_only and policy != idle.POLICY_HOME:
            bad = ", ".join(f"arm {x}: {','.join(v['violations'])}"
                            for x, v in sorted(rep["frozen"].items())
                            if not v["ok"])
            print(f"  !! {res['name']} is clear and certified all the way "
                  f"through, and then {rep['frozen_failed']} arm(s) stop in a "
                  f"pose that fails its own gates ({bad}); re-conducting the "
                  "whole phase with conductor v1's go-home")
            allocate.resequence(res, q_start=q_start, return_home=True)
            return build_phase(a, res, dt, pens, q_start=q_start,
                               policy=idle.POLICY_HOME)
        raise SystemExit(f"scene_check REFUSED {res['name']}; nothing rendered")
    return dict(res=res, progs=progs, samp=samp, paths=paths, sch=sch,
                qtraj=qtraj, rep=rep, M=M, worst_tip=worst_tip, idle=out)


def nominal_floor(a, res, pens, q_start=None, policy=idle.POLICY_FREEZE):
    """The longest per-arm frozen programme of an allocation. -> seconds.

    AN EXACT LOWER BOUND ON THAT ALLOCATION'S MAKESPAN, and a cheap one.  The
    conductor's only move is to insert pauses — it never shortens a path — so no
    schedule of these programmes can finish before the busiest arm's own
    programme does.  It costs one `writing.arm_program` per arm and not a single
    collision image, which is what lets `build_phases` decide whether a second
    allocation is even worth conducting.
    """
    best = 0.0
    for aid in res["arms"]:
        segs = res["programs"].get(aid, [])
        if not segs:
            continue
        p = writing.arm_program(FLEET[aid], segs, a.draw_speed, a.transit_speed,
                                H_INV_DEFAULT, qd_frac=a.qd_frac,
                                pen_ext=pens[aid],
                                q_start=(q_start or {}).get(aid), park=policy)
        best = max(best, float(p["duration"]))
    return best


def build(a):
    phases, strokes, info = run_allocation(a, verbose=False)
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
    # THE UNSPLIT ALLOCATION IS THE FALLBACK THE CONDUCTOR JUDGES v2 AGAINST.
    # It is only built for the phases that were actually cut — a phase the
    # splitter left alone is its own alternative — and it costs seconds, because
    # allocating is cheap and it is CONDUCTING that is expensive.
    alt = {}
    if not getattr(a, "no_split", False) and not getattr(a, "no_verify", False):
        cut = [k for k, p in enumerate(phases)
               if (p.get("balance") or {}).get("n_splits", 0)]
        if cut:
            print(f"\nallocating {len(cut)} split phase(s) again without cutting, "
                  "so the conductor can rule on whether the cuts paid")
            base, _, _ = run_allocation(a, verbose=False, split=False)
            for k in cut:
                base[k].update(name=phases[k]["name"] + " [unsplit]",
                               ink=phases[k]["ink"], strokes=phases[k]["strokes"])
                alt[k] = base[k]
    built = build_phases(a, phases, dt, pens, alt=alt)
    return phases, strokes, info, built, dt, pens


def build_phases(a, phases, dt, pens, alt=None):
    """Already-allocated phases -> the conducted, signed-off list. -> built.

    Split out of `build` so that a drawing which did not come from the tracer
    can be put through the IDENTICAL conductor: `scripts/bench.py` allocates its
    five generated test drawings and hands them here, and the regression numbers
    it prints are therefore produced by the same code path as the logo's — not
    by a second implementation that has to be kept in step with this one.

    THE CONDUCTOR HAS THE LAST WORD ON THE ALLOCATION.  `alt[k]` is a second
    allocation of the same phase drawing the same ink — in practice the same
    phase allocated WITHOUT stroke splitting — and when it is given, the phase is
    conducted both ways from the same starting pose and the faster one ships.

    That is the objective hierarchy applied where the information actually is.
    `allocate.balance_loads` minimises the busiest ARM; the makespan is the
    busiest arm PLUS everything the conductor must insert to keep six of them out
    of each other's way, and the balancer cannot see the second term — it prices
    every arm as though it had the paper to itself.  On the CSAIL orange pass
    that gap is the whole story: splitting takes the phase's floor from 65.1 s
    to 55.2 s and its conducted makespan from 65.1 s to 95.3 s, because four
    inverted arms that used to take turns now all work the middle of the sheet
    at once and spend 110 s waiting for each other.  A lower floor is a promise;
    only the conductor knows whether it can be kept, so only the conductor is
    allowed to accept it.

    The second conduct is usually not paid for.  `nominal_floor` is an exact
    lower bound on an allocation's makespan and costs no collision images at
    all, so an alternative that could not win even at its floor is dropped
    before a single one is built — which is the normal case, because the normal
    case is that splitting worked.
    """
    # A PASS STARTS WHERE THE LAST ONE STOPPED.  Under freeze-in-place the fleet
    # does not return to `q_seed` between passes, so pass 2 is sequenced from the
    # poses pass 1 actually froze in — including any minimal retreat, which is
    # only known once pass 1 has been conducted.  Re-ordering from the bag costs
    # a Held-Karp per arm and changes nothing about who draws what.
    # FREEZING ONLY PAYS FOR THE PASS NOBODY FOLLOWS.  An arm that stops where
    # it finished has not gone home, and the NEXT pass has to start from
    # somewhere: either from the frozen pose (which is what the sequencer is
    # re-priced for, and which on this logo the conductor refuses outright — the
    # four inverted arms end up inside each other's swept tubes), or from
    # `q_seed`, which means walking home anyway.  Walking home BETWEEN passes is
    # strictly worse than walking home during one, because during a pass the
    # trip overlaps somebody else's drawing and between passes nothing overlaps
    # it.  So an intermediate pass goes home — which is also what a human
    # walking in to swap the pens would ask for — and the LAST pass, the one
    # with nothing after it, freezes.
    built, q_start = [], None
    for k, ph in enumerate(phases):
        last = k == len(phases) - 1
        policy_k = a.idle_policy if (last or a.freeze_all_phases) \
            else idle.POLICY_HOME
        home_k = policy_k == idle.POLICY_HOME

        def prepare(res):
            if policy_k != a.idle_policy:
                print(f"\n{res['name']} is not the last pass, so it goes home "
                      "at the end: the pass after it starts from the ready pose")
                allocate.resequence(res, q_start=q_start, return_home=True)
            elif k and q_start is not None and a.idle_policy != idle.POLICY_HOME:
                print(f"\nre-sequencing {res['name']} from the poses pass "
                      f"{k} froze in (the allocation is untouched)")
                allocate.resequence(res, q_start=q_start, return_home=False)

        def conduct(res):
            try:
                return build_phase(a, res, dt, pens, q_start=q_start,
                                   policy=policy_k)
            except (SystemExit, idle.Unconductable, RuntimeError) as exc:
                print(f"  !! {res['name']} could not be conducted as allocated: "
                      f"{exc}")
                return None

        prepare(ph)
        B = conduct(ph)
        other = (alt or {}).get(k) if isinstance(alt, dict) else \
            (alt[k] if alt and k < len(alt) else None)
        if other is not None:
            prepare(other)
            lb = nominal_floor(a, other, pens, q_start, policy_k)
            if B is not None and lb >= float(B["sch"]["duration"]) - 1e-9:
                print(f"  the unsplit allocation of {ph['name']} floors at "
                      f"{lb:.1f} s, which is already the conducted "
                      f"{B['sch']['duration']:.1f} s or worse — not conducting it")
            else:
                print(f"  conducting {ph['name']} again WITHOUT stroke splitting "
                      f"(its floor is {lb:.1f} s"
                      + ("" if B is None else
                         f" against the {B['sch']['duration']:.1f} s splitting "
                         "achieved") + ")")
                C = conduct(other)
                if C is not None and (B is None or float(C["sch"]["duration"])
                                      < float(B["sch"]["duration"]) - 1e-9):
                    n = int((ph.get("balance") or {}).get("n_splits", 0))
                    print(f"  !! the conductor prefers the UNSPLIT allocation of "
                          f"{ph['name']}: {C['sch']['duration']:.1f} s against "
                          + ("a refusal" if B is None else
                             f"{B['sch']['duration']:.1f} s")
                          + f" — {n} split(s) discarded")
                    B, ph = C, other
                    phases[k] = other
        if B is None:
            raise SystemExit(f"{ph['name']} could not be conducted")
        B["split_kept"] = B["res"] is not other
        built.append(B)
        q_start = built[-1]["idle"]["q_end"]

        if k + 1 < len(phases):
            hold = {aid: built[-1]["qtraj"][aid][-1] for aid in FLEET}
            rep = scene_check.check_static(hold, built[0]["sch"]["margin"],
                                           pen_ext=pens, verbose=False)
            where = ("every arm parked at q_seed"
                     if a.idle_policy == idle.POLICY_HOME else
                     "every arm frozen where it finished")
            print(f"\npen-swap pause: {where}, min clearance "
                  f"{1000 * rep['min_clearance']:.1f} mm "
                  f"(margin {1000 * built[0]['sch']['margin']:.0f} mm), "
                  f"{len(FLEET) - rep['poses_failed']}/{len(FLEET)} poses pass "
                  f"their own gates -> {'PASS' if rep['ok'] else 'FAIL'}"
                  + (f"   !! arms {rep['pen_below_paper']} hold the pen BELOW "
                     "the paper plane" if rep["pen_below_paper"] else ""))
            if not rep["ok"]:
                bad = "; ".join(f"arm {x}: {','.join(v['violations'])}"
                                for x, v in rep["poses"].items() if not v["ok"])
                raise SystemExit("the pen-swap hold pose is not safe"
                                 + (f" ({bad})" if bad else ""))
    return built


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
    # --draw-speed, --transit-speed and --qd-frac all come from add_args: the
    # allocator balances the fleet on them and the sequencer prices its
    # transits with them before this script ever freezes a timeline.
    ap.add_argument("--safety", type=float, default=coordination.SAFETY_M)
    ap.add_argument("--calib", type=float, default=coordination.CALIB_M)
    ap.add_argument("--pause", type=float, default=2.0,
                    help="seconds of every-arm-parked between two passes, "
                         "while a human swaps the pens")
    # ---- the idle policy (aris_sixarm/idle.py); --idle-policy is in
    # csail_allocate.add_args, because the sequencer prices it too ---------
    ap.add_argument("--no-jit", action="store_true",
                    help="do not spend an arm's slack on slow pen-up taxiing")
    ap.add_argument("--no-retreat", action="store_true",
                    help="do not offer a minimal retreat to a frozen pose that "
                         "is in another arm's way")
    ap.add_argument("--jit-frac", type=float, default=idle.JIT_FRAC,
                    help="fraction of an arm's measured slack the slow taxi may "
                         "spend")
    ap.add_argument("--freeze-all-phases", action="store_true",
                    help="freeze at the end of EVERY pass, not just the last "
                         "one; the pass after a frozen one is then sequenced "
                         "from the poses it froze in")
    ap.add_argument("--no-verify", action="store_true",
                    help="ship the split allocation without conducting the "
                         "unsplit one as well.  The A/B is the only thing that "
                         "knows whether cutting a stroke paid: the balancer "
                         "prices each arm alone on the paper and cannot see the "
                         "contention a better-balanced fleet creates")
    ap.add_argument("--reseq-tries", type=int, default=3,
                    help="times a refused phase may be re-sequenced without the "
                         "pen-up transits the conductor could not run")
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
        makespan_s=float(sum(B["sch"]["duration"] for B in built)
                         + (n_pause * dt if n_pause else 0.0)),
        balanced=any(B["res"].get("balance") for B in built),
        idle_policy=a.idle_policy, jit=not a.no_jit, retreat=not a.no_retreat,
        jit_frac=a.jit_frac,
        logo=dict(w=info["logo_w"], h=info["logo_h"],
                  center=[float(x) for x in info["center"]],
                  offset=[float(x) for x in info["offset"]]),
        phases=[], sequencer=phases[0].get("sequencer", "opt"))
    for B in built:
        res, sch, rep, progs = B["res"], B["sch"], B["rep"], B["progs"]
        idl = B["idle"]
        summary["phases"].append(dict(
            idle=dict(policy=idl["policy"], asked=a.idle_policy,
                      passes=idl["passes"],
                      retreats=idl["retreats"], taxi=idl["taxi"],
                      rest_delay_s={str(k): float(v["delay_s"])
                                    for k, v in idl["rest"].items()},
                      frozen_ok=bool(rep["frozen_failed"] == 0),
                      parks=idl["parks"], sent_home=idl["sent_home"],
                      pen_below_paper=rep["frozen_pen_below_paper"]),
            arm_taxi_s={str(x): float(progs[x]["taxi_s"]) for x in res["arms"]},
            arm_retreat_s={str(x): float(progs[x]["retreat_s"])
                           for x in res["arms"]},
            name=res["name"], ink=res["ink"],
            split_kept=bool(B.get("split_kept", True)),
            traced_m=res["total_len"], drawn_m=res["drawn_len"],
            dropped_m=res["dropped_len"],
            duration_s=float(sch["duration"]),
            floor_s=float(max(sch["nominal"].values())),
            pause_total=float(sch["pause_total"]),
            pauses={str(k): float(v) for k, v in sch["pauses"].items()},
            priority=[int(x) for x in sch["order"]],
            priority_search=sch.get("search"),
            balance=_balance_json(res.get("balance")),
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
