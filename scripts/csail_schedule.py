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

With `--select-profile`, steps 1-4 are run for each of FOUR execution profiles
— `qd_frac` 0.30 or 0.60, fiber menus off or on — and the fastest one step 4
certifies is what ships (section 5).  The other three are recorded in the
schedule JSON with their makespan, or with the reason they were refused, or
with the floor that proves they could not have won.

Writes out/csail_schedule<tag>.npz, (with --program) the shipped allocation as
out/csail_program<tag>.json, and (with --final) the end-state still.
"""
import argparse
import contextlib
import copy
import io
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from aris_sixarm import (allocate, artwork, coordination, idle, pwl,  # noqa: E402
                         scene_check, sequence, trace, writing)
from aris_sixarm import fleet as fleet_mod                          # noqa: E402
from aris_sixarm.fleet import FLEET, SHEET, H_INV_DEFAULT           # noqa: E402
from csail_allocate import (add_args, run_allocation, final_png,    # noqa: E402
                            program_json, totals, _override,
                            _park_groups, alloc_kwargs)

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
                search_max_n=a.search_max_n,
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
                      + ", ".join(f"arm {x}: {sorted(v, key=idle.edge_key)}"
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
                                forbid={x: sorted(v, key=idle.edge_key)
                                        for x, v in forbid.items()})
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
    # the drawing mask is REPORTED, never gated: it separates "the pen tip is
    # low because it is drawing" from "the pen tip is low mid-flight", which is
    # the one number a human reading the paper-clearance line wants.
    draw_mask = {aid: samp[aid]["seg"][np.clip(sch["progress"][aid][:M], 0,
                                               samp[aid]["n"] - 1)] >= 0
                 for aid in FLEET}
    rep = scene_check.check_timeline(
        qtraj, dt, sch["margin"], programs=res["programs"], pen_ext=pens,
        progress={k: v[:M] for k, v in sch["progress"].items()}, sub=a.subcheck,
        drawing=draw_mask)
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


# ---------------------------------------------------------------------------
# 5.  THE EXECUTION PROFILE IS A PROPERTY OF THE PROGRAMME, NOT OF THE REPO
#
# Two knobs move the makespan more than anything else the pipeline chooses, and
# neither has one right value for every drawing:
#
#   `qd_frac`   the fraction of the FR3 joint-velocity limit a move may use.
#               `writing.draw_duration` stretches the ink until no joint exceeds
#               it, so it is the cap that decides whether a segment's clock is
#               the material's `length / draw_speed` or the joint limit's
#               (`docs/BENCH.md`, "Speed-dependent verdict").  0.60 is worth
#               −18.9 % of the CSAIL demo makespan and is what `README.md`
#               reproduces the animation with; 0.30 is the module default.
#   `cluster`   the entry/exit fiber menus and the (segment, direction,
#               VARIANT) DP, with the min-travel band they are variants OF —
#               `--cluster --band-objective min_travel`, the pair, because the
#               menus are variants of the band the objective chooses and
#               measuring one without the other measures neither
#               (`scripts/speed_sweep.py`, `docs/CONCURRENCY.md`).
#
# The evidence that they belong here rather than in a default: at 0.60 with the
# cluster features the CSAIL logo conducts in 77.792 s against 104.021 s on the
# shipped defaults, `scene_check` PASS at 82.4 mm — and the SAME 0.60 is
# REFUSED outright on `bench`'s spiral, after the whole re-sequence ladder,
# conductor v1's go-home and the unsplit allocation.  One number cannot be both,
# so the choice is made per programme, by conducting the candidates and keeping
# the fastest one that certifies.  A profile that cannot be conducted, or whose
# timeline `scene_check` refuses, is not a slower answer — it is not an answer,
# and it is recorded as REFUSED with the reason rather than quietly dropped.
#
# WHAT MAKES IT AFFORDABLE IS `nominal_floor`.  Conducting a cell is the
# expensive half of this pipeline (minutes to an hour); allocating one is
# seconds.  The busiest arm's own frozen programme is an exact lower bound on
# what any schedule of that allocation can achieve, so a candidate whose FLOOR
# is already the incumbent's certified MAKESPAN or worse cannot win and is
# never conducted.  Ordering the candidates by that floor is what makes the
# pruning bite, and it bites hard: over `bench`'s five drawings and the logo,
# 8 of the 24 cells were conducted and three of the six drawings decided on a
# single conduct (`docs/BENCH.md`).
#
# AND THE ORDER THEY ARE TRIED IN IS NOT THE ORDER THEY ARE LISTED IN.  Floor
# order decides which candidates are worth conducting, and it cannot be known
# until every cell has been allocated; but WHICH CELL TO FINISH FIRST is a
# different question, and the answer has been the same on this rig for every
# picture in the corpus.  `qd0.60+cluster` is what the CSAIL logo ships at
# (77.792 s against 104.021 s on the defaults, `docs/BENCH.md`), what the
# Trollface ships at, and what two-pass ships at.  So it is allocated,
# conducted and certified BEFORE the other three are allocated at all, and the
# moment `scene_check` signs its timeline off the programme is written to disk
# as the provisional best.
#
# That is the number this pipeline is now measured on: not how long the grid
# takes, but how long until there is a certified programme the fleet could run.
# The remaining three then run against it — pruned by their floors, conducted
# in parallel because they no longer prune each other — and replace it only if
# one of them is faster.
PROFILES = (dict(qd_frac=0.30, cluster=False), dict(qd_frac=0.30, cluster=True),
            dict(qd_frac=0.60, cluster=False), dict(qd_frac=0.60, cluster=True))
FIRST_PROFILE = "qd0.60+cluster"


class NoProfile(SystemExit):
    """No candidate certified.  Carries the grid, because that IS the result.

    A `SystemExit` so that every caller which already stops on a refused phase
    stops here too, and an object rather than a string so that a caller which
    reports rather than stops — `scripts/bench.py` prints a REFUSED row — can
    still say WHICH four things were tried and why each one failed.
    """

    def __init__(self, msg, grid):
        super().__init__(msg)
        self.grid = grid


def profile_name(p):
    """'qd0.60+cluster'. -> str.  The key every table and JSON record uses."""
    return f"qd{float(p['qd_frac']):.2f}" + ("+cluster" if p["cluster"] else "")


def profile_args(a, p):
    """`a` with one profile's knobs set. -> a shallow copy of `a`.

    `--cluster` carries `--band-objective min_travel` with it and the profile is
    the PAIR: the fiber menus are variants of the band the objective chooses, so
    a cluster run under the bottleneck objective is neither of the two things
    this grid is comparing.  With the menus off the caller's own objective is
    left alone, because then it is the only band there is.
    """
    b = copy.copy(a)
    # A PROFILE'S ARGUMENTS TRAVEL TO A CONDUCT WORKER, SO THEY MUST PICKLE.
    # `draw.py` hangs its "a programme certified" callback on the args object,
    # and a closure is exactly what `mp.Pool` cannot send.  The callback belongs
    # to the run, not to the cell, and `build` passes it separately.
    b.on_certified = None
    b.qd_frac = float(p["qd_frac"])
    b.cluster = bool(p["cluster"])
    if p["cluster"]:
        b.band_objective = "min_travel"
    else:
        b.band_objective = getattr(a, "band_objective", pwl.OBJECTIVE)
    return b


def profile_pause_s(a, n_phases):
    """The pen-swap seconds a makespan of `n_phases` phases carries. -> float."""
    if n_phases <= 1:
        return 0.0
    fps = float(getattr(a, "fps", 0.0) or 0.0)
    pause = float(getattr(a, "pause", 0.0))
    # the payload lays the pause down in whole animation frames, so quote the
    # number the schedule will actually have rather than the one asked for
    return round(pause * fps) / fps if fps else pause


def profile_floor(a, phases, pens, alt=None):
    """A lower bound on the conducted makespan of an allocation. -> seconds.

    THE BOUND HAS TO BE A BOUND, or the pruning it drives is a guess.  Three
    things are deliberately left out, and each of them only ever ADDS seconds to
    the number this is bounding: the conductor's pauses (it never shortens a
    path, it only waits), the idle policy's taxi and retreat, and the trip home
    that every pass but the last one pays for at its end — this prices every
    phase as though it froze in place, which is the cheaper park.  What it does
    assume is that a pass starts from the ready pose, which is true under the
    shipped policy (an intermediate pass goes home, so the next one starts
    there) and not under `--freeze-all-phases`, where this is an estimate and
    the pruning it drives should be turned off.

    `alt[k]` is the unsplit allocation `build_phases` may ship instead, so the
    bound has to be the cheaper of the two floors — the conductor is allowed to
    prefer either one and the bound must hold for whichever it ships.
    """
    tot = profile_pause_s(a, len(phases))
    for k, ph in enumerate(phases):
        f = nominal_floor(a, ph, pens, None, idle.POLICY_FREEZE)
        other = (alt or {}).get(k) if isinstance(alt, dict) else None
        if other is not None:
            f = min(f, nominal_floor(a, other, pens, None, idle.POLICY_FREEZE))
        tot += float(f)
    return float(tot)


def profile_makespan(a, built):
    """What the shipped summary will call `makespan_s` for this conduct."""
    return float(sum(B["sch"]["duration"] for B in built)
                 + profile_pause_s(a, len(built)))


def profile_order(profiles, first=FIRST_PROFILE):
    """`profiles` with the preferred cell first, the rest in their own order."""
    names = [profile_name(p) for p in profiles]
    if first in names:
        k = names.index(first)
        return [profiles[k]] + [p for i, p in enumerate(profiles) if i != k]
    return list(profiles)


def _conduct_one(args):
    """One profile's conduct, in its own process. -> (name, built, reason, log).

    Module level and returning rather than raising, for `_alloc_profile`'s
    reason; and its stdout comes back as TEXT rather than going to the terminal,
    because three conducts talking at once is three conducts nobody can read.
    """
    name, b, phases, dt, pens, alt = args
    buf, t0 = io.StringIO(), time.time()
    try:
        with contextlib.redirect_stdout(buf):
            built = build_phases(b, phases, dt, pens, alt=alt)
    except (SystemExit, idle.Unconductable, RuntimeError) as exc:
        return (name, None, f"{type(exc).__name__}: {exc}", buf.getvalue(),
                time.time() - t0)
    return name, built, None, buf.getvalue(), time.time() - t0


def _record(r, built, reason, conduct_s):
    """Fold one conduct's outcome into its grid row. -> the row."""
    r["conduct_s"] = conduct_s
    if built is None:
        r.update(status="refused", reason=reason)
        return r
    ok = all(bool(B["rep"]["ok"]) for B in built)
    r.update(built=built, makespan_s=profile_makespan(r["args"], built),
             scene_check=ok,
             min_clearance=float(min(B["rep"]["min_clearance"] for B in built)))
    if not ok:
        # `build_phase` already has the veto and raises rather than returning a
        # refused timeline; this is the belt to that braces, so that a future
        # conductor which reports instead of raising cannot ship an uncertified
        # profile through this door.
        r.update(status="refused",
                 reason="scene_check refused the conducted timeline")
    else:
        r["status"] = "certified"
    return r


def select_profile(a, alloc, dt, conduct=None, floor=None, profiles=PROFILES,
                   prune=True, verbose=True, first=FIRST_PROFILE, jobs=None,
                   on_certified=None):
    """Conduct the execution profiles and ship the fastest CERTIFIED one.

        alloc(b, p) -> (phases, alt, pens)     allocate `p` under the args `b`
        conduct(b, phases, dt, pens, alt=alt) -> built        (`build_phases`)

    Returns `dict(chosen=<grid row>, grid=[<grid row> x len(profiles)],
    first_certified_s=...)`, in which the chosen row carries `built`, `phases`,
    `alt` and `pens` — the conduct that WON is the conduct that ships, never
    re-run, so what the payload is written from is the timeline `scene_check`
    signed off on here.

    TWO STAGES, AND THE FIRST ONE IS THE ANSWER.  `first` names the cell this
    rig has always shipped (see the section note); it is allocated and conducted
    on its own, and `on_certified(row)` is called the moment its timeline
    certifies so the caller can put a runnable programme on disk.  Only then are
    the other three allocated, pruned against that certified makespan, and
    conducted — in `jobs` processes, because a floor that has already been
    beaten prunes just as well from a parallel batch as from a serial one, and
    what the remaining candidates cannot do is prune each other.  `jobs=1`
    conducts them one at a time in floor order, which is the serial chain every
    published number was measured on.

    Deterministic in what it ships: the allocator and the sequencer are
    functions of their input (up to the local-search budgets, which is the same
    caveat every row of `docs/BENCH.md` carries), the candidate order is by
    floor with ties broken on the fixed `PROFILES` order, and running the
    conducts concurrently cannot change any one of their timelines — only how
    many of them get run.
    """
    conduct = build_phases if conduct is None else conduct
    floor = profile_floor if floor is None else floor
    t_start = time.time()
    rows = {}
    for p in profiles:
        b = profile_args(a, p)
        rows[profile_name(p)] = dict(
            profile=profile_name(p), qd_frac=float(p["qd_frac"]),
            cluster=bool(p["cluster"]),
            band_objective=getattr(b, "band_objective", None),
            status="allocated", floor_s=None, makespan_s=None, reason=None,
            alloc_s=0.0, conduct_s=0.0, args=b)
    grid = [rows[profile_name(p)] for p in profiles]
    order = profile_order(profiles, first)

    def do_alloc(p, tag):
        row = rows[profile_name(p)]
        if verbose:
            print(f"\n{'=' * 74}\n=== allocating profile {profile_name(p)} "
                  f"({tag})\n{'=' * 74}")
        t0 = time.time()
        try:
            phases, alt, pens = alloc(row["args"], p)
            row.update(phases=phases, alt=alt, pens=pens,
                       floor_s=float(floor(row["args"], phases, pens, alt)))
        except (SystemExit, idle.Unconductable, RuntimeError) as exc:
            # AN ALLOCATION CAN REFUSE TOO, and a profile that cannot be
            # allocated (or that draws less of the picture than this run was
            # asked for) is refused for the same reason a refused conduct is:
            # it is not a slower answer, it is not an answer.
            row.update(status="refused", reason=f"{type(exc).__name__}: {exc}")
        row["alloc_s"] = time.time() - t0
        if verbose:
            print(f"\n--- profile {row['profile']} allocated in "
                  f"{row['alloc_s']:.1f} s"
                  + (f", floor {row['floor_s']:.3f} s"
                     if row["floor_s"] is not None else f": {row['reason']}"))
        return row

    best, first_s = None, None

    def keep(r):
        """Adopt `r` as the incumbent and tell the caller it may be shipped."""
        nonlocal best, first_s
        best = r
        if first_s is None:
            first_s = time.time() - t_start
        if on_certified is not None:
            on_certified(r)

    # ---- stage 1: the cell most likely to win, all the way to a programme ---
    lead = do_alloc(order[0], f"1 of {len(profiles)}, the provisional best")
    if lead["status"] == "allocated":
        lead["rank"] = 0
        if verbose:
            print(f"\n{'=' * 74}\n=== conducting profile {lead['profile']} "
                  f"(qd_frac {lead['qd_frac']:.2f}, cluster "
                  f"{'on' if lead['cluster'] else 'off'}, band "
                  f"{lead['band_objective']}), floor {lead['floor_s']:.3f} s"
                  f"\n{'=' * 74}")
        t0 = time.time()
        try:
            built = conduct(lead["args"], lead["phases"], dt, lead["pens"],
                            alt=lead["alt"])
            _record(lead, built, None, time.time() - t0)
        except (SystemExit, idle.Unconductable, RuntimeError) as exc:
            _record(lead, None, f"{type(exc).__name__}: {exc}",
                    time.time() - t0)
        if lead["status"] == "certified":
            if verbose:
                print(f"  profile {lead['profile']} CERTIFIED: "
                      f"{lead['makespan_s']:.3f} s against a floor of "
                      f"{lead['floor_s']:.3f} s, clearance "
                      f"{1000 * lead['min_clearance']:.1f} mm "
                      f"({lead['conduct_s']:.0f} s)")
            keep(lead)
            if verbose:
                print(f"\n*** PROVISIONAL BEST after {first_s:.1f} s: "
                      f"{lead['profile']} at {lead['makespan_s']:.3f} s — a "
                      "certified programme exists from here on ***")
        elif verbose:
            print(f"  !! profile {lead['profile']} {lead['status'].upper()}: "
                  f"{lead['reason']}")

    # ---- stage 2: the rest, against a makespan that already certified -------
    for i, p in enumerate(order[1:]):
        do_alloc(p, f"{i + 2} of {len(profiles)}")
    ranked = sorted((r for r in grid
                     if r["status"] == "allocated" and r is not lead),
                    key=lambda r: (r["floor_s"], grid.index(r)))
    if verbose:
        print(f"\n{'=' * 74}\n=== execution profiles: "
              f"{sum(1 for r in grid if r['floor_s'] is not None)} of "
              f"{len(grid)} allocated\n{'=' * 74}")
        for r in grid:
            print(f"  {r['profile']:<16} "
                  + (f"floor {r['floor_s']:8.3f} s   (allocated in "
                     f"{r['alloc_s']:.1f} s)" if r["floor_s"] is not None
                     else f"REFUSED at allocation: {r['reason']}"))
    todo = []
    for rank, r in enumerate(ranked, start=1):
        r["rank"] = rank
        if prune and best is not None \
                and r["floor_s"] >= best["makespan_s"] - 1e-9:
            r.update(status="pruned",
                     reason=f"floor {r['floor_s']:.3f} s cannot beat the "
                            f"certified {best['makespan_s']:.3f} s of "
                            f"{best['profile']}")
            if verbose:
                print(f"\n  {r['profile']}: {r['reason']} — not conducted")
            continue
        todo.append(r)

    def report(r):
        if not verbose:
            return
        if r["status"] == "certified":
            print(f"  profile {r['profile']} CERTIFIED: "
                  f"{r['makespan_s']:.3f} s against a floor of "
                  f"{r['floor_s']:.3f} s, clearance "
                  f"{1000 * r['min_clearance']:.1f} mm ({r['conduct_s']:.0f} s)")
        else:
            print(f"  !! profile {r['profile']} {r['status'].upper()}: "
                  f"{r['reason']}")

    # `conduct is build_phases` because only the real conductor is known to be
    # forkable: a caller that passes its own (the tests do) gets the serial path
    n_jobs = len(todo) if jobs is None else int(jobs)
    if todo and n_jobs > 1 and conduct is build_phases:
        if verbose:
            print(f"\n{'=' * 74}\n=== conducting "
                  + ", ".join(r["profile"] for r in todo)
                  + f" on {min(n_jobs, len(todo))} processes — the provisional "
                    "best has pruned them already and they cannot prune each "
                    f"other\n{'=' * 74}")
        with mp.get_context("fork").Pool(min(n_jobs, len(todo))) as pool:
            out = pool.map(_conduct_one,
                           [(r["profile"], r["args"], r["phases"], dt,
                             r["pens"], r["alt"]) for r in todo])
        for r, (_name, built, reason, log, took) in zip(todo, out):
            if verbose:
                print(f"\n{'-' * 74}\n--- profile {r['profile']}\n{log.rstrip()}")
            _record(r, built, reason, took)
            report(r)
        # judged together, in floor order, so which one ships does not depend
        # on which one happened to finish first
        for r in todo:
            if r["status"] == "certified" \
                    and (best is None
                         or r["makespan_s"] < best["makespan_s"] - 1e-9):
                keep(r)
    else:
        for r in todo:
            if prune and best is not None \
                    and r["floor_s"] >= best["makespan_s"] - 1e-9:
                r.update(status="pruned",
                         reason=f"floor {r['floor_s']:.3f} s cannot beat the "
                                f"certified {best['makespan_s']:.3f} s of "
                                f"{best['profile']}")
                if verbose:
                    print(f"\n  {r['profile']}: {r['reason']} — not conducted")
                continue
            if verbose:
                print(f"\n{'=' * 74}\n=== conducting profile {r['profile']} "
                      f"(qd_frac {r['qd_frac']:.2f}, cluster "
                      f"{'on' if r['cluster'] else 'off'}, band "
                      f"{r['band_objective']}), floor {r['floor_s']:.3f} s"
                      + ("" if best is None else
                         f", to beat {best['makespan_s']:.3f} s")
                      + f"\n{'=' * 74}")
            t0 = time.time()
            try:
                built = conduct(r["args"], r["phases"], dt, r["pens"],
                                alt=r["alt"])
                _record(r, built, None, time.time() - t0)
            except (SystemExit, idle.Unconductable, RuntimeError) as exc:
                _record(r, None, f"{type(exc).__name__}: {exc}",
                        time.time() - t0)
            report(r)
            if r["status"] == "certified" \
                    and (best is None
                         or r["makespan_s"] < best["makespan_s"] - 1e-9):
                keep(r)

    if best is None:
        raise NoProfile(
            "no execution profile could be certified: "
            + "; ".join(f"{r['profile']}: {r['reason']}" for r in grid), grid)
    best["chosen"] = True
    if verbose:
        print(f"\n{'=' * 74}\n=== SHIPPING {best['profile']}: "
              f"{best['makespan_s']:.3f} s\n" + "\n".join(profile_table(grid))
              + f"\n=== first certified programme at {first_s:.1f} s, "
                f"grid finished at {time.time() - t_start:.1f} s\n{'=' * 74}")
    return dict(chosen=best, grid=grid, first_certified_s=first_s,
                grid_s=time.time() - t_start)


def profile_table(grid):
    """The four outcomes as markdown. -> list of lines."""
    out = ["| profile | qd_frac | cluster | floor | makespan | outcome |",
           "|---|---|---|---|---|---|"]
    for r in grid:
        ms = ("**%.3f s**" % r["makespan_s"]) if r.get("makespan_s") is not None \
            else "—"
        if r.get("chosen"):
            ms += " (shipped)"
        out.append(f"| {r['profile']} | {r['qd_frac']:.2f} | "
                   f"{'on' if r['cluster'] else 'off'} | "
                   + (f"{r['floor_s']:.3f} s" if r["floor_s"] is not None else "—")
                   + f" | {ms} | "
                   + {"certified": "certified",
                      "pruned": "not conducted",
                      "refused": "REFUSED"}.get(r["status"], r["status"])
                   + (f" — {r['reason']}" if r["reason"] else "") + " |")
    return out


def profile_json(sel):
    """The grid as it is recorded in the program and schedule JSON. -> dict.

    Every cell of it, refusals and prunings included with their reason: the
    record of what shipped is only worth anything beside the record of what did
    not, and a run that quietly dropped the three losers would be
    indistinguishable from one that never tried them.
    """
    def row(r):
        return dict(
            profile=r["profile"], qd_frac=r["qd_frac"], cluster=r["cluster"],
            band_objective=r["band_objective"], status=r["status"],
            chosen=bool(r.get("chosen", False)),
            floor_rank=r.get("rank"),
            floor_s=r["floor_s"], makespan_s=r.get("makespan_s"),
            min_clearance=r.get("min_clearance"),
            scene_check=r.get("scene_check"), reason=r["reason"],
            alloc_s=round(float(r["alloc_s"]), 3),
            conduct_s=round(float(r["conduct_s"]), 3))
    ch = sel.get("chosen")
    return dict(chosen=ch["profile"] if ch else None,
                qd_frac=ch["qd_frac"] if ch else None,
                cluster=ch["cluster"] if ch else None,
                band_objective=ch["band_objective"] if ch else None,
                makespan_s=ch["makespan_s"] if ch else None,
                floor_s=ch["floor_s"] if ch else None,
                n_conducted=int(sum(r["status"] in ("certified", "refused")
                                    and r.get("rank") is not None
                                    for r in sel["grid"])),
                grid=[row(r) for r in sel["grid"]])


def arm_groups(mode, arms, near=0.70, fleet=None):
    """--arm-phases -> [[arm ids]] (or None for 'off').

    'disjoint' is a graph colouring, not a table: two arms are adjacent when
    their BASES are within `near` metres, and the groups are the colour
    classes of a greedy colouring in registry order.  On the proposed rig's
    2 x 3 grid that is exactly the two columns — the three transverse pairs
    are the only edges — and on a rig whose arms are all far apart it is one
    group, i.e. 'off', which is the right answer there.
    """
    if mode in (None, "", "off", "none"):
        return None
    ids = sorted(set(arms))          # a two-pass run names its arms twice
    if mode == "solo":
        return [[a] for a in ids]
    if mode != "disjoint":
        groups = [[int(x) for x in g.split(",") if x.strip()]
                  for g in str(mode).split("/") if g.strip()]
        if not groups:
            raise SystemExit(f"--arm-phases {mode!r}: no groups")
        return groups
    fl = FLEET if fleet is None else fleet
    xy = {a: np.asarray(fl[a].xy, float) for a in ids}
    colour = {}
    for a in ids:
        taken = {colour[b] for b in colour
                 if float(np.linalg.norm(xy[a] - xy[b])) < near}
        colour[a] = next(c for c in range(len(ids)) if c not in taken)
    return [[a for a in ids if colour[a] == c]
            for c in sorted(set(colour.values()))]


# ==========================================================================
# A BAG THE ARM CANNOT FLY IN ONE TOUR IS NOT INK THE ARM CANNOT FLY
# ==========================================================================
# `allocate.prune_unflyable` asks ONE question — "is there a paper-legal
# Hamiltonian path through this arm's whole bag" — and when the answer is no it
# takes ink away until the answer is yes.  For the spans with no finite
# predecessor that is exactly right and there is nothing else to do.  For the
# rest it is a much stronger conclusion than the measurement supports, and on
# the proposed rig at h = 0.940 it is the entire coverage story:
#
#   `out/residual_anatomy.py` routes every stroke of the logo from every arm's
#   own DEPOT with a bag of ONE, nobody parked.  All 12.7783 m of it — every
#   stroke, both inks — has at least one arm that certifies the ink and can fly
#   to it.  Arm 31 alone certifies and can reach all thirty-eight.
#
# So none of the 4.84 m the 6-mover pass leaves empty is unreachable.  It is
# ink whose arm could not visit it IN THE SAME TOUR as the rest of its bag:
# `prune_unflyable` finds no isolated node, falls through to its
# shortest-segment fallback, and drops eighteen strokes one at a time until a
# path exists through what is left.
#
# A TOUR IS A PROPERTY OF A PASS, AND A RUN MAY HAVE MORE THAN ONE PASS.  Every
# pass starts from the depot — an intermediate pass goes home, which is already
# what `build_phases` makes it do — so the ink one tour could not thread is ink
# a SECOND tour can be given, from the same arm, in the same scene, at the cost
# of a go-home and a pause.  That is what this does: allocate the hole the last
# pass left, again, with the identical allocator, and hand the conductor
# another phase.  It adds ink and it costs makespan, which is the objective
# hierarchy this repository already runs on — coverage is a constraint, and
# the clock is what it is spent from.
#
# ESTIMATES PRUNE, CONDUCTION DECIDES, as always: a residual pass is an
# ALLOCATION and the conductor still rules on it like any other phase.  A pass
# that certifies no new ink is dropped before it costs a conduct.
def residual_strokes(results, min_len=None):
    """The ink a round left empty, as a stroke set. -> (strokes, refused).

    `results` is EVERY pass of the round, not the last one: a two-pass round
    allocates grey and orange independently and each leaves its own holes, so
    chaining off `phases[-1]` alone would hand the next round the orange holes
    and quietly abandon the grey ones.

    `allocate.leftover` already carries the GEOMETRY of every hole — it
    truncates the original polyline to the uncovered span — so the next pass
    is handed real strokes and not a bookkeeping range, and it re-probes,
    re-partitions the colours and re-sequences them from scratch.  Ids are
    fresh and dense because `allocate.allocate` keys its interval map on them
    and the next pass is a self-contained problem.

    `refused` is every hole too short to be worth a segment of its own, handed
    BACK rather than dropped on the floor: it is still empty paper and the
    composed programme's coverage has to keep saying so.
    """
    min_len = allocate.MIN_SEG_M if min_len is None else float(min_len)
    out, refused = [], []
    for res in results:
        for d in res.get("dropped") or ():
            pts = np.asarray(d["pts"], float)
            if len(pts) < 2 or allocate.polyline_length(pts) < min_len:
                refused.append(d)
                continue
            out.append(dict(id=len(out), color=d["color"],
                            kind=d.get("kind", ""), pts=pts))
    return out, refused


def residual_passes(a, phases, share=None, rounds=0, min_gain=None):
    """Re-allocate what the last pass left empty. -> ([phase], refused holes).

    Each round is a full `allocate.allocate` on the previous round's holes,
    made with `alloc_kwargs` so it is the same allocator asking the same
    question of the same fleet.  It stops early on two conditions: nothing
    left to allocate, and a round that certified less than `min_gain` metres
    (a pass costs a go-home and a pause and is not worth confetti).  A round
    that gives back only a little is still KEPT if it clears that floor — the
    conductor is the one that decides whether the phase is runnable, and it
    has not been asked yet.

    THE PICTURE'S BOOKKEEPING STAYS ON THE FIRST PASS.  `totals` sums
    `total_len` and `dropped_len` over phases, so a residual pass must not
    claim its input as newly traced metres — it is the same paper.  Every
    returned pass carries `total_len = 0`, and the caller moves the FINAL
    hole list onto phase 0, so "traced = drawn + dropped" still holds over the
    composed programme by construction.
    """
    rounds = int(rounds or 0)
    if rounds <= 0 or not phases:
        return [], []
    min_gain = allocate.MIN_SEG_M if min_gain is None else float(min_gain)
    arms = allocate.active_arms(_override(a.arms))
    # A ROUND HAS THE SAME SHAPE AS THE PASS IT FOLLOWS.  Under `--two-pass`
    # every arm may hold either colour, one phase at a time, and the holes are
    # re-allocated the same way — one sub-pass per ink — so the residual keeps
    # the whole fleet on each colour instead of falling back to the
    # one-pen-per-arm partition the two-pass run exists to escape.  Ordering by
    # ink also means consecutive same-colour phases need no swap between them.
    def holes(results):
        return [d for res in results for d in (res.get("dropped") or [])]

    out, refused, prev = [], [], list(phases)
    for r in range(rounds):
        st, tiny = residual_strokes(prev)
        if not st:
            print(f"\nresidual pass {r + 1}: nothing left to allocate")
            break
        left = trace.total_length(st)
        print(f"\n=== residual pass {r + 1}: the {left:.4f} m in {len(st)} "
              f"span(s) the last round left empty, allocated again over all "
              f"{len(arms)} arms ===")
        inks = artwork.inks_of(st)
        if getattr(a, "two_pass", False) and len(inks) > 1:
            bags = [(ink, [s for s in st if s["color"] == ink]) for ink in inks]
        else:
            bags = [(inks[0] if len(inks) == 1 else None, st)]
        kw = alloc_kwargs(a, share=share)
        made, gain = [], 0.0
        for ink, sub in bags:
            if not sub:
                continue
            tag = (f"residual pass {r + 1}" if len(bags) == 1
                   else f"residual pass {r + 1}: {ink}")
            print(f"\n  --- {tag} — {len(sub)} span(s), "
                  f"{trace.total_length(sub):.4f} m ---")
            res = allocate.allocate(
                sub, colors=(None if ink is None else {x: ink for x in arms}),
                **kw)
            res.update(name=tag, ink=ink, strokes=sub, residual_round=r + 1,
                       total_len=0.0)
            for line in allocate.report(res, sub):
                print(line)
            made.append(res)
            gain += float(res["drawn_len"])
        if gain < min_gain:
            print(f"  residual pass {r + 1} certifies {gain:.4f} m, under the "
                  f"{min_gain:.4f} m a go-home and a pause are worth — dropped")
            break
        # A SUB-PASS THAT CERTIFIED NOTHING IS NOT A PHASE — it would buy a
        # go-home and a pause and no ink — but its HOLES are still the next
        # round's input, so it is dropped from the programme and kept in the
        # chain.
        kept = [x for x in made if x["drawn_len"] > 1e-9]
        out += kept
        print(f"\n  residual pass {r + 1} gives back {gain:.4f} m of the "
              f"{left:.4f} m it was handed, in {len(kept)} phase(s)")
        # THE CONFETTI IS ABANDONED ONLY NOW.  `tiny` is the part of this
        # round's input too short to re-offer; it is permanently empty paper
        # from here, whereas `prev`'s other holes have just been answered by
        # `made` and must not be counted again.  Adding it before the round
        # was known to be kept would double-count it against the refusal
        # branches above, which return `prev`'s holes whole.
        refused += tiny
        prev = made
    # what the composed programme finally leaves empty: every hole the last
    # round still has, plus every scrap earlier rounds were too small to offer
    return out, refused + holes(prev)


def phase_by_arms(phases, alt, groups):
    """Split every phase by arm group. -> (phases, alt), indices re-mapped."""
    out, new_alt = [], {}
    for k, ph in enumerate(phases):
        parts = allocate.split_by_arms(ph, groups)
        other = (alt or {}).get(k)
        alt_parts = allocate.split_by_arms(other, groups) if other else None
        for j, p in enumerate(parts):
            if alt_parts is not None and j < len(alt_parts):
                new_alt[len(out)] = alt_parts[j]
            out.append(p)
    return out, new_alt


def allocate_all(a, verbose=False, share=None):
    """Trace, allocate, and build the unsplit fallback. -> (phases, alt, pens).

    Everything `build_phases` needs and nothing it does not, so that ONE
    allocation is what the single-profile path and every cell of the profile
    grid are made of.  Refuses on coverage here rather than after conducting:
    a run that draws less of the picture than it was asked for is not a faster
    run, and finding that out costs one allocation instead of one conduct.
    """
    share = {} if share is None else share
    phases, strokes, info = run_allocation(a, verbose=False,
                                           px=getattr(a, "traced_px", None),
                                           share=share)
    for ph in phases:
        print(f"\n=== {ph['name']} ===")
        for line in allocate.report(ph, ph["strokes"]):
            print(line)
    n_primary = len(phases)
    # ...and then the same allocator, again, on the holes (see `residual_passes`)
    extra, empty = residual_passes(a, phases, share=share,
                                   rounds=getattr(a, "residual_passes", 0))
    if extra:
        # THE HOLE THE COMPOSED PROGRAMME LEAVES IS THE LAST ONE, NOT THE FIRST.
        # Every pass after the first was handed the pass before it as its
        # picture, so the intermediate hole lists are already answered; leaving
        # them on their phases would count the same empty paper once per pass.
        # The picture's bookkeeping rides on phase 0 (the convention
        # `allocate.split_by_arms` already uses), so the FINAL hole list moves
        # there and every other phase's is emptied.
        for ph in phases + extra:
            ph["dropped"], ph["dropped_len"] = [], 0.0
        phases[0]["dropped"] = list(empty)
        phases[0]["dropped_len"] = float(sum(d["length"] for d in empty))
        phases = phases + extra
    T = totals(phases)
    print(f"\nALL PHASES: {T['traced']:.4f} m traced, {T['dropped']:.4f} m left "
          f"empty -> COVERAGE {100 * T['covered']:.4f} %"
          + (f"  ({n_primary} pass(es) + {len(extra)} residual)" if extra
             else ""))
    want = 1.0 if a.require_full else float(a.min_coverage)
    if T["covered"] < want - 1e-12:
        raise SystemExit(
            f"coverage is {100 * T['covered']:.4f} %, below the "
            f"{100 * want:.2f} % this run was asked for; {T['dropped']:.4f} m "
            f"in {T['n_dropped']} spans left empty (lower --min-coverage to "
            "render it anyway)")

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
            base, _, _ = run_allocation(a, verbose=False, split=False,
                                        px=getattr(a, "traced_px", None),
                                        share=share)
            for k in cut:
                base[k].update(name=phases[k]["name"] + " [unsplit]",
                               ink=phases[k]["ink"], strokes=phases[k]["strokes"])
                alt[k] = base[k]

    # ...and THEN, if asked, the same ink drawn by fewer arms at a time.  After
    # the coverage gate and after the A/B, because neither is about the
    # schedule: phasing by arm changes when an arm draws and nothing about
    # what is drawn, so the coverage this run refuses on is the same number
    # either way and both allocations split the same way.
    # THE SAME GROUPING THE ALLOCATION WAS PRUNED AGAINST.  `csail_allocate`
    # asks who will be parked before it allocates (`_parks`), and a colouring
    # taken over a DIFFERENT arm set can come out differently — an arm that
    # drew nothing changes the greedy order.  So the grouping is computed once,
    # over the arms this run may use, and both stages read that one.
    groups = _park_groups(a, allocate.active_arms(_override(a.arms)))
    if groups:
        phases, alt = phase_by_arms(phases, alt, groups)
        print(f"\nconducting in {len(phases)} phase(s), "
              f"{len(groups)} arm group(s) per pass: "
              + "  ".join("{" + ",".join(str(x) for x in g) + "}"
                          for g in groups)
              + "  (everybody outside the drawing group waits at the depot)")
    return phases, alt, pens, strokes, info


def _alloc_profile(args):
    """One profile's allocation, in its own process. -> (name, result | error).

    Module level and returning rather than raising, so `mp.Pool` can both send
    it and get an answer back from a cell that refused — `select_profile`
    records a refused allocation as an outcome with a reason, and losing that
    to a worker traceback would turn a recorded refusal into a dead run.
    """
    a, p = args
    try:
        return profile_name(p), allocate_all(profile_args(a, p)), None
    except (SystemExit, idle.Unconductable, RuntimeError) as exc:
        return profile_name(p), None, f"{type(exc).__name__}: {exc}"


def build(a, on_certified=None):
    """Allocate and conduct, at one profile or at the best of four. -> tuple.

    `on_certified(row, strokes, info, dt)` is called the moment a profile's
    timeline certifies, so a caller can put a runnable programme on disk before
    the grid has finished (see `select_profile`).  It is a PARAMETER and not an
    attribute of `a` because the per-profile argument objects are pickled to the
    conduct workers and a closure will not go.
    """
    # A MODULE GLOBAL AND NOT A PARAMETER, because `coordinate` is four calls
    # below here and the image pool is a property of the MACHINE rather than of
    # any allocation.  It is read at every dispatch, so setting it here reaches
    # the conducts of every profile, including the forked ones.
    if getattr(a, "image_jobs", None) is not None:
        coordination.IMAGE_JOBS = int(a.image_jobs)
    dt = 1.0 / (a.fps * a.substeps)
    if not getattr(a, "select_profile", False):
        phases, alt, pens, strokes, info = allocate_all(a)
        built = build_phases(a, phases, dt, pens, alt=alt)
        return phases, strokes, info, built, dt, pens, None

    # THE PROFILE IS SELECTED ON THIS PROGRAMME, BY CONDUCTING IT.  The tracer
    # and the placement are the same in every cell — only the two knobs move —
    # so the strokes and the placement info are kept per cell only because the
    # winner's are the ones that get written out beside its allocation.
    #
    # THE FOUR ALLOCATIONS ARE INDEPENDENT, AND ON A DENSE PICTURE THEY ARE THE
    # RUN.  `docs/BENCH.md`'s premise is that conducting a cell is the expensive
    # half and allocating one is seconds; that holds for the logo's 42 segments
    # and inverts for a picture with 63, where `balance_loads`' split search is
    # 676.8 s of a 724.1 s allocation and the four cells are 56 minutes of
    # strictly serial single-core work.  `--profile-jobs` maps them over
    # processes instead.  It cannot change any result — each cell is a pure
    # function of `(a, profile)` — so the default stays 1 and every published
    # number reproduces on the serial path.
    #
    # THE FOUR ALLOCATIONS ARE NOT INDEPENDENT ANY MORE, ON PURPOSE.  Two of the
    # knobs the grid moves are invisible to the planner: `plan_family` says two
    # profiles that agree on the band objective ask `plan_stroke` the same
    # questions, and `paper`'s route memo says all four ask `paper.route` the
    # same ones.  Run in ONE process with a shared bag they cost the geometry
    # once between them, which is worth more than running them in four processes
    # that each pay for it — `--profile-jobs` is still there for a machine where
    # it is not.
    seen, pre, share = {}, {}, {}
    jobs = int(getattr(a, "profile_jobs", 1) or 1)
    if jobs > 1:
        print(f"\nallocating {len(PROFILES)} execution profiles on {jobs} "
              "processes (they are independent; this trades the shared plan and "
              "route memos for cores)")
        t0 = time.time()
        with mp.get_context("fork").Pool(min(jobs, len(PROFILES))) as pool:
            for name, res, err in pool.map(_alloc_profile,
                                           [(a, p) for p in PROFILES]):
                pre[name] = (res, err)
        print(f"  {len(pre)} allocations in {time.time() - t0:.1f} s")

    def alloc(b, p):
        name = profile_name(p)
        if name in pre:
            res, err = pre[name]
            if err is not None:
                raise SystemExit(err)
            phases, alt, pens, strokes, info = res
        else:
            phases, alt, pens, strokes, info = allocate_all(b, share=share)
        seen[name] = (strokes, info)
        return phases, alt, pens

    hook = on_certified if on_certified is not None \
        else getattr(a, "on_certified", None)

    def certified(row):
        """Tell the caller a runnable programme exists, with what it needs."""
        if hook is not None:
            strokes, info = seen[row["profile"]]
            hook(row, strokes, info, dt)

    sel = select_profile(a, alloc, dt,
                         prune=not getattr(a, "no_profile_prune", False),
                         first=getattr(a, "first_profile", FIRST_PROFILE),
                         jobs=getattr(a, "conduct_jobs", None),
                         on_certified=certified if hook is not None else None)
    best = sel["chosen"]
    strokes, info = seen[best["profile"]]
    return (best["phases"], strokes, info, best["built"], dt, best["pens"], sel)


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
            """-> True if the phase is ready to conduct.

            RE-SEQUENCING CAN REFUSE, AND THAT IS A PHASE VERDICT (2026-08-26).
            `resequence` prices every segment order against the transit
            router, and on a rig whose parked arms stand in the transit
            corridors the Held-Karp cost matrix can come out with no finite
            tour at all — `no feasible order over N segments`.  That is the
            same statement as "the conductor refuses this phase", reached one
            stage earlier, so it is caught in the same place and by the same
            dial: with `--skip-unconductable` the phase is dropped and the
            rest of the picture still ships, without it the run stops.  It
            used to propagate out of `build` as an unhandled RuntimeError and
            take the whole programme with it.
            """
            try:
                if policy_k != a.idle_policy:
                    print(f"\n{res['name']} is not the last pass, so it goes "
                          "home at the end: the pass after it starts from the "
                          "ready pose")
                    allocate.resequence(res, q_start=q_start, return_home=True)
                elif k and q_start is not None \
                        and a.idle_policy != idle.POLICY_HOME:
                    print(f"\nre-sequencing {res['name']} from the poses pass "
                          f"{k} froze in (the allocation is untouched)")
                    allocate.resequence(res, q_start=q_start,
                                        return_home=False)
            except (SystemExit, idle.Unconductable, RuntimeError) as exc:
                print(f"  !! {res['name']} could not even be re-sequenced: "
                      f"{exc}")
                return False
            return True

        def conduct(res):
            try:
                return build_phase(a, res, dt, pens, q_start=q_start,
                                   policy=policy_k)
            except (SystemExit, idle.Unconductable, RuntimeError) as exc:
                print(f"  !! {res['name']} could not be conducted as allocated: "
                      f"{exc}")
                return None

        B = conduct(ph) if prepare(ph) else None
        other = (alt or {}).get(k) if isinstance(alt, dict) else \
            (alt[k] if alt and k < len(alt) else None)
        if other is not None and not prepare(other):
            other = None
        if other is not None:
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
            # A PHASE IS NOT A RUN.  With `--arm-phases` a phase is one group
            # of arms drawing part of the same picture, and a group the
            # conductor cannot schedule does not make the groups that CONDUCT
            # unsafe — it makes their ink the only ink there is.  Shipping them
            # is the honest floor (the programme is certified for what it
            # draws, and the coverage says what the refusal cost); refusing the
            # whole run is the honest default, because a picture missing a
            # quarter of itself is usually not the picture that was asked for.
            if getattr(a, "skip_unconductable", False):
                ph["conducted"] = False
                print(f"  !! SKIPPING {ph['name']}: "
                      f"{ph['drawn_len']:.4f} m of ink nobody will draw "
                      "(--skip-unconductable)")
                continue
            raise SystemExit(f"{ph['name']} could not be conducted")
        ph["conducted"] = True
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
    if not built:
        raise SystemExit("no phase could be conducted")
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

    # THE PALETTE TRAVELS WITH THE PAYLOAD.  The drake demo builds one pen per
    # (arm, ink) and switches between them at the swap, so it has to know which
    # inks exist and what colour each one is — and on an arbitrary picture that
    # is measured off the source by the tracer, not looked up in a table.  Two
    # extra arrays here are what let the demo stop hard-coding grey and orange
    # without it having to import the tracer or read the program JSON.
    pal = getattr(a, "palette", None) or INK
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
    used_inks = []
    for B in built:
        for c in ([B["res"]["ink"]] if B["res"]["ink"]
                  else sorted(set(B["res"]["colors"].values()))):
            if c and c not in used_inks:
                used_inks.append(c)
    used_inks = used_inks or ["grey"]
    d["ink_names"] = np.array(used_inks)
    d["ink_palette"] = np.array([artwork.hex_of(c, pal) for c in used_inks])

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
    d["ink_hex"] = np.array([artwork.hex_of(c, pal) for c in ink_col])
    np.savez_compressed(out, **d)
    return nF, len(ink_t), M_tot, n_pause


def schedule_args(ap):
    """The conducting + animation arguments, shared with `scripts/draw.py`.

    Split out of `main` for the same reason `csail_allocate.add_args` was: the
    generic front door conducts with THIS conductor and must be able to be
    handed every knob it has, and a second copy of the list would drift.
    """
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
    ap.add_argument("--image-jobs", type=int, default=None,
                    help="processes the conductor builds its collision images "
                         "on (default: one per core).  1 is the serial build "
                         "every number published before 2026-08-25 was "
                         "measured on; the images are the same images either "
                         "way, so this changes wall clock and nothing else")
    ap.add_argument("--search-max-n", type=int,
                    default=coordination.PRIORITY_SEARCH_MAX,
                    help="moving arms below which EVERY priority order is "
                         "enumerated.  The prefix walk is 64 DP solves at four "
                         "moving arms and up to 1956 at six, so on a six-arm "
                         "rig this is the knob between an exhaustive search and "
                         "busiest-first with promotion on refusal")
    ap.add_argument("--pause", type=float, default=2.0,
                    help="seconds of every-arm-parked between two passes, "
                         "while a human swaps the pens")
    ap.add_argument("--skip-unconductable", action="store_true",
                    help="ship the phases that DO conduct instead of refusing "
                         "the whole run, and count the rest as ink nobody "
                         "drew.  Only meaningful with --arm-phases, where a "
                         "phase is one group of arms: the programme that comes "
                         "out is certified for what it draws and its coverage "
                         "says what that cost")
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
    ap.add_argument("--residual-passes", type=int, default=0,
                    metavar="N",
                    help="after the last pass, allocate WHAT IT LEFT EMPTY as "
                         "a further pass, up to N times.  A span an arm can "
                         "ink and can fly to from its depot is still dropped "
                         "when no Hamiltonian path threads it into the rest of "
                         "that arm's bag (allocate.prune_unflyable), and a "
                         "second pass is a second tour from the same depot.  "
                         "Costs a go-home and a --pause per pass and adds "
                         "certified ink; 0 (the default) is the single-tour "
                         "programme every published number was measured on")
    ap.add_argument("--min-coverage", type=float, default=0.0,
                    help="refuse to write a payload below this certified "
                         "coverage (fraction of traced metres)")
    ap.add_argument("--require-full", action="store_true",
                    help="shorthand for --min-coverage 1.0")
    # ---- the execution profile (section 5) ------------------------------
    # OPT-IN, because --qd-frac and --cluster are arguments and a script that
    # ignored the flags it was handed would be worse than one that does not
    # search: `scripts/speed_sweep.py` conducts NAMED cells of exactly this
    # grid and has to keep getting the cell it asked for.
    ap.add_argument("--select-profile", action="store_true",
                    help="conduct the four execution profiles (qd_frac 0.30 / "
                         "0.60 x cluster off / on) and ship the fastest one "
                         "scene_check certifies, recording all four outcomes.  "
                         "Overrides --qd-frac, --cluster and --band-objective")
    ap.add_argument("--no-profile-prune", action="store_true",
                    help="conduct every profile even when its floor already "
                         "says it cannot win (see csail_schedule.profile_floor)")
    ap.add_argument("--first-profile", default=FIRST_PROFILE,
                    help="the execution profile allocated, conducted and "
                         "certified BEFORE the others, so that a runnable "
                         "programme exists as early as possible.  The rest run "
                         "against its certified makespan and replace it only if "
                         "one of them is faster.  Give a profile name "
                         "('qd0.60+cluster') or anything else to keep the "
                         "listed order")
    ap.add_argument("--conduct-jobs", type=int, default=None,
                    help="processes the REMAINING profiles are conducted on "
                         "(default: all of them at once).  They cannot prune "
                         "each other — only the provisional best prunes them — "
                         "so conducting them together costs the longest one "
                         "instead of the sum.  1 conducts them serially in "
                         "floor order, which is what every published number was "
                         "measured on")
    ap.add_argument("--profile-jobs", type=int, default=1,
                    help="processes to ALLOCATE the four execution profiles on. "
                         "They are independent and each is a pure function of "
                         "(args, profile), so this changes wall clock and "
                         "nothing else.  USUALLY A LOSS NOW, and left at 1 for "
                         "that reason: run in one process the four cells share "
                         "the plan bag (allocate.plan_family) and the route memo "
                         "that most of an allocation is made of, and a worker "
                         "may not fork the route screen's own pool at all — so "
                         "each of the four pays serially for geometry the "
                         "shared run buys once (see docs/FAST_PLANNING.md)")
    ap.add_argument("--program", action="store_true",
                    help="also write out/csail_program<tag>.json from the "
                         "allocation that SHIPPED, so the programme and the "
                         "schedule cannot describe different runs")
    ap.add_argument("--final", default=None)
    ap.add_argument("--verbose", action="store_true")
    return ap


def main(argv=None):
    ap = schedule_args(add_args(argparse.ArgumentParser()))
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    phases, strokes, info, built, dt, pens, sel = build(a)
    prof = profile_json(sel) if sel else None
    if prof:
        # the shipped profile's own knobs, so that everything written below
        # reports the run that happened and not the run that was asked for
        a.qd_frac, a.cluster = prof["qd_frac"], prof["cluster"]
        a.band_objective = prof["band_objective"]
    path = out / f"csail_schedule{a.tag}.npz"
    nF, nInk, M_tot, n_pause = payload(built, dt, pens, a, path)
    pal = getattr(a, "palette", None)
    nm = getattr(a, "name", None) or "CSAIL logo"
    if a.program:
        doc = program_json(phases, strokes, info,
                           out / f"csail_program{a.tag}.json",
                           palette=pal, name=nm, source=a.image)
        if prof:
            doc["profile"] = prof
            (out / f"csail_program{a.tag}.json").write_text(json.dumps(doc))
        print(f"wrote {out}/csail_program{a.tag}.json "
              f"({(out / f'csail_program{a.tag}.json').stat().st_size / 1e3:.0f} kB)")
    if a.final:
        final_png(phases, strokes, a.final, palette=pal, name=nm)
    print(f"\nwrote {path} ({path.stat().st_size / 1e6:.1f} MB): {nF} frames @ "
          f"{a.fps:g} fps = {(nF - 1) / a.fps:.1f} s, {nInk} ink chunks"
          + (f", including {n_pause * dt:.1f} s of pen-swap pause"
             if n_pause else "")
          + (f"; {a.final}" if a.final else ""))
    print(f"SCHEDULE WALL CLOCK {time.time() - t0:.1f} s")

    summary = summary_json(a, phases, strokes, info, built, dt, pens, prof,
                           nF, nInk, n_pause)
    (out / f"csail_schedule{a.tag}.json").write_text(json.dumps(summary, indent=1))
    return summary


def summary_json(a, phases, strokes, info, built, dt, pens, prof, nF, nInk,
                 n_pause):
    """The run's own record. -> dict, ready for `json.dumps`.

    Split out of `main` so `scripts/draw.py` writes the IDENTICAL document under
    its own name: the drake demo reads this file for its legend and every
    downstream reader (docs, `scripts/bench.py`, the README tables) knows this
    shape, so a generic front door that invented a second one would be a second
    schema to keep in step.
    """
    T = totals(phases)
    # ...and what was actually CONDUCTED, which is the same thing unless a
    # phase was skipped (`--skip-unconductable`): the ink of a phase nobody
    # could schedule is ink nobody draws, and the coverage this run reports
    # has to say so rather than counting the allocation's intention.
    drawn = float(sum(B["res"]["drawn_len"] for B in built))
    skipped = [p for p in phases if p.get("conducted") is False]
    T = dict(T, drawn=drawn, dropped=T["traced"] - drawn,
             covered=drawn / max(T["traced"], 1e-9),
             n_segments=int(sum(len(B["res"]["programs"][x])
                                for B in built for x in B["res"]["arms"])))
    summary = dict(
        profile=prof, qd_frac=float(a.qd_frac),
        skipped_phases=[p["name"] for p in skipped],
        skipped_m=float(sum(p["drawn_len"] for p in skipped)),
        name=getattr(a, "name", None), source=getattr(a, "image", None),
        inks=artwork.inks_of(strokes),
        palette={k: artwork.hex_of(k, getattr(a, "palette", None) or INK)
                 for k in artwork.inks_of(strokes)},
        cluster=bool(getattr(a, "cluster", False)),
        band_objective=getattr(a, "band_objective", None),
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
        rig=fleet_mod.ACTIVE_RIG, sheet=[float(SHEET[0]), float(SHEET[1])],
        arms=[int(x) for x in sorted(FLEET)],
        logo=dict(w=info["logo_w"], h=info["logo_h"],
                  rotate_deg=float(info.get("rotate_deg", 0.0)),
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
            paper_clearance={str(k): v for k, v in
                             rep.get("paper_clearance", {}).items()},
            paper_failed=[int(x) for x in rep.get("paper_failed", [])],
            frame_failed=[int(x) for x in rep.get("frame_failed", [])],
            # every OTHER arm's base column, including the arms this phase
            # left at the depot (scene_check's own gate)
            column_clearance={str(k): v for k, v in
                              rep.get("column_clearance", {}).items()},
            column_failed=[int(x) for x in rep.get("column_failed", [])],
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
            # the reconfiguration the transit floors hide (`sequence.reconfiguration`)
            arm_reconfig_rad={str(x): float(sequence.reconfiguration(
                res["programs"][x])) for x in res["arms"]},
            arm_variants={str(x): list(res["sequence"][x].get("variants", []))
                          for x in res["arms"]},
            menu_stats={str(x): (res.get("menu_stats") or {}).get(x)
                        for x in res["arms"]},
            colors={str(k): v for k, v in res["colors"].items()}))
    # what the old single-phase readers look for, pointed at phase 1
    summary["reconfig_rad"] = float(sum(
        sum(ph["arm_reconfig_rad"].values()) for ph in summary["phases"]))
    summary["transit_s"] = float(sum(
        sum(ph["arm_transit_s"].values()) for ph in summary["phases"]))
    summary.update(pauses=summary["phases"][0]["pauses"],
                   priority=summary["phases"][0]["priority"],
                   arm_metres=summary["phases"][0]["arm_metres"],
                   arm_segments=summary["phases"][0]["arm_segments"],
                   per_pair=summary["phases"][0]["per_pair"],
                   colors=summary["phases"][0]["colors"])
    return summary


if __name__ == "__main__":
    main()
