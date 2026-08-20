#!/usr/bin/env python3
"""How much of the run is ONE arm drawing alone, and did it have to be?

    python3 scripts/solo_time.py --arms all --two-pass \
        --pens 2:300,31:200,71:200,97:200 --max-probes 5 \
        --target-width 1.2969246423461636 --offset 0.1 0.05 \
        --schedule out/csail_schedule_full.npz

Read-only.  Nothing here re-plans the piece or re-conducts it: the timeline is
read back out of the shipped npz, and the allocation is re-run only to recover
the one thing the pipeline computes and then throws away — FOR EVERY DRAWN SPAN,
THE SET OF ARMS THAT COULD HAVE DRAWN IT.  `allocate.rebalance` builds that set
(probe coverage plus a clean re-plan that gives back not one millimetre), uses
it, and returns a count.  This rebuilds it span by span and keeps it.

Three questions, in order:

  1. CONCURRENCY.  At each instant, how many arms have a pen down?  The
     time-share of 0/1/2/3/4 is what "six arms" is actually worth.

  2. WHY SOLO.  Every stretch with exactly one pen down is either
       STRUCTURAL   — no other arm of that colour certifies the span, so no
                      allocator could have shared it and only a different
                      PLACEMENT would change the answer; or
       ALLOCATABLE  — somebody else certifies it and the balancer left it where
                      it was.  The balancer's only move is a WHOLE segment, so a
                      span it cannot hand over without making the receiver the
                      new busiest arm is a span it must leave alone even when
                      half of it would have fitted.

  3. WHAT A SPLIT WOULD BUY.  With spans divisible, the min-max load problem has
     a closed form: for every subset A of arms, the work that can ONLY be done
     inside A must fit inside A, so

         T* = max over non-empty A of ( W(A) + O(A) ) / |A|

     where W(A) is the ink seconds of every span whose eligible set is contained
     in A and O(A) the fixed overhead of the arms in A.  (It is the min-cut of
     the obvious transportation network, read off directly — with six arms there
     are 63 subsets, so there is no search.)  Taking each span's ink time to be
     the FASTEST eligible arm's makes it a lower bound on any assignment,
     splitting or not.
"""
import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from aris_sixarm import allocate, writing                      # noqa: E402
from aris_sixarm.fleet import FLEET, H_INV_DEFAULT             # noqa: E402
from csail_allocate import add_args, run_allocation            # noqa: E402


# ==========================================================================
# 1. who has a pen down, when
# ==========================================================================
def concurrency(d):
    """-> dict(shares, per_phase, drawing (F,), n (F,), dt, arms)."""
    fps = float(d["fps"])
    arms = [int(x) for x in d["arms"]]
    seg = np.stack([d[f"seg_{a}"] for a in arms])          # (A, F)
    down = seg >= 0
    n = down.sum(axis=0)
    ph = d["phase"]
    out = dict(fps=fps, arms=arms, n=n, down=down, phase=ph, dt=1.0 / fps)
    for name, m in [("all", np.ones(len(n), bool))] + \
                   [(f"phase{k + 1}", ph == k) for k in range(int(d["n_phases"]))]:
        tot = int(m.sum())
        out[name] = dict(seconds=tot / fps,
                         share={int(k): float((n[m] == k).mean())
                                for k in range(len(arms) + 1)
                                if (n[m] == k).any()},
                         sec={int(k): float((n[m] == k).sum() / fps)
                              for k in range(len(arms) + 1)
                              if (n[m] == k).any()})
    return out


def solo_stretches(c, min_frames=1):
    """Maximal runs of "exactly one pen down". -> [dict(t0, t1, arm, phase, segs)]."""
    n, down, ph = c["n"], c["down"], c["phase"]
    arms, fps = c["arms"], c["fps"]
    solo = n == 1
    out, i = [], 0
    F = len(n)
    while i < F:
        if not solo[i]:
            i += 1
            continue
        j = i
        who = int(np.argmax(down[:, i]))
        while j < F and solo[j] and int(np.argmax(down[:, j])) == who \
                and ph[j] == ph[i]:
            j += 1
        if j - i >= min_frames:
            out.append(dict(t0=i / fps, t1=j / fps, frames=int(j - i),
                            seconds=float((j - i) / fps), arm=arms[who],
                            phase=int(ph[i]), i0=i, i1=j))
        i = j
    return out


# ==========================================================================
# 2. who ELSE could have drawn each span
# ==========================================================================
def _sig(pts, nd=5):
    """A drawn span's identity, independent of who draws it or which way round.

    The re-run of the allocation is deterministic but its ORDER is not the
    shipped one — the second pass is sequenced from wherever the first pass
    froze, which this read-only script does not reproduce — so a segment cannot
    be found by its index in a programme.  Its geometry, on the other hand, is
    the same polyline whichever end you start from, and that is what the npz
    carries.  Endpoints (unordered) plus length to 10 um: the spans in this
    piece are centimetres apart, so a collision is not a near thing.
    """
    pts = np.asarray(pts, float)
    a = (round(float(pts[0][0]), nd), round(float(pts[0][1]), nd))
    b = (round(float(pts[-1][0]), nd), round(float(pts[-1][1]), nd))
    n = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
    return (min(a, b), max(a, b), round(n, nd))


def _clip_cover(ivs, s0, s1, eps=1e-9):
    """Merge intervals and clip them to [s0, s1]. -> [(a, b)] disjoint, sorted."""
    got = sorted((max(u, s0), min(v, s1)) for u, v in ivs
                 if min(v, s1) - max(u, s0) > eps)
    out = []
    for u, v in got:
        if out and u <= out[-1][1] + eps:
            out[-1] = (out[-1][0], max(out[-1][1], v))
        else:
            out.append((u, v))
    return out


def eligibility(res, draw_speed, qd_frac, h_inv=H_INV_DEFAULT, verbose=True):
    """Re-probe one phase. -> {(arm, k): dict(arms, draw_s, length, stroke)}.

    `arms` is the set that certifies the span end to end — the same two tests
    `allocate.rebalance` applies (`frac_covers` over the probe intervals, then a
    clean re-plan that must not lose a millimetre), applied to the SHIPPED
    segments instead of to the pre-balance ones.  `draw_s[b]` is what arm b
    would need for the ink, from `writing.segment_draw_time`, which is the
    number the timeline pays and the balancer scored on.
    """
    ivmap, colors = res["ivmap"], res["colors"]
    by_id = {s["id"]: s for s in res["strokes"]}
    out, n_replan = {}, 0
    for a in res["arms"]:
        for k, ent in enumerate(res["programs"][a]):
            st = by_id[ent["stroke_id"]]
            s0, s1 = min(ent["s_range"]), max(ent["s_range"])
            sp = dict(s0=float(s0), s1=float(s1), direction=int(ent["direction"]))
            cand = {a: ent}
            for b in res["arms"]:
                if b == a or colors.get(b) != st["color"]:
                    continue
                ivs = [v for v in ivmap.get(st["id"], []) if v.arm == b]
                if not allocate.frac_covers(ivs, sp["s0"], sp["s1"]):
                    continue
                e, n = allocate.replan_same_span(st, sp, FLEET[b],
                                                 res["aopts"][b])
                n_replan += n
                if e is not None:
                    cand[b] = e
            # ...and, separately, WHICH PART of the span each other arm reaches.
            # An arm that certifies 60 % of a span is invisible to a balancer
            # whose only move is the whole segment, and is exactly the arm a
            # splitting one would use.
            part = {}
            for b in res["arms"]:
                if b == a or colors.get(b) != st["color"]:
                    continue
                cov = _clip_cover([(v.s0, v.s1) for v in ivmap.get(st["id"], [])
                                   if v.arm == b], s0, s1)
                if cov:
                    part[b] = cov
            out[(a, k)] = dict(
                arms=sorted(cand), length=float(ent["length"]),
                sig=_sig(np.asarray(ent["plan"]["pts"], float)),
                stroke=int(ent["stroke_id"]), s_range=(float(s0), float(s1)),
                part={b: [(float(u), float(v)) for u, v in cov]
                      for b, cov in part.items()},
                part_frac={b: float(sum(v - u for u, v in cov) / max(s1 - s0, 1e-12))
                           for b, cov in part.items()},
                draw_s={b: float(writing.segment_draw_time(
                    FLEET[b], e, draw_speed, qd_frac, h_inv, res["pens"][b]))
                    for b, e in cand.items()})
    if verbose:
        print(f"  re-probed {len(out)} spans with {n_replan} clean re-plans; "
              f"{sum(1 for v in out.values() if len(v['arms']) > 1)} have an "
              "alternative arm")
    return out


# ==========================================================================
# 3. what a split-capable balancer could reach
# ==========================================================================
def atoms_of(elig, eps=1e-9):
    """Cut every span at every certified boundary. -> [(seconds, frozenset(arms))].

    The balancer's unit is the whole span; a SPLITTING balancer's unit is
    whatever piece of it somebody else can reach, and the probe data already
    says where those pieces start and stop (`allocate.Interval`).  Cutting each
    span at the union of its neighbours' interval endpoints gives the atoms:
    inside one atom the eligible set is constant, so the same subset bound
    applies with no further search.

    Ink seconds are charged at the OWNER's own rate for that span (its
    `segment_draw_time` per metre), which is the timeline's rate for that ink;
    what a different arm's redundancy resolution would do to it is not knowable
    without planning the piece, and the difference is second-order next to the
    length.
    """
    out = []
    for (a, k), v in elig.items():
        s0, s1 = v["s_range"]
        span = max(s1 - s0, 1e-12)
        rate = v["draw_s"][a] / span                  # seconds per unit of s
        cuts = {s0, s1}
        for cov in v["part"].values():
            for u, w in cov:
                cuts.update((u, w))
        cuts = sorted(x for x in cuts if s0 - eps <= x <= s1 + eps)
        for u, w in zip(cuts, cuts[1:]):
            if w - u <= eps:
                continue
            mid = 0.5 * (u + w)
            who = {a} | {b for b, cov in v["part"].items()
                         if any(x - eps <= mid <= y + eps for x, y in cov)}
            out.append(((w - u) * rate, frozenset(who)))
    return out


def subset_floor(work, arms, overhead=None):
    """max over non-empty A of (W(A) + O(A)) / |A|, the transportation min-cut."""
    over = overhead or {}
    best, binding = 0.0, ()
    for r in range(1, len(arms) + 1):
        for A in combinations(sorted(arms), r):
            S = set(A)
            W = sum(w for w, who in work if who <= S)
            if W <= 0:
                continue
            val = (W + sum(float(over.get(b, 0.0)) for b in A)) / len(A)
            if val > best:
                best, binding = val, A
    return dict(floor_s=float(best), binding=list(binding))


def split_floor(elig, overhead=None):
    """Closed-form min-max load when spans are divisible. -> dict.

    The transportation min-cut, enumerated over subsets: work whose eligible set
    lies inside A has nowhere to go but A.  `overhead[a]` is arm a's fixed
    non-ink time (its pen-up tour); pass None for the pure-ink floor.
    """
    arms = sorted({b for v in elig.values() for b in v["arms"]})
    work = [(min(v["draw_s"][b] for b in v["arms"]), frozenset(v["arms"]))
            for v in elig.values()]
    return subset_floor(work, arms, overhead)


def main(argv=None):
    ap = add_args(argparse.ArgumentParser())
    ap.add_argument("--schedule", default=str(ROOT / "out/csail_schedule_full.npz"))
    ap.add_argument("--summary", default=str(ROOT / "out/csail_schedule_full.json"))
    ap.add_argument("--json", default=str(ROOT / "out/solo_time.json"))
    ap.add_argument("--tag", default="_full")
    a = ap.parse_args(argv)

    d = np.load(a.schedule, allow_pickle=False)
    S = json.loads(Path(a.summary).read_text())
    c = concurrency(d)
    print(f"{a.schedule}: {S['duration']:.1f} s, {len(c['arms'])} arms, "
          f"{int(d['n_phases'])} phases")
    for name in ["all"] + [f"phase{k + 1}" for k in range(int(d["n_phases"]))]:
        r = c[name]
        print(f"  {name:>7} {r['seconds']:6.1f} s: "
              + "  ".join(f"{k} drawing {100 * v:5.1f} % ({r['sec'][k]:5.1f} s)"
                          for k, v in sorted(r["share"].items())))

    stretches = solo_stretches(c)
    print(f"\n{len(stretches)} solo stretches, "
          f"{sum(s['seconds'] for s in stretches):.1f} s in total")

    t0 = time.time()
    phases, strokes, info = run_allocation(a, verbose=False)
    print(f"\nre-ran the allocation in {time.time() - t0:.0f} s "
          f"({len(phases)} phases)")
    elig = []
    for k, res in enumerate(phases):
        print(f"  {res['name']}:")
        elig.append(eligibility(res, a.draw_speed, a.qd_frac))

    # ---- classify every solo second -------------------------------------
    by_sig = [{v["sig"]: v for v in E.values()} for E in elig]
    per = {"structural": 0.0, "splittable": 0.0, "allocatable": 0.0,
           "unattributed": 0.0}
    rows = []
    for s in stretches:
        k, arm = s["phase"], s["arm"]
        g = d[f"seg_{arm}"][s["i0"]:s["i1"]]
        off, pts = d[f"segoff_{arm}"], d[f"segpts_{arm}"]
        secs = {}
        for gi in np.unique(g[g >= 0]):
            loc = int(gi)
            v = (by_sig[k].get(_sig(pts[off[loc]:off[loc + 1]]))
                 if loc + 1 < len(off) else None)
            n = float((g == gi).sum()) / c["fps"]
            kind = ("unattributed" if v is None else
                    "allocatable" if len(v["arms"]) > 1 else
                    "splittable" if v["part"] else "structural")
            secs[kind] = secs.get(kind, 0.0) + n
            per[kind] = per.get(kind, 0.0) + n
            rows.append(dict(phase=k, arm=arm, seg=loc, kind=kind, seconds=n,
                             alt=[] if v is None else
                             [x for x in v["arms"] if x != arm],
                             part={} if v is None else
                             {str(b): round(f, 3)
                              for b, f in v["part_frac"].items()},
                             length=None if v is None else v["length"]))
        s["kinds"] = secs
    tot_solo = sum(per.values())
    print(f"\nsolo seconds by cause: "
          + "  ".join(f"{k} {v:.1f} s ({100 * v / max(tot_solo, 1e-9):.0f} %)"
                      for k, v in per.items() if v))

    out = dict(schedule=a.schedule, duration=float(S["duration"]),
               shares={n: c[n] for n in ["all"] + [f"phase{k + 1}" for k in
                                                   range(int(d["n_phases"]))]},
               solo_total_s=float(tot_solo), solo_by_cause=per,
               stretches=[{k: v for k, v in s.items() if k not in ("i0", "i1")}
                          for s in stretches], spans=rows, phases=[])
    for name in out["shares"]:
        out["shares"][name] = dict(seconds=c[name]["seconds"],
                                   share=c[name]["share"], sec=c[name]["sec"])

    # ---- the split-capable floor, per phase ------------------------------
    for k, res in enumerate(phases):
        ph = S["phases"][k]
        over = {int(x): float(v) for x, v in ph["arm_transit_s"].items()}
        ink = split_floor(elig[k])
        both = split_floor(elig[k], over)
        at = atoms_of(elig[k])
        arms_k = sorted({b for v in elig[k].values() for b in v["arms"]}
                        | {b for v in elig[k].values() for b in v["part"]})
        a_ink = subset_floor(at, arms_k)
        a_both = subset_floor(at, arms_k, over)
        movable = sum(1 for v in elig[k].values() if len(v["arms"]) > 1)
        splittable = [v for v in elig[k].values()
                      if len(v["arms"]) == 1 and v["part"]]
        print(f"\n{res['name']}: makespan {ph['duration_s']:.1f} s, floor "
              f"{ph['floor_s']:.1f} s; {movable}/{len(elig[k])} spans have an "
              f"alternative arm for ALL of them, {len(splittable)} for part "
              f"({sum(v['length'] for v in splittable):.2f} m)")
        print(f"  whole-span floor: {ink['floor_s']:.1f} s of ink alone "
              f"(binding {ink['binding']}), {both['floor_s']:.1f} s with "
              "today's pen-up tours")
        print(f"  split-capable floor ({len(at)} atoms): {a_ink['floor_s']:.1f} s "
              f"of ink alone (binding {a_ink['binding']}), "
              f"{a_both['floor_s']:.1f} s with today's pen-up tours")
        out["phases"].append(dict(
            name=res["name"], makespan_s=float(ph["duration_s"]),
            floor_s=float(ph["floor_s"]), pause_s=float(ph["pause_total"]),
            n_spans=len(elig[k]), n_movable=int(movable),
            n_splittable=len(splittable),
            splittable_m=float(sum(v["length"] for v in splittable)),
            whole_floor_ink_s=ink["floor_s"], whole_floor_binding=ink["binding"],
            whole_floor_with_transit_s=both["floor_s"],
            n_atoms=len(at), split_floor_ink_s=a_ink["floor_s"],
            split_floor_binding=a_ink["binding"],
            split_floor_with_transit_s=a_both["floor_s"],
            arm_loads_s={str(x): float(ph["arm_nominal_s"][x])
                         for x in ph["arm_nominal_s"]}))
    Path(a.json).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {a.json}")
    return out


if __name__ == "__main__":
    main()
