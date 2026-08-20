#!/usr/bin/env python3
"""WHERE THE PAUSES COME FROM: a read-only autopsy of the conducted timeline.

    python3 scripts/concurrency_diag.py              # the shipped _full run
    python3 scripts/concurrency_diag.py --no-perms   # skip the 24 orders

The two-pass CSAIL run pays 90.6 s of pauses in a 63.5 s phase 1 and 57.4 s in
an 88.0 s phase 2.  `coordination.coordinate` reports those as one number per
arm, which says WHO waited but not WHY, and the improvement one would build
next depends entirely on why.  This script re-runs the conductor on the very
same frozen paths -- the allocation is deterministic, and the reconstruction is
ASSERTED against out/csail_schedule_full.json arm by arm, to the float, so a
number below is either the shipped run's or a crash -- and instruments it:

  SPLIT    a pause is one of two unrelated things.  EN ROUTE, the arm is held
           at a path index because the next cell is occupied.  REST HOLD, the
           arm could be at its last index by `m_reach` but may not ARRIVE until
           `m_end`, because arriving means standing in that final pose for the
           rest of the phase and the pose is not clear yet.  The first is fixed
           by getting out of each other's way; the second by not standing
           there.  Nothing that reports one number per arm can tell them apart.
  BLAME    the shipped backtrack places pauses as EARLY as it can, which is
           fine for running and useless for blaming, so each arm's own waits
           are re-walked to the instant they are forced (`asap_progress`) while
           every blocker keeps the schedule it shipped with -- same arrival,
           same collision image, a different reading of the same run.  A wait
           is charged to the arm whose swept clearance at the blocking cell is
           tightest, so the seconds add to the pause total exactly.
  DISTANCE both the margin test's own number (worst corner clearance minus the
           swept-motion slack) and the raw pose clearance, because the gap
           between them is how much of the pausing is conservatism rather than
           geometry.

and then re-runs the scheduler (never the planner) under counterfactuals:
margins, all 24 priority permutations, a 25 %-speed crawl, and arms that tuck
out of the way instead of standing where they stopped.

Writes out/concurrency_diagnostic.png + .json; docs/CONCURRENCY.md reads them.
"""
import argparse
import itertools
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.patches import Rectangle              # noqa: E402

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from aris_sixarm import coordination, writing         # noqa: E402
from aris_sixarm.fleet import FLEET, H_INV_DEFAULT    # noqa: E402
from csail_allocate import add_args, parse_pens, run_allocation   # noqa: E402

# the shipped run, exactly (README's --two-pass line plus the placement the
# commit message records as the winner; --max-probes 5 is what reproduces its
# 49 certified segments)
SHIPPED = ["--arms", "all", "--two-pass",
           "--pens", "2:300,31:200,71:200,97:200",
           "--target-width", "1.2969246423461636", "--offset", "0.1", "0.05",
           "--max-probes", "5"]
MARGINS = (0.080, 0.065, 0.050, 0.035)     # m; 0.080 = the shipped 0.05 + 0.03
CRAWL = 0.25                               # fraction of nominal speed in a crawl


# ==========================================================================
# geometry: one field per unordered pair, thresholded many times
# ==========================================================================
def pair_fields(paths, sweep=coordination.SWEEP_K, cap=coordination.BROAD_CAP):
    """{(i,j): (D4, si, sj)} for i < j -- everything a margin test ever needs.

    `free_cells` costs a clearance matrix and then throws it away behind a
    boolean.  Every counterfactual in this script is the SAME geometry read at
    a different threshold or with a different sweep slack, so the corner
    clearance D4 and the two per-arm step lengths are kept and the boolean is
    made on demand.  That is what makes four margins and twenty-four priority
    orders cost one geometry pass between them.
    """
    ids = sorted(paths)
    out = {}
    for x, i in enumerate(ids):
        for j in ids[x + 1:]:
            D = coordination.clearance_matrix(paths[i], paths[j], cap)
            D4 = np.minimum(np.minimum(D[:-1, :-1], D[1:, :-1]),
                            np.minimum(D[:-1, 1:], D[1:, 1:]))
            out[(i, j)] = (D4, paths[i].step.copy(), paths[j].step.copy())
    return out


def swept(fields, a, b, sweep=coordination.SWEEP_K, k_a=1.0, k_b=1.0):
    """(na-1, nb-1) swept clearance S for arm `a` against arm `b`.

    `k_a`/`k_b` scale each arm's swept-motion slack: k = CRAWL is "this arm
    creeps through the conflict", which is the ONLY thing a velocity schedule
    buys geometrically -- a slower link sweeps a thinner tube, so a cell that
    is refused for the tube and not for the pose becomes usable.
    """
    if (a, b) in fields:
        D4, sa, sb = fields[(a, b)]
        return D4 - sweep * (k_a * sa[:, None] + k_b * sb[None, :])
    D4, sb, sa = fields[(b, a)]
    return D4.T - sweep * (k_a * sa[:, None] + k_b * sb[None, :])


# ==========================================================================
# the conductor, re-implemented so the DP internals are inspectable
# ==========================================================================
def okf_for(a, blockers, prog, free, n, M):
    """(n, M) bool: may arm `a` occupy cell p at step m, given fixed blockers."""
    ok = np.ones((n - 1, M), bool)
    for b in blockers:
        F = free[(a, b)]
        ok &= F[:, np.clip(prog[b], 0, F.shape[1] - 1)]
    return np.vstack([ok, ok[-1:]])           # index n-1 sits in cell n-2


def dp_arrive(okf, n, M, rest_free=False):
    """Earliest arrival for one arm. -> (m_end, reach, rest) or (None, ...)."""
    rest = (np.ones(M, bool) if rest_free
            else np.logical_and.accumulate(okf[n - 1, ::-1])[::-1])
    reach = np.zeros((n, M), bool)
    reach[0, 0] = True
    for m in range(1, M):
        av = reach[:, m - 1] & okf[:, m - 1]
        col = av.copy()
        col[1:] |= av[:-1]
        reach[:, m] = col
        if reach[n - 1, m] and rest[m]:
            break
    hits = np.flatnonzero(reach[n - 1] & rest)
    if not len(hits):
        return None, reach, rest
    return int(hits[0]), reach, rest


def dp_backtrack(reach, okf, n, m_end, M):
    """`coordination._dp`'s own backtrack, restated: pauses land as EARLY as
    they can.  This is the schedule that shipped, and the one every arm after
    this one was conducted around, so the reconstruction has to use it."""
    prog = np.full(M, n - 1, int)
    p, m = n - 1, m_end
    while m > 0:
        if p > 0 and reach[p - 1, m - 1] and okf[p - 1, m - 1]:
            p -= 1
        elif reach[p, m - 1] and okf[p, m - 1]:
            pass
        else:
            raise RuntimeError("backtrack fell off the reachable set")
        m -= 1
        prog[m] = p
    return prog


def asap_progress(okf, n, M, m_end):
    """The SAME arrival, with every pause pushed to the instant it is forced.

    Backward feasibility G[p, m] ("from cell p at step m, can this arm still be
    at the last index by m_end") turns the schedule into a forward walk that
    advances whenever advancing keeps the deadline reachable.  A wait step in
    THIS walk is a wait the geometry demanded, which is what makes it blamable;
    the shipped backtrack's waits sit wherever the recursion happened to leave
    them.  Arrival is m_end either way, and the arm's OWN okf is unchanged (it
    is a function of the blockers, whose schedules stay exactly as shipped), so
    re-walking is a re-reading of the same run and not a different one.
    """
    G = np.zeros((n, m_end + 1), bool)
    G[n - 1, m_end] = True
    for m in range(m_end - 1, -1, -1):
        nxt = G[:, m + 1].copy()
        nxt[:-1] |= G[1:, m + 1]              # advancing lands one index higher
        G[:, m] = okf[:, m] & nxt
    prog = np.full(M, n - 1, int)
    p = 0
    for m in range(m_end):
        if p < n - 1 and G[p + 1, m + 1]:
            p += 1
        prog[m + 1] = p
    return prog


def make_free(paths, fields, margin, ids, sweep=coordination.SWEEP_K,
              crawl=None, vanish_finished=False):
    """{(a, b): free-cell image} for every ordered pair of `ids`."""
    free = {}
    for a in ids:
        for b in ids:
            if a == b:
                continue
            F = swept(fields, a, b, sweep, k_a=(crawl or 1.0), k_b=1.0) >= margin
            if vanish_finished:
                F[:, -1] = True               # b at its last index has tucked
            free[(a, b)] = F
    return free


def conduct(paths, fields, margin, order=None, sweep=coordination.SWEEP_K,
            drop=(), rest_free=False, vanish_finished=False, crawl=None,
            horizon_mult=3.0, retry=True, free=None):
    """One scheduling run. -> dict(progress, finish, pauses, order, ...) or None.

    A faithful restatement of `coordination.coordinate` -- same DP, same
    backtrack, same deadlock retry, and the priority order either handed in
    (`order`, which is how the shipped run is reproduced now that the conductor
    SEARCHES its order) or the old busiest-first guess -- with the images
    handed in rather than built, so a counterfactual is a re-threshold and not a
    re-derivation.  The knobs are the counterfactuals:

    `drop`               arms deleted from the world (tucked right out of it).
    `rest_free`          an arm that has arrived stops constraining ITSELF, and
                         `vanish_finished` does the same for it as a BLOCKER --
                         together, "every arm retreats to a tuck pose the
                         instant it finishes".
    `crawl`              scale on the moving arm's own swept-motion slack.
    """
    ids = [a for a in sorted(paths) if a not in drop]
    dt = paths[ids[0]].dt
    static = [a for a in ids if not paths[a].moves]
    moving = sorted([a for a in ids if paths[a].moves],
                    key=lambda a: (-paths[a].motion, a))
    if order is not None:
        moving = [a for a in order if a in moving]
    nom = {a: (paths[a].n - 1) * dt for a in ids}
    M = int(np.ceil(max(nom.values()) * horizon_mult / dt)) + 64
    if free is None:
        free = make_free(paths, fields, margin, ids, sweep, crawl,
                         vanish_finished)

    failed = None
    for _ in range(len(moving) + 1 if retry else 1):
        prog = {a: np.zeros(M, int) for a in static}
        finish, okfs, failed = {}, {}, None
        for k, a in enumerate(moving):
            blk = static + moving[:k]
            okf = okf_for(a, blk, prog, free, paths[a].n, M)
            m_end, reach, _ = dp_arrive(okf, paths[a].n, M, rest_free)
            if m_end is None:
                failed = a
                break
            prog[a] = dp_backtrack(reach, okf, paths[a].n, m_end, M)
            finish[a] = m_end * dt
            # m_reach: the earliest step the arm could BE at its last index at
            # all.  m_end - m_reach is the delay bought purely by the rest
            # requirement -- the arm may not arrive until the pose it will then
            # stand in is clear for good.  Splitting the two is the whole
            # diagnosis, so it is recorded at the source.
            m_reach = int(np.flatnonzero(reach[paths[a].n - 1])[0])
            okfs[a] = (okf, m_end, blk, m_reach)
        if failed is None:
            break
        if not retry or moving[0] == failed:
            return None
        moving = [failed] + [a for a in moving if a != failed]
    if failed is not None:
        return None
    for a in static:
        finish[a] = 0.0
    pauses = {a: float(finish[a] - nom[a]) for a in moving}
    return dict(progress=prog, finish=finish, nominal=nom, pauses=pauses,
                order=static + moving, moving=moving, parked=static,
                duration=float(max(finish.values())), margin=float(margin),
                dt=float(dt), M=M, okfs=okfs, free=free,
                pause_total=float(sum(pauses.values())))


# ==========================================================================
# blame
# ==========================================================================
def arm_state(aid, p, m, paths, samp, prog, moved_by):
    """What arm `aid` is doing at path index `p`, time step `m`."""
    if not paths[aid].moves:
        return "parked"
    if p >= paths[aid].n - 1:
        return "rest"
    if m < moved_by.get(aid, 0):
        return "prestart"
    return "draw" if samp[aid]["seg"][min(p, samp[aid]["n"] - 1)] >= 0 else "transit"


CAUSE = {("draw", "draw"): "draw-draw",
         ("draw", "transit"): "draw-transit",
         ("transit", "draw"): "draw-transit",
         ("transit", "transit"): "transit-transit"}


def blame(sch, paths, samp, fields, margin, sweep=coordination.SWEEP_K):
    """Every wait step of every arm, charged to the arm that caused it.

    A pause splits cleanly in two, and the two want completely different fixes:

      EN ROUTE   the arm is held at some path index because the next cell is
                 occupied.  Its own pauses are re-walked to the instant they
                 are forced (`asap_progress`, deadline `m_reach`), so the
                 blocker at that instant IS the reason.  The blockers' own
                 schedules stay exactly as shipped, so this is a re-reading of
                 the run, not a different run.
      REST HOLD  the arm could physically be at its last index by `m_reach` but
                 may not ARRIVE until `m_end`, because arriving means standing
                 in that final pose for the rest of the phase and the pose is
                 not clear yet.  No amount of creeping along the path helps: the
                 conflict is at the destination.  Every step of that window is
                 charged to whoever is not clear of the arm's parked pose.

    -> (events, steps, asap); an event is a maximal run of wait steps with the
    same tightest blocker.
    """
    dt, prog = sch["dt"], sch["progress"]
    asap = {a: prog[a] for a in prog}
    for a in sch["moving"]:
        okf, m_end, _, m_reach = sch["okfs"][a]
        asap[a] = asap_progress(okf, paths[a].n, sch["M"], m_reach)
    moved_by = {}
    for a in sch["moving"]:
        nz = np.flatnonzero(np.diff(asap[a]) > 0)
        moved_by[a] = int(nz[0]) + 1 if len(nz) else 0
    S = {}

    def Sab(a, b):
        if (a, b) not in S:
            S[(a, b)] = swept(fields, a, b, sweep)
        return S[(a, b)]

    def occupants(a, p, m, blk):
        """Which already-scheduled arms are not clear of cell p at step m."""
        out = []
        for b in blk:
            pb = int(min(prog[b][m], paths[b].n - 2))
            s = float(Sab(a, b)[p, pb])
            if s < margin:
                out.append((s, b, pb))
        return out

    def charge(a, at, m, blk, kind, cands, m_wall, p_wall):
        """One wait step, charged to the tightest arm in `cands`."""
        if not cands:
            return dict(arm=a, t=m * dt, m=m, p=at, blocker=a, S=float("nan"),
                        D4=float("nan"), n_blockers=0, kind=kind, ahead=0,
                        sa="transit", sb="unattributed")
        s, b, pb = min(cands)
        return dict(arm=a, t=m * dt, m=m, p=at, blocker=b, S=s, kind=kind,
                    D4=float(Sab(a, b)[p_wall, pb]
                             + sweep * (paths[a].step[p_wall] + paths[b].step[pb])),
                    n_blockers=len(cands), ahead=int(m_wall - m),
                    seg=int(samp[a]["seg"][min(at, samp[a]["n"] - 1)]),
                    seg_b=int(samp[b]["seg"][min(pb, samp[b]["n"] - 1)]),
                    sa=("draw" if samp[a]["seg"][min(at, samp[a]["n"] - 1)] >= 0
                        else "transit"),
                    sb=arm_state(b, pb, m_wall, paths, samp, asap, moved_by))

    steps = []
    for a in sch["moving"]:
        okf, m_end, blk, m_reach = sch["okfs"][a]
        n, P = paths[a].n, asap[a]
        for m in range(m_reach):               # ---- held on its path
            if P[m + 1] != P[m] or P[m] >= n - 1:
                continue
            p = min(P[m] + 1, n - 2)
            cands = occupants(a, p, m + 1, blk)
            pw, mw = p, m + 1
            if not cands:
                # NOT BLOCKED THIS INSTANT.  The DP refused the move because
                # every continuation from it dies later, so the honest answer to
                # "what is in the way" is what this arm would run into if it set
                # off now at full speed: walk the diagonal and name the first
                # cell somebody else is standing in.
                k = np.arange(1, min(n - 1 - p, m_end - m - 1))
                if len(k):
                    rows, cols = p + k, m + 1 + k
                    bad = np.flatnonzero(~okf[rows, cols])
                    if len(bad):
                        pw, mw = int(rows[bad[0]]), int(cols[bad[0]])
                        cands = occupants(a, pw, mw, blk)
            steps.append(charge(a, p, m + 1, blk, "route", cands, mw, pw))
        if m_end > m_reach:                    # ---- cannot park yet
            # `rest` is a suffix condition: the arm may not ARRIVE at m until
            # its final pose is clear at every step from m to the horizon.  The
            # binding instant is therefore the LAST step before m_end at which
            # that pose is occupied -- one wall, holding up the whole window,
            # and blaming each step of the window on whoever happens to be near
            # at that step would name arms that are merely passing through.
            bad = np.flatnonzero(~okf[n - 1, :m_end])
            mw = int(bad[-1]) if len(bad) else m_end - 1
            cands = occupants(a, n - 2, mw, blk)
            for m in range(m_reach, m_end):
                steps.append(charge(a, n - 2, m, blk, "rest", cands, mw, n - 2))
    for s in steps:
        s["cause"] = (
            "unattributed" if s["sb"] == "unattributed" else
            "own rest pose" if s["kind"] == "rest" else
            "blocker standing" if s["sb"] in ("rest", "parked", "prestart") else
            CAUSE[(s["sa"], s["sb"])])

    events, cur = [], None
    for s in steps:
        if (cur and s["arm"] == cur["arm"] and s["blocker"] == cur["blocker"]
                and s["kind"] == cur["kind"] and s["m"] - cur["m1"] <= 1):
            cur["m1"], cur["n"] = s["m"], cur["n"] + 1
            cur["S"] = np.fmin(cur["S"], s["S"])
        else:
            if cur:
                events.append(cur)
            cur = dict(arm=s["arm"], blocker=s["blocker"], m0=s["m"], m1=s["m"],
                       n=1, S=s["S"], cause=s["cause"], sa=s["sa"], sb=s["sb"],
                       t0=s["t"], p=s["p"], kind=s["kind"],
                       seg=s.get("seg", -1), seg_b=s.get("seg_b", -1))
    if cur:
        events.append(cur)
    for e in events:
        e["dur"] = e["n"] * dt
    return events, steps, asap


# ==========================================================================
# counterfactuals
# ==========================================================================
def crawl_bound(events, paths, dt):
    """Idealised delay if a blocked arm crept at CRAWL speed instead of stopping.

    An arm that keeps moving at 25 % of nominal covers a quarter of the path it
    would have covered anyway, so a stop of L steps becomes a delay of 0.75 L --
    OPTIMISTIC on purpose: it assumes the creep is admissible for the whole
    event and that the conflict clears on the same clock.  It is capped by the
    path the arm has left, because an arm cannot creep past its own last index.

    A REST HOLD is credited nothing, and that is not a detail.  The arm is not
    being held up along its path; it is being held out of the pose it wants to
    finish in.  Creeping the last centimetre more slowly still puts it there,
    so no velocity schedule can recover that second.
    """
    saved = defaultdict(float)
    for e in events:
        if e["kind"] != "route":
            continue
        left = paths[e["arm"]].n - 1 - e["p"]
        saved[e["arm"]] += min(e["n"] * (1.0 - CRAWL), left) * dt
    return saved


def realized_clearance(sch, paths, fields, sweep=coordination.SWEEP_K):
    """min over the schedule of the swept clearance -- what scene_check reports."""
    worst = np.inf
    ids = sorted(sch["progress"])
    for x, i in enumerate(ids):
        for j in ids[x + 1:]:
            S = swept(fields, i, j, sweep)
            pi = np.clip(sch["progress"][i][:sch["M"]], 0, S.shape[0] - 1)
            pj = np.clip(sch["progress"][j][:sch["M"]], 0, S.shape[1] - 1)
            worst = min(worst, float(S[pi, pj].min()))
    return worst


# ==========================================================================
# figure
# ==========================================================================
ARMC = {a: FLEET[a].color for a in FLEET}
CAUSEC = {"own rest pose": "#d62728", "blocker standing": "#e377c2",
          "draw-draw": "#1f77b4", "draw-transit": "#ff7f0e",
          "transit-transit": "#2ca02c", "unattributed": "#999999"}
CAUSEK = ["own rest pose", "blocker standing", "draw-draw", "draw-transit",
          "transit-transit"]


def gantt(ax, ph, title):
    """Per-arm bar: moving (pale) vs paused (blocker's colour).

    Solid = held on its path by the blocker; hatched = held OUT of its finished
    pose by the blocker.  The two look different because they are fixed by
    different machinery.
    """
    sch, paths = ph["sch"], ph["paths"]
    arms = sorted(sch["progress"])
    ymap = {a: k for k, a in enumerate(arms)}
    for a in arms:
        y = ymap[a]
        if not paths[a].moves:
            ax.add_patch(Rectangle((0, y - 0.3), ph["dur"], 0.6,
                                   fc="#e8e8e8", ec="none"))
            ax.text(ph["dur"] * 0.5, y, "parked at q_seed all phase — obstacle "
                    "for everyone", fontsize=7, ha="center", va="center",
                    color="#777")
            continue
        ax.add_patch(Rectangle((0, y - 0.3), sch["finish"][a], 0.6,
                               fc=ARMC[a], ec="none", alpha=0.30))
        ax.add_patch(Rectangle((sch["finish"][a], y - 0.13),
                               ph["dur"] - sch["finish"][a], 0.26,
                               fc="#c4c4c4", ec="none"))
    for e in ph["events"]:
        y = ymap[e["arm"]]
        x0, w = e["m0"] * sch["dt"], e["n"] * sch["dt"]
        if e["kind"] == "route":
            ax.add_patch(Rectangle((x0, y - 0.3), w, 0.6,
                                   fc=ARMC[e["blocker"]], ec="none"))
        else:
            ax.add_patch(Rectangle((x0, y - 0.3), w, 0.6, fc="none",
                                   ec=ARMC[e["blocker"]], lw=0.7,
                                   hatch="////"))
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels([f"arm {a}" + ("" if paths[a].moves else "  (parked)")
                        for a in arms], fontsize=8)
    ax.set_xlim(0, ph["dur"])
    ax.set_ylim(-0.6, len(arms) - 0.4)
    ax.invert_yaxis()
    ax.set_xlabel("t (s)", fontsize=8)
    ax.set_title(title, fontsize=9.5, loc="left")
    ax.tick_params(labelsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def pair_matrix(ax, phs, title):
    arms = sorted({a for ph in phs for a in ph["sch"]["progress"]})
    Mx = np.zeros((len(arms), len(arms)))
    idx = {a: k for k, a in enumerate(arms)}
    for ph in phs:
        for s in ph["steps"]:
            Mx[idx[s["arm"]], idx[s["blocker"]]] += ph["sch"]["dt"]
    im = ax.imshow(Mx, cmap="magma_r", vmin=0)
    for i in range(len(arms)):
        for j in range(len(arms)):
            if Mx[i, j] > 0.05:
                ax.text(j, i, f"{Mx[i, j]:.0f}", ha="center", va="center",
                        fontsize=8, color="white" if Mx[i, j] > Mx.max() * 0.55
                        else "black")
    ax.set_xticks(range(len(arms)))
    ax.set_yticks(range(len(arms)))
    ax.set_xticklabels(arms, fontsize=8)
    ax.set_yticklabels(arms, fontsize=8)
    ax.set_xlabel("blocked BY arm", fontsize=8)
    ax.set_ylabel("arm that waited", fontsize=8)
    ax.set_title(title, fontsize=9.5, loc="left")
    plt.colorbar(im, ax=ax, fraction=0.046, label="pause s")
    return Mx


def figure(phs, cf, path):
    fig = plt.figure(figsize=(17.5, 12.4))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.05],
                          hspace=0.42, wspace=0.30,
                          left=0.055, right=0.985, top=0.925, bottom=0.055)
    for k, ph in enumerate(phs):
        ax = fig.add_subplot(gs[k, 0:2])
        ideal = max(ph["sch"]["nominal"].values())
        gantt(ax, ph, f"{ph['name']} — makespan {ph['dur']:.1f} s against "
                      f"{ideal:.1f} s of perfect concurrency "
                      f"({ph['dur'] - ideal:.1f} s lost); "
                      f"{ph['sch']['pause_total']:.1f} s of pause "
                      f"(bar colour = the arm that blocked)")
        ax.axvline(ideal, color="#111", lw=1.1, ls="--")
        ax.annotate("longest arm's nominal time", (ideal, -0.55),
                    fontsize=7, ha="right", va="top", color="#111",
                    xytext=(-4, 0), textcoords="offset points")
        if k == 0:
            ax.legend(handles=[
                Rectangle((0, 0), 1, 1, fc="#777", ec="none"),
                Rectangle((0, 0), 1, 1, fc="none", ec="#777", hatch="////"),
                Rectangle((0, 0), 1, 1, fc="#c4c4c4", ec="none")],
                labels=["held ON ITS PATH by that arm",
                        "held OUT OF ITS FINISHED POSE by that arm",
                        "done, standing in that pose"],
                fontsize=7.2, ncol=3, loc="upper right",
                bbox_to_anchor=(1.005, -0.28), frameon=False)
    pair_matrix(fig.add_subplot(gs[0, 2]), phs, "pause seconds by pair (both phases)")

    # cause split
    ax = fig.add_subplot(gs[1, 2])
    keys = list(CAUSEK)
    tot = defaultdict(lambda: [0.0, 0.0])
    for k, ph in enumerate(phs):
        for s in ph["steps"]:
            tot[s["cause"]][k] += ph["sch"]["dt"]
    keys += [k for k in tot if k not in keys]
    y = np.arange(len(keys))
    b1 = [tot[k][0] for k in keys]
    b2 = [tot[k][1] for k in keys]
    col = [CAUSEC.get(k, "#888888") for k in keys]
    ax.barh(y, b1, color=col, height=0.62)
    ax.barh(y, b2, left=b1, color=col, height=0.62, alpha=0.45)
    for i, k in enumerate(keys):
        ax.text(b1[i] + b2[i] + 1.5, i, f"{b1[i] + b2[i]:.0f} s", va="center",
                fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(keys, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("pause s  (solid = phase 1, pale = phase 2)", fontsize=8)
    ax.set_title("what the two arms were doing", fontsize=9.5, loc="left")
    ax.tick_params(labelsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # distance histogram: the margin test's number, and the raw pose clearance
    ax = fig.add_subplot(gs[2, 0])
    dt = phs[0]["sch"]["dt"]
    allS = np.array([1000 * s["S"] for ph in phs for s in ph["steps"]
                     if np.isfinite(s["S"])])
    allD = np.array([1000 * s["D4"] for ph in phs for s in ph["steps"]
                     if np.isfinite(s["S"])])
    edges = np.arange(-10, 122, 4.0)
    hS, _ = np.histogram(np.clip(allS, -9, 120), edges)
    hD, _ = np.histogram(np.clip(allD, -9, 120), edges)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    ax.bar(ctr, hS * dt, width=3.6, color="#f0a202",
           label="swept clearance S (what the margin test reads)")
    ax.step(np.append(edges, edges[-1]), np.append(np.append([0], hD * dt), 0),
            where="pre", color="#333", lw=1.3,
            label="pose clearance (the four cell corners)")
    for x, c in ((0, "#7f1d1d"), (35, "#666"), (50, "#666"), (65, "#666"),
                 (80, "#d62728")):
        ax.axvline(x, color=c, lw=1.0, ls=":" if x not in (0, 80) else "-")
    ax.annotate("margin\n80 mm", (80, ax.get_ylim()[1] * 0.86), fontsize=7,
                color="#d62728", ha="left", va="top",
                xytext=(3, 0), textcoords="offset points")
    ax.annotate("capsules touch", (0, ax.get_ylim()[1] * 0.86), fontsize=7,
                color="#7f1d1d", ha="left", va="top",
                xytext=(3, 0), textcoords="offset points")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_xlabel("clearance at the blocking cell (mm)", fontsize=8)
    ax.set_ylabel("pause s", fontsize=8)
    ax.set_title("how close was the conflict really?", fontsize=9.5, loc="left")
    ax.tick_params(labelsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # counterfactual bars
    ax = fig.add_subplot(gs[2, 1:])
    labs = [c[0] for c in cf]
    p1 = [c[1] for c in cf]
    p2 = [c[2] for c in cf]
    mk = [c[3] for c in cf]
    x = np.arange(len(cf))
    ax.bar(x, p1, 0.62, color="#4c78a8", label="phase 1 pause s")
    ax.bar(x, p2, 0.62, bottom=p1, color="#9ecae9", label="phase 2 pause s")
    ideal = sum(max(ph["sch"]["nominal"].values()) for ph in phs)
    ax.axhline(ideal, color="#111", lw=1.1, ls="--")
    ax.annotate(f"perfect concurrency = {ideal:.0f} s  (each phase = its "
                "longest arm)", (len(cf) - 0.4, ideal), fontsize=7.5,
                color="#111", ha="right", va="bottom")
    ax.plot(x, mk, "o-", color="#d62728", ms=5, lw=1.6,
            label="total makespan (s)")
    for i in range(len(cf)):
        ax.text(i, p1[i] + p2[i] + 2.5, f"{p1[i] + p2[i]:.0f}", ha="center",
                fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=7.6, rotation=22, ha="right")
    ax.set_ylabel("seconds", fontsize=8)
    ax.set_title("counterfactuals — scheduler re-runs on the SAME frozen paths",
                 fontsize=9.5, loc="left")
    ax.legend(fontsize=7.5, ncol=3, loc="upper right")
    ax.tick_params(labelsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    fig.suptitle("Where the two-pass CSAIL schedule spends its 148 s of pauses",
                 fontsize=13, x=0.055, ha="left", y=0.972)
    fig.text(0.055, 0.947, "six FR3s, frozen certified paths, conductor v1 "
             "(advance-or-wait); every panel re-derived from the shipped "
             "programme, nothing re-planned", fontsize=8.6, color="#555")
    fig.savefig(path, dpi=125)
    plt.close(fig)


# ==========================================================================
def build_phase(res, dt, pens, draw_speed, transit_speed, qd_frac,
                park=writing.PARK_HOME):
    """THIS FILE IS ABOUT CONDUCTOR v1, so it parks arms the way v1 did.

    `writing.arm_program` now freezes an arm where it finishes by default
    (`aris_sixarm/idle.py`); every number in docs/CONCURRENCY.md was measured
    with the go-home behaviour, and reconstructing it under a different idle
    policy would be a different run wearing this document's labels.
    """
    progs, samp = {}, {}
    for aid in FLEET:
        segs = res["programs"].get(aid, []) if aid in res["arms"] else []
        p = writing.arm_program(FLEET[aid], segs, draw_speed=draw_speed,
                                transit_speed=transit_speed, qd_frac=qd_frac,
                                h_inv=H_INV_DEFAULT, pen_ext=pens[aid],
                                park=park, verbose=False)
        progs[aid], samp[aid] = p, writing.uniform_samples(p, dt)
    paths = coordination.arm_paths({a: samp[a]["q"] for a in FLEET}, dt, pens=pens)
    return progs, samp, paths


def main(argv=None):
    ap = add_args(argparse.ArgumentParser())
    ap.add_argument("--fps", type=float, default=12.0)
    # 12 fps x 4 substeps = the 1/48 s conducting clock the shipped run used
    # (every pause in out/csail_schedule_full.json is an exact multiple of it)
    ap.add_argument("--substeps", type=int, default=4)
    ap.add_argument("--draw-speed", type=float, default=writing.DRAW_SPEED_FLEET)
    ap.add_argument("--safety", type=float, default=coordination.SAFETY_M)
    ap.add_argument("--calib", type=float, default=coordination.CALIB_M)
    ap.add_argument("--ref", default=str(ROOT / "out/csail_schedule_full.json"))
    ap.set_defaults(idle_policy="home")   # see build_phase: this is a v1 study
    ap.add_argument("--png", default=str(ROOT / "out/concurrency_diagnostic.png"))
    ap.add_argument("--json", default=str(ROOT / "out/concurrency_diagnostic.json"))
    ap.add_argument("--no-perms", action="store_true")
    # SHIPPED first, so anything typed on the command line overrides it
    a = ap.parse_args(SHIPPED + list(sys.argv[1:] if argv is None else argv))
    dt = 1.0 / (a.fps * a.substeps)
    margin0 = float(a.safety + a.calib)
    pen = parse_pens(a.pens) or {}
    pens = {x: float(pen.get(x, 0.110)) for x in FLEET}
    ref = json.loads(Path(a.ref).read_text())
    # The reference has to be a run of the SAME policy or the reconstruction is
    # of a different piece of work; say so once, loudly, instead of failing an
    # assertion twenty lines further down with a number that looks like a bug.
    same = ref.get("idle_policy", "home") == "home"
    if not same:
        print(f"\n!! {a.ref} was scheduled with idle_policy="
              f"{ref.get('idle_policy')!r}; this study reconstructs conductor "
              "v1's go-home behaviour, so the per-arm checks against it are "
              "SKIPPED.  Re-run csail_schedule.py --idle-policy home for a "
              "reference this can be pinned to.\n")

    t00 = time.time()
    phases, _, _ = run_allocation(a, verbose=False)
    out, cfrows = [], []
    report = dict(margin=margin0, dt=dt, phases=[], counterfactuals={})

    for k, res in enumerate(phases):
        R = ref["phases"][k]
        print(f"\n=== {res['name']} " + "=" * 52)
        progs, samp, paths = build_phase(res, dt, pens, a.draw_speed,
                                         a.transit_speed, a.qd_frac)
        for x in res["arms"]:                 # the reconstruction is exact or nothing
            got, want = progs[x]["duration"], R["arm_nominal_s"][str(x)]
            assert not same or abs(got - want) < 1e-9, \
                f"arm {x} nominal {got} != {want}"
        t0 = time.time()
        fields = pair_fields(paths)
        free0 = make_free(paths, fields, margin0, sorted(paths))
        # THE SHIPPED ORDER IS AN INPUT, NOT A RE-DERIVATION.  The conductor
        # searches its priority order now (`coordination._search_priority`), so
        # re-deriving it here with this file's own busiest-first restatement
        # would reconstruct a different run and the assertions below would be
        # comparing two schedules rather than checking one.  The order the run
        # shipped with is read off the reference summary and handed in.
        sch = conduct(paths, fields, margin0, order=R["priority"], free=free0)
        assert sch is not None
        print(f"  geometry + conduct in {time.time() - t0:.1f} s; "
              f"priority {sch['order']}")
        for x, v in sorted(sch["pauses"].items()):
            want = R["pauses"].get(str(x), float("nan"))
            print(f"  arm {x:>2}: nominal {sch['nominal'][x]:6.2f} s  "
                  f"pause {v:6.2f} s  (shipped {want:6.2f} s)"
                  + ("" if abs(v - want) < 1e-9 else "   <-- MISMATCH"))
            assert not same or abs(v - want) < 1e-9, \
                f"arm {x} pause {v} != shipped {want}"
        assert not same or abs(sch["duration"] - R["duration_s"]) < 1e-9
        events, steps, asap = blame(sch, paths, samp, fields, margin0)
        ph = dict(name=res["name"], sch=sch, paths=paths, samp=samp,
                  fields=fields, events=events, steps=steps, asap=asap,
                  free0=free0, dur=sch["duration"])
        out.append(ph)

        ideal = max(sch["nominal"].values())
        print(f"  makespan {sch['duration']:.1f} s vs {ideal:.1f} s of perfect "
              f"concurrency -> {sch['duration'] - ideal:.1f} s of makespan lost; "
              f"{sch['pause_total']:.1f} s of pause, of which "
              f"{sch['pause_total'] - (sch['duration'] - ideal):.1f} s sits "
              f"inside slack the arm had anyway")
        n_un = sum(1 for s in steps if s["cause"] == "unattributed")
        route = sum(dt for s in steps if s["kind"] == "route")
        print(f"  {len(steps)} wait steps = {len(steps) * dt:.1f} s "
              f"(pause total {sch['pause_total']:.1f} s), {len(events)} events; "
              f"{route:.1f} s held on the path, "
              f"{len(steps) * dt - route:.1f} s held out of the parked pose; "
              f"{n_un * dt:.1f} s unattributed")
        print(f"  {'arm':>4} {'from':>7} {'for':>7} {'blocked by':>11} "
              f"{'doing':>18} {'blocker doing':>18} {'min gap':>9}")
        for e in sorted(events, key=lambda e: -e["n"]):
            what = (f"seg {e['seg']}" if e["seg"] >= 0 else "transit")
            wb = (f"seg {e['seg_b']}" if e["seg_b"] >= 0 and
                  e["sb"] in ("draw", "transit") else e["sb"])
            print(f"  {e['arm']:>4} {e['t0']:>6.1f}s {e['dur']:>6.2f}s "
                  f"{e['blocker']:>11} "
                  f"{(what if e['kind'] == 'route' else 'parking'):>18} "
                  f"{wb:>18} {1000 * e['S']:>8.1f}mm")
        per_pair = defaultdict(float)
        per_cause = defaultdict(float)
        for s in steps:
            per_pair[(s["arm"], s["blocker"])] += dt
            per_cause[s["cause"]] += dt
        for (x, b), v in sorted(per_pair.items(), key=lambda kv: -kv[1]):
            print(f"    arm {x:>2} blocked by arm {b:>2}: {v:6.2f} s")
        for c, v in sorted(per_cause.items(), key=lambda kv: -kv[1]):
            print(f"    cause {c:<18} {v:6.2f} s")
        fin = np.array([s["S"] for s in steps if np.isfinite(s["S"])])
        f4 = np.array([s["D4"] for s in steps if np.isfinite(s["S"])])
        sweep_only = float(np.sum((fin < margin0) & (f4 >= margin0)) * dt)
        touching = float(np.sum(f4 < 0) * dt)
        print(f"    blocked only by the swept-motion slack (the two POSES are "
              f"{1000 * margin0:.0f} mm clear): {sweep_only:.2f} s")
        print(f"    capsules that actually overlap (pose clearance < 0): "
              f"{touching:.2f} s; tightest pose clearance over all blocking "
              f"cells {1000 * float(f4.min()):.1f} mm")
        for lo, hi in ((-1.0, 0.0), (0.0, 0.035), (0.035, 0.050),
                       (0.050, 0.065), (0.065, 0.080)):
            v = float(np.sum((fin >= lo) & (fin < hi)) * dt)
            print(f"    swept clearance {1000 * lo:>6.0f}..{1000 * hi:>3.0f} mm: "
                  f"{v:6.2f} s")
        report["phases"].append(dict(
            name=res["name"], makespan=sch["duration"], ideal_makespan=ideal,
            pause_total=sch["pause_total"], priority=[int(x) for x in sch["order"]],
            pauses={str(x): float(v) for x, v in sch["pauses"].items()},
            wait_s=len(steps) * dt, n_events=len(events), route_s=route,
            rest_s=len(steps) * dt - route, unattributed_s=n_un * dt,
            sweep_only_s=sweep_only, overlapping_s=touching,
            tightest_pose_mm=1000 * float(f4.min()),
            per_pair={f"{x}<-{b}": v for (x, b), v in per_pair.items()},
            per_cause=dict(per_cause),
            events=[dict(arm=int(e["arm"]), blocker=int(e["blocker"]),
                         t0=e["t0"], dur=e["dur"], kind=e["kind"],
                         cause=e["cause"], min_gap_mm=1000 * float(e["S"]),
                         seg=e["seg"], blocker_seg=e["seg_b"],
                         blocker_state=e["sb"]) for e in events]))

    tot = sum(p["sch"]["pause_total"] for p in out)
    mk = sum(p["dur"] for p in out)
    cfrows.append(("shipped\n(margin 80 mm)", out[0]["sch"]["pause_total"],
                   out[1]["sch"]["pause_total"], mk))

    # ---- (a) margin sensitivity -----------------------------------------
    print("\n=== margin sensitivity " + "=" * 44)
    print(f"  {'margin':>8} {'phase1 pause':>13} {'phase2 pause':>13} "
          f"{'total pause':>12} {'ph1 mk':>8} {'ph2 mk':>8} {'makespan':>10} "
          f"{'min clear':>10}")
    report["counterfactuals"]["margin"] = []
    for mg in MARGINS:
        row, ok = [], True
        for ph in out:
            s = conduct(ph["paths"], ph["fields"], mg)
            if s is None:
                ok = False
                break
            row.append((s["pause_total"], s["duration"],
                        realized_clearance(s, ph["paths"], ph["fields"])))
        if not ok:
            print(f"  {1000 * mg:>7.0f}  INFEASIBLE")
            continue
        print(f"  {1000 * mg:>7.0f} {row[0][0]:>12.1f}s {row[1][0]:>12.1f}s "
              f"{row[0][0] + row[1][0]:>11.1f}s {row[0][1]:>7.1f}s "
              f"{row[1][1]:>7.1f}s {row[0][1] + row[1][1]:>9.1f}s "
              f"{1000 * min(r[2] for r in row):>9.1f}mm")
        report["counterfactuals"]["margin"].append(dict(
            margin=mg, pause=[r[0] for r in row], makespan=[r[1] for r in row],
            min_clearance=min(r[2] for r in row)))
        if mg != margin0:
            cfrows.append((f"margin {1000 * mg:.0f} mm", row[0][0], row[1][0],
                           row[0][1] + row[1][1]))

    # ---- (b) priority permutations ---------------------------------------
    if not a.no_perms:
        print("\n=== priority permutations " + "=" * 41)
        report["counterfactuals"]["perms"] = []
        best_tot = [0.0, 0.0]
        best_mk = [0.0, 0.0]
        best_orders = []
        for ph in out:
            mv = ph["sch"]["moving"]
            rows = []
            for perm in itertools.permutations(mv):
                s = conduct(ph["paths"], ph["fields"], margin0, order=list(perm),
                            retry=False, free=ph["free0"])
                if s is None:
                    rows.append((np.inf, np.inf, perm))
                    continue
                rows.append((s["pause_total"], s["duration"], perm))
            rows.sort()
            fez = [r for r in rows if np.isfinite(r[0])]
            cur = next(r for r in rows if list(r[2]) == list(mv))
            print(f"  {ph['name']}: {len(fez)}/{len(rows)} orders feasible; "
                  f"shipped {list(mv)} -> {cur[0]:.1f} s pause / "
                  f"{cur[1]:.1f} s makespan")
            print(f"    best pause   {list(fez[0][2])} -> {fez[0][0]:.1f} s "
                  f"pause / {fez[0][1]:.1f} s makespan")
            bm = min(fez, key=lambda r: r[1])
            print(f"    best makespan{list(bm[2])} -> {bm[0]:.1f} s pause / "
                  f"{bm[1]:.1f} s makespan")
            print(f"    worst        {list(fez[-1][2])} -> {fez[-1][0]:.1f} s "
                  f"pause / {fez[-1][1]:.1f} s makespan")
            best_tot[0] += fez[0][0]
            best_mk[0] += fez[0][1]
            best_tot[1] += bm[0]
            best_mk[1] += bm[1]
            best_orders.append(list(bm[2]))
            report["counterfactuals"]["perms"].append(dict(
                phase=ph["name"], n_feasible=len(fez), n=len(rows),
                shipped=dict(order=[int(x) for x in cur[2]], pause=cur[0],
                             makespan=cur[1]),
                best_pause=dict(order=[int(x) for x in fez[0][2]],
                                pause=fez[0][0], makespan=fez[0][1]),
                best_makespan=dict(order=[int(x) for x in bm[2]], pause=bm[0],
                                   makespan=bm[1]),
                worst=dict(order=[int(x) for x in fez[-1][2]], pause=fez[-1][0],
                           makespan=fez[-1][1])))
        cfrows.append(("best priority\norder", *_split(report, "perms", "best_pause"),
                       best_mk[0]))

    # ---- (c) crawl bound --------------------------------------------------
    print("\n=== 25 % crawl (optimistic lower bound on the delay) " + "=" * 14)
    c_p, c_mk = [], []
    for ph in out:
        saved = crawl_bound(ph["events"], ph["paths"], dt)
        pt = ph["sch"]["pause_total"] - sum(saved.values())
        mkn = max(ph["sch"]["finish"][x] - saved.get(x, 0.0)
                  for x in ph["sch"]["finish"])
        c_p.append(pt)
        c_mk.append(mkn)
        print(f"  {ph['name']}: pause {ph['sch']['pause_total']:.1f} -> "
              f"{pt:.1f} s, makespan {ph['dur']:.1f} -> {mkn:.1f} s")
        for x, v in sorted(saved.items()):
            print(f"    arm {x:>2}: {v:.1f} s of the delay recoverable")
    print(f"  TOTAL pause {tot:.1f} -> {sum(c_p):.1f} s, "
          f"makespan {mk:.1f} -> {sum(c_mk):.1f} s")
    cfrows.append(("25 % crawl\n(lower bound)", c_p[0], c_p[1], sum(c_mk)))
    report["counterfactuals"]["crawl"] = dict(pause=c_p, makespan=c_mk)

    # the geometric half of the same idea: re-conduct with the moving arm's own
    # swept slack cut to a quarter (a creeping link sweeps a thinner tube)
    print("\n=== thin-sweep re-conduct (what a crawl buys GEOMETRICALLY) " + "=" * 7)
    s_p, s_mk = [], []
    for ph in out:
        s = conduct(ph["paths"], ph["fields"], margin0, crawl=CRAWL)
        s_p.append(s["pause_total"])
        s_mk.append(s["duration"])
        print(f"  {ph['name']}: pause {ph['sch']['pause_total']:.1f} -> "
              f"{s['pause_total']:.1f} s, makespan {ph['dur']:.1f} -> "
              f"{s['duration']:.1f} s")
    cfrows.append(("thin sweep\n(crawl geometry)", s_p[0], s_p[1], sum(s_mk)))
    report["counterfactuals"]["thin_sweep"] = dict(pause=s_p, makespan=s_mk)

    # ---- (d) tuck poses ---------------------------------------------------
    print("\n=== tuck poses " + "=" * 52)
    report["counterfactuals"]["tuck"] = {}
    for lab, short, kw in (
            ("no parked arms 13/17", "delete parked\narms 13/17",
             dict(drop=(13, 17))),
            ("finished arms tuck", "finished arms\ntuck away",
             dict(rest_free=True, vanish_finished=True)),
            ("both", "tuck +\ndelete 13/17",
             dict(drop=(13, 17), rest_free=True, vanish_finished=True))):
        t_p, t_mk = [], []
        for ph in out:
            s = conduct(ph["paths"], ph["fields"], margin0, **kw)
            t_p.append(s["pause_total"] if s else float("nan"))
            t_mk.append(s["duration"] if s else float("nan"))
        print(f"  {lab:<22}: pause {t_p[0]:6.1f} + {t_p[1]:6.1f} = "
              f"{sum(t_p):6.1f} s, makespan {sum(t_mk):6.1f} s")
        cfrows.append((short, t_p[0], t_p[1], sum(t_mk)))
        report["counterfactuals"]["tuck"][lab] = dict(pause=t_p, makespan=t_mk)

    # ---- stacked: do the cheap ones add up? -------------------------------
    if not a.no_perms:
        print("\n=== stacked " + "=" * 55)
        report["counterfactuals"]["stacked"] = {}
        combos = (("best order + tuck",
                   dict(rest_free=True, vanish_finished=True), margin0),
                  ("best order + 50 mm margin", {}, 0.050),
                  ("best order + tuck + 50 mm",
                   dict(rest_free=True, vanish_finished=True), 0.050))
        for lab, kw, mg in combos:
            s_p, s_mk = [], []
            for ph, order in zip(out, best_orders):
                s = conduct(ph["paths"], ph["fields"], mg, order=order,
                            retry=True, **kw)
                s_p.append(s["pause_total"] if s else float("nan"))
                s_mk.append(s["duration"] if s else float("nan"))
            print(f"  {lab:<26}: pause {sum(s_p):6.1f} s, makespan "
                  f"{s_mk[0]:5.1f} + {s_mk[1]:5.1f} = {sum(s_mk):6.1f} s")
            cfrows.append((lab.replace(" + ", "\n+ ", 1), s_p[0], s_p[1],
                           sum(s_mk)))
            report["counterfactuals"]["stacked"][lab] = dict(pause=s_p,
                                                             makespan=s_mk)
    ideal = sum(max(ph["sch"]["nominal"].values()) for ph in out)
    print(f"\nperfect concurrency (each phase = its longest arm) would be "
          f"{ideal:.1f} s; the shipped run is {mk:.1f} s")
    report["ideal_makespan"] = ideal

    figure(out, cfrows, a.png)
    Path(a.json).write_text(json.dumps(report, indent=1, default=float))
    print(f"\nwrote {a.png} and {a.json}")
    print(f"DIAGNOSTIC WALL CLOCK {time.time() - t00:.1f} s")
    return report


def _split(report, key, which):
    """(phase1, phase2) pause totals of a per-phase counterfactual entry."""
    r = report["counterfactuals"][key]
    return (r[0][which]["pause"], r[1][which]["pause"])


if __name__ == "__main__":
    main()
