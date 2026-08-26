"""Per-arm segment sequencing: what to draw next, and which way round.

An arm's programme arrives from `allocate` as a bag of certified segments with
no order.  Choosing that order is a travelling-salesman problem, and the two
things this module insists on are WHAT the salesman pays and THAT he is allowed
to walk a street either way.

  THE COST IS THE TRANSIT, IN SECONDS.  Not paper distance.  The arm does not
  travel along the paper between two strokes: it lifts to a hover pose,
  joint-interpolates to the hover pose above the next stroke, and lowers.  What
  that costs is `writing.transit_time` — the capped joint-space move, floored by
  the pen-tip hop at `transit_speed` and by the lift/lower beats — and two
  strokes whose ends are 3 cm apart on the paper can be a long way apart in the
  redundancy band (different branch, q7 half a turn round) while two that look
  far apart cost the minimum beat.  The old nearest-neighbour-in-xy chaining
  optimised a quantity the arm never pays.  The transit-duration computation
  lives in `writing` and is imported, not re-derived, because a sequencer
  optimising numbers the timeline does not use is worse than no sequencer.

  A SEGMENT IS DIRECTION-AGNOSTIC.  Every gate `stroke_api` certifies is
  symmetric in s (see `stroke_api.reverse_plan`), so each segment offers TWO
  nodes — drawn forward, drawn backward — with entry and exit configurations
  swapped, and exactly one of the two must be visited.  That doubles the node
  count and it is worth it: the direction choice is often a bigger saving than
  the order, because it decides which end of the stroke the arm has to reach
  for next.

  EXACT WHERE EXACT IS AFFORDABLE.  n <= 16 segments is solved to optimality by
  Held-Karp over states (subset, last segment, last direction) — 2^n * 2n
  states, which for the CSAIL logo's biggest arm (12 segments) is 98 304 states
  and a fraction of a second.  Above that the same cost matrix is handed to a
  nearest-neighbour seed plus 2-opt and Or-opt passes with direction flips in
  the move set, under a wall-clock budget.  Both walk the SAME matrix, so the
  fallback is a different search over an identical objective, not a different
  objective.

  THE MOVES ARE WRITTEN FOR AN ASYMMETRIC COST.  Reversing a stretch of the
  tour does not merely re-traverse it: every segment inside it flips direction
  too, and the cost of an internal edge (i,di)->(j,dj) becomes
  (j,!dj)->(i,!di).  Those need not be equal, so the passes carry a matrix of
  per-edge reversal penalties and a prefix sum of it along the current tour,
  which makes the exact delta of a block reversal an O(1) lookup instead of an
  O(n) re-sum.  (With `writing`'s current lift/lower beats the penalty happens
  to be identically zero; the code does not assume it, and a test pins that the
  deltas match a brute-force re-evaluation either way.)

Everything here is deterministic: no randomness, no time-dependent tie-breaks,
ties resolved toward the lowest segment index and then the forward direction.
"""
import atexit
import os
import time

import numpy as np

from . import paper
from .fleet import H_INV_DEFAULT
from .frames import PEN_EXT, QD_MAX
from .writing import (LIFT_Z, QD_FRAC, T_HOME_F, T_LIFT_F, T_LOWER_F,
                      T_TRAVEL_MIN, TRANSIT_SPEED, dq_time_many,
                      lifted_or_lower)

EXACT_MAX_N = 16         # segments solved exactly by Held-Karp
TIME_BUDGET = 2.0        # s of local search per arm, above EXACT_MAX_N
OR_OPT_MAX = 3           # longest run Or-opt relocates
EPS = 1e-9               # s; an "improvement" below this is float noise
FK_BLOCK = 4_000_000     # configurations the paper screen may sample at once


# ==========================================================================
# 1. the cost matrix
# ==========================================================================
def _row_time(Q0, Q1, frac, tmin):
    """`writing._dq_time` row by row: (K,7) x (K,7) -> (K,)."""
    d = np.abs(np.asarray(Q1, float) - np.asarray(Q0, float))
    return np.maximum(tmin, (d / (QD_MAX * max(frac, 1e-6))).max(axis=-1))


def endpoints(spec, segs, h_inv=H_INV_DEFAULT, pen_ext=PEN_EXT):
    """The two ends of every segment, with their hover poses.

    -> dict(q (n,2,7), xy (n,2,2), hover (n,2,7), z (n,2), n)
    where index [i, 0] is the plan's FIRST sample and [i, 1] its last.  The
    hover pose is `writing.lifted_or_lower`'s — the same call, at the same
    heights, that `writing.arm_program` will make when it lays the transit
    down, so the cost the sequencer minimises is the cost the timeline pays.
    """
    n = len(segs)
    q = np.zeros((n, 2, 7))
    xy = np.zeros((n, 2, 2))
    hov = np.zeros((n, 2, 7))
    z = np.zeros((n, 2))
    for i, s in enumerate(segs):
        qs = np.asarray(s["plan"]["qs"], float)
        pts = np.asarray(s["plan"]["pts"], float)
        # "THE SAME CALL" INCLUDES THE PEN ORIENTATION IT ASKS FOR.  Both
        # sides hover VERTICALLY, above a leaning stroke as above a flat one
        # (`writing.lifted_config`'s `tilt` argument records why), and the
        # cross-check in `csail_schedule` is what keeps them honest: when this
        # function and `arm_program` briefly disagreed about that, it said so
        # immediately — "sequencer priced transits the timeline does not pay:
        # arm 71: 0.6230 s", on the one arm drawing a 15-degree span.
        for e, k in ((0, 0), (1, -1)):
            q[i, e], xy[i, e] = qs[k], pts[k]
            h, zz = lifted_or_lower(spec, qs[k], pts[k], h_inv=h_inv,
                                    pen_ext=pen_ext)
            hov[i, e], z[i, e] = h, zz
    return dict(q=q, xy=xy, hover=hov, z=z, n=n)


def _leg_surcharge(spec, Q0, Q1, tip_floor, floor, qd_frac, h_inv, pen_ext,
                   q_home=None):
    """Extra seconds for routing each (Q0[i] -> Q1[i]) move. -> (K,).

    A TRANSIT IS THREE MOVES AND ALL THREE CAN NEED A DETOUR.  The first cut of
    this priced only the hover-to-hover crossing, on the assumption that a lift
    off the paper and a lower onto it are short and vertical and never in
    trouble.  Mostly true, and when it is not, `arm_program` inserts a via there
    too and the timeline pays seconds the matrix never charged — which the
    pipeline's own cross-check caught immediately: "sequencer priced transits
    the timeline does not pay: arm 2: 1.1225 s".  Every leg `writing` routes is
    now priced by the same call with the same floors, including the two the
    depot pays (ready pose -> first entry, last exit -> ready pose).
    """
    Q0 = np.asarray(Q0, float).reshape(-1, 7)
    Q1 = np.asarray(Q1, float).reshape(-1, 7)
    K = len(Q0)
    out = np.zeros(K)
    tf = np.broadcast_to(np.asarray(tip_floor, float).reshape(-1), (K,))
    fl = np.broadcast_to(np.asarray(floor, float).reshape(-1), (K,))
    for i in range(K):
        ok, _, _ = paper.move_ok(spec, Q0[i], Q1[i], pen_ext, h_inv,
                                 tip_floor=float(tf[i]))
        if ok:
            continue
        r = paper.route(spec, Q0[i], Q1[i], pen_ext=pen_ext, h_inv=h_inv,
                        tip_floor=float(tf[i]), q_home=q_home)
        if r is None:
            out[i] = np.inf
            continue
        qs = [Q0[i]] + list(r["vias"]) + [Q1[i]]
        direct = float(_row_time(Q0[i][None, :], Q1[i][None, :], qd_frac, 0.0)[0])
        routed = float(sum(_row_time(np.asarray(u)[None, :],
                                     np.asarray(v)[None, :], qd_frac, 0.0)[0]
                           for u, v in zip(qs[:-1], qs[1:])))
        out[i] = max(0.0, max(fl[i], routed) - max(fl[i], direct))
    return out


def _paper_surcharge(spec, exi_h, ent_h, same, floor, qd_frac, h_inv, pen_ext):
    """What routing each crossing around the paper adds. -> (M,N) seconds.

    THE SEQUENCER HAS TO PAY FOR THE DETOUR IT CAUSES.  A hover-to-hover move
    that dives through the canvas is not free to fix: `paper.route` climbs and
    flies over, and those via-configurations are extra joint-space seconds.  If
    the matrix priced the straight line the tour would be chosen against a
    fiction and `arm_program` would then lay down — and charge the fleet for —
    something the sequencer never considered.  So every cell that needs a
    detour carries its cost, and every cell that CANNOT be routed at all
    carries `inf`, which is how an unflyable crossing stops being an ordering
    the search can propose rather than an exception it hits later.

    THE HOVER HEIGHTS ARE MEASURED, NOT LOOKED UP.  Both this and the fiber
    variant hand in nothing but the two hover poses; the height each one
    actually reached is the FK tip z of the pose itself.  That is what lets the
    plain and the cluster matrices share one definition of the surcharge, which
    is the property `tests/test_menu.py` pins when it asserts a one-variant
    menu reproduces `cost_matrix` cell for cell.

    Only the crossings that violate cost anything: the screen is ONE batched FK
    over every cell's sampled line, and `paper.route` memoises, so on the CSAIL
    logo (13 segments, 26 nodes) it is milliseconds and zero in all but a
    handful of cells.

    RECTANGULAR ON PURPOSE.  `exi_h` and `ent_h` need not be the same node list:
    the incremental matrix in `allocate` grows an arm's costs one span at a time
    and asks only for the NEW rows against the old columns and vice versa.  The
    square call every sequencing pass makes is the special case, and it computes
    exactly what it always did.
    """
    M, N = len(exi_h), len(ent_h)
    out = np.zeros((M, N))
    if M == 0 or N == 0:
        return out
    c_exi, z_exi = paper.chain_tip_z(exi_h, spec, pen_ext, h_inv)
    c_ent, z_ent = paper.chain_tip_z(ent_h, spec, pen_ext, h_inv)
    tip_floor = np.minimum(np.minimum(z_exi[:, None], z_ent[None, :]),
                           paper.TIP_CLEAR)
    # `paper.route` clamps its floors to what the two endpoints can hold before
    # it keys its memo on them; done here in bulk, the per-cell key is a tuple
    # build rather than two forward-kinematics calls (`paper.key_maker`).
    chain_floor = np.minimum(np.minimum(c_exi[:, None], c_ent[None, :]),
                             paper.CHAIN_CLEAR)

    # ---- one batched screen over every cell, in row blocks -----------------
    # The screen is (M, N, K, 7) floats if it is built at once, which is a
    # gigabyte before an arm has fifty spans; the FK is per configuration, so
    # blocking the rows changes nothing but the size of the temporary.
    K = paper.SAMPLES
    f = np.linspace(0.0, 1.0, K).reshape(1, 1, K, 1)
    cz = np.empty((M, N))
    tz = np.empty((M, N))
    rows = max(1, int(FK_BLOCK // max(N * K, 1)))
    for a0 in range(0, M, rows):
        a1 = min(a0 + rows, M)
        L = exi_h[a0:a1, None, None, :] * (1.0 - f) + ent_h[None, :, None, :] * f
        c, t = paper.chain_tip_z(L.reshape(-1, 7), spec, pen_ext, h_inv)
        cz[a0:a1] = c.reshape(a1 - a0, N, K).min(-1)
        tz[a0:a1] = t.reshape(a1 - a0, N, K).min(-1)
    bad = (~same) & ((cz < paper.CHAIN_CLEAR - paper.EPS)
                     | (tz < tip_floor - paper.EPS))

    cells = [(int(a), int(b)) for a, b in zip(*np.where(bad))]
    for a, b, val in _screen_routes(spec, exi_h, ent_h, tip_floor, chain_floor,
                                    floor, qd_frac, h_inv, pen_ext, cells):
        out[a, b] = val
    return out


def dive_screen(spec, exi_h, ent_h, same, h_inv=H_INV_DEFAULT,
                pen_ext=PEN_EXT):
    """Which crossings of this block go through the canvas. -> (cells, tip, chain).

    The cheap half of `_paper_surcharge`, exposed because `allocate._ArmMatrix`
    wants to know which crossings of ALL the blocks it is about to build need
    routing before it builds any of them — see `prewarm`.
    """
    M, N = len(exi_h), len(ent_h)
    if M == 0 or N == 0:
        return [], np.zeros((M, N)), np.zeros((M, N))
    c_exi, z_exi = paper.chain_tip_z(exi_h, spec, pen_ext, h_inv)
    c_ent, z_ent = paper.chain_tip_z(ent_h, spec, pen_ext, h_inv)
    tip = np.minimum(np.minimum(z_exi[:, None], z_ent[None, :]),
                     paper.TIP_CLEAR)
    chain = np.minimum(np.minimum(c_exi[:, None], c_ent[None, :]),
                       paper.CHAIN_CLEAR)
    K = paper.SAMPLES
    f = np.linspace(0.0, 1.0, K).reshape(1, 1, K, 1)
    cz = np.empty((M, N))
    tz = np.empty((M, N))
    rows = max(1, int(FK_BLOCK // max(N * K, 1)))
    for a0 in range(0, M, rows):
        a1 = min(a0 + rows, M)
        L = exi_h[a0:a1, None, None, :] * (1.0 - f) + ent_h[None, :, None, :] * f
        c, t = paper.chain_tip_z(L.reshape(-1, 7), spec, pen_ext, h_inv)
        cz[a0:a1] = c.reshape(a1 - a0, N, K).min(-1)
        tz[a0:a1] = t.reshape(a1 - a0, N, K).min(-1)
    bad = (~same) & ((cz < paper.CHAIN_CLEAR - paper.EPS)
                     | (tz < tip - paper.EPS))
    return [(int(a), int(b)) for a, b in zip(*np.where(bad))], tip, chain


def prewarm(spec, blocks, qd_frac=QD_FRAC, h_inv=H_INV_DEFAULT,
            pen_ext=PEN_EXT):
    """Route every diving crossing of these blocks in ONE dispatch.

    THE BATCH IS WHAT MAKES THE POOL WORTH HAVING.  A pair that routes does so on
    the first or second shape of `paper.route`'s ladder; a pair that CANNOT
    route walks all of it, and on this rig that is tens of milliseconds against
    a few.  Dispatched a hundred and seventy cells at a time — which is what
    growing an arm's matrix by one span asks for — the wall clock of each batch
    is its slowest crossing, and the twenty-four workers spend most of it idle.
    Handed every block of a growth step at once they do not.

    `blocks` is [(exi_h, ent_h, same)].  Nothing is returned: the answers go
    into the memos, and the `leg_costs` calls that follow find them there.
    """
    tasks = []
    for exi_h, ent_h, same in blocks:
        cells, tip, chain = dive_screen(spec, exi_h, ent_h, same, h_inv, pen_ext)
        if not cells:
            continue
        key_of = paper.key_maker(spec, exi_h, ent_h, pen_ext, h_inv)
        for a, b in cells:
            tasks.append((key_of(a, b, tip[a, b], chain[a, b]),
                          exi_h[a], ent_h[b], float(tip[a, b])))
    price_crossings(spec, tasks, pen_ext, h_inv, qd_frac)


# --------------------------------------------------------------------------
# 1a. the screen's crossings, priced on as many cores as there are
#
# ROUTING IS THE COST MATRIX, AND IT IS EMBARRASSINGLY PARALLEL.  On the
# Trollface at its shipped placement 57 % of the crossings between one arm's
# spans dive through the canvas, and each of those walks `paper.route`'s ladder
# — lift, retract-and-go, arc, Cartesian traverse, fold-through-home — at about
# 15 ms.  Eighteen thousand of them is 280 s, and measured with a profiler it is
# 94 % of what building an arm's matrix costs; everything else in this module is
# array arithmetic beside it.
#
# Nothing about the answer depends on anything else in the matrix: a cell's
# route is a function of two hover poses, the arm and the floors.  So the cells
# are farmed out to a fork pool and the answers filed back into `paper`'s own
# memo, which makes them free to the next matrix, to `prune_unflyable`, to
# `merge_remainders` and to the sequencing pass — and, because a route knows
# nothing about `qd_frac`, free to the other execution profiles as well.
#
# The parent screens the cells it has already routed itself (a memo hit is
# microseconds and shipping it to a worker is not), and a worker never forks a
# pool of its own.
ROUTE_JOBS = 0             # 0: decide from the machine.  1: never fork
ROUTE_PAR_MIN = 24         # crossings a batch needs before the pool is worth
#   the round trip.  A whole Trollface allocation walks the ladder about 21 000
#   times at 15 ms each — 570 s if it is done one crossing at a time, and the
#   thing the clock is actually made of — so the pool matters; what does not pay
#   is shipping a handful of cells to it.  The WORKER COUNT is deliberately not
#   scaled to the batch: a pool is only cheap if it is the same pool next time.
_PARALLEL = True           # cleared inside a worker, so nothing nests
_PRICED = {}               # (route key, qd_frac) -> (routed s, direct s)
_POOL, _POOL_JOBS, _POOL_PID = None, 0, 0


def clear_cache():
    """Drop the priced-crossing memo.  `paper.clear_cache` is its other half.

    The workers are shut down too: each holds a forked copy of this process,
    memos and fleet included, and the reason the memos are being dropped is
    always that something they were derived from has changed.
    """
    _PRICED.clear()
    close_pool()


paper.on_clear(clear_cache)
atexit.register(lambda: close_pool())


def route_jobs(n_cells):
    """Processes to screen `n_cells` crossings on. -> int (1 = do it here)."""
    import multiprocessing as mp
    if not _PARALLEL or mp.current_process().daemon:
        return 1                  # a pool worker may not have children
    if int(n_cells) < int(ROUTE_PAR_MIN):
        return 1
    cap = int(ROUTE_JOBS) if int(ROUTE_JOBS) > 0 \
        else min((os.cpu_count() or 1), 24)
    return max(1, min(cap, int(n_cells)))


def _no_nesting():
    global _PARALLEL
    _PARALLEL = False


def screen_pool(jobs):
    """The screen's worker pool, kept alive between matrices. -> Pool.

    ONE POOL, NOT ONE PER BLOCK.  An arm's matrix is built in blocks — new rows
    against old columns, old against new, new against new — and a picture with a
    hundred spans builds a hundred of them.  Forking twenty-four workers for each
    cost 173 s of a 248 s run in `posix.read` and `fork` alone, all of it
    waiting for workers that had a few dozen crossings to do.  The pool is made
    once and reused, so the fork is paid once; the workers accumulate their own
    `paper` memos as they go, which is the second reason not to keep throwing
    them away.
    """
    global _POOL, _POOL_JOBS, _POOL_PID
    if _POOL is not None and _POOL_JOBS == jobs and _POOL_PID == os.getpid():
        return _POOL
    close_pool()
    import multiprocessing as mp
    _POOL = mp.get_context("fork").Pool(jobs, initializer=_no_nesting)
    _POOL_JOBS, _POOL_PID = jobs, os.getpid()
    return _POOL


def close_pool():
    """Shut the screen's workers down (they hold a copy of this process).

    ONLY IN THE PROCESS THAT MADE THEM.  A conduct worker or a profile worker is
    forked out of a process that may already have a pool, and it inherits the
    Pool object — pids and all.  Its `atexit` would then terminate workers
    belonging to its parent, which would take the route screen down under a run
    that is still using it.  The pid check is what stops a child tidying up
    somebody else's processes.
    """
    global _POOL, _POOL_JOBS, _POOL_PID
    if _POOL is not None and _POOL_PID == os.getpid():
        try:
            _POOL.terminate()
            _POOL.join()
        except Exception:                                   # pragma: no cover
            pass
    _POOL, _POOL_JOBS, _POOL_PID = None, 0, 0


def _screen_task(spec, pen_ext, h_inv, qd_frac, q_home, task):
    """One crossing: route it and price the detour, floor aside.

    -> (key, routed seconds, direct seconds, route), `routed` infinite where no
    shape on the ladder certified.  The travel FLOOR is deliberately left out:
    it belongs to the cell, while everything here belongs to the pair of poses
    and the joint-speed cap, which is what makes the answer cacheable across
    every matrix that asks about the same crossing.

    `key` is computed by the CALLER and handed straight back.  `paper`'s memo
    keys on `id(spec)`, and a worker's copy of the arm is a different object at
    a different address; the parent's key is the only one that means anything
    where the answer is going to be filed.
    """
    key, q0, q1, tip = task
    r = paper.route(spec, q0, q1, pen_ext=pen_ext, h_inv=h_inv,
                    tip_floor=float(tip), q_home=q_home)
    if r is None:
        return key, np.inf, 0.0, None
    qs = [q0] + list(r["vias"]) + [q1]
    direct = float(_row_time(q0[None, :], q1[None, :], qd_frac, 0.0)[0])
    routed = float(sum(_row_time(np.asarray(u)[None, :],
                                 np.asarray(v)[None, :], qd_frac, 0.0)[0]
                       for u, v in zip(qs[:-1], qs[1:])))
    return key, routed, direct, r


def _screen_routes(spec, exi_h, ent_h, tip_floor, chain_floor, floor, qd_frac,
                   h_inv, pen_ext, cells):
    """Price every crossing in `cells`. -> [(a, b, seconds)].

    Two memos, in the order that costs least.  A crossing this process has
    already PRICED at this joint-speed cap is a dict lookup and two float
    comparisons; only what is left is routed, here or in the pool, and what
    comes back is filed under both `paper`'s memo and this module's, so a
    crossing is walked exactly once per run however many matrices ask about it.
    """
    if not cells:
        return []
    key_of = paper.key_maker(spec, exi_h, ent_h, pen_ext, h_inv)
    qk = round(float(qd_frac), 9)

    def priced(a, b, routed, direct):
        fl = float(floor[a, b])
        return (a, b, np.inf if not np.isfinite(routed)
                else max(0.0, max(fl, routed) - max(fl, direct)))

    keys = [key_of(a, b, tip_floor[a, b], chain_floor[a, b]) for a, b in cells]
    price_crossings(spec, [(k, exi_h[a], ent_h[b], float(tip_floor[a, b]))
                           for k, (a, b) in zip(keys, cells)],
                    pen_ext, h_inv, qd_frac)
    out = []
    for k, (a, b) in zip(keys, cells):
        routed, direct = _PRICED[(k, qk)]
        out.append(priced(a, b, routed, direct))
    return out


def price_crossings(spec, tasks, pen_ext=PEN_EXT, h_inv=H_INV_DEFAULT,
                    qd_frac=QD_FRAC):
    """Fill the priced-crossing memo for these (key, q0, q1, tip_floor) tasks.

    A crossing already in it is left alone; the rest are routed, here or in the
    pool, and filed under both `paper`'s memo and this module's.  `chunksize=1`
    on purpose: a failed route walks the whole ladder and a successful one stops
    at the second rung, so handing a worker a fixed block of them is how one
    unlucky crossing ends up deciding the batch.
    """
    qk = round(float(qd_frac), 9)
    seen, todo = set(), []
    for t in tasks:
        if (t[0], qk) in _PRICED or t[0] in seen:
            continue
        seen.add(t[0])
        todo.append(t)
    if not todo:
        return
    import functools
    run = functools.partial(_screen_task, spec, pen_ext, h_inv, qd_frac,
                            np.asarray(spec.q_seed, float).reshape(7))
    # A WORKER'S MEMO IS NOT THIS PROCESS'S.  The pool is forked once and the
    # routes bought after that live only in the parent, so a crossing THIS
    # process has already walked is done here — it is a dict lookup and a few
    # microseconds of arithmetic, against a round trip to a worker that would
    # walk the ladder again.  It is also what makes the other three execution
    # profiles cheap: a route knows nothing about `qd_frac`, so the second
    # profile of a picture finds every crossing already routed and only has to
    # re-price it.
    here = [t for t in todo
            if paper.cached_route(t[0]) is not paper.NOT_CACHED]
    fresh = [t for t in todo
             if paper.cached_route(t[0]) is paper.NOT_CACHED]
    rows = [run(t) for t in here]
    if fresh:
        jobs = route_jobs(len(fresh))
        rows += ([run(t) for t in fresh] if jobs <= 1
                 else screen_pool(jobs).map(run, fresh, chunksize=1))
    for key, routed, direct, r in rows:
        paper.cache_route(key, r)
        _PRICED[(key, qk)] = (routed, direct)


# --------------------------------------------------------------------------
# 1b. the matrix, in the two pieces it is actually made of
#
# A CELL OF THE COST MATRIX DEPENDS ON TWO NODES AND NOTHING ELSE.  `C[a, b]` is
# the lift off node a, the hover-to-hover crossing, and the lower onto node b:
# every term is a function of a's exit pose and b's entry pose, of the arm, and
# of the run's fixed speeds — never of which OTHER segments happen to be in the
# bag.  That is a strong property and until now nothing used it: `arm_load`
# rebuilt the whole matrix for every candidate bag the balancer priced, and on a
# picture with sixty spans the same few thousand crossings were re-screened
# through forward kinematics hundreds of times each.
#
# So the three pieces are separated here and `cost_matrix` is assembled from
# them.  `node_lift` and `node_lower` are per node, `leg_costs` is per PAIR and
# rectangular, and `depot_legs` is per node again.  `allocate._ArmMatrix` keeps
# one growing instance of each per arm and slices a bag's matrix out of it, so a
# span's crossings are screened once per run instead of once per price.  Nothing
# about the arithmetic moved: `cost_matrix` below is the same four array
# operations in the same order, and `tests/test_sequence.py` pins that a sliced
# matrix equals the freshly built one cell for cell.
def node_lift(spec, exi_q, exi_h, qd_frac=QD_FRAC, h_inv=H_INV_DEFAULT,
              pen_ext=PEN_EXT, paper_safe=True):
    """Seconds to lift the pen off each node's exit onto its hover. -> (N,)."""
    lift = _row_time(exi_q, exi_h, qd_frac, T_LIFT_F)
    if paper_safe:
        lift = lift + _leg_surcharge(spec, exi_q, exi_h, paper.CONTACT_FLOOR,
                                     T_LIFT_F, qd_frac, h_inv, pen_ext)
    return lift


def node_lower(spec, ent_h, ent_q, qd_frac=QD_FRAC, h_inv=H_INV_DEFAULT,
               pen_ext=PEN_EXT, paper_safe=True):
    """Seconds to lower from each node's hover onto its entry. -> (N,)."""
    lower = _row_time(ent_h, ent_q, qd_frac, T_LOWER_F)
    if paper_safe:
        lower = lower + _leg_surcharge(spec, ent_h, ent_q, paper.CONTACT_FLOOR,
                                       T_LOWER_F, qd_frac, h_inv, pen_ext)
    return lower


def leg_costs(spec, exi_h, exi_xy, ent_h, ent_xy, lift, lower, same,
              transit_speed=TRANSIT_SPEED, qd_frac=QD_FRAC,
              h_inv=H_INV_DEFAULT, pen_ext=PEN_EXT, paper_safe=True):
    """The (M, N) block of exit-to-entry transit seconds, paper detours included.

    `same[i, j]` marks a pair that no tour may contain (the two nodes are the
    same segment); those cells come back `inf` exactly as `cost_matrix` sets
    them, and the paper screen skips them.
    """
    M, N = len(exi_h), len(ent_h)
    if M == 0 or N == 0:
        return np.zeros((M, N))
    hop = np.linalg.norm(ent_xy[None, :, :] - exi_xy[:, None, :], axis=-1)
    floor = np.maximum(T_TRAVEL_MIN, hop / max(transit_speed, 1e-9))
    travel = dq_time_many(exi_h, ent_h, qd_frac, floor)
    B = lift[:, None] + travel + lower[None, :]
    B[same] = np.inf
    if paper_safe:
        B = B + _paper_surcharge(spec, exi_h, ent_h, same, floor, qd_frac,
                                 h_inv, pen_ext)
    return B


def depot_legs(spec, ent_h, exi_h, lift, lower, qd_frac=QD_FRAC,
               h_inv=H_INV_DEFAULT, pen_ext=PEN_EXT, q_start=None,
               return_home=True, paper_safe=True):
    """(into, outof): the depot's entry legs and its exit legs, per node."""
    N = len(ent_h)
    if N == 0:
        return np.zeros(0), np.zeros(0)
    q0 = np.asarray(spec.q_seed if q_start is None else q_start, float)
    depot = np.repeat(q0.reshape(1, 7), N, axis=0)
    home = np.repeat(np.asarray(spec.q_seed, float)[None, :], len(exi_h), axis=0)
    into = _row_time(depot, ent_h, qd_frac, T_HOME_F) + lower
    outof = lift + (_row_time(exi_h, home, qd_frac, T_HOME_F)
                    if return_home else 0.0)
    if paper_safe:
        into = into + _leg_surcharge(spec, depot, ent_h,
                                     paper.travel_floor(LIFT_Z, LIFT_Z),
                                     T_HOME_F, qd_frac, h_inv, pen_ext)
        if return_home:
            outof = outof + _leg_surcharge(spec, exi_h, home,
                                           paper.travel_floor(LIFT_Z, LIFT_Z),
                                           T_HOME_F, qd_frac, h_inv, pen_ext)
    return into, outof


def cost_matrix(spec, segs, transit_speed=TRANSIT_SPEED, qd_frac=QD_FRAC,
                h_inv=H_INV_DEFAULT, ends=None, pen_ext=PEN_EXT, q_start=None,
                return_home=True, paper_safe=True):
    """Every transit time an ordering could possibly pay. -> (2n+1, 2n+1).

    Node `2i + d` is segment i drawn forward (d = 0) or backward (d = 1); node
    `2n` is the depot, the pose the arm starts the pass in.  `C[a, b]` is the
    seconds from node a's EXIT to node b's ENTRY, hover overhead included;
    `C[2n, b]` is the entry lift from the depot and `C[a, 2n]` the exit lift.
    A segment cannot follow itself in either direction, so those cells are inf.

    THE DEPOT IS NOT ALWAYS THE READY POSE, AND THE TOUR DOES NOT ALWAYS CLOSE.
    `q_start` is where the arm is standing when the pass begins — `spec.q_seed`
    for the first pass, and wherever the previous pass froze it for the second.
    `return_home=False` is the freeze-in-place idle policy (`idle.py`): the arm
    lifts its pen at the last stroke and stops, so the last leg costs the lift
    and nothing else.  Both belong here rather than in a correction downstream,
    because the ORDER the sequencer picks depends on them: which segment is
    cheapest to start from depends on where the arm is, and an arm that never
    goes home should not be paying for the trip when it decides which segment
    to finish on.

    Built as four array operations rather than 4n^2 scalar calls, which is what
    keeps a few hundred segments per arm affordable: the local search then only
    ever looks costs up.
    """
    ends = endpoints(spec, segs, h_inv, pen_ext) if ends is None else ends
    n = ends["n"]
    N = 2 * n
    if n == 0:
        return np.zeros((1, 1))
    ent_q = ends["q"][:, [0, 1]].reshape(N, 7)
    ent_h = ends["hover"][:, [0, 1]].reshape(N, 7)
    ent_xy = ends["xy"][:, [0, 1]].reshape(N, 2)
    exi_q = ends["q"][:, [1, 0]].reshape(N, 7)
    exi_h = ends["hover"][:, [1, 0]].reshape(N, 7)
    exi_xy = ends["xy"][:, [1, 0]].reshape(N, 2)

    lift = node_lift(spec, exi_q, exi_h, qd_frac, h_inv, pen_ext, paper_safe)
    lower = node_lower(spec, ent_h, ent_q, qd_frac, h_inv, pen_ext, paper_safe)
    seg = np.arange(N) // 2
    C = np.full((N + 1, N + 1), np.inf)
    C[:N, :N] = leg_costs(spec, exi_h, exi_xy, ent_h, ent_xy, lift, lower,
                          seg[:, None] == seg[None, :], transit_speed, qd_frac,
                          h_inv, pen_ext, paper_safe)
    C[N, :N], C[:N, N] = depot_legs(spec, ent_h, exi_h, lift, lower, qd_frac,
                                    h_inv, pen_ext, q_start, return_home,
                                    paper_safe)
    return C


def flip_index(N):
    """Node -> the same segment drawn the other way (the depot maps to itself)."""
    fl = np.arange(N + 1)
    fl[:N] ^= 1
    return fl


def nodes_of(order, dirs):
    """(order, dirs in +-1) -> the node ids a tour visits, in order."""
    return [2 * int(i) + (0 if d > 0 else 1) for i, d in zip(order, dirs)]


def tour_of(order, dirs, n):
    """(order, dirs) -> the closed tour, depot first."""
    return [2 * n] + nodes_of(order, dirs)


def cycle_cost(C, tour):
    """Total transit seconds of a closed tour (depot at index 0)."""
    t = np.asarray(tour, int)
    if len(t) < 2:
        return 0.0
    return float(C[t, np.roll(t, -1)].sum())


def sequence_cost(C, n, order, dirs):
    """Transit seconds of an explicit (order, directions), depot to depot."""
    return cycle_cost(C, tour_of(order, dirs, n))


def _split(tour, n):
    """A tour (depot first) -> (order, dirs)."""
    body = [k for k in tour if k != 2 * n]
    return ([k // 2 for k in body], [1 if k % 2 == 0 else -1 for k in body])


# ==========================================================================
# 2. exact: Held-Karp over (subset, last segment, last direction)
# ==========================================================================
def held_karp(C, n):
    """Minimum-transit order AND directions, exactly. -> dict.

    dp[mask, node] is the cheapest way to leave the ready pose, draw exactly
    the segments in `mask`, and be standing at `node`'s exit.  The recursion is
    the textbook one; the only twist is that a segment contributes two nodes
    and the mask counts the SEGMENT, so choosing a node commits its direction.

    Ties go to the lowest node id (segment index first, forward before
    backward), which makes the result a function of the cost matrix alone.

    RAISES `RuntimeError` when no ordering is finite — including over ONE
    segment, where the only ordering is "fly there and draw it".  That case has
    its own two-line branch and the branch used to hand back its `inf` as a
    cost instead of refusing (2026-08-25): a bag whose single span the arm can
    ink and cannot reach was reported as a tour, `prune_unflyable` only ever
    drops what makes `solve` RAISE so the span stayed, and the refusal surfaced
    a stage later as `writing.PaperRefused` with the whole run already
    allocated.  On the proposed rig, where an arm ending a pass with one
    segment is ordinary, that killed all four execution profiles.
    `cluster_held_karp` has no such shortcut and always raised, so the two
    solvers disagreed about the same bag — which is how it was found.
    """
    N = 2 * n
    if n == 0:
        return dict(order=[], dirs=[], cost=0.0, states=0, method="held_karp")
    if n == 1:
        tot = np.array([C[N, 0] + C[0, N], C[N, 1] + C[1, N]], float)
        if not np.isfinite(tot).any():
            raise RuntimeError(f"no feasible order over {n} segments")
        k = int(np.argmin(tot))
        return dict(order=[0], dirs=[1 if k == 0 else -1],
                    cost=float(tot[k]), states=2, method="held_karp")
    T = C[:N, :N]
    dp = np.full((1 << n, N), np.inf)
    par = np.full((1 << n, N), -1, np.int32)
    node = np.arange(N)
    dp[1 << (node // 2), node] = C[N, :N]
    for mask in range(1, 1 << n):
        row = dp[mask]
        if not np.isfinite(row).any():
            continue
        cand = row[:, None] + T
        best = cand.min(axis=0)
        arg = cand.argmin(axis=0)
        for j in range(n):
            if (mask >> j) & 1:
                continue
            nm = mask | (1 << j)
            for k in (2 * j, 2 * j + 1):
                if best[k] < dp[nm, k]:
                    dp[nm, k] = best[k]
                    par[nm, k] = arg[k]
    full = (1 << n) - 1
    tot = dp[full] + C[:N, N]
    if not np.isfinite(tot).any():
        raise RuntimeError(f"no feasible order over {n} segments")
    k = int(np.argmin(tot))
    cost, body, mask = float(tot[k]), [], full
    while k >= 0:
        body.append(k)
        p = int(par[mask, k])
        mask ^= 1 << (k // 2)
        k = p
    body.reverse()
    order, dirs = _split(body, n)
    return dict(order=order, dirs=dirs, cost=cost, states=int((1 << n) * N),
                method="held_karp")


# ==========================================================================
# 3. heuristic: nearest neighbour, then 2-opt + Or-opt with direction flips
# ==========================================================================
def nearest_neighbour(C, n):
    """Greedy seed from the ready pose. -> tour (depot first).

    At every step the cheapest unvisited (segment, direction) from where the
    pen currently is; ties to the lowest node id.
    """
    N = 2 * n
    used = np.zeros(n, bool)
    tour, cur = [N], N
    for _ in range(n):
        row = C[cur, :N].copy()
        row[np.repeat(used, 2)] = np.inf
        k = int(np.argmin(row))
        if not np.isfinite(row[k]):
            raise RuntimeError("nearest neighbour found no reachable segment")
        tour.append(k)
        used[k // 2] = True
        cur = k
    return tour


def reversal_penalty(C, N):
    """R[a, b] = C[flip(b), flip(a)] - C[a, b], the cost of traversing an edge
    backwards with both its segments turned round.

    NaN where C is infinite (a pair no tour can contain), so a mistaken lookup
    poisons a comparison into False instead of quietly costing zero.
    """
    fl = flip_index(N)
    fin = np.isfinite(C)
    Cf = np.where(fin, C, 0.0)
    R = Cf[np.ix_(fl, fl)].T - Cf
    R[~fin] = np.nan
    return R


def _prefix(R, tour):
    """Cumulative reversal penalty along the tour's open edges."""
    t = np.asarray(tour, int)
    return np.concatenate([[0.0], np.cumsum(R[t[:-1], t[1:]])])


def two_opt_deltas(C, R, fl, tour, i, PR=None):
    """Cost change of reversing tour[i..j], for every j >= i. -> (m-i,) array.

    Two boundary edges are replaced, and every edge INSIDE the block is walked
    backwards with both of its segments turned round — that is what the prefix
    sum of `R` pays for in O(1).  j == i is the degenerate case: no internal
    edge, so it is exactly "draw this one segment the other way round".
    """
    t = np.asarray(tour, int)
    m = len(t)
    PR = _prefix(R, tour) if PR is None else PR
    js = np.arange(i, m)
    prev, a = t[i - 1], t[i]
    b, nxt = t[js], t[(js + 1) % m]
    return (C[prev, fl[b]] + C[fl[a], nxt] - C[prev, a] - C[b, nxt]
            + PR[js] - PR[i])


def apply_two_opt(tour, fl, i, j):
    """The move `two_opt_deltas` prices: walk the block backwards, turned round."""
    tour[i:j + 1] = [int(fl[x]) for x in reversed(tour[i:j + 1])]
    return tour


def or_opt_deltas(C, R, fl, tour, p, q, PR=None):
    """Cost of relocating the run tour[p..q] into every other gap.

    -> (fwd, rev, rest): `rest` is the tour without the run, and fwd[k] / rev[k]
    the cost change of dropping the run back in after `rest[k]`, the same way
    round / turned round.  k = p - 1 forward is the null move and prices at 0.
    """
    t = np.asarray(tour, int)
    m = len(t)
    PR = _prefix(R, tour) if PR is None else PR
    prev, after = t[p - 1], t[(q + 1) % m]
    first, last = t[p], t[q]
    gain = C[prev, first] + C[last, after] - C[prev, after]   # saved by removal
    rest = np.concatenate([t[:p], t[q + 1:]])
    uu = rest
    vv = rest[(np.arange(len(rest)) + 1) % len(rest)]
    base = C[uu, vv]
    fwd = C[uu, first] + C[last, vv] - base - gain
    rev = C[uu, fl[last]] + C[fl[first], vv] - base - gain + PR[q] - PR[p]
    return fwd, rev, rest


def apply_or_opt(tour, fl, p, q, k, reverse):
    """The move `or_opt_deltas` prices: the run goes in after position k of the
    tour-without-the-run."""
    run = tour[p:q + 1]
    if reverse:
        run = [int(fl[x]) for x in reversed(run)]
    body = tour[:p] + tour[q + 1:]
    tour[:] = body[:k + 1] + run + body[k + 1:]
    return tour


def _two_opt_pass(C, R, fl, tour, deadline):
    """One sweep of block reversals, best j for each i, applied as found."""
    moved = False
    i = 1
    while i < len(tour):
        delta = two_opt_deltas(C, R, fl, tour, i)
        k = int(np.argmin(delta))
        if delta[k] < -EPS:
            apply_two_opt(tour, fl, i, i + k)
            moved = True
        i += 1
        if time.time() > deadline:
            break
    return moved


def _or_opt_pass(C, R, fl, tour, deadline, or_max=OR_OPT_MAX):
    """One sweep of run relocations: lift 1..`or_max` consecutive segments out
    and drop them in anywhere else, forward or turned round."""
    moved = False
    m = len(tour)
    for L in range(1, min(or_max, m - 2) + 1):
        p = 1
        while p + L - 1 < len(tour):
            q = p + L - 1
            fwd, rev, _ = or_opt_deltas(C, R, fl, tour, p, q)
            kf, kr = int(np.argmin(fwd)), int(np.argmin(rev))
            take_rev = bool(rev[kr] < fwd[kf] - EPS)
            k, delta = (kr, rev[kr]) if take_rev else (kf, fwd[kf])
            if delta < -EPS:
                apply_or_opt(tour, fl, p, q, k, take_rev)
                moved = True
            p += 1
            if time.time() > deadline:
                return moved
    return moved


def descend(C, R, fl, tour, deadline, or_max=OR_OPT_MAX, max_rounds=100):
    """Alternate the two passes until neither finds anything. -> rounds used."""
    rounds = 0
    for rounds in range(1, max_rounds + 1):
        a = _two_opt_pass(C, R, fl, tour, deadline)
        b = _or_opt_pass(C, R, fl, tour, deadline, or_max)
        if not (a or b) or time.time() > deadline:
            break
    return rounds


def double_bridge(tour, rng):
    """The classic 4-opt kick: cut into four and re-join A C B D.

    No 2-opt and no Or-opt move can undo it in one step, which is exactly why
    it is the right way to leave a local optimum — a random restart throws away
    everything the descent learned, and a smaller perturbation gets walked
    straight back.  The depot stays at index 0 and no segment is turned round,
    so the result is always a valid tour.
    """
    m = len(tour)
    a, b, c = sorted(rng.choice(np.arange(1, m), size=3, replace=False))
    return tour[:a] + tour[b:c] + tour[a:b] + tour[c:]


def local_search(C, n, tour, budget=TIME_BUDGET, or_max=OR_OPT_MAX,
                 kick_seed=0, stall=60):
    """2-opt + Or-opt to a local optimum, then double-bridge kicks until the
    budget runs out or `stall` kicks in a row fail. -> (tour, stats).

    The kicks are what make the budget mean something: a descent from the
    greedy seed lands in the first local optimum it meets and stops, and on
    these instances that is routinely 15 % worse than optimal.  The kick
    sequence comes from a FIXED seed, so the answer is still a function of the
    cost matrix alone.
    """
    N = 2 * n
    fl = flip_index(N)
    R = reversal_penalty(C, N)
    t0 = time.time()
    deadline = t0 + budget
    tour = list(tour)
    rounds = descend(C, R, fl, tour, deadline, or_max)
    best, best_cost = list(tour), cycle_cost(C, tour)
    rng = np.random.default_rng(kick_seed)
    kicks = accepted = since = 0
    while n >= 4 and time.time() < deadline and since < stall:
        cand = double_bridge(best, rng)
        rounds += descend(C, R, fl, cand, deadline, or_max)
        kicks += 1
        c = cycle_cost(C, cand)
        if c < best_cost - EPS:
            best, best_cost, since, accepted = cand, c, 0, accepted + 1
        else:
            since += 1
    return best, dict(rounds=rounds, kicks=kicks, accepted=accepted,
                      wall=float(time.time() - t0),
                      stalled=bool(since >= stall))


# ==========================================================================
# 4. the entry point
# ==========================================================================
def seed_from(C, groups, ref):
    """A starting tour over `groups`, grown out of the tour `ref`. -> tour|None.

    THE DELTA THE BALANCER PRICES IS ONE SEGMENT, NOT ONE BAG.  Every candidate
    the load balancer considers is the arm's current bag with a span added,
    removed, or cut in two, and the tour it already has over that bag is a far
    better place to start a descent from than the greedy seed — the arm's other
    twenty spans are ordered the same way in both.  So the reference tour is
    filtered down to the segments this bag still has and the newcomers are
    dropped in at their cheapest position; the descent then does the rest.

    `groups[k]` is the interchangeable node ids of segment k (its two
    directions, and with fiber menus its variants too), in the SAME numbering as
    `C`.  `ref` may name nodes this bag does not have; they are skipped.  -> None
    when the result would be an infinite tour, which is the caller's signal to
    fall back to the full search rather than price a flyable bag as unflyable.
    """
    N = C.shape[0] - 1
    where = {int(x): k for k, g in enumerate(groups) for x in g}
    tour, seen = [N], set()
    for x in (ref or []):
        k = where.get(int(x))
        if k is not None and k not in seen:
            tour.append(int(x))
            seen.add(k)
    for k in range(len(groups)):
        if k in seen:
            continue
        best = None
        for pos in range(1, len(tour) + 1):
            prev = tour[pos - 1]
            nxt = tour[pos] if pos < len(tour) else N
            base = C[prev, nxt]
            for x in groups[k]:
                d = C[prev, int(x)] + C[int(x), nxt] - (base if np.isfinite(base)
                                                        else 0.0)
                if np.isfinite(d) and (best is None or d < best[0] - EPS):
                    best = (float(d), pos, int(x))
        if best is None:
            return None
        tour.insert(best[1], best[2])
    return tour if np.isfinite(cycle_cost(C, tour)) else None


def _greedy_or_insertion(C, groups, greedy):
    """The greedy seed, or cheapest insertion when greedy paints itself in.

    NEAREST NEIGHBOUR CAN FAIL ON A BAG THAT HAS A TOUR.  It only ever appends,
    so one early choice can leave a segment whose every remaining predecessor
    has been used up — and it then reports "no reachable segment" for an
    instance `seed_from` orders without trouble, because insertion may put a
    segment ANYWHERE in the partial tour rather than only at its end.  The
    greedy seed is still tried first, so every tour this module used to build it
    still builds; this is only what happens instead of an exception.
    """
    try:
        return greedy()
    except RuntimeError:
        tour = seed_from(C, groups, None)
        if tour is None:
            raise
        return tour


def solve(C, n, exact_max_n=EXACT_MAX_N, budget=TIME_BUDGET, or_max=OR_OPT_MAX,
          warm=None):
    """Cost matrix -> dict(order, dirs, cost, method, ...).

    Exact below `exact_max_n` segments, nearest neighbour + local search above.
    `warm` is a tour over a NEARBY bag (see `seed_from`): given one, the search
    starts from it instead of from the greedy seed, which is what makes pricing
    the balancer's next candidate cost a descent instead of a search.
    """
    t0 = time.time()
    if n == 0:
        return dict(order=[], dirs=[], cost=0.0, method="empty", n=0,
                    nn_cost=0.0, wall=0.0)
    if n <= exact_max_n:
        out = held_karp(C, n)
        out.update(n=n, nn_cost=float("nan"), wall=float(time.time() - t0))
        return out
    groups = [[2 * k, 2 * k + 1] for k in range(n)]
    tour = None if warm is None else seed_from(C, groups, warm)
    method = "warm+2opt+oropt" if tour is not None else "nn+2opt+oropt"
    if tour is None:
        tour = _greedy_or_insertion(C, groups, lambda: nearest_neighbour(C, n))
    nn_cost = cycle_cost(C, tour)
    tour, stats = local_search(C, n, tour, budget, or_max)
    order, dirs = _split(tour, n)
    return dict(order=order, dirs=dirs, cost=cycle_cost(C, tour), n=n,
                method=method, nn_cost=float(nn_cost), stats=stats,
                wall=float(time.time() - t0))


# ==========================================================================
# 5. clusters: the state is (segment, direction, VARIANT)
# ==========================================================================
# A segment used to offer two nodes.  With `menu.py` it offers 2 x V: each
# certified (entry, exit) fiber pair, drawn either way round.  Everything below
# is the same machinery over a longer node list — the SAME cost model, the same
# Held-Karp, the same 2-opt/Or-opt move set — plus one new move that swaps a
# segment's variant without touching the order.
#
# THE OBJECTIVE IS STILL MAKESPAN, AND THE TIE-BREAKS ARE INFINITESIMAL.  The
# matrix the DP minimises is the transit seconds the timeline will pay PLUS
# 1e-7 per radian of two things the clock cannot see: the reconfiguration
# ||q_exit - q_entry_next||_inf across each transit, and the interior travel
# surcharge of the variant being entered.  Both are weighted far below any
# transit difference that could matter (a whole tour's tie-break tops out
# around 1e-5 s against transits of seconds), so they can never buy a slower
# schedule — they only choose among schedules the clock calls equal.  And
# equal is the common case, not the rare one: `writing`'s lift, lower and
# travel beats are FLOORED at 0.20/0.20/0.25 s, so a joint move that fits
# inside the floor is free, and the arm has been wandering inside that freedom.
# `cluster_solve` reports `cost` re-derived from the pure transit matrix, so
# nothing downstream ever sees the tie-break — `csail_schedule.cross_check`
# compares it against `writing.arm_program`'s own total to 1e-6.
W_RECONFIG = 1e-7        # s per rad of ||q_exit - q_entry||_inf  (tie-break)
W_SURCHARGE = 1.0        # the interior surcharge is SECONDS; see below
EXACT_MAX_STATES = 20_000_000    # 2^n * nodes before Held-Karp is refused


def cluster_endpoints(spec, menus, h_inv=H_INV_DEFAULT, pen_ext=PEN_EXT):
    """Flatten per-segment menus into one node list. -> dict.

    Node `base[i] + 2*v + d` is segment i, variant v, drawn forward (d = 0) or
    backward (d = 1).  Putting the direction in the LOW bit is what lets
    `flip_index` stay exactly what it was: turning a segment round is still
    `node ^ 1`, so every asymmetric-cost move in section 3 works unchanged.

    -> dict(seg, var, dirn, base, nv, ent_q, exi_q, ent_xy, exi_xy, ent_h,
            exi_h, surcharge, n, N)
    """
    vlists = [list(m.variants) if hasattr(m, "variants") else list(m)
              for m in menus]
    if any(not v for v in vlists):
        raise ValueError("every segment needs at least one certified variant; "
                         "segment(s) "
                         f"{[i for i, v in enumerate(vlists) if not v]} have none")
    nv = [len(v) for v in vlists]
    base = np.concatenate([[0], np.cumsum([2 * k for k in nv])]).astype(int)
    N = int(base[-1])
    seg = np.zeros(N, int)
    var = np.zeros(N, int)
    dirn = np.zeros(N, int)
    ent_q = np.zeros((N, 7))
    exi_q = np.zeros((N, 7))
    ent_xy = np.zeros((N, 2))
    exi_xy = np.zeros((N, 2))
    sur = np.zeros(N)
    ent_h = np.zeros((N, 7))
    exi_h = np.zeros((N, 7))
    for i, vs in enumerate(vlists):
        for v, x in enumerate(vs):
            # a variant has two ends, not four: the hover above each is solved
            # ONCE and handed to both directions, which is the same call at the
            # same heights `writing.arm_program` will make when it lays the
            # transit down (that identity is what `cross_check` relies on).
            hov = {}
            for e_ in ("entry", "exit"):
                hov[e_], _ = lifted_or_lower(spec, np.asarray(x[f"{e_}_q"], float),
                                             np.asarray(x[f"{e_}_xy"], float),
                                             h_inv=h_inv, pen_ext=pen_ext)
            for d in (0, 1):
                k = base[i] + 2 * v + d
                seg[k], var[k], dirn[k] = i, v, d
                sur[k] = float(x.get("surcharge", 0.0))
                # drawn backward, the plan's LAST sample is the one entered at
                a, b = ("entry", "exit") if d == 0 else ("exit", "entry")
                ent_q[k] = np.asarray(x[f"{a}_q"], float)
                exi_q[k] = np.asarray(x[f"{b}_q"], float)
                ent_xy[k] = np.asarray(x[f"{a}_xy"], float)
                exi_xy[k] = np.asarray(x[f"{b}_xy"], float)
                ent_h[k], exi_h[k] = hov[a], hov[b]
    return dict(seg=seg, var=var, dirn=dirn, base=base, nv=nv, ent_q=ent_q,
                exi_q=exi_q, ent_xy=ent_xy, exi_xy=exi_xy, ent_h=ent_h,
                exi_h=exi_h, surcharge=sur, n=len(menus), N=N)


def reconfig_block(exi_q, ent_q, w_reconfig=1e-7):
    """The (M, N) reconfiguration tie-break, ||q_exit - q_entry||_inf x weight."""
    if not len(exi_q) or not len(ent_q):
        return np.zeros((len(exi_q), len(ent_q)))
    return w_reconfig * np.max(np.abs(np.asarray(exi_q)[:, None, :]
                                      - np.asarray(ent_q)[None, :, :]), axis=-1)


def cluster_cost_matrix(spec, menus, transit_speed=TRANSIT_SPEED,
                        qd_frac=QD_FRAC, h_inv=H_INV_DEFAULT, pen_ext=PEN_EXT,
                        q_start=None, return_home=True, ends=None,
                        w_reconfig=W_RECONFIG, w_surcharge=W_SURCHARGE,
                        paper_safe=True):
    """Every transit an ordering-and-variant choice could pay. -> (C, T, ends).

    `C` is the pure transit seconds, built by the SAME four array operations
    as `cost_matrix` — same `lift`/`lower` row times, same pen-tip hop floor,
    same `writing.dq_time_many` for the hover-to-hover move — so a one-variant
    menu reproduces `cost_matrix` cell for cell.  `T` is the infinitesimal
    tie-break described above, and is what the search adds to `C`; nothing
    reports it.
    """
    e = cluster_endpoints(spec, menus, h_inv, pen_ext) if ends is None else ends
    N = e["N"]
    if N == 0:
        return np.zeros((1, 1)), np.zeros((1, 1)), e
    ent_q, ent_h, ent_xy = e["ent_q"], e["ent_h"], e["ent_xy"]
    exi_q, exi_h, exi_xy = e["exi_q"], e["exi_h"], e["exi_xy"]

    lift = node_lift(spec, exi_q, exi_h, qd_frac, h_inv, pen_ext, paper_safe)
    lower = node_lower(spec, ent_h, ent_q, qd_frac, h_inv, pen_ext, paper_safe)
    same = e["seg"][:, None] == e["seg"][None, :]
    C = np.full((N + 1, N + 1), np.inf)
    C[:N, :N] = leg_costs(spec, exi_h, exi_xy, ent_h, ent_xy, lift, lower, same,
                          transit_speed, qd_frac, h_inv, pen_ext, paper_safe)
    C[N, :N], C[:N, N] = depot_legs(spec, ent_h, exi_h, lift, lower, qd_frac,
                                    h_inv, pen_ext, q_start, return_home,
                                    paper_safe)

    # the tie-break: reconfiguration on the edge, surcharge on the node entered
    T = np.zeros((N + 1, N + 1))
    T[:N, :N] = reconfig_block(exi_q, ent_q, w_reconfig)
    # THE INTERIOR IS NOT INVARIANT, AND ASSUMING IT WAS COST 47 % OF THE CLOCK.
    # The first version of this priced only transit, on the argument that every
    # variant of a stroke draws the same polyline at the same speed.  It does
    # not: a variant is a different path through the band, so it has a
    # different |dq/ds|, and `writing.draw_duration` stretches the ink until no
    # joint exceeds `qd_frac` of its limit.  Measured on the CSAIL grey phase,
    # choosing fibers on transit alone moved the busiest arm's DRAW time
    # 20.2 -> 30.9 s while saving 8 s of transit across the whole fleet — a
    # trade the clock refuses.  With `pwl.TRAVEL_MODE = "time"` the band DP
    # already costs an edge in seconds at full joint speed, so the menu's
    # surcharge IS the extra draw time this variant will cost, and it enters at
    # full weight, scaled by the same `qd_frac` the timeline will apply.
    T[:, :N] += (w_surcharge / max(qd_frac, 1e-6)) * e["surcharge"][None, :]
    if not return_home:
        # WHERE AN ARM STOPS IS A DECISION, AND UNDER FREEZE NOBODY WAS MAKING
        # IT.  With `return_home=False` the last leg costs the lift and nothing
        # else, so `C[a, N]` is the SAME number for every node and the DP is
        # perfectly indifferent about which stroke it finishes on and which
        # fiber it comes off.  That was harmless when a segment offered two
        # nodes; with 2 x V it makes the set of equal-transit tours enormous,
        # the search picks an arbitrary member, and the arm freezes in an
        # arbitrary posture — which the conductor then has to certify as an
        # OBSTACLE for the rest of the run, and refuses ("frozen pose sits in
        # another arm's tube").  So the same infinitesimal tie-break that
        # prices reconfiguration on every other edge is applied to the last
        # one, measured against the ready pose: among tours the clock cannot
        # separate, finish in the posture nearest the one the arm is known to
        # be able to stand in.  It buys no seconds and is not meant to; it
        # replaces an arbitrary choice with a defensible one.
        home = np.repeat(np.asarray(spec.q_seed, float)[None, :], N, axis=0)
        T[:N, N] += w_reconfig * np.max(np.abs(exi_q - home), axis=-1)
    T[~np.isfinite(C)] = 0.0
    return C, T, e


def cluster_held_karp(C, e):
    """Exact order, direction AND variant. -> dict.

    dp[mask, node] is the cheapest way to leave the depot, draw exactly the
    segments in `mask`, and be standing at `node`'s exit.  Identical in shape
    to `held_karp`; the only difference is that a segment now contributes
    2 x V nodes instead of 2, and choosing one commits its direction AND its
    entry/exit fiber.

    The relaxation is done for every successor node at once — each node belongs
    to exactly one segment, so `mask | (1 << seg[node])` is a per-node target
    mask and the scatter has no duplicate (mask, node) pairs to collide on.
    That keeps the inner loop six array operations instead of one per segment.
    """
    seg, N, n = e["seg"], e["N"], e["n"]
    if n == 0:
        return dict(order=[], dirs=[], variants=[], cost=0.0, states=0,
                    method="cluster_held_karp")
    T = C[:N, :N]
    dp = np.full((1 << n, N), np.inf)
    par = np.full((1 << n, N), -1, np.int32)
    nodes = np.arange(N)
    dp[1 << seg, nodes] = C[N, :N]
    seg_bit = (1 << seg).astype(np.int64)
    for mask in range(1, 1 << n):
        row = dp[mask]
        if not np.isfinite(row).any():
            continue
        cand = row[:, None] + T
        best = cand.min(axis=0)
        arg = cand.argmin(axis=0)
        free = (mask & seg_bit) == 0
        if not free.any():
            continue
        nms = (mask | seg_bit)[free]
        nds = nodes[free]
        b = best[free]
        upd = b < dp[nms, nds]
        if upd.any():
            dp[nms[upd], nds[upd]] = b[upd]
            par[nms[upd], nds[upd]] = arg[free][upd]
    full = (1 << n) - 1
    tot = dp[full] + C[:N, N]
    if not np.isfinite(tot).any():
        raise RuntimeError(f"no feasible order over {n} segments")
    k = int(np.argmin(tot))
    body, mask = [], full
    while k >= 0:
        body.append(k)
        p = int(par[mask, k])
        mask ^= 1 << int(seg[k])
        k = p
    body.reverse()
    return dict(cost=float(tot[int(np.argmin(tot))]), nodes=body,
                states=int((1 << n) * N), method="cluster_held_karp",
                **_cluster_split(body, e))


def _cluster_split(nodes, e):
    """Node ids -> (order, dirs, variants)."""
    seg, var, dirn = e["seg"], e["var"], e["dirn"]
    return dict(order=[int(seg[k]) for k in nodes],
                dirs=[1 if dirn[k] == 0 else -1 for k in nodes],
                variants=[int(var[k]) for k in nodes])


def cluster_siblings(e):
    """node -> the other variants of the same segment, same direction."""
    base, nv, N = e["base"], e["nv"], e["N"]
    out = []
    for k in range(N):
        i, v, d = int(e["seg"][k]), int(e["var"][k]), int(e["dirn"][k])
        out.append(np.array([base[i] + 2 * w + d for w in range(nv[i])
                             if w != v], int))
    return out


def _variant_pass(C, tour, sibs, deadline):
    """One sweep of variant re-selection: keep the order and both neighbours,
    swap which certified fiber pair this segment is drawn on.

    THE MOVE THE OLD SEARCH COULD NOT MAKE.  2-opt re-orders and Or-opt
    relocates; both can only pick from the two nodes a segment offered.  With a
    menu, the cheapest thing to do is often to draw the same stroke in the same
    place at the same time and simply come off it somewhere else, which is a
    move in neither of the other two neighbourhoods.  It is also the cheapest
    to price — two edges — so it runs in the same descent loop as the others.
    """
    moved = False
    m = len(tour)
    for p in range(1, m):
        k = tour[p]
        sib = sibs[k]
        if not len(sib):
            continue
        prev, nxt = tour[p - 1], tour[(p + 1) % m]
        cur = C[prev, k] + C[k, nxt]
        alt = C[prev, sib] + C[sib, nxt]
        j = int(np.argmin(alt))
        if alt[j] < cur - EPS:
            tour[p] = int(sib[j])
            moved = True
        if time.time() > deadline:
            break
    return moved


def cluster_descend(C, R, fl, tour, sibs, deadline, or_max=OR_OPT_MAX,
                    max_rounds=100):
    """2-opt, Or-opt and variant re-selection until none of the three bites."""
    rounds = 0
    for rounds in range(1, max_rounds + 1):
        a = _two_opt_pass(C, R, fl, tour, deadline)
        b = _or_opt_pass(C, R, fl, tour, deadline, or_max)
        c = _variant_pass(C, tour, sibs, deadline)
        if not (a or b or c) or time.time() > deadline:
            break
    return rounds


def cluster_nearest_neighbour(C, e):
    """Greedy seed over nodes, one segment each. -> tour (depot first)."""
    seg, N, n = e["seg"], e["N"], e["n"]
    used = np.zeros(n, bool)
    tour, cur = [N], N
    for _ in range(n):
        row = C[cur, :N].copy()
        row[used[seg]] = np.inf
        k = int(np.argmin(row))
        if not np.isfinite(row[k]):
            raise RuntimeError("nearest neighbour found no reachable segment")
        tour.append(k)
        used[seg[k]] = True
        cur = k
    return tour


def cluster_local_search(C, e, tour, budget=TIME_BUDGET, or_max=OR_OPT_MAX,
                         kick_seed=0, stall=60):
    """Descent + double-bridge kicks, with variant re-selection in the move set."""
    N, n = e["N"], e["n"]
    fl = flip_index(N)
    R = reversal_penalty(C, N)
    sibs = cluster_siblings(e)
    t0 = time.time()
    deadline = t0 + budget
    tour = list(tour)
    rounds = cluster_descend(C, R, fl, tour, sibs, deadline, or_max)
    best, best_cost = list(tour), cycle_cost(C, tour)
    rng = np.random.default_rng(kick_seed)
    kicks = accepted = since = 0
    while n >= 4 and time.time() < deadline and since < stall:
        cand = double_bridge(best, rng)
        rounds += cluster_descend(C, R, fl, cand, sibs, deadline, or_max)
        kicks += 1
        c = cycle_cost(C, cand)
        if c < best_cost - EPS:
            best, best_cost, since, accepted = cand, c, 0, accepted + 1
        else:
            since += 1
    return best, dict(rounds=rounds, kicks=kicks, accepted=accepted,
                      wall=float(time.time() - t0),
                      stalled=bool(since >= stall))


def cluster_groups(e):
    """`seed_from`'s groups for a cluster bag: every node of each segment."""
    base, nv = np.asarray(e["base"], int), list(e["nv"])
    return [list(range(int(base[k]), int(base[k]) + 2 * int(nv[k])))
            for k in range(int(e["n"]))]


def cluster_solve(C, T, e, exact_max_n=EXACT_MAX_N, budget=TIME_BUDGET,
                  or_max=OR_OPT_MAX, exact_max_states=EXACT_MAX_STATES,
                  warm=None):
    """(C, T, ends) -> dict(order, dirs, variants, cost, method, ...).

    The search runs on `C + T`; `cost` is re-derived from `C` alone, so the
    number returned is the transit seconds the timeline will pay and the
    tie-break never leaves this function.  `warm` is `seed_from`'s reference
    tour; see `solve`.
    """
    t0 = time.time()
    n, N = e["n"], e["N"]
    if n == 0:
        return dict(order=[], dirs=[], variants=[], cost=0.0, method="empty",
                    n=0, nn_cost=0.0, wall=0.0)
    W = C + T
    exact = n <= exact_max_n and (1 << n) * max(N, 1) <= exact_max_states
    if exact:
        out = cluster_held_karp(W, e)
        out.update(n=n, nn_cost=float("nan"), wall=float(time.time() - t0),
                   cost=cycle_cost(C, tour_of_nodes(out["nodes"], N)))
        return out
    groups = cluster_groups(e)
    tour = None if warm is None else seed_from(W, groups, warm)
    method = ("cluster_warm+2opt+oropt+variant" if tour is not None
              else "cluster_nn+2opt+oropt+variant")
    if tour is None:
        tour = _greedy_or_insertion(W, groups,
                                    lambda: cluster_nearest_neighbour(W, e))
    nn_cost = cycle_cost(C, tour)
    tour, stats = cluster_local_search(W, e, tour, budget, or_max)
    body = [k for k in tour if k != N]
    return dict(cost=cycle_cost(C, tour), n=n, nodes=body,
                method=method, nn_cost=float(nn_cost),
                stats=stats, wall=float(time.time() - t0),
                **_cluster_split(body, e))


def tour_of_nodes(nodes, N):
    """Node ids -> the closed tour, depot first."""
    return [N] + list(nodes)


def reconfiguration(segs):
    """Sum of ||q_exit - q_entry_next||_inf over consecutive segments. -> rad.

    THE QUANTITY THE CLOCK CANNOT SEE, AND THE ONE THE ARM IS DOING.  Between
    two strokes the arm lifts, flies to the next hover and lowers, and
    `writing`'s beats FLOOR each of those at 0.20/0.25/0.20 s.  A pen-up whose
    joint move fits inside its floor therefore costs the same as one that
    barely moves at all — so the transit-seconds objective is blind, over a
    wide band, to how far round the null space the arm actually swings.  This
    measures that swing directly: the largest single-joint jump between the
    configuration a segment ends in and the one the next begins in, summed
    along the programme.

    It is a diagnostic, not a gate and not a term in the objective.  It is
    reported because it is what a person watching the rig sees — an arm that
    finishes a stroke and then rolls its wrist most of a turn before starting
    the next one — and because the fiber menus exist to lower it.
    """
    segs = list(segs)
    tot = 0.0
    for a, b in zip(segs[:-1], segs[1:]):
        qa = np.asarray(a["plan"]["qs"], float)[-1]
        qb = np.asarray(b["plan"]["qs"], float)[0]
        tot += float(np.max(np.abs(qa - qb)))
    return tot


def order_arm(spec, segs, transit_speed=TRANSIT_SPEED, qd_frac=QD_FRAC,
              h_inv=H_INV_DEFAULT, exact_max_n=EXACT_MAX_N, budget=TIME_BUDGET,
              baseline=None):
    """One arm's segments -> the order and directions to draw them in.

    `baseline` (an order, all forward implied, or a (order, dirs) pair) is
    costed on the same matrix and returned as `baseline_cost`, so "how much did
    the sequencer save" is a difference of two numbers from one model rather
    than a comparison of two models.
    """
    n = len(segs)
    t0 = time.time()
    if n == 0:
        return dict(order=[], dirs=[], cost=0.0, method="empty", n=0,
                    baseline_cost=0.0, wall=0.0, C=np.zeros((1, 1)))
    C = cost_matrix(spec, segs, transit_speed, qd_frac, h_inv)
    t_mat = time.time() - t0
    out = solve(C, n, exact_max_n, budget)
    if baseline is not None:
        order, dirs = (baseline if isinstance(baseline, tuple)
                       else (list(baseline), [1] * n))
        out["baseline_cost"] = sequence_cost(C, n, order, dirs)
    out.update(C=C, matrix_wall=float(t_mat), wall=float(time.time() - t0))
    return out


def report(per_arm):
    """{arm: order_arm result} -> terse lines."""
    out = [f"{'arm':>5} {'segs':>5} {'method':>14} {'transit':>9} "
           f"{'baseline':>9} {'saved':>8} {'wall':>7}"]
    for a in sorted(per_arm):
        r = per_arm[a]
        b = r.get("baseline_cost")
        sav = "" if b is None or not b else f"{100 * (b - r['cost']) / b:6.1f} %"
        out.append(f"{a:>5} {r['n']:>5} {r['method']:>14} {r['cost']:>8.2f}s "
                   f"{('-' if b is None else f'{b:8.2f}s'):>9} {sav:>8} "
                   f"{r.get('wall', 0.0):>6.2f}s")
    return out
