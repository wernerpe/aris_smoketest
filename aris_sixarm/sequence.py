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
import time

import numpy as np

from .fleet import H_INV_DEFAULT
from .frames import PEN_EXT, QD_MAX
from .writing import (QD_FRAC, T_HOME_F, T_LIFT_F, T_LOWER_F, T_TRAVEL_MIN,
                      TRANSIT_SPEED, dq_time_many, lifted_or_lower)

EXACT_MAX_N = 16         # segments solved exactly by Held-Karp
TIME_BUDGET = 2.0        # s of local search per arm, above EXACT_MAX_N
OR_OPT_MAX = 3           # longest run Or-opt relocates
EPS = 1e-9               # s; an "improvement" below this is float noise


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
        for e, k in ((0, 0), (1, -1)):
            q[i, e], xy[i, e] = qs[k], pts[k]
            h, zz = lifted_or_lower(spec, qs[k], pts[k], h_inv=h_inv,
                                    pen_ext=pen_ext)
            hov[i, e], z[i, e] = h, zz
    return dict(q=q, xy=xy, hover=hov, z=z, n=n)


def cost_matrix(spec, segs, transit_speed=TRANSIT_SPEED, qd_frac=QD_FRAC,
                h_inv=H_INV_DEFAULT, ends=None, pen_ext=PEN_EXT):
    """Every transit time an ordering could possibly pay. -> (2n+1, 2n+1).

    Node `2i + d` is segment i drawn forward (d = 0) or backward (d = 1); node
    `2n` is the depot, the arm's ready pose.  `C[a, b]` is the seconds from
    node a's EXIT to node b's ENTRY, hover overhead included; `C[2n, b]` is the
    entry lift from the ready pose and `C[a, 2n]` the exit lift back to it.
    A segment cannot follow itself in either direction, so those cells are inf.

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

    lift = _row_time(exi_q, exi_h, qd_frac, T_LIFT_F)        # (N,)
    lower = _row_time(ent_h, ent_q, qd_frac, T_LOWER_F)      # (N,)
    hop = np.linalg.norm(ent_xy[None, :, :] - exi_xy[:, None, :], axis=-1)
    floor = np.maximum(T_TRAVEL_MIN, hop / max(transit_speed, 1e-9))
    travel = dq_time_many(exi_h, ent_h, qd_frac, floor)      # (N,N)

    C = np.full((N + 1, N + 1), np.inf)
    C[:N, :N] = lift[:, None] + travel + lower[None, :]
    seg = np.arange(N) // 2
    C[:N, :N][seg[:, None] == seg[None, :]] = np.inf         # no self-succession
    home = np.repeat(np.asarray(spec.q_seed, float)[None, :], N, axis=0)
    C[N, :N] = _row_time(home, ent_h, qd_frac, T_HOME_F) + lower
    C[:N, N] = lift + _row_time(exi_h, home, qd_frac, T_HOME_F)
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
    """
    N = 2 * n
    if n == 0:
        return dict(order=[], dirs=[], cost=0.0, states=0, method="held_karp")
    if n == 1:
        k = int(np.argmin([C[N, 0] + C[0, N], C[N, 1] + C[1, N]]))
        return dict(order=[0], dirs=[1 if k == 0 else -1],
                    cost=float(C[N, k] + C[k, N]), states=2, method="held_karp")
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
def solve(C, n, exact_max_n=EXACT_MAX_N, budget=TIME_BUDGET, or_max=OR_OPT_MAX):
    """Cost matrix -> dict(order, dirs, cost, method, ...).

    Exact below `exact_max_n` segments, nearest neighbour + local search above.
    """
    t0 = time.time()
    if n == 0:
        return dict(order=[], dirs=[], cost=0.0, method="empty", n=0,
                    nn_cost=0.0, wall=0.0)
    if n <= exact_max_n:
        out = held_karp(C, n)
        out.update(n=n, nn_cost=float("nan"), wall=float(time.time() - t0))
        return out
    tour = nearest_neighbour(C, n)
    nn_cost = cycle_cost(C, tour)
    tour, stats = local_search(C, n, tour, budget, or_max)
    order, dirs = _split(tour, n)
    return dict(order=order, dirs=dirs, cost=cycle_cost(C, tour), n=n,
                method="nn+2opt+oropt", nn_cost=float(nn_cost), stats=stats,
                wall=float(time.time() - t0))


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
