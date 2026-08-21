"""Conductor v1: make six frozen per-arm timelines safe to run at once.

THE ONE THING THIS MODULE MAY CHANGE IS THE CLOCK.  Every arm's path — its
segment order, the certified joint trajectory of each segment, the hover
transits between them — arrives frozen from `writing.arm_program`, and leaves
frozen.  All the conductor does is decide, for each instant, whether an arm
advances along its path or waits where it is.  That keeps every guarantee the
per-segment planner earned (`stroke_api`'s certificate is about a path, not a
schedule) and makes the coordination problem small enough to solve exactly.

  MODEL     Each arm is 7 capsules built from `frames.fk`'s own chain points:
            base column, upper arm, elbow offset, forearm (r = LINK_R), wrist
            and hand (r = WRIST_R), and the pen (r = PEN_R).  These are
            CONSERVATIVE envelopes of the real links, not CAD.  Two arms are
            "clear" when every capsule pair is at least
            `safety + calib` apart:
              safety (default 0.05 m) is the operating margin;
              calib  (default 0.03 m) is what we owe the four bases whose XY
                     came from a preset file and has never been surveyed.
            Raise `calib` for a rig you have not measured; drop it to zero the
            day a survey lands.  Neither number is a guess about the arm — it
            is a guess about our knowledge of where the arm is.

  IMAGE     For each pair, the collision image over PROGRESS INDICES: cell
            (a, b) is free iff arm i anywhere in [a, a+1] and arm j anywhere in
            [b, b+1] are clear.  Cells, not points, is what makes this
            tunnel-proof: the cell's clearance is the smallest of its four
            corner clearances minus the two half-step motions (clearance is
            1-Lipschitz in point displacement), so nothing can pass through
            anything between two samples.

  SCHEDULE  Parked arms first (they are obstacles, and their schedule is a
            constant), then the moving arms in a PRIORITY ORDER.  Each arm
            gets the earliest-arrival monotone schedule that stays in free
            cells against every arm already scheduled — which, because the
            parked ones come first, means against every arm in the rig.  That
            is a reachability DP over (progress index x time), exact for the
            "advance or wait" move set — no local heuristic, no iteration to
            convergence, and it reports infeasibility instead of shipping an
            unsafe timeline.  A conflict with a PARKED arm is reported rather
            than scheduled around, because waiting cannot resolve one.

  PRIORITY  The order is SEARCHED, not guessed.  Priority is the conductor's
            only free variable and it is worth as much as the safety margin:
            on the CSAIL logo's phase 1 only 3 of the 24 orders are feasible at
            all and "busiest first, promote whoever deadlocks" landed on the
            worst of the three (63.5 s against 48.3 s).  With at most
            `PRIORITY_SEARCH_MAX` moving arms every permutation is enumerated
            and the minimum-makespan one kept (`_search_priority`); above that
            the old busiest-first heuristic with deadlock promotion is the
            fallback, because 7! schedules is no longer cheap.

  IDLE      An arm that has arrived has NOT left: the rest suffix in `_dp`
            keeps checking its final pose against everyone still moving, which
            is correct and is also what made three quarters of the fleet's
            pauses "waiting for permission to go home".  WHERE an arm stops is
            not this module's choice — `idle.py` makes it and hands the paths
            in — but two things here serve it: `rest_delays` measures what the
            rest rule cost each arm (by re-running the same DP with the rule
            dropped), and `hard_blocks` names the progress indices no schedule
            can ever reach, which is the difference between "this is slow" and
            "this cannot run".

  NOT v1    No re-timing of a stroke, no path change, and no dynamics.  The
            only recovery from a deadlock IN HERE is re-ordering the priorities
            and trying again — exhaustively where that is affordable, by
            promoting the arm that failed (`retry_orders`) where it is not.
            Either way it resolves an ordering deadlock and cannot resolve a
            geometric one; a refusal now carries the evidence (`err.free`,
            `err.paths`) so a caller that CAN change a path — the sequencer, via
            `idle.unrunnable` — gets told which one.  A pause here is
            instantaneous in the animation's kinematic playback; a real run
            needs the acceleration-limited version of the same schedule.
"""
import math
import time

import numpy as np

from .frames import PEN_EXT, fk_many
from .fleet import FLEET, H_INV_DEFAULT
from .rig_final import PEN_R_FINAL

LINK_R = 0.09        # m, capsule radius for base/upper arm/forearm
WRIST_R = 0.07       # m, wrist + hand
PEN_R = 0.03         # m, the pen itself
SAFETY_M = 0.05      # m, operating clearance between two arms
CALIB_M = 0.03       # m, unsurveyed base positions (see module docstring)
SWEEP_K = 0.55       # sweep slack factor: 0.5 for the chord, +10 % for the arc
BROAD_CAP = 0.25     # m, clearances above this are not computed exactly
PRIORITY_SEARCH_MAX = 6   # moving arms whose 6! = 720 orders are all enumerated

# (chain point i, chain point j, radius); indices into the 10-point chain
# (frames.fk's 9 points + the pen tip).  Points 1/2 and 5/6 coincide by
# construction (zero DH offset), so they are not given their own capsule.
CAPSULES = ((0, 1, LINK_R), (1, 3, LINK_R), (3, 4, LINK_R), (4, 5, LINK_R),
            (5, 7, WRIST_R), (7, 8, WRIST_R), (8, 9, PEN_R))


# ==========================================================================
# geometry
# ==========================================================================
def chain_world(qs, spec, h_inv=H_INV_DEFAULT, pen_ext=PEN_EXT):
    """(N,7) joints -> (N,10,3) chain points in WORLD, pen tip included."""
    qs = np.asarray(qs, float).reshape(-1, 7)
    T, P = fk_many(qs)
    tip = T[:, :3, 3] + T[:, :3, :3] @ np.array([0.0, 0.0, pen_ext])
    P = np.concatenate([P, tip[:, None, :]], axis=1)
    Twb = spec.T_world_base(h_inv)
    return P @ Twb[:3, :3].T + Twb[:3, 3]


def seg_seg_dist(p0, p1, q0, q1):
    """Distance between segments [p0,p1] and [q0,q1], broadcast over leading axes.

    Ericson's clamped parametrisation (Real-Time Collision Detection 5.1.9),
    written out so it vectorises: solve the unconstrained least squares for the
    two parameters, clamp each into [0,1], and re-solve the other against the
    clamp.  The degenerate cases (either segment a point, the two parallel) fall
    out of the same expression once the denominator is floored.
    """
    d1, d2, r = p1 - p0, q1 - q0, p0 - q0
    a = np.sum(d1 * d1, -1)
    e = np.sum(d2 * d2, -1)
    f = np.sum(d2 * r, -1)
    b = np.sum(d1 * d2, -1)
    c = np.sum(d1 * r, -1)
    den = a * e - b * b
    s = np.where(den > 1e-12, np.clip((b * f - c * e) / np.where(den > 1e-12, den, 1.0),
                                      0.0, 1.0), 0.0)
    t = (b * s + f) / np.maximum(e, 1e-12)
    t_c = np.clip(t, 0.0, 1.0)
    s = np.clip((b * t_c - c) / np.maximum(a, 1e-12), 0.0, 1.0)
    t = np.clip((b * s + f) / np.maximum(e, 1e-12), 0.0, 1.0)
    w = r + s[..., None] * d1 - t[..., None] * d2
    return np.sqrt(np.maximum(np.sum(w * w, -1), 0.0))


class ArmPath:
    """One arm's frozen path, pre-chewed for the collision image."""

    def __init__(self, arm_id, q, dt, h_inv=H_INV_DEFAULT, pen_ext=PEN_EXT,
                 spec=None):
        spec = FLEET[arm_id] if spec is None else spec
        self.arm, self.dt = arm_id, float(dt)
        self.q = np.asarray(q, float).reshape(-1, 7)
        if len(self.q) < 2:                       # a parked arm still occupies space
            self.q = np.vstack([self.q, self.q[-1:]])
        P = chain_world(self.q, spec, h_inv, pen_ext)
        self.n = len(P)
        self.A = np.ascontiguousarray(P[:, [c[0] for c in CAPSULES], :], np.float32)
        self.B = np.ascontiguousarray(P[:, [c[1] for c in CAPSULES], :], np.float32)
        self.r = np.array([c[2] for c in CAPSULES], np.float32)
        if getattr(spec, "rig", "sixarm") == "final":
            # the pen capsule carries the HOLDER envelope union (both CAD
            # builds, clutch extended): r 0.05, not the bare-pen 0.03
            self.r = self.r.copy()
            self.r[-1] = PEN_R_FINAL
        self.center = P.mean(axis=1).astype(np.float32)
        self.radius = (np.linalg.norm(P - P.mean(axis=1, keepdims=True), axis=2).max(1)
                       + self.r.max()).astype(np.float32)
        self.step = np.linalg.norm(np.diff(P, axis=0), axis=2).max(1).astype(np.float32)
        self.moves = bool(np.max(np.abs(self.q - self.q[0])) > 1e-9)
        self.motion = float(self.step.sum())


def arm_paths(q_by_arm, dt, h_inv=H_INV_DEFAULT, pens=None, fleet=None):
    """{arm: (N,7)} -> {arm: ArmPath}, each built with THAT ARM's pen.

    The pen is the last capsule of the chain, so the length is geometry and not
    bookkeeping: conducting a fleet in which arm 31 carries 300 mm against a
    110 mm capsule would schedule 19 cm of the arm out of the collision image
    entirely.  `pens` is {arm_id: metres}; an arm it does not name keeps
    `frames.PEN_EXT`.
    """
    pens = pens or {}
    fl = FLEET if fleet is None else fleet
    return {a: ArmPath(a, q, dt, h_inv, float(pens.get(a, PEN_EXT)), fl[a])
            for a, q in q_by_arm.items()}


def clearance_matrix(pi, pj, cap=BROAD_CAP, chunk=15000):
    """(Ni, Nj) capsule-to-capsule clearance, CLIPPED at `cap`.

    The clip is the whole point: a broad phase on per-sample bounding spheres
    throws away every pair that cannot possibly be closer than `cap`, and only
    the survivors pay for 49 exact segment distances.  Anything reported as
    `cap` is "at least cap", which is all a margin test needs to know.
    """
    Ni, Nj = pi.n, pj.n
    D = np.full((Ni, Nj), cap, np.float32)
    rr = (pi.r[:, None] + pj.r[None, :]).astype(np.float32)
    rows = max(1, int(2e6 // max(Nj, 1)))
    for a0 in range(0, Ni, rows):
        a1 = min(a0 + rows, Ni)
        d = pi.center[a0:a1, None, :] - pj.center[None, :, :]
        gap = (np.sqrt(np.einsum("ijk,ijk->ij", d, d))
               - pi.radius[a0:a1, None] - pj.radius[None, :])
        ia, ib = np.nonzero(gap < cap)
        if not len(ia):
            continue
        ia = ia + a0
        for k0 in range(0, len(ia), chunk):
            u, v = ia[k0:k0 + chunk], ib[k0:k0 + chunk]
            d = seg_seg_dist(pi.A[u][:, :, None, :], pi.B[u][:, :, None, :],
                             pj.A[v][:, None, :, :], pj.B[v][:, None, :, :]) - rr
            D[u, v] = np.minimum(d.reshape(len(u), -1).min(1), cap)
    return D


def free_cells(pi, pj, margin, sweep=SWEEP_K, cap=BROAD_CAP):
    """(Ni-1, Nj-1) boolean: is the whole cell clear by `margin`?

    Clearance is 1-Lipschitz in the displacement of the bodies, so the cell's
    worst case is bounded by its best corner minus the two half-step motions.
    `sweep` > 0.5 pays for the arc-vs-chord difference of a rotating link.
    """
    D = clearance_matrix(pi, pj, cap)
    S = np.minimum(np.minimum(D[:-1, :-1], D[1:, :-1]),
                   np.minimum(D[:-1, 1:], D[1:, 1:]))
    S -= sweep * (pi.step[:, None] + pj.step[None, :])
    return S >= margin


# ==========================================================================
# scheduling
# ==========================================================================
def _dp(free_ab, prog_hi, n, horizon, deadline=None, rest=True):
    """Earliest-arrival monotone schedule for one arm. -> (progress (M,), m_end).

    `free_ab` maps each already-scheduled arm to its (n-1, nb-1) free-cell
    image; `prog_hi` maps it to its progress at every time step.  State is
    (progress index, time step); the only moves are "advance one index" and
    "wait", which is exactly what a pause schedule is allowed to do.

    `deadline` (a step index) says "do not bother unless this arm can arrive by
    then", and is how the priority search pays for itself: an order whose arm
    finishes later than the best complete order already found cannot win, so
    the forward pass is allocated and run over `deadline + 1` columns instead
    of the whole horizon and returns `(None, None)` if it does not arrive.  It
    is a BOUND, not a shortcut — the rest-suffix row below is still built over
    the WHOLE horizon, because the run does not end at the deadline and an arm
    that arrives has to stand in its final pose until it does.  With
    `deadline=None` this is exactly the unbounded search, to the index.
    """
    M = horizon
    D = M if deadline is None else int(min(max(deadline, 0) + 1, M))
    # AN ARM THAT HAS FINISHED HAS NOT LEFT.  Arrival is not the end of the
    # arm's participation: it then stands at the last sample of its path for
    # the rest of the run while other arms are still moving, and the earliest
    # arrival is only safe if that resting pose stays clear too.  Requiring the
    # suffix of the last row is what makes "arrives at m_end" mean "is safe
    # from 0 to the horizon" rather than "is safe until it stops".  It is one
    # row, so it is cheap to keep at full length even when the forward pass is
    # bounded.
    #
    # `rest=False` drops that requirement, and is NOT a schedule anyone may
    # run: it is the counterfactual "when could this arm have arrived if the
    # pose it stops in cost nothing", which is exactly the seconds an idle
    # policy is trying to buy back (`idle.rest_delays`).
    rest_row = np.ones(M, bool)
    if rest:
        for b, F in free_ab.items():
            rest_row &= F[n - 2, np.clip(prog_hi[b], 0, F.shape[1] - 1)]
    rest_ok = np.logical_and.accumulate(rest_row[::-1])[::-1][:D]
    ok = np.ones((n - 1, D), bool)
    for b, F in free_ab.items():
        ok &= F[:, np.clip(prog_hi[b][:D], 0, F.shape[1] - 1)]
    okf = np.vstack([ok, ok[-1:]])                # index n-1 sits in cell n-2
    reach = np.zeros((n, D), bool)
    reach[0, 0] = True
    for m in range(1, D):
        av = reach[:, m - 1] & okf[:, m - 1]
        col = av.copy()
        col[1:] |= av[:-1]
        reach[:, m] = col
        if reach[n - 1, m] and rest_ok[m]:
            break
    hits = np.flatnonzero(reach[n - 1] & rest_ok)
    if not len(hits):
        return None, None
    m_end = int(hits[0])
    prog = np.full(M, n - 1, int)
    p, m = n - 1, m_end
    while m > 0:
        if p > 0 and reach[p - 1, m - 1] and okf[p - 1, m - 1]:
            p -= 1                                # arrived by moving: pauses go early
        elif reach[p, m - 1] and okf[p, m - 1]:
            pass
        else:                                     # cannot happen if reach is right
            raise RuntimeError("backtrack fell off the reachable set")
        m -= 1
        prog[m] = p
    return prog, m_end


def _schedule(paths, moving, static, cells, prog0, M, dt, verbose=False):
    """One priority order, scheduled. -> (prog, finish) or (None, failing arm)."""
    prog, finish = dict(prog0), {}
    for k, a in enumerate(moving):
        fab = {b: cells(a, b) for b in static + moving[:k]}
        P, m_end = _dp(fab, prog, paths[a].n, M)
        if P is None:
            return None, a
        prog[a] = P
        finish[a] = m_end * dt
    return (prog, finish), None


def _search_priority(paths, moving, static, cells, prog0, M, dt):
    """Every priority order of `moving`, ranked on MAKESPAN. -> (best, stats).

    THE CONDUCTOR'S ONLY FREE VARIABLE IS WHO GOES FIRST, so it should not stop
    at the first order that happens to work.  `n!` orders for n <= 6 is at most
    720, and they share prefixes: arm `a`'s schedule depends on the arms
    scheduled BEFORE it and on nothing else, so a depth-first walk over
    permutation PREFIXES costs at most sum_k P(n, k) DP solves (64 for n = 4,
    1956 for n = 6) instead of n * n! (96 and 4320), and the collision images
    are computed once for the whole search.

    Two prunings, both exact rather than heuristic:

      BOUND    makespan is the max over per-arm finishes, so a prefix's max is
               a LOWER BOUND on every order that extends it.  A prefix already
               worse than the best complete order is abandoned, and the DP for
               each new arm is given that bound as a `deadline` so it stops
               looking as soon as it is beaten.
      TIES     the bound is not strict, so orders that TIE on makespan are all
               explored and ranked on total pause and then on the order itself.
               The winner is therefore a function of the geometry alone, with
               no dependence on the order the permutations were walked in.

    Children are tried busiest-first, which is the old heuristic's guess: it
    costs nothing and finding a good incumbent in the first descent is what
    makes the bound bite for the rest of the search.

    -> (dict(order, prog, finish, makespan, pause) or None, stats).
    """
    n = len(moving)
    nom = {a: (paths[a].n - 1) * dt for a in moving}
    by_motion = sorted(moving, key=lambda a: (-paths[a].motion, a))
    stats = dict(n_dp=0, n_orders=0, n_pruned=0, fail_depth=n + 1, failed=None)
    best = {}

    def note_fail(a, depth):
        if depth < stats["fail_depth"]:
            stats["fail_depth"], stats["failed"] = depth, a

    def rec(order, prog, finish, m_max, pause):
        if len(order) == n:
            stats["n_orders"] += 1
            key = (m_max, round(pause, 9), tuple(order))
            if not best or key < best["key"]:
                best.update(key=key, order=list(order), prog=dict(prog),
                            finish=dict(finish), makespan=m_max * dt,
                            pause=float(pause))
            return
        if best and m_max > best["key"][0]:       # cannot beat what we have
            stats["n_pruned"] += 1
            return
        placed = set(order)
        for a in by_motion:
            if a in placed:
                continue
            cap = best["key"][0] if best else None
            fab = {b: cells(a, b) for b in static + order}
            stats["n_dp"] += 1
            P, m_end = _dp(fab, prog, paths[a].n, M, deadline=cap)
            if P is None:
                if cap is None:       # a real refusal, not "would be too slow"
                    note_fail(a, len(order))
                continue
            prog[a], finish[a] = P, m_end * dt
            rec(order + [a], prog, finish, max(m_max, m_end),
                pause + (m_end * dt - nom[a]))
            del prog[a], finish[a]

    rec([], dict(prog0), {}, 0, 0.0)
    return (best or None), stats


def coordinate(paths, safety=SAFETY_M, calib=CALIB_M, sweep=SWEEP_K,
               horizon_mult=3.0, retry_orders=True, priority_search=True,
               search_max_n=PRIORITY_SEARCH_MAX, verbose=True):
    """Frozen paths -> a merged, pause-scheduled timeline.

    -> dict(progress {arm: (M,) index}, order, moving, parked, pauses, finish,
            duration, free, margin, M, dt, attempts, search).  `free[(a, b)]` is
    the collision image the schedule for `a` was built against, kept so a caller
    (or a test) can check WHICH pairs were considered rather than trusting that
    they were.

    PRIORITY IS A CHOICE, AND IT IS THE ONLY ONE THE CONDUCTOR MAKES.  Every arm
    after the first is conducted around arms whose schedules are already fixed,
    so the order decides both whether a schedule exists at all (an arm late in
    the order can find that the earlier ones are never simultaneously out of its
    way — a deadlock of the ordering, not of the geometry) and how long it
    takes.  With `priority_search` and at most `search_max_n` moving arms every
    permutation is enumerated and the MINIMUM-MAKESPAN one shipped
    (`_search_priority`); the collision images are computed once per pair and
    shared by the whole search, so the enumeration costs DPs and not a
    re-derivation of the geometry.

    Above `search_max_n` moving arms — where n! stops being cheap — the old
    behaviour is the fallback: busiest first, and `retry_orders` promotes the
    arm that deadlocked to the front and tries again, up to once per moving
    arm, because the arm at the front is the one nobody has to avoid.  Either
    path reports infeasibility rather than shipping an unsafe timeline.
    """
    margin = float(safety + calib)
    dt = next(iter(paths.values())).dt
    # A PARKED ARM IS AN OBSTACLE, NOT A NON-PARTICIPANT.  Priority used to be
    # "most motion first", which put every idle arm at the BACK of the queue —
    # and since an arm is only ever checked against the arms scheduled BEFORE
    # it, an arm that draws nothing ended up in nobody's collision image at
    # all.  That is harmless while the idle arms sit at the edge of the rig and
    # wrong the moment one of them is parked in the middle of the sheet, which
    # is exactly what a two-pass run does (an arm that draws only orange stands
    # still through the whole grey phase).  Static arms are therefore scheduled
    # FIRST, at zero cost: their "schedule" is a constant, and every moving arm
    # is then conducted around them.
    static = [a for a in paths if not paths[a].moves]
    moving = sorted([a for a in paths if paths[a].moves],
                    key=lambda a: (-paths[a].motion, a))
    order = static + moving
    nom = {a: (paths[a].n - 1) * dt for a in paths}
    M = int(np.ceil(max(nom.values()) * horizon_mult / dt)) + 64

    free = {}

    def cells(a, b):
        if (a, b) not in free:
            free[(a, b)] = free_cells(paths[a], paths[b], margin, sweep)
        return free[(a, b)]

    prog0 = {a: np.zeros(M, int) for a in static}
    attempts, res, failed = [], None, None
    search, n_attempts = None, 0
    if priority_search and len(moving) <= int(search_max_n):
        heuristic = list(moving)
        t0 = time.time()
        best, stats = _search_priority(paths, moving, static, cells, prog0, M, dt)
        search = dict(n_arms=len(moving), n_permutations=math.factorial(len(moving)),
                      n_dp=stats["n_dp"], n_orders_costed=stats["n_orders"],
                      n_pruned=stats["n_pruned"], wall=float(time.time() - t0),
                      heuristic_order=[int(a) for a in heuristic])
        n_attempts = stats["n_orders"]
        if best is not None:
            moving = best["order"]
            res, failed = (best["prog"], best["finish"]), None
            search.update(order=[int(a) for a in moving],
                          makespan=float(best["makespan"]),
                          pause_total=float(best["pause"]))
            if verbose:
                print(f"  priority search: {stats['n_dp']} DP solves over the "
                      f"{search['n_permutations']} orders of {len(moving)} "
                      f"moving arms ({stats['n_orders']} costed to the end, "
                      f"{stats['n_pruned']} prefixes cut) in {search['wall']:.1f} s"
                      f" -> {best['makespan']:.1f} s with "
                      + "".join(f"{a} " for a in moving).strip()
                      + (" (the busiest-first guess)" if moving == heuristic else
                         f" instead of {' '.join(str(a) for a in heuristic)}"))
        else:
            failed = stats["failed"] or (moving[0] if moving else None)
    else:
        for _ in range(len(moving) + 1 if retry_orders else 1):
            attempts.append(list(moving))
            res, failed = _schedule(paths, moving, static, cells, prog0, M, dt)
            if res is not None:
                break
            if not retry_orders or moving[0] == failed:
                break
            if verbose:
                print(f"  arm {failed} deadlocked at priority "
                      f"{moving.index(failed) + 1}; promoting it to the front "
                      "and re-conducting")
            moving = [failed] + [a for a in moving if a != failed]
        n_attempts = len(attempts)
    if res is None:
        a = failed
        blocked = [b for b in static + moving if b != a and not cells(a, b).any()]
        err = RuntimeError(
            f"arm {a} has no monotone pause schedule inside {M * dt:.0f} s"
            + (f"; its path is never clear of arm{'s' if len(blocked) > 1 else ''} "
               f"{', '.join(str(b) for b in blocked)}, which no amount of "
               "waiting can fix" if blocked else
               f"; {n_attempts} priority order"
               f"{'' if n_attempts == 1 else 's'} were tried")
            + "; v1 does not re-route — try a placement that keeps the arms "
            "further apart")
        # THE REFUSAL IS EVIDENCE, NOT JUST A VERDICT.  The caller can often act
        # on WHERE the arm got stuck (`hard_blocks` below turns these images
        # into "this progress index is impossible whatever anyone else does"),
        # and re-deriving the images to find out would cost as much as the
        # conduct that just failed.
        err.free, err.paths, err.arm = free, paths, a
        err.margin, err.sweep = margin, float(sweep)
        raise err
    prog, finish = res
    for a in static:
        finish[a] = 0.0
    order = static + moving
    if verbose:
        for k, a in enumerate(moving):
            note = ("no pauses" if finish[a] <= nom[a] + 1e-9
                    else f"+{finish[a] - nom[a]:.1f} s of pauses")
            print(f"  arm {a:>2}: priority {k + 1} of {len(moving)} moving "
                  f"({len(static)} parked arms are obstacles for all of them), "
                  f"{paths[a].n} steps, {nom[a]:.1f} s nominal -> {finish[a]:.1f} s "
                  f"scheduled ({note})")
        if search is None and n_attempts > 1:
            print(f"  ({n_attempts} priority orders tried before one worked)")

    pauses = {a: float(finish[a] - nom[a]) for a in paths if paths[a].moves}
    duration = float(max(finish.values()))
    m_last = int(round(duration / dt)) + 1
    return dict(progress={a: prog[a][:m_last] for a in paths}, order=order,
                pauses=pauses, finish=finish, nominal=nom, duration=duration,
                margin=margin, safety=float(safety), calib=float(calib),
                sweep=float(sweep), M=m_last, dt=float(dt), free=free,
                moving=moving, parked=static, attempts=int(n_attempts),
                search=search, pause_total=float(sum(pauses.values())))


def hard_blocks(paths, margin=SAFETY_M + CALIB_M, sweep=SWEEP_K, free=None):
    """Progress indices no schedule can ever reach. -> {(a, b): rows}.

    A row of `a`'s collision image against `b` with NO free cell in it says
    something a pause schedule can never answer: wherever `b` is on its own
    path, `a` may not be at that index.  Waiting does not help, priority does
    not help, and the conductor is right to refuse — but it is not a placement
    failure either, because the index belongs to a PATH somebody chose, and the
    sequencer chooses paths.  A blocked index inside a pen-up transit is an
    edge of a tour, and `allocate.resequence(forbid=...)` can be asked for the
    best tour that does not use it.

    `free` re-uses images already computed (the ones the failed `coordinate`
    attached to its exception); missing pairs are derived here.
    """
    free = {} if free is None else free
    out = {}
    for a, pa in paths.items():
        if not pa.moves:
            continue
        for b in paths:
            if b == a:
                continue
            F = free.get((a, b))
            if F is None:
                F = free[(a, b)] = free_cells(paths[a], paths[b], margin, sweep)
            rows = np.flatnonzero(~F.any(axis=1))
            if len(rows):
                out[(a, b)] = rows
    return out


def rest_delays(paths, res):
    """How much the rest-suffix rule cost each arm, and who made it pay.

    -> {arm: dict(arrive_s, free_arrive_s, delay_s, blockers {arm: seconds})}

    THE POSE AN ARM STOPS IN IS THE ONE THING A PAUSE SCHEDULE CANNOT FIX.  An
    arm held up on its path gets there eventually; an arm whose FINAL pose is
    inside somebody's corridor is refused permission to arrive at all until that
    somebody has gone past, and the DP pays for it in front-loaded waiting.
    Re-running each arm's own DP with the rest requirement dropped (`_dp(...,
    rest=False)`) separates the two exactly: `delay_s` is the seconds that are
    about where the arm STOPS rather than where it goes, and `blockers` names
    the arms whose swept tube the frozen pose sits in during those seconds.

    That number, and not a geometric guess, is what `idle.plan_retreat` triggers
    on — a frozen pose nobody is waiting on is a frozen pose to leave alone.
    Nothing here re-schedules: it re-reads the images the schedule was built
    from, so it cannot change what runs.
    """
    dt, M = float(res["dt"]), int(res["M"])
    prog = {a: np.asarray(p, int) for a, p in res["progress"].items()}
    margin, sweep = float(res["margin"]), float(res["sweep"])
    free = res["free"]

    def cells(a, b):
        if (a, b) not in free:
            free[(a, b)] = free_cells(paths[a], paths[b], margin, sweep)
        return free[(a, b)]

    static, moving = list(res["parked"]), list(res["moving"])
    out = {}
    for k, a in enumerate(moving):
        others = static + moving[:k]
        fab = {b: cells(a, b) for b in others}
        n = paths[a].n
        m_arr = int(round(res["finish"][a] / dt))
        _, m_free = _dp(fab, prog, n, M, rest=False)
        m_free = m_arr if m_free is None else int(m_free)
        blk = {}
        for b, F in fab.items():
            j = np.clip(prog[b][m_free:max(m_arr, m_free)], 0, F.shape[1] - 1)
            bad = int(np.count_nonzero(~F[n - 2, j]))
            if bad:
                blk[b] = bad * dt
        out[a] = dict(arrive_s=float(m_arr * dt), free_arrive_s=float(m_free * dt),
                      delay_s=float(max(0, m_arr - m_free) * dt), blockers=blk)
    return out


def report(res, paths):
    """Terse conductor report -> list of printable lines."""
    out = [f"conductor v1: margin {res['margin']:.3f} m "
           f"(safety {res['safety']:.3f} + calib {res['calib']:.3f}), "
           f"sweep slack {res['sweep']:.2f} x step, dt {res['dt']:.4f} s"]
    out.append(f"{'arm':>5} {'prio':>5} {'steps':>7} {'nominal':>9} "
               f"{'scheduled':>10} {'pause':>8}")
    k = 0
    for a in res["order"]:
        if not paths[a].moves:
            out.append(f"{a:>5} {'-':>5} {paths[a].n:>7} "
                       f"{'parked (draws nothing, still an obstacle)':>41}")
            continue
        k += 1
        out.append(f"{a:>5} {k:>5} {paths[a].n:>7} {res['nominal'][a]:>8.1f}s "
                   f"{res['finish'][a]:>9.1f}s {res['pauses'][a]:>7.1f}s")
    out.append(f"merged timeline {res['duration']:.1f} s, "
               f"{res['pause_total']:.1f} s of pauses inserted in total")
    return out
