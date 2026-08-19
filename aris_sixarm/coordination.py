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

  SCHEDULE  Priority = longest programme first.  The busiest arm runs its
            nominal clock untouched; each next arm gets the earliest-arrival
            monotone schedule that stays in free cells against every arm
            already scheduled.  That is a reachability DP over
            (progress index x time), which is exact for the "advance or wait"
            move set — no local heuristic, no iteration to convergence, and it
            reports infeasibility instead of shipping an unsafe timeline.

  NOT v1    No re-ordering, no re-timing of a stroke, no path change, no
            deadlock recovery beyond priority order, and no dynamics.  A pause
            here is instantaneous in the animation's kinematic playback; a real
            run needs the acceleration-limited version of the same schedule.
"""
import numpy as np

from .frames import PEN_EXT, fk_many
from .fleet import FLEET, H_INV_DEFAULT

LINK_R = 0.09        # m, capsule radius for base/upper arm/forearm
WRIST_R = 0.07       # m, wrist + hand
PEN_R = 0.03         # m, the pen itself
SAFETY_M = 0.05      # m, operating clearance between two arms
CALIB_M = 0.03       # m, unsurveyed base positions (see module docstring)
SWEEP_K = 0.55       # sweep slack factor: 0.5 for the chord, +10 % for the arc
BROAD_CAP = 0.25     # m, clearances above this are not computed exactly

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

    def __init__(self, arm_id, q, dt, h_inv=H_INV_DEFAULT, pen_ext=PEN_EXT):
        spec = FLEET[arm_id]
        self.arm, self.dt = arm_id, float(dt)
        self.q = np.asarray(q, float).reshape(-1, 7)
        if len(self.q) < 2:                       # a parked arm still occupies space
            self.q = np.vstack([self.q, self.q[-1:]])
        P = chain_world(self.q, spec, h_inv, pen_ext)
        self.n = len(P)
        self.A = np.ascontiguousarray(P[:, [c[0] for c in CAPSULES], :], np.float32)
        self.B = np.ascontiguousarray(P[:, [c[1] for c in CAPSULES], :], np.float32)
        self.r = np.array([c[2] for c in CAPSULES], np.float32)
        self.center = P.mean(axis=1).astype(np.float32)
        self.radius = (np.linalg.norm(P - P.mean(axis=1, keepdims=True), axis=2).max(1)
                       + self.r.max()).astype(np.float32)
        self.step = np.linalg.norm(np.diff(P, axis=0), axis=2).max(1).astype(np.float32)
        self.moves = bool(np.max(np.abs(self.q - self.q[0])) > 1e-9)
        self.motion = float(self.step.sum())


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
def _dp(free_ab, prog_hi, n, horizon):
    """Earliest-arrival monotone schedule for one arm. -> (progress (M,), m_end).

    `free_ab` maps each already-scheduled arm to its (n-1, nb-1) free-cell
    image; `prog_hi` maps it to its progress at every time step.  State is
    (progress index, time step); the only moves are "advance one index" and
    "wait", which is exactly what a pause schedule is allowed to do.
    """
    M = horizon
    ok = np.ones((n - 1, M), bool)
    for b, F in free_ab.items():
        ok &= F[:, np.clip(prog_hi[b], 0, F.shape[1] - 1)]
    okf = np.vstack([ok, ok[-1:]])                # index n-1 sits in cell n-2
    reach = np.zeros((n, M), bool)
    reach[0, 0] = True
    for m in range(1, M):
        av = reach[:, m - 1] & okf[:, m - 1]
        col = av.copy()
        col[1:] |= av[:-1]
        reach[:, m] = col
        if reach[n - 1, m]:
            break
    hits = np.flatnonzero(reach[n - 1])
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


def coordinate(paths, safety=SAFETY_M, calib=CALIB_M, sweep=SWEEP_K,
               horizon_mult=3.0, verbose=True):
    """Frozen paths -> a merged, pause-scheduled timeline.

    -> dict(progress {arm: (M,) index}, order, pauses, finish, duration,
            free, margin, M, dt)
    """
    margin = float(safety + calib)
    dt = next(iter(paths.values())).dt
    order = sorted(paths, key=lambda a: (-paths[a].motion, a))
    nom = {a: (paths[a].n - 1) * dt for a in paths}
    M = int(np.ceil(max(nom.values()) * horizon_mult / dt)) + 64

    free, prog, finish = {}, {}, {}
    for k, a in enumerate(order):
        pa = paths[a]
        if not pa.moves:
            prog[a] = np.zeros(M, int)
            finish[a] = 0.0
            continue
        if k == 0:
            prog[a] = np.minimum(np.arange(M), pa.n - 1)
            finish[a] = nom[a]
            if verbose:
                print(f"  arm {a:>2}: priority 1, {pa.n} steps, "
                      f"{nom[a]:.1f} s nominal, no pauses (busiest arm)")
            continue
        fab = {}
        for b in order[:k]:
            if not paths[b].moves and not pa.moves:
                continue
            fab[b] = free_cells(pa, paths[b], margin, sweep)
            free[(a, b)] = fab[b]
        P, m_end = _dp(fab, prog, pa.n, M)
        if P is None:
            raise RuntimeError(
                f"arm {a} has no monotone pause schedule inside {M * dt:.0f} s; "
                "v1 does not re-order or re-route — try a different priority "
                "order, or a placement that keeps the arms further apart")
        prog[a] = P
        finish[a] = m_end * dt
        if verbose:
            print(f"  arm {a:>2}: priority {k + 1}, {pa.n} steps, {nom[a]:.1f} s "
                  f"nominal -> {finish[a]:.1f} s scheduled "
                  f"(+{finish[a] - nom[a]:.1f} s of pauses)")

    pauses = {a: float(finish[a] - nom[a]) for a in paths if paths[a].moves}
    duration = float(max(finish.values()))
    m_last = int(round(duration / dt)) + 1
    return dict(progress={a: prog[a][:m_last] for a in paths}, order=order,
                pauses=pauses, finish=finish, nominal=nom, duration=duration,
                margin=margin, safety=float(safety), calib=float(calib),
                sweep=float(sweep), M=m_last, dt=float(dt),
                pause_total=float(sum(pauses.values())))


def report(res, paths):
    """Terse conductor report -> list of printable lines."""
    out = [f"conductor v1: margin {res['margin']:.3f} m "
           f"(safety {res['safety']:.3f} + calib {res['calib']:.3f}), "
           f"sweep slack {res['sweep']:.2f} x step, dt {res['dt']:.4f} s"]
    out.append(f"{'arm':>5} {'prio':>5} {'steps':>7} {'nominal':>9} "
               f"{'scheduled':>10} {'pause':>8}")
    for k, a in enumerate(res["order"]):
        if not paths[a].moves:
            out.append(f"{a:>5} {'-':>5} {paths[a].n:>7} {'idle (draws nothing)':>29}")
            continue
        out.append(f"{a:>5} {k + 1:>5} {paths[a].n:>7} {res['nominal'][a]:>8.1f}s "
                   f"{res['finish'][a]:>9.1f}s {res['pauses'][a]:>7.1f}s")
    out.append(f"merged timeline {res['duration']:.1f} s, "
               f"{res['pause_total']:.1f} s of pauses inserted in total")
    return out
