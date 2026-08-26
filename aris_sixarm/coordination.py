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
import atexit
import hashlib
import math
import multiprocessing as mp
import os
import time
from collections import OrderedDict

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

# THE CONDUCT IS THE IMAGES, NOT THE SEARCH.  Measured on the Trollface's first
# conduct: the 12 DP solves take 2.1 s and the 15 collision images they read
# take 187 s.  The log line that says "13 DP solves ... in 163.1 s" was always
# reporting the geometry, because `cells()` builds an image the first time the
# search asks for it and the search is what happens to be on the clock.  Four
# things were wrong with those 187 s and the constants below are three of them;
# the fourth, that `free_cells(b, a)` was built from scratch when
# `free_cells(a, b)` was already in hand, needed no constant at all.
CAPSULE_TILE = 32         # samples a side of the block the capsule filter uses
DP_BLOCK = 1024           # time steps a schedule DP gathers at a time
IMAGE_JOBS = 0            # 0: decide from the machine.  1: never fork
IMAGE_JOBS_MAX = 32       # ceiling however many cores there are
IMAGE_PAR_MIN = 2_000_000  # cells in a batch before a fork pays for itself
IMAGE_CELLS_PER_JOB = 500_000    # cells one worker should be given
IMAGE_CACHE_BYTES = 1 << 31   # 2 GiB of images kept between conducts

_IMAGES = OrderedDict()   # (key_i, key_j, margin, sweep) -> free-cell image
_IMAGE_BYTES = 0
_BATCH = None             # the paths an image worker reads, inherited by fork
_IMAGE_STATS = dict(built=0, cached=0, transposed=0, cells=0, wall=0.0, jobs=0)

# (chain point i, chain point j, radius); indices into the 10-point chain
# (frames.fk's 9 points + the pen tip).  Points 1/2 and 5/6 coincide by
# construction (zero DH offset), so they are not given their own capsule.
CAPSULES = ((0, 1, LINK_R), (1, 3, LINK_R), (3, 4, LINK_R), (4, 5, LINK_R),
            (5, 7, WRIST_R), (7, 8, WRIST_R), (8, 9, PEN_R))
# LATERAL HOLDER: an 11-point chain (bracket corner at index 10) and a
# TWO-capsule tool, bracket TCP->corner + pen corner->tip, both at the
# holder envelope radius.  Mirrors rig_final.STATIC_CAPSULES_LAT and
# scene_check.RADII_LAT (a test pins the three together).
CAPSULES_LAT = CAPSULES[:-1] + ((8, 10, PEN_R_FINAL), (10, 9, PEN_R_FINAL))


# ==========================================================================
# geometry
# ==========================================================================
def chain_world(qs, spec, h_inv=H_INV_DEFAULT, pen_ext=PEN_EXT):
    """(N,7) joints -> (N,10|11,3) chain points in WORLD, tool included.

    10 points for the inline pen; with the lateral holder ACTIVE
    (frames.PEN_LAT != 0) the bracket corner joins as point 10 and the
    capsule tables select on the width.
    """
    qs = np.asarray(qs, float).reshape(-1, 7)
    T, P = fk_many(qs)
    from .frames import tool_points_many
    tool = tool_points_many(T, pen_ext)
    P = np.concatenate([P] + [t[:, None, :] for t in tool], axis=1)
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


def _box_of(p, r0, r1):
    """Each capsule's axis-aligned box over samples [r0, r1), grown by r."""
    A, B = p.A[r0:r1], p.B[r0:r1]
    return (np.minimum(A.min(0), B.min(0)) - p.r[:, None],
            np.maximum(A.max(0), B.max(0)) + p.r[:, None])


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
        caps = CAPSULES_LAT if P.shape[1] >= 11 else CAPSULES
        self.A = np.ascontiguousarray(P[:, [c[0] for c in caps], :], np.float32)
        self.B = np.ascontiguousarray(P[:, [c[1] for c in caps], :], np.float32)
        self.r = np.array([c[2] for c in caps], np.float32)
        if getattr(spec, "rig", "sixarm") == "final" and caps is CAPSULES:
            # the pen capsule carries the HOLDER envelope union (both CAD
            # builds, clutch extended): r 0.05, not the bare-pen 0.03
            # (the lateral table already carries the envelope radius)
            self.r = self.r.copy()
            self.r[-1] = PEN_R_FINAL
        self.center = P.mean(axis=1).astype(np.float32)
        self.radius = (np.linalg.norm(P - P.mean(axis=1, keepdims=True), axis=2).max(1)
                       + self.r.max()).astype(np.float32)
        self.step = np.linalg.norm(np.diff(P, axis=0), axis=2).max(1).astype(np.float32)
        self.moves = bool(np.max(np.abs(self.q - self.q[0])) > 1e-9)
        self.motion = float(self.step.sum())
        # PER-CAPSULE BOXES, because the per-SAMPLE one never rejects anything.
        # `clearance_matrix`'s broad phase tests one bounding sphere per sample
        # covering the whole chain — and a Franka is a metre long standing a
        # metre from its neighbour, so those spheres always overlap: on the
        # Trollface 100.0 % of the 19.3 M sample pairs its three moving arms
        # make survived it and paid all 49 exact capsule distances.  A box per
        # CAPSULE over the whole path is a different question with a useful
        # answer (the base column is 0.85 m from anything arm 2 ever does),
        # and it is exact: a capsule pair whose
        # boxes are `cap` apart cannot contribute a value below `cap`, and a
        # value at or above `cap` is clipped to `cap` either way.
        self._box = self._key = None
        self._tiles = {}

    def boxes(self, r0=0, r1=None):
        """Each capsule's swept box over samples [r0, r1). -> (lo, hi) (7,3)."""
        r1 = self.n if r1 is None else int(r1)
        if r0 == 0 and r1 == self.n:
            if getattr(self, "_box", None) is None:
                self._box = _box_of(self, 0, self.n)
            return self._box
        return _box_of(self, r0, r1)

    def tile_boxes(self, tile):
        """`boxes` over every block of `tile` samples. -> (lo, hi) (T,7,3).

        Memoised on the path, because one image is built in row blocks on
        several processes and every one of them tiles the same columns.
        """
        if getattr(self, "_tiles", None) is None:
            self._tiles = {}
        got = self._tiles.get(tile)
        if got is None:
            lo, hi = [], []
            for s in range(0, self.n, tile):
                b = _box_of(self, s, min(s + tile, self.n))
                lo.append(b[0])
                hi.append(b[1])
            got = self._tiles[tile] = (np.array(lo), np.array(hi))
        return got

    @property
    def key(self):
        """A hash of what an image depends on. -> str.

        TWO PATHS WITH THIS KEY HAVE THE SAME IMAGE, whoever built them.  The
        conductor is run three or four times over one fleet (baseline, JIT,
        retreat) and each pass re-programmes ONE or TWO arms — `idle._programs`
        hands the others back the identical programme object — but every pass
        re-samples all six and rebuilds every image from scratch.  Keying on the
        world-frame capsule endpoints and the per-sample step, which is exactly
        what `free_cells` reads, makes the unchanged pairs free.  Keying on
        `id()` would not: `_capsules` builds new objects every pass.
        """
        if getattr(self, "_key", None) is None:
            h = hashlib.blake2b(digest_size=16)
            h.update(np.asarray([self.arm, self.n], np.int64).tobytes())
            for arr in (self.A, self.B, self.r, self.step):
                h.update(np.ascontiguousarray(arr).tobytes())
            self._key = h.hexdigest()
        return self._key


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


def live_capsules(pi, pj, cap=BROAD_CAP):
    """Which of the 49 capsule pairs can EVER come within `cap`. -> (P, Q).

    Box-to-box distance is a lower bound on capsule-to-capsule distance — a
    capsule lies inside the box of its two endpoints grown by its radius — so a
    pair this rejects contributes nothing a `cap`-clipped matrix would record.
    It costs 49 box tests for a whole matrix and on the Trollface's three real
    pairs it retires 14 %, 51 % and 45 % of the exact work: the base column
    never approaches anything, and neither does the upper arm of an arm working
    the far side of its sheet.
    """
    return _live(pi.boxes(), pj.boxes(), float(cap) ** 2)


def _live(bi, bj, cap2):
    """The capsule pairs two box sets can bring within sqrt(cap2). -> (P, Q)."""
    (lo_i, hi_i), (lo_j, hi_j) = bi, bj
    sep = np.maximum(np.maximum(lo_i[:, None, :] - hi_j[None, :, :],
                                lo_j[None, :, :] - hi_i[:, None, :]), 0.0)
    return np.nonzero(np.einsum("pqk,pqk->pq", sep, sep) < cap2)


def clearance_matrix(pi, pj, cap=BROAD_CAP, chunk=15000, rows=None, tile=None):
    """(Ni, Nj) capsule-to-capsule clearance, CLIPPED at `cap`.

    The clip is the whole point: a broad phase throws away every pair that
    cannot possibly be closer than `cap`, and only the survivors pay for the
    exact segment distances.  Anything reported as `cap` is "at least cap",
    which is all a margin test needs to know.

    There are two broad phases and they reject different things.  The
    per-CAPSULE one (`live_capsules`) is a property of the two whole paths and
    decides which of the 49 capsule pairs are worth carrying at all; the
    per-SAMPLE one below is a property of two instants.  On arms that share a
    sheet the second rejects nothing and the first does the work.

    `rows` is a half-open `(r0, r1)` range of ROWS of the matrix, so one image
    can be built in blocks on several processes; the returned array is those
    rows only.  `None` is the whole matrix, to the index.
    """
    Ni, Nj = pi.n, pj.n
    tile = int(CAPSULE_TILE if tile is None else tile)
    r0, r1 = (0, Ni) if rows is None else (int(rows[0]), int(rows[1]))
    D = np.full((r1 - r0, Nj), cap, np.float32)
    cap2 = float(cap) ** 2
    if not len(_live(pi.boxes(), pj.boxes(), cap2)[0]):
        return D                       # these two never come near each other
    work = max(1, int(chunk) * len(CAPSULES) ** 2)   # distances per numpy call
    col = pj.tile_boxes(int(tile))
    for a0 in range(r0, r1, tile):
        a1 = min(a0 + tile, r1)
        row = pi.boxes(a0, a1)
        for t, b0 in enumerate(range(0, Nj, tile)):
            b1 = min(b0 + tile, Nj)
            # WHICH CAPSULES CAN MEET, HERE.  Over the whole path an arm's
            # forearm gets everywhere; over `tile` consecutive samples of it it
            # does not, and the tile-local boxes retire most of the capsule
            # pairs the whole-path boxes have to keep — on the Trollface's
            # biggest pair, 42 of 49 globally against 14 of 49 per tile.
            # Rejection is exact either way, so the matrix is the same matrix
            # however it was tiled; only the arithmetic skipped changes.
            P, Q = _live(row, (col[0][t], col[1][t]), cap2)
            if not len(P):
                continue
            d = pi.center[a0:a1, None, :] - pj.center[None, b0:b1, :]
            gap = (np.sqrt(np.einsum("ijk,ijk->ij", d, d))
                   - pi.radius[a0:a1, None] - pj.radius[None, b0:b1])
            ia, ib = np.nonzero(gap < cap)
            if not len(ia):
                continue
            ia, ib = ia + a0, ib + b0
            rr = (pi.r[P] + pj.r[Q]).astype(np.float32)
            step = max(1, work // len(P))
            for k0 in range(0, len(ia), step):
                u, v = ia[k0:k0 + step], ib[k0:k0 + step]
                d = seg_seg_dist(pi.A[u[:, None], P[None, :]],
                                 pi.B[u[:, None], P[None, :]],
                                 pj.A[v[:, None], Q[None, :]],
                                 pj.B[v[:, None], Q[None, :]]) - rr
                D[u - r0, v] = np.minimum(d.min(1), cap)
    return D


def free_cells(pi, pj, margin, sweep=SWEEP_K, cap=BROAD_CAP, rows=None):
    """(Ni-1, Nj-1) boolean: is the whole cell clear by `margin`?

    Clearance is 1-Lipschitz in the displacement of the bodies, so the cell's
    worst case is bounded by its best corner minus the two half-step motions.
    `sweep` > 0.5 pays for the arc-vs-chord difference of a rotating link.

    `rows` is a half-open range of CELL rows, and reads one clearance row more
    than it returns because a cell is bounded by its four corners.
    """
    r0, r1 = (0, pi.n - 1) if rows is None else (int(rows[0]), int(rows[1]))
    D = clearance_matrix(pi, pj, cap, rows=(r0, r1 + 1))
    S = np.minimum(np.minimum(D[:-1, :-1], D[1:, :-1]),
                   np.minimum(D[:-1, 1:], D[1:, 1:]))
    S -= sweep * (pi.step[r0:r1, None] + pj.step[None, :])
    return S >= margin


# ==========================================================================
# the image bag: once per unordered pair, on every core, and kept
# ==========================================================================
def clear_images():
    """Drop the collision-image memo.  -> None."""
    global _IMAGE_BYTES
    _IMAGES.clear()
    _IMAGE_BYTES = 0


def _remember(k, F):
    """File one image under its content key, oldest out first. -> the image."""
    global _IMAGE_BYTES
    F.flags.writeable = False       # it is handed to every caller that asks
    _IMAGES[k] = F
    _IMAGE_BYTES += F.nbytes
    while _IMAGE_BYTES > IMAGE_CACHE_BYTES and len(_IMAGES) > 1:
        _, old = _IMAGES.popitem(last=False)
        _IMAGE_BYTES -= old.nbytes
    return F


def image_jobs(n_cells):
    """Processes to build `n_cells` of image on. -> int (1 = do it here).

    SCALED TO THE BATCH, not to the machine.  The kernel is memory-bound —
    measured on the Trollface, the build stops getting faster somewhere around
    eight to twelve processes and forking more only costs the fork — so the
    rule is a process per `IMAGE_CELLS_PER_JOB` cells, floored at one and
    capped at what `IMAGE_JOBS` (or the machine) allows.
    """
    if int(IMAGE_JOBS) == 1 or mp.current_process().daemon:
        return 1                  # a pool worker may not have children
    if int(n_cells) < int(IMAGE_PAR_MIN):
        return 1
    cap = int(IMAGE_JOBS) if int(IMAGE_JOBS) > 0 else (os.cpu_count() or 1)
    return max(1, min(int(cap), IMAGE_JOBS_MAX,
                      int(n_cells) // int(IMAGE_CELLS_PER_JOB) or 1))


def _image_rows(task):
    """One block of one image, in a forked worker. -> (a, b, r0, r1, rows)."""
    a, b, r0, r1, margin, sweep = task
    return a, b, r0, r1, free_cells(_BATCH[a], _BATCH[b], margin, sweep,
                                    rows=(r0, r1))


def _build(paths, todo, key_of, margin, sweep, jobs):
    """Build the images `todo` names, in row blocks over a fork pool. -> None.

    ONE POOL FOR THE WHOLE BATCH, AND THE BATCH IS ROW BLOCKS AND NOT IMAGES.
    A conduct wants three big images at once and they are not the same size —
    on the Trollface 13.9 s, 4.8 s and 4.0 s — so a task per image leaves most
    of the machine idle for the length of the slowest one.  Row blocks of one
    image are independent (a cell reads two clearance rows and nothing else),
    so the batch is sliced to about four blocks per worker and the wall clock
    becomes the total over the cores rather than the largest image.

    The pool is FORKED with the paths already in memory: an `ArmPath` is a
    megabyte of float32 and there are six of them, and copy-on-write ships them
    for nothing where `pool.map` would pickle them once per task.
    """
    global _BATCH
    cells = sum((paths[a].n - 1) * (paths[b].n - 1) for a, b in todo)
    njobs = image_jobs(cells) if jobs is None else max(1, int(jobs))
    aim = max(1, cells // (njobs * 4))      # cells a block should carry
    tasks = []
    for a, b in todo:
        ni, nj = paths[a].n - 1, paths[b].n - 1
        blk = max(1, min(ni, aim // max(nj, 1)))
        for r0 in range(0, ni, blk):
            tasks.append((a, b, r0, min(r0 + blk, ni),
                          float(margin), float(sweep)))
    part = {c: np.empty((paths[c[0]].n - 1, paths[c[1]].n - 1), bool)
            for c in todo}
    t0 = time.time()
    if njobs > 1 and len(tasks) > 1:
        _BATCH = paths
        try:
            with mp.get_context("fork").Pool(njobs) as pool:
                for a, b, r0, r1, F in pool.imap_unordered(_image_rows, tasks,
                                                           chunksize=1):
                    part[(a, b)][r0:r1] = F
        finally:
            _BATCH = None
    else:
        for a, b, r0, r1, m, s in tasks:
            part[(a, b)][r0:r1] = free_cells(paths[a], paths[b], m, s,
                                             rows=(r0, r1))
    for c, F in part.items():
        _remember(key_of[c], F)
    _IMAGE_STATS.update(built=_IMAGE_STATS["built"] + len(todo),
                        cells=_IMAGE_STATS["cells"] + cells,
                        wall=_IMAGE_STATS["wall"] + (time.time() - t0),
                        jobs=njobs)


def build_images(paths, pairs, margin, sweep, jobs=None):
    """Every image `pairs` asks for. -> {(a, b): (na-1, nb-1) bool}.

    THREE THINGS THE OLD `cells()` DID NOT DO, and between them they are most
    of a conduct.

      ONCE PER UNORDERED PAIR.  `free_cells(b, a)` is `free_cells(a, b)`
      transposed — the four-corner minimum and the two sweep terms are both
      symmetric — and the priority search asks for BOTH directions of every
      moving pair, because an order that schedules `a` before `b` and one that
      schedules `b` before `a` are both in the enumeration.  The old code built
      each from scratch: on the Trollface, 46.0 s for (31, 2) and then 45.8 s
      for (2, 31).  The image is now built for the pair sorted by arm id and
      read the other way round, which also makes it a function of the UNORDERED
      pair — `seg_seg_dist` clamps one parameter before the other, so the two
      directions could disagree in the last bit of a float32, and now they
      cannot.

      KEPT BETWEEN CONDUCTS.  Keyed on `ArmPath.key`, so the JIT and retreat
      passes — which re-programme one or two arms and leave the rest alone —
      pay only for the pairs that actually changed.

      BUILT ON EVERY CORE.  See `_build`.

    The pairs are gathered UP FRONT rather than on first use.  Every one of them
    is needed by any complete schedule (arm k is conducted against the k-1 arms
    before it and every parked arm, and the union over k is exactly this set),
    so nothing is built on speculation — and asking for them together is what
    makes one dispatch out of what was six serial waits.
    """
    canon, key_of = {}, {}
    for a, b in pairs:
        c = (a, b) if a <= b else (b, a)
        canon[(a, b)] = c
        key_of[c] = (paths[c[0]].key, paths[c[1]].key,
                     float(margin), float(sweep))
    todo = [c for c in dict.fromkeys(canon.values()) if key_of[c] not in _IMAGES]
    hit = len(key_of) - len(todo)
    if todo:
        _build(paths, todo, key_of, margin, sweep, jobs)
    _IMAGE_STATS["cached"] += hit
    out = {}
    for ab, c in canon.items():
        F = _IMAGES[key_of[c]]
        _IMAGES.move_to_end(key_of[c])
        if ab == c:
            out[ab] = F
        else:
            out[ab] = np.ascontiguousarray(F.T)
            _IMAGE_STATS["transposed"] += 1
    return out


def _at_exit():                                            # pragma: no cover
    clear_images()


atexit.register(_at_exit)


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
    # THE COLUMNS AFTER THE ARM ARRIVES ARE NEVER READ, so they are never built.
    # `ok` is one gather per already-scheduled arm over (n-1) x D booleans — on
    # the Trollface 29.5 MB apiece, and the measured 184 ms of a 253 ms solve —
    # while the forward pass below breaks the moment the arm can stop, which on
    # that same solve was step 3525 of a horizon of 10639.  Building it in
    # blocks makes the gather cost what the schedule actually uses.  The first
    # block is the arm's own length because it cannot possibly arrive sooner:
    # arriving means advancing n-1 times, one index per step.
    okf = np.empty((n, D), bool)          # index n-1 sits in cell n-2
    reach = np.zeros((n, D), bool)        # both are lazily paged, not touched
    reach[0, 0] = True
    have = 0

    def extend(upto):                     # -> columns [0, upto) of okf are real
        nonlocal have
        hi = min(D, max(int(upto), have + max(n, DP_BLOCK)))
        col = okf[:, have:hi]
        col[:] = True
        for b, F in free_ab.items():
            G = F[:, np.clip(prog_hi[b][have:hi], 0, F.shape[1] - 1)]
            col[:n - 1] &= G
            col[n - 1] &= G[n - 2]
        have = hi

    for m in range(1, D):
        if have < m:
            extend(m)
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
               search_max_n=PRIORITY_SEARCH_MAX, jobs=None, verbose=True):
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

    # EVERY IMAGE ANY ORDER COULD ASK FOR, IN ONE DISPATCH.  The set is not a
    # guess: arm k is conducted against the arms before it and every parked
    # arm, so the union over every order is exactly `moving x everyone else`,
    # and the search used to discover that one image at a time — six serial
    # waits, two of which were the transpose of two others.
    t_img = time.time()
    n_img0, n_hit0 = _IMAGE_STATS["built"], _IMAGE_STATS["cached"]
    free = build_images(paths, [(a, b) for a in moving for b in paths if b != a],
                        margin, sweep, jobs=jobs)
    t_img = time.time() - t_img
    images = dict(wall=float(t_img), built=_IMAGE_STATS["built"] - n_img0,
                  cached=_IMAGE_STATS["cached"] - n_hit0,
                  jobs=int(_IMAGE_STATS["jobs"]))
    if verbose and free:
        print(f"  collision images: {images['built']} built on "
              f"{images['jobs']} process(es), {images['cached']} already in "
              f"hand, {len(free)} read (both ways round) in {t_img:.1f} s")

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
        # WHAT THE SEARCH DID, NOT WHAT IT FINISHED.  `n_attempts` is
        # `stats["n_orders"]` — COMPLETE orders costed — which is zero by
        # construction whenever the search fails, so this used to report "0
        # priority orders were tried" for a search that had just walked every
        # one of them.  It read as "the DFS cut every prefix", and a refusal
        # whose evidence points at the wrong suspect is worse than one with no
        # evidence at all.  The bound only ever prunes against a complete order
        # already found, so a failed search has pruned NOTHING: every
        # permutation really was explored, and `n_dp` is how many prefixes that
        # took.
        tried = (f"; all {math.factorial(len(moving))} priority orders of the "
                 f"{len(moving)} moving arms were searched "
                 f"({search['n_dp']} prefix DP solves) and none completed"
                 if search is not None else
                 f"; {n_attempts} priority order"
                 f"{'' if n_attempts == 1 else 's'} were tried")
        err = RuntimeError(
            f"arm {a} has no monotone pause schedule inside {M * dt:.0f} s"
            + (f"; its path is never clear of arm{'s' if len(blocked) > 1 else ''} "
               f"{', '.join(str(b) for b in blocked)}, which no amount of "
               "waiting can fix" if blocked else tried)
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
                search=search, images=images,
                pause_total=float(sum(pauses.values())))


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
