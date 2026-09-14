"""A leader's REALISED TRAJECTORY as its exact swept capsules, indexed.

WHAT THIS REPLACES, AND WHY.  `staged.trajectory_room` hands a follower the
leader's trajectory reduced to `cluster_capsules` SPHERES: one sphere per
0.075 m grid cell, its radius the cell half-diagonal plus the biggest capsule
radius that fell in the cell plus the sweep pad.  Measured on 2026-09-14
(docs/V2_STAGED.md section 22.3, `scripts/diag_lf_follower.py`): the median
sphere is 186 mm and the largest 417 mm, and the reduction costs a MEDIAN
195 mm of clearance against the capsules it contains.  On follower arm 31
against same-row leader 71, 0 % of the follower's ink samples clear the 50 mm
gate against the spheres and 55.9 % clear it against the exact capsules.  The
sphere is not a conservative refinement of the leader's trajectory, it is a
different and much fatter obstacle, and it is the single biggest correctable
term between "the follower keeps nothing" and "the follower keeps something".

WHAT THIS IS.  The same capsule chain, per timeline sample, with the same
per-sample 1-Lipschitz sweep pad `scene_check.check_timeline` charges itself
(`SWEEP_FRAC x` the larger of the step into and the step out of the sample) —
so the room still covers the motion BETWEEN samples and the reduction to a
grid of spheres is the only thing given up.  Nothing is decimated at the
default stride: a stride > 1 grows every retained sample's pad by the motion
it now has to span, which is conservative and is asserted by a test.

WHY IT IS STILL FAST.  A stage trajectory is a few thousand samples times nine
capsules, and `frozen.partner_clearance` is linear in the partner's capsule
count — which is what the sphere reduction was bought to avoid.  So the
spheres are kept, as the BROAD PHASE: the same grid cells, each holding its
members and its own bounding sphere, and a query prunes a cell whole whenever
the distance to its sphere already exceeds the best gap found so far.  The
sphere bound that was the answer is now the pruning test, and the answer is
exact.  A cell's sphere CONTAINS its members by construction, so the prune is
sound; the only thing the clustering can now cost is time.

THE INDEPENDENT CHECKER DOES NOT SHARE THIS CODE.  `scene_check` certifies the
final timelines pair-wise on its own sampling with its own residual, and it
stays the judge: if this module were wrong, that check is what would say so.
"""
import numpy as np

from . import coordination

# THE BROAD-PHASE GRID IS NOT THE SPHERE ROOM'S GRID, and it must not be.
# `cluster_capsules` bounds at 0.075 m because the sphere IS the answer there
# and a fat sphere is a wrong answer.  Here the cell is only a PRUNE, so it is
# chosen for cost alone, and the cost has two sides: a fine grid makes the
# broad phase (every observer segment against every cell) dominate, a coarse
# one makes each surviving cell drag hundreds of members into the narrow phase.
# Measured on a 26 000-capsule stage-sized room, floored at the router's 63 mm:
# 0.050 m -> 0.25, 0.075 -> 0.13, 0.100 -> 0.09, 0.150 -> 0.05 ms per pose, and
# the UNFLOORED query runs the other way (0.10 m is 5.0 ms, 0.15 m is 8.2).
# 0.10 m is the knee.  The answer is bit-identical at every setting.
CELL = 0.10                # m, the broad-phase grid
BOX_CELL = 0.30            # m, the grid the go-around's pseudo-boxes are cut on
MAX_BOXES = 12             # ...and how many of them a detour search is offered
CHUNK = 4096               # observer segments measured per broad-phase block
SHELL = 0.05               # m, the first bound shell the narrow phase opens
SHELLS = 6                 # ...and how many times it may double before giving up
PAIR_BLOCK = 4_000_000     # capsule distances evaluated in one vectorised call


def seg_point_dist(p0, p1, X):
    """Distance from segments to points, broadcast. -> array.

    `seg_seg_dist` with a degenerate second segment, written out: the broad
    phase asks this of every (segment, cell centre) pair and it is the single
    hottest expression in the module, so it does not pay the general form's
    clamp-and-re-solve.  Project, clamp, measure.
    """
    d = p1 - p0
    w = X - p0
    dd = np.sum(d * d, -1)
    t = np.clip(np.sum(w * d, -1) / np.maximum(dd, 1e-12), 0.0, 1.0)
    w = w - t[..., None] * d
    return np.sqrt(np.maximum(np.sum(w * w, -1), 0.0))


def _group(mid, cell):
    """Grid-cell membership of a point cloud. -> (inv (M,), n_cells).

    `np.unique(..., axis=0)` on the integer cell key, which is
    `cluster_capsules`' own grouping and has to stay it: the broad-phase
    spheres are only a sound prune if they are the same spheres that bound
    these members.
    """
    key = np.floor(np.asarray(mid, float) / float(cell)).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    return np.asarray(inv).reshape(-1), int(inv.max()) + 1 if len(inv) else 0


class ExactRoom:
    """A capsule cloud with a grid-hash broad phase. Immutable once built.

    `A`, `B` are (M,3) capsule endpoints in WORLD and `R` (M,) their radii with
    the sweep pad already added, which is the whole of what an obstacle is
    here.  A degenerate capsule (A == B) is a sphere, so this shape also holds
    everything `cluster_capsules` used to return — which is what lets the flag
    be one flag.
    """

    __slots__ = ("A", "B", "R", "cell", "cen", "rad", "lo", "hi", "rmax",
                 "start", "count", "members", "digest", "kind")

    def __init__(self, A, B, R, cell=CELL, digest="", kind="capsules"):
        self.A = np.ascontiguousarray(np.asarray(A, float).reshape(-1, 3))
        self.B = np.ascontiguousarray(np.asarray(B, float).reshape(-1, 3))
        self.R = np.ascontiguousarray(np.asarray(R, float).reshape(-1))
        self.cell = float(cell)
        self.digest = str(digest)
        self.kind = str(kind)
        m = len(self.A)
        if not m:
            self.cen = np.zeros((0, 3))
            self.rad = np.zeros(0)
            self.lo = np.zeros((0, 3))
            self.hi = np.zeros((0, 3))
            self.rmax = np.zeros(0)
            self.start = np.zeros(0, np.int64)
            self.count = np.zeros(0, np.int64)
            self.members = np.zeros(0, np.int64)
            return
        inv, n = _group(0.5 * (self.A + self.B), self.cell)
        # the cell's own bounding sphere: the tight AABB centre of its members'
        # ENDPOINTS, the largest endpoint offset from it, plus the largest
        # member radius.  Identical to `staged.cluster_capsules`, deliberately.
        P = np.vstack([self.A, self.B])
        g = np.concatenate([inv, inv])
        lo = np.full((n, 3), np.inf)
        hi = np.full((n, 3), -np.inf)
        np.minimum.at(lo, g, P)
        np.maximum.at(hi, g, P)
        cen = 0.5 * (lo + hi)
        rad = np.zeros(n)
        np.maximum.at(rad, g, np.linalg.norm(P - cen[g], axis=1))
        rmax = np.zeros(n)
        np.maximum.at(rmax, g, np.concatenate([self.R, self.R]))
        self.cen = np.ascontiguousarray(cen)
        self.rad = np.ascontiguousarray(rad + rmax)
        # ...AND THE CELL'S AABB, WHICH IS THE BOUND THAT ACTUALLY PRUNES.  The
        # bounding sphere costs a 65 mm half-diagonal it does not need: an AABB
        # is CONVEX, so a capsule whose two endpoints are both in the cell's
        # endpoint AABB has its whole axis in there too, and the box is exact
        # where the sphere is slack.  Measured on a stage-sized room: swapping
        # the prune from the sphere to the max of the two takes an unfloored
        # query from 7 ms per pose to well under 1.  Both bounds are valid, so
        # their MAX is valid, and the narrow phase still decides.
        self.lo = np.ascontiguousarray(lo)
        self.hi = np.ascontiguousarray(hi)
        self.rmax = np.ascontiguousarray(rmax)
        # CSR membership, so a cell's members are one contiguous slice
        srt = np.argsort(inv, kind="stable")
        self.members = np.ascontiguousarray(srt.astype(np.int64))
        cnt = np.bincount(inv, minlength=n).astype(np.int64)
        self.count = cnt
        self.start = np.concatenate([[0], np.cumsum(cnt)[:-1]]).astype(np.int64)

    # -- shape ------------------------------------------------------------
    def __len__(self):
        return len(self.A)

    @property
    def n_cells(self):
        return len(self.cen)

    @property
    def spheres(self):
        """The broad phase, as the sphere room it used to be. -> (cen, rad)."""
        return self.cen, self.rad

    def signature(self):
        """What a cache has to key on to tell two rooms apart. -> tuple."""
        return (self.kind, int(len(self.A)),
                round(float(np.sum(self.A) + np.sum(self.B)), 6),
                round(float(np.sum(self.R)), 6))

    # -- the query --------------------------------------------------------
    def _pairs(self, p0, p1, ro, ti, ki, best):
        """Measure a sparse set of (segment, cell) pairs exactly, in one call.

        THE PYTHON LOOP OVER CELLS WAS THE COST, not the geometry.  A stage
        trajectory lands in a few thousand grid cells and a loop that visits
        each one to apply the prune costs more than the distances it saves
        (measured: 2.29 ms per pose).  So the prune is evaluated as a MASK and
        the survivors are expanded — cell -> its members — into one flat pair
        list, measured by a single vectorised `seg_seg_dist`, and reduced back
        per segment.  Same answers, no loop.
        """
        if not len(ti):
            return best
        cnt = self.count[ki]
        cum = np.cumsum(cnt)
        tot = int(cum[-1])
        if not tot:
            return best
        # split the pair list so no single vectorised call exceeds PAIR_BLOCK
        # capsule distances, whatever the cells happen to hold
        edges = [0]
        while edges[-1] < len(ti):
            nxt = int(np.searchsorted(cum, cum[edges[-1] - 1] if edges[-1] else 0,
                                      "left"))
            nxt = int(np.searchsorted(
                cum, (cum[edges[-1] - 1] if edges[-1] else 0) + PAIR_BLOCK,
                "right"))
            edges.append(max(nxt, edges[-1] + 1))
        for lo, hi in zip(edges[:-1], edges[1:]):
            tb, kb = ti[lo:hi], ki[lo:hi]
            c = cnt[lo:hi]
            n = int(c.sum())
            if not n:
                continue
            base = np.cumsum(c) - c
            seq = np.arange(n) - np.repeat(base, c)
            mem = self.members[np.repeat(self.start[kb], c) + seq]
            tt = np.repeat(tb, c)
            d = coordination.seg_seg_dist(p0[tt], p1[tt], self.A[mem],
                                          self.B[mem])
            np.minimum.at(best, tt, d - self.R[mem] - ro[tt])
        return best

    def segment_clearance(self, s0, s1, r_obs, floor=None):
        """Surface gap from observer segments to the room. -> (T,).

        `s0`, `s1` are (T,3) and `r_obs` (T,) the observer capsule radii.  With
        `floor` the answer is only guaranteed to be on the right side of it —
        `paper.leg_static_lb`'s own contract, and what lets a cell whose sphere
        is already further than the floor be skipped without being measured.
        """
        s0 = np.asarray(s0, float).reshape(-1, 3)
        s1 = np.asarray(s1, float).reshape(-1, 3)
        r_obs = np.asarray(r_obs, float).reshape(-1)
        T = len(s0)
        out = np.full(T, np.inf)
        if not len(self.A) or not T:
            return out
        cut = np.inf if floor is None else float(floor)
        K = len(self.cen)
        for lo in range(0, T, CHUNK):
            hi = min(T, lo + CHUNK)
            p0, p1, ro = s0[lo:hi], s1[lo:hi], r_obs[lo:hi]
            t = hi - lo
            # BROAD PHASE: the segment against every cell's bounding sphere,
            # which is a lower bound on its gap to any member of that cell,
            # because the sphere CONTAINS them (`cluster_capsules`' own bound).
            d = seg_point_dist(p0[:, None, :], p1[:, None, :], self.cen[None])
            lb = d - self.rad[None, :]
            # ...and the AABB bound, which is the tight one: the gap between
            # the segment's own AABB and the cell's, axis by axis.
            s_lo = np.minimum(p0, p1)
            s_hi = np.maximum(p0, p1)
            gap = np.maximum(np.maximum(self.lo[None] - s_hi[:, None, :],
                                        s_lo[:, None, :] - self.hi[None]), 0.0)
            np.maximum(lb, np.sqrt(np.einsum("tkc,tkc->tk", gap, gap))
                       - self.rmax[None, :], out=lb)
            lb -= ro[:, None]
            # ITERATIVE DEEPENING, AND IT IS WHAT MAKES THIS EXACT AND FAST.
            # Measuring every cell whose bound is below `inf` measures the whole
            # room; measuring the nearest few and pruning on the result still
            # pays for the nearest few, which on a stage-sized cell is hundreds
            # of capsules per segment.  So the shell is grown instead: measure
            # only the cells whose bound falls in [prev, cut_v), and STOP a row
            # the moment its best gap is below `cut_v` — every cell left has a
            # bound at or above `cut_v`, hence a true gap above the best
            # already found, so the answer cannot improve.  That is an exact
            # termination test, not a heuristic one, and a row typically
            # settles in the first shell.
            best = np.full(t, np.inf)
            lbmin = lb.min(axis=1)
            prev = np.full(t, -np.inf)     # every cell below this is MEASURED
            cut_v = np.minimum(cut, lbmin + SHELL)
            for rnd in range(SHELLS):
                if rnd == SHELLS - 1:
                    cut_v = np.full(t, cut)
                # THE TERMINATION TEST IS AGAINST `prev`, NOT `cut_v`, and the
                # difference is a wrong answer.  What is known after a shell is
                # that every cell with `lb < prev` has been measured; an
                # unmeasured cell therefore holds nothing nearer than `prev`.
                # So a row is finished exactly when `best < prev`.  Testing
                # against the shell about to be opened instead declares a row
                # finished while cells between `prev` and `cut_v` are still
                # unmeasured, and those cells can hold the true minimum — which
                # made the unfloored query read HIGH on about one pose in eight.
                need = (best >= prev) & (lbmin < cut)
                if not need.any():
                    break
                m = need[:, None] & (lb >= prev[:, None]) & (lb < cut_v[:, None])
                ti, ki = np.nonzero(m)
                best = self._pairs(p0, p1, ro, ti, ki, best)
                prev = cut_v
                cut_v = np.minimum(cut, cut_v + SHELL * (1 << (rnd + 1)))
            out[lo:hi] = best
        return out

    def chain_clearance(self, P, caps, floor=None):
        """An observer CHAIN against the room. -> (N,).

        `P` is (N, 10|11, 3) world chain points and `caps` the observer's own
        capsule table — exactly `frozen.partner_clearance`'s arguments, so the
        two models are interchangeable at that seam and nowhere else.
        """
        P = np.asarray(P, float)
        if P.ndim == 2:
            P = P[None]
        N = len(P)
        if not len(self.A) or not N or not len(caps):
            return np.full(N, np.inf)
        C = len(caps)
        s0 = np.stack([P[:, i] for i, j, r in caps], axis=1).reshape(-1, 3)
        s1 = np.stack([P[:, j] for i, j, r in caps], axis=1).reshape(-1, 3)
        ro = np.tile(np.array([r for i, j, r in caps], float), N)
        g = self.segment_clearance(s0, s1, ro, floor)
        return g.reshape(N, C).min(axis=1)

    def sphere_clearance(self, C, radii, floor=None):
        """Observer SPHERE centres against the room. -> (N,).

        A sphere is a degenerate segment, which is the whole adaptation.  Only
        reached under the `link_spheres` flag, which ships off.
        """
        C = np.asarray(C, float)
        if C.ndim == 2:
            C = C[None]
        N, S = C.shape[0], C.shape[1]
        if not len(self.A) or not N or not S:
            return np.full(N, np.inf)
        p = C.reshape(-1, 3)
        ro = np.tile(np.asarray(radii, float).reshape(-1), N)
        return self.segment_clearance(p, p, ro, floor).reshape(N, S).min(axis=1)

    # -- what the go-around tier can see ----------------------------------
    def boxes(self, name="room", coarse=BOX_CELL, max_boxes=MAX_BOXES):
        """The room's occupied airspace, as a few AABBs. -> [box dict].

        FOR DETOUR GENERATION ONLY, never for a gate.  `paper._skirt` builds
        its go-around waypoints from the footprints of the boxes in the way,
        and it was blind to the room: with the leader modelled as anything but
        a box the tier had nothing to walk around, so every crossing fell
        through to the RRT tier with no hint about WHERE the obstacle was.
        These are the room's cells merged on a coarser grid, biggest first —
        an honest picture of where the leader is, at the resolution a 0.18 m
        sidestep can act on.  The gate that judges the detour is still the
        exact room, so a box here that is too big or too small costs a detour
        that is tried and refused, never a certificate.
        """
        if not len(self.A):
            return []
        P = np.vstack([self.A, self.B])
        inv, n = _group(P, float(coarse))
        lo = np.full((n, 3), np.inf)
        hi = np.full((n, 3), -np.inf)
        np.minimum.at(lo, inv, P)
        np.maximum.at(hi, inv, P)
        pad = np.zeros(n)
        np.maximum.at(pad, inv, np.concatenate([self.R, self.R]))
        vol = np.prod(np.maximum(hi - lo, 1e-6), axis=1)
        keep = np.argsort(-vol)[:int(max_boxes)]
        return [dict(name=f"{name}:cell{int(k)}",
                     lo=lo[k] - pad[k], hi=hi[k] + pad[k])
                for k in keep]


def union(rooms, digest=""):
    """One room out of several. -> ExactRoom.

    An arm that PRE-POSITIONS before a stage has two trajectories — the
    clear-out from its park to its tuck, and the tour it flies from there — and
    a partner has to avoid both.  Concatenating the capsule blocks is the whole
    of it; the index is rebuilt over the union, which is what keeps the prune
    sound.
    """
    rooms = [r for r in rooms if r is not None and len(r)]
    if not rooms:
        return ExactRoom(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0),
                         digest=digest)
    if len(rooms) == 1 and not digest:
        return rooms[0]
    return ExactRoom(np.vstack([r.A for r in rooms]),
                     np.vstack([r.B for r in rooms]),
                     np.concatenate([r.R for r in rooms]),
                     cell=float(rooms[0].cell),
                     digest=digest or rooms[0].digest,
                     kind=rooms[0].kind)


def from_spheres(cen, rad, digest=""):
    """The SPHERE room, in this shape. -> ExactRoom.

    The A/B arm of the flag: `frozen` then has one partner model instead of
    two, and `ARIS_ROOM=spheres` reproduces the shipped numbers exactly because
    a degenerate capsule measured by `seg_seg_dist` IS a sphere.
    """
    C = np.asarray(cen, float).reshape(-1, 3)
    return ExactRoom(C, C.copy(), np.asarray(rad, float).reshape(-1),
                     digest=digest, kind="spheres")


def sweep_pads(A3, B3, frac):
    """The per-sample 1-Lipschitz residual of a sampled trajectory. -> (N,).

    `scene_check.check_timeline`'s own charge, and `staged.active_pair_gap`
    writes it the same way: a chain point moves at most `step[i]` between
    samples i-1 and i, so sample i has to carry the larger of the step into it
    and the step out of it, times `frac`.  Sample 0 and sample N-1 carry only
    the one step they have.
    """
    A3 = np.asarray(A3, float)
    B3 = np.asarray(B3, float)
    n = len(A3)
    if n < 2:
        return np.zeros(max(n, 0))
    d = np.maximum(np.linalg.norm(np.diff(A3, axis=0), axis=2).max(axis=1),
                   np.linalg.norm(np.diff(B3, axis=0), axis=2).max(axis=1))
    step = np.concatenate([[0.0], d])                 # motion INTO sample i
    nxt = np.concatenate([d, [0.0]])                  # ...and out of it
    return float(frac) * np.maximum(step, nxt)


def from_samples(A3, B3, r, frac, stride=1, digest="", cell=CELL):
    """A sampled capsule chain -> the room it sweeps. -> ExactRoom.

    `A3`, `B3` are (N, K, 3) capsule endpoints per timeline sample and `r` (K,)
    the link radii.  `stride` keeps every `stride`-th sample and grows each
    retained sample's pad by the WHOLE motion it now has to span, which is a
    sum of steps and therefore never smaller than the steps it replaced: the
    decimation is conservative, not merely cheaper.
    """
    A3 = np.asarray(A3, float)
    B3 = np.asarray(B3, float)
    r = np.asarray(r, float).reshape(-1)
    if not len(A3):
        return ExactRoom(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0),
                         cell, digest)
    pad = sweep_pads(A3, B3, frac)
    s = max(1, int(stride))
    if s > 1:
        d = np.maximum(np.linalg.norm(np.diff(A3, axis=0), axis=2).max(axis=1),
                       np.linalg.norm(np.diff(B3, axis=0), axis=2).max(axis=1))
        cum = np.concatenate([[0.0], np.cumsum(d)])
        idx = np.arange(0, len(A3), s)
        if idx[-1] != len(A3) - 1:
            idx = np.concatenate([idx, [len(A3) - 1]])
        lo = np.concatenate([[cum[idx[0]]], cum[idx[:-1]]])
        hi = np.concatenate([cum[idx[1:]], [cum[idx[-1]]]])
        pad = float(frac) * np.maximum(cum[idx] - lo, hi - cum[idx])
        A3, B3 = A3[idx], B3[idx]
    R = (r[None, :] + pad[:, None]).reshape(-1)
    return ExactRoom(A3.reshape(-1, 3), B3.reshape(-1, 3), R, cell, digest)
