"""STAGE ASSIGNMENT, WIRED END TO END: pieces -> plans -> legs -> a stage programme.

Build item 4 of docs/ARCHITECTURE_V2.md.  `traces.py` answers "how few pieces
can this picture be drawn in, and who draws which, in which stage".  This module
takes that answer and turns it into something an arm could actually be handed: a
certified joint trajectory per (stage, arm), the pen-up legs between the pieces,
the order they are drawn in, and the two independent checks a stage has to pass
before anybody believes it.

THERE ARE TWO SAFETY QUESTIONS IN A STAGE AND THEY HAVE TWO DIFFERENT TOOLS.
This is the correction docs/V2_WORKCELLS.md section 5 needs and it is the whole
design of this module:

  BETWEEN TWO ACTIVE ARMS, whose poses are not yet chosen when the stage is
  designed, the criterion is the ENVELOPE one: the two work cells' envelopes --
  the union over every pose either arm could hold anywhere inside its cell --
  must be at least `PAIR_MARGIN` apart.  The zigzag's 0.40 m dead band gives
  +85.8 mm and that is the stage design; it is settled and this module does not
  re-open it.  What this module DOES do is re-measure it from the trajectories
  the planner actually produced, which is a strictly weaker claim than the
  envelope and therefore a check rather than a certificate: `active_pair_gap`.

  BETWEEN AN ACTIVE ARM AND A PARKED PARTNER, the partner is not an envelope at
  all.  It is one known, measured, barrier-verified pose, and the right tool is
  the per-stroke SOLO certification this repo already ships: `frozen.freeze`
  puts the partner's REAL capsules into the static set (aris_sixarm/frozen.py),
  the planner chooses poses and legs that clear 50 mm against them, and
  `scene_check.check_timeline` re-derives the result with no planner state.
  The certified 3-layer map at h = 0.970 with the shipped parks frozen is
  already 100 % of the block at the 50 mm gate, so NO park search, NO cell
  erosion and NO gate change is needed for the parked half of the problem.

  THE "+4.7 mm ENVELOPE VS A PARKED PARTNER" OF V2_WORKCELLS section 5 IS THE
  WRONG TEST FOR A PARKED PARTNER.  It measures an active arm's whole work-cell
  envelope against a parked arm's whole envelope -- a union over poses the
  parked arm will not hold, because it is holding one.  A parked arm is a
  constant, and a constant is certified cell by cell and leg by leg, not as a
  union.  See docs/DECISIONS.md, the 2026-09-11 correction entry.

WHAT IS NOT HERE, ON PURPOSE.  No balancer, no split and no merge: the DP in
`traces.py` already minimised the piece count exactly, and a second optimiser
over the same pieces would be optimising against a cost the first one did not
pay.  No conductor for the main stages either -- three actives in a stage are
static keep-outs for each other, so there is no clock to share (the seam stages
are two-active and `idle.conduct` is still the tool for them; this module does
not invoke it, and says so in its output rather than pretending).

    ARIS_RIG=proposed ARIS_TOOL=lateral python3 -m aris_sixarm.staged --help
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from . import allocate, coordination, frozen, paper, scene_check, stroke_api
from . import traces as traces_mod
from . import writing
from .fleet import FLEET, H_INV_DEFAULT

# The programme this module writes is NOT `program_schema.Bundle`: that schema's
# `_from_dict` refuses unknown AND missing keys, so a stage id and a per-piece
# hover are a `SCHEMA_VERSION` bump rather than an extension, and folding the
# two together is build item 5.  Until then the staged programme carries its own
# version, and it is a superset of what item 5 has to absorb: stage id, barrier
# list, per-piece entry/exit hover configurations and per-piece q_first/q_last.
STAGED_SCHEMA_VERSION = 1

ATLAS_DEFAULT = "out/atlas_proposed_h0970_lat0860_gated63"

PAIR_MARGIN = coordination.PAIR_MARGIN      # 50 mm, the arm-to-arm gate
SWEEP_FRAC = 0.55                           # `scene_check.check_timeline`'s own
CHECK_DT = 0.05                             # s, the clock the checks sample on
MAX_CHECK_POSES = 900                       # per arm, per stage, for check (a)


# ---------------------------------------------------------------------------
# 1.  THE PIECES, OUT OF THE DP
# ---------------------------------------------------------------------------
@dataclass
class Piece:
    """One maximal run of a line, with the (stage, arm) the DP gave it."""
    stage: int
    arm: int
    line: int
    k: int                              # which piece of that line
    pts: np.ndarray                     # (N, 2) paper metres, already lapped
    length_m: float
    state: int = -1                     # the DP state (a (stage, arm) cell)
    s0: float = 0.0                     # ...and its span in the LINE's arc
    s1: float = 0.0                     # length, which is what a mask names

    @property
    def key(self) -> tuple[int, int, int, int]:
        return (self.stage, self.arm, self.line, self.k)


def pieces_of(plan: "traces_mod.Plan") -> list[Piece]:
    """Every drawn piece of every line, in line order. -> [Piece].

    `LinePlan.piece_points` is what the arm is actually told to draw -- the run
    lapped by `OVERDRAW_M` at an overlap seam and abutting exactly at a hard
    edge -- so this is the geometry, not the run.
    """
    out: list[Piece] = []
    for lp in plan.lines:
        for k, p in enumerate(lp.pieces):
            pts = lp.piece_points(k)
            if len(pts) < 2:
                continue
            out.append(Piece(int(p.stage), int(p.arm), int(lp.index), int(k),
                             np.asarray(pts, float),
                             float(traces_mod.cumlen(pts)[-1]),
                             int(p.state), float(p.s0), float(p.s1)))
    return out


def bucket(pieces: Sequence[Piece]) -> dict[tuple[int, int], list[Piece]]:
    """-> {(stage, arm): [Piece]}, the unit this module plans and orders."""
    out: dict[tuple[int, int], list[Piece]] = {}
    for p in pieces:
        out.setdefault((p.stage, p.arm), []).append(p)
    return out


def stage_actives(pattern: "traces_mod.Pattern", stage: int) -> tuple[int, ...]:
    """The arms a stage names.  Everybody else is parked, by definition."""
    return tuple(sorted({int(c.arm) for c in pattern.stage(stage)}))


def same_row_pairs(arms: Iterable[int]) -> list[tuple[int, int]]:
    """Pairs of arms on the same ROW of the base lattice. -> [(a, b)].

    THE TRANSVERSE PAIR IS UNSEPARABLE ON THIS RIG, on either axis
    (docs/V2_WORKCELLS.md section 4b): 0.80 m of x between two certified
    drawing poses is -160.8 mm and 0.81 m of y is -163.1 mm, because both
    elbows stand in the same column about the mid-line whatever the pens do.
    No stage may ever put one in the air together, so this is checked rather
    than assumed.
    """
    a = sorted(int(x) for x in arms)
    return [(i, j) for n, i in enumerate(a) for j in a[n + 1:]
            if traces_mod.ROW_OF.get(i) is not None
            and traces_mod.ROW_OF.get(i) == traces_mod.ROW_OF.get(j)]


# ---------------------------------------------------------------------------
# 2.  THE PARKED PARTNERS, AS REAL CAPSULES
# ---------------------------------------------------------------------------
def shipped_parks(specs=None) -> dict[int, np.ndarray]:
    """{arm: q} of the pose each arm holds when it is not the one drawing.

    `spec.q_seed` IS the shipped park on the proposed rig -- `layout.build_fleet`
    is called with `q_park=Q_PARK_PROPOSED` and stamps it in as `q_ready` -- and
    it is what `writing.arm_program` flies home to, so taking it from the spec
    is what keeps the pose this module freezes and the pose the timeline ends at
    the same pose by construction rather than by agreement.
    """
    fl = FLEET if specs is None else specs
    return {int(a): np.asarray(fl[a].q_seed, float).reshape(7) for a in fl}


def freeze_partners(arm: int, parks: dict[int, np.ndarray], specs=None,
                    pens=None, h_inv=H_INV_DEFAULT, leg_cache=True,
                    leg_cache_root=None) -> tuple[int, ...]:
    """Model every arm but `arm` by its REAL capsules at its park. -> the ids.

    This is the parked half of the safety argument and it is the whole of it:
    the five partners stop being 0.32 m pose-invariant bounding bands and become
    the capsules of the pose they are actually holding, checked at the same
    floors the bands were checked at (`frozen.py`).  Everything downstream that
    reaches `paper.static_boxes` -- the router, the hover solver, the screen --
    then sees the room the arm is really in.

    THE LEG STORE IS RE-NAMESPACED ON THE FROZEN SET, and it has to be.
    `paper.route_key` does not contain the frozen set (the set changes
    `static_boxes` without changing any memo key), so the store refuses to
    answer at all while `frozen` is in a state it was not opened under
    (`paper._dyn_state`).  Opening it under a signature that CONTAINS the frozen
    poses gives each (stage, arm) its own directory, which is both correct --
    arm 13's routes past a parked 17 are not arm 17's routes past a parked 13 --
    and warm across runs, which is the whole point of build item 1.
    """
    fl = FLEET if specs is None else specs
    pens = {a: fl[a].pen for a in fl} if pens is None else pens
    others = {int(a): np.asarray(q, float).reshape(7)
              for a, q in parks.items() if int(a) != int(arm) and a in fl}
    frozen.freeze(others, fl, pens, h_inv)
    frozen.observe(int(arm))
    paper.clear_cache()                 # the memos are not keyed on the set
    if leg_cache:
        paper.disk_cache_open(leg_cache_root,
                              signature=leg_cache_signature(others))
    else:
        paper.disk_cache_close()
    return tuple(sorted(others))


def leg_cache_signature(frozen_poses: dict[int, np.ndarray],
                        envelopes: dict | None = None) -> str:
    """`paper.cache_signature()` plus the frozen room it is valid under.

    The room is the parked partners' poses AND, in a stage, the active
    partners' envelopes — both change `static_boxes` without changing any memo
    key, so both have to be in the store's namespace or a leg bought in one
    room would be served in another.
    """
    blob = json.dumps(dict(
        base=paper.cache_signature(),
        frozen={str(int(a)): [round(float(x), 9)
                              for x in np.asarray(q, float).ravel()]
                for a, q in sorted(frozen_poses.items())},
        envelopes={str(int(a)): [int(len(v[1])),
                                 round(float(np.sum(v[0])), 6),
                                 round(float(np.sum(v[1])), 6)]
                   for a, v in sorted((envelopes or {}).items())}),
        sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def thaw():
    """Back to the shipped pose-invariant partner model."""
    frozen.thaw()
    paper.clear_cache()


# ---------------------------------------------------------------------------
# 2b.  THE OTHER ACTIVE ARMS, AS AN ENVELOPE
# ---------------------------------------------------------------------------
# THE GAP THIS CLOSES.  Measured on 2026-09-11 (docs/V2_STAGED.md): ink against
# ink in a three-active stage clears by +273.5 mm, and the binding number of the
# whole trajectory is +12.9 mm — TWO PEN-UP LEGS flying into each other.  Each
# arm's legs were certified against the PARKED fleet, which is the right room at
# a barrier and the wrong one mid-stage, and nothing asked whether one active
# arm's leg crossed another's.  docs/ARCHITECTURE_V2.md section 2f names both
# the gap and the fix: an envelope is a union over POSES and a leg is a PATH, so
# route every leg against "the other arms' envelopes as static boxes".
#
# WHAT THE ENVELOPE IS HERE.  Not a box — a box round an arm's whole reach
# swallows its neighbour and would refuse everything.  It is the union of the
# partner's LINK CAPSULES over every pose it could hold in this stage: its
# certified drawing pose at each strict-GO cell of its work cell, the hover
# above each of those, and its park.  That is exactly the pose stack
# `scripts/workcell_envelopes.py` measured +85.8 mm with, and `frozen.py` will
# take it unchanged now that `freeze_sets` accepts more than one pose.
ENVELOPE_STRIDE = 1         # every certified cell: a skipped pose needs a pad
ENVELOPE_CLUSTER = 0.075    # m, the grid the capsules are bounded on
ENVELOPE_PAD = 0.0          # m; ZERO is honest only at stride 1 (see below)
ENVELOPE_DIR = "out/stage_envelopes"


def cluster_capsules(A, B, R, cell=ENVELOPE_CLUSTER, pad=0.0):
    """Bound a capsule cloud by grid-local SPHERES. -> (centres (M,3), r (M,)).

    A full row band's envelope is some 9 000 capsules and
    `frozen.partner_clearance` is linear in them, which would put a tenth of a
    second on every `paper.route` call.  Each sphere here CONTAINS every capsule
    whose midpoint fell in its grid cell — both endpoints and the capsule radius
    — so the reduction is conservative by construction: a query that clears the
    spheres clears the capsules.  `pad` is added on top, for the poses the
    lattice stride skipped.
    """
    A = np.asarray(A, float).reshape(-1, 3)
    B = np.asarray(B, float).reshape(-1, 3)
    R = np.asarray(R, float).reshape(-1)
    if not len(A):
        return np.zeros((0, 3)), np.zeros(0)
    key = np.floor(0.5 * (A + B) / float(cell)).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    n = int(inv.max()) + 1
    # VECTORISED, AND IT HAS TO BE.  A stride-1 row band is a few hundred
    # thousand capsule endpoints in tens of thousands of cells, and a
    # `inv == k` scan per cell is O(cells x endpoints) -- minutes per arm, which
    # is what made the first sweep look like it had hung.  `np.minimum.at` and
    # friends do the same grouping in one pass each.
    P = np.vstack([A, B])
    g = np.concatenate([inv, inv])
    lo = np.full((n, 3), np.inf)
    hi = np.full((n, 3), -np.inf)
    np.minimum.at(lo, g, P)
    np.maximum.at(hi, g, P)
    cen = 0.5 * (lo + hi)
    rad = np.zeros(n)
    np.maximum.at(rad, g, np.linalg.norm(P - cen[g], axis=1))
    rmax = np.zeros(n)
    np.maximum.at(rmax, g, np.concatenate([R, R]))
    return cen, rad + rmax + float(pad)


def envelope_poses(arm, region, atlas_dir, spec, h_inv, park,
                   stride=ENVELOPE_STRIDE, verbose=False):
    """Every pose the arm could hold inside `region`. -> Q (N, 7).

    The drawing pose the atlas certified at each strict-GO cell of the region,
    the hover above each of them, and the park.  `writing.lifted_config`'s own
    two-tier search, exactly as `scripts/workcell_envelopes.arm_cells` does it,
    with the drawing pose kept where no hover exists at all — an arm that can
    draw a cell but not lift off it still stands there.
    """
    from . import atlas as atlas_mod
    from . import ik
    arr, meta = atlas_mod.load(Path(atlas_dir), int(arm))
    cur, why = atlas_mod.is_current(meta)
    if not cur:
        raise ValueError(f"atlas for arm {arm} is stale: {why}")
    arr = arr[atlas_mod.strict_go(arr)]
    g = float(meta["grid"]) * max(1, int(stride))
    keep = traces_mod.rect_contains(tuple(region), arr[:, 0], arr[:, 1])
    on = (np.abs(np.rint(arr[:, 0] / g) * g - arr[:, 0]) < 1e-6) & \
         (np.abs(np.rint(arr[:, 1] / g) * g - arr[:, 1]) < 1e-6)
    arr = arr[keep & on]
    qd = np.ascontiguousarray(arr[:, atlas_mod.QCOL:atlas_mod.QCOL + 7])
    out = [qd] if len(qd) else []
    qh = []
    for k in range(len(arr)):
        xy = (arr[k, 0], arr[k, 1])
        q, _ = writing.lifted_config(spec, qd[k], xy, z=writing.LIFT_Z,
                                     h_inv=h_inv, pen_ext=spec.pen)
        if q is None:
            q, _ = writing.lifted_config(spec, qd[k], xy, z=writing.LIFT_Z,
                                         h_inv=h_inv, pen_ext=spec.pen,
                                         phis=writing.HOVER_YAWS,
                                         q7s=ik.Q7_GRID)
        qh.append(qd[k] if q is None else np.asarray(q, float).reshape(7))
    if qh:
        out.append(np.asarray(qh, float))
    out.append(np.asarray(park, float).reshape(1, 7))
    Q = np.vstack(out)
    if verbose:
        print(f"    envelope arm {arm}: {len(arr)} cells at {g:.2f} m "
              f"-> {len(Q)} poses")
    return Q


def _env_sig(**kw) -> str:
    return hashlib.sha256(json.dumps(kw, sort_keys=True).encode()).hexdigest()


def cached_envelope_poses(arm, region, atlas_dir, spec, h_inv, park,
                          stride=ENVELOPE_STRIDE, cache_dir=ENVELOPE_DIR,
                          verbose=False) -> np.ndarray:
    """`envelope_poses`, filed on disk. -> Q (N, 7).

    THE POSES ARE THE EXPENSIVE HALF AND THEY DO NOT DEPEND ON THE CLUSTERING.
    One `writing.lifted_config` per certified cell is about 9 ms and a row band
    has a few hundred to a few thousand of them; the spheres that bound them
    are a hundredth of that.  Keying the two together made a sweep over the
    cluster cell re-solve every hover, which is a cache-key bug and not a cost.
    """
    sig = _env_sig(what="poses", atlas=str(atlas_dir), arm=int(arm),
                   region=[[float(v) for v in r] for r in region],
                   park=[round(float(x), 9)
                         for x in np.asarray(park, float).ravel()],
                   cache=paper.cache_signature(), stride=int(stride),
                   lift=float(writing.LIFT_Z))
    p = Path(cache_dir) / f"q{sig[:24]}.npz"
    if p.exists():
        try:
            return np.asarray(np.load(p)["q"], float)
        except Exception:
            pass
    Q = envelope_poses(arm, region, atlas_dir, spec, h_inv, park, stride,
                       verbose)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, q=Q)
    return Q


def stage_envelope(arm, region, atlas_dir, spec, h_inv, park, pens,
                   stride=ENVELOPE_STRIDE, cluster=ENVELOPE_CLUSTER,
                   pad=ENVELOPE_PAD, cache_dir=ENVELOPE_DIR, verbose=False):
    """One arm's work-cell envelope, as bounding spheres. -> (centres, radii).

    Cached on disk under the atlas, the rig, the tool and the region, because
    it is a property of the STAGE and not of the picture: one build per rig and
    pattern, re-read by every run and every arm that has to avoid it.  The
    POSES are cached separately and more coarsely (`cached_envelope_poses`), so
    a sweep over the clustering costs the clustering and not the IK.
    """
    sig = _env_sig(what="spheres", atlas=str(atlas_dir), arm=int(arm),
                   region=[[float(v) for v in r] for r in region],
                   park=[round(float(x), 9)
                         for x in np.asarray(park, float).ravel()],
                   cache=paper.cache_signature(), stride=int(stride),
                   cluster=float(cluster), pad=float(pad),
                   lift=float(writing.LIFT_Z))
    p = Path(cache_dir) / f"{sig[:24]}.npz"
    if p.exists():
        try:
            z = np.load(p)
            return np.asarray(z["c"], float), np.asarray(z["r"], float)
        except Exception:
            pass
    Q = cached_envelope_poses(arm, region, atlas_dir, spec, h_inv, park,
                              stride, cache_dir, verbose)
    path = coordination.ArmPath(int(arm), Q, 0.01, h_inv,
                                float(pens.get(arm, spec.pen)), spec)
    keep = [k for k in range(len(path.r))
            if k not in coordination.FROZEN_SWEEP_BANDS]
    A = np.asarray(path.A, float)[:, keep].reshape(-1, 3)
    B = np.asarray(path.B, float)[:, keep].reshape(-1, 3)
    R = np.tile(np.asarray(path.r, float)[keep], len(Q))
    c, r = cluster_capsules(A, B, R, cluster, pad)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, c=c, r=r)
    if verbose:
        print(f"    envelope arm {arm}: {len(A)} capsules -> {len(c)} spheres")
    return c, r


def stage_envelopes(pattern, stage, atlas_dir, specs=None, pens=None,
                    parks=None, h_inv=H_INV_DEFAULT, **kw) -> dict:
    """{arm: (centres, radii)} for every ACTIVE arm of a stage."""
    fl = FLEET if specs is None else specs
    pens = {a: fl[a].pen for a in fl} if pens is None else pens
    parks = shipped_parks(fl) if parks is None else parks
    regions = allocate.work_cell_regions(pattern, stage) or {}
    return {int(a): stage_envelope(a, regions[a], atlas_dir, fl[a], h_inv,
                                   parks[a], pens, **kw)
            for a in sorted(regions)}


def freeze_stage(arm, parks, envelopes=None, specs=None, pens=None,
                 h_inv=H_INV_DEFAULT, leg_cache=True, leg_cache_root=None):
    """Install the room `arm` has to fly in for one stage. -> the frozen ids.

    Every partner is modelled by what it IS for the whole of this stage: a
    PARKED partner by the capsules of the one pose it holds, an ACTIVE partner
    by the bounding spheres of its whole work-cell envelope.  A leg certified
    against this clears every pose the other actives could possibly be in, so
    the stage's pair clearance is a property of the routing rather than
    something to be re-discovered by the check afterwards.
    """
    fl = FLEET if specs is None else specs
    pens = {a: fl[a].pen for a in fl} if pens is None else pens
    env = {int(a): v for a, v in (envelopes or {}).items() if int(a) != int(arm)}
    sets = {int(a): np.asarray(parks[a], float).reshape(1, 7)
            for a in fl if int(a) != int(arm)}
    frozen.freeze_sets(sets, fl, pens, h_inv, clusters=env)
    frozen.observe(int(arm))
    paper.clear_cache()
    if leg_cache:
        paper.disk_cache_open(leg_cache_root,
                              signature=leg_cache_signature(
                                  {a: sets[a] for a in sets}, env))
    else:
        paper.disk_cache_close()
    return tuple(sorted(sets))


# ---------------------------------------------------------------------------
# 2d.  THE OTHER ACTIVE ARMS, AS WHAT THEY ACTUALLY DO
# ---------------------------------------------------------------------------
# MEASURED, 2026-09-11 (docs/V2_STAGED.md §14): tightening the pose-union
# envelope from a 0.15 m cluster cell with a 40 mm pad to 0.075 m with none
# moves the ink-vs-envelope minimum by 74 mm, moves every park clear of the
# gate, and moves the number of buckets that fly BY NOTHING -- the same six
# refuse at every cell from 0.10 m down, with the RRT tier on as well as off.
#
# SO THE UNION IS THE WRONG OBJECT, not a badly bounded one.  An arm's work-cell
# envelope is every pose it COULD hold anywhere in its cell, and two arms in
# adjacent row bands own overlapping airspace between the parks and the paper;
# no bound on a set that large leaves a neighbour room to fly through it.  What
# the neighbour actually holds is a few hundred poses out of that union -- one
# trajectory -- and THAT is a room a leg can be routed around.
#
# WHAT IT COSTS IN CERTIFICATION, AND IT IS NOT FREE.  A pose-union envelope is
# a property of the STAGE: it is true whatever the neighbour is asked to draw,
# so an arm's plan survives its neighbour being re-planned.  A trajectory room
# is a property of the neighbour's SPECIFIC PLAN, so an arm's certificate is
# void the moment that plan changes.  That dependency is recorded explicitly --
# `ArmStage.depends_on` carries a digest of each neighbour's trajectory -- so a
# re-plan invalidates its neighbours' certificates by name rather than silently.
def trajectory_digest(st: "ArmStage", dt=CHECK_DT) -> str:
    """A digest of the trajectory an arm's neighbours were certified against."""
    if st.timeline is None:
        Q = np.asarray(st.q_park, float).reshape(1, 7)
    else:
        Q = np.asarray(writing.uniform_samples(st.timeline, dt)["q"], float)
    return hashlib.sha256(np.round(Q, 9).tobytes()).hexdigest()[:16]


def trajectory_room(st: "ArmStage", specs=None, pens=None, h_inv=H_INV_DEFAULT,
                    dt=CHECK_DT, cluster=ENVELOPE_CLUSTER, max_n=None):
    """One arm's ACTUAL stage timeline, as swept bounding spheres.
    -> (centres, radii, digest).

    The same reduction the pose-union envelope uses (`cluster_capsules`, which
    CONTAINS what it replaces), over the poses the arm actually holds instead of
    the poses it could hold.  The pad is not a guess here: it is
    `scene_check.check_timeline`'s own 1-Lipschitz between-sample residual,
    `SWEEP_FRAC x` the largest step any capsule endpoint takes between two
    samples of the timeline, so the spheres cover the motion BETWEEN the samples
    and not only at them.
    """
    fl = FLEET if specs is None else specs
    pens = {a: fl[a].pen for a in fl} if pens is None else pens
    spec = fl[st.arm]
    Q = _samples(st, dt, max_n or 10 ** 9)
    path = coordination.ArmPath(int(st.arm), Q, dt, h_inv,
                                float(pens.get(st.arm, spec.pen)), spec)
    keep = [k for k in range(len(path.r))
            if k not in coordination.FROZEN_SWEEP_BANDS]
    A3 = np.asarray(path.A, float)[:, keep]
    B3 = np.asarray(path.B, float)[:, keep]
    step = 0.0
    if len(A3) > 1:
        step = max(float(np.max(np.linalg.norm(np.diff(A3, axis=0), axis=2))),
                   float(np.max(np.linalg.norm(np.diff(B3, axis=0), axis=2))))
    R = np.tile(np.asarray(path.r, float)[keep], len(A3))
    c, r = cluster_capsules(A3.reshape(-1, 3), B3.reshape(-1, 3), R, cluster,
                            SWEEP_FRAC * step)
    return c, r, trajectory_digest(st, dt)


def stage_rooms(arms: dict, specs=None, pens=None, h_inv=H_INV_DEFAULT,
                dt=CHECK_DT, cluster=ENVELOPE_CLUSTER) -> dict:
    """{arm: (centres, radii, digest)} from one stage's ACTUAL timelines."""
    return {int(a): trajectory_room(st, specs, pens, h_inv, dt, cluster)
            for a, st in arms.items()}


def priority_order(actives, buckets, stage) -> tuple[int, ...]:
    """The order the actives of a stage choose their paths in. -> (arm, ...).

    BUSIEST FIRST, because the arm with the most ink has the least freedom.
    It plans against the parked fleet alone and keeps whatever it finds; every
    arm after it plans around what the earlier ones FINALLY did.  Ties break on
    the arm id so the order is a function of the plan and not of dict ordering.
    """
    def ink(a):
        return sum(p.length_m for p in buckets.get((int(stage), int(a)), ()))
    return tuple(sorted((int(a) for a in actives), key=lambda a: (-ink(a), a)))


def ink_vs_envelope(plan: dict, spec, h_inv=H_INV_DEFAULT, pen=None) -> float:
    """The clearance of one certified piece's INK against the live frozen room.

    `plan_stroke` never consults the static set — it gates the paper, the arm's
    own metal and the joint limits, and nothing else — so a piece the atlas
    certifies can still be drawn THROUGH another active arm's envelope.  This is
    the missing half of the same question the legs already answer, and it is
    cheap: the dense joint samples are already in hand.
    """
    if not frozen.active():
        return float("inf")
    P = coordination.chain_world(np.asarray(plan["qs"], float), spec, h_inv,
                                 float(spec.pen if pen is None else pen))
    return float(np.min(frozen.partner_clearance(P)))


# ---------------------------------------------------------------------------
# 2c.  THE PLAN MEMO
# ---------------------------------------------------------------------------
# THE REFUSAL LOOP RE-RUNS THE DP, AND THE DP MOSTLY RETURNS THE SAME PIECES.
# Striking one (stage, arm) out of one stretch of one line changes that line and
# no other, so a round that re-plans every piece from scratch pays for hundreds
# of `plan_stroke` calls it has already made.  The memo is keyed on everything
# the answer depends on -- the arm, its tool, the options, the geometry to 1 um,
# and the frozen room, because the ink-vs-envelope verdict is part of the answer.
_PLAN_MEMO: dict = {}


def plan_memo_clear():
    _PLAN_MEMO.clear()


def _memo_key(arm, pts, o, room):
    return (int(arm), room,
            tuple(sorted((k, repr(v)) for k, v in o.items())),
            np.round(np.asarray(pts, float), 6).tobytes())


def _room_key(envelopes=None) -> str:
    """What room the planner is standing in, for the memo. -> str.

    THE FROZEN IDS ARE NOT THE ROOM.  A bucket planned against five PARKED
    partners and the same bucket planned against three parked partners and two
    ENVELOPES freeze the same five arm ids, so keying on the ids alone would
    serve one room's verdict in the other — and the ink-vs-envelope verdict is
    exactly what differs.  The envelope's own digest goes in with them.
    """
    if not frozen.active():
        return "-"
    env = "".join(f"{int(a)}:{len(v[1])}:{float(np.sum(v[1])):.6f};"
                  for a, v in sorted((envelopes or {}).items()))
    return f"{tuple(frozen.frozen_ids())}|{frozen.observer()}|{env}"


# ---------------------------------------------------------------------------
# 3.  ONE (STAGE, ARM) BUCKET: PLAN, ORDER, FLY
# ---------------------------------------------------------------------------
@dataclass
class PiecePlan:
    """A piece and what `plan_stroke` made of it."""
    piece: Piece
    status: str                         # "ok" | "split" | "degenerate" | "bug"
    reason: str = ""
    plan: dict | None = None
    seconds: float = 0.0
    s_star: float = 0.0     # the fraction `plan_stroke` DID certify, if any

    @property
    def ok(self) -> bool:
        return self.status == "ok"


@dataclass
class ArmStage:
    """One arm's whole job in one stage: plans, order, legs, trajectory."""
    stage: int
    arm: int
    q_park: np.ndarray
    planned: list[PiecePlan] = field(default_factory=list)
    programme: list[dict] = field(default_factory=list)   # ordered seg dicts
    order: list[int] = field(default_factory=list)
    timeline: dict | None = None
    hovers: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    frozen_partners: tuple[int, ...] = ()
    envelope_partners: tuple[int, ...] = ()
    depends_on: dict[int, str] = field(default_factory=dict)
    room_kind: str = "parked"          # parked | envelope | trajectory
    priority: int | None = None        # where in the stage's order it planned
    frozen_poses: dict[int, list[float]] = field(default_factory=dict)
    lift_ladder: tuple[float, ...] | None = None
    ink_clearance: list[float] = field(default_factory=list)
    plan_s: float = 0.0
    seq_s: float = 0.0
    prog_s: float = 0.0
    wall_s: float = 0.0
    note: str = ""

    @property
    def refused(self) -> list[PiecePlan]:
        return [p for p in self.planned if not p.ok]

    @property
    def accepted(self) -> list[PiecePlan]:
        return [p for p in self.planned if p.ok]

    @property
    def duration(self) -> float:
        return 0.0 if self.timeline is None else float(self.timeline["duration"])

    @property
    def ink_m(self) -> float:
        return float(sum(p.piece.length_m for p in self.accepted))


def _seg_of(pp: PiecePlan, index: int) -> dict:
    """A `PiecePlan` in the `segs` contract `sequence` and `writing` already take."""
    return dict(stroke_id=int(pp.piece.line), seg=int(index),
                color="", kind="piece", s_range=(0.0, 1.0), direction=1,
                flipped=False, length=float(pp.plan["arc_len"]),
                pts=np.asarray(pp.plan["pts"], float),
                stage=int(pp.piece.stage), piece=int(pp.piece.k),
                plan=pp.plan)


def plan_bucket(stage: int, arm: int, pieces: Sequence[Piece], specs=None,
                pens=None, parks=None, h_inv=H_INV_DEFAULT, opts=None,
                sequencer="opt", seq_opts=None, leg_cache=True,
                leg_cache_root=None, fly=True, on_piece: Callable | None = None,
                envelopes=None, lift_ladder=None, ink_gate=None,
                verbose=False) -> ArmStage:
    """Plan, order and fly one (stage, arm) bucket. -> ArmStage.

    The four steps, and none of them is new machinery:

      1. the FIVE PARTNERS ARE FROZEN at their parks before a single call is
         made, so every leg this bucket certifies is certified against the real
         capsules of the arms that are really standing there;
      2. every piece goes through `stroke_api.plan_stroke` -- the same funnel
         `scripts/csail_allocate.py` uses -- and a piece it refuses is RECORDED
         AND DROPPED rather than split: the DP's atlas-permitted piece count is
         a prefilter, and what fraction of it the local planner refuses is a
         measurement this module exists to take (docs/V2_STAGED.md);
      3. the accepted pieces are ordered by `allocate.sequence_arm`, which is
         `sequence.cost_matrix` + `sequence.solve` -- one tour per bucket, the
         same `segs` contract the allocator hands it today;
      4. `writing.arm_program` lays the timeline down: park -> entry hover ->
         draw -> exit hover -> ... -> park, with every pen-up leg a
         `paper.route` against the frozen room and the persistent store.

    `park=PARK_HOME` is not the idle policy this repo usually runs, and it is
    not a preference here either: a stage BARRIER is defined as every arm
    pen-up, stopped, at the park the NEXT stage's envelopes were certified
    against (docs/ARCHITECTURE_V2.md section 2e), so a stage programme that did
    not end there would not be a stage programme.
    """
    fl = FLEET if specs is None else specs
    pens = {a: fl[a].pen for a in fl} if pens is None else pens
    parks = shipped_parks(fl) if parks is None else parks
    spec = fl[arm]
    q_park = np.asarray(parks[arm], float).reshape(7)
    st = ArmStage(int(stage), int(arm), q_park)
    t_all = time.perf_counter()

    # THE ROOM.  With `envelopes` the other ACTIVES are in it as their whole
    # work-cell envelopes and not only the parked five, which is what makes the
    # stage's pair clearance a property of the routing (see `freeze_stage`).
    if envelopes:
        st.frozen_partners = freeze_stage(arm, parks, envelopes, fl, pens,
                                          h_inv, leg_cache, leg_cache_root)
        st.envelope_partners = tuple(sorted(int(a) for a in envelopes
                                            if int(a) != int(arm)))
        # THE CERTIFICATE NAMES WHAT IT DEPENDS ON.  A pose-union envelope is a
        # property of the stage and survives a neighbour re-planning; a
        # TRAJECTORY room does not, so the digest of the trajectory each
        # neighbour was holding rides on the result and `programme()` writes it
        # out.  A re-plan of one arm changes its digest, and every neighbour
        # whose `depends_on` still names the old one is stale by inspection.
        st.depends_on = {int(a): (str(v[2]) if len(v) > 2 else "")
                         for a, v in envelopes.items() if int(a) != int(arm)}
        st.room_kind = ("trajectory" if any(len(v) > 2
                                            for v in envelopes.values())
                        else "envelope")
    else:
        st.frozen_partners = freeze_partners(arm, parks, fl, pens, h_inv,
                                             leg_cache, leg_cache_root)
    st.frozen_poses = {int(a): [float(x) for x in q]
                       for a, q in frozen.poses().items()}
    ladder0 = writing.HOVER_LADDER
    if lift_ladder is not None:
        # THE SECOND LEVER, AND IT IS A CALLER'S KNOB NOT A CONSTANT CHANGE.
        # Two adjacent row bands that cannot be separated in the plane can be
        # separated in z: the hover ladder is where the pen-up legs fly, and
        # `lifted_or_lower`'s memo is keyed on the heights, so raising it for
        # one stage moves that stage's legs and nothing else.
        writing.HOVER_LADDER = tuple(float(z) for z in lift_ladder)
        paper.clear_cache()
        st.lift_ladder = tuple(float(z) for z in lift_ladder)

    o = dict(opts or {})
    o.setdefault("h_inv", h_inv)
    o.setdefault("pen_ext", pens.get(arm))
    # THE INK CHECK IS A MEASUREMENT BY DEFAULT AND A GATE ONLY IF ASKED.
    # The envelope this compares against is a CONSERVATIVE bound -- a stride-2
    # pose sample, bounded by grid-local spheres, inflated by `ENVELOPE_PAD` --
    # so a piece that reads under the gate against it has not been shown to
    # violate anything; it has been shown to be close to a bound that is
    # deliberately larger than the thing it bounds.  Refusing ink on that would
    # throw away certified metres to a conservatism.  `ink_gate=PAIR_MARGIN`
    # turns it into a refusal for a caller that wants the strict reading.
    gate = None if ink_gate is None else float(ink_gate)
    t0 = time.perf_counter()
    for pc in pieces:
        if on_piece is not None:
            on_piece(stage, arm, pc)
        t1 = time.perf_counter()
        mk = _memo_key(arm, pc.pts, o, _room_key(envelopes))
        hit = _PLAN_MEMO.get(mk)
        if hit is not None:
            st.planned.append(PiecePlan(pc, hit[0], hit[1], hit[2], 0.0,
                                        hit[3]))
            continue
        r = stroke_api.plan_stroke(pc.pts, spec, o)
        status = str(r.get("status"))
        reason = str(r.get("reason") or "")
        if status == "ok" and envelopes:
            # ...AND THE INK ITSELF HAS TO CLEAR THE OTHER ACTIVES.
            # `plan_stroke` never consults the static set, so a piece can be
            # certified and still be drawn through a neighbour's envelope.  A
            # piece that does is a REFUSAL like any other and goes back to the
            # DP with this (stage, arm) struck out of its capability set.
            d = ink_vs_envelope(r, spec, h_inv, pens.get(arm))
            st.ink_clearance.append(float(d))
            if gate is not None and d < gate:
                status, reason = "refused", "ink_vs_active_envelope"
        keep = r if status == "ok" else None
        star = float(r.get("s_star", 0.0) or 0.0)
        _PLAN_MEMO[mk] = (status, reason, keep, star)
        st.planned.append(PiecePlan(pc, status, reason, keep,
                                    time.perf_counter() - t1, star))
    st.plan_s = time.perf_counter() - t0
    good = st.accepted
    if verbose:
        print(f"  stage {stage} arm {arm}: {len(good)}/{len(pieces)} pieces "
              f"planned in {st.plan_s:.1f} s")
    if not good or not fly:
        st.programme = [_seg_of(p, i) for i, p in enumerate(good)]
        st.order = list(range(len(good)))
        return _finish(st, t_all, ladder0)

    segs = [_seg_of(p, i) for i, p in enumerate(good)]
    so = dict(h_inv=h_inv, pen_ext=pens.get(arm), q_start=q_park,
              return_home=True)
    so.update(seq_opts or {})
    t0 = time.perf_counter()
    try:
        seq = allocate.sequence_arm(segs, spec, sequencer=sequencer, opts=o,
                                    seq_opts=so)
        st.programme = list(seq["programme"])
        st.order = [int(i) for i in seq["order"]]
    except RuntimeError as exc:
        # NO FEASIBLE TOUR IS A STATEMENT ABOUT THE ROOM, NOT A CRASH.  With the
        # other actives' envelopes in the static set a bucket can become
        # unflyable — every order needs a depot leg the router refuses — and
        # `sequence.solve` says so by raising.  The bucket falls back to the
        # nearest-neighbour order so the timeline is still attempted and the
        # refusal is reported, rather than taking the whole run down.
        st.note = f"sequencer refused ({exc}); nearest-neighbour order used"
        order, _ = allocate.order_nearest(segs, spec.xy)
        st.order = [int(i) for i in order]
        st.programme = [dict(segs[i], flipped=False) for i in order]
    st.seq_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    try:
        st.timeline = writing.arm_program(
            spec, st.programme, h_inv=h_inv, pen_ext=pens.get(arm),
            q_start=q_park, park=writing.PARK_HOME,
            transit_speed=so.get("transit_speed", writing.TRANSIT_SPEED),
            qd_frac=so.get("qd_frac", writing.QD_FRAC))
    except writing.PaperRefused as exc:
        st.note = f"arm_program refused: {exc}"
    st.prog_s = time.perf_counter() - t0
    st.hovers = [(writing.lifted_or_lower(spec, np.asarray(s["plan"]["qs"])[0],
                                          np.asarray(s["plan"]["pts"])[0],
                                          h_inv=h_inv,
                                          pen_ext=pens.get(arm))[0],
                  writing.lifted_or_lower(spec, np.asarray(s["plan"]["qs"])[-1],
                                          np.asarray(s["plan"]["pts"])[-1],
                                          h_inv=h_inv,
                                          pen_ext=pens.get(arm))[0])
                 for s in st.programme]
    return _finish(st, t_all, ladder0)


def _finish(st: ArmStage, t_all: float, ladder0=None) -> ArmStage:
    if ladder0 is not None and writing.HOVER_LADDER != ladder0:
        writing.HOVER_LADDER = ladder0      # a knob, borrowed and given back
        paper.clear_cache()
    st.wall_s = time.perf_counter() - t_all
    return st


# ---------------------------------------------------------------------------
# 4.  THE TWO CHECKS A STAGE HAS TO PASS
# ---------------------------------------------------------------------------
def _samples(st: ArmStage, dt: float, max_n: int, with_pen=False):
    """One arm's stage trajectory on a uniform clock, decimated. -> (N, 7).

    `with_pen` also returns the pen-down mask: `seg >= 0` is the sample the
    timeline is DRAWING at, which is the half of a trajectory the work-cell
    envelope is a claim about.
    """
    if st.timeline is None:
        Q = np.asarray(st.q_park, float).reshape(1, 7)
        return (Q, np.zeros(1, bool)) if with_pen else Q
    u = writing.uniform_samples(st.timeline, dt)
    Q, seg = np.asarray(u["q"], float), np.asarray(u["seg"], int)
    if len(Q) > max_n:
        keep = np.unique(np.linspace(0, len(Q) - 1, max_n).astype(int))
        Q, seg = Q[keep], seg[keep]
    return (Q, seg >= 0) if with_pen else Q


def _chains(Q: np.ndarray, spec, h_inv: float, pen: float) -> np.ndarray:
    return np.array([scene_check._chain(q, spec, h_inv, pen) for q in Q])


def active_pair_gap(stages: dict[int, ArmStage], specs=None, pens=None,
                    h_inv=H_INV_DEFAULT, dt=CHECK_DT, max_n=MAX_CHECK_POSES
                    ) -> dict:
    """Two ACTIVE arms' realised clearance, with NO assumption about timing.

    -> dict(min_m, per_pair, n_samples).

    WHY THIS IS NOT `check_timeline` ON A MERGED CLOCK.  Inside a stage the
    actives are asynchronous by construction -- that is the whole reason the
    stage needs no conductor -- so there is no single alignment of their two
    timelines to check, and checking one would certify a schedule nobody is
    running.  The claim that has to hold is the ENVELOPE claim restricted to
    the poses the planner actually produced: for EVERY pair of instants, one on
    each arm's timeline, the two arms clear the gate.  That is the minimum over
    the CROSS PRODUCT of the two pose sets, and it is what this computes.

    The between-sample bound is `check_timeline`'s own, applied on both axes:
    a chain point moves at most `step[i]` between grid samples, so
    `d(A(t), B(t')) >= d(A_i, B_j) - 0.55 (step^A_i + step^B_j)` for the grid
    cell containing (t, t').  Decimating makes the steps larger and the bound
    more conservative, never wrong.

    INK-VS-INK IS REPORTED SEPARATELY, and it is the number that answers the
    envelope's own question.  A work-cell envelope is a union over POSES -- the
    drawing pose at each certified cell, the hover above it, the park -- and the
    pen-up LEG between two hovers is a PATH whose interior is in none of them
    (docs/ARCHITECTURE_V2.md section 2f names this gap in as many words).  So
    `min_ink_m` is the two arms measured only where BOTH pens are down, which is
    the claim `scripts/workcell_envelopes.py` made and the one the 0.40 m dead
    band was chosen for; `min_m` is the whole trajectory, legs included, which
    is a strictly stronger statement and the one a stage has to pass.
    """
    fl = FLEET if specs is None else specs
    pens = {a: fl[a].pen for a in fl} if pens is None else pens
    arms = sorted(stages)
    rr = scene_check._radii_for(fl, arms)
    Q, DOWN = {}, {}
    for a in arms:
        Q[a], DOWN[a] = _samples(stages[a], dt, max_n, with_pen=True)
    P, SW = {}, {}
    for a in arms:
        P[a] = _chains(Q[a], fl[a], h_inv, scene_check.pen_len(pens.get(a), a))
        step = np.concatenate([[0.0], np.linalg.norm(np.diff(P[a], axis=0),
                                                     axis=2).max(1)])
        SW[a] = SWEEP_FRAC * np.maximum(step, np.concatenate([step[1:], [0.0]]))
    per: dict[str, float] = {}
    per_ink: dict[str, float] = {}
    worst = worst_ink = np.inf
    at = None
    for n, ai in enumerate(arms):
        for aj in arms[n + 1:]:
            Pi, Pj = P[ai], P[aj]
            m, mi = np.inf, np.inf
            for j in range(len(Pj)):
                d = scene_check.pair_clearance(
                    Pi, np.repeat(Pj[j][None], len(Pi), axis=0), rr) \
                    - SW[ai] - SW[aj][j]
                k = int(np.argmin(d))
                if float(d[k]) < m:
                    m = float(d[k])
                    if m < worst:
                        at = (int(ai), int(aj),
                              "ink" if DOWN[ai][k] else "leg",
                              "ink" if DOWN[aj][j] else "leg")
                if DOWN[aj][j] and DOWN[ai].any():
                    mi = min(mi, float(np.min(d[DOWN[ai]])))
            per[f"{ai}-{aj}"] = m
            per_ink[f"{ai}-{aj}"] = mi
            worst = min(worst, m)
            worst_ink = min(worst_ink, mi)
    return dict(min_m=float(worst), min_ink_m=float(worst_ink),
                worst_at=at, per_pair=per, per_pair_ink=per_ink,
                n_samples={int(a): int(len(Q[a])) for a in arms},
                n_ink={int(a): int(DOWN[a].sum()) for a in arms})


def drop_bands(spec, owners: Iterable[int]):
    """`spec` with the pose-invariant BODY BANDS of `owners` removed.

    A `body:<aid>_column<k>` box is a 0.32 m AABB standing in for a neighbour's
    shoulder and upper links AT ANY POSE, written in by
    `mounts.attach_body_columns` because at atlas-sweep time nobody has decided
    what pose the neighbour will hold (`frozen.py`).  In a stage timeline
    EVERYBODY'S POSE IS DECIDED -- the actives are in the trajectory and the
    parked partners are at an identified park -- so every one of those arms is
    already in the check as its real capsules, and its band is the same arm
    counted a second time in its most conservative form.

    This is `frozen.filter_boxes` applied to the CHECK instead of to the
    planner, and it is the same convention `scene_check.neighbour_columns`
    already uses for the cylinder version of the same object ("in the room" =
    not in this timeline).  TRUE STRUCTURE IS NEVER DROPPED: mounts, plates,
    the drop cluster and the runway all stay, and so does the band of any arm
    NOT named in `owners`.
    """
    import dataclasses
    own = {int(x) for x in owners}
    kw = {}
    for fname in ("mount_boxes", "column_boxes"):
        cur = getattr(spec, fname, None)
        if not cur:
            continue
        new = tuple(b for b in cur
                    if frozen.band_owner(b.get("name")
                                         if isinstance(b, dict) else None)
                    not in own)
        if len(new) != len(cur):
            kw[fname] = new
    return dataclasses.replace(spec, **kw) if kw else spec


REFINE_MAX = 3              # halvings of dt a borderline verdict may buy
REFINE_FRAMES = 40000       # ...and the frame count at which it stops trying


def solo_check(st: ArmStage, parks: dict[int, np.ndarray], specs=None,
               pens=None, h_inv=H_INV_DEFAULT, margin=PAIR_MARGIN,
               dt=CHECK_DT, sub=2, bands=False, refine=True) -> dict:
    """One active arm's whole stage timeline against the PARKED fleet and the steel.

    `scene_check.check_timeline` with the five partners held at their parks for
    every frame.  This is the independent re-derivation of the claim the planner
    made with `frozen` switched on: the same poses, no planner state, and the
    parked partners measured as the capsules they ARE rather than as the
    envelope they are not.

    `bands=False` (the default) drops the partners' pose-invariant body bands
    from the STATIC set, because every arm those bands stand for is in this
    timeline as its own capsules -- see `drop_bands`.  `bands=True` is the
    shipped pose-invariant reading, kept because the difference between the two
    is exactly what the frozen model buys and it should be reported rather than
    assumed: the band refuses what the arm does not, by up to 127 mm
    (`frozen.py`, measured 2026-09-09).
    """
    fl = FLEET if specs is None else specs
    pens = {a: fl[a].pen for a in fl} if pens is None else pens
    Q = _samples(st, dt, 10 ** 9)
    traj = {int(st.arm): Q}
    for a in sorted(fl):
        if int(a) == int(st.arm):
            continue
        traj[int(a)] = np.repeat(
            np.asarray(parks[a], float).reshape(1, 7), len(Q), axis=0)
    room = ({a: drop_bands(s, traj) for a, s in fl.items()} if not bands
            else fl)
    rep = scene_check.check_timeline(
        traj, dt, margin, programs=None, h_inv=h_inv,
        pen_ext={int(a): scene_check.pen_len(pens.get(a), a) for a in fl},
        sub=sub, verbose=False, fleet=room)
    # THE COLUMN CYLINDER IS THE SAME OBJECT AS THE BAND and it is reported the
    # same way: `neighbour_columns` hands out a cylinder for every other arm
    # whether or not it is in the timeline, so under the frozen model it is
    # informational and not a gate.  Everything else in the report -- inter-arm
    # (real capsules), steel, paper, self, joint limits -- is the verdict.
    rep["column_is_gate"] = bool(bands)
    rep["dt"] = float(dt)
    # A BORDERLINE VERDICT IS LOOKED AT MORE CLOSELY, NOT BELIEVED.
    # `check_timeline` subtracts a 1-Lipschitz residual, `0.55 x (step_i +
    # step_j)`, computed at whatever rate the timeline was handed in at -- and
    # unlike its own frame and paper gates, its INTER-ARM gate does not refine.
    # So a leg sampled coarsely is charged for being sampled coarsely.  Measured
    # on CSAIL stage 2, arm 97 reads 28.23 mm at dt = 0.05, 36.58 at 0.02 and
    # 39.38 at 0.01: ELEVEN MILLIMETRES OF THAT WAS THE SAMPLING.  Refining only
    # where the verdict binds is `block_screen`'s own rule -- "a cell between
    # the bounds has to be LOOKED AT rather than believed either way" -- and it
    # costs nothing on the runs that pass.
    if refine and float(rep["min_clearance"]) < float(margin) \
            and int(rep.get("n_frames", 0)) < REFINE_FRAMES:
        for _ in range(REFINE_MAX):
            dt *= 0.5
            fine = solo_check(st, parks, specs, pens, h_inv, margin, dt, sub,
                              bands, refine=False)
            better = float(fine["min_clearance"]) > float(rep["min_clearance"])
            rep, fine = (fine, rep) if better else (rep, fine)
            if float(rep["min_clearance"]) >= float(margin) or \
                    int(rep.get("n_frames", 0)) >= REFINE_FRAMES:
                break
    if "full_ok" not in rep:           # check_timeline's own verdict, which
        rep["full_ok"] = bool(rep["ok"])  # also gates the column CYLINDER
    if not bands and not rep.get("_regated"):
        rep["_regated"] = True
        rep["ok"] = bool(
            float(rep["min_clearance"]) >= float(margin)
            and not rep.get("frame_failed") and not rep.get("paper_failed")
            and not rep.get("self_failed") and int(rep.get("frozen_failed", 0)) == 0
            and min(rep["joint_margin"].values()) > 0.0
            and bool(rep.get("monotone", True)))
    return rep


# ---------------------------------------------------------------------------
# 4b.  THE REFUSAL LOOP
# ---------------------------------------------------------------------------
# EVERY NUMBER IN docs/V2_TRACES.md IS WHAT THE 2 cm ATLAS *PERMITS*, NOT A PLAN.
# `plan_stroke` refuses 11.1 % of the logo's pieces and 18.1 % of a 1 000-line
# set's, and two thirds of that is the redundancy band giving out somewhere
# along a piece the lattice certified cell by cell.  A refused piece is not ink
# that cannot be drawn -- it is ink that cannot be drawn BY THAT ARM IN THAT
# STAGE, which is one bit of one atom's capability set.  So the honest close of
# the loop is to strike that bit and ask the DP again: the neighbours absorb
# what they can, the piece is re-cut where they cannot, and the whole thing
# costs 10 ms a round (docs/V2_TRACES.md section 4) plus the pieces that
# actually moved, which the plan memo is what keeps cheap.
REFUSAL_ROUNDS = 4          # a cap, not a convergence criterion
MASK_EPS = 1e-9


def mask_atoms(atoms, masks):
    """Strike `(state, s0, s1)` out of the span, CUTTING the atoms. -> [Atom].

    AN ATOM IS INDIVISIBLE WITH RESPECT TO THE CAPABILITY MAP, AND A REFUSAL IS
    A NEW TRANSITION IN IT.  Clearing the bit on every atom the ban merely
    TOUCHES throws the state out of the part of the atom the ban does not
    cover, and on a picture whose atoms are 0.2 m long that is most of the ink:
    measured on the CSAIL logo, banning whole atoms took the loop from 100 % to
    80.6 % coverage and cutting them keeps it at 100 %.  So the ban cuts the
    atom at both its ends and clears the bit only on the middle.

    Returns a NEW list; the caller must use it.
    """
    out = list(atoms)
    for (k, b0, b1) in masks:
        b0, b1, bit = float(b0), float(b1), ~(1 << int(k))
        nxt = []
        for a in out:
            if a.s1 <= b0 + MASK_EPS or a.s0 >= b1 - MASK_EPS:
                nxt.append(a)
                continue
            lo, hi = max(a.s0, b0), min(a.s1, b1)
            if a.s0 < lo - MASK_EPS:
                nxt.append(traces_mod.Atom(a.s0, lo, a.bits))
            nxt.append(traces_mod.Atom(lo, hi, a.bits & bit))
            if hi < a.s1 - MASK_EPS:
                nxt.append(traces_mod.Atom(hi, a.s1, a.bits))
        out = nxt
    return out


def plan_lines_masked(lines, cap, opts=None, masks=None):
    """`traces.plan_lines`, with `(state, s0, s1)` bans per line. -> Plan.

    Identical to `traces.plan_lines` -- same longest-first visit order, same
    load tie-break, same absorption and seam placement -- with one extra step
    between the atoms and the DP: the bits a previous round's refusals struck
    out are cleared before the chain is solved.  An atom that loses its LAST
    bit becomes `UNCOVERED`, which is the DP's own way of saying "nobody can
    draw this", and it shows up in the summary as a gap rather than silently.
    """
    opts = opts or traces_mod.Options()
    masks = masks or {}
    t0 = time.perf_counter()
    L = [np.asarray(p, float) for p in lines]
    total = sum(float(traces_mod.cumlen(p)[-1]) for p in L) or 1.0
    load = np.zeros(cap.n_states)
    order = sorted(range(len(L)),
                   key=lambda i: -float(traces_mod.cumlen(L[i])[-1]))
    out: list = [None] * len(L)
    for i in order:
        w = (load / total) if (opts.balance or opts.balance_w) else None
        atoms, p, cum = traces_mod.atoms_of(L[i], cap, opts.ds, opts.tol)
        atoms = mask_atoms(atoms, masks.get(i, ()))
        assign = traces_mod.solve_line(atoms, cap, w, opts.balance_w)
        assign = traces_mod.absorb_short(atoms, assign, cap, opts.min_piece_m)
        pieces, seams = traces_mod.place_seams(atoms, assign, cap,
                                               opts.overdraw_m)
        lp = traces_mod.LinePlan(i, p, cum, atoms, assign, pieces, seams)
        for pc in lp.pieces:
            load[pc.state] += pc.length
        out[i] = lp
    return traces_mod.Plan(cap, out, time.perf_counter() - t0, opts)


def resolve_refusals(lines, cap, specs=None, pens=None, parks=None,
                     h_inv=H_INV_DEFAULT, opts=None, trace_opts=None,
                     stages=None, pattern=None, envelopes=None,
                     rounds=REFUSAL_ROUNDS, leg_cache=True,
                     leg_cache_root=None, verbose=True):
    """Iterate DP -> `plan_stroke` -> strike the refusal, to a fixed point.

    -> (plan, masks, log).  Nothing is flown here: the loop is over the PIECES
    and the pieces are all that change, so paying for the legs and the ordering
    inside it would be paying for them `rounds` times over.
    """
    fl = FLEET if specs is None else specs
    pens = {a: fl[a].pen for a in fl} if pens is None else pens
    parks = shipped_parks(fl) if parks is None else parks
    masks: dict[int, list[tuple[int, float, float]]] = {}
    seen: set = set()
    log: list[dict] = []
    plan = None
    stuck: list = []
    for rnd in range(int(rounds) + 1):
        plan = plan_lines_masked(lines, cap, trace_opts, masks)
        pcs = pieces_of(plan)
        buckets = bucket(pcs)
        want = (sorted({s for (s, _) in buckets}) if stages is None
                else list(stages))
        new = 0
        stuck = []
        for s in want:
            env = (envelopes or {}).get(s) if envelopes else None
            for a in sorted({b for (t, b) in buckets if t == s}):
                st = plan_bucket(s, a, buckets[(s, a)], fl, pens, parks, h_inv,
                                 opts, leg_cache=leg_cache,
                                 leg_cache_root=leg_cache_root, fly=False,
                                 envelopes=env, verbose=False)
                for pp in st.refused:
                    pc = pp.piece
                    span = ban_span(pp)
                    if pc.state < 0 or span is None:
                        stuck.append(pp)
                        continue
                    b = (pc.state, round(float(span[0]), 6),
                         round(float(span[1]), 6))
                    if b in seen:
                        stuck.append(pp)
                        continue
                    seen.add(b)
                    masks.setdefault(pc.line, []).append(b)
                    new += 1
        d = plan.summary()
        log.append(dict(round=int(rnd), pieces=int(d["n_pieces"]),
                        refused=int(new), unplannable=int(len(stuck)),
                        gaps=int(d["gaps"]),
                        covered_frac=float(d["covered_frac"]),
                        drawn_m=float(d["drawn_m"]),
                        uncovered_m=float(d["uncovered_m"])))
        if verbose:
            print(f"  refusal round {rnd}: {d['n_pieces']} pieces, "
                  f"{new} refused, coverage {100 * d['covered_frac']:.2f} %")
        if not new:
            break
    return plan, masks, log


def ban_span(pp: "PiecePlan"):
    """What a refusal actually bans. -> (s0, s1) in the LINE's arc, or None.

    A REFUSAL IS NOT ALWAYS ABOUT THE WHOLE PIECE, and banning the whole piece
    is what turns an 11 % refusal rate into a 19 % coverage hole.  `plan_stroke`
    hands back `s_star`, "the normalised arc length up to which it IS planned",
    and a head plan is an "ok" result with all of its guarantees -- so a `split`
    at s* = 0.6 means the arm CAN draw the first 60 % and the DP should only be
    told about the other 40 %.

    A `degenerate` piece bans NOTHING and comes back as `stuck`: "too short",
    "off sheet" and "too short after clip" are statements about the PIECE, not
    about the arm, so no neighbour would do better and striking the state out
    would only spread the hole.  The same for a ban that has already been made
    and did not help -- a span may not be banned twice, or the loop would walk
    the same piece down the state list for ever.
    """
    if pp.status == "degenerate":
        return None
    pc = pp.piece
    lo, hi = float(pc.s0), float(pc.s1)
    if pp.status == "split" and 0.0 < pp.s_star < 1.0:
        return (lo + pp.s_star * (hi - lo), hi)
    return (lo, hi)


SHIPPED_LADDER = tuple(writing.HOVER_LADDER)
ROW_LIFT_STEP = 0.14        # m of z between two adjacent row bands' hovers


def row_lift_ladder(arm, step=ROW_LIFT_STEP, base=None):
    """A hover ladder that separates the three ROW bands in z. -> tuple.

    THE SECOND LEVER, AND THE CHEAPER ONE.  Two adjacent row bands cannot be
    separated in the plane -- the elbow reaches across, which is
    docs/V2_WORKCELLS.md section 1 -- but the pen-up legs do not have to fly at
    the same height in both.  `writing.HOVER_LADDER` is where they fly, and
    `lifted_or_lower` tries its rungs in order, so putting this row's height
    first and the shipped ladder behind it raises the legs WHERE THE ARM CAN
    HOLD THE POSE and changes nothing where it cannot.
    """
    b = float(writing.LIFT_Z if base is None else base)
    z = b + float(step) * int(traces_mod.ROW_OF.get(int(arm), 0))
    return (z, max(0.02, z - 0.02), z + 0.03) + SHIPPED_LADDER


# ---------------------------------------------------------------------------
# 5.  A WHOLE STAGE, AND THE WHOLE PROGRAMME
# ---------------------------------------------------------------------------
@dataclass
class StageResult:
    stage: int
    actives: tuple[int, ...]
    arms: dict[int, ArmStage]
    pair: dict = field(default_factory=dict)
    solo: dict = field(default_factory=dict)
    wall_s: float = 0.0
    lift_used: bool = False
    room_passes: list = field(default_factory=list)
    order: tuple[int, ...] = ()
    order_rank: int = 0        # 0 = the ink-first order was enough
    orders_tried: int = 0

    @property
    def duration(self) -> float:
        """The stage costs its BUSIEST arm: the actives never wait for each other."""
        return max((a.duration for a in self.arms.values()), default=0.0)

    @property
    def n_pieces(self) -> int:
        return sum(len(a.accepted) for a in self.arms.values())

    @property
    def ink_m(self) -> float:
        return float(sum(a.ink_m for a in self.arms.values()))

    @property
    def flown(self) -> int:
        return sum(1 for st in self.arms.values() if st.timeline is not None)

    @property
    def with_ink(self) -> int:
        return sum(1 for st in self.arms.values() if st.accepted)

    @property
    def complete(self) -> bool:
        """Did every bucket that HAS ink produce a timeline?

        A VACUOUS PASS MUST NOT READ AS A PASS.  A stage in which nothing flew
        is six arms standing at their parks, and both checks clear it by a
        quarter of a metre -- which is true, and says nothing whatever about the
        programme it was supposed to certify.  Measured on the v3 control: two
        stages of eight flew nothing at all and both were labelled PASS.
        """
        return all(st.timeline is not None
                   for st in self.arms.values() if st.accepted)

    @property
    def ok(self) -> bool:
        pair_ok = self.pair.get("min_m", np.inf) >= PAIR_MARGIN
        solo_ok = all(r.get("ok", False) for r in self.solo.values())
        return bool(self.complete and pair_ok and solo_ok)


@dataclass
class StagedResult:
    pattern: str
    stages: list[StageResult]
    pieces: list[Piece]
    dp_s: float = 0.0
    plan_s: float = 0.0
    check_s: float = 0.0
    wall_s: float = 0.0
    ttfm_s: float | None = None
    parks: dict[int, list[float]] = field(default_factory=dict)
    refusals: list = field(default_factory=list)
    envelope_s: float = 0.0
    summary_dp: dict = field(default_factory=dict)

    @property
    def makespan(self) -> float:
        """The barrier serialises the stages, so the sequence costs the SUM."""
        return float(sum(s.duration for s in self.stages))

    @property
    def parallel_plan_s(self) -> float:
        """Planning wall clock if each stage's actives plan on their own worker."""
        return float(sum(max((a.wall_s for a in s.arms.values()), default=0.0)
                         for s in self.stages))

    def barriers(self) -> list[dict]:
        """The rendezvous between consecutive stages. -> [dict].

        A barrier is not a time, it is a STATE: every arm pen-up, stopped, at
        the specific park the next stage's envelope set was certified against,
        with its queue drained and no un-cleared fault (docs/ARCHITECTURE_V2.md
        section 2e).  Recorded here as the park identity per arm, because park
        IDENTITY is what the guarantee is indexed by.
        """
        out = []
        for k, s in enumerate(self.stages):
            out.append(dict(kind="stage", index=k, before_stage=s.stage,
                            actives=[int(a) for a in s.actives],
                            parks={str(a): q for a, q in self.parks.items()}))
        out.append(dict(kind="end", index=len(self.stages), before_stage=None,
                        actives=[], parks={str(a): q
                                           for a, q in self.parks.items()}))
        return out


def run(lines, pattern=None, coverage=None, specs=None, pens=None, parks=None,
        h_inv=H_INV_DEFAULT, opts=None, trace_opts=None, stages=None,
        leg_cache=True, leg_cache_root=None, check=True, fly=True,
        measure_ttfm=True, dt=CHECK_DT, max_check_poses=MAX_CHECK_POSES,
        route_jobs=None, envelopes=True, atlas_dir=ATLAS_DEFAULT,
        refusal_rounds=REFUSAL_ROUNDS, lift_retry=True, env_kw=None,
        trajectory_rooms=False, room_iterations=1,
        room_order="priority", order_search=6,
        verbose=True, on_piece=None) -> StagedResult:
    """The whole of build item 4: lines -> pieces -> plans -> legs -> checks.

    `lines` are polylines in paper metres, `pattern` a `traces.Pattern` (the
    eight-stage zigzag by default) and `coverage` a `traces.Coverage` (the
    shipped atlas by default).  Everything else is the fleet's own.

    `envelopes` puts the OTHER ACTIVE arms into the room as their whole
    work-cell envelopes, which is what makes the stage's pair clearance a
    property of the routing rather than of luck; `lift_retry` is the fallback
    for a stage that still does not clear, and it raises that stage's hover
    ladder per ROW BAND so the two bands' legs fly at different heights.
    `refusal_rounds` closes the loop on the pieces `plan_stroke` will not take.
    """
    t_all = time.perf_counter()
    fl = FLEET if specs is None else specs
    pens = {a: fl[a].pen for a in fl} if pens is None else pens
    parks = shipped_parks(fl) if parks is None else parks
    pattern = traces_mod.zigzag_pattern() if pattern is None else pattern
    # THE ROUTE SCREEN'S FORK COUNT IS A CALLER'S BUSINESS.  `sequence.route_jobs`
    # defaults to `min(cpu_count, 24)`, which is the right answer for a machine
    # running one allocation and the wrong one for a box running several stages
    # (or several of these) at once; `sequence.ROUTE_JOBS` is the knob that is
    # already there for it and `tests/test_balance.py` already sets it the same
    # way.  `None` leaves the module exactly as shipped.
    from . import sequence as _sequence
    jobs0 = _sequence.ROUTE_JOBS
    if route_jobs is not None:
        _sequence.ROUTE_JOBS = int(route_jobs)
    if coverage is None:
        coverage = traces_mod.coverage_from_atlas(ATLAS_DEFAULT,
                                                  arms=tuple(sorted(fl)))
    cap = traces_mod.capability(coverage, pattern)
    want = list(range(pattern.n_stages) if stages is None else stages)
    for s in want:
        bad = same_row_pairs(stage_actives(pattern, s))
        if bad:
            raise ValueError(f"stage {s} puts a same-ROW pair in the air "
                             f"together: {bad} -- see traces.zigzag_pattern")

    env: dict[int, dict] = {}
    env_s = 0.0
    # PASS 1 IS SOLO AGAINST THE PARKED FLEET, ALWAYS.  With trajectory rooms
    # the pose-union envelope is not merely unnecessary, it is the thing being
    # replaced: building pass 1 against it would start the iteration from the
    # room that does not fly, and there would be no pass-1 trajectory to derive
    # a room from.
    if trajectory_rooms:
        envelopes = False
    if envelopes and atlas_dir:
        t0 = time.perf_counter()
        for s in want:
            env[s] = stage_envelopes(pattern, s, atlas_dir, fl, pens, parks,
                                     h_inv, verbose=verbose, **(env_kw or {}))
        env_s = time.perf_counter() - t0
        if verbose:
            print(f"  stage envelopes built in {env_s:.1f} s: "
                  + "  ".join(f"s{s}=" + "/".join(str(len(v[1]))
                                                  for v in env[s].values())
                              for s in want))

    plan_memo_clear()
    t0 = time.perf_counter()
    if refusal_rounds:
        tplan, masks, rlog = resolve_refusals(
            lines, cap, fl, pens, parks, h_inv, opts, trace_opts, want,
            pattern, env or None, refusal_rounds, leg_cache, leg_cache_root,
            verbose)
    else:
        tplan, masks, rlog = plan_lines_masked(lines, cap, trace_opts), {}, []
    pcs = pieces_of(tplan)
    dp_s = time.perf_counter() - t0
    buckets = bucket(pcs)

    ttfm = None
    if measure_ttfm:
        ttfm = time_to_first_motion(buckets, want, fl, pens, parks, h_inv, opts,
                                    leg_cache, leg_cache_root,
                                    float(tplan.seconds))

    out: list[StageResult] = []
    plan_s = check_s = 0.0
    for s in want:
        acts = stage_actives(pattern, s)
        t1 = time.perf_counter()
        arms = _plan_stage(s, acts, buckets, fl, pens, parks, h_inv, opts,
                           leg_cache, leg_cache_root, fly, on_piece,
                           env.get(s), None, verbose)
        # THE TRAJECTORY ROOM, AND WHY IT IS TWO PASSES.  Pass 1 is every active
        # planned SOLO against the parked fleet, which is the room that flies
        # but does not separate the actives.  Pass 2 re-plans each active
        # against what the OTHERS ACTUALLY DID in pass 1 -- a few hundred poses
        # instead of the whole work-cell union -- and pass 3 repeats it on pass
        # 2's trajectories.  It is a fixed-point iteration and it is NOT the
        # certificate: re-planning A moves A, which is a room B was certified
        # against, so the claim is only ever closed by the INDEPENDENT
        # whole-timeline pair check below.  The iteration is what gets the
        # trajectories apart; `active_pair_gap` is what proves they are.
        rooms, order = [], ()
        order_rank, orders_tried = 0, 0
        if fly and trajectory_rooms:
            if room_order == "priority":
                arms, order, rm, rank, tried = _order_search(
                    s, acts, buckets, fl, pens, parks, h_inv, opts, leg_cache,
                    leg_cache_root, on_piece, dt, ENVELOPE_CLUSTER,
                    order_search, verbose)
                rooms.append({int(a): str(v[2]) for a, v in rm.items()})
                order_rank, orders_tried = rank, tried
            else:
                # THE SUPERSEDED SIMULTANEOUS SCHEME, kept because the
                # measurement that retired it is worth being able to reproduce:
                # on CSAIL stage 0 it is strictly worse than planning solo.
                for it in range(max(1, int(room_iterations))):
                    rm = stage_rooms(arms, fl, pens, h_inv, dt)
                    rooms.append({int(a): str(v[2]) for a, v in rm.items()})
                    if verbose:
                        print(f"  stage {s} room pass {it + 1}: "
                              + "  ".join(f"{a}={len(v[1])}sph" for a, v in
                                          sorted(rm.items())))
                    arms = _plan_stage(s, acts, buckets, fl, pens, parks,
                                       h_inv, opts, leg_cache, leg_cache_root,
                                       fly, on_piece, rm, None, verbose)
        plan_s += time.perf_counter() - t1
        sr = StageResult(int(s), acts, arms, wall_s=time.perf_counter() - t1,
                         room_passes=rooms, order=tuple(order),
                         order_rank=int(order_rank),
                         orders_tried=int(orders_tried))
        if check and fly:
            t2 = time.perf_counter()
            _check_stage(sr, acts, fl, pens, parks, h_inv, dt, max_check_poses)
            # THE FALLBACK, AND ONLY WHERE IT IS NEEDED.  A stage whose legs
            # still do not clear is re-planned once with the row-band hover
            # ladder; whichever of the two passes is kept, and `lift_used`
            # records which was needed so the report can say so.
            if lift_retry and len(arms) > 1 and \
                    sr.pair.get("min_m", np.inf) < PAIR_MARGIN:
                if verbose:
                    print(f"  stage {s}: {1000 * sr.pair['min_m']:+.1f} mm "
                          "-> retrying with the row-band hover ladder")
                lad = {a: row_lift_ladder(a) for a in acts}
                arms2 = _plan_stage(s, acts, buckets, fl, pens, parks, h_inv,
                                    opts, leg_cache, leg_cache_root, fly,
                                    on_piece, env.get(s), lad, verbose)
                sr2 = StageResult(int(s), acts, arms2,
                                  wall_s=time.perf_counter() - t1)
                _check_stage(sr2, acts, fl, pens, parks, h_inv, dt,
                             max_check_poses)
                if sr2.pair.get("min_m", -np.inf) > sr.pair.get("min_m",
                                                               -np.inf):
                    sr, sr2.lift_used = sr2, True
                    sr.lift_used = True
            check_s += time.perf_counter() - t2
        out.append(sr)
        if verbose:
            _report_stage(sr)
    thaw()
    _sequence.close_pool()
    _sequence.ROUTE_JOBS = jobs0
    res = StagedResult(pattern.name, out, pcs, dp_s, plan_s, check_s,
                       time.perf_counter() - t_all, ttfm,
                       {int(a): [float(x) for x in np.asarray(q).ravel()]
                        for a, q in parks.items()},
                       refusals=rlog, envelope_s=env_s,
                       summary_dp=tplan.summary())
    return res


def _priority_stage(s, acts, buckets, fl, pens, parks, h_inv, opts, leg_cache,
                    leg_cache_root, on_piece, dt, cluster, verbose):
    """One stage, planned in PRIORITY ORDER against the rooms already fixed.

    -> (arms, order, rooms).

    WHY THIS AND NOT AN ITERATION.  Planning every active against every other
    active's PREVIOUS trajectory is circular: re-planning A moves a room B was
    certified against, so a pass never closes and -- measured on CSAIL stage 0 --
    it can be strictly worse than planning solo, because each arm is asked to
    yield to a neighbour's unconstrained path and nobody actually yields.

    A PRIORITY ORDER CLOSES IT IN ONE SWEEP.  Arm 1 plans against the parked
    fleet and its trajectory is then FINAL.  Arm k plans against the parked
    fleet plus the final trajectories of arms 1..k-1, so when it finishes, every
    pair (i, k) with i < k is certified against the path arm i actually flies --
    and arm i never moves again.  Every pair is therefore certified, exactly,
    with no iteration and no circularity, and the dependency graph is a DAG:
    arm k depends on 1..k-1 and on nothing after it.

    This is the spatial analogue of what `coordination.coordinate` already does
    in time -- a priority search over orders -- with the order fixed by ink
    rather than searched, because the busiest arm is the one with least room to
    give and there are at most three actives in a stage.
    """
    order = priority_order(acts, buckets, s)
    return _sweep_in_order(s, order, buckets, fl, pens, parks, h_inv, opts,
                           leg_cache, leg_cache_root, on_piece, dt, cluster,
                           verbose)


def _sweep_in_order(s, order, buckets, fl, pens, parks, h_inv, opts, leg_cache,
                    leg_cache_root, on_piece, dt, cluster, verbose):
    """One stage, planned in the given order. -> (arms, order, rooms)."""
    fixed: dict[int, tuple] = {}
    arms: dict[int, ArmStage] = {}
    for k, a in enumerate(order):
        st = plan_bucket(s, a, buckets.get((s, a), []), fl, pens, parks, h_inv,
                         opts, leg_cache=leg_cache,
                         leg_cache_root=leg_cache_root, fly=True,
                         on_piece=on_piece,
                         envelopes=(dict(fixed) if fixed else None),
                         verbose=verbose)
        st.priority = int(k)
        arms[int(a)] = st
        fixed[int(a)] = trajectory_room(st, fl, pens, h_inv, dt, cluster)
        if verbose:
            print(f"  stage {s} priority {k}: arm {a} "
                  f"({len(st.accepted)} pieces, {st.ink_m:.3f} m) "
                  f"-> {len(fixed[int(a)][1])} spheres"
                  + (f", avoiding {sorted(st.depends_on)}" if st.depends_on
                     else ", free"))
    return arms, order, fixed


def _order_search(s, acts, buckets, fl, pens, parks, h_inv, opts, leg_cache,
                  leg_cache_root, on_piece, dt, cluster, max_orders, verbose):
    """Try the stage's orders until one flies every ink bucket.

    -> (arms, order, rooms, rank, tried).

    THE GREEDY ORDER IS A HEURISTIC AND IT HAS A KNOWN WEAK SPOT: the arm that
    plans LAST has the least freedom left, so a stage can fail on its third arm
    while its first two fly.  Measured on CSAIL stage 0, that is exactly what
    happened -- 2 of 3, with 0.385 m stranded.

    A stage has at most three actives, so the whole order space is six
    permutations and searching it is cheap where searching a six-arm priority
    order is not (`idle._conduct`'s `sum_k P(n, k)` is 720 for six and 15 for
    three).  Ink-first is tried FIRST, so a stage that did not need the search
    pays one extra comparison and nothing else, and `rank` records which order
    was taken so a stage that needed a non-ink-first one says so.
    """
    import itertools
    base = priority_order(acts, buckets, s)
    cands = [base] + [o for o in itertools.permutations(base) if o != base]
    cands = cands[:max(1, int(max_orders))]
    best = None
    for rank, o in enumerate(cands):
        arms, _, rm = _sweep_in_order(s, o, buckets, fl, pens, parks, h_inv,
                                      opts, leg_cache, leg_cache_root,
                                      on_piece, dt, cluster, verbose)
        stranded = [st for st in arms.values()
                    if st.accepted and st.timeline is None]
        flown = float(sum(st.ink_m for st in arms.values()
                          if st.timeline is not None))
        if verbose:
            print(f"  stage {s} order {rank} {list(o)}: "
                  f"{flown:.3f} m flown, {len(stranded)} stranded")
        if not stranded:
            return arms, o, rm, rank, rank + 1
        if best is None or flown > best[3]:
            best = (arms, o, rm, flown, rank)
    arms, o, rm, _, rank = best
    return arms, o, rm, rank, len(cands)


def _plan_stage(s, acts, buckets, fl, pens, parks, h_inv, opts, leg_cache,
                leg_cache_root, fly, on_piece, env, ladders, verbose):
    return {a: plan_bucket(s, a, buckets.get((s, a), []), fl, pens, parks,
                           h_inv, opts, leg_cache=leg_cache,
                           leg_cache_root=leg_cache_root, fly=fly,
                           on_piece=on_piece, envelopes=env,
                           lift_ladder=(ladders or {}).get(a), verbose=verbose)
            for a in acts}


def _check_stage(sr, acts, fl, pens, parks, h_inv, dt, max_check_poses):
    thaw()              # the CHECK is not allowed to inherit the planner's room
    sr.pair = (active_pair_gap(sr.arms, fl, pens, h_inv, dt, max_check_poses)
               if len(sr.arms) > 1
               else dict(min_m=float("inf"), min_ink_m=float("inf"),
                         worst_at=None, per_pair={}, per_pair_ink={},
                         n_samples={}, n_ink={}))
    sr.solo = {int(a): solo_check(sr.arms[a], parks, fl, pens, h_inv,
                                  PAIR_MARGIN, dt) for a in acts}


def time_to_first_motion(buckets, want, specs, pens, parks, h_inv, opts,
                         leg_cache, leg_cache_root, dp_s: float) -> float:
    """Seconds from "the picture is in hand" to "the first arm may move".

    The DP, the first piece of the first stage's first arm, its entry hover and
    the park -> hover leg.  Nothing else gates the first motion: the rest of
    that arm's bucket, and every other arm, is planned behind the pens
    (docs/ARCHITECTURE_V2.md section 1.2).
    """
    first = None
    for s in want:
        cand = sorted(k for k in buckets if k[0] == s and buckets[k])
        if cand:
            first = cand[0]
            break
    if first is None:
        return float("nan")
    s, a = first
    spec = specs[a]
    t0 = time.perf_counter()
    freeze_partners(a, parks, specs, pens, h_inv, leg_cache, leg_cache_root)
    o = dict(opts or {})
    o.setdefault("h_inv", h_inv)
    o.setdefault("pen_ext", pens.get(a))
    pc = buckets[first][0]
    r = stroke_api.plan_stroke(pc.pts, spec, o)
    if r.get("status") != "ok":
        return float("nan")
    qs, pts = np.asarray(r["qs"], float), np.asarray(r["pts"], float)
    hov, z = writing.lifted_or_lower(spec, qs[0], pts[0], h_inv=h_inv,
                                     pen_ext=pens.get(a))
    leg = paper.route(spec, np.asarray(parks[a], float).reshape(7), hov,
                      pen_ext=pens.get(a), h_inv=h_inv,
                      tip_floor=paper.travel_floor(z, z))
    if leg is None:
        return float("nan")
    return float(dp_s + time.perf_counter() - t0)


def _report_stage(sr: StageResult) -> None:
    pm = sr.pair.get("min_m")
    sm = min((r.get("min_clearance", np.inf) for r in sr.solo.values()),
             default=np.inf)
    print(f"stage {sr.stage}: arms {list(sr.actives)}  "
          f"{sr.n_pieces} pieces  {sr.ink_m:.3f} m  "
          f"{sr.duration:.1f} s"
          + ("" if pm is None else
             f"  active-pair {1000 * pm:+.1f} mm "
             f"(ink {1000 * sr.pair['min_ink_m']:+.1f})")
          + (f"  solo {1000 * sm:+.1f} mm" if np.isfinite(sm) else "")
          + f"  [{sr.flown}/{sr.with_ink} flown]"
          + ("" if not sr.solo else
             f"  [{'PASS' if sr.ok else ('EMPTY' if sr.with_ink == 0 else 'FAIL')}]"))


# ---------------------------------------------------------------------------
# 6.  THE TYPED PROGRAMME
# ---------------------------------------------------------------------------
def programme(res: StagedResult, trajectories: bool = True) -> dict:
    """The staged programme, as the document item 5 has to absorb. -> dict.

    Per stage: the actives, and per arm its ordered pieces (each with the stage
    it belongs to, the piece's own polyline, the first and last joint vectors of
    its certified plan, and the ENTRY and EXIT HOVER CONFIGURATIONS that the
    pen-up legs fly between), the pen-up legs as the waypoint blocks
    `writing.arm_program` laid down, and the joint trajectory.  Plus the
    barrier list.

    `Segment` in `program_schema` carries no joint vector at all and
    `export_bundle` drops `q_first`/`q_last`, so this is a superset of that
    schema rather than an instance of it; folding the two together is build
    item 5 and it is a `SCHEMA_VERSION` bump, not a silent extension.
    """
    doc = dict(schema=int(STAGED_SCHEMA_VERSION), pattern=res.pattern,
               n_stages=len(res.stages), parks=res.parks,
               pair_margin_m=float(PAIR_MARGIN),
               makespan_s=float(res.makespan),
               ttfm_s=(None if res.ttfm_s is None else float(res.ttfm_s)),
               timing=dict(dp_s=float(res.dp_s), plan_s=float(res.plan_s),
                           check_s=float(res.check_s),
                           parallel_plan_s=float(res.parallel_plan_s),
                           wall_s=float(res.wall_s)),
               barriers=res.barriers(), stages=[])
    for sr in res.stages:
        one = dict(stage=int(sr.stage), actives=[int(a) for a in sr.actives],
                   duration_s=float(sr.duration), n_pieces=int(sr.n_pieces),
                   ink_m=float(sr.ink_m),
                   checks=dict(active_pair=_jsonable(sr.pair),
                               solo={str(a): dict(
                                   ok=bool(r.get("ok")),
                                   min_clearance_m=float(r.get("min_clearance",
                                                               np.nan)))
                                   for a, r in sr.solo.items()}),
                   arms={})
        for a, st in sr.arms.items():
            hov = st.hovers or [(None, None)] * len(st.programme)
            pieces = []
            for i, sg in enumerate(st.programme):
                qs = np.asarray(sg["plan"]["qs"], float)
                pieces.append(dict(
                    stage=int(sr.stage), arm=int(a), line=int(sg["stroke_id"]),
                    piece=int(sg.get("piece", i)), order=int(i),
                    flipped=bool(sg.get("flipped", False)),
                    home_before=bool(sg.get("home_before", False)),
                    length_m=float(sg["length"]),
                    q_first=[float(x) for x in qs[0]],
                    q_last=[float(x) for x in qs[-1]],
                    hover_in=(None if hov[i][0] is None
                              else [float(x) for x in np.asarray(hov[i][0]).ravel()]),
                    hover_out=(None if hov[i][1] is None
                               else [float(x) for x in np.asarray(hov[i][1]).ravel()]),
                    pts=np.round(np.asarray(sg["pts"], float), 5).tolist()))
            tl = st.timeline
            one["arms"][str(a)] = dict(
                arm=int(a), q_park=[float(x) for x in np.asarray(st.q_park).ravel()],
                frozen_partners=[int(x) for x in st.frozen_partners],
                room_kind=str(st.room_kind),
                priority=(None if st.priority is None else int(st.priority)),
                trajectory_digest=trajectory_digest(st),
                depends_on={str(k): str(v) for k, v in st.depends_on.items()},
                n_pieces=len(pieces), ink_m=float(st.ink_m),
                duration_s=float(st.duration),
                refused=[dict(line=int(p.piece.line), piece=int(p.piece.k),
                              length_m=float(p.piece.length_m),
                              status=p.status, reason=p.reason)
                         for p in st.refused],
                wall=dict(plan_s=float(st.plan_s), seq_s=float(st.seq_s),
                          prog_s=float(st.prog_s), total_s=float(st.wall_s)),
                legs=([dict(kind=ph["kind"], seg=int(ph["seg"]),
                            t0=float(ph["t0"]), t1=float(ph["t1"]))
                       for ph in tl["phases"] if ph["kind"] != "stroke"]
                      if tl else []),
                pieces=pieces,
                trajectory=(dict(t=np.round(tl["t"], 6).tolist(),
                                 q=np.round(tl["q"], 6).tolist(),
                                 seg=[int(x) for x in tl["seg"]])
                            if (tl and trajectories) else None))
        doc["stages"].append(one)
    return doc


def _jsonable(d: dict) -> dict:
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            out[k] = {str(kk): (float(vv) if isinstance(vv, (int, float))
                                else vv) for kk, vv in v.items()}
        elif isinstance(v, (int, float, np.floating)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def summary(res: StagedResult) -> dict:
    """The table docs/V2_STAGED.md quotes."""
    rows = []
    for sr in res.stages:
        rows.append(dict(
            stage=int(sr.stage), actives=[int(a) for a in sr.actives],
            pieces=int(sr.n_pieces), ink_m=round(sr.ink_m, 4),
            duration_s=round(sr.duration, 3),
            plan_s_per_arm={str(a): round(st.wall_s, 2)
                            for a, st in sr.arms.items()},
            plan_s_parallel=round(max((st.wall_s for st in sr.arms.values()),
                                      default=0.0), 2),
            plan_s_serial=round(sum(st.wall_s for st in sr.arms.values()), 2),
            refused=int(sum(len(st.refused) for st in sr.arms.values())),
            active_pair_mm=(None if not sr.pair else
                            round(1000 * sr.pair["min_m"], 2)),
            active_pair_ink_mm=(None if not sr.pair else
                                round(1000 * sr.pair["min_ink_m"], 2)),
            active_pair_at=(None if not sr.pair else sr.pair.get("worst_at")),
            solo_min_mm=(None if not sr.solo else round(1000 * min(
                r.get("min_clearance", np.nan) for r in sr.solo.values()), 2)),
            lift_used=bool(sr.lift_used),
            buckets=len(sr.arms),
            complete=bool(sr.complete),
            order=[int(a) for a in sr.order],
            order_rank=int(sr.order_rank),
            orders_tried=int(sr.orders_tried),
            buckets_flown=int(sum(1 for st in sr.arms.values()
                                  if st.timeline is not None)),
            buckets_with_ink=int(sum(1 for st in sr.arms.values()
                                     if st.accepted)),
            room_kind=next((st.room_kind for st in sr.arms.values()), "parked"),
            depends_on={str(a): st.depends_on
                        for a, st in sr.arms.items() if st.depends_on},
            ink_vs_envelope_mm=(None if not any(
                st.ink_clearance for st in sr.arms.values()) else round(
                1000 * min(min(st.ink_clearance) for st in sr.arms.values()
                           if st.ink_clearance), 2)),
            ok=bool(sr.ok)))
    n_pieces = len(res.pieces)
    refused = sum(len(st.refused) for sr in res.stages
                  for st in sr.arms.values())
    planned = sum(len(st.planned) for sr in res.stages
                  for st in sr.arms.values())
    return dict(pattern=res.pattern, stages=rows,
                n_pieces_dp=int(n_pieces), n_pieces_planned=int(planned),
                n_refused=int(refused),
                refused_frac=(refused / planned if planned else 0.0),
                makespan_s=round(res.makespan, 3),
                dp_s=round(res.dp_s, 4), plan_s=round(res.plan_s, 2),
                check_s=round(res.check_s, 2), wall_s=round(res.wall_s, 2),
                parallel_plan_s=round(res.parallel_plan_s, 2),
                ttfm_s=(None if res.ttfm_s is None else round(res.ttfm_s, 3)),
                envelope_s=round(res.envelope_s, 2),
                refusal_rounds=res.refusals,
                coverage=res.summary_dp.get("covered_frac"),
                drawn_m=res.summary_dp.get("drawn_m"),
                ink_m=res.summary_dp.get("ink_m"),
                gaps=res.summary_dp.get("gaps"),
                stage_overhead=stage_overhead(res),
                buckets_flown=int(sum(1 for s_ in res.stages
                                      for st in s_.arms.values()
                                      if st.timeline is not None)),
                buckets_total=int(sum(len(s_.arms) for s_ in res.stages)),
                buckets_with_ink=int(sum(1 for s_ in res.stages
                                         for st in s_.arms.values()
                                         if st.accepted)),
                all_ok=bool(all(s.ok for s in res.stages)))


def stage_overhead(res: StagedResult) -> dict:
    """What a stage costs BEFORE any ink: the park -> out -> back trip.

    A barrier is a rendezvous at the parks, so every active arm of every stage
    pays an entry leg out of its park and an exit leg back to it, whatever it
    has to draw in between.  At eight stages and three actives that is up to 24
    of them, and the interesting question at scale is how much of the makespan
    they still are once the buckets are large.
    """
    rows, sums = [], dict(park_s=0.0, draw_s=0.0, transit_s=0.0)
    crit = 0.0
    for sr in res.stages:
        worst, crit_arm = 0.0, 0.0
        for a, st in sorted(sr.arms.items()):
            tl = st.timeline
            if tl is None:
                continue
            ph = tl["phases"]
            strokes = [q for q in ph if q["kind"] == "stroke"]
            trans = [q for q in ph if q["kind"] == "transit"]
            entry = float(strokes[0]["t0"]) if strokes else 0.0
            exit_ = float(trans[-1]["t1"] - trans[-1]["t0"]) if trans else 0.0
            rows.append(dict(stage=int(sr.stage), arm=int(a),
                             pieces=len(strokes),
                             duration_s=round(float(tl["duration"]), 3),
                             draw_s=round(float(tl["draw_s"]), 3),
                             transit_s=round(float(tl["transit_s"]), 3),
                             entry_s=round(entry, 3), exit_s=round(exit_, 3),
                             park_overhead_s=round(entry + exit_, 3)))
            sums["park_s"] += entry + exit_
            sums["draw_s"] += float(tl["draw_s"])
            sums["transit_s"] += float(tl["transit_s"])
            if float(tl["duration"]) >= worst:
                worst = float(tl["duration"])
                crit_arm = entry + exit_
        crit += crit_arm
    mk = res.makespan or 1.0
    return dict(per_arm=rows,
                park_s_total=round(sums["park_s"], 2),
                park_s_on_the_critical_path=round(crit, 2),
                park_frac_of_makespan=round(crit / mk, 4),
                draw_s_total=round(sums["draw_s"], 2),
                transit_s_total=round(sums["transit_s"], 2))


def refusal_table(res: StagedResult) -> dict:
    """What refuses a piece the 2 cm atlas permitted. -> {reason: count}."""
    out: dict[str, int] = {}
    for sr in res.stages:
        for st in sr.arms.values():
            for p in st.refused:
                k = f"{p.status}:{p.reason}" if p.reason else p.status
                out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


# ---------------------------------------------------------------------------
# 7.  CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--atlas", default=ATLAS_DEFAULT)
    ap.add_argument("--lines", default=None,
                    help="a JSON file of polylines in paper metres")
    ap.add_argument("--synthetic", default=None, help="kind[:n]")
    ap.add_argument("--stages", default=None, help="e.g. '0,1' (default: all)")
    ap.add_argument("--no-fly", action="store_true",
                    help="plan the pieces and stop: no ordering, no legs, no "
                         "timeline and no checks (the refusal measurement)")
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--no-leg-cache", action="store_true")
    ap.add_argument("--leg-cache", default=None)
    ap.add_argument("--tilt-max-deg", type=float, default=0.0)
    ap.add_argument("--gate", choices=("strict", "flat"), default="strict")
    ap.add_argument("--dt", type=float, default=CHECK_DT)
    ap.add_argument("--no-envelopes", action="store_true",
                    help="do NOT put the other ACTIVE arms into the room as "
                         "their work-cell envelopes (the pre-2026-09-11 "
                         "behaviour, which leaves the legs uncertified against "
                         "each other)")
    ap.add_argument("--no-lift-retry", action="store_true")
    ap.add_argument("--trajectory-rooms", action="store_true",
                    help="build each active arm's room from the OTHER actives' "
                         "ACTUAL pass-1 trajectories rather than from their "
                         "pose-union work-cell envelopes (docs/V2_STAGED.md "
                         "section 15).  Certifies against a SPECIFIC plan, so "
                         "the dependency is recorded per arm")
    ap.add_argument("--room-iterations", type=int, default=1)
    ap.add_argument("--order-search", type=int, default=6,
                    help="orders a stage may try before keeping the best "
                         "(1 = ink-first only, the pre-search behaviour)")
    ap.add_argument("--room-order", choices=("priority", "simultaneous"),
                    default="priority",
                    help="'priority' (default): the actives choose in ink order "
                         "and arm k avoids the FINAL trajectories of 1..k-1, "
                         "which closes the fixed point in one sweep; "
                         "'simultaneous' is the superseded scheme that plans "
                         "everybody against everybody's previous pass")
    ap.add_argument("--refusal-rounds", type=int, default=REFUSAL_ROUNDS)
    ap.add_argument("--env-stride", type=int, default=ENVELOPE_STRIDE)
    ap.add_argument("--route-jobs", type=int, default=6,
                    help="processes the pen-up route screen may fork "
                         "(sequence.ROUTE_JOBS; 0 = the machine decides)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--programme", default=None)
    a = ap.parse_args(argv)

    if a.lines:
        lines = traces_mod.load_lines(a.lines)
        name = Path(a.lines).name
    elif a.synthetic:
        kind, _, n = a.synthetic.partition(":")
        lines = traces_mod.synthetic(kind, int(n or 1000))
        name = f"synthetic {kind} x{n or 1000}"
    else:
        raise SystemExit("one of --lines / --synthetic is required")
    cov = traces_mod.coverage_from_atlas(a.atlas, arms=tuple(sorted(FLEET)),
                                         gate=a.gate,
                                         tilt_max_deg=a.tilt_max_deg)
    stages = None if not a.stages else [int(x) for x in a.stages.split(",")]
    print(f"=== {name}: {len(lines)} lines, "
          f"{sum(float(traces_mod.cumlen(np.asarray(p, float))[-1]) for p in lines):.2f} m ===")
    res = run(lines, coverage=cov, opts=dict(tilt_max_deg=a.tilt_max_deg),
              stages=stages, leg_cache=not a.no_leg_cache,
              leg_cache_root=a.leg_cache, check=not a.no_check,
              fly=not a.no_fly, dt=a.dt, route_jobs=a.route_jobs,
              envelopes=not a.no_envelopes, atlas_dir=a.atlas,
              refusal_rounds=a.refusal_rounds,
              lift_retry=not a.no_lift_retry,
              env_kw=dict(stride=a.env_stride),
              trajectory_rooms=a.trajectory_rooms,
              room_iterations=a.room_iterations,
              room_order=a.room_order, order_search=a.order_search)
    d = summary(res)
    print(json.dumps(d, indent=1))
    print("refusals:", json.dumps(refusal_table(res)))
    if a.json:
        Path(a.json).write_text(json.dumps(
            dict(d, refusals=refusal_table(res), case=name), indent=1))
        print("wrote", a.json)
    if a.programme:
        Path(a.programme).write_text(json.dumps(programme(res)))
        print("wrote", a.programme)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
