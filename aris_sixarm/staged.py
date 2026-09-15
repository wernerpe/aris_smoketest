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

import dataclasses
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from . import (allocate, coordination, exact_room, frozen, paper,
               scene_check, stroke_api)
from . import traces as traces_mod
from . import writing
from .fleet import FLEET, H_INV_DEFAULT

# The programme this module writes is NOT `program_schema.Bundle`: that schema's
# `_from_dict` refuses unknown AND missing keys, so a stage id and a per-piece
# hover are a `SCHEMA_VERSION` bump rather than an extension, and folding the
# two together is build item 5.  Until then the staged programme carries its own
# version, and it is a superset of what item 5 has to absorb: stage id, barrier
# list, per-piece entry/exit hover configurations and per-piece q_first/q_last.
#
# 2 (2026-09-14) adds, for the leader/follower pattern: a per-arm `role`
# ("leader" | "follower" | "conductor" | "active"), a per-stage `conducted`
# flag, and a per-arm `deferred` list.  NOTHING WAS RENAMED OR REMOVED -- every
# field version 1 wrote is still written, in the same place and the same shape.
STAGED_SCHEMA_VERSION = 2

ATLAS_DEFAULT = "out/atlas_proposed_h0970_lat0860_gated63"

PAIR_MARGIN = coordination.PAIR_MARGIN      # 50 mm, the arm-to-arm gate
SWEEP_FRAC = 0.55                           # `scene_check.check_timeline`'s own
CHECK_DT = 0.05                             # s, the clock the checks sample on
MAX_CHECK_POSES = 900                       # per arm, per stage, for check (a)
LF_MAX_DROPS = 4            # pieces a bucket may shed before it is all deferred
# THE SHEET CHOICE IS A CHAIN, NOT A PIECE-AT-A-TIME DECISION (2026-09-14).
# `stroke_api.plan_stroke` picks each piece's (phi, q7, branch) sheet on its own
# merits, so adjacent pieces in a tour land on different sheets and the pen-up
# between them folds the arm over.  Once `sequence_arm` has fixed the order,
# `allocate.chain_sheets` re-picks the sheets along the tour with a Viterbi pass
# priced on the real capped transit time.  Every alternative it may choose is a
# `menu` variant materialised through `stroke_api` (same sigma, margin and
# validator certificate) AND put back through this module's ink-vs-room test.
CHAIN_SHEETS = True
CHAIN_K = allocate.CHAIN_K  # certified alternatives offered per piece
CONDUCT_CAP_S = 1200.0      # s of WALL CLOCK any one conduct may take
BAND_SHARE_MAX = 0.05       # dead-band ink above which it gets its own stage


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
                    leg_cache_root=None, standoff=None,
                    partners=None) -> tuple[int, ...]:
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

    `partners` NAMES WHICH ARMS ARE STANDING STILL, and it is not cosmetic.  In
    a two-arm conduct the OTHER MOVER is not a pose, it is a trajectory, and the
    conductor separates it in time; putting its entry pose in the room would
    route this arm's legs around a pose the arm leaves.  `None` keeps the
    shipped behaviour -- every other arm in `parks`.
    """
    fl = FLEET if specs is None else specs
    pens = {a: fl[a].pen for a in fl} if pens is None else pens
    keep = None if partners is None else {int(a) for a in partners}
    others = {int(a): np.asarray(q, float).reshape(7)
              for a, q in parks.items() if int(a) != int(arm) and a in fl
              and (keep is None or int(a) in keep)}
    frozen.freeze(others, fl, pens, h_inv)
    frozen.observe(int(arm))
    frozen.set_standoff(_standoff_for(int(arm), standoff, others))
    paper.clear_cache()                 # the memos are not keyed on the set
    if leg_cache:
        paper.disk_cache_open(leg_cache_root,
                              signature=leg_cache_signature(
                                  others, None, frozen.standoff()))
    else:
        paper.disk_cache_close()
    return tuple(sorted(others))


def freeze_conduct(partners, poses, specs=None, pens=None,
                   h_inv=H_INV_DEFAULT, rooms=None, leg_cache=True,
                   leg_cache_root=None, keep_bands=False) -> tuple[int, ...]:
    """The room a CONDUCT flies in, installed for the whole of it. -> the ids.

    THE DEFECT THIS CLOSES, MEASURED 2026-09-14 (docs/V2_STAGED.md §26).
    `idle.conduct` does not merely re-time the timelines `plan_bucket` built --
    it calls `writing.arm_program` again and ROUTES EVERY PEN-UP LEG AGAIN.  The
    row conductor thawed the frozen set before handing over, so every leg the
    conductor emitted was routed against the pose-invariant base columns and
    nothing else: the four arms outside the group were neither routed around
    (thawed) nor scheduled against (not in the conduct), and arm 31 flew through
    arm 2's standing chain at -204.6 mm with every gate in the stack satisfied.
    A six-arm conduct never showed it because there is no arm outside the group.

    So the arms this conduct does NOT move stay in the room for the length of
    it.  `poses` is what each of them holds; `rooms` names the ones that are
    themselves MOVING in this stage under an earlier priority group, and carries
    `trajectory_room`'s `(room, radii, digest)` for each -- a static swept
    obstacle that contains every pose that arm takes, which is the whole pair
    certificate and not a snapshot of it.

    No observer is named: the arms this conduct moves are not in the frozen set
    at all, so there is nothing for an observer to exclude, and `static_boxes`
    already drops a spec's own band.

    `keep_bands` keeps the frozen partners' band AABBs in the static room ON TOP
    of their real capsules -- `frozen.set_keep_bands` says why, and
    `CONDUCT_BANDS` says when.
    """
    fl = FLEET if specs is None else specs
    pens = {a: fl[a].pen for a in fl} if pens is None else pens
    sets = {int(a): np.asarray(poses[a], float).reshape(1, 7)
            for a in sorted(int(x) for x in partners) if a in fl}
    env = {int(a): v for a, v in (rooms or {}).items() if int(a) in sets}
    frozen.freeze_sets(sets, fl, pens, h_inv, clusters=env or None)
    frozen.observe(None)
    frozen.set_standoff(None)
    frozen.set_keep_bands(bool(keep_bands))
    paper.clear_cache()
    if leg_cache:
        paper.disk_cache_open(leg_cache_root,
                              signature=leg_cache_signature(
                                  {a: sets[a][0] for a in sets}, env or None,
                                  None, bool(keep_bands)))
    else:
        paper.disk_cache_close()
    return tuple(sorted(sets))


def _standoff_for(arm, standoff, partners) -> dict:
    """The standoff this arm actually carries. -> {aid: S}.

    A standoff names PARTNERS, and only ones that are in the room: an entry for
    the observer itself, or for an arm nobody froze, is dropped here so the
    memo and cache keys record what was really demanded.
    """
    return {int(a): float(s) for a, s in (standoff or {}).items()
            if float(s) > 0.0 and int(a) != int(arm) and int(a) in partners}


def leg_cache_signature(frozen_poses: dict[int, np.ndarray],
                        envelopes: dict | None = None,
                        standoff: dict | None = None,
                        keep_bands: bool = False) -> str:
    """`paper.cache_signature()` plus the frozen room it is valid under.

    The room is the parked partners' poses AND, in a stage, the active
    partners' envelopes — both change `static_boxes` without changing any memo
    key, so both have to be in the store's namespace or a leg bought in one
    room would be served in another.

    ...AND SO DOES THE PARTNER STANDOFF, for exactly the same reason: it makes
    `paper.route` refuse legs it used to grant without touching one memo key.
    It is written into the blob ONLY when it is non-empty, so every signature an
    S = 0 run builds is bit-identical to the one it built before the standoff
    existed and a warm leg store still answers.
    """
    d = dict(
        base=paper.cache_signature(),
        frozen={str(int(a)): [round(float(x), 9)
                              for x in np.asarray(q, float).ravel()]
                for a, q in sorted(frozen_poses.items())},
        envelopes={str(int(a)): list(room_sig(v))
                   for a, v in sorted((envelopes or {}).items())})
    if standoff:
        d["standoff"] = {str(int(a)): round(float(s), 9)
                         for a, s in sorted(standoff.items())
                         if float(s) > 0.0}
        if not d["standoff"]:
            d.pop("standoff")
    # ...AND THE KEPT BANDS, for the third time and the same reason: a leg
    # bought in the relaxed room may not be served in the room that keeps the
    # bands.  Absent when they are dropped, so an old signature is unchanged.
    if keep_bands:
        d["keep_bands"] = True
    blob = json.dumps(d, sort_keys=True)
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
                 h_inv=H_INV_DEFAULT, leg_cache=True, leg_cache_root=None,
                 standoff=None, partners=None, keep_bands=False):
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
    who = fl if partners is None else [int(a) for a in partners]
    sets = {int(a): np.asarray(parks[a], float).reshape(1, 7)
            for a in who if int(a) != int(arm)}
    env = {int(a): v for a, v in (envelopes or {}).items()
           if int(a) != int(arm) and int(a) in sets}
    frozen.freeze_sets(sets, fl, pens, h_inv, clusters=env)
    frozen.observe(int(arm))
    frozen.set_standoff(_standoff_for(int(arm), standoff, sets))
    # THE BANDS ARE A PROPERTY OF THE CONDUCT, NOT OF THE FREEZE.  `freeze_sets`
    # clears the flag, so a caller flying under `CONDUCT_BANDS = "keep"` has to
    # restate it here or its re-plan would silently route in the relaxed room
    # the retry exists to leave behind (docs/V2_STAGED.md §28.3).
    frozen.set_keep_bands(bool(keep_bands))
    paper.clear_cache()
    if leg_cache:
        paper.disk_cache_open(leg_cache_root,
                              signature=leg_cache_signature(
                                  {a: sets[a] for a in sets}, env,
                                  frozen.standoff(), bool(keep_bands)))
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


ROOM_STRIDE = 1            # timeline samples the exact room keeps out of each
INK_CAP = 0.30             # m, above which the ink gate stops resolving exactly


def room_mode():
    """Which obstacle a realised trajectory is modelled as. -> str.

    `ARIS_ROOM=spheres` reproduces every number earned before 2026-09-14
    exactly — `cluster_capsules` unchanged, `frozen` measuring degenerate
    capsules — and `capsules`, the default, is the leader's real swept chain.
    One flag, read in one place, so the A/B is one flag and not a branch per
    consumer.  An empty value is NOT SET, not off.
    """
    v = os.environ.get("ARIS_ROOM", "").strip().lower()
    return "spheres" if v in ("spheres", "sphere") else "capsules"


def room_sig(v) -> tuple:
    """What a cache has to key on to tell two rooms apart. -> tuple.

    `(kind, n primitives, position sum, radius sum)`, which is the shape both
    the plan memo and the leg store want and the only thing either of them ever
    knew about a room.  It exists because `v[0]` used to be an array of centres
    that `np.sum` would take, and an `ExactRoom` is not one.
    """
    if isinstance(v[0], exact_room.ExactRoom):
        return v[0].signature()
    return ("spheres", int(len(v[1])),
            round(float(np.sum(v[0])), 6), round(float(np.sum(v[1])), 6))


def room_size(v) -> str:
    """How big a room is, for a log line. -> "861 sph" | "23220 cap"."""
    return (f"{len(v[0])} cap" if isinstance(v[0], exact_room.ExactRoom)
            else f"{len(v[1])} sph")


def trajectory_room(st: "ArmStage", specs=None, pens=None, h_inv=H_INV_DEFAULT,
                    dt=CHECK_DT, cluster=ENVELOPE_CLUSTER, max_n=None,
                    mode=None, stride=None):
    """One arm's ACTUAL stage timeline, as the room a partner must avoid.
    -> (room, radii, digest).

    Under `ARIS_ROOM=capsules` (the default) `room` is an
    `exact_room.ExactRoom` holding the arm's REAL swept capsule chain, and
    `radii` its per-capsule radii; under `spheres` it is the shipped
    `cluster_capsules` reduction and `radii` the sphere radii.  The tuple shape
    is the same either way because every consumer downstream takes it apart the
    same way (`room_sig`, `room_size`, `freeze_stage`).

    THE PAD IS NOT A GUESS IN EITHER MODE.  It is
    `scene_check.check_timeline`'s own 1-Lipschitz between-sample residual, so
    the room covers the motion BETWEEN the samples and not only at them — the
    sphere form charges `SWEEP_FRAC x` the LARGEST step anywhere in the
    timeline to every sphere, the capsule form charges each sample the larger
    of its own two steps, which is the same bound applied where it is earned
    rather than globally.

    WHY THE SPHERES WERE NOT ENOUGH.  A sphere per 0.075 m cell is the cell's
    half-diagonal (65 mm) plus the biggest capsule radius that fell in it (up
    to 177 mm) plus the pad, and measured on 2026-09-14 that reduction costs a
    MEDIAN 195 mm of clearance against the capsules it contains: on follower
    arm 31 against leader 71, 0 % of the follower's ink clears the 50 mm gate
    against the spheres and 55.9 % clears it against the capsules.  The
    reduction, not the leader, was refusing the follower's ink.
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
    dig = trajectory_digest(st, dt)
    if (room_mode() if mode is None else str(mode)) == "capsules":
        rm = exact_room.from_samples(
            A3, B3, np.asarray(path.r, float)[keep], SWEEP_FRAC,
            stride=ROOM_STRIDE if stride is None else int(stride), digest=dig)
        return rm, rm.R, dig
    step = 0.0
    if len(A3) > 1:
        step = max(float(np.max(np.linalg.norm(np.diff(A3, axis=0), axis=2))),
                   float(np.max(np.linalg.norm(np.diff(B3, axis=0), axis=2))))
    R = np.tile(np.asarray(path.r, float)[keep], len(A3))
    c, r = cluster_capsules(A3.reshape(-1, 3), B3.reshape(-1, 3), R, cluster,
                            SWEEP_FRAC * step)
    return c, r, dig


def stage_rooms(arms: dict, specs=None, pens=None, h_inv=H_INV_DEFAULT,
                dt=CHECK_DT, cluster=ENVELOPE_CLUSTER) -> dict:
    """{arm: (room, radii, digest)} from one stage's ACTUAL timelines."""
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
    # CAPPED, THE WAY `coordination.BROAD_CAP` CAPS EVERY OTHER CLEARANCE IN
    # THIS REPO.  The exact room resolves a gap below the cap exactly and stops
    # scanning outward above it; the gate this feeds is 50 mm and the number is
    # only ever read to rank pieces by how deeply they are buried, so paying
    # for an exact 400 mm buys nothing.  Measured: 4.6 ms per pose uncapped,
    # 0.18 ms at this cap, same verdict.
    return float(np.min(frozen.partner_clearance(P, floor=INK_CAP)))


# ---------------------------------------------------------------------------
# 2b.  THE ROOM BOUNDARY, AS A CUT LINE
# ---------------------------------------------------------------------------
# `ink_vs_envelope` returns the MINIMUM over a piece's poses and the gate then
# refuses the piece WHOLE.  Measured on the CSAIL logo (docs/V2_STAGED.md
# section 23.3): 55.9 % of follower arm 31's ink poses clear the 50 mm gate
# against leader 71's exact room, and NONE of its pieces do, because every one
# of them contains a stretch that does not.  A pose-wise fraction only becomes
# flown ink if the piece is CUT where it enters the room -- which is the
# principle the whole trace DP is built on ("cut wherever the capability set
# changes"), applied to one more kind of transition.
SPLIT_MIN_M = traces_mod.MIN_PIECE_M    # a clear stretch shorter than this is
SPLIT_ROUNDS = 1                        # ...not a piece.  Rounds of re-cutting.
SPLIT_ID0 = 1000                        # piece ids a split part may be given
_SPLIT_ID = [SPLIT_ID0]


def split_ids_reset(start: int = SPLIT_ID0) -> None:
    """Restart the part-id counter.  Called once per `run`.

    A SPLIT PART NEEDS AN ID OF ITS OWN.  `Piece.k` is "which piece of that
    line" and `(line, piece)` is how the programme, the animation's duplicate
    check and every report identify ink; two halves of one cut piece sharing a
    `k` would read as the same ink drawn twice.  The counter is global and
    monotone, which is all uniqueness needs, and it starts at 1 000 so a part
    can never collide with a DP piece id (the CSAIL logo has 71 pieces in
    total, the 1 000-line synthetic 1 661).
    """
    _SPLIT_ID[0] = int(start)


def _next_split_id() -> int:
    _SPLIT_ID[0] += 1
    return _SPLIT_ID[0]


def ink_residual(P, frac: float = SWEEP_FRAC) -> np.ndarray:
    """A pose sequence's own 1-Lipschitz between-sample residual. -> (N,).

    THE ROOM ALREADY CARRIES THE LEADER'S SWEEP (`exact_room.from_samples` pads
    every capsule by `SWEEP_FRAC x` its own step); this is the other half, the
    FOLLOWER's.  A clear stretch is a claim about the motion between the ink
    samples and not only at them, so the pose-wise clearance has to be charged
    the follower's own travel the same way `scene_check.check_timeline` charges
    it.  `ink_vs_envelope`'s whole-piece minimum never needed it -- it is a
    number that is reported, and the gate it feeds refuses on the raw pose --
    so this is strictly stricter than the gate it refines.
    """
    P = np.asarray(P, float)
    return exact_room.sweep_pads(P, P, float(frac))


def ink_clearance_profile(plan: dict, spec, h_inv=H_INV_DEFAULT, pen=None,
                          frac: float = SWEEP_FRAC) -> np.ndarray:
    """Per-pose clearance of a certified piece's ink against the live room.
    -> (N,), the sweep residual ALREADY SUBTRACTED.

    `ink_vs_envelope` is `min` of this without the residual; a stretch of it is
    what a split is cut on.
    """
    if not frozen.active():
        return np.full(len(np.asarray(plan["qs"], float)), np.inf)
    P = coordination.chain_world(np.asarray(plan["qs"], float), spec, h_inv,
                                 float(spec.pen if pen is None else pen))
    return frozen.partner_clearance(P, floor=INK_CAP) - ink_residual(P, frac)


def clear_runs(d, cum, gate: float, min_len: float = SPLIT_MIN_M):
    """Contiguous sample runs that clear `gate`, at least `min_len` long.
    -> [(i0, i1)], inclusive index pairs into the pose/point arrays.

    `cum` is the cumulative arc length of the piece's dense points, which is
    the same length as `d` by `plan_stroke`'s contract (`qs` and `pts` are the
    same dense sampling).  A run shorter than the minimum piece length is
    dropped rather than kept: `traces.absorb_short` makes the same judgement
    about the DP's own runs, and a 3 mm stroke is a pen-down, a pen-up and no
    picture.
    """
    d = np.asarray(d, float)
    cum = np.asarray(cum, float)
    ok = d >= float(gate)
    out = []
    i = 0
    n = len(ok)
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and ok[j + 1]:
            j += 1
        if j > i and float(cum[j] - cum[i]) >= float(min_len):
            out.append((int(i), int(j)))
        i = j + 1
    return out


# A HOVER IS A POSE THE ARM HOLDS, AND A POSE IS JUDGED AT THE POSE GATE.
# `paper.FRAME_FLOOR` (63 mm) is `STATIC_MARGIN` plus the CHECKER's sweep
# residual and it is a ROUTING floor -- and `paper.effective_static_floor`
# already clamps it down to whatever a leg's own endpoints can hold, precisely
# so that a certified pose 51 mm from a neighbour is not stranded by a bar its
# own endpoint cannot meet.  Gating a new piece end at 63 mm would therefore
# refuse ends the router itself accepts: measured on CSAIL stage A
# (2026-09-14), every hover over follower 31's clear stretches reads +53.7 mm,
# which is over the arm-to-arm gate and under the routing floor, and at 63 mm
# the cut threw all four of them away.  The bar here is `PAIR_MARGIN`, which is
# `hold_gap`'s own bar for a held pose and the one `scene_check` re-derives.
HOVER_FLOOR = PAIR_MARGIN


def hover_clears(spec, q_ref, xy, h_inv=H_INV_DEFAULT, pen=None,
                 floor=None, boxes=None) -> bool:
    """Can the arm HOLD a certified hover over `xy` in the installed room?

    THE PIECE-END RULE, AND IT IS WHY A CUT IS NOT FREE.  Every piece end is a
    pen-up: `writing.arm_program` lifts to `writing.lifted_or_lower`'s hover
    and `paper.route` flies the leg from there.  A cut makes two new ends, and
    an end whose hover stands inside the leader's room is a leg the router will
    refuse -- which costs the WHOLE bucket its timeline, not just the piece.
    So the new ends are gated here, before the part is ever offered, against
    exactly the room the router will use and at the gate a held pose is judged
    at (see `HOVER_FLOOR`).
    """
    q = writing.lifted_or_lower(spec, np.asarray(q_ref, float).reshape(7),
                                np.asarray(xy, float).reshape(2), h_inv=h_inv,
                                pen_ext=pen)[0]
    if q is None:
        return False
    fl_ = HOVER_FLOOR if floor is None else float(floor)
    boxes = paper.static_boxes(spec) if boxes is None else boxes
    c = float(paper.chain_static(np.asarray(q, float).reshape(1, 7), spec,
                                 pen, h_inv, boxes)[0])
    return c >= fl_


HOVER_TRIES = 8             # inward steps each new end may take to find a hover


def _trim_to_hover(spec, qs, pts, cum, i0, i1, h_inv, pen, boxes, min_len,
                   tries=HOVER_TRIES):
    """Walk a clear run's ends inward until both can hold a hover.
    -> (i0, i1) or (None, None).

    WHERE A CUT LANDS INSIDE THE CLEAR RUN IS FREE -- every sample of it clears
    the gate -- but WHERE THE PIECE ENDS is not, because a piece end is a
    pen-up and `writing.arm_program` has to lift there.  So an end whose hover
    stands inside the room moves in, by a fraction of the run at a time, and
    the run is only given up when there is no end left that is long enough to
    be a piece.
    """
    n = max(1, (int(i1) - int(i0)) // max(1, int(tries)))
    a, b = int(i0), int(i1)
    for _ in range(int(tries) + 1):
        if float(cum[b] - cum[a]) < float(min_len):
            return None, None
        ok_a = hover_clears(spec, qs[a], pts[a], h_inv, pen, boxes=boxes)
        ok_b = hover_clears(spec, qs[b], pts[b], h_inv, pen, boxes=boxes)
        if ok_a and ok_b:
            return a, b
        if not ok_a:
            a += n
        if not ok_b:
            b -= n
    return None, None


def split_at_room(pc: "Piece", plan: dict, spec, h_inv=H_INV_DEFAULT, pen=None,
                  gate: float = PAIR_MARGIN, min_len: float = SPLIT_MIN_M,
                  hover_gate: bool = True):
    """A room-refused piece, re-cut at the room boundary. -> (keep, drop, info).

    `keep` are the CERTIFIED CLEAR stretches as new `Piece`s -- contiguous
    s-intervals whose every pose clears `gate` against the live frozen room
    with the sweep residual, each at least `min_len` long and each with a
    certified hover at both of its new ends.  `drop` is the complement, as
    `Piece`s too, so that what is deferred is the ink that was actually refused
    and not the whole line.  Their geometry comes from the plan's own dense
    points, so a part is a sub-polyline of exactly what the arm was going to
    draw.

    NOTHING HERE IS A CERTIFICATE.  The parts are re-planned through
    `plan_stroke` and re-gated by `ink_vs_envelope` like any other piece, and a
    part whose fresh plan resolves the redundancy differently and lands back
    inside the room is refused again and deferred.  This function chooses WHERE
    to cut; the gate stays the judge.
    """
    pts = np.asarray(plan["pts"], float)
    cum = traces_mod.cumlen(pts)
    L = float(cum[-1])
    info = dict(n_poses=int(len(pts)), clear_frac=0.0, n_keep=0, n_drop=0,
                hover_refused=0, keep_m=0.0, drop_m=0.0)
    if L <= 0 or len(pts) < 2:
        return [], [pc], info
    d = ink_clearance_profile(plan, spec, h_inv, pen)
    info["clear_frac"] = float(np.mean(d >= float(gate)))
    runs = clear_runs(d, cum, gate, min_len)
    lo, hi = float(pc.s0), float(pc.s1)

    def part(i0, i1):
        """[i0, i1] of the dense points, as a `Piece` of the same line."""
        q = pts[int(i0):int(i1) + 1]
        return dataclasses.replace(
            pc, k=_next_split_id(), pts=np.asarray(q, float),
            length_m=float(traces_mod.cumlen(q)[-1]),
            s0=lo + (hi - lo) * float(cum[i0]) / L,
            s1=lo + (hi - lo) * float(cum[i1]) / L)

    keep, edges = [], []
    boxes = paper.static_boxes(spec) if hover_gate else None
    qs = np.asarray(plan["qs"], float)
    for (i0, i1) in runs:
        if hover_gate:
            # THE CUT POINT IS FREE INSIDE THE CLEAR RUN, so an end whose hover
            # will not stand is walked INWARD rather than losing the stretch.
            # Measured on CSAIL stage A (2026-09-14): without this the one
            # clear stretch arm 31 had was thrown away for its ends alone.
            i0, i1 = _trim_to_hover(spec, qs, pts, cum, i0, i1, h_inv, pen,
                                    boxes, min_len)
            if i0 is None:
                info["hover_refused"] += 1
                continue
        keep.append(part(i0, i1))
        edges.append((i0, i1))
    if not keep:
        return [], [pc], info
    # THE COMPLEMENT, PIECE BY PIECE, so the deferral is the refused ink and
    # not the whole line.  A gap shorter than the minimum length is ink nobody
    # can draw as a piece; it is still deferred, because the next stage may
    # merge it back with a neighbour the DP re-cuts.
    drop, prev = [], 0
    for (i0, i1) in edges:
        if i0 - prev >= 1 and float(cum[i0] - cum[prev]) > 0:
            drop.append(part(prev, i0))
        prev = i1
    if prev < len(pts) - 1:
        drop.append(part(prev, len(pts) - 1))
    info.update(n_keep=len(keep), n_drop=len(drop),
                keep_m=float(sum(p.length_m for p in keep)),
                drop_m=float(sum(p.length_m for p in drop)))
    return keep, drop, info


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
    env = "".join(f"{int(a)}:{room_sig(v)};"
                  for a, v in sorted((envelopes or {}).items()))
    # THE STANDOFF IS PART OF THE ROOM for the same reason the envelope is: it
    # changes what `ink_vs_envelope` and the legs answer without changing an id.
    # Empty appends nothing, so an S = 0 key is the key it always was.
    # ...AND SO ARE THE KEPT BANDS: they change what every leg answers without
    # touching an id, a pose or an envelope.  Empty when they are dropped, so
    # the key the shipped behaviour builds is the key it always built.
    return (f"{tuple(frozen.frozen_ids())}|{frozen.observer()}|{env}"
            + (f"|S{frozen.standoff_sig()}" if frozen.standoff() else "")
            + (f"|B{frozen.keep_bands_sig()}" if frozen.keep_bands() else ""))


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
    residue: bool = False              # ran ALONE, after the others parked
    role: str = "active"               # active | leader | follower | conductor
    deferred: list = field(default_factory=list)   # [Piece] sent to the final pass
    conducted: bool = False            # its timeline came out of `idle.conduct`
    frozen_poses: dict[int, list[float]] = field(default_factory=dict)
    lift_ladder: tuple[float, ...] | None = None
    # WHERE THE ARM WENT BEFORE THE STAGE STARTED, and what that cost.  `None`
    # for every arm that did not pre-position -- a leader, or a follower whose
    # entry pose was already outside its leaders' rooms.  `q_park` remains the
    # pose the arm STARTED the stage at, which IS the tuck when there is one,
    # because that is what `plan_bucket` planned the tour from; `q_tuck` says so
    # explicitly and `clear_out_s` is the pre-move's seconds.
    q_tuck: np.ndarray | None = None
    clear_out_s: float = 0.0
    ink_clearance: list[float] = field(default_factory=list)
    ink_clear: dict = field(default_factory=dict)  # {piece.key: clearance_m}
    # WHAT THE ROOM-BOUNDARY CUT DID, per bucket: how many pieces were re-cut,
    # into how many parts, how much of their ink stayed and how much left.
    split: dict = field(default_factory=dict)
    # WHAT THE CONTINUITY PASS DID: pieces re-sheeted, transit seconds and
    # reconfiguration radians before and after.  {} when it did not run.
    chain: dict = field(default_factory=dict)
    # THE EXTRA CLEARANCE THIS BUCKET WAS HELD TO, per partner, on top of every
    # gate -- {} for every arm that carried none, which is every arm at S = 0.
    standoff: dict = field(default_factory=dict)
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

    @property
    def q_end(self) -> np.ndarray:
        """The pose the arm HOLDS when the stage ends. -> (7,).

        Its start pose if it never moved, the timeline's last sample otherwise.
        Under `PARK_HOME` that is the park by construction; under `PARK_FREEZE`
        it is the hover above the arm's last stroke, and it is the pose the NEXT
        stage starts from and the pose every other arm's next-stage room is
        built against.
        """
        if self.timeline is None:
            return np.asarray(self.q_park, float).reshape(7)
        return np.asarray(self.timeline["q"], float)[-1].reshape(7)


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
                park_policy=writing.PARK_HOME, standoff=None, partners=None,
                keep_bands=False, verbose=False) -> ArmStage:
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

    `park_policy` IS THE BARRIER, AND IT HAS TWO READINGS.  `PARK_HOME` sends
    the arm back to `q_park` at the end of the stage, which is what the zigzag's
    envelope argument needs: its guarantee is indexed by WHICH park the next
    stage's envelopes were certified against (docs/ARCHITECTURE_V2.md section
    2e).  `PARK_FREEZE` stops the arm at the hover above its last stroke and
    HOLDS it there, which is what a trajectory room needs and all it needs --
    the room contains the arm's whole path, its endpoint included, so a held
    pose is already inside the volume every neighbour was routed around.  Pete,
    2026-09-14, watching the v6 animation: "a lot of excessive parking ... we
    should be just executing that plan as efficiently as possible."  The park
    trip then survives only where it is load-bearing: at the very start of the
    programme and at the very end.
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
                                          h_inv, leg_cache, leg_cache_root,
                                          standoff, partners, keep_bands)
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
                                             leg_cache, leg_cache_root,
                                             standoff, partners)
        if keep_bands:
            # ...AND THE SAME RESTATEMENT `freeze_stage` makes, one door over:
            # `frozen.freeze` clears the flag, so the room a `CONDUCT_BANDS =
            # "keep"` re-plan flies in has to be re-asked for -- along with the
            # leg store's namespace, which the flag is part of.
            frozen.set_keep_bands(True)
            paper.clear_cache()
            if leg_cache:
                paper.disk_cache_open(leg_cache_root,
                                      signature=leg_cache_signature(
                                          frozen.poses(), None,
                                          frozen.standoff(), True))
    st.frozen_poses = {int(a): [float(x) for x in q]
                       for a, q in frozen.poses().items()}
    st.standoff = {int(a): float(s) for a, s in frozen.standoff().items()}
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
            # THE INK CLEARANCE RIDES ON THE MEMO, and it has to.  It is a
            # function of exactly the memo key (the arm, the geometry and the
            # room), and the drop-and-defer rule picks the piece that stands
            # CLOSEST to the room -- so a memo hit that dropped the number would
            # make that choice on whichever pieces happened to miss the cache.
            if len(hit) > 4 and hit[4] is not None:
                st.ink_clearance.append(float(hit[4]))
                st.ink_clear[pc.key] = float(hit[4])
            continue
        r = stroke_api.plan_stroke(pc.pts, spec, o)
        status = str(r.get("status"))
        reason = str(r.get("reason") or "")
        dist = None
        if status == "ok" and envelopes:
            # ...AND THE INK ITSELF HAS TO CLEAR THE OTHER ACTIVES.
            # `plan_stroke` never consults the static set, so a piece can be
            # certified and still be drawn through a neighbour's envelope.  A
            # piece that does is a REFUSAL like any other and goes back to the
            # DP with this (stage, arm) struck out of its capability set.
            d = ink_vs_envelope(r, spec, h_inv, pens.get(arm))
            dist = float(d)
            st.ink_clearance.append(float(d))
            st.ink_clear[pc.key] = float(d)
            if gate is not None and d < gate:
                status, reason = "refused", "ink_vs_active_envelope"
        # A ROOM REFUSAL KEEPS ITS PLAN, and nothing else does.  `split_at_room`
        # cuts the piece on the per-pose clearance of exactly this plan, so
        # throwing it away here would mean re-planning the piece to find out
        # where it entered the room.  `ok` is still False, so nothing flies it.
        keep = (r if (status == "ok" or reason == "ink_vs_active_envelope")
                else None)
        star = float(r.get("s_star", 0.0) or 0.0)
        _PLAN_MEMO[mk] = (status, reason, keep, star, dist)
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
    # THE SEQUENCER'S COST MODEL IS THE TIMELINE'S CLOCK OR IT OPTIMISED A
    # FICTION: it prices the go-home leg only if the timeline is going to fly
    # one, so the tour and the programme have to agree about the barrier.
    so = dict(h_inv=h_inv, pen_ext=pens.get(arm), q_start=q_park,
              return_home=(str(park_policy) == writing.PARK_HOME))
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

    # ...AND THE SHEETS, ALONG THE TOUR THE SEQUENCER JUST FIXED.  Every
    # alternative is materialised through `stroke_api` and then asked the SAME
    # ink-vs-room question the piece was accepted on, so a re-sheeted piece
    # carries the certificate of the one it replaced and never less.
    if CHAIN_SHEETS and len(st.programme) > 1:
        t0 = time.perf_counter()

        _base_clear: dict[int, float] = {}
        _prog0 = list(st.programme)

        def _accept(plan, i):
            if not envelopes:
                return True
            d = float(ink_vs_envelope(plan, spec, h_inv, pens.get(arm)))
            if gate is not None:
                return d >= gate
            # No gate means the ink check is a MEASUREMENT (see above), so the
            # rule is "no worse than the plan it replaces" and the bucket's
            # reported clearance stays true of what it ships.
            if i not in _base_clear:
                _base_clear[i] = float(ink_vs_envelope(
                    _prog0[i]["plan"], spec, h_inv, pens.get(arm)))
            return d >= _base_clear[i] - 1e-9
        try:
            st.programme, st.chain = allocate.chain_sheets(
                st.programme, spec, opts=o, seq_opts=so, k=int(CHAIN_K),
                accept=_accept, verbose=verbose)
        except Exception as exc:                # a tie-break, not a promise
            st.note = (st.note + "; " if st.note else "") + \
                f"chain_sheets skipped ({type(exc).__name__}: {exc})"
        st.seq_s += time.perf_counter() - t0

    t0 = time.perf_counter()
    try:
        st.timeline = writing.arm_program(
            spec, st.programme, h_inv=h_inv, pen_ext=pens.get(arm),
            q_start=q_park, park=str(park_policy),
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
    # THE LAST EXIT HOVER IS WHATEVER THE TRAJECTORY ACTUALLY ENDS ON.  The list
    # above re-derives every hover with the ORDINARY rule, and under
    # `PARK_FREEZE` `writing.arm_program` asks a DIFFERENT question of the one
    # pose the stage stops and holds: the strict joint margin and the room
    # (`HOVER_HOLD_MARGIN`, `HOVER_ROOM_FLOOR`).  The two answers differ -- 0.136
    # rad against 0.6023 on the pinned toy case -- and the trajectory is the
    # truth: it is what `scene_check` reads, what the barrier holds and what the
    # next stage plans from.  So the convenience field is taken FROM it rather
    # than re-derived beside it (docs/V2_STAGED.md section 28).
    if (st.hovers and st.timeline is not None
            and str(park_policy) == writing.PARK_FREEZE):
        st.hovers[-1] = (st.hovers[-1][0],
                         np.asarray(st.timeline["q"], float)[-1].reshape(7))
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


def hold_gap(poses: dict, specs=None, pens=None, h_inv=H_INV_DEFAULT) -> dict:
    """Is the HELD POSE SET at a barrier pairwise clear? -> dict(min_m, ...).

    THE BARRIER USED TO BE A PARK SET AND IT IS A HELD SET NOW (Pete,
    2026-09-14), so the thing that was true of the parks by construction --
    `Q_PARK_PROPOSED` was searched to be mutually clear -- has to be PROVED of
    the poses the arms actually stop in.  It is proved the same way and against
    the same gate: the real capsules of the six poses, `scene_check`'s own
    pair clearance, `PAIR_MARGIN`.

    It should never fail, and the reason it should never fail is worth writing
    down: an arm's trajectory room contains its whole path, its last sample
    included, and every later arm was routed clear of that room, so the held
    poses are separated by the same certificate that separated the motion.  This
    is the assertion that the reasoning held, not a new hope.

    ...AND A BARRIER MAY NOT HOLD A POSE THAT IS NOT CERTIFIABLE ON ITS OWN.
    Pairwise clearance is the half of the question that is about the FLEET; the
    other half is about each pose by itself, and it is the half that failed
    (`lf6_s150` stage B, arm 31 at 0.1064 rad of joint margin against
    `validate.MARGIN_GATE`'s 0.15 -- docs/V2_STAGED.md §29).  `validate_pose` is
    exactly the gate `scene_check` judges a held pose with, so it is asked here,
    where the barrier is actually declared, rather than discovered afterwards by
    the judge.  `writing.hold_candidates` is what an arm does about it.
    """
    fl = FLEET if specs is None else specs
    pens = {a: fl[a].pen for a in fl} if pens is None else pens
    arms = sorted(int(a) for a in poses)
    rr = scene_check._radii_for(fl, arms)
    P = {a: _chains(np.asarray(poses[a], float).reshape(1, 7), fl[a], h_inv,
                    scene_check.pen_len(pens.get(a), a)) for a in arms}
    per, worst = {}, float("inf")
    for n, ai in enumerate(arms):
        for aj in arms[n + 1:]:
            d = float(scene_check.pair_clearance(P[ai], P[aj], rr)[0])
            per[f"{ai}-{aj}"] = d
            worst = min(worst, d)
    # THE SAME READING `scene_check` TAKES, DOWN TO THE ONE EXEMPTION: a pen
    # resting ON the paper is reported and is not a hard violation there, so it
    # is not one here either, or the barrier would refuse a pose the judge
    # passes.
    pose_ok, margins, kinds = {}, {}, {}
    for a in arms:
        rep = scene_check.validate_pose(
            np.asarray(poses[a], float).reshape(7), fl[a], h_inv,
            scene_check.pen_len(pens.get(a), a))
        hard = [v["kind"] for v in rep["violations"]
                if v["kind"] != scene_check.PEN_PAPER]
        pose_ok[a] = not hard
        kinds[a] = hard
        margins[a] = float((rep.get("worst") or {}).get("joint_margin",
                                                        float("nan")))
    bad = sorted(a for a in arms if not pose_ok[a])
    return dict(min_m=float(worst), per_pair=per,
                poses_ok=bool(not bad), bad_poses=bad,
                bad_why={str(a): kinds[a] for a in bad},
                joint_margin={str(a): margins[a] for a in arms},
                ok=bool(worst >= PAIR_MARGIN and not bad))


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
    conducted: bool = False    # this stage went through `idle.conduct`
    conducted_check: dict = field(default_factory=dict)
    deferred_in: float = 0.0   # metres this stage took FROM an earlier one
    conduct_s: float = 0.0

    @property
    def roles(self) -> dict[int, str]:
        return {int(a): str(st.role) for a, st in sorted(self.arms.items())}

    @property
    def deferred_m(self) -> float:
        """Metres this stage could not fly and handed to the final pass."""
        return float(sum(p.length_m for st in self.arms.values()
                         for p in st.deferred))

    def role_ink(self, role: str, flown: bool = False) -> float:
        return float(sum(st.ink_m for st in self.arms.values()
                         if st.role == role
                         and (st.timeline is not None or not flown)))

    @property
    def duration(self) -> float:
        """The stage costs its BUSIEST CONCURRENT arm, plus any residue.

        The actives never wait for each other -- that is the whole point of the
        envelope argument -- so the concurrent part costs its busiest arm.  A
        RESIDUE bucket is not concurrent with anything: it runs after the others
        have parked, so it is added rather than maxed.
        """
        if self.conducted:
            # A CONDUCTED STAGE HAS ONE CLOCK AND THE MERGE IS WHERE IT IS SET.
            # `_merge_conducts` pads every arm to the same frame count and lays
            # a group the conductor REFUSED down one arm at a time INSIDE that
            # clock, so a residue arm's seconds are already in the maximum;
            # adding them again would bill the same wall twice (measured
            # 2026-09-14: 680.9 s reported for a 227.0 s timeline).
            return max((a.duration for a in self.arms.values()), default=0.0)
        conc = max((a.duration for a in self.arms.values() if not a.residue),
                   default=0.0)
        return conc + sum(a.duration for a in self.arms.values() if a.residue)

    @property
    def residue_m(self) -> float:
        return float(sum(a.ink_m for a in self.arms.values() if a.residue))

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
        if self.conducted:
            # A CONDUCTED STAGE HAS ONE CERTIFICATE AND IT IS THE WHOLE
            # TIMELINE.  `active_pair_gap` is the right question for arms that
            # never wait for each other and the WRONG one here: the conductor's
            # whole job is to choose when each arm is where, so the cross
            # product of two timelines is a set of instants the fleet never
            # holds.  `scene_check.check_timeline` on the conducted clock is
            # what v19 shipped on, and it is what this stage passes or fails on.
            return bool(self.complete
                        and self.conducted_check.get("ok", False))
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
    holds: list = field(default_factory=list)   # the HELD pose set per barrier
    partner_standoff: float = 0.0   # m demanded of a leader, above every gate

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

        A barrier is not a time, it is a STATE: every arm pen-up, stopped, with
        its queue drained and no un-cleared fault (docs/ARCHITECTURE_V2.md
        section 2e), in the pose the next stage was certified against -- and the
        pose IDENTITY is what the guarantee is indexed by.

        UNDER THE LEADER/FOLLOWER PATTERN THAT POSE IS NOT THE PARK.  An arm
        holds the hover above its last stroke and the next stage starts from
        there, so the barrier carries the HELD set (`holds`) and its pairwise
        proof alongside the parks, which are still what the programme starts and
        ends at and still the fault-recovery home.
        """
        out = []
        for k, s in enumerate(self.stages):
            one = dict(kind="stage", index=k, before_stage=s.stage,
                       actives=[int(a) for a in s.actives],
                       parks={str(a): q for a, q in self.parks.items()})
            if k < len(self.holds):
                h = self.holds[k]
                one["holds"] = h["q"]
                one["hold_min_mm"] = round(1000 * float(h["min_m"]), 2)
                one["hold_ok"] = bool(h["ok"])
            out.append(one)
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
        room_order="priority", order_search=6, residue=True,
        follower_ink_gate=PAIR_MARGIN, lf_max_drops=LF_MAX_DROPS, sub=2,
        tuck=True, split_rounds=SPLIT_ROUNDS, split_min_m=SPLIT_MIN_M,
        conduct_rows=True, conduct_cap_s=CONDUCT_CAP_S, conduct_jobs=3,
        row_compose="priority",
        partner_standoff=0.0, verbose=True, on_piece=None) -> StagedResult:
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
    roles = {s: pattern.roles(s) for s in want}
    conducted = {s for s in want if pattern.is_conducted(s)}
    # THE LEADER/FOLLOWER MODE IS DECLARED BY THE PATTERN, not by a flag.  A
    # pattern that names a leader and a follower is asking for the six-arm
    # stage, the role-ordered sweep and the conducted final pass, and asking for
    # them together: half of that scheme is not a scheme.
    lf = any(r in ("leader", "follower") for s in want for r in roles[s].values())
    for s in want:
        bad = same_row_pairs(stage_actives(pattern, s))
        if bad and not getattr(pattern, "same_row_ok", False):
            raise ValueError(f"stage {s} puts a same-ROW pair in the air "
                             f"together: {bad} -- see traces.zigzag_pattern")
    if lf:
        # The room IS the scheme here: a follower is certified against what its
        # leader ACTUALLY does, which is `trajectory_room` and nothing else.
        trajectory_rooms, envelopes, residue = True, False, False

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
    split_ids_reset()
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
    deferred: dict[int, list] = {}
    # THE HELD POSE SET, AND IT IS THE BARRIER NOW.  Stage A starts from the
    # parks; every stage after it starts from wherever the last one stopped, and
    # no arm flies home in between (Pete, 2026-09-14).  A barrier is still a
    # rendezvous -- everybody finished, pen up, stopped -- but it is no longer a
    # TRIP: the guarantee is indexed by the held pose set instead of the park
    # set, which is why the set is recorded per barrier and proved pairwise.
    held = {int(a): np.asarray(parks[a], float).reshape(7) for a in fl}
    holds: list[dict] = []
    for s in want:
        acts = stage_actives(pattern, s)
        if lf:
            t1 = time.perf_counter()
            entry = {int(a): np.asarray(q, float).reshape(7)
                     for a, q in held.items()}
            holds.append(dict(before_stage=int(s),
                              q={str(a): [float(x) for x in q]
                                 for a, q in sorted(entry.items())},
                              **hold_gap(entry, fl, pens, h_inv)))
            if s in conducted:
                took = {int(a): list(v) for a, v in deferred.items() if v}
                took_m = float(sum(p.length_m for v in took.values()
                                   for p in v))
                band: dict[int, list] = {}
                if conduct_rows:
                    # THE PARTITION IS A MEASUREMENT FIRST.  Row-safe ink goes
                    # to three two-arm conductors at once; dead-band ink cannot,
                    # and gets its own short stage rather than dragging every
                    # arm back into one 720-order search.
                    byrow, band, pstats = partition_deferred(took)
                    if verbose:
                        print(f"  stage {s} deferred {pstats['total_m']:.3f} m:"
                              + "".join(f"  row {j} {m:.3f} m"
                                        for j, m in sorted(
                                            pstats["row_m"].items()))
                              + f"  dead band {sum(pstats['band_m'].values()):.3f}"
                              f" m ({100 * pstats['band_frac']:.1f} %)")
                    rowdef = {a: v for r in byrow.values()
                              for a, v in r.items()}
                    arms, chk, c_s = _conduct_groups(
                        s, [ROW_ARMS[j] for j in sorted(ROW_ARMS)], buckets,
                        rowdef, fl, pens, entry, parks, h_inv, opts, leg_cache,
                        leg_cache_root, dt, sub, conduct_jobs, conduct_cap_s,
                        verbose, mode=row_compose)
                    chk["partition"] = {str(k): v for k, v in pstats.items()}
                else:
                    arms, chk, c_s = _conduct_stage(
                        s, buckets, took, fl, pens, entry, parks, h_inv, opts,
                        leg_cache, leg_cache_root, on_piece, dt, sub, verbose)
                sr = StageResult(int(s), tuple(sorted(int(a) for a in fl)),
                                 arms, wall_s=time.perf_counter() - t1,
                                 conducted=True, conducted_check=chk,
                                 deferred_in=took_m, conduct_s=float(c_s))
                deferred = {int(a): list(v) for a, v in band.items() if v}
            else:
                arms, order, rm, defer = _lf_stage(
                    s, acts, roles[s], buckets, fl, pens, entry, h_inv, opts,
                    leg_cache, leg_cache_root, on_piece, dt, ENVELOPE_CLUSTER,
                    follower_ink_gate, lf_max_drops, writing.PARK_FREEZE,
                    tuck, verbose, split_rounds, split_min_m,
                    partner_standoff)
                # A DEFERRAL GOES TO THE NEXT STAGE THIS ARM *LEADS*, and only
                # then to the conductor.  A follower keeps what fits and the
                # rest is its own remainder: in stage B the same arm has
                # priority, so the ink it could not take while yielding is ink
                # it can plan free.  That is Pete's literal version of the
                # pattern -- "the new leaders draw their remainder" -- and it is
                # what keeps the conducted final pass to what nothing else could
                # take.  Everything else waits for stage C.
                for a, v in defer.items():
                    nxt = next((t for t in want if t > s
                                and roles[t].get(int(a)) == "leader"), None)
                    if nxt is None:
                        deferred.setdefault(int(a), []).extend(v)
                        continue
                    buckets.setdefault((int(nxt), int(a)), []).extend(
                        dataclasses.replace(p, stage=int(nxt)) for p in v)
                sr = StageResult(int(s), acts, arms,
                                 wall_s=time.perf_counter() - t1,
                                 room_passes=[{int(a): str(v[2])
                                               for a, v in rm.items()}],
                                 order=tuple(order))
                if check and fly:
                    t2 = time.perf_counter()
                    _check_stage(sr, acts, fl, pens, entry, h_inv, dt,
                                 max_check_poses)
                    check_s += time.perf_counter() - t2
            for a, st in sr.arms.items():
                held[int(a)] = np.asarray(st.q_end, float).reshape(7)
            plan_s += time.perf_counter() - t1
            out.append(sr)
            if verbose:
                _report_stage(sr)
            continue
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
                if residue:
                    _residue_pass(s, arms, buckets, fl, pens, parks, h_inv,
                                  opts, leg_cache, leg_cache_root, on_piece,
                                  verbose)
            else:
                # THE SUPERSEDED SIMULTANEOUS SCHEME, kept because the
                # measurement that retired it is worth being able to reproduce:
                # on CSAIL stage 0 it is strictly worse than planning solo.
                for it in range(max(1, int(room_iterations))):
                    rm = stage_rooms(arms, fl, pens, h_inv, dt)
                    rooms.append({int(a): str(v[2]) for a, v in rm.items()})
                    if verbose:
                        print(f"  stage {s} room pass {it + 1}: "
                              + "  ".join(f"{a}={room_size(v)}" for a, v in
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
    # ---- STAGE D: THE DEAD-BAND INK, WHICH NO ROW CONDUCTOR MAY TAKE --------
    # A piece that straddles a row's dead band puts its arm where the cross-row
    # measurement does not reach, so it cannot ride with the three row
    # conductors.  Two forms, and the share chooses: a small residue is one
    # conduct over whoever owns it, and a large one is Pete's own sequence --
    # the OUTER rows draw their band pieces while the middle row holds, then
    # the middle row draws its own.
    if lf and fly and deferred and conduct_rows \
            and any(sr.conducted for sr in out):
        band_m = float(sum(p.length_m for v in deferred.values() for p in v))
        who = sorted(int(a) for a, v in deferred.items() if v)
        outer = tuple(a for a in who if traces_mod.ROW_OF.get(a) != 1)
        mid = tuple(a for a in who if traces_mod.ROW_OF.get(a) == 1)
        share = band_m / max(band_m + sum(sr.ink_m for sr in out
                                          if sr.conducted), 1e-12)
        phases = ([tuple(who)] if (share < BAND_SHARE_MAX
                                   or not (outer and mid))
                  else [outer, mid])
        if verbose:
            print(f"  stage D: {band_m:.3f} m of dead-band ink "
                  f"({100 * share:.1f} % of the final pass) in "
                  f"{len(phases)} phase(s): "
                  + " then ".join(str(sorted(ph)) for ph in phases))
        for ph in phases:
            sd = max(sr.stage for sr in out) + 1
            t1 = time.perf_counter()
            entry = {int(a): np.asarray(q, float).reshape(7)
                     for a, q in held.items()}
            holds.append(dict(before_stage=int(sd),
                              q={str(a): [float(x) for x in q]
                                 for a, q in sorted(entry.items())},
                              **hold_gap(entry, fl, pens, h_inv)))
            take = {int(a): list(v) for a, v in deferred.items()
                    if int(a) in set(ph) and v}
            arms, chk, c_s = _conduct_groups(
                sd, [tuple(ph)], {}, take, fl, pens, entry, parks, h_inv, opts,
                leg_cache, leg_cache_root, dt, sub, 1, conduct_cap_s, verbose)
            sr = StageResult(int(sd), tuple(sorted(int(a) for a in fl)), arms,
                             wall_s=time.perf_counter() - t1, conducted=True,
                             conducted_check=chk,
                             deferred_in=float(sum(p.length_m for v in
                                                   take.values() for p in v)),
                             conduct_s=float(c_s))
            for a in take:
                deferred.pop(int(a), None)
            for a, st in sr.arms.items():
                held[int(a)] = np.asarray(st.q_end, float).reshape(7)
            plan_s += time.perf_counter() - t1
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
                       summary_dp=tplan.summary(), holds=holds,
                       partner_standoff=float(partner_standoff or 0.0))
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
                  f"-> {room_size(fixed[int(a)])}"
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


def _residue_pass(s, arms, buckets, fl, pens, parks, h_inv, opts, leg_cache,
                  leg_cache_root, on_piece, verbose):
    """Ink that no order could fly CONCURRENTLY is flown ALONE, at the end.

    -> the metres that took this path.

    PETE'S ORIGINAL FINAL PASS, and the honest floor under the whole scheme: a
    bucket that cannot share the stage with its neighbours does not have to be
    abandoned, it has to be SERIALISED.  The other actives have finished and are
    back at their parks by then -- that is what the stage barrier means -- so
    the residue arm plans against exactly the room pass 1 flies in, the parked
    fleet, and its timeline is appended to the stage rather than overlapped with
    anybody's.

    `idle.conduct` IS the tool for this and it reduces to nothing here: with one
    arm moving the priority search enumerates `sum_k P(1, k)` = ONE order and
    there is no second mover to schedule against.  So the residue is laid down
    directly and checked the same way -- `solo_check` against the parked fleet
    is precisely the certificate a one-mover conduct would produce, and
    `active_pair_gap` does not see it at all, because nothing else is moving.
    """
    done = 0.0
    for a, st in sorted(arms.items()):
        if not st.accepted or st.timeline is not None:
            continue
        if verbose:
            print(f"  stage {s} RESIDUE: arm {a} ({len(st.accepted)} pieces, "
                  f"{st.ink_m:.3f} m) alone, after the others park")
        alone = plan_bucket(s, a, buckets.get((s, a), []), fl, pens, parks,
                            h_inv, opts, leg_cache=leg_cache,
                            leg_cache_root=leg_cache_root, fly=True,
                            on_piece=on_piece, envelopes=None, verbose=verbose)
        alone.residue = True
        alone.priority = st.priority
        arms[int(a)] = alone
        if alone.timeline is not None:
            done += float(alone.ink_m)
    return done


# ---------------------------------------------------------------------------
# 5b.  THE LEADER/FOLLOWER STAGE, AND THE CONDUCTED FINAL PASS
# ---------------------------------------------------------------------------
# PETE'S SPECIFICATION, 2026-09-14 (see `traces.leader_follower_pattern` for the
# pattern and for why the zigzag's premise was wrong).  Two things differ from
# `_priority_stage` and they are the whole of the difference:
#
#   1. ALL SIX ARMS MOVE.  The priority order is by ROLE first -- the three
#      leaders, busiest first, then the three followers, busiest first -- rather
#      than by ink alone, because a follower is defined as the arm that yields
#      and a leader as the one that does not.  Everything else is the existing
#      sweep: arm k plans against the FINAL trajectory rooms of arms 1..k-1, so
#      every pair is certified against a path that never moves again.
#
#   2. A PIECE THAT DOES NOT FIT IS DEFERRED, NOT SERIALISED.  The residue pass
#      (docs/V2_STAGED.md section 20.3) flies a stranded bucket ALONE after the
#      others park, which is correct and is exactly what this pattern exists to
#      avoid: serialising inside a stage gives back the concurrency the six-arm
#      stage was built to buy.  Here the piece leaves the stage altogether and
#      goes to the conducted final pass, where all six arms are available and a
#      shared clock -- not a static keep-out -- carries the safety argument.


# ---------------------------------------------------------------------------
# 5b-i.  THE FOLLOWER'S PRE-POSITION
# ---------------------------------------------------------------------------
# THE PARK IS INSIDE THE ROOM, and that is where stage A actually died.  Measured
# 2026-09-14 on arm 31 (docs/V2_STAGED.md section 22.3): the follower's own start
# pose reads -128.4 mm against its same-row leader's room before it has moved a
# joint.  `paper.effective_static_floor` then clamps EVERY leg's static floor to
# that negative number -- correctly, because a gate no endpoint can meet refuses
# ink the arm can draw -- so the router is asked to fly out of a hole it is
# already in, and the entry legs fail first.  No amount of routing fixes a start
# pose; the arm has to be somewhere else before the leader starts.
#
# SO IT MOVES FIRST.  The follower flies park -> TUCK while the leader is still
# parked -- a few seconds, gated against the parked fleet exactly like any other
# leg -- and the leader starts after that.  The tuck is chosen OUTSIDE the
# leader's whole trajectory room by the routing floor, which is a stronger claim
# than "clear at each instant": it holds for the entire stage whatever the
# leader is doing, so the follower may stand there for the whole tour and the
# pair is separated by construction.  The cost is the clear-out's seconds on the
# time to first motion, and it is reported.
TUCK_HEIGHTS = (writing.LIFT_Z, 0.09, 0.12, 0.15, 0.20)
TUCK_XY_PER_PIECE = 3        # ink points per piece offered as a tuck station
TUCK_MARGIN = 0.010          # m ABOVE the routing floor a tuck must stand clear
TUCK_MAX_S = 10.0            # s the clear-out may cost the time to first motion


def _ink_xy(pieces, per=TUCK_XY_PER_PIECE):
    """Paper points spread over a bucket's ink. -> [(x, y)].

    The tuck wants to be near the arm's OWN ink -- it is a start line, not a
    parking bay -- so the candidate stations are the ink itself: the ends of
    each piece and a point or two along it.
    """
    out = []
    for p in pieces:
        P = np.asarray(p.pts, float).reshape(-1, 2)
        if not len(P):
            continue
        idx = np.unique(np.linspace(0, len(P) - 1, max(2, int(per))).astype(int))
        out += [(float(P[i, 0]), float(P[i, 1])) for i in idx]
    return out


def tuck_pose(spec, pieces, q_from, h_inv=H_INV_DEFAULT, pen_ext=None,
              floor=None, heights=TUCK_HEIGHTS, margin=TUCK_MARGIN):
    """Where a follower should stand while its leader draws. -> (q, info).

    `(None, info)` when nothing clears.  THE ROOM MUST ALREADY BE INSTALLED:
    this reads `paper.static_boxes` and `frozen.chain_clearance` through the
    ordinary accessors, so it measures against exactly what the router will,
    the leaders' exact rooms included.

    The candidates are hovers over the arm's own ink at a ladder of heights
    (`writing.hover_solve`, the same solver `paper._traverse` lifts with), plus
    the pose the arm is already standing in -- because a follower whose park is
    ALREADY outside the room should not move at all, and this is where that is
    decided.  A candidate has to clear the whole installed room by the routing
    floor plus `margin`, and the winner is the one whose pen tip is nearest the
    ink it is about to draw.
    """
    fl_ = paper.FRAME_FLOOR if floor is None else float(floor)
    need = fl_ + float(margin)
    boxes = paper.static_boxes(spec)
    xys = _ink_xy(pieces)
    info = dict(need_mm=1000 * need, n_xy=len(xys), tried=0, kept=0,
                park_mm=None, best_mm=None, moved=False)
    q_from = np.asarray(q_from, float).reshape(7)
    cen = (np.mean(np.asarray(xys, float), axis=0) if xys
           else paper.tip_xy(q_from, spec, pen_ext, h_inv))

    def score(q):
        """(clears?, clearance, distance from the ink centroid)."""
        q = np.asarray(q, float).reshape(7)
        c = float(paper.chain_static(q[None], spec, pen_ext, h_inv, boxes)[0])
        d = float(np.linalg.norm(paper.tip_xy(q, spec, pen_ext, h_inv) - cen))
        return c, d

    c0, d0 = score(q_from)
    info["park_mm"] = 1000 * c0
    best, best_key, best_c = None, None, None
    if c0 >= need:
        # ALREADY OUTSIDE.  Nothing to do, and nothing is charged for it.
        info.update(tried=1, kept=1, best_mm=1000 * c0, moved=False)
        return q_from, info
    for z in heights:
        for xy in xys:
            q = writing.hover_solve(spec, q_from, xy, z=z, h_inv=h_inv,
                                    pen_ext=pen_ext)
            info["tried"] += 1
            if q is None:
                continue
            c, d = score(q)
            if c < need:
                continue
            info["kept"] += 1
            key = (d, -c)
            if best_key is None or key < best_key:
                best, best_key, best_c = np.asarray(q, float).reshape(7), key, c
    if best is not None:
        info.update(best_mm=1000 * best_c, moved=True)
    return best, info


def splice_timeline(pre, main):
    """Lay `main` down after `pre` on one clock. -> timeline dict.

    `writing.arm_program` builds ONE pass from ONE start pose and there is no
    other way in; a pre-move is therefore a second programme, and the two have
    to become one object or nothing downstream sees the first.  That matters
    for more than tidiness: `trajectory_room`, `active_pair_gap` and
    `scene_check` all read the timeline, so a clear-out that is not in it is a
    few seconds of six-arm motion that no gate ever looked at.

    The junction sample is dropped rather than repeated -- `pre` ends at the
    tuck and `main` starts at it, the same pose -- and every phase, ink chunk
    and second of `main` is shifted by `pre`'s duration.  Monotonicity is
    re-established the way `arm_program` establishes it in the first place.
    """
    if pre is None or float(pre.get("duration", 0.0)) <= 0.0:
        return main
    if main is None:
        return pre
    off = float(pre["duration"])
    t = np.concatenate([np.asarray(pre["t"], float),
                        np.asarray(main["t"], float)[1:] + off])
    t = np.maximum.accumulate(t + 1e-9 * np.arange(len(t)))
    out = dict(main)
    out["t"] = t
    for k in ("q", "seg", "u"):
        out[k] = np.concatenate([np.asarray(pre[k]), np.asarray(main[k])[1:]])
    out["phases"] = list(pre.get("phases", [])) + [
        dict(p, t0=float(p["t0"]) + off, t1=float(p["t1"]) + off)
        for p in main.get("phases", [])]
    out["ink"] = list(pre.get("ink", [])) + [
        (float(ti) + off, ch) for ti, ch in main.get("ink", [])]
    out["lifts"] = list(pre.get("lifts", [])) + list(main.get("lifts", []))
    out["paper_modes"] = (list(pre.get("paper_modes", []))
                          + list(main.get("paper_modes", [])))
    for k in ("transit_s", "taxi_s", "retreat_s", "aside_s", "draw_s",
              "draw_len", "transit_len", "paper_vias", "fallbacks", "n_home"):
        out[k] = pre.get(k, 0) + main.get(k, 0)
    out["duration"] = float(t[-1])
    out["dense_tip_err"] = max(float(pre.get("dense_tip_err", 0.0)),
                               float(main.get("dense_tip_err", 0.0)))
    return out


def clear_out(spec, q_from, q_tuck, h_inv=H_INV_DEFAULT, pen_ext=None):
    """The pre-move park -> tuck, as a certified timeline. -> dict | None.

    `writing.arm_program`'s `aside` path, which is the one place in this repo
    that lays a standalone routed move down on the real clock: it routes with
    `paper.route` at the flying floor and raises rather than teleporting.  IT
    IS PLANNED IN WHATEVER ROOM IS INSTALLED WHEN IT IS CALLED, and the caller
    installs the fleet with every leader still PARKED -- which is true, because
    this is the move that happens before any leader starts.
    """
    if float(np.max(np.abs(np.asarray(q_from, float).reshape(7)
                           - np.asarray(q_tuck, float).reshape(7)))) <= 1e-9:
        return None
    try:
        return writing.arm_program(spec, [], h_inv=h_inv, pen_ext=pen_ext,
                                   q_start=np.asarray(q_from, float).reshape(7),
                                   park=writing.PARK_FREEZE,
                                   aside=np.asarray(q_tuck, float).reshape(7))
    except writing.PaperRefused:
        return None


def lf_standoffs(actives, roles: dict, standoff_m: float) -> dict:
    """Who owes an extra standoff to whom, in one stage. -> {leader: {foll: S}}.

    ONLY A LEADER OWES IT, AND ONLY TO ITS SAME-ROW FOLLOWER.  The leader is the
    arm that plans first and is certified at exactly `PAIR_MARGIN` against the
    partner's held pose, so it is the arm whose plan eats the room the follower
    then has to route its legs in; and the pair that has no room to give is the
    same-ROW one, because a cross-row pair clears by +194 mm with each row
    inside its own band (docs/V2_WORKCELLS.md section 4b).  Charging a cross-row
    partner would cost a leader ink and buy a follower nothing.

    `standoff_m` = 0 returns {}, which is what makes S = 0 a no-op all the way
    down: no standoff is set, no cache signature changes, no key moves.
    """
    S = float(standoff_m or 0.0)
    if S <= 0.0:
        return {}
    foll = {int(a) for a in actives
            if str(roles.get(int(a), "active")) == "follower"}
    out = {}
    for a in (int(x) for x in actives):
        if str(roles.get(a, "active")) != "leader":
            continue
        row = traces_mod.ROW_OF.get(a)
        mates = {p: S for p in sorted(foll)
                 if p != a and row is not None
                 and traces_mod.ROW_OF.get(p) == row}
        if mates:
            out[a] = mates
    return out


def role_order(roles: dict, actives, buckets, stage) -> tuple[int, ...]:
    """The order the six actives of a main stage choose their paths in.

    ROLE FIRST, INK SECOND.  `priority_order` ranks by ink because the busiest
    arm has the least room to give; that is still the tie-break, but it is a
    tie-break WITHIN a role now.  A follower that happened to carry more ink
    than a leader must still plan after it, or "leader" and "follower" would be
    labels on an order nobody enforced.
    """
    rank = {"leader": 0, "active": 1, "follower": 2, "conductor": 3}

    def ink(a):
        return sum(p.length_m for p in buckets.get((int(stage), int(a)), ()))
    return tuple(sorted((int(a) for a in actives),
                        key=lambda a: (rank.get(roles.get(int(a), "active"), 1),
                                       -ink(a), a)))


def _fly_or_defer(s, a, pieces, fl, pens, parks, h_inv, opts, leg_cache,
                  leg_cache_root, on_piece, envelopes, gate, max_drops,
                  park_policy, verbose, split_rounds=SPLIT_ROUNDS,
                  split_min_m=SPLIT_MIN_M, standoff=None):
    """Plan one bucket in the room it is given, deferring what will not fit.

    -> (ArmStage, [Piece] deferred).

    TWO WAYS A PIECE FAILS THE ROOM AND BOTH END IN THE SAME PLACE.  Its INK can
    pass through the room -- `plan_stroke` never consults the static set, so a
    piece can be certified end to end and still be drawn through the leader's
    trajectory -- and `ink_vs_envelope` catches that per piece, at the gate.  Or
    its pen-up LEG can be unroutable, which `paper.route` reports by refusing
    and `writing.arm_program` by raising, and which is a statement about the
    bucket rather than about any one piece.  The first is a refusal and is
    exact.  The second is answered by dropping the piece that stands CLOSEST to
    the room and asking again, up to `max_drops` times, because the leg that
    cannot be flown is the leg into or out of the piece that is buried deepest
    in somebody else's trajectory.  A bucket that still will not fly is deferred
    whole.

    AND A PIECE THAT FAILS THE INK GATE IS CUT BEFORE IT IS DEFERRED.  The gate
    is a minimum over the piece's poses, so a piece that is 56 % clear is
    refused entire; `split_at_room` re-cuts it at the room boundary and the
    clear stretches come back through this same loop as ordinary pieces, to be
    re-planned and re-gated like any other.  The certificate is unchanged --
    every part that flies passed `ink_vs_envelope` on its own plan -- and what
    is deferred is now the refused ink rather than the line it was part of.
    """
    keep = list(pieces)
    out: dict = {}          # piece.key -> Piece, deduplicated by construction
    tally = dict(parents=0, parts=0, kept_m=0.0, dropped_m=0.0,
                 hover_refused=0, rounds=0)

    def plan(ps):
        return plan_bucket(s, a, ps, fl, pens, parks, h_inv, opts,
                           leg_cache=leg_cache, leg_cache_root=leg_cache_root,
                           fly=True, on_piece=on_piece, envelopes=envelopes,
                           ink_gate=gate, park_policy=park_policy,
                           standoff=standoff, verbose=False)

    st = plan(keep)
    # THE CUT, BEFORE ANYTHING IS DEFERRED.  Only where there is a gate to fail
    # and a room to fail it against; a leader plans free and has neither.
    for _ in range(int(split_rounds) if (gate is not None and envelopes) else 0):
        cut, kept_now, done = [], [], True
        for pp in st.refused:
            if pp.reason != "ink_vs_active_envelope" or pp.plan is None:
                continue
            kp, dp, info = split_at_room(pp.piece, pp.plan, fl[int(a)], h_inv,
                                         pens.get(int(a)), float(gate),
                                         float(split_min_m))
            tally["hover_refused"] += int(info["hover_refused"])
            if not kp:
                continue
            done = False
            tally["parents"] += 1
            tally["parts"] += len(kp)
            cut.append(pp.piece.key)
            kept_now += kp
            for p in dp:
                out[p.key] = p
            if verbose:
                print(f"  stage {s} arm {a}: line {pp.piece.line} piece "
                      f"{pp.piece.k} ({pp.piece.length_m:.3f} m) cut at the "
                      f"room boundary -> {len(kp)} clear part(s) "
                      f"{info['keep_m']:.3f} m, {len(dp)} deferred "
                      f"{info['drop_m']:.3f} m "
                      f"({100 * info['clear_frac']:.1f} % of poses clear)")
        if done:
            break
        tally["rounds"] += 1
        keep = [p for p in keep if p.key not in set(cut)] + kept_now
        st = plan(keep)
    st.split = dict(tally)
    for pp in st.refused:
        if pp.reason == "ink_vs_active_envelope":
            out[pp.piece.key] = pp.piece
    drops = 0
    while st.timeline is None and st.accepted and drops < int(max_drops):
        acc = [p.piece for p in st.accepted]
        worst = min(acc, key=lambda p: st.ink_clear.get(p.key, float("inf")))
        out[worst.key] = worst
        keep = [p for p in keep if p.key != worst.key]
        drops += 1
        if verbose:
            print(f"  stage {s} arm {a}: bucket will not fly; deferring "
                  f"line {worst.line} piece {worst.k} "
                  f"({worst.length_m:.3f} m) and retrying ({drops})")
        st = plan(keep)
        for pp in st.refused:
            if pp.reason == "ink_vs_active_envelope":
                out[pp.piece.key] = pp.piece
    if st.timeline is None and st.accepted:
        for pp in st.accepted:
            out[pp.piece.key] = pp.piece
        st = plan([])
    st.split = dict(tally)
    st.split["kept_m"] = float(sum(p.piece.length_m for p in st.accepted
                                   if p.piece.k >= SPLIT_ID0))
    st.split["dropped_m"] = float(sum(p.length_m for p in out.values()
                                      if p.k >= SPLIT_ID0))
    st.deferred = [out[k] for k in sorted(out)]
    return st, st.deferred


def _lf_stage(s, actives, roles, buckets, fl, pens, parks, h_inv, opts,
              leg_cache, leg_cache_root, on_piece, dt, cluster, gate,
              max_drops, park_policy, tuck, verbose,
              split_rounds=SPLIT_ROUNDS, split_min_m=SPLIT_MIN_M,
              partner_standoff=0.0):
    """One six-arm main stage of the leader/follower pattern.

    -> (arms, order, rooms, {arm: [Piece] deferred}).

    `parks` here is the HELD POSE SET the stage begins at, not the shipped
    parks: the arm's own entry is its start pose, and every partner that has not
    moved yet in this stage is in the static room as the pose it is actually
    holding (Pete, 2026-09-14).  In stage A those are the parks; after that they
    are the hovers the previous stage stopped at.

    `partner_standoff` (S, metres) is the one lever of docs/V2_STAGED.md §25: a
    LEADER is held to `gate + S` against its SAME-ROW FOLLOWER's pose-invariant
    capsules and to the ordinary gate against everything else.  The leader's
    plan is what eats the follower's routing room -- it was certified at exactly
    `PAIR_MARGIN` against that partner's held pose and had no reason to keep
    more -- so this is where the room is bought back.  S = 0 asks for nothing
    and reproduces every number this module measured before it existed.
    """
    order = role_order(roles, actives, buckets, s)
    # WHO OWES WHOM, and only same-row: the cross-row pairs clear by +194 mm
    # with each row inside its own band (docs/V2_WORKCELLS.md section 4b), so
    # charging them would cost a leader ink and buy a follower nothing.
    S = float(partner_standoff or 0.0)
    standoffs = lf_standoffs(actives, roles, S)
    if verbose and standoffs:
        print(f"  stage {s} partner standoff S = {1000 * S:.0f} mm: "
              + "  ".join(f"{a}->{sorted(v)}"
                          for a, v in sorted(standoffs.items())))
    fixed: dict[int, tuple] = {}
    arms: dict[int, ArmStage] = {}
    deferred: dict[int, list] = {}
    start = {int(a): np.asarray(q, float).reshape(7) for a, q in parks.items()}
    for k, a in enumerate(order):
        role = str(roles.get(int(a), "active"))
        pieces = list(buckets.get((s, int(a)), []))
        pre = None
        # THE PRE-POSITION, AND ONLY FOR A FOLLOWER WITH A ROOM TO GET OUT OF.
        # The clear-out is planned with the LEADERS STILL PARKED -- which is
        # when it happens -- so the room it is routed in is the entry fleet and
        # no trajectory rooms at all; the tuck it flies to is then chosen in
        # the room WITH the leaders' trajectories in it, because that is the
        # room the follower has to stand in for the rest of the stage.
        if role == "follower" and fixed and pieces and tuck:
            freeze_stage(int(a), start, None, fl, pens, h_inv,
                         leg_cache, leg_cache_root)
            q_pre = start[int(a)].copy()
            freeze_stage(int(a), start, dict(fixed), fl, pens, h_inv,
                         leg_cache, leg_cache_root)
            q_t, info = tuck_pose(fl[int(a)], pieces, q_pre, h_inv,
                                  pens.get(int(a)))
            if q_t is not None and info["moved"]:
                freeze_stage(int(a), start, None, fl, pens, h_inv,
                             leg_cache, leg_cache_root)
                pre = clear_out(fl[int(a)], q_pre, q_t, h_inv, pens.get(int(a)))
                if pre is not None and float(pre["duration"]) <= TUCK_MAX_S:
                    start[int(a)] = np.asarray(q_t, float).reshape(7)
                else:
                    pre = None
            if verbose:
                print(f"  stage {s} arm {a} [follower] tuck: park "
                      f"{info['park_mm']:+.1f} mm, {info['kept']}/{info['tried']}"
                      f" stations clear >= {info['need_mm']:.0f} mm"
                      + (f" -> {info['best_mm']:+.1f} mm, clear-out "
                         f"{pre['duration']:.2f} s" if pre is not None
                         else " -> none (staying put)"))
        # THE FOLLOWER IS THE ONE THAT YIELDS, so the ink gate is ITS gate.  A
        # leader's ink is measured against the rooms of the leaders before it
        # and reported (that is the pre-existing reading, docs/V2_STAGED.md
        # section 8); a follower's is REFUSED at the gate, because "only try to
        # knock out lines in its cell that were safe to draw" is a per-piece
        # instruction and this is the per-piece test.
        g = gate if (role == "follower" and fixed) else None
        st, drop = _fly_or_defer(s, a, pieces,
                                 fl, pens, start, h_inv, opts, leg_cache,
                                 leg_cache_root, on_piece,
                                 (dict(fixed) if fixed else None), g,
                                 max_drops, park_policy, verbose,
                                 split_rounds, split_min_m,
                                 standoffs.get(int(a)))
        st.role, st.priority = role, int(k)
        # THE CLEAR-OUT IS PART OF THE TRAJECTORY, not a prologue to it.  Spliced
        # in here, it reaches `trajectory_room` (so the next arm avoids it),
        # `active_pair_gap` and `scene_check` (so it is certified like every
        # other second of motion) and the programme (so it is flown).
        if pre is not None:
            st.q_tuck = start[int(a)].copy()
            st.timeline = splice_timeline(pre, st.timeline)
            st.clear_out_s = float(pre["duration"])
        arms[int(a)] = st
        if drop:
            deferred[int(a)] = list(drop)
        fixed[int(a)] = trajectory_room(st, fl, pens, h_inv, dt, cluster)
        if verbose:
            print(f"  stage {s} priority {k}: arm {a} [{role}] "
                  f"({len(st.accepted)} pieces, {st.ink_m:.3f} m"
                  + (f", {sum(p.length_m for p in drop):.3f} m deferred"
                     if drop else "")
                  + f") -> {room_size(fixed[int(a)])}"
                  + (f", avoiding {sorted(st.depends_on)}" if st.depends_on
                     else ", free")
                  + ("" if st.timeline is not None else "  [NO TIMELINE]"))
    return arms, order, fixed, deferred


def _conducted_phases(phases, prog_idx, dt, M):
    """Nominal phase times, re-stamped on the CONDUCTED clock. -> [dict].

    `sch["progress"][a][m]` is the NOMINAL sample index arm `a` has reached at
    conducted frame `m`, and it is non-decreasing, so the conducted instant a
    nominal sample p is first reached at is `dt x searchsorted(progress, p)`.
    The phases are the only part of a conducted timeline that would otherwise
    still be quoting the clock the conductor threw away.
    """
    p = np.asarray(prog_idx, int)
    out = []
    for ph in phases:
        i0 = int(np.searchsorted(p, int(round(float(ph["t0"]) / dt)), "left"))
        i1 = int(np.searchsorted(p, int(round(float(ph["t1"]) / dt)), "left"))
        out.append(dict(ph, t0=float(dt * min(i0, M - 1)),
                        t1=float(dt * min(max(i1, i0), M - 1))))
    return out


def _bucket_ink(a, s, buckets, deferred) -> float:
    """The metres one arm is offered in one conducted stage. -> m."""
    return float(sum(p.length_m
                     for p in list((buckets or {}).get((int(s), int(a)), []))
                     + list((deferred or {}).get(int(a), []))))


def _serialise_group(s, who, buckets, deferred, fl, pens, held, parks, h_inv,
                     opts, leg_cache, leg_cache_root, dt, sub, verbose,
                     outside, op, rooms, bands, why):
    """A group the conductor REFUSED, re-planned IN PRIORITY ORDER.

    -> (arms, report).

    THE DEFECT THIS CLOSES, MEASURED 2026-09-14 (docs/V2_STAGED.md §28.8).  When
    `idle.conduct` refuses, the fallback flies the group's arms one at a time --
    but their routes came off `plan_bucket`, which runs BEFORE `freeze_conduct`
    and is given `partners=outside` only, so a row's own two arms were never in
    each other's room.  The moving arm then routed straight through its standing
    partner: 2 ↔ 97 at **-161.3 mm**, 31 ↔ 71 at **-21.0 mm**, and in both the
    other arm's per-frame travel is 0.00 mm -- it is standing still.
    `freeze_conduct`'s `still` set catches the arms that produce NO timeline; it
    cannot catch a pair where both arms have ink, because neither is still at
    freeze time and the refusal is only discovered afterwards.

    SO THE GROUP GETS THE SAME PRIORITY ARGUMENT `_priority_stage` MAKES BETWEEN
    ARMS IN A STAGE, one level down.  Busiest arm first, against the frozen
    outside fleet plus its partner AT THE POSE THAT PARTNER HOLDS; then the
    second arm against the outside fleet plus the first arm's REALISED
    TRAJECTORY as an exact swept room (`trajectory_room`, the same machinery the
    main stages use).  Every leg is re-routed under that room -- nothing is
    reused from `plan_bucket`, because what `plan_bucket` produced is exactly
    what was wrong.

    ...AND THE ROOM IS THE CONSERVATIVE READING OF A SLOT THAT DOES NOT OVERLAP.
    `_merge_conducts` lays a refused group's arms END TO END, so while arm two
    flies, arm one has finished and is standing at its park -- which its own
    trajectory room contains.  Certifying against the whole room is therefore
    strictly stronger than the programme needs, and it is what is asked FIRST.
    Where it costs the second arm a piece, that piece is not thrown away: the
    arm is re-planned against the first arm's FINISHING POSE alone, which is the
    honest statement of the slot it actually flies in -- a solo against a frozen
    fleet, always routable or honestly refused.  Which of the two shipped is
    recorded per arm (`room_kind`, `note`).

    `hold_gap` proves the pose set at every slot boundary, because a boundary is
    a barrier: six arms standing in one scene while one of them hands over.
    """
    order = sorted((int(a) for a in who),
                   key=lambda a: (-_bucket_ink(a, s, buckets, deferred), int(a)))
    held7 = {int(a): np.asarray(q, float).reshape(7) for a, q in held.items()}
    parks7 = {int(a): np.asarray(parks[a], float).reshape(7) for a in fl}
    op = dict(op or {})
    out_pose = {int(a): np.asarray(op.get(int(a), held7[int(a)]),
                                   float).reshape(7) for a in outside}
    base_env = {int(a): v for a, v in (rooms or {}).items() if int(a) in out_pose}
    keep = (str(bands) == "keep")
    arms: dict[int, ArmStage] = {}
    fin: dict[int, np.ndarray] = {}          # where an arm that has flown stands
    fin_room: dict[int, tuple] = {}          # ...and the room it swept to get there
    slots: list[dict] = []
    for k, a in enumerate(order):
        pcs = list(buckets.get((int(s), int(a)), [])) \
            + list((deferred or {}).get(int(a), []))
        # WHAT EVERY OTHER ARM IS DOING WHILE THIS ONE FLIES ITS SLOT: an arm
        # outside the group holds `op`; an arm of this group that has already
        # flown is home and holding; one that has not started is standing at its
        # stage entry pose, which is the thing nobody was routing around.
        poses = dict(out_pose)
        env = dict(base_env)
        for b in order:
            if b == a:
                continue
            poses[b] = fin.get(b, held7[b])
            if b in fin_room:
                env[b] = fin_room[b]
        partners = sorted(poses)
        start = dict(poses)
        start[int(a)] = held7[int(a)]

        def _plan(use_env):
            return plan_bucket(s, a, pcs, fl, pens, start, h_inv, opts,
                               leg_cache=leg_cache,
                               leg_cache_root=leg_cache_root, fly=True,
                               envelopes=(dict(use_env) or None),
                               partners=partners,
                               park_policy=writing.PARK_HOME,
                               keep_bands=keep, verbose=False)

        st = _plan(env)
        if pcs and env and (st.timeline is None
                            or len(st.accepted) < len(pcs)):
            # THE PIECE THE ROOM COST IS OFFERED THE SLOT INSTEAD.  The room is
            # a claim about arms that are no longer moving when this arm flies;
            # dropping it for their finishing POSES is not a relaxation of the
            # programme, it is the programme.
            alt = _plan(base_env)
            better = ((st.timeline is None and alt.timeline is not None)
                      or (alt.timeline is not None
                          and len(alt.accepted) > len(st.accepted)))
            if better:
                alt.note = (alt.note + "; " if alt.note else "") + \
                    ("in-group priority: re-planned against the partners' "
                     "finishing poses after the swept room refused "
                     f"{len(pcs) - len(st.accepted)} piece(s)")
                st = alt
        st.role, st.conducted, st.residue = "conductor", False, True
        st.priority = int(k)
        # AN ARM WITH NOTHING TO DRAW STILL HAS TO GET HOME, and it gets there in
        # the room it was just frozen into -- see `_conduct_stage`'s own aside.
        q_home = parks7[int(a)]
        if st.timeline is None and float(np.max(np.abs(held7[int(a)]
                                                       - q_home))) > 1e-9:
            try:
                st.timeline = writing.arm_program(
                    fl[a], [], h_inv=h_inv, pen_ext=pens.get(a),
                    q_start=held7[int(a)], park=writing.PARK_HOME, aside=q_home)
            except writing.PaperRefused as exc:
                st.note = (st.note + "; " if st.note else "") + \
                    f"go-home refused: {exc}"
        arms[int(a)] = st
        slots.append(dict(arm=int(a), poses=dict(poses),
                          frames=_clock_frames(st.timeline, dt)))
        fin[int(a)] = np.asarray(st.q_end, float).reshape(7)
        if st.timeline is not None:
            fin_room[int(a)] = trajectory_room(st, fl, pens, h_inv, dt)
        if verbose:
            print(f"  stage {s} group {sorted(int(x) for x in who)} priority "
                  f"{k}: arm {a} ({len(st.accepted)}/{len(pcs)} pieces, "
                  f"{st.ink_m:.3f} m) against {sorted(partners)}"
                  + (f", rooms {sorted(env)}" if env else ", poses only"))
    thaw()              # the CHECK does not inherit the planner's own room
    worst, ok, holds = float("inf"), True, []
    for sl in slots:
        st = arms[int(sl["arm"])]
        if st.timeline is None:
            continue
        rep = solo_check(st, sl["poses"], fl, pens, h_inv, PAIR_MARGIN, dt, sub)
        worst = min(worst, float(rep.get("min_clearance", np.nan)))
        ok = ok and bool(rep.get("ok", False))
        sl["min_clearance_m"] = float(rep.get("min_clearance", np.nan))
        sl["ok"] = bool(rep.get("ok", False))
    # ...AND THE SLOT BOUNDARIES, WHICH ARE BARRIERS.  At the instant one arm
    # stops and the next starts, six arms are standing in one scene and nothing
    # is moving; `hold_gap` is the same proof the stage barriers carry.
    for k, sl in enumerate(slots):
        q = dict(sl["poses"])
        q[int(sl["arm"])] = held7[int(sl["arm"])]
        holds.append(dict(before=int(sl["arm"]), **hold_gap(q, fl, pens, h_inv)))
    q_end = dict(out_pose)
    q_end.update({int(a): fin[int(a)] for a in order})
    holds.append(dict(before=None, **hold_gap(q_end, fl, pens, h_inv)))
    hold_ok = all(bool(h["ok"]) for h in holds)
    return arms, dict(
        ok=bool(ok and hold_ok), serialised=True, reason=str(why),
        min_clearance=float(worst), in_group_priority=True,
        serial_order=[int(a) for a in order],
        slots=[dict(arm=int(sl["arm"]), frames=int(sl["frames"]),
                    min_clearance_m=float(sl.get("min_clearance_m", np.nan)),
                    ok=bool(sl.get("ok", True))) for sl in slots],
        slot_holds=[dict(before=h["before"], min_m=float(h["min_m"]),
                         ok=bool(h["ok"])) for h in holds])


def _conduct_stage(s, buckets, deferred, fl, pens, held, parks, h_inv, opts,
                   leg_cache, leg_cache_root, on_piece, dt, sub, verbose,
                   only=None, rooms=None, outside_poses=None, bands="drop"):
    """THE FINAL PASS: everything A and B could not fly, conducted.

    -> (arms, check_report, seconds).

    `only` names the arms this conduct is over; `None` is all six, which is
    `ARCHITECTURE_V2` section 2d's combinatorial case and the one §22 measured
    at 3 075 s of wall for 126.6 s of motion.  A two-arm `only` is four
    priority orders instead of 720 and is what `_conduct_rows` runs three of.

    Pete's own words: "in the end we would do the coordination of all arms to
    fill in the gaps if needed".  This is the one stage in the programme where
    there IS a shared clock, and `idle.conduct` -- the machinery that made v19 --
    is the tool, unchanged.  Its cost is the combinatorial one
    `ARCHITECTURE_V2` section 2d warns about, and it is paid exactly once, on
    exactly the ink nothing else could take.

    THIS IS ALSO WHERE THE PARKS COME BACK.  Stages A and B end wherever the
    ink ended -- that is the whole of Pete's "no excessive parking" -- so the
    fleet is not at its parks when the final pass starts, and it has to be when
    the programme ends: the park is the fault-recovery home and the pose the
    next programme will be planned from.  So the conduct runs under
    `POLICY_HOME` for the arms that draw, and every arm that draws NOTHING is
    given its park as an `aside`, which `writing.arm_program` lays down as a
    certified routed move on the real clock rather than as a teleport.

    IF NOTHING WAS DEFERRED, THE SEAMS ARE EMPTY AND THE FLEET IS ALREADY HOME,
    THIS STAGE COSTS NOTHING.  There is no conduct, no check and no barrier.
    """
    from . import idle as idle_mod
    t0 = time.perf_counter()
    who = sorted(int(a) for a in (fl if only is None else only))
    # THE ARMS THIS CONDUCT DOES NOT MOVE ARE THE ROOM, and they stay in it for
    # the whole conduct -- `freeze_conduct` says why.  `outside_poses` is what
    # each of them holds while this group runs (its stage entry pose for a group
    # that has not started, its park for one that has already finished and gone
    # home), and `rooms` upgrades an outside arm that is itself moving in this
    # stage from a pose to its whole realised swept trajectory.
    outside = sorted(int(a) for a in fl if int(a) not in set(who))
    op = dict(held if outside_poses is None else outside_poses)
    rooms = {int(a): v for a, v in (rooms or {}).items() if int(a) in outside}
    arms: dict[int, ArmStage] = {}
    segs: dict[int, list] = {}
    room_poses = {int(a): np.asarray(q, float).reshape(7)
                  for a, q in held.items()}
    room_poses.update({int(a): np.asarray(q, float).reshape(7)
                       for a, q in op.items()})
    for a in who:
        pcs = list(buckets.get((s, int(a)), [])) + list(deferred.get(int(a), []))
        st = plan_bucket(s, a, pcs, fl, pens, room_poses, h_inv, opts,
                         leg_cache=leg_cache, leg_cache_root=leg_cache_root,
                         fly=True, on_piece=on_piece,
                         envelopes=(rooms or None), partners=outside,
                         park_policy=writing.PARK_HOME, verbose=False)
        st.role, st.conducted = "conductor", True
        arms[int(a)] = st
        segs[int(a)] = list(st.programme)
    home = {int(a): np.asarray(parks[a], float).reshape(7)
            for a in who if not segs.get(int(a))}
    # AN ARM IN THE GROUP THAT IS NOT GOING TO MOVE IS A WALL, NOT A MOVER, AND
    # THE ROOM HAS TO SAY SO.  `freeze_conduct` freezes the arms OUTSIDE the
    # group because the movers cannot be in their own room -- but an arm inside
    # the group whose bucket came out empty and whose park it is already standing
    # at moves no joint in this conduct, and it was in neither half of the safety
    # argument: not in the frozen set (it is nominally a mover) and not in the
    # conductor's schedule (it has no timeline).  Measured 2026-09-14 on
    # `lf6_whole` stage C: arm 31 drew nothing, stood at its park for the whole
    # conduct, and arm 71 routed a leg **-41.8 mm** through it.
    # ...AND THAT IS NOT ONLY THE EMPTY BUCKETS.  An arm whose bucket WOULD NOT
    # FLY has pieces, so it is not offered a go-home `aside` either -- it simply
    # stands at its entry pose for the whole conduct with no timeline at all.
    # Measured 2026-09-14 on `lf6_whole` stage C: arm 31 carried 8 pieces, flew
    # none, stood still, and arm 71 routed through it at -20.7 mm.
    still = sorted(int(a) for a in who
                   if arms[int(a)].timeline is None
                   and (int(a) not in home
                        or float(np.max(np.abs(
                            np.asarray(held[a], float).reshape(7)
                            - home[int(a)]))) <= 1e-9))
    room = sorted(set(outside) | set(still))
    op_all = dict(op)
    op_all.update({int(a): np.asarray(held[a], float).reshape(7)
                   for a in still})
    if room:
        freeze_conduct(room, op_all, fl, pens, h_inv, rooms, leg_cache,
                       leg_cache_root, keep_bands=(str(bands) == "keep"))
    else:
        thaw()
    # AN ARM WITH NOTHING TO DRAW STILL HAS TO GET HOME.  `plan_bucket` returns
    # no timeline for an empty bucket, so an arm that drew in stage A or B and
    # has no residue would simply stand at its hover for ever.  `arm_program`
    # with no segments and an `aside` lays the go-home down as one certified
    # routed move on the real clock -- the same object `idle.conduct` builds for
    # it -- so the fallback path below has one too, and the programme ends at
    # the parks whether the conductor takes it or not.
    for a, q in sorted(home.items()):
        if float(np.max(np.abs(np.asarray(held[a], float).reshape(7) - q))) \
                <= 1e-9:
            continue
        try:
            arms[a].timeline = writing.arm_program(
                fl[a], [], h_inv=h_inv, pen_ext=pens.get(a),
                q_start=np.asarray(held[a], float).reshape(7),
                park=writing.PARK_HOME, aside=q)
        except writing.PaperRefused as exc:
            arms[a].note = f"go-home refused: {exc}"
    moved = any(float(np.max(np.abs(np.asarray(held[a], float).reshape(7)
                                    - np.asarray(parks[a], float).reshape(7))))
                > 1e-9 for a in who)
    if not any(segs.values()) and not moved:
        for st in arms.values():
            st.timeline = None
        thaw()
        return arms, dict(ok=True, empty=True, min_clearance=float("inf")), \
            time.perf_counter() - t0
    if verbose:
        print("  stage %d CONDUCT: %s" % (s, ", ".join(
            f"arm {a} {len(v)} pieces" for a, v in sorted(segs.items()) if v)))
    try:
        out = idle_mod.conduct(segs, {int(a): float(pens[a]) for a in who}, dt,
                               q_start={int(a): np.asarray(held[a],
                                                           float).reshape(7)
                                        for a in who},
                               policy=idle_mod.POLICY_HOME, aside=home,
                               specs=fl, h_inv=h_inv, verbose=verbose)
    except (idle_mod.Unconductable, RuntimeError) as exc:
        # THE FLOOR UNDER THE WHOLE SCHEME IS THE ONE-ARM-AT-A-TIME PROGRAMME.
        # A conductor that refuses has said "these timelines cannot share a
        # clock", which is a statement about sharing and not about the ink: each
        # arm's bucket was already planned and certified ALONE against the
        # parked fleet, so the final pass falls back to flying them in turn.
        # `solo_check` is the certificate for that, `active_pair_gap` never sees
        # it because nothing else is moving, and the cost is the sum rather than
        # the max -- which is announced, not hidden.
        if verbose:
            print(f"  stage {s}: the conductor refused ({exc}); the final pass "
                  "falls back to one arm at a time, IN PRIORITY ORDER")
        arms, rep = _serialise_group(
            s, who, buckets, deferred, fl, pens, held, parks, h_inv, opts,
            leg_cache, leg_cache_root, dt, sub, verbose, outside, op, rooms,
            bands, exc)
        return arms, rep, time.perf_counter() - t0
    thaw()              # the CHECK does not inherit the conductor's own room
    sch, samp, progs = out["sch"], out["samp"], out["progs"]
    M = int(sch["M"])
    ts = dt * np.arange(M)
    qtraj, prog_idx = {}, {}
    for a, st in arms.items():
        idx = np.clip(np.asarray(sch["progress"][a], int)[:M], 0,
                      int(samp[a]["n"]) - 1)
        prog_idx[int(a)] = idx
        q = np.asarray(samp[a]["q"], float)[idx]
        qtraj[int(a)] = q
        p = progs[a]
        st.timeline = dict(
            t=ts, q=q, seg=np.asarray(samp[a]["seg"], int)[idx],
            u=np.asarray(samp[a]["u"], float)[idx],
            phases=_conducted_phases(p["phases"], idx, dt, M),
            # EVERY ARM'S STAGE LASTS AS LONG AS THE STAGE.  The conductor
            # pauses arms rather than shortening them, so an arm that finished
            # early is still standing in this stage's timeline and still has to
            # be drawn by the animation and seen by the check.
            duration=float(ts[-1]) if M > 1 else 0.0,
            draw_s=float(p["draw_s"]), transit_s=float(p["transit_s"]),
            ink=p.get("ink"), q_end=np.asarray(q[-1], float))
    rep = scene_check.check_timeline(
        qtraj, dt, float(sch["margin"]), programs=segs,
        h_inv=h_inv,
        pen_ext={int(a): scene_check.pen_len(pens.get(a), a) for a in fl},
        sub=int(sub), progress={int(a): v for a, v in prog_idx.items()},
        verbose=False)
    rep = dict(rep)
    # WHY IT FAILED, IN THE STAGE LINE.  `check_timeline`'s verdict is one bool
    # over seven gates and a FAIL that does not say which is a FAIL nobody can
    # act on -- the conducted pass is the one stage where the inter-arm gate is
    # not the likely culprit, because the conductor's whole job is that gate.
    rep["failed"] = sorted(
        ([f"clearance:{1000 * float(rep.get('min_clearance', np.nan)):+.1f}mm"]
         if float(rep.get("min_clearance", np.inf)) < float(sch["margin"])
         else [])
        + [k.replace("_failed", "") for k in ("frame_failed", "paper_failed",
                                              "self_failed")
           if rep.get(k)]
        + (["frozen"] if int(rep.get("frozen_failed", 0)) else [])
        + (["joint_limit"]
           if min(rep.get("joint_margin", {1: 1.0}).values()) <= 0.0 else [])
        + ([] if rep.get("monotone", True) else ["monotone"]))
    rep["margin_m"] = float(sch["margin"])
    rep["makespan_s"] = float(ts[-1]) if M > 1 else 0.0
    rep["pause_s"] = float(sch.get("pause_total", 0.0))
    return arms, rep, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# 7b.  THE FINAL PASS, PER ROW
# ---------------------------------------------------------------------------
# WHAT IS DEFERRED IS, BY CONSTRUCTION, THE CONTESTED STRIP BETWEEN ONE ROW'S
# TWO ARMS: a follower yields the ink that lies inside its same-row leader's
# trajectory, and a same-row pair is the pair no work cell separates.  Rows,
# though, ARE separated -- by the 0.40 m y dead band, measured at +194.3 and
# +201.1 mm between cross-row pairs drawing in their own bands
# (docs/V2_WORKCELLS.md section 4b) -- so the final pass does not need one
# conductor over six arms.  It needs three over two, and they can run at once.
#
# The exception is ink that does not stay in a row band.  A piece straddling a
# dead band puts its arm's elbow where the next row's argument does not reach,
# so it cannot go to a row conductor; it goes to a short stage D instead.
ROW_ARMS = {j: tuple(sorted(a for a in traces_mod.ARMS
                           if traces_mod.ROW_OF[a] == j))
            for j in sorted(set(traces_mod.ROW_OF.values()))}
BAND_EPS = 1e-9


def piece_row(pc: "Piece", dead_band_m: float = traces_mod.DEAD_BAND_M):
    """Which row conductor may take this piece. -> int, or None for the band.

    The arm's row, IF the piece's whole geometry stays inside that row's band.
    A piece with a point in the dead band is `None`: the cross-row clearance
    that licenses three conductors was measured with each row drawing inside
    its own band, and this piece is not.
    """
    j = traces_mod.ROW_OF.get(int(pc.arm))
    if j is None:
        return None
    _, y0, _, y1 = traces_mod.row_band(int(j), dead_band_m)
    ys = np.asarray(pc.pts, float).reshape(-1, 2)[:, 1]
    return int(j) if (float(ys.min()) >= y0 - BAND_EPS
                      and float(ys.max()) <= y1 + BAND_EPS) else None


def partition_deferred(deferred: dict, dead_band_m: float = traces_mod.DEAD_BAND_M):
    """{arm: [Piece]} -> ({row: {arm: [Piece]}}, {arm: [Piece]}, stats)."""
    rows: dict[int, dict[int, list]] = {}
    band: dict[int, list] = {}
    stats = dict(row_m={}, band_m={}, n_row=0, n_band=0)
    for a, ps in (deferred or {}).items():
        for p in ps:
            j = piece_row(p, dead_band_m)
            if j is None:
                band.setdefault(int(a), []).append(p)
                stats["band_m"][int(a)] = (stats["band_m"].get(int(a), 0.0)
                                           + float(p.length_m))
                stats["n_band"] += 1
            else:
                rows.setdefault(j, {}).setdefault(int(a), []).append(p)
                stats["row_m"][j] = stats["row_m"].get(j, 0.0) + float(p.length_m)
                stats["n_row"] += 1
    tot = sum(stats["row_m"].values()) + sum(stats["band_m"].values())
    stats["total_m"] = float(tot)
    stats["band_frac"] = float(sum(stats["band_m"].values()) / max(tot, 1e-12))
    return rows, band, stats


def _thin_plan(pl: dict) -> dict:
    """The part of a `plan_stroke` result anything downstream of a conduct reads.

    A ROW CONDUCT RUNS IN ANOTHER PROCESS AND ITS ANSWER HAS TO COME BACK.  The
    full plan carries the planner's own working objects (`lat`, `sheet_obj`,
    `pwl`, the smoother's state); nothing outside the planner reads them, and
    they are not worth pickling.  `programme()` wants `qs`; `scene_check`'s
    per-segment re-validation wants `pts`, `qs`, `times` and the lean cone.
    """
    return dict(qs=np.asarray(pl["qs"], float), pts=np.asarray(pl["pts"], float),
                times=np.asarray(pl["times"], float),
                arc_len=float(pl.get("arc_len", 0.0)),
                tilt_max_deg=float(pl.get("tilt_max_deg", 0.0) or 0.0))


def _thin_stage(st: ArmStage) -> ArmStage:
    """`st` with every planner working object dropped, so it can be pickled."""
    out = dataclasses.replace(
        st,
        planned=[dataclasses.replace(
            pp, plan=(None if pp.plan is None else _thin_plan(pp.plan)))
            for pp in st.planned])
    thin = {id(pp0.plan): pp.plan for pp0, pp in zip(st.planned, out.planned)
            if pp0.plan is not None}
    out.programme = [dict(sg, plan=thin.get(id(sg.get("plan")),
                                            _thin_plan(sg["plan"])))
                     for sg in st.programme]
    if st.timeline is not None:
        tl = st.timeline
        out.timeline = {k: tl[k] for k in
                        ("t", "q", "seg", "u", "phases", "duration", "draw_s",
                         "transit_s") if k in tl}
        out.timeline["q_end"] = np.asarray(tl["q"], float)[-1]
    return out


def _row_job(payload):
    """One row's whole final pass, in its own process. -> (arms, rep, seconds).

    Forked, so it inherits the parent's fleet, atlas, leg store and environment
    exactly; only the answer crosses back, and `_thin_stage` is what makes the
    answer crossable.
    """
    (s, who, buckets, defer, fl, pens, held, parks, h_inv, opts, leg_cache,
     leg_cache_root, dt, sub, verbose, rooms, op, bands) = payload
    arms, rep, secs = _conduct_stage(s, buckets, defer, fl, pens, held, parks,
                                     h_inv, opts, leg_cache, leg_cache_root,
                                     None, dt, sub, verbose, only=who,
                                     rooms=rooms, outside_poses=op, bands=bands)
    return {int(a): _thin_stage(st) for a, st in arms.items()}, rep, float(secs)


def _pad(v, M):
    v = np.asarray(v)
    if len(v) >= M:
        return v[:M]
    return np.concatenate([v, np.repeat(v[-1:], M - len(v), axis=0)])


def _on_clock(tl, dt):
    """One arm's timeline, on the MERGE's clock. -> dict.

    A CONDUCTED timeline is already `dt`-spaced -- `_conduct_stage` builds it by
    indexing `idle.conduct`'s own uniform samples -- but a timeline the
    conductor REFUSED, and a bare go-home `aside`, come straight out of
    `writing.arm_program` on its WAYPOINT clock, where two consecutive rows can
    be a whole reconfiguration apart.  Merging those as if they were frames puts
    4.3 rad between two samples, and `scene_check`'s 1-Lipschitz residual then
    reads half a metre of self-collision that is not there (measured
    2026-09-14: arm 2 at -531.6 mm, arm 97 at -512.4 mm).  `uniform_samples` is
    the same resampler the conductor uses, and it is applied here so that every
    arm in the merged scene is on one clock whatever produced it.
    """
    t = np.asarray(tl["t"], float)
    if len(t) > 1:
        want = dt * np.arange(len(t))
        if float(np.max(np.abs(t - want))) <= 1e-6:
            return tl
    s = writing.uniform_samples(dict(tl, t=t, q=np.asarray(tl["q"], float),
                                     seg=np.asarray(tl["seg"], int),
                                     u=np.asarray(tl["u"], float),
                                     duration=float(tl.get("duration",
                                                           t[-1] if len(t)
                                                           else 0.0))), dt)
    return dict(tl, t=dt * np.arange(int(s["n"])), q=s["q"], seg=s["seg"],
                u=s["u"])


def _lead(v, n, first):
    """`n` frames of `first` in front of `v`. -> array."""
    v = np.asarray(v)
    if n <= 0:
        return v
    return np.concatenate([np.repeat(np.asarray(first).reshape((1,) + v.shape[1:]),
                                     n, axis=0), v])


def _clock_frames(tl, dt) -> int:
    """How many MERGE frames one timeline takes. -> int.

    `_on_clock`'s answer, WITHOUT resampling anything -- so a caller that has to
    lay slots end to end can ask before the merge has run.  A conducted timeline
    is already `dt`-spaced and is its own length; a timeline the conductor
    REFUSED comes off `writing.arm_program`'s waypoint clock, where the row count
    says nothing at all about the seconds (measured 2026-09-14: 354 waypoints
    over 59.7 s, which is 1 195 frames).
    """
    if tl is None:
        return 1
    t = np.asarray(tl["t"], float)
    if len(t) > 1:
        want = dt * np.arange(len(t))
        if float(np.max(np.abs(t - want))) <= 1e-6:
            return len(t)
    dur = float(tl.get("duration", t[-1] if len(t) else 0.0))
    return max(int(np.ceil(dur / float(dt))) + 1, 1)


def _slot_frames(part, dt) -> int:
    """How long one group's SLOT is, in merge frames. -> int.

    THE SLOT IS THE MERGE'S CLOCK AND NOTHING ELSE.  `--row-compose serial` lays
    the groups end to end, so a slot measured in the wrong unit overlaps the
    next group -- and a group the conductor REFUSED is doubly wrong, because its
    arms fly one after another (`_merge_conducts` says so) and its slot is
    therefore the SUM of their lengths rather than the max.

    MEASURED 2026-09-14 on `lf6_whole` stage C: group [31, 71] was refused, its
    354-waypoint timeline was read as 354 frames, group [13, 17] was laid down
    at 17.65 s -- and arm 71 was still flying until 59.70 s.  The merge read
    17 <-> 71 at **-108.6 mm**, which is not a composition failure of the scheme
    but of this arithmetic.
    """
    ns = [_clock_frames(st.timeline, dt) for st in part[0].values()
          if st.timeline is not None]
    if not ns:
        return 1
    if part[1].get("serialised"):
        return sum(n - 1 for n in ns) + 1
    return max(ns)


def _merge_conducts(stage, parts, fl, pens, held, parks, h_inv, dt, sub,
                    offsets=None):
    """Several disjoint conducts on one clock. -> (arms, report).

    Each group was conducted alone; the groups overlap in TIME, which is the
    whole point, and what licenses that is the row dead band.  So the merge is
    an assertion and the whole-timeline `scene_check` below is what discharges
    it -- over all six arms, on the conductor's own margin, with the sweep
    residual, sharing no code with any of this.

    `offsets` is {arm: frames} and is what `--row-compose serial` needs: a
    group that runs AFTER another one starts its own clock where the earlier one
    stopped, holds its stage entry pose until then, and holds its finishing pose
    afterwards.  `None` is the overlapping form, every group starting at frame
    zero -- which is the composition §24.5 measured and refused.
    """
    arms: dict[int, ArmStage] = {}
    for a, st in ((int(a), st) for g in parts for a, st in g[0].items()):
        arms[a] = st
    for st in arms.values():
        if st.timeline is not None:
            st.timeline = _on_clock(st.timeline, dt)
    off = {int(a): int(v) for a, v in (offsets or {}).items()}
    # A GROUP THE CONDUCTOR REFUSED FLIES ONE ARM AT A TIME, AND THE MERGE HAS
    # TO SAY SO.  `_conduct_stage`'s fallback hands back each arm's timeline on
    # ITS OWN clock -- that is the whole content of "serialised" -- so laying
    # them all down from frame zero merges a programme nobody is going to run.
    # Measured 2026-09-14: group [2, 97] refuses the conductor, and merged
    # concurrently its two arms read -208.3 mm against each other.
    for g in parts:
        if not g[1].get("serialised"):
            continue
        base = max([0] + [int(off.get(int(a), 0)) for a in g[0]])
        # THE ORDER THE SLOTS ARE LAID DOWN IN IS THE ORDER THEY WERE PLANNED
        # IN, and it is not the arm ids.  `_serialise_group` plans the busiest
        # arm first and certifies the second against the first's realised
        # trajectory; laying them down the other way round would merge a
        # programme with no certificate at all.
        seq = [int(a) for a in (g[1].get("serial_order") or [])
               if int(a) in g[0]]
        for a in seq + [a for a in sorted(int(x) for x in g[0])
                        if a not in set(seq)]:
            st = g[0][a]
            off[int(a)] = int(base)
            if st.timeline is not None:
                base += len(np.asarray(st.timeline["t"], float)) - 1
    M = max([1] + [int(off.get(int(a), 0))
                   + len(np.asarray(st.timeline["t"], float))
                   for a, st in arms.items() if st.timeline is not None])
    ts = dt * np.arange(M)
    qtraj, segs = {}, {}
    for a in sorted(fl):
        st = arms.get(int(a))
        if st is None:
            st = ArmStage(int(stage), int(a),
                          np.asarray(held[a], float).reshape(7))
            st.role, st.conducted = "conductor", True
            arms[int(a)] = st
        if st.timeline is None:
            # AN ARM IN NO GROUP IS NOT ABSENT, IT IS STANDING THERE.  It has
            # to be in the merged check as the pose it holds, or the check
            # would certify a scene with four arms in it.
            qtraj[int(a)] = np.repeat(
                np.asarray(held[a], float).reshape(1, 7), M, axis=0)
            continue
        n = int(off.get(int(a), 0))
        tl = st.timeline
        q0 = np.asarray(held[a], float).reshape(7)
        q = _pad(_lead(np.asarray(tl["q"], float), n, q0), M)
        tl = dict(tl, t=ts,
                  q=q,
                  seg=_pad(_lead(np.asarray(tl["seg"], int), n, -1), M),
                  u=_pad(_lead(np.asarray(tl["u"], float), n, 0.0), M),
                  phases=[dict(ph, t0=float(ph["t0"]) + n * dt,
                               t1=float(ph["t1"]) + n * dt)
                          for ph in (tl.get("phases") or [])],
                  duration=float(ts[-1]) if M > 1 else 0.0,
                  q_end=np.asarray(q[-1], float))
        st.timeline = tl
        qtraj[int(a)] = q
        if st.programme:
            segs[int(a)] = list(st.programme)
    # EVERY ARM'S t = 0 POSE, PAIRWISE, BEFORE ANYTHING IS BELIEVED ABOUT THE
    # MOTION.  The merge stands six arms in one scene for the first time, and
    # `hold_gap` is the same proof the stage barriers already carry.
    t0h = hold_gap({int(a): np.asarray(v, float)[0] for a, v in qtraj.items()},
                   fl, pens, h_inv)
    rep = scene_check.check_timeline(
        qtraj, dt, PAIR_MARGIN, programs=segs or None, h_inv=h_inv,
        pen_ext={int(a): scene_check.pen_len(pens.get(a), a) for a in fl},
        sub=int(sub), verbose=False)
    rep = dict(rep)
    rep["failed"] = sorted(
        ([f"clearance:{1000 * float(rep.get('min_clearance', np.nan)):+.1f}mm"]
         if float(rep.get("min_clearance", np.inf)) < PAIR_MARGIN else [])
        + [k.replace("_failed", "") for k in ("frame_failed", "paper_failed",
                                              "self_failed") if rep.get(k)]
        + (["frozen"] if int(rep.get("frozen_failed", 0)) else [])
        + (["joint_limit"]
           if min(rep.get("joint_margin", {1: 1.0}).values()) <= 0.0 else []))
    rep["margin_m"] = float(PAIR_MARGIN)
    rep["makespan_s"] = float(ts[-1]) if M > 1 else 0.0
    rep["t0_holds"] = dict(min_m=float(t0h["min_m"]), ok=bool(t0h["ok"]))
    rep["serialised_groups"] = [sorted(int(a) for a in g[0]) for g in parts
                                if g[1].get("serialised")]
    rep["offsets_s"] = {str(a): round(dt * int(v), 3)
                        for a, v in sorted(off.items()) if int(v)}
    if not t0h["ok"]:
        rep["failed"] = sorted(set(rep["failed"]) | {"t0_holds"})
        rep["ok"] = False
    rep["groups"] = [dict(arms=sorted(int(a) for a in g[0]),
                          ok=bool(g[1].get("ok", False)),
                          min_clearance_m=float(g[1].get("min_clearance",
                                                         np.nan)),
                          wall_s=float(g[2])) for g in parts]
    return arms, rep


ROW_COMPOSE = ("parallel", "priority", "serial")


def group_order(groups, deferred, buckets, stage) -> list[tuple]:
    """The order the row groups choose their paths in. -> [(arm, ...)].

    BUSIEST FIRST, exactly as `priority_order` orders the arms inside a stage
    and for the same reason: the group with the most ink has the least freedom,
    so it plans first and its trajectory is then final.  Ties break on the
    sorted arm ids, so the order is a function of the plan.
    """
    def ink(who):
        return sum(p.length_m
                   for a in who
                   for p in (list((deferred or {}).get(int(a), []))
                             + list((buckets or {}).get((int(stage), int(a)),
                                                        []))))
    return sorted((tuple(g) for g in groups),
                  key=lambda g: (-ink(g), tuple(sorted(g))))


def _group_payload(s, who, buckets, deferred, fl, pens, held, parks, h_inv,
                   opts, leg_cache, leg_cache_root, dt, sub, rooms, op,
                   bands="drop"):
    return (s, tuple(who), buckets,
            {int(a): list(v) for a, v in (deferred or {}).items()
             if int(a) in set(int(x) for x in who)},
            fl, pens, held, parks, h_inv, opts, leg_cache, leg_cache_root,
            dt, sub, False, (rooms or None), (op or None), str(bands))


CONDUCT_BANDS = "auto"     # drop | keep | auto -- see `_conduct_groups`


def _conduct_groups(s, groups, buckets, deferred, fl, pens, held, parks, h_inv,
                    opts, leg_cache, leg_cache_root, dt, sub, jobs, cap_s,
                    verbose, mode="parallel"):
    """`_conduct_set`, with the band retry `CONDUCT_BANDS` asks for.

    THE ROOM A CONDUCT ROUTES IN AND THE ROOM THE JUDGE MEASURES IT IN ARE NOT
    THE SAME ROOM, and stage D of the serial programme is where that showed
    (docs/V2_STAGED.md section 28).  `freeze_conduct` drops a frozen partner's
    band AABB and replaces it by that partner's real capsules -- 127 mm of
    honest room, and the whole reason `frozen` exists -- while
    `scene_check`'s FRAME gate measures every arm against
    `spec.static_obstacles()`, bands included, knowing nothing about any of it.

    So: route in the relaxed room, and if the judge's frame gate refuses what
    comes out, route the whole set AGAIN with the bands kept as well and take
    that instead -- but only if it actually clears the gate.  `keep` and `drop`
    force one room; `auto` (the default) pays for the second pass only where
    the first produced something that cannot ship.  The retry can only ever be
    more conservative than the run it replaces.
    """
    want = str(CONDUCT_BANDS or "auto")
    first = "drop" if want == "auto" else want
    arms, rep, secs = _conduct_set(s, groups, buckets, deferred, fl, pens, held,
                                   parks, h_inv, opts, leg_cache,
                                   leg_cache_root, dt, sub, jobs, cap_s,
                                   verbose, mode, first)
    rep["bands"] = first
    if want != "auto" or not rep.get("frame_failed"):
        return arms, rep, secs
    if verbose:
        print(f"  stage {s}: frame gate refuses arms {rep['frame_failed']} in "
              "the relaxed room; re-conducting with the partners' bands kept")
    a2, r2, s2 = _conduct_set(s, groups, buckets, deferred, fl, pens, held,
                              parks, h_inv, opts, leg_cache, leg_cache_root,
                              dt, sub, jobs, cap_s, verbose, mode, "keep")
    r2["bands"] = "keep"
    r2["bands_retry"] = dict(
        from_frame_failed=list(rep.get("frame_failed") or []),
        to_frame_failed=list(r2.get("frame_failed") or []),
        ink_m_drop=float(sum(st.ink_m for st in arms.values())),
        ink_m_keep=float(sum(st.ink_m for st in a2.values())))
    if r2.get("frame_failed"):
        rep["bands_retry"] = dict(r2["bands_retry"], taken=False)
        return arms, rep, secs + s2      # the retry did not fix it either
    r2["bands_retry"]["taken"] = True
    return a2, r2, secs + s2


def _conduct_set(s, groups, buckets, deferred, fl, pens, held, parks, h_inv,
                 opts, leg_cache, leg_cache_root, dt, sub, jobs, cap_s,
                 verbose, mode="parallel", bands="drop"):
    """Run several disjoint conducts. -> (arms, report, seconds).

    `groups` is [(arm, ...)] and `mode` is how they COMPOSE, which is the whole
    subject of docs/V2_STAGED.md §26:

    * `parallel` -- every group starts at stage time zero and plans with the
      other groups' arms STANDING at their stage entry poses.  That is what the
      code has always claimed and (with `freeze_conduct` in) now actually does;
      it is still an assertion that four arms leaving four held poses at once
      stay apart, and §24.5 measured that assertion failing at -191.9 mm.  Kept
      as the control.
    * `priority` -- the groups are conducted BUSIEST FIRST, in a pipeline, and
      each later group sees every earlier group's REALISED trajectory as an
      exact swept room.  A pair (i, k) with i < k is then certified against the
      path arm i actually flies, at every instant, which is the same argument
      `_priority_stage` makes between arms inside a stage.  The wall is the SUM
      of the groups rather than the max; the motion still overlaps.
    * `serial` -- the groups run one after another in TIME.  Each group plans
      with the earlier groups at their PARKS (they have gone home) and the later
      ones at their entry poses, so all three still plan CONCURRENTLY; what is
      paid is the motion, which becomes the sum.

    A group that overruns `cap_s` is ABANDONED and its ink is reported as
    residue, because a final pass nobody can wait for is not a final pass.
    """
    import concurrent.futures as _fut
    import multiprocessing as _mp
    t0 = time.perf_counter()
    mode = str(mode or "parallel")
    if mode not in ROW_COMPOSE:
        raise ValueError(f"row compose {mode!r} is not one of {ROW_COMPOSE}")
    if mode in ("priority", "serial") and len(groups) > 1:
        return _compose_groups(s, groups, buckets, deferred, fl, pens, held,
                               parks, h_inv, opts, leg_cache, leg_cache_root,
                               dt, sub, jobs, cap_s, verbose, mode, t0, bands)
    jobs = max(1, min(int(jobs), len(groups)))
    payloads = [_group_payload(s, who, buckets, deferred, fl, pens, held, parks,
                               h_inv, opts, leg_cache, leg_cache_root, dt, sub,
                               None, None, bands) for who in groups]
    parts, lost = [], []
    if jobs == 1 or len(groups) == 1:
        for pay in payloads:
            parts.append(_row_job(pay))
    else:
        ctx = _mp.get_context("fork")
        ex = _fut.ProcessPoolExecutor(max_workers=jobs, mp_context=ctx)
        futs = {ex.submit(_row_job, pay): pay[1] for pay in payloads}
        try:
            # THE CAP IS ON THE WHOLE SET, AND IT HAS TO BE HERE.  `as_completed`
            # with no timeout blocks for ever on a group that hangs, and a
            # per-future timeout never runs because the future it would time out
            # is the one that has not been yielded.
            for f in _fut.as_completed(futs, timeout=max(1.0, float(cap_s))):
                who = futs[f]
                try:
                    parts.append(f.result())
                except Exception as exc:               # noqa: BLE001
                    lost.append((who, f"{type(exc).__name__}: {exc}"))
                    if verbose:
                        print(f"  stage {s} group {sorted(who)}: ABANDONED "
                              f"({type(exc).__name__}: {exc})")
        except TimeoutError:
            for f, who in futs.items():
                if not f.done():
                    f.cancel()
                    lost.append((who, f"over the {cap_s:.0f} s cap"))
                    if verbose:
                        print(f"  stage {s} group {sorted(who)}: ABANDONED "
                              f"(over the {cap_s:.0f} s wall-clock cap); its "
                              "ink is reported as residue")
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
    if verbose:
        for g in parts:
            print(f"  stage {s} group {sorted(int(a) for a in g[0])}: "
                  f"{sum(st.ink_m for st in g[0].values()):.3f} m, "
                  f"{g[2]:.1f} s wall, "
                  f"{1000 * float(g[1].get('min_clearance', np.nan)):+.1f} mm, "
                  + ("PASS" if g[1].get("ok") else "FAIL"))
    if not parts:
        return {}, dict(ok=False, empty=True, min_clearance=float("inf"),
                        abandoned=[sorted(w) for w, _ in lost]), \
            time.perf_counter() - t0
    arms, rep = _merge_conducts(s, parts, fl, pens, held, parks, h_inv, dt, sub)
    rep["compose"] = "parallel"
    rep["abandoned"] = [sorted(int(a) for a in w) for w, _ in lost]
    rep["abandoned_why"] = [why for _, why in lost]
    return arms, rep, time.perf_counter() - t0


def _one_group(pay, cap_s):
    """One group's conduct in its own forked process, under a wall cap.

    -> (part, why_lost).  Exactly one of the two is `None`.
    """
    import concurrent.futures as _fut
    import multiprocessing as _mp
    ex = _fut.ProcessPoolExecutor(max_workers=1,
                                  mp_context=_mp.get_context("fork"))
    try:
        f = ex.submit(_row_job, pay)
        try:
            return f.result(timeout=max(1.0, float(cap_s))), None
        except _fut.TimeoutError:
            return None, f"over the {cap_s:.0f} s cap"
        except Exception as exc:                       # noqa: BLE001
            return None, f"{type(exc).__name__}: {exc}"
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def _compose_groups(s, groups, buckets, deferred, fl, pens, held, parks, h_inv,
                    opts, leg_cache, leg_cache_root, dt, sub, jobs, cap_s,
                    verbose, mode, t0, bands="drop"):
    """`priority` and `serial`. -> (arms, report, seconds).  See `_conduct_set`."""
    order = group_order(groups, deferred, buckets, s)
    parks7 = {int(a): np.asarray(parks[a], float).reshape(7) for a in fl}
    held7 = {int(a): np.asarray(held[a], float).reshape(7) for a in fl}
    parts, lost, rooms, done = [], [], {}, set()
    offsets: dict[int, int] = {}
    if verbose:
        print(f"  stage {s} compose {mode}: "
              + " then ".join(str(sorted(g)) for g in order))
    if mode == "serial":
        # EVERY GROUP'S ROOM IS KNOWN BEFORE ANY OF THEM RUNS, so they still
        # plan at once: the groups before it have gone home (`PARK_HOME`), the
        # groups after it have not moved yet.
        pay = []
        for k, who in enumerate(order):
            op = {}
            for j, other in enumerate(order):
                if j == k:
                    continue
                for a in other:
                    op[int(a)] = parks7[int(a)] if j < k else held7[int(a)]
            pay.append(_group_payload(s, who, buckets, deferred, fl, pens,
                                      held7, parks, h_inv, opts, leg_cache,
                                      leg_cache_root, dt, sub, None, op,
                                      bands))
        import concurrent.futures as _fut
        import multiprocessing as _mp
        n = max(1, min(int(jobs), len(pay)))
        if n == 1:
            got = [(_row_job(p), None) for p in pay]
        else:
            ex = _fut.ProcessPoolExecutor(max_workers=n,
                                          mp_context=_mp.get_context("fork"))
            futs = [ex.submit(_row_job, p) for p in pay]
            got = []
            for f in futs:
                try:
                    got.append((f.result(timeout=max(1.0, float(cap_s))), None))
                except Exception as exc:               # noqa: BLE001
                    got.append((None, f"{type(exc).__name__}: {exc}"))
            ex.shutdown(wait=False, cancel_futures=True)
        m = 0
        for who, (part, why) in zip(order, got):
            if part is None:
                lost.append((who, why))
                continue
            parts.append(part)
            for a in who:
                offsets[int(a)] = m
            m += _slot_frames(part, dt) - 1
    else:
        for who in order:
            op = {int(a): (parks7[int(a)] if int(a) in done else held7[int(a)])
                  for g in order for a in g if int(a) not in set(who)}
            # An arm an earlier group MOVED is in the room as its whole realised
            # trajectory, not as either endpoint.
            op.update({int(a): held7[int(a)] for a in rooms})
            pay = _group_payload(s, who, buckets, deferred, fl, pens, held7,
                                 parks, h_inv, opts, leg_cache, leg_cache_root,
                                 dt, sub, dict(rooms), op, bands)
            part, why = _one_group(pay, cap_s)
            if part is None:
                lost.append((who, why))
                if verbose:
                    print(f"  stage {s} group {sorted(who)}: ABANDONED ({why})")
                continue
            parts.append(part)
            for a, st in part[0].items():
                done.add(int(a))
                if st.timeline is not None:
                    rooms[int(a)] = trajectory_room(st, fl, pens, h_inv, dt)
            if verbose:
                print(f"  stage {s} group {sorted(who)} -> room "
                      + ", ".join(f"{a}={room_size(rooms[a])}"
                                  for a in sorted(int(x) for x in who)
                                  if a in rooms))
    if verbose:
        for g in parts:
            print(f"  stage {s} group {sorted(int(a) for a in g[0])}: "
                  f"{sum(st.ink_m for st in g[0].values()):.3f} m, "
                  f"{g[2]:.1f} s wall, "
                  f"{1000 * float(g[1].get('min_clearance', np.nan)):+.1f} mm, "
                  + ("PASS" if g[1].get("ok") else "FAIL"))
    if not parts:
        return {}, dict(ok=False, empty=True, compose=mode,
                        min_clearance=float("inf"),
                        abandoned=[sorted(w) for w, _ in lost]), \
            time.perf_counter() - t0
    arms, rep = _merge_conducts(s, parts, fl, pens, held7, parks, h_inv, dt, sub,
                                offsets=(offsets if mode == "serial" else None))
    rep["compose"] = mode
    rep["group_order"] = [sorted(int(a) for a in g) for g in order]
    rep["pipeline_wall_s"] = float(sum(g[2] for g in parts)) \
        if mode == "priority" else float(max([0.0] + [g[2] for g in parts]))
    rep["abandoned"] = [sorted(int(a) for a in w) for w, _ in lost]
    rep["abandoned_why"] = [why for _, why in lost]
    return arms, rep, time.perf_counter() - t0


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
    # AN ARM THAT DOES NOT MOVE IS NOT AN ACTIVE PAIR, AND MEASURING IT AS ONE
    # MAKES THE ANSWER WORSE FOR NOTHING.  `active_pair_gap` takes the cross
    # product of two timelines and subtracts a 1-Lipschitz residual computed on
    # the DECIMATED grid, because a pair of asynchronous arms has no common
    # clock.  Against an arm that is standing still there is no cross product to
    # take: it is one pose, `solo_check` already measures the whole moving
    # timeline against it at the full rate AND refines a borderline verdict, and
    # `hold_gap` measures the held poses against each other.  Under the
    # leader/follower pattern six arms are named active and most of them may be
    # holding, so charging a still arm the moving one's decimation residual cost
    # 30 mm of a 50 mm gate on the first CSAIL run and measured nothing.
    conc = {a: st for a, st in sr.arms.items()
            if not st.residue and st.timeline is not None}
    sr.pair = (active_pair_gap(conc, fl, pens, h_inv, dt, max_check_poses)
               if len(conc) > 1
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
    if sr.conducted:
        c = sr.conducted_check
        print(f"stage {sr.stage}: CONDUCTED  arms "
              + ", ".join(f"{a}({len(st.accepted)})"
                          for a, st in sorted(sr.arms.items()) if st.accepted)
              + f"  {sr.n_pieces} pieces  {sr.ink_m:.3f} m  "
              f"{sr.duration:.1f} s"
              + ("  (empty)" if c.get("empty") else
                 f"  whole-timeline {1000 * float(c.get('min_clearance', np.nan)):+.1f} mm")
              + ("  serialised" if c.get("serialised") else "")
              + (f"  failed: {','.join(c['failed'])}" if c.get("failed") else "")
              + ("" if sr.complete else "  incomplete")
              + f"  [{'PASS' if sr.ok else 'FAIL'}]")
        return
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
                   conducted=bool(sr.conducted),
                   roles={str(a): r for a, r in sr.roles.items()},
                   deferred_m=float(sr.deferred_m),
                   checks=dict(active_pair=_jsonable(sr.pair),
                               conducted=dict(
                                   ok=bool(sr.conducted_check.get("ok", False)),
                                   empty=bool(sr.conducted_check.get("empty",
                                                                     False)),
                                   min_clearance_m=float(
                                       sr.conducted_check.get("min_clearance",
                                                              np.nan)))
                               if sr.conducted else None,
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
                # `q_park` IS THE POSE THE STAGE STARTS FROM, which is the park
                # in stage A and under the zigzag and is the pose the previous
                # stage HELD otherwise; `q_hold` is where this stage leaves the
                # arm.  The name is kept because it is the field the animation
                # reads for "where this arm stands in this stage".
                arm=int(a), q_park=[float(x) for x in np.asarray(st.q_park).ravel()],
                q_hold=[float(x) for x in np.asarray(st.q_end).ravel()],
                q_tuck=(None if st.q_tuck is None else
                        [float(x) for x in np.asarray(st.q_tuck).ravel()]),
                clear_out_s=float(st.clear_out_s),
                # WHAT KIND OF POSE THE BARRIER IS HOLDING.  "hover" is the
                # ordinary freeze; "hover_z", "hover_prev" and "park" are
                # `writing.hold_candidates`' three retreats, taken where the
                # hover the stage would have stopped on is not a pose a barrier
                # may hold (docs/V2_STAGED.md §29).
                hold_kind=(None if tl is None else str(tl.get("hold_kind", ""))),
                frozen_partners=[int(x) for x in st.frozen_partners],
                room_kind=str(st.room_kind),
                residue=bool(st.residue),
                role=str(st.role), conducted=bool(st.conducted),
                deferred=[dict(line=int(p.line), piece=int(p.k),
                               length_m=float(p.length_m))
                          for p in st.deferred],
                deferred_m=float(sum(p.length_m for p in st.deferred)),
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
            residue_m=round(sr.residue_m, 4),
            residue_arms=[int(a) for a, st in sorted(sr.arms.items())
                          if st.residue],
            orders_tried=int(sr.orders_tried),
            buckets_flown=int(sum(1 for st in sr.arms.values()
                                  if st.timeline is not None)),
            buckets_with_ink=int(sum(1 for st in sr.arms.values()
                                     if st.accepted)),
            conducted=bool(sr.conducted),
            roles={str(a): r for a, r in sr.roles.items()},
            conducted_min_mm=(None if not sr.conducted else
                              (None if sr.conducted_check.get("empty") else
                               round(1000 * float(sr.conducted_check.get(
                                   "min_clearance", np.nan)), 2))),
            conducted_failed=(sr.conducted_check.get("failed")
                              if sr.conducted else None),
            conducted_serialised=bool(sr.conducted_check.get("serialised")),
            conducted_groups=(sr.conducted_check.get("groups")
                              if sr.conducted else None),
            conducted_abandoned=(sr.conducted_check.get("abandoned")
                                 if sr.conducted else None),
            partition=(sr.conducted_check.get("partition")
                       if sr.conducted else None),
            split={str(a): st.split for a, st in sorted(sr.arms.items())
                   if st.split and st.split.get("parents")},
            standoff={str(a): {str(p): round(float(v), 4)
                               for p, v in sorted(st.standoff.items())}
                      for a, st in sorted(sr.arms.items()) if st.standoff},
            deferred_m=round(sr.deferred_m, 4),
            deferred_in_m=round(sr.deferred_in, 4),
            deferred_arms={str(a): round(float(sum(p.length_m
                                                   for p in st.deferred)), 4)
                           for a, st in sorted(sr.arms.items()) if st.deferred},
            leader_ink_m=round(sr.role_ink("leader"), 4),
            leader_ink_flown_m=round(sr.role_ink("leader", True), 4),
            follower_ink_m=round(sr.role_ink("follower"), 4),
            follower_ink_flown_m=round(sr.role_ink("follower", True), 4),
            follower_ink_deferred_m=round(float(sum(
                p.length_m for st in sr.arms.values() if st.role == "follower"
                for p in st.deferred)), 4),
            conduct_s=round(sr.conduct_s, 2),
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
                partner_standoff_m=round(float(res.partner_standoff), 4),
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
                roles=role_summary(res),
                holds=[dict(before_stage=int(h["before_stage"]),
                            min_mm=round(1000 * float(h["min_m"]), 2),
                            # ...AND WHAT EACH HELD POSE IS ON ITS OWN, which is
                            # the half of the barrier `validate_pose` judges and
                            # the half that failed (docs/V2_STAGED.md §29).
                            poses_ok=bool(h.get("poses_ok", True)),
                            bad_poses=[int(a) for a in
                                       (h.get("bad_poses") or [])],
                            joint_margin_min=round(min(
                                [float(v) for v in
                                 (h.get("joint_margin") or {}).values()]
                                or [float("nan")]), 4),
                            ok=bool(h["ok"])) for h in res.holds],
                holds_ok=bool(all(h["ok"] for h in res.holds)),
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


def role_summary(res: StagedResult) -> dict:
    """The number Pete's design hinges on: how much FOLLOWER ink fitted.

    A follower is offered a bag and keeps what fits the leaders' rooms; the
    fraction it keeps is the whole question the pattern asks, per stage, because
    a follower that keeps nothing is a parked partner with extra steps.  The
    deferred metres and what the conducted final pass cost to absorb them are
    the other side of the same ledger.
    """
    def ink_mm(stages_, role):
        """The clearance every piece of one role stood at, from the room. -> dict.

        THE FIT FRACTION ON ITS OWN DOES NOT SAY WHY.  A follower that keeps
        nothing has either been refused at the gate (its ink passes through the
        occupied volume) or been unable to route a leg, and those are different
        findings with different fixes.  The clearance distribution separates
        them: a piece at -200 mm is inside the leader's trajectory and no gate
        setting saves it; a piece at +45 mm is one the gate and nothing else
        refused.
        """
        v = sorted(float(x) for sr_ in stages_ for st in sr_.arms.values()
                   if st.role == role for x in st.ink_clearance)
        if not v:
            return None
        q = np.percentile(np.asarray(v), [5, 25, 50, 75, 95])
        return dict(n=len(v), min_mm=round(1000 * v[0], 2),
                    p05_mm=round(1000 * float(q[0]), 2),
                    p25_mm=round(1000 * float(q[1]), 2),
                    median_mm=round(1000 * float(q[2]), 2),
                    p75_mm=round(1000 * float(q[3]), 2),
                    p95_mm=round(1000 * float(q[4]), 2),
                    max_mm=round(1000 * v[-1], 2),
                    under_gate=int(sum(1 for x in v if x < PAIR_MARGIN)))

    per, total = {}, dict(leader_m=0.0, leader_flown_m=0.0, follower_m=0.0,
                          follower_flown_m=0.0, deferred_m=0.0,
                          conducted_m=0.0, conducted_s=0.0)
    for sr in res.stages:
        if sr.conducted:
            total["conducted_m"] += float(sr.ink_m)
            total["conducted_s"] += float(sr.duration)
            continue
        fdef = float(sum(p.length_m for st in sr.arms.values()
                         if st.role == "follower" for p in st.deferred))
        off = sr.role_ink("follower", True) + fdef
        per[str(sr.stage)] = dict(
            leader_m=round(sr.role_ink("leader"), 4),
            leader_flown_m=round(sr.role_ink("leader", True), 4),
            follower_offered_m=round(off, 4),
            follower_flown_m=round(sr.role_ink("follower", True), 4),
            follower_fit_frac=(round(sr.role_ink("follower", True) / off, 4)
                               if off > 1e-9 else None),
            deferred_m=round(sr.deferred_m, 4),
            follower_ink_clearance=ink_mm([sr], "follower"),
            busiest_arm_s=round(max((st.duration
                                     for st in sr.arms.values()), default=0.0), 3))
        total["leader_m"] += sr.role_ink("leader")
        total["leader_flown_m"] += sr.role_ink("leader", True)
        total["follower_m"] += off
        total["follower_flown_m"] += sr.role_ink("follower", True)
        total["deferred_m"] += sr.deferred_m
    total = {k: round(float(v), 4) for k, v in total.items()}
    total["follower_fit_frac"] = (round(total["follower_flown_m"]
                                        / total["follower_m"], 4)
                                  if total["follower_m"] > 1e-9 else None)
    main = [sr for sr in res.stages if not sr.conducted]
    return dict(per_stage=per, total=total, split=split_summary(res),
                follower_ink_clearance=ink_mm(main, "follower"),
                leader_ink_clearance=ink_mm(main, "leader"))


def split_summary(res: StagedResult) -> dict:
    """What cutting the pieces at the room boundary bought. -> dict.

    `parents` pieces went in, `parts` clear stretches came out, `kept_m` of
    their ink was flown and `dropped_m` deferred.  `pieces_before` /
    `pieces_after` is the pen-up bill: every extra piece is an entry hover, an
    exit hover and a leg between them.
    """
    out = dict(parents=0, parts=0, kept_m=0.0, dropped_m=0.0,
               hover_refused=0, per_stage={})
    for sr in res.stages:
        if sr.conducted:
            continue
        one = dict(parents=0, parts=0, kept_m=0.0, dropped_m=0.0,
                   hover_refused=0, pieces=0, part_pieces=0)
        for st in sr.arms.values():
            d = st.split or {}
            for k in ("parents", "parts", "hover_refused"):
                one[k] += int(d.get(k, 0))
                out[k] += int(d.get(k, 0))
            for k in ("kept_m", "dropped_m"):
                one[k] = round(one[k] + float(d.get(k, 0.0)), 4)
                out[k] = round(out[k] + float(d.get(k, 0.0)), 4)
            one["pieces"] += len(st.accepted)
            one["part_pieces"] += sum(1 for p in st.accepted
                                      if p.piece.k >= SPLIT_ID0)
        out["per_stage"][str(sr.stage)] = one
    return out


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
# 6c.  THE FINAL PASS, RE-RUN OFF A PROGRAMME
# ---------------------------------------------------------------------------
# WHY THIS EXISTS.  Stages A and B cost 1 980 s of planning wall on the CSAIL
# logo and stage C is the only stage the composition question is about, so
# measuring three compositions against each other must not mean three whole
# programmes.  Everything the final pass needs is already written down: the
# stage's entry pose set is the barrier's `holds`, and the ink it was offered is
# the pieces the conducted stages actually carry -- each with its own polyline,
# in paper metres, which is exactly what a `Piece` is.
#
# WHAT IT CANNOT RECOVER, STATED RATHER THAN HIDDEN.  A piece the conduct
# REFUSED is written out without its geometry (`programme()` records a refusal
# by id, status and reason), so a re-run is offered the accepted set.  On the
# lf3 s150 programme that is 1 piece of 50, and a piece `plan_stroke` refused
# once it refuses again, so nothing that flew is missing.
def pieces_from_programme(doc: dict, stage: int, also=()) -> dict:
    """The ink a conducted stage was offered. -> {arm: [Piece]}."""
    out: dict[int, list] = {}
    want = {int(stage)} | {int(s) for s in also}
    for one in doc.get("stages", []):
        if int(one.get("stage", -1)) not in want:
            continue
        for a, v in (one.get("arms") or {}).items():
            for p in v.get("pieces") or []:
                pts = np.asarray(p["pts"], float).reshape(-1, 2)
                out.setdefault(int(a), []).append(
                    Piece(stage=int(stage), arm=int(a), line=int(p["line"]),
                          k=int(p["piece"]), pts=pts,
                          length_m=float(p.get("length_m",
                                               traces_mod.cumlen(pts)[-1]))))
    return out


def holds_from_programme(doc: dict, stage: int) -> dict:
    """The pose set a stage starts from. -> {arm: q}."""
    for b in doc.get("barriers", []):
        if b.get("before_stage") is not None \
                and int(b["before_stage"]) == int(stage) and b.get("holds"):
            return {int(a): np.asarray(q, float).reshape(7)
                    for a, q in b["holds"].items()}
    raise SystemExit(f"no barrier with holds before stage {stage}")


def run_conducted(doc: dict, stage: int = 2, band_stage: int | None = None,
                  specs=None, pens=None, parks=None, h_inv=H_INV_DEFAULT,
                  opts=None, leg_cache=True, leg_cache_root=None,
                  dt=CHECK_DT, sub=2, conduct_jobs=3,
                  conduct_cap_s=CONDUCT_CAP_S, row_compose="priority",
                  verbose=True) -> StagedResult:
    """Re-run ONE conducted stage (and its dead-band stage) off a programme."""
    t_all = time.perf_counter()
    fl = FLEET if specs is None else specs
    pens = {a: fl[a].pen for a in fl} if pens is None else pens
    parks = shipped_parks(fl) if parks is None else parks
    band_stage = (stage + 1) if band_stage is None else band_stage
    held = holds_from_programme(doc, stage)
    # THE PARTITION IS NOT RE-TAKEN, IT IS READ BACK.  `partition_deferred` ran
    # once in the original programme and its answer is written into that
    # programme's own stages: what the row conductors took is stage C's pieces
    # and what the dead band took is stage D's.  Re-partitioning here would ask
    # `piece_row` about the stage's DP bucket as well, which never went through
    # it, and would move ink between the stages the comparison is about.
    offered = pieces_from_programme(doc, stage)
    band = pieces_from_programme(doc, band_stage)
    buckets = {(int(stage), int(a)): list(v) for a, v in offered.items()}
    row_m = sum(p.length_m for v in offered.values() for p in v)
    band_m = sum(p.length_m for v in band.values() for p in v)
    pstats = dict(row_m={str(traces_mod.ROW_OF.get(int(a))): float(
        sum(p.length_m for p in v)) for a, v in sorted(offered.items())},
        band_m={str(int(a)): float(sum(p.length_m for p in v))
                for a, v in sorted(band.items())},
        total_m=float(row_m + band_m),
        band_frac=float(band_m / max(row_m + band_m, 1e-12)))
    if verbose:
        print(f"  stage {stage} offered {row_m:.3f} m in "
              f"{sum(len(v) for v in offered.values())} pieces; dead band "
              f"{band_m:.3f} m in {sum(len(v) for v in band.values())} pieces "
              f"({100 * pstats['band_frac']:.1f} %)")
    out, holds, plan_s = [], [], 0.0
    entry = {int(a): np.asarray(q, float).reshape(7) for a, q in held.items()}
    holds.append(dict(before_stage=int(stage),
                      q={str(a): [float(x) for x in q]
                         for a, q in sorted(entry.items())},
                      **hold_gap(entry, fl, pens, h_inv)))
    t1 = time.perf_counter()
    arms, chk, c_s = _conduct_groups(
        stage, [ROW_ARMS[j] for j in sorted(ROW_ARMS)], buckets, {}, fl, pens,
        entry, parks, h_inv, opts, leg_cache, leg_cache_root, dt, sub,
        conduct_jobs, conduct_cap_s, verbose, mode=row_compose)
    chk["partition"] = {str(k): v for k, v in pstats.items()}
    sr = StageResult(int(stage), tuple(sorted(int(a) for a in fl)), arms,
                     wall_s=time.perf_counter() - t1, conducted=True,
                     conducted_check=chk,
                     deferred_in=float(pstats["total_m"]), conduct_s=float(c_s))
    plan_s += time.perf_counter() - t1
    out.append(sr)
    if verbose:
        _report_stage(sr)
    for a, st in sr.arms.items():
        entry[int(a)] = np.asarray(st.q_end, float).reshape(7)
    if band:
        who = sorted(int(a) for a, v in band.items() if v)
        holds.append(dict(before_stage=int(band_stage),
                          q={str(a): [float(x) for x in q]
                             for a, q in sorted(entry.items())},
                          **hold_gap(entry, fl, pens, h_inv)))
        t1 = time.perf_counter()
        arms, chk, c_s = _conduct_groups(
            band_stage, [tuple(who)], {}, band, fl, pens, entry, parks, h_inv,
            opts, leg_cache, leg_cache_root, dt, sub, 1, conduct_cap_s, verbose)
        sd = StageResult(int(band_stage), tuple(sorted(int(a) for a in fl)),
                         arms, wall_s=time.perf_counter() - t1, conducted=True,
                         conducted_check=chk,
                         deferred_in=float(sum(p.length_m for v in band.values()
                                               for p in v)),
                         conduct_s=float(c_s))
        plan_s += time.perf_counter() - t1
        out.append(sd)
        if verbose:
            _report_stage(sd)
    thaw()
    from . import sequence as _sequence
    _sequence.close_pool()
    return StagedResult(str(doc.get("pattern", "")) + f"/stage{stage}-only",
                        out, [p for v in offered.values() for p in v],
                        0.0, plan_s, 0.0, time.perf_counter() - t_all, None,
                        {int(a): [float(x) for x in np.asarray(q).ravel()]
                         for a, q in parks.items()},
                        holds=holds)


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
    ap.add_argument("--no-residue", action="store_true",
                    help="do not serialise a bucket no order could fly "
                         "concurrently; leave it stranded instead")
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
    ap.add_argument("--pattern", choices=("zigzag", "leader_follower"),
                    default="zigzag",
                    help="'zigzag': the eight-stage pattern of "
                         "docs/V2_WORKCELLS.md section 4b, one arm per row with "
                         "its same-row partner PARKED.  'leader_follower': "
                         "Pete's specification (docs/V2_STAGED.md section 22) -- "
                         "six arms in every main stage, leaders 13/71/2 planned "
                         "with priority, followers 17/31/97 planned against the "
                         "leaders' realised trajectories, and one conducted "
                         "final pass for everything that did not fit")
    ap.add_argument("--split-m", type=float, default=0.0,
                    help="leader_follower only: how far OUTWARD from each arm's "
                         "own base column its bag divides into leader ink "
                         "(toward the mid-line) and follower ink (its outer "
                         "strip).  Larger means more leader ink and a narrower, "
                         "safer follower strip")
    ap.add_argument("--whole-bag", action="store_true",
                    help="leader_follower only: PETE'S LITERAL BASELINE -- no "
                         "split at all.  Every arm is offered its WHOLE cell in "
                         "both roles, so each arm's bag lands in the stage it "
                         "plans first in and what a follower cannot fit is "
                         "deferred to the stage where that same arm leads")
    ap.add_argument("--no-follower-gate", action="store_true",
                    help="leader_follower only: measure a follower's ink "
                         "against the leaders' rooms instead of refusing it")
    ap.add_argument("--refusal-rounds", type=int, default=REFUSAL_ROUNDS)
    ap.add_argument("--env-stride", type=int, default=ENVELOPE_STRIDE)
    ap.add_argument("--route-jobs", type=int, default=6,
                    help="processes the pen-up route screen may fork "
                         "(sequence.ROUTE_JOBS; 0 = the machine decides)")
    ap.add_argument("--no-tuck", action="store_true",
                    help="leader_follower only: do NOT let a follower "
                         "pre-position before its leaders start.  The shipped "
                         "park can be deep inside the same-row leader's room "
                         "(-128.4 mm on arm 31, measured), which clamps every "
                         "leg's static floor negative before routing begins; "
                         "this turns the clear-out off to reproduce that.")
    ap.add_argument("--split-rounds", type=int, default=SPLIT_ROUNDS,
                    help="leader_follower only: how many times a piece the "
                         "room ink gate refuses may be CUT at the room "
                         "boundary and its certified clear stretches offered "
                         "back (0 = the pre-2026-09-14 whole-piece refusal)")
    ap.add_argument("--split-min-m", type=float, default=SPLIT_MIN_M,
                    help="the shortest clear stretch a cut may keep")
    ap.add_argument("--no-conduct-rows", action="store_true",
                    help="leader_follower only: run the final pass as ONE "
                         "six-arm conduct (the pre-2026-09-14 behaviour, "
                         "3 075 s of wall on the CSAIL logo) instead of three "
                         "two-arm conductors, one per row, in parallel")
    ap.add_argument("--partner-standoff", type=float, default=0.0,
                    help="leader_follower only: metres of clearance a LEADER "
                         "must keep, ON TOP OF every gate it already meets, "
                         "from its SAME-ROW follower's pose-invariant capsules "
                         "(base column + shoulder->elbow).  No gate constant "
                         "moves; 0.0 (the default) is exactly the pre-existing "
                         "behaviour.  The leader loses the ink that needs to "
                         "reach in under its partner's shoulder and that ink "
                         "is deferred as usual (docs/V2_STAGED.md section 25)")
    ap.add_argument("--conduct-jobs", type=int, default=3,
                    help="row conductors run at once (<= 3)")
    ap.add_argument("--conduct-cap-s", type=float, default=CONDUCT_CAP_S,
                    help="wall-clock seconds any one conduct may take before "
                         "it is abandoned and its ink reported as residue")
    ap.add_argument("--row-compose", choices=ROW_COMPOSE, default="priority",
                    help="how the row conductors COMPOSE (docs/V2_STAGED.md "
                         "section 26).  'parallel': every group starts at stage "
                         "time zero and plans with the others STANDING at their "
                         "entry poses -- the pre-2026-09-14 behaviour, and the "
                         "one whose merge failed at -191.9 mm.  'priority' "
                         "(default): busiest group first, each later group "
                         "conducted against the earlier groups' REALISED "
                         "trajectories as exact swept rooms; the wall becomes a "
                         "pipeline.  'serial': the groups run one after another "
                         "in TIME and still plan at once; the motion becomes "
                         "the sum of the rows")
    ap.add_argument("--stage-c-only", default=None, metavar="PROGRAMME.json",
                    help="re-run ONLY the conducted final pass (and its "
                         "dead-band stage) off an existing programme JSON: the "
                         "stage entry poses come from its barrier and the ink "
                         "from the pieces it carries.  Stages A and B are not "
                         "re-planned")
    ap.add_argument("--stage-c-index", type=int, default=2,
                    help="--stage-c-only: which stage of that programme is the "
                         "conducted final pass")
    ap.add_argument("--json", default=None)
    ap.add_argument("--programme", default=None)
    a = ap.parse_args(argv)

    if a.stage_c_only:
        doc = json.loads(Path(a.stage_c_only).read_text())
        print(f"=== {Path(a.stage_c_only).name}: stage {a.stage_c_index} only, "
              f"compose {a.row_compose} ===")
        from . import sequence as _sequence
        if a.route_jobs is not None:
            _sequence.ROUTE_JOBS = int(a.route_jobs)
        res = run_conducted(doc, stage=a.stage_c_index,
                            opts=dict(tilt_max_deg=a.tilt_max_deg),
                            leg_cache=not a.no_leg_cache,
                            leg_cache_root=a.leg_cache, dt=a.dt, sub=2,
                            conduct_jobs=a.conduct_jobs,
                            conduct_cap_s=a.conduct_cap_s,
                            row_compose=a.row_compose)
        d = summary(res)
        d["compose"] = a.row_compose
        d["source_programme"] = str(a.stage_c_only)
        d["stage_ab_s"] = round(float(sum(
            float(one.get("duration_s", 0.0))
            for one in doc.get("stages", [])
            if int(one.get("stage", -1)) < int(a.stage_c_index))), 3)
        d["makespan_abcd_s"] = round(d["stage_ab_s"] + res.makespan, 3)
        print(json.dumps(d, indent=1))
        if a.json:
            Path(a.json).write_text(json.dumps(d, indent=1))
            print("wrote", a.json)
        if a.programme:
            Path(a.programme).write_text(json.dumps(programme(res)))
            print("wrote", a.programme)
        return 0

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
    pat = (traces_mod.leader_follower_pattern(split_m=a.split_m,
                                              whole_bag=a.whole_bag)
           if a.pattern == "leader_follower" else traces_mod.zigzag_pattern())
    print(f"=== {name}: {len(lines)} lines, "
          f"{sum(float(traces_mod.cumlen(np.asarray(p, float))[-1]) for p in lines):.2f} m ===")
    print(f"=== pattern {pat.name}: {pat.n_stages} stages ===")
    res = run(lines, pattern=pat, coverage=cov,
              opts=dict(tilt_max_deg=a.tilt_max_deg),
              stages=stages, leg_cache=not a.no_leg_cache,
              leg_cache_root=a.leg_cache, check=not a.no_check,
              fly=not a.no_fly, dt=a.dt, route_jobs=a.route_jobs,
              envelopes=not a.no_envelopes, atlas_dir=a.atlas,
              refusal_rounds=a.refusal_rounds,
              lift_retry=not a.no_lift_retry,
              env_kw=dict(stride=a.env_stride),
              trajectory_rooms=a.trajectory_rooms,
              room_iterations=a.room_iterations,
              room_order=a.room_order, order_search=a.order_search,
              residue=not a.no_residue, sub=2,
              follower_ink_gate=(None if a.no_follower_gate else PAIR_MARGIN),
              tuck=not a.no_tuck, split_rounds=a.split_rounds,
              split_min_m=a.split_min_m,
              conduct_rows=not a.no_conduct_rows,
              conduct_jobs=a.conduct_jobs, conduct_cap_s=a.conduct_cap_s,
              row_compose=a.row_compose,
              partner_standoff=a.partner_standoff)
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
