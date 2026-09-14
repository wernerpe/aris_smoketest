"""A POSE-AWARE neighbour model, for partners that are known to be frozen.

OFF BY DEFAULT.  Nothing in this module runs until `freeze()` is called, and
`thaw()` restores the shipped behaviour exactly.

WHAT IT IS FOR.  `mounts.attach_body_columns` gives every arm the OTHER arms'
pose-invariant BASE COLUMN BANDS: a 0.32 m AABB per band standing in for a
neighbour's shoulder and upper links AT ANY POSE, because at atlas-sweep time
nobody has decided what pose the neighbour will hold.  That is the honest model
when a neighbour might be moving, and it is the obstacle that actually blocks
pen-ups on the all-ceiling rig.

MEASURED (2026-09-09, cell (0.60, 1.82) at h = 0.970): arm 71 is the only arm
that can draw that cell, and every one of its 8 pen-up legs is refused against
`body:31_column3` — while arm 31's ACTUAL parked capsules clear arm 71's route
by 127 mm.  The band refuses what the arm does not.  When the neighbour is
frozen at a certified park — which is exactly what the solo feasibility map
assumes and what the conductor's phases enforce (`scene_check` reports
"frozen 6/6") — the band is over-conservative by construction.

WHAT IT REPLACES, AND WHAT IT KEEPS.  Only the `body:<aid>_column<k>` bands of
arms named in the frozen set are dropped, and they are replaced by that arm's
real link capsules at its real pose, checked with the same floors the bands
were checked at.  TRUE STRUCTURE IS NEVER DROPPED: mounts, plates, the drop
cluster, the runway, and the bands of any partner NOT in the frozen set all
stay exactly as they were.

WHAT IT COSTS IN CERTIFICATION.  A cell certified with this on depends on the
NAMED NEIGHBOUR HOLDING THE NAMED POSE for the whole stroke and both its
pen-up legs.  That is a real obligation on the conductor and it must be
recorded with the result — `scripts/certified_area.py` writes it into the
JSON's `frozen_dependency` block.  It is not a free 127 mm.
"""
import re

import numpy as np

from . import coordination, exact_room, link_spheres, rig_final

# {aid: (A (C,3), B (C,3), R (C,))} of every frozen partner's world capsules
_CAPS = {}
_POSES = {}
_OBSERVER = None

_BAND = re.compile(r"^body:(\d+)_column\d+$")

# ==========================================================================
# THE PARTNER STANDOFF — an EXTRA requirement, never a gate change
# ==========================================================================
# WHAT IT IS FOR (docs/V2_STAGED.md §25).  In the leader/follower pattern the
# LEADER plans first and is certified against its same-row partner's held pose
# at exactly `PAIR_MARGIN` / `STATIC_MARGIN`, because it has no reason to keep
# more.  Then the FOLLOWER has to route its own pen-up legs in what is left,
# and `paper.route`'s floor is `paper.FRAME_FLOOR` (63 mm) — higher than the
# 50 mm pose gate the leader was held to.  Measured on CSAIL stage A: every
# pose arm 31 can hold near its own ink stands +53.7 mm from leader 71's exact
# room, at every hover rung and over every x-y, which is legal to stand in and
# impossible to route through.  The number is CONSTANT in the follower's pose,
# so the binding geometry is the follower's links that do not move.
#
# MEASURED, which set those are.  Leader 71's realised stage-A trajectory
# against follower 31's capsules at its held pose, per capsule (mm of surface
# gap): base bands 207.0 / 243.8 / 192.2, **upper arm (chain 1->3) 103.2**,
# elbow 314.2, forearm 331.4, wrist 613.3, hand 560.1, tool 519.8.  The
# binding pair is the leader's FOREARM against the follower's UPPER ARM, whose
# shoulder end (chain point 1) is where the base column ends and does not move
# whatever the partner's joints do.  So the pose-invariant set carried here is
# every capsule both of whose chain endpoints are in {0, 1, 3} — the four base
# column bands (0->1) and the shoulder->elbow link (1->3), which is "base
# column + link 0/1".  The elbow link (3->4) is 3x further away and is not in
# it.
#
# WHAT IT IS NOT.  No gate constant moves: `PAIR_MARGIN`, `STATIC_MARGIN` and
# `FRAME_FLOOR` are untouched, and `S = 0` reproduces every earlier number
# exactly (`tests/test_staged_standoff.py`).  `S` is an ADDITIONAL requirement
# on ONE named partner's pose-invariant capsules: `partner_clearance` returns
# `min(gap_everywhere, gap_to_that_set - S)`, so every call site that compares
# the result against its own floor `f` is thereby demanding `f + S` of that
# set and `f` of everything else.  The leader loses the ink that needs to
# reach in under its partner's shoulder; deferral already knows what to do
# with it.
STANDOFF_POINTS = (0, 1, 3)   # chain points a POSE-INVARIANT capsule spans

_STANDOFF = {}      # {aid: S metres} — extra clearance, that partner's set only
_INVARIANT = {}     # {aid: bool mask over that partner's capsule rows}


def invariant_mask(tab, n_poses=1):
    """Which rows of a partner's flat capsule block are pose-invariant.

    `tab` is the capsule table the block was built from and `n_poses` how many
    times it was tiled (`freeze_sets` stacks pose-major, which is what
    `np.tile` on the radii already assumes).  -> (len(tab) * n_poses,) bool.
    """
    row = np.array([bool(c[0] in STANDOFF_POINTS and c[1] in STANDOFF_POINTS)
                    for c in tab], bool)
    return np.tile(row, int(n_poses)) if int(n_poses) != 1 else row


def set_standoff(mapping):
    """Demand S metres MORE of these partners' pose-invariant capsules.

    `mapping` is {aid: S}; anything falsy clears it.  Call it AFTER `freeze` /
    `freeze_sets`, which reset it — so a caller that does not ask for a
    standoff cannot inherit one from the previous bucket.
    """
    global _STANDOFF
    _STANDOFF = {int(a): float(s) for a, s in (mapping or {}).items()
                 if float(s) > 0.0}


def standoff():
    """The live standoff. -> {aid: S}."""
    return dict(_STANDOFF)


def standoff_sig():
    """A stable string for the memo and leg-cache keys. -> str.

    EMPTY WHEN THERE IS NO STANDOFF, so every key an S = 0 run builds is the
    key it built before this existed and a warm leg store still answers.
    """
    return "".join(f"{a}:{_STANDOFF[a]:.6f};" for a in sorted(_STANDOFF))


# ==========================================================================
# KEEPING THE BANDS — the judge's frame gate does not know about this module
# ==========================================================================
# MEASURED 2026-09-14 (docs/V2_STAGED.md §28), stage D of the serial programme.
# Arm 31's dead-band conduct routes with the other five arms frozen, so
# `filter_boxes` drops `body:71_column3` and replaces it by arm 71's real parked
# capsules — the whole point of this module, and 127 mm of honest room on this
# rig.  `scene_check`'s FRAME gate re-derives everything from the timeline and
# knows nothing about any of it: it measures every arm against
# `spec.static_obstacles()`, band AABBs included, and refused that conduct at
# +31.0 mm while its own COLUMN gate — the same metal, written as cylinders
# rather than as the 0.32 m box that contains them — passed the identical
# trajectory at +84.4 mm.  The gate did not change and the route did.
#
# SO A CALLER MAY ASK FOR BOTH.  `set_keep_bands(True)` keeps every band in the
# static room AND keeps the pose-aware partner term, which is strictly more
# conservative than either half: it is the room the conduct flew in before
# `freeze_conduct` existed, plus the partner bodies `freeze_conduct` added.  It
# can only refuse legs, never admit one, so it needs no separate certificate —
# it costs routing room, and `staged.CONDUCT_BANDS` spends it only where the
# relaxed room produced something the judge refuses.
_KEEP_BANDS = False


def set_keep_bands(flag):
    """Keep the frozen partners' band AABBs in the static room as well.

    Call it AFTER `freeze` / `freeze_sets`, which reset it — so a caller that
    does not ask for the bands cannot inherit them from the previous bucket.
    """
    global _KEEP_BANDS
    _KEEP_BANDS = bool(flag)


def keep_bands():
    """Are the frozen partners' bands being kept as well? -> bool."""
    return bool(_KEEP_BANDS)


def keep_bands_sig():
    """A stable string for the memo and leg-cache keys. -> str.

    EMPTY when the bands are dropped, so every key the shipped behaviour builds
    is the key it built before this existed and a warm leg store still answers.
    """
    return "bands;" if _KEEP_BANDS else ""


def band_owner(name):
    """The arm a pose-invariant body band belongs to. -> int | None.

    `None` for every box that is NOT such a band — mounts, plates, the drop
    cluster, the runway — which is what keeps true structure out of this.
    """
    m = _BAND.match(str(name))
    return int(m.group(1)) if m else None


def active():
    """Is the pose-aware model switched on? -> bool."""
    return bool(_CAPS)


def frozen_ids():
    """The arms currently modelled by their actual pose. -> sorted list."""
    return sorted(_CAPS)


def poses():
    """{aid: q} of the frozen set, for provenance. -> dict."""
    return {a: np.asarray(q, float).copy() for a, q in _POSES.items()}


def freeze_sets(sets, fleet, pens, h_inv, clusters=None):
    """Model these arms by the UNION of their capsules over a SET of poses.

    `freeze` is the N = 1 case of this and nothing else: a PARKED partner holds
    one pose, so its obstacle is that pose's capsules.  An ACTIVE partner in the
    same stage holds no single pose — it holds an ENVELOPE, the union over every
    pose it could take anywhere inside its work cell — and that union is the
    object `docs/ARCHITECTURE_V2.md` section 2f says a pen-up leg has to be
    routed against.  The capsule block is the same shape either way, so nothing
    downstream (`filter_boxes`, `partner_clearance`, `chain_clearance`) changes.

    `sets` is {aid: q} or {aid: Q (N, 7)}.  `clusters` is an optional
    {aid: (centres (M, 3), radii (M,))} of BOUNDING SPHERES that already contain
    that arm's capsules — a caller that has reduced a 9 000-capsule envelope to
    a few hundred spheres hands them in here rather than paying the full block
    on every query (see `staged.cluster_capsules`).  A sphere is a degenerate
    capsule (A == B), which is all the adapting this needs.

    ...OR AN `exact_room.ExactRoom`, which is the same object without the
    reduction: the partner's real swept capsules behind a grid-hash prune.  The
    sphere cloud cost a median 195 mm of clearance against the capsules it
    contained (docs/V2_STAGED.md section 23), so the room the follower is
    certified against is the exact one by default and the spheres are kept only
    for the A/B (`ARIS_ROOM=spheres`).  Both arrive here and both leave through
    `partner_clearance`, which is the only seam either of them has.
    """
    global _CAPS, _POSES, _INVARIANT
    _CAPS, _POSES, _INVARIANT = {}, {}, {}
    set_standoff(None)
    set_keep_bands(False)
    for aid, Q in (sets or {}).items():
        aid = int(aid)
        if aid not in fleet:
            continue
        Q = np.asarray(Q, float).reshape(-1, 7)
        cl = (clusters or {}).get(aid)
        if cl is not None and isinstance(cl[0], exact_room.ExactRoom):
            # `(room, radii, digest)` — `staged.trajectory_room`'s tuple, whose
            # first slot is the room itself rather than a cloud of centres
            _CAPS[aid] = cl[0]
            _POSES[aid] = Q[0].copy()
            continue
        if cl is not None:
            C = np.asarray(cl[0], float).reshape(-1, 3)
            R = np.asarray(cl[1], float).reshape(-1)
            _CAPS[aid] = (C, C.copy(), R)
            _POSES[aid] = Q[0].copy()
            continue
        P = coordination.chain_world(Q, fleet[aid], h_inv,
                                     float(pens.get(aid, 0.110)))
        tab = (coordination.CAPSULES_LAT if P.shape[1] >= 11
               else coordination.CAPSULES)
        A, B = coordination.cap_endpoints(P, tab)
        tab, keep = coordination.known_pose_capsules(tab)
        A = np.asarray(A, float)[:, keep].reshape(-1, 3)
        B = np.asarray(B, float)[:, keep].reshape(-1, 3)
        R = np.tile(np.array([c[2] for c in tab], float), len(Q))
        _CAPS[aid] = (A, B, R)
        _INVARIANT[aid] = invariant_mask(tab, len(Q))
        _POSES[aid] = Q[0].copy()


def freeze(parks, fleet, pens, h_inv):
    """Model these arms by their actual capsules at these poses.

    `parks` is {aid: q}; `fleet` and `pens` supply each arm's base transform and
    tool.  Replaces any previous frozen set.
    """
    global _CAPS, _POSES, _INVARIANT
    _CAPS, _POSES, _INVARIANT = {}, {}, {}
    set_standoff(None)
    set_keep_bands(False)
    for aid, q in (parks or {}).items():
        aid = int(aid)
        if aid not in fleet:
            continue
        q = np.asarray(q, float).reshape(1, 7)
        P = coordination.chain_world(q, fleet[aid], h_inv,
                                     float(pens.get(aid, 0.110)))
        tab = (coordination.CAPSULES_LAT if P.shape[1] >= 11
               else coordination.CAPSULES)
        A, B = coordination.cap_endpoints(P, tab)
        # a FROZEN partner is a known pose: drop link1's revolution sweep, its
        # real upper arm is already in the table
        tab, keep = coordination.known_pose_capsules(tab)
        A, B = np.asarray(A, float)[0][keep], np.asarray(B, float)[0][keep]
        R = np.array([c[2] for c in tab], float)
        if link_spheres.enabled():
            # ...and under the sphere model the partner carries BOTH blocks:
            # its shipped capsules and its spheres.  `partner_clearance` asks
            # each model its own question and keeps the larger answer, which
            # is the intersection of the two envelopes (see `link_spheres`).
            tab2, keep2 = link_spheres.moving_capsules(tab)
            SC = link_spheres.centres_world(
                q, fleet[aid].T_world_base(h_inv))[0]
            _CAPS[aid] = (np.concatenate([A[keep2], SC]),
                          np.concatenate([B[keep2], SC]),
                          np.concatenate([R[keep2], link_spheres.RADII]),
                          (A, B, R))
        else:
            _CAPS[aid] = (A, B, R)
        # THE MASK IS OVER `blk[3] or blk[:3]`, the SHIPPED capsule block, which
        # is the block the standoff pass below reads.  Under the sphere model
        # that is `blk[3]` and it is `(A, B, R)` either way.
        _INVARIANT[aid] = invariant_mask(tab, 1)
        _POSES[aid] = np.asarray(q, float).reshape(7).copy()


def thaw():
    """Back to the shipped pose-invariant model."""
    global _CAPS, _POSES, _OBSERVER, _INVARIANT
    _CAPS, _POSES, _OBSERVER, _INVARIANT = {}, {}, None, {}
    set_standoff(None)
    set_keep_bands(False)


def observe(aid):
    """Name the arm that is MOVING, so it is not checked against itself."""
    global _OBSERVER
    _OBSERVER = None if aid is None else int(aid)


def observer():
    return _OBSERVER


def keep_box(b):
    """Does this box survive the swap? -> bool.

    A band belonging to a frozen partner is dropped (its capsules take over);
    everything else — true structure, and the bands of partners that are NOT
    frozen — is kept.
    """
    if not _CAPS or _KEEP_BANDS:
        return True
    o = band_owner(b.get("name") if isinstance(b, dict) else None)
    return not (o is not None and o in _CAPS and o != _OBSERVER)


def filter_boxes(boxes):
    """`boxes` minus the frozen partners' own bands. -> list."""
    if not _CAPS or not boxes:
        return list(boxes) if boxes else boxes
    return [b for b in boxes if keep_box(b)]


def room_kinds():
    """What model each frozen partner is carried as. -> {aid: str}.

    `capsules` for an exact room, `spheres` for a cluster cloud, `pose` for the
    one-pose capsule block of a parked partner.  A cache that namespaces on the
    room has to see this: the same partner ids and the same poses describe two
    different obstacles under the two room models.
    """
    out = {}
    for a, blk in _CAPS.items():
        out[int(a)] = (blk.kind if isinstance(blk, exact_room.ExactRoom)
                       else ("spheres" if len(blk) == 3
                             and blk[0] is not blk[1]
                             and np.array_equal(blk[0], blk[1]) else "pose"))
    return out


def room_boxes(observer=None):
    """The frozen ROOMS' occupied airspace, as a few AABBs. -> [box dict].

    FOR DETOUR GENERATION ONLY.  `paper._skirt` reads box footprints to decide
    which way to walk around an obstacle, and a room is not a box, so the tier
    was blind to the one obstacle that mattered: every crossing the leader's
    trajectory blocked fell through to the RRT with no hint about WHERE to go.
    These are each room's own coarse cells (`ExactRoom.boxes`), handed to the
    waypoint generator and to nothing else — every leg the generator proposes
    is still gated against the exact room by `chain_clearance`, so a box here
    can cost a wasted detour and can never buy a certificate.
    """
    obs = _OBSERVER if observer is None else int(observer)
    out = []
    for a, blk in sorted(_CAPS.items()):
        if a == obs or not isinstance(blk, exact_room.ExactRoom):
            continue
        out += blk.boxes(name=f"room:{int(a)}")
    return out


def partner_clearance(P, C=None, floor=None):
    """Observer chain -> clearance to every frozen partner's real capsules.

    `P` is (N, 10 or 11, 3) world chain points, exactly what
    `rig_final.chain_static_clearance` takes.  -> (N,) surface gap, `inf` when
    the model is off or nobody but the observer is frozen.

    `C` is the OBSERVER's sphere centres (N,S,3) when the sphere model is
    active.  The partner side is already spheres whenever it is — `freeze`
    builds its capsule block under the same flag — so passing `C` is what
    makes the query spheres on BOTH sides instead of one.  Without it the
    observer is still measured as five sausages, which is valid (they contain
    the arm) and merely leaves half the accuracy on the table.
    """
    P = np.asarray(P, float)
    if P.ndim == 2:
        P = P[None]
    others = [a for a in _CAPS if a != _OBSERVER]
    if not others:
        return np.full(len(P), np.inf)
    caps = (rig_final.STATIC_CAPSULES_LAT if P.shape[1] >= 11
            else rig_final.STATIC_CAPSULES)
    lean = (link_spheres.static_capsules(caps)[0] if C is not None else caps)
    worst = np.full(len(P), np.inf)          # the shipped capsule model
    lean_w = np.full(len(P), np.inf)         # ...and the sphere one
    if C is not None:
        C = np.asarray(C, float)
    # THE CAPSULE MODEL FIRST, OVER EVERY PARTNER.  The floor short-circuit
    # below has to see the FINISHED capsule number: `worst` is a running
    # minimum, so a later partner can pull it under the floor after an earlier
    # one looked safe, and skipping that earlier partner's sphere terms would
    # leave `lean_w` a minimum over too few terms — which is larger, and would
    # make `max(worst, lean_w)` OVERSTATE the clearance.  Two passes, so the
    # decision is made once and on the complete number.
    for aid in others:
        blk = _CAPS[aid]
        # AN EXACT ROOM ANSWERS THE WHOLE QUESTION ITSELF, prune included; the
        # loop below is the flat-block form and would defeat the index.
        if isinstance(blk, exact_room.ExactRoom):
            worst = np.minimum(worst, blk.chain_clearance(P, caps, floor))
            continue
        Acap, Bcap, Rcap = blk[3] if len(blk) > 3 else (blk[0], blk[1], blk[2])
        for (i, j, r) in caps:
            # (N,1,3) observer segment against (1,C,3) partner segments
            d = coordination.seg_seg_dist(P[:, i][:, None, :],
                                          P[:, j][:, None, :],
                                          Acap[None, :, :], Bcap[None, :, :])
            worst = np.minimum(worst, (d - (Rcap[None, :] + r)).min(axis=1))
    # ...AND THE STANDOFF, WHICH IS A SECOND PASS OVER A SUBSET.  Nothing above
    # changed: this only folds in `gap_to_the_invariant_set - S`, so the number
    # that comes back is still a clearance and every caller's own floor is what
    # decides.  The subset is a subset of what the loop above already measured,
    # so with S = 0 this is a no-op by construction and with S > 0 it can only
    # LOWER the answer -- it never licenses anything.
    so = np.full(len(P), np.inf)
    for aid, S in _STANDOFF.items():
        if aid == _OBSERVER or aid not in _CAPS or S <= 0.0:
            continue
        blk = _CAPS[aid]
        if isinstance(blk, exact_room.ExactRoom):
            # A ROOM IS NOT SPLIT INTO ITS INVARIANT ROWS HERE, so the whole
            # room wears the standoff.  That is conservative, and in the
            # leader/follower sweep it never fires: a leader plans before every
            # follower, so a partner it holds a standoff against is always a
            # single held POSE and never a trajectory room.  The floor is
            # raised with it, or a floored query could return a bound above
            # `floor` that subtraction pushes below it.
            f = None if floor is None else float(floor) + float(S)
            so = np.minimum(so, blk.chain_clearance(P, caps, f) - S)
            continue
        m = _INVARIANT.get(aid)
        if m is None or not m.any():
            continue
        Acap, Bcap, Rcap = blk[3] if len(blk) > 3 else (blk[0], blk[1], blk[2])
        Ai, Bi, Ri = Acap[m], Bcap[m], Rcap[m]
        for (i, j, r) in caps:
            d = coordination.seg_seg_dist(P[:, i][:, None, :],
                                          P[:, j][:, None, :],
                                          Ai[None, :, :], Bi[None, :, :])
            so = np.minimum(so, (d - (Ri[None, :] + r)).min(axis=1) - float(S))
    worst = np.minimum(worst, so)
    if C is None:
        return worst
    sel = slice(None) if floor is None else np.flatnonzero(worst < float(floor))
    if floor is not None and not len(sel):
        return worst                     # `max` cannot change a passing row
    Ps, Cs = P[sel], C[sel]
    lean_w = np.full(len(Ps), np.inf)
    for aid in others:
        blk = _CAPS[aid]
        if isinstance(blk, exact_room.ExactRoom):
            lean_w = np.minimum(lean_w, blk.chain_clearance(Ps, lean, floor))
            lean_w = np.minimum(lean_w, blk.sphere_clearance(
                Cs, link_spheres.RADII, floor))
            continue
        A, B, R = blk[0], blk[1], blk[2]
        for (i, j, r) in lean:
            d = coordination.seg_seg_dist(Ps[:, i][:, None, :],
                                          Ps[:, j][:, None, :],
                                          A[None, :, :], B[None, :, :])
            lean_w = np.minimum(lean_w, (d - (R[None, :] + r)).min(axis=1))
        # the observer's spheres against the partner's own block
        d = coordination.seg_seg_dist(Cs[:, :, None, :], Cs[:, :, None, :],
                                      A[None, None, :, :], B[None, None, :, :])
        d = d - (R[None, None, :] + link_spheres.RADII[None, :, None])
        lean_w = np.minimum(lean_w, d.reshape(len(Ps), -1).min(axis=1))
    out = worst.copy()
    out[sel] = np.maximum(worst[sel], lean_w)
    # THE SPHERE MODEL MAY NOT UNDO THE STANDOFF.  `max` is the intersection of
    # the two envelopes and is right for the obstacle; the standoff is not an
    # obstacle, it is an extra requirement, so it survives both readings.
    return np.minimum(out, so)


def chain_clearance(P, room, C=None, floor=None):
    """The whole static room, measured. -> (N,).

    `room` is what `paper.static_boxes` hands out: real steel as boxes, and —
    when `envelope` is installed — body columns as CYLINDERS rather than their
    bounding boxes.  Each is measured with its own exact primitive, and a
    frozen partner's real capsules are folded in on top.

    `C` is the arm's sphere centres when the sphere model is active; it is
    passed straight through to all three, which is the single seam the whole
    arm-vs-room half of the flag runs through.

    With everything off this IS `rig_final.chain_static_clearance` called with
    the identical arguments, so every shipped number is reproduced exactly.
    """
    from . import envelope
    boxes, cyls = envelope.split(room)
    d = rig_final.chain_static_clearance(P, boxes, C=C, floor=floor)
    if cyls:
        d = np.minimum(d, envelope.chain_cyl_clearance(P, cyls, C=C,
                                                       floor=floor))
    if not _CAPS:
        return d
    return np.minimum(d, partner_clearance(P, C, floor))
