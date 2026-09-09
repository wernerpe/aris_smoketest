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

from . import coordination, link_spheres, rig_final

# {aid: (A (C,3), B (C,3), R (C,))} of every frozen partner's world capsules
_CAPS = {}
_POSES = {}
_OBSERVER = None

_BAND = re.compile(r"^body:(\d+)_column\d+$")


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


def freeze(parks, fleet, pens, h_inv):
    """Model these arms by their actual capsules at these poses.

    `parks` is {aid: q}; `fleet` and `pens` supply each arm's base transform and
    tool.  Replaces any previous frozen set.
    """
    global _CAPS, _POSES
    _CAPS, _POSES = {}, {}
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
            # ...and under the sphere model the "real upper arm" is 64 fitted
            # spheres rather than five chain sausages.  Applied AFTER the
            # known-pose drop so the two filters compose in the one order that
            # makes sense: band 3 goes because the pose is known, the moving
            # sausages go because something tighter stands in for them.
            tab, keep2 = link_spheres.moving_capsules(tab)
            SC = link_spheres.centres_world(
                q, fleet[aid].T_world_base(h_inv))[0]
            A = np.concatenate([A[keep2], SC])
            B = np.concatenate([B[keep2], SC])
            R = np.concatenate([R[keep2], link_spheres.RADII])
        _CAPS[aid] = (A, B, R)
        _POSES[aid] = np.asarray(q, float).reshape(7).copy()


def thaw():
    """Back to the shipped pose-invariant model."""
    global _CAPS, _POSES, _OBSERVER
    _CAPS, _POSES, _OBSERVER = {}, {}, None


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
    if not _CAPS:
        return True
    o = band_owner(b.get("name") if isinstance(b, dict) else None)
    return not (o is not None and o in _CAPS and o != _OBSERVER)


def filter_boxes(boxes):
    """`boxes` minus the frozen partners' own bands. -> list."""
    if not _CAPS or not boxes:
        return list(boxes) if boxes else boxes
    return [b for b in boxes if keep_box(b)]


def partner_clearance(P):
    """Observer chain -> clearance to every frozen partner's real capsules.

    `P` is (N, 10 or 11, 3) world chain points, exactly what
    `rig_final.chain_static_clearance` takes.  -> (N,) surface gap, `inf` when
    the model is off or nobody but the observer is frozen.
    """
    P = np.asarray(P, float)
    if P.ndim == 2:
        P = P[None]
    others = [a for a in _CAPS if a != _OBSERVER]
    if not others:
        return np.full(len(P), np.inf)
    caps = (rig_final.STATIC_CAPSULES_LAT if P.shape[1] >= 11
            else rig_final.STATIC_CAPSULES)
    worst = np.full(len(P), np.inf)
    for aid in others:
        A, B, R = _CAPS[aid]
        for (i, j, r) in caps:
            # (N,1,3) observer segment against (1,C,3) partner segments
            d = coordination.seg_seg_dist(P[:, i][:, None, :],
                                          P[:, j][:, None, :],
                                          A[None, :, :], B[None, :, :])
            worst = np.minimum(worst, (d - (R[None, :] + r)).min(axis=1))
    return worst


def chain_clearance(P, room):
    """The whole static room, measured. -> (N,).

    `room` is what `paper.static_boxes` hands out: real steel as boxes, and —
    when `envelope` is installed — body columns as CYLINDERS rather than their
    bounding boxes.  Each is measured with its own exact primitive, and a
    frozen partner's real capsules are folded in on top.

    With everything off this IS `rig_final.chain_static_clearance` called with
    the identical arguments, so every shipped number is reproduced exactly.
    """
    from . import envelope
    boxes, cyls = envelope.split(room)
    d = rig_final.chain_static_clearance(P, boxes)
    if cyls:
        d = np.minimum(d, envelope.chain_cyl_clearance(P, cyls))
    if not _CAPS:
        return d
    return np.minimum(d, partner_clearance(P))
