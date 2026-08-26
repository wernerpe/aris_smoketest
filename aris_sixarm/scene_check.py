"""Independent check of a MERGED multi-arm timeline.  Nothing here trusts the
conductor.

`coordination.py` decides the schedule from a collision image it built itself;
if the image were wrong, the schedule would be confidently wrong with it.  So
this module re-derives everything from the merged timeline alone — the joint
trajectories that will actually be played back — with its own kinematics call,
its own capsule geometry, and a segment-distance routine written from a
different derivation (endpoint distances plus the interior critical point,
rather than the clamped parametrisation `coordination.seg_seg_dist` uses).
Where the two agree, they agree by geometry rather than by shared code.

What it verifies, and refuses to pass without:

  CLEARANCE   every pair of arms, at every sampled instant AND across the
              sweeps between instants, at least `margin` apart.  The timeline
              is re-sampled `sub` times finer than it was scheduled on, and the
              residual between those samples is covered by the same
              1-Lipschitz displacement bound, so "densely sampled" here means
              "bounded everywhere", not "probably fine".
  COLUMNS     every arm against every OTHER FLEET ARM'S BASE COLUMN, whether
              or not that arm appears in the timeline — the one 0.333 m of a
              robot that is in the same place in every configuration, and the
              part a phase conducted by a subset of the fleet would otherwise
              be checked as if it had been removed from the room.
  MONOTONE    each arm's progress only ever advances, and never by more than
              one index per step — the property the schedule's whole safety
              argument rests on.
  PER-ARM     `validate.validate_plan` re-run on every drawn segment (pen on
              the curve, joint margin, sigma, step size, paper and boom
              clearance, velocity limits), because a schedule cannot repair a
              segment and must not be allowed to mask one.
  PLAYBACK    the joints actually played back at each frame match the frozen
              path sample the schedule points at.

`check_timeline` returns a report; `ok` is the only thing the animation is
allowed to condition on.
"""
import numpy as np

from . import paper
from . import frames as _frames
from .frames import FR3_MAX, FR3_MIN, PEN_EXT, fk, tool_offset
from .fleet import FLEET, H_INV_DEFAULT
from .rig_final import STATIC_MARGIN
from .validate import check_pose as validate_pose, validate_plan

# same envelope as the conductor, restated here on purpose: if someone widens
# a capsule there and the two disagree, this check is supposed to notice.
#
# THE BASE COLUMN IS `COLUMN_BANDS`, WRITTEN AS SUB-SEGMENT CAPSULES.  Entries
# are `(i, j, r)` for a whole segment between two chain points, or
# `(i, j, r, t0, t1)` for the sub-segment `[t0, t1]` of it; the base column's
# bands all ride the pose-invariant [0, 1] axis at `t = base z / d1`, and t
# leaves [0, 1] at both ends because the metal does (see COLUMN_BANDS).  The
# table is built from THIS module's own band literals, not imported from
# `mounts` or `coordination` — a test pins all three together.
COLUMN_D1 = 0.333                    # m, base flange -> shoulder (DH)
# (z0, z1, r) in base z, flange downwards — the measured envelope, no margin.
COLUMN_BANDS = ((-0.2325, 0.0667, 0.177), (0.0667, 0.0988, 0.118),
                (0.0988, 0.2590, 0.078), (0.2590, 0.3875, 0.130))
N_BASE = len(COLUMN_BANDS)           # leading RADII entries that are the column
RADII = tuple((0, 1, r, z0 / COLUMN_D1, z1 / COLUMN_D1)
              for z0, z1, r in COLUMN_BANDS) \
    + ((1, 3, 0.130), (3, 4, 0.117), (4, 5, 0.131),
       (5, 7, 0.091), (7, 8, 0.104), (8, 9, 0.03))
# FINAL-RIG pen capsule: the holder envelope union (r 0.05), restated from
# rig_final.PEN_R_FINAL on purpose — a test pins the two together.
RADII_FINAL = RADII[:-1] + ((8, 9, 0.05),)
# LATERAL HOLDER: an 11-point chain (9 FK + tip@9 + bracket corner@10) and a
# TWO-capsule tool — bracket TCP->corner, pen corner->tip — restated from
# rig_final.STATIC_CAPSULES_LAT on purpose; a test pins the two together.
RADII_LAT = RADII[:-1] + ((8, 10, 0.05), (10, 9, 0.05))


def _moving(radii):
    """`radii` without the base-column bands — the capsules that MOVE.

    The base is bolted where it is, so every gate about an arm's own mount,
    the frame steel or a neighbour's column skips it.  A filter on the chain
    index rather than a slice, because the column is `N_BASE` entries now and
    "the first one" stopped being a safe way to say it.
    """
    return tuple(c for c in radii if c[0] != 0)


def _cap_ends(P, radii):
    """Chain points (...,K,3) -> the capsules' (A, B) endpoints (...,C,3).

    Own derivation, on purpose (see the module docstring): the sub-segment
    parameters are applied here as an explicit affine step rather than through
    `coordination.cap_endpoints`.
    """
    P = np.asarray(P, float)
    A = P[..., [c[0] for c in radii], :]
    D = P[..., [c[1] for c in radii], :] - A
    t0 = np.array([c[3] if len(c) > 3 else 0.0 for c in radii], float)
    t1 = np.array([c[4] if len(c) > 4 else 1.0 for c in radii], float)
    return A + t0[:, None] * D, A + t1[:, None] * D


def _radii_for(fleet_dict, arms):
    if _frames.PEN_LAT != 0.0:          # the ACTIVE tool is the lateral holder
        return RADII_LAT
    final = any(getattr(fleet_dict[a], "rig", "sixarm") == "final" for a in arms)
    return RADII_FINAL if final else RADII

# REPORTED, NOT GATED, AND DELIBERATELY SO.  `validate.check_pose` measures the
# pen tip against the paper plane, which nothing did before: the paper-clearance
# gate looks at the nine FK chain points and the pen is not one of them.  It
# turns out the INVERTED READY POSE fails it — the tip is 16 mm below the paper
# with a 200 mm pen and 113 mm below with a 300 mm one — so conductor v1 parked
# four arms through the table three times a run and no check noticed.  Making it
# a gate here would retroactively refuse the last release rather than the thing
# it is warning about, so it is separated out, reported loudly, and left for the
# rig owner to answer (a shorter pen, a different ready pose, or a survey that
# says the table is lower than the model thinks).
PEN_PAPER = "pen_below_paper"

# ==========================================================================
# ...AND THE GATE THAT SHOULD HAVE EXISTED WHEN THAT COMMENT WAS WRITTEN
# ==========================================================================
# The note above is about a POSE.  What it did not cover, and what nothing
# covered, is the MOTION between poses: a pen-up transit is a straight line in
# joint space and no check in this repo ever looked at what that line does on
# the way.  On the shipped `csail_final6` timeline three of forty-two pen-up
# blocks go through the table — pen tip 253.6 mm under the canvas (arm 97
# heading home, t = 49.33 s), a chain point 156.1 mm under it (arm 2
# mid-transit, t = 29.43 s) — and one of them is held down there by a
# conductor pause for three quarters of a second.  `scene_check` passed that
# timeline, and passed it on the same run in which it correctly caught a 40.9 mm
# frame clip, because the paper was the one piece of the scene it did not model.
#
# It does now, and it is a HARD gate, unlike `PEN_PAPER`.  The distinction that
# makes that safe is contact: the pen is *supposed* to be on the paper while
# drawing and passes through z = 0 at the ends of every lift and lower, so the
# tip is gated at the contact band and the CHAIN — which is never in contact,
# and sits one pen length up even when the tip is down — at the paper's own
# 2 cm keep-out.  Measured that way the shipped timeline misses by 175-254 mm
# and a correctly routed one clears by tens of millimetres, so the gate
# discriminates by two orders of magnitude rather than by a hair.
PAPER_CHAIN = paper.CHAIN_CLEAR      # m, chain points (1..8) above the paper
PAPER_TIP = paper.TIP_TOL            # m, how far under the tip may ever be
PAPER_STEP = 0.005                   # m of per-point motion the check refines to
FRAME_STEP = 0.005                   # ...and the same for the frame gate, which
#   was rate-dependent until the two-pass work found it refusing clear transits

# ==========================================================================
# ...AND THE SAME AGAIN FOR THE ARMS THAT ARE NOT IN THE TIMELINE
# ==========================================================================
# `pair_clearance` covers every arm the timeline CONTAINS.  An arm the timeline
# leaves out is not out of the room: it is standing wherever it stands, and
# 0.333 m of it — base flange to shoulder, the axis q1 turns about — is in the
# same place whatever pose that is.  A phase conducted with three of six arms,
# or a solo pass, would otherwise be checked as if the other three had been
# carried out of the building.
#
# So this gate re-derives that column HERE, from the fleet's own base
# transform and this module's own radii, and holds every arm's chain to the
# same `margin` the inter-arm gate uses.  It shares no code with
# `mounts.column_bands` (which inflates the same body into AABBs for the
# planner to consume) and no number with it either: the planner's boxes are
# each 0.03 wider than these because a box gate is compared against
# STATIC_MARGIN = 0.05 and this one against the inter-arm 0.08.  Two
# derivations, one geometric statement — which is the whole point of the
# module (`d1` and the radii are restated from `frames.DH[0][2]` and
# `coordination.LINK_R`; a test pins them).
#
# THE COLUMN IS NOT 0.09, DOES NOT STOP AT THE SHOULDER, AND IS NOT A POLE
# (2026-08-26).  The mesh audit measured link0 + link1's q1 sweep: the body is
# 0.155 m at its widest, its connector and cable stub reach 0.177 m over the
# first 67 mm below the plate, the whole assembly runs 0.3875 m from the
# flange — 54.5 mm PAST the `d1` this segment used to stop at — and through
# the middle third it WAISTS to 0.057.  `d1` is still the DH constant it
# always was; the column's shape is a separate, measured thing, and it is
# `COLUMN_BANDS` at the top of this module, where the capsule table that needs
# it is built.  Nothing here reads `mounts`: the planner's boxes are each
# `calib` = 0.03 wider than these bands because a box gate is compared against
# STATIC_MARGIN = 0.05 and this one against the inter-arm 0.08.  Two
# derivations, one geometric statement.
COLUMN_Z1 = COLUMN_BANDS[-1][1]      # m, MEASURED far end of the body column
COLUMN_R = 0.155                     # m, the widest the casting proper gets —
#   the radius a SINGLE-capsule column wore, kept as the default for the
#   legacy flat `(p0, p1)` form `_as_bands` still understands


def _chain(q, spec, h_inv, pen_ext):
    """10 (inline) or 11 (lateral tool) chain points of one configuration, in
    world.  Scalar `fk`, not the batch path, so a bug in the batch kernel
    cannot hide here; the tool points are built here from the raw pose for
    the same reason (`frames.tool_points_many` is the planner's helper)."""
    T, P = fk(np.asarray(q, float))
    off = tool_offset(pen_ext)              # ACTIVE tool decides the width
    tip = T[:3, 3] + T[:3, :3] @ off
    rows = [P, tip[None]]
    if off[0] != 0.0:
        corner = T[:3, 3] + T[:3, :3] @ np.array([off[0], 0.0, 0.0])
        rows.append(corner[None])
    P = np.vstack(rows)
    Twb = spec.T_world_base(h_inv)
    return P @ Twb[:3, :3].T + Twb[:3, 3]


def _pt_seg(p, a, b):
    """Distance from points p (...,3) to segment [a,b] (...,3)."""
    ab = b - a
    denom = np.sum(ab * ab, -1)
    t = np.where(denom > 1e-15, np.sum((p - a) * ab, -1) / np.where(denom > 1e-15,
                                                                   denom, 1.0), 0.0)
    t = np.clip(t, 0.0, 1.0)
    d = p - (a + t[..., None] * ab)
    return np.sqrt(np.sum(d * d, -1))


def segment_distance(p0, p1, q0, q1):
    """Distance between two segments — independent derivation.

    The minimum of a convex quadratic over the unit square is attained either
    at the interior stationary point (when it exists and lies inside) or on the
    boundary; the boundary minimum of THIS quadratic is one of the four
    point-to-segment distances.  Taking the smaller of the two candidates is
    therefore exact, and needs none of the clamp-and-re-solve bookkeeping.
    """
    d1, d2, r = p1 - p0, q1 - q0, p0 - q0
    a = np.sum(d1 * d1, -1)
    e = np.sum(d2 * d2, -1)
    b = np.sum(d1 * d2, -1)
    c = np.sum(d1 * r, -1)
    f = np.sum(d2 * r, -1)
    den = a * e - b * b
    inside = den > 1e-12
    s = np.where(inside, (b * f - c * e) / np.where(inside, den, 1.0), -1.0)
    t = np.where(inside, (a * f - b * c) / np.where(inside, den, 1.0), -1.0)
    good = inside & (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)
    w = r + s[..., None] * d1 - t[..., None] * d2
    interior = np.where(good, np.sqrt(np.maximum(np.sum(w * w, -1), 0.0)), np.inf)
    edge = np.minimum(np.minimum(_pt_seg(p0, q0, q1), _pt_seg(p1, q0, q1)),
                      np.minimum(_pt_seg(q0, p0, p1), _pt_seg(q1, p0, p1)))
    return np.minimum(interior, edge)


def pair_clearance(Pi, Pj, radii=RADII):
    """Min capsule clearance between two arms, for chain points (...,10,3).

    Broadcasts over any leading axis, so a whole timeline costs one call.
    """
    rr = np.array([c[2] for c in radii])
    Ai, Bi = _cap_ends(Pi, radii)
    Aj, Bj = _cap_ends(Pj, radii)
    a0, a1 = Ai[..., :, None, :], Bi[..., :, None, :]
    b0, b1 = Aj[..., None, :, :], Bj[..., None, :, :]
    d = segment_distance(a0, a1, b0, b1) - rr[:, None] - rr[None, :]
    return d.reshape(d.shape[:-2] + (-1,)).min(-1)


def static_clearance_lb(P, boxes, step=0.02):
    """LOWER BOUND on capsule-to-frame-box clearance — own derivation.

    The planner minimises distance along each capsule segment analytically
    (ternary search on a convex profile, `rig_final.segment_box_clearance`);
    here the same quantity is bounded by SAMPLING the segment every <= `step`
    metres and subtracting the 1-Lipschitz displacement residual, so the two
    code paths share nothing but the geometry itself.  The base-column capsule
    (0,1) is skipped for the same reason the planner skips it: the base is
    bolted to its mount by construction.

    P: (...,10,3) world chain points.  -> (...,) clearance lower bound.
    """
    if not boxes:
        return np.full(np.asarray(P).shape[:-2], np.inf)
    lo = np.stack([b["lo"] for b in boxes])
    hi = np.stack([b["hi"] for b in boxes])
    P = np.asarray(P, float)
    out = np.full(P.shape[:-2], np.inf)
    radii = RADII_LAT if P.shape[-2] >= 11 else RADII_FINAL
    for (i, j, r) in _moving(radii):               # skip the base column bands
        a, b = P[..., i, :], P[..., j, :]
        L = float(np.max(np.linalg.norm(b - a, axis=-1)))
        K = max(2, int(np.ceil(L / step)) + 1)
        ts = np.linspace(0.0, 1.0, K)
        pts = a[..., None, :] + ts[:, None] * (b - a)[..., None, :]
        d = np.maximum(np.maximum(lo - pts[..., None, :], pts[..., None, :] - hi),
                       0.0)
        dist = np.sqrt(np.sum(d * d, -1)).min(-1)          # over boxes
        slack = np.linalg.norm(b - a, axis=-1) / (2 * (K - 1))
        out = np.minimum(out, dist.min(-1) - r - slack)
    return out


def base_column(spec):
    """The pose-invariant base column of one arm -> [(p0 (3,), p1 (3,), r)].

    `spec.T_world_base()` is the only thing read: each band runs along the
    base z axis, which is the axis joint 1 rotates about, so no joint value
    can move it.  One entry per band of COLUMN_BANDS — the measured profile,
    fat at the plate where the connector is, 0.155 the rest of the way down.
    """
    T = np.asarray(spec.T_world_base(), float)
    p0, zc = T[:3, 3], T[:3, 2]
    return [(p0 + z0 * zc, p0 + z1 * zc, r) for z0, z1, r in COLUMN_BANDS]


def _as_bands(c):
    """One arm's column entry -> [(p0, p1, r)].

    Accepts the banded form `base_column` returns and the legacy single
    segment `(p0, p1)` / `(p0, p1, r)` a caller may still be holding.  The
    discriminator is whether the first element IS a 3-vector.
    """
    try:
        a0 = np.asarray(c[0], float)
    except (TypeError, ValueError):
        a0 = None
    if a0 is not None and a0.shape == (3,):
        return [(c[0], c[1], float(c[2]) if len(c) > 2 else COLUMN_R)]
    return [(b[0], b[1], float(b[2])) for b in c]


def column_clearance(P, columns, radii):
    """Chain points (...,10|11,3) vs base columns: [[(p0, p1, r)]] per arm,
    or the legacy flat [(p0, p1)] at COLUMN_R.

    -> (...,) worst capsule-to-column clearance.  The mover's OWN base column
    (capsule 0) is skipped — it is the one segment that is bolted where it is —
    exactly as `static_clearance_lb` and the planner's table skip it.

    A BAND IS A CAPSULE HERE, NOT A CYLINDER, and that is deliberate: capsule
    caps make the union of the stack more conservative than the body it stands
    for, which is the direction this gate is allowed to be wrong in.  The
    planner's boxes (`mounts.column_bands`) contain these bands grown by
    `calib`, so anything the planner certifies passes here.
    """
    bands = []
    for c in (columns or ()):
        bands.extend(_as_bands(c))
    if not bands:
        return np.full(np.asarray(P).shape[:-2], np.inf)
    P = np.asarray(P, float)
    out = np.full(P.shape[:-2], np.inf)
    for (i, j, r) in _moving(radii):
        a, b = P[..., i, :], P[..., j, :]
        for p0, p1, cr in bands:
            c0 = np.broadcast_to(np.asarray(p0, float), a.shape)
            c1 = np.broadcast_to(np.asarray(p1, float), a.shape)
            out = np.minimum(out, segment_distance(a, b, c0, c1) - r - cr)
    return out


def neighbour_columns(fleet_dict, arm, present):
    """[(p0, p1)] for every arm in the room that is not `arm`.

    "In the room" = the spec says `active`, or the arm is in the timeline
    (`present`) — an inactive registry entry is hardware that is not installed,
    and the legacy six-arm preset has two of them.
    """
    return [base_column(s) for a, s in sorted(fleet_dict.items())
            if a != arm and (getattr(s, "active", True) or a in present)]


def pen_len(pen_ext, arm):
    """This arm's pen, whether `pen_ext` is one length or {arm_id: length}."""
    if isinstance(pen_ext, dict):
        return float(pen_ext.get(arm, PEN_EXT))
    return float(pen_ext)


def check_static(q_by_arm, margin, h_inv=H_INV_DEFAULT, pen_ext=PEN_EXT,
                 verbose=True, fleet=None):
    """A whole fleet standing still, all at once. -> report dict.

    A POSE IS A CLAIM EVEN WHEN NOTHING IS MOVING.  Two of them are load-bearing
    and neither is a stroke: the configuration every arm holds while a human
    walks in to swap the pens, and — since `idle.py` — the pose each arm freezes
    in when it finishes, which the next pass then starts from and which no
    stroke validator ever saw.  Both get the same treatment as a timeline: pair
    clearance from this module's own capsules and its own segment distance, plus
    `validate.check_pose` per arm for the gates that are about the configuration
    alone (joint limits with the planner's comfort margin, chain height above
    the paper, the inverted arms' boom keep-out).

    No sweep slack is subtracted, because nothing sweeps.
    """
    fl = FLEET if fleet is None else fleet
    arms = sorted(q_by_arm)
    P = {a: _chain(np.asarray(q_by_arm[a], float).reshape(7), fl[a], h_inv,
                   pen_len(pen_ext, a)) for a in arms}
    rr = _radii_for(fl, arms)
    per_pair, worst, worst_at = {}, np.inf, None
    for i, ai in enumerate(arms):
        for aj in arms[i + 1:]:
            d = float(pair_clearance(P[ai], P[aj], rr))
            per_pair[(ai, aj)] = d
            if d < worst:
                worst, worst_at = d, (ai, aj)
    poses, bad, dipped = {}, 0, []
    for a in arms:
        rep = validate_pose(np.asarray(q_by_arm[a], float).reshape(7), fl[a],
                            h_inv, pen_len(pen_ext, a))
        kinds = [v["kind"] for v in rep["violations"]]
        hard = [k for k in kinds if k != PEN_PAPER]
        bad += 1 if hard else 0
        dipped += [a] if PEN_PAPER in kinds else []
        poses[a] = dict(ok=not hard, worst=rep["worst"], violations=kinds)
    # ...and the arms that are standing there without being in this dict
    col_clear, col_bad = {}, []
    for a in arms:
        cols = neighbour_columns(fl, a, set(arms))
        if not cols:
            continue
        d = float(column_clearance(P[a][None], cols, rr)[0])
        col_clear[a] = d
        if d < margin:
            col_bad.append(a)
    ok = bool(worst >= margin and bad == 0 and not col_bad)
    rep = dict(ok=ok, min_clearance=float(worst), margin=float(margin),
               worst_pair=worst_at, poses=poses, poses_failed=int(bad),
               pen_below_paper=sorted(dipped),
               column_clearance={int(a): v for a, v in col_clear.items()},
               column_failed=sorted(col_bad),
               per_pair={f"{i}-{j}": v for (i, j), v in per_pair.items()})
    if verbose:
        print(f"scene_check(static): {len(arms)} arms, min clearance "
              f"{1000 * worst:.1f} mm (margin {1000 * margin:.0f} mm)"
              + (f" between arms {worst_at[0]} and {worst_at[1]}" if worst_at else "")
              + f"; {len(arms) - bad}/{len(arms)} poses pass their own gates"
              + (f"; base columns {1000 * min(col_clear.values()):.1f} mm"
                 + (f" FAIL {col_bad}" if col_bad else "") if col_clear else "")
              + f" -> {'PASS' if ok else 'FAIL'}")
        if dipped:
            print(f"  !! WARNING: arms {dipped} hold a pose whose PEN TIP is "
                  "below the paper plane (see `PEN_PAPER` in scene_check)")
    return rep


def check_timeline(qtraj, dt, margin, programs=None, h_inv=H_INV_DEFAULT,
                   pen_ext=PEN_EXT, sub=2, progress=None, verbose=True,
                   fleet=None, drawing=None):
    """Verify a merged timeline. -> report dict (`ok` gates the animation).

    `qtraj` is {arm_id: (M,7)} exactly as it will be played back; `sub` sets how
    many extra samples are interpolated between two scheduled steps.

    `pen_ext` may be a single length or {arm_id: length}.  The pen is a capsule
    of the arm and the last 30 cm of its geometry, so an arm carrying a 300 mm
    pen checked at 110 mm is checked as a shorter robot than the one that runs:
    the clearance this module reports would be about a machine that does not
    exist.  It is also the length the per-segment validator re-derives the pen
    tip with, and that has to be the length the segment was certified at.
    """
    arms = sorted(qtraj)
    M = len(next(iter(qtraj.values())))
    fine = {}
    for a in arms:
        Q = np.asarray(qtraj[a], float)
        if sub > 1 and M > 1:
            g = np.linspace(0, M - 1, (M - 1) * sub + 1)
            i0 = np.clip(g.astype(int), 0, M - 2)
            fr = (g - i0)[:, None]
            fine[a] = Q[i0] * (1 - fr) + Q[i0 + 1] * fr
        else:
            fine[a] = Q
    F = len(next(iter(fine.values())))

    fl = FLEET if fleet is None else fleet
    P = {a: np.array([_chain(q, fl[a], h_inv, pen_len(pen_ext, a))
                      for q in fine[a]]) for a in arms}
    stepd = {a: np.concatenate([[0.0], np.linalg.norm(np.diff(P[a], axis=0),
                                                      axis=2).max(1)]) for a in arms}

    rr = _radii_for(fl, arms)
    worst, worst_at = np.inf, None
    per_pair = {}
    for i, ai in enumerate(arms):
        for aj in arms[i + 1:]:
            # ...minus the sweep back to the previous fine sample, so the bound
            # holds between samples and not only at them
            lo = (pair_clearance(P[ai], P[aj], rr)
                  - 0.55 * (stepd[ai] + stepd[aj]))
            k = int(np.argmin(lo))
            per_pair[(ai, aj)] = float(lo[k])
            if lo[k] < worst:
                worst = float(lo[k])
                worst_at = (ai, aj, float(k * dt / max(sub, 1)))

    lim = {a: float(min(np.min(np.asarray(fine[a]) - FR3_MIN),
                        np.min(FR3_MAX - np.asarray(fine[a])))) for a in arms}

    # STATIC STRUCTURE: every frame of every arm against the rig's frame
    # boxes — covers the transits and hovers no per-segment validator sees.
    # Same lower-bound logic as the inter-arm check, and — since the two-pass
    # work — the same AUTO-REFINEMENT as the paper gate below, for the same
    # reason and with the same consequence if it is left out.
    #
    # THE VERDICT HAS TO BE A PROPERTY OF THE TRAJECTORY, NOT OF THE RATE IT
    # WAS HANDED IN AT.  `0.55 * stepd[a]` is a 1-Lipschitz sweep residual, and
    # it scales with the sampling step: at `sub=2` it charges an arm nearly
    # 4.5 mm of phantom approach that is not in the motion at all.  That is not
    # academic — it is what refused the two-pass logo eight times over.  Arm 71
    # in the orange pass reads 49.1 mm against a 50 mm margin at `sub=2` and
    # 53.5 mm at `sub=8`, the SAME timeline at the same instant (t = 22.75 s);
    # the transit was always clear and the check was charging it for being
    # sampled coarsely.  `docs/MERGED_CANVAS.md` §3.3's 40.9 mm veto is very
    # likely the same artefact.  Refining to <= FRAME_STEP of per-point motion
    # caps the residual under a tenth of the margin and the numbers stop moving.
    frame_clear, frame_bad, frame_refine = {}, [], {}
    for a in arms:
        boxes = (fl[a].static_obstacles()
                 if hasattr(fl[a], "static_obstacles") else [])
        if not boxes:
            continue
        Pa, res, ex = P[a], 0.55 * stepd[a], 1
        Q = np.asarray(fine[a], float)
        if len(Q) > 1:
            mv = float(np.max(np.linalg.norm(np.diff(P[a], axis=0), axis=2)))
            ex = int(np.clip(np.ceil(mv / FRAME_STEP), 1, 32))
        if ex > 1:
            g = np.linspace(0, len(Q) - 1, (len(Q) - 1) * ex + 1)
            i0 = np.clip(g.astype(int), 0, len(Q) - 2)
            fr = (g - i0)[:, None]
            Pa = np.array([_chain(q, fl[a], h_inv, pen_len(pen_ext, a))
                           for q in Q[i0] * (1 - fr) + Q[i0 + 1] * fr])
            res = 0.55 * float(np.max(np.linalg.norm(np.diff(Pa, axis=0),
                                                     axis=2), initial=0.0))
        lb = static_clearance_lb(Pa, boxes) - res
        k = int(np.argmin(lb))
        # report the time on the SCHEDULED clock whatever the refinement was
        frame_clear[a] = (float(lb[k]),
                          float(k * dt / (max(sub, 1) * max(ex, 1))))
        frame_refine[a] = int(ex)
        if lb[k] < STATIC_MARGIN:
            frame_bad.append(a)

    # EVERY OTHER ARM'S BASE COLUMN, in the room whether or not it is in this
    # timeline (see the note by COLUMN_R).  Same sweep residual as the
    # inter-arm gate, because it is the same kind of statement.
    col_clear, col_bad = {}, []
    for a in arms:
        cols = neighbour_columns(fl, a, set(arms))
        if not cols:
            continue
        lb = column_clearance(P[a], cols, rr) - 0.55 * stepd[a]
        k = int(np.argmin(lb))
        col_clear[a] = (float(lb[k]), float(k * dt / max(sub, 1)))
        if lb[k] < margin:
            col_bad.append(a)

    # THE PAPER: every arm's tip and chain, at every instant, including the
    # transits and hovers no per-segment validator ever sees.  The residual is
    # PER POINT — each body's own displacement between two fine samples, not
    # the worst body's — because the tip travels slowly along the paper while
    # it draws and charging it the elbow's sweep would refuse the ink for being
    # ink.
    paper_clear, paper_bad = {}, []
    for a in arms:
        # AUTO-REFINE UNTIL THE RESIDUAL IS SMALL, so the verdict is a property
        # of the trajectory and not of the rate it was handed in at.  The
        # residual is a 1-Lipschitz bound on 3D displacement, which over-charges
        # a pen tip SLIDING ALONG the paper (large step, no vertical motion);
        # at the animation's own 12 fps that alone can read 14.7 mm of false
        # dip.  Refining to <= PAPER_STEP of per-point motion caps it at about
        # 2.8 mm, an order below the tip floor, and the numbers stop moving.
        Q = np.asarray(fine[a], float)
        ex = 1
        if len(Q) > 1:
            mv = float(np.max(np.linalg.norm(np.diff(P[a], axis=0), axis=2)))
            ex = int(np.clip(np.ceil(mv / PAPER_STEP), 1, 32))
        if ex > 1:
            g = np.linspace(0, len(Q) - 1, (len(Q) - 1) * ex + 1)
            i0 = np.clip(g.astype(int), 0, len(Q) - 2)
            fr = (g - i0)[:, None]
            Pa = np.array([_chain(q, fl[a], h_inv, pen_len(pen_ext, a))
                           for q in Q[i0] * (1 - fr) + Q[i0 + 1] * fr])
        else:
            Pa = P[a]
        d = np.concatenate([np.zeros((1, Pa.shape[1])),
                            np.linalg.norm(np.diff(Pa, axis=0), axis=2)])
        # chain = links 1..8 plus, on the lateral tool, the bracket corner
        # (index 10) — a rigid body that is never in contact; the tip (9)
        # keeps its own separate gate as always.
        ccols = list(range(1, 9)) + ([10] if Pa.shape[1] >= 11 else [])
        chain = (Pa[:, ccols, 2] - 0.55 * d[:, ccols]).min(axis=1)
        tip = Pa[:, 9, 2] - 0.55 * d[:, 9]
        kc, kt = int(np.argmin(chain)), int(np.argmin(tip))
        step_t = dt / (max(sub, 1) * ex)
        up = None
        if drawing is not None:
            # map each refined sample back to the scheduled step it lies in,
            # rather than assuming the refinement is a clean integer repeat of
            # the schedule (it is not: `fine` has (M-1)*sub+1 samples, not M*sub)
            dm = np.asarray(drawing[a], bool)
            src = np.clip((np.arange(len(tip)) / (max(sub, 1) * ex)).astype(int),
                          0, len(dm) - 1)
            up = ~dm[src]
        paper_clear[a] = dict(
            chain=float(chain[kc]), chain_t=float(kc * step_t),
            tip=float(tip[kt]), tip_t=float(kt * step_t), refine=int(ex),
            tip_penup=(float(tip[up].min()) if up is not None and up.any()
                       else None))
        if chain[kc] < PAPER_CHAIN or tip[kt] < -PAPER_TIP:
            paper_bad.append(a)

    mono = True
    if progress is not None:
        for a, p in progress.items():
            d = np.diff(np.asarray(p, int))
            mono &= bool(np.all(d >= 0) and np.all(d <= 1))

    # WHERE THE ARMS STOP IS PART OF THE TIMELINE.  Under the freeze-in-place
    # idle policy the last sample of each arm is a pose nothing ever certified —
    # it is a hover, not a stroke, so `validate_plan` below never sees it — and
    # the fleet then stands in it for the rest of the run and (in a two-pass
    # piece) through the pen swap and into the next pass.  Its inter-arm
    # clearance is already covered by the sweep above; these are the gates that
    # are about the configuration alone.
    frozen, frozen_bad, frozen_dip = {}, 0, []
    for a in arms:
        rep_p = validate_pose(np.asarray(qtraj[a], float)[-1], fl[a], h_inv,
                              pen_len(pen_ext, a))
        kinds = [v["kind"] for v in rep_p["violations"]]
        hard = [k for k in kinds if k != PEN_PAPER]
        frozen_bad += 1 if hard else 0
        frozen_dip += [a] if PEN_PAPER in kinds else []
        frozen[a] = dict(ok=not hard, worst=rep_p["worst"], violations=kinds)

    seg_reports, seg_bad = [], 0
    if programs:
        for a, segs in programs.items():
            for k, s in enumerate(segs):
                pl = s["plan"]
                # THE CONE TRAVELS WITH THE PLAN.  A tilt-rescued segment is
                # allowed the lean it was granted and not one degree more, and
                # a segment that never leaned is checked against a
                # perpendicular pen — which is every segment in every run made
                # before `aris_sixarm/tilt.py` existed, and all but a handful
                # in the runs made since (`tilt.plan_adaptive` records the cone
                # the plan NEEDS, not the one the run allowed).
                cone = float(pl.get("tilt_max_deg", 0.0) or 0.0)
                rep = validate_plan(np.asarray(pl["pts"], float), fl[a],
                                    np.asarray(pl["qs"], float),
                                    times=np.asarray(pl["times"], float),
                                    h_inv=None, pen_ext=pen_len(pen_ext, a),
                                    tilt_max_deg=cone)
                seg_bad += 0 if rep["ok"] else 1
                seg_reports.append(dict(
                    arm=a, seg=k, ok=bool(rep["ok"]), cone_deg=cone,
                    lean_deg=float(rep["worst"].get("max_lean_deg", 0.0))))

    ok = bool(worst >= margin and mono and seg_bad == 0 and frozen_bad == 0
              and min(lim.values()) > 0.0 and not frame_bad and not paper_bad
              and not col_bad)
    rep = dict(ok=ok, min_clearance=float(worst), margin=float(margin),
               worst_pair=worst_at, per_pair={f"{i}-{j}": v for (i, j), v in
                                              per_pair.items()},
               joint_margin=lim, monotone=bool(mono), n_frames=M, n_fine=F,
               pens={int(a): pen_len(pen_ext, a) for a in arms},
               n_segments=len(seg_reports), segments_failed=int(seg_bad),
               segments=seg_reports, frozen=frozen,
               frozen_failed=int(frozen_bad),
               frozen_pen_below_paper=sorted(frozen_dip),
               frame_clearance={int(a): v for a, v in frame_clear.items()},
               frame_margin=float(STATIC_MARGIN),
               frame_refine={int(a): int(v) for a, v in frame_refine.items()},
               frame_failed=sorted(frame_bad),
               paper_clearance={int(a): v for a, v in paper_clear.items()},
               paper_chain_margin=float(PAPER_CHAIN),
               paper_tip_margin=float(PAPER_TIP),
               paper_failed=sorted(paper_bad),
               column_clearance={int(a): v for a, v in col_clear.items()},
               column_failed=sorted(col_bad))
    if verbose:
        print(f"scene_check: {M} scheduled steps re-sampled to {F}, "
              f"{len(arms)} arms, {len(per_pair)} pairs")
        print(f"  min inter-arm clearance {worst * 1000:.1f} mm "
              f"(margin {margin * 1000:.0f} mm)"
              + (f" at t={worst_at[2]:.2f} s between arms {worst_at[0]} and "
                 f"{worst_at[1]}" if worst_at else ""))
        print(f"  per-arm plan validation: {len(seg_reports) - seg_bad}/"
              f"{len(seg_reports)} segments ok; progress monotone: {mono}")
        if frame_clear:
            wa = min(frame_clear, key=lambda a: frame_clear[a][0])
            print(f"  min frame clearance {frame_clear[wa][0] * 1000:.1f} mm "
                  f"(margin {STATIC_MARGIN * 1000:.0f} mm), arm {wa} at "
                  f"t={frame_clear[wa][1]:.2f} s"
                  + (f"; FAIL: arms {frame_bad}" if frame_bad else ""))
        if col_clear:
            wc = min(col_clear, key=lambda a: col_clear[a][0])
            print(f"  min neighbour-base-column clearance "
                  f"{col_clear[wc][0] * 1000:.1f} mm (margin "
                  f"{margin * 1000:.0f} mm), arm {wc} at "
                  f"t={col_clear[wc][1]:.2f} s"
                  + (f"; FAIL: arms {col_bad}" if col_bad else ""))
        wp = min(paper_clear, key=lambda a: paper_clear[a]["chain"])
        wt = min(paper_clear, key=lambda a: paper_clear[a]["tip"])
        print(f"  paper clearance: chain {paper_clear[wp]['chain'] * 1000:.1f} mm "
              f"(margin {PAPER_CHAIN * 1000:.0f} mm) arm {wp} at "
              f"t={paper_clear[wp]['chain_t']:.2f} s; tip "
              f"{paper_clear[wt]['tip'] * 1000:.1f} mm (floor "
              f"{-PAPER_TIP * 1000:.0f} mm) arm {wt} at "
              f"t={paper_clear[wt]['tip_t']:.2f} s"
              + (f"; FAIL: arms {paper_bad}" if paper_bad else ""))
        print(f"  frozen poses: {len(arms) - frozen_bad}/{len(arms)} pass the "
              "joint-limit, paper and boom gates"
              + ("" if not frozen_bad else "  <- "
                 + "; ".join(f"arm {a}: {','.join(v['violations'])}"
                             for a, v in frozen.items() if not v["ok"]))
              + ("" if not frozen_dip else
                 f"; !! arms {frozen_dip} stop with the PEN TIP BELOW THE "
                 "PAPER (reported, not gated — see scene_check.PEN_PAPER)"))
        print(f"  VERDICT {'PASS' if ok else 'FAIL'}")
    return rep
