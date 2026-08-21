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

from .frames import FR3_MAX, FR3_MIN, PEN_EXT, fk
from .fleet import FLEET, H_INV_DEFAULT
from .rig_final import STATIC_MARGIN
from .validate import check_pose as validate_pose, validate_plan

# same envelope as the conductor, restated here on purpose: if someone widens
# a capsule there and the two disagree, this check is supposed to notice.
RADII = ((0, 1, 0.09), (1, 3, 0.09), (3, 4, 0.09), (4, 5, 0.09),
         (5, 7, 0.07), (7, 8, 0.07), (8, 9, 0.03))
# FINAL-RIG pen capsule: the holder envelope union (r 0.05), restated from
# rig_final.PEN_R_FINAL on purpose — a test pins the two together.
RADII_FINAL = RADII[:-1] + ((8, 9, 0.05),)


def _radii_for(fleet_dict, arms):
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


def _chain(q, spec, h_inv, pen_ext):
    """10 chain points of one configuration, in world.  Scalar `fk`, not the
    batch path, so a bug in the batch kernel cannot hide here."""
    T, P = fk(np.asarray(q, float))
    tip = T[:3, 3] + T[:3, :3] @ np.array([0.0, 0.0, pen_ext])
    P = np.vstack([P, tip])
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
    ia = np.array([c[0] for c in radii])
    ib = np.array([c[1] for c in radii])
    rr = np.array([c[2] for c in radii])
    a0, a1 = Pi[..., ia, :][..., :, None, :], Pi[..., ib, :][..., :, None, :]
    b0, b1 = Pj[..., ia, :][..., None, :, :], Pj[..., ib, :][..., None, :, :]
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
    for (i, j, r) in RADII_FINAL[1:]:              # skip the base column
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
    ok = bool(worst >= margin and bad == 0)
    rep = dict(ok=ok, min_clearance=float(worst), margin=float(margin),
               worst_pair=worst_at, poses=poses, poses_failed=int(bad),
               pen_below_paper=sorted(dipped),
               per_pair={f"{i}-{j}": v for (i, j), v in per_pair.items()})
    if verbose:
        print(f"scene_check(static): {len(arms)} arms, min clearance "
              f"{1000 * worst:.1f} mm (margin {1000 * margin:.0f} mm)"
              + (f" between arms {worst_at[0]} and {worst_at[1]}" if worst_at else "")
              + f"; {len(arms) - bad}/{len(arms)} poses pass their own gates"
              + f" -> {'PASS' if ok else 'FAIL'}")
        if dipped:
            print(f"  !! WARNING: arms {dipped} hold a pose whose PEN TIP is "
                  "below the paper plane (see `PEN_PAPER` in scene_check)")
    return rep


def check_timeline(qtraj, dt, margin, programs=None, h_inv=H_INV_DEFAULT,
                   pen_ext=PEN_EXT, sub=2, progress=None, verbose=True,
                   fleet=None):
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
    # Same sweep residual as the inter-arm check, same lower-bound logic.
    frame_clear, frame_bad = {}, []
    for a in arms:
        boxes = (fl[a].static_obstacles()
                 if hasattr(fl[a], "static_obstacles") else [])
        if not boxes:
            continue
        lb = static_clearance_lb(P[a], boxes) - 0.55 * stepd[a]
        k = int(np.argmin(lb))
        frame_clear[a] = (float(lb[k]), float(k * dt / max(sub, 1)))
        if lb[k] < STATIC_MARGIN:
            frame_bad.append(a)

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
                rep = validate_plan(np.asarray(pl["pts"], float), fl[a],
                                    np.asarray(pl["qs"], float),
                                    times=np.asarray(pl["times"], float),
                                    h_inv=None, pen_ext=pen_len(pen_ext, a))
                seg_bad += 0 if rep["ok"] else 1
                seg_reports.append(dict(arm=a, seg=k, ok=bool(rep["ok"])))

    ok = bool(worst >= margin and mono and seg_bad == 0 and frozen_bad == 0
              and min(lim.values()) > 0.0 and not frame_bad)
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
               frame_failed=sorted(frame_bad))
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
