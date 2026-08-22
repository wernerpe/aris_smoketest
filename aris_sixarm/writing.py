"""Write the word ARIS: per-stroke DP plans, pen-up transits, and a
time-indexed joint schedule for every arm in the fleet.

Pure kinematics + numpy (no drake), so it runs under the system python3 and
under the pydrake venv alike.  The drake demo calls `plan_word()` once, caches
the result as JSON, and replays `sample_schedule()` frame by frame.

Letters are written SEQUENTIALLY (A, R, I, S); an arm that is not currently
writing holds its ready pose `spec.q_seed`.

PEN-UP TRANSITS ARE A PLACEHOLDER, NOT A PLANNED MOTION.  Between two strokes
we lift the pen `LIFT_Z` above the paper at the stroke end, joint-interpolate
straight to the lifted pose above the next stroke start, and descend.  Linear
joint interpolation makes no collision or self-collision guarantee and the
Cartesian path in between is whatever the interpolation happens to trace — the
real system needs the RRT that item 3 of the roadmap calls for.  It is
adequate here because the lifted poses are close together on the same letter
and the workspace above the paper is empty.
"""
import json

import numpy as np

from . import ik, letters, paper, planner
from .fleet import FLEET, H_INV_DEFAULT
from .frames import (FR3_MIN, FR3_MAX, PEN_EXT, QD_MAX, joint_margin, rotx,
                     tip_pos)

DS = 0.01               # stroke resampling step, m
DRAW_SPEED = 0.08       # m/s along a stroke
FPS = 30.0
LIFT_Z = 0.06           # pen-up height above the paper, m
TIP_TOL = 2e-3          # on-curve tolerance for the FK cross-check, m
INK_Z = 0.0015          # ink is drawn just above the paper (z-fighting)
INK_CHUNK = 0.025       # target ink chunk length, m (~15 chunks per stroke)

# timeline (see module docstring): one pen-up transit is 1.0 s total
T_LIFT, T_TRAVEL, T_LOWER = 0.25, 0.50, 0.25
T_HOME = 1.5            # ready pose <-> first/last lifted pose
T_LETTER_GAP = 0.4      # idle beat between letters

COMFORT_R = 0.50        # "comfortable" pen radius from the base, m
NUDGE_STEPS = (0.05, 0.10, 0.15)
FALLBACK_HEIGHT = 0.30


# --------------------------------------------------------------------------
# pen-up poses
# --------------------------------------------------------------------------
HOVER_MARGIN = 0.10     # rad, the joint-limit margin a HOVER pose must keep.
#   Looser than `validate.MARGIN_GATE` (0.15) on purpose and historically: a
#   hover is a place to stand, not a curve to be dragged along at a commanded
#   speed.  Callers that are CHOOSING a pose rather than accepting one — the
#   idle policy's retreat — ask for the stricter gate instead, because there is
#   no reason to spend margin you do not have to.


def lifted_config(spec, q_ref, xy, z=LIFT_Z, h_inv=H_INV_DEFAULT,
                  pen_ext=PEN_EXT, span=0.6, n_q7=25, margin_min=HOVER_MARGIN):
    """IK pose with the pen tip at (x, y, z), R = rotx(pi), nearest to q_ref.

    Scans q7 around q_ref[6] (the redundancy that q_ref already picked) and
    all analytic branches; returns the solution with the smallest ||dq||_inf
    among those keeping at least `margin_min` rad of joint-limit margin.  The
    filter is inside the scan and not applied afterwards, so raising it does not
    merely reject the nearest solution — it picks the nearest ACCEPTABLE one,
    which is usually a different q7 rather than no answer at all.
    """
    Twb = spec.T_world_base(h_inv)
    R_w = rotx(np.pi)
    T_w = np.eye(4)
    T_w[:3, :3] = R_w
    T_w[:3, 3] = np.array([xy[0], xy[1], z]) - pen_ext * R_w[:, 2]
    T_b = np.linalg.inv(Twb) @ T_w
    q7s = np.clip(q_ref[6] + np.linspace(-span, span, n_q7),
                  FR3_MIN[6] + 1e-3, FR3_MAX[6] - 1e-3)
    best, best_d = None, np.inf
    for q7 in q7s:
        for q in ik.solve(T_b, q7, q_ref):
            if joint_margin(q) < margin_min:
                continue
            d = float(np.max(np.abs(q - q_ref)))
            if d < best_d:
                best, best_d = q, d
    return best, best_d


# --------------------------------------------------------------------------
# per-letter planning (with placement nudges)
# --------------------------------------------------------------------------
def _placement_candidates(spec, center, height):
    """Nominal placement first, then centre nudges of up to +-0.15 m along the
    base->centre ray (toward the comfortable radius first), then the same
    sequence at the fallback height."""
    base = np.asarray(spec.xy, float)
    c0 = np.asarray(center, float)
    u = c0 - base
    r0 = float(np.linalg.norm(u))
    u = u / max(r0, 1e-9)
    sgn = 1.0 if COMFORT_R > r0 else -1.0     # +1 = away from base
    for h in (height, FALLBACK_HEIGHT):
        yield c0, h, ("nominal" if h == height else f"height {h:.2f} m")
        for d in NUDGE_STEPS:
            for s in (sgn, -sgn):
                yield (c0 + s * d * u, h,
                       f"centre {s * d:+.2f} m along base->centre"
                       + ("" if h == height else f", height {h:.2f} m"))


def _plan_strokes(strokes, spec, ds=DS, h_inv=H_INV_DEFAULT, verbose=True):
    """Plan every stroke of one placement. -> (list of per-stroke dicts | None)."""
    out = []
    for k, poly in enumerate(strokes):
        pts, _ = planner.resample(poly, ds)
        lat = planner.build_lattice(pts, spec, h_inv=h_inv)
        res = planner.plan(lat, objective="maximin_sigma")
        if not res["ok"]:
            if verbose:
                print(f"    stroke {k}: FAILED, cut_s={res['cut_s']:.3f}")
            return None
        rep = planner.path_report(lat, res)
        err = tip_error(res["qs"], lat)
        if err > TIP_TOL:
            if verbose:
                print(f"    stroke {k}: tip off curve by {err:.2e} m")
            return None
        out.append(dict(pts=lat["pts"], qs=res["qs"], sigmas=res["sigmas"],
                        margins=res["margins"], min_sigma=rep["min_sigma"],
                        min_margin=rep["min_margin"], tip_err=err,
                        max_step=rep["max_step"],
                        length=float(ds * (len(lat["pts"]) - 1))))
        if verbose:
            print(f"    stroke {k}: {len(pts)} steps, {out[-1]['length']:.3f} m, "
                  f"min_sigma={rep['min_sigma']:.4f} min_margin={rep['min_margin']:.3f} "
                  f"tip_err={err:.2e} m")
    return out


MAX_DQ_FRAME = 0.04       # rad, per sub-step of the densified stroke


def densify(qs, pts, spec, h_inv=H_INV_DEFAULT, pen_ext=PEN_EXT,
            max_dq=MAX_DQ_FRAME):
    """Sub-sample a planned stroke so that FRAME interpolation stays on the curve.

    The DP's continuity window allows up to JUMP_THRESH rad between two
    consecutive 10 mm steps, and it uses that budget: advancing one q7 index
    is a null-space self-motion (q1 and q3 counter-rotate, q5/q7 compensate)
    that moves the pen tip barely at all.  Those configurations are NOT
    collinear in joint space, so linearly interpolating between them for
    animation frames leaves the null-space manifold and swings the tip off the
    paper by several mm.

    Both endpoints are the same analytic-IK branch (verified: zero branch
    changes along every ARIS stroke), so the gap can be filled exactly:
    re-solve the case-consistent IK at Cartesian points along the segment with
    q7 interpolated.  Returns (qs_dense (M,7), u_dense (M,)) with u the
    normalised arc position, so the caller can keep constant pen speed.
    """
    Twb = spec.T_world_base(h_inv)
    Twb_inv = np.linalg.inv(Twb)
    R_w = rotx(np.pi)
    T_w = np.eye(4)
    T_w[:3, :3] = R_w
    n = len(qs) - 1
    out_q, out_u, fallbacks = [qs[0]], [0.0], 0
    for i in range(n):
        dq = float(np.max(np.abs(qs[i + 1] - qs[i])))
        k = max(1, int(np.ceil(dq / max_dq)))
        for m in range(1, k):
            f = m / k
            p = pts[i] + f * (pts[i + 1] - pts[i])
            T_w[:3, 3] = np.array([p[0], p[1], 0.0]) - pen_ext * R_w[:, 2]
            q7 = qs[i, 6] + f * (qs[i + 1, 6] - qs[i, 6])
            q = ik.solve_cc(Twb_inv @ T_w, q7, out_q[-1])
            if q is None:                       # no case-consistent solution
                q = qs[i] + f * (qs[i + 1] - qs[i])
                fallbacks += 1
            out_q.append(q)
            out_u.append((i + f) / n)
        out_q.append(qs[i + 1])
        out_u.append((i + 1) / n)
    return np.array(out_q), np.array(out_u), fallbacks


def tip_error_pts(qs, pts_ref, spec, h_inv=H_INV_DEFAULT, pen_ext=PEN_EXT):
    """Max world tip-to-reference distance for an arbitrary (q, xy) pairing."""
    Twb = spec.T_world_base(h_inv)
    tip = np.array([Twb[:3, :3] @ tip_pos(q, pen_ext) + Twb[:3, 3] for q in qs])
    exy = np.linalg.norm(tip[:, :2] - pts_ref, axis=1)
    return float(np.max(np.hypot(exy, tip[:, 2])))


def tip_error(qs, lat):
    """Max world-frame distance from the FK pen tip to the commanded curve."""
    Twb = lat["Twb"]
    tip = np.array([Twb[:3, :3] @ tip_pos(q, lat["pen_ext"]) + Twb[:3, 3] for q in qs])
    exy = np.linalg.norm(tip[:, :2] - lat["pts"][:len(qs)], axis=1)
    return float(np.max(np.hypot(exy, tip[:, 2])))


def plan_letter(name, arm_id, center, height=letters.DEFAULT_HEIGHT,
                aspect=letters.DEFAULT_ASPECT, ds=DS, h_inv=H_INV_DEFAULT,
                verbose=True):
    """Plan one glyph, nudging the placement until every stroke plans."""
    spec = FLEET[arm_id]
    for ctr, h, tag in _placement_candidates(spec, center, height):
        r = float(np.linalg.norm(np.asarray(ctr) - np.asarray(spec.xy)))
        if verbose:
            print(f"  '{name}' arm {arm_id} @ ({ctr[0]:.3f}, {ctr[1]:.3f}) "
                  f"h={h:.2f} r={r:.3f} [{tag}]")
        polys = letters.place(name, ctr, h, aspect)
        got = _plan_strokes(polys, spec, ds, h_inv, verbose)
        if got is not None:
            for S in got:                       # animation-grade sub-sampling
                qd, ud, fb = densify(S["qs"], S["pts"], spec, h_inv)
                ref = np.column_stack([
                    np.interp(ud, np.linspace(0, 1, len(S["pts"])), S["pts"][:, 0]),
                    np.interp(ud, np.linspace(0, 1, len(S["pts"])), S["pts"][:, 1])])
                S["qs_dense"], S["u_dense"] = qd, ud
                S["dense_tip_err"] = tip_error_pts(qd, ref, spec, h_inv)
                S["dense_fallbacks"] = fb
                if verbose:
                    print(f"      densified {len(S['qs'])} -> {len(qd)} steps, "
                          f"tip_err={S['dense_tip_err']:.2e} m"
                          + (f", {fb} IK fallbacks" if fb else ""))
            return dict(name=name, arm_id=arm_id, center=[float(ctr[0]), float(ctr[1])],
                        height=float(h), aspect=float(aspect), nudge=tag,
                        nudged=tag != "nominal", radius=r, strokes=got,
                        min_sigma=min(s["min_sigma"] for s in got),
                        min_margin=min(s["min_margin"] for s in got),
                        max_tip_err=max(s["tip_err"] for s in got))
    raise RuntimeError(f"letter {name!r} on arm {arm_id} did not plan at any "
                       f"of the candidate placements")


def plan_word(placement=None, height=letters.DEFAULT_HEIGHT,
              aspect=letters.DEFAULT_ASPECT, ds=DS, h_inv=H_INV_DEFAULT,
              verbose=True):
    return [plan_letter(n, a, c, height, aspect, ds, h_inv, verbose)
            for n, a, c in (placement or letters.PLACEMENT)]


# --------------------------------------------------------------------------
# schedule
# --------------------------------------------------------------------------
def build_schedule(plan, fps=FPS, draw_speed=DRAW_SPEED, h_inv=H_INV_DEFAULT,
                   verbose=True):
    """Time-index the whole word.

    Returns dict with
      t          (nT,) frame times
      q          {arm_id: (nT, 7)} joint trajectory for every fleet arm
      ink        list of (t_visible, letter_idx, stroke_idx, chunk_pts (M,3))
      duration, phases (bookkeeping)
    """
    keys = {aid: [] for aid in FLEET}          # arm -> [(t, q)] waypoints
    ink, phases = [], []
    t = 0.0

    def add(aid, tt, q):
        keys[aid].append((float(tt), np.asarray(q, float)))

    for aid, spec in FLEET.items():
        add(aid, 0.0, spec.q_seed)

    for li, L in enumerate(plan):
        spec = FLEET[L["arm_id"]]
        aid = L["arm_id"]
        strokes = L["strokes"]
        lifts = []      # (q_lift_start, q_lift_end) per stroke
        for S in strokes:
            qs = S["qs"]
            q0, d0 = lifted_config(spec, qs[0], S["pts"][0], h_inv=h_inv)
            q1, d1 = lifted_config(spec, qs[-1], S["pts"][-1], h_inv=h_inv)
            if q0 is None or q1 is None:
                raise RuntimeError(f"no lifted IK for letter {L['name']}")
            lifts.append((q0, q1))
        t0_letter = t

        # ready -> lifted above the first stroke start
        add(aid, t, spec.q_seed)
        t += T_HOME
        add(aid, t, lifts[0][0])
        t += T_LOWER
        add(aid, t, strokes[0]["qs"][0])

        for k, S in enumerate(strokes):
            qs, pts = S["qs"], S["pts"]
            dur = S["length"] / draw_speed
            # the densified trajectory carries its own normalised arc position,
            # so constant pen speed survives the sub-sampling
            qdraw = S.get("qs_dense", qs)
            u = S.get("u_dense", np.linspace(0.0, 1.0, len(qs)))
            ts = t + np.asarray(u, float) * dur
            for tt, q in zip(ts, qdraw):
                add(aid, tt, q)
            phases.append(dict(kind="stroke", letter=li, stroke=k,
                               t0=float(t), t1=float(t + dur), arm=aid))
            # ink chunks become visible as the tip reaches the end of each
            n_ch = int(np.clip(round(S["length"] / INK_CHUNK), 4, 40))
            edges = np.linspace(0, len(pts) - 1, n_ch + 1).astype(int)
            for c in range(n_ch):
                i0, i1 = edges[c], edges[c + 1]
                seg = pts[i0:i1 + 1]
                xyz = np.column_stack([seg, np.full(len(seg), INK_Z)])
                t_vis = t + dur * i1 / (len(pts) - 1)   # tip reaches the far end
                ink.append((float(t_vis), li, k, c, xyz))
            t += dur

            if k + 1 < len(strokes):                # pen-up transit
                t += T_LIFT
                add(aid, t, lifts[k][1])
                t += T_TRAVEL
                add(aid, t, lifts[k + 1][0])
                t += T_LOWER
                add(aid, t, strokes[k + 1]["qs"][0])
                phases.append(dict(kind="transit", letter=li, stroke=k,
                                   t0=float(t - T_LIFT - T_TRAVEL - T_LOWER),
                                   t1=float(t), arm=aid))

        t += T_LIFT
        add(aid, t, lifts[-1][1])
        t += T_HOME
        add(aid, t, spec.q_seed)
        phases.append(dict(kind="letter", letter=li, t0=float(t0_letter),
                           t1=float(t), arm=aid))
        t += T_LETTER_GAP

    duration = float(t)
    for aid, spec in FLEET.items():
        add(aid, duration, spec.q_seed)

    nT = int(round(duration * fps)) + 1
    ts = np.arange(nT) / fps
    q = {}
    for aid, kk in keys.items():
        kk.sort(key=lambda e: e[0])
        kt = np.array([e[0] for e in kk])
        kq = np.array([e[1] for e in kk])
        q[aid] = np.column_stack([np.interp(ts, kt, kq[:, j]) for j in range(7)])
    if verbose:
        print(f"schedule: {duration:.2f} s, {nT} frames @ {fps:g} fps, "
              f"{len(ink)} ink chunks")
    return dict(t=ts, q=q, ink=ink, duration=duration, phases=phases, fps=fps)


# ==========================================================================
# concurrent fleet programmes: one FROZEN timeline per arm
# ==========================================================================
# The word "ARIS" above is written one letter at a time by one arm at a time.
# A whole logo allocated to six arms is not: every arm has its own programme
# and they all run at once.  What follows builds ONE arm's timeline — entry
# lift, its segments in the order and the direction `sequence.py` chose, a
# pen-up transit between each pair, exit lift — as a frozen path with a nominal
# clock.  (The transit durations it lays down are `transit_time` below, which
# is also the cost function the sequencer minimised; `csail_schedule.py` checks
# the two against each other every run.)
#
# FROZEN is the operative word: `coordination.py` may only stretch this clock
# (insert pauses), never re-order the segments or re-route a transit, so the
# certified per-segment plans stay exactly what `stroke_api` certified.
DRAW_SPEED_FLEET = 0.12     # m/s along a stroke, the concurrent default
TRANSIT_SPEED = 0.80        # m/s of pen-tip travel between strokes
T_LIFT_F, T_LOWER_F = 0.20, 0.20
T_TRAVEL_MIN = 0.25
T_HOME_F = 1.20             # ready pose <-> first/last lifted pose
QD_FRAC = 0.30              # fraction of the FR3 joint-velocity limit a
#   PEN-UP move is allowed to use.  This is not decoration: a transit is a
#   straight line in joint space, so "1.2 seconds from the ready pose to the
#   paper" asks joint 1 for several rad/s and throws the elbow across a metre
#   of workspace between two animation frames.  The conductor bounds what can
#   happen BETWEEN two samples by the distance the bodies move, so an
#   unpaced transit does not just look wrong, it costs real clearance —
#   every millimetre of per-step motion is a millimetre off the margin.
#   Drawing is paced by `draw_speed` and then stretched, by the same rule, if
#   the redundancy resolution asks a joint to move faster than this.
#
#   0.60 IS THE NUMBER THE MEASUREMENT SAYS, AND 0.30 IS THE NUMBER HERE.
#   Both halves of that are deliberate, and neither was written down before
#   2026-08-20.  THE MEASUREMENT (`README.md`, "Pacing is a clearance
#   budget"): the first cut of the conductor ran transits as 1.2 s joint-space
#   lines and moved the elbow 295 mm between two frames, so the sweep slack
#   alone exceeded the 80 mm margin four times over and the schedule was
#   infeasible; capping every move at 60 % of the FR3 joint velocity limit
#   brings that displacement to <= 32 mm and the problem becomes easy.  0.30
#   is a SECOND halving on top of that, which no measurement asked for, and it
#   is not free: `draw_duration` stretches the ink until no joint exceeds this
#   fraction, and the critical speed v* = length / need below which the cap
#   never binds is LINEAR in it.  On the shipped CSAIL two-pass run at the
#   0.12 m/s animation speed the whole piece is 104.0 s at 0.30 and 84.4 s at
#   0.60 — 18.9 % of the makespan, at identical coverage (99.2139 %) and with
#   `scene_check` PASS either way (82.1 mm and 82.7 mm).  At the rig's own
#   0.02 m/s the cap barely matters: the smallest v* on the logo is 0.0149 m/s
#   at 0.30, where 2 of 58 segments still cap (ink 1.006x the material's
#   time), against 0.0299 m/s at 0.60, where none do.
#
#   ADOPTING 0.60 WAS TRIED ON 2026-08-20 AND THE CORPUS REFUSED IT, so the
#   halving stays until the refusal does.  The logo passes and `bench`'s
#   `scatter` passes and gets 26 % faster; `bench`'s `spiral` conducts in
#   89.4 s at 0.30 and CANNOT BE CONDUCTED AT ALL at 0.60 — and spiral is one
#   of the three rows that are exact end to end (`sequence.EXACT_MAX_N`), so
#   that is a reproducible answer and not a budget flake.  The mechanism is
#   not the pacing but what the pacing buys: `allocate.rebalance` prices in
#   SECONDS, halving the transit term changes which assignment is cheapest
#   (six cuts proposed, where at 0.30 the conductor handed all nine back and
#   the spiral shipped uncut), and the arms then finish in poses they
#   cannot stop clear of — "arm 2 cannot stop clear of 97 (-94 mm)", with the
#   unsplit alternative and conductor v1's go-home both refused after it.  So
#   this is blocked on the ALLOCATION being conductable at 0.60, not on the
#   cap being safe.  Pass `--qd-frac 0.6` per run, exactly as `README.md`'s
#   demo recipe does, until that is fixed.  docs/BENCH.md has both halves.


def _dq_time(q0, q1, frac=QD_FRAC, tmin=0.0):
    """Shortest time a straight joint-space move may take. -> seconds."""
    d = np.abs(np.asarray(q1, float) - np.asarray(q0, float))
    return float(max(tmin, np.max(d / (QD_MAX * max(frac, 1e-6)))))


def dq_time_many(Q0, Q1, frac=QD_FRAC, tmin=0.0):
    """`_dq_time` for every pair. (A,7) x (B,7) -> (A,B) seconds.

    The same arithmetic as `_dq_time`, done as one array operation because the
    sequencer (`sequence.py`) needs the whole matrix of "how long from every
    exit pose to every entry pose" before it can choose an order, and asking
    for it one pair at a time is what makes a few hundred segments per arm
    expensive.  `tmin` broadcasts, so the per-pair travel floor (the pen-tip
    hop at `transit_speed`) goes in as an (A,B) array.
    """
    Q0 = np.asarray(Q0, float).reshape(-1, 7)
    Q1 = np.asarray(Q1, float).reshape(-1, 7)
    lim = QD_MAX * max(frac, 1e-6)
    out = np.empty((len(Q0), len(Q1)))
    rows = max(1, int(4e6 // max(7 * len(Q1), 1)))       # cap the temporary
    for a0 in range(0, len(Q0), rows):
        a1 = min(a0 + rows, len(Q0))
        d = np.abs(Q1[None, :, :] - Q0[a0:a1, None, :]) / lim
        out[a0:a1] = d.max(axis=-1)
    return np.maximum(tmin, out)


def _draw_time(qd, ud, dur, frac=QD_FRAC):
    """`dur` stretched until no joint exceeds `frac` of its velocity limit."""
    du = np.diff(np.asarray(ud, float))
    dq = np.abs(np.diff(np.asarray(qd, float), axis=0))
    need = dq / (QD_MAX * max(frac, 1e-6)) / np.maximum(du, 1e-12)[:, None]
    return float(max(dur, need.max() if need.size else 0.0))


def draw_duration(qd, ud, length, draw_speed=DRAW_SPEED_FLEET, frac=QD_FRAC):
    """Seconds of ink for one densified segment. -> float.

    `length / draw_speed` is what the MATERIAL allows; `_draw_time` stretches it
    until no joint exceeds `frac` of its velocity limit, which is why 4.5 m of
    ink at 0.12 m/s can take 64 s and not 38.  The draw-speed limit is a cap,
    never a promise, and this is the one place that arithmetic lives —
    `arm_program` lays the clock down with it and `allocate.balance_loads`
    scores a candidate assignment with it, so the allocator's idea of "how long
    would this arm take" is the timeline's, not a proxy for it.
    """
    return _draw_time(qd, ud, float(length) / max(draw_speed, 1e-9), frac)


def segment_draw_time(spec, seg, draw_speed=DRAW_SPEED_FLEET, qd_frac=QD_FRAC,
                      h_inv=H_INV_DEFAULT, pen_ext=PEN_EXT):
    """How long THIS arm needs to lay THIS certified segment's ink. -> seconds.

    Densifies exactly as `arm_program` does (same IK, same pen) and paces the
    result the same way, so summing this over an arm's segments and adding the
    sequencer's tour cost reproduces `arm_program`'s `duration` to the float.
    """
    qs = np.asarray(seg["plan"]["qs"], float)
    pts = np.asarray(seg["plan"]["pts"], float)
    qd, ud, _ = densify(qs, pts, spec, h_inv, pen_ext)
    return draw_duration(qd, ud, seg["length"], draw_speed, qd_frac)


# --------------------------------------------------------------------------
# what a pen-up costs — ONE definition, used by the timeline and the sequencer
# --------------------------------------------------------------------------
# `arm_program` below lays these three moves down as waypoints; `sequence.py`
# adds them up to decide which segment should follow which.  They have to be
# the same numbers or the sequencer is optimising a fiction, so they are
# computed here and nowhere else.
def transit_time(q_exit, q_hover_exit, q_hover_entry, q_entry, hop,
                 transit_speed=TRANSIT_SPEED, qd_frac=QD_FRAC):
    """One pen-up transit -> (lift, travel, lower) seconds.

    Lift off the paper at the segment's exit, travel between the two hover
    poses (never faster than `transit_speed` over the `hop` metres of paper,
    and never faster than `qd_frac` of the joint velocity limits), lower onto
    the next segment's entry.
    """
    return (_dq_time(q_exit, q_hover_exit, qd_frac, T_LIFT_F),
            _dq_time(q_hover_exit, q_hover_entry, qd_frac,
                     max(T_TRAVEL_MIN, hop / max(transit_speed, 1e-9))),
            _dq_time(q_hover_entry, q_entry, qd_frac, T_LOWER_F))


def enter_time(q_home, q_hover_entry, q_entry, qd_frac=QD_FRAC):
    """Ready pose -> hover -> first segment's entry. -> (home, lower) seconds."""
    return (_dq_time(q_home, q_hover_entry, qd_frac, T_HOME_F),
            _dq_time(q_hover_entry, q_entry, qd_frac, T_LOWER_F))


def exit_time(q_exit, q_hover_exit, q_home, qd_frac=QD_FRAC):
    """Last segment's exit -> hover -> ready pose. -> (lift, home) seconds."""
    return (_dq_time(q_exit, q_hover_exit, qd_frac, T_LIFT_F),
            _dq_time(q_hover_exit, q_home, qd_frac, T_HOME_F))


# --------------------------------------------------------------------------
# ...AND WHAT IT IS ALLOWED TO FLY THROUGH ON THE WAY
# --------------------------------------------------------------------------
# THE PAPER IS AN OBSTACLE, AND UNTIL 2026-08-21 NOTHING SAID SO.  The three
# beats above are straight lines in joint space between poses that are each
# certified on their own, and `transit_time` prices them without ever asking
# what the line does in between.  On the shipped `csail_final6` timeline what
# it does, three times, is go through the table — 253.6 mm of pen tip and
# 156.1 mm of wrist below the canvas, once with a conductor pause holding the
# arm down there for three quarters of a second.
#
# `paper.route` is the answer: it certifies the straight line and, when the
# line is refused, returns VIA-CONFIGURATIONS that are themselves gated IK
# solutions.  The beats below carry those vias as extra waypoints, and every
# one of them is priced with the same `_dq_time` cap as the move it replaced,
# so a routed transit is still velocity-capped and the seconds it costs are
# seconds the sequencer sees (`sequence.cost_matrix` calls straight into here).
#
# A beat is a list of (seconds, q) steps.  With no vias it is one step and the
# arithmetic is bit-identical to `transit_time`'s, which is what keeps every
# pinned number in the corpus reproducible on a transit that never needed
# fixing — 39 of the shipped run's 42 pen-up blocks are exactly that case.
PAPER_SAFE = True           # route pen-ups around the paper plane


def _beat(qs, qd_frac=QD_FRAC, floor=0.0):
    """A chain of configurations -> [(seconds, q)], total >= `floor`.

    Each hop is capped at `qd_frac` of the joint velocity limits exactly as a
    single-hop move would be; the floor (the lift/lower beat minimum, or the
    pen-tip hop at `transit_speed`) applies to the beat AS A WHOLE and is spread
    proportionally, so inserting a via never makes the arm move faster than the
    un-routed move would have.
    """
    qs = [np.asarray(q, float).reshape(7) for q in qs]
    ts = [_dq_time(a, b, qd_frac, 0.0) for a, b in zip(qs[:-1], qs[1:])]
    tot = float(sum(ts))
    if floor > tot:
        ts = [floor / len(ts)] * len(ts) if tot <= 1e-12 else \
            [t * (floor / tot) for t in ts]
    return list(zip([float(x) for x in ts], qs[1:]))


def _route(spec, q0, q1, pen_ext, h_inv, tip_floor, qd_frac, floor,
           paper_safe=True, q_home=None):
    """One certified pen-up move as a beat. -> ([(s, q), ...], mode) | None.

    `None` is a REFUSAL and callers must propagate it: `sequence.cost_matrix`
    turns it into an infinite edge so the tour never proposes the move, and
    `arm_program` raises rather than lay a path it cannot certify.
    """
    if not paper_safe:
        return _beat([q0, q1], qd_frac, floor), "unchecked"
    r = paper.route(spec, q0, q1, pen_ext=pen_ext, h_inv=h_inv,
                    tip_floor=tip_floor, q_home=q_home)
    if r is None:
        return None
    return _beat([q0] + list(r["vias"]) + [q1], qd_frac, floor), r["mode"]


def transit_beats(spec, q_exit, q_hover_exit, q_hover_entry, q_entry, hop,
                  z_exit=LIFT_Z, z_entry=LIFT_Z, pen_ext=PEN_EXT,
                  h_inv=H_INV_DEFAULT, transit_speed=TRANSIT_SPEED,
                  qd_frac=QD_FRAC, paper_safe=PAPER_SAFE, q_home=None):
    """`transit_time`, with the paper as an obstacle. -> dict | None.

    -> dict(lift, travel, lower, total, steps, modes) where each of the three
    beats is a list of (seconds, q) steps and `total` is what the transit costs.
    `None` when the crossing cannot be certified at all — the honest answer, and
    the one that lets the sequencer pick a different order instead of flying it.

    The three beats keep different floors because they are doing different
    things: a lift STARTS with the pen on the paper and a lower ENDS there, so
    both are allowed inside the contact band, while the travel between two
    hovers has no business near the plane at all and keeps `paper.TIP_CLEAR` —
    capped by the hover the arm actually reached, since `lifted_or_lower` gives
    up height near the edge of reach.
    """
    lift = _route(spec, q_exit, q_hover_exit, pen_ext, h_inv, -paper.TIP_TOL,
                  qd_frac, T_LIFT_F, paper_safe)
    if lift is None:
        return None
    trav = _route(spec, q_hover_exit, q_hover_entry, pen_ext, h_inv,
                  paper.travel_floor(z_exit, z_entry), qd_frac,
                  max(T_TRAVEL_MIN, hop / max(transit_speed, 1e-9)),
                  paper_safe, q_home)
    if trav is None:
        return None
    low = _route(spec, q_hover_entry, q_entry, pen_ext, h_inv, -paper.TIP_TOL,
                 qd_frac, T_LOWER_F, paper_safe)
    if low is None:
        return None
    beats = [lift[0], trav[0], low[0]]
    return dict(lift=lift[0], travel=trav[0], lower=low[0],
                steps=[s for b in beats for s in b],
                total=float(sum(s[0] for b in beats for s in b)),
                modes=(lift[1], trav[1], low[1]))


def enter_beats(spec, q_home, q_hover_entry, q_entry, pen_ext=PEN_EXT,
                h_inv=H_INV_DEFAULT, qd_frac=QD_FRAC, paper_safe=PAPER_SAFE):
    """Ready pose -> hover -> the first segment's entry. -> dict | None."""
    # NEITHER END OF THIS ONE IS IN CONTACT.  The arm starts in its ready pose,
    # high over the paper, and finishes at a hover — so the trip in keeps the
    # FLYING floor, not the contact band.  Giving it the contact band (which the
    # first cut did) would have let an entry swing through the canvas with
    # nothing at construction time objecting.
    home = _route(spec, q_home, q_hover_entry, pen_ext, h_inv,
                  paper.travel_floor(LIFT_Z, LIFT_Z), qd_frac, T_HOME_F,
                  paper_safe)
    if home is None:
        return None
    low = _route(spec, q_hover_entry, q_entry, pen_ext, h_inv, -paper.TIP_TOL,
                 qd_frac, T_LOWER_F, paper_safe)
    if low is None:
        return None
    return dict(home=home[0], lower=low[0], steps=home[0] + low[0],
                total=float(sum(s[0] for s in home[0] + low[0])),
                modes=(home[1], low[1]))


def exit_beats(spec, q_exit, q_hover_exit, q_home, pen_ext=PEN_EXT,
               h_inv=H_INV_DEFAULT, qd_frac=QD_FRAC, paper_safe=PAPER_SAFE):
    """Last exit -> hover -> the ready pose. -> dict | None.

    THE TRIP HOME IS A TRANSIT LIKE ANY OTHER, and on the shipped timeline it
    is two of the three that went through the table: arms 2 and 97 both dive on
    their way to `q_seed`, because that pose is a metre away and on the far side
    of a branch change.  It gets the same treatment and the same refusal.
    """
    lift = _route(spec, q_exit, q_hover_exit, pen_ext, h_inv, -paper.TIP_TOL,
                  qd_frac, T_LIFT_F, paper_safe)
    if lift is None:
        return None
    home = _route(spec, q_hover_exit, q_home, pen_ext, h_inv,
                  paper.travel_floor(LIFT_Z, LIFT_Z), qd_frac, T_HOME_F,
                  paper_safe)
    if home is None:
        return None
    return dict(lift=lift[0], home=home[0], steps=lift[0] + home[0],
                total=float(sum(s[0] for s in lift[0] + home[0])),
                modes=(lift[1], home[1]))


def lifted_or_lower(spec, q_ref, xy, heights=(LIFT_Z, 0.045, 0.03), h_inv=H_INV_DEFAULT,
                    pen_ext=PEN_EXT):
    """`lifted_config`, retrying at lower heights. -> (q, height_used).

    Near the edge of an arm's reach the 6 cm hover has no IK solution with the
    margin we insist on even though the stroke's endpoint does; dropping the
    hover is strictly better than dropping the transit.  A last resort of "do
    not lift at all" keeps the timeline well-formed (the pen scuffs the paper,
    which the report says out loud rather than hiding).
    """
    for z in heights:
        q, _ = lifted_config(spec, q_ref, xy, z=z, h_inv=h_inv, pen_ext=pen_ext)
        if q is not None:
            return np.asarray(q, float), float(z)
    return np.asarray(q_ref, float), 0.0


PARK_FREEZE = "freeze"      # stop at the hover pose above the last stroke
PARK_HOME = "home"          # the old behaviour: transit back to `spec.q_seed`


class PaperRefused(RuntimeError):
    """A pen-up move that cannot be flown without entering the paper.

    Raised by `arm_program` rather than returning a timeline that would have to
    be vetoed downstream.  `sequence.cost_matrix` prices the same refusal as an
    infinite edge, so an order the sequencer chooses can never raise this for a
    transit — it is the entry, the go-home and the retreat, whose endpoints the
    tour does not get to choose, that can still hit it.
    """


def arm_program(spec, segs, draw_speed=DRAW_SPEED_FLEET, transit_speed=TRANSIT_SPEED,
                h_inv=H_INV_DEFAULT, ink_chunk=INK_CHUNK, qd_frac=QD_FRAC,
                pen_ext=PEN_EXT, q_start=None, park=PARK_FREEZE, retreat=None,
                taxi_stretch=0.0, paper_safe=PAPER_SAFE, verbose=False):
    """One arm's frozen nominal timeline from its allocated segments.

    `segs` are `allocate.allocate`'s programme entries, already in the order the
    arm will draw them; each carries the certified `plan` whose dense `qs`/`pts`
    are the joint trajectory and the curve it was certified against.

    `pen_ext` is THIS ARM's pen, and it has to be the pen the segments were
    certified with: the densifier re-solves IK at intermediate Cartesian points
    (tip = TCP + pen_ext along tool z) and the hover poses are IK too, so a
    default pen here against a 300 mm plan would silently re-plan the transit
    for a tool the arm is not holding.

    WHERE THE ARM STARTS AND WHERE IT STOPS ARE NOW ARGUMENTS, because both used
    to be `spec.q_seed` and that cost more clock than anything else in the piece
    (`docs/IDLE.md`):

      `q_start`      the pose the arm is standing in when the pass begins.  The
                     second pass of a two-pass run starts wherever the first one
                     left the arm, so this is not `q_seed` there and the
                     animation would teleport if it pretended otherwise.
      `park`         "freeze" (default) lifts the pen at the last stroke's exit
                     and STOPS THERE; "home" is the old transit back to
                     `spec.q_seed`.  Freezing is not a saving of a second and a
                     half of home transit — it is a saving of the WAIT for
                     permission to arrive, because the conductor may not let an
                     arm reach its final pose until that pose is clear all the
                     way to the horizon, and the hover pose above the ink an arm
                     has just laid is a pose the fleet was already avoiding.
      `retreat`      a (7,) pose to creep to after that lift, for the rare
                     frozen pose that is genuinely in another arm's way.
                     `idle.plan_retreat` chooses it; this only lays it down.
      `taxi_stretch` seconds to spread ACROSS THE PEN-UP BLOCKS (never across
                     the ink).  An arm with slack against the phase's floor
                     arrives at each entry just in time instead of racing there
                     and standing still: same path, same certificate, a fraction
                     of the per-step motion, and therefore a fraction of the
                     swept-tube slack it costs everybody else.

    -> dict(t, q, seg, u, phases, ink, duration, lifts, dense_tip_err, q_end,
            transit_s, taxi_s, retreat_s, draw_s, ...)
       t     (K,)    waypoint times, strictly increasing
       q     (K,7)   waypoint joints
       seg   (K,)    index into `segs` while drawing, -1 while not
       u     (K,)    normalised arc position within that segment
       ink   list of (t_visible, chunk_xyz (M,3))
       `transit_s` is the pen-up time the SEQUENCER priced; `taxi_s` and
       `retreat_s` are the two idle-policy additions on top of it, kept apart
       so "the sequencer's model is the timeline's clock" stays checkable.
    """
    t, T, Q, S, U = 0.0, [], [], [], []
    ink, phases, lifts, worst = [], [], [], 0.0
    q0 = np.asarray(spec.q_seed if q_start is None else q_start, float)

    def add(tt, q, s=-1, u=0.0):
        T.append(float(tt))
        Q.append(np.asarray(q, float))
        S.append(int(s))
        U.append(float(u))

    if not segs:                                  # an arm that reaches nothing
        add(0.0, q0)
        add(1.0, q0)
        return dict(t=np.array(T), q=np.array(Q), seg=np.array(S), u=np.array(U),
                    phases=[], ink=[], duration=0.0, lifts=[], dense_tip_err=0.0,
                    draw_len=0.0, transit_len=0.0, transit_s=0.0, draw_s=0.0,
                    taxi_s=0.0, retreat_s=0.0, q_end=q0, park=str(park),
                    pen=float(pen_ext), paper_modes=[], paper_vias=0,
                    fallbacks=0)

    dense = []
    for k, s in enumerate(segs):
        qs = np.asarray(s["plan"]["qs"], float)
        pts = np.asarray(s["plan"]["pts"], float)
        qd, ud, fb = densify(qs, pts, spec, h_inv, pen_ext)
        ref = np.column_stack([np.interp(ud, np.linspace(0, 1, len(pts)), pts[:, 0]),
                               np.interp(ud, np.linspace(0, 1, len(pts)), pts[:, 1])])
        err = tip_error_pts(qd, ref, spec, h_inv, pen_ext)
        worst = max(worst, err)
        dense.append(dict(qd=qd, ud=ud, pts=pts, fallbacks=fb, tip_err=err,
                          length=float(s["length"])))
        if verbose:
            print(f"    seg {k}: {len(qs)} -> {len(qd)} samples, "
                  f"{s['length']:.3f} m, tip_err={err:.2e} m"
                  + (f", {fb} IK fallbacks" if fb else ""))

    # ---- every pen-up beat, priced before any of it is laid down ----------
    # The taxi stretch is a FACTOR on the pen-up blocks, so the total has to be
    # known first; and knowing it first is also what keeps `transit_s` equal to
    # the number the sequencer minimised whatever the stretch turns out to be.
    hov = [lifted_or_lower(spec, D["qd"][0], D["pts"][0], h_inv=h_inv,
                           pen_ext=pen_ext) for D in dense]
    hox = [lifted_or_lower(spec, D["qd"][-1], D["pts"][-1], h_inv=h_inv,
                           pen_ext=pen_ext) for D in dense]
    q_home = np.asarray(spec.q_seed, float).reshape(7)

    def need(beat, what, k):
        """Refuse rather than lay down a move the paper gate would not pass."""
        if beat is None:
            raise PaperRefused(f"arm {getattr(spec, 'arm_id', '?')}: {what} at "
                               f"segment {k} cannot clear the paper plane")
        return beat

    ent = need(enter_beats(spec, q0, hov[0][0], dense[0]["qd"][0], pen_ext,
                           h_inv, qd_frac, paper_safe), "entry", 0)
    beats, modes = [ent["steps"]], [ent["modes"]]
    hops = []
    for k, D in enumerate(dense):
        if k + 1 < len(dense):
            hop = float(np.linalg.norm(dense[k + 1]["pts"][0] - D["pts"][-1]))
            hops.append(hop)
            b = need(transit_beats(spec, D["qd"][-1], hox[k][0], hov[k + 1][0],
                                   dense[k + 1]["qd"][0], hop, hox[k][1],
                                   hov[k + 1][1], pen_ext, h_inv, transit_speed,
                                   qd_frac, paper_safe, q_home), "transit", k)
        elif park == PARK_HOME:
            b = need(exit_beats(spec, D["qd"][-1], hox[k][0], q_home, pen_ext,
                                h_inv, qd_frac, paper_safe), "go-home", k)
        else:                                     # freeze: lift, and stop
            r = _route(spec, D["qd"][-1], hox[k][0], pen_ext, h_inv,
                       -paper.TIP_TOL, qd_frac, T_LIFT_F, paper_safe)
            b = need(None if r is None else dict(steps=r[0], modes=(r[1],)),
                     "final lift", k)
        beats.append(b["steps"])
        modes.append(b["modes"])
    transit_s = float(sum(s[0] for b in beats for s in b))
    stretch = max(0.0, float(taxi_stretch))
    kf = 1.0 + stretch / transit_s if transit_s > 1e-12 and stretch else 1.0
    taxi_s = transit_s * (kf - 1.0)

    lifts.append(hov[0][1])
    add(0.0, q0)
    for i, (dt_, q_) in enumerate(beats[0]):
        t += kf * dt_
        if i == len(beats[0]) - 1:
            add(t, q_, 0, 0.0)
        else:
            add(t, q_)

    draw_len = transit_len = 0.0
    for k, D in enumerate(dense):
        dur = draw_duration(D["qd"], D["ud"], D["length"], draw_speed, qd_frac)
        for uu, q in zip(D["ud"][1:], D["qd"][1:]):
            add(t + uu * dur, q, k, uu)
        n_ch = int(np.clip(round(D["length"] / ink_chunk), 3, 60))
        edges = np.linspace(0, len(D["pts"]) - 1, n_ch + 1).astype(int)
        for c in range(n_ch):
            i0, i1 = edges[c], edges[c + 1]
            if i1 <= i0:
                continue
            xyz = np.column_stack([D["pts"][i0:i1 + 1],
                                   np.full(i1 - i0 + 1, INK_Z)])
            ink.append((float(t + dur * i1 / (len(D["pts"]) - 1)), xyz))
        phases.append(dict(kind="stroke", seg=k, t0=float(t), t1=float(t + dur)))
        t += dur
        draw_len += D["length"]

        b, t0 = beats[k + 1], t
        lifts.append(hox[k][1])
        if k + 1 < len(dense):
            lifts.append(hov[k + 1][1])
            transit_len += hops[k]
        for i, (dt_, q_) in enumerate(b):
            t += kf * dt_
            if i == len(b) - 1 and k + 1 < len(dense):
                add(t, q_, k + 1, 0.0)
            else:
                add(t, q_)
        phases.append(dict(kind="transit", seg=k, t0=float(t0), t1=float(t)))

    retreat_s = 0.0
    if retreat is not None and park != PARK_HOME:
        q_ret = np.asarray(retreat, float).reshape(7)
        z_ret = float(paper.chain_tip_z(q_ret[None, :], spec, pen_ext, h_inv)[1][0])
        r = _route(spec, hox[-1][0], q_ret, pen_ext, h_inv,
                   paper.travel_floor(hox[-1][1], z_ret), qd_frac, T_LIFT_F,
                   paper_safe)
        need(None if r is None else dict(steps=r[0]), "retreat", len(dense) - 1)
        retreat_s = float(sum(s[0] for s in r[0]))
        t0 = t
        for dt_, q_ in r[0]:
            t += dt_
            add(t, q_)
        beats.append(r[0])          # keep `paper_vias` counting the retreat too
        modes.append((r[1],))
        phases.append(dict(kind="retreat", seg=len(dense) - 1, t0=float(t0),
                           t1=float(t)))

    T = np.maximum.accumulate(np.asarray(T, float) + 1e-9 * np.arange(len(T)))
    return dict(t=T, q=np.array(Q), seg=np.array(S), u=np.array(U), phases=phases,
                ink=ink, duration=float(T[-1]), lifts=lifts, dense_tip_err=worst,
                draw_len=draw_len, transit_len=transit_len,
                transit_s=float(transit_s), taxi_s=float(taxi_s),
                retreat_s=float(retreat_s), q_end=np.array(Q[-1], float),
                park=str(park), pen=float(pen_ext),
                draw_s=float(T[-1] - transit_s - taxi_s - retreat_s),
                paper_modes=[m for mm in modes for m in mm],
                paper_vias=int(sum(len(b) for b in beats)
                               - sum(len(mm) for mm in modes)),
                fallbacks=int(sum(D["fallbacks"] for D in dense)))


def uniform_samples(prog, dt):
    """The frozen timeline on a uniform clock -> dict(q (N,7), seg, u, n).

    The coordination grid and the animation frames both live on this clock, so
    "progress index" means one unambiguous thing everywhere downstream: a
    scheduled arm at index p is at `q[p]`, full stop.
    """
    T, Q = prog["t"], prog["q"]
    n = max(int(np.ceil(prog["duration"] / dt)) + 1, 1)
    ts = np.arange(n) * dt
    q = np.column_stack([np.interp(ts, T, Q[:, j]) for j in range(7)])
    i = np.clip(np.searchsorted(T, ts, side="right") - 1, 0, len(T) - 2)
    same = prog["seg"][i] == prog["seg"][i + 1]
    f = (ts - T[i]) / np.maximum(T[i + 1] - T[i], 1e-12)
    seg = np.where(same, prog["seg"][i], -1)
    u = np.where(same, prog["u"][i] + f * (prog["u"][i + 1] - prog["u"][i]), 0.0)
    return dict(q=q, seg=seg.astype(int), u=u, n=n, dt=float(dt))


# --------------------------------------------------------------------------
# JSON cache (interpreter- and numpy-version-independent)
# --------------------------------------------------------------------------
def save_plan(plan, path):
    def enc(L):
        d = {k: v for k, v in L.items() if k != "strokes"}
        d["strokes"] = [{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                         for k, v in S.items()} for S in L["strokes"]]
        return d
    with open(path, "w") as f:
        json.dump([enc(L) for L in plan], f)


def load_plan(path):
    with open(path) as f:
        raw = json.load(f)
    for L in raw:
        for S in L["strokes"]:
            for k in ("pts", "qs", "sigmas", "margins", "qs_dense", "u_dense"):
                if k in S:
                    S[k] = np.asarray(S[k], float)
    return raw


def report(plan):
    """Terse per-letter summary lines."""
    out = []
    for L in plan:
        out.append(
            f"{L['name']} arm {L['arm_id']:>2}: {len(L['strokes'])} strokes, "
            f"{sum(S['length'] for S in L['strokes']):.2f} m, "
            f"centre ({L['center'][0]:.3f}, {L['center'][1]:.3f}) h={L['height']:.2f} "
            f"[{L['nudge']}], min_sigma={L['min_sigma']:.4f}, "
            f"min_margin={L['min_margin']:.3f}, tip_err<={L['max_tip_err']:.1e} m")
    return out
