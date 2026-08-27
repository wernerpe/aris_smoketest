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

from . import frames as _frames
from . import ik, letters, paper, planner
from .fleet import FLEET, H_INV_DEFAULT
from .frames import (FR3_MIN, FR3_MAX, PEN_EXT, QD_MAX, joint_margin, rotx,
                     rotz, tip_pos)

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


HOVER_YAWS = tuple(np.linspace(0, 2 * np.pi, 8, endpoint=False))
#   The TOOL YAW axis of the hover fiber, and it only became a real axis when
#   the pen holder went lateral: with the tip 110 mm off the wrist axis, phi
#   moves the whole arm around an 11 cm circle while the pen stays put.  Eight
#   of them is what `atlas._candidates` and `layout.certified_ready_pose` both
#   scan, so a hover is now searched over the same fiber every other CHOSEN
#   pose in this package is searched over.


def lifted_config(spec, q_ref, xy, z=LIFT_Z, h_inv=H_INV_DEFAULT,
                  pen_ext=PEN_EXT, span=0.6, n_q7=25, margin_min=HOVER_MARGIN,
                  tilt=None, phis=None, q7s=None, ok=None):
    """IK pose with the pen tip at (x, y, z), R = rotx(pi), nearest to q_ref.

    Scans q7 around q_ref[6] (the redundancy that q_ref already picked) and
    all analytic branches; returns the solution with the smallest ||dq||_inf
    among those keeping at least `margin_min` rad of joint-limit margin.  The
    filter is inside the scan and not applied afterwards, so raising it does not
    merely reject the nearest solution — it picks the nearest ACCEPTABLE one,
    which is usually a different q7 rather than no answer at all.

    `phis`, `q7s` and `ok` WIDEN THAT SCAN TO THE WHOLE FIBER, and they default
    to exactly what this function always did (one orientation, a local q7
    window, no extra gate) so the pose it returns where nothing is in the way
    is the pose it has always returned.

      `phis`  tool yaws to scan, each giving `rotz(phi) @ R`.  The pen points
              the same way and the ARM does not: with the lateral holder that
              is a metre of elbow travel for a tip that has not moved.
      `q7s`   the q7 values to scan; `None` is the local window above.  The
              whole grid is what a search that has ALREADY failed locally
              should be asking for.
      `ok`    a VECTORISED SCORE, (N,7) -> (N,) float, `-inf` for a pose that
              is not allowed at all and higher-is-better for one that is.  It
              is applied like the margin filter and for the same reason: a
              hover that fails must be replaced by the best acceptable hover,
              not by nothing.  `writing.static_gate` builds the one the lift
              layer needs.

              IT IS A SCORE AND NOT A PREDICATE BECAUSE THE THRESHOLD IS NOT
              THE WHOLE STORY.  "Nearest pose that clears 53 mm" lands hovers a
              millimetre over the gate, and then every straight line between
              two of them dips under it and the cost matrix spends its budget
              routing crossings between poses that were only just legal.  The
              score is the clearance CAPPED at a comfortable value, so
              candidates that are comfortable all tie and the nearest of them
              wins — one scan, one forward-kinematics pass, and hovers that
              stand off the metal wherever standing off is free.

    `tilt` LEANS THE HOVER, AND THE FLEET DOES NOT USE IT.  The hover above a
    tilt-rescued segment's endpoint could be asked for at the same lean, which
    would make the lift a pure translation of the tool along the paper normal.
    It was tried and it is NOT what ships, for two reasons that turned out to
    point the same way:

      A HOVER IS A TRAVEL POSE, AND THE ROUTER'S VIAS ARE VERTICAL.
      `paper.route` builds every detour it inserts out of `lifted_config`
      solutions at `rotx(pi)`, so a leaning hover does not avoid reorienting
      the pen during a transit — it only moves where the reorientation
      happens, from the lift into the crossing.  ONE hover convention for the
      whole fleet is worth more than a locally tidier lift.

      IT COST A CONDUCT.  With leaning hovers on the 100 %-coverage
      allocation, phase 1 was refused with "arm 31: go-home at segment 0
      cannot clear the paper plane" — the leaning hover is a different pose to
      leave from and `paper.route` would not fly it home.  The same allocation
      conducts with vertical hovers.

    So the pen reorients during the LIFT, with the tip off the paper and the
    whole move swept by `scene_check`, which is the cheapest place to put it.
    The argument is kept here rather than deleted because "why is the hover
    above a leaning stroke not leaning" is a question worth an answer.
    """
    Twb = spec.T_world_base(h_inv)
    if tilt is None:
        R_w = rotx(np.pi)
    else:
        from .tilt import pen_rot
        R_w = pen_rot(np.asarray(tilt, float).reshape(1, 2))[0]
    from .frames import tool_offset
    Twb_inv = np.linalg.inv(Twb)
    off = tool_offset(pen_ext)
    if q7s is None:
        q7s = np.clip(q_ref[6] + np.linspace(-span, span, n_q7),
                      FR3_MIN[6] + 1e-3, FR3_MAX[6] - 1e-3)
    else:
        q7s = np.clip(np.asarray(q7s, float).reshape(-1),
                      FR3_MIN[6] + 1e-3, FR3_MAX[6] - 1e-3)
    best, best_d, cand = None, np.inf, []
    for phi in ((None,) if phis is None else np.atleast_1d(phis)):
        R = R_w if phi is None else rotz(float(phi)) @ R_w
        T_w = np.eye(4)
        T_w[:3, :3] = R
        # tip = TCP + R @ tool_offset (ACTIVE tool): the hover puts the TIP at
        # (x, y, z) whichever holder is mounted, at whichever yaw.
        T_w[:3, 3] = np.array([xy[0], xy[1], z]) - R @ off
        T_b = Twb_inv @ T_w
        for q7 in q7s:
            for q in ik.solve(T_b, q7, q_ref):
                if joint_margin(q) < margin_min:
                    continue
                d = float(np.max(np.abs(q - q_ref)))
                if ok is None:
                    if d < best_d:
                        best, best_d = q, d
                elif d < best_d:
                    cand.append((d, q))
    if ok is not None and cand:
        # RANK FIRST, SCORE ONCE.  The score is a forward-kinematics pass
        # against thirty boxes, and the widened scan hands it five hundred
        # candidates; asking it one at a time was 70 % of a route's clock.
        # Sorted by ||dq||_inf, `argmax` returns the FIRST maximiser — so among
        # equally comfortable candidates the nearest wins, and where nothing is
        # comfortable the clearest wins.
        cand.sort(key=lambda t: t[0])
        Q = np.array([q for _, q in cand])
        s = np.asarray(ok(Q), float).reshape(-1)
        if np.isfinite(s).any():
            i = int(np.argmax(np.where(np.isfinite(s), s, -np.inf)))
            best, best_d = Q[i], float(cand[i][0])
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
            max_dq=MAX_DQ_FRAME, tilt=None, phi=0.0):
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

    `tilt` IS THE PEN ORIENTATION AND IT HAS TO COME ALONG.  The pose this
    re-solves at was hard-coded to `rotx(pi)` — the perpendicular pen — which
    is right for every plan the planner has ever made and WRONG for a
    tilt-rescued one (`aris_sixarm/tilt.py`): the plan's own samples would
    survive verbatim and every sample inserted between them would be solved
    for a different tool frame, so the executed stroke would rock the pen back
    to vertical and out to the lean between every pair of commanded points.
    Pass the plan's `tilt` field — (N,2), one lean vector per commanded sample
    — and the fill is solved on the interpolated orientation instead.

    THE INTERPOLATION IS OF THE VECTOR, never of (lean, azimuth): the chart
    (tx, ty) is regular at the apex and the polar one is not, so interpolating
    the angles through a zero crossing would swing the azimuth half a turn and
    spin the pen on the paper while the lean passed through nothing.
    """
    from .frames import rotz, tool_offset
    Twb = spec.T_world_base(h_inv)
    Twb_inv = np.linalg.inv(Twb)
    off = tool_offset(pen_ext)              # ACTIVE tool
    # `phi` IS THE TOOL YAW AND IT HAS TO COME ALONG for the same reason
    # `tilt` does: a lateral plan was solved at its phi, and filling between
    # its samples at phi = 0 would spin the wrist a quarter turn and back
    # between every pair of commanded points.  Scalar phi (a fixed-phi plan)
    # or per-sample (the phi-varying rescue, interpolated like tilt).
    ph = np.asarray(phi, float)
    ph_arr = ph.reshape(-1) if ph.ndim else None
    if ph_arr is not None and len(ph_arr) != len(qs):
        raise ValueError(f"phi has {len(ph_arr)} rows for {len(qs)} samples")
    phi0 = float(ph) if ph.ndim == 0 else 0.0
    R_w = rotz(phi0) @ rotx(np.pi) if phi0 else rotx(np.pi)
    T_w = np.eye(4)
    T_w[:3, :3] = R_w
    tl = None if tilt is None else np.asarray(tilt, float).reshape(-1, 2)
    if tl is not None and len(tl) != len(qs):
        raise ValueError(f"tilt has {len(tl)} rows for {len(qs)} samples")
    n = len(qs) - 1
    out_q, out_u, fallbacks = [qs[0]], [0.0], 0
    for i in range(n):
        dq = float(np.max(np.abs(qs[i + 1] - qs[i])))
        k = max(1, int(np.ceil(dq / max_dq)))
        for m in range(1, k):
            f = m / k
            p = pts[i] + f * (pts[i + 1] - pts[i])
            if tl is not None:
                from .tilt import pen_rot
                R_w = pen_rot(tl[i] + f * (tl[i + 1] - tl[i]))[0]
                T_w[:3, :3] = R_w
            elif ph_arr is not None:
                R_w = rotz(ph_arr[i] + f * (ph_arr[i + 1] - ph_arr[i])) \
                    @ rotx(np.pi)
                T_w[:3, :3] = R_w
            T_w[:3, 3] = np.array([p[0], p[1], 0.0]) - R_w @ off
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
    qd, ud, _ = densify(qs, pts, spec, h_inv, pen_ext,
                        tilt=seg["plan"].get("tilt"),
                        phi=seg["plan"].get("phi", 0.0))
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
    lift = _route(spec, q_exit, q_hover_exit, pen_ext, h_inv, paper.CONTACT_FLOOR,
                  qd_frac, T_LIFT_F, paper_safe)
    if lift is None:
        return None
    trav = _route(spec, q_hover_exit, q_hover_entry, pen_ext, h_inv,
                  paper.travel_floor(z_exit, z_entry), qd_frac,
                  max(T_TRAVEL_MIN, hop / max(transit_speed, 1e-9)),
                  paper_safe, q_home)
    if trav is None:
        return None
    low = _route(spec, q_hover_entry, q_entry, pen_ext, h_inv, paper.CONTACT_FLOOR,
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
    low = _route(spec, q_hover_entry, q_entry, pen_ext, h_inv, paper.CONTACT_FLOOR,
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
    lift = _route(spec, q_exit, q_hover_exit, pen_ext, h_inv, paper.CONTACT_FLOOR,
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


# ==========================================================================
# THE HOVER IS A POSE THE ARM HOLDS, AND NOTHING WAS CERTIFYING IT
# ==========================================================================
# `atlas.solve_cell` gates a DRAWING pose against the static set — every
# neighbour's steel and every neighbour's pose-invariant base column — and the
# hover 6 cm above it is a DIFFERENT CONFIGURATION.  Same tip, same paper cell,
# a completely different elbow: the analytic IK nearest the drawing pose at
# z = +0.06 can put the forearm a hundred millimetres inside a neighbour's
# column while the ink under it clears by fifty.
#
# Measured on the proposed rig at h = 0.940 (`out/transit_block.py`,
# `out/hover_fiber.py`): every one of the six arms' certified cells clears the
# columns while DRAWING — 0 failures, worst 97.3 mm — and 4 to 18 % of the same
# cells have a derived hover that does not, worst 131 mm INSIDE.  That is what
# every conduct refusal on this rig has been naming: "the impossible indices
# are pen-up transits", identically with one moving arm as with three, which
# can only be a stationary neighbour doing it.
#
# THE FIX IS NOT A HIGHER HOVER, IT IS THE FIBER THE DRAWING POSE ALREADY
# SPENT.  A stroke sample has to hold a tip position AND stay on a continuous
# band, so the band DP spends q7 and the tool yaw buying continuity.  A hover
# has to hold a tip position and nothing else — phi, q7, the IK branch and the
# height are all free up there.  So where the derived hover is blocked, SEARCH
# that fiber for one that is not: 8 yaws x the whole q7 grid x every branch,
# keeping the NEAREST acceptable pose so the lift stays short and the elbow
# stays near the ink the conductor already cleared.  It recovers 92 % of the
# blocked cells (173 of 189 sampled) and refuses the rest honestly.
HOVER_LADDER = (LIFT_Z, 0.045, 0.03, 0.09, 0.12)
#   ...and the height is on the fiber too.  The first three rungs are the ones
#   this function has always had; the two above them only ever run when all
#   three have failed, which used to mean "do not lift at all".

# A HOVER THAT SITS ON THE GATE MAKES EVERY TRANSIT OUT OF IT MARGINAL.  The
# fiber search keeps the NEAREST acceptable pose, and "acceptable" is a
# threshold, so left alone it lands hovers a millimetre over `FRAME_FLOOR` —
# and then the straight line between two such hovers dips under it, every
# crossing needs routing, and the cost matrix spends its whole budget proving
# that transits between poses which are only just legal are only just illegal.
#
# So the search asks for COMFORT first and settles for the floor second.  The
# tier costs nothing where the plain solver already had a comfortable answer
# (which is most cells: a hover 6 cm over the paper is usually nowhere near a
# column), and where it does run it buys 4 cm of clearance for a slightly
# longer lift.  Measured on the proposed rig it is the difference between half
# of every arm's crossings needing a route and a fifth of them.
HOVER_COMFORT = 0.04    # m of static clearance ABOVE the floor, if it is there

_HOVERS = {}                # (arm, pen, tool, h, q_ref, xy, tilt) -> (q, z)


def _clear_hovers():
    _HOVERS.clear()


paper.on_clear(_clear_hovers)


def static_gate(spec, pen_ext=PEN_EXT, h_inv=H_INV_DEFAULT, floor=None,
                comfort=None):
    """How good a CHOSEN pen-up pose is. -> fn (N,7) -> (N,) float.

    `-inf` for a pose that may not be held at all; otherwise the static
    clearance capped at `floor + comfort`, which is the ranking key
    `lifted_config` maximises (see there for why a threshold alone is not
    enough).

    Two conditions decide the `-inf`, and the second one is only here because
    the first widened the search:

      THE STATIC SET.  Every neighbour's steel and base column, at
      `paper.FRAME_FLOOR` — the FULL floor, not the endpoint-clamped one, since
      this pose is being CHOSEN and there is no reason to spend clearance you
      do not have to.  `paper.effective_static_floor` clamps only where a pose
      is being ACCEPTED: the certified ink at the ends of a leg.

      THE PAPER.  `atlas.solve_cell` refuses a drawing pose whose chain dips
      under `CHAIN_CLEAR`, and while the hover was pinned to phi = 0 directly
      above such a pose it inherited that for free.  A search over eight tool
      yaws does not: the same tip at the same height with the arm swung a
      quarter turn round the holder's 11 cm circle can put an elbow through the
      table.  So the hover is held to the floor its own ink is held to, and the
      widening cannot buy column clearance with the canvas.

    `None` when there is nothing in the room to hit and no gate to apply, which
    is also what makes the widened scan cost nothing on the legacy rigs: with
    no predicate to fail, `hover_solve` still reaches its second stage only
    when the first found no pose at all.
    """
    boxes = paper.static_boxes(spec) if paper.STATIC_SAFE else []
    fl = paper.FRAME_FLOOR if floor is None else float(floor)
    cap = fl + (HOVER_COMFORT if comfort is None else float(comfort))

    def score(Q):
        Q = np.asarray(Q, float).reshape(-1, 7)
        cz, _, sc = paper.chain_screen(Q, spec, pen_ext, h_inv, boxes)
        good = (cz >= paper.CHAIN_CLEAR - paper.EPS) & (sc >= fl - paper.EPS)
        # ...AND THE ARM AGAINST ITSELF.  A hover is a pose nobody drew with:
        # `lifted_config`'s only gate is the joint margin, and this score is
        # the whole of the rest.  A pen-up that lifts into a fold would have
        # passed everything (`aris_sixarm/selfcoll.py`).
        from . import selfcoll
        good &= selfcoll.self_ok(Q, margin=selfcoll.SELF_PLAN_MARGIN,
                                 pen_ext=pen_ext)
        return np.where(good, np.minimum(sc, cap), -np.inf)
    score.cap = cap          # `hover_solve` reads it to decide when to widen
    return score


def hover_solve(spec, q_ref, xy, z=LIFT_Z, h_inv=H_INV_DEFAULT,
                pen_ext=PEN_EXT, tilt=None, margin_min=HOVER_MARGIN, ok=None):
    """The best hover over (xy, z) that `ok` allows. -> q (7,) | None.

    Two stages, cheap first, and the first stage is bit-for-bit the call this
    module has always made: one orientation, q7 near the drawing pose's own.
    Where that answer exists and is COMFORTABLE — clear of the static set by
    `ok.cap`, which most of a canvas is — nothing else runs and nothing else
    changes.  Where it does not exist, or exists only just inside the gate, the
    second stage opens the whole fiber: 8 tool yaws x the whole q7 grid x every
    branch, ranked by the same score.  The escalation is bounded by geometry —
    it is the cells near a neighbour's column, and no others, that pay for it.

    AND "WHERE IT DOES NOT" IS NOT A CORNER CASE ON THE LATERAL TOOL.  phi = 0
    is a free choice for an INLINE pen — the pen is on the wrist axis, so
    rotating the tool about it moves nothing but q7 — and it is a hard
    constraint for a holder that puts the tip 110 mm off that axis: it pins the
    whole arm to one point of an 11 cm circle.  `atlas.solve_cell` certifies a
    drawing pose over EIGHT yaws for exactly that reason; the hover above it
    was pinned to one, and measured over 600 certified cells of the proposed
    rig, 330 of them — 55 % — had no hover at any of the three heights.  Not a
    blocked hover: no hover.  Those transits lifted the pen by nothing at all
    and dragged it across the paper to the next stroke, and the only thing that
    ever said so was `arm_program`'s `lifts` list full of zeros.  With the
    fiber open it is 15 of 1200.
    """
    q, _ = lifted_config(spec, q_ref, xy, z=z, h_inv=h_inv, pen_ext=pen_ext,
                         margin_min=margin_min, tilt=tilt, ok=ok)
    cap = getattr(ok, "cap", None)
    if q is not None and (cap is None
                          or float(np.asarray(ok(q[None, :]), float)[0])
                          >= cap - paper.EPS):
        return q
    w, _ = lifted_config(spec, q_ref, xy, z=z, h_inv=h_inv, pen_ext=pen_ext,
                         margin_min=margin_min, tilt=tilt, phis=HOVER_YAWS,
                         q7s=ik.Q7_GRID, ok=ok)
    if w is None or q is None:
        return w if w is not None else q
    # both exist: keep whichever the score prefers, the narrow one on a tie —
    # it is the nearer pose and the shorter lift
    s = np.asarray(ok(np.stack([q, w])), float)
    return q if s[0] >= s[1] else w


# ==========================================================================
# A HOVER IS A PLACE THE ARM HAS TO BE ABLE TO GET TO, AND FROM
# ==========================================================================
# `hover_solve` picks the hover for STATIC CLEARANCE and for nothing else, and
# on the proposed rig that leaves some of them in pockets: the pose is fine,
# and `paper.route` will not fly `q_seed -> hover` or `hover -> q_seed` at any
# shape on its ladder.  A span with an end like that can be drawn and cannot be
# left, so it can only ever be the last thing an arm does in a pass — which is
# what `allocate.fly_shrink` gives ink back to avoid.
#
# MEASURED, BECAUSE IT DECIDES WHETHER THIS TIER IS WORTH ITS COST.  Over the
# 128 certified span ends arms 31 and 71 hold at the shipped placement, 114
# join the depot both ways and 14 do not; of those 14, TEN have a pose on the
# same fiber — same tip, same height, a different tool yaw and q7 and IK branch
# — that does, found in 2 to 8 candidates.  The other four are at the edge of
# reach, where the ladder had already given up height (z = 0.03 or none), and
# nothing on the fiber helps them.
#
# So the fiber can be searched again, ONLY where the chosen hover is in a
# pocket, and the acceptance test is the whole approach and not half of it: the
# arm has to be able to LIFT onto the pose from the ink it just drew as well as
# fly home from it, or this would trade a dead depot leg for a dead lift.
# Where the plain choice is fine — which is most of a canvas — it costs one
# direct `paper.route` each way and changes nothing at all.
#
# ...AND IT IS OFF BY DEFAULT, BECAUSE THE COVERAGE IT BUYS IS NOT COVERAGE.
# Measured end to end at the shipped placement, primary allocation, against the
# same run with it off:
#
#     grey    4.7929 -> 4.8504 m drawn     (+57.5 mm)
#     orange  9.4349 -> 9.3534 m drawn     (-81.5 mm)
#     ALLOCATED  96.50 % -> 96.34 %
#
# It recovers the ends it was built for and it MOVES EVERY OTHER HOVER IT
# TOUCHES, and a hover 3.1 rad from the pose the arm just drew in is a
# different transit for the whole rest of the bag — the grey gain and the
# orange loss are the same mechanism twice.  So the machinery stays, measured,
# behind `--depot-hover`, and the fleet keeps the hover it had: `fly_shrink`
# already answers the pocket by giving the far end back, for a few centimetres
# of ink that the residual pass then offers to somebody else.
#
# ...AND THAT IS AN ARGUMENT AGAINST THE SWITCH, NOT AGAINST THE TIER.  Read
# the A/B again: the tier fires only at a pocket, and a pocket is only a
# PROBLEM where the allocator is about to pay for it.  Most of the 14 are not —
# `fly_shrink` gives a few centimetres back, or the span is the last thing the
# arm does anyway — and at those the tier still swaps a 6 cm lift for a pose
# most of a radian away and re-prices the whole bag around it.  That is the
# -81.5 mm.  So the tier gets an ALLOW-SET: `HOVER_DEPOT_SITES` is None for the
# old global switch and a `set()` of span-END identities for the selective one,
# which `allocate.fly_shrink` fills in one end at a time and only when the
# admission is what makes the span round-trippable.  A site outside the set
# takes the `sel = False` branch and the `sel = False` memo slot — the same
# code and the same key a tier-off run computes, so "every other hover is
# untouched" is a property of the key and not a measurement.
HOVER_DEPOT_AWARE = False
# HOW DEEP THE SEARCH GOES, AND WHY IT IS NOT 24 ANY MORE.  The budget was
# picked when the tier was measured on 14 pockets and the ten it could fix were
# found "in 2 to 8 candidates".  Re-measured under the self-collision guard over
# 1 127 certified cells of all six arms (`out/guard_pocket.py`), the fiber holds
# a depot-joining pose at 267 pockets' worth of span end and finds it at:
#
#     <=  4 tries  52.2 %      <= 24 tries  93.5 %
#     <=  8 tries  71.7 %      <= 48 tries  98.6 %
#     <= 16 tries  88.4 %      <= 64 tries 100.0 %
#
# so 24 leaves 6.5 % of the recoverable ends on the table.  The budget is spent
# only where a pocket is ABOUT TO COST INK (`allocate._rescue_pocket`'s
# allow-set), the median success is at 3.5 candidates, and it is the FAILURES
# that pay it — so 48 doubles the cost of a pocket nothing can fix and buys back
# most of what 24 was giving away.  64 buys the last 1.4 % for another third
# again, and is not taken.
HOVER_DEPOT_TRIES = 48      # fiber poses routed before the pocket is accepted
HOVER_DEPOT_SITES = None    # None = every pocket; a set = only these ends

# ...AND THE LEAN IS ON THE FIBER TOO, LAST, AND ONLY IF THE RUN ALLOWS ONE.
# `lifted_config`'s note argues at length that the FLEET should not hover at a
# lean — the router's vias are vertical, one hover convention is worth more than
# a locally tidier lift, and a leaning hover once cost a conduct.  Every word of
# that is about the hover a span gets when a vertical one exists.  It says
# nothing about the end where NO vertical pose on the whole fiber joins the
# depot, which is the only place this rung runs: there the alternative is not a
# tidier lift, it is giving the ink back.
#
# Measured over the same 1 127 cells, the lean rungs find 9 of the 138
# recoverable pockets that no vertical pose could reach (7 at 7.5 deg, 2 at 15).
# Small, and it is the difference between a span being drawn and not at the ends
# it fires on.  `HOVER_LEAN_MAX_DEG` is the RUN's cone — `csail_allocate` sets
# it from `--tilt-max-deg`, so a flat run stays flat and gets exactly the
# answers it got before this existed.
HOVER_LEAN_MAX_DEG = 0.0    # deg; the cone the run permits (0 = vertical only)
HOVER_DEPOT_LEANS = (7.5, 15.0)      # deg, tried in order, after the flat fiber
HOVER_DEPOT_LEAN_TRIES = 16          # ...each with its own, smaller budget


def _lean_rungs(cap=None):
    """The pen leans a rescue may try, inside the run's cone. -> [(pitch, 0)].

    Degrees to the (pitch, roll) pair `tilt.pen_rot` takes, dropping any rung
    the run's `--tilt-max-deg` does not permit.  Empty on a flat run, which is
    what keeps this inert there.
    """
    cap = HOVER_LEAN_MAX_DEG if cap is None else float(cap)
    return [(float(np.deg2rad(d)), 0.0) for d in HOVER_DEPOT_LEANS
            if d <= cap + 1e-9]


def hover_site(spec, q_ref, xy):
    """The identity of ONE SPAN END, for the depot-hover allow-set. -> tuple.

    The part of `lifted_or_lower`'s memo key that names the end rather than the
    tool or the run: which arm, the drawing pose it lifts off, and the point it
    lifts over.  `allocate.fly_shrink` admits these; nothing else may.
    """
    return (int(getattr(spec, "arm_id", -1)),
            np.round(np.asarray(q_ref, float), 9).tobytes(),
            np.round(np.asarray(xy, float), 9).tobytes())


def depot_hover_selected(spec, q_ref, xy):
    """Does the depot-aware tier fire at this end? -> bool."""
    if not HOVER_DEPOT_AWARE:
        return False
    return (HOVER_DEPOT_SITES is None
            or hover_site(spec, q_ref, xy) in HOVER_DEPOT_SITES)


def hover_fiber(spec, q_ref, xy, z, ok, h_inv=H_INV_DEFAULT, pen_ext=PEN_EXT,
                tilt=None, margin_min=HOVER_MARGIN):
    """Every gated pose on the hover fiber over (xy, z), nearest first. -> (K,7).

    `hover_solve` keeps one; this keeps them all, in the order `hover_solve`
    would have preferred them, so a caller with a SECOND question to ask (can
    the arm get home from it?) can walk the same ranking instead of inventing
    another one.
    """
    Twb_inv = np.linalg.inv(spec.T_world_base(h_inv))
    R_w = rotx(np.pi)
    if tilt is not None:
        from .tilt import pen_rot
        R_w = pen_rot(np.asarray(tilt, float).reshape(1, 2))[0]
    from .frames import tool_offset
    off = tool_offset(pen_ext)
    q7s = np.clip(np.asarray(ik.Q7_GRID, float).reshape(-1),
                  FR3_MIN[6] + 1e-3, FR3_MAX[6] - 1e-3)
    cand = []
    for phi in HOVER_YAWS:
        R = rotz(float(phi)) @ R_w
        T_w = np.eye(4)
        T_w[:3, :3] = R
        T_w[:3, 3] = np.array([xy[0], xy[1], z]) - R @ off
        T_b = Twb_inv @ T_w
        for q7 in q7s:
            for q in ik.solve(T_b, q7, q_ref):
                if joint_margin(q) < margin_min:
                    continue
                cand.append((float(np.max(np.abs(q - q_ref))), q))
    if not cand:
        return np.zeros((0, 7))
    cand.sort(key=lambda t: t[0])
    Q = np.array([q for _, q in cand])
    if ok is None:
        return Q
    return Q[np.isfinite(np.asarray(ok(Q), float).reshape(-1))]


def hover_joins_depot(spec, q_ref, h, h_inv=H_INV_DEFAULT, pen_ext=PEN_EXT):
    """Can the arm lift onto this hover, fly home from it, and fly back? -> bool.

    The three legs `writing` will actually lay down and `sequence.depot_legs`
    will actually price — the lift at the contact floor, and the two halves of
    a trip home at the flying floor — asked of one candidate pose.
    """
    if paper.route(spec, q_ref, h, pen_ext=pen_ext, h_inv=h_inv,
                   tip_floor=paper.CONTACT_FLOOR) is None:
        return False
    q0 = np.asarray(spec.q_seed, float).reshape(7)
    fl = paper.travel_floor(LIFT_Z, LIFT_Z)
    return (paper.route(spec, q0, h, pen_ext=pen_ext, h_inv=h_inv,
                        tip_floor=fl) is not None
            and paper.route(spec, h, q0, pen_ext=pen_ext, h_inv=h_inv,
                            tip_floor=fl) is not None)


def lifted_or_lower(spec, q_ref, xy, heights=HOVER_LADDER, h_inv=H_INV_DEFAULT,
                    pen_ext=PEN_EXT, tilt=None):
    """A CERTIFIED hover over `xy`, as high as the arm can hold one.
    -> (q, height_used).

    Near the edge of an arm's reach the 6 cm hover has no IK solution with the
    margin we insist on even though the stroke's endpoint does; dropping the
    hover is strictly better than dropping the transit.  A last resort of "do
    not lift at all" keeps the timeline well-formed (the pen scuffs the paper,
    which the report says out loud rather than hiding) — and it is a pose the
    atlas certified, so the leg out of it is clamped rather than contradictory.

    Memoised because the balancer prices the same span dozens of times and the
    second stage of `hover_solve` is a 500-solution scan; `paper.clear_cache`
    drops this with the rest.

    THE MEMO IS KEYED ON WHETHER THE TIER FIRES *HERE*, not on whether it is
    switched on.  `allocate.fly_shrink` admits a site in the middle of a run and
    does not clear anything: every other end keeps computing under `sel = False`
    and hitting the `sel = False` slot it already filled, which is bit for bit
    the entry a tier-off run stores.  Only the admitted end gets a second slot.
    """
    sel = depot_hover_selected(spec, q_ref, xy)
    key = (id(spec), float(pen_ext), float(_frames.PEN_LAT), float(h_inv),
           np.round(np.asarray(q_ref, float), 9).tobytes(),
           np.round(np.asarray(xy, float), 9).tobytes(),
           None if tilt is None else np.round(np.asarray(tilt, float),
                                              9).tobytes(),
           tuple(float(z) for z in heights), bool(paper.STATIC_SAFE),
           bool(sel),
           # ...AND EVERYTHING THE ANSWER DEPENDS ON, which is the lesson the
           # q_home flag cost this module once already: the rescue's depth and
           # the run's lean cone both change what comes back, so a run that
           # changes either may not read an answer computed under the other.
           int(HOVER_DEPOT_TRIES), float(HOVER_LEAN_MAX_DEG),
           tuple(float(d) for d in HOVER_DEPOT_LEANS))
    hit = _HOVERS.get(key)
    if hit is not None:
        return hit

    def ladder(ok):
        for z in heights:
            q = hover_solve(spec, q_ref, xy, z=z, h_inv=h_inv, pen_ext=pen_ext,
                            tilt=tilt, ok=ok)
            if q is not None:
                return np.asarray(q, float), float(z)
        return None

    gate = static_gate(spec, pen_ext, h_inv)
    out = ladder(gate)
    if out is None and gate is not None:
        # ...AND THE HOVER IS CLAMPED TO ITS OWN INK, for the third time and
        # the same reason (`paper.effective_static_floor`).  A cell certified
        # at 51 mm has no hover anywhere on its fiber that keeps 53, so the
        # first ladder returns nothing and the arm gets "do not lift at all" —
        # a pen dragged across the paper to save a clearance the ink under it
        # never had.  Asking the hover for what the drawing pose holds is the
        # honest question: such a hover ships exactly when its own ink does.
        fl = float(paper.chain_static(np.asarray(q_ref, float).reshape(1, 7),
                                      spec, pen_ext, h_inv)[0])
        if fl < paper.FRAME_FLOOR:
            out = ladder(static_gate(spec, pen_ext, h_inv, floor=fl))
    if out is None:
        out = (np.asarray(q_ref, float), 0.0)
    if sel and gate is not None \
            and not hover_joins_depot(spec, q_ref, out[0], h_inv, pen_ext):
        # ...AND THE POSE HAS TO BE SOMEWHERE THE ARM CAN GET TO AND FROM.
        # Only ever reached where the chosen hover is in a pocket AND this end
        # is one the allow-set names; the ranking is `hover_solve`'s own, so
        # the NEAREST acceptable pose still wins and the lift stays as short as
        # the pocket allows.  `sel`, not `HOVER_DEPOT_AWARE`: the branch and
        # the memo key have to agree about whether the tier fired HERE, or an
        # answer computed under one is filed under the other.
        #
        # ...AND THE RUNGS ARE IN ASCENDING DISTURBANCE.  The vertical fiber
        # first — same tip, same height, a different tool yaw and q7 and IK
        # branch, and the nearest acceptable pose still wins — then the heights
        # the ladder already had, and only then a LEAN, which is the one rung
        # that changes what the PEN is doing rather than where the elbow is.
        # Each rung carries its own budget, so a pocket nothing can fix does
        # not spend the leaned budget over and over on the way to finding out.
        #
        # ...AND IT RUNS AT AN END WITH NO HOVER AT ALL, which the first cut
        # excluded with `out[1] > 0`.  Such an end is not a comfortable case to
        # leave alone: it is `(q_ref, 0.0)` — the arm drags the pen across the
        # paper to the next stroke — and if a leaned pose on the fiber can be
        # lifted onto and flown home from, it is better by every measure.
        for lean, budget in ([(tilt, HOVER_DEPOT_TRIES)]
                             + [(t, HOVER_DEPOT_LEAN_TRIES)
                                for t in _lean_rungs()]):
            n, got = 0, None
            for z in heights:
                for cand in hover_fiber(spec, q_ref, xy, z, gate, h_inv,
                                        pen_ext, lean):
                    if n >= budget:
                        break
                    n += 1
                    if hover_joins_depot(spec, q_ref, cand, h_inv, pen_ext):
                        got = (np.asarray(cand, float), float(z))
                        break
                if got is not None or n >= budget:
                    break
            if got is not None:
                out = got
                break
    _HOVERS[key] = out
    return out


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

    A BAG IS NOT A TOUR.  A segment carrying `home_before=True` is entered from
    the READY POSE rather than from the segment before it: the arm lifts, flies
    home, and flies out again — `exit_beats` then `enter_beats`, the same two
    legs the pass pays at its own ends, laid end to end as one pen-up block.
    `sequence.close_depot` is what puts the flag there and
    `sequence.home_legs` prices it, so the seconds this lays down are the
    seconds the sequencer minimised and `csail_schedule.cross_check` still
    holds to the float.  The flag is ignored on the FIRST segment, which is
    entered from `q_start` by definition; slicing a programme (`split_by_arms`,
    `split_by_segments`) therefore needs no fixing up.

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
                    fallbacks=0, n_home=0)

    dense = []
    for k, s in enumerate(segs):
        qs = np.asarray(s["plan"]["qs"], float)
        pts = np.asarray(s["plan"]["pts"], float)
        tl = s["plan"].get("tilt")
        qd, ud, fb = densify(qs, pts, spec, h_inv, pen_ext, tilt=tl,
                             phi=s["plan"].get("phi", 0.0))
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
    # VERTICAL, even above a leaning stroke — see `lifted_config`'s `tilt`.
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
    hops, n_home = [], 0
    for k, D in enumerate(dense):
        if k + 1 < len(dense):
            hop = float(np.linalg.norm(dense[k + 1]["pts"][0] - D["pts"][-1]))
            hops.append(hop)
            if segs[k + 1].get("home_before"):
                go = need(exit_beats(spec, D["qd"][-1], hox[k][0], q_home,
                                     pen_ext, h_inv, qd_frac, paper_safe),
                          "go-home", k)
                come = need(enter_beats(spec, q_home, hov[k + 1][0],
                                        dense[k + 1]["qd"][0], pen_ext, h_inv,
                                        qd_frac, paper_safe), "entry", k + 1)
                b = dict(steps=go["steps"] + come["steps"],
                         modes=tuple(go["modes"]) + tuple(come["modes"]))
                n_home += 1
            else:
                b = need(transit_beats(spec, D["qd"][-1], hox[k][0],
                                       hov[k + 1][0], dense[k + 1]["qd"][0],
                                       hop, hox[k][1], hov[k + 1][1], pen_ext,
                                       h_inv, transit_speed, qd_frac,
                                       paper_safe, q_home), "transit", k)
        elif park == PARK_HOME:
            b = need(exit_beats(spec, D["qd"][-1], hox[k][0], q_home, pen_ext,
                                h_inv, qd_frac, paper_safe), "go-home", k)
        else:                                     # freeze: lift, and stop
            r = _route(spec, D["qd"][-1], hox[k][0], pen_ext, h_inv,
                       paper.CONTACT_FLOOR, qd_frac, T_LIFT_F, paper_safe)
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
                fallbacks=int(sum(D["fallbacks"] for D in dense)),
                n_home=int(n_home))


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
