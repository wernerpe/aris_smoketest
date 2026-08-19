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

from . import ik, letters, planner
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
def lifted_config(spec, q_ref, xy, z=LIFT_Z, h_inv=H_INV_DEFAULT,
                  pen_ext=PEN_EXT, span=0.6, n_q7=25):
    """IK pose with the pen tip at (x, y, z), R = rotx(pi), nearest to q_ref.

    Scans q7 around q_ref[6] (the redundancy that q_ref already picked) and
    all analytic branches; returns the solution with the smallest ||dq||_inf.
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
            if joint_margin(q) < 0.10:
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
# lift, its segments in the allocator's nearest-neighbour order, a pen-up
# transit between each pair, exit lift — as a frozen path with a nominal clock.
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


def _dq_time(q0, q1, frac=QD_FRAC, tmin=0.0):
    """Shortest time a straight joint-space move may take. -> seconds."""
    d = np.abs(np.asarray(q1, float) - np.asarray(q0, float))
    return float(max(tmin, np.max(d / (QD_MAX * max(frac, 1e-6)))))


def _draw_time(qd, ud, dur, frac=QD_FRAC):
    """`dur` stretched until no joint exceeds `frac` of its velocity limit."""
    du = np.diff(np.asarray(ud, float))
    dq = np.abs(np.diff(np.asarray(qd, float), axis=0))
    need = dq / (QD_MAX * max(frac, 1e-6)) / np.maximum(du, 1e-12)[:, None]
    return float(max(dur, need.max() if need.size else 0.0))


def lifted_or_lower(spec, q_ref, xy, heights=(LIFT_Z, 0.045, 0.03), h_inv=H_INV_DEFAULT):
    """`lifted_config`, retrying at lower heights. -> (q, height_used).

    Near the edge of an arm's reach the 6 cm hover has no IK solution with the
    margin we insist on even though the stroke's endpoint does; dropping the
    hover is strictly better than dropping the transit.  A last resort of "do
    not lift at all" keeps the timeline well-formed (the pen scuffs the paper,
    which the report says out loud rather than hiding).
    """
    for z in heights:
        q, _ = lifted_config(spec, q_ref, xy, z=z, h_inv=h_inv)
        if q is not None:
            return np.asarray(q, float), float(z)
    return np.asarray(q_ref, float), 0.0


def arm_program(spec, segs, draw_speed=DRAW_SPEED_FLEET, transit_speed=TRANSIT_SPEED,
                h_inv=H_INV_DEFAULT, ink_chunk=INK_CHUNK, qd_frac=QD_FRAC,
                verbose=False):
    """One arm's frozen nominal timeline from its allocated segments.

    `segs` are `allocate.allocate`'s programme entries, already in the order the
    arm will draw them; each carries the certified `plan` whose dense `qs`/`pts`
    are the joint trajectory and the curve it was certified against.

    -> dict(t, q, seg, u, phases, ink, duration, lifts, dense_tip_err)
       t     (K,)    waypoint times, strictly increasing
       q     (K,7)   waypoint joints
       seg   (K,)    index into `segs` while drawing, -1 while not
       u     (K,)    normalised arc position within that segment
       ink   list of (t_visible, chunk_xyz (M,3))
    """
    t, T, Q, S, U = 0.0, [], [], [], []
    ink, phases, lifts, worst = [], [], [], 0.0

    def add(tt, q, s=-1, u=0.0):
        T.append(float(tt))
        Q.append(np.asarray(q, float))
        S.append(int(s))
        U.append(float(u))

    if not segs:                                  # an arm that reaches nothing
        add(0.0, spec.q_seed)
        add(1.0, spec.q_seed)
        return dict(t=np.array(T), q=np.array(Q), seg=np.array(S), u=np.array(U),
                    phases=[], ink=[], duration=0.0, lifts=[], dense_tip_err=0.0,
                    draw_len=0.0, transit_len=0.0)

    dense = []
    for k, s in enumerate(segs):
        qs = np.asarray(s["plan"]["qs"], float)
        pts = np.asarray(s["plan"]["pts"], float)
        qd, ud, fb = densify(qs, pts, spec, h_inv)
        ref = np.column_stack([np.interp(ud, np.linspace(0, 1, len(pts)), pts[:, 0]),
                               np.interp(ud, np.linspace(0, 1, len(pts)), pts[:, 1])])
        err = tip_error_pts(qd, ref, spec, h_inv)
        worst = max(worst, err)
        dense.append(dict(qd=qd, ud=ud, pts=pts, fallbacks=fb, tip_err=err,
                          length=float(s["length"])))
        if verbose:
            print(f"    seg {k}: {len(qs)} -> {len(qd)} samples, "
                  f"{s['length']:.3f} m, tip_err={err:.2e} m"
                  + (f", {fb} IK fallbacks" if fb else ""))

    q_lift0, z0 = lifted_or_lower(spec, dense[0]["qd"][0], dense[0]["pts"][0], h_inv=h_inv)
    lifts.append(z0)
    add(0.0, spec.q_seed)
    t += _dq_time(spec.q_seed, q_lift0, qd_frac, T_HOME_F)
    add(t, q_lift0)
    t += _dq_time(q_lift0, dense[0]["qd"][0], qd_frac, T_LOWER_F)
    add(t, dense[0]["qd"][0], 0, 0.0)

    draw_len = transit_len = 0.0
    for k, D in enumerate(dense):
        dur = _draw_time(D["qd"], D["ud"], D["length"] / draw_speed, qd_frac)
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

        nxt = dense[k + 1] if k + 1 < len(dense) else None
        q_end, z1 = lifted_or_lower(spec, D["qd"][-1], D["pts"][-1], h_inv=h_inv)
        lifts.append(z1)
        t0 = t
        t += _dq_time(D["qd"][-1], q_end, qd_frac, T_LIFT_F)
        add(t, q_end)
        if nxt is None:
            t += _dq_time(q_end, spec.q_seed, qd_frac, T_HOME_F)
            add(t, spec.q_seed)
        else:
            q_next, z2 = lifted_or_lower(spec, nxt["qd"][0], nxt["pts"][0], h_inv=h_inv)
            lifts.append(z2)
            hop = float(np.linalg.norm(nxt["pts"][0] - D["pts"][-1]))
            transit_len += hop
            t += _dq_time(q_end, q_next, qd_frac,
                          max(T_TRAVEL_MIN, hop / transit_speed))
            add(t, q_next)
            t += _dq_time(q_next, nxt["qd"][0], qd_frac, T_LOWER_F)
            add(t, nxt["qd"][0], k + 1, 0.0)
        phases.append(dict(kind="transit", seg=k, t0=float(t0), t1=float(t)))

    T = np.maximum.accumulate(np.asarray(T, float) + 1e-9 * np.arange(len(T)))
    return dict(t=T, q=np.array(Q), seg=np.array(S), u=np.array(U), phases=phases,
                ink=ink, duration=float(T[-1]), lifts=lifts, dense_tip_err=worst,
                draw_len=draw_len, transit_len=transit_len,
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
