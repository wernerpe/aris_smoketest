"""The paper plane as an obstacle for PEN-UP motion.

THE HOLE THIS FILLS.  Everything that touches the paper was certified against
it and nothing that flies over it was.  `stroke_api` certifies a stroke, and
`validate.validate_plan` re-derives its chain clearance (`Z_CLEAR`, 2 cm) sample
by sample; `validate.check_pose` re-derives the same gate for a pose an arm
stands in.  But a pen-up transit is none of those things: `writing.arm_program`
lays it down as a STRAIGHT LINE IN JOINT SPACE between two hover poses, both of
which are certified, and no one ever looked at what the line does in between.

What it does, on the shipped `csail_final6` timeline, is go through the table.
Three of its forty-two pen-up blocks swing the wrist and the pen below z = 0 —
worst pen tip 253.6 mm under the canvas (arm 97 on its way home, t = 49.33 s),
worst chain point 156.1 mm under it (arm 2 mid-transit, t = 29.43 s) — and one
of them PARKS THERE, held 207 mm below the paper for three quarters of a second
by a conductor pause the conductor had no reason to think was unsafe.  The
entry and exit corridors are safe because they are short and vertical; it is
the long reconfiguration — a metre of travel that swaps elbow branch on the way
— that dives.

So the paper joins the frame boxes as a thing a pen-up move has to be certified
against, and the certification has the same shape as the rest of this repo's:
sample densely, measure with the same kinematics the timeline will be played
back with, and REFUSE rather than hope.  When a direct move is refused, `route`
buys clearance the cheapest way there is — it lifts higher and flies over —
and hands back via-configurations that are themselves IK solutions through
`writing.lifted_config`'s gates, so nothing downstream has to trust an
interpolation.

The two floors, and why they are what they are:

  CHAIN_CLEAR   2 cm, and it is `validate.Z_CLEAR` imported rather than
                restated.  This is the number every certified stroke already
                keeps; a transit that keeps less is a transit that would have
                been refused had it been ink.
  TIP_CLEAR     2 cm for the pen tip while it is FLYING.  The tip is not a
                chain point (the FK chain stops at the flange, one pen length
                short of the paper), which is exactly how a 110 mm pen can be
                under the table while every link the validator looks at is
                above it.  2 cm is the paper's own keep-out, applied to the one
                body that is closest to it, and it is never demanded above the
                hover the arm actually achieved: `lifted_or_lower` drops to
                45 mm, 30 mm, or to no lift at all near the edge of reach, and
                a floor higher than the hover would refuse a move whose
                endpoints are both fine.
  TIP_TOL       10 mm, the CONTACT band.  A lift starts and a lower ends
                with the pen ON the paper by construction, and the certified
                ink is only on the curve to `validate.TIP_TOL` (2 mm), so a
                gate at exactly zero would refuse the drawing.  10 mm is an
                order of magnitude above that band and the resampling residual
                on top of it (measured: 3.3 mm at the animation's own frame
                rate), and an order of magnitude below the smallest real
                violation there has ever been (175 mm).  It is deliberately not
                a tight number: the tip gate is a second net, and the CHAIN
                gate is what does the precise work — all three of the shipped
                timeline's bad blocks miss it by 67 to 156 mm.

Measured against those floors the shipped timeline's own numbers are not close
to the line: 39 of its 42 pen-up blocks clear 20 mm with room to spare and the
other 3 are 175-254 mm the wrong side of it.  The gate discriminates by two
orders of magnitude, which is the property that makes it worth having.
"""
import numpy as np

from . import rig_final
from .fleet import H_INV_DEFAULT
from .frames import PEN_EXT, fk_many, tip_pos_many
from .validate import Z_CLEAR

CHAIN_CLEAR = Z_CLEAR      # m, every chain point above the paper (= 0.02)
TIP_CLEAR = 0.02           # m, the pen tip while flying
TIP_TOL = 0.010            # m, the contact band (lift/lower/ink)
EPS = 1e-9

# Extra hover heights `route` climbs to when a direct move is refused.  Low
# first: a via costs joint-space seconds and the sequencer pays them, so the
# cheapest certified escape wins.  The top of the ladder is 40 cm, well above
# anything the logo needs, so "no route" means the geometry and not the ladder.
VIA_HEIGHTS = (0.08, 0.12, 0.15, 0.18, 0.25, 0.32, 0.40)

SAMPLES = 33               # configurations sampled along one straight move

_CACHE = {}                # (spec key, pen, h_inv, q0, q1, floors) -> result
_LIFTS = {}                # (spec key, pen, h_inv, q, z) -> hover pose | None


def clear_cache():
    """Drop the memos.  Tests that mutate a fleet in place need this."""
    _CACHE.clear()
    _LIFTS.clear()


def _key(spec, q0, q1, pen_ext, h_inv, tip_floor, chain_floor):
    return (id(spec), float(pen_ext), float(h_inv),
            np.round(np.asarray(q0, float), 9).tobytes(),
            np.round(np.asarray(q1, float), 9).tobytes(),
            round(float(tip_floor), 9), round(float(chain_floor), 9))


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------
def chain_tip_z(qs, spec, pen_ext=PEN_EXT, h_inv=H_INV_DEFAULT):
    """World z of the lowest chain point and of the pen tip. -> (M,), (M,).

    The chain minimum skips point 0 (the base origin, bolted to its mount) for
    the same reason `validate.validate_plan` does; the pen tip is returned
    SEPARATELY and never folded into that minimum, because the whole failure
    mode this module exists for is a tip below a chain that is above.
    """
    qs = np.asarray(qs, float).reshape(-1, 7)
    Twb = spec.T_world_base(h_inv)
    R, t = Twb[:3, :3], Twb[:3, 3]
    _, p = fk_many(qs)                              # (M,9,3), base frame
    chain_z = (p @ R.T + t)[:, 1:, 2].min(axis=1)
    tip_z = (tip_pos_many(qs, pen_ext) @ R.T + t)[:, 2]
    return chain_z, tip_z


def line_samples(q0, q1, n=SAMPLES):
    """The straight joint-space move, sampled. -> (n,7)."""
    f = np.linspace(0.0, 1.0, int(n))[:, None]
    return np.asarray(q0, float).reshape(1, 7) * (1.0 - f) \
        + np.asarray(q1, float).reshape(1, 7) * f


def line_clearance(spec, q0, q1, pen_ext=PEN_EXT, h_inv=H_INV_DEFAULT,
                   n=SAMPLES):
    """Worst chain and tip height along one straight move. -> (chain, tip)."""
    cz, tz = chain_tip_z(line_samples(q0, q1, n), spec, pen_ext, h_inv)
    return float(cz.min()), float(tz.min())


def effective_floors(spec, q0, q1, pen_ext=PEN_EXT, h_inv=H_INV_DEFAULT,
                     tip_floor=TIP_CLEAR, chain_floor=CHAIN_CLEAR):
    """The floors a move can actually be held to. -> (tip, chain).

    A MOVE IS NEVER ASKED TO KEEP MORE CLEARANCE THAN ITS OWN ENDPOINTS HAVE.
    The endpoints are given — a certified stroke pose, a hover the arm could
    reach, the ready pose it parks in — and a floor above them is not a
    constraint, it is a contradiction: no route can satisfy it, so `route`
    refuses, `cost_matrix` prices the edge `inf`, and every ordering disappears.

    That is not hypothetical.  With a 200 mm pen the INVERTED READY POSE holds
    its pen tip 16 mm BELOW the paper (the long-standing warning behind
    `scene_check.PEN_PAPER`), so asking the trip home to keep the flying floor
    of 20 mm made every depot edge infinite and the sequencer reported "no
    feasible order over 5 segments".  Clamping to the endpoints keeps the gate
    meaningful where it can be met and inert where the geometry already lost —
    and the timeline-wide gate in `scene_check` still has the last word on
    whether such a pose may ship at all.
    """
    ends = np.stack([np.asarray(q0, float).reshape(7),
                     np.asarray(q1, float).reshape(7)])
    cz, tz = chain_tip_z(ends, spec, pen_ext, h_inv)
    return (float(min(tip_floor, tz.min())), float(min(chain_floor, cz.min())))


def move_ok(spec, q0, q1, pen_ext=PEN_EXT, h_inv=H_INV_DEFAULT,
            tip_floor=TIP_CLEAR, chain_floor=CHAIN_CLEAR, n=SAMPLES):
    """Does the straight move keep both floors? -> (ok, chain, tip)."""
    tip_floor, chain_floor = effective_floors(spec, q0, q1, pen_ext, h_inv,
                                              tip_floor, chain_floor)
    cz, tz = line_clearance(spec, q0, q1, pen_ext, h_inv, n)
    return bool(cz >= chain_floor - EPS and tz >= tip_floor - EPS), cz, tz


def path_clearance(spec, qs, pen_ext=PEN_EXT, h_inv=H_INV_DEFAULT, n=SAMPLES):
    """Worst chain and tip height along a polyline of configurations."""
    qs = [np.asarray(q, float).reshape(7) for q in qs]
    cz, tz = np.inf, np.inf
    for a, b in zip(qs[:-1], qs[1:]):
        c, t = line_clearance(spec, a, b, pen_ext, h_inv, n)
        cz, tz = min(cz, c), min(tz, t)
    return float(cz), float(tz)


def frame_clearance(qs, spec, pen_ext=PEN_EXT, h_inv=H_INV_DEFAULT):
    """Worst clearance from the chain (pen tip included) to the rig's steel.

    A VIA THAT TRADES A PAPER HIT FOR A FRAME HIT IS NOT A FIX, and the two
    failure modes pull in opposite directions: the way out of the paper is UP,
    and up is where the top rails, the corner posts and the booms are.  On the
    first routed conduct of this logo `scene_check` refused a profile at
    49.9 mm of frame clearance against a 50 mm margin — 0.1 mm — with the paper
    gate passing comfortably, which is exactly the trade this exists to stop
    the router making.  Same geometry as `validate.validate_plan`'s frame gate,
    via `rig_final.chain_static_clearance`.
    """
    boxes = spec.static_obstacles() if hasattr(spec, "static_obstacles") else []
    if not boxes:
        return np.inf
    qs = np.asarray(qs, float).reshape(-1, 7)
    Twb = spec.T_world_base(h_inv)
    R, t = Twb[:3, :3], Twb[:3, 3]
    _, p = fk_many(qs)
    pw = p @ R.T + t
    tips = tip_pos_many(qs, pen_ext) @ R.T + t
    P10 = np.concatenate([pw, tips[:, None, :]], axis=1)
    return float(rig_final.chain_static_clearance(P10, boxes).min())


def path_frame_clearance(spec, qs, pen_ext=PEN_EXT, h_inv=H_INV_DEFAULT,
                         n=SAMPLES):
    """`frame_clearance` along every straight leg of a polyline of configs."""
    qs = [np.asarray(q, float).reshape(7) for q in qs]
    out = np.inf
    for a, b in zip(qs[:-1], qs[1:]):
        out = min(out, frame_clearance(line_samples(a, b, n), spec, pen_ext,
                                       h_inv))
    return float(out)


def tip_xy(q, spec, pen_ext=PEN_EXT, h_inv=H_INV_DEFAULT):
    """Where this configuration's pen tip is on the paper. -> (2,)."""
    Twb = spec.T_world_base(h_inv)
    return (Twb[:3, :3] @ tip_pos_many(np.asarray(q, float).reshape(1, 7),
                                       pen_ext)[0] + Twb[:3, 3])[:2]


# --------------------------------------------------------------------------
# routing
# --------------------------------------------------------------------------
TRAVERSE_STEPS = (0.30, 0.20, 0.12)   # m of paper per hop, coarsest first


def _traverse(spec, q_from, xy0, xy1, z, pen_ext, h_inv, mm, step, lift,
              ends=True):
    """Hovers walking the xy line from `xy0` to `xy1` at height `z`.

    Each is solved nearest to the one BEFORE it, so the walk stays on one
    analytic branch and the joint-space line between two neighbouring hovers is
    short.  That is the whole trick: a long reconfiguration dives because its
    endpoints are on different branches and the straight line between them is
    not a motion anybody chose, whereas a chain of 20-30 cm hops never leaves
    the hover plane it was solved on.

    `ends` decides who owns the last hover, and it is the difference between
    this working and not.  With `ends=False` the caller supplies both end
    hovers, and the one over the TARGET is solved nearest to the target — which
    puts it on the target's branch and hands the walk exactly the branch flip
    it was inserted to avoid.  With `ends=True` the walk solves its own last
    hover by continuing, and the single short leg from there to the target is
    the only place a branch has to reconcile.  On arm 2's go-home — 0.94 m back
    across the mirror plane — `ends=False` finds no route at any height and
    `ends=True` clears the paper by 20-60 mm.
    """
    hop = float(np.linalg.norm(np.asarray(xy1) - np.asarray(xy0)))
    K = max(2, int(np.ceil(hop / max(step, 1e-6))))
    out, ref = [], q_from
    for i in (range(0, K + 1) if ends else range(1, K)):
        xy = np.asarray(xy0) + (i / K) * (np.asarray(xy1) - np.asarray(xy0))
        v, _ = lift(spec, ref, xy, z, pen_ext, h_inv, mm)
        if v is None:
            return None
        out.append(v)
        ref = v
    return out


def route(spec, q0, q1, pen_ext=PEN_EXT, h_inv=H_INV_DEFAULT,
          tip_floor=TIP_CLEAR, chain_floor=CHAIN_CLEAR, heights=VIA_HEIGHTS,
          n=SAMPLES, margin_min=None, cache=True, q_home=None,
          steps=TRAVERSE_STEPS):
    """A pen-up route from q0 to q1 that clears the paper. -> dict | None.

    -> dict(vias, mode, chain_z, tip_z, tried) where `vias` is the (possibly
    empty) list of intermediate configurations such that EVERY consecutive
    straight joint-space move in [q0, *vias, q1] keeps the pen tip at least
    `tip_floor` and every chain point at least `chain_floor` above the paper.
    `None` means no shape on the ladder certified — the caller must refuse the
    move rather than fly it, which is the entire point, and `sequence`'s cost
    matrix turns that refusal into an infinite edge so the tour goes round it.

    The shapes tried, cheapest first, at each height of the ladder in turn:

        []            the direct move, when it was never a problem
        [a0]          lift higher over the exit, then straight down-range
        [a1]          fly to a high point over the entry, then descend
        [a0, a1]      lift, cross at height, descend  — retract-then-go
        [a0, m, a1]   the same with a hover over the midpoint of the hop
        [a0, ..., a1] a CARTESIAN TRAVERSE: hovers every `step` metres along
                      the xy line, each solved nearest to the last

    and, when `q_home` is given (the arm's ready pose), `[q_home]` and
    `[a0, q_home, a1]` — folding back to the pose the arm parks in is the one
    reconfiguration that is certified by construction.

    THE TRAVERSE IS THE ONE THAT EARNS ITS KEEP.  On the shipped timeline the
    two offending crossings have endpoints on different IK branches — arm 2
    swings joint 1 from +0.85 to −1.83 rad and joint 7 half a turn — and no
    amount of lifting the two ENDS fixes a straight line between branches.
    Walking the hover plane in 20 cm hops does, because every hop is short
    enough that the analytic solution nearest the previous one is the same
    branch continued.

    Every via is a `writing.lifted_config` IK solution at the arm's own tool
    orientation, subject to the same joint-limit margin every other hover
    keeps, so a via is a pose the arm may legitimately stand in and not an
    interpolation artefact.  The assembled route is then re-certified WHOLE by
    sampling, so nothing is assumed from the parts.
    """
    # writing imports this module, so the hover solver is fetched on use.
    from .writing import HOVER_MARGIN, lifted_config
    mm = HOVER_MARGIN if margin_min is None else float(margin_min)

    def lift(sp, ref, xy, z, pe, hi, margin):
        """`lifted_config`, memoised on (arm, reference pose, target, height).

        THE FAILURE PATH IS THE HOT ONE.  A pair that routes does so on the
        first or second shape; a pair that CANNOT route walks the whole ladder,
        and the fiber-menu matrices ask about tens of thousands of pairs whose
        endpoints are drawn from a few hundred distinct hover poses.  The lift
        over a given pose at a given height is the same solution every time it
        is asked for — `lifted_config` is deterministic — so solving it once
        per (pose, height) instead of once per PAIR is the difference between
        the cluster profile costing seconds and costing minutes.  Nothing about
        the answer changes; a test pins the memo against the direct call.
        """
        k = (id(sp), float(pe), float(hi), float(z), round(float(margin), 9),
             np.round(np.asarray(ref, float), 9).tobytes(),
             np.round(np.asarray(xy, float), 9).tobytes())
        if k not in _LIFTS:
            _LIFTS[k] = lifted_config(sp, ref, xy, z=z, h_inv=hi, pen_ext=pe,
                                      margin_min=margin)[0]
        return _LIFTS[k], None

    q0 = np.asarray(q0, float).reshape(7)
    q1 = np.asarray(q1, float).reshape(7)
    tip_floor, chain_floor = effective_floors(spec, q0, q1, pen_ext, h_inv,
                                              tip_floor, chain_floor)
    ck = _key(spec, q0, q1, pen_ext, h_inv, tip_floor, chain_floor) if cache \
        else None
    if ck is not None and ck in _CACHE:
        return _CACHE[ck]

    def legs_ok(seq, frame=True):
        qs = [q0] + list(seq) + [q1]
        cz, tz = path_clearance(spec, qs, pen_ext, h_inv, n)
        ok = cz >= chain_floor - EPS and tz >= tip_floor - EPS
        # The frame is only asked about a route we are INSERTING.  A direct move
        # that already clears the paper is left exactly as it was — routing is
        # not the place to start refusing transits the rest of the pipeline has
        # always flown, and `scene_check` still has the last word on the frame.
        if ok and frame and seq:
            ok = path_frame_clearance(spec, qs, pen_ext, h_inv, n) \
                >= rig_final.STATIC_MARGIN - EPS
        return ok, cz, tz

    def done(seq, name, cz, tz, tried):
        out = dict(vias=[np.asarray(v, float).reshape(7) for v in seq],
                   mode=name, chain_z=float(cz), tip_z=float(tz), tried=tried)
        if ck is not None:
            _CACHE[ck] = out
        return out

    ok, cz, tz = legs_ok([], frame=False)
    if ok:
        return done([], "direct", cz, tz, 1)

    xy0 = tip_xy(q0, spec, pen_ext, h_inv)
    xy1 = tip_xy(q1, spec, pen_ext, h_inv)
    xym = 0.5 * (xy0 + xy1)
    tried = 1
    if q_home is not None:
        qh = np.asarray(q_home, float).reshape(7)
        tried += 1
        ok, cz, tz = legs_ok([qh])
        if ok:
            return done([qh], "home", cz, tz, tried)
    for z in heights:
        a0, _ = lift(spec, q0, xy0, z, pen_ext, h_inv, mm)
        a1, _ = lift(spec, q1, xy1, z, pen_ext, h_inv, mm)
        shapes = []
        if a0 is not None:
            shapes.append(("lift_exit", [a0]))
        if a1 is not None:
            shapes.append(("lift_entry", [a1]))
        if a0 is not None and a1 is not None:
            shapes.append(("retract_go", [a0, a1]))
            m, _ = lift(spec, a0, xym, z, pen_ext, h_inv, mm)
            if m is not None:
                shapes.append(("arc", [a0, m, a1]))
            for st in steps:
                mid = _traverse(spec, a0, xy0, xy1, z, pen_ext, h_inv, mm, st,
                                lift, ends=False)
                if mid:
                    shapes.append((f"traverse{100 * st:.0f}", [a0] + mid + [a1]))
            if q_home is not None:
                shapes.append(("lift_home", [a0, np.asarray(q_home, float)
                                             .reshape(7), a1]))
        # the WALK: its own hovers at both ends, the target reached in one
        # short leg.  Tried after the cheap shapes and before giving up on this
        # height, because it is the shape that survives a branch change.
        for st in steps:
            walk = _traverse(spec, q0, xy0, xy1, z, pen_ext, h_inv, mm, st,
                             lift, ends=True)
            if walk:
                shapes.append((f"walk{100 * st:.0f}", walk))
        for name, seq in shapes:
            tried += 1
            ok, cz, tz = legs_ok(seq)
            if ok:
                return done(seq, f"{name}@{100 * z:.0f}cm", cz, tz, tried)
    if ck is not None:
        _CACHE[ck] = None
    return None


def travel_floor(z_exit, z_entry, tip_clear=TIP_CLEAR):
    """The tip floor a hover-to-hover move must keep. -> metres.

    Never more than the lower of the two hovers the arm actually reached:
    `writing.lifted_or_lower` gives up 60 mm for 45, 30, or none at all near
    the edge of an arm's reach, and demanding 20 mm of a move whose own
    endpoint is at 0 would refuse a transit for being what it was asked to be.
    """
    return float(min(float(z_exit), float(z_entry), float(tip_clear)))
