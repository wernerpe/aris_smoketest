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

from . import envelope
from . import frozen
from . import rig_final
from . import frames as _frames
from .fleet import H_INV_DEFAULT
from .frames import PEN_EXT, ext_of, fk_many, tip_pos_many, tool_points_many
from .validate import Z_CLEAR

CHAIN_CLEAR = Z_CLEAR      # m, every chain point above the paper (= 0.02)
TIP_CLEAR = 0.02           # m, the pen tip while flying
TIP_TOL = 0.010            # m, the contact band (lift/lower/ink)
EPS = 1e-9

# THE ROUTER HAS TO LEAVE THE CHECKER ROOM TO CHECK.  `move_ok` compares
# SAMPLES of a straight move against a floor; `scene_check`'s paper gate
# compares the same quantity MINUS a 1-Lipschitz sweep residual (up to
# 0.55 * scene_check.PAPER_STEP = 2.75 mm) against the same number.  So a
# lower beat that sits exactly ON the contact band is certified by the router
# and refused by the checker, and neither of them is wrong.
#
# That is not hypothetical: it is the whole of the first occupancy-aware CSAIL
# run's refusal on the proposed rig.  Arm 31's entry read tip -10.3 mm against
# a -10 mm floor — a 0.3 mm veto of a 7.5 mm dip, with every other gate
# passing by tens of millimetres.
#
# The band a ROUTE may use is therefore the band minus that residual.  The
# direction is the safe one: this can only refuse routes, never certify one
# the checker would fail, and 7 mm of contact is still 3.5x the 2 mm the ink
# itself is held to (`validate.TIP_TOL`).
TIP_SWEEP_PAD = 0.003      # m, scene_check's worst refined sweep residual
CONTACT_FLOOR = -(TIP_TOL - TIP_SWEEP_PAD)   # -0.007 m, lift/lower tip floor
# ...and the same argument, one obstacle over: `scene_check`'s FRAME gate
# subtracts the same kind of residual from the same STATIC_MARGIN this module
# compares its samples against (`FRAME_STEP` there).  A route that clears the
# steel by exactly the margin is a route the checker refuses — arm 31 read
# 48.8 mm against 50 mm on the run that found this — so a route must clear it
# by the margin plus the residual.
#
# ...AND THAT PAD WAS THE WRONG ONE, measured 2026-08-26.  `TIP_SWEEP_PAD`
# covers the checker's trajectory residual and nothing else, and the checker's
# STATIC bound carries a second, larger piece of slack: it samples each capsule
# every 2 cm instead of minimising along it, and subtracts half a step.  A
# route that cleared 53 mm exactly read 47.4 mm there.  The floor is now
# `rig_final.STATIC_PLAN_MARGIN` — the same 63 mm the stroke planner keeps, so
# the ink and the pen-up over it are held to one number.
FRAME_FLOOR = rig_final.STATIC_PLAN_MARGIN               # 0.063 m
SKIRT_STEP = 0.30          # m of paper per hop when a route SKIRTS an obstacle

# ...AND A PAD IS NOT A BOUND.  `TIP_SWEEP_PAD` covers the residual the CHECKER
# loses between two of ITS samples.  It says nothing about the residual this
# module loses between two of its own, and this module samples a whole pen-up
# leg — a metre of tip travel and more of elbow — at a fixed 33 points.  33
# points over a metre is 30 mm of chain motion per step, and a 1-Lipschitz
# quantity can be 16 mm lower between two samples than at either of them.  So
# the router could read 53 mm at its samples on a leg whose true minimum was
# 37, certify it, and hand `scene_check` — which refines until its own residual
# is under 2.75 mm — a timeline to refuse.
#
# THAT IS NOT A HYPOTHESIS.  It is `out/run_solo_p0.log`: arm 2's conducted solo
# timeline, every other gate passing by 60 mm, refused at "min frame clearance
# 41.0 mm (margin 50 mm)".  The router's gate has to be STRICTLY TIGHTER than
# the checker's on every path or the checker is not a second opinion, it is a
# lottery — and a 9 mm disagreement about the same metal is the loosest a gate
# in this repo has ever been.
#
# So the static measurement refines the way the checker refines, to the same
# per-point step, and subtracts the same 1-Lipschitz coefficient — and then
# asks for `FRAME_FLOOR`, which is 3 mm MORE than the checker asks for.  Both
# sides now compute a lower bound on the same quantity, at the same density,
# and the router's is held to the higher number.  That ordering is the whole
# property.
STATIC_STEP = 0.005        # m of per-point motion the static gate refines to
#                            (= scene_check.FRAME_STEP, deliberately)
SWEEP_K = 0.55             # ...and its 1-Lipschitz coefficient, likewise
REFINE_CAP = 32            # ...and its cap on the refinement factor

# THE SAME HOLE, ONE OBSTACLE OVER.  This module made the paper a thing a pen-up
# is certified against, and left the STEEL exactly where it found it: `route`
# asks `path_frame_clearance` only about a detour it is INSERTING, never about
# the direct move (see `legs_ok`).  So a transit that clears the paper and
# grazes a frame box is priced at zero seconds by `sequence.cost_matrix`, flown
# by `writing.arm_program`, and discovered only by `scene_check` at the end —
# which is the defect `docs/PAPER_PLANE.md` §1.1 describes, with "the paper"
# replaced by "the frame".
#
# It is not a rare corner.  Every two-pass configuration tried on `final6_opt`
# — five placements x three execution profiles, split and unsplit — was refused
# by `scene_check` on ONE arm's pen-up against the frame, at 34.1, 37.4, 45.2,
# 49.1 and 49.4 mm against a 50 mm margin.  Eight refusals, and not one of them
# is a transit anybody priced.
#
# `STATIC_SAFE` closes it: the direct move is held to the same static floor,
# so a steel-grazing pen-up is routed around the metal exactly as a
# paper-breaking one is routed around the table, and refused honestly when no
# shape on the ladder clears both.
#
# IT IS ON NOW, AND TWO THINGS HAD TO CHANGE FIRST (2026-08-26).  It shipped
# OFF as `FRAME_SAFE` because turning it on refused everything: the CSAIL logo
# would not route a single pen-up on the proposed rig.  Both reasons were bugs
# in this module, not facts about the rig.
#
#   1. THE FLOOR WAS A CONTRADICTION AT THE ENDPOINTS.  `FRAME_FLOOR` is
#      53 mm and `atlas.solve_cell` certifies a drawing pose at exactly
#      `STATIC_MARGIN` = 50 mm, so 4-6 of every 300 certified cells stand
#      between 50 and 53 mm from a neighbour's column.  A lift OUT of such a
#      pose was asked to keep 53 mm at its own first sample and no shape on
#      any ladder could — the identical failure `effective_floors` was written
#      for, one obstacle over.  `effective_static_floor` clamps it the same
#      way: never ask a move to clear the metal by more than its own endpoints
#      already do.
#   2. THE LADDER ONLY WENT UP.  Every escape this module offered was a higher
#      hover, and on a rig with a ceiling grid up is where the steel is.  The
#      obstacle that actually blocks a pen-up here is a neighbour's BASE
#      COLUMN — a vertical body 0.6 m long standing between two arms 0.61 m
#      apart — and the way past a column is around it.  `_skirt` reads the
#      blocking boxes' own footprints and walks the hover plane around them.
#
# With those two, the static gate converges: it is the honest question and it
# now has honest answers.  `--no-static-safe` restores the old behaviour for a
# reproduction of a pre-2026-08-26 number.
STATIC_SAFE = True

# Extra hover heights `route` climbs to when a direct move is refused.  Low
# first: a via costs joint-space seconds and the sequencer pays them, so the
# cheapest certified escape wins.  The top of the ladder is 40 cm, well above
# anything the logo needs, so "no route" means the geometry and not the ladder.
VIA_HEIGHTS = (0.08, 0.12, 0.15, 0.18, 0.25, 0.32, 0.40)

SAMPLES = 33               # configurations sampled along one straight move

_CACHE = {}                # (spec key, pen, h_inv, q0, q1, floors) -> result
_LIFTS = {}                # (spec key, pen, h_inv, q, z) -> hover pose | None
_LEGS = {}                 # (spec key, pen, tool, h_inv, q0, q1, floor)
#                            -> (chain z, tip z, static lower bound)
_SELF = {}                 # (pen, tool, self floor, n, q0, q1) -> self LB
#   THE LEGS ARE SHARED, NOT THE ROUTES.  Every shape on the ladder is built
#   out of the same handful of hover poses, and so is every OTHER crossing
#   between the same two spans: `[a0]`, `[a0, a1]` and `[a0, m, a1]` all begin
#   with q0 -> a0, and the return crossing begins with the same a1 -> q1 the
#   outward one ended with.  Measuring a leg is a forward-kinematics pass over
#   33 configurations against the static set, so measuring each distinct one
#   once instead of once per shape is most of what makes the static gate
#   affordable at all.


_ON_CLEAR = []             # memos elsewhere that are derived from these two


def on_clear(fn):
    """Register a memo to be dropped whenever this module's are.

    `sequence` caches the PRICED detour of a crossing, which is this module's
    answer plus arithmetic; a fleet mutated in place invalidates both, and a
    module that had to remember to clear the other one would eventually forget.
    """
    _ON_CLEAR.append(fn)


def clear_cache():
    """Drop the memos.  Tests that mutate a fleet in place need this."""
    _CACHE.clear()
    _LIFTS.clear()
    _LEGS.clear()
    _SELF.clear()
    for fn in _ON_CLEAR:
        fn()


def route_key(spec, q0, q1, pen_ext=None, h_inv=H_INV_DEFAULT,
              tip_floor=TIP_CLEAR, chain_floor=CHAIN_CLEAR):
    """The memo key `route` would file this question under. -> hashable.

    Public because the crossings of one cost matrix are screened in PARALLEL
    (`sequence._paper_surcharge`) and a worker's memo dies with the worker: the
    parent files each answer it gets back under this key, so the routes bought
    by one matrix are still free to the next one and to the timeline that
    finally flies them.  `route` clamps the floors to what the endpoints can
    hold before it keys on them, and so does this — the two must agree or the
    parent would file the answer where nothing looks for it.
    """
    tf, cf = effective_floors(spec, q0, q1, pen_ext, h_inv, tip_floor,
                              chain_floor)
    # the depot flag defaults to True for the same reason `key_maker` bakes it:
    # every caller of the public form is the sequencer's parallel screen, and
    # `sequence._screen_task` routes with `spec.q_seed`
    return _key(spec, q0, q1, pen_ext, h_inv, tf, cf)


NOT_CACHED = object()      # `None` is a legitimate answer ("no route exists")


def cache_route(key, out):
    """File a route computed elsewhere under `route_key`'s key."""
    _CACHE[key] = out


def cached_route(key):
    """The memoised answer for that key, or `NOT_CACHED`."""
    return _CACHE.get(key, NOT_CACHED)


def _pose_bytes(q):
    return np.round(np.asarray(q, float), 9).tobytes()


def _rrt_key(rrt=True):
    """The RRT tier's contribution to a memo key. -> hashable.

    `False` whenever the tier could not have been used, so that a run with
    `--no-rrt` and a `fold_home` sub-route (which never offers the tier) share
    the pre-planner memo instead of splitting it.
    """
    if not (RRT_SAFE and rrt):
        return False
    return (True, RRT_PROBE[2] if RRT_PROBE is not None else None)


def _key(spec, q0, q1, pen_ext, h_inv, tip_floor, chain_floor, q_home=True,
         rrt=True):
    # STATIC_SAFE is part of the question, so it is part of the key: the same
    # pair has a different answer with the steel gated in, and a memo that
    # forgot that would hand a run the other run's route.  So is the ACTIVE
    # TOOL (frames.PEN_LAT): the lateral holder sweeps a different envelope.
    # ...and SO IS WHETHER THE DEPOT WAS OFFERED.  `q_home` adds three shapes
    # to the ladder, the last of them the routed fold that is the difference
    # between an arm reaching a span and not, so the same pair genuinely has
    # two answers.  Caching them under one key means whichever caller asked
    # FIRST decides what every later one is told — and the dangerous order is
    # the common one, a q_home-less call filing `None` that a q_home call then
    # reads instead of trying the fold.  A bool is enough: `q_home` is always
    # the arm's own `q_seed` and `id(spec)` already says which arm that is.
    # ...AND SO IS THE SELF GATE, for the third time and the same argument.
    # ...AND SO IS THE C-SPACE TIER, for the fourth time and the same argument.
    # A pair the ladder cannot route has two answers now — `None` without the
    # planner, a path with it — and a run that turned it off must not read the
    # run that left it on.  The probe, when there is one, rides along: it is a
    # fourth obstacle and a route certified without it is not the same route.
    return (id(spec), ext_of(pen_ext), float(_frames.PEN_LAT), float(h_inv),
            _pose_bytes(q0), _pose_bytes(q1),
            round(float(tip_floor), 9), round(float(chain_floor), 9),
            bool(STATIC_SAFE), bool(q_home), bool(SELF_SAFE), _rrt_key(rrt))


def key_maker(spec, q_rows, q_cols, pen_ext=None, h_inv=H_INV_DEFAULT):
    """`_key` for a whole BLOCK of pose pairs, with each pose hashed once.

    A cost matrix asks about tens of thousands of pairs drawn from a few
    hundred poses, and rounding-and-hashing a pose is not free; the block form
    does it once per pose instead of once per cell.  It builds the same tuple
    `_key` does, out of the same helper, so the two cannot drift apart.
    """
    rb = [_pose_bytes(q) for q in q_rows]
    cb = [_pose_bytes(q) for q in q_cols]
    base = (id(spec), ext_of(pen_ext), float(_frames.PEN_LAT), float(h_inv))

    def key(a, b, tip_floor, chain_floor):
        # `True` for the depot, because every caller of the BLOCK form is the
        # sequencer's crossing screen and `sequence._screen_task` routes with
        # `spec.q_seed` — see `_key`, whose tuple this has to reproduce exactly.
        return base + (rb[a], cb[b], round(float(tip_floor), 9),
                       round(float(chain_floor), 9), bool(STATIC_SAFE), True,
                       bool(SELF_SAFE), _rrt_key(True))
    return key


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------
def chain_tip_z(qs, spec, pen_ext=None, h_inv=H_INV_DEFAULT):
    """World z of the lowest chain point and of the pen tip. -> (M,), (M,).

    The chain minimum skips point 0 (the base origin, bolted to its mount) for
    the same reason `validate.validate_plan` does; the pen tip is returned
    SEPARATELY and never folded into that minimum, because the whole failure
    mode this module exists for is a tip below a chain that is above.
    """
    qs = np.asarray(qs, float).reshape(-1, 7)
    Twb = spec.T_world_base(h_inv)
    R, t = Twb[:3, :3], Twb[:3, 3]
    T, p = fk_many(qs)                              # (M,9,3), base frame
    chain_z = (p @ R.T + t)[:, 1:, 2].min(axis=1)
    tool = tool_points_many(T, pen_ext)             # [tip(, bracket corner)]
    tip_z = (tool[0] @ R.T + t)[:, 2]
    if len(tool) > 1:
        # the lateral holder's bracket corner is a rigid body of the CHAIN —
        # it can dip where no FK point does, so it joins the chain minimum
        chain_z = np.minimum(chain_z, (tool[1] @ R.T + t)[:, 2])
    return chain_z, tip_z


def world_chain(qs, spec, pen_ext=None, h_inv=H_INV_DEFAULT):
    """(M,7) -> (M,K,3) world chain points, tool points included.

    The 10- or 11-point chain every gate in this package measures: `frames.fk`'s
    nine, the pen tip, and (lateral holder) the bracket corner.
    """
    qs = np.asarray(qs, float).reshape(-1, 7)
    Twb = spec.T_world_base(h_inv)
    R, t = Twb[:3, :3], Twb[:3, 3]
    T, p = fk_many(qs)
    pw = p @ R.T + t
    tool = [tp @ R.T + t for tp in tool_points_many(T, pen_ext)]
    return np.concatenate([pw] + [tp[:, None, :] for tp in tool], axis=1)


def sample_residual(P):
    """The 1-Lipschitz residual between consecutive samples of a path. -> m.

    `scene_check`'s own expression, restated: the worst distance any single
    chain point travels between two samples, times the coefficient the checker
    charges.  A clearance measured AT the samples can be this much lower
    between them, and no further.
    """
    if len(P) < 2:
        return 0.0
    return SWEEP_K * float(np.max(np.linalg.norm(np.diff(P, axis=0), axis=2)))


def refine_n(P, n, step=STATIC_STEP, cap=REFINE_CAP):
    """How many samples this leg needs for a residual under `step`. -> int."""
    if len(P) < 2:
        return n
    mv = float(np.max(np.linalg.norm(np.diff(P, axis=0), axis=2)))
    return (n - 1) * int(np.clip(np.ceil(mv / max(step, 1e-9)), 1, cap)) + 1


def sphere_centres(qs, spec, h_inv=H_INV_DEFAULT):
    """The sphere centres a static gate needs, or None. (N,7) -> (N,S,3)|None.

    One place, because every static funnel in this module builds its chain
    from a `Q` and then hands it to `frozen.chain_clearance`, and under the
    sphere model that call needs the centres for the SAME `Q`.  `None` when
    the flag is off, which is what makes every gate below reduce exactly to
    the number it shipped with.
    """
    from . import link_spheres
    if not link_spheres.enabled():
        return None
    return link_spheres.centres_world(np.asarray(qs, float).reshape(-1, 7),
                                      spec.T_world_base(h_inv))


NEAR_SLACK = 0.35          # m of clearance the broad phase stays exact to


def near_boxes(P, boxes, slack=NEAR_SLACK, C=None):
    """The boxes these configurations could come within `slack` of.

    A BROAD PHASE, because the static set is thirty boxes on this rig and a
    pen-up leg is within reach of half a dozen of them, and the narrow phase is
    a 36-step ternary search per capsule per box.

    THE BOUND IS PER CAPSULE AND EXACT.  A capsule's segment runs between two
    chain points, so every point on it is within `|P_i - P_j|` of the nearer
    end: if BOTH ends are farther from a box than that plus the capsule radius
    plus `slack`, no point of that capsule is within `slack` of it, in any of
    these configurations.  A box no capsule can reach is dropped, and the
    minimum the narrow phase then reports is unchanged wherever it is under
    `slack` — which is every floor this module compares against, seven times
    over.  Above `slack` the answer may come back `inf` instead of a large
    number, and nothing asks.
    """
    if not boxes or not len(P):
        return boxes
    caps = (rig_final.STATIC_CAPSULES_LAT if P.shape[1] >= 11
            else rig_final.STATIC_CAPSULES)
    lo = np.stack([np.asarray(b["lo"], float) for b in boxes])
    hi = np.stack([np.asarray(b["hi"], float) for b in boxes])
    d = rig_final._point_box_d(P[:, :, None, :], lo, hi)      # (M,K,B)
    keep = np.zeros(len(boxes), bool)
    if C is not None:
        # THE SPHERES GET THEIR OWN SCREEN, and they need one: the sphere set
        # is NOT a subset of the sausages it replaces (a fitted sphere reaches
        # a little past a capsule's hemispherical cap at the flange and the
        # finger tips), so a box kept only because some capsule could reach it
        # is not proof that no sphere can.  A sphere is a point, so its screen
        # is the point-to-box distance and nothing else.
        from . import link_spheres
        C = np.asarray(C, float)
        dc = rig_final._point_box_d(C[:, :, None, :], lo, hi)  # (M,S,B)
        keep |= (dc - link_spheres.RADII[None, :, None]
                 <= slack).any(axis=(0, 1))
        caps, _ = link_spheres.static_capsules(caps)
    for i, j, r in caps:
        L = np.linalg.norm(P[:, i] - P[:, j], axis=1)         # (M,)
        m = np.minimum(d[:, i, :], d[:, j, :]) - L[:, None]
        keep |= (m <= r + slack).any(axis=0)
        if keep.all():
            break
    return [b for b, k in zip(boxes, keep) if k]


def leg_static_lb(spec, q0, q1, pen_ext=None, h_inv=H_INV_DEFAULT,
                  boxes=None, n=SAMPLES, floor=None):
    """A LOWER BOUND on the static clearance over the whole straight move.

    -> metres.  Not "the clearance at 33 sampled configurations" — the
    clearance anywhere along the move, refined and residual-corrected the way
    `scene_check` does it (see STATIC_STEP).  This is the quantity the router
    gates on, and it is a bound the checker cannot beat by finding a dip
    between two of the router's samples.

    `floor` IS WHAT MAKES IT AFFORDABLE, and it costs nothing in rigour.  A
    refinement to 32x is a thousand configurations against thirty boxes, and
    almost every leg is decided without it: if the COARSE bound already clears
    the floor the leg is certified (refining can only raise a lower bound
    towards the truth, never lower the verdict), and if the coarse MINIMUM is
    already under the floor the leg is refused (refining can only lower it).
    Only a leg in the band between those two — sampled clear, bounded unclear —
    has to be looked at closely.  Given a floor the returned number is only
    guaranteed to be on the right side of it, which is all any caller asks.
    """
    if boxes is None:
        boxes = static_boxes(spec)
    if not boxes:
        return np.inf
    Q = line_samples(q0, q1, n)
    P = world_chain(Q, spec, pen_ext, h_inv)
    C = sphere_centres(Q, spec, h_inv)
    boxes = near_boxes(P, boxes, C=C)
    if not boxes:
        return np.inf
    m = float(frozen.chain_clearance(P, boxes, C).min())
    res = sample_residual(P)
    if floor is not None:
        if m - res >= float(floor) - EPS:
            return m - res                       # certified without refining
        if m < float(floor) - EPS:
            return m                             # refused without refining
    # ...and the band between those two is where the whole-leg residual used to
    # decide the answer by itself.  `adaptive_static_lb` subdivides only the
    # intervals still under the floor instead (see ADAPT_TOL).
    return adaptive_static_lb(spec, q0, q1, pen_ext, h_inv, boxes, n, floor)


# ==========================================================================
# ...AND THE THIRD OBSTACLE, WHICH IS THE ARM ITSELF
# ==========================================================================
# This module's whole subject is a motion nobody planned: a pen-up is a
# STRAIGHT LINE IN JOINT SPACE between two poses that were each certified on
# their own, and `writing.py` says outright that its interpolation "makes no
# collision or self-collision guarantee".  Two of the three things such a line
# can hit are gated above — the paper it flies over and the neighbours' steel.
# The third is the arm's own metal, and until now the only thing that looked at
# it was `writing.static_gate`, which gates the POSES `route` chooses as vias
# and says nothing about the line between two of them.
#
# THE GAP IS REAL AND IT IS LARGE.  Sampled along straight joint-space moves
# between certified drawing cells of arm 31 — poses the self guard passes at
# 63.7 mm or better, at both ends — `selfcoll` reads -194.7 mm at the worst
# sample: the line folds the wrist through the shoulder on its way from one
# legal pose to another.  Those particular pairs are not transits anybody flew;
# what they establish is that the gate is not inert on a joint-space line, which
# is the only claim needed to put it in the stack.
#
# THE FLOOR IS THE PRODUCER'S, `selfcoll.SELF_PLAN_MARGIN` (23 mm against the
# checker's 20), and the residual is charged the way `scene_check` charges its
# own — `SWEEP_K` times the worst distance any CAPSULE ENDPOINT travels between
# two samples, refined to `STATIC_STEP` exactly as the static bound is.  So the
# router's bound is computed at the checker's density and held to a higher
# number, which is the ordering the static gate already lives under.
#
# AND IT IS AFFORDABLE BECAUSE OF THE SPHERE SCREEN (`selfcoll.sphere_bounds`).
# The exact 165-pair segment arithmetic is 95 us a configuration and would have
# doubled the clock of every route on the ladder; screened, it is 10 us, because
# the arm is almost never near itself and a bounding ball settles 99.9 % of the
# pairs without any segment math.
#
# ...AND IT IS NOT OPTIONAL, WHICH IS THE THING THE MEASUREMENT SETTLED.  The
# first instinct was that it could ship OFF the way `FRAME_SAFE` did — right,
# unaffordable, written down — because `scene_check` sweeps self-clearance on
# every conducted frame and would refuse anything that folded.  Run end to end
# on the CSAIL logo at the v8 placement, everything else identical, that is
# exactly backwards:
#
#     SELF_SAFE off   ALLOCATES 100.0000 % ... and scene_check REFUSES phase 1
#                     at -177.8 mm on arms 13, 71 and 97.  Nothing renders.
#     SELF_SAFE on    ALLOCATES  94.6994 % and every conducted phase reads
#                     21.0 to 60.5 mm against the checker's 20 mm margin.
#
# The checker catching it does not make the coverage real; it makes the coverage
# a phase that is thrown away three stages later, which is the inf-pricing
# lesson this module already learned twice (the paper, then the metal).  A gate
# the ROUTER does not know about is a gate the allocator spends its whole budget
# walking into.
#
# WHAT IT COSTS AND WHY, STATED RATHER THAN HIDDEN.  5.30 points of allocated
# logo, and it is not the gate being wrong — every crossing it refuses is a
# pen-up whose straight joint-space line puts the arm inside itself.  It is the
# ROUTER having nothing to offer such a crossing, and that is measured too: the
# shape ladder is exhausted, not merely unlucky.  Adding 8 cm and 5 cm rungs to
# `TRAVERSE_STEPS` — the obvious fix, since a fold is what a LONG interpolation
# commits — recovers 2 of 520 crossings for 1.9x the clock, because a fold is
# not a long hop on the hover plane, it is a change of IK branch that no walk
# through hover poses avoids.  What buys those crossings back is a pen-up
# planner that searches configuration space, which this package does not have
# and which is a project rather than a flag.
SELF_SAFE = True           # certify pen-up LEGS against the arm's own metal

# ==========================================================================
# ...AND THE PEN-UP PLANNER THAT PARAGRAPH ASKS FOR
# ==========================================================================
# "What buys those crossings back is a pen-up planner that searches
# configuration space, which this package does not have and which is a project
# rather than a flag."  It has one now: `aris_sixarm.transit`, a bidirectional
# RRT-Connect in the seven-dimensional joint space whose every edge is
# certified by the bounds this module already computes.
#
# IT IS THE LAST TIER AND IT HAS TO BE.  Measured on the CSAIL logo at the v8
# placement the shape ladder settles 480 of 520 crossings, most of them on the
# first or second rung, for a few milliseconds each; the planner costs
# hundreds of milliseconds to seconds and it is worth every one of them only
# on the crossings where the ladder is exhausted.  So it runs after
# `fold_home`, on the way to returning `None`, and it turns some of those
# `None`s into routes.
#
# THE CERTIFICATION IS NOT NEW AND THAT IS THE WHOLE DESIGN.  `transit` hands
# back a polyline of configurations and `legs_ok` — the same function that
# grades every shape on the ladder — certifies it, leg by leg, against the
# same three obstacles at the same floors.  A path the planner finds and
# `legs_ok` refuses is refused; nothing about this tier can put a motion into
# a timeline that the ladder's own certifier would not have taken.  What is
# new is only where the candidate shapes come from.
#
# `--no-rrt` reproduces a pre-2026-08-26 number exactly, which is why the flag
# is part of `_key`: a run with the tier off must not be handed a route that a
# run with it on paid for.
RRT_SAFE = True            # offer the C-space planner when the ladder is out

# THE FOURTH OBSTACLE, WHICH ONLY SOME CALLERS HAVE.  The ladder knows about
# the paper, the neighbours' steel and the arm's own metal; it does not know
# about the neighbours' PARKED CHAINS, because `allocate.ParkProbe` lives a
# layer up and screens a route after the fact (`scripts/feasible_workspace.py`
# is where the pipeline puts one).  A caller that has a probe may hand it to
# the planner here, and then a route the probe would have refused three stages
# later is a route the search never proposes.
#
#   RRT_PROBE = (fn(qs, sweep) -> clearance, margin, key)
#
# `key` is a hashable that goes into the memo, for the reason `q_home` is in
# there: a memo that forgot which obstacles were in the room when it filed an
# answer would hand the next caller the other caller's route.  The LADDER is
# deliberately NOT probed — its answers are pinned numbers and this must not
# move them; the probe only ever makes the new tier stricter.
RRT_PROBE = None


def self_floor(spec, q0, q1, pen_ext=None):
    """The self-clearance floor a move can actually be held to. -> metres.

    `effective_static_floor`'s argument for the third obstacle, and it is here
    for the same reason: a pose certified at exactly the producer's margin
    cannot be asked for more at its own first sample.  Inert on this rig today
    (the tightest certified pose in the shipped map holds 63.7 mm), and the
    clamp is what keeps it inert rather than contradictory if that ever stops
    being true.
    """
    if not SELF_SAFE:
        return -np.inf
    from . import selfcoll
    ends = np.stack([np.asarray(q0, float).reshape(7),
                     np.asarray(q1, float).reshape(7)])
    A, B, R = selfcoll.capsule_ends(ends, pen_ext)
    # the screened form, because the answer is a `min` against the margin and
    # the screen is exact on everything below it — an endpoint that clears the
    # margin only has to be KNOWN to, not measured
    lo = selfcoll.min_clearance(A, B, R, selfcoll.SELF_PLAN_MARGIN)
    return float(min(selfcoll.SELF_PLAN_MARGIN, lo))


# ==========================================================================
# THE CERTIFICATE WAS REFUSING, NOT THE GEOMETRY (2026-09-09)
# ==========================================================================
# Both bounds above price a straight move by sampling it and charging a
# 1-Lipschitz residual for what happens in between, and both charged ONE
# residual for the WHOLE leg: `min over every sample` minus `SWEEP_K` times
# `the worst point-motion over every interval`.  On a short hop that is
# nothing.  On a branch change it is the whole answer, because the residual
# bottoms out at `SWEEP_K * STATIC_STEP` = 2.75 mm and no further refinement
# was on offer at any price.
#
# MEASURED, and it is why this got written: arm 31, cell (0.52, 1.48) at
# h = 0.970, the descent from every one of 53 certified hovers refused on a
# 63.0 mm static floor by a bound of 61.1 mm — while the leg's true minimum
# over 2001 dense samples is 63.7 mm and not one sampled configuration on it
# touches anything.  Pete looked at the scene and said what the numbers say:
# "there is more than enough space to reach there".  The arm was not the
# problem and neither was the floor.  The certificate was.
#
# THE FIX IS TO SPEND SAMPLES WHERE THE BOUND BINDS.  Every interval carries
# its own residual, so every interval gets its own bound —
# `min(c[i], c[i+1]) - SWEEP_K * motion_i` — which is the identical Lipschitz
# argument stated locally and is never looser than the global form (the global
# form is this one with the two terms taken worst-case INDEPENDENTLY, a
# minimum from one end of the leg against a residual from the other).  Then
# only the intervals whose own bound is still under the floor are bisected,
# and their residual halves every round.  A leg that was never in doubt costs
# what it always cost; a leg like that descent converges on the four or five
# intervals that actually decide it.
#
# It is a TIGHTENING OF A LOWER BOUND, so it can only ever turn refusals into
# certificates and never the reverse — `scene_check`, which samples the real
# timeline and clamps nothing, still has the last word.

ADAPT_TOL = 0.0005         # m: how close to the sampled minimum the bound is
#                            driven before it stops buying samples.  Half a
#                            millimetre is an order under the 13 mm the
#                            producers already carry and under every gate step
#                            on this rig, so no verdict can turn on it.
ADAPT_CAP = 4097           # most configurations one adaptive leg may sample
ADAPT_SPLIT = 512          # most intervals bisected in one round
SELF_SCREEN_PAD = 0.05     # m of headroom the self screen stays EXACT over,
#                            so an interval can be certified against the floor
#                            rather than against the screen's own bound


def interval_bounds(c, X, k=SWEEP_K):
    """`sample_residual`'s argument, made once per interval. -> (lb, res).

    `c` is (N,) clearance at the samples of one straight move and `X` is
    (N, P, 3) the points whose travel bounds how fast it can change — the chain
    for the static gate, the capsule ends for the self gate.  Both returns are
    (N-1,), one per interval.

    Between samples i and i+1 no watched point moves further than `res[i]`, so
    the clearance inside that interval cannot fall below the lower of its two
    ends by more than that.  The whole-leg residual is this quantity maximised
    over every interval and then subtracted from every interval's minimum,
    which is the same statement made worst-case twice.
    """
    c = np.asarray(c, float)
    if len(c) < 2:
        return c, np.zeros(max(len(c) - 1, 0))
    res = float(k) * np.linalg.norm(np.diff(np.asarray(X, float), axis=0),
                                    axis=2).max(axis=1)
    return np.minimum(c[:-1], c[1:]) - res, res


def adaptive_lb(sample, floor=None, n=SAMPLES, tol=ADAPT_TOL, cap=ADAPT_CAP,
                k=SWEEP_K):
    """A CONVERGED 1-Lipschitz lower bound along one straight move.

    -> (lb, sampled_min, n_used).  `sample(ts)` takes a (M,) array of
    parameters in [0, 1] and returns `(c (M,), X (M, P, 3))`: the clearance at
    those points of the move, and the points whose travel bounds it.

    Stops the moment a finer grid cannot change a verdict — the bound clears
    `floor`; or the SAMPLED minimum is already under it, which no refinement
    can rescue because the truth is at most the sampled minimum; or the bound
    has closed to within `tol` of the sampled minimum and there is nothing left
    to win.  Without a `floor` it simply converges to `tol`.
    """
    ts = np.linspace(0.0, 1.0, int(n))
    c, X = sample(ts)
    c = np.asarray(c, float)
    X = np.asarray(X, float)
    while True:
        lb, res = interval_bounds(c, X, k)
        m = float(c.min())
        best = float(lb.min()) if len(lb) else m
        if floor is not None and m < float(floor) - EPS:
            return m, m, len(ts)               # no density rescues a collision
        if floor is not None and best >= float(floor) - EPS:
            return best, m, len(ts)            # certified
        if m - best <= tol or len(ts) >= cap:
            return best, m, len(ts)
        # only the intervals that could still be the binding one, and only
        # while halving their residual can still buy anything
        thr = float(floor) if floor is not None else m - tol
        split = np.where((lb < thr) & (res > 0.5 * tol))[0]
        if not len(split):
            return best, m, len(ts)
        room = int(cap) - len(ts)
        if room <= 0:
            return best, m, len(ts)
        budget = min(ADAPT_SPLIT, room)
        if len(split) > budget:                # tightest first
            split = split[np.argsort(lb[split])[:budget]]
        mid = 0.5 * (ts[split] + ts[split + 1])
        cm, Xm = sample(mid)
        ts = np.concatenate([ts, mid])
        order = np.argsort(ts, kind="stable")
        ts = ts[order]
        c = np.concatenate([c, np.asarray(cm, float)])[order]
        X = np.concatenate([X, np.asarray(Xm, float)])[order]


def adaptive_static_lb(spec, q0, q1, pen_ext=None, h_inv=H_INV_DEFAULT,
                       boxes=None, n=SAMPLES, floor=None, tol=ADAPT_TOL):
    """`leg_static_lb`, converged. -> metres."""
    if boxes is None:
        boxes = static_boxes(spec)
    if not boxes:
        return np.inf
    q0 = np.asarray(q0, float).reshape(7)
    q1 = np.asarray(q1, float).reshape(7)
    dq = q1 - q0

    def sample(ts):
        Q = q0[None, :] + np.asarray(ts, float)[:, None] * dq[None, :]
        P = world_chain(Q, spec, pen_ext, h_inv)
        return frozen.chain_clearance(P, boxes,
                                      sphere_centres(Q, spec, h_inv)), P

    return float(adaptive_lb(sample, floor, n, tol)[0])


def leg_self_lb(spec, q0, q1, pen_ext=None, n=SAMPLES, floor=None,
                tol=ADAPT_TOL):
    """A LOWER BOUND on the arm's self-clearance over the whole straight move.

    -> metres.  `adaptive_lb` on the capsule ends, and given a `floor` only
    guaranteed to land on the right side of it.

    The screen is held EXACT to `SELF_SCREEN_PAD` above the floor rather than
    to the floor itself, because an interval is certified by its ENDS MINUS ITS
    RESIDUAL: a sample screened off at the floor comes back as a bound at the
    floor and the subtraction then puts the interval under it forever.
    """
    if not SELF_SAFE:
        return np.inf
    from . import selfcoll
    q0 = np.asarray(q0, float).reshape(7)
    q1 = np.asarray(q1, float).reshape(7)
    dq = q1 - q0
    fl = None if floor is None else float(floor)
    screen = None if fl is None else fl + SELF_SCREEN_PAD

    def sample(ts):
        Q = q0[None, :] + np.asarray(ts, float)[:, None] * dq[None, :]
        A, B, R = selfcoll.capsule_ends(Q, pen_ext)
        return (selfcoll.clearance_screened(A, B, R, screen),
                np.concatenate([A, B], axis=1))

    return float(adaptive_lb(sample, fl, n, tol)[0])


def block_self_lb(L, pen_ext=None, floor=None, k=SWEEP_K):
    """`leg_self_lb`'s coarse half for a whole BLOCK of sampled lines.

    -> (min (R,N), residual (R,N)), and the two come back APART for the reason
    `block_screen` keeps its static pair apart: `min` is an upper bound on the
    truth and `min - residual` a lower one, and a cell between them has to be
    LOOKED AT rather than believed either way.  Handed together they would
    flag every long move on this rig, because 33 samples of a metre of
    reconfiguration lose 30-60 mm to the residual and the floor is 23.

    `L` is (R, N, K, 7), the same array `block_screen` takes.
    """
    from . import selfcoll
    L = np.asarray(L, float)
    R_, N_, K_ = L.shape[:3]
    if not (R_ and N_):
        return np.zeros((R_, N_)), np.zeros((R_, N_))
    A, B, rad = selfcoll.capsule_ends(L.reshape(-1, 7), pen_ext)
    Ab = A.reshape(R_, N_, K_, -1, 3)
    Bb = B.reshape(R_, N_, K_, -1, 3)
    res = k * np.maximum(
        np.linalg.norm(np.diff(Ab, axis=2), axis=4).max(axis=(2, 3)),
        np.linalg.norm(np.diff(Bb, axis=2), axis=4).max(axis=(2, 3))) \
        if K_ > 1 else np.zeros((R_, N_))
    # THE SCREEN'S FLOOR CARRIES THE WORST RESIDUAL IN THE BLOCK, because the
    # verdict is `min - res >= floor` and the screen has to be exact wherever
    # that could go either way.  One number for the block rather than one per
    # cell: `clearance_screened` takes a scalar, and being exact on a few extra
    # pairs is cheaper than a per-cell dispatch.
    fl = None if floor is None else float(floor) + float(np.max(res))
    d = selfcoll.clearance_screened(A, B, rad, fl).reshape(R_, N_, K_)
    return d.min(axis=2), res


def leg_bounds(spec, q0, q1, pen_ext=None, h_inv=H_INV_DEFAULT, boxes=None,
               n=SAMPLES, floor=None):
    """Every gate's question about one straight move, off ONE FK pass.

    -> (chain_z, tip_z, static_lb).  `route` asks about thirty shapes built out
    of a handful of distinct legs and every one of them used to cost two
    forward-kinematics passes over the same configurations — one for the paper
    and one for the metal.  They are the same 11 points; this computes them
    once.  The static term keeps `leg_static_lb`'s early-outs and its broad
    phase, so a leg that is decided coarsely still is.
    """
    Q = line_samples(q0, q1, n)
    P = world_chain(Q, spec, pen_ext, h_inv)
    ccols = list(range(1, 9)) + ([10] if P.shape[1] >= 11 else [])
    cz = float(P[:, ccols, 2].min())
    tz = float(P[:, 9, 2].min())
    if boxes is None:
        boxes = static_boxes(spec)
    if not boxes:
        return cz, tz, np.inf
    C = sphere_centres(Q, spec, h_inv)
    bx = near_boxes(P, boxes, C=C)
    if not bx:
        return cz, tz, np.inf
    m = float(frozen.chain_clearance(P, bx, C).min())
    res = sample_residual(P)
    if floor is not None:
        if m - res >= float(floor) - EPS or m < float(floor) - EPS:
            return cz, tz, (m - res if m - res >= float(floor) - EPS else m)
    # the undecided band: subdivide where the bound binds, not everywhere
    return cz, tz, adaptive_static_lb(spec, q0, q1, pen_ext, h_inv, bx, n,
                                      floor)


def path_static_lb(spec, qs, pen_ext=None, h_inv=H_INV_DEFAULT, boxes=None,
                   n=SAMPLES, floor=None):
    """`leg_static_lb` along every straight leg of a polyline. -> metres.

    Stops at the first leg that fails `floor`: a polyline is only as clear as
    its worst leg, and the shapes this is asked about are mostly going to be
    refused (that is what a ladder is).
    """
    if boxes is None:
        boxes = static_boxes(spec)
    if not boxes:
        return np.inf
    qs = [np.asarray(q, float).reshape(7) for q in qs]
    out = np.inf
    for a, b in zip(qs[:-1], qs[1:]):
        out = min(out, leg_static_lb(spec, a, b, pen_ext, h_inv, boxes, n,
                                     floor))
        if floor is not None and out < float(floor) - EPS:
            break
    return float(out)


def chain_screen(qs, spec, pen_ext=None, h_inv=H_INV_DEFAULT, boxes=None):
    """`chain_tip_z` and `chain_static` off ONE forward-kinematics pass.

    -> (chain_z (M,), tip_z (M,), static (M,)).  The cost-matrix screens sample
    tens of thousands of configurations along the lines they are pricing and
    now have to ask both questions of every one of them; asking them separately
    is two FK passes over the same array, which on a full cluster matrix is
    seconds of the allocator's latency budget for nothing.
    """
    boxes = static_boxes(spec) if boxes is None else boxes
    qs = np.asarray(qs, float).reshape(-1, 7)
    P = world_chain(qs, spec, pen_ext, h_inv)
    ccols = list(range(1, 9)) + ([10] if P.shape[1] >= 11 else [])
    chain_z = P[:, ccols, 2].min(axis=1)
    tip_z = P[:, 9, 2]
    C = sphere_centres(qs, spec, h_inv)
    boxes = near_boxes(P, boxes, C=C) if boxes else boxes
    if not boxes:
        return chain_z, tip_z, np.full(len(qs), np.inf)
    return chain_z, tip_z, frozen.chain_clearance(P, boxes, C)


def block_screen(L, spec, pen_ext=None, h_inv=H_INV_DEFAULT, boxes=None):
    """A whole block of sampled lines. -> (chain_z, tip_z, static, residual).

    `L` is (R, N, K, 7): R x N straight moves, each sampled at K points.  One
    forward-kinematics pass answers every question, and the static minimum
    comes back with its 1-Lipschitz RESIDUAL alongside rather than folded into
    it, because the caller needs the two apart.  `min` is an upper bound on the
    truth and `min - residual` a lower one; a cell between them is a cell that
    has to be LOOKED AT more closely rather than believed either way, and
    `dive_screen` is where that decision belongs.

    The screen itself does not refine — that would be R x N x 32
    configurations for a question whose only job is to decide what to look at.
    """
    L = np.asarray(L, float)
    R, N, K = L.shape[:3]
    P = world_chain(L.reshape(-1, 7), spec, pen_ext, h_inv)
    Pb = P.reshape(R, N, K, P.shape[1], 3)
    ccols = list(range(1, 9)) + ([10] if P.shape[1] >= 11 else [])
    cz = Pb[:, :, :, ccols, 2].min(axis=(2, 3))
    tz = Pb[:, :, :, 9, 2].min(axis=2)
    C = sphere_centres(L.reshape(-1, 7), spec, h_inv)
    boxes = near_boxes(P, boxes, C=C) if boxes else boxes
    if not boxes:
        return cz, tz, np.full((R, N), np.inf), np.zeros((R, N))
    sc = frozen.chain_clearance(P, boxes, C).reshape(R, N, K).min(axis=2)
    res = SWEEP_K * np.linalg.norm(np.diff(Pb, axis=2), axis=4).max(axis=(2, 3))
    return cz, tz, sc, res


def line_samples(q0, q1, n=SAMPLES):
    """The straight joint-space move, sampled. -> (n,7)."""
    f = np.linspace(0.0, 1.0, int(n))[:, None]
    return np.asarray(q0, float).reshape(1, 7) * (1.0 - f) \
        + np.asarray(q1, float).reshape(1, 7) * f


def line_clearance(spec, q0, q1, pen_ext=None, h_inv=H_INV_DEFAULT,
                   n=SAMPLES):
    """Worst chain and tip height along one straight move. -> (chain, tip)."""
    cz, tz = chain_tip_z(line_samples(q0, q1, n), spec, pen_ext, h_inv)
    return float(cz.min()), float(tz.min())


def effective_floors(spec, q0, q1, pen_ext=None, h_inv=H_INV_DEFAULT,
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


def move_ok(spec, q0, q1, pen_ext=None, h_inv=H_INV_DEFAULT,
            tip_floor=TIP_CLEAR, chain_floor=CHAIN_CLEAR, n=SAMPLES):
    """Does the straight move keep its floors? -> (ok, chain, tip).

    "Fine as it is, do not route it" — which is why `STATIC_SAFE` has to be
    answered HERE and not only in `route`.  `sequence._leg_surcharge` and
    `_paper_surcharge` call this first and skip the router entirely when it says
    yes, so a frame-grazing move that clears the paper would never reach the
    ladder and the sequencer would price it at zero while `writing._route`
    routed it — the exact disagreement `csail_schedule.cross_check` exists to
    catch.  Both sides ask the same question, so both get the same answer.
    """
    tip_floor, chain_floor = effective_floors(spec, q0, q1, pen_ext, h_inv,
                                              tip_floor, chain_floor)
    boxes = static_boxes(spec) if STATIC_SAFE else []
    sf = (effective_static_floor(spec, q0, q1, pen_ext, h_inv, boxes=boxes)
          if boxes else None)
    cz, tz, sc = leg_bounds(spec, q0, q1, pen_ext, h_inv, boxes, n, sf)
    ok = bool(cz >= chain_floor - EPS and tz >= tip_floor - EPS)
    if ok and boxes:
        ok = bool(sc >= sf - EPS)
    # ...AND THE THIRD OBSTACLE, HERE TOO AND FOR THE SAME REASON `STATIC_SAFE`
    # is answered here: `sequence._leg_surcharge` and `_paper_surcharge` skip
    # the router entirely when this says yes, so a move that folds the arm
    # through itself would be priced at zero seconds while `writing._route`
    # routed it — the exact disagreement `csail_schedule.cross_check` exists to
    # catch, one obstacle over.
    if ok and SELF_SAFE:
        sfl = self_floor(spec, q0, q1, pen_ext)
        ok = bool(leg_self_lb(spec, q0, q1, pen_ext, n, sfl) >= sfl - EPS)
    return ok, cz, tz


def path_clearance(spec, qs, pen_ext=None, h_inv=H_INV_DEFAULT, n=SAMPLES):
    """Worst chain and tip height along a polyline of configurations."""
    qs = [np.asarray(q, float).reshape(7) for q in qs]
    cz, tz = np.inf, np.inf
    for a, b in zip(qs[:-1], qs[1:]):
        c, t = line_clearance(spec, a, b, pen_ext, h_inv, n)
        cz, tz = min(cz, c), min(tz, t)
    return float(cz), float(tz)


def static_boxes(spec):
    """The full static set this arm must clear. -> list of boxes.

    Every neighbour's steel AND every neighbour's pose-invariant base column,
    which on the all-ceiling rig is the obstacle that actually blocks pen-ups
    (`mounts.attach_body_columns`).  One accessor, so the router, the hover
    solver and the screen cannot end up asking about different rooms.
    """
    boxes = spec.static_obstacles() if hasattr(spec, "static_obstacles") else []
    # ...and with `frozen` switched on, a partner known to be holding a pose is
    # modelled by that pose's capsules instead of its pose-invariant band (see
    # aris_sixarm/frozen.py).  Off by default: `filter_boxes` is the identity.
    #
    # `envelope.swap` then replaces whatever body bands are LEFT — the partners
    # that are still modelled pose-invariantly — with the band's own CYLINDER
    # rather than its bounding box.  The AABB padded a 128 mm cylinder into a
    # 449 mm box and that padding is what refused the pen-ups under a base
    # (see aris_sixarm/envelope.py).  Off by default; identity when off.
    return envelope.swap(frozen.filter_boxes(boxes))


def chain_static(qs, spec, pen_ext=None, h_inv=H_INV_DEFAULT, boxes=None):
    """Per-configuration clearance to the static set. -> (M,).

    The PER-POSE form, which `frame_clearance` then minimises over.  It is the
    per-pose one that the lift layer needs: a hover is a configuration an arm
    HOLDS, and "is this pose legal" is not a question about a path.
    """
    boxes = static_boxes(spec) if boxes is None else boxes
    qs = np.asarray(qs, float).reshape(-1, 7)
    if not boxes or not len(qs):
        return np.full(len(qs), np.inf)
    Twb = spec.T_world_base(h_inv)
    R, t = Twb[:3, :3], Twb[:3, 3]
    T, p = fk_many(qs)
    pw = p @ R.T + t
    tool = [tp @ R.T + t for tp in tool_points_many(T, pen_ext)]
    P10 = np.concatenate([pw] + [tp[:, None, :] for tp in tool], axis=1)
    return frozen.chain_clearance(P10, boxes, sphere_centres(qs, spec, h_inv))


def frame_clearance(qs, spec, pen_ext=None, h_inv=H_INV_DEFAULT, boxes=None):
    """Worst clearance from the chain (pen tip included) to the static set.

    A VIA THAT TRADES A PAPER HIT FOR A FRAME HIT IS NOT A FIX, and the two
    failure modes pull in opposite directions: the way out of the paper is UP,
    and up is where the top rails, the corner posts and the booms are.  On the
    first routed conduct of this logo `scene_check` refused a profile at
    49.9 mm of frame clearance against a 50 mm margin — 0.1 mm — with the paper
    gate passing comfortably, which is exactly the trade this exists to stop
    the router making.  Same geometry as `validate.validate_plan`'s frame gate,
    via `rig_final.chain_static_clearance`.
    """
    if boxes is None:
        boxes = static_boxes(spec)
    if not boxes:
        return np.inf
    return float(chain_static(qs, spec, pen_ext, h_inv, boxes).min())


def pose_static_ok(q, spec, pen_ext=None, h_inv=H_INV_DEFAULT,
                   floor=FRAME_FLOOR, boxes=None):
    """May the arm STAND here, as far as the static set is concerned? -> bool.

    The gate the lift layer was missing.  `atlas.solve_cell` asks it of every
    drawing pose it certifies; nothing asked it of the hover 6 cm above that
    pose, which is a DIFFERENT configuration — same tip, different elbow — and
    on the proposed rig it is inside a neighbour's base column for 4 to 15 % of
    every arm's certified cells (`out/transit_block.py`).
    """
    if not STATIC_SAFE:
        return True
    return bool(chain_static(np.asarray(q, float).reshape(1, 7), spec, pen_ext,
                             h_inv, boxes)[0] >= float(floor) - EPS)


def effective_static_floor(spec, q0, q1, pen_ext=None, h_inv=H_INV_DEFAULT,
                           floor=FRAME_FLOOR, boxes=None):
    """The static floor a move can actually be held to. -> metres.

    `effective_floors`' argument, one obstacle over, and it is the whole reason
    the static gate could not be switched on before.  `FRAME_FLOOR` is 53 mm —
    `STATIC_MARGIN` plus the checker's sweep residual — and the ATLAS certifies
    a drawing pose at exactly `STATIC_MARGIN`.  So a certified cell may stand
    51 mm from a neighbour's column (4-6 of every 300 do), and a lift out of it
    that is asked for 53 mm is asked for something its own first sample does
    not have: no route can satisfy it, every edge out of that span prices
    `inf`, and the arm is stranded on ink it can draw.

    Clamping to the endpoints keeps the gate meaningful where it can be met and
    inert where the geometry already lost — and `scene_check`, which does not
    clamp anything, still has the last word on whether such a pose may ship.
    """
    if boxes is None:
        boxes = static_boxes(spec)
    if not boxes:
        return -np.inf
    ends = np.stack([np.asarray(q0, float).reshape(7),
                     np.asarray(q1, float).reshape(7)])
    return float(min(float(floor), float(chain_static(ends, spec, pen_ext,
                                                      h_inv, boxes).min())))


def path_frame_clearance(spec, qs, pen_ext=None, h_inv=H_INV_DEFAULT,
                         n=SAMPLES, boxes=None):
    """`frame_clearance` along every straight leg of a polyline of configs."""
    if boxes is None:
        boxes = static_boxes(spec)
    if not boxes:
        return np.inf
    qs = [np.asarray(q, float).reshape(7) for q in qs]
    out = np.inf
    for a, b in zip(qs[:-1], qs[1:]):
        out = min(out, frame_clearance(line_samples(a, b, n), spec, pen_ext,
                                       h_inv, boxes))
    return float(out)


def tip_xy(q, spec, pen_ext=None, h_inv=H_INV_DEFAULT):
    """Where this configuration's pen tip is on the paper. -> (2,)."""
    Twb = spec.T_world_base(h_inv)
    return (Twb[:3, :3] @ tip_pos_many(np.asarray(q, float).reshape(1, 7),
                                       pen_ext)[0] + Twb[:3, 3])[:2]


# --------------------------------------------------------------------------
# routing
# --------------------------------------------------------------------------
TRAVERSE_STEPS = (0.30, 0.20, 0.12)   # m of paper per hop, coarsest first


SKIRT_PADS = (0.18, 0.30)      # m the hover plane keeps around a blocking box
SKIRT_HEIGHTS = (0.08, 0.15)   # hover heights a sidestep is tried at
SKIRT_TRIES = 16               # detour shapes one crossing may be offered


def _box_xy(boxes):
    """The xy footprints of a box list. -> (B,2,2) [lo/hi] x [x/y]."""
    if not boxes:
        return np.zeros((0, 2, 2))
    return np.stack([np.stack([np.asarray(b["lo"], float)[:2],
                               np.asarray(b["hi"], float)[:2]])
                     for b in boxes])


def _seg_box_xy(xy0, xy1, lo, hi, n=17):
    """Does the xy segment come within nothing of this footprint? -> bool."""
    f = np.linspace(0.0, 1.0, n)[:, None]
    P = np.asarray(xy0, float) * (1 - f) + np.asarray(xy1, float) * f
    d = np.maximum(lo[None] - P, P - hi[None])
    return bool((np.linalg.norm(np.maximum(d, 0.0), axis=1) <= 1e-9).any())


def _skirt(spec, xy0, xy1, boxes, pads=SKIRT_PADS):
    """xy waypoint lists that walk AROUND the boxes in the way. -> [[xy, ...]].

    THE WAY PAST A COLUMN IS AROUND IT.  Everything this module offered before
    was a higher hover, which is the right escape from the PAPER — the paper is
    below and the air above it is empty — and the wrong escape from a neighbour
    that stands 0.6 m tall between two arms 0.61 m apart.  Worse, on this rig up
    is where the ceiling grid is, so the vertical ladder trades one static hit
    for another and the router reports "no route" for a crossing whose obstacle
    a 25 cm sidestep clears.

    The obstacles are known: they are the same boxes every other gate measures
    against, with published footprints.  For each one the straight hover walk
    would cross, this offers the four corners of its footprint grown by `pad`,
    singly (round one corner) and in adjacent pairs (round one whole side), for
    three pads.  The waypoints are xy only — `_traverse` turns each into a
    chain of certified hovers, and a waypoint that no hover can reach simply
    drops out.
    """
    xy0 = np.asarray(xy0, float).reshape(2)
    xy1 = np.asarray(xy1, float).reshape(2)
    fp = _box_xy(boxes)
    out, seen = [], set()

    def add(pts):
        k = np.round(np.asarray(pts, float), 4).tobytes()
        if k not in seen:
            seen.add(k)
            out.append([np.asarray(p, float).reshape(2) for p in pts])

    for pad in pads:
        # cluster every footprint the walk would cross, at this pad, into one
        # rectangle: two columns side by side are one obstacle to go round.
        hit = [f for f in fp
               if _seg_box_xy(xy0, xy1, f[0] - pad, f[1] + pad)]
        if not hit:
            continue
        lo = np.min([f[0] for f in hit], axis=0) - pad
        hi = np.max([f[1] for f in hit], axis=0) + pad
        c = [np.array([lo[0], lo[1]]), np.array([hi[0], lo[1]]),
             np.array([hi[0], hi[1]]), np.array([lo[0], hi[1]])]
        # nearest corner first: the cheapest way round is the short way
        order = sorted(range(4), key=lambda i: float(
            np.linalg.norm(c[i] - xy0) + np.linalg.norm(c[i] - xy1)))
        for i in order:
            add([c[i]])
        for i in order[:2]:                  # round ONE side, the near two
            for j in ((i + 1) % 4, (i - 1) % 4):
                add([c[i], c[j]])
    # ...and a plain perpendicular sidestep, for an obstacle whose footprint is
    # not what the chain is hitting (a boom overhead, a plate off to one side)
    u = xy1 - xy0
    nrm = float(np.linalg.norm(u))
    if nrm > 1e-6:
        nvec = np.array([-u[1], u[0]]) / nrm
        for d in pads:
            for s in (1.0, -1.0):
                add([0.5 * (xy0 + xy1) + s * d * nvec])
    return out[:SKIRT_TRIES]


def _walk(spec, q_from, xy_pts, z, pen_ext, h_inv, mm, step, lift):
    """Hovers walking a POLYLINE of xy waypoints at height `z`. -> [q] | None.

    `_traverse` for more than one leg, each leg solved nearest to the hover the
    previous one ended on, so the whole detour stays on one analytic branch for
    the same reason a straight traverse does.
    """
    out, ref = [], q_from
    for a, b in zip(xy_pts[:-1], xy_pts[1:]):
        leg = _traverse(spec, ref, a, b, z, pen_ext, h_inv, mm, step, lift,
                        ends=True)
        if not leg:
            return None
        out += leg
        ref = leg[-1]
    return out


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


def route(spec, q0, q1, pen_ext=None, h_inv=H_INV_DEFAULT,
          tip_floor=TIP_CLEAR, chain_floor=CHAIN_CLEAR, heights=VIA_HEIGHTS,
          n=SAMPLES, margin_min=None, cache=True, q_home=None,
          steps=TRAVERSE_STEPS, rrt=True):
    """A pen-up route from q0 to q1 that clears the paper AND the metal.
    -> dict | None.

    -> dict(vias, mode, chain_z, tip_z, tried) where `vias` is the (possibly
    empty) list of intermediate configurations such that EVERY consecutive
    straight joint-space move in [q0, *vias, q1] keeps the pen tip at least
    `tip_floor` and every chain point at least `chain_floor` above the paper —
    and, under `STATIC_SAFE`, at least `effective_static_floor` from every
    neighbour's steel and base column.  `None` means no shape on the ladder
    certified — the caller must refuse the move rather than fly it, which is
    the entire point, and `sequence`'s cost matrix turns that refusal into an
    infinite edge so the tour goes round it.

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

    ...and then, when the whole ladder has failed, a SKIRT: the same hover walk
    routed around the footprint of whatever static box is in the way.  Every
    shape above it goes UP, which is the answer to a table and not to a column;
    see `_skirt`.

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
    from .writing import HOVER_MARGIN, hover_solve, static_gate
    mm = HOVER_MARGIN if margin_min is None else float(margin_min)
    boxes = static_boxes(spec)
    # A VIA IS A POSE THIS FUNCTION CHOOSES, so it is held to the full static
    # floor and searched over the whole fiber when the nearest one is blocked —
    # the same treatment `writing.lifted_or_lower` now gives a stroke's own
    # hover, for the same reason.  `None` on the legacy rigs, where the scan
    # never widens because there is nothing to fail.
    gate = static_gate(spec, pen_ext, h_inv) if boxes else None

    def lift(sp, ref, xy, z, pe, hi, margin):
        """`hover_solve`, memoised on (arm, reference pose, target, height).

        THE FAILURE PATH IS THE HOT ONE.  A pair that routes does so on the
        first or second shape; a pair that CANNOT route walks the whole ladder,
        and the fiber-menu matrices ask about tens of thousands of pairs whose
        endpoints are drawn from a few hundred distinct hover poses.  The lift
        over a given pose at a given height is the same solution every time it
        is asked for — `hover_solve` is deterministic — so solving it once
        per (pose, height) instead of once per PAIR is the difference between
        the cluster profile costing seconds and costing minutes.  Nothing about
        the answer changes; a test pins the memo against the direct call.
        """
        k = (id(sp), ext_of(pe), float(_frames.PEN_LAT), float(hi), float(z),
             round(float(margin), 9), gate is not None,
             np.round(np.asarray(ref, float), 9).tobytes(),
             np.round(np.asarray(xy, float), 9).tobytes())
        if k not in _LIFTS:
            _LIFTS[k] = hover_solve(sp, ref, xy, z=z, h_inv=hi, pen_ext=pe,
                                    margin_min=margin, ok=gate)
        return _LIFTS[k], None

    q0 = np.asarray(q0, float).reshape(7)
    q1 = np.asarray(q1, float).reshape(7)
    tip_floor, chain_floor = effective_floors(spec, q0, q1, pen_ext, h_inv,
                                              tip_floor, chain_floor)
    # ...and the same clamp for the metal.  See `effective_static_floor`: the
    # atlas certifies ink at STATIC_MARGIN and this module asks for
    # STATIC_MARGIN + the checker's residual, so an unclamped floor refuses
    # every leg out of the tightest certified cells.
    static_floor = (effective_static_floor(spec, q0, q1, pen_ext, h_inv,
                                           boxes=boxes)
                    if boxes and STATIC_SAFE else -np.inf)
    ck = _key(spec, q0, q1, pen_ext, h_inv, tip_floor, chain_floor,
              q_home is not None, rrt) if cache \
        else None
    if ck is not None and ck in _CACHE:
        return _CACHE[ck]

    gate_floor = static_floor if STATIC_SAFE else FRAME_FLOOR
    lk = (id(spec), ext_of(pen_ext), float(_frames.PEN_LAT), float(h_inv),
          round(float(gate_floor), 9), int(n))
    # THE SELF MEMO IS NOT KEYED ON THE ARM, and that is a property of the
    # question rather than an optimisation: self-collision is one arm against
    # its own metal in its own base frame, so two arms holding the same joints
    # have the same answer and `id(spec)`/`h_inv` have nothing to say about it.
    self_fl = self_floor(spec, q0, q1, pen_ext)
    sk = (ext_of(pen_ext), float(_frames.PEN_LAT), round(float(self_fl), 9),
          int(n))

    def leg(a, b):
        """`leg_bounds`, memoised on the pair (see `_LEGS`)."""
        k = lk + (_pose_bytes(a), _pose_bytes(b))
        if k not in _LEGS:
            _LEGS[k] = leg_bounds(spec, a, b, pen_ext, h_inv, boxes, n,
                                  gate_floor if boxes else None)
        return _LEGS[k]

    def leg_self(a, b):
        """`leg_self_lb`, memoised on the pair (see `_SELF`)."""
        k = sk + (_pose_bytes(a), _pose_bytes(b))
        if k not in _SELF:
            _SELF[k] = leg_self_lb(spec, a, b, pen_ext, n, self_fl)
        return _SELF[k]

    def legs_ok(seq, frame=True):
        qs = [q0] + list(seq) + [q1]
        cz, tz, sc = np.inf, np.inf, np.inf
        for a, b in zip(qs[:-1], qs[1:]):
            c, t, s = leg(a, b)
            cz, tz, sc = min(cz, c), min(tz, t), min(sc, s)
            if cz < chain_floor - EPS or tz < tip_floor - EPS:
                break
        ok = cz >= chain_floor - EPS and tz >= tip_floor - EPS
        # With STATIC_SAFE off the metal is only asked about a route we are
        # INSERTING, and a direct move that already clears the paper is left
        # exactly as it was.  With it on — the default since 2026-08-26 — the
        # direct move is asked too, which is the whole point: `scene_check`
        # having the last word on the frame is no use to a tour that was
        # costed, chosen and frozen before anybody looked.  That is not an
        # abstraction: arm 2's conducted solo timeline was refused at 41.0 mm
        # against the checker's 50, on a pen-up nobody had priced.
        if ok and frame and boxes and (seq or STATIC_SAFE):
            ok = bool(sc >= gate_floor - EPS)
        # ...AND THE ARM AGAINST ITSELF, LAST, because it is the only one of
        # the three that needs its own forward kinematics: a shape refused by
        # the paper or the metal never pays for it.  Every leg of the assembled
        # shape, the direct move included — a fold is exactly the thing a
        # straight joint-space line commits and a via cannot inherit.
        if ok and SELF_SAFE:
            for a, b in zip(qs[:-1], qs[1:]):
                if leg_self(a, b) < self_fl - EPS:
                    ok = False
                    break
        return ok, cz, tz

    def done(seq, name, cz, tz, tried):
        out = dict(vias=[np.asarray(v, float).reshape(7) for v in seq],
                   mode=name, chain_z=float(cz), tip_z=float(tz), tried=tried)
        if ck is not None:
            _CACHE[ck] = out
        return out

    ok, cz, tz = legs_ok([], frame=STATIC_SAFE)
    if ok:
        return done([], "direct", cz, tz, 1)

    xy0 = tip_xy(q0, spec, pen_ext, h_inv)
    xy1 = tip_xy(q1, spec, pen_ext, h_inv)
    xym = 0.5 * (xy0 + xy1)
    tried = 1
    # WHICH OBSTACLE IS IT?  The two failure modes want opposite escapes: the
    # way off the paper is up, and the way past a neighbour's base column is
    # round.  A crossing that clears the canvas and fails only on the metal
    # therefore walks the vertical ladder for nothing — seven heights and forty
    # shapes of IK — before reaching the one family that can help it, and on
    # this rig that is most crossings that need routing at all.  So the order
    # follows the diagnosis: metal-only, skirt first.
    metal_only = bool(cz >= chain_floor - EPS and tz >= tip_floor - EPS)

    def skirts(base):
        """The sidestep family. -> (result, tried) with result None if none fit."""
        t = base
        if not boxes:
            return None, t
        for z in SKIRT_HEIGHTS:
            for wp in _skirt(spec, xy0, xy1, boxes):
                seq = _walk(spec, q0, [xy0] + wp + [xy1], z, pen_ext, h_inv,
                            mm, SKIRT_STEP, lift)
                if not seq:
                    continue
                t += 1
                good, c, tt = legs_ok(seq)
                if good:
                    return done(seq, f"skirt{len(wp)}@{100 * z:.0f}cm", c, tt,
                                t), t
        return None, t

    if metal_only:
        got, tried = skirts(tried)
        if got is not None:
            return got
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
    # ...and for a crossing that was blocked by the TABLE, going round is the
    # last thing left rather than the first: it is longer than a lift and the
    # sequencer pays the difference, so it is tried only once every shape that
    # buys clearance more cheaply has failed.  Skipped entirely when nothing
    # static is in the room, which is every legacy rig, and already spent above
    # when the metal was the whole problem.
    if not metal_only:
        got, tried = skirts(tried)
        if got is not None:
            return got
    # THE FOLD THROUGH THE DEPOT, ROUTED RATHER THAN FLOWN STRAIGHT.
    #
    # `q_home` has been offered twice already — as the bare via `[qh]` and as
    # `[a0, qh, a1]` — and both ask `legs_ok` to fly a STRAIGHT joint-space line
    # into the depot and another one out of it.  That is a far stronger demand
    # than "the arm can get home and set off again", and the difference is the
    # whole of this rig's coverage: `out/residual_anatomy.py` routes every one
    # of the logo's 38 strokes from every arm's own depot, and arm 31 reaches
    # all of them — yet 29 % of its span-to-span crossings have no route, so
    # `prune_unflyable` finds no Hamiltonian path through its bag and gives the
    # ink back.  Both halves of the journey exist.  Only the straight line
    # between them does not.
    #
    # So the last shape on the ladder is the composite: route q0 -> depot and
    # depot -> q1 with this same function, and hand the concatenation to the
    # same `legs_ok` every other shape is certified by.  Three things make it
    # honest rather than a special case:
    #
    #   * it is CERTIFIED, not assumed.  The sub-routes are certified against
    #     their own endpoints; the composite is then re-checked end to end at
    #     THIS call's floors, which are the ones the caller asked about.
    #   * it is EXECUTED, not merely priced.  Every consumer — the sequencer's
    #     matrix, `prune_unflyable`, and `writing.arm_program`'s transit — is
    #     the same `route` call, so the vias the tour was costed on are the
    #     vias the timeline lays down.  That equality is what the pipeline's
    #     "sequencer priced transits the timeline does not pay" cross-check
    #     exists to catch, and it is preserved by construction here.
    #   * it is LAST, because it is dear.  A trip to the depot and back is the
    #     most expensive escape on the ladder, the sequencer pays for it in
    #     real seconds, and so a tour takes it only where nothing cheaper
    #     certified.
    #
    # `q_home=None` in the two sub-calls is what stops the recursion at one
    # level: a fold through the depot on the way to the depot is not a shape.
    if q_home is not None:
        qh = np.asarray(q_home, float).reshape(7)
        if not (np.allclose(qh, q0) or np.allclose(qh, q1)):
            sub = dict(pen_ext=pen_ext, h_inv=h_inv, tip_floor=tip_floor,
                       chain_floor=chain_floor, heights=heights, n=n,
                       margin_min=margin_min, cache=cache, steps=steps,
                       q_home=None, rrt=False)
            r0 = route(spec, q0, qh, **sub)
            r1 = route(spec, qh, q1, **sub)
            if r0 is not None and r1 is not None:
                seq = list(r0["vias"]) + [qh] + list(r1["vias"])
                tried += 1
                ok, cz, tz = legs_ok(seq)
                if ok:
                    return done(seq, "fold_home", cz, tz, tried)
    # ==================================================================
    # THE LADDER IS OUT.  SEARCH THE CONFIGURATION SPACE.
    # ==================================================================
    # Everything above walks a two-dimensional surface — a tip position on a
    # hover plane, with the elbow following whatever the analytic solver hands
    # back — and the crossings that survive it are the ones whose endpoints
    # are in different components of THAT surface while being perfectly well
    # connected in the seven-dimensional space the arm actually moves in.
    # `transit.plan` searches the seven.
    #
    # It is offered the depot as an extra root for the same reason `fold_home`
    # exists: the park pose is the one configuration on the far side of a
    # branch change that is certified by construction, and a tree that already
    # contains it starts halfway across the reconfiguration.  A root the gate
    # refuses is simply dropped.
    #
    # And the result is certified by `legs_ok`, not by the planner.  The
    # planner holds every edge to these floors plus `transit.PAD`; `legs_ok`
    # then re-derives the same three bounds over the assembled polyline at
    # `SAMPLES` per leg with refinement, exactly as it does for a skirt or a
    # traverse.  The two cannot disagree in the dangerous direction, and if
    # they disagree at all this refuses.
    if rrt and RRT_SAFE:
        from . import transit
        probe = None if RRT_PROBE is None else RRT_PROBE[0]
        pmargin = 0.0 if RRT_PROBE is None else float(RRT_PROBE[1])
        seq = transit.plan(spec, q0, q1, pen_ext=pen_ext, h_inv=h_inv,
                           boxes=boxes, chain_floor=chain_floor,
                           tip_floor=tip_floor,
                           static_floor=static_floor if boxes else -np.inf,
                           self_floor=self_fl if SELF_SAFE else -np.inf,
                           probe=probe, probe_margin=pmargin,
                           extra_roots=() if q_home is None
                           else (np.asarray(q_home, float).reshape(7),))
        if seq:
            tried += 1
            ok, cz, tz = legs_ok(seq)
            if ok:
                return done(seq, f"rrt{len(seq)}", cz, tz, tried)
            transit._STATS["recert_failed"] += 1
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
