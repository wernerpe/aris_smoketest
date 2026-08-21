"""What an arm does when it is not drawing — the conductor's idle policy.

An arm that has finished is not a non-participant: it is a metre and a half of
steel standing in a shared workspace for the rest of the run.  Conductor v1
handled that the only way it knew how — every arm went back to `spec.q_seed`
when its programme ended — and the bill came in `docs/CONCURRENCY.md`: 79 % of
every pause-second the fleet paid was an arm waiting for permission to reach its
PARK pose, not an arm held up on its way to ink.  The rest-suffix rule is not
the bug (an arm may not arrive until the pose it will then stand in is clear all
the way to the horizon, and that is exactly right); the bug is choosing to
arrive somewhere unhelpful.

Three policies, in the order they are applied, all of them still conducted by
the same DP against the same collision images — none of this weakens a gate:

  FREEZE-IN-PLACE   the default.  An arm that finishes lifts its pen to hover
                    height and STOPS THERE.  The pose above the ink it has just
                    laid is a pose every other arm was already avoiding for the
                    length of that stroke, so the horizon-long clearance the
                    rest rule demands is usually free; `q_seed`, in the middle
                    of the rig, is precisely where everybody else's transits go.
                    The frozen arm is then static geometry, which the machinery
                    already models correctly: `coordination._dp`'s rest suffix
                    keeps checking it after arrival, and `scene_check` re-derives
                    it from the played-back trajectory.

  MINIMAL RETREAT   for the frozen pose that is genuinely in the way.  "Genuinely"
                    is measured, not guessed: `coordination.rest_delays` re-runs
                    each arm's own DP with the rest requirement dropped, and only
                    an arm that actually LOST SECONDS to where it stops is
                    offered a retreat.  The search is outward along a few
                    candidate directions — lift straight up, then back toward the
                    arm's own base — ordered by how far the arm has to move, and
                    the first candidate that certifies as a pose AND leaves every
                    other arm's remaining swept tube wins.  It is a few
                    centimetres, not a trip home.

  JIT PRE-POSITION  an arm with slack does not need to race to its next entry
                    and stand there.  Every pen-up block of its programme is
                    stretched — the ink never is — by the slack the conductor
                    measured it to have, so it arrives at each entry at or before
                    its slot.  Same path, same certificate, a fraction of the
                    per-step motion, and therefore a fraction of the swept-tube
                    slack it charges everybody else (99.7 % of all blocking in
                    the shipped run was sweep slack over poses already 80 mm
                    clear).  DRAWING HAS PRIORITY OVER TAXIING, twice over: the
                    stretch is only ever spent out of slack an arm demonstrably
                    has in a schedule that has already been conducted, and if the
                    stretched fleet conducts SLOWER than the unstretched one the
                    stretch is thrown away.

AND THE POLICY IS PER ARM.  Some finishing poses cannot be made safe at all —
the arm ends its programme somewhere its neighbour has to sweep through, and no
lift or step back gets out of the way.  That arm, and only that arm, takes
conductor v1's answer and goes home; the rest keep what freezing bought them.
An all-or-nothing policy would let one awkward stroke undo the whole change.

`conduct` is the driver: it builds the frozen timelines, conducts them, and
applies the policies in that order, each one paid for by a re-conduct it only
keeps if the clock agrees.  `policy="home"` restores conductor v1's behaviour
exactly, which is what the A/B in `tests/test_csail.py` compares against.
"""
import numpy as np

from . import coordination, validate, writing
from .fleet import FLEET, H_INV_DEFAULT
from .frames import PEN_EXT

POLICY_FREEZE = writing.PARK_FREEZE
POLICY_HOME = writing.PARK_HOME

JIT_FRAC = 0.75          # of the slack a conducted schedule measured, not of
#   the nominal floor: an arm that finishes 20 s before the fleet does may
#   spend 15 of them taxiing slowly and still keep 5 s of pause budget.  Spend
#   all of it and the first second of pause the arm then takes becomes a second
#   of makespan, which is how a policy meant to buy time gives it back.
RETREAT_UP = (0.04, 0.08, 0.14, 0.20)      # m of extra hover height to try
RETREAT_BACK = (0.06, 0.12, 0.20, 0.30)    # m back along base->pen, to try
RETREAT_MIN_DELAY = 1e-9   # s of rest delay below which a frozen pose is fine


def _h(h_inv):
    """`None` means the default inverted height, as it does in `validate`.

    Not decoration: `ArmSpec.T_world_base(None)` puts a nan in the translation
    of an inverted arm's base rather than raising, which turns every IK call
    downstream into a silent "no solution" — a retreat search that quietly
    finds nothing is exactly the failure mode this module must not have.
    """
    return H_INV_DEFAULT if h_inv is None else h_inv


# ==========================================================================
# is a static pose clear of what is left of another arm's path?
# ==========================================================================
def tube_clearance(q, arm, pen, other, j0=0, sweep=coordination.SWEEP_K,
                   h_inv=H_INV_DEFAULT, spec=None):
    """Clearance from ONE pose to the tail of another arm's path. -> metres.

    `other` is an `ArmPath`; `j0` is the first progress index it has not yet
    passed.  The bound is the cell bound the conductor uses — the worse of the
    two corner clearances minus the sweep slack of the moving arm — so a pose
    this calls clear by `margin` is a pose the collision image agrees is clear,
    not a pose that merely samples clear.  The static arm contributes no sweep,
    which is the whole point of standing still.
    """
    pa = coordination.ArmPath(arm, np.asarray(q, float)[None, :], other.dt,
                              _h(h_inv), pen, spec)
    D = coordination.clearance_matrix(pa, other)[0]           # (n_other,)
    if len(D) < 2:
        return float(D.min())
    cell = np.minimum(D[:-1], D[1:]) - sweep * other.step
    j0 = int(np.clip(j0, 0, len(cell) - 1))
    return float(cell[j0:].min())


def frozen_interference(q, arm, pen, paths, tails, margin,
                        sweep=coordination.SWEEP_K, h_inv=H_INV_DEFAULT,
                        spec=None):
    """Which arms' remaining tubes a frozen pose sits inside. -> {arm: clearance}.

    `tails[b]` is the first index of b's path that b has not yet reached.  Only
    pairs closer than `margin` are returned, so an empty dict means the pose is
    clear of the whole rest of the run.
    """
    out = {}
    for b, pb in paths.items():
        if b == arm:
            continue
        c = tube_clearance(q, arm, pen, pb, tails.get(b, 0), sweep, h_inv, spec)
        if c < margin:
            out[b] = c
    return out


# ==========================================================================
# the retreat: outward, a few centimetres, certified
# ==========================================================================
def retreat_candidates(spec, q_frozen, h_inv=H_INV_DEFAULT, pen_ext=PEN_EXT,
                       ups=RETREAT_UP, backs=RETREAT_BACK,
                       margin_min=validate.MARGIN_GATE):
    """Poses to try, nearest first. -> [dict(q, tag, dq)].

    Two directions and no more, because a retreat that needs a search is a
    retreat that should have been a re-placement:

      UP    straight up off the paper at the same (x, y).  The cheapest thing an
            arm can do, it keeps the pen over its own ink (so it cannot wander
            into a neighbour's sheet), and height is where the room is.
      BACK  the same lift, with the pen tip pulled toward the arm's own base
            along the base->pen ray.  An arm folded over its own base is out of
            the middle of the rig, which is where the contention is.

    Every candidate is a `lifted_config` IK solution nearest to the pose the arm
    is already in, so "smallest" is measured in the joint space the arm actually
    has to move through, and the list is sorted by it.  Nothing is certified
    here — `plan_retreat` does that, because a candidate is only interesting if
    it also gets the arm out of the way.
    """
    q_frozen = np.asarray(q_frozen, float)
    h_inv = _h(h_inv)
    Twb = spec.T_world_base(h_inv)
    tip = Twb[:3, :3] @ writing.tip_pos(q_frozen, pen_ext) + Twb[:3, 3]
    xy, z0 = tip[:2], float(tip[2])
    base = np.asarray(spec.xy, float)
    u = xy - base
    r = float(np.linalg.norm(u))
    u = u / max(r, 1e-9)

    tries = [(xy, z0 + dz, f"up {100 * dz:.0f} cm") for dz in ups]
    for db in backs:
        if db > r - 0.15:                       # do not fold onto the base column
            continue
        for dz in (ups[0], ups[1]):
            tries.append((xy - db * u, z0 + dz,
                          f"back {100 * db:.0f} cm, up {100 * dz:.0f} cm"))
    out = []
    for pxy, z, tag in tries:
        q, _ = writing.lifted_config(spec, q_frozen, pxy, z=z, h_inv=h_inv,
                                     pen_ext=pen_ext, margin_min=margin_min)
        if q is None:
            continue
        out.append(dict(q=np.asarray(q, float), tag=tag,
                        dq=float(np.max(np.abs(q - q_frozen)))))
    out.sort(key=lambda c: c["dq"])
    return out


def plan_retreat(spec, q_frozen, arm, pen, paths, tails, margin,
                 sweep=coordination.SWEEP_K, h_inv=H_INV_DEFAULT,
                 ups=RETREAT_UP, backs=RETREAT_BACK):
    """The smallest certified pose that leaves the interference set. -> dict|None.

    -> dict(q, tag, dq, clearance, before, tried) or None if nothing offered
    gets clear.  A candidate has to survive three questions in this order, and
    the order is the cheap-first one:

      1. is it a pose this arm may stand in at all — joint limits with the
         planner's comfort margin, chain above the paper, boom keep-out
         (`validate.check_pose`);
      2. is it clear, by the conductor's own cell bound, of every other arm's
         REMAINING path (not its whole path: an arm cannot be blocked by a tube
         its neighbour has already flown through);
      3. is it the smallest such move offered.

    Returning None is a legitimate answer and not a failure — it means freezing
    in place is the best this policy has, and the conductor will schedule the
    wait it implies exactly as before.
    """
    before = frozen_interference(q_frozen, arm, pen, paths, tails, margin,
                                 sweep, h_inv, spec)
    if not before:                  # nothing to get out of the way of
        return None
    cands = retreat_candidates(spec, q_frozen, h_inv, pen, ups, backs)
    for c in cands:
        rep = validate.check_pose(c["q"], spec, h_inv, pen)
        if not rep["ok"]:
            c["refused"] = "pose: " + ",".join(v["kind"] for v in rep["violations"])
            continue
        hit = frozen_interference(c["q"], arm, pen, paths, tails, margin,
                                  sweep, h_inv, spec)
        if hit:
            c["refused"] = "still inside " + ",".join(str(b) for b in sorted(hit))
            continue
        worst = min((tube_clearance(c["q"], arm, pen, pb, tails.get(b, 0), sweep,
                                    h_inv, spec) for b, pb in paths.items()
                     if b != arm), default=float("inf"))
        return dict(q=c["q"], tag=c["tag"], dq=c["dq"], clearance=float(worst),
                    before={int(b): float(v) for b, v in before.items()},
                    tried=len(cands))
    return None


# ==========================================================================
# the driver
# ==========================================================================
def _programs(specs, segs_by_arm, pens, q_start, parks, stretch, retreats,
              draw_speed, transit_speed, qd_frac, h_inv, only=None, prev=None,
              verbose=False):
    """`writing.arm_program` for every arm, re-using `prev` where nothing moved.

    THE POLICY IS PER ARM, and that is not a detail.  One arm whose finishing
    pose cannot be made safe would otherwise force the whole fleet back to
    conductor v1's behaviour; instead that arm goes home and the other five keep
    what freezing bought them.  `parks[a]` is "freeze" or "home".
    """
    out = dict(prev or {})
    for a, segs in segs_by_arm.items():
        if only is not None and a not in only and a in out:
            continue
        out[a] = writing.arm_program(
            specs[a], segs, draw_speed=draw_speed, transit_speed=transit_speed,
            h_inv=h_inv, qd_frac=qd_frac, pen_ext=pens.get(a, PEN_EXT),
            q_start=(q_start or {}).get(a), park=parks[a],
            retreat=(retreats or {}).get(a),
            taxi_stretch=float((stretch or {}).get(a, 0.0)), verbose=verbose)
    return out


def _better(new, old, eps=1e-9):
    """Is `new` a schedule worth keeping over `old`? -> bool.

    THE OBJECTIVE HIERARCHY, WRITTEN DOWN ONCE.  Makespan first — that is what
    the piece is judged on — and total pause as the tie-break, because two
    schedules that finish at the same instant are not equally good: the one that
    leaves the fleet standing still for less of it is closer to a run that could
    survive being re-timed with real accelerations.  Safety is not in here at
    all, because it is not traded: both candidates came out of the same DP
    against the same margin, or they did not come out at all.
    """
    a, b = float(new["duration"]), float(old["duration"])
    if a < b - eps:
        return True
    if a > b + eps:
        return False
    return float(new["pause_total"]) <= float(old["pause_total"]) + eps


def _capsules(progs, pens, dt, fleet=None):
    """Sample every frozen timeline onto the conducting clock. -> (samp, paths)."""
    samp = {a: writing.uniform_samples(p, dt) for a, p in progs.items()}
    return samp, coordination.arm_paths({a: s["q"] for a, s in samp.items()}, dt,
                                        pens=pens, fleet=fleet)


def _conduct(progs, pens, dt, safety, calib, sweep, verbose, specs=None,
             search_max_n=coordination.PRIORITY_SEARCH_MAX):
    """Sample, build capsule paths, conduct. -> (sch, paths, samp).

    `search_max_n` goes straight through to `coordination.coordinate`: the
    number of MOVING arms below which every priority order is enumerated.  It
    is exposed here because it is the one conductor knob whose cost is
    super-exponential in the fleet size — the prefix walk is up to 1956 DP
    solves at six moving arms against 64 at four — and six arms drawing at once
    is now a rig somebody actually runs, not a hypothetical.
    """
    samp, paths = _capsules(progs, pens, dt, fleet=specs)
    sch = coordination.coordinate(paths, safety=safety, calib=calib, sweep=sweep,
                                  search_max_n=search_max_n, verbose=verbose)
    return sch, paths, samp


class Unconductable(RuntimeError):
    """A refusal with the evidence attached. -> .transits, .draws, .blocks.

    `transits[arm]` is a list of `(i, j)` bag-index edges — "this arm's pen-up
    move from segment i to segment j passes through a place no schedule can put
    it" — in the form `allocate.resequence(forbid=...)` takes, so a caller can
    hand the refusal back to the sequencer and ask for a different tour.
    `draws[arm]` is the same thing for INK, and nothing but a different
    allocation or placement will fix one of those.
    """

    def __init__(self, msg, transits=None, draws=None, blocks=None):
        super().__init__(msg)
        self.transits = transits or {}
        self.draws = draws or {}
        self.blocks = blocks or {}


def unrunnable(progs, blocks, dt, orders=None):
    """Blocked progress indices -> the tour edges and the ink they belong to.

    A progress index is a moment in a frozen timeline, and `arm_program` already
    recorded which block of the programme every moment belongs to.  Walking that
    record turns "index 1112 is impossible" into either an edge of a tour (which
    the sequencer can be told to avoid) or a stroke (which it cannot).
    """
    transits, draws = {}, {}
    for (a, b), rows in blocks.items():
        p = progs.get(a)
        if p is None or not p["phases"]:
            continue
        order = list((orders or {}).get(a) or range(len(p["phases"])))
        first = p["phases"][0]["t0"]
        for r in rows:
            t = float(r) * dt
            if t < first:                    # the entry taxi, before any ink
                if order:
                    transits.setdefault(a, set()).add((None, int(order[0])))
                continue
            hit = next((ph for ph in p["phases"] if ph["t0"] <= t <= ph["t1"]),
                       None)
            if hit is None:
                continue
            k = int(hit["seg"])
            if hit["kind"] == "stroke":
                draws.setdefault(a, set()).add(k)
            elif k + 1 < len(order):         # a transit BETWEEN two segments
                transits.setdefault(a, set()).add((int(order[k]),
                                                   int(order[k + 1])))
            else:                            # the exit lift or the retreat
                draws.setdefault(a, set()).add(k)
    return ({a: sorted(v, key=edge_key) for a, v in transits.items()},
            {a: sorted(v) for a, v in draws.items()})


def edge_key(e):
    """Sort key for a tour edge, INCLUDING the one that starts from nowhere.

    `unrunnable` emits `(None, j)` for a block hit during the entry taxi — the
    move from wherever the arm is standing to its first segment, which has no
    source segment.  `sorted` on a mix of `(None, j)` and `(i, j)` raises
    `TypeError: '<' not supported between instances of 'int' and 'NoneType'`,
    and it raises it INSIDE THE REFUSAL PATH: a conductable-looking run dies
    with a type error instead of reporting the refusal it had already
    diagnosed.  `sequence_arm` already understands `i is None` (it forbids the
    START row of the cost matrix); only the sort did not.
    """
    return tuple(-1 if x is None else int(x) for x in e)


def _refusal(exc, progs, dt, orders, specs=None):
    """Turn `coordinate`'s RuntimeError into one a caller can act on."""
    free = getattr(exc, "free", None)
    paths = getattr(exc, "paths", None)
    if paths is None:
        return exc
    blocks = coordination.hard_blocks(paths, getattr(exc, "margin", 0.08),
                                      getattr(exc, "sweep", coordination.SWEEP_K),
                                      free)
    tr, dr = unrunnable(progs, blocks, dt, orders)
    note = ""
    if tr:
        note = ("; the impossible indices are pen-up transits: "
                + ", ".join(f"arm {a} " + " ".join(
                    f"{'start' if i is None else i}->{j}" for i, j in v)
                            for a, v in tr.items())
                + " — re-sequencing without those edges may run")
    elif dr:
        note = ("; the impossible indices are INK: "
                + ", ".join(f"arm {a} segment(s) {v}" for a, v in dr.items())
                + " — no order fixes that, only a different allocation")
    else:
        # No index is impossible on its own, so the refusal is about the pose
        # the arm STOPS in or about the ordering.  Say which, rather than
        # leaving the caller with a verdict and no evidence.
        margin = getattr(exc, "margin", 0.08)
        sweep = getattr(exc, "sweep", coordination.SWEEP_K)
        stuck = []
        for a, p in progs.items():
            if p["duration"] <= 0.0 or "q_end" not in p:
                continue
            bad = frozen_interference(p["q_end"], a, p.get("pen", PEN_EXT),
                                      {b: pb for b, pb in paths.items()
                                       if b != a}, {}, margin, sweep,
                                      spec=specs.get(a) if specs else None)
            if bad:
                stuck.append(f"arm {a} cannot stop clear of "
                             + ",".join(f"{b} ({1000 * v:.0f} mm)"
                                        for b, v in sorted(bad.items())))
        note = ("; no single progress index is impossible, so this is an "
                + ("FROZEN POSE problem: " + "; ".join(stuck)
                   if stuck else "ORDERING deadlock with no geometric cause "
                   "this module can name"))
    return Unconductable(str(exc) + note, tr, dr, blocks)


def _blocking_freezes(progs, paths, pens, margin, sweep, h_inv, specs=None):
    """Frozen poses that sit in ANOTHER arm's tube anywhere. -> {arm: {b: gap}}.

    The conservative reading of "remaining tube", and the only one available
    before a schedule exists: an arm that has been delayed is BEHIND where it
    would nominally be, so nothing can be assumed to have been flown through
    yet.  This is what the infeasibility rescue triggers on; once there IS a
    schedule, `coordination.rest_delays` gives the exact, much narrower answer.
    """
    out = {}
    for a, p in progs.items():
        if p["duration"] <= 0.0 or "q_end" not in p:
            continue
        hit = frozen_interference(p["q_end"], a, pens.get(a, PEN_EXT), paths,
                                  {b: 0 for b in paths}, margin, sweep, h_inv,
                                  spec=specs.get(a) if specs else None)
        if hit:
            out[a] = hit
    return out


def conduct(segs_by_arm, pens, dt, q_start=None, policy=POLICY_FREEZE,
            retreat=True, jit=True, jit_frac=JIT_FRAC, specs=None,
            draw_speed=writing.DRAW_SPEED_FLEET,
            transit_speed=writing.TRANSIT_SPEED, qd_frac=writing.QD_FRAC,
            h_inv=H_INV_DEFAULT, safety=coordination.SAFETY_M,
            calib=coordination.CALIB_M, sweep=coordination.SWEEP_K,
            search_max_n=coordination.PRIORITY_SEARCH_MAX,
            on_programs=None, orders=None, verbose=True):
    """Freeze the timelines, conduct them, and spend the idle time better.

    -> dict(progs, samp, paths, sch, rest, retreats, taxi, passes, q_end, policy)

    Up to three conducts, and every one of them is the same exact DP over the
    same images — the policy chooses PATHS, it never chooses a schedule:

      0. freeze (or go home) and conduct.  This is the baseline, and it is kept
         if neither of the other two beats it.
      1. JIT: spend `jit_frac` of the slack pass 0 measured on stretching each
         arm's pen-up blocks, and re-conduct.  Kept only if the makespan does
         not get worse — that is the "drawing has priority over taxiing" rule
         with the fleet's own clock as the judge.
      2. RETREAT: any arm the rest-suffix rule actually cost seconds gets the
         smallest certified pose that leaves everyone's remaining tube, and the
         fleet is re-conducted with it.  Kept on the same terms.

    `passes` records every one of them — what it was worth and whether it was
    kept — because a policy that cannot be shown to have paid for itself is a
    policy nobody can take back out.
    """
    specs = FLEET if specs is None else specs
    prog_kw = dict(draw_speed=draw_speed, transit_speed=transit_speed,
                   qd_frac=qd_frac, h_inv=h_inv)
    parks = {a: str(policy) for a in segs_by_arm}
    progs = _programs(specs, segs_by_arm, pens, q_start, parks, None, None,
                      only=None, **prog_kw)
    if on_programs is not None:       # the caller's cross-checks, before the
        on_programs(progs)            # first (expensive) conduct rather than after
    passes, taxi, retreats = [], {}, {}
    margin = float(safety + calib)
    try:
        sch, paths, samp = _conduct(progs, pens, dt, safety, calib, sweep,
                                    verbose, specs, search_max_n)
    except RuntimeError as exc:
        # FREEZING CAN MAKE A SCHEDULE IMPOSSIBLE, WHERE GOING HOME ONLY MADE IT
        # SLOW.  An arm parked on top of the ink another arm still has to draw
        # is not a delay, it is a wall, and no amount of waiting gets past it.
        # This is the one place a retreat is planned WITHOUT a schedule to
        # measure — against every other arm's whole path, because a conductor
        # that has refused has told us nothing about who goes where when.
        if not retreat or policy == POLICY_HOME:
            raise _refusal(exc, progs, dt, orders, specs)
        samp, paths = _capsules(progs, pens, dt, fleet=specs)
        blocked = _blocking_freezes(progs, paths, pens, margin, sweep, h_inv, specs)
        if verbose:
            print(f"  conductor refused ({exc}); {len(blocked)} frozen pose(s) "
                  "sit in another arm's tube — offering each a retreat")
        want, sent_home = {}, []
        for a in blocked:
            got = plan_retreat(specs[a], progs[a]["q_end"], a,
                               pens.get(a, PEN_EXT), paths,
                               {b: 0 for b in paths}, margin, sweep, h_inv)
            if got is not None:
                want[a] = got
            elif parks[a] != POLICY_HOME:
                # NOTHING NEARBY IS OUT OF THE WAY, so this one arm takes
                # conductor v1's answer and goes home.  The ready pose is a
                # known-good place to stand — it is where every arm stood for
                # the whole of the last release — and one arm paying for it is
                # a great deal better than six.
                parks[a], _ = POLICY_HOME, sent_home.append(a)
        if not want and not sent_home:
            raise _refusal(exc, progs, dt, orders, specs)
        progs = _programs(specs, segs_by_arm, pens, q_start, parks, None,
                          {a: g["q"] for a, g in want.items()},
                          only=set(want) | set(sent_home), prev=progs, **prog_kw)
        try:
            sch, paths, samp = _conduct(progs, pens, dt, safety, calib, sweep,
                                        verbose, specs, search_max_n)
        except RuntimeError as exc2:
            raise _refusal(exc2, progs, dt, orders, specs)
        retreats = dict(want)
        passes.append(dict(name="rescue", makespan=float(sch["duration"]),
                           pause=float(sch["pause_total"]), kept=True,
                           arms={int(a): dict(tag=g["tag"], dq=g["dq"],
                                              clearance=g["clearance"])
                                 for a, g in want.items()}))
        if verbose:
            for a, g in sorted(want.items()):
                print(f"  rescue retreat: arm {a} {g['tag']} ({g['dq']:.3f} rad)")
            for a in sorted(sent_home):
                print(f"  rescue: arm {a} has nowhere clear to freeze near its "
                      "last stroke — it goes home, alone")
    if not passes:                       # no rescue was needed
        passes.append(dict(name="baseline", policy=str(policy),
                           makespan=float(sch["duration"]),
                           pause=float(sch["pause_total"]), kept=True))
    best = dict(progs=progs, sch=sch, paths=paths, samp=samp)

    # ---- 1. just-in-time pre-positioning ---------------------------------
    if jit and float(jit_frac) > 0.0:
        M = float(sch["duration"])
        want = {a: max(0.0, float(jit_frac) * (M - sch["finish"].get(a, 0.0)))
                for a in progs if progs[a]["duration"] > 0.0}
        want = {a: s for a, s in want.items() if s > dt}
        if want:
            pj = _programs(specs, segs_by_arm, pens, q_start, parks, want,
                           {a: g["q"] for a, g in retreats.items()} or None,
                           only=set(want), prev=progs, **prog_kw)
            try:
                sj, paj, saj = _conduct(pj, pens, dt, safety, calib, sweep,
                                        verbose, specs, search_max_n)
            except RuntimeError:          # a slower taxi is never worth a refusal
                sj = dict(duration=float("inf"), pause_total=float("inf"))
            keep = _better(sj, sch)
            passes.append(dict(name="jit", makespan=float(sj["duration"]),
                               pause=float(sj["pause_total"]), kept=bool(keep),
                               stretch={int(a): float(s) for a, s in want.items()},
                               frac=float(jit_frac)))
            if keep:
                taxi = {a: float(s) for a, s in want.items()}
                best = dict(progs=pj, sch=sj, paths=paj, samp=saj)
            if verbose:
                print(f"  JIT taxi: {len(want)} arm(s) stretched by "
                      + ", ".join(f"{a}:+{s:.1f}s" for a, s in sorted(want.items()))
                      + f" -> {sj['duration']:.1f} s vs {sch['duration']:.1f} s "
                      + ("(kept)" if keep else "(thrown away)"))

    sch, progs, paths, samp = (best["sch"], best["progs"], best["paths"],
                               best["samp"])
    rest = coordination.rest_delays(paths, sch)

    # ---- 2. minimal retreat, for the frozen poses that actually cost -------
    if retreat and policy != POLICY_HOME:
        want = {}
        for a, r in rest.items():
            if r["delay_s"] <= RETREAT_MIN_DELAY or not r["blockers"]:
                continue
            if a in retreats or parks.get(a) == POLICY_HOME:
                continue           # already moved once, or not freezing at all
            m_free = int(round(r["free_arrive_s"] / dt))
            tails = {b: int(np.asarray(sch["progress"][b])[
                min(m_free, sch["M"] - 1)]) for b in paths if b != a}
            got = plan_retreat(specs[a], progs[a]["q_end"], a,
                               pens.get(a, PEN_EXT), paths, tails, sch["margin"],
                               sweep, h_inv)
            if got is not None:
                want[a] = got
        if want:
            merged = dict(retreats)
            merged.update(want)
            pr = _programs(specs, segs_by_arm, pens, q_start, parks, taxi or None,
                           {a: g["q"] for a, g in merged.items()},
                           only=set(want), prev=progs, **prog_kw)
            try:
                sr, par, sar = _conduct(pr, pens, dt, safety, calib, sweep,
                                        verbose, specs, search_max_n)
            except RuntimeError:
                sr = dict(duration=float("inf"), pause_total=float("inf"))
            was = float(sch["duration"])
            keep = _better(sr, sch)
            passes.append(dict(name="retreat", makespan=float(sr["duration"]),
                               pause=float(sr["pause_total"]), kept=bool(keep),
                               arms={int(a): dict(tag=g["tag"], dq=g["dq"],
                                                  clearance=g["clearance"])
                                     for a, g in want.items()}))
            if keep:
                retreats = merged
                sch, progs, paths, samp = sr, pr, par, sar
                rest = coordination.rest_delays(paths, sch)
            if verbose:
                for a, g in sorted(want.items()):
                    print(f"  retreat: arm {a} {g['tag']} ({g['dq']:.3f} rad) "
                          f"-> {1000 * g['clearance']:.0f} mm clear of every "
                          f"remaining tube")
                print(f"  retreat pass -> {sr['duration']:.1f} s vs {was:.1f} s "
                      + ("(kept)" if keep else "(thrown away)"))

    return dict(progs=progs, samp=samp, paths=paths, sch=sch, rest=rest,
                retreats={int(a): dict(tag=g["tag"], dq=float(g["dq"]),
                                       clearance=float(g["clearance"]),
                                       before=g["before"])
                          for a, g in retreats.items()},
                taxi={int(a): float(s) for a, s in taxi.items()},
                passes=passes, policy=str(policy),
                parks={int(a): v for a, v in parks.items()},
                sent_home=sorted(int(a) for a, v in parks.items()
                                 if v == POLICY_HOME and str(policy) != POLICY_HOME),
                q_end={a: np.asarray(p["q_end"], float) if "q_end" in p
                       else np.asarray(p["q"][-1], float)
                       for a, p in progs.items()})


def report(res):
    """Terse idle-policy report -> list of printable lines."""
    out = [f"idle policy: {res['policy']}"
           + (f", {len(res['retreats'])} retreat(s)" if res["retreats"] else "")
           + (f", {len(res['taxi'])} arm(s) taxiing slowly" if res["taxi"] else "")
           + (f", arm(s) {res['sent_home']} sent home (nowhere clear to freeze)"
              if res.get("sent_home") else "")]
    for p in res["passes"]:
        out.append(f"  pass {p['name']:>8}: makespan {p['makespan']:6.1f} s, "
                   f"pause {p['pause']:6.1f} s  "
                   f"{'KEPT' if p['kept'] else 'discarded'}")
    for a, r in sorted(res["rest"].items()):
        if r["delay_s"] > 1e-9:
            out.append(f"  arm {a:>2} still waits {r['delay_s']:.1f} s for "
                       f"permission to stop, on "
                       + ", ".join(f"arm {b} ({v:.1f} s)"
                                   for b, v in sorted(r["blockers"].items())))
    return out
