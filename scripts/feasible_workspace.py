"""THE FEASIBLE CANVAS, SOLO: what one arm at a time can actually draw.

A canvas cell is FEASIBLE when SOME arm can draw it while the other five stand
at their certified parks.  Three layers, all of them the pipeline's own — this
script composes existing seams and invents no gate of its own:

  1. DRAW POSE   a certified strict-GO pose from the banded atlas
                 (`atlas.strict_go`, gated at `rig_final.STATIC_MARGIN`
                 against every neighbour's steel and base column).  The atlas
                 carries `atlas.model_signature`; a stale one is refused here
                 rather than trusted.

  2. HOVER       a certified lift over that cell, `writing.lifted_or_lower` —
                 the v3 full-fiber solver (8 tool yaws x the whole q7 grid x
                 every branch x the 5-height ladder), NOT the phi = 0 pin.  Its
                 last resort is "do not lift at all" (height 0), which is a
                 refusal dressed as an answer: this script counts z > 0 only.

  3. PEN-UP      `writing.enter_beats(park -> hover -> draw pose)`, which is
                 two `paper.route` calls: park to hover on the flying floor,
                 hover to ink on the contact band.  `paper.route` walks its
                 whole ladder (direct, lift, descend, retract-then-go, mid-hop,
                 cartesian traverse, fold-through-home, skirt) against the
                 static metal at `STATIC_PLAN_MARGIN`.
                 ...and then the FIVE PARKED PARTNERS, which the router does
                 not know about: `allocate.ParkProbe` at the conductor's own
                 SAFETY_M + CALIB_M = 0.08 m, over the route densified to the
                 router's own `paper.SAMPLES` per leg.  This is where the
                 pipeline puts a parked arm (commit 8c1e7b1) and this script
                 puts it in the same place.

NO PROXY.  The first cut of this budgeted a hover-plane roadmap because
per-cell routing looked unaffordable; measured, it is a couple of seconds a
cell and the whole fleet is 23 376 certified cells, so every cell is routed for
real and the map has no approximation in it to validate.
`scripts/feasible_workspace_validate.py` recomputes a stratified sample in a
cold process and reports what the straight-line proxy WOULD have got wrong,
as a diagnostic rather than as a substitute.

`--every N` sweeps every Nth certified cell for a pilot; the full map needs
`--every 1` (the default), because a skipped cell is indistinguishable from a
cell no arm can draw.

    ARIS_RIG=proposed ARIS_TOOL=lateral python3 scripts/feasible_workspace.py

Writes out/feasible_workspace.{png,json}, _raw.npz, _map.npz, and
checkpoints _ckpt.npz every ten chunks so a long sweep loses nothing.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# the rig and the tool are read at import time, so they are set before the
# package is touched — a caller who forgot the env vars gets the rig they asked
# for rather than the default one silently
os.environ.setdefault("ARIS_RIG", "proposed")
os.environ.setdefault("ARIS_TOOL", "lateral")

from aris_sixarm import allocate, atlas, layout, paper, writing   # noqa: E402
from aris_sixarm import envelope, frozen
from aris_sixarm import transit                                   # noqa: E402
from aris_sixarm import frames                                    # noqa: E402

ATLAS_DIR = ROOT / "out" / "atlas_proposed_h0940_gated"
OUT = ROOT / "out"
GRID = 0.02
SHEET = (1.8034, 3.63064)

# cause codes, in the order a cell is decided
FEASIBLE = 0
NO_DRAW = 1        # no arm has a certified drawing pose here
NO_HOVER = 2       # some arm can draw it, none can lift off it
NO_ROUTE = 3       # some arm can draw and lift, none can fly here

CAUSE_NAME = {FEASIBLE: "feasible", NO_DRAW: "no draw pose",
              NO_HOVER: "draw ok, no hover", NO_ROUTE: "hover ok, unreachable"}

# THE NEIGHBOUR MODEL, and both halves are ON by default since 2026-09-09.
# CYLM  the body column measured as its own cylinder instead of its bounding
#       box (aris_sixarm/envelope.py) -- the AABB padded a 128 mm cylinder
#       into a 449 mm box and refused pen-ups under a base against 160 mm of
#       nothing.  Strictly tighter and still a valid outer envelope.
# FROZEN  partners modelled by their ACTUAL park capsules (aris_sixarm/
#       frozen.py).  A solo feasibility map assumes every other arm is parked
#       BY CONSTRUCTION, so this is the map's own premise made explicit; the
#       dependency is recorded in the JSON.  `--legacy-bands` turns both off
#       and reproduces the pre-2026-09-09 model exactly.
CYLM = True
FROZEN = True
RRT_CELL_PLANS = 3      # C-space plans one (arm, cell) may spend (see `_cell`)


def axes(sheet=SHEET, grid=GRID):
    """The atlas's own cell centres, so a row of it indexes straight in."""
    xs = np.arange(0.0, sheet[0] + 1e-9, grid)
    ys = np.arange(0.0, sheet[1] + 1e-9, grid)
    return xs, ys


def _ij(x, y, xs, ys):
    return (int(round(float(y) / GRID)), int(round(float(x) / GRID)))


# ---------------------------------------------------------------------------
# one arm's cells, in a worker
# ---------------------------------------------------------------------------
_W = {}
SHIPPED_PITCH = 0.61


# THE PARK OVERRIDE IS A MODULE GLOBAL BECAUSE THE WORKERS FORK.
# `--parks` has to reach `rig`, and `rig` is called inside every pool worker's
# `_init` — in the sweep AND in the rescue ladder.  Threading it through three
# signatures and two `initargs` tuples is how the FIRST version of this went
# wrong: main() replaced its own `fl`/`parks`, printed the searched depots, and
# every worker went on deriving `PARK_GRID_PROPOSED`'s.  The map that came out
# said one thing in its header and routed against another.  A module global set
# once before any pool is created cannot be half-applied, and it is the idiom
# this file already uses for `paper.RRT_SAFE` and `transit.TIME_BUDGET`.
_PARK_OVERRIDE = None


def set_park_override(parks):
    """Depots every later `rig()` must use, or None for the derived recipe."""
    global _PARK_OVERRIDE
    _PARK_OVERRIDE = (None if parks is None else
                      {int(k): np.asarray(v, float) for k, v in parks.items()})


def rig(pitch=None, h=None, calib=None):
    """The fleet and its parks. -> (fleet, parks, h, pitch).

    At the SHIPPED pitch this is the committed pair — `layout.FLEET_PROPOSED`
    and the baked `Q_PARK_PROPOSED` literals — and nothing is re-solved, so the
    headline number is measured against the rig in the repository and not
    against a re-derivation of it.

    At any other pitch the layout is `paired_grid` at that spacing and the
    parks are re-derived by `layout.certified_park_poses` from the SAME
    (radius, hover, bearing) recipe.  That recipe was searched at 0.61 and is
    not re-optimised here, so a comparison run answers "the shipped park recipe
    at a wider pitch", not "the best rig at a wider pitch".

    `calib` REPLACES the unsurveyed-base allowance in the OBSTACLE BOXES, and
    it is here so the commissioning survey can be priced without editing a
    constant.  `mounts.MOUNTS.calib` = 0.03 m is the allowance for base
    positions nobody has measured yet; every neighbour's body column is carried
    as boxes one `calib` fatter than the metal (`mounts.MountModel.column_bands`
    and `column_r`).  A run with `calib=0.0` is asking "what would the canvas be
    if the six bases were surveyed?" — and it is a PROJECTION, built here on a
    local model, never written back into the package and never the shipped
    number.  The park poses are the shipped literals either way: a survey moves
    what the arms may fly through, not where they were told to wait.
    """
    h = layout.LAYOUT_PROPOSED["h"] if h is None else float(h)
    pitch = SHIPPED_PITCH if pitch is None else float(pitch)
    shipped = (abs(pitch - SHIPPED_PITCH) < 1e-9 and abs(h - 0.940) < 1e-9
               and _PARK_OVERRIDE is None)
    if calib is None:
        if shipped:
            return layout.FLEET_PROPOSED, layout.Q_PARK_PROPOSED, h, pitch
        lay = layout.paired_grid(spacing=pitch, rows=3, h=h)
        if _PARK_OVERRIDE is not None:
            parks = _PARK_OVERRIDE
        else:
            parks = layout.certified_park_poses(layout.build_fleet(lay),
                                                layout.PARK_GRID_PROPOSED)
        return layout.build_fleet(lay, q_park=parks), parks, h, pitch
    from aris_sixarm import mounts
    model = mounts.MOUNTS.scaled(calib=float(calib))
    lay = (layout.LAYOUT_PROPOSED if shipped
           else layout.paired_grid(spacing=pitch, rows=3, h=h))
    parks = (_PARK_OVERRIDE if _PARK_OVERRIDE is not None else
             layout.Q_PARK_PROPOSED if shipped else
             layout.certified_park_poses(layout.build_fleet(lay),
                                         layout.PARK_GRID_PROPOSED))
    return (layout.build_fleet(lay, mount_model=model, q_park=parks),
            parks, h, pitch)


def park_hovers(fleet, parks, h):
    """Where each parked pen sits on the canvas. -> {arm: (x, y)}.

    Baked as `layout.PARK_HOVER_PROPOSED` for the shipped rig; derived here for
    a comparison rig, by the same forward kinematics that put it there."""
    out = {}
    for a, q in parks.items():
        xy = paper.tip_xy(np.asarray(q, float), fleet[a],
                          pen_ext=fleet[a].pen, h_inv=h)
        out[int(a)] = (round(float(xy[0]), 3), round(float(xy[1]), 3))
    return out


def _init(atlas_dir, h, redundant=True, pitch=None, fiber=True, lean=0.0,
          tries=12, rrt=0.0, rrt_nodes=600, attempts=1, plans=None,
          shortcut=None, calib=None, frozen_partners=True,
          cyl_model=True):
    """Per-worker state: the fleet, the parks, the probe.  Built once."""
    writing.HOVER_LEAN_MAX_DEG = float(lean)
    fl, parks, h, _ = rig(pitch, h, calib)
    pens = {a: fl[a].pen for a in fl}
    _W["fleet"] = fl
    _W["parks"] = parks
    _W["pens"] = pens
    _W["h"] = h
    _W["calib"] = calib
    # THE INTER-ARM SHARE OF THE SAME ALLOWANCE.  `SAFETY_M + CALIB_M` is the
    # 0.08 m the conductor holds every pair to, and 0.03 of it is the same
    # unsurveyed-base term the boxes carry.  A projection that took it out of
    # the metal and left it in the pair margin would only have done half the
    # sum, so this does both — and, being an argument, neither is a package
    # constant that some other run could inherit.
    if calib is None:
        _W["margin"] = None
    else:
        from aris_sixarm import coordination
        _W["margin"] = float(coordination.SAFETY_M + float(calib))
    _W["probe"] = allocate.ParkProbe(parks, fl, pens, h_inv=h,
                                     margin=_W["margin"])
    # THE POSE-AWARE NEIGHBOUR MODEL, off unless asked for.  The solo map's own
    # assumption is that every OTHER arm is parked, and `frozen` makes the
    # obstacle set say so: a partner's pose-invariant band is replaced by its
    # actual capsules at its actual park.  See aris_sixarm/frozen.py for what
    # this costs in certification.
    _W["frozen"] = bool(frozen_partners)
    if _W["frozen"]:
        frozen.freeze(parks, fl, pens, h)
    else:
        frozen.thaw()
    _W["cyl"] = bool(cyl_model)
    if _W["cyl"]:
        envelope.install(fl, h)
    else:
        envelope.uninstall()
    _W["probes"] = {}
    _W["atlas_dir"] = atlas_dir
    _W["redundant"] = bool(redundant)
    _W["fiber"] = bool(fiber)
    _W["tries"] = int(tries)
    _W["plans"] = RRT_CELL_PLANS if plans is None else int(plans)
    # THE C-SPACE TIER, WITH A BUDGET THAT IS A POLICY AND NOT A DEFAULT.
    #
    # This map routes every one of 23 376 certified cells and retries each
    # refusal over a dozen more hovers on the fiber, so the number of pen-up
    # questions asked here is two orders of magnitude above a logo run's.  The
    # planner's own default (2.5 s x 2 attempts) is sized for a crossing a tour
    # cannot do without; a CELL is not that, and a budget that big would put
    # the sweep past a day.
    #
    # So the map gets `--rrt SECONDS` per plan, ONE attempt, and a smaller
    # tree.  What that means for the number has to be said plainly: a cell this
    # reports as unreachable is a cell no ladder shape and no RRT-within-budget
    # could fly to.  It is a LOWER bound on feasibility, tighter than the one
    # before it and still not a proof of impossibility — which is exactly the
    # claim the layer above it (the hover ladder, the fiber retry) already
    # makes about itself.
    # ...AND THE BUDGET IS NOW REAL.  Until 2026-08-27 `transit.plan` bound
    # these three in its SIGNATURE, so every one of these assignments was
    # inert and the v11 map ran the module defaults (2.5 s x 2 attempts x 900
    # nodes) while its log said 0.80 s x 1 x 500.  `transit.plan` resolves them
    # at call time now, which is what makes the escalation ladder below an
    # escalation and not a wish.
    _W["rrt"] = float(rrt)
    paper.RRT_SAFE = float(rrt) > 0.0
    if paper.RRT_SAFE:
        transit.TIME_BUDGET = float(rrt)
        transit.ATTEMPTS = int(attempts)
        transit.MAX_NODES = int(rrt_nodes)
        # Smoothing is what a TOUR pays for and this map does not buy tours: it
        # asks whether a cell can be flown to at all.  Two rounds of shortcut
        # keep the path from being absurd and cost a tenth of what the default
        # spends making it short.
        transit.SHORTCUT_TIME = (min(0.20, 0.25 * float(rrt))
                                 if shortcut is None else float(shortcut))
        transit.SHORTCUT_ROUNDS = 24


def _park_sig(parks):
    """A hashable identity for one parked SET. -> tuple.

    IT GOES IN `paper`'s MEMO KEY, and that is not decoration.  `paper.route`
    files its answer under a key that carries `RRT_PROBE[2]`, so two questions
    that differ only in WHERE THE OTHER FIVE ARMS ARE STANDING would otherwise
    collide — the region-aware rung would ask about a fleet that had moved and
    be handed the answer for the fleet that had not.
    """
    return tuple((int(a), np.round(np.asarray(q, float), 6).tobytes())
                 for a, q in sorted(parks.items()))


def _probe_for(parks):
    """The `ParkProbe` for one parked set, built once per worker. -> probe."""
    sig = _park_sig(parks)
    got = _W["probes"].get(sig)
    if got is None:
        got = (allocate.ParkProbe(parks, _W["fleet"], _W["pens"],
                                  h_inv=_W["h"], margin=_W.get("margin")), sig)
        _W["probes"][sig] = got
    return got


def _park_probe_hook(arm, probe=None, sig=None):
    """Point `paper.RRT_PROBE` at the five arms standing behind THIS one.

    The ladder does not know about parked partners — `allocate.ParkProbe`
    screens a route after the fact, three stages later — and a route that is
    going to be refused there is a route the search should not have spent its
    budget finding.  The planner can be told, so it is.

    The key is what goes into `paper`'s memo: the arm decides which five
    partners are in the room, and `sig` says WHICH five poses they are in — the
    parks are no longer fixed for a run (see `_park_sig`).
    """
    probe = _W["probe"] if probe is None else probe
    if not probe or not probe.partners(int(arm)):
        paper.RRT_PROBE = None
        return
    paper.RRT_PROBE = (lambda qs, sweep: probe.clearance(int(arm), qs, sweep),
                       probe.margin, ("park", int(arm),
                                      round(float(probe.margin), 9), sig))


def _dense(steps, q0, n=None):
    """A beat's waypoints -> the router's own sampling density. -> (M, 7).

    `writing._beat` returns one row per HOP, and a hop is a metre-long joint
    move.  `ParkProbe.clearance` charges half the worst per-sample motion as
    its 1-Lipschitz residual, so handing it the hops themselves would refuse
    every route for being long.  Sampled at `paper.SAMPLES` per hop it is the
    same density `paper.route` certified the route at.
    """
    n = paper.SAMPLES if n is None else int(n)
    qs = [np.asarray(q0, float).reshape(7)] + [np.asarray(s[1], float).reshape(7)
                                               for s in steps]
    out = [qs[0][None, :]]
    for a, b in zip(qs[:-1], qs[1:]):
        f = np.linspace(0.0, 1.0, n)[1:, None]
        out.append(a[None, :] * (1.0 - f) + b[None, :] * f)
    return np.vstack(out)


def _certified_hovers(spec, q_draw, xy, h, heights=None):
    """EVERY certified hover over this cell, ladder order. -> [(q, z)].

    `writing.lifted_or_lower` stops at the first rung that certifies, because
    a timeline wants one hover and the highest it can hold.  A FEASIBILITY map
    wants the redundancy: a cell whose 6 cm hover cannot be flown to may have a
    3 cm one that can, and refusing it would blame the ladder's preference on
    the canvas.  Same solver, same gate, same rungs, same floor-clamp fallback
    (`writing.lifted_or_lower`'s second ladder) — it just does not stop.
    """
    heights = writing.HOVER_LADDER if heights is None else heights
    gate = writing.static_gate(spec, spec.pen, h)
    out, seen = [], set()

    def ladder(ok):
        for z in heights:
            q = writing.hover_solve(spec, q_draw, xy, z=z, h_inv=h,
                                    pen_ext=spec.pen, ok=ok)
            if q is None:
                continue
            k = np.round(q, 6).tobytes()
            if k in seen:
                continue
            seen.add(k)
            out.append((np.asarray(q, float), float(z)))

    ladder(gate)
    if not out:
        fl = float(paper.chain_static(np.asarray(q_draw, float).reshape(1, 7),
                                      spec, spec.pen, h)[0])
        if fl < paper.FRAME_FLOOR:
            ladder(writing.static_gate(spec, spec.pen, h, floor=fl))
    return out


def _fiber_hovers(spec, q_draw, xy, h, tries=None):
    """The rest of the fiber, ladder order, nearest first. -> [(q, z)].

    `_certified_hovers` walks the HEIGHTS; this walks the whole fiber at each
    of them — 8 tool yaws x the entire q7 grid x every branch, and then the
    leans the run's cone allows — which is what `writing.lifted_or_lower`'s
    depot-aware tier does when a span end is about to cost ink.

    IT IS THE SAME QUESTION THE MAP IS ASKING.  A cell whose derived hover
    cannot be flown to is not a cell the arm cannot reach; it is a cell whose
    FIRST hover cannot be flown to, and the pipeline does not accept that answer
    either.  Measured over 1 127 certified cells (`out/guard_pocket.py`), 51.7 %
    of the span ends the router calls unreachable hold a pose on their own fiber
    that it can fly to — so a map that did not look here would blame the canvas
    for a search that stopped early.
    """
    tries = writing.HOVER_DEPOT_TRIES if tries is None else int(tries)
    gate = writing.static_gate(spec, spec.pen, h)
    out = []
    for lean in [None] + writing._lean_rungs():
        for z in writing.HOVER_LADDER:
            for q in writing.hover_fiber(spec, q_draw, xy, z, gate, h,
                                         spec.pen, lean):
                out.append((np.asarray(q, float), float(z)))
                if len(out) >= tries:
                    return out
    return out


def _cell(arm, row, qcol, redundant=True):
    """The three layers for one (arm, cell). -> (code, z_hover, park_clear).

    `code` is the FIRST layer that refused, as a cause code; FEASIBLE when all
    three hold.  The draw pose is layer 1 by construction — this is only ever
    called on a strict-GO row.  A cell passes if ANY certified hover on the
    ladder — and then anywhere on the FIBER — can be flown to, which is the
    redundancy the brief asks for and the search the pipeline itself makes.
    """
    fl, parks, h = _W["fleet"], _W["parks"], _W["h"]
    probe = _W["probe"]
    if _W.get("frozen"):
        frozen.observe(arm)          # never check the mover against itself
    if _W.get("rrt", 0.0) > 0.0:
        _park_probe_hook(arm)
    spec = fl[arm]
    x, y = float(row[0]), float(row[1])
    q_draw = np.asarray(row[qcol:qcol + 7], float)
    q_park = np.asarray(parks[arm], float)

    if redundant:
        hovers = _certified_hovers(spec, q_draw, (x, y), h)
    else:
        q, z = writing.lifted_or_lower(spec, q_draw, (x, y), h_inv=h,
                                       pen_ext=spec.pen)
        hovers = [(q, z)] if z > 0.0 else []
    if not hovers:
        return NO_HOVER, 0.0, float("nan")

    # THE PER-CELL BUDGET, COUNTED IN PLANS AND NOT IN SECONDS.
    #
    # A cell that cannot be flown to walks its whole hover ladder and then a
    # dozen more poses off the fiber, and each of those is two `paper.route`
    # calls that can reach the planner — so an unbounded tier would spend half
    # a minute on one refused cell and days on the map.  The cap is a COUNT
    # because the map has to reproduce: `scripts/feasible_workspace_validate.py`
    # re-derives a stratified sample in a cold process and compares cell for
    # cell, and a budget measured in seconds would give a different answer on a
    # loaded box than on an idle one.  A count gives the same answer on both.
    #
    # What the number means, said plainly: a cell this reports as unreachable
    # is a cell that no ladder shape could fly to and that the planner could
    # not fly to on the FIRST `RRT_CELL_PLANS` hovers it was offered.  That is
    # a lower bound on feasibility — the same kind of claim the hover ladder
    # and the fiber retry above it already make about themselves.
    budget0 = transit.stats()["calls"]
    rrt_on = paper.RRT_SAFE
    try:
        return _cell_hovers(arm, spec, h, probe, q_park, q_draw, x, y, hovers,
                            budget0, rrt_on)
    finally:
        # THE FLAG IS A MODULE GLOBAL AND THE WORKER OUTLIVES THE CELL.  A cell
        # that raised on its way out would leave the tier off for every cell
        # after it in this worker, and the map would silently become a
        # different measurement halfway through a chunk.
        paper.RRT_SAFE = rrt_on


def _cell_hovers(arm, spec, h, probe, q_park, q_draw, x, y, hovers,
                 budget0, rrt_on, plans=None):
    """`_cell`'s search over the hover ladder and then the fiber."""
    plans = _W.get("plans", RRT_CELL_PLANS) if plans is None else int(plans)
    best = float("-inf")
    z0 = float(hovers[0][1])
    seen = set()
    tier = 0
    while True:
        for q_hov, z in hovers:
            k = np.round(q_hov, 9).tobytes()
            if k in seen:
                continue
            seen.add(k)
            if rrt_on and paper.RRT_SAFE \
                    and transit.stats()["calls"] - budget0 >= plans:
                paper.RRT_SAFE = False
            beats = writing.enter_beats(spec, q_park, q_hov, q_draw,
                                        pen_ext=spec.pen, h_inv=h)
            if beats is None:
                continue
            clear = float(probe.clearance(arm, _dense(beats["steps"], q_park)))
            best = max(best, clear)
            if clear >= probe.margin:
                return FEASIBLE, float(z), clear
        if tier or not _W.get("fiber", True):
            break
        tier = 1
        hovers = _fiber_hovers(spec, q_draw, (x, y), h, _W.get("tries", 12))
    return NO_ROUTE, z0, (float("nan") if best == float("-inf") else best)


# ---------------------------------------------------------------------------
# THE ESCALATION LADDER
# ---------------------------------------------------------------------------
# THE SWEEP ABOVE IS A BUDGET APPLIED UNIFORMLY, AND THE DEAD SET IS 4.66 % OF
# THE CANVAS.  Spending the same seconds on the cell an arm inks on its first
# try as on the cell it cannot reach is what makes the budget a compromise; so
# once the map exists, the cells it refused get the budget they were never
# offered, and only they.
#
# EVERY RUNG IS A COUNT, NOT A CLOCK.  A cell's answer is the FIRST RUNG THAT
# CERTIFIES, and that has to be reproducible on a loaded box: the tree bound is
# `max_nodes`, the restart bound is `attempts` (whose seeds are derived from the
# scene and the endpoints, so they replay), the per-cell bound is a PLAN COUNT,
# and the shortcutter is bounded by rounds.  `transit.TIME_BUDGET` is set well
# above what any rung can spend, so the clock never decides an answer — and
# `transit.stats()["deadline"]` is reported so the claim is checked rather than
# asserted.  A run that reports `deadline 0` gave the only answer it could give.
#
# What each rung buys is measured, not assumed: `--rescue` prints per-rung hit
# rates and per-rung seconds, and the map's JSON carries them.
#
# THE RUNGS ARE SIZED ON A MEASURED PRICE.  A SOLVED tree costs 39 nodes and is
# free; a REFUSED one spends its whole node budget, measured at 11.7 ms a node
# on this rig — so a plan that is going to fail costs 3.5 s at 300 nodes x 1
# seed, 21 s at 900 x 2, 63 s at 1800 x 3.  The ladder climbs the two that
# matter (how many hovers, how many plans) before it climbs the one that is
# mostly a tax (how big a tree), because the measurement says the tree finds
# its answer early or not at all.
# ...AND THE ASIDE RUNG COMES SECOND, NOT THIRD, because it is cheap where it
# fires and free where it does not.  A cell whose DESCENT the ladder cannot do
# is behind a wall no park can move (see `_cell_escalated`), so an aside rung
# skips it outright rather than re-proving it one park set at a time; measured
# on the v11 dead set that is 73 % of the arm-cells, which is what makes the
# rung affordable at all.
RESCUE_RUNGS = (
    # name      tries  plans  nodes  attempts  aside
    ("fiber",      48,     8,   300,        1,     0),
    ("aside",      96,     8,   300,        1,     3),
    ("seeds",      96,    16,   900,        2,     0),
    ("deep",      192,    24,  1800,        2,     3),
)
RESCUE_TIME = 120.0     # s per plan: a ceiling that must never bind (see above)


def rung_settings(rung):
    """-> dict for one rung of `RESCUE_RUNGS`."""
    name, tries, plans, nodes, attempts, aside = RESCUE_RUNGS[int(rung)]
    return dict(name=name, tries=int(tries), plans=int(plans),
                nodes=int(nodes), attempts=int(attempts), aside=int(aside))


def _apply_rung(r):
    """Put one rung's budget in force in this worker."""
    _W["tries"] = int(r["tries"])
    _W["plans"] = int(r["plans"])
    transit.MAX_NODES = int(r["nodes"])
    transit.ATTEMPTS = int(r["attempts"])
    transit.TIME_BUDGET = RESCUE_TIME
    transit.SHORTCUT_TIME = 5.0      # never binds; SHORTCUT_ROUNDS does
    transit.SHORTCUT_ROUNDS = 24


def _park_sets(arm, xy, n_aside):
    """The parked fleets this rung may try, best first. -> [(parks, sig)].

    The first is always the SHIPPED set, so a rung that gains nothing from
    moving anybody gives the shipped answer and the ladder cannot regress.
    """
    base = _W["parks"]
    out = [(base, None)]
    if n_aside <= 0:
        return out
    fl = _W["fleet"]
    tx = np.asarray(xy, float).reshape(2)
    fired = [a for a, s in sorted(fl.items())
             if a != arm
             and float(np.linalg.norm(np.asarray(s.xy, float) - tx))
             <= layout.ASIDE_DISC_R]
    if not fired:
        return out
    seen = {_park_sig(base)}
    # `max_moved` walks 1 then 2: one arm aside is the cheap answer and the one
    # a phase can actually fly, and the second only ever runs when the first
    # bought nothing.
    for moved in (1, 2):
        for k in range(1, int(n_aside) + 1):
            parks, info = layout.region_aware_parks(
                fl, base, tx, drawing=arm, h_inv=_W["h"], grid=None,
                max_moved=moved, rank=k)
            if not info or not any(v["moved"] for v in info.values()):
                continue
            sig = _park_sig(parks)
            if sig in seen:
                continue
            seen.add(sig)
            out.append((parks, sig))
    return out


def _descent_ok(spec, q_hov, q_draw, h):
    """Can the pen get from this hover down onto the ink, ON THE LADDER?

    The second of `writing.enter_beats`' two legs, asked on its own and with
    the planner OFF, so the answer costs a ladder walk and not a tree search.
    """
    return writing._route(spec, q_hov, q_draw, spec.pen, h,
                          paper.CONTACT_FLOOR, writing.QD_FRAC,
                          writing.T_LOWER_F, writing.PAPER_SAFE) is not None


def _enter_clear(arm, spec, h, probe, q_park, q_hov, q_draw):
    """`enter_beats` + the parked-fleet verdict for one hover. -> clearance."""
    beats = writing.enter_beats(spec, q_park, q_hov, q_draw,
                                pen_ext=spec.pen, h_inv=h)
    if beats is None:
        return float("-inf")
    return float(probe.clearance(arm, _dense(beats["steps"], q_park)))


def _cell_escalated(arm, row, qcol, rung):
    """One (arm, cell) at one rung. -> (code, z_hover, park_clear).

    The rung's budget is already in force; what this adds over `_cell` is three
    things the sweep's evaluator cannot do.

    THE FIBER, EVEN WITH AN EMPTY LADDER.  `_cell` returns NO_HOVER the moment
    `_certified_hovers` is empty, so those cells never reach `_fiber_hovers`
    and never see a LEAN.  107 canvas cells died in that gap and it is a search
    that stopped, not a canvas that ended.

    THE DESCENT IS SCREENED BEFORE THE CORRIDOR IS PLANNED.  `enter_beats` is
    two routes — a metre of corridor from the park to the hover, and 6 cm of
    descent from the hover onto the ink — and it walks them in that order, so a
    hover whose DESCENT is impossible still costs a full corridor plan.  It is
    not a corner case: measured over 30 route-dead cells at 48 fiber tries, 22
    of them have no hover at all whose descent the ladder can do, and the
    planner was being spent proving corridors to hovers that were never going
    to be usable.  Screening every hover's descent on the ladder costs 2.4 s
    for all 48 of them and says which of the two walls this cell is behind.

    THE PARKED FLEET IS A VARIABLE, and it is one for exactly the cells it can
    help.  A parked arm is in the corridor, never in the 6 cm of descent — the
    descent's ladder is park-blind, and the only way a park reaches it at all
    is by RESTRICTING the C-space tier.  So a cell walled at the descent is
    offered no aside park: moving somebody could not have helped it, and the
    rung would spend four searches to prove that one at a time.
    """
    fl, h = _W["fleet"], _W["h"]
    if _W.get("frozen"):
        frozen.observe(arm)          # never check the mover against itself
    spec = fl[arm]
    x, y = float(row[0]), float(row[1])
    q_draw = np.asarray(row[qcol:qcol + 7], float)

    hovers = _certified_hovers(spec, q_draw, (x, y), h)
    hovers = hovers + list(_fiber_hovers(spec, q_draw, (x, y), h, _W["tries"]))
    if not hovers:
        return NO_HOVER, 0.0, float("nan")
    seen, uniq = set(), []
    for q_hov, z in hovers:
        k = np.round(q_hov, 9).tobytes()
        if k in seen:
            continue
        seen.add(k)
        uniq.append((q_hov, float(z)))
    hovers = uniq
    z0 = float(hovers[0][1])

    rrt_on = paper.RRT_SAFE
    best = float("-inf")
    try:
        # ---- the descent screen, ladder only ---------------------------
        paper.RRT_SAFE = False
        landable = [hv for hv in hovers if _descent_ok(spec, hv[0], q_draw, h)]
        paper.RRT_SAFE = rrt_on
        if rung["aside"] and not landable:
            # AN ASIDE RUNG HAS NOTHING FOR THIS CELL.  The wall is the 6 cm
            # between the hover and the ink, and no park pose is in there — the
            # rung below already spent its budget on that descent with the same
            # parks this one would use.  Charging the cell four park sets to
            # re-prove it is how an escalation ladder stops being affordable.
            return NO_ROUTE, z0, float("nan")
        sets = _park_sets(arm, (x, y), rung["aside"])
        if rung["aside"] and len(sets) < 2:
            # nobody is standing over this cell, so this rung has no question
            # to ask about it that the rung below did not already ask
            return NO_ROUTE, z0, float("nan")
        for parks, sig in sets:
            probe, _psig = _probe_for(parks)
            # AND THE POSE-AWARE MODEL FOLLOWS THE PARKS IT IS MODELLING.  An
            # aside rung MOVES the partners, so a frozen set derived from the
            # shipped depots would be describing arms that are no longer there
            # — the obstacle set and the ParkProbe would disagree about the
            # same six poses.  Re-freeze on the set actually in force.
            if _W.get("frozen"):
                frozen.freeze(parks, _W["fleet"], _W["pens"], h)
                frozen.observe(arm)
                paper.clear_cache()
            if _W.get("rrt", 0.0) > 0.0:
                _park_probe_hook(arm, probe, sig)
            q_park = np.asarray(parks[arm], float)
            # ---- the corridor, for hovers that can land -----------------
            paper.RRT_SAFE = rrt_on
            budget0 = transit.stats()["calls"]
            for q_hov, z in landable:
                if rrt_on and paper.RRT_SAFE and \
                        transit.stats()["calls"] - budget0 >= rung["plans"]:
                    paper.RRT_SAFE = False
                c = _enter_clear(arm, spec, h, probe, q_park, q_hov, q_draw)
                best = max(best, c)
                if c >= probe.margin:
                    return FEASIBLE, float(z), float(c)
            # ---- and then the descent itself, in the C-space ------------
            # Only where the ladder found NO landable hover: that cell is
            # behind the descent, and the seven-dimensional tier is the only
            # thing left that has not been asked about it.
            if landable:
                continue
            paper.RRT_SAFE = rrt_on
            budget0 = transit.stats()["calls"]
            for q_hov, z in hovers:
                if rrt_on and paper.RRT_SAFE and \
                        transit.stats()["calls"] - budget0 >= rung["plans"]:
                    break
                c = _enter_clear(arm, spec, h, probe, q_park, q_hov, q_draw)
                best = max(best, c)
                if c >= probe.margin:
                    return FEASIBLE, float(z), float(c)
    finally:
        paper.RRT_SAFE = rrt_on
    return NO_ROUTE, z0, (float("nan") if best == float("-inf") else best)


def _rescue_chunk(job):
    """(rung index, [(arm, row index)]) -> per-cell results, in a worker."""
    ri, items = job
    r = rung_settings(ri)
    _apply_rung(r)
    qcol = atlas.QCOL
    s0 = dict(transit.stats())
    out = []
    for arm, i in items:
        rows = _go_rows(arm)
        row = rows[i]
        t0 = time.time()
        code, z, clear = _cell_escalated(arm, row, qcol, r)
        out.append((int(arm), float(row[0]), float(row[1]), int(code),
                    float(z), float(clear), int(i), float(time.time() - t0)))
    s1 = transit.stats()
    # THE CLOCK MUST NOT HAVE DECIDED ANYTHING (see `RESCUE_RUNGS`), so the
    # count of searches that stopped on it comes back with the answers.
    return ri, out, {k: s1[k] - s0.get(k, 0) for k in
                     ("calls", "solved", "failed", "deadline", "nodes")}


_ROWS = {}


def _go_rows(arm):
    if arm not in _ROWS:
        arr, _ = atlas.load(_W["atlas_dir"], arm)
        _ROWS[arm] = arr[atlas.strict_go(arr)]
    return _ROWS[arm]


def _chunk(job):
    """(arm, [row indices]) -> per-row results, in a worker process."""
    arm, idx = job
    rows = _go_rows(arm)
    qcol = atlas.QCOL
    out = []
    for i in idx:
        r = rows[i]
        code, z, clear = _cell(arm, r, qcol, _W["redundant"])
        out.append((float(r[0]), float(r[1]), int(code), float(z),
                    float(clear), int(i)))
    return arm, out


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------
def sweep(arms, atlas_dir, h, workers, chunk=24, every=1, redundant=True,
          ckpt=None, pitch=None, log=print, fiber=True, lean=0.0, tries=12,
          rrt=0.0, rrt_nodes=600):
    """-> {arm: (N,6) array of (x, y, code, z_hover, park_clear, atlas_row)}.

    Chunks are STRIDED, not contiguous: an atlas row block is one band of y, and
    a band at the edge of reach is all refusals (the whole route ladder, 4 s a
    cell) while a band under the arm is all first-shape hits (0.1 s).  Handed
    contiguous blocks the pool finishes the cheap arms and then waits on one
    worker for twenty minutes.  Strided, every chunk is a fair sample of the
    canvas and the pool drains evenly.
    """
    import multiprocessing as mp

    sel = {}
    for a in arms:
        arr, meta = atlas.load(atlas_dir, a)
        ok, why = atlas.is_current(meta)
        if not ok:
            raise SystemExit(f"atlas for arm {a} is STALE: {why}\n"
                             f"  re-sweep it into {atlas_dir} before trusting it")
        n = int(atlas.strict_go(arr).sum())
        sel[a] = np.arange(0, n, max(1, int(every)))
        log(f"  arm {a}: {n} strict-GO cells "
            f"({len(sel[a])} swept, model signature current)")

    jobs = []
    for a in arms:
        idx = sel[a]
        nchunk = max(1, int(np.ceil(len(idx) / chunk)))
        for k in range(nchunk):
            jobs.append((a, [int(v) for v in idx[k::nchunk]]))
    jobs = [j for j in jobs if j[1]]
    total = sum(len(j[1]) for j in jobs)
    log(f"{total} arm-cells, {len(jobs)} chunks, {workers} workers", flush=True)

    res = {a: [] for a in arms}
    t0 = time.time()
    done = ncell = 0
    ctx = mp.get_context("fork")
    with ctx.Pool(workers, initializer=_init,
                  initargs=(str(atlas_dir), h, redundant, pitch, fiber,
                            lean, tries, rrt, rrt_nodes, 1, None, None, None,
                            FROZEN, CYLM)) as pool:
        for arm, out in pool.imap_unordered(_chunk, jobs, chunksize=1):
            res[arm].extend(out)
            done += 1
            ncell += len(out)
            el = time.time() - t0
            log(f"  {done}/{len(jobs)} chunks  {ncell}/{total} cells  "
                f"{el:.0f}s  eta {el / done * (len(jobs) - done):.0f}s",
                flush=True)
            # NOTHING IS LOST: a partial grid on disk after every chunk
            if ckpt and (done % 10 == 0 or done == len(jobs)):
                np.savez_compressed(
                    ckpt, **{f"arm{k}": (np.array(v) if v else np.zeros((0, 6)))
                             for k, v in res.items()})
    return {a: (np.array(v) if v else np.zeros((0, 6))) for a, v in res.items()}


def rescue(per_arm, arms, atlas_dir, h, workers, rungs=None, chunk=6,
           ckpt=None, pitch=None, log=print, lean=0.0, rrt=1.0, calib=None):
    """Escalate the DEAD cells of a finished map. -> (per_arm, ladder).

    Only cells the map REFUSED are touched, and a cell stops being touched the
    moment any arm carries it: the ladder's answer for a cell is the first rung
    that certifies it, and the rungs above that one are never charged for it.

    `per_arm` is updated in place and returned; `ladder` is the per-rung
    accounting the JSON carries — how many arm-cells the rung was offered, how
    many it turned, what it cost, and whether the CLOCK ever decided an answer.
    """
    import multiprocessing as mp

    rungs = list(range(len(RESCUE_RUNGS))) if rungs is None else list(rungs)
    idx = {a: {int(r[5]): i for i, r in enumerate(per_arm[a])} for a in arms}
    ladder = []
    for ri in rungs:
        r = rung_settings(ri)
        comp = compose(per_arm, arms)
        dead = comp["cause"] != FEASIBLE
        # every (arm, cell) that is still refused ON A DEAD CELL
        items = []
        for a in arms:
            d = per_arm[a]
            if not len(d):
                continue
            code = d[:, 2].astype(int)
            ii = np.round(d[:, 1] / GRID).astype(int)
            jj = np.round(d[:, 0] / GRID).astype(int)
            sel = (code != FEASIBLE) & dead[ii, jj]
            items += [(int(a), int(v)) for v in d[sel][:, 5].astype(int)]
        if not items:
            log(f"rung {ri} {r['name']}: nothing left to escalate")
            break
        # STRIDED, for the same reason `sweep` strides: a rung's cost is wildly
        # uneven per cell and a contiguous block leaves one worker holding it.
        nchunk = max(1, int(np.ceil(len(items) / chunk)))
        jobs = [(ri, items[k::nchunk]) for k in range(nchunk)]
        jobs = [j for j in jobs if j[1]]
        log(f"rung {ri} {r['name']}: {len(items)} arm-cells on "
            f"{int(dead.sum())} dead cells, {len(jobs)} chunks "
            f"(tries {r['tries']}, plans {r['plans']}, nodes {r['nodes']}, "
            f"attempts {r['attempts']}, aside {r['aside']})", flush=True)
        if r["aside"]:
            # BUILT IN THE PARENT, ON PURPOSE.  Certifying one arm's 61 aside
            # candidates is 9 s of IK, the pool is forked, and a table built
            # here is inherited by every worker instead of being rebuilt in
            # each of them.
            ta = time.time()
            fl = rig(pitch, h, calib)[0]
            for a in sorted(fl):
                layout.aside_candidates(
                    fl[a], pen_lat=frames.PEN_LAT_HOLDER,
                    extra=(layout.PARK_GRID_PROPOSED[a],)
                    if a in layout.PARK_GRID_PROPOSED else ())
            log(f"  aside park candidates for {len(fl)} arms in "
                f"{time.time() - ta:.0f}s", flush=True)
        t0 = time.time()
        got = []
        done = 0
        tstat = dict(calls=0, solved=0, failed=0, deadline=0, nodes=0)
        ctx = mp.get_context("fork")
        with ctx.Pool(workers, initializer=_init,
                      initargs=(str(atlas_dir), h, True, pitch, True, lean,
                                r["tries"], rrt, r["nodes"], r["attempts"],
                                r["plans"], 5.0, calib, FROZEN,
                                CYLM)) as pool:
            for _ri, out, st in pool.imap_unordered(_rescue_chunk, jobs,
                                                    chunksize=1):
                got += out
                for k in tstat:
                    tstat[k] += int(st.get(k, 0))
                done += 1
                el = time.time() - t0
                log(f"  rung {ri} {done}/{len(jobs)} chunks  {len(got)} cells  "
                    f"{el:.0f}s  eta {el / done * (len(jobs) - done):.0f}s",
                    flush=True)
        turned = 0
        secs = 0.0
        for arm, x, y, code, z, clear, i, dt in got:
            secs += dt
            k = idx[arm].get(int(i))
            if k is None:
                continue
            was = int(per_arm[arm][k, 2])
            if code == FEASIBLE or (was == NO_HOVER and code == NO_ROUTE):
                per_arm[arm][k, 2] = code
                per_arm[arm][k, 3] = z
                per_arm[arm][k, 4] = clear
                turned += code == FEASIBLE
            elif np.isfinite(clear):
                # a better witness for the SAME refusal: the map reports the
                # closest anyone got, and a deeper search got closer
                old = per_arm[arm][k, 4]
                if not np.isfinite(old) or clear > old:
                    per_arm[arm][k, 4] = clear
        after = compose(per_arm, arms)
        row = dict(rung=int(ri), name=r["name"], offered=len(items),
                   turned=int(turned),
                   dead_before=int(dead.sum()),
                   dead_after=int((after["cause"] != FEASIBLE).sum()),
                   wall_s=round(time.time() - t0, 1),
                   cell_s=round(secs / max(1, len(got)), 3),
                   rrt=dict(tstat),
                   **{k: int(v) for k, v in
                      dict(tries=r["tries"], plans=r["plans"], nodes=r["nodes"],
                           attempts=r["attempts"], aside=r["aside"]).items()})
        ladder.append(row)
        log(f"  rung {ri} {r['name']}: {turned} arm-cells turned, dead "
            f"{row['dead_before']} -> {row['dead_after']} "
            f"({row['wall_s']:.0f}s, {row['cell_s']:.2f}s/arm-cell); "
            f"RRT {tstat['calls']} plans {tstat['solved']} solved, "
            f"{tstat['deadline']} stopped on the CLOCK", flush=True)
        if ckpt:
            np.savez_compressed(ckpt, **{f"arm{k}": v
                                         for k, v in per_arm.items()})
    return per_arm, ladder


# ---------------------------------------------------------------------------
# the map
# ---------------------------------------------------------------------------
def compose(per_arm, arms, sheet=SHEET, grid=GRID):
    """Per-arm cell verdicts -> the canvas rasters.

    -> dict(n_arms, cause, draw_any, hover_any, per_arm_mask) where `cause` is
    the union verdict per cell: a cell is FEASIBLE if ANY arm carried it all
    the way, and otherwise blamed on the FURTHEST layer any arm reached.
    """
    xs, ys = axes(sheet, grid)
    H, W = len(ys), len(xs)
    n_arms = np.zeros((H, W), int)
    draw_any = np.zeros((H, W), bool)
    hover_any = np.zeros((H, W), bool)
    masks = {}
    for a in arms:
        m = np.zeros((H, W), bool)
        d = per_arm[a]
        for row in d:
            x, y, code = float(row[0]), float(row[1]), int(row[2])
            i, j = _ij(x, y, xs, ys)
            draw_any[i, j] = True
            if code != NO_HOVER:
                hover_any[i, j] = True
            if code == FEASIBLE:
                m[i, j] = True
                n_arms[i, j] += 1
        masks[a] = m
    cause = np.full((H, W), NO_DRAW, int)
    cause[hover_any] = NO_ROUTE
    cause[draw_any & ~hover_any] = NO_HOVER
    cause[n_arms > 0] = FEASIBLE
    return dict(n_arms=n_arms, cause=cause, draw_any=draw_any,
                hover_any=hover_any, per_arm=masks, xs=xs, ys=ys)


def _runs(row):
    """Longest run of True in a bool row. -> (start, length)."""
    if not row.any():
        return 0, 0
    d = np.diff(np.concatenate(([0], row.view(np.int8), [0])))
    s = np.where(d == 1)[0]
    e = np.where(d == -1)[0]
    k = int(np.argmax(e - s))
    return int(s[k]), int(e[k] - s[k])


def all_rects(mask):
    """Every maximal all-True rectangle, by row band. -> [(i0, j0, h, w)].

    O(H^2 * W) and the grid is 182 x 91, so it is exact and instant.  The
    histogram scan is asymptotically better and only ever finds the largest by
    AREA; the brief wants the largest at each ORIENTATION too, and those are
    different rectangles, so every band is enumerated instead.
    """
    H, W = mask.shape
    out = []
    for i0 in range(H):
        band = mask[i0].copy()
        for i1 in range(i0, H):
            if i1 > i0:
                band &= mask[i1]
            if not band.any():
                break
            j0, w = _runs(band)
            if w:
                out.append((i0, j0, i1 - i0 + 1, w))
    return out


def rect_report(mask, grid=GRID):
    """The largest clean rectangle, and the largest in each ORIENTATION.

    A rectangle turned 90 degrees is not a new rectangle in a raster — it is
    the same one with its sides swapped, so "both orientations" can only mean
    what artwork shape fits: the biggest PORTRAIT block (long side up the
    canvas) and the biggest LANDSCAPE one (long side across it).  On a
    1.80 x 3.63 m canvas those are genuinely different answers.
    """
    R = all_rects(mask)

    def pack(r):
        i0, j0, hh, ww = r
        return dict(cells=int(hh * ww), x0=round(float(j0 * grid), 3),
                    y0=round(float(i0 * grid), 3),
                    w=round(float(ww * grid), 3), h=round(float(hh * grid), 3),
                    area_m2=round(float(ww * hh * grid * grid), 4),
                    aspect=round(float(ww / hh), 3))

    def best(pred):
        c = [r for r in R if pred(r)]
        return pack(max(c, key=lambda r: r[2] * r[3])) if c else None

    return (best(lambda r: True),                       # largest by area
            best(lambda r: r[2] >= r[3]),               # portrait  (h >= w)
            best(lambda r: r[3] >= r[2]))               # landscape (w >= h)


def blobs(mask):
    """Sizes of the 4-connected components of a bool mask. -> sorted desc.

    Two-pass union-find rather than scipy, which is not installed here.
    """
    H, W = mask.shape
    lab = np.full((H, W), -1, int)
    par = []

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a

    for i in range(H):
        row = mask[i]
        for j in np.where(row)[0]:
            up = lab[i - 1, j] if i and mask[i - 1, j] else -1
            lf = lab[i, j - 1] if j and mask[i, j - 1] else -1
            if up < 0 and lf < 0:
                lab[i, j] = len(par)
                par.append(len(par))
            elif up < 0 or lf < 0:
                lab[i, j] = max(up, lf)
            else:
                lab[i, j] = up
                a, b = find(up), find(lf)
                if a != b:
                    par[b] = a
    if not par:
        return np.zeros(0, int)
    flat = lab[mask]
    roots = np.array([find(int(v)) for v in flat])
    return np.sort(np.bincount(roots)[np.bincount(roots) > 0])[::-1]


def numbers(comp, per_arm, arms, sheet=SHEET, grid=GRID, fleet=None,
            park_hover=None, pitch=SHIPPED_PITCH, h=0.940, lean=None):
    cell_a = grid * grid
    n_arms, cause = comp["n_arms"], comp["cause"]
    H, W = cause.shape
    tot = H * W
    feas = int((cause == FEASIBLE).sum())

    def pct(n):
        return round(100.0 * n / tot, 2)

    fleet = layout.FLEET_PROPOSED if fleet is None else fleet
    park_hover = (layout.PARK_HOVER_PROPOSED if park_hover is None
                  else park_hover)
    out = dict(
        rig="proposed", tool="lateral", h=float(h), pitch=float(pitch),
        grid=grid,
        sheet=list(sheet), cells=tot, cell_area_m2=cell_a,
        canvas_area_m2=round(tot * cell_a, 4),
        feasible_cells=feas, feasible_pct=pct(feas),
        feasible_m2=round(feas * cell_a, 4),
    )
    out["layers"] = {
        "draw_pose_only": dict(cells=int(comp["draw_any"].sum()),
                               pct=pct(int(comp["draw_any"].sum())),
                               m2=round(int(comp["draw_any"].sum()) * cell_a, 4)),
        "plus_hover": dict(cells=int(comp["hover_any"].sum()),
                           pct=pct(int(comp["hover_any"].sum())),
                           m2=round(int(comp["hover_any"].sum()) * cell_a, 4)),
        "plus_reachability": dict(cells=feas, pct=pct(feas),
                                  m2=round(feas * cell_a, 4)),
    }
    out["dead_by_cause"] = {
        CAUSE_NAME[c]: dict(cells=int((cause == c).sum()),
                            pct=pct(int((cause == c).sum())),
                            m2=round(int((cause == c).sum()) * cell_a, 4))
        for c in (NO_DRAW, NO_HOVER, NO_ROUTE)}

    if lean:
        # THE CELLS THAT ARE WAITING ON A DECISION, KEPT SEPARATE FROM THE ONES
        # THAT ARE WAITING ON A SEARCH.  `scripts/lean_study.py` asks the atlas
        # the same question at a wider cone; a cell that certifies at 17.5 or
        # 20 degrees is not a hole in the canvas, it is a hole in what the pen
        # is allowed to do, and the two must not be added up.
        per = lean.get("per_cell", [])
        by = {}
        for r in per:
            by.setdefault(f"{float(r['min_lean_deg']):g}", 0)
            by[f"{float(r['min_lean_deg']):g}"] += 1
        out["lean_pending"] = dict(
            shipped_cone_deg=float(lean.get("shipped_cone_deg", 15.0)),
            cells=len(per), pct=pct(len(per)),
            m2=round(len(per) * cell_a, 4),
            min_lean_histogram=dict(sorted(by.items(),
                                           key=lambda kv: float(kv[0]))),
            cumulative_by_cone=lean.get("cumulative_by_cone", {}),
            # A WIDER CONE BUYS A POSE, NOT A CELL.  Granting it puts these
            # cells on layer 1 and they still have to earn layers 2 and 3 like
            # everybody else, so the honest headline is what it does to the
            # DRAW-POSE layer.  Quoting it against `feasible` would be claiming
            # a hover and a route nobody has planned.
            draw_pose_pct_if_granted=round(
                100.0 * (int(comp["draw_any"].sum()) + len(per)) / tot, 2),
            note="cells with NO certified drawing pose inside the shipped "
                 "15-degree cone that have one inside a wider one.  Not "
                 "counted as feasible anywhere in this file; the pen's cone is "
                 "a hardware decision and this is what it is worth.  Granting "
                 "it would put them on layer 1; the hover and the route are "
                 "still theirs to earn.")

    out["histogram"] = {str(k): int((n_arms == k).sum())
                        for k in range(0, len(arms) + 1)}
    out["ge2_pct"] = pct(int((n_arms >= 2).sum()))
    out["ge3_pct"] = pct(int((n_arms >= 3).sum()))
    out["max_arms"] = int(n_arms.max())

    pa = {}
    for a in arms:
        d = per_arm[a]
        m = comp["per_arm"][a]
        n = int(m.sum())
        codes = d[:, 2].astype(int) if len(d) else np.zeros(0, int)
        pa[str(a)] = dict(
            name=fleet[a].name,
            xy=[round(float(v), 4) for v in fleet[a].xy],
            park_hover=list(park_hover.get(a, ())),
            draw_cells=int(len(d)), draw_pct=pct(len(d)),
            hover_cells=int((codes != NO_HOVER).sum()),
            hover_pct=pct(int((codes != NO_HOVER).sum())),
            feasible_cells=n, feasible_pct=pct(n),
            feasible_m2=round(n * cell_a, 4),
            lost_to_hover=int((codes == NO_HOVER).sum()),
            lost_to_route=int((codes == NO_ROUTE).sum()),
        )
    out["per_arm"] = pa

    # canvas thirds, along the long axis
    ys = comp["ys"]
    edges = [0, len(ys) // 3, 2 * len(ys) // 3, len(ys)]
    thirds = []
    for k in range(3):
        sl = slice(edges[k], edges[k + 1])
        sub = cause[sl]
        n = int((sub == FEASIBLE).sum())
        thirds.append(dict(
            name=["south end", "middle", "north end"][k],
            y_range=[round(float(ys[edges[k]]), 3),
                     round(float(ys[edges[k + 1] - 1]), 3)],
            cells=int(sub.size), feasible_cells=n,
            feasible_pct=round(100.0 * n / sub.size, 2),
            feasible_m2=round(n * cell_a, 4),
            dead_no_draw=int((sub == NO_DRAW).sum()),
            dead_no_hover=int((sub == NO_HOVER).sum()),
            dead_no_route=int((sub == NO_ROUTE).sum())))
    out["thirds"] = thirds

    a, p, l = rect_report(cause == FEASIBLE, grid)
    out["largest_rect"] = a
    out["largest_rect_portrait"] = p
    out["largest_rect_landscape"] = l

    # WHERE THE DEAD AREA IS, in the three shapes the rig actually makes:
    # the six discs an arm cannot reach under its own boom, the route-dead
    # middle where the two middle-row arms park in everybody's way, and
    # whatever is left over.  Disjoint, in that order, so the three add up.
    dead = cause != FEASIBLE
    XS, YS = np.meshgrid(comp["xs"], ys)
    under = np.zeros_like(dead)
    for arm in arms:
        bx, by = fleet[arm].xy
        under |= ((XS - bx) ** 2 + (YS - by) ** 2) <= 0.30 ** 2
    mid = np.zeros_like(dead)
    mid[edges[1]:edges[2]] = True

    def blk(m):
        n = int(m.sum())
        return dict(cells=n, m2=round(n * cell_a, 4), pct=pct(n))

    d_under = dead & under
    d_mid = dead & ~under & mid
    d_rest = dead & ~under & ~mid
    out["dead_split"] = dict(
        total=blk(dead),
        under_base_discs_r30=blk(d_under),
        middle_third_outside_the_discs=blk(d_mid),
        elsewhere=blk(d_rest),
        middle_third_route_dead=blk(d_mid & (cause == NO_ROUTE)),
        route_dead_everywhere=blk(dead & (cause == NO_ROUTE)),
        under_base_discs_area_m2=round(float(under.sum()) * cell_a, 4))

    # ...AND WHAT SHAPE IT IS, which is the number that explains the
    # rectangles.  81.84 % drawable and a biggest clean block of 0.74 m2 only
    # reconcile if the dead area is not one hole: it is six solid discs under
    # the booms plus a confetti of isolated refusals scattered through canvas
    # that is otherwise fine.  A rectangle is stopped by one bad pixel, so the
    # confetti costs far more artwork than its area does.
    sz = blobs(dead)
    small = int(sz[sz <= 50].sum())
    big = int(sz[sz > 50].sum())
    nbr = np.zeros_like(cause, int)
    F = (cause == FEASIBLE)
    P = np.pad(F, 1)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if (di, dj) != (0, 0):
                nbr += P[1 + di:1 + di + H, 1 + dj:1 + dj + W]
    speck = int((dead & (nbr >= 5)).sum())
    out["dead_shape"] = dict(
        blobs=int(len(sz)),
        largest_blobs=[int(v) for v in sz[:8]],
        solid_cells=big, solid_m2=round(big * cell_a, 4),
        solid_pct_of_dead=round(100.0 * big / max(1, int(dead.sum())), 1),
        confetti_cells=small, confetti_m2=round(small * cell_a, 4),
        confetti_pct_of_dead=round(100.0 * small / max(1, int(dead.sum())), 1),
        dead_cells_with_5plus_feasible_neighbours=speck,
        speckle_pct_of_dead=round(100.0 * speck / max(1, int(dead.sum())), 1),
        note=("blobs of more than 50 cells are the six under-boom discs; the "
              "rest is scattered refusals, and a rectangle is stopped by one "
              "of them"))
    return out


# ---------------------------------------------------------------------------
# the figure
# ---------------------------------------------------------------------------
# THE PALETTE IS COMPUTED, NOT PICKED.  Two ORDINAL families share one raster —
# how many arms can draw a cell, and, where none can, how far the cell got — so
# each is a single-hue ramp with monotone lightness, and the two hues are the
# blue/amber pair rather than the green/red one that collapses under
# deuteranopia.  Checked, not eyeballed (out/palette_build.py): both ramps
# monotone in OKLCH L with adjacent dL >= 0.08, hue span <= 15 deg, light end
# >= 2.0:1 on white; worst cross-family pair dE 14.4 under protanopia and 15.9
# under deuteranopia (target 8); adjacent feasible steps dE 8.3-8.7.
FEAS_COLS = ["#90b4e0", "#7199cb", "#537fb5", "#3465a0", "#114c8a", "#003275"]
DEAD_COLS = {3: "#e1ab7b", 2: "#ab6d30", 1: "#763100"}   # NO_ROUTE/HOVER/DRAW
INK, INK2, INK3 = "#1a1a19", "#4a4a48", "#77776f"
METAL = "#111827"


def _bar_panel(ax, labels, vals, cols, title, note=None, xmax=100.0,
               fmt="{:.2f}%", hatch=None):
    """A thin horizontal bar row with selective direct labels.

    `hatch` is per-bar and is how a PROJECTION is drawn beside a measurement:
    the same axis, so the two are comparable at a glance, and a different
    surface, so nobody reads a hatched bar as a number this rig has.
    """
    y = np.arange(len(labels))[::-1]
    bars = ax.barh(y, vals, height=0.52, color=cols, zorder=3)
    for b, h in zip(bars, hatch or [None] * len(labels)):
        if h:
            b.set_hatch(h)
            b.set_edgecolor("white")
            b.set_linewidth(0.0)
    for yy, v in zip(y, vals):
        ax.text(v + xmax * 0.015, yy, fmt.format(v), va="center", ha="left",
                fontsize=9, color=INK, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9, color=INK)
    ax.set_xlim(0, xmax * 1.22)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    ax.set_xticks([])
    ax.tick_params(left=False)
    for sp in ("top", "right", "bottom", "left"):
        ax.spines[sp].set_visible(False)
    ax.set_title(title, fontsize=10, color=INK, fontweight="bold", loc="left",
                 pad=6)
    if note:
        ax.text(0, -0.62, note, transform=ax.get_yaxis_transform(),
                fontsize=8, color=INK3, va="top")


def figure(comp, nums, arms, path, sheet=SHEET, grid=GRID, fleet=None,
           park_hover=None, lean=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle
    from matplotlib.lines import Line2D
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from aris_sixarm import mounts

    fleet = layout.FLEET_PROPOSED if fleet is None else fleet
    park_hover = (layout.PARK_HOVER_PROPOSED if park_hover is None
                  else park_hover)
    pitch, h = nums["pitch"], nums["h"]
    n_arms, cause = comp["n_arms"], comp["cause"]
    xs, ys = comp["xs"], comp["ys"]
    ext = [xs[0] - grid / 2, xs[-1] + grid / 2, ys[0] - grid / 2,
           ys[-1] + grid / 2]

    # the right column's panels are sized by the rows they actually carry, so
    # a rig whose deepest cell is three arms deep does not get a six-row gap
    nmax = max(1, int(n_arms.max()))
    fig = plt.figure(figsize=(15.2, 13.8))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.0, 1.06],
                          height_ratios=[3.2, float(len(arms)),
                                         max(2.4, float(nmax)), 3.2],
                          left=0.055, right=0.975, top=0.878, bottom=0.175,
                          wspace=0.20, hspace=0.40)
    ax = fig.add_subplot(gs[:, 0])

    # ONE raster: 1..6 = how many arms, -1..-3 = which layer refused
    img = np.where(n_arms > 0, n_arms, -cause).astype(float)
    cmap = ListedColormap([DEAD_COLS[3], DEAD_COLS[2], DEAD_COLS[1],
                           "#ffffff"] + FEAS_COLS)
    norm = BoundaryNorm([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5,
                         6.5], cmap.N)
    ax.imshow(img, origin="lower", extent=ext, cmap=cmap, norm=norm,
              interpolation="nearest", zorder=1)
    ax.add_patch(Rectangle((ext[0], ext[2]), ext[1] - ext[0], ext[3] - ext[2],
                           fill=False, ec=INK2, lw=1.1, zorder=10))

    # the metal, in plan: boom footprint, base plate, and where the pen parks
    m = mounts.MOUNTS
    br, (pw, pd) = m.boom_r, m.plate_xy
    for arm in arms:
        bx, by = fleet[arm].xy
        ax.add_patch(Rectangle((bx - pw / 2, by - pd / 2), pw, pd, fill=False,
                               ec=METAL, lw=0.9, ls=(0, (3, 2)), zorder=6))
        ax.add_patch(Circle((bx, by), br, fill=False, ec="white", lw=3.2,
                            zorder=6))
        ax.add_patch(Circle((bx, by), br, fill=False, ec=METAL, lw=1.6,
                            zorder=7))
        ax.plot([bx], [by], marker="x", ms=7, mew=2.0, color=METAL, zorder=8)
        pa = nums["per_arm"][str(arm)]
        ax.annotate(f"arm {arm}   {pa['feasible_pct']:.1f}%", (bx, by),
                    xytext=(0, 13), textcoords="offset points", ha="center",
                    va="bottom", fontsize=8.4, color=INK, fontweight="bold",
                    zorder=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=INK3,
                              lw=0.6, alpha=0.94))
        ph = park_hover.get(arm)
        if ph:
            ax.plot([ph[0]], [ph[1]], marker="P", ms=11, color="#8a1c1c",
                    mec="white", mew=1.4, zorder=9)

    # THE LEAN-PENDING SET, MARKED AND NOT COLOURED.  These cells are dead in
    # the raster like any other and they are dead for a different reason —
    # a decision nobody has taken, not a geometry nobody can fly.  A ring over
    # the cell says so without pretending the cell is drawable.
    lp = nums.get("lean_pending")
    if lp and lean and lean.get("per_cell"):
        P = np.array([[r["x"], r["y"]] for r in lean["per_cell"]], float)
        ax.plot(P[:, 0], P[:, 1], linestyle="none", marker="o", ms=3.4,
                mfc="none", mec="#0b6e4f", mew=1.1, zorder=8)

    # THE SIZES LIVE IN THE LEGEND, not on top of the map: the two rectangles
    # overlap wherever the feasible region is one blob, and two label boxes in
    # the same 40 cm of canvas hide the cells they are describing.
    for key, ls, tag in (("largest_rect_portrait", "-", "P"),
                         ("largest_rect_landscape", (0, (4, 2)), "L")):
        r = nums.get(key)
        if not r:
            continue
        ax.add_patch(Rectangle((r["x0"], r["y0"]), r["w"], r["h"], fill=False,
                               ec="white", lw=3.6, zorder=9))
        ax.add_patch(Rectangle((r["x0"], r["y0"]), r["w"], r["h"], fill=False,
                               ec=METAL, lw=2.0, ls=ls, zorder=10))
        ax.text(r["x0"] + 0.03, r["y0"] + r["h"] - 0.03, tag, ha="left",
                va="top", fontsize=10, color=METAL, fontweight="bold",
                zorder=11,
                bbox=dict(boxstyle="circle,pad=0.16", fc="white", ec=METAL,
                          lw=1.0))

    ax.set_xlim(ext[0] - 0.05, ext[1] + 0.05)
    ax.set_ylim(ext[2] - 0.05, ext[3] + 0.05)
    ax.set_aspect("equal")
    ax.set_xlabel("canvas x (m)", fontsize=9.5, color=INK2)
    ax.set_ylabel("canvas y (m)", fontsize=9.5, color=INK2)
    ax.tick_params(labelsize=8.5, colors=INK3)
    for sp in ax.spines.values():
        sp.set_visible(False)
    # ---- the legend, under the map ----------------------------------------
    # THE FEASIBLE SCALE IS ORDINAL, so it is drawn as a ramp with a numbered
    # axis and not as six unrelated swatches: the reader is meant to see 1 -> 6
    # as an order, which a column of legend keys hides.
    # ...and only as far as the fleet actually gets.  A six-step key over a
    # canvas whose deepest cell is reachable by four arms promises a redundancy
    # this rig does not have.
    cb = fig.add_axes([0.055, 0.112, 0.20 * nmax / 6.0, 0.017])
    for k in range(nmax):
        cb.add_patch(Rectangle((k, 0), 1, 1, fc=FEAS_COLS[k], ec="white",
                               lw=1.2))
    cb.set_xlim(0, nmax)
    cb.set_ylim(0, 1)
    cb.set_yticks([])
    cb.set_xticks(np.arange(nmax) + 0.5)
    cb.set_xticklabels([str(k + 1) for k in range(nmax)], fontsize=9,
                       color=INK)
    cb.tick_params(length=0, pad=3)
    for sp in cb.spines.values():
        sp.set_visible(False)
    cb.set_title(f"FEASIBLE — how many arms can solo-draw the cell "
                 f"(never more than {nmax})",
                 fontsize=9.2, color=INK, fontweight="bold", loc="left", pad=5)

    d = nums["dead_by_cause"]
    hs = [Rectangle((0, 0), 1, 1, fc=DEAD_COLS[c], ec="none")
          for c in (NO_ROUTE, NO_HOVER, NO_DRAW)]
    ls_ = [f"can draw and lift, cannot FLY there   "
           f"({d['hover ok, unreachable']['pct']:.2f}%,  "
           f"{d['hover ok, unreachable']['m2']:.2f} m2)",
           f"can draw, cannot LIFT off it   "
           f"({d['draw ok, no hover']['pct']:.2f}%,  "
           f"{d['draw ok, no hover']['m2']:.2f} m2)",
           f"no certified DRAWING pose, any arm   "
           f"({d['no draw pose']['pct']:.2f}%,  "
           f"{d['no draw pose']['m2']:.2f} m2)"]
    if lp:
        hs.append(Line2D([], [], color="#0b6e4f", marker="o", ls="none",
                         ms=6, mfc="none", mew=1.4))
        ls_.append(f"...of which THE PEN'S CONE, not the canvas\n"
                   f"({lp['cells']} cells, {lp['m2']:.3f} m2:  certified at "
                   f"{min(float(k) for k in lp['min_lean_histogram']):g}"
                   f"-{max(float(k) for k in lp['min_lean_histogram']):g} deg, "
                   f"not at {lp['shipped_cone_deg']:g})")
    lg1 = fig.legend(hs, ls_, loc="upper left", bbox_to_anchor=(0.055, 0.090),
                     ncol=1, fontsize=8.8, frameon=False, handlelength=1.5,
                     handleheight=1.0, labelspacing=0.4, borderpad=0.0,
                     title="DEAD — the last layer the cell got through",
                     title_fontproperties=dict(size=9.2, weight="bold"))
    lg1._legend_box.align = "left"

    hs2 = [Line2D([], [], color=METAL, marker="x", ls="none", ms=7, mew=2),
           Line2D([], [], color="#8a1c1c", marker="P", ls="none", ms=9,
                  mec="white", mew=1.2),
           Line2D([], [], color=METAL, lw=2.2),
           Line2D([], [], color=METAL, lw=2.2, ls=(0, (4, 2)))]
    rp, rl = nums.get("largest_rect_portrait"), nums.get("largest_rect_landscape")
    ls2 = ["arm base: boom r = 0.10 m, plate 0.226 x 0.190 m",
           "where that arm's pen waits while another draws",
           "P  largest PORTRAIT artwork, zero dead spots"
           + (f"   ({rp['w']:.2f} x {rp['h']:.2f} m)" if rp else ""),
           "L  largest LANDSCAPE artwork, zero dead spots"
           + (f"   ({rl['w']:.2f} x {rl['h']:.2f} m)" if rl else "")]
    lg2 = fig.legend(hs2, ls2, loc="upper left",
                     bbox_to_anchor=(0.375, 0.090), ncol=1,
                     fontsize=8.8, frameon=False, handlelength=1.8,
                     labelspacing=0.4, borderpad=0.0,
                     title="THE RIG", title_fontproperties=dict(
                         size=9.2, weight="bold"))
    lg2._legend_box.align = "left"

    # ---- right column ------------------------------------------------------
    L = nums["layers"]
    ax1 = fig.add_subplot(gs[0, 1])
    lab = ["1. a certified DRAWING pose\n     (banded atlas, strict GO)",
           "2. + a certified HOVER over it\n     (v3 full-fiber lift)",
           "3. + the arm can FLY there\n     (routed from its park)"]
    val = [L["draw_pose_only"]["pct"], L["plus_hover"]["pct"],
           L["plus_reachability"]["pct"]]
    col = ["#9aa5b1", "#5b6b7d", FEAS_COLS[3]]
    hat = [None, None, None]
    note = "each layer is a subset of the one above it"
    # THE PROJECTION IS A SECOND MEASUREMENT AND IT IS DRAWN AS ONE.  Same
    # axis as layer 3, because it is a layer-3 number and the whole point is
    # that the two are comparable; hatched, because nobody may read it as a
    # margin this rig has been granted.
    pp = nums.get("post_survey_projection")
    if pp and pp.get("feasible_pct") is not None:
        lab.append("PROJECTION: the same map with the\n     30 mm "
                   "unsurveyed-base allowance out")
        val.append(float(pp["feasible_pct"]))
        col.append(FEAS_COLS[3])
        hat.append("////")
        d = pp.get("dead_by_cause") or {}
        rest = int(d.get("no draw pose", 0))
        note = ("each layer is a subset of the one above it;  the hatched bar "
                "is REPORT-ONLY —\nnobody has bought that survey"
                + (f", and all {rest} cells it still leaves are the "
                   "pen-cone cells below" if rest else ""))
    _bar_panel(ax1, lab, val, col, "What each layer costs, as % of canvas",
               note=note, hatch=hat)

    ax2 = fig.add_subplot(gs[1, 1])
    pa = nums["per_arm"]
    order = sorted(arms, key=lambda a: -pa[str(a)]["feasible_pct"])
    _bar_panel(ax2, [f"arm {a}  ({pa[str(a)]['name']})" for a in order],
               [pa[str(a)]["feasible_pct"] for a in order],
               [FEAS_COLS[3]] * len(order),
               "Canvas each arm can solo-draw, alone",
               note="they overlap: the union above is not the sum",
               xmax=max(pa[str(a)]["feasible_pct"] for a in order))

    ax3 = fig.add_subplot(gs[2, 1])
    hist = nums["histogram"]
    ks = [k for k in range(1, nmax + 1)]
    tot = nums["cells"]
    _bar_panel(ax3, [f"{k} arm" + ("" if k == 1 else "s") for k in ks],
               [100.0 * hist.get(str(k), 0) / tot for k in ks],
               [FEAS_COLS[k - 1] for k in ks],
               "Redundancy: how many arms reach each cell",
               note="a cell only one arm can reach has no fallback",
               xmax=max(100.0 * hist.get(str(k), 0) / tot for k in ks))

    ax4 = fig.add_subplot(gs[3, 1])
    th = nums["thirds"]
    _bar_panel(ax4, [f"{t['name']}\n  y {t['y_range'][0]:.2f} - "
                     f"{t['y_range'][1]:.2f} m" for t in th],
               [t["feasible_pct"] for t in th], [FEAS_COLS[3]] * 3,
               "Feasible by canvas third",
               note="the middle third is where the seam and the two "
                    "middle-row arms are")

    r = nums["largest_rect"]
    fig.suptitle(
        f"THE FEASIBLE DRAWING WORKSPACE  —  {nums['feasible_pct']:.1f}% of the "
        f"canvas, one arm at a time",
        fontsize=18, fontweight="bold", color=INK, x=0.055, y=0.972,
        ha="left")
    fig.text(0.055, 0.937,
             f"proposed rig, lateral pen  ·  h = {h:.3f} m  ·  pitch "
             f"{pitch:.2f} m  ·  canvas {sheet[0]:.4f} x {sheet[1]:.5f} m in "
             f"2 cm cells  ·  {nums['feasible_m2']:.2f} of "
             f"{nums['canvas_area_m2']:.2f} m2 drawable  ·  biggest clean "
             f"block {r['w']:.2f} x {r['h']:.2f} m",
             fontsize=11, color=INK2, ha="left")
    fig.text(0.055, 0.913,
             "SOLO: one arm inks while the other five hold their certified "
             "parks — an upper bound on what any concurrent schedule can ink.   "
             "NOT MODELLED: ceiling cross-members (steel undesigned), cable "
             "dress, fingertip cradle, paper transport.",
             fontsize=8.8, color=INK3, ha="left", style="italic")
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default=str(ATLAS_DIR))
    ap.add_argument("--out", default=str(OUT / "feasible_workspace"))
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 4))
    ap.add_argument("--every", type=int, default=1,
                    help="sweep every Nth certified cell (uniform pilot)")
    ap.add_argument("--chunk", type=int, default=24)
    ap.add_argument("--arms", default="")
    ap.add_argument("--no-redundant-hover", action="store_true",
                    help="stop at the first ladder rung, as a timeline would")
    ap.add_argument("--no-fiber-hover", action="store_true",
                    help="do not search the rest of the hover fiber when no "
                         "ladder rung can be flown to (the pre-2026-08-27 map)")
    ap.add_argument("--fiber-tries", type=int, default=12,
                    help="hover-fiber candidates a cell may spend on the "
                         "pen-up layer before it is called unreachable.  "
                         "Measured (out/guard_pocket.py) 12 finds 82.6 %% of "
                         "the ends an unbounded search finds and 48 finds "
                         "98.6 %%, so a budgeted map is a LOWER bound on the "
                         "drawable canvas and never an overstatement")
    ap.add_argument("--hover-lean-deg", type=float, default=0.0,
                    help="pen lean the FIBER search may use as its last rung, "
                         "matching the run's --tilt-max-deg (0 = vertical only)")
    ap.add_argument("--pitch", type=float, default=None,
                    help="transverse pair spacing; default is the shipped "
                         "0.61 and its baked parks. Any other value re-derives "
                         "the layout AND the park poses — pass a matching "
                         "--atlas swept at that pitch.")
    ap.add_argument("--sweep-atlas", action="store_true",
                    help="re-sweep the reachability atlas for --pitch into "
                         "--atlas first (needed for any pitch but the shipped "
                         "one: the committed 0.65 atlas predates the mesh "
                         "audit and its model signature is refused)")
    ap.add_argument("--rrt", type=float, default=0.0, metavar="SECONDS",
                    help="per-plan budget for the C-space pen-up planner "
                         "(aris_sixarm.transit), the tier below the shape "
                         "ladder.  0 disables it, which reproduces a "
                         "pre-2026-08-26 map exactly.  A cell refused under a "
                         "budget is refused UNDER THAT BUDGET and the map says "
                         "so.")
    ap.add_argument("--rrt-nodes", type=int, default=600,
                    help="nodes per tree for --rrt (default 600)")
    ap.add_argument("--from-raw", default=None, metavar="NPZ",
                    help="skip the sweep and rebuild the map, the numbers and "
                         "the figure from an existing _raw.npz (or a _ckpt.npz "
                         "of a run still going)")
    ap.add_argument("--ladder", default="", metavar="JSON[,JSON...]",
                    help="the runs whose ESCALATION RUNGS produced this "
                         "--from-raw npz, in the order they were spent.  A "
                         "rebuild does not re-walk the ladder, so without this "
                         "the map it writes cannot say what its own cells cost "
                         "— and a rung that turned 148 cells for 737 s and one "
                         "that turned none for 150 s are the whole argument "
                         "for where the ladder ends.  Each entry is stamped "
                         "with the file it was read from; nothing is "
                         "recomputed and nothing is summed.")
    ap.add_argument("--rescue", default="", metavar="RUNGS",
                    help="after loading/sweeping, walk the ESCALATION LADDER "
                         "over the cells the map refused and only those: a "
                         "comma-separated list of rung indices into "
                         "RESCUE_RUNGS (\"all\" for every rung).  A cell's "
                         "answer is the FIRST rung that certifies it; the "
                         "rungs above never see it.")
    ap.add_argument("--rescue-chunk", type=int, default=6)
    ap.add_argument("--projection", default="", metavar="JSON",
                    help="another run of this script — a `--calib` one — whose "
                         "headline is carried into this map's JSON as a "
                         "PROJECTION beside the shipped number.  It is never "
                         "added to anything: the map's own percentage is "
                         "measured at today's margins and the projection sits "
                         "next to it saying what a survey would be worth.")
    ap.add_argument("--lean-study", default=str(OUT / "lean_study.json"),
                    metavar="JSON",
                    help="scripts/lean_study.py's output.  The cells it names "
                         "are marked DISTINCTLY on the map and counted in "
                         "their own JSON block: they have no certified drawing "
                         "pose inside the shipped cone and do have one inside "
                         "a wider one, which is a decision pending and not a "
                         "canvas that ended.  They are never added to the "
                         "feasible number.")
    ap.add_argument("--calib", type=float, default=None, metavar="M",
                    help="REPORT-ONLY PROJECTION.  Replace the 0.03 m "
                         "unsurveyed-base allowance in the obstacle boxes "
                         "(mounts.MountModel.calib) and in the pair margin "
                         "(SAFETY_M + this) with another value; 0 asks what a "
                         "commissioning survey would buy.  The model is built "
                         "LOCALLY and no package constant is touched, so this "
                         "cannot leak into a shipped number — write it to its "
                         "own --out and quote it as a projection.")
    ap.add_argument("--h", type=float, default=None, metavar="M",
                    help="REPORT-ONLY.  Map a mounting height other than the "
                         "shipped `LAYOUT_PROPOSED['h']`.  `rig` has always "
                         "taken an `h`; only this CLI never passed one, so a "
                         "height study had to re-implement the map.  Nothing "
                         "in aris_sixarm/ is written — write it to its own "
                         "--out and quote it as a projection.")
    ap.add_argument("--legacy-bands", action="store_true",
                    help="reproduce the pre-2026-09-09 neighbour model exactly:"
                         " body columns as their bounding BOXES and no "
                         "pose-aware frozen partners.  For comparison runs.")
    ap.add_argument("--frozen-partners", action="store_true",
                    help="model every OTHER arm by its ACTUAL park capsules "
                         "instead of its pose-invariant body band (see "
                         "aris_sixarm/frozen.py).  A cell certified this way "
                         "DEPENDS on those arms holding those poses for the "
                         "whole stroke and both its pen-up legs; the JSON "
                         "records the dependency.  Off by default.")
    ap.add_argument("--parks", default=None, metavar="JSON",
                    help="a `scripts/height_sweep.py park` result whose depots "
                         "to use.  Needed with --h, because the shipped "
                         "(radius, hover, bearing) grid was searched at 0.940 "
                         "and does not transfer: at 0.850 it has no certified "
                         "ready pose for arm 97 at all.  Parked arms are "
                         "obstacles for the route layer, so mapping one height "
                         "with another's depots is not that height's map.")
    a = ap.parse_args()

    global FROZEN, CYLM
    if a.legacy_bands:
        FROZEN = CYLM = False
    print("neighbour model: body columns as %s ; partners %s"
          % ("CYLINDERS" if CYLM else "bounding boxes",
             "FROZEN at their parks" if FROZEN else "pose-invariant"))

    fl, parks, h, pitch = rig(a.pitch, a.h, a.calib)
    if a.parks:
        doc = json.load(open(a.parks))
        if not doc.get("certifies"):
            raise SystemExit(f"{a.parks}: that search found no fleet park set "
                             "clearing the gate; refusing to map on it")
        set_park_override({int(k): v["q"] for k, v in doc["best"].items()})
        fl, parks, h, pitch = rig(a.pitch, a.h, a.calib)
        print(f"parks from {a.parks} (searched at h = {doc['h']}) — in force "
              "for every worker, not just this header")
    arms = [int(v) for v in a.arms.split(",")] if a.arms else sorted(fl)
    ph = (layout.PARK_HOVER_PROPOSED
          if abs(pitch - SHIPPED_PITCH) < 1e-9 and a.h is None and not a.parks
          else park_hovers(fl, parks, h))
    print(f"rig=proposed tool=lateral PEN_LAT={frames.PEN_LAT} h={h} "
          f"pitch={pitch}"
          + ("  (SHIPPED: baked parks)" if abs(pitch - SHIPPED_PITCH) < 1e-9
             else "  (re-derived layout AND parks)"))
    print(f"STATIC_SAFE={paper.STATIC_SAFE} PAPER_SAFE={writing.PAPER_SAFE} "
          f"FRAME_FLOOR={paper.FRAME_FLOOR} SELF_SAFE={paper.SELF_SAFE}")
    # WHICH FLOOR THE INK WAS CERTIFIED AT, said out loud.  An atlas swept at
    # `rig_final.STATIC_MARGIN` offers cells the ROUTER may not be able to fly
    # to (`scripts/regate_atlas.py`); one swept at `paper.FRAME_FLOOR` does not.
    # A map that does not say which it read is a map nobody can reproduce.
    atlas_floor = None
    try:
        _m = atlas.load(Path(a.atlas), sorted(fl)[0])[1]
        atlas_floor = (float(_m["static_margin"])
                       if "static_margin" in _m.files
                       else float(__import__("aris_sixarm").rig_final
                                  .STATIC_MARGIN))
    except Exception:
        pass
    print(f"atlas {a.atlas}: drawing poses gated at "
          + (f"{1000 * atlas_floor:.0f} mm" if atlas_floor else "unknown")
          + (f"  (the router's own floor)" if atlas_floor
             and abs(atlas_floor - paper.FRAME_FLOOR) < 1e-9
             else "  (the CHECKER's floor: an optimistic prefilter, see "
                  "scripts/regate_atlas.py)"))
    print("RRT tier: " + (f"ON, {a.rrt:.2f} s x 1 attempt, "
                          f"{a.rrt_nodes} nodes/tree, "
                          f"{RRT_CELL_PLANS} plans/cell, park-probed"
                          if a.rrt > 0 else "OFF"))
    if a.rescue:
        print("ESCALATION LADDER: " + " | ".join(
            f"{i}:{r[0]}(tries {r[1]}, plans {r[2]}, nodes {r[3]}, "
            f"seeds {r[4]}, aside {r[5]})"
            for i, r in enumerate(RESCUE_RUNGS)))
        print(f"  running rungs {a.rescue}; "
              f"transit.TIME_BUDGET is pinned at {RESCUE_TIME:.0f} s so the "
              "CLOCK never decides an answer (the deadline count is reported)")

    if a.sweep_atlas:
        # ONE ATLAS PER COLLISION MODEL, and the model is this build's.
        # `atlas.sweep_arm` stamps `model_signature` into every npz, so a
        # comparison rig gets a comparison atlas rather than a stale one.
        import multiprocessing as mp
        d = Path(a.atlas)
        d.mkdir(parents=True, exist_ok=True)
        print(f"sweeping atlas for pitch {pitch} into {d}", flush=True)
        ta = time.time()
        with mp.get_context("fork").Pool(min(6, a.workers)) as pool:
            pool.starmap(atlas.sweep_arm,
                         [(x, str(d), GRID, 1.05, h, 15.0, fl[x].pen, fl,
                           SHEET, frames.PEN_LAT) for x in arms])
        print(f"atlas {time.time() - ta:.0f}s", flush=True)

    t0 = time.time()
    if a.from_raw:
        z = np.load(a.from_raw)
        per_arm = {int(k[3:]): z[k] for k in z.files}
        arms = sorted(per_arm)
        print(f"rebuilt from {a.from_raw}: "
              f"{sum(len(v) for v in per_arm.values())} arm-cells")
    else:
        per_arm = sweep(arms, Path(a.atlas), h, a.workers, a.chunk, a.every,
                        not a.no_redundant_hover, ckpt=a.out + "_ckpt.npz",
                        pitch=pitch, fiber=not a.no_fiber_hover,
                        lean=a.hover_lean_deg, tries=a.fiber_tries,
                        rrt=a.rrt, rrt_nodes=a.rrt_nodes)
        print(f"sweep {time.time() - t0:.0f}s", flush=True)
        np.savez_compressed(a.out + "_raw.npz",
                            **{f"arm{k}": v for k, v in per_arm.items()})

    ladder = []
    if a.rescue and a.every <= 1:
        rungs = (list(range(len(RESCUE_RUNGS))) if a.rescue.strip() == "all"
                 else [int(v) for v in a.rescue.split(",") if v.strip() != ""])
        tr = time.time()
        per_arm, ladder = rescue(per_arm, arms, Path(a.atlas), h, a.workers,
                                 rungs=rungs, chunk=a.rescue_chunk,
                                 ckpt=a.out + "_ckpt.npz", pitch=pitch,
                                 lean=a.hover_lean_deg,
                                 rrt=max(a.rrt, 1e-9), calib=a.calib)
        print(f"rescue {time.time() - tr:.0f}s", flush=True)
        np.savez_compressed(a.out + "_raw.npz",
                            **{f"arm{k}": v for k, v in per_arm.items()})

    if a.every > 1:
        # A PILOT IS NOT A MAP.  Every Nth certified cell says what fraction of
        # the CERTIFIED cells survive the two layers above the atlas; it cannot
        # say what fraction of the CANVAS does, because the cells it skipped are
        # indistinguishable from cells no arm can draw.
        print(f"\nPILOT (every {a.every}th cell) — per-arm-cell rates, "
              f"NOT canvas coverage")
        allc = np.vstack([v for v in per_arm.values() if len(v)])
        for arm in arms:
            d = per_arm[arm]
            c = d[:, 2].astype(int)
            print(f"  arm {arm}: {len(d)} cells  feasible "
                  f"{100.0 * (c == FEASIBLE).mean():5.1f}%  "
                  f"no-hover {100.0 * (c == NO_HOVER).mean():4.1f}%  "
                  f"no-route {100.0 * (c == NO_ROUTE).mean():5.1f}%")
        c = allc[:, 2].astype(int)
        print(f"  ALL {len(allc)} arm-cells: feasible "
              f"{100.0 * (c == FEASIBLE).mean():.2f}%, "
              f"no-hover {100.0 * (c == NO_HOVER).mean():.2f}%, "
              f"no-route {100.0 * (c == NO_ROUTE).mean():.2f}%")
        return

    lean = None
    if a.lean_study:
        p = Path(a.lean_study)
        if p.exists():
            with open(p) as f:
                lean = json.load(f)
            print(f"lean study: {lean['cells_no_pose_at_15']} cells with no "
                  f"pose at {lean['shipped_cone_deg']:g} deg, "
                  f"{lean['cells_with_a_pose_in_the_wider_cone']} of them "
                  f"certified inside {max(lean['cone']):g} deg")
        else:
            print(f"lean study: {p} not found; the >cone set is not marked")

    comp = compose(per_arm, arms)
    nums = numbers(comp, per_arm, arms, fleet=fl, park_hover=ph, pitch=pitch,
                   h=h, lean=lean)
    nums["sweep_seconds"] = round(time.time() - t0, 1)
    nums["atlas"] = dict(
        dir=str(a.atlas),
        pose_static_floor_m=atlas_floor,
        route_static_floor_m=float(paper.FRAME_FLOOR),
        note=("the floor a DRAWING pose was gated at, and the one every pen-up "
              "LEG is held to.  They were 50 and 63 mm for every map before "
              "v12, and `paper.effective_static_floor` clamps the difference "
              "away rather than contradicting it — which leaves a pose on the "
              "atlas gate with nothing for its own descent to spend."))
    # THE DEPENDENCY, RECORDED WITH THE RESULT.  A cell certified against a
    # partner's actual parked capsules is certified ONLY while that partner
    # holds that pose — for the whole stroke and both its pen-up legs.  That is
    # an obligation on the conductor, so it travels with the number.
    nums["neighbour_model"] = dict(
        body_columns=("CYLINDERS (aris_sixarm/envelope.py)" if CYLM
                      else "bounding boxes (pre-2026-09-09)"),
        partners=("frozen at their certified parks" if FROZEN
                  else "pose-invariant bands"),
        note=("the AABB of a band padded a 128.5 mm cylinder into a 449 mm "
              "box -- 160 mm of nothing below where the arm's body ends, "
              "which is where a neighbour's forearm passes under a base.  "
              "The cylinder is the same measured profile, exactly."))
    if FROZEN:
        # the workers froze in `_init`; the PARENT has to as well, or the block
        # it writes names no arms at all
        frozen.freeze(parks, fl, {a: fl[a].pen for a in fl}, h)
        frozen_qs = frozen.poses()
        nums["frozen_dependency"] = dict(
            model="pose-aware neighbour capsules (aris_sixarm/frozen.py)",
            arms=sorted(int(k) for k in frozen_qs),
            poses={str(k): [round(float(v), 6) for v in q]
                   for k, q in frozen_qs.items()},
            parks_source=(str(a.parks) if a.parks
                          else "layout.Q_PARK_PROPOSED"),
            replaced="body:<aid>_column<k> bands of the named arms only",
            kept="mounts, plates, drop cluster, runway, and the bands of any "
                 "arm NOT in this list",
            floor_m=float(paper.FRAME_FLOOR),
            obligation=("every cell in this map is certified ONLY while each "
                        "named arm holds its named pose for the whole stroke "
                        "and both of its pen-up legs.  The conductor enforces "
                        "this with its phase freeze — `scene_check` reports "
                        "'frozen N/N' over the merged timeline, and "
                        "`allocate.ParkProbe` is what holds a parked partner "
                        "to the pair margin while another arm draws."))
    if not ladder and a.ladder:
        # A REBUILD DOES NOT RE-WALK THE LADDER, and the rungs a map's cells
        # were actually bought with are provenance, not decoration.  Carried
        # across verbatim, each rung stamped with the run that spent it.
        for p in [Path(v) for v in a.ladder.split(",") if v.strip()]:
            if not p.exists():
                print(f"ladder: {p} not found")
                continue
            with open(p) as f:
                lj = json.load(f)
            got = lj.get("escalation_ladder") or []
            ladder += [dict(r, from_run=str(p)) for r in got]
            print(f"ladder: {len(got)} rung(s) carried from {p}")
        ladder.sort(key=lambda r: r.get("rung", 0))
    if ladder:
        nums["escalation_ladder"] = ladder
        nums["escalation_rungs"] = [dict(zip(
            ("name", "tries", "plans", "nodes", "attempts", "aside"), r))
            for r in RESCUE_RUNGS]
    if a.projection:
        p = Path(a.projection)
        if p.exists():
            with open(p) as f:
                pj = json.load(f)
            nums["post_survey_projection"] = dict(
                source=str(p),
                feasible_pct=pj.get("feasible_pct"),
                feasible_cells=pj.get("feasible_cells"),
                dead_by_cause={k: v["cells"]
                               for k, v in pj.get("dead_by_cause", {}).items()},
                calibration=pj.get("calibration_projection"),
                note="MEASURED, NOT SHIPPED.  The same map with the 30 mm "
                     "unsurveyed-base allowance out of the obstacle boxes and "
                     "out of the pair margin.  The number above it is this "
                     "rig's canvas at today's margins; this is what a "
                     "commissioning survey of the six base positions would "
                     "buy, and nobody has bought it.")
            print(f"post-survey projection ({p}): "
                  f"{pj.get('feasible_pct')}%  against this map's "
                  f"{nums['feasible_pct']}%")
        else:
            print(f"projection: {p} not found")
    if a.calib is not None:
        from aris_sixarm import mounts, coordination
        nums["calibration_projection"] = dict(
            calib_m=float(a.calib), shipped_calib_m=float(mounts.MOUNTS.calib),
            pair_margin_m=float(coordination.SAFETY_M + a.calib),
            shipped_pair_margin_m=float(coordination.SAFETY_M
                                        + coordination.CALIB_M),
            what_moved=("the obstacle boxes every neighbour's body column is "
                        "carried as, and the pair margin the parked fleet is "
                        "held to"),
            what_did_not=("rig_final.STATIC_MARGIN, and therefore the atlas "
                          "gate every DRAWING pose was certified at: this "
                          "projection moves the hover and the route, not the "
                          "ink"),
            note="REPORT-ONLY.  Not a shipped number and not a margin anyone "
                 "has bought; it is what a commissioning survey of the six "
                 "base positions would be worth if it removed the allowance "
                 "entirely.")
    nums["caveats"] = [
        "Ceiling cross-members are NOT modelled: the steel that carries the six "
        "booms has no design yet, so only the booms themselves (r = 0.10 m, "
        "gated as circumscribed squares) and their base plates are obstacles.",
        "Cable dress is NOT modelled: no umbilical, drag chain or festoon "
        "occupies any of this volume, and every arm is a bare kinematic chain "
        "plus its measured link capsules.",
        "The fingertip cradle is NOT modelled: the pen is the certified "
        "tip = TCP + R @ (0.110, 0, 0.110) lateral holder and nothing around it.",
        "Paper transport, table frame and operator access are NOT modelled.",
        "SOLO only: this is one arm drawing with five parked. It is an upper "
        "bound on what any concurrent schedule can ink.",
        "The router plans against the metal and is then VETOED by the parked "
        "partners rather than routing around them, so a park-aware router could "
        "only recover cells, never lose them.",
    ]
    with open(a.out + ".json", "w") as f:
        json.dump(nums, f, indent=2)
    np.savez_compressed(a.out + "_map.npz", n_arms=comp["n_arms"],
                        cause=comp["cause"], draw_any=comp["draw_any"],
                        hover_any=comp["hover_any"], xs=comp["xs"],
                        ys=comp["ys"],
                        **{f"mask{k}": v for k, v in comp["per_arm"].items()})
    figure(comp, nums, arms, a.out + ".png", fleet=fl, park_hover=ph,
           lean=lean)

    print(f"\nFEASIBLE {nums['feasible_pct']:.2f}%  "
          f"({nums['feasible_m2']:.3f} / {nums['canvas_area_m2']:.3f} m2)")
    for k, v in nums["layers"].items():
        print(f"  {k:>18}: {v['pct']:6.2f}%  {v['m2']:.3f} m2")
    for k, v in nums["dead_by_cause"].items():
        print(f"  DEAD {k:>22}: {v['pct']:6.2f}%  {v['m2']:.3f} m2")
    for k, v in nums["per_arm"].items():
        print(f"  arm {k:>3}: {v['feasible_pct']:6.2f}%  "
              f"(draw {v['draw_pct']:.2f}%, -{v['lost_to_hover']} hover, "
              f"-{v['lost_to_route']} route)")
    for k in ("largest_rect", "largest_rect_portrait",
              "largest_rect_landscape"):
        r = nums.get(k)
        if r:
            print(f"  {k:>24}: {r['w']:.2f} x {r['h']:.2f} m "
                  f"= {r['area_m2']:.3f} m2 at ({r['x0']:.2f}, {r['y0']:.2f})")
    ds = nums["dead_split"]
    print(f"  DEAD {ds['total']['m2']:.3f} m2 splits as: "
          f"{ds['under_base_discs_r30']['m2']:.3f} under the six base discs "
          f"(r=0.30), {ds['middle_third_outside_the_discs']['m2']:.3f} in the "
          f"middle third outside them "
          f"(of which {ds['middle_third_route_dead']['m2']:.3f} route-dead), "
          f"{ds['elsewhere']['m2']:.3f} elsewhere")
    print(f"  wrote {a.out}.png / .json")


if __name__ == "__main__":
    main()
