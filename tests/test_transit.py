"""The CONFIGURATION-SPACE pen-up planner: determinism, certification, order.

WHAT THIS FILE IS DEFENDING.  `aris_sixarm.transit` is the first randomised
search in this package, and it lives inside a pipeline whose whole discipline
is that the same question produces the same answer — `tests/test_balance.py`
asserts a cost matrix is bit-identical on one core and on four, and
`scripts/csail_schedule.py`'s `cross_check` raises if the seconds the
sequencer priced and the seconds the timeline pays differ by 1e-6.  A planner
seeded from entropy, or from `hash()` (which is salted per process), breaks
both of those and does it intermittently.  So the seed is a pure function of
the question, and the first four tests here are about nothing else.

The rest are about the two orderings the tier has to preserve:

  * THE LADDER GOES FIRST, AND UNCHANGED.  Every crossing the shape ladder
    settles must settle the same way, to the bit, with the planner available —
    otherwise every pinned number in the corpus moved for a tier that was
    never asked.
  * THE PRODUCER IS TIGHTER THAN THE CHECKER.  A path this planner returns is
    certified by `paper.route`'s own `legs_ok` and then re-derived here by the
    INDEPENDENT checkers — `scene_check.static_clearance_lb`, its own self
    model, its own paper gate — which is the same treatment every other route
    in this package gets.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from aris_sixarm import fleet as fleet_mod
from aris_sixarm import paper, rig_final, scene_check, selfcoll, transit, writing
from aris_sixarm.fleet import FLEET
from aris_sixarm.frames import joint_margin

ROOT = Path(__file__).parents[1]

# The same probe set `tests/test_paper.py` uses on the merged canvas: hover
# points spanning arm 2's mirror plane at y = 1.81532, where its crossings get
# into trouble.  A planner test on a happy-path pair proves nothing.
PROBE_XY = [(0.80, 1.60), (0.90, 1.70), (0.95, 1.55), (0.95, 1.93),
            (1.00, 1.80), (1.10, 1.65), (0.85, 1.90)]
PEN = 0.110


@pytest.fixture()
def six():
    """The six-arm extended-pole rig, restored afterwards.

    `paper`'s memos are keyed on `id(spec)` and the fleet dict is mutated in
    place, so they are dropped on the way in and on the way out — and so is
    `transit`'s tally, which these tests read.
    """
    before = fleet_mod.ACTIVE_RIG
    paper.clear_cache()
    transit.reset_stats()
    fleet_mod.activate("final6_opt")
    try:
        yield fleet_mod.FLEET
    finally:
        fleet_mod.activate(before)
        paper.clear_cache()
        transit.reset_stats()


def _hover(spec, xy, z=writing.LIFT_Z, pen=PEN):
    q, _ = writing.lifted_config(spec, np.asarray(spec.q_seed, float), xy, z=z,
                                 pen_ext=pen)
    return q


def _gate(spec, pen=PEN, **kw):
    """The gate a real route pays, for a pair of real hovers."""
    return transit.Gate(spec, pen, boxes=paper.static_boxes(spec),
                        static_floor=paper.FRAME_FLOOR,
                        self_floor=selfcoll.SELF_PLAN_MARGIN, **kw)


# ==========================================================================
# 1. determinism — the property the pipeline cannot survive without
# ==========================================================================
def test_the_seed_is_a_pure_function_of_the_question(six):
    """Same question, same seed; any change to the question, a different one."""
    spec = FLEET[2]
    boxes = paper.static_boxes(spec)
    floors = (0.02, 0.02, 0.063, 0.023, 0.0)
    sig = transit.scene_signature(spec, boxes, PEN, 1.0, floors)
    assert sig == transit.scene_signature(spec, boxes, PEN, 1.0, floors)
    q0 = np.asarray(spec.q_seed, float)
    q1 = q0 + 0.1
    s = transit.seed_for(sig, q0, q1)
    assert s == transit.seed_for(sig, q0, q1)
    assert s != transit.seed_for(sig, q1, q0), "direction is part of the question"
    assert s != transit.seed_for(sig, q0, q1, attempt=1)
    assert s != transit.seed_for(
        transit.scene_signature(spec, boxes, PEN, 1.0,
                                (0.02, 0.02, 0.064, 0.023, 0.0)), q0, q1), \
        "a different floor is a different question"
    # ...and a box that MOVED re-seeds, because the tree would be a different
    # tree.  A box list in a different ORDER does not: it is the same room.
    moved = [dict(b) for b in boxes]
    moved[0] = dict(lo=np.asarray(moved[0]["lo"], float) + 0.01,
                    hi=np.asarray(moved[0]["hi"], float) + 0.01)
    assert transit.scene_signature(spec, moved, PEN, 1.0, floors) != sig
    assert transit.scene_signature(spec, boxes[::-1], PEN, 1.0, floors) == sig


def test_the_signature_does_not_depend_on_the_object_address(six):
    """`id(spec)` is fine for a memo and fatal for a seed.

    `sequence` screens crossings in a fork pool and the worker's copy of an arm
    is a different object at a different address; a seed drawn from `id` would
    grow a different tree in every worker and the bit-identity test in
    `tests/test_balance.py` would fail intermittently, which is the worst way
    for it to fail.
    """
    import copy
    spec = FLEET[2]
    twin = copy.deepcopy(spec)
    assert id(twin) != id(spec)
    boxes = paper.static_boxes(spec)
    floors = (0.02, 0.02, 0.063, 0.023, 0.0)
    assert transit.scene_signature(twin, paper.static_boxes(twin), PEN, 1.0,
                                   floors) \
        == transit.scene_signature(spec, boxes, PEN, 1.0, floors)


def test_the_same_question_plans_the_same_path_twice(six):
    """Bit-identical, memo bypassed, tally reset in between."""
    spec = FLEET[2]
    a = _hover(spec, (0.95, 1.55))
    b = _hover(spec, (0.95, 1.93))
    if a is None or b is None:
        pytest.skip("no hover IK for the probe pair on this rig")
    g = _gate(spec)
    one = transit.plan(spec, a, b, pen_ext=PEN, gate=g, time_budget=1.0,
                       max_nodes=200)
    two = transit.plan(spec, a, b, pen_ext=PEN, gate=_gate(spec),
                       time_budget=1.0, max_nodes=200)
    assert (one is None) == (two is None)
    if one is None:
        pytest.skip("this pair needs no plan on this rig")
    assert len(one) == len(two)
    for u, v in zip(one, two):
        assert np.array_equal(u, v), "the same question planned a different path"


def test_the_same_question_plans_the_same_path_in_another_process(six):
    """PYTHONHASHSEED is per process, and this is what catches a `hash()`.

    Run in a cold interpreter with a deliberately different hash salt: the
    digest of the planned path must be the one this process gets.
    """
    spec = FLEET[2]
    a = _hover(spec, (0.95, 1.55))
    b = _hover(spec, (0.95, 1.93))
    if a is None or b is None:
        pytest.skip("no hover IK for the probe pair on this rig")
    here = transit.plan(spec, a, b, pen_ext=PEN, gate=_gate(spec),
                        time_budget=1.0, max_nodes=200)
    if here is None:
        pytest.skip("this pair needs no plan on this rig")
    src = f"""
import hashlib, numpy as np
from aris_sixarm import fleet, paper, selfcoll, transit
fleet.activate("final6_opt")
spec = fleet.FLEET[2]
a = np.frombuffer(bytes.fromhex("{np.asarray(a, float).tobytes().hex()}"))
b = np.frombuffer(bytes.fromhex("{np.asarray(b, float).tobytes().hex()}"))
g = transit.Gate(spec, {PEN}, boxes=paper.static_boxes(spec),
                 static_floor=paper.FRAME_FLOOR,
                 self_floor=selfcoll.SELF_PLAN_MARGIN)
p = transit.plan(spec, a, b, pen_ext={PEN}, gate=g, time_budget=1.0,
                 max_nodes=200)
print(hashlib.blake2b(np.stack(p).tobytes() if p else b"", digest_size=8).hexdigest())
"""
    env = {"PYTHONHASHSEED": "12345", "PATH": "/usr/bin:/bin",
           "PYTHONPATH": str(ROOT)}
    out = subprocess.run([sys.executable, "-c", src], capture_output=True,
                         text=True, env=env, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    import hashlib
    mine = hashlib.blake2b(np.stack(here).tobytes(), digest_size=8).hexdigest()
    assert out.stdout.strip() == mine, "a cold process planned a different path"


# ==========================================================================
# 2. certification — the producer is tighter than every checker that grades it
# ==========================================================================
def _planned(spec, a, b, **kw):
    g = _gate(spec)
    p = transit.plan(spec, a, b, pen_ext=PEN, gate=g, time_budget=1.5,
                     max_nodes=400, **kw)
    return None if p is None else [np.asarray(a, float)] + list(p) + \
        [np.asarray(b, float)]


def test_a_planned_path_passes_the_independent_checkers(six):
    """`scene_check`'s derivations, not `paper`'s, on the planner's output.

    This is the ordering the whole package rests on: the producer computes a
    bound at a density it chooses and holds it to a floor 13 mm above the
    checker's, and the checker — a separate derivation, with its own segment
    sampling and its own half-step slack — has to agree.  Anything else and
    the second opinion is a lottery.
    """
    spec = FLEET[2]
    n_checked = 0
    hs = [_hover(spec, p) for p in PROBE_XY]
    hs = [q for q in hs if q is not None]
    for a in hs:
        for b in hs:
            if a is b:
                continue
            path = _planned(spec, a, b)
            if path is None:
                continue
            n_checked += 1
            dense = np.vstack([paper.line_samples(u, v, 65)
                               for u, v in zip(path[:-1], path[1:])])
            # the paper, the checker's way
            P = paper.world_chain(dense, spec, PEN)
            cc = list(range(1, 9)) + ([10] if P.shape[1] >= 11 else [])
            assert P[:, cc, 2].min() >= scene_check.PAPER_CHAIN - 1e-9
            assert P[:, 9, 2].min() >= -scene_check.PAPER_TIP - 1e-9
            # the metal, `scene_check`'s own lower bound
            boxes = paper.static_boxes(spec)
            if boxes:
                lb = float(scene_check.static_clearance_lb(P, boxes).min())
                assert lb >= rig_final.STATIC_MARGIN - 1e-9, (
                    f"planner cleared the steel by {1000 * lb:.1f} mm against "
                    f"the checker's {1000 * rig_final.STATIC_MARGIN:.0f}")
            # the arm against itself, `scene_check`'s own model
            sc = float(scene_check.self_clearance(dense, PEN).min())
            assert sc >= scene_check.SELF_MARGIN - 1e-9, (
                f"planner folded the arm to {1000 * sc:.1f} mm")
    if not n_checked:
        pytest.skip("no pair on this rig needed the planner")


def test_every_planned_via_is_a_pose_the_arm_may_stand_in(six):
    """`tests/test_paper.py` pins this for the ladder; it holds here too.

    A ladder via is a `hover_solve` IK solution gated on `HOVER_MARGIN`.  A
    planner via is a point in the joint box and nothing would enforce it, so
    `transit.VIA_MARGIN` does — on the sampler, on every node a CONNECT keeps,
    and therefore on every via that comes out.
    """
    spec = FLEET[2]
    hs = [_hover(spec, p) for p in PROBE_XY]
    hs = [q for q in hs if q is not None]
    n = 0
    for a in hs:
        for b in hs:
            if a is b:
                continue
            p = transit.plan(spec, a, b, pen_ext=PEN, gate=_gate(spec),
                             time_budget=1.0, max_nodes=250)
            if not p:
                continue
            n += 1
            for v in p:
                assert joint_margin(np.asarray(v, float)) \
                    >= writing.HOVER_MARGIN - 1e-9
    if not n:
        pytest.skip("no pair on this rig needed the planner")


def test_shortcutting_cannot_loosen_a_gate(six):
    """Smoothing is re-certified, not assumed.

    A shortcut replaces k legs with one, and the one is longer than any of
    them: the sampled bound on it is a different number computed over a
    different set of configurations.  Every accepted shortcut goes through the
    same `line_prefix` the search does, so this asserts the property rather
    than trusting the loop.
    """
    rng = np.random.default_rng(3)
    spec = FLEET[2]
    g = _gate(spec)
    a = _hover(spec, (0.95, 1.55))
    b = _hover(spec, (0.95, 1.93))
    if a is None or b is None:
        pytest.skip("no hover IK for the probe pair on this rig")
    raw = transit._rrt_connect(g, np.asarray(a, float), np.asarray(b, float),
                               rng, transit.FR3_MIN + transit.LIMIT_MARGIN,
                               transit.FR3_MAX - transit.LIMIT_MARGIN,
                               transit.STEP, 300,
                               __import__("time").perf_counter() + 2.0)
    if raw is None:
        pytest.skip("no tree on this pair inside the test budget")
    short = transit._shortcut(g, raw, rng, transit.STEP)
    assert len(short) <= len(raw)
    assert np.array_equal(short[0], raw[0]) and np.array_equal(short[-1],
                                                               raw[-1])
    for u, v in zip(short[:-1], short[1:]):
        assert g.line_prefix(u, v, transit.STEP)[1], \
            "a shortcut leg does not certify at the gate that accepted it"


def test_the_planner_refuses_a_gate_nothing_can_meet(six):
    """A budget spent is reported as a refusal, not as a path.

    Asked to keep half a metre of clearance from the paper — which no pose in
    the room has — the planner must come back `None` inside its budget rather
    than return something it did not certify.
    """
    spec = FLEET[2]
    a = _hover(spec, (0.95, 1.55))
    b = _hover(spec, (0.95, 1.93))
    if a is None or b is None:
        pytest.skip("no hover IK for the probe pair on this rig")
    g = transit.Gate(spec, PEN, boxes=paper.static_boxes(spec),
                     chain_floor=0.50, tip_floor=0.50,
                     static_floor=paper.FRAME_FLOOR,
                     self_floor=selfcoll.SELF_PLAN_MARGIN)
    assert transit.plan(spec, a, b, pen_ext=PEN, gate=g, time_budget=0.5,
                        max_nodes=120, attempts=1) is None


# ==========================================================================
# 3. the order of the tiers, and the memo that remembers which ran
# ==========================================================================
def test_the_ladder_still_wins_first_and_bit_identically(six):
    """Every crossing the ladder settles settles the same way, to the bit.

    This is the test that protects the whole corpus.  The planner is a tier
    BELOW forty shapes that were pinning numbers before it existed; if turning
    it on moved any of their answers, every measurement in `docs/` would have
    to be re-earned for a search that was never even called.
    """
    spec = FLEET[2]
    hs = [_hover(spec, p) for p in PROBE_XY]
    hs = [q for q in hs if q is not None]
    assert len(hs) >= 4

    def sweep(flag):
        before = paper.RRT_SAFE
        paper.RRT_SAFE = flag
        paper.clear_cache()
        try:
            out = {}
            for i, a in enumerate(hs):
                for j, b in enumerate(hs):
                    if i == j:
                        continue
                    out[(i, j)] = paper.route(spec, a, b, pen_ext=PEN,
                                              q_home=spec.q_seed)
            return out
        finally:
            paper.RRT_SAFE = before
            paper.clear_cache()

    off, on = sweep(False), sweep(True)
    n_same = n_new = 0
    for k, r0 in off.items():
        r1 = on[k]
        if r0 is None:
            # the only permitted difference: a crossing the ladder refused
            assert r1 is None or r1["mode"].startswith("rrt")
            n_new += r1 is not None
            continue
        assert r1 is not None, "the planner cost the ladder an answer"
        assert r0["mode"] == r1["mode"], f"{k}: {r0['mode']} -> {r1['mode']}"
        assert len(r0["vias"]) == len(r1["vias"])
        for u, v in zip(r0["vias"], r1["vias"]):
            assert np.array_equal(u, v), f"{k}: the ladder's via moved"
        assert r0["chain_z"] == r1["chain_z"] and r0["tip_z"] == r1["tip_z"]
        assert r0["tried"] == r1["tried"], "the ladder walked a different ladder"
        n_same += 1
    assert n_same > 0, "nothing was routed, so nothing was compared"


def test_the_tier_is_part_of_the_memo_key(six):
    """The q_home lesson, for the fourth time.

    A pair has two answers now — `None` without the planner and a path with it
    — so a memo that filed them under one key would hand whichever caller
    asked first to every caller after it.  `paper._key` and `paper.key_maker`
    have to say the same thing about it, or the sequencer's parallel screen
    files answers where nothing looks for them.
    """
    spec = FLEET[2]
    a = np.asarray(spec.q_seed, float)
    b = a + 0.2
    args = (spec, a, b, PEN, 1.0, 0.02, 0.02)
    before = paper.RRT_SAFE
    try:
        paper.RRT_SAFE = True
        on = paper._key(*args)
        km_on = paper.key_maker(spec, [a], [b], PEN, 1.0)(0, 0, 0.02, 0.02)
        paper.RRT_SAFE = False
        off = paper._key(*args)
        km_off = paper.key_maker(spec, [a], [b], PEN, 1.0)(0, 0, 0.02, 0.02)
    finally:
        paper.RRT_SAFE = before
    assert on != off
    assert km_on == on and km_off == off, \
        "the block key and the single key disagree about the tier"
    # ...and the recursion's sub-routes, which never get the tier, share the
    # tier-off memo rather than splitting it
    paper.RRT_SAFE = True
    try:
        assert paper._key(*args, rrt=False) == off
    finally:
        paper.RRT_SAFE = before


def test_a_probe_is_part_of_the_memo_key_too(six):
    """A fourth obstacle in the room is a different question."""
    spec = FLEET[2]
    a = np.asarray(spec.q_seed, float)
    b = a + 0.2
    args = (spec, a, b, PEN, 1.0, 0.02, 0.02)
    before = paper.RRT_PROBE
    try:
        bare = paper._key(*args)
        paper.RRT_PROBE = (lambda qs, sweep: 1.0, 0.08, ("park", 1))
        with_probe = paper._key(*args)
        paper.RRT_PROBE = (lambda qs, sweep: 1.0, 0.08, ("park", 2))
        other = paper._key(*args)
    finally:
        paper.RRT_PROBE = before
    assert len({bare, with_probe, other}) == 3


def test_a_route_the_planner_found_is_priced_and_flown_the_same_way(six):
    """The sequencer prices what the timeline pays, planner or not.

    `sequence` and `writing._route` read the SAME memo entry, so a planned
    transit is priced hop by hop at `_dq_time` exactly as a skirt is.  This
    pins the arithmetic rather than the seconds: whatever the vias are, the
    beat's total is the sum of its capped hops.
    """
    spec = FLEET[2]
    a = _hover(spec, (0.95, 1.55))
    b = _hover(spec, (0.95, 1.93))
    if a is None or b is None:
        pytest.skip("no hover IK for the probe pair on this rig")
    r = paper.route(spec, a, b, pen_ext=PEN, q_home=spec.q_seed)
    if r is None:
        pytest.skip("unroutable on this rig")
    qs = [a] + list(r["vias"]) + [b]
    beat = writing._beat(qs, writing.QD_FRAC, 0.0)
    assert len(beat) == len(qs) - 1
    for (dt, v), u in zip(beat, qs[:-1]):
        assert dt >= writing._dq_time(u, v, writing.QD_FRAC) - 1e-12
    assert np.array_equal(np.asarray(beat[-1][1], float), np.asarray(b, float))
