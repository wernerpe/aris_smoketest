"""The incremental improvement loop: same answers, a fraction of the clock.

Run: python3 tests/test_balance.py     (or pytest tests/)

`docs/FAST_PLANNING.md` is what these pin.  The balancer used to rebuild an
arm's whole cost matrix and re-solve its tour from scratch for every candidate
it priced, scan every relocation and every swap before taking one, and have no
idea when it had finished; on the Trollface that was 676.8 s of an 826 s
allocation and it ended with `0 moves and 0 splits taken`.

Three claims are tested here and none of them needs the robot:

  THE SLICE IS THE MATRIX.  `allocate._ArmMatrix` keeps one growing matrix per
  arm and hands out submatrices of it.  Every cell of the submatrix must equal
  the cell `sequence.cost_matrix` would have built for that bag alone, under
  both cost models, and growing the union must not disturb what was in it.

  FIRST IMPROVEMENT IS STILL IMPROVEMENT.  `balance_loads` scanned every
  candidate to find the best; it now takes the first that helps, in an order
  predicted from the ink.  On synthetic instances it must reach a maximum load
  no worse than the exhaustive scan's by more than the tolerance the mandate
  allows, and never worse than where it started.

  THE FLOOR IS A FLOOR.  `_phase_floor` is a lower bound on the busiest arm of
  ANY assignment, so stopping within epsilon of it cannot be leaving an
  improvement on the table that the loop would have found.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from aris_sixarm import allocate, sequence                      # noqa: E402
from aris_sixarm.fleet import FLEET_SIXARM as FLEET             # noqa: E402

FLOOR = FLEET[13]


# ==========================================================================
# 1. the growing matrix hands out exactly the matrix that was there before
# ==========================================================================
def _seg_ends(rng):
    """One segment's `sequence.endpoints`, without going near the IK."""
    lo, hi = -1.5, 1.5
    return dict(n=1, q=rng.uniform(lo, hi, (1, 2, 7)),
                xy=rng.uniform(0.0, 2.0, (1, 2, 2)),
                hover=rng.uniform(lo, hi, (1, 2, 7)),
                z=np.full((1, 2), 0.06))


def _cluster_seg_ends(rng, nv):
    """One segment's `sequence.cluster_endpoints`, with `nv` fiber variants."""
    N = 2 * nv
    lo, hi = -1.5, 1.5
    return dict(seg=np.zeros(N, int),
                var=np.repeat(np.arange(nv), 2),
                dirn=np.tile([0, 1], nv),
                base=np.array([0, N]), nv=[nv], n=1, N=N,
                ent_q=rng.uniform(lo, hi, (N, 7)),
                exi_q=rng.uniform(lo, hi, (N, 7)),
                ent_xy=rng.uniform(0.0, 2.0, (N, 2)),
                exi_xy=rng.uniform(0.0, 2.0, (N, 2)),
                ent_h=rng.uniform(lo, hi, (N, 7)),
                exi_h=rng.uniform(lo, hi, (N, 7)),
                surcharge=rng.uniform(0.0, 0.4, N))


def _same(A, B, tol=1e-12):
    """allclose that treats matching infinities as equal."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    fa, fb = np.isfinite(A), np.isfinite(B)
    return bool(np.array_equal(fa, fb)
                and np.allclose(A[fa], B[fb], atol=tol, rtol=0.0))


def _plain_want(ends, return_home=True):
    """The matrix `_ArmMatrix` should slice, built from scratch. -> (2n+1)^2.

    DEPOT-CLOSED, because that is the model the run uses: with multi-tour bags
    on (`allocate.MULTI_TOUR`) a crossing may be flown via the ready pose, so
    the freshly built matrix these tests compare against has to be the one the
    sequencer will actually walk.  With the feature off both sides are the
    plain `cost_matrix` again and the assertions read exactly as they did.
    """
    n = len(ends["z"])
    C = sequence.cost_matrix(FLOOR, None, 0.30, 0.30, 1.0, ends=ends,
                             pen_ext=0.110, return_home=return_home)
    if not allocate.MULTI_TOUR:
        return C
    outof, into = sequence.home_legs(FLOOR, None, 0.30, 0.30, 1.0, ends=ends,
                                     pen_ext=0.110)
    return sequence.close_depot(C, outof, into, sequence.seg_index(n))


def test_arm_matrix_slices_the_plain_cost_matrix():
    """A bag's matrix out of the union == the bag's matrix built alone."""
    rng = np.random.default_rng(5)
    ends = [_seg_ends(rng) for _ in range(6)]
    M = allocate._ArmMatrix(FLOOR, False, 0.30, 0.30, 1.0, 0.110, None, True)
    M.ensure([(i, allocate._ArmMatrix.plain_nodes(e))
              for i, e in enumerate(ends)])
    for keys in ([0, 1, 2, 3, 4, 5], [4, 1, 0], [2], [5, 3]):
        C, _T, e, idx = M.matrix(keys)
        want = _plain_want(allocate._stack_ends([ends[k] for k in keys]))
        assert C.shape == want.shape == (2 * len(keys) + 1,) * 2
        assert _same(C, want), f"bag {keys} sliced a different matrix"
        assert len(idx) == 2 * len(keys)
        assert e["n"] == len(keys) and e["N"] == 2 * len(keys)


def test_arm_matrix_slices_the_cluster_cost_matrix():
    """The same, with the fiber menus on and a different variant count each."""
    rng = np.random.default_rng(7)
    ce = [_cluster_seg_ends(rng, nv) for nv in (1, 3, 2, 4)]
    M = allocate._ArmMatrix(FLOOR, True, 0.30, 0.60, 1.0, 0.110, None, False)
    M.ensure([(i, allocate._ArmMatrix.cluster_nodes(c))
              for i, c in enumerate(ce)])
    for keys in ([0, 1, 2, 3], [3, 0], [2, 1, 3]):
        C, T, e, _idx = M.matrix(keys)
        stacked = allocate._stack_cluster_ends([ce[k] for k in keys])
        wC, wT, we = sequence.cluster_cost_matrix(
            FLOOR, None, 0.30, 0.60, 1.0, 0.110, q_start=None,
            return_home=False, ends=stacked)
        if allocate.MULTI_TOUR:
            outof, into = sequence.cluster_home_legs(
                FLOOR, None, 0.30, 0.60, 1.0, 0.110, ends=stacked)
            wC = sequence.close_depot(wC, outof, into, stacked["seg"])
        assert _same(C, wC), f"bag {keys}: transit matrix differs"
        assert _same(T, wT), f"bag {keys}: tie-break matrix differs"
        assert list(e["nv"]) == list(we["nv"]) and e["N"] == we["N"]
        assert np.array_equal(e["seg"], we["seg"])
        assert np.allclose(e["surcharge"], we["surcharge"])


def test_arm_matrix_growth_does_not_disturb_what_was_there():
    """Adding spans later must leave the earlier ones' cells untouched."""
    rng = np.random.default_rng(11)
    ends = [_seg_ends(rng) for _ in range(5)]
    kw = dict()
    M = allocate._ArmMatrix(FLOOR, False, 0.30, 0.30, 1.0, 0.110, None, True,
                            **kw)
    M.ensure([(i, allocate._ArmMatrix.plain_nodes(ends[i])) for i in (0, 1)])
    first, _, _, _ = M.matrix([0, 1])
    M.ensure([(i, allocate._ArmMatrix.plain_nodes(ends[i])) for i in (2, 3, 4)])
    again, _, _, _ = M.matrix([0, 1])
    assert _same(first, again), "growing the union moved an existing cell"
    whole, _, _, _ = M.matrix([0, 1, 2, 3, 4])
    want = _plain_want(allocate._stack_ends(ends))
    assert _same(whole, want), "the grown union is not the matrix it slices"
    # ensure() is idempotent: asking again adds nothing and changes nothing
    n_before = len(M.span)
    M.ensure([(0, allocate._ArmMatrix.plain_nodes(ends[0]))])
    assert len(M.span) == n_before


def test_the_route_screen_is_the_same_answer_on_one_core_or_many():
    """Farming the crossings out must not move a single second.

    The paper screen is where the clock of this pipeline actually is, so it runs
    on a fork pool; a pool that changed an answer would be changing the tour the
    fleet flies.  Same block, both ways, and the memos dropped in between so the
    second run really does the work again.
    """
    from aris_sixarm import paper
    rng = np.random.default_rng(19)
    ends = [_seg_ends(rng) for _ in range(9)]
    stacked = allocate._stack_ends(ends)
    jobs, par_min = sequence.ROUTE_JOBS, sequence.ROUTE_PAR_MIN
    try:
        sequence.ROUTE_JOBS, sequence.ROUTE_PAR_MIN = 1, 10 ** 9
        paper.clear_cache()
        one = sequence.cost_matrix(FLOOR, None, 0.30, 0.30, 1.0, ends=stacked,
                                   pen_ext=0.110)
        sequence.ROUTE_JOBS, sequence.ROUTE_PAR_MIN = 4, 1
        paper.clear_cache()
        many = sequence.cost_matrix(FLOOR, None, 0.30, 0.30, 1.0, ends=stacked,
                                    pen_ext=0.110)
    finally:
        sequence.ROUTE_JOBS, sequence.ROUTE_PAR_MIN = jobs, par_min
        paper.clear_cache()
    assert _same(one, many, tol=0.0), "the pool priced a crossing differently"


def test_the_batched_key_is_the_key_route_files_under():
    """`key_maker` must land where `route_key` looks, or the memo never hits."""
    from aris_sixarm import paper
    rng = np.random.default_rng(31)
    exi = rng.uniform(-1.5, 1.5, (4, 7))
    ent = rng.uniform(-1.5, 1.5, (3, 7))
    key = paper.key_maker(FLOOR, exi, ent, 0.110, 1.0)
    for a in range(4):
        for b in range(3):
            tip, chain = -0.01, paper.CHAIN_CLEAR
            assert key(a, b, tip, chain) == paper._key(
                FLOOR, exi[a], ent[b], 0.110, 1.0, tip, chain)


# ==========================================================================
# 2. the warm start is a tour over the bag it was asked for
# ==========================================================================
def _random_matrix(n, rng, nodes=2):
    """A cost matrix over n segments with `nodes` interchangeable nodes each."""
    N = n * nodes
    C = rng.uniform(0.5, 4.0, (N + 1, N + 1))
    grp = np.arange(N) // nodes
    C[:N, :N][grp[:, None] == grp[None, :]] = np.inf
    C[N, N] = np.inf
    return C, [list(range(k * nodes, (k + 1) * nodes)) for k in range(n)]


def test_seed_from_visits_every_segment_exactly_once():
    """Whatever the reference was, the seed is a tour over THIS bag."""
    rng = np.random.default_rng(3)
    C, groups = _random_matrix(7, rng)
    tour = sequence.seed_from(C, groups, None)
    assert tour[0] == C.shape[0] - 1, "the depot comes first"
    body = tour[1:]
    assert sorted(b // 2 for b in body) == list(range(7))
    assert np.isfinite(sequence.cycle_cost(C, tour))

    # a reference over a bag with one segment gone and one node renamed
    ref = [g[0] for g in groups[:5]][::-1] + [999]
    tour = sequence.seed_from(C, groups, ref)
    assert sorted(b // 2 for b in tour[1:]) == list(range(7))
    assert np.isfinite(sequence.cycle_cost(C, tour))


def test_warm_started_solve_is_never_a_worse_tour_than_its_seed():
    """The descent may only improve on what it was handed."""
    rng = np.random.default_rng(17)
    C, groups = _random_matrix(20, rng)
    ref = sequence.nearest_neighbour(C, 20)
    seed = sequence.seed_from(C, groups, ref)
    r = sequence.solve(C, 20, exact_max_n=8, budget=0.15, warm=ref)
    assert r["method"] == "warm+2opt+oropt"
    assert r["cost"] <= sequence.cycle_cost(C, seed) + 1e-9


def test_seed_from_refuses_an_infinite_tour():
    """An unreachable segment gives back None, not an infinite price."""
    rng = np.random.default_rng(23)
    C, groups = _random_matrix(5, rng)
    C[:, groups[3]] = np.inf              # nothing can enter segment 3
    assert sequence.seed_from(C, groups, None) is None
    # and `solve` then falls back to the search that raises properly
    try:
        sequence.solve(C, 5, exact_max_n=0, budget=0.05, warm=[0, 2, 4])
    except RuntimeError:
        pass


# ==========================================================================
# 3. first improvement against the exhaustive scan
# ==========================================================================
def _instance(rng, n_seg=14, n_arm=4):
    """A synthetic balancing instance: ink per (arm, segment) and a transit."""
    ink = rng.uniform(1.0, 9.0, (n_arm, n_seg))
    opts = []
    for i in range(n_seg):
        k = rng.integers(1, n_arm + 1)
        opts.append(set(int(x) for x in
                        rng.choice(n_arm, size=int(k), replace=False)))
    owner = [int(sorted(o)[0]) for o in opts]

    def load_fn(a, idx):
        """Ink plus a pen-up cost that grows with the number of pieces."""
        return float(sum(ink[a, i] for i in idx) + 0.8 * len(idx))

    def draw_fn(a, i):
        return float(ink[a, i])

    return owner, opts, load_fn, draw_fn


def test_first_improvement_matches_the_exhaustive_scan():
    """Same-or-better than where it started, and within 2 % of best-improvement."""
    for seed in range(12):
        rng = np.random.default_rng(100 + seed)
        owner, opts, load_fn, draw_fn = _instance(rng)
        _o1, best = allocate.balance_loads(owner, opts, load_fn,
                                           strategy="best")
        _o2, first = allocate.balance_loads(owner, opts, load_fn,
                                            strategy="first", draw_fn=draw_fn)
        assert first["max_after"] <= first["max_before"] + 1e-9, \
            "the pass made the busiest arm busier"
        assert first["max_after"] <= best["max_after"] * 1.02 + 1e-9, \
            f"seed {seed}: first {first['max_after']:.3f} vs best " \
            f"{best['max_after']:.3f}"
        # and it gets there having priced fewer candidates
        assert first["n_scanned"] <= best["n_scanned"]


def test_first_improvement_without_a_hint_still_terminates():
    """`draw_fn` is an ordering, not a requirement."""
    rng = np.random.default_rng(41)
    owner, opts, load_fn, _draw = _instance(rng)
    o, info = allocate.balance_loads(owner, opts, load_fn, strategy="first")
    assert len(o) == len(owner)
    assert info["max_after"] <= info["max_before"] + 1e-9
    for i, a in enumerate(o):
        assert a in opts[i], "a segment was given to an arm that cannot draw it"


# ==========================================================================
# 4. the floor is a floor
# ==========================================================================
class _StubPricer:
    """Just the one method `_phase_floor` asks for."""

    def __init__(self, ink):
        self.ink = ink

    def draw_s(self, arm, it, ent):
        return float(self.ink[arm][it["k"]])


def test_phase_floor_is_a_lower_bound_on_every_assignment():
    """Enumerate small instances outright and check nothing gets under it."""
    import itertools
    rng = np.random.default_rng(9)
    for trial in range(8):
        n_seg, n_arm = 6, 3
        ink = rng.uniform(1.0, 6.0, (n_arm, n_seg))
        items = [dict(k=i, arm=0, entry=None) for i in range(n_seg)]
        options = [set(range(n_arm)) for _ in range(n_seg)]
        entries = [{a: None for a in range(n_arm)} for _ in range(n_seg)]
        whole, mean = allocate._phase_floor(items, options, entries,
                                            _StubPricer(ink))
        worst = np.inf
        for assign in itertools.product(range(n_arm), repeat=n_seg):
            loads = [sum(ink[a, i] for i in range(n_seg) if assign[i] == a)
                     for a in range(n_arm)]
            worst = min(worst, max(loads))
        assert whole <= worst + 1e-9, f"trial {trial}: floor above the optimum"
        assert mean <= whole + 1e-9, "the mean bound must be the weaker one"


def test_balance_stops_at_the_floor():
    """Given a floor it is already at, the pass prices nothing at all."""
    rng = np.random.default_rng(2)
    owner, opts, load_fn, draw_fn = _instance(rng)
    _o, info = allocate.balance_loads(owner, opts, load_fn, draw_fn=draw_fn,
                                      floor=1e9, strategy="first")
    assert info["stopped"] == "floor" and info["n_scanned"] == 0
    assert info["rounds"] == 0


# ==========================================================================
# 5. what the four execution profiles may share, and what they may not
# ==========================================================================
def _fixture_strokes():
    """Three short lines arms 31 and 71 both reach (the `test_csail` fixture)."""
    return [dict(pts=np.column_stack([np.linspace(1.74, 1.87, 14),
                                      np.full(14, y)]),
                 color="grey", kind="outline", id=i)
            for i, y in enumerate((1.631, 1.700, 1.560))]


def _fixture_kw():
    from aris_sixarm.fleet import FLEET_SIXARM
    return dict(arms=[31, 71], fleet=FLEET_SIXARM, pens={31: 0.200, 71: 0.200},
                colors={31: "grey", 71: "grey"}, verbose=False)


def test_plan_family_ignores_qd_frac_and_not_the_band():
    """The key that decides what two profiles may share is the planner's."""
    from aris_sixarm.fleet import FLEET_SIXARM
    strokes = _fixture_strokes()
    arms = [31, 71]
    pens = {31: 0.200, 71: 0.200}

    def fam(objective):
        opts = dict(objective=objective, tilt_max_deg=0.0)
        aopts = {a: allocate.pen_opts(opts, pens, a) for a in arms}
        return allocate.plan_family(strokes, arms, aopts, pens)

    assert fam("min_travel") == fam("min_travel"), "not a function of its input"
    assert fam("min_travel") != fam("maximin_sigma"), \
        "the band objective changes what plan_stroke returns and must key it"
    # a different picture is a different question
    other = _fixture_strokes()
    other[0]["pts"] = other[0]["pts"] + 0.01
    opts = dict(objective="min_travel", tilt_max_deg=0.0)
    aopts = {a: allocate.pen_opts(opts, pens, a) for a in arms}
    assert allocate.plan_family(other, arms, aopts, pens) != fam("min_travel")
    del FLEET_SIXARM


def test_sharing_a_plan_bag_does_not_change_the_allocation():
    """Two profiles over one bag must allocate exactly what they allocate alone.

    The whole point of `share` is that `qd_frac` is invisible to the planner, so
    the second profile of a family may have the first one's probes and clean
    re-plans.  If that were only nearly true the two runs would drift, so it is
    checked on the spans themselves and not on a summary of them.
    """
    strokes, kw = _fixture_strokes(), _fixture_kw()

    def spans(r):
        return sorted((a, s["stroke_id"], round(min(s["s_range"]), 9),
                       round(max(s["s_range"]), 9))
                      for a in r["arms"] for s in r["programs"][a])

    solo = [allocate.allocate(strokes, seq_opts=dict(qd_frac=q), **kw)
            for q in (0.30, 0.60)]
    bag = {}
    together = [allocate.allocate(strokes, seq_opts=dict(qd_frac=q), share=bag,
                                  **kw) for q in (0.30, 0.60)]
    for q, r0, r1 in zip((0.30, 0.60), solo, together):
        assert spans(r0) == spans(r1), f"qd {q}: sharing moved a span"
        assert abs(r0["drawn_len"] - r1["drawn_len"]) < 1e-12, f"qd {q}"
        assert abs(r0["dropped_len"] - r1["dropped_len"]) < 1e-12, f"qd {q}"
    assert together[1]["timing"]["probe"] == 0.0, \
        "the second profile re-probed what the first one had already asked"
    assert together[0]["timing"]["probe"] > 0.0
    assert any(k[0] == "probe" for k in bag), "nothing was actually shared"


# ==========================================================================
# 6. the profile that ships first
# ==========================================================================
class _Args:
    qd_frac = 0.30
    cluster = False
    band_objective = "maximin_sigma"
    fps = 12.0
    pause = 0.0


def _grid_stubs(makespan, floors):
    events = []

    def alloc(b, p):
        import csail_schedule as cs
        events.append(("alloc", cs.profile_name(p)))
        return [dict(name=cs.profile_name(p))], {}, {13: 0.110}

    def floor(b, phases, pens, alt=None):
        return floors[phases[0]["name"]]

    def conduct(b, phases, dt, pens, alt=None):
        events.append(("conduct", phases[0]["name"]))
        return [dict(sch=dict(duration=makespan[phases[0]["name"]]),
                     rep=dict(ok=True, min_clearance=0.0829))]

    return events, alloc, floor, conduct


def test_first_ink_ordering_certifies_before_the_rest_are_allocated():
    """The historically-best cell goes all the way to a programme first."""
    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
    import csail_schedule as cs
    ms = {"qd0.30": 120.0, "qd0.30+cluster": 100.0,
          "qd0.60": 95.0, "qd0.60+cluster": 90.0}
    fl = {"qd0.30": 80.0, "qd0.30+cluster": 70.0,
          "qd0.60": 60.0, "qd0.60+cluster": 65.0}
    events, alloc, floor, conduct = _grid_stubs(ms, fl)
    seen = []
    sel = cs.select_profile(_Args(), alloc, 1 / 48.0, conduct=conduct,
                            floor=floor, verbose=False, jobs=1,
                            on_certified=lambda r: seen.append(
                                (r["profile"], list(events))))
    assert events[0] == ("alloc", "qd0.60+cluster")
    assert events[1] == ("conduct", "qd0.60+cluster")
    assert seen and seen[0][0] == "qd0.60+cluster"
    assert seen[0][1] == events[:2], \
        "the programme was offered only after other cells had been allocated"
    assert sel["chosen"]["profile"] == "qd0.60+cluster"
    assert sel["first_certified_s"] is not None
    assert [r["profile"] for r in sel["grid"]] == [
        cs.profile_name(p) for p in cs.PROFILES], "the report order moved"


def test_a_faster_profile_replaces_the_provisional_best():
    """...and the ones whose floor cannot beat it are never conducted."""
    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
    import csail_schedule as cs
    ms = {"qd0.30": 120.0, "qd0.30+cluster": 70.0,
          "qd0.60": 95.0, "qd0.60+cluster": 90.0}
    fl = {"qd0.30": 95.0, "qd0.30+cluster": 60.0,
          "qd0.60": 88.0, "qd0.60+cluster": 65.0}
    events, alloc, floor, conduct = _grid_stubs(ms, fl)
    seen = []
    sel = cs.select_profile(_Args(), alloc, 1 / 48.0, conduct=conduct,
                            floor=floor, verbose=False, jobs=1,
                            on_certified=lambda r: seen.append(r["profile"]))
    assert seen == ["qd0.60+cluster", "qd0.30+cluster"], \
        "the caller was not told about the programme that finally shipped"
    assert sel["chosen"]["profile"] == "qd0.30+cluster"
    by = {r["profile"]: r for r in sel["grid"]}
    assert by["qd0.30"]["status"] == "pruned", \
        "a floor of 95 s cannot beat a certified 90 s and must not be conducted"
    assert "qd0.60+cluster" in by["qd0.30"]["reason"], \
        "the provisional best is what pruned it, and the reason must say so"
    assert ("conduct", "qd0.30") not in events
    # serially, each certified cell still prunes the next: 88 s cannot beat the
    # 70 s qd0.30+cluster just certified
    assert by["qd0.60"]["status"] == "pruned"
    assert ("conduct", "qd0.60") not in events


def test_profile_order_puts_the_preferred_cell_first():
    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
    import csail_schedule as cs
    got = [cs.profile_name(p)
           for p in cs.profile_order(cs.PROFILES, "qd0.60+cluster")]
    assert got[0] == "qd0.60+cluster"
    assert sorted(got) == sorted(cs.profile_name(p) for p in cs.PROFILES)
    assert got[1:] == [cs.profile_name(p) for p in cs.PROFILES
                       if cs.profile_name(p) != "qd0.60+cluster"]
    # an unknown name leaves the listed order alone rather than guessing
    assert [cs.profile_name(p) for p in cs.profile_order(cs.PROFILES, "nope")] \
        == [cs.profile_name(p) for p in cs.PROFILES]


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                fails += 1
                print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if fails else 0)
