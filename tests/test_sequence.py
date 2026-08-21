"""Direction-agnostic segments and the per-arm sequencer.

Run: python3 tests/test_sequence.py     (or pytest tests/)

Fast on purpose: one real certified plan (for the reversal), and everything
else on synthetic cost matrices so the search is tested against arithmetic
somebody can check by hand rather than against the IK.
"""
import itertools
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from aris_sixarm import allocate, sequence, writing            # noqa: E402
from aris_sixarm.fleet import FLEET_SIXARM as FLEET            # noqa: E402
from aris_sixarm.stroke_api import plan_stroke, reverse_plan   # noqa: E402
from aris_sixarm.validate import validate_plan                 # noqa: E402

FLOOR = FLEET[13]
_CACHE = {}


def plan_floor():
    """A comfortable 0.30 m line for the floor arm — certified end to end."""
    if "floor" not in _CACHE:
        bx, by = FLOOR.xy
        pts = np.array([[bx + 0.55, by - 0.15], [bx + 0.55, by + 0.15]])
        r = plan_stroke(pts, FLOOR)
        assert r["status"] == "ok", f"fixture did not plan: {r['status']}"
        _CACHE["floor"] = r
    return _CACHE["floor"]


# ==========================================================================
# 1. a certified plan, executed the other way round
# ==========================================================================
def test_reversed_plan_is_a_flip_that_still_validates():
    """The gates are symmetric in s, so reversal is array flipping — and the
    INDEPENDENT validator has to agree, not just the arithmetic."""
    p = plan_floor()
    r = reverse_plan(p, FLOOR)

    assert np.array_equal(r["qs"], p["qs"][::-1])
    assert np.array_equal(r["pts"], p["pts"][::-1])
    assert np.array_equal(r["sigmas"], p["sigmas"][::-1])
    assert np.array_equal(r["margins"], p["margins"][::-1])
    assert np.allclose(r["s"], 1.0 - p["s"][::-1], atol=1e-12)
    assert np.allclose(r["knots"][:, 0], 1.0 - p["knots"][::-1, 0], atol=1e-12)
    assert r["reversed"] is True and p.get("reversed") is None

    # the clock is re-derived, not flipped; |dq/ds| is reversal-invariant so it
    # must come out the same length
    assert r["times"][0] == 0.0
    assert np.all(np.diff(r["times"]) > 0), "the reversed clock must increase"
    assert abs(r["total_time"] - p["total_time"]) < 1e-9

    rep = validate_plan(r["pts"], FLOOR, r["qs"], times=r["times"])
    assert rep["ok"], f"reversed plan rejected: {rep['violations'][:2]}"
    assert r["validation"]["ok"], "reverse_plan must carry its own certificate"
    assert r["status"] == "ok"

    # the original is untouched, and reversing twice is the identity
    assert np.array_equal(p["qs"], plan_floor()["qs"])
    rr = reverse_plan(r, FLOOR)
    assert np.array_equal(rr["qs"], p["qs"]) and rr["reversed"] is False


def test_reverse_segment_turns_a_programme_entry_round():
    """An allocator programme entry flips its geometry AND its direction flag."""
    p = plan_floor()
    seg = dict(stroke_id=7, color="grey", kind="outline", s_range=(0.0, 1.0),
               direction=1, pts=np.asarray(p["stroke"], float),
               length=float(p["arc_len"]), plan=p)
    rev = allocate.reverse_segment(seg, FLOOR)
    assert rev is not None
    assert rev["direction"] == -1 and rev["flipped"] is True
    assert np.array_equal(rev["pts"], seg["pts"][::-1])
    assert np.array_equal(rev["plan"]["qs"], p["qs"][::-1])
    assert rev["s_range"] == seg["s_range"] and rev["length"] == seg["length"]
    assert seg["direction"] == 1, "the input entry must not be mutated"


# ==========================================================================
# 2. the cost matrix IS the timeline's transit
# ==========================================================================
def _fake_ends(n, rng):
    """Endpoint configurations/hovers without going near the IK."""
    lo, hi = -1.5, 1.5
    return dict(n=n, q=rng.uniform(lo, hi, (n, 2, 7)),
                xy=rng.uniform(0.0, 2.0, (n, 2, 2)),
                hover=rng.uniform(lo, hi, (n, 2, 7)),
                z=np.full((n, 2), 0.06))


def test_cost_matrix_is_the_transit_the_timeline_will_pay():
    """Every cell must equal `writing`'s own lift + travel + lower for that
    pair — the sequencer is not allowed a cost model of its own."""
    rng = np.random.default_rng(3)
    n = 5
    ends = _fake_ends(n, rng)
    C = sequence.cost_matrix(FLOOR, None, ends=ends)
    N = 2 * n
    assert C.shape == (N + 1, N + 1)

    def end_of(node, entry):
        i, d = node // 2, node % 2
        e = d if entry else 1 - d
        return ends["q"][i, e], ends["xy"][i, e], ends["hover"][i, e]

    for a in range(N):
        for b in range(N):
            if a // 2 == b // 2:
                assert not np.isfinite(C[a, b]), "a segment cannot follow itself"
                continue
            qa, xya, ha = end_of(a, entry=False)
            qb, xyb, hb = end_of(b, entry=True)
            hop = float(np.linalg.norm(xyb - xya))
            want = sum(writing.transit_time(qa, ha, hb, qb, hop))
            assert abs(C[a, b] - want) < 1e-12, f"cell ({a},{b})"
    for b in range(N):
        qb, _, hb = end_of(b, entry=True)
        assert abs(C[N, b] - sum(writing.enter_time(FLOOR.q_seed, hb, qb))) < 1e-12
    for a in range(N):
        qa, _, ha = end_of(a, entry=False)
        assert abs(C[a, N] - sum(writing.exit_time(qa, ha, FLOOR.q_seed))) < 1e-12
    assert not np.isfinite(C[N, N])


def test_batched_and_scalar_joint_times_agree():
    """`dq_time_many` is `_dq_time` in bulk, and nothing else."""
    rng = np.random.default_rng(11)
    A, B = rng.uniform(-2, 2, (6, 7)), rng.uniform(-2, 2, (4, 7))
    M = writing.dq_time_many(A, B, 0.3, 0.25)
    for i in range(len(A)):
        for j in range(len(B)):
            assert abs(M[i, j] - writing._dq_time(A[i], B[j], 0.3, 0.25)) < 1e-15


# ==========================================================================
# 3. exact sequencing
# ==========================================================================
def _geo_matrix(P, depot, w=None):
    """Segments as endpoint pairs P (n,2,2); cost = pen-tip distance from one
    exit to the next entry, plus optional per-node weights `w = (exit, entry)`
    which make the matrix genuinely asymmetric under a direction flip."""
    P = np.asarray(P, float)
    n = len(P)
    N = 2 * n
    ent = P[:, [0, 1]].reshape(N, 2)
    exi = P[:, [1, 0]].reshape(N, 2)
    C = np.full((N + 1, N + 1), np.inf)
    D = np.linalg.norm(ent[None, :, :] - exi[:, None, :], axis=-1)
    we, wn = (np.zeros(N), np.zeros(N)) if w is None else w
    C[:N, :N] = D + we[:, None] + wn[None, :]
    seg = np.arange(N) // 2
    C[:N, :N][seg[:, None] == seg[None, :]] = np.inf
    C[N, :N] = np.linalg.norm(ent - depot, axis=1) + wn
    C[:N, N] = np.linalg.norm(exi - depot, axis=1) + we
    return C


def _brute_force(C, n, forward_only=False):
    """Every order x every direction assignment. -> (cost, order, dirs)."""
    best = (np.inf, None, None)
    dirsets = [[1] * n] if forward_only else [list(d) for d in
                                              itertools.product((1, -1), repeat=n)]
    for order in itertools.permutations(range(n)):
        for dirs in dirsets:
            c = sequence.sequence_cost(C, n, list(order), dirs)
            if c < best[0]:
                best = (float(c), list(order), list(dirs))
    return best


def test_held_karp_on_a_square_anybody_can_check():
    """FOUR SIDES OF A SQUARE, two of them certified the wrong way round.

    Segments: A->B, B->C, D->C, A->D on the unit square, the pen parked at A.
    Walk the square A-B-C-D-A and every transit is zero — but only if the last
    two segments are DRAWN BACKWARDS, because that is not the direction they
    were certified in.  So the optimum is exactly 0.0 s and it is unreachable
    without direction agnosticism; the best all-forward tour costs 2 + sqrt(2).
    """
    A, B, C_, D = (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)
    P = np.array([[A, B], [B, C_], [D, C_], [A, D]])
    C = _geo_matrix(P, np.array(A))

    r = sequence.held_karp(C, 4)
    assert abs(r["cost"]) < 1e-12, f"optimum should be 0, got {r['cost']}"
    assert abs(sequence.sequence_cost(C, 4, r["order"], r["dirs"])
               - r["cost"]) < 1e-12, "reported cost must match the tour"
    assert sorted(r["order"]) == [0, 1, 2, 3]
    assert r["dirs"].count(-1) == 2, "two sides have to be drawn backwards"

    exact, _, _ = _brute_force(C, 4)
    assert abs(exact - r["cost"]) < 1e-12
    fwd, _, _ = _brute_force(C, 4, forward_only=True)
    assert abs(fwd - 2 * np.sqrt(2)) < 1e-12, f"all-forward best {fwd}"
    assert fwd > exact + 1.0, "direction agnosticism has to be worth something"


def test_held_karp_matches_brute_force_on_asymmetric_instances():
    """Exactness, on matrices with no geometry to be lucky with."""
    for seed in (0, 1, 2):
        rng = np.random.default_rng(seed)
        n = 5
        P = rng.uniform(0, 1, (n, 2, 2))
        w = (rng.uniform(0, 0.4, 2 * n), rng.uniform(0, 0.4, 2 * n))
        C = _geo_matrix(P, rng.uniform(0, 1, 2), w)
        r = sequence.held_karp(C, n)
        best, _, _ = _brute_force(C, n)
        assert abs(r["cost"] - best) < 1e-9, f"seed {seed}: {r['cost']} vs {best}"
        assert sorted(r["order"]) == list(range(n))
        assert abs(sequence.sequence_cost(C, n, r["order"], r["dirs"])
                   - r["cost"]) < 1e-9


def test_exact_solver_reaches_its_advertised_ceiling():
    """16 segments is where the exact path stops: 2^16 x 32 = 2.1 M states.

    Worth pinning, because the whole design rests on the claim that the case
    this repo actually has (12 segments on the busiest arm) is nowhere near it.
    """
    rng = np.random.default_rng(16)
    n = 16
    C = _geo_matrix(rng.uniform(0, 1, (n, 2, 2)), rng.uniform(0, 1, 2))
    t0 = time.time()
    r = sequence.solve(C, n)
    w = time.time() - t0
    print(f"      16 segments exactly: {w:.2f} s over {r['states']} states")
    assert r["method"] == "held_karp" and sorted(r["order"]) == list(range(n))
    assert r["states"] == (1 << n) * 2 * n
    assert w < 5.0, f"exact solve took {w:.1f} s at the ceiling"
    h = sequence.solve(C, n, exact_max_n=0, budget=0.5)
    assert h["cost"] >= r["cost"] - 1e-9, "the heuristic cannot beat the optimum"


def test_held_karp_is_deterministic():
    rng = np.random.default_rng(5)
    C = _geo_matrix(rng.uniform(0, 1, (7, 2, 2)), np.zeros(2))
    a = sequence.held_karp(C, 7)
    b = sequence.held_karp(C, 7)
    assert a["order"] == b["order"] and a["dirs"] == b["dirs"]


# ==========================================================================
# 4. the heuristic fallback
# ==========================================================================
def test_move_deltas_match_a_full_re_evaluation():
    """The passes price a move in O(1) off a prefix sum; that price has to be
    the actual change in the tour's cost, on a matrix where turning a block
    round is NOT free."""
    rng = np.random.default_rng(17)
    n = 7
    w = (rng.uniform(0, 0.5, 2 * n), rng.uniform(0, 0.5, 2 * n))
    C = _geo_matrix(rng.uniform(0, 1, (n, 2, 2)), rng.uniform(0, 1, 2), w)
    N = 2 * n
    fl = sequence.flip_index(N)
    R = sequence.reversal_penalty(C, N)
    assert np.abs(np.nan_to_num(R)).max() > 1e-6, "the fixture must be asymmetric"

    tour = sequence.nearest_neighbour(C, n)
    base = sequence.cycle_cost(C, tour)
    for i in range(1, len(tour)):
        d = sequence.two_opt_deltas(C, R, fl, tour, i)
        for k, j in enumerate(range(i, len(tour))):
            got = sequence.cycle_cost(
                C, sequence.apply_two_opt(list(tour), fl, i, j))
            assert abs((got - base) - d[k]) < 1e-9, f"2-opt ({i},{j})"
    for L in (1, 2, 3):
        for p in range(1, len(tour) - L + 1):
            q = p + L - 1
            fwd, rev, rest = sequence.or_opt_deltas(C, R, fl, tour, p, q)
            for k in range(len(rest)):
                for arr, flip in ((fwd, False), (rev, True)):
                    got = sequence.cycle_cost(
                        C, sequence.apply_or_opt(list(tour), fl, p, q, k, flip))
                    assert abs((got - base) - arr[k]) < 1e-9, \
                        f"or-opt L={L} p={p} k={k} rev={flip}"


def test_fallback_finds_the_exact_optimum_on_a_small_instance():
    """Forced onto the heuristic path, it should still land on (or within a
    whisker of) what Held-Karp proves is optimal."""
    for seed in (4, 5, 6):
        rng = np.random.default_rng(seed)
        n = 11
        C = _geo_matrix(rng.uniform(0, 1, (n, 2, 2)), rng.uniform(0, 1, 2))
        exact = sequence.solve(C, n)
        heur = sequence.solve(C, n, exact_max_n=0, budget=0.3)
        assert exact["method"] == "held_karp" and heur["method"].startswith("nn+")
        assert sorted(heur["order"]) == list(range(n))
        assert heur["cost"] >= exact["cost"] - 1e-9, "nothing beats the optimum"
        assert heur["cost"] <= exact["cost"] * 1.05 + 1e-9, \
            f"seed {seed}: heuristic {heur['cost']:.4f} vs exact {exact['cost']:.4f}"


def test_fallback_improves_on_its_nearest_neighbour_seed():
    """Twenty segments: past the exact solver, so the local search has to earn
    its keep against the greedy seed it starts from."""
    rng = np.random.default_rng(20)
    n = 20
    C = _geo_matrix(rng.uniform(0, 1, (n, 2, 2)), rng.uniform(0, 1, 2))
    r = sequence.solve(C, n, budget=0.5)
    assert r["method"] == "nn+2opt+oropt" and r["n"] == n
    assert sorted(r["order"]) == list(range(n)), "every segment exactly once"
    assert len(r["dirs"]) == n and set(r["dirs"]) <= {1, -1}
    assert abs(sequence.sequence_cost(C, n, r["order"], r["dirs"])
               - r["cost"]) < 1e-9
    gain = 100 * (r["nn_cost"] - r["cost"]) / r["nn_cost"]
    print(f"      20 segments: NN {r['nn_cost']:.3f} -> {r['cost']:.3f} "
          f"({gain:.1f} % better, {r['wall']:.2f} s)")
    assert gain > 2.0, f"local search only found {gain:.1f} %"


def test_fallback_scales_to_a_few_hundred_segments():
    """THE PICTURE WILL GET BIGGER.  300 segments on one arm: the matrix is
    360 000 cells built as array ops, every move is an O(1) lookup, and the
    search stops on a wall-clock budget rather than on a pass count."""
    rng = np.random.default_rng(300)
    n = 300
    w = (rng.uniform(0, 0.05, 2 * n), rng.uniform(0, 0.05, 2 * n))
    P = rng.uniform(0, 1, (n, 2, 2)) * np.array([1.0, 1.0])
    P[:, 1] = P[:, 0] + rng.uniform(-0.08, 0.08, (n, 2))     # short segments
    t0 = time.time()
    C = _geo_matrix(P, rng.uniform(0, 1, 2), w)
    t_mat = time.time() - t0
    r = sequence.solve(C, n, budget=2.0)
    gain = 100 * (r["nn_cost"] - r["cost"]) / r["nn_cost"]
    print(f"      300 segments: matrix {t_mat:.2f} s, search {r['wall']:.2f} s, "
          f"NN {r['nn_cost']:.2f} -> {r['cost']:.2f} ({gain:.1f} % better, "
          f"{r['stats']['kicks']} kicks, {r['stats']['rounds']} rounds)")
    assert sorted(r["order"]) == list(range(n))
    assert abs(sequence.sequence_cost(C, n, r["order"], r["dirs"])
               - r["cost"]) < 1e-6
    assert gain > 5.0, f"only {gain:.1f} % better than the greedy seed"
    assert r["wall"] < 4.0, f"search took {r['wall']:.1f} s against a 2 s budget"
    assert t_mat < 2.0, f"matrix build took {t_mat:.1f} s"


def test_solver_handles_the_degenerate_arms():
    C0 = np.zeros((1, 1))
    assert sequence.solve(C0, 0)["order"] == []
    rng = np.random.default_rng(1)
    C1 = _geo_matrix(rng.uniform(0, 1, (1, 2, 2)), np.zeros(2))
    r = sequence.solve(C1, 1)
    assert r["order"] == [0] and r["dirs"][0] in (1, -1)
    assert abs(r["cost"] - sequence.sequence_cost(C1, 1, [0], r["dirs"])) < 1e-12


if __name__ == "__main__":
    t0 = time.time()
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            t1 = time.time()
            try:
                fn()
                print(f"{name} PASS ({time.time() - t1:.2f} s)")
            except AssertionError as e:
                fails += 1
                print(f"{name} FAIL: {e}")
    print(f"\n{'ALL PASS' if not fails else f'{fails} FAILURES'} "
          f"in {time.time() - t0:.1f} s")
