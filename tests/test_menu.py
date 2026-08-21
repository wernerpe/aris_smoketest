"""The 2026-08-20 refactor: a gated min-travel band, and fiber menus into L2.

Five things are pinned here, and each of them is a claim the refactor makes
that could quietly stop being true:

  1. the band objective changed, and the GATES did not — a shorter path may
     spend sigma down to the gate and not one step past it;
  2. a menu prunes exactly the (entry, exit) pairs the band cannot connect,
     no more and no fewer;
  3. the cluster DP beats the (segment, direction) DP where entry matching is
     what is on the table, and lowers the reconfiguration while it does;
  4. materialising a variant lazily gives back the plan eager planning gives,
     to the float, certificate included;
  5. with one variant per segment the whole cluster apparatus REDUCES to the
     sequencer it replaced — same matrix, same tour, same seconds;
  6. and the price the ALLOCATOR puts on a bag is the tour the sequencer will
     actually reach in it — for whichever of the two models the run is using,
     because one object carries both (`allocate.cost_model`).

(5) is the one that makes the rest safe to ship: it says the new code path is a
generalisation of the old one and not a rewrite of it.  (6) is the one that
keeps them honest afterwards: the balancer used to price every candidate with
the single-variant DP while `--cluster` sequenced with the cluster DP, so it
balanced a load the sequencer then moved.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from aris_sixarm import (allocate, menu, planner, pwl, sequence, stroke_api,
                         trace, writing)
from aris_sixarm.fleet import (FLEET_SIXARM as FLEET, H_INV_DEFAULT,
                               SHEET_SIXARM as SHEET)

PEN = 0.200
SPEC = FLEET[31]


# --------------------------------------------------------------------------
# 1. the synthetic band: gates are hard, travel is the objective
# --------------------------------------------------------------------------
def _synthetic_sheet(Ns=40, Nq=11, valley=(10, 30), valley_j=6, hole=None):
    """A band with a low-sigma valley at low q7 in the middle of the stroke.

    Every cell clears both gates, so nothing here is about feasibility.  What
    differs is sigma: 0.30 everywhere except a valley of 0.12 (still ABOVE the
    0.10 gate) at j < `valley_j` for s in `valley`.  Pin both ends at j = 0 and
    the two objectives must disagree — maximin climbs out of the valley and
    back to keep its bottleneck at 0.30, min-travel stays in it, because 0.12
    is already good enough and the climb is 0.60 rad of null-space swing that
    draws nothing.

    Q is built so the arithmetic is checkable by hand: one q7 index is 0.05 rad
    of joint travel, one arc-length step is 0.002 rad of it.
    """
    mask = np.ones((Ns, Nq), bool)
    sigma = np.full((Ns, Nq), 0.30)
    lo, hi = valley
    sigma[lo:hi, :valley_j] = 0.12
    margin = np.full((Ns, Nq), 0.30)
    Q = np.zeros((Ns, Nq, 7))
    Q[:, :, 6] = np.arange(Nq)[None, :] * 0.05
    Q[:, :, 0] = np.arange(Ns)[:, None] * 0.002
    if hole is not None:
        i0, j0, j1 = hole
        sigma[i0, j0:j1] = 0.05           # below the gate: a real hole
    return dict(id=0, mask=mask, sigma=sigma, margin=margin, Q=Q,
                spans_s=True, nodes=int(mask.sum()), i0=0, i1=Ns - 1)


def _band(sheet):
    free = pwl.band_free(sheet)
    clear = pwl.clearance_map(free)
    edges = pwl._edge_ok(sheet, planner.JUMP_THRESH) & free[:-1][:, :, None]
    return free, clear, edges, pwl._edge_travel(sheet)


def _path_travel(sheet, js):
    etr = pwl._edge_travel(sheet, "chord")
    return float(etr[np.arange(len(js) - 1), js[:-1], js[1:] - js[:-1] + 1].sum())


def test_gated_min_travel_beats_maximin_and_never_spends_a_gate():
    """The objective change, on a band whose answer is known by construction."""
    sheet = _synthetic_sheet()
    free, clear, edges, etr = _band(sheet)
    Ns, Nq = free.shape

    js_t, last_t, cost = pwl._dense_dp_travel(free, clear, edges, etr,
                                              pwl.W_CLEARANCE, j_start=0, j_end=0)
    js_m, last_m = pwl._dense_dp(free, sheet["sigma"], clear, edges,
                                 pwl.W_CLEARANCE, pwl.W_TRAVEL,
                                 j_start=0, j_end=0)
    assert js_t is not None and last_t == Ns - 1, "min-travel found no path"
    assert js_m is not None and last_m == Ns - 1, "maximin found no path"

    # the whole point: less joint travel, and less q7 wander with it
    t_travel, m_travel = _path_travel(sheet, js_t), _path_travel(sheet, js_m)
    assert t_travel < m_travel - 1e-9, \
        f"min-travel did not beat maximin: {t_travel:.4f} vs {m_travel:.4f} rad"
    assert np.abs(np.diff(js_t)).sum() < np.abs(np.diff(js_m)).sum(), \
        "min-travel wandered at least as far in q7 as maximin did"
    # it stays in the valley; maximin climbs out of it and back
    assert js_t.max() == 0, f"min-travel left j = 0 (max j = {js_t.max()})"
    assert js_m.max() >= 6, f"maximin never climbed out (max j = {js_m.max()})"

    # ... and NEITHER of them spends a gate to do it
    for name, js in (("min_travel", js_t), ("maximin", js_m)):
        ii = np.arange(Ns)
        assert np.all(sheet["sigma"][ii, js] >= pwl.SIGMA_GATE), \
            f"{name} walked through a sub-gate sigma"
        assert np.all(sheet["margin"][ii, js] >= pwl.MARGIN_GATE), \
            f"{name} walked through a sub-gate margin"
        assert np.all(free[ii, js]), f"{name} left the gated free region"


def test_a_hole_in_the_cheap_lane_is_a_hard_gate_not_a_penalty():
    """The shortest path is refused where the gate is, and detours around it.

    Same band, with sigma driven below the gate for three cells across the lane
    min-travel wanted.  A soft penalty would pay the cost and walk through; a
    gate cannot be paid, so the path has to go round — and the assertion is
    that it goes round rather than that it gets more expensive.
    """
    sheet = _synthetic_sheet(hole=(20, 0, 3))
    free, clear, edges, etr = _band(sheet)
    js, last, _ = pwl._dense_dp_travel(free, clear, edges, etr, pwl.W_CLEARANCE,
                                       j_start=0, j_end=0)
    assert js is not None and last == len(free) - 1
    assert js[20] >= 3, f"the path walked into the hole (j = {js[20]} at i = 20)"
    ii = np.arange(len(free))
    assert np.all(sheet["sigma"][ii, js] >= pwl.SIGMA_GATE)
    assert np.all(free[ii, js])


# --------------------------------------------------------------------------
# real geometry: a few CSAIL strokes on one arm
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def csail_menus():
    """Menus + the eager plan for a handful of real segments on arm 31."""
    gif = Path(__file__).parents[1] / "assets/csail/csail_old_med.gif"
    px, _ = trace.trace_logo(str(gif))
    strokes, _ = trace.to_sheet(px, SHEET, margin=0.06,
                                target_width=1.2969246423461636,
                                offset=(0.1, 0.05))
    opts = dict(pen_ext=PEN)
    menus, segs = [], []
    for st in strokes:
        m = menu.stroke_menu(st["pts"], SPEC, opts, max_variants=99,
                             max_surcharge=None)
        if m.status != "ok":
            continue
        p = m.materialize(0)
        if p["status"] != "ok":
            continue
        menus.append(m)
        segs.append(dict(plan=p, pts=p["pts"], length=p["arc_len"],
                         direction=1, s_range=(0.0, 1.0)))
        if len(segs) >= 5:
            break
    assert len(segs) >= 4, "not enough real segments to test on"
    return menus, segs, opts


_CORPUS = []


def _corpus_menus():
    """Every CSAIL stroke arm 31 can carry, memoised.

    GENUINE PRUNING IS RARE, AND THAT IS WHY THE TEST HAS TO LOOK WIDELY.  Over
    the whole corpus only 3 of 469 candidate (entry, exit) pairs are actually
    unreachable — 0.64 % — so a handful of segments will not contain one, and a
    test that only ever sees connected bands is not testing pruning at all.
    (An earlier version of this file DID see plenty of "pruning", which turned
    out to be a bug in the separable edge price deleting certified edges; the
    numbers here are what pruning looks like once that is fixed.)
    """
    if not _CORPUS:
        gif = Path(__file__).parents[1] / "assets/csail/csail_old_med.gif"
        px, _ = trace.trace_logo(str(gif))
        strokes, _ = trace.to_sheet(px, SHEET, margin=0.06,
                                    target_width=1.2969246423461636,
                                    offset=(0.1, 0.05))
        for st in strokes:
            m = menu.stroke_menu(st["pts"], SPEC, dict(pen_ext=PEN),
                                 max_variants=99, max_surcharge=None)
            if m.status == "ok":
                _CORPUS.append(m)
    return _CORPUS


def test_the_menu_keeps_exactly_the_pairs_the_band_connects(csail_menus):
    """Pruning is the band DP's own answer, not a heuristic about it.

    For every menu, the candidate entries and exits are re-derived here and
    every (entry, exit) pair is re-tested with an independent forward sweep.
    A pair is in the menu IF AND ONLY IF that sweep leaves a finite cost at the
    exit — so nothing reachable is dropped and nothing unreachable is offered.
    The menus here are built with `max_variants=99` AND `max_surcharge=None` on
    purpose: with the shipped caps the absence of a pair would be ambiguous
    between "pruned" (the band does not connect it), "trimmed" (the menu was
    full) and "too slow" (the variant would have cost draw time), and this test
    is about the first of those three.
    """
    menus, _segs, _opts = csail_menus
    checked = pruned = 0
    for m in menus + _corpus_menus():
        have = {(v["sheet"], v["j0"], v["j1"]) for v in m.variants}
        Ns = m.ctx["Ns"]
        spanning = [sh for sh in m.ctx["order"] if sh["spans_s"]][:menu.MAX_SHEETS]
        for sh in spanning:
            free, clear, edges, etr = _band(sh)
            if not (free[0].any() and free[-1].any()):
                continue
            exits = menu.end_candidates(free[-1], menu.N_CAND)
            for j0 in menu.end_candidates(free[0], menu.N_CAND):
                A, _c, _p, last = pwl.travel_forward(free, clear, edges, etr,
                                                     pwl.W_CLEARANCE,
                                                     j_start=int(j0))
                for j1 in exits:
                    reachable = bool(last == Ns - 1 and np.isfinite(A[j1]))
                    present = (int(sh["id"]), int(j0), int(j1)) in have
                    assert present == reachable, (
                        f"sheet {sh['id']} pair ({j0}, {j1}): menu says "
                        f"{present}, the band says {reachable}")
                    checked += 1
                    pruned += (not reachable)
    assert checked > 0, "no pairs were examined"
    assert pruned > 0, ("nothing was pruned anywhere, so this test never "
                        "exercised the branch it exists for")


def test_lazy_materialisation_is_the_eager_plan_to_the_float(csail_menus):
    """A promise of endpoints, redeemed, is the plan planning eagerly gives.

    The menu advertises an entry and an exit configuration before anything is
    planned, and the sequencer optimises against them.  If materialising later
    produced a DIFFERENT plan, every transit second the DP costed would be
    fiction.  So: materialise, plan the same variant eagerly from scratch, and
    require the two to be identical arrays — then require the advertised
    endpoints to be the plan's actual endpoints.
    """
    menus, _segs, opts = csail_menus
    for m in menus[:3]:
        for k in range(min(2, len(m))):
            v = m.variants[k]
            lazy = m.materialize(k)
            eager = stroke_api.plan_stroke(
                m.poly, SPEC, dict(opts, j_start=v["j0"], j_end=v["j1"],
                                   sheet_id=v["sheet"]))
            assert lazy["status"] == eager["status"] == "ok"
            assert np.array_equal(lazy["qs"], eager["qs"]), \
                "the lazily materialised trajectory is not the eager one"
            assert np.array_equal(lazy["knots"], eager["knots"])
            assert lazy["validation"]["ok"] and eager["validation"]["ok"], \
                "a materialised variant is not independently validated"
            assert lazy["min_sigma"] >= pwl.SIGMA_GATE
            assert lazy["min_margin"] >= pwl.MARGIN_GATE
            # the numbers L2 optimised against are the numbers L1 delivered
            assert np.allclose(v["entry_q"], lazy["qs"][0], atol=1e-9), \
                "the advertised entry configuration is not the plan's"
            assert np.allclose(v["exit_q"], lazy["qs"][-1], atol=1e-9), \
                "the advertised exit configuration is not the plan's"


def test_one_variant_menus_reduce_the_cluster_dp_to_the_old_one(csail_menus):
    """The safety property: the new path is a generalisation, not a rewrite.

    Hand the cluster machinery one variant per segment — the plan the allocator
    would have had anyway — and it must produce the SAME cost matrix cell for
    cell, and the same tour, as `sequence.cost_matrix` + `sequence.held_karp`.
    If this ever fails, every A/B measured with the feature turned off is
    measuring a different sequencer rather than a different menu.
    """
    menus, segs, _opts = csail_menus
    n = len(segs)
    one = [menu.PlanMenu(s["plan"]) for s in segs]
    C1, T1, e1 = sequence.cluster_cost_matrix(SPEC, one, pen_ext=PEN)
    C0 = sequence.cost_matrix(SPEC, segs, pen_ext=PEN)

    assert np.array_equal(np.isfinite(C1), np.isfinite(C0)), \
        "the two matrices disagree about which transits are possible"
    fin = np.isfinite(C0)
    assert np.allclose(C1[fin], C0[fin], atol=0, rtol=0), \
        f"cost matrices differ by up to {np.abs(C1[fin] - C0[fin]).max():.3e} s"

    hk = sequence.held_karp(C0, n)
    cl = sequence.cluster_solve(C1, np.zeros_like(T1), e1)
    assert cl["order"] == hk["order"] and cl["dirs"] == hk["dirs"], \
        "the cluster DP picked a different tour from the same matrix"
    assert cl["cost"] == pytest.approx(hk["cost"], abs=1e-9)
    assert all(v == 0 for v in cl["variants"])


def test_the_cluster_dp_beats_single_variant_where_the_entry_matters(csail_menus):
    """The claim the whole L1->L2 interface is for.

    Same segments, same arm, same cost model; the only difference is whether
    the sequencer may choose which certified fiber to come off each stroke on.
    It must buy transit seconds, and it must buy RECONFIGURATION — the sum of
    ||q_exit - q_entry_next||_inf that the floored transit beats hide, and the
    thing a person watching the rig actually sees.
    """
    menus, segs, opts = csail_menus
    n = len(segs)
    assert max(len(m) for m in menus) > 1, \
        "every menu is a singleton here; this instance cannot test the feature"

    one = [menu.PlanMenu(s["plan"]) for s in segs]
    C1, T1, e1 = sequence.cluster_cost_matrix(SPEC, one, pen_ext=PEN)
    single = sequence.cluster_solve(C1, T1, e1)

    Cm, Tm, em = sequence.cluster_cost_matrix(SPEC, menus, pen_ext=PEN)
    multi = sequence.cluster_solve(Cm, Tm, em)

    assert multi["cost"] <= single["cost"] + 1e-9, \
        (f"more choice cost more transit: {multi['cost']:.4f} s vs "
         f"{single['cost']:.4f} s — the singleton menus are a SUBSET of the "
         f"full ones, so this is impossible unless the DP is wrong")
    assert multi["cost"] < single["cost"] - 1e-6, \
        (f"the cluster DP found nothing to win here ({multi['cost']:.4f} s vs "
         f"{single['cost']:.4f} s); the instance no longer tests the feature")

    def reconfig(e, nodes):
        return float(sum(np.max(np.abs(e["exi_q"][a] - e["ent_q"][b]))
                         for a, b in zip(nodes[:-1], nodes[1:])))

    r1, rm = reconfig(e1, single["nodes"]), reconfig(em, multi["nodes"])
    assert rm < r1 - 1e-6, \
        f"reconfiguration did not fall: {rm:.3f} rad vs {r1:.3f} rad"


# --------------------------------------------------------------------------
# the large-n fallback, where Held-Karp is not an option
# --------------------------------------------------------------------------
def _cluster_instance(n, V, seed, spread=0.9):
    """A synthetic (segment, direction, variant) instance. -> (C, e).

    Built the way `test_sequence.py` builds its matrices — arithmetic, no
    kinematics — but through the REAL cost shape: a lift beat, a
    `writing.dq_time_many` hover-to-hover move floored by the pen-tip hop, and
    a lower beat.  The floor is the whole point: it is what makes a large share
    of the pairs cost exactly the same, which is the regime the variant move
    exists to exploit and the regime a synthetic matrix of pure distances would
    not reproduce.
    """
    from aris_sixarm import writing as W
    rng = np.random.default_rng(seed)
    nv = [V] * n
    base = np.concatenate([[0], np.cumsum([2 * k for k in nv])]).astype(int)
    N = int(base[-1])
    seg = np.repeat(np.arange(n), 2 * V)
    var = np.tile(np.repeat(np.arange(V), 2), n)
    dirn = np.tile(np.array([0, 1]), n * V)

    xy0 = rng.uniform(0, 1, (n, 2))                      # the two ends, on paper
    xy1 = xy0 + rng.uniform(-0.08, 0.08, (n, 2))
    q0 = rng.uniform(-spread, spread, (n, V, 7))         # per-variant postures
    q1 = rng.uniform(-spread, spread, (n, V, 7))
    ent_q, exi_q = np.zeros((N, 7)), np.zeros((N, 7))
    ent_xy, exi_xy = np.zeros((N, 2)), np.zeros((N, 2))
    for k in range(N):
        i, v, d = int(seg[k]), int(var[k]), int(dirn[k])
        ent_q[k], exi_q[k] = (q0[i, v], q1[i, v]) if d == 0 else (q1[i, v], q0[i, v])
        ent_xy[k], exi_xy[k] = (xy0[i], xy1[i]) if d == 0 else (xy1[i], xy0[i])

    e = dict(seg=seg, var=var, dirn=dirn, base=base, nv=nv, n=n, N=N,
             ent_q=ent_q, exi_q=exi_q, ent_xy=ent_xy, exi_xy=exi_xy,
             ent_h=ent_q, exi_h=exi_q, surcharge=np.zeros(N))
    hop = np.linalg.norm(ent_xy[None, :, :] - exi_xy[:, None, :], axis=-1)
    floor = np.maximum(sequence.T_TRAVEL_MIN, hop / sequence.TRANSIT_SPEED)
    C = np.full((N + 1, N + 1), np.inf)
    C[:N, :N] = (sequence.T_LIFT_F
                 + W.dq_time_many(exi_q, ent_q, sequence.QD_FRAC, floor)
                 + sequence.T_LOWER_F)
    C[:N, :N][seg[:, None] == seg[None, :]] = np.inf
    C[N, :N] = sequence.T_HOME_F
    C[:N, N] = sequence.T_HOME_F
    return C, e


def _subset_to_variant0(C, e):
    """The same instance with every segment pinned to its first variant."""
    keep = np.flatnonzero(e["var"] == 0)
    idx = np.concatenate([keep, [e["N"]]])
    n, V = e["n"], 1
    sub = dict(seg=e["seg"][keep], var=np.zeros(len(keep), int),
               dirn=e["dirn"][keep], nv=[1] * n, n=n, N=len(keep),
               base=np.arange(n + 1) * 2,
               ent_q=e["ent_q"][keep], exi_q=e["exi_q"][keep],
               surcharge=np.zeros(len(keep)))
    return C[np.ix_(idx, idx)], sub


def test_the_large_n_fallback_reselects_variants_and_scales():
    """300 segments x 3 variants: Held-Karp is refused, the search still wins.

    1800 nodes, a 3.24 M-cell matrix, and 2^300 states — so `cluster_solve` has
    to take the local-search branch, and the assertion is that the branch is
    (a) actually taken, (b) valid (every segment drawn exactly once), (c) inside
    its wall-clock budget, and (d) worth taking.

    (d) IS MEASURED AGAINST THE RIGHT BASELINE, WHICH IS NOT THE GREEDY SEED.
    On a dense random matrix nearest-neighbour is already within about a per
    cent of what any local search reaches, so "beats NN by 5 %" — the bar the
    geometric instance in `test_sequence.py` clears easily — measures the
    instance's structure and not this code.  What IS this code is the variant
    move, so the baseline here is the SAME search over the same segments with
    every variant frozen to the first: a strict subset of the choices, which
    the full menu therefore cannot lose to, and beats by about 6 %.
    """
    n, V = 300, 3
    C, e = _cluster_instance(n, V, seed=300)
    assert e["N"] == 2 * n * V

    r = sequence.cluster_solve(C, np.zeros_like(C), e, budget=2.0)
    assert r["method"] == "cluster_nn+2opt+oropt+variant", \
        f"the exact branch was taken at n = {n} ({r['method']})"
    assert sorted(r["order"]) == list(range(n)), "a segment was dropped or repeated"
    assert len(r["variants"]) == n and all(0 <= v < V for v in r["variants"])
    assert abs(sequence.cycle_cost(C, sequence.tour_of_nodes(r["nodes"], e["N"]))
               - r["cost"]) < 1e-6, "the reported cost is not the tour's cost"
    assert r["cost"] < r["nn_cost"] - 1e-6, "the descent improved nothing at all"
    assert r["wall"] < 8.0, f"search took {r['wall']:.1f} s against a 2 s budget"

    # the variant move earns its place: freeze the variants and it gets worse
    frozen_C, frozen_e = _subset_to_variant0(C, e)
    f = sequence.cluster_solve(frozen_C, np.zeros_like(frozen_C), frozen_e,
                               budget=2.0)
    assert r["cost"] < f["cost"] * 0.98, \
        (f"choosing among {V} variants bought less than 2 %: "
         f"{r['cost']:.3f} s vs {f['cost']:.3f} s frozen")
    # more than one variant is actually in use, i.e. the move fired
    assert len(set(r["variants"])) > 1, "the search never re-selected a variant"


def test_menus_never_lose_a_segment(csail_menus):
    """`build_menus` degrades to the certified plan, it does not drop work.

    A segment whose band offers no menu still has to be drawn.  The fallback is
    a one-variant `PlanMenu` around the plan the allocator already certified,
    so the count of menus always equals the count of segments and every one of
    them can be materialised.
    """
    _menus, segs, opts = csail_menus
    mus, stats = allocate.build_menus(segs, SPEC, opts)
    assert len(mus) == len(segs)
    assert stats["n_menu"] + stats["n_plan"] == len(segs)
    for m, s in zip(mus, segs):
        assert m.status == "ok" and len(m) >= 1
        p = m.materialize(0)
        assert p["status"] == "ok"


# --------------------------------------------------------------------------
# 6. the allocator and the sequencer price the same bag the same way
# --------------------------------------------------------------------------
def _bag(segs):
    """The fixture's segments, given the identity a cached menu is keyed on."""
    return [dict(s, stroke_id=i) for i, s in enumerate(segs)]


_SEQ_KW = dict(transit_speed=0.80, qd_frac=0.30, pen_ext=PEN,
               q_start=None, return_home=True)


@pytest.mark.parametrize("cluster", [False, True])
def test_the_price_the_allocator_pays_is_the_tour_that_will_be_drawn(
        csail_menus, cluster):
    """The costing hook's whole reason to exist (`allocate.cost_model`).

    `rebalance` decides who draws what by pricing candidate bags in seconds,
    and `allocate` then hands each arm's bag to a sequencer.  Those used to be
    two independent choices of cost model — the balancer always priced with
    `sequence.solve`, while the pass that ran could be the cluster DP — so the
    balancer balanced a load the sequencer then moved.  Here BOTH halves come
    off one object, and this pins the identity for each model it offers: the
    seconds a bag is priced at are the seconds its own DP reaches on it.
    """
    _menus, raw, opts = csail_menus
    segs = _bag(raw)
    model = allocate.cost_model(cluster)
    assert model.cluster is cluster
    draw = [writing.segment_draw_time(SPEC, s, 0.12, _SEQ_KW["qd_frac"],
                                      H_INV_DEFAULT, PEN) for s in segs]

    price = model.load(31, SPEC, segs, draw, **_SEQ_KW)
    seq = model.sequence(31, segs, SPEC, "opt", opts, dict(_SEQ_KW))
    assert seq["n_refused"] == 0, \
        "a refused reversal makes the tour differ from the one that was priced"

    if not cluster:
        assert price - sum(draw) == pytest.approx(seq["cost"], abs=1e-9)
        return

    # the cluster price is the DP's own objective, decomposed: transit, plus
    # the interior surcharge of the variant each segment is entered on, which
    # is extra DRAW time and is charged at the same `w_surcharge / qd_frac`
    # `sequence.cluster_cost_matrix` charges it at
    mus = model.menus(31, SPEC, segs, opts)
    sur = sum(float(mus[k].variants[v]["surcharge"])
              for k, v in zip(seq["order"], seq["variants"]))
    extra = (sequence.W_SURCHARGE / _SEQ_KW["qd_frac"]) * sur
    assert price - sum(draw) - extra == pytest.approx(seq["cluster_cost"],
                                                      abs=1e-9)


def test_pricing_a_cluster_bag_with_the_single_variant_dp_is_a_different_number(
        csail_menus):
    """And the defect the hook closes was worth closing.

    If the two models happened to agree on every bag, threading one object
    through both stages would be tidiness rather than a fix.  They do not: the
    cluster DP may choose the fiber each stroke is entered on, so it reaches a
    tour the single-variant DP cannot, and pricing one bag under both models
    gives two different numbers.  That difference is exactly what the balancer
    used to be blind to (`docs/BENCH.md`: 1.40x -> 1.49x phase-2 imbalance at
    the rig's own draw speed).
    """
    _menus, raw, opts = csail_menus
    segs = _bag(raw)
    draw = [writing.segment_draw_time(SPEC, s, 0.12, _SEQ_KW["qd_frac"],
                                      H_INV_DEFAULT, PEN) for s in segs]
    flat = allocate.cost_model(False).load(31, SPEC, segs, draw, **_SEQ_KW)
    clus = allocate.cost_model(True).load(31, SPEC, segs, draw, opts=opts,
                                          **_SEQ_KW)
    assert clus < flat - 1e-6, \
        (f"the cluster model priced this bag at {clus:.4f} s and the "
         f"single-variant model at {flat:.4f} s; if these are equal the "
         "balancer cannot be blind to the difference and this test is stale")


def test_the_allocation_records_which_model_priced_it():
    """A result carries the name of the model that balanced it, not a guess."""
    assert allocate.cost_model(None).name == (
        "cluster" if allocate.CLUSTER else "segment")
    assert allocate.cost_model(True).name == "cluster"
    assert allocate.cost_model(False).name == "segment"
    # a model instance passes through unchanged, so `rebalance` and `allocate`
    # cannot end up holding two of them
    m = allocate.cost_model(True)
    assert allocate.cost_model(m) is m


def test_per_span_cluster_endpoints_stack_into_the_whole_bag_exactly(
        csail_menus):
    """What makes pricing with the cluster DP affordable at all.

    The balancer re-prices a bag once per candidate move, and a bag is a set of
    SPANS: solving the hover IK for every variant of every segment each time
    would make the cluster model cost what the cluster model was accused of
    costing.  `allocate._stack_cluster_ends` assembles the many-segment node
    list out of memoised one-segment ones, and that is only legitimate if it
    reproduces `sequence.cluster_endpoints` on the whole bag — field for field,
    not approximately.
    """
    menus, _segs, _opts = csail_menus
    full = sequence.cluster_endpoints(SPEC, menus, pen_ext=PEN)
    stacked = allocate._stack_cluster_ends(
        [sequence.cluster_endpoints(SPEC, [m], pen_ext=PEN) for m in menus])
    assert set(stacked) == set(full)
    for k, v in full.items():
        got = stacked[k]
        if isinstance(v, np.ndarray):
            assert np.asarray(got).shape == v.shape, f"{k}: shape"
            assert np.array_equal(np.asarray(got), v), f"{k}: values"
        else:
            assert list(got) == list(v) if isinstance(v, list) else got == v, k

    C0, _T0, _e0 = sequence.cluster_cost_matrix(SPEC, menus, pen_ext=PEN)
    C1, _T1, _e1 = sequence.cluster_cost_matrix(SPEC, menus, pen_ext=PEN,
                                                ends=stacked)
    assert np.array_equal(np.isfinite(C0), np.isfinite(C1))
    fin = np.isfinite(C0)
    assert np.array_equal(C0[fin], C1[fin])
