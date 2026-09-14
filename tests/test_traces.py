"""CUT AT EVERY TRANSITION, THEN MERGE: the DP, the seams and the minimum.

NO ENVIRONMENT VARIABLES AND NO ATLAS HERE.  Every capability map below is
rasterised from rectangles by `traces.coverage_from_rects`, which is the same
`Coverage` object and the same lookup the shipped atlas produces -- so these
pin the ALGORITHM, at a geometry a reader can check by hand, and a re-sweep of
the atlas can never turn one of them red for a reason that is not a bug.

The one thing that cannot be checked by hand is the DP's optimality, so it is
checked against brute force on every tiny case a small alphabet can produce.
"""
import itertools

import numpy as np
import pytest

from aris_sixarm import traces as T

# A toy paper: two arms, a left-right split with a controllable overlap.  Arm 13
# owns x <= LEFT, arm 17 owns x >= RIGHT.  LEFT < RIGHT is a hard edge with a
# hole between; LEFT > RIGHT is an overlap zone [RIGHT, LEFT].
Y = 0.50


def two_arm_cov(left, right, grid=T.GRID):
    return T.coverage_from_rects(
        {13: [(0.0, 0.0, left, 1.0)], 17: [(right, 0.0, 2.0, 1.0)]},
        grid=grid, extent=(0.0, 0.0, 2.0, 1.0))


def flat_pattern(arms=(13, 17)):
    """One stage, each arm everywhere -- so the CELLS never bind, only coverage."""
    return T.Pattern("flat", tuple(
        T.StageCell(0, a, ((0.0, 0.0, 2.0, 1.0),), "ALL") for a in arms))


def hline(x0, x1, y=Y, n=2):
    return np.column_stack([np.linspace(x0, x1, n), np.full(n, y)])


# ---------------------------------------------------------------------------
# 1.  THE DP IS OPTIMAL -- against brute force, on every tiny case
# ---------------------------------------------------------------------------
def _fake_cap(n_states):
    """A `Capability` with no geometry: only `n_states` and `members` are used."""
    cells = tuple(T.StageCell(k, 10 + k, ((0.0, 0.0, 1.0, 1.0),)) for k in
                  range(n_states))
    cov = T.coverage_from_rects({c.arm: [] for c in cells},
                                extent=(0.0, 0.0, 0.1, 0.1))
    return T.Capability(cov, T.Pattern("fake", cells), cells)


@pytest.mark.parametrize("n_states,n_atoms", [(2, 4), (2, 6), (3, 4), (3, 5),
                                              (4, 4)])
def test_dp_matches_brute_force_on_every_tiny_case(n_states, n_atoms):
    """Exhaustive: every assignment of capability sets to a short chain.

    The DP's contraction -- "stay, or switch from the cheapest predecessor" --
    is the whole reason this is milliseconds and not minutes, and it is exactly
    the step a hand-rolled greedy gets wrong.  So it is checked against the
    definition, on all 15^4 .. 15^6 capability chains the alphabet admits.
    """
    cap = _fake_cap(n_states)
    sets = [b for b in range(1, 1 << n_states)]
    n_checked = 0
    for combo in itertools.product(sets, repeat=n_atoms):
        atoms = [T.Atom(i * 0.1, (i + 1) * 0.1, b) for i, b in enumerate(combo)]
        got = T.runs_of(atoms, T.solve_line(atoms, cap))
        assert len(got) == T.brute_force_line(atoms, cap), combo
        for k, a, b in got:                       # and it is a LEGAL assignment
            assert all(atoms[i].bits >> k & 1 for i in range(a, b))
        n_checked += 1
    assert n_checked == len(sets) ** n_atoms


def test_dp_is_optimal_with_uncovered_atoms_too():
    """An atom nobody can draw is forced, and it breaks the run."""
    cap = _fake_cap(2)
    for combo in itertools.product([0, 1, 2, 3], repeat=5):
        atoms = [T.Atom(i * 0.1, (i + 1) * 0.1, b) for i, b in enumerate(combo)]
        assign = T.solve_line(atoms, cap)
        assert len(T.runs_of(atoms, assign)) == T.brute_force_line(atoms, cap)
        for i, b in enumerate(combo):
            assert (assign[i] == T.UNCOVERED) == (b == 0)


def test_the_tie_break_costs_no_pieces_and_moves_load():
    """`balance_w = 0` buys balance only where the piece count does not care."""
    cap = _fake_cap(2)
    atoms = [T.Atom(0.0, 1.0, 0b11)]            # either state, one piece either way
    assert T.solve_line(atoms, cap, weight=[0.0, 0.0]) == [0]
    assert T.solve_line(atoms, cap, weight=[5.0, 0.0]) == [1]
    assert T.solve_line(atoms, cap, weight=[0.0, 5.0]) == [0]
    # ...and it never buys balance WITH a piece: state 1 is loaded to the moon,
    # but taking state 0 for the middle atom would cost two changes.
    atoms = [T.Atom(0.0, 1.0, 0b10), T.Atom(1.0, 2.0, 0b11),
             T.Atom(2.0, 3.0, 0b10)]
    assert T.solve_line(atoms, cap, weight=[0.0, 9.0]) == [1, 1, 1]
    # a real weight WILL buy it, which is the whole difference between the two
    assert T.solve_line(atoms, cap, weight=[0.0, 9.0], balance_w=1.0) == [1, 0, 1]


# ---------------------------------------------------------------------------
# 2.  A LINE INSIDE ONE ARM'S CELL IS ONE PIECE
# ---------------------------------------------------------------------------
def test_a_line_fully_inside_one_cell_is_one_piece():
    cap = T.capability(two_arm_cov(1.2, 1.4), flat_pattern())
    lp = T.plan_line(hline(0.20, 1.00), cap)
    assert len(lp.atoms) == 1
    assert lp.n_pieces == 1
    p = lp.pieces[0]
    assert p.arm == 13 and p.stage == 0
    assert lp.seams == []
    assert p.draw0 == pytest.approx(0.0) and p.draw1 == pytest.approx(lp.length)
    assert p.length == pytest.approx(0.80, abs=1e-9)


def test_a_line_inside_one_cell_stays_one_piece_under_the_zigzag():
    """The same claim on the real pattern shape: a row band holds a whole line."""
    band = T.row_band(1)                       # y in [1.4103, 2.2204]
    cov = T.coverage_from_rects({71: [T.BLOCK]}, extent=(0.0, 0.0, 2.0, 3.8))
    cap = T.capability(cov, T.Pattern("one", (T.StageCell(0, 71, (band,), "R1"),)))
    y = 0.5 * (band[1] + band[3])
    lp = T.plan_line(hline(0.30, 1.40, y=y), cap)
    assert lp.n_pieces == 1 and len(lp.atoms) == 1
    assert lp.pieces[0].arm == 71


# ---------------------------------------------------------------------------
# 3.  A CROSSING IN AN OVERLAP ZONE: one hand-over, seam in the middle, delta
# ---------------------------------------------------------------------------
def test_an_overlap_crossing_is_one_handover_seamed_in_the_middle():
    """Arm 13 to x <= 1.20, arm 17 from x >= 0.80: the overlap is [0.80, 1.20].

    Both cells are rasterised on the 2 cm lattice, so the overlap the sampler
    sees is [0.79, 1.21] -- a cell is capable when its CENTRE is inside, and a
    point rounds to its nearest centre.  The seam therefore lands at the middle
    of THAT, which is 1.00 either way, and the test asserts the middle rather
    than the endpoints for exactly that reason.
    """
    cap = T.capability(two_arm_cov(1.20, 0.80), flat_pattern())
    lp = T.plan_line(hline(0.20, 1.80), cap)
    assert lp.n_pieces == 2
    assert [p.arm for p in lp.pieces] == [13, 17]
    assert len(lp.seams) == 1
    s = lp.seams[0]
    assert s.kind == "overlap"
    assert s.hi - s.lo == pytest.approx(0.42, abs=2e-3)      # 0.40 + one cell
    assert s.s == pytest.approx(0.5 * (s.lo + s.hi))
    # the line starts at x = 0.20, so arc length s == x - 0.20
    assert s.s + 0.20 == pytest.approx(1.00, abs=2e-3)
    # ...and each side laps delta past it, in its own drawing direction
    a, b = lp.pieces
    assert a.draw1 == pytest.approx(s.s + T.OVERDRAW_M)
    assert b.draw0 == pytest.approx(s.s - T.OVERDRAW_M)
    assert a.draw1 - b.draw0 == pytest.approx(2 * T.OVERDRAW_M)
    assert a.draw0 == pytest.approx(0.0)
    assert b.draw1 == pytest.approx(lp.length)
    # the overdraw is INK, not bookkeeping: the two pieces are longer than the line
    assert a.length + b.length == pytest.approx(lp.length + 2 * T.OVERDRAW_M)
    # and neither is asked to draw outside the zone it certifies
    assert a.draw1 <= s.hi + 1e-12 and b.draw0 >= s.lo - 1e-12


def test_the_lap_never_leaves_the_overlap_zone():
    """A 6 mm overlap cannot absorb two 5 mm laps, and the lap is what yields."""
    cap = T.capability(two_arm_cov(1.002, 0.998, grid=0.002), flat_pattern())
    lp = T.plan_line(hline(0.20, 1.80), cap, T.Options(ds=0.0005))
    assert lp.n_pieces == 2
    s = lp.seams[0]
    assert s.kind == "overlap" and s.hi - s.lo < 2 * T.OVERDRAW_M
    a, b = lp.pieces
    assert a.draw1 == pytest.approx(s.hi) and b.draw0 == pytest.approx(s.lo)


# ---------------------------------------------------------------------------
# 4.  A HARD EDGE: two pieces meeting at the transition point exactly
# ---------------------------------------------------------------------------
def test_a_hard_edge_crossing_is_two_pieces_meeting_exactly():
    """Arm 13 stops where arm 17 starts: no overlap, so nothing to lap into.

    On a 2 mm lattice arm 13's last capable cell centre is 0.998 and arm 17's
    first is 1.000, so the transition -- the point where a sample stops rounding
    to 13's cell and starts rounding to 17's -- is 0.999 exactly.
    """
    cap = T.capability(two_arm_cov(0.999, 0.999, grid=0.002), flat_pattern())
    lp = T.plan_line(hline(0.20, 1.80), cap, T.Options(ds=0.0005))
    assert lp.n_pieces == 2
    assert [p.arm for p in lp.pieces] == [13, 17]
    s = lp.seams[0]
    assert s.kind == "hard"
    assert s.lo == pytest.approx(s.hi) and s.s == pytest.approx(s.lo)
    a, b = lp.pieces
    assert a.draw1 == pytest.approx(b.draw0, abs=1e-9)      # they MEET
    assert a.draw1 == pytest.approx(s.s, abs=1e-9)          # ...at the transition
    # the transition is at x = 0.999, to the bisection tolerance and not to the
    # 0.5 mm sample, and no ink is added or lost at a hard edge
    assert a.draw1 + 0.20 == pytest.approx(0.999, abs=2e-5)
    assert a.length + b.length == pytest.approx(lp.length, abs=1e-9)


def test_a_hole_between_the_two_cells_is_a_gap_not_a_handover():
    """Neither arm reaches (1.01, 1.19): two pieces, and the ink between is lost.

    Arm 13's cells stop at centre 1.00 and arm 17's start at centre 1.20, so on
    the 2 cm lattice the hole a sample sees is 0.18 m wide, not 0.20.
    """
    cap = T.capability(two_arm_cov(1.00, 1.20), flat_pattern())
    lp = T.plan_line(hline(0.20, 1.80), cap)
    assert lp.n_pieces == 2
    assert [s.kind for s in lp.seams] == ["gap", "gap"]
    d = T.summarise(T.Plan(cap, [lp]))
    assert d["handovers"] == 0 and d["gaps"] == 2
    assert d["uncovered_m"] == pytest.approx(0.18, abs=2e-3)
    assert d["overdraw_m"] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 5.  A DEAD BAND WITH NO CAPABLE ARM IN THE STAGE DEFERS TO A SEAM STAGE
# ---------------------------------------------------------------------------
def test_a_line_through_the_dead_band_is_deferred_to_a_seam_stage():
    """The dead band belongs to NO main stage, and the seam stages come back for it.

    A vertical line down the middle of the paper crosses row band 0, the first
    dead band, row band 1, the second dead band and row band 2.  Every arm
    certifies everything here, so the ONLY thing cutting this line is the stage
    cells -- which is the point: the pieces in the dead bands carry a SEAM
    stage, and the pieces either side carry a main one.
    """
    cov = T.coverage_from_rects({a: [T.BLOCK] for a in T.ARMS},
                                extent=(0.0, 0.0, 2.0, 3.8))
    cap = T.capability(cov, T.zigzag_pattern())
    x = 0.5 * (T.BLOCK[0] + T.BLOCK[2])
    line = np.array([[x, T.BLOCK[1]], [x, T.BLOCK[3]]])
    lp = T.plan_line(line, cap, T.Options(ds=0.002))
    assert all(k != T.UNCOVERED for k in lp.assign), "nothing is undrawable here"
    stages = [p.stage for p in lp.pieces]
    assert lp.n_pieces == 5, stages
    # the dead bands are the 2nd and 4th piece, and they are SEAM stages (>= 2)
    assert stages[1] >= 2 and stages[3] >= 2
    assert stages[0] <= 1 and stages[2] <= 1 and stages[4] <= 1
    for k in (1, 3):
        p = lp.pieces[k]
        band = T.seam_band(0) if k == 1 else T.seam_band(1)
        mid = 0.5 * (p.s0 + p.s1) + T.BLOCK[1]
        assert band[1] <= mid <= band[3]
        assert p.run_length == pytest.approx(T.DEAD_BAND_M, abs=3e-3)
    # every dead-band crossing is a HARD edge -- the cells abut by construction,
    # so there is nothing to lap into and the seams are as good as the calibration
    assert [s.kind for s in lp.seams] == ["hard"] * 4


def test_the_seam_stages_are_the_only_drawers_of_the_dead_band():
    """No main-stage cell contains a dead-band point, and a seam cell does."""
    pat = T.zigzag_pattern()
    y = T.Y_CUT[0]
    for c in pat.cells:
        inside = bool(c.contains(0.9, y))
        assert inside == (c.stage >= 2 and c.region[0][1] <= y <= c.region[0][3])
    assert any(c.contains(0.9, y) for c in pat.cells if c.stage >= 2)


def test_the_zigzag_is_one_arm_per_row_with_the_columns_alternating():
    pat = T.zigzag_pattern()
    assert pat.n_stages == 8            # two main stages, six seam stages
    assert {c.arm for c in pat.stage(0)} == {13, 71, 2}
    assert {c.arm for c in pat.stage(1)} == {17, 31, 97}
    for s in (0, 1):                       # one arm per row, columns alternating
        rows = sorted(T.ROW_OF[c.arm] for c in pat.stage(s))
        cols = [T.COL_OF[c.arm] for c in
                sorted(pat.stage(s), key=lambda c: T.ROW_OF[c.arm])]
        assert rows == [0, 1, 2]
        assert cols in ([0, 1, 0], [1, 0, 1])
    # the dead band is 0.40 m and the row bands and seams tile the block exactly
    assert T.row_band(0)[3] == pytest.approx(T.Y_CUT[0] - 0.20)
    assert T.row_band(1)[1] == pytest.approx(T.Y_CUT[0] + 0.20)
    assert T.seam_band(0)[3] - T.seam_band(0)[1] == pytest.approx(T.DEAD_BAND_M)
    covered = sum(r[3] - r[1] for r in
                  [T.row_band(j) for j in (0, 1, 2)] +
                  [T.seam_band(k) for k in (0, 1)])
    assert covered == pytest.approx(T.BLOCK[3] - T.BLOCK[1])


def test_the_duplicate_seam_stages_are_one_state():
    """Stages 4 and 5 re-offer arms 97 and 2 exactly what stages 2 and 3 do,
    and stages 6 and 7 re-offer arms 13 and 17 exactly what stages 2 and 3 do.

    The seam correction of docs/V2_WORKCELLS.md section 4b adds two stages and
    four cells, but only TWO of those four are new states: 31 on SEAM1 and 71
    on SEAM1, which are what covers y in [2.24, 2.40].  The other two are 13
    and 17 back on SEAM0, which they were already offered in stages 2 and 3, so
    they merge and cost neither a state nor a piece."""
    cov = T.coverage_from_rects({a: [T.BLOCK] for a in T.ARMS},
                                extent=(0.0, 0.0, 2.0, 3.8))
    cap = T.capability(cov, T.zigzag_pattern())
    assert len(T.zigzag_pattern().cells) == 18
    assert cap.n_states == 14
    assert sorted(cap.merged) == [(4, 2), (5, 3), (6, 2), (7, 3)]


def test_the_two_extra_seam_stages_put_a_middle_arm_on_seam1():
    """The whole point of stages 6 and 7 (docs' 7 and 8): SEAM1's floor is
    reachable only by a MIDDLE arm, and the four-seam version offered SEAM1 to
    nobody but 2 and 97."""
    pat = T.zigzag_pattern()
    assert {c.arm for c in pat.stage(6)} == {13, 31}
    assert {c.arm for c in pat.stage(7)} == {17, 71}
    seam1 = T.seam_band(1)
    on_seam1 = {c.arm for c in pat.cells if c.stage >= 2 and c.region[0] == seam1}
    assert on_seam1 == {2, 97, 31, 71}
    # ...and no stage ever puts a same-row (transverse) pair in the air: that
    # pair is at -262 mm however the paper is cut (docs/V2_WORKCELLS.md 1-2)
    for s in range(pat.n_stages):
        rows = [T.ROW_OF[c.arm] for c in pat.stage(s)]
        assert len(rows) == len(set(rows)), f"stage {s} has a same-row pair"
    assert pat.name.endswith("+6seams")


# ---------------------------------------------------------------------------
# 6.  THE MINIMUM PIECE
# ---------------------------------------------------------------------------
def test_a_short_piece_is_absorbed_when_a_neighbour_can_take_it():
    """Three atoms; the middle one is 4 mm and both neighbours could draw it."""
    cap = _fake_cap(2)
    atoms = [T.Atom(0.000, 0.300, 0b01), T.Atom(0.300, 0.304, 0b11),
             T.Atom(0.304, 0.310, 0b10)]
    assign = T.solve_line(atoms, cap)
    assert len(T.runs_of(atoms, assign)) == 2       # the DP already merges it
    # force the awkward case: the middle atom assigned against both neighbours
    bad = [0, 1, 1]
    assert len(T.runs_of(atoms, bad)) == 2
    got = T.absorb_short(atoms, bad, cap, min_piece_m=0.010)
    assert len(T.runs_of(atoms, got)) == 2
    # a 6 mm run of state 1 at the end is under the minimum but state 0 cannot
    # draw its last atom, so it STAYS -- the alternative is not drawing it
    assert got[2] == 1


def test_a_short_piece_no_neighbour_can_draw_is_kept():
    cap = _fake_cap(3)
    atoms = [T.Atom(0.00, 0.30, 0b001), T.Atom(0.30, 0.304, 0b010),
             T.Atom(0.304, 0.60, 0b100)]
    assign = T.absorb_short(atoms, T.solve_line(atoms, cap), cap, 0.010)
    assert len(T.runs_of(atoms, assign)) == 3
    assert [k for k, _, _ in T.runs_of(atoms, assign)] == [0, 1, 2]


def test_absorption_never_raises_the_piece_count():
    """On random chains: the minimum never costs a piece, and often saves one."""
    rng = np.random.default_rng(7)
    cap = _fake_cap(3)
    saved = 0
    for _ in range(400):
        n = int(rng.integers(2, 9))
        atoms, s = [], 0.0
        for _ in range(n):
            ln = float(rng.choice([0.004, 0.008, 0.05, 0.2]))
            atoms.append(T.Atom(s, s + ln, int(rng.integers(1, 8))))
            s += ln
        a0 = T.solve_line(atoms, cap)
        a1 = T.absorb_short(atoms, a0, cap, 0.010)
        n0, n1 = len(T.runs_of(atoms, a0)), len(T.runs_of(atoms, a1))
        assert n1 <= n0
        saved += n0 - n1
        for k, i0, i1 in T.runs_of(atoms, a1):    # still a LEGAL assignment
            assert all(atoms[i].bits >> k & 1 for i in range(i0, i1))
    assert saved == 0, "the DP's answer already has no absorbable short piece"


# ---------------------------------------------------------------------------
# 7.  THE SEGMENTATION ITSELF
# ---------------------------------------------------------------------------
def test_the_cut_lands_on_the_transition_not_on_a_sample():
    """`ds` is 4 mm and the bisection takes the cut to 0.01 mm of the truth."""
    cap = T.capability(two_arm_cov(0.999, 0.999, grid=0.002), flat_pattern())
    atoms, _, _ = T.atoms_of(hline(0.0, 2.0), cap, ds=0.004, tol=1e-5)
    cuts = [a.s1 for a in atoms[:-1]]
    assert len(cuts) == 1
    assert cuts[0] == pytest.approx(0.999, abs=2e-5)   # NOT to the 4 mm sample
    assert abs(cuts[0] - 0.999) < 1e-4                 # a sample would be 40x off


def test_atoms_tile_the_line_and_carry_distinct_capability_sets():
    cov = T.coverage_from_rects(
        {13: [(0.0, 0.0, 0.70, 1.0)], 17: [(0.50, 0.0, 1.30, 1.0)],
         71: [(1.10, 0.0, 2.00, 1.0)]}, extent=(0.0, 0.0, 2.0, 1.0))
    cap = T.capability(cov, flat_pattern((13, 17, 71)))
    atoms, _, cum = T.atoms_of(hline(0.0, 2.0), cap)
    assert atoms[0].s0 == 0.0
    assert atoms[-1].s1 == pytest.approx(float(cum[-1]))
    for a, b in zip(atoms, atoms[1:]):
        assert a.s1 == b.s0
        assert a.bits != b.bits
    assert len(atoms) == 5          # 13 | 13+17 | 17 | 17+71 | 71


def test_sub_polyline_is_exact_at_the_ends_and_keeps_the_vertices():
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    cum = T.cumlen(pts)
    assert cum[-1] == pytest.approx(2.0)
    sub = T.sub_polyline(pts, cum, 0.5, 1.5)
    assert sub[0] == pytest.approx([0.5, 0.0])
    assert sub[-1] == pytest.approx([1.0, 0.5])
    assert len(sub) == 3 and sub[1] == pytest.approx([1.0, 0.0])
    assert T.cumlen(sub)[-1] == pytest.approx(1.0)


def test_piece_points_are_the_ink_the_arm_is_handed():
    cap = T.capability(two_arm_cov(1.20, 0.80), flat_pattern())
    lp = T.plan_line(hline(0.20, 1.80, n=9), cap)
    for k, p in enumerate(lp.pieces):
        P = lp.piece_points(k)
        assert T.cumlen(P)[-1] == pytest.approx(p.length, abs=1e-9)
        assert cap.states_at(P[:, 0], P[:, 1]).min() >> p.state & 1, \
            "an arm is never handed a point it cannot draw"


# ---------------------------------------------------------------------------
# 8.  THE WHOLE-PICTURE ACCOUNTING
# ---------------------------------------------------------------------------
def test_the_summary_adds_up():
    cov = T.coverage_from_rects(
        {13: [(0.0, 0.0, 1.10, 1.0)], 17: [(0.90, 0.0, 2.00, 1.0)]},
        extent=(0.0, 0.0, 2.0, 1.0))
    cap = T.capability(cov, flat_pattern())
    lines = [hline(0.1, 1.9, y=y) for y in np.linspace(0.1, 0.9, 9)]
    plan = T.plan_lines(lines, cap)
    d = plan.summary()
    assert d["n_lines"] == 9
    assert d["n_pieces"] == 18 and d["handovers"] == 9
    assert d["handovers_overlap"] == 9 and d["handovers_hard"] == 0
    assert d["uncovered_m"] == pytest.approx(0.0)
    assert d["covered_frac"] == pytest.approx(1.0)
    assert d["drawn_m"] == pytest.approx(d["ink_m"] + 9 * 2 * T.OVERDRAW_M)
    assert sum(d["load_m"].values()) == pytest.approx(d["drawn_m"], abs=1e-3)
    assert sum(d["arm_load_m"].values()) == pytest.approx(d["drawn_m"], abs=1e-3)
    # nine identical lines, two arms, and the tie-break has nothing to tie:
    # every line's split is forced by the overlap, so the loads are equal
    assert d["arm_load_m"]["13"] == pytest.approx(d["arm_load_m"]["17"], abs=1e-3)


def test_staging_can_only_cost_pieces_never_save_them():
    """The single-stage floor is a floor: a cell restriction cannot help.

    Every state the zigzag offers is a subset of a state the single-stage
    pattern offers (same arm, smaller region), so any assignment legal under
    the zigzag is legal unstaged with the same number of changes.
    """
    cov = T.coverage_from_rects({a: [T.BLOCK] for a in T.ARMS},
                                extent=(0.0, 0.0, 2.0, 3.8))
    rng = np.random.default_rng(3)
    lines = [np.column_stack([
        rng.uniform(T.BLOCK[0], T.BLOCK[2], 4),
        rng.uniform(T.BLOCK[1], T.BLOCK[3], 4)]) for _ in range(40)]
    a = T.plan_lines(lines, T.capability(cov, T.zigzag_pattern())).summary()
    b = T.plan_lines(lines, T.capability(cov, T.single_stage_pattern())).summary()
    assert b["n_pieces"] <= a["n_pieces"]
    assert b["n_pieces"] == b["n_lines"], "unstaged, every line is one piece"


def test_a_line_off_the_paper_is_all_uncovered_and_draws_nothing():
    cap = T.capability(two_arm_cov(1.2, 1.4), flat_pattern())
    lp = T.plan_line(np.array([[0.2, 2.5], [1.0, 2.5]]), cap)
    assert lp.n_pieces == 0
    assert lp.assign == [T.UNCOVERED]
    d = T.summarise(T.Plan(cap, [lp]))
    assert d["uncovered_m"] == pytest.approx(0.8) and d["drawn_m"] == 0.0
    assert d["covered_frac"] == pytest.approx(0.0)


def test_a_degenerate_line_does_not_explode():
    cap = T.capability(two_arm_cov(1.2, 1.4), flat_pattern())
    lp = T.plan_line(np.array([[0.5, 0.5], [0.5, 0.5]]), cap)
    assert lp.length == pytest.approx(0.0)
    assert lp.n_pieces == 1 and lp.pieces[0].length == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 9.  THE SYNTHETIC PICTURES ARE ON THE PAPER
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ["strokes", "hatch", "scribble"])
def test_the_synthetic_sets_land_inside_the_certified_block(kind):
    lines = T.synthetic(kind, 50, seed=1)
    assert len(lines) >= 40
    P = np.vstack(lines)
    assert P[:, 0].min() >= T.BLOCK[0] - 1e-9
    assert P[:, 0].max() <= T.BLOCK[2] + 1e-9
    assert P[:, 1].min() >= T.BLOCK[1] - 1e-9
    assert P[:, 1].max() <= T.BLOCK[3] + 1e-9
    assert T.synthetic(kind, 50, seed=1)[0].tolist() == lines[0].tolist()


# ---------------------------------------------------------------------------
# 9.  THE WORK CELL, ON THE ALLOCATOR'S OWN PREFILTER (build item 3a)
# ---------------------------------------------------------------------------
def test_a_work_cell_normalises_from_every_shape_it_arrives_in():
    from aris_sixarm import allocate
    pat = T.zigzag_pattern()
    assert allocate.work_cell_regions(None) is None
    # a Pattern, with the stage named
    got = allocate.work_cell_regions(pat, 0)
    assert sorted(got) == [2, 13, 71]
    assert got[13] == (T.row_band(0),)
    # an iterable of StageCell, filtered to one stage
    assert allocate.work_cell_regions(pat.cells, 1) == \
        allocate.work_cell_regions(pat, 1)
    # a plain mapping, one rect or several
    r = (0.2, 0.3, 0.4, 0.5)
    assert allocate.work_cell_regions({13: r}) == {13: (r,)}
    assert allocate.work_cell_regions({13: (r, r)}) == {13: (r, r)}
    # a stage names its actives; everybody else may draw NOWHERE
    assert 17 not in allocate.work_cell_regions(pat, 0)


def test_a_point_is_in_exactly_one_of_two_abutting_work_cells():
    """Half-open at the high edge, exactly as `rect_contains` is."""
    from aris_sixarm import allocate
    lo, hi = (0.0, 0.0, 1.0, 1.0), (1.0, 0.0, 2.0, 1.0)
    xy = [(0.5, 0.5), (1.0, 0.5), (1.5, 0.5), (2.0, 0.5)]
    a = allocate.in_work_cell(xy, (lo,))
    b = allocate.in_work_cell(xy, (hi,))
    assert list(a) == [True, False, False, False]
    assert list(b) == [False, True, True, False]
    assert not (a & b).any()
    assert not allocate.in_work_cell(xy, ()).any()


def test_the_prefilter_returns_only_cells_inside_the_work_cell(tmp_path):
    """`atlas_cells` with a work cell is the atlas INTERSECTED with the region.

    Written against a synthetic two-arm atlas rather than a shipped one, so the
    property is pinned whether or not `out/` holds a current sweep.
    """
    import numpy as np
    from aris_sixarm import allocate, atlas as A
    g = 0.02
    xs, ys = np.meshgrid(np.arange(0.10, 0.90, g), np.arange(0.10, 0.90, g))
    n = xs.size
    row = np.zeros((n, max(A.FLATCOL, A.LEANCOL) + 1))
    row[:, 0], row[:, 1] = xs.ravel(), ys.ravel()
    row[:, 2] = 0.30
    row[:, A.FLATCOL] = 0.30
    row[:, A.LEANCOL] = -1.0
    for aid in (13, 17):
        np.savez(tmp_path / f"atlas_arm{aid}.npz", data=row, grid=g,
                 h=0.97, pen=0.11)
    full = allocate.atlas_cells((13, 17), str(tmp_path))
    assert full is not None and len(full[13][1]) == n
    box = (0.20, 0.20, 0.50, 0.50)
    cut = allocate.atlas_cells((13, 17), str(tmp_path),
                               work_cells={13: box, 17: (0.50, 0.20, 0.80, 0.50)})
    assert 0 < len(cut[13][1]) < len(full[13][1])
    for ix, iy in cut[13][1]:
        x, y = ix * g, iy * g
        assert box[0] - 1e-9 <= x < box[2] - 1e-9
        assert box[1] - 1e-9 <= y < box[3] - 1e-9
    assert not (cut[13][1] & cut[17][1])        # abutting cells never overlap
    # an arm the work cell does not name draws nowhere
    only13 = allocate.atlas_cells((13, 17), str(tmp_path), work_cells={13: box})
    assert only13[17][1] == set()
    # ...and a stage of the real pattern restricts every arm it names
    staged = allocate.atlas_cells((13, 17), str(tmp_path),
                                  work_cells=T.zigzag_pattern(), stage=0)
    assert staged[17][1] == set() and len(staged[13][1]) <= len(full[13][1])


def test_the_park_erosion_gives_back_the_side_the_partner_stands_on():
    """`park_erode` is the x-erosion frontier's knob, and it is off by default.

    An arm in column 0 gives back the HIGH-x end of its cell (its transverse
    partner stands at x = 1.2067) and an arm in column 1 the LOW-x end, so the
    two columns' eroded cells still tile the row band until the erosion passes
    half its width.  Measured 2026-09-11: the frontier is 1.04 m of a 1.48 m
    block, which is why this is a parameter and not the default.
    """
    r = (0.16, 0.0, 1.64, 1.0)
    assert T.park_erode(r, 13, 0.0) == r            # off by default
    assert T.park_erode(r, 13, 0.40) == pytest.approx((0.16, 0.0, 1.24, 1.0))
    assert T.park_erode(r, 17, 0.40) == pytest.approx((0.56, 0.0, 1.64, 1.0))
    for a in T.ARMS:                                 # never inverted
        x0, _, x1, _ = T.park_erode(r, a, 2.0)
        assert x0 <= x1
    # the shipped pattern is unchanged, name included
    assert T.zigzag_pattern().name == "zigzag-rowband-y20+6seams"
    assert T.zigzag_pattern(park_erode_m=0.0) == T.zigzag_pattern()
    eroded = T.zigzag_pattern(park_erode_m=0.40)
    assert eroded.name == "zigzag-rowband-y20+6seams-parkx040"
    assert eroded.n_stages == T.zigzag_pattern().n_stages
    for c, e in zip(T.zigzag_pattern().cells, eroded.cells):
        assert c.stage == e.stage and c.arm == e.arm and c.name == e.name
        w0 = c.region[0][2] - c.region[0][0]
        assert e.region[0][2] - e.region[0][0] == pytest.approx(w0 - 0.40)
    # ...and the two columns still tile the row band up to half its width
    lo = T.park_erode(T.row_band(0), 13, 0.74)
    hi = T.park_erode(T.row_band(0), 17, 0.74)
    assert lo[2] == pytest.approx(hi[0])


# ---------------------------------------------------------------------------
# PETE'S LEADER/FOLLOWER PATTERN
# ---------------------------------------------------------------------------
def test_the_main_stages_tile_the_row_bands_and_the_seams_go_to_the_conductor():
    """Nothing is offered twice and nothing is offered to nobody.

    The two main stages between them hand every point of every row band to
    exactly one (arm, role) -- the arm whose column half it is in, leading or
    following -- and the y dead bands, which no row band covers, go to the
    conducted final pass.  A point offered twice is a line drawn twice; a point
    offered never is a hole.
    """
    pat = T.leader_follower_pattern()
    main = [c for c in pat.cells if c.stage < 2]
    seam = [c for c in pat.cells if c.stage == 2]
    xs = np.linspace(T.BLOCK[0] + 0.01, T.BLOCK[2] - 0.01, 40)
    ys = np.linspace(T.BLOCK[1] + 0.01, T.BLOCK[3] - 0.01, 60)
    X, Y = np.meshgrid(xs, ys)
    n_main = sum(c.contains(X, Y).astype(int) for c in main)
    n_seam = sum(c.contains(X, Y).astype(int) for c in seam)
    in_band = np.zeros_like(X, bool)
    for j in (0, 1, 2):
        in_band |= T.rect_contains((T.row_band(j),), X, Y)
    assert (n_main[in_band] == 1).all(), "a row-band point is offered once"
    assert (n_main[~in_band] == 0).all(), "the dead band is not a row band"
    assert (n_seam[~in_band] >= 1).all(), "every dead-band point has a drawer"
    assert (n_seam[in_band] == 0).all(), "a seam cell never leaves the band"


def test_the_leader_owns_the_contested_middle_and_the_follower_the_outer_strip():
    """The rule, as geometry: the leader draws toward its partner.

    A follower's safe pieces are the ones FAR from the arm it is yielding to, so
    an arm draws its half of the contested middle -- the paper between the two
    base columns, which only a priority planner can be trusted with -- as
    leader, and the strip between its own base column and the rim as follower.
    """
    for a in T.ARMS:
        lead = T.role_region(a, "leader")[0]
        foll = T.role_region(a, "follower")[0]
        base = T.COL_X[T.COL_OF[a]]
        # the boundary is the arm's own base column when the split is zero
        if T.COL_OF[a] == 0:
            assert lead[0] == pytest.approx(base) and foll[2] == pytest.approx(base)
            assert lead[2] == pytest.approx(T.X_MID)     # in to the mid-line
            assert foll[0] == pytest.approx(T.BLOCK[0])  # out to the rim
        else:
            assert lead[2] == pytest.approx(base) and foll[0] == pytest.approx(base)
            assert lead[0] == pytest.approx(T.X_MID)
            assert foll[2] == pytest.approx(T.BLOCK[2])
    # the leaders are the 1-2-1 -- one arm per row, columns alternating -- which
    # is the zigzag's own stage 0, so the LEADERS are separated exactly as before
    assert T.LEADERS == (13, 71, 2)
    assert [T.ROW_OF[a] for a in T.LEADERS] == [0, 1, 2]
    assert [T.COL_OF[a] for a in T.LEADERS] == [0, 1, 0]
    assert sorted(T.LEADERS + T.FOLLOWERS) == sorted(T.ARMS)


def test_the_pattern_name_records_the_split_it_was_built_with():
    assert T.leader_follower_pattern().name == "leader-follower-y20-split+000"
    assert T.leader_follower_pattern(split_m=0.15).name == \
        "leader-follower-y20-split+150"
    assert T.leader_follower_pattern(whole_bag=True).name == \
        "leader-follower-y20-whole"
    assert T.leader_follower_pattern(split_m=0.0) == T.leader_follower_pattern()
