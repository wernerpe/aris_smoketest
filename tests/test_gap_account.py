"""THE COVERAGE ACCOUNT: what is on the paper, measured on the paper.

Pete, 2026-09-15, looking at the certified programme's animation: "why are there
so many gaps in the lines?"  The answer could not be read off the programme,
because the programme's own book-keeping says the ink is there.  A piece that is
LISTED in an arm's stage is not ink: `staged._merge_conducts` pads an arm whose
bucket produced no timeline out to the stage's frame count, so the arm stands in
the conducted clock with eight pieces on its list, an `ink_m` of 1.914 m, and
not one pen-down sample anywhere -- which is exactly what `lf6b_s150` shipped
for arm 31 in stage C.

So the account is GEOMETRIC.  Drawn means a frame whose `seg` is a real segment
index, and the residual is the logo minus every drawn piece's own polyline.
These are the two claims that makes, on a toy picture whose hole is known.
"""
import numpy as np
import pytest

from scripts import gap_account as G


def _line(x0, x1, y, n=9):
    return np.stack([np.linspace(x0, x1, n), np.full(n, float(y))], axis=1)


def _doc(pieces):
    """A programme with one stage and one arm. `pieces` are (pts, flown)."""
    n = len(pieces)
    seg = [i for i, (_, ok) in enumerate(pieces) if ok] or [-1]
    return dict(stages=[dict(stage=0, arms={"31": dict(
        trajectory=dict(t=list(range(len(seg))), q=[], seg=seg),
        pieces=[dict(line=0 if i < 1 else 1, piece=i, order=i, length_m=1.0,
                     pts=np.asarray(p, float).tolist())
                for i, (p, _) in enumerate(pieces)],
        refused=[])})])


def test_a_piece_with_no_pen_down_samples_is_not_ink():
    """THE HOLE THE PROGRAMME'S OWN BOOK-KEEPING CANNOT SEE.

    Both pieces are listed with a length and both are in the arm's `pieces`
    list; only the first has a frame that draws it.  The second's metre is a
    hole, and `ink_m` would have counted it.
    """
    lines = [_line(0.0, 1.0, 0.0), _line(0.0, 1.0, 1.0)]
    doc = _doc([(lines[0], True), (lines[1], False)])
    ink, listed = G.drawn_pieces(doc)
    assert sorted(ink) == [0]                       # line 1 has no ink at all
    assert [p["flown"] for p in listed] == [True, False]
    gaps, total, inked = G.residual(lines, ink)
    assert total == pytest.approx(2.0, abs=1e-6)
    assert inked == pytest.approx(1.0, abs=2 * G.DS)
    assert [g[0] for g in gaps] == [1]
    assert gaps[0][3] == pytest.approx(1.0, abs=2 * G.DS)


def test_the_residual_finds_a_hole_in_the_middle_of_a_line():
    """A join that does not close is a gap, and it is found where it is."""
    line = _line(0.0, 1.0, 0.0, 101)
    head = line[line[:, 0] <= 0.40 + 1e-9]
    tail = line[line[:, 0] >= 0.55 - 1e-9]
    doc = dict(stages=[dict(stage=0, arms={"31": dict(
        trajectory=dict(t=[0, 1], q=[], seg=[0, 1]),
        pieces=[dict(line=0, piece=0, order=0, length_m=0.40,
                     pts=head.tolist()),
                dict(line=0, piece=1, order=1, length_m=0.45,
                     pts=tail.tolist())], refused=[])})])
    ink, _ = G.drawn_pieces(doc)
    gaps, total, inked = G.residual([line], ink)
    assert total == pytest.approx(1.0, abs=1e-6)
    assert len(gaps) == 1
    li, s0, s1, m = gaps[0]
    assert li == 0
    assert m == pytest.approx(0.15, abs=2 * (G.DS + G.TOL))
    assert s0 == pytest.approx(0.40, abs=G.DS + G.TOL)
    assert s1 == pytest.approx(0.55, abs=G.DS + G.TOL)
    assert inked == pytest.approx(0.85, abs=4 * G.DS)


def test_a_fully_drawn_picture_leaves_no_residual():
    """The control: the same machinery must report ZERO on a complete cover."""
    lines = [_line(0.0, 1.0, 0.0), _line(0.0, 0.5, 1.0)]
    doc = _doc([(lines[0], True), (lines[1], True)])
    ink, _ = G.drawn_pieces(doc)
    gaps, total, inked = G.residual(lines, ink)
    assert gaps == []
    assert inked == pytest.approx(total, abs=1e-9)


def test_ink_is_matched_PER_LINE_and_never_across_neighbours():
    """TWO LINES 1 mm APART ARE TWO LINES.

    The residual is taken per line against the pieces that name that line, so
    ink lying beside a line can never be read as covering it -- which a purely
    geometric nearest-point test over the whole picture would do.
    """
    lines = [_line(0.0, 1.0, 0.0), _line(0.0, 1.0, 0.0005)]
    doc = _doc([(lines[0], True), (lines[1], False)])
    ink, _ = G.drawn_pieces(doc)
    gaps, total, inked = G.residual(lines, ink)
    assert [g[0] for g in gaps] == [1]
    assert inked == pytest.approx(1.0, abs=2 * G.DS)
