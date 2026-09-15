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


# ---------------------------------------------------------------------------
# THE INVARIANT `staged.run` ASSERTS: EVERY STRETCH IS FLOWN OR IN THE GAP LIST
# ---------------------------------------------------------------------------
from aris_sixarm import staged, traces          # noqa: E402


def _arm_stage(stage, arm, segs, flown):
    """An `ArmStage` whose `programme` is `segs` and which draws `flown`."""
    st = staged.ArmStage(int(stage), int(arm), np.zeros(7))
    st.programme = [dict(stroke_id=int(li), seg=i, piece=int(k),
                         length=float(traces.cumlen(p)[-1]),
                         pts=np.asarray(p, float))
                    for i, (li, k, p) in enumerate(segs)]
    n = max(2, len(segs) + 1)
    st.timeline = dict(t=0.05 * np.arange(n), q=np.zeros((n, 7)),
                       seg=np.array([(i if i in flown else -1)
                                     for i in range(n)], int),
                       u=np.zeros(n), phases=[], duration=0.05 * (n - 1),
                       draw_s=0.0, transit_s=0.0)
    return st


def _plan_with_uncovered(lines, uncovered):
    """A `traces.Plan` in which the named lines have NO drawer at all."""
    lp = []
    for i, l in enumerate(lines):
        cum = traces.cumlen(l)
        atoms = [traces.Atom(0.0, float(cum[-1]), 0)]
        assign = [traces.UNCOVERED if i in uncovered else 0]
        lp.append(traces.LinePlan(i, np.asarray(l, float), cum, atoms, assign,
                                  [], []))
    return traces.Plan(None, lp)


def test_a_line_no_stage_can_take_is_in_the_gap_list_with_a_reason():
    """THE DELIBERATE HOLE MUST NOT VANISH.

    Line 1 is offered to nobody -- the DP marks it `UNCOVERED` -- so no stage
    ever lists it, no arm ever refuses it and no book-keeping anywhere would
    mention it.  The account is over the INPUT PICTURE, so it is a gap with a
    reason, and `flown + missing` is still the whole picture.
    """
    lines = [_line(0.0, 1.0, 0.0), _line(0.0, 1.0, 1.0)]
    sr = staged.StageResult(0, (31,),
                            {31: _arm_stage(0, 31, [(0, 0, lines[0])], {0})})
    res = staged.StagedResult("toy", [sr], [])
    acc = staged.coverage_account(lines, res, _plan_with_uncovered(lines, {1}))
    assert acc["total_m"] == pytest.approx(2.0, abs=1e-6)
    assert acc["flown_m"] == pytest.approx(1.0, abs=2 * staged.INK_DS)
    assert acc["gaps_m"] == pytest.approx(1.0, abs=2 * staged.INK_DS)
    assert [g["line"] for g in acc["gap_list"]] == [1]
    assert acc["gap_list"][0]["reason"] == "no_drawer"
    assert acc["unattributed_m"] == pytest.approx(0.0, abs=1e-9)
    # the invariant `run` asserts
    assert acc["flown_m"] + acc["gaps_m"] == pytest.approx(acc["total_m"],
                                                           abs=1e-9)


def test_a_listed_bucket_that_never_flew_is_a_gap_not_ink():
    """`lf6b_s150` STAGE C, IN MINIATURE.

    The arm's bucket is listed with a length and the arm has a trajectory --
    `_merge_conducts` pads it -- and not one frame draws it.  `ink_m` counts
    the metre; the account must not.
    """
    lines = [_line(0.0, 1.0, 0.0), _line(0.0, 1.0, 1.0)]
    st = _arm_stage(0, 31, [(0, 0, lines[0]), (1, 0, lines[1])], {0})
    sr = staged.StageResult(0, (31,), {31: st})
    res = staged.StagedResult("toy", [sr], [])
    acc = staged.coverage_account(lines, res, _plan_with_uncovered(lines, set()))
    assert acc["gaps_m"] == pytest.approx(1.0, abs=2 * staged.INK_DS)
    g = acc["gap_list"][0]
    assert g["line"] == 1 and g["reason"] == "listed_not_flown"
    assert g["listed_not_flown"][0]["arm"] == 31
    assert acc["listed_not_flown_m"] == pytest.approx(1.0, abs=1e-6)


def test_every_gap_carries_one_of_the_declared_reason_codes():
    """No gap is ever silent, and no reason is ever invented."""
    lines = [_line(0.0, 1.0, 0.0)]
    sr = staged.StageResult(0, (31,), {31: _arm_stage(0, 31, [], set())})
    res = staged.StagedResult("toy", [sr], [])
    acc = staged.coverage_account(lines, res, None)
    assert len(acc["gap_list"]) == 1
    assert acc["gap_list"][0]["reason"] in staged.GAP_REASONS
    assert acc["flown_m"] == pytest.approx(0.0, abs=1e-9)


def _dp_piece(stage, arm, line, k, pts):
    cum = traces.cumlen(np.asarray(pts, float))
    return staged.Piece(int(stage), int(arm), int(line), int(k),
                        np.asarray(pts, float), float(cum[-1]),
                        state=0, s0=0.0, s1=float(cum[-1]))


def test_a_bucket_no_stage_ever_opened_is_named_and_not_unattributed():
    """`bench/starburst`, IN MINIATURE -- and it is what `unattributed` WAS.

    The DP gives line 1 to (stage 2, arm 71) and the conducted group that owns
    arm 71 produces no arms at all, so the piece is listed by nobody, refused
    by nobody and deferred by nobody: every book in the account is silent about
    a metre that is plainly missing from the paper.  Measured on `starburst`
    2026-09-15, that was 1.594 m of `unattributed` -- a hole in the REASON
    chain, not in the picture -- and it is `bucket_never_planned`.
    """
    lines = [_line(0.0, 1.0, 0.0), _line(0.0, 1.0, 1.0)]
    sr = staged.StageResult(0, (31,),
                            {31: _arm_stage(0, 31, [(0, 0, lines[0])], {0})})
    pcs = [_dp_piece(0, 31, 0, 0, lines[0]), _dp_piece(2, 71, 1, 0, lines[1])]
    res = staged.StagedResult("toy", [sr], pcs)
    acc = staged.coverage_account(lines, res, _plan_with_uncovered(lines, set()))
    assert len(acc["gap_list"]) == 1
    g = acc["gap_list"][0]
    assert g["line"] == 1 and g["reason"] == "bucket_never_planned"
    assert g["never_planned"][0]["stage"] == 2
    assert g["never_planned"][0]["arm"] == 71
    assert acc["unattributed_m"] == pytest.approx(0.0, abs=1e-9)
    # ...and the piece the stage DID list is not accused of the same thing
    assert not any(u["line"] == 0 for v in acc["gap_list"]
                   for u in v["never_planned"])


def test_every_missing_metre_of_a_toy_picture_has_a_NAMED_reason():
    """THE ACCOUNT HAS NO UNEXPLAINED BUCKET.

    One picture with all five failures on it at once -- a line nobody was
    offered, a bucket listed and never flown, a stub `plan_stroke` refused, a
    piece deferred and never re-listed, and a bucket no stage ever opened --
    and every missing metre lands in a named reason.  `unattributed` is a BUG
    and the assertion is that it is empty, not that it is small.
    """
    lines = [_line(0.0, 1.0, 0.0),      # 0  drawn
             _line(0.0, 1.0, 1.0),      # 1  listed, never flown
             _line(0.0, 1.0, 2.0),      # 2  no drawer at all
             _line(0.0, 1.0, 3.0),      # 3  plan_stroke refused it
             _line(0.0, 1.0, 4.0),      # 4  deferred and never re-listed
             _line(0.0, 1.0, 5.0)]      # 5  bucket no stage ever opened
    st = _arm_stage(0, 31, [(0, 0, lines[0]), (1, 0, lines[1])], {0})
    st.planned = [staged.PiecePlan(_dp_piece(0, 31, 3, 0, lines[3]),
                                   "degenerate", "too_short")]
    st.deferred = [_dp_piece(0, 31, 4, 0, lines[4])]
    sr = staged.StageResult(0, (31,), {31: st})
    pcs = [_dp_piece(0, 31, i, 0, lines[i]) for i in (0, 1, 3, 4)]
    pcs.append(_dp_piece(2, 71, 5, 0, lines[5]))
    res = staged.StagedResult("toy", [sr], pcs)
    acc = staged.coverage_account(lines, res, _plan_with_uncovered(lines, {2}))
    assert acc["flown_m"] + acc["gaps_m"] == pytest.approx(acc["total_m"],
                                                           abs=1e-9)
    got = {g["line"]: g["reason"] for g in acc["gap_list"]}
    assert got == {1: "listed_not_flown", 2: "no_drawer", 3: "plan_refused",
                   4: "deferred_never_taken", 5: "bucket_never_planned"}
    assert acc["unattributed_m"] == pytest.approx(0.0, abs=1e-9)
    assert "unattributed" not in acc["by_reason"]
    assert sum(v["m"] for v in acc["by_reason"].values()) == \
        pytest.approx(acc["gaps_m"], abs=6 * staged.INK_DS)


def test_all_ok_is_false_over_ink_the_timeline_never_draws_without_a_picture():
    """THE GUARD MAY NOT DEPEND ON HAVING THE LINES.

    `staged.run_conducted` re-runs one stage off a programme and never sees the
    input polylines, so it has no coverage account -- and `lf7c_s150` shipped
    2.520 m listed and never drawn while reporting `all_ok: true`, because the
    guard read the account instead of the stages.
    """
    lines = [_line(0.0, 1.0, 0.0), _line(0.0, 1.0, 1.0)]
    st = _arm_stage(0, 31, [(0, 0, lines[0]), (1, 0, lines[1])], {0})
    res = staged.StagedResult("toy", [staged.StageResult(0, (31,), {31: st})], [])
    assert res.coverage == {}                    # no picture, no account
    assert staged.listed_not_flown_m(res) == pytest.approx(1.0, abs=1e-6)
    d = staged.summary(res)
    assert d["listed_not_flown_m"] == pytest.approx(1.0, abs=1e-4)
    assert d["all_ok"] is False
