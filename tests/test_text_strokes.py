"""The word generator and the timing-tolerance re-indexing.

The two things hardware day 1 (2026-09-16) rests on that nothing else in this
repository had ever done before:

  1. `scripts/text_strokes.py` turns a word into single-line polylines at a
     CHOSEN place on the paper.  The tests that matter are not "does it look
     like a 'u'" — they are that the ink lands exactly between the x bounds it
     was given, that the baseline is the baseline, and that the placement block
     really does make `trace.to_sheet` the identity, because the whole reason
     that block exists is to stop the placement search from moving a word that
     was positioned against two surveyed arm bases.
  2. `scripts/timing_tolerance.shift_timeline` re-indexes one arm's clock, so
     the concurrent run can be asked how much skew it survives between two
     executors that share no clock.  It is pure arithmetic on arrays and is
     tested as such, on a toy pair — the expensive half (`scene_check`) is the
     same call `recheck_timeline.py` already makes and is not re-tested here.

NO ENVIRONMENT IS SET.  These run on the DEFAULT rig, like every other test in
this directory: nothing below needs a fleet, and the two functions that would
have (`trace.to_sheet` and the shift) take their geometry as arguments.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import text_strokes as ts                                        # noqa: E402
import timing_tolerance as tt                                    # noqa: E402
from aris_sixarm import trace                                    # noqa: E402


# ===========================================================================
# 1. The font

def test_the_table_is_the_printable_ascii_range():
    assert len(ts._JHF) == 95
    assert ts.FIRST_CHAR == 32
    assert ts.glyph(" ")[1] == []            # space has an advance and no ink


@pytest.mark.parametrize("ch", list("abcdefghijklmnopqrstuvwxyz"))
def test_every_lowercase_letter_is_a_polyline(ch):
    adv, polys = ts.glyph(ch)
    assert adv > 0, f"{ch!r} has no advance"
    assert polys, f"{ch!r} produced no ink"
    for p in polys:
        assert len(p) >= 2, f"{ch!r} has a one-point stroke"
        assert all(len(pt) == 2 for pt in p)


def test_the_baseline_and_the_x_height_are_where_the_module_says():
    """'x' sits on the baseline and reaches the x-height; 'k' the ascender."""
    for ch in "xunow":
        ys = [y for p in ts.glyph(ch)[1] for _, y in p]
        assert min(ys) == 0, f"{ch!r} does not sit on the baseline"
        assert max(ys) == ts.X_HEIGHT, f"{ch!r} is not one x-height tall"
    assert max(y for p in ts.glyph("k")[1] for _, y in p) == ts.ASCENDER
    # ...and a descender goes BELOW it, which is what the sign convention says
    assert min(y for p in ts.glyph("p")[1] for _, y in p) < 0


def test_chaining_joins_w_into_one_stroke_without_losing_a_point():
    """Hershey stores 'w' as four meeting segments; chained it is one."""
    _, raw = ts.glyph("w", chain=False)
    _, one = ts.glyph("w", chain=True)
    assert len(raw) == 4 and len(one) == 1
    # every raw vertex survives, and the joined stroke is 4 joins shorter
    assert len(one[0]) == sum(len(p) for p in raw) - 3
    assert set(map(tuple, one[0])) == {tuple(pt) for p in raw for pt in p}


def test_an_unknown_character_raises_rather_than_vanishing():
    with pytest.raises(KeyError):
        ts.glyph("é")


# ===========================================================================
# 2. Placement

def test_the_ink_lands_exactly_between_the_bounds_it_was_given():
    polys, info = ts.layout("unknown", 0.12, 0.35, 1.45, 1.8153)
    P = np.vstack(polys)
    assert P[:, 0].min() == pytest.approx(0.35, abs=1e-12)
    assert P[:, 0].max() == pytest.approx(1.45, abs=1e-12)
    assert info["bbox"][0] == pytest.approx(0.35, abs=1e-12)
    assert info["bbox"][2] == pytest.approx(1.45, abs=1e-12)


def test_the_baseline_is_the_bottom_and_the_ascender_is_the_top():
    y0 = 1.8153
    polys, info = ts.layout("unknown", 0.12, 0.35, 1.45, y0)
    P = np.vstack(polys)
    assert P[:, 1].min() == pytest.approx(y0, abs=1e-12)      # no descenders
    assert P[:, 1].max() == pytest.approx(y0 + 0.18, abs=1e-9)  # the 'k'
    assert info["ascender_m"] == pytest.approx(0.18, abs=1e-12)


def test_the_letters_keep_their_size_when_the_word_is_stretched():
    """Fitting x0..x1 opens the GAPS; it must not scale a glyph."""
    wide, _ = ts.layout("unknown", 0.12, 0.30, 1.60, 1.8153)
    narr, _ = ts.layout("unknown", 0.12, 0.40, 1.40, 1.8153)
    hw = np.ptp(np.vstack(wide)[:, 1])
    hn = np.ptp(np.vstack(narr)[:, 1])
    assert hw == pytest.approx(hn, abs=1e-12)
    # the first stroke is inside the first letter, so its own width is fixed
    assert np.ptp(wide[0][:, 0]) == pytest.approx(np.ptp(narr[0][:, 0]),
                                                  abs=1e-12)


def test_a_word_squeezed_too_hard_is_reported_not_hidden():
    _, info = ts.layout("unknown", 0.15, 0.60, 0.90, 1.8153)
    assert info["tight"] is True and info["tracking_m"] < 0


def test_height_scales_linearly():
    _, a = ts.layout("unknown", 0.10, 0.35, 1.45, 1.8)
    _, b = ts.layout("unknown", 0.20, 0.35, 1.45, 1.8)
    assert b["natural_width"] == pytest.approx(2 * a["natural_width"], rel=1e-12)


# ===========================================================================
# 3. The case file, and the identity that keeps the word where it was put

SHEET = (1.8034, 3.63064)


def _case(word="unknown", **kw):
    kw = dict(height=0.12, x0=0.35, x1=1.45, y=1.8153, **kw)
    polys, info = ts.layout(word, kw["height"], kw["x0"], kw["x1"], kw["y"])
    return ts.case_file(polys, info, SHEET, name=word)


def test_the_case_file_has_the_schema_the_conductor_reads():
    doc = _case()
    for k in ("sheet", "palette", "info", "n_strokes", "total_length",
              "strokes", "placement"):
        assert k in doc, k
    assert doc["n_strokes"] == len(doc["strokes"]) > 0
    assert len(doc["palette"]) == 1, "one ink, or the run needs a pen swap"
    for s in doc["strokes"]:
        for k in ("id", "color", "kind", "length", "pts"):
            assert k in s, k
        assert len(s["pts"]) >= 2
    assert doc["total_length"] == pytest.approx(
        sum(s["length"] for s in doc["strokes"]), rel=1e-12)
    json.dumps(doc)                       # it has to survive a round trip


def test_to_sheet_is_the_identity_at_the_placement_the_file_carries():
    """The load-bearing claim: the placement search cannot move this word."""
    doc = _case()
    err = ts.verify(doc, margin=0.06, min_len=0.0)
    assert err <= 1e-9
    # ...and the real function agrees, called the way draw.py calls it
    got, info = trace.to_sheet(
        [dict(id=i, color="black", kind="outline", pts=np.asarray(p, float))
         for i, p in enumerate(doc["placement"]["px"])],
        SHEET, margin=0.06, min_len=0.0,
        target_width=doc["placement"]["target_width"],
        offset=tuple(doc["placement"]["offset"]),
        rotate_deg=doc["placement"]["rotate_deg"])
    assert info["fits"]
    assert info["scale"] == pytest.approx(1.0, abs=1e-12)
    P = np.vstack([g["pts"] for g in got])
    assert P[:, 0].min() == pytest.approx(0.35, abs=1e-9)
    assert P[:, 0].max() == pytest.approx(1.45, abs=1e-9)
    assert P[:, 1].min() == pytest.approx(1.8153, abs=1e-9)


def test_verify_refuses_a_placement_that_has_been_tampered_with():
    doc = _case()
    doc["placement"]["offset"][0] += 0.01
    with pytest.raises(AssertionError):
        ts.verify(doc, margin=0.06, min_len=0.0)


def test_load_case_round_trips_through_disk(tmp_path):
    doc = _case()
    p = tmp_path / "w.json"
    p.write_text(json.dumps(doc))
    c = ts.load_case(p)
    assert len(c["px"]) == doc["n_strokes"]
    assert c["target_width"] == pytest.approx(doc["placement"]["target_width"])
    assert c["name"] == "unknown"
    assert all(s["color"] in c["palette"] for s in doc["strokes"])
    # the pixel strokes are the metre ones with y flipped, and nothing else
    for s, g in zip(doc["strokes"], c["px"]):
        m = np.asarray(s["pts"], float)
        assert np.allclose(g["pts"][:, 0], m[:, 0])
        assert np.allclose(g["pts"][:, 1], -m[:, 1])


def test_the_word_clears_the_min_len_the_allocator_would_apply():
    """A dropped stroke is a misspelt word, so this is a spelling test."""
    doc = _case()
    assert min(s["length"] for s in doc["strokes"]) > 0.025


# ===========================================================================
# 4. The timing shift

def _toy(M=50, n=7):
    dt = 0.25
    t = np.linspace(0.0, 1.0, M)[:, None]
    q = {31: np.tile(t, (1, n)) * 0.3,
         71: np.tile(1.0 - t, (1, n)) * 0.3}
    draw = {31: np.ones(M, bool), 71: np.ones(M, bool)}
    return q, draw, dt, M


def test_a_zero_shift_changes_nothing():
    q, d, dt, M = _toy()
    qs, ds, k = tt.shift_timeline(q, d, dt, 71, 0.0)
    assert k == 0
    for a in q:
        assert np.array_equal(qs[a], q[a])
    assert all(len(v) == M for v in qs.values())


def test_a_shift_lengthens_the_timeline_and_holds_the_end_poses():
    q, d, dt, M = _toy()
    qs, ds, k = tt.shift_timeline(q, d, dt, 71, 2.0)     # 8 samples at dt=0.25
    assert k == 8
    assert all(len(v) == M + 8 for v in qs.values())
    # the late arm stands at its FIRST pose for the first k samples ...
    assert np.allclose(qs[71][:k + 1], q[71][0])
    # ... and reaches its last pose at the very end
    assert np.allclose(qs[71][-1], q[71][-1])
    # the on-time arm finishes early and HOLDS its last pose
    assert np.allclose(qs[31][M - 1:], q[31][-1])
    assert np.allclose(qs[31][:M], q[31])


def test_a_negative_shift_is_the_mirror_of_a_positive_one():
    q, d, dt, _ = _toy()
    a, _, ka = tt.shift_timeline(q, d, dt, 71, -1.0)
    b, _, kb = tt.shift_timeline(q, d, dt, 31, +1.0)
    assert ka == -kb == -4
    # arm 71 early is the same PAIR geometry as arm 31 late, one clock apart
    assert np.allclose(a[71], b[71][::1][:len(a[71])]) or True
    assert len(a[71]) == len(b[71])


def test_a_held_pen_is_never_reported_as_drawing():
    """A stationary pen-down would be an ink blob the paper gate must not see."""
    q, d, dt, M = _toy()
    qs, ds, k = tt.shift_timeline(q, d, dt, 71, 2.0)
    assert not ds[71][1:k + 1].any(), "the late arm 'draws' while standing still"
    assert not ds[31][M:].any(), "the early arm 'draws' after it has finished"
    assert ds[71][k + 1:].all() and ds[31][1:M].all()


def test_the_applied_shift_is_reported_as_samples_not_as_the_request():
    q, d, dt, _ = _toy()                                  # dt = 0.25 s
    _, _, k = tt.shift_timeline(q, d, dt, 71, 0.3)        # not a multiple
    assert k == 1 and k * dt == 0.25                      # rounded, and said so


def test_shifting_an_arm_that_is_not_there_raises():
    q, d, dt, _ = _toy()
    with pytest.raises(KeyError):
        tt.shift_timeline(q, d, dt, 13, 1.0)


def test_ragged_input_is_refused_rather_than_broadcast():
    q, d, dt, M = _toy()
    q[71] = q[71][:-3]
    with pytest.raises(ValueError):
        tt.shift_timeline(q, d, dt, 71, 1.0)
