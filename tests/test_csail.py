"""Tracer + allocator regressions (fast: no IK, no image files).

Run: python3 tests/test_csail.py     (or pytest tests/)
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from aris_sixarm import allocate, trace              # noqa: E402
from aris_sixarm.allocate import Interval            # noqa: E402
from aris_sixarm.fleet import SHEET                  # noqa: E402


def _bar(mask, r0, r1, c0, c1):
    mask[r0:r1, c0:c1] = True
    return mask


def _dense(p, ds=0.5):
    p = np.asarray(p, float)
    t = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    s = np.arange(0, t[-1], ds)
    return np.column_stack([np.interp(s, t, p[:, 0]), np.interp(s, t, p[:, 1])])


def _plus(n=81, w=5):
    m = np.zeros((n, n), bool)
    mid = n // 2
    _bar(m, mid - w // 2, mid + w // 2 + 1, 6, n - 6)
    _bar(m, 6, n - 6, mid - w // 2, mid + w // 2 + 1)
    return m


def _ex(n=81, w=5):
    m = np.zeros((n, n), bool)
    for i in range(6, n - 6):
        for d in range(-(w // 2), w // 2 + 1):
            m[i, min(max(i + d, 0), n - 1)] = True
            m[i, min(max(n - 1 - i + d, 0), n - 1)] = True
    return m


def test_crossing_stays_two_strokes():
    """The whole point of the junction pairing: an X leaves as 2 strokes, not 4.

    A skeleton graph cut at every junction would return four ~35 px stubs from
    each of these; the straightest-continuation pairing has to carry both
    branches through the crossing.
    """
    for name, m in (("plus", _plus()), ("ex", _ex())):
        strokes, skel = trace.trace_lines(m, bridge=False)
        assert len(strokes) == 2, f"{name}: {len(strokes)} strokes, want 2"
        for s in strokes:
            assert trace.plen(s) > 0.8 * (m.shape[0] - 12), \
                f"{name}: stroke only {trace.plen(s):.0f} px long"
        # and the two strokes really do cross (RDP leaves the crossing point
        # implicit — it is collinear — so compare the densified paths)
        a, b = (_dense(s) for s in strokes)
        d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
        assert d.min() < 2.0, f"{name}: the two strokes never meet ({d.min():.1f} px)"


def test_thinning_and_thick_split():
    """Zhang-Suen returns a 1 px skeleton; solid glyphs split off from line art.

    `split_thick` judges whole CONNECTED COMPONENTS, which is what the logo
    needs (the wordmark does not touch the buildings) and is why the two
    shapes here are disjoint.
    """
    m = np.zeros((60, 130), bool)
    _bar(m, 28, 33, 5, 60)               # a 5 px line
    _bar(m, 5, 40, 80, 115)              # a 35 px solid glyph, not touching it
    sk = trace.thin(m)
    assert sk.sum() < 0.2 * m.sum(), "skeleton is not thin"
    assert sk[28:33, 20:40].sum() == 20, "the line should thin to one row"
    assert trace.label_components(m)[1] == 2
    thick, thin_ = trace.split_thick(m, n_erode=4)
    assert thick[20, 90] and not thick[30, 10], "the glyph is not the thick part"
    assert thin_[30, 10] and not thin_[20, 90], "the line is not the thin part"
    assert thick.sum() + thin_.sum() == m.sum(), "the split must partition"


def test_rdp_and_sheet_mapping():
    """RDP keeps corners and drops collinear points; the sheet map keeps aspect."""
    line = np.column_stack([np.arange(0, 50, 1.0), np.zeros(50)])
    line[25:, 1] = np.arange(25) * 1.0
    assert len(trace.rdp(line, 1.2)) == 3
    px = [dict(pts=np.array([[0.0, 0.0], [200.0, 0.0], [200.0, 100.0],
                             [0.0, 100.0], [0.0, 0.0]]), color="grey",
               kind="outline")]
    out, info = trace.to_sheet(px, SHEET, margin=0.06)
    p = out[0]["pts"]
    assert abs((p[:, 0].max() - p[:, 0].min()) /
               (p[:, 1].max() - p[:, 1].min()) - 2.0) < 1e-9   # aspect kept
    assert p[:, 0].min() > 0.059 and p[:, 0].max() < SHEET[0] - 0.059
    assert p[:, 1].min() > 0.059 and p[:, 1].max() < SHEET[1] - 0.059
    assert abs(p[:, 0].mean() * 0 + (p[:, 0].max() + p[:, 0].min()) / 2
               - SHEET[0] / 2) < 1e-9                          # centred


def test_greedy_cover_is_minimal_and_finds_gaps():
    """Fewest pieces, and everything nobody covers comes back as a gap."""
    ivs = [Interval(0.0, 0.4, 13), Interval(0.3, 0.5, 17), Interval(0.35, 0.8, 31),
           Interval(0.9, 1.0, 97)]
    chosen, gaps = allocate.greedy_cover(ivs)
    assert [(v.s0, v.s1) for v in chosen] == [(0.0, 0.4), (0.35, 0.8), (0.9, 1.0)]
    assert gaps == [(0.8, 0.9)]
    # a single interval that spans everything must win outright
    chosen, gaps = allocate.greedy_cover(ivs + [Interval(0.0, 1.0, 97)])
    assert len(chosen) == 1 and not gaps
    # ties go to the arm with less work so far
    tie = [Interval(0.0, 1.0, 13), Interval(0.0, 1.0, 31)]
    assert allocate.greedy_cover(tie, load={13: 5.0, 31: 1.0})[0][0].arm == 31
    assert allocate.uncovered([Interval(0.2, 0.6, 13)]) == [(0.0, 0.2), (0.6, 1.0)]


def test_handoff_cut_lands_mid_overlap():
    """Seams go to the middle of the overlap, with both sides grown to meet."""
    chosen = [Interval(0.0, 0.6, 13), Interval(0.4, 1.0, 31)]
    spans = allocate.place_cuts(chosen, L=1.0, overlap=0.01)
    assert abs(spans[0]["s1"] - 0.51) < 1e-9      # mid-overlap 0.5, +1 cm
    assert abs(spans[1]["s0"] - 0.49) < 1e-9      # mid-overlap 0.5, -1 cm
    assert spans[0]["s1"] > spans[1]["s0"], "the two segments must overlap"
    for sp, iv in zip(spans, chosen):             # never outside what was certified
        assert iv.s0 - 1e-12 <= sp["s0"] and sp["s1"] <= iv.s1 + 1e-12


def test_colour_partitions():
    """14 non-trivial pen assignments for four arms, 16 with the degenerate two."""
    arms = [13, 17, 31, 97]
    p = allocate.partitions(arms)
    assert len(p) == 14 and len(allocate.partitions(arms, nontrivial=False)) == 16
    assert all(set(d) == set(arms) for d in p)
    assert all(len(set(d.values())) == 2 for d in p)
    assert len({tuple(sorted(d.items())) for d in p}) == 14      # all distinct
    assert len(allocate.partitions([13, 17])) == 2


def test_best_partition_follows_the_drops():
    """The enumeration must pick the pen split that leaves least paper empty."""
    strokes = [dict(id=0, color="grey", kind="outline",
                    pts=np.array([[0.0, 0.0], [1.0, 0.0]])),
               dict(id=1, color="orange", kind="outline",
                    pts=np.array([[0.0, 1.0], [3.0, 1.0]]))]
    ivmap = {0: [Interval(0.0, 1.0, 13)], 1: [Interval(0.0, 1.0, 31)]}
    colors, cover, table = allocate.best_partition(strokes, ivmap,
                                                   [13, 17, 31, 97])
    assert colors[13] == "grey" and colors[31] == "orange"
    assert cover["dropped_len"] < 1e-9
    assert table[0]["dropped"] <= table[-1]["dropped"]


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
    sys.exit(1 if fails else 0)
