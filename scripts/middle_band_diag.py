#!/usr/bin/env python3
"""Is the new cross-seam overlap CONCURRENCY, or only redundancy?

    python3 scripts/middle_band_diag.py            # -> out/middle_band_diag.json

The six-arm atlas on the merged canvas says the seam strip is strict-GO for
77.2 % of its cells by two or more arms, and 74.9 % of it by an arm from EACH
unit (docs/MERGED_CANVAS.md).  That is a statement about REACH: either arm can
stand there.  It is not a statement about the two of them being there at the
same time, and the conductor has opinions about the difference — the first
six-arm CSAIL run was refused with

    arm 97 has no monotone pause schedule ...; the impossible indices are INK:
    arm 2 segment(s) [3] — no order fixes that, only a different allocation

which says arm 2 drawing one stroke near the seam leaves arm 97 nowhere at all
to be.  This script measures that directly, off the atlas's own certified
poses: for every pair of cells (one each arm) in a band, it computes the
capsule clearance between the two poses the atlas stored for them.  No planner,
no schedule — the pure geometric question "can these two arms hold these two
poses at once".

Reported per arm pair:
  shared        cells both arms are strict-GO on          (the redundancy)
  pairs_ok      (i, j) pose pairs at or above the margin  (the concurrency)
  worst / best  the clearance range over those pairs
  blocked_i     cells of arm A whose pose clears NO pose of arm B anywhere in
                the band — the shape the refusal above takes
"""
import argparse
import itertools
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
_RIG = "final6_opt"
for _i, _a in enumerate(sys.argv):
    if _a == "--rig" and _i + 1 < len(sys.argv):
        _RIG = sys.argv[_i + 1]
os.environ["ARIS_RIG"] = _RIG
sys.path.insert(0, str(ROOT))

import numpy as np                                            # noqa: E402
from aris_sixarm import coordination, rig_final6 as r6        # noqa: E402
from aris_sixarm.atlas import QCOL, load, strict_go           # noqa: E402
from aris_sixarm.fleet import ACTIVE_RIG, FLEET               # noqa: E402


def poses_in(arr, y0, y1, stride):
    """Strict-GO rows with y in [y0, y1] -> (xy (N,2), q (N,7))."""
    g = strict_go(arr)
    m = g & (arr[:, 1] >= y0 - 1e-9) & (arr[:, 1] <= y1 + 1e-9)
    sel = arr[m][::stride]
    return sel[:, :2], sel[:, QCOL:QCOL + 7]


def pair_clearance(a, qa, b, qb, pen=0.110):
    """(Na,7) x (Nb,7) -> (Na, Nb) capsule clearance, metres."""
    pa = coordination.ArmPath(a, np.ascontiguousarray(qa), 1.0, 1.0, pen,
                              FLEET[a])
    pb = coordination.ArmPath(b, np.ascontiguousarray(qb), 1.0, 1.0, pen,
                              FLEET[b])
    return coordination.clearance_matrix(pa, pb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rig", default=_RIG)
    ap.add_argument("--atlas", default=None)
    ap.add_argument("--out", default=str(ROOT / "out/middle_band_diag.json"))
    ap.add_argument("--stride", type=int, default=3,
                    help="subsample of the band's cells per arm")
    ap.add_argument("--pad", type=float, default=0.40,
                    help="metres either side of the seam that count as the "
                         "middle band")
    ap.add_argument("--band", action="append", metavar="Y0,Y1",
                    help="an extra band to measure, e.g. '0.60,1.50' for one "
                         "unit's own working half; repeatable")
    ap.add_argument("--margin", type=float, default=None,
                    help="required clearance (default safety + calib)")
    a = ap.parse_args()
    atlas = Path(a.atlas or (ROOT / "out" / f"atlas_{ACTIVE_RIG}"))
    margin = a.margin if a.margin is not None \
        else coordination.SAFETY_M + coordination.CALIB_M

    y0s, y1s = r6.WEB_A[1][1], r6.WEB_B[0][1]
    bands = [("SEAM", y0s, y1s),
             ("middle band", y0s - a.pad, y1s + a.pad)]
    for spec in (a.band or []):
        lo, hi = (float(v) for v in spec.split(","))
        bands.append((f"band y {lo:.2f}-{hi:.2f}", lo, hi))
    arms = sorted(FLEET)
    arr = {x: load(atlas, x)[0] for x in arms}

    doc = dict(rig=ACTIVE_RIG, atlas=str(atlas), margin=margin,
               stride=a.stride, bands={})
    for name, y0, y1 in bands:
        print(f"\n=== {name}: canvas y {y0:.3f} .. {y1:.3f} m, "
              f"clearance margin {1000 * margin:.0f} mm ===")
        P = {x: poses_in(arr[x], y0, y1, a.stride) for x in arms}
        rows = []
        for x, y in itertools.combinations(arms, 2):
            (xya, qa), (xyb, qb) = P[x], P[y]
            if not len(qa) or not len(qb):
                continue
            # cells BOTH arms can stand on: the redundancy the atlas reports
            sa = {tuple(np.round(p, 4)) for p in xya}
            sb = {tuple(np.round(p, 4)) for p in xyb}
            shared = len(sa & sb)
            D = pair_clearance(x, qa, y, qb)
            ok = D >= margin
            blocked_a = int((~ok).all(axis=1).sum())
            blocked_b = int((~ok).all(axis=0).sum())
            r = dict(a=int(x), b=int(y), n_a=int(len(qa)), n_b=int(len(qb)),
                     shared_cells=shared,
                     pairs=int(D.size), pairs_ok=int(ok.sum()),
                     pairs_ok_pct=round(100 * float(ok.mean()), 2),
                     best_mm=round(1000 * float(D.max()), 1),
                     worst_mm=round(1000 * float(D.min()), 1),
                     median_mm=round(1000 * float(np.median(D)), 1),
                     blocked_a=blocked_a, blocked_b=blocked_b,
                     same_unit=r6.UNIT_OF[x] == r6.UNIT_OF[y])
            rows.append(r)
            print(f"  {x:>2} vs {y:>2} ({'same' if r['same_unit'] else 'CROSS'} "
                  f"unit): {r['n_a']:>4} x {r['n_b']:<4} poses, "
                  f"{shared:>4} shared cells, "
                  f"{r['pairs_ok_pct']:>6.2f} % of pose pairs clear "
                  f"(worst {r['worst_mm']:>7.1f} mm, median "
                  f"{r['median_mm']:>6.1f}), "
                  f"{blocked_a} of arm {x}'s poses clear none of arm {y}'s")
        doc["bands"][name] = dict(y0=y0, y1=y1, pairs=rows)

    Path(a.out).write_text(json.dumps(doc, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
