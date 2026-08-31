#!/usr/bin/env python3
"""THE INK NOBODY CONDUCTED, LOOKED UP IN THE SOLO MAP.

Two numbers have been quoted side by side all campaign and never joined up:
the feasibility map's "99.46 % of the canvas is drawable by SOME arm" and the
conducted logo's "93.83 % of the artwork got drawn".  They are not the same
question and the gap between them is not a contradiction, but nothing measured
WHICH cells the second one loses, so nothing could say whether the artwork's
residual is a reach limit (the map would call those cells dead too) or a
scheduling one (the map calls them alive and the conductor still cannot get an
arm there inside a phase).

This asks.  It regenerates the artwork the run traced — same image, same
placement, and it checks the total against the run's own `traced_m` so a
placement drift cannot pass silently — resamples every stroke at 2 mm, reads
the ink the schedule actually laid out of its npz `segpts_*`, and calls a
sample UNCOVERED when no laid point lies within 4 mm of it.  Then it looks
each uncovered sample up in the map: how many arms can draw that cell solo,
which ones, and how far it is from each base.

Run:  python3 scripts/ink_vs_map.py out/csail_schedule_h094_v11.npz \
          [--map out/feasible_workspace_v12] [--against OTHER.npz]

`--against` prints the SECOND run's uncovered set minus the first's, which is
how an A/B says where the ink one configuration loses and the other keeps
actually sits.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aris_sixarm import trace                                      # noqa: E402
from aris_sixarm.fleet import FLEET, SHEET, H_INV_DEFAULT          # noqa: E402

# The placement the campaign's logo runs were traced at
# (`out/csail_place_v4_placement.json`, "chosen").  A run whose npz was made at
# a different placement is caught by the traced-length check in `artwork`.
PLACEMENT = dict(image=str(ROOT / "assets/csail/csail_old_med.gif"),
                 target_width=1.4308899999999998, rotate_deg=90.0,
                 offset=(-0.1, 0.0), margin=0.06, min_len=0.025)
STEP = 0.002               # m between artwork samples
NEAR = 0.004               # m: a sample this close to laid ink is covered


def artwork(expect_m=None):
    """The strokes the run traced. -> ([stroke], total metres)."""
    px, _ = trace.trace_logo(PLACEMENT["image"])
    strokes, info = trace.to_sheet(
        px, SHEET, margin=PLACEMENT["margin"],
        target_width=PLACEMENT["target_width"], offset=PLACEMENT["offset"],
        rotate_deg=PLACEMENT["rotate_deg"], min_len=PLACEMENT["min_len"])
    total = float(trace.total_length(strokes))
    if expect_m is not None and abs(total - expect_m) > 1e-3:
        raise SystemExit(f"the artwork this script regenerates is {total:.4f} m "
                         f"and the run traced {expect_m:.4f} m — the placement "
                         f"in PLACEMENT is not the one that run used")
    return strokes, total


def resample(pts, step=STEP):
    """A polyline as points at most `step` apart. -> (N,2)."""
    P = np.asarray(pts, float).reshape(-1, 2)
    if len(P) < 2:
        return P
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] <= 0:
        return P[:1]
    t = np.linspace(0.0, s[-1], max(2, int(np.ceil(s[-1] / step)) + 1))
    return np.column_stack([np.interp(t, s, P[:, 0]), np.interp(t, s, P[:, 1])])


def drawn_points(npz_path):
    """Every canvas point a schedule inks. -> (M,2)."""
    z = np.load(npz_path)
    out = []
    for k in z.files:
        if k.startswith("segpts_"):
            P = np.asarray(z[k], float)
            P = P.reshape(-1, P.shape[-1])[:, :2]
            if len(P):
                out.append(resample(P))
    return np.vstack(out) if out else np.zeros((0, 2))


def _nearest(A, D, block=512):
    """Distance from each row of A to the nearest row of D. -> (len(A),)."""
    if not len(D):
        return np.full(len(A), np.inf)
    out = np.empty(len(A))
    for i in range(0, len(A), block):
        b = A[i:i + block]
        out[i:i + block] = np.sqrt(
            ((b[:, None, :] - D[None, :, :]) ** 2).sum(-1)).min(axis=1)
    return out


def uncovered(npz_path, A):
    """The artwork samples this schedule never inks. -> (K,2)."""
    return A[_nearest(A, drawn_points(npz_path)) > NEAR]


def report(miss, A, total, z, xs, ys, label):
    if not len(miss):
        print(f"\n{label}: nothing uncovered")
        return
    ix = np.array([int(np.argmin(np.abs(xs - v))) for v in miss[:, 0]])
    iy = np.array([int(np.argmin(np.abs(ys - v))) for v in miss[:, 1]])
    cells = sorted(set(zip(ix.tolist(), iy.tolist())))
    na = np.asarray(z["n_arms"])[iy, ix]
    print(f"\n{label}")
    print(f"  {len(miss)} of {len(A)} samples uncovered "
          f"({100.0 * len(miss) / len(A):.2f} % ~= "
          f"{total * len(miss) / len(A):.4f} m), over {len(cells)} cells")
    print(f"  bbox x [{miss[:, 0].min():.3f}, {miss[:, 0].max():.3f}]  "
          f"y [{miss[:, 1].min():.3f}, {miss[:, 1].max():.3f}]  "
          f"centroid ({miss[:, 0].mean():.3f}, {miss[:, 1].mean():.3f})")
    print(f"  ON CELLS THE SOLO MAP CALLS DEAD: {int((na == 0).sum())} samples "
          f"({100.0 * (na == 0).mean():.2f} %)")
    v, c = np.unique(na, return_counts=True)
    print("  arms that could draw it, solo:  "
          + "   ".join(f"{int(a)} arm(s): {int(n)}" for a, n in zip(v, c)))
    for x in sorted(FLEET):
        m = np.asarray(z[f"mask{x}"])[iy, ix]
        b = FLEET[x].T_world_base(H_INV_DEFAULT)[:2, 3]
        d = float(np.linalg.norm(b - miss.mean(axis=0)))
        print(f"     arm {x:>3}  base ({b[0]:.3f}, {b[1]:.3f})  "
              f"{d:.3f} m from the centroid   draws solo "
              f"{int(m.sum()):5d}/{len(miss)} ({100.0 * m.mean():5.1f} %)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--map", default=str(ROOT / "out/feasible_workspace_v12"))
    ap.add_argument("--against", default=None,
                    help="a second schedule npz; also prints ITS uncovered set "
                         "and the part of it the first run covered")
    a = ap.parse_args()

    sch = Path(a.npz).with_suffix(".json")
    expect = float(json.loads(sch.read_text())["traced_m"]) if sch.exists() \
        else None
    strokes, total = artwork(expect)
    A = np.vstack([resample(s["pts"]) for s in strokes])
    print(f"artwork {len(strokes)} strokes, {total:.4f} m, {len(A)} samples "
          f"at {1000 * STEP:.0f} mm")

    z = np.load(a.map + "_map.npz")
    xs, ys = np.asarray(z["xs"], float), np.asarray(z["ys"], float)
    j = json.load(open(a.map + ".json"))
    print(f"map {a.map}: {j['feasible_pct']} % feasible, grid {j['grid']} m")

    m0 = uncovered(a.npz, A)
    report(m0, A, total, z, xs, ys, f"=== {Path(a.npz).name} ===")
    if a.against:
        m1 = uncovered(a.against, A)
        report(m1, A, total, z, xs, ys, f"=== {Path(a.against).name} ===")
        extra = m1[_nearest(m1, m0) > NEAR] if len(m0) else m1
        report(extra, A, total, z, xs, ys,
               f"=== what {Path(a.against).name} loses and "
               f"{Path(a.npz).name} keeps ===")


if __name__ == "__main__":
    main()
