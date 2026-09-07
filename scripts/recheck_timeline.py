#!/usr/bin/env python3
"""INDEPENDENT `scene_check` OVER A SHIPPED SCHEDULE'S WHOLE MERGED TIMELINE.

    ARIS_RIG=proposed ARIS_TOOL=lateral python3 scripts/recheck_timeline.py \
        out/csail_schedule_h094_v18.npz [--sub 2] [--json OUT.json]

WHY THIS IS NOT THE CHECK THE RUN ALREADY DID.  `csail_schedule.build` runs
`scene_check.check_timeline` ONCE PER PHASE, on that phase's own frames.  The
`.npz` holds every frame of every arm end to end — through the pen swap, through
the freeze each arm holds while another phase draws, and through the seam
between two phases — and **no per-phase check covers a seam**.  Measured, that
is not academic: v15's tightest phase read 82.0 mm and the whole timeline read
**80.26 mm**, 1.7 mm tighter, against an 80 mm gate; and the h = 0.880 re-plan
of 2026-09-04 passed every per-phase check and then FAILED the merged one on
arm 2's self-collision at 19.4 mm.  A programme is only as good as its worst
instant, and some of those instants exist only between the phases.

WHAT IT READS AND WHAT IT ASSUMES.  Everything comes out of the `.npz` the run
shipped — `q_<arm>`, `dt`, `margin`, `pen_ext`, `seg_<arm>` for the drawing mask
— so this cannot silently check a different trajectory from the one on disk.
The FLEET is the shipped rig unless `--h` says otherwise, which is there for the
report-only re-plans at another mounting height (`scripts/replan_at_height.py`):
those `.npz` files were planned against a fleet that is not
`layout.FLEET_PROPOSED`, and checking them against the shipped one would be
measuring a machine that never ran.  `--parks` takes the depot set from a
`scripts/height_sweep.py park` JSON for the same reason.

`programs=None` deliberately: the per-segment re-validation needs the in-memory
plans, which the `.npz` does not carry.  Everything the merged timeline is FOR —
inter-arm, self, frame, neighbour columns, paper, joint limits, the frozen
poses — is computed here from the frames alone.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("ARIS_RIG", "proposed")
os.environ.setdefault("ARIS_TOOL", "lateral")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from aris_sixarm import layout, scene_check                      # noqa: E402


def fleet_for(h=None, parks=None):
    """The rig to check against. -> (fleet, h).

    `h=None` is the shipped rig, which is what every normal run wants.
    """
    if h is None:
        return layout.FLEET_PROPOSED, float(layout.LAYOUT_PROPOSED["h"])
    lay = layout.paired_grid(spacing=layout.PAIR_SPACING, rows=3, h=float(h))
    if parks:
        doc = json.loads(Path(parks).read_text())
        pk = {int(k): np.asarray(v["q"], float)
              for k, v in doc["best"].items()}
    else:
        pk = layout.certified_park_poses(layout.build_fleet(lay),
                                         layout.PARK_GRID_PROPOSED)
    fl = layout.build_fleet(lay, q_park=pk)
    zs = {round(float(s.T_world_base()[2, 3]), 6) for s in fl.values()}
    if zs != {round(float(h), 6)}:
        raise SystemExit(f"fleet is not at h = {h}: {sorted(zs)}")
    return fl, float(h)


def recheck(npz, sub=2, h=None, parks=None, verbose=True):
    """-> the `scene_check.check_timeline` report for the whole timeline."""
    z = np.load(npz, allow_pickle=False)
    arms = [int(v) for v in z["arms"]]
    dt = float(z["dt"])
    margin = float(z["margin"])
    pens = {int(x): float(p) for x, p in zip(z["arms"], z["pen_ext"])}
    q = {x: np.asarray(z[f"q_{x}"], float) for x in arms}
    draw = {x: np.asarray(z[f"seg_{x}"]) >= 0 for x in arms}
    fl, hh = fleet_for(h, parks)
    if verbose:
        M = len(next(iter(q.values())))
        print(f"{npz}: {M} frames, dt={dt:.4f}, margin={1000 * margin:.0f} mm, "
              f"h={hh}, phases={int(z['n_phases'])}, "
              f"duration={float(z['duration']):.3f} s")
        print(f"  pens {{{', '.join(f'{k}: {1000 * v:.1f}' for k, v in pens.items())}}} mm")
    return scene_check.check_timeline(q, dt, margin, programs=None, h_inv=hh,
                                      pen_ext=pens, sub=sub, verbose=verbose,
                                      fleet=fl, drawing=draw), margin


def summarise(rep, margin):
    """The six gates and their margins, one line each. -> list[str]."""
    out = [f"VERDICT {'PASS' if rep['ok'] else 'FAIL'}"]
    wp = rep.get("worst_pair")
    out.append(f"  min inter-arm {1000 * rep['min_clearance']:.2f} mm "
               f"(gate {1000 * margin:.0f}) "
               + (f"pair {wp[0]}-{wp[1]} at t = {wp[2]:.2f} s " if wp else "")
               + f"-> margin {1000 * (rep['min_clearance'] - margin):+.2f} mm")
    sc = rep.get("self_clearance") or {}
    if sc:
        a = min(sc, key=lambda k: sc[k])
        out.append(f"  self-collision {1000 * sc[a]:.1f} mm "
                   f"(gate {1000 * rep['self_margin']:.0f}) arm {a}"
                   + ("  FAIL" if rep.get("self_failed") else ""))
    fc = rep.get("frame_clearance") or {}
    if fc:
        a = min(fc, key=lambda k: fc[k][0])
        out.append(f"  frame {1000 * fc[a][0]:.1f} mm "
                   f"(gate {1000 * rep['frame_margin']:.0f}) arm {a} "
                   f"at t={fc[a][1]:.2f} s")
    cc = rep.get("column_clearance") or {}
    if cc:
        a = min(cc, key=lambda k: cc[k][0])
        out.append(f"  neighbour base column {1000 * cc[a][0]:.1f} mm "
                   f"(gate {1000 * margin:.0f}) arm {a} at t={cc[a][1]:.2f} s")
    pc = rep.get("paper_clearance") or {}
    if pc:
        ca = min(pc, key=lambda k: pc[k]["chain"])
        ta = min(pc, key=lambda k: pc[k]["tip"])
        out.append(f"  paper chain {1000 * pc[ca]['chain']:.1f} mm "
                   f"(gate {1000 * rep['paper_chain_margin']:.0f}) arm {ca}; "
                   f"tip {1000 * pc[ta]['tip']:.1f} mm "
                   f"(floor {1000 * rep['paper_tip_margin']:.0f}) arm {ta}")
    jm = rep.get("joint_margin") or {}
    if jm:
        out.append(f"  joint margin min {min(jm.values()):.4f}")
    fr = rep.get("frozen") or {}
    out.append(f"  frozen poses {sum(1 for v in fr.values() if v['ok'])}/"
               f"{len(fr)}, pen-below-paper {rep.get('frozen_pen_below_paper')}")
    out.append(f"  failures: frame {rep.get('frame_failed')} paper "
               f"{rep.get('paper_failed')} column {rep.get('column_failed')} "
               f"self {rep.get('self_failed')}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("npz")
    ap.add_argument("--sub", type=int, default=2,
                    help="extra samples interpolated between scheduled steps")
    ap.add_argument("--json", default=None)
    ap.add_argument("--h", type=float, default=None,
                    help="check against a fleet at this height instead of the "
                         "shipped one (for report-only re-plans)")
    ap.add_argument("--parks", default=None,
                    help="height_sweep.py park JSON supplying that fleet's depots")
    a = ap.parse_args(argv)
    rep, margin = recheck(a.npz, a.sub, a.h, a.parks)
    print()
    for line in summarise(rep, margin):
        print(line)
    if a.json:
        Path(a.json).write_text(json.dumps(
            {k: v for k, v in rep.items() if k != "segments"},
            default=str, indent=1))
        print("wrote", a.json)
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
