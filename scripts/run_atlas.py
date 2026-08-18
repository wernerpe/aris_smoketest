#!/usr/bin/env python3
"""Run the reachability sweep. Usage: run_atlas.py [arm_id ...] [--h-inv 0.924]"""
import argparse
import json
import multiprocessing as mp
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from aris_sixarm.atlas import sweep_arm, strict_go  # noqa: E402
from aris_sixarm.fleet import FLEET, H_INV_DEFAULT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="*", type=int, default=None)
    ap.add_argument("--h-inv", type=float, default=H_INV_DEFAULT)
    ap.add_argument("--grid", type=float, default=0.02)
    ap.add_argument("--tilt", type=float, default=15.0, help="max pen lean (deg)")
    ap.add_argument("--pen", type=float, default=0.110, help="pen tip below TCP (m)")
    ap.add_argument("--out", default=str(Path(__file__).parents[1] / "out"))
    a = ap.parse_args()
    arms = a.arms or list(FLEET)
    fn = partial(sweep_arm, out_dir=a.out, grid=a.grid, h_inv=a.h_inv,
                 tilt_max_deg=a.tilt, pen_ext=a.pen)
    if len(arms) > 1:
        with mp.Pool(min(6, len(arms))) as pool:
            arrs = pool.map(fn, arms)
    else:
        arrs = [fn(arms[0])]
    summary = {aid: dict(cells=len(arr), strict_go=int(strict_go(arr).sum()))
               for aid, arr in zip(arms, arrs)}
    json.dump(summary, open(Path(a.out) / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
