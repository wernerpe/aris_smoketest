#!/usr/bin/env python
"""The bench corpus as POLYLINE FILES the staged pipeline can be pointed at.

    .venv/bin/python -m scripts.bench_lines --out-dir out --sheet installed

`aris_sixarm/bench` generates its five drawings into `bench.SHEET`, which is
**1.8034 x 1.700 m** -- the sheet the corpus was pinned on.  The paper the
fleet actually has is **1.8034 x 3.63064 m** (`csail_schedule_h097_v19_strokes`
carries it), and a drawing that lives in the bottom half of it is a test of two
rows rather than of the fleet.  So `--sheet installed` regenerates the same
seeded drawings into the real sheet: the seeds are pinned, the generators are
untouched, and only the rectangle they fill changes.  `--sheet corpus` keeps
docs/BENCH.md's own rectangle.

Colour is dropped on the way out, because `traces` plans GEOMETRY: the staged
pipeline has no ink concept and `duotone`'s two colours are two sets of
polylines to it.  That is stated here rather than discovered downstream.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from aris_sixarm import bench

INSTALLED_SHEET = (1.8034, 3.63064)


def dump(name: str, sheet, path: Path) -> dict:
    strokes, meta = bench.make(name, sheet=tuple(sheet))
    out = [dict(id=int(i), color=str(s.get("color", "grey")),
                kind=str(s.get("kind", "bench")),
                pts=np.asarray(s["pts"], float).reshape(-1, 2).tolist())
           for i, s in enumerate(strokes)]
    total = float(bench.total_length(strokes))
    path.write_text(json.dumps(dict(
        sheet=[float(x) for x in sheet], n_strokes=len(out),
        total_length=total, info=dict(bench=name, regime=meta.get("regime")),
        strokes=out)))
    return dict(name=name, n=len(out), total_m=round(total, 3),
                path=str(path))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--sheet", choices=("installed", "corpus"),
                    default="installed")
    ap.add_argument("--names", default=",".join(bench.ORDER))
    a = ap.parse_args(argv)
    sheet = INSTALLED_SHEET if a.sheet == "installed" else bench.SHEET
    d = Path(a.out_dir)
    d.mkdir(parents=True, exist_ok=True)
    for n in a.names.split(","):
        r = dump(n.strip(), sheet, d / f"bench_{n.strip()}_lines.json")
        print(f"  {r['name']:10s} {r['n']:4d} strokes  {r['total_m']:7.3f} m"
              f"  -> {r['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
