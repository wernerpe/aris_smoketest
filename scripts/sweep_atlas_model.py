"""Re-sweep an atlas under the 2026-09-09 NEIGHBOUR MODEL.

The drawing-pose gate has always run against the neighbours' body bands as
their BOUNDING BOXES.  `aris_sixarm/envelope.py` showed what that costs — band
3's AABB hangs 160 mm below where the arm's body ends — and the router and the
hover gate were moved off it.  The atlas was not, and `NO_DRAW` is where most
of the dead canvas lives (378 of 400 cells at h = 0.970).  This sweeps it
again with the same room the router now uses:

  * body columns as CYLINDERS for a partner that could be moving, and
  * the partner's REAL capsules at its certified park, which is the solo map's
    own premise (`aris_sixarm/frozen.py`) — so the atlas a solo map reads is
    gated against the arms that will actually be standing there.

A cell certified this way carries the same dependency the map does: the named
arms must hold the named poses.  `--legacy-bands` reproduces the shipped sweep.

    sweep_atlas_model.py --h 0.970 --out DIR [--parks JSON] [--jobs 6]
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aris_sixarm import atlas, envelope, frames, frozen, layout  # noqa: E402
from aris_sixarm.atlas import sweep_arm  # noqa: E402

SHEET = (1.8034, 3.63064)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parks", default=None,
                    help="height_sweep.py park JSON; default is the shipped "
                         "layout.Q_PARK_PROPOSED")
    ap.add_argument("--grid", type=float, default=0.02)
    ap.add_argument("--tilt", type=float, default=15.0)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--legacy-bands", action="store_true",
                    help="reproduce the shipped sweep: AABB bands, no frozen "
                         "partners")
    a = ap.parse_args()

    h = float(a.h)
    fl = layout.build_fleet(layout.paired_grid(spacing=0.61, rows=3, h=h))
    pens = {x: fl[x].pen for x in fl}
    if a.parks:
        doc = json.load(open(a.parks))
        if not doc.get("certifies"):
            raise SystemExit(f"{a.parks}: that park search certifies nothing; "
                             "refusing to gate an atlas on it")
        parks = {int(k): np.asarray(v["q"], float)
                 for k, v in doc["best"].items()}
    else:
        parks = {int(k): np.asarray(v, float)
                 for k, v in layout.Q_PARK_PROPOSED.items()}

    if a.legacy_bands:
        envelope.uninstall()
        frozen.thaw()
    else:
        envelope.install(fl, h)
        frozen.freeze(parks, fl, pens, h)
    print(f"h={h:.3f} tool PEN_LAT={frames.PEN_LAT:.7f} grid={a.grid} "
          f"tilt<={a.tilt:.0f}deg")
    print(f"  neighbour model: body columns as "
          f"{'CYLINDERS' if envelope.active() else 'bounding boxes'} ; "
          f"partners {'FROZEN at their parks' if frozen.active() else 'pose-invariant'}")
    print(f"  parks: {a.parks or 'layout.Q_PARK_PROPOSED'}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    arms = sorted(fl)
    # the HOLDER's own axial depth, explicitly: `None` would be stored in the
    # npz as an object array and no reader can load it without allow_pickle
    fn = partial(sweep_arm, out_dir=str(out), grid=a.grid, h_inv=h,
                 tilt_max_deg=a.tilt, pen_ext=frames.PEN_EXT_HOLDER,
                 fleet=fl, sheet=SHEET, pen_lat=frames.PEN_LAT_HOLDER)
    t0 = time.time()
    with mp.get_context("fork").Pool(min(a.jobs, len(arms))) as pool:
        arrs = pool.map(fn, arms)
    tot = sum(int(atlas.strict_go(x).sum()) for x in arrs)
    print(f"  {tot} strict-GO arm-cells over {len(arms)} arms "
          f"in {time.time() - t0:.0f}s -> {out}")
    (out / "sweep_provenance.json").write_text(json.dumps(dict(
        h=h, grid=a.grid, tilt_max_deg=a.tilt,
        neighbour_model=("aabb+bands" if a.legacy_bands else "cyl+frozen"),
        parks_source=(a.parks or "layout.Q_PARK_PROPOSED"),
        frozen_arms=sorted(int(k) for k in frozen.poses()),
        frozen_poses={str(k): [round(float(v), 6) for v in q]
                      for k, q in frozen.poses().items()},
        obligation=("cells certified here depend on each named arm holding "
                    "its named pose while another arm draws; the conductor's "
                    "phase freeze is what enforces it (scene_check reports "
                    "'frozen N/N')")), indent=1) + "\n")


if __name__ == "__main__":
    main()
