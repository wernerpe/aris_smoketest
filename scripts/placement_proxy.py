#!/usr/bin/env python3
"""WHICH PLACEMENTS CAN REACH 100 %, IN FIVE MINUTES INSTEAD OF SEVEN HOURS.

    ARIS_RIG=proposed ARIS_TOOL=lateral python3 scripts/placement_proxy.py \
        assets/csail/csail_old_med.gif --atlas out/atlas_proposed_h0940_lat0860 \
        --scales 0.85 --rotate 90 --top 10

THE METRIC IS THE LONGEST CONTIGUOUS DEAD RUN INSIDE ONE STROKE, and the choice
of "contiguous" is the whole idea.  Sample the traced ink every centimetre and
look each sample up in the 2 cm atlas union.  A SCATTERED dead sample is drawn
anyway — the atlas is a sampling of a continuous pose search, and a stroke that
clips the corner of a dead cell still has a certified pose almost everywhere
along it — but a RUN of them is a span no arm can cover, and that is what comes
back as ink left on the floor.

IT IS CALIBRATED, WHICH IS WHY IT IS WORTH TRUSTING.  On v15's own placement
(90 deg, scale 0.85, offset -0.10/0.00) it predicts **82 mm**; the v15 run
measured **80.3 mm** left empty, at the same place.  Right to 2 mm on the one
case where the answer was already known.

WHAT IT SAVED.  `--placement auto` at the run's own flag set cost 27 243 s in
v16 and chose a placement that lost 0.65 pp, because the search ranks on a
balance-free allocation and on an atlas PROXY of liveness, and neither can see
that one stroke crosses a base disc.  This ranks on the thing that actually
decides, costs minutes, and in v17 found a zero-dead-run offset that drew
100.0000 %.  Run it BEFORE paying for a search or a plan.

IT IS A PROXY AND IT IS NOT A PLAN.  Zero dead run does not promise 100 %: the
allocator still has to certify a pose at every point, fly to it and conduct it.
It promises only that no span is dead on the atlas, which is the failure that
cost v15 and v16 their ink.
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

from aris_sixarm import atlas as atlas_mod                       # noqa: E402
from aris_sixarm import layout, trace                            # noqa: E402
from aris_sixarm.fleet import SHEET                              # noqa: E402

GRID = 0.02
STEP = 0.01          # metres between ink samples


def union_mask(atlas_dir, fleet):
    """Cells with a certified drawing pose for ANY arm. -> (H, W) bool."""
    xs = np.arange(0.0, SHEET[0] + 1e-9, GRID)
    ys = np.arange(0.0, SHEET[1] + 1e-9, GRID)
    u = np.zeros((len(ys), len(xs)), bool)
    for a in sorted(fleet):
        arr, _ = atlas_mod.load(Path(atlas_dir), a)
        if len(arr):
            ii = np.rint(arr[:, 1] / GRID).astype(int)
            jj = np.rint(arr[:, 0] / GRID).astype(int)
            u[ii, jj] |= atlas_mod.strict_go(arr)
    return u


def dead_runs(strokes, u):
    """-> (longest contiguous dead run m, total dead m, min clearance cells).

    `clear` is how many cells the ink keeps away from the nearest dead cell,
    reported as the 5th percentile, so two zero-run placements can still be
    ranked on how much room they have before a modelling change bites.
    """
    from scipy import ndimage
    dist = ndimage.distance_transform_edt(u) * GRID
    H, W = u.shape
    longest = total = 0.0
    ds = []
    for s in strokes:
        p = np.asarray(s["pts"], float)
        if len(p) < 2:
            continue
        seg = np.diff(p, axis=0)
        L = np.hypot(seg[:, 0], seg[:, 1])
        run = 0.0
        for i in range(len(seg)):
            n = max(2, int(L[i] / STEP) + 1)
            d = L[i] / n
            q = p[i] + seg[i] * np.linspace(0, 1, n)[:, None]
            ii = np.rint(q[:, 1] / GRID).astype(int)
            jj = np.rint(q[:, 0] / GRID).astype(int)
            ok = (ii >= 0) & (ii < H) & (jj >= 0) & (jj < W)
            live = np.zeros(len(q), bool)
            live[ok] = u[ii[ok], jj[ok]]
            ds.append(dist[np.clip(ii, 0, H - 1), np.clip(jj, 0, W - 1)])
            for v in live:
                if v:
                    run = 0.0
                else:
                    run += d
                    total += d
                    longest = max(longest, run)
    clear = np.concatenate(ds) if ds else np.zeros(1)
    return longest, total, float(np.percentile(clear, 5)), float(clear.min())


def check_certified(info, cert, margin=0.0):
    """Is this placement's box inside the certified rectangle? -> dict.

    THE CERTIFIED RECTANGLE IS A PROMISE ABOUT ANYWHERE INSIDE IT, so the only
    question a placement has to answer is whether its bounding box fits — and
    if it does not, by how much and on which side, because "move it 3 cm east"
    is actionable and "refused" is not.

    `cert` is `scripts/certified_area.py`'s JSON (or its `rect.largest`).
    """
    r = cert.get("rect", {}).get("largest", cert) if isinstance(cert, dict) \
        else cert
    cx, cy = float(info["center"][0]), float(info["center"][1])
    w, h = float(info["logo_w"]), float(info["logo_h"])
    box = dict(x0=cx - w / 2, x1=cx + w / 2, y0=cy - h / 2, y1=cy + h / 2)
    over = dict(
        west=round(max(0.0, (r["x0"] + margin) - box["x0"]), 4),
        east=round(max(0.0, box["x1"] - (r["x1"] - margin)), 4),
        south=round(max(0.0, (r["y0"] + margin) - box["y0"]), 4),
        north=round(max(0.0, box["y1"] - (r["y1"] - margin)), 4))
    worst = max(over.values())
    return dict(inside=worst <= 0.0, overhang_m=over, worst_overhang_m=worst,
                box={k: round(v, 4) for k, v in box.items()},
                certified={k: r[k] for k in ("x0", "x1", "y0", "y1", "w", "h",
                                             "area_m2") if k in r})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("source")
    ap.add_argument("--atlas", required=True)
    ap.add_argument("--certified-area", default=None, metavar="JSON",
                    help="scripts/certified_area.py output.  Every candidate "
                         "is also checked against that hole-free rectangle "
                         "and the ones that do not fit are REFUSED, with the "
                         "overhang per side reported")
    ap.add_argument("--certified-margin", type=float, default=0.0,
                    help="metres of the certified rectangle to keep in hand")
    ap.add_argument("--rotate", type=float, nargs="+", default=[90.0])
    ap.add_argument("--scales", type=float, nargs="+", default=[0.85])
    ap.add_argument("--offset-range", type=float, default=0.30)
    ap.add_argument("--offset-step", type=float, default=0.01)
    ap.add_argument("--dy-step", type=float, default=0.05)
    ap.add_argument("--margin", type=float, default=0.06)
    ap.add_argument("--min-len", type=float, default=0.025)
    ap.add_argument("--base-width", type=float, default=1.6834,
                    help="scale 1.0 width; scale s means target_width s*this")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    fl = layout.FLEET_PROPOSED
    u = union_mask(a.atlas, fl)
    px, _ = trace.trace_any(a.source)
    cert = None
    if a.certified_area:
        cert = json.loads(Path(a.certified_area).read_text())
        cr = cert.get("rect", {}).get("largest", cert)
        print(f"certified rectangle: {cr['w']:.2f} x {cr['h']:.2f} m = "
              f"{cr['area_m2']:.3f} m² at ({cr['x0']:.2f}, {cr['y0']:.2f})"
              + (f", margin {a.certified_margin:.3f} m"
                 if a.certified_margin else ""))
    print(f"atlas {a.atlas}: {100 * u.mean():.2f} % of {u.size} cells live")
    print(f"traced {len(px)} strokes from {a.source}")
    rows, n_fit, n_refused = [], 0, 0
    R = a.offset_range
    for rot in a.rotate:
        for f in a.scales:
            tw = f * a.base_width
            for dx in np.arange(-R, R + 1e-9, a.offset_step):
                for dy in np.arange(-R, R + 1e-9, a.dy_step):
                    st, info = trace.to_sheet(
                        px, SHEET, margin=a.margin, min_len=a.min_len,
                        target_width=tw, offset=(float(dx), float(dy)),
                        rotate_deg=rot)
                    if not info["fits"]:
                        continue
                    n_fit += 1
                    if cert is not None:
                        c = check_certified(info, cert, a.certified_margin)
                        if not c["inside"]:
                            n_refused += 1
                            continue
                    lo, tot, p5, mn = dead_runs(st, u)
                    rows.append(dict(rot=rot, scale=f, dx=round(float(dx), 3),
                                     dy=round(float(dy), 3),
                                     longest_mm=round(1000 * lo, 1),
                                     total_mm=round(1000 * tot, 1),
                                     p5_clear_mm=round(1000 * p5, 1),
                                     min_clear_mm=round(1000 * mn, 1),
                                     logo_w=round(info["logo_w"], 4),
                                     logo_h=round(info["logo_h"], 4),
                                     centre=[round(v, 4)
                                             for v in info["center"]]))
    # zero dead run first, then the most room to spare, then the biggest logo
    rows.sort(key=lambda r: (r["longest_mm"], r["total_mm"],
                             -r["p5_clear_mm"], -r["logo_w"]))
    zero = [r for r in rows if r["longest_mm"] == 0.0]
    print(f"{n_fit} placements fit the sheet; "
          + (f"{n_refused} refused as outside the certified rectangle; "
             if cert is not None else "")
          + f"{len(zero)} of the rest have a ZERO contiguous dead run\n")
    if cert is not None and not rows:
        print("NOTHING FITS THE CERTIFIED RECTANGLE.  The overhang of the "
              "largest candidate says which way to move or shrink:")
        for rot in a.rotate:
            for f in a.scales:
                _st, info = trace.to_sheet(
                    px, SHEET, margin=a.margin, min_len=a.min_len,
                    target_width=f * a.base_width, offset=(0.0, 0.0),
                    rotate_deg=rot)
                c = check_certified(info, cert, a.certified_margin)
                print(f"  rot {rot:.0f} scale {f:.2f} centred: "
                      f"{info['logo_w']:.3f} x {info['logo_h']:.3f} m, "
                      f"overhang {c['overhang_m']}")
        return 1
    print(f"{'rot':>4} {'scale':>6} {'dx':>7} {'dy':>7} {'longest':>8} "
          f"{'total':>7} {'p5 clr':>7}  centre")
    for r in rows[:a.top]:
        print(f"{r['rot']:4.0f} {r['scale']:6.2f} {r['dx']:+7.2f} "
              f"{r['dy']:+7.2f} {r['longest_mm']:7.0f}m {r['total_mm']:6.0f}m "
              f"{r['p5_clear_mm']:6.0f}m  ({r['centre'][0]:.4f}, "
              f"{r['centre'][1]:.4f})")
    if a.json:
        Path(a.json).write_text(json.dumps(
            dict(atlas=a.atlas, source=a.source, n_fit=n_fit,
                 n_zero=len(zero), rows=rows[:200]), indent=1))
        print("\nwrote", a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
