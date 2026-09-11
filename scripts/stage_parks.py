"""Per-stage park sets for the eight-stage work-cell pattern — build item 2.

docs/V2_WORKCELLS.md §5 measured the shipped `Q_PARK_PROPOSED` against what an
arm could be TOLD to draw inside a work cell and got **+4.7 mm** (13 against
17) against a 50 mm gate.  That, and not the +85.8 mm of ink-vs-ink, is what
fails the gate, and it fails identically for every pattern measured.  This
script searches a park PER STAGE, ranked against that stage's envelopes, and
writes the set with its provenance.

What it reports, per stage:

  * the worst park-vs-ACTIVE-ENVELOPE clearance (the +4.7 mm number, re-earned);
  * the worst park-vs-park clearance (expected at the 250 mm broad-phase cap);
  * whether each arm can FLY from its park into its own cell and back
    (`paper.route` against the steel and its own metal — not against the other
    arms, which is build item 6's `check_timeline`);
  * the transition property: whether each arm's park for stage k is already
    inside its own cell for stage k+1, or needs a certified route.

    ARIS_RIG=proposed ARIS_TOOL=lateral python3 scripts/stage_parks.py \
        --atlas out/atlas_proposed_h0970_lat0860_gated63 --stride 2 \
        --json out/stage_parks_h0970.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from aris_sixarm import coordination as co          # noqa: E402
from aris_sixarm import layout, paper, traces, writing  # noqa: E402

import workcell_envelopes as we                     # noqa: E402


def cell_poses(atlas_dir, stride, fleet, h_inv, cache=None, verbose=True):
    """{arm: dict(x, y, q_draw, q_hover)} on the common lattice."""
    if cache and Path(cache).exists():
        z = np.load(cache)
        return {a: {k: z[f"{a}_{k}"] for k in ("x", "y", "q_draw", "q_hover")}
                for a in we.ARMS}
    xs, ys = we.lattice(stride)
    out = {a: we.arm_cells(a, atlas_dir, xs, ys, fleet[a], h_inv, verbose)
           for a in we.ARMS}
    if cache:
        np.savez_compressed(cache, **{f"{a}_{k}": v for a, d in out.items()
                                      for k, v in d.items()})
    return out


def tip_xy(spec, q, h_inv):
    return [float(v) for v in paper.tip_xy(np.asarray(q, float).reshape(7),
                                           spec, spec.pen, h_inv)[:2]]


def in_region(xy, region):
    x, y = float(xy[0]), float(xy[1])
    return any(x0 - 1e-9 <= x < x1 - 1e-9 and y0 - 1e-9 <= y < y1 - 1e-9
               for x0, y0, x1, y1 in region)


def nearest_hover(d, region, xy):
    """The hover over the certified cell of `region` nearest `xy`. -> (7,)|None."""
    x, y = np.asarray(d["x"], float), np.asarray(d["y"], float)
    m = np.zeros(len(x), bool)
    for x0, y0, x1, y1 in region:
        m |= (x >= x0 - 1e-9) & (x < x1 - 1e-9) & (y >= y0 - 1e-9) & (y < y1 - 1e-9)
    if not m.any():
        return None
    i = np.nonzero(m)[0]
    j = i[int(np.argmin((x[i] - xy[0]) ** 2 + (y[i] - xy[1]) ** 2))]
    return np.asarray(d["q_hover"], float)[j]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas",
                    default="out/atlas_proposed_h0970_lat0860_gated63")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--dead-band", type=float, default=traces.DEAD_BAND_M)
    ap.add_argument("--json", default="out/stage_parks_h0970.json")
    ap.add_argument("--cell-cache", default=None,
                    help="npz of the per-arm cell poses, written if absent")
    ap.add_argument("--no-routes", action="store_true",
                    help="skip the flyability and transition routes (the slow "
                         "half; the clearances are unaffected)")
    ap.add_argument("--no-leg-cache", action="store_true")
    a = ap.parse_args()

    fleet = layout.FLEET_PROPOSED
    h = float(layout.LAYOUT_PROPOSED["h"])
    gate = co.PAIR_MARGIN
    if not a.no_leg_cache:
        paper.disk_cache_open()
    pattern = traces.zigzag_pattern(a.dead_band)
    print(f"pattern {pattern.name}: {pattern.n_stages} stages, gate "
          f"{1000 * gate:.0f} mm, h = {h:.3f}")

    t0 = time.time()
    poses = cell_poses(a.atlas, a.stride, fleet, h, a.cell_cache)
    print(f"cells read in {time.time() - t0:.1f} s")

    t0 = time.time()
    parks, rows = layout.stage_parks(pattern, poses, fleet=fleet, h_inv=h,
                                     gate=gate, verbose=True)
    print(f"stage park search in {time.time() - t0:.1f} s")

    # ---- the shipped set, measured the same way, for the comparison
    shipped = {}
    for s in range(pattern.n_stages):
        env = layout.stage_envelopes(pattern, poses, s)
        paths = {b: layout._stage_path(b, env[b], fleet, h) for b in env}
        worst = np.inf
        pair = None
        for arm in sorted(fleet):
            q = np.asarray(layout.Q_PARK_PROPOSED[arm], float).reshape(1, 7)
            p = layout._stage_path(arm, q, fleet, h)
            for b in paths:
                if b == arm:
                    continue
                m = float(co.clearance_matrix(p, paths[b]).min())
                if m < worst:
                    worst, pair = m, (int(arm), int(b))
        shipped[s] = dict(worst_m=float(worst), pair=pair)

    out = dict(
        pattern=pattern.name, n_stages=pattern.n_stages, gate_m=float(gate),
        h=h, atlas=a.atlas, stride=int(a.stride),
        dead_band_m=float(a.dead_band),
        cache_signature=paper.cache_signature(),
        spec_signature={str(k): paper.spec_signature(v)
                        for k, v in sorted(fleet.items())},
        aside_recipe_grid=dict(bearings=list(layout.ASIDE_BEARINGS),
                               radii=list(layout.ASIDE_RADII),
                               hovers=list(layout.ASIDE_HOVERS)),
        shipped_park_vs_envelope=shipped, rows=rows, stages={})

    for s in range(pattern.n_stages):
        cells = {int(c.arm): c for c in pattern.stage(s)}
        mine = [r for r in rows if r["stage"] == s]
        pp, pair = layout.park_pair_clearance(parks[s], fleet, h)
        rec = dict(active=sorted(cells),
                   regions={str(k): [list(map(float, r)) for r in v.region]
                            for k, v in cells.items()},
                   worst_park_vs_envelope_m=float(
                       min(r["env_clear_m"] for r in mine)),
                   worst_park_vs_envelope_arm=int(
                       min(mine, key=lambda r: r["env_clear_m"])["arm"]),
                   worst_park_vs_park_m=float(pp),
                   worst_park_vs_park_pair=list(pair) if pair else None,
                   all_certified=all(r["certified"] for r in mine),
                   parks={}, )
        for arm in sorted(fleet):
            q = parks[s][arm]
            e = dict(q=[float(v) for v in q],
                     tip_xy=tip_xy(fleet[arm], q, h),
                     env_clear_m=float(
                         next(r["env_clear_m"] for r in mine
                              if r["arm"] == arm)),
                     recipe=next(r["recipe"] for r in mine if r["arm"] == arm))
            if arm in cells:
                e["in_own_cell"] = in_region(e["tip_xy"], cells[arm].region)
                if not a.no_routes:
                    qh = nearest_hover(poses[arm], cells[arm].region,
                                       e["tip_xy"])
                    r = None if qh is None else layout.repark_route(
                        fleet[arm], q, qh, h_inv=h, q_home=fleet[arm].q_seed)
                    e["flyable"] = r is not None
                    e["fly_mode"] = None if r is None else r["mode"]
            rec["parks"][str(arm)] = e
        out["stages"][str(s)] = rec

    # ---- the transition property
    trans = []
    for s in range(pattern.n_stages - 1):
        nxt = {int(c.arm): c for c in pattern.stage(s + 1)}
        for arm in sorted(fleet):
            q0, q1 = parks[s][arm], parks[s + 1][arm]
            same = bool(np.allclose(q0, q1, atol=1e-12))
            xy = tip_xy(fleet[arm], q0, h)
            inside = (arm in nxt) and in_region(xy, nxt[arm].region)
            row = dict(stage=s, arm=int(arm), park_unchanged=same,
                       in_next_cell=bool(inside), needs_route=not same)
            if not same and not a.no_routes:
                r = layout.repark_route(fleet[arm], q0, q1, h_inv=h,
                                        q_home=fleet[arm].q_seed)
                row["route"] = None if r is None else r["mode"]
                row["routed"] = r is not None
            trans.append(row)
    out["transitions"] = trans
    out["leg_cache"] = paper.cache_report()

    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {a.json}")
    print(f"{'stage':>5} {'active':>16} {'park-vs-env':>12} "
          f"{'shipped':>10} {'park-vs-park':>13} {'flyable':>9}")
    for s in range(pattern.n_stages):
        r = out["stages"][str(s)]
        fly = [v.get("flyable") for v in r["parks"].values()
               if "flyable" in v]
        print(f"{s:>5} {str(r['active']):>16} "
              f"{1000 * r['worst_park_vs_envelope_m']:>+11.1f} "
              f"{1000 * shipped[s]['worst_m']:>+9.1f} "
              f"{1000 * r['worst_park_vs_park_m']:>+12.1f} "
              f"{sum(1 for f in fly if f)}/{len(fly):>7}")
    need = [t for t in trans if t["needs_route"]]
    print(f"\ntransitions needing a route: {len(need)} of {len(trans)}; "
          f"routed {sum(1 for t in need if t.get('routed'))}")


if __name__ == "__main__":
    main()
