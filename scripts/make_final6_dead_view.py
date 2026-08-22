#!/usr/bin/env python3
"""The six-arm scene with WHAT CANNOT BE DRAWN drawn on it.

Two layers `scripts/make_final6_scene.py` does not carry, over the same rig,
the same frames and the same continuous canvas:

  dead_zones/   every 2 cm atlas cell no arm can reach, in the two tiers of
                `aris_sixarm/dead.py` — solid dark red where not even the
                PLANNER's own gates (margin 0.15, sigma 0.10) admit an arm, and
                orange where the planner does but the strict atlas
                (0.30 / 0.14) does not.  The headline "24.07 % dead" is the
                union of the two, and the split is the point: one tier is
                geometry and the other is a gate.

  undrawn_ink/  the shipped CSAIL programme's own dropped spans, laid on the
                paper where they would have been drawn — the 13.03 % that the
                86.97 % coverage number is the complement of, with every span
                attributed to a cause by `dead.attribute`.

    ARIS_RIG=final6_opt python3 scripts/make_final6_dead_view.py
        -> out/final6_dead_view.html, out/final6_dead_zones.npz,
           out/final6_undrawn_attribution.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from aris_sixarm import dead as dead_mod  # noqa: E402
from aris_sixarm.fleet import ACTIVE_RIG, FLEET, SHEET  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
import make_final6_scene as scene  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default=str(ROOT / "out/atlas_final6_opt"))
    ap.add_argument("--program", default=str(ROOT / "out/csail_program_final6.json"))
    ap.add_argument("--html", default=str(ROOT / "out/final6_dead_view.html"))
    ap.add_argument("--masks", default=str(ROOT / "out/final6_dead_zones.npz"))
    ap.add_argument("--json", default=str(ROOT / "out/final6_undrawn_attribution.json"))
    ap.add_argument("--grid", type=float, default=0.02)
    ap.add_argument("--probe-stride", type=int, default=dead_mod.PROBE_STRIDE)
    ap.add_argument("--probe-limit", type=int, default=120)
    ap.add_argument("--no-probe", action="store_true")
    ap.add_argument("--no-span-probe", action="store_true")
    ap.add_argument("--no-coverage", action="store_true")
    a = ap.parse_args(argv)

    arms = sorted(FLEET)
    print(f"rig {ACTIVE_RIG}, sheet {SHEET[0]:.4f} x {SHEET[1]:.5f} m, "
          f"arms {arms}")
    T = dead_mod.tiers(a.atlas, arms, SHEET, a.grid)
    c = T["counts"]
    print(f"cells {c['cells']}: strict-GO {c['strict_go']} "
          f"({c['strict_go_pct']:.2f} %), permissive-GO {c['permissive_go']} "
          f"({c['permissive_go_pct']:.2f} %)")
    print(f"  atlas-strict dead {c['dead_strict']} ({c['dead_strict_pct']:.2f} %)"
          f"  ->  permissive-dead {c['dead_permissive']} "
          f"({c['dead_permissive_pct']:.2f} %)")
    print(f"  strict-only-dead  {c['strict_only_dead']} "
          f"({c['strict_only_dead_pct']:.2f} %) — reachable paper the strict "
          f"sweep disowns")
    dead_mod.save(T, a.masks)

    probe = None
    if not a.no_probe:
        pens = {x: 0.110 for x in arms}
        probe = dead_mod.probe_cells(T, FLEET, pens, stride=a.probe_stride,
                                     limit=a.probe_limit)

    spans, prog = dead_mod.load_dropped(a.program)
    ink = dead_mod.arms_with_ink(prog)
    rows, att = dead_mod.attribute(spans, T, ink)
    print(f"undrawn: {att['n_spans']} spans, {att['total_m']:.4f} m")
    for k, v in sorted(att["by_cause_m"].items(), key=lambda kv: -kv[1]):
        print(f"    {k:18s} {v:7.4f} m  "
              f"{100 * v / max(att['total_m'], 1e-9):5.1f} %")

    # WHY "other" IS NOT A SHRUG.  Every leftover span is handed back to the
    # real planner, so the residual category carries the planner's own verdict
    # instead of an absence of one.
    sp = None
    if not a.no_span_probe:
        pens = {x: 0.110 for x in arms}
        sp = dead_mod.probe_spans(spans, rows, FLEET, ink, pens,
                                  causes=dead_mod.CAUSES)
        why = {}
        for g in sp:
            for v in g["per_arm"].values():
                k = f"{v['status']}/{v['reason']}"
                why[k] = why.get(k, 0) + 1
        for k, n in sorted(why.items(), key=lambda kv: -kv[1]):
            print(f"    planner says {k:32s} x{n}")
        att["span_probe_certified_m"] = float(
            sum(g["length_m"] for g in sp if g["certified_by"] is not None))

    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(dict(
        rig=ACTIVE_RIG, grid=a.grid, sheet=[float(SHEET[0]), float(SHEET[1])],
        arms=arms, gates=dict(strict_margin=dead_mod.STRICT_MARGIN,
                              strict_sigma=dead_mod.STRICT_SIGMA,
                              permissive_margin=dead_mod.PERMISSIVE_MARGIN,
                              permissive_sigma=dead_mod.PERMISSIVE_SIGMA),
        counts=c, probe=probe, span_probe=sp, totals=att,
        spans=[{k: (v if not isinstance(v, np.ndarray) else v.tolist())
                for k, v in r.items()} for r in rows]), indent=1))
    print(f"wrote {a.json}")

    scene.build(a.html, a.atlas, dead=T, dropped=spans, attribution=att,
                probe=probe, coverage=not a.no_coverage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
