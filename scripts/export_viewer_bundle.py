#!/usr/bin/env python3
"""Turn a finished schedule into ONE viewer bundle (json + bin).

    ARIS_RIG=proposed ARIS_TOOL=lateral python3 scripts/export_viewer_bundle.py \
        out/csail_schedule_h094_v14.npz out/csail_program_h094_v14.json \
        --out out/v14_bundle.json --summary out/csail_schedule_h094_v14.json

THE RIG AND THE TOOL MUST BE THE ONES THE RUN USED.  A bundle carries per-arm
base transforms and the active pen offset, both read off the process globals
`fleet.FLEET` and `frames.PEN_LAT`; exported under the wrong rig it is a
picture of a different room, and nothing downstream would notice.  The GUI's
worker does this in-process for exactly that reason (it is already the right
process); this script is for a run that was made from a terminal, or for
re-exporting an old programme after the schema changes.

    --scene DIR    also write the rig-dependent scene (meshes, arm bases, the
                   static installation, the golden FK samples) beside it.  The
                   GUI builds this itself and caches it per (rig, tool); you
                   want it only when serving the viewer from somewhere else.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from aris_sixarm import fleet as fleet_mod                   # noqa: E402
from aris_sixarm import frames                               # noqa: E402
from aris_sixarm.program_schema import (export_bundle,       # noqa: E402
                                        export_scene)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", help="<name>_schedule.npz — the conducted timeline")
    ap.add_argument("program", help="<name>_program.json — the allocation")
    ap.add_argument("--summary", default=None,
                    help="<name>_schedule.json; without it the bundle has no "
                         "per-phase rows (everything else is unaffected)")
    ap.add_argument("--out", default=None,
                    help="the bundle json (default: <name>_bundle.json beside "
                         "the npz).  The binary is written with the same stem "
                         "and a .bin suffix")
    ap.add_argument("--scene", default=None, metavar="PATH",
                    help="also write the scene json (+ .bin) here")
    ap.add_argument("--no-clearance", action="store_true",
                    help="skip the per-frame clearance series; it is the only "
                         "part that costs anything (seconds, and about a third "
                         "of the binary)")
    ap.add_argument("--max-clearance-frames", type=int, default=4000,
                    help="stride the clearance series to at most this many "
                         "samples")
    a = ap.parse_args(argv)

    npz = Path(a.npz)
    summary = a.summary
    if summary is None:
        # BOTH WRITERS PUT THE SUMMARY BESIDE THE npz WITH THE SAME STEM:
        # `draw.py` writes `<name>_schedule.{npz,json}` and
        # `csail_schedule.py` writes `csail_schedule<tag>.{npz,json}`, so
        # swapping the suffix is the one guess that is right for both.
        # (Substring-replacing "_schedule.npz" is not: the tag comes AFTER
        # "schedule" in the CSAIL convention, so the replace is a no-op and
        # the guess is the npz itself.)
        guess = npz.with_suffix(".json")
        summary = str(guess) if guess.exists() and guess != npz else None
    out = Path(a.out) if a.out else npz.with_name(npz.stem + "_bundle.json")

    print(f"rig {fleet_mod.ACTIVE_RIG!r}, tool "
          f"{'lateral' if frames.PEN_LAT else 'inline'}, "
          f"{len(fleet_mod.FLEET)} arms in the fleet")
    b = export_bundle(npz, a.program, out, summary=summary,
                      clearance=not a.no_clearance,
                      max_clearance_frames=a.max_clearance_frames)
    bin_ = out.with_suffix(".bin")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} kB) and "
          f"{bin_} ({bin_.stat().st_size / 1024:.0f} kB)")
    print(f"  {b.meta.n_frames} frames at {b.meta.fps:g} fps "
          f"({b.meta.duration_s:.1f} s), {len(b.arms)} arms, "
          f"{len(b.segments)} segments, {len(b.strokes)} strokes, "
          f"{len(b.ink.hex)} ink chunks, {len(b.clearance.pairs)} arm pairs")
    print(f"  coverage {b.meta.coverage_pct:.4f} %, makespan "
          f"{b.meta.makespan_s:.3f} s, min clearance "
          f"{1000 * b.meta.min_clearance_m:.1f} mm "
          f"(margin {1000 * b.meta.margin_m:.0f} mm)")
    lean = [s for s in b.segments if s.lean_deg > 1e-6]
    if lean:
        print(f"  {len(lean)} of {len(b.segments)} segments lean, worst "
              f"{max(s.lean_deg for s in lean):.1f} deg")

    if a.scene:
        doc = export_scene(a.scene)
        print(f"wrote {a.scene} — {len(doc['arms'])} arms, "
              f"{len(doc['bodies'])} static bodies, "
              f"{len(doc['meshes'])} meshes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
