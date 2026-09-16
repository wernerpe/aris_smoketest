"""The one adapter between a hardware-day artefact and the browser viewer.

`program_schema.export_bundle` reads the npz `scripts/csail_schedule.py`
writes, and neither of the two files hardware day actually flies is quite that
npz:

  * `scripts/day1.py line` writes a six-arm timeline with the ANIMATION arrays
    left out on purpose — see its `_payload` docstring: a one-line run has no
    tracer behind it, so `segpts_`/`segoff_` (the plan's own dense tip path per
    segment) and the `ink_*` chunks would have to be fabricated, and a
    fabricated array is one a viewer believes.
  * the alternating word, `out/unknown_h0970_home_alt.npz`, is
    `scripts/serialise_timeline.py`'s re-ordering of the conducted file into
    two blocks that never overlap.  It carries the joint trajectories and the
    segment index and drops everything the animation used, `n_frames`
    included.

So this module fills those keys IN A COPY and hands the copy to the shipped
exporter.  It invents no geometry: every array it adds is EMPTY, which the
viewer reads as "this run recorded no dense tip path" and falls back to
slicing the target stroke by each segment's s-range (`scene3d.addSpan`).  The
joint trajectories, the segment index, the clearance series and every gate
number are the file's own.

THE RIG IS THIS PROCESS'S.  `export_bundle` reads `fleet.FLEET` and
`frames.PEN_LAT`, which are chosen at import time from ARIS_RIG / ARIS_TOOL —
so this is called from the GUI worker, which is already a process started for
the job's rig, and never from the server.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np


def normalise(npz_path, out_npz):
    """Copy an npz, adding whatever `export_bundle` needs and it lacks.

    -> the list of key names that had to be added, so a caller can say so.
    """
    z = np.load(Path(npz_path), allow_pickle=False)
    d = {k: z[k] for k in z.files}
    arms = [int(a) for a in np.atleast_1d(d["arms"])]
    added = []

    def put(key, value):
        if key not in d:
            d[key] = value
            added.append(key)

    n = int(len(np.asarray(d[f"q_{arms[0]}"])))
    put("n_frames", np.int64(n))
    put("min_clearance", np.float64(d.get("margin", 0.0)))
    put("pause_s", np.float64(0.0))
    put("phase_start_s", np.zeros(int(d.get("n_phases", 1)), float))
    # the ink chunks: none were recorded, and none are invented
    put("ink_t", np.zeros(0, float))
    put("ink_arm", np.zeros(0, np.int64))
    put("ink_off", np.zeros(1, np.int64))
    put("ink_xyz", np.zeros((0, 3), float))
    put("ink_hex", np.array([], dtype="<U7"))
    for a in arms:
        # (0, 2) points and a single CSR offset: zero segments of dense tip
        # path.  `web/viewer/js/app.js` tests `drawnOff.length > 1` before it
        # draws one, so this is the shape that means "there is none".
        put(f"segpts_{a}", np.zeros((0, 2), float))
        put(f"segoff_{a}", np.zeros(1, np.int64))
        put(f"u_{a}", np.zeros(n, float))
    out_npz = Path(out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **d)
    return added


def export(npz_path, program_json, out_json, summary=None):
    """npz + programme json -> `bundle.json` and `bundle.bin`. -> the Bundle.

    `out_json` names the JSON; the exporter writes the binary beside it.
    """
    from ..program_schema import export_bundle
    npz_path, out_json = Path(npz_path), Path(out_json)
    summary = Path(summary) if summary and Path(summary).exists() else None
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / npz_path.name
        normalise(npz_path, tmp)
        return export_bundle(tmp, program_json, out_json, summary=summary)
