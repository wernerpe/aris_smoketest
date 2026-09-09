"""The two reporting instruments whose output `docs/DECISIONS.md` quotes.

A number a committed document states should come from something the repository
can run again, and these two produce a lot of them: every published
whole-timeline `scene_check` verdict (v15's 80.26 mm, v16's 80.25, v17's 80.91)
and every planner-cost table (v15's balance 65.5 %, v16's flycheck 57.7 %).

The schedule re-checker needs a shipped `.npz`, which lives in gitignored
`out/`, so that test SKIPS on a clean checkout rather than asserting against a
file that is not there — the same rule the atlas-backed tests in
`test_layout.py` follow.  The substage extractor needs nothing but a log it
writes itself, so it always runs.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


# ---------------------------------------------------------------------------
# the substage extractor
# ---------------------------------------------------------------------------
def _events(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_substages_sums_per_sub_and_per_stage(tmp_path):
    import job_substages
    p = tmp_path / "events.jsonl"
    _events(p, [
        {"kind": "substage_end", "stage": "allocation",
         "payload": {"sub": "balance", "elapsed_s": 10.0}},
        {"kind": "substage_end", "stage": "allocation",
         "payload": {"sub": "balance", "elapsed_s": 5.5}},
        {"kind": "substage_end", "stage": "allocation",
         "payload": {"sub": "probe", "elapsed_s": 2.0}},
        {"kind": "stage_end", "stage": "allocation",
         "payload": {"elapsed_s": 20.0}},
        {"kind": "stage_end", "stage": "conduction",
         "payload": {"elapsed_s": 4.0}},
        {"kind": "job_end", "stage": "",
         "payload": {"ok": True, "elapsed_s": 25.0}},
    ])
    tot, n, stage, stage_n, job = job_substages.read(p)
    assert tot["balance"] == pytest.approx(15.5)
    assert n["balance"] == 2
    assert tot["probe"] == pytest.approx(2.0)
    assert stage["allocation"] == pytest.approx(20.0)
    assert stage_n["conduction"] == 1
    assert job["ok"] is True and job["elapsed_s"] == 25.0


def test_substages_takes_a_directory_and_survives_a_corrupt_line(tmp_path):
    """A truncated line is not a stream: the worker flushes per event and a
    reader can catch one half-written, which must not abort the tail."""
    import job_substages
    d = tmp_path / "job"
    d.mkdir()
    p = d / "events.jsonl"
    with open(p, "w") as f:
        f.write(json.dumps({"kind": "substage_end", "stage": "a",
                            "payload": {"sub": "merge",
                                        "elapsed_s": 3.0}}) + "\n")
        f.write('{"kind": "substage_end", "payl\n')          # corrupt
        f.write(json.dumps({"kind": "substage_end", "stage": "a",
                            "payload": {"sub": "merge",
                                        "elapsed_s": 4.0}}) + "\n")
    tot, _, _, _, _ = job_substages.read(d)                  # a DIRECTORY
    assert tot["merge"] == pytest.approx(7.0)


def test_substages_runs_as_a_script(tmp_path):
    p = tmp_path / "events.jsonl"
    _events(p, [{"kind": "substage_end", "stage": "allocation",
                 "payload": {"sub": "flycheck", "elapsed_s": 1.25}}])
    r = subprocess.run([sys.executable,
                        str(ROOT / "scripts/job_substages.py"), str(p)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "flycheck" in r.stdout and "1.2" in r.stdout


# ---------------------------------------------------------------------------
# the whole-timeline scene_check re-checker
# ---------------------------------------------------------------------------
def _newest_schedule():
    out = ROOT / "out"
    cands = sorted(out.glob("csail_schedule_h*_v*.npz"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def test_recheck_reads_the_shipped_npz_and_gates_on_the_whole_timeline():
    """The merged timeline is at least as tight as the tightest phase.

    That is the entire reason this script exists: a per-phase check cannot see
    the seam between two phases, and every shipped programme so far has been
    tighter end-to-end than in any one phase.
    """
    npz = _newest_schedule()
    if npz is None:
        pytest.skip("no shipped schedule in gitignored out/ to re-check")
    import recheck_timeline
    rep, margin = recheck_timeline.recheck(str(npz), sub=2, verbose=False)
    z = np.load(npz, allow_pickle=False)
    n_frames = len(z[f"q_{int(z['arms'][0])}"])
    assert rep["n_frames"] == n_frames, "must check EVERY frame on disk"
    assert rep["min_clearance"] <= float(z["min_clearance"]) + 1e-9, (
        "the merged timeline cannot be looser than the per-phase minimum the "
        "run recorded; if it is, this is not checking the whole timeline")
    assert set(rep["pens"]) == {int(v) for v in z["arms"]}
    for v in rep["pens"].values():                 # the pen the run shipped
        assert v == pytest.approx(float(z["pen_ext"][0]))
    lines = recheck_timeline.summarise(rep, margin)
    assert lines[0].startswith("VERDICT")
    assert any("min inter-arm" in ln for ln in lines)


def test_certified_area_gate_accepts_what_fits_and_measures_what_does_not():
    """A placement is inside the hole-free rectangle, or it is refused WITH the
    overhang — "move it 3 cm east" is actionable and "refused" is not."""
    import placement_proxy
    cert = {"rect": {"largest": {"x0": 0.60, "x1": 1.20, "y0": 0.50,
                                 "y1": 2.50, "w": 0.60, "h": 2.00,
                                 "area_m2": 1.20}}}
    inside = {"center": (0.90, 1.50), "logo_w": 0.40, "logo_h": 1.00}
    c = placement_proxy.check_certified(inside, cert)
    assert c["inside"] and c["worst_overhang_m"] == 0.0

    # a box too wide, off-centre to the west: west overhang only
    wide = {"center": (0.85, 1.50), "logo_w": 0.80, "logo_h": 1.00}
    c = placement_proxy.check_certified(wide, cert)
    assert not c["inside"]
    assert c["overhang_m"]["west"] == pytest.approx(0.15)
    assert c["overhang_m"]["east"] == pytest.approx(0.05)
    assert c["overhang_m"]["north"] == 0.0 and c["overhang_m"]["south"] == 0.0
    assert c["worst_overhang_m"] == pytest.approx(0.15)

    # ...and the margin eats into the rectangle, never out of it
    tight = placement_proxy.check_certified(inside, cert, margin=0.15)
    assert not tight["inside"]
    assert tight["overhang_m"]["west"] == pytest.approx(0.05)

    # the flat form (just the rect) is accepted as well as the whole document
    flat = placement_proxy.check_certified(inside, cert["rect"]["largest"])
    assert flat["inside"]


@pytest.mark.parametrize("name", ["certified_area_h0940.json",
                                  "certified_area_h0970.json"])
def test_certified_area_rectangle_really_has_no_dead_cell_in_it(name):
    """The promise the JSON makes, checked against THE MAP IT CAME FROM.

    THE MAP IS READ OUT OF THE DOCUMENT, not hardcoded here, and that is the
    whole point of `certified_area.py` writing a `source`.  A certified area is
    a statement about one map; re-issue the map — a tighter certificate, a
    finished rung, a different park set — and a rectangle checked against some
    OTHER map is comparing two different canvases.  Hardcoding a path made this
    test fail on 2026-09-09 against a rectangle that was in fact clean: the
    JSON had been re-issued from `fw_h0940_final_map.npz` (the adaptive
    certificate) and the test was still reading `feasible_workspace_v14_map`,
    where that one cell was still dead.

    EVERY rectangle in the document is checked, not just the largest, because
    `placement_proxy` will happily hand a caller the near-square or the
    landscape one.
    """
    import certified_area
    j = ROOT / "out" / name
    if not j.exists():
        pytest.skip(f"no {name} in gitignored out/")
    doc = json.loads(j.read_text())
    src = doc.get("source")
    if not src or not Path(src).is_file() or not str(src).endswith(".npz"):
        pytest.skip(f"{name}: source {src!r} is not a readable map npz "
                    "(the draw-pose-only variants are sourced from an atlas)")
    live, xs, ys = certified_area.live_from_map(str(src))
    for which, r in doc["rect"].items():
        j0 = int(np.argmin(np.abs(xs - r["x0"])))
        j1 = int(np.argmin(np.abs(xs - r["x1"])))
        i0 = int(np.argmin(np.abs(ys - r["y0"])))
        i1 = int(np.argmin(np.abs(ys - r["y1"])))
        block = live[i0:i1 + 1, j0:j1 + 1]
        assert block.all(), (
            f"{name} [{which}]: {int((~block).sum())} dead cells inside a "
            f"rectangle the JSON calls certified (source {src}); a drawing "
            "placed there would not be drawable")
        assert block.size == r["cells"], (
            f"{name} [{which}]: rectangle spans {block.size} cells but the "
            f"JSON says {r['cells']}")


def test_recheck_refuses_a_fleet_that_is_not_at_the_height_asked_for():
    """`--h` exists for the report-only re-plans, and it must prove itself."""
    import recheck_timeline
    fl, h = recheck_timeline.fleet_for(0.880)
    assert h == 0.880
    zs = {round(float(s.T_world_base()[2, 3]), 6) for s in fl.values()}
    assert zs == {0.880}
    fl0, h0 = recheck_timeline.fleet_for(None)     # the shipped rig
    from aris_sixarm import layout
    assert fl0 is layout.FLEET_PROPOSED
    assert h0 == pytest.approx(float(layout.LAYOUT_PROPOSED["h"]))
