"""`scripts/day1.py` — the hardware-day front door.

WHAT THESE TESTS ARE FOR.  `day1.py` is the only thing a person at the rig
types, so the two properties that matter are (a) a line it says is certified
really did go through the certified planner, the solo programme and the
INDEPENDENT scene check, and came out with a pathway CSV that carries the
joint reference; and (b) everything it cannot certify is a refusal with a
non-zero exit and no file — never a CSV for a line nobody graded.

The word half is deliberately not re-planned here: `word` plans nothing, and
its re-check of the shipped 2026-09-15 asset is exactly the thing to pin.  The
coverage refusal is pinned on a SYNTHETIC schedule JSON, because the real
failure it exists to catch (`--skip-unconductable` shipping a 1.2 %-covered
file under a log that says 84 %) costs an hour to reproduce and one dict to
state.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import day1                                                      # noqa: E402

# Lines both arms certify, checked by hand on 2026-09-16.  They sit OFF the
# seam line (y = 1.8153), which is the middle row's own line and where the
# go-home leg is tightest: arm 31 refuses a line there, and that refusal is a
# real property of the rig rather than a bad test fixture (see
# `test_line_refuses_an_unreachable_line`'s sibling note).
GOOD = {31: ((0.50, 1.70), (0.65, 1.70)),
        71: ((1.15, 1.95), (1.30, 1.95))}


def _read_csv(path):
    rows = [ln.rstrip("\n").split(",") for ln in open(path) if ln.strip()]
    return rows[0], rows[1:]


@pytest.mark.parametrize("arm", [31, 71])
def test_line_certifies_and_writes_a_joint_csv(arm, tmp_path):
    """A known-good short line: certified, a CSV with rows, q1..q7 on each."""
    p0, p1 = GOOD[arm]
    r = day1.plan_line(arm, np.array(p0), np.array(p1), name="t",
                       out_dir=tmp_path, verbose=False)
    s = r["summary"]
    assert s["certified"] and s["gates"]["ok"]
    # every gate has room, and the tightest one is named
    assert s["gates"]["min_inter_arm_m"] > day1.coordination.PAIR_MARGIN
    assert not s["gates"]["frame_failed"] and not s["gates"]["paper_failed"]
    assert s["duration_s"] > 0.0

    csv_path = tmp_path / f"t_{arm}.csv"
    assert csv_path.exists(), "a certified line must write its CSV"
    hdr, rows = _read_csv(csv_path)
    assert hdr == day1.pathway.CSV_COLUMNS
    assert len(rows) > 0
    # THE JOINT COLUMNS ARE THE POINT.  This is the file that answers "test our
    # redundancy resolution on the real thing": the planner's own choice of arm
    # configuration, carried alongside the Cartesian pose instead of thrown
    # away and re-solved by the controller.
    assert hdr[11:18] == ["q1", "q2", "q3", "q4", "q5", "q6", "q7"]
    for row in rows:
        assert all(c != "" for c in row[11:18])
        assert len(row) == len(day1.pathway.CSV_COLUMNS)
    assert any(row[2] == day1.pathway.KIND_DRAW for row in rows)

    # ...and the summary says what it assumed, so a wrong number is findable
    assert (tmp_path / f"t_{arm}.json").exists()
    js = json.loads((tmp_path / f"t_{arm}.json").read_text())
    assert js["tool_tip"]["pen_ext_m"] == pytest.approx(
        float(day1.frames.PEN_EXT_HOLDER))
    assert js["base"]["translation_m"][2] == pytest.approx(0.970)
    assert js["h_m"] == pytest.approx(0.970)
    assert len(js["frozen_arms"]) == 5      # the other five stand at their parks


def test_line_refuses_an_unreachable_line(tmp_path):
    """Off in the far corner: refused, non-zero, and NOTHING written.

    An uncertified line must never get a CSV — a CSV is the file somebody
    streams to a stiff controller, and its existence is the only signal that
    the line was graded.
    """
    with pytest.raises(day1.Refused):
        day1.plan_line(31, np.array([0.10, 3.50]), np.array([0.25, 3.50]),
                       name="bad", out_dir=tmp_path, verbose=False)
    assert not list(tmp_path.glob("*.csv"))
    assert not list(tmp_path.glob("*.npz"))


def test_line_refusal_is_a_non_zero_exit_with_a_plain_message(tmp_path):
    """What a person at the rig actually sees: one line, and exit 1."""
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "day1.py"), "line",
         "--arm", "31", "--from", "0.10,3.50", "--to", "0.25,3.50",
         "--out", str(tmp_path)],
        cwd=str(ROOT), capture_output=True, text=True, timeout=900)
    assert r.returncode != 0
    assert "REFUSED" in (r.stderr + r.stdout)
    assert "Traceback" not in r.stderr
    assert not list(tmp_path.glob("*.csv"))


def test_hover_line_flies_above_the_paper():
    """`--hover 0.03` is graded with the REAL pen and reads ~30 mm of tip.

    The hover is a 30 mm LONGER PEN in the plan, not a lower plate — the same
    trajectory HARDWARE_DAY1 §4.1 gets by planning at h = 0.940 and flying at
    0.970.  The proof that the two really are the same construction is that the
    independent check, run with the REAL pen at the REAL height, reports the
    tip 30 mm clear of the paper.  Nothing is written: `write=False`.
    """
    p0, p1 = GOOD[31]
    r = day1.plan_line(31, np.array(p0), np.array(p1), name="h",
                       hover=0.030, write=False, verbose=False)
    tip = r["summary"]["gates"]["min_paper_tip_m"]
    assert tip == pytest.approx(0.030, abs=0.004), \
        f"a 30 mm hover should ride 30 mm off the paper; got {1000 * tip:.1f} mm"


@pytest.mark.skipif(not day1.VARIANTS["alt"]["npz"].exists(),
                    reason="the 2026-09-15 alternating asset is not in out/")
def test_word_alt_recheck_passes_on_the_shipped_asset(tmp_path):
    """The deliverable re-certifies today, and to the published number.

    155.56 mm is `docs/HARDWARE_DAY1.md` §4.3's own figure for the alternating
    programme.  Pinning it here means a change anywhere in the gate stack that
    moves the day's headline clearance breaks a test instead of surfacing on a
    robot.
    """
    rep, margin = day1.rt.recheck(str(day1.VARIANTS["alt"]["npz"]),
                                  sub=day1.SUB, verbose=False)
    assert rep["ok"], day1.rt.summarise(rep, margin)
    assert 1000 * rep["min_clearance"] == pytest.approx(155.56, abs=0.1)
    assert tuple(rep["worst_pair"][:2]) == (31, 71)


def test_coverage_refusal_on_a_schedule_that_drew_nothing(tmp_path):
    """The h = 0.850 failure mode, stated as a dict.

    That run logged `COVERAGE 84.2074 %` — the ALLOCATOR's number — over a file
    whose conducted coverage was 0.0123 and in which arm 71 never moved.
    `--skip-unconductable` is exactly the flag that lets that gap open quietly,
    so the only number `day1 word --replan` is allowed to believe is
    `coverage` out of the schedule JSON.
    """
    p = tmp_path / "s.json"
    p.write_text(json.dumps(dict(
        coverage=0.01, traced_m=2.5922, drawn_m=0.0319,
        arm_metres={"31": 0.0319, "71": 0.0},
        arm_segments={"31": 1, "71": 0},
        skipped_phases=["single pass"], skipped_m=2.15)))
    ok, lines = day1.check_replan(p)
    assert not ok
    body = "\n".join(lines)
    assert "REFUSED" in body
    assert "1.0000 %" in body or "1.00" in body
    assert "[71]" in body          # the arm that drew nothing is named

    # ...and a real one passes, so the gate is not simply always-refuse
    p2 = tmp_path / "ok.json"
    p2.write_text(json.dumps(dict(
        coverage=1.0, arm_metres={"31": 1.09, "71": 1.50},
        arm_segments={"31": 7, "71": 6}, skipped_phases=[], skipped_m=0.0)))
    ok2, _ = day1.check_replan(p2)
    assert ok2


def test_the_site_assets_resolve_on_a_clone_with_no_out(tmp_path,
                                                        monkeypatch):
    """THE ROBOT PC HAS NO `out/`, AND `word` MUST STILL RUN THERE.

    `out/` is gitignored, so a fresh clone carries none of the 0.970 artefacts
    `word` re-checks and copies.  They are tracked under `assets/site/h0970/`
    with the same names, and `day1.asset` prefers `out/` when it is there — so
    a workstation that has just re-planned sees its own new file and the robot
    PC sees the shipped one.
    """
    # every file `word` reads is in the tracked tree
    site = ROOT / "assets" / "site" / "h0970"
    assert (site / "unknown_h0970_home_alt.npz").is_file()
    assert (site / "pathways" / "unknown_h0970_home_alt_arm31.csv").is_file()
    assert (site / "pathways" / "unknown_h0970_home_alt_arm71.csv").is_file()
    assert (site / "unknown_strokes.json").is_file()
    assert (site / "atlas_proposed_h0970_lat0860"
            / "atlas_arm31.npz").is_file()

    # out/ wins when it is there...
    out_copy = ROOT / "out" / "unknown_h0970_home_alt.npz"
    if out_copy.exists():
        assert day1.asset("unknown_h0970_home_alt.npz") == out_copy
    # ...and the fallback is the tracked tree when it is not
    monkeypatch.setattr(day1, "ASSETS", tmp_path)
    (tmp_path / "nothing_is_in_out.json").write_text("{}")
    assert day1.asset("nothing_is_in_out.json") == \
        tmp_path / "nothing_is_in_out.json"
    # a file in NEITHER place names the place it is normally written, so the
    # refusal points at out/ and not at the fallback
    assert day1.asset("no_such_file.json") == ROOT / "out" / "no_such_file.json"


def test_help_is_readable_without_env_vars():
    """One `--help` a person can read in a minute, with no rig in the env."""
    import os
    env = {k: v for k, v in os.environ.items()
           if k not in ("ARIS_RIG", "ARIS_TOOL")}
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "day1.py"), "--help"],
                       cwd=str(ROOT), capture_output=True, text=True, env=env,
                       timeout=300)
    assert r.returncode == 0
    for want in ("line", "word", "--hover", "1.8153", "out/day1"):
        assert want in r.stdout
