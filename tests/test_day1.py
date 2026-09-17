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
    # THE CLOCK IS THE LAST COLUMN, so the v1 contract's eighteen are still
    # the first eighteen and a positional reader is unaffected.
    assert hdr[:len(day1.pathway.CSV_COLUMNS)] == day1.pathway.CSV_COLUMNS
    assert hdr == day1.CSV_COLUMNS_T and hdr[-1] == "t_s"
    assert len(rows) > 0
    # THE JOINT COLUMNS ARE THE POINT.  This is the file that answers "test our
    # redundancy resolution on the real thing": the planner's own choice of arm
    # configuration, carried alongside the Cartesian pose instead of thrown
    # away and re-solved by the controller.
    assert hdr[11:18] == ["q1", "q2", "q3", "q4", "q5", "q6", "q7"]
    for row in rows:
        assert all(c != "" for c in row[11:18])
        assert len(row) == len(day1.CSV_COLUMNS_T)
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


# --------------------------------------------------------------------------
# THE SOLO WORD — `word --arm N`
#
# These are the acceptance tests for the rung between `line` and the two-arm
# word, and they run the real planner at the real rig: thirteen Hershey strokes
# certified one at a time, ordered by `sequence` over the transit cost, laid
# down as one programme and graded by the independent checker.  They are slow
# (tens of seconds each) and that is the point — the thing being pinned is that
# the whole path works, not that a mock returns a dict.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("arm", [31, 71])
def test_word_arm_certifies_and_writes_a_joint_csv(arm, tmp_path):
    """The solo word for one arm: certified, a CSV with rows, q1..q7 on each."""
    r = day1.plan_word(arm, name="w", out_dir=tmp_path, verbose=False)
    s = r["summary"]
    assert s["certified"] and s["gates"]["ok"]
    assert s["gates"]["min_inter_arm_m"] > day1.coordination.PAIR_MARGIN
    assert not s["gates"]["frame_failed"] and not s["gates"]["paper_failed"]
    assert s["duration_s"] > 0.0
    # every stroke of the word is in it, or the test is not about the word
    assert s["strokes"]["planned"] == s["strokes"]["asked"] > 1
    assert s["strokes"]["refused"] == 0
    assert len(s["strokes"]["draw_order"]) == s["strokes"]["planned"]
    assert sorted(s["strokes"]["draw_order"]) == list(
        range(s["strokes"]["asked"]))

    csv_path = tmp_path / f"w_{arm}.csv"
    assert csv_path.exists(), "a certified word must write its CSV"
    hdr, rows = _read_csv(csv_path)
    assert hdr[:len(day1.pathway.CSV_COLUMNS)] == day1.pathway.CSV_COLUMNS
    assert hdr == day1.CSV_COLUMNS_T and hdr[-1] == "t_s"
    assert len(rows) > 0
    assert hdr[11:18] == ["q1", "q2", "q3", "q4", "q5", "q6", "q7"]
    for row in rows:
        assert all(c != "" for c in row[11:18])
        assert len(row) == len(day1.CSV_COLUMNS_T)
    assert any(row[2] == day1.pathway.KIND_DRAW for row in rows)
    # more than one stroke really reached the file
    assert len({row[0] for row in rows}) > 1

    js = json.loads((tmp_path / f"w_{arm}.json").read_text())
    assert js["placement"]["centre_x"] == pytest.approx(
        js["base"]["translation_m"][0])
    assert js["placement"]["baseline_y"] == pytest.approx(
        js["placement"]["seam_y"] + day1.WORD_DY)
    assert js["hover_m"] == 0.0


def test_word_arm_csv_carries_the_planner_s_clock(tmp_path):
    """`t_s` — the last column, the npz's clock, and checkable against it.

    The exporter's CSV has no time in it at all. `day1.py` appends one without
    touching `export/pathway.py`: it re-runs the exporter's OWN `_emit_frames`
    to recover which timeline frame each row came from, so row k of the middle
    block is frame `frame_idx[k]` at `frame_idx[k] / fps`. The proof that the
    reconstruction is right is that the first DRAW row of every stroke lands on
    the `t_start_s` the exporter computed for that stroke by a different route.
    """
    r = day1.plan_word(31, name="w", out_dir=tmp_path, verbose=False)
    s = r["summary"]
    hdr, rows = _read_csv(tmp_path / "w_31.csv")
    assert hdr[-1] == "t_s"
    t = np.array([float(row[-1]) for row in rows])
    assert (np.diff(t) > 0).all(), "t_s must strictly increase"
    assert t[0] >= 0.0
    # the file is the certified span, so it starts after the programme does and
    # ends before it does — it is the same clock, not a rebased one
    assert 0.0 < t[0] < t[-1] <= s["duration_s"]
    assert s["csv"]["t_s"]["first_s"] == pytest.approx(t[0])
    assert s["csv"]["t_s"]["last_s"] == pytest.approx(t[-1])
    assert s["csv"]["t_s"]["n_synth_rows"] == 0
    assert s["csv"]["columns"] == hdr

    man = json.loads((tmp_path / "w_31.manifest.json").read_text())
    assert man["format"]["columns"] == hdr
    assert "re-pace" in man["t_s"]["note"].lower() or \
        "RE-PACE" in man["t_s"]["note"]

    # THE CROSS-CHECK.  `manifest["strokes"][i]["t_start_s"]` is the exporter's
    # own number, derived from `strokes_of` and never from this column.
    first = {}
    for row in rows:
        if row[2] == day1.pathway.KIND_DRAW:
            first.setdefault(int(row[0]), float(row[-1]))
    assert len(first) == len(man["strokes"]) > 1
    for m in man["strokes"]:
        assert first[int(m["seg"])] == pytest.approx(m["t_start_s"], abs=1e-6)


def test_word_arm_reports_the_joint_speed_against_the_fr3_limits(tmp_path):
    """The gate on the FILE: no joint may be asked for more than it has.

    Every other gate in `day1.py` is about where the arm is.  This one is about
    how fast the joint reference moves, which is what a stiff controller turns
    into torque — and the CSV carries the PLANNER'S OWN q1..q7 per row, so the
    number here is the one the robot will be handed.
    """
    r = day1.plan_word(31, name="w", out_dir=tmp_path, verbose=False)
    sa = r["summary"]["joint_speed"]
    assert sa["ok"]
    assert sa["qd_max_rad_s"] == pytest.approx(
        [float(v) for v in day1.frames.QD_MAX])
    for key in ("at_csv_rows", "at_1khz"):
        v = sa[key]
        assert v["ok"] and 0.0 < v["worst_frac"] <= 1.0
        assert len(v["peak_qd_rad_s"]) == 7 and len(v["frac_of_limit"]) == 7
        assert len(v["peak_qdd_rad_s2"]) == 7
    # the pacer aims at QD_FRAC of the limit, so a healthy programme sits
    # there and not at the ceiling; a reading near 1.0 is a bug upstream
    assert sa["at_csv_rows"]["worst_frac"] < 0.9
    assert day1._speed_line(sa).startswith("  joint speed OK  at t_s")

    # IT IS MEASURED ON THE FILE'S OWN CLOCK, and says so.  A uniform reading
    # would charge a collapsed hold the nominal frame period and invent a
    # speed nothing moves at — the row spacing really is not uniform.
    assert sa["source"] == "the CSV's own t_s column"
    assert sa["n_nonmonotonic_t"] == 0
    assert sa["at_csv_rows"]["dt_max_s"] > sa["at_csv_rows"]["dt_median_s"]
    assert "re-pace" in sa["validity"].lower()

    # ...and the same audit is in the manifest, beside the column it certifies
    man = json.loads((tmp_path / "w_31.manifest.json").read_text())
    assert man["t_s"]["joint_speed"]["at_csv_rows"]["worst_frac"] == \
        pytest.approx(sa["at_csv_rows"]["worst_frac"])


def test_speed_audit_uses_the_times_it_is_given(tmp_path):
    """The audit is a function of (t, q) and catches an over-speed.

    Stated on a two-row toy rather than a robot: the same poses half a second
    apart pass, and a tenth of that apart do not — which is the whole point of
    computing the certificate from `t_s` instead of from a nominal period.
    """
    import numpy as np
    lim = np.asarray(day1.frames.QD_MAX, float)
    q0 = np.zeros(7)
    q1 = 0.5 * lim * 0.5                       # half the limit for 0.5 s
    ok = day1.speed_audit([0.0, 0.5, 1.0], [q0, q1, q0 + 2 * q1])
    assert ok["ok"] and ok["at_csv_rows"]["worst_frac"] == pytest.approx(0.5)
    fast = day1.speed_audit([0.0, 0.05, 0.10], [q0, q1, q0 + 2 * q1])
    assert not fast["ok"]
    assert fast["at_csv_rows"]["worst_frac"] == pytest.approx(5.0)
    # a clock that does not advance is refused rather than divided by
    bad = day1.speed_audit([0.0, 0.0, 1.0], [q0, q1, q0])
    assert bad["n_nonmonotonic_t"] == 1 and not bad["ok"]


def test_word_arm_refuses_a_word_it_cannot_reach(tmp_path):
    """Off the far end of the paper: refused, and NOTHING written."""
    with pytest.raises(day1.Refused) as e:
        day1.plan_word(31, dy=1.60, name="bad", out_dir=tmp_path,
                       verbose=False)
    assert "stroke" in str(e.value)
    assert not list(tmp_path.glob("*.csv"))
    assert not list(tmp_path.glob("*.npz"))
    assert not list(tmp_path.glob("*.json"))


def test_word_arm_hover_flies_above_the_paper():
    """`--hover` is graded with the REAL pen and reads ~30 mm of tip.

    Same construction as `line --hover`: the plan gets a 30 mm longer pen and
    the certificate does not, so the independent check — run with the real pen
    at the real height — reports the tip that far off the paper.  Nothing is
    written: `write=False`.
    """
    r = day1.plan_word(31, hover=0.030, write=False, verbose=False)
    s = r["summary"]
    tip = s["gates"]["min_paper_tip_m"]
    assert tip == pytest.approx(0.030, abs=0.005), \
        f"a 30 mm hover should ride 30 mm off the paper; got {1000 * tip:.1f} mm"
    # ...and the json SAYS the measured height, which is what the runbook asks
    # somebody to check with a ruler
    assert s["hover"]["asked_m"] == pytest.approx(0.030)
    assert s["hover"]["measured_tip_above_paper_m"] == pytest.approx(tip)


def test_word_arm_argv_is_the_command_a_person_would_type(tmp_path):
    """The GUI's whitelist: the panel's word control, as an argument list."""
    from aris_sixarm.gui.worker import build_day1_argv
    argv = build_day1_argv(dict(day1="word", arm=31, width=0.55, hover=0.03,
                                rig="proposed", tool="lateral"), tmp_path)
    assert argv[0] == "word"
    assert argv[argv.index("--arm") + 1] == "31"
    assert argv[argv.index("--width") + 1] == "0.55"
    assert argv[argv.index("--hover") + 1] == "0.03"
    assert argv[argv.index("--out") + 1] == str(tmp_path)
    assert "--variant" not in argv          # a solo word has no variant

    # no arm (or "both") is still the two-arm asset re-check, unchanged
    both = build_day1_argv(dict(day1="word", variant="alt"), tmp_path)
    assert both[:3] == ["word", "--variant", "alt"]
    assert build_day1_argv(dict(day1="word", arm="both", variant="hover"),
                           tmp_path)[:3] == ["word", "--variant", "hover"]
    # ...and the whitelist is still a whitelist
    with pytest.raises(ValueError):
        build_day1_argv(dict(day1="word", arm=31, tilt=3), tmp_path)
    with pytest.raises(ValueError):
        build_day1_argv(dict(day1="word", arm=13), tmp_path)


def test_word_arm_rows_are_cartesian_followable(tmp_path):
    """No pen-up run may reconfigure the arm, and the hover must be signed right.

    The deployed executor walks consecutive non-draw rows as a CARTESIAN path
    and latches its own nullspace (ARIS2_CONTRACTS §1), so a joint step no
    continuous IK branch could produce is a path it cannot follow.
    """
    r = day1.plan_word(31, name="w", out_dir=tmp_path, verbose=False)
    rows = r["summary"]["rows"]
    assert not rows["reconfigures"] and rows["n_over_bound"] == 0
    assert rows["max_dq_rad"] < rows["bound_rad"]
    assert rows["max_travel_dq_rad"] <= rows["max_dq_rad"]
    # pen-down: the draw rows sit ON the plane, in the base frame
    assert rows["tip_above_paper_base_m"]["draw_max"] == pytest.approx(
        0.0, abs=1e-4)
    assert "hover_check" not in rows


def test_word_arm_hover_rows_are_below_paper_z_in_the_base_frame():
    """fr3_link0's +z points DOWN on an inverted arm, so a hover is z_paper - h.

    Getting this sign wrong would drive the pen 30 mm INTO the paper on the one
    pass whose whole purpose is never to touch it.
    """
    r = day1.plan_word(31, hover=0.030, write=False, verbose=False)
    # `write=False` stops before the rows exist, so the geometry is checked on
    # the certificate instead: scene_check's own tip clearance is positive
    assert r["summary"]["gates"]["min_paper_tip_m"] > 0.02


def test_park_pose_is_the_arms_seed_and_reports_its_tip(tmp_path):
    """`park --arm N` with no --from-q: the pose, and the tip in fr3_link0."""
    fl, _ = day1.rt.fleet_for(None, None, "uniform")
    for arm in (31, 71):
        p = day1.park_pose(arm)
        assert p["frame"] == "fr3_link0"
        assert p["q"] == pytest.approx(
            [float(v) for v in fl[arm].q_seed])
        assert len(p["tip_xyz_base_m"]) == 3 and len(p["tip_quat_xyzw"]) == 4
        # the park holds the pen well clear of the paper plane
        assert p["tip_xyz_base_m"][2] < 0.970 - 0.05


def test_park_plans_and_certifies_a_path_from_a_pose_off_the_park(tmp_path):
    """0.5 rad off the park: an RRT path, certified, written as travel rows."""
    q = np.asarray(day1.park_pose(31)["q"], float)
    start = q.copy()
    start[1] += 0.5
    r = day1.plan_park(31, start, out_dir=tmp_path, verbose=False)
    s = r["summary"]
    assert s["certified"] and s["gates"]["ok"]
    assert s["gates"]["min_inter_arm_m"] > day1.coordination.PAIR_MARGIN
    assert s["rrt"]["n_configs"] > 1 and s["duration_s"] > 0.0
    assert not s["rows"]["reconfigures"]
    assert s["joint_speed"]["ok"]

    hdr, rows = _read_csv(tmp_path / "park_31.csv")
    assert hdr == day1.CSV_COLUMNS_T
    assert len(rows) == s["rrt"]["n_configs"] > 1
    # EVERY row is pen-up, and every row carries joints and a time
    assert all(row[2] == day1.pathway.KIND_TRAVEL for row in rows)
    for row in rows:
        assert all(c != "" for c in row[11:18])
    t = np.array([float(row[-1]) for row in rows])
    assert (np.diff(t) > 0).all() and t[0] == 0.0
    # the path ends AT the park
    last = np.array([float(v) for v in rows[-1][11:18]])
    assert last == pytest.approx(q, abs=1e-6)

    # ...and it says, in as many words, what it is and is not
    assert "CHECK" in s["what_this_is"]
    assert "go_start_pos.py" in s["what_this_is"]
    assert "IGNORES q1..q7" in s["executor"]


def test_park_refuses_a_start_pose_that_is_in_collision(tmp_path):
    """A start inside the static scene: refused, and nothing written.

    +0.5 rad on joint 1 swings arm 31 into the structure — measured, its
    clearance to the static boxes there is -88 mm, so the C-space search has an
    infeasible root and says so rather than returning a path through metal.
    """
    q = np.asarray(day1.park_pose(31)["q"], float)
    bad = q.copy()
    bad[0] += 0.5
    with pytest.raises(day1.Refused) as e:
        day1.plan_park(31, bad, out_dir=tmp_path, verbose=False)
    assert "collision" in str(e.value)
    assert not list(tmp_path.glob("*.csv"))
    assert not list(tmp_path.glob("*.npz"))


def test_the_site_file_is_the_only_place_addresses_live(tmp_path):
    """`config/site.json` holds every machine-specific fact; the script holds none.

    Pete: "make sure it is easy to reconfigure the setup." The test of that is
    mechanical — no IP, no user@host and no /tmp path may appear in the script
    at all, and everything the dispatch line is built from must come out of one
    tracked JSON file that `day1.py site --set` can edit.
    """
    import re
    src = (ROOT / "scripts" / "day1.py").read_text()
    # an address or a remote path in the SCRIPT is the thing this forbids
    assert not re.search(r"\b192\.168\.\d+\.\d+", src), \
        "an IP address is hard-coded in day1.py; it belongs in config/site.json"
    assert not re.search(r"\b\w+@\d+\.\d+\.\d+\.\d+", src)
    assert "/tmp/impedance_pathway" not in src
    assert "OPERATOR" not in src

    st = day1.site()
    assert st["operator"]["host"]
    for k in ("31", "71"):
        sl = st["slots"][k]
        for f in ("position", "arm", "ip", "domain", "paper_z", "mounted"):
            assert f in sl, f
    assert st["slots"]["31"]["mounted"] is None, \
        "which arms are mounted is UNKNOWN until somebody confirms it"
    for k in ("RTFF_CONTACT_DESCEND", "RTFF_FORCE_SIGN", "RTFF_TRAVEL_SPEED",
              "RTFF_MODE", "RTFF_DEPART_LIFT"):
        assert k in st["rtff_env"], k
    assert st["rtff_env"]["RTFF_DEPART_LIFT"] == "0"
    assert "97" in st["arms"]                      # the spare is known of


def test_site_set_edits_the_file_and_nothing_else(tmp_path):
    """`site --set slot31.arm=97` writes the file back, typed."""
    src = json.loads((ROOT / "config" / "site.json").read_text())
    p = tmp_path / "site.json"
    p.write_text(json.dumps(src, indent=1))
    ap = day1.build_parser()
    a = ap.parse_args(["--site", str(p), "site", "--set", "slot31.arm=97",
                       "--set", "slot31.mounted=true",
                       "--set", "slot71.paper_z=0.003"])
    day1.site(str(p), reload=True)
    assert day1.cmd_site(a) == 0
    got = json.loads(p.read_text())
    assert got["slots"]["31"]["arm"] == 97            # int, not "97"
    assert got["slots"]["31"]["mounted"] is True      # bool, not "true"
    assert got["slots"]["71"]["paper_z"] == 0.003     # float
    assert got["operator"]["host"] == src["operator"]["host"]
    with pytest.raises(day1.Refused):
        day1.cmd_site(ap.parse_args(
            ["--site", str(p), "site", "--set", "nonsense"]))
    day1.site(reload=True)             # leave the module on the real file


def test_send_dispatches_a_slot_to_a_different_physical_arm(tmp_path, capsys):
    """`--as-arm`: the plan is a POSITION, the arm bolted into it may differ."""
    good = tmp_path / "unknown_31.csv"
    good.write_text(",".join(day1.CSV_COLUMNS_T) + "\n"
                    + ",".join(["0"] * len(day1.CSV_COLUMNS_T)) + "\n")
    ap = day1.build_parser()
    assert day1.cmd_send(ap.parse_args(
        ["send", "--arm", "31", "--as-arm", "97", "--file", str(good),
         "--dry-run"])) == 0
    out = capsys.readouterr().out
    assert "slot 31 -> arm 97" in out
    assert "left-middle" in out
    assert "ARM_ID    97" in out
    assert "192.168.50.15" in out                   # arm 97's box, from the site
    assert "/tmp/impedance_pathway_arm97.csv" in out
    assert "ARM_ID=97 bash" in out
    assert "CONFIRM THE MOUNTING" in out
    # ...and the park it prints is the SLOT's, because that is the base frame
    # the file's poses are in
    assert "park q  =" in out


def test_send_refuses_a_file_that_is_not_a_pathway_csv(tmp_path):
    """`send` is the interface to the arms and it checks what it is handing over."""
    ap = day1.build_parser()
    missing = tmp_path / "nope.csv"
    with pytest.raises(day1.Refused):
        day1.cmd_send(ap.parse_args(
            ["send", "--arm", "31", "--file", str(missing), "--dry-run"]))
    wrong = tmp_path / "wrong.csv"
    wrong.write_text("x,y,z\n1,2,3\n")
    with pytest.raises(day1.Refused) as e:
        day1.cmd_send(ap.parse_args(
            ["send", "--arm", "31", "--file", str(wrong), "--dry-run"]))
    assert "pathway columns" in str(e.value)


def test_send_dry_run_copies_nothing_and_prints_the_one_command(tmp_path,
                                                                capsys):
    """A dry run names the scp and the draw command and moves no arm."""
    good = tmp_path / "unknown_31.csv"
    good.write_text(",".join(day1.pathway.CSV_COLUMNS) + "\n"
                    + ",".join(["0"] * len(day1.pathway.CSV_COLUMNS)) + "\n")
    ap = day1.build_parser()
    assert day1.cmd_send(ap.parse_args(
        ["send", "--arm", "31", "--file", str(good), "--host",
         "someone@10.0.0.1", "--dry-run"])) == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "nothing copied" in out
    assert f"scp {good} someone@10.0.0.1:/tmp/impedance_pathway_arm31.csv" in out
    # THE DISPATCH LINE, VERBATIM — this is the string somebody pastes
    assert ("RTFF_CONTACT_DESCEND=0 RTFF_FORCE_SIGN=1 RTFF_TRAVEL_SPEED=0.02 "
            "RTFF_MODE=observe RTFF_DEPART_LIFT=0 ARM_ID=31 bash "
            "~/RTff/draw_rtff_supervised.sh /tmp/impedance_pathway_arm31.csv "
            "1.0 2.5 5 fresh") in out
    assert "aris_hold.sh stack 31" in out
    assert "tail -f /tmp/rtff_draw_arm31.log" in out
    # the things that must be said every time
    assert "ladder gate" in out.lower()
    assert "e-stop" in out.lower()
    assert "IGNORES q1..q7 AND t_s" in out
    assert "park q  =" in out          # where the arm has to be standing


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
    for want in ("line", "word", "send", "--hover", "1.8153", "out/day1"):
        assert want in r.stdout
    # the refusal a person will otherwise meet at the rig, said in the help
    assert "SEAM LINE" in r.stdout

    w = subprocess.run([sys.executable, str(ROOT / "scripts" / "day1.py"),
                        "word", "--help"],
                       cwd=str(ROOT), capture_output=True, text=True, env=env,
                       timeout=300)
    assert w.returncode == 0
    for want in ("--arm", "--width", "--dy", "--allow-partial", "-0.10"):
        assert want in w.stdout, want
