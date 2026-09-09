"""The two commissioning scripts, on synthetic input.  NO ROBOT IS TOUCHED.

`asbuilt_layout.py` reads a survey and hands the planner a fleet; this pins that
a survey it has not been given is refused, that a deviation is REPORTED rather
than absorbed, and that the two things the shipped layout code cannot express —
a per-arm height and a per-arm yaw — actually reach the base transform.

`touchdown_calibrate.py` solves the pen tip; this pins the round trip, the
sensitivity to touchdown noise, and — the important one — that a set of
touchdowns which cannot separate the lateral offset from the axial one is
REFUSED instead of answered.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import asbuilt_layout as ab                                       # noqa: E402
import touchdown_calibrate as tc                                  # noqa: E402
from aris_sixarm import frames                                    # noqa: E402
from aris_sixarm.fleet import FLEET                               # noqa: E402


# ---------------------------------------------------------------------------
# asbuilt_layout
# ---------------------------------------------------------------------------
def _survey(moves=None):
    """The template with `moves` = {arm: {field: value}} written in."""
    s = ab.template()
    for v in s["arms"].values():
        v["measured"] = True
    for a, fields in (moves or {}).items():
        s["arms"][str(a)].update(fields)
    return s


def test_the_nominal_is_derived_from_the_layout_constant_not_typed():
    from aris_sixarm import layout
    nom = ab.build_sheet_nominal()
    assert set(nom) == set(layout.build_fleet(layout.LAYOUT_PROPOSED))
    for a, v in nom.items():
        assert v["z"] == pytest.approx(layout.LAYOUT_PROPOSED["h"])
    xs = sorted({round(v["x"], 4) for v in nom.values()})
    assert len(xs) == 2 and xs[1] - xs[0] == pytest.approx(
        layout.PAIR_SPACING, abs=1e-9)


def test_an_unmeasured_survey_is_not_within_tolerance():
    s = ab.template()                    # every arm still `measured: false`
    lines, _, ok = ab.deviations(s)
    assert not ok
    assert any("NOT MEASURED" in ln for ln in lines)


def test_a_survey_at_the_nominal_passes():
    lines, dev, ok = ab.deviations(_survey())
    assert ok, lines
    assert all(abs(v) < 1e-12 for d in dev.values() for v in d.values())


def test_a_base_out_of_tolerance_is_flagged_and_not_absorbed():
    s = _survey({31: {"x": ab.build_sheet_nominal()[31]["x"] + 0.025}})
    lines, dev, ok = ab.deviations(s)
    assert not ok
    assert any("arm  31" in ln and "xy" in ln for ln in lines)
    assert dev[31]["x"] == pytest.approx(0.025)
    # the OTHER arms are untouched: the build sheet forbids re-centring them
    assert all(abs(dev[a]["x"]) < 1e-12 for a in dev if a != 31)


def test_coplanarity_is_its_own_gate():
    nom = ab.build_sheet_nominal()
    s = _survey({2: {"z": nom[2]["z"] + 0.008}})     # inside the +-10 mm...
    lines, _, ok = ab.deviations(s)
    assert not ok                                      # ...but not coplanar
    assert any("coplanarity" in ln and "OUT" in ln for ln in lines)


def test_per_arm_height_and_yaw_reach_the_base_transform():
    nom = ab.build_sheet_nominal()
    s = _survey({31: {"z": nom[31]["z"] - 0.09, "yaw_deg": 2.5}})
    fleet, meta = ab.build_asbuilt(s)
    T = fleet[31].T_world_base()
    assert T[2, 3] == pytest.approx(nom[31]["z"] - 0.09)
    # yaw 2.5 deg about the base z shows up in the rotation.  The inverted
    # base is Ry(pi) @ Rz(yaw) = [[-c, s, 0], [s, c, 0], [0, 0, -1]], so the
    # yaw reads off as atan2(R10, -R00) and NOT as atan2(R10, R00) — which
    # would say 177.5 deg and look like a bug in the survey rather than in the
    # reader.
    R = T[:3, :3]
    assert np.degrees(np.arctan2(R[1, 0], -R[0, 0])) == pytest.approx(2.5,
                                                                      abs=1e-6)
    assert R[2, 2] == pytest.approx(-1.0)          # still hanging
    # ...and the other five did NOT move
    for a in fleet:
        if a != 31:
            assert fleet[a].T_world_base()[2, 3] == pytest.approx(nom[a]["z"])
    assert meta["h_min"] < meta["h_max"]


def test_the_as_built_height_is_the_one_the_hardware_was_built_at():
    """0.850 is what stands in the room; 0.940 is what ships in layout.py."""
    from aris_sixarm import layout
    assert layout.LAYOUT_PROPOSED["h"] == 0.940
    s = _survey({a: {"z": 0.850} for a in ab.build_sheet_nominal()})
    fleet, meta = ab.build_asbuilt(s)
    assert meta["h_mean"] == pytest.approx(0.850)
    assert all(f.T_world_base()[2, 3] == pytest.approx(0.850)
               for f in fleet.values())
    _, _, ok = ab.deviations(s)
    assert not ok            # 90 mm is a deviation and must be reported as one


def test_the_survey_allowance_is_what_the_survey_buys():
    """A surveyed rig may carry a thinner column than an unsurveyed one."""
    from aris_sixarm import mounts
    s = _survey()
    fat, _ = ab.build_asbuilt(s)                       # calib = MOUNTS.calib
    thin, meta = ab.build_asbuilt(s, calib=0.005)
    assert meta["calib"] == pytest.approx(0.005)
    assert mounts.MOUNTS.calib == pytest.approx(0.03)  # unchanged on disk
    assert len(thin[31].mount_boxes) == len(fat[31].mount_boxes)
    vol = lambda boxes: sum(float(np.prod(np.asarray(b["hi"], float)
                                          - np.asarray(b["lo"], float)))
                            for b in boxes)
    assert vol(thin[31].mount_boxes) < vol(fat[31].mount_boxes)


def test_the_document_round_trips_and_refuses_a_half_edited_file(tmp_path):
    s = _survey({71: {"z": 0.851, "yaw_deg": 0.4}})
    doc = ab.to_document(s)
    p = tmp_path / "asbuilt.json"
    p.write_text(json.dumps(doc))
    fleet, back = ab.load_asbuilt(p)
    assert back["kind"] == "aris_sixarm.asbuilt_layout"
    assert fleet[71].T_world_base()[2, 3] == pytest.approx(0.851)

    doc["bases"]["71"][2][3] = 0.5           # edit one half only
    p.write_text(json.dumps(doc))
    with pytest.raises(SystemExit, match="one half"):
        ab.load_asbuilt(p)

    (tmp_path / "other.json").write_text(json.dumps({"kind": "something else"}))
    with pytest.raises(SystemExit, match="not an as-built"):
        ab.load_asbuilt(tmp_path / "other.json")


# ---------------------------------------------------------------------------
# touchdown_calibrate
# ---------------------------------------------------------------------------
def test_a_noise_free_round_trip_returns_the_tip_exactly():
    tip = (frames.PEN_LAT_HOLDER, 0.0, frames.PEN_EXT_HOLDER)
    fit = tc.solve(tc.synthetic(FLEET, tip=tip, n=10), FLEET)
    assert fit["well_posed"]
    assert fit["pen_lat_m"] == pytest.approx(tip[0], abs=1e-9)
    assert fit["pen_ext_m"] == pytest.approx(tip[2], abs=1e-9)
    assert fit["off_jaw_plane_m"] == pytest.approx(0.0, abs=1e-9)
    assert fit["resid_max_m"] < 1e-12


def test_it_recovers_a_tip_that_is_NOT_the_shipped_one():
    """The case that matters: the photograph was wrong and the fit says so."""
    tip = (0.0805, 0.0021, 0.0512)          # 5.5 mm out, 5.9 mm deeper
    fit = tc.solve(tc.synthetic(FLEET, tip=tip, n=12, seed=3), FLEET)
    assert fit["pen_lat_m"] == pytest.approx(tip[0], abs=1e-9)
    assert fit["pen_ext_m"] == pytest.approx(tip[2], abs=1e-9)
    assert fit["off_jaw_plane_m"] == pytest.approx(tip[1], abs=1e-9)
    assert fit["d_ext_m"] == pytest.approx(tip[2] - frames.PEN_EXT_HOLDER,
                                           abs=1e-9)


def test_a_deeper_tip_eats_the_v18_paper_chain_margin():
    """docs/DECISIONS.md: v18 holds 20.5 mm against a 20 mm gate."""
    shallow = tc.solve(tc.synthetic(
        FLEET, tip=(0.086, 0.0, frames.PEN_EXT_HOLDER + 0.0002), n=10), FLEET)
    assert shallow["chain_ok"]                      # 0.2 mm of the 0.5 eaten
    deep = tc.solve(tc.synthetic(
        FLEET, tip=(0.086, 0.0, frames.PEN_EXT_HOLDER + 0.002), n=10), FLEET)
    assert not deep["chain_ok"]
    assert deep["chain_clearance_after_m"] < tc.CHAIN_GATE_M
    assert any("RE-CONDUCTED" in ln for ln in tc.report(deep))


def test_touchdown_noise_maps_to_tip_error_at_about_one_to_one():
    """What a millimetre of contact-detection error costs the tip."""
    tip = (frames.PEN_LAT_HOLDER, 0.0, frames.PEN_EXT_HOLDER)
    errs = []
    for seed in range(6):
        fit = tc.solve(tc.synthetic(FLEET, tip=tip, n=10, noise_m=0.001,
                                    seed=seed), FLEET)
        errs.append(np.abs(np.array(fit["tip_hand_tcp"]) - tip).max())
    assert 1e-4 < float(np.mean(errs)) < 5e-3        # same order, not amplified


def test_too_few_touchdowns_is_refused_rather_than_answered():
    with pytest.raises(SystemExit, match="three unknowns"):
        tc.solve(tc.synthetic(FLEET, n=2), FLEET)


def test_touchdowns_that_cannot_separate_the_offsets_are_refused():
    """Ten touchdowns at ONE tool orientation are one equation, not ten."""
    recs = tc.synthetic(FLEET, n=1, seed=11)
    q = np.asarray(recs[0]["q"], float)
    same = []
    for i in range(10):
        # move only joint 1: the tool's ORIENTATION relative to the paper
        # normal is unchanged for a base whose z axis is vertical, so every
        # row of A is (nearly) the same direction
        qq = q.copy()
        qq[0] += 0.05 * i
        T_wb = FLEET[31].T_world_base()
        T, _ = frames.fk(qq)
        p = np.array([frames.PEN_LAT_HOLDER, 0.0, frames.PEN_EXT_HOLDER])
        z = float((T_wb[:3, :3] @ (T[:3, :3] @ p + T[:3, 3]) + T_wb[:3, 3])[2])
        same.append(dict(arm=31, q=[float(x) for x in qq], paper_z=z,
                         label=f"joint-1 only {i}"))
    fit = tc.solve(same, FLEET)
    assert not fit["well_posed"], fit["cond"]
    assert any("ILL-POSED" in ln for ln in tc.report(fit))
    assert any("TOOL YAWS" in ln for ln in tc.report(fit))


def test_the_self_test_runs_and_reports():
    assert tc.main(["--self-test"]) == 0
