"""The typed programme record: strict on the way in, lossless on the way out.

NO ENVIRONMENT VARIABLES HERE.  `ARIS_RIG` and `ARIS_TOOL` are read once, at
`import aris_sixarm`, so a test that set them would either do nothing (the
package is already imported) or change the rig for every test that runs after
it in the same process.  These tests use whatever fleet is active and build
their fixture around it, which is also the honest thing: the schema must not
care which rig it is describing.
"""
import json

import numpy as np
import pytest

from aris_sixarm import frames
from aris_sixarm.fleet import FLEET, SHEET
from aris_sixarm.program_schema import (Bundle, BufferTable, Meta, Segment,
                                        SchemaError, export_bundle,
                                        export_scene, _from_dict)


# --------------------------------------------------------------------------
# a tiny synthetic programme, in the shape `csail_schedule.payload` writes
# --------------------------------------------------------------------------
def _fake_run(tmp_path, n_frames=12, n_arms=2):
    arms = sorted(FLEET)[:n_arms]
    fleet_arms = sorted(FLEET)
    F = n_frames
    d = dict(fps=np.float64(24.0), dt=np.float64(1 / 48.0),
             stride=np.int64(2), n_phases=np.int64(1),
             pause_s=np.float64(0.0), margin=np.float64(0.08),
             min_clearance=np.float64(0.113), pause_total=np.float64(0.0),
             arms=np.array(arms, np.int64),
             drawing_arms=np.array(arms, np.int64),
             pen_ext=np.array([0.110] * len(fleet_arms), float),
             sheet=np.array(SHEET, float),
             n_frames=np.int64(F), duration=np.float64((F - 1) / 24.0),
             phase=np.zeros(F, np.int64),
             phase_start_s=np.array([0.0]), phase_ink=np.array(["grey"]),
             ink_names=np.array(["grey"]), ink_palette=np.array(["#666665"]))
    rng = np.random.default_rng(3)
    for a in arms:
        q = np.tile(np.asarray(FLEET[a].q_seed, np.float32), (F, 1))
        q += 0.01 * rng.standard_normal((F, 7)).astype(np.float32)
        d[f"q_{a}"] = q
        d[f"seg_{a}"] = np.where(np.arange(F) < F // 2, 0, -1).astype(np.int64)
        d[f"u_{a}"] = np.linspace(0, 1, F)
        d[f"segpts_{a}"] = np.column_stack([np.linspace(0.2, 0.6, 8),
                                            np.linspace(0.3, 0.9, 8)])
        d[f"segoff_{a}"] = np.array([0, 8], np.int64)
    d["ink_t"] = np.array([0.5, 1.0])
    d["ink_arm"] = np.array([arms[0], arms[-1]], np.int64)
    d["ink_off"] = np.array([0, 3, 6], np.int64)
    d["ink_xyz"] = np.zeros((6, 3))
    d["ink_hex"] = np.array(["#666665", "#666665"])
    npz = tmp_path / "t_schedule.npz"
    np.savez_compressed(npz, **d)

    pts = np.column_stack([np.linspace(0.2, 0.6, 8),
                           np.linspace(0.3, 0.9, 8)]).round(5).tolist()
    seg = lambda a: dict(                                   # noqa: E731
        stroke_id=0, seg=0, color="grey", kind="outline", s_range=[0.0, 1.0],
        direction=1, flipped=False, length_m=0.4, home_before=False,
        n_points=8, plan_ok=True, validated=True, pen_ext_m=0.110,
        n_dense=80, n_knots=6, min_sigma=0.24, min_margin=0.51,
        tip_err_m=1e-9, max_lean_deg=7.5, tilt_cone_deg=15.0,
        draw_time_s=3.3, pts=pts, q_first=[0.0] * 7, q_last=[0.0] * 7)
    prog = dict(
        rig="test", arms_in_fleet=fleet_arms, sheet=list(SHEET), h_inv=0.94,
        n_phases=1, two_pass=False, pens_mm={str(a): 110.0 for a in arms},
        name="tiny", source="synthetic", inks=["grey"],
        palette={"grey": "#666665"},
        logo=dict(logo_w=0.4, logo_h=0.6, rotate_deg=0.0),
        totals=dict(traced_m=0.4, drawn_m=0.4, dropped_m=0.0, dropped_pct=0.0,
                    coverage_pct=100.0, n_strokes=1, n_segments=len(arms),
                    n_probes=4, wall_s=12.5, transit_s=1.0,
                    baseline_transit_s=2.0),
        strokes=[dict(id=0, color="grey", kind="outline", length=0.4)],
        sequencer="opt",
        phases=[dict(name="single pass", ink="grey",
                     arms={str(a): [seg(a)] for a in arms},
                     dropped=[], wall_s=12.5)])
    pj = tmp_path / "t_program.json"
    pj.write_text(json.dumps(prog))

    strokes = dict(sheet=list(SHEET), palette={"grey": "#666665"},
                   info={}, n_strokes=1, total_length=0.4,
                   strokes=[dict(id=0, color="grey", kind="outline",
                                 length=0.4, pts=pts)])
    (tmp_path / "t_strokes.json").write_text(json.dumps(strokes))

    summ = dict(makespan_s=0.46,
                phases=[dict(name="single pass", ink="grey", duration_s=0.46,
                             floor_s=0.40, scene_check_ok=True,
                             min_clearance=0.113,
                             per_pair={f"{arms[0]}-{arms[-1]}": 0.113},
                             arm_metres={str(a): 0.2 for a in arms},
                             arm_segments={str(a): 1 for a in arms},
                             arm_draw_s={str(a): 1.6 for a in arms},
                             arm_transit_s={str(a): 0.3 for a in arms},
                             paper_failed=[], frame_failed=[],
                             column_failed=[])])
    sj = tmp_path / "t_schedule.json"
    sj.write_text(json.dumps(summ))
    return npz, pj, sj, arms


# --------------------------------------------------------------------------
# strictness
# --------------------------------------------------------------------------
def test_unknown_key_raises_and_names_itself():
    from aris_sixarm.program_schema import BufferRef
    good = dict(offset=0, length=8, dtype="f4", shape=[2])
    assert _from_dict(BufferRef, good, "BufferRef").dtype == "f4"
    with pytest.raises(SchemaError) as e:
        _from_dict(BufferRef, dict(good, tilt=3), "BufferRef")
    assert "tilt" in str(e.value)
    assert "BufferRef" in str(e.value)


def test_missing_key_raises():
    from aris_sixarm.program_schema import BufferRef
    with pytest.raises(SchemaError) as e:
        _from_dict(BufferRef, dict(offset=0, length=8, dtype="f4"), "BufferRef")
    assert "shape" in str(e.value)


def test_the_lean_has_exactly_one_name():
    """No `tilt`, no `lean_vec`, no `max_lean_deg`, no `tilt_max_deg`."""
    names = {f for f in Segment.__dataclass_fields__}
    assert "lean_deg" in names and "cone_deg" in names
    assert not (names & {"tilt", "lean_vec", "max_lean_deg", "tilt_max_deg",
                         "lean", "tilt_deg"})


def test_buffer_table_is_eight_byte_aligned():
    """A misaligned offset makes `new Float64Array(buf, off, n)` throw in JS."""
    t = BufferTable()
    refs = [t.add(np.arange(n), "f4" if n % 2 else "f8") for n in range(1, 20)]
    for r in refs:
        assert r.offset % 8 == 0
    assert len(t.blob()) >= refs[-1].offset + refs[-1].length


# --------------------------------------------------------------------------
# the export
# --------------------------------------------------------------------------
def test_export_bundle_round_trips(tmp_path):
    npz, pj, sj, arms = _fake_run(tmp_path)
    out = tmp_path / "bundle.json"
    b = export_bundle(npz, pj, out, summary=sj)

    assert out.exists() and out.with_suffix(".bin").exists()
    assert isinstance(b.meta, Meta)
    assert b.meta.schema_version == 1
    assert b.meta.n_frames == 12
    assert [a.arm for a in b.arms] == arms
    assert len(b.segments) == len(arms)
    assert len(b.strokes) == 1

    b2 = Bundle.from_json(out.read_text())
    assert b2.meta.name == b.meta.name
    assert [s.arm for s in b2.segments] == [s.arm for s in b.segments]
    assert b2.segments[0].lean_deg == 7.5
    assert b2.segments[0].cone_deg == 15.0


def test_export_resolves_the_lean_from_the_program_spelling(tmp_path):
    """`max_lean_deg` on disk becomes `lean_deg` in the schema, once."""
    npz, pj, sj, arms = _fake_run(tmp_path)
    b = export_bundle(npz, pj, tmp_path / "b.json", summary=sj)
    assert all(s.lean_deg == 7.5 for s in b.segments)
    raw = json.loads((tmp_path / "b.json").read_text())
    assert "max_lean_deg" not in json.dumps(raw["segments"][0])


def test_bundle_json_refuses_an_unknown_key(tmp_path):
    npz, pj, sj, _ = _fake_run(tmp_path)
    export_bundle(npz, pj, tmp_path / "b.json", summary=sj)
    doc = json.loads((tmp_path / "b.json").read_text())
    doc["meta"]["tilt"] = 3.0
    with pytest.raises(SchemaError) as e:
        Bundle.from_dict(doc)
    assert "tilt" in str(e.value)

    doc2 = json.loads((tmp_path / "b.json").read_text())
    doc2["extra_top_level"] = 1
    with pytest.raises(SchemaError):
        Bundle.from_dict(doc2)


def test_buffers_are_readable_at_the_offsets_they_declare(tmp_path):
    """The invariant a browser depends on: a view, not a copy."""
    npz, pj, sj, arms = _fake_run(tmp_path)
    b = export_bundle(npz, pj, tmp_path / "b.json", summary=sj)
    blob = (tmp_path / "b.bin").read_bytes()
    a = b.arms[0]
    q = np.frombuffer(blob, np.float32, count=a.q.length // 4,
                      offset=a.q.offset).reshape(a.q.shape)
    z = np.load(npz)
    assert q.shape == (12, 7)
    np.testing.assert_allclose(q, z[f"q_{a.arm}"], rtol=0, atol=0)
    assert a.q.offset % 8 == 0


def test_segment_indices_match_the_timeline(tmp_path):
    """`Segment.index` must name the same segment `seg_<arm>` does."""
    npz, pj, sj, arms = _fake_run(tmp_path)
    b = export_bundle(npz, pj, tmp_path / "b.json", summary=sj)
    z = np.load(npz)
    for s in b.segments:
        assert s.index in set(np.unique(z[f"seg_{s.arm}"]).tolist())


def test_clearance_series_is_per_frame_and_pairwise(tmp_path):
    npz, pj, sj, arms = _fake_run(tmp_path)
    b = export_bundle(npz, pj, tmp_path / "b.json", summary=sj)
    if len(arms) >= 2:
        assert len(b.clearance.pairs) >= 1
        d = b.clearance.pairs[0]["d"]
        assert d.shape == [12]
    assert b.clearance.margin_m == pytest.approx(0.08)
    assert set(b.clearance.self_d) == {str(a) for a in arms}


def test_export_without_a_summary_still_works(tmp_path):
    npz, pj, _, _ = _fake_run(tmp_path)
    b = export_bundle(npz, pj, tmp_path / "b.json")
    assert b.phases == []
    assert b.meta.n_frames == 12


# --------------------------------------------------------------------------
# the scene
# --------------------------------------------------------------------------
def test_export_scene_describes_the_active_rig(tmp_path):
    out = tmp_path / "scene.json"
    doc = export_scene(out)
    assert out.exists() and out.with_suffix(".bin").exists()
    assert [a["arm"] for a in doc["arms"]] == sorted(FLEET)
    assert doc["tool"] == ("lateral" if frames.PEN_LAT else "inline")
    assert doc["pen_lat_m"] == pytest.approx(frames.PEN_LAT)
    # every mesh the viewer will pose has an FK index to pose it by
    for name in doc["meshes"]:
        assert name in doc["link_index"] or name in doc["finger_T"]
    assert doc["link_index"]["panda_hand"] == 9
    assert len(doc["dh"]) == 7


def test_scene_carries_a_golden_fk_check(tmp_path):
    """The viewer's own FK is checked against these on every load."""
    out = tmp_path / "scene.json"
    doc = export_scene(out)
    blob = out.with_suffix(".bin").read_bytes()
    ref = doc["fk_check"]["T"]
    T = np.frombuffer(blob, np.float64, count=ref["length"] // 8,
                      offset=ref["offset"]).reshape(ref["shape"])
    qs = np.asarray(doc["fk_check"]["q"], float)
    assert T.shape == (len(qs), 10, 4, 4)
    np.testing.assert_allclose(T, frames.link_frames_many(qs), atol=0, rtol=0)


def test_scene_radii_index_into_the_chain_it_describes(tmp_path):
    doc = export_scene(tmp_path / "scene.json")
    width = 11 if doc["pen_lat_m"] else 10
    for row in doc["radii"]:
        assert len(row) in (3, 5)
        assert 0 <= row[0] < width and 0 <= row[1] < width
        assert row[2] > 0
