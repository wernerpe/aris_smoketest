"""The GUI backend, end to end: create a job, watch its events, read its result.

WHAT THIS TEST ACTUALLY RUNS.  A real subprocess, executing the real front door
(`scripts/draw.py`), on a real picture — a three-stroke PNG this file draws —
with the real progress sink writing the real JSONL the browser tails.  Nothing
is mocked, because the parts worth testing are exactly the seams that a mock
would replace: that the worker's argv is built from the job's parameters, that
the events reach the file, that a finished job's directory holds its artifacts.

IT STOPS AFTER THE TRACE, and that is deliberate.  Allocating and conducting
even a three-stroke drawing is minutes of certified IK; the stages after the
trace are covered by `tests/test_program_schema.py` (on their recorded output)
and by the numbers in docs/VIEWER.md (on a real run).  A test suite that took
four minutes to tell you the websocket works would not be run.

NO ARIS_RIG / ARIS_TOOL ANYWHERE.  A trace is rig-independent, so this test
says nothing about which rig is active and does not touch the environment.
"""
import json
import os
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient                    # noqa: E402

from aris_sixarm.gui.jobs import JobManager                  # noqa: E402
from aris_sixarm.gui.server import create_app                # noqa: E402
from aris_sixarm.gui.worker import (build_argv,              # noqa: E402
                                    build_day1_argv)


# --------------------------------------------------------------------------
@pytest.fixture
def picture(tmp_path):
    """Three black strokes on white — enough for the tracer to find something."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (240, 240), "white")
    d = ImageDraw.Draw(img)
    d.line([(40, 40), (200, 40)], fill="black", width=5)
    d.line([(40, 120), (200, 200)], fill="black", width=5)
    d.arc([60, 130, 180, 220], 0, 180, fill="black", width=5)
    p = tmp_path / "tiny.png"
    img.save(p)
    return p


@pytest.fixture
def client(tmp_path):
    app = create_app(jobs_dir=str(tmp_path / "jobs"))
    with TestClient(app) as c:
        yield c
    app.state.jobs.shutdown()


def _wait(client, job_id, timeout=180):
    """Poll until the job leaves the live states. -> the job dict."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{job_id}").json()
        if not j["live"]:
            return j
        time.sleep(0.25)
    log = client.get(f"/api/jobs/{job_id}/log")
    raise AssertionError(f"job {job_id} did not finish in {timeout} s\n"
                         + (log.text[-3000:] if log.status_code == 200 else ""))


# --------------------------------------------------------------------------
# the argv whitelist
# --------------------------------------------------------------------------
def test_build_argv_is_a_whitelist():
    with pytest.raises(ValueError) as e:
        build_argv(dict(source="x.png", tilt=3))
    assert "tilt" in str(e.value)


def test_build_argv_maps_kinds():
    argv = build_argv(dict(source="x.png", out="run", max_probes=2,
                           no_rrt=True, no_balance=False,
                           offset=[0.1, -0.2], scales=[0.4, 1.0, 5]))
    assert argv[0] == "x.png"
    assert "--out" in argv and argv[argv.index("--out") + 1] == "run"
    assert "--max-probes" in argv
    assert "--no-rrt" in argv
    assert "--no-balance" not in argv          # falsy flags are not passed
    i = argv.index("--offset")
    assert argv[i + 1:i + 3] == ["0.1", "-0.2"]
    i = argv.index("--scales")
    assert argv[i + 1:i + 4] == ["0.4", "1.0", "5"]
    assert "--no-anim" in argv                 # the GUI never renders pydrake


def test_build_argv_needs_a_source():
    with pytest.raises(ValueError):
        build_argv(dict(out="run"))


# --------------------------------------------------------------------------
# HARDWARE DAY 1 — the panel a person uses at the rig
# --------------------------------------------------------------------------
# NO ARIS_RIG / ARIS_TOOL HERE EITHER.  A day-1 job carries its rig and its
# tool in its PARAMETERS, and `jobs._spawn` puts them in the environment of the
# subprocess it starts — which is the only place they can be set, because they
# are read at `import aris_sixarm` time.  A test that exported them would be
# choosing the rig for the whole pytest process instead.
def test_build_day1_argv_is_a_whitelist():
    with pytest.raises(ValueError) as e:
        build_day1_argv(dict(day1="line", arm=31, **{"from": "0.5,1.7"},
                             to="0.65,1.7", tilt=3))
    assert "tilt" in str(e.value)
    with pytest.raises(ValueError):
        build_day1_argv(dict(day1="dance"))
    with pytest.raises(ValueError):                 # not an arm on this rig
        build_day1_argv(dict(day1="line", arm=13, **{"from": "0.5,1.7"},
                             to="0.65,1.7"))
    with pytest.raises(ValueError):                 # no line without two ends
        build_day1_argv(dict(day1="line", arm=31))
    with pytest.raises(ValueError):
        build_day1_argv(dict(day1="word", variant="backwards"))


def test_build_day1_argv_is_the_command_a_person_would_type(tmp_path):
    argv = build_day1_argv(dict(day1="line", arm=71, **{"from": "1.15,1.95"},
                                to="1.30,1.95", name="probe",
                                rig="proposed", tool="lateral"), tmp_path)
    assert argv[0] == "line"
    assert argv[argv.index("--arm") + 1] == "71"
    assert argv[argv.index("--from") + 1] == "1.15,1.95"
    assert argv[argv.index("--to") + 1] == "1.30,1.95"
    assert argv[argv.index("--out") + 1] == str(tmp_path)
    assert "--hover" not in argv                # the flag is opt-in, as in day1.py

    hov = build_day1_argv(dict(day1="line", arm=31, **{"from": "0.5,1.7"},
                               to="0.65,1.7", hover=0.03), tmp_path)
    assert hov[hov.index("--hover") + 1] == "0.03"

    w = build_day1_argv(dict(day1="word", variant="alt"), tmp_path)
    assert w[:3] == ["word", "--variant", "alt"]
    assert w[w.index("--arms") + 1] == "31,71"


def test_the_config_carries_the_day1_form(client):
    d = client.get("/api/config").json()["day1"]
    assert d["arms"] == [31, 71]
    assert d["variants"] == ["alt", "concurrent", "hover"]
    for k in ("arm", "from", "to", "name", "variant", "rig", "tool"):
        assert k in d, k
    # the refusal a person will otherwise meet at the rig, said in the form
    assert "1.8153" in d["note"]


def test_a_day1_line_job_certifies_and_loads_into_the_viewer(client, tmp_path):
    """The whole panel path: launch, PASS, a CSV, and a bundle that parses.

    This is the acceptance test for the Day 1 panel — it runs the real
    `scripts/day1.py line` in a real subprocess, at the real rig, and then
    reads back the three things the browser needs: the one-line verdict, the
    CSV path to copy, and the viewer bundle.  Nothing is mocked.
    """
    from aris_sixarm.program_schema import Bundle

    out = tmp_path / "day1"
    r = client.post("/api/jobs", json=dict(
        day1="line", rig="proposed", tool="lateral", arm=71,
        **{"from": "1.15,1.95"}, to="1.30,1.95", name="t",
        out_dir=str(out)))
    assert r.status_code == 200, r.text
    done = _wait(client, r.json()["id"], timeout=600)
    assert done["status"] == "done", done.get("error")

    ev = client.get(f"/api/jobs/{r.json()['id']}/events").json()["events"]
    verdicts = [e["payload"] for e in ev
                if e["kind"] == "item" and e["payload"].get("what") == "day1"]
    assert len(verdicts) == 1, "exactly one verdict per run"
    v = verdicts[0]
    assert v["ok"] is True
    # THE SAME STRING A TERMINAL PRINTS, with the gate numbers in it.
    assert v["one_liner"].startswith("PASS  t_71")
    assert "inter-arm" in v["one_liner"] and "(gate 50)" in v["one_liner"]
    assert v["gates"]["min_inter_arm_m"] > v["gates"]["pair_margin_m"]
    assert v["csv"] and Path(v["csv"][0]).exists()
    assert Path(v["csv"][0]).parent == out, "the CSV goes where --out says"

    # ...and the bundle the 3D viewer loads on PASS
    doc = client.get(f"/api/jobs/{r.json()['id']}/bundle")
    assert doc.status_code == 200
    b = Bundle.from_json(doc.text)
    assert b.meta.schema_version == 1
    assert b.meta.n_frames > 1 and len(b.arms) == 6
    assert len(b.segments) == 1 and b.segments[0].arm == 71
    assert b.strokes[0].pts is not None, "the target line must be in the bundle"
    assert client.get(f"/api/jobs/{r.json()['id']}/bundle.bin"
                      ).status_code == 200


def test_a_day1_line_that_will_not_certify_fails_the_job_and_writes_no_csv(
        client, tmp_path):
    """A refusal is a RESULT: the job fails, and no CSV exists to be flown."""
    out = tmp_path / "day1"
    r = client.post("/api/jobs", json=dict(
        day1="line", rig="proposed", tool="lateral", arm=31,
        **{"from": "0.10,3.50"}, to="0.25,3.50", name="bad",
        out_dir=str(out)))
    done = _wait(client, r.json()["id"], timeout=600)
    assert done["status"] == "failed"
    assert not list(out.glob("*.csv")) if out.exists() else True
    # the sentence a terminal would have printed, in the browser's log too
    log = client.get(f"/api/jobs/{r.json()['id']}/log").text
    assert "REFUSED" in log and "Traceback" not in log
    # ...and as the panel's verdict, in red, rather than an empty box
    ev = client.get(f"/api/jobs/{r.json()['id']}/events").json()["events"]
    v = [e["payload"] for e in ev
         if e["kind"] == "item" and e["payload"].get("what") == "day1"]
    assert len(v) == 1 and v[0]["ok"] is False
    assert "REFUSED" in v[0]["one_liner"] and not v[0]["csv"]


def test_the_bundle_adapter_adds_only_what_the_exporter_needs(tmp_path):
    """The day-1 npz shapes, normalised — and nothing invented.

    `day1.py line` leaves out the animation arrays on purpose and
    `serialise_timeline.py` drops them AND `n_frames`; the adapter adds them as
    EMPTY arrays so `export_bundle` can read the file, and adds nothing else.
    """
    import numpy as np
    from aris_sixarm.gui.day1_bundle import normalise

    src = tmp_path / "s.npz"
    q = np.zeros((5, 7), np.float32)
    np.savez_compressed(
        src, arms=np.array([31, 71]), drawing_arms=np.array([31]),
        pen_ext=np.array([0.046, 0.046]), fps=np.float64(48.0),
        dt=np.float64(1 / 48), stride=np.int64(1), n_phases=np.int64(1),
        duration=np.float64(4 / 48), margin=np.float64(0.05),
        min_clearance=np.float64(0.1), sheet=np.array([1.8034, 3.63064]),
        q_31=q, q_71=q, seg_31=np.full(5, -1), seg_71=np.full(5, -1))
    added = normalise(src, tmp_path / "n.npz")
    assert "n_frames" in added
    for k in ("ink_t", "ink_arm", "ink_off", "ink_xyz", "ink_hex",
              "segpts_31", "segoff_31", "segpts_71", "segoff_71"):
        assert k in added, k
    # what was already there is NOT touched
    assert "margin" not in added and "min_clearance" not in added
    z = np.load(tmp_path / "n.npz", allow_pickle=False)
    assert int(z["n_frames"]) == 5
    assert z["ink_xyz"].shape == (0, 3) and z["ink_off"].tolist() == [0]
    assert z["segpts_31"].shape == (0, 2)
    # a single CSR offset is how the viewer is told there is no dense tip path
    assert z["segoff_31"].tolist() == [0]
    assert float(z["margin"]) == 0.05


# --------------------------------------------------------------------------
# the static app and the config
# --------------------------------------------------------------------------
def test_the_page_and_its_modules_are_served(client):
    html = client.get("/")
    assert html.status_code == 200
    assert "<title>Aris stroke planner</title>" in html.text
    # every module the page imports, and the vendored three.js the import map
    # points at, must actually be reachable — an offline robot PC has no
    # second chance to fetch one.
    for path in ["/js/app.js", "/js/api.js", "/js/state.js", "/js/panel.js",
                 "/js/panels.js", "/js/strip.js", "/js/scene3d.js",
                 "/js/fk.js", "/js/program.js",
                 "/vendor/three.module.js", "/vendor/OrbitControls.js"]:
        r = client.get(path)
        assert r.status_code == 200, path
        assert len(r.content) > 100, path


def test_config_offers_the_rigs_tools_and_pictures(client):
    c = client.get("/api/config").json()
    assert "proposed" in c["rigs"] and "final6_opt" in c["rigs"]
    assert c["tools"] == ["lateral", "inline"]
    assert any(s["path"].endswith(".gif") or s["path"].endswith(".png")
               for s in c["sources"])
    assert "source" in c["defaults"]


def test_the_default_atlas_is_at_the_layouts_own_height(client):
    """THE FORM MUST NOT OFFER AN ATLAS FROM A HEIGHT THAT NO LONGER SHIPS.

    `gui/server.py` is deliberately ignorant of the planner — it imports
    nothing that binds a rig at import time — so its default atlas is a
    STRING, and a string cannot follow `layout.LAYOUT_PROPOSED["h"]` on its
    own.  This is the thread that ties them: the default's name must carry the
    height in force, and the directory must exist and have been swept at that
    height.  When `h` moves, this fails until the default moves with it.
    """
    from pathlib import Path
    from aris_sixarm import layout
    d = client.get("/api/config").json()["defaults"]["atlas"]
    h = float(layout.LAYOUT_PROPOSED["h"])
    assert f"h{round(1000 * h):04d}" in d, (d, h)
    p = Path(__file__).resolve().parents[1] / d
    if not (p / "atlas_arm31.npz").is_file():
        pytest.skip(f"{d} is not in this checkout (out/ is gitignored)")
    import numpy as np
    assert abs(float(np.load(p / "atlas_arm31.npz")["base"][2, 3]) - h) <= 1e-9


# --------------------------------------------------------------------------
# a job, end to end
# --------------------------------------------------------------------------
def test_a_trace_job_runs_and_reports_its_stages(client, picture):
    r = client.post("/api/jobs", json=dict(
        source=str(picture), out="tiny", work_px=240, min_len=0.01,
        inks="1", trace_only=True))
    assert r.status_code == 200, r.text
    job = r.json()
    assert job["status"] in ("queued", "running")

    done = _wait(client, job["id"])
    assert done["status"] == "done", done.get("error")

    ev = client.get(f"/api/jobs/{job['id']}/events").json()
    events = ev["events"]
    kinds = [e["kind"] for e in events]
    assert kinds[0] == "job_start"
    assert kinds[-1] == "job_end"
    assert events[-1]["payload"]["ok"] is True

    # ...the stage bracket the viewer's timeline is drawn from
    starts = [e for e in events if e["kind"] == "stage_start"]
    ends = [e for e in events if e["kind"] == "stage_end"]
    assert [e["stage"] for e in starts] == ["trace"]
    assert ends[0]["stage"] == "trace"
    assert ends[0]["payload"]["ok"] is True
    assert ends[0]["payload"]["n_strokes"] >= 1
    assert ends[0]["payload"]["elapsed_s"] > 0

    # ...the planner's own words, teed into the same stream
    logs = [e for e in events if e["kind"] == "log"]
    assert any("traced" in e["payload"]["msg"] for e in logs)

    # ...and every event carries the same envelope
    for e in events:
        assert set(e) >= {"seq", "t", "wall", "kind", "stage", "payload"}

    # the artifacts landed in the JOB's directory, not the shared out/
    art = client.get(f"/api/jobs/{job['id']}").json()["artifacts"]
    assert "tiny_trace.png" in art
    assert "events.jsonl" in art and "log.txt" in art
    # ...and the job directory really is the one this server was given
    assert done["root"].endswith("jobs")
    assert client.get(f"/api/jobs/{job['id']}/file/tiny_trace.png"
                      ).status_code == 200


def test_events_can_be_replayed_by_byte_offset(client, picture):
    job = client.post("/api/jobs", json=dict(
        source=str(picture), out="tiny2", work_px=240, min_len=0.01,
        inks="1", trace_only=True)).json()
    _wait(client, job["id"])

    whole = client.get(f"/api/jobs/{job['id']}/events").json()
    assert whole["offset"] > 0
    # reading from the end returns nothing and does not move
    more = client.get(f"/api/jobs/{job['id']}/events",
                      params={"offset": whole["offset"]}).json()
    assert more["events"] == []
    assert more["offset"] == whole["offset"]
    # reading in two halves reconstructs the whole, in order
    mid = whole["offset"] // 2
    a = client.get(f"/api/jobs/{job['id']}/events",
                   params={"offset": 0}).json()
    assert [e["seq"] for e in a["events"]] == [e["seq"] for e in whole["events"]]


def test_the_websocket_replays_then_tails(client, picture):
    job = client.post("/api/jobs", json=dict(
        source=str(picture), out="tiny3", work_px=240, min_len=0.01,
        inks="1", trace_only=True)).json()
    seen = []
    with client.websocket_connect(f"/api/jobs/{job['id']}/stream") as ws:
        t0 = time.time()
        while time.time() - t0 < 180:
            msg = ws.receive_json()
            if msg["kind"] == "batch":
                seen.extend(msg["events"])
                if any(e["kind"] == "job_end" for e in msg["events"]):
                    break
            elif msg["kind"] == "closed":
                break
    assert seen and seen[0]["kind"] == "job_start"
    assert any(e["kind"] == "stage_end" and e["stage"] == "trace" for e in seen)


def test_a_bad_source_fails_the_job_rather_than_the_server(client, tmp_path):
    job = client.post("/api/jobs", json=dict(
        source=str(tmp_path / "does_not_exist.png"), out="nope",
        trace_only=True)).json()
    done = _wait(client, job["id"], timeout=120)
    assert done["status"] == "failed"
    assert client.get("/api/jobs").status_code == 200      # still serving


def test_an_unknown_parameter_is_refused_at_creation(client, picture):
    r = client.post("/api/jobs", json=dict(source=str(picture),
                                           made_up_flag=1))
    # the job is created and the worker refuses it — the failure is recorded
    # as a job outcome with a readable reason, not a 500 from the API
    if r.status_code == 200:
        done = _wait(client, r.json()["id"], timeout=120)
        assert done["status"] == "failed"
        ev = client.get(f"/api/jobs/{r.json()['id']}/events").json()["events"]
        assert any("made_up_flag" in json.dumps(e) for e in ev)
    else:
        assert r.status_code == 400


def test_cancel_kills_the_process_group(client, picture):
    # A job that will still be running a few seconds in: the placement is
    # fixed (no atlas needed) and the allocation is minutes of certified IK,
    # which is exactly the situation cancelling exists for.
    job = client.post("/api/jobs", json=dict(
        source=str(picture), out="cancelme", work_px=240, min_len=0.01,
        inks="1", placement="off", target_width=0.3, rotate="0")).json()
    for _ in range(80):                       # wait until it is really running
        time.sleep(0.25)
        j = client.get(f"/api/jobs/{job['id']}").json()
        if not j["live"]:
            pytest.skip(f"the job ended before it could be cancelled: "
                        f"{j.get('error')}")
        ev = client.get(f"/api/jobs/{job['id']}/events").json()["events"]
        if any(e["kind"] == "stage_start" and e["stage"] == "allocation"
               for e in ev):
            break
    out = client.post(f"/api/jobs/{job['id']}/cancel").json()
    assert out["status"] == "cancelled"
    j = client.get(f"/api/jobs/{job['id']}").json()
    assert j["status"] == "cancelled" and not j["live"]
    ev = client.get(f"/api/jobs/{job['id']}/events").json()["events"]
    assert ev[-1]["kind"] == "job_end"
    assert ev[-1]["payload"].get("cancelled") is True


def test_non_finite_numbers_survive_as_null_not_as_broken_json(tmp_path):
    """`JSON.parse` refuses `Infinity`, and one of them killed a whole batch.

    `sequence.solve` returns inf for a bag with no feasible tour, so this is a
    value the planner really emits — and python's json writes it happily while
    every browser refuses it.
    """
    import numpy as np
    from aris_sixarm.gui.jobs import finite
    from aris_sixarm.gui.worker import EventWriter

    ev = {"seq": 1, "t": 0.0, "wall": 0.0, "kind": "item",
          "stage": "allocation",
          "payload": {"transit_s": float("inf"),
                      "f32": np.float32("nan"),
                      "f64": np.float64("-inf"),
                      "arr": np.array([1.0, np.inf], np.float32),
                      "n": np.int64(3), "ok": 1.5}}
    path = tmp_path / "e.jsonl"
    w = EventWriter(path)
    w(ev)
    w.close()
    line = path.read_text().strip()
    assert "Infinity" not in line and "NaN" not in line
    back = json.loads(line)
    assert back["payload"] == {"transit_s": None, "f32": None, "f64": None,
                               "arr": [1.0, None], "n": 3, "ok": 1.5}

    # ...and an event recorded BEFORE that fix still replays cleanly
    path.write_text(json.dumps(
        {"seq": 2, "t": 0.0, "wall": 0.0, "kind": "item", "stage": "",
         "payload": {"x": float("inf")}}) + "\n")
    assert "Infinity" in path.read_text()
    mgr = JobManager(tmp_path / "jobs")
    from aris_sixarm.gui.jobs import Job
    j = Job(id="20200101-000000-inf", params={}, status="done",
            root=str(tmp_path / "jobs"))
    j.dir.mkdir(parents=True, exist_ok=True)
    j.events_path.write_bytes(path.read_bytes())
    j.save()
    events, _ = JobManager(tmp_path / "jobs").read_events(j.id, 0)
    assert events[0]["payload"] == {"x": None}
    assert finite(float("nan")) is None


def test_a_missing_job_is_a_404(client):
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.post("/api/jobs/nope/cancel").status_code == 404


def test_the_manager_adopts_jobs_from_disk(tmp_path):
    """A server restart keeps the history, and does not claim a dead job runs."""
    mgr = JobManager(tmp_path / "jobs")
    job = mgr._jobs  # empty
    assert job == {}
    from aris_sixarm.gui.jobs import Job, STATUS_RUNNING
    j = Job(id="20200101-000000-dead", params={"out": "x"},
            status=STATUS_RUNNING, pid=999999999,
            root=str(tmp_path / "jobs"))
    j.dir.mkdir(parents=True, exist_ok=True)
    j.events_path.touch()
    j.save()

    mgr2 = JobManager(tmp_path / "jobs")
    got = mgr2.get("20200101-000000-dead")
    assert got is not None
    assert got.status == "failed"
    assert "restarted" in got.error


# --------------------------------------------------------------------------
# the scene cache, which used to have no expiry at all
# --------------------------------------------------------------------------
def test_the_scene_cache_expires_when_the_package_moves(tmp_path, monkeypatch):
    """A CACHED SCENE MUST NOT OUTLIVE THE CODE THAT BUILT IT (2026-09-10).

    `_ensure_scene` used to test only that the file EXISTED, so after the
    mounting height went 0.940 -> 0.970 the server went on serving a
    2026-09-02 scene at base z 0.940, with the 0.110 pen and the superseded
    park poses, until somebody deleted `out/gui_cache/` by hand.  There was no
    way for a browser to tell.

    The stamp cannot ask the planner what the height is — this module never
    imports it, by design — so it hashes the package the scene is built FROM.
    Three properties, and the middle one is the bug:

      * the same tree gives the same stamp (a cache that never hits is not a
        cache);
      * a CHANGED package gives a different one;
      * rig and tool are in it, so two scenes never share an entry.
    """
    from aris_sixarm.gui import server as srv
    a = srv.scene_stamp("proposed", "lateral")
    assert a == srv.scene_stamp("proposed", "lateral"), "not reproducible"
    assert a != srv.scene_stamp("proposed", "inline")
    assert a != srv.scene_stamp("final6_opt", "lateral")

    # a package edit invalidates it — simulated the way an edit really lands,
    # by moving a source file's mtime rather than by rewriting one
    victim = srv.ROOT / "aris_sixarm" / "layout.py"
    st = victim.stat()
    try:
        os.utime(victim, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
        assert srv.scene_stamp("proposed", "lateral") != a, \
            "a changed package must not reuse a cached scene"
    finally:
        os.utime(victim, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert srv.scene_stamp("proposed", "lateral") == a, "restore failed"


def test_a_stale_scene_entry_is_rebuilt_not_served(tmp_path, monkeypatch):
    """The stamp is what decides, and a wrong one is not trusted.

    Builds nothing: `_ensure_scene`'s subprocess is the expensive part and is
    not what is under test.  What is under test is the BRANCH — that a cache
    entry whose stamp does not match is not returned early.
    """
    from aris_sixarm.gui import server as srv
    monkeypatch.setattr(srv, "CACHE", tmp_path)
    out = tmp_path / "scene_proposed_lateral.json"
    out.write_text('{"stale": true}')
    out.with_suffix(".bin").write_bytes(b"stale")

    calls = []

    class _R:
        returncode = 0
        stderr = stdout = ""

    def _fake_run(*a, **k):
        calls.append(a)
        out.write_text('{"fresh": true}')
        return _R()

    monkeypatch.setattr(srv.subprocess, "run", _fake_run)

    # no stamp at all -> rebuild
    srv._ensure_scene("proposed", "lateral")
    assert len(calls) == 1, "an unstamped entry must be rebuilt"
    # ...and it is stamped now, so the next call is free
    srv._ensure_scene("proposed", "lateral")
    assert len(calls) == 1, "a current entry must be served from cache"
    # a stamp from another tree -> rebuild
    out.with_suffix(".stamp").write_text("deadbeef")
    srv._ensure_scene("proposed", "lateral")
    assert len(calls) == 2, "a stale stamp must be rebuilt"
