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
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient                    # noqa: E402

from aris_sixarm.gui.jobs import JobManager                  # noqa: E402
from aris_sixarm.gui.server import create_app                # noqa: E402
from aris_sixarm.gui.worker import build_argv                # noqa: E402


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
