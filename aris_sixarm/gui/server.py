"""The GUI server: job control, a live event bus, and the static viewer.

DELIBERATELY IGNORANT OF THE PLANNER.  Nothing in this module imports
`fleet`, `allocate` or anything else that binds a rig at import time, and that
is a design constraint rather than an accident: `aris_sixarm/__init__.py`
chooses the rig and the tool from the environment the FIRST time the package is
imported, so a server that imported the planner would be stuck at one rig for
its whole life and would quietly answer questions about arm positions in the
wrong room.  Every rig-dependent answer — the scene geometry, the bundle — is
produced by a SUBPROCESS started for that rig, and cached by (rig, tool).

THE EVENT BUS IS A FILE, TAILED.  See `jobs.py` for why.  A websocket client
gets the whole history first and then the tail, so a browser that connects
halfway through an hour-long run sees the same thing as one that was there
from the start, and reloading the page is not a way to lose the record.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               PlainTextResponse, Response)
from fastapi.staticfiles import StaticFiles

from .jobs import JobManager

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "web" / "viewer"
CACHE = ROOT / "out" / "gui_cache"

RIGS = ("proposed", "final6_opt", "final6", "final", "sixarm")
TOOLS = ("lateral", "inline")

POLL_S = 0.15          # how often a websocket looks for new events


def create_app(jobs_dir=None):
    app = FastAPI(title="Aris stroke planner")
    mgr = JobManager(jobs_dir) if jobs_dir else JobManager()
    app.state.jobs = mgr

    # ---- the viewer ----------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    def index():
        p = WEB / "index.html"
        if not p.exists():
            return HTMLResponse("<h1>web/viewer/index.html is missing</h1>",
                                status_code=500)
        return HTMLResponse(p.read_text())

    if (WEB / "js").is_dir():
        app.mount("/js", StaticFiles(directory=str(WEB / "js")), name="js")
    if (WEB / "vendor").is_dir():
        app.mount("/vendor", StaticFiles(directory=str(WEB / "vendor")),
                  name="vendor")

    # ---- what the parameter form needs ---------------------------------
    @app.get("/api/config")
    def config():
        return dict(rigs=list(RIGS), tools=list(TOOLS),
                    root=str(ROOT), jobs_dir=str(mgr.dir),
                    sources=_sources(),
                    atlases=sorted(str(p.relative_to(ROOT))
                                   for p in (ROOT / "out").glob("atlas_*")
                                   if p.is_dir()),
                    placements=sorted(
                        str(p.relative_to(ROOT))
                        for p in (ROOT / "out").glob("*placement*.json")),
                    defaults=DEFAULT_PARAMS)

    # ---- jobs ----------------------------------------------------------
    @app.get("/api/jobs")
    def list_jobs(limit: int = 100):
        return mgr.list(limit)

    @app.post("/api/jobs")
    async def create_job(body: dict):
        try:
            job = mgr.create(body)
        except Exception as exc:
            raise HTTPException(400, f"{type(exc).__name__}: {exc}")
        return job.to_dict()

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        job = mgr.get(job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        d = job.to_dict()
        d["artifacts"] = sorted(p.name for p in job.dir.iterdir()) \
            if job.dir.exists() else []
        return d

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        job = mgr.cancel(job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        return job.to_dict()

    @app.get("/api/jobs/{job_id}/events")
    def job_events(job_id: str, offset: int = 0):
        job = mgr.get(job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        events, off = mgr.read_events(job_id, offset)
        return dict(events=events, offset=off, status=job.status,
                    live=job.status in ("queued", "running"))

    @app.get("/api/jobs/{job_id}/log", response_class=PlainTextResponse)
    def job_log(job_id: str):
        job = mgr.get(job_id)
        if job is None or not job.log_path.exists():
            raise HTTPException(404, "no log")
        return job.log_path.read_text(errors="replace")

    @app.get("/api/jobs/{job_id}/bundle")
    def job_bundle(job_id: str):
        return _job_file(mgr, job_id, "bundle.json", "application/json")

    @app.get("/api/jobs/{job_id}/bundle.bin")
    def job_bundle_bin(job_id: str):
        return _job_file(mgr, job_id, "bundle.bin", "application/octet-stream")

    @app.get("/api/jobs/{job_id}/file/{name}")
    def job_file(job_id: str, name: str):
        if "/" in name or ".." in name:
            raise HTTPException(400, "bad name")
        return _job_file(mgr, job_id, name, None)

    # ---- the rig-dependent scene ---------------------------------------
    @app.get("/api/scene")
    def scene(rig: str = "proposed", tool: str = "lateral"):
        path = _ensure_scene(rig, tool)
        return FileResponse(path, media_type="application/json")

    @app.get("/api/scene.bin")
    def scene_bin(rig: str = "proposed", tool: str = "lateral"):
        path = _ensure_scene(rig, tool).with_suffix(".bin")
        return FileResponse(path, media_type="application/octet-stream")

    # ---- the live stream -------------------------------------------------
    @app.websocket("/api/jobs/{job_id}/stream")
    async def stream(ws: WebSocket, job_id: str):
        await ws.accept()
        job = mgr.get(job_id)
        if job is None:
            await ws.send_text(json.dumps({"kind": "error",
                                           "payload": {"msg": "no such job"}}))
            await ws.close()
            return
        offset, idle_after_end = 0, 0
        try:
            while True:
                events, offset = mgr.read_events(job_id, offset)
                if events:
                    # ONE FRAME PER POLL, NOT ONE PER EVENT.  A busy stage
                    # emits thousands of log lines a second and a websocket
                    # message each would spend the browser's whole frame
                    # budget in the parser.  Batching is what keeps the
                    # timeline animating while the planner is shouting.
                    await ws.send_text(json.dumps({"kind": "batch",
                                                   "events": events,
                                                   "offset": offset}))
                job = mgr.get(job_id)
                live = job is not None and job.status in ("queued", "running")
                if not live:
                    # Drain whatever landed after the process exited, then say
                    # so once and stop; a client that keeps the socket open on
                    # a finished job is polling a file that will never grow.
                    idle_after_end += 1
                    if idle_after_end > 3 and not events:
                        await ws.send_text(json.dumps(
                            {"kind": "closed",
                             "payload": job.to_dict() if job else {}}))
                        break
                await asyncio.sleep(POLL_S)
        except WebSocketDisconnect:
            return
        except Exception:
            pass
        finally:
            try:
                await ws.close()
            except Exception:
                pass

    @app.on_event("shutdown")
    def _shutdown():
        mgr.shutdown()

    return app


# --------------------------------------------------------------------------
DEFAULT_PARAMS = dict(
    source="assets/csail/csail_old_med.gif",
    rig="proposed", tool="lateral",
    out="gui", title="", inks="auto",
    placement="off", target_width=0.9, offset=[0.0, 0.0], rotate="90",
    # THE ATLAS DEFAULT FOLLOWS THE LAYOUT'S HEIGHT, and it is a STRING here
    # on purpose: this module never imports the planner (see the module
    # docstring), so it cannot ask `layout.LAYOUT_PROPOSED["h"]` what the
    # height is.  Moved to the 0.970 sweep on 2026-09-10 with
    # `LAYOUT_PROPOSED["h"]`; `tests/test_gui.py` pins the two together so a
    # height change that misses this line fails rather than silently offering
    # a stale atlas in the form.
    arms="all", atlas="out/atlas_proposed_h0970_lat0860",
    max_probes=1, tilt_max_deg=0.0, min_len=0.025,
    band_objective="maximin_sigma", sequencer="opt",
    no_rrt=True, no_verify=True, two_pass=False,
    arm_phases="solo", idle_policy="freeze", image_jobs=6, fps=24.0,
    substeps=2, subcheck=2, min_coverage=0.0, select_profile=False,
    skip_unconductable=True, trace_only=False, place_only=False,
    scales=[0.4, 1.0, 13], top=3, slack=0.01, jobs=6, margin=0.06)


def _sources():
    """Pictures the form can offer. -> [{path, label, bytes}]."""
    out = []
    for pat in ("assets/artworks/**/*", "assets/csail/*"):
        for p in sorted(ROOT.glob(pat)):
            if p.is_file() and p.suffix.lower() in (
                    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg"):
                out.append(dict(path=str(p.relative_to(ROOT)),
                                label=p.name, bytes=p.stat().st_size))
    return out


def _job_file(mgr, job_id, name, media):
    job = mgr.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    p = job.dir / name
    if not p.exists():
        raise HTTPException(404, f"{name} not written for this job")
    return FileResponse(p, media_type=media)


_SCENE_LOCK = {}


def _ensure_scene(rig, tool):
    """The scene for one (rig, tool), built by a subprocess and cached.

    A SUBPROCESS BECAUSE THE RIG IS AN IMPORT-TIME DECISION.  See the module
    docstring: the only honest way for one server to answer about five rigs is
    to ask five interpreters.  The cache is a file, so the second browser tab
    and the next server restart both pay nothing.
    """
    if rig not in RIGS or tool not in TOOLS:
        raise HTTPException(400, f"unknown rig/tool {rig!r}/{tool!r}")
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f"scene_{rig}_{tool}.json"
    if out.exists() and out.with_suffix(".bin").exists():
        return out
    env = dict(os.environ)
    env["ARIS_RIG"] = rig
    env["ARIS_TOOL"] = tool
    env.setdefault("MPLBACKEND", "Agg")
    code = ("from aris_sixarm.program_schema import export_scene; "
            f"export_scene({str(out)!r})")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env,
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0 or not out.exists():
        raise HTTPException(
            500, f"could not build the scene for rig={rig} tool={tool}:\n"
                 + (r.stderr or r.stdout)[-2000:])
    return out


app = None


def get_app():
    global app
    if app is None:
        app = create_app()
    return app
