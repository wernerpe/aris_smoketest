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
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               PlainTextResponse, Response)
from fastapi.staticfiles import StaticFiles

from . import operator                 # RUN ON ARM: ssh, day1.py send, tails
from .jobs import JobManager

ROOT = Path(__file__).resolve().parents[2]
WEB = ROOT / "web" / "viewer"
CACHE = ROOT / "out" / "gui_cache"

RIGS = ("proposed", "final6_opt", "final6", "final", "sixarm")
TOOLS = ("lateral", "inline")

POLL_S = 0.15          # how often a websocket looks for new events

# THE DRAKE MESHCAT SCENE LIVES ON ONE PORT AND ONLY ONE.
#
# `scripts/meshcat_drake.py` is the high-quality view of a programme — the real
# FR3 glTFs in Drake's own meshcat, as against `web/viewer/js/scene3d.js`,
# which draws the fleet out of three.js primitives.  The GUI does not manage a
# pool of them: 7000-7008 belong to other people's scenes on this machine and
# 8765 is this server, so there is exactly one slot, 7009, and asking for a
# second programme REPLACES what is in it.  That is also what a person means by
# the button — they want to look at THIS run.
MESHCAT_PORT = 7009
MESHCAT_SCRIPT = ROOT / "scripts" / "meshcat_drake.py"

# ---------------------------------------------------------------------------
# THE SCENE CACHE HAS TO EXPIRE, AND IT DID NOT (2026-09-10)
# ---------------------------------------------------------------------------
# `_ensure_scene` wrote `out/gui_cache/scene_<rig>_<tool>.{json,bin}` once and
# returned it forever — the only test was that the file EXISTS.  So the moment
# the mounting height, the tool or a park pose moved, every browser got the old
# rig with no warning and no way to tell.  Found for real: after h went
# 0.940 -> 0.970 the served scene was still a 2026-09-02 build at base
# z = 0.940 with the 0.110 pen and the pre-2026-09-07 parks, and it had to be
# deleted by hand.
#
# THE FIX HAS TO WORK WITHOUT IMPORTING THE PLANNER, which is this module's
# whole design (see the module docstring): the server cannot ask
# `layout.LAYOUT_PROPOSED["h"]` what the height is, because importing that
# binds a rig at import time and the server answers about five.  So the key is
# a CONTENT stamp over the package the subprocess would import — every .py
# under `aris_sixarm/`, by size and mtime — plus the rig, the tool and a
# version scalar.  It is CONSERVATIVE by construction: an edit to an unrelated
# module rebuilds a scene that would not have changed, which costs one
# subprocess and is the right way round.  What it can never do is serve a
# scene built from code that is no longer on disk.
#
# `SCENE_CACHE_V` is bumped by hand when the SHAPE of the exported scene
# changes (`program_schema.export_scene`), which the file stamps cannot see
# if the change is to a schema this module does not import.
# v2 (2026-09-16): the scene carries `tool_model` — the pen holder read out of
# `assets/system_model/installation_fatfingers.urdf`.  The stamp hashes only
# `aris_sixarm/**.py`, so a change to that URDF or to the holder meshes is
# invisible to it; bump this by hand when the model itself moves.
SCENE_CACHE_V = 2


def scene_stamp(rig, tool):
    """What a cached scene depends on. -> hex digest, no planner import."""
    h = hashlib.sha256()
    h.update(f"v{SCENE_CACHE_V}|{rig}|{tool}\n".encode())
    pkg = ROOT / "aris_sixarm"
    for f in sorted(pkg.rglob("*.py")):
        try:
            st = f.stat()
        except OSError:                      # vanished mid-walk: rebuild
            return h.hexdigest() + "-racing"
        h.update(f"{f.relative_to(ROOT)}|{st.st_size}|{st.st_mtime_ns}\n"
                 .encode())
    return h.hexdigest()


def create_app(jobs_dir=None):
    app = FastAPI(title="Aris stroke planner")
    mgr = JobManager(jobs_dir) if jobs_dir else JobManager()
    app.state.jobs = mgr
    app.state.meshcat_proc = None
    # WHICH ARM HAS A GREEN STACK CHECK, AND WHEN.  Session state, deliberately
    # not persisted: a check is a statement about the rig a minute ago, and a
    # server that restarted has not made one.
    app.state.op_checks = {}
    app.state.tails = operator.Tails()

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
                    defaults=DEFAULT_PARAMS, day1=DAY1_DEFAULTS)

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

    # ---- the Drake meshcat view ----------------------------------------
    @app.post("/api/meshcat")
    def open_meshcat(body: dict):
        """(Re)launch `meshcat_drake.py` on 7009 for one programme.

        NOT A JOB.  `JobManager` owns things that run to completion and leave
        artifacts; this is a viewer that runs until it is replaced, and giving
        it a job id would put a permanently-"running" row in the job list.  It
        gets the same detached-process-group treatment so that replacing it
        kills the whole tree.
        """
        npz = _under_root(body.get("npz"), ".npz")
        program = (_under_root(body.get("program"), ".json")
                   if body.get("program") else None)
        rig = str(body.get("rig") or "proposed")
        tool = str(body.get("tool") or "lateral")
        if rig not in RIGS or tool not in TOOLS:
            raise HTTPException(400, f"unknown rig/tool {rig!r}/{tool!r}")
        if not MESHCAT_SCRIPT.exists():
            raise HTTPException(500, f"{MESHCAT_SCRIPT} is missing")
        arms = str(body.get("only_arms") or "")
        argv = [sys.executable, "-u", str(MESHCAT_SCRIPT),
                "--npz", str(npz), "--port", str(MESHCAT_PORT), "--loop"]
        if program is not None:
            argv += ["--program", str(program)]
        if arms:
            argv += ["--only-arms", arms]
        replaced = _kill_meshcat(app.state)
        env = dict(os.environ)
        env["ARIS_RIG"] = rig
        env["ARIS_TOOL"] = tool
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("MPLBACKEND", "Agg")
        log_path = ROOT / "out" / f"meshcat_drake_{MESHCAT_PORT}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # APPEND, like `jobs._spawn`.  Truncating destroys the log of a scene
        # somebody started from a terminal and is still watching — which is
        # exactly what `tests/test_gui_backend.py` did to the live 7009 scene
        # the first time it ran, because the spawn is faked in that test and
        # this `open` is not.
        log = open(log_path, "ab", buffering=0)
        try:
            proc = subprocess.Popen(
                argv, cwd=str(ROOT), env=env, stdout=log,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                start_new_session=True)
        finally:
            log.close()
        app.state.meshcat_proc = proc
        return dict(port=MESHCAT_PORT, pid=proc.pid, replaced=replaced,
                    url=f"http://{socket.getfqdn()}:{MESHCAT_PORT}/",
                    npz=str(npz.relative_to(ROOT)),
                    log=str(log_path.relative_to(ROOT)),
                    only_arms=arms, rig=rig, tool=tool)

    @app.get("/api/meshcat")
    def meshcat_status():
        proc = getattr(app.state, "meshcat_proc", None)
        live = proc is not None and proc.poll() is None
        return dict(port=MESHCAT_PORT, live=live,
                    pid=(proc.pid if live else None),
                    url=f"http://{socket.getfqdn()}:{MESHCAT_PORT}/")

    # ---- RUN ON ARM: the operator box ------------------------------------
    #
    # FIVE BUTTONS AND A TAIL, AND THE SERVER HOLDS THE GATE.  `gui/operator.py`
    # builds every argument list and owns the refusal; this layer only records
    # which arm has a green stack check in THIS server's lifetime, because that
    # is a fact about the session and not about a request.  A client cannot
    # assert it: `stack_ok` is written here, from `app.state.op_checks`, and
    # whatever the body said about it is dropped.
    @app.get("/api/operator")
    def operator_config():
        st = getattr(app.state, "op_checks", {})
        now = time.time()
        return dict(
            **operator.config(),
            checks={str(a): dict(ok=bool(v.get("ok")),
                                 age_s=round(now - v.get("wall", now), 1),
                                 fresh=_check_fresh(v, now),
                                 cmd=v.get("cmd"), output=v.get("output"))
                    for a, v in st.items()},
            tails={str(a): dict(live=app.state.tails.live(a))
                   for a in operator.CANDIDATE_IDS},
            csv=operator.last_pass_csv(mgr))

    @app.post("/api/operator/site")
    def operator_site(body: dict):
        """Write `config/site.json` — the Site setup form's Save button.

        ONE TRACKED FILE FOR THE WHOLE INSTALLATION, and the GUI edits it in
        place rather than keeping a second copy of the same facts.  The answer
        is the file as it now reads, so the form shows what was actually
        stored and not what it hoped to store.
        """
        try:
            st, source = operator.save_site(body or {})
        except OSError as exc:
            raise HTTPException(500, f"could not write the site file: {exc}")
        return dict(site=st, source=source, map={
            str(s): dict(arm=operator.arm_of(s, st),
                         line=operator.mapping_line(s, st))
            for s in operator.SLOTS})

    @app.post("/api/operator/identify")
    def operator_identify(body: dict):
        """Poll 31 / 71 / 97 for live joints: which robot is where.

        THE ONE HONEST ANSWER TO "WHICH ARM IS THAT".  Every id is asked over
        ssh, the ones that answer come back with their joints, and the browser
        poses them in the viewer with the tool model drawn.  Moving an arm by
        hand and watching which row changes is the identification.
        """
        ids = body.get("ids") if body else None
        rows = operator.identify(ids)
        st, _ = operator.site()
        parks = {}
        rig = str((body or {}).get("rig") or "proposed")
        tool = str((body or {}).get("tool") or "lateral")
        if rig in RIGS and tool in TOOLS:
            try:
                doc = json.loads(_ensure_scene(rig, tool).read_text())
                parks = {int(a["arm"]): [float(x) for x in a["q_seed"]]
                         for a in doc.get("arms", [])}
            except Exception:
                parks = {}
        for r in rows:
            park = parks.get(r["arm"])
            r["park"] = park
            r["delta"] = ([round(q - p, 6) for q, p in zip(r["q"], park)]
                          if r.get("q") and park else None)
        return dict(arms=rows, slots={str(s): operator.arm_of(s, st)
                                      for s in operator.SLOTS},
                    map={str(s): operator.mapping_line(s, st)
                         for s in operator.SLOTS})

    @app.post("/api/operator/check")
    def operator_check(body: dict):
        arm = _op_arm(body)
        r = operator.run_capture(operator.check_argv(arm))
        r["healthy"] = operator.is_healthy(r["output"])
        # ONLY "STACK HEALTHY" IS GREEN.  A zero exit code from a script that
        # printed something else is not a healthy stack, and an ssh that could
        # not connect is certainly not one.
        app.state.op_checks[arm] = dict(ok=r["healthy"], wall=time.time(),
                                        cmd=r["cmd"], output=r["output"])
        return r

    @app.post("/api/operator/copy")
    def operator_copy(body: dict):
        """`day1.py send --dry-run`: copies nothing, prints the run line.

        ADDRESSED BY SLOT, dispatched to whatever arm `config/site.json` says
        is bolted into it — `send_argv` adds `--as-arm` when those differ, and
        refuses when they differ and `day1.py` cannot say so.
        """
        slot = _op_slot(body)
        csv = str(body.get("csv") or "")
        try:
            argv = operator.send_argv(slot, csv)
        except operator.GateRefused as exc:
            raise HTTPException(400, str(exc))
        r = operator.run_capture(argv)
        r["mapping"] = operator.mapping_line(slot)
        return r

    @app.post("/api/operator/hold")
    def operator_hold(body: dict):
        return operator.run_capture(operator.hold_argv(_op_arm(body)))

    @app.post("/api/operator/kill")
    def operator_kill(body: dict):
        return operator.run_capture(operator.kill_argv(_op_arm(body)))

    @app.post("/api/operator/run")
    def operator_run(body: dict):
        """THE ONLY ENDPOINT ON THIS PAGE THAT CAN MOVE AN ARM.

        The slot is what was chosen and typed; the ARM is what
        `config/site.json` maps it to, and the stack check that gates it is the
        check for THAT arm — checking slot 31 while 97 is bolted into it would
        be a green light from a robot nobody is about to move.
        """
        slot = _op_slot(body)
        arm = operator.arm_of(slot)
        chk = app.state.op_checks.get(arm, {})
        params = dict(operator="run", slot=slot, arm=arm,
                      csv=str(body.get("csv") or ""),
                      confirm=str(body.get("confirm") or ""),
                      rig=str(body.get("rig") or "proposed"),
                      tool=str(body.get("tool") or "lateral"),
                      # NOT FROM THE BODY.  This server ran the check and this
                      # server remembers the answer.
                      stack_ok=bool(chk.get("ok"))
                      and _check_fresh(chk, time.time()))
        try:
            argv = operator.build_run_argv(params)
        except operator.GateRefused as exc:
            raise HTTPException(409, str(exc))
        job = mgr.create(params)
        d = job.to_dict()
        d["cmd"] = operator.shown(argv)
        d["mapping"] = operator.mapping_line(slot)
        return d

    @app.post("/api/operator/pose")
    def operator_pose(body: dict):
        """SHOW CURRENT POSE — the arm's live joints, and what they mean.

        A MEASUREMENT, NOT A MODEL.  The answer is whatever the operator box
        published, together with the raw text it came in; the park it is
        compared against is the scene's own `q_seed` for that arm, which is the
        pose the viewer draws when nothing is playing.  The difference per
        joint is the number a person reads while holding the rendered hand next
        to the real one — the tool model is drawn in both pictures, which is the
        whole point of the button.
        """
        arm = _op_arm(body)
        rig = str(body.get("rig") or "proposed")
        tool = str(body.get("tool") or "lateral")
        if rig not in RIGS or tool not in TOOLS:
            raise HTTPException(400, f"unknown rig/tool {rig!r}/{tool!r}")
        r = operator.read_pose(arm)
        parks = {}
        try:
            doc = json.loads(_ensure_scene(rig, tool).read_text())
            parks = {int(a["arm"]): [float(x) for x in a["q_seed"]]
                     for a in doc.get("arms", [])}
        except Exception as exc:
            r["park_error"] = f"{type(exc).__name__}: {exc}"
        r["rig"], r["tool"] = rig, tool
        r["park"] = parks.get(arm)
        r["delta"] = ([round(q - p, 6) for q, p in zip(r["q"], r["park"])]
                      if r.get("q") and r.get("park") else None)
        r["meshcat"] = None
        # THE DRAKE SCENE TOO, WHEN ONE IS ASKED FOR.  It is the same launch
        # path the panel's own button uses, handed a two-frame npz of this
        # pose, so `scripts/meshcat_drake.py` is untouched by this feature.
        if r.get("q") and body.get("meshcat"):
            try:
                npz = operator.pose_npz(
                    arm, r["q"], parks,
                    ROOT / "out" / "gui_operator" / f"pose_arm{arm}.npz")
                r["meshcat"] = open_meshcat(dict(
                    npz=str(npz.relative_to(ROOT)), rig=rig, tool=tool,
                    only_arms=str(arm)))
            except HTTPException as exc:
                r["meshcat_error"] = str(exc.detail)
            except Exception as exc:
                r["meshcat_error"] = f"{type(exc).__name__}: {exc}"
        return r

    @app.post("/api/operator/tail")
    def operator_tail(body: dict):
        arm = _op_arm(body)
        action = str(body.get("action") or "start")
        if action == "stop":
            return app.state.tails.stop(arm)
        return app.state.tails.start(arm, operator.tail_argv(arm))

    @app.get("/api/operator/tail")
    def operator_tail_read(arm: int, offset: int = 0):
        try:
            return app.state.tails.read(arm, offset)
        except operator.GateRefused as exc:
            raise HTTPException(400, str(exc))

    # AN ARM IS A ROBOT AND A SLOT IS A POSITION, and the two endpoints that
    # take one must not take the other: `check`/`hold`/`kill`/`tail`/`pose`
    # address a DDS domain, `copy`/`run` address a plan.
    def _op_arm(body):
        try:
            return operator._id((body or {}).get("arm"), operator.site()[0])
        except operator.GateRefused as exc:
            raise HTTPException(400, str(exc))

    def _op_slot(body):
        b = body or {}
        try:
            return operator._slot(b.get("slot", b.get("arm")))
        except operator.GateRefused as exc:
            raise HTTPException(400, str(exc))

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
        _kill_meshcat(app.state)
        app.state.tails.shutdown()
        mgr.shutdown()

    return app


def _check_fresh(check, now=None):
    """A stack check is only good for `CHECK_TTL_S`.  -> bool.

    A HEALTHY STACK IS A PERISHABLE FACT.  The panel's gate is "somebody
    checked this arm and it was healthy", and an hour-old check is not that —
    the arm may have been stopped, guided by hand, or had its controllers
    swapped since.  Going stale disarms the RUN button rather than refusing at
    the end of a long press, which is the order a person wants.
    """
    if not check:
        return False
    return ((now or time.time()) - float(check.get("wall", 0.0))
            <= operator.CHECK_TTL_S)


def _under_root(name, suffix):
    """A caller-supplied path -> an absolute path inside the repo, or 400.

    The browser hands back a path the SERVER told it about, but that is not a
    reason to trust it: the string makes a round trip through a page anyone on
    the lab network can open.  Resolve it and refuse anything that leaves the
    repo or is not the expected kind of file.
    """
    if not name:
        raise HTTPException(400, "no npz given")
    p = Path(str(name))
    p = (p if p.is_absolute() else ROOT / p).resolve()
    if not p.is_relative_to(ROOT):
        raise HTTPException(400, f"{name} is outside the repo")
    if p.suffix != suffix:
        raise HTTPException(400, f"{name} is not a {suffix} file")
    if not p.exists():
        raise HTTPException(404, f"{name} does not exist")
    return p


def _kill_meshcat(state):
    """Kill the scene currently on 7009, if this server started it.

    -> True if there was one.  The process group, not the process: the script
    is started with `start_new_session=True` exactly so that this reaches
    whatever it spawned.
    """
    proc = getattr(state, "meshcat_proc", None)
    state.meshcat_proc = None
    if proc is None or proc.poll() is not None:
        return False
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    return True


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


# HARDWARE DAY 1 — what the panel's form starts at.
#
# STRINGS AND NUMBERS ONLY, for the same reason `DEFAULT_PARAMS["atlas"]` is a
# string: this module never imports the planner (see the module docstring), so
# it cannot ask `layout` where arm 71's base is.  Nothing here is a second
# definition of anything — `scripts/day1.py` plans, grades and REFUSES whatever
# the form offers, so a stale default costs one refusal and not a wrong robot.
# The starting line is one both arms certify (tests/test_day1.py GOOD[71]).
DAY1_DEFAULTS = {
    "arms": [31, 71],
    "variants": ["alt", "concurrent", "hover"],
    "arm": 71, "from": "1.15,1.95", "to": "1.30,1.95",
    "name": "probe", "hover": False, "hover_m": 0.03,
    "variant": "alt", "rig": "proposed", "tool": "lateral",
    # the SOLO word: "both" is the two-arm asset re-check (the `variant`
    # control), 31 or 71 plans the word for that one arm here and now.
    "word_arm": "both", "word_width": 0.55, "word_hover": False,
    # arm 31 REFUSES a line exactly on the seam (y = 1.8153): the go-home leg
    # does not clear the paper plane there.  Said in the form so it is read
    # before the refusal rather than after it.
    "note": "metres, canvas datum.  x across 1.8034, y along 3.63064.  "
            "seam y = 1.8153 — arm 31 refuses a line exactly on it.",
}


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
    stamp = out.with_suffix(".stamp")
    want = scene_stamp(rig, tool)
    if (out.exists() and out.with_suffix(".bin").exists()
            and stamp.exists() and stamp.read_text().strip() == want):
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
    # LAST, so a half-written scene is never stamped as current: a crash
    # between the two leaves a cache entry that simply rebuilds next time.
    stamp.write_text(want)
    return out


app = None


def get_app():
    global app
    if app is None:
        app = create_app()
    return app
