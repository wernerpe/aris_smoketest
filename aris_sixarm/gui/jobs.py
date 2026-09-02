"""Planning jobs: one directory, one subprocess, one append-only event log.

THE JOB DIRECTORY IS THE JOB.  Everything a job produces — its parameters, its
event stream, the planner's stdout, and every artifact `scripts/draw.py`
writes — lands under `out/gui_jobs/<job_id>/`, and nothing about a finished job
lives anywhere else.  That is what makes replay free: the viewer asks for a
finished job's events and gets the same JSONL the browser was streamed live,
because it IS the same file.  A server restart loses no history, and a job
whose browser tab was closed halfway is not a job whose record has a hole in it.

WHY A SUBPROCESS AND NOT A THREAD.  Three reasons, and any one of them would be
enough.  The planner is CPU-bound python (`allocate.balance_loads` is minutes of
one core), so a thread would hold the GIL and the websocket would go silent
exactly while the interesting stage was running.  A cancel has to be immediate
and unconditional — there is no cooperative check inside a Held-Karp DP — and
only a signal to another process gives that.  And the rig and the tool are
chosen by environment variables read at `import aris_sixarm` time
(`aris_sixarm/__init__.py`), so two jobs at different rigs cannot share an
interpreter at all; a fresh process per job is not overhead here, it is the
only correct answer.

WHY A FILE AND NOT A PIPE.  A pipe between the worker and the server would
deadlock the moment nobody drained it (the planner prints tens of thousands of
lines), would lose everything emitted before the browser connected, and would
have to be re-invented for replay.  An append-only JSONL that the worker flushes
and any number of readers tail by byte offset has none of those properties, and
`tail -f out/gui_jobs/<id>/events.jsonl | jq` works from a terminal while the
GUI is running, which is worth something on its own.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JOBS_DIR = ROOT / "out" / "gui_jobs"

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"
LIVE = (STATUS_QUEUED, STATUS_RUNNING)


def new_job_id():
    """Sortable, unique, and readable in a directory listing."""
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]


@dataclass
class Job:
    id: str
    params: dict
    status: str = STATUS_QUEUED
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    pid: int | None = None
    error: str | None = None
    returncode: int | None = None
    # THE ROOT IS A PROPERTY OF THE JOB, NOT A MODULE CONSTANT.  It used to be
    # `JOBS_DIR` here and the manager's own `jobs_dir` there, so a server (or a
    # test) started with a different jobs directory listed one place and wrote
    # to another — the listing was empty and the files went into the shared
    # `out/gui_jobs` anyway.  One place decides where a job lives, and it
    # travels with the job.
    root: str = str(JOBS_DIR)

    @property
    def dir(self):
        return Path(self.root) / self.id

    @property
    def events_path(self):
        return self.dir / "events.jsonl"

    @property
    def log_path(self):
        return self.dir / "log.txt"

    @property
    def meta_path(self):
        return self.dir / "job.json"

    def to_dict(self):
        d = asdict(self)
        d["elapsed_s"] = round(
            (self.finished or time.time()) - (self.started or self.created), 3)
        d["live"] = self.status in LIVE
        return d

    def save(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.meta_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=1))
        tmp.replace(self.meta_path)          # never a half-written job.json

    @classmethod
    def load(cls, path):
        path = Path(path)
        d = json.loads(path.read_text())
        # The root is taken from WHERE THE FILE IS, not from what it says: a
        # job directory that has been moved or a jobs tree that has been
        # copied should still read, and the parent of `job.json` is the only
        # answer that is true in both cases.
        return cls(id=d["id"], params=d.get("params", {}),
                   status=d.get("status", STATUS_FAILED),
                   created=d.get("created", 0.0), started=d.get("started"),
                   finished=d.get("finished"), pid=d.get("pid"),
                   error=d.get("error"), returncode=d.get("returncode"),
                   root=str(path.parent.parent))


class JobManager:
    """Create, watch, cancel.  Holds no planner state and imports no planner."""

    def __init__(self, jobs_dir=JOBS_DIR):
        self.dir = Path(jobs_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._procs: dict[str, subprocess.Popen] = {}
        self._jobs: dict[str, Job] = {}
        self._rescan()

    # -- registry ---------------------------------------------------------
    def _rescan(self):
        """Adopt the job directories on disk, so a restart keeps the history.

        A job that was RUNNING when the server died is not running now: no
        supervisor survived to reap it and its pid means nothing after a
        reboot.  It is marked failed with a reason rather than left claiming
        to be live forever, which is the failure mode of every progress UI
        that trusts a status field it never rewrites.
        """
        for meta in sorted(self.dir.glob("*/job.json")):
            try:
                job = Job.load(meta)
            except Exception:
                continue
            if job.id in self._jobs:
                continue
            if job.status in LIVE and job.id not in self._procs:
                if not _pid_alive(job.pid):
                    job.status = STATUS_FAILED
                    job.error = ("the server restarted while this job was "
                                 "running; its process is gone")
                    job.finished = job.finished or time.time()
                    job.save()
            self._jobs[job.id] = job

    def list(self, limit=100):
        self._rescan()
        self._reap()
        return [j.to_dict() for j in
                sorted(self._jobs.values(), key=lambda j: -j.created)[:limit]]

    def get(self, job_id):
        self._rescan()
        self._reap()
        return self._jobs.get(job_id)

    # -- lifecycle --------------------------------------------------------
    def create(self, params):
        job = Job(id=new_job_id(), params=dict(params), root=str(self.dir))
        job.dir.mkdir(parents=True, exist_ok=True)
        job.events_path.touch()
        job.save()
        self._jobs[job.id] = job
        self._spawn(job)
        return job

    def _spawn(self, job):
        env = dict(os.environ)
        # THE ONLY PLACE ARIS_RIG / ARIS_TOOL ARE EVER SET.  They are read at
        # `import aris_sixarm` time and cannot be changed afterwards, so they
        # belong to a process and not to a call; a fresh process per job is
        # what makes them safe.  Library code and tests must keep passing the
        # rig and the tool explicitly — see docs/VIEWER.md.
        rig = str(job.params.get("rig") or "").strip()
        tool = str(job.params.get("tool") or "").strip()
        if rig:
            env["ARIS_RIG"] = rig
        if tool:
            env["ARIS_TOOL"] = tool
        env["ARIS_GUI_JOB_DIR"] = str(job.dir)
        env["PYTHONUNBUFFERED"] = "1"
        # Matplotlib must not try to open a window from a headless worker, and
        # the planner writes several figures.
        env.setdefault("MPLBACKEND", "Agg")
        cmd = [sys.executable, "-m", "aris_sixarm.gui.worker", str(job.dir)]
        log = open(job.log_path, "ab", buffering=0)
        proc = subprocess.Popen(
            cmd, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            # OWN PROCESS GROUP, SO A CANCEL REACHES THE CHILDREN.  The
            # allocator forks a pool for the placement search and the profile
            # grid; killing only the worker would leave those running and the
            # cores busy while the GUI says the job is cancelled.
            start_new_session=True)
        self._procs[job.id] = proc
        job.pid = proc.pid
        job.status = STATUS_RUNNING
        job.started = time.time()
        job.save()

    def cancel(self, job_id):
        job = self.get(job_id)
        if job is None:
            return None
        proc = self._procs.get(job_id)
        if job.status not in LIVE:
            return job
        if proc is not None and proc.poll() is None:
            _kill_group(proc.pid)
        elif _pid_alive(job.pid):
            _kill_group(job.pid)
        job.status = STATUS_CANCELLED
        job.finished = time.time()
        job.error = "cancelled by the operator"
        job.save()
        _append_event(job.events_path, {"kind": "job_end", "stage": "",
                                        "payload": {"ok": False,
                                                    "cancelled": True,
                                                    "error": job.error}})
        return job

    def _reap(self):
        """Turn finished processes into finished jobs.

        The worker writes its own `job_end` event and its own exit reason; this
        only has to notice that the process is gone and copy the return code
        across, so a worker that died without saying anything (OOM, SIGKILL)
        still ends up in a terminal state instead of spinning forever.
        """
        for jid, proc in list(self._procs.items()):
            rc = proc.poll()
            if rc is None:
                continue
            del self._procs[jid]
            job = self._jobs.get(jid)
            if job is None or job.status not in LIVE:
                continue
            job.returncode = rc
            job.finished = time.time()
            if rc == 0:
                job.status = STATUS_DONE
            else:
                job.status = STATUS_FAILED
                job.error = job.error or _last_error(job) or \
                    f"the planner exited with code {rc}"
                _append_event(job.events_path,
                              {"kind": "job_end", "stage": "",
                               "payload": {"ok": False, "error": job.error,
                                           "returncode": rc}})
            job.save()

    def shutdown(self):
        for proc in self._procs.values():
            if proc.poll() is None:
                _kill_group(proc.pid)

    # -- the event stream -------------------------------------------------
    def read_events(self, job_id, offset=0):
        """Every complete JSON line from byte `offset`. -> (events, new offset).

        Reading by BYTE OFFSET and stopping at the last newline is what makes a
        tail safe against a partially-written line: the worker appends with one
        `write` per event, but nothing in POSIX promises that a reader cannot
        see half of one, and a truncated line would otherwise abort the stream
        rather than being picked up whole on the next poll.
        """
        job = self.get(job_id)
        if job is None:
            return [], offset
        path = job.events_path
        if not path.exists():
            return [], offset
        size = path.stat().st_size
        if size <= offset:
            return [], offset
        with open(path, "rb") as f:
            f.seek(offset)
            blob = f.read(size - offset)
        cut = blob.rfind(b"\n")
        if cut < 0:
            return [], offset
        out = []
        for line in blob[:cut].split(b"\n"):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue                      # a corrupt line is not a stream
        return out, offset + cut + 1


# --------------------------------------------------------------------------
def _append_event(path, ev):
    ev.setdefault("seq", -1)
    ev.setdefault("t", 0.0)
    ev.setdefault("wall", time.time())
    with open(path, "a") as f:
        f.write(json.dumps(ev) + "\n")


def _pid_alive(pid):
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def _kill_group(pid):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(int(pid)), sig)
        except (OSError, ValueError):
            return
        for _ in range(20):
            if not _pid_alive(pid):
                return
            time.sleep(0.05)


def _last_error(job):
    """The tail of the worker's log, for a job that died without an event."""
    try:
        txt = job.log_path.read_text(errors="replace").strip().splitlines()
    except OSError:
        return None
    return "\n".join(txt[-12:]) if txt else None
