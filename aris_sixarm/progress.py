"""A progress hook the pipeline stages call, and nothing else.

WHY A MODULE AND NOT A CALLBACK ARGUMENT.  The pipeline is five stages spread
over three scripts and a dozen library modules, and the interesting numbers —
which stroke certified, which arm got it, how long the balancer spent moving
nothing — are produced three and four calls below the front door.  Threading a
callback down to them would mean editing every signature in between, and every
signature in between is part of the published API that the CSAIL scripts, the
bench and the tests all call.  A module-level sink is the one change that
reaches the inner loops without touching what anything means.

THE CONTRACT IS THAT THE PLANNER DOES NOT NOTICE.  With no sink attached
`emit()` is a global load and a comparison against None, and every call site is
written so that the payload is not BUILT when there is no sink to hand it to
(see `active()` and the `stage()` context manager, whose payload callables are
only invoked once someone is listening).  A run with the GUI off must produce
byte-identical outputs to a run before this module existed; `tests/test_progress.py`
pins the no-op half of that and `docs/VIEWER.md` records the diff that pins the
other half.

THE SINK IS PROCESS-GLOBAL AND THAT IS DELIBERATE.  A planning job runs in its
own subprocess (`aris_sixarm.gui.worker`), which attaches exactly one sink at
startup and never removes it, so there is no ambiguity about which job an event
belongs to and no lock to contend for in the inner loop.  In-process users
(tests, notebooks) get `recording()`, which restores the previous sink.

EVENT SHAPE.  Every event is a plain dict, JSON-round-trippable:

    {"seq": int,          monotonic per-process counter, gapless
     "t":   float,        seconds since the sink was attached
     "wall": float,       time.time(), for correlating with logs
     "kind": str,         one of KINDS below
     "stage": str,        one of STAGES below, or "" for job-level events
     "payload": dict}     typed per (kind, stage); see docs/VIEWER.md

`job_id` is NOT carried per event: it is a property of the sink, and the server
stamps it on once when the event leaves the worker.  Putting it in every event
would be 400 bytes of the same string per stroke.
"""
from __future__ import annotations

import time
from contextlib import contextmanager

# --------------------------------------------------------------------------
# the vocabulary — the viewer switches on these, so they are a schema
# --------------------------------------------------------------------------
STAGES = (
    "day1",         # scripts/day1.py: one line, or the word re-checked on site
    "trace",        # picture -> pixel strokes -> strokes on the sheet
    "placement",    # the rotation x scale x translation search
    "allocation",   # probe, cover, repair, balance, split, merge, sequence
    "conduction",   # per-phase timelines, idle policy, execution profiles
    "scene_check",  # the inter-arm / paper / frame veto
)

KINDS = (
    "job_start",     # payload: params
    "job_end",       # payload: {ok, error?, elapsed_s, out_files}
    "stage_start",   # payload: stage-specific, e.g. {n_strokes}
    "stage_end",     # payload: {elapsed_s, ok, ...stage-specific}
    "substage_start",  # payload: {sub}            — a named phase inside a stage
    "substage_end",    # payload: {sub, elapsed_s, ok}
    "progress",      # payload: {done, total, label?}  — a loop tick
    "item",          # payload: one unit of work finished; see docs/VIEWER.md
    "metric",        # payload: {name, value, unit?}   — a scalar worth a gauge
    "log",           # payload: {level, msg}
    "artifact",      # payload: {name, path}           — a file appeared on disk
)

# --------------------------------------------------------------------------
# the sink
# --------------------------------------------------------------------------
_sink = None            # callable(event: dict) -> None, or None
_seq = 0
_t0 = 0.0


def set_sink(fn):
    """Attach a sink. -> the previous one (None if there was none).

    `fn` is called with one dict per event, from whatever thread emitted it.
    It MUST NOT raise: an exception here would propagate into the planner and
    turn a reporting failure into a planning failure.  `emit` guards anyway,
    but a sink that swallows its own errors keeps the event stream honest
    about which events were dropped.
    """
    global _sink, _seq, _t0
    prev = _sink
    _sink = fn
    if fn is not None and prev is None:
        _seq = 0
        _t0 = time.monotonic()
    return prev


def clear_sink():
    global _sink
    _sink = None


def active():
    """True when someone is listening.

    Guard any payload that costs something to build:

        if progress.active():
            progress.emit("item", "allocation", arm=a, span=_expensive(x))
    """
    return _sink is not None


def elapsed():
    """Seconds since the sink was attached (0.0 with no sink)."""
    return time.monotonic() - _t0 if _sink is not None else 0.0


def emit(kind, stage="", **payload):
    """Send one event.  A no-op — two loads and a branch — with no sink."""
    if _sink is None:
        return
    global _seq
    _seq += 1
    ev = {"seq": _seq, "t": time.monotonic() - _t0, "wall": time.time(),
          "kind": kind, "stage": stage, "payload": payload}
    try:
        _sink(ev)
    except Exception:
        # A broken pipe to a browser that closed its tab is not a planning
        # error.  Drop the sink rather than raising into the inner loop, and
        # never let the same failure be raised twice.
        pass


# convenience wrappers, all no-ops without a sink -------------------------
def log(msg, level="info", stage=""):
    if _sink is not None:
        emit("log", stage, level=level, msg=str(msg))


def metric(name, value, stage="", unit=None):
    if _sink is not None:
        emit("metric", stage, name=name, value=value, unit=unit)


def progress(stage, done, total=None, label=None):
    if _sink is not None:
        emit("progress", stage, done=int(done),
             total=(None if total is None else int(total)), label=label)


def item(stage, **payload):
    if _sink is not None:
        emit("item", stage, **payload)


def artifact(name, path, stage=""):
    if _sink is not None:
        emit("artifact", stage, name=name, path=str(path))


@contextmanager
def stage(name, **payload):
    """Bracket a stage with stage_start / stage_end and its wall time.

    The yielded dict is the stage_end payload under construction: a stage that
    learns something worth reporting (how many strokes it produced, how many
    certified) writes it in, and it rides out on the closing event.  On an
    exception the stage_end still goes out, with `error` set, so the viewer's
    timeline never shows a stage that started and never finished.

        with progress.stage("trace", source=path) as st:
            strokes = do_the_work()
            st["n_strokes"] = len(strokes)
    """
    if _sink is None:
        yield {}
        return
    t0 = time.monotonic()
    emit("stage_start", name, **payload)
    end = {}
    try:
        yield end
    except BaseException as exc:
        end["error"] = f"{type(exc).__name__}: {exc}"
        emit("stage_end", name, elapsed_s=time.monotonic() - t0, ok=False, **end)
        raise
    emit("stage_end", name, elapsed_s=time.monotonic() - t0, ok=True, **end)


@contextmanager
def substage(stage_name, label, **payload):
    """A named phase INSIDE a stage, timed the same way.

    This is the one that drives the algorithm iteration: `docs/FAST_PLANNING.md`
    found 93 % of an allocation inside `balance` only by timing the six
    sub-phases separately, and the viewer draws exactly these as the
    "where the time goes" bar.
    """
    if _sink is None:
        yield {}
        return
    t0 = time.monotonic()
    emit("substage_start", stage_name, sub=label, **payload)
    end = {}
    try:
        yield end
    except BaseException as exc:
        end["error"] = f"{type(exc).__name__}: {exc}"
        emit("substage_end", stage_name, sub=label,
             elapsed_s=time.monotonic() - t0, ok=False, **end)
        raise
    emit("substage_end", stage_name, sub=label,
         elapsed_s=time.monotonic() - t0, ok=True, **end)


# --------------------------------------------------------------------------
# in-process recording, for tests and notebooks
# --------------------------------------------------------------------------
@contextmanager
def recording():
    """Collect events into a list, restoring whatever sink was there before.

        with progress.recording() as events:
            run_the_thing()
        assert [e["kind"] for e in events] == [...]
    """
    events = []
    prev = set_sink(events.append)
    try:
        yield events
    finally:
        set_sink(prev)
