"""The subprocess that actually plans.  One job, one process, one event log.

    python -m aris_sixarm.gui.worker out/gui_jobs/<job_id>

Reads `job.json`, attaches a `aris_sixarm.progress` sink that appends one JSON
line per event to `events.jsonl`, turns the parameters into the argument list
`scripts/draw.py` already takes, and calls its `main()`.  Nothing is
re-implemented here: the GUI runs the SAME front door a person runs from a
terminal, so a run started from the browser and a run started from a shell
differ in exactly one thing, which is that somebody is listening.

THE PLANNER'S STDOUT IS PART OF THE PROGRESS.  Every stage of this pipeline
prints a great deal, and most of it is the good stuff — which arm gave a span
back and why, what the balancer moved, what the sequencer saved.  Re-deriving
that into typed events would be a second implementation of the same reporting,
so instead stdout is teed: it goes to `log.txt` verbatim AND becomes one `log`
event per line, interleaved in `events.jsonl` in the order it happened relative
to the typed events.  The browser gets a scrolling log that is the planner's
own words, with the typed events as the structure over them.
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------
# the sink
# --------------------------------------------------------------------------
class EventWriter:
    """Append-only JSONL, one line per event, flushed every time.

    Flushing per event is the point: a viewer tailing this file is the only
    way anybody sees a stage that is still running, and a buffered writer would
    deliver the whole allocation in one burst after it finished — which is
    precisely the case the GUI exists to make visible.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.f = open(self.path, "a", buffering=1)      # line buffered

    def __call__(self, ev):
        try:
            # STRICT JSON, BECAUSE THE READER IS A BROWSER.  `json.dumps`
            # happily writes `Infinity` and `NaN`; `JSON.parse` refuses them,
            # and one such value in one event kills the whole batch the
            # websocket delivers — measured: an arm whose bag has no feasible
            # tour reports `transit_s = inf` from `sequence.solve`, and the
            # GUI's entire stage timeline went blank for the rest of the run.
            # The strict attempt is the fast path; only a payload that
            # actually contains a non-finite number pays for the walk.
            try:
                line = json.dumps(ev, default=_jsonable, allow_nan=False)
            except ValueError:
                line = json.dumps(_finite(ev), default=_jsonable,
                                  allow_nan=False)
            self.f.write(line + "\n")
        except Exception as exc:                        # never raise at a caller
            try:
                self.f.write(json.dumps(
                    {"seq": ev.get("seq", -1), "t": ev.get("t", 0.0),
                     "wall": time.time(), "kind": "log", "stage": "",
                     "payload": {"level": "error",
                                 "msg": f"event dropped: {type(exc).__name__}: "
                                        f"{exc}"}}) + "\n")
            except Exception:
                pass

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass


def _finite(x):
    """One recursive walk that resolves numpy AND kills non-finite floats.

    It cannot delegate to `jobs.finite`: that one is for events read back from
    disk, where every value is already a plain JSON type.  Here the sink is
    handed whatever a call site passed, and `np.float32` is not a `float` to
    `isinstance` (though `np.float64` is), so a nested one would sail past a
    float-only check and straight into the encoder.
    """
    import math
    try:
        import numpy as np
        if isinstance(x, np.generic):
            x = x.item()
        elif isinstance(x, np.ndarray):
            x = x.tolist()
    except Exception:
        pass
    if isinstance(x, float):
        return None if not math.isfinite(x) else x
    if isinstance(x, dict):
        return {str(k): _finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_finite(v) for v in x]
    return x


def _jsonable(x):
    """numpy and Path survive the trip; anything else becomes its repr."""
    try:
        import numpy as np
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
    except Exception:
        pass
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (set, frozenset)):
        return sorted(x)
    return repr(x)


class Tee(io.TextIOBase):
    """A stdout that is also a stream of `log` events, line by line."""

    def __init__(self, raw, emit, level="info"):
        self.raw, self.emit, self.level = raw, emit, level
        self.buf = ""

    def write(self, s):
        self.raw.write(s)
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            if line.strip():
                self.emit(line.rstrip(), self.level)
        return len(s)

    def flush(self):
        self.raw.flush()

    def isatty(self):
        return False


# --------------------------------------------------------------------------
# parameters -> the front door's argument list
# --------------------------------------------------------------------------
# (json key, flag, kind).  "flag" args pass the flag when the value is truthy;
# "value" args pass `--flag VALUE`; "pair" args pass two numbers.  Anything the
# form does not send keeps `scripts/draw.py`'s own default, so the GUI cannot
# silently move a planner constant by having a stale default of its own.
ARGSPEC = [
    ("out", "--out", "value"),
    ("title", "--title", "value"),
    ("outdir", "--outdir", "value"),
    # ---- trace ----
    ("inks", "--inks", "value"),
    ("work_px", "--work-px", "value"),
    ("min_len", "--min-len", "value"),
    ("rdp", "--rdp", "value"),
    ("min_px", "--min-px", "value"),
    ("fill_erode", "--fill-erode", "value"),
    ("no_bridge", "--no-bridge", "flag"),
    ("trace_only", "--trace-only", "flag"),
    # ---- placement ----
    ("placement", "--placement", "value"),
    ("rotate", "--rotate", "value"),
    ("scales", "--scales", "triple"),
    ("search_offset", "--search-offset", "value"),
    ("offset_step", "--offset-step", "value"),
    ("top", "--top", "value"),
    ("slack", "--slack", "value"),
    ("radius", "--radius", "value"),
    ("jobs", "--jobs", "value"),
    ("place_only", "--place-only", "flag"),
    ("target_width", "--target-width", "value"),
    ("offset", "--offset", "pair"),
    ("margin", "--margin", "value"),
    # ---- allocation ----
    ("arms", "--arms", "value"),
    ("atlas", "--atlas", "value"),
    ("arm_phases", "--arm-phases", "value"),
    ("arm_phase_near", "--arm-phase-near", "value"),
    ("max_probes", "--max-probes", "value"),
    ("tilt_max_deg", "--tilt-max-deg", "value"),
    ("draw_speed", "--draw-speed", "value"),
    ("transit_speed", "--transit-speed", "value"),
    ("sequencer", "--sequencer", "value"),
    ("band_objective", "--band-objective", "value"),
    ("min_split", "--min-split", "value"),
    ("pens", "--pens", "value"),
    ("no_balance", "--no-balance", "flag"),
    ("no_split", "--no-split", "flag"),
    ("no_merge", "--no-merge", "flag"),
    ("no_rrt", "--no-rrt", "flag"),
    ("rrt_budget", "--rrt-budget", "value"),
    ("two_pass", "--two-pass", "flag"),
    ("cluster", "--cluster", "flag"),
    ("depot_hover_selective", "--depot-hover-selective", "flag"),
    ("no_multi_tour", "--no-multi-tour", "flag"),
    ("residual_passes", "--residual-passes", "value"),
    ("residual_min_gain", "--residual-min-gain", "value"),
    # ---- conduction ----
    ("fps", "--fps", "value"),
    ("substeps", "--substeps", "value"),
    ("subcheck", "--subcheck", "value"),
    ("qd_frac", "--qd-frac", "value"),
    ("idle_policy", "--idle-policy", "value"),
    ("min_coverage", "--min-coverage", "value"),
    ("require_full", "--require-full", "flag"),
    ("skip_unconductable", "--skip-unconductable", "flag"),
    ("freeze_all_phases", "--freeze-all-phases", "flag"),
    ("freeze_refused_phase", "--freeze-refused-phase", "flag"),
    ("aside_parks", "--aside-parks", "flag"),
    ("reseq_tries", "--reseq-tries", "value"),
    ("pause", "--pause", "value"),
    ("select_profile", "--select-profile", "flag"),
    ("profile_jobs", "--profile-jobs", "value"),
    ("conduct_jobs", "--conduct-jobs", "value"),
    ("image_jobs", "--image-jobs", "value"),
    ("no_verify", "--no-verify", "flag"),
    ("verbose", "--verbose", "flag"),
]


def build_argv(params):
    """Job parameters -> `scripts/draw.py` argv.  -> list[str].

    Explicit, and a whitelist: a key the form invents does not become a flag,
    it becomes an error.  The alternative — passing unknown keys through —
    would let a typo in the browser turn into a silently different planner
    configuration, which is the class of bug `program_schema` exists to end.
    """
    p = dict(params)
    src = p.pop("source", None)
    if not src:
        raise ValueError("no source picture given")
    known = {k for k, _, _ in ARGSPEC} | {"rig", "tool", "no_anim", "note"}
    unknown = sorted(set(p) - known)
    if unknown:
        raise ValueError(f"unknown job parameters: {unknown}")
    argv = [str(src)]
    for key, flag, kind in ARGSPEC:
        if key not in p:
            continue
        v = p[key]
        if v is None or v == "":
            continue
        if kind == "flag":
            if v:
                argv.append(flag)
        elif kind == "pair":
            argv += [flag, str(v[0]), str(v[1])]
        elif kind == "triple":
            argv += [flag, str(v[0]), str(v[1]), str(v[2])]
        else:
            argv += [flag, str(v)]
    # The animation is rendered by a pydrake venv that a GUI machine need not
    # have, and the browser viewer replaces it anyway.  It is off unless the
    # job explicitly asks.
    if not p.get("no_anim", True):
        pass
    else:
        argv.append("--no-anim")
    return argv


# --------------------------------------------------------------------------
def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    job_dir = Path(argv[0]).resolve()
    meta = json.loads((job_dir / "job.json").read_text())
    params = meta.get("params", {})

    # EVERY ARTIFACT LANDS IN THE JOB DIRECTORY.  `scripts/draw.py` writes
    # eight files named after `--out` under `--outdir`, and left to its default
    # that is the shared `out/`, where two jobs planning the same picture would
    # overwrite each other's programme while both were running.  The job
    # directory is the job (see `jobs.py`), so the default belongs here and not
    # in the form.
    params.setdefault("outdir", str(job_dir))

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    from aris_sixarm import progress

    writer = EventWriter(job_dir / "events.jsonl")
    progress.set_sink(writer)

    def log_line(line, level):
        progress.emit("log", "", level=level, msg=line)

    sys.stdout = Tee(sys.stdout, log_line, "info")
    sys.stderr = Tee(sys.stderr, log_line, "error")

    t0 = time.time()
    ok, err = True, None
    try:
        args = build_argv(params)
        progress.emit("job_start", "", params=params, argv=args,
                      rig=os.environ.get("ARIS_RIG", "final"),
                      tool=os.environ.get("ARIS_TOOL", "inline"),
                      pid=os.getpid(), job_dir=str(job_dir))
        import draw                                       # scripts/draw.py
        draw.main(args)
    except SystemExit as exc:
        # `scripts/draw.py` and the conductor refuse with SystemExit and a
        # sentence that says why (coverage below the gate, scene_check's veto).
        # That is a RESULT, not a crash, and it is reported as one.
        code = exc.code
        ok = code in (0, None)
        err = None if ok else str(code)
    except BaseException as exc:
        ok, err = False, f"{type(exc).__name__}: {exc}"
        progress.emit("log", "", level="error", msg=traceback.format_exc())
    finally:
        files = sorted(p.name for p in job_dir.iterdir()) \
            if job_dir.exists() else []
        bundle_err = None
        if ok:
            try:
                bundle_err = _export_bundle(job_dir, params, progress)
            except BaseException as exc:
                bundle_err = f"{type(exc).__name__}: {exc}"
                progress.emit("log", "", level="error",
                              msg=traceback.format_exc())
        progress.emit("job_end", "", ok=bool(ok), error=err,
                      bundle_error=bundle_err,
                      elapsed_s=round(time.time() - t0, 3), files=files)
        progress.clear_sink()
        sys.stdout.flush()
        writer.close()
    return 0 if ok else 1


def _export_bundle(job_dir, params, progress):
    """Turn what the run wrote into the one file the viewer reads. -> error|None.

    In-process rather than as a second subprocess, because the rig and the tool
    are already right here and nowhere else: the bundle carries per-arm base
    transforms, and re-deriving those under a default rig is exactly the kind of
    quietly-wrong record `program_schema` was written to make impossible.
    """
    from aris_sixarm.program_schema import export_bundle
    stem = params.get("out") or Path(str(params.get("source"))).stem
    npz = job_dir / f"{stem}_schedule.npz"
    prog = job_dir / f"{stem}_program.json"
    if not npz.exists() or not prog.exists():
        return (f"no schedule to bundle ({npz.name} / {prog.name} not written) "
                "— a trace-only or place-only run has no programme")
    with progress.stage("export", npz=npz.name):
        b = export_bundle(npz, prog, job_dir / "bundle.json",
                          summary=job_dir / f"{stem}_schedule.json")
    progress.artifact("bundle", job_dir / "bundle.json")
    progress.metric("bundle_frames", b.meta.n_frames)
    return None


if __name__ == "__main__":
    sys.exit(main())
