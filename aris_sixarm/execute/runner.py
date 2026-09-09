"""The loop: one `FleetProgram`, one `Backend`, one clock.

Small on purpose.  Every decision that could hurt somebody is made BEFORE the
loop starts (`preflight`) or AT a barrier (`_at_barrier`), and the loop itself
does exactly three things per tick: advance the governor, sample every arm at
the same program time, and hand the whole fleet to the backend in one call.

WHY THE BARRIER LOGIC LOOKS PARANOID.  A barrier is the only moment the plan
and the world are compared — everywhere else the executor is open loop, because
a conducted timeline IS open loop and re-planning against measured state would
be flying a path nothing certified.  So the barrier does all the checking that
the rest of the run does not: it stops the clock EARLY ENOUGH to stop (the
governor ramps, so the stop distance is finite and known), it verifies every
arm against the pose the timeline says it should be holding, and it will not
resume through a mismatch or through an un-acknowledged pen swap.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .backends import RateKeeper
from .trajectory import Governor


@dataclass
class RunLog:
    """What happened.  Written beside every run, hardware or not."""
    program: str
    backend: str
    ok: bool = False
    reason: str = ""
    t_program_s: float = 0.0
    t_wall_s: float = 0.0
    frames: int = 0
    barriers_passed: list = field(default_factory=list)
    #: [(barrier name, {arm: max |dq| rad})] — moves made at a barrier that
    #: the certified timeline does not contain.  Empty is the good case.
    repositioned: list = field(default_factory=list)
    problems: list = field(default_factory=list)
    rate_report: str = ""

    def report(self):
        out = [f"run {self.program!r} on {self.backend}: "
               + ("OK" if self.ok else f"STOPPED — {self.reason}"),
               f"  {self.frames} frames, {self.t_program_s:.2f} s of programme "
               f"in {self.t_wall_s:.2f} s of wall clock",
               f"  barriers passed: {self.barriers_passed or '(none)'}"]
        for nm, d in self.repositioned:
            out.append(f"  REPOSITIONED at {nm}: " + ", ".join(
                f"arm {a} by {v:.4f} rad" for a, v in sorted(d.items()))
                + "  (uncertified move)")
        if self.rate_report:
            out.append(f"  {self.rate_report}")
        out += [f"  ! {p}" for p in self.problems]
        return out


def _confirm_never(barrier):
    """The default acknowledgement: refuse.  A pen swap needs a person."""
    return False


def confirm_console(barrier):
    """An acknowledgement a human types.  Passed in explicitly, never default."""
    ans = input(f"  barrier {barrier.name}: {barrier.note}\n"
                f"  type 'go' to resume: ")
    return ans.strip().lower() == "go"


def play(program, backend, *, hz=None, realtime=True, rate=1.0,
         arms=None, t_end=None, confirm=_confirm_never, goto_start=True,
         goto_duration_s=8.0, barrier_tol=0.02, skip_preflight=False,
         max_wall_s=None):
    """Drive `program` into `backend`.  -> `RunLog`.

    `hz` defaults to the programme's own frame rate, which is the rate its
    samples were certified at; asking for more only interpolates chords.
    `realtime=False` runs the loop as fast as it can and is what the tests use.
    `confirm` is called at every barrier with `requires_ack`; the default
    REFUSES, so a programme with a pen swap stops there unless a caller has
    supplied a human.
    """
    arms = list(program.arms if arms is None else arms)
    hz = float(program.fps if hz is None else hz)
    log = RunLog(program=program.name, backend=getattr(backend, "name",
                                                       type(backend).__name__))

    problems = [] if skip_preflight else backend.preflight(program)
    if problems:
        log.problems = list(problems)
        log.reason = "preflight refused the programme"
        return log

    tracks = {a: program.track(a) for a in arms}
    t_end = float(program.duration_s if t_end is None else t_end)
    barriers = [b for b in program.barriers if b.t_s <= t_end + 1e-9]

    backend.connect(arms)
    state = backend.read_state()

    # -- the start barrier is the one place a free move is allowed ---------
    start = barriers[0] if barriers and barriers[0].kind == "start" else None
    if start is not None:
        miss = start.mismatch(state, tol=barrier_tol)
        if miss and not goto_start:
            log.problems, log.reason = miss, "not at the start pose"
            return log
        if miss:
            for a in arms:
                backend.goto(a, tracks[a].sample(0.0), goto_duration_s)
            state = backend.read_state()
            miss = start.mismatch(state, tol=barrier_tol)
            if miss:
                log.problems, log.reason = miss, "goto did not reach the start"
                backend.stop(log.reason)
                return log
        if start.requires_ack and not confirm(start):
            log.reason = "the start barrier was not acknowledged"
            backend.stop(log.reason)
            return log
        log.barriers_passed.append(start.name)
        barriers = barriers[1:]

    gov = Governor(max_rate=float(rate))
    gov.resume()
    keeper = RateKeeper(hz, realtime=realtime)
    backend.start_stream(arms, hz)

    import time
    t_wall0 = time.perf_counter()
    pending = list(barriers)
    try:
        while gov.t_program < t_end - 1e-9 or pending:
            wall_dt = keeper.tick()
            nxt = pending[0] if pending else None
            if nxt is not None and (gov.t_program + gov.stop_distance_s
                                    >= nxt.t_s - 1e-9):
                gov.hold()
            gov.step(wall_dt)
            if nxt is not None:
                gov.t_program = min(gov.t_program, nxt.t_s)
            gov.t_program = min(gov.t_program, t_end)
            backend.send(gov.t_program,
                         {a: tracks[a].sample(gov.t_program) for a in arms})
            log.frames += 1

            if nxt is not None and gov.holding and \
                    gov.t_program >= nxt.t_s - 1e-9:
                ok, why = _at_barrier(backend, nxt, confirm, barrier_tol, log,
                                      goto_duration_s)
                if not ok:
                    log.reason = why
                    backend.stop(why)
                    log.t_program_s = gov.t_program
                    log.t_wall_s = time.perf_counter() - t_wall0
                    log.rate_report = keeper.report()
                    return log
                pending.pop(0)
                if nxt.kind != "end":
                    gov.resume()
            if gov.t_program >= t_end - 1e-9 and not pending:
                break
            if max_wall_s is not None and \
                    time.perf_counter() - t_wall0 > max_wall_s:
                log.reason = f"wall-clock budget {max_wall_s:g} s exhausted"
                backend.stop(log.reason)
                return log
    finally:
        log.t_program_s = gov.t_program
        log.t_wall_s = time.perf_counter() - t_wall0
        log.rate_report = keeper.report()

    log.ok = True
    backend.stop("")
    return log


def _at_barrier(backend, barrier, confirm, tol, log, goto_duration_s=8.0):
    """-> (may we continue, why not).

    The order is deliberate: VERIFY, then ask the human, then REPOSITION.  The
    reposition is a supervised move onto a pose the certified timeline does not
    connect to the held one (`Barrier.reposition`), so it must not happen
    before somebody has looked, and the verification must be against where the
    arms actually are rather than where they are about to go.
    """
    measured = backend.read_state()
    problems = [p for p in (backend.barrier(barrier, measured) or [])]
    if problems:
        log.problems += problems
        return False, f"barrier {barrier.name} refused: {problems[0]}"
    if barrier.requires_ack and not confirm(barrier):
        return False, f"barrier {barrier.name} was not acknowledged"
    rep = barrier.reposition(tol=1e-9)
    if rep:
        log.repositioned.append(
            (barrier.name, {int(a): float(np.abs(q - barrier.hold_q[a]).max())
                            for a, q in rep.items()}))
        for a, q in sorted(rep.items()):
            backend.goto(a, q, goto_duration_s)
        after = backend.read_state()
        miss = [f"barrier {barrier.name}: arm {a} did not reach the "
                f"reposition target"
                for a, q in sorted(rep.items())
                if a not in after
                or np.abs(np.asarray(after[a], float) - q).max() > tol]
        if miss:
            log.problems += miss
            return False, miss[0]
    log.barriers_passed.append(barrier.name)
    return True, ""


def dry_run(npz_path, program_path=None, **kw):
    """The convenience the ladder's rungs actually call. -> (program, RunLog)."""
    from .backends import MeshcatDryRun
    from .program import from_schedule
    prog = from_schedule(npz_path, program_path)
    be = MeshcatDryRun()
    starts = {a: prog.phases[0].tracks[a].q[0] for a in prog.arms} \
        if prog.phases else {}
    be._state = {a: np.asarray(q, float) for a, q in starts.items()}
    return prog, play(prog, be, **kw)
