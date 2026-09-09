"""A conducted schedule, in the shape a robot executor needs.

`scripts/csail_schedule.py` ends with `np.savez_compressed` and a viewer; this
module is the other consumer of the same file.  It reads the conducted
timeline and re-expresses it as:

    FleetProgram -> [PhaseTrack] -> {arm: JointTrajectory}   +   [Barrier]

and NOTHING ELSE.  It plans nothing, re-times nothing and certifies nothing.
Everything here is a re-shaping of numbers `coordination` already produced and
`scene_check` already graded, which is the property that makes the executor
auditable: if the arm flies somewhere unexpected, the timeline it was given is
byte-for-byte the timeline the checker passed.

THREE THINGS A READER MUST KNOW BEFORE TRUSTING THIS FILE.

1. **The npz is DECIMATED and the certificate is not.**  `csail_schedule`
   conducts at `dt = 1/(fps * substeps)` — 1/48 s on every programme shipped —
   and writes every `substeps`-th frame, so the file is 24 Hz.  `scene_check`
   ran on the FULL-rate path.  Executing the 24 Hz samples therefore flies the
   chords between certified frames, not the certified frames' own path.  The
   chords are short (`writing.MAX_DQ_FRAME` = 0.04 rad bounds the conductor's
   own step and the swept-cell bound in `coordination` is subtracted rather
   than assumed) but they are NOT what was proved.  `from_schedule` records
   this in `warnings` on every load, and `DECIMATED_NOTE` is the one-line form.
   The fix is upstream and cheap: conduct with `--substeps 1`, or have
   `csail_schedule` write the un-strided `qtraj` beside the animation one.

2. **The clock is the fleet's.**  See `trajectory.Governor`.  A phase's tracks
   share `start_s`, and the arms are only mutually safe at equal program time.

3. **A pause is a BARRIER, not a gap.**  Between phases the npz holds every arm
   at its last pose for `pause_s` seconds while a human swaps pens.  That is
   the one place in the programme where the fleet is stationary by design, and
   it is therefore the only place an executor may legally stop, resynchronise,
   or hand control to a person.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .trajectory import JointTrajectory

DECIMATED_NOTE = (
    "the schedule npz is decimated by `stride` (the conductor ran at "
    "1/(fps*stride) s and scene_check graded THAT path); executing these "
    "samples flies the chords between certified frames")

BARRIER_KINDS = ("start", "pen_swap", "phase_end", "end")


@dataclass
class Barrier:
    """A program time at which the whole fleet must be stationary.

    `hold_q` is the configuration EVERY arm is expected to be holding, read out
    of the timeline itself rather than assumed to be the park pose — on v18 it
    is each arm's last pose of the outgoing phase, which the conductor's idle
    policy has already flown home.  An executor arriving at a barrier compares
    measured state against `hold_q` and refuses to continue on a mismatch:
    that comparison is the only place in the whole stack where the real robot's
    position is checked against the plan's, so it is where drift, a slipped
    pen, and a wrong arm mapping all surface.
    """
    index: int
    kind: str
    name: str
    t_s: float                       # program time on the fleet clock
    hold_q: dict = field(default_factory=dict)      # arm -> (7,)
    hold_s: float = 0.0              # how long the timeline stands still here
    requires_ack: bool = False       # a human must confirm before resuming
    note: str = ""
    #: where the NEXT phase begins.  Usually `hold_q`; on v18 it is not, for
    #: two arms, and that discrepancy is the reason this field exists — see
    #: `reposition`.
    resume_q: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in BARRIER_KINDS:
            raise ValueError(f"barrier kind {self.kind!r} not in {BARRIER_KINDS}")
        self.hold_q = {int(a): np.asarray(q, float).reshape(7)
                       for a, q in self.hold_q.items()}
        self.resume_q = {int(a): np.asarray(q, float).reshape(7)
                         for a, q in self.resume_q.items()}

    def reposition(self, tol=1e-6):
        """{arm: q} the fleet must be MOVED to before the clock restarts.

        MEASURED, NOT HYPOTHETICAL.  The conductor lays the phases end to end
        and holds each arm at its outgoing pose through the pause, but the
        incoming phase starts from whatever ITS first frame is, and those are
        not always the same configuration: on `csail_schedule_h094_v18` arm 31
        differs by 0.0320 rad across the first pen swap and arm 2 by 0.0329 rad
        across the second.

        In the animation that is one invisible frame.  On a robot it is a step
        command into a stiff controller, and it is motion NOTHING CERTIFIED —
        `scene_check` graded the frames on either side of the gap, not the move
        between them.  So the executor treats it as an explicit, supervised
        `goto` at barrier speed rather than as part of the stream, and this
        method is the list of arms that need one.
        """
        out = {}
        for a, q in sorted(self.resume_q.items()):
            h = self.hold_q.get(a)
            if h is None or np.abs(np.asarray(q) - h).max() > tol:
                out[a] = np.asarray(q, float)
        return out

    def mismatch(self, measured, tol=0.02):
        """-> [] when every arm is where the barrier says.  Else what is off.

        `tol` is radians per joint.  0.02 rad is ~1.1 deg, about 2 mm at the
        pen for a wrist joint and 15 mm for joint 1 — loose enough that a
        settled arm passes, tight enough that a wrong pose does not.
        """
        bad = []
        for a, q in sorted(self.hold_q.items()):
            m = measured.get(a)
            if m is None:
                bad.append(f"barrier {self.name}: arm {a} did not report a state")
                continue
            d = np.abs(np.asarray(m, float).reshape(7) - q)
            if d.max() > tol:
                j = int(np.argmax(d))
                bad.append(f"barrier {self.name}: arm {a} joint {j + 1} is "
                           f"{d[j]:.4f} rad off the held pose (tol {tol:g})")
        return bad


@dataclass
class PhaseTrack:
    """One conducted phase: every arm's slice of it, on the phase's own clock."""
    index: int
    name: str
    ink: str | None
    start_s: float                   # on the FLEET clock
    duration_s: float
    tracks: dict = field(default_factory=dict)      # arm -> JointTrajectory
    drawing: dict = field(default_factory=dict)     # arm -> bool
    n_segments: dict = field(default_factory=dict)  # arm -> int

    @property
    def arms(self):
        return sorted(self.tracks)

    @property
    def drawing_arms(self):
        return sorted(a for a in self.tracks if self.drawing.get(a))


@dataclass
class FleetProgram:
    """Everything an executor needs, and nothing it must not be trusted with."""
    name: str
    source: str
    rig: str
    tool: str
    sheet_m: tuple
    fps: float
    conductor_dt: float              # 1/(fps*stride) — what scene_check graded
    stride: int
    arms: list
    pen_ext: dict
    phases: list
    barriers: list
    margin_m: float = 0.0
    min_clearance_m: float = 0.0
    pause_s: float = 0.0
    warnings: list = field(default_factory=list)

    # -- the whole programme on one clock ----------------------------------
    @property
    def duration_s(self):
        return max((p.start_s + p.duration_s for p in self.phases), default=0.0)

    def track(self, arm_id):
        """One arm's WHOLE programme, pauses included, on the fleet clock.

        The pauses are represented honestly — two samples holding the same
        configuration — rather than skipped, because a controller streaming
        this array must keep commanding through the swap or the arm goes limp
        in front of the person changing its pen.
        """
        ts, qs = [], []
        for p in self.phases:
            tr = p.tracks.get(arm_id)
            if tr is None or not len(tr):
                continue
            if ts and p.start_s > ts[-1][-1] + 1e-9:      # a barrier hold
                # ONE FRAME before the new phase, not an epsilon: the held pose
                # and the incoming pose are not always equal (see
                # `Barrier.reposition`), and an epsilon-wide step between two
                # different configurations is an infinite commanded velocity
                # that the envelope check would then report as the programme's
                # rather than as this function's.
                ts.append(np.array([p.start_s - 1.0 / self.fps]))
                qs.append(qs[-1][-1:][:])
            ts.append(tr.t + p.start_s)
            qs.append(tr.q)
        if not ts:
            return JointTrajectory(arm_id, [0.0], np.zeros((1, 7)),
                                   source=f"{self.name} (idle)")
        t = np.concatenate(ts)
        q = np.concatenate(qs)
        keep = np.concatenate([[True], np.diff(t) > 0])
        return JointTrajectory(arm_id, t[keep], q[keep],
                               dt_nominal=0.0, source=f"{self.name} whole")

    def barrier_before(self, t_s):
        prev = [b for b in self.barriers if b.t_s <= t_s + 1e-9]
        return prev[-1] if prev else None

    # -- gates --------------------------------------------------------------
    def check(self, **kw):
        """Every arm's trajectory against the joint envelope. -> [problems]."""
        bad = []
        for p in self.phases:
            for a in p.arms:
                bad += [f"phase {p.index} ({p.name}): {m}"
                        for m in p.tracks[a].check(**kw)]
        seen = set()
        for b in self.barriers:
            if b.t_s in seen:
                bad.append(f"two barriers at t = {b.t_s:.4f} s")
            seen.add(b.t_s)
        for p in self.phases:
            for a in p.arms:
                if len(p.tracks[a]) and abs(p.tracks[a].duration_s
                                            - p.duration_s) > 2.0 / self.fps:
                    bad.append(f"phase {p.index}: arm {a} track is "
                               f"{p.tracks[a].duration_s:.3f} s against the "
                               f"phase's {p.duration_s:.3f} s — the arms are "
                               "not on one clock")
        return bad

    # -- the single-arm rungs ------------------------------------------------
    def solo(self, arm_id, phase=None):
        """This arm's track, plus the poses the OTHER five must be holding.

        THIS IS NOT A CERTIFICATE AND THE RETURN VALUE SAYS SO.  The conducted
        timeline is certified with all six arms MOVING; freezing five of them
        at their phase-start poses is a different scene, and inter-arm
        clearance for it has never been evaluated.  What comes back is the
        material for that evaluation — the mover's path and the frozen set —
        and `recheck_required` is True on every one of them.  The re-check is
        `scripts/recheck_timeline.py`'s job, against a timeline in which the
        five are constant.
        """
        phases = self.phases if phase is None else [self.phases[phase]]
        others = {}
        for p in phases:
            for a in p.arms:
                if a != arm_id and len(p.tracks[a]):
                    others.setdefault(a, p.tracks[a].q[0])
        return SoloProgram(
            parent=self.name, arm_id=int(arm_id),
            phases=[PhaseTrack(p.index, p.name, p.ink, p.start_s, p.duration_s,
                               {arm_id: p.tracks[arm_id]},
                               {arm_id: p.drawing.get(arm_id, False)},
                               {arm_id: p.n_segments.get(arm_id, 0)})
                    for p in phases if arm_id in p.tracks],
            others_hold={a: np.asarray(q, float) for a, q in others.items()},
            recheck_required=True)

    # -- reporting ----------------------------------------------------------
    def report(self):
        out = [f"{self.name}  [{self.source}]",
               f"  rig {self.rig}, tool {self.tool}, sheet "
               f"{self.sheet_m[0]:.4f} x {self.sheet_m[1]:.4f} m",
               f"  {len(self.arms)} arms, {len(self.phases)} phases, "
               f"{len(self.barriers)} barriers, {self.duration_s:.2f} s at "
               f"{self.fps:g} fps (conductor dt {1000 * self.conductor_dt:.2f} "
               f"ms, stride {self.stride})",
               f"  scene_check margin {1000 * self.margin_m:.0f} mm, worst "
               f"clearance {1000 * self.min_clearance_m:.1f} mm"]
        for p in self.phases:
            out.append(f"  phase {p.index} {p.name!r} ink={p.ink} "
                       f"t {p.start_s:8.3f} + {p.duration_s:7.3f} s  "
                       f"drawing {p.drawing_arms}")
        for b in self.barriers:
            out.append(f"  barrier {b.index} {b.kind:<10s} t {b.t_s:8.3f} s  "
                       f"hold {b.hold_s:5.2f} s"
                       + ("  ACK REQUIRED" if b.requires_ack else "")
                       + (f"  — {b.note}" if b.note else ""))
        for w in self.warnings:
            out.append(f"  ! {w}")
        return out

    def summary(self):
        """A JSON-round-trippable dict for the run log.  No arrays."""
        return dict(
            name=self.name, source=self.source, rig=self.rig, tool=self.tool,
            sheet_m=list(self.sheet_m), fps=self.fps,
            conductor_dt=self.conductor_dt, stride=self.stride,
            arms=list(self.arms), pen_ext={str(k): v for k, v in self.pen_ext.items()},
            duration_s=self.duration_s, margin_m=self.margin_m,
            min_clearance_m=self.min_clearance_m, pause_s=self.pause_s,
            phases=[dict(index=p.index, name=p.name, ink=p.ink,
                         start_s=p.start_s, duration_s=p.duration_s,
                         arms=p.arms, drawing=p.drawing_arms,
                         n_samples={str(a): len(p.tracks[a]) for a in p.arms})
                    for p in self.phases],
            barriers=[dict(index=b.index, kind=b.kind, name=b.name, t_s=b.t_s,
                           hold_s=b.hold_s, requires_ack=b.requires_ack,
                           note=b.note, arms=sorted(b.hold_q))
                      for b in self.barriers],
            warnings=list(self.warnings))


@dataclass
class SoloProgram:
    """One arm's phases, and the frozen pose set they must be re-checked against."""
    parent: str
    arm_id: int
    phases: list
    others_hold: dict
    recheck_required: bool = True

    @property
    def duration_s(self):
        return max((p.start_s + p.duration_s for p in self.phases), default=0.0)

    def report(self):
        return ([f"SOLO arm {self.arm_id} of {self.parent!r}: "
                 f"{len(self.phases)} phase(s), {self.duration_s:.2f} s",
                 f"  the other arms must be VERIFIED AT: "
                 + ", ".join(f"{a}" for a in sorted(self.others_hold))]
                + (["  ! NOT CERTIFIED: the conducted timeline moves all six "
                    "arms; a frozen fleet is a different scene and needs "
                    "scripts/recheck_timeline.py against it"]
                   if self.recheck_required else []))


# ---------------------------------------------------------------------------
# the reader
# ---------------------------------------------------------------------------
def from_schedule(npz_path, program_path=None, name=None):
    """`csail_schedule.py`'s npz (+ its program json) -> a `FleetProgram`.

    `program_path` is optional and adds only labels — phase names, the rig and
    the tool the run used.  Everything load-bearing comes out of the npz, which
    is the file `scene_check` graded.
    """
    npz_path = Path(npz_path)
    z = np.load(npz_path, allow_pickle=False)
    prog = json.loads(Path(program_path).read_text()) if program_path else {}

    fps = float(z["fps"])
    stride = int(z["stride"])
    conductor_dt = float(z["dt"])
    arms = [int(a) for a in z["arms"]]
    drawing_arms = {int(a) for a in z["drawing_arms"]}
    pen_ext = {int(a): float(v) for a, v in zip(arms, z["pen_ext"])}
    ph = np.asarray(z["phase"], int)
    ink_of = [str(x) for x in z["phase_ink"]]
    pause_s = float(z.get("pause_s", 0.0))

    warnings = []
    if stride > 1:
        warnings.append(f"stride = {stride}: " + DECIMATED_NOTE)
    tool = "lateral" if max(pen_ext.values(), default=0.0) < 0.09 else "inline"
    tool = str(prog.get("tool", tool))

    # -- the phases, taken from the `phase` array rather than from the header.
    # `phase_start_s` is written in FULL-rate units and the animation frames
    # are strided, so the two disagree by up to one conductor step; the array
    # is the one that says which frame belongs to which phase, so it wins and
    # the header is only cross-checked.
    phases, bounds = [], []
    for k in range(int(z["n_phases"])):
        idx = np.flatnonzero(ph == k)
        if not len(idx):
            continue
        i0, i1 = int(idx[0]), int(idx[-1])
        if i1 - i0 + 1 != len(idx):
            warnings.append(f"phase {k} is not contiguous in the timeline")
        bounds.append((k, i0, i1))
        t = (np.arange(i0, i1 + 1) - i0) / fps
        pj = (prog.get("phases") or [{}] * (k + 1))[k] if k < len(
            prog.get("phases") or []) else {}
        tracks, draw, nseg = {}, {}, {}
        for a in arms:
            q = np.asarray(z[f"q_{a}"][i0:i1 + 1], float)
            tracks[a] = JointTrajectory(
                a, t, q, dt_nominal=1.0 / fps,
                source=f"{npz_path.name}:phase{k}")
            seg = np.asarray(z[f"seg_{a}"][i0:i1 + 1], int)
            draw[a] = a in drawing_arms and bool((seg >= 0).any())
            nseg[a] = int(len(np.unique(seg[seg >= 0])))
        phases.append(PhaseTrack(
            index=k, name=str(pj.get("name", f"phase {k}")),
            ink=ink_of[k] if k < len(ink_of) else None,
            start_s=i0 / fps, duration_s=(i1 - i0) / fps,
            tracks=tracks, drawing=draw, n_segments=nseg))

    # -- the barriers.  One at the start, one in every gap between phases, one
    # at the end.  A gap whose incoming and outgoing ink differ is a PEN SWAP
    # and needs a person, which is what `requires_ack` means.
    barriers, n = [], 0
    if phases:
        barriers.append(Barrier(
            n, "start", "start", 0.0,
            {a: phases[0].tracks[a].q[0] for a in arms}, 0.0, True,
            "every arm must be at its first commanded pose before the clock "
            "starts; this is the only pose the executor may drive to freely"))
        n += 1
    for (k, _, i1), (k2, j0, _) in zip(bounds, bounds[1:]):
        gap = (j0 - i1 - 1) / fps
        swap = phases[k].ink != phases[k2].ink
        b = Barrier(
            n, "pen_swap" if swap else "phase_end",
            f"{'swap' if swap else 'end'}-of-phase-{k}", (i1) / fps,
            {a: phases[k].tracks[a].q[-1] for a in arms}, gap, swap,
            (f"ink {phases[k].ink} -> {phases[k2].ink}: a human changes the "
             f"pens; the arms hold their last pose for {gap:.2f} s")
            if swap else f"phases {k} -> {k2}, {gap:.2f} s of hold",
            resume_q={a: phases[k2].tracks[a].q[0] for a in arms})
        rep = b.reposition(tol=1e-9)
        if rep:
            worst = max(float(np.abs(v - b.hold_q[a]).max())
                        for a, v in rep.items())
            b.note += (f"; arms {sorted(rep)} must be REPOSITIONED here "
                       f"(worst {worst:.4f} rad) — the outgoing hold pose and "
                       "the incoming phase's first pose differ and the move "
                       "between them is not in the certified timeline")
            warnings.append(
                f"barrier {b.name}: {len(rep)} arm(s) step up to {worst:.4f} "
                "rad across the pause; the executor repositions them under a "
                "supervised goto, and that move is UNCERTIFIED")
        barriers.append(b)
        n += 1
    if phases:
        barriers.append(Barrier(
            n, "end", "end", phases[-1].start_s + phases[-1].duration_s,
            {a: phases[-1].tracks[a].q[-1] for a in arms}, 0.0, False,
            "the programme is over; the arms hold until something stops them"))

    fp = FleetProgram(
        name=name or str(prog.get("name") or npz_path.stem),
        source=str(npz_path), rig=str(prog.get("rig", "?")), tool=tool,
        sheet_m=tuple(float(x) for x in z["sheet"]), fps=fps,
        conductor_dt=conductor_dt, stride=stride, arms=arms, pen_ext=pen_ext,
        phases=phases, barriers=barriers,
        margin_m=float(z.get("margin", 0.0)),
        min_clearance_m=float(z.get("min_clearance", 0.0)),
        pause_s=pause_s, warnings=warnings)

    hdr = np.asarray(z["phase_start_s"], float)
    for p, h in zip(phases, hdr):
        if abs(p.start_s - float(h)) > 2 * conductor_dt:
            fp.warnings.append(
                f"phase {p.index}: the `phase` array puts the start at "
                f"{p.start_s:.4f} s and the header at {float(h):.4f} s — "
                "more than one conductor step apart")
    return fp
