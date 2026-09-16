"""Conducted fleet schedule -> per-arm pathway CSV v2 + manifest.

This is the planner side of `docs/ARIS2_CONTRACTS.md` §1: it turns the file the
conductor wrote (`out/<name>_schedule.npz` + `out/<name>_program.json`) into the
same CSV the deployed GUI produces and `rtff_pathway_exec.py` loads, plus the
seven optional joint columns.

    stroke_idx,wp_idx,kind,x_m,y_m,z_m,qx,qy,qz,qw,intensity,q1..q7

WHAT IS READ, AND FROM WHERE
----------------------------
`aris_sixarm.execute.program.from_schedule` parses the npz + program json into a
`FleetProgram`; that is the ONLY parser and nothing here re-implements it.  One
array it does not carry is `seg_<arm>` — the per-frame segment index, -1 when the
pen is up — and that array IS the pen state and the stroke id, so it is read
straight off the npz and its alignment with the `FleetProgram` tracks is
ASSERTED (`_arm_frames`), not assumed.

FRAMES
------
The schedule lives in the planner's canvas/world frame (z = 0 IS the paper
plane, `aris_sixarm.fleet`); the CSV lives in the ARM's own `fr3_link0`.  The
bridge is `ArmSpec.T_world_base()` — the same 4x4 `planner.build_lattice`,
`scene_check._chain` and `execute` use — and the exporter never applies it: the
row poses come from `frames.fk(q)`, which IS the base frame, so the CSV needs no
transform at all.  `T_world_base` is recorded in the manifest and used only for
`paper_z_base_m` (the world paper plane expressed in the base) and for the
h_inv-invariance assertion in `base_transform`.

POSE
----
Row pose = the EE frame the robot is configured with (`setEE` = NOMINAL pen
tip, contract §4):

    position    = frames.tip_pos(q) = T_tcp[:3,3] + R_tcp @ tool_offset()
    orientation = R_tcp, the hand-TCP rotation of `frames.fk`

The hand-TCP frame carries the flange's Rz(-pi/4) twist — the Franka Hand's own
nominal-end-effector frame, libfranka's `F_T_NE`, which `setEE` (`NE_T_EE`) then
extends to the pen tip without turning it.  So the repo's tool frame and the
deployed EE frame are the SAME frame and NO yaw conversion is applied.  The
EVIDENCE is not that argument, it is the measurement:
`tests/test_export_pathway.py::test_floor_arm_pen_down_is_the_deployed_1000`
puts a floor arm with base yaw 0 and the pen straight down through this
exporter and gets the deployed generator's constant `(1,0,0,0)` back, and its
sibling test shows a yawed base gives that constant times a rotation about the
pen axis.  Applying a yaw fudge here would break the FK-vs-pose gate below,
which is the whole point of the joint columns.

THE TRANSITS ARE IN THE FILE, AND THEY HAVE TO BE
-------------------------------------------------
v1 of this exporter emitted only the draw rows, bracketed by two synthesized
`lift_start` / `lift_end` rows at the raised point, and let the executor
synthesize the travel between strokes.  THAT FILE IS NOT EXECUTABLE.  The
planner RECONFIGURES THE ARM inside its pen-up transits — measured on arm 31 of
`csail_schedule_h094_v18`, the last draw pose of stroke 2 and the first of
stroke 3 are 4.39 rad apart on joint 3, stroke 5 -> 6 is 4.70 rad on joint 5,
10 -> 11 is 3.14 rad on joint 3 — and a straight-line hover travel cannot flip a
wrist.  In the Drake SIL the arm hit its joint limits on the way to stroke 4 and
never recovered (touch 0.29, 211 mm rms in-plane error).

So every certified pen-up frame between the first and last drawing frame is now
a `travel` row, in order, with its own q1..q7: the executor walks consecutive
non-draw rows as a Cartesian path (`spd = travel_speed if not is_draw`), which
is exactly what the certified transit is.  Consecutive frames with identical q
(the barrier holds, where the fleet stands still for a pen swap) collapse to one
row.

NOTHING IS SYNTHESIZED ANY MORE, except at the two ends and only when the
timeline itself does not provide the lift: the file starts at the last part of
the planner's own descent onto stroke 1 and ends with its own retract off the
last stroke, and only if that retract does not exist (arm 71 finishes with the
pen down and holds there) is a short RAMP of rows solved for with
`ik.solve_cc` off the adjacent draw pose — same branch, two millimetres apart,
FK-gated like every other row.  EVERY ROW CARRIES q1..q7; there are no empty
joint cells, because §2 publishes no joint reference for a segment whose
endpoint lacks one.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import frames as _frames
from ..execute.program import from_schedule

# --- the file format -------------------------------------------------------
CSV_COLUMNS = ["stroke_idx", "wp_idx", "kind", "x_m", "y_m", "z_m",
               "qx", "qy", "qz", "qw", "intensity",
               "q1", "q2", "q3", "q4", "q5", "q6", "q7"]
KIND_LIFT_START = "lift_start"
KIND_DRAW = "draw"
KIND_TRAVEL = "travel"
KIND_LIFT_END = "lift_end"

#: how far past the first / last drawing frame the exporter will follow the
#: planner's own approach and retract looking for `lift_m` of clearance.  If it
#: is not found inside the window the frames are NOT emitted (they would drag
#: the pen along the paper) and a solved ramp is used instead.
END_WINDOW_S = 3.0

#: a solved end ramp is only accepted if it stays within this of the drawing
#: pose it was chained from.  It is a TOTAL, not a per-step bound, and it is
#: the one that matters: holding q7 fixed while the tip climbs is a poor
#: constraint near a shoulder singularity, where a 6 cm lift can cost 1.8 rad
#: of joint 1 against joint 3 no matter how finely it is stepped.  That is not
#: a retract worth inventing, so the ramp is refused and the file says so.
END_ROW_DQ_MAX = 0.35

#: ...and no single solved row may step further than this, which is the
#: conductor's own per-sub-step bound (`writing.MAX_DQ_FRAME`).  Not imported:
#: this module has no other reason to pull in the 1700-line planner.
END_ROW_STEP_DQ = 0.04

#: the deployed generator's `RAISING_AMOUNT` (svg_to_pathway_csv.py): the lift
#: rows sit this far off the paper, ALONG THE PEN AXIS — which for a floor arm
#: is base +z (raised = down + 0.05) and for an inverted arm base -z (raised =
#: down - 0.05), reproducing both of the generator's branches from one rule and
#: matching the executor's own `hover_pt = surface_pt - hover * pen`.
LIFT_M = 0.05

#: contract §1: FK(q) must reproduce the row pose to better than this.
FK_TOL_M = 0.5e-3
FK_TOL_DEG = 0.5

EXPORT_VERSION = "2.0"
GENERATOR = "aris_sixarm.export.pathway"

_FMT_XYZ = "%.6f"        # micrometres, the generator's own precision
_FMT_QUAT = "%.9f"
_FMT_Q = "%.9f"
_FMT_INTENSITY = "%.4f"


# ---------------------------------------------------------------------------
# rotations
# ---------------------------------------------------------------------------
def quat_xyzw(R):
    """Rotation matrix (3,3) -> unit quaternion (x, y, z, w).

    Shepperd's branch, so no square root is ever taken of a near-zero pivot.
    The SIGN is canonicalised by `_canonical_sign`, not here.
    """
    R = np.asarray(R, float).reshape(3, 3)
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w, x, y, z = (0.25 * s, (R[2, 1] - R[1, 2]) / s,
                      (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s)
    elif R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w, x, y, z = ((R[2, 1] - R[1, 2]) / s, 0.25 * s,
                      (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s)
    elif R[1, 1] >= R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w, x, y, z = ((R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                      0.25 * s, (R[1, 2] + R[2, 1]) / s)
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w, x, y, z = ((R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                      (R[1, 2] + R[2, 1]) / s, 0.25 * s)
    q = np.array([x, y, z, w], float)
    return q / np.linalg.norm(q)


def quat_R(q):
    """Unit quaternion (x, y, z, w) -> rotation matrix.  The executor's own."""
    x, y, z, w = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def quat_angle_deg(a, b):
    """Angle between the rotations two quaternions name, in degrees."""
    d = abs(float(np.dot(np.asarray(a, float), np.asarray(b, float))))
    return float(np.degrees(2.0 * np.arccos(min(1.0, max(-1.0, d)))))


def _canonical_sign(q):
    """q or -q, picking the one whose LARGEST component is positive.

    A quaternion and its negative are the same rotation and the deployed file
    writes one of the two: `(1,0,0,0)` for a floor arm and `(0,0,1,0)` for an
    inverted one.  Both have w = 0, so the usual "w >= 0" rule cannot decide
    them — at w = 0 it is deciding on rounding noise, and it picks (-1,0,0,0)
    about half the time.  The largest component is the one furthest from a sign
    change, so this rule reproduces both deployed constants and is stable.
    """
    q = np.asarray(q, float)
    return q if q[int(np.argmax(np.abs(q)))] >= 0.0 else -q


def quats_from_R(Rs):
    """(N,3,3) -> (N,4) xyzw, canonical at row 0 and CONTINUOUS after it.

    Continuity matters because the executor publishes the row's quaternion
    verbatim and slerps between rows; a sign flip mid-stroke is the same
    rotation but a needlessly ugly interpolation.
    """
    Rs = np.asarray(Rs, float).reshape(-1, 3, 3)
    out = np.empty((len(Rs), 4))
    prev = None
    for i, R in enumerate(Rs):
        q = quat_xyzw(R)
        q = _canonical_sign(q) if prev is None else (
            q if float(np.dot(q, prev)) >= 0.0 else -q)
        out[i] = q
        prev = q
    return out


# ---------------------------------------------------------------------------
# the rig
# ---------------------------------------------------------------------------
def tool_offsets(tool):
    """Tool name -> (pen_lat, pen_ext) in the hand-TCP frame, metres.

    Reads `aris_sixarm.frames`' own constants; it does NOT mutate the process's
    ACTIVE tool, so the exporter is safe to call from a test that has a
    different tool active.
    """
    if tool == "inline":
        return 0.0, float(_frames.PEN_EXT)
    if tool == "lateral":
        return float(_frames.PEN_LAT_HOLDER), float(_frames.PEN_EXT_HOLDER)
    raise ValueError(f"unknown tool {tool!r}; want one of {_frames.TOOL_NAMES}")


def base_transform(spec, h_inv=None):
    """The arm's `T_world_base` (4x4), and a refusal if it is ambiguous.

    `ArmSpec.T_world_base` is THE function — the one `planner.build_lattice`,
    `scene_check._chain`, `atlas` and `execute` all call.  It takes an `h_inv`
    that DEFAULTS to `fleet.H_INV_DEFAULT = 1.00`, and a spec whose `R` is None
    AND whose mount hangs (the legacy `inv`/`wall` branch) puts its base at
    whatever that argument says.  Every current rig — proposed, final, final6 —
    carries `R` and `z` explicitly and ignores `h_inv` (`layout.study_spec`,
    `rig_final`), and a legacy FLOOR spec ignores it too; that is the whole of
    the trap docs/DECISIONS.md records, and it is CHECKED here rather than
    argued: the transform is probed at three heights and only accepted as
    h_inv-free if it does not move.  A spec that does move must be given the
    run's own `h_inv` (from the program json) or it is refused.
    """
    probes = [np.asarray(spec.T_world_base(h), float) for h in (0.5, 1.0, 1.5)]
    if all(np.allclose(probes[0], T) for T in probes[1:]):
        T = np.asarray(spec.T_world_base(), float)
        if not np.allclose(T, probes[0]):
            raise ValueError(f"arm {spec.arm_id}: T_world_base() disagrees "
                             "with T_world_base(h) at every probed h")
        if h_inv is not None and not np.allclose(
                T, np.asarray(spec.T_world_base(float(h_inv)), float)):
            raise ValueError(f"arm {spec.arm_id}: T_world_base moved at the "
                             f"run's own h_inv = {h_inv}")
        return T
    if h_inv is None:
        raise ValueError(
            f"arm {spec.arm_id}: this spec's base pose DEPENDS on `h_inv` "
            "(no explicit R, hanging mount) and the run recorded none. Pass "
            "--h-inv: the default is 1.00 m and would silently hang the fleet "
            "in the wrong place (docs/DECISIONS.md).")
    return np.asarray(spec.T_world_base(float(h_inv)), float)


def paper_z_base(T_world_base):
    """The world paper plane (z = 0) expressed in the arm's base frame.

    Only a number when the base z axis is parallel to the world's, which is
    true of every `floor` and `inv` mount (R = rotz(yaw) or roty(pi) @ rotz(yaw)).
    Returns (z, tilt_deg); a caller must refuse a tilt worth mentioning.
    """
    T = np.asarray(T_world_base, float)
    zb_w = T[:3, 2]                       # base z axis, in world
    tilt = float(np.degrees(np.arccos(min(1.0, abs(float(zb_w[2]))))))
    # base coords of the world origin's plane: z_b = zb_w . (p_w - t)
    z = float(np.dot(zb_w, -T[:3, 3]))
    return z, tilt


# ---------------------------------------------------------------------------
# the schedule
# ---------------------------------------------------------------------------
@dataclass
class Stroke:
    """One contiguous run of drawing frames for one arm."""
    seg: int                      # the schedule's own per-arm segment index
    phase: int
    ink: str | None
    i0: int                       # first frame, on the npz timeline
    n: int
    q: np.ndarray                 # (n,7)
    t_start_s: float
    t_end_s: float
    stroke_id: int | None = None  # the ARTWORK stroke, from the program json
    kind: str | None = None       # "outline" | "fill", from the program json


def _arm_frames(prog, z, arm_id):
    """-> (Q (F,7), SEG (F,), PHASE (F,)) for one arm, on the npz timeline.

    The q comes THROUGH `FleetProgram` (so the exporter ships the same numbers
    the executor would stream) and the segment index off the npz, and the two
    are cross-checked frame by frame: `from_schedule` derives a phase's start
    from the `phase` array as `i0 / fps`, so `round(start_s * fps)` recovers
    `i0` and the track must equal the npz slice.  If that ever stops holding,
    the pen state and the joints are misaligned — the one failure this file
    must not be able to make quietly.
    """
    key = f"q_{arm_id}"
    if key not in z.files:
        raise ValueError(f"arm {arm_id} is not in the schedule "
                         f"(arms: {sorted(prog.arms)})")
    Q = np.asarray(z[key], float)
    SEG = np.asarray(z[f"seg_{arm_id}"], int)
    if len(SEG) != len(Q):
        raise ValueError(f"arm {arm_id}: seg/q length mismatch "
                         f"({len(SEG)} vs {len(Q)})")
    PH = np.full(len(Q), -1, int)
    for p in prog.phases:
        tr = p.tracks.get(arm_id)
        if tr is None or not len(tr):
            continue
        i0 = int(round(p.start_s * prog.fps))
        n = len(tr)
        if i0 < 0 or i0 + n > len(Q):
            raise ValueError(f"arm {arm_id}: phase {p.index} runs {i0}..{i0+n} "
                             f"outside the {len(Q)}-frame timeline")
        if not np.allclose(np.asarray(tr.q, float), Q[i0:i0 + n], atol=1e-9):
            raise ValueError(
                f"arm {arm_id}: phase {p.index}'s FleetProgram track does not "
                f"match q_{arm_id}[{i0}:{i0 + n}] — the phase-to-frame mapping "
                "this exporter uses to read `seg` is wrong")
        PH[i0:i0 + n] = p.index
    return Q, SEG, PH


def _program_stroke_labels(program_json, arm_id):
    """{global seg index: (artwork stroke_id, kind)} from the program json.

    `csail_schedule` makes the segment indices GLOBAL across phases by adding
    the running count of the arm's earlier segments (see its npz writer), and
    the program json lists those segments per phase in the same order — so the
    two line up by construction and this walk reproduces the offset.
    """
    out, base = {}, 0
    for ph in (program_json.get("phases") or []):
        entries = ((ph.get("arms") or {}).get(str(arm_id))
                   or (ph.get("arms") or {}).get(arm_id) or [])
        for j, e in enumerate(entries):
            out[base + j] = (e.get("stroke_id"), e.get("kind"))
        base += len(entries)
    return out


def strokes_of(prog, z, arm_id, program_json=None):
    """Every contiguous drawing run of one arm, in timeline order.

    `seg >= 0` IS the pen-down flag: `execute.program.from_schedule` uses the
    same test, and the tip of every such frame is on the world paper plane
    (checked in `build_arm_pathway`).  A segment is expected to be one
    contiguous run; if the conductor ever splits one, each run becomes its own
    CSV stroke and keeps the segment's index, which the manifest then shows as
    a repeated `seg`.
    """
    Q, SEG, PH = _arm_frames(prog, z, arm_id)
    labels = _program_stroke_labels(program_json or {}, arm_id)
    out, i = [], 0
    while i < len(SEG):
        if SEG[i] < 0:
            i += 1
            continue
        k = int(SEG[i])
        j = i
        while j + 1 < len(SEG) and SEG[j + 1] == k:
            j += 1
        sid, kind = labels.get(k, (None, None))
        ph = int(PH[i])
        out.append(Stroke(seg=k, phase=ph,
                          ink=(prog.phases[ph].ink if 0 <= ph < len(prog.phases)
                               else None),
                          i0=i, n=j - i + 1, q=Q[i:j + 1].copy(),
                          t_start_s=i / prog.fps, t_end_s=j / prog.fps,
                          stroke_id=sid, kind=kind))
        i = j + 1
    return out


def _tip_of(qs, pen_ext, pen_lat):
    T, _ = _frames.fk_many(np.asarray(qs, float).reshape(-1, 7))
    return T, T[:, :3, 3] + T[:, :3, :3] @ _frames.tool_offset(pen_ext, pen_lat)


END_ROW_STEP_M = 0.002    # the solved ramp's tip spacing


def _end_rows(q_adj, lift_m, step_m=END_ROW_STEP_M):
    """A RAMP of poses climbing `lift_m` off the paper along the pen axis.

    Used ONLY where the certified timeline has no lift of its own (an arm that
    finishes with the pen down and holds there).  `ik.solve_cc` is the
    case-consistent single-branch solve, chained from the draw pose's own joints
    and its own q7, so every answer is the same IK branch a centimetre further
    out — not a new configuration this file would then have to fly to.

    A ramp rather than one row because one row is one 5 cm jump, and on arm 71
    that was 0.13 rad in a single step: four times the conductor's own per-frame
    bound and the very thing this file exists to keep out of the stream.

    Refused — an empty list, and the caller says so in the manifest — if the
    solver gives up, if any one step exceeds `END_ROW_STEP_DQ`, or if the climb
    drifts more than `END_ROW_DQ_MAX` from where it started.  A 5 cm retract
    that costs a radian is a reconfiguration, not a retract, and the whole
    point of this file is that reconfigurations come from the planner.
    -> list of q, in order away from the paper (empty if refused).
    """
    from .. import ik
    q_adj = np.asarray(q_adj, float).reshape(7)
    n = max(1, int(np.ceil(lift_m / max(step_m, 1e-6))))
    T0, _ = _frames.fk(q_adj)
    out, q_prev = [], q_adj
    for k in range(1, n + 1):
        T = T0.copy()
        T[:3, 3] = T0[:3, 3] - (lift_m * k / n) * T0[:3, 2]   # frame slides
        q = ik.solve_cc(T, float(q_adj[6]), q_prev)
        if (q is None
                or float(np.abs(q - q_prev).max()) > END_ROW_STEP_DQ
                or float(np.abs(q - q_adj).max()) > END_ROW_DQ_MAX):
            return []
        out.append(q)
        q_prev = q
    return out


def _emit_frames(Q, SEG, pen_ext, pen_lat, z_paper, lift_m, window,
                 decimate_m=0.0):
    """Which schedule frames become rows. -> (idx, is_draw, ends, notes).

    THE WHOLE CERTIFIED SPAN, first drawing frame to last, is emitted: draw
    frames and the pen-up transits between them, in timeline order.  Then:

      * consecutive frames with IDENTICAL q collapse to one row, but only when
        the later one is pen-up — a barrier hold is 1177 frames of the same
        pose on arm 31 and one row says it exactly, while a drawing frame is
        never dropped (one draw row per schedule frame is the contract).
      * the file is extended backwards from the first drawing frame, and
        forwards from the last, along the planner's OWN approach and retract
        until the tip is `lift_m` clear of the paper.  If that clearance is not
        reached inside `window` frames the extension is DROPPED rather than
        truncated — those frames would drag the pen along the paper — and
        `ends` asks for one solved row instead.
      * `decimate_m` (default 0 = off) thins pen-up rows to that tip spacing,
        but only across frames whose joints also barely moved.  It is off by
        default because the motion this file exists to carry — a wrist flip
        with the tip nearly stationary — has no tip spacing to be thinned by.
    """
    F = len(SEG)
    draw = np.flatnonzero(np.asarray(SEG) >= 0)
    notes = []
    if not len(draw):
        return [], [], dict(pre=[], post=[]), notes
    f0, f1 = int(draw[0]), int(draw[-1])

    _, tip = _tip_of(Q, pen_ext, pen_lat)
    height = np.abs(tip[:, 2] - z_paper)

    def _reach(start, step):
        """Walk from `start` until the tip is lift_m clear. -> index or None."""
        i = start
        for _ in range(window):
            i += step
            if i < 0 or i >= F:
                return None
            if height[i] >= lift_m:
                return i
        return None

    lo, hi = _reach(f0, -1), _reach(f1, +1)
    if lo is None:
        notes.append(f"no certified approach: the tip is never {1000 * lift_m:.0f} "
                     f"mm clear of the paper within {window} frames before the "
                     "first drawing frame, so a lift_start ramp is solved for")
    if hi is None:
        notes.append(f"no certified retract: the tip is never {1000 * lift_m:.0f} "
                     f"mm clear of the paper within {window} frames after the "
                     "last drawing frame (the arm holds the pen down), so a "
                     "lift_end ramp is solved for")
    a, b = (f0 if lo is None else lo), (f1 if hi is None else hi)

    idx, drw = [], []
    last = None
    for i in range(a, b + 1):
        d = bool(SEG[i] >= 0)
        if not d and last is not None:
            if not np.any(Q[i] != Q[last]):          # a hold: one row says it
                continue
            if decimate_m > 0.0 and not (SEG[i - 1] >= 0 or (
                    i + 1 <= b and SEG[i + 1] >= 0)):
                if (np.linalg.norm(tip[i] - tip[last]) < decimate_m
                        and np.abs(Q[i] - Q[last]).max() < 0.02):
                    continue
        idx.append(i)
        drw.append(d)
        last = i
    # the approach ramp is written outward-in, so reverse the solved climb
    ends = dict(pre=(list(reversed(_end_rows(Q[f0], lift_m))) if lo is None
                     else []),
                post=(_end_rows(Q[f1], lift_m) if hi is None else []))
    for k, missing in (("pre", lo is None), ("post", hi is None)):
        if missing and not ends[k]:
            notes.append(f"the {k} lift ramp could NOT be solved for "
                         "(ik.solve_cc refused or moved too far); the file "
                         "starts/ends with the pen on the paper")
    return idx, drw, ends, notes


# ---------------------------------------------------------------------------
# the rows
# ---------------------------------------------------------------------------
def _fmt_row(stroke_idx, wp_idx, kind, xyz, quat, intensity, q):
    """One formatted CSV row.  `q` is never optional: contract §1 allows empty
    joint columns, but this exporter no longer writes any — `verify_rows` still
    checks for them because a file can be corrupted after it is written."""
    return [str(int(stroke_idx)), str(int(wp_idx)), kind,
            _FMT_XYZ % xyz[0], _FMT_XYZ % xyz[1], _FMT_XYZ % xyz[2],
            _FMT_QUAT % quat[0], _FMT_QUAT % quat[1],
            _FMT_QUAT % quat[2], _FMT_QUAT % quat[3],
            _FMT_INTENSITY % intensity] + [_FMT_Q % v for v in q]


def verify_rows(rows, pen_lat, pen_ext, tol_m=FK_TOL_M, tol_deg=FK_TOL_DEG):
    """Contract §1's gate, run on the FORMATTED rows. -> [problems].

    Re-parses every row exactly as the executor's `_load` does and re-runs the
    forward kinematics, so what is checked is the text that will be written,
    rounding included — not the floats it came from.  A row with empty joint
    columns is skipped (contract §1 permits them, though this exporter writes
    none) and a HALF-empty one is reported: that is corruption, not a v1 row.
    """
    bad = []
    qs, idx = [], []
    for i, r in enumerate(rows):
        jc = r[11:18]
        if any(c == "" for c in jc):
            if any(c != "" for c in jc):
                bad.append(f"row {i}: joint columns are partly filled")
            continue
        qs.append([float(c) for c in jc])
        idx.append(i)
    if not qs:
        return bad
    T, _ = _frames.fk_many(np.asarray(qs, float))
    tip = T[:, :3, 3] + T[:, :3, :3] @ _frames.tool_offset(pen_ext, pen_lat)
    for n, i in enumerate(idx):
        r = rows[i]
        p = np.array([float(r[3]), float(r[4]), float(r[5])])
        d = float(np.linalg.norm(p - tip[n]))
        if d > tol_m:
            bad.append(f"row {i} (stroke {r[0]} wp {r[1]} {r[2]}): FK(q) is "
                       f"{1000 * d:.4f} mm from the row's xyz "
                       f"(tol {1000 * tol_m:g} mm)")
        a = quat_angle_deg([float(r[6]), float(r[7]), float(r[8]),
                            float(r[9])], quat_xyzw(T[n, :3, :3]))
        if a > tol_deg:
            bad.append(f"row {i} (stroke {r[0]} wp {r[1]} {r[2]}): FK(q) is "
                       f"{a:.4f} deg from the row's quaternion "
                       f"(tol {tol_deg:g} deg)")
    return bad


# ---------------------------------------------------------------------------
# one arm
# ---------------------------------------------------------------------------
@dataclass
class ArmPathway:
    arm_id: int
    rows: list
    manifest: dict
    stats: dict = field(default_factory=dict)
    #: filled in by `write_pathway`: {"csv": ..., "manifest": ...}.  Kept OFF
    #: `stats`, which is the same object the manifest carries and has already
    #: been serialised by then.
    paths: dict = field(default_factory=dict)

    def summary(self):
        s = self.stats
        bb = s["bbox_base_m"]
        if not s["n_rows"]:
            return (f"arm {self.arm_id:>3d}: NO DRAWING FRAMES in this "
                    "schedule — the file is a header and nothing else")
        return (f"arm {self.arm_id:>3d}: {s['n_rows']:5d} rows "
                f"({s['n_draw_rows']:5d} draw + {s['n_travel_rows']:5d} travel,"
                f" {s['n_strokes']:3d} strokes), "
                f"draw {s['draw_length_m']:7.3f} m, "
                f"paper z {s['paper_z_base_m']:+.4f} m\n"
                f"          bbox x[{bb['min'][0]:+.3f},{bb['max'][0]:+.3f}] "
                f"y[{bb['min'][1]:+.3f},{bb['max'][1]:+.3f}] "
                f"z[{bb['min'][2]:+.3f},{bb['max'][2]:+.3f}] m, "
                f"r_xy<={s['max_radius_xy_m']:.3f} |p|<={s['max_radius_m']:.3f} m"
                + (f", lean<={s['max_lean_deg']:.1f} deg"
                   if s["max_lean_deg"] > 1e-6 else "")
                + f"\n          step<={s['max_row_dq_rad']:.4f} rad/row, "
                f"biggest stroke boundary {s['max_boundary_dq_rad']:.3f} rad "
                f"(flown over {s['n_travel_rows']} certified travel rows)")


def build_arm_pathway(prog, z, arm_id, spec, *, tool, rig, intensity=1.0,
                      lift_m=LIFT_M, z_mode="plane", h_inv=None,
                      transit_decimate_m=0.0, program_json=None,
                      source_paths=None, extra=None):
    """One arm's CSV rows + manifest.  Refuses to return a file it cannot verify."""
    if z_mode not in ("plane", "fk"):
        raise ValueError(f"z_mode {z_mode!r}; want 'plane' or 'fk'")
    pen_lat, pen_ext = tool_offsets(tool)
    Twb = base_transform(spec, h_inv)
    z_paper, tilt_deg = paper_z_base(Twb)
    if tilt_deg > 1e-6:
        raise ValueError(
            f"arm {arm_id}: its base z axis is {tilt_deg:.3f} deg off the "
            "world's, so the world paper plane is NOT a constant z in this "
            "base frame and `z_m` has no single value (contract §1). Only "
            "floor and inverted mounts are exportable.")

    # the tool the RUN was planned with, straight off the npz.  Exporting a
    # plan with a different tip model moves every point of it.
    plan_ext = float(prog.pen_ext.get(arm_id, np.nan))
    if np.isfinite(plan_ext) and abs(plan_ext - pen_ext) > 1e-6:
        raise ValueError(
            f"arm {arm_id}: --tool {tool} has an axial tip depth of "
            f"{pen_ext:.7f} m but the schedule was planned at {plan_ext:.7f} m "
            f"(npz `pen_ext`). Exporting it would move every row along the pen "
            f"axis by {1000 * (pen_ext - plan_ext):+.1f} mm.")

    strokes = strokes_of(prog, z, arm_id, program_json)
    # the paper's own normal, in the base frame, pointing AWAY from the paper.
    # A vertical pen is exactly -n_b; the planner leans the pen up to a per
    # stroke cone (`tilt.py`), and a synthesized end row follows the PEN rather
    # than the base z so the tip leaves the paper perpendicular to the ink.
    n_b = np.asarray(Twb, float)[:3, :3][2, :].copy()
    Q, SEG, _PH = _arm_frames(prog, z, arm_id)

    frame_idx, is_draw, ends, notes = _emit_frames(
        Q, SEG, pen_ext, pen_lat, z_paper, lift_m,
        window=int(round(END_WINDOW_S * prog.fps)),
        decimate_m=transit_decimate_m)

    rows, stroke_meta = [], []
    draw_len = 0.0
    z_dev = 0.0
    boundaries = []
    if frame_idx:
        qk = np.asarray([Q[i] for i in frame_idx], float)
        # the synthesized end ramps, if the timeline gave no lift (see `ends`)
        pre, post = ends["pre"], ends["post"]
        allq = np.asarray(list(pre) + list(qk) + list(post), float)
        kinds = ([KIND_LIFT_START] * len(pre)
                 + [KIND_DRAW if d else KIND_TRAVEL for d in is_draw]
                 + [KIND_LIFT_END] * len(post))

        T, _ = _frames.fk_many(allq)
        tip = T[:, :3, 3] + T[:, :3, :3] @ _frames.tool_offset(pen_ext, pen_lat)
        quats = quats_from_R(T[:, :3, :3])          # continuous over the file
        pen = T[:, :3, 2]                           # the pen axis, EE +Z
        drawm = np.array([k == KIND_DRAW for k in kinds])
        z_dev = float(np.abs(tip[drawm, 2] - z_paper).max()) if drawm.any() else 0.0
        xyz = tip.copy()
        if z_mode == "plane":
            xyz[drawm, 2] = z_paper

        # stroke_idx: the seg of the stroke a row BELONGS TO — its own if it is
        # a draw row, otherwise the NEXT stroke's (a transit is that stroke's
        # approach, which is what the generator's `lift_start` was), and the
        # last stroke's for anything trailing.  wp_idx counts inside the group.
        segs = np.array([SEG[i] if d else -1
                         for i, d in zip(frame_idx, is_draw)], int)
        segs = np.concatenate([np.full(len(pre), -1), segs,
                               np.full(len(post), -1)]).astype(int)
        owner = segs.copy()
        nxt = -1
        for i in range(len(owner) - 1, -1, -1):     # backward fill
            if owner[i] >= 0:
                nxt = owner[i]
            elif nxt >= 0:
                owner[i] = nxt
        last = owner[0] if owner[0] >= 0 else 0
        for i in range(len(owner)):                 # forward fill the tail
            if owner[i] < 0:
                owner[i] = last
            else:
                last = owner[i]

        wp, prev_owner = 0, None
        for i, k in enumerate(kinds):
            if owner[i] != prev_owner:
                wp, prev_owner = 0, owner[i]
            rows.append(_fmt_row(owner[i], wp, k, xyz[i], quats[i],
                                 intensity if k == KIND_DRAW else 1.0, allq[i]))
            wp += 1

        # -- per-stroke and per-boundary bookkeeping ------------------------
        # the drawing runs, as ROWS.  A run breaks when the segment changes OR
        # when the rows stop being adjacent — grouping on the segment alone
        # would silently merge a segment the conductor split in two, which
        # `strokes_of` is careful to keep apart.
        di = np.flatnonzero(drawm)
        runs = []
        for k in range(len(di)):
            if (k and segs[di[k]] == segs[di[k - 1]]
                    and di[k] == di[k - 1] + 1):
                runs[-1] = (runs[-1][0], runs[-1][1], k)
            else:
                runs.append((int(segs[di[k]]), k, k))
        if len(runs) != len(strokes):
            raise ValueError(
                f"arm {arm_id}: {len(runs)} drawing runs among the rows but "
                f"{len(strokes)} strokes in the schedule — a draw frame was "
                "lost between `strokes_of` and the row builder")
        for (sv, a, b), st in zip(runs, strokes):
            sl = di[a:b + 1]
            length = float(np.linalg.norm(np.diff(tip[sl], axis=0),
                                          axis=1).sum()) if len(sl) > 1 else 0.0
            draw_len += length
            stroke_meta.append(dict(
                seg=int(sv), stroke_id=st.stroke_id, kind=st.kind,
                phase=st.phase, ink=st.ink, n_draw_rows=len(sl),
                length_m=length,
                max_lean_deg=float(np.degrees(np.arccos(np.clip(
                    pen[sl] @ (-n_b), -1.0, 1.0))).max()),
                t_start_s=st.t_start_s, t_end_s=st.t_end_s, frame0=st.i0))
        for (s0, _a0, b0), (s1, a1, _b1) in zip(runs, runs[1:]):
            i0, i1 = int(di[b0]), int(di[a1])       # last / first draw ROW
            gap = allq[i0:i1 + 1]
            steps = np.abs(np.diff(gap, axis=0)).max(axis=1) if len(gap) > 1 \
                else np.zeros(1)
            L = float(np.linalg.norm(np.diff(tip[i0:i1 + 1], axis=0),
                                     axis=1).sum())
            dq = float(np.abs(allq[i1] - allq[i0]).max())
            boundaries.append(dict(
                from_seg=int(s0), to_seg=int(s1),
                boundary_dq_rad=dq, n_transit_rows=int(i1 - i0 - 1),
                max_row_dq_rad=float(steps.max()), transit_tip_len_m=L,
                max_lift_m=float(np.abs(tip[i0:i1 + 1, 2] - z_paper).max()),
                # a reconfiguration with no tip travel to hide it in: the
                # executor paces by Cartesian arc length, so this is the ratio
                # that says how hard the hop is on the joint reference
                reconfig_rad_per_m=(dq / L) if L > 1e-9 else None))

    bad = verify_rows(rows, pen_lat, pen_ext)
    if bad:
        hint = ("" if z_mode == "fk" else
                f"\n  (the draw rows are snapped to the nominal paper plane by "
                f"--z-mode plane and the plan's own tip sits up to "
                f"{1000 * z_dev:.3f} mm off it; --z-mode fk writes the "
                f"certified pose instead)")
        raise ValueError(f"arm {arm_id}: {len(bad)} row(s) fail the contract's "
                         f"FK gate; refusing to write.\n  "
                         + "\n  ".join(bad[:10]) + hint)

    P = np.array([[float(r[3]), float(r[4]), float(r[5])] for r in rows]) \
        if rows else np.zeros((0, 3))
    bbox = dict(min=[float(v) for v in P.min(axis=0)] if len(P) else [],
                max=[float(v) for v in P.max(axis=0)] if len(P) else [])
    max_r = float(np.linalg.norm(P, axis=1).max()) if len(P) else 0.0
    n_draw = sum(1 for r in rows if r[2] == KIND_DRAW)
    speeds = _speeds(program_json, prog, draw_len, n_draw)

    # HOW HARD THE ORIENTATION STEPS.  A v1 file has ONE quaternion for the
    # whole run; this one carries the planner's own per-stroke tool yaw, which
    # is near-constant along a stroke and can flip by a lot at a stroke
    # boundary.  The executor's streaming loop publishes the waypoint's
    # quaternion as-is (no slerp), so this number is what the impedance
    # controller is asked to swallow in one tick at the hop.  Measured, not
    # assumed, and put where an operator sees it before running the file.
    Qr = np.array([[float(r[6]), float(r[7]), float(r[8]), float(r[9])]
                   for r in rows]) if rows else np.zeros((0, 4))
    step_in = step_between = 0.0
    if len(Qr) > 1:
        d = np.degrees(2 * np.arccos(np.clip(
            np.abs(np.sum(Qr[:-1] * Qr[1:], axis=1)), 0.0, 1.0)))
        dr = np.array([r[2] == KIND_DRAW for r in rows])
        ink = dr[:-1] & dr[1:]              # both ends on the paper
        step_in = float(d[ink].max()) if ink.any() else 0.0
        step_between = float(d[~ink].max()) if (~ink).any() else 0.0

    # THE NUMBER THIS FILE LIVES OR DIES BY: the joint step the executor is
    # asked to swallow between two consecutive rows.  v1 dropped the transits
    # and this was 4.7 rad at a stroke boundary; with them it is the
    # conductor's own per-frame bound.
    Qrows = np.array([[float(c) for c in r[11:18]] for r in rows]) \
        if rows else np.zeros((0, 7))
    max_row_dq = float(np.abs(np.diff(Qrows, axis=0)).max()) \
        if len(Qrows) > 1 else 0.0
    max_bnd = max((b["boundary_dq_rad"] for b in boundaries), default=0.0)

    max_lean = max((s["max_lean_deg"] for s in stroke_meta), default=0.0)
    stats = dict(n_rows=len(rows), n_draw_rows=n_draw,
                 n_travel_rows=sum(1 for r in rows if r[2] == KIND_TRAVEL),
                 n_strokes=len(strokes),
                 draw_length_m=draw_len, bbox_base_m=bbox,
                 max_radius_m=max_r, paper_z_base_m=z_paper,
                 max_radius_xy_m=(float(np.hypot(P[:, 0], P[:, 1]).max())
                                  if len(P) else 0.0),
                 max_lean_deg=max_lean,
                 max_row_dq_rad=max_row_dq,
                 max_boundary_dq_rad=max_bnd,
                 max_quat_step_within_stroke_deg=step_in,
                 max_quat_step_between_strokes_deg=step_between)
    manifest = dict(
        # --- contract §1's required set ---
        arm_id=int(arm_id),
        rig=str(rig),
        tool=dict(name=str(tool),
                  tip_offset_hand_tcp_m=[pen_lat, 0.0, pen_ext],
                  pen_lat_m=pen_lat, pen_ext_m=pen_ext,
                  plan_pen_ext_m=(None if not np.isfinite(plan_ext)
                                  else plan_ext)),
        T_world_base=[float(v) for v in np.asarray(Twb, float).reshape(-1)],
        paper_z_base_m=z_paper,
        joint_columns=True,
        generator=dict(module=GENERATOR, version=EXPORT_VERSION, **_git_info()),
        source=dict(name=prog.name, planner_source=prog.source,
                    schedule=(source_paths or {}).get("schedule"),
                    program=(source_paths or {}).get("program"),
                    fps=prog.fps, stride=prog.stride,
                    conductor_dt=prog.conductor_dt,
                    sheet_m=list(prog.sheet_m)),
        speeds=speeds,
        created=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        # --- everything else a reader of this file will want ---
        format=dict(version=EXPORT_VERSION, columns=CSV_COLUMNS,
                    frame="fr3_link0", units="m, rad, unit quaternion"),
        quaternion=dict(order="xyzw", ee_frame="nominal pen tip (setEE)",
                        pen_axis="EE +Z",
                        vertical_pen_axis_base=[float(v) for v in -n_b],
                        max_lean_deg=max_lean,
                        note="the repo's hand-TCP frame carries the flange's "
                             "Rz(-pi/4) twist — the Franka Hand's nominal EE "
                             "frame (libfranka F_T_NE), which setEE extends to "
                             "the tip without turning it — so no yaw "
                             "conversion is applied; the yaw about the pen axis is the "
                             "planner's own and is NOT the deployed "
                             "generator's constant. Unlike a v1 file the "
                             "quaternion is NOT constant: it carries the "
                             "planner's yaw per frame and, on rescued "
                             "strokes, up to `max_lean_deg` of pen lean "
                             "(aris_sixarm.tilt)."),
        z=dict(mode=z_mode, lift_m=lift_m,
               max_fk_deviation_from_plane_m=z_dev,
               note="draw rows carry the paper plane in the base frame and "
                    "nothing else; the executor adds the press depth. The "
                    "lift rows sit `lift_m` off the paper ALONG THE PEN AXIS "
                    "(the executor's own hover convention), which for a "
                    "vertical pen is the generator's z +/- 0.05."),
        intensity=dict(mode="constant", value=float(intensity),
                       note="the schedule carries no tone channel today; "
                            "tone mapping is future work"),
        transits=dict(
            in_file=True,
            kind=KIND_TRAVEL,
            n_rows=stats["n_travel_rows"],
            decimate_m=float(transit_decimate_m),
            synthesized_end_rows={k: len(ends.get(k) or ())
                                  for k in ("pre", "post")},
            note="the planner's CERTIFIED pen-up transit frames ARE in this "
                 "file, in order, as `travel` rows with their own q1..q7. They "
                 "have to be: the plan RECONFIGURES the arm between strokes "
                 "(see stats.max_boundary_dq_rad) and a synthesized "
                 "straight-line hover travel cannot fly that. Consecutive "
                 "frames with identical q (barrier holds) are collapsed to one "
                 "row. Nothing is synthesized except the end rows listed in "
                 "`synthesized_end_rows`, which exist only where the timeline "
                 "itself never lifts the pen."),
        boundaries=boundaries,
        stats=stats,
        strokes=stroke_meta,
        warnings=(list(prog.warnings) + list(notes)
                  + ([] if strokes else
                     [f"arm {arm_id} has no drawing frames in this schedule; "
                      "the CSV is a header row and nothing else"])
                  + ([f"this file crosses {len(set(s['ink'] for s in stroke_meta)) - 1} "
                      "ink change(s); the CSV format cannot express the pen-swap "
                      "barrier, so the executor will fly straight through it"]
                     if len(set(s["ink"] for s in stroke_meta)) > 1 else [])),
    )
    if extra:
        manifest.update(extra)
    return ArmPathway(int(arm_id), rows, manifest, stats)


def _speeds(program_json, prog, draw_len, n_draw):
    """Contract §1's `speeds`: what the plan was paced at."""
    phases = (program_json or {}).get("phases") or []
    ds = sorted({float(p["draw_speed"]) for p in phases
                 if p.get("draw_speed") is not None})
    measured = (draw_len / (n_draw / prog.fps)) if n_draw else None
    return dict(
        draw_m_s=(ds[0] if len(ds) == 1 else (ds or None)),
        draw_m_s_measured=measured,
        travel_m_s=None,
        note="the plan paces TRANSITS in joint space (aris_sixarm.pacing / "
             "transit), not at a Cartesian travel speed, so there is no "
             "travel_m_s to record; the executor uses its own --travel-speed. "
             "`draw_m_s_measured` is this arm's tip length over its drawing "
             "frames on the fleet clock.")


def _git_info():
    """The repo sha the file was generated from (contract §1's `generator`)."""
    root = Path(__file__).resolve().parents[2]
    def _g(*a):
        try:
            return subprocess.run(["git", "-C", str(root), *a],
                                  capture_output=True, text=True,
                                  timeout=10).stdout.strip() or None
        except Exception:
            return None
    sha = _g("rev-parse", "HEAD")
    status = _g("status", "--porcelain")
    return dict(git_sha=sha, git_branch=_g("rev-parse", "--abbrev-ref", "HEAD"),
                git_dirty=(None if status is None else bool(status)))


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------
def write_pathway(pw, out_dir, name):
    """-> (csv path, manifest path)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}_arm{pw.arm_id}.csv"
    man_path = out_dir / f"{name}_arm{pw.arm_id}.manifest.json"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        w.writerows(pw.rows)
    man = dict(pw.manifest)
    man["csv"] = csv_path.name
    man_path.write_text(json.dumps(man, indent=1) + "\n")
    return csv_path, man_path


def export(schedule, program=None, *, rig=None, tool=None, arms=None,
           out="out/pathways", name=None, intensity=1.0, lift_m=LIFT_M,
           z_mode="plane", h_inv=None, transit_decimate_m=0.0, write=True):
    """The whole job. -> list[ArmPathway] (each with `.paths` when written)."""
    from .. import fleet as fleet_mod

    schedule = Path(schedule)
    program = None if program is None else Path(program)
    prog = from_schedule(schedule, program)
    pj = json.loads(program.read_text()) if program else {}

    rig = rig or os.environ.get("ARIS_RIG") or pj.get("rig") or prog.rig
    if rig in (None, "", "?"):
        raise ValueError("no rig: pass --rig (the base transforms come from it)")
    plan_rig = pj.get("rig") or (prog.rig if prog.rig != "?" else None)
    if plan_rig and str(plan_rig) != str(rig):
        raise ValueError(f"--rig {rig!r} but the run was planned on "
                         f"{plan_rig!r}; every base transform would be wrong")
    tool = tool or os.environ.get("ARIS_TOOL") or prog.tool
    if tool not in _frames.TOOL_NAMES:
        raise ValueError(f"unknown tool {tool!r}; want one of {_frames.TOOL_NAMES}")
    if h_inv is None and pj.get("h_inv") is not None:
        h_inv = float(pj["h_inv"])

    fl, _sheet = fleet_mod.rig(rig)
    z = np.load(schedule, allow_pickle=False)
    name = name or schedule.stem
    want = list(prog.arms) if not arms else [int(a) for a in arms]

    out_list = []
    for aid in want:
        if aid not in fl:
            raise ValueError(f"arm {aid} is not in rig {rig!r} "
                             f"(has {sorted(fl)})")
        pw = build_arm_pathway(
            prog, z, aid, fl[aid], tool=tool, rig=rig, intensity=intensity,
            lift_m=lift_m, z_mode=z_mode, h_inv=h_inv,
            transit_decimate_m=transit_decimate_m, program_json=pj,
            source_paths=dict(schedule=str(schedule),
                              program=None if program is None else str(program)))
        if write:
            c, m = write_pathway(pw, out, name)
            pw.paths = dict(csv=str(c), manifest=str(m))
        out_list.append(pw)
    return out_list


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m aris_sixarm.export.pathway",
        description="conducted schedule -> per-arm pathway CSV v2 + manifest "
                    "(docs/ARIS2_CONTRACTS.md §1, docs/EXPORT_PATHWAY.md)")
    ap.add_argument("--schedule", required=True, help="out/<name>_schedule.npz")
    ap.add_argument("--program", default=None, help="out/<name>_program.json")
    ap.add_argument("--rig", default=None,
                    help="fleet registry the run was planned on (env ARIS_RIG; "
                         "the flag wins). Must match the program json.")
    ap.add_argument("--tool", default=None, choices=list(_frames.TOOL_NAMES),
                    help="pen model (env ARIS_TOOL; the flag wins). Must match "
                         "the schedule's own pen_ext.")
    ap.add_argument("--arms", nargs="*", type=int, default=None,
                    help="arm ids; default every arm in the schedule")
    ap.add_argument("--out", default="out/pathways", help="output directory")
    ap.add_argument("--name", default=None,
                    help="file stem; default the schedule's own")
    ap.add_argument("--intensity", type=float, default=1.0,
                    help="constant tone written on every draw row (0..1). The "
                         "schedule has no tone channel; see the doc.")
    ap.add_argument("--lift", type=float, default=LIFT_M,
                    help="the clearance off the paper (m) that ends the file's "
                         "approach and retract, and the height of a solved end "
                         "row where the timeline provides none; the deployed "
                         "generator's 0.05")
    ap.add_argument("--transit-decimate-mm", type=float, default=0.0,
                    help="thin pen-up rows to this tip spacing, and only where "
                         "the joints also barely moved. DEFAULT 0 = keep every "
                         "certified transit frame: the motion this file exists "
                         "to carry is a wrist flip with a near-stationary tip, "
                         "which has no tip spacing to thin by.")
    ap.add_argument("--z-mode", default="plane", choices=("plane", "fk"),
                    help="'plane' (default) writes the paper plane in the base "
                         "frame on every draw row; 'fk' writes the certified "
                         "pose's own z")
    ap.add_argument("--h-inv", type=float, default=None,
                    help="inverted-base height, for a legacy rig whose specs "
                         "carry no explicit base pose")
    a = ap.parse_args(argv)

    if not 0.0 <= a.intensity <= 1.0:
        ap.error("--intensity must be in [0, 1]")

    # the tool is process-global state in `frames`; set it before anything the
    # rig registry imports can capture it (aris_sixarm/__init__.py's door).
    tool = a.tool or os.environ.get("ARIS_TOOL") or None
    if tool:
        _frames.activate_tool(tool)

    try:
        pws = export(a.schedule, a.program, rig=a.rig, tool=a.tool,
                     arms=a.arms, out=a.out, name=a.name,
                     intensity=a.intensity, lift_m=a.lift, z_mode=a.z_mode,
                     h_inv=a.h_inv,
                     transit_decimate_m=a.transit_decimate_mm / 1000.0)
    except ValueError as e:                 # every refusal in this module
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    if not pws:
        print("nothing to export", file=sys.stderr)
        return 1
    m0 = pws[0].manifest
    print(f"{m0['source']['name']}")
    print(f"  rig {m0['rig']}, tool {m0['tool']['name']} "
          f"(tip {m0['tool']['tip_offset_hand_tcp_m']} in hand TCP), "
          f"z-mode {m0['z']['mode']}, intensity {a.intensity:g}")
    for pw in pws:
        print("  " + pw.summary())
        worst = sorted(pw.manifest["boundaries"],
                       key=lambda b: -b["boundary_dq_rad"])[:3]
        for b in worst:
            if b["boundary_dq_rad"] < 0.5:
                continue
            r = b["reconfig_rad_per_m"]
            print(f"          stroke {b['from_seg']}->{b['to_seg']}: "
                  f"{b['boundary_dq_rad']:.2f} rad over "
                  f"{b['n_transit_rows']} rows / "
                  f"{b['transit_tip_len_m']:.2f} m of tip travel"
                  + (f"  [{r:.1f} rad/m]" if r else ""))
    tot = sum(p.stats["draw_length_m"] for p in pws)
    print(f"  {len(pws)} arm(s), {sum(p.stats['n_rows'] for p in pws)} rows, "
          f"{tot:.3f} m of ink -> {a.out}")
    seen = set()
    for pw in pws:
        for w in pw.manifest.get("warnings", []):
            if w not in seen:
                seen.add(w)
                print(f"  ! {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
