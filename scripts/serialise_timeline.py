#!/usr/bin/env python3
"""A conducted fleet timeline -> one in which only ONE arm ever moves.

WHY, IN ONE SENTENCE.  `scene_check`'s inter-arm certificate is a statement
about two arms at the SAME instant on ONE clock, and on hardware day 1 there is
no such clock — two arms are two `fr3_sender` processes started by two hands —
so the safe first drawing run is the one whose certificate does not depend on a
clock at all.

WHAT IT BUILDS.  The concurrent programme is cut into two blocks:

    block 1   arm A flies its whole conducted track;  arm B stands at the pose
              it was conducted to START from.
    block 2   arm B flies its whole conducted track;  arm A stands at the pose
              it was conducted to END at.

Both frozen poses are poses the conductor already put that arm in and
`scene_check` already graded, and the seam between the blocks has NO step in
either arm: A finishes block 1 at its last pose and holds exactly that; B holds
its first pose through block 1 and then starts from exactly that.  So there is
no `Barrier.reposition` to supervise — which matters, because
`docs/HARDWARE_LADDER.md` §2.4 records that repositioning at a barrier is
uncertified motion into a stiff controller.

WHAT IS AND IS NOT CERTIFIED.  The re-indexing moves no control point: every
sample in the output is a sample from the input.  So every per-arm gate the
conductor earned — self-collision, joint limits, the paper chain, the tip
floor, the neighbours' base columns — still holds sample for sample.  What does
NOT carry over is the INTER-ARM gate, because a frozen fleet is a different
scene from a moving one (`execute.program.SoloProgram.recheck_required` says
exactly this).  Therefore this script re-runs `scripts/recheck_timeline.py`'s
own checker on the result and refuses to claim anything it has not measured.

THE COARSE CUT IS THE DELIBERATE ONE.  Finer interleaving — alternating per
stroke, or per tour — would be shorter, and it would also multiply the number
of places where a human has to decide that the other arm really has stopped.
Two blocks is one decision, and on a first hardware day the decision count is
what matters.  Nothing here implements a finer cut, on purpose.

    ARIS_RIG=proposed ARIS_TOOL=lateral python3 scripts/serialise_timeline.py \\
        out/unknown_h0970_schedule.npz --out out/unknown_h0970_alt.npz \\
        --order 31,71 --recheck
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The keys an execution timeline has to carry: what
# `execute.program.from_schedule` parses and what `recheck_timeline.recheck`
# reads.  Animation-only arrays (`segpts_`, `segoff_`, `u_`, `ink_*`) are NOT
# rebuilt -- they describe where ink was laid on a clock this file no longer
# has, and a stale copy of them would be a lie a viewer would believe.
SCALARS = ("fps", "stride", "dt", "margin", "min_clearance", "pause_s",
           "sheet", "h", "calib", "safety")


def moving_mask(q, tol=1e-9):
    """Which samples this arm is actually moving at. -> (M,) bool."""
    q = np.asarray(q, float)
    d = np.zeros(len(q), bool)
    d[1:] = np.abs(np.diff(q, axis=0)).max(axis=1) > tol
    return d


def serialise(z, order=None, tol=1e-9):
    """The npz payload -> the alternating one. -> dict of arrays.

    `order` names the arms that TAKE THE PAPER, in turn.  It is not a
    permutation of the timeline's arms and must not be: a conducted fleet npz
    carries every arm in the rig, and on installation day 1 four of the six are
    PARKED — present as obstacles, 2 steps long, 0.00 m of chain motion. Those
    arms have no turn to take; they stand where the conductor parked them from
    the first frame to the last, which is what they were already doing.

    Left None, the movers are the arms that actually move, in ascending id —
    which on day 1 puts arm 31 (the west arm, under the start of the word)
    first, so the ink appears left to right the way a person reads it.
    """
    arms = [int(v) for v in z["arms"]]
    Q = {a: np.asarray(z[f"q_{a}"], float) for a in arms}
    SEG = {a: np.asarray(z[f"seg_{a}"]).astype(np.int64) for a in arms}
    M = len(Q[arms[0]])
    for a in arms:
        if len(Q[a]) != M or len(SEG[a]) != M:
            raise SystemExit(f"arm {a} is not on the same clock as arm {arms[0]}")

    moves = [a for a in arms
             if np.abs(np.diff(Q[a], axis=0)).max(initial=0.0) > tol]
    order = moves if order is None else [int(a) for a in order]
    unknown = sorted(set(order) - set(arms))
    if unknown:
        raise SystemExit(f"--order names arms {unknown} that are not in this "
                         f"timeline ({arms})")
    missed = sorted(set(moves) - set(order))
    if missed:
        raise SystemExit(f"--order leaves out arms {missed}, which MOVE in "
                         f"this timeline; they would be frozen at their first "
                         f"pose and their ink would never be drawn")
    parked = [a for a in arms if a not in order]

    blocks = []
    for k, mover in enumerate(order):
        q_out, seg_out = {}, {}
        for a in arms:
            if a == mover:
                q_out[a] = Q[a].copy()
                seg_out[a] = SEG[a].copy()
            else:
                # BEFORE ITS TURN an arm stands where it will START; AFTER its
                # turn it stands where it FINISHED.  Both are conducted poses,
                # and both make the seam between blocks a no-op for that arm.
                # A PARKED arm has no turn, and its first and last pose are the
                # same pose, so this is its park either way.
                done = a in order and order.index(a) < k
                hold = Q[a][-1] if done else Q[a][0]
                q_out[a] = np.tile(hold, (M, 1))
                seg_out[a] = np.full(M, -1, np.int64)   # a still pen draws nothing
        blocks.append((mover, q_out, seg_out))

    out = {}
    for a in arms:
        out[f"q_{a}"] = np.concatenate([b[1][a] for b in blocks]).astype(np.float32)
        out[f"seg_{a}"] = np.concatenate([b[2][a] for b in blocks])
    N = len(out[f"q_{arms[0]}"])
    out["arms"] = np.array(arms, np.int64)
    out["drawing_arms"] = np.array(arms, np.int64)
    out["pen_ext"] = np.asarray(z["pen_ext"], float)
    # ONE PHASE PER BLOCK, so `from_schedule` shows the operator two tracks and
    # puts a barrier between them -- which is exactly the hand-over the run
    # ladder asks a person to witness.
    out["phase"] = np.concatenate([np.full(M, k, np.int64)
                                   for k in range(len(blocks))])
    fps = float(z["fps"])
    out["phase_start_s"] = np.array([k * M / fps for k in range(len(blocks))])
    out["n_phases"] = np.int64(len(blocks))
    out["duration"] = np.float64((N - 1) / fps)
    ink = str(z["phase_ink"][0]) if "phase_ink" in z.files else "black"
    out["phase_ink"] = np.array([ink] * len(blocks))
    for k in SCALARS:
        if k in z.files:
            out[k] = z[k]
    out["serialised_order"] = np.array(order, np.int64)
    return out, [b[0] for b in blocks], M, N, fps


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--order", default=None, metavar="A,B",
                    help="who takes the paper first (default: ascending id)")
    ap.add_argument("--recheck", action="store_true",
                    help="run scene_check over the result and REFUSE to write "
                         "a file it fails")
    ap.add_argument("--sub", type=int, default=2)
    ap.add_argument("--json", default=None, help="the re-check report")
    a = ap.parse_args(argv)

    z = np.load(a.npz, allow_pickle=False)
    order = (None if not a.order
             else [int(x) for x in a.order.replace(",", " ").split()])
    out, blocks, M, N, fps = serialise(z, order)
    parked = [int(x) for x in out["arms"] if int(x) not in blocks]
    print(f"{a.npz}: {len(out['arms'])} arms, {M} frames "
          f"({(M - 1) / fps:.2f} s) concurrent"
          + (f"; arms {parked} are PARKED and take no turn" if parked else ""))
    print(f"  -> {len(blocks)} blocks in the order {blocks}, {N} frames "
          f"({(N - 1) / fps:.2f} s): only one arm moves at a time")
    for k, mover in enumerate(blocks):
        held = [int(x) for x in out["arms"] if int(x) != mover]
        print(f"     block {k + 1}: arm {mover} flies; arms {held} HELD")

    if a.recheck:
        from recheck_timeline import summarise
        from aris_sixarm import scene_check
        from recheck_timeline import fleet_for
        fl, hh = fleet_for(None, None, "uniform")
        q = {int(x): np.asarray(out[f"q_{int(x)}"], float) for x in out["arms"]}
        dr = {int(x): out[f"seg_{int(x)}"] >= 0 for x in out["arms"]}
        pens = {int(x): float(p) for x, p in zip(out["arms"], out["pen_ext"])}
        margin = float(out["margin"])
        rep = scene_check.check_timeline(q, float(out["dt"]), margin,
                                         programs=None, h_inv=hh,
                                         pen_ext=pens, sub=a.sub, verbose=True,
                                         fleet=fl, drawing=dr)
        for line in summarise(rep, margin):
            print(line)
        if a.json:
            Path(a.json).write_text(json.dumps(
                {k: v for k, v in rep.items() if k != "segments"},
                default=str, indent=1))
            print(f"wrote {a.json}")
        if not rep["ok"]:
            raise SystemExit("the serialised timeline does NOT certify; "
                             "nothing written")

    np.savez_compressed(a.out, **out)
    print(f"wrote {a.out} ({Path(a.out).stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
