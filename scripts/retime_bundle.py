#!/usr/bin/env python3
"""Certified timeline -> per-arm fr3 bundle, slowed until the gate accepts it.

THE NUMBER THIS EXISTS FOR.  `fr3drivers`' command gate calls
`--fr3_max_joint_acceleration = 10.0` *"the single most dangerous field"*, and
`docs/ARCHITECTURE_V2.md` §5 Q4 records that the shipped six-arm programme
demands **33-37 rad/s²** through this same adapter.  `pacing.py` says why in
its own words: it bounds joint VELOCITY and nothing else, and *"the v profile
here is a ceiling, not a trajectory"*.  So a certified path is not a flyable
one, and this script measures the gap and closes as much of it as a TIME
SCALING can.

WHY A TIME SCALING AND NOTHING ELSE.  The bundle format's own argument, quoted
in `execute/backends.py`: a time scaling *"leaves every control point
untouched, so every collision certificate the planner earned survives"*.  That
is the whole reason this is a re-timing script and not a re-planning one — the
arm visits exactly the poses `scene_check` graded, in exactly the same order,
and only the clock changes.  Scaling every sample time by 1/s scales the
sampled speed by s and the sampled acceleration by s², so the smallest s that
clears both gates is a one-dimensional search and the duration grows by 1/s.

WHAT THIS DOES NOT FIX, SAID PLAINLY.  The conducted path is piecewise linear
in joint space, so at a corner the TRUE acceleration is impulsive no matter how
slowly it is flown; what the scaling bounds is the acceleration the driver
actually measures between the samples it is sent, which is the quantity the
gate tests and `trajectory.JointTrajectory.check` screens.  A genuinely C1 path
needs a blend or a real TOPP pass, and that is a planning change this script
deliberately does not make.  If the scaling needed is large enough to be
absurd, the honest output is the number and a bundle marked NOT FLYABLE, which
is what `--max-slowdown` produces.

    ARIS_RIG=proposed ARIS_TOOL=lateral python3 scripts/retime_bundle.py \\
        out/unknown_h0970_schedule.npz out/unknown_h0970_program.json \\
        --out out/bundles/unknown_h0970 --json out/unknown_h0970_retime.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

QDD_GATE = 10.0          # rad/s^2, fr3drivers --fr3_max_joint_acceleration
SPEED_SAFETY = 0.8       # trajectory.SPEED_SAFETY, the fraction of QD_MAX used


def peaks(traj):
    """-> (peak |qd| per joint, peak |qdd| per joint), both (7,)."""
    v = traj.joint_speed()
    a = traj.joint_accel()
    return (v.max(axis=0) if len(v) else np.zeros(7),
            a.max(axis=0) if len(a) else np.zeros(7))


def scale_for(traj, qd_max, qdd_gate=QDD_GATE, safety=SPEED_SAFETY):
    """The largest rate s <= 1 at which both gates hold. -> (s, before).

    Closed form, not a search: scaling the clock by 1/s multiplies every
    sampled speed by s and every sampled acceleration by s^2, so

        s <= min_j (safety * qd_max_j / v_j)      and
        s <= min_j sqrt(qdd_gate / a_j)

    and the binding one is the min.  Reported to four digits because the
    duration growth is 1/s and a hardware day is a 3-hour box.
    """
    v, a = peaks(traj)
    sv = np.inf if not np.any(v > 0) else float(
        np.min(safety * np.asarray(qd_max, float)[v > 0] / v[v > 0]))
    sa = np.inf if not np.any(a > 0) else float(
        np.min(np.sqrt(qdd_gate / a[a > 0])))
    s = min(1.0, sv, sa)
    return s, dict(peak_qd=v.tolist(), peak_qdd=a.tolist(),
                   speed_scale_from_velocity=sv,
                   speed_scale_from_acceleration=sa,
                   binding=("acceleration" if sa <= sv else "velocity"))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz")
    ap.add_argument("program", nargs="?", default=None)
    ap.add_argument("--out", default="out/bundles", help="bundle directory")
    ap.add_argument("--tag", default=None, help="file stem (default: the npz's)")
    ap.add_argument("--arms", type=int, nargs="*", default=None)
    ap.add_argument("--qdd-gate", type=float, default=QDD_GATE)
    ap.add_argument("--hz", type=float, default=None, metavar="HZ",
                    help="measure (and export) the trajectory RESAMPLED to this "
                         "rate -- the driver's own 1000 Hz is what its gate "
                         "actually sees, and it is a harsher number than the "
                         "conductor's 48 Hz")
    ap.add_argument("--max-slowdown", type=float, default=8.0, metavar="X",
                    help="refuse to claim a bundle is flyable if it needs to be "
                         "slowed by more than this (default 8x)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    from aris_sixarm.execute.program import from_schedule
    from aris_sixarm.execute.backends import Fr3BundleBackend
    from aris_sixarm.frames import QD_MAX

    prog = from_schedule(a.npz, a.program)
    tag = a.tag or Path(a.npz).stem
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    arms = a.arms or list(prog.arms)

    print(f"{a.npz}: {len(prog.arms)} arms, {len(prog.phases)} phases, "
          f"{prog.duration_s:.2f} s at {prog.fps:g} fps (stride {prog.stride})")
    print(f"  gate: |qdd| <= {a.qdd_gate:g} rad/s^2, "
          f"|qd| <= {SPEED_SAFETY:g} x QD_MAX"
          + (f", measured at {a.hz:g} Hz" if a.hz else
             ", measured on the conducted samples"))
    if prog.stride > 1:
        print(f"  !! stride {prog.stride}: this npz is DECIMATED and "
              "Fr3BundleBackend.preflight refuses it.  Re-conduct with "
              "--substeps 1.")

    rows, worst = [], 1.0
    for aid in arms:
        traj = prog.track(aid)
        meas = traj.resample(a.hz) if a.hz else traj
        s, before = scale_for(meas, QD_MAX, a.qdd_gate)
        slow = 1.0 / max(s, 1e-12)
        worst = max(worst, slow)
        after_qd = np.asarray(before["peak_qd"]) * s
        after_qdd = np.asarray(before["peak_qdd"]) * s * s
        flyable = slow <= a.max_slowdown
        path = outdir / f"{tag}_arm{aid}{'' if flyable else '_NOT_FLYABLE'}.npz"
        d = Fr3BundleBackend.export(
            traj, path, speed_scale=s,
            source=f"aris_sixarm retime_bundle {tag} arm {aid} "
                   f"(time scaling only, control points untouched)")
        rows.append(dict(
            arm=int(aid), samples=int(len(traj)),
            duration_before_s=float(traj.duration_s),
            duration_after_s=float(traj.duration_s / max(s, 1e-12)),
            speed_scale=float(s), slowdown=float(slow),
            peak_qd_before=float(max(before["peak_qd"])),
            peak_qd_after=float(after_qd.max()),
            peak_qdd_before=float(max(before["peak_qdd"])),
            peak_qdd_after=float(after_qdd.max()),
            binding=before["binding"], flyable=bool(flyable),
            bundle=str(path), segments=int(len(d["t_start"]))))
        print(f"\n  arm {aid}: {len(traj)} samples, {traj.duration_s:.2f} s")
        print(f"    peak |qd|   {max(before['peak_qd']):7.3f} -> "
              f"{after_qd.max():7.3f} rad/s   "
              f"(gate {SPEED_SAFETY * QD_MAX.min():.3f}..{SPEED_SAFETY * QD_MAX.max():.3f})")
        print(f"    peak |qdd|  {max(before['peak_qdd']):7.3f} -> "
              f"{after_qdd.max():7.3f} rad/s^2 (gate {a.qdd_gate:g})  "
              f"BINDING: {before['binding']}")
        print(f"    rate {s:.4f} = {slow:.2f}x slower, "
              f"{traj.duration_s:.1f} s -> {traj.duration_s / max(s, 1e-12):.1f} s")
        print(f"    wrote {path}  (degree 1, {len(d['t_start'])} segments"
              + ("" if flyable else
                 f"; NOT FLYABLE: needs {slow:.1f}x, over the "
                 f"{a.max_slowdown:g}x limit") + ")")

    pf = Fr3BundleBackend().preflight(prog)
    print(f"\nFr3BundleBackend.preflight: "
          + ("clean" if not pf else f"{len(pf)} problem(s)"))
    for line in pf[:12]:
        print(f"  - {line}")

    doc = dict(source=str(a.npz), program=(None if not a.program
                                           else str(a.program)),
               tag=tag, qdd_gate=float(a.qdd_gate),
               speed_safety=SPEED_SAFETY, measured_hz=a.hz,
               stride=int(prog.stride), fps=float(prog.fps),
               worst_slowdown=float(worst), arms=rows, preflight=pf)
    if a.json:
        Path(a.json).write_text(json.dumps(doc, indent=1))
        print(f"wrote {a.json}")
    # THE FLEET FLIES ON ONE CLOCK, SO IT FLIES AT THE SLOWEST ARM'S RATE.
    # Re-timing one arm and not the other moves them against each other at
    # instants nobody certified -- `execute.Governor`'s whole argument.
    print(f"\nFLEET RATE {1.0 / worst:.4f} ({worst:.2f}x slower): one scalar "
          "for every arm, because a per-arm rate breaks the inter-arm "
          "certificate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
