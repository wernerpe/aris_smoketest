#!/usr/bin/env python3
"""Inspect or rehearse a conducted schedule.  NOTHING HERE TOUCHES A ROBOT.

    ARIS_RIG=proposed ARIS_TOOL=lateral python3 -m aris_sixarm.execute \
        out/csail_schedule_h094_v18.npz out/csail_program_h094_v18.json

    ... --play                 rehearse it in meshcat at wall-clock speed
    ... --play --solo 31       one arm, with the other five frozen (see below)
    ... --check                the joint-envelope gate only, no viewer
    ... --json out/x.json      write the summary beside the run

THE RIG AND THE TOOL MUST BE THE ONES THE RUN USED, for the same reason
`export_viewer_bundle.py` says so: the base transforms come off `fleet.FLEET`
in this process.
"""
import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("npz")
    ap.add_argument("program", nargs="?", default=None)
    ap.add_argument("--play", action="store_true",
                    help="rehearse in meshcat at wall-clock speed")
    ap.add_argument("--check", action="store_true",
                    help="run the joint-envelope gate and print the verdict")
    ap.add_argument("--solo", type=int, default=None, metavar="ARM",
                    help="one arm only; prints the frozen set the run would "
                         "have to be re-certified against")
    ap.add_argument("--rate", type=float, default=1.0,
                    help="clock rate, fleet-wide (0.25 is a sensible rehearsal)")
    ap.add_argument("--hz", type=float, default=None)
    ap.add_argument("--fast", action="store_true",
                    help="do not sleep; play as fast as the loop runs")
    ap.add_argument("--json", default=None, help="write the summary here")
    a = ap.parse_args(argv)

    from aris_sixarm import fleet as fleet_mod
    from aris_sixarm import frames
    from aris_sixarm.execute import from_schedule

    print(f"rig {fleet_mod.ACTIVE_RIG!r}, tool "
          f"{'lateral' if frames.PEN_LAT else 'inline'}, "
          f"{len(fleet_mod.FLEET)} arms in the fleet")
    prog = from_schedule(a.npz, a.program)
    print("\n".join(prog.report()))

    if a.solo is not None:
        solo = prog.solo(a.solo)
        print("\n".join(solo.report()))

    rc = 0
    if a.check or a.play:
        bad = prog.check()
        print(f"joint envelope: {'PASS' if not bad else f'{len(bad)} PROBLEM(S)'}")
        for m in bad[:20]:
            print(f"  ! {m}")
        rc = 1 if bad else 0

    if a.json:
        Path(a.json).write_text(json.dumps(prog.summary(), indent=1))
        print(f"wrote {a.json}")

    if a.play:
        from aris_sixarm.execute import MeshcatDryRun, confirm_console, play
        arms = [a.solo] if a.solo is not None else None
        be = MeshcatDryRun()
        log = play(prog, be, arms=arms, rate=a.rate, hz=a.hz,
                   realtime=not a.fast, confirm=confirm_console,
                   skip_preflight=True)
        print("\n".join(log.report()))
        rc = rc or (0 if log.ok else 1)
    return rc


if __name__ == "__main__":
    sys.exit(main())
