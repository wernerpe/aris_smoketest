#!/usr/bin/env python3
"""Loop a conducted programme in a meshcat scene on a port of your choosing.

WHY THIS AND NOT `python -m aris_sixarm.execute --play`.  That CLI is the
rehearsal: it plays ONCE, at wall-clock speed, and stops at every barrier for a
human to type "go" — which is exactly right when a person is standing in front
of it deciding whether to fly the thing, and exactly wrong for a scene left up
on a screen all afternoon.  It also calls `meshcat.Visualizer()` with no URL,
which picks the first free port in 7000-7005; on this machine those belong to
other people's scenes.

So this script does three things that one does not: it binds an EXPLICIT port
through `animate_staged.start_bridge` (the bridge `meshcat-server` cannot be
told to bind, since it has no `--port`), it auto-acknowledges barriers instead
of waiting on stdin, and it loops.  Everything else is the shipped executor —
`execute.play` into `execute.MeshcatDryRun` — so what is on the screen is the
same numbers the same loop would send to a robot.

    ARIS_RIG=proposed ARIS_TOOL=lateral python3 scripts/meshcat_loop.py \\
        out/unknown_h0970_schedule.npz out/unknown_h0970_program.json \\
        --port 7008 --rate 1.0

Then open  http://<host>:7008/static/
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz")
    ap.add_argument("program", nargs="?", default=None)
    ap.add_argument("--port", type=int, default=7008,
                    help="7000-7007 belong to other scenes on this machine")
    ap.add_argument("--host", default="0.0.0.0",
                    help="0.0.0.0 so the scene is reachable off the box")
    ap.add_argument("--rate", type=float, default=1.0)
    ap.add_argument("--arms", type=int, nargs="*", default=None)
    ap.add_argument("--loops", type=int, default=0,
                    help="0 = forever")
    ap.add_argument("--pause", type=float, default=3.0,
                    help="seconds held on the last frame between loops")
    a = ap.parse_args(argv)

    if 7000 <= a.port <= 7007:
        raise SystemExit(f"port {a.port} is inside 7000-7007, which belong to "
                         "other scenes on this machine; pick another")

    from animate_staged import start_bridge
    from aris_sixarm import fleet as fleet_mod
    from aris_sixarm import frames
    from aris_sixarm.execute import MeshcatDryRun, from_schedule, play

    url = start_bridge(a.port, host=a.host)
    prog = from_schedule(a.npz, a.program)
    print(f"rig {fleet_mod.ACTIVE_RIG!r}, tool "
          f"{'lateral' if frames.PEN_LAT else 'inline'}, "
          f"{len(fleet_mod.FLEET)} arms in the fleet")
    print("\n".join(prog.report()))
    print(f"\nmeshcat bridge on {url}\n  -> http://frankastation.drl.csail.mit.edu:"
          f"{a.port}/static/\n  -> http://localhost:{a.port}/static/")

    be = MeshcatDryRun(url=url)
    # AUTO-ACKNOWLEDGE.  `play`'s default REFUSES a barrier, on purpose, so a
    # pen swap cannot be flown past without a human.  Nothing here flies; the
    # backend's every robot-facing call is a no-op, and a looping scene that
    # stopped forever at the first barrier would show one frame all afternoon.
    n = 0
    while a.loops <= 0 or n < a.loops:
        n += 1
        log = play(prog, be, arms=a.arms, rate=a.rate, realtime=True,
                   confirm=lambda b: True, skip_preflight=True)
        print(f"  loop {n}: {'ok' if log.ok else 'stopped — ' + str(log.reason)}"
              f"  ({log.n_sent if hasattr(log, 'n_sent') else '?'} samples)")
        if not log.ok:
            break
        time.sleep(max(0.0, a.pause))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
