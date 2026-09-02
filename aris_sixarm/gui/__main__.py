"""`python -m aris_sixarm.gui` — start the server and print the URL.

THE SERVER IS NOT THE PLANNER, so it takes no rig and no tool: those belong to
a job, and a job is a subprocess.  Starting the server with ARIS_RIG set does
no harm and no good; the arms you see are the arms of the rig the JOB names.
"""
import argparse
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="python -m aris_sixarm.gui",
        description="Interactive browser GUI for the Aris stroke planner.")
    ap.add_argument("--host", default="127.0.0.1",
                    help="127.0.0.1 (default) is reachable only from this "
                         "machine; 0.0.0.0 exposes the planner to the network")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--jobs-dir", default=None,
                    help="where job directories go (default out/gui_jobs)")
    ap.add_argument("--reload", action="store_true",
                    help="restart on a source change (development only)")
    a = ap.parse_args(argv)

    import uvicorn
    from .server import create_app

    print(f"\n  Aris stroke planner GUI:  http://{a.host}:{a.port}/\n")
    uvicorn.run(create_app(a.jobs_dir), host=a.host, port=a.port,
                log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
