"""The browser GUI for the stroke planner: a server, a worker, a viewer.

    .venv/bin/python -m aris_sixarm.gui          # then open the URL it prints

`server.py` serves `web/viewer/` and the job API; `worker.py` is the
subprocess that plans; `jobs.py` is the bookkeeping between them.  Nothing in
this package plans anything itself — the GUI runs `scripts/draw.py`, which is
the same front door a person runs from a terminal.

See docs/VIEWER.md for the event schema, the programme schema and what each
milestone of the viewer does.
"""
from .server import create_app                              # noqa: F401

__all__ = ["create_app"]
