"""The browser GUI for the stroke planner: a server, a worker, a viewer.

    .venv/bin/python -m aris_sixarm.gui          # then open the URL it prints

`server.py` serves `web/viewer/` and the job API; `worker.py` is the
subprocess that plans; `jobs.py` is the bookkeeping between them.  Nothing in
this package plans anything itself — the GUI runs `scripts/draw.py`, which is
the same front door a person runs from a terminal.

`create_app` IS IMPORTED LAZILY, and that is not tidiness.  A worker is started
as `python -m aris_sixarm.gui.worker`, which runs this file first; importing
the server here would make every planning subprocess pull in fastapi,
starlette and pydantic before it traces a pixel — and would make a job fail on
a machine that has the planner but not the `gui` extra, which is exactly the
machine the planner is supposed to run on.

See docs/VIEWER.md for the event schema, the programme schema and what each
milestone of the viewer does.
"""

__all__ = ["create_app"]


def __getattr__(name):
    if name == "create_app":
        from .server import create_app
        return create_app
    raise AttributeError(name)
