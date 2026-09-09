"""The execution adapter: a conducted schedule -> something that could be flown.

    from aris_sixarm.execute import from_schedule, play, MeshcatDryRun
    prog = from_schedule("out/csail_schedule_h094_v18.npz",
                         "out/csail_program_h094_v18.json")
    print("\\n".join(prog.report()))
    play(prog, MeshcatDryRun())            # rehearsal, at wall-clock speed

WHERE THE PLANNER ENDS AND THIS BEGINS.  `coordination.conduct` produces the
fleet timeline; `scene_check` grades it; `csail_schedule` writes it to an npz
and `program_schema.export_bundle` turns the same npz into a scrubber for the
browser.  This package is the third consumer of that file and the only one
whose output is meant to be commanded: it re-expresses the timeline as per-arm
`JointTrajectory` objects on one shared clock, marks the pen swaps as
`Barrier`s, and hands them to a `Backend`.

WHAT IS DELIBERATELY ABSENT.  No live control.  `Fr3BundleBackend` is a
skeleton whose every robot-touching method raises, and it will stay that way
until `docs/HARDWARE_LADDER.md`'s rung 1 has produced a measured pen-tip
transform: a stiff position controller flying a plan whose tip offset is
user-specified from a photograph is the one combination this project must not
build.
"""
from .backends import (INSTALLATION_IPS, Backend,                 # noqa: F401
                       Fr3BundleBackend, LibfrankaBackend, MeshcatDryRun,
                       RateKeeper, RecordingBackend)
from .program import (Barrier, DECIMATED_NOTE, FleetProgram,      # noqa: F401
                      PhaseTrack, SoloProgram, from_schedule)
from .runner import RunLog, confirm_console, dry_run, play        # noqa: F401
from .trajectory import (JOINT_MARGIN, SPEED_SAFETY, Governor,    # noqa: F401
                         JointTrajectory)

__all__ = [
    "Backend", "Barrier", "DECIMATED_NOTE", "FleetProgram", "Fr3BundleBackend",
    "Governor", "INSTALLATION_IPS", "JOINT_MARGIN", "JointTrajectory",
    "LibfrankaBackend", "MeshcatDryRun",
    "PhaseTrack", "RateKeeper", "RecordingBackend", "RunLog", "SPEED_SAFETY",
    "SoloProgram", "confirm_console", "dry_run", "from_schedule", "play",
]
