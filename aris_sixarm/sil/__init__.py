"""Software-in-the-loop simulator for the deployed drawing stack.

WHAT THIS ANSWERS.  The physical pencil tip is not where the robot believes it
is.  `setEE` puts the end effector at the NOMINAL tip; the graphite wears, the
holder slips, the protrusion was set by eye.  This package puts the deployed
Cartesian impedance law (a numpy replica of the live C++, contract §5) and the
executor's press-into-paper geometry (an open-loop replica, briefing §8.5) on
top of a Drake plant with real contact, and measures what a tip-length error
does to contact force, penetration, air time and tracking.

    python -m aris_sixarm.sil --csv aris_sixarm/sil/examples/line10cm_arm31.csv \\
        --rig proposed --arm 31 --tool lateral --compare 0.0 -0.010 \\
        --out out/sil/line10cm

NO ROS.  Nothing here imports `rclpy`; the ROS wrapper that closes the loop
with the real executor is a later task and lives in `ros_node.py`.

Public API:
    SilParams, ArmMount, Tool, Paper, Joints, Controller, Walker   parameters
    build_simulator(params) -> SilPlant                            the plant
    PathwayWalker, SetpointLog, Setpoint                           the stream
    CartesianImpedanceLaw                                          the law
    run(params, source) -> Trace                                   one run
    Trace, metrics(trace), plot(trace, png)                        the output
"""
from .impedance import CartesianImpedanceLaw
from .params import (ArmMount, Controller, Joints, Paper, SilParams, Tool,
                     Walker, DEFAULT_CONTROLLER_YAML)
from .plant import SilPlant, build_simulator
from .setpoints import (PathwayWalker, Setpoint, SetpointLog, read_csv_v2,
                        runs_of)
from .simulator import Activation, run, seed_configuration
from .trace import (Trace, format_metrics, format_per_stroke, metrics,
                    per_stroke_metrics, plot)

__all__ = [
    "ArmMount", "Activation", "CartesianImpedanceLaw", "Controller",
    "DEFAULT_CONTROLLER_YAML", "Joints", "Paper", "PathwayWalker",
    "Setpoint", "SetpointLog", "SilParams", "SilPlant", "Tool", "Trace",
    "Walker", "build_simulator", "format_metrics", "format_per_stroke",
    "metrics", "per_stroke_metrics", "plot", "read_csv_v2", "run", "runs_of",
    "seed_configuration",
]
