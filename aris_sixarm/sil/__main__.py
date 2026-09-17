"""CLI: run one pathway CSV through the SIL and report the metrics.

    .venv/bin/python -m aris_sixarm.sil \\
        --csv aris_sixarm/sil/examples/line10cm_arm31.csv \\
        --rig proposed --arm 31 --tool lateral --tip-error -0.010 \\
        --out out/sil/line10cm_arm31

    # the study: the same stroke with a correct and a 1 cm short pencil
    .venv/bin/python -m aris_sixarm.sil --csv <csv> --compare 0.0 -0.010 \\
        --out out/sil/line10cm_arm31

    # a whole plan on a synthetic mount, deployed stage-0 nullspace, per stroke
    .venv/bin/python -m aris_sixarm.sil --csv out/pathways/<plan>.csv \\
        --mount inverted --h 0.94 --tool lateral --tip-error 0.0 \\
        --nullspace latched --per-stroke --out out/sil/<plan>

Writes `trace_<tag>.npz`, `setpoints_<tag>.npz`, `plot_<tag>.png` and one
`summary.json` per invocation, and prints the metric table.  `--rig` and
`--tool` default to the ACTIVE rig and tool, i.e. they honour `ARIS_RIG` and
`ARIS_TOOL` exactly as the rest of the package does; nothing else here reads
the environment.

TWO DEFAULTS THAT DEPEND ON THE INPUT, because getting them wrong is silent:
  the SHEET  -- `--rig` gets that rig's canvas (corner at the world origin);
                `--mount` gets a 4 x 4 m sheet centred on the base, because a
                synthetic mount sits at the origin and the canvas would cover
                one quadrant of its reach.  `--paper-size` / `--paper-center`
                override either.
  the NULLSPACE -- `reference` (contract §2) when the stream carries q columns,
                `latched` (today's deployed law) when it does not.
                `--nullspace` forces it either way.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .. import fleet, frames
from .params import (ArmMount, Controller, Joints, Paper, SilParams, Tool,
                     Walker)
from .setpoints import PathwayWalker, SetpointLog
from .simulator import run
from .trace import (format_metrics, format_per_stroke, metrics,
                    per_stroke_metrics, plot)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m aris_sixarm.sil",
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    src = ap.add_argument_group("input")
    src.add_argument("--csv", type=Path,
                     help="pathway CSV v2 to walk (contract §1)")
    src.add_argument("--setpoint-log", type=Path,
                     help="replay a recorded (t, pose, q_ref) stream instead")

    arm = ap.add_argument_group("arm and tool")
    arm.add_argument("--rig", default=fleet.ACTIVE_RIG, choices=fleet.RIG_NAMES)
    arm.add_argument("--arm", type=int, default=31, help="arm id in that rig")
    arm.add_argument("--mount", choices=("floor", "inverted"),
                     help="ignore --rig/--arm and synthesise a mount")
    arm.add_argument("--h", type=float, default=0.94,
                     help="m, base height above the paper for --mount")
    arm.add_argument("--tool", default=frames.ACTIVE_TOOL,
                     choices=("inline", "lateral"))
    arm.add_argument("--tip-error", type=float, default=-0.010,
                     help="m along EE +Z; negative = the physical tip is "
                          "closer to the hand than the robot believes")
    arm.add_argument("--compare", type=float, nargs="+", metavar="TIP_ERROR",
                     help="run several tip errors and tabulate side by side")

    paper = ap.add_argument_group("paper and contact")
    paper.add_argument("--paper-z", type=float, default=0.0,
                       help="m, the sheet's top surface in the world frame")
    paper.add_argument("--paper-size", type=float, nargs=2, metavar=("SX", "SY"),
                       help="m; default = the rig canvas for --rig, or 4x4 "
                            "centred on the base for --mount")
    paper.add_argument("--paper-center", type=float, nargs=2,
                       metavar=("X", "Y"), help="m, world frame")
    paper.add_argument("--stiffness", type=float, default=1.0e5,
                       help="N/m, compliant point-contact stiffness")
    paper.add_argument("--dissipation", type=float, default=20.0,
                       help="s/m, Hunt-Crossley dissipation")
    paper.add_argument("--friction", type=float, nargs=2, default=(0.6, 0.5),
                       metavar=("STATIC", "DYNAMIC"))

    ctl = ap.add_argument_group("controller")
    ctl.add_argument("--controller-yaml", type=Path,
                     help="the deployed schema; default = the shipped replica")
    ctl.add_argument("--k", type=float, nargs=6, metavar="K",
                     help="override k_cartesian (x y z rx ry rz)")
    ctl.add_argument("--nullspace", choices=("latched", "reference"),
                     help="nullspace target: `latched` = today's deployed "
                          "stage-0 behaviour (posture at activation, q columns "
                          "ignored); `reference` = contract §2. Default: "
                          "`reference` when the CSV carries q, else `latched`")
    ctl.add_argument("--qref-rate-limit", type=float, default=1.0,
                     help="rad/s cap on how fast the effective nullspace "
                          "target may follow the reference; <= 0 disables")
    ctl.add_argument("--joint-viscous", type=float, default=0.0,
                     help="Nm.s/rad on every joint (UNCALIBRATED, see Joints)")
    ctl.add_argument("--joint-coulomb", type=float, default=0.0,
                     help="Nm on every joint (UNCALIBRATED, see Joints)")
    ctl.add_argument("--settle", type=float, default=0.5,
                     help="s held at the activation pose before the stream")

    walk = ap.add_argument_group("executor walk (briefing §8.5)")
    walk.add_argument("--hover", type=float, default=0.030)
    walk.add_argument("--press", type=float, default=0.010)
    walk.add_argument("--dmax", type=float, default=0.012)
    walk.add_argument("--draw-speed", type=float, default=0.02)
    walk.add_argument("--travel-speed", type=float, default=0.04)
    walk.add_argument("--touchdown-speed", type=float, default=0.002)
    walk.add_argument("--lift-max", type=float, default=0.015)

    out = ap.add_argument_group("output")
    out.add_argument("--out", type=Path, default=Path("out/sil/run"))
    out.add_argument("--no-plot", action="store_true")
    out.add_argument("--per-stroke", action="store_true",
                     help="also print the per-stroke table for each run")
    return ap


def walker_for(args) -> Walker:
    return Walker(hover_m=args.hover, press_m=args.press, dmax_m=args.dmax,
                  draw_speed=args.draw_speed, travel_speed=args.travel_speed,
                  touchdown_speed=args.touchdown_speed,
                  lift_max_m=args.lift_max)


def params_for(args, tip_error: float, has_q: bool) -> SilParams:
    """Assemble a `SilParams` from parsed CLI arguments.

    `has_q` decides the nullspace default: the joint reference of contract §2
    only exists when the stream carries one, so a file without q columns runs
    the deployed stage-0 law and a file with them runs stage 1.
    """
    friction = dict(point_stiffness=args.stiffness,
                    dissipation=args.dissipation,
                    friction_static=args.friction[0],
                    friction_dynamic=args.friction[1])
    if args.mount:
        mount = ArmMount.synthetic(args.mount, args.h, args.arm)
        paper = Paper.under(mount, args.paper_z, **friction)
    else:
        mount = ArmMount.from_fleet(args.rig, args.arm)
        paper = Paper.for_rig(args.rig, args.paper_z, **friction)
    if args.paper_size:
        paper = replace(paper, size=tuple(args.paper_size))
    if args.paper_center:
        paper = replace(paper, center_xy=tuple(args.paper_center))

    controller = Controller.from_yaml(args.controller_yaml)
    if args.k:
        controller = controller.with_k(args.k)
    controller = replace(
        controller,
        nullspace_mode=args.nullspace or ("reference" if has_q else "latched"),
        qref_rate_limit_rad_s=args.qref_rate_limit)
    return SilParams(
        mount=mount, tool=Tool.from_frames(args.tool, tip_error), paper=paper,
        controller=controller, walker=walker_for(args),
        joints=Joints(viscous=np.full(7, args.joint_viscous),
                      coulomb=np.full(7, args.joint_coulomb)),
        settle_s=args.settle)


def source_for(args):
    if args.setpoint_log:
        return SetpointLog(args.setpoint_log)
    if not args.csv:
        raise SystemExit("one of --csv / --setpoint-log is required")
    return PathwayWalker(args.csv, walker_for(args))


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    errors = args.compare if args.compare else [args.tip_error]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    table, summary = {}, {}
    for tip_error in errors:
        tag = f"tip{tip_error * 1e3:+.0f}mm"
        source = source_for(args)
        params = params_for(args, tip_error, source.has_joint_columns)
        print(f"[sil] {tag}: {params.mount.rig} arm {params.mount.arm_id} "
              f"({params.mount.mount}), tool {params.tool.name}, "
              f"press {1e3 * params.walker.press_applied_m:.1f} mm, "
              f"nullspace {params.controller.nullspace_mode}"
              f"@{params.controller.qref_rate_limit_rad_s:g} rad/s, "
              f"paper {params.paper.size[0]:.2f}x{params.paper.size[1]:.2f} m, "
              f"{source.duration_s + params.settle_s:.1f} s to simulate")
        trace = run(params, source, label=tag)
        m = metrics(trace)
        table[tag] = m
        summary[tag] = {"metrics": m, "meta": trace.meta,
                        "per_stroke": per_stroke_metrics(trace)}
        trace.save(out / f"trace_{tag}.npz")
        trace.save_setpoint_log(out / f"setpoints_{tag}.npz")
        if not args.no_plot:
            plot(trace, out / f"plot_{tag}.png")
        if args.per_stroke:
            print(f"\nper stroke, {tag}:")
            print(format_per_stroke(trace))

    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print()
    print(format_metrics(table))
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
