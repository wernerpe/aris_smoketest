#!/usr/bin/env python3
"""The commissioning survey -> a layout the planner can be run at.  REPORT ONLY.

    ARIS_RIG=proposed ARIS_TOOL=lateral python3 scripts/asbuilt_layout.py \
        --survey out/survey_20260910.json --out out/asbuilt_20260910.json

    --template            print an empty survey file to fill in and stop
    --survey F --check    read it, compare against the build sheet, and stop

WHY THIS IS NOT A ONE-LINE EDIT OF `layout.py`.

`docs/BUILD_SHEET.md` gives six base positions to +-10 mm, one mounting height
to +-10 mm and one orientation for all six, and says in as many words: *"If a
base ends up outside that, measure and report the as-built offset — do not
silently re-centre others."*  This is the thing that reads that report.

THREE THINGS THE SHIPPED CODE CANNOT EXPRESS, and this script is what makes
them expressible without editing a committed constant:

1. **PER-ARM HEIGHT.**  `layout.paired_grid` takes ONE `h` and
   `layout.build_fleet` writes it into all six specs.  Six plates that are
   mutually coplanar to +-3 mm are six DIFFERENT heights, and the one that
   matters for a given arm's ink is its own.

2. **PER-ARM YAW.**  `layout.study_spec` sets `yaw = 0.0` for every inverted
   arm, unconditionally — which is right, because the build sheet says all six
   are clocked identically and calls that "load-bearing".  An arm bolted 2 deg
   out is not a variant the shipped code has a way to say.

3. **THE SURVEY ALLOWANCE.**  `mounts.MOUNTS.calib` = 0.03 m is carried on
   every neighbour's body column *because nobody has measured where the bases
   are*.  A survey is exactly what retires it, and
   `feasible_workspace.rig(..., calib=...)` already takes the number.  Dropping
   30 mm of fat off five columns is the largest single thing this survey buys,
   and it is why the survey comes before anything flies.

WHAT THIS WRITES, AND WHAT IT DOES NOT.  It writes ONE json — the as-built
layout, with provenance and the deviation report — and `load_asbuilt` turns
that back into a fleet.  `aris_sixarm/layout.py` is not touched, and no
certified number is recomputed here: re-planning at the as-built rig is
`scripts/replan_at_height.py`'s job and it needs an atlas and a park set
searched at that height (see docs/HARDWARE_LADDER.md rung 0).
"""
import argparse
import dataclasses
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aris_sixarm import layout, mounts                          # noqa: E402
from aris_sixarm.frames import roty, rotz                       # noqa: E402
from aris_sixarm.layout import StudySpec                        # noqa: E402

#: `docs/BUILD_SHEET.md` §2 and §1: position and height tolerance, in metres.
TOL_XY = 0.010
TOL_H = 0.010
#: §5: plates level and mutually coplanar within +-3 mm.
TOL_COPLANAR = 0.003
#: §3: every arm clocked the same.  A survey that reports otherwise is a
#: finding, not a configuration: the certified coverage assumes uniform yaw.
TOL_YAW_DEG = 1.0

SCHEMA = 1


def build_sheet_nominal():
    """The build sheet's own table, from `layout.LAYOUT_PROPOSED`. -> {id: ...}.

    Derived rather than typed: the sheet is a rendering of the layout constant,
    so a change to the constant must move the nominal a survey is judged
    against, and typing the mm here would let those two drift apart.
    """
    lay = layout.LAYOUT_PROPOSED
    fids, iids = layout.arm_ids(lay)
    out = {}
    for aid, xy in zip(fids, lay["floor"]):
        out[int(aid)] = dict(x=float(xy[0]), y=float(xy[1]),
                             z=float(layout.Z_FLOOR_BASE), yaw_deg=0.0,
                             mount="floor")
    for aid, xy in zip(iids, lay["inv"]):
        out[int(aid)] = dict(x=float(xy[0]), y=float(xy[1]),
                             z=float(lay["h"]), yaw_deg=0.0, mount="inv")
    return out


def template():
    """An empty survey, with the nominal pre-filled as the thing to overwrite."""
    nom = build_sheet_nominal()
    return dict(
        schema=SCHEMA, kind="aris_sixarm.asbuilt_survey",
        surveyed=str(date.today()), by="",
        datum=("docs/BUILD_SHEET.md §0: origin at the marked canvas corner, "
               "x across 0..1.8034, y along 0..3.63064, z UP from the TOP "
               "SURFACE OF THE PAPER. Base xy is the JOINT-1 AXIS (centre of "
               "the base bolt circle), NOT a plate edge. Base z is the "
               "UNDERSIDE OF THE MOUNTING PLATE, which is where the robot "
               "bolts."),
        units="m, degrees",
        arms={str(a): dict(nom[a], measured=False,
                           note="overwrite x/y/z/yaw_deg with the measurement; "
                                "set measured true")
              for a in sorted(nom)},
        paper=dict(z_at_corners=[0.0, 0.0, 0.0, 0.0],
                   note="probed paper height at the four canvas corners; the "
                        "planner's z = 0 is the paper surface, so a table "
                        "that is not flat is a finding the touchdown "
                        "calibration has to carry"),
        gripper=dict(grasp_width_m=None,
                     note="the width libfranka REPORTS after the pen is "
                          "clamped. The GUI commands 0.0432 m at 70 N "
                          "(Aris_Kindt/franka_control_gui.py _PEN_GRASP_WIDTH); "
                          "a grasp only succeeds above width - epsilon_inner, "
                          "so the read-back is a LOWER BOUND on the real jaw "
                          "gap and it is what pins the holder build."),
        notes="")


def deviations(survey):
    """-> (report lines, {arm: {field: delta}}, ok).  The build sheet's own gates."""
    nom, dev, lines, ok = build_sheet_nominal(), {}, [], True
    arms = {int(k): v for k, v in survey["arms"].items()}
    unmeasured = sorted(a for a, v in arms.items() if not v.get("measured"))
    if unmeasured:
        lines.append(f"  NOT MEASURED: arms {unmeasured} still carry the "
                     "nominal; every number below is about the drawing, not "
                     "the room")
        ok = False
    zs = [float(v["z"]) for v in arms.values()]
    spread = max(zs) - min(zs)
    for a in sorted(arms):
        v, n = arms[a], nom.get(a)
        if n is None:
            lines.append(f"  arm {a}: not in the build sheet")
            ok = False
            continue
        d = {k: float(v[k]) - float(n[k]) for k in ("x", "y", "z", "yaw_deg")}
        dev[a] = d
        dxy = float(np.hypot(d["x"], d["y"]))
        flags = []
        if dxy > TOL_XY:
            flags.append(f"xy {1000 * dxy:.1f} mm > {1000 * TOL_XY:.0f}")
        if abs(d["z"]) > TOL_H:
            flags.append(f"z {1000 * d['z']:+.1f} mm > {1000 * TOL_H:.0f}")
        if abs(d["yaw_deg"]) > TOL_YAW_DEG:
            flags.append(f"yaw {d['yaw_deg']:+.2f} deg > {TOL_YAW_DEG}")
        if v.get("mount") != n["mount"]:
            flags.append(f"mount {v.get('mount')!r} != {n['mount']!r}")
        ok = ok and not flags
        lines.append(
            f"  arm {a:>3}: dx {1000 * d['x']:+6.1f}  dy {1000 * d['y']:+6.1f}  "
            f"dz {1000 * d['z']:+6.1f} mm  dyaw {d['yaw_deg']:+5.2f} deg"
            + ("   <-- " + "; ".join(flags) if flags else ""))
    lines.append(f"  plate coplanarity: {1000 * spread:.1f} mm spread "
                 f"(build sheet asks {1000 * TOL_COPLANAR:.0f})"
                 + ("" if spread <= TOL_COPLANAR else "   <-- OUT"))
    if spread > TOL_COPLANAR:
        ok = False
    return lines, dev, ok


def build_asbuilt(survey, calib=None):
    """-> ({arm: StudySpec}, meta).  A fleet at the SURVEYED poses.

    `layout.build_fleet`'s recipe, with the two things it cannot express put
    back: a per-arm z and a per-arm yaw.  The mount-hardware obstacle boxes are
    still `mounts.obstacles_for`'s, taken at the MEAN height — those boxes are
    the neighbours' plates and booms and they are a schematic either way, so
    using the mean rather than six of them is the same approximation the
    shipped rig already makes, stated here rather than hidden.
    """
    arms = {int(k): v for k, v in survey["arms"].items()}
    h_mean = float(np.mean([float(v["z"]) for v in arms.values()]))
    model = (mounts.MOUNTS if calib is None
             else dataclasses.replace(mounts.MOUNTS, calib=float(calib)))

    bare = {}
    for a in sorted(arms):
        v = arms[a]
        yaw = float(np.radians(v.get("yaw_deg", 0.0)))
        z = float(v["z"])
        if v.get("mount") == "floor":
            R = rotz(yaw)
        else:
            R = roty(np.pi) @ rotz(yaw)
        bare[a] = StudySpec(
            a, f"{v.get('mount', 'inv')}{a}", v.get("mount", "inv"),
            (float(v["x"]), float(v["y"])), yaw, True,
            layout.COLORS.get(a, (0.5, 0.5, 0.5)),
            z=z, R=tuple(np.asarray(R, float).flatten()))
    fleet = {a: StudySpec(
        s.arm_id, s.name, s.mount, s.xy, s.yaw, True, s.color,
        z=s.z, R=s.R,
        mount_boxes=tuple(mounts.obstacles_for(a, bare, h_mean, model)))
        for a, s in bare.items()}
    meta = dict(h_mean=h_mean, h_min=min(float(v["z"]) for v in arms.values()),
                h_max=max(float(v["z"]) for v in arms.values()),
                calib=float(model.calib), n_arms=len(fleet))
    return fleet, meta


def to_document(survey, calib=None):
    """The as-built layout, ready to write and to re-load. -> dict."""
    lines, dev, ok = deviations(survey)
    fleet, meta = build_asbuilt(survey, calib)
    arms = {int(k): v for k, v in survey["arms"].items()}
    return dict(
        schema=SCHEMA, kind="aris_sixarm.asbuilt_layout",
        generated_by="scripts/asbuilt_layout.py",
        survey=survey, within_build_sheet=ok, deviations_mm={
            str(a): {k: 1000.0 * v if k != "yaw_deg" else v
                     for k, v in d.items()} for a, d in dev.items()},
        report=lines, meta=meta,
        # the layout dict shape `layout.build_fleet` understands, for the arms
        # that happen to sit at one height; carried as a CONVENIENCE and marked
        # as lossy, because it cannot hold the per-arm z and yaw.
        layout_approx=dict(
            floor=[[float(arms[a]["x"]), float(arms[a]["y"])]
                   for a in sorted(arms) if arms[a].get("mount") == "floor"],
            inv=[[float(arms[a]["x"]), float(arms[a]["y"])]
                 for a in sorted(arms) if arms[a].get("mount") != "floor"],
            h=meta["h_mean"],
            lossy="per-arm z and yaw are NOT in this dict; use load_asbuilt"),
        bases={str(a): [[float(x) for x in row]
                        for row in fleet[a].T_world_base()]
               for a in sorted(fleet)})


def load_asbuilt(path, calib=None):
    """An as-built json -> ({arm: StudySpec}, doc).  The planner's entry point.

    The bases are RE-DERIVED from the survey rather than read out of `bases`,
    and then checked against it: the stored matrices are for a reader, and a
    file whose two halves disagree is refused rather than silently believed.
    """
    doc = json.loads(Path(path).read_text())
    if doc.get("kind") != "aris_sixarm.asbuilt_layout":
        raise SystemExit(f"{path}: not an as-built layout ({doc.get('kind')!r})")
    fleet, meta = build_asbuilt(doc["survey"], calib)
    for a, T in doc.get("bases", {}).items():
        got = fleet[int(a)].T_world_base()
        if not np.allclose(got, np.asarray(T, float), atol=1e-9):
            raise SystemExit(
                f"{path}: the stored base for arm {a} is not what the survey "
                "re-derives. The file has been edited in one half only.")
    return fleet, doc


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--survey", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--template", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="report the deviations and stop")
    ap.add_argument("--calib", type=float, default=None,
                    help="the unsurveyed-base allowance to carry on the "
                         "neighbours' columns (mounts.MOUNTS.calib = 0.03 by "
                         "default). A SURVEY is what earns a smaller one.")
    a = ap.parse_args(argv)

    if a.template:
        print(json.dumps(template(), indent=1))
        return 0
    if not a.survey:
        ap.error("--survey is required (or --template)")

    survey = json.loads(Path(a.survey).read_text())
    lines, _, ok = deviations(survey)
    print(f"AS-BUILT SURVEY {a.survey}  (surveyed {survey.get('surveyed')} "
          f"by {survey.get('by') or '?'})")
    print("\n".join(lines))
    print(f"  verdict: {'WITHIN the build sheet' if ok else 'OUT of tolerance — report it, do not re-centre the others'}")
    if a.check:
        return 0 if ok else 1

    doc = to_document(survey, a.calib)
    print(f"  fleet: h {doc['meta']['h_min']:.4f} .. {doc['meta']['h_max']:.4f} "
          f"(mean {doc['meta']['h_mean']:.4f}), calib {doc['meta']['calib']:.3f} m")
    out = a.out or str(Path(a.survey).with_name(
        Path(a.survey).stem + "_asbuilt.json"))
    Path(out).write_text(json.dumps(doc, indent=1))
    print(f"wrote {out}")
    print("  NEXT: an atlas and a park search AT THIS HEIGHT before anything "
          "is re-planned — scripts/height_sweep.py sweep/park, then "
          "scripts/replan_at_height.py. See docs/HARDWARE_LADDER.md rung 0.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
