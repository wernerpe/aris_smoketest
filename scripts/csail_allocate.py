#!/usr/bin/env python3
"""Trace the CSAIL logo, allocate it to the active fleet, and draw the result.

    python3 scripts/csail_allocate.py [--no-scene] [--margin 0.06]

Writes
    out/csail_program.json   per-arm certified segment programs + dropped list
    out/csail_allocation.png the sheet, coloured by the arm that draws it,
                             dropped spans dashed
    out/csail_scene.html     static meshcat: paper, the four active arms posed
                             mid-stroke, every allocated stroke on the paper

Single-arm-sequential throughout: this is an ALLOCATION plus a certified plan
per segment.  No multi-arm timing, no collision reasoning between arms, and
the pen-up transits are counted but not planned.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402
from matplotlib.lines import Line2D          # noqa: E402

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from aris_sixarm import allocate, idle, pwl, trace, writing   # noqa: E402
from aris_sixarm.fleet import FLEET, SHEET, H_INV_DEFAULT  # noqa: E402
from csail_trace import sheet_axes           # noqa: E402

# the logo's own two inks, straight off the source image (trace.GREY_RGB /
# trace.ORANGE_RGB); one of them is what each arm's pen is filled with
PEN = {"grey": "#%02x%02x%02x" % trace.GREY_RGB,
       "orange": "#%02x%02x%02x" % trace.ORANGE_RGB}


def _hex(rgb):
    return int(rgb[0] * 255) << 16 | int(rgb[1] * 255) << 8 | int(rgb[2] * 255)


def _dense(pts, ds=0.0015):
    p = np.asarray(pts, float)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    t = np.concatenate([[0], np.cumsum(seg)])
    if t[-1] <= 0:
        return p
    s = np.arange(0, t[-1], ds)
    return np.column_stack([np.interp(s, t, p[:, 0]), np.interp(s, t, p[:, 1])])


# ---------------------------------------------------------------------------
def allocation_png(phases, strokes, path):
    """One sheet, coloured by the arm that draws it; a phase per line style.

    Two passes are drawn on ONE picture on purpose: the question the picture
    answers is "who draws what", and an arm that draws grey before the swap and
    orange after is the whole point of drawing in two passes.  Phase 2 is
    dashed so the two are still tellable apart at a glance.
    """
    fig, ax = plt.subplots(figsize=(14.2, 8.8))
    sheet_axes(ax)
    used_arms = sorted({a for p in phases for a in p["arms"]})
    for ph in phases:
        for d in ph["dropped"]:
            p = np.asarray(d["pts"], float)
            ax.plot(p[:, 0], p[:, 1], ls=(0, (3, 3)), color="#c9c9c9", lw=2.0,
                    zorder=2)
    for k, ph in enumerate(phases):
        style = "-" if k == 0 else (0, (6, 2))
        for aid in ph["arms"]:
            c = FLEET[aid].color
            for s in ph["programs"][aid]:
                p = s["pts"]
                ax.plot(p[:, 0], p[:, 1], ls=style, color=c, lw=3.0, zorder=4,
                        solid_capstyle="round", solid_joinstyle="round")
                ax.plot(p[0, 0], p[0, 1], "o", color=c, ms=4.5, mec="white",
                        mew=0.9, zorder=5)
    # filled = used by THIS run; "(parked)" = the registry says it is down today
    for aid, spec in FLEET.items():
        used = aid in used_arms
        ax.plot(*spec.xy, marker="o" if spec.mount == "floor" else "s", ms=14,
                mfc=spec.color if used else "white", mec=spec.color,
                mew=2.2, zorder=6, clip_on=False)
        ax.annotate(str(aid), spec.xy, color="white" if used else spec.color,
                    fontsize=7.5, fontweight="bold", ha="center", va="center",
                    zorder=7, clip_on=False)
        note = ("" if spec.active else "(parked)") if used else "not in this run"
        if note:
            ax.annotate(note, (spec.xy[0], spec.xy[1] - 0.13), color=spec.color,
                        fontsize=7, ha="center", va="top", alpha=0.7, zorder=7,
                        clip_on=False)

    T = totals(phases)
    handles, labels = [], []
    for aid in used_arms:
        pen = next(p["pens"][aid] for p in phases if aid in p["pens"])
        bits, m = [], 0.0
        for ph in phases:
            segs = ph["programs"].get(aid, [])
            m += sum(s["length"] for s in segs)
            if segs:
                bits.append(f"{ph['ink'] or ph['colors'][aid]} "
                            f"{sum(s['length'] for s in segs):.2f} m "
                            f"({len(segs)} seg)")
        handles.append(Line2D([], [], color=FLEET[aid].color, lw=3.4))
        labels.append(f"arm {aid} ({FLEET[aid].name}) — {1000 * pen:.0f} mm pen — "
                      + (" then ".join(bits) if bits
                         else "reaches none of the logo"))
    handles.append(Line2D([], [], color="#c9c9c9", lw=2.0, ls=(0, (3, 3))))
    labels.append(f"left empty (out of reach) — {T['dropped']:.3f} m, "
                  f"{100 * T['dropped'] / max(T['traced'], 1e-9):.2f} % of "
                  f"{T['traced']:.2f} m traced, {T['n_dropped']} spans")
    if len(phases) > 1:
        handles.append(Line2D([], [], color="#555555", lw=2.4, ls=(0, (6, 2))))
        labels.append("phase 2 (after the pen swap); solid = phase 1")
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.015),
              ncol=2, fontsize=9, framealpha=0.94, borderpad=0.8,
              labelspacing=0.5, columnspacing=2.0)
    ax.set_title(f"CSAIL logo — allocation to {len(used_arms)} arms in "
                 f"{len(phases)} pass{'' if len(phases) == 1 else 'es'} "
                 f"({100 * T['covered']:.2f} % of {T['traced']:.2f} m traced, "
                 f"{T['n_segments']} certified segments); colour = arm",
                 fontsize=11.5)
    fig.tight_layout()
    fig.savefig(path, dpi=135)
    plt.close(fig)


# ---------------------------------------------------------------------------
def final_png(phases, strokes, path, title=None):
    """The end state: the paper as it looks when every arm has stopped.

    Ink is drawn in the PEN colour (the logo's own grey/orange), not the arm
    colour, because that is what is on the paper; what nobody reached stays a
    dashed ghost so the holes are visible rather than merely absent.
    """
    fig, ax = plt.subplots(figsize=(13.0, 7.6))
    sheet_axes(ax)
    for ph in phases:
        for d in ph["dropped"]:
            p = np.asarray(d["pts"], float)
            ax.plot(p[:, 0], p[:, 1], ls=(0, (4, 4)), color="#cfcfcf", lw=1.8,
                    zorder=2)
    drawn = {"grey": 0.0, "orange": 0.0}
    for ph in phases:
        for aid in ph["arms"]:
            for s in ph["programs"][aid]:
                p = s["pts"]
                drawn[s["color"]] += s["length"]
                ax.plot(p[:, 0], p[:, 1], "-", color=PEN[s["color"]], lw=3.6,
                        zorder=4, solid_capstyle="round", solid_joinstyle="round")
    T = totals(phases)
    handles = [Line2D([], [], color=PEN["grey"], lw=4),
               Line2D([], [], color=PEN["orange"], lw=4)]
    labels = [f"grey ink #{PEN['grey'][1:]} — {drawn['grey']:.2f} m",
              f"orange ink #{PEN['orange'][1:]} — {drawn['orange']:.2f} m"]
    if T["dropped"] > 0:
        handles.append(Line2D([], [], color="#cfcfcf", lw=1.8, ls=(0, (4, 4))))
        labels.append(f"left empty — {T['dropped']:.3f} m, "
                      f"{100 * T['dropped'] / max(T['traced'], 1e-9):.2f} % of "
                      f"{T['traced']:.2f} m traced, {T['n_dropped']} spans")
    else:
        handles.append(Line2D([], [], color="none"))
        labels.append(f"nothing left empty — {T['traced']:.2f} m traced, "
                      f"{T['traced']:.2f} m certified")
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=3, fontsize=9.5, framealpha=0.94, borderpad=0.7)
    ax.set_title(title or ("CSAIL logo — the paper when the arms stop "
                           f"({100 * T['covered']:.2f} % of {T['traced']:.2f} m "
                           "traced, every metre plan_stroke-certified)"),
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
LEGEND = """
<div style="position:fixed;top:12px;left:12px;z-index:1000;background:rgba(255,255,255,0.93);
border:1px solid #bbb;border-radius:8px;padding:10px 14px;font:12px/1.55 sans-serif;color:#222;max-width:360px">
<b>Aris Kindt &mdash; CSAIL logo, allocated</b><br>
%(rows)s
<hr style="margin:6px 0">
<span style="color:#b0b0b0">&#9632;</span> left empty &mdash; %(drop).2f m,
%(dropf).1f %% of the %(tot).2f m traced (no active arm reaches it)<br>
<hr style="margin:6px 0">
%(nstroke)d traced strokes, %(nseg)d certified segments, %(ncut)d handoff cuts.
Each arm carries ONE pen colour for the whole piece; the grey/orange split of
the %(narm)d arms is the best of the %(npart)d non-trivial partitions.
Arms are posed mid-stroke on a segment of their own program (pen tip on the
paper, pen = TCP + 0.110 m).<br>
%(note)s
</div>
"""


def scene_html(phases, strokes, path, h_inv=H_INV_DEFAULT):
    res = phases[0]
    used = sorted({a for p in phases for a in p["arms"]})
    import meshcat
    import meshcat.geometry as g
    import meshcat.transformations as tf
    from aris_sixarm.viz import robot_model
    from aris_sixarm.frames import tip_pos

    links, joints = robot_model.load_model()
    vis = meshcat.Visualizer()
    vis["/Background"].set_property("top_color", [0.95, 0.95, 0.97])
    vis["/Background"].set_property("bottom_color", [0.85, 0.85, 0.9])
    vis["paper"].set_object(g.Box([SHEET[0], SHEET[1], 0.004]),
                            g.MeshLambertMaterial(color=0xFAFAF5))
    vis["paper"].set_transform(tf.translation_matrix([SHEET[0] / 2, SHEET[1] / 2, -0.002]))
    vis["table"].set_object(g.Box([SHEET[0] + 0.45, SHEET[1] + 0.25, 0.05]),
                            g.MeshLambertMaterial(color=0x8A7358))
    vis["table"].set_transform(tf.translation_matrix([SHEET[0] / 2, SHEET[1] / 2, -0.029]))

    def cloud(path_, pts_xy, rgb, z, size):
        if not len(pts_xy):
            return
        xyz = np.column_stack([pts_xy[:, 0], pts_xy[:, 1], np.full(len(pts_xy), z)])
        col = np.tile(np.asarray(rgb, float), (len(pts_xy), 1))
        vis[path_].set_object(g.PointCloud(position=xyz.T.astype(np.float32),
                                           color=col.T.astype(np.float32), size=size))

    dr = [_dense(d["pts"]) for p in phases for d in p["dropped"]]
    if dr:
        cloud("dropped", np.vstack(dr), [0.80, 0.80, 0.80], 0.003, 0.006)
    for aid in used:
        segs = [x for p in phases for x in p["programs"].get(aid, [])]
        spec = FLEET[aid]
        col = np.array(spec.color)
        if segs:
            cloud(f"strokes/arm{aid}", np.vstack([_dense(s["pts"]) for s in segs]),
                  col, 0.005, 0.009)
        Twb = spec.T_world_base(h_inv)
        vis[f"bases/arm{aid}"].set_object(g.Sphere(0.045),
                                          g.MeshLambertMaterial(color=_hex(col)))
        vis[f"bases/arm{aid}"].set_transform(Twb)
        if spec.mount == "inv":
            vis[f"booms/arm{aid}"].set_object(
                g.Cylinder(1.3, 0.012),
                g.MeshLambertMaterial(color=_hex(col), opacity=0.55))
            vis[f"booms/arm{aid}"].set_transform(
                tf.translation_matrix([spec.xy[0], spec.xy[1], Twb[2, 3] - 0.65])
                @ tf.rotation_matrix(np.pi / 2, [1, 0, 0]))
        if segs:      # posed mid-stroke on the arm's longest segment
            s = max(segs, key=lambda s: s["length"])
            q = np.asarray(s["plan"]["qs"])[len(s["plan"]["qs"]) // 2]
            tip = Twb[:3, :3] @ tip_pos(q) + Twb[:3, 3]
            assert abs(tip[2]) < 2e-3, f"arm {aid} pose is not on the paper: {tip}"
        else:         # nothing allocated: show it parked at its ready pose
            q = spec.q_seed
        robot_model.add_robot(vis, f"robots/arm{aid}", links, joints, q, Twb,
                              pen_color=_hex(col))

    rows = []
    for aid in used:
        segs = [x for p in phases for x in p["programs"].get(aid, [])]
        ids = {s["stroke_id"] for s in segs}
        c = FLEET[aid].color
        pen = next(p["pens"][aid] for p in phases if aid in p["pens"])
        inks = " then ".join(p["ink"] or p["colors"][aid] for p in phases
                             if p["programs"].get(aid))
        rows.append(
            f'<span style="color:rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})">'
            f"&#9632;</span> arm {aid} {FLEET[aid].name} &mdash; "
            f"<b>{1000 * pen:.0f} mm</b> pen &mdash; <b>{inks or 'idle'}</b> &mdash; "
            f"{sum(s['length'] for s in segs):.2f} m, {len(ids)} strokes, "
            f"{len(segs) - len(ids)} cuts")
    T = totals(phases)
    n_seg = T["n_segments"]
    n_cut = n_seg - sum(len({s["stroke_id"] for s in p["programs"][a]})
                        for p in phases for a in p["arms"])
    parked = [a for a in used if not FLEET[a].active]
    html = vis.static_html().replace("</body>", LEGEND % dict(
        rows="<br>".join(rows), drop=T["dropped"], tot=T["traced"],
        dropf=100 * T["dropped"] / max(T["traced"], 1e-9),
        nstroke=len(strokes), nseg=n_seg, ncut=n_cut, narm=len(used),
        npart=(1 << len(used)) - 2,
        note=("Single-arm-sequential: no multi-arm timing, transits not planned."
              if not parked else
              f"HYPOTHETICAL FLEET: arm{'s' if len(parked) > 1 else ''} "
              f"{', '.join(str(a) for a in parked)} "
              f"{'are' if len(parked) > 1 else 'is'} parked in the registry and "
              "brought up for this run only.")) + "</body>")
    open(path, "w").write(html)
    return len(html)


# ---------------------------------------------------------------------------
def _phase_json(res):
    out = dict(
        name=res["name"], ink=res["ink"],
        colors={str(a): res["colors"][a] for a in res["arms"]},
        pens_mm={str(a): round(1000 * res["pens"][a], 1) for a in res["arms"]},
        traced_m=res["total_len"], drawn_m=res["drawn_len"],
        dropped_m=res["dropped_len"], n_probes=res["n_probes"],
        draw_speed=res.get("draw_speed"),
        balance=(None if not res.get("balance") else dict(
            n_movable=int(res["balance"]["n_movable"]),
            rounds=int(res["balance"]["rounds"]),
            n_splits=int(res["balance"].get("n_splits", 0)),
            n_segments_before=int(res["balance"].get("n_segments_before", 0)),
            n_segments_after=int(res["balance"].get("n_segments_after", 0)),
            coverage_lost_m=float(res["balance"].get("coverage_lost_m", 0.0)),
            max_before_s=float(res["balance"]["max_before"]),
            max_after_s=float(res["balance"]["max_after"]),
            loads_before_s={str(k): float(v)
                            for k, v in res["balance"]["loads_before"].items()},
            loads_after_s={str(k): float(v)
                           for k, v in res["balance"]["loads_after"].items()})),
        wall_s=res["timing"]["total"],
        transit_s=float(sum(res["transit_time"].values())),
        baseline_transit_s=float(sum(q["baseline_cost"]
                                     for q in res["sequence"].values())),
        sequence={str(a): dict(
            method=q["method"], n=int(q["n"]),
            order=[int(x) for x in q["order"]], dirs=[int(x) for x in q["dirs"]],
            transit_s=float(q["cost"]), baseline_transit_s=float(q["baseline_cost"]),
            n_reversed=int(q["n_reversed"]), n_refused=int(q["n_refused"]),
            wall_s=float(q["wall"])) for a, q in res["sequence"].items()},
        arms={}, dropped=[])
    for aid in res["arms"]:
        out["arms"][str(aid)] = [dict(
            stroke_id=int(s["stroke_id"]), seg=i, color=s["color"],
            kind=s["kind"], s_range=[float(x) for x in s["s_range"]],
            direction=int(s["direction"]),
            flipped=bool(s.get("flipped", False)), length_m=float(s["length"]),
            n_points=int(len(s["pts"])), plan_ok=s["plan"]["status"] == "ok",
            validated=bool(s["plan"]["validation"]["ok"]),
            pen_ext_m=float(res["pens"][aid]),
            n_dense=int(s["plan"]["n_dense"]), n_knots=int(s["plan"]["n_knots"]),
            min_sigma=float(s["plan"]["min_sigma"]),
            min_margin=float(s["plan"]["min_margin"]),
            tip_err_m=float(s["plan"]["tip_err"]),
            draw_time_s=float(s["plan"]["total_time"]),
            pts=np.round(s["pts"], 5).tolist(),
            q_first=list(np.round(s["plan"]["qs"][0], 6)),
            q_last=list(np.round(s["plan"]["qs"][-1], 6)))
            for i, s in enumerate(res["programs"][aid])]
    for d in res["dropped"]:
        out["dropped"].append(dict(
            stroke_id=int(d["stroke_id"]), color=d["color"], kind=d["kind"],
            s_range=[float(x) for x in d["s_range"]], length_m=float(d["length"]),
            at=[float(x) for x in d["at"]],
            pts=np.round(d["pts"], 5).tolist()))
    return out


def program_json(phases, strokes, info, path):
    T = totals(phases)
    pens = {}
    for p in phases:
        pens.update({str(a): round(1000 * p["pens"][a], 1) for a in p["arms"]})
    doc = dict(
        sheet=list(SHEET), h_inv=H_INV_DEFAULT, n_phases=len(phases),
        two_pass=len(phases) > 1, pens_mm=pens,
        logo={k: ([float(x) for x in v] if isinstance(v, (tuple, list))
                  else v if isinstance(v, bool) else float(v))
              for k, v in info.items()},
        totals=dict(traced_m=T["traced"], drawn_m=T["drawn"],
                    dropped_m=T["dropped"],
                    dropped_pct=100 * T["dropped"] / max(T["traced"], 1e-9),
                    coverage_pct=100 * T["covered"],
                    n_strokes=len(strokes), n_segments=T["n_segments"],
                    n_probes=int(sum(p["n_probes"] for p in phases)),
                    wall_s=float(sum(p["timing"]["total"] for p in phases)),
                    transit_s=float(sum(v for p in phases
                                        for v in p["transit_time"].values())),
                    baseline_transit_s=float(sum(
                        q["baseline_cost"] for p in phases
                        for q in p["sequence"].values()))),
        strokes=[dict(id=s["id"], color=s["color"], kind=s["kind"],
                      length=trace.plen(s["pts"])) for s in strokes],
        sequencer=phases[0].get("sequencer", "opt"),
        phases=[_phase_json(p) for p in phases])
    # the single-pass keys the older readers use, kept pointing at phase 1
    doc["colors"] = doc["phases"][0]["colors"]
    doc["arms"] = doc["phases"][0]["arms"]
    doc["dropped"] = [d for p in doc["phases"] for d in p["dropped"]]
    doc["sequence"] = doc["phases"][0]["sequence"]
    with open(path, "w") as f:
        json.dump(doc, f)
    return doc


def add_args(ap):
    """The trace + placement + fleet arguments, shared with the demo scripts."""
    ap.add_argument("--image", default=str(ROOT / "assets/csail/csail_old_med.gif"))
    ap.add_argument("--margin", type=float, default=0.06)
    ap.add_argument("--out", default=str(ROOT / "out"))
    ap.add_argument("--arms", default=None,
                    help="'all', or e.g. '13,31,97' — a HYPOTHETICAL fleet state "
                         "for this run (default: the registry's active flags)")
    ap.add_argument("--placement", default=None,
                    help="csail_placement_*.json from scripts/csail_place.py")
    ap.add_argument("--target-width", type=float, default=None)
    ap.add_argument("--offset", type=float, nargs=2, default=None, metavar=("DX", "DY"))
    ap.add_argument("--no-prefilter", action="store_true")
    # these two are the SEQUENCER's cost model as much as the timeline's: the
    # transit it prices is the transit `writing.arm_program` will lay down, so
    # they have to be the same numbers in both places (see run_allocation).
    ap.add_argument("--transit-speed", type=float, default=writing.TRANSIT_SPEED)
    # the material's limit along a stroke.  It is the ALLOCATOR's cost model
    # too, now that the allocator balances the fleet on seconds instead of
    # metres (allocate.rebalance), so it is defined here with the other two
    # rather than only where the timeline is frozen.
    ap.add_argument("--draw-speed", type=float, default=writing.DRAW_SPEED_FLEET)
    ap.add_argument("--no-balance", action="store_true",
                    help="skip the min-max load pass and ship the raw interval "
                         "cover — which is optimal for pen-ups and blind to "
                         "the clock (see allocate.balance_loads)")
    ap.add_argument("--no-split", action="store_true",
                    help="allocation v1: balance with whole-segment moves and "
                         "swaps only.  v2 may also CUT a span on the busiest "
                         "arm and hand one piece to an arm that certifies it, "
                         "which is the only move that reaches the 48 %% of solo "
                         "drawing time docs/SOLO_TIME.md measured as splittable")
    ap.add_argument("--min-split", type=float, default=allocate.MIN_SPLIT_M,
                    help="metres; the shortest piece a cut may create")
    ap.add_argument("--qd-frac", type=float, default=writing.QD_FRAC,
                    help="fraction of the FR3 joint-velocity limit any move may use")
    ap.add_argument("--sequencer", default=allocate.SEQUENCER,
                    choices=("opt", "nn"),
                    help="'opt' (default): per-arm order AND per-segment "
                         "direction chosen to minimise transit TIME, exact to "
                         "16 segments; 'nn': the old nearest-neighbour chain "
                         "in paper distance, every segment drawn forward")
    ap.add_argument("--pens", default=None, metavar="ARM:MM,...",
                    help="per-arm pen length, e.g. '2:300,31:200,71:200,97:200'; "
                         "arms not named keep frames.PEN_EXT (110 mm).  A pen "
                         "is a fixture: it is the same length in both phases, "
                         "only the ink changes at the swap")
    ap.add_argument("--two-pass", action="store_true",
                    help="draw grey, stop for a human to swap the pens, then "
                         "draw orange.  Lifts the one-colour-per-arm constraint "
                         "ACROSS phases (not within one), so allocation becomes "
                         "two independent single-colour problems on all arms")
    ap.add_argument("--max-probes", type=int, default=3,
                    help="plan calls per (stroke, arm); above 2 they walk the "
                         "largest remaining gap (see allocate.probe_stroke)")
    # THE IDLE POLICY IS A SEQUENCING QUESTION BEFORE IT IS A CONDUCTING ONE.
    # Under "freeze" an arm does not go home when it finishes, so the tour the
    # sequencer prices must not pay for the trip — which changes WHICH segment
    # it chooses to finish on.  It lives here, with the other two halves of the
    # shared cost model, so the allocator and the schedule cannot disagree.
    # THE TWO HALVES OF THE 2026-08-20 REFACTOR, EACH WITH ITS OWN OFF SWITCH,
    # because the A/B that justifies them has to be runnable from one binary.
    ap.add_argument("--cluster", action="store_true",
                    help="sequence on (segment, direction, VARIANT): build the "
                         "entry/exit fiber menus and run the cluster DP. Off by "
                         "default — it buys transit and reconfiguration and "
                         "costs draw time; see allocate.CLUSTER")
    ap.add_argument("--band-objective", choices=pwl.OBJECTIVES,
                    default=pwl.OBJECTIVE,
                    help="the (s, q7) band objective: 'min_travel' (default) is "
                         "the gated shortest path, 'maximin_sigma' the older "
                         "bottleneck objective")
    ap.add_argument("--idle-policy", choices=(idle.POLICY_FREEZE,
                                              idle.POLICY_HOME),
                    default=idle.POLICY_FREEZE,
                    help="what an arm does when it finishes: 'freeze' (default) "
                         "lifts the pen and stops where it is; 'home' is "
                         "conductor v1's transit back to the ready pose")
    return ap


def _override(arms_arg):
    if arms_arg is None:
        return None
    return arms_arg if arms_arg == "all" else [int(x) for x in arms_arg.split(",")]


def parse_pens(s):
    """'2:300,31:200' -> {2: 0.300, 31: 0.200}.  Millimetres in, metres out."""
    if not s:
        return None
    out = {}
    for part in s.split(","):
        arm, mm = part.split(":")
        out[int(arm)] = float(mm) / 1000.0
    unknown = set(out) - set(FLEET)
    if unknown:
        raise SystemExit(f"--pens names arms not in the fleet: {sorted(unknown)}")
    return out


def run_allocation(a, verbose=False, split=None):
    """Trace -> place -> allocate. -> (phases, strokes, info).

    One code path for the allocation PNG/JSON and for the animation, so the
    programme that gets scheduled is the programme that gets reported.

    `phases` is a LIST because the piece may be drawn in two passes with a pen
    swap between them.  Single-pass is the one-element case and takes the same
    path through every writer downstream; nothing here special-cases it.  Each
    phase is a full `allocate.allocate` result with `name`, `ink` and its own
    stroke subset attached.
    """
    tw, off = a.target_width, a.offset
    if a.placement:
        doc = json.loads(Path(a.placement).read_text())["chosen"]
        tw = tw if tw is not None else doc["target_width"]
        off = off if off is not None else doc["offset"]
    pens = parse_pens(getattr(a, "pens", None))
    t0 = time.time()
    px, _ = trace.trace_logo(a.image)
    strokes, info = trace.to_sheet(px, SHEET, margin=a.margin, target_width=tw,
                                   offset=tuple(off or (0.0, 0.0)))
    if not info["fits"]:
        raise SystemExit(f"placement does not fit the sheet: {info}")
    print(f"traced {len(strokes)} strokes, {trace.total_length(strokes):.2f} m, "
          f"logo {info['logo_w']:.3f} x {info['logo_h']:.3f} m at "
          f"({info['center'][0]:.3f}, {info['center'][1]:.3f})  ({time.time() - t0:.2f} s)")
    if pens:
        print("  pens: " + "  ".join(f"arm {k} = {1000 * v:.0f} mm"
                                     for k, v in sorted(pens.items())))
    kw = dict(opts=dict(objective=getattr(a, "band_objective", pwl.OBJECTIVE)),
              cluster=getattr(a, "cluster", allocate.CLUSTER),
              verbose=verbose, pens=pens,
              active_override=_override(a.arms),
              sequencer=getattr(a, "sequencer", allocate.SEQUENCER),
              max_probes=getattr(a, "max_probes", 3),
              balance=not getattr(a, "no_balance", False),
              # `split=False` here is how the caller asks for the SAME ink
              # allocated without cutting, which is what the conducted A/B in
              # `csail_schedule.build_phases` needs a second copy of
              split=(not getattr(a, "no_split", False)) if split is None
                    else bool(split),
              min_split=getattr(a, "min_split", allocate.MIN_SPLIT_M),
              draw_speed=getattr(a, "draw_speed", writing.DRAW_SPEED_FLEET),
              seq_opts=dict(transit_speed=a.transit_speed, qd_frac=a.qd_frac),
              return_home=getattr(a, "idle_policy",
                                  idle.POLICY_FREEZE) == idle.POLICY_HOME,
              atlas_dir=None if a.no_prefilter else str(Path(a.out)))
    if not getattr(a, "two_pass", False):
        res = allocate.allocate(strokes, **kw)
        res.update(name="single pass", ink=None, strokes=strokes)
        return [res], strokes, info

    arms = allocate.active_arms(_override(a.arms))
    phases = []
    for ink in allocate.COLORS:
        sub = [s for s in strokes if s["color"] == ink]
        print(f"  phase {len(phases) + 1}/{len(allocate.COLORS)} — {ink} ink, "
              f"{len(sub)} strokes, {trace.total_length(sub):.2f} m, "
              f"all {len(arms)} arms available")
        r = allocate.allocate(sub, colors={x: ink for x in arms}, **kw)
        r.update(name=f"phase {len(phases) + 1}: {ink}", ink=ink, strokes=sub)
        phases.append(r)
    return phases, strokes, info


def totals(phases):
    """Traced / drawn / dropped over every phase. -> dict."""
    t = float(sum(p["total_len"] for p in phases))
    return dict(traced=t, drawn=float(sum(p["drawn_len"] for p in phases)),
                dropped=float(sum(p["dropped_len"] for p in phases)),
                covered=1.0 - float(sum(p["dropped_len"] for p in phases)) / max(t, 1e-9),
                n_segments=int(sum(len(p["programs"][x]) for p in phases
                                   for x in p["arms"])),
                n_dropped=int(sum(len(p["dropped"]) for p in phases)))


def main(argv=None):
    ap = add_args(argparse.ArgumentParser())
    ap.add_argument("--tag", default="", help="output-name suffix, e.g. _6arm")
    ap.add_argument("--final", default=None, help="also write the end-state still here")
    ap.add_argument("--alloc", default=None,
                    help="also write the allocation picture here (as well as "
                         "out/csail_allocation<tag>.png)")
    ap.add_argument("--no-scene", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    phases, strokes, info = run_allocation(a, verbose=a.verbose)
    for ph in phases:
        print(f"\n=== {ph['name']} ===")
        for line in allocate.report(ph, ph["strokes"]):
            print(line)
    T = totals(phases)
    print(f"\nALL PHASES: {T['traced']:.4f} m traced, {T['dropped']:.4f} m left "
          f"empty in {T['n_dropped']} spans -> COVERAGE {100 * T['covered']:.4f} % "
          f"over {T['n_segments']} certified segments")

    t1 = time.time()
    program_json(phases, strokes, info, out / f"csail_program{a.tag}.json")
    allocation_png(phases, strokes, out / f"csail_allocation{a.tag}.png")
    if a.alloc:
        allocation_png(phases, strokes, a.alloc)
    if a.final:
        final_png(phases, strokes, a.final)
    n = scene_html(phases, strokes, out / f"csail_scene{a.tag}.html") \
        if not a.no_scene else 0
    print(f"\nwrote {out}/csail_program{a.tag}.json, {out}/csail_allocation{a.tag}.png"
          + (f", {a.final}" if a.final else "")
          + (f", {out}/csail_scene{a.tag}.html ({n / 1e6:.1f} MB)" if n else "")
          + f"  ({time.time() - t1:.1f} s)")
    print(f"PIPELINE WALL CLOCK {time.time() - t0:.1f} s")
    return phases


if __name__ == "__main__":
    main()
