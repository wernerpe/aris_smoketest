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
from aris_sixarm import allocate, trace      # noqa: E402
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
def allocation_png(res, strokes, path):
    fig, ax = plt.subplots(figsize=(14.2, 8.4))
    sheet_axes(ax)
    for d in res["dropped"]:
        p = np.asarray(d["pts"], float)
        ax.plot(p[:, 0], p[:, 1], ls=(0, (3, 3)), color="#c9c9c9", lw=2.0,
                zorder=2)
    for aid in res["arms"]:
        c = FLEET[aid].color
        for s in res["programs"][aid]:
            p = s["pts"]
            ax.plot(p[:, 0], p[:, 1], "-", color=c, lw=3.0, zorder=4,
                    solid_capstyle="round", solid_joinstyle="round")
            ax.plot(p[0, 0], p[0, 1], "o", color=c, ms=4.5, mec="white",
                    mew=0.9, zorder=5)
    # filled = used by THIS run; "(parked)" = the registry says it is down today
    for aid, spec in FLEET.items():
        used = aid in res["arms"]
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

    tot, drp = res["total_len"], res["dropped_len"]
    handles, labels = [], []
    n_arms = len(res["arms"])
    for aid in res["arms"]:
        segs = res["programs"][aid]
        ids = {s["stroke_id"] for s in segs}
        handles.append(Line2D([], [], color=FLEET[aid].color, lw=3.4))
        labels.append(f"arm {aid} ({FLEET[aid].name}) — {res['colors'][aid]} pen — "
                      f"{sum(s['length'] for s in segs):.2f} m, {len(ids)} strokes, "
                      f"{len(segs) - len(ids)} handoff cut"
                      f"{'' if len(segs) - len(ids) == 1 else 's'}"
                      + ("   (reaches none of the logo)" if not segs else ""))
    handles.append(Line2D([], [], color="#c9c9c9", lw=2.0, ls=(0, (3, 3))))
    labels.append(f"left empty (out of reach) — {drp:.2f} m, "
                  f"{100 * drp / max(tot, 1e-9):.1f} % of {tot:.2f} m traced, "
                  f"{len(res['dropped'])} spans")
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.015),
              ncol=2, fontsize=9, framealpha=0.94, borderpad=0.8,
              labelspacing=0.5, columnspacing=2.0)
    ax.set_title(f"CSAIL logo — allocation to {n_arms} arms "
                 f"({res['drawn_len']:.2f} m drawn of {tot:.2f} m traced); "
                 "colour = arm, dashed = nobody reaches it", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(path, dpi=135)
    plt.close(fig)


# ---------------------------------------------------------------------------
def final_png(res, strokes, path, title=None):
    """The end state: the paper as it looks when every arm has stopped.

    Ink is drawn in the PEN colour (the logo's own grey/orange), not the arm
    colour, because that is what is on the paper; what nobody reached stays a
    dashed ghost so the holes are visible rather than merely absent.
    """
    fig, ax = plt.subplots(figsize=(13.0, 7.6))
    sheet_axes(ax)
    for d in res["dropped"]:
        p = np.asarray(d["pts"], float)
        ax.plot(p[:, 0], p[:, 1], ls=(0, (4, 4)), color="#cfcfcf", lw=1.8, zorder=2)
    drawn = {"grey": 0.0, "orange": 0.0}
    for aid in res["arms"]:
        for s in res["programs"][aid]:
            p = s["pts"]
            drawn[s["color"]] += s["length"]
            ax.plot(p[:, 0], p[:, 1], "-", color=PEN[s["color"]], lw=3.6, zorder=4,
                    solid_capstyle="round", solid_joinstyle="round")
    tot, drp = res["total_len"], res["dropped_len"]
    handles = [Line2D([], [], color=PEN["grey"], lw=4),
               Line2D([], [], color=PEN["orange"], lw=4),
               Line2D([], [], color="#cfcfcf", lw=1.8, ls=(0, (4, 4)))]
    labels = [f"grey ink #{PEN['grey'][1:]} — {drawn['grey']:.2f} m",
              f"orange ink #{PEN['orange'][1:]} — {drawn['orange']:.2f} m",
              f"left empty — {drp:.2f} m, {100 * drp / max(tot, 1e-9):.1f} % of "
              f"{tot:.2f} m traced, {len(res['dropped'])} spans"]
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=3, fontsize=9.5, framealpha=0.94, borderpad=0.7)
    ax.set_title(title or ("CSAIL logo — the paper when the arms stop "
                           f"({res['drawn_len']:.2f} m of {tot:.2f} m traced, "
                           f"{100 * res['drawn_len'] / max(tot, 1e-9):.1f} %)"),
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


def scene_html(res, strokes, path, h_inv=H_INV_DEFAULT):
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

    dr = [_dense(d["pts"]) for d in res["dropped"]]
    if dr:
        cloud("dropped", np.vstack(dr), [0.80, 0.80, 0.80], 0.003, 0.006)
    for aid in res["arms"]:
        segs = res["programs"][aid]
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
    for aid in res["arms"]:
        segs = res["programs"][aid]
        ids = {s["stroke_id"] for s in segs}
        c = FLEET[aid].color
        rows.append(
            f'<span style="color:rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})">'
            f"&#9632;</span> arm {aid} {FLEET[aid].name} &mdash; "
            f"<b>{res['colors'][aid]}</b> pen &mdash; "
            f"{sum(s['length'] for s in segs):.2f} m, {len(ids)} strokes, "
            f"{len(segs) - len(ids)} cuts")
    n_seg = sum(len(res["programs"][a]) for a in res["arms"])
    n_cut = n_seg - sum(len({s["stroke_id"] for s in res["programs"][a]})
                        for a in res["arms"])
    parked = [a for a in res["arms"] if not FLEET[a].active]
    html = vis.static_html().replace("</body>", LEGEND % dict(
        rows="<br>".join(rows), drop=res["dropped_len"], tot=res["total_len"],
        dropf=100 * res["dropped_len"] / max(res["total_len"], 1e-9),
        nstroke=len(strokes), nseg=n_seg, ncut=n_cut, narm=len(res["arms"]),
        npart=(1 << len(res["arms"])) - 2,
        note=("Single-arm-sequential: no multi-arm timing, transits not planned."
              if not parked else
              f"HYPOTHETICAL FLEET: arm{'s' if len(parked) > 1 else ''} "
              f"{', '.join(str(a) for a in parked)} "
              f"{'are' if len(parked) > 1 else 'is'} parked in the registry and "
              "brought up for this run only.")) + "</body>")
    open(path, "w").write(html)
    return len(html)


# ---------------------------------------------------------------------------
def program_json(res, strokes, info, path):
    doc = dict(
        sheet=list(SHEET), h_inv=H_INV_DEFAULT,
        logo={k: ([float(x) for x in v] if isinstance(v, (tuple, list))
                  else v if isinstance(v, bool) else float(v))
              for k, v in info.items()},
        colors={str(a): res["colors"][a] for a in res["arms"]},
        totals=dict(traced_m=res["total_len"], drawn_m=res["drawn_len"],
                    dropped_m=res["dropped_len"],
                    dropped_pct=100 * res["dropped_len"] / max(res["total_len"], 1e-9),
                    n_strokes=len(strokes), n_probes=res["n_probes"],
                    wall_s=res["timing"]["total"]),
        strokes=[dict(id=s["id"], color=s["color"], kind=s["kind"],
                      length=trace.plen(s["pts"])) for s in strokes],
        arms={}, dropped=[])
    for aid in res["arms"]:
        doc["arms"][str(aid)] = [dict(
            stroke_id=int(s["stroke_id"]), seg=i, color=s["color"],
            kind=s["kind"], s_range=[float(x) for x in s["s_range"]],
            direction=int(s["direction"]), length_m=float(s["length"]),
            n_points=int(len(s["pts"])), plan_ok=s["plan"]["status"] == "ok",
            validated=bool(s["plan"]["validation"]["ok"]),
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
        doc["dropped"].append(dict(
            stroke_id=int(d["stroke_id"]), color=d["color"], kind=d["kind"],
            s_range=[float(x) for x in d["s_range"]], length_m=float(d["length"]),
            at=[float(x) for x in d["at"]],
            pts=np.round(d["pts"], 5).tolist()))
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
    return ap


def _override(arms_arg):
    if arms_arg is None:
        return None
    return arms_arg if arms_arg == "all" else [int(x) for x in arms_arg.split(",")]


def run_allocation(a, verbose=False):
    """Trace -> place -> allocate, exactly as the demo scripts need it.

    One code path for the allocation PNG/JSON and for the animation, so the
    programme that gets scheduled is the programme that gets reported.
    """
    tw, off = a.target_width, a.offset
    if a.placement:
        doc = json.loads(Path(a.placement).read_text())["chosen"]
        tw = tw if tw is not None else doc["target_width"]
        off = off if off is not None else doc["offset"]
    t0 = time.time()
    px, _ = trace.trace_logo(a.image)
    strokes, info = trace.to_sheet(px, SHEET, margin=a.margin, target_width=tw,
                                   offset=tuple(off or (0.0, 0.0)))
    if not info["fits"]:
        raise SystemExit(f"placement does not fit the sheet: {info}")
    print(f"traced {len(strokes)} strokes, {trace.total_length(strokes):.2f} m, "
          f"logo {info['logo_w']:.3f} x {info['logo_h']:.3f} m at "
          f"({info['center'][0]:.3f}, {info['center'][1]:.3f})  ({time.time() - t0:.2f} s)")
    res = allocate.allocate(strokes, opts=None, verbose=verbose,
                            active_override=_override(a.arms),
                            atlas_dir=None if a.no_prefilter else str(Path(a.out)))
    return res, strokes, info


def main(argv=None):
    ap = add_args(argparse.ArgumentParser())
    ap.add_argument("--tag", default="", help="output-name suffix, e.g. _6arm")
    ap.add_argument("--final", default=None, help="also write the end-state still here")
    ap.add_argument("--no-scene", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    res, strokes, info = run_allocation(a, verbose=a.verbose)
    print()
    for line in allocate.report(res, strokes):
        print(line)

    t1 = time.time()
    program_json(res, strokes, info, out / f"csail_program{a.tag}.json")
    allocation_png(res, strokes, out / f"csail_allocation{a.tag}.png")
    if a.final:
        final_png(res, strokes, a.final)
    n = scene_html(res, strokes, out / f"csail_scene{a.tag}.html") \
        if not a.no_scene else 0
    print(f"\nwrote {out}/csail_program{a.tag}.json, {out}/csail_allocation{a.tag}.png"
          + (f", {a.final}" if a.final else "")
          + (f", {out}/csail_scene{a.tag}.html ({n / 1e6:.1f} MB)" if n else "")
          + f"  ({time.time() - t1:.1f} s)")
    print(f"PIPELINE WALL CLOCK {time.time() - t0:.1f} s")
    return res


if __name__ == "__main__":
    main()
