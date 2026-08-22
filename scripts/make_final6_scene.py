#!/usr/bin/env python3
"""Static meshcat scene of the MIRRORED SIX-ARM installation.

Two FINAL-RIG units abutting back-to-back at the hanging-arm-side outer face
(aris_sixarm/rig_final6.py): both frames as grey collision boxes, both paper
webs, all six arms at their ready poses with the real pen holders, per-arm
identity colours, and a strict-GO coverage layer obtained by MIRRORING unit
A's atlases (no new sweep — see scripts/make_atlas6_preview.py).

Same construction as out/final_rig_view.html, doubled.

    python3 scripts/make_final6_scene.py            # -> out/final_rig6_view.html
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import trimesh

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from aris_sixarm import atlas, dead as dead_mod, rig_final  # noqa: E402
from aris_sixarm import rig_final6 as r6  # noqa: E402
from aris_sixarm.fleet import ACTIVE_RIG  # noqa: E402
from aris_sixarm.viz import robot_model  # noqa: E402

BOX_COLOR = {"A": 0x9AA0A6, "B": 0x8B9096}      # two greys, one per unit
BOX_OPACITY = 0.42

# THE DEAD-ZONE TIERS ARE TWO COLOURS BECAUSE THEY ARE TWO CLAIMS.
# Dark red is paper no arm can certify at the gates the PLANNER itself uses:
# geometry, and nothing about a different allocation or a slower pen recovers
# it.  Orange is paper the strict atlas (margin 0.30, sigma 0.14) disowns but
# the planner (0.15, 0.10) can reach — real paper that the headline "24.07 %
# dead" number counts as lost.  Keeping them apart is the whole point of the
# layer; a single red blob would repeat the overstatement it exists to correct.
DEAD_COLOR = 0x8B0000          # solid dark red — permissive-dead
STRICT_ONLY_COLOR = 0xFF8C00   # orange — strict-only-dead
UNDRAWN_COLOR = 0xE0217D       # the undrawn spans of the shipped programme
Z_DEAD = 0.0020                # just over the paper, under the coverage clouds
Z_UNDRAWN = 0.0320             # over everything: it is the highlight


def _hex(rgb):
    return int(rgb[0] * 255) << 16 | int(rgb[1] * 255) << 8 | int(rgb[2] * 255)


def _add_boxes(vis, boxes):
    for b in boxes:
        lo, hi = np.asarray(b["lo"]), np.asarray(b["hi"])
        size = np.maximum(hi - lo, 1e-4)
        unit = b.get("unit", "A")
        vis[f"frame_{unit}/{b['name']}"].set_object(
            g.Box(list(size)),
            g.MeshLambertMaterial(color=BOX_COLOR[unit], opacity=BOX_OPACITY,
                                  transparent=True))
        vis[f"frame_{unit}/{b['name']}"].set_transform(
            tf.translation_matrix((lo + hi) / 2))


_HOLDER = []


def _holder_mesh(face_target=5000):
    """The CAD pen holder, decimated once and reused for all six hands."""
    if not _HOLDER:
        p = ROOT / "assets/final_rig" / rig_final.TOOL["visual_mesh"]
        m = trimesh.load(p, force="mesh")
        m.merge_vertices(merge_tex=True, merge_norm=True)
        if len(m.faces) > face_target:
            m = m.simplify_quadric_decimation(face_count=face_target)
        _HOLDER.append(m)
    return _HOLDER[0]


def _add_holder(vis, path, T_world_hand, color):
    """The CAD pen holder mesh, already expressed in the panda_hand frame."""
    m = _holder_mesh()
    vis[path].set_object(
        g.TriangularMeshGeometry(np.asarray(m.vertices, np.float32),
                                 np.asarray(m.faces, np.uint32)),
        g.MeshLambertMaterial(color=color))
    vis[path].set_transform(T_world_hand)


def _cells(vis, path, mask, xs, ys, grid, color, z):
    """One dead tier as a dense point cloud, one sprite per 2 cm atlas cell."""
    iy, ix = np.nonzero(mask)
    if not len(iy):
        return 0
    xyz = np.column_stack([xs[ix], ys[iy], np.full(len(iy), z)])
    rgb = np.tile(np.array([[(color >> 16 & 255) / 255.0,
                             (color >> 8 & 255) / 255.0,
                             (color & 255) / 255.0]]), (len(iy), 1))
    vis[path].set_object(
        g.PointCloud(position=xyz.T.astype(np.float32),
                     color=rgb.T.astype(np.float32), size=float(grid)))
    return len(iy)


def _spans(vis, path, spans, color, z):
    """The undrawn spans as line segments lying on the paper."""
    segs = []
    for sp in spans:
        P = np.asarray(sp["pts"], float).reshape(-1, 2)
        for a, b in zip(P[:-1], P[1:]):
            segs.append([a[0], a[1], z])
            segs.append([b[0], b[1], z])
    if not segs:
        return 0
    V = np.asarray(segs, np.float32).T
    vis[path].set_object(g.LineSegments(
        g.PointsGeometry(V), g.LineBasicMaterial(color=color, linewidth=6)))
    return len(segs) // 2


def dead_legend(T, att, probe):
    c = T["counts"]
    by = att["by_cause_m"]
    tot = max(att["total_m"], 1e-9)
    rows = "".join(
        f"<tr><td>{k}</td><td align=right>{v:.3f} m</td>"
        f"<td align=right>{100 * v / tot:.1f} %</td></tr>"
        for k, v in sorted(by.items(), key=lambda kv: -kv[1]))
    sp = ("" if "span_probe_certified_m" not in att else
          "<hr style='margin:6px 0'><b>And 'other' is not a shrug</b>: every "
          "leftover span was handed back to the real planner and "
          f"<b>{att['span_probe_certified_m']:.3f} m</b> of it certified. A "
          "cell being GO does not make a STROKE through it certifiable &mdash; "
          "the planner needs a continuous band from end to end, and these "
          "spans have an infeasible endpoint or no surviving q7 fiber.")
    pr = ("" if probe is None else
          f"Probe check: {probe['n']} permissive-dead cells sampled "
          f"(every {probe['stride']}th, {100 * probe['length']:.0f} mm stroke), "
          f"<b>{probe['n_certified']}</b> certified by the real planner.<br>")
    return f"""
<div style="position:fixed;top:12px;right:12px;z-index:1000;
background:rgba(255,255,255,0.95);border:1px solid #bbb;border-radius:8px;
padding:10px 14px;font:12px/1.5 sans-serif;color:#222;max-width:420px">
<b>DEAD ZONES &mdash; two tiers, because they mean two things</b><br>
<span style="color:#8B0000">&#9632;</span> <b>permissive-dead</b>
{c['dead_permissive']} cells ({c['dead_permissive_pct']:.2f} %) &mdash; no arm
certifies even at the PLANNER's own hard gates (margin
{dead_mod.PERMISSIVE_MARGIN:.2f}, &sigma;<sub>min</sub>
{dead_mod.PERMISSIVE_SIGMA:.2f}). Geometry; nothing recovers it.<br>
<span style="color:#FF8C00">&#9632;</span> <b>strict-only-dead</b>
{c['strict_only_dead']} cells ({c['strict_only_dead_pct']:.2f} %) &mdash;
reachable at the planner's gates, refused by the atlas's
({dead_mod.STRICT_MARGIN:.2f} / {dead_mod.STRICT_SIGMA:.2f}). Real paper the
headline dead number throws away.<br>
{pr}
<hr style="margin:6px 0">
<b>Atlas-strict dead</b> {c['dead_strict_pct']:.2f} % &rarr;
<b>truly dead</b> {c['dead_permissive_pct']:.2f} %. The strict sweep overstates
death by {c['strict_only_dead_pct']:.2f} pp of the canvas.<br>
<hr style="margin:6px 0">
<span style="color:#E0217D">&#9632;</span> <b>undrawn_ink</b> &mdash; the
{att['n_spans']} spans of the shipped CSAIL programme that were traced and
never drawn, {att['total_m']:.3f} m of {100 * (1 - 0.869704):.2f} %:<br>
<table style="font:11px/1.4 sans-serif">{rows}</table>
<i>conductor refusal: {att['conductor_refusal_m']:.3f} m &mdash;
{att['conductor_refusal_note']}</i>
{sp}
</div>
"""


def legend():
    mp = r6.MIRROR_PLANE_CANVAS_Y
    rows = "".join(
        f'<span style="color:{c}">&#9632;</span> {t}<br>' for c, t in [
            ("#1f77b4", 'arm <b>13</b> floor &mdash; unit A, outer front end'),
            ("#d62728", 'arm <b>31</b> inverted &mdash; unit A, middle band'),
            ("#2ca02c", 'arm <b>2</b> side &mdash; unit A, middle band'),
            ("#17becf", 'arm <b>17</b> floor &mdash; unit B, outer back end *'),
            ("#ff7f0e", 'arm <b>71</b> inverted &mdash; unit B, middle band *'),
            ("#9467bd", 'arm <b>97</b> side &mdash; unit B, middle band *'),
        ])
    return f"""
<div style="position:fixed;top:12px;left:12px;z-index:1000;
background:rgba(255,255,255,0.94);border:1px solid #bbb;border-radius:8px;
padding:10px 14px;font:12px/1.5 sans-serif;color:#222;max-width:390px">
<b>THE REAL INSTALLATION &mdash; two FINAL-RIG units, mirrored</b><br>
{rows}
<hr style="margin:6px 0">
<b>Mirror plane</b>: canvas y = <b>{mp:.5f} m</b> (W frame Y = 208.28 cm) &mdash;
the outer face on the hanging-arm side. Provenance: PDF dimension "208,3"
(<i>entire table</i>), DXF-measured 208.280; the corner posts
(post_BL/post_BR) and top_slab all stop exactly there.
The hanging mounts sit at Y&nbsp;152.45 (back half of the 208.28 depth), the
floor mount at Y&nbsp;14.07 (front half) &mdash; so the frames abut BACK to
back and the four hanging arms cluster in the middle.<br>
<hr style="margin:6px 0">
<b>ASSUMPTIONS &mdash; need your confirmation</b><br>
1. The two frames <b>abut exactly, zero gap</b>. The levelling-foot pads
overhang the leg lines by 0.33 cm, so pads touching would hold the structural
faces 0.66 cm apart. Knob: <code>rig_final6.GAP_CM</code>.<br>
2. * <b>Unit B arm ids 17 / 71 / 97</b> are ASSUMED (the legacy registry's
other three physical arms). No drawing names them.<br>
3. <b>Web seam</b>: <code>MERGE_WEBS = {r6.MERGE_WEBS}</code> &rarr;
{"ONE continuous surface spanning both units and the "
 f"<b>{r6.SEAM_M * 100:.2f} cm</b> strip between the webs (swept for real by "
 "<code>scripts/run_atlas6.py</code>). What is still assumed is physical: that "
 "the paper is flat and drawable over that strip, which lies on the two "
 "frames' abutting top rails and not on a tabletop."
 if r6.MERGE_WEBS else
 f"two webs with a <b>{r6.SEAM_M * 100:.2f} cm</b> gap between them, not one "
 "continuous surface. Flip the flag if it should be one web."}<br>
4. Unit-B base rotations are realised as <b>proper</b> rotations
(S&middot;R&middot;S): a reflection is improper and no arm can be built
left-handed. Confirm each real base plate's yaw.<br>
<hr style="margin:6px 0">
Grey boxes = both frames' collision model (conservative). Webs
1.803&times;1.700 m each, combined extent 1.803&times;{r6.SHEET_FINAL6[1]:.3f} m.
<b>coverage</b> layer: bright = strict-GO, faint = reachable only; unit B's is
unit A's atlas MIRRORED (exact &mdash; the mirrored joint vector
q&middot;(-1,1,-1,1,-1,1,-1) reproduces the chain to 3e-16 and preserves joint
margin, &sigma;<sub>min</sub> and f<sub>max</sub>; adding the second frame
removes 0 cells). Toggle layers: Open Controls &rarr; Scene.
</div>
"""


def build(html_path, atlas_dir, dead=None, dropped=None, attribution=None,
          probe=None, coverage=True):
    links, joints = robot_model.load_model()
    vis = meshcat.Visualizer()
    vis["/Background"].set_property("top_color", [0.95, 0.95, 0.97])
    vis["/Background"].set_property("bottom_color", [0.85, 0.85, 0.90])

    # --- the drawable surface: two webs, or one merged canvas ------------
    # `r6.webs()` returns ONE rectangle under MERGE_WEBS and TWO without it, so
    # it is enumerated rather than zipped against a fixed pair of names — a zip
    # against ("webA", "webB") silently drops half a two-web rig or draws only
    # the first half of a merged one.
    webs = r6.webs()
    for i, ((x0, y0), (w, h)) in enumerate(webs):
        name = f"web{chr(ord('A') + i)}" if len(webs) > 1 else "canvas"
        vis[f"paper/{name}"].set_object(
            g.Box([w, h, 0.004]), g.MeshLambertMaterial(color=0xFAFAF5))
        vis[f"paper/{name}"].set_transform(
            tf.translation_matrix([x0 + w / 2, y0 + h / 2, -0.002]))

    # --- the mirror plane, drawn so it can be SEEN -----------------------
    ym = r6.MIRROR_PLANE_CANVAS_Y
    # sized to the frame footprint (canvas x -0.212..1.972, z 0..1.72) so it
    # reads as the seam between the two units, not as scenery
    vis["mirror_plane"].set_object(
        g.Box([2.20, 0.002, 1.74]),
        g.MeshLambertMaterial(color=0xE0217D, opacity=0.10, transparent=True))
    vis["mirror_plane"].set_transform(
        tf.translation_matrix([0.880, ym, 0.85]))

    # --- opening view: the whole 3.63 m installation, long axis across ---
    vis["/Cameras/default"].set_transform(
        tf.translation_matrix([0.9017, ym, 0.45]))
    # NB the camera offset is expressed in meshcat's ROTATED (Y-up) frame:
    # rotated (a, b, c) == world (a, -c, b).  This is world (2.5, -2.7, 2.2),
    # i.e. front-right and above, looking down the 3.63 m long axis.
    vis["/Cameras/default/rotated/<object>"].set_property(
        "position", [2.5, 2.2, 2.7])

    # --- both frames, in the BUILD the active rig stands in ---------------
    # `ARIS_RIG=final6_opt` lowers the side arms 20 cm and lengthens both poles;
    # drawing the as-drawn boxes under that rig would show two arms hanging in
    # mid-air below the end of their own pole, which is the one picture this
    # scene exists to prevent.
    opt = ACTIVE_RIG == "final6_opt"
    fleet6 = r6.FLEET_FINAL6_OPT if opt else r6.FLEET_FINAL6
    _add_boxes(vis, r6.frame_boxes6_canvas(
        zmin=-10, boxes_w=r6.FRAME_BOXES6_OPT_W_CM if opt else None))

    # --- six arms at ready, with pen holders -----------------------------
    for aid, spec in fleet6.items():
        Twb = spec.T_world_base()
        col = _hex(spec.color)
        q = np.asarray(spec.q_seed, float)
        robot_model.add_robot(vis, f"robots/arm{aid}", links, joints, q, Twb,
                              pen_color=col)
        poses = robot_model.link_poses(joints, q)
        _add_holder(vis, f"holders/arm{aid}", Twb @ poses["panda_hand"], col)
        vis[f"bases/arm{aid}"].set_object(
            g.Sphere(0.035), g.MeshLambertMaterial(color=col))
        vis[f"bases/arm{aid}"].set_transform(Twb)

    # --- dead zones, under the coverage clouds ---------------------------
    n_dead = n_so = n_sp = 0
    if dead is not None:
        n_dead = _cells(vis, "dead_zones/permissive_dead",
                        dead["dead_permissive"], dead["xs"], dead["ys"],
                        dead["grid"], DEAD_COLOR, Z_DEAD)
        n_so = _cells(vis, "dead_zones/strict_only_dead",
                      dead["strict_only_dead"], dead["xs"], dead["ys"],
                      dead["grid"], STRICT_ONLY_COLOR, Z_DEAD + 0.0008)
    if dropped:
        n_sp = _spans(vis, "undrawn_ink/spans", dropped, UNDRAWN_COLOR,
                      Z_UNDRAWN)

    # --- coverage layer, mirrored (no new sweep) -------------------------
    for i, (aid, spec) in (enumerate(fleet6.items()) if coverage else []):
        # PREFER THE ARM'S OWN SWEEP.  `out/atlas_final6_opt` has all six, over
        # the whole continuous canvas; `out/atlas_final` has only unit A's
        # three, and unit B has to be mirrored from its twin (exact, but blind
        # to the seam and to the other unit's half).
        try:
            arr, _ = atlas.load(atlas_dir, aid)
            mirrored = False
        except FileNotFoundError:
            arr, _ = atlas.load(atlas_dir, r6.TWIN[aid])
            mirrored = True
        go = atlas.strict_go(arr)
        x, y = arr[:, 0].copy(), arr[:, 1].copy()
        if mirrored:
            y = r6.mirror_y(y)
        col = np.array(spec.color)
        xyz = np.column_stack([x, y, np.full(len(arr), 0.004 + 0.0045 * i)])
        rgb = np.where(go[:, None], col[None, :],
                       (0.35 * col + 0.55)[None, :] * 0.45)
        vis[f"coverage/arm{aid}"].set_object(
            g.PointCloud(position=xyz.T.astype(np.float32),
                         color=rgb.T.astype(np.float32), size=0.016))

    extra = "" if dead is None and not dropped else dead_legend(
        dead, attribution, probe)
    html = vis.static_html().replace("</body>", legend() + extra + "</body>")
    Path(html_path).parent.mkdir(parents=True, exist_ok=True)
    open(html_path, "w").write(html)
    print(f"{html_path}  ({len(html) / 1e6:.1f} MB)  "
          f"6 arms, {len(r6.frame_boxes6_canvas(zmin=-10))} frame boxes, "
          f"mirror plane canvas y = {r6.MIRROR_PLANE_CANVAS_Y:.6f} m"
          + (f", dead {n_dead} + strict-only {n_so} cells, "
             f"{n_sp} undrawn segments" if dead is not None or n_sp else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", default=str(ROOT / "out/final_rig6_view.html"))
    ap.add_argument("--atlas", default=str(ROOT / "out/atlas_final"))
    a = ap.parse_args()
    build(a.html, a.atlas)
