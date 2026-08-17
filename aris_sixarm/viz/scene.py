"""Static meshcat scene of the atlases: paper, posed arms, reach layers, legend.

Layers (toggle under Open Controls -> Scene):
  reach_by_arm/arm<ID>  arm color; bright = strict-GO, faint = reachable only
  sigma_min/arm<ID>     viridis by force controllability (hidden by default)
  f_max/arm<ID>         viridis by max downward force (hidden by default)
"""
import numpy as np
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
from matplotlib import cm

from ..atlas import load, strict_go
from ..fleet import FLEET, SHEET, H_INV_DEFAULT
from ..frames import tip_pos
from ..metrics import GATE_MARGIN, GATE_SIGMA
from . import robot_model

LEGEND = """
<div style="position:fixed;top:12px;left:12px;z-index:1000;background:rgba(255,255,255,0.92);
border:1px solid #bbb;border-radius:8px;padding:10px 14px;font:12px/1.5 sans-serif;color:#222;max-width:330px">
<b>Aris Kindt &mdash; 6-arm reachability atlas</b><br>
<span style="color:#1f77b4">&#9632;</span> 13 front (floor) &nbsp;
<span style="color:#17becf">&#9632;</span> 17 back (floor) &nbsp;
<span style="color:#d62728">&#9632;</span> 31 L-inv-front<br>
<span style="color:#2ca02c">&#9632;</span> 2 R-wall-front &nbsp;
<span style="color:#ff7f0e">&#9632;</span> 71 L-inv-back &nbsp;
<span style="color:#9467bd">&#9632;</span> 97 R-inv-back<br>
<hr style="margin:6px 0">
<b>reach_by_arm</b>: bright = strict-GO (joint margin &ge; 0.30 rad AND
&sigma;<sub>min</sub> &ge; 0.14), faint = reachable only.<br>
<b>sigma_min</b> layer &mdash; force controllability, smallest singular value of the
3&times;7 pen-tip Jacobian (low = force-blind fringe):<br>
<div style="background:linear-gradient(to right,#440154,#31688e,#35b779,#fde725);
height:10px;border-radius:3px;margin:2px 0"></div>
<div style="display:flex;justify-content:space-between"><span>0.05</span>
<span>0.14 = gate</span><span>&ge;0.25</span></div>
<b>f_max</b> layer &mdash; max sustainable downward force from &tau;-limits via
J&#7488;n (same colormap, 0&ndash;100 N).<br>
<hr style="margin:6px 0">
Arms shown in a real drawing pose (atlas IK solution, pen tip on paper;
pen = TCP+0.110 m). Toggle layers: Open Controls &rarr; Scene.
</div>
"""


def _hex(rgb):
    return int(rgb[0] * 255) << 16 | int(rgb[1] * 255) << 8 | int(rgb[2] * 255)


def _cloud(vis, path, xyz, rgb, size=0.016):
    vis[path].set_object(g.PointCloud(position=xyz.T.astype(np.float32),
                                      color=rgb.T.astype(np.float32), size=size))


def _drawing_pose(a, go, bx, by):
    cand = a[go]
    r = np.hypot(cand[:, 0] - bx, cand[:, 1] - by)
    row = cand[np.argmax(cand[:, 2] - 0.6 * np.abs(r - 0.5))]
    return row[:2], row[7:14]


def build(out_dir, html_path, h_inv=H_INV_DEFAULT):
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

    grid_go = {}
    for i, (aid, spec) in enumerate(FLEET.items()):
        a, _ = load(out_dir, aid)
        col = np.array(spec.color)
        go = strict_go(a)
        z = 0.004 + 0.0045 * i
        name = f"arm{aid}"
        Twb = spec.T_world_base(h_inv)

        vis[f"bases/{name}"].set_object(g.Sphere(0.045),
                                        g.MeshLambertMaterial(color=_hex(col)))
        vis[f"bases/{name}"].set_transform(Twb)
        if spec.mount == "inv":
            vis[f"booms/{name}"].set_object(
                g.Cylinder(1.3, 0.012), g.MeshLambertMaterial(color=_hex(col), opacity=0.55))
            vis[f"booms/{name}"].set_transform(
                tf.translation_matrix([spec.xy[0], spec.xy[1], Twb[2, 3] - 0.65])
                @ tf.rotation_matrix(np.pi / 2, [1, 0, 0]))

        cell, q_draw = _drawing_pose(a, go, *spec.xy)
        robot_model.add_robot(vis, f"robots/{name}", links, joints, q_draw, Twb,
                              pen_color=_hex(col))
        tip_w = Twb[:3, :3] @ tip_pos(q_draw) + Twb[:3, 3]
        assert np.linalg.norm(tip_w - [cell[0], cell[1], 0.0]) < 1e-3

        xyz = np.column_stack([a[:, 0], a[:, 1], np.full(len(a), z)])
        rgb = np.where(go[:, None], col[None, :], (0.35 * col + 0.55)[None, :] * 0.45)
        _cloud(vis, f"reach_by_arm/{name}", xyz, rgb)
        sm = np.clip((a[:, 3] - 0.05) / 0.20, 0, 1)
        _cloud(vis, f"sigma_min/{name}", xyz, cm.viridis(sm)[:, :3])
        _cloud(vis, f"f_max/{name}", xyz, cm.viridis(np.clip(a[:, 4] / 100, 0, 1))[:, :3])
        for x, y in a[go][:, :2]:
            grid_go.setdefault((round(x, 3), round(y, 3)), []).append(aid)

    vis["sigma_min"].set_property("visible", False)
    vis["f_max"].set_property("visible", False)

    n_sheet = int(SHEET[0] / 0.02 + 1) * int(SHEET[1] / 0.02 + 1)
    counts = np.array([len(v) for v in grid_go.values()])
    stats = dict(coverage=100 * len(grid_go) / n_sheet,
                 overlap2=100 * (counts >= 2).sum() / n_sheet,
                 overlap3=100 * (counts >= 3).sum() / n_sheet)
    html = vis.static_html().replace("</body>", LEGEND + "</body>")
    open(html_path, "w").write(html)
    print("coverage %.1f%%  >=2-arm %.1f%%  >=3-arm %.1f%%  -> %s (%.1f MB)" % (
        stats["coverage"], stats["overlap2"], stats["overlap3"], html_path, len(html) / 1e6))
    return stats
