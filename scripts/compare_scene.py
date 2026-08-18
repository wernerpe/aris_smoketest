#!/usr/bin/env python3
"""One meshcat scene comparing reach improvements across configs.

Layers (toggle in Open Controls -> Scene):
  baseline        strict-GO today: pen 110mm, tilt<=15deg (dim arm colors)
  gain_pen200     cells GAINED by a 200mm pen (bright arm colors)
  gain_pen300     cells gained by a 300mm pen
  gain_tilt30     cells gained by allowing 30deg lean (pen 110mm)
  lost_pen300     cells LOST by the 300mm pen (under-base holes) — dark red
"""
import sys
from pathlib import Path

import numpy as np
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

sys.path.insert(0, str(Path(__file__).parents[1]))
from aris_sixarm.atlas import load, strict_go, QCOL  # noqa: E402
from aris_sixarm.fleet import FLEET, SHEET  # noqa: E402
from aris_sixarm.viz import robot_model  # noqa: E402
from aris_sixarm.viz.scene import _cloud, _hex, _drawing_pose  # noqa: E402

ROOT = Path(__file__).parents[1]
CONFIGS = {"baseline": "out_pen110", "pen200": "out_pen200",
           "pen300": "out_pen300", "tilt30": "out_tilt30"}

LEGEND = """
<div style="position:fixed;top:12px;left:12px;z-index:1000;background:rgba(255,255,255,0.92);
border:1px solid #bbb;border-radius:8px;padding:10px 14px;font:12px/1.5 sans-serif;color:#222;max-width:330px">
<b>Reach improvements &mdash; what each change buys</b><br>
<span style="color:#1f77b4">&#9632;</span>13 <span style="color:#17becf">&#9632;</span>17
<span style="color:#d62728">&#9632;</span>31 <span style="color:#2ca02c">&#9632;</span>2
<span style="color:#ff7f0e">&#9632;</span>71 <span style="color:#9467bd">&#9632;</span>97<br>
<hr style="margin:6px 0">
<b>baseline</b> (dim): strict-GO today &mdash; pen 110 mm, lean &le;15&deg;. 75.9% of sheet.<br>
<b>gain_pen200</b> (bright): +4.5% coverage, inverted arms +9 cm outer radius.<br>
<b>gain_pen300</b>: rim grows further (+15 cm) but&hellip;<br>
<b>lost_pen300</b> (dark red): &hellip;the under-base holes it re-opens.<br>
<b>gain_tilt30</b>: 30&deg; lean, stock pen &mdash; fills the under-base holes instead (+3.9%).<br>
<hr style="margin:6px 0">
Toggle layers under Open Controls &rarr; Scene. All layers use the same strict
gates (margin &ge;0.30 rad, &sigma;<sub>min</sub> &ge;0.14). Arms posed from the baseline atlas.
</div>
"""


def go_cells(out_dir, aid):
    a, _ = load(ROOT / out_dir, aid)
    go = strict_go(a)
    return {(round(x, 3), round(y, 3)) for x, y in a[go][:, :2]}, a, go


vis = meshcat.Visualizer()
vis["/Background"].set_property("top_color", [0.95, 0.95, 0.97])
vis["/Background"].set_property("bottom_color", [0.85, 0.85, 0.9])
vis["paper"].set_object(g.Box([SHEET[0], SHEET[1], 0.004]),
                        g.MeshLambertMaterial(color=0xFAFAF5))
vis["paper"].set_transform(tf.translation_matrix([SHEET[0] / 2, SHEET[1] / 2, -0.002]))
vis["table"].set_object(g.Box([SHEET[0] + 0.45, SHEET[1] + 0.25, 0.05]),
                        g.MeshLambertMaterial(color=0x8A7358))
vis["table"].set_transform(tf.translation_matrix([SHEET[0] / 2, SHEET[1] / 2, -0.029]))
links, joints = robot_model.load_model()

for i, (aid, spec) in enumerate(FLEET.items()):
    col = np.array(spec.color)
    name = f"arm{aid}"
    Twb = spec.T_world_base()
    base, a, go = go_cells("out_pen110", aid)
    p200 = go_cells("out_pen200", aid)[0]
    p300 = go_cells("out_pen300", aid)[0]
    t30 = go_cells("out_tilt30", aid)[0]

    vis[f"bases/{name}"].set_object(g.Sphere(0.045), g.MeshLambertMaterial(color=_hex(col)))
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

    def layer(path, cells, rgb, z):
        if not cells:
            return
        xy = np.array(sorted(cells))
        xyz = np.column_stack([xy, np.full(len(xy), z)])
        _cloud(vis, path, xyz, np.tile(rgb, (len(xy), 1)))

    zi = 0.004 + 0.0045 * i
    layer(f"baseline/{name}", base, 0.45 * col + 0.4, zi)
    layer(f"gain_pen200/{name}", p200 - base, col, zi + 0.002)
    layer(f"gain_pen300/{name}", p300 - base, col, zi + 0.002)
    layer(f"gain_tilt30/{name}", t30 - base, col, zi + 0.002)
    layer(f"lost_pen300/{name}", base - p300, np.array([0.55, 0.05, 0.05]), zi + 0.003)

for hidden in ("gain_pen300", "gain_tilt30", "lost_pen300"):
    vis[hidden].set_property("visible", False)

html = vis.static_html().replace("</body>", LEGEND + "</body>")
out = ROOT / "out/reach_improvements.html"
out.write_text(html)
print(f"wrote {out} ({len(html)/1e6:.1f} MB)")
