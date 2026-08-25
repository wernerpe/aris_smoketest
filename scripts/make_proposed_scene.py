#!/usr/bin/env python3
"""Static meshcat scene of the PROPOSED layout (docs/LAYOUT_STUDY.md).

2 floor + 4 ceiling-inverted arms on the merged canvas, LATERAL pen holder
visible (bracket + offset pen), every arm in a READY pose derived for this
tool: hover 0.10 m over a comfortable point of the arm's own annulus, IK'd
with the lateral offset and gated by `validate.check_pose` — margin >= 0.30,
chain above the paper, PEN TIP above the paper (the check that caught the
legacy inverted ready pose dipping 16 mm under).

GREEN FIELD: the ceiling is drawn as a nominal slab at z = 2.0 m and each
inverted arm hangs from a schematic boom; no real structure exists for these
positions yet.

    python3 scripts/make_proposed_scene.py         -> out/proposed_scene.html
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from aris_sixarm import frames  # noqa: E402
frames.activate_tool("lateral")                    # the tool of the study

import meshcat  # noqa: E402
import meshcat.geometry as g  # noqa: E402
import meshcat.transformations as tf  # noqa: E402

from aris_sixarm import atlas, ik, metrics  # noqa: E402
from aris_sixarm.frames import joint_margin, rotx, rotz, tool_offset  # noqa: E402
from aris_sixarm.layout import build_fleet, LAYOUT_PROPOSED  # noqa: E402
from aris_sixarm.rig_final6 import SHEET_FINAL6  # noqa: E402
from aris_sixarm.validate import check_pose  # noqa: E402
from aris_sixarm.viz import robot_model  # noqa: E402

W, H = SHEET_FINAL6
CEILING_Z = 2.0


def _hex(rgb):
    return int(rgb[0] * 255) << 16 | int(rgb[1] * 255) << 8 | int(rgb[2] * 255)


def ready_pose(spec, h_inv):
    """Hover 0.10 m over the arm's comfortable patch, lateral tool, gated.

    Scans 8 tool yaws x the q7 grid x all branches — with the lateral holder
    phi is a real DOF and a hover pinned to phi = 0 can be unreachable at a
    spot the arm covers at another phi (`writing.lifted_config` keeps the
    one-convention phi = 0 for the INLINE fleet; a phi-aware hover solver
    for the full pipeline is follow-up work).  Best min(margin, 2.5 sigma)
    among poses that pass `validate.check_pose` with the pen above paper —
    the frames.py ready-pose recipe, at the new tool.
    """
    b = np.asarray(spec.xy, float)
    c = np.array([W / 2, H / 2])
    u = c - b
    u = u / max(np.linalg.norm(u), 1e-9)
    Twb = spec.T_world_base(h_inv)
    Twb_inv = np.linalg.inv(Twb)
    off = tool_offset()                      # ACTIVE tool (lateral)
    best = None
    for r in (0.55, 0.62, 0.48, 0.70, 0.40):
        xy = np.clip(b + r * u, [0.05, 0.05], [W - 0.05, H - 0.05])
        for phi in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            R = rotz(phi) @ rotx(np.pi)
            T_w = np.eye(4)
            T_w[:3, :3] = R
            T_w[:3, 3] = np.array([xy[0], xy[1], 0.10]) - R @ off
            T_b = Twb_inv @ T_w
            for q7 in ik.Q7_GRID:
                for q in ik.solve(T_b, q7, spec.q_seed):
                    m = joint_margin(q)
                    if m < 0.30:
                        continue
                    rep = check_pose(q, spec, h_inv=h_inv)
                    if not rep["ok"] or rep["worst"]["tip_z"] <= 0.0:
                        continue
                    s = metrics.sigma_min(metrics.tip_jacobian(q))
                    key = min(m, 2.5 * s)
                    if best is None or key > best[0]:
                        best = (key, q, xy, rep)
        if best is not None:
            return best[1], best[2], best[3]
    raise RuntimeError(f"no certified ready pose for arm {spec.arm_id}")


def main():
    lay = LAYOUT_PROPOSED
    cand = ROOT / "out" / "layout_candidates.json"
    if cand.exists():
        rec = json.loads(cand.read_text())
        if rec.get("fine"):
            lay = rec["fine"][0]["layout"]
    h = float(lay.get("h", 0.922))
    fl = build_fleet(lay)
    links, joints = robot_model.load_model()

    vis = meshcat.Visualizer()
    vis["/Background"].set_property("top_color", [0.95, 0.95, 0.97])
    vis["/Background"].set_property("bottom_color", [0.85, 0.85, 0.9])
    vis["paper"].set_object(g.Box([W, H, 0.004]),
                            g.MeshLambertMaterial(color=0xFAFAF5))
    vis["paper"].set_transform(tf.translation_matrix([W / 2, H / 2, -0.002]))
    vis["table"].set_object(g.Box([W + 0.7, H + 0.7, 0.05]),
                            g.MeshLambertMaterial(color=0x8A7358))
    vis["table"].set_transform(tf.translation_matrix([W / 2, H / 2, -0.029]))
    vis["ceiling"].set_object(g.Box([W + 0.7, H + 0.7, 0.03]),
                              g.MeshLambertMaterial(color=0xB9B4AC,
                                                    opacity=0.35))
    vis["ceiling"].set_transform(
        tf.translation_matrix([W / 2, H / 2, CEILING_Z + 0.015]))

    # strict-GO cloud from the fine atlas of the winner, if present
    fine_dir = ROOT / "out" / "layout_study" / "fine_0"
    if fine_dir.exists():
        for f in sorted(fine_dir.glob("atlas_arm*.npz")):
            d = np.load(f)
            arr, aid = d["data"], int(d["arm_id"])
            if not len(arr) or aid not in fl:
                continue
            go = atlas.strict_go(arr)
            col = np.array(fl[aid].color)
            xyz = np.column_stack([arr[go, 0], arr[go, 1],
                                   np.full(go.sum(), 0.004)])
            rgb = np.tile(col, (go.sum(), 1))
            vis[f"strict_go/arm{aid}"].set_object(
                g.PointCloud(position=xyz.T.astype(np.float32),
                             color=rgb.T.astype(np.float32), size=0.014))

    for aid, spec in fl.items():
        col = _hex(spec.color)
        Twb = spec.T_world_base(h)
        name = f"arm{aid}"
        vis[f"bases/{name}"].set_object(g.Sphere(0.045),
                                        g.MeshLambertMaterial(color=col))
        vis[f"bases/{name}"].set_transform(Twb)
        if spec.mount == "inv":
            length = CEILING_Z - h
            vis[f"booms/{name}"].set_object(
                g.Cylinder(length, 0.05),
                g.MeshLambertMaterial(color=col, opacity=0.55))
            vis[f"booms/{name}"].set_transform(
                tf.translation_matrix([spec.xy[0], spec.xy[1],
                                       h + length / 2])
                @ tf.rotation_matrix(np.pi / 2, [1, 0, 0]))
        else:
            vis[f"stands/{name}"].set_object(
                g.Box([0.23, 0.19, 0.013]),
                g.MeshLambertMaterial(color=0x555555))
            vis[f"stands/{name}"].set_transform(
                tf.translation_matrix([spec.xy[0], spec.xy[1], 0.006]))
        q, xy, rep = ready_pose(spec, h)
        robot_model.add_robot(vis, f"robots/{name}", links, joints, q, Twb,
                              pen_color=col)
        tip = Twb[:3, :3] @ frames.tip_pos(q) + Twb[:3, 3]
        print(f"arm {aid:>2} ({spec.mount:5s}) ready over ({xy[0]:.2f}, "
              f"{xy[1]:.2f}): margin {rep['worst']['joint_margin']:.3f}, "
              f"tip z {tip[2] * 1000:+.0f} mm, chain z "
              f"{rep['worst']['min_chain_z'] * 1000:.0f} mm")

    html = vis.static_html()
    out = ROOT / "out" / "proposed_scene.html"
    out.write_text(html)
    print(f"wrote {out} ({len(html) / 1e6:.1f} MB), layout h={h}")


if __name__ == "__main__":
    main()
