#!/usr/bin/env python3
"""Static meshcat scene of the PROPOSED layout (docs/LAYOUT_STUDY.md).

The winning layout on the merged canvas with the LATERAL pen holder visible
(bracket + offset pen), every arm in the CERTIFIED ready pose
(`layout.certified_ready_pose` — the same one the study's fine stage gates).

v2: THE SCHEMATIC MOUNT HARDWARE IS DRAWN, at the dimensions the study
treated as obstacles (`aris_sixarm/mounts.py`): each inverted arm's base
plate (0.226 x 0.190 x 0.05, sitting on the base flange) and its boom
(r = 0.10 cylinder up to the 2.34 m ceiling grid), each floor arm's pedestal
(0.30 x 0.25, from the plate top down).  Each arm's ready pose is reported
with its clearance to every OTHER arm's hardware, so the picture and the
numbers cannot drift apart.

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

from aris_sixarm import atlas, mounts, rig_final  # noqa: E402
from aris_sixarm.layout import (build_fleet, certified_ready_pose,  # noqa: E402
                                LAYOUT_PROPOSED)
from aris_sixarm.rig_final6 import SHEET_FINAL6  # noqa: E402
from aris_sixarm.viz import robot_model  # noqa: E402

W, H = SHEET_FINAL6
M = mounts.MOUNTS
CEILING_Z = M.ceiling_z


def _hex(rgb):
    return int(rgb[0] * 255) << 16 | int(rgb[1] * 255) << 8 | int(rgb[2] * 255)


def main():
    # ALWAYS the registered recommendation — the scene must show what
    # `ARIS_RIG=proposed` gives, not whatever happened to rank first in the
    # last search.  The atlas cloud is the certified sweep OF THAT layout,
    # matched by coordinates rather than by index.
    lay = LAYOUT_PROPOSED
    h = float(lay.get("h", 0.922))
    fine_dir = None
    cand = ROOT / "out" / "layout_candidates.json"
    if cand.exists():
        rec = json.loads(cand.read_text())
        want = np.round(np.asarray(sorted(map(tuple, lay["inv"]))), 4)
        for entry in list(rec.get("grid", [])) + list(rec.get("fine", [])):
            got = np.round(np.asarray(sorted(map(tuple,
                                                 entry["layout"]["inv"]))), 4)
            if got.shape == want.shape and np.allclose(got, want, atol=2e-3) \
                    and abs(entry["layout"]["h"] - h) < 1e-9:
                fine_dir = Path(entry["dir"])
                break
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

    # strict-GO cloud from the certified atlas of THIS layout, if present
    if fine_dir is not None and fine_dir.exists():
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
        # THE SCHEMATIC MOUNT HARDWARE, at the dimensions the study treated
        # as obstacles — drawn so the picture and the numbers cannot drift
        if spec.mount == "inv":
            vis[f"plates/{name}"].set_object(
                g.Box([*M.plate_xy, M.plate_t]),
                g.MeshLambertMaterial(color=0x555555))
            vis[f"plates/{name}"].set_transform(tf.translation_matrix(
                [spec.xy[0], spec.xy[1], h + M.plate_t / 2]))
            length = CEILING_Z - (h + M.plate_t)
            vis[f"booms/{name}"].set_object(
                g.Cylinder(length, M.boom_r),
                g.MeshLambertMaterial(color=col, opacity=0.55))
            vis[f"booms/{name}"].set_transform(
                tf.translation_matrix([spec.xy[0], spec.xy[1],
                                       h + M.plate_t + length / 2])
                @ tf.rotation_matrix(np.pi / 2, [1, 0, 0]))
        else:
            vis[f"stands/{name}"].set_object(
                g.Box([*M.ped_xy, M.ped_drop]),
                g.MeshLambertMaterial(color=0x555555))
            vis[f"stands/{name}"].set_transform(tf.translation_matrix(
                [spec.xy[0], spec.xy[1], M.z_floor - M.ped_drop / 2]))
        q, xy, rep = certified_ready_pose(spec, h)
        robot_model.add_robot(vis, f"robots/{name}", links, joints, q, Twb,
                              pen_color=col)
        T, pts = frames.fk(q)
        P = np.vstack([pts] + list(frames.tool_points_many(
            T[None], pen_lat=frames.PEN_LAT_HOLDER)))
        Pw = (Twb[:3, :3] @ P.T).T + Twb[:3, 3]
        cl = float(rig_final.chain_static_clearance(
            Pw, spec.static_obstacles())[0])
        gate = "OK" if cl >= rig_final.STATIC_MARGIN else "VIOLATION"
        print(f"arm {aid:>2} ({spec.mount:5s}) ready over ({xy[0]:.2f}, "
              f"{xy[1]:.2f}): margin {rep['worst']['joint_margin']:.3f}, "
              f"tip z {rep['worst']['tip_z'] * 1000:+.0f} mm, "
              f"mount clearance {cl:+.3f} m {gate}")

    html = vis.static_html()
    out = ROOT / "out" / "proposed_scene.html"
    out.write_text(html)
    nf = sum(s.mount == "floor" for s in fl.values())
    print(f"wrote {out} ({len(html) / 1e6:.1f} MB), {nf}+{6 - nf} layout "
          f"h={h}, mounts drawn (boom r={M.boom_r}, ceiling {CEILING_Z} m)")


if __name__ == "__main__":
    main()
