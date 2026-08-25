"""Franka arm visuals for meshcat from drake's panda_arm_hand.urdf.

Meshes are glTF (Y-up): v_link = T_visual_origin @ Rx(+90deg) @ v_gltf.
Kinematics: Panda visual model, identical link geometry to FR3.
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
import meshcat.geometry as g

import os

# clean assets copied from franka_manipulation_station (vendored in-repo)
_D = Path(os.environ.get(
    "ARIS_FRANKA_MESHES",
    str(Path(__file__).parents[2] / "assets/franka_description")))
_RX90 = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1.0]])
_FACE_TARGET = 5000


def _rpy_T(xyz, rpy):
    r, p, y = rpy
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3] = Rz @ Ry @ Rx
    T[:3, 3] = xyz
    return T


def _parse(el, tag_default="0 0 0"):
    o = el.find("origin")
    xyz = [float(v) for v in (o.get("xyz", tag_default) if o is not None else tag_default).split()]
    rpy = [float(v) for v in (o.get("rpy", tag_default) if o is not None else tag_default).split()]
    return _rpy_T(xyz, rpy)


def load_model():
    """-> (links: {name: (vertices Nx3 f32, faces Mx3 u32)}, joints: ordered list)."""
    root = ET.parse(_D / "urdf/panda_arm_hand.urdf").getroot()
    links, joints = {}, []
    cache = {}
    for link in root.iter("link"):
        vis = link.find("visual")
        if vis is None:
            continue
        mesh_el = vis.find("geometry/mesh")
        if mesh_el is None:
            continue
        fname = mesh_el.get("filename").split("/")[-1]
        if fname not in cache:
            m = trimesh.load(_D / "meshes/visual" / fname, force="mesh")
            # weld duplicated seam vertices BEFORE decimating, else the
            # simplifier tears the surface apart at every normal/uv seam
            m.merge_vertices(merge_tex=True, merge_norm=True)
            if len(m.faces) > _FACE_TARGET:
                m = m.simplify_quadric_decimation(face_count=_FACE_TARGET)
            cache[fname] = m
        m = cache[fname]
        Tv = _parse(vis) @ _RX90
        v = (Tv[:3, :3] @ m.vertices.T).T + Tv[:3, 3]
        f = np.asarray(m.faces, np.uint32)
        if fname == "link0.gltf":
            # the visual includes dangling cables below the base plate — clip
            keep = ~np.any(v[f][:, :, 2] < -0.005, axis=1)
            f = f[keep]
        links[link.get("name")] = (v.astype(np.float32), f)
    for j in root.iter("joint"):
        if j.find("parent") is None or j.find("child") is None:
            continue  # transmission stubs
        axis_el = j.find("axis")
        axis = [float(v) for v in axis_el.get("xyz").split()] if axis_el is not None else [0, 0, 1]
        joints.append(dict(name=j.get("name"), type=j.get("type"),
                           parent=j.find("parent").get("link"),
                           child=j.find("child").get("link"),
                           T=_parse(j), axis=np.array(axis, float)))
    return links, joints


def link_poses(joints, q, finger_open=0.02):
    """FK over the URDF tree -> {link_name: T} (base = panda_link0 frame)."""
    qmap = {f"panda_joint{i+1}": q[i] for i in range(7)}
    qmap["panda_finger_joint1"] = qmap["panda_finger_joint2"] = finger_open
    T = {"panda_link0": np.eye(4)}
    pending = list(joints)
    while pending:
        progressed = False
        for j in list(pending):
            if j["parent"] not in T:
                continue
            Tj = T[j["parent"]] @ j["T"]
            if j["type"] == "revolute":
                th = qmap.get(j["name"], 0.0)
                c, s = np.cos(th), np.sin(th)
                ax = j["axis"]
                K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
                R = np.eye(3) + s * K + (1 - c) * (K @ K)
                Tq = np.eye(4)
                Tq[:3, :3] = R
                Tj = Tj @ Tq
            elif j["type"] == "prismatic":
                Tq = np.eye(4)
                Tq[:3, 3] = j["axis"] * qmap.get(j["name"], 0.0)
                Tj = Tj @ Tq
            T[j["child"]] = Tj
            pending.remove(j)
            progressed = True
        if not progressed:
            break
    return T


def add_robot(vis, path, links, joints, q, T_world_base, pen_color=0x202020,
              pen_len=0.110, body_color=0xF4F4F2, dark_color=0x2E2E2E,
              pen_lat=None):
    poses = link_poses(joints, q)
    dark = {"panda_hand", "panda_leftfinger", "panda_rightfinger", "panda_link7"}
    for name, (v, f) in links.items():
        if name not in poses:
            continue
        col = dark_color if name in dark else body_color
        vis[f"{path}/{name}"].set_object(
            g.TriangularMeshGeometry(v, f), g.MeshLambertMaterial(color=col))
        vis[f"{path}/{name}"].set_transform(T_world_base @ poses[name])
    # pen (TCP frame, +z toward tip).  `pen_lat=None` -> the ACTIVE tool
    # (frames.PEN_LAT): with the LATERAL holder the pen hangs pen_lat along
    # hand x — a bracket bar from the grip out to the offset, then the pen
    # down to the tip, so the visual matches the planned geometry.
    from ..frames import lat_of
    lat = lat_of(pen_lat)
    T_tcp = np.eye(4)
    T_tcp[2, 3] = 0.1034
    T_pen = poses["panda_hand"] @ T_tcp
    rx = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1.0]])
    if lat == 0.0:
        seg = np.eye(4)
        seg[2, 3] = pen_len / 2 - 0.025      # extends 0.05 up into the fingers
        vis[f"{path}/pen"].set_object(
            g.Cylinder(pen_len + 0.05, 0.0045),
            g.MeshLambertMaterial(color=pen_color))
        vis[f"{path}/pen"].set_transform(T_world_base @ T_pen @ seg @ rx)
    else:
        rz = np.array([[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0],
                       [0, 0, 0, 1.0]])     # cylinder axis y -> x
        brk = np.eye(4)
        brk[0, 3] = lat / 2 - 0.01
        vis[f"{path}/pen_bracket"].set_object(
            g.Cylinder(lat + 0.02, 0.008),
            g.MeshLambertMaterial(color=dark_color))
        vis[f"{path}/pen_bracket"].set_transform(T_world_base @ T_pen @ brk @ rz)
        seg = np.eye(4)
        seg[0, 3] = lat
        seg[2, 3] = pen_len / 2 - 0.01
        vis[f"{path}/pen"].set_object(
            g.Cylinder(pen_len + 0.02, 0.0045),
            g.MeshLambertMaterial(color=pen_color))
        vis[f"{path}/pen"].set_transform(T_world_base @ T_pen @ seg @ rx)
