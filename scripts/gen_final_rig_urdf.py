#!/usr/bin/env python3
"""Generate the FINAL-RIG installation URDFs from aris_sixarm/rig_final.py.

Outputs (assets/final_rig/):
  installation.urdf   world + paper + frame boxes + THREE namespaced Franka
                      arms (cloned from the vendored panda_arm_hand.urdf,
                      finger joints fixed) welded at the drawing's mounts
  environment.urdf    the static geometry alone (paper + frame boxes), for
                      composing with drake_models arms in the demo pipeline

Frame: the URDF world frame IS the canvas frame C (origin = paper front-left
corner, z=0 the paper top, meters) — the planning convention.

Everything geometric is read from rig_final.py; nothing here is a number.
Regenerate after any rig_final change:  python3 scripts/gen_final_rig_urdf.py
Verify in the station venv afterwards:  <venv>/bin/python scripts/check_final_rig_urdf.py
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aris_sixarm import rig_final  # noqa: E402
from aris_sixarm.frames import PEN_EXT  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC_URDF = ROOT / "assets/franka_description/urdf/panda_arm_hand.urdf"
OUT_DIR = ROOT / "assets/final_rig"
MESH_REL = "../franka_description/meshes/visual"   # relative to OUT_DIR
FINGER_FIX = 0.0285       # m, finger half-width holding the holder (CAD-derived)

# tool config (filled by the pen-holder CAD step; see rig_final/tool notes)
from aris_sixarm.rig_final import TOOL  # noqa: E402


def rpy_from_R(R):
    """URDF rpy (extrinsic XYZ: R = Rz(y) Ry(p) Rx(r)) from a matrix."""
    p = -np.arcsin(np.clip(R[2, 0], -1, 1))
    if abs(np.cos(p)) > 1e-9:
        r = np.arctan2(R[2, 1] / np.cos(p), R[2, 2] / np.cos(p))
        y = np.arctan2(R[1, 0] / np.cos(p), R[0, 0] / np.cos(p))
    else:                                  # gimbal: fold roll into yaw
        r = 0.0
        y = np.arctan2(-R[0, 1], R[1, 1]) if R[2, 0] < 0 else \
            np.arctan2(R[0, 1], R[1, 1])
    # verify (the caller's rotations are exact axis combinations)
    cr, sr, cp, sp, cy, sy = (np.cos(r), np.sin(r), np.cos(p), np.sin(p),
                              np.cos(y), np.sin(y))
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    assert np.allclose(Rz @ Ry @ Rx, R, atol=1e-9), (r, p, y, R)
    return r, p, y


def _fmt(v):
    return f"{v:.6f}".rstrip("0").rstrip(".") or "0"


def _origin(el, xyz=None, rpy=None):
    o = ET.SubElement(el, "origin")
    if xyz is not None:
        o.set("xyz", " ".join(_fmt(v) for v in xyz))
    if rpy is not None:
        o.set("rpy", " ".join(_fmt(v) for v in rpy))


def add_box_link(robot, name, lo, hi, rgba, collision=True, visual=True):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    c, size = (lo + hi) / 2.0, hi - lo
    link = ET.SubElement(robot, "link", name=name)
    for kind, want in (("visual", visual), ("collision", collision)):
        if not want:
            continue
        e = ET.SubElement(link, kind)
        _origin(e, xyz=c)
        g = ET.SubElement(e, "geometry")
        ET.SubElement(g, "box", size=" ".join(_fmt(s) for s in size))
        if kind == "visual":
            m = ET.SubElement(e, "material", name=f"{name}_mat")
            ET.SubElement(m, "color", rgba=" ".join(str(x) for x in rgba))
    j = ET.SubElement(robot, "joint", name=f"{name}_weld", type="fixed")
    ET.SubElement(j, "parent", link="world")
    ET.SubElement(j, "child", link=name)
    return link


def add_environment(robot):
    """Paper + every frame box (zmin=-10: below-paper structure included)."""
    W, H = rig_final.SHEET_FINAL
    t = rig_final.PAPER_THICK_CM / 100.0
    add_box_link(robot, "paper", (0, 0, -t), (W, H, 0.0),
                 (0.98, 0.97, 0.94, 1.0))
    steel = (0.35, 0.37, 0.40, 1.0)
    grey = (0.55, 0.55, 0.55, 1.0)
    for b in rig_final.frame_boxes_canvas(zmin=-10):
        rgba = grey if b["name"] in ("table_block", "feed_roll") else steel
        add_box_link(robot, f"frame_{b['name']}", b["lo"], b["hi"], rgba)


def clone_arm(robot, arm_key, arm_id):
    """Clone the vendored panda into `robot` with prefix arm{arm_id}_,
    welded to world at the rig_final mount pose."""
    pfx = f"arm{arm_id}_"
    src = ET.parse(SRC_URDF).getroot()
    for el in list(src):
        if el.tag not in ("link", "joint",
                          "{http://drake.mit.edu}collision_filter_group"):
            continue
        el = ET.fromstring(ET.tostring(el))          # deep copy
        el.set("name", pfx + el.get("name"))
        for sub in el.iter():
            if sub is el:
                continue
            for attr in ("link", "joint"):           # parent/child/mimic refs
                if sub.get(attr):
                    sub.set(attr, pfx + sub.get(attr))
            if sub.get("name") and ("collision_filter_group" in sub.tag
                                    or sub.tag == "material"):
                sub.set("name", pfx + sub.get("name"))
            if sub.tag == "mesh" and sub.get("filename"):
                f = sub.get("filename").rsplit("/", 1)[-1]
                sub.set("filename", f"{MESH_REL}/{f}")
        if el.tag == "joint" and "finger" in el.get("name"):
            el.set("type", "fixed")                  # pen rig: fingers fixed
            for m in list(el):
                if m.tag in ("mimic", "limit", "axis", "dynamics"):
                    el.remove(m)
            for o in el.iter("origin"):
                pass
            # displace the finger by the grip opening along the joint axis (y)
            o = el.find("origin")
            if o is None:
                o = ET.SubElement(el, "origin")
            xyz = [float(v) for v in (o.get("xyz") or "0 0 0").split()]
            sign = 1.0 if el.get("name").endswith("joint1") else -1.0
            xyz[1] += sign * FINGER_FIX
            o.set("xyz", " ".join(_fmt(v) for v in xyz))
        robot.append(el)
    # weld at the mount
    p, R = rig_final.arm_base_canvas(arm_key)
    j = ET.SubElement(robot, "joint", name=f"{pfx}mount_weld", type="fixed")
    _origin(j, xyz=p, rpy=rpy_from_R(R))
    ET.SubElement(j, "parent", link="world")
    ET.SubElement(j, "child", link=f"{pfx}panda_link0")
    return pfx


def add_pen_holder(robot, pfx):
    """The pen holder (rig_final.TOOL): visual = the CAD-extracted mesh in the
    panda_hand frame; collision = the union-envelope cylinder.  Attached as a
    fixed link on the hand."""
    if TOOL is None:
        return
    name = f"{pfx}pen_holder"
    link = ET.SubElement(robot, "link", name=name)
    v = ET.SubElement(link, "visual")
    _origin(v, xyz=(0, 0, 0))
    g = ET.SubElement(v, "geometry")
    ET.SubElement(g, "mesh", filename=TOOL["visual_mesh"])
    m = ET.SubElement(v, "material", name=f"{name}_mat")
    ET.SubElement(m, "color", rgba="0.25 0.25 0.28 1.0")
    cc = TOOL["collision"]
    c = ET.SubElement(link, "collision")
    _origin(c, xyz=(0, 0, (cc["z0"] + cc["z1"]) / 2.0))
    g = ET.SubElement(c, "geometry")
    ET.SubElement(g, "cylinder", radius=_fmt(cc["radius"]),
                  length=_fmt(cc["z1"] - cc["z0"]))
    j = ET.SubElement(robot, "joint", name=f"{name}_weld", type="fixed")
    ET.SubElement(j, "parent", link=f"{pfx}{TOOL['parent']}")
    ET.SubElement(j, "child", link=name)


def build(with_arms):
    robot = ET.Element("robot", name="aris_final_rig" if with_arms
                       else "aris_final_rig_env")
    ET.SubElement(robot, "link", name="world")
    add_environment(robot)
    if with_arms:
        for key, aid in rig_final.ARM_IDS.items():
            pfx = clone_arm(robot, key, aid)
            add_pen_holder(robot, pfx)
    ET.indent(robot)
    return ET.ElementTree(robot)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, with_arms in (("installation.urdf", True),
                            ("environment.urdf", False)):
        tree = build(with_arms)
        out = OUT_DIR / name
        tree.write(out, xml_declaration=True, encoding="unicode" if False
                   else "utf-8")
        n_link = len(tree.getroot().findall("link"))
        print(f"wrote {out.relative_to(ROOT)}  ({n_link} links)")


if __name__ == "__main__":
    main()
