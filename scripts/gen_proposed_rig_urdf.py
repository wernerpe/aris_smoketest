#!/usr/bin/env python3
"""Generate the PROPOSED-RIG installation URDFs from the code that defines it.

The PROPOSED rig (`ARIS_RIG=proposed`, docs/LAYOUT_STUDY.md v2) is the
all-ceiling layout: SIX inverted FR3 arms on one 2 x 3 grid over the merged
1.8034 x 3.63064 m paper web, carrying the LATERAL pen holder.  Nothing here
is a number — every dimension is read from:

    aris_sixarm/layout.py    LAYOUT_PROPOSED / FLEET_PROPOSED (base poses, h)
    aris_sixarm/mounts.py    MOUNTS (plate, boom, ceiling grid height)
    aris_sixarm/rig_final6.py SHEET_FINAL6 (the canvas)
    aris_sixarm/rig_final.py  paper thickness, tool capsule radii
    aris_sixarm/frames.py     TCP chain, pen offsets, FR3 joint limits

Outputs (assets/proposed_rig/):
  installation.urdf   world + paper + the six arms' mount hardware + ceiling
                      reference plane + SIX namespaced Franka arms (cloned
                      from the vendored panda_arm_hand.urdf, fingers fixed,
                      **joint limits replaced with the FR3 values**) welded at
                      the fleet base poses, each carrying the pen holder
                      (the 22-deg CAD as visual meshes, the planner's L-shaped
                      capsule envelope + a fitted primitive envelope as
                      collision, the graphite, and a pen_tip frame)
  environment.urdf    the static geometry alone (paper + mounts + ceiling
                      reference), for composing with drake_models arms

Frame: the URDF world frame IS the canvas frame C (origin = paper corner,
z = 0 the paper top surface, metres) — the planning convention, same as the
final rig's URDF.

WHAT IS AND IS NOT MODELLED
---------------------------
  * paper: a 2 mm web at z in [-0.002, 0].  There is NO table, roll, winder or
    transport structure for this layout — none is designed yet.
  * mount hardware: exactly what the layout study gated against
    (`mounts.arm_mount_boxes`): a 0.226 x 0.190 x 0.05 base plate sitting ON
    TOP of each arm's base flange, and a vertical boom from the plate top to
    the nominal 2.34 m ceiling grid.  The study carries the boom as the
    CIRCUMSCRIBED SQUARE column of the r = 0.10 cylinder so its box machinery
    stays conservative, so the URDF writes the SQUARE as the collision
    geometry (what was certified) and the CYLINDER as the visual (what will
    be built).  A test pins the collision box against `mounts`.
  * ceiling grid: NOT MODELLED.  The steel design is pending; the study says
    so explicitly and gated no cross-member.  `ceiling_grid_ref` is a
    VISUAL-ONLY reference plate at MOUNTS.ceiling_z marking the level the
    booms terminate at — it has no collision geometry and must not be read as
    structure.
  * each arm's OWN plate and boom are collision-filtered against that arm
    (it is bolted to them by construction — `mounts.obstacles_for`); every
    OTHER arm still sees them, which is the study's convention exactly.

FR3, NOT PANDA.  The vendored `panda_arm_hand.urdf` carries Panda joint
limits, which are the wrong robot.  Every revolute limit is rewritten here
from `frames.FR3_MIN/FR3_MAX` (position), `frames.QD_MAX` (velocity) and
`frames.TAU_MAX` (effort).  The Panda `drake:acceleration` attribute is
DROPPED rather than restated as an FR3 number: no FR3 acceleration source
exists in this repo.

THE TOOL.  The pen tip stays exactly where `frames` puts it:
TCP + R @ (PEN_LAT_HOLDER, 0, PEN_EXT_HOLDER) — USER-SPECIFIED, not
gate-validated (frames.py), and untouched by any of this.  What the 2026-08-19 CAD delivery adds is the LOOK: the real 22-deg
clutch holder as two decimated hand-frame meshes, placed by
`rig_final.penholder22_T_hand` — an INFERENCE, because the delivery has no
assembly file.  Read `rig_final.PENHOLDER22` before trusting the picture: the
housing's own clocking is 23 deg where the planner's tool transform implies
45, and the difference is parked in a fingertip cradle nobody has sent us.

Inertials and arm meshes are the vendored ones, untouched — no mass property
is invented here, and the tool links carry no inertial at all (all welded).

Regenerate after any layout/mounts change:
    python3 scripts/gen_proposed_rig_urdf.py
Verify in the station venv afterwards:
    <venv>/bin/python scripts/check_proposed_rig_urdf.py
"""
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# the rpy extraction is the FINAL rig's, verbatim (and it self-verifies).  The
# final-rig generator is NOT edited, so its two outputs stay byte-identical.
from gen_final_rig_urdf import rpy_from_R  # noqa: E402

from aris_sixarm import mounts, rig_final  # noqa: E402
from aris_sixarm.frames import (D_HAND_TCP, FR3_MAX, FR3_MIN,  # noqa: E402
                                PEN_EXT_HOLDER, PEN_LAT_HOLDER, QD_MAX, TAU_MAX)
from aris_sixarm.layout import LAYOUT_PROPOSED, FLEET_PROPOSED  # noqa: E402
from aris_sixarm.rig_final6 import SHEET_FINAL6  # noqa: E402

# DRAKE READS THE PREFIX LITERALLY.  Its URDF parser is tinyxml2-based and not
# namespace-aware: it matches the element name "drake:collision_filter_group"
# as a STRING.  ElementTree, left alone, serialises that namespace under an
# invented prefix (xmlns:ns0=...), and every filter group in the file is then
# silently ignored — which is exactly what has happened to the vendored panda's
# own group_link57 / group_link68 in assets/final_rig/installation.urdf
# (verified: drake reports those pairs as unfiltered).  Registering the prefix
# is the whole fix, and it must happen before anything is serialised.
ET.register_namespace("drake", "http://drake.mit.edu")

# WARNING — THE ARM COLLISION SPHERES IN HERE ARE A THIRD SCHEMATIC, AND THEY
# ARE UNAUDITED.  The vendored panda carries its arm collision geometry as 402
# spheres (14 on link0, r = 0.06, and so on down the chain), and everything
# this repo certifies is measured against a COMPLETELY DIFFERENT model: the
# capsules of `coordination.CAPSULES` and the boxes of `mounts`.  On
# 2026-08-26 `scripts/collision_audit.py` put the capsules on the instrument
# against the manufacturer's collision MESHES and found them optimistic by up
# to 78 mm; the capsules were corrected, the spheres were not, and nobody has
# ever checked the two against each other.
#
# What that means in practice: drake's own collision queries on this file —
# `scripts/check_proposed_rig_urdf.py`, the meshcat playback, anything that
# asks the plant for a distance — are answering with the SPHERES, not with the
# model the programme was certified under.  Read a drake clearance from this
# URDF as an independent opinion, never as a confirmation.
#
# The fix, when it is worth doing, is to replace the sphere sets with the
# manufacturer's collision meshes (`vamp/resources/panda/meshes/collision/
# link{0..7}.obj` + hand/fingers — the same geometry the audit used as ground
# truth, and already proven to reproduce the FR3 collision boxes to 0.5 mm).
# That is a nice-to-have, not a blocker: nothing in the certification path
# reads this file.
SRC_URDF = ROOT / "assets/franka_description/urdf/panda_arm_hand.urdf"
OUT_DIR = ROOT / "assets/proposed_rig"
MESH_REL = "../franka_description/meshes/visual"   # relative to OUT_DIR
# NOT 0.018, DELIBERATELY.  The mount post is 50 mm long with a 7.000 mm socket
# in each end, so the faces a fingertip seats on are 36.000 mm apart, and the
# 10-deg assembly measures 36.0008 (rig_final.PENHOLDER22, docs/SYSTEM_MODEL.md
# 7a).  0.0285 is the fingertip's BACK face and comes from 50 + 2 x 3.5, which
# adds the tip left OUTSIDE the socket instead of subtracting the 7 mm inside
# it.  `gen_system_model.py` carries the corrected 0.018.  This generator does
# NOT, because assets/proposed_rig/ is what every certified number in the repo
# was checked against and re-cutting it is a re-certification, not an edit.
FINGER_FIX = 0.0285   # m, finger half-width holding the holder (final-rig CAD)

M = mounts.MOUNTS
H_INV = float(LAYOUT_PROPOSED["h"])       # 0.850, the study's winning height
CEILING_REF_T = 0.03      # m, thickness of the visual-only reference plate
CEILING_REF_PAD = 0.35    # m, how far it overhangs the canvas (viz only)
TIP_MARKER_R = 0.004      # m, the pen_tip frame's visual marker


# ---------------------------------------------------------------------------
# WRITER PRECISION.  The final rig writes 1e-6 because its numbers came off a
# drawing quoted in whole hundredths of a centimetre.  This layout's numbers
# are DERIVED (H/6, H/2, 5H/6 of a 3.63064 m web; pi in the inverted seat), so
# 1e-6 truncation is real error: it costs ~0.9 um at the pen tip, which is
# the same order as the check's own tolerance.  1e-9 puts the round trip at
# nanometres and leaves nothing to argue about.
# ---------------------------------------------------------------------------
def _fmt(v):
    return f"{float(v):.9f}".rstrip("0").rstrip(".") or "0"


def _origin(el, xyz=None, rpy=None):
    o = ET.SubElement(el, "origin")
    if xyz is not None:
        o.set("xyz", " ".join(_fmt(v) for v in xyz))
    if rpy is not None:
        o.set("rpy", " ".join(_fmt(v) for v in rpy))


def add_box_link(robot, name, lo, hi, rgba, collision=True, visual=True):
    """A world-welded box, from its (lo, hi) corners.  Mirrors the final rig's
    `add_box_link` — same element order, this writer's precision."""
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


ARM_LINKS = tuple(f"panda_link{i}" for i in range(9)) + (
    "panda_hand", "panda_leftfinger", "panda_rightfinger")
TOOL_LINKS = ("pen_bracket", "pen_body", "pen_holder", "pen_lead")
WRIST_LINKS = ("panda_link6", "panda_link7", "panda_link8", "panda_hand",
               "panda_leftfinger", "panda_rightfinger")


# ---------------------------------------------------------------------------
# static geometry
# ---------------------------------------------------------------------------
def add_environment(robot):
    """Paper web + every arm's mount hardware + the ceiling reference plate."""
    W, H = SHEET_FINAL6
    t = rig_final.PAPER_THICK_CM / 100.0
    add_box_link(robot, "paper", (0, 0, -t), (W, H, 0.0),
                 (0.98, 0.97, 0.94, 1.0))
    for aid, spec in FLEET_PROPOSED.items():
        add_mount(robot, aid, spec)
    # the ceiling grid LEVEL, not the grid: visual only, no collision.
    add_box_link(robot, "ceiling_grid_ref",
                 (-CEILING_REF_PAD, -CEILING_REF_PAD, M.ceiling_z),
                 (W + CEILING_REF_PAD, H + CEILING_REF_PAD,
                  M.ceiling_z + CEILING_REF_T),
                 (0.72, 0.71, 0.67, 0.25), collision=False)


def add_mount(robot, aid, spec):
    """One arm's schematic hardware: base plate + boom.

    Geometry straight from `mounts.arm_mount_boxes` — the same boxes the
    layout study handed the atlas as obstacles.  The boom's collision is that
    box (the circumscribed square column); its visual is the r = boom_r
    cylinder the box conservatively encloses.
    """
    boxes = {b["name"]: b for b in mounts.arm_mount_boxes(
        spec.mount, spec.xy, spec.yaw, H_INV, tag=f"mount{aid}", m=M)}
    plate = boxes[f"mount{aid}_plate"]
    add_box_link(robot, plate["name"], plate["lo"], plate["hi"],
                 (0.33, 0.33, 0.35, 1.0))
    boom = boxes[f"mount{aid}_boom"]
    lo, hi = np.asarray(boom["lo"], float), np.asarray(boom["hi"], float)
    c = (lo + hi) / 2.0
    link = ET.SubElement(robot, "link", name=boom["name"])
    v = ET.SubElement(link, "visual")                 # what will be built
    _origin(v, xyz=c)
    g = ET.SubElement(v, "geometry")
    ET.SubElement(g, "cylinder", radius=_fmt(M.boom_r),
                  length=_fmt(hi[2] - lo[2]))
    m = ET.SubElement(v, "material", name=f"{boom['name']}_mat")
    ET.SubElement(m, "color", rgba="0.45 0.46 0.48 1.0")
    col = ET.SubElement(link, "collision")            # what was certified
    _origin(col, xyz=c)
    g = ET.SubElement(col, "geometry")
    ET.SubElement(g, "box", size=" ".join(_fmt(s) for s in (hi - lo)))
    j = ET.SubElement(robot, "joint", name=f"{boom['name']}_weld",
                      type="fixed")
    ET.SubElement(j, "parent", link="world")
    ET.SubElement(j, "child", link=boom["name"])


# ---------------------------------------------------------------------------
# the arms
# ---------------------------------------------------------------------------
def _set_fr3_limits(el, idx):
    """Rewrite a cloned panda revolute joint's <limit> with the FR3 values."""
    lim = el.find("limit")
    if lim is None:
        return
    lim.set("effort", _fmt(TAU_MAX[idx]))
    lim.set("lower", _fmt(FR3_MIN[idx]))
    lim.set("upper", _fmt(FR3_MAX[idx]))
    lim.set("velocity", _fmt(QD_MAX[idx]))
    # the Panda acceleration hint is a different robot's number and this repo
    # has no FR3 source for it — drop it rather than restate it.
    lim.attrib.pop("{http://drake.mit.edu}acceleration", None)


def clone_arm(robot, arm_id, spec):
    """Clone the vendored panda into `robot` with prefix arm{arm_id}_, welded
    to world at the fleet base pose, with FR3 limits and fixed fingers."""
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
        name = el.get("name")
        if el.tag == "joint" and name.startswith(f"{pfx}panda_joint") \
                and name[-1].isdigit() and 1 <= int(name[-1]) <= 7:
            _set_fr3_limits(el, int(name[-1]) - 1)
        if el.tag == "joint" and "finger" in name:
            el.set("type", "fixed")                  # pen rig: fingers fixed
            for m in list(el):
                if m.tag in ("mimic", "limit", "axis", "dynamics"):
                    el.remove(m)
            # displace the finger by the grip opening along the joint axis (y)
            o = el.find("origin")
            if o is None:
                o = ET.SubElement(el, "origin")
            xyz = [float(v) for v in (o.get("xyz") or "0 0 0").split()]
            sign = 1.0 if name.endswith("joint1") else -1.0
            xyz[1] += sign * FINGER_FIX
            o.set("xyz", " ".join(_fmt(v) for v in xyz))
        robot.append(el)
    T = spec.T_world_base(H_INV)          # the fleet's own base transform
    j = ET.SubElement(robot, "joint", name=f"{pfx}mount_weld", type="fixed")
    _origin(j, xyz=T[:3, 3], rpy=rpy_from_R(T[:3, :3]))
    ET.SubElement(j, "parent", link="world")
    ET.SubElement(j, "child", link=f"{pfx}panda_link0")
    return pfx


def _cyl_link(robot, name, parent, joint_xyz, cyl_xyz, cyl_rpy, radius,
              length, rgba, r_collision=None):
    """A cylinder on a fixed joint.  `rgba=None` writes NO visual — the link
    is a collision envelope and has nothing to show."""
    link = ET.SubElement(robot, "link", name=name)
    for kind in (("visual", "collision") if rgba is not None
                 else ("collision",)):
        e = ET.SubElement(link, kind)
        _origin(e, xyz=cyl_xyz, rpy=cyl_rpy)
        g = ET.SubElement(e, "geometry")
        r = radius if kind == "visual" or r_collision is None else r_collision
        ET.SubElement(g, "cylinder", radius=_fmt(r), length=_fmt(length))
        if kind == "visual":
            m = ET.SubElement(e, "material", name=f"{name}_mat")
            ET.SubElement(m, "color", rgba=" ".join(str(x) for x in rgba))
    j = ET.SubElement(robot, "joint", name=f"{name}_weld", type="fixed")
    _origin(j, xyz=joint_xyz)
    ET.SubElement(j, "parent", link=parent)
    ET.SubElement(j, "child", link=name)
    return link


def _mesh_visual(link, filename, rgba):
    v = ET.SubElement(link, "visual")
    _origin(v, xyz=(0, 0, 0))                # the mesh is baked in hand frame
    g = ET.SubElement(v, "geometry")
    ET.SubElement(g, "mesh", filename=filename)
    stem = filename.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    m = ET.SubElement(v, "material", name=f"{link.get('name')}_{stem}_mat")
    ET.SubElement(m, "color", rgba=" ".join(str(x) for x in rgba))


def add_cad_holder(robot, pfx, pen_ext=PEN_EXT_HOLDER, pen_lat=PEN_LAT_HOLDER):
    """The REAL 22-deg clutch holder: CAD visual + a conservative primitive
    envelope, welded to the hand at the identity.

    Visual = the two decimated, hand-frame meshes `rig_final.PENHOLDER22`
    names (`scripts/extract_penholder22_meshes.py` bakes the placement in, the
    final rig's own convention).  There is NO assembly file in the delivery,
    so that placement is INFERRED — `rig_final.penholder22_T_hand` states the
    three assumptions and what they cost.

    Collision = `rig_final.penholder22_collision`: three cylinders coaxial
    with the bore whose radii are the largest distance any vertex of either
    mesh reaches from that axis inside its band, plus a fourth for the pencil
    tail (7c).  A 7 420-triangle concave part is not a collision geometry, and
    the final rig's answer to exactly that problem is a primitive envelope too
    (`TOOL["collision"]`).  Nothing here is fitted by eye; the extract script
    re-proves the enclosure on every run.  No inertial is written (welded
    link).

    RE-CERTIFIED 2026-09-03.  The housing was mounted END-FOR-END until then
    and the flip moves these cylinders 30 mm along the bore — see
    `rig_final.penholder22_T_hand` and docs/SYSTEM_MODEL.md 7c.  It is the one
    change this tree has taken since it was certified, and it is a change
    because the alternative was leaving a known-wrong body in the asset every
    certified number was checked against.
    """
    P = rig_final.PENHOLDER22
    name = f"{pfx}pen_holder"
    link = ET.SubElement(robot, "link", name=name)
    for mesh, rgba in zip(P["visual_meshes"], ((0.24, 0.25, 0.28, 1.0),
                                               (0.13, 0.14, 0.16, 1.0))):
        _mesh_visual(link, mesh, rgba)
    for kind, T, par in rig_final.penholder22_collision(pen_ext, pen_lat,
                                                        D_HAND_TCP):
        c = ET.SubElement(link, "collision")
        _origin(c, xyz=T[:3, 3], rpy=rpy_from_R(T[:3, :3]))
        g = ET.SubElement(c, "geometry")
        assert kind == "cylinder", kind
        ET.SubElement(g, "cylinder", radius=_fmt(par[0]), length=_fmt(par[1]))
    j = ET.SubElement(robot, "joint", name=f"{name}_weld", type="fixed")
    _origin(j, xyz=(0.0, 0.0, 0.0))          # the meshes carry the placement
    ET.SubElement(j, "parent", link=f"{pfx}{P['parent']}")
    ET.SubElement(j, "child", link=name)


def add_pen_holder(robot, pfx, pen_ext=PEN_EXT_HOLDER, pen_lat=PEN_LAT_HOLDER):
    """The tool: the PLANNER's envelope, the CAD holder, and the pen tip.

        tcp         = panda_hand + (0, 0, D_HAND_TCP)      [frames.TCP_D]
        pen_bracket = tcp -> tcp + (pen_lat, 0, 0)         [along hand x]
        pen_body    = corner -> corner + (0, 0, pen_ext)   [along tool z]
        pen_tip     = the tool tip frame, = TCP + R @ (pen_lat, 0, pen_ext)

    `pen_bracket` / `pen_body` are `rig_final.STATIC_CAPSULES_LAT` written out
    as geometry: the L-shaped two-capsule envelope the layout study actually
    gated every pose against, at BRACKET_R_LAT / PEN_R_LAT.  They are
    COLLISION-ONLY — the real holder is not an L and drawing one would be a
    lie — but they stay in the file so that anything collision-checking this
    URDF is checking at least what the planner certified.  The CAD holder's
    own envelope is added alongside (`add_cad_holder`); the union is what a
    checker sees, and the union is conservative for both models.

    `pen_lead` is the graphite stick: the one part of the assembly that is NOT
    in the CAD (it is a consumable, and its protrusion is the adjustable
    setting), drawn at the clutch bore's own radius from the CAP'S OUTER FACE
    — where the pen leaves — to the planning tip.  Its length is the
    delivery's headline problem, and correcting the housing's sense made it
    worse rather than better: 125.6 mm, against 100.5 mm end-for-end.
    """
    tcp = f"{pfx}tcp"
    link = ET.SubElement(robot, "link", name=tcp)      # pure frame, no geometry
    j = ET.SubElement(robot, "joint", name=f"{tcp}_weld", type="fixed")
    _origin(j, xyz=(0.0, 0.0, D_HAND_TCP))
    ET.SubElement(j, "parent", link=f"{pfx}panda_hand")
    ET.SubElement(j, "child", link=tcp)

    _cyl_link(robot, f"{pfx}pen_bracket", tcp, (0.0, 0.0, 0.0),
              (pen_lat / 2.0, 0.0, 0.0), (0.0, np.pi / 2, 0.0),
              rig_final.BRACKET_R_LAT, pen_lat, None)
    _cyl_link(robot, f"{pfx}pen_body", tcp, (pen_lat, 0.0, 0.0),
              (0.0, 0.0, pen_ext / 2.0), (0.0, 0.0, 0.0),
              rig_final.PEN_R_LAT, pen_ext, None)

    add_cad_holder(robot, pfx, pen_ext, pen_lat)

    # the graphite: the cap's outer face -> tip, along the BORE.  Not the
    # TCP -> tip ray: since 2026-09-04 the grip sits at the far end of the Fat
    # finger plates (PENHOLDER22["grip_hand_x"]) and the two are 20 deg apart.
    P = rig_final.PENHOLDER22
    la, lb = rig_final.penholder22_lead(pen_ext, pen_lat, D_HAND_TCP)
    tcp_o = np.array([0.0, 0.0, D_HAND_TCP])       # the link is welded to `tcp`
    lean = float(np.arctan2(lb[0] - la[0], lb[2] - la[2]))   # the BORE's own
    lctr = tuple(0.5 * (la + lb) - tcp_o)
    lL = float(np.linalg.norm(lb - la))
    _cyl_link(robot, f"{pfx}pen_lead", tcp, (0.0, 0.0, 0.0),
              lctr, (0.0, lean, 0.0),
              P["lead_r"], lL, (0.16, 0.16, 0.17, 1.0),
              r_collision=P["lead_r_coll"])

    tip = f"{pfx}pen_tip"                              # the tool tip FRAME
    link = ET.SubElement(robot, "link", name=tip)
    v = ET.SubElement(link, "visual")
    _origin(v, xyz=(0.0, 0.0, 0.0))
    g = ET.SubElement(v, "geometry")
    ET.SubElement(g, "sphere", radius=_fmt(TIP_MARKER_R))
    m = ET.SubElement(v, "material", name=f"{tip}_mat")
    ET.SubElement(m, "color", rgba="0.90 0.10 0.10 1.0")
    j = ET.SubElement(robot, "joint", name=f"{tip}_weld", type="fixed")
    _origin(j, xyz=(0.0, 0.0, pen_ext))
    ET.SubElement(j, "parent", link=f"{pfx}pen_body")
    ET.SubElement(j, "child", link=tip)


def add_collision_filters(robot, arm_id):
    """Two exclusions per arm, both of them the planning model's own:

      * the arm is BOLTED to its own plate and boom (`mounts.obstacles_for`
        excludes the own mount; every other arm still sees it);
      * the pen holder is CLAMPED IN THE HAND, so its 5 cm capsule envelope
        overlapping the wrist is construction, not contact.
    """
    pfx = f"arm{arm_id}_"
    for name, members, ignores in (
            (f"{pfx}body", [pfx + n for n in ARM_LINKS], [f"mount{arm_id}"]),
            (f"mount{arm_id}", [f"mount{arm_id}_plate", f"mount{arm_id}_boom"],
             []),
            (f"{pfx}tool", [pfx + n for n in TOOL_LINKS], [f"{pfx}wrist"]),
            (f"{pfx}wrist", [pfx + n for n in WRIST_LINKS], [])):
        g = ET.SubElement(robot, "{http://drake.mit.edu}collision_filter_group",
                          name=name)
        for lk in members:
            ET.SubElement(g, "{http://drake.mit.edu}member", link=lk)
        for other in ignores:
            ET.SubElement(g, "{http://drake.mit.edu}"
                             "ignored_collision_filter_group", name=other)


# ---------------------------------------------------------------------------
def build(with_arms):
    robot = ET.Element("robot", name="aris_proposed_rig" if with_arms
                       else "aris_proposed_rig_env")
    ET.SubElement(robot, "link", name="world")
    add_environment(robot)
    if with_arms:
        for aid, spec in FLEET_PROPOSED.items():
            pfx = clone_arm(robot, aid, spec)
            add_pen_holder(robot, pfx)
            add_collision_filters(robot, aid)
    ET.indent(robot)
    return ET.ElementTree(robot)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, with_arms in (("installation.urdf", True),
                            ("environment.urdf", False)):
        tree = build(with_arms)
        out = OUT_DIR / name
        tree.write(out, xml_declaration=True, encoding="utf-8")
        n_link = len(tree.getroot().findall("link"))
        shown = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
        print(f"wrote {shown}  ({n_link} links)")
    print(f"  {len(FLEET_PROPOSED)} inverted arms at h={H_INV} m, "
          f"canvas {SHEET_FINAL6[0]} x {SHEET_FINAL6[1]} m, "
          f"boom r={M.boom_r} to the {M.ceiling_z} m grid, "
          f"pen tip = TCP + R @ ({PEN_LAT_HOLDER}, 0, {PEN_EXT_HOLDER})")


if __name__ == "__main__":
    main()
