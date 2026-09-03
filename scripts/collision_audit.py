#!/usr/bin/env python3
"""COLLISION AUDIT — ground-truth the schematic collision models against MESHES.

WHY THIS EXISTS.  Every inter-arm clearance number in this repo is a CAPSULE
number: `coordination.CAPSULES(_LAT)` is seven segments per arm at radii
0.09 / 0.07 / 0.05, and `mounts.arm_column_box` reduces a neighbour's whole
base to a 0.12 m box over 0.333 m of base z.  The decision to raise the
ceiling grid from h = 0.850 to h ~ 0.922 (commit de6225f) rests entirely on
those models: at h = 0.85 a drawing arm folds, its elbow capsule enters the
neighbour column's band and the corrected union falls to 98.10 %; at h = 0.922
the chain hangs below the band and the penalty goes to zero.  Nobody had ever
checked the capsules against the geometry they claim to envelope.

THE INSTRUMENT.  Exact triangle-mesh minimum distance (FCL BVH, `python-fcl`)
on meshes posed by THIS REPO'S OWN FK (`frames.fk`'s modified-DH chain, which
is joint-for-joint the URDF's link frames — `--validate` re-proves that against
the URDF's own joint origins and, if drake is reachable, against drake's FK).

THE GROUND TRUTH, and why it is what it is.  `assets/proposed_rig/
installation.urdf` carries SPHERE collision geometry for the arms (14 spheres
on link0, 6 on link1, ...), which is a second schematic model and not a
ground truth, so it is audited, not trusted.  The mesh ground truth is, per
link, the UNION of

  * the manufacturer's COLLISION mesh, `vamp/resources/panda/meshes/collision/
    link{0..7}.obj` + hand/finger — the geometry franka ships for exactly this
    question.  Panda, not FR3, and that is CHECKED not assumed: every one of
    the ten link AABBs reproduces the FR3 collision boxes in the station's own
    `assets/urdfs/fr3_franka_hand.urdf` to <= 0.5 mm (`--validate` prints the
    table), i.e. the two robots' collision shells are the same object;
  * the full-resolution VISUAL mesh, `assets/franka_description/meshes/visual/
    link{0..7}.gltf` (40 k vertices on link0, not decimated).  glTF is Y-up, so
    the link-frame transform is Rx(+90) after the file's own node rotation;
    `--validate` re-derives it by matching the collision AABB.

For link1..link7 and the hand the two agree to <= 1.3 mm (link6: 7.1 mm), so
the choice does not matter there.  On LINK 0 it matters a great deal: the
visual mesh carries the base connector and its cable stub, which reach
0.1769 m from the base z axis and 0.2307 m up the back of the base, and the
manufacturer's collision mesh covers NEITHER.  Both are kept; the cable
components are tagged so the report can quote the answer with and without.

ERROR BUDGET (metres, one-sided unless stated):
  FCL BVH distance            exact to float64 round-off, < 1e-7
  Panda-vs-FR3 shell identity <= 5e-4 (measured, see --validate)
  visual tessellation         CAD chord error, ~1e-4 (40 k verts on a 0.2 m part)
  FK vs URDF link frames      exact by construction, re-proved < 1e-12
  pen holder placement        NOT a measurement error: the holder has no
                              assembly file, so its pose on the hand is
                              INFERRED (rig_final.penholder22_T_hand).  Read
                              every tool number as "given that inference".
  penetration                 mesh distance is CLIPPED at 0 when two meshes
                              interpenetrate, which UNDER-states optimism.

READ-ONLY on the package.  Nothing in `aris_sixarm/` is touched.

REQUIREMENTS beyond the repo's own: `trimesh` and `python-fcl` (and `numpy`,
`scipy`, `networkx`, which trimesh's proximity queries want).  The repo's
system interpreter carries trimesh but NOT fcl, so this was run from a
throwaway venv:  `python3 -m venv v && v/bin/pip install numpy scipy trimesh
python-fcl networkx matplotlib && v/bin/python scripts/collision_audit.py`.
The `aris_sixarm` package (C++ IK extension included) imports there unchanged.

    python3 scripts/collision_audit.py --validate
    python3 scripts/collision_audit.py --part column
    python3 scripts/collision_audit.py --part fidelity --poses 400
    python3 scripts/collision_audit.py --part height --jobs 24
    python3 scripts/collision_audit.py --part all --jobs 24
"""
import argparse
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("ARIS_TOOL", "lateral")

import trimesh                                                   # noqa: E402

from aris_sixarm import (coordination, ik, layout, metrics,       # noqa: E402
                         mounts, rig_final)
from aris_sixarm.coordination import (CAPSULES_LAT, LINK_R,       # noqa: E402
                                      seg_seg_dist)
from aris_sixarm.frames import (DH, D_HAND_TCP, FR3_MAX, FR3_MIN, # noqa: E402
                                PEN_EXT, PEN_LAT_HOLDER, TCP_D,
                                fk, fk_many, joint_margin, rotx, rotz,
                                tool_offset, tool_points_many)
from aris_sixarm.metrics import GATE_MARGIN, GATE_SIGMA           # noqa: E402
from aris_sixarm.rig_final6 import SHEET_FINAL6                   # noqa: E402

trimesh.util.log.setLevel(50)

# ---------------------------------------------------------------------------
# where the ground truth lives
# ---------------------------------------------------------------------------
COLL_DIRS = ("/home/franka/git/vamp/resources/panda/meshes/collision",
             "/home/franka/git/franka_manipulation_station/vamp/resources/"
             "panda/meshes/collision",
             "/home/franka/git/franka_manipulation_station/.venv/lib/"
             "python3.10/site-packages/pybullet_data/franka_panda/meshes/"
             "collision")
VIS_DIR = ROOT / "assets/franka_description/meshes/visual"
FR3_BOX_URDF = ("/home/franka/git/franka_manipulation_station/assets/urdfs/"
                "fr3_franka_hand.urdf")
INSTALL_URDF = ROOT / "assets/proposed_rig/installation.urdf"
CAD_DIR = (ROOT.parent / "raw_slack_file_dump"
           / "Pen holder all parts 2026.08.19")
HOUSING_STL = "pen holder housing - 22 deg - reinforced - v20260429.STL"
CAP_STL = "pen holder cap v20250903.STL"
CLUTCH_STL = "pen clutch - Creatcolor monolith graphite v1.01.STL"
MM = 0.001

OUT = ROOT / "out"
GLTF_ZUP = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])   # Y-up -> Z-up

# the arm's own frames, in the order `link_transforms` returns them
FRAMES = ("link0", "link1", "link2", "link3", "link4", "link5", "link6",
          "link7", "link8", "hand")
# which frame each mesh part rides on
MESH_FRAME = {f"link{i}": f"link{i}" for i in range(8)}
MESH_FRAME.update(hand="hand", leftfinger="hand", rightfinger="hand")

MARGIN = 0.08              # coordination.SAFETY_M + CALIB_M
D1 = float(DH[0][2])       # 0.333, base flange -> shoulder
COLUMN_R = mounts.MOUNTS.column_r        # the legacy one-capsule radius
CAP_R = np.array([c[2] for c in CAPSULES_LAT], float)
NCAP = len(CAPSULES_LAT)
N_BASE = coordination.N_BASE   # leading entries that are the base column


def _coll_dir():
    for d in COLL_DIRS:
        if Path(d, "link0.obj").exists():
            return Path(d)
    raise SystemExit("no franka collision meshes found in %s" % (COLL_DIRS,))


# ===========================================================================
# 1. FORWARD KINEMATICS THAT CARRIES FRAMES, NOT JUST POINTS
# ===========================================================================
def link_transforms(q, tcp=TCP_D):
    """(7,) joints -> (10,4,4) link frames in the BASE frame.

    A literal transcription of `frames.fk`'s loop that keeps the whole T at
    every step instead of only its translation, plus the two fixed frames the
    chain hangs the hand on:

        link0 = I ... link7 = the DH chain
        link8 = link7 * trans(0, 0, D_FLANGE)
        hand  = link8 * Rz(-pi/4)

    `--validate` pins its translations against `frames.fk`'s chain points and
    its joint origins against the URDF's own <joint><origin> values.
    """
    T = np.eye(4)
    out = [T.copy()]
    for (al, a, d), th in zip(DH, q):
        ca, sa, ct, st = np.cos(al), np.sin(al), np.cos(th), np.sin(th)
        T = T @ np.array([[ct, -st, 0, a],
                          [st * ca, ct * ca, -sa, -sa * d],
                          [st * sa, ct * sa, ca, ca * d],
                          [0, 0, 0, 1.0]])
        out.append(T.copy())
    flange = np.eye(4)
    flange[2, 3] = tcp - D_HAND_TCP          # 0.107, link7 -> link8
    T8 = out[-1] @ flange
    c, s = np.cos(-np.pi / 4), np.sin(-np.pi / 4)
    twist = np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1.0, 0],
                      [0, 0, 0, 1.0]])
    out.append(T8)
    out.append(T8 @ twist)
    return np.array(out)


# ===========================================================================
# 2. THE MESH GROUND TRUTH
# ===========================================================================
def load_collision_meshes():
    """{part: trimesh} — the manufacturer's collision shells, link frames."""
    d = _coll_dir()
    out = {}
    for n in [f"link{i}" for i in range(8)] + ["hand", "finger"]:
        p = d / f"{n}.obj"
        if p.exists():
            out[n] = trimesh.load(p, process=False, force="mesh")
    return out


def load_visual_meshes():
    """{part: trimesh} — the full-res visual shells, in the LINK frames.

    glTF is Y-up.  The link-frame mesh is `Rx(+90) @ node_rotation @ raw`;
    `validate_meshes` re-derives that from the collision AABBs rather than
    trusting it.
    """
    out = {}
    for n in [f"link{i}" for i in range(8)] + ["hand"]:
        p = VIS_DIR / f"{n}.gltf"
        if not p.exists():
            continue
        sc = trimesh.load(p, process=False)
        V, F, off = [], [], 0
        for node in sc.graph.nodes_geometry:
            T, gname = sc.graph[node]
            g = sc.geometry[gname]
            V.append(trimesh.transform_points(g.vertices, np.asarray(T)))
            F.append(g.faces + off)
            off += len(g.vertices)
        m = trimesh.Trimesh(np.concatenate(V) @ GLTF_ZUP.T,
                            np.concatenate(F), process=False)
        out[n] = m
    return out


def split_link0_visual(m, z_cut=-0.002):
    """link0's visual -> (body, cable).  The base CONNECTOR and its cable stub
    are separate shells that reach behind and above the base casting; the
    manufacturer's collision mesh models neither, so they are carried apart so
    the report can quote both answers."""
    comps = m.split(only_watertight=False)
    body, cable = [], []
    for c in comps:
        (cable if c.bounds[0][2] < z_cut else body).append(c)
    cat = trimesh.util.concatenate
    return (cat(body) if body else None, cat(cable) if cable else None)


def arm_parts(gt="union"):
    """The audited arm as {part_name: (frame_name, trimesh)}.

    `gt`: "col" manufacturer collision only, "vis" visual only, "union" both
    (the default and the one every headline number uses).
    """
    col, vis = load_collision_meshes(), load_visual_meshes()
    parts = {}
    if gt in ("col", "union"):
        for n, m in col.items():
            if n == "finger":
                for side, y in (("leftfinger", 0.0285), ("rightfinger", -0.0285)):
                    T = np.eye(4)
                    T[:3, 3] = (0.0, y, 0.0584)
                    if side == "rightfinger":       # mirrored about hand x-z
                        T[:3, :3] = np.diag([1.0, -1.0, 1.0])
                    parts[f"col:{side}"] = ("hand", m.copy().apply_transform(T))
            else:
                parts[f"col:{n}"] = (n, m)
    if gt in ("vis", "union"):
        for n, m in vis.items():
            if n == "link0":
                body, cable = split_link0_visual(m)
                if body is not None:
                    parts["vis:link0"] = ("link0", body)
                if cable is not None:
                    parts["vis:link0_cable"] = ("link0", cable)
            else:
                parts[f"vis:{n}"] = (n, m)
    return parts


def tool_parts(pen_ext=PEN_EXT, pen_lat=PEN_LAT_HOLDER, with_lead=True):
    """The 22-deg pen holder as RAW CAD in the hand frame -> {name: mesh}.

    Raw STLs from the slack dump (mm), placed by `rig_final.penholder22_T_hand`
    — the same inference the URDF's visual meshes were baked with, but WITHOUT
    the 5 k-face decimation those went through.  Plus the graphite stick: the
    `lead`, the only part of the tool that reaches the paper, and the `tail`,
    the only part that reaches back at the wrist (docs/SYSTEM_MODEL.md 7c).
    """
    if not CAD_DIR.exists():
        return {}
    T_h, T_c, exit_x, reach = rig_final.penholder22_T_hand(pen_ext, pen_lat,
                                                           D_HAND_TCP)
    out = {}
    for nm, fn, T in (("housing", HOUSING_STL, T_h), ("cap", CAP_STL, T_c)):
        m = trimesh.load(CAD_DIR / fn, force="mesh", process=False)
        m.apply_scale(MM)
        out[nm] = m.apply_transform(T)
    if with_lead:
        P = rig_final.PENHOLDER22
        by, bz = P["bore_yz"]
        for nm, x0, x1 in (("lead", P["cap_end_x"],
                            P["post_xy"][0] + reach),
                           ("tail", -P["tail_len"], P["tail_x"])):
            cyl = trimesh.creation.cylinder(radius=P["lead_r"], sections=32,
                                            height=x1 - x0)
            T = np.eye(4)
            T[:3, :3] = np.array([[0, 0, 1.0], [0, 1.0, 0], [-1.0, 0, 0]])
            T[:3, 3] = (0.5 * (x0 + x1), by, bz)
            out[nm] = cyl.apply_transform(T_h @ T)
    return out


# ===========================================================================
# 3. THE DISTANCE ENGINE
# ===========================================================================
class Body:
    """One posed rigid part, as an FCL BVH plus its bounding sphere."""

    __slots__ = ("name", "obj", "c", "r", "mesh")

    def __init__(self, name, mesh):
        import fcl
        V = np.ascontiguousarray(mesh.vertices, np.float64)
        F = np.ascontiguousarray(mesh.faces, np.int32)
        bvh = fcl.BVHModel()
        bvh.beginModel(len(V), len(F))
        bvh.addSubModel(V, F)
        bvh.endModel()
        self.name, self.mesh = name, mesh
        self.obj = fcl.CollisionObject(bvh, fcl.Transform())
        self.c = 0.5 * (mesh.bounds[0] + mesh.bounds[1])
        self.r = float(np.linalg.norm(mesh.vertices - self.c, axis=1).max())

    def place(self, T):
        self.obj.setTransform(__import__("fcl").Transform(
            np.ascontiguousarray(T[:3, :3]), np.ascontiguousarray(T[:3, 3])))
        return self


def _dist(a, b):
    """Exact min distance between two posed BVHs; < 0 means they collide."""
    import fcl
    req = fcl.DistanceRequest(enable_nearest_points=False)
    res = fcl.DistanceResult()
    d = fcl.distance(a.obj, b.obj, req, res)
    return float(d)


def _world_centre(body, T):
    return T[:3, :3] @ body.c + T[:3, 3]


_BODY_CACHE = {}


def body_sets(parts, tools):
    """Two re-posable BVH sets (one per side of a pair) -> (names, BA, BB).

    Built ONCE and re-posed: an FCL object carries its transform, not its
    geometry, so a whole fleet at a whole pose set costs two copies of the
    robot and no more.  The raw holder STL is a million triangles of printed
    part; its CONVEX HULL goes into the BVH instead — conservative by
    construction (the hull contains the part), and the part measured 13.5 mm
    INSIDE the tool capsules, so tightening it could not change an answer.
    """
    if "sets" in _BODY_CACHE:
        return _BODY_CACHE["sets"]
    geom = {n: m for n, (f, m) in parts.items()}
    for n, m in tools.items():
        geom[f"tool:{n}"] = m.convex_hull if len(m.vertices) > 20000 else m
    names = list(geom)
    BA = {n: Body(n, m) for n, m in geom.items()}
    BB = {n: Body(n, m) for n, m in geom.items()}
    _BODY_CACHE["sets"] = (names, BA, BB)
    return names, BA, BB


def place_arm(names, src, spec, q, frame_index=None):
    """One arm at one pose -> (bodies, world transforms), in `names` order."""
    idx = frame_index or {f: i for i, f in enumerate(FRAMES)}
    Twb = np.asarray(spec.T_world_base(), float)
    L = link_transforms(q)
    bod, pos = [], []
    for n in names:
        if n.startswith("tool:"):
            T = Twb @ L[idx["hand"]]
        else:
            T = Twb @ L[idx[MESH_FRAME.get(_part_key(n), _part_key(n))]]
        bod.append(src[n])
        pos.append(T)
    return bod, pos


def pair_distance(bodies_a, poses_a, bodies_b, poses_b, cap=0.40):
    """Min mesh distance between two posed arms. -> (d, name_a, name_b).

    Bounding-sphere broad phase against the running best, so a 50 k-face
    link0 is only ever paid for when it can win.  `cap` clips: a value at or
    above it is reported as `cap` (nothing in this audit reads further).
    Distances are clipped at 0 when two meshes interpenetrate, which
    UNDER-states optimism and never over-states it.
    """
    ca = [(_world_centre(b, T), b.r, b, T) for b, T in zip(bodies_a, poses_a)]
    cb = [(_world_centre(b, T), b.r, b, T) for b, T in zip(bodies_b, poses_b)]
    order = []
    for xa, ra, ba, Ta in ca:
        for xb, rb, bb, Tb in cb:
            lo = float(np.linalg.norm(xa - xb)) - ra - rb
            if lo < cap:
                order.append((lo, ba, Ta, bb, Tb))
    order.sort(key=lambda t: t[0])
    best, na, nb = cap, None, None
    for lo, ba, Ta, bb, Tb in order:
        if lo >= best:
            break
        d = _dist(ba.place(Ta), bb.place(Tb))
        d = cap if d < 0 and d != d else d
        d = max(0.0, d)
        if d < best:
            best, na, nb = d, ba.name, bb.name
    return best, na, nb


# ===========================================================================
# 4. THE CAPSULE MODEL, MEASURED THE WAY THE PACKAGE MEASURES IT
# ===========================================================================
def world_chain(qs, spec):
    """(N,7) -> (N,11,3) chain points in WORLD, lateral tool included."""
    qs = np.asarray(qs, float).reshape(-1, 7)
    T, P = fk_many(qs)
    tool = tool_points_many(T, PEN_EXT, PEN_LAT_HOLDER)
    P11 = np.concatenate([P] + [t[:, None, :] for t in tool], axis=1)
    Twb = np.asarray(spec.T_world_base(), float)
    return P11 @ Twb[:3, :3].T + Twb[:3, 3]


def capsule_clearance(Pa, Pb):
    """Two (11,3) chains -> the conductor's own clearance, and its argmin.

    -> (clearance, capsule index on a, capsule index on b).  Exactly what
    `coordination.clearance_matrix` computes for one sample pair.
    """
    A0, A1 = coordination.cap_endpoints(Pa, CAPSULES_LAT)
    B0, B1 = coordination.cap_endpoints(Pb, CAPSULES_LAT)
    n = len(CAP_R)
    d = seg_seg_dist(A0[:, None], A1[:, None], B0[None], B1[None])
    d = d - CAP_R[:, None] - CAP_R[None, :]
    k = int(np.argmin(d))
    return float(d.flat[k]), k // n, k % n


def pose_bodies(parts, tools, q, spec):
    """(bodies, world transforms) for one arm at one pose."""
    Twb = np.asarray(spec.T_world_base(), float)
    L = link_transforms(q)
    idx = {f: i for i, f in enumerate(FRAMES)}
    bodies, poses = [], []
    for name, (frame, body) in parts.items():
        bodies.append(body)
        poses.append(Twb @ L[idx[frame]])
    Thand = Twb @ L[idx["hand"]]
    for name, body in tools.items():
        bodies.append(body)
        poses.append(Thand)
    return bodies, poses


# ===========================================================================
# 5. VALIDATION — the instrument, before any measurement
# ===========================================================================
def validate(rec):
    """Prove the FK, the mesh frames and the distance engine before using them."""
    v = rec.setdefault("validate", {})
    rng = np.random.default_rng(7)

    # --- FK translations == frames.fk's chain points -----------------------
    err = 0.0
    for _ in range(200):
        q = rng.uniform(FR3_MIN, FR3_MAX)
        T, P = fk(q)
        L = link_transforms(q)
        err = max(err, float(np.abs(L[:8, :3, 3] - P[:8]).max()))
        # the TCP: hand frame translated D_HAND_TCP along its own z
        tcp = L[9][:3, 3] + L[9][:3, 2] * D_HAND_TCP
        err = max(err, float(np.abs(tcp - T[:3, 3]).max()),
                  float(np.abs(L[9][:3, :3] - T[:3, :3]).max()))
    v["fk_vs_frames_fk_m"] = err
    print(f"[validate] link_transforms vs frames.fk        max err {err:.2e} m")

    # --- the URDF's own joint origins == the DH chain ----------------------
    r = ET.parse(INSTALL_URDF).getroot()
    want = [((0, 0, 0.333), (0, 0, 0)), ((0, 0, 0), (-np.pi / 2, 0, 0)),
            ((0, -0.316, 0), (np.pi / 2, 0, 0)),
            ((0.0825, 0, 0), (np.pi / 2, 0, 0)),
            ((-0.0825, 0.384, 0), (-np.pi / 2, 0, 0)),
            ((0, 0, 0), (np.pi / 2, 0, 0)), ((0.088, 0, 0), (np.pi / 2, 0, 0))]
    ju = {}
    for j in r.findall("joint"):
        n = j.get("name", "")
        if n.startswith("arm13_panda_joint"):
            o = j.find("origin")
            ju[n] = (np.fromstring(o.get("xyz"), sep=" "),
                     np.fromstring(o.get("rpy"), sep=" "))
    e = 0.0
    for i, (xyz, rpy) in enumerate(want, start=1):
        gx, gr = ju[f"arm13_panda_joint{i}"]
        e = max(e, float(np.abs(gx - xyz).max()), float(np.abs(gr - rpy).max()))
    v["urdf_joint_origins_vs_dh"] = e
    print(f"[validate] URDF joint origins vs frames.DH     max err {e:.2e}")

    # --- the mount weld == StudySpec.T_world_base --------------------------
    fleet = layout.FLEET_PROPOSED
    e = 0.0
    for j in r.findall("joint"):
        n = j.get("name", "")
        if n.endswith("_mount_weld"):
            aid = int(n.split("_")[0][3:])
            o = j.find("origin")
            xyz = np.fromstring(o.get("xyz"), sep=" ")
            e = max(e, float(np.abs(np.asarray(
                fleet[aid].T_world_base())[:3, 3] - xyz).max()))
    v["urdf_mount_weld_vs_spec_m"] = e
    print(f"[validate] URDF mount welds vs FLEET_PROPOSED  max err {e:.2e} m")

    # --- Panda collision shells == the station's FR3 collision boxes -------
    col = load_collision_meshes()
    fr3 = {}
    for L in ET.parse(FR3_BOX_URDF).getroot().findall("link"):
        nm = L.get("name", "").replace("fr3_", "")
        cs = L.findall("collision")
        if len(cs) == 1 and cs[0].find("geometry/box") is not None:
            fr3[nm] = np.fromstring(cs[0].find("geometry/box").get("size"),
                                    sep=" ")
    tab, worst = {}, 0.0
    for nm, box in fr3.items():
        if nm not in col:
            continue
        ext = col[nm].bounds[1] - col[nm].bounds[0]
        d = float(np.abs(np.sort(ext) - np.sort(box)).max())
        tab[nm] = dict(panda_extent=ext.round(6).tolist(),
                       fr3_box=box.round(6).tolist(), err_mm=1000 * d)
        worst = max(worst, d)
    v["panda_vs_fr3_boxes"] = tab
    v["panda_vs_fr3_worst_mm"] = 1000 * worst
    print(f"[validate] Panda collision AABB vs FR3 boxes   worst "
          f"{1000 * worst:.2f} mm over {len(tab)} links")

    # --- the visual glTF really does land in the link frame ---------------
    vis = load_visual_meshes()
    worst = 0.0
    for nm in ("link1", "link2", "link3", "link4", "link5", "link7", "hand"):
        a, b = vis[nm].bounds, col[nm].bounds
        worst = max(worst, float(np.abs(a - b).max()))
    v["visual_vs_collision_aabb_worst_mm"] = 1000 * worst
    print(f"[validate] visual glTF AABB vs collision AABB  worst "
          f"{1000 * worst:.2f} mm (frame check, link0/6 excluded)")

    # --- the distance engine on a case with a closed form -----------------
    a = Body("boxA", trimesh.creation.box((0.1, 0.1, 0.1)))
    b = Body("boxB", trimesh.creation.box((0.1, 0.1, 0.1)))
    errs = []
    for gap in (0.001, 0.01, 0.05, 0.2, 0.5):
        Ta, Tb = np.eye(4), np.eye(4)
        Tb[0, 3] = 0.1 + gap
        errs.append(abs(_dist(a.place(Ta), b.place(Tb)) - gap))
    ang = np.pi / 4                      # rotated, so the answer is a corner
    Tb = np.eye(4)
    Tb[:3, :3] = rotz(ang)
    Tb[0, 3] = 0.05 + 0.05 * np.sqrt(2) + 0.03
    errs.append(abs(_dist(a.place(np.eye(4)), b.place(Tb)) - 0.03))
    v["fcl_box_gap_max_err_m"] = float(max(errs))
    print(f"[validate] FCL box-gap closed form            max err "
          f"{max(errs):.2e} m")

    # --- visual outside collision, per link (the link0 story) -------------
    tab = {}
    for nm, m in vis.items():
        q = trimesh.proximity.ProximityQuery(col[nm])
        sd = -q.signed_distance(m.vertices)
        tab[nm] = dict(max_mm=float(1000 * sd.max()),
                       p99_mm=float(1000 * np.percentile(sd, 99)),
                       frac_outside=float((sd > 0).mean()))
    v["visual_outside_collision"] = tab
    print("[validate] visual vertices outside the collision mesh:")
    for nm, t in tab.items():
        print(f"           {nm:8s} max {t['max_mm']:7.1f} mm  "
              f"p99 {t['p99_mm']:6.1f} mm  frac {t['frac_outside']:.3f}")
    return v


def validate_drake(rec):
    """Cross-check `link_transforms` against drake's own FK, if reachable."""
    import subprocess
    py = ("/home/franka/git/franka_manipulation_station/.venv/bin/python")
    if not Path(py).exists():
        print("[validate] drake venv not found — skipped")
        return
    rng = np.random.default_rng(11)
    Q = rng.uniform(FR3_MIN, FR3_MAX, size=(8, 7))
    script = f'''
import json, numpy as np
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.framework import DiagramBuilder
b = DiagramBuilder()
plant, sg = AddMultibodyPlantSceneGraph(b, 0.0)
Parser(plant).AddModels({str(INSTALL_URDF)!r})
plant.Finalize()
ctx = plant.CreateDefaultContext()
Q = np.array({Q.tolist()!r})
names = [f"arm13_panda_link{{i}}" for i in range(9)] + ["arm13_panda_hand"]
out = []
for q in Q:
    for i in range(7):
        plant.GetJointByName(f"arm13_panda_joint{{i+1}}").set_angle(ctx, q[i])
    out.append([plant.EvalBodyPoseInWorld(
        ctx, plant.GetBodyByName(n)).GetAsMatrix4().tolist() for n in names])
print("JSON" + json.dumps(out))
'''
    p = subprocess.run([py, "-c", script], capture_output=True, text=True)
    line = [l for l in p.stdout.splitlines() if l.startswith("JSON")]
    if not line:
        print("[validate] drake FK failed:", p.stderr.strip()[-400:])
        return
    got = np.array(json.loads(line[0][4:]))
    spec = layout.FLEET_PROPOSED[13]
    Twb = np.asarray(spec.T_world_base(), float)
    err = 0.0
    for k, q in enumerate(Q):
        L = link_transforms(q)
        for i in range(10):
            err = max(err, float(np.abs(Twb @ L[i] - got[k, i]).max()))
    rec["validate"]["drake_fk_max_err_m"] = err
    print(f"[validate] drake FK vs link_transforms         max err {err:.2e} m")


# ===========================================================================
# 6. PART B — WHAT THE BASE COLUMN ACTUALLY IS
# ===========================================================================
def part_column(rec, nq1=181, dz=0.005):
    """The pose-invariant base envelope, as a radial profile about base z.

    link0 is fixed; link1 rotates about base z with q1 and sweeps a solid of
    revolution.  The union of the two IS the "base column" every neighbour
    obstacle in this repo claims to model, and the profile r_max(z) is the
    only cylinder that bounds it.  link2 is measured too, but reported apart:
    it is pose DEPENDENT and belongs to part C, not to a pose-invariant
    obstacle.
    """
    print("\n=== PART B: the base column, measured ===")
    parts = arm_parts("union")
    q1s = np.linspace(FR3_MIN[0], FR3_MAX[0], nq1)

    def profile(names, sweep_q1):
        """-> (z grid, r_max(z)) in the BASE frame, over the sweep."""
        pts = []
        for nm in names:
            frame, m = parts[nm]
            V = np.asarray(m.vertices, float)
            if frame == "link0":
                pts.append(V)
            elif frame == "link1" and sweep_q1:
                V1 = V + np.array([0.0, 0.0, D1])       # joint1 origin
                for a in q1s:
                    c, s = np.cos(a), np.sin(a)
                    pts.append(np.column_stack([c * V1[:, 0] - s * V1[:, 1],
                                                s * V1[:, 0] + c * V1[:, 1],
                                                V1[:, 2]]))
        P = np.concatenate(pts)
        r = np.hypot(P[:, 0], P[:, 1])
        zlo = np.floor(P[:, 2].min() / dz) * dz
        zhi = np.ceil(P[:, 2].max() / dz) * dz
        edges = np.arange(zlo, zhi + dz / 2, dz)
        k = np.clip(np.searchsorted(edges, P[:, 2], "right") - 1, 0,
                    len(edges) - 2)
        rmax = np.zeros(len(edges) - 1)
        np.maximum.at(rmax, k, r)
        return 0.5 * (edges[:-1] + edges[1:]), rmax

    sets = {
        "link0_collision": ["col:link0"],
        "link0_visual_body": ["vis:link0"],
        "link0_visual_cable": ["vis:link0_cable"],
        "link0_all": [n for n in parts if "link0" in n],
        "link1_swept": [n for n in parts if n.endswith("link1")],
        "column_all": [n for n in parts if "link0" in n or n.endswith("link1")],
    }
    prof = {}
    for nm, names in sets.items():
        names = [n for n in names if n in parts]
        if not names:
            continue
        z, r = profile(names, sweep_q1=True)
        prof[nm] = dict(z=z.tolist(), r=r.tolist(),
                        r_max=float(r.max()),
                        z_range=[float(z[r > 0].min() - dz / 2),
                                 float(z[r > 0].max() + dz / 2)])
        print(f"  {nm:20s} r_max {r.max():.4f} m   base-frame z "
              f"[{prof[nm]['z_range'][0]:+.4f}, {prof[nm]['z_range'][1]:+.4f}]")

    # the same, restricted to the band the obstacle claims: base z in [0, D1]
    band = {}
    for nm, p in prof.items():
        z, r = np.array(p["z"]), np.array(p["r"])
        s = (z >= 0.0) & (z <= D1) & (r > 0)
        band[nm] = float(r[s].max()) if s.any() else 0.0
    rec["column"] = dict(profiles=prof, band_r_max=band, d1=D1,
                         claimed_capsule_r=LINK_R, claimed_box_r=COLUMN_R,
                         nq1=nq1, dz=dz)

    # what a corrected obstacle would have to be
    z = np.array(prof["column_all"]["z"])
    r = np.array(prof["column_all"]["r"])
    s = r > 0
    z, r = z[s], r[s]
    print(f"\n  CLAIMED   capsule r = {LINK_R:.3f} over base z [0, {D1:.3f}] "
          f"(world z [h-{D1:.3f}, h])")
    print(f"  CLAIMED   box     r = {COLUMN_R:.3f}, AABB with +-r caps")
    print(f"  MEASURED  r_max   = {r.max():.4f} over base z "
          f"[{z.min():.4f}, {z.max():.4f}]")
    over_cap = r.max() - LINK_R
    print(f"  the real body reaches {1000 * over_cap:.1f} mm outside the "
          f"r = {LINK_R} capsule and "
          f"{1000 * (r.max() - COLUMN_R):.1f} mm outside the r = {COLUMN_R} box")

    # is 0.517 (= h - D1) the true bottom?
    zbot = float(z.max())
    print(f"  the column's true FAR end (base z, = below the plate for a "
          f"hanging arm) is {zbot:.4f} m, not {D1:.4f}: "
          f"{'PAST' if zbot > D1 else 'SHORT OF'} the claim by "
          f"{1000 * abs(zbot - D1):.1f} mm")

    # a conservative CYLINDER STACK that DOES bound it, for part C.  Only the
    # part BELOW the plate (base z >= 0) is a column a drawing arm can meet;
    # what is above it is inside the mount, where the plate and boom boxes
    # gate — and is reported separately because it does not fit in them.
    s = z >= 0.0
    stack = column_stack(z[s], r[s], nband=12)
    rec["column"]["corrected_stack"] = [
        dict(z0=a, z1=b, r=c) for a, b, c in stack]
    rec["column"]["corrected_stack_full"] = [
        dict(z0=a, z1=b, r=c) for a, b, c in column_stack(z, r, nband=8)]
    print("\n  a conservative replacement below the plate (cylinder bands, "
          "base-frame z):")
    for a, b, c in stack:
        print(f"    z [{a:+.3f}, {b:+.3f}]  r = {c:.4f}")
    print(f"  ABOVE the plate (base z < 0: the connector and its cable) the "
          f"body reaches r = {r[~s].max():.4f} m over "
          f"{1000 * abs(z[~s].min()):.0f} mm — the plate box is only "
          f"{mounts.MOUNTS.plate_xy[0] / 2:.3f} x "
          f"{mounts.MOUNTS.plate_xy[1] / 2:.3f} m in half-extent and "
          f"{1000 * mounts.MOUNTS.plate_t:.0f} mm thick")
    return rec["column"]


def column_stack(z, r, nband=8, pad=0.0):
    """A radial profile -> a few (z0, z1, radius) bands whose CAPSULE union
    contains it.  Capsule end caps bulge `r` past the axis, so the axis of
    each band is inset by its own radius where that stays inside the band;
    otherwise the cap overhang is simply accepted (it is conservative)."""
    lo, hi = float(z.min()), float(z.max())
    edges = np.linspace(lo, hi, nband + 1)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        s = (z >= a - 1e-9) & (z <= b + 1e-9)
        if not s.any():
            continue
        rr = float(r[s].max()) + pad
        out.append((float(a), float(b), rr))
    # merge adjacent bands of equal radius
    merged = []
    for a, b, rr in out:
        if merged and abs(merged[-1][2] - rr) < 1e-4:
            merged[-1] = (merged[-1][0], b, rr)
        else:
            merged.append((a, b, rr))
    return merged


# ===========================================================================
# 7. PART A — CAPSULE FIDELITY
# ===========================================================================
def capsule_excess(parts, tools, q, sample=4000, rng=None):
    """How far each part's real surface sticks OUT of the capsule union.

    -> {part: max over its surface of min_c (dist to capsule c axis - r_c)}.
    Positive is the danger direction: geometry the capsule model does not
    contain, which is exactly how much clearance a pairwise capsule distance
    can invent.  Measured in the BASE frame, so it is a property of the pose
    and not of where the arm is bolted.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    T, P = fk(q)
    tp = tool_points_many(T[None], PEN_EXT, PEN_LAT_HOLDER)
    P11 = np.concatenate([P] + [t[0][None] for t in tp], axis=0)
    A, B = P11[CAP_I], P11[CAP_J]
    L = link_transforms(q)
    idx = {f: i for i, f in enumerate(FRAMES)}
    out = {}
    items = [(n, f, m) for n, (f, m) in parts.items()]
    items += [(f"tool:{n}", "hand", m) for n, m in tools.items()]
    for name, frame, m in items:
        V = np.asarray(m.vertices, float)
        if len(V) > sample:
            V = V[rng.choice(len(V), sample, replace=False)]
        Tf = L[idx[frame]]
        W = V @ Tf[:3, :3].T + Tf[:3, 3]
        d = _pt_seg_dist(W, A, B) - CAP_R[None, :]
        out[name] = float(d.min(axis=1).max())
    return out


# WHICH CAPSULE IS RESPONSIBLE FOR WHICH LINK.  `CAPSULES_LAT` runs the base
# column's `N_BASE` BANDS — all of them sub-segments of (0,1), base->shoulder
# — and then 1:(1,3) upper arm  2:(3,4) elbow offset  3:(4,5) forearm
# 4:(5,7) wrist  5:(7,8) hand  6:(8,10) bracket  7:(10,9) pen, and
# `frames.fk`'s chain points are the URDF link-frame origins, so each link
# body lies along one or two consecutive capsules.  The attribution has to be
# STRUCTURAL and not nearest-capsule: in a folded pose the base casting's
# nearest capsule can be the hand's, and inflating the hand capsule by the
# base's overhang is arithmetically valid and physically meaningless.
#
# The numbers below are written in the LEGACY one-capsule-per-base indexing
# (0 = the whole base column) and translated by `_caps`, so this table still
# reads like the arm and the shipped fixture
# (tests/data/collision_audit_geometry.json, an 8-entry record) stays
# comparable across the banding.
_CAP_OF_LEGACY = {"link0": (0,), "link1": (0, 1), "link2": (1,),
                  "link3": (1, 2), "link4": (2, 3), "link5": (3, 4),
                  "link6": (4,), "link7": (4, 5), "link8": (5,),
                  "hand": (5,), "leftfinger": (5,), "rightfinger": (5,),
                  "housing": (6, 7), "cap": (6, 7), "lead": (6, 7)}


def _caps(legacy):
    """Legacy capsule indices -> indices into the shipped `CAPSULES_LAT`."""
    out = []
    for k in legacy:
        out.extend(range(N_BASE) if k == 0 else [N_BASE + k - 1])
    return tuple(sorted(set(out)))


CAP_OF = {k: _caps(v) for k, v in _CAP_OF_LEGACY.items()}


def _part_key(name):
    k = name.split(":", 1)[-1]
    return k.replace("link0_cable", "link0")


def capsule_inflation(parts, tools, Q, sample=6000, rng=None, exclude=()):
    """How much each capsule must GROW to contain the links it is FOR.

    -> (8,) metres.  Every surface point is charged to the nearest of the
    capsules `CAP_OF` says own its link, and that capsule's inflation is the
    worst such point over the pose set.  Growing each capsule by its own
    number makes the union contain the arm — the property every clearance
    number in this repo assumes and, as measured here, has never had.
    """
    rng = np.random.default_rng(1) if rng is None else rng
    idx = {f: i for i, f in enumerate(FRAMES)}
    items = [(_part_key(n), f, m) for n, (f, m) in parts.items()
             if not any(x in n for x in exclude)]
    items += [(n, "hand", m) for n, m in tools.items()]
    grown = np.zeros(len(CAP_R))
    for q in np.asarray(Q, float).reshape(-1, 7):
        T, P = fk(q)
        tp = tool_points_many(T[None], PEN_EXT, PEN_LAT_HOLDER)
        P11 = np.concatenate([P] + [t[0][None] for t in tp], axis=0)
        A, B = P11[CAP_I], P11[CAP_J]
        L = link_transforms(q)
        for key, frame, m in items:
            allowed = np.array(CAP_OF[key], int)
            V = np.asarray(m.vertices, float)
            if len(V) > sample:
                V = V[rng.choice(len(V), sample, replace=False)]
            Tf = L[idx[frame]]
            W = V @ Tf[:3, :3].T + Tf[:3, 3]
            d = (_pt_seg_dist(W, A[allowed], B[allowed])
                 - CAP_R[None, allowed])
            k = d.argmin(axis=1)
            np.maximum.at(grown, allowed[k], d[np.arange(len(W)), k])
    return np.maximum(grown, 0.0)


def _pt_seg_dist(P, A, B):
    """(N,3) points vs (M,3),(M,3) segments -> (N,M) distances."""
    d = B - A                                  # (M,3)
    L2 = np.sum(d * d, axis=1)
    L2 = np.where(L2 < 1e-18, 1.0, L2)
    w = P[:, None, :] - A[None, :, :]          # (N,M,3)
    t = np.clip(np.sum(w * d[None], axis=2) / L2[None], 0.0, 1.0)
    proj = A[None] + t[..., None] * d[None]
    return np.linalg.norm(P[:, None, :] - proj, axis=2)


def load_pose_sets(fleet, n_random=120, seed=3, atlas_dir="atlas_proposed_lat"):
    """{arm: (K,7)} — certified drawing poses, the baked parks, random gated.

    The atlas rows are certified for the arm they were swept for and carry
    the winning q directly.  Random poses are drawn from the joint box and
    kept only if they pass the same joint-margin gate and put every chain
    point above the paper, so the sample is poses the rig could actually be
    IN and not arbitrary configurations.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for aid, spec in fleet.items():
        rows = []
        f = OUT / atlas_dir / f"atlas_arm{aid}.npz"
        if f.exists():
            d = np.load(f)
            cols = list(d["columns"])
            Q = d["data"][:, [cols.index(f"q{i}") for i in range(1, 8)]]
            k = min(len(Q), 240)
            rows.append(Q[rng.choice(len(Q), k, replace=False)])
        rows.append(np.asarray(layout.Q_PARK_PROPOSED[aid], float)[None])
        got, tries = [], 0
        while len(got) < n_random and tries < 60 * n_random:
            tries += 1
            q = rng.uniform(FR3_MIN, FR3_MAX)
            if joint_margin(q) < GATE_MARGIN:
                continue
            Pw = world_chain(q[None], spec)[0]
            if Pw[1:9, 2].min() < 0.02:
                continue
            got.append(q)
        if got:
            rows.append(np.asarray(got))
        out[aid] = np.vstack(rows)
    return out


def part_fidelity(rec, n_pose=220, n_pair=1500, seed=5, jobs=1):
    """Capsule model vs mesh, on the poses the rig will actually hold."""
    print("\n=== PART A: capsule fidelity ===")
    rng = np.random.default_rng(seed)
    parts, tools = arm_parts("union"), tool_parts()
    print(f"  ground truth: {len(parts)} arm parts + {len(tools)} tool parts, "
          f"{sum(len(m.faces) for _, m in parts.values()) + sum(len(m.faces) for m in tools.values())} faces")

    res = {}
    for h in (0.850, 0.922):
        lay = dict(layout.LAYOUT_PROPOSED)
        lay["h"] = h
        fleet = layout.build_fleet(lay, q_park=layout.Q_PARK_PROPOSED)
        poses = load_pose_sets(fleet, n_random=80, seed=seed + int(h * 1000))
        print(f"\n  h = {h:.3f}: pose sets "
              f"{ {a: len(v) for a, v in poses.items()} }")

        # --- A1: per-part capsule containment excess -----------------------
        exc = {}
        sub = rng.choice(len(poses[13]), min(n_pose, len(poses[13])),
                         replace=False)
        for k in sub:
            e = capsule_excess(parts, tools, poses[13][k], rng=rng)
            for nm, v in e.items():
                exc[nm] = max(exc.get(nm, -9.9), v)
        print("  A1  how far each part's real surface reaches OUTSIDE the "
              "capsule union (mm, + = optimistic):")
        for nm, v in sorted(exc.items(), key=lambda t: -t[1]):
            flag = "  <-- OPTIMISTIC" if v > 0.0005 else ""
            print(f"      {nm:22s} {1000 * v:+8.1f}{flag}")

        # --- A2: pairwise, capsule vs mesh ---------------------------------
        names, BA, BB = body_sets(parts, tools)
        ids = sorted(fleet)
        pairs = [(a, b) for i, a in enumerate(ids) for b in ids[i + 1:]]
        recs = []
        t0 = time.time()
        for (a, b) in pairs:
            Qa, Qb = poses[a], poses[b]
            ka = rng.integers(0, len(Qa), n_pair)
            kb = rng.integers(0, len(Qb), n_pair)
            for i, j in zip(ka, kb):
                Pa = world_chain(Qa[i][None], fleet[a])[0]
                Pb = world_chain(Qb[j][None], fleet[b])[0]
                dc, ci, cj = capsule_clearance(Pa, Pb)
                if dc > 0.30:                # nowhere near: nothing to learn
                    continue
                ba, pa = place_arm(names, BA, fleet[a], Qa[i])
                bb, pb = place_arm(names, BB, fleet[b], Qb[j])
                dm, na, nb = pair_distance(ba, pa, bb, pb, cap=0.40)
                recs.append(dict(a=int(a), b=int(b), ia=int(i), ib=int(j),
                                 d_caps=dc, d_mesh=dm, cap_a=int(ci),
                                 cap_b=int(cj), part_a=na, part_b=nb))
        dt = time.time() - t0
        dc = np.array([r["d_caps"] for r in recs])
        dm = np.array([r["d_mesh"] for r in recs])
        err = dc - dm                              # + = optimistic
        print(f"  A2  {len(recs)} near pairs in {dt:.1f} s; "
              f"capsule - mesh (mm): "
              f"min {1000 * err.min():+.1f}  med {1000 * np.median(err):+.1f}  "
              f"p99 {1000 * np.percentile(err, 99):+.1f}  "
              f"max {1000 * err.max():+.1f}")
        opt = err > 0.0005
        print(f"      OPTIMISTIC (capsule claims more clearance than the "
              f"meshes have): {opt.sum()}/{len(recs)} = {100 * opt.mean():.1f} %")
        # THE ONLY CLASS THAT CAN HURT ANYONE: a pair the conductor would
        # sign off (capsule clearance at or above the 80 mm margin) whose
        # real geometry is closer than that, or touching.
        clear = dc >= MARGIN
        false_clear = clear & (dm < MARGIN)
        false_touch = clear & (dm <= 1e-9)
        near = dc < 0.16
        print(f"      FALSE CLEAR (capsule >= {MARGIN} m, mesh < {MARGIN} m): "
              f"{int(false_clear.sum())} of {int(clear.sum())} signed-off pairs"
              f"   FALSE CLEAR TO CONTACT: {int(false_touch.sum())}")
        if near.any():
            print(f"      near band (capsule < 160 mm, {int(near.sum())} pairs)"
                  f": capsule - mesh med {1000 * np.median(err[near]):+.1f} mm, "
                  f"max {1000 * err[near].max():+.1f} mm")
        if opt.any():
            k = int(np.argmax(err))
            print(f"      worst: arms {recs[k]['a']}-{recs[k]['b']}, capsule "
                  f"{recs[k]['cap_a']} vs {recs[k]['cap_b']}, parts "
                  f"{recs[k]['part_a']} / {recs[k]['part_b']}: capsule "
                  f"{1000 * recs[k]['d_caps']:+.1f} mm, mesh "
                  f"{1000 * recs[k]['d_mesh']:+.1f} mm")
        # who is optimistic, by part
        bypart = {}
        for r, e in zip(recs, err):
            for p in (r["part_a"], r["part_b"]):
                if p:
                    bypart[p] = max(bypart.get(p, -9.9), float(e))
        res[f"h{h:.3f}"] = dict(
            excess_mm={k: 1000 * v for k, v in exc.items()},
            n_pairs=len(recs), seconds=dt,
            err_mm=dict(min=1000 * float(err.min()),
                        median=1000 * float(np.median(err)),
                        p50=1000 * float(np.percentile(err, 50)),
                        p90=1000 * float(np.percentile(err, 90)),
                        p99=1000 * float(np.percentile(err, 99)),
                        max=1000 * float(err.max())),
            optimistic_frac=float(opt.mean()),
            optimistic_n=int(opt.sum()),
            n_signed_off=int(clear.sum()),
            n_false_clear=int(false_clear.sum()),
            n_false_touch=int(false_touch.sum()),
            near_n=int(near.sum()),
            near_err_median_mm=(1000 * float(np.median(err[near]))
                                if near.any() else None),
            near_err_max_mm=(1000 * float(err[near].max())
                             if near.any() else None),
            worst_by_part_mm={k: 1000 * v for k, v in
                              sorted(bypart.items(), key=lambda t: -t[1])},
            samples=recs)

    # --- the number part C needs: per-capsule inflation, over ALL poses ----
    lay = dict(layout.LAYOUT_PROPOSED)
    fleet = layout.build_fleet(lay, q_park=layout.Q_PARK_PROPOSED)
    poses = load_pose_sets(fleet, n_random=60, seed=seed)
    Qall = np.vstack([poses[a] for a in sorted(poses)])
    k = rng.choice(len(Qall), min(400, len(Qall)), replace=False)
    # The base CONNECTOR AND CABLE are left out of the MOVER's inflation and
    # kept in the NEIGHBOUR's column (part B): they sit at base z < 0, which
    # for a hanging arm is above its own mount plate, where the plate and
    # boom boxes already gate.  Charging capsule 0 for them would inflate the
    # mover's own base by 0.17 m for geometry that is bolted into the ceiling.
    grow = capsule_inflation(parts, tools, Qall[k], rng=rng,
                             exclude=("link0_cable",))
    grow_all = capsule_inflation(parts, tools, Qall[k], rng=rng)
    res["capsule_inflation_m"] = grow.tolist()
    res["capsule_inflation_with_cable_m"] = grow_all.tolist()
    print("\n  A3  per-capsule inflation the real meshes demand (mm), over "
          f"{len(k)} poses:")
    for c, (i, j, r) in enumerate(CAPSULES_LAT):
        print(f"      capsule {c} ({i}->{j})  r {r:.3f} -> "
              f"{r + grow[c]:.4f}   (+{1000 * grow[c]:.1f} mm)"
              + (f"   [with base cable: +{1000 * grow_all[c]:.1f} mm]"
                 if grow_all[c] > grow[c] + 1e-4 else ""))
    rec["fidelity"] = res
    return res


# ===========================================================================
# 8. PART A (tool) — the L-capsules and the URDF cylinders vs raw CAD
# ===========================================================================
def _l_capsules():
    """The planner's tool model, in the hand frame -> (A, B, r)."""
    tcp = np.array([0.0, 0.0, D_HAND_TCP])
    tip = tcp + tool_offset(PEN_EXT, PEN_LAT_HOLDER)
    corner = tcp + np.array([PEN_LAT_HOLDER, 0.0, 0.0])
    return (np.array([tcp, corner]), np.array([corner, tip]),
            np.array([rig_final.BRACKET_R_LAT, rig_final.PEN_R_LAT]))


def _cyl_sdist(V, cyl):
    """(N,3) points vs the URDF's cylinder list -> (N,) outside distance."""
    best = np.full(len(V), np.inf)
    for _, T, (r, ln) in cyl:
        Ti = np.linalg.inv(T)
        P = V @ Ti[:3, :3].T + Ti[:3, 3]
        dr = np.hypot(P[:, 0], P[:, 1]) - r
        dz = np.abs(P[:, 2]) - 0.5 * ln
        best = np.minimum(best, np.where(
            (dr <= 0) & (dz <= 0), np.maximum(dr, dz),
            np.hypot(np.maximum(dr, 0), np.maximum(dz, 0))))
    return best


def part_tool(rec):
    """The pen holder: two schematic envelopes against the raw STL.

    Also the question the CAD delivery leaves open (`rig_final.PENHOLDER22`):
    the holder is drawn along the PLANNER's 45-degree ray because the
    fingertip cradle that would set its real clocking is not in the delivery.
    If the cradle turns out square to the hand the bore leans 23 degrees
    instead, the tool lands somewhere else entirely, and the planner's
    L-capsules are a model of a tool that is not there.  Both are measured.
    """
    print("\n=== PART A (tool): the pen holder envelopes vs raw CAD ===")
    if not CAD_DIR.exists():
        print("  raw CAD not present — skipped")
        return {}
    A, B, R = _l_capsules()
    cyl = rig_final.penholder22_collision(PEN_EXT, PEN_LAT_HOLDER, D_HAND_TCP)
    reach = float(np.linalg.norm(tool_offset(PEN_EXT, PEN_LAT_HOLDER)))
    out = {}
    cases = [("planner 45-deg ray (as built in the URDF)",
              PEN_EXT, PEN_LAT_HOLDER)]
    lean = rig_final.PENHOLDER22["post_clock"]        # 23.00 deg, measured
    cases.append((f"cradle square to the hand ({np.degrees(lean):.1f}-deg lean)",
                  reach * np.cos(lean), reach * np.sin(lean)))
    for label, pe, pl in cases:
        tools = tool_parts(pe, pl)
        rows = {}
        print(f"\n  -- {label}: bore -> tip at "
              f"({pl:.4f}, 0, {pe:.4f}) m from the TCP")
        for nm, m in tools.items():
            V = np.asarray(m.vertices, float)
            exc = (_pt_seg_dist(V, A, B) - R[None]).min(axis=1)
            cy = _cyl_sdist(V, cyl)
            rows[nm] = dict(n=int(len(V)),
                            l_capsule_max_mm=float(1000 * exc.max()),
                            l_capsule_n_out=int((exc > 1e-6).sum()),
                            urdf_cyl_max_mm=float(1000 * cy.max()),
                            urdf_cyl_n_out=int((cy > 1e-6).sum()))
            print(f"     {nm:8s} n={len(V):7d}   L-capsules "
                  f"{1000 * exc.max():+8.2f} mm "
                  f"({(exc > 1e-6).sum():6d} out)   URDF cylinders "
                  f"{1000 * cy.max():+8.2f} mm ({(cy > 1e-6).sum():6d} out)")
        body = np.concatenate([np.asarray(tools[n].vertices, float)
                               for n in ("housing", "cap")])
        exc = (_pt_seg_dist(body, A, B) - R[None]).min(axis=1)
        cy = _cyl_sdist(body, cyl)
        print(f"     housing+cap (no graphite): L-capsules "
              f"{1000 * exc.max():+.2f} mm   URDF cylinders "
              f"{1000 * cy.max():+.3f} mm")
        need = []
        own = (_pt_seg_dist(body, A, B) - R[None]).argmin(axis=1)
        for k in range(2):
            dk = _pt_seg_dist(body, A[k:k + 1], B[k:k + 1])[:, 0]
            need.append(float(dk[own == k].max()) if (own == k).any()
                        else float(R[k]))
        rows["_body"] = dict(l_capsule_max_mm=float(1000 * exc.max()),
                             urdf_cyl_max_mm=float(1000 * cy.max()),
                             needed_r=need)
        print(f"     the L-capsules would need r = {need[0]:.4f} / "
              f"{need[1]:.4f} (they carry {R[0]:.3f} / {R[1]:.3f})")
        # ...AND WITH THE PENCIL TAIL, which is a body of the assembled tool
        # and reaches back at the wrist (docs/SYSTEM_MODEL.md 7c).  Reported
        # separately so the housing's own escape stays readable next to it.
        if "tail" in tools:
            wt = np.concatenate([body, np.asarray(tools["tail"].vertices,
                                                  float)])
            ex2 = (_pt_seg_dist(wt, A, B) - R[None]).min(axis=1)
            own2 = (_pt_seg_dist(wt, A, B) - R[None]).argmin(axis=1)
            need2 = []
            for k in range(2):
                dk = _pt_seg_dist(wt, A[k:k + 1], B[k:k + 1])[:, 0]
                need2.append(float(dk[own2 == k].max()) if (own2 == k).any()
                             else float(R[k]))
            rows["_body_with_tail"] = dict(
                l_capsule_max_mm=float(1000 * ex2.max()), needed_r=need2)
            print(f"     + the pencil tail: L-capsules "
                  f"{1000 * ex2.max():+.2f} mm, they would need r = "
                  f"{need2[0]:.4f} / {need2[1]:.4f}")
        out[label] = rows
    rec["tool"] = out
    return out


# ===========================================================================
# 9. PART C — the decisive re-check
# ===========================================================================
def _clear_caps(A, B, caps):
    """(N,7,3) chain capsules vs a list of (A,B,r) obstacle capsules."""
    out = np.full(len(A), np.inf)
    for (oa, ob, orr) in caps:
        d = seg_seg_dist(A[:, :, None, :], B[:, :, None, :],
                         oa[None, None], ob[None, None])
        out = np.minimum(out, (d - CAP_R[None, :, None]
                               - orr[None, None, :]).min(axis=(1, 2)))
    return out


def _column_caps(xy, h, stack=None):
    """A neighbour's base column as a list of (A,B,r) capsules, world frame.

    The PUBLISHED model — one capsule of radius LINK_R over base z [0, D1],
    which for a hanging arm is world z [h - D1, h].
    """
    x, y = float(xy[0]), float(xy[1])
    return (np.array([[x, y, h]]), np.array([[x, y, h - D1]]),
            np.array([LINK_R]))


def _column_cyls(xy, h, stack):
    """The CORRECTED column as flat-ended CYLINDER bands -> (xy, r, z0, z1).

    Capsules are the wrong primitive for a stack: a fat band's spherical cap
    bulges its own radius into the neighbouring thin band and the union comes
    out far more conservative than the geometry it is standing in for.  A
    solid cylinder band is convex, so `seg_cyl_clearance` can still measure
    it exactly.  Bands at base z < 0 are DROPPED: for a hanging arm that is
    above the mount plate, where the plate and boom boxes already gate.
    """
    x, y = float(xy[0]), float(xy[1])
    out = []
    for band in stack:
        z0, z1, r = band["z0"], band["z1"], band["r"]
        z0, z1 = max(z0, 0.0), max(z1, 0.0)
        if z1 - z0 < 1e-9:
            continue
        out.append((x, y, r, h - z1, h - z0))     # world z, low to high
    return np.array(out, float)


def _seg_pt_2d(A, B, x, y):
    """(...,3),(...,3) segments vs (M,) axis positions -> (...,M) XY distance."""
    ax, ay = A[..., 0, None], A[..., 1, None]
    dx, dy = B[..., 0, None] - ax, B[..., 1, None] - ay
    L2 = np.maximum(dx * dx + dy * dy, 1e-18)
    t = np.clip(((x - ax) * dx + (y - ay) * dy) / L2, 0.0, 1.0)
    return np.hypot(ax + t * dx - x, ay + t * dy - y)


def seg_cyl_clearance(A, B, cyl, cap=1.0, iters=34):
    """(...,3),(...,3) segments vs (M,5) vertical cylinder bands -> (...,M).

    A solid cylinder band is CONVEX, so t -> dist(A + t(B-A), band) is convex
    on [0, 1] and ternary section converges to the true minimum — the same
    argument `rig_final.segment_box_clearance` runs for boxes.  34 sections
    shrink the bracket by 1.5e-6 of the segment.

    Capsules, not cylinders, would be the obvious primitive for a stack and
    are the wrong one: a fat band's spherical cap bulges its own radius into
    the thin band above it, and the union of the stack comes out markedly
    more conservative than the body it stands for.

    Values at or above `cap` are clipped, and the cheap lower bound
    sqrt(max(d_xy - r, 0)^2 + max(dz, 0)^2) retires everything that cannot
    beat it before any iteration is paid for.
    """
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    cx, cy, cr, cz0, cz1 = (np.ascontiguousarray(cyl[:, k]) for k in range(5))
    d2 = _seg_pt_2d(A, B, cx, cy) - cr
    zlo = np.minimum(A[..., 2, None], B[..., 2, None])
    zhi = np.maximum(A[..., 2, None], B[..., 2, None])
    dz0 = np.maximum(cz0 - zhi, zlo - cz1)
    lb = np.hypot(np.maximum(d2, 0.0), np.maximum(dz0, 0.0))
    out = np.full(lb.shape, cap)
    live = lb < cap
    if not live.any():
        return out
    idx = np.nonzero(live)
    Ai = A[idx[:-1]]                    # (K,3)
    Bi = B[idx[:-1]]
    m = idx[-1]
    px, py, pr, pz0, pz1 = cx[m], cy[m], cr[m], cz0[m], cz1[m]
    D = Bi - Ai

    def d_at(t):
        P = Ai + t[:, None] * D
        dr = np.hypot(P[:, 0] - px, P[:, 1] - py) - pr
        dz = np.maximum(pz0 - P[:, 2], P[:, 2] - pz1)
        inside = (dr <= 0) & (dz <= 0)
        return np.where(inside, np.maximum(dr, dz),
                        np.hypot(np.maximum(dr, 0.0), np.maximum(dz, 0.0)))

    lo = np.zeros(len(Ai))
    hi = np.ones(len(Ai))
    for _ in range(iters):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        take = d_at(m1) < d_at(m2)
        hi = np.where(take, m2, hi)
        lo = np.where(take, lo, m1)
    out[idx] = np.minimum(d_at(0.5 * (lo + hi)), cap)
    return out


def _cell_job(job):
    """One arm, one layout, several occupancy scenarios -> a coverage mask.

    A faithful re-implementation of `scripts/layout_rescore.py`'s gated fiber
    (same yaws, same q7 grid, same gate order, same own-boom proxy, same
    steel), so the published numbers are reproduced as a CONTROL before the
    corrected column is swapped in.  Rebuilt here rather than imported for
    the reason that file gives: `spec.static_obstacles()` has carried the
    neighbours' columns since 14b01cd and would double-count them.
    """
    name, lay, aid, scen, stack, grow = job
    h = float(lay["h"])
    grow = np.zeros(len(CAP_R)) if grow is None else np.asarray(grow, float)
    capr = CAP_R + grow
    fleet = layout.build_fleet(lay)
    spec = fleet[aid]
    Twb = np.asarray(spec.T_world_base(), float)
    Twb_inv = np.linalg.inv(Twb)
    steel = [b for o, s in fleet.items() if o != aid
             for b in mounts.arm_mount_boxes(s.mount, s.xy, s.yaw, h,
                                             tag=f"mount:{o}")]
    off = tool_offset(PEN_EXT, PEN_LAT_HOLDER)
    W, H = SHEET_FINAL6
    XS = np.arange(0.0, W + 1e-9, 0.02)
    YS = np.arange(0.0, H + 1e-9, 0.02)
    RMAX = 1.05 + PEN_LAT_HOLDER
    sets = {}
    for nm in scen:
        if nm == "air":
            sets[nm] = []
        elif nm == "mounts":
            sets[nm] = [_column_caps(fleet[b].xy, h) for b in fleet if b != aid]
        elif nm in ("mesh", "mesh_col"):
            sets[nm] = np.concatenate([_column_cyls(fleet[b].xy, h, stack)
                                       for b in fleet if b != aid])
    bx, by = spec.xy
    cells, ix, iy = [], [], []
    for j, y in enumerate(YS):
        for i, x in enumerate(XS):
            if (x - bx) ** 2 + (y - by) ** 2 <= RMAX ** 2:
                cells.append((x, y))
                ix.append(i)
                iy.append(j)
    ok = {nm: np.zeros(len(cells), bool) for nm in scen}
    YAWS = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    for k, (x, y) in enumerate(cells):
        sols = []
        tip = np.array([x, y, 0.0])
        for yaw in YAWS:
            R = rotz(yaw) @ rotx(np.pi)
            T_w = np.eye(4)
            T_w[:3, :3] = R
            T_w[:3, 3] = tip - R @ off
            Tb = Twb_inv @ T_w
            for q7 in ik.Q7_GRID:
                sols.extend(ik.solve(Tb, q7, spec.q_seed))
        if not sols:
            continue
        Q = np.asarray(sols, float).reshape(-1, 7)
        jm = np.array([joint_margin(q) for q in Q])
        keep = jm >= GATE_MARGIN
        Q, jm = Q[keep], jm[keep]
        if not len(Q):
            continue
        T, P = fk_many(Q)
        tl = tool_points_many(T, PEN_EXT, PEN_LAT_HOLDER)
        P11 = np.concatenate([P] + [t[:, None, :] for t in tl], axis=1)
        Pw = P11 @ Twb[:3, :3].T + Twb[:3, 3]
        keep = Pw[:, 1:9, 2].min(1) >= 0.02
        rb = np.hypot(P[:, :, 0], P[:, :, 1])
        keep &= ~np.any((P[:, :, 2] < -0.02) & (rb < 0.12), axis=1)
        Q, jm, Pw = Q[keep], jm[keep], Pw[keep]
        if not len(Q):
            continue
        if steel:
            keep = (rig_final.chain_static_clearance(Pw, steel)
                    >= rig_final.STATIC_MARGIN)
            Q, jm, Pw = Q[keep], jm[keep], Pw[keep]
            if not len(Q):
                continue
        A, B = Pw[:, CAP_I, :], Pw[:, CAP_J, :]
        order = np.argsort(-jm)
        sig = {}
        for nm in scen:
            obs = sets[nm]
            if nm in ("mesh", "mesh_col"):
                rr = capr if nm == "mesh" else CAP_R
                # the clip has to sit STRICTLY above the decision threshold,
                # or a clipped far pair lands on MARGIN and loses to the
                # round-off (0.17 - 0.09 = 0.0799999... < 0.08 refuses the
                # whole canvas, which is exactly what the first run did).
                gap = (seg_cyl_clearance(A, B, obs,
                                         cap=MARGIN + float(rr.max()) + 0.05)
                       - rr[None, :, None]).min(axis=(1, 2))
            elif len(obs):
                gap = _clear_caps(A, B, obs)
            else:
                gap = np.full(len(Q), np.inf)
            for t in order:
                if gap[t] < MARGIN:
                    continue
                if t not in sig:
                    sig[t] = metrics.sigma_min(metrics.tip_jacobian(
                        Q[t], pen_ext=PEN_EXT, pen_lat=PEN_LAT_HOLDER))
                if sig[t] >= GATE_SIGMA:
                    ok[nm][k] = True
                    break
    return name, aid, np.array(ix), np.array(iy), {n: ok[n] for n in scen}


def part_height(rec, jobs=8, heights=(0.850, 0.922), stack=None, grow=None):
    """The decisive re-check: does the height verdict survive the geometry?

    Three scenarios per height.  `air` and `mounts` are `layout_rescore.py`'s
    own, rebuilt here, and reproducing its published 99.98 / 98.10 (h = 0.85)
    and 99.86 / 99.86 (h = 0.922) is the CONTROL that says this file is
    measuring the same thing.  `mesh` swaps BOTH sides of the comparison for
    what the geometry actually is: the neighbour column becomes the measured
    cylinder stack of part B, and the mover's own seven capsules are grown by
    the per-capsule inflation part A measured, so neither side is a schematic
    any more.
    """
    import multiprocessing as mp
    print("\n=== PART C: the height verdict, re-checked ===")
    if stack is None:
        stack = rec.get("column", {}).get("corrected_stack")
    if stack is None:
        stack = part_column(rec)["corrected_stack"]
    if grow is None:
        grow = rec.get("fidelity", {}).get("capsule_inflation_m")
    grow = np.zeros(len(CAP_R)) if grow is None else np.asarray(grow, float)
    print("  mover capsule radii: "
          + ", ".join(f"{a:.3f}->{a + g:.3f}" for a, g in zip(CAP_R, grow)))
    # air        nothing but the arm                     (the published objective)
    # mounts     published column, published capsules    (the published penalty)
    # mesh_col   MEASURED column, published capsules     (what the column costs)
    # mesh       MEASURED column, GROWN capsules         (both sides mesh-true)
    scen = ("air", "mounts", "mesh_col", "mesh")
    W, H = SHEET_FINAL6
    nx = len(np.arange(0.0, W + 1e-9, 0.02))
    ny = len(np.arange(0.0, H + 1e-9, 0.02))
    jobs_list = []
    for h in heights:
        lay = dict(layout.LAYOUT_PROPOSED)
        lay["h"] = float(h)
        for aid in sorted(layout.FLEET_PROPOSED):
            jobs_list.append((f"h{h:.3f}", lay, aid, scen, stack,
                              grow.tolist()))
    t0 = time.time()
    if jobs > 1:
        with mp.Pool(jobs) as pool:
            res = pool.map(_cell_job, jobs_list)
    else:
        res = [_cell_job(j) for j in jobs_list]
    print(f"  {len(jobs_list)} arm-sweeps in {time.time() - t0:.1f} s")
    cnt = {}
    for name, aid, ix, iy, ok in res:
        for nm, m in ok.items():
            g = cnt.setdefault((name, nm), np.zeros((ny, nx), np.int16))
            g[iy[m], ix[m]] += 1
    out = {}
    for h in heights:
        name = f"h{h:.3f}"
        row = {}
        for nm in scen:
            g = cnt[(name, nm)]
            row[f"union_{nm}"] = float(100 * (g > 0).mean())
            row[f"ge2_{nm}"] = float(100 * (g >= 2).mean())
            row[f"cells_{nm}"] = int((g > 0).sum())
        row["n_cells"] = int(nx * ny)
        out[name] = row
        print(f"  h = {h:.3f}   union %: air {row['union_air']:.2f}   "
              f"published column {row['union_mounts']:.2f}   "
              f"MEASURED column {row['union_mesh_col']:.2f}   "
              f"+ grown capsules {row['union_mesh']:.2f}")
    rec["height"] = dict(rows=out, scenarios=list(scen), stack=stack,
                         capsule_inflation_m=grow.tolist(),
                         grid=0.02, margin=MARGIN)
    np.savez_compressed(OUT / "collision_audit_height.npz",
                        **{f"{a}__{b}": cnt[(a, b)] for (a, b) in cnt})
    return out


def part_pairs(rec, n=900, seed=17):
    """The 71-vs-31-style minimum clearances, capsule vs mesh."""
    print("\n=== PART C (pairs): worst inter-arm clearance, capsule vs mesh ===")
    rng = np.random.default_rng(seed)
    parts, tools = arm_parts("union"), tool_parts()
    names, BA, BB = body_sets(parts, tools)
    res = {}
    for h in (0.850, 0.922):
        lay = dict(layout.LAYOUT_PROPOSED)
        lay["h"] = h
        fleet = layout.build_fleet(lay, q_park=layout.Q_PARK_PROPOSED)
        poses = load_pose_sets(fleet, n_random=0, seed=seed)
        ids = sorted(fleet)
        rows = []
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                Qa, Qb = poses[a], poses[b]
                ka = rng.integers(0, len(Qa), n)
                kb = rng.integers(0, len(Qb), n)
                wc, wm, wrec = np.inf, np.inf, None
                for p, qq in zip(ka, kb):
                    Pa = world_chain(Qa[p][None], fleet[a])[0]
                    Pb = world_chain(Qb[qq][None], fleet[b])[0]
                    dc, ci, cj = capsule_clearance(Pa, Pb)
                    if dc > 0.30:
                        continue
                    ba, pa = place_arm(names, BA, fleet[a], Qa[p])
                    bb, pb = place_arm(names, BB, fleet[b], Qb[qq])
                    dm, na, nb = pair_distance(ba, pa, bb, pb, cap=0.40)
                    if dm < wm:
                        wm, wrec = dm, (na, nb)
                    wc = min(wc, dc)
                rows.append(dict(a=int(a), b=int(b),
                                 worst_caps_mm=1000 * float(wc),
                                 worst_mesh_mm=1000 * float(wm),
                                 mesh_parts=wrec,
                                 base_dist=float(np.hypot(
                                     *(np.array(fleet[a].xy)
                                       - np.array(fleet[b].xy))))))
        rows.sort(key=lambda r: r["worst_mesh_mm"])
        print(f"  h = {h:.3f} (certified drawing poses + parks, {n} pairs each)")
        for r in rows[:6]:
            print(f"    {r['a']:>3}-{r['b']:<3} base {r['base_dist']:.2f} m   "
                  f"capsule worst {r['worst_caps_mm']:+8.1f} mm   "
                  f"mesh worst {r['worst_mesh_mm']:+8.1f} mm   "
                  f"{r['mesh_parts']}")
        res[f"h{h:.3f}"] = rows
    rec["pairs"] = res
    return res


# ===========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all",
                    choices=("all", "column", "fidelity", "tool", "height",
                             "pairs", "none"))
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--poses", type=int, default=220)
    ap.add_argument("--pair-samples", type=int, default=900)
    ap.add_argument("--heights", default="0.850,0.875,0.900,0.922,0.950")
    ap.add_argument("--out", default=str(OUT / "collision_audit.json"))
    a = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    p = Path(a.out)
    rec = json.loads(p.read_text()) if p.exists() else {}
    rec["meta"] = dict(when=time.strftime("%Y-%m-%d %H:%M:%S"),
                       tool=os.environ.get("ARIS_TOOL"),
                       collision_dir=str(_coll_dir()),
                       visual_dir=str(VIS_DIR),
                       cad_dir=str(CAD_DIR), margin=MARGIN,
                       pen_ext=PEN_EXT, pen_lat=PEN_LAT_HOLDER)

    def save():
        p.write_text(json.dumps(rec, indent=1, default=float))
        print(f"[saved] {p}")

    if a.validate or a.part in ("all",):
        validate(rec)
        validate_drake(rec)
        save()
    if a.part in ("all", "column"):
        part_column(rec)
        save()
    if a.part in ("all", "tool"):
        part_tool(rec)
        save()
    if a.part in ("all", "fidelity"):
        part_fidelity(rec, n_pose=a.poses)
        save()
    if a.part in ("all", "pairs"):
        part_pairs(rec, n=a.pair_samples)
        save()
    if a.part in ("all", "height"):
        part_height(rec, jobs=a.jobs,
                    heights=tuple(float(x) for x in a.heights.split(",")))
        save()
    save()


if __name__ == "__main__":
    main()
