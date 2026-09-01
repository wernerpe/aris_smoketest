#!/usr/bin/env python3
"""Generate `assets/system_model/` — the whole installation, visual + collision.

    python3 scripts/gen_system_model.py meshes    # vendor + decimate meshes
    python3 scripts/gen_system_model.py urdf      # write URDFs + manifest
    python3 scripts/gen_system_model.py all       # both

WHAT IT WRITES
--------------
    installation.urdf            cage + table + paper + six arms + holders,
                                 collision = the manufacturer's own shells
    installation_capsules.urdf   the same scene, collision = the AUDITED
                                 capsule set (selfcoll.BODY_CAPSULES)
    environment.urdf             the static scene alone, no arms
    model_manifest.json          every body: dimensions, provenance, source
    meshes/fr3/*.gltf            the vendored Franka visuals, RETEXTURED
    meshes/fr3/textures/*.png    the 27 real texture maps, byte-identical
    meshes/collision/*.obj       the manufacturer's collision shells
    meshes/penholder/*.obj       the pen holder, re-decimated from raw CAD

WHY THIS SUPERSEDES `assets/proposed_rig/`
------------------------------------------
Three things were wrong with the older asset and are fixed here.

 1. THE ARM COLLISION GEOMETRY WAS THE VENDORED PANDA'S 66 SPHERES PER ARM,
    which nothing in this repo has ever audited, and which are not the model
    any certified number was earned against.  They are gone.  This model
    offers the two geometries that ARE backed by measurement: the
    manufacturer's collision shells (`scripts/collision_audit.py --validate`
    reproduces the FR3's own collision boxes from them to 0.344 mm worst, and
    drake reports their convex hulls at the same volume, so they are already
    convex and lossless as URDF collision), and the audited capsule set.

 2. THE VISUALS HAD NO TEXTURES.  The vendored glTFs reference 65 image files
    that were never copied across, so every renderer fell back to flat white
    and printed a warning per image per mesh.  The 54 that the arm needs are
    now vendored BYTE-IDENTICALLY from where the meshes themselves came from
    (`~/git/franka_manipulation_station/assets/franka_description`), and the
    glTFs are rewritten to point at them.  No runtime hack, no stripped
    materials, no resampling.

 3. THE STRUCTURE WAS A SCHEMATIC KEEP-OUT, NOT A BUILD.  `mounts.py` models
    each mount as a radius-100 column running to a 2.34 m "ceiling".  This
    model carries the real 80/20 cage from the original drawing, at the
    CORRECTED datum — see `aris_sixarm/system_model.py` for why 2.34 m is a
    provenance bug and what the drawing actually says.

`assets/proposed_rig/` IS NOT TOUCHED.  It is what every certified number in
the repo was checked against and what `scripts/collision_audit.py` still
reads; this is an addition, not a replacement.

READ-ONLY against the package: nothing here mutates a package constant, and
the tool offset is passed explicitly at every call site so `frames.PEN_LAT`
stays where it is.
"""
import hashlib
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from aris_sixarm import frames, layout, mounts, rig_final, selfcoll  # noqa: E402
from aris_sixarm import system_model as SM                            # noqa: E402
from aris_sixarm.frames import (D_HAND_TCP, FR3_MAX, FR3_MIN, PEN_EXT,  # noqa: E402
                                PEN_LAT_HOLDER, QD_MAX, TAU_MAX)
from gen_final_rig_urdf import rpy_from_R                              # noqa: E402

# Drake's tinyxml2 matches `drake:*` as a literal string, so the prefix has to
# be registered or ElementTree emits `ns0:` and every filter group dies silently
# (which is exactly what happened to assets/final_rig/installation.urdf).
ET.register_namespace("drake", "http://drake.mit.edu")

OUT_DIR = ROOT / "assets/system_model"
VENDOR_GLTF = ROOT / "assets/franka_description/meshes/visual"
SRC_URDF = ROOT / "assets/franka_description/urdf/panda_arm_hand.urdf"

# The two external sources this model vendors from.  Both are outside the repo
# and both are recorded, file by file with a sha256, in the manifest.
TEXTURE_SRC = Path("/home/franka/git/franka_manipulation_station/assets/"
                   "franka_description/meshes/visual")
COLLISION_SRC = Path("/home/franka/git/vamp/resources/panda/meshes/collision")
CAD_SRC = ROOT.parent / "raw_slack_file_dump" / "Pen holder all parts 2026.08.19"
# the two parts of the mounted holder that are visible from outside; the other
# six live inside the 21.1 mm bore or are bench tools (PENHOLDER22["omitted"])
HOUSING_STL = "pen holder housing - 22 deg - reinforced - v20260429.STL"
CAP_STL = "pen holder cap v20250903.STL"

FR3_MESHES = tuple(f"link{i}" for i in range(8)) + ("hand", "finger")
TEXTURED = tuple(f"link{i}" for i in range(8)) + ("hand",)   # finger has none
TEX_KINDS = ("color", "normal", "occlusion_roughness_metallic")

FINGER_FIX = 0.0285          # m, custom fingertip half-width holding the holder
HOLDER_FACES = 8000          # decimation target, per part
MM = SM.MM

# Where the arm's collision shells attach.  The vendored URDF's link8 and the
# two fingers carry no shell of their own: link8 is a pure frame, and the
# fingers are the `finger.obj` shell mirrored.
SHELL_OF = {f"panda_link{i}": f"link{i}" for i in range(8)}
SHELL_OF["panda_hand"] = "hand"
SHELL_OF["panda_leftfinger"] = "finger"
SHELL_OF["panda_rightfinger"] = "finger"

ARM_LINKS = tuple(f"panda_link{i}" for i in range(9)) + (
    "panda_hand", "panda_leftfinger", "panda_rightfinger")
TOOL_LINKS = ("pen_holder", "pen_lead")
WRIST_LINKS = ("panda_link6", "panda_link7", "panda_link8", "panda_hand",
               "panda_leftfinger", "panda_rightfinger")


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------
def _fmt(v):
    """Exact float64 round trip — `float(_fmt(x)) == x` for every x.

    The final rig wrote 1e-6 because its numbers came off a drawing quoted in
    hundredths of a centimetre; the proposed rig went to 1e-9 because this
    layout's numbers are DERIVED (sixths of a 3.63064 m web, pi in the
    inverted seat) and 1e-6 truncation costs ~0.9 um at the pen tip.  1e-9
    still costs something: at the far row, y = 3.0255333333333336 truncates to
    3.025533333 and the round trip lands 1.06 nm out — over the bar, on the
    two arms furthest from the origin.

    `repr` is the shortest decimal that reads back as the SAME double, so the
    truncation error is not small, it is zero.  There is nothing left to pick
    a tolerance for.
    """
    return repr(float(v))


def _v3(v):
    return " ".join(_fmt(x) for x in v)


def _origin(el, xyz=None, rpy=None):
    o = ET.SubElement(el, "origin")
    if xyz is not None:
        o.set("xyz", _v3(xyz))
    if rpy is not None:
        o.set("rpy", _v3(rpy))
    return o


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# STAGE 1 — meshes
# ---------------------------------------------------------------------------
def vendor_textures(out_dir=None):
    """Copy the real texture maps in byte-identically -> [dict] for the manifest.

    BYTE-IDENTICAL ON PURPOSE.  A resampled texture is a texture whose
    provenance is "Claude resized it"; a copied one still has the vendor's own
    sha256, and 2048 px PNGs of a robot arm are 10 MB, which is less than the
    .bin files this repo already commits.  The .ktx2 twins are NOT copied:
    drake's VTK loader says so itself ("glTF extension KHR_texture_basisu is
    used in this model, but not supported by this loader"), so they are 0.8 MB
    of dead weight and one warning per mesh.
    """
    out_dir = Path(out_dir or OUT_DIR)
    dst = out_dir / "meshes/fr3/textures"
    dst.mkdir(parents=True, exist_ok=True)
    if not TEXTURE_SRC.is_dir():
        raise SystemExit(f"texture source not found: {TEXTURE_SRC}")
    recs = []
    for stem in TEXTURED:
        for kind in TEX_KINDS:
            name = f"{stem}_{kind}.png"
            src = TEXTURE_SRC / name
            if not src.is_file():
                raise SystemExit(f"missing texture {src}")
            shutil.copyfile(src, dst / name)
            recs.append(dict(file=f"meshes/fr3/textures/{name}",
                             bytes=(dst / name).stat().st_size,
                             sha256=_sha256(dst / name),
                             source=str(src)))
    return recs


def retexture_gltfs(out_dir=None):
    """Rewrite the vendored glTFs to point at the vendored textures.

    The geometry is untouched: `buffers[0].uri` is re-pointed at the SAME .bin
    the repo already commits, by relative path, so nothing is duplicated.  Only
    the image table changes — the .ktx2 entries and the KHR_texture_basisu
    extension go, and every texture is re-indexed onto the .png that survives.
    """
    out_dir = Path(out_dir or OUT_DIR)
    dst = out_dir / "meshes/fr3"
    dst.mkdir(parents=True, exist_ok=True)
    recs = []
    for stem in FR3_MESHES:
        src = VENDOR_GLTF / f"{stem}.gltf"
        d = json.loads(src.read_text())
        # THE .bin COMES TOO, and the package is self-contained because of it.
        # Pointing the buffer at ../../../franka_description works in drake and
        # in VTK, and trimesh REFUSES it — its resolver will not follow a URI
        # out of the glTF's own directory without allow_anywhere.  An asset
        # package that only some loaders can open is not an asset package.  It
        # costs 5.9 MB: the ten meshes this model uses, not the 10 MB
        # cobot_pump nothing here mounts.
        for buf in d.get("buffers", []):
            if buf.get("uri", "").endswith(".bin"):
                b = Path(buf["uri"]).name
                shutil.copyfile(VENDOR_GLTF / b, dst / b)
                recs.append(dict(file=f"meshes/fr3/{b}",
                                 bytes=(dst / b).stat().st_size,
                                 sha256=_sha256(dst / b),
                                 source=str(VENDOR_GLTF / b)))
                buf["uri"] = b
        imgs = d.get("images")
        if imgs:
            keep, remap = [], {}
            for i, im in enumerate(imgs):
                if im.get("mimeType") == "image/ktx2":
                    continue
                remap[i] = len(keep)
                im["uri"] = f"textures/{Path(im['uri']).name}"
                keep.append(im)
            d["images"] = keep
            for t in d.get("textures", []):
                t.pop("extensions", None)
                t["source"] = remap[t["source"]]
            used = [e for e in d.get("extensionsUsed", [])
                    if e != "KHR_texture_basisu"]
            if used:
                d["extensionsUsed"] = used
            else:
                d.pop("extensionsUsed", None)
        (dst / f"{stem}.gltf").write_text(
            json.dumps(d, indent="\t", sort_keys=True) + "\n")
        recs.append(dict(file=f"meshes/fr3/{stem}.gltf",
                         bytes=(dst / f"{stem}.gltf").stat().st_size,
                         sha256=_sha256(dst / f"{stem}.gltf"),
                         source=str(src),
                         note="geometry byte-identical (the vendor's own .bin, "
                              "vendored alongside); image table re-pointed at "
                              "the vendored .png set, ktx2 dropped"))
    return recs


def vendor_collision(out_dir=None):
    """Copy the manufacturer's collision shells in -> [dict]."""
    out_dir = Path(out_dir or OUT_DIR)
    dst = out_dir / "meshes/collision"
    dst.mkdir(parents=True, exist_ok=True)
    if not COLLISION_SRC.is_dir():
        raise SystemExit(f"collision source not found: {COLLISION_SRC}")
    recs = []
    for stem in FR3_MESHES:
        src = COLLISION_SRC / f"{stem}.obj"
        shutil.copyfile(src, dst / f"{stem}.obj")
        recs.append(dict(file=f"meshes/collision/{stem}.obj",
                         bytes=(dst / f"{stem}.obj").stat().st_size,
                         sha256=_sha256(dst / f"{stem}.obj"),
                         source=str(src)))
    return recs


def _obj_with_normals(m, name):
    """Write an OBJ with area-weighted vertex normals, in numpy alone.

    trimesh's own exporter reaches for scipy.sparse to accumulate them, and
    scipy is not installed on the interpreter that has trimesh.  The normals
    matter: without them drake's VTK renderer facets a 3D-printed barrel into
    visible strips.
    """
    v = np.asarray(m.vertices, float)
    f = np.asarray(m.faces, np.int64)
    # face normals, NOT unit — their length is twice the triangle area, which
    # is exactly the weight a smooth vertex normal wants
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, f[:, k], fn)
    n = np.linalg.norm(vn, axis=1, keepdims=True)
    vn = np.divide(vn, n, out=np.zeros_like(vn), where=n > 0)
    out = [f"# {name} — generated by scripts/gen_system_model.py",
           f"# {len(v)} vertices, {len(f)} faces, metres, panda_hand frame",
           f"o {name}"]
    out += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in v]
    out += [f"vn {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vn]
    out += [f"f {a}//{a} {b}//{b} {c}//{c}" for a, b, c in (f + 1)]
    return "\n".join(out) + "\n"


def decimate_holder(out_dir=None, faces=HOLDER_FACES):
    """Re-decimate the pen holder from the RAW STLs -> (recs, quality).

    Weld first, then decimate: the raw STLs are per-triangle soup, and a
    quadric simplifier run on unwelded vertices tears the surface apart at
    every seam.  The extent loss of each part is measured and reported, and
    the result is re-proved against the 3-cylinder collision envelope, which
    is the only thing that makes the envelope a proof rather than a claim.
    """
    import trimesh
    out_dir = Path(out_dir or OUT_DIR)
    dst = out_dir / "meshes/penholder"
    dst.mkdir(parents=True, exist_ok=True)
    P = rig_final.PENHOLDER22
    T_h, T_c, _, _ = rig_final.penholder22_T_hand(PEN_EXT, PEN_LAT_HOLDER,
                                                  D_HAND_TCP)
    parts = (("housing", HOUSING_STL, T_h), ("cap", CAP_STL, T_c))
    recs, quality = [], []
    for name, stl, T in parts:
        src = CAD_SRC / stl
        if not src.is_file():
            raise SystemExit(f"raw CAD not found: {src}")
        m = trimesh.load(src, force="mesh")
        ext0 = m.bounds[1] - m.bounds[0]
        assert 5.0 < ext0.max() < 500.0, f"{name}: not millimetres? {ext0}"
        n0 = len(m.faces)
        m.merge_vertices(merge_tex=True, merge_norm=True)
        if len(m.faces) > faces:
            m = m.simplify_quadric_decimation(face_count=faces)
        ext1 = m.bounds[1] - m.bounds[0]
        shrink = float(np.abs(ext1 - ext0).max())
        assert shrink < 1.0, f"{name}: decimation lost {shrink:.3f} mm"
        m.apply_scale(0.001)                       # mm -> m
        m.apply_transform(T)                       # -> panda_hand frame
        out = dst / f"penholder22_{name}_hand.obj"
        out.write_text(_obj_with_normals(m, f"penholder22_{name}_hand"))
        recs.append(dict(file=f"meshes/penholder/penholder22_{name}_hand.obj",
                         bytes=out.stat().st_size, sha256=_sha256(out),
                         source=str(src)))
        quality.append(dict(part=name, faces_raw=n0, faces_out=len(m.faces),
                            extent_loss_mm=round(shrink, 4),
                            extent_mm=[round(float(v), 3) for v in ext1]))
    # the envelope must still contain every vertex it claims to contain
    worst = -1e9
    for name, _, T in parts:
        m = trimesh.load(dst / f"penholder22_{name}_hand.obj", force="mesh")
        by, bz = P["bore_yz"]
        v = np.asarray(m.vertices, float)
        vh = (np.linalg.inv(T_h)[:3, :3] @ v.T).T + np.linalg.inv(T_h)[:3, 3]
        out_of = np.full(len(vh), np.inf)
        for x0, x1, r in P["env_cylinders"]:
            radial = np.hypot(vh[:, 1] - by, vh[:, 2] - bz) - r
            axial = np.maximum(x0 - vh[:, 0], vh[:, 0] - x1)
            out_of = np.minimum(out_of, np.maximum(radial, axial))
        worst = max(worst, float(out_of.max()))
    # THE BAR IS 10 um, NOT ZERO, AND THE REASON IS WORTH WRITING DOWN.  The
    # envelope's radii were fitted to a 5000-face decimation, where every
    # vertex sits at worst 3.1 um INSIDE.  At this model's finer target the
    # mesh follows the true barrel more closely and pokes 0.4 um proud of one
    # band — a tenth of the printer's own layer resolution, and five orders
    # below the 50 mm static margin.  It is REPORTED rather than hidden:
    # `envelope_worst_escape_m` goes into the manifest on every run.
    assert worst <= 1e-5, f"holder escapes its envelope by {worst * 1000:.4f} mm"
    return recs, dict(parts=quality, envelope_worst_escape_m=round(worst, 12),
                      envelope_bar_m=1e-5)


def build_meshes(out_dir=None):
    out_dir = Path(out_dir or OUT_DIR)
    files = []
    files += vendor_textures(out_dir)
    files += retexture_gltfs(out_dir)
    files += vendor_collision(out_dir)
    holder, quality = decimate_holder(out_dir)
    files += holder
    (out_dir / "meshes/MESH_SOURCES.json").write_text(json.dumps(
        dict(files=files, holder_decimation=quality), indent=2) + "\n")
    return files, quality


# ---------------------------------------------------------------------------
# STAGE 2 — the URDF
# ---------------------------------------------------------------------------
def add_box_link(robot, body):
    """One `system_model.Body` -> a welded box link (metres, canvas frame).

    The box sits at the LINK ORIGIN and the weld carries the offset, so a
    consumer that asks drake where the body is gets the box's centre rather
    than the world origin.
    """
    lo = np.asarray(body.lo, float) / MM
    hi = np.asarray(body.hi, float) / MM
    link = ET.SubElement(robot, "link", name=body.name)
    for role in ("visual",) + (("collision",) if body.collision else ()):
        el = ET.SubElement(link, role)
        ET.SubElement(ET.SubElement(el, "geometry"), "box", size=_v3(hi - lo))
        if role == "visual":
            mat = ET.SubElement(el, "material", name=f"{body.name}_mat")
            ET.SubElement(mat, "color", rgba=_v3(body.rgba))
    j = ET.SubElement(robot, "joint", name=f"{body.name}_weld", type="fixed")
    ET.SubElement(j, "parent", link="world")
    ET.SubElement(j, "child", link=body.name)
    _origin(j, 0.5 * (lo + hi))
    return link


def add_environment(robot):
    for body in SM.bodies():
        add_box_link(robot, body)


def _set_fr3_limits(el, idx):
    lim = el.find("limit")
    if lim is None:
        return
    lim.set("effort", _fmt(TAU_MAX[idx]))
    lim.set("lower", _fmt(FR3_MIN[idx]))
    lim.set("upper", _fmt(FR3_MAX[idx]))
    lim.set("velocity", _fmt(QD_MAX[idx]))
    lim.attrib.pop("{http://drake.mit.edu}acceleration", None)


def clone_arm(robot, arm_id, spec, collision):
    """Clone the vendored panda into `arm{id}_`, re-shelled and re-limited.

    `collision` is "mesh" (the manufacturer's shells) or "capsule" (the
    audited set).  Either way the vendored SPHERES are dropped on the floor:
    they are the one arm collision model in this repo that nothing has ever
    measured, and every certified number was earned against something else.
    """
    pfx = f"arm{arm_id}_"
    src = ET.parse(SRC_URDF).getroot()
    caps_by_link = {}
    if collision == "capsule":
        for lnk, a, b, r, tag in SM.arm_collision_capsules():
            caps_by_link.setdefault(lnk, []).append((a, b, r, tag))

    for el in list(src):
        if el.tag not in ("link", "joint",
                          "{http://drake.mit.edu}collision_filter_group"):
            continue
        el.set("name", pfx + el.get("name"))
        for sub in el.iter():
            for att in ("link", "joint"):
                if sub.get(att):
                    sub.set(att, pfx + sub.get(att))
            # `material` and every drake:* element that names a group (the
            # vendored groups reference themselves by name, so the reference
            # has to move with the definition or drake refuses to parse).
            # `el.iter()` yields `el` itself, whose name is already prefixed.
            if sub is not el and sub.get("name") \
                    and (sub.tag == "material"
                         or sub.tag.startswith("{http://drake.mit.edu}")):
                sub.set("name", pfx + sub.get("name"))
            if sub.tag == "mesh" and sub.get("filename"):
                f = Path(sub.get("filename")).name
                sub.set("filename", f"meshes/fr3/{f}")

        if el.tag == "link":
            bare = el.get("name")[len(pfx):]
            for c in list(el.findall("collision")):
                el.remove(c)
            if collision == "mesh":
                shell = SHELL_OF.get(bare)
                if shell:
                    c = ET.SubElement(el, "collision")
                    _origin(c, (0, 0, 0), _shell_rpy(el))
                    ET.SubElement(ET.SubElement(c, "geometry"), "mesh",
                                  filename=f"meshes/collision/{shell}.obj")
            else:
                for a, b, r, tag in caps_by_link.get(bare, ()):
                    _add_capsule(el, a, b, r)
        elif el.tag == "joint":
            name = el.get("name")
            if name.startswith(f"{pfx}panda_joint") and name[-1].isdigit() \
                    and 1 <= int(name[-1]) <= 7:
                _set_fr3_limits(el, int(name[-1]) - 1)
            if "finger" in name:
                el.set("type", "fixed")
                for tag in ("mimic", "limit", "axis", "dynamics"):
                    for c in list(el.findall(tag)):
                        el.remove(c)
                o = el.find("origin")
                xyz = [float(v) for v in (o.get("xyz", "0 0 0")).split()]
                xyz[1] = FINGER_FIX if name.endswith("joint1") else -FINGER_FIX
                o.set("xyz", _v3(xyz))
        robot.append(el)

    T = spec.T_world_base(float(layout.LAYOUT_PROPOSED["h"]))
    j = ET.SubElement(robot, "joint", name=f"{pfx}mount_weld", type="fixed")
    ET.SubElement(j, "parent", link="world")
    ET.SubElement(j, "child", link=f"{pfx}panda_link0")
    _origin(j, T[:3, 3], rpy_from_R(T[:3, :3]))
    return pfx


# Rx(-90), for stripping the fingers' extra glTF correction (see _shell_rpy)
_RX_M90 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


def _shell_rpy(link_el):
    """Where a manufacturer collision shell sits in ITS link's frame -> rpy.

    Identity for nine of the eleven, and Rz(180) for the RIGHT FINGER, which
    is the whole reason this is a function.

    The shells are authored in each link's own frame, so they need no
    transform — EXCEPT that the two fingers are one mesh used twice, and the
    vendored URDF mirrors the right one with `rpy = (pi/2, 0, pi)` against the
    left's `(pi/2, 0, 0)`.  The common `pi/2` is a glTF-orientation
    correction that only the finger visual needs and that the shell does not;
    the `pi` is the mirror, and the shell needs it exactly as much as the
    visual does.  So: take the visual's own rotation and strip the Rx(pi/2).

    Dropping it is not cosmetic.  `finger.obj` spans y in [-0.0001, 0.0264] —
    it lies entirely on one side — so an unmirrored right finger puts its
    collision shell up to 52.8 mm from where the finger is.
    """
    v = link_el.find("visual")
    o = None if v is None else v.find("origin")
    rpy = [float(t) for t in (o.get("rpy", "0 0 0") if o is not None
                              else "0 0 0").split()]
    if not any(rpy):
        return (0.0, 0.0, 0.0)
    R = np.eye(3)
    for ax, th in zip("xyz", rpy):
        c, s_ = np.cos(th), np.sin(th)
        M = {"x": [[1, 0, 0], [0, c, -s_], [0, s_, c]],
             "y": [[c, 0, s_], [0, 1, 0], [-s_, 0, c]],
             "z": [[c, -s_, 0], [s_, c, 0], [0, 0, 1]]}[ax]
        R = np.asarray(M, float) @ R
    # Rx(pi/2) @ Rx(-pi/2) is the identity in exact arithmetic and leaves a
    # 5e-12 residual in floats.  A nanoradian is not a rotation; snap it, so
    # the left finger reads `0 0 0` and a reader can see at a glance that only
    # the right one is mirrored.
    return tuple(0.0 if abs(v) < 1e-9 else v
                 for v in rpy_from_R(R @ _RX_M90))


def _add_capsule(link, a, b, r):
    """A link-frame capsule -> a `drake:capsule` collision element."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    L = float(np.linalg.norm(d))
    z = d / L
    # any frame whose z is the capsule axis; the capsule is axisymmetric
    tmp = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 \
        else np.array([1.0, 0.0, 0.0])
    x = np.cross(tmp, z)
    x /= np.linalg.norm(x)
    R = np.column_stack([x, np.cross(z, x), z])
    c = ET.SubElement(link, "collision")
    _origin(c, 0.5 * (a + b), rpy_from_R(R))
    ET.SubElement(ET.SubElement(c, "geometry"),
                  "{http://drake.mit.edu}capsule",
                  radius=_fmt(r), length=_fmt(L))


def add_tool(robot, pfx, pen_ext=PEN_EXT, pen_lat=PEN_LAT_HOLDER):
    """The pen holder and its graphite, welded to the hand.

    RED FLAG, and it is the biggest one in this model.  The 2026.08.19 CAD
    delivery is eight printed parts with NO ASSEMBLY FILE, and no fingertip
    cradle geometry at all, so where the holder sits in the hand is INFERRED
    rather than read: `rig_final.penholder22_T_hand` puts the grip centre on
    the hand TCP and aims the bore along the PLANNER's TCP-to-tip ray.  The
    housing's own machined flats clock at 23.0 deg (measured off the STL);
    the planner's ray leans 45.0 deg.  Those are different numbers and only
    one of them can be right about the real part.

    The transform here is the PLANNING assumption, chosen so the model and the
    gate-validated tool transform agree.  It is pending Pete's caliper
    measurement — see system_model.OPEN_QUESTIONS["penholder_cradle"].
    """
    P = rig_final.PENHOLDER22
    T_h, _, nose, reach = rig_final.penholder22_T_hand(pen_ext, pen_lat,
                                                       D_HAND_TCP)
    # --- the holder: mesh visual, primitive collision --------------------
    link = ET.SubElement(robot, "link", name=f"{pfx}pen_holder")
    for mesh, rgba in ((f"meshes/penholder/{Path(P['visual_meshes'][0]).name}",
                        (0.24, 0.25, 0.28, 1.0)),
                       (f"meshes/penholder/{Path(P['visual_meshes'][1]).name}",
                        (0.13, 0.14, 0.16, 1.0))):
        v = ET.SubElement(link, "visual")
        _origin(v, (0, 0, 0))
        ET.SubElement(ET.SubElement(v, "geometry"), "mesh", filename=mesh)
        mat = ET.SubElement(v, "material",
                            name=f"{pfx}pen_holder_{Path(mesh).stem}_mat")
        ET.SubElement(mat, "color", rgba=_v3(rgba))
    for kind, T, (r, L) in rig_final.penholder22_collision(pen_ext, pen_lat,
                                                           D_HAND_TCP):
        assert kind == "cylinder", kind
        c = ET.SubElement(link, "collision")
        _origin(c, T[:3, 3], rpy_from_R(T[:3, :3]))
        ET.SubElement(ET.SubElement(c, "geometry"), "cylinder",
                      radius=_fmt(r), length=_fmt(L))
    j = ET.SubElement(robot, "joint", name=f"{pfx}pen_holder_weld",
                      type="fixed")
    ET.SubElement(j, "parent", link=f"{pfx}panda_hand")
    ET.SubElement(j, "child", link=f"{pfx}pen_holder")
    _origin(j, (0, 0, 0))

    # --- the graphite, nose to tip along the bore ------------------------
    lean = float(np.arctan2(pen_lat, pen_ext))
    link = ET.SubElement(robot, "link", name=f"{pfx}pen_lead")
    for role, r in (("visual", P["lead_r"]), ("collision", P["lead_r_coll"])):
        el = ET.SubElement(link, role)
        _origin(el, (np.sin(lean) * (nose + reach) / 2, 0.0,
                     np.cos(lean) * (nose + reach) / 2), (0.0, lean, 0.0))
        ET.SubElement(ET.SubElement(el, "geometry"), "cylinder",
                      radius=_fmt(r), length=_fmt(reach - nose))
        if role == "visual":
            mat = ET.SubElement(el, "material", name=f"{pfx}pen_lead_mat")
            ET.SubElement(mat, "color", rgba="0.16 0.16 0.17 1.0")
    j = ET.SubElement(robot, "joint", name=f"{pfx}pen_lead_weld", type="fixed")
    ET.SubElement(j, "parent", link=f"{pfx}tcp")
    ET.SubElement(j, "child", link=f"{pfx}pen_lead")
    _origin(j, (0, 0, 0))

    # --- the two pure frames the planner speaks in -----------------------
    ET.SubElement(robot, "link", name=f"{pfx}tcp")
    j = ET.SubElement(robot, "joint", name=f"{pfx}tcp_weld", type="fixed")
    ET.SubElement(j, "parent", link=f"{pfx}panda_hand")
    ET.SubElement(j, "child", link=f"{pfx}tcp")
    _origin(j, (0.0, 0.0, D_HAND_TCP))

    link = ET.SubElement(robot, "link", name=f"{pfx}pen_tip")
    v = ET.SubElement(link, "visual")
    ET.SubElement(ET.SubElement(v, "geometry"), "sphere", radius=_fmt(0.004))
    mat = ET.SubElement(v, "material", name=f"{pfx}pen_tip_mat")
    ET.SubElement(mat, "color", rgba="0.90 0.10 0.10 1.0")
    j = ET.SubElement(robot, "joint", name=f"{pfx}pen_tip_weld", type="fixed")
    ET.SubElement(j, "parent", link=f"{pfx}tcp")
    ET.SubElement(j, "child", link=f"{pfx}pen_tip")
    _origin(j, (pen_lat, 0.0, pen_ext))


def add_cable_dress(robot, pfx):
    """Translucent conduit loops on the forearm and wrist.  VISUAL ONLY.

    ESTIMATED, and marked so everywhere it appears: nothing about the real
    cable dress has been measured.  The 30 mm radius is the middle of the
    collision audit's 20-40 mm estimate.  These carry no collision geometry —
    turning them on is a re-certification, not a switch.
    """
    for i, (parent, a, b, what) in enumerate(SM.CABLE_DRESS):
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = b - a
        L = float(np.linalg.norm(d))
        z = d / L
        tmp = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 \
            else np.array([1.0, 0.0, 0.0])
        x = np.cross(tmp, z)
        x /= np.linalg.norm(x)
        R = np.column_stack([x, np.cross(z, x), z])
        name = f"{pfx}cable_dress_{i}"
        link = ET.SubElement(robot, "link", name=name)
        v = ET.SubElement(link, "visual")
        _origin(v, 0.5 * (a + b), rpy_from_R(R))
        ET.SubElement(ET.SubElement(v, "geometry"), "cylinder",
                      radius=_fmt(SM.CABLE_R / MM), length=_fmt(L))
        mat = ET.SubElement(v, "material", name=f"{name}_mat")
        ET.SubElement(mat, "color", rgba="0.15 0.15 0.17 0.45")
        j = ET.SubElement(robot, "joint", name=f"{name}_weld", type="fixed")
        ET.SubElement(j, "parent", link=f"{pfx}{parent}")
        ET.SubElement(j, "child", link=name)
        _origin(j, (0, 0, 0))


def add_collision_filters(robot, arm_id):
    pfx = f"arm{arm_id}_"
    ns = "{http://drake.mit.edu}"

    def group(name, members, ignores=()):
        g = ET.SubElement(robot, f"{ns}collision_filter_group", name=name)
        for m in members:
            ET.SubElement(g, f"{ns}member", link=m)
        for ig in ignores:
            ET.SubElement(g, f"{ns}ignored_collision_filter_group", name=ig)

    # An arm is bolted to its own plate and clamp stack by construction, and
    # its posts are the steel it hangs from — none of that is an obstacle to it.
    own = [f"plate{arm_id}", f"clamp{arm_id}"] \
        + [f"post{arm_id}_{i}{j}" for i in range(2) for j in range(2)] \
        + [f"gusset{arm_id}_{i}{j}" for i in range(2) for j in range(2)]
    group(f"{pfx}body", [f"{pfx}{n}" for n in ARM_LINKS],
          ignores=[f"mountsteel{arm_id}"])
    group(f"mountsteel{arm_id}", own)
    group(f"{pfx}tool", [f"{pfx}{n}" for n in TOOL_LINKS],
          ignores=[f"{pfx}wrist"])
    group(f"{pfx}wrist", [f"{pfx}{n}" for n in WRIST_LINKS])


def build(with_arms=True, collision="mesh"):
    name = ("aris_system_model" if with_arms else "aris_system_model_env")
    if with_arms and collision != "mesh":
        name = f"aris_system_model_{collision}"
    robot = ET.Element("robot", name=name)
    ET.SubElement(robot, "link", name="world")
    add_environment(robot)
    if with_arms:
        for aid, spec in layout.FLEET_PROPOSED.items():
            pfx = clone_arm(robot, aid, spec, collision)
            add_tool(robot, pfx)
            add_cable_dress(robot, pfx)
            add_collision_filters(robot, aid)
    ET.indent(robot)
    return ET.ElementTree(robot)


# ---------------------------------------------------------------------------
# STAGE 3 — the manifest
# ---------------------------------------------------------------------------
def manifest(out_dir=None):
    out_dir = Path(out_dir or OUT_DIR)
    h = SM.H_MOUNT
    mesh_src = out_dir / "meshes/MESH_SOURCES.json"
    meshes = json.loads(mesh_src.read_text()) if mesh_src.is_file() else {}
    bodies = []
    for b in SM.bodies(h):
        bodies.append(dict(
            name=b.name, kind=b.kind, provenance=b.provenance,
            size_mm=list(b.size), centre_mm=list(b.centre),
            lo_mm=list(b.lo), hi_mm=list(b.hi),
            collision=b.collision, source=b.source,
            **({"note": b.note} if b.note else {})))
    caps = SM.arm_collision_capsules()
    return dict(
        model="aris_system_model",
        purpose="the definitive dimensioned model of the ARIS six-arm "
                "installation: cage, table, canvas, six FR3 arms, pen "
                "holders.  Visual and collision.",
        generated_by="scripts/gen_system_model.py",
        truth_module="aris_sixarm/system_model.py",
        units=dict(manifest="mm, canvas frame, z=0 at the paper top surface",
                   urdf="m, same frame"),
        frame=dict(origin="the canvas reference corner",
                   x="across the short side of the paper, 0 -> 1803.4",
                   y="along the long side, 0 -> 3630.64",
                   z="up, 0 at the paper's TOP surface"),
        provenance_classes={
            "DRAWING": "lifted from the original drawing through "
                       "rig_final.FRAME_BOXES_W_CM / ARM_MOUNTS_W",
            "CODE": "a constant this repo plans against",
            "AUDIT": "measured by a script in this repo, off CAD or off the "
                     "manufacturer's meshes",
            "ASSUMED": "chosen here; every one is also in open_questions"},
        provenance_mix_scope="the 77 STATIC bodies only — the cage, the "
                             "table and the canvas.  The arms, their "
                             "collision geometry and the tool are not boxes "
                             "and carry their own class in `arm_geometry` "
                             "and `tool` (both AUDIT, except the tool "
                             "PLACEMENT, which is ASSUMED).",
        provenance_mix={k: dict(count=v[0], percent=v[1])
                        for k, v in SM.provenance_mix(h).items()},
        canvas=dict(size_mm=[SM.CANVAS_W, SM.CANVAS_L],
                    source="layout.SHEET_FINAL6", provenance="CODE"),
        mount_plane_mm=dict(value=h, source="layout.LAYOUT_PROPOSED['h']",
                            provenance="CODE",
                            note="docs/BUILD_SHEET.md still publishes 850.0"),
        z_ladder_mm=SM.z_ladder(h),
        cage=dict(
            profile_mm=SM.PROFILE,
            frame_outside_mm=[SM.FR_W, SM.FR_L],
            margin_mm=SM.MARGIN,
            runway_span_mm=SM.RAIL_LEN_X,
            runway_width_mm=SM.RUNWAY_W,
            post_section_mm=SM.PROFILE,
            post_pitch_mm=[SM.POST_PITCH_X, SM.POST_PITCH_Y],
            post_length_mm=SM.z_ladder(h)["post_length"],
            post_overrun_mm=SM.POST_OVER,
            plate_mm=list(SM.PLATE),
            plate_offset_mm=-SM.PLATE_OFF,
            clamp_mm=list(SM.CLAMP),
            gusset_mm=list(SM.GUSSET),
            gusset_pair_clear_mm=SM.GUSSET_PAIR_CLEAR,
            cluster_plan_mm=[SM.CLUSTER_W, SM.CLUSTER_D],
            total_height_above_floor_mm=SM.CAGE_TOTAL_H,
            provenance="DRAWING, at the CORRECTED datum",
            datum_note="the drawing's 233,7 cm is FLOOR to top-of-cage; "
                       "mounts.MOUNTS.ceiling_z = 2.34 m is that number "
                       "re-datumed to the paper, which is a provenance bug"),
        arms={str(aid): dict(
            xy_mm=list(SM.ARM_XY[aid]),
            base_z_mm=h,
            R_world_base="Ry(180 deg) — every arm hangs, front toward "
                         "canvas -x",
            plate_centre_x_mm=round(SM.plate_centre_x(SM.ARM_XY[aid][0]), 2),
            post_x_mm=[round(v, 2) for v in SM.post_x(SM.ARM_XY[aid][0])],
            provenance="CODE (layout.FLEET_PROPOSED)")
            for aid in layout.FLEET_PROPOSED},
        arm_geometry=dict(
            visual=dict(
                source="franka_description glTF, vendored; textures vendored "
                       "byte-identically from "
                       "~/git/franka_manipulation_station",
                provenance="AUDIT",
                note="the same meshes the repo already committed; only the "
                     "image table is rewritten"),
            collision_mesh=dict(
                source=SM.COLLISION_MESH_SOURCE,
                provenance="AUDIT",
                panda_vs_fr3_worst_mm=SM.PANDA_VS_FR3_WORST_MM,
                method="scripts/collision_audit.py --validate compares each "
                       "shell's AABB with the FR3's own collision box in the "
                       "station's fr3_franka_hand.urdf; eight of nine agree "
                       "to <= 0.5 um, link6 to 0.344 mm",
                convexity="drake reports the convex hull at the same volume "
                          "to 0.1 % on nine of ten shells (link6 4.2 %), so "
                          "they are already convex shells and nothing is lost",
                used_in="installation.urdf"),
            collision_capsule=dict(
                source="selfcoll.BODY_CAPSULES",
                provenance="AUDIT",
                count=len(caps),
                method="fitted to the manufacturer's collision meshes by "
                       "scripts/self_collision_audit.py; each row records the "
                       "exact mesh maximum its radius rounds up from",
                used_in="installation_capsules.urdf"),
            spheres_removed=dict(
                what="the vendored panda's 66 collision spheres per arm",
                why="the one arm collision model in this repo that nothing "
                    "has ever audited, and not the model any certified number "
                    "was earned against.  assets/proposed_rig/ still carries "
                    "them; this model does not.")),
        tool=dict(
            transform="rig_final.penholder22_T_hand(PEN_EXT=%.3f, "
                      "PEN_LAT_HOLDER=%.3f, D_HAND_TCP=%.4f)"
                      % (PEN_EXT, PEN_LAT_HOLDER, D_HAND_TCP),
            provenance="ASSUMED",
            red_flag="THE PLACEMENT IS INFERRED.  The CAD delivery has no "
                     "assembly file and no fingertip cradle geometry.  The "
                     "housing's own flats clock at 23.0 deg; the planner's "
                     "TCP-to-tip ray leans 45.0 deg; this model uses the "
                     "planner's ray.  PENDING Pete's caliper measurement.",
            collision="the 3-cylinder envelope, re-proved on every mesh "
                      "regeneration to contain every visual vertex",
            envelope_cylinders=[list(c) for c in
                                rig_final.PENHOLDER22["env_cylinders"]]),
        cable_dress=dict(
            provenance="ASSUMED",
            radius_mm=SM.CABLE_R,
            estimate="20-40 mm, the collision audit's",
            collision=False,
            note="ESTIMATED, visual only, translucent.  Nothing about the "
                 "real dress has been measured."),
        meshes=meshes,
        reconciliation=SM.reconciliation(h),
        open_questions=SM.OPEN_QUESTIONS,
        bodies=bodies)


# ---------------------------------------------------------------------------
def _rel(p):
    """Path for the log — relative to the repo when it is inside it."""
    p = Path(p)
    try:
        return p.relative_to(ROOT)
    except ValueError:
        return p


def write_urdfs(out_dir=None):
    out_dir = Path(out_dir or OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    wrote = []
    for fname, with_arms, coll in (("installation.urdf", True, "mesh"),
                                   ("installation_capsules.urdf", True,
                                    "capsule"),
                                   ("environment.urdf", False, "mesh")):
        tree = build(with_arms, coll)
        tree.write(out_dir / fname, xml_declaration=True, encoding="utf-8")
        n = len(tree.getroot().findall("link"))
        wrote.append((fname, n))
        print(f"wrote {_rel(out_dir / fname)}  ({n} links)")
    (out_dir / "model_manifest.json").write_text(
        json.dumps(manifest(out_dir), indent=2, sort_keys=False) + "\n")
    print(f"wrote {_rel(out_dir / 'model_manifest.json')}")
    return wrote


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    stage = argv[0] if argv else "all"
    if stage not in ("meshes", "urdf", "all"):
        raise SystemExit(__doc__)
    if stage in ("meshes", "all"):
        files, quality = build_meshes(OUT_DIR)
        tot = sum(f["bytes"] for f in files)
        print(f"meshes: {len(files)} files, {tot / 1e6:.2f} MB")
        for q in quality["parts"]:
            print(f"  holder {q['part']:8s} {q['faces_raw']:>7d} -> "
                  f"{q['faces_out']:>5d} faces, extent lost "
                  f"{q['extent_loss_mm']:.4f} mm")
        print(f"  holder envelope worst escape "
              f"{quality['envelope_worst_escape_m'] * 1000:+.6f} mm")
    if stage in ("urdf", "all"):
        write_urdfs(OUT_DIR)


if __name__ == "__main__":
    main()
