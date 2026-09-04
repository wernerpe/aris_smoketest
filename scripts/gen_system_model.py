#!/usr/bin/env python3
"""Generate `assets/system_model/` — the whole installation, visual + collision.

    python3 scripts/gen_system_model.py meshes    # vendor + decimate meshes
    python3 scripts/gen_system_model.py urdf      # write URDFs + manifest
    python3 scripts/gen_system_model.py all       # both
    python3 scripts/gen_system_model.py urdf --fingers stock|fat|both

WHAT IT WRITES
--------------
    installation.urdf            cage + table + paper + six arms + holders,
                                 collision = the manufacturer's own shells
    installation_capsules.urdf   the same scene, collision = the AUDITED
                                 capsule set (selfcoll.BODY_CAPSULES)
    environment.urdf             the static scene alone, no arms
    installation_fatfingers.urdf installation.urdf with the printed 90 mm
                                 "Fat Franka Finger" in place of the stock one
    model_manifest.json          every body: dimensions, provenance, source
    meshes/fr3/*.gltf            the vendored Franka visuals, RETEXTURED
    meshes/fr3/textures/*.png    the 27 real texture maps, byte-identical
    meshes/collision/*.obj       the manufacturer's collision shells
    meshes/penholder/*.obj       the pen holder, re-decimated from raw CAD
    meshes/fatfinger/*.obj       the Fat finger, in the LEFT finger's frame

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
    and printed a warning per image per mesh.  The 54 the arm's own meshes
    name — 27 PNG maps and their 27 .ktx2 twins — were all still sitting in
    the directory the meshes themselves came from
    (`~/git/franka_manipulation_station/assets/franka_description`).  The 27
    PNGs are vendored BYTE-IDENTICALLY and the glTFs re-pointed at them; the
    .ktx2 twins are dropped, because VTK says in as many words that it cannot
    read them.  No runtime hack, no stripped materials, no resampling.

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
# guarded: the tests load this module by path three times, and an unguarded
# insert leaves six copies of the same two directories on sys.path
for _p in (str(ROOT / "scripts"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from aris_sixarm import frames, layout, mounts, rig_final, selfcoll  # noqa: E402
from aris_sixarm import system_model as SM                            # noqa: E402
from aris_sixarm.frames import (D_HAND_TCP, FR3_MAX, FR3_MIN,  # noqa: E402
                                PEN_EXT_HOLDER, PEN_LAT_HOLDER, QD_MAX,
                                TAU_MAX)
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
# the two parts of the mounted holder that are visible from outside.  The
# stack inside the 21.1 mm bore is drawn too, as primitives, out of
# `rig_final.penholder22_internals`; only the shim set and the bench extractor
# stay out (PENHOLDER22["omitted"]).
HOUSING_STL = "pen holder housing - 22 deg - reinforced - v20260429.STL"
CAP_STL = "pen holder cap v20250903.STL"
# The whole-finger replacement that arrived 2026-09-02.  It is a VARIANT, not a
# replacement for the stock finger: `--fingers` picks which one the URDF gets,
# and `installation.urdf` stays the stock build.  See rig_final.FATFINGER and
# docs/SYSTEM_MODEL.md 7d.
FAT_FINGER_STL = "Fat Franka Finger v250904.STL"

FR3_MESHES = tuple(f"link{i}" for i in range(8)) + ("hand", "finger")
TEXTURED = tuple(f"link{i}" for i in range(8)) + ("hand",)   # finger has none
TEX_KINDS = ("color", "normal", "occlusion_roughness_metallic")

# WHERE THE FINGERS ACTUALLY CLOSE TO, m.  The mount post is 50 mm long with a
# 7.000 mm socket in each end (measured on this delivery's own STL: the socket
# floors are at post z = 7.00 and 43.00), and the 10-deg assembly seats an FR3
# fingertip in each socket — its two grip faces come out 36.0008 mm apart.  So
# the half-width is 18.000, not the 28.500 this constant used to carry: 28.500
# is the fingertip's BACK face, 3.5 mm proud of the post's end, and a finger
# parked there holds nothing because 57 mm is 7 mm wider than the post is long.
# The URDF's `finger.gltf` includes its own tip, so it wants the grip plane.
# At 0.018 the mesh lands where the assembly puts it in all three axes: the
# finger's distal 18.1 mm covers panda_hand z 94.2..112.3 and the CAD block
# covers 94.2..112.3.  See rig_final.PENHOLDER22's docstring.
#
# THIS IS THE ASSEMBLY'S NUMBER, NOT NECESSARILY THE RIG'S.  The running robot
# grasps with width 0.0432 and epsilon_inner 0.0, and libfranka only calls a
# grasp successful above width - epsilon_inner, so the real jaw gap is at least
# 43.2 mm and 36.0 would fail every time.  50 mm — the post's BARE ENDS — does
# not, and 50 mm is what the newest finger part can clamp: "Fat Franka Finger
# v250904" is an 18.4 mm blade and 18.4 does not enter an 18.0 socket.  See
# system_model.OPEN_QUESTIONS["penholder_cradle"]; it changes no angle.
FINGER_FIX = 0.018           # m, fingertip grip half-width holding the holder

# WHERE THE FAT FINGER'S JOINT SITS, m — and it is DERIVED, not chosen.  The
# Fat blade's innermost feature is not its contact plate but the 2.5839 mm rib
# along the plate's proximal edge (rig_final.FATFINGER), 8.0663 mm out from the
# stock grip plane.  Closing two of these on the mount post's bare 50 mm ends,
# the RIB lands first, at half the post less that reach — and bisecting the
# blade against the committed holder meshes returns the same 16.9337 mm and
# names the rib crest as the touching vertex.  The plates never get there; they
# stop 2.5839 mm off.  libfranka would report `width` 0.0339.
#
# THIS IS NOT A CLAIM ABOUT WHAT THE ARMS DO.  It is the tightest the BARE
# blade can close on this holder, which is what the URDF draws.  If a fingertip
# is bolted to the plate — and four measurements say the plate is a seat for
# one — the tip grips instead, at 0.0357 seated in the sockets or 0.0497 flat
# on the bare ends.  `rig_final.fatfinger_widths()` lists all six, and
# system_model.OPEN_QUESTIONS["penholder_cradle"] says which one measurement
# settles it.
FAT_FINGER_FIX = round(
    0.5 * (rig_final.PENHOLDER22["post_z"][1]
           - rig_final.PENHOLDER22["post_z"][0])
    - rig_final.FATFINGER["rib_offset"], 7)          # 0.0169337
HOLDER_FACES = 8000          # decimation target, per part
FAT_FINGER_FACES = 8000      # same budget; 8234 raw, and it costs 0.0 mm
FINGER_VARIANTS = ("stock", "fat")
MM = SM.MM

# the stack inside the bore: spring steel, printed black, printed grey, lead
INTERNAL_RGBA = dict(spring=(0.72, 0.74, 0.78, 1.0),
                     spacer=(0.30, 0.31, 0.34, 1.0),
                     sleeve=(0.20, 0.21, 0.24, 1.0),
                     clutch=(0.55, 0.57, 0.60, 1.0),
                     graphite_buried=(0.16, 0.16, 0.17, 1.0),
                     # the tail is the same stick, drawn a shade lighter so
                     # the render says which end of it is which
                     graphite_tail=(0.34, 0.24, 0.16, 1.0))

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
        if not src.is_file():
            raise SystemExit(f"missing vendored glTF {src}")
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
                if t["source"] not in remap:
                    raise SystemExit(
                        f"{src.name}: texture {t} names only a ktx2 image, "
                        "which VTK cannot read and this model does not "
                        "vendor; it has no PNG to fall back to")
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
        if not src.is_file():
            raise SystemExit(f"missing collision shell {src}")
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
    T_h, T_c, _, _ = rig_final.penholder22_T_hand(PEN_EXT_HOLDER, PEN_LAT_HOLDER,
                                                  D_HAND_TCP)
    parts = (("housing", HOUSING_STL, T_h), ("cap", CAP_STL, T_c))
    recs, quality = [], []
    for name, stl, T in parts:
        src = CAD_SRC / stl
        if not src.is_file():
            raise SystemExit(f"raw CAD not found: {src}")
        m = trimesh.load(src, force="mesh")
        ext0 = m.bounds[1] - m.bounds[0]
        # NOT an assert: a units check that `python -O` switches off is not a
        # units check.  A printed pen holder is a 10 cm object; if these were
        # metres the part would be 80 m long, if centimetres 80 cm.
        if not 5.0 < ext0.max() < 500.0:
            raise SystemExit(f"{name}: extent {ext0} is not millimetres — "
                             "has the CAD been re-exported in other units?")
        n0 = len(m.faces)
        m.merge_vertices(merge_tex=True, merge_norm=True)
        if len(m.faces) > faces:
            m = m.simplify_quadric_decimation(face_count=faces)
        ext1 = m.bounds[1] - m.bounds[0]
        shrink = float(np.abs(ext1 - ext0).max())
        if shrink >= 1.0:
            raise SystemExit(f"{name}: decimation lost {shrink:.3f} mm of "
                             "extent; raise the face target")
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
    if worst > 1e-5:
        raise SystemExit("the holder escapes its collision envelope by "
                         f"{worst * 1000:.4f} mm — the envelope is no longer "
                         "conservative and rig_final.PENHOLDER22 must be "
                         "re-fitted before this mesh is used")
    return recs, dict(parts=quality, envelope_worst_escape_m=round(worst, 12),
                      envelope_bar_m=1e-5)


def _fat_boxes(m):
    """The Fat blade's collision envelope -> [(lo, hi)], panda_leftfinger, m.

    MEASURED off the mesh, not typed in.  Three contiguous z bands at the CAD's
    own steps (foot / web / plate), the foot band split in two at the
    part's own mirror plane because the two feet have 48.6 mm of air between
    them, and each box the AABB of its cell.  The AABB is fitted over a z
    window `overlap` wider than the band it is clamped to, so a triangle that
    straddles a boundary is inside BOTH boxes rather than between them.
    """
    F = rig_final.FATFINGER
    z0, z1 = F["collision_bands"]
    ov = F["collision_overlap"]
    lo_z, hi_z = float(m.bounds[0][2]), float(m.bounds[1][2])
    P = np.vstack([np.asarray(m.vertices, float), m.triangles.mean(axis=1)])
    split = F["collision_foot_split"]
    cells = [((lo_z, z0), (-np.inf, split)),
             ((lo_z, z0), (split, np.inf)),
             ((z0, z1), (-np.inf, np.inf)),
             ((z1, hi_z), (-np.inf, np.inf))]
    out = []
    for (za, zb), (xa, xb) in cells:
        sel = ((P[:, 2] >= za - ov) & (P[:, 2] <= zb + ov)
               & (P[:, 0] >= xa) & (P[:, 0] < xb))
        if not sel.any():
            raise SystemExit(f"fat finger: collision cell z[{za},{zb}] "
                             f"x[{xa},{xb}] is empty — has the part changed?")
        q = P[sel]
        x0 = float(q[:, 0].min()) if np.isinf(xa) else float(xa)
        x1 = float(q[:, 0].max()) if np.isinf(xb) else float(xb)
        # the two foot boxes ABUT on the split rather than stopping at their
        # own geometry: a triangle drawn across the 48 mm of air between the
        # feet would otherwise have its middle in neither box, which is a
        # 100 nm hole in the proof and no less a hole for being small
        out.append(((x0, float(q[:, 1].min()), float(za)),
                    (x1, float(q[:, 1].max()), float(zb))))
    return out


def _fat_escape(m, boxes):
    """How far the blade reaches outside `boxes` -> metres (<= 0 is contained).

    Vertices, face centroids AND edge midpoints, because a box set proved on
    vertices alone says nothing about a long triangle's middle.
    """
    v = np.asarray(m.vertices, float)
    t = m.triangles
    P = np.vstack([v, t.mean(axis=1),
                   0.5 * (t[:, 0] + t[:, 1]), 0.5 * (t[:, 1] + t[:, 2]),
                   0.5 * (t[:, 2] + t[:, 0])])
    out = np.full(len(P), np.inf)
    for lo, hi in boxes:
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        out = np.minimum(out, np.maximum(lo - P, P - hi).max(axis=1))
    return float(out.max())


def vendor_fat_finger(out_dir=None, faces=FAT_FINGER_FACES):
    """The Fat Franka Finger, raw STL -> panda_leftfinger frame OBJ.

    Same treatment as the holder — weld, decimate, units-check, measure what
    the decimation cost — and one thing more: the STL and the SLDPRT of this
    part DO NOT SHARE A DATUM (the STL is exported 10.5 mm along +y off the
    part origin), and it is the SLDPRT's datum that puts the plate hole on the
    finger centreline.  `rig_final.fatfinger_T_finger` carries that shift, so
    the mesh written here is already in the LEFT finger's link frame; the right
    finger gets the same file under `Rz(pi)` about x = 34.5 mm, which is the
    other foot bolted down and the reason the part has two of everything.
    """
    import trimesh
    out_dir = Path(out_dir or OUT_DIR)
    dst = out_dir / "meshes/fatfinger"
    dst.mkdir(parents=True, exist_ok=True)
    src = CAD_SRC / FAT_FINGER_STL
    if not src.is_file():
        raise SystemExit(f"raw CAD not found: {src}")
    m = trimesh.load(src, force="mesh")
    ext0 = m.bounds[1] - m.bounds[0]
    if not 5.0 < ext0.max() < 500.0:
        raise SystemExit(f"fat finger: extent {ext0} is not millimetres — "
                         "has the CAD been re-exported in other units?")
    n0 = len(m.faces)
    m.merge_vertices(merge_tex=True, merge_norm=True)
    if len(m.faces) > faces:
        m = m.simplify_quadric_decimation(face_count=faces)
    ext1 = m.bounds[1] - m.bounds[0]
    shrink = float(np.abs(ext1 - ext0).max())
    if shrink >= 1.0:
        raise SystemExit(f"fat finger: decimation lost {shrink:.3f} mm of "
                         "extent; raise the face target")
    m.apply_scale(0.001)                                    # mm -> m
    m.apply_transform(rig_final.fatfinger_T_finger())       # -> left finger
    out = dst / "fatfinger_leftfinger.obj"
    out.write_text(_obj_with_normals(m, "fatfinger_leftfinger"))
    # fit and prove the envelope on the mesh AS WRITTEN, the way
    # `decimate_holder` re-loads before re-proving.  The OBJ writer rounds to
    # 1 um, and a box fitted to the unrounded mesh is 0.1 um too small for the
    # file that ships — which is a hull that does not contain its own asset.
    m = trimesh.load(out, force="mesh")
    boxes = _fat_boxes(m)
    esc = _fat_escape(m, boxes)
    if esc > 0.0:
        raise SystemExit(f"the fat finger escapes its own box envelope by "
                         f"{esc * 1000:.4f} mm — a hull that does not contain "
                         "the part is not a hull")
    rec = dict(file="meshes/fatfinger/fatfinger_leftfinger.obj",
               bytes=out.stat().st_size, sha256=_sha256(out), source=str(src))
    quality = dict(faces_raw=n0, faces_out=len(m.faces),
                   extent_loss_mm=round(shrink, 4),
                   extent_mm=[round(float(v), 4) for v in ext1],
                   frame="panda_leftfinger",
                   stl_y_shift_m=rig_final.FATFINGER["stl_y_shift"],
                   collision_boxes_m=[[list(lo), list(hi)]
                                      for lo, hi in boxes],
                   collision_worst_escape_m=round(esc, 12),
                   joint_value_m=FAT_FINGER_FIX)
    return [rec], quality


def build_meshes(out_dir=None):
    out_dir = Path(out_dir or OUT_DIR)
    files = []
    files += vendor_textures(out_dir)
    files += retexture_gltfs(out_dir)
    files += vendor_collision(out_dir)
    holder, quality = decimate_holder(out_dir)
    files += holder
    fat, fat_quality = vendor_fat_finger(out_dir)
    files += fat
    (out_dir / "meshes/MESH_SOURCES.json").write_text(json.dumps(
        dict(files=files, holder_decimation=quality,
             fat_finger=fat_quality), indent=2) + "\n")
    return files, quality, fat_quality


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


def _fat_finger_geometry(link_el, right, boxes):
    """Re-body one finger link as the Fat blade — visual mesh + box envelope.

    ONE MESH SERVES BOTH FINGERS, and that is the part's own design rather than
    an economy here: it is mirror-symmetric about its middle (0.44 mm over
    every mating feature), so bolting the FAR foot down instead of the near one
    gives the mirror placement — and only that lets both blades reach the same
    way in the hand.  In the right finger's own link frame it is `Rz(pi)` about
    x = mirror_pitch / 2.  See rig_final.FATFINGER.
    """
    F = rig_final.FATFINGER
    for old in list(link_el.findall("visual")) + list(
            link_el.findall("collision")):
        link_el.remove(old)
    xyz = (F["mirror_pitch"], 0.0, 0.0) if right else (0.0, 0.0, 0.0)
    rpy = (0.0, 0.0, np.pi) if right else (0.0, 0.0, 0.0)
    v = ET.SubElement(link_el, "visual")
    _origin(v, xyz, rpy)
    ET.SubElement(ET.SubElement(v, "geometry"), "mesh",
                  filename="meshes/fatfinger/fatfinger_leftfinger.obj")
    mat = ET.SubElement(v, "material", name=f"{link_el.get('name')}_fat_mat")
    ET.SubElement(mat, "color", rgba="0.28 0.29 0.32 1.0")
    R = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]) \
        if right else np.eye(3)
    for lo, hi in boxes:
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        c = R @ (0.5 * (lo + hi)) + np.asarray(xyz, float)
        col = ET.SubElement(link_el, "collision")
        _origin(col, c, rpy)
        ET.SubElement(ET.SubElement(col, "geometry"), "box", size=_v3(hi - lo))


def clone_arm(robot, arm_id, spec, collision, fingers="stock",
              fat_boxes=None):
    """Clone the vendored panda into `arm{id}_`, re-shelled and re-limited.

    `collision` is "mesh" (the manufacturer's shells) or "capsule" (the
    audited set).  Either way the vendored SPHERES are dropped on the floor:
    they are the one arm collision model in this repo that nothing has ever
    measured, and every certified number was earned against something else.

    `fingers` is "stock" (the manufacturer's `finger.gltf` at FINGER_FIX) or
    "fat" (the printed whole-finger replacement at FAT_FINGER_FIX).  The fat
    blade carries its own MEASURED box envelope in both variants, because
    neither of the two arm collision models has anything to say about it:
    `finger.obj` is the wrong shape and `selfcoll.BODY_CAPSULES` has no finger
    row at all.
    """
    if fingers not in FINGER_VARIANTS:
        raise ValueError(f"fingers must be one of {FINGER_VARIANTS}, "
                         f"not {fingers!r}")
    fix = FINGER_FIX if fingers == "stock" else FAT_FINGER_FIX
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
            if fingers == "fat" and bare in ("panda_leftfinger",
                                             "panda_rightfinger"):
                _fat_finger_geometry(el, bare.endswith("rightfinger"),
                                     fat_boxes)
                robot.append(el)
                continue
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
                    el.append(ET.Comment(f" selfcoll.BODY_CAPSULES {tag} "))
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
                xyz[1] = fix if name.endswith("joint1") else -fix
                o.set("xyz", _v3(xyz))
        robot.append(el)

    T = spec.T_world_base(float(layout.LAYOUT_PROPOSED["h"]))
    j = ET.SubElement(robot, "joint", name=f"{pfx}mount_weld", type="fixed")
    ET.SubElement(j, "parent", link="world")
    ET.SubElement(j, "child", link=f"{pfx}panda_link0")
    _origin(j, T[:3, 3], _rpy_checked(T[:3, :3]))
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
    vis = link_el.find("visual")
    o = None if vis is None else vis.find("origin")
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
    return tuple(0.0 if abs(t) < 1e-9 else t
                 for t in _rpy_checked(R @ _RX_M90))


def _axis_frame(a, b):
    """Segment a->b -> (centre, rpy, length) for a z-axis-aligned primitive.

    Any frame whose z is the segment will do — a capsule and a cylinder are
    both axisymmetric — so the other two axes are picked off whichever world
    axis is least parallel to it.  The `0.9` pivot keeps the cross product
    away from zero: the worst case leaves it at 0.436 of unit length.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    L = float(np.linalg.norm(d))
    if L <= 0.0:
        raise ValueError(f"degenerate segment {a} -> {b}")
    z = d / L
    tmp = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 \
        else np.array([1.0, 0.0, 0.0])
    x = np.cross(tmp, z)
    x /= np.linalg.norm(x)
    R = np.column_stack([x, np.cross(z, x), z])
    return 0.5 * (a + b), _rpy_checked(R), L


def _rpy_checked(R):
    """`rpy_from_R`, with its own decomposition verified here.

    The only thing that checks a decomposition today is a bare `assert` inside
    `gen_final_rig_urdf`, which `python -O` removes — and every rotation this
    generator writes goes through it: base welds, shell mirrors, capsule
    frames, holder cylinders.  Re-composing is three matrix products.
    """
    rpy = rpy_from_R(R)
    r, p, y = (float(v) for v in rpy)
    cr, sr, cp, sp, cy, sy = (np.cos(r), np.sin(r), np.cos(p), np.sin(p),
                              np.cos(y), np.sin(y))
    back = (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.0]])
            @ np.array([[cp, 0, sp], [0, 1.0, 0], [-sp, 0, cp]])
            @ np.array([[1.0, 0, 0], [0, cr, -sr], [0, sr, cr]]))
    if not np.allclose(back, R, atol=1e-9):
        raise ValueError(f"rpy {rpy} does not recompose to\n{R}")
    return rpy


def _add_capsule(link, a, b, r):
    """A link-frame capsule -> a `drake:capsule` collision element."""
    ctr, rpy, L = _axis_frame(a, b)
    c = ET.SubElement(link, "collision")
    _origin(c, ctr, rpy)
    ET.SubElement(ET.SubElement(c, "geometry"),
                  "{http://drake.mit.edu}capsule",
                  radius=_fmt(r), length=_fmt(L))


def add_tool(robot, pfx, pen_ext=PEN_EXT_HOLDER, pen_lat=PEN_LAT_HOLDER):
    """The pen holder and its graphite, welded to the hand.

    RED FLAG, and it is still the biggest one in this model.  The 2026.08.19
    CAD delivery is eight printed parts with NO ASSEMBLY FILE, so where the
    holder sits in the hand is INFERRED rather than read:
    `rig_final.penholder22_T_hand` puts the grip centre on the hand TCP and
    aims the bore along the PLANNER's TCP-to-tip ray.  The housing's own
    machined flats clock at 23.0 deg (measured off the STL); the planner's ray
    leans 45.0 deg.  Those are different numbers and only one can be right.

    WHAT CHANGED.  The 10-deg "natural hold" ASSEMBLY has now been read
    (`rig_final.penholder22_stack`), and it took away the place that
    difference used to be parked.  There is no fingertip cradle: the stock FR3
    fingertips seat square in the post's own 18 x 18 x 7 mm sockets, so the
    housing's clocking reaches the hand undivided — and on the build that HAS
    an assembly the file's name-angle IS the lean (10.0000 deg) with the grip
    centre on the TCP to 0.14 mm.  ...AND 2026-09-03 CLOSED IT THE OTHER WAY.
    A photo of the real gripper shows NO fingertip fitted — the Fat blades'
    bare plates clamp the post's end faces — so on the deployed build nothing
    transmits the clocking, and what sets the lean instead is the hand: the
    housing's 55.1 mm of barrel behind the grip is inside the manufacturer's
    own hand collision shell at every lean under 35.17 deg (-11.90 mm at 23,
    +6.16 at 45).  The 45 deg ray stays, now for a hardware reason.  What DID
    move is the axial depth: PEN_EXT_HOLDER, 0.0588421 m, the tip ~5 cm below
    the blades' contact plates.  Neither half was ever gate-validated — only
    the INLINE pen's 0.110 is.  See
    system_model.OPEN_QUESTIONS["penholder_cradle"] and
    docs/SYSTEM_MODEL.md 7e.

    ...AND THE HOUSING IS NO LONGER MOUNTED END-FOR-END (2026-09-03).  The
    transform used to point the housing's +X away from the tip, which put the
    17.00 mm tail land toward the paper and the cap toward the wrist.  The pen
    leaves through the CAP — the assembly's preview, the spring's direction,
    what the 17.00 land will and will not pass, and the 25 mm grip-to-cap all
    say so — and the model now says so too.  It is a 180 deg rotation about
    the post axis; the tip does not move.  docs/SYSTEM_MODEL.md 7c.

    The INTERNALS are drawn as primitives from `penholder22_internals` and are
    VISUAL ONLY: every bore body is inside the 21.148 mm bore and the pencil
    tail is inside its own envelope cylinder, and
    `penholder22_internals_escape` re-proves on every run that the hull the
    collision model ships still contains all of them.
    """
    P = rig_final.PENHOLDER22
    _, _, exit_x, reach = rig_final.penholder22_T_hand(pen_ext, pen_lat,
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
        _origin(c, T[:3, 3], _rpy_checked(T[:3, :3]))
        ET.SubElement(ET.SubElement(c, "geometry"), "cylinder",
                      radius=_fmt(r), length=_fmt(L))
    # --- the stack inside the bore.  VISUAL ONLY, and inside the hull -----
    esc = rig_final.penholder22_internals_escape()
    if esc > 0.0:
        raise SystemExit(f"the holder's internals escape the 3-cylinder "
                         f"envelope by {esc * 1000:.4f} mm — either the stack "
                         "or PENHOLDER22['env_cylinders'] is wrong, and a "
                         "hull that does not contain the assembly is not a "
                         "hull")
    for name, T, (r, L), note in rig_final.penholder22_internals(
            pen_ext, pen_lat, D_HAND_TCP):
        link.append(ET.Comment(f" {name}: {note} "))
        v = ET.SubElement(link, "visual")
        _origin(v, T[:3, 3], _rpy_checked(T[:3, :3]))
        ET.SubElement(ET.SubElement(v, "geometry"), "cylinder",
                      radius=_fmt(r), length=_fmt(L))
        mat = ET.SubElement(v, "material", name=f"{pfx}pen_{name}_mat")
        ET.SubElement(mat, "color", rgba=_v3(INTERNAL_RGBA[name]))
    j = ET.SubElement(robot, "joint", name=f"{pfx}pen_holder_weld",
                      type="fixed")
    ET.SubElement(j, "parent", link=f"{pfx}panda_hand")
    ET.SubElement(j, "child", link=f"{pfx}pen_holder")
    _origin(j, (0, 0, 0))

    # --- the graphite, the CAP's outer face to the tip, along the BORE ----
    # `penholder22_lead` owns the arithmetic, because the bore stopped being
    # the TCP -> tip ray on 2026-09-04 (the grip is at the far end of the Fat
    # blades now, `PENHOLDER22["grip_hand_x"]`) and rebuilding it here from
    # `arctan2(pen_lat, pen_ext)` would draw the graphite down the wrong line.
    la, lb = rig_final.penholder22_lead(pen_ext, pen_lat, D_HAND_TCP)
    tcp_o = np.array([0.0, 0.0, D_HAND_TCP])       # the link is welded to `tcp`
    lean = float(np.arctan2(lb[0] - la[0], lb[2] - la[2]))   # the BORE's own
    lctr = tuple(0.5 * (la + lb) - tcp_o)
    lL = float(np.linalg.norm(lb - la))
    link = ET.SubElement(robot, "link", name=f"{pfx}pen_lead")
    for role, r in (("visual", P["lead_r"]), ("collision", P["lead_r_coll"])):
        el = ET.SubElement(link, role)
        _origin(el, lctr, (0.0, lean, 0.0))
        ET.SubElement(ET.SubElement(el, "geometry"), "cylinder",
                      radius=_fmt(r), length=_fmt(lL))
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
        ctr, rpy, L = _axis_frame(a, b)
        name = f"{pfx}cable_dress_{i}"
        link = ET.SubElement(robot, "link", name=name)
        link.append(ET.Comment(f" ESTIMATED: {what} "))
        v = ET.SubElement(link, "visual")
        _origin(v, ctr, rpy)
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


def fat_boxes_from(out_dir):
    """The measured Fat-finger envelope, out of the mesh stage's own record.

    The URDF stage never re-reads the STL — the boxes were measured when the
    mesh was vendored and they live in `meshes/MESH_SOURCES.json` beside the
    sha256 of the mesh they were measured on, so the two cannot drift apart.
    """
    src = Path(out_dir) / "meshes/MESH_SOURCES.json"
    if not src.is_file():
        raise SystemExit(f"{src} is missing — run "
                         "`gen_system_model.py meshes` before `urdf`")
    d = json.loads(src.read_text()).get("fat_finger")
    if not d:
        raise SystemExit(f"{src} carries no fat_finger record — re-run "
                         "`gen_system_model.py meshes`")
    return [(tuple(lo), tuple(hi)) for lo, hi in d["collision_boxes_m"]]


def build(with_arms=True, collision="mesh", fingers="stock", fat_boxes=None):
    name = ("aris_system_model" if with_arms else "aris_system_model_env")
    if with_arms and collision != "mesh":
        name = f"aris_system_model_{collision}"
    if with_arms and fingers != "stock":
        name = f"aris_system_model_{fingers}fingers"
    robot = ET.Element("robot", name=name)
    ET.SubElement(robot, "link", name="world")
    add_environment(robot)
    if with_arms:
        for aid, spec in layout.FLEET_PROPOSED.items():
            pfx = clone_arm(robot, aid, spec, collision, fingers, fat_boxes)
            add_tool(robot, pfx)
            add_cable_dress(robot, pfx)
            add_collision_filters(robot, aid)
    ET.indent(robot)
    return ET.ElementTree(robot)


# ---------------------------------------------------------------------------
# STAGE 3 — the manifest
# ---------------------------------------------------------------------------
def _fat_finger_manifest(meshes):
    """The Fat finger's own block — geometry, placement, and what is OPEN."""
    F = rig_final.FATFINGER
    q = meshes.get("fat_finger", {})
    return dict(
        provenance="AUDIT for the geometry and the placement, ASSUMED for the "
                   "grasp",
        what="a printed whole-finger replacement, 18.4339 x 90.0003 x 50.000 "
             "mm: two mounting feet 69 mm apart, a slanted web, and a flat "
             "contact plate with a 2.5839 mm rib along its proximal edge",
        source=F["source"],
        used_in="installation_fatfingers.urdf",
        stl_y_shift_m=F["stl_y_shift"],
        stl_y_shift_why="the STL and the SLDPRT of this part DO NOT share a "
                        "datum: the STL is exported 10.5000 mm along +y off "
                        "the part origin (x and z agree to 0.0002 mm).  The "
                        "SLDPRT datum is the one that matters — it puts the "
                        "plate hole on the finger centreline.",
        placement="rig_final.fatfinger_T_finger(); link x = cad y, link y = "
                  "0.0785578 - cad x, link z = cad z - 0.0101579",
        placement_evidence="the part is drawn in the SAME frame as "
                           "'Franka_Finger_FR3 Fingertip only.SLDPRT': the "
                           "plate's 6.000 mm hole is on the axis of that "
                           "tip's 93514A130 brass insert to 0.0002 mm, the "
                           "plate occupies exactly the tip's z band, and it "
                           "lands 0.1502 mm outboard of the tip's back face.  "
                           "So the map into the finger is fixed by placing "
                           "the FINGERTIP, which the 10-deg assembly already "
                           "did.",
        placement_residual_mm=dict(
            foot_outer_face_vs_finger_back=0.155,
            plate_face_vs_fingertip_back=0.150,
            grip_plane_vs_finger_inner_face=0.084,
            fingertip_distal_face_vs_finger_tip=0.0,
            note="worst 0.155 mm, against manufacturer meshes that disagree "
                 "with each other by 0.051 mm.  Independent check: the "
                 "plate's z centre lands at panda_hand z = 103.242 mm "
                 "against the 10-deg assembly's grip centre of 103.26 and "
                 "the stock TCP's 103.4."),
        mirror=dict(plane_cad_y_m=F["mirror_y"], worst_m=F["mirror_worst"],
                    volume_fraction=F["mirror_volume_fraction"],
                    why="the part is its own mirror image, so bolting the FAR "
                        "foot down gives the mirror placement — which is what "
                        "a LEFT and a RIGHT finger need if both blades are to "
                        "reach the same way in the hand.  That is why there "
                        "are two feet, four foot holes and two plate holes."),
        plate_offset_m=F["plate_offset"],
        rib_offset_m=F["rib_offset"],
        jaw_gap_m="2 q + %.7f at the plates, 2 q + %.7f at the ribs"
                  % (2 * F["plate_offset"], 2 * F["rib_offset"]),
        joint_value_m=FAT_FINGER_FIX,
        joint_value_why="the tightest the BARE blade can close on the mount "
                        "post's 50 mm ends before the rib fouls it (25.000 "
                        "less the rib's own 8.0663).  Bisecting the blade "
                        "against the committed holder meshes returns the same "
                        "16.9337 mm and names the rib crest as the touching "
                        "vertex.  NOT a claim about the arms — see "
                        "open_questions.penholder_cradle.",
        locates_post=F["locates_post"],
        locates_post_why=F["locates_post_why"],
        grasp_width_hypotheses_m=rig_final.fatfinger_widths(),
        grasp_width_note="the running GUI commands width 0.0432 with "
                         "epsilon_inner 0.0.  Only the two grips on the "
                         "post's BARE 50 mm ends clear that — and a bare post "
                         "end is a flat 26 mm square with nothing to key "
                         "into, so the pen's lean is set at grasp time.",
        collision=dict(
            kind="4 axis-aligned boxes, panda_leftfinger frame",
            why="a printed Z-bracket is not a collision geometry and neither "
                "arm collision model has anything to say about it: "
                "finger.obj is the wrong shape and selfcoll.BODY_CAPSULES has "
                "no finger row at all",
            boxes_m=q.get("collision_boxes_m"),
            worst_escape_m=q.get("collision_worst_escape_m"),
            method="measured off the vendored mesh on every run, then "
                   "re-proved against its vertices, face centroids and edge "
                   "midpoints"),
        decimation=dict(faces_raw=q.get("faces_raw"),
                        faces_out=q.get("faces_out"),
                        extent_loss_mm=q.get("extent_loss_mm")),
    )


def manifest(out_dir=None):
    out_dir = Path(out_dir or OUT_DIR)
    h = SM.H_MOUNT
    mesh_src = out_dir / "meshes/MESH_SOURCES.json"
    if not mesh_src.is_file():
        # emitting a manifest with no provenance record at all, quietly, is
        # the one failure this file exists to make impossible
        raise SystemExit(f"{mesh_src} is missing — run "
                         "`gen_system_model.py meshes` before `urdf`")
    meshes = json.loads(mesh_src.read_text())
    bodies = []
    recert_worst = 0.0
    for b in SM.bodies(h):
        esc = SM.recert_escape_mm(b, h)
        rec = {}
        if esc is not None and esc > 0.0:
            recert_worst = max(recert_worst, esc)
            rec = dict(status="RE-CERT-PENDING",
                       escapes_modelled_envelope_mm=esc,
                       envelope="mounts.arm_mount_boxes — the schematic "
                                "0.226x0.190x0.05 plate and the 0.2x0.2 "
                                "column every certified number was earned "
                                "against")
        elif esc is not None:
            rec = dict(status="inside the modelled envelope")
        bodies.append(dict(
            name=b.name, kind=b.kind, provenance=b.provenance,
            size_mm=list(b.size), centre_mm=list(b.centre),
            lo_mm=list(b.lo), hi_mm=list(b.hi),
            collision=b.collision, source=b.source,
            **({"note": b.note} if b.note else {}), **rec))
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
            transform="rig_final.penholder22_T_hand(PEN_EXT_HOLDER=%.7f, "
                      "PEN_LAT_HOLDER=%.7f, D_HAND_TCP=%.4f)"
                      % (PEN_EXT_HOLDER, PEN_LAT_HOLDER, D_HAND_TCP),
            provenance="ASSUMED",
            red_flag="THE PLACEMENT IS INFERRED.  The 2026.08.19 delivery has "
                     "no assembly file.  The housing's own flats clock at "
                     "23.0 deg; the planner's TCP-to-tip ray leans 45.0 deg; "
                     "this model uses the planner's ray.  THE 10-DEG ASSEMBLY "
                     "HAS NOW BEEN READ and it removes the fingertip cradle "
                     "as the place that difference could hide: the stock FR3 "
                     "fingertips seat square in the post's own 18x18x7 mm "
                     "sockets, and on that build the file's name-angle IS the "
                     "lean to 4 decimals with the grip centre on the TCP to "
                     "0.14 mm.  PENDING one measurement on the mounted "
                     "holder — see open_questions.penholder_cradle.",
            end_for_end_fixed="2026-09-03.  The housing used to be mounted "
                              "END-FOR-END: penholder22_T_hand pointed its "
                              "+X away from the tip, so the model put the "
                              "17.00 mm tail land 55.099 mm toward the paper "
                              "and the cap 30.001 mm back toward the wrist.  "
                              "The pen leaves through the CAP and the model "
                              "now does too — a 180 deg rotation about the "
                              "post axis, the grip centre and the pen tip "
                              "unmoved.  COST, measured, not absorbed: the "
                              "two lateral tool capsules "
                              "(rig_final.STATIC_CAPSULES_LAT, r 0.050) "
                              "CONTAINED the old placement by 1.720 mm and do "
                              "NOT contain this one — housing+cap escape by "
                              "6.546 mm (bracket r 0.0565 needed) and the "
                              "pencil tail by 77.661 mm (r 0.1277).  No "
                              "capsule radius was widened; see "
                              "docs/SYSTEM_MODEL.md 7c.",
            collision="four cylinders: the 3-cylinder housing envelope, "
                      "re-proved on every mesh regeneration to contain every "
                      "visual vertex, plus the pencil tail's own",
            envelope_cylinders=[list(c) for c in
                                rig_final.penholder22_hull()],
            finger_half_width_m=FINGER_FIX,
            finger_half_width_source="AUDIT: post 50 mm less 2 x 7.000 mm of "
                                     "socket = 36.0008 mm of jaw in the "
                                     "10-deg assembly; was 0.0285, which is "
                                     "the fingertip's back face",
            fat_finger=_fat_finger_manifest(meshes),
            internals=dict(
                provenance="AUDIT",
                order="tail shoulder -> spring -> [shim] -> sleeve (clutch "
                      "inside it) -> cap shoulder -> THE PEN LEAVES.  The "
                      "graphite runs right through the barrel and out both "
                      "ends: pen_lead past the cap, graphite_tail past the "
                      "tail face (72.514 mm, measured off the assembly)",
                order_source='the resolved component transforms of "Natural '
                             'hold assembly - closed.SLDASM" (10-deg build, '
                             "same architecture); every interface then "
                             "re-measured on this delivery's own STLs",
                spacer_fitted_m=rig_final.PENHOLDER22["spacer_fitted"],
                spacer_choices_m=list(rig_final.PENHOLDER22["spacers"]),
                envelope_escape_m=rig_final.penholder22_internals_escape(),
                collision=False,
                bodies=[dict(name=n, radius_m=r, length_m=L, note=note)
                        for n, _, (r, L), note
                        in rig_final.penholder22_internals(
                            PEN_EXT_HOLDER, PEN_LAT_HOLDER, D_HAND_TCP)])),
        cable_dress=dict(
            provenance="ASSUMED",
            radius_mm=SM.CABLE_R,
            estimate="20-40 mm, the collision audit's",
            collision=False,
            note="ESTIMATED, visual only, translucent.  Nothing about the "
                 "real dress has been measured."),
        meshes=meshes,
        recert=dict(
            bodies_pending=sum(1 for b in bodies
                               if b.get("status") == "RE-CERT-PENDING"),
            worst_escape_mm=round(recert_worst, 2),
            what="every piece of mount hardware reaches outside the schematic "
                 "keep-out `mounts.arm_mount_boxes` models, measured per body "
                 "in `bodies[].escapes_modelled_envelope_mm`",
            note="INCLUDING THE PLATE.  The re-issued layout sheet reads the "
                 "plate as inside because it compared thicknesses (12.7 real "
                 "against 50 modelled); in PLAN it escapes by 25.06 mm, "
                 "because the modelled plate is centred on the J1 axis and "
                 "the real one is offset 25.15 mm off it."),
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


def write_urdfs(out_dir=None, fingers="both"):
    """Write the URDFs and the manifest.  `fingers` is stock | fat | both.

    The default writes all four, which is what a clean-room rebuild has to
    reproduce.  `--fingers stock` writes only the three the stock build needs
    and `--fingers fat` only the variant, so either can be regenerated alone
    without touching the other's bytes.
    """
    out_dir = Path(out_dir or OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    if fingers not in FINGER_VARIANTS + ("both",):
        raise SystemExit(f"--fingers must be one of "
                         f"{FINGER_VARIANTS + ('both',)}, not {fingers!r}")
    want = FINGER_VARIANTS if fingers == "both" else (fingers,)
    files = (("installation.urdf", True, "mesh", "stock"),
             ("installation_capsules.urdf", True, "capsule", "stock"),
             ("environment.urdf", False, "mesh", "stock"),
             ("installation_fatfingers.urdf", True, "mesh", "fat"))
    boxes = fat_boxes_from(out_dir) if "fat" in want else None
    wrote = []
    for fname, with_arms, coll, fng in files:
        if fng not in want:
            continue
        tree = build(with_arms, coll, fng, boxes)
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
    fingers = "both"
    if "--fingers" in argv:
        i = argv.index("--fingers")
        if i + 1 >= len(argv):
            raise SystemExit(__doc__)
        fingers = argv[i + 1]
        del argv[i:i + 2]
    stage = argv[0] if argv else "all"
    if stage not in ("meshes", "urdf", "all"):
        raise SystemExit(__doc__)
    if stage in ("meshes", "all"):
        files, quality, fat = build_meshes(OUT_DIR)
        tot = sum(f["bytes"] for f in files)
        print(f"meshes: {len(files)} files, {tot / 1e6:.2f} MB")
        for q in quality["parts"]:
            print(f"  holder {q['part']:8s} {q['faces_raw']:>7d} -> "
                  f"{q['faces_out']:>5d} faces, extent lost "
                  f"{q['extent_loss_mm']:.4f} mm")
        print(f"  holder envelope worst escape "
              f"{quality['envelope_worst_escape_m'] * 1000:+.6f} mm")
        print(f"  fat finger      {fat['faces_raw']:>7d} -> "
              f"{fat['faces_out']:>5d} faces, extent lost "
              f"{fat['extent_loss_mm']:.4f} mm, "
              f"{len(fat['collision_boxes_m'])} collision boxes, escape "
              f"{fat['collision_worst_escape_m'] * 1000:+.6f} mm")
    if stage in ("urdf", "all"):
        write_urdfs(OUT_DIR, fingers)


if __name__ == "__main__":
    main()
