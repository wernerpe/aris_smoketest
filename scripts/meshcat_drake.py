#!/usr/bin/env python3
"""The installation in DRAKE'S meshcat, with the real FR3 assets.

    ARIS_RIG=proposed ARIS_TOOL=lateral python3 scripts/meshcat_drake.py \\
        --npz out/unknown_h0970_home_alt.npz --only-arms 31,71 --loop

then open  http://frankastation.drl.csail.mit.edu:7009/

WHY THIS EXISTS.  `web/viewer/js/scene3d.js` draws the fleet out of three.js
PRIMITIVES — 1 box, 3 cylinders, 3 spheres and some line buffers, and not one
mesh file in the whole `web/` tree.  It is a schematic, and it reads as one:
the arms are stacks of tubes.  This script puts the SAME numbers (the same npz,
the same joint angles) through the same model the renders and the collision
gates use — `assets/system_model/installation_fatfingers.urdf` — in a pydrake
`MultibodyPlant`, and serves it through Drake's own `MeshcatVisualizer`, which
loads the URDF's glTF visuals with their PBR textures.  That is the "other
project" look: these are byte-for-byte the meshes
`~/git/franka_manipulation_station` renders (see `--write-provenance`).

The three.js viewer is untouched and still comes up with the GUI; this is a
second window, not a replacement.

WHAT IS ON THE SCREEN.  Everything the URDF has: the 80/20 cage, the table and
its paper, the seam bars, six inverted FR3s, the Fat Franka Finger blades, the
penholder22 housing and cap, the graphite, and the welded pen tip.  `--npz`
plays a conducted programme on ONE clock at `--rate` real time, and the ink is
drawn as the tip goes pen-down — a poly-line per stroke, from the PLANT's own
tip body rather than from a second kinematic chain, so what you see is the
model's tip and not a recomputation of it.

`--only-arms 31,71` hides the four unmounted arms.  It hides them: it sets
`visible = false` on their meshcat paths and changes NOTHING about the plant,
so the joint angles, the clock and the ink are identical either way.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pydrake.geometry import (Meshcat, MeshcatParams,  # noqa: E402
                              MeshcatVisualizer, Rgba)
from pydrake.multibody.parsing import Parser  # noqa: E402
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph  # noqa: E402
from pydrake.systems.framework import DiagramBuilder  # noqa: E402

URDF = ROOT / "assets/system_model/installation_fatfingers.urdf"
MESH_DIR = ROOT / "assets/system_model/meshes"
PROVENANCE = MESH_DIR / "HQ_FR3_SOURCES.json"

# 7000-7008 belong to other scenes on this machine and 8765 is the GUI.
PORT = 7009

# --- THE HIGH-QUALITY FR3 VISUAL SET ---------------------------------------
#
# link -> the visual mesh the six arms must be wearing.  This is a
# SUBSTITUTION MAP, applied to the URDF text at parse time (`_hq_urdf`), and
# on today's `installation_fatfingers.urdf` it is a NO-OP: `gen_system_model.py`
# already writes these paths, because the decimated `meshes/collision/*.obj`
# set it also vendors is for the gates, not for looking at.  The map is here so
# that (a) the substitution is checked rather than assumed — `--check-meshes`
# fails loudly the day a scene ships with collision hulls in its <visual>, and
# (b) a scene that does carry the low-poly set can be viewed properly without
# regenerating it.
#
# WHERE THEY COME FROM.  `~/git/franka_manipulation_station/assets/
# franka_description/meshes/visual/` — the station project, i.e. "the other
# project where we used the drake meshcat thing".  The geometry `.bin` files
# here are byte-identical to the station's (sha256, recorded by
# `--write-provenance`); the `.gltf` headers differ only in that their image
# table points at the vendored `.png` set with the `.ktx2` entries dropped, so
# a Drake that has no KTX2 reader still gets the textures.  Nothing was
# downloaded: every byte was copied off this machine.
HQ_FR3 = {
    "link0": "meshes/fr3/link0.gltf",
    "link1": "meshes/fr3/link1.gltf",
    "link2": "meshes/fr3/link2.gltf",
    "link3": "meshes/fr3/link3.gltf",
    "link4": "meshes/fr3/link4.gltf",
    "link5": "meshes/fr3/link5.gltf",
    "link6": "meshes/fr3/link6.gltf",
    "link7": "meshes/fr3/link7.gltf",
    "hand": "meshes/fr3/hand.gltf",
}
HQ_UPSTREAM = Path("/home/franka/git/franka_manipulation_station/assets/"
                   "franka_description/meshes/visual")

# what a low-poly scene would have in its <visual> instead, per link
LOWPOLY = {k: f"meshes/collision/{k}.obj" for k in HQ_FR3}

INK_LIFT = 0.0002      # m, above the tip, so the line does not z-fight the
                       # paper's own top face.  0.2 mm: the pen-down tip sits
                       # within 0.03 mm of z = 0, so the drawn line stays
                       # inside the 0.5 mm the check allows.
INK_WIDTH = 3.0
INK_HEX = {"black": "#111111", "red": "#cc2222", "blue": "#2244cc"}
ARM_TINT = {2: "#111111", 13: "#111111", 17: "#111111",
            31: "#111111", 71: "#111111", 97: "#111111"}


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _rel(p):
    try:
        return Path(p).relative_to(ROOT)
    except ValueError:
        return Path(p)


def _rgba(hex_str, alpha=1.0):
    h = hex_str.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return Rgba(r, g, b, alpha)


# --- the model --------------------------------------------------------------

def _visual_meshes(tree):
    """Every <mesh> that is inside a <visual> -> (link name, element).

    STRUCTURAL, not textual.  `meshes/collision/link3.obj` appears in this
    URDF six times over in <collision> elements where it belongs; a string
    search for it would report a substitution on a scene that needs none.
    """
    for link in tree.getroot().iter("link"):
        for vis in link.findall("visual"):
            for mesh in vis.iter("mesh"):
                yield link.get("name"), mesh


def substitutions(urdf):
    """-> {low-poly path: high-quality path} for what `urdf` actually needs.

    `{}` means the scene already wears the good set and must be parsed
    untouched.
    """
    import xml.etree.ElementTree as ET
    return {m.get("filename"): HQ_FR3[LOWPOLY_INV[m.get("filename")]]
            for _, m in _visual_meshes(ET.parse(urdf))
            if m.get("filename") in LOWPOLY_INV}


def _hq_urdf(urdf):
    """-> (path to parse, temp path to delete or None).

    A rewrite is written NEXT TO the original because every mesh path in the
    file is relative to it; a temp file in /tmp would parse to a scene with no
    meshes at all.  `load` removes it.
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(urdf)
    n = 0
    for _, mesh in _visual_meshes(tree):
        hit = LOWPOLY_INV.get(mesh.get("filename"))
        if hit:
            mesh.set("filename", HQ_FR3[hit])
            n += 1
    if not n:
        return Path(urdf), None
    tmp = Path(urdf).with_suffix(".hq.urdf")
    tree.write(tmp)
    print(f"substituted {n} visual mesh(es) for the high-quality FR3 set")
    return tmp, tmp


LOWPOLY_INV = {v: k for k, v in LOWPOLY.items()}


def load(urdf=URDF, meshcat=None):
    """-> (diagram, plant, meshcat, visualizer, model_instance_name)."""
    parse_path, tmp = _hq_urdf(urdf)
    try:
        b = DiagramBuilder()
        plant, sg = AddMultibodyPlantSceneGraph(b, time_step=0.0)
        models = Parser(plant).AddModels(str(parse_path))
        plant.Finalize()
        mi = plant.GetModelInstanceName(models[0])
        vis = MeshcatVisualizer.AddToBuilder(b, sg, meshcat)
        return b.Build(), plant, meshcat, vis, mi
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def arm_ids(plant):
    """The arm prefixes the plant actually has, e.g. [2, 13, 17, 31, 71, 97]."""
    out = set()
    for i in plant.GetJointIndices():
        n = plant.get_joint(i).name()
        if n.startswith("arm") and "_panda_joint" in n:
            out.add(int(n[3:n.index("_panda_joint")]))
    return sorted(out)


def set_q(plant, ctx, aid, q):
    for j in range(7):
        plant.GetJointByName(f"arm{aid}_panda_joint{j + 1}").set_angle(
            ctx, float(q[j]))


def tip_world(plant, ctx, aid):
    return plant.EvalBodyPoseInWorld(
        ctx, plant.GetBodyByName(f"arm{aid}_pen_tip")).translation()


def frame_camera(meshcat, plant, ctx, keep):
    """Open on the arms being SHOWN, not on the middle of a 3.6 m table.

    Meshcat's default camera is a fixed offset from the origin, which for this
    scene is a corner of the paper — Pete's first look would be up the inside
    of the cage.  Aim at the centroid of the shown arms' bases and stand off it
    from the -y side, the same quarter render_system_model's `three_quarter`
    shoots from.
    """
    bases = [plant.EvalBodyPoseInWorld(
        ctx, plant.GetBodyByName(f"arm{a}_panda_link0")).translation()
        for a in sorted(keep)]
    if not bases:
        return
    b = np.asarray(bases)
    c = b.mean(axis=0)
    target = np.array([c[0], c[1], 0.20])
    span = float(np.ptp(b[:, 1])) if len(b) > 1 else 0.0

    # STAY UNDER THE ROOF.  The cage's top beams start 1.62 m above the paper
    # and are 76 mm of solid section: an eye at or above that height looks
    # ALONG the slab and the top half of the picture is a black wedge (which is
    # what the first framing of this did).  Hold the eye under 1.45, stand off
    # down the -y end where the end rail's opening is, and look slightly down.
    back = 2.35 + 0.50 * span
    up = min(1.45, 0.75 + 0.22 * span)
    eye = target + np.array([0.95, -back, up])
    meshcat.SetCameraPose(eye, target)


def hide_arms(meshcat, plant, mi, keep):
    """VISUAL ONLY: `visible = false` on every body of an arm not in `keep`.

    The plant is untouched — those arms still have joints, still get their
    commanded angles, and still appear in the ink bookkeeping.  Hiding is a
    property on a meshcat path and nothing more.
    """
    hidden = 0
    for aid in arm_ids(plant):
        if aid in keep:
            continue
        for body in plant.GetBodyIndices(plant.GetModelInstanceByName(mi)):
            name = plant.get_body(body).name()
            if not name.startswith(f"arm{aid}_"):
                continue
            path = f"/drake/visualizer/{mi}/{name}"
            if meshcat.HasPath(path):
                meshcat.SetProperty(path, "visible", False)
                hidden += 1
    return hidden


# --- the programme ----------------------------------------------------------

def load_programme(npz_path, program_path=None):
    """-> dict with the frames, the clock and the per-arm pen-down flags.

    Deliberately reads the npz DIRECTLY rather than through
    `execute.program.from_schedule`: that builds a FleetProgram with barriers
    and acknowledgements, which is the rehearsal's business.  A viewer wants
    one array of frames on one clock, which is what the npz already is.
    """
    z = np.load(npz_path, allow_pickle=False)
    arms = [int(a) for a in z["arms"]]
    fps = float(z["fps"])
    q = {a: np.asarray(z[f"q_{a}"], float) for a in arms}
    seg = {a: np.asarray(z[f"seg_{a}"], int) for a in arms
           if f"seg_{a}" in z.files}
    n = len(next(iter(q.values())))
    ink = [str(x) for x in z["phase_ink"]] if "phase_ink" in z.files else []
    phase = (np.asarray(z["phase"], int) if "phase" in z.files
             else np.zeros(n, int))
    prog = json.loads(Path(program_path).read_text()) if program_path else {}
    names = [str(p.get("name", "")) for p in (prog.get("phases") or [])]
    return dict(arms=arms, drawing=[int(a) for a in z["drawing_arms"]],
                q=q, seg=seg, n=n, fps=fps, phase=phase, ink=ink,
                phase_names=names, stride=int(z["stride"]),
                duration=n / fps, name=Path(npz_path).stem)


class Ink:
    """The strokes, grown a frame at a time and pushed to meshcat as lines.

    One meshcat path per finished stroke plus one for the stroke in progress,
    so a 100-stroke programme costs 100 objects and one message per drawing arm
    per frame — not one object per frame.
    """

    def __init__(self, meshcat, root="/ink"):
        self.m, self.root = meshcat, root
        self.open = {}        # aid -> (segid, [xyz, ...])
        self.done = {}        # aid -> n finished strokes
        self.polylines = []   # every stroke, for the checks

    def clear(self):
        self.m.Delete(self.root)
        self.open, self.done, self.polylines = {}, {}, []

    def _flush(self, aid, colour):
        seg = self.open.pop(aid, None)
        if seg is None:
            return
        _, pts = seg
        if len(pts) >= 2:
            k = self.done.get(aid, 0)
            self._line(f"{self.root}/arm{aid}/s{k}", pts, colour)
            self.done[aid] = k + 1
            self.polylines.append((aid, np.asarray(pts, float)))

    def _line(self, path, pts, colour):
        v = np.asarray(pts, float).T.copy()       # (3, N), meshcat's order
        self.m.SetLine(path, v, INK_WIDTH, _rgba(colour))

    def step(self, aid, segid, tip, colour):
        """One frame of one arm.  `segid < 0` means the pen is up."""
        cur = self.open.get(aid)
        if segid < 0:
            self._flush(aid, colour)
            return
        if cur is None or cur[0] != segid:
            self._flush(aid, colour)
            cur = (segid, [])
            self.open[aid] = cur
        p = np.asarray(tip, float).copy()
        p[2] += INK_LIFT
        cur[1].append(p)
        if len(cur[1]) >= 2:
            self._line(f"{self.root}/arm{aid}/live{self.done.get(aid, 0)}",
                       cur[1], colour)

    def finish(self, colour_of):
        for aid in list(self.open):
            self._flush(aid, colour_of(aid))


def ink_polylines(plant, ctx, prog):
    """Every stroke of `prog` as world-frame points, WITH the display lift.

    The same points `Ink` pushes to meshcat, computed in one pass and with no
    meshcat in the way — this is what `tests/test_meshcat_drake.py` measures
    against the paper plane.
    """
    out = []
    for aid in prog["drawing"]:
        seg = prog["seg"].get(aid)
        if seg is None:
            continue
        cur, run = None, []
        for i in range(prog["n"]):
            s = int(seg[i])
            if s != cur:
                if len(run) >= 2:
                    out.append((aid, np.asarray(run)))
                cur, run = s, []
            if s < 0:
                continue
            set_q(plant, ctx, aid, prog["q"][aid][i])
            p = tip_world(plant, ctx, aid).copy()
            p[2] += INK_LIFT
            run.append(p)
        if len(run) >= 2:
            out.append((aid, np.asarray(run)))
    return out


# --- provenance -------------------------------------------------------------

def write_provenance(path=PROVENANCE):
    """Record WHICH FR3 meshes this viewer shows and where they came from."""
    files = []
    for link, rel in HQ_FR3.items():
        p = MESH_DIR.parent / rel
        up = HQ_UPSTREAM / f"{link}.bin"
        binp = p.with_suffix(".bin")
        rec = dict(link=link, file=str(_rel(p)), urdf_ref=rel,
                   bytes=p.stat().st_size, sha256=_sha(p),
                   geometry=str(_rel(binp)), geometry_sha256=_sha(binp))
        if up.exists():
            rec["upstream"] = str(up)
            rec["upstream_sha256"] = _sha(up)
            rec["geometry_byte_identical"] = (rec["geometry_sha256"]
                                              == rec["upstream_sha256"])
        files.append(rec)
    doc = dict(
        what="the high-quality FR3 visual meshes scripts/meshcat_drake.py "
             "shows, and the substitution map it applies to get them",
        source_project=str(HQ_UPSTREAM),
        source_note="copied off this machine; nothing was downloaded.  The "
                    "geometry .bin files are byte-identical to the station "
                    "project's; the .gltf headers differ only in pointing at "
                    "the vendored .png textures with the .ktx2 entries "
                    "dropped, so a Drake without a KTX2 reader still textures "
                    "them.",
        substitution=dict(
            low_poly=LOWPOLY, high_quality=HQ_FR3,
            note="applied to <visual> only, at parse time.  A no-op on "
                 "assets/system_model/installation_fatfingers.urdf, which "
                 "already carries the high-quality set — the check is the "
                 "point."),
        urdf=str(_rel(URDF)), files=files)
    path.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {_rel(path)}  ({len(files)} meshes)")
    return doc


def check_meshes(urdf=URDF):
    """-> list of complaints.  Empty means the scene is wearing the good set."""
    import xml.etree.ElementTree as ET
    bad, seen = [], set()
    for link_name, mesh in _visual_meshes(ET.parse(urdf)):
        f = mesh.get("filename", "")
        if f in LOWPOLY_INV:
            bad.append(f"{link_name}: <visual> uses the low-poly {f}")
        if f.startswith("meshes/fr3/"):
            seen.add(f)
    missing = sorted(set(HQ_FR3.values()) - seen)
    if missing:
        bad.append(f"the scene never references {missing}")
    for rel in HQ_FR3.values():
        if not (MESH_DIR.parent / rel).exists():
            bad.append(f"missing on disk: {rel}")
    return bad


# --- playing ----------------------------------------------------------------

def colour_of(prog, i, aid):
    if prog["ink"]:
        k = int(prog["phase"][i])
        if k < len(prog["ink"]):
            return INK_HEX.get(prog["ink"][k], "#111111")
    return ARM_TINT.get(aid, "#111111")


def play(diagram, plant, meshcat, prog, rate, loop, ink=True, record=False,
         vis=None):
    root = diagram.CreateDefaultContext()
    ctx = plant.GetMyContextFromRoot(root)
    pen = Ink(meshcat)
    dt = 1.0 / prog["fps"]

    def one_pass(realtime):
        if ink:
            pen.clear()
        t0 = time.time()
        for i in range(prog["n"]):
            for aid in prog["arms"]:
                set_q(plant, ctx, aid, prog["q"][aid][i])
            root.SetTime(i * dt)
            diagram.ForcedPublish(root)
            if ink:
                for aid in prog["drawing"]:
                    seg = prog["seg"].get(aid)
                    if seg is None:
                        continue
                    pen.step(aid, int(seg[i]), tip_world(plant, ctx, aid),
                             colour_of(prog, i, aid))
            if realtime:
                due = t0 + (i * dt) / max(rate, 1e-6)
                slack = due - time.time()
                if slack > 0:
                    time.sleep(slack)
        if ink:
            pen.finish(lambda a: colour_of(prog, prog["n"] - 1, a))

    if record and vis is not None:
        # The scrub slider: one pass as fast as the machine will go, recorded
        # into the visualiser, then published as a meshcat animation.  The ink
        # is NOT part of the recording — meshcat animates transforms, not
        # objects — so it ends up drawn in full, which is the right still.
        print("recording one pass for the scrub slider...")
        vis.StartRecording(set_transforms_while_recording=False)
        one_pass(realtime=False)
        vis.StopRecording()
        vis.PublishRecording()
        print("recording published: use the animation panel to scrub")
        while True:
            time.sleep(3600)

    n = 0
    while True:
        n += 1
        one_pass(realtime=True)
        print(f"pass {n} done ({prog['duration']:.1f} s of programme)")
        if not loop:
            return
        time.sleep(1.5)


def pose_static(diagram, plant, prog=None):
    root = diagram.CreateDefaultContext()
    ctx = plant.GetMyContextFromRoot(root)
    if prog is not None:
        for aid in prog["arms"]:
            set_q(plant, ctx, aid, prog["q"][aid][0])
        where = "the programme's first frame"
    else:
        from aris_sixarm.layout import FLEET_PROPOSED, Q_PARK_PROPOSED
        for aid in FLEET_PROPOSED:
            if aid in arm_ids(plant):
                set_q(plant, ctx, aid, Q_PARK_PROPOSED[aid])
        where = "Q_PARK_PROPOSED"
    diagram.ForcedPublish(root)
    print(f"posed at {where}")
    return root, ctx


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default=None, help="a conducted programme")
    ap.add_argument("--program", default=None, help="its _program.json, for "
                                                    "phase names only")
    ap.add_argument("--urdf", default=str(URDF))
    ap.add_argument("--port", type=int, default=PORT,
                    help="7000-7008 belong to other scenes, 8765 to the GUI")
    ap.add_argument("--host", default="*",
                    help="'*' = every interface, which is what makes the URL "
                         "work off the box")
    ap.add_argument("--rate", type=float, default=1.0)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--static", action="store_true",
                    help="pose and hold; no programme")
    ap.add_argument("--only-arms", default="",
                    help="e.g. 31,71 — hide the others (VISUAL only)")
    ap.add_argument("--no-ink", action="store_true")
    ap.add_argument("--no-camera", action="store_true",
                    help="leave meshcat's default camera where it is")
    ap.add_argument("--record", action="store_true",
                    help="publish a scrubbable meshcat animation instead of "
                         "playing live")
    ap.add_argument("--write-provenance", action="store_true")
    ap.add_argument("--check-meshes", action="store_true")
    a = ap.parse_args(argv)

    urdf = Path(a.urdf) if Path(a.urdf).is_absolute() else ROOT / a.urdf

    if a.write_provenance:
        write_provenance()
        return 0
    if a.check_meshes:
        bad = check_meshes(urdf)
        for b in bad:
            print("FAIL:", b)
        print("the scene wears the high-quality FR3 set" if not bad
              else f"{len(bad)} problem(s)")
        return 1 if bad else 0

    if 7000 <= a.port <= 7008 or a.port == 8765:
        raise SystemExit(f"port {a.port} belongs to another scene or the GUI; "
                         f"this viewer's port is {PORT}")

    meshcat = Meshcat(MeshcatParams(host=a.host, port=a.port))
    diagram, plant, meshcat, vis, mi = load(urdf, meshcat)
    print(f"meshcat: http://frankastation.drl.csail.mit.edu:{a.port}/")

    prog = None
    if a.npz:
        prog = load_programme(ROOT / a.npz if not Path(a.npz).is_absolute()
                              else a.npz, a.program)
        print(f"programme {prog['name']}: {prog['n']} frames @ "
              f"{prog['fps']:g} fps = {prog['duration']:.1f} s, drawing arms "
              f"{prog['drawing']}")

    _, ctx0 = pose_static(diagram, plant, prog)  # something to look at at once
    keep = {int(x) for x in a.only_arms.replace(",", " ").split()} \
        if a.only_arms else set(arm_ids(plant))
    if keep != set(arm_ids(plant)):
        n = hide_arms(meshcat, plant, mi, keep)
        print(f"hid {n} bodies; showing arms {sorted(keep)} (visual only)")
    if not a.no_camera:
        frame_camera(meshcat, plant, ctx0, keep)

    if a.static or prog is None:
        print("static: holding.  ^C to stop.")
        while True:
            time.sleep(3600)

    play(diagram, plant, meshcat, prog, a.rate, a.loop,
         ink=not a.no_ink, record=a.record, vis=vis)
    print("done; holding the last frame.  ^C to stop.")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    sys.exit(main() or 0)
