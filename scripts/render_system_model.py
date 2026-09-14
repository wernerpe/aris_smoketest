#!/usr/bin/env python3
"""Render `assets/system_model/` — offscreen VTK stills, and a meshcat scene.

    /home/franka/git/franka_manipulation_station/.venv/bin/python \
        scripts/render_system_model.py [--out out] [--width 2400]

Writes to `out/` (gitignored — these are a look, not an artefact):

    system_model_three_quarter.png   the whole installation from above-corner
    system_model_elevation.png       square on the long side, the z ladder
    system_model_plan.png            from above, the 2x3 grid and the cage
    system_model_drop_cluster.png    one drop cluster: posts, plate, clamp
                                     stack, gussets, the arm and its holder
    system_model_holder.png          the pen holder alone, on the hand
    system_model_holder_side.png     the same, straight down the hand's x axis:
                                     fingers left and right, the barrel above
                                     the grip and the graphite below it, which
                                     is the shot that says which way round the
                                     housing is (docs/SYSTEM_MODEL.md 7c)
    system_model.html                a static meshcat scene of everything

The arms are posed at `Q_PARK_PROPOSED` — the certified park poses, which is
what the rig actually looks like standing idle.  Shadows and PBR are on, which
is only worth anything because the textures are now really there.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pydrake.geometry import (ClippingRange, ColorRenderCamera,  # noqa: E402
                              DepthRange, DepthRenderCamera, LightParameter,
                              MakeRenderEngineVtk, Meshcat, MeshcatVisualizer,
                              RenderCameraCore, RenderEngineVtkParams, Rgba)
from pydrake.math import RigidTransform, RotationMatrix  # noqa: E402
from pydrake.multibody.parsing import Parser  # noqa: E402
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph  # noqa: E402
from pydrake.systems.framework import DiagramBuilder  # noqa: E402
from pydrake.systems.sensors import CameraInfo, RgbdSensor  # noqa: E402
from PIL import Image  # noqa: E402

from aris_sixarm import rig_final  # noqa: E402
from aris_sixarm import system_model as SM  # noqa: E402
from aris_sixarm.frames import (D_HAND_TCP, PEN_EXT_HOLDER,  # noqa: E402
                                PEN_LAT_HOLDER)
from aris_sixarm.layout import FLEET_PROPOSED, Q_PARK_PROPOSED  # noqa: E402

# WHERE THE TOOL ACTUALLY IS, and the two close-ups aim at it rather than at
# the hand's own axis.  Since 2026-09-04 the holder is clamped at the far end
# of the Fat finger plates, 66.5 mm out along hand x — a camera still pointed
# down the wrist centreline puts the subject at the edge of its own frame.
GRIP_HAND = np.array([rig_final.PENHOLDER22["grip_hand_x"], 0.0, D_HAND_TCP])
TIP_HAND = np.array([PEN_LAT_HOLDER, 0.0, D_HAND_TCP + PEN_EXT_HOLDER])


def _rel(p):
    """Path for the log — relative to the repo when it is inside it.

    `--out` may point anywhere; `relative_to` raises rather than falling back,
    and raising after a successful render is a bad trade.
    """
    try:
        return Path(p).relative_to(ROOT)
    except ValueError:
        return Path(p)

URDF = ROOT / "assets/system_model/installation.urdf"
TAG = ""          # inserted into every output name; see --tag
MM = SM.MM
CW, CL = SM.CANVAS_W / MM, SM.CANVAS_L / MM     # 1.8034 x 3.63064 m
MID = np.array([CW / 2, CL / 2, 0.45])


def look_at(eye, target, up=(0, 0, 1)):
    """World<-camera for a camera at `eye` pointing at `target`.

    Drake's camera looks down its own +z with +y down the image.  `up` must
    not be parallel to the view ray — a camera looking straight down the world
    z with up = +z produces a zero cross product and a NaN pose, which renders
    as an empty frame rather than an error.
    """
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    f = target - eye
    f /= np.linalg.norm(f)
    up = np.asarray(up, float)
    if abs(float(np.dot(f, up / np.linalg.norm(up)))) > 0.999:
        raise ValueError(f"look_at: up {up} is parallel to the view ray {f}")
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    return RigidTransform(RotationMatrix(np.column_stack([r, np.cross(f, r),
                                                          f])), eye)


# (name, eye, target, vertical field of view in degrees)
VIEWS = (
    ("three_quarter", (4.30, -2.30, 3.10), (CW / 2, CL / 2 - 0.15, 0.55), 42),
    ("elevation", (8.20, CL / 2, 0.72), (CW / 2, CL / 2, 0.55), 29),
    # a HIGH OBLIQUE, not a true plan: straight down, the runway beams roof
    # the arms over completely and the picture is six grey rectangles
    ("plan", (CW / 2 + 1.15, CL / 2 - 3.30, 5.70), (CW / 2, CL / 2, 0.35), 44),
    # A DROP CLUSTER FROM OUTSIDE THE FRAME — from inside, the gussets fill
    # the picture.  The subject is ARM 13, row 0, west column: axis
    # (0.5967, 0.60511), and the eye stands off it exactly as it stood off
    # arm 31 before 2026-09-14.
    #
    # WHY IT IS NO LONGER ARM 31.  The seam bar stands on the MIDDLE ROW's own
    # line, at the frame's west corner — 0.79 m due west of arm 31's axis and
    # 1.65 m tall.  Any eye west of the frame aimed at that cluster looks
    # along the row and the bar is between the two: at this camera's 30 deg it
    # sat 15.8 deg off axis, inside the 22 deg half-width, and filled the left
    # third of the frame with a grey slab.  A sweep of every eye outside the
    # frame between 2.4 and 3.2 m and 10 to 40 deg of elevation clears the bar
    # only from the SOUTH, where arm 13's own cluster then stands in the way.
    # The cluster is the same part on every row, so the shot moved one row
    # instead — the picture is of the hardware, not of arm 31.
    ("drop_cluster", (-1.55, -0.59, 2.05), (0.62, 0.57, 1.12), 30),
    ("holder", None, None, 33),      # framed on the arm's own hand, below
    # THE ONE VIEW THAT SETTLES WHICH WAY ROUND THE HOLDER IS.  Straight down
    # the hand's own x axis, so the finger-travel axis (hand y) lies across the
    # image and the approach axis (hand z) runs down it: the two fingers left
    # and right, the mount post between them, and the barrel and the graphite
    # separated top from bottom.  The bore is in the hand's x-z plane, so this
    # projection foreshortens it by cos 45 deg and NOTHING ELSE — the 55.1 mm
    # of barrel behind the grip reads as 39 mm of barrel ABOVE the fingers,
    # with the cap, the graphite and the pen tip below them.  Mounted
    # end-for-end (as this model was until 2026-09-03, docs/SYSTEM_MODEL.md 7c)
    # the same shot puts the fat end below the fingers instead, which is why it
    # is worth its own camera.
    ("holder_side", None, None, 30),
    # THE PHOTO'S OWN VIEWPOINT (2026-09-03, docs/SYSTEM_MODEL.md 7e).  Pete's
    # photograph of the real gripper is an oblique from slightly BELOW the
    # hand, looking roughly along the hand's x axis, so the jaw axis lies
    # left-to-right in the frame.  That is the shot in which the Fat blades
    # read as a V — feet apart at the carriages, plates together at the
    # paper — and in which the pencil's tail is seen standing out of the back
    # of the housing toward the wrist.  Rendering the model from the same
    # place is how "does the model agree with the photograph" stops being a
    # matter of opinion.  From the -x side, where the barrel and the tail
    # lean; "below" means further along +z_hand, which is toward the paper.
    ("photo", None, None, 34),
    # THE VIEW PETE JUDGES THE TOOL FROM.  Straight down the JAW axis
    # (hand y), aimed between the grip and the tip, with up = -z_hand: the
    # bore's lean and the graphite's protrusion are both IN the image plane
    # here and neither is foreshortened, which is what makes it the shot to
    # hold against a photograph.  `holder_side` looks down hand X and shows
    # the V of the blades; this one looks down hand Y and shows the ANGLE.
    ("holder_jaw", None, None, 32),
)

LIGHTS = [
    LightParameter(type="point", color=Rgba(1.0, 0.99, 0.96),
                   intensity=0.75, position=[-1.6, -1.2, 3.2],
                   attenuation_values=[1, 0, 0], frame="world"),
    LightParameter(type="point", color=Rgba(0.92, 0.95, 1.0),
                   intensity=0.55, position=[3.4, 4.8, 2.8],
                   attenuation_values=[1, 0, 0], frame="world"),
    LightParameter(type="point", color=Rgba(1.0, 1.0, 1.0),
                   intensity=0.35, position=[3.0, -1.5, 0.9],
                   attenuation_values=[1, 0, 0], frame="world"),
]


def build():
    b = DiagramBuilder()
    plant, sg = AddMultibodyPlantSceneGraph(b, time_step=0.0)
    Parser(plant).AddModels(str(URDF))
    plant.Finalize()
    params = RenderEngineVtkParams(
        default_clear_color=[0.94, 0.945, 0.95],
        lights=LIGHTS, cast_shadows=True, shadow_map_size=2048,
        exposure=1.15, force_to_pbr=True)
    sg.AddRenderer("vtk", MakeRenderEngineVtk(params))
    return b, plant, sg


def camera(width, height, fov_deg):
    core = RenderCameraCore("vtk", CameraInfo(width, height,
                                              np.deg2rad(float(fov_deg))),
                            ClippingRange(0.05, 60.0), RigidTransform())
    return (ColorRenderCamera(core, False),
            DepthRenderCamera(core, DepthRange(0.05, 50.0)))


def pose_fleet(plant, ctx):
    for aid in FLEET_PROPOSED:
        q = Q_PARK_PROPOSED[aid]
        for i in range(7):
            plant.GetJointByName(f"arm{aid}_panda_joint{i + 1}").set_angle(
                ctx, float(q[i]))


def stills(out_dir, width, height, views=None):
    want = None if not views else set(views)
    b, plant, sg = build()
    # the holder close-up needs a pose to aim at, so build the plant once,
    # solve the fleet, then read arm 31's hand out of it
    probe = b.Build().CreateDefaultContext()
    pctx = plant.GetMyContextFromRoot(probe)
    pose_fleet(plant, pctx)
    X_hand = plant.EvalBodyPoseInWorld(pctx,
                                       plant.GetBodyByName("arm31_panda_hand"))
    hand = X_hand.translation()
    R_hand = X_hand.rotation().matrix()
    tip = plant.EvalBodyPoseInWorld(
        pctx, plant.GetBodyByName("arm31_pen_tip")).translation()

    b, plant, sg = build()
    sensors = {}
    for name, eye, target, fov in VIEWS:
        if want is not None and name not in want:
            continue
        up = (0.0, 0.0, 1.0)
        if name == "holder":
            # Frame the hand-plus-holder-plus-graphite, about 0.19 m of
            # subject.  Hold the EYE at a fixed 0.50 m from the subject centre
            # rather than at a fixed offset from a moving point — otherwise
            # re-weighting the centre silently re-zooms the shot.
            ctr = hand + 0.45 * (tip - hand)
            d = np.array([0.30, -0.26, 0.14])
            eye = ctr + 0.50 * d / np.linalg.norm(d)
            target = ctr
        elif name == "holder_side":
            # Down the hand's OWN x axis, from the side the pen leans toward.
            # `up` is -z_hand, which puts the wrist at the top of the frame and
            # the paper at the bottom whatever the arm is doing — and it is
            # perpendicular to the view ray by construction, so `look_at`'s
            # parallel-up rule cannot be tripped by a park pose changing.
            # ...and from the -x side, which is the side the barrel and the
            # pencil tail lean to.  From +x the hand's own body stands in
            # front of both of them and the shot proves nothing.
            ctr = hand + R_hand @ (GRIP_HAND + np.array([0.0, 0.0, 0.020]))
            eye = ctr - 0.50 * R_hand[:, 0]
            target, up = ctr, -R_hand[:, 2]
        elif name == "holder_jaw":
            # centred on the HOLDER's own middle — the tail face is 55.1 mm
            # behind the grip and the tip 65.0 in front, so that is the grip
            # plus a whisker down the bore, not the grip-to-tip midpoint
            u = (TIP_HAND - GRIP_HAND) / np.linalg.norm(TIP_HAND - GRIP_HAND)
            ctr = hand + R_hand @ (GRIP_HAND + 0.005 * u)
            eye = ctr + 0.42 * R_hand[:, 1]
            target, up = ctr, -R_hand[:, 2]
        elif name == "photo":
            # the grip centre, seen from -x and from below (+z_hand), the way
            # the photograph was taken.  `up` is -z_hand as in holder_side, so
            # the wrist is at the top of the frame and the paper at the bottom.
            ctr = hand + R_hand @ (GRIP_HAND + np.array([-0.030, 0.0, 0.008]))
            eye = ctr + R_hand @ np.array([-0.300, 0.045, 0.140])
            target, up = ctr, -R_hand[:, 2]
        w, h = (width, height)
        cc, dc = camera(w, h, fov)
        s = b.AddSystem(RgbdSensor(sg.world_frame_id(),
                                   look_at(eye, target, up), cc, dc))
        b.Connect(sg.get_query_output_port(), s.query_object_input_port())
        sensors[name] = s
    dia = b.Build()
    root = dia.CreateDefaultContext()
    pose_fleet(plant, plant.GetMyContextFromRoot(root))
    for name, s in sensors.items():
        img = s.color_image_output_port().Eval(s.GetMyContextFromRoot(root))
        p = out_dir / f"system_model_{TAG}{name}.png"
        Image.fromarray(np.asarray(img.data)[:, :, :3]).save(p)
        print(f"wrote {_rel(p)}  ({p.stat().st_size / 1e6:.2f} MB)")


def meshcat_html(out_dir):
    b = DiagramBuilder()
    plant, sg = AddMultibodyPlantSceneGraph(b, time_step=0.0)
    Parser(plant).AddModels(str(URDF))
    plant.Finalize()
    m = Meshcat()
    MeshcatVisualizer.AddToBuilder(b, sg, m)
    dia = b.Build()
    root = dia.CreateDefaultContext()
    pose_fleet(plant, plant.GetMyContextFromRoot(root))
    dia.ForcedPublish(root)
    p = out_dir / "system_model.html"
    p.write_text(m.StaticHtml())
    print(f"wrote {_rel(p)}  ({p.stat().st_size / 1e6:.1f} MB)")


def main():
    global URDF, TAG
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--width", type=int, default=2400)
    ap.add_argument("--no-html", action="store_true")
    # the fat-finger variant is a SECOND scene, not a replacement, so it gets
    # its own file names rather than overwriting the stock renders
    ap.add_argument("--urdf", default=None,
                    help="which scene to render (default installation.urdf; "
                         "installation_fatfingers.urdf is the other one)")
    ap.add_argument("--tag", default="",
                    help="inserted into every output name, e.g. 'fatfingers_'")
    ap.add_argument("--views", default="",
                    help="comma-separated subset of "
                         + ",".join(v[0] for v in VIEWS))
    a = ap.parse_args()
    if a.urdf:
        URDF = Path(a.urdf) if Path(a.urdf).is_absolute() else ROOT / a.urdf
    TAG = a.tag
    out_dir = ROOT / a.out
    out_dir.mkdir(parents=True, exist_ok=True)
    stills(out_dir, a.width, int(a.width * 0.66),
           [v for v in a.views.split(",") if v])
    if not a.no_html:
        meshcat_html(out_dir)


if __name__ == "__main__":
    main()
