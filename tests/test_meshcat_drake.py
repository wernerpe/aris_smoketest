"""`scripts/meshcat_drake.py`: the right meshes, the right tip, ink on paper.

WHAT IS WORTH TESTING HERE.  A viewer is mostly a look, and a look is not
testable.  Three things about this one are:

  * that the six arms are wearing the HIGH-QUALITY FR3 visuals and not the
    decimated collision hulls — which is the whole reason the script exists,
    and which a regenerated scene could silently undo;
  * that the tip the viewer draws from is THE SAME TIP the planner plans to.
    The viewer reads its tip out of the `MultibodyPlant` (the URDF's welded
    `arm<id>_pen_tip` body) while everything upstream uses `frames.tip_pos`;
    if those two ever drift, the ink on the screen stops being evidence;
  * that the ink lands on the paper.

NO ARIS_RIG / ARIS_TOOL ANYWHERE.  The URDF welds the penholder22 tip onto all
six arms at `(PEN_LAT_HOLDER, 0, PEN_EXT_HOLDER)`, so the reference offsets are
passed to `frames.tip_pos` EXPLICITLY rather than read out of the process-wide
active tool.  That is also the stricter test: it pins the model's own geometry
instead of whatever `activate_tool` last did.
"""
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pydrake")

from pydrake.multibody.parsing import Parser                  # noqa: E402
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph  # noqa: E402
from pydrake.systems.framework import DiagramBuilder          # noqa: E402

from aris_sixarm.frames import (PEN_EXT_HOLDER, PEN_LAT_HOLDER,  # noqa: E402
                                tip_pos)

import importlib.util                                         # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "out/unknown_h0970_home_alt.npz"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "meshcat_drake", ROOT / "scripts/meshcat_drake.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MD = _load_module()


@pytest.fixture(scope="module")
def plant_ctx():
    """The scene with no meshcat at all — `load` needs a server, this does not.

    Parsing is a few seconds, so the whole module shares one.
    """
    b = DiagramBuilder()
    plant, _ = AddMultibodyPlantSceneGraph(b, time_step=0.0)
    Parser(plant).AddModels(str(MD.URDF))
    plant.Finalize()
    diagram = b.Build()
    root = diagram.CreateDefaultContext()
    return plant, plant.GetMyContextFromRoot(root)


# --- the meshes -------------------------------------------------------------

def test_the_scene_wears_the_high_quality_fr3_visuals():
    assert MD.check_meshes(MD.URDF) == []


def test_no_substitution_is_needed_on_the_shipped_scene():
    """The substitution map is a CHECK on this scene, not a repair of it.

    If this ever fails, `gen_system_model.py` has started writing collision
    hulls into <visual> and the viewer will quietly rewrite them — which is
    the intended fallback, but it should be noticed.
    """
    assert MD.substitutions(MD.URDF) == {}


def test_substitution_is_structural_not_textual():
    """`meshes/collision/*.obj` in a <collision> must not look like a hit."""
    text = MD.URDF.read_text()
    assert 'filename="meshes/collision/link3.obj"' in text   # it IS in there
    assert "meshes/collision/link3.obj" not in MD.substitutions(MD.URDF)


def test_provenance_records_the_station_project_meshes():
    import json
    doc = json.loads(MD.PROVENANCE.read_text())
    assert "franka_manipulation_station" in doc["source_project"]
    got = {f["link"] for f in doc["files"]}
    assert got == set(MD.HQ_FR3)
    for f in doc["files"]:
        assert (ROOT / f["file"]).exists()
        # every one of the nine is byte-identical to the station's own
        assert f.get("geometry_byte_identical") is True, f["link"]


# --- the tip ----------------------------------------------------------------

def test_plant_tip_agrees_with_frames_tip_pos(plant_ctx):
    """10 random certified poses, plant tip vs `frames.tip_pos`, < 1e-6 m.

    Compared in the arm's own link0 frame, which is what `tip_pos` returns, so
    the check is of the TOOL geometry and does not drag the mount transform in
    with it.
    """
    plant, ctx = plant_ctx
    z = np.load(NPZ)
    arms = [int(a) for a in z["arms"]]
    rng = np.random.default_rng(20260916)
    worst = 0.0
    for _ in range(10):
        aid = int(rng.choice(arms))
        Q = z[f"q_{aid}"]
        q = Q[int(rng.integers(0, len(Q)))]
        MD.set_q(plant, ctx, aid, q)
        X0 = plant.EvalBodyPoseInWorld(
            ctx, plant.GetBodyByName(f"arm{aid}_panda_link0"))
        tip_link0 = X0.inverse() @ MD.tip_world(plant, ctx, aid)
        ref = tip_pos(q, pen_ext=PEN_EXT_HOLDER, pen_lat=PEN_LAT_HOLDER)
        worst = max(worst, float(np.linalg.norm(tip_link0 - ref)))
    assert worst < 1e-6, f"tip disagrees by {worst:.3e} m"


# --- the ink ----------------------------------------------------------------

def test_ink_lies_on_the_paper_plane(plant_ctx):
    """Every drawn point, in every pen-down frame, within 0.5 mm of z = 0.

    World z = 0 IS the paper's top surface in this URDF (`paper_weld` puts the
    2 mm sheet's centre at -0.001), so the tolerance is on |z| directly.  The
    points measured are the ones the viewer actually pushes to meshcat, the
    0.2 mm display lift included.
    """
    plant, ctx = plant_ctx
    prog = MD.load_programme(NPZ)
    lines = MD.ink_polylines(plant, ctx, prog)
    assert lines, "the programme drew nothing"
    worst = max(float(np.abs(p[:, 2]).max()) for _, p in lines)
    assert worst < 5e-4, f"ink stands {worst * 1000:.3f} mm off the paper"


def test_ink_is_one_polyline_per_stroke(plant_ctx):
    """The stroke count out of the ink matches the npz's own `seg` bookkeeping.

    A viewer that dropped or merged strokes would still look plausible; this
    is the only thing that notices.
    """
    plant, ctx = plant_ctx
    prog = MD.load_programme(NPZ)
    lines = MD.ink_polylines(plant, ctx, prog)
    for aid in prog["drawing"]:
        seg = prog["seg"][aid]
        want = len(np.unique(seg[seg >= 0]))
        got = sum(1 for a, _ in lines if a == aid)
        assert got == want, f"arm {aid}: {got} polylines for {want} strokes"


def test_ink_only_grows_while_the_pen_is_down():
    """`Ink.step` opens on seg >= 0, closes on -1, and splits on a new seg."""
    class FakeMeshcat:
        def __init__(self):
            self.lines = {}

        def SetLine(self, path, v, w, rgba):
            self.lines[path] = np.asarray(v)

        def Delete(self, path):
            self.lines.clear()

    m = FakeMeshcat()
    ink = MD.Ink(m)
    for seg, x in [(-1, 0), (0, 1), (0, 2), (0, 3), (-1, 4), (-1, 5),
                   (1, 6), (1, 7)]:
        ink.step(71, seg, np.array([float(x), 0.0, 0.0]), "#111111")
    ink.finish(lambda a: "#111111")
    assert len(ink.polylines) == 2
    assert [len(p) for _, p in ink.polylines] == [3, 2]
    # and the lift is applied to the drawn points, not to the tip
    assert np.allclose(ink.polylines[0][1][:, 2], MD.INK_LIFT)


# --- --only-arms is visual only ---------------------------------------------

def test_only_arms_touches_meshcat_and_nothing_else(plant_ctx):
    """Hiding four arms must not move a joint or drop a body from the plant."""
    plant, ctx = plant_ctx
    before_q = plant.GetPositions(ctx).copy()
    before_n = plant.num_bodies()

    class FakeMeshcat:
        def __init__(self):
            self.hidden = []

        def HasPath(self, p):
            return True

        def SetProperty(self, path, prop, value):
            assert (prop, value) == ("visible", False)
            self.hidden.append(path)

    m = FakeMeshcat()
    mi = "aris_system_model_fatfingers"
    n = MD.hide_arms(m, plant, mi, {31, 71})
    assert n == len(m.hidden) > 0
    assert plant.num_bodies() == before_n
    assert np.array_equal(plant.GetPositions(ctx), before_q)
    # exactly the four unmounted arms, and no body of 31 or 71
    assert not any(("/arm31_" in p or "/arm71_" in p) for p in m.hidden)
    for aid in (2, 13, 17, 97):
        assert any(f"/arm{aid}_panda_link0" in p for p in m.hidden)


def test_arm_ids_finds_the_whole_fleet(plant_ctx):
    plant, _ = plant_ctx
    assert MD.arm_ids(plant) == [2, 13, 17, 31, 71, 97]
