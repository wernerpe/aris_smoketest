"""The Drake plant: one FR3, welded at its mount, over a sheet of paper.

MODEL, AND WHY EACH PIECE IS WHAT IT IS
  ROBOT.  `assets/franka_description/urdf/panda_arm_hand.urdf` -- the same
  vendored file `scripts/gen_proposed_rig_urdf.py` clones into every rig URDF
  in this repo, so the sim's kinematics ARE the planner's: Drake's `panda_hand`
  frame reproduces `frames.link_frames_many(...)[:, 9]` and the nominal tip
  reproduces `frames.tip_pos` to ~6e-12 m (pinned by tests/test_sil_plant.py).
  Inertias are the vendored Panda's, untouched; no FR3 inertia source exists
  in this repo and none is invented here.

  STRIPPED.  Visual and collision geometry are removed from the arm before
  parsing, and the two finger joints are welded shut.  The SIL asks one
  question -- what the pen tip does against the paper -- and the ONLY proximity
  pair in the scene is the tip sphere against the sheet.  Self-collision and
  arm-vs-structure are `aris_sixarm.validate` / `selfcoll` / the rig URDFs'
  job, and the vendored arm's 402 collision spheres are explicitly unaudited
  (see the warning in `gen_proposed_rig_urdf.py`), so letting them answer a
  contact question here would be worse than not asking them.

  TIP.  Two frames ride the hand.  `pen_tip_nominal` is what the robot
  believes and what `o_t_ee` reports (contract §4); the controller's Jacobian
  is taken there.  `pen_tip_actual` is nominal + `tip_error_m` along EE +Z and
  carries the only robot-side collision geometry: a sphere of radius
  `tip_radius_m` whose centre sits one radius BACK along the pen axis, so the
  sphere's forward pole is exactly the physical tip and contact begins exactly
  when that point reaches the sheet.

  PAPER.  A box welded to the world, top face at `paper.z_world`; it stands
  for the table as much as the sheet, so it is deep (`Paper.thickness`).

  CONTACT: COMPLIANT POINT CONTACT, not hydroelastic.  A 1 mm sphere on a flat
  sheet is a point contact in the physical sense -- one normal, one patch
  smaller than the discretisation any hydroelastic mesh could carry -- so the
  pressure field would be integrated over nothing and would only add mesh
  parameters to tune.  Point contact instead exposes the two numbers that
  matter directly (`point_stiffness` N/m, Hunt-Crossley `dissipation` s/m) and
  they are `params.Paper` fields.  The default 1e5 N/m puts 0.08 mm of
  penetration under 8 N: two orders below the press depths this study is
  about, i.e. stiff enough that the paper is not the compliance being measured,
  soft enough that SAP is comfortable at 1 ms.

  SOLVER.  Discrete plant at `params.dt` (1 ms, the controller's own tick),
  discrete contact approximation `kLagged`.  NOT `kSap`, and the difference is
  not cosmetic: SAP models dissipation as a LINEAR relaxation time (default
  0.1 s), which at 5e4 N/m is a 5 kNs/m dashpot across the contact, and this
  study's own 10 cm stroke chatters against it -- 0 to 32 N at 45 % touch,
  with the tip vibrating +-5 um about the sheet.  `kLagged` (and `kSimilar`,
  which agrees with it to 0.7 % here) uses the Hunt-Crossley model, i.e. the
  `dissipation` parameter this package exposes, and the same stroke comes out
  at a steady 7.84 N with 100 % touch.  If SAP is ever wanted here, its
  `relaxation_time` proximity property has to be set deliberately first.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydrake.geometry import (AddContactMaterial, Box, ProximityProperties,
                              Sphere)
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import (AddMultibodyPlantSceneGraph,
                                     CoulombFriction,
                                     DiscreteContactApproximation)
from pydrake.multibody.tree import FixedOffsetFrame, JacobianWrtVariable
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder

from ..frames import FR3_MAX, FR3_MIN, QD_MAX, TAU_MAX
from .params import SilParams
from .rotations import quat_from_matrix

REPO_ROOT = Path(__file__).resolve().parents[2]
FR3_URDF = REPO_ROOT / "assets/franka_description/urdf/panda_arm_hand.urdf"
TIP_RADIUS_M = 0.001


def sil_urdf_xml(path=FR3_URDF, strip_visual: bool = True,
                 strip_collision: bool = True) -> str:
    """-> the vendored arm URDF as a string, stripped and re-limited for the SIL.

    Removes `<transmission>` (the SIL drives generalized forces, not
    actuators), welds `panda_finger_joint1/2` shut (the pen holder is clamped;
    the fingers are not a degree of freedom of this problem) and removes the
    geometry roles asked for.  With both stripped the file has no external mesh
    references at all, so it is handed to the parser as a string and no
    temporary file is ever written.

    FR3, NOT PANDA, ON THE LIMITS.  The vendored file carries PANDA joint
    limits, which are the wrong robot -- `gen_proposed_rig_urdf.py` rewrites
    them for the same reason and this does it the same way, from
    `frames.FR3_MIN/MAX` (position), `QD_MAX` (velocity) and `TAU_MAX`
    (effort).  It matters here more than it does in a static URDF: Drake's
    discrete plant ENFORCES position limits as constraints, so with the
    vendored numbers a failing run walks 0.56 rad past an FR3 stop and keeps
    going, in a place where the real robot would have tripped its reflex.
    Measured on the arm-31 v18 pathway before this was fixed: joint 6 at
    -0.56 rad of slack for three consecutive strokes.
    """
    root = ET.parse(path).getroot()
    for link in root.findall("link"):
        for tag in (("visual",) if strip_visual else ()) + \
                (("collision",) if strip_collision else ()):
            for elem in list(link.findall(tag)):
                link.remove(elem)
    for tr in root.findall("transmission"):
        root.remove(tr)
    joint_index = 0
    for joint in root.findall("joint"):
        if "finger" in joint.get("name", ""):
            joint.set("type", "fixed")
            for tag in ("limit", "axis", "dynamics"):
                elem = joint.find(tag)
                if elem is not None:
                    joint.remove(elem)
        elif joint.get("type") == "revolute":
            limit = joint.find("limit")
            limit.set("lower", repr(float(FR3_MIN[joint_index])))
            limit.set("upper", repr(float(FR3_MAX[joint_index])))
            limit.set("velocity", repr(float(QD_MAX[joint_index])))
            limit.set("effort", repr(float(TAU_MAX[joint_index])))
            joint_index += 1
    if joint_index != 7:
        raise ValueError(f"{path}: expected 7 revolute joints, got {joint_index}")
    return ET.tostring(root, encoding="unicode")


@dataclass
class ContactSample:
    """What the paper did to the pen this tick."""

    normal_force_n: float        # N along the paper normal, + = paper pushing
    penetration_m: float         # m, contact-query depth (>= 0)
    n_pairs: int


class SilPlant:
    """A built, finalized single-arm plant plus its context and queries.

    Every kinematic query is answered in the ARM BASE frame, because that is
    the frame the controller, the CSV and `o_t_ee` all live in; the world
    (canvas) frame is used only for the paper and for reporting tip heights.
    """

    def __init__(self, params: SilParams, tip_radius_m: float = TIP_RADIUS_M,
                 contact_approximation: str = "kLagged"):
        self.params = params
        self.tip_radius_m = float(tip_radius_m)
        self.contact_approximation = contact_approximation
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(
            builder, time_step=params.dt)
        plant.set_discrete_contact_approximation(
            getattr(DiscreteContactApproximation, contact_approximation))
        Parser(plant).AddModelsFromString(sil_urdf_xml(), "urdf")

        X_wb = RigidTransform(
            RotationMatrix(params.mount.T_world_base[:3, :3]),
            params.mount.T_world_base[:3, 3])
        plant.WeldFrames(plant.world_frame(),
                         plant.GetFrameByName("panda_link0"), X_wb)

        props = ProximityProperties()
        AddContactMaterial(
            dissipation=params.paper.dissipation,
            point_stiffness=params.paper.point_stiffness,
            friction=CoulombFriction(params.paper.friction_static,
                                     params.paper.friction_dynamic),
            properties=props)
        paper = params.paper
        plant.RegisterCollisionGeometry(
            plant.world_body(),
            RigidTransform([paper.center_xy[0], paper.center_xy[1],
                            paper.z_world - 0.5 * paper.thickness]),
            Box(paper.size[0], paper.size[1], paper.thickness), "paper", props)

        hand = plant.GetFrameByName("panda_hand")
        tool = params.tool
        self.frame_nominal = plant.AddFrame(FixedOffsetFrame(
            "pen_tip_nominal", hand, RigidTransform(tool.nominal_tip_hand)))
        self.frame_actual = plant.AddFrame(FixedOffsetFrame(
            "pen_tip_actual", hand, RigidTransform(tool.actual_tip_hand)))
        # centre one radius back along the pen axis: the sphere's forward pole
        # IS the physical tip, so contact starts when that point reaches paper.
        plant.RegisterCollisionGeometry(
            plant.GetBodyByName("panda_hand"),
            RigidTransform(tool.actual_tip_hand
                           - np.array([0.0, 0.0, self.tip_radius_m])),
            Sphere(self.tip_radius_m), "pen_tip", props)

        plant.Finalize()
        self.plant = plant
        self.scene_graph = scene_graph
        self.diagram = builder.Build()
        self.simulator = Simulator(self.diagram)
        self.simulator.Initialize()
        self.context = self.simulator.get_mutable_context()
        self.plant_context = plant.GetMyMutableContextFromRoot(self.context)
        self.frame_base = plant.GetFrameByName("panda_link0")
        self.hand_body = plant.GetBodyByName("panda_hand")
        self._tau_port = plant.get_applied_generalized_force_input_port()
        self._tau_port.FixValue(self.plant_context, np.zeros(plant.num_velocities()))
        self._contact_port = plant.get_contact_results_output_port()
        self._t = 0.0

    # -- state -------------------------------------------------------------
    def set_state(self, q: np.ndarray, dq: np.ndarray | None = None) -> None:
        """Set the 7 joint positions (rad) and velocities (rad/s)."""
        self.plant.SetPositions(self.plant_context, np.asarray(q, float))
        self.plant.SetVelocities(
            self.plant_context,
            np.zeros(7) if dq is None else np.asarray(dq, float))

    @property
    def q(self) -> np.ndarray:
        return self.plant.GetPositions(self.plant_context).copy()

    @property
    def dq(self) -> np.ndarray:
        return self.plant.GetVelocities(self.plant_context).copy()

    @property
    def t(self) -> float:
        """Simulated time, s."""
        return self._t

    # -- kinematics --------------------------------------------------------
    def tip_pose_base(self):
        """The NOMINAL tip in the base frame -> (position (3,) m, xyzw (4,)).

        This is `o_t_ee`: what the robot reports and what the controller
        regulates.  The physical tip is never reported (contract §4).
        """
        X = self.frame_nominal.CalcPose(self.plant_context, self.frame_base)
        return X.translation().copy(), quat_from_matrix(X.rotation().matrix())

    def tip_jacobian_base(self) -> np.ndarray:
        """(6,7) nominal-tip Jacobian in the base frame, [linear; angular].

        `getZeroJacobian(kEndEffector)` order, which is the order `k_cartesian`
        indexes; Drake returns [angular; linear] and the two blocks are
        swapped here, once, so nothing downstream has to remember.
        """
        J = self.plant.CalcJacobianSpatialVelocity(
            self.plant_context, JacobianWrtVariable.kV, self.frame_nominal,
            np.zeros(3), self.frame_base, self.frame_base)
        return np.vstack([J[3:], J[:3]])

    def coriolis(self) -> np.ndarray:
        """(7,) Nm, C(q, dq) dq -- `getCoriolisForceVector`, gravity excluded."""
        return self.plant.CalcBiasTerm(self.plant_context)

    def gravity_compensation(self) -> np.ndarray:
        """(7,) Nm, the torque the hardware layer adds to cancel gravity.

        Drake puts gravity on the RIGHT of `M v_dot + C v = tau + tau_g`, so
        the cancelling torque is `-tau_g`.  libfranka does this inside the
        robot; the controller must not (briefing §3).
        """
        return -self.plant.CalcGravityGeneralizedForces(self.plant_context)

    def tip_world(self, actual: bool = False) -> np.ndarray:
        """(3,) m, the tip in the WORLD (canvas) frame."""
        frame = self.frame_actual if actual else self.frame_nominal
        return frame.CalcPoseInWorld(self.plant_context).translation().copy()

    # -- contact -----------------------------------------------------------
    def contact(self) -> ContactSample:
        """Read the contact results for this tick."""
        results = self._contact_port.Eval(self.plant_context)
        n = results.num_point_pair_contacts()
        if n == 0:
            return ContactSample(0.0, 0.0, 0)
        normal = self.params.paper.normal_world
        f_total, depth = 0.0, 0.0
        hand_index = self.hand_body.index()
        for i in range(n):
            info = results.point_pair_contact_info(i)
            # `contact_force()` is the force ON body B, expressed in world.
            sign = 1.0 if info.bodyB_index() == hand_index else -1.0
            f_total += sign * float(np.dot(info.contact_force(), normal))
            depth = max(depth, float(info.point_pair().depth))
        return ContactSample(f_total, depth, n)

    # -- stepping ----------------------------------------------------------
    def step(self, tau: np.ndarray) -> None:
        """Apply generalized force `tau` (7, Nm) for one `dt` and advance."""
        self._tau_port.FixValue(self.plant_context, np.asarray(tau, float))
        self._t += self.params.dt
        self.simulator.AdvanceTo(self._t)


def build_simulator(params: SilParams, tip_radius_m: float = TIP_RADIUS_M,
                    contact_approximation: str = "kLagged") -> SilPlant:
    """Build the SIL plant for `params`. -> `SilPlant`.

    `contact_approximation` is a `DiscreteContactApproximation` member name;
    see the module docstring on why the default is not `kSap`.
    """
    return SilPlant(params, tip_radius_m, contact_approximation)
