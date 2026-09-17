"""The Drake plant, against `frames` — the sim's tool convention IS the
planner's, or the whole study measures the wrong pen.

Needs the `sil` extra (pydrake).
"""
import numpy as np
import pytest

pytest.importorskip("pydrake")

from aris_sixarm import frames                                  # noqa: E402
from aris_sixarm.sil import (ArmMount, Controller, Paper, SilParams,  # noqa: E402
                             Tool, build_simulator)
from aris_sixarm.sil.plant import sil_urdf_xml                   # noqa: E402

TOOLS = ("inline", "lateral")


def a_params(tool="lateral", tip_error=-0.010, rig="proposed", arm=31):
    mount = ArmMount.from_fleet(rig, arm)
    return SilParams(mount=mount, tool=Tool.from_frames(tool, tip_error),
                     paper=Paper.for_rig(rig), controller=Controller.from_yaml())


def test_urdf_is_stripped_to_seven_dof():
    xml = sil_urdf_xml()
    assert "<collision" not in xml and "<visual" not in xml
    assert "transmission" not in xml
    plant = build_simulator(a_params()).plant
    assert plant.num_positions() == 7 and plant.num_velocities() == 7


@pytest.mark.parametrize("tool", TOOLS)
def test_nominal_tip_frame_reproduces_frames_tip_pos(tool):
    """FK of `pen_tip_nominal` == `frames.tip_pos` at 20 random q, < 1e-6 m.

    This is the pin that says the sim's EE is the planner's EE: same hand
    chain, same `Rz(-pi/4)` flange twist, same tool offset.  The tolerance the
    task asks for is 1e-6 m; what the two actually agree to is ~1e-11.
    """
    params = a_params(tool=tool)
    sim = build_simulator(params)
    lat, ext = params.tool.offset_tcp[0], params.tool.offset_tcp[2]
    rng = np.random.default_rng(20260909)
    worst_p, worst_r = 0.0, 0.0
    for _ in range(20):
        q = rng.uniform(frames.FR3_MIN + 0.05, frames.FR3_MAX - 0.05)
        sim.set_state(q)
        p_base, _ = sim.tip_pose_base()
        worst_p = max(worst_p, float(np.abs(
            p_base - frames.tip_pos(q, pen_ext=ext, pen_lat=lat)).max()))
        X = sim.frame_nominal.CalcPose(sim.plant_context, sim.frame_base)
        worst_r = max(worst_r, float(np.abs(
            X.rotation().matrix() - frames.fk(q)[0][:3, :3]).max()))
    assert worst_p < 1e-6, worst_p
    assert worst_r < 1e-6, worst_r


@pytest.mark.parametrize("tip_error", (0.0, -0.010, +0.004))
def test_actual_tip_is_the_nominal_one_shifted_along_ee_z(tip_error):
    """actual - nominal == tip_error * (EE +Z), exactly, at every pose."""
    params = a_params(tip_error=tip_error)
    sim = build_simulator(params)
    rng = np.random.default_rng(7)
    for _ in range(10):
        q = rng.uniform(frames.FR3_MIN + 0.05, frames.FR3_MAX - 0.05)
        sim.set_state(q)
        d = sim.tip_world(actual=True) - sim.tip_world(actual=False)
        R = sim.frame_nominal.CalcPoseInWorld(sim.plant_context).rotation()
        assert np.allclose(d, tip_error * R.matrix()[:, 2], atol=1e-12)
        assert float(np.linalg.norm(d)) == pytest.approx(abs(tip_error),
                                                         abs=1e-12)


def test_jacobian_is_linear_over_angular_and_matches_metrics():
    """`tip_jacobian_base()` is [linear; angular] (libfranka's
    `getZeroJacobian` order), and its linear block is the repo's own tip
    Jacobian rotated into the base frame -- which it already is."""
    from aris_sixarm.metrics import tip_jacobian
    params = a_params()
    sim = build_simulator(params)
    lat, ext = params.tool.offset_tcp[0], params.tool.offset_tcp[2]
    q = np.array([0.3, -0.8, 0.4, -2.0, 0.2, 1.9, 0.5])
    sim.set_state(q)
    J = sim.tip_jacobian_base()
    assert J.shape == (6, 7)
    assert np.allclose(J[:3], tip_jacobian(q, pen_ext=ext, pen_lat=lat),
                       atol=1e-8)


def test_gravity_compensation_holds_the_arm_still():
    """With -tau_g applied and nothing else, the arm does not move: the
    hardware layer's job, done the way libfranka does it."""
    params = a_params(tip_error=0.0)
    sim = build_simulator(params)
    q0 = np.array([-0.6858, -1.2181, 1.0783, -2.5881, -2.1652, 1.5101, 0.9886])
    sim.set_state(q0)
    for _ in range(500):
        sim.step(sim.gravity_compensation())
    assert float(np.abs(sim.q - q0).max()) < 1e-9
    assert sim.contact().n_pairs == 0          # the pen is nowhere near paper


def test_paper_contact_reports_a_normal_force():
    """Driven straight down onto the sheet, the tip reports a push along +z of
    the order the compliant contact predicts."""
    params = a_params(tool="inline", tip_error=0.0, rig="final6_opt", arm=13)
    sim = build_simulator(params)
    q0 = np.array([-0.4749, 0.6034, -0.3287, -1.8726, 0.2844, 2.4303, 2.175])
    sim.set_state(q0)
    assert sim.tip_world()[2] == pytest.approx(params.paper.z_world, abs=2e-3)
    for _ in range(400):                        # push the wrist down gently
        tau = sim.gravity_compensation()
        tau += sim.tip_jacobian_base()[:3].T @ np.array([0.0, 0.0, -5.0])
        tau -= 3.0 * sim.dq                     # keep the free arm from bolting
        sim.step(tau)
    contact = sim.contact()
    assert contact.n_pairs == 1
    assert contact.normal_force_n > 0.5
    assert contact.penetration_m > 0.0


def test_joint_limits_are_the_fr3_ones_and_drake_enforces_them():
    """The vendored URDF carries PANDA limits, which are the wrong robot, and
    Drake's discrete plant enforces whatever is in the file.  Joint 4 is the
    clearest case: FR3 stops at -0.1518 rad, Panda at -0.0698."""
    params = a_params(tool="inline", tip_error=0.0, rig="final6_opt", arm=13)
    plant = build_simulator(params).plant
    for i in range(7):
        joint = plant.GetJointByName(f"panda_joint{i + 1}")
        assert joint.position_lower_limit() == pytest.approx(frames.FR3_MIN[i])
        assert joint.position_upper_limit() == pytest.approx(frames.FR3_MAX[i])

    sim = build_simulator(params)
    sim.set_state(np.array([0.0, 0.0, 0.0, -1.5, 0.0, 1.6, 0.0]))
    for _ in range(2000):                       # shove joint 4 into its stop
        tau = sim.gravity_compensation() - 1.0 * sim.dq
        tau[3] += 60.0
        sim.step(tau)
    assert sim.q[3] == pytest.approx(frames.FR3_MAX[3], abs=1e-6)
