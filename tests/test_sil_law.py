"""The impedance-law replica, against the C++ it transcribes.

Pure numpy — no Drake, no plant, no CSV.  What is pinned here is the structure
of `cartesian_impedance_controller.cpp::update()`: what the law returns when
there is nothing to correct, that the pose error is clamped, that the two
safety limits hold, and that the nullspace projector really is one.
"""
import dataclasses

import numpy as np
import pytest

from aris_sixarm.sil.impedance import CartesianImpedanceLaw
from aris_sixarm.sil.params import Controller
from aris_sixarm.sil.rotations import quat_from_matrix, rotx, rotz
from aris_sixarm.sil.setpoints import Setpoint

Q0 = np.array([0.1, -0.7, 0.2, -2.1, -0.3, 1.6, 0.8])
P0 = np.array([0.45, 0.02, 0.31])
QUAT0 = quat_from_matrix(rotx(np.pi))


def a_jacobian(seed=0):
    """A well-conditioned 6x7 [linear; angular] Jacobian of plausible scale."""
    rng = np.random.default_rng(seed)
    J = rng.normal(size=(6, 7))
    u, s, vt = np.linalg.svd(J, full_matrices=False)
    return u @ np.diag(np.linspace(0.5, 1.0, 6)) @ vt


def a_law(**kw):
    cfg = dataclasses.replace(Controller.from_yaml(), **kw)
    law = CartesianImpedanceLaw(cfg)
    law.activate(Q0, P0, QUAT0)
    return law, cfg


def test_yaml_is_the_draw_phase_of_the_briefing():
    cfg = Controller.from_yaml()
    assert cfg.k_cartesian.tolist() == [2800.0, 2800.0, 800.0, 600.0, 600.0,
                                        200.0]
    assert cfg.max_torques.tolist() == [75.0] * 4 + [11.0] * 3
    assert cfg.filter_alpha == 0.05 and cfg.damping_pinv == 0.05
    assert cfg.max_pose_error_pos == 0.05 and cfg.max_pose_error_rot == 0.30


def test_zero_error_zero_velocity_gives_coriolis_plus_nullspace():
    """e = 0, dq = 0  =>  tau = N (Kn (q_null - q)) + coriolis, exactly.

    Held at the activation pose with no setpoint, so `position_d_` never
    moves off `position`.  The law is stepped until the SLEW limiter has let
    the command reach its steady value (max_torque_rate * dt = 1 Nm per tick);
    what is compared is that steady value.
    """
    law, cfg = a_law()
    J = a_jacobian()
    coriolis = np.array([0.3, -1.2, 0.4, 0.9, -0.1, 0.05, -0.02])
    q = Q0 + 0.05                      # posture away from the latched target
    for _ in range(400):
        tau, dbg = law.step(q, np.zeros(7), P0, QUAT0, J, coriolis, 1e-3, None)
    want = law.nullspace_projector(J) @ (cfg.k_nullspace * (Q0 - q)) + coriolis
    assert np.allclose(dbg.error, 0.0, atol=1e-12)
    assert np.allclose(dbg.wrench, 0.0, atol=1e-12)
    assert np.allclose(dbg.tau_task, 0.0, atol=1e-12)
    assert np.allclose(tau, want, atol=1e-10)


def test_pose_error_is_clamped_in_both_halves():
    """|e_pos| <= max_pose_error_pos and |e_rot| <= max_pose_error_rot.

    The clamp is a force ceiling in disguise: F_max,z = K_z * 0.05, i.e. 40 N
    of spring authority at the draw-phase K_z of 800 (briefing §7.2).
    """
    law, cfg = a_law()
    J = a_jacobian(1)
    far = Setpoint(0.0, P0 + np.array([10.0, -10.0, 10.0]),
                   quat_from_matrix(rotz(np.pi) @ rotx(np.pi)))
    worst_pos, worst_rot = 0.0, 0.0
    for _ in range(300):
        _, dbg = law.step(Q0, np.zeros(7), P0, QUAT0, J, np.zeros(7), 1e-3, far)
        worst_pos = max(worst_pos, float(np.abs(dbg.error[:3]).max()))
        worst_rot = max(worst_rot, float(np.abs(dbg.error[3:]).max()))
    assert worst_pos == pytest.approx(cfg.max_pose_error_pos, abs=1e-12)
    assert worst_rot > 0.2                       # the rotation really is large
    assert worst_rot <= cfg.max_pose_error_rot + 1e-12


def test_magnitude_and_slew_saturation_hold():
    """Per-joint |tau| <= max_torques and |dtau| <= max_torque_rate * dt."""
    law, cfg = a_law()
    J = a_jacobian(2)
    far = Setpoint(0.0, P0 + np.array([5.0, 5.0, -5.0]), QUAT0)
    last = np.zeros(7)
    peak_step = 0.0
    for _ in range(500):
        tau, _ = law.step(Q0, np.zeros(7), P0, QUAT0, J, np.zeros(7), 1e-3, far)
        peak_step = max(peak_step, float(np.abs(tau - last).max()))
        assert np.all(np.abs(tau) <= cfg.max_torques + 1e-12)
        assert np.all(np.abs(tau - last) <= cfg.max_torque_rate * 1e-3 + 1e-12)
        last = tau
    # the limiter is actually engaged by this input, not vacuously satisfied
    assert peak_step == pytest.approx(cfg.max_torque_rate * 1e-3, rel=1e-9)
    assert np.any(np.abs(last) >= cfg.max_torques - 1e-9)


def test_nullspace_projector_annihilates_task_torques():
    """N J^T ~ 0.  The damped pinv leaves a residual of order
    d^2 / sigma^2 -- exact in the limit, bounded at the shipped damping."""
    law, cfg = a_law()
    J = a_jacobian(3)
    N = law.nullspace_projector(J)
    scale = float(np.abs(J.T).max())
    assert float(np.abs(N @ J.T).max()) <= 0.05 * scale

    tight, _ = a_law(damping_pinv=1e-6)
    assert float(np.abs(tight.nullspace_projector(J) @ J.T).max()) < 1e-8
    # and it is a projector: idempotent, symmetric
    Nt = tight.nullspace_projector(J)
    assert np.allclose(Nt @ Nt, Nt, atol=1e-6)
    assert np.allclose(Nt, Nt.T, atol=1e-6)


def test_joint_reference_replaces_the_latched_posture_then_expires():
    """Contract §2 with the guard OFF: a fresh reference low-passes
    `q_nullspace_` toward it with `filter_alpha`; a stale one leaves the
    activation posture standing."""
    law, cfg = a_law(qref_rate_limit_rad_s=0.0)
    J = a_jacobian(4)
    q_ref = Q0 + 0.4
    sp = Setpoint(0.0, P0, QUAT0, q_ref)
    law.step(Q0, np.zeros(7), P0, QUAT0, J, np.zeros(7), 1e-3, sp)
    assert np.allclose(law.q_nullspace, cfg.filter_alpha * q_ref
                       + (1 - cfg.filter_alpha) * Q0)
    assert np.allclose(law.q_reference_raw, q_ref)
    for _ in range(2000):
        law.step(Q0, np.zeros(7), P0, QUAT0, J, np.zeros(7), 1e-3, sp)
    assert np.allclose(law.q_nullspace, q_ref, atol=1e-6)

    # nothing published for 0.5 s -> the target freezes where it was
    frozen = law.q_nullspace
    for _ in range(2000):
        law.step(Q0, np.zeros(7), P0, QUAT0, J, np.zeros(7), 1e-3, None)
    assert np.allclose(law.q_nullspace, frozen)


def test_latched_mode_ignores_the_joint_reference():
    """`nullspace_mode='latched'` is the DEPLOYED stage-0 law: the q columns
    exist and are not read."""
    law, _ = a_law(nullspace_mode="latched")
    J = a_jacobian(5)
    sp = Setpoint(0.0, P0, QUAT0, Q0 + 4.5)
    for _ in range(3000):
        law.step(Q0, np.zeros(7), P0, QUAT0, J, np.zeros(7), 1e-3, sp)
    assert np.allclose(law.q_nullspace, Q0)
    assert np.allclose(law.q_reference_raw, Q0)      # nothing was latched
    assert law.qref_ticks == 0 and law.qref_divergence_ticks == 0


def test_qref_rate_limit_slews_instead_of_stepping_and_counts_divergence():
    """A 4.5 rad reference step -- what a plan whose transits are not in the
    file hands the controller -- is followed at exactly the rate limit, and
    every tick it is more than `qref_divergence_rad` ahead is counted."""
    rate = 1.0
    law, cfg = a_law(qref_rate_limit_rad_s=rate)
    J = a_jacobian(6)
    q_ref = Q0.copy()
    q_ref[0] += 4.5
    sp = Setpoint(0.0, P0, QUAT0, q_ref)
    prev = law.q_nullspace
    for _ in range(1000):                                    # 1 s
        law.step(Q0, np.zeros(7), P0, QUAT0, J, np.zeros(7), 1e-3, sp)
        assert float(np.abs(law.q_nullspace - prev).max()) <= rate * 1e-3 + 1e-12
        prev = law.q_nullspace
    moved = float(law.q_nullspace[0] - Q0[0])
    assert moved == pytest.approx(rate * 1.0, rel=1e-6)      # exactly the cap
    assert law.qref_ticks == 1000
    assert law.qref_divergence_ticks == 1000                 # still 3.5 rad away
    assert np.allclose(law.q_reference_raw, q_ref)           # raw is unfiltered

    # with the guard off the same step is taken 20x faster (alpha = 0.05)
    fast, _ = a_law(qref_rate_limit_rad_s=0.0)
    fast.step(Q0, np.zeros(7), P0, QUAT0, J, np.zeros(7), 1e-3, sp)
    assert fast.q_nullspace[0] - Q0[0] == pytest.approx(0.05 * 4.5)


def test_activation_latches_pose_and_posture():
    law, _ = a_law()
    pos_d, quat_d = law.filtered_setpoint
    assert np.allclose(pos_d, P0) and np.allclose(quat_d, QUAT0)
    assert np.allclose(law.q_nullspace, Q0)
    with pytest.raises(RuntimeError):
        CartesianImpedanceLaw(Controller.from_yaml()).step(
            Q0, np.zeros(7), P0, QUAT0, a_jacobian(), np.zeros(7), 1e-3)
