"""Numpy replica of the deployed `CartesianImpedanceController`.

Line-for-line from `cartesian_impedance_controller.cpp` `update()` and
`on_activate()` (the stale-but-structurally-correct copy in the `aris2-rtff`
worktree; the live 378-line source is unversioned, briefing §2.3).  The order
of operations is the C++ order and the arithmetic is the C++ arithmetic; where
this file differs from the C++ it says so in a comment.

WHAT IS AND IS NOT IN THE LAW
  IN:   setpoint low-pass (filter_alpha, slerp), sign-continuous quaternion,
        pose-error clamp, F = K e - D J dq, tau = J^T F + N tau_posture +
        coriolis, per-joint magnitude clamp and slew limit.
  OUT:  GRAVITY.  libfranka's hardware layer compensates it and the controller
        must not (briefing §3, §7.2); `simulator.py` adds -tau_g outside the
        law, which is what that layer does.

FRAMES.  Everything is expressed in the arm's BASE frame (`fr3_link0`), which
is where `getZeroJacobian` and `getPoseMatrix` deliver and where the
equilibrium-pose topic lives.  The Jacobian is 6x7 ordered [linear; angular]
to match `K = diag(kx,ky,kz,krx,kry,krz)` acting on [position; rotation]
error -- libfranka's `getZeroJacobian` order, NOT Drake's.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .params import Controller
from .rotations import (matrix_from_quat, quat_align, quat_conj, quat_mul,
                        quat_from_matrix, quat_normalize, rotvec_from_quat,
                        slerp)


@dataclass
class LawDebug:
    """What one `step` did, for the trace. SI units, base frame."""

    error: np.ndarray            # (6,) clamped pose error [m, m, m, rad x3]
    wrench: np.ndarray           # (6,) F = K e - D J dq, [N x3, Nm x3]
    tau_task: np.ndarray         # (7,) Nm
    tau_null: np.ndarray         # (7,) Nm
    tau_raw: np.ndarray          # (7,) Nm, before saturation
    pos_d: np.ndarray            # (3,) m, the FILTERED setpoint position
    quat_d: np.ndarray           # (4,) xyzw, the FILTERED setpoint orientation


class CartesianImpedanceLaw:
    """Stateful 1 kHz control law. One instance per simulated controller.

    State, and it is exactly the C++ member set: the filtered setpoint
    (`position_d_`, `orientation_d_`), its unfiltered target
    (`position_d_target_`, `orientation_d_target_`), the latched nullspace
    posture (`q_nullspace_`) and the last commanded torque (`last_tau_`).
    """

    def __init__(self, cfg: Controller):
        self.cfg = cfg
        self._pos_d = np.zeros(3)
        self._quat_d = np.array([0.0, 0.0, 0.0, 1.0])
        self._pos_target = np.zeros(3)
        self._quat_target = np.array([0.0, 0.0, 0.0, 1.0])
        self._q_null = np.asarray(cfg.q_nullspace, float).copy()
        self._last_tau = np.zeros(7)
        self._q_ref = None                  # contract §2 joint reference, RAW
        self._q_ref_age_s = np.inf
        self._active = False
        self.qref_divergence_ticks = 0      # |raw - effective| > threshold
        self.qref_ticks = 0                 # ticks with a fresh reference

    # -- on_activate -------------------------------------------------------
    def activate(self, q: np.ndarray, p_tip: np.ndarray,
                 quat_tip: np.ndarray) -> None:
        """Latch the current tip pose as the desired and the current posture
        as the nullspace target, and zero the torque history.

        `on_activate` does exactly this, and the comment on why the CONFIGURED
        `q_nullspace` is overwritten is worth carrying: a fixed posture target
        ~2 rad from the actual configuration makes the posture spring
        reconfigure the elbow at the torque clamp the moment torques start, and
        the swing trips the robot's joint_velocity_violation reflex within a
        second (seen 2026-06-12).
        """
        self._pos_d = np.asarray(p_tip, float).copy()
        self._quat_d = quat_normalize(quat_tip)
        self._pos_target = self._pos_d.copy()
        self._quat_target = self._quat_d.copy()
        self._q_null = np.asarray(q, float).copy()
        self._last_tau = np.zeros(7)
        self._q_ref = None
        self._q_ref_age_s = np.inf
        self._active = True

    # -- read-only view of the state --------------------------------------
    @property
    def q_nullspace(self) -> np.ndarray:
        """(7,) rad, the current nullspace posture target."""
        return self._q_null.copy()

    @property
    def q_reference_raw(self) -> np.ndarray:
        """(7,) rad, the joint reference AS RECEIVED — the effective target
        (`q_nullspace`) is what the guard and the low-pass actually let
        through.  Falls back to the effective target when nothing is
        published, so the two are equal exactly when there is nothing to
        catch up with."""
        return self._q_null.copy() if self._q_ref is None else self._q_ref.copy()

    @property
    def filtered_setpoint(self):
        """(position (3,) m, orientation (4,) xyzw) after the low-pass."""
        return self._pos_d.copy(), self._quat_d.copy()

    # -- the projector, exposed so a test can measure it -------------------
    def nullspace_projector(self, J: np.ndarray) -> np.ndarray:
        """-> (7,7) N = I - J^T pinv_damped(J^T).

        `jacobianTransposePseudoInverse`: M = (J J^T + d^2 I)^-1 J, so that
        M J^T -> I as the damping vanishes.  The damping is what keeps the
        projector finite through the wrist band, and it is also why N J^T is
        only approximately zero -- N J^T = J^T (J J^T + d^2 I)^-1 d^2, i.e. a
        residual of order d^2 / sigma_min^2.
        """
        d2 = self.cfg.damping_pinv ** 2
        JJt = J @ J.T + d2 * np.eye(6)
        Jt_pinv = np.linalg.solve(JJt, J)
        return np.eye(7) - J.T @ Jt_pinv

    # -- saturate ----------------------------------------------------------
    def saturate(self, tau: np.ndarray, last: np.ndarray,
                 dt: float) -> np.ndarray:
        """Per-joint magnitude clamp then slew limit, in that order.

        Verbatim from `CartesianImpedanceController::saturate`.  The magnitude
        limit is PER JOINT (`max_torques`) because a uniform scalar starved the
        shoulders at stretched poses -- 8-13 mm of tracking error, 2026-06-12 --
        while letting the wrist exceed its 12 Nm rating.
        """
        lim = self.cfg.max_torques
        max_step = self.cfg.max_torque_rate * dt
        clamped = np.clip(tau, -lim, lim)
        delta = np.clip(clamped - last, -max_step, max_step)
        return last + delta

    # -- update ------------------------------------------------------------
    def step(self, q: np.ndarray, dq: np.ndarray, p_tip: np.ndarray,
             quat_tip: np.ndarray, J_tip: np.ndarray, coriolis: np.ndarray,
             dt: float, setpoint=None):
        """One 1 kHz tick. -> (tau (7,) Nm, LawDebug).

        Args:
          q, dq      (7,) rad, rad/s -- the joint state the controller reads.
          p_tip      (3,) m, the NOMINAL tip position in the base frame
                     (`o_t_ee`: the robot reports the tip it BELIEVES in).
          quat_tip   (4,) xyzw, that frame's orientation.
          J_tip      (6,7) the nominal-tip Jacobian in the base frame,
                     [linear; angular].
          coriolis   (7,) Nm, C(q,dq) dq.  Gravity is NOT included and must
                     not be: the hardware layer adds it.
          dt         s.
          setpoint   a `setpoints.Setpoint`, or None to hold the last target
                     (which is what the RT buffer does between publishes).
        """
        if not self._active:
            raise RuntimeError("activate() the law before stepping it")
        cfg = self.cfg
        dt = max(float(dt), 1e-4)          # the C++ floor on `period`
        q = np.asarray(q, float)
        dq = np.asarray(dq, float)

        # --- latch the latest desired (the RT buffer read) -----------------
        latched = cfg.nullspace_mode == "latched"
        if setpoint is not None:
            self._pos_target = np.asarray(setpoint.p_xyz, float)
            self._quat_target = quat_align(setpoint.quat_xyzw, self._quat_target)
            if setpoint.q_ref is not None and not latched:
                self._q_ref = np.asarray(setpoint.q_ref, float)
                self._q_ref_age_s = 0.0
        self._q_ref_age_s += dt

        # --- low-pass the running desired toward the target ---------------
        a = cfg.filter_alpha
        self._pos_d = a * self._pos_target + (1.0 - a) * self._pos_d
        self._quat_d = quat_normalize(slerp(self._quat_d, self._quat_target, a))

        # --- nullspace target: contract §2, plus the continuity guard ------
        # A FRESH joint reference replaces the latched posture, low-passed with
        # the same filter_alpha; a stale or absent one leaves the activation
        # posture standing, which is today's deployed behaviour exactly.
        # `qref_rate_limit_rad_s` then bounds how fast the EFFECTIVE target may
        # move, so a plan that steps the posture between strokes is followed at
        # a rate the arm can absorb instead of being thrown at the joint stops.
        if self._q_ref is not None and \
                self._q_ref_age_s <= cfg.joint_reference_timeout_s:
            self.qref_ticks += 1
            gap = float(np.abs(self._q_ref - self._q_null).max())
            if gap > cfg.qref_divergence_rad:
                self.qref_divergence_ticks += 1
            target = a * self._q_ref + (1.0 - a) * self._q_null
            if cfg.qref_rate_limit_rad_s > 0.0:
                step = cfg.qref_rate_limit_rad_s * dt
                target = self._q_null + np.clip(target - self._q_null,
                                                -step, step)
            self._q_null = target

        # --- pose error ---------------------------------------------------
        error = np.zeros(6)
        error[:3] = self._pos_d - np.asarray(p_tip, float)
        quat_meas = quat_align(quat_normalize(quat_tip), self._quat_d)
        error[3:] = rotvec_from_quat(quat_mul(self._quat_d, quat_conj(quat_meas)))
        error[:3] = np.clip(error[:3], -cfg.max_pose_error_pos,
                            cfg.max_pose_error_pos)
        error[3:] = np.clip(error[3:], -cfg.max_pose_error_rot,
                            cfg.max_pose_error_rot)

        # --- Cartesian wrench + task torque -------------------------------
        F = cfg.k_cartesian * error - cfg.d_cartesian * (J_tip @ dq)
        tau_task = J_tip.T @ F

        # --- nullspace ----------------------------------------------------
        N = self.nullspace_projector(J_tip)
        tau_null = N @ (cfg.k_nullspace * (self._q_null - q)
                        - cfg.d_nullspace * dq)

        tau_raw = tau_task + tau_null + np.asarray(coriolis, float)
        # gravity is added by the hardware layer; NOT here.

        tau = self.saturate(tau_raw, self._last_tau, dt)
        self._last_tau = tau
        return tau, LawDebug(error, F, tau_task, tau_null, tau_raw,
                             self._pos_d.copy(), self._quat_d.copy())


def pose_from_matrix(T: np.ndarray):
    """(4,4) -> (position (3,), quaternion (4,) xyzw). A convenience the
    plant and the setpoint sources share."""
    T = np.asarray(T, float)
    return T[:3, 3].copy(), quat_from_matrix(T[:3, :3])


def matrix_from_pose(p: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """(position, xyzw quaternion) -> (4,4)."""
    T = np.eye(4)
    T[:3, :3] = matrix_from_quat(quat)
    T[:3, 3] = np.asarray(p, float)
    return T
