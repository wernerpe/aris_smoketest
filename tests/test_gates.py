"""Validation gates — the same real-touchdown checks the IKA toolkit passed.

Run: pytest tests/  (or python3 tests/test_gates.py)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from aris_sixarm.frames import fk, tip_pos, PEN_EXT  # noqa: E402
from aris_sixarm import ik  # noqa: E402

# real arm-31 contact joints, h=92 rig (2026-07-07 touchdown)
Q_CONTACT = np.array([-2.307, -1.608, -0.860, -2.091, 1.685, 2.332, 1.048])


def test_gate_a_fk():
    """FK convention: known pose reaches [0.30, 0, 0.40] with tool z down."""
    q0 = np.array([-0.001, -0.842, 0.001, -2.611, 0.0, 1.769, 0.785])
    T, _ = fk(q0)
    assert np.linalg.norm(T[:3, 3] - [0.30, 0, 0.40]) < 5e-3
    assert T[2, 2] < -0.99


def test_gate_c_ik_roundtrip():
    """Analytic IK reproduces the real contact configuration exactly."""
    T_c, _ = fk(Q_CONTACT)
    best = min((float(np.max(np.abs(q - Q_CONTACT)))
                for q in ik.solve(T_c, Q_CONTACT[6], Q_CONTACT)), default=None)
    assert best is not None and best < 0.01


def test_pen_tip_offset():
    """Tip sits exactly PEN_EXT beyond the TCP along tool z."""
    T, _ = fk(Q_CONTACT)
    assert np.allclose(tip_pos(Q_CONTACT), T[:3, 3] + T[:3, :3] @ [0, 0, PEN_EXT])


def test_urdf_visual_matches_dh():
    """The drake-URDF visual chain agrees with the DH chain at the TCP."""
    from aris_sixarm.viz import robot_model
    _, joints = robot_model.load_model()
    poses = robot_model.link_poses(joints, Q_CONTACT)
    Th = poses["panda_hand"]
    tcp_urdf = Th[:3, 3] + Th[:3, :3] @ [0, 0, 0.1034]
    T, _ = fk(Q_CONTACT)
    assert np.linalg.norm(tcp_urdf - T[:3, 3]) < 1e-6


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name} PASS")
