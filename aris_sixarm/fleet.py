"""The six-arm fleet: world layout and mount transforms.

World frame ("canvas"): z = 0 is the paper plane, z up, origin at the
paper corner, sheet extends +x, +y.

Sources & provenance (the upstream repos disagree; picks documented in
docs/DECISIONS.md):
  - arm XY: Aris_Kindt arm_orchestrator/presets/six_arm_display_ADJUSTED_h80.json
  - H_INV = 1.00 m: the IKA 2026-07-12 governing answer (h=100cm + <=15deg
    tilt => solid ~154cm field; "do NOT lower the mount"). The rig measured
    0.924 on 2026-07-07 — pass h_inv=0.924 to model today's rig instead.
  - floor base z = +0.0127 m: booth URDF standing seat (0.6477) minus the
    revised table top (0.635) = one base-plate thickness.
  - arm 2 is a wall-plate mount in the registry but modelled topdown here,
    matching every preset (the wall experiment draws on the same paper).
"""
from dataclasses import dataclass, field

import numpy as np

from .frames import Q_READY_FLOOR, Q_READY_INV, rotz, roty

SHEET = (3.607, 1.961)      # paper, metres (six-arm planner preset)
H_INV_DEFAULT = 1.00        # inverted base height above paper
Z_FLOOR_BASE = 0.0127       # floor base plate top above paper


@dataclass(frozen=True)
class ArmSpec:
    arm_id: int
    name: str
    mount: str                      # "floor" | "inv"
    xy: tuple
    yaw: float = 0.0
    active: bool = True             # arm_registry.py active flag (2026-08)
    color: tuple = (0.5, 0.5, 0.5)  # stable viz identity color

    @property
    def q_seed(self):
        return Q_READY_FLOOR if self.mount == "floor" else Q_READY_INV

    def T_world_base(self, h_inv=H_INV_DEFAULT):
        T = np.eye(4)
        if self.mount == "floor":
            T[:3, :3] = rotz(self.yaw)
            T[:3, 3] = [*self.xy, Z_FLOOR_BASE]
        else:  # booth URDF hanging seats: rpy (0, pi, 0), then local yaw
            T[:3, :3] = roty(np.pi) @ rotz(self.yaw)
            T[:3, 3] = [*self.xy, h_inv]
        return T


FLEET = {
    13: ArmSpec(13, "front", "floor", (-0.1118, 1.0008), 0.0, True, (0.12, 0.47, 0.71)),
    17: ArmSpec(17, "back", "floor", (3.7186, 1.0008), np.pi, True, (0.09, 0.75, 0.81)),
    31: ArmSpec(31, "L-inv-front", "inv", (1.3030, 1.6307), 0.0, True, (0.84, 0.15, 0.16)),
    2:  ArmSpec(2, "R-wall-front", "inv", (1.3030, 0.3505), 0.0, False, (0.17, 0.63, 0.17)),
    71: ArmSpec(71, "L-inv-back", "inv", (2.3038, 1.6307), 0.0, False, (1.00, 0.50, 0.05)),
    97: ArmSpec(97, "R-inv-back", "inv", (2.3038, 0.3505), 0.0, True, (0.58, 0.40, 0.74)),
}
