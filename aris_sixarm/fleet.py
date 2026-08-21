"""The fleet registries: world layout and mount transforms.

World frame ("canvas"): z = 0 is the paper plane, z up, origin at the
paper corner, sheet extends +x, +y — for BOTH rigs.

TWO rigs live here:

  FLEET_FINAL   the 3-arm FINAL INSTALLATION from the authoritative drawing
                (docs/FINAL_RIG.md, aris_sixarm/rig_final.py): arm 13 upright
                on the tabletop, arm 31 inverted under the central beam,
                arm 2 side-mounted with J1 horizontal.  Base poses are the
                drawing's, expressed in the canvas frame; the structure they
                bolt to is `rig_final.frame_boxes_canvas()`.
  FLEET_SIXARM  the legacy six-arm layout (preset six_arm_display_ADJUSTED_h80
                + IKA h_inv; docs/DECISIONS.md) — kept VERBATIM for
                regression: every historical number in the repo was earned on
                it, and the arm ids overlap with the final rig (the drawing
                reuses physical arms 13, 31, 2), so it is a separate dict,
                never a subset.

`FLEET`/`SHEET` are the ACTIVE rig — the final one.  Modules that accept a
`fleet=` argument default to it; passing `FLEET_SIXARM` reproduces the legacy
behaviour bit for bit (tests pin that).
"""
from dataclasses import dataclass

import numpy as np

from . import rig_final
from .frames import (Q_READY_FLOOR, Q_READY_INV, Q_READY_INV_FINAL,
                     Q_READY_WALL, PEN_EXT, rotz, roty)

SHEET_SIXARM = (3.607, 1.961)   # legacy paper, metres (six-arm planner preset)
SHEET_FINAL = rig_final.SHEET_FINAL     # (1.8034, 1.700) from the drawing
H_INV_DEFAULT = 1.00        # legacy inverted base height above paper
Z_FLOOR_BASE = 0.0127       # legacy floor base plate top above paper


@dataclass(frozen=True)
class ArmSpec:
    arm_id: int
    name: str
    mount: str                      # "floor" | "inv" | "wall"
    xy: tuple
    yaw: float = 0.0
    active: bool = True
    color: tuple = (0.5, 0.5, 0.5)  # stable viz identity color
    rig: str = "sixarm"             # "sixarm" (legacy) | "final"
    z: float = None                 # explicit base height; None -> legacy rules
    R: tuple = None                 # explicit base rotation (9-tuple, row-major)
    pen_ext: float = None           # tool config; None -> frames.PEN_EXT
    q_ready: tuple = None           # explicit ready pose; None -> mount rule

    @property
    def q_seed(self):
        if self.q_ready is not None:
            return np.asarray(self.q_ready, float)
        return {"floor": Q_READY_FLOOR, "inv": Q_READY_INV,
                "wall": Q_READY_WALL}[self.mount]

    @property
    def pen(self):
        return PEN_EXT if self.pen_ext is None else self.pen_ext

    def T_world_base(self, h_inv=H_INV_DEFAULT):
        T = np.eye(4)
        if self.R is not None:          # final rig: the drawing's pose, exact
            T[:3, :3] = np.asarray(self.R, float).reshape(3, 3)
            T[:3, 3] = [*self.xy, self.z]
            return T
        if self.mount == "floor":
            T[:3, :3] = rotz(self.yaw)
            T[:3, 3] = [*self.xy, Z_FLOOR_BASE]
        else:  # legacy booth URDF hanging seats: rpy (0, pi, 0) + local yaw
            T[:3, :3] = roty(np.pi) @ rotz(self.yaw)
            T[:3, 3] = [*self.xy, h_inv]
        return T

    def static_obstacles(self):
        """Frame boxes this arm must clear (canvas frame, m).

        Final rig: every structure box above the paper plane, minus this
        arm's own mount hardware (its base is bolted there by construction;
        the mount boxes still constrain every OTHER arm).  Legacy rig: none —
        the sixarm proxies (paper plane + own-boom cylinder) stay in force
        unchanged.
        """
        if self.rig != "final":
            return []
        key = {13: "up", 31: "down", 2: "side"}[self.arm_id]
        return rig_final.frame_boxes_canvas(exclude_tag=f"mount:{key}")


def _final(key, arm_id, name, color, active=True):
    p, R = rig_final.arm_base_canvas(key)
    mount = {"up": "floor", "down": "inv", "side": "wall"}[key]
    # arm 31's ready pose is FINAL-RIG-SPECIFIC: the legacy Q_READY_INV dips
    # the pen below the paper at this rig's h = 0.922 (see frames.py)
    ready = {"down": tuple(Q_READY_INV_FINAL)}.get(key)
    return ArmSpec(arm_id, name, mount, (float(p[0]), float(p[1])), 0.0,
                   active, color, rig="final", z=float(p[2]),
                   R=tuple(np.asarray(R).flatten()), q_ready=ready)


# ids and mounts straight from the drawing's own text: "Three robot arms:
# 13 floor, 31 left - upside down, 2 right side position".  Identity colors
# carried over from the legacy registry.
FLEET_FINAL = {
    13: _final("up", 13, "up-front", (0.12, 0.47, 0.71)),
    31: _final("down", 31, "down-left", (0.84, 0.15, 0.16)),
    2:  _final("side", 2, "side-right", (0.17, 0.63, 0.17)),
}

# the legacy six-arm layout, VERBATIM (sources: docs/DECISIONS.md)
FLEET_SIXARM = {
    13: ArmSpec(13, "front", "floor", (-0.1118, 1.0008), 0.0, True, (0.12, 0.47, 0.71)),
    17: ArmSpec(17, "back", "floor", (3.7186, 1.0008), np.pi, True, (0.09, 0.75, 0.81)),
    31: ArmSpec(31, "L-inv-front", "inv", (1.3030, 1.6307), 0.0, True, (0.84, 0.15, 0.16)),
    2:  ArmSpec(2, "R-wall-front", "inv", (1.3030, 0.3505), 0.0, False, (0.17, 0.63, 0.17)),
    71: ArmSpec(71, "L-inv-back", "inv", (2.3038, 1.6307), 0.0, False, (1.00, 0.50, 0.05)),
    97: ArmSpec(97, "R-inv-back", "inv", (2.3038, 0.3505), 0.0, True, (0.58, 0.40, 0.74)),
}

FLEET = FLEET_FINAL             # the ACTIVE rig
SHEET = SHEET_FINAL


def sheet_for(spec):
    """The paper an arm draws on, decided by the spec's rig — so a legacy
    six-arm spec plans against the legacy 3.607 x 1.961 sheet whatever the
    active rig is, and vice versa."""
    return SHEET_SIXARM if getattr(spec, "rig", "sixarm") == "sixarm" \
        else SHEET_FINAL
