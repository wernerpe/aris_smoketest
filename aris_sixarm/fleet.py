"""The fleet registries: world layout and mount transforms.

World frame ("canvas"): z = 0 is the paper plane, z up, origin at the
paper corner, sheet extends +x, +y — for BOTH rigs.

FOUR rigs are selectable; two of them are built here and two in
`rig_final6.py` (which reflects this one's geometry and cannot be imported
from here at module scope without a cycle):

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

  rig_final6.FLEET_FINAL6      two of those units mirrored back to back: six
                arms, one continuous 1.8034 x 3.63064 m canvas.
  rig_final6.FLEET_FINAL6_OPT  the same six with both side poles lengthened
                20 cm and both side arms re-clamped at canvas z 0.576.

`FLEET`/`SHEET` are the ACTIVE rig, which DEFAULTS to the 3-arm final one so
that every published number still reproduces.  `activate(name)` (or the
`ARIS_RIG` env var, read by `aris_sixarm/__init__.py`) switches it; modules
that accept a `fleet=` argument default to whatever is active, and passing
`FLEET_SIXARM` reproduces the legacy behaviour bit for bit (tests pin that).
"""
from dataclasses import dataclass, field

import numpy as np

from . import mounts
from . import rig_final
from .frames import (Q_READY_FLOOR, Q_READY_INV, Q_READY_INV_FINAL,
                     Q_READY_WALL, PEN_EXT, ext_of, rotz, roty)

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
    pen_ext: float = None           # tool config; None -> the ACTIVE tool
    q_ready: tuple = None           # explicit ready pose; None -> mount rule
    # the OTHER arms' pose-invariant base columns, written in by
    # `mounts.attach_body_columns` once the registry exists (see there).
    column_boxes: tuple = field(default=(), compare=False, repr=False)

    @property
    def q_seed(self):
        if self.q_ready is not None:
            return np.asarray(self.q_ready, float)
        return {"floor": Q_READY_FLOOR, "inv": Q_READY_INV,
                "wall": Q_READY_WALL}[self.mount]

    @property
    def pen(self):
        """This arm's axial pen depth; unset -> the ACTIVE tool's own.

        Unset used to mean the INLINE pen's `PEN_EXT` even under
        `ARIS_TOOL=lateral`; it now means `frames.ext_of()`, which is the
        holder's `PEN_EXT_HOLDER` when the lateral tool is active.
        """
        return ext_of(self.pen_ext)

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

        PLUS, on any rig whose registry called `mounts.attach_body_columns`,
        the other arms' pose-invariant BASE COLUMNS: an arm is an obstacle
        even when it is not moving, and 0.333 m of it is an obstacle even
        when nobody has decided what pose it will hold.
        """
        cols = list(self.column_boxes)
        if self.rig != "final":
            return cols
        key = {13: "up", 31: "down", 2: "side"}[self.arm_id]
        return rig_final.frame_boxes_canvas(exclude_tag=f"mount:{key}") + cols


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
FLEET_FINAL = mounts.attach_body_columns({
    13: _final("up", 13, "up-front", (0.12, 0.47, 0.71)),
    31: _final("down", 31, "down-left", (0.84, 0.15, 0.16)),
    2:  _final("side", 2, "side-right", (0.17, 0.63, 0.17)),
})

# the legacy six-arm layout, VERBATIM (sources: docs/DECISIONS.md) — and
# verbatim includes its obstacle model, which is the two legacy proxies and
# nothing else: no structure, and no body columns.  Every published number in
# this repo was earned against exactly that, and this registry exists so they
# still reproduce.
FLEET_SIXARM = {
    13: ArmSpec(13, "front", "floor", (-0.1118, 1.0008), 0.0, True, (0.12, 0.47, 0.71)),
    17: ArmSpec(17, "back", "floor", (3.7186, 1.0008), np.pi, True, (0.09, 0.75, 0.81)),
    31: ArmSpec(31, "L-inv-front", "inv", (1.3030, 1.6307), 0.0, True, (0.84, 0.15, 0.16)),
    2:  ArmSpec(2, "R-wall-front", "inv", (1.3030, 0.3505), 0.0, False, (0.17, 0.63, 0.17)),
    71: ArmSpec(71, "L-inv-back", "inv", (2.3038, 1.6307), 0.0, False, (1.00, 0.50, 0.05)),
    97: ArmSpec(97, "R-inv-back", "inv", (2.3038, 0.3505), 0.0, True, (0.58, 0.40, 0.74)),
}

# the ACTIVE rig — see `activate` below.  A COPY, not `FLEET_FINAL` itself:
# `activate` mutates this dict in place (it is the object half the package
# holds a reference to), and the four registries must survive that.
FLEET = dict(FLEET_FINAL)
SHEET = SHEET_FINAL
ACTIVE_RIG = "final"

# ===========================================================================
# THE ACTIVE RIG IS A CHOICE, AND IT IS MADE IN ONE PLACE
# ===========================================================================
# Four rigs now exist and they are not variations of one number: they differ in
# WHICH ARMS EXIST, in where those arms are bolted, and in how big the paper is.
# Every module downstream reads `FLEET` and `SHEET`, so switching rigs is
# switching those two — and the only honest way to do it is at import time,
# before anything has captured either.
#
#   final        the 3-arm FINAL INSTALLATION from the drawing (the DEFAULT).
#                Every historical number in this repo and every pinned test was
#                earned on it; it stays the default so that all of them still
#                reproduce by running the command that produced them.
#   final6       the real installation: two of those units mirrored back to
#                back, six arms, ONE continuous 1.8034 x 3.63064 m canvas
#                (rig_final6.MERGE_WEBS), the side arms as drawn.
#   final6_opt   the same six arms with BOTH side poles lengthened 20 cm and
#                both side arms re-clamped at canvas z 0.576 — the optimum of
#                docs/ARM2_HEIGHT.md, adopted by the user.  ASSUMES THE TWO
#                POLE EXTENSIONS ARE PHYSICALLY INSTALLED.
#   sixarm       the legacy six-arm preset, kept verbatim for regression.
#
# Two ways in, and they are the same door:
#   ARIS_RIG=final6_opt python3 scripts/whatever.py     (import time, total)
#   fleet.activate("final6_opt")                        (in process, for tests)
# The env var is the one to use from a script, because a script's `from
# aris_sixarm.fleet import SHEET` runs before its first line does and only the
# env var is earlier than that; `aris_sixarm/__init__.py` is what reads it.
RIG_NAMES = ("final", "final6", "final6_opt", "sixarm", "proposed")


def rig(name):
    """-> (fleet dict, sheet) for a named rig.  Nothing is mutated."""
    if name == "final":
        return FLEET_FINAL, SHEET_FINAL
    if name == "sixarm":
        return FLEET_SIXARM, SHEET_SIXARM
    if name in ("final6", "final6_opt"):
        # imported HERE, not at module scope: `rig_final6` reads `ArmSpec` from
        # this module, so a top-level import would be a cycle
        from . import rig_final6
        return ((rig_final6.FLEET_FINAL6_OPT if name == "final6_opt"
                 else rig_final6.FLEET_FINAL6),
                rig_final6.SHEET_FINAL6)
    if name == "proposed":
        # the GREEN-FIELD layout study's winner (docs/LAYOUT_STUDY.md):
        # 2 floor + 4 ceiling-inverted arms, no wall mounts, LATERAL tool.
        # No structure exists for these positions yet — see aris_sixarm/layout.py.
        from . import layout, rig_final6
        return layout.FLEET_PROPOSED, rig_final6.SHEET_FINAL6
    raise ValueError(f"unknown rig {name!r}; want one of {RIG_NAMES}")


# the modules that bind `SHEET` INTO THEIR OWN NAMESPACE at import time.  A
# rebind of `fleet.SHEET` does not reach them, so `activate` walks this list —
# explicitly, so that a module which starts doing it has to be added here and
# cannot be silently missed.  (Everything else either reads `fleet.SHEET`
# lazily inside a function — `planner`, `allocate` — or takes it as an
# argument.)  Scripts are NOT on this list and cannot be: they import before
# they can call anything, which is what `ARIS_RIG` is for — and `bench` bakes
# SHEET into DEFAULT ARGUMENTS as well, which are evaluated at def time and
# which nothing can rebind afterwards, so a bench run must use the env var.
_SHEET_BINDERS = ("aris_sixarm.atlas", "aris_sixarm.viz.scene",
                  "aris_sixarm.bench")


def activate(name):
    """Make a named rig the ACTIVE one, in this process. -> the fleet dict.

    `FLEET` is mutated IN PLACE rather than rebound, because half the package
    did `from .fleet import FLEET` and holds the dict object itself; mutating
    it is the only edit all of them see.  `SHEET` is a tuple and cannot be
    mutated, so it is rebound here and in `_SHEET_BINDERS`.

    Prefer `ARIS_RIG` in anything with a `__main__`.  This exists for tests and
    for a caller that has not yet imported the modules that bind SHEET.
    """
    import sys
    fl, sheet = rig(name)
    global SHEET, ACTIVE_RIG
    FLEET.clear()
    FLEET.update(fl)
    SHEET = tuple(sheet)
    ACTIVE_RIG = name
    for m in _SHEET_BINDERS:
        mod = sys.modules.get(m)
        if mod is not None and hasattr(mod, "SHEET"):
            mod.SHEET = SHEET
    al = sys.modules.get("aris_sixarm.allocate")
    if al is not None:                      # a module-level derived constant
        al.ACTIVE = [aid for aid, s in FLEET.items() if s.active]
    bench = sys.modules.get("aris_sixarm.bench")
    if bench is not None and hasattr(bench, "HATCH_ARM"):
        bench.HATCH_ARM = next(a for a in FLEET if FLEET[a].mount == "inv")
    return FLEET


def sheet_for(spec):
    """The paper an arm draws on, decided by the spec's rig — so a legacy
    six-arm spec plans against the legacy 3.607 x 1.961 sheet whatever the
    active rig is, and vice versa.

    A MIRRORED-rig spec (`rig_final6.Arm6Spec`) draws on the COMBINED canvas.
    Under `MERGE_WEBS` that is one continuous surface from the canvas origin,
    which is exactly what `planner.clip_to_sheet` measures against.  A
    `layout.StudySpec` carries `unit` too (and the legacy proxies, so its
    `rig` reads "sixarm") — the unit check must therefore come first.
    """
    if hasattr(spec, "unit"):
        from . import rig_final6
        return rig_final6.SHEET_FINAL6
    if getattr(spec, "rig", "sixarm") == "sixarm":
        return SHEET_SIXARM
    return SHEET_FINAL
