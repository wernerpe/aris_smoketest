"""aris_sixarm — motion analysis & planning for the Aris Kindt 6-arm drawing installation."""
import os as _os

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# THE ACTIVE RIG, CHOSEN BEFORE ANYTHING CAN CAPTURE IT
# ---------------------------------------------------------------------------
# `fleet.FLEET` and `fleet.SHEET` decide which arms exist and how big the paper
# is, and several modules bind them into their own namespace at import time
# (`atlas`, `viz.scene`, `bench`) — so the switch has to happen before the
# first of those imports, and the only thing earlier than a script's import
# block is the environment it was launched with.  See `fleet.activate`.
#
#     ARIS_RIG=final6_opt python3 scripts/csail_schedule.py ...
#
# It lives HERE rather than at the bottom of `fleet.py` because activating a
# six-arm rig imports `rig_final6`, which imports `fleet` — doing it inside
# `fleet`'s own module body makes that circular whenever `rig_final6` is the
# first module imported.  The package `__init__` runs before either.
_rig = _os.environ.get("ARIS_RIG", "").strip()
if _rig and _rig != "final":
    from . import fleet as _fleet
    _fleet.activate(_rig)
del _rig

# ---------------------------------------------------------------------------
# THE ACTIVE TOOL, SAME DOOR AS THE RIG
# ---------------------------------------------------------------------------
# The pen holder model: "inline" (the legacy tip = TCP + 0.110 along tool z,
# the DEFAULT so every published number reproduces) or "lateral" (the real
# holder: tip = TCP + R @ (PEN_LAT_HOLDER, 0, PEN_EXT_HOLDER), the pen off the
# wrist axis along hand x at a 45 deg lean).  BOTH halves switch: until
# 2026-09-03 only the lateral one did, and a lateral run silently kept the
# inline pen's axial 0.110.  See frames.activate_tool / frames.PEN_LAT.
#
#     ARIS_TOOL=lateral python3 scripts/whatever.py
_tool = _os.environ.get("ARIS_TOOL", "").strip()
if _tool and _tool != "inline":
    from . import frames as _frames
    _frames.activate_tool(_tool)
del _tool
