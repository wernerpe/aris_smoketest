"""THE ENDPOINT CLAMP MAY RELAX THE ROUTING FLOOR, NEVER BELOW THE PAIR GATE.

`paper.effective_static_floor` clamps a leg's static floor to what its own two
endpoints hold, and the argument for that is a CERTIFICATE: the atlas certified
every drawing pose at `STATIC_MARGIN` against the fixed frame, so refusing the
lift out of one at `STATIC_MARGIN + residual` refuses ink the arm demonstrably
has.  Nothing certified anything against a FROZEN PARTNER'S ROOM, and clamping
there is how a leg whose two endpoints are clear gets flown straight through
another arm.

Measured on `bench/scatter` stage 1 (2026-09-15): arm 2 leaves its stage entry
pose at +63.3 mm against arm 97's realised room, dips to -53.1 mm 0.7 s later
inside its `aside` block, and is back over +190 mm by t = 3.4 s.  Both
endpoints legal; the middle through the other arm.
"""
import numpy as np
import pytest

from aris_sixarm import fleet as fleet_mod
from aris_sixarm import frames, frozen, paper, staged


@pytest.fixture(scope="module")
def rig():
    rig0, tool0 = fleet_mod.ACTIVE_RIG, frames.ACTIVE_TOOL
    fleet_mod.activate("proposed")
    frames.activate_tool("lateral")
    paper.clear_cache()
    yield fleet_mod.FLEET
    staged.thaw()
    paper.disk_cache_close()
    frames.activate_tool(tool0)
    fleet_mod.activate(rig0)
    paper.clear_cache()


def test_room_floor_is_the_pair_gate_plus_the_checkers_residual():
    """NOT A NEW NUMBER.  STATIC_MARGIN and PAIR_MARGIN are both 50 mm."""
    from aris_sixarm import coordination
    assert paper.ROOM_FLOOR == paper.FRAME_FLOOR
    assert coordination.PAIR_MARGIN == pytest.approx(0.050)
    assert paper.ROOM_FLOOR > coordination.PAIR_MARGIN      # the residual


def test_the_clamp_still_relaxes_against_the_metal_with_nobody_frozen(rig):
    """THE BUG THE CLAMP EXISTS FOR IS STILL FIXED.

    With no partner frozen the floor is exactly what it always was: the lesser
    of `FRAME_FLOOR` and what the leg's own endpoints hold.
    """
    staged.thaw()
    assert not frozen.active()
    parks = staged.shipped_parks(rig)
    spec = rig[31]
    q0, q1 = parks[31], parks[31] + 0.05
    f = paper.effective_static_floor(spec, q0, q1, spec.pen)
    d = paper.chain_static(np.stack([q0, q1]), spec, spec.pen).min()
    assert f == pytest.approx(min(paper.FRAME_FLOOR, float(d)), abs=1e-9)


def test_a_frozen_partner_holds_the_floor_up_to_the_room_gate(rig):
    """THE FIX.  A partner in the room may not be clamped away.

    The endpoints are chosen so the STATIC clamp would drop the floor under the
    gate; with a partner frozen the floor must come back up to `ROOM_FLOOR`.
    """
    parks = staged.shipped_parks(rig)
    pens = {a: rig[a].pen for a in rig}
    spec = rig[31]
    q0 = np.asarray(parks[31], float)
    q1 = q0 + 0.05
    staged.thaw()
    free = paper.effective_static_floor(spec, q0, q1, spec.pen)
    try:
        frozen.freeze({a: parks[a] for a in rig if a != 31}, rig, pens,
                      staged.H_INV_DEFAULT)
        assert frozen.active()
        held = paper.effective_static_floor(spec, q0, q1, spec.pen)
    finally:
        staged.thaw()
    # the partner may only ever RAISE the floor, and never above FRAME_FLOOR
    assert held >= free - 1e-12
    assert held >= min(paper.ROOM_FLOOR, paper.FRAME_FLOOR) - 1e-12
    assert held <= paper.FRAME_FLOOR + 1e-12


def test_room_only_clearance_is_chain_clearance_without_the_partners(rig):
    """The seam the fix runs through, pinned."""
    parks = staged.shipped_parks(rig)
    pens = {a: rig[a].pen for a in rig}
    spec = rig[31]
    qs = np.stack([np.asarray(parks[31], float)])
    boxes = paper.static_boxes(spec)
    staged.thaw()
    a = paper.chain_static(qs, spec, spec.pen, boxes=boxes)
    b = paper._static_only(qs, spec, spec.pen, boxes=boxes)
    assert a == pytest.approx(b)            # nobody frozen: identical
    try:
        frozen.freeze({x: parks[x] for x in rig if x != 31}, rig, pens,
                      staged.H_INV_DEFAULT)
        c = paper.chain_static(qs, spec, spec.pen, boxes=boxes)
        d = paper._static_only(qs, spec, spec.pen, boxes=boxes)
    finally:
        staged.thaw()
    assert d == pytest.approx(b)            # the metal never moved
    assert float(c.min()) <= float(d.min()) + 1e-12   # partners only subtract
