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


# ---------------------------------------------------------------------------
# THE CLEAR-OUT IS ROUTED IN THE ROOM ITS DESTINATION IS CHOSEN IN
# ---------------------------------------------------------------------------
def test_the_clear_out_is_routed_in_the_leaders_room(rig, monkeypatch):
    """"IT HAPPENS FIRST" IS NOT A DEFENCE AGAINST A CROSS-PRODUCT CLAIM.

    `_lf_stage` used to route a follower's clear-out with `rooms=None` -- the
    entry fleet and no trajectory rooms at all -- while choosing the tuck POSE
    it flies to WITH the leaders' trajectories in the scene, on the argument
    that the clear-out happens while the leaders are still parked.  But the
    clear-out is spliced into the timeline precisely so `active_pair_gap` sees
    it, and that check's claim is the minimum over the CROSS PRODUCT of two
    arms' pose sets, because inside a stage the actives are asynchronous by
    construction.

    Measured on `bench/scatter` stage 1: follower arm 2's clear-out left its
    entry pose at +63.3 mm against leader 97's realised room, dipped to
    -53.1 mm 0.7 s later and was back over +190 mm by t = 3.4 s; the stage read
    **-53.13 mm** and failed.  Routed in the leaders' room the clear-out is
    refused, the arm stays put, and the stage reads **+57.3 mm** and passes.

    This pins the ROOM the clear-out is routed in, which is the fix.
    """
    seen = []
    real_freeze = staged.freeze_stage

    def spy(arm, poses, rooms, *a, **kw):
        seen.append(None if rooms is None else tuple(sorted(rooms)))
        return real_freeze(arm, poses, rooms, *a, **kw)

    real_clear = staged.clear_out
    calls = []

    def clear_spy(*a, **kw):
        calls.append(tuple(seen))          # the room in force AT THE CALL
        return None                        # refuse: the arm stays put

    monkeypatch.setattr(staged, "freeze_stage", spy)
    monkeypatch.setattr(staged, "clear_out", clear_spy)
    monkeypatch.setattr(staged, "tuck_pose",
                        lambda *a, **kw: (np.asarray(a[2], float).reshape(7)
                                          + 0.01,
                                          dict(moved=True, park_mm=1.0, kept=1,
                                               tried=1, need_mm=73.0,
                                               best_mm=99.0)))
    parks = staged.shipped_parks(rig)
    pens = {a: rig[a].pen for a in rig}
    pcs = [staged.Piece(stage=0, arm=31, line=0, k=0,
                        pts=np.array([[0.7, 1.7], [0.8, 1.7]]), length_m=0.1)]
    staged._lf_stage(
        0, (71, 31), {71: "leader", 31: "follower"},
        {(0, 71): list(pcs), (0, 31): list(pcs)}, rig, pens, dict(parks),
        staged.H_INV_DEFAULT, None, False, None, None, 0.05,
        staged.ENVELOPE_CLUSTER, staged.PAIR_MARGIN, 4, writing_park(),
        True, False, 0, staged.SPLIT_MIN_M, 0.0)
    assert calls, "the clear-out was never reached"
    # the freeze immediately before `clear_out` must carry the leader's room
    last = calls[0][-1]
    assert last is not None and 71 in last, (
        f"the clear-out was routed with rooms={last!r}; it must see the "
        "leaders' trajectories")


def writing_park():
    from aris_sixarm import writing as _w
    return _w.PARK_FREEZE
