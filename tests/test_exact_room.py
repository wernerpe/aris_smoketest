"""The exact trajectory room: is it the distance it claims, and is it fast?

The claims this file has to hold down are the ones the room's whole argument
rests on:

  1. A CAPSULE IS MEASURED AS A CAPSULE.  A hand-built case with a distance
     anybody can compute on paper, so the index cannot quietly answer a
     different question than the geometry.
  2. THE INDEX IS AN INDEX, NOT AN APPROXIMATION.  Whatever the grid cell, the
     answer is bit-identical to the brute-force minimum over every capsule.
  3. THE SPHERE ROOM IS A BOUND ON IT.  `cluster_capsules` CONTAINS the
     capsules it replaces, so the sphere answer is never above the exact one —
     that is the property that makes the old room conservative and the new one
     an improvement rather than a relaxation.
  4. THE FLOOR CONTRACT.  Given a floor the number is only guaranteed to be on
     the right side of it (`paper.leg_static_lb`'s own words), and below the
     floor it is exact.
  5. THE SWEEP PAD IS STILL HONEST, decimation included.
"""
import time

import numpy as np
import pytest

from aris_sixarm import coordination, exact_room, frames, paper, rig_final, staged
from aris_sixarm import fleet as fleet_mod
from aris_sixarm.fleet import H_INV_DEFAULT

CAPS = rig_final.STATIC_CAPSULES_LAT

# NO ENVIRONMENT VARIABLES.  The rig and the tool are switched IN PROCESS and
# put back afterwards, exactly as `tests/test_staged.py` does it: `ARIS_RIG` /
# `ARIS_TOOL` are the launcher's door and a test that set them would be
# changing the run it is part of.


@pytest.fixture(scope="module")
def rig():
    rig0, tool0 = fleet_mod.ACTIVE_RIG, frames.ACTIVE_TOOL
    fleet_mod.activate("proposed")
    frames.activate_tool("lateral")
    paper.clear_cache()
    yield fleet_mod.FLEET
    staged.thaw()
    frames.activate_tool(tool0)
    fleet_mod.activate(rig0)
    paper.clear_cache()


def brute(room, P, caps=CAPS):
    """The minimum over every capsule, with no index at all. -> (N,)."""
    P = np.asarray(P, float)
    out = np.full(len(P), np.inf)
    for i, j, r in caps:
        d = coordination.seg_seg_dist(P[:, i][:, None, :], P[:, j][:, None, :],
                                      room.A[None], room.B[None])
        out = np.minimum(out, (d - room.R[None, :]).min(axis=1) - r)
    return out


def tube(fleet, arm=71, n=400, scale=0.5, seed=3):
    """A realistic swept capsule chain for one arm. -> (A3, B3, r)."""
    rng = np.random.default_rng(seed)
    q0 = np.asarray(fleet[arm].q_seed, float)
    t = np.linspace(0, 1, n)[:, None]
    amp = scale * np.array([0.6, 0.35, 0.5, 0.4, 0.6, 0.4, 0.8])
    w = np.array([1.0, 2.0, 3.0, 1.5, 2.5, 2.0, 1.2])
    Q = q0[None] + amp[None] * np.sin(2 * np.pi * (t * w[None]
                                                   + rng.random(7)[None]))
    p = coordination.ArmPath(arm, Q, 0.05, H_INV_DEFAULT,
                             float(fleet[arm].pen), fleet[arm])
    keep = [k for k in range(len(p.r))
            if k not in coordination.FROZEN_SWEEP_BANDS]
    return (np.asarray(p.A, float)[:, keep], np.asarray(p.B, float)[:, keep],
            np.asarray(p.r, float)[keep])


def observer(fleet, arm=31, n=120, scale=0.6, seed=5):
    rng = np.random.default_rng(seed)
    Q = np.asarray(fleet[arm].q_seed, float)[None] + scale * rng.normal(
        0, 1, (n, 7))
    return coordination.chain_world(Q, fleet[arm], H_INV_DEFAULT,
                                    float(fleet[arm].pen))


# --------------------------------------------------------------------------
# 1.  a capsule, measured by hand
# --------------------------------------------------------------------------
def test_one_capsule_reads_the_distance_you_can_compute_on_paper(rig):
    # the unit segment on the x axis, radius 50 mm
    room = exact_room.ExactRoom([[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]], [0.05])
    # an observer POINT 300 mm away in y, radius 20 mm -> 300 - 50 - 20
    g = room.segment_clearance([[0.5, 0.3, 0.0]], [[0.5, 0.3, 0.0]], [0.02])
    assert g[0] == pytest.approx(0.23, abs=1e-12)
    # ...off the end, where the capsule is a hemisphere: the axis endpoint is
    # at (1,0,0), so a point at (1.4,0,0) is 400 mm from the axis
    g = room.segment_clearance([[1.4, 0.0, 0.0]], [[1.4, 0.0, 0.0]], [0.0])
    assert g[0] == pytest.approx(0.35, abs=1e-12)
    # ...and a point INSIDE reads negative, which is the sign the gates need
    g = room.segment_clearance([[0.5, 0.02, 0.0]], [[0.5, 0.02, 0.0]], [0.0])
    assert g[0] == pytest.approx(-0.03, abs=1e-12)


def test_a_degenerate_capsule_is_a_sphere(rig):
    c = np.array([[0.3, 0.4, 0.5]])
    room = exact_room.from_spheres(c, [0.1])
    assert room.kind == "spheres"
    g = room.segment_clearance([[0.3, 0.4, 0.9]], [[0.3, 0.4, 0.9]], [0.0])
    assert g[0] == pytest.approx(0.3, abs=1e-12)


# --------------------------------------------------------------------------
# 2.  the index answers what brute force answers, at every cell size
# --------------------------------------------------------------------------
@pytest.mark.parametrize("cell", [0.05, 0.075, 0.10, 0.20, 0.50])
def test_the_grid_hash_is_exact_at_every_cell_size(cell, rig):
    A3, B3, r = tube(rig, n=250)
    room = exact_room.from_samples(A3, B3, r, staged.SWEEP_FRAC, cell=cell)
    P = observer(rig, n=60)
    got = room.chain_clearance(P, CAPS)
    assert np.abs(got - brute(room, P)).max() == 0.0


def test_the_cell_size_changes_the_cost_and_not_the_answer(rig):
    A3, B3, r = tube(rig, n=250)
    P = observer(rig, n=60)
    ref = None
    for cell in (0.06, 0.12, 0.30):
        room = exact_room.from_samples(A3, B3, r, staged.SWEEP_FRAC, cell=cell)
        got = room.chain_clearance(P, CAPS)
        if ref is None:
            ref = got
        assert np.array_equal(got, ref)


# --------------------------------------------------------------------------
# 3.  the sphere room is a BOUND on the capsule room
# --------------------------------------------------------------------------
def test_the_sphere_reduction_never_reads_above_the_capsules_it_contains(rig):
    """`cluster_capsules` CONTAINS its members, so it can only under-read.

    This is the property the old room was conservative by, and the number
    between the two is what the follower was paying: a median of ~0.1-0.2 m of
    clearance that the leader's metal was not actually occupying.
    """
    A3, B3, r = tube(rig, n=300)
    exact = exact_room.from_samples(A3, B3, r, staged.SWEEP_FRAC)
    step = max(float(np.max(np.linalg.norm(np.diff(A3, axis=0), axis=2))),
               float(np.max(np.linalg.norm(np.diff(B3, axis=0), axis=2))))
    c, rad = staged.cluster_capsules(A3.reshape(-1, 3), B3.reshape(-1, 3),
                                     np.tile(r, len(A3)),
                                     staged.ENVELOPE_CLUSTER,
                                     staged.SWEEP_FRAC * step)
    sph = exact_room.from_spheres(c, rad)
    P = observer(rig, n=120)
    ge, gs = exact.chain_clearance(P, CAPS), sph.chain_clearance(P, CAPS)
    assert bool(np.all(gs <= ge + 1e-9)), "the sphere room read ABOVE the exact"
    assert float(np.median(ge - gs)) > 0.05, "the reduction should cost real mm"


# --------------------------------------------------------------------------
# 4.  the floor contract
# --------------------------------------------------------------------------
@pytest.mark.parametrize("floor", [0.05, 0.063, 0.25])
def test_a_floored_query_is_exact_below_the_floor_and_honest_above(floor, rig):
    A3, B3, r = tube(rig, n=250)
    room = exact_room.from_samples(A3, B3, r, staged.SWEEP_FRAC)
    P = observer(rig, n=120, scale=0.9)
    ref = room.chain_clearance(P, CAPS)
    got = room.chain_clearance(P, CAPS, floor=floor)
    below = ref < floor
    assert np.array_equal(got[below], ref[below])
    assert bool(np.all(got[~below] >= floor))


# --------------------------------------------------------------------------
# 5.  the sweep pad, and what decimation costs
# --------------------------------------------------------------------------
def test_the_pad_is_the_checkers_own_per_sample_residual(rig):
    A3, B3, r = tube(rig, n=60)
    pad = exact_room.sweep_pads(A3, B3, staged.SWEEP_FRAC)
    d = np.maximum(np.linalg.norm(np.diff(A3, axis=0), axis=2).max(axis=1),
                   np.linalg.norm(np.diff(B3, axis=0), axis=2).max(axis=1))
    # every interior sample carries the larger of the step in and the step out
    assert pad[3] == pytest.approx(staged.SWEEP_FRAC * max(d[2], d[3]))
    assert pad[0] == pytest.approx(staged.SWEEP_FRAC * d[0])
    assert pad[-1] == pytest.approx(staged.SWEEP_FRAC * d[-1])
    room = exact_room.from_samples(A3, B3, r, staged.SWEEP_FRAC)
    assert room.R.reshape(len(A3), -1)[3, 0] == pytest.approx(r[0] + pad[3])


def test_decimation_grows_the_pad_so_the_room_still_contains_the_motion(rig):
    """A stride > 1 must be CONSERVATIVE, not merely cheaper.

    The retained samples span more motion, so each one has to carry it; the
    test is that the decimated room never reads FURTHER from an observer than
    the stride-1 room it replaces.
    """
    A3, B3, r = tube(rig, n=201)
    full = exact_room.from_samples(A3, B3, r, staged.SWEEP_FRAC, stride=1)
    P = observer(rig, n=80, scale=0.8)
    g_full = full.chain_clearance(P, CAPS)
    for stride in (2, 4, 8):
        thin = exact_room.from_samples(A3, B3, r, staged.SWEEP_FRAC,
                                       stride=stride)
        assert len(thin) < len(full)
        g = thin.chain_clearance(P, CAPS)
        assert bool(np.all(g <= g_full + 1e-9)), \
            f"stride {stride} read further than stride 1"


def test_the_union_of_two_rooms_contains_both(rig):
    A3, B3, r = tube(rig, n=120, seed=1)
    a = exact_room.from_samples(A3[:60], B3[:60], r, staged.SWEEP_FRAC)
    b = exact_room.from_samples(A3[60:], B3[60:], r, staged.SWEEP_FRAC)
    u = exact_room.union([a, b])
    assert len(u) == len(a) + len(b)
    P = observer(rig, n=60)
    ga, gb, gu = (x.chain_clearance(P, CAPS) for x in (a, b, u))
    assert np.abs(gu - np.minimum(ga, gb)).max() == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------
# 6.  the query is fast enough for the planner's gates
# --------------------------------------------------------------------------
def test_one_pose_costs_the_planner_under_a_millisecond(rig):
    """The budget the build was given: <= 1 ms per pose at the router's floor.

    A stage-sized room (a few thousand samples) against one arm's chain, at
    `paper.FRAME_FLOOR`, which is how every gate in `paper.route` asks.
    """
    A3, B3, r = tube(rig, n=2000)
    room = exact_room.from_samples(A3, B3, r, staged.SWEEP_FRAC)
    assert len(room) > 15000, "not a stage-sized room"
    P = observer(rig, n=200)
    room.chain_clearance(P[:5], CAPS, floor=0.063)          # warm
    t0 = time.perf_counter()
    room.chain_clearance(P, CAPS, floor=0.063)
    per = (time.perf_counter() - t0) / len(P)
    assert per < 1e-3, f"{1000 * per:.3f} ms per pose"


# --------------------------------------------------------------------------
# 7.  the boxes the go-around tier is shown
# --------------------------------------------------------------------------
def test_the_go_around_boxes_cover_the_room_they_stand_for(rig):
    A3, B3, r = tube(rig, n=200)
    room = exact_room.from_samples(A3, B3, r, staged.SWEEP_FRAC)
    bx = room.boxes()
    assert 0 < len(bx) <= exact_room.MAX_BOXES
    assert all(set(b) >= {"name", "lo", "hi"} for b in bx)
    lo = np.min([b["lo"] for b in bx], axis=0)
    hi = np.max([b["hi"] for b in bx], axis=0)
    # the boxes kept are the biggest ones, so they need not cover every
    # capsule -- but they must lie inside the room's own extent plus its radii
    rm = float(room.R.max())
    assert bool(np.all(lo >= room.A.min(axis=0).min()
                       - rm - exact_room.BOX_CELL - 1e-9))
    assert bool(np.all(hi <= room.A.max(axis=0).max()
                       + rm + exact_room.BOX_CELL + 1e-9))


def test_an_empty_room_is_infinitely_far_and_has_no_boxes(rig):
    room = exact_room.ExactRoom(np.zeros((0, 3)), np.zeros((0, 3)),
                                np.zeros(0))
    assert len(room) == 0 and room.boxes() == []
    P = observer(rig, n=4)
    assert bool(np.all(np.isinf(room.chain_clearance(P, CAPS))))


# --------------------------------------------------------------------------
# 8.  the flag, and the follower's pre-position
# --------------------------------------------------------------------------
def test_the_room_flag_selects_the_model_and_nothing_else(rig, monkeypatch):
    """`ARIS_ROOM` picks the obstacle; both arrive through the same tuple."""
    monkeypatch.delenv("ARIS_ROOM", raising=False)
    assert staged.room_mode() == "capsules"
    monkeypatch.setenv("ARIS_ROOM", "spheres")
    assert staged.room_mode() == "spheres"
    monkeypatch.setenv("ARIS_ROOM", "")          # empty is NOT SET, not off
    assert staged.room_mode() == "capsules"


def _stub_stage(fleet, arm=71, n=200):
    q0 = np.asarray(fleet[arm].q_seed, float)
    t = np.linspace(0, 1, n)[:, None]
    Q = q0[None] + 0.4 * np.sin(2 * np.pi * t * np.arange(1, 8)[None])
    st = staged.ArmStage(0, arm, q0)
    st.timeline = dict(t=np.linspace(0, 10, n), q=Q, duration=10.0,
                       seg=np.full(n, -1, int), u=np.zeros(n))
    return st


def test_both_room_modes_build_the_same_tuple_shape(rig):
    st = _stub_stage(rig)
    pens = {a: rig[a].pen for a in rig}
    cap = staged.trajectory_room(st, rig, pens, H_INV_DEFAULT, mode="capsules")
    sph = staged.trajectory_room(st, rig, pens, H_INV_DEFAULT, mode="spheres")
    assert cap[2] == sph[2], "the digest is of the TRAJECTORY, not the room"
    assert isinstance(cap[0], exact_room.ExactRoom) and cap[0].kind == "capsules"
    assert len(cap[0]) > len(sph[1]), "capsules should outnumber spheres"
    # the two rooms must be distinguishable to every cache that keys on one
    assert staged.room_sig(cap) != staged.room_sig(sph)
    assert "cap" in staged.room_size(cap) and "sph" in staged.room_size(sph)


def test_an_exact_room_reaches_the_planner_through_the_frozen_seam(rig):
    """The swap is inside `frozen`, so every gate sees it without knowing."""
    from aris_sixarm import frozen
    st = _stub_stage(rig)
    pens = {a: rig[a].pen for a in rig}
    parks = staged.shipped_parks(rig)
    room = staged.trajectory_room(st, rig, pens, H_INV_DEFAULT, mode="capsules")
    try:
        staged.freeze_stage(31, parks, {71: room}, rig, pens, H_INV_DEFAULT,
                            leg_cache=False)
        assert frozen.room_kinds()[71] == "capsules"
        assert all(v == "pose" for a, v in frozen.room_kinds().items() if a != 71)
        # the go-around tier gets boxes for the room and for nothing else
        bx = frozen.room_boxes()
        assert bx and all(b["name"].startswith("room:71") for b in bx)
        P = observer(rig, n=20)
        g = frozen.partner_clearance(P, floor=0.063)
        assert np.isfinite(g).all()
    finally:
        staged.thaw()


def test_a_tuck_is_only_searched_when_the_arm_is_not_already_clear(rig):
    """An arm already outside the room does not move, and says so."""
    spec = rig[31]
    q = np.asarray(spec.q_seed, float)
    # nothing frozen: the only room is the shipped static set, and the park
    # clears it comfortably, so `tuck_pose` returns the pose it was given
    staged.thaw()
    got, info = staged.tuck_pose(spec, [], q, H_INV_DEFAULT, spec.pen)
    assert got is not None and np.array_equal(got, q)
    assert info["moved"] is False
    assert info["park_mm"] >= info["need_mm"]


def test_a_tuck_that_cannot_clear_returns_none_rather_than_a_bad_pose(rig):
    """A refusal is a refusal; it must not hand back something it rejected."""
    spec = rig[31]
    q = np.asarray(spec.q_seed, float)
    staged.thaw()
    # an impossible bar: nothing in the workspace stands 10 m from the steel
    got, info = staged.tuck_pose(spec, [], q, H_INV_DEFAULT, spec.pen,
                                 floor=10.0)
    assert got is None and info["kept"] == 0


def test_splicing_the_clear_out_puts_it_on_one_clock(rig):
    """The pre-move has to be IN the timeline or no gate ever sees it."""
    n, m = 5, 7
    pre = dict(t=np.linspace(0, 2.0, n), q=np.zeros((n, 7)),
               seg=np.full(n, -1, int), u=np.zeros(n), duration=2.0,
               phases=[dict(kind="aside", seg=-1, t0=0.0, t1=2.0)], ink=[],
               lifts=[], paper_modes=["skirt"], transit_s=0.0, taxi_s=0.0,
               retreat_s=0.0, aside_s=2.0, draw_s=0.0, draw_len=0.0,
               transit_len=0.3, paper_vias=0, fallbacks=0, n_home=0,
               dense_tip_err=0.001)
    main = dict(t=np.linspace(0, 5.0, m), q=np.ones((m, 7)),
                seg=np.zeros(m, int), u=np.linspace(0, 1, m), duration=5.0,
                phases=[dict(kind="stroke", seg=0, t0=0.0, t1=5.0)],
                ink=[(1.0, np.zeros((3, 3)))], lifts=[0.06],
                paper_modes=["direct"], transit_s=1.0, taxi_s=0.0,
                retreat_s=0.0, aside_s=0.0, draw_s=4.0, draw_len=0.5,
                transit_len=0.2, paper_vias=1, fallbacks=0, n_home=0,
                dense_tip_err=0.002, q_end=np.ones(7), park="freeze", pen=0.11)
    out = staged.splice_timeline(pre, main)
    assert len(out["t"]) == n + m - 1, "the junction sample is shared, not doubled"
    assert bool(np.all(np.diff(out["t"]) > 0)), "the clock must be monotone"
    assert out["duration"] == pytest.approx(7.0, abs=1e-6)
    assert [p["kind"] for p in out["phases"]] == ["aside", "stroke"]
    assert out["phases"][1]["t0"] == pytest.approx(2.0)
    assert out["ink"][0][0] == pytest.approx(3.0), "ink must move with the clock"
    assert out["aside_s"] == pytest.approx(2.0)
    assert out["draw_s"] == pytest.approx(4.0)
    assert np.array_equal(out["q_end"], main["q_end"])
    # a zero-length pre-move is not a splice at all
    assert staged.splice_timeline(None, main) is main
