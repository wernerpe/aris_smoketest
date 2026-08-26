"""The arm against itself: the model, its pair rule, and its three copies.

The guard is new geometry AND a new gate in five certifying stages, so what
these pin is (a) that the capsule table still contains the metal it was fitted
to, (b) that the pair rule is what the measurement said it should be, (c) that
the producer and the two checkers agree, and (d) that the gate is neither
vacuous (it refuses folded poses) nor expensive (it refuses nothing the
shipped certified map already contains).

Nothing here needs python-fcl: the mesh work lives in
`scripts/self_collision_audit.py`, and its record is `out/self_collision_audit.json`.
"""
import json
import os

import numpy as np
import pytest

from aris_sixarm import frames, scene_check, selfcoll, validate
from aris_sixarm.frames import FR3_MAX, FR3_MIN, D_HAND_TCP, fk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT = os.path.join(ROOT, "out", "self_collision_audit.json")


def _rand(n, seed=0, margin=0.0):
    rng = np.random.default_rng(seed)
    Q = FR3_MIN + rng.random((n, 7)) * (FR3_MAX - FR3_MIN)
    if margin > 0:
        m = np.min(np.minimum(Q - FR3_MIN, FR3_MAX - Q), axis=1)
        Q = Q[m >= margin]
    return Q


# --------------------------------------------------------------------------
# the link frames the whole model hangs on
# --------------------------------------------------------------------------
def test_link_frames_reproduce_fk_exactly():
    """`link_frames_many` keeps what `fk` throws away and nothing else."""
    Q = _rand(200, seed=1)
    L = frames.link_frames_many(Q)
    assert L.shape == (len(Q), 10, 4, 4)
    F = np.eye(4)
    F[2, 3] = D_HAND_TCP
    for i, q in enumerate(Q):
        T, P = fk(q)
        # the first eight chain points ARE the first eight link origins
        assert np.array_equal(L[i, :8, :3, 3], P[:8])
        # and the hand frame stepped D_HAND_TCP along its own z is fk's TCP
        assert np.allclose(L[i, 9] @ F, T, atol=1e-14)
        for k in range(10):
            R = L[i, k, :3, :3]
            assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)


def test_link_frames_are_rigid_under_a_base_rotation():
    """Self-clearance is a base-frame quantity: q1 turns the whole arm."""
    q = _rand(1, seed=2)[0]
    a = selfcoll.self_clearance(q[None])[0]
    for d in (0.3, -1.1, 2.0):
        q2 = q.copy()
        q2[0] = np.clip(q[0] + d, FR3_MIN[0], FR3_MAX[0])
        assert selfcoll.self_clearance(q2[None])[0] == pytest.approx(a, abs=1e-9)


# --------------------------------------------------------------------------
# the measured table
# --------------------------------------------------------------------------
def test_the_table_matches_the_audit_record():
    if not os.path.exists(AUDIT):
        pytest.skip("out/self_collision_audit.json not present (gitignored)")
    rows = json.load(open(AUDIT))["capsules"]["rows"]
    assert len(rows) >= len(selfcoll.BODY_CAPSULES)
    by = {(r["body"], r["band"]): r for r in rows}
    for name, band, frame, a, b, r in selfcoll.BODY_CAPSULES:
        rec = by[(name, band)]
        assert rec["frame_index"] == frame
        assert np.allclose(rec["a"], a, atol=1e-4)
        assert np.allclose(rec["b"], b, atol=1e-4)
        # shipped radius is the mesh maximum rounded UP to the millimetre
        assert r >= rec["r_fit"] - 1e-12
        assert r - rec["r_fit"] < 1e-3
        assert abs(r * 1000 - round(r * 1000)) < 1e-9


def test_every_body_is_banded_on_a_real_frame():
    seen = {}
    for name, band, frame, a, b, r in selfcoll.BODY_CAPSULES:
        seen.setdefault(name, []).append(band)
        assert 0 <= frame < len(frames.LINK_FRAMES)
        assert 0.03 < r < 0.20
        assert np.linalg.norm(np.array(b) - np.array(a)) > 1e-3
    assert set(seen) == {f"link{i}" for i in range(8)} | {"hand"}
    for name, bands in seen.items():
        assert sorted(bands) == list(range(len(bands))), name
        assert len(bands) == (7 if name == "link0" else 3), name
    # link0 never moves, so its bands ride the base axis itself
    for name, band, frame, a, b, r in selfcoll.BODY_CAPSULES:
        if name == "link0":
            assert a[0] == a[1] == b[0] == b[1] == 0.0
            assert b[2] > a[2]


def test_the_tool_capsules_are_the_packages_own_envelope():
    """Not the CAD: `rig_final`'s L, which the audit proved contains it."""
    from aris_sixarm import rig_final
    assert selfcoll.TOOL_R_LAT == rig_final.BRACKET_R_LAT == rig_final.PEN_R_LAT
    from aris_sixarm import coordination
    assert selfcoll.TOOL_R_INLINE == coordination.PEN_R
    lat = 0.110
    A, B, R = selfcoll.capsule_ends(np.zeros((1, 7)), pen_ext=0.110, pen_lat=lat)
    T, _ = fk(np.zeros(7))
    tip = T[:3, 3] + T[:3, :3] @ np.array([lat, 0.0, 0.110])
    corner = T[:3, 3] + T[:3, :3] @ np.array([lat, 0.0, 0.0])
    assert np.allclose(A[0, selfcoll.BRACKET], T[:3, 3], atol=1e-12)
    assert np.allclose(B[0, selfcoll.BRACKET], corner, atol=1e-12)
    assert np.allclose(B[0, selfcoll.PEN], tip, atol=1e-12)
    assert R[selfcoll.BRACKET] == selfcoll.TOOL_R_LAT
    # the inline pen collapses the bracket into the pen and changes no answer
    A0, B0, R0 = selfcoll.capsule_ends(np.zeros((1, 7)), pen_ext=0.110,
                                       pen_lat=0.0)
    assert np.allclose(A0[0, selfcoll.BRACKET], B0[0, selfcoll.BRACKET])
    assert R0[selfcoll.PEN] == selfcoll.TOOL_R_INLINE


# --------------------------------------------------------------------------
# the pair rule
# --------------------------------------------------------------------------
def test_the_pair_rule_is_four_joints_and_nothing_else():
    pos = selfcoll.CHAIN_POS
    assert selfcoll.WATCH_CHAIN_D == 4
    for i, j in selfcoll.SELF_PAIRS:
        d = abs(pos[selfcoll.BODY_OF[j]] - pos[selfcoll.BODY_OF[i]])
        assert d >= 4, (selfcoll.NAMES[i], selfcoll.NAMES[j])
    assert len(selfcoll.SELF_PAIRS) == 233


def test_closer_pairs_could_not_have_informed_the_gate():
    """WHY four joints, re-measured rather than asserted.

    Two claims, and the rule stands on both.  (1) Every WATCHED pair opens up:
    a gate over them is answering a question that has more than one answer.
    (2) A gate that also watched the nearer pairs would refuse almost every
    ordinary configuration, for metal that `scripts/self_collision_audit.py`
    measures at 114 mm and more — so it would be a refusal machine, not a
    guard.
    """
    reach = selfcoll.pair_reach(n=40000, d_min=selfcoll.WATCH_CHAIN_D)
    assert len(reach) == len(selfcoll.SELF_PAIRS)
    assert min(reach.values()) > selfcoll.SELF_MARGIN
    assert np.median(list(reach.values())) > 0.10

    Q = _rand(4000, seed=7, margin=0.30)          # ordinary comfortable poses
    keep = selfcoll.self_clearance(Q) >= selfcoll.SELF_MARGIN
    closer = selfcoll.pair_clearance(
        Q, pairs=selfcoll._pairs(selfcoll.WATCH_CHAIN_D - 1)).min(axis=1)
    assert keep.mean() > 0.95, "the shipped rule must pass ordinary poses"
    # ...and a three-joint rule passes ~9 % of the same poses: it is not a
    # stricter guard, it is a different (and wrong) claim about the metal
    assert (closer >= selfcoll.SELF_MARGIN).mean() < 0.15
    assert keep.mean() - (closer >= selfcoll.SELF_MARGIN).mean() > 0.75


# --------------------------------------------------------------------------
# the three copies agree
# --------------------------------------------------------------------------
def test_the_two_checkers_bound_the_producer():
    """Independent derivations: both checkers are LOWER bounds, and close ones.

    `selfcoll` solves the segment pair's stationary point, `validate` samples
    one segment against the other's exact distance and subtracts the residual,
    and `scene_check` runs its own closed form.  A checker may read low — that
    is what `SELF_PLAN_PAD` exists to pay for — and must never read high.
    """
    Q = _rand(400, seed=3)
    a = selfcoll.self_clearance(Q)
    b = validate.self_clearance(Q)
    c = scene_check.self_clearance(Q)
    assert np.all(b <= a + 1e-12)
    assert np.max(a - b) <= selfcoll.SELF_PLAN_PAD
    assert np.allclose(c, a, atol=1e-9)      # a second exact derivation


def test_the_three_margins_are_the_same_number():
    assert validate.SELF_MARGIN == selfcoll.SELF_MARGIN
    assert scene_check.SELF_MARGIN == selfcoll.SELF_MARGIN
    assert validate.SELF_CHAIN_D == selfcoll.WATCH_CHAIN_D
    assert scene_check.SELF_CHAIN_D == selfcoll.WATCH_CHAIN_D
    # the producer pays more than the checker demands, and by enough
    assert selfcoll.SELF_PLAN_MARGIN > selfcoll.SELF_MARGIN
    assert selfcoll.SELF_PLAN_PAD >= selfcoll.SELF_SEG_SLACK


# --------------------------------------------------------------------------
# the gate is neither vacuous nor expensive
# --------------------------------------------------------------------------
def test_the_gate_refuses_folded_poses():
    """It is not decoration: the joint limits alone do not prevent this."""
    Q = _rand(20000, seed=4)
    c = selfcoll.self_clearance(Q)
    assert (c < 0).sum() > 100, "a uniform sample should contain real folds"
    assert c.min() < -0.05
    # ...and the STRICT comfort margin does not save you either
    m = np.min(np.minimum(Q - FR3_MIN, FR3_MAX - Q), axis=1)
    comfy = c[m >= 0.30]
    assert len(comfy) > 200
    assert comfy.min() < selfcoll.SELF_MARGIN


def test_the_gate_passes_every_shipped_ready_pose():
    """The poses an arm is asked to stand in when it is not drawing."""
    for name in ("Q_READY_FLOOR", "Q_READY_INV", "Q_READY_INV_FINAL",
                 "Q_READY_WALL", "Q_READY_WALL_LOW"):
        q = np.asarray(getattr(frames, name), float)
        c = selfcoll.self_clearance(q[None])[0]
        assert c > selfcoll.SELF_PLAN_MARGIN, (name, c)


def test_signature_moves_when_the_model_moves():
    s = selfcoll.signature(pen_lat=0.110)
    assert s.ndim == 1 and len(s) > 100
    assert not np.array_equal(s, selfcoll.signature(pen_lat=0.0))
    from aris_sixarm import atlas
    assert len(atlas.model_signature()) > len(s)


# --------------------------------------------------------------------------
# the atlas invariant the gated search owes its callers
# --------------------------------------------------------------------------
def test_a_gated_atlas_row_is_a_strict_go_row():
    """`min_lean_deg >= 0` MEANS strict-GO, on every row of every atlas.

    The gated search screens with the analytic batch Jacobian and records the
    finite-difference one; they agree to ~3e-10, which is not the same as
    agreeing.  This is the invariant that gap could break.
    """
    import glob
    from aris_sixarm import atlas
    dirs = sorted(glob.glob(os.path.join(ROOT, "out", "atlas_gated*")))
    found = False
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "atlas_arm*.npz"))):
            arr = np.load(f)["data"]
            if not len(arr) or arr.shape[1] <= atlas.LEANCOL:
                continue
            found = True
            gated = arr[:, atlas.LEANCOL] >= 0.0
            go = atlas.strict_go(arr)
            assert np.all(go[gated]), f"{f}: a gated row is not strict-GO"
            # ...and the lean recorded is the lean the pose was found at
            assert np.allclose(arr[gated, 8], arr[gated, atlas.LEANCOL])
            assert set(np.unique(arr[gated, atlas.LEANCOL]).tolist()) <= \
                set((0.0,) + atlas.GATE_CONE_DEG)
    if not found:
        pytest.skip("no gated atlas on disk (out/ is gitignored)")
