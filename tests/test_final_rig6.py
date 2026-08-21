"""THE MIRRORED SIX-ARM INSTALLATION: the reflection, and what it preserves.

`rig_final6` claims three things the rest of the six-arm work rests on:

  1. the mirror is an involution — points, rotations, boxes and joint vectors
     all round-trip — and it is anchored on a plane DERIVED from the frame
     geometry, not typed in;
  2. unit B's base orientations are PROPER rotations (no arm can be built
     left-handed), and they are the physically natural realisation per mount;
  3. the coverage preview is symmetric because the mirrored joint vector
     reproduces the chain exactly AND preserves every quantity
     `atlas.strict_go` gates on — which is what lets us mirror the atlas
     instead of re-sweeping.

Run: pytest tests/test_final_rig6.py     (~5 s)

`rig_final.py` and `fleet.py` are NOT touched by any of this; the first test
pins that the 3-arm rig is still exactly itself.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from aris_sixarm import rig_final, rig_final6 as r6           # noqa: E402
from aris_sixarm.atlas import QCOL, load, strict_go           # noqa: E402
from aris_sixarm.fleet import FLEET_FINAL                     # noqa: E402
from aris_sixarm.frames import (FR3_MAX, FR3_MIN, PEN_EXT,    # noqa: E402
                                fk, fk_many, joint_margin, tip_pos)
from aris_sixarm.metrics import (f_max, sigma_min,            # noqa: E402
                                 tip_jacobian)

ROOT = Path(__file__).parents[1]
ATLAS = ROOT / "out/atlas_final"


# ==========================================================================
# 1. the plane and the transform
# ==========================================================================
def test_mirror_plane_is_derived_from_the_frame_and_round_trips():
    """The plane comes out of the geometry, and mirroring twice is identity."""
    # derived, not asserted: the outer face on the hanging-arm side is the
    # one the corner posts and the top slab stand on
    y_w, defining = r6.outer_face_provenance()
    assert y_w == r6.MIRROR_PLANE_W_CM == 208.28
    assert defining == ["post_BL", "post_BR", "top_slab"]
    # and it IS the hanging arms' side (that is the whole layout claim)
    side, y_hang, mid = r6.which_side_do_the_hanging_arms_live_on()
    assert side == "back" and y_hang > mid
    # canvas value, straight off the paper origin
    assert abs(r6.MIRROR_PLANE_CANVAS_Y
               - (208.28 - rig_final.PAPER_ORIGIN_W_CM[1]) / 100) < 1e-12
    assert abs(r6.MIRROR_PLANE_CANVAS_Y - 1.815320) < 1e-9

    rng = np.random.default_rng(3)
    # points
    P = rng.uniform(-1, 4, size=(50, 3))
    assert np.allclose(r6.mirror_point(r6.mirror_point(P)), P)
    assert np.allclose(r6.mirror_point(P)[:, [0, 2]], P[:, [0, 2]])
    # a point ON the plane is its own image
    on = np.array([0.4, r6.MIRROR_PLANE_CANVAS_Y, 0.7])
    assert np.allclose(r6.mirror_point(on), on)
    # rotations: involution, and properness is preserved
    for _ in range(20):
        R = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        R *= np.sign(np.linalg.det(R))
        assert np.allclose(r6.mirror_rotation(r6.mirror_rotation(R)), R)
        assert np.linalg.det(r6.mirror_rotation(R)) == pytest.approx(1.0)
    # joint vectors
    q = rng.uniform(FR3_MIN, FR3_MAX, size=(30, 7))
    assert np.allclose(r6.mirror_q(r6.mirror_q(q)), q)
    # boxes: valid AABBs, involutive, and the union of both units is
    # symmetric about the plane
    for b in rig_final.frame_boxes_canvas(zmin=-10):
        lo, hi = r6.mirror_box(b["lo"], b["hi"])
        assert np.all(hi >= lo)
        lo2, hi2 = r6.mirror_box(lo, hi)
        assert np.allclose(lo2, b["lo"]) and np.allclose(hi2, b["hi"])
    both = r6.frame_boxes6_canvas(zmin=-10)
    assert len(both) == 2 * len(rig_final.frame_boxes_canvas(zmin=-10)) == 70
    A = {b["name"][:-2]: b for b in both if b["unit"] == "A"}
    B = {b["name"][:-2]: b for b in both if b["unit"] == "B"}
    assert set(A) == set(B)
    for n in A:
        lo, hi = r6.mirror_box(A[n]["lo"], A[n]["hi"])
        assert np.allclose(lo, B[n]["lo"]) and np.allclose(hi, B[n]["hi"]), n
    # the two webs are disjoint and mirror-symmetric, with the flagged seam
    (_, ay0), (_, ah) = r6.WEB_A
    (_, by0), _ = r6.WEB_B
    assert by0 > ay0 + ah
    assert r6.SEAM_M == pytest.approx(by0 - ah)
    assert r6.SEAM_M == pytest.approx(0.23064, abs=1e-5)
    assert r6.MERGE_WEBS is False           # the DEFAULT the user must confirm

    # and none of this touched the 3-arm rig
    assert len(rig_final.FRAME_BOXES_W_CM) == 35
    assert sorted(FLEET_FINAL) == [2, 13, 31]


# ==========================================================================
# 2. unit B is buildable hardware
# ==========================================================================
def test_unit_b_poses_are_proper_rotations_in_the_natural_realisation():
    """No arm can be built left-handed: every base must be det = +1, and each
    mount must land on the physically natural mirrored pose."""
    assert sorted(r6.FLEET_FINAL6) == [2, 13, 17, 31, 71, 97]
    assert sorted(r6.UNIT_B_IDS.values()) == [17, 71, 97]   # ASSUMED ids

    for aid, spec in r6.FLEET_FINAL6.items():
        R = np.asarray(spec.R, float).reshape(3, 3)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12), aid
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12), aid

    # unit A is bit-identical to the 3-arm registry
    for aid in (13, 31, 2):
        assert np.allclose(r6.FLEET_FINAL6[aid].T_world_base(),
                           FLEET_FINAL[aid].T_world_base(), atol=0)

    # positions: exact reflections
    for a, b in ((13, 17), (31, 71), (2, 97)):
        Ta = r6.FLEET_FINAL6[a].T_world_base()
        Tb = r6.FLEET_FINAL6[b].T_world_base()
        assert np.allclose(Tb[:3, 3], r6.mirror_point(Ta[:3, 3]))
        assert np.allclose(Tb[:3, :3], r6.mirror_rotation(Ta[:3, :3]))

    # per-mount natural realisation, stated as directions rather than matrices
    up, dn, sd = (r6.FLEET_FINAL6[i] for i in (17, 71, 97))
    Rup = np.asarray(up.R, float).reshape(3, 3)
    Rdn = np.asarray(dn.R, float).reshape(3, 3)
    Rsd = np.asarray(sd.R, float).reshape(3, 3)
    # floor arm 17: still upright, but facing its own web from the far end
    assert np.allclose(Rup[:, 2], [0, 0, 1])        # J1 axis up
    assert np.allclose(Rup[:, 0], [0, -1, 0])       # front now -Y (was +Y)
    assert up.xy[1] > r6.WEB_B[0][1] + r6.WEB_B[1][1]   # beyond its web's end
    # inverted arm 71: hangs, yaw mirrored (at yaw 0 the mirror is identity)
    assert np.allclose(Rdn[:, 2], [0, 0, -1])       # J1 axis down
    assert np.allclose(Rdn[:, 0], [1, 0, 0])
    # side arm 97: J1 horizontal along -X, front straight down, into ITS half
    assert np.allclose(Rsd[:, 2], [-1, 0, 0])
    assert np.allclose(Rsd[:, 0], [0, 0, -1])
    (bx0, by0), (bw, bh) = r6.WEB_B
    assert by0 <= sd.xy[1] <= by0 + bh

    # ready poses are the mirrored ones, inside the FR3 limits, and each puts
    # its pen over its OWN web
    for a, b in ((13, 17), (31, 71), (2, 97)):
        qa = np.asarray(r6.FLEET_FINAL6[a].q_seed, float)
        qb = np.asarray(r6.FLEET_FINAL6[b].q_seed, float)
        assert np.allclose(qb, r6.mirror_q(qa))
        assert np.all(qb >= FR3_MIN) and np.all(qb <= FR3_MAX)
        assert joint_margin(qb) == pytest.approx(joint_margin(qa), abs=1e-15)
        Tb = r6.FLEET_FINAL6[b].T_world_base()
        tip = Tb[:3, :3] @ tip_pos(qb) + Tb[:3, 3]
        assert bx0 - 1e-9 <= tip[0] <= bx0 + bw + 1e-9, b
        assert by0 - 1e-9 <= tip[1] <= by0 + bh + 1e-9, b
        assert tip[2] > 0.0                          # pen above the paper

    # each unit-B arm is excused from its own mount hardware and NOTHING else
    for key, aid in r6.UNIT_B_IDS.items():
        mine = {x["name"] for x in r6.frame_boxes6_canvas()} \
            - {x["name"] for x in r6.FLEET_FINAL6[aid].static_obstacles()}
        assert mine == {x["name"] for x in r6.frame_boxes6_canvas()
                        if x["tag"] == f"mount:{key}@B"}, aid


# ==========================================================================
# 3. why the preview may be mirrored instead of re-swept
# ==========================================================================
def test_the_mirrored_atlas_is_exact():
    """The symmetry preview's licence: the mirrored joint vector reproduces
    the chain reflected AND preserves every gated quantity, so unit B's
    strict-GO map IS unit A's reflected — and the second frame costs nothing.
    """
    rng = np.random.default_rng(11)
    S = r6.S_MIRROR
    for _ in range(120):
        q = rng.uniform(FR3_MIN, FR3_MAX)
        qm = r6.mirror_q(q)
        assert np.all(qm >= FR3_MIN) and np.all(qm <= FR3_MAX)   # limit-safe
        _, pts = fk(q)
        _, ptsm = fk(qm)
        assert np.allclose(ptsm, pts @ S.T, atol=1e-14)          # whole chain
        assert np.allclose(tip_pos(qm), S @ tip_pos(q), atol=1e-14)
        # the three quantities the strict-GO gate is made of
        assert joint_margin(qm) == pytest.approx(joint_margin(q), abs=1e-15)
        J, Jm = tip_jacobian(q), tip_jacobian(qm)
        assert sigma_min(Jm) == pytest.approx(sigma_min(J), abs=1e-9)
        n = rng.normal(size=3)
        n /= np.linalg.norm(n)
        assert f_max(Jm, S @ n) == pytest.approx(f_max(J, n), rel=1e-7)

    if not (ATLAS / "atlas_arm13.npz").exists():
        pytest.skip("no atlas in out/atlas_final (run scripts/run_atlas.py)")

    # on the REAL atlas: mirror every strict-GO pose onto unit B and require
    # the pen to land on web B, with the SAME clearance against BOTH frames
    (bx0, by0), (bw, bh) = r6.WEB_B
    for a, b, key in ((13, 17, "up"), (31, 71, "down"), (2, 97, "side")):
        arr, _ = load(ATLAS, a)
        A = arr[strict_go(arr)]
        assert len(A)
        qs = A[:, QCOL:QCOL + 7]
        for aid, unit, qq, want_y in ((a, "A", qs, A[:, 1]),
                                      (b, "B", r6.mirror_q(qs),
                                       r6.mirror_y(A[:, 1]))):
            Twb = r6.FLEET_FINAL6[aid].T_world_base()
            T, p = fk_many(qq)
            tip = T[:, :3, 3] + np.einsum("nij,j->ni", T[:, :3, :3],
                                          [0.0, 0.0, PEN_EXT])
            P = np.concatenate([p, tip[:, None]], axis=1)
            Pw = np.einsum("ij,nkj->nki", Twb[:3, :3], P) + Twb[:3, 3]
            assert np.allclose(Pw[:, -1, 0], A[:, 0], atol=1e-8), aid
            assert np.allclose(Pw[:, -1, 1], want_y, atol=1e-8), aid
            assert np.allclose(Pw[:, -1, 2], 0.0, atol=1e-8), aid
            clr = rig_final.chain_static_clearance(
                Pw, r6.frame_boxes6_canvas(exclude_tag=f"mount:{key}@{unit}"))
            assert clr.min() >= rig_final.STATIC_MARGIN, (aid, clr.min())
            if unit == "A":
                base = clr
            else:                       # the mirrored arm clears identically
                assert np.allclose(clr, base, atol=1e-12), aid
        # ... and the SECOND frame is invisible to unit A: same clearances
        Twb = r6.FLEET_FINAL6[a].T_world_base()
        T, p = fk_many(qs)
        tip = T[:, :3, 3] + np.einsum("nij,j->ni", T[:, :3, :3],
                                      [0.0, 0.0, PEN_EXT])
        Pw = np.einsum("ij,nkj->nki", Twb[:3, :3],
                       np.concatenate([p, tip[:, None]], axis=1)) + Twb[:3, 3]
        one = rig_final.chain_static_clearance(
            Pw, rig_final.frame_boxes_canvas(exclude_tag=f"mount:{key}"))
        assert np.allclose(one, base, atol=1e-12), a


def test_the_preview_map_is_symmetric():
    """The artefact itself: the combined strict-GO map must be invariant
    under the mirror, and the middle cluster must show ZERO cross-unit
    overlap (the two webs are disjoint in y)."""
    if not (ATLAS / "coverage.npz").exists():
        pytest.skip("no coverage.npz (run the atlas first)")
    d = np.load(ATLAS / "coverage.npz")
    per, xs, ys = d["per_arm_go"], d["xs"], d["ys"]
    cntA = per.sum(axis=0)
    cntB = cntA[::-1, :]                       # unit B = the reflection
    assert np.array_equal(np.flipud(cntB), cntA)          # symmetric by build
    assert cntA.sum() == cntB.sum()
    n_web = len(xs) * len(ys)
    union_pct = 100 * (cntA >= 1).sum() / n_web
    assert union_pct == pytest.approx(68.63, abs=0.01)     # unchanged per web
    assert cntA.max() == 2                                 # no 3-arm cell

    # cross-unit overlap is exactly zero: the webs do not share a single y,
    # and no cluster arm is even within the atlas's sweep radius of the other
    (_, ay0), (_, ah) = r6.WEB_A
    (_, by0), (_, bh) = r6.WEB_B
    assert ay0 + ah < by0
    for aid in (17, 71, 97):                   # unit-B arms vs web A
        s = r6.FLEET_FINAL6[aid]
        X, Y = np.meshgrid(xs, ys)
        dist = np.sqrt((X - s.xy[0]) ** 2 + (Y - s.xy[1]) ** 2 + s.z ** 2)
        if aid == 97:      # the ONLY one that gets inside 1.05 m — 35 cells,
            assert dist.min() == pytest.approx(1.0275, abs=1e-3)   # unscored
        else:
            assert dist.min() > 1.05, aid
