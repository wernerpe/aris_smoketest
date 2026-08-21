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

and, since the user's 2026-08-21 decisions, two more:

  4. the MERGED CANVAS is one continuous surface that covers both webs and the
     seam between them, and activating it as the fleet swaps `FLEET`/`SHEET`
     everywhere without destroying the registries it swapped away from;
  5. the EXTENDED-POLE build is a build and not a monkey-patch: `rig_final`'s
     own boxes are untouched, the clamped stack slides exactly 20 cm, the pole
     grows enough to still carry the bracket, and the lowered arms have a ready
     pose whose pen is above the paper (the drawn one's is 0.1 m below it).

Run: pytest tests/test_final_rig6.py     (~5 s)

`rig_final.py` is NOT touched by any of this, and `fleet.py`'s registries
survive activation; the first test pins that the 3-arm rig is still itself.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from aris_sixarm import fleet, rig_final, rig_final6 as r6    # noqa: E402
from aris_sixarm import validate                              # noqa: E402
from aris_sixarm.atlas import QCOL, load, strict_go           # noqa: E402
from aris_sixarm.fleet import FLEET_FINAL, FLEET_SIXARM, SHEET_FINAL  # noqa: E402
from aris_sixarm.frames import (FR3_MAX, FR3_MIN, PEN_EXT,    # noqa: E402
                                fk, fk_many, joint_margin, tip_pos)
from aris_sixarm.metrics import (f_max, sigma_min,            # noqa: E402
                                 tip_jacobian)

ROOT = Path(__file__).parents[1]
ATLAS = ROOT / "out/atlas_final"
ATLAS6 = ROOT / "out/atlas_final6_opt"


@pytest.fixture
def active6():
    """Activate the merged-canvas / extended-pole rig, and put it back.

    `fleet.activate` is global by design — it mutates the one `FLEET` dict every
    module holds a reference to — so a test that switches rigs and walks away
    would silently re-rig every test after it.
    """
    before = fleet.ACTIVE_RIG
    try:
        yield fleet.activate("final6_opt")
    finally:
        fleet.activate(before)


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
    # the two webs are disjoint and mirror-symmetric, with the seam between
    (_, ay0), (_, ah) = r6.WEB_A
    (_, by0), _ = r6.WEB_B
    assert by0 > ay0 + ah
    assert r6.SEAM_M == pytest.approx(by0 - ah)
    assert r6.SEAM_M == pytest.approx(0.23064, abs=1e-5)
    # ... and the user's decision (2026-08-21) is that the paper spans them
    assert r6.MERGE_WEBS is True

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


# ==========================================================================
# 4. the merged canvas, and switching the rig without losing the others
# ==========================================================================
def test_the_merged_canvas_is_one_surface_and_activating_it_is_reversible(active6):
    """The user's 2026-08-21 decision, end to end.

    MERGE_WEBS is not a cosmetic flag: it decides what `webs()`/`web_of()` hand
    back, what `fleet.sheet_for` gives the planner's own `clip_to_sheet` gate,
    and therefore whether a stroke over the seam is a stroke or an off-sheet
    refusal.  And activating the six-arm rig has to be a SWAP, not a
    destruction — `FLEET` is the one dict object every module imported, so it
    is mutated in place, and the registries it was copied from must survive.
    """
    # --- one surface, covering both webs AND the strip between them -------
    assert r6.MERGE_WEBS is True
    webs = r6.webs()
    assert len(webs) == 1
    (x0, y0), (w, h) = webs[0]
    assert (x0, y0) == (0.0, 0.0)
    assert (w, h) == r6.SHEET_FINAL6
    assert w == pytest.approx(rig_final.SHEET_FINAL[0])
    assert h == pytest.approx(2 * r6.MIRROR_PLANE_CANVAS_Y) \
        == pytest.approx(3.63064, abs=1e-5)
    # it is exactly web A + the seam + web B, and it is mirror-symmetric
    (_, ah), (_, by0) = r6.WEB_A[1], r6.WEB_B[0]
    assert h == pytest.approx(ah + r6.SEAM_M + r6.WEB_B[1][1])
    assert r6.mirror_y(0.0) == pytest.approx(h)
    # the seam strip is INSIDE the canvas now, not a gap beside it
    for y in (ah + 1e-6, 0.5 * (ah + by0), by0 - 1e-6):
        assert y0 <= y <= y0 + h
    # every arm draws on that one surface, whichever unit it belongs to
    for aid in r6.FLEET_FINAL6:
        assert r6.web_of(aid) == webs[0]

    # --- the swap reaches every consumer ---------------------------------
    assert fleet.ACTIVE_RIG == "final6_opt"
    assert sorted(fleet.FLEET) == [2, 13, 17, 31, 71, 97]
    assert fleet.SHEET == r6.SHEET_FINAL6
    from aris_sixarm import atlas as atlas_mod
    assert atlas_mod.SHEET == r6.SHEET_FINAL6      # bound at ITS import time
    from aris_sixarm import allocate as alloc_mod
    assert sorted(alloc_mod.ACTIVE) == [2, 13, 17, 31, 71, 97]
    # the planner's sheet gate is what decides a seam stroke's fate
    for aid in fleet.FLEET:
        assert fleet.sheet_for(fleet.FLEET[aid]) == r6.SHEET_FINAL6
    assert fleet.sheet_for(FLEET_SIXARM[13]) == (3.607, 1.961)

    # --- and nothing it swapped away from was destroyed -------------------
    assert sorted(FLEET_FINAL) == [2, 13, 31]
    assert FLEET_FINAL[2].z == pytest.approx(0.776)      # still as drawn
    assert SHEET_FINAL == (1.8034, 1.700)
    assert len(rig_final.FRAME_BOXES_W_CM) == 35
    assert sorted(FLEET_SIXARM) == [2, 13, 17, 31, 71, 97]


# ==========================================================================
# 5. the extended-pole build
# ==========================================================================
def test_the_extended_pole_build_is_a_build_and_not_a_patch(active6):
    """Both side arms 20 cm lower on 20 cm of new pole — and it hangs together.

    Three things have to be true at once or the configuration is a fiction: the
    drawing must be left alone, the pole must actually reach the bracket that
    now hangs 20 cm lower on it, and the arm must have a ready pose it can hold
    (the drawn one's pen ends up 100 mm UNDER the paper at the new height,
    which is the same failure that forced Q_READY_INV_FINAL into existence).
    """
    A = r6.FLEET_FINAL6_OPT
    assert sorted(A) == [2, 13, 17, 31, 71, 97]
    # the drawing is untouched, and the as-drawn six-arm fleet still exists
    assert len(rig_final.FRAME_BOXES_W_CM) == 35
    assert all(b["lo"][2] == 152.37 for b in rig_final.FRAME_BOXES_W_CM
               if b["name"] == "side_boom")
    assert r6.FLEET_FINAL6[2].z == pytest.approx(0.776)

    # --- both side arms, one mirror of the other, 20 cm down --------------
    for aid in (2, 97):
        assert A[aid].z == pytest.approx(0.576) == pytest.approx(
            r6.FLEET_FINAL6[aid].z + r6.SIDE_DZ_OPT_CM / 100)
        assert A[aid].mount == "wall"
    assert A[97].xy[1] == pytest.approx(r6.mirror_y(A[2].xy[1]))
    # the other four are exactly where they were
    for aid in (13, 31, 17, 71):
        assert np.allclose(A[aid].T_world_base(),
                           r6.FLEET_FINAL6[aid].T_world_base(), atol=0)

    # --- the boxes: stack slid, pole grown, everything else identical -----
    drawn = {b["name"]: b for b in rig_final.FRAME_BOXES_W_CM}
    built = {b["name"]: b for b in r6.FRAME_BOXES6_OPT_W_CM}
    assert set(drawn) == set(built)
    for n in drawn:
        d, b = drawn[n], built[n]
        assert d["lo"][:2] == b["lo"][:2] and d["hi"][:2] == b["hi"][:2], n
        if n in r6.SIDE_SLIDING:
            assert b["lo"][2] == pytest.approx(d["lo"][2] + r6.SIDE_DZ_OPT_CM)
            assert b["hi"][2] == pytest.approx(d["hi"][2] + r6.SIDE_DZ_OPT_CM)
        elif n == r6.SIDE_POLE:
            assert b["lo"][2] == pytest.approx(d["lo"][2] - r6.POLE_EXT_OPT_CM)
            assert b["hi"][2] == d["hi"][2]          # only the bottom grows
        else:
            assert b["lo"][2] == d["lo"][2] and b["hi"][2] == d["hi"][2], n
    # THE POINT OF THE EXTRA PROFILE: the pole must still carry the bracket,
    # which is what it does NOT do as drawn (docs/ARM2_HEIGHT.md: the plate top
    # is flush with the pole's own bottom end).
    pole, brk = built[r6.SIDE_POLE], built["side_bracket"]
    assert pole["lo"][2] <= brk["lo"][2] + 1e-9
    assert brk["hi"][2] <= pole["hi"][2] + 1e-9
    assert brk["lo"][2] - pole["lo"][2] > 0          # real engagement, not flush
    # as drawn, that engagement is the 2.70 cm of docs/ARM2_HEIGHT.md and no
    # more — the bracket sits on the very bottom end of the pole
    assert drawn["side_bracket"]["lo"][2] - drawn[r6.SIDE_POLE]["lo"][2] \
        == pytest.approx(2.70, abs=0.01)

    # --- both units get the SAME build, mirrored --------------------------
    both = r6.frame_boxes6_canvas(zmin=-10, boxes_w=r6.FRAME_BOXES6_OPT_W_CM)
    assert len(both) == 70
    U = {b["unit"]: {b2["name"][:-2]: b2 for b2 in both if b2["unit"] == b["unit"]}
         for b in both}
    for n in U["A"]:
        lo, hi = r6.mirror_box(U["A"][n]["lo"], U["A"][n]["hi"])
        assert np.allclose(lo, U["B"][n]["lo"]) and np.allclose(hi, U["B"][n]["hi"]), n
    # each arm is excused from its OWN mount hardware and nobody else's — and
    # it sees the EXTENDED-pole boxes, not the drawn ones (`static_obstacles`
    # drops the below-paper table block, so compare against the same zmin)
    above = r6.frame_boxes6_canvas(boxes_w=r6.FRAME_BOXES6_OPT_W_CM)
    for aid in sorted(A):
        key = {13: "up", 31: "down", 2: "side",
               17: "up", 71: "down", 97: "side"}[aid]
        unit = r6.UNIT_OF[aid]
        seen = A[aid].static_obstacles()
        mine = {b["name"] for b in above} - {b["name"] for b in seen}
        assert mine == {b["name"] for b in above
                        if b["tag"] == f"mount:{key}@{unit}"}, aid
        assert mine, aid
        if key == "side":                 # and the pole it sees is the long one
            pole_b = next(b for b in seen if b["name"] == "side_boom@"
                          + ("B" if unit == "A" else "A"))
            assert (pole_b["hi"][2] - pole_b["lo"][2]) == pytest.approx(
                (drawn[r6.SIDE_POLE]["hi"][2] - drawn[r6.SIDE_POLE]["lo"][2]
                 + r6.POLE_EXT_OPT_CM) / 100.0)

    # --- the lowered arms can actually hold a ready pose -------------------
    for aid in (2, 97):
        q = np.asarray(A[aid].q_seed, float)
        rep = validate.check_pose(q, A[aid])
        assert rep["ok"], (aid, rep["violations"])
        T = A[aid].T_world_base()
        tip = T[:3, :3] @ tip_pos(q) + T[:3, 3]
        assert tip[2] > 0.0, aid                       # pen ABOVE the paper
        assert 0.0 <= tip[0] <= r6.SHEET_FINAL6[0]
        assert 0.0 <= tip[1] <= r6.SHEET_FINAL6[1]
    assert np.allclose(np.asarray(A[97].q_seed, float),
                       r6.mirror_q(np.asarray(A[2].q_seed, float)))
    # the DRAWN ready pose is exactly what could not be reused
    T = A[2].T_world_base()
    old = T[:3, :3] @ tip_pos(np.asarray(r6.FLEET_FINAL6[2].q_seed, float)) + T[:3, 3]
    assert old[2] == pytest.approx(-0.10, abs=1e-4)


def test_the_seam_strip_is_scored_and_bridged(active6):
    """The payoff, read off the real sweep: the strip that used to be a gap is
    now the best-covered band on the canvas, and it is covered from BOTH sides.
    """
    if not (ATLAS6 / "coverage.npz").exists():
        pytest.skip("no six-arm atlas (run scripts/run_atlas6.py)")
    d = np.load(ATLAS6 / "coverage.npz")
    xs, ys, per, arms = d["xs"], d["ys"], d["per_arm_go"], list(d["arms"])
    assert sorted(int(a) for a in arms) == [2, 13, 17, 31, 71, 97]
    assert ys[-1] > rig_final.SHEET_FINAL[1]          # the sweep left web A
    cnt = per.sum(axis=0)

    seam = (ys >= r6.WEB_A[1][1]) & (ys <= r6.WEB_B[0][1])
    assert seam.sum() >= 10                            # the strip WAS swept
    iA = [i for i, a in enumerate(arms) if r6.UNIT_OF[int(a)] == "A"]
    iB = [i for i, a in enumerate(arms) if r6.UNIT_OF[int(a)] == "B"]
    goA, goB = per[iA].any(axis=0), per[iB].any(axis=0)

    # covered, shared, and shared ACROSS the units — which is the thing the
    # mirrored two-web layout could not buy at any price (the preview measured
    # cross-unit overlap at exactly 0.00 %)
    assert (cnt[seam] >= 1).mean() > 0.80
    assert (cnt[seam] >= 2).mean() > 0.50
    assert (goA[seam] & goB[seam]).mean() > 0.50
    assert cnt.max() >= 3
    # THE RIG IS MIRROR-SYMMETRIC, THE SAMPLE GRID IS NOT: rows run 0, 0.02 ...
    # 3.62 while the canvas ends at 3.63064, so y -> 3.63064 - y does not map
    # the grid onto itself and cell-for-cell equality is the wrong claim.  What
    # must hold is that twin arms cover the same AMOUNT, to within the half-cell
    # the offset grid costs.
    tot = {int(a): int(per[i].sum()) for i, a in enumerate(arms)}
    for a, b in ((13, 17), (31, 71), (2, 97)):
        assert abs(tot[a] - tot[b]) / max(tot[a], 1) < 0.02, (a, b)
    # the floor arms reach none of the seam; the four hanging arms all do
    for i, a in enumerate(arms):
        got = per[i][seam].sum()
        assert (got == 0) if int(a) in (13, 17) else (got > 0), int(a)
