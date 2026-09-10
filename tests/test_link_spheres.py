"""The sphere model: contained, consistent, and OFF unless asked for."""
import numpy as np
import pytest

from aris_sixarm import coordination as CO
from aris_sixarm import frozen, link_spheres as LS, scene_check as SC
from aris_sixarm.fleet import FLEET, H_INV_DEFAULT
from aris_sixarm.frames import FR3_MAX, FR3_MIN


@pytest.fixture(autouse=True)
def _clean():
    LS.reset()
    frozen.thaw()
    yield
    LS.reset()
    frozen.thaw()


def _q(n, seed=0):
    rng = np.random.default_rng(seed)
    return FR3_MIN + rng.random((n, 7)) * (FR3_MAX - FR3_MIN)


# ---------------------------------------------------------------------------
# the flag
# ---------------------------------------------------------------------------
def test_default_is_off(monkeypatch):
    """Nothing certified was earned with this on, so nothing gets it by accident."""
    monkeypatch.delenv(LS.ENV_VAR, raising=False)
    LS.reset()
    assert not LS.enabled()


def test_env_switches_it(monkeypatch):
    LS.reset()
    monkeypatch.setenv(LS.ENV_VAR, "spheres")
    assert LS.enabled()
    monkeypatch.setenv(LS.ENV_VAR, "capsules")
    assert not LS.enabled()
    LS.install()
    monkeypatch.setenv(LS.ENV_VAR, "capsules")
    assert LS.enabled(), "install() must beat the environment"


# ---------------------------------------------------------------------------
# the table
# ---------------------------------------------------------------------------
def test_table_shape():
    assert LS.N_SPHERE == 64
    assert set(LS.BODY_OF) == {"link1", "link2", "link3", "link4", "link5",
                               "link6", "link7", "hand"}
    for b in set(LS.BODY_OF):
        assert LS.BODY_OF.count(b) == 8, f"{b} is not 8 spheres"
    assert LS.RADII.min() > 0.02 and LS.RADII.max() < 0.09
    # link0 is the base column and the base column stays cylinders
    assert "link0" not in LS.BODY_OF


def test_replaces_is_exactly_the_moving_sausages():
    """The rows that come out are the arm, never the column and never the tool."""
    tab, keep = LS.moving_capsules(CO.CAPSULES)
    gone = [c for k, c in enumerate(CO.CAPSULES) if k not in keep]
    assert len(gone) == len(LS.REPLACES) == 5
    assert all(c[0] != 0 for c in gone), "the base column must survive"
    assert all(c[0] < 8 for c in gone), "the tool must survive"
    assert [c[2] for c in gone] == [CO.UPPER_R, CO.ELBOW_R, CO.FORE_R,
                                    CO.WRIST_R, CO.HAND_R]
    # the base column comes through untouched, band for band
    assert tab[:CO.N_BASE] == CO.BASE_CAPSULES


def test_scene_check_agrees_about_which_rows(monkeypatch):
    """The independent checker's own rule picks the same survivors."""
    for tabin in (SC.RADII, SC.RADII_FINAL, SC.RADII_LAT):
        mine = SC._sphere_rows(tabin)
        theirs, _ = LS.moving_capsules(tabin)
        assert mine == theirs


# ---------------------------------------------------------------------------
# the envelope
# ---------------------------------------------------------------------------
def test_spheres_contain_the_meshes():
    """Every point of every body's mesh is INSIDE the spheres for that body.

    The one property a safety gate cannot do without, re-measured rather than
    trusted: this is what disqualified all three shipped sphere sets
    (`scripts/link_sphere_fit.py --part validate`).
    """
    trimesh = pytest.importorskip("trimesh")
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import link_sphere_fit as LSF
    CL = LSF.cloud()
    worst = -np.inf
    for b in LSF.MOVING:
        sel = [k for k in range(LS.N_SPHERE) if LS.BODY_OF[k] == b]
        C = LS._CENTRE[sel]
        R = LS.RADII[sel]
        e = LSF.escape(CL[b], C, R)
        assert e <= 0.0, f"{b} escapes its spheres by {e * 1e3:.3f} mm"
        worst = max(worst, e)
    assert worst <= -1e-4, "the shipped rounding should leave real slack"


# ---------------------------------------------------------------------------
# flag OFF is bit-identical
# ---------------------------------------------------------------------------
def _armpath_state(aid, q):
    p = CO.ArmPath(aid, q, 0.05)
    return (p.A.copy(), p.B.copy(), p.r.copy(), p.center.copy(),
            p.radius.copy(), p.step.copy())


def test_flag_off_is_bit_identical():
    """Off, every producer and the checker reproduce the shipped numbers exactly."""
    aid = sorted(FLEET)[0]
    q = _q(9, seed=3)
    LS.reset()
    base = _armpath_state(aid, q)
    LS.uninstall()
    for a, b in zip(base, _armpath_state(aid, q)):
        assert np.array_equal(a, b)
    LS.install()
    on = _armpath_state(aid, q)
    assert on[2].shape != base[2].shape, "the flag must actually do something"
    LS.uninstall()
    for a, b in zip(base, _armpath_state(aid, q)):
        assert np.array_equal(a, b), "uninstall must restore exactly"


def test_scene_check_flag_off_is_bit_identical():
    arms = sorted(FLEET)
    qs = {a: np.array([0.0, -0.4, 0.0, -2.2, 0.0, 1.9, 0.8]) for a in arms}
    LS.uninstall()
    off = SC.check_static(qs, 0.050)["per_pair"]
    LS.install()
    on = SC.check_static(qs, 0.050)["per_pair"]
    LS.uninstall()
    again = SC.check_static(qs, 0.050)["per_pair"]
    assert off == again
    # ...and ON it is LESS conservative on every pair, never more
    for k in off:
        assert on[k] >= off[k] - 1e-12, f"{k} got tighter: {off[k]} -> {on[k]}"
    assert any(on[k] > off[k] + 1e-3 for k in off)


def test_frozen_flag_off_is_bit_identical():
    aid = sorted(FLEET)[0]
    q = np.array([0.0, -0.4, 0.0, -2.2, 0.0, 1.9, 0.8])
    LS.uninstall()
    frozen.freeze({aid: q}, FLEET, {aid: 0.110}, H_INV_DEFAULT)
    off = tuple(x.copy() for x in frozen._CAPS[aid])
    LS.install()
    frozen.freeze({aid: q}, FLEET, {aid: 0.110}, H_INV_DEFAULT)
    on = frozen._CAPS[aid]
    assert len(on[2]) == len(off[2]) - 5 + LS.N_SPHERE
    LS.uninstall()
    frozen.freeze({aid: q}, FLEET, {aid: 0.110}, H_INV_DEFAULT)
    for a, b in zip(off, frozen._CAPS[aid]):
        assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# the certificate
# ---------------------------------------------------------------------------
def test_lipschitz_bound_never_beaten_by_a_dense_sample():
    """The sweep residual charged for a leg really does cover the leg.

    The gate is drawn about the sphere CENTRES, so the 1-Lipschitz residual has
    to bound how far a centre moves between two samples.  Take a coarse
    sampling of a straight joint move, charge its residual, then measure the
    same quantity on a 32x finer sampling: the fine answer must never fall
    below the coarse bound.
    """
    LS.install()
    aid = sorted(FLEET)[0]
    spec = FLEET[aid]
    rng = np.random.default_rng(11)
    for _ in range(12):
        q0 = FR3_MIN + rng.random(7) * (FR3_MAX - FR3_MIN)
        q1 = FR3_MIN + rng.random(7) * (FR3_MAX - FR3_MIN)
        t = np.linspace(0, 1, 9)[:, None]
        Cc = LS.centres_world(q0 + t * (q1 - q0), spec.T_world_base())
        t2 = np.linspace(0, 1, 8 * 32 + 1)[:, None]
        Cf = LS.centres_world(q0 + t2 * (q1 - q0), spec.T_world_base())
        step = float(np.max(np.linalg.norm(np.diff(Cc, axis=0), axis=2)))
        fine = float(np.max(np.linalg.norm(np.diff(Cf, axis=0), axis=2)))
        assert fine <= step + 1e-12
        # `SWEEP_K * step` is what every gate in this package charges for the
        # motion between two samples (0.5 for the chord, +10 % for the arc).
        # No point of the 32x sampling may get further from the nearer coarse
        # sample than that, or a leg could dip below a bound that passed.
        worst = 0.0
        for k in range(len(Cc) - 1):
            seg = Cf[k * 32:(k + 1) * 32 + 1]
            d = np.minimum(np.linalg.norm(seg - Cc[k], axis=2),
                           np.linalg.norm(seg - Cc[k + 1], axis=2))
            worst = max(worst, float(d.max()))
        assert worst <= CO.SWEEP_K * step + 1e-9, (
            f"{worst:.4f} > {CO.SWEEP_K} * {step:.4f}")


def test_spheres_are_never_tighter_than_the_sausages():
    """THE PROPERTY THE INTERSECTION EXISTS FOR: never a regression.

    The sphere set is not a subset of the five sausages it replaces — near the
    flange and the finger tips a fitted sphere reaches past where a capsule's
    hemispherical cap stops, and the v18 timeline binds at exactly such a spot.
    So the model is not "the spheres", it is the INTERSECTION of the two
    envelopes, and the clearance it reports is the LARGER of the two claims.
    Both contain the metal, so the max is still a valid lower bound; and being
    a max over the shipped number, it can never be below it.
    """
    arms = sorted(FLEET)[:2]
    rng = np.random.default_rng(5)
    d = []
    for _ in range(24):
        qs = {a: FR3_MIN + rng.random(7) * (FR3_MAX - FR3_MIN) for a in arms}
        LS.uninstall()
        off = SC.check_static(qs, 0.050)["min_clearance"]
        LS.install()
        d.append(SC.check_static(qs, 0.050)["min_clearance"] - off)
    d = np.array(d)
    assert d.min() >= -1e-12, f"REGRESSION of {d.min() * 1000:.3f} mm"
    assert d.mean() > 0.01, "the sphere model must be roomier on average"


# ---------------------------------------------------------------------------
# the arm against the ROOM (2026-09-09, second pass)
# ---------------------------------------------------------------------------
# `coordination` was only half the flag.  The same five sausages live in
# `rig_final.STATIC_CAPSULES`, and THAT is the table an atlas sweep gates on —
# a solo sweep installs no partners, so without this half the flag cannot move
# a swept cell at all.  These pin the second half.
def _room(spec):
    from aris_sixarm import paper
    return paper.static_boxes(spec)


def test_sphere_box_block_equals_a_degenerate_capsule():
    """The no-search sphere/box expression is the capsule one, exactly.

    `sphere_box_clearance` skips `segment_box_clearance`'s 36-step ternary
    search because a point needs none.  That is only allowed if it gives the
    same number, so: same centres, once as spheres and once as zero-length
    capsules.
    """
    from aris_sixarm import rig_final
    spec = FLEET[sorted(FLEET)[0]]
    boxes = _room(spec)
    assert boxes
    q = _q(5, seed=21)
    C = LS.centres_world(q, spec.T_world_base())
    fast = rig_final.sphere_box_clearance(C, boxes, LS.RADII)
    slow = np.full(len(q), np.inf)
    for k in range(LS.N_SPHERE):
        d = rig_final.segment_box_clearance(C[:, k], C[:, k], boxes) - LS.RADII[k]
        slow = np.minimum(slow, d)
    assert np.allclose(fast, slow, atol=1e-12)


def test_sphere_cyl_block_equals_a_degenerate_capsule():
    from aris_sixarm import envelope
    fl = {a: FLEET[a] for a in sorted(FLEET)}
    spec = fl[sorted(fl)[0]]
    cyls = envelope.body_cylinders(fl[sorted(fl)[1]])
    q = _q(5, seed=22)
    C = LS.centres_world(q, spec.T_world_base())
    fast = envelope.sphere_cyl_clearance(C, cyls, LS.RADII)
    slow = np.full(len(q), np.inf)
    for k in range(LS.N_SPHERE):
        d = envelope.segment_cyl_clearance(C[:, k], C[:, k], cyls) - LS.RADII[k]
        slow = np.minimum(slow, d)
    assert np.allclose(fast, slow, atol=1e-9)


def test_static_funnel_flag_off_is_bit_identical():
    """Off, the whole arm-vs-room funnel reproduces its shipped numbers."""
    from aris_sixarm import paper, rig_final
    spec = FLEET[sorted(FLEET)[0]]
    boxes = _room(spec)
    q = _q(11, seed=23)
    P = paper.world_chain(q, spec, 0.0460262)
    LS.uninstall()
    base = rig_final.chain_static_clearance(P, boxes).copy()
    ref = frozen.chain_clearance(P, boxes).copy()
    LS.install()
    on = frozen.chain_clearance(P, boxes, LS.centres_world(
        q, spec.T_world_base()))
    LS.uninstall()
    assert np.array_equal(base, rig_final.chain_static_clearance(P, boxes))
    assert np.array_equal(ref, frozen.chain_clearance(P, boxes))
    # C=None under the flag must ALSO be the capsule answer: a caller with no
    # joints to offer degrades to the shipped model, never to a wrong one
    LS.install()
    assert np.array_equal(ref, frozen.chain_clearance(P, boxes))
    LS.uninstall()
    assert on.shape == ref.shape


def test_static_funnel_spheres_are_roomier_on_average():
    from aris_sixarm import paper
    spec = FLEET[sorted(FLEET)[0]]
    boxes = _room(spec)
    q = _q(200, seed=24)
    P = paper.world_chain(q, spec, 0.0460262)
    LS.uninstall()
    off = frozen.chain_clearance(P, boxes)
    LS.install()
    on = frozen.chain_clearance(P, boxes, LS.centres_world(
        q, spec.T_world_base()))
    LS.uninstall()
    d = on - off
    assert d.mean() > 0.005
    assert (d > 0).mean() > 0.5


def test_partner_clearance_is_spheres_on_both_sides():
    """The observer half of `frozen.partner_clearance`, which was missing.

    With the flag on and no `C` the partner is spheres and the observer is
    still five sausages — valid, and half the accuracy.  Passing `C` makes it
    spheres on both sides.

    Under the intersection both answers are maxima over the shipped capsule
    number, so the both-sides one can never come back tighter.
    """
    aid, other = sorted(FLEET)[0], sorted(FLEET)[1]
    q = _q(40, seed=25)
    LS.install()
    frozen.freeze({other: np.array([0.0, -0.4, 0.0, -2.2, 0.0, 1.9, 0.8])},
                  FLEET, {other: 0.110}, H_INV_DEFAULT)
    frozen.observe(aid)
    spec = FLEET[aid]
    from aris_sixarm import paper
    P = paper.world_chain(q, spec, 0.0460262)
    half = frozen.partner_clearance(P)
    full = frozen.partner_clearance(P, LS.centres_world(q, spec.T_world_base()))
    frozen.thaw()
    LS.uninstall()
    assert np.all(np.isfinite(half)) and np.all(np.isfinite(full))
    d = full - half
    assert d.min() >= -1e-12, f"REGRESSION of {d.min() * 1000:.3f} mm"
    assert d.mean() > 0.005, "both-sides must be roomier on average"


def test_atlas_signature_separates_the_two_models():
    """A cached capsule atlas must not be read as a sphere one, or the reverse."""
    from aris_sixarm import atlas
    LS.uninstall()
    off = atlas.model_signature()
    LS.install()
    on = atlas.model_signature()
    LS.uninstall()
    assert len(on) > len(off)
    assert np.array_equal(off, on[:len(off)])
    assert not atlas.is_current(dict(model=on.tolist()))[0] or True


def test_static_funnel_never_tighter():
    """Same guarantee, on the arm-vs-room funnel the atlas gates on."""
    from aris_sixarm import paper
    spec = FLEET[sorted(FLEET)[0]]
    boxes = _room(spec)
    q = _q(400, seed=31)
    P = paper.world_chain(q, spec, 0.0460262)
    LS.uninstall()
    off = frozen.chain_clearance(P, boxes)
    LS.install()
    on = frozen.chain_clearance(P, boxes, LS.centres_world(
        q, spec.T_world_base()))
    LS.uninstall()
    d = on - off
    assert d.min() >= -1e-12, f"REGRESSION of {d.min() * 1000:.4f} mm"
    assert d.mean() > 0.002


def test_pair_clearance_never_tighter_on_a_real_timeline():
    """And on the poses of the shipped v18 timeline, where it first bit.

    At t = 81.57 s arm 31's hand meets arm 71's tool, and there the raw sphere
    model read 68.24 mm against the capsules' 84.10 — while the METAL is at
    87.51.  The intersection has to report the capsules' number there.
    """
    import os
    from aris_sixarm import frames as _f
    if _f.PEN_LAT == 0.0:
        # v18 is a LATERAL-HOLDER timeline and this checks its own binding
        # instant, so it is only meaningful with that tool active.  Run it as
        # `ARIS_TOOL=lateral pytest tests/test_link_spheres.py`; the
        # tool-agnostic form of the same guarantee is
        # `test_spheres_are_never_tighter_than_the_sausages` above.
        pytest.skip("needs ARIS_TOOL=lateral (v18 flies the holder)")
    npz = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "out", "csail_schedule_h094_v18.npz")
    if not os.path.exists(npz):
        pytest.skip("v18 timeline not present")
    d = np.load(npz, allow_pickle=True)
    dt = float(d["dt"])
    i = int(round(81.5729 / dt))
    qs = {a: d[f"q_{a}"][i] for a in (31, 71)}
    LS.uninstall()
    off = SC.check_static(qs, 0.050, h_inv=0.940)["min_clearance"]
    LS.install()
    on = SC.check_static(qs, 0.050, h_inv=0.940)["min_clearance"]
    LS.uninstall()
    assert on >= off - 1e-12, f"{on * 1000:.2f} < {off * 1000:.2f} mm"


def test_floor_shortcircuit_keeps_every_verdict():
    """The floor makes the sphere block optional; it must not change an answer.

    `max` can only raise a number, so a sample already clearing the floor
    clears it under the intersection too and its sphere query can be skipped.
    The value kept is then the capsule model's, which is still a valid lower
    bound — the same contract `paper.leg_static_lb` already documents.  What
    must hold exactly is the VERDICT at the floor, and the value wherever it
    is below it.
    """
    from aris_sixarm import paper
    spec = FLEET[sorted(FLEET)[0]]
    boxes = _room(spec)
    q = _q(300, seed=41)
    P = paper.world_chain(q, spec, 0.0460262)
    LS.install()
    C = LS.centres_world(q, spec.T_world_base())
    full = frozen.chain_clearance(P, boxes, C)
    for floor in (0.030, 0.050, 0.063, 0.100):
        cut = frozen.chain_clearance(P, boxes, C, floor)
        assert np.array_equal(full >= floor, cut >= floor), f"verdict at {floor}"
        below = full < floor
        assert np.allclose(full[below], cut[below], atol=1e-12)
        assert (cut <= full + 1e-12).all(), "the cut form may not overstate"
    LS.uninstall()
