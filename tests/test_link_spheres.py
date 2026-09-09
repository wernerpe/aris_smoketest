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


def test_spheres_are_mostly_but_not_always_roomier():
    """The gain is an AVERAGE, and the exceptions are real and small.

    The sphere set is NOT a subset of the five sausages it replaces: a sausage
    is a finite segment with hemispherical caps, and near the flange and the
    finger tips a fitted sphere reaches a little past where the cap stops.
    That is the old model being THIN there, not this one being wrong — the
    2026-08-26 mesh audit had to inflate those same capsules because they were
    optimistic by up to 78 mm — and it is why the claim this test makes is
    about the distribution rather than about every pose.
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
    assert d.mean() > 0.01, "the sphere model must be roomier on average"
    assert (d > 0).mean() >= 0.6
    assert d.min() > -0.02, "and never much tighter than the sausages"
