"""Regression tests distilled from the fuzz campaign (scripts/fuzz_planner.py).

Every test here is a case the campaign either broke on or had to be taught to
survive.  They are deterministic and fast (< 3 min all in) so the whole file can
run after any change to the planning stack; the campaign is what finds new ones,
this is what stops the old ones coming back.

Run: python3 tests/test_planner_robustness.py   (or pytest tests/)
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from aris_sixarm import ik, letters, planner, pwl  # noqa: E402
from aris_sixarm.fleet import FLEET_SIXARM as FLEET, SHEET_SIXARM  # noqa: E402
from aris_sixarm.frames import QD_MAX, fk  # noqa: E402
from aris_sixarm.stroke_api import (DEFAULTS, plan_stroke,  # noqa: E402
                                    polyline_length, truncate_polyline)
from aris_sixarm.validate import validate_plan  # noqa: E402

FLOOR = FLEET[13]        # floor mount
INV = FLEET[31]          # inverted mount
_CACHE = {}


# --------------------------------------------------------------------------
# the strokes the tests share (built once)
# --------------------------------------------------------------------------
def floor_stroke():
    """A comfortable 0.30 m line for the floor arm, well inside its annulus."""
    bx, by = FLOOR.xy
    return np.array([[bx + 0.55, by - 0.15], [bx + 0.55, by + 0.15]])


def inv_stroke():
    """The rim arc from demo_pwl.py, clipped to the sheet (1.56 m)."""
    bx, by = INV.xy
    th = np.linspace(-0.6 * np.pi, 1.05 * np.pi, 400)
    return planner.clip_to_sheet(
        np.column_stack([bx + 0.66 * np.cos(th), by + 0.66 * np.sin(th)]),
        verbose=False, sheet=SHEET_SIXARM)


def under_base_line():
    """demo_stroke.py's stroke B: straight through the arm-31 dead zone."""
    bx, by = INV.xy
    return np.array([[bx - 0.55, by - 0.35], [bx + 0.55, by + 0.35]])


def cached(key, fn):
    if key not in _CACHE:
        _CACHE[key] = fn()
    return _CACHE[key]


def plan_floor():
    return cached("floor", lambda: plan_stroke(floor_stroke(), FLOOR))


def plan_inv():
    return cached("inv", lambda: plan_stroke(inv_stroke(), INV))


def plan_split():
    return cached("split", lambda: plan_stroke(under_base_line(), INV))


# --------------------------------------------------------------------------
# 1. degenerate inputs — never a crash, never a plan
# --------------------------------------------------------------------------
def test_degenerate_inputs():
    """Empty, single-point, all-duplicate, zero-length and sub-millimetre
    strokes are `degenerate`.  Each of these makes `planner.resample` divide by
    a zero arc length and hand back NaN parameters if it is called at all."""
    bx, by = FLOOR.xy
    cases = {
        "empty": np.zeros((0, 2)),
        "one_point": np.array([[bx + 0.5, by]]),
        "duplicates": np.array([[bx + 0.5, by]] * 6),
        "zero_length": np.array([[bx + 0.5, by], [bx + 0.5, by]]),
        "sub_ds": np.array([[bx + 0.5, by], [bx + 0.5, by + 2e-3]]),
        "bad_shape": np.zeros((4, 3)),
    }
    for name, poly in cases.items():
        r = plan_stroke(poly, FLOOR)
        assert r["status"] == "degenerate", f"{name}: got {r['status']} " \
                                            f"{r.get('reason')} {r.get('error', '')}"


def test_multi_stroke_input_is_rejected_cleanly():
    """letters.place() returns a LIST of polylines; the single-stroke API says
    so instead of asarray-ing it into a (2, N, 2) block."""
    r = plan_stroke(letters.place("R", (1.30, 1.05)), INV)
    assert r["status"] == "degenerate" and r["reason"] == "not_a_single_stroke"
    for poly in letters.place("R", (1.30, 1.05)):     # each one on its own is fine
        assert plan_stroke(poly, INV)["status"] in ("ok", "split")


def test_non_finite_points_are_dropped():
    """A NaN/inf vertex is dropped, and what is left is planned normally."""
    bx, by = FLOOR.xy
    poly = np.array([[bx + 0.55, by - 0.15], [np.nan, by], [bx + 0.55, by],
                     [np.inf, 3.0], [bx + 0.55, by + 0.15]])
    r = plan_stroke(poly, FLOOR)
    assert r["status"] == "ok", r.get("reason")
    assert r["validation"]["ok"]


def test_off_sheet_and_gigantic_are_bounded():
    """Nothing off the paper is planned, and a kilometre-long input never gets
    resampled at 10 mm (that would be 10^5 lattice steps)."""
    t0 = time.time()
    assert plan_stroke(np.array([[-2.0, -2.0], [-1.0, -2.0]]), FLOOR)["status"] \
        == "degenerate"
    r = plan_stroke(np.array([[0.5, 0.5], [1e5, 1e5]]), FLOOR)
    assert r["status"] in ("degenerate", "split"), r["status"]
    assert time.time() - t0 < 20.0, "unbounded resampling of a huge stroke"


# --------------------------------------------------------------------------
# 2. one certified stroke per mount type
# --------------------------------------------------------------------------
def _assert_certified(r, spec):
    assert r["status"] == "ok", f"{r['status']}: {r.get('reason')}"
    v = r["validation"]
    assert v["ok"], v["violations"][:3]
    # re-validate independently of the dict the planner handed back
    v2 = validate_plan(r["pts"], spec, r["qs"], times=r["times"])
    assert v2["ok"], v2["violations"][:3]
    assert v2["worst"]["tip_err"] < 2e-3
    assert v2["worst"]["min_margin"] >= 0.15 - 1e-6
    assert v2["worst"]["min_sigma"] >= 0.10 - 1e-6
    assert v2["worst"]["max_step"] <= 0.35 + 1e-6
    assert v2["worst"]["max_qd_frac"] <= 1.0 + 1e-6


def test_floor_mount_stroke_is_certified():
    _assert_certified(plan_floor(), FLOOR)


def test_inverted_mount_stroke_is_certified():
    r = plan_inv()
    _assert_certified(r, INV)
    assert r["arc_len"] > 1.5 and r["n_knots"] <= 4     # the PWL stays compact


def test_plan_covers_the_whole_stroke():
    """The dense samples land on BOTH ends of the stroke that was asked for.

    `planner.resample` steps by a fixed ds and stops at the last whole step, so
    the lattice used to cover 0.160 m of a 0.168 m stroke while the 5 mm
    back-out covered 0.165 m — the tail was planned on faith, at exactly the
    place where the reach boundary usually is.
    """
    for r, poly in ((plan_floor(), floor_stroke()), (plan_inv(), inv_stroke())):
        assert r["coverage_gap"] < 1e-6
        assert np.linalg.norm(r["pts"][0] - poly[0]) < 1e-6
        assert np.linalg.norm(r["pts"][-1] - poly[-1]) < 1e-6
        span = polyline_length(r["pts"])
        assert abs(span - r["arc_len"]) < 1e-3 * r["arc_len"] + 1e-6


# --------------------------------------------------------------------------
# 3. splits
# --------------------------------------------------------------------------
def test_known_split_point():
    """The under-base line splits where the fiber empties (golden s*)."""
    r = plan_split()
    assert r["status"] == "split" and r["reason"] == "empty_fiber"
    assert abs(r["s_star"] - 0.385) < 0.03, r["s_star"]
    assert r["s_reach"] >= r["s_star"]
    h = r["head"]
    assert h is not None and h["status"] == "ok"
    _assert_certified(h, INV)


def test_split_head_replans_from_scratch():
    """[0, s* - eps] planned fresh comes back certified: the contract's promise
    that everything below s_star really is plannable, not just believed."""
    r = plan_split()
    eps = 0.01 / r["arc_len"]
    sub = truncate_polyline(np.asarray(r["stroke"], float), 0.0, r["s_star"] - eps)
    hr = plan_stroke(sub, INV)
    assert hr["status"] == "ok", f"{hr['status']} {hr.get('reason')}"
    _assert_certified(hr, INV)


def test_stroke_starting_in_a_dead_zone():
    """s* = 0: nothing is certified, no head, and the caller is told where the
    arm could pick the stroke up again."""
    bx, by = INV.xy
    poly = np.array([[bx, by], [bx + 0.60, by]])       # starts under the base
    r = plan_stroke(poly, INV)
    assert r["status"] == "split" and r["reason"] == "start_infeasible"
    assert r["s_star"] == 0.0 and r["head"] is None
    assert 0.0 < r.get("s_resume", 0.0) < 1.0


# --------------------------------------------------------------------------
# 4. determinism
# --------------------------------------------------------------------------
def test_determinism_bitwise():
    """Same input, same bytes out — twice for a plan and once for a split."""
    a = plan_stroke(floor_stroke(), FLOOR)
    b = plan_stroke(floor_stroke(), FLOOR)
    for k in ("knots", "qs", "times", "sigmas"):
        assert np.array_equal(a[k], b[k]), f"{k} differs between runs"
    s1, s2 = plan_split(), plan_stroke(under_base_line(), INV)
    assert s1["s_star"] == s2["s_star"]
    assert np.array_equal(s1["head"]["qs"], s2["head"]["qs"])


# --------------------------------------------------------------------------
# 5. the IK filter and the validator itself
# --------------------------------------------------------------------------
def test_ik_rejects_boundary_clamped_solutions():
    """The analytic solver clamps at the workspace boundary: at this pose it
    returns q2 = 0.0 exactly, inside every joint limit and NaN-free, for a
    target it misses by 1.9 cm.  `ik.solve` must not pass that on."""
    Twb_inv = np.linalg.inv(FLOOR.T_world_base())
    T = pwl.pen_down_poses(np.array([[0.21198008, 1.20778365]]), Twb_inv, 0.110)[0]
    q7 = 1.3467
    raw = ik._IK.solve_ik(T.flatten(order="F"), q7, FLOOR.q_seed)
    miss = [q for q in raw if np.all(np.isfinite(q)) and not ik.reaches(T, q)]
    assert miss, "the solver no longer clamps here — retire this test"
    Tf, _ = fk(np.asarray(miss[0], float))
    assert np.linalg.norm(Tf[:3, 3] - T[:3, 3]) > 1e-3
    assert ik.solve(T, q7, FLOOR.q_seed) == [], "clamped solution leaked through"


def test_validator_catches_a_broken_plan():
    """The validator is only worth what it rejects: bend one sample off the
    curve, blow the continuity budget, and hand it an impossible clock."""
    r = plan_floor()
    qs = np.array(r["qs"], float)
    assert validate_plan(r["pts"], FLOOR, qs, times=r["times"])["ok"]

    bad = qs.copy()
    bad[len(bad) // 2, 1] += 0.02                     # 1 cm off the paper
    rep = validate_plan(r["pts"], FLOOR, bad, times=r["times"])
    kinds = {v["kind"] for v in rep["violations"]}
    assert not rep["ok"] and "tip_off_curve" in kinds and "continuity" not in kinds

    bad = qs.copy()
    bad[len(bad) // 2:, 0] += 0.5                     # a 0.5 rad jump in q1
    kinds = {v["kind"] for v in validate_plan(r["pts"], FLOOR, bad)["violations"]}
    assert "continuity" in kinds and "tip_off_curve" in kinds

    rep = validate_plan(r["pts"], FLOOR, qs, times=np.arange(len(qs)) * 1e-4)
    assert not rep["ok"] and "velocity" in {v["kind"] for v in rep["violations"]}
    rep = validate_plan(r["pts"], FLOOR, qs, times=np.zeros(len(qs)))
    assert "time_monotonic" in {v["kind"] for v in rep["violations"]}


def test_validator_never_raises():
    """Garbage in, a violation record out — a validator that throws loses the
    campaign's evidence exactly when it matters."""
    r = plan_floor()
    for pts, qs, t in ((r["pts"], np.zeros((3, 2)), None),
                       (np.zeros((0, 2)), r["qs"], None),
                       (r["pts"], np.full_like(r["qs"], np.nan), None),
                       (r["pts"], r["qs"], np.zeros(3)),
                       ("nonsense", r["qs"], None)):
        rep = validate_plan(pts, FLOOR, qs, times=t)
        assert rep["ok"] is False and isinstance(rep["violations"], list)


def test_velocity_limits_hold_at_the_planned_clock():
    """The clock is not decoration: recomputed from the raw samples, every
    finite-difference joint velocity is inside the FR3 URDF limits."""
    for r in (plan_floor(), plan_inv()):
        qs, t = np.asarray(r["qs"]), np.asarray(r["times"])
        assert np.all(np.diff(t) > 0)
        qd = np.abs(np.diff(qs, axis=0)) / np.diff(t)[:, None]
        assert np.all(qd <= QD_MAX[None, :] + 1e-9), (qd / QD_MAX).max()


def test_opts_are_not_mutated():
    """A caller's dict survives a call (the recursion passes copies around)."""
    opts = {"ds_dense": 0.005}
    plan_stroke(under_base_line(), INV, opts)
    assert opts == {"ds_dense": 0.005}
    assert DEFAULTS["ds_dense"] == 0.005


# --------------------------------------------------------------------------
# 7. the batched kinematics — the fast path must BE the old path
# --------------------------------------------------------------------------
def _small_lattice_args(spec, stroke, ds=0.01, n_q7=48):
    pts, _ = planner.resample(stroke, ds)
    return (pts, spec, None, 0.110, n_q7, True)


def test_batched_lattice_matches_the_scalar_path():
    """The whole point of the rewrite: `_build_lattice_batch` and
    `_build_lattice_scalar` must produce the SAME lattice, not a similar one.

    The valid mask and the joint values have to agree exactly — they come from
    the same C++ solver and the same gates, so anything but zero difference is
    a bug in the vectorisation.  sigma is allowed 1e-8 because the fast path
    uses the analytic tip Jacobian where the scalar path finite-differences it;
    the measured gap is ~2e-11, i.e. the finite differences' own truncation
    error (see test_analytic_tip_jacobian_matches_fd).
    """
    for name, spec, stroke in (("floor line", FLOOR, floor_stroke()),
                               ("inv rim", INV, inv_stroke()),
                               ("under base", INV, under_base_line())):
        args = _small_lattice_args(spec, stroke)
        new = planner._build_lattice_batch(*args)
        old = planner._build_lattice_scalar(*args)
        v = old["valid"]
        assert np.array_equal(new["valid"], v), f"{name}: valid mask differs"
        assert v.any(), f"{name}: nothing valid — the test proves nothing"
        assert np.max(np.abs(new["Q"][v] - old["Q"][v])) < 1e-10, name
        assert np.array_equal(new["margin"], old["margin"]), name
        assert np.max(np.abs(new["sigma"][v] - old["sigma"][v])) < 1e-8, name
        # invalid nodes keep their sentinels, so nothing downstream can read a
        # stale value out of a gap in the band
        assert np.all(np.isnan(new["Q"][~v])) and np.all(new["sigma"][~v] == -1)


def test_scalar_fallback_when_the_extension_has_no_batch():
    """The station venv runs a cp310 wheel with only the scalar entry points.
    With the batch functions taken away, the planner must still plan — the same
    plan — and every wrapper must fall back rather than raise."""
    q = np.array([plan_floor()["qs"][0], plan_floor()["qs"][-1]])

    class _NoBatch:                     # a stand-in for the older extension
        solve_ik = staticmethod(ik._IK.solve_ik)
        solve_ik_cc = staticmethod(ik._IK.solve_ik_cc)

    assert ik.has_batch(), "this build has no batch entry points to hide"
    real = ik._IK
    try:
        ik._IK = _NoBatch()
        assert not ik.has_batch()
        T_fb, P_fb = ik.fk_batch(q)
        J_fb = ik.tip_jacobian_batch(q, pen_ext=0.110)
        lat_fb = planner.build_lattice(*_small_lattice_args(FLOOR,
                                                            floor_stroke())[:2],
                                       pen_ext=0.110)
        r_fb = plan_stroke(floor_stroke(), FLOOR)
    finally:
        ik._IK = real
    assert ik.has_batch()

    # the fallback path took the scalar route and got the scalar answers
    T, P = ik.fk_batch(q)
    assert np.array_equal(T, T_fb) and np.array_equal(P, P_fb)
    assert np.max(np.abs(ik.tip_jacobian_batch(q, pen_ext=0.110) - J_fb)) < 1e-8
    lat = planner.build_lattice(*_small_lattice_args(FLOOR, floor_stroke())[:2],
                                pen_ext=0.110)
    assert np.array_equal(lat["valid"], lat_fb["valid"])
    v = lat["valid"]
    assert np.max(np.abs(lat["Q"][v] - lat_fb["Q"][v])) < 1e-10
    assert r_fb["status"] == "ok"
    assert np.max(np.abs(r_fb["qs"] - plan_floor()["qs"])) < 1e-10


def test_batched_fk_is_bit_identical_to_frames_fk():
    """`fk_many` is a C++ transcription of `frames.fk`.  Bit-identical, not
    close: the lattice gates chain points on a 2 cm threshold, and a plan must
    not change because a clearance check was vectorised."""
    from aris_sixarm.frames import FR3_MAX, FR3_MIN, fk_many, tip_pos, tip_pos_many
    Q = np.random.default_rng(11).uniform(FR3_MIN, FR3_MAX, size=(200, 7))
    T, P = fk_many(Q)
    for i, q in enumerate(Q):
        Tr, Pr = fk(q)
        assert np.array_equal(T[i], Tr) and np.array_equal(P[i], Pr), i
    assert np.array_equal(tip_pos_many(Q, 0.110),
                          np.array([tip_pos(q, 0.110) for q in Q]))


def test_analytic_tip_jacobian_matches_fd():
    """`metrics.tip_jacobian` (central differences) stays the reference; the
    analytic z_i x (p_tip - p_i) Jacobian the planner actually uses has to
    agree with it to well inside the gates it feeds."""
    from aris_sixarm.frames import FR3_MAX, FR3_MIN
    from aris_sixarm.metrics import (sigma_min, sigma_min_many, tip_jacobian,
                                     tip_jacobian_many)
    Q = np.random.default_rng(12).uniform(FR3_MIN, FR3_MAX, size=(100, 7))
    Jfd = np.array([tip_jacobian(q, pen_ext=0.110) for q in Q])
    Jan = tip_jacobian_many(Q, pen_ext=0.110)
    assert np.max(np.abs(Jan - Jfd)) < 1e-8
    s_fd = np.array([sigma_min(J) for J in Jfd])
    assert np.max(np.abs(sigma_min_many(Jan) - s_fd)) < 1e-8
    assert np.array_equal(sigma_min_many(Jfd), s_fd)   # same SVD, same numbers


def test_solve_batch_agrees_with_solve():
    """`ik.solve_batch` is `ik.solve` for a whole array, compacted the same
    way — the lattice indexes branches by slot, so the order is contractual."""
    pts, _ = planner.resample(inv_stroke(), 0.02)
    Twb_inv = np.linalg.inv(INV.T_world_base())
    poses = pwl.pen_down_poses(pts, Twb_inv, 0.110)
    q7 = np.full(len(poses), 0.4)
    Q, valid = ik.solve_batch(poses, q7, INV.q_seed)
    n_sol = 0
    for i, T in enumerate(poses):
        ref = ik.solve(T, q7[i], INV.q_seed)
        assert int(valid[i].sum()) == len(ref), i
        for k, q in enumerate(ref):
            assert np.array_equal(Q[i, k], q), (i, k)
            n_sol += 1
        assert np.all(np.isnan(Q[i, len(ref):]))
    assert n_sol > 50, "the sample found almost no solutions — widen it"


if __name__ == "__main__":
    t0 = time.time()
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            t1 = time.time()
            try:
                fn()
                print(f"{name} PASS ({time.time() - t1:.1f} s)")
            except AssertionError as e:
                fails += 1
                print(f"{name} FAIL: {e}")
    print(f"\n{'ALL PASS' if not fails else f'{fails} FAILURES'} "
          f"in {time.time() - t0:.1f} s")
    sys.exit(1 if fails else 0)
