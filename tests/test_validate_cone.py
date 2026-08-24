"""The INDEPENDENT validator learns about the pen's orientation.

Until pen tilt existed, every plan the pipeline made commanded the pen
perpendicular to the paper, and nothing downstream ever asked whether it had
stayed there — it did not need to.  `validate.validate_plan`'s tip test cannot
answer the question either, and that is not an oversight: `frames.tip_pos_many`
steps `pen_ext` along the tool z OF THE FK'D POSE, so a pen leaning forty
degrees that still puts its tip on the curve passes it exactly as a
perpendicular one does.  Every other gate in the certificate is about the CHAIN.

So "the plan kept the lean the material allows" was the one invariant of a
tilted plan that nothing re-derived, and these tests are about the gate that
now does.  The load-bearing property is the DEFAULT: `tilt_max_deg = 0` means
every plan written before this existed is checked against the perpendicular
pen it was actually asked for, and only a plan that was granted a cone is
allowed to use one.
"""
import numpy as np
import pytest

from aris_sixarm import fleet as fleet_mod
from aris_sixarm import planner, stroke_api, tilt, validate
from aris_sixarm.frames import PEN_EXT, fk_many, tip_pos_many


@pytest.fixture
def final6():
    prev = fleet_mod.ACTIVE_RIG
    fleet_mod.activate("final6_opt")
    yield fleet_mod.FLEET
    fleet_mod.activate(prev)


def donut_arc(cx=1.2768, cy=1.2572, r=0.13, a0=170.0, a1=240.0, n=70):
    """An arc inside arm 2's comfort donut — the strokes tilt exists for."""
    th = np.linspace(np.deg2rad(a0), np.deg2rad(a1), n)
    return np.column_stack([cx + r * np.cos(th), cy + r * np.sin(th)])


# --------------------------------------------------------------------------
# 1. the refactor that made the check free
# --------------------------------------------------------------------------
def test_tip_matches_tip_pos_many(final6):
    """One FK now serves the tip, the pen axis and the chain points.

    `validate_plan` used to call `fk_many` three times for three readings of
    the same kinematics.  Folding them into one is only allowed if the numbers
    do not move, so the tip expression is `tip_pos_many`'s body verbatim —
    pinned here BIT for bit rather than to a tolerance, because a tolerance
    would hide exactly the kind of drift this is guarding against.
    """
    spec = final6[2]
    r = stroke_api.plan_stroke(donut_arc(), spec, dict(ds_lattice=0.008))
    assert r["status"] == "ok"
    qs = np.asarray(r["qs"], float)
    T, _ = fk_many(qs)
    folded = T[:, :3, 3] + T[:, :3, :3] @ np.array([0.0, 0.0, PEN_EXT])
    assert np.array_equal(folded, tip_pos_many(qs, PEN_EXT))


# --------------------------------------------------------------------------
# 2. the default is the perpendicular pen
# --------------------------------------------------------------------------
def test_flat_plans_pass_a_zero_degree_cone(final6):
    """The whole shipping corpus is perpendicular, and the gate says so.

    `arccos` of the down-component reports ~1e-8 rad of phantom lean for a pen
    that is vertical to the last bit of double precision, which is enough to
    fail its own zero-degree cone; `arctan2` of (sideways, down) is conditioned
    the other way round.  Measured over every certified CSAIL plan the worst
    flat lean is 6.3e-11 deg, so `CONE_EPS_DEG = 1e-6` is five orders of
    magnitude of headroom.
    """
    spec = final6[2]
    r = stroke_api.plan_stroke(donut_arc(), spec, dict(ds_lattice=0.008))
    assert r["status"] == "ok"
    rep = validate.validate_plan(r["pts"], spec, r["qs"], times=r["times"])
    assert rep["ok"]
    assert rep["worst"]["max_lean_deg"] < 1e-6
    assert rep["worst"]["max_lean_deg"] < validate.CONE_EPS_DEG


def test_a_tilted_plan_fails_the_default_gate_and_passes_its_own(final6):
    """The gate has to BITE, or the default proves nothing.

    A tilt-rescued plan handed to the validator with no cone must be refused —
    that is what makes "flat plans are checked against a perpendicular pen" a
    statement with content.  Handed the cone it was granted, the same samples
    pass.
    """
    spec = final6[2]
    poly = donut_arc()
    from aris_sixarm.metrics import GATE_MARGIN, GATE_SIGMA
    r = tilt.plan_adaptive(poly, spec, 15.0, pitch_deg=7.5,
                           opts=dict(ds_lattice=0.008, margin_gate=GATE_MARGIN,
                                     sigma_gate=GATE_SIGMA))
    assert r["status"] == "ok" and r["tilt_used"] is True
    assert r["max_lean_deg"] > 1.0                   # it really did lean

    strict = validate.validate_plan(r["pts"], spec, r["qs"], times=r["times"],
                                    margin_gate=GATE_MARGIN,
                                    sigma_gate=GATE_SIGMA)
    assert not strict["ok"]
    kinds = {v["kind"] for v in strict["violations"]}
    assert kinds == {"pen_cone"}                     # the ONLY thing wrong
    assert strict["worst"]["max_lean_deg"] == pytest.approx(r["max_lean_deg"],
                                                            abs=1e-9)

    granted = validate.validate_plan(r["pts"], spec, r["qs"], times=r["times"],
                                     margin_gate=GATE_MARGIN,
                                     sigma_gate=GATE_SIGMA, tilt_max_deg=15.0)
    assert granted["ok"]


def test_the_lean_is_re_derived_not_read_off_the_plan(final6):
    """`validate_plan` never reads `tilt`, `lean` or `max_lean_deg`.

    Corrupt every one of them and the verdict must not move: the validator FKs
    the joint samples and measures the tool z itself, which is the entire
    reason it is worth running a second time.
    """
    spec = final6[2]
    from aris_sixarm.metrics import GATE_MARGIN, GATE_SIGMA
    r = tilt.plan_adaptive(donut_arc(), spec, 15.0, pitch_deg=7.5,
                           opts=dict(ds_lattice=0.008, margin_gate=GATE_MARGIN,
                                     sigma_gate=GATE_SIGMA))
    assert r["status"] == "ok"
    lie = dict(r, tilt=np.zeros_like(np.asarray(r["tilt"], float)),
               lean=np.zeros(len(r["qs"])), max_lean_deg=0.0)
    rep = validate.validate_plan(lie["pts"], spec, lie["qs"],
                                 times=lie["times"], margin_gate=GATE_MARGIN,
                                 sigma_gate=GATE_SIGMA)
    assert not rep["ok"]
    assert rep["worst"]["max_lean_deg"] > 1.0


def test_cone_agrees_with_tilt_cone_check(final6):
    """Two implementations, one number.

    `tilt.cone_check` is the planner's own reading and `validate._pen_lean_deg`
    the certificate's; they are written independently and must not drift.
    """
    spec = final6[2]
    from aris_sixarm.metrics import GATE_MARGIN, GATE_SIGMA
    r = tilt.plan_adaptive(donut_arc(), spec, 15.0, pitch_deg=7.5,
                           opts=dict(ds_lattice=0.008, margin_gate=GATE_MARGIN,
                                     sigma_gate=GATE_SIGMA))
    assert r["status"] == "ok"
    _, worst, _ = tilt.cone_check(r["qs"], spec, 15.0)
    rep = validate.validate_plan(r["pts"], spec, r["qs"], times=r["times"],
                                 margin_gate=GATE_MARGIN, sigma_gate=GATE_SIGMA,
                                 tilt_max_deg=15.0)
    assert rep["worst"]["max_lean_deg"] == pytest.approx(worst, abs=1e-12)


# --------------------------------------------------------------------------
# 3. the gates reach the shipping entry point
# --------------------------------------------------------------------------
def test_plan_stroke_validates_at_the_gate_it_was_asked_for(final6):
    """A strict-gate plan carries a strict-gate certificate.

    The three places a gate has to arrive are the band DP, the 5 mm chase and
    the independent validator.  If the first two tightened and the third did
    not, an "ok" at 0.30 would be re-derived at 0.15 and the certificate would
    be about a weaker claim than the plan makes.
    """
    spec = final6[2]
    r = stroke_api.plan_stroke(donut_arc(r=0.30), spec,
                               dict(ds_lattice=0.008, margin_gate=0.20))
    plan = r if r["status"] == "ok" else r.get("head")
    if plan is None:
        pytest.skip("this arc does not certify at 0.20 on this rig")
    assert plan["min_margin"] >= 0.20 - 1e-9
    rep = validate.validate_plan(plan["pts"], spec, plan["qs"],
                                 times=plan["times"], margin_gate=0.20)
    assert rep["ok"] and rep["worst"]["min_margin"] >= 0.20 - 1e-6


def test_reversing_a_tilted_plan_keeps_its_certificate(final6):
    """`reverse_plan` used to raise KeyError on a tilted plan.

    It subscripts `windows` without a default, and the tilt path had none — so
    a tilt-rescued segment the sequencer wanted to draw backwards took the
    whole allocation down.  It also named knot column 1 explicitly, silently
    dropping the two tilt columns, and left `tilt`/`lean` in forward order next
    to reversed joints.
    """
    spec = final6[2]
    from aris_sixarm.metrics import GATE_MARGIN, GATE_SIGMA
    r = tilt.plan_adaptive(donut_arc(), spec, 15.0, pitch_deg=7.5,
                           opts=dict(ds_lattice=0.008, margin_gate=GATE_MARGIN,
                                     sigma_gate=GATE_SIGMA))
    assert r["status"] == "ok" and r["tilt_used"] is True
    rev = stroke_api.reverse_plan(r, spec,
                                  opts=dict(margin_gate=GATE_MARGIN,
                                            sigma_gate=GATE_SIGMA))
    assert rev["status"] == "ok" and rev["validation"]["ok"]
    assert np.array_equal(rev["qs"], np.asarray(r["qs"])[::-1])
    assert np.array_equal(rev["tilt"], np.asarray(r["tilt"])[::-1])
    assert np.asarray(rev["knots"]).shape == np.asarray(r["knots"]).shape
    assert rev["max_lean_deg"] == pytest.approx(r["max_lean_deg"])


def test_a_tilted_plan_is_shaped_like_a_flat_one(final6):
    """Same door, same keys.  A tilted plan is handed to `allocate`,
    `sequence`, `writing` and `reverse_plan` exactly as a flat one is."""
    spec = final6[2]
    from aris_sixarm.metrics import GATE_MARGIN, GATE_SIGMA
    flat = stroke_api.plan_stroke(donut_arc(), spec, dict(ds_lattice=0.008))
    tilted = tilt.plan_adaptive(donut_arc(), spec, 15.0, pitch_deg=7.5,
                                opts=dict(ds_lattice=0.008,
                                          margin_gate=GATE_MARGIN,
                                          sigma_gate=GATE_SIGMA))
    assert flat["status"] == tilted["status"] == "ok"
    need = {"windows", "stroke", "clip_s", "sheet", "notes", "knots", "qs",
            "pts", "times", "s", "q7", "sigmas", "margins", "min_sigma",
            "min_margin", "tip_err", "max_step", "sum_travel", "n_knots",
            "n_dense", "arc_len", "total_time", "frac_slowed", "headroom",
            "validation", "coverage_gap"}
    assert need <= set(tilted), f"missing {sorted(need - set(tilted))}"
    # the sharp polyline is the documented state of a tilted plan
    assert np.all(np.asarray(tilted["windows"], float) == 0.0)


def test_the_tilt_chase_lands_on_both_ends_of_the_stroke(final6):
    """`planner.resample` stops at the last WHOLE step, so chasing at a raw
    5 mm left up to 5 mm of the far end unplanned — and every gate here is
    POINTWISE, so nothing could see a tail that was never sampled.  The step
    is fitted to the length now, as `stroke_api.prepare` has always done."""
    spec = final6[2]
    from aris_sixarm.metrics import GATE_MARGIN, GATE_SIGMA
    poly = donut_arc()
    r = tilt.plan_adaptive(poly, spec, 15.0, pitch_deg=7.5,
                           opts=dict(ds_lattice=0.008, margin_gate=GATE_MARGIN,
                                     sigma_gate=GATE_SIGMA))
    assert r["status"] == "ok"
    assert r["coverage_gap"] == pytest.approx(0.0, abs=1e-9)
    # the commanded curve and the resampled one agree at both ends
    stroke = np.asarray(r["stroke"], float)
    pts = np.asarray(r["pts"], float)
    assert np.linalg.norm(pts[0] - stroke[0]) < 1e-9
    assert np.linalg.norm(pts[-1] - stroke[-1]) < 1e-9
