"""End-to-end runs: the question the package exists to answer.

(a) hold in the air     — gravity compensation + the law + damping are sane
(b) the 10 cm line with the pencil the robot believes in
(c) the same line with a pencil 1 cm shorter than the robot believes

Needs the `sil` extra (pydrake).  Total runtime ~15 s.
"""
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pydrake")

from aris_sixarm.sil import (ArmMount, Controller, Paper, PathwayWalker,  # noqa: E402
                             SetpointLog, SilParams, Tool, Trace, Walker,
                             metrics, plot, read_csv_v2, run)
from aris_sixarm.sil.__main__ import main                       # noqa: E402

EXAMPLE = (Path(__file__).resolve().parents[1]
           / "aris_sixarm/sil/examples/line10cm_arm31.csv")
RIG, ARM, TOOL = "proposed", 31, "lateral"


def a_params(tip_error, press_m=0.010):
    return SilParams(
        mount=ArmMount.from_fleet(RIG, ARM),
        tool=Tool.from_frames(TOOL, tip_error),
        paper=Paper.for_rig(RIG),
        controller=Controller.from_yaml(),
        walker=Walker(press_m=press_m))


def test_a_hold_in_air_tracks_within_two_millimetres(tmp_path):
    """2 s at the first draw pose, pen 30 mm above the plane.

    No contact, no press: what is left is gravity compensation, the Cartesian
    spring and its damping.  A steady-state tip error above a couple of
    millimetres would mean one of those three is wrong, and the rig's own
    free-air number for comparison is 5-20 mm of friction lag (briefing §8.4),
    which this sim does not model and therefore must beat.
    """
    row = next(r for r in read_csv_v2(EXAMPLE) if r.is_draw)
    hover = row.p - 0.030 * row.pen_axis
    log = tmp_path / "hold.npz"
    np.savez(log, t_s=np.array([0.0, 2.0]),
             pos=np.vstack([hover, hover]),
             quat=np.vstack([row.quat, row.quat]))

    params = a_params(0.0)
    trace = run(params, SetpointLog(log), label="hold")
    tail = trace.t >= trace.t[-1] - 0.5
    steady = float(trace.track_err_m[tail].max())
    assert steady < 2e-3, f"steady-state tip error {1e3 * steady:.3f} mm"
    assert not trace.in_contact.any()
    # The wrist carries a bounded ~80 Hz limit cycle of a few tenths of a
    # milliradian: the Cartesian damping term is evaluated on the PREVIOUS
    # tick's velocity (one-tick delay, as in the deployed loop) and
    # D_rot * dt / I_wrist ~ 4, which is the classic sampled-damping ringing.
    # It is a property of the 1 kHz law on this arm, not of this simulator --
    # what the real machine has and this model does not is the joint friction
    # that damps it out (see `params.Joints`).  It must stay far below
    # anything the study measures, and 0.1 mm at the tip is.
    swing = float(np.ptp(trace.tip_nom_world[tail], axis=0).max())
    assert swing < 5e-4, f"tip swing {1e3 * swing:.3f} mm"


def test_b_correct_tip_draws_the_line(tmp_path):
    """tip_error 0, press 10 mm: the pen is on the paper the whole stroke.

    K_z 800 x 10 mm of geometric press is ~8 N of spring authority and the
    contact is stiff enough not to give any of it back, so that is what comes
    out.  THE RIG WOULD NOT SHOW 8 N: its measured effective stiffness is
    0.40 x commanded (briefing §8.4), i.e. ~320 N/m along the pen, so the same
    10 mm press reads ~3.2 N there.  The sim realises the COMMANDED stiffness
    and does not reproduce that 0.40 factor -- whatever mechanism costs the
    rig 60 % of its stiffness (joint friction, the wrench estimate, the
    hardware layer) is not in this model.
    """
    params = a_params(0.0)
    trace = run(params, PathwayWalker(EXAMPLE, params.walker), label="tip+0mm")
    m = metrics(trace)
    assert m["touch_frac"] > 0.9, m
    assert 5.0 < m["f_mean_n"] < 12.0, m
    assert m["max_penetration_m"] < 1e-3            # 0.16 mm: the sheet is stiff
    assert m["mean_actual_tip_height_m"] < 0.0      # the tip is IN the paper
    # the briefing's bowing mechanism is reproduced: pen friction against the
    # lateral spring displaces the stroke by millimetres (§7.1, K_xy comment)
    assert 1e-3 < m["xy_rms_m"] < 5e-3, m

    out = tmp_path / "trace.npz"
    trace.save(out)
    again = Trace.load(out)
    assert np.allclose(again.f_normal, trace.f_normal)
    assert metrics(again)["touch_frac"] == pytest.approx(m["touch_frac"])
    assert plot(trace, tmp_path / "plot.png").exists()
    assert trace.save_setpoint_log(tmp_path / "sp.npz").exists()
    assert SetpointLog(tmp_path / "sp.npz").duration_s > 1.0


def test_c_short_tip_metrics_are_finite(tmp_path):
    """tip_error -10 mm, the same press.  No assertion on the OUTCOME -- that
    outcome is the finding, and it is reported, not pinned."""
    params = a_params(-0.010)
    trace = run(params, PathwayWalker(EXAMPLE, params.walker), label="tip-10mm")
    m = metrics(trace)
    for key, value in m.items():
        if key == "first_near_limit_s":
            continue                 # nan is the GOOD outcome: never near one
        assert np.isfinite(value), (key, value)
    assert 0.0 <= m["touch_frac"] <= 1.0
    assert m["max_penetration_m"] >= 0.0
    assert m["near_limit_frac"] == 0.0


def test_cli_compare_writes_everything(tmp_path):
    """The exact invocation the report quotes, on a trimmed stroke."""
    trimmed = tmp_path / "short.csv"
    lines = EXAMPLE.read_text().splitlines()
    trimmed.write_text("\n".join(lines[:1] + lines[1:12] + lines[-1:]) + "\n")
    out = tmp_path / "run"
    assert main(["--csv", str(trimmed), "--rig", RIG, "--arm", str(ARM),
                 "--tool", TOOL, "--compare", "0.0", "-0.010",
                 "--out", str(out), "--no-plot"]) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert set(summary) == {"tip+0mm", "tip-10mm"}
    assert summary["tip+0mm"]["meta"]["k_cartesian"][2] == 800.0
    for tag in summary:
        assert (out / f"trace_{tag}.npz").exists()
        assert (out / f"setpoints_{tag}.npz").exists()
