"""The execution adapter, against a synthetic timeline and against v18.

NOTHING IN THIS FILE TOUCHES A ROBOT, and one test asserts that: every
robot-facing method of the hardware backend must raise.
"""
import json

import numpy as np
import pytest

from aris_sixarm import frames
from aris_sixarm.execute import (Barrier, Fr3BundleBackend, Governor,
                                 JointTrajectory, RecordingBackend,
                                 from_schedule, play)
from aris_sixarm.execute.program import DECIMATED_NOTE

V18_NPZ = "out/csail_schedule_h094_v18.npz"
V18_PROG = "out/csail_program_h094_v18.json"


# ---------------------------------------------------------------------------
# a synthetic conducted schedule, in the npz's own shape
# ---------------------------------------------------------------------------
def _fake_npz(tmp_path, n=(40, 30), fps=24.0, stride=2, pause_frames=6,
              arms=(13, 31), swap_step=0.0):
    """Two phases with a pause, in exactly `csail_schedule.py`'s layout."""
    rng = np.random.default_rng(7)
    q0 = np.array([0.0, -0.6, 0.0, -2.0, 0.0, 1.6, 0.6])
    ph, blocks = [], []
    for k, m in enumerate(n):
        ph += [k] * m
        if k + 1 < len(n):
            ph += [-1] * pause_frames
    ph = np.array(ph, np.int64)
    nF = len(ph)
    d = dict(fps=np.float64(fps), dt=np.float64(1.0 / (fps * stride)),
             stride=np.int64(stride), n_phases=np.int64(len(n)),
             pause_s=np.float64(pause_frames / fps), margin=np.float64(0.08),
             min_clearance=np.float64(0.0819), pause_total=np.float64(1.0),
             arms=np.array(sorted(arms), np.int64),
             drawing_arms=np.array(sorted(arms), np.int64),
             pen_ext=np.array([frames.PEN_EXT_HOLDER] * len(arms)),
             sheet=np.array([1.8034, 3.63064]),
             ink_names=np.array(["grey", "orange"]),
             ink_palette=np.array(["#5c5c5b", "#ca6207"]),
             n_frames=np.int64(nF), duration=np.float64((nF - 1) / fps),
             phase=ph, phase_ink=np.array(["grey", "orange"][:len(n)]))
    starts = [int(np.flatnonzero(ph == k)[0]) for k in range(len(n))]
    d["phase_start_s"] = np.array([s / fps for s in starts], float)
    for a in arms:
        Q = np.zeros((nF, 7))
        cur = q0 + 0.01 * a
        for k, m in enumerate(n):
            i0 = starts[k]
            if k:                                   # the swap step under test
                cur = cur + swap_step * np.eye(7)[3]
            step = 0.004 * rng.standard_normal((m, 7))
            step[0] = 0.0            # phase starts exactly at `cur`
            walk = np.cumsum(step, axis=0)
            Q[i0:i0 + m] = cur + walk
            if k + 1 < len(n):
                Q[i0 + m:starts[k + 1]] = Q[i0 + m - 1]
            cur = Q[i0 + m - 1]
        d[f"q_{a}"] = Q.astype(np.float32)
        seg = np.full(nF, -1, np.int64)
        seg[starts[0] + 5:starts[0] + 15] = 0
        d[f"seg_{a}"] = seg
        d[f"u_{a}"] = np.zeros(nF)
        d[f"segpts_{a}"] = np.zeros((0, 2))
        d[f"segoff_{a}"] = np.array([0], np.int64)
    d.update(ink_t=np.zeros(0), ink_arm=np.zeros(0, np.int64),
             ink_off=np.zeros(1, np.int64), ink_xyz=np.zeros((0, 3)),
             ink_hex=np.array([], dtype="<U7"))
    p = tmp_path / "fake_schedule.npz"
    np.savez_compressed(p, **d)
    return p


# ---------------------------------------------------------------------------
# JointTrajectory
# ---------------------------------------------------------------------------
def test_sample_is_linear_and_clamped():
    tr = JointTrajectory(31, [0.0, 1.0, 2.0], np.array(
        [np.zeros(7), np.ones(7), 2 * np.ones(7)]))
    assert np.allclose(tr.sample(0.5), 0.5)
    assert np.allclose(tr.sample(-5.0), 0.0)         # clamped, not extrapolated
    assert np.allclose(tr.sample(99.0), 2.0)
    assert tr.duration_s == 2.0


def test_resample_keeps_the_endpoints_and_the_chords():
    t = np.linspace(0, 1, 5)
    q = np.outer(t, np.ones(7))
    tr = JointTrajectory(13, t, q)
    r = tr.resample(100.0)
    assert np.allclose(r.q[0], tr.q[0]) and np.allclose(r.q[-1], tr.q[-1])
    assert abs(r.duration_s - tr.duration_s) < 1e-12
    # on a straight path, resampling is exact
    assert np.abs(r.sample(0.37) - tr.sample(0.37)).max() < 1e-12


def test_check_catches_the_four_things_it_is_for():
    good = JointTrajectory(31, np.linspace(0, 2, 100),
                           np.tile(frames.Q_READY_FLOOR, (100, 1)))
    assert good.check() == []

    over = JointTrajectory(31, [0.0, 1.0],
                           np.array([frames.Q_READY_FLOOR,
                                     frames.FR3_MAX + 0.5]))
    assert any("over the" in m for m in over.check())

    fast = JointTrajectory(31, [0.0, 0.01],
                           np.array([np.zeros(7), np.full(7, 1.0)]))
    assert any("speed" in m for m in fast.check())

    nonmono = JointTrajectory(31, [0.0, 1.0, 0.5], np.zeros((3, 7)))
    assert any("not strictly increasing" in m for m in nonmono.check())


def test_governor_is_monotone_and_ramps():
    g = Governor(accel=2.0)
    g.resume()
    ts = [g.step(0.05) for _ in range(40)]
    assert all(b >= a for a, b in zip(ts, ts[1:]))       # monotone clock
    assert g.rate == pytest.approx(1.0)
    g.hold()
    d = g.stop_distance_s
    before = g.t_program
    for _ in range(100):
        g.step(0.05)
    assert g.holding
    # the clock advanced by the stop distance and then stopped
    assert g.t_program - before == pytest.approx(d, abs=2e-2)


# ---------------------------------------------------------------------------
# the exporter
# ---------------------------------------------------------------------------
def test_from_schedule_shapes_a_synthetic_run(tmp_path):
    p = from_schedule(_fake_npz(tmp_path), name="fake")
    assert [ph.index for ph in p.phases] == [0, 1]
    assert p.arms == [13, 31]
    assert p.phases[0].ink == "grey" and p.phases[1].ink == "orange"
    assert p.phases[0].start_s == 0.0
    # the pause is a gap in the fleet clock, and it is a barrier
    assert p.phases[1].start_s > p.phases[0].start_s + p.phases[0].duration_s
    kinds = [b.kind for b in p.barriers]
    assert kinds == ["start", "pen_swap", "end"]
    assert p.barriers[1].requires_ack                 # a human swaps the pens
    assert any(DECIMATED_NOTE in w for w in p.warnings)
    assert p.check() == []


def test_a_pose_step_across_the_pause_becomes_a_reposition(tmp_path):
    """The v18 finding, reproduced on a synthetic run and on the real one."""
    (tmp_path / "a").mkdir()
    clean = from_schedule(_fake_npz(tmp_path / "a", swap_step=0.0))
    assert clean.barriers[1].reposition(tol=1e-9) == {}

    (tmp_path / "b").mkdir()
    stepped = from_schedule(_fake_npz(tmp_path / "b", swap_step=0.05))
    rep = stepped.barriers[1].reposition(tol=1e-9)
    assert set(rep) == {13, 31}
    assert np.abs(rep[13] - stepped.barriers[1].hold_q[13]).max() == \
        pytest.approx(0.05, abs=1e-6)
    assert any("REPOSITIONED" in b.note for b in stepped.barriers)
    assert any("UNCERTIFIED" in w for w in stepped.warnings)


def test_track_stitches_the_phases_without_an_infinite_velocity(tmp_path):
    p = from_schedule(_fake_npz(tmp_path, swap_step=0.05))
    tr = p.track(31)
    assert tr.duration_s == pytest.approx(p.duration_s)
    # the gap fill must be a frame wide, not an epsilon: an epsilon-wide step
    # between two different poses would read as ~1e6 rad/s here
    assert tr.joint_speed().max() < frames.QD_MAX.max()


def test_solo_names_the_frozen_set_and_refuses_to_claim_a_certificate(tmp_path):
    p = from_schedule(_fake_npz(tmp_path))
    s = p.solo(31)
    assert s.arm_id == 31
    assert set(s.others_hold) == {13}
    assert s.recheck_required
    assert any("NOT CERTIFIED" in line for line in s.report())


def test_summary_round_trips_through_json(tmp_path):
    p = from_schedule(_fake_npz(tmp_path))
    d = json.loads(json.dumps(p.summary()))
    assert d["arms"] == [13, 31]
    assert len(d["barriers"]) == len(p.barriers)


# ---------------------------------------------------------------------------
# the runner
# ---------------------------------------------------------------------------
def test_play_reaches_the_end_and_stops_at_every_barrier(tmp_path):
    p = from_schedule(_fake_npz(tmp_path))
    be = RecordingBackend(initial={a: p.phases[0].tracks[a].q[0]
                                   for a in p.arms})
    log = play(p, be, realtime=False, confirm=lambda b: True,
               skip_preflight=True)
    assert log.ok, log.report()
    assert log.barriers_passed == [b.name for b in p.barriers]
    assert log.t_program_s == pytest.approx(p.duration_s)
    # every send carried the WHOLE fleet at ONE program time
    assert all(set(q) == set(p.arms) for _, q in be.sent)
    assert all(b >= a for (a, _), (b, _) in zip(be.sent, be.sent[1:]))


def test_an_unacknowledged_pen_swap_stops_the_run(tmp_path):
    p = from_schedule(_fake_npz(tmp_path))
    be = RecordingBackend(initial={a: p.phases[0].tracks[a].q[0]
                                   for a in p.arms})
    log = play(p, be, realtime=False, skip_preflight=True)   # default refuses
    assert not log.ok
    assert "not acknowledged" in log.reason
    assert be.stopped is not None


def test_an_arm_in_the_wrong_place_is_driven_there_once_and_only_once(tmp_path):
    p = from_schedule(_fake_npz(tmp_path))
    be = RecordingBackend(initial={a: np.zeros(7) for a in p.arms})
    log = play(p, be, realtime=False, confirm=lambda b: True,
               skip_preflight=True)
    assert log.ok
    # one supervised goto per arm to reach the start, and nothing else: this
    # programme has no reposition barrier, so the start is the only free move
    assert not log.repositioned
    assert sorted(a for a, _, _ in be.gotos) == p.arms


def test_a_barrier_mismatch_refuses_to_resume(tmp_path):
    p = from_schedule(_fake_npz(tmp_path))
    b = p.barriers[1]
    wrong = {a: q + 0.5 for a, q in b.hold_q.items()}
    assert b.mismatch(wrong)
    assert b.mismatch(b.hold_q) == []
    assert b.mismatch({}) and "did not report" in b.mismatch({})[0]


def test_repositioning_is_a_goto_and_is_logged_as_uncertified(tmp_path):
    p = from_schedule(_fake_npz(tmp_path, swap_step=0.05))
    be = RecordingBackend(initial={a: p.phases[0].tracks[a].q[0]
                                   for a in p.arms})
    log = play(p, be, realtime=False, confirm=lambda b: True,
               skip_preflight=True)
    assert log.ok
    assert len(log.repositioned) == 1
    name, per_arm = log.repositioned[0]
    assert name == "swap-of-phase-0"
    assert all(v == pytest.approx(0.05, abs=1e-6) for v in per_arm.values())
    assert any("uncertified" in line for line in log.report())


# ---------------------------------------------------------------------------
# the hardware backend must not work
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("call", [
    lambda b: b.connect([13]),
    lambda b: b.read_state(),
    lambda b: b.goto(13, np.zeros(7), 1.0),
    lambda b: b.start_stream([13], 1000.0),
    lambda b: b.send(0.0, {13: np.zeros(7)}),
    lambda b: b.stop(""),
])
def test_every_robot_facing_call_raises(call):
    with pytest.raises(NotImplementedError, match="STUB"):
        call(Fr3BundleBackend())


def test_the_hardware_preflight_refuses_a_decimated_programme(tmp_path):
    p = from_schedule(_fake_npz(tmp_path, stride=2))
    bad = Fr3BundleBackend().preflight(p)
    assert any("decimated" in m for m in bad)

    # and the arm that has never had an IP is named rather than assumed
    (tmp_path / "c").mkdir()
    p71 = from_schedule(_fake_npz(tmp_path / "c", arms=(13, 71)))
    assert any("no IP configured" in m and "71" in m
               for m in Fr3BundleBackend().preflight(p71))


def test_the_bundle_export_is_the_chords_exactly(tmp_path):
    t = np.linspace(0, 1, 11)
    q = np.outer(t, np.arange(7) * 0.1)
    tr = JointTrajectory(31, t, q, source="test")
    d = Fr3BundleBackend.export(tr, tmp_path / "b.npz")
    assert d["format"] == "fr3_bundle" and d["version"] == 1
    assert d["control_points"].shape == (10, 2, 7)
    # degree 1: the control points ARE the samples, so the curve is the chords
    assert np.allclose(d["control_points"][:, 0], q[:-1])
    assert np.allclose(d["control_points"][:, 1], q[1:])
    assert np.allclose(d["t_start"], t[:-1]) and np.allclose(d["t_end"], t[1:])
    # and it says so
    assert "NOT flyable" in json.loads(d["meta_json"])["warning"]
    # a speed scale is a reparametrisation: the control points do not move
    slow = Fr3BundleBackend.export(tr, None, speed_scale=0.5)
    assert np.allclose(slow["control_points"], d["control_points"])
    assert np.allclose(slow["t_end"], d["t_end"] * 2)


# ---------------------------------------------------------------------------
# the real thing
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not __import__("pathlib").Path(V18_NPZ).exists(),
                    reason="v18 schedule not in out/")
def test_v18_loads_and_says_what_is_wrong_with_it():
    p = from_schedule(V18_NPZ, V18_PROG)
    assert len(p.phases) == 3 and len(p.arms) == 6
    assert [b.kind for b in p.barriers] == ["start", "pen_swap", "pen_swap",
                                            "end"]
    assert p.duration_s == pytest.approx(244.1667, abs=1e-3)
    assert p.min_clearance_m == pytest.approx(0.0819, abs=1e-3)

    # the two findings this module exists to surface, pinned as numbers
    assert any(DECIMATED_NOTE in w for w in p.warnings)
    reps = {b.name: b.reposition(tol=1e-9) for b in p.barriers}
    assert set(reps["swap-of-phase-0"]) == {31}
    assert set(reps["swap-of-phase-1"]) == {2}

    # velocity is comfortable; ACCELERATION IS NOT, and that is the open item
    for a in p.arms:
        tr = p.track(a)
        assert (tr.joint_speed() / frames.QD_MAX).max() < 0.5
        assert tr.joint_accel().max() > 10.0        # the fr3drivers gate
    assert not any("speed" in m for m in p.check())
    assert any("accel SCREEN" in m for m in p.check())


@pytest.mark.skipif(not __import__("pathlib").Path(V18_NPZ).exists(),
                    reason="v18 schedule not in out/")
def test_v18_plays_end_to_end_into_a_recording_backend():
    p = from_schedule(V18_NPZ, V18_PROG)
    be = RecordingBackend(initial={a: p.phases[0].tracks[a].q[0]
                                   for a in p.arms})
    log = play(p, be, realtime=False, confirm=lambda b: True,
               skip_preflight=True)
    assert log.ok, log.report()
    assert log.t_program_s == pytest.approx(244.1667, abs=1e-3)
    assert len(log.repositioned) == 2
