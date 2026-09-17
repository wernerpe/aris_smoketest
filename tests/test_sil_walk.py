"""The setpoint sources and the paper geometry — both Drake-free.

What is pinned: the walker's rules (`PathwayWalker` docstring) and which sheet
a rig mount and a synthetic mount get by default.
"""
from pathlib import Path

import numpy as np
import pytest

from aris_sixarm.sil.params import ArmMount, Paper, Walker
from aris_sixarm.sil.setpoints import PathwayWalker, read_csv_v2, runs_of

HEAD = ("stroke_idx,wp_idx,kind,x_m,y_m,z_m,qx,qy,qz,qw,intensity,"
        "q1,q2,q3,q4,q5,q6,q7")
W = Walker()


def write_csv(path: Path, rows) -> Path:
    """rows: (stroke, wp, kind, x, y, z, q_or_None)."""
    lines = [HEAD]
    for s, wp, kind, x, y, z, q in rows:
        joints = ",".join(f"{v:.6f}" for v in q) if q is not None else "," * 6
        lines.append(f"{s},{wp},{kind},{x:.6f},{y:.6f},{z:.6f},"
                     f"1.0,0.0,0.0,0.0,1.0,{joints}")
    path.write_text("\n".join(lines) + "\n")
    return path


Q_A = np.full(7, 0.1)
Q_B = np.full(7, 0.2)


def phases(walker) -> list:
    """-> [(phase, n_ticks)] runs, in order."""
    out = []
    for sp in walker:
        if out and out[-1][0] == sp.phase:
            out[-1][1] += 1
        else:
            out.append([sp.phase, 1])
    return [tuple(x) for x in out]


# --- paper ---------------------------------------------------------------
def test_rig_paper_is_the_canvas_with_its_corner_at_the_origin():
    paper = Paper.for_rig("proposed")
    assert paper.size == pytest.approx((1.8034, 3.63064))
    assert paper.center_xy == pytest.approx((0.9017, 1.81532))


def test_synthetic_mount_gets_a_big_sheet_centred_on_the_base():
    """A synthetic mount has no canvas: the rig sheet would leave the arm
    drawing over the edge of the world."""
    mount = ArmMount.synthetic("inverted", 0.94, 31)
    paper = Paper.under(mount)
    assert paper.size == (4.0, 4.0)
    assert paper.center_xy == (0.0, 0.0)             # the base is at the origin
    reach = 1.07                                     # FR3 tip reach, lateral
    assert paper.size[0] / 2 > reach and paper.size[1] / 2 > reach
    off = Paper.under(ArmMount.synthetic("floor", 0.0, 13), span=2.0)
    assert off.size == (2.0, 2.0)


# --- the walk ------------------------------------------------------------
def test_runs_of_splits_on_the_draw_flag_only(tmp_path):
    rows = read_csv_v2(write_csv(tmp_path / "r.csv", [
        (0, 0, "lift_start", 0.4, 0.0, 0.0, None),
        (0, 1, "draw", 0.4, 0.0, 0.05, Q_A),
        (0, 2, "draw", 0.4, 0.0, 0.05, Q_A),
        (0, 3, "lift_end", 0.4, 0.0, 0.0, None),
        (1, 0, "travel", 0.5, 0.0, 0.0, Q_B),
        (1, 1, "draw", 0.5, 0.0, 0.05, Q_B),
    ]))
    runs = runs_of(rows)
    assert [(is_draw, len(r)) for is_draw, r in runs] == [
        (False, 1), (True, 2), (False, 2), (True, 1)]


def test_pen_up_rows_are_walked_as_a_cartesian_path_at_travel_speed(tmp_path):
    """A `travel` run is flown as written -- no hover added, no press -- and
    its duration is its own arc length over `travel_speed`."""
    csv = write_csv(tmp_path / "p.csv", [
        (0, 0, "draw", 0.40, 0.0, 0.05, Q_A),
        (0, 1, "draw", 0.42, 0.0, 0.05, Q_A),
        (1, 0, "travel", 0.42, 0.0, 0.00, Q_B),
        (1, 1, "travel", 0.60, 0.0, 0.00, Q_B),      # 180 mm of transit
        (2, 0, "draw", 0.60, 0.0, 0.05, Q_B),
        (2, 1, "draw", 0.62, 0.0, 0.05, Q_B),
    ])
    walker = PathwayWalker(csv, W)
    # stroke 0 needs no connect leg (the pen starts at its own approach point);
    # the lift, the transit run and stroke 2's approach are all `travel` and
    # therefore one contiguous phase.
    assert [p for p, _ in phases(walker)] == [
        "touchdown", "draw", "lift", "travel", "touchdown", "draw", "lift"]

    travel = [sp for sp in walker if sp.phase == "travel"]
    # 80 mm down from the lift (hover 30 above z=0.05) to the transit plane
    # at z=0, 180 mm of transit, 80 mm back up to stroke 2's approach point
    assert len(travel) == pytest.approx(round(0.34 / 0.04 / 1e-3), abs=4)
    assert all(sp.press_cmd_m <= 0.0 for sp in travel)      # never pressed
    on_plane = [sp for sp in travel if abs(sp.p_xyz[2]) < 1e-9]
    # the run's own rows are commanded AS WRITTEN on their z = 0 pen-up plane
    assert len(on_plane) == pytest.approx(round(0.18 / 0.04 / 1e-3), abs=3)
    assert all(sp.press_cmd_m == 0.0 for sp in on_plane)
    assert min(sp.p_xyz[0] for sp in on_plane) == pytest.approx(0.42, abs=1e-6)
    assert max(sp.p_xyz[0] for sp in on_plane) == pytest.approx(0.60, abs=1e-3)


def test_q_ref_is_interpolated_along_a_transit_and_absent_without_columns(
        tmp_path):
    csv = write_csv(tmp_path / "q.csv", [
        (0, 0, "draw", 0.40, 0.0, 0.05, Q_A),
        (0, 1, "draw", 0.42, 0.0, 0.05, Q_A),
        (1, 0, "travel", 0.42, 0.0, 0.00, Q_A),
        (1, 1, "travel", 0.60, 0.0, 0.00, Q_B),
        (2, 0, "draw", 0.60, 0.0, 0.05, Q_B),
        (2, 1, "draw", 0.62, 0.0, 0.05, Q_B),
    ])
    transit = [sp for sp in PathwayWalker(csv, W)
               if sp.phase == "travel" and sp.stroke_idx == 1]
    q = np.array([sp.q_ref for sp in transit])
    assert np.all(np.diff(q[:, 0]) >= -1e-12)          # monotone A -> B
    assert q[0, 0] == pytest.approx(Q_A[0], abs=1e-6)
    assert q[-1, 0] == pytest.approx(Q_B[0], abs=1e-3)

    noq = write_csv(tmp_path / "noq.csv", [
        (0, 0, "draw", 0.40, 0.0, 0.05, Q_A),
        (0, 1, "draw", 0.42, 0.0, 0.05, Q_A),
        (1, 0, "travel", 0.42, 0.0, 0.00, None),       # contract §1: no q
        (1, 1, "travel", 0.60, 0.0, 0.00, None),
        (2, 0, "draw", 0.60, 0.0, 0.05, Q_B),
        (2, 1, "draw", 0.62, 0.0, 0.05, Q_B),
    ])
    walker = PathwayWalker(noq, W)
    assert walker.has_joint_columns                    # draw rows still carry q
    assert all(sp.q_ref is None for sp in walker
               if sp.phase == "travel" and sp.stroke_idx == 1)


def test_draw_rows_are_pressed_and_a_missing_pen_up_run_is_synthesised(
        tmp_path):
    """No pen-up rows at all -> the walker makes the hover itself."""
    csv = write_csv(tmp_path / "d.csv", [
        (0, 0, "draw", 0.40, 0.0, 0.05, Q_A),
        (0, 1, "draw", 0.42, 0.0, 0.05, Q_A),
    ])
    walker = PathwayWalker(csv, W)
    assert [p for p, _ in phases(walker)] == [
        "touchdown", "draw", "lift"]                   # no travel: already there
    draw = [sp for sp in walker if sp.phase == "draw"]
    assert all(sp.press_cmd_m == pytest.approx(W.press_applied_m)
               for sp in draw)
    # the pen axis of quat (1,0,0,0) is -z, so the press goes DOWN from z=0.05
    assert draw[0].p_xyz[2] == pytest.approx(0.05 - W.press_applied_m, abs=1e-9)
    lift = [sp for sp in walker if sp.phase == "lift"]
    assert lift[-1].p_xyz[2] == pytest.approx(0.05 + W.hover_m, abs=1e-4)
    assert W.lift_max_m < W.hover_m                    # the lift clears the press


def test_real_pathway_walks_every_row():
    """The exporter's own arm-31 file, whatever shape it currently has: every
    draw run is landed and lifted, and the draw time is the file's own draw arc
    length over `draw_speed`.  Nothing here is hard-coded to a file version --
    the exporter regenerates it."""
    csv = Path("out/pathways/csail_schedule_h094_v18_arm31.csv")
    if not csv.exists():
        pytest.skip("exporter output not present")
    walker = PathwayWalker(csv, W)
    n_draw_runs = sum(1 for is_draw, _ in runs_of(walker.rows) if is_draw)
    seq = [p for p, _ in phases(walker)]
    assert seq.count("draw") == n_draw_runs
    assert seq.count("touchdown") == n_draw_runs   # the two legs are adjacent
    assert seq.count("lift") == n_draw_runs
    assert walker.has_joint_columns

    length = sum(
        float(np.linalg.norm(np.diff(np.array([r.p for r in rows]), axis=0),
                             axis=1).sum())
        for is_draw, rows in runs_of(walker.rows) if is_draw and len(rows) > 1)
    draw_ticks = sum(n for p, n in phases(walker) if p == "draw")
    assert draw_ticks * 1e-3 == pytest.approx(length / W.draw_speed,
                                              rel=1e-3, abs=0.05)
