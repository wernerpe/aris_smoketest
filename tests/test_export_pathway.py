"""Pathway CSV v2 exporter — `aris_sixarm.export.pathway`.

The gates, in the order they matter:

  * the file is EXECUTABLE: every row carries q, and no two consecutive rows
    ask the arm for a joint step it cannot make.  v1 of the exporter dropped
    the planner's pen-up transits and asked for 4.7 rad in one hop; the Drake
    SIL hit joint limits on the way to stroke 4 and never recovered.
  * the DEPLOYED READER parses what we write.  `_load_verbatim` below is a copy
    of the executor's own parser, not a paraphrase of it.
  * the QUATERNION means on paper what the deployed generator means by it.
  * the FK gate of `docs/ARIS2_CONTRACTS.md` §1 actually refuses a bad row.
  * the manifest carries the contract's field set.
  * the shipped programme exports, and lands where the arm can reach.

Runs without env vars and without a robot; ~5 s.
"""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from aris_sixarm import fleet, frames, ik, planner, writing
from aris_sixarm.export import pathway as ep

_ROOT = Path(__file__).resolve().parents[1]
SHIPPED_NPZ = _ROOT / "out" / "csail_schedule_h094_v18.npz"
SHIPPED_JSON = _ROOT / "out" / "csail_program_h094_v18.json"
# `out/` is gitignored, so the shipped programme is a local artefact: present on
# the planning machine, absent in a bare checkout.  The synthetic fixtures below
# need nothing but the package, so only the tests that read it are skipped.
needs_shipped = pytest.mark.skipif(
    not (SHIPPED_NPZ.exists() and SHIPPED_JSON.exists()),
    reason=f"{SHIPPED_NPZ.name} is not in this checkout (out/ is gitignored)")

#: the largest joint step the exporter may put between two consecutive rows.
#: DERIVED, not chosen: the conductor bounds its own sub-step at
#: `writing.MAX_DQ_FRAME` and the npz keeps every `stride`-th one, so a row-to-
#: row step cannot legitimately exceed their product.  On
#: `csail_schedule_h094_v18` the exported files actually reach 0.0658 rad
#: (both arm 31 and arm 71), i.e. 82% of this.
MAX_ROW_DQ = 2 * writing.MAX_DQ_FRAME


# ---------------------------------------------------------------------------
# THE DEPLOYED READER, VERBATIM
# ---------------------------------------------------------------------------
# Copied character for character from `PathwayExec._load`, lines 312-328 of
#   /home/franka/aris_project/worktrees/aris2-rtff/rtff_pathway_exec.py
# at commit 5d1580e25a3544bc56af4d713a946ceef2bde47f — the worktree's HEAD and
# also `git -C /home/franka/aris_project/Aris_Kindt rev-parse origin/diemut-operator-rtff`,
# i.e. the DEPLOYED v1 reader (`git show 5d1580e:rtff_pathway_exec.py`).  That
# is deliberately the v1 one: the contract's promise is that our seven extra
# columns are INVISIBLE to it.  The worktree's working copy has since grown the
# optional q1..q7 parse of contract §2, which is purely additive; it is that
# work item's own tests' business, not this file's.
# Only two things are changed here, and neither touches the parse: `self` is
# dropped, and the trailing `return rim_tilt_waypoints(wps)` becomes
# `return wps` — that helper is a no-op unless RTFF_RIM_TILT=1 is in the
# environment (its own first two lines), and this test does not set it.
def _load_verbatim(path):
    wps = []
    with open(path) as f:
        for row in csv.DictReader(f):
            kind = row["kind"]
            xyz = np.array([float(row["x_m"]), float(row["y_m"]),
                            float(row["z_m"])])
            q = [float(row["qx"]), float(row["qy"]),
                 float(row["qz"]), float(row["qw"])]
            try:
                inten = float(row.get("intensity", "") or 1.0)
            except (TypeError, ValueError):
                inten = 1.0
            wps.append({"kind": kind, "xyz": xyz, "q": q,
                        "draw": kind == "draw",
                        "intensity": min(max(inten, 0.0), 1.0)})
    return wps


def _rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def _joints(rows):
    return np.array([[float(r[f"q{i}"]) for i in range(1, 8)] for r in rows])


# ---------------------------------------------------------------------------
# a synthetic schedule, in the conductor's own npz shape
# ---------------------------------------------------------------------------
def _floor_arm_stroke(rig="sixarm", arm=13, y=0.9, x0=0.35, n=8, dx=0.005,
                     max_step=0.03, far_from=None):
    """A certified joint path for a FLOOR arm drawing a straight line.

    Built by `planner.build_lattice` — the planner's own IK + gates — so the
    pen is vertical and the tip is on the world paper plane, exactly as a real
    schedule's drawing frames are.  ONE (q7, branch) is used for the whole
    stroke, as the planner does, so the chain is smooth: picking a fresh
    solution per point produces a path with 0.37 rad holes in it, which is not
    what a conducted schedule contains.

    `far_from`: among the smooth candidates, take the one whose FIRST pose is
    furthest from this configuration.  That is how a real programme ends up
    reconfiguring between strokes, and the fixture needs it to be able to test
    that the transit rows carry the reconfiguration.
    """
    fl, _ = fleet.rig(rig)
    spec = fl[arm]
    pts = np.array([[x0 + dx * i, y] for i in range(n)])
    lat = planner.build_lattice(pts, spec)
    V = lat["valid"]
    cands = []
    for j, b in np.argwhere(V.all(axis=0)):
        Q = np.asarray(lat["Q"][:, j, b], float)
        step = float(np.abs(np.diff(Q, axis=0)).max())
        if step <= max_step:
            cands.append((Q, step))
    assert cands, "no smooth certified chain covers the fixture line"
    if far_from is None:
        Q = min(cands, key=lambda t: t[1])[0]
    else:
        Q = max(cands,
                key=lambda t: np.abs(t[0][0] - np.asarray(far_from)).max())[0]
    return spec, Q


def _climb(q, height=0.06, n=30):
    """A pen-up climb off a drawing pose, by the repo's own case-consistent IK.

    This is TEST INPUT — it stands in for what the conductor's transit router
    actually produces (measured on the shipped programme: every inter-stroke
    transit lifts the tip 58..60 mm before travelling).  Independent of the
    exporter: `ik.solve_cc` is the repo primitive, called here directly.

    Stepped at 2 mm and CHECKED, because holding q7 fixed while the tip climbs
    is a poor constraint near a shoulder singularity: at some certified drawing
    poses a 6 cm lift costs 1.8 rad however finely it is stepped, and a fixture
    that quietly contained one would not be testing the exporter.
    """
    q = np.asarray(q, float).reshape(7)
    T0, _ = frames.fk(q)
    out, prev = [], q
    for k in range(1, n + 1):
        T = T0.copy()
        T[:3, 3] = T0[:3, 3] - (height * k / n) * T0[:3, 2]
        s = ik.solve_cc(T, float(q[6]), prev)
        assert s is not None, "the fixture climb is not solvable"
        assert np.abs(s - prev).max() < MAX_ROW_DQ, (
            "the fixture climb is not smooth — pick a different stroke")
        out.append(s)
        prev = s
    return np.asarray(out)


def _bridge(qa, qb, max_dq=0.03):
    """Joint-space interpolation between two pen-up poses, finely enough.

    Stands in for the conductor's transit, which is what really fills this gap;
    what matters for the exporter is that the frames exist and step gently.
    """
    n = max(2, int(np.ceil(np.abs(np.asarray(qb) - np.asarray(qa)).max()
                           / max_dq)))
    t = np.linspace(0.0, 1.0, n + 1)[1:-1]
    return np.asarray(qa) + t[:, None] * (np.asarray(qb) - np.asarray(qa))


def _write_npz(path, arm, Q, seg, *, pen_ext=None, fps=24.0,
               sheet=(3.607, 1.961)):
    """A minimal but REAL schedule npz: the keys `from_schedule` reads."""
    pen_ext = frames.PEN_EXT if pen_ext is None else float(pen_ext)
    Q = np.asarray(Q, float)
    seg = np.asarray(seg, int)
    F = len(Q)
    assert len(seg) == F
    np.savez_compressed(
        path, fps=fps, dt=1.0 / (2 * fps), stride=2, n_phases=1,
        pause_s=0.0, margin=0.08, min_clearance=0.1,
        arms=np.array([arm]), drawing_arms=np.array([arm]),
        pen_ext=np.array([pen_ext]), sheet=np.array(sheet, float),
        n_frames=F, duration=F / fps, phase=np.zeros(F, int),
        phase_start_s=np.array([0.0]), phase_ink=np.array(["grey"]),
        **{f"q_{arm}": Q.astype(np.float32), f"seg_{arm}": seg,
           f"u_{arm}": np.linspace(0, 1, F)})
    return path


def _two_stroke(tmp, arm=13):
    """Two strokes with a REAL certified transit between them, and a lift at
    each end — the shape every arm of a conducted schedule has.

    The two strokes sit far enough apart that the boundary joint jump exceeds
    0.5 rad, which is the case that makes transit rows mandatory.
    """
    spec, qa = _floor_arm_stroke(y=0.90, x0=0.45)
    _, qb = _floor_arm_stroke(y=1.10, x0=0.55, far_from=qa[-1])
    assert np.abs(qb[0] - qa[-1]).max() > 0.5, "fixture strokes are too close"
    up_a, up_b = _climb(qa[-1]), _climb(qb[0])
    lead = _climb(qa[0])[::-1]                      # descend onto stroke A
    tail = _climb(qb[-1])                           # retract off stroke B
    hold = np.repeat(up_a[-1:], 4, axis=0)          # a barrier hold: dedupe me
    blocks = [(lead, False), (qa, True),
              (up_a, False), (hold, False),
              (_bridge(up_a[-1], up_b[-1]), False), (up_b[::-1], False),
              (qb, True), (tail, False)]
    Q = np.concatenate([b for b, _ in blocks])
    seg, k = [], 0
    for b, drawing in blocks:
        seg += [k if drawing else -1] * len(b)
        k += 1 if drawing else 0
    _write_npz(tmp / "two.npz", arm, Q, seg)
    return spec, tmp / "two.npz", qa, qb, len(hold)


@pytest.fixture(scope="module")
def floor_export(tmp_path_factory):
    """A written CSV + manifest for a floor arm, base yaw 0, inline pen."""
    d = tmp_path_factory.mktemp("floor")
    spec, npz, qa, qb, n_hold = _two_stroke(d)
    pws = ep.export(npz, rig="sixarm", tool="inline", out=d / "pathways",
                    name="synth", intensity=0.75)
    assert len(pws) == 1
    return dict(dir=d, spec=spec, qa=qa, qb=qb, n_hold=n_hold, pw=pws[0],
                csv=d / "pathways" / "synth_arm13.csv",
                manifest=d / "pathways" / "synth_arm13.manifest.json")


# ---------------------------------------------------------------------------
# 1. round trip through the deployed reader
# ---------------------------------------------------------------------------
def test_round_trip_through_the_executors_own_parser(floor_export):
    pw = floor_export["pw"]
    wps = _load_verbatim(floor_export["csv"])
    rows = _rows(floor_export["csv"])
    assert len(wps) == len(pw.rows) == len(rows)
    assert list(rows[0].keys()) == ep.CSV_COLUMNS

    # the seven extra columns are INVISIBLE to the v1 reader: it builds exactly
    # the v1 waypoint dict, so the executor "behaves exactly as v1" (§1)
    assert set(wps[0]) == {"kind", "xyz", "q", "draw", "intensity"}

    kinds = [w["kind"] for w in wps]
    assert set(kinds) <= {"lift_start", "travel", "draw", "lift_end"}
    assert kinds[0] == "travel" and kinds[-1] == "travel"   # certified ends
    n_draw = sum(1 for w in wps if w["draw"])
    assert n_draw == len(floor_export["qa"]) + len(floor_export["qb"])
    # the two strokes are separated by real travel rows, not by a hop
    runs = [k for k, _ in __import__("itertools").groupby(
        ["draw" if w["draw"] else "up" for w in wps])]
    assert runs == ["up", "draw", "up", "draw", "up"]

    # xyz, quaternion and intensity survive the text
    for w, r in zip(wps, pw.rows):
        assert w["xyz"] == pytest.approx([float(r[3]), float(r[4]),
                                          float(r[5])], abs=0)
        assert w["q"] == pytest.approx([float(r[6]), float(r[7]),
                                        float(r[8]), float(r[9])], abs=0)
        assert np.linalg.norm(w["q"]) == pytest.approx(1.0, abs=1e-9)
    assert {w["intensity"] for w in wps if w["draw"]} == {0.75}
    assert {w["intensity"] for w in wps if not w["draw"]} == {1.0}

    # the draw rows ARE the schedule's frames, in order, undecimated
    T, _ = frames.fk_many(np.concatenate([floor_export["qa"],
                                          floor_export["qb"]]))
    tip = T[:, :3, 3] + T[:, :3, :3] @ frames.tool_offset(frames.PEN_EXT, 0.0)
    got = np.array([w["xyz"] for w in wps if w["draw"]])
    assert got[:, :2] == pytest.approx(tip[:, :2], abs=1e-6)

    # stroke / waypoint indexing is monotone and resets per stroke
    sidx = [int(r["stroke_idx"]) for r in rows]
    assert sidx == sorted(sidx) and set(sidx) == {0, 1}
    for s in (0, 1):
        w = [int(r["wp_idx"]) for r in rows if int(r["stroke_idx"]) == s]
        assert w == list(range(len(w)))


def test_the_barrier_hold_collapses_to_one_row(floor_export):
    """A pen swap is 1177 identical frames on arm 31.  One row says it."""
    Q = _joints(_rows(floor_export["csv"]))
    assert not (np.abs(np.diff(Q, axis=0)).max(axis=1) == 0).any()


# ---------------------------------------------------------------------------
# 2. THE FILE IS EXECUTABLE
# ---------------------------------------------------------------------------
def test_every_row_carries_seven_joints(floor_export):
    rows = _rows(floor_export["csv"])
    for i, r in enumerate(rows):
        cells = [r[f"q{j}"] for j in range(1, 8)]
        assert all(c != "" for c in cells), f"row {i} ({r['kind']}) has gaps"
        q = np.array([float(c) for c in cells])
        assert q.shape == (7,) and np.isfinite(q).all()
        assert (q >= frames.FR3_MIN).all() and (q <= frames.FR3_MAX).all()
    assert floor_export["pw"].manifest["joint_columns"] is True


def test_no_row_to_row_joint_step_the_arm_cannot_make(floor_export):
    Q = _joints(_rows(floor_export["csv"]))
    step = np.abs(np.diff(Q, axis=0)).max()
    assert step < MAX_ROW_DQ, f"{step:.4f} rad between two rows"
    assert floor_export["pw"].stats["max_row_dq_rad"] == pytest.approx(step)


def test_transit_rows_bridge_a_big_stroke_boundary(floor_export):
    """The reconfiguration lives INSIDE the transit, so the transit must be in
    the file: a boundary jump over 0.5 rad with no rows between the strokes is
    the defect the SIL found."""
    m = floor_export["pw"].manifest
    assert m["transits"]["in_file"] is True
    assert m["stats"]["n_travel_rows"] > 0
    assert m["boundaries"], "a two-stroke fixture has one boundary"
    for b in m["boundaries"]:
        if b["boundary_dq_rad"] > 0.5:
            assert b["n_transit_rows"] > 0, b
            assert b["max_row_dq_rad"] < MAX_ROW_DQ
    assert max(b["boundary_dq_rad"] for b in m["boundaries"]) > 0.5


# ---------------------------------------------------------------------------
# 3. the quaternion convention
# ---------------------------------------------------------------------------
def test_floor_arm_pen_down_is_the_deployed_1000(floor_export):
    """`(1,0,0,0)` = pen straight down on a floor arm (contract §1).

    The legacy `sixarm` arm 13 is bolted to the floor with base yaw 0, so its
    base frame IS the world frame up to a translation and every DRAWING row's
    quaternion must be the deployed generator's constant EXACTLY, not merely up
    to a yaw.  (Pen-up rows are free to turn, and the fixture's bridge does.)
    """
    Twb = floor_export["spec"].T_world_base()
    assert np.allclose(Twb[:3, :3], np.eye(3), atol=1e-12)
    draw = [w for w in _load_verbatim(floor_export["csv"]) if w["draw"]]
    assert draw
    for w in draw:
        # 1e-6 on the components, not 0: the residual is the analytic IK's own
        # (~2e-8), i.e. 2.4e-6 deg against the contract's 0.5 deg budget.
        assert w["q"] == pytest.approx([1.0, 0.0, 0.0, 0.0], abs=1e-6)
        assert ep.quat_angle_deg(w["q"], [1.0, 0.0, 0.0, 0.0]) < 1e-4
        # and the pen points DOWN in the base frame
        assert ep.quat_R(w["q"]) @ [0, 0, 1] == pytest.approx([0, 0, -1],
                                                              abs=1e-6)


def test_pen_down_is_1000_up_to_a_yaw_about_the_pen_axis(tmp_path):
    """The same on a floor arm whose base is YAWED (final rig, arm 13).

    The free parameter the deployed file bakes to zero is the rotation about
    the pen axis; the planner's own yaw is whatever its IK chose, and the
    exporter must NOT convert it (that would break the FK gate). So the
    invariant is `R_export = Rx(pi) @ Rz(psi)`, i.e. both frames agree on
    where +Z points.
    """
    spec, q = _floor_arm_stroke(rig="final", arm=13, x0=0.88, y=0.30)
    assert not np.allclose(spec.T_world_base()[:3, :3], np.eye(3))
    lead, tail = _climb(q[0])[::-1], _climb(q[-1])
    Q = np.concatenate([lead, q, tail])
    seg = [-1] * len(lead) + [0] * len(q) + [-1] * len(tail)
    _write_npz(tmp_path / "s.npz", 13, Q, seg)
    pw, = ep.export(tmp_path / "s.npz", rig="final", tool="inline",
                    out=tmp_path / "p", name="yaw")

    R_ref = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])   # (1,0,0,0)
    assert ep.quat_xyzw(R_ref) == pytest.approx([1, 0, 0, 0], abs=1e-12)
    seen_yaw = False
    for w in _load_verbatim(pw.paths["csv"]):
        R = ep.quat_R(w["q"])
        Rz = R_ref.T @ R                      # must be a rotation about z
        assert Rz[:, 2] == pytest.approx([0, 0, 1], abs=1e-6)
        assert Rz[2, :] == pytest.approx([0, 0, 1], abs=1e-6)
        seen_yaw |= abs(np.arctan2(Rz[1, 0], Rz[0, 0])) > 1e-3
    assert seen_yaw, "this fixture is supposed to exercise a NON-zero yaw"


@needs_shipped
def test_inverted_arm_pen_points_plus_z_in_link0(tmp_path):
    """`svg_to_pathway_csv --inverted`: "the pen must point +Z in fr3_link0".

    Not exactly +Z on every DRAWING row, and that is a real difference from the
    deployed file: `aris_sixarm.tilt` leans the pen inside a per-stroke cone to
    rescue strokes at the rim, so the quaternion varies. The invariant is that
    every drawing row still descends toward the overhead table — pen . +Z > 0 —
    and that the lean never exceeds the cone the programme itself recorded.
    """
    pw, = ep.export(SHIPPED_NPZ, SHIPPED_JSON, rig="proposed", tool="lateral",
                    arms=[31], out=tmp_path, name="inv")
    assert fleet.rig("proposed")[0][31].mount == "inv"
    wps = _load_verbatim(pw.paths["csv"])
    axes = np.array([ep.quat_R(w["q"]) @ [0, 0, 1]
                     for w in wps if w["draw"]])
    assert axes[:, 2].min() > 0.0
    lean = np.degrees(np.arccos(np.clip(axes[:, 2], -1, 1)))

    prog = json.loads(SHIPPED_JSON.read_text())
    cone = max(e.get("tilt_cone_deg", 0.0) for ph in prog["phases"]
               for e in ph["arms"]["31"])
    assert cone > 0.0, "this fixture is supposed to contain a LEANING stroke"
    assert lean.max() <= cone + 1e-6
    assert pw.manifest["quaternion"]["max_lean_deg"] == pytest.approx(
        lean.max(), abs=1e-6)
    assert lean.min() == pytest.approx(0.0, abs=1e-3)   # vertical strokes


# ---------------------------------------------------------------------------
# 4. the FK gate
# ---------------------------------------------------------------------------
def test_fk_gate_fires_on_a_corrupted_row(floor_export):
    pw = floor_export["pw"]
    lat, ext = ep.tool_offsets("inline")
    assert ep.verify_rows(pw.rows, lat, ext) == []

    # a millimetre of position error, on a draw row
    bad = [list(r) for r in pw.rows]
    draw = next(i for i, r in enumerate(bad) if r[2] == "draw")
    bad[draw][3] = "%.6f" % (float(bad[draw][3]) + 0.001)
    msgs = ep.verify_rows(bad, lat, ext)
    assert len(msgs) == 1 and "mm from the row's xyz" in msgs[0]

    # a degree of orientation error: spin the row's quaternion about its own z
    bad = [list(r) for r in pw.rows]
    q = np.array([float(bad[draw][6 + i]) for i in range(4)])
    s = np.sin(np.radians(0.5))
    spin = np.array([0.0, 0.0, s, np.cos(np.radians(0.5))])   # Rz(1 deg), xyzw
    x1, y1, z1, w1 = q
    x2, y2, z2, w2 = spin
    bad[draw][6:10] = ["%.9f" % v for v in (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2)]
    msgs = ep.verify_rows(bad, lat, ext)
    assert len(msgs) == 1 and "deg from the row's quaternion" in msgs[0]

    # a half-filled joint column set is a corruption of its own
    bad = [list(r) for r in pw.rows]
    bad[draw][14] = ""
    assert any("partly filled" in m for m in ep.verify_rows(bad, lat, ext))


def test_build_refuses_to_return_a_file_that_fails_the_gate(monkeypatch,
                                                            tmp_path):
    """The gate is INSIDE the exporter, not just available to a caller."""
    spec, q = _floor_arm_stroke()
    _write_npz(tmp_path / "s.npz", 13, q, [0] * len(q))
    real = ep.verify_rows

    def _one_bad(rows, *a, **kw):
        return real(rows, *a, **kw) + ["synthetic failure"]

    monkeypatch.setattr(ep, "verify_rows", _one_bad)
    with pytest.raises(ValueError, match="refusing to write"):
        ep.export(tmp_path / "s.npz", rig="sixarm", tool="inline",
                  out=tmp_path / "p", name="bad")


def test_a_tool_that_is_not_the_planned_one_is_refused(tmp_path):
    spec, q = _floor_arm_stroke()
    _write_npz(tmp_path / "s.npz", 13, q, [0] * len(q))   # the inline pen
    with pytest.raises(ValueError, match="axial tip depth"):
        ep.export(tmp_path / "s.npz", rig="sixarm", tool="lateral",
                  out=tmp_path / "p", name="wrongtool")


def test_an_arm_that_never_lifts_gets_a_solved_end_ramp(tmp_path):
    """Arm 71 finishes its last stroke and holds the pen down for 216 frames.

    There is no certified retract to take, so the exporter solves one — a ramp
    of `ik.solve_cc` rows a centimetre apart, not one 5 cm jump, because one
    jump was 0.13 rad in a single row (4x the conductor's own bound).
    """
    spec, q = _floor_arm_stroke(y=0.90, x0=0.45)
    _write_npz(tmp_path / "s.npz", 13, q, [0] * len(q))   # starts AND ends down
    pw, = ep.export(tmp_path / "s.npz", rig="sixarm", tool="inline",
                    out=tmp_path / "p", name="flat")
    kinds = [w["kind"] for w in _load_verbatim(pw.paths["csv"])]
    assert kinds[0] == "lift_start" and kinds[-1] == "lift_end"
    assert kinds.count("draw") == len(q)
    n = pw.manifest["transits"]["synthesized_end_rows"]
    assert n["pre"] > 1 and n["post"] > 1
    Q = _joints(_rows(pw.paths["csv"]))
    assert np.abs(np.diff(Q, axis=0)).max() < MAX_ROW_DQ
    # the ramp really does climb, along the pen axis
    P = np.array([w["xyz"] for w in _load_verbatim(pw.paths["csv"])])
    assert P[0, 2] == pytest.approx(P[n["pre"], 2] + ep.LIFT_M, abs=1e-6)
    assert P[-1, 2] == pytest.approx(P[-1 - n["post"], 2] + ep.LIFT_M, abs=1e-6)


# ---------------------------------------------------------------------------
# 5. the manifest
# ---------------------------------------------------------------------------
def test_manifest_carries_the_contract_fields(floor_export):
    m = json.loads(floor_export["manifest"].read_text())
    for k in ("arm_id", "rig", "tool", "T_world_base", "paper_z_base_m",
              "joint_columns", "generator", "source", "speeds", "created"):
        assert k in m, k
    assert m["arm_id"] == 13
    assert m["rig"] == "sixarm"
    assert m["tool"]["name"] == "inline"
    assert m["tool"]["tip_offset_hand_tcp_m"] == [0.0, 0.0, frames.PEN_EXT]
    assert len(m["T_world_base"]) == 16
    T = np.array(m["T_world_base"]).reshape(4, 4)
    assert np.allclose(T, floor_export["spec"].T_world_base())
    assert m["joint_columns"] is True
    assert m["generator"]["module"] == "aris_sixarm.export.pathway"
    assert "git_sha" in m["generator"]
    assert m["source"]["schedule"].endswith("two.npz")
    assert set(m["speeds"]) >= {"draw_m_s", "travel_m_s"}
    assert m["created"].startswith("20")

    # the paper plane, and the promise that nothing else is in z_m
    assert m["paper_z_base_m"] == pytest.approx(-fleet.Z_FLOOR_BASE, abs=1e-9)
    for w in _load_verbatim(floor_export["csv"]):
        if w["draw"]:
            assert w["xyz"][2] == pytest.approx(m["paper_z_base_m"], abs=1e-9)
        else:
            assert w["xyz"][2] > m["paper_z_base_m"]      # pen up is pen up

    # and the things a reader of this file has to be told
    assert m["transits"]["in_file"] is True
    assert m["transits"]["n_rows"] == m["stats"]["n_travel_rows"]
    assert m["intensity"]["value"] == 0.75
    assert m["quaternion"]["order"] == "xyzw"
    assert m["stats"]["n_strokes"] == 2
    assert len(m["strokes"]) == 2
    assert m["strokes"][0]["n_draw_rows"] == len(floor_export["qa"])
    assert m["strokes"][1]["n_draw_rows"] == len(floor_export["qb"])
    b, = m["boundaries"]
    assert set(b) >= {"from_seg", "to_seg", "boundary_dq_rad",
                      "n_transit_rows", "max_row_dq_rad", "transit_tip_len_m"}
    assert (b["from_seg"], b["to_seg"]) == (0, 1)


# ---------------------------------------------------------------------------
# 6. the shipped programme
# ---------------------------------------------------------------------------
@needs_shipped
def test_shipped_schedule_exports_and_lands_inside_the_arms_reach(tmp_path):
    pw, = ep.export(SHIPPED_NPZ, SHIPPED_JSON, rig="proposed", tool="lateral",
                    arms=[31], out=tmp_path, name="csail")
    s = pw.stats
    assert s["n_draw_rows"] > 0
    assert s["n_travel_rows"] > 0
    assert s["n_strokes"] > 0
    assert s["draw_length_m"] > 1.0

    wps = _load_verbatim(pw.paths["csv"])
    P = np.array([w["xyz"] for w in wps])
    assert len(P) == s["n_rows"]
    lo, hi = np.array(s["bbox_base_m"]["min"]), np.array(s["bbox_base_m"]["max"])
    assert (P.min(axis=0) == pytest.approx(lo)) and (P.max(axis=0)
                                                     == pytest.approx(hi))

    # REACH.  Arm 31 hangs 0.940 m over the paper, so every tip sits ~0.94 m
    # down the base z axis and the 3D distance from link0 is 0.94..1.25 m: the
    # familiar "< 0.9 m from the origin" is a FLOOR arm's number and no
    # inverted row could ever satisfy it.  What is 0.9 m here is the radius off
    # the base's own z axis, and the DRAWING rows — the ones the atlas gated —
    # stay inside it; a transit is allowed to swing wider (arm 71 reaches
    # 0.910 m).  The second bound is the chain's straight-line extension, a
    # sanity ceiling rather than a strict kinematic maximum.
    D = np.array([w["xyz"] for w in wps if w["draw"]])
    assert np.hypot(D[:, 0], D[:, 1]).max() < 0.9
    reach = (frames.DH[0][2] + frames.DH[2][2] + frames.DH[4][2]
             + frames.TCP_D + frames.PEN_EXT_HOLDER + frames.PEN_LAT_HOLDER)
    assert np.linalg.norm(P, axis=1).max() < reach
    # and the poses really are reachable, because they ARE certified poses: FK
    # of each row's own joints put it there (the exporter's gate, re-run here)
    assert ep.verify_rows(pw.rows, *ep.tool_offsets("lateral")) == []

    # every draw row is on the paper, inside the sheet
    Twb = fleet.rig("proposed")[0][31].T_world_base()
    W = D @ Twb[:3, :3].T + Twb[:3, 3]
    assert np.abs(W[:, 2]).max() < 1e-6
    sheet = fleet.rig("proposed")[1]
    assert W[:, 0].min() > -1e-6 and W[:, 0].max() < sheet[0] + 1e-6
    assert W[:, 1].min() > -1e-6 and W[:, 1].max() < sheet[1] + 1e-6


@needs_shipped
@pytest.mark.parametrize("arm", [31, 71])
def test_shipped_schedule_is_executable(tmp_path, arm):
    """The regression the Drake SIL found, on the real programme.

    (a) no row has empty joint cells, (b) no row-to-row joint step exceeds the
    conductor's own bound, (c) every stroke boundary worth more than 0.5 rad is
    flown over certified transit rows rather than jumped.
    """
    pw, = ep.export(SHIPPED_NPZ, SHIPPED_JSON, rig="proposed", tool="lateral",
                    arms=[arm], out=tmp_path, name="x")
    rows = _rows(pw.paths["csv"])
    assert all(r[f"q{i}"] != "" for r in rows for i in range(1, 8))

    Q = _joints(rows)
    step = np.abs(np.diff(Q, axis=0)).max()
    assert step < MAX_ROW_DQ, f"arm {arm}: {step:.4f} rad between two rows"
    assert pw.stats["max_row_dq_rad"] == pytest.approx(step)
    assert (Q >= frames.FR3_MIN).all() and (Q <= frames.FR3_MAX).all()

    big = [b for b in pw.manifest["boundaries"] if b["boundary_dq_rad"] > 0.5]
    assert big, "the shipped programme reconfigures between strokes"
    assert max(b["boundary_dq_rad"] for b in big) > 3.0
    for b in big:
        assert b["n_transit_rows"] > 0, b
        assert b["max_row_dq_rad"] < MAX_ROW_DQ, b


@needs_shipped
def test_shipped_schedule_stroke_bookkeeping(tmp_path):
    """The stroke ids in the manifest are the programme's own."""
    pw, = ep.export(SHIPPED_NPZ, SHIPPED_JSON, rig="proposed", tool="lateral",
                    arms=[31], out=tmp_path, name="csail")
    m = pw.manifest
    assert [s["seg"] for s in m["strokes"]] == list(range(len(m["strokes"])))
    prog = json.loads(SHIPPED_JSON.read_text())
    want = [e["stroke_id"] for ph in prog["phases"] for e in ph["arms"]["31"]]
    assert [s["stroke_id"] for s in m["strokes"]] == want
    assert {s["ink"] for s in m["strokes"]} == {"grey", "orange"}

    # the orientation is held along a stroke and turns during the transit —
    # where it belongs, and where the executor now has rows to slerp over
    assert m["stats"]["max_quat_step_within_stroke_deg"] < 1.0

    rows = _rows(pw.paths["csv"])
    idx = [int(r["stroke_idx"]) for r in rows]
    assert sorted(set(idx)) == [s["seg"] for s in m["strokes"]]
    assert sum(1 for r in rows if r["kind"] == "draw") == sum(
        s["n_draw_rows"] for s in m["strokes"])


def test_base_transform_is_h_inv_invariant_on_the_proposed_rig():
    """The TRAP of docs/DECISIONS.md, pinned: `layout.study_spec` carries the
    base pose EXPLICITLY, so a bare `T_world_base()` is right and no `h_inv`
    can move it.  If that ever regresses, `base_transform` refuses."""
    fl, _ = fleet.rig("proposed")
    for aid, spec in fl.items():
        T = ep.base_transform(spec, h_inv=1.0)
        assert np.allclose(T, spec.T_world_base())
        assert np.allclose(T, spec.T_world_base(0.5))
        z, tilt = ep.paper_z_base(T)
        assert tilt < 1e-9
        assert z == pytest.approx(0.940, abs=1e-9)
