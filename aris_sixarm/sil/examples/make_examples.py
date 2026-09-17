#!/usr/bin/env python3
"""Generate the SIL's example pathway CSVs (contract §1, CSV v2).

    .venv/bin/python aris_sixarm/sil/examples/make_examples.py
    .venv/bin/python aris_sixarm/sil/examples/make_examples.py --search

WHAT IT WRITES.  One 10 cm straight draw stroke per example, rows every 1 mm
(101 draw rows) bracketed by a `lift_start` / `lift_end` pen-up row at the same
poses, in the ARM's own base frame, with the `q1..q7` columns filled from this
repo's analytic IK.  Plus the `<name>.manifest.json` sidecar of contract §1.

  line10cm_arm31.csv        rig `proposed`, arm 31 (ceiling), LATERAL holder
  line10cm_arm13_floor.csv  rig `final6_opt`, arm 13 (floor), INLINE pen

HOW THE SPOT WAS CHOSEN.  `--search` scans a 0.45 m half-span grid of canvas
points around the arm's base, 8 tool yaws (`atlas._YAWS`), the whole
`ik.Q7_GRID` and every FR3-valid branch, and picks the configuration with the
best `min(joint_margin, 2.5 * sigma_min)` -- the same two gates the atlas uses
(`metrics.GATE_MARGIN` 0.30 rad, `GATE_SIGMA` 0.14).  The winners are the
defaults below, so the normal run is deterministic and needs no search:

  arm 31 lateral   canvas (1.047, 1.815), yaw 135 deg, margin 0.612, sigma 0.233
  arm 13 inline    canvas (1.330, 0.323), yaw 315 deg, margin 0.841, sigma 0.284

ORIENTATION.  `R_world_tcp = rotz(yaw) @ rotx(pi)` -- the atlas's own
perpendicular candidate, pen straight down, no lean.  The row quaternion is
that rotation expressed in the arm's base frame, which is the deployed
convention: `(1,0,0,0)` = pen straight down on an axis-aligned floor arm.

WHAT EACH ROW IS.  Position = the NOMINAL pen tip (`setEE`, contract §4) ON
the paper plane; `z_m` is therefore the paper plane in the base frame, as
contract §1 asks.  Every row is FK-verified through `frames.tip_pos` /
`frames.fk` to < 0.5 mm and < 0.5 deg -- the assertion the real exporter makes.
"""
import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aris_sixarm import fleet, frames, ik                      # noqa: E402
from aris_sixarm.metrics import sigma_min, tip_jacobian        # noqa: E402
from aris_sixarm.sil.rotations import quat_from_matrix         # noqa: E402

HERE = Path(__file__).resolve().parent
COLUMNS = ["stroke_idx", "wp_idx", "kind", "x_m", "y_m", "z_m",
           "qx", "qy", "qz", "qw", "intensity"] + [f"q{i}" for i in range(1, 8)]
YAWS = np.linspace(0, 2 * np.pi, 8, endpoint=False)

EXAMPLES = (
    dict(name="line10cm_arm31", rig="proposed", arm=31, tool="lateral",
         center=(1.047, 1.815), yaw_deg=135.0, direction=(1.0, 0.0)),
    dict(name="line10cm_arm13_floor", rig="final6_opt", arm=13, tool="inline",
         center=(1.330, 0.323), yaw_deg=315.0, direction=(1.0, 0.0)),
)


def tool_offset(tool: str) -> np.ndarray:
    """The nominal tip in the hand-TCP frame, without touching the globals."""
    if tool == "lateral":
        return np.array([frames.PEN_LAT_HOLDER, 0.0, frames.PEN_EXT_HOLDER])
    return np.array([0.0, 0.0, frames.PEN_EXT])


def search(rig: str, arm: int, tool: str, span: float = 0.45, n: int = 13):
    """Scan for the most comfortable perpendicular draw pose. -> printout."""
    off = tool_offset(tool)
    T_wb = fleet.rig(rig)[0][arm].T_world_base()
    T_bw = np.linalg.inv(T_wb)
    seed = fleet.rig(rig)[0][arm].q_seed
    bx, by = T_wb[0, 3], T_wb[1, 3]
    best = None
    for x in np.linspace(bx - span, bx + span, n):
        for y in np.linspace(by - span, by + span, n):
            for yaw in YAWS:
                R = frames.rotz(yaw) @ frames.rotx(np.pi)
                T_w = np.eye(4)
                T_w[:3, :3], T_w[:3, 3] = R, np.array([x, y, 0.0]) - R @ off
                T_b = T_bw @ T_w
                for q7 in ik.Q7_GRID:
                    for q in ik.solve(T_b, q7, seed):
                        m = frames.joint_margin(q)
                        s = sigma_min(tip_jacobian(q, pen_ext=off[2],
                                                   pen_lat=off[0]))
                        score = min(m, 2.5 * s)
                        if best is None or score > best[0]:
                            best = (score, m, s, x, y, np.degrees(yaw))
    score, m, s, x, y, yaw = best
    print(f"{rig} arm {arm} {tool}: canvas ({x:.3f}, {y:.3f}) yaw {yaw:.0f} deg"
          f"  margin {m:.3f} sigma {s:.3f} score {score:.3f}")


def build_rows(spec: dict, length_m: float = 0.10, pitch_m: float = 0.001):
    """-> (rows, meta). Rows are dicts keyed by `COLUMNS`."""
    off = tool_offset(spec["tool"])
    arm_spec = fleet.rig(spec["rig"])[0][spec["arm"]]
    T_wb = arm_spec.T_world_base()
    T_bw = np.linalg.inv(T_wb)
    R_w = frames.rotz(np.deg2rad(spec["yaw_deg"])) @ frames.rotx(np.pi)
    R_b = T_bw[:3, :3] @ R_w
    quat = quat_from_matrix(R_b)

    d = np.asarray(spec["direction"], float)
    d = d / np.linalg.norm(d)
    n_pts = int(round(length_m / pitch_m)) + 1
    s = np.linspace(-0.5 * length_m, 0.5 * length_m, n_pts)
    cx, cy = spec["center"]
    tips_w = np.stack([cx + s * d[0], cy + s * d[1], np.zeros(n_pts)], axis=1)
    tips_b = tips_w @ T_bw[:3, :3].T + T_bw[:3, 3]

    seed = arm_spec.q_seed
    q_prev, qs = None, []
    for p_b in tips_b:
        T_b = np.eye(4)
        T_b[:3, :3], T_b[:3, 3] = R_b, p_b - R_b @ off
        if q_prev is None:
            cands = [q for q7 in ik.Q7_GRID for q in ik.solve(T_b, q7, seed)]
            if not cands:
                raise SystemExit(f"{spec['name']}: no IK at the first row")
            q = max(cands, key=frames.joint_margin)
        else:
            q = ik.solve_cc(T_b, q_prev[6], q_prev)
            if q is None:
                raise SystemExit(f"{spec['name']}: IK lost continuity")
        qs.append(q)
        q_prev = q
    qs = np.array(qs)

    # the exporter's own assertion: FK(q) through the same EE frame reproduces
    # the row to < 0.5 mm / < 0.5 deg.
    tip_fk = np.array([frames.tip_pos(q, pen_ext=off[2], pen_lat=off[0])
                       for q in qs])
    pos_err = float(np.abs(tip_fk - tips_b).max())
    rot_err = max(float(np.arccos(np.clip(
        (np.trace(frames.fk(q)[0][:3, :3].T @ R_b) - 1) / 2, -1, 1)))
        for q in qs)
    if pos_err > 5e-4 or rot_err > np.deg2rad(0.5):
        raise SystemExit(f"{spec['name']}: FK check failed "
                         f"{pos_err * 1e3:.3f} mm / {np.degrees(rot_err):.3f} deg")

    rows = []

    def row(kind, i, p, q, tone):
        rows.append(dict(zip(COLUMNS, [
            0, i, kind, f"{p[0]:.6f}", f"{p[1]:.6f}", f"{p[2]:.6f}",
            f"{quat[0]:.6f}", f"{quat[1]:.6f}", f"{quat[2]:.6f}",
            f"{quat[3]:.6f}", f"{tone:.3f}",
            *[f"{v:.6f}" for v in q]])))

    row("lift_start", 0, tips_b[0], qs[0], 0.0)
    for i, (p, q) in enumerate(zip(tips_b, qs)):
        row("draw", i + 1, p, q, 1.0)
    row("lift_end", n_pts + 1, tips_b[-1], qs[-1], 0.0)

    meta = {
        "arm_id": spec["arm"], "rig": spec["rig"],
        "tool": {"name": spec["tool"], "tip_offset_hand_tcp_m": off.tolist()},
        "T_world_base": [float(v) for v in T_wb.reshape(-1)],
        "paper_z_base_m": float(tips_b[0, 2]),
        "joint_columns": True,
        "generator": "aris_sixarm/sil/examples/make_examples.py",
        "source": "synthetic 10 cm line, SIL example",
        "speeds": {"draw_m_s": 0.02, "travel_m_s": 0.04},
        "created": date.today().isoformat(),
        "fk_check": {"max_pos_err_m": pos_err,
                     "max_rot_err_rad": float(rot_err)},
        "joint_margin_rad": float(min(frames.joint_margin(q) for q in qs)),
        "sigma_min": float(min(sigma_min(tip_jacobian(q, pen_ext=off[2],
                                                      pen_lat=off[0]))
                               for q in qs)),
    }
    return rows, meta


def write(spec: dict, out_dir: Path) -> None:
    rows, meta = build_rows(spec)
    csv_path = out_dir / f"{spec['name']}.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / f"{spec['name']}.manifest.json").write_text(
        json.dumps(meta, indent=2) + "\n")
    print(f"{csv_path.name}: {len(rows)} rows, margin "
          f"{meta['joint_margin_rad']:.3f} rad, sigma {meta['sigma_min']:.3f}, "
          f"FK {meta['fk_check']['max_pos_err_m'] * 1e3:.4f} mm")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--search", action="store_true",
                    help="rescan for the most comfortable spot and print it")
    ap.add_argument("--out", type=Path, default=HERE)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for spec in EXAMPLES:
        if args.search:
            search(spec["rig"], spec["arm"], spec["tool"])
        else:
            write(spec, args.out)


if __name__ == "__main__":
    main()
