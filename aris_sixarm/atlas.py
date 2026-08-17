"""Reachability atlas: sweep the paper plane per arm with analytic IK.

Per cell (2cm grid default), candidates = 8 tool-yaws x 16 q7 x 4 IK branches
with the pen perpendicular; if none survives, a tilt-cone rescue up to
`tilt_max_deg` (half- and full-tilt about tool x/y). Metrics:

  margin      worst joint-limit distance (rad) of the best solution
  sigma_min   force controllability at the best solution (pen-tip Jacobian)
  f_max       max downward force before a torque limit saturates (N)
  n_sol       count of limit-valid (candidate, q7, branch) solutions
  valid_frac  fraction of the (candidate x q7) grid with a comfortable
              solution (margin >= 0.15) — "how many options do we have here"
  q7_window   widest CONTIGUOUS q7 interval (rad) that stays comfortable at a
              single tool orientation — the self-motion corridor a stroke can
              glide through without branch jumps
  tilt_deg    pen lean that was needed (0 = perpendicular reached)

Clearance: chain points >= 2cm above paper; inverted arms keep out of the
boom cylinder (r < 0.12 above the mount plate). Proxy checks inherited from
the IKA toolkit — replace with real scene geometry when it matters.
"""
import time
from pathlib import Path

import numpy as np

from . import ik
from .fleet import FLEET, SHEET, H_INV_DEFAULT
from .frames import fk, rotx, rotz, rot_axis, PEN_EXT, joint_margin
from .metrics import tip_jacobian, sigma_min, f_max, GATE_MARGIN, GATE_SIGMA

COLUMNS = ["x", "y", "margin", "sigma_min", "f_max", "n_sol", "valid_frac",
           "q7_window", "tilt_deg", "q1", "q2", "q3", "q4", "q5", "q6", "q7"]
QCOL = 9                      # index of q1 in COLUMNS
PERMISSIVE_MARGIN = 0.15      # option counted as "comfortable enough"

_YAWS = np.linspace(0, 2 * np.pi, 8, endpoint=False)
_DQ7 = float(ik.Q7_GRID[1] - ik.Q7_GRID[0])


def _candidates(tilt_max_deg):
    """(locked, tilted) lists of (tilt_deg, R_world_tcp)."""
    locked, tilted = [], []
    for yaw in _YAWS:
        R = rotz(yaw) @ rotx(np.pi)          # pen straight down
        locked.append((0.0, R))
        if tilt_max_deg > 0:
            for ang_deg in (0.5 * tilt_max_deg, tilt_max_deg):
                a = np.deg2rad(ang_deg)
                for ax in ((1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)):
                    tilted.append((ang_deg, rot_axis(R @ np.array(ax), a) @ R))
    return locked, tilted


def _q7_window(valid_row):
    """Longest contiguous run of True -> corridor width in rad."""
    best = run = 0
    for v in valid_row:
        run = run + 1 if v else 0
        best = max(best, run)
    return best * _DQ7


def solve_cell(x, y, Twb, Twb_inv, mount, seed, cand_sets):
    """-> (margin, sigma_min, f_max, n_sol, valid_frac, q7_window, tilt_deg, q)
    or None."""
    tip_w = np.array([x, y, 0.0])
    press_b = Twb_inv[:3, :3] @ np.array([0, 0, -1.0])
    for cand in cand_sets:
        if not cand:
            continue
        sols, n_sol = [], 0
        valid = np.zeros((len(cand), len(ik.Q7_GRID)), bool)
        for i, (tilt_deg, R_w) in enumerate(cand):
            T_w = np.eye(4)
            T_w[:3, :3] = R_w
            T_w[:3, 3] = tip_w - PEN_EXT * R_w[:, 2]
            T_b = Twb_inv @ T_w
            T16 = T_b  # ik.solve flattens
            for j, q7 in enumerate(ik.Q7_GRID):
                for q in ik.solve(T16, q7, seed):
                    n_sol += 1
                    m = joint_margin(q)
                    if m >= PERMISSIVE_MARGIN:
                        valid[i, j] = True
                    sols.append((m, tilt_deg, q))
        if not sols:
            continue
        sols.sort(key=lambda t: -t[0])
        for m, tilt_deg, q in sols[:6]:       # clearance-check best few
            _, pts = fk(q)
            pts_w = (Twb[:3, :3] @ pts.T).T + Twb[:3, 3]
            if np.any(pts_w[1:, 2] < 0.02):
                continue
            if mount == "inv":
                rb = np.hypot(pts[:, 0], pts[:, 1])
                if np.any((pts[:, 2] < -0.02) & (rb < 0.12)):
                    continue
            J = tip_jacobian(q)
            vf = float(valid.mean())
            qw = float(max(_q7_window(row) for row in valid))
            return (m, sigma_min(J), f_max(J, press_b), n_sol, vf, qw,
                    tilt_deg, q)
    return None


def sweep_arm(arm_id, out_dir, grid=0.02, rmax=1.05, h_inv=H_INV_DEFAULT,
              tilt_max_deg=15.0):
    spec = FLEET[arm_id]
    Twb = spec.T_world_base(h_inv)
    Twb_inv = np.linalg.inv(Twb)
    cand_sets = _candidates(tilt_max_deg)
    bx, by = spec.xy
    rows = []
    t0 = time.time()
    for y in np.arange(0.0, SHEET[1] + 1e-9, grid):
        for x in np.arange(0.0, SHEET[0] + 1e-9, grid):
            if (x - bx) ** 2 + (y - by) ** 2 > rmax ** 2:
                continue
            r = solve_cell(x, y, Twb, Twb_inv, spec.mount, spec.q_seed, cand_sets)
            if r is not None:
                rows.append([x, y, *r[:7], *r[7]])
    arr = np.array(rows) if rows else np.zeros((0, len(COLUMNS)))
    out = Path(out_dir) / f"atlas_arm{arm_id}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, data=arr, columns=np.array(COLUMNS), arm_id=arm_id,
                        mount=spec.mount, base=Twb, grid=grid, h_inv=h_inv,
                        tilt_max_deg=tilt_max_deg)
    go = strict_go(arr)
    print(f"arm {arm_id} ({spec.name}): {len(arr)} reachable, "
          f"{int(go.sum())} strict-GO, tilt<={tilt_max_deg:.0f}deg, "
          f"{time.time() - t0:.0f}s")
    return arr


def strict_go(arr):
    if not len(arr):
        return np.zeros(0, bool)
    return (arr[:, 2] >= GATE_MARGIN) & (arr[:, 3] >= GATE_SIGMA)


def load(out_dir, arm_id):
    d = np.load(Path(out_dir) / f"atlas_arm{arm_id}.npz")
    return d["data"], d
