"""Reachability atlas: sweep the paper plane per arm with analytic IK.

Per cell (2cm grid default), best over 8 tool-yaws x 16 q7 x 4 IK branches
(15deg tilt-cone rescue if the perpendicular pen fails):
  margin, sigma_min, f_max, n_sol (redundancy richness), tilt_deg, q_best.
Clearance: chain points >= 2cm above paper; inverted arms keep out of the
boom cylinder (r < 0.12 above the mount plate). Proxy checks inherited from
the IKA toolkit — replace with real scene geometry when it matters.
"""
import time
from pathlib import Path

import numpy as np

from . import ik
from .fleet import FLEET, SHEET, H_INV_DEFAULT
from .frames import fk, rotx, rotz, rot_axis, PEN_EXT
from .metrics import tip_jacobian, sigma_min, f_max, GATE_MARGIN, GATE_SIGMA

COLUMNS = ["x", "y", "margin", "sigma_min", "f_max", "n_sol", "tilt_deg",
           "q1", "q2", "q3", "q4", "q5", "q6", "q7"]

_YAWS = np.linspace(0, 2 * np.pi, 8, endpoint=False)
_TILTS = [(ang, ax) for ang in (np.deg2rad(7.5), np.deg2rad(15.0))
          for ax in ((1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0))]


def _rotations():
    locked, tilted = [], []
    for yaw in _YAWS:
        R = rotz(yaw) @ rotx(np.pi)          # pen straight down
        locked.append((0.0, R))
        for ang, ax in _TILTS:
            tilted.append((np.rad2deg(ang), rot_axis(R @ np.array(ax), ang) @ R))
    return locked, tilted


_LOCKED, _TILTED = _rotations()


def solve_cell(x, y, Twb, Twb_inv, mount, seed):
    """-> (margin, sigma_min, f_max, n_sol, tilt_deg, q) or None."""
    tip_w = np.array([x, y, 0.0])
    press_b = Twb_inv[:3, :3] @ np.array([0, 0, -1.0])
    for cand in (_LOCKED, _TILTED):
        sols, n_sol = [], 0
        for tilt_deg, R_w in cand:
            T_w = np.eye(4)
            T_w[:3, :3] = R_w
            T_w[:3, 3] = tip_w - PEN_EXT * R_w[:, 2]
            T_b = Twb_inv @ T_w
            for _, q, m in ik.scan(T_b, seed):
                n_sol += 1
                sols.append((m, tilt_deg, q))
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
            return m, sigma_min(J), f_max(J, press_b), n_sol, tilt_deg, q
    return None


def sweep_arm(arm_id, out_dir, grid=0.02, rmax=1.05, h_inv=H_INV_DEFAULT):
    spec = FLEET[arm_id]
    Twb = spec.T_world_base(h_inv)
    Twb_inv = np.linalg.inv(Twb)
    bx, by = spec.xy
    rows = []
    t0 = time.time()
    for y in np.arange(0.0, SHEET[1] + 1e-9, grid):
        for x in np.arange(0.0, SHEET[0] + 1e-9, grid):
            if (x - bx) ** 2 + (y - by) ** 2 > rmax ** 2:
                continue
            r = solve_cell(x, y, Twb, Twb_inv, spec.mount, spec.q_seed)
            if r is not None:
                m, sm, fm, n, tilt, q = r
                rows.append([x, y, m, sm, fm, n, tilt, *q])
    arr = np.array(rows) if rows else np.zeros((0, len(COLUMNS)))
    out = Path(out_dir) / f"atlas_arm{arm_id}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, data=arr, columns=np.array(COLUMNS), arm_id=arm_id,
                        mount=spec.mount, base=Twb, grid=grid, h_inv=h_inv)
    go = strict_go(arr)
    print(f"arm {arm_id} ({spec.name}): {len(arr)} reachable, "
          f"{int(go.sum())} strict-GO, {time.time() - t0:.0f}s")
    return arr


def strict_go(arr):
    if not len(arr):
        return np.zeros(0, bool)
    return (arr[:, 2] >= GATE_MARGIN) & (arr[:, 3] >= GATE_SIGMA)


def load(out_dir, arm_id):
    d = np.load(Path(out_dir) / f"atlas_arm{arm_id}.npz")
    return d["data"], d
