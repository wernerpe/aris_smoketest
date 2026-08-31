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
from . import rig_final
from .fleet import FLEET, SHEET, H_INV_DEFAULT
from .frames import (fk, rotx, rotz, rot_axis, PEN_EXT, joint_margin,
                     lat_of, tool_offset, FR3_MIN, FR3_MAX)
from .metrics import tip_jacobian, sigma_min, f_max, GATE_MARGIN, GATE_SIGMA

COLUMNS = ["x", "y", "margin", "sigma_min", "f_max", "n_sol", "valid_frac",
           "q7_window", "tilt_deg", "q1", "q2", "q3", "q4", "q5", "q6", "q7",
           "min_lean_deg", "flat_margin"]
QCOL = 9                      # index of q1 in COLUMNS
LEANCOL = 16                  # index of min_lean_deg — APPENDED, on purpose:
#   every consumer that slices `arr[:, QCOL:QCOL + 7]` or reads a column by the
#   index it has always had keeps working, and the new column is only seen by
#   code that asks for it.
FLATCOL = 17                  # ...and `flat_margin`, which exists because the
#   gated search took an answer away.  `tilt_deg` used to mean "the lean of the
#   best-margin pose that cleared metal here", and for the great majority of
#   reachable cells that was 0 — so `tilt_deg == 0 and margin >= 0.15` was a
#   usable statement of FLAT REACHABILITY, and `allocate.atlas_cells` has read
#   it as one for as long as there has been a prefilter.  The gated search
#   changed what the row records: a cell that only certifies at 5 degrees now
#   carries `tilt_deg = 5`, and the flat pose it can still reach — uncomfortably,
#   below the strict gate, but reachable — stopped being written down.  The
#   allocator's prefilter shrank, arms stopped being offered ink they can draw,
#   and the logo lost 2.5 points of coverage to a map that had just gained 7.6.
#   So the flat answer is recorded explicitly: the joint margin of the best
#   PERPENDICULAR pose that clears every obstacle here, or -1 if none does.


# ---------------------------------------------------------------------------
# AN ATLAS IS ONLY VALID FOR THE COLLISION MODEL IT WAS SWEPT UNDER
# ---------------------------------------------------------------------------
# Nothing recorded that until 2026-08-26, and then the mesh audit widened
# every capsule and re-derived the neighbour column, and a directory full of
# .npz files that had been true the day before became a directory full of
# poses this package would now refuse — with no way to tell, because an atlas
# carries its grid, its height and its pen length but never carried the
# geometry it was gated against.  It does now.  `model_signature` is the
# whole collision model as one flat array; `is_current` compares a loaded
# atlas's stored copy against the running one, and a caller that gets False
# is holding stale certifications, not a difference of opinion.
#
# ...AND NOT ONLY FOR THE COLLISION MODEL: FOR THE SEARCH POLICY TOO.
# 2026-08-26 added a second way for an atlas to be stale.  `solve_cell` used to
# return the single best-MARGIN pose that cleared metal, and the strict gates
# were applied to that one row afterwards; it now SEARCHES the fiber and the
# tilt cone for a pose that PASSES those gates.  Two atlases swept under
# identical geometry by those two policies certify different sets of cells, and
# nothing in the file would have said so.  So the signature carries the search
# policy — the cone grid the gated search may use, and a version scalar for the
# policy itself — alongside the geometry, and `SEARCH_POLICY` is bumped by hand
# whenever the search changes what it is willing to certify.
SEARCH_POLICY = 2.0           # 1 = best-margin-then-gate, 2 = gated search


def model_signature():
    """The collision model AND search policy this process sweeps with. -> (N,)"""
    from . import mounts
    from . import selfcoll
    caps = rig_final.STATIC_CAPSULES_LAT + rig_final.STATIC_CAPSULES
    flat = [v for c in caps for v in c]
    flat += [v for band in mounts.MOUNTS.column_bands for v in band]
    flat += [rig_final.STATIC_MARGIN, GATE_MARGIN, GATE_SIGMA]
    flat += [SEARCH_POLICY, *GATE_CONE_DEG]
    return np.concatenate([np.asarray(flat, float), selfcoll.signature()])


def is_current(meta):
    """Was this loaded atlas swept under the model this process is running?

    -> (bool, reason).  An atlas with no signature at all predates the mesh
    audit and is reported as such rather than trusted.
    """
    if "model" not in getattr(meta, "files", ()):
        return False, ("swept before 2026-08-26: no collision-model signature "
                       "(pre-mesh-audit capsules)")
    got = np.asarray(meta["model"], float)
    want = model_signature()
    if got.shape != want.shape:
        return False, (f"model signature is {got.shape}, this build "
                       f"wants {want.shape}")
    if not np.allclose(got, want, atol=1e-12):
        bad = int(np.argmax(np.abs(got - want)))
        return False, (f"collision model differs at entry {bad}: atlas has "
                       f"{got[bad]:.4f}, this build has {want[bad]:.4f}")
    return True, "current"
PERMISSIVE_MARGIN = 0.15      # option counted as "comfortable enough"

_YAWS = np.linspace(0, 2 * np.pi, 8, endpoint=False)
_DQ7 = float(ik.Q7_GRID[1] - ik.Q7_GRID[0])


def _candidates(tilt_max_deg):
    """(locked, tilted) lists of (tilt_deg, R_world_tcp).

    THE LEGACY CANDIDATE SET, kept exactly as it was.  It is what the
    best-margin fallback below still searches, so a cell that certifies at no
    lean records the same row it always did.
    """
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


# THE CONE THE GATED SEARCH MAY USE, AND WHY IT IS FINER THAN THE LEGACY ONE.
# `_candidates` offers two lean magnitudes, half the cone and all of it, which
# is all a CLEARANCE fallback ever needed: it was reached only when no
# perpendicular pose cleared metal, and then any pose that cleared would do.
# A GATE fallback is a different question — "what is the LEAST lean at which
# this cell certifies" — and the answer is spread across the whole cone.
# `scripts/dead_disc_anatomy.py` measured it on the 1 321 dead cells of the
# shipped map: 56 of them need no lean at all (the old search simply missed
# them), and the rest are spread 2.5-15 deg with the mode at 2.5.  Two
# magnitudes would round most of that up to 7.5 and lean the pen three times
# further than the geometry asks.  So the gated search walks this grid in
# ascending order and stops at the first lean that certifies, and the lean it
# stopped at is recorded per cell as `min_lean_deg`.
GATE_CONE_DEG = (2.5, 5.0, 7.5, 10.0, 12.5, 15.0)
_TILT_AXES = ((1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0))

# THE FILTER AND THE RECORD MUST NOT DISAGREE ABOUT THE GATE.  The gated search
# screens the fiber with the ANALYTIC batch Jacobian (one SVD for the whole
# fiber) and the row it finally writes carries `metrics.sigma_min` of the
# FINITE-DIFFERENCE Jacobian, because that is the number every atlas has ever
# stored and every consumer compares against `GATE_SIGMA`.  The two agree to
# ~3e-10 — but "agree to 3e-10" is not "agree", and a pose screened in at
# exactly 0.140000000 could be recorded at 0.139999999 and then fail the very
# gate it was selected for.  The invariant this file owes its callers is that
# `min_lean_deg >= 0` means strict-GO, so the screen keeps a margin thirty
# times the known disagreement and the invariant is pinned by a test.
_SIGMA_EPS = 1e-8


def _gated_groups(tilt_max_deg, cone=GATE_CONE_DEG):
    """Orientation groups for the GATED search, ordered by ascending lean.

    -> [(lean_deg, [(lean_deg, R_world_tcp), ...]), ...].  Group 0 is the
    perpendicular pen at the same 8 tool yaws the atlas has always used, so
    FLAT IS TRIED FIRST AND WHOLE: a cell that certifies without leaning never
    reaches the cone, and its row is the row the legacy search would have
    returned whenever the legacy search's own best-margin pick was gate-clean.
    """
    R0 = [rotz(yaw) @ rotx(np.pi) for yaw in _YAWS]
    groups = [(0.0, [(0.0, R) for R in R0])]
    for deg in cone:
        if deg > tilt_max_deg + 1e-9:
            break
        a = np.deg2rad(deg)
        cand = [(deg, rot_axis(R @ np.array(ax), a) @ R)
                for R in R0 for ax in _TILT_AXES]
        groups.append((float(deg), cand))
    return groups


def _q7_window(valid_row):
    """Longest contiguous run of True -> corridor width in rad."""
    best = run = 0
    for v in valid_row:
        run = run + 1 if v else 0
        best = max(best, run)
    return best * _DQ7


def _clears(q, Twb, legacy_inv, boxes, off, lat, static=None):
    """The per-pose geometric gates: paper, the legacy boom, the frame steel.

    `static` is the floor the pose must keep to the neighbours' boxes;
    `None` is `rig_final.STATIC_MARGIN`, the gate every shipped atlas was swept
    at.  IT IS A PARAMETER BECAUSE THE ATLAS AND THE ROUTER DO NOT AGREE ABOUT
    IT AND THE DISAGREEMENT IS LOAD-BEARING.  `rig_final` says so itself: the
    checker keeps `STATIC_MARGIN` = 50 mm, the PRODUCER must keep
    `STATIC_PLAN_MARGIN` = 63 mm (50 + the 13 mm of slack `scene_check`'s own
    independent lower bound carries), and "an atlas is now an OPTIMISTIC
    prefilter by up to 13 mm".

    AND THE BAND IS THE DEAD SET, EXACTLY.  Over the v11 map's whole raw:
    772 of the 1 112 refused arm-cells on dead canvas cells have a drawing pose
    in [50, 63), and ZERO of the 22 437 FEASIBLE arm-cells do — not a sample,
    all of them.  Every cell the map could fly to already had a pose clear of
    the router's floor, so re-gating there cannot cost a cell.

    A POSE IN THAT BAND IS NOT REFUSED — `paper.effective_static_floor` clamps
    a leg's floor down to what its own ENDPOINTS hold, and exists to stop
    exactly that contradiction.  What it is, is a pose with NOTHING LEFT OVER.
    Clamped, the leg out of it must hold the pose's own clearance along its
    whole swing, and a swing dips.  Measured on 36 route-dead cells of the v11
    map, over all 48 hovers on each one's fiber: 27 are walled at that clamped
    static floor and nothing else binds on any of them — not the paper, not the
    tip, not the arm against itself — and the best hover misses by a MEDIAN OF
    1.4 mm.  Their drawing poses sit at a median 58.5 mm, on the gate.

    AND THE POSE IS NOT THE CELL.  This function returns the FIRST gated pose
    that clears, in descending joint-margin order, and never asks whether
    another clears more.  Probed on all 772 banded arm-cells of the v11 dead
    set, 746 (96.6 %) have a different pose at the same cell — another tool
    yaw, another q7, another IK branch, or a lean inside the same 15-degree
    cone — that clears 63 mm, at a median of 120 mm, and 446 of them need no
    lean at all.  A sweep whose cells are meant to be FLOWN to should be given
    the producer's floor, and then the descent is not threading a gap.

    THE ARM AGAINST ITSELF IS NOT HERE, and deliberately.  It is the one gate
    whose cost is dominated by numpy overhead rather than by the pose (233
    capsule pairs, one broadcast), so it is applied to the WHOLE gate-passing
    fiber in one call by `_self_mask` below before this is reached — the same
    "stage the gates onto a shrinking index set" shape `planner` uses.  It is
    applied BEFORE the search settles rather than after it, because a gated
    search that leans the pen goes looking in exactly the folded corner of the
    fiber the fat comfort margins used to keep it out of
    (`aris_sixarm/selfcoll.py`).
    """
    T, pts = fk(q)
    pts_w = (Twb[:3, :3] @ pts.T).T + Twb[:3, 3]
    if np.any(pts_w[1:, 2] < 0.02):
        return None
    if legacy_inv:
        rb = np.hypot(pts[:, 0], pts[:, 1])
        if np.any((pts[:, 2] < -0.02) & (rb < 0.12)):
            return None
    if boxes:
        tool_pts = [T[:3, 3] + T[:3, :3] @ off]
        if lat != 0.0:
            tool_pts.append(T[:3, 3] + T[:3, :3] @ np.array([lat, 0.0, 0.0]))
        tool_w = [Twb[:3, :3] @ t + Twb[:3, 3] for t in tool_pts]
        P10 = np.vstack([pts_w] + [t[None] for t in tool_w])
        floor = rig_final.STATIC_MARGIN if static is None else float(static)
        if rig_final.chain_static_clearance(P10, boxes)[0] < floor:
            return None
    return T


def _self_mask(Q, pen_ext, lat):
    """The self-collision gate for a whole fiber at once. (N,7) -> (N,) bool."""
    from . import selfcoll
    if not len(Q):
        return np.zeros(0, bool)
    return selfcoll.self_ok(Q, margin=selfcoll.SELF_PLAN_MARGIN,
                            pen_ext=pen_ext, pen_lat=lat)


def solve_cell(x, y, Twb, Twb_inv, spec, cand_sets, pen_ext=PEN_EXT,
               boxes=(), pen_lat=None, gate_groups=None,
               gate_margin=GATE_MARGIN, gate_sigma=GATE_SIGMA,
               static_margin=None):
    """-> (margin, sigma_min, f_max, n_sol, valid_frac, q7_window, tilt_deg,
    q, min_lean_deg) or None.  `boxes`: the arm's static frame obstacles.

    THE SEARCH LOOKS FOR A POSE THAT PASSES THE GATES (2026-08-26).  It used to
    look for the best-MARGIN pose that cleared metal and let `strict_go` judge
    that one row afterwards, and the difference is not academic: on the shipped
    proposed-rig map that policy left 1 321 cells uncertified, and
    `scripts/dead_disc_anatomy.py` proved by exhaustive search that 1 261 of
    them have a pose which passes the SAME gates under the SAME obstacle set
    inside the SAME 15-degree cone.  56 of those need no lean at all — the
    best-margin pick simply happened to be an uncomfortable one — and the rest
    are reachable at leans the two-magnitude clearance fallback never offered,
    because that fallback only ever fired when NOTHING perpendicular cleared
    metal, which is a different question from whether anything perpendicular
    CERTIFIES.

    So the order is: for each lean in `gate_groups` (0 first, then the cone
    ascending), take every IK solution with margin >= `gate_margin`, keep those
    with sigma >= `gate_sigma`, and clearance-check them best-margin first; the
    first that clears is the answer and its lean is `min_lean_deg`.  Only if no
    lean in the whole cone yields a gated pose does it fall back to the LEGACY
    pick — the best-margin clearing pose over `cand_sets`, verbatim — so a cell
    that certifies nothing still records exactly the row it always did, and
    `min_lean_deg` is -1 to say the row is a reachability record and not a
    certificate.

    The consequence is monotone: no cell that was strict-GO can stop being one
    (the legacy pick that certified it is still in the gated search's own
    candidate set, and still wins its own margin ordering), and cells that were
    not can become one.  The only thing that takes a cell away is the new
    self-collision gate, which is a real refusal and not a change of policy.

    With the LATERAL tool (`pen_lat` != 0) the 8 tool yaws stop being redundant
    with q7 and become the REAL phi axis: each yaw puts the TCP at a different
    point of the 11 cm circle around the tip, so the same candidate sweep that
    always ran genuinely searches the new DOF.
    """
    mount, seed = spec.mount, spec.q_seed
    legacy_inv = mount == "inv" and getattr(spec, "rig", "sixarm") == "sixarm"
    lat = lat_of(pen_lat)
    off = tool_offset(pen_ext, lat)
    tip_tgt = np.array([x, y, 0.0])
    press_b = Twb_inv[:3, :3] @ np.array([0, 0, -1.0])

    nq7 = len(ik.Q7_GRID)

    def _solve_set(cand):
        """-> (sols, n_sol, valid) for one orientation set.

        ONE BATCHED SOLVE FOR THE WHOLE SET.  `ik.solve_batch` guarantees that
        `Q[i][:valid[i].sum()]` is exactly `ik.solve(...)` — same solutions,
        same order, same slots — so `sols` comes out in the identical order the
        nested scalar loop produced it in, and the margin sort that follows is
        therefore identical too.  It is worth saying out loud because the sort
        is what picks the pose: a different tie order would be a different
        atlas.  On an older wheel with no batch entry points the scalar loop
        below is used instead and produces the same list.
        """
        T_b = np.empty((len(cand), 4, 4))
        for i, (_, R_w) in enumerate(cand):
            T_w = np.eye(4)
            T_w[:3, :3] = R_w
            T_w[:3, 3] = tip_tgt - R_w @ off
            T_b[i] = Twb_inv @ T_w
        valid = np.zeros((len(cand), nq7), bool)
        sols = []
        if ik.has_batch():
            flat = np.repeat(ik._flat16(T_b), nq7, axis=0)
            Q, vld = ik.solve_batch(flat, np.tile(ik.Q7_GRID, len(cand)), seed)
            Q = Q.reshape(len(cand), nq7, 4, 7)
            vld = vld.reshape(len(cand), nq7, 4)
            n_sol = int(vld.sum())
            ii, jj, kk = np.nonzero(vld)
            if len(ii):
                q = Q[ii, jj, kk]
                m = np.min(np.minimum(q - FR3_MIN, FR3_MAX - q), axis=1)
                good = m >= PERMISSIVE_MARGIN
                valid[ii[good], jj[good]] = True
                td = np.array([cand[i][0] for i in ii])
                sols = list(zip(m.tolist(), td.tolist(), list(q)))
        else:
            n_sol = 0
            for i, (tilt_deg, _) in enumerate(cand):
                for j, q7 in enumerate(ik.Q7_GRID):
                    for q in ik.solve(T_b[i], q7, seed):
                        n_sol += 1
                        m = joint_margin(q)
                        if m >= PERMISSIVE_MARGIN:
                            valid[i, j] = True
                        sols.append((m, tilt_deg, q))
        return sols, n_sol, valid

    def _flat_best(sols, valid):
        """The old flat answer: best-margin perpendicular pose that clears.

        -> margin, or -1.0.  Same candidate ordering and same top-six
        truncation the legacy pick uses, because it IS the legacy pick
        restricted to the locked set — the number `allocate.atlas_cells` has
        always read out of `tilt_deg == 0 and margin >= 0.15`.
        """
        if not sols:
            return -1.0
        top = sorted(sols, key=lambda t: -t[0])[:6]
        ok = _self_mask(np.array([t[2] for t in top]), pen_ext, lat)
        for n, (m, _, q) in enumerate(top):
            if ok[n] and _clears(q, Twb, legacy_inv, boxes, off, lat, static_margin) is not None:
                return float(m)
        return -1.0

    def _pack(m, q, tilt_deg, n_sol, valid, lean):
        J = tip_jacobian(q, pen_ext=pen_ext, pen_lat=lat)
        s = sigma_min(J)
        if lean >= 0.0 and (m < gate_margin or s < gate_sigma):
            return None            # the screen and the record disagreed; the
            #                        caller falls through to the legacy pick
        return (m, s, f_max(J, press_b), n_sol, float(valid.mean()),
                float(max(_q7_window(row) for row in valid)), tilt_deg, q, lean)

    # ---- the GATED search: least lean that certifies ----------------------
    flat_m, flat_done, keep0 = -1.0, False, None
    for lean, cand in (gate_groups if gate_groups is not None
                       else _gated_groups(15.0)):
        sols, n_sol, valid = _solve_set(cand)
        if lean == 0.0:
            keep0 = (sols, valid)
        keep = [s for s in sols if s[0] >= gate_margin]
        if not keep:
            continue
        keep.sort(key=lambda t: -t[0])
        Q = np.array([s[2] for s in keep])
        # ONE BATCHED SVD AND ONE BATCHED SELF-CHECK for the whole gate-passing
        # fiber; only what survives both pays for a per-pose clearance FK.
        sig = np.linalg.svd(ik.tip_jacobian_batch(Q, pen_ext=pen_ext,
                                                  pen_lat=lat),
                            compute_uv=False)[:, -1]
        idx = np.flatnonzero(sig >= gate_sigma + _SIGMA_EPS)
        if not len(idx):
            continue
        idx = idx[_self_mask(Q[idx], pen_ext, lat)]
        for k in idx:
            m, tilt_deg, q = keep[k]
            if _clears(q, Twb, legacy_inv, boxes, off, lat, static_margin) is not None:
                out = _pack(m, q, tilt_deg, n_sol, valid, lean)
                if out is not None:
                    # a cell certified FLAT is flat-reachable by definition
                    return out + (float(m) if lean == 0.0
                                  else _flat_best(*keep0),)

    # ---- the LEGACY pick, verbatim: best margin among the best six --------
    for cand in cand_sets:
        if not cand:
            continue
        sols, n_sol, valid = _solve_set(cand)
        if not sols:
            continue
        sols.sort(key=lambda t: -t[0])
        top = sols[:6]
        ok = _self_mask(np.array([t[2] for t in top]), pen_ext, lat)
        for n, (m, tilt_deg, q) in enumerate(top):
            if not ok[n]:
                continue
            if _clears(q, Twb, legacy_inv, boxes, off, lat, static_margin) is not None:
                return _pack(m, q, tilt_deg, n_sol, valid, -1.0) + (
                    float(m) if tilt_deg == 0.0
                    else (_flat_best(*keep0) if keep0 else -1.0),)
    return None


def sweep_arm(arm_id, out_dir, grid=0.02, rmax=1.05, h_inv=H_INV_DEFAULT,
              tilt_max_deg=15.0, pen_ext=PEN_EXT, fleet=None, sheet=None,
              pen_lat=None, cone=None):
    """One arm's reachability atlas, swept and stamped. -> (N, len(COLUMNS)).

    `cone` is the LADDER of lean magnitudes the gated search may climb, in
    ascending degrees; `None` is `GATE_CONE_DEG`, the shipped 2.5 -> 15 grid,
    and `tilt_max_deg` still caps whatever is offered.  It is a parameter and
    not a constant because the cone is a PEN-HARDWARE decision — how far the
    material lets the nib lean before the line stops being a line — and asking
    "what would 20 degrees buy?" must not mean editing a module to find out.
    `_gated_groups` has always taken it; only `sweep_arm` did not pass it on.
    """
    spec = (FLEET if fleet is None else fleet)[arm_id]
    sheet = SHEET if sheet is None else sheet
    boxes = spec.static_obstacles() if hasattr(spec, "static_obstacles") else []
    Twb = spec.T_world_base(h_inv)
    Twb_inv = np.linalg.inv(Twb)
    cand_sets = _candidates(tilt_max_deg)
    gate_groups = _gated_groups(tilt_max_deg,
                                GATE_CONE_DEG if cone is None
                                else tuple(float(c) for c in cone))
    pen_lat = lat_of(pen_lat)
    # the lateral tool extends reach by up to pen_lat in every direction
    rmax = rmax + abs(pen_lat)
    bx, by = spec.xy
    rows = []
    t0 = time.time()
    for y in np.arange(0.0, sheet[1] + 1e-9, grid):
        for x in np.arange(0.0, sheet[0] + 1e-9, grid):
            if (x - bx) ** 2 + (y - by) ** 2 > rmax ** 2:
                continue
            r = solve_cell(x, y, Twb, Twb_inv, spec, cand_sets, pen_ext, boxes,
                           pen_lat=pen_lat, gate_groups=gate_groups)
            if r is not None:
                rows.append([x, y, *r[:7], *r[7], r[8], r[9]])
    arr = np.array(rows) if rows else np.zeros((0, len(COLUMNS)))
    out = Path(out_dir) / f"atlas_arm{arm_id}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, data=arr, columns=np.array(COLUMNS), arm_id=arm_id,
                        mount=spec.mount, base=Twb, grid=grid, h_inv=h_inv,
                        tilt_max_deg=tilt_max_deg, pen_ext=pen_ext,
                        pen_lat=pen_lat, model=model_signature(),
                        cone=np.array(GATE_CONE_DEG if cone is None
                                      else [float(c) for c in cone], float))
    go = strict_go(arr)
    flat = int((go & (arr[:, LEANCOL] == 0.0)).sum()) if len(arr) else 0
    print(f"arm {arm_id} ({spec.name}): {len(arr)} reachable, "
          f"{int(go.sum())} strict-GO ({flat} flat, "
          f"{int(go.sum()) - flat} leaning), tilt<={tilt_max_deg:.0f}deg, "
          f"{time.time() - t0:.0f}s")
    return arr


def min_lean(arr):
    """Per row, the least lean (deg) at which the cell certifies; -1 if none.

    The column the maps and the drawing planner read to answer "does this cell
    need the pen leaned, and by how much" — the question `solve_cell` now
    answers per cell and nothing could ask before.
    """
    if not len(arr):
        return np.zeros(0)
    return arr[:, LEANCOL]


def strict_go(arr):
    if not len(arr):
        return np.zeros(0, bool)
    return (arr[:, 2] >= GATE_MARGIN) & (arr[:, 3] >= GATE_SIGMA)


def load(out_dir, arm_id):
    d = np.load(Path(out_dir) / f"atlas_arm{arm_id}.npz")
    return d["data"], d
