#!/usr/bin/env python3
"""Does the stroke planner ever REALLY switch IK branches mid-stroke?

THE QUESTION.  `planner.build_lattice` carries a BRANCH axis — the analytic
solver's up-to-four solutions at each (arc length, q7) — and `planner.plan`'s
ladder DP is free to change branch index from step to step.  Reading a planned
path one sees the index move, which looks like the arm swapping IK branch
half way down a line.  This script measures whether any of that is real.

WHAT A "BRANCH" ACTUALLY IS HERE, AND WHY THE INDEX LIES.  The He/Liu solver
enumerates four solutions in a FIXED order, `q_all[2*i6 + i1]`, where

    i6 = 0  iff  (V6H x V62).Z6 <= 0        (the q6 = pi - Theta6 - Phi6 root)
    i1 = 1  iff  q2 < 0                     (the q2 = -acos(...) shoulder root)

(franka_ik_He.hpp; `franka_IK_EE_CC` computes exactly these two predicates off
the seed configuration to stay on one branch).  That pair IS a stable case
label and `case_id` below reproduces it bit for bit.  What the lattice stores
is NOT that label: `ik.solve_batch` COMPACTS the surviving solutions to the
front of the branch axis, so when a case dies at some arc length the ones
behind it shuffle down a slot.  Slot churn is therefore expected and means
nothing; only the case label is worth counting.

THE THREE MEASUREMENTS, from weakest to strongest:

  DENSE PATH   Every shipped plan's joint trajectory comes out of
               `pwl.chase_cc` -> `ik.solve_cc`, which is case-consistent by
               construction.  Its case label can still flip, and does — when
               the path crosses the locus where the two roots of the flipped
               predicate MERGE.  So each flip is classified by the joint step
               that carries it: an ALIAS (the configuration is continuous
               across it) or a TRUE SWITCH (a jump).
  LATTICE      The same census on the ladder-DP path and on the band DP's
               dense path, plus the slot churn each of them shows.
  EDGE CENSUS  The strongest form, and the one that does not depend on which
               path was chosen: over EVERY admissible edge of the gated
               lattice, how many join two different case labels, and how far
               apart in joint space are their endpoints?  A genuine
               branch-to-branch transition would be a cross-case edge whose
               endpoints are far apart; a fold crossing is a cross-case edge
               whose endpoints coincide.

AND THE STRUCTURAL COROLLARY, which the same run checks empirically: the
ladder DP's edge test (grid neighbour + ||dq||_inf <= jump) is a SUBSET of the
relation `pwl.sheet_fields` builds its connected components from, so no DP path
can leave the sheet it starts on.  Planning each sheet separately and keeping
the best is then not an approximation of the joint lattice — it is the same
optimum.  `--sheets` measures it rather than asserting it.

    ARIS_RIG=proposed ARIS_TOOL=lateral python3 scripts/branch_switch_stat.py \
        --program out/csail_program_h094_v14.json --fuzz 200 \
        --out out/branch_switch_v14.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from aris_sixarm import ik, metrics, planner, pwl, stroke_api  # noqa: E402
from aris_sixarm.fleet import FLEET  # noqa: E402
from aris_sixarm.frames import FR3_MIN, FR3_MAX  # noqa: E402

# ---------------------------------------------------------------------------
# the solver's own case label
# ---------------------------------------------------------------------------
_D1, _D3, _D5, _A4 = 0.3330, 0.3160, 0.3840, 0.0825


def _cc_chain(q):
    """`franka_IK_EE_CC`'s own As_a/Ts_a chain.  q (N,7) -> Ts (7,N,4,4)."""
    q = np.asarray(q, float).reshape(-1, 7)
    N = len(q)
    c, s = np.cos(q), np.sin(q)
    A = np.zeros((7, N, 4, 4))
    A[:, :, 3, 3] = 1.0
    A[0, :, 0, 0], A[0, :, 0, 1] = c[:, 0], -s[:, 0]          # O1
    A[0, :, 1, 0], A[0, :, 1, 1] = s[:, 0], c[:, 0]
    A[0, :, 2, 2], A[0, :, 2, 3] = 1.0, _D1
    A[1, :, 0, 0], A[1, :, 0, 1] = c[:, 1], -s[:, 1]          # O2
    A[1, :, 1, 2] = 1.0
    A[1, :, 2, 0], A[1, :, 2, 1] = -s[:, 1], -c[:, 1]
    A[2, :, 0, 0], A[2, :, 0, 1] = c[:, 2], -s[:, 2]          # O3
    A[2, :, 1, 2], A[2, :, 1, 3] = -1.0, -_D3
    A[2, :, 2, 0], A[2, :, 2, 1] = s[:, 2], c[:, 2]
    A[3, :, 0, 0], A[3, :, 0, 1], A[3, :, 0, 3] = c[:, 3], -s[:, 3], _A4   # O4
    A[3, :, 1, 2] = -1.0
    A[3, :, 2, 0], A[3, :, 2, 1] = s[:, 3], c[:, 3]
    A[4, :, 0, 0], A[4, :, 0, 3] = 1.0, -_A4                  # H
    A[4, :, 1, 1] = 1.0
    A[4, :, 2, 2] = 1.0
    A[5, :, 0, 0], A[5, :, 0, 1] = c[:, 4], -s[:, 4]          # O5
    A[5, :, 1, 2], A[5, :, 1, 3] = 1.0, _D5
    A[5, :, 2, 0], A[5, :, 2, 1] = -s[:, 4], -c[:, 4]
    A[6, :, 0, 0], A[6, :, 0, 1] = c[:, 5], -s[:, 5]          # O6
    A[6, :, 1, 2] = -1.0
    A[6, :, 2, 0], A[6, :, 2, 1] = s[:, 5], c[:, 5]
    Ts = np.empty((7, N, 4, 4))
    Ts[0] = A[0]
    for j in range(1, 7):
        Ts[j] = Ts[j - 1] @ A[j]
    return Ts


def wrist_triple(q):
    """(V6H x V62).Z6 — the signed quantity `is_case6_0` tests. -> (N,)"""
    Ts = _cc_chain(q)
    V62 = Ts[1][:, :3, 3] - Ts[6][:, :3, 3]
    V6H = Ts[4][:, :3, 3] - Ts[6][:, :3, 3]
    return np.einsum("ni,ni->n", np.cross(V6H, V62), Ts[6][:, :3, 2])


def case_id(q):
    """The solver's stable case label of each configuration -> (N,) int 0..3.

    Verified against the raw `_franka_ik.solve_ik` slot a configuration comes
    back in (`--selftest`): exact on every configuration the solver claims.
    """
    q = np.asarray(q, float).reshape(-1, 7)
    if not len(q):
        return np.zeros(0, int)
    i6 = (wrist_triple(q) > 0).astype(int)      # is_case6_0 == (triple <= 0)
    return 2 * i6 + (q[:, 1] < 0).astype(int)


def selftest(n=3000, seed=0):
    """case_id vs the raw solver's own slot index. -> (checked, matched)."""
    from aris_sixarm.frames import fk
    rng = np.random.default_rng(seed)
    checked = matched = 0
    for _ in range(n):
        q = rng.uniform(FR3_MIN + 0.05, FR3_MAX - 0.05)
        T, _ = fk(q)
        sols = ik._IK.solve_ik(T.flatten(order="F"), float(q[6]), q)
        hit = [k for k, s in enumerate(sols)
               if np.all(np.isfinite(s)) and np.max(np.abs(np.asarray(s) - q)) < 1e-7]
        if len(hit) != 1:
            continue
        checked += 1
        matched += int(case_id(q[None])[0] == hit[0])
    return checked, matched


# ---------------------------------------------------------------------------
# 1. the dense path: flips, and whether the joints jump across them
# ---------------------------------------------------------------------------
def dense_flips(qs, sigmas=None, margins=None, jump=planner.JUMP_THRESH,
                local_mult=10.0):
    """Case-label census along one dense joint path. -> dict.

    A flip is a TRUE SWITCH if the joint step carrying it exceeds the
    continuity budget every certified plan is held to (`jump`), and an ALIAS
    otherwise.  `jump` is a generous bar for a plan that has already been
    certified under it, so a second, LOCAL test is reported alongside: a step
    more than `local_mult` times the segment's own median step is anomalous
    whatever its absolute size.
    """
    qs = np.asarray(qs, float)
    n = len(qs)
    out = dict(n=int(n), n_flips=0, labels=[], true_switch=0, alias=0,
               local_anomaly=0, flip_dq=[], flip_sigma=[], flip_margin=[],
               max_step=0.0, median_step=0.0, labels_seen=[])
    if n < 2:
        return out
    lab = case_id(qs)
    step = np.max(np.abs(np.diff(qs, axis=0)), axis=1)
    out["max_step"] = float(step.max())
    out["median_step"] = float(np.median(step))
    out["labels_seen"] = sorted(int(x) for x in np.unique(lab))
    idx = np.flatnonzero(lab[1:] != lab[:-1])
    out["n_flips"] = int(len(idx))
    if not len(idx):
        return out
    thr = local_mult * max(float(np.median(step)), 1e-12)
    for i in idx:
        d = float(step[i])
        out["flip_dq"].append(d)
        out["labels"].append([int(lab[i]), int(lab[i + 1])])
        if sigmas is not None:
            out["flip_sigma"].append(float(min(sigmas[i], sigmas[i + 1])))
        if margins is not None:
            out["flip_margin"].append(float(min(margins[i], margins[i + 1])))
        if d > jump:
            out["true_switch"] += 1
        else:
            out["alias"] += 1
        out["local_anomaly"] += int(d > thr)
    return out


# ---------------------------------------------------------------------------
# 2. the gated lattice: every admissible edge, and which of them cross a case
# ---------------------------------------------------------------------------
def cross_reproducible(lat, edges, jump=planner.JUMP_THRESH):
    """Would the CERTIFICATION chase reproduce these cross-case edges? -> dict.

    Every shipped joint path is walked by `pwl.chase_cc` -> `ik.solve_cc`,
    which reads the two case predicates off the previous sample and re-solves
    on THAT case.  So a cross-case edge the search is entitled to propose is
    only real if `solve_cc`, seeded at the edge's source, lands on the edge's
    target.  `edges` is [(i, j_src, kb_src, j_dst, kb_dst)].
    """
    poses = pwl.pen_down_poses(lat["pts"], np.linalg.inv(lat["Twb"]),
                               lat["pen_ext"], phi=float(lat.get("phi", 0.0)),
                               pen_lat=float(lat.get("pen_lat", 0.0)),
                               tilt=lat.get("tilt"))
    q7s, Q = lat["q7s"], lat["Q"]
    n_ok = n_none = n_other = 0
    dists = []
    for i, ja, ka, jb, kb in edges:
        qa, qb = Q[i, ja, ka], Q[i + 1, jb, kb]
        q = ik.solve_cc(poses[i + 1], float(q7s[jb]), qa)
        if q is None:
            n_none += 1
            continue
        d = float(np.max(np.abs(q - qb)))
        dists.append(d)
        if d <= 1e-9:
            n_ok += 1
        else:
            n_other += 1
    return dict(n=len(edges), reproduced=int(n_ok), no_solution=int(n_none),
                other_branch=int(n_other),
                worst_miss=float(max(dists)) if dists else 0.0)


def edge_census(lat, margin_gate, sigma_gate, jump=planner.JUMP_THRESH,
                probe=False):
    """Cross-case census over the ladder DP's own edge set. -> dict.

    Edges are the DP's: (i, j, kb) -> (i+1, j+dj, kb'), dj in {-1, 0, +1},
    both endpoints past the gates, ||dq||_inf <= jump.  For every edge whose
    two endpoints carry DIFFERENT case labels we record the joint distance:
    a genuine branch-to-branch transition is a cross-case edge whose endpoints
    are FAR APART, a fold crossing is one where they have merged.

    With `probe`, each cross-case edge is additionally handed to
    `cross_reproducible` — the question of whether the certification chase
    could walk it at all.
    """
    V, Q, S, M = lat["valid"], lat["Q"], lat["sigma"], lat["margin"]
    Ns, Nq, Nb = V.shape
    ok = V & (np.nan_to_num(M, nan=-1.0) >= margin_gate) \
        & (np.nan_to_num(S, nan=-1.0) >= sigma_gate)
    Qf = np.where(ok[..., None], Q, 0.0)
    lab = np.full((Ns, Nq, Nb), -1, int)
    if ok.any():
        lab[ok] = case_id(Q[ok])
    n_edge = n_cross = 0
    cross_d, cross_sig, cross_mar = [], [], []
    same_d, cross_ix = [], []
    for dj in (-1, 0, 1):
        lo, hi = max(0, -dj), min(Nq, Nq - dj)
        if lo >= hi:
            continue
        a = (slice(0, Ns - 1), slice(lo, hi))
        b = (slice(1, Ns), slice(lo + dj, hi + dj))
        d = np.max(np.abs(Qf[a][:, :, :, None, :] - Qf[b][:, :, None, :, :]),
                   axis=-1)
        good = ok[a][:, :, :, None] & ok[b][:, :, None, :] & (d <= jump)
        if not good.any():
            continue
        la = np.broadcast_to(lab[a][:, :, :, None], good.shape)
        lb = np.broadcast_to(lab[b][:, :, None, :], good.shape)
        sa = np.broadcast_to(S[a][:, :, :, None], good.shape)
        sb = np.broadcast_to(S[b][:, :, None, :], good.shape)
        ma = np.broadcast_to(M[a][:, :, :, None], good.shape)
        mb = np.broadcast_to(M[b][:, :, None, :], good.shape)
        cross = good & (la != lb)
        n_edge += int(good.sum())
        n_cross += int(cross.sum())
        if cross.any():
            cross_d.append(d[cross])
            cross_sig.append(np.minimum(sa[cross], sb[cross]))
            cross_mar.append(np.minimum(ma[cross], mb[cross]))
            if probe:
                ii, jj, ka, kb = np.nonzero(cross)
                cross_ix += [(int(i), int(lo + j), int(x), int(lo + dj + j),
                              int(y)) for i, j, x, y in zip(ii, jj, ka, kb)]
        same = good & (la == lb)
        if same.any():
            same_d.append(d[same])
    cd = np.concatenate(cross_d) if cross_d else np.zeros(0)
    cs = np.concatenate(cross_sig) if cross_sig else np.zeros(0)
    cm = np.concatenate(cross_mar) if cross_mar else np.zeros(0)
    sd = np.concatenate(same_d) if same_d else np.zeros(0)
    chase = cross_reproducible(lat, cross_ix, jump) if (probe and cross_ix) \
        else None
    return dict(
        chase=chase,
        n_nodes=int(ok.sum()), n_edges=int(n_edge), n_cross=int(n_cross),
        cross_dq_max=float(cd.max()) if len(cd) else 0.0,
        cross_dq_p50=float(np.median(cd)) if len(cd) else 0.0,
        cross_dq_p99=float(np.quantile(cd, 0.99)) if len(cd) else 0.0,
        cross_sigma_max=float(cs.max()) if len(cs) else 0.0,
        cross_margin_max=float(cm.max()) if len(cm) else 0.0,
        same_dq_p99=float(np.quantile(sd, 0.99)) if len(sd) else 0.0,
        # the number that decides the question: a cross-case edge that is BOTH
        # far apart in joint space and comfortably inside the gates would be a
        # real branch-to-branch transition the gates permit.
        n_cross_far=int((cd > 0.10).sum()) if len(cd) else 0,
        n_cross_far_comfy=int(((cd > 0.10) & (cs >= sigma_gate + 0.04)).sum())
        if len(cd) else 0,
        cross_dq_hist=[int(x) for x in np.histogram(
            cd, bins=[0, 1e-6, 1e-4, 1e-3, 1e-2, 0.05, 0.10, 0.20, 0.35])[0]]
        if len(cd) else [0] * 8)


def sheet_label_mix(lat, sheets):
    """How many sheets carry more than one case label. -> dict.

    If sheets are label-mixed, the continuity relation has already merged the
    labels the DP's branch axis is supposed to be choosing between, which is
    the whole of Pete's claim in one number.
    """
    Q = lat["Q"]
    n_mixed, sizes, per = 0, [], []
    for sh in sheets:
        sel = sh["sel"]
        if not sel.any():
            per.append(0)
            continue
        labs = sorted(int(x) for x in np.unique(case_id(Q[sel])))
        per.append(len(labs))
        sizes.append(int(sh["nodes"]))
        n_mixed += int(len(labs) > 1)
    # cells one sheet occupies with MORE THAN ONE node: what the 2-D
    # representative (`sheet_fields` keeps the highest-sigma branch) throws away
    dup = 0
    tot = 0
    for sh in sheets:
        k = sh["sel"].sum(axis=2)
        dup += int((k > 1).sum())
        tot += int((k > 0).sum())
    return dict(n_sheets=len(sheets), n_mixed=int(n_mixed),
                labels_per_sheet=per, dup_cells=int(dup), mask_cells=int(tot))


# ---------------------------------------------------------------------------
# 3. per-sheet planning vs the joint lattice
# ---------------------------------------------------------------------------
def _restrict(lat, sel):
    out = dict(lat)
    out["valid"] = lat["valid"] & sel
    return out


def sheet_vs_joint(lat, sheets, objective="maximin_sigma"):
    """`planner.plan` on the whole lattice vs the best single sheet. -> dict."""
    joint = planner.plan(lat, objective=objective)
    best = dict(cut_index=-1, bottleneck=-np.inf, ok=False, sheet=-1)
    for sh in sheets:
        r = planner.plan(_restrict(lat, sh["sel"]), objective=objective)
        key = (int(r.get("cut_index", -1)), float(r.get("bottleneck", -np.inf)))
        cur = (int(best["cut_index"]), float(best["bottleneck"]))
        if key > cur:
            best = dict(cut_index=key[0], bottleneck=key[1],
                        ok=bool(r.get("ok", False)), sheet=int(sh["id"]))
    jb = float(joint.get("bottleneck", -np.inf))
    return dict(joint_ok=bool(joint.get("ok", False)),
                joint_cut=int(joint.get("cut_index", -1)),
                joint_bottleneck=jb,
                sheet_ok=bool(best["ok"]), sheet_cut=int(best["cut_index"]),
                sheet_bottleneck=float(best["bottleneck"]),
                sheet_id=int(best["sheet"]),
                extent_loss=int(joint.get("cut_index", -1)) - int(best["cut_index"]),
                sigma_loss=(jb - float(best["bottleneck"]))
                if np.isfinite(jb) and np.isfinite(best["bottleneck"]) else 0.0,
                joint_path=joint.get("path"))


def path_slot_churn(path, Q):
    """Slot churn vs case churn on a lattice DP path. -> dict."""
    if not path:
        return dict(n=0, slot_changes=0, case_changes=0)
    kb = np.array([p[-1] for p in path])
    qs = np.array([Q[i][p] for i, p in enumerate(path)])
    lab = case_id(qs)
    step = np.max(np.abs(np.diff(qs, axis=0)), axis=1) if len(qs) > 1 \
        else np.zeros(0)
    flip = np.flatnonzero(lab[1:] != lab[:-1]) if len(lab) > 1 else np.zeros(0, int)
    return dict(n=int(len(path)), slot_changes=int((np.diff(kb) != 0).sum()),
                case_changes=int(len(flip)),
                case_flip_dq=[float(step[i]) for i in flip],
                max_step=float(step.max()) if len(step) else 0.0)


# ---------------------------------------------------------------------------
# 4. one stroke, end to end
# ---------------------------------------------------------------------------
def analyse_stroke(pts, spec, opts, strict=(metrics.GATE_MARGIN, metrics.GATE_SIGMA),
                   perm=(pwl.MARGIN_GATE, pwl.SIGMA_GATE), do_sheets=True):
    """Plan one stroke and run every census on it. -> dict (never raises)."""
    rec = dict(arm=int(getattr(spec, "arm_id", -1)),
               length=float(stroke_api.polyline_length(pts)))
    t0 = time.perf_counter()
    o = dict(opts)
    o["keep_debug"] = True
    try:
        r = stroke_api.plan_stroke(np.asarray(pts, float), spec, o)
    except Exception as exc:                       # pragma: no cover
        rec.update(status="error", error=str(exc))
        return rec
    rec["status"] = r.get("status")
    rec["t_plan"] = time.perf_counter() - t0
    if r.get("status") == "ok":
        rec["dense"] = dense_flips(r["qs"], r.get("sigmas"), r.get("margins"))
        rec.update(sheet=int(r.get("sheet", -1)), n_sheets=int(r.get("n_sheets", 0)),
                   min_sigma=float(r.get("min_sigma", 0.0)),
                   min_margin=float(r.get("min_margin", 0.0)),
                   n_dense=int(r.get("n_dense", 0)),
                   phi=float(np.mean(np.atleast_1d(r.get("phi", 0.0)))),
                   lean_deg=float(r.get("lean_deg", 0.0) or 0.0))
    lat = r.get("lat")
    if lat is None:
        return rec
    try:
        rec["edges_strict"] = edge_census(lat, strict[0], strict[1])
        rec["edges_perm"] = edge_census(lat, perm[0], perm[1], probe=True)
        rec["edges_raw"] = edge_census(lat, planner.HARD_MARGIN, planner.HARD_SIGMA)
        sheets = pwl.sheet_fields(lat)
        rec["mix"] = sheet_label_mix(lat, sheets)
        if do_sheets:
            sj = sheet_vs_joint(lat, sheets)
            jp = sj.pop("joint_path", None)
            rec["sheet_vs_joint"] = sj
            rec["ladder_path"] = path_slot_churn(jp, lat["Q"]) if jp else None
        band = r.get("pwl")
        if band is not None and band.get("ok"):
            sh = r["sheet_obj"]
            js = band["dense_idx"]
            ii = np.arange(len(js))
            rec["band_path"] = dict(
                slot_changes=int((np.diff(sh["branch"][ii, js]) != 0).sum()),
                **{k: v for k, v in dense_flips(sh["Q"][ii, js]).items()
                   if k in ("n_flips", "true_switch", "alias", "max_step",
                            "labels_seen")})
    except Exception as exc:                       # pragma: no cover
        rec["census_error"] = f"{type(exc).__name__}: {exc}"
    return rec


# ---------------------------------------------------------------------------
# 5. inputs
# ---------------------------------------------------------------------------
def program_segments(path):
    """(arm id, world polyline, cone deg) per shipped segment."""
    prog = json.load(open(path))
    out = []
    for ph in prog.get("phases", []):
        for arm, segs in ph.get("arms", {}).items():
            for s in segs:
                out.append((int(arm), np.asarray(s["pts"], float),
                            float(s.get("tilt_cone_deg", 0.0) or 0.0), s))
    return out, prog


def fuzz_strokes(n, seed, arms):
    """Random strokes from the fuzz campaign's own generators."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import fuzz_planner as fz
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        arm = int(rng.choice(arms))
        spec = FLEET[arm]
        gen = str(rng.choice(fz.GENS))
        cls = str(rng.choice(("inside", "inside", "inside", "straddle")))
        try:
            p, meta = fz.make_stroke(rng, spec, gen, cls)
        except Exception:
            continue
        if len(np.asarray(p, float)) < 2:
            continue
        if stroke_api.polyline_length(p) < 0.05:
            continue
        out.append((arm, np.asarray(p, float), meta))
    return out


def walk_npz(path):
    """Case-label census over the SHIPPED executed joint trajectories.

    `seg_<arm> >= 0` marks the frames the arm is drawing on; transits are
    reported separately because they are not this planner's output.
    """
    z = np.load(path, allow_pickle=True)
    arms = [int(a) for a in z["arms"]]
    out = dict(arms={}, ink=dict(n=0, flips=0, true=0, alias=0),
               transit=dict(n=0, flips=0, true=0, alias=0))
    for a in arms:
        qk, sk = f"q_{a}", f"seg_{a}"
        if qk not in z.files or sk not in z.files:
            continue
        q, seg = np.asarray(z[qk], float), np.asarray(z[sk])
        lab = case_id(q)
        step = np.max(np.abs(np.diff(q, axis=0)), axis=1)
        flip = lab[1:] != lab[:-1]
        ink = (seg[:-1] >= 0) & (seg[1:] >= 0) & (seg[:-1] == seg[1:])
        rec = {}
        for name, m in (("ink", ink), ("transit", ~ink)):
            f = flip & m
            rec[name] = dict(n=int(m.sum()), flips=int(f.sum()),
                             true=int((f & (step > planner.JUMP_THRESH)).sum()),
                             alias=int((f & (step <= planner.JUMP_THRESH)).sum()),
                             max_flip_dq=float(step[f].max()) if f.any() else 0.0)
            out[name]["n"] += rec[name]["n"]
            out[name]["flips"] += rec[name]["flips"]
            out[name]["true"] += rec[name]["true"]
            out[name]["alias"] += rec[name]["alias"]
        out["arms"][str(a)] = rec
    return out


# ---------------------------------------------------------------------------
def summarise(recs, name):
    ok = [r for r in recs if r.get("status") == "ok"]
    lines = [f"--- {name}: {len(recs)} strokes, {len(ok)} planned ok ---"]
    if not ok:
        return lines
    d = [r["dense"] for r in ok if "dense" in r]
    n_samp = sum(x["n"] for x in d)
    n_flip = sum(x["n_flips"] for x in d)
    n_true = sum(x["true_switch"] for x in d)
    n_alias = sum(x["alias"] for x in d)
    n_loc = sum(x["local_anomaly"] for x in d)
    fdq = [q for x in d for q in x["flip_dq"]]
    fsg = [q for x in d for q in x["flip_sigma"]]
    multi = sum(1 for x in d if len(x["labels_seen"]) > 1)
    lines.append(f"  dense samples {n_samp}, case-label changes {n_flip} "
                 f"(on {multi}/{len(d)} strokes); TRUE switches {n_true}, "
                 f"aliases {n_alias}, local anomalies {n_loc}")
    if fdq:
        lines.append(f"  ||dq||_inf across a flip: max {max(fdq):.2e} rad, "
                     f"median {float(np.median(fdq)):.2e} rad "
                     f"(continuity budget {planner.JUMP_THRESH})")
        lines.append(f"  sigma_min at the flips: max {max(fsg):.4f}, "
                     f"min {min(fsg):.4f}")
    for key, gate in (("edges_raw", "0.15/0.08 lattice"),
                      ("edges_perm", "0.15/0.10 band"),
                      ("edges_strict", "0.30/0.14 strict")):
        e = [r[key] for r in recs if key in r]
        if not e:
            continue
        ne = sum(x["n_edges"] for x in e)
        nc = sum(x["n_cross"] for x in e)
        far = sum(x["n_cross_far"] for x in e)
        comfy = sum(x["n_cross_far_comfy"] for x in e)
        dmax = max([x["cross_dq_max"] for x in e] or [0.0])
        smax = max([x["cross_sigma_max"] for x in e] or [0.0])
        lines.append(f"  edges @ {gate}: {ne} admissible, {nc} cross-case "
                     f"({100.0 * nc / max(ne, 1):.3f} %), of those "
                     f"{far} with ||dq||>0.10 rad, {comfy} of those also "
                     f"sigma-comfortable; worst cross-case ||dq|| {dmax:.3f} rad, "
                     f"best cross-case sigma {smax:.4f}")
        ch = [x["chase"] for x in e if x.get("chase")]
        if ch:
            lines.append(f"    of those cross-case edges, the case-consistent "
                         f"chase reproduces {sum(x['reproduced'] for x in ch)}/"
                         f"{sum(x['n'] for x in ch)}; it lands on another "
                         f"configuration for {sum(x['other_branch'] for x in ch)} "
                         f"and refuses {sum(x['no_solution'] for x in ch)}")
    m = [r["mix"] for r in recs if "mix" in r]
    if m:
        lines.append(f"  sheets: {sum(x['n_sheets'] for x in m)} total, "
                     f"{sum(x['n_mixed'] for x in m)} carry >1 case label; "
                     f"cells with 2+ nodes of one sheet: "
                     f"{sum(x['dup_cells'] for x in m)} of "
                     f"{sum(x['mask_cells'] for x in m)}")
    sv = [r["sheet_vs_joint"] for r in recs if "sheet_vs_joint" in r]
    if sv:
        loss = [x["extent_loss"] for x in sv]
        sig = [x["sigma_loss"] for x in sv]
        lines.append(f"  per-sheet vs joint ladder DP on {len(sv)} lattices: "
                     f"extent loss max {max(loss)} steps "
                     f"({sum(1 for x in loss if x > 0)} strokes lose any), "
                     f"bottleneck-sigma loss max {max(sig):.2e}")
    lp = [r["ladder_path"] for r in recs if r.get("ladder_path")]
    if lp:
        lines.append(f"  ladder-DP path: {sum(x['slot_changes'] for x in lp)} "
                     f"branch-SLOT changes vs "
                     f"{sum(x['case_changes'] for x in lp)} case-label changes "
                     f"over {sum(x['n'] for x in lp)} steps")
    bp = [r["band_path"] for r in recs if r.get("band_path")]
    if bp:
        lines.append(f"  band-DP path: {sum(x['slot_changes'] for x in bp)} "
                     f"slot changes vs {sum(x['n_flips'] for x in bp)} "
                     f"case-label changes, {sum(x['true_switch'] for x in bp)} true")
    sh = [r for r in ok if "sheet" in r]
    if sh:
        nz = sum(1 for r in sh if int(r["sheet"]) != 0)
        lines.append(f"  chosen sheet != 0 on {nz}/{len(sh)} strokes "
                     f"(mean {np.mean([r['n_sheets'] for r in sh]):.1f} sheets each)")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--program", default=None, help="shipped program JSON")
    ap.add_argument("--npz", default=None, help="shipped schedule NPZ")
    ap.add_argument("--fuzz", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260831)
    ap.add_argument("--tilt-max-deg", type=float, default=15.0)
    ap.add_argument("--objective", default=pwl.OBJECTIVE, choices=pwl.OBJECTIVES)
    ap.add_argument("--no-sheets", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="out/branch_switch.json")
    a = ap.parse_args(argv)

    report = dict(argv=vars(a))
    if a.selftest:
        c, m = selftest()
        print(f"case_id selftest: {m}/{c} configurations match the raw solver's "
              f"own slot")
        report["selftest"] = dict(checked=c, matched=m)

    opts = dict(objective=a.objective, tilt_max_deg=float(a.tilt_max_deg))

    if a.program:
        segs, prog = program_segments(a.program)
        if a.limit:
            segs = segs[:a.limit]
        recs = []
        for k, (arm, pts, cone, meta) in enumerate(segs):
            r = analyse_stroke(pts, FLEET[arm], dict(opts, tilt_max_deg=cone or
                                                     a.tilt_max_deg),
                               do_sheets=not a.no_sheets)
            r["shipped"] = dict(stroke_id=meta["stroke_id"],
                                min_sigma=meta["min_sigma"],
                                n_dense=meta["n_dense"], arm=arm)
            recs.append(r)
            print(f"  [{k + 1}/{len(segs)}] arm {arm} L={r['length']:.3f} m "
                  f"{r['status']} flips="
                  f"{r.get('dense', {}).get('n_flips', '-')} "
                  f"true={r.get('dense', {}).get('true_switch', '-')}", flush=True)
        report["program"] = recs
        for line in summarise(recs, f"shipped program {Path(a.program).name}"):
            print(line)

    if a.npz:
        report["npz"] = walk_npz(a.npz)
        w = report["npz"]
        print(f"--- shipped trajectories {Path(a.npz).name} ---")
        for k in ("ink", "transit"):
            x = w[k]
            print(f"  {k}: {x['n']} frame steps, {x['flips']} case-label "
                  f"changes, {x['true']} true switches, {x['alias']} aliases")

    if a.fuzz:
        arms = sorted(FLEET)
        strokes = fuzz_strokes(a.fuzz, a.seed, arms)
        recs = []
        for k, (arm, pts, meta) in enumerate(strokes):
            r = analyse_stroke(pts, FLEET[arm], opts, do_sheets=not a.no_sheets)
            r["gen"] = meta
            recs.append(r)
            if (k + 1) % 10 == 0 or k == 0:
                print(f"  fuzz [{k + 1}/{len(strokes)}]", flush=True)
        report["fuzz"] = recs
        for line in summarise(recs, f"fuzz {a.fuzz} random strokes"):
            print(line)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    def _j(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        raise TypeError(type(o))

    with open(a.out, "w") as f:
        json.dump(report, f, default=_j)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
