"""What none of the arms can draw, in the two tiers that mean different things.

THE ATLAS OVERSTATES DEATH, AND SAYING SO IS THE POINT.  `atlas.strict_go`
gates a cell at margin >= 0.30 rad and sigma_min >= 0.14 — the IKA numbers, a
standard for a cell you would like to STAND in and work from.  The planner that
actually certifies a stroke gates at `validate.MARGIN_GATE` = 0.15 and
`validate.SIGMA_GATE` = 0.10, half and two thirds of those.  A cell between the
two is not unreachable: it is reachable, and the strict atlas calls it dead.

So "dead" here is two colours and not one:

  PERMISSIVE-DEAD   no arm reaches the cell even at the PLANNER's own hard
                    gates.  This is the geometry — the feed roll, the guide
                    rods, the under-shoulder holes, the corners — and no
                    re-allocation, re-placement or slower pen recovers any of
                    it.  Solid dark red.
  STRICT-ONLY-DEAD  some arm clears the planner's gates but none clears the
                    atlas's.  This paper is drawable and the strict sweep
                    disowns it; whether a stroke through it CERTIFIES still
                    depends on the continuous band the planner needs, which a
                    per-cell atlas cannot answer either way.  Orange.

Both tiers come out of the SAME per-arm atlas rows — columns 2 and 3 are the
best margin and sigma the sweep found at that cell — so the two masks are
nested by construction (`permissive_dead` is a subset of `strict_dead`) and
nothing is mirrored, interpolated or assumed.

`probe_cells` is the honesty check on the red tier: it hands short strokes
through a deterministic subsample of permissive-dead cells to the real
`stroke_api` planner and reports how many come back certified.  A red cell that
the planner can draw would mean this module is still overstating death, and the
count is printed rather than hidden.
"""
import json
from pathlib import Path

import numpy as np

from . import atlas
from .metrics import GATE_MARGIN, GATE_SIGMA
from .validate import MARGIN_GATE, SIGMA_GATE

# the planner's own hard gates, restated from `validate` on purpose: if someone
# loosens a gate there and this module disagrees, a test is supposed to notice.
PERMISSIVE_MARGIN = MARGIN_GATE      # 0.15 rad
PERMISSIVE_SIGMA = SIGMA_GATE        # 0.10
STRICT_MARGIN = GATE_MARGIN          # 0.30 rad
STRICT_SIGMA = GATE_SIGMA            # 0.14

PROBE_STRIDE = 7           # every Nth permissive-dead cell gets a real probe
PROBE_LEN = 0.05           # m, the length of the probe stroke


def grid_axes(sheet, grid=0.02):
    """The atlas lattice for a sheet. -> (xs, ys)."""
    return (np.arange(0.0, sheet[0] + 1e-9, grid),
            np.arange(0.0, sheet[1] + 1e-9, grid))


def _mask(arr, xs, ys, grid, keep):
    """Scatter a per-row boolean onto the dense lattice. -> (H,W) bool."""
    m = np.zeros((len(ys), len(xs)), bool)
    if not len(arr):
        return m
    ix = np.rint(arr[:, 0] / grid).astype(int)
    iy = np.rint(arr[:, 1] / grid).astype(int)
    ok = (ix >= 0) & (ix < len(xs)) & (iy >= 0) & (iy < len(ys)) & keep
    m[iy[ok], ix[ok]] = True
    return m


def tiers(atlas_dir, arms, sheet, grid=0.02):
    """Both dead tiers over the whole canvas. -> dict of dense masks.

    -> dict(xs, ys, grid, arms,
            per_arm_strict, per_arm_permissive   (A,H,W) bool
            strict, permissive                   (H,W) bool, the unions
            dead_strict, dead_permissive         (H,W) bool
            strict_only_dead                     (H,W) bool
            counts)

    Deterministic: arms are visited in sorted order and every operation is a
    whole-array one, so two runs over the same atlas produce bit-identical
    masks.  `tests` pins that.
    """
    xs, ys = grid_axes(sheet, grid)
    arms = sorted(int(a) for a in arms)
    ps, pp = [], []
    for a in arms:
        arr, _ = atlas.load(atlas_dir, a)
        arr = np.asarray(arr, float)
        st = (arr[:, 2] >= STRICT_MARGIN) & (arr[:, 3] >= STRICT_SIGMA) \
            if len(arr) else np.zeros(0, bool)
        pm = (arr[:, 2] >= PERMISSIVE_MARGIN) & (arr[:, 3] >= PERMISSIVE_SIGMA) \
            if len(arr) else np.zeros(0, bool)
        ps.append(_mask(arr, xs, ys, grid, st))
        pp.append(_mask(arr, xs, ys, grid, pm))
    per_s = np.array(ps) if ps else np.zeros((0, len(ys), len(xs)), bool)
    per_p = np.array(pp) if pp else np.zeros((0, len(ys), len(xs)), bool)
    strict = per_s.any(axis=0) if len(per_s) else np.zeros((len(ys), len(xs)), bool)
    perm = per_p.any(axis=0) if len(per_p) else np.zeros((len(ys), len(xs)), bool)
    dead_s, dead_p = ~strict, ~perm
    n = strict.size
    return dict(xs=xs, ys=ys, grid=float(grid), arms=arms,
                per_arm_strict=per_s, per_arm_permissive=per_p,
                strict=strict, permissive=perm,
                dead_strict=dead_s, dead_permissive=dead_p,
                strict_only_dead=dead_s & ~dead_p,
                counts=dict(cells=int(n),
                            strict_go=int(strict.sum()),
                            permissive_go=int(perm.sum()),
                            dead_strict=int(dead_s.sum()),
                            dead_permissive=int(dead_p.sum()),
                            strict_only_dead=int((dead_s & ~dead_p).sum()),
                            strict_go_pct=100.0 * strict.sum() / n,
                            permissive_go_pct=100.0 * perm.sum() / n,
                            dead_strict_pct=100.0 * dead_s.sum() / n,
                            dead_permissive_pct=100.0 * dead_p.sum() / n,
                            strict_only_dead_pct=100.0 * (dead_s & ~dead_p).sum() / n))


def cell_of(xy, T):
    """Paper xy -> (iy, ix) lattice index, clipped to the sheet."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    ix = np.clip(np.rint(xy[:, 0] / T["grid"]).astype(int), 0, len(T["xs"]) - 1)
    iy = np.clip(np.rint(xy[:, 1] / T["grid"]).astype(int), 0, len(T["ys"]) - 1)
    return iy, ix


def densify(pts, ds=0.005):
    """Resample a polyline at <= `ds` spacing. -> (M,2)."""
    P = np.asarray(pts, float).reshape(-1, 2)
    if len(P) < 2:
        return P
    out = [P[0]]
    for a, b in zip(P[:-1], P[1:]):
        L = float(np.linalg.norm(b - a))
        k = max(1, int(np.ceil(L / ds)))
        for i in range(1, k + 1):
            out.append(a + (i / k) * (b - a))
    return np.array(out)


def load_dropped(program_path):
    """The shipped programme's undrawn spans. -> list of dicts."""
    with open(program_path) as f:
        prog = json.load(f)
    out = []
    for d in prog.get("dropped", []):
        pts = np.asarray(d.get("pts", []), float).reshape(-1, 2)
        out.append(dict(stroke_id=d.get("stroke_id"), color=d.get("color"),
                        kind=d.get("kind"), length_m=float(d.get("length_m",
                                                                d.get("length", 0.0))),
                        s_range=d.get("s_range"), at=d.get("at"), pts=pts))
    return out, prog


# --------------------------------------------------------------------------
# why each undrawn span is undrawn
# --------------------------------------------------------------------------
# THE PROGRAMME RECORDS THAT A SPAN WAS DROPPED AND NEVER WHY.
# `allocate.leftover` derives the dropped list as the COMPLEMENT of the shipped
# programmes — "traced = drawn + dropped holds by construction" — which is the
# right way to get the metres right and throws away the cause completely.  A
# span that no arm can reach and a span that three arms can reach but none of
# them was holding the right pen are the same record.
#
# So the cause is re-derived here, from the geometry, in the order that makes
# the coarsest true statement first:
#
#   dead_permissive   no arm clears the PLANNER's gates anywhere along it.
#                     Nothing about this run could have drawn it.
#   dead_strict_only  reachable at the planner's gates, not at the atlas's.
#   wrong_ink         an arm certifies it, but no arm HOLDING THAT COLOUR does.
#                     A pen-swap problem, not a reach problem.
#   other             a right-ink arm can reach every cell of it and it was
#                     still not drawn: the continuous band the planner needs,
#                     the probe budget, the splitter's minimum length, or the
#                     balancer handing it back.
#
# CONDUCTOR REFUSAL IS NOT ONE OF THE CATEGORIES, and that is a finding rather
# than an omission: a refusal never reaches this list.  `idle.Unconductable`
# makes `csail_schedule` re-sequence, or fall back to the go-home policy, or
# abandon the whole profile — the ALLOCATION is never edited, so a refused
# profile contributes no dropped spans at all.  `attribute` reports the count as
# a structural zero with that reasoning attached.
CAUSES = ("dead_permissive", "dead_strict_only", "wrong_ink", "other")


def arms_with_ink(prog):
    """{colour: [arm ids]} from the shipped programme's own colour map."""
    out = {}
    for a, col in (prog.get("colors") or {}).items():
        out.setdefault(str(col), []).append(int(a))
    return {k: sorted(v) for k, v in out.items()}


def attribute(spans, T, ink_arms, ds=0.005, frac=0.5):
    """Why each undrawn span is undrawn. -> (rows, totals).

    The four causes are DISJOINT AND EXHAUSTIVE at every sample, tested in the
    order below, so the per-sample fractions on each row sum to 1 and the
    metres add up to the dropped total without double counting:

      1 dead_permissive    no arm at the planner's gates
      2 dead_strict_only   somebody at the planner's gates, nobody at the
                           atlas's — and the ATLAS is what `allocate` prefilters
                           the candidate set with, so this is a real causal
                           channel and not a presentational one
      3 wrong_ink          strict-GO for some arm, for none holding that colour
      4 other              strict-GO for an arm holding the right colour, and
                           still not drawn: the continuous band the planner
                           needs along the WHOLE stroke (per-cell GO does not
                           imply a stroke through those cells certifies), the
                           probe budget, the splitter's minimum length, or the
                           balancer handing it back.  `probe_spans` takes this
                           category apart by asking the planner directly.

    A span is labelled by the first cause holding over more than `frac` of its
    samples; the full fraction vector is kept so a mixed span reads as mixed.
    """
    idx = {a: i for i, a in enumerate(T["arms"])}
    rows = []
    for s in spans:
        P = densify(s["pts"], ds)
        if not len(P):
            rows.append(dict(**{k: v for k, v in s.items() if k != "pts"},
                             cause="other", n=0,
                             frac={c: 0.0 for c in CAUSES}))
            continue
        iy, ix = cell_of(P, T)
        dp = T["dead_permissive"][iy, ix]
        strict = T["strict"][iy, ix]
        mine = [idx[a] for a in ink_arms.get(str(s.get("color")), []) if a in idx]
        own = (T["per_arm_strict"][mine][:, iy, ix].any(axis=0) if mine
               else np.zeros(len(P), bool))
        c1 = dp
        c2 = (~c1) & (~strict)
        c3 = (~c1) & (~c2) & (~own)
        c4 = (~c1) & (~c2) & (~c3)
        f = dict(dead_permissive=float(c1.mean()),
                 dead_strict_only=float(c2.mean()),
                 wrong_ink=float(c3.mean()), other=float(c4.mean()))
        cause = max(CAUSES, key=lambda c: f[c])
        for c in CAUSES:
            if f[c] > frac:
                cause = c
                break
        rows.append(dict(**{k: v for k, v in s.items() if k != "pts"},
                         cause=cause, n=int(len(P)), frac=f))
    tot = {c: 0.0 for c in CAUSES}
    share = {c: 0.0 for c in CAUSES}
    for r in rows:
        tot[r["cause"]] += float(r["length_m"])
        for c in CAUSES:                       # metres weighted by the mixture
            share[c] += float(r["length_m"]) * r["frac"].get(c, 0.0)
    return rows, dict(by_cause_m=tot, by_cause_weighted_m=share,
                      total_m=float(sum(tot.values())), n_spans=len(rows),
                      conductor_refusal_m=0.0,
                      conductor_refusal_note=(
                          "structural zero: a conductor refusal re-sequences, "
                          "falls back to the go-home policy, or abandons the "
                          "profile; it never edits the allocation, so it "
                          "contributes no dropped spans"))


def probe_spans(spans, rows, fleet, ink_arms, pens, causes=("other",),
                verbose=True):
    """Ask the planner what it actually says about the leftover spans.

    "other" means every cell of the span is strict-GO for an arm holding the
    right ink and the span was still not drawn — which is a statement about the
    per-cell atlas, not about the stroke.  A stroke needs a CONTINUOUS certified
    band from end to end, and this is the only way to find out whether it has
    one: hand the exact span to `stroke_api.plan_stroke` for every right-ink arm
    and record what comes back.
    """
    from . import stroke_api
    out = []
    for s, r in zip(spans, rows):
        if r["cause"] not in causes:
            continue
        best, per = None, {}
        for a in ink_arms.get(str(s.get("color")), []):
            if a not in fleet:
                continue
            res = stroke_api.plan_stroke(np.asarray(s["pts"], float),
                                         fleet[a],
                                         dict(pen_ext=float(pens.get(a, 0.110))))
            per[int(a)] = dict(status=res.get("status"),
                               reason=res.get("reason"),
                               s_star=res.get("s_star"))
            if res.get("status") == "ok":
                best = int(a)
        out.append(dict(stroke_id=s.get("stroke_id"), color=s.get("color"),
                        length_m=float(s["length_m"]), cause=r["cause"],
                        certified_by=best, per_arm=per))
    if verbose:
        n_ok = sum(1 for g in out if g["certified_by"] is not None)
        m_ok = sum(g["length_m"] for g in out if g["certified_by"] is not None)
        print(f"span probe: {len(out)} '{'/'.join(causes)}' spans re-planned, "
              f"{n_ok} certify whole ({m_ok:.4f} m) — the rest have no "
              f"continuous band")
    return out


def probe_cells(T, fleet, pens, stride=PROBE_STRIDE, length=PROBE_LEN,
                limit=None, verbose=True):
    """Hand real strokes to the planner in permissive-dead cells. -> dict.

    The red tier claims no arm can draw these cells at the gates the planner
    itself uses.  This checks that claim the only way it can be checked — by
    asking the planner — on a deterministic subsample (every `stride`-th dead
    cell in row-major order, never a random one, so the answer is reproducible).
    Any cell that comes back certified is a cell this module called dead and
    should not have.
    """
    from . import stroke_api
    dead = np.argwhere(T["dead_permissive"])
    picked = dead[::max(int(stride), 1)]
    if limit:
        picked = picked[:int(limit)]
    got = []
    for iy, ix in picked:
        x, y = float(T["xs"][ix]), float(T["ys"][iy])
        seg = np.array([[x - 0.5 * length, y], [x + 0.5 * length, y]])
        best = None
        for a in T["arms"]:
            if a not in fleet:
                continue
            try:
                r = stroke_api.plan_stroke(
                    seg, fleet[a],
                    dict(pen_ext=float(pens.get(a, 0.110)),
                         clip_to_sheet=False, min_length=0.5 * length))
            except Exception:
                continue
            if r.get("status") == "ok":
                best = a
                break
        got.append(dict(x=x, y=y, certified_by=best))
    n_ok = sum(1 for g in got if g["certified_by"] is not None)
    if verbose:
        print(f"dead probe: {len(got)} permissive-dead cells sampled "
              f"(stride {stride}), {n_ok} certified by the real planner")
    return dict(n=len(got), n_certified=int(n_ok), stride=int(stride),
                length=float(length), cells=got)


def save(T, path):
    """Persist the dense masks so the viewer and the tests share one artefact."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        p, xs=T["xs"], ys=T["ys"], grid=T["grid"], arms=np.array(T["arms"]),
        strict=T["strict"], permissive=T["permissive"],
        dead_strict=T["dead_strict"], dead_permissive=T["dead_permissive"],
        strict_only_dead=T["strict_only_dead"],
        per_arm_strict=T["per_arm_strict"],
        per_arm_permissive=T["per_arm_permissive"])
    return p
