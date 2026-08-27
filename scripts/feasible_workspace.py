"""THE FEASIBLE CANVAS, SOLO: what one arm at a time can actually draw.

A canvas cell is FEASIBLE when SOME arm can draw it while the other five stand
at their certified parks.  Three layers, all of them the pipeline's own — this
script composes existing seams and invents no gate of its own:

  1. DRAW POSE   a certified strict-GO pose from the banded atlas
                 (`atlas.strict_go`, gated at `rig_final.STATIC_MARGIN`
                 against every neighbour's steel and base column).  The atlas
                 carries `atlas.model_signature`; a stale one is refused here
                 rather than trusted.

  2. HOVER       a certified lift over that cell, `writing.lifted_or_lower` —
                 the v3 full-fiber solver (8 tool yaws x the whole q7 grid x
                 every branch x the 5-height ladder), NOT the phi = 0 pin.  Its
                 last resort is "do not lift at all" (height 0), which is a
                 refusal dressed as an answer: this script counts z > 0 only.

  3. PEN-UP      `writing.enter_beats(park -> hover -> draw pose)`, which is
                 two `paper.route` calls: park to hover on the flying floor,
                 hover to ink on the contact band.  `paper.route` walks its
                 whole ladder (direct, lift, descend, retract-then-go, mid-hop,
                 cartesian traverse, fold-through-home, skirt) against the
                 static metal at `STATIC_PLAN_MARGIN`.
                 ...and then the FIVE PARKED PARTNERS, which the router does
                 not know about: `allocate.ParkProbe` at the conductor's own
                 SAFETY_M + CALIB_M = 0.08 m, over the route densified to the
                 router's own `paper.SAMPLES` per leg.  This is where the
                 pipeline puts a parked arm (commit 8c1e7b1) and this script
                 puts it in the same place.

NO PROXY.  The first cut of this budgeted a hover-plane roadmap because
per-cell routing looked unaffordable; measured, it is a couple of seconds a
cell and the whole fleet is 23 376 certified cells, so every cell is routed for
real and the map has no approximation in it to validate.
`scripts/feasible_workspace_validate.py` recomputes a stratified sample in a
cold process and reports what the straight-line proxy WOULD have got wrong,
as a diagnostic rather than as a substitute.

`--every N` sweeps every Nth certified cell for a pilot; the full map needs
`--every 1` (the default), because a skipped cell is indistinguishable from a
cell no arm can draw.

    ARIS_RIG=proposed ARIS_TOOL=lateral python3 scripts/feasible_workspace.py

Writes out/feasible_workspace.{png,json}, _raw.npz, _map.npz, and
checkpoints _ckpt.npz every ten chunks so a long sweep loses nothing.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# the rig and the tool are read at import time, so they are set before the
# package is touched — a caller who forgot the env vars gets the rig they asked
# for rather than the default one silently
os.environ.setdefault("ARIS_RIG", "proposed")
os.environ.setdefault("ARIS_TOOL", "lateral")

from aris_sixarm import allocate, atlas, layout, paper, writing   # noqa: E402
from aris_sixarm import frames                                    # noqa: E402

ATLAS_DIR = ROOT / "out" / "atlas_proposed_h0940_gated"
OUT = ROOT / "out"
GRID = 0.02
SHEET = (1.8034, 3.63064)

# cause codes, in the order a cell is decided
FEASIBLE = 0
NO_DRAW = 1        # no arm has a certified drawing pose here
NO_HOVER = 2       # some arm can draw it, none can lift off it
NO_ROUTE = 3       # some arm can draw and lift, none can fly here

CAUSE_NAME = {FEASIBLE: "feasible", NO_DRAW: "no draw pose",
              NO_HOVER: "draw ok, no hover", NO_ROUTE: "hover ok, unreachable"}


def axes(sheet=SHEET, grid=GRID):
    """The atlas's own cell centres, so a row of it indexes straight in."""
    xs = np.arange(0.0, sheet[0] + 1e-9, grid)
    ys = np.arange(0.0, sheet[1] + 1e-9, grid)
    return xs, ys


def _ij(x, y, xs, ys):
    return (int(round(float(y) / GRID)), int(round(float(x) / GRID)))


# ---------------------------------------------------------------------------
# one arm's cells, in a worker
# ---------------------------------------------------------------------------
_W = {}
SHIPPED_PITCH = 0.61


def rig(pitch=None, h=None):
    """The fleet and its parks. -> (fleet, parks, h, pitch).

    At the SHIPPED pitch this is the committed pair — `layout.FLEET_PROPOSED`
    and the baked `Q_PARK_PROPOSED` literals — and nothing is re-solved, so the
    headline number is measured against the rig in the repository and not
    against a re-derivation of it.

    At any other pitch the layout is `paired_grid` at that spacing and the
    parks are re-derived by `layout.certified_park_poses` from the SAME
    (radius, hover, bearing) recipe.  That recipe was searched at 0.61 and is
    not re-optimised here, so a comparison run answers "the shipped park recipe
    at a wider pitch", not "the best rig at a wider pitch".
    """
    h = layout.LAYOUT_PROPOSED["h"] if h is None else float(h)
    pitch = SHIPPED_PITCH if pitch is None else float(pitch)
    if abs(pitch - SHIPPED_PITCH) < 1e-9 and abs(h - 0.940) < 1e-9:
        return layout.FLEET_PROPOSED, layout.Q_PARK_PROPOSED, h, pitch
    lay = layout.paired_grid(spacing=pitch, rows=3, h=h)
    fl = layout.build_fleet(lay)
    parks = layout.certified_park_poses(fl, layout.PARK_GRID_PROPOSED)
    return layout.build_fleet(lay, q_park=parks), parks, h, pitch


def park_hovers(fleet, parks, h):
    """Where each parked pen sits on the canvas. -> {arm: (x, y)}.

    Baked as `layout.PARK_HOVER_PROPOSED` for the shipped rig; derived here for
    a comparison rig, by the same forward kinematics that put it there."""
    out = {}
    for a, q in parks.items():
        xy = paper.tip_xy(np.asarray(q, float), fleet[a],
                          pen_ext=fleet[a].pen, h_inv=h)
        out[int(a)] = (round(float(xy[0]), 3), round(float(xy[1]), 3))
    return out


def _init(atlas_dir, h, redundant=True, pitch=None, fiber=True, lean=0.0,
          tries=12):
    """Per-worker state: the fleet, the parks, the probe.  Built once."""
    writing.HOVER_LEAN_MAX_DEG = float(lean)
    fl, parks, h, _ = rig(pitch, h)
    pens = {a: fl[a].pen for a in fl}
    _W["fleet"] = fl
    _W["parks"] = parks
    _W["h"] = h
    _W["probe"] = allocate.ParkProbe(parks, fl, pens, h_inv=h)
    _W["atlas_dir"] = atlas_dir
    _W["redundant"] = bool(redundant)
    _W["fiber"] = bool(fiber)
    _W["tries"] = int(tries)


def _dense(steps, q0, n=None):
    """A beat's waypoints -> the router's own sampling density. -> (M, 7).

    `writing._beat` returns one row per HOP, and a hop is a metre-long joint
    move.  `ParkProbe.clearance` charges half the worst per-sample motion as
    its 1-Lipschitz residual, so handing it the hops themselves would refuse
    every route for being long.  Sampled at `paper.SAMPLES` per hop it is the
    same density `paper.route` certified the route at.
    """
    n = paper.SAMPLES if n is None else int(n)
    qs = [np.asarray(q0, float).reshape(7)] + [np.asarray(s[1], float).reshape(7)
                                               for s in steps]
    out = [qs[0][None, :]]
    for a, b in zip(qs[:-1], qs[1:]):
        f = np.linspace(0.0, 1.0, n)[1:, None]
        out.append(a[None, :] * (1.0 - f) + b[None, :] * f)
    return np.vstack(out)


def _certified_hovers(spec, q_draw, xy, h, heights=None):
    """EVERY certified hover over this cell, ladder order. -> [(q, z)].

    `writing.lifted_or_lower` stops at the first rung that certifies, because
    a timeline wants one hover and the highest it can hold.  A FEASIBILITY map
    wants the redundancy: a cell whose 6 cm hover cannot be flown to may have a
    3 cm one that can, and refusing it would blame the ladder's preference on
    the canvas.  Same solver, same gate, same rungs, same floor-clamp fallback
    (`writing.lifted_or_lower`'s second ladder) — it just does not stop.
    """
    heights = writing.HOVER_LADDER if heights is None else heights
    gate = writing.static_gate(spec, spec.pen, h)
    out, seen = [], set()

    def ladder(ok):
        for z in heights:
            q = writing.hover_solve(spec, q_draw, xy, z=z, h_inv=h,
                                    pen_ext=spec.pen, ok=ok)
            if q is None:
                continue
            k = np.round(q, 6).tobytes()
            if k in seen:
                continue
            seen.add(k)
            out.append((np.asarray(q, float), float(z)))

    ladder(gate)
    if not out:
        fl = float(paper.chain_static(np.asarray(q_draw, float).reshape(1, 7),
                                      spec, spec.pen, h)[0])
        if fl < paper.FRAME_FLOOR:
            ladder(writing.static_gate(spec, spec.pen, h, floor=fl))
    return out


def _fiber_hovers(spec, q_draw, xy, h, tries=None):
    """The rest of the fiber, ladder order, nearest first. -> [(q, z)].

    `_certified_hovers` walks the HEIGHTS; this walks the whole fiber at each
    of them — 8 tool yaws x the entire q7 grid x every branch, and then the
    leans the run's cone allows — which is what `writing.lifted_or_lower`'s
    depot-aware tier does when a span end is about to cost ink.

    IT IS THE SAME QUESTION THE MAP IS ASKING.  A cell whose derived hover
    cannot be flown to is not a cell the arm cannot reach; it is a cell whose
    FIRST hover cannot be flown to, and the pipeline does not accept that answer
    either.  Measured over 1 127 certified cells (`out/guard_pocket.py`), 51.7 %
    of the span ends the router calls unreachable hold a pose on their own fiber
    that it can fly to — so a map that did not look here would blame the canvas
    for a search that stopped early.
    """
    tries = writing.HOVER_DEPOT_TRIES if tries is None else int(tries)
    gate = writing.static_gate(spec, spec.pen, h)
    out = []
    for lean in [None] + writing._lean_rungs():
        for z in writing.HOVER_LADDER:
            for q in writing.hover_fiber(spec, q_draw, xy, z, gate, h,
                                         spec.pen, lean):
                out.append((np.asarray(q, float), float(z)))
                if len(out) >= tries:
                    return out
    return out


def _cell(arm, row, qcol, redundant=True):
    """The three layers for one (arm, cell). -> (code, z_hover, park_clear).

    `code` is the FIRST layer that refused, as a cause code; FEASIBLE when all
    three hold.  The draw pose is layer 1 by construction — this is only ever
    called on a strict-GO row.  A cell passes if ANY certified hover on the
    ladder — and then anywhere on the FIBER — can be flown to, which is the
    redundancy the brief asks for and the search the pipeline itself makes.
    """
    fl, parks, h = _W["fleet"], _W["parks"], _W["h"]
    probe = _W["probe"]
    spec = fl[arm]
    x, y = float(row[0]), float(row[1])
    q_draw = np.asarray(row[qcol:qcol + 7], float)
    q_park = np.asarray(parks[arm], float)

    if redundant:
        hovers = _certified_hovers(spec, q_draw, (x, y), h)
    else:
        q, z = writing.lifted_or_lower(spec, q_draw, (x, y), h_inv=h,
                                       pen_ext=spec.pen)
        hovers = [(q, z)] if z > 0.0 else []
    if not hovers:
        return NO_HOVER, 0.0, float("nan")

    best = float("-inf")
    z0 = float(hovers[0][1])
    seen = set()
    tier = 0
    while True:
        for q_hov, z in hovers:
            k = np.round(q_hov, 9).tobytes()
            if k in seen:
                continue
            seen.add(k)
            beats = writing.enter_beats(spec, q_park, q_hov, q_draw,
                                        pen_ext=spec.pen, h_inv=h)
            if beats is None:
                continue
            clear = float(probe.clearance(arm, _dense(beats["steps"], q_park)))
            best = max(best, clear)
            if clear >= probe.margin:
                return FEASIBLE, float(z), clear
        if tier or not _W.get("fiber", True):
            break
        tier = 1
        hovers = _fiber_hovers(spec, q_draw, (x, y), h, _W.get("tries", 12))
    return NO_ROUTE, z0, (float("nan") if best == float("-inf") else best)


_ROWS = {}


def _go_rows(arm):
    if arm not in _ROWS:
        arr, _ = atlas.load(_W["atlas_dir"], arm)
        _ROWS[arm] = arr[atlas.strict_go(arr)]
    return _ROWS[arm]


def _chunk(job):
    """(arm, [row indices]) -> per-row results, in a worker process."""
    arm, idx = job
    rows = _go_rows(arm)
    qcol = atlas.QCOL
    out = []
    for i in idx:
        r = rows[i]
        code, z, clear = _cell(arm, r, qcol, _W["redundant"])
        out.append((float(r[0]), float(r[1]), int(code), float(z),
                    float(clear), int(i)))
    return arm, out


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------
def sweep(arms, atlas_dir, h, workers, chunk=24, every=1, redundant=True,
          ckpt=None, pitch=None, log=print, fiber=True, lean=0.0, tries=12):
    """-> {arm: (N,6) array of (x, y, code, z_hover, park_clear, atlas_row)}.

    Chunks are STRIDED, not contiguous: an atlas row block is one band of y, and
    a band at the edge of reach is all refusals (the whole route ladder, 4 s a
    cell) while a band under the arm is all first-shape hits (0.1 s).  Handed
    contiguous blocks the pool finishes the cheap arms and then waits on one
    worker for twenty minutes.  Strided, every chunk is a fair sample of the
    canvas and the pool drains evenly.
    """
    import multiprocessing as mp

    sel = {}
    for a in arms:
        arr, meta = atlas.load(atlas_dir, a)
        ok, why = atlas.is_current(meta)
        if not ok:
            raise SystemExit(f"atlas for arm {a} is STALE: {why}\n"
                             f"  re-sweep it into {atlas_dir} before trusting it")
        n = int(atlas.strict_go(arr).sum())
        sel[a] = np.arange(0, n, max(1, int(every)))
        log(f"  arm {a}: {n} strict-GO cells "
            f"({len(sel[a])} swept, model signature current)")

    jobs = []
    for a in arms:
        idx = sel[a]
        nchunk = max(1, int(np.ceil(len(idx) / chunk)))
        for k in range(nchunk):
            jobs.append((a, [int(v) for v in idx[k::nchunk]]))
    jobs = [j for j in jobs if j[1]]
    total = sum(len(j[1]) for j in jobs)
    log(f"{total} arm-cells, {len(jobs)} chunks, {workers} workers", flush=True)

    res = {a: [] for a in arms}
    t0 = time.time()
    done = ncell = 0
    ctx = mp.get_context("fork")
    with ctx.Pool(workers, initializer=_init,
                  initargs=(str(atlas_dir), h, redundant, pitch, fiber,
                            lean, tries)) as pool:
        for arm, out in pool.imap_unordered(_chunk, jobs, chunksize=1):
            res[arm].extend(out)
            done += 1
            ncell += len(out)
            el = time.time() - t0
            log(f"  {done}/{len(jobs)} chunks  {ncell}/{total} cells  "
                f"{el:.0f}s  eta {el / done * (len(jobs) - done):.0f}s",
                flush=True)
            # NOTHING IS LOST: a partial grid on disk after every chunk
            if ckpt and (done % 10 == 0 or done == len(jobs)):
                np.savez_compressed(
                    ckpt, **{f"arm{k}": (np.array(v) if v else np.zeros((0, 6)))
                             for k, v in res.items()})
    return {a: (np.array(v) if v else np.zeros((0, 6))) for a, v in res.items()}


# ---------------------------------------------------------------------------
# the map
# ---------------------------------------------------------------------------
def compose(per_arm, arms, sheet=SHEET, grid=GRID):
    """Per-arm cell verdicts -> the canvas rasters.

    -> dict(n_arms, cause, draw_any, hover_any, per_arm_mask) where `cause` is
    the union verdict per cell: a cell is FEASIBLE if ANY arm carried it all
    the way, and otherwise blamed on the FURTHEST layer any arm reached.
    """
    xs, ys = axes(sheet, grid)
    H, W = len(ys), len(xs)
    n_arms = np.zeros((H, W), int)
    draw_any = np.zeros((H, W), bool)
    hover_any = np.zeros((H, W), bool)
    masks = {}
    for a in arms:
        m = np.zeros((H, W), bool)
        d = per_arm[a]
        for row in d:
            x, y, code = float(row[0]), float(row[1]), int(row[2])
            i, j = _ij(x, y, xs, ys)
            draw_any[i, j] = True
            if code != NO_HOVER:
                hover_any[i, j] = True
            if code == FEASIBLE:
                m[i, j] = True
                n_arms[i, j] += 1
        masks[a] = m
    cause = np.full((H, W), NO_DRAW, int)
    cause[hover_any] = NO_ROUTE
    cause[draw_any & ~hover_any] = NO_HOVER
    cause[n_arms > 0] = FEASIBLE
    return dict(n_arms=n_arms, cause=cause, draw_any=draw_any,
                hover_any=hover_any, per_arm=masks, xs=xs, ys=ys)


def _runs(row):
    """Longest run of True in a bool row. -> (start, length)."""
    if not row.any():
        return 0, 0
    d = np.diff(np.concatenate(([0], row.view(np.int8), [0])))
    s = np.where(d == 1)[0]
    e = np.where(d == -1)[0]
    k = int(np.argmax(e - s))
    return int(s[k]), int(e[k] - s[k])


def all_rects(mask):
    """Every maximal all-True rectangle, by row band. -> [(i0, j0, h, w)].

    O(H^2 * W) and the grid is 182 x 91, so it is exact and instant.  The
    histogram scan is asymptotically better and only ever finds the largest by
    AREA; the brief wants the largest at each ORIENTATION too, and those are
    different rectangles, so every band is enumerated instead.
    """
    H, W = mask.shape
    out = []
    for i0 in range(H):
        band = mask[i0].copy()
        for i1 in range(i0, H):
            if i1 > i0:
                band &= mask[i1]
            if not band.any():
                break
            j0, w = _runs(band)
            if w:
                out.append((i0, j0, i1 - i0 + 1, w))
    return out


def rect_report(mask, grid=GRID):
    """The largest clean rectangle, and the largest in each ORIENTATION.

    A rectangle turned 90 degrees is not a new rectangle in a raster — it is
    the same one with its sides swapped, so "both orientations" can only mean
    what artwork shape fits: the biggest PORTRAIT block (long side up the
    canvas) and the biggest LANDSCAPE one (long side across it).  On a
    1.80 x 3.63 m canvas those are genuinely different answers.
    """
    R = all_rects(mask)

    def pack(r):
        i0, j0, hh, ww = r
        return dict(cells=int(hh * ww), x0=round(float(j0 * grid), 3),
                    y0=round(float(i0 * grid), 3),
                    w=round(float(ww * grid), 3), h=round(float(hh * grid), 3),
                    area_m2=round(float(ww * hh * grid * grid), 4),
                    aspect=round(float(ww / hh), 3))

    def best(pred):
        c = [r for r in R if pred(r)]
        return pack(max(c, key=lambda r: r[2] * r[3])) if c else None

    return (best(lambda r: True),                       # largest by area
            best(lambda r: r[2] >= r[3]),               # portrait  (h >= w)
            best(lambda r: r[3] >= r[2]))               # landscape (w >= h)


def blobs(mask):
    """Sizes of the 4-connected components of a bool mask. -> sorted desc.

    Two-pass union-find rather than scipy, which is not installed here.
    """
    H, W = mask.shape
    lab = np.full((H, W), -1, int)
    par = []

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a

    for i in range(H):
        row = mask[i]
        for j in np.where(row)[0]:
            up = lab[i - 1, j] if i and mask[i - 1, j] else -1
            lf = lab[i, j - 1] if j and mask[i, j - 1] else -1
            if up < 0 and lf < 0:
                lab[i, j] = len(par)
                par.append(len(par))
            elif up < 0 or lf < 0:
                lab[i, j] = max(up, lf)
            else:
                lab[i, j] = up
                a, b = find(up), find(lf)
                if a != b:
                    par[b] = a
    if not par:
        return np.zeros(0, int)
    flat = lab[mask]
    roots = np.array([find(int(v)) for v in flat])
    return np.sort(np.bincount(roots)[np.bincount(roots) > 0])[::-1]


def numbers(comp, per_arm, arms, sheet=SHEET, grid=GRID, fleet=None,
            park_hover=None, pitch=SHIPPED_PITCH, h=0.940):
    cell_a = grid * grid
    n_arms, cause = comp["n_arms"], comp["cause"]
    H, W = cause.shape
    tot = H * W
    feas = int((cause == FEASIBLE).sum())

    def pct(n):
        return round(100.0 * n / tot, 2)

    fleet = layout.FLEET_PROPOSED if fleet is None else fleet
    park_hover = (layout.PARK_HOVER_PROPOSED if park_hover is None
                  else park_hover)
    out = dict(
        rig="proposed", tool="lateral", h=float(h), pitch=float(pitch),
        grid=grid,
        sheet=list(sheet), cells=tot, cell_area_m2=cell_a,
        canvas_area_m2=round(tot * cell_a, 4),
        feasible_cells=feas, feasible_pct=pct(feas),
        feasible_m2=round(feas * cell_a, 4),
    )
    out["layers"] = {
        "draw_pose_only": dict(cells=int(comp["draw_any"].sum()),
                               pct=pct(int(comp["draw_any"].sum())),
                               m2=round(int(comp["draw_any"].sum()) * cell_a, 4)),
        "plus_hover": dict(cells=int(comp["hover_any"].sum()),
                           pct=pct(int(comp["hover_any"].sum())),
                           m2=round(int(comp["hover_any"].sum()) * cell_a, 4)),
        "plus_reachability": dict(cells=feas, pct=pct(feas),
                                  m2=round(feas * cell_a, 4)),
    }
    out["dead_by_cause"] = {
        CAUSE_NAME[c]: dict(cells=int((cause == c).sum()),
                            pct=pct(int((cause == c).sum())),
                            m2=round(int((cause == c).sum()) * cell_a, 4))
        for c in (NO_DRAW, NO_HOVER, NO_ROUTE)}

    out["histogram"] = {str(k): int((n_arms == k).sum())
                        for k in range(0, len(arms) + 1)}
    out["ge2_pct"] = pct(int((n_arms >= 2).sum()))
    out["ge3_pct"] = pct(int((n_arms >= 3).sum()))
    out["max_arms"] = int(n_arms.max())

    pa = {}
    for a in arms:
        d = per_arm[a]
        m = comp["per_arm"][a]
        n = int(m.sum())
        codes = d[:, 2].astype(int) if len(d) else np.zeros(0, int)
        pa[str(a)] = dict(
            name=fleet[a].name,
            xy=[round(float(v), 4) for v in fleet[a].xy],
            park_hover=list(park_hover.get(a, ())),
            draw_cells=int(len(d)), draw_pct=pct(len(d)),
            hover_cells=int((codes != NO_HOVER).sum()),
            hover_pct=pct(int((codes != NO_HOVER).sum())),
            feasible_cells=n, feasible_pct=pct(n),
            feasible_m2=round(n * cell_a, 4),
            lost_to_hover=int((codes == NO_HOVER).sum()),
            lost_to_route=int((codes == NO_ROUTE).sum()),
        )
    out["per_arm"] = pa

    # canvas thirds, along the long axis
    ys = comp["ys"]
    edges = [0, len(ys) // 3, 2 * len(ys) // 3, len(ys)]
    thirds = []
    for k in range(3):
        sl = slice(edges[k], edges[k + 1])
        sub = cause[sl]
        n = int((sub == FEASIBLE).sum())
        thirds.append(dict(
            name=["south end", "middle", "north end"][k],
            y_range=[round(float(ys[edges[k]]), 3),
                     round(float(ys[edges[k + 1] - 1]), 3)],
            cells=int(sub.size), feasible_cells=n,
            feasible_pct=round(100.0 * n / sub.size, 2),
            feasible_m2=round(n * cell_a, 4),
            dead_no_draw=int((sub == NO_DRAW).sum()),
            dead_no_hover=int((sub == NO_HOVER).sum()),
            dead_no_route=int((sub == NO_ROUTE).sum())))
    out["thirds"] = thirds

    a, p, l = rect_report(cause == FEASIBLE, grid)
    out["largest_rect"] = a
    out["largest_rect_portrait"] = p
    out["largest_rect_landscape"] = l

    # WHERE THE DEAD AREA IS, in the three shapes the rig actually makes:
    # the six discs an arm cannot reach under its own boom, the route-dead
    # middle where the two middle-row arms park in everybody's way, and
    # whatever is left over.  Disjoint, in that order, so the three add up.
    dead = cause != FEASIBLE
    XS, YS = np.meshgrid(comp["xs"], ys)
    under = np.zeros_like(dead)
    for arm in arms:
        bx, by = fleet[arm].xy
        under |= ((XS - bx) ** 2 + (YS - by) ** 2) <= 0.30 ** 2
    mid = np.zeros_like(dead)
    mid[edges[1]:edges[2]] = True

    def blk(m):
        n = int(m.sum())
        return dict(cells=n, m2=round(n * cell_a, 4), pct=pct(n))

    d_under = dead & under
    d_mid = dead & ~under & mid
    d_rest = dead & ~under & ~mid
    out["dead_split"] = dict(
        total=blk(dead),
        under_base_discs_r30=blk(d_under),
        middle_third_outside_the_discs=blk(d_mid),
        elsewhere=blk(d_rest),
        middle_third_route_dead=blk(d_mid & (cause == NO_ROUTE)),
        route_dead_everywhere=blk(dead & (cause == NO_ROUTE)),
        under_base_discs_area_m2=round(float(under.sum()) * cell_a, 4))

    # ...AND WHAT SHAPE IT IS, which is the number that explains the
    # rectangles.  81.84 % drawable and a biggest clean block of 0.74 m2 only
    # reconcile if the dead area is not one hole: it is six solid discs under
    # the booms plus a confetti of isolated refusals scattered through canvas
    # that is otherwise fine.  A rectangle is stopped by one bad pixel, so the
    # confetti costs far more artwork than its area does.
    sz = blobs(dead)
    small = int(sz[sz <= 50].sum())
    big = int(sz[sz > 50].sum())
    nbr = np.zeros_like(cause, int)
    F = (cause == FEASIBLE)
    P = np.pad(F, 1)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if (di, dj) != (0, 0):
                nbr += P[1 + di:1 + di + H, 1 + dj:1 + dj + W]
    speck = int((dead & (nbr >= 5)).sum())
    out["dead_shape"] = dict(
        blobs=int(len(sz)),
        largest_blobs=[int(v) for v in sz[:8]],
        solid_cells=big, solid_m2=round(big * cell_a, 4),
        solid_pct_of_dead=round(100.0 * big / max(1, int(dead.sum())), 1),
        confetti_cells=small, confetti_m2=round(small * cell_a, 4),
        confetti_pct_of_dead=round(100.0 * small / max(1, int(dead.sum())), 1),
        dead_cells_with_5plus_feasible_neighbours=speck,
        speckle_pct_of_dead=round(100.0 * speck / max(1, int(dead.sum())), 1),
        note=("blobs of more than 50 cells are the six under-boom discs; the "
              "rest is scattered refusals, and a rectangle is stopped by one "
              "of them"))
    return out


# ---------------------------------------------------------------------------
# the figure
# ---------------------------------------------------------------------------
# THE PALETTE IS COMPUTED, NOT PICKED.  Two ORDINAL families share one raster —
# how many arms can draw a cell, and, where none can, how far the cell got — so
# each is a single-hue ramp with monotone lightness, and the two hues are the
# blue/amber pair rather than the green/red one that collapses under
# deuteranopia.  Checked, not eyeballed (out/palette_build.py): both ramps
# monotone in OKLCH L with adjacent dL >= 0.08, hue span <= 15 deg, light end
# >= 2.0:1 on white; worst cross-family pair dE 14.4 under protanopia and 15.9
# under deuteranopia (target 8); adjacent feasible steps dE 8.3-8.7.
FEAS_COLS = ["#90b4e0", "#7199cb", "#537fb5", "#3465a0", "#114c8a", "#003275"]
DEAD_COLS = {3: "#e1ab7b", 2: "#ab6d30", 1: "#763100"}   # NO_ROUTE/HOVER/DRAW
INK, INK2, INK3 = "#1a1a19", "#4a4a48", "#77776f"
METAL = "#111827"


def _bar_panel(ax, labels, vals, cols, title, note=None, xmax=100.0,
               fmt="{:.2f}%"):
    """A thin horizontal bar row with selective direct labels."""
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, height=0.52, color=cols, zorder=3)
    for yy, v in zip(y, vals):
        ax.text(v + xmax * 0.015, yy, fmt.format(v), va="center", ha="left",
                fontsize=9, color=INK, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9, color=INK)
    ax.set_xlim(0, xmax * 1.22)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    ax.set_xticks([])
    ax.tick_params(left=False)
    for sp in ("top", "right", "bottom", "left"):
        ax.spines[sp].set_visible(False)
    ax.set_title(title, fontsize=10, color=INK, fontweight="bold", loc="left",
                 pad=6)
    if note:
        ax.text(0, -0.62, note, transform=ax.get_yaxis_transform(),
                fontsize=8, color=INK3, va="top")


def figure(comp, nums, arms, path, sheet=SHEET, grid=GRID, fleet=None,
           park_hover=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle
    from matplotlib.lines import Line2D
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from aris_sixarm import mounts

    fleet = layout.FLEET_PROPOSED if fleet is None else fleet
    park_hover = (layout.PARK_HOVER_PROPOSED if park_hover is None
                  else park_hover)
    pitch, h = nums["pitch"], nums["h"]
    n_arms, cause = comp["n_arms"], comp["cause"]
    xs, ys = comp["xs"], comp["ys"]
    ext = [xs[0] - grid / 2, xs[-1] + grid / 2, ys[0] - grid / 2,
           ys[-1] + grid / 2]

    # the right column's panels are sized by the rows they actually carry, so
    # a rig whose deepest cell is three arms deep does not get a six-row gap
    nmax = max(1, int(n_arms.max()))
    fig = plt.figure(figsize=(15.2, 13.8))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.0, 1.06],
                          height_ratios=[3.2, float(len(arms)),
                                         max(2.4, float(nmax)), 3.2],
                          left=0.055, right=0.975, top=0.878, bottom=0.175,
                          wspace=0.20, hspace=0.40)
    ax = fig.add_subplot(gs[:, 0])

    # ONE raster: 1..6 = how many arms, -1..-3 = which layer refused
    img = np.where(n_arms > 0, n_arms, -cause).astype(float)
    cmap = ListedColormap([DEAD_COLS[3], DEAD_COLS[2], DEAD_COLS[1],
                           "#ffffff"] + FEAS_COLS)
    norm = BoundaryNorm([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5,
                         6.5], cmap.N)
    ax.imshow(img, origin="lower", extent=ext, cmap=cmap, norm=norm,
              interpolation="nearest", zorder=1)
    ax.add_patch(Rectangle((ext[0], ext[2]), ext[1] - ext[0], ext[3] - ext[2],
                           fill=False, ec=INK2, lw=1.1, zorder=10))

    # the metal, in plan: boom footprint, base plate, and where the pen parks
    m = mounts.MOUNTS
    br, (pw, pd) = m.boom_r, m.plate_xy
    for arm in arms:
        bx, by = fleet[arm].xy
        ax.add_patch(Rectangle((bx - pw / 2, by - pd / 2), pw, pd, fill=False,
                               ec=METAL, lw=0.9, ls=(0, (3, 2)), zorder=6))
        ax.add_patch(Circle((bx, by), br, fill=False, ec="white", lw=3.2,
                            zorder=6))
        ax.add_patch(Circle((bx, by), br, fill=False, ec=METAL, lw=1.6,
                            zorder=7))
        ax.plot([bx], [by], marker="x", ms=7, mew=2.0, color=METAL, zorder=8)
        pa = nums["per_arm"][str(arm)]
        ax.annotate(f"arm {arm}   {pa['feasible_pct']:.1f}%", (bx, by),
                    xytext=(0, 13), textcoords="offset points", ha="center",
                    va="bottom", fontsize=8.4, color=INK, fontweight="bold",
                    zorder=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=INK3,
                              lw=0.6, alpha=0.94))
        ph = park_hover.get(arm)
        if ph:
            ax.plot([ph[0]], [ph[1]], marker="P", ms=11, color="#8a1c1c",
                    mec="white", mew=1.4, zorder=9)

    # THE SIZES LIVE IN THE LEGEND, not on top of the map: the two rectangles
    # overlap wherever the feasible region is one blob, and two label boxes in
    # the same 40 cm of canvas hide the cells they are describing.
    for key, ls, tag in (("largest_rect_portrait", "-", "P"),
                         ("largest_rect_landscape", (0, (4, 2)), "L")):
        r = nums.get(key)
        if not r:
            continue
        ax.add_patch(Rectangle((r["x0"], r["y0"]), r["w"], r["h"], fill=False,
                               ec="white", lw=3.6, zorder=9))
        ax.add_patch(Rectangle((r["x0"], r["y0"]), r["w"], r["h"], fill=False,
                               ec=METAL, lw=2.0, ls=ls, zorder=10))
        ax.text(r["x0"] + 0.03, r["y0"] + r["h"] - 0.03, tag, ha="left",
                va="top", fontsize=10, color=METAL, fontweight="bold",
                zorder=11,
                bbox=dict(boxstyle="circle,pad=0.16", fc="white", ec=METAL,
                          lw=1.0))

    ax.set_xlim(ext[0] - 0.05, ext[1] + 0.05)
    ax.set_ylim(ext[2] - 0.05, ext[3] + 0.05)
    ax.set_aspect("equal")
    ax.set_xlabel("canvas x (m)", fontsize=9.5, color=INK2)
    ax.set_ylabel("canvas y (m)", fontsize=9.5, color=INK2)
    ax.tick_params(labelsize=8.5, colors=INK3)
    for sp in ax.spines.values():
        sp.set_visible(False)
    # ---- the legend, under the map ----------------------------------------
    # THE FEASIBLE SCALE IS ORDINAL, so it is drawn as a ramp with a numbered
    # axis and not as six unrelated swatches: the reader is meant to see 1 -> 6
    # as an order, which a column of legend keys hides.
    # ...and only as far as the fleet actually gets.  A six-step key over a
    # canvas whose deepest cell is reachable by four arms promises a redundancy
    # this rig does not have.
    cb = fig.add_axes([0.055, 0.112, 0.20 * nmax / 6.0, 0.017])
    for k in range(nmax):
        cb.add_patch(Rectangle((k, 0), 1, 1, fc=FEAS_COLS[k], ec="white",
                               lw=1.2))
    cb.set_xlim(0, nmax)
    cb.set_ylim(0, 1)
    cb.set_yticks([])
    cb.set_xticks(np.arange(nmax) + 0.5)
    cb.set_xticklabels([str(k + 1) for k in range(nmax)], fontsize=9,
                       color=INK)
    cb.tick_params(length=0, pad=3)
    for sp in cb.spines.values():
        sp.set_visible(False)
    cb.set_title(f"FEASIBLE — how many arms can solo-draw the cell "
                 f"(never more than {nmax})",
                 fontsize=9.2, color=INK, fontweight="bold", loc="left", pad=5)

    d = nums["dead_by_cause"]
    hs = [Rectangle((0, 0), 1, 1, fc=DEAD_COLS[c], ec="none")
          for c in (NO_ROUTE, NO_HOVER, NO_DRAW)]
    ls_ = [f"can draw and lift, cannot FLY there   "
           f"({d['hover ok, unreachable']['pct']:.2f}%,  "
           f"{d['hover ok, unreachable']['m2']:.2f} m2)",
           f"can draw, cannot LIFT off it   "
           f"({d['draw ok, no hover']['pct']:.2f}%,  "
           f"{d['draw ok, no hover']['m2']:.2f} m2)",
           f"no certified DRAWING pose, any arm   "
           f"({d['no draw pose']['pct']:.2f}%,  "
           f"{d['no draw pose']['m2']:.2f} m2)"]
    lg1 = fig.legend(hs, ls_, loc="upper left", bbox_to_anchor=(0.055, 0.090),
                     ncol=1, fontsize=8.8, frameon=False, handlelength=1.5,
                     handleheight=1.0, labelspacing=0.4, borderpad=0.0,
                     title="DEAD — the last layer the cell got through",
                     title_fontproperties=dict(size=9.2, weight="bold"))
    lg1._legend_box.align = "left"

    hs2 = [Line2D([], [], color=METAL, marker="x", ls="none", ms=7, mew=2),
           Line2D([], [], color="#8a1c1c", marker="P", ls="none", ms=9,
                  mec="white", mew=1.2),
           Line2D([], [], color=METAL, lw=2.2),
           Line2D([], [], color=METAL, lw=2.2, ls=(0, (4, 2)))]
    rp, rl = nums.get("largest_rect_portrait"), nums.get("largest_rect_landscape")
    ls2 = ["arm base: boom r = 0.10 m, plate 0.226 x 0.190 m",
           "where that arm's pen waits while another draws",
           "P  largest PORTRAIT artwork, zero dead spots"
           + (f"   ({rp['w']:.2f} x {rp['h']:.2f} m)" if rp else ""),
           "L  largest LANDSCAPE artwork, zero dead spots"
           + (f"   ({rl['w']:.2f} x {rl['h']:.2f} m)" if rl else "")]
    lg2 = fig.legend(hs2, ls2, loc="upper left",
                     bbox_to_anchor=(0.375, 0.090), ncol=1,
                     fontsize=8.8, frameon=False, handlelength=1.8,
                     labelspacing=0.4, borderpad=0.0,
                     title="THE RIG", title_fontproperties=dict(
                         size=9.2, weight="bold"))
    lg2._legend_box.align = "left"

    # ---- right column ------------------------------------------------------
    L = nums["layers"]
    ax1 = fig.add_subplot(gs[0, 1])
    _bar_panel(ax1,
               ["1. a certified DRAWING pose\n     (banded atlas, strict GO)",
                "2. + a certified HOVER over it\n     (v3 full-fiber lift)",
                "3. + the arm can FLY there\n     (routed from its park)"],
               [L["draw_pose_only"]["pct"], L["plus_hover"]["pct"],
                L["plus_reachability"]["pct"]],
               ["#9aa5b1", "#5b6b7d", FEAS_COLS[3]],
               "What each layer costs, as % of canvas",
               note="each layer is a subset of the one above it")

    ax2 = fig.add_subplot(gs[1, 1])
    pa = nums["per_arm"]
    order = sorted(arms, key=lambda a: -pa[str(a)]["feasible_pct"])
    _bar_panel(ax2, [f"arm {a}  ({pa[str(a)]['name']})" for a in order],
               [pa[str(a)]["feasible_pct"] for a in order],
               [FEAS_COLS[3]] * len(order),
               "Canvas each arm can solo-draw, alone",
               note="they overlap: the union above is not the sum",
               xmax=max(pa[str(a)]["feasible_pct"] for a in order))

    ax3 = fig.add_subplot(gs[2, 1])
    hist = nums["histogram"]
    ks = [k for k in range(1, nmax + 1)]
    tot = nums["cells"]
    _bar_panel(ax3, [f"{k} arm" + ("" if k == 1 else "s") for k in ks],
               [100.0 * hist.get(str(k), 0) / tot for k in ks],
               [FEAS_COLS[k - 1] for k in ks],
               "Redundancy: how many arms reach each cell",
               note="a cell only one arm can reach has no fallback",
               xmax=max(100.0 * hist.get(str(k), 0) / tot for k in ks))

    ax4 = fig.add_subplot(gs[3, 1])
    th = nums["thirds"]
    _bar_panel(ax4, [f"{t['name']}\n  y {t['y_range'][0]:.2f} - "
                     f"{t['y_range'][1]:.2f} m" for t in th],
               [t["feasible_pct"] for t in th], [FEAS_COLS[3]] * 3,
               "Feasible by canvas third",
               note="the middle third is where the seam and the two "
                    "middle-row arms are")

    r = nums["largest_rect"]
    fig.suptitle(
        f"THE FEASIBLE DRAWING WORKSPACE  —  {nums['feasible_pct']:.1f}% of the "
        f"canvas, one arm at a time",
        fontsize=18, fontweight="bold", color=INK, x=0.055, y=0.972,
        ha="left")
    fig.text(0.055, 0.937,
             f"proposed rig, lateral pen  ·  h = {h:.3f} m  ·  pitch "
             f"{pitch:.2f} m  ·  canvas {sheet[0]:.4f} x {sheet[1]:.5f} m in "
             f"2 cm cells  ·  {nums['feasible_m2']:.2f} of "
             f"{nums['canvas_area_m2']:.2f} m2 drawable  ·  biggest clean "
             f"block {r['w']:.2f} x {r['h']:.2f} m",
             fontsize=11, color=INK2, ha="left")
    fig.text(0.055, 0.913,
             "SOLO: one arm inks while the other five hold their certified "
             "parks — an upper bound on what any concurrent schedule can ink.   "
             "NOT MODELLED: ceiling cross-members (steel undesigned), cable "
             "dress, fingertip cradle, paper transport.",
             fontsize=8.8, color=INK3, ha="left", style="italic")
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default=str(ATLAS_DIR))
    ap.add_argument("--out", default=str(OUT / "feasible_workspace"))
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 4))
    ap.add_argument("--every", type=int, default=1,
                    help="sweep every Nth certified cell (uniform pilot)")
    ap.add_argument("--chunk", type=int, default=24)
    ap.add_argument("--arms", default="")
    ap.add_argument("--no-redundant-hover", action="store_true",
                    help="stop at the first ladder rung, as a timeline would")
    ap.add_argument("--no-fiber-hover", action="store_true",
                    help="do not search the rest of the hover fiber when no "
                         "ladder rung can be flown to (the pre-2026-08-27 map)")
    ap.add_argument("--fiber-tries", type=int, default=12,
                    help="hover-fiber candidates a cell may spend on the "
                         "pen-up layer before it is called unreachable.  "
                         "Measured (out/guard_pocket.py) 12 finds 82.6 %% of "
                         "the ends an unbounded search finds and 48 finds "
                         "98.6 %%, so a budgeted map is a LOWER bound on the "
                         "drawable canvas and never an overstatement")
    ap.add_argument("--hover-lean-deg", type=float, default=0.0,
                    help="pen lean the FIBER search may use as its last rung, "
                         "matching the run's --tilt-max-deg (0 = vertical only)")
    ap.add_argument("--pitch", type=float, default=None,
                    help="transverse pair spacing; default is the shipped "
                         "0.61 and its baked parks. Any other value re-derives "
                         "the layout AND the park poses — pass a matching "
                         "--atlas swept at that pitch.")
    ap.add_argument("--sweep-atlas", action="store_true",
                    help="re-sweep the reachability atlas for --pitch into "
                         "--atlas first (needed for any pitch but the shipped "
                         "one: the committed 0.65 atlas predates the mesh "
                         "audit and its model signature is refused)")
    ap.add_argument("--from-raw", default=None, metavar="NPZ",
                    help="skip the sweep and rebuild the map, the numbers and "
                         "the figure from an existing _raw.npz (or a _ckpt.npz "
                         "of a run still going)")
    a = ap.parse_args()

    fl, parks, h, pitch = rig(a.pitch)
    arms = [int(v) for v in a.arms.split(",")] if a.arms else sorted(fl)
    ph = (layout.PARK_HOVER_PROPOSED if abs(pitch - SHIPPED_PITCH) < 1e-9
          else park_hovers(fl, parks, h))
    print(f"rig=proposed tool=lateral PEN_LAT={frames.PEN_LAT} h={h} "
          f"pitch={pitch}"
          + ("  (SHIPPED: baked parks)" if abs(pitch - SHIPPED_PITCH) < 1e-9
             else "  (re-derived layout AND parks)"))
    print(f"STATIC_SAFE={paper.STATIC_SAFE} PAPER_SAFE={writing.PAPER_SAFE} "
          f"FRAME_FLOOR={paper.FRAME_FLOOR}")

    if a.sweep_atlas:
        # ONE ATLAS PER COLLISION MODEL, and the model is this build's.
        # `atlas.sweep_arm` stamps `model_signature` into every npz, so a
        # comparison rig gets a comparison atlas rather than a stale one.
        import multiprocessing as mp
        d = Path(a.atlas)
        d.mkdir(parents=True, exist_ok=True)
        print(f"sweeping atlas for pitch {pitch} into {d}", flush=True)
        ta = time.time()
        with mp.get_context("fork").Pool(min(6, a.workers)) as pool:
            pool.starmap(atlas.sweep_arm,
                         [(x, str(d), GRID, 1.05, h, 15.0, fl[x].pen, fl,
                           SHEET, frames.PEN_LAT) for x in arms])
        print(f"atlas {time.time() - ta:.0f}s", flush=True)

    t0 = time.time()
    if a.from_raw:
        z = np.load(a.from_raw)
        per_arm = {int(k[3:]): z[k] for k in z.files}
        arms = sorted(per_arm)
        print(f"rebuilt from {a.from_raw}: "
              f"{sum(len(v) for v in per_arm.values())} arm-cells")
    else:
        per_arm = sweep(arms, Path(a.atlas), h, a.workers, a.chunk, a.every,
                        not a.no_redundant_hover, ckpt=a.out + "_ckpt.npz",
                        pitch=pitch, fiber=not a.no_fiber_hover,
                        lean=a.hover_lean_deg, tries=a.fiber_tries)
        print(f"sweep {time.time() - t0:.0f}s", flush=True)
        np.savez_compressed(a.out + "_raw.npz",
                            **{f"arm{k}": v for k, v in per_arm.items()})

    if a.every > 1:
        # A PILOT IS NOT A MAP.  Every Nth certified cell says what fraction of
        # the CERTIFIED cells survive the two layers above the atlas; it cannot
        # say what fraction of the CANVAS does, because the cells it skipped are
        # indistinguishable from cells no arm can draw.
        print(f"\nPILOT (every {a.every}th cell) — per-arm-cell rates, "
              f"NOT canvas coverage")
        allc = np.vstack([v for v in per_arm.values() if len(v)])
        for arm in arms:
            d = per_arm[arm]
            c = d[:, 2].astype(int)
            print(f"  arm {arm}: {len(d)} cells  feasible "
                  f"{100.0 * (c == FEASIBLE).mean():5.1f}%  "
                  f"no-hover {100.0 * (c == NO_HOVER).mean():4.1f}%  "
                  f"no-route {100.0 * (c == NO_ROUTE).mean():5.1f}%")
        c = allc[:, 2].astype(int)
        print(f"  ALL {len(allc)} arm-cells: feasible "
              f"{100.0 * (c == FEASIBLE).mean():.2f}%, "
              f"no-hover {100.0 * (c == NO_HOVER).mean():.2f}%, "
              f"no-route {100.0 * (c == NO_ROUTE).mean():.2f}%")
        return

    comp = compose(per_arm, arms)
    nums = numbers(comp, per_arm, arms, fleet=fl, park_hover=ph, pitch=pitch,
                   h=h)
    nums["sweep_seconds"] = round(time.time() - t0, 1)
    nums["caveats"] = [
        "Ceiling cross-members are NOT modelled: the steel that carries the six "
        "booms has no design yet, so only the booms themselves (r = 0.10 m, "
        "gated as circumscribed squares) and their base plates are obstacles.",
        "Cable dress is NOT modelled: no umbilical, drag chain or festoon "
        "occupies any of this volume, and every arm is a bare kinematic chain "
        "plus its measured link capsules.",
        "The fingertip cradle is NOT modelled: the pen is the certified "
        "tip = TCP + R @ (0.110, 0, 0.110) lateral holder and nothing around it.",
        "Paper transport, table frame and operator access are NOT modelled.",
        "SOLO only: this is one arm drawing with five parked. It is an upper "
        "bound on what any concurrent schedule can ink.",
        "The router plans against the metal and is then VETOED by the parked "
        "partners rather than routing around them, so a park-aware router could "
        "only recover cells, never lose them.",
    ]
    with open(a.out + ".json", "w") as f:
        json.dump(nums, f, indent=2)
    np.savez_compressed(a.out + "_map.npz", n_arms=comp["n_arms"],
                        cause=comp["cause"], draw_any=comp["draw_any"],
                        hover_any=comp["hover_any"], xs=comp["xs"],
                        ys=comp["ys"],
                        **{f"mask{k}": v for k, v in comp["per_arm"].items()})
    figure(comp, nums, arms, a.out + ".png", fleet=fl, park_hover=ph)

    print(f"\nFEASIBLE {nums['feasible_pct']:.2f}%  "
          f"({nums['feasible_m2']:.3f} / {nums['canvas_area_m2']:.3f} m2)")
    for k, v in nums["layers"].items():
        print(f"  {k:>18}: {v['pct']:6.2f}%  {v['m2']:.3f} m2")
    for k, v in nums["dead_by_cause"].items():
        print(f"  DEAD {k:>22}: {v['pct']:6.2f}%  {v['m2']:.3f} m2")
    for k, v in nums["per_arm"].items():
        print(f"  arm {k:>3}: {v['feasible_pct']:6.2f}%  "
              f"(draw {v['draw_pct']:.2f}%, -{v['lost_to_hover']} hover, "
              f"-{v['lost_to_route']} route)")
    for k in ("largest_rect", "largest_rect_portrait",
              "largest_rect_landscape"):
        r = nums.get(k)
        if r:
            print(f"  {k:>24}: {r['w']:.2f} x {r['h']:.2f} m "
                  f"= {r['area_m2']:.3f} m2 at ({r['x0']:.2f}, {r['y0']:.2f})")
    ds = nums["dead_split"]
    print(f"  DEAD {ds['total']['m2']:.3f} m2 splits as: "
          f"{ds['under_base_discs_r30']['m2']:.3f} under the six base discs "
          f"(r=0.30), {ds['middle_third_outside_the_discs']['m2']:.3f} in the "
          f"middle third outside them "
          f"(of which {ds['middle_third_route_dead']['m2']:.3f} route-dead), "
          f"{ds['elsewhere']['m2']:.3f} elsewhere")
    print(f"  wrote {a.out}.png / .json")


if __name__ == "__main__":
    main()
