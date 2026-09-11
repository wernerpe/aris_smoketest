"""CUT AT EVERY TRANSITION, THEN MERGE: the fewest pieces a picture draws in.

Pete, 2026-09-11: *"what if we cut lines everywhere where there is a transition
and then figured out some sort of greedy merging and allocation trying to
minimize the overall number of lines?"*

THE POINT.  A line is handed to the fleet whole, and the fleet hands it back in
pieces, because no single arm certifies all of it.  Today that pieceing is a
by-product of the allocator's cost search: a span is offered, planned, refused,
split, re-offered, and the number of pieces that fall out is whatever falls
out.  Pete's question turns it the other way round.  The pieces are not a
by-product; the pieces are the objective.  Fewer pieces is fewer pen-ups, fewer
hand-overs, fewer seams to register, and a shorter tour for every arm.

AND IT IS NOT A GREEDY PROBLEM.  It looks like one, and the greedy answer is
even quite good, but the exact answer is an eight-line dynamic programme:

  1.  SAMPLE the line finely and read, at each sample, its CAPABILITY SET --
      the set of (stage, arm) pairs that may draw that point.  A pair may draw
      it when the arm has a certified drawing pose there in the atlas AND the
      point is inside that arm's work cell for that stage (docs/V2_WORKCELLS).
  2.  CUT wherever the capability set CHANGES.  Between two cuts every point of
      the line has exactly the same set of possible drawers, so the whole
      stretch -- an ATOM -- is indivisible: nothing is ever gained by cutting
      inside it.  The atoms are the alphabet, and there are very few of them.
  3.  ASSIGN each atom one pair from its set so that the number of maximal runs
      -- PIECES -- is smallest.  Unit cost per change, states are the (stage,
      arm) pairs: a textbook chain DP, O(atoms x states) with the standard
      "stay, or switch from the best predecessor" contraction, milliseconds for
      a thousand lines, and provably optimal per line.

WHY PER LINE IS ENOUGH.  The piece count is a SUM over lines, and the atoms of
one line constrain no other line, so minimising each line's pieces minimises
the total.  Cross-line coupling -- load balance, stage timing -- is a SECOND
objective, and it enters here exactly where it cannot hurt the first: as the
DP's lexicographic tie-break (`balance_w = 0`, the default), or as a small
weight on the same term (`balance_w > 0`) for a caller that would rather trade
pieces for balance.  See `docs/V2_TRACES.md` for which one Pete should want.

SEAMS.  Two pieces of one line meet somewhere, and where they meet is a joint
in the ink.  There are exactly two kinds and they want opposite treatment:

  IN AN OVERLAP ZONE, both arms certify a stretch of the line, and the DP's
  choice of switch point inside it is arbitrary.  Put the seam in the MIDDLE of
  the overlap -- the furthest either arm is from the edge of what it can do --
  and OVERDRAW each side by `OVERDRAW_M` in its own drawing direction, so the
  two strokes lap rather than abut and a registration error of less than delta
  leaves no white gap.
  AT A HARD EDGE, the overlap is empty: the last point one arm certifies is the
  first the other does.  There is nothing to overdraw into; the pieces meet at
  the transition point exactly and the seam is as good as the calibration.

The summary counts the two separately, because the second is the one that shows
up in a photograph.

WHAT THIS MODULE IS NOT.  It does not plan, certify, sequence or move anything,
and it changes no constant and no planner semantic.  It reads the atlas and a
stage pattern and answers "how few pieces can this picture be drawn in, and who
draws which".  `plan_stroke` still has to accept each piece, and a piece this
module hands out is a span the ATLAS certifies at 2 cm resolution, which is the
allocator's own prefilter (`allocate.atlas_cells`) and not a plan.

    python3 -m aris_sixarm.traces --help
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# the geometry these numbers live in -- ALL OF IT COPIED, NONE OF IT NEW
# ---------------------------------------------------------------------------
# Every constant below is the one `scripts/workcell_envelopes.py` measured
# docs/V2_WORKCELLS.md's recommendation with.  They are repeated rather than
# imported because that file is a script, not a module, and because a stage
# pattern that drifted from the geometry it was certified against would be a
# pattern nobody measured.
ARMS = (2, 13, 17, 31, 71, 97)
GRID = 0.02                                     # the atlas lattice, metres
BLOCK = (0.16, 0.00, 1.64, 3.62)                # the certified block
COL_X = (0.5967, 1.2067)                        # the base lattice: 2 columns,
ROW_Y = (0.6051, 1.8153, 3.0255)                # 3 rows (layout.paired_grid)
X_MID = float(np.mean(COL_X))                                   # 0.9017
Y_CUT = (float(np.mean(ROW_Y[:2])), float(np.mean(ROW_Y[1:])))  # 1.2102, 2.4204
ARM_AT = {(0, 0): 13, (1, 0): 17, (0, 1): 31, (1, 1): 71, (0, 2): 2, (1, 2): 97}
COL_OF = {a: i for (i, j), a in ARM_AT.items()}
ROW_OF = {a: j for (i, j), a in ARM_AT.items()}

# THE DEAD BAND IS A CLIFF, NOT A SLOPE (docs/V2_WORKCELLS.md section 4): two
# adjacent rows' ink clears by -127.2 mm at 0.20 m, -3.8 mm at 0.30 m and
# +85.8 mm at 0.40 m, which is the moment the two elbow sweeps stop
# overlapping.  0.40 m is the recommendation and anything less does not clear.
DEAD_BAND_M = 0.40

OVERDRAW_M = 0.005      # delta: how far each side of a seam laps the other
MIN_PIECE_M = 0.010     # a piece shorter than this is absorbed if it can be
SAMPLE_DS = 0.004       # capability is read every 4 mm -- a fifth of the grid
REFINE_TOL = 1e-5       # ...and every transition is then bisected to 0.01 mm
SPAN_EPS = 1e-9         # "the same parameter", for the hard-edge test

UNCOVERED = -1          # the state of an atom no (stage, arm) pair can draw


# ---------------------------------------------------------------------------
# 1.  THE CAPABILITY MAP
# ---------------------------------------------------------------------------
Rect = tuple[float, float, float, float]        # (x0, y0, x1, y1), inclusive


def rect_contains(rects: Sequence[Rect], x, y) -> np.ndarray:
    """Is (x, y) inside the union of `rects`?  -> bool array of x's shape."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    out = np.zeros(np.shape(x), bool)
    for (x0, y0, x1, y1) in rects:
        out |= (x >= x0 - SPAN_EPS) & (x <= x1 + SPAN_EPS) & \
               (y >= y0 - SPAN_EPS) & (y <= y1 + SPAN_EPS)
    return out


@dataclass
class Coverage:
    """Which 2 cm cells each arm has a CERTIFIED DRAWING POSE at.

    One boolean lattice per arm, indexed `[iy, ix]` at `round(x / grid)`.  TWO
    GATES EXIST IN THIS REPO AND THEY ANSWER DIFFERENT QUESTIONS, so both are
    offered and the caller says which:

      `strict` -- the atlas's own `strict_go`, `margin >= GATE_MARGIN` (0.30)
          and `sigma_min >= GATE_SIGMA` (0.14).  This is the gate every
          envelope in docs/V2_WORKCELLS.md was measured under, so a cell
          offered here is a cell whose clearance numbers that document owns.
          It is the honest floor for a HAND-OVER, which is a claim about where
          two arms BOTH work.
      `flat` -- `allocate.atlas_cells`' reading, `flat_margin >= 0.15` or a
          lean the run permits.  The planner's permissive prefilter, and the
          one v19's 46 segments were allocated against; comparing this module's
          piece count with v19's on anything else is comparing two maps.

    `erode` shrinks each arm's mask by one cell in the 8-neighbourhood, which is
    `dead_spans.go_cells`' default and exists because a CONTINUOUS stroke
    sample rounds to its nearest cell and can therefore sit up to 14 mm outside
    the region that cell certifies.  Off by default, to match `certified_area`
    and `workcell_envelopes`; on, every number in this module gets more
    conservative and none of them gets wrong.
    """
    grid: float
    mask: dict[int, np.ndarray]         # arm -> (ny, nx) bool
    source: str = ""

    @property
    def arms(self) -> tuple[int, ...]:
        return tuple(sorted(self.mask))

    def capable(self, arm: int, x, y) -> np.ndarray:
        m = self.mask[arm]
        ix = np.rint(np.asarray(x, float) / self.grid).astype(int)
        iy = np.rint(np.asarray(y, float) / self.grid).astype(int)
        ok = (ix >= 0) & (ix < m.shape[1]) & (iy >= 0) & (iy < m.shape[0])
        out = np.zeros(np.shape(ix), bool)
        if out.ndim == 0:
            return np.bool_(bool(ok) and bool(m[int(iy), int(ix)]))
        out[ok] = m[iy[ok], ix[ok]]
        return out

    def area_m2(self, arm: int) -> float:
        return float(self.mask[arm].sum()) * self.grid ** 2


def coverage_from_atlas(atlas_dir, arms: Iterable[int] = ARMS,
                        gate: str = "strict", tilt_max_deg: float = 0.0,
                        erode: bool = False,
                        require_current: bool = True) -> Coverage:
    """The certified-drawing-pose lattice, read from a swept atlas directory.

    `require_current` is on for the same reason `workcell_envelopes.py` has it
    on: an atlas carries the collision model and search policy it was swept
    under, and a stale one is not a difference of opinion, it is a set of
    certifications this build would refuse.  Reading the shipped atlas needs
    `ARIS_RIG=proposed ARIS_TOOL=lateral` in the environment, because that is
    the rig it was swept for.
    """
    from . import atlas as atlas_mod
    if gate not in ("strict", "flat"):
        raise ValueError(f"gate is 'strict' or 'flat', not {gate!r}")
    d = Path(atlas_dir)
    rows, grid = {}, None
    for a in arms:
        arr, meta = atlas_mod.load(d, a)
        cur, why = atlas_mod.is_current(meta)
        if require_current and not cur:
            raise ValueError(f"atlas for arm {a} is stale: {why}")
        g = float(meta["grid"])
        grid = g if grid is None else grid
        if abs(g - grid) > 1e-12:
            raise ValueError(f"arm {a}'s atlas is on a {g} m grid, not {grid}")
        if gate == "strict":
            keep = atlas_mod.strict_go(arr)
        else:                       # allocate.atlas_cells' own reading
            keep = (arr[:, atlas_mod.FLATCOL] >= 0.15) | \
                   ((arr[:, atlas_mod.LEANCOL] >= 0.0) &
                    (arr[:, atlas_mod.LEANCOL] <= float(tilt_max_deg) + 1e-9))
        rows[a] = arr[keep][:, :2]
    nx = max(int(np.rint(r[:, 0].max() / grid)) for r in rows.values()) + 1
    ny = max(int(np.rint(r[:, 1].max() / grid)) for r in rows.values()) + 1
    mask = {}
    for a, r in rows.items():
        m = np.zeros((ny, nx), bool)
        m[np.rint(r[:, 1] / grid).astype(int),
          np.rint(r[:, 0] / grid).astype(int)] = True
        mask[a] = erode_mask(m) if erode else m
    return Coverage(grid=grid, mask=mask, source=f"{d}:{gate}")


def erode_mask(m: np.ndarray) -> np.ndarray:
    """Drop every cell with a dead 8-neighbour -- `dead_spans.go_cells`' erosion."""
    p = np.zeros((m.shape[0] + 2, m.shape[1] + 2), bool)
    p[1:-1, 1:-1] = m
    out = np.ones_like(m)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            out &= p[dy:dy + m.shape[0], dx:dx + m.shape[1]]
    return out


def coverage_from_rects(per_arm: dict[int, Sequence[Rect]], grid: float = GRID,
                        extent: Rect = (0.0, 0.0, 2.0, 3.8)) -> Coverage:
    """A coverage lattice rasterised from rectangles -- for tests and toys.

    Same object, same lookup, no atlas and no rig: a test that wants "arm 13 can
    draw the left half" should not need a swept sweep to say so.
    """
    nx = int(np.rint(extent[2] / grid)) + 1
    ny = int(np.rint(extent[3] / grid)) + 1
    xs = np.arange(nx) * grid
    ys = np.arange(ny) * grid
    X, Y = np.meshgrid(xs, ys)
    return Coverage(grid=grid,
                    mask={a: rect_contains(r, X, Y) for a, r in per_arm.items()},
                    source="rects")


# ---------------------------------------------------------------------------
# 2.  THE STAGE CELLS, AS DATA
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StageCell:
    """"Arm `arm` may draw inside `region` during stage `stage`", and nothing else.

    The work-cell object docs/V2_WORKCELLS.md section 5 lists as missing.  A
    region is a union of axis-aligned rectangles in PAPER coordinates, because
    every cell the recommendation needs is one, and because a rectangle is the
    only region shape the envelope script ever measured.
    """
    stage: int
    arm: int
    region: tuple[Rect, ...]
    name: str = ""

    def contains(self, x, y) -> np.ndarray:
        return rect_contains(self.region, x, y)


@dataclass(frozen=True)
class Pattern:
    """A stage sequence: who may draw where, in which stage."""
    name: str
    cells: tuple[StageCell, ...]
    note: str = ""

    @property
    def n_stages(self) -> int:
        return 1 + max((c.stage for c in self.cells), default=-1)

    def stage(self, s: int) -> tuple[StageCell, ...]:
        return tuple(c for c in self.cells if c.stage == s)


def row_band(j: int, dead_band_m: float = DEAD_BAND_M) -> Rect:
    """Full-width row band `j`, eroded in y by half the dead band each side.

    `workcell_envelopes.block_rect(i, j, 0.0, s)` unioned over the two columns,
    with `s = dead_band_m / 2`: only the INTERNAL boundaries erode, because
    giving back paper at the rim buys clearance from nobody.
    """
    h = dead_band_m / 2.0
    y0 = BLOCK[1] if j == 0 else Y_CUT[j - 1] + h
    y1 = (Y_CUT[j] - h) if j < 2 else BLOCK[3]
    return (BLOCK[0], y0, BLOCK[2], y1)


def seam_band(k: int, dead_band_m: float = DEAD_BAND_M) -> Rect:
    """The dead band between rows `k` and `k+1`, which a seam stage returns for."""
    h = dead_band_m / 2.0
    return (BLOCK[0], Y_CUT[k] - h, BLOCK[2], Y_CUT[k] + h)


def zigzag_pattern(dead_band_m: float = DEAD_BAND_M) -> Pattern:
    """THE RECOMMENDATION of docs/V2_WORKCELLS.md, as a `Pattern`.

    One arm per full-width row band, a 0.40 m dead band in y between bands, the
    two columns alternating between the two main stages, then SIX 2-active seam
    stages that come back for the dead bands.  Verbatim the stage list
    `workcell_envelopes.py` evaluates as `3-active-rowband-y20+6seams`:
    +85.8 mm of ink-vs-ink, 100.0 % of the block, 2.35x the serial makespan,
    tightest seam stage +194.3 mm.

    Stage 0 is 13 / 71 / 2 and stage 1 is 17 / 31 / 97 -- one arm per row,
    columns alternating DOWN THE ROWS, which is the partition that works.  "One
    arm per COLUMN" puts a transverse pair in the air at once and that pair is
    at -262 mm however the paper is cut.

    A SEAM NEEDS AN OUTER ARM *AND* A MIDDLE ARM, which is why stages 6 and 7
    exist (docs/V2_WORKCELLS.md section 4b, commit 07da40e).  Tip reach over
    the block is y in [0.000, 1.320] for 13/17, [1.080, 2.560] for 31/71 and
    [2.280, 3.600] for 2/97, so SEAM1's floor -- y in [2.24, 2.40], 92 cells,
    2.7 % of the block, and CSAIL stroke 17 entirely -- is reachable only by a
    MIDDLE arm, and the four-seam version offered SEAM1 to nobody but 2 and 97.
    The pairing that covers it has to put 31 or 71 on SEAM1 against a ROW-0 arm
    on SEAM0: 13 against 31 is +194.3 mm and 17 against 71 is +201.1 mm.  The
    tempting alternative -- crossing 31 and 71 over the two seams to keep six
    stages -- is -163.1 and -201.5 mm: the transverse pair is unseparable on
    this rig on either axis.

    The two extra stages cost 6.4 % of the parallel speedup (2.51x -> 2.35x)
    and buy the last 2.7 % of the block AND lift stage-compatible redundancy
    from 41.6 % to 47.1 %, which is the atlas's own ceiling.  Arms 13 and 17
    are offered SEAM0 twice (stages 2 and 6, 3 and 7); `Capability` merges a
    repeated (arm, region) into one state, so that costs no state and no piece.
    """
    R = [row_band(j, dead_band_m) for j in (0, 1, 2)]
    S = [seam_band(k, dead_band_m) for k in (0, 1)]
    plan = [
        (0, ((13, R[0], "R0"), (71, R[1], "R1"), (2, R[2], "R2"))),
        (1, ((17, R[0], "R0"), (31, R[1], "R1"), (97, R[2], "R2"))),
        (2, ((13, S[0], "SEAM0"), (97, S[1], "SEAM1"))),
        (3, ((17, S[0], "SEAM0"), (2, S[1], "SEAM1"))),
        (4, ((31, S[0], "SEAM0"), (97, S[1], "SEAM1"))),
        (5, ((71, S[0], "SEAM0"), (2, S[1], "SEAM1"))),
        (6, ((13, S[0], "SEAM0"), (31, S[1], "SEAM1"))),
        (7, ((17, S[0], "SEAM0"), (71, S[1], "SEAM1"))),
    ]
    cells = tuple(StageCell(s, arm, (rect,), nm)
                  for s, row in plan for (arm, rect, nm) in row)
    return Pattern(f"zigzag-rowband-y{int(dead_band_m * 50):02d}+6seams", cells,
                   "docs/V2_WORKCELLS.md section 4b recommendation: one arm per "
                   f"full-width row band, {dead_band_m:.2f} m dead band in y, "
                   "columns alternating, six 2-active seam stages")


def single_stage_pattern() -> Pattern:
    """NO STAGES AND NO CELLS: every arm, everywhere its atlas allows.

    The control the DP is measured against, and NOT a runnable pattern -- six
    arms drawing at once never clears (docs/V2_WORKCELLS.md section 4: eroding
    every Voronoi block by 0.60 m still leaves the worst pair at -23.2 mm).
    Its piece count is the floor staging is charged against, nothing more.
    """
    cells = tuple(StageCell(0, a, (BLOCK,), "ALL") for a in ARMS)
    return Pattern("single-stage", cells,
                   "every arm everywhere in the certified block; NOT collision-"
                   "certified, this is the DP's unconstrained floor")


# ---------------------------------------------------------------------------
# 3.  THE CAPABILITY INDEX -- coverage AND cell, as a bitmask per point
# ---------------------------------------------------------------------------
@dataclass
class Capability:
    """`states_at(x, y)` -> the capability set, packed one bit per state.

    A state is a (stage, arm) pair with a region.  Two cells that offer the SAME
    arm the SAME region in different stages are ONE state: the recommendation's
    stages 4 and 5 re-offer arm 97 and arm 2 exactly what stages 2 and 3 do, and
    splitting a seam's ink between two interchangeable stages would be noise in
    the load table and an arbitrary choice in the DP.  `merged` records them.
    """
    coverage: Coverage
    pattern: Pattern
    states: tuple[StageCell, ...]
    merged: tuple[tuple[int, int], ...] = ()    # (dropped stage, kept stage), arm

    @property
    def n_states(self) -> int:
        return len(self.states)

    def label(self, k: int) -> str:
        if k == UNCOVERED:
            return "uncovered"
        c = self.states[k]
        return f"s{c.stage}/arm{c.arm}"

    def states_at(self, x, y) -> np.ndarray:
        """-> uint32 bitmask array; bit k set iff state k may draw that point."""
        x = np.atleast_1d(np.asarray(x, float))
        y = np.atleast_1d(np.asarray(y, float))
        bits = np.zeros(x.shape, np.uint32)
        arm_ok: dict[int, np.ndarray] = {}
        for k, c in enumerate(self.states):
            if c.arm not in arm_ok:
                arm_ok[c.arm] = self.coverage.capable(c.arm, x, y)
            ok = arm_ok[c.arm] & c.contains(x, y)
            bits |= ok.astype(np.uint32) << np.uint32(k)
        return bits

    def members(self, bits: int) -> tuple[int, ...]:
        b = int(bits)
        return tuple(k for k in range(self.n_states) if b >> k & 1)


def capability(coverage: Coverage, pattern: Pattern) -> Capability:
    """Fold a coverage lattice and a stage pattern into one lookup."""
    seen: dict[tuple[int, tuple[Rect, ...]], int] = {}
    states: list[StageCell] = []
    merged: list[tuple[int, int]] = []
    for c in sorted(pattern.cells, key=lambda c: (c.stage, c.arm)):
        key = (c.arm, tuple(sorted(c.region)))
        if key in seen:
            merged.append((c.stage, states[seen[key]].stage))
            continue
        seen[key] = len(states)
        states.append(c)
    if len(states) > 31:
        raise ValueError(f"{len(states)} states; the bitmask holds 31")
    missing = {c.arm for c in states} - set(coverage.mask)
    if missing:
        raise ValueError(f"pattern names arms with no coverage: {sorted(missing)}")
    return Capability(coverage, pattern, tuple(states), tuple(merged))


# ---------------------------------------------------------------------------
# 4.  ARC LENGTH, AND CUTTING A POLYLINE AT A PARAMETER
# ---------------------------------------------------------------------------
def cumlen(pts: np.ndarray) -> np.ndarray:
    p = np.asarray(pts, float)
    d = np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1]))
    return np.concatenate([[0.0], np.cumsum(d)])


def points_at(pts: np.ndarray, cum: np.ndarray, s) -> np.ndarray:
    """Linear interpolation along the polyline at arc lengths `s`. -> (n, 2)."""
    p = np.asarray(pts, float)
    s = np.atleast_1d(np.asarray(s, float))
    L = cum[-1]
    if L <= 0:
        return np.repeat(p[:1], len(s), axis=0)
    s = np.clip(s, 0.0, L)
    return np.column_stack([np.interp(s, cum, p[:, 0]),
                            np.interp(s, cum, p[:, 1])])


def sub_polyline(pts: np.ndarray, cum: np.ndarray, s0: float, s1: float
                 ) -> np.ndarray:
    """The stretch of the polyline between two arc lengths, endpoints exact."""
    p = np.asarray(pts, float)
    L = float(cum[-1])
    s0, s1 = max(0.0, min(s0, L)), max(0.0, min(s1, L))
    if s1 <= s0 or L <= 0:
        return points_at(p, cum, [s0, max(s1, s0)])
    inner = p[(cum > s0 + SPAN_EPS) & (cum < s1 - SPAN_EPS)]
    ends = points_at(p, cum, [s0, s1])
    return np.vstack([ends[:1], inner, ends[1:]])


# ---------------------------------------------------------------------------
# 5.  ATOMIC SEGMENTATION -- cut wherever the capability set changes
# ---------------------------------------------------------------------------
@dataclass
class Atom:
    """An indivisible stretch of one line: one capability set from end to end."""
    s0: float
    s1: float
    bits: int

    @property
    def length(self) -> float:
        return self.s1 - self.s0


def atoms_of(pts: np.ndarray, cap: Capability, ds: float = SAMPLE_DS,
             tol: float = REFINE_TOL) -> tuple[list[Atom], np.ndarray, np.ndarray]:
    """Sample finely, cut at every capability change, bisect each cut. -> atoms.

    The sampling is the only approximation in this module and it is stated
    plainly: a feature of the capability map narrower than `ds` along the line
    can be stepped over.  `ds` is 4 mm against a 2 cm atlas lattice, so the
    map's own features are five samples wide; the cut ITSELF is then bisected
    to `tol` (0.01 mm), which is what makes a hard edge land on the transition
    point rather than on a sample.
    """
    p = np.asarray(pts, float)
    cum = cumlen(p)
    L = float(cum[-1])
    if L <= 0:
        b = int(cap.states_at(p[0, 0], p[0, 1])[0])
        return [Atom(0.0, 0.0, b)], p, cum
    n = max(2, int(math.ceil(L / ds)) + 1)
    s = np.linspace(0.0, L, n)
    XY = points_at(p, cum, s)
    bits = cap.states_at(XY[:, 0], XY[:, 1])
    idx = np.flatnonzero(bits[1:] != bits[:-1])
    if not len(idx):
        return [Atom(0.0, L, int(bits[0]))], p, cum
    lo, hi = s[idx].copy(), s[idx + 1].copy()
    blo = bits[idx]
    iters = max(1, int(math.ceil(math.log2(max(ds, tol) / tol))))
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        M = points_at(p, cum, mid)
        same = cap.states_at(M[:, 0], M[:, 1]) == blo
        lo = np.where(same, mid, lo)
        hi = np.where(same, hi, mid)
    cuts = 0.5 * (lo + hi)
    bounds = np.concatenate([[0.0], cuts, [L]])
    vals = np.concatenate([[bits[0]], bits[idx + 1]])
    atoms = [Atom(float(bounds[i]), float(bounds[i + 1]), int(vals[i]))
             for i in range(len(vals)) if bounds[i + 1] - bounds[i] > SPAN_EPS]
    return (atoms or [Atom(0.0, L, int(bits[0]))]), p, cum


# ---------------------------------------------------------------------------
# 6.  THE DYNAMIC PROGRAMME
# ---------------------------------------------------------------------------
_INF = (float("inf"), float("inf"))


def solve_line(atoms: Sequence[Atom], cap: Capability,
               weight: Sequence[float] | None = None,
               balance_w: float = 0.0) -> list[int]:
    """Assign each atom a state so the number of PIECES is smallest. -> [state].

    EXACT, not greedy.  `dp[i][k]` is the cheapest way to have atom `i` drawn by
    state `k`; the only transition cost is 1 per CHANGE, and since that cost is
    the same whatever we change FROM, the O(states^2) relaxation collapses to
    "stay where you are, or switch from the cheapest predecessor" -- O(states)
    a step, which is why a thousand lines cost milliseconds.

    The value is a PAIR and it is compared lexicographically, which is how the
    second objective gets in without ever outranking the first:

        (pieces + balance_w * imbalance,  imbalance)

    With the default `balance_w = 0` the first component is exactly the piece
    count and the second breaks its ties toward the least-loaded state -- free
    balance, no pieces given up.  A caller that would rather trade sets
    `balance_w > 0` and says how many pieces a unit of imbalance is worth.
    `weight[k]` is that unit: the caller's running load on state k, normalised.

    An atom no state can draw is `UNCOVERED`; it is forced, and it BREAKS the
    run, because a stretch nobody can draw is not a hand-over, it is a hole.
    """
    w = [0.0] * cap.n_states if weight is None else list(weight)
    prev: dict[int, tuple[float, float]] = {}
    back: list[dict[int, int]] = []
    for i, at in enumerate(atoms):
        allow = cap.members(at.bits) or (UNCOVERED,)
        cur: dict[int, tuple[float, float]] = {}
        step: dict[int, int] = {}
        if not prev:
            for k in allow:
                own = 0.0 if k == UNCOVERED else w[k] * at.length
                cur[k] = (balance_w * own, own)
                step[k] = k
        else:
            bs = min(prev, key=lambda k: prev[k])
            bv = prev[bs]
            switch = (bv[0] + 1.0, bv[1])
            for k in allow:
                stay = prev.get(k, _INF)
                if stay <= switch:
                    val, par = stay, k
                else:
                    val, par = switch, bs
                own = 0.0 if k == UNCOVERED else w[k] * at.length
                cur[k] = (val[0] + balance_w * own, val[1] + own)
                step[k] = par
        back.append(step)
        prev = cur
    k = min(prev, key=lambda k: prev[k])
    out = [k] * len(atoms)
    for i in range(len(atoms) - 1, 0, -1):
        k = back[i][k]
        out[i - 1] = k
    return out


def brute_force_line(atoms: Sequence[Atom], cap: Capability) -> int:
    """The piece count by exhaustive enumeration. -> int.  For tests only."""
    import itertools
    opts = [cap.members(a.bits) or (UNCOVERED,) for a in atoms]
    best = None
    for combo in itertools.product(*opts):
        n = 1 + sum(1 for a, b in zip(combo, combo[1:]) if a != b)
        best = n if best is None else min(best, n)
    return int(best or 0)


# ---------------------------------------------------------------------------
# 7.  PIECES, SEAMS AND THE MINIMUM
# ---------------------------------------------------------------------------
@dataclass
class Piece:
    """A maximal run of atoms on one line drawn by one (stage, arm) pair."""
    state: int
    stage: int
    arm: int
    s0: float                   # the run's own extent, before any overdraw
    s1: float
    draw0: float                # what the arm is actually told to draw --
    draw1: float                # the run, lapped by delta at overlap seams
    i0: int = 0                 # atom index range [i0, i1)
    i1: int = 0

    @property
    def length(self) -> float:
        return self.draw1 - self.draw0

    @property
    def run_length(self) -> float:
        return self.s1 - self.s0


@dataclass
class Seam:
    """Where two consecutive pieces of one line meet."""
    s: float
    kind: str                   # "overlap" | "hard" | "gap"
    lo: float                   # the overlap zone [lo, hi] the seam sits in
    hi: float
    before: int                 # state index either side
    after: int


@dataclass
class LinePlan:
    index: int
    pts: np.ndarray
    cum: np.ndarray
    atoms: list[Atom]
    assign: list[int]
    pieces: list[Piece]
    seams: list[Seam]

    @property
    def length(self) -> float:
        return float(self.cum[-1])

    @property
    def n_pieces(self) -> int:
        return len(self.pieces)

    def piece_points(self, k: int) -> np.ndarray:
        p = self.pieces[k]
        return sub_polyline(self.pts, self.cum, p.draw0, p.draw1)


def runs_of(atoms: Sequence[Atom], assign: Sequence[int]
            ) -> list[tuple[int, int, int]]:
    """Maximal runs of equal state. -> [(state, i0, i1)] with i1 exclusive."""
    out: list[tuple[int, int, int]] = []
    for i, k in enumerate(assign):
        if out and out[-1][0] == k:
            out[-1] = (k, out[-1][1], i + 1)
        else:
            out.append((k, i, i + 1))
    return out


def absorb_short(atoms: Sequence[Atom], assign: list[int], cap: Capability,
                 min_piece_m: float = MIN_PIECE_M) -> list[int]:
    """Melt runs shorter than `min_piece_m` into a neighbour that can take them.

    A 6 mm piece is not a stroke, it is a hand-over with ink on it: two pen-ups,
    two registrations and a lift for less ink than the lift costs.  If either
    neighbour's state is in EVERY atom of the short run -- which the DP did not
    use because the run's own set offered something the DP liked better -- the
    run is given to the longer neighbour and the two merge.  If neither can, the
    piece stays, because the alternative is not drawing it.

    Absorption can only ever REDUCE the piece count, so it cannot break the DP's
    optimality; what it can do is raise the load on one state, which is why it
    runs before the seams and after the assignment.
    """
    assign = list(assign)
    for _ in range(len(assign) + 1):
        runs = runs_of(atoms, assign)
        lens = [sum(atoms[i].length for i in range(a, b)) for _, a, b in runs]
        hit = False
        for r, (k, a, b) in enumerate(runs):
            if k == UNCOVERED or lens[r] >= min_piece_m:
                continue
            best = None
            for nb in (r - 1, r + 1):
                if not 0 <= nb < len(runs) or runs[nb][0] == UNCOVERED:
                    continue
                st = runs[nb][0]
                if all(atoms[i].bits >> st & 1 for i in range(a, b)):
                    cand = (lens[nb], -abs(nb - r), st)
                    best = cand if best is None or cand > best else best
            if best is not None:
                for i in range(a, b):
                    assign[i] = best[2]
                hit = True
                break
        if not hit:
            break
    return assign


def place_seams(atoms: Sequence[Atom], assign: Sequence[int], cap: Capability,
                overdraw_m: float = OVERDRAW_M
                ) -> tuple[list[Piece], list[Seam]]:
    """Turn an assignment into pieces, and decide where each joint in the ink goes.

    For a boundary between runs A and B at `s*`, the OVERLAP ZONE is the maximal
    stretch around `s*` that BOTH states can draw, clipped to the two runs
    (never take ink from a third piece).  Seam at its middle; each side laps
    `overdraw_m` past the seam, clipped to the zone so nobody is asked to draw
    where it cannot.  An empty zone is a HARD EDGE and the pieces abut exactly.
    """
    runs = runs_of(atoms, assign)
    drawn = [(k, a, b) for (k, a, b) in runs if k != UNCOVERED]
    pieces = [Piece(k, cap.states[k].stage, cap.states[k].arm,
                    atoms[a].s0, atoms[b - 1].s1, atoms[a].s0, atoms[b - 1].s1,
                    a, b) for (k, a, b) in drawn]
    seams: list[Seam] = []
    for r in range(len(runs) - 1):
        kA, aA, bA = runs[r]
        kB, aB, bB = runs[r + 1]
        star = atoms[bA - 1].s1
        if kA == UNCOVERED or kB == UNCOVERED:
            seams.append(Seam(star, "gap", star, star, kA, kB))
            continue
        lo = star
        for i in range(bA - 1, aA - 1, -1):
            if not (atoms[i].bits >> kB & 1):
                break
            lo = atoms[i].s0
        hi = star
        for i in range(aB, bB):
            if not (atoms[i].bits >> kA & 1):
                break
            hi = atoms[i].s1
        kind = "hard" if hi - lo <= SPAN_EPS else "overlap"
        s = 0.5 * (lo + hi)
        seams.append(Seam(s, kind, lo, hi, kA, kB))
        pa = next(p for p in pieces if p.i0 == aA)
        pb = next(p for p in pieces if p.i0 == aB)
        pa.draw1 = min(hi, s + overdraw_m, pb.s1)
        pb.draw0 = max(lo, s - overdraw_m, pa.s0)
    for p in pieces:                     # a lap must never invert a piece
        if p.draw1 < p.draw0:
            p.draw0 = p.draw1 = 0.5 * (p.draw0 + p.draw1)
    return pieces, seams


# ---------------------------------------------------------------------------
# 8.  THE WHOLE PICTURE
# ---------------------------------------------------------------------------
@dataclass
class Options:
    ds: float = SAMPLE_DS
    tol: float = REFINE_TOL
    overdraw_m: float = OVERDRAW_M
    min_piece_m: float = MIN_PIECE_M
    balance_w: float = 0.0
    balance: bool = True        # tie-break toward the least-loaded state


@dataclass
class Plan:
    cap: Capability
    lines: list[LinePlan]
    seconds: float = 0.0
    opts: Options = field(default_factory=Options)

    @property
    def n_pieces(self) -> int:
        return sum(l.n_pieces for l in self.lines)

    def summary(self) -> dict:
        return summarise(self)


def plan_line(pts, cap: Capability, opts: Options | None = None, index: int = 0,
              weight: Sequence[float] | None = None) -> LinePlan:
    opts = opts or Options()
    atoms, p, cum = atoms_of(pts, cap, opts.ds, opts.tol)
    assign = solve_line(atoms, cap, weight, opts.balance_w)
    assign = absorb_short(atoms, assign, cap, opts.min_piece_m)
    pieces, seams = place_seams(atoms, assign, cap, opts.overdraw_m)
    return LinePlan(index, p, cum, atoms, assign, pieces, seams)


def plan_lines(lines: Sequence, cap: Capability, opts: Options | None = None
               ) -> Plan:
    """Cut, assign, absorb and seam every line. -> Plan.

    The lines are independent -- a line's atoms constrain no other line -- so
    the loop IS the optimum for the piece count, and the only thing that crosses
    a line boundary is the load vector the tie-break reads.  Lines are visited
    longest-first so the biggest commitments are made while the load table is
    still flat, which is the whole of the cross-line coupling in this module and
    is exactly as principled as it sounds (see docs/V2_TRACES.md, open
    questions).
    """
    opts = opts or Options()
    t0 = time.perf_counter()
    L = [np.asarray(p, float) for p in lines]
    total = sum(float(cumlen(p)[-1]) for p in L) or 1.0
    load = np.zeros(cap.n_states)
    order = sorted(range(len(L)), key=lambda i: -float(cumlen(L[i])[-1]))
    out: list[LinePlan] = [None] * len(L)       # type: ignore[list-item]
    for i in order:
        w = (load / total) if (opts.balance or opts.balance_w) else None
        lp = plan_line(L[i], cap, opts, i, w)
        for p in lp.pieces:
            load[p.state] += p.length
        out[i] = lp
    return Plan(cap, out, time.perf_counter() - t0, opts)


def summarise(plan: Plan) -> dict:
    """Pieces, loads, hand-overs, holes -- what docs/V2_TRACES.md quotes."""
    cap = plan.cap
    per_line = [l.n_pieces for l in plan.lines]
    load = np.zeros(cap.n_states)
    stage_load: dict[int, float] = {}
    arm_load: dict[int, float] = {}
    kinds = {"overlap": 0, "hard": 0, "gap": 0}
    same_arm = 0     # a "hand-over" an arm makes to itself, one stage later
    ink = uncovered = 0.0
    for l in plan.lines:
        ink += l.length
        for p in l.pieces:
            load[p.state] += p.length
            stage_load[p.stage] = stage_load.get(p.stage, 0.0) + p.length
            arm_load[p.arm] = arm_load.get(p.arm, 0.0) + p.length
        for i, k in enumerate(l.assign):
            if k == UNCOVERED:
                uncovered += l.atoms[i].length
        for s in l.seams:
            kinds[s.kind] += 1
            if s.kind != "gap" and \
                    cap.states[s.before].arm == cap.states[s.after].arm:
                same_arm += 1
    n_lines = len(plan.lines)
    drawn = sum(p.length for l in plan.lines for p in l.pieces)
    # THE MAKESPAN A STAGE SEQUENCE COSTS, in metres of ink: within a stage the
    # active arms are static keep-outs for each other and draw asynchronously,
    # so the stage costs its BUSIEST arm; the barrier between stages makes the
    # sequence cost the sum.  The same accounting `workcell_envelopes.evaluate`
    # uses, with ink metres in place of block cells.
    makespan = 0.0
    for st in range(cap.pattern.n_stages):
        makespan += max((load[k] for k, c in enumerate(cap.states)
                         if c.stage == st), default=0.0)
    return dict(
        pattern=cap.pattern.name,
        n_stages=cap.pattern.n_stages,
        n_states=cap.n_states,
        merged_stages=[list(m) for m in cap.merged],
        n_lines=n_lines,
        n_atoms=sum(len(l.atoms) for l in plan.lines),
        n_pieces=sum(per_line),
        pieces_per_line=dict(
            mean=(sum(per_line) / n_lines if n_lines else 0.0),
            max=max(per_line, default=0),
            one=sum(1 for n in per_line if n == 1),
            zero=sum(1 for n in per_line if n == 0)),
        handovers=kinds["overlap"] + kinds["hard"],
        handovers_overlap=kinds["overlap"],
        handovers_hard=kinds["hard"],
        handovers_same_arm=same_arm,
        gaps=kinds["gap"],
        ink_m=ink,
        drawn_m=drawn,
        uncovered_m=uncovered,
        covered_frac=(1.0 - uncovered / ink) if ink else 1.0,
        overdraw_m=drawn - (ink - uncovered),
        load_m={cap.label(k): round(float(load[k]), 4)
                for k in range(cap.n_states)},
        stage_load_m={str(k): round(v, 4) for k, v in sorted(stage_load.items())},
        arm_load_m={str(k): round(v, 4) for k, v in sorted(arm_load.items())},
        stage_makespan_m=round(float(makespan), 4),
        seconds=round(plan.seconds, 4),
    )


def format_summary(d: dict) -> str:
    out = [f"  pattern            {d['pattern']}  "
           f"({d['n_stages']} stages, {d['n_states']} states)",
           f"  lines / atoms      {d['n_lines']} / {d['n_atoms']}",
           f"  PIECES             {d['n_pieces']}  "
           f"(mean {d['pieces_per_line']['mean']:.2f} a line, "
           f"max {d['pieces_per_line']['max']}, "
           f"{d['pieces_per_line']['one']} lines in one)",
           f"  hand-overs         {d['handovers']}  "
           f"({d['handovers_overlap']} in an overlap, "
           f"{d['handovers_hard']} at a hard edge; "
           f"{d['handovers_same_arm']} are the same arm in a later stage)",
           f"  gaps (no drawer)   {d['gaps']}",
           f"  ink                {d['ink_m']:.3f} m, drawn {d['drawn_m']:.3f} m, "
           f"uncovered {d['uncovered_m']:.3f} m "
           f"({100 * d['covered_frac']:.2f} % covered)",
           f"  overdraw           {1000 * d['overdraw_m']:.1f} mm total",
           "  load per (stage, arm), metres:"]
    for k, v in d["load_m"].items():
        out.append(f"      {k:<14} {v:8.3f}")
    out.append("  per stage: " + "  ".join(
        f"s{k}={v:.2f}" for k, v in d["stage_load_m"].items()))
    out.append("  per arm:   " + "  ".join(
        f"{k}={v:.2f}" for k, v in d["arm_load_m"].items()))
    out.append(f"  stage makespan (busiest arm a stage, summed)  "
               f"{d['stage_makespan_m']:.2f} m")
    out.append(f"  time               {1000 * d['seconds']:.1f} ms")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 9.  THE PICTURES THIS IS MEASURED ON
# ---------------------------------------------------------------------------
def load_lines(path) -> list[np.ndarray]:
    """Polylines in PAPER metres, out of whatever JSON the repo wrote them in.

    Understands a programme (`out/csail_program_*.json`, strokes under
    `strokes`/`segments` with `pts`/`points`/`xy`), a GUI job's traced strokes,
    and a bare list of point lists.
    """
    obj = json.loads(Path(path).read_text())

    def as_line(o):
        if isinstance(o, dict):
            for k in ("pts", "points", "xy", "polyline", "pts_xy"):
                if k in o:
                    return np.asarray(o[k], float)
            return None
        a = np.asarray(o, float)
        return a if a.ndim == 2 and a.shape[1] >= 2 else None

    def harvest(o):
        if isinstance(o, dict):
            for k in ("strokes", "lines", "segments", "polylines", "traces"):
                if k in o and isinstance(o[k], list):
                    got = [as_line(e) for e in o[k]]
                    got = [g[:, :2] for g in got if g is not None and len(g) > 1]
                    if got:
                        return got
            for v in o.values():
                got = harvest(v)
                if got:
                    return got
        elif isinstance(o, list):
            got = [as_line(e) for e in o]
            got = [g[:, :2] for g in got if g is not None and len(g) > 1]
            if got:
                return got
        return []

    got = harvest(obj)
    if not got:
        raise ValueError(f"no polylines found in {path}")
    return got


def synthetic(kind: str, n: int = 1000, seed: int = 0) -> list[np.ndarray]:
    """Deterministic test pictures that cover the certified block.

    `strokes` -- n long random polylines, each a smooth 6-point wander;
    `hatch`   -- n parallel lines at 15 degrees, wall to wall;
    `scribble`-- n short curved arcs scattered over the block.
    """
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = BLOCK
    out: list[np.ndarray] = []
    if kind == "strokes":
        for _ in range(n):
            a = rng.uniform([x0, y0], [x1, y1])
            th = rng.uniform(0, 2 * np.pi)
            L = rng.uniform(0.30, 1.20)
            t = np.linspace(0, L, 6)
            P = a + np.column_stack([t * np.cos(th), t * np.sin(th)])
            P += rng.normal(0, 0.02, P.shape) * np.linspace(0, 1, 6)[:, None]
            out.append(np.clip(P, [x0, y0], [x1, y1]))
    elif kind == "hatch":
        th = np.deg2rad(15.0)
        d = np.array([np.cos(th), np.sin(th)])
        nrm = np.array([-d[1], d[0]])
        c = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        span = np.hypot(x1 - x0, y1 - y0)
        for k in np.linspace(-span / 2, span / 2, n):
            P = np.array([c + nrm * k - d * span, c + nrm * k + d * span])
            P = _clip_to_rect(P, BLOCK)
            if P is not None:
                out.append(P)
    elif kind == "scribble":
        for _ in range(n):
            a = rng.uniform([x0, y0], [x1, y1])
            r = rng.uniform(0.03, 0.14)
            t0 = rng.uniform(0, 2 * np.pi)
            t = t0 + np.linspace(0, rng.uniform(1.0, 4.0), 24)
            P = a + np.column_stack([r * np.cos(t), r * np.sin(t)])
            out.append(np.clip(P, [x0, y0], [x1, y1]))
    else:
        raise ValueError(kind)
    return [p for p in out if cumlen(p)[-1] > 1e-6]


def _clip_to_rect(seg: np.ndarray, rect: Rect):
    """Liang-Barsky, for the hatch. -> (2, 2) or None."""
    (x0, y0, x1, y1) = rect
    p0, p1 = seg[0], seg[1]
    d = p1 - p0
    t0, t1 = 0.0, 1.0
    for (pp, qq) in ((-d[0], p0[0] - x0), (d[0], x1 - p0[0]),
                     (-d[1], p0[1] - y0), (d[1], y1 - p0[1])):
        if abs(pp) < 1e-12:
            if qq < 0:
                return None
            continue
        r = qq / pp
        if pp < 0:
            t0 = max(t0, r)
        else:
            t1 = min(t1, r)
    if t0 >= t1:
        return None
    return np.array([p0 + t0 * d, p0 + t1 * d])


# ---------------------------------------------------------------------------
# 10.  CLI
# ---------------------------------------------------------------------------
def _run(name, lines, cap, opts, js):
    plan = plan_lines(lines, cap, opts)
    d = plan.summary()
    d["case"] = name
    print(f"\n{name}")
    print(format_summary(d))
    js.append(d)
    return plan


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--atlas", default="out/atlas_proposed_h0970_lat0860_gated63")
    ap.add_argument("--lines", action="append", default=[],
                    help="a JSON file of polylines in paper metres (repeatable)")
    ap.add_argument("--synthetic", action="append", default=[],
                    help="kind[:n]  -- strokes | hatch | scribble")
    ap.add_argument("--dead-band", type=float, default=DEAD_BAND_M)
    ap.add_argument("--ds", type=float, default=SAMPLE_DS)
    ap.add_argument("--overdraw", type=float, default=OVERDRAW_M)
    ap.add_argument("--min-piece", type=float, default=MIN_PIECE_M)
    ap.add_argument("--balance-w", type=float, default=0.0)
    ap.add_argument("--no-balance", action="store_true")
    ap.add_argument("--gate", choices=("strict", "flat"), default="strict")
    ap.add_argument("--erode", action="store_true")
    ap.add_argument("--tilt-max-deg", type=float, default=0.0)
    ap.add_argument("--stale-ok", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    cov = coverage_from_atlas(a.atlas, gate=a.gate, erode=a.erode,
                              tilt_max_deg=a.tilt_max_deg,
                              require_current=not a.stale_ok)
    print(f"coverage from {a.atlas} [{a.gate}"
          f"{', eroded' if a.erode else ''}]: " + "  ".join(
              f"{k}={cov.area_m2(k):.3f} m2" for k in cov.arms))
    pats = [zigzag_pattern(a.dead_band), single_stage_pattern()]
    opts = Options(ds=a.ds, overdraw_m=a.overdraw, min_piece_m=a.min_piece,
                   balance_w=a.balance_w, balance=not a.no_balance)
    jobs: list[tuple[str, list[np.ndarray]]] = []
    for p in a.lines:
        jobs.append((Path(p).name, load_lines(p)))
    for s in a.synthetic:
        kind, _, n = s.partition(":")
        jobs.append((f"synthetic {kind} x{n or 1000}",
                     synthetic(kind, int(n or 1000))))
    js: list[dict] = []
    for nm, lines in jobs:
        tot = sum(float(cumlen(p)[-1]) for p in lines)
        print(f"\n=== {nm}: {len(lines)} lines, {tot:.2f} m of ink ===")
        for pat in pats:
            _run(f"{nm}  [{pat.name}]", lines, capability(cov, pat), opts, js)
    if a.json:
        Path(a.json).write_text(json.dumps(js, indent=1))
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
