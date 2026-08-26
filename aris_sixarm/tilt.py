"""Pen ORIENTATION freedom: tilt within a material cone, for one stroke.

The shipping planner pins the pen perpendicular to the paper (`planner.py`:
R = rotx(pi), yaw 0) and lets q7 carry the whole 1-D self-motion.  A real pen
does not need to be perpendicular: most media draw happily at a lean of 10-20
degrees, and the drawing is identical as long as the TIP still traces the
curve.  This module adds that lean as a planning axis.

WHAT THE FREEDOM ACTUALLY IS.  Constrain the tip to a point and the pen AXIS to
a direction and the task has 5 DOF, not 6 — the arm has 7, so the self-motion
is 2-D.  One of those dimensions is the spin of the tool about the pen axis,
which is EXACTLY q7 (see `planner.py`'s "WHY NO YAW AXIS": joint 7 rotates the
flange about the tool z, the TCP lies on that axis, so q7 -> q7 + d maps the
pose (R, p) to (R @ rotz(d), p) with the tip and the pen axis untouched).  So
fixing a tool-frame CONVENTION per pen-axis direction and letting q7 range over
its grid covers that whole dimension exactly once, with no aliasing and no
double counting.  The other dimension is the pen-axis direction itself, which
lives on a disc: the set of unit vectors within `tilt_max` of straight down.

WHY THE TILT IS A VECTOR AND NEVER (theta, phi).  (theta, phi) has a
coordinate singularity at the apex: every phi names the same pen axis at
theta = 0, so a grid in those coordinates puts N nodes on one pose, gives them
N different tool frames, and lets a +-1-index DP window "travel" round the apex
for free.  The vector (tx, ty) = theta * (cos phi, sin phi) has no such point.
It is the rotation-vector chart:

    w = (ty, -tx, 0),   R_tilt = exp(hat(w)),   R = R_tilt @ rotx(pi)

so tx leans the pen toward +x of the canvas, ty toward +y, (0, 0) is exactly
perpendicular, and R is an ANALYTIC function of (tx, ty) everywhere on the
disc.  The convention is complete: `tilt_rot` is single-valued, so each
(tx, ty) names one tool frame and q7 sweeps the spin around it.

THE DISC IS SAMPLED ON A HEX LATTICE, not on rings-of-(theta, phi) and not on
tool-x/tool-y axes (which is what `atlas.py`'s tilt rescue does, and it biases
the answer toward four compass directions).  A hex lattice has one neighbour
distance, six neighbours everywhere, and its Euclidean rings inside radius
`tilt_max` land at 19 points (n_ring = 2) or 37 (n_ring = 3).  Adjacency for
the DP is "same cell or one of the six neighbours".

THE LATTICE IS (s x q7 x tilt-point x branch) and still a DAG in s: edges only
join consecutive arc-length steps, |dq7| <= 1 grid index, tilt to itself or a
hex neighbour, and ||dq||_inf <= JUMP_THRESH as ever.  One DP sweep is
globally optimal.

TWO WAYS TO USE IT, AND THEY COST THREE ORDERS OF MAGNITUDE APART.

  * THE ORACLE (`build_lattice(..., cells=None)`): materialise every tilt point
    at every arc-length step.  ~20-40x the IK of the flat lattice, seconds per
    stroke.  This is the ground truth the evidence in docs/TILT_EXPLORATION.md
    is measured against; it is NOT a shipping path.
  * THE ADAPTIVE PLANNER (`plan_adaptive`): plan flat first — literally today's
    lattice, at today's cost — and open the tilt dimensions ONLY on the
    arc-length collar around wherever the flat plan died or ran thin, and only
    on the coarse ring of the disc until a winner shows up.  `extend_lattice`
    is the primitive: it solves the (s, tilt) COLUMNS that were not
    materialised before and merges them in, so a lattice grows toward the
    answer instead of being paid for up front.

Everything here is inert unless asked for: `tilt_max_deg = 0` collapses the
disc to its single centre point and reproduces the flat lattice node for node.
"""
import numpy as np

from . import ik, planner, rig_final
from .frames import FR3_MIN, FR3_MAX, PEN_EXT, rotx
from .pwl import MARGIN_GATE, SIGMA_GATE

OBJECTIVE = "min_travel"         # default band objective; see `plan_lattice`
OBJECTIVES = ("min_travel", "maximin_sigma")
N_RING = 2               # hex rings on the disc: 2 -> 19 points, 3 -> 37
W_TILT = 0.05            # tie-break cost per rad^2 of lean (artwork quality)
JUMP_SEARCH = 0.25       # rad, the DP's edge budget when tilt is open.
# WHY IT IS TIGHTER THAN THE EXECUTION GATE.  planner.JUMP_THRESH = 0.35 is
# what a finished trajectory must satisfy between dense samples; using the same
# number as the SEARCH edge test was fine while q7 was the only moving axis,
# but with tilt open two axes move at once and the DP starts taking edges whose
# endpoints are 0.35 apart across an IK branch boundary — legal on the grid,
# and `ik.solve_cc` will not walk them, so the certification chase leaves the
# planned path and dies.  Measured on the R bowl at tilt 15: at 0.35 the DP
# path chases to ||dq||_inf = 2.56 rad and stops on the margin gate; at 0.25
# the same lattice certifies end to end with min margin 0.308 (the flat plan's
# is 0.211).  Reserving 0.10 rad of headroom between what the search may
# propose and what the certificate demands is the whole fix.
COLLAR_M = 0.08          # m of arc length opened either side of a weak cell
THIN_FRAC = 0.20         # a cell is "thin" below this fraction of the median
                         # free-fiber width along the stroke

_HEX_DIRS = ((1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1))


# --------------------------------------------------------------------------
# 1. the disc: hex sampling and its adjacency
# --------------------------------------------------------------------------
def hex_disc(tilt_max_deg, n_ring=N_RING, pitch_deg=None):
    """Hex-lattice sampling of the tilt disc |t| <= tilt_max.

    The lattice pitch is `tilt_max / n_ring`, so the outermost cells sit
    exactly ON the cone and nothing outside it is ever proposed.  Points are
    ordered by radius then angle, so index 0 is always the perpendicular pen
    and a prefix of the list is a coarser disc (which is what the adaptive
    planner's coarse-to-fine stage refines through).

    `pitch_deg` FIXES the lattice spacing instead, so that a bigger cone is a
    strict SUPERSET of a smaller one.  This matters for any 0/15/30 comparison:
    `atlas._candidates` builds its cone as {tilt_max/2, tilt_max}, which makes
    its 30-degree candidate set {15, 30} and its 15-degree set {7.5, 15} — NOT
    nested, and that alone can make a wider cone score worse (it is why the
    shipped tilt study's 45-degree column is non-monotone).  With a fixed
    pitch, tilt <= 30 contains every pose tilt <= 15 offered, so a loss is a
    real loss.

    Returns dict:
        tilt      (Nt,2) the (tx, ty) rotation-vector components, rad
        nbr       (Nt,7) int32: column 0 is the point itself, columns 1..6 its
                  hex neighbours inside the disc, -1 where a neighbour falls
                  outside.  THE DP'S TILT EDGE SET, in one array.
        ij        (Nt,2) axial hex coordinates (debug/plotting)
        ring      (Nt,)  which hex ring each point is on (0 = centre)
        pitch     the lattice spacing in rad
        tilt_max  the cone half-angle in rad
    """
    tmax = np.deg2rad(float(tilt_max_deg))
    if pitch_deg is not None and tmax > 0:
        n_ring = max(1, int(round(float(tilt_max_deg) / float(pitch_deg))))
    if tmax <= 0 or n_ring <= 0:
        # Column 0 is the point ITSELF and must stay 0 even here: it is the
        # "tilt does not change" edge, and without it the perpendicular-only
        # disc has no edges at all and every DP cuts at step 1.
        nbr = np.full((1, 7), -1, np.int32)
        nbr[0, 0] = 0
        return dict(tilt=np.zeros((1, 2)), nbr=nbr,
                    ij=np.zeros((1, 2), int), ring=np.zeros(1, int),
                    pitch=0.0, tilt_max=0.0)
    a = tmax / n_ring
    cells = []
    for i in range(-n_ring, n_ring + 1):
        for j in range(-n_ring, n_ring + 1):
            # hex ring index in axial coordinates
            ring = (abs(i) + abs(j) + abs(i + j)) // 2
            if ring > n_ring:
                continue
            x, y = a * (i + 0.5 * j), a * (np.sqrt(3) / 2) * j
            r = float(np.hypot(x, y))
            if r > tmax + 1e-12:
                continue                      # outside the material cone
            cells.append((r, float(np.arctan2(y, x)) % (2 * np.pi), i, j, x, y,
                          ring))
    cells.sort(key=lambda c: (round(c[0], 12), c[1]))
    ij = np.array([[c[2], c[3]] for c in cells], int)
    tilt = np.array([[c[4], c[5]] for c in cells], float)
    ring = np.array([c[6] for c in cells], int)
    index = {(int(i), int(j)): k for k, (i, j) in enumerate(ij)}
    nbr = np.full((len(cells), 7), -1, np.int32)
    nbr[:, 0] = np.arange(len(cells))
    for k, (i, j) in enumerate(ij):
        for d, (di, dj) in enumerate(_HEX_DIRS, start=1):
            nbr[k, d] = index.get((int(i) + di, int(j) + dj), -1)
    return dict(tilt=tilt, nbr=nbr, ij=ij, ring=ring, pitch=float(a),
                tilt_max=float(tmax))


def sub_disc(disc, m):
    """The hex SUBLATTICE of pitch `m` x the disc's own. -> (indices, nbr).

    THE COARSE STAGE MUST SPAN THE CONE, NOT THE MIDDLE OF IT.  The obvious
    coarse-to-fine ladder — "hex ring <= 1, then everything" — samples only out
    to one pitch, which at pitch 7.5 deg means the first stage can lean 7.5 deg
    and no further.  A donut that needs 15 deg then fails stage 0 for a reason
    that has nothing to do with how finely the disc is sampled, and every
    rescue pays for both stages.

    The sublattice fixes that: taking every m-th hex cell keeps the CENTRE and
    the six cells on the cone edge (m = n_ring gives exactly 7 points at the
    full lean), and it is a strict SUBSET of the fine disc, so nothing solved
    here is wasted if the fine stage runs.  Adjacency is the sublattice's own —
    a coarse step moves a full m pitches — and it is returned as a neighbour
    table indexed by FINE disc index so `plan_lattice` needs no other change.
    """
    ij = np.asarray(disc["ij"], int)
    key = {(int(a), int(b)): k for k, (a, b) in enumerate(ij)}
    m = max(1, int(m))
    sel = np.array([k for k, (a, b) in enumerate(ij)
                    if a % m == 0 and b % m == 0], int)
    nbr = np.full((len(ij), 7), -1, np.int32)
    for k in sel:
        a, b = int(ij[k, 0]), int(ij[k, 1])
        nbr[k, 0] = k
        for d, (da, db) in enumerate(_HEX_DIRS, start=1):
            nbr[k, d] = key.get((a + da * m, b + db * m), -1)
    return sel, nbr


def coarse_prefix(disc, n_ring):
    """Indices of the points of `disc` on hex ring <= n_ring (centre first).

    The disc is radius-ordered, so this is a prefix only when the coarse ring
    happens to be the innermost Euclidean shell; it is computed from the hex
    ring instead, which is what the neighbour structure is built on.
    """
    return np.flatnonzero(disc["ring"] <= int(n_ring))


# --------------------------------------------------------------------------
# 2. the pen convention
# --------------------------------------------------------------------------
def tilt_rot(tilt):
    """(N,2) tilt vectors -> (N,3,3) R_tilt = exp(hat((ty, -tx, 0))).

    Rodrigues, with the identity taken at |w| = 0 — the chart is regular there
    (that is the whole point of the vector parameterisation), the FORMULA just
    has a removable 0/0.
    """
    t = np.asarray(tilt, float).reshape(-1, 2)
    w = np.column_stack([t[:, 1], -t[:, 0], np.zeros(len(t))])
    th = np.linalg.norm(w, axis=1)
    k = np.where(th[:, None] > 1e-12, w / np.where(th[:, None] > 1e-12,
                                                   th[:, None], 1.0), 0.0)
    K = np.zeros((len(t), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    s, c = np.sin(th)[:, None, None], (1 - np.cos(th))[:, None, None]
    return np.eye(3)[None] + s * K + c * (K @ K)


def pen_rot(tilt):
    """(N,2) -> (N,3,3) the WORLD tool rotation R = R_tilt @ rotx(pi).

    Column 2 is the pen axis (TCP -> tip): (0,0,-1) at zero tilt.
    """
    return tilt_rot(tilt) @ rotx(np.pi)


def pen_poses(pts_xy, tilt, Twb_inv, pen_ext):
    """(N,4,4) base-frame hand-TCP poses whose PEN TIP is exactly on pts_xy.

    `tilt` is (N,2), or (2,) broadcast to every point, or None for the
    perpendicular pen — in which case this is `pwl.pen_down_poses` verbatim.
    """
    p = np.asarray(pts_xy, float).reshape(-1, 2)
    t = np.zeros((len(p), 2)) if tilt is None else \
        np.broadcast_to(np.asarray(tilt, float).reshape(-1, 2), (len(p), 2))
    R = pen_rot(t)
    T = np.tile(np.eye(4), (len(p), 1, 1))
    T[:, :3, :3] = R
    T[:, :3, 3] = np.column_stack([p[:, 0], p[:, 1], np.zeros(len(p))]) \
        - pen_ext * R[:, :, 2]
    return np.asarray(Twb_inv, float) @ T


def cone_check(qs, spec, tilt_max_deg, h_inv=None):
    """Per-sample pen lean (deg) from vertical IN WORLD, and the worst one.

    The check a tilted plan needs and a flat one does not: the tip-on-curve
    test in `validate.py` is ALREADY orientation-aware (it FKs the tool frame
    and steps `pen_ext` along the tool z), so a tilted plan passes it
    unchanged — what nothing downstream currently checks is that the lean
    stayed inside the MATERIAL cone the artist allowed.
    """
    from .fleet import H_INV_DEFAULT
    from .frames import fk_many
    Twb = spec.T_world_base(H_INV_DEFAULT if h_inv is None else h_inv)
    T, _ = fk_many(np.asarray(qs, float).reshape(-1, 7))
    axis_w = (Twb[:3, :3] @ T[:, :3, 2].T).T          # pen axis in world
    # arctan2 of (sideways, down), NOT arccos of the down component: arccos is
    # ill-conditioned exactly where a flat plan lives, and reports ~1e-8 rad of
    # lean for a pen that is perpendicular to the last bit of double precision.
    deg = np.rad2deg(np.arctan2(np.linalg.norm(axis_w[:, :2], axis=1),
                                -axis_w[:, 2]))
    return deg, float(deg.max()), bool(deg.max() <= float(tilt_max_deg) + 1e-6)


# --------------------------------------------------------------------------
# 3. the lattice, materialised by (s, tilt) COLUMN
# --------------------------------------------------------------------------
def _setup(spec, h_inv, n_q7):
    from .fleet import H_INV_DEFAULT
    Twb = spec.T_world_base(H_INV_DEFAULT if h_inv is None else h_inv)
    return (Twb, np.linalg.inv(Twb),
            np.linspace(FR3_MIN[6] + 0.05, FR3_MAX[6] - 0.05, n_q7))


def empty_lattice(pts_xy, spec, disc, h_inv=None, pen_ext=PEN_EXT,
                  n_q7=planner.N_Q7):
    """An all-invalid (s x q7 x tilt x branch) lattice, ready to be filled."""
    Twb, Twb_inv, q7s = _setup(spec, h_inv, n_q7)
    pts = np.asarray(pts_xy, float)
    Ns, Nt, Nb = len(pts), len(disc["tilt"]), planner.N_BRANCH
    shape = (Ns, n_q7, Nt, Nb)
    return dict(Q=np.full((*shape, 7), np.nan), valid=np.zeros(shape, bool),
                margin=np.full(shape, -1.0), sigma=np.full(shape, -1.0),
                cells=np.zeros((Ns, Nt), bool), q7s=q7s, disc=disc,
                Twb=Twb, Twb_inv=Twb_inv, pts=pts, pen_ext=pen_ext, spec=spec,
                n_ik=0, n_solved_cells=0)


def extend_lattice(lat, cells, clearance=True):
    """Materialise the (s, tilt) COLUMNS marked in `cells` that are not yet in.

    A "column" is one arc-length step at one tilt point: `n_q7 x N_BRANCH`
    lattice nodes sharing a single target pose.  Columns already solved are
    skipped, so this is idempotent and the adaptive planner can ask for a
    collar without tracking what it already paid for.

    The gates are exactly `planner.build_lattice`'s — margin >= HARD_MARGIN,
    the paper / boom / frame clearances, sigma_min >= HARD_SIGMA — applied in
    the same shrinking-index-set order, so a tilt-0 column here is bit for bit
    the tilt-0 column there.

    Returns the number of columns newly solved.
    """
    want = np.asarray(cells, bool) & ~lat["cells"]
    ii, tt = np.nonzero(want)
    if not len(ii):
        return 0
    spec, disc, pen_ext = lat["spec"], lat["disc"], lat["pen_ext"]
    n_q7 = len(lat["q7s"])
    Nb = planner.N_BRANCH
    Twb, Twb_inv = lat["Twb"], lat["Twb_inv"]

    # (a) one pose per requested column, then one q7 row per pose
    T_b = pen_poses(lat["pts"][ii], disc["tilt"][tt], Twb_inv, pen_ext)
    flat = np.repeat(ik._flat16(T_b), n_q7, axis=0)
    q7 = np.tile(lat["q7s"], len(ii))

    # (b) one batched solve, FK-verified in C++, re-filtered against FR3 limits
    Q, valid = ik.solve_batch(flat, q7, spec.q_seed)
    lat["n_ik"] += len(flat)
    lat["n_solved_cells"] += len(ii)

    # global node indices of this block, so the gates can shrink an index set
    # exactly the way the flat lattice's do
    Nq_t = n_q7 * len(disc["tilt"]) * Nb
    base = (ii[:, None] * Nq_t + np.arange(n_q7)[None, :] * len(disc["tilt"]) * Nb
            + tt[:, None] * Nb)                          # (K, n_q7)
    gidx = (base[:, :, None] + np.arange(Nb)[None, None, :]).reshape(-1)

    keep = valid.reshape(-1)
    idx, q = gidx[keep], Q.reshape(-1, 7)[keep]

    m = np.min(np.minimum(q - FR3_MIN, FR3_MAX - q), axis=1)
    k = m >= planner.HARD_MARGIN
    idx, q, m = idx[k], q[k], m[k]

    # THE GATES ARE ANDs OVER INDEPENDENT PER-NODE QUANTITIES, so the order is
    # free and only the cost differs (`planner._build_lattice_batch` says the
    # same).  It puts sigma last because on the flat lattice the clearance
    # stage is cheap; with tilt open the frame check is the dominant cost and
    # one batched SVD is not, so sigma goes first and the steel only ever sees
    # postures that already control.
    if clearance and len(idx):
        T, p = ik.fk_batch(q)
        pw = p @ Twb[:3, :3].T + Twb[:3, 3]
        k = pw[:, 1:, 2].min(axis=1) >= planner.Z_PAPER
        if spec.mount == "inv" and getattr(spec, "rig", "sixarm") == "sixarm":
            rb = np.hypot(p[:, :, 0], p[:, :, 1])
            k &= ~np.any((p[:, :, 2] < planner.BOOM_Z)
                         & (rb < planner.BOOM_R), axis=1)
        idx, q, m, T, pw = idx[k], q[k], m[k], T[k], pw[k]
    else:
        T = pw = None

    if len(idx):
        s = np.linalg.svd(ik.tip_jacobian_batch(q, pen_ext=pen_ext),
                          compute_uv=False)[:, -1]
        k = s >= planner.HARD_SIGMA
        idx, q, m, s = idx[k], q[k], m[k], s[k]
        if T is not None:
            T, pw = T[k], pw[k]
    else:
        s = np.zeros(0)

    if clearance and len(idx) and T is not None:
        boxes = spec.static_obstacles() if hasattr(spec, "static_obstacles") \
            else []
        if boxes:
            tip = T[:, :3, 3] + T[:, :3, :3] @ np.array([0.0, 0.0, pen_ext])
            tip_w = tip @ Twb[:3, :3].T + Twb[:3, 3]
            P10 = np.concatenate([pw, tip_w[:, None, :]], axis=1)
            k = _frame_clear(P10, boxes)
            idx, q, m, s = idx[k], q[k], m[k], s[k]

    if len(idx):                       # ...and the arm against itself
        from . import selfcoll
        k = selfcoll.self_ok(q, margin=selfcoll.SELF_PLAN_MARGIN,
                             pen_ext=pen_ext)
        idx, q, m, s = idx[k], q[k], m[k], s[k]

    Qf = lat["Q"].reshape(-1, 7)
    Qf[idx] = q
    lat["valid"].reshape(-1)[idx] = True
    lat["margin"].reshape(-1)[idx] = m
    lat["sigma"].reshape(-1)[idx] = s
    lat["cells"][ii, tt] = True
    return len(ii)


def _frame_clear(P10, boxes, margin=None):
    """(N,10,3) chain points -> (N,) bool "clears the frame", screened first.

    `rig_final.chain_static_clearance` is EXACT and it is the whole cost of a
    tilt column: it ternary-searches each capsule against each box for 36
    iterations, and profiling the adaptive planner on a donut stroke put 73 %
    of the wall clock inside it (the analytic IK it exists to gate was 2 %).

    Almost every node is nowhere near the steel, and two cheap bounds settle
    those without any search.  For a capsule (A, B, r) against a box set, with
    dA / dB the point distances and L = |B - A|:

        d_seg <= min(dA, dB)                 -> UPPER bound  (reject if < margin)
        d_seg >= min(dA, dB) - L/2           -> LOWER bound  (accept if >= margin)

    Both come from `rig_final.box_clearance`, which is vectorised over points
    x boxes with no search at all.  Only the nodes the two bounds straddle are
    handed to the exact routine, so this is a pure speed change: the accept
    and reject sets are proved, not approximated, and
    `tests/test_tilt.py::test_frame_clear_matches_exact` pins the mask against
    `chain_static_clearance` node for node.
    """
    margin = rig_final.STATIC_MARGIN if margin is None else margin
    P10 = np.asarray(P10, float)
    N = len(P10)
    if not N:
        return np.zeros(0, bool)
    # Only the points a capsule actually ends on are worth measuring, and the
    # min over boxes is accumulated ONE BOX AT A TIME.  `rig_final.box_clearance`
    # is vectorised the other way — it builds an (N x boxes x 3) array — which
    # on a tilt column is tens of millions of floats and became the screen's
    # own hotspot once it had removed everyone else's.
    used = sorted({i for c in rig_final.STATIC_CAPSULES for i in c[:2]})
    col = {k: n for n, k in enumerate(used)}
    dp = np.full((N, len(used)), np.inf)
    for b in boxes:
        lo, hi = b["lo"], b["hi"]
        for k in used:
            P = P10[:, k]
            d = np.maximum(np.maximum(lo - P, P - hi), 0.0)
            np.minimum(dp[:, col[k]], np.sqrt(np.einsum("ij,ij->i", d, d)),
                       out=dp[:, col[k]])
    ub = np.full(N, np.inf)
    lb = np.full(N, np.inf)
    for i, j, r in rig_final.STATIC_CAPSULES:
        ends = np.minimum(dp[:, col[i]], dp[:, col[j]])
        L = np.linalg.norm(P10[:, j] - P10[:, i], axis=1)
        ub = np.minimum(ub, ends - r)
        lb = np.minimum(lb, ends - 0.5 * L - r)
    out = np.zeros(N, bool)
    out[lb >= margin] = True                       # proved clear
    todo = np.flatnonzero((lb < margin) & (ub >= margin))
    if len(todo):
        out[todo] = (rig_final.chain_static_clearance(P10[todo], boxes)
                     >= margin)
    return out                                     # ub < margin stays False


def build_lattice(pts_xy, spec, tilt_max_deg=0.0, n_ring=N_RING, h_inv=None,
                  pen_ext=PEN_EXT, n_q7=planner.N_Q7, clearance=True,
                  cells=None, disc=None):
    """THE ORACLE: the whole (s x q7 x tilt x branch) lattice in one solve.

    `cells` (Ns, Nt) bool restricts which columns are materialised; None means
    all of them.  With `tilt_max_deg = 0` the disc is one point and the result
    is the flat lattice — the same nodes, the same gates, the same numbers as
    `planner.build_lattice`, which `tests/test_tilt.py` pins.
    """
    disc = hex_disc(tilt_max_deg, n_ring) if disc is None else disc
    lat = empty_lattice(pts_xy, spec, disc, h_inv, pen_ext, n_q7)
    if cells is None:
        cells = np.ones(lat["cells"].shape, bool)
    extend_lattice(lat, cells, clearance=clearance)
    return lat


def free_mask(lat, sigma_gate=SIGMA_GATE, margin_gate=MARGIN_GATE):
    """The band-gated node set: `pwl.band_free`, on the 4-D lattice."""
    return (lat["valid"] & (lat["sigma"] >= sigma_gate)
            & (lat["margin"] >= margin_gate))


# --------------------------------------------------------------------------
# 4. the DP: one sweep over a DAG in s
# --------------------------------------------------------------------------
def plan_lattice(lat, free=None, jump=planner.JUMP_THRESH, w_smooth=4.0,
                 w_tilt=W_TILT, sigma_gate=SIGMA_GATE,
                 margin_gate=MARGIN_GATE, objective=OBJECTIVE, nbr=None):
    """DP over (s x q7 x tilt x branch): one sweep, globally optimal.

    TWO OBJECTIVES, AND THE GATES ARE HARD IN BOTH.  `free` is the admissible
    node set; neither objective can trade a gate for anything, which is what
    makes "raise margin_gate to 0.30 and ask whether a plan still exists" the
    honest way to pose the donut question.

      "maximin_sigma"  maximise the smallest sigma_min along the path, ties
          broken by sum w_smooth*||dq||^2 + w_tilt*||tilt||^2.  This is
          `planner.plan`'s objective, and with a tilt axis open it will spend a
          LOT of joint travel to buy a little bottleneck: measured on the rim
          arc at tilt 15, sigma 0.196 -> 0.212 for travel 5.40 -> 19.22 rad.
          Travel is the clock (`writing.draw_duration` stretches the ink until
          no joint exceeds its velocity cap), so that is not a free win.
      "min_travel"     minimise total joint travel sum ||dq||_1 over the gated
          free region, plus the tilt penalty.  The same objective `pwl.py`
          offers on the flat band, and the right default once tilt widens the
          feasible set: with more room to move, "shortest" is a better ask than
          "flattest bottleneck".

    Exactly `planner.plan`'s lexicographic bottleneck DP with a tilt axis
    added to the fiber:  V[n] = min(sigma[n], max_{p->n} V[p]), ties broken by
    sum w_smooth*||dq||^2 + w_tilt*||tilt||^2.  The tilt term is what makes a
    plan prefer the perpendicular pen when leaning buys nothing.

    THE CANDIDATE AXIS IS UNIFORM AND THAT IS WHY THIS IS ALSO THE ADAPTIVE
    PLANNER'S DP.  Every target node has the same 3 (dq7) x 7 (tilt: itself or
    a hex neighbour) x N_BRANCH slots, and the work per arc-length step is
    proportional to the number of tilt points ACTIVE there — which is 1 wherever
    only the perpendicular column was materialised.  A flat lattice therefore
    costs what the flat DP costs, and opening a collar costs the collar.

    Returns dict: ok, cut_index, cut_s, and on success path (Ns,3) of
    (q7 index, tilt index, branch index), qs, q7, tilt, margins, sigmas,
    bottleneck, travel.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}, want {OBJECTIVES}")
    maximin = objective == "maximin_sigma"
    free = free_mask(lat, sigma_gate, margin_gate) if free is None else free
    Q, S = lat["Q"], lat["sigma"]
    Ns, Nq, Nt, Nb = free.shape
    nbr = lat["disc"]["nbr"] if nbr is None else np.asarray(nbr, np.int32)
    tilt_pen = w_tilt * np.sum(np.asarray(lat["disc"]["tilt"], float) ** 2,
                               axis=1)                      # (Nt,)
    Qf = np.where(free[..., None], Q, 0.0)
    NEG, INF = -np.inf, np.inf
    Sq = (np.where(free, np.round(S * planner._SIGMA_Q), NEG) if maximin
          else np.where(free, 0.0, NEG))

    act = [np.flatnonzero(free[i].any(axis=(0, 2))) for i in range(Ns)]
    if not len(act[0]):
        return dict(ok=False, cut_index=-1, cut_s=0.0, reach_counts=np.zeros(Ns, int))

    # Only tilt directions that SOMEWHERE have a neighbour get a candidate
    # slot.  On a one-point disc that is direction 0 alone, so the candidate
    # axis is 3 x N_BRANCH and a flat lattice pays a flat lattice's DP.
    dirs = [d for d in range(7) if (nbr[:, d] >= 0).any()]
    NC = 3 * len(dirs) * Nb
    parent = [None] * Ns
    B = Sq[0][:, act[0], :].copy()
    C = np.where(free[0][:, act[0], :],
                 tilt_pen[act[0]][None, :, None], INF)
    reach = np.zeros(Ns, int)
    reach[0] = int(np.isfinite(C).sum())
    last_ok = 0

    for i in range(1, Ns):
        ta, ts = act[i], act[i - 1]
        if not len(ta) or not len(ts):
            break
        # where each active source tilt sits in the previous step's fiber
        pos = np.full(Nt, -1, np.int64)
        pos[ts] = np.arange(len(ts))
        cb = np.full((Nq, len(ta), Nb, NC), NEG)
        cc = np.full((Nq, len(ta), Nb, NC), INF)
        for dslot, d in enumerate(dirs):         # tilt: self, then 6 neighbours
            src_t = nbr[ta, d]                   # (A,) tilt index, -1 = none
            ok_t = src_t >= 0
            if ok_t.any():                       # ... and materialised at i-1
                ok_t[ok_t] = pos[src_t[ok_t]] >= 0
            if not ok_t.any():
                continue
            aa = np.flatnonzero(ok_t)            # target slots using this dir
            sp = pos[src_t[aa]]                  # their source fiber positions
            for u, dj in enumerate((-1, 0, 1)):
                lo, hi = max(0, dj), min(Nq, Nq + dj)
                if lo >= hi:
                    continue
                tgt, src = slice(lo, hi), slice(lo - dj, hi - dj)
                A = Qf[i][tgt][:, ta[aa], :, :]                    # (nq,a,Nb,7)
                Bq = Qf[i - 1][src][:, src_t[aa], :, :]
                dd = A[:, :, :, None, :] - Bq[:, :, None, :, :]
                dinf = np.max(np.abs(dd), axis=-1)
                stepc = (np.sum(dd * dd, axis=-1) if maximin
                         else np.sum(np.abs(dd), axis=-1))
                feas = (free[i][tgt][:, ta[aa], :, None]
                        & free[i - 1][src][:, src_t[aa], None, :]
                        & (dinf <= jump)
                        & (B[src][:, sp, None, :] > NEG))
                sl = slice((dslot * 3 + u) * Nb, (dslot * 3 + u + 1) * Nb)
                blk_b = np.where(feas, np.minimum(B[src][:, sp, None, :],
                                                  Sq[i][tgt][:, ta[aa], :, None]),
                                 NEG)
                blk_c = np.where(feas, C[src][:, sp, None, :] + w_smooth * stepc
                                 + tilt_pen[ta[aa]][None, :, None, None], INF)
                # cb[tgt] is a basic slice, hence a VIEW: this writes through.
                cb[tgt][:, aa, :, sl] = blk_b
                cc[tgt][:, aa, :, sl] = blk_c
        best_b = cb.max(axis=-1)
        k = np.argmin(np.where(cb == best_b[..., None], cc, INF), axis=-1)
        Cn = np.take_along_axis(cc, k[..., None], -1)[..., 0]
        if not np.isfinite(Cn).any():
            break
        Bn = np.take_along_axis(cb, k[..., None], -1)[..., 0]
        B = np.where(np.isfinite(Cn), Bn, NEG)
        C = Cn
        parent[i] = np.where(np.isfinite(Cn), k, -1).astype(np.int32)
        last_ok = i
        reach[i] = int(np.isfinite(Cn).sum())

    out = dict(ok=last_ok == Ns - 1, cut_index=last_ok,
               cut_s=last_ok / max(Ns - 1, 1), reach_counts=reach)
    if last_ok < 1:
        return out
    tie = B == B.max()
    idx = np.unravel_index(int(np.argmin(np.where(tie, C, INF))), C.shape)
    out["dp_cost"] = float(C[idx])
    # backtrack: decode (tilt slot, dq7, branch) out of the packed parent
    jj, aa, bb = int(idx[0]), int(idx[1]), int(idx[2])
    path = [(jj, int(act[last_ok][aa]), bb)]
    for i in range(last_ok, 0, -1):
        k = int(parent[i][jj, aa, bb])
        assert k >= 0, f"missing parent at step {i}"
        b_src = k % Nb
        rest = k // Nb
        dj, d = rest % 3 - 1, dirs[rest // 3]
        t_tgt = act[i][aa]
        t_src = int(nbr[t_tgt, d])
        jj = jj - dj
        aa = int(np.flatnonzero(act[i - 1] == t_src)[0])
        bb = b_src
        path.append((jj, t_src, bb))
    path = np.array(path[::-1], int)
    n = len(path)
    qs = np.array([Q[i, path[i, 0], path[i, 1], path[i, 2]] for i in range(n)])
    assert np.all(np.isfinite(qs)), "planned path visits an invalid node"
    out_sigmas = np.array([lat["sigma"][i, path[i, 0], path[i, 1], path[i, 2]]
                           for i in range(n)])
    out.update(path=path, qs=qs,
               q7=lat["q7s"][path[:, 0]],
               tilt=lat["disc"]["tilt"][path[:, 1]],
               margins=np.array([lat["margin"][i, path[i, 0], path[i, 1],
                                               path[i, 2]] for i in range(n)]),
               sigmas=out_sigmas,
               bottleneck=float(out_sigmas.min()),
               travel=float(np.abs(np.diff(qs, axis=0)).sum()),
               max_step=float(np.abs(np.diff(qs, axis=0)).max()) if n > 1 else 0.0,
               max_lean_deg=float(np.rad2deg(
                   np.linalg.norm(lat["disc"]["tilt"][path[:, 1]], axis=1).max())))
    return out


# --------------------------------------------------------------------------
# 5. from a lattice path to a certified trajectory
# --------------------------------------------------------------------------
def _knot_values(lat, path):
    """(Ns,4) [s, q7, tx, ty] implied by a lattice index path."""
    Ns = len(path)
    s = np.arange(Ns) / max(Ns - 1, 1)
    return np.column_stack([s, lat["q7s"][path[:, 0]],
                            lat["disc"]["tilt"][path[:, 1]]])


def _interp_plan(knots, s):
    """Read (q7, tx, ty) off the knot polyline at normalised arc lengths `s`.

    ONLY the redundancy parameters are interpolated — never q, and never the
    pose.  (tx, ty) is interpolated as a VECTOR, which is the second reason the
    chart matters: interpolating (theta, phi) across the apex would swing the
    azimuth through half a turn while the lean passes through zero, and the
    pen would spin on the paper for nothing.
    """
    k = np.asarray(knots, float)
    s = np.atleast_1d(np.asarray(s, float))
    return (np.interp(s, k[:, 0], k[:, 1]),
            np.column_stack([np.interp(s, k[:, 0], k[:, 2]),
                             np.interp(s, k[:, 0], k[:, 3])]))


def chase(stroke_pts, spec, knots, q_seed, ds=0.005, lat=None, pen_ext=None,
          h_inv=None, margin_gate=MARGIN_GATE, sigma_gate=SIGMA_GATE,
          jump=planner.JUMP_THRESH, gated=True, fallback=False):
    """Walk a tilted plan with case-consistent IK. -> `pwl.chase_report` dict.

    THE ONLY THING TILT CHANGES HERE IS THE POSE ARRAY.  `pwl.chase_cc` takes
    poses and a commanded q7 and knows nothing about where the poses came
    from, so the certification machinery — the gates, the branch consistency,
    the tip-error report — carries over untouched.  That is the whole
    compatibility story for the back-out: `pwl.pen_down_poses` gains a tilt
    argument and everything downstream of it already works.
    """
    from . import pwl
    Twb = lat["Twb"] if lat is not None else spec.T_world_base() if h_inv is None \
        else spec.T_world_base(h_inv)
    pen_ext = (lat["pen_ext"] if lat is not None else PEN_EXT) \
        if pen_ext is None else pen_ext
    pts, s = planner.resample(stroke_pts, ds)
    q7, tl = _interp_plan(knots, s)
    poses = pen_poses(pts, tl, np.linalg.inv(Twb), pen_ext)
    ch = pwl.chase_cc(poses, q7, q_seed, pen_ext=pen_ext,
                      margin_gate=margin_gate if gated else None,
                      sigma_gate=sigma_gate if gated else None,
                      jump_gate=jump if gated else None, fallback=fallback)
    setup = dict(pts=pts, s=s, q7=q7, poses=poses, Twb=Twb, pen_ext=pen_ext,
                 ds=ds, arc_len=float(np.linalg.norm(
                     np.diff(np.asarray(stroke_pts, float), axis=0),
                     axis=1).sum()), q_seed=q_seed)
    if not ch["n"]:
        return dict(ok=False, certified=False, n=0, fails=ch["fails"],
                    cut_index=-1, stop=ch["stop"], tilt=tl[:0], q7=q7[:0])
    rep = pwl.chase_report(ch, setup, jump=jump)
    rep["tilt"] = tl[:ch["n"]]
    rep["stop"] = ch["stop"]
    rep["lean"] = np.rad2deg(np.linalg.norm(rep["tilt"], axis=1))
    rep["max_lean_deg"] = float(rep["lean"].max()) if len(rep["lean"]) else 0.0
    rep["certified"] = bool(rep["ok"] and rep["min_sigma"] >= sigma_gate
                            and rep["min_margin"] >= margin_gate
                            and rep["continuous"])
    return rep


def simplify(lat, path, ds_corridor=None, eps=1.0, margin_gate=MARGIN_GATE,
             sigma_gate=SIGMA_GATE, jump=planner.JUMP_THRESH, exact=True):
    """RDP on the (q7, tx, ty) index path, with a chase veto on every chord.

    Deviation is measured in GRID INDICES per axis — q7 in q7-grid steps, tilt
    in hex-lattice pitches — and the worst axis is what the tolerance is
    compared against, so one `eps` covers a mixed-unit plan the way
    `pwl._rdp_corridor`'s single q7 tolerance covers a scalar one.

    A chord is accepted only if it is within `eps` AND a case-consistent chase
    along it clears both gates and the continuity budget — `pwl.Corridor`'s
    exact stage, with the pose array now depending on the interpolated tilt as
    well as on s.  Returns the knot array (K,4).
    """
    from . import pwl
    vals = _knot_values(lat, path)
    Ns = len(path)
    q7step = float(lat["q7s"][1] - lat["q7s"][0]) if len(lat["q7s"]) > 1 else 1.0
    pitch = lat["disc"]["pitch"] or 1.0
    scale = np.array([1.0 / q7step, 1.0 / pitch, 1.0 / pitch])
    idx = vals[:, 1:] * scale                      # the path in index units
    Q0 = lat["Q"][0, path[0, 0], path[0, 1], path[0, 2]]
    poses_pts = lat["pts"]
    Twb_inv = lat["Twb_inv"]

    # THE GRID PROXY, WITHOUT WHICH THIS IS THE WHOLE COST OF THE FEATURE.
    # `pwl.Corridor` screens a candidate segment against the lattice fields
    # before it will pay for a chase, and that ordering is not a micro-
    # optimisation: a chase is a python loop with one `ik.solve_cc`, one
    # analytic Jacobian and one SVD per sample, and RDP proposes far more
    # chords than it keeps.  Screening first took the adaptive planner's rim
    # arc from 830 ms to well under the budget.  The proxy is the free-mask
    # lookup at the nearest lattice cell — nearest q7 index by rounding,
    # nearest tilt point by brute force over the disc (Nt is 7 to 61).
    free = free_mask(lat, sigma_gate, margin_gate)
    any_free = free.any(axis=3)                        # (Ns, Nq, Nt)
    disc_t = lat["disc"]["tilt"]
    Nq = free.shape[1]

    def _grid_ok(ii, q7, tl):
        j = np.clip(np.rint((q7 - lat["q7s"][0]) / q7step).astype(int), 0, Nq - 1)
        d = np.linalg.norm(tl[:, None, :] - disc_t[None, :, :], axis=2)
        t = np.argmin(d, axis=1)
        return bool(any_free[ii, j, t].all())

    def ok(i0, i1):
        if not exact or i1 <= i0 + 1:
            return True
        ii = np.arange(i0, i1 + 1)
        t = (ii - i0) / (i1 - i0)
        q7 = vals[i0, 1] + (vals[i1, 1] - vals[i0, 1]) * t
        tl = vals[i0, 2:] + (vals[i1, 2:] - vals[i0, 2:]) * t[:, None]
        if not _grid_ok(ii, q7, tl):
            return False
        q = lat["Q"][i0, path[i0, 0], path[i0, 1], path[i0, 2]]
        if not np.all(np.isfinite(q)):
            return False
        poses = pen_poses(poses_pts[ii], tl, Twb_inv, lat["pen_ext"])
        return pwl.chase_cc(poses, q7, q, pen_ext=lat["pen_ext"],
                            margin_gate=margin_gate, sigma_gate=sigma_gate,
                            jump_gate=jump)["ok"]

    keep = [0, Ns - 1]

    def rec(i0, i1):
        if i1 <= i0 + 1:
            return
        ii = np.arange(i0, i1 + 1)
        t = (ii - i0) / (i1 - i0)
        chord = idx[i0] + (idx[i1] - idx[i0]) * t[:, None]
        dev = np.max(np.abs(idx[ii] - chord), axis=1)
        k = int(np.argmax(dev))
        if dev[k] <= eps and ok(i0, i1):
            return
        cut = i0 + k if 0 < k < i1 - i0 else (i0 + i1) // 2
        keep.append(cut)
        rec(i0, cut)
        rec(cut, i1)

    rec(0, Ns - 1)
    ki = np.array(sorted(set(keep)))
    return vals[ki], ki, Q0


def from_flat(flat, disc):
    """A tilt lattice whose perpendicular column IS an existing flat lattice.

    `planner.build_lattice`'s output is, node for node, this module's tilt
    index 0 (`tests/test_tilt.py::test_tilt0_lattice_matches_planner` pins it),
    so a rescue never re-solves the IK the flat pass already paid for.  That is
    what makes the adaptive planner's overhead the COLLAR and nothing else.
    """
    lat = empty_lattice(flat["pts"], flat["spec"], disc,
                        pen_ext=flat["pen_ext"], n_q7=len(flat["q7s"]))
    lat["Twb"] = flat["Twb"]
    lat["Twb_inv"] = np.linalg.inv(flat["Twb"])
    lat["Q"][:, :, 0, :, :] = flat["Q"]
    lat["valid"][:, :, 0, :] = flat["valid"]
    lat["margin"][:, :, 0, :] = flat["margin"]
    lat["sigma"][:, :, 0, :] = flat["sigma"]
    lat["cells"][:, 0] = True
    lat["n_solved_cells"] = int(len(flat["pts"]))
    return lat


def weak_cells(lat, sigma_gate=SIGMA_GATE, margin_gate=MARGIN_GATE,
               thin_frac=THIN_FRAC):
    """(Ns,) bool: arc-length steps whose free fiber is empty or unusually thin.

    "Thin" is relative to the stroke's own median width, because an absolute
    node count means different things on a 48-sample q7 grid at different
    places on the paper.  A step with no free node at all is always weak.
    """
    free = free_mask(lat, sigma_gate, margin_gate)
    w = free.sum(axis=(1, 2, 3))
    med = float(np.median(w[w > 0])) if np.any(w > 0) else 0.0
    return (w == 0) | (w < max(1.0, thin_frac * med)), w


def collar(mask, ds, collar_m=COLLAR_M):
    """Dilate a per-step boolean mask by `collar_m` METRES of arc length.

    Opening tilt exactly on the failing step is useless: the plan has to reach
    the tilted posture and come back, and both take arc length at
    ||dq||_inf <= 0.35 per step.  The collar is that runway.
    """
    mask = np.asarray(mask, bool)
    r = max(1, int(round(float(collar_m) / max(float(ds), 1e-9))))
    out = np.zeros_like(mask)
    for k in range(-r, r + 1):
        out |= np.roll(mask, k) if k == 0 else _shift(mask, k)
    return out


def _shift(m, k):
    out = np.zeros_like(m)
    if k > 0:
        out[k:] = m[:-k]
    elif k < 0:
        out[:k] = m[-k:]
    else:
        out[:] = m
    return out


# --------------------------------------------------------------------------
# 6. the entry points
# --------------------------------------------------------------------------
def _jump_ladder(tilt_max_deg, o):
    """The DP edge budgets to try, in order. -> tuple of rad.

    THE FULL BUDGET FIRST, THE TIGHT ONE AS A FALLBACK.  `planner.JUMP_THRESH`
    = 0.35 is what a finished trajectory must satisfy, and planning with it is
    what lets the rim arc keep its cheap 2-knot path (travel 5.40 rad; forcing
    0.25 on the same stroke costs 19.4).  But a DP that may propose a 0.35 step
    while BOTH q7 and tilt move is proposing a step the certification chase
    often cannot follow — `ik.solve_cc` walks whichever branch is continuous,
    not whichever branch the DP scored, and on the R bowl at tilt 15 the two
    part company and the chase dies at ||dq||_inf = 2.56 rad.

    So the tight budget is not a tax on every stroke, it is the second thing
    tried when the first does not certify.  Same idea as `stroke_api`'s
    `fallback_objective`: a plan is never refused because the FIRST search was
    greedy, and the extra DP is only paid by strokes that needed it.
    """
    if o.get("jump_search") is not None:
        return (float(o["jump_search"]),)
    if float(tilt_max_deg) <= 0:
        return (planner.JUMP_THRESH,)
    return (planner.JUMP_THRESH, JUMP_SEARCH)


def _solve(poly, spec, lat, o, extra, tilt_max_deg, nbr=None):
    """Run the DP and certify it, walking the jump ladder until one sticks."""
    import time
    best = None
    for jump in _jump_ladder(tilt_max_deg, o):
        t0 = time.perf_counter()
        res = plan_lattice(lat, w_tilt=o.get("w_tilt", W_TILT), jump=jump,
                           objective=o.get("objective_tilt", OBJECTIVE),
                           margin_gate=o.get("margin_gate", MARGIN_GATE),
                           sigma_gate=o.get("sigma_gate", SIGMA_GATE), nbr=nbr)
        t_dp = time.perf_counter() - t0
        out = _finish(poly, spec, lat, res, o,
                      dict(extra, t_dp=t_dp, jump_search=jump,
                           n_ik=int(lat["n_ik"])))
        if out["status"] == "ok":
            return out
        best = out if best is None else best
    return best


def _finish(poly, spec, lat, res, o, extra):
    """DP path -> knots -> dense chase -> clock -> independent certificate."""
    from . import pacing, stroke_api
    from .validate import validate_plan
    if not res.get("ok"):
        return dict(extra, status="split", reason="tilt_lattice_cut",
                    s_star=0.0, s_reach=float(res.get("cut_s", 0.0)),
                    head=None)
    # THE SIMPLIFICATION LADDER, AND WHY IT IS NOT OPTIONAL.  `simplify`
    # vetoes each candidate chord with a chase seeded from that chord's OWN
    # lattice node; the certification chase instead walks the whole stroke
    # from s = 0 and arrives at each knot with whatever configuration the
    # preceding chords left it in.  Chord-wise feasibility therefore does not
    # compose into stroke-wise feasibility — measured on the R bowl: nine
    # knots each individually chaseable, ||dq||_inf = 2.56 rad end to end.
    # `pwl`/`smooth` meet the same problem and answer it with window
    # bisection and a sharp fallback; the ladder here is the same idea in the
    # simplest form that works.  eps = 0 keeps every lattice step as a knot,
    # i.e. commands exactly the path the DP certified, and is the last word
    # before a split is conceded.
    # THE CHASE MUST LAND ON BOTH ENDS OF THE STROKE.  `planner.resample` steps
    # by a fixed ds and stops at the last WHOLE step, so chasing a 0.1373 m
    # stroke at a raw 5 mm leaves the last 2.3 mm unplanned — and every gate
    # here is POINTWISE, so not one of them can see a tail that was never
    # sampled.  `stroke_api.prepare` fits the step to the length for exactly
    # this reason and the tilt path was not doing it; the `coverage_gap` check
    # at the bottom of this function is what found it.
    ds_dense = stroke_api._fit_ds(stroke_api.polyline_length(poly),
                                  o["ds_dense"])
    ch = None
    for eps in (o.get("eps_idx", 1.0), 0.0):
        knots, ki, q_seed = simplify(
            lat, res["path"], eps=eps,
            exact=o.get("exact", True) and eps > 0,
            margin_gate=o.get("margin_gate", MARGIN_GATE),
            sigma_gate=o.get("sigma_gate", SIGMA_GATE))
        ch = chase(poly, spec, knots, q_seed, ds=ds_dense, lat=lat,
                   margin_gate=o.get("margin_gate", MARGIN_GATE),
                   sigma_gate=o.get("sigma_gate", SIGMA_GATE))
        if ch.get("certified"):
            break
    if not ch.get("certified"):
        return dict(extra, status="split", reason="chase_failed",
                    s_star=0.0, n_knots=int(len(knots)),
                    s_reach=float(max(ch.get("cut_index", 0), 0))
                    / max(len(ch.get("q7", [1])) - 1, 1),
                    stop=ch.get("stop"), head=None)
    qs = ch["qs"]
    pc = pacing.pace(qs, ch["arc_len"], v_draw=o["v_draw"], safety=o["safety"],
                     ds_m=ch["ds"], dqds=ch["dqds"])
    cone = float(extra["tilt_max_deg"])
    rep = validate_plan(ch["pts"], spec, qs, times=pc["t"],
                        pen_ext=lat["pen_ext"],
                        margin_gate=o.get("margin_gate", MARGIN_GATE),
                        sigma_gate=o.get("sigma_gate", SIGMA_GATE),
                        tilt_max_deg=cone) if o.get("validate", True) \
        else dict(ok=True)
    lean, worst, inside = cone_check(qs, spec, cone)
    out = dict(extra, status="ok", reason="", knots=knots, qs=qs,
               pts=ch["pts"], times=pc["t"], s=ch["s"], q7=ch["q7"],
               tilt=ch["tilt"], sigmas=ch["sigmas"], margins=ch["margins"],
               min_sigma=float(ch["min_sigma"]), min_margin=float(ch["min_margin"]),
               tip_err=float(ch["tip_err"]), max_step=float(ch["max_step"]),
               sum_travel=float(ch["sum_travel"]), n_knots=int(len(knots)),
               n_dense=int(len(qs)), arc_len=float(ch["arc_len"]),
               max_lean_deg=float(worst), lean=lean, cone_ok=bool(inside),
               bottleneck=float(res.get("bottleneck", np.nan)),
               total_time=float(pc["total_time"]),
               frac_slowed=float(pc["frac_slowed"]),
               headroom=float(pc["headroom"]), validation=rep,
               **_flat_shaped_fields(poly, knots, ch))
    # coverage: an "ok" plan draws the WHOLE stroke, and `stroke_api` checks
    # that the same way -- pointwise tip error cannot see a plan that quietly
    # stopped short of an END, because it never samples there.
    gap = max(float(np.linalg.norm(np.asarray(ch["pts"])[0] - poly[0])),
              float(np.linalg.norm(np.asarray(ch["pts"])[-1] - poly[-1])))
    out["coverage_gap"] = gap
    if gap > 1e-6:
        out.update(status="bug", reason="incomplete_coverage")
    if not rep["ok"]:
        out.update(status="bug", reason="validation_failed")
    if not inside:
        out.update(status="bug", reason="cone_violation")
    return out


# CORNER ROUNDING IS THE ONE THING A TILTED PLAN DOES NOT GET, AND IT IS SAID
# HERE RATHER THAN LEFT TO BE INFERRED FROM A MISSING KEY.  `smooth.RoundedPWL`
# rounds a knot by fitting a quadratic Bezier in a WINDOW around it, and the
# plan it rounds is one scalar function of s (q7).  A tilted plan is three
# (q7, tx, ty) and wants the same windows applied per component with the tilt
# pair interpolated as a VECTOR — interpolating (theta, phi) through the apex
# would swing the azimuth half a turn while the lean passed through zero and
# spin the pen on the paper for nothing.  That generalisation is real work and
# it is NOT done: a tilted plan ships as the SHARP polyline the DP certified,
# with every window zero.
#
# WHAT THAT COSTS, EXACTLY.  Nothing in the certificate: `chase` walks the same
# 5 mm samples and enforces the same gates whether the corners are rounded or
# not, and `stroke_api` already ships sharp polylines when rounding fails to
# certify ("corner rounding did not certify; kept the sharp polyline").  What
# it costs is |dq/ds| at the knots, and therefore the clock, because
# `pacing.pace` slows the ink until no joint exceeds its velocity cap.  On the
# spans this feature exists for -- 50 mm and 10 mm rescues at the edge of an
# arm's reach -- that is a few knots on a few centimetres, and the measured
# `frac_slowed` says whether it bound at all.  On a long tilted stroke it would
# matter and the generalisation would have to be done first.
def _flat_shaped_fields(poly, knots, ch):
    """The keys a FLAT plan carries that the tilt path would otherwise omit.

    A tilted plan is handed to `allocate`, `sequence`, `writing` and
    `stroke_api.reverse_plan` through the same door as a flat one, so it has to
    be shaped like one.  `windows` is the load-bearing member: `reverse_plan`
    subscripts it without a default, so a tilt-rescued segment the sequencer
    wanted to draw backwards used to raise `KeyError` in the middle of an
    allocation.  Zero windows is not a placeholder -- it is the truth about
    this plan (see the note above).
    """
    return dict(windows=np.zeros(len(knots)), stroke=np.asarray(poly, float),
                clip_s=(0.0, 1.0), depth=0, sheet=-1,
                arc_len_input=float(ch["arc_len"]),
                notes=["tilted plan: sharp polyline, no corner rounding "
                       "(aris_sixarm/tilt.py, _flat_shaped_fields)"])


def _prep(pts_xy, spec, o):
    """Hygiene + resample + sheet clip, exactly as `stroke_api.prepare` does."""
    from . import fleet, stroke_api
    poly, notes = stroke_api.sanitize(np.asarray(pts_xy, float), o["dup_tol"])
    if len(poly) < 2:
        return None, None
    L = stroke_api.polyline_length(poly)
    if L < o["min_length"]:
        return None, None
    ds = stroke_api._fit_ds(L, o["ds_lattice"])
    pts, _ = planner.resample(poly, ds)
    if o["clip_to_sheet"]:
        kept = planner.clip_to_sheet(pts, verbose=False,
                                     sheet=fleet.sheet_for(spec))
        if len(kept) < 2:
            return None, None
        # ONLY WHEN SOMETHING WAS ACTUALLY CLIPPED, which is what
        # `stroke_api.prepare` does and what this function claims to do.
        # Replacing `poly` unconditionally handed the certification chase the
        # 10 mm LATTICE resample in place of the stroke the caller asked for,
        # so a tilted plan drew the chords of the curve rather than the curve:
        # 0.65 mm of arc length on a 138 mm span of stroke 26, and a commanded
        # path that cuts every corner by the lattice's chord error.  The
        # lattice is built on `pts` either way; `poly` is the curve the plan is
        # certified AGAINST and it has to be the one that came in.
        if len(kept) != len(pts):
            poly, pts = kept, kept
    return poly, pts


def plan_oracle(pts_xy, spec, tilt_max_deg=15.0, n_ring=N_RING,
                pitch_deg=None, opts=None):
    """GROUND TRUTH: the whole tilt lattice, one DP, one certification.

    Seconds per stroke.  Use it to answer "what does the freedom BUY", never
    to draw with.  `plan_adaptive` is the one with a runtime budget.
    """
    from . import stroke_api
    o = dict(stroke_api.DEFAULTS)
    o.update(opts or {})
    poly, pts = _prep(pts_xy, spec, o)
    if poly is None:
        return dict(status="degenerate", reason="too_short_or_off_sheet",
                    tilt_max_deg=tilt_max_deg)
    import time
    disc = hex_disc(tilt_max_deg, n_ring, pitch_deg)
    t0 = time.perf_counter()
    lat = build_lattice(pts, spec, disc=disc, n_q7=o["n_q7"],
                        pen_ext=o["pen_ext"])
    t_lat = time.perf_counter() - t0
    extra = dict(arm=getattr(spec, "arm_id", None), tilt_max_deg=tilt_max_deg,
                 n_tilt=int(len(disc["tilt"])), n_lattice=int(len(pts)),
                 t_lattice=t_lat, mode="oracle")
    return _solve(poly, spec, lat, o, extra, tilt_max_deg)


def plan_adaptive(pts_xy, spec, tilt_max_deg=15.0, n_ring=N_RING,
                  pitch_deg=None, opts=None, collar_m=COLLAR_M,
                  coarse_ring=1, always=False):
    """THE SHIPPING CANDIDATE: flat first, tilt only where flat was not enough.

    1. Plan the stroke flat.  At the SHIPPING gates that means calling the
       shipping pipeline itself — same lattice, sheets, PWL, corner rounding
       and certificate — so a stroke that works today returns today's answer,
       bit for bit, having paid one status comparison for the privilege.  At
       stricter gates the shipping entry point cannot express the question
       (its gates are module constants), so the flat pass is this module's own
       DP run on a one-point disc, which is the same search with the gates
       moved.
    2. If that certifies, ship it.  Otherwise reuse the flat lattice
       (`from_flat` re-solves no IK), mark the arc-length steps whose free
       fiber is empty or thin, dilate them into a COLLAR, and materialise the
       tilt columns THERE and nowhere else — at the coarse ring of the disc
       first.
    3. Re-DP and certify.  Still failing, refine the collar to the full disc
       and try once more before conceding the flat pass's split.

    THE FALLBACK IS THE WHOLE SAFETY ARGUMENT.  Every exit that is not an
    improvement returns the flat result unchanged, so opening the tilt axis
    can add certified strokes and can never remove one.

    `always=True` opens the collar even when the flat plan succeeded, which is
    how the study measures what tilt does to strokes that already work.
    """
    import time
    from . import stroke_api
    o = dict(stroke_api.DEFAULTS)
    o.update(opts or {})
    mg = o.get("margin_gate", MARGIN_GATE)
    sg = o.get("sigma_gate", SIGMA_GATE)
    shipping_gates = (mg == MARGIN_GATE and sg == SIGMA_GATE)
    t_start = time.perf_counter()

    def meta(r, **kw):
        r = dict(r)
        r.update(tilt_max_deg=tilt_max_deg, mode="adaptive",
                 t_total=time.perf_counter() - t_start, **kw)
        # THE CONE A PLAN CARRIES IS THE ONE IT NEEDS, NOT THE ONE THE RUN WAS
        # WILLING TO ALLOW.  `tilt_max_deg` travels with the plan into
        # `validate.validate_plan` and `scene_check`, which check the lean
        # against it; if every plan in a tilt-enabled run carried 15 degrees,
        # turning the flag on would quietly weaken the certificate of the
        # hundreds of strokes that never leaned.  The allowance is kept as
        # `tilt_allowance` so the run's setting is still legible.
        r["tilt_allowance"] = float(tilt_max_deg)
        if float(r.get("max_lean_deg", 0.0) or 0.0) <= 0.0:
            r["tilt_max_deg"] = 0.0
        return r

    poly = pts = flat_lat = None
    # ---- 1. the flat pass -------------------------------------------------
    if shipping_gates:
        ctx, early = stroke_api.prepare(pts_xy, spec, o)
        if early is not None and early.get("status") == "degenerate":
            return meta(early, tilt_used=False)
        if ctx is not None:
            flat = stroke_api.plan_from_ctx(ctx, spec, o)
            poly, pts, flat_lat = ctx["poly"], ctx["pts"], ctx["lat"]
        else:
            # `prepare` refuses before it can hand back a context when the
            # fiber is empty at s = 0 — which is EXACTLY the stroke tilt is
            # for, so this path must fall through to the rescue and not out.
            flat = early
    else:
        poly, pts = _prep(pts_xy, spec, o)
        if poly is None:
            return meta(dict(status="degenerate",
                             reason="too_short_or_off_sheet"), tilt_used=False)
        flat_lat = planner.build_lattice(pts, spec, h_inv=o["h_inv"],
                                         pen_ext=o["pen_ext"], n_q7=o["n_q7"])
        flat = _solve(poly, spec, from_flat(flat_lat, hex_disc(0.0)), o,
                      dict(arm=getattr(spec, "arm_id", None), tilt_max_deg=0.0,
                           n_tilt=1, mode="adaptive-flat"), 0.0)
    t_flat = time.perf_counter() - t_start
    if flat.get("status") == "ok" and not always:
        return meta(flat, tilt_used=False, max_lean_deg=0.0, t_flat=t_flat)
    if float(tilt_max_deg) <= 0:
        return meta(flat, tilt_used=False, t_flat=t_flat)

    # ---- 2. the collar ----------------------------------------------------
    if poly is None or pts is None:
        poly, pts = _prep(pts_xy, spec, o)
        if poly is None:
            return meta(flat, tilt_used=False, t_flat=t_flat)
    if flat_lat is None:
        flat_lat = planner.build_lattice(pts, spec, h_inv=o["h_inv"],
                                         pen_ext=o["pen_ext"], n_q7=o["n_q7"])
    Ns = len(pts)
    ds = float(stroke_api.polyline_length(poly)) / max(Ns - 1, 1)
    disc = hex_disc(tilt_max_deg, n_ring, pitch_deg)
    lat = from_flat(flat_lat, disc)
    weak, width = weak_cells(lat, sigma_gate=sg, margin_gate=mg)
    if flat.get("status") == "split":
        cut = int(round(float(flat.get("s_reach", 0.0)) * (Ns - 1)))
        weak[max(0, cut - 1):min(Ns, cut + 2)] = True
    if not weak.any():
        weak[:] = True                       # nothing looked weak: open it all
    band = collar(weak, ds, collar_m)

    # ---- 3. coarse ring on the collar, then the full disc -----------------
    out = None
    n_ring_eff = int(max(disc["ring"])) or 1
    for stage, m in enumerate(sorted({n_ring_eff, 1}, reverse=True)):
        keep, nbr = sub_disc(disc, m)
        cells = np.zeros(lat["cells"].shape, bool)
        cells[np.ix_(np.flatnonzero(band), keep)] = True
        extend_lattice(lat, cells)
        extra = dict(arm=getattr(spec, "arm_id", None),
                     tilt_max_deg=tilt_max_deg, n_tilt=int(len(keep)),
                     n_lattice=Ns, mode="adaptive", tilt_used=True,
                     stage=stage, sub_pitch=int(m),
                     collar_steps=int(band.sum()),
                     collar_frac=float(band.mean()), t_flat=t_flat,
                     flat_status=flat.get("status"))
        out = _solve(poly, spec, lat, o, extra, tilt_max_deg, nbr=nbr)
        if out["status"] == "ok":
            return meta(out)
    return meta(flat, tilt_used=True, tilt_failed=True, t_flat=t_flat,
                collar_steps=int(band.sum()), n_ik=int(lat["n_ik"]))
