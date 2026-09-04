"""Raster logo -> pen strokes: colour unmixing, thinning, skeleton-graph tracing.

The CSAIL mark is line art drawn with ONE pen width in two colours, plus a
solid wordmark.  Those are two different tracing problems and this module
keeps them apart:

  building outlines   uniform-width strokes.  The ink is the stroke, so the
                      centreline is what a pen should follow -> thin the mask
                      (Zhang-Suen) and trace the skeleton graph.
  "CSAIL" wordmark    solid letterforms, ~30 px wide at the working scale.  A
                      skeleton of a filled glyph is a stick figure, not a
                      letter, so these are traced as BOUNDARY CONTOURS
                      (marching squares on the sub-pixel coverage field, so
                      the letterforms come out smooth rather than stepped).
                      The two are told apart by how much of a connected
                      component survives erosion — see `split_thick`.

WHY THE JUNCTION LOGIC IS THE WHOLE GAME.  The logo is a pile of overlapping
building silhouettes: the skeleton of the grey mask alone has ~50 junctions.
A naive "cut at every junction" tracer returns confetti — dozens of 20 px
fragments, each of which becomes its own pen-up/pen-down for the fleet, and
none of which looks like a building edge.  So at every junction node the
incident branches are PAIRED by straightness (greedy over the turn angle,
capped at MAX_TURN) and the pairing is followed through: an X crossing leaves
as two strokes that pass through each other, a T leaves as the bar plus the
stem.  Nearby junctions are contracted first (a crossing thins into two T's a
few pixels apart, not one X), spurs shorter than SPUR_PX are pruned, and
endpoints that face each other across a small gap are bridged — the orange
lines are painted OVER the grey ones, so every grey line that passes under an
orange one is interrupted by exactly one stroke width of white.

Coordinates: pixel space is (x = column, y = row, y down) at the upsampled
resolution.  `to_sheet` flips y, applies one uniform scale and centres the
result on the paper.  Nothing here imports IK or drake.
"""
import numpy as np
from PIL import Image

# ---- palette (measured off assets/csail/csail_old_med.gif) ------------------
GREY_RGB = (102, 102, 101)
ORANGE_RGB = (203, 102, 8)
BG_RGB = (255, 255, 255)
ALPHA_ON = 0.45            # ink coverage at which a pixel counts as drawn

UPSCALE = 4
LETTER_ERODE = 6           # 3x3 erosions that a building line must not survive
                           # (lines are ~11 px wide at 4x, letters ~31 px)
SPUR_PX = 7                # dead-end branches shorter than this are hairs
MERGE_PX = 6               # junction pair closer than this = one junction
MAX_TURN = np.deg2rad(62)  # straightest-continuation cap at a junction
DIR_PX = 9                 # branch pixels used to estimate its direction
BRIDGE_PX = 11             # endpoint gap that occlusion can explain
BRIDGE_TURN = np.deg2rad(38)
MIN_PX = 8                 # specks
RDP_TOL = 1.2              # px, at the upsampled scale


# ===========================================================================
# 1. colour -> masks
# ===========================================================================
def unmix(rgb):
    """(H,W,3) uint8 -> (alpha_grey, alpha_orange), each (H,W) float.

    Antialiased pixels are mixtures `bg + a_g*(grey-bg) + a_o*(orange-bg)`;
    the least-squares solve of that 3x2 system recovers the two coverages, so
    a half-covered orange pixel and a fully covered light-grey one are told
    apart by *hue* rather than by a brightness threshold that would have to
    straddle both palettes.
    """
    c = np.asarray(rgb, float) - np.asarray(BG_RGB, float)
    M = np.column_stack([np.asarray(GREY_RGB, float) - BG_RGB,
                         np.asarray(ORANGE_RGB, float) - BG_RGB])
    a = c.reshape(-1, 3) @ np.linalg.pinv(M).T
    a = a.reshape(rgb.shape[0], rgb.shape[1], 2)
    return a[..., 0], a[..., 1]


def _upsample(alpha, k):
    im = Image.fromarray(alpha.astype(np.float32), mode="F")
    return np.asarray(im.resize((alpha.shape[1] * k, alpha.shape[0] * k),
                                Image.BICUBIC), float)


def load_masks(path, upscale=UPSCALE, alpha_on=ALPHA_ON):
    """Image file -> dict(grey, orange, shape) of boolean masks, upsampled."""
    rgb = np.asarray(Image.open(path).convert("RGB"))
    ag, ao = unmix(rgb)
    ag, ao = _upsample(ag, upscale), _upsample(ao, upscale)
    grey, orange = ag > alpha_on, ao > alpha_on
    both = grey & orange                       # antialiased hue collisions
    grey &= ~(both & (ao > ag))
    orange &= ~(both & (ag >= ao))
    return dict(grey=grey, orange=orange, alpha_grey=ag, alpha_orange=ao,
                shape=grey.shape, upscale=upscale)


# ---- binary morphology (numpy only; no scipy in this environment) ----------
def _shifts(m):
    p = np.pad(m, 1)
    return [p[0:-2, 1:-1], p[2:, 1:-1], p[1:-1, 0:-2], p[1:-1, 2:],
            p[0:-2, 0:-2], p[0:-2, 2:], p[2:, 0:-2], p[2:, 2:]]


def dilate(m):
    s = _shifts(m)
    out = m.copy()
    for x in s:
        out |= x
    return out


def erode(m):
    s = _shifts(m)
    out = m.copy()
    for x in s:
        out &= x
    return out


def label_components(mask):
    """8-connected component labels (run-length union-find) -> (labels, n)."""
    H, W = mask.shape
    parent = [0]

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    lab = np.zeros((H, W), np.int32)
    prev = []
    for r in range(H):
        row = mask[r]
        idx = np.flatnonzero(np.diff(np.concatenate([[0], row.view(np.int8), [0]])))
        runs = list(zip(idx[0::2], idx[1::2]))
        cur = []
        for a, b in runs:
            hits = [pl for (pa, pb, pl) in prev if pa - 1 < b and a - 1 < pb]
            if hits:
                L = min(find(h) for h in hits)
                for h in hits:
                    union(L, h)
            else:
                L = len(parent)
                parent.append(L)
            lab[r, a:b] = L
            cur.append((a, b, L))
        prev = cur
    roots = np.array([find(i) for i in range(len(parent))])
    uniq = np.unique(roots[roots > 0])
    remap = np.zeros(len(parent), np.int32)
    remap[uniq] = np.arange(1, len(uniq) + 1)
    out = np.where(mask, remap[roots[lab]], 0)
    return out, len(uniq)


def split_thick(mask, n_erode=LETTER_ERODE, frac=0.15):
    """-> (thick, thin): whole components that are mostly fat, and the rest.

    A component counts as solid when more than `frac` of it survives
    `n_erode` erosions.  Reconstructing the eroded seed inside the mask —
    the textbook opening-by-reconstruction — does NOT work here: the orange
    building outlines are one single connected component, so one surviving
    seed pixel at a crossing floods the whole silhouette network.  Judging by
    area fraction per component separates a 30 px letterform (>0.5 survives)
    from a 5 px line whose crossings happen to be fat (<0.02).
    """
    seed = mask.copy()
    for _ in range(n_erode):
        seed = erode(seed)
    lab, n = label_components(mask)
    area = np.bincount(lab.ravel(), minlength=n + 1)
    surv = np.bincount(lab[seed].ravel(), minlength=n + 1)
    solid = np.zeros(n + 1, bool)
    solid[1:] = surv[1:] > frac * np.maximum(area[1:], 1)
    thick = solid[lab]
    return thick, mask & ~thick


# ===========================================================================
# 2. Zhang-Suen thinning
# ===========================================================================
def _ring(m):
    """The 8-ring P2..P9 clockwise from north, as boolean arrays."""
    p = np.pad(m, 1)
    return (p[0:-2, 1:-1], p[0:-2, 2:], p[1:-1, 2:], p[2:, 2:],
            p[2:, 1:-1], p[2:, 0:-2], p[1:-1, 0:-2], p[0:-2, 0:-2])


def _AB(P):
    B = sum(x.astype(np.uint8) for x in P)
    seq = list(P) + [P[0]]
    A = sum(((~seq[i]) & seq[i + 1]).astype(np.uint8) for i in range(8))
    return A, B


def thin(mask):
    """Zhang-Suen thinning -> 1 px wide, 8-connected skeleton (bool array)."""
    m = mask.copy()
    for _ in range(200):
        changed = False
        for step in (0, 1):
            P2, P3, P4, P5, P6, P7, P8, P9 = _ring(m)
            A, B = _AB((P2, P3, P4, P5, P6, P7, P8, P9))
            cond = m & (A == 1) & (B >= 2) & (B <= 6)
            if step == 0:
                cond &= ~(P2 & P4 & P6) & ~(P4 & P6 & P8)
            else:
                cond &= ~(P2 & P4 & P8) & ~(P2 & P6 & P8)
            if cond.any():
                m &= ~cond
                changed = True
        if not changed:
            break
    return m


def crossing_number(skel):
    """Per-pixel 0->1 transitions around the 8-ring: 1 = tip, 2 = path, >=3 = junction."""
    A, _ = _AB(_ring(skel))
    return np.where(skel, A, 0)


# ===========================================================================
# 3. skeleton -> graph -> long strokes
# ===========================================================================
_NB = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def _neighbours(skel, r, c):
    H, W = skel.shape
    out = []
    for dr, dc in _NB:
        rr, cc = r + dr, c + dc
        if 0 <= rr < H and 0 <= cc < W and skel[rr, cc]:
            out.append((rr, cc))
    return out


def _adjacent(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) <= 1


def skeleton_graph(skel):
    """-> (nodes, edges).  nodes: list of (r,c).  edges: dict id -> dict(a, b, path).

    `path` is the pixel chain INCLUDING both node pixels.  Closed curves with
    no node of their own become edges with a == b and a synthetic node.
    """
    cn = crossing_number(skel)
    deg = np.zeros(skel.shape, int)
    for dr, dc in _NB:
        deg += np.roll(np.roll(skel, dr, 0), dc, 1).astype(int)
    deg = np.where(skel, deg, 0)
    is_node = skel & ((cn != 2) | (deg > 2))
    node_id = {}
    nodes = []
    for r, c in zip(*np.nonzero(is_node)):
        if deg[r, c] == 0:
            continue                                   # isolated speck
        node_id[(int(r), int(c))] = len(nodes)
        nodes.append((int(r), int(c)))

    edges, seen = {}, set()
    for start in list(node_id):
        for nb in _neighbours(skel, *start):
            if (start, nb) in seen:
                continue
            path = [start]
            prev, cur = start, nb
            while True:
                path.append(cur)
                if cur in node_id:
                    break
                cand = [p for p in _neighbours(skel, *cur)
                        if p != prev and p not in (path[-3:-1])]
                if not cand:
                    break
                if len(cand) > 1:      # staircase: the true continuation is
                    cand.sort(key=lambda p: _adjacent(p, prev))  # not next to prev
                nxt = cand[0]
                prev, cur = cur, nxt
            end = path[-1]
            seen.add((start, nb))
            if end in node_id and len(path) > 1:
                seen.add((end, path[-2]))
            edges[len(edges)] = dict(a=node_id[start],
                                     b=node_id.get(end, node_id[start]),
                                     path=path)

    # closed loops carry no node at all — seed one arbitrarily
    used = np.zeros(skel.shape, bool)
    for e in edges.values():
        for r, c in e["path"]:
            used[r, c] = True
    left = skel & ~used
    while left.any():
        r, c = (int(x[0]) for x in np.nonzero(left))
        start = (r, c)
        node_id[start] = len(nodes)
        nodes.append(start)
        path = [start]
        left[r, c] = False
        prev, cur = None, None
        nbs = [p for p in _neighbours(skel, r, c) if left[p]]
        if not nbs:
            continue
        prev, cur = start, nbs[0]
        while True:
            path.append(cur)
            left[cur] = False
            cand = [p for p in _neighbours(skel, *cur) if left[p]]
            if not cand:
                if _adjacent(cur, start):
                    path.append(start)
                break
            if len(cand) > 1:
                cand.sort(key=lambda p: _adjacent(p, prev))
            prev, cur = cur, cand[0]
        edges[len(edges)] = dict(a=node_id[start], b=node_id[start], path=path)
    return nodes, edges


def _prune_spurs(nodes, edges, spur_px=SPUR_PX):
    """Drop dead-end branches shorter than `spur_px` (thinning hairs at corners)."""
    while True:
        ends = {}
        for i, e in edges.items():
            ends.setdefault(e["a"], []).append(i)
            ends.setdefault(e["b"], []).append(i)
        drop = None
        for i, e in edges.items():
            if e["a"] == e["b"]:
                continue
            leaf = (len(ends[e["a"]]) == 1) ^ (len(ends[e["b"]]) == 1)
            if leaf and len(e["path"]) < spur_px and \
                    max(len(ends[e["a"]]), len(ends[e["b"]])) > 2:
                drop = i
                break
        if drop is None:
            return edges
        edges.pop(drop)


def _merge_nodes(nodes, edges, merge_px=MERGE_PX):
    """Contract short junction-to-junction edges: a thinned X is two T's."""
    parent = list(range(len(nodes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    deg = {}
    for e in edges.values():
        deg[e["a"]] = deg.get(e["a"], 0) + 1
        deg[e["b"]] = deg.get(e["b"], 0) + 1
    for i, e in list(edges.items()):
        if e["a"] != e["b"] and len(e["path"]) <= merge_px \
                and deg.get(e["a"], 0) >= 3 and deg.get(e["b"], 0) >= 3:
            ra, rb = find(e["a"]), find(e["b"])
            if ra != rb:
                parent[rb] = ra
                edges.pop(i)
    for e in edges.values():
        e["a"], e["b"] = find(e["a"]), find(e["b"])
    return edges


def _dir(path, from_start, k=DIR_PX):
    p = np.array(path, float)[:, ::-1]          # (r,c) -> (x,y)
    if from_start:
        seg = p[:min(k, len(p))]
        d = seg[-1] - seg[0]
    else:
        seg = p[max(0, len(p) - k):]
        d = seg[0] - seg[-1]
    n = np.linalg.norm(d)
    return d / n if n > 1e-9 else np.array([0.0, 0.0])


def pair_at_junctions(edges, max_turn=MAX_TURN):
    """Greedy straightest-continuation pairing of branch ends at every node.

    An end is `(edge_id, 0|1)`; the returned dict maps end -> end (symmetric).
    Cost is the turn angle a pen would make going in along one branch and out
    along the other, so a 4-way crossing pairs into two through-strokes and a
    T-junction keeps its bar intact.
    """
    at = {}
    for i, e in edges.items():
        at.setdefault(e["a"], []).append((i, 0))
        at.setdefault(e["b"], []).append((i, 1))
    pair = {}
    for node, ends in at.items():
        if len(ends) < 2:
            continue
        dirs = {end: _dir(edges[end[0]]["path"], end[1] == 0) for end in ends}
        cands = []
        for x in range(len(ends)):
            for y in range(x + 1, len(ends)):
                ea, eb = ends[x], ends[y]
                if ea[0] == eb[0] and edges[ea[0]]["a"] != edges[ea[0]]["b"]:
                    continue
                turn = np.arccos(np.clip(-float(dirs[ea] @ dirs[eb]), -1, 1))
                cands.append((turn, x, y))
        cands.sort()
        taken = set()
        for turn, x, y in cands:
            if turn > max_turn or x in taken or y in taken:
                continue
            taken.update((x, y))
            pair[ends[x]] = ends[y]
            pair[ends[y]] = ends[x]
    return pair


def chain(edges, pair):
    """Follow the pairing into maximal strokes -> list of pixel polylines (x,y)."""
    def pts(eid, flip):
        p = np.array(edges[eid]["path"], float)[:, ::-1]
        return p[::-1] if flip else p

    used, out = set(), []
    ends = [(i, s) for i in edges for s in (0, 1)]
    for start in [e for e in ends if e not in pair] + ends:
        eid, side = start
        if eid in used:
            continue
        chunks, cur = [], (eid, side)
        while True:
            e, s = cur
            if e in used:
                break
            used.add(e)
            chunks.append(pts(e, flip=(s == 1)))
            nxt = pair.get((e, 1 - s))
            if nxt is None:
                break
            cur = nxt
        if chunks:
            out.append(np.vstack(chunks))
    return out


# ===========================================================================
# 4. polyline post-processing
# ===========================================================================
def rdp(pts, tol):
    """Ramer-Douglas-Peucker simplification, endpoints fixed."""
    p = np.asarray(pts, float)
    n = len(p)
    if n < 3:
        return p
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        d = p[j] - p[i]
        L = float(np.hypot(*d))
        seg = p[i + 1:j] - p[i]
        dist = (np.abs(d[0] * seg[:, 1] - d[1] * seg[:, 0]) / L if L > 1e-12
                else np.linalg.norm(seg, axis=1))
        k = int(np.argmax(dist))
        if dist[k] > tol:
            stack += [(i, i + 1 + k), (i + 1 + k, j)]
            keep[i + 1 + k] = True
    return p[keep]


def plen(p):
    p = np.asarray(p, float)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0


def bridge_endpoints(strokes, gap=BRIDGE_PX, turn=BRIDGE_TURN):
    """Join strokes whose ends face each other across a small gap.

    The orange line art is painted over the grey, so a grey edge running under
    an orange one is cut into two collinear pieces separated by one stroke
    width of white.  Re-joining them is what keeps the grey silhouettes
    reading as single building outlines.

    ONE MERGE PER PASS, and the search for it is vectorised rather than looped.
    Taking the globally best pair each time and re-deriving every end afterwards
    is what makes the result independent of stroke order, and it is also what
    makes the naive form quadratic PER MERGE — on a picture that comes out of
    the skeleton as 350 fragments (which is what a mis-detected background does
    to any tracer) that is 180 million Python-level distance evaluations and a
    full minute.  The arrays below are indexed `[i, j, ei, ej]` precisely so
    that C-order flattening reproduces the old loop nesting, and the winner is
    chosen on `(turn + approach, gap, position)` — so this is the same answer
    the loop gave, ties included, at numpy speed.
    """
    S = [np.asarray(s, float) for s in strokes]
    alive = np.ones(len(S), bool)
    if len(S) < 2:
        return list(S)

    def ends_of(i):
        p = S[i]
        h, t = p[:DIR_PX], p[max(0, len(p) - DIR_PX):]
        d0, d1 = h[0] - h[-1], t[-1] - t[0]
        n0, n1 = np.linalg.norm(d0), np.linalg.norm(d1)
        return (np.array([p[0], p[-1]]),
                np.array([d0 / n0 if n0 > 1e-9 else np.zeros(2),
                          d1 / n1 if n1 > 1e-9 else np.zeros(2)]))

    while True:
        idx = np.flatnonzero(alive)
        if len(idx) < 2:
            break
        n = len(idx)
        P = np.empty((n, 2, 2))          # [stroke, end, coord]
        D = np.empty((n, 2, 2))          # outward unit direction at that end
        for k, i in enumerate(idx):
            P[k], D[k] = ends_of(int(i))
        # V[i, j, ei, ej] = tip(j, ej) - tip(i, ei)
        V = P[None, :, None, :, :] - P[:, None, :, None, :]
        G = np.linalg.norm(V, axis=-1)
        ok = (G <= gap) & (G >= 1e-9)
        ok[np.arange(n), np.arange(n)] = False
        T = np.arccos(np.clip(-np.einsum("iek,jfk->ijef", D, D), -1.0, 1.0))
        link = V / np.where(ok, G, 1.0)[..., None]
        T2 = np.arccos(np.clip(np.einsum("iek,ijefk->ijef", D, link), -1.0, 1.0))
        ok &= (T < turn) & (T2 < turn)
        cand = np.flatnonzero(ok.ravel())
        if not len(cand):
            break
        sc = (T + T2).ravel()[cand]
        gg = G.ravel()[cand]
        pick = int(cand[np.lexsort((np.arange(len(cand)), gg, sc))[0]])
        i, j, ei, ej = np.unravel_index(pick, (n, n, 2, 2))
        i, j = int(idx[i]), int(idx[j])
        a = S[i] if ei else S[i][::-1]
        b = S[j][::-1] if ej else S[j]
        S[i] = np.vstack([a, b])
        alive[j] = False
    return [S[i] for i in range(len(S)) if alive[i]]


# ===========================================================================
# 5. the two tracers
# ===========================================================================
def trace_lines(mask, rdp_tol=RDP_TOL, min_px=MIN_PX, bridge=True):
    """Uniform-width line art -> centreline polylines in pixel (x, y)."""
    skel = thin(mask)
    nodes, edges = skeleton_graph(skel)
    edges = _prune_spurs(nodes, edges)
    edges = _merge_nodes(nodes, edges)
    strokes = chain(edges, pair_at_junctions(edges))
    strokes = [s for s in strokes if plen(s) >= min_px]
    if bridge:
        strokes = bridge_endpoints(strokes)
    return [rdp(s, rdp_tol) for s in strokes if plen(s) >= min_px], skel


def trace_contours(field, rdp_tol=RDP_TOL, min_px=MIN_PX, level=0.5):
    """Solid shapes -> closed boundary contours (marching squares, incl. holes).

    `field` is the ink-coverage map, not the thresholded mask: iso-lines of
    the continuous alpha land between pixels and follow the diagonal of a
    letterform smoothly, where the same contour of a boolean mask staircases
    one pixel at a time and RDP then preserves every step of the staircase.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Z = np.pad(np.asarray(field, float), 1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cs = ax.contour(Z, levels=[level])
    segs = [np.asarray(s, float) - 1.0 for s in cs.allsegs[0]]
    plt.close(fig)
    out = []
    for s in segs:
        if plen(s) < min_px or len(s) < 4:
            continue
        if np.linalg.norm(s[0] - s[-1]) > 1e-6:
            s = np.vstack([s, s[0]])
        out.append(rdp(s, rdp_tol))
    return out


# ===========================================================================
# 6. image -> strokes on the paper
# ===========================================================================
def trace_logo(path, upscale=UPSCALE, rdp_tol=RDP_TOL, min_px=MIN_PX,
               letter_erode=LETTER_ERODE):
    """-> (strokes_px, debug).  strokes_px: [{pts, color, kind}] in pixel (x,y)."""
    m = load_masks(path, upscale)
    letters, orange_lines = split_thick(m["orange"], letter_erode)
    grey_lines = m["grey"]
    out, dbg = [], dict(masks=m, letters=letters, orange_lines=orange_lines)
    gs, dbg["skel_grey"] = trace_lines(grey_lines, rdp_tol, min_px)
    os_, dbg["skel_orange"] = trace_lines(orange_lines, rdp_tol, min_px)
    for p in gs:
        out.append(dict(pts=p, color="grey", kind="outline"))
    for p in os_:
        out.append(dict(pts=p, color="orange", kind="outline"))
    near = letters
    for _ in range(3):
        near = dilate(near)
    field = np.where(near, m["alpha_orange"], 0.0)
    for p in trace_contours(field, rdp_tol, min_px, level=ALPHA_ON):
        out.append(dict(pts=p, color="orange", kind="letter"))
    dbg["shape"] = m["shape"]
    return out, dbg


def to_sheet(strokes_px, sheet, margin=0.06, min_len=0.025, target_width=None,
             offset=(0.0, 0.0), rotate_deg=0.0):
    """Pixel strokes -> paper strokes in metres: one uniform scale, centred.

    Keeping the aspect ratio is non-negotiable (it is a logo), so the scale is
    whichever of the two sheet dimensions binds first; `target_width` is an
    upper bound, not a target, and the caller is told when it did not bind.

    `offset` (dx, dy) in metres moves the logo off the sheet centre — the knob
    the placement search turns, together with `target_width`.  It is applied as
    asked and NOT clamped; `info["fits"]` reports whether the placed logo is
    still inside the margin, which is the caller's cue to reject the candidate.

    `rotate_deg` TURNS THE WHOLE LOGO before it is fitted, which matters as
    soon as the paper stops being roughly the logo's own shape.  The CSAIL
    logo is 1.31 times wider than it is tall; the merged six-arm canvas is
    1.8034 x 3.63064 m, i.e. twice as TALL as it is wide.  Upright, the logo's
    width binds and two thirds of the canvas is unusable at any size; turned
    90 degrees its long axis runs along the canvas's long axis and the same
    width limit buys a logo 1.31x bigger in every dimension (1.71x the area).
    Applied to the pixel polylines about their own bounding-box centre, so
    everything after this line — the aspect-preserving fit, the margin test,
    the offset — is untouched and sees only a differently-shaped logo.
    """
    if rotate_deg:
        a = np.deg2rad(float(rotate_deg))
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        strokes_px = [dict(s, pts=np.asarray(s["pts"], float) @ R.T)
                      for s in strokes_px]
    P = np.vstack([s["pts"] for s in strokes_px])
    x0, x1 = P[:, 0].min(), P[:, 0].max()
    y0, y1 = P[:, 1].min(), P[:, 1].max()
    w, h = max(x1 - x0, 1e-9), max(y1 - y0, 1e-9)
    avail = (sheet[0] - 2 * margin, sheet[1] - 2 * margin)
    scale = min(avail[0] / w, avail[1] / h)
    if target_width is not None:
        scale = min(scale, target_width / w)
    cx = sheet[0] / 2 + float(offset[0])
    cy = sheet[1] / 2 + float(offset[1])
    out = []
    for k, s in enumerate(strokes_px):
        p = np.asarray(s["pts"], float)
        xy = np.column_stack([cx + (p[:, 0] - (x0 + x1) / 2) * scale,
                              cy - (p[:, 1] - (y0 + y1) / 2) * scale])
        if plen(xy) < min_len:
            continue
        out.append(dict(pts=xy, color=s["color"], kind=s["kind"], id=len(out)))
    lw, lh = w * scale, h * scale
    info = dict(scale=scale, logo_w=lw, logo_h=lh, center=(cx, cy),
                rotate_deg=float(rotate_deg),
                offset=(float(offset[0]), float(offset[1])),
                px_per_m=1.0 / scale, dropped_short=len(strokes_px) - len(out),
                fits=bool(cx - lw / 2 >= margin - 1e-9
                          and cx + lw / 2 <= sheet[0] - margin + 1e-9
                          and cy - lh / 2 >= margin - 1e-9
                          and cy + lh / 2 <= sheet[1] - margin + 1e-9))
    return out, info


def total_length(strokes):
    return float(sum(plen(s["pts"]) for s in strokes))


# ===========================================================================
# 7. ANY picture -> strokes:  the CSAIL tracer's machinery, without its palette
#
# `trace_logo` above is the CSAIL mark's tracer and it knows three things about
# that mark that are not true of a picture in general: there are exactly two
# inks and it knows their RGB; the source is a 222 px gif that wants upsampling
# by a fixed 4; and a solid shape is a letterform, told from a line by a
# hand-tuned six-erosion test.  Everything BELOW the palette — thinning, the
# skeleton graph, junction-through pairing, spur pruning, endpoint bridging,
# marching-squares contours, RDP, `to_sheet` — is about ink and not about that
# logo, so this section reuses all of it and replaces only the three
# assumptions:
#
#   PALETTE     `detect_inks` clusters the picture's own ink colours (k-means
#               on the pixels that are not background) and `unmix_palette`
#               generalises `unmix`'s 3x2 least squares to 3xk, so an
#               antialiased pixel is still resolved by HUE rather than by a
#               brightness threshold.  `--inks 1` on black line art is the case
#               that matters most and it is the cheapest one.
#   SCALE       `load_art` resizes to a WORKING RESOLUTION instead of
#               upsampling by a constant.  4x is right for a 222 px gif and
#               absurd for a 2000 px png; the pipeline's tuned pixel constants
#               (SPUR_PX, MERGE_PX, BRIDGE_PX, RDP_TOL, MIN_PX) are all lengths
#               in working pixels, so fixing the working resolution is what
#               makes them mean the same thing on every source.
#   SOLID       `auto_fill_erode` MEASURES the erosion count instead of
#               assuming it: the distance transform sampled ON THE SKELETON is
#               the local half-width of whatever the skeleton is running down,
#               so its median over the whole mask is the picture's own line
#               half-width, and a shape whose inscribed disk is much bigger
#               than that is a fill.  It reproduces LETTER_ERODE on the CSAIL
#               gif without being told about it.
#
# WHY A FILL IS TRACED AS ITS BOUNDARY AND A LINE AS ITS CENTRELINE.  A pen has
# one width; the ink is the stroke, so a uniform-width stroke wants its
# centreline.  A solid region has no centreline worth drawing — the skeleton of
# a filled letterform is a stick figure and the skeleton of the Trollface's
# mouth is a fork — but its BOUNDARY is exactly what a person drawing it with a
# pen would draw: the grin curve plus one closed loop per tooth.  Which of the
# two a component gets is the only judgement in here, and it is measured.
# ===========================================================================
WORK_PX = 900              # long side of the working raster.  Every pixel
                           # constant above is a length at THIS resolution.
FILL_K = 1.35              # fill-erode radius, in median line half-widths
FILL_FRAC = 0.10           # area fraction surviving the erosion -> "solid"
MAX_INKS = 4               # ink clusters `--inks auto` will consider
INK_TOL = 46.0             # RGB radius inside which one cluster is one ink
ART_MIN_PX = 10            # working pixels; shorter than this is a speck

# Ink names, so a report reads "black 12.4 m" and not "cluster 0".  Nearest of
# these in RGB wins; ties and repeats get a numeric suffix.
INK_NAMES = (
    ("black", (0, 0, 0)), ("grey", (128, 128, 128)), ("white", (255, 255, 255)),
    ("red", (220, 30, 30)), ("orange", (235, 130, 20)), ("yellow", (240, 220, 40)),
    ("green", (40, 160, 60)), ("cyan", (40, 190, 210)), ("blue", (40, 80, 200)),
    ("purple", (130, 60, 180)), ("magenta", (215, 50, 160)), ("brown", (130, 80, 40)),
)


def ink_name(rgb, taken=()):
    """Nearest named colour, uniquified against `taken`. -> str."""
    c = np.asarray(rgb, float)
    base = min(INK_NAMES, key=lambda kv: float(np.sum((c - kv[1]) ** 2)))[0]
    if base not in taken:
        return base
    k = 2
    while f"{base}{k}" in taken:
        k += 1
    return f"{base}{k}"


# ---------------------------------------------------------------------------
PAPER_RGB = (255, 255, 255)    # what transparency is composited onto


def load_art(path, work_px=WORK_PX, bg_rgb=None, paper=PAPER_RGB):
    """Any raster -> (rgb uint8 (H,W,3) on an opaque background, bg tuple).

    Transparency is COMPOSITED onto `paper` rather than thresholded, so the same
    drawing exported with a white surround and with a transparent one traces to
    the same strokes.

    THE BACKGROUND IS READ AFTER THE COMPOSITE, NOT BEFORE, and that ordering is
    the whole of this function.  `assets/csail/csail_old_med.gif` has a
    TRANSPARENT white surround: of its 1 480 border pixels 1 245 are transparent
    and the 235 that are opaque are the orange rule that runs off the edge — so
    the modal opaque border colour is ORANGE, the unmixer is handed the logo's
    own ink as its paper, and four grey "inks" are detected in a two-ink
    picture.  Composited first, the same border is 84 % white and the answer is
    white.  Reading a background off pixels that are not there is the failure
    mode; there are no pixels that are not there after a composite.
    """
    im = Image.open(path).convert("RGBA")
    flat = Image.new("RGBA", im.size, (*tuple(int(v) for v in paper), 255))
    flat.alpha_composite(im)
    rgb = np.asarray(flat.convert("RGB"))
    if bg_rgb is None:
        border = np.zeros(rgb.shape[:2], bool)
        border[:2, :] = border[-2:, :] = border[:, :2] = border[:, -2:] = True
        px = rgb[border]
        q = px.astype(int) // 16
        key = q[:, 0] * 4096 + q[:, 1] * 64 + q[:, 2]
        mode = int(np.bincount(key).argmax())
        bg_rgb = tuple(int(v) for v in np.median(px[key == mode], axis=0))
    w, h = im.size
    k = float(work_px) / max(w, h)
    if abs(k - 1.0) > 1e-9:
        flat = flat.resize((max(1, int(round(w * k))), max(1, int(round(h * k)))),
                           Image.LANCZOS)
        rgb = np.asarray(flat.convert("RGB"))
    return rgb, tuple(int(v) for v in bg_rgb)


CORE_ERODE = 2             # erosions that leave only a stroke's INTERIOR
INK_MIN_FRAC = 0.02        # share of core ink a real ink must hold


def ink_core(rgb, bg, alpha_on=ALPHA_ON, erode_n=CORE_ERODE):
    """Pixels that are ink and are not an EDGE of it. -> bool (H,W).

    THE ANTIALIASED RIM IS WHERE THE FALSE INKS COME FROM.  Every dark line on
    white paper is bordered by a band of every grey between the two, and on a
    900 px working raster that band is 30 % of the ink by area — enough for
    k-means to find "a mid grey", "a dark grey" and "black" in a picture drawn
    with one pen, and then for `ink_masks` to cut every stroke into three
    interleaved fragments.  Two erosions remove the rim and leave the interior,
    where a pen's colour is the only colour there is: on `csail_old_med.gif`
    that takes the k = 2 cluster spread from 50/47 to 19/25 (so the ink count
    stops at the two that are really there instead of running to four), and on
    the Trollface from 12.7 to 0.9.
    """
    d = np.linalg.norm(rgb.astype(float) - np.asarray(bg, float), axis=2)
    ref = np.asarray(bg, float)
    ref = max(float(np.linalg.norm(ref)), float(np.linalg.norm(255 - ref)), 1e-9)
    ink = d >= alpha_on * ref
    core = ink
    for _ in range(int(erode_n)):
        e = erode(core)
        if e.sum() < 200:          # a hairline drawing has no interior to spare
            break
        core = e
    return core if core.any() else ink


def detect_inks(rgb, bg, n=None, max_inks=MAX_INKS, tol=INK_TOL,
                alpha_on=ALPHA_ON, min_frac=INK_MIN_FRAC):
    """The picture's own ink colours. -> list of (r, g, b) ints, darkest first.

    `n` fixes the count; `None` chooses it as the SMALLEST k whose clusters are
    each tighter than `tol` in RGB — the question "is this drawn with one pen or
    with three" asked of the pixels rather than of the caller.  Only interior
    pixels vote (`ink_core`), a cluster holding less than `min_frac` of them is
    not an ink, and centres that end up within `tol` of each other are merged,
    so an over-large `n` degrades to the right answer instead of shredding one
    ink into three.
    """
    px = rgb[ink_core(rgb, bg, alpha_on)].astype(float)
    if len(px) == 0:
        return [(0, 0, 0)]
    if len(px) > 40000:                        # k-means on a fixed sample
        px = px[np.linspace(0, len(px) - 1, 40000).astype(int)]
    best = _kmeans(px, 1)
    for k in range(1, int(max_inks) + 1 if n is None else int(n) + 1):
        if n is not None and k != int(n):
            continue
        C = _kmeans(px, k)
        lab = np.argmin(((px[:, None, :] - C[None]) ** 2).sum(2), axis=1)
        share = np.array([float(np.mean(lab == j)) for j in range(k)])
        spread = max((float(np.percentile(np.linalg.norm(px[lab == j] - C[j],
                                                         axis=1), 90))
                      if share[j] > 0 else 0.0) for j in range(k))
        if n is None and k > 1 and share.min() < min_frac:
            break                              # the k-th "ink" is a rounding rim
        best = C
        if n is not None or spread <= tol:
            break
    return _merge_close(best, tol)


def _merge_close(C, tol):
    """Single-linkage merge of centres within `tol`. -> list of (r,g,b), dark first."""
    out = []
    for c in sorted(np.asarray(C, float), key=lambda v: float(np.sum(v))):
        if out and float(np.linalg.norm(c - out[-1])) <= tol:
            out[-1] = (out[-1] + c) / 2.0
        else:
            out.append(c)
    return [tuple(int(round(v)) for v in c) for c in out]


def _kmeans(px, k, iters=25, seed=0):
    """k-means++ over (N,3) float. -> (k,3) centres.  Deterministic."""
    rng = np.random.default_rng(seed)
    C = [px[rng.integers(len(px))]]
    for _ in range(k - 1):
        d2 = np.min(((px[:, None, :] - np.asarray(C)[None]) ** 2).sum(2), axis=1)
        s = float(d2.sum())
        C.append(px[rng.choice(len(px), p=d2 / s) if s > 0
                    else rng.integers(len(px))])
    C = np.asarray(C, float)
    for _ in range(iters):
        lab = np.argmin(((px[:, None, :] - C[None]) ** 2).sum(2), axis=1)
        new = np.array([px[lab == j].mean(0) if np.any(lab == j) else C[j]
                        for j in range(k)])
        if np.allclose(new, C):
            break
        C = new
    return C


def unmix_palette(rgb, palette, bg=BG_RGB):
    """(H,W,3) -> (k, H, W) ink coverages.  `unmix` generalised to k inks.

    Same model and same reason: an antialiased pixel is `bg + sum a_i (c_i-bg)`
    and the least-squares solve of that 3xk system recovers the coverages, so a
    half-covered dark pixel and a fully covered mid-grey one are told apart by
    where they sit in colour space rather than by how bright they are.  With
    ONE ink the solve degenerates to a projection onto that ink's own axis,
    which is exactly the right answer for black line art.
    """
    c = np.asarray(rgb, float) - np.asarray(bg, float)
    M = np.column_stack([np.asarray(p, float) - np.asarray(bg, float)
                         for p in palette])
    a = c.reshape(-1, 3) @ np.linalg.pinv(M).T
    a = a.reshape(rgb.shape[0], rgb.shape[1], len(palette))
    return np.moveaxis(np.clip(a, 0.0, 1.5), 2, 0)


def ink_masks(alphas, alpha_on=ALPHA_ON):
    """(k,H,W) coverages -> (k,H,W) bool masks, each pixel to at most one ink."""
    A = np.asarray(alphas, float)
    on = A > alpha_on
    if len(A) > 1:
        win = np.argmax(A, axis=0)
        on &= (np.arange(len(A))[:, None, None] == win[None])
    return on


# ---------------------------------------------------------------------------
def distance_transform(mask, max_r=200):
    """Chessboard distance to the background, by iterated erosion. -> int array.

    numpy only (there is no scipy here), and it costs one erosion per pixel of
    the widest thing in the picture, which on line art is a few dozen.
    """
    dt = np.zeros(mask.shape, np.int32)
    cur = np.asarray(mask, bool)
    for _ in range(int(max_r)):
        if not cur.any():
            break
        dt += cur
        cur = erode(cur)
    return dt


def line_halfwidth(mask, skel=None):
    """The picture's own line half-width in working pixels. -> float.

    The distance transform sampled ON THE SKELETON is the local half-width of
    whatever the skeleton runs down, and the skeleton is one pixel per unit
    LENGTH — so the median over it is length-weighted and a picture that is
    mostly line reports its lines, however large a fill it also carries.
    """
    sk = thin(mask) if skel is None else skel
    if not sk.any():
        return 1.0
    return float(np.median(distance_transform(mask)[sk]))


def auto_fill_erode(mask, k=FILL_K, skel=None, lo=3, hi=40):
    """Erosions a LINE must not survive, measured off this picture. -> int.

    On `assets/csail/csail_old_med.gif` at the 900 px working resolution this
    returns 5 against the hand-tuned LETTER_ERODE of 6 (which was measured at
    888 px), and on the Trollface it returns 8 — the value at which both filled
    eyes and the solid mouth are fills and the bold jaw outline, whose half
    width is only 3 px less than an eye's, is still a line.
    """
    return int(np.clip(round(k * line_halfwidth(mask, skel)), lo, hi))


# ---------------------------------------------------------------------------
def trace_ink(mask, alpha, rdp_tol=RDP_TOL, min_px=ART_MIN_PX, fill_erode=None,
              fill_frac=FILL_FRAC, bridge=True):
    """One ink's mask -> (strokes, debug).  Fills by contour, lines by centreline.

    `strokes` is [{pts, kind}] with `kind` in {"outline", "fill"}; the two are
    the two tracers of section 5 and the split between them is section 7's one
    judgement.
    """
    mask = np.asarray(mask, bool)
    out = dict(fill_erode=0, thick=np.zeros(mask.shape, bool), skel=None)
    if not mask.any():
        return [], out
    skel_all = thin(mask)
    ne = auto_fill_erode(mask, skel=skel_all) if fill_erode is None \
        else int(fill_erode)
    thick, thin_m = split_thick(mask, ne, fill_frac)
    out.update(fill_erode=ne, thick=thick,
               halfwidth=line_halfwidth(mask, skel_all))
    strokes = []
    if thin_m.any():
        lines, out["skel"] = trace_lines(thin_m, rdp_tol, min_px, bridge=bridge)
        strokes += [dict(pts=p, kind="outline") for p in lines]
    if thick.any():
        # the contour is taken off the COVERAGE field, not the boolean mask, so
        # a diagonal edge comes out smooth instead of stepped (see
        # `trace_contours`); dilating the region first is what lets the iso-line
        # sit OUTSIDE the last ink pixel instead of being clipped to it.
        near = thick
        for _ in range(3):
            near = dilate(near)
        field = np.where(near, np.asarray(alpha, float), 0.0)
        strokes += [dict(pts=p, kind="fill")
                    for p in trace_contours(field, rdp_tol, min_px, level=ALPHA_ON)]
    return strokes, out


def trace_art(path, n_inks=None, work_px=WORK_PX, rdp_tol=RDP_TOL,
              min_px=ART_MIN_PX, fill_erode=None, fill_frac=FILL_FRAC,
              alpha_on=ALPHA_ON, palette=None, bg_rgb=None, bridge=True):
    """Any raster -> (strokes_px, debug).  The generic front door's tracer.

    `strokes_px` is [{pts, color, kind}] in working-pixel (x, y), ready for
    `to_sheet` — the same shape `trace_logo` returns, so everything downstream
    (placement, allocation, the schedule, the animation) is unchanged.

    `n_inks` fixes the ink count; None measures it (`detect_inks`).  A picture
    with ONE ink needs no colour partition and no pen swap, which is the whole
    difference between drawing the Trollface and drawing the CSAIL mark.
    """
    rgb, bg = load_art(path, work_px, bg_rgb)
    pal = list(palette) if palette else detect_inks(rgb, bg, n_inks,
                                                   alpha_on=alpha_on)
    names, taken = [], []
    for c in pal:
        names.append(ink_name(c, taken))
        taken.append(names[-1])
    alphas = unmix_palette(rgb, pal, bg)
    masks = ink_masks(alphas, alpha_on)
    out, dbg = [], dict(shape=rgb.shape[:2], rgb=rgb, bg=bg, palette=pal,
                        names=names, alphas=alphas, masks=masks, per_ink={})
    for i, nm in enumerate(names):
        st, d = trace_ink(masks[i], alphas[i], rdp_tol, min_px, fill_erode,
                          fill_frac, bridge)
        dbg["per_ink"][nm] = d
        for s in st:
            out.append(dict(pts=s["pts"], color=nm, kind=s["kind"]))
    return out, dbg


# ===========================================================================
# 8. vector in, the same strokes out
#
# An SVG is already the thing the raster tracer spends its effort recovering —
# a set of paths — so tracing one is FLATTENING, not thinning: every curve is
# subdivided until it is straight to `flat_tol` and handed on as a polyline in
# the same working-pixel frame.  The ink is the element's own `stroke` (or its
# `fill` where it has no stroke), snapped to the nearest name, so `--inks` means
# the same thing on both inputs.
#
# WHAT THIS DELIBERATELY DOES NOT DO: strokes are drawn as their PATH and fills
# as their OUTLINE, which is the vector analogue of section 7's split and is
# right for line art; it is wrong for a poster made of overlapping filled
# shapes, where the visible edge is a boolean of many outlines and only the
# raster path can see it.  `--rasterize` is the answer there and it is the
# caller's call, not this function's.
# ===========================================================================
import re as _re                                                # noqa: E402
import xml.etree.ElementTree as _ET                             # noqa: E402

_NUM = _re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")
_CMD = _re.compile(r"([MmZzLlHhVvCcSsQqTtAa])")
SVG_FLAT_TOL = 0.25        # user units; curve subdivision flatness


def _cross2(u, v):
    """The z of a 2-D cross product, as a magnitude. -> float.

    `np.cross` accepted 2-vectors and returned this scalar until NumPy 2.0
    REMOVED that overload; on 2.x the same call raises "Both input arrays must
    be (arrays of) 3-dimensional vectors".  The arithmetic is one line and
    exact, so it is written out rather than depending on which NumPy is
    installed.  (This is the whole of `tests/test_draw.py`'s SVG failure — an
    environment change, not a planner one.)
    """
    return abs(float(u[0]) * float(v[1]) - float(u[1]) * float(v[0]))


def _bezier(p0, p1, p2, p3, tol, depth=0):
    """Adaptive cubic subdivision -> list of points EXCLUDING p0."""
    d = _cross2(p3 - p0, p1 - p0) + _cross2(p3 - p0, p2 - p0)
    L = np.linalg.norm(p3 - p0)
    if depth >= 12 or (L > 1e-12 and d / L <= tol) or L <= tol:
        return [p3]
    a, b = (p0 + p1) / 2, (p1 + p2) / 2
    c = (p2 + p3) / 2
    ab, bc = (a + b) / 2, (b + c) / 2
    m = (ab + bc) / 2
    return (_bezier(p0, a, ab, m, tol, depth + 1)
            + _bezier(m, bc, c, p3, tol, depth + 1))


def _arc(p0, r, rot, laf, sf, p1, tol):
    """SVG elliptical arc -> list of points EXCLUDING p0 (F.6.5 + sampling)."""
    rx, ry = abs(float(r[0])), abs(float(r[1]))
    if rx < 1e-12 or ry < 1e-12 or np.allclose(p0, p1):
        return [p1]
    th = np.deg2rad(float(rot))
    R = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
    d = R @ ((p0 - p1) / 2.0)
    lam = (d[0] / rx) ** 2 + (d[1] / ry) ** 2
    if lam > 1:
        rx, ry = rx * np.sqrt(lam), ry * np.sqrt(lam)
    num = max(rx ** 2 * ry ** 2 - rx ** 2 * d[1] ** 2 - ry ** 2 * d[0] ** 2, 0.0)
    den = rx ** 2 * d[1] ** 2 + ry ** 2 * d[0] ** 2
    co = np.sqrt(num / den) if den > 1e-18 else 0.0
    if laf == sf:
        co = -co
    cp = co * np.array([rx * d[1] / ry, -ry * d[0] / rx])
    c = R.T @ cp + (p0 + p1) / 2.0

    def ang(v):
        return np.arctan2(v[1], v[0])

    v0 = np.array([(d[0] - cp[0]) / rx, (d[1] - cp[1]) / ry])
    v1 = np.array([(-d[0] - cp[0]) / rx, (-d[1] - cp[1]) / ry])
    a0, da = ang(v0), ang(v1) - ang(v0)
    if not sf and da > 0:
        da -= 2 * np.pi
    elif sf and da < 0:
        da += 2 * np.pi
    n = max(2, int(np.ceil(abs(da) / max(2 * np.arccos(
        np.clip(1 - tol / max(max(rx, ry), 1e-9), -1, 1)), 1e-3))))
    t = a0 + da * np.arange(1, n + 1) / n
    P = np.column_stack([rx * np.cos(t), ry * np.sin(t)]) @ R + c
    return [np.asarray(p, float) for p in P]


def svg_path_points(d, tol=SVG_FLAT_TOL):
    """One `d` attribute -> list of (N,2) polylines (subpaths), user units."""
    toks = [t for t in _CMD.split(d) if t.strip()]
    subs, cur, start = [], [], None
    p = np.zeros(2)
    prev_c = prev_q = None
    i = 0
    cmd = None
    while i < len(toks):
        if _CMD.fullmatch(toks[i]):
            cmd = toks[i]
            i += 1
            args = []
            if i < len(toks) and not _CMD.fullmatch(toks[i]):
                args = [float(x) for x in _NUM.findall(toks[i])]
                i += 1
        else:
            args = [float(x) for x in _NUM.findall(toks[i])]
            i += 1
        rel = cmd.islower()
        C = cmd.upper()
        k = dict(M=2, L=2, H=1, V=1, C=6, S=4, Q=4, T=2, A=7, Z=0)[C]
        if C == "Z":
            if len(cur) > 1:
                if start is not None and not np.allclose(cur[-1], start):
                    cur.append(start.copy())
                subs.append(np.array(cur, float))
            cur, p = [], (start.copy() if start is not None else p)
            if start is not None:
                cur = [start.copy()]
            prev_c = prev_q = None
            continue
        j = 0
        first = True
        while j + k <= len(args):
            a = args[j:j + k]
            j += k
            if C == "M":
                q = (p + a) if rel else np.array(a, float)
                if first:
                    if len(cur) > 1:
                        subs.append(np.array(cur, float))
                    cur, start = [q.copy()], q.copy()
                    C, cmd = "L", ("l" if rel else "L")   # implicit lineto after
                else:
                    cur.append(q.copy())
                p, prev_c, prev_q = q, None, None
            elif C == "L":
                q = (p + a) if rel else np.array(a, float)
                cur.append(q.copy())
                p, prev_c, prev_q = q, None, None
            elif C in ("H", "V"):
                q = p.copy()
                idx = 0 if C == "H" else 1
                q[idx] = (p[idx] + a[0]) if rel else a[0]
                cur.append(q.copy())
                p, prev_c, prev_q = q, None, None
            elif C in ("C", "S"):
                if C == "C":
                    c1 = (p + a[0:2]) if rel else np.array(a[0:2], float)
                    c2 = (p + a[2:4]) if rel else np.array(a[2:4], float)
                    q = (p + a[4:6]) if rel else np.array(a[4:6], float)
                else:
                    c1 = 2 * p - prev_c if prev_c is not None else p.copy()
                    c2 = (p + a[0:2]) if rel else np.array(a[0:2], float)
                    q = (p + a[2:4]) if rel else np.array(a[2:4], float)
                cur += [x.copy() for x in _bezier(p, c1, c2, q, tol)]
                p, prev_c, prev_q = q, c2, None
            elif C in ("Q", "T"):
                if C == "Q":
                    c1 = (p + a[0:2]) if rel else np.array(a[0:2], float)
                    q = (p + a[2:4]) if rel else np.array(a[2:4], float)
                else:
                    c1 = 2 * p - prev_q if prev_q is not None else p.copy()
                    q = (p + a[0:2]) if rel else np.array(a[0:2], float)
                cur += [x.copy() for x in _bezier(p, p + 2 * (c1 - p) / 3,
                                                  q + 2 * (c1 - q) / 3, q, tol)]
                p, prev_c, prev_q = q, None, c1
            elif C == "A":
                q = (p + a[5:7]) if rel else np.array(a[5:7], float)
                cur += [x.copy() for x in _arc(p, a[0:2], a[2], bool(a[3]),
                                               bool(a[4]), q, tol)]
                p, prev_c, prev_q = q, None, None
            first = False
    if len(cur) > 1:
        subs.append(np.array(cur, float))
    return subs


def _svg_transform(s):
    """A `transform` attribute -> 3x3 matrix (translate/scale/rotate/matrix)."""
    T = np.eye(3)
    for name, body in _re.findall(r"(\w+)\s*\(([^)]*)\)", s or ""):
        v = [float(x) for x in _NUM.findall(body)]
        M = np.eye(3)
        if name == "translate":
            M[0, 2], M[1, 2] = v[0], (v[1] if len(v) > 1 else 0.0)
        elif name == "scale":
            M[0, 0], M[1, 1] = v[0], (v[1] if len(v) > 1 else v[0])
        elif name == "rotate":
            a = np.deg2rad(v[0])
            R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            M[:2, :2] = R
            if len(v) > 2:
                M[:2, 2] = np.array(v[1:3]) - R @ np.array(v[1:3])
        elif name == "matrix" and len(v) >= 6:
            M = np.array([[v[0], v[2], v[4]], [v[1], v[3], v[5]], [0, 0, 1]])
        elif name in ("skewX", "skewY"):
            M[0, 1 if name == "skewX" else 0] = np.tan(np.deg2rad(v[0]))
            if name == "skewY":
                M[1, 0], M[0, 1] = np.tan(np.deg2rad(v[0])), 0.0
        T = T @ M
    return T


_CSS = {"black": (0, 0, 0), "white": (255, 255, 255), "red": (255, 0, 0),
        "green": (0, 128, 0), "blue": (0, 0, 255), "yellow": (255, 255, 0),
        "orange": (255, 165, 0), "grey": (128, 128, 128), "gray": (128, 128, 128),
        "purple": (128, 0, 128), "cyan": (0, 255, 255), "magenta": (255, 0, 255),
        "brown": (165, 42, 42), "silver": (192, 192, 192), "navy": (0, 0, 128)}


def _svg_color(v):
    """A paint value -> (r,g,b) or None for none/unset."""
    if not v:
        return None
    v = v.strip().lower()
    if v in ("none", "transparent", "currentcolor", "inherit"):
        return None
    if v.startswith("#"):
        h = v[1:]
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        if len(h) >= 6:
            return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
        return None
    m = _re.match(r"rgba?\(([^)]*)\)", v)
    if m:
        n = [float(x) for x in _NUM.findall(m.group(1))]
        if len(n) >= 3:
            return tuple(int(round(x)) for x in n[:3])
    return _CSS.get(v)


def _svg_style(el, inherited):
    """Element paint, with `style=` overriding presentation attributes."""
    out = dict(inherited)
    for k in ("stroke", "fill"):
        if el.get(k) is not None:
            out[k] = el.get(k)
    for part in (el.get("style") or "").split(";"):
        if ":" in part:
            k, v = part.split(":", 1)
            if k.strip() in ("stroke", "fill"):
                out[k.strip()] = v.strip()
    return out


def trace_svg(path, work_px=WORK_PX, flat_tol=SVG_FLAT_TOL, rdp_tol=RDP_TOL,
              min_px=ART_MIN_PX, n_inks=None, palette=None):
    """An SVG file -> (strokes_px, debug), the same shape as `trace_art`.

    The result is scaled so its long side is `work_px`, which is what makes the
    downstream pixel constants (RDP tolerance, speck length) mean the same on a
    vector input as on a raster one.
    """
    root = _ET.parse(path).getroot()
    ns = {"s": "http://www.w3.org/2000/svg"}
    del ns
    subs = []

    def walk(el, T, style):
        st = _svg_style(el, style)
        T = T @ _svg_transform(el.get("transform"))
        tag = el.tag.split("}")[-1]
        polys = []
        if tag == "path" and el.get("d"):
            polys = svg_path_points(el.get("d"), flat_tol)
        elif tag in ("polyline", "polygon") and el.get("points"):
            v = [float(x) for x in _NUM.findall(el.get("points"))]
            P = np.array(v[:2 * (len(v) // 2)], float).reshape(-1, 2)
            if tag == "polygon" and len(P) > 2:
                P = np.vstack([P, P[0]])
            polys = [P] if len(P) > 1 else []
        elif tag == "line":
            polys = [np.array([[float(el.get("x1", 0)), float(el.get("y1", 0))],
                               [float(el.get("x2", 0)), float(el.get("y2", 0))]])]
        elif tag == "rect":
            x, y = float(el.get("x", 0)), float(el.get("y", 0))
            w, h = float(el.get("width", 0)), float(el.get("height", 0))
            polys = [np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h],
                               [x, y]], float)]
        elif tag in ("circle", "ellipse"):
            cx, cy = float(el.get("cx", 0)), float(el.get("cy", 0))
            rx = float(el.get("rx", el.get("r", 0)))
            ry = float(el.get("ry", el.get("r", 0)))
            t = np.linspace(0, 2 * np.pi, 129)
            polys = [np.column_stack([cx + rx * np.cos(t), cy + ry * np.sin(t)])]
        for P in polys:
            if len(P) < 2:
                continue
            Q = (T @ np.column_stack([P, np.ones(len(P))]).T).T[:, :2]
            paint = _svg_color(st.get("stroke")) or _svg_color(st.get("fill"))
            subs.append((Q, paint if paint is not None else (0, 0, 0),
                         "outline" if _svg_color(st.get("stroke")) else "fill"))
        for ch in el:
            walk(ch, T, st)

    walk(root, np.eye(3), dict(stroke=None, fill="black"))
    if not subs:
        return [], dict(shape=(0, 0), palette=[], names=[])
    P = np.vstack([q for q, _, _ in subs])
    lo, hi = P.min(0), P.max(0)
    k = float(work_px) / max(float(hi[0] - lo[0]), float(hi[1] - lo[1]), 1e-9)
    cols = [c for _, c, _ in subs]
    pal = list(palette) if palette else _svg_palette(cols, n_inks)
    names, taken = [], []
    for c in pal:
        names.append(ink_name(c, taken))
        taken.append(names[-1])
    out = []
    for q, c, kind in subs:
        p = rdp((q - lo) * k, rdp_tol)
        if plen(p) < min_px:
            continue
        j = int(np.argmin([sum((a - b) ** 2 for a, b in zip(c, pc))
                           for pc in pal]))
        out.append(dict(pts=p, color=names[j], kind=kind))
    return out, dict(shape=(int(round((hi[1] - lo[1]) * k)),
                            int(round((hi[0] - lo[0]) * k))),
                     palette=pal, names=names, bg=BG_RGB, per_ink={})


def _svg_palette(cols, n_inks=None, tol=INK_TOL):
    """The distinct paints an SVG uses. -> list of (r,g,b), darkest first."""
    px = np.array(cols, float)
    if n_inks is not None:
        C = _kmeans(px, min(int(n_inks), len(np.unique(px, axis=0))))
        return [tuple(int(round(v)) for v in c)
                for c in sorted(C, key=lambda c: float(np.sum(c)))]
    out = []
    for c in px:
        if not any(float(np.linalg.norm(c - np.asarray(o))) <= tol for o in out):
            out.append(tuple(int(round(v)) for v in c))
    return sorted(out, key=sum)[:MAX_INKS] or [(0, 0, 0)]


def trace_any(path, **kw):
    """Raster or vector in, `(strokes_px, debug)` out.  Dispatch on suffix."""
    from pathlib import Path as _P
    if _P(path).suffix.lower() in (".svg", ".svgz"):
        return trace_svg(path, **{k: v for k, v in kw.items()
                                  if k in ("work_px", "flat_tol", "rdp_tol",
                                           "min_px", "n_inks", "palette")})
    return trace_art(path, **{k: v for k, v in kw.items()
                              if k not in ("flat_tol",)})
