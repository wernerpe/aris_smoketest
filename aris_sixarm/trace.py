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
    """
    S = [np.asarray(s, float) for s in strokes]
    alive = [True] * len(S)

    def tip(i, end):
        p = S[i]
        return p[-1] if end else p[0]

    def tdir(i, end):
        p = S[i]
        q = p[max(0, len(p) - DIR_PX):] if end else p[:DIR_PX]
        d = (q[-1] - q[0]) if end else (q[0] - q[-1])
        n = np.linalg.norm(d)
        return d / n if n > 1e-9 else np.zeros(2)

    changed = True
    while changed:
        changed = False
        best = None
        for i in range(len(S)):
            if not alive[i]:
                continue
            for j in range(len(S)):
                if j == i or not alive[j]:
                    continue
                for ei in (0, 1):
                    for ej in (0, 1):
                        pi, pj = tip(i, ei), tip(j, ej)
                        g = float(np.linalg.norm(pj - pi))
                        if g > gap or g < 1e-9:
                            continue
                        di, dj = tdir(i, ei), tdir(j, ej)
                        t = np.arccos(np.clip(-float(di @ dj), -1, 1))
                        link = (pj - pi) / g
                        t2 = np.arccos(np.clip(float(di @ link), -1, 1))
                        if t < turn and t2 < turn and (best is None or
                                                       (t + t2, g) < best[0]):
                            best = ((t + t2, g), i, ei, j, ej)
        if best is not None:
            _, i, ei, j, ej = best
            a = S[i] if ei else S[i][::-1]
            b = S[j][::-1] if ej else S[j]
            S[i] = np.vstack([a, b])
            alive[j] = False
            changed = True
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
             offset=(0.0, 0.0)):
    """Pixel strokes -> paper strokes in metres: one uniform scale, centred.

    Keeping the aspect ratio is non-negotiable (it is a logo), so the scale is
    whichever of the two sheet dimensions binds first; `target_width` is an
    upper bound, not a target, and the caller is told when it did not bind.

    `offset` (dx, dy) in metres moves the logo off the sheet centre — the knob
    the placement search turns, together with `target_width`.  It is applied as
    asked and NOT clamped; `info["fits"]` reports whether the placed logo is
    still inside the margin, which is the caller's cue to reject the candidate.
    """
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
                offset=(float(offset[0]), float(offset[1])),
                px_per_m=1.0 / scale, dropped_short=len(strokes_px) - len(out),
                fits=bool(cx - lw / 2 >= margin - 1e-9
                          and cx + lw / 2 <= sheet[0] - margin + 1e-9
                          and cy - lh / 2 >= margin - 1e-9
                          and cy + lh / 2 <= sheet[1] - margin + 1e-9))
    return out, info


def total_length(strokes):
    return float(sum(plen(s["pts"]) for s in strokes))
