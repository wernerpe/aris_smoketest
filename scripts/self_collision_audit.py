"""SELF-COLLISION GROUND TRUTH: an arm's own metal against its own metal.

`scripts/collision_audit.py` measured every capsule against the manufacturer's
meshes for arm-vs-arm and arm-vs-structure.  It never asked the third question,
because until 2026-08-26 nothing in the package asked it either: CAN AN ARM HIT
ITSELF?  `scripts/dead_disc_anatomy.py` says so in as many words — "SELF-
COLLISION is not modelled anywhere in the package ... a 15 deg leaned pose
folded under its own shoulder has not been checked".  This script is the
measurement the guard in `aris_sixarm/selfcoll.py` is built on.

WHY THE SHIPPED CAPSULE TABLE CANNOT BE THE SELF MODEL
------------------------------------------------------
The obvious move is to reuse `coordination.CAPSULES_LAT` — the audited,
mesh-validated table the conductor already trusts — against itself.  It does
not work, and the reason is not a detail:

  * THREE PAIRS ARE ALWAYS OVERLAPPING.  (1,3)-(4,5) is pinned at -178.5 mm by
    the 0.0825 m elbow offset against 0.130 + 0.131 of radius; (4,5)-(7,8) at
    -147.0 by the 0.088 m wrist offset; (7,8)-(10,9) at -44.0 by the 0.110 m
    bracket.  Their clearance is negative for EVERY configuration in the joint
    box (`--part invariance` measures it), so they cannot distinguish a safe
    pose from an unsafe one.
  * AND THE BASE COLUMN IS A BODY OF REVOLUTION ABOUT THE WRONG AXIS.  Band 3
    (base z 0.2590-0.3875, r 0.130) is link1's envelope swept about the base
    axis — the right model for a NEIGHBOUR, which sees link1 at every q1 and
    must clear all of them, and the wrong one for the arm itself, which sees
    link1 at ONE q1 and knows which side the casting bulges to.  Measured on
    the certified atlas, that band against the forearm capsule reads -22.5 mm
    at its worst while THE METAL IS 161 mm APART: the model is 158-180 mm of
    phantom shoulder.  A guard built on it would have refused 1 021 strict-GO
    cells of the shipped map for metal that is nowhere near.

So the self model is measured fresh, in each link's OWN frame, where a capsule
can be tight: `--part capsules` fits the smallest enclosing capsule of each
body's mesh (the manufacturer's collision meshes UNION the full-resolution
visual meshes, link0's cable included) by searching segment directions and
shrinking the segment.  The radii it finds are roughly HALF the inter-arm
table's, because the inter-arm radii are drawn about the kinematic chain's
segments and these are drawn about the metal's own principal axis.

Containment is checked against EVERY vertex of every mesh, not the fitting
sample, and the shipped radii are that exact maximum rounded UP to the
millimetre — the same convention `collision_audit` used for the inter-arm
inflation.

WHICH PAIRS THE GUARD WATCHES
-----------------------------
`--part touching` measures, over the joint box, the largest separation each
body pair ever reaches.  A pair that never separates is held together by the
kinematics itself — consecutive links meet at their joint, link1/link2 and
link5/link6 share an origin, the fingers ride the hand, the holder is gripped
by the fingers — and carries no information, exactly as an always-overlapping
capsule pair does.  What guards those is the FR3 joint limits, and that is
said out loud here rather than assumed.

The shipped rule is FOUR JOINTS OF SEPARATION, and it is chosen by
measurement, not by taste.  Against the certified map:

    chain distance >= 2   294 capsule pairs   refuses ALL 23 376 strict-GO
    chain distance >= 3   225 capsule pairs   refuses 17 807 of 23 376
    chain distance >= 4   165 capsule pairs   refuses NONE; tightest certified
                                              pose still holds 63.7 mm

— while `--part truth` puts the METAL at those same poses no closer than
114.3 mm.  Everything the three- and two-joint rules refuse is capsule fat
around a joint, and everything the four-joint rule watches is the fold: the
wrist, the hand and the pen coming back at the base, the shoulder and the
upper arm.  That is the self-collision an arm drawing on a table can actually
commit, and the one a leaning pen makes reachable.

Run (needs python-fcl, which the repo interpreter does not carry):

    python3 -m venv v && v/bin/pip install numpy scipy networkx trimesh python-fcl
    v/bin/python scripts/self_collision_audit.py --part all

    --part invariance   why the inter-arm table cannot be the self model
    --part capsules     fit the self capsule table from the meshes
    --part touching     which body pairs are held together by the kinematics
    --part truth        the model against the metal, pose by pose
    --part cost         what the finished guard costs the certified atlas
    --part near         what it does NOT watch, and whether the metal cares
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("ARIS_RIG", "proposed")
os.environ.setdefault("ARIS_TOOL", "lateral")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np                                                  # noqa: E402

import collision_audit as CA                                        # noqa: E402
from aris_sixarm import atlas, coordination                         # noqa: E402
from aris_sixarm.fleet import FLEET                                 # noqa: E402
from aris_sixarm.frames import (FR3_MAX, FR3_MIN, fk_many,          # noqa: E402
                                tool_points_many)

OUT = ROOT / "out"
CAPS = coordination.CAPSULES_LAT
NB = coordination.N_BASE
PEN_EXT, PEN_LAT = 0.110, 0.110
ATLAS_DIR = "atlas_proposed_h0940_gated"

# The BODIES of the self model: the rigid groups of an arm.  `hand` folds in
# the fingers and `tool` the whole holder, because each group is one rigid body
# on the hand frame and no joint separates its parts.
BODIES = [("link0", "link0", ["link0", "link0_cable"]),
          ("link1", "link1", ["link1"]),
          ("link2", "link2", ["link2"]),
          ("link3", "link3", ["link3"]),
          ("link4", "link4", ["link4"]),
          ("link5", "link5", ["link5"]),
          ("link6", "link6", ["link6"]),
          ("link7", "link7", ["link7"]),
          ("hand", "hand", ["hand", "leftfinger", "rightfinger"]),
          ("tool", "hand", ["housing", "cap", "lead"])]
BODY_NAMES = [b[0] for b in BODIES]

# The z-band edges link0 is measured in (base frame, metres).  Chosen on the
# metal: the cable stub runs to -0.2307, the plate face sits at 0, the casting
# steps in at 0.070 and 0.100, and the collision mesh stops at 0.1400 — above
# which the arm's own metal is link1's and belongs to link1's capsules.
LINK0_Z = (-0.2380, -0.0750, 0.0000, 0.0350, 0.0700, 0.1000, 0.1200, 0.1440)


def cap_label(k):
    c = CAPS[k]
    s = f"({c[0]},{c[1]})r{c[2]:.3f}"
    if len(c) > 3:
        s += f"[{c[3]:.3f},{c[4]:.3f}]"
    return s


def chain11(qs):
    T, P = fk_many(np.atleast_2d(qs))
    tool = tool_points_many(T, PEN_EXT, PEN_LAT)
    return np.concatenate([P] + [t[:, None, :] for t in tool], axis=1)


def _pt_seg(p, a, b):
    ab = b - a
    den = np.sum(ab * ab, -1)
    t = np.where(den > 1e-15,
                 np.sum((p - a) * ab, -1) / np.where(den > 1e-15, den, 1.0), 0.0)
    t = np.clip(t, 0.0, 1.0)
    d = p - (a + t[..., None] * ab)
    return np.sqrt(np.sum(d * d, -1))


def seg_seg(p0, p1, q0, q1):
    d1, d2, r = p1 - p0, q1 - q0, p0 - q0
    a = np.sum(d1 * d1, -1)
    e = np.sum(d2 * d2, -1)
    b = np.sum(d1 * d2, -1)
    c = np.sum(d1 * r, -1)
    f = np.sum(d2 * r, -1)
    den = a * e - b * b
    ins = den > 1e-12
    s = np.where(ins, (b * f - c * e) / np.where(ins, den, 1.0), -1.0)
    t = np.where(ins, (a * f - b * c) / np.where(ins, den, 1.0), -1.0)
    good = ins & (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)
    w = r + s[..., None] * d1 - t[..., None] * d2
    interior = np.where(good, np.sqrt(np.maximum(np.sum(w * w, -1), 0.0)), np.inf)
    edge = np.minimum(np.minimum(_pt_seg(p0, q0, q1), _pt_seg(p1, q0, q1)),
                      np.minimum(_pt_seg(q0, p0, p1), _pt_seg(q1, p0, p1)))
    return np.minimum(interior, edge)


# ---------------------------------------------------------------------------
# pose sets
# ---------------------------------------------------------------------------
def atlas_poses(atlas_dir=ATLAS_DIR, strict=True):
    Q = []
    for aid in FLEET:
        arr, meta = atlas.load(OUT / atlas_dir, aid)
        ok, why = atlas.is_current(meta)
        if not ok:
            print(f"  [warn] atlas {atlas_dir} arm {aid}: {why}")
        if strict:
            arr = arr[atlas.strict_go(arr)]
        Q.append(arr[:, atlas.QCOL:atlas.QCOL + 7])
    return np.vstack(Q)


def random_poses(n, seed=7, margin=0.0):
    """Uniform in the joint box.  DELIBERATELY NO PAPER FILTER: the poses that
    fold an arm under its own shoulder are the ones this audit exists to
    sample, and `collision_audit.load_pose_sets` filters exactly those out."""
    rng = np.random.default_rng(seed)
    keep = []
    while len(keep) < n:
        Q = FR3_MIN + rng.random((2 * n + 64, 7)) * (FR3_MAX - FR3_MIN)
        if margin > 0:
            m = np.min(np.minimum(Q - FR3_MIN, FR3_MAX - Q), axis=1)
            Q = Q[m >= margin]
        keep.extend(Q)
    return np.array(keep[:n])


# ---------------------------------------------------------------------------
# mesh plumbing
# ---------------------------------------------------------------------------
def _mesh_groups():
    """-> {body: (frame_index, [(name, trimesh), ...])} for the 10 bodies."""
    parts = CA.arm_parts("union")
    tools = CA.tool_parts(PEN_EXT, PEN_LAT)
    idx = {f: i for i, f in enumerate(CA.FRAMES)}
    got = {}
    for name, (frame, mesh) in parts.items():
        got.setdefault(CA._part_key(name), []).append((name, mesh))
    for name, mesh in tools.items():
        got.setdefault(name, []).append((f"tool:{name}", mesh))
    out = {}
    for body, frame, keys in BODIES:
        ms = []
        for k in keys:
            ms.extend(got.get(k, []))
        out[body] = (idx[frame], ms)
    return out


def _bodies_fcl(groups):
    names, BA, BB, fidx = [], {}, {}, {}
    for body, (fi, ms) in groups.items():
        for nm, mesh in ms:
            m = mesh.convex_hull if len(mesh.vertices) > 20000 else mesh
            names.append((body, nm))
            BA[nm] = CA.Body(nm, m)
            BB[nm] = CA.Body(nm, m)
            fidx[nm] = fi
    return names, BA, BB, fidx


# ---------------------------------------------------------------------------
# part 1 — why the inter-arm table cannot be the self model
# ---------------------------------------------------------------------------
def part_invariance(rec, n=400000, seed=11):
    pairs = []
    for i in range(len(CAPS)):
        for j in range(i + 1, len(CAPS)):
            if i < NB and j < NB:
                continue
            if {CAPS[i][0], CAPS[i][1]} & {CAPS[j][0], CAPS[j][1]}:
                continue
            pairs.append((i, j))
    rng = np.random.default_rng(seed)
    lo = np.full(len(pairs), np.inf)
    hi = np.full(len(pairs), -np.inf)
    rr = np.array([c[2] for c in CAPS], float)
    t0 = np.array([c[3] if len(c) > 3 else 0.0 for c in CAPS])
    t1 = np.array([c[4] if len(c) > 4 else 1.0 for c in CAPS])
    for _ in range(0, n, 20000):
        Q = FR3_MIN + rng.random((20000, 7)) * (FR3_MAX - FR3_MIN)
        P = chain11(Q)
        A0 = P[:, [c[0] for c in CAPS], :]
        D = P[:, [c[1] for c in CAPS], :] - A0
        A, B = A0 + t0[:, None] * D, A0 + t1[:, None] * D
        for k, (i, j) in enumerate(pairs):
            d = seg_seg(A[:, i], B[:, i], A[:, j], B[:, j]) - rr[i] - rr[j]
            lo[k] = min(lo[k], d.min())
            hi[k] = max(hi[k], d.max())
    rows = [dict(i=i, j=j, label=f"{cap_label(i)} <-> {cap_label(j)}",
                 min_mm=float(lo[k] * 1e3), max_mm=float(hi[k] * 1e3),
                 always_overlapping=bool(hi[k] < 0))
            for k, (i, j) in enumerate(pairs)]
    print(f"\nINTER-ARM CAPSULE PAIRS THAT CAN NEVER DISCRIMINATE "
          f"({n} random configurations):")
    for r in rows:
        if r["always_overlapping"]:
            print(f"   {r['label']:48s} min {r['min_mm']:8.1f}  "
                  f"max {r['max_mm']:8.1f} mm")
    n_dead = sum(r["always_overlapping"] for r in rows)
    print(f"   -> {n_dead} of {len(rows)} non-adjacent pairs are always "
          f"overlapping")
    rec["invariance"] = dict(n=n, rows=rows)
    return rows


# ---------------------------------------------------------------------------
# part 2 — fit the self capsule table
# ---------------------------------------------------------------------------
def fit_capsule(V, n_dir=400, seed=0):
    """Smallest enclosing capsule of a point cloud -> (a, b, r).

    Search over segment DIRECTIONS (the three principal axes plus a random
    sphere), and for each one shrink the segment inwards while the enclosing
    radius falls — a point past an end is covered by that end's hemispherical
    cap, so the tightest capsule is generally SHORTER than the projection
    extent.  `r` is the exact max distance from any point to the segment, so
    the capsule contains the cloud by construction.
    """
    V = np.asarray(V, float)
    c = V.mean(0)
    X = V - c
    vt = np.linalg.svd(X, full_matrices=False)[2]
    rng = np.random.default_rng(seed)
    dirs = list(vt) + [d / np.linalg.norm(d)
                       for d in rng.normal(size=(n_dir, 3))]
    best = None
    for d in dirs:
        d = d / np.linalg.norm(d)
        t = X @ d
        perp = np.linalg.norm(X - np.outer(t, d), axis=1)
        t0, t1 = float(t.min()), float(t.max())

        def rad(a, b, t=t, perp=perp):
            dd = np.where(t < a, np.hypot(a - t, perp),
                          np.where(t > b, np.hypot(t - b, perp), perp))
            return float(dd.max())

        r = rad(t0, t1)
        for step in (0.04, 0.01, 0.0025, 0.0005):
            while True:
                cand = [(rad(t0 + step, t1), t0 + step, t1),
                        (rad(t0, t1 - step), t0, t1 - step)]
                cand = [c2 for c2 in cand if c2[2] > c2[1]]
                if not cand:
                    break
                r2, a2, b2 = min(cand)
                if r2 < r - 1e-9:
                    r, t0, t1 = r2, a2, b2
                else:
                    break
        if best is None or r < best[0]:
            best = (r, d, t0, t1)
    r, d, t0, t1 = best
    return c + t0 * d, c + t1 * d, r


def _sub(V, n, seed=0):
    if len(V) <= n:
        return V
    return V[np.random.default_rng(seed).choice(len(V), n, replace=False)]


def part_capsules(rec, bands=3, seed=0, n_dir=400):
    """Fit `bands` capsules per body, along the body's own principal axis.

    ONE CAPSULE PER LINK IS NOT ENOUGH, and that is measured rather than
    assumed.  A single fitted capsule per body is already half the radius of
    the inter-arm table (link4 0.185 -> 0.086), and it STILL reads negative on
    every certified pose for link0-link2, link2-link4, link3-link5 and
    link5-hand, and on 3 230 of them for link1-link3 — because a sausage drawn
    round an L-shaped casting contains a great deal of air near the joints,
    which is exactly where two links two apart pass each other.  Splitting each
    body into three bands along its own fitted axis and re-fitting each band
    takes the radii to 0.05-0.08 and, more to the point, puts the metal where
    the metal is.  Every vertex lands in exactly one band, so the union still
    contains the body by construction.
    """
    groups = _mesh_groups()
    rows = []
    print(f"\n{'body':7s} {'band':>4s} {'frame':6s} {'verts':>8s} {'r_fit':>8s} "
          f"{'r_ship':>8s} {'len':>7s}  a -> b (link frame, m)")
    for body, (fi, ms) in groups.items():
        V = np.vstack([np.asarray(m.vertices, float) for _, m in ms])
        if body == "link0":
            # LINK0 IS THE ONE BODY THAT NEVER MOVES, so its natural model is
            # the one the column audit already uses for it: a stack of z-bands
            # about the base axis, each at the largest radius the metal reaches
            # in that slice.  A principal-axis fit draws chords ACROSS the base
            # disc instead, and a chord through a disc is mostly air: on arm
            # 71's certified park pose the chord fit reads -120.4 mm against
            # the wrist where the metal is 26.9 mm, and this stack reads -6.3.
            for k, (z0, z1) in enumerate(zip(LINK0_Z[:-1], LINK0_Z[1:])):
                sel = (V[:, 2] >= z0 - 1e-9) & (V[:, 2] <= z1 + 1e-9)
                if sel.sum() < 4:
                    continue
                a = np.array([0.0, 0.0, z0])
                b = np.array([0.0, 0.0, z1])
                r_all = float(_pt_seg(V[sel], a[None], b[None]).max())
                r_ship = float(np.ceil(r_all * 1000.0) / 1000.0)
                rows.append(dict(body=body, band=k, frame=CA.FRAMES[fi],
                                 frame_index=int(fi),
                                 a=[float(v) for v in a],
                                 b=[float(v) for v in b],
                                 r_fit=r_all, r=r_ship,
                                 n_vertices=int(sel.sum())))
                print(f"{body:7s} {k:4d} {CA.FRAMES[fi]:6s} {int(sel.sum()):8d} "
                      f"{r_all:8.4f} {r_ship:8.3f} {z1-z0:7.4f}  "
                      f"z-band [{z0:6.3f}, {z1:6.3f}] about the base axis")
            continue
        a0, b0, _ = fit_capsule(_sub(V, 60000, seed), n_dir=n_dir, seed=seed)
        d = (b0 - a0) / np.linalg.norm(b0 - a0)
        t = (V - a0) @ d
        edges = np.linspace(t.min(), t.max(), bands + 1)
        edges[0] -= 1.0
        edges[-1] += 1.0
        for k in range(bands):
            sel = (t >= edges[k]) & (t < edges[k + 1])
            if sel.sum() < 16:
                continue
            W = V[sel]
            a, b, _ = fit_capsule(_sub(W, 60000, seed), n_dir=n_dir, seed=seed)
            # CONTAINMENT AGAINST EVERY VERTEX OF THE BAND, not the sample.
            r_all = float(_pt_seg(W, a[None], b[None]).max())
            r_ship = float(np.ceil(r_all * 1000.0) / 1000.0)
            rows.append(dict(body=body, band=k, frame=CA.FRAMES[fi],
                             frame_index=int(fi),
                             a=[float(v) for v in a], b=[float(v) for v in b],
                             r_fit=r_all, r=r_ship, n_vertices=int(sel.sum())))
            print(f"{body:7s} {k:4d} {CA.FRAMES[fi]:6s} {int(sel.sum()):8d} "
                  f"{r_all:8.4f} {r_ship:8.3f} {np.linalg.norm(b-a):7.4f}  "
                  f"[{a[0]:6.3f},{a[1]:6.3f},{a[2]:6.3f}] -> "
                  f"[{b[0]:6.3f},{b[1]:6.3f},{b[2]:6.3f}]")
    rec["capsules"] = dict(bands=bands, rows=rows)
    print("\n--- paste into aris_sixarm/selfcoll.py -------------------------")
    for r in rows:
        print(f'    ("{r["body"]}", {r["band"]}, {r["frame_index"]}, '
              f'({r["a"][0]:.4f}, {r["a"][1]:.4f}, {r["a"][2]:.4f}), '
              f'({r["b"][0]:.4f}, {r["b"][1]:.4f}, {r["b"][2]:.4f}), '
              f'{r["r"]:.3f}),   # mesh {r["r_fit"]:.4f}')
    return rows


# ---------------------------------------------------------------------------
# part 3 — which body pairs are held together by the kinematics
# ---------------------------------------------------------------------------
def part_touching(rec, n=600, seed=13, thresh=0.05):
    import fcl                                                   # noqa: F401
    groups = _mesh_groups()
    names, BA, BB, fidx = _bodies_fcl(groups)
    by_body = {}
    for body, nm in names:
        by_body.setdefault(body, []).append(nm)
    Q = random_poses(n, seed=seed)
    hi = {}
    for q in Q:
        L = CA.link_transforms(q)
        for _, nm in names:
            BA[nm].place(L[fidx[nm]])
            BB[nm].place(L[fidx[nm]])
        for x in range(len(BODY_NAMES)):
            for y in range(x + 1, len(BODY_NAMES)):
                bx, by = BODY_NAMES[x], BODY_NAMES[y]
                d = min(max(0.0, CA._dist(BA[na], BB[nb]))
                        for na in by_body[bx] for nb in by_body[by])
                k = f"{bx}|{by}"
                hi[k] = max(hi.get(k, 0.0), d)
    held = sorted(k for k, v in hi.items() if v < thresh)
    print(f"\nBODY PAIRS THAT NEVER SEPARATE BY {thresh*1e3:.0f} mm over "
          f"{n} configurations (held by the kinematics, excluded):")
    for k in held:
        print(f"   {k:22s} max {hi[k]*1e3:7.1f} mm")
    print(f"\nthe pairs the guard watches ({len(hi)-len(held)} of {len(hi)}):")
    for k, v in sorted(hi.items(), key=lambda kv: kv[1]):
        if k not in held:
            print(f"   {k:22s} max {v*1e3:7.1f} mm")
    rec["touching"] = dict(n=n, thresh=thresh, held=held,
                           max_mm={k: float(v * 1e3)
                                   for k, v in sorted(hi.items())})
    return held


# ---------------------------------------------------------------------------
# part 4 — the model against the metal
# ---------------------------------------------------------------------------
def part_truth(rec, n_atlas=1200, n_random=1200, seed=5, cap=0.5):
    import fcl                                                   # noqa: F401
    from aris_sixarm import selfcoll
    groups = _mesh_groups()
    names, BA, BB, fidx = _bodies_fcl(groups)
    by_body = {}
    for body, nm in names:
        by_body.setdefault(body, []).append(nm)
    watch = sorted({tuple(sorted((selfcoll.BODY_OF[i], selfcoll.BODY_OF[j])))
                    for i, j in selfcoll.SELF_PAIRS})
    print(f"\n{len(selfcoll.SELF_PAIRS)} watched capsule pairs over "
          f"{len(watch)} body pairs")

    rng = np.random.default_rng(seed)
    Qa = atlas_poses()
    Qa = Qa[rng.choice(len(Qa), min(n_atlas, len(Qa)), replace=False)]
    Qr = random_poses(n_random, seed=seed + 1)
    Q = np.vstack([Qa, Qr])
    src = ["atlas"] * len(Qa) + ["random"] * len(Qr)
    Dc = selfcoll.self_clearance(Q)
    t0 = time.time()
    dm = np.full(len(Q), cap)
    which = [None] * len(Q)
    for n, q in enumerate(Q):
        L = CA.link_transforms(q)
        for _, nm in names:
            BA[nm].place(L[fidx[nm]])
            BB[nm].place(L[fidx[nm]])
        best, bw = cap, None
        for (x, y) in watch:
            for na in by_body[x]:
                for nb in by_body[y]:
                    d = max(0.0, CA._dist(BA[na], BB[nb]))
                    if d < best:
                        best, bw = d, f"{na}|{nb}"
        dm[n], which[n] = best, bw
        if n and n % 400 == 0:
            print(f"  {n}/{len(Q)}, {time.time()-t0:.0f}s")
    opt = Dc > dm + 1e-9
    a = np.array([s == "atlas" for s in src])
    print(f"\nMODEL vs METAL over {len(Q)} poses ({time.time()-t0:.0f}s)")
    for lab, m in (("certified atlas ", a), ("random joint box", ~a)):
        print(f"  {lab}: model min {Dc[m].min()*1e3:7.1f} mm, "
              f"metal min {dm[m].min()*1e3:7.1f} mm, model <= metal on "
              f"{int((Dc[m] <= dm[m] + 1e-9).sum())}/{int(m.sum())}")
    print(f"  OPTIMISTIC (model > metal): {int(opt.sum())} poses "
          f"({opt.sum()/len(Q)*100:.2f} %)")
    if opt.any():
        k = int(np.argmax(Dc - dm))
        print(f"    worst: {src[k]} pose, model {Dc[k]*1e3:.1f} mm, "
              f"metal {dm[k]*1e3:.1f} mm via {which[k]}")
    rec["truth"] = dict(n_poses=int(len(Q)), seconds=float(time.time() - t0),
                        n_optimistic=int(opt.sum()),
                        model_min_mm=float(Dc.min() * 1e3),
                        metal_min_mm=float(dm.min() * 1e3),
                        atlas_model_min_mm=float(Dc[a].min() * 1e3),
                        atlas_metal_min_mm=float(dm[a].min() * 1e3),
                        rows=[dict(src=src[i], model_mm=float(Dc[i] * 1e3),
                                   metal_mm=float(dm[i] * 1e3), parts=which[i])
                              for i in range(len(Q))])


# ---------------------------------------------------------------------------
# part 5 — what the guard costs the certified atlas
# ---------------------------------------------------------------------------
def part_cost(rec, atlas_dir=ATLAS_DIR):
    from aris_sixarm import selfcoll
    print(f"\nself-collision guard vs out/{atlas_dir}")
    print(f"  {selfcoll.N_CAP} capsules, {len(selfcoll.SELF_PAIRS)} "
          f"watched pairs, margin {selfcoll.SELF_MARGIN*1e3:.0f} mm")
    tot = go = tot_bad = go_bad = 0
    per, worst = {}, np.inf
    for aid in FLEET:
        arr, _ = atlas.load(OUT / atlas_dir, aid)
        Q = arr[:, atlas.QCOL:atlas.QCOL + 7]
        c = selfcoll.self_clearance(Q)
        bad = c < selfcoll.SELF_MARGIN
        g = atlas.strict_go(arr)
        per[str(aid)] = dict(rows=int(len(arr)), strict_go=int(g.sum()),
                             rows_fail=int(bad.sum()),
                             strict_go_fail=int((bad & g).sum()),
                             min_mm=float(c.min() * 1e3),
                             min_go_mm=float(c[g].min() * 1e3) if g.any() else None)
        tot += len(arr); go += int(g.sum())
        tot_bad += int(bad.sum()); go_bad += int((bad & g).sum())
        worst = min(worst, float(c[g].min()) if g.any() else np.inf)
        print(f"  arm {aid:>2}: {len(arr):5d} rows / {int(g.sum()):5d} strict-GO"
              f"   refused {int(bad.sum()):4d} / {int((bad & g).sum()):4d}"
              f"   min {c.min()*1e3:7.1f} mm  (strict-GO {c[g].min()*1e3:7.1f})")
    print(f"  TOTAL {tot} rows, {go} strict-GO: guard refuses {tot_bad} rows "
          f"({tot_bad/max(tot,1)*100:.3f} %), {go_bad} strict-GO cells "
          f"({go_bad/max(go,1)*100:.3f} %); tightest certified pose "
          f"{worst*1e3:.1f} mm")
    rec["cost"] = dict(atlas=atlas_dir, per_arm=per, rows=tot, strict_go=go,
                       rows_fail=tot_bad, strict_go_fail=go_bad,
                       tightest_certified_mm=float(worst * 1e3))


# ---------------------------------------------------------------------------
# part 6 — what the guard does NOT watch, and whether that matters
# ---------------------------------------------------------------------------
def part_near(rec, n=12000, seed=4):
    """Closest the METAL ever gets, per body pair, by chain distance.

    The number that decides `selfcoll.WATCH_CHAIN_D` and the number that has to
    be published next to it: a pair the guard does not watch is only defensible
    if the mechanism keeps it apart, and this says which ones it does not.
    """
    import fcl                                                   # noqa: F401
    groups = _mesh_groups()
    names, BA, BB, fidx = _bodies_fcl(groups)
    by_body = {}
    for body, nm in names:
        by_body.setdefault(body, []).append(nm)
    pos = dict(link0=0, link1=1, link2=2, link3=3, link4=4, link5=5, link6=6,
               link7=7, hand=8, tool=8)
    pairs = [(BODY_NAMES[i], BODY_NAMES[j])
             for i in range(len(BODY_NAMES)) for j in range(i + 1, len(BODY_NAMES))]
    Q = random_poses(n, seed=seed)
    lo = {k: 1e9 for k in pairs}
    t0 = time.time()
    for k, q in enumerate(Q):
        L = CA.link_transforms(q)
        for _, nm in names:
            BA[nm].place(L[fidx[nm]])
            BB[nm].place(L[fidx[nm]])
        for (x, y) in pairs:
            d = min(max(0.0, CA._dist(BA[na], BB[nb]))
                    for na in by_body[x] for nb in by_body[y])
            lo[(x, y)] = min(lo[(x, y)], d)
        if k and k % 1500 == 0:
            print(f"  {k}/{n}  {time.time()-t0:.0f}s", flush=True)
    from aris_sixarm import selfcoll
    watched = {tuple(sorted((selfcoll.BODY_OF[i], selfcoll.BODY_OF[j])))
               for i, j in selfcoll.SELF_PAIRS}
    print(f"\nCLOSEST THE METAL EVER GETS ({n} configurations, "
          f"{time.time()-t0:.0f}s)")
    print(f"{'pair':22s} {'chain d':>8s} {'watched':>8s} {'min mm':>9s}")
    rows = []
    for (x, y) in sorted(pairs, key=lambda k: (abs(pos[k[1]] - pos[k[0]]), lo[k])):
        w = tuple(sorted((x, y))) in watched
        rows.append(dict(a=x, b=y, chain_d=int(abs(pos[y] - pos[x])),
                         watched=bool(w), min_mm=float(lo[(x, y)] * 1e3)))
        flag = "  <== CONTACT, NOT WATCHED" if (lo[(x, y)] <= 1e-9 and not w) \
            else ("  <== contact, caught" if lo[(x, y)] <= 1e-9 else "")
        print(f"{x+'|'+y:22s} {abs(pos[y]-pos[x]):8d} {str(w):>8s} "
              f"{lo[(x,y)]*1e3:9.2f}{flag}")
    rec["near"] = dict(n=n, rows=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all",
                    choices=["all", "invariance", "capsules", "touching",
                             "truth", "cost", "near"])
    ap.add_argument("--poses", type=int, default=1200)
    ap.add_argument("--out", default=str(OUT / "self_collision_audit.json"))
    a = ap.parse_args()
    p = Path(a.out)
    rec = json.loads(p.read_text()) if p.exists() else {}
    rec["meta"] = dict(when=time.strftime("%Y-%m-%d %H:%M"), pen_ext=PEN_EXT,
                       pen_lat=PEN_LAT, bodies=BODY_NAMES)

    def save():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rec, indent=1))
        print(f"[saved] {p}")

    if a.part in ("all", "invariance"):
        part_invariance(rec); save()
    if a.part in ("all", "capsules"):
        part_capsules(rec); save()
    if a.part in ("all", "touching"):
        part_touching(rec); save()
    if a.part in ("all", "truth"):
        part_truth(rec, n_atlas=a.poses, n_random=a.poses); save()
    if a.part in ("all", "cost"):
        part_cost(rec); save()
    if a.part in ("all", "near"):
        part_near(rec); save()


if __name__ == "__main__":
    main()
