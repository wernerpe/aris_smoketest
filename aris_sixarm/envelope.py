"""The neighbour's body column as what it actually is: a solid of REVOLUTION.

WHAT THE BANDS ARE.  `mounts.MOUNTS.body_bands` is the measured radial profile
of an FR3's base casting and shoulder — four (z0, z1, r) bands in BASE z,
running from the connector/cable stub 0.2325 m ABOVE the flange down to
0.3875 m below it, each grown by `calib` (30 mm of unsurveyed-base allowance).
Joint 1 turns about base z, so everything link0 and link1 carry sweeps into a
SOLID OF REVOLUTION about that axis: the profile is not an approximation of
the swept body, it IS the swept body, and a cylinder per band is exact for it.

WHAT WENT WRONG.  `mounts.arm_column_boxes` then hands each band to the box
machinery as `lo = min(a, b) - r`, `hi = max(a, b) + r` — the band's AXIS-
ALIGNED BOUNDING BOX, inflated by the band's radius in ALL THREE axes.  In x
and y that circumscribes the circle (a corner sits r*sqrt(2) from the axis
instead of r).  In Z IT IS SIMPLY WRONG-HEADED: a cylinder needs no z padding
at all, and the bottom band's box therefore hangs a further `r` = 160 mm below
where the arm's body actually ends.

That padding is what refused the pen-ups.  A neighbour's forearm passing UNDER
a base at z ~ 0.42-0.58 m is nowhere near any steel — it is under the 160 mm of
nothing that the AABB added.  Pete, looking at the :7004 scene: "ohh that is a
fake collision. the box is very over conservative!!!"

WHAT THIS MODULE DOES.  The same four bands, as FINITE CYLINDERS, with an
exact segment-to-cylinder distance in the same ternary-search style
`rig_final.segment_box_clearance` uses (the distance is convex along the
segment, so the search finds the true minimum).  It is strictly tighter than
the AABB and still a valid OUTER envelope of exactly what the bands always
enveloped — see `tests/test_mounts.py`, which samples the neighbour's joint
range and checks nothing escapes.

WHAT IT DOES NOT CLAIM.  The column has never enveloped links 2 and beyond,
and this does not start: those are handled where they always were — the
conductor's pair clearance while two arms move, and `allocate.ParkProbe` for a
parked partner.  This is the same obstacle, measured honestly.
"""
import numpy as np

from . import mounts, rig_final


def body_cylinders(spec, m=mounts.MOUNTS, h_inv=None, tag=None):
    """The pose-INVARIANT body column of ONE arm -> finite cylinders.

    Each entry is dict(name, p0, zc, z0, z1, r): the axis point (the base
    flange), the base z direction, the band's extent along it, and its radius.
    The same bands `mounts.arm_column_boxes` uses, minus the bounding box.
    """
    T = spec.T_world_base() if h_inv is None else spec.T_world_base(h_inv)
    p0 = np.asarray(T[:3, 3], float)
    zc = np.asarray(T[:3, 2], float)
    aid = spec.arm_id
    return [dict(name=f"body:{aid}_column{k}", p0=p0, zc=zc,
                 z0=float(z0), z1=float(z1), r=float(r),
                 source=(f"arm {aid} base column band {k}: r = {r:.3f} m over "
                         f"base z [{z0:+.4f}, {z1:+.4f}] — MEASURED body "
                         f"(mesh audit) inflated by calib {m.calib}, "
                         "pose-invariant, carried as the band's own CYLINDER"),
                 tag=f"body:{aid}" if tag is None else tag)
            for k, (z0, z1, r) in enumerate(m.column_bands)]


def fleet_body_cylinders(fleet, m=mounts.MOUNTS, h_inv=None):
    """{arm_id: [cylinders]} for a whole fleet."""
    return {aid: body_cylinders(s, m, h_inv) for aid, s in fleet.items()}


def _pack(cyls):
    """-> (p0 (M,3), zc (M,3), z0 (M,), z1 (M,), r (M,))."""
    return (np.stack([np.asarray(c["p0"], float) for c in cyls]),
            np.stack([np.asarray(c["zc"], float) for c in cyls]),
            np.array([c["z0"] for c in cyls], float),
            np.array([c["z1"] for c in cyls], float),
            np.array([c["r"] for c in cyls], float))


def point_cyl_d(P, p0, zc, z0, z1, r):
    """Distance from points to finite cylinders. (N,1,3) x (M,..) -> (N,M).

    Zero inside, exact outside: split the offset into its axial and radial
    parts, clamp each to the cylinder's extent, and take the hypotenuse of the
    two deficits — the standard exact distance to a convex capped cylinder.
    """
    w = P - p0
    t = np.einsum("...i,...i->...", w, zc)
    radial = w - t[..., None] * zc
    d = np.linalg.norm(radial, axis=-1)
    dz = np.maximum(np.maximum(z0 - t, t - z1), 0.0)
    dr = np.maximum(d - r, 0.0)
    return np.hypot(dz, dr)


def segment_cyl_clearance(A, B, cyls, iters=36):
    """(N,3),(N,3) segments -> (N,) exact min distance to the cylinder set.

    `rig_final.segment_box_clearance`'s argument, one shape over: d(t) is
    convex along the segment (the axial and radial deficits are each convex,
    and a norm of nonnegative convex components is convex), so a ternary
    search finds the true minimum, and a Lipschitz bound prunes the pairs that
    cannot come close.
    """
    A = np.asarray(A, float).reshape(-1, 3)
    B = np.asarray(B, float).reshape(-1, 3)
    if not cyls:
        return np.full(len(A), np.inf)
    p0, zc, z0, z1, r = _pack(cyls)
    dA = point_cyl_d(A[:, None], p0, zc, z0, z1, r)
    dB = point_cyl_d(B[:, None], p0, zc, z0, z1, r)
    L = np.linalg.norm(B - A, axis=1)
    ends = np.minimum(dA, dB)
    out = ends.copy()
    ni, mi = np.nonzero(ends - 0.5 * L[:, None] < out.min(axis=1)[:, None])
    if len(ni):
        a, d = A[ni], (B - A)[ni]
        t0 = np.zeros(len(ni))
        t1 = np.ones(len(ni))
        cp, cz, c0, c1, cr = p0[mi], zc[mi], z0[mi], z1[mi], r[mi]
        for _ in range(iters):
            m1 = t0 + (t1 - t0) / 3.0
            m2 = t1 - (t1 - t0) / 3.0
            f1 = point_cyl_d(a + m1[:, None] * d, cp, cz, c0, c1, cr)
            f2 = point_cyl_d(a + m2[:, None] * d, cp, cz, c0, c1, cr)
            take1 = f1 <= f2
            t1 = np.where(take1, m2, t1)
            t0 = np.where(take1, t0, m1)
        dmin = point_cyl_d(a + (0.5 * (t0 + t1))[:, None] * d, cp, cz, c0, c1, cr)
        np.minimum.at(out, (ni, mi), dmin)
    return out.min(axis=1)


def sphere_cyl_clearance(C, cyls, radii):
    """(N,S,3) sphere centres vs finite cylinders -> (N,).  Exact, no search."""
    C = np.asarray(C, float)
    if not cyls:
        return np.full(len(C), np.inf)
    p0, zc, z0, z1, r = _pack(cyls)
    d = point_cyl_d(C.reshape(-1, 1, 3), p0, zc, z0, z1, r)
    d = d.reshape(C.shape[0], C.shape[1], -1) - np.asarray(radii)[None, :, None]
    return d.reshape(len(C), -1).min(axis=1)


def chain_cyl_clearance(P, cyls, capsules=None, C=None, floor=None):
    """`rig_final.chain_static_clearance`, against cylinders. (N,C,3) -> (N,).

    `C` is the sphere centres when the sphere model is active — same
    substitution, same exactness, and the orange cylinders on the other side
    of the query are untouched (see `link_spheres`).
    """
    P = np.asarray(P, float)
    if P.ndim == 2:
        P = P[None]
    if capsules is None:
        capsules = (rig_final.STATIC_CAPSULES_LAT if P.shape[1] >= 11
                    else rig_final.STATIC_CAPSULES)
    if not cyls:
        return np.full(len(P), np.inf)
    worst = np.full(len(P), np.inf)
    for i, j, rad in capsules:
        worst = np.minimum(worst,
                           segment_cyl_clearance(P[:, i], P[:, j], cyls) - rad)
    if C is None:
        return worst
    from . import link_spheres                     # the intersection, see above
    sel = slice(None) if floor is None else np.flatnonzero(worst < float(floor))
    if floor is not None and not len(sel):
        return worst
    lean, _ = link_spheres.static_capsules(capsules)
    sph = sphere_cyl_clearance(np.asarray(C, float)[sel], cyls,
                               link_spheres.RADII)
    for i, j, rad in lean:
        sph = np.minimum(sph, segment_cyl_clearance(P[sel, i], P[sel, j],
                                                    cyls) - rad)
    out = worst.copy()
    out[sel] = np.maximum(worst[sel], sph)
    return out


# ==========================================================================
# INSTALLING THE MODEL
# ==========================================================================
# `paper.static_boxes` hands the router a flat list of obstacle dicts.  With
# the cylinder model installed, every `body:<aid>_column<k>` AABB in that list
# is swapped for the band's own cylinder — carrying `lo`/`hi` as well, so the
# broad phase (`paper.near_boxes`) and every other box-shaped consumer keep
# working unchanged and only the NARROW phase gets the honest geometry.

_BY_NAME = {}


def install(fleet, h_inv):
    """Model every arm's body column as cylinders instead of AABBs."""
    global _BY_NAME
    _BY_NAME = {}
    for aid, spec in fleet.items():
        for c in body_cylinders(spec, h_inv=h_inv):
            a = np.asarray(c["p0"]) + c["z0"] * np.asarray(c["zc"])
            b = np.asarray(c["p0"]) + c["z1"] * np.asarray(c["zc"])
            # the AABB is kept ONLY as a broad-phase proxy; it is never what
            # the clearance is measured against once `cyl` is set
            c["lo"] = np.minimum(a, b) - c["r"]
            c["hi"] = np.maximum(a, b) + c["r"]
            c["cyl"] = True
            _BY_NAME[c["name"]] = c


def uninstall():
    """Back to the shipped bounding boxes."""
    global _BY_NAME
    _BY_NAME = {}


def active():
    return bool(_BY_NAME)


def swap(boxes):
    """Replace every body-column AABB in `boxes` with its cylinder. -> list."""
    if not _BY_NAME or not boxes:
        return boxes
    return [_BY_NAME.get(b.get("name"), b) if isinstance(b, dict) else b
            for b in boxes]


def split(room):
    """A mixed obstacle list -> (boxes, cylinders)."""
    if not room:
        return room, ()
    cyl = [b for b in room if isinstance(b, dict) and b.get("cyl")]
    if not cyl:
        return room, ()
    return [b for b in room if not (isinstance(b, dict) and b.get("cyl"))], cyl
