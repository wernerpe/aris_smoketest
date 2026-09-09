"""WHERE `aris_sixarm/link_spheres.py`'s 64 NUMBERS CAME FROM.

Run:
    python scripts/link_sphere_fit.py --part validate   # the shipped sets
    python scripts/link_sphere_fit.py --part fit        # re-fit and print
    python scripts/link_sphere_fit.py --part accuracy   # volume / offset

REQUIREMENTS beyond the repo's own: `trimesh` (the repo venv carries it).  The
meshes are loaded through `scripts/collision_audit.py`, which is where this
package's mesh plumbing already lives — the manufacturer's collision shells
UNION the full-resolution visual meshes, link0's cable included, exactly the
ground truth `scripts/self_collision_audit.py --part capsules` fits against.
`assets/system_model/meshes/fr3/*.gltf` is byte-identical geometry to
`assets/franka_description/meshes/visual/*.gltf` (checked), so which of the two
is read does not change a number.

--part validate answers the question Pete's instruction actually raises: can a
shipped sphere set be scavenged?  It measures ESCAPE — the largest distance any
mesh point of a body lies OUTSIDE the spheres attached to that body — for the
three sphere models on this machine.  All three are optimistic on every body.
See the `link_spheres` docstring for the table.

--part fit produces the shipped table: eight spheres a moving body, seeded by
farthest-point, descended by Lloyd, polished by Badoiu-Clarkson MEB, and then
each radius set to the EXACT maximum distance from its centre to any point
assigned to it and rounded UP to the millimetre.  Containment is by
construction and `tests/test_link_spheres.py` re-measures it.
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import trimesh                                                   # noqa: E402
import collision_audit as CA                                     # noqa: E402

trimesh.util.log.setLevel(50)

MOVING = ("link1", "link2", "link3", "link4", "link5", "link6", "link7", "hand")
FRAME_OF = dict(link1=1, link2=2, link3=3, link4=4, link5=5, link6=6,
                link7=7, hand=9)          # frames.LINK_FRAMES indices
K = 8                                     # spheres per moving body
HOST = {f"link{i}": f"link{i}" for i in range(8)}
HOST.update(hand="hand", leftfinger="hand", rightfinger="hand", link8="link7")

SHIPPED = (
    ("franka66 (Drake stock Panda)",
     ROOT / "assets/franka_description/urdf/panda_arm_hand.urdf", "panda_"),
    ("vamp59",
     Path("/home/franka/git/vamp/resources/panda/panda_spherized.urdf"), "panda_"),
    ("mmt35 (genuinely FR3)",
     Path("/home/franka/git/mmt_gcs/assets/"
          "fr3_franka_hand_sphere_collisions.urdf"), "fr3_"),
)


def cloud(seed=0):
    """{body: (N,3) points in the link frame} — vertices AND surface samples.

    Vertices alone are not enough: a sphere containing three vertices contains
    their triangle (a ball is convex), but only if all three are charged to the
    SAME sphere, and nothing here guarantees that.  Sampling the surface closes
    it empirically — 8 samples a vertex, ~1.3 M points over the eight bodies.
    """
    got = defaultdict(list)
    for name, (_frame, mesh) in CA.arm_parts("union").items():
        got[CA._part_key(name)].append(mesh)
    rng = np.random.default_rng(seed)
    out = defaultdict(list)
    for key, ms in got.items():
        host = HOST.get(key)
        if host is None:
            continue
        for m in ms:
            out[host].append(np.asarray(m.vertices, float))
            n = min(200000, max(4000, 8 * len(m.vertices)))
            s, _ = trimesh.sample.sample_surface(
                m, n, seed=int(rng.integers(1 << 30)))
            out[host].append(np.asarray(s, float))
    return {k: np.concatenate(v) for k, v in out.items()}


def escape(P, C, R, chunk=20000):
    """Largest distance any point of P lies OUTSIDE the spheres. -> m.

    <= 0 means the union contains the cloud.  This is the number that decides
    whether a sphere set may stand in a safety gate at all.
    """
    worst = -np.inf
    for s in range(0, len(P), chunk):
        d = np.linalg.norm(P[s:s + chunk, None, :] - C[None], axis=2) - R[None]
        worst = max(worst, float(d.min(1).max()))
    return worst


def radii_for(P, C, chunk=20000):
    """Exact radii so every point of P lies inside its NEAREST sphere. -> (k,)"""
    R = np.zeros(len(C))
    for s in range(0, len(P), chunk):
        d = np.linalg.norm(P[s:s + chunk, None, :] - C[None], axis=2)
        lab = d.argmin(1)
        for i in range(len(C)):
            m = lab == i
            if m.any():
                R[i] = max(R[i], float(d[m, i].max()))
    return R


def meb(P, iters=120):
    """Badoiu-Clarkson approximate minimum-enclosing-ball centre."""
    c = P.mean(0)
    for i in range(1, iters + 1):
        c = c + (P[int(np.argmax(((P - c) ** 2).sum(1)))] - c) / (i + 1)
    return c


def fit(P, k=K, seed=1, rounds=25):
    """k centres covering P: farthest-point seed, Lloyd, then an MEB polish."""
    rng = np.random.default_rng(seed)
    C = [P[rng.integers(len(P))]]
    d = np.linalg.norm(P - C[0], axis=1)
    while len(C) < k:
        C.append(P[int(np.argmax(d))])
        d = np.minimum(d, np.linalg.norm(P - C[-1], axis=1))
    C = np.stack(C)
    for _ in range(rounds):
        lab = np.argmin(((P[:, None, :] - C[None]) ** 2).sum(2), axis=1)
        S, n = np.zeros_like(C), np.zeros(k)
        np.add.at(S, lab, P)
        np.add.at(n, lab, 1.0)
        good = n > 0
        new = C.copy()
        new[good] = S[good] / n[good, None]
        if np.max(np.linalg.norm(new - C, axis=1)) < 1e-5:
            C = new
            break
        C = new
    lab = np.argmin(((P[:, None, :] - C[None]) ** 2).sum(2), axis=1)
    for i in range(k):
        if (lab == i).sum():
            C[i] = meb(P[lab == i])
    return C


def _rpy(r, p, y):
    import math
    Rz = np.array([[math.cos(y), -math.sin(y), 0],
                   [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    Ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0],
                   [-math.sin(p), 0, math.cos(p)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)],
                   [0, math.sin(r), math.cos(r)]])
    return Rz @ Ry @ Rx


def urdf_spheres(path, pref):
    """{body: [(centre, radius)]} with link8/fingers folded into their host."""
    import xml.etree.ElementTree as ET
    root = ET.parse(path).getroot()
    per, chain = defaultdict(list), {}
    for link in root.iter('link'):
        nm = link.get('name').replace(pref, '')
        for col in link.findall('collision'):
            g = col.find('geometry')
            sp = None if g is None else g.find('sphere')
            if sp is None:
                continue
            o = col.find('origin')
            xyz = (o.get('xyz') if o is not None else None) or '0 0 0'
            per[nm].append((np.array([float(v) for v in xyz.split()]),
                            float(sp.get('radius'))))
    for j in root.iter('joint'):
        if j.find('parent') is None or j.find('child') is None:
            continue
        o = j.find('origin')
        T = np.eye(4)
        if o is not None:
            T[:3, 3] = [float(v) for v in (o.get('xyz') or '0 0 0').split()]
            T[:3, :3] = _rpy(*[float(v) for v in
                               (o.get('rpy') or '0 0 0').split()])
        chain[j.find('child').get('link').replace(pref, '')] = (
            j.find('parent').get('link').replace(pref, ''), T)
    out = defaultdict(list)
    for nm, sl in per.items():
        host = HOST.get(nm)
        if host is None:
            continue
        T, cur = np.eye(4), nm
        while cur != host and cur in chain:
            p, Tj = chain[cur]
            T, cur = Tj @ T, p
        for c, r in sl:
            out[host].append((T[:3, :3] @ c + T[:3, 3], r))
    return out


def part_validate(CL):
    print("ESCAPE of the shipped sphere sets (positive = the metal is OUTSIDE)")
    for label, path, pref in SHIPPED:
        if not Path(path).exists():
            print(f"  {label}: NOT PRESENT at {path}")
            continue
        S = urdf_spheres(path, pref)
        n = sum(len(v) for v in S.values())
        print(f"  {label}  ({n} spheres, folded)")
        for b in ("link0",) + MOVING:
            sl = S.get(b, [])
            if not sl:
                print(f"      {b:7s}  n=  0   NO SPHERES")
                continue
            C = np.stack([c for c, _ in sl])
            R = np.array([r for _, r in sl])
            print(f"      {b:7s}  n={len(sl):3d}   escape "
                  f"{escape(CL[b], C, R) * 1e3:+8.2f} mm")


def part_fit(CL, seed=1):
    print("    # (body, frame, centre, radius)")
    worst = -np.inf
    for b in MOVING:
        P = CL[b]
        rng = np.random.default_rng(3)
        C = fit(P[rng.choice(len(P), min(12000, len(P)), replace=False)],
                K, seed=seed)
        R = radii_for(P, C)
        Rr = np.ceil(R * 1000.0) / 1000.0            # UP to the millimetre
        worst = max(worst, escape(P, C, Rr))
        for i in np.argsort(-Rr):
            print(f'    ("{b}", {FRAME_OF[b]}, ({C[i][0]:+.4f}, {C[i][1]:+.4f},'
                  f' {C[i][2]:+.4f}), {Rr[i]:.3f}),   # mesh {R[i]:.4f}')
    print(f"    # worst escape over all eight bodies: {worst * 1e3:+.4f} mm")


def part_accuracy(CL, n_pose=6, seed=4):
    """The five moving chain capsules against these spheres, in the metal."""
    from aris_sixarm import coordination as CO, link_spheres as LS
    from aris_sixarm.frames import FR3_MAX, FR3_MIN, fk, link_frames_many
    caps = CO.CAPSULES[CO.N_BASE:-1]
    rng = np.random.default_rng(seed)

    def pt_seg(P, a, b):
        ab = b - a
        den = ab @ ab
        t = np.clip(((P - a) @ ab) / den, 0, 1) if den > 1e-15 else np.zeros(len(P))
        return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)

    Vs, Vc, Os, Oc = [], [], [], []
    for _ in range(n_pose):
        q = FR3_MIN + rng.random(7) * (FR3_MAX - FR3_MIN)
        _T, pts = fk(q)
        C = LS.centres_base(q.reshape(1, 7))[0]
        R = LS.RADII
        L = link_frames_many(q.reshape(1, 7))[0]
        M = np.concatenate([
            CL[b][rng.choice(len(CL[b]), 4000, replace=False)]
            @ L[FRAME_OF[b], :3, :3].T + L[FRAME_OF[b], :3, 3] for b in MOVING])
        lo = np.minimum(C.min(0) - R.max(), M.min(0)) - 0.2
        hi = np.maximum(C.max(0) + R.max(), M.max(0)) + 0.2
        Q = lo + rng.random((500000, 3)) * (hi - lo)
        box = float(np.prod(hi - lo))
        ds = (np.linalg.norm(Q[:, None, :] - C[None], axis=2) - R[None]).min(1)
        dc = np.full(len(Q), np.inf)
        for i, j, r in caps:
            dc = np.minimum(dc, pt_seg(Q, pts[i], pts[j]) - r)
        Vs.append(box * float((ds <= 0).mean()))
        Vc.append(box * float((dc <= 0).mean()))
        Os.append(-(np.linalg.norm(M[:, None, :] - C[None], axis=2)
                    - R[None]).min(1))
        oc = np.full(len(M), np.inf)
        for i, j, r in caps:
            oc = np.minimum(oc, pt_seg(M, pts[i], pts[j]) - r)
        Oc.append(-oc)
    os_, oc_ = np.concatenate(Os), np.concatenate(Oc)
    print(f"  five moving capsules  {np.mean(Vc) * 1e3:6.2f} L   "
          f"offset mean {oc_.mean() * 1e3:5.1f} mm  max {oc_.max() * 1e3:6.1f} mm")
    print(f"  {LS.N_SPHERE} fitted spheres     {np.mean(Vs) * 1e3:6.2f} L   "
          f"offset mean {os_.mean() * 1e3:5.1f} mm  max {os_.max() * 1e3:6.1f} mm")
    print(f"  ratio                 {np.mean(Vs) / np.mean(Vc):6.3f}     "
          f"          {(os_.mean() - oc_.mean()) * 1e3:+5.1f} mm      "
          f"{(os_.max() - oc_.max()) * 1e3:+6.1f} mm")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="validate",
                    choices=("validate", "fit", "accuracy"))
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    CL = cloud()
    {"validate": lambda: part_validate(CL),
     "fit": lambda: part_fit(CL, a.seed),
     "accuracy": lambda: part_accuracy(CL)}[a.part]()
