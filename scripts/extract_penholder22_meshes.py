#!/usr/bin/env python3
"""Turn the 22-deg pen-holder CAD into two committed, hand-frame visual meshes.

Input (NOT in the repo — 65 MB of raw CAD stays in the dump):
    raw_slack_file_dump/"Pen holder all parts 2026.08.19"/*.STL
Output (committed, ~5k faces each, metres, ALREADY IN THE panda_hand FRAME):
    assets/proposed_rig/meshes/penholder22_housing_hand.obj
    assets/proposed_rig/meshes/penholder22_cap_hand.obj

Baking the placement into the mesh is the final rig's own convention
(`assets/final_rig/meshes/penholder_rig10_panda_hand_frame.stl`): the URDF then
carries the tool as one visual at the identity, and there is exactly one place
— `rig_final.penholder22_T_hand` — where the inferred placement lives.

OBJ, not STL: drake's VTK render engine and its meshcat both IGNORE .stl
("RenderEngineVtk only supports .obj and .gltf"), so the final rig's committed
STL tool mesh does not actually appear in a drake render.  This one does.

THE DELIVERY HAS NO ASSEMBLY FILE.  Every placement here is INFERRED; the
inference, its three assumptions and what it costs are written out in
`rig_final.PENHOLDER22` / `penholder22_T_hand`.  This script MEASURES the
housing rather than trusting those constants, and fails loudly if the CAD ever
stops agreeing with them.

    python3 scripts/extract_penholder22_meshes.py [--faces 5000]
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import trimesh

logging.getLogger("trimesh").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aris_sixarm import rig_final  # noqa: E402
from aris_sixarm.frames import D_HAND_TCP, PEN_EXT, PEN_LAT_HOLDER  # noqa: E402

CAD = (ROOT.parent / "raw_slack_file_dump" / "Pen holder all parts 2026.08.19")
HOUSING = "pen holder housing - 22 deg - reinforced - v20260429.STL"
CAP = "pen holder cap v20250903.STL"
CLUTCH = "pen clutch - Creatcolor monolith graphite v1.01.STL"
OUT_DIR = ROOT / "assets/proposed_rig/meshes"
MM = 0.001
P = rig_final.PENHOLDER22


def load_mm(name):
    m = trimesh.load(CAD / name, force="mesh")
    ext = m.bounds[1] - m.bounds[0]
    # UNITS.  A printed pen holder is a 10 cm object.  If these numbers were
    # metres the part would be 80 m long; if centimetres, 80 cm.  Millimetres
    # is the only reading that survives, and SolidWorks STL defaults to mm.
    assert 5.0 < ext.max() < 500.0, f"{name}: extent {ext} is not millimetres"
    return m


def measure_housing(m):
    """Re-derive every PENHOLDER22 housing constant from the mesh itself."""
    V, n, a, c = m.vertices, m.face_normals, m.area_faces, m.triangles_center
    out = {}
    # THE CLOCKING.  Cluster the triangles into PLANES (same normal, same
    # offset) by descending total area; the four biggest planes whose normal
    # lies in the housing's xy plane and is off-axis are the mount post's four
    # flats.  A square has four of them 90 deg apart, so fold the angle into
    # [0, 45] and the answer is unique.
    d = np.einsum("ij,ij->i", n, c)
    used = np.zeros(len(n), bool)
    planes = []
    for i in np.argsort(-a):
        if used[i]:
            continue
        sel = (n @ n[i] > 0.9995) & (np.abs(d - d[i]) < 0.05) & ~used
        used |= sel
        planes.append((a[sel].sum(), n[i].copy(), float(d[i])))
    planes.sort(key=lambda t: -t[0])
    post = [p for p in planes if abs(p[1][2]) < 1e-6
            and min(abs(p[1][0]), abs(p[1][1])) > 0.05][:4]
    assert len(post) == 4, "the mount post's four flats are not the four "\
                           "largest off-axis planes any more"
    ang = [np.degrees(np.arctan2(nv[1], nv[0])) % 90.0 for _, nv, _ in post]
    out["clock_deg"] = float(min(np.median(ang), 90.0 - np.median(ang)))
    th = np.radians(out["clock_deg"])
    u1 = np.array([np.sin(th), np.cos(th)])          # a flat normal
    u2 = np.array([u1[1], -u1[0]])
    # each flat's SIGNED offset along its own axis -> the square's two pairs
    off = {}
    for _, nv, dv in post:
        for k, u in ((1, u1), (2, u2)):
            if abs(nv[:2] @ u) > 0.999:
                off.setdefault(k, []).append(dv * np.sign(nv[:2] @ u))
    out["post_side_mm"] = float(max(max(v) - min(v) for v in off.values()))
    ctr = 0.5 * (max(off[1]) + min(off[1])) * u1 \
        + 0.5 * (max(off[2]) + min(off[2])) * u2
    out["post_x_mm"], out["post_y_mm"] = float(ctr[0]), float(ctr[1])
    out["post_len_mm"] = float(np.ptp(V[:, 2]))
    # THE BARREL, over its plain span only (the post starts at x ~ 38)
    r = np.linalg.norm(V[:, 1:] - np.array(P["bore_yz"]) * 1000.0, axis=1)
    plain = V[:, 0] <= 36.0
    out["barrel_r_mm"] = float(r[plain].max())
    out["bore_r_mm"] = float(r[(V[:, 0] > 8) & (V[:, 0] < 28)].min())
    out["nose_bore_r_mm"] = float(r[V[:, 0] < 2].min())
    out["thread_r_mm"] = float(r[V[:, 0] >= P["thread_x"][0] * 1000.0].max())
    out["len_mm"] = float(V[:, 0].max())
    return out


def decimate(m, faces):
    """Weld, THEN decimate — the order the repo learned the hard way (the
    simplifier tears the surface at every seam otherwise; see
    aris_sixarm/viz/robot_model.py)."""
    before = len(m.faces)
    m.merge_vertices(merge_tex=True, merge_norm=True)
    if len(m.faces) > faces:
        m = m.simplify_quadric_decimation(face_count=faces)
    return m, before, len(m.faces)


def export(m, T, path):
    """Scale to metres, place in the hand frame, write OBJ with normals."""
    m = m.copy()
    m.apply_scale(MM)
    m.apply_transform(T)
    m.vertex_normals
    path.write_text(trimesh.exchange.obj.export_obj(m, include_normals=True))
    return m


def outside_envelope(pts):
    """How far each hand-frame point lies OUTSIDE the HOUSING's envelope.

    -> (N,) metres, 0 where the point is inside at least one primitive.  This
    is the conservatism check the brief asks for: the primitives must enclose
    the visual, and a number is the only way to know that they do.

    The first THREE primitives only, deliberately.  `penholder22_collision`
    returns four now — the fourth is the pencil tail's own envelope, off at
    negative x — and letting it into this proof would mean the housing was
    being checked against a hull that was not fitted to it.  The order is part
    of `penholder22_hull`'s contract.
    """
    prims = rig_final.penholder22_collision(PEN_EXT, PEN_LAT_HOLDER,
                                            D_HAND_TCP)
    assert len(prims) == 4, "penholder22_hull has changed shape"
    worst = np.full(len(pts), np.inf)
    for kind, T, par in prims[:3]:
        q = (pts - T[:3, 3]) @ T[:3, :3]            # into the primitive frame
        if kind == "cylinder":
            r, L = par
            d = np.maximum(np.linalg.norm(q[:, :2], axis=1) - r,
                           np.abs(q[:, 2]) - L / 2.0)
        else:
            h = np.asarray(par, float) / 2.0
            d = np.max(np.abs(q) - h, axis=1)
        worst = np.minimum(worst, d)
    return np.maximum(worst, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faces", type=int, default=5000)
    args = ap.parse_args()
    if not CAD.is_dir():
        sys.exit(f"CAD not found: {CAD}\n(the raw delivery stays in "
                 f"raw_slack_file_dump/ and is not part of the repo)")

    housing, cap = load_mm(HOUSING), load_mm(CAP)
    clutch = load_mm(CLUTCH)
    print(f"UNITS: millimetres (housing extent "
          f"{np.round(housing.bounds[1] - housing.bounds[0], 2)} mm) -> x{MM}")

    meas = measure_housing(housing)
    print("MEASURED housing (mm / deg):")
    for k, v in meas.items():
        print(f"   {k:16s} {v:9.3f}")
    Vc = clutch.vertices
    mid = (Vc[:, 0] > 10.0) & (Vc[:, 0] < 30.0)      # away from the jaws
    lead_r = float(np.linalg.norm(
        Vc[mid][:, 1:] - np.array([8.53, 8.53]), axis=1).min())
    print(f"   clutch bore r   {lead_r:9.3f}  (the graphite stick)")

    # the constants must be the measurements, or PENHOLDER22 is a fiction
    for got, want, tol, tag in (
            (meas["clock_deg"], np.degrees(P["post_clock"]), 0.02, "clocking"),
            (meas["barrel_r_mm"], P["barrel_r"] * 1000, 0.02, "barrel_r"),
            (meas["post_side_mm"], P["post_side"] * 1000, 0.05, "post_side"),
            (meas["post_x_mm"], P["post_xy"][0] * 1000, 0.01, "post_x"),
            (meas["post_y_mm"], P["post_xy"][1] * 1000, 0.01, "post_y"),
            (meas["post_len_mm"], (P["post_z"][1] - P["post_z"][0]) * 1000,
             0.01, "post_len"),
            (meas["thread_r_mm"], P["thread_r"] * 1000, 0.02, "thread_r"),
            (meas["len_mm"], P["thread_x"][1] * 1000, 0.02, "housing length"),
            (lead_r, P["lead_r"] * 1000, 0.02, "lead_r")):
        assert abs(got - want) <= tol, f"{tag}: CAD {got} vs PENHOLDER22 {want}"
    print("   PASS  every PENHOLDER22 constant reproduces from the CAD")
    print(f"\n'22 deg' VERDICT: the flats clock {meas['clock_deg']:.2f} deg "
          f"about the POST axis relative to the bore (the file is named "
          f"'22 deg'); the post axis is perpendicular to the bore, so this is "
          f"a lean about y_hand, not a tilt of the mount.")

    T_h, T_c, exit_x, reach = rig_final.penholder22_T_hand(
        PEN_EXT, PEN_LAT_HOLDER, D_HAND_TCP)
    lean = np.degrees(np.arctan2(PEN_LAT_HOLDER, PEN_EXT))
    print(f"PLANNER asks for a {lean:.2f} deg lean and {reach * 1000:.2f} mm "
          f"of reach from the TCP; the housing gives {meas['clock_deg']:.2f} "
          f"deg and {exit_x * 1000:.2f} mm grip-to-exit (the CAP's outer face; "
          f"{P['post_xy'][0] * 1000:.2f} mm of barrel stands the other way, "
          f"behind the grip), so the drawing needs "
          f"{(reach - exit_x) * 1000:.1f} mm of graphite past the cap and "
          f"{lean - meas['clock_deg']:.2f} deg out of the fingertip cradle.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outs = []
    print(f"\nDECIMATION (weld first, target {args.faces} faces):")
    for m, T, name in ((housing, T_h, "penholder22_housing_hand.obj"),
                       (cap, T_c, "penholder22_cap_hand.obj")):
        ext0 = m.bounds[1] - m.bounds[0]
        d, before, after = decimate(m, args.faces)
        ext1 = d.bounds[1] - d.bounds[0]
        out = export(d, T, OUT_DIR / name)
        shrink = float(np.max(np.abs(ext1 - ext0)))
        print(f"   {name}: {before} -> {after} faces, extent lost "
              f"{shrink:.3f} mm, hand-frame bbox "
              f"{np.round(out.bounds[0], 4)} .. {np.round(out.bounds[1], 4)} m")
        assert shrink < 1.5, f"{name}: decimation ate {shrink:.2f} mm"
        # CONSERVATISM: the primitive envelope must contain the visual
        outs.append((name, outside_envelope(np.asarray(out.vertices))))

    print("\nCOLLISION ENVELOPE (rig_final.penholder22_collision: three "
          "cylinders coaxial with the bore):")
    for name, o in outs:
        print(f"   {name}: {int((o > 0).sum())} of {len(o)} visual vertices "
              f"outside, worst {o.max() * 1000:.3f} mm")
        assert o.max() <= 1e-9, f"{name}: the envelope does not enclose it"
    print("   PASS  the envelope encloses every visual vertex")

    tip = np.array([PEN_LAT_HOLDER, 0.0, D_HAND_TCP + PEN_EXT])
    print(f"\npen tip (planning, hand frame): {np.round(tip, 4)} m — the "
          f"meshes are placed to point at it, and nothing here moves it.")


if __name__ == "__main__":
    main()
