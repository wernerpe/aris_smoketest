#!/usr/bin/env python3
"""Single-arm stroke planner demo on arm 31 (inverted).

Stroke A: rim arc at r = 0.66 around the arm-31 base — passable, but only with
          global redundancy planning; greedy (diffIK-like) gets stuck almost
          immediately.
Stroke B: straight line through the under-base dead zone — the fiber is
          literally empty for ~0.34 m, so the band disconnects and the planner
          reports the split point s* instead of a plan.

Outputs: out/stroke_band.png, out/stroke_demo.html
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parents[1]))
from aris_sixarm import planner  # noqa: E402
from aris_sixarm.fleet import FLEET, SHEET  # noqa: E402
from aris_sixarm.frames import tip_pos  # noqa: E402

ROOT = Path(__file__).parents[1]
spec = FLEET[31]
bx, by = spec.xy
clip_to_sheet = planner.clip_to_sheet     # longest contiguous in-sheet run


# Stroke A: long rim arc, clipped to the sheet
th = np.linspace(-0.6 * np.pi, 1.05 * np.pi, 400)
arcA = clip_to_sheet(np.column_stack([bx + 0.66 * np.cos(th),
                                      by + 0.66 * np.sin(th)]))
# Stroke B: line straight through the under-base dead zone
lineB = np.array([[bx - 0.55, by - 0.35], [bx + 0.55, by + 0.35]])

results = {}
for name, pts_in, ds in [("A_rim_arc", arcA, 0.012), ("B_center_line", lineB, 0.008)]:
    pts, _ = planner.resample(pts_in, ds)
    Ns = len(pts)
    print(f"stroke {name}: {Ns} steps, len {ds * (Ns - 1):.2f} m")
    lat = planner.build_lattice(pts, spec)
    nvalid = lat["valid"].sum(axis=(1, 2))
    print(f"  fiber: {nvalid.mean():.1f} nodes/step (min {nvalid.min()}, "
          f"{int((nvalid == 0).sum())} empty steps)")
    dp = planner.plan(lat, objective="maximin_sigma")
    gr = planner.greedy(lat)
    for tag, res in (("DP  ", dp), ("greedy", gr)):
        rep = planner.path_report(lat, res)
        if not rep.get("feasible") and "min_sigma" not in rep:
            print(f"  {tag}: ok={res['ok']} cut_s={res['cut_s']:.3f}  (no path)")
            continue
        print(f"  {tag}: ok={res['ok']} cut_s={res['cut_s']:.3f} "
              f"min_sigma={rep['min_sigma']:.4f} min_margin={rep['min_margin']:.3f} "
              f"max_step={rep['max_step']:.3f} travel={rep['sum_travel']:.1f} rad")
    if not dp["ok"]:
        cut = lat["pts"][dp["cut_index"]]
        print(f"  -> SPLIT at s*={dp['cut_s']:.4f}, paper xy=({cut[0]:.4f}, {cut[1]:.4f}), "
              f"r={np.hypot(*(cut - spec.xy)):.3f} m from the base")

    # --- verification: every step on the curve, every step continuous ---
    Twb = lat["Twb"]
    qs = dp["qs"]
    tip_w = np.array([Twb[:3, :3] @ tip_pos(q, lat["pen_ext"]) + Twb[:3, 3] for q in qs])
    err = np.linalg.norm(tip_w[:, :2] - lat["pts"][:len(qs)], axis=1)
    jmp = np.abs(np.diff(qs, axis=0)).max()
    assert err.max() < 2e-3, f"tip off stroke by {err.max():.2e} m"
    assert jmp <= planner.JUMP_THRESH + 1e-12, f"joint jump {jmp:.3f}"
    print(f"  verify: tip-on-curve max {err.max():.2e} m (all {len(qs)} steps), "
          f"|dq|_inf max {jmp:.3f} <= {planner.JUMP_THRESH}, |z| max {abs(tip_w[:,2]).max():.2e} m")
    results[name] = (lat, dp, gr)

# ---- band figure: per-(s, q7) best sigma_min + the two paths ----
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
for ax, (name, (lat, dp, gr)) in zip(axes, results.items()):
    V, S = lat["valid"], lat["sigma"]
    Ns, Nq, _ = V.shape
    best = np.where(V.any(axis=2), np.max(np.where(V, S, -np.inf), axis=2), np.nan)
    band = np.ma.masked_invalid(best)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.88")
    im = ax.imshow(band.T, origin="lower", aspect="auto", cmap=cmap,
                   extent=[0, 1, lat["q7s"][0], lat["q7s"][-1]],
                   vmin=planner.HARD_SIGMA, vmax=float(np.nanmax(best)))
    plt.colorbar(im, ax=ax, label=r"best $\sigma_{min}$ over branches")
    if "q7s" in dp:
        s = np.arange(len(dp["q7s"])) / (Ns - 1)
        ax.plot(s, dp["q7s"], "-", color="#1f4fd8", lw=2.2,
                label=f"DP maximin-$\\sigma$ ({'ok' if dp['ok'] else 'to split'})")
    if "q7s" in gr:
        s = np.arange(len(gr["q7s"])) / (Ns - 1)
        ax.plot(s, gr["q7s"], "--", color="#d62728", lw=1.8,
                label=f"greedy ({'ok' if gr['ok'] else 'STUCK'})")
        if not gr["ok"]:
            ax.plot(s[-1], gr["q7s"][-1], "x", color="#d62728", ms=13, mew=3)
    empty = ~V.any(axis=(1, 2))
    if not dp["ok"]:
        ax.axvline(dp["cut_s"], color="magenta", ls=":", lw=2.2,
                   label=f"split $s^*$={dp['cut_s']:.3f}"
                         + (" (fiber empty)" if empty.any() else ""))
    ax.set_xlabel("arc length s (normalised)")
    ax.set_ylabel(r"$q_7$ (rad)")
    ax.set_title(f"stroke {name}  —  {Ns} steps, fiber = $q_7\\times$branch")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
plt.tight_layout()
plt.savefig(ROOT / "out/stroke_band.png", dpi=140)
print(f"wrote out/stroke_band.png")

# ---- meshcat: ghosts along stroke A ----
import meshcat  # noqa: E402
import meshcat.geometry as g  # noqa: E402
import meshcat.transformations as tf  # noqa: E402
from aris_sixarm.viz import robot_model  # noqa: E402
from aris_sixarm.viz.scene import _hex  # noqa: E402

lat, dp, gr = results["A_rim_arc"]
latB, dpB, _ = results["B_center_line"]
vis = meshcat.Visualizer()
vis["/Background"].set_property("top_color", [0.95, 0.95, 0.97])
vis["/Background"].set_property("bottom_color", [0.85, 0.85, 0.9])
vis["paper"].set_object(g.Box([SHEET[0], SHEET[1], 0.004]),
                        g.MeshLambertMaterial(color=0xFAFAF5))
vis["paper"].set_transform(tf.translation_matrix([SHEET[0] / 2, SHEET[1] / 2, -0.002]))
col = np.array(spec.color)
Twb = lat["Twb"]
vis["base"].set_object(g.Sphere(0.045), g.MeshLambertMaterial(color=_hex(col)))
vis["base"].set_transform(Twb)
vis["boom"].set_object(g.Cylinder(1.3, 0.012),
                       g.MeshLambertMaterial(color=_hex(col), opacity=0.5))
vis["boom"].set_transform(tf.translation_matrix([bx, by, Twb[2, 3] - 0.65])
                          @ tf.rotation_matrix(np.pi / 2, [1, 0, 0]))
pv = np.column_stack([lat["pts"], np.full(len(lat["pts"]), 0.004)])
vis["strokeA"].set_object(g.Line(g.PointsGeometry(pv.T.astype(np.float32)),
                                 g.LineBasicMaterial(color=0x111111)))
pvB = np.column_stack([latB["pts"], np.full(len(latB["pts"]), 0.004)])
vis["strokeB"].set_object(g.Line(g.PointsGeometry(pvB.T.astype(np.float32)),
                                 g.LineBasicMaterial(color=0x999999)))
cutB = latB["pts"][dpB["cut_index"]]
vis["splitB"].set_object(g.Sphere(0.02), g.MeshLambertMaterial(color=0xCC00CC))
vis["splitB"].set_transform(tf.translation_matrix([cutB[0], cutB[1], 0.01]))
links, joints = robot_model.load_model()
ghost_ids = np.linspace(0, len(dp["qs"]) - 1, 6).astype(int)
for n, gi in enumerate(ghost_ids):
    q = dp["qs"][gi]
    tip_w = Twb[:3, :3] @ tip_pos(q, lat["pen_ext"]) + Twb[:3, 3]
    assert np.linalg.norm(tip_w[:2] - lat["pts"][gi]) < 2e-3, "tip off stroke!"
    robot_model.add_robot(vis, f"ghosts/{n:02d}", links, joints, q, Twb,
                          pen_color=_hex(col))
LEG = f"""<div style="position:fixed;top:12px;left:12px;z-index:1000;background:rgba(255,255,255,0.92);
border:1px solid #bbb;border-radius:8px;padding:10px 14px;font:12px/1.5 sans-serif;max-width:340px">
<b>Stroke planner demo &mdash; arm 31</b><br>
Black arc: {0.012 * (len(lat['pts']) - 1):.2f} m rim stroke (r = 0.66) planned end to end by the
ladder-graph DP on the (s &times; q<sub>7</sub> &times; branch) lattice, maximin-&sigma;
objective (min &sigma;<sub>min</sub> = {planner.path_report(lat, dp)['min_sigma']:.3f}) &mdash;
6 ghost poses along it, pen tip verified on the curve at <b>every</b> step.<br>
Grey line: the under-base stroke. Its fiber is empty for ~0.34 m, so the band
disconnects; the magenta dot is the reported split s* = {dpB['cut_s']:.3f}.<br>
See out/stroke_band.png for the (s, q<sub>7</sub>) bands and the greedy-vs-DP comparison.</div>"""
html = vis.static_html().replace("</body>", LEG + "</body>")
(ROOT / "out/stroke_demo.html").write_text(html)
print(f"wrote out/stroke_demo.html ({len(html) / 1e6:.1f} MB)")
