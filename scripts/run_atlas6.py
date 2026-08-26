#!/usr/bin/env python3
"""Sweep ALL SIX arms over the CONTINUOUS canvas — the seam included.

    python3 scripts/run_atlas6.py                 # -> out/atlas_final6_opt/
    python3 scripts/run_atlas6.py --rig final6    # the as-drawn side arms

Everything before this was either a three-arm sweep of one 1.8034 x 1.700 m web
(`scripts/run_atlas.py` -> out/atlas_final) or a PREVIEW that mirrored it
(`scripts/make_atlas6_preview.py`), and the preview said so in its own title:
the seam strip and the cross-web region were REACH-BOUND ONLY, never scored.
Two user decisions turn that into a real question:

  * MERGE_WEBS — the paper is ONE surface spanning both units, so the 23.064 cm
    strip between the two webs is drawable and has to be swept, not assumed;
  * the extended poles — both side arms re-clamped 20 cm lower (canvas z 0.576)
    on 20 cm of new profile, which is what puts the middle cluster in reach of
    each other's half.

Neither is expressible as a reflection of the old atlas, so this re-sweeps all
six arms from scratch at the same 2 cm grid, the same gates (margin >= 0.30,
sigma_min >= 0.14), the same 15-degree tilt cone and the same 110 mm pen, with
BOTH frames' collision boxes active and the pole extension in them.

Writes
    <out>/atlas_arm<id>.npz   one per arm (aris_sixarm.atlas's own format)
    <out>/coverage.npz        xs, ys, per_arm_go/reach, union, count, arms
    <out>/summary.json        every number this prints
    out/atlas_final6_opt.png  the combined coverage map
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).parents[1]
# THE RIG IS CHOSEN BEFORE THE FIRST aris_sixarm IMPORT, because `atlas.py`
# binds SHEET into its own namespace at import time and nothing later can
# reach that binding.  See fleet.activate's docstring.
_RIG = "final6_opt"
for _i, _a in enumerate(sys.argv):
    if _a == "--rig" and _i + 1 < len(sys.argv):
        _RIG = sys.argv[_i + 1]
os.environ["ARIS_RIG"] = _RIG
sys.path.insert(0, str(ROOT))

import numpy as np                                             # noqa: E402
from aris_sixarm import rig_final6 as r6                       # noqa: E402
from aris_sixarm.atlas import strict_go, sweep_arm             # noqa: E402
from aris_sixarm.fleet import ACTIVE_RIG, FLEET, SHEET         # noqa: E402


def masks(arr, xs, ys, grid):
    """One arm's atlas rows -> (reach, go) boolean (ny, nx) canvas masks."""
    reach = np.zeros((len(ys), len(xs)), bool)
    go = np.zeros_like(reach)
    if len(arr):
        ix = np.rint(arr[:, 0] / grid).astype(int)
        iy = np.rint(arr[:, 1] / grid).astype(int)
        reach[iy, ix] = True
        go[iy, ix] = strict_go(arr)
    return reach, go


def band(ys, y0, y1):
    """Row mask for a y band, cell centres."""
    return (ys >= y0 - 1e-9) & (ys <= y1 + 1e-9)


def summarise(xs, ys, per_go, per_reach, arms, grid):
    """Every number the report quotes. -> dict."""
    cnt = per_go.sum(axis=0)
    union = cnt >= 1
    n = union.size
    cell_m2 = grid * grid
    A = [i for i, a in enumerate(arms) if r6.UNIT_OF[a] == "A"]
    B = [i for i, a in enumerate(arms) if r6.UNIT_OF[a] == "B"]
    goA, goB = per_go[A].any(axis=0), per_go[B].any(axis=0)

    y_seam0, y_seam1 = r6.WEB_A[1][1], r6.WEB_B[0][1]     # 1.700 .. 1.93064
    seam = band(ys, y_seam0, y_seam1)
    webA = band(ys, 0.0, y_seam0)
    webB = band(ys, y_seam1, ys[-1])
    # the MIDDLE BAND: everything the four hanging arms share, seam +- 0.4 m
    mid = band(ys, y_seam0 - 0.40, y_seam1 + 0.40)

    def blk(rows, name):
        u, c = union[rows], cnt[rows]
        return dict(name=name, cells=int(u.size),
                    area_m2=round(float(u.size) * cell_m2, 4),
                    go_pct=round(100 * float(u.mean()), 2),
                    ge2_pct=round(100 * float((c >= 2).mean()), 2),
                    ge3_pct=round(100 * float((c >= 3).mean()), 2),
                    max_arms=int(c.max()) if c.size else 0,
                    cross_unit_pct=round(
                        100 * float((goA[rows] & goB[rows]).mean()), 2))

    out = dict(
        rig=ACTIVE_RIG, grid=grid, sheet=[float(SHEET[0]), float(SHEET[1])],
        cells=int(n), area_m2=round(n * cell_m2, 4),
        merge_webs=bool(r6.MERGE_WEBS), seam_m=round(float(r6.SEAM_M), 5),
        side_z=float(FLEET[2].z), pole_ext_cm=float(r6.POLE_EXT_OPT_CM)
        if ACTIVE_RIG == "final6_opt" else 0.0,
        union_go_pct=round(100 * float(union.mean()), 2),
        union_go_cells=int(union.sum()),
        union_reach_pct=round(100 * float(per_reach.any(axis=0).mean()), 2),
        ge2_pct=round(100 * float((cnt >= 2).mean()), 2),
        ge3_pct=round(100 * float((cnt >= 3).mean()), 2),
        ge4_pct=round(100 * float((cnt >= 4).mean()), 2),
        max_arms=int(cnt.max()),
        cross_unit_pct=round(100 * float((goA & goB).mean()), 2),
        dead_pct=round(100 * float((~union).mean()), 2),
        per_arm={}, bands=[blk(webA, "web A (y 0.000-1.700)"),
                           blk(seam, f"SEAM strip (y {y_seam0:.3f}-{y_seam1:.5f})"),
                           blk(webB, f"web B (y {y_seam1:.5f}-{ys[-1]:.3f})"),
                           blk(mid, "middle band (seam +- 0.40 m)")],
        histogram={str(k): int((cnt == k).sum()) for k in range(int(cnt.max()) + 1)})
    for i, a in enumerate(arms):
        g, r = per_go[i], per_reach[i]
        out["per_arm"][str(a)] = dict(
            name=FLEET[a].name, mount=FLEET[a].mount, unit=r6.UNIT_OF[a],
            reach=int(r.sum()), go=int(g.sum()),
            go_pct=round(100 * float(g.mean()), 2),
            go_m2=round(float(g.sum()) * cell_m2, 4),
            seam_go=int(g[seam].sum()),
            seam_go_pct=round(100 * float(g[seam].mean()), 2),
            own_web_go_pct=round(100 * float(
                g[webA if r6.UNIT_OF[a] == "A" else webB].mean()), 2),
            other_web_go=int(g[webB if r6.UNIT_OF[a] == "A" else webA].sum()))
    return out


def figure(png, xs, ys, per_go, arms, S, S_):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    cnt = per_go.sum(axis=0)
    half = 0.5 * (xs[1] - xs[0])
    ext = (ys[0] - half, ys[-1] + half, xs[0] - half, xs[-1] + half)
    ym = r6.MIRROR_PLANE_CANVAS_Y
    y_seam0, y_seam1 = r6.WEB_A[1][1], r6.WEB_B[0][1]
    col = {a: np.array(FLEET[a].color) for a in arms}

    fig = plt.figure(figsize=(17.2, 12.4), facecolor="white")
    gs = fig.add_gridspec(3, 2, height_ratios=[1.28, 1.28, 1.0],
                          width_ratios=[1.68, 1.0], hspace=0.50, wspace=0.09,
                          left=0.045, right=0.995, top=0.925, bottom=0.048)
    axA, axB, axP = (fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]),
                     fig.add_subplot(gs[2, 0]))
    axT = fig.add_subplot(gs[:, 1])

    # --- who ---------------------------------------------------------------
    img = np.ones(cnt.shape + (3,)) * 0.93
    for i, a in enumerate(arms):
        img[per_go[i] & (cnt == 1)] = col[a]
    img[cnt == 2] = (0.13, 0.13, 0.16)
    img[cnt >= 3] = (0.62, 0.00, 0.42)
    axA.imshow(np.transpose(img, (1, 0, 2)), origin="lower", extent=ext,
               aspect="equal", interpolation="nearest")
    axA.set_title("Strict-GO coverage, six arms, ONE continuous canvas — a real "
                  "sweep, seam included", fontsize=12.5, fontweight="bold",
                  loc="left")

    # --- how many ----------------------------------------------------------
    lut = np.array([[0.93, 0.93, 0.93], [0.62, 0.76, 0.88], [0.95, 0.55, 0.10],
                    [0.75, 0.10, 0.10], [0.36, 0.05, 0.36], [0.10, 0.02, 0.20]])
    axB.imshow(lut[np.clip(cnt, 0, 5)].transpose(1, 0, 2), origin="lower",
               extent=ext, aspect="equal", interpolation="nearest")
    axB.set_title("How many arms can strictly draw a cell", fontsize=12.5,
                  fontweight="bold", loc="left")

    for ax in (axA, axB):
        ax.set_facecolor("#F2F2F0")
        ax.add_patch(Rectangle((y_seam0, xs[0] - half), y_seam1 - y_seam0,
                               xs[-1] - xs[0] + 2 * half, facecolor="none",
                               edgecolor="#B0006E", hatch="////", lw=1.2,
                               zorder=3))
        ax.axvline(ym, color="#E0217D", lw=1.5, ls="--", zorder=4)
        for a in arms:
            s = FLEET[a]
            mk = {"floor": "s", "inv": "v", "wall": ">"}[s.mount]
            ax.plot(s.xy[1], s.xy[0], mk, ms=9, mfc=col[a], mec="k", mew=1.0,
                    zorder=5, clip_on=False)
            ax.annotate(f"{a}", (s.xy[1], s.xy[0]), textcoords="offset points",
                        xytext=(0, 11), ha="center", fontsize=9,
                        fontweight="bold", zorder=6, clip_on=False)
        ax.add_patch(Rectangle((0, 0), S[1], S[0], fill=False, ec="#333",
                               lw=1.2, zorder=2))
        ax.set_xlim(-0.30, S[1] + 0.30)
        ax.set_ylim(-0.20, S[0] + 0.22)
        ax.set_xlabel("canvas y  [m]   (the long axis)")
        ax.set_ylabel("canvas x  [m]")
        ax.grid(alpha=0.16, lw=0.5)
        ax.text(0.90, S[0] + 0.10, "UNIT A", ha="center", fontsize=10,
                fontweight="bold", color="#555")
        ax.text(2 * ym - 0.90, S[0] + 0.10, "UNIT B", ha="center", fontsize=10,
                fontweight="bold", color="#555")
    axB.text(ym, -0.18, f"seam {100 * r6.SEAM_M:.1f} cm — NOW ONE WEB",
             color="#B0006E", ha="center", va="bottom", fontsize=8.5,
             fontweight="bold")
    axA.legend(handles=[Patch(fc=col[a], label=f"arm {a} only") for a in arms]
               + [Patch(fc=(0.13, 0.13, 0.16), label="2 arms"),
                  Patch(fc=(0.62, 0.0, 0.42), label="3+ arms"),
                  Patch(fc=(0.93, 0.93, 0.93), label="no arm")],
               ncol=9, fontsize=8.2, loc="upper center", handlelength=1.3,
               columnspacing=1.0, bbox_to_anchor=(0.5, -0.185), frameon=False)
    axB.legend(handles=[Patch(fc=lut[i], label=f"{i} arm{'' if i == 1 else 's'}")
                        for i in range(min(6, int(cnt.max()) + 1))],
               ncol=6, fontsize=8.2, loc="upper center", handlelength=1.3,
               bbox_to_anchor=(0.5, -0.185), frameon=False)

    # --- profile -----------------------------------------------------------
    axP.fill_between(ys, 0, 100 * (cnt >= 1).mean(axis=1), color="#8FB4D9",
                     label="$\\geq$1 arm")
    axP.fill_between(ys, 0, 100 * (cnt >= 2).mean(axis=1), color="#F28C1A",
                     label="$\\geq$2 arms")
    axP.fill_between(ys, 0, 100 * (cnt >= 3).mean(axis=1), color="#8B1A5A",
                     label="$\\geq$3 arms")
    axP.axvspan(y_seam0, y_seam1, color="#B0006E", alpha=0.14)
    axP.axvline(ym, color="#E0217D", lw=1.4, ls="--")
    axP.set_xlim(-0.03, S[1] + 0.03)
    axP.set_ylim(0, 100)
    axP.set_xlabel("canvas y  [m]")
    axP.set_ylabel("% of the canvas width covered")
    axP.set_title("Coverage profile along the long axis", fontsize=11,
                  fontweight="bold", loc="left")
    axP.legend(fontsize=9, loc="upper center", ncol=3, frameon=False)
    axP.grid(alpha=0.25, lw=0.5)

    # --- numbers -----------------------------------------------------------
    lines = [
        "ONE CONTINUOUS CANVAS",
        f"  {S[0]:.4f} x {S[1]:.5f} m = {S_['area_m2']:.3f} m²"
        f"  ({S_['cells']} cells @ 2 cm)",
        f"  union strict-GO   {S_['union_go_pct']:.2f} %"
        f"   ({S_['union_go_cells']} cells)",
        f"  reachable at all  {S_['union_reach_pct']:.2f} %",
        f"  dead (nobody)     {S_['dead_pct']:.2f} %",
        "",
        "PER ARM, of the whole canvas",
    ]
    for a in arms:
        p = S_["per_arm"][str(a)]
        lines.append(f"  {a:>2} {p['name']:<12} {p['go_pct']:5.2f} %"
                     f"  ({p['go_m2']:.3f} m²)  unit {p['unit']}")
    lines += [
        "",
        "OVERLAP STRUCTURE",
        f"  >=2 arms {S_['ge2_pct']:.2f} %    >=3 arms {S_['ge3_pct']:.2f} %"
        f"    >=4 {S_['ge4_pct']:.2f} %",
        f"  most arms over one cell: {S_['max_arms']}",
        f"  CROSS-UNIT (an A arm and a B arm both GO): "
        f"{S_['cross_unit_pct']:.2f} %",
        "",
        "BY BAND",
    ]
    for b in S_["bands"]:
        lines.append(f"  {b['name']}")
        lines.append(f"     {b['area_m2']:6.3f} m²  GO {b['go_pct']:5.2f} %"
                     f"  >=2 {b['ge2_pct']:5.2f} %  cross-unit "
                     f"{b['cross_unit_pct']:5.2f} %")
    lines += [
        "",
        "SEAM STRIP, ARM BY ARM  (the payoff)",
    ]
    for a in arms:
        p = S_["per_arm"][str(a)]
        lines.append(f"  {a:>2}  {p['seam_go_pct']:5.2f} % of the strip"
                     f"   ({p['seam_go']} cells)")
    lines += ["", "ASSUMPTIONS — FLAGGED IN EVERY OUTPUT"]
    lines += ([
        f"  * BOTH side poles LENGTHENED {S_['pole_ext_cm']:.0f} cm and arms 2",
        f"    and 97 RE-CLAMPED at canvas z {S_['side_z']:.3f} m.  That steel is",
        "    not in the drawing.  Every number above assumes it is fitted.",
    ] if S_["pole_ext_cm"] else [
        f"  * side arms AS DRAWN at canvas z {S_['side_z']:.3f} m — hanging off",
        "    the ends of their poles, which the drawing itself calls",
        "    'not stiff and stable' (rig_final6.FLEET_FINAL6_OPT is the fix).",
    ])
    lines += [
        "  * the paper is ONE web, flat and drawable across the",
        f"    {100 * r6.SEAM_M:.1f} cm strip over the two frames' abutting rails.",
        "  * the two frames abut with ZERO gap (rig_final6.GAP_CM).",
        "  * unit-B ids 17 / 71 / 97 assumed; no drawing names them.",
        "  * base-plate yaws unconfirmed (FINAL_RIG Flags #6), and the",
        "    mirror propagates that to unit B unchanged.",
    ]
    axT.axis("off")
    axT.text(0.0, 1.0, "\n".join(lines), va="top", ha="left",
             family="monospace", fontsize=8.5, linespacing=1.38,
             transform=axT.transAxes)

    fig.suptitle("ARIS — six arms, one continuous canvas, "
                 + ("extended poles" if S_["pole_ext_cm"] else "poles as drawn")
                 + ": the real sweep", fontsize=14.5, fontweight="bold",
                 x=0.045, ha="left", y=0.975)
    Path(png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=140, facecolor="white")
    plt.close(fig)
    print("wrote", png)


def main():
    ap = argparse.ArgumentParser()
    # `proposed` is a six-arm rig on the same canvas, so every number this
    # script computes is defined for it — the seam bands simply describe a
    # seam its one continuous web does not have.
    ap.add_argument("--rig", default=_RIG,
                    choices=("final6", "final6_opt", "proposed"))
    ap.add_argument("--grid", type=float, default=0.02)
    ap.add_argument("--tilt", type=float, default=15.0)
    ap.add_argument("--pen", type=float, default=0.110)
    ap.add_argument("--pen-lat", type=float, default=None,
                    help="lateral tool offset (None = the ACTIVE tool, i.e. 0 unless ARIS_TOOL=lateral)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--png", default=None)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--replot", action="store_true",
                    help="reuse the atlases already under --out")
    a = ap.parse_args()
    out = Path(a.out or (ROOT / "out" / f"atlas_{ACTIVE_RIG}"))
    png = Path(a.png or (ROOT / "out" / f"atlas_{ACTIVE_RIG}.png"))
    out.mkdir(parents=True, exist_ok=True)
    arms = sorted(FLEET)

    print(f"rig {ACTIVE_RIG}: {len(arms)} arms {arms}, canvas "
          f"{SHEET[0]:.4f} x {SHEET[1]:.5f} m, MERGE_WEBS={r6.MERGE_WEBS}, "
          f"side arms at canvas z {FLEET[2].z:.3f}")
    print(f"  frame boxes per arm: "
          + ", ".join(f"{x}:{len(FLEET[x].static_obstacles())}" for x in arms))

    if a.replot:
        from aris_sixarm.atlas import load
        arrs = [load(out, x)[0] for x in arms]
    else:
        fn = partial(sweep_arm, out_dir=str(out), grid=a.grid,
                     tilt_max_deg=a.tilt, pen_ext=a.pen, fleet=FLEET,
                     sheet=SHEET, pen_lat=a.pen_lat)
        with mp.get_context("fork").Pool(min(a.jobs, len(arms))) as pool:
            arrs = pool.map(fn, arms)

    xs = np.arange(0.0, SHEET[0] + 1e-9, a.grid)
    ys = np.arange(0.0, SHEET[1] + 1e-9, a.grid)
    per_reach, per_go = [], []
    for arr in arrs:
        r, g = masks(arr, xs, ys, a.grid)
        per_reach.append(r)
        per_go.append(g)
    per_reach, per_go = np.array(per_reach), np.array(per_go)
    cnt = per_go.sum(axis=0)
    np.savez_compressed(out / "coverage.npz", xs=xs, ys=ys,
                        per_arm_go=per_go, per_arm_reach=per_reach,
                        union_go=cnt >= 1, union_reach=per_reach.any(axis=0),
                        count_go=cnt, arms=np.array(arms, np.int64),
                        rig=np.array(ACTIVE_RIG))

    S = summarise(xs, ys, per_go, per_reach, arms, a.grid)
    (out / "summary.json").write_text(json.dumps(S, indent=1))

    print(f"\n{'arm':>4} {'name':<13} {'unit':>4} {'reach':>7} {'GO':>7} "
          f"{'% canvas':>9} {'% own web':>10} {'% SEAM':>8} {'other web':>10}")
    for x in arms:
        p = S["per_arm"][str(x)]
        print(f"{x:>4} {p['name']:<13} {p['unit']:>4} {p['reach']:>7} "
              f"{p['go']:>7} {p['go_pct']:>8.2f}% {p['own_web_go_pct']:>9.2f}% "
              f"{p['seam_go_pct']:>7.2f}% {p['other_web_go']:>10}")
    print(f"\nUNION strict-GO {S['union_go_pct']:.2f} % of {S['cells']} cells "
          f"({S['area_m2']:.3f} m²); reachable {S['union_reach_pct']:.2f} %; "
          f"dead {S['dead_pct']:.2f} %")
    print(f"  >=2 arms {S['ge2_pct']:.2f} %   >=3 {S['ge3_pct']:.2f} %   "
          f">=4 {S['ge4_pct']:.2f} %   max {S['max_arms']}   "
          f"CROSS-UNIT {S['cross_unit_pct']:.2f} %")
    print("  overlap histogram (cells by arm count): "
          + ", ".join(f"{k}:{v}" for k, v in S["histogram"].items()))
    for b in S["bands"]:
        print(f"  {b['name']:<40} {b['area_m2']:6.3f} m²  GO {b['go_pct']:6.2f} %"
              f"  >=2 {b['ge2_pct']:6.2f} %  >=3 {b['ge3_pct']:5.2f} %  "
              f"cross-unit {b['cross_unit_pct']:6.2f} %")
    print("\nASSUMES: "
          + (f"both side poles extended {S['pole_ext_cm']:.0f} cm and arms 2/97 "
             f"re-clamped at canvas z {S['side_z']:.3f} m (NOT DRAWN STEEL); "
             if S["pole_ext_cm"] else
             f"side arms AS DRAWN at canvas z {S['side_z']:.3f} m; ")
          + f"one continuous web across the {100 * r6.SEAM_M:.1f} cm seam; "
            "zero gap between the frames.")

    figure(png, xs, ys, per_go, arms, SHEET, S)
    print(f"wrote {out}/coverage.npz, {out}/summary.json")


if __name__ == "__main__":
    main()
