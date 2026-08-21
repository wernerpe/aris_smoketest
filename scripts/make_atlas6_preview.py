#!/usr/bin/env python3
"""Six-arm coverage PREVIEW by symmetry — no new atlas sweep.

Unit B's per-arm strict-GO map is unit A's atlas REFLECTED across the mirror
plane (aris_sixarm/rig_final6.py).  That is EXACT, not an approximation:

  * the mirrored joint vector q * (-1,1,-1,1,-1,1,-1) reproduces the whole FR3
    chain reflected, to 3e-16 (frames.fk), and the pen tip with it;
  * the four sign-flipped joints are exactly the four whose FR3 limits are
    antisymmetric, so `joint_margin` is IDENTICAL, cell for cell;
  * the mirrored Jacobian is S J diag(sign), so sigma_min is identical and
    f_max is identical against the mirrored press direction — those are the
    two quantities `atlas.strict_go` gates on;
  * the atlas candidate set is closed under the mirror (8 tool yaws at 45 deg
    steps map psi -> pi/2 - psi; the 4 tilt axes map onto themselves; Q7_GRID
    is symmetric);
  * unit B's frame boxes are the exact reflection of unit A's, and adding the
    SECOND frame removes 0 of unit A's 5693 strict-GO cells (re-checked here:
    the chain clearance is bit-identical with 35 boxes and with 70).

The one thing that is NOT exact is what lies BEYOND each unit's own web, and
this script refuses to invent it: the cross-web and seam regions are reported
as REACH-BOUND ONLY (inside the sweep radius, never scored).

SUPERSEDED, AND KEPT AS THE BEFORE PICTURE.  The user has since merged the webs
and lengthened both side poles, and `scripts/run_atlas6.py` sweeps all six arms
over the whole continuous canvas for real — seam included, no mirroring, the
extended poles in the obstacle set (out/atlas_final6_opt.png,
docs/MERGED_CANVAS.md).  This preview is what the AS-DRAWN two-web layout looked
like, and its headline — cross-unit overlap 0.00 %, the middle cluster buys no
redundancy — is the number the real sweep moved to 8.46 % of the canvas and
74.91 % of the seam strip.

    python3 scripts/make_atlas6_preview.py   # -> out/atlas_final6_preview.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.patches import Patch, Rectangle                  # noqa: E402

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from aris_sixarm import rig_final, rig_final6 as r6              # noqa: E402
from aris_sixarm.atlas import strict_go, load                    # noqa: E402
from aris_sixarm.frames import fk_many, PEN_EXT                  # noqa: E402
from aris_sixarm.atlas import QCOL                               # noqa: E402

RMAX = 1.05          # the atlas's own sweep radius (atlas.sweep_arm default)
BG = "#F2F2F0"


def verify_second_frame_costs_nothing(atlas_dir):
    """Re-derive the exactness claim -> (n_cells, n_lost, worst_delta)."""
    n = lost = 0
    worst = 0.0
    for aid, key in ((13, "up"), (31, "down"), (2, "side")):
        a, _ = load(atlas_dir, aid)
        A = a[strict_go(a)]
        spec = r6.FLEET_FINAL6[aid]
        Twb = spec.T_world_base()
        T, p = fk_many(A[:, QCOL:QCOL + 7])
        tip = T[:, :3, 3] + np.einsum("nij,j->ni", T[:, :3, :3],
                                      np.array([0.0, 0.0, PEN_EXT]))
        P = np.concatenate([p, tip[:, None]], axis=1)
        Pw = np.einsum("ij,nkj->nki", Twb[:3, :3], P) + Twb[:3, 3]
        c3 = rig_final.chain_static_clearance(
            Pw, rig_final.frame_boxes_canvas(exclude_tag=f"mount:{key}"))
        c6 = rig_final.chain_static_clearance(
            Pw, r6.frame_boxes6_canvas(exclude_tag=f"mount:{key}@A"))
        g = rig_final.STATIC_MARGIN
        lost += int(((c3 >= g) & (c6 < g)).sum())
        worst = max(worst, float(np.abs(c3 - c6).max()))
        n += len(A)
    return n, lost, worst


def _within_rmax(aid, X, Y):
    s = r6.FLEET_FINAL6[aid]
    bx, by = s.xy
    return (X - bx) ** 2 + (Y - by) ** 2 + s.z ** 2 <= RMAX ** 2


def build(png_path, atlas_dir):
    d = np.load(Path(atlas_dir) / "coverage.npz")
    xs, ys = d["xs"], d["ys"]
    per = d["per_arm_go"]                       # (3, ny, nx) unit A
    arms = [int(a) for a in d["arms"]]
    ny, nx = per.shape[1:]
    n_web = nx * ny
    ym = r6.MIRROR_PLANE_CANVAS_Y
    (_, yB0), (wW, wH) = r6.WEB_B
    half = 0.01                                  # half a 2 cm cell

    cnt = per.sum(axis=0)                        # unit A overlap count
    union = cnt >= 1
    # unit B = the reflection: same array, y axis reversed
    perB = per[:, ::-1, :]
    cntB = cnt[::-1, :]

    # cell-centre grids -> pixel-edge extents; web B is the exact mirror of A
    extA = (ys[0] - half, ys[-1] + half, xs[0] - half, xs[-1] + half)
    extB = (2 * ym - (ys[-1] + half), 2 * ym - (ys[0] - half),
            xs[0] - half, xs[-1] + half)

    col = {aid: np.array(r6.FLEET_FINAL6[aid].color) for aid in r6.FLEET_FINAL6}
    twin = {2: 97, 13: 17, 31: 71}

    fig = plt.figure(figsize=(16.4, 12.6), facecolor="white")
    gs = fig.add_gridspec(3, 2, height_ratios=[1.30, 1.30, 0.98],
                          width_ratios=[1.62, 1.0],
                          hspace=0.52, wspace=0.10,
                          left=0.045, right=0.995, top=0.925, bottom=0.045)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[1, 0])
    axP = fig.add_subplot(gs[2, 0])
    axT = fig.add_subplot(gs[:, 1])

    # ================= panel A: per-arm identity ==========================
    def identity_rgba(pa, ct, ids):
        img = np.ones(pa.shape[1:] + (4,))
        img[..., :3] = 0.93
        img[..., 3] = 1.0
        for m, aid in zip(pa, ids):
            sel = m & (ct == 1)
            img[sel, :3] = col[aid]
        img[ct >= 2, :3] = (0.10, 0.10, 0.12)     # handoff band
        return img

    for ax in (axA, axB):
        ax.set_facecolor(BG)
    axA.imshow(np.transpose(identity_rgba(per, cnt, arms), (1, 0, 2)),
               origin="lower", extent=extA, aspect="equal", interpolation="nearest")
    axA.imshow(np.transpose(identity_rgba(perB, cntB,
                                          [twin[a] for a in arms]), (1, 0, 2)),
               origin="lower", extent=extB, aspect="equal", interpolation="nearest")
    axA.set_title("Strict-GO coverage, six arms — unit B is unit A MIRRORED "
                  "across the plane (exact; no new sweep)",
                  fontsize=12, fontweight="bold", loc="left")

    # ================= panel B: overlap count =============================
    lut = np.array([[0.93, 0.93, 0.93], [0.62, 0.76, 0.88],
                    [0.95, 0.55, 0.10], [0.75, 0.10, 0.10]])
    axB.imshow(lut[np.clip(cnt, 0, 3)].transpose(1, 0, 2), origin="lower",
               extent=extA, aspect="equal", interpolation="nearest")
    axB.imshow(lut[np.clip(cntB, 0, 3)].transpose(1, 0, 2), origin="lower",
               extent=extB, aspect="equal", interpolation="nearest")
    axB.set_title("How many arms can strictly draw a cell — the middle "
                  "cluster adds NO shared coverage", fontsize=12,
                  fontweight="bold", loc="left")

    # seam + reach-bound annotations on both maps
    seam = Rectangle((wH, xs[0] - half), r6.SEAM_M, xs[-1] - xs[0] + 2 * half,
                     facecolor="none", edgecolor="#B0006E", hatch="////",
                     lw=1.2, zorder=3)
    axB.add_patch(seam)
    for ax in (axA, axB):
        ax.axvline(ym, color="#E0217D", lw=1.6, ls="--", zorder=4)
        for aid, s in r6.FLEET_FINAL6.items():
            mk = {"floor": "s", "inv": "v", "wall": ">"}[s.mount]
            ax.plot(s.xy[1], s.xy[0], mk, ms=9, mfc=col[aid], mec="k",
                    mew=1.0, zorder=5, clip_on=False)
            ax.annotate(f"{aid}", (s.xy[1], s.xy[0]), textcoords="offset points",
                        xytext=(0, 11), ha="center", fontsize=9,
                        fontweight="bold", zorder=6, clip_on=False)
        ax.set_xlim(-0.32, 2 * ym + 0.32)
        ax.set_ylim(-0.17, 1.99)
        ax.set_xlabel("canvas y  [m]   (the mirrored axis)")
        ax.set_ylabel("canvas x  [m]")
        ax.grid(alpha=0.18, lw=0.5)
        for (x0, y0), (w, h) in r6.webs():
            ax.add_patch(Rectangle((y0, x0), h, w, fill=False,
                                   ec="#444", lw=1.0, zorder=2))
    axA.text(ym, -0.15, f"mirror plane  y = {ym:.5f} m", color="#B0006E",
             ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    axB.text(ym, -0.15, f"seam {r6.SEAM_M*100:.1f} cm — no web here",
             color="#B0006E", ha="center", va="bottom", fontsize=8.5,
             fontweight="bold")
    for ax in (axA, axB):
        ax.text(0.90, 1.83, "UNIT A", ha="center", va="bottom", fontsize=10,
                fontweight="bold", color="#555")
        ax.text(2 * ym - 0.90, 1.83, "UNIT B  (mirrored)", ha="center",
                va="bottom", fontsize=10, fontweight="bold", color="#555")

    axA.legend(handles=[Patch(fc=col[a], label=f"arm {a} only")
                        for a in (13, 31, 2, 17, 71, 97)]
               + [Patch(fc=(0.10, 0.10, 0.12), label="2 arms (handoff)"),
                  Patch(fc=(0.93, 0.93, 0.93), label="no arm")],
               ncol=8, fontsize=8.2, loc="upper center", handlelength=1.4,
               columnspacing=1.1, bbox_to_anchor=(0.5, -0.19), frameon=False)
    axB.legend(handles=[Patch(fc=lut[i], label=lab) for i, lab in
                        enumerate(["0 arms", "1 arm", "2 arms", "3 arms"])],
               ncol=4, fontsize=8.2, loc="upper center", handlelength=1.4,
               bbox_to_anchor=(0.5, -0.19), frameon=False)

    # ================= panel C: profile along the mirrored axis ===========
    fr1A = union.mean(axis=1)
    fr2A = (cnt >= 2).mean(axis=1)
    axP.fill_between(ys, 0, 100 * fr1A, color="#8FB4D9", label="≥1 arm")
    axP.fill_between(2 * ym - ys, 0, 100 * fr1A, color="#8FB4D9")
    axP.fill_between(ys, 0, 100 * fr2A, color="#F28C1A", label="≥2 arms")
    axP.fill_between(2 * ym - ys, 0, 100 * fr2A, color="#F28C1A")
    axP.axvspan(wH, wH + r6.SEAM_M, color="#B0006E", alpha=0.13)
    axP.axvline(ym, color="#E0217D", lw=1.4, ls="--")
    axP.set_xlim(-0.05, 2 * ym + 0.05)
    axP.set_ylim(0, 100)
    axP.set_xlabel("canvas y  [m]")
    axP.set_ylabel("% of the web's width covered")
    axP.set_title("Coverage profile along the mirrored axis", fontsize=11,
                  fontweight="bold", loc="left")
    axP.legend(fontsize=9, loc="upper center", ncol=2, frameon=False)
    axP.grid(alpha=0.25, lw=0.5)

    # ================= panel D: the numbers ===============================
    tot = 2 * n_web
    u_pct = 100 * 2 * union.sum() / tot
    back = ys >= wH * 2 / 3
    nb = int(back.sum()) * nx
    b1 = 100 * union[back].sum() / nb
    b2 = 100 * (cnt[back] >= 2).sum() / nb
    X, Y = np.meshgrid(xs, ys)
    lensA = {aid: int(_within_rmax(aid, X, Y).sum()) for aid in (17, 71, 97)}
    ys_seam = np.arange(wH + 0.02, yB0, 0.02)
    XS, YS = np.meshgrid(xs, ys_seam)
    seam_cnt = sum(_within_rmax(a, XS, YS) for a in r6.FLEET_FINAL6)
    n_cells, n_lost, worst = verify_second_frame_costs_nothing(atlas_dir)

    axT.axis("off")
    txt = (
        f"COMBINED DRAWABLE SURFACE\n"
        f"  2 webs x 1.8034 x 1.700 m = 6.132 m²  ({tot} cells @ 2 cm)\n"
        f"  union strict-GO   {u_pct:.2f} %\n"
        f"      identical on each web — the mirror is EXACT\n"
        f"  per arm, of its own web:\n"
        f"      13 / 17   {100*per[arms.index(13)].sum()/n_web:5.1f} %   floor\n"
        f"      31 / 71   {100*per[arms.index(31)].sum()/n_web:5.1f} %   inverted\n"
        f"      2  / 97   {100*per[arms.index(2)].sum()/n_web:5.1f} %   side\n"
        f"  ≥2 arms  {100*(cnt>=2).sum()/n_web:.2f} %    ≥3 arms  "
        f"{100*(cnt>=3).sum()/n_web:.2f} %\n"
        f"  most arms over any one cell:  {int(cnt.max())}\n"
        f"\n"
        f"THE MIDDLE CLUSTER   31 · 2 | 71 · 97\n"
        f"the interesting new number, and it is zero:\n"
        f"  cross-unit overlap on paper      0.00 %\n"
        f"  · the two webs are disjoint in y\n"
        f"    (A ≤ 1.700 m, B ≥ {yB0:.3f} m)\n"
        f"  · the nearest cell of the OTHER web is\n"
        f"    1.142 m from arm 31/71 and 1.028 m from\n"
        f"    arm 2/97 — at or past the atlas's\n"
        f"    {RMAX:.2f} m sweep radius.\n"
        f"  So the 4-arm cluster sits back-to-back, not\n"
        f"  overlapping: the mirror DOUBLES AREA at the\n"
        f"  same quality and buys NO redundancy in the\n"
        f"  middle band.\n"
        f"  central band (back third of each web, the\n"
        f"  part nearest the seam):\n"
        f"      ≥1 arm {b1:.1f} %   ≥2 arms {b2:.1f} %   ≥3 arms 0.0 %\n"
        f"      — unchanged from the single unit.\n"
        f"\n"
        f"REACH-BOUND ONLY — NEVER SCORED\n"
        f"(would need a real sweep to become a number)\n"
        f"  unit-B arms within {RMAX:.2f} m of web A:\n"
        f"      17 → {lensA[17]} cells   71 → {lensA[71]}   "
        f"97 → {lensA[97]} ({lensA[97]*4e-4:.3f} m²)\n"
        f"  MERGED, the {r6.SEAM_M*100:.1f} cm seam adds\n"
        f"  {1.8034*r6.SEAM_M:.3f} m² of paper, of which "
        f"{100*(seam_cnt>=1).mean():.0f} % lies inside\n"
        f"  ≥1 arm's sweep radius and {100*(seam_cnt>=2).mean():.0f} % inside ≥2 — "
        f"and THAT\n"
        f"  overlap IS cross-unit (arms 2 and 97 reach\n"
        f"  ~34 % of it each).  It is the only place the\n"
        f"  mirrored layout could buy real redundancy.\n"
        f"\n"
        f"EXACTNESS, RE-DERIVED HERE\n"
        f"  {n_cells} unit-A strict-GO poses re-checked\n"
        f"  against BOTH frames (70 boxes, not 35):\n"
        f"  {n_lost} cells lost, worst clearance change\n"
        f"  {worst:.0e} m.  The second frame is invisible.\n"
        f"\n"
        f"ASSUMPTIONS — CONFIRM BEFORE DEEP RE-DERIVATION\n"
        f"  · frames abut with ZERO gap at W Y = "
        f"{r6.MIRROR_PLANE_W_CM} cm\n"
        f"    (canvas y {ym:.5f} m).  Foot pads overhang\n"
        f"    0.33 cm, so touching pads ⇒ 0.66 cm gap.\n"
        f"  · unit-B ids 17 / 71 / 97 assumed from the\n"
        f"    legacy six-arm registry — no drawing says.\n"
        f"  · this preview assumes TWO webs with a seam.\n"
        f"    MERGE_WEBS is now {r6.MERGE_WEBS}; the merged\n"
        f"    canvas is swept for real by run_atlas6.py.\n"
        f"  · unit-B base rotations realised as PROPER\n"
        f"    rotations S·R·S (a reflection is improper).\n"
    )
    axT.text(0.0, 1.0, txt, va="top", ha="left", family="monospace",
             fontsize=8.6, linespacing=1.40, transform=axT.transAxes)

    fig.suptitle("ARIS — the real installation: two FINAL-RIG units mirrored "
                 "back-to-back  ·  coverage preview by symmetry",
                 fontsize=14, fontweight="bold", x=0.055, ha="left", y=0.975)
    Path(png_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=145, facecolor="white")
    plt.close(fig)
    print(f"{png_path}\n  union strict-GO {u_pct:.2f}% of {tot} cells "
          f"(2 webs); >=2 arms {100*(cnt>=2).sum()/n_web:.2f}%, max {int(cnt.max())} arms; "
          f"cross-unit overlap 0.00%; second frame lost {n_lost} cells")
    return dict(union_pct=u_pct, overlap2_pct=100 * (cnt >= 2).sum() / n_web,
                max_arms=int(cnt.max()), cross_unit_pct=0.0, lost=n_lost)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", default=str(ROOT / "out/atlas_final6_preview.png"))
    ap.add_argument("--atlas", default=str(ROOT / "out/atlas_final"))
    a = ap.parse_args()
    build(a.png, a.atlas)
