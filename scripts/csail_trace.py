#!/usr/bin/env python3
"""Trace the old MIT CSAIL logo into pen strokes on the paper.

    python3 scripts/csail_trace.py [--image PATH] [--upscale 4] [--margin 0.06]

Writes
    out/csail_trace.png    the traced stroke set at sheet scale — LOOK AT THIS
    out/csail_masks.png    debug: colour masks, skeletons, trace over the source
    out/csail_strokes.json the stroke set in metres (input to the allocator)

Runs under the SYSTEM python3: numpy + matplotlib + PIL only, no IK, no drake.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from PIL import Image                    # noqa: E402

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from aris_sixarm import trace            # noqa: E402
from aris_sixarm.fleet import FLEET, SHEET  # noqa: E402

INK = {"grey": "#7f8184", "orange": "#c8651b"}


def sheet_axes(ax, title=""):
    ax.add_patch(plt.Rectangle((0, 0), *SHEET, facecolor="#fdfdf7",
                               edgecolor="#333333", lw=1.4, zorder=0))
    ax.set_aspect("equal")
    ax.set_xlim(-0.12, SHEET[0] + 0.12)
    ax.set_ylim(-0.12, SHEET[1] + 0.12)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=10)


def draw_strokes(ax, strokes, lw=2.6, alpha=1.0, ends=False):
    for s in strokes:
        p = np.asarray(s["pts"], float)
        ax.plot(p[:, 0], p[:, 1], "-", color=INK[s["color"]], lw=lw,
                alpha=alpha, solid_capstyle="round", solid_joinstyle="round",
                zorder=3)
        if ends:
            ax.plot(p[0, 0], p[0, 1], ".", color="#111111", ms=4, zorder=4)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=str(ROOT / "assets/csail/csail_old_med.gif"))
    ap.add_argument("--upscale", type=int, default=trace.UPSCALE)
    ap.add_argument("--margin", type=float, default=0.06)
    ap.add_argument("--rdp", type=float, default=trace.RDP_TOL)
    ap.add_argument("--out", default=str(ROOT / "out"))
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    px, dbg = trace.trace_logo(a.image, upscale=a.upscale, rdp_tol=a.rdp)
    strokes, info = trace.to_sheet(px, SHEET, margin=a.margin)
    dt = time.time() - t0

    L = trace.total_length(strokes)
    n_grey = sum(1 for s in strokes if s["color"] == "grey")
    n_let = sum(1 for s in strokes if s["kind"] == "letter")
    print(f"traced {len(strokes)} strokes ({n_grey} grey, "
          f"{len(strokes) - n_grey} orange, of which {n_let} letter contours), "
          f"{L:.2f} m of path, {sum(len(s['pts']) for s in strokes)} points, "
          f"in {dt:.2f} s")
    print(f"logo {info['logo_w']:.3f} x {info['logo_h']:.3f} m on a "
          f"{SHEET[0]:.3f} x {SHEET[1]:.3f} m sheet "
          f"({info['px_per_m']:.0f} px/m at {a.upscale}x, margin {a.margin} m)")
    if info["logo_w"] < 3.2:
        print(f"  note: the 3.3 m width asked for needs "
              f"{3.3 * info['logo_h'] / info['logo_w']:.2f} m of height at this "
              f"aspect; the sheet has {SHEET[1]:.3f}, so HEIGHT binds and the "
              f"logo is as big as it can be without distorting it.")

    # ---- out/csail_trace.png -------------------------------------------
    fig, ax = plt.subplots(figsize=(13.0, 7.4))
    sheet_axes(ax)
    draw_strokes(ax, strokes)
    for aid, spec in FLEET.items():
        ax.plot(*spec.xy, marker="o" if spec.mount == "floor" else "s",
                ms=12, mfc=spec.color if spec.active else "white",
                mec=spec.color, mew=2.0, zorder=6, clip_on=False)
        ax.annotate(str(aid), spec.xy, color="white" if spec.active else spec.color,
                    fontsize=7, fontweight="bold", ha="center", va="center",
                    zorder=7, clip_on=False)
    ax.set_title(f"CSAIL logo traced to {len(strokes)} pen strokes, {L:.1f} m of "
                 f"path — {info['logo_w']:.2f} x {info['logo_h']:.2f} m on the "
                 f"{SHEET[0]:.3f} x {SHEET[1]:.3f} m sheet", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "csail_trace.png", dpi=130)
    plt.close(fig)

    # ---- out/csail_masks.png -------------------------------------------
    src = np.asarray(Image.open(a.image).convert("RGB")
                     .resize((dbg["shape"][1], dbg["shape"][0]), Image.LANCZOS))
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    ax[0, 0].imshow(src)
    ax[0, 0].set_title(f"source, upsampled {a.upscale}x")
    m = dbg["masks"]
    rgb = np.ones((*dbg["shape"], 3))
    rgb[m["grey"]] = [0.49, 0.51, 0.52]
    rgb[dbg["orange_lines"]] = [0.78, 0.40, 0.11]
    rgb[dbg["letters"]] = [0.95, 0.65, 0.30]
    ax[0, 1].imshow(rgb)
    ax[0, 1].set_title("unmixed masks: grey / orange lines / solid letters")
    sk = np.ones((*dbg["shape"], 3))
    sk[dbg["skel_grey"]] = [0.20, 0.20, 0.22]
    sk[dbg["skel_orange"]] = [0.78, 0.40, 0.11]
    ax[1, 0].imshow(sk)
    ax[1, 0].set_title("Zhang-Suen skeletons (letters excluded)")
    ax[1, 1].imshow(src, alpha=0.35)
    for s in px:
        p = s["pts"]
        ax[1, 1].plot(p[:, 0], p[:, 1], "-", lw=1.3,
                      color="#1a9850" if s["kind"] == "outline" else "#0570b0")
        ax[1, 1].plot(p[0, 0], p[0, 1], ".", ms=4, color="k")
    ax[1, 1].set_title(f"traced polylines over the source ({len(px)} strokes; "
                       "dots = stroke starts)")
    for x in ax.ravel():
        x.set_xticks([])
        x.set_yticks([])
    fig.tight_layout()
    fig.savefig(out / "csail_masks.png", dpi=100)
    plt.close(fig)

    with open(out / "csail_strokes.json", "w") as f:
        json.dump(dict(sheet=list(SHEET), info={k: float(v) for k, v in info.items()},
                       n_strokes=len(strokes), total_length=L,
                       strokes=[dict(id=s["id"], color=s["color"], kind=s["kind"],
                                     length=trace.plen(s["pts"]),
                                     pts=np.round(s["pts"], 5).tolist())
                                for s in strokes]), f)
    print(f"wrote {out}/csail_trace.png, {out}/csail_masks.png, "
          f"{out}/csail_strokes.json")
    return strokes, info


if __name__ == "__main__":
    main()
