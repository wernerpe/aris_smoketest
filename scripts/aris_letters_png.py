#!/usr/bin/env python3
"""Layout figure for the ARIS writing demo: out/aris_letters.png

    python3 scripts/aris_letters_png.py

Runs under the SYSTEM python3 (numpy + matplotlib); it needs no drake.  If
out/aris_plan.json exists (written by scripts/aris_writing_demo.py) the actual
PLANNED stroke geometry is drawn, so any placement nudge the planner had to
make shows up here; otherwise the nominal letterforms are drawn.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from aris_sixarm import letters  # noqa: E402
from aris_sixarm.fleet import FLEET, SHEET  # noqa: E402
from aris_sixarm.writing import COMFORT_R  # noqa: E402

PLAN = ROOT / "out/aris_plan.json"
OUT = ROOT / "out/aris_letters.png"


def load():
    """-> list of (name, arm_id, [strokes], nudge_tag, nominal_centre)."""
    nominal = {n: c for n, _, c in letters.PLACEMENT}
    if PLAN.exists():
        with open(PLAN) as f:
            plan = json.load(f)
        return [(L["name"], L["arm_id"],
                 [np.asarray(S["pts"], float) for S in L["strokes"]],
                 L["nudge"], nominal[L["name"]]) for L in plan], True
    return [(n, a, letters.place(n, c), "nominal (unplanned)", c)
            for n, a, c in letters.PLACEMENT], False


def main():
    data, planned = load()
    fig, ax = plt.subplots(figsize=(12.4, 7.4))

    ax.add_patch(plt.Rectangle((0, 0), *SHEET, facecolor="#fdfdf7",
                               edgecolor="#333333", lw=1.6, zorder=0))
    ax.text(0.02, SHEET[1] - 0.06, f"paper  {SHEET[0]:.3f} x {SHEET[1]:.3f} m "
            "(z = 0, world origin at the corner)", fontsize=8, color="#555555",
            va="top")

    for aid, spec in FLEET.items():
        bx, by = spec.xy
        c = spec.color
        on = spec.active
        ax.plot([bx], [by], marker="o" if spec.mount == "floor" else "s",
                ms=15, mfc=c if on else "white", mec=c,
                mew=2.2, alpha=1.0 if on else 0.55, zorder=6)
        ax.annotate(f"{aid}", (bx, by), color="white" if on else c, fontsize=8,
                    fontweight="bold", ha="center", va="center", zorder=7)
        ax.annotate(f"{spec.mount}{'' if on else ' (parked)'}", (bx, by - 0.13),
                    color=c, fontsize=7.5, ha="center", va="top", zorder=7,
                    alpha=1.0 if on else 0.6)
        if on:
            ax.add_patch(plt.Circle((bx, by), COMFORT_R, fill=False, ls=(0, (4, 4)),
                                    lw=0.9, ec=c, alpha=0.35, zorder=1))

    for name, aid, strokes, nudge, nom in data:
        c = FLEET[aid].color
        for k, pts in enumerate(strokes):
            ax.plot(pts[:, 0], pts[:, 1], "-", color=c, lw=3.4, zorder=4,
                    solid_capstyle="round")
            # pen-down / pen-up: the transit endpoints of every stroke
            ax.plot(pts[0, 0], pts[0, 1], "o", color=c, ms=7, mec="black",
                    mew=1.0, zorder=5)
            ax.plot(pts[-1, 0], pts[-1, 1], "s", color="white", ms=7, mec=c,
                    mew=2.0, zorder=5)
            mid = pts[len(pts) // 2]
            ax.annotate(f"{name}{k + 1}", mid + np.array([0.035, 0.035]),
                        fontsize=7, color=c, zorder=8,
                        bbox=dict(boxstyle="round,pad=0.12", fc="white",
                                  ec="none", alpha=0.75))
        ctr = np.mean([p.mean(axis=0) for p in strokes], axis=0)
        # a nudged letter: show where it was asked to go
        if nudge != "nominal" and np.linalg.norm(np.array(nom) - ctr) > 1e-3:
            ax.annotate("", xy=tuple(ctr), xytext=tuple(nom), zorder=3,
                        arrowprops=dict(arrowstyle="->", color="#888888",
                                        lw=1.2, ls=":"))
            ax.plot(*nom, "x", color="#888888", ms=8, mew=1.6, zorder=3)
            d = np.linalg.norm(np.array(nom) - ctr)
            ax.annotate(f"nudged {d:.2f} m\ntoward base", (nom[0] - 0.06, nom[1]),
                        fontsize=6.5, color="#777777", ha="right", va="center",
                        zorder=8, bbox=dict(boxstyle="round,pad=0.18", fc="white",
                                            ec="#cccccc", lw=0.5, alpha=0.9))
        ymax = max(p[:, 1].max() for p in strokes)      # clear the glyph itself
        ax.annotate(f"'{name}' → arm {aid}", (ctr[0], ymax + 0.075),
                    fontsize=10, fontweight="bold", color=c, ha="center", zorder=5)

    handles = [
        Line2D([], [], color="#444444", lw=3.4, label="planned stroke (pen down)"),
        Line2D([], [], marker="o", color="#444444", ls="", ms=7, mec="black",
               label="stroke start (pen down)"),
        Line2D([], [], marker="s", color="white", ls="", ms=7, mec="#444444",
               mew=2, label="stroke end (pen up / lift)"),
        Line2D([], [], marker="o", color="#444444", ls="", ms=11,
               label="floor-mounted base"),
        Line2D([], [], marker="s", color="#444444", ls="", ms=11,
               label="inverted base (h = 1.00 m)"),
        Line2D([], [], color="#888888", lw=0.9, ls=(0, (4, 4)),
               label=f"comfortable radius r = {COMFORT_R:.2f} m"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.94,
              ncol=3, bbox_to_anchor=(0.005, 0.005))
    ax.set_xlim(-0.42, SHEET[0] + 0.42)
    ax.set_ylim(-0.30, SHEET[1] + 0.34)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("ARIS letter layout — stroke allocation over the six-arm fleet"
                 + ("  (planned geometry)" if planned else "  (nominal geometry)"),
                 fontsize=12)
    ax.grid(alpha=0.15, lw=0.6)
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=150)
    print(f"wrote {OUT} ({'planned' if planned else 'nominal'} geometry, "
          f"{sum(len(s) for _, _, s, _, _ in data)} strokes)")


if __name__ == "__main__":
    main()
