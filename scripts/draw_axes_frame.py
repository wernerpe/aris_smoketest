#!/usr/bin/env python3
"""A one-page, minimal plan: the table outline, the canvas, and the six arm
rotation axes, dimensioned from the TABLE CENTRE.  Nothing else.

    python scripts/draw_axes_frame.py            -> out/drawings/axes_frame.{pdf,png}

Numbers (mm): table 2188 x 4165.6 (two butted half-frames, Pete's tape 416.6 cm);
canvas 1803.4 x 3630.6 centred on the table; axes at x = +-305.0 (610 apart) and
y = 0, +-1210.2 (the middle row on the seam line).
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

TABLE_W, TABLE_L = 2188.0, 4165.6
CANVAS_W, CANVAS_L = 1803.4, 3630.6
AX_X = (-305.0, 305.0)
AX_Y = (-1210.2, 0.0, 1210.2)
STRUT_X, STRUT_Y = 76.2, 152.4          # one hanging strut from above: 3 x 6 in
STRUT_FAR, STRUT_NEAR = 240.0, 156.0     # Pete's tape 2026-09-16: outside face -> axis
ARMS = {(-305.0, -1210.2): 13, (305.0, -1210.2): 17,
        (-305.0, 0.0): 31, (305.0, 0.0): 71,
        (-305.0, 1210.2): 2, (305.0, 1210.2): 97}


def main(out_dir="out/drawings"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11.7, 16.5))          # A3 portrait
    ax.set_aspect("equal")
    ax.add_patch(Rectangle((-TABLE_W / 2, -TABLE_L / 2), TABLE_W, TABLE_L,
                           fill=False, lw=2.5, ec="black"))
    ax.add_patch(Rectangle((-CANVAS_W / 2, -CANVAS_L / 2), CANVAS_W, CANVAS_L,
                           fill=False, lw=1.2, ec="gray", ls="--"))
    ax.axvline(0, color="gray", lw=0.8, ls=":")
    # runway / strut centrelines: one per row, horizontal, dashed
    for y in AX_Y:
        ax.plot([-TABLE_W / 2, TABLE_W / 2], [y, y], color="black", lw=1.0, ls="--")
    # hanging struts seen from above: 76.2 (x) x 152.4 (y) each = 3 x 6 in,
    # placed from Pete's tape: outside face -> axis 240 on one side, 156 on the other
    for (x, y), arm in ARMS.items():
        for x0 in (x - STRUT_FAR, x + STRUT_NEAR - STRUT_X):
            ax.add_patch(Rectangle((x0, y - STRUT_Y / 2), STRUT_X, STRUT_Y,
                                   fill=True, fc="#d9d9d9", ec="black", lw=1.2))
    for (x, y), arm in ARMS.items():
        ax.plot([x - 90, x + 90], [y, y], color="crimson", lw=2)
        ax.plot([x, x], [y - 90, y + 90], color="crimson", lw=2)
        ax.add_patch(plt.Circle((x, y), 95, fill=False, ec="crimson", lw=1))
        ax.text(x + 110, y + 110, f"arm {arm}\n({x:+.0f}, {y:+.0f})",
                fontsize=13, color="crimson", va="bottom")
    # dimensions
    def dim(x0, y0, x1, y1, text, off=(0, 0)):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="<->", lw=1.5, color="navy"))
        ax.text((x0 + x1) / 2 + off[0], (y0 + y1) / 2 + off[1], text,
                fontsize=14, color="navy", ha="center", va="center",
                bbox=dict(fc="white", ec="none", pad=1))
    dim(-305, 1700, 305, 1700, "610.0", off=(0, 60))
    # one strut pair dimensioned (top-left arm): far / near / outer width
    xa, ya = -305.0, 1210.2
    dim(xa - STRUT_FAR, ya - 200, xa, ya - 200, f"{STRUT_FAR:.0f}", off=(0, -60))
    dim(xa, ya - 200, xa + STRUT_NEAR, ya - 200, f"{STRUT_NEAR:.0f}", off=(0, -60))
    dim(xa - STRUT_FAR, ya - 330, xa + STRUT_NEAR, ya - 330,
        f"{STRUT_FAR + STRUT_NEAR:.0f} outside to outside", off=(0, -60))
    ax.text(xa - STRUT_FAR, ya + 120, f"strut {STRUT_X} x {STRUT_Y} (3 x 6 in)",
            fontsize=12, color="black")
    dim(-700, 0, -700, 1210.2, "1210.2", off=(-120, 0))
    dim(-700, -1210.2, -700, 0, "1210.2", off=(-120, 0))
    dim(-TABLE_W / 2, -2250, TABLE_W / 2, -2250, f"table {TABLE_W:.0f}", off=(0, -70))
    dim(1400, -TABLE_L / 2, 1400, TABLE_L / 2, f"table {TABLE_L:.1f}", off=(0, 250))
    dim(-CANVAS_W / 2, -1950, CANVAS_W / 2, -1950, f"canvas {CANVAS_W}", off=(0, -70))
    dim(1000, -CANVAS_L / 2, 1000, CANVAS_L / 2, f"canvas {CANVAS_L}", off=(0, -250))
    ax.text(0, TABLE_L / 2 + 120, "origin (0,0) = table centre: seam line x long centre line. "
            "x across, y along. mm.", ha="center", fontsize=13)
    ax.set_xlim(-1500, 1500)
    ax.set_ylim(-2450, 2350)
    ax.axis("off")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"axes_frame.{ext}", dpi=150)
    return out / "axes_frame.pdf"


if __name__ == "__main__":
    print(main())
