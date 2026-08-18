#!/usr/bin/env python3
"""Compare strict-GO reach across pen lengths (out_pen{110,200,300}/)."""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parents[1]))
from aris_sixarm.atlas import load, strict_go  # noqa: E402
from aris_sixarm.fleet import FLEET, SHEET  # noqa: E402

ROOT = Path(__file__).parents[1]
PENS = [110, 200, 300]

n_sheet = int(SHEET[0] / 0.02 + 1) * int(SHEET[1] / 0.02 + 1)
print(f"{'arm':>4} {'mount':>6} | " + " | ".join(
    f"pen {p}mm: GO / r_out / r_in / med-sigma" for p in PENS))
cov = {}
for aid, spec in FLEET.items():
    line = f"{aid:>4} {spec.mount:>6} |"
    for p in PENS:
        a, _ = load(ROOT / f"out_pen{p}", aid)
        go = strict_go(a)
        r = np.hypot(a[go, 0] - spec.xy[0], a[go, 1] - spec.xy[1])
        line += (f"  {int(go.sum()):5d} / {r.max():.3f} / {r.min():.3f} / "
                 f"{np.median(a[go, 3]):.3f} |")
        cov.setdefault(p, set()).update(
            (round(x, 3), round(y, 3)) for x, y in a[go][:, :2])
    print(line)
print("\nsheet strict-GO coverage: " + "  ".join(
    f"pen {p}mm: {100 * len(cov[p]) / n_sheet:.1f}%" for p in PENS))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(PENS, [100 * len(cov[p]) / n_sheet for p in PENS], "o-")
axes[0].set_xlabel("pen tip below TCP (mm)"), axes[0].set_ylabel("% sheet strict-GO")
axes[0].set_title("Coverage vs pen length (tilt<=15deg)"), axes[0].grid(alpha=0.3)
for aid, spec in FLEET.items():
    r_outs = []
    for p in PENS:
        a, _ = load(ROOT / f"out_pen{p}", aid)
        go = strict_go(a)
        r_outs.append(np.hypot(a[go, 0] - spec.xy[0], a[go, 1] - spec.xy[1]).max())
    axes[1].plot(PENS, r_outs, "o-", color=spec.color, label=f"{aid} {spec.mount}")
axes[1].set_xlabel("pen tip below TCP (mm)"), axes[1].set_ylabel("outer GO radius (m)")
axes[1].set_title("Reach vs pen length"), axes[1].legend(fontsize=8), axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(ROOT / "out/pen_study.png", dpi=140)
print("wrote out/pen_study.png")
