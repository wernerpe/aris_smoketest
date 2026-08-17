#!/usr/bin/env python3
"""Compare strict-GO reach across pen-tilt allowances (out_tilt{0,15,30,45}/)."""
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
TILTS = [0, 15, 30, 45]

n_sheet = int(SHEET[0] / 0.02 + 1) * int(SHEET[1] / 0.02 + 1)
print(f"{'arm':>4} {'mount':>6} | " + " | ".join(f"tilt<={t:2d}: GO cells / r_out / r_in" for t in TILTS))
cov = {}
for aid, spec in FLEET.items():
    line = f"{aid:>4} {spec.mount:>6} |"
    for t in TILTS:
        a, _ = load(ROOT / f"out_tilt{t}", aid)
        go = strict_go(a)
        r = np.hypot(a[go, 0] - spec.xy[0], a[go, 1] - spec.xy[1])
        r_out = r.max() if go.any() else 0
        # inner hole radius: smallest r with a GO cell (inverted arms)
        r_in = r.min() if go.any() else np.nan
        line += f"  {int(go.sum()):5d} / {r_out:.3f} / {r_in:.3f} |"
        cov.setdefault(t, set()).update(
            (round(x, 3), round(y, 3)) for x, y in a[go][:, :2])
    print(line)
print("\nsheet strict-GO coverage: " + "  ".join(
    f"tilt<={t}: {100 * len(cov[t]) / n_sheet:.1f}%" for t in TILTS))

# figure: coverage + per-arm outer radius vs tilt
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(TILTS, [100 * len(cov[t]) / n_sheet for t in TILTS], "o-")
axes[0].set_xlabel("max pen tilt (deg)"), axes[0].set_ylabel("% sheet strict-GO")
axes[0].set_title("Coverage vs pen lean"), axes[0].grid(alpha=0.3)
for aid, spec in FLEET.items():
    r_outs = []
    for t in TILTS:
        a, _ = load(ROOT / f"out_tilt{t}", aid)
        go = strict_go(a)
        r_outs.append(np.hypot(a[go, 0] - spec.xy[0], a[go, 1] - spec.xy[1]).max())
    axes[1].plot(TILTS, r_outs, "o-", color=spec.color, label=f"{aid} {spec.mount}")
axes[1].set_xlabel("max pen tilt (deg)"), axes[1].set_ylabel("outer GO radius (m)")
axes[1].set_title("Reach vs pen lean"), axes[1].legend(fontsize=8), axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(ROOT / "out/tilt_study.png", dpi=140)
print("wrote out/tilt_study.png")
