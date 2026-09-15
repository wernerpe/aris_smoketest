#!/usr/bin/env python3
"""A conducted timeline -> a top-down GIF of the arms and the ink they lay.

WHY NOT A SCREEN RECORDING OF MESHCAT.  The meshcat scene is the rehearsal and
it is the right thing to stand in front of; it is also a 3-D perspective view
in a browser, which is the wrong thing to put in a document and impossible to
capture headless.  What a person actually needs to check before a hardware day
is a PLAN view: which arm is over which letter, when the two chains come near
each other, and whether the ink arrives in the order the programme says.  That
is two dimensions, and matplotlib draws it without a renderer.

Every chain point comes from `scene_check._chain` — the same function the
checker itself measures clearances with, scalar `fk` and all — so the picture
is drawn from the geometry that was certified rather than from a second model
that could drift from it.

    ARIS_RIG=proposed ARIS_TOOL=lateral python3 scripts/topdown_gif.py \\
        out/unknown_h0970_schedule.npz --out out/unknown_h0970_topdown.gif
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARM_COLORS = {31: "#1f77b4", 71: "#d62728", 13: "#2ca02c", 17: "#9467bd",
              2: "#8c564b", 97: "#e377c2"}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", type=int, nargs="*", default=None,
                    help="draw only these (default: every arm that moves)")
    ap.add_argument("--fps", type=float, default=12.0, help="GIF frame rate")
    ap.add_argument("--max-frames", type=int, default=240,
                    help="decimate to at most this many frames")
    ap.add_argument("--pad", type=float, default=0.25,
                    help="metres of paper to show around the ink")
    ap.add_argument("--full-sheet", action="store_true",
                    help="show the whole sheet instead of framing the ink")
    a = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    from aris_sixarm import scene_check
    from aris_sixarm.fleet import FLEET, SHEET

    z = np.load(a.npz, allow_pickle=False)
    arms = [int(v) for v in z["arms"]]
    Q = {x: np.asarray(z[f"q_{x}"], float) for x in arms}
    SEG = {x: np.asarray(z[f"seg_{x}"]).astype(np.int64) for x in arms}
    pens = {int(x): float(p) for x, p in zip(z["arms"], z["pen_ext"])}
    dt_frame = 1.0 / float(z["fps"])
    M = len(Q[arms[0]])
    # ONLY THE ARMS THAT MOVE, unless asked otherwise: a parked arm is a
    # stationary blob that hides the two that matter.
    movers = a.arms or [x for x in arms
                        if np.abs(np.diff(Q[x], axis=0)).max(initial=0.0) > 1e-9]
    if not movers:
        raise SystemExit(f"{a.npz}: no arm moves in this timeline")

    step = max(1, int(np.ceil(M / a.max_frames)))
    idx = np.arange(0, M, step)
    h_inv = float(z["h"]) if "h" in z.files else None
    print(f"{a.npz}: {M} frames, {len(arms)} arms, movers {movers}; "
          f"{len(idx)} GIF frames at every {step}th")

    # Every chain, once, up front -- 240 frames x 2 arms of scalar fk is
    # seconds, and doing it inside the draw callback would do it twice.
    chains = {x: np.stack([scene_check._chain(Q[x][i], FLEET[x], h_inv, pens[x])
                           for i in idx]) for x in movers}
    # `scene_check._chain` returns `frames.fk`'s 9 link points, then the TIP,
    # then -- only for a LATERAL tool, whose offset has a non-zero x -- the
    # holder corner.  So the tip is row 9 and the arm is rows 0..8, measured
    # here rather than assumed: `frames.tip_pos` agrees with row 9 to 0.0 m.
    n_fk = 9
    tips = {x: chains[x][:, n_fk] for x in movers}
    has_corner = {x: chains[x].shape[1] > n_fk + 1 for x in movers}
    down = {x: SEG[x][idx] >= 0 for x in movers}

    ink = []
    for k in range(len(idx)):
        for x in movers:
            if down[x][k]:
                ink.append((tips[x][k, 0], tips[x][k, 1], x, k))
    ink = np.array([(p[0], p[1], p[2], p[3]) for p in ink]) if ink \
        else np.zeros((0, 4))

    fig, ax = plt.subplots(figsize=(9.0, 4.2), dpi=130)
    ax.add_patch(plt.Rectangle((0, 0), SHEET[0], SHEET[1], fc="#fbfaf7",
                               ec="#bdb9b0", lw=1.0, zorder=0))
    if a.full_sheet or not len(ink):
        ax.set_xlim(-0.05, SHEET[0] + 0.05)
        ax.set_ylim(-0.05, SHEET[1] + 0.05)
    else:
        ax.set_xlim(ink[:, 0].min() - a.pad, ink[:, 0].max() + a.pad)
        ax.set_ylim(ink[:, 1].min() - a.pad, ink[:, 1].max() + a.pad)
    for x in movers:
        T = FLEET[x].T_world_base(h_inv)
        ax.plot([T[0, 3]], [T[1, 3]], "x", color=ARM_COLORS.get(x, "#333"),
                ms=11, mew=2.4, zorder=5)
        ax.annotate(f"arm {x}", (T[0, 3], T[1, 3]), fontsize=8, zorder=6,
                    textcoords="offset points", xytext=(8, 6),
                    color=ARM_COLORS.get(x, "#333"))
    ax.set_aspect("equal")
    ax.grid(True, lw=0.3, color="#e6e4e0")
    ax.set_xlabel("x  [m]")
    ax.set_ylabel("y  [m]")

    links = {x: ax.plot([], [], "-o", color=ARM_COLORS.get(x, "#333"), lw=2.0,
                        ms=3.0, alpha=0.85, zorder=4)[0] for x in movers}
    pens_ln = {x: ax.plot([], [], "-", color=ARM_COLORS.get(x, "#333"), lw=3.2,
                          alpha=0.95, zorder=5)[0] for x in movers}
    laid = {x: ax.plot([], [], ".", color=ARM_COLORS.get(x, "#333"), ms=2.2,
                       zorder=3)[0] for x in movers}
    title = ax.set_title("", fontsize=9)

    def draw(k):
        for x in movers:
            C = chains[x][k]
            links[x].set_data(C[:n_fk, 0], C[:n_fk, 1])
            # the holder, drawn as flange -> corner -> tip so the 86 mm the pen
            # reaches ACROSS the hand is visible; inline tools have no corner
            rows = ([n_fk - 1, n_fk + 1, n_fk] if has_corner[x]
                    else [n_fk - 1, n_fk])
            pens_ln[x].set_data(C[rows, 0], C[rows, 1])
            m = (ink[:, 2] == x) & (ink[:, 3] <= k) if len(ink) else None
            if m is not None:
                laid[x].set_data(ink[m, 0], ink[m, 1])
        pen = ", ".join(f"{x} {'DOWN' if down[x][k] else 'up  '}"
                        for x in movers)
        title.set_text(f"t = {idx[k] * dt_frame:6.2f} s   pen: {pen}")
        return (list(links.values()) + list(pens_ln.values())
                + list(laid.values()) + [title])

    anim = FuncAnimation(fig, draw, frames=len(idx), blit=False,
                         interval=1000.0 / a.fps)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    anim.save(a.out, writer=PillowWriter(fps=a.fps))
    plt.close(fig)
    print(f"wrote {a.out} ({Path(a.out).stat().st_size / 1e6:.2f} MB, "
          f"{len(idx)} frames at {a.fps:g} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
