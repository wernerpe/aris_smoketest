#!/usr/bin/env python3
"""Slide arm 2 up/down its boom pole and ask what the paper gains.

The final rig's biggest dead region sits under and to the right of the
SIDE-mounted arm 2 (canvas base (1.60984, 1.25723, 0.776), roty(-pi/2)rotz(pi),
J1 horizontal along -X, front facing straight down).  Arm 2 hangs off a
vertical T-slot pole; the only free parameter the fabricator can change
without touching anything else is HOW FAR DOWN THE POLE the plate is clamped.

This script sweeps that one parameter.

WHAT MOVES WITH THE ARM
  The pole (`side_boom`) and its top bracing (`side_gusset`) are structure:
  they stay.  The clamped stack — `side_plate`, `side_clamps`,
  `side_bracket` — slides with the arm, so every OTHER arm's collision model
  stays honest (those boxes are excluded only from arm 2's own check, see
  fleet.ArmSpec.static_obstacles / rig_final.frame_boxes_canvas).

FEASIBLE RANGE (see `feasibility()` and docs/FINAL_RIG.md appendix)
  UP    +65.19 cm, to bracket top = the central top beam underside 226.03.
        The -X mounting face is empty the whole way (DXF-swept).
  DOWN  +2.70 cm at best, 0.00 cm on the printed reading — the plate top is
        already flush with the pole's own bottom end.  The arm hangs OFF THE
        END of the pole.  Going lower is not a slide, it is a longer pole.

USAGE
  scripts/arm2_height_sweep.py --neighbours 999    # the run behind the figure
  scripts/arm2_height_sweep.py --dz -8 -4 0 4 8    # explicit offsets, cm
  scripts/arm2_height_sweep.py --replot            # redraw from cached sweeps
"""
import argparse
import json
import multiprocessing as mp
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from aris_sixarm import rig_final                      # noqa: E402
from aris_sixarm.atlas import strict_go, sweep_arm     # noqa: E402
from aris_sixarm.fleet import FLEET, SHEET             # noqa: E402

ROOT = Path(__file__).parents[1]
GRID = 0.02
Z0_CANVAS = 0.776                 # arm-2 J1 axis, canvas m (= 141.268 cm W)

# the hardware that is CLAMPED to the pole and therefore slides with the arm.
# `side_boom` (the pole) and `side_gusset` (its top brace) do NOT slide.
SLIDING = ("side_plate", "side_clamps", "side_bracket")

# --- what the pole allows -------------------------------------------------
# Independently re-derived from the binary DXF (ezdxf + ACIS decode of all 646
# 3DSOLIDs; the pole beams are the ONLY 4 solids of 315 where the drawing's two
# model copies disagree -- docs/FINAL_RIG.md Flags #1):
#
#   pole = 2 x 3"x3" alu profile stacked in Y, X 187.31-194.95, Y 144.83-160.07
#     FRONT copy (= what the PDF prints)   z 155.0675 .. 226.0675  (71.00 cm)
#     TOP copy   (= same part as the four  z 152.3675 .. 226.0275  (73.66 cm
#                   arm-31 boom beams)                              = 29.00 in)
#   ceiling: central double top beam underside      z 226.0275
#   the arm's stack, all four pieces confirmed in BOTH copies:
#     base plate            z 132.4852 .. 155.0675   (1.27 thick)
#     lower spacer block    z 132.4852 .. 139.6165   (3.81, hangs in AIR)
#     upper spacer block    z 147.9362 .. 155.0675   (3.81)
#     BRACKET, 2 halves     z 155.0675 .. 160.8338   <- the only piece that
#                                                       grips the pole
# The plate top is EXACTLY flush with the pole's bottom end in the front copy.
# That is the drawing's own warning made geometric: the arm hangs off the end
# of the pole on one 5.77 cm bracket, with two spacer blocks bolted to nothing.
POLE_Z_FRONT = (155.0675, 226.0675)
POLE_Z_TOP = (152.3675, 226.0275)
POLE_Z = (152.37, 226.07)         # the union, = rig_final's side_boom box
CEIL_Z = 226.0275                 # central double top beam underside, W cm
BRACKET_Z = (155.0675, 160.8338)  # the gripping piece; must stay fully on
STACK_Z = (132.4852, 160.8338)    # plate bottom .. bracket top

# assumed pole bottom.  TOP-copy reading = the union already in rig_final, and
# the more generous of the two: it buys 2.70 cm of downward travel where the
# printed front view buys none.  Flip to POLE_Z_FRONT for the strict reading.
POLE_BOTTOM = POLE_Z_TOP[0]
DZ_DOWN = BRACKET_Z[0] - POLE_BOTTOM              # 2.70 cm (0.00 front copy)
DZ_UP = CEIL_Z - BRACKET_Z[1]                     # 65.19 cm


def feasibility(dz_cm):
    """Can the plate be clamped here WITHOUT changing the pole?

    -> (bracket_engagement_cm, ok, why).  dz_cm > 0 = slide UP.
    The bracket must sit fully on the pole; above, the bracket top must clear
    the central double top beam.  Everything between 160.83 and 226.03 on the
    -X mounting face is empty (swept in the DXF), so the up range is solid.
    """
    lo, hi = BRACKET_Z[0] + dz_cm, BRACKET_Z[1] + dz_cm
    eng = min(hi, POLE_Z[1]) - max(lo, POLE_BOTTOM)
    if lo < POLE_BOTTOM - 1e-9:
        return eng, False, (f"bracket bottom {lo:.2f} is below the pole's own "
                            f"end {POLE_BOTTOM:.2f} - nothing to clamp to")
    if hi > CEIL_Z:
        return eng, False, f"bracket top {hi:.2f} > top beam {CEIL_Z}"
    return eng, True, ""


# --- the modified rig -----------------------------------------------------
def shifted_boxes(dz_cm):
    """FRAME_BOXES_W_CM with arm 2's CLAMPED hardware slid by dz_cm."""
    out = []
    for b in rig_final.FRAME_BOXES_W_CM:
        if b["name"] in SLIDING:
            b = dict(b, lo=(b["lo"][0], b["lo"][1], b["lo"][2] + dz_cm),
                     hi=(b["hi"][0], b["hi"][1], b["hi"][2] + dz_cm),
                     source=b["source"] + f" [slid {dz_cm:+.1f} cm]")
        out.append(b)
    return out


def shifted_fleet(dz_cm):
    """FLEET with arm 2's base z shifted; xy and R untouched."""
    f = dict(FLEET)
    f[2] = replace(f[2], z=Z0_CANVAS + dz_cm / 100.0)
    return f


def _mask(arr, shape):
    """(reach, go) boolean sheet masks from an atlas row array."""
    reach = np.zeros(shape, bool)
    go = np.zeros(shape, bool)
    if len(arr):
        ix = np.rint(arr[:, 0] / GRID).astype(int)
        iy = np.rint(arr[:, 1] / GRID).astype(int)
        reach[iy, ix] = True
        go[iy, ix] = strict_go(arr)
    return reach, go


def _tag_dir(out_dir, dz_cm):
    return Path(out_dir) / f"dz{dz_cm:+05.1f}".replace(".", "p")


def run_one(job):
    """(dz_cm, arm_ids, out_dir) -> {arm_id: rows}.  Runs in a fork worker, so
    the module patch is private to this process."""
    dz_cm, arms, out_dir = job
    rig_final.FRAME_BOXES_W_CM = shifted_boxes(dz_cm)
    fleet = shifted_fleet(dz_cm)
    d = _tag_dir(out_dir, dz_cm)
    return dz_cm, {a: sweep_arm(a, d, grid=GRID, fleet=fleet) for a in arms}


# --- metrics --------------------------------------------------------------
def score(go2, base, engage):
    """Per-height numbers, against the baseline coverage masks."""
    dead0, pg = base["dead0"], base["pg_other"]
    n = dead0.size
    union = pg[0] | pg[1] | go2                 # arms 13, 31 fixed + new arm 2
    cnt = pg[0].astype(int) + pg[1] + go2
    recovered = dead0 & go2
    lost = base["ug0"] & ~union                 # was covered, now nobody
    return dict(
        arm2_go=int(go2.sum()), arm2_go_m2=round(go2.sum() * GRID * GRID, 4),
        arm2_go_pct=round(100 * go2.mean(), 2),
        recovered=int(recovered.sum()),
        recovered_m2=round(recovered.sum() * GRID * GRID, 4),
        union_go_pct=round(100 * union.mean(), 2),
        union_go=int(union.sum()),
        overlap2_pct=round(100 * (cnt >= 2).mean(), 2),
        new_dead=int(lost.sum()),
        new_dead_m2=round(lost.sum() * GRID * GRID, 4),
        dead_pct=round(100 * (~union).mean(), 2),
        engage_cm=round(engage, 2), cells=n)


def shoulder_xy(spec):
    """Canvas xy of the arm's J2 (shoulder) axis — the centre of its
    under-base hole.  For arm 2 the J1 axis is horizontal, so the shoulder
    sits 0.333 m in -X of the plate, not under it; sliding the base along z
    does not move it in xy, which is why that hole never goes away."""
    from aris_sixarm.frames import fk
    T = spec.T_world_base()
    p = fk(np.zeros(7))[1][1]                     # J2 origin in the base frame
    return (T[:3, :3] @ p + T[:3, 3])[:2]


def why_dead(mask, base, r_hole=0.34):
    """Attribute dead cells to the structure / singularity that owns them.

    Categories are tried in order and each cell is claimed once, so the
    numbers add up to the total."""
    X, Y = np.meshgrid(base["xs"], base["ys"])
    n = max(int(mask.sum()), 1)
    sh = {a: shoulder_xy(FLEET[a]) for a in (13, 31, 2)}
    order = [
        ("left strip x<=0.12: feed roll + arm-31 boom", X <= 0.12),
        ("right strip x>=1.66: guide rods + rising paper curl", X >= 1.66),
        ("arm-2 under-shoulder hole (wrist fold, not reach)",
         np.hypot(X - sh[2][0], Y - sh[2][1]) <= r_hole),
        ("arm-31 under-shoulder hole", np.hypot(X - sh[31][0],
                                                Y - sh[31][1]) <= r_hole),
        ("arm-13 under-shoulder arc", np.hypot(X - sh[13][0],
                                               Y - sh[13][1]) <= r_hole + 0.06),
        ("front edge y<=0.06", Y <= 0.06),
        ("back edge y>=1.62", Y >= 1.62),
    ]
    out, left = {}, mask
    for k, m in order:
        hit = left & m
        out[k] = dict(cells=int(hit.sum()),
                      pct_of_dead=round(100 * hit.sum() / n, 1))
        left = left & ~m
    out["everything else (mid-sheet, between the three lobes)"] = dict(
        cells=int(left.sum()), pct_of_dead=round(100 * left.sum() / n, 1))
    out["_shoulders_xy"] = {str(k): [round(float(v[0]), 3),
                                     round(float(v[1]), 3)] for k, v in sh.items()}
    return out


def load_baseline(atlas_dir):
    d = np.load(Path(atlas_dir) / "coverage.npz")
    arms = list(d["arms"])
    pg = d["per_arm_go"]
    other = np.stack([pg[arms.index(13)], pg[arms.index(31)]])
    return dict(xs=d["xs"], ys=d["ys"], ug0=d["union_go"],
                cg0=d["count_go"], dead0=~d["union_go"],
                go2_0=pg[arms.index(2)], pg_other=other,
                shape=d["union_go"].shape)


# --- figure ---------------------------------------------------------------
def figure(rows, base, best, masks, path, best_feas):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    xs, ys = base["xs"], base["ys"]
    ext = [xs[0] - GRID / 2, xs[-1] + GRID / 2, ys[0] - GRID / 2, ys[-1] + GRID / 2]
    zc = np.array([r["z_canvas"] for r in rows])
    ok = np.array([r["feasible"] for r in rows])

    fig = plt.figure(figsize=(15.5, 8.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05], hspace=0.30,
                          wspace=0.22, left=0.055, right=0.985,
                          top=0.905, bottom=0.075)

    # ---- (a) coverage vs z ----
    ax = fig.add_subplot(gs[0, :2])
    for key, lab, c in (("union_go_pct", "union strict-GO (3 arms)", "#1f77b4"),
                        ("arm2_go_pct", "arm 2 strict-GO", "#2ca02c"),
                        ("overlap2_pct", "cells GO by $\\geq$2 arms", "#9467bd")):
        v = np.array([r[key] for r in rows])
        ax.plot(zc, v, "-o", ms=4, color=c, label=lab)
    if (~ok).any():
        for lo, hi in _spans(zc, ~ok):
            ax.axvspan(lo, hi, color="0.88", zorder=0)
    ax.set_ylim(0, 88)
    ax.axvline(Z0_CANVAS, color="k", lw=1.2, ls="--")
    ax.annotate("as drawn 0.776", (Z0_CANVAS, 86), xytext=(4, 0),
                textcoords="offset points", fontsize=8, va="top", ha="left")
    ax.axvline(best_feas["z_canvas"], color="#555", lw=1.3)
    ax.annotate(f"{best_feas['z_canvas']:.3f} = as low as the\n"
                f"drawn pole allows ({best_feas['dz_cm']:+.0f} cm)",
                (best_feas["z_canvas"], 86), xytext=(-5, 0),
                textcoords="offset points", fontsize=8, color="#333",
                va="top", ha="right")
    ax.axvline(best["z_canvas"], color="#d62728", lw=1.8)
    ax.annotate(f"optimum {best['z_canvas']:.3f}\n"
                f"needs {best['pole_extension_cm']:.0f} cm more pole",
                (best["z_canvas"], 62), xytext=(6, 0),
                textcoords="offset points", fontsize=8.5, weight="bold",
                color="#d62728", va="center", ha="left")
    ax.set_xlabel("arm-2 J1 axis height above the paper, canvas z (m)  "
                  "$\\leftarrow$ lower on the pole")
    ax.set_ylabel("% of the 1.8034 $\\times$ 1.700 m sheet")
    ax.set_title("(a) coverage vs. how far down the pole arm 2 is clamped\n"
                 "grey = the bracket runs off the pole's own bottom end "
                 f"(z$_W$ {POLE_BOTTOM:.2f} cm): reachable only by lengthening "
                 "the pole", fontsize=10, loc="left")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")

    # ---- (b) recovered / lost ----
    ax = fig.add_subplot(gs[0, 2])
    rec = np.array([r["recovered"] for r in rows])
    lost = np.array([r["new_dead"] for r in rows])
    ax.plot(zc, rec, "-o", ms=4, color="#2ca02c", label="dead cells recovered")
    ax.plot(zc, -lost, "-o", ms=4, color="#d62728", label="cells newly dead")
    if (~ok).any():
        for lo, hi in _spans(zc, ~ok):
            ax.axvspan(lo, hi, color="0.88", zorder=0)
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(Z0_CANVAS, color="k", lw=1.2, ls="--")
    ax.axvline(best["z_canvas"], color="#d62728", lw=1.4)
    ax.set_xlabel("canvas z (m)")
    ax.set_ylabel("cells (2 cm grid, 4 cm$^2$ each)")
    ax.set_title("(b) net cells won / lost", fontsize=10, loc="left")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # ---- (c)(d)(e) before / after / delta maps ----
    def paint(ax, cnt, title):
        img = np.zeros(cnt.shape + (3,))
        img[...] = 0.16                      # dead = near-black
        img[cnt == 1] = (0.55, 0.72, 0.86)
        img[cnt == 2] = (0.15, 0.45, 0.72)
        img[cnt >= 3] = (0.05, 0.25, 0.45)
        ax.imshow(img, origin="lower", extent=ext, interpolation="nearest")
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("canvas x (m)")
        ax.set_aspect("equal")
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])

    ax = fig.add_subplot(gs[1, 0])
    paint(ax, base["cg0"], f"(c) as drawn, z = {Z0_CANVAS:.3f}\n"
          f"union GO {100*base['ug0'].mean():.1f} %, "
          f"$\\geq$2 arms {100*(base['cg0']>=2).mean():.1f} %")
    ax.set_ylabel("canvas y (m)")
    _arm_marks(ax)

    ax = fig.add_subplot(gs[1, 1])
    paint(ax, masks["cnt"], f"(d) optimum, z = {best['z_canvas']:.3f} "
          f"({best['dz_cm']:+.0f} cm on the pole)\n"
          f"union GO {best['union_go_pct']:.1f} %, "
          f"$\\geq$2 arms {best['overlap2_pct']:.1f} %")
    _arm_marks(ax)

    ax = fig.add_subplot(gs[1, 2])
    img = np.zeros(base["cg0"].shape + (3,))
    img[...] = 0.93
    img[~base["ug0"] & ~masks["union"]] = (0.16, 0.16, 0.16)
    img[masks["recovered"]] = (0.13, 0.65, 0.20)
    img[masks["lost"]] = (0.84, 0.10, 0.11)
    ax.imshow(img, origin="lower", extent=ext, interpolation="nearest")
    ax.set_title(f"(e) what the move changes\n"
                 f"+{best['recovered']} cells ({best['recovered_m2']:.3f} m$^2$)"
                 f"  /  -{best['new_dead']} ({best['new_dead_m2']:.3f} m$^2$)",
                 fontsize=10, loc="left")
    ax.set_xlabel("canvas x (m)")
    ax.set_aspect("equal")
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    _arm_marks(ax)
    ax.legend(handles=[Patch(fc=(0.13, 0.65, 0.20), label="recovered"),
                       Patch(fc=(0.84, 0.10, 0.11), label="newly dead"),
                       Patch(fc=(0.16, 0.16, 0.16), label="dead before & after")],
              fontsize=7, loc="center left", framealpha=0.95,
              borderpad=0.4, handlelength=1.2)

    fig.suptitle("Arm 2 (side, green) — sliding the base along its boom pole: "
                 "what the paper gains", fontsize=13, x=0.055, ha="left")
    fig.savefig(path, dpi=150)
    print("wrote", path)


def _spans(x, m):
    out, i = [], 0
    while i < len(x):
        if m[i]:
            j = i
            while j + 1 < len(x) and m[j + 1]:
                j += 1
            lo = x[i] if i == 0 else 0.5 * (x[i - 1] + x[i])
            hi = x[j] if j == len(x) - 1 else 0.5 * (x[j] + x[j + 1])
            out.append((lo, hi))
            i = j + 1
        else:
            i += 1
    return out


def _arm_marks(ax):
    """Mark each arm's SHOULDER (J2 axis) — the centre of its under-base hole.
    Arm 2's is 0.333 m in -X of its plate, because its J1 axis is horizontal."""
    import matplotlib.patheffects as pe
    glow = [pe.withStroke(linewidth=2.4, foreground="w")]
    for aid in (13, 31, 2):
        s = FLEET[aid]
        p = shoulder_xy(s)
        off = not (0 <= p[0] <= SHEET[0] and 0 <= p[1] <= SHEET[1])
        lx = min(max(p[0], 0.05), SHEET[0] - 0.05)
        ly = min(max(p[1], 0.04), SHEET[1] - 0.04)
        ax.plot(lx, ly, "v" if off else "o", ms=7 if off else 6, mfc=s.color,
                mec="w", mew=1.2, zorder=6)
        ax.annotate(f"{aid}" + (" (off-sheet)" if off else ""), (lx, ly),
                    xytext=(8, 3), zorder=7, textcoords="offset points",
                    fontsize=8.5, weight="bold", color=s.color,
                    path_effects=glow)


# --- main -----------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dz", nargs="*", type=float, default=None,
                    help="pole offsets in cm (+ = up); default = the sweep")
    ap.add_argument("--atlas", default=str(ROOT / "out" / "atlas_final"))
    ap.add_argument("--out", default=str(ROOT / "out"))
    ap.add_argument("--work", default=str(ROOT / "out" / "arm2_sweep"))
    ap.add_argument("--neighbours", type=float, default=None,
                    help="also re-run arms 13/31 at this dz (cm) and report; "
                         "999 = at the best height found")
    ap.add_argument("--replot", action="store_true",
                    help="reuse the atlases already under --work, just "
                         "recompute the numbers and redraw the figure")
    a = ap.parse_args()

    base = load_baseline(a.atlas)
    dzs = a.dz if a.dz is not None else \
        [-26, -24, -22, -20, -18, -16, -14, -12, -10, -8, -6, -4, -3, -2,
         -1, 0, 1, 2, 4, 6, 8, 12]

    jobs = [(float(dz), [2], a.work) for dz in dzs]
    if a.replot:                       # reuse the atlases already in --work
        results = {}
        for dz, _, w in jobs:
            p = _tag_dir(w, dz) / "atlas_arm2.npz"
            results[dz] = {2: np.load(p)["data"]}
        print(f"replot: reused {len(results)} cached atlases from {a.work}")
    else:
        with mp.Pool(min(len(jobs), max(1, (mp.cpu_count() or 4) - 2))) as pool:
            results = dict(pool.map(run_one, jobs))

    rows, keep = [], {}
    for dz in sorted(results):
        arr = results[dz][2]
        reach, go2 = _mask(arr, base["shape"])
        eng, ok, why = feasibility(dz)
        r = dict(dz_cm=dz, z_canvas=round(Z0_CANVAS + dz / 100.0, 4),
                 z_w_cm=round(141.268 + dz, 3), feasible=bool(ok), why=why,
                 arm2_reach=int(reach.sum()), **score(go2, base, eng))
        rows.append(r)
        keep[dz] = go2

    # two optima: what the pole ALLOWS today, and where the coverage actually
    # peaks (which needs the pole extended downward -- see the doc appendix).
    feas = [r for r in rows if r["feasible"]]
    key = lambda r: (r["union_go_pct"], -abs(r["dz_cm"]))          # noqa: E731
    best_feas = max(feas or rows, key=key)
    best = max(rows, key=key)
    go2b = keep[best["dz_cm"]]
    union = base["pg_other"][0] | base["pg_other"][1] | go2b
    masks = dict(union=union,
                 cnt=base["pg_other"][0].astype(int) + base["pg_other"][1] + go2b,
                 recovered=base["dead0"] & go2b,
                 lost=base["ug0"] & ~union)
    best["pole_extension_cm"] = round(
        max(0.0, POLE_BOTTOM - (BRACKET_Z[0] + best["dz_cm"])), 2)
    best["still_dead"] = why_dead(~union, base)

    for r in rows:
        r["pole_extension_cm"] = round(
            max(0.0, POLE_BOTTOM - (BRACKET_Z[0] + r["dz_cm"])), 2)
    hdr = ("  dz    z_c  needs+pole   arm2GO  %sheet  recov  union%   >=2%  "
           "newdead")
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for r in rows:
        pe = ("         -" if r["feasible"]
              else f"  +{r['pole_extension_cm']:5.1f}cm")
        print(f"{r['dz_cm']:+5.0f} {r['z_canvas']:6.3f}{pe}"
              f"{r['arm2_go']:8d}{r['arm2_go_pct']:8.2f}"
              f"{r['recovered']:7d}{r['union_go_pct']:8.2f}"
              f"{r['overlap2_pct']:7.2f}{r['new_dead']:8d}")
    u0 = 100 * base["ug0"].mean()
    print(f"\nbest WITHOUT touching the pole: dz {best_feas['dz_cm']:+.0f} cm "
          f"-> canvas z {best_feas['z_canvas']:.3f}, union "
          f"{best_feas['union_go_pct']:.2f} % (from {u0:.2f} %)")
    print(f"unconstrained optimum:          dz {best['dz_cm']:+.0f} cm "
          f"-> canvas z {best['z_canvas']:.3f}, union "
          f"{best['union_go_pct']:.2f} % (from {u0:.2f} %) -- needs the pole "
          f"extended {best['pole_extension_cm']:.0f} cm downward")
    d0 = why_dead(base["dead0"], base)
    print(f"\ndead-cell accounting   as drawn ({100*base['dead0'].mean():.1f} % "
          f"of the sheet)  ->  at the optimum ({best['dead_pct']:.1f} %):")
    for k in best["still_dead"]:
        if not k.startswith("_"):
            a_, b_ = d0[k]["cells"], best["still_dead"][k]["cells"]
            print(f"  {a_:5d} -> {b_:5d}  ({b_ - a_:+5d})  {k}")

    out = dict(baseline=dict(union_go_pct=round(u0, 2),
                             arm2_go=int(base["go2_0"].sum()),
                             dead=int(base["dead0"].sum()),
                             cells=int(base["ug0"].size),
                             still_dead=why_dead(base["dead0"], base)),
               pole=dict(sliding_boxes=list(SLIDING),
                         z_w_cm_front_copy=list(POLE_Z_FRONT),
                         z_w_cm_top_copy=list(POLE_Z_TOP),
                         assumed_bottom_w_cm=POLE_BOTTOM,
                         bracket_z_w_cm=list(BRACKET_Z),
                         stack_z_w_cm=list(STACK_Z), ceiling_z_w_cm=CEIL_Z,
                         travel_down_cm=round(DZ_DOWN, 2),
                         travel_up_cm=round(DZ_UP, 2)),
               rows=rows, best=best, best_no_pole_change=best_feas)

    if a.neighbours is not None:
        dz = best["dz_cm"] if a.neighbours == 999 else float(a.neighbours)
        cached = {i: _tag_dir(a.work, dz) / f"atlas_arm{i}.npz" for i in (13, 31)}
        if a.replot and all(p.exists() for p in cached.values()):
            res = {i: np.load(p)["data"] for i, p in cached.items()}
        else:
            _, res = run_one((dz, [13, 31], a.work))
        nb = {}
        for aid in (13, 31):
            _, g = _mask(res[aid], base["shape"])
            i = 0 if aid == 13 else 1
            b = base["pg_other"][i]
            nb[str(aid)] = dict(go=int(g.sum()), go_before=int(b.sum()),
                                delta=int(g.sum()) - int(b.sum()),
                                lost=int((b & ~g).sum()),
                                gained=int((g & ~b).sum()))
            print(f"neighbour arm {aid} @ dz {dz:+.0f}: strict-GO "
                  f"{int(b.sum())} -> {int(g.sum())} "
                  f"(lost {nb[str(aid)]['lost']}, gained {nb[str(aid)]['gained']})")
        out["neighbours"] = dict(dz_cm=dz, **nb)

    Path(a.out).mkdir(parents=True, exist_ok=True)
    json.dump(out, open(Path(a.out) / "arm2_height_sweep.json", "w"), indent=1)
    figure(rows, base, best, masks, Path(a.out) / "arm2_height_sweep.png",
           best_feas)


if __name__ == "__main__":
    main()
