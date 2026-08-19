#!/usr/bin/env python3
"""Smoothing + pacing on top of the PWL redundancy plan — arm 31, two strokes.

Continues scripts/demo_pwl.py (whose stroke constructions this imports, so the
two demos cannot drift apart) one step further down the stack:

    plan_pwl  ->  smooth_q7_of_s  ->  certify  ->  pace

  * SMOOTH rounds every interior corner of the polyline q7(s) with a quadratic
    Bezier over a window in s.  q7(s) is a graph over s, so the result is still
    a function — monotone in s by construction, never as a constraint.
  * CERTIFY re-solves the case-consistent IK along the smoothed curve at 5 mm
    with the same gates the polyline was accepted under (sigma >= 0.10,
    margin >= 0.15, ||dq||_inf <= 0.35), and bisects any window whose corner
    fails back toward the sharp knot, which plan_pwl already certified.
  * PACE turns the certified geometry into a schedule: constant tip speed
    except where dq/ds would push a joint past its FR3 velocity limit.

Outputs out/smooth_band.png and the report numbers.  System python3 only.
"""
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from aris_sixarm import pacing, planner, pwl, smooth  # noqa: E402
from aris_sixarm.frames import QD_MAX  # noqa: E402
from demo_pwl import BACKOUT_DS, demo_strokes  # noqa: E402

FRAC_SWEEP = (0.25, 0.5, 1.0)
JC = plt.get_cmap("tab10").colors
spec, STROKES = demo_strokes()
out = {}

for name, poly, ds in STROKES:
    print(f"\n=== stroke {name} ===")
    pts, _ = planner.resample(poly, ds)
    t0 = time.time()
    lat = planner.build_lattice(pts, spec)
    sheets = pwl.sheet_fields(lat)
    sh = pwl.dominant_sheet(sheets)
    res = pwl.plan_pwl(lat, sh)
    assert res["ok"], f"no PWL path on sheet {sh['id']}"
    print(f"  PWL  : {res['n_knots']} knots over {len(pts)} lattice steps, "
          f"{len(sheets)} sheets, on sheet {sh['id']} ({time.time() - t0:.1f} s)")

    # ---- reference: back the RAW polyline out, same resolution -----------
    raw = pwl.backout(poly, spec, res["knots"], lat=lat, sheet=sh, ds=BACKOUT_DS)
    assert raw["ok"] and raw["tip_err"] < 2e-3
    print(f"  raw  : {len(raw['qs'])} samples @ {BACKOUT_DS * 1e3:.0f} mm, "
          f"L={raw['arc_len']:.3f} m, min_sigma={raw['min_sigma']:.4f} "
          f"min_margin={raw['min_margin']:.3f} max|dq/ds|={raw['max_dqds']:.2f} rad/m")

    # ---- smooth + certify -------------------------------------------------
    curve = smooth.smooth_q7_of_s(res["knots"], frac=smooth.ROUND_FRAC)
    t0 = time.time()
    sm = smooth.certify(poly, spec, curve, lat=lat, sheet=sh, ds=BACKOUT_DS,
                        verbose=True)
    print("\n".join(smooth.smooth_report(name, sm, raw))
          + f"  ({time.time() - t0:.1f} s)")
    assert sm["certified"], f"{name}: smoothed curve failed certification"
    assert sm["tip_err"] < 2e-3, "smoothed path left the stroke"

    # how the rounding window trades against the corner step in dq/ds
    sweep = []
    for f in FRAC_SWEEP:
        c = smooth.smooth_q7_of_s(res["knots"], frac=f)
        r = smooth.certify(poly, spec, c, lat=lat, sheet=sh, ds=BACKOUT_DS)
        sweep.append((f, r))
    print("          window sweep  frac  max|dq/ds|  corner step  min sigma  cert")
    print(f"          {'':13} sharp {raw['max_dqds']:10.2f} "
          f"{raw['max_dqds_jump']:12.3f} {raw['min_sigma']:10.4f}   -")
    for f, r in sweep:
        print(f"          {'':13}{f:5.2f} {r['max_dqds']:10.2f} "
              f"{r['max_dqds_jump']:12.3f} {r['min_sigma']:10.4f} "
              f"{'  y' if r['certified'] else '  N'}")

    # ---- pace -------------------------------------------------------------
    pc = pacing.pace(sm["qs"], sm["arc_len"], ds_m=sm["ds"])
    print("\n".join(pacing.pace_report(name, pc)))
    v_break = float(pc["v_limit"].min())
    stress = pacing.pace(sm["qs"], sm["arc_len"], ds_m=sm["ds"],
                         v_draw=2.0 * v_break)
    print(f"         the joints only become the constraint above "
          f"v = {v_break:.3f} m/s ({v_break / pc['v_draw']:.0f}x the drawing "
          f"speed); at 2x that, {100 * stress['frac_slowed']:.0f} % of the "
          f"stroke slows, worst factor {stress['worst_factor']:.2f}x, "
          f"{stress['total_time']:.2f} s vs {stress['naive_time']:.2f} s naive")
    out[name] = dict(lat=lat, sh=sh, res=res, raw=raw, sm=sm, pc=pc,
                     stress=stress, v_break=v_break, sheets=sheets)

# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------
fig, axes = plt.subplots(2, len(out), figsize=(8.2 * len(out), 9.6),
                         gridspec_kw=dict(height_ratios=[1.12, 1.0]))
axes = np.atleast_2d(axes)
for c, (name, d) in enumerate(out.items()):
    lat, sh, res = d["lat"], d["sh"], d["res"]
    sm, raw, pc, st = d["sm"], d["raw"], d["pc"], d["stress"]
    curve = sm["curve"]
    nint = max(len(curve.knots) - 2, 0)

    # ---- top: the band, the polyline, and the rounded curve --------------
    ax = axes[0, c]
    Ns, Nq = sh["mask"].shape
    s_grid, q7s = np.arange(Ns) / (Ns - 1), lat["q7s"]
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.88")
    im = ax.imshow(np.ma.masked_invalid(sh["sigma"]).T, origin="lower",
                   aspect="auto", cmap=cmap, extent=[0, 1, q7s[0], q7s[-1]],
                   vmin=pwl.SIGMA_GATE, vmax=float(np.nanmax(sh["sigma"])))
    plt.colorbar(im, ax=ax, label=r"$\sigma_{min}$ on sheet %d" % sh["id"],
                 pad=0.01)
    ax.contour(s_grid, q7s, res["free"].T.astype(float), levels=[0.5],
               colors="white", linewidths=1.1)
    occ = q7s[np.flatnonzero(sh["mask"].any(axis=0))]
    nint_ = max(len(curve.knots) - 2, 0)
    ylo = max(q7s[0], occ.min() - 0.7)
    # extra sky when a zoom inset has to live above the band
    yhi = min(q7s[-1], occ.max() + (1.25 if nint_ else 0.7))

    def draw_plan(a, lw_pwl=1.1, lw_sm=2.4, ms=5.5, ss=None):
        a.plot(res["knots"][:, 0], res["knots"][:, 1], "-", color="#111111",
               lw=lw_pwl, alpha=0.85, label=f"PWL plan ({res['n_knots']} knots)")
        a.plot(res["knots"][:, 0], res["knots"][:, 1], "o", ms=ms, mfc="#ffffff",
               mec="#111111", mew=1.1, zorder=5)
        ss = np.linspace(0, 1, 4000) if ss is None else ss
        a.plot(ss, curve(ss), "-", color="#ffffff", lw=lw_sm + 2.0, alpha=0.85,
               solid_capstyle="round")
        a.plot(ss, curve(ss), "-", color="#ff2d55", lw=lw_sm,
               label="C1 corner-rounded $q_7(s)$")

    # window "ruler" along the top edge — visible without hiding the field
    ybar = yhi - 0.055 * (yhi - ylo)
    for k in range(1, len(curve.knots) - 1):
        lo, hi = curve.corner_span(k)
        if hi <= lo:
            continue
        shrunk = bool(sm["shrunk"][k - 1])
        col = "#ff2d55" if shrunk else "#0b7fd4"
        ax.axvspan(lo, hi, color=col, alpha=0.13, lw=0)
        for x in (lo, hi):
            ax.axvline(x, color=col, lw=0.7, ls=":", alpha=0.85)
        ax.plot([lo, hi], [ybar, ybar], "-", color=col, lw=4.0,
                solid_capstyle="butt", zorder=6)
        ax.plot([lo, hi], [ybar, ybar], "|", color=col, ms=7, mew=1.4, zorder=6)
        if shrunk:
            ax.annotate(f"corner {k}: window shrunk\n"
                        f"{sm['windows0'][k - 1]:.4f} → {sm['windows'][k - 1]:.4f}",
                        xy=(curve.s[k], curve.q[k]), xytext=(0, 30),
                        textcoords="offset points", ha="center", fontsize=7,
                        color="#7a0022", zorder=7,
                        bbox=dict(fc="white", ec="#ff2d55", lw=0.8, alpha=0.95),
                        arrowprops=dict(arrowstyle="->", color="#ff2d55", lw=0.9))
    if nint:
        ax.plot([], [], "-", color="#0b7fd4", lw=4.0,
                label="corner rounding window (none shrunk)"
                if not sm["shrunk"].any() else "rounding window")
    draw_plan(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel("arc length $s$ (normalised)")
    ax.set_ylabel(r"$q_7$ (rad)")
    ax.set_title(f"{name} — smoothed plan in the band "
                 f"({nint} interior corner{'s' if nint != 1 else ''}, "
                 f"{int(sm['shrunk'].sum())} shrunk, "
                 + ("certified" if sm["certified"] else "NOT certified") + ")")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.93)

    if nint:
        # zoom on the sharpest corner — the rounding is ~2 % of the stroke wide
        # and simply invisible at full scale
        k = 1 + int(np.argmax(np.abs(np.diff(curve.slope))))
        w = float(curve.windows[k - 1])
        lo, hi = curve.s[k] - 3.2 * w, curve.s[k] + 3.2 * w
        # upper right: the only corner of the panel with neither the band, the
        # legend, nor the window ruler in it
        axz = ax.inset_axes([0.63, 0.67, 0.30, 0.26])
        ssz = np.linspace(lo, hi, 800)
        axz.axvspan(*curve.corner_span(k), color="#0b7fd4", alpha=0.16, lw=0)
        draw_plan(axz, lw_pwl=1.3, lw_sm=2.6, ms=6.0, ss=ssz)
        axz.set_xlim(lo, hi)
        qz = curve(ssz)
        pad = 0.35 * (qz.max() - qz.min() + 1e-9)
        axz.set_ylim(min(qz.min(), curve.q[k]) - pad,
                     max(qz.max(), curve.q[k]) + pad)
        axz.set_title(f"corner {k}, ×{1 / (6.4 * w):.0f} zoom "
                      f"(window ±{w:.4f})", fontsize=7.0, pad=2.5)
        axz.tick_params(labelsize=6)
        axz.set_facecolor("white")
        axz.patch.set_alpha(0.93)
        for sp in axz.spines.values():
            sp.set_color("#0b7fd4")
        ax.indicate_inset_zoom(axz, edgecolor="#0b7fd4", alpha=0.8)
    else:
        ax.text(0.5, 0.955, "single segment — no corner to round:\n"
                "the smoothed curve IS the polyline",
                transform=ax.transAxes, ha="center", va="top", fontsize=9,
                bbox=dict(fc="white", ec="0.55", alpha=0.95))

    # ---- bottom: joint velocities at the paced timing --------------------
    ax = axes[1, c]
    t, qd = pc["t"], np.abs(pc["qd"])
    floor = max(qd.max() / 80.0, 1e-3)
    for j in range(7):
        ax.plot(t, np.maximum(qd[:, j], floor), lw=1.3, color=JC[j],
                label=f"$|\\dot q_{j + 1}|$")
    caption = []
    for lim in sorted(set(QD_MAX.tolist())):
        js = ",".join(str(j + 1) for j in range(7) if QD_MAX[j] == lim)
        ax.axhline(lim, color="#d62728", lw=1.2, ls="--")
        ax.axhline(pc["safety"] * lim, color="#d62728", lw=0.8, ls=":", alpha=0.6)
        caption.append(f"j{js} {lim:.2f}")
    # one caption beats three labels: the three limits are within a factor 2 of
    # each other and their labels collide on a log axis at any font size
    ax.text(0.012, QD_MAX.max() * 1.35,
            "FR3 QD_MAX (dashed):  " + " · ".join(caption) + " rad/s\n"
            "80 % safety ceiling (dotted)", fontsize=7.2, color="#d62728",
            va="bottom", ha="left", linespacing=1.35,
            transform=ax.get_yaxis_transform())
    ax.set_yscale("log")
    ax.set_ylim(floor * 0.85, QD_MAX.max() * 25.0)
    ax.set_xlim(0, t[-1])
    ax.set_xlabel(f"time (s), paced at $v_{{draw}}$ = {1e3 * pc['v_draw']:.0f} mm/s")
    ax.set_ylabel(r"$|\dot q_j|$ (rad/s, log)")
    ax.set_title(f"{name} — paced {pc['total_time']:.1f} s "
                 f"(naive {pc['naive_time']:.1f} s), "
                 f"{100 * pc['frac_slowed']:.0f} % of the stroke slowed\n"
                 f"peak $|\\dot q|$ = {qd.max():.3f} rad/s = "
                 f"{100 * pc['headroom']:.1f} % of the limit "
                 f"({1 / pc['headroom']:.0f}× headroom)", fontsize=10.5)
    ax.legend(loc="lower left", fontsize=7.5, ncol=4, framealpha=0.93,
              columnspacing=1.0, handlelength=1.4)

    # inset: the tip-speed profiles, where the slowdown zones live
    axv = ax.inset_axes([0.55, 0.66, 0.43, 0.31])
    s_ax = sm["s"]
    axv.plot(s_ax, 1e3 * pc["v"], "-", color="#111111", lw=1.8,
             label=f"$v$ @ {1e3 * pc['v_draw']:.0f} mm/s")
    axv.plot(s_ax, 1e3 * pc["v_limit"], "-", color="#0b7fd4", lw=1.4,
             label="ceiling $v_{lim}(s)$")
    axv.plot(s_ax, 1e3 * st["v"], "-", color="#ff2d55", lw=1.6,
             label=f"$v$ @ {1e3 * st['v_draw']:.0f} mm/s")
    if st["slow"].any():
        axv.fill_between(s_ax, 1e3 * st["v"], 1e3 * st["v_draw"],
                         where=st["slow"], color="#ff2d55", alpha=0.30, lw=0,
                         label="slowdown zone")
    axv.set_yscale("log")
    axv.set_xlim(0, 1)
    axv.set_xlabel("$s$", fontsize=7, labelpad=1)
    axv.set_ylabel("mm/s", fontsize=7, labelpad=1)
    axv.set_title(f"tip speed: joints bind above {1e3 * d['v_break']:.0f} mm/s "
                  f"({d['v_break'] / pc['v_draw']:.0f}× $v_{{draw}}$)", fontsize=7.5)
    axv.tick_params(labelsize=6)
    axv.set_facecolor("white")
    axv.patch.set_alpha(0.93)
    axv.legend(fontsize=6.0, loc="lower left", framealpha=0.9,
               handlelength=1.2, borderpad=0.3, labelspacing=0.25)
plt.tight_layout()
plt.savefig(ROOT / "out/smooth_band.png", dpi=140)
print("\nwrote out/smooth_band.png")

# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------
print(f"\n{'stroke':<11} {'corners':>8} {'shrunk':>7} {'min sigma':>19} "
      f"{'min margin':>19} {'travel (rad)':>17} {'max|dq/ds| rad/m':>21}")
print(f"{'':<11} {'':>8} {'':>7} {'PWL -> smoothed':>19} {'PWL -> smoothed':>19} "
      f"{'PWL -> smoothed':>17} {'PWL -> smoothed':>21}")
for name, d in out.items():
    raw, sm = d["raw"], d["sm"]
    nint = max(len(sm["curve"].knots) - 2, 0)
    print(f"{name:<11} {nint:>8} {int(sm['shrunk'].sum()):>7} "
          f"{raw['min_sigma']:>9.4f} -> {sm['min_sigma']:<7.4f} "
          f"{raw['min_margin']:>9.3f} -> {sm['min_margin']:<7.3f} "
          f"{raw['sum_travel']:>7.2f} -> {sm['sum_travel']:<7.2f} "
          f"{raw['max_dqds']:>10.2f} -> {sm['max_dqds']:<8.2f}")
print(f"\n{'stroke':<11} {'tip err (m)':>12} {'unpaced max|qd|':>16} "
      f"{'% of limit':>11} {'paced (s)':>10} {'naive (s)':>10} {'% slowed':>9} "
      f"{'v_break (mm/s)':>15}")
for name, d in out.items():
    sm, pc = d["sm"], d["pc"]
    print(f"{name:<11} {sm['tip_err']:>12.1e} "
          f"{pc['max_qd_unpaced'].max():>16.4f} "
          f"{100 * pc['unpaced_headroom']:>10.1f}% {pc['total_time']:>10.2f} "
          f"{pc['naive_time']:>10.2f} {100 * pc['frac_slowed']:>8.1f}% "
          f"{1e3 * d['v_break']:>15.0f}")
