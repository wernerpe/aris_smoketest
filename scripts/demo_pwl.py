#!/usr/bin/env python3
"""Piecewise-linear redundancy planning in (s, q7) — arm 31, two strokes.

Runs the whole new pipeline against the existing lattice DP on

  A_rim_arc:  the r = 0.66 rim arc from demo_stroke.py (clipped to the sheet) —
              the stroke greedy diffIK cannot do at all;
  R_bowl:     stroke 1 of the letter R as the ARIS demo actually places it on
              arm 31 (bowl + diagonal leg, one continuous move).

For each stroke: build the (s x q7 x branch) lattice, plan it the old way
(planner.plan), then label IK sheets, plan a PWL path q7(s) on the dominant
sheet, and back joint configurations out of the polyline with dense
case-consistent IK.

Outputs: out/pwl_band.png  (bands + paths on top, sigma profiles below)
No drake and no meshcat needed — system python3 is enough.
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
from aris_sixarm import letters, planner, pwl, writing  # noqa: E402
from aris_sixarm.fleet import FLEET  # noqa: E402
from aris_sixarm.metrics import sigma_min, tip_jacobian  # noqa: E402

spec = FLEET[31]
bx, by = spec.xy

# stroke A — the demo_stroke.py rim arc, same construction
th = np.linspace(-0.6 * np.pi, 1.05 * np.pi, 400)
arcA = planner.clip_to_sheet(np.column_stack([bx + 0.66 * np.cos(th),
                                              by + 0.66 * np.sin(th)]))
# stroke B — letter R, stroke 1, at the placement aris_writing_demo.py settles on
# (the nudge search moves R 0.10 m back toward the arm-31 base)
R_CENTER = (1.30052, 1.14999)
bowlB = letters.place("R", R_CENTER, letters.DEFAULT_HEIGHT, letters.DEFAULT_ASPECT)[1]

STROKES = [("A_rim_arc", arcA, 0.012), ("R_bowl", bowlB, writing.DS)]
BACKOUT_DS = 0.005
out = {}

for name, poly, ds in STROKES:
    print(f"\n=== stroke {name} ===")
    pts, _ = planner.resample(poly, ds)
    Ns = len(pts)
    t0 = time.time()
    lat = planner.build_lattice(pts, spec)
    print(f"  lattice: {Ns} steps, {ds * (Ns - 1):.2f} m, "
          f"{lat['valid'].sum()} valid nodes ({time.time() - t0:.1f} s)")

    # ---- reference: the existing ladder-graph DP -------------------------
    dp = planner.plan(lat, objective="maximin_sigma")
    rep = planner.path_report(lat, dp)
    print(f"  DP   : ok={dp['ok']} min_sigma={rep['min_sigma']:.4f} "
          f"min_margin={rep['min_margin']:.3f} travel={rep['sum_travel']:.2f} rad "
          f"({len(dp['qs'])} nodes)")
    qd, ud, dfb = writing.densify(dp["qs"], lat["pts"], spec)
    sig_dpd = np.array([sigma_min(tip_jacobian(q, pen_ext=lat["pen_ext"])) for q in qd])
    trav_dpd = float(np.abs(np.diff(qd, axis=0)).sum())
    print(f"  DP densified to {len(qd)} steps: min_sigma={sig_dpd.min():.4f}, "
          f"travel={trav_dpd:.2f} rad" + (f", {dfb} IK fallbacks" if dfb else ""))

    # ---- 1. sheets -------------------------------------------------------
    t0 = time.time()
    sheets = pwl.sheet_fields(lat)
    lines = pwl.sheet_report(sheets, Ns)
    print(f"  {lines[0]} ({time.time() - t0:.1f} s)")
    print("\n".join(f"  {ln}" for ln in lines[1:]))
    sh = pwl.dominant_sheet(sheets)
    dp_sheet, on = pwl.sheet_of_path(sheets, dp["path"])
    print(f"  chosen sheet {sh['id']}: {sh['nodes']} nodes, {sh['mask'].sum()} cells, "
          f"spans_s={sh['spans_s']}; the lattice DP ran on sheet {dp_sheet} "
          f"({100 * on:.0f} % of its steps)"
          + ("  <- same sheet" if dp_sheet == sh["id"] else "  <- DIFFERENT sheet"))

    # ---- 2. PWL on that sheet -------------------------------------------
    res = pwl.plan_pwl(lat, sh)
    assert res["ok"], f"no PWL path on sheet {sh['id']} (cut_s={res['cut_s']:.3f})"
    print(f"  PWL  : {res['n_knots']} knots vs {res['n_dense']} dense steps "
          f"({res['n_dense'] / res['n_knots']:.0f}x), lattice bottleneck "
          f"sigma={res['bottleneck']:.4f}, min clearance along it "
          f"{res['dense_clearance'].min():.2f} grid units; corridor certified by "
          f"{res['exact_calls']} IK chases / {res['exact_samples']} samples")
    print("         knots (s, q7): "
          + "  ".join(f"({a:.3f}, {b:+.3f})" for a, b in res["knots"]))

    # ---- 3. back out joint configurations --------------------------------
    t0 = time.time()
    bo = pwl.backout(poly, spec, res["knots"], lat=lat, sheet=sh, ds=BACKOUT_DS)
    print(f"  back : ok={bo['ok']} {len(bo['qs'])} samples @ {BACKOUT_DS * 1e3:.0f} mm "
          f"({time.time() - t0:.1f} s), min_sigma={bo['min_sigma']:.4f} "
          f"mean={bo['mean_sigma']:.4f} min_margin={bo['min_margin']:.3f}")
    print(f"         tip err max {bo['tip_err']:.2e} m, |dq|_inf max {bo['max_step']:.4f}, "
          f"travel={bo['sum_travel']:.2f} rad, {bo['fallbacks']} solve_cc fallbacks, "
          f"{bo['fails']} failures")
    assert bo["ok"] and bo["tip_err"] < 2e-3, "backed-out path left the stroke"
    assert bo["continuous"], "backed-out path is not continuous"
    print(f"         joint travel {bo['sum_travel']:.2f} rad vs {trav_dpd:.2f} rad for the "
          f"densified DP ({trav_dpd / max(bo['sum_travel'], 1e-9):.1f}x): the DP spends its "
          f"continuity budget on q7 staircase jitter, the polyline cannot")
    out[name] = (lat, dp, sheets, sh, res, bo, sig_dpd, ud)

# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------
fig, axes = plt.subplots(2, len(out), figsize=(7.0 * len(out), 8.4),
                         gridspec_kw=dict(height_ratios=[2.1, 1.0]))
for c, (name, (lat, dp, sheets, sh, res, bo, sig_dpd, ud)) in enumerate(out.items()):
    ax = axes[0, c]
    Ns, Nq = sh["mask"].shape
    s_grid, q7s = np.arange(Ns) / (Ns - 1), lat["q7s"]
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.88")
    im = ax.imshow(np.ma.masked_invalid(sh["sigma"]).T, origin="lower", aspect="auto",
                   cmap=cmap, extent=[0, 1, q7s[0], q7s[-1]],
                   vmin=pwl.SIGMA_GATE, vmax=float(np.nanmax(sh["sigma"])))
    plt.colorbar(im, ax=ax, label=r"$\sigma_{min}$ on sheet %d" % sh["id"], pad=0.01)
    ax.contour(s_grid, q7s, res["free"].T.astype(float), levels=[0.5],
               colors="white", linewidths=1.2, linestyles="-")
    ax.plot([], [], "-", color="white", lw=1.2, label="free-region boundary")
    ax.plot(res["dense"][:, 0], res["dense"][:, 1], "-", color="#ff9d00", lw=1.0,
            label=f"dense max-clearance DP ({res['n_dense']} steps)")
    s_dp = np.arange(len(dp["q7s"])) / (Ns - 1)
    ax.plot(s_dp, dp["q7s"], "--", color="#d62728", lw=1.6,
            label="lattice DP (reference)")
    ax.plot(res["knots"][:, 0], res["knots"][:, 1], "-", color="#ffffff", lw=4.2,
            alpha=0.75, solid_capstyle="round")
    ax.plot(res["knots"][:, 0], res["knots"][:, 1], "-o", color="#111111", lw=2.4,
            ms=7, mfc="#ff2d55", mec="#111111", mew=1.2,
            label=f"PWL plan: {res['n_knots']} knots")
    ax.set_xlim(0, 1)
    occ = q7s[np.flatnonzero(sh["mask"].any(axis=0))]      # zoom to the band
    ax.set_ylim(max(q7s[0], occ.min() - 0.7), min(q7s[-1], occ.max() + 0.7))
    ax.set_xlabel("arc length s (normalised)")
    ax.set_ylabel(r"$q_7$ (rad)")
    ax.text(0.995, 0.02, f"full FR3 $q_7$ range: [{q7s[0]:.2f}, {q7s[-1]:.2f}] rad",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color="0.25")
    ax.set_title(f"{name} — band on sheet {sh['id']} of {len(sheets)} "
                 f"({sh['nodes']} nodes)")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.92)

    ax = axes[1, c]
    ax.plot(np.linspace(0, 1, len(sig_dpd)), sig_dpd, "--", color="#d62728", lw=1.4,
            label=f"lattice DP, densified (min {sig_dpd.min():.3f})")
    ax.plot(bo["s"], bo["sigmas"], "-", color="#111111", lw=1.8,
            label=f"PWL backed out (min {bo['min_sigma']:.3f})")
    ax.plot(res["dense"][:, 0], res["sigma"], ":", color="#ff9d00", lw=1.4,
            label="PWL, lattice field (predicted)")
    ax.axhline(pwl.SIGMA_GATE, color="0.4", lw=0.9, ls="-.")
    ax.text(0.005, pwl.SIGMA_GATE, " band gate", fontsize=7, va="bottom", color="0.35")
    ax.set_xlim(0, 1)
    ax.set_xlabel("arc length s (normalised)")
    ax.set_ylabel(r"$\sigma_{min}$")
    ax.set_title(f"{name} — controllability along the FINAL configurations")
    ax.legend(loc="best", fontsize=8, framealpha=0.92)
plt.tight_layout()
plt.savefig(ROOT / "out/pwl_band.png", dpi=140)
print("\nwrote out/pwl_band.png")

# --------------------------------------------------------------------------
# summary table
# --------------------------------------------------------------------------
print(f"\n{'stroke':<11} {'sheets':>6} {'knots/dense':>12} {'DP sig':>8} "
      f"{'PWL sig':>8} {'DP marg':>8} {'PWL marg':>9} {'tip err':>9} {'fallb':>6}")
for name, (lat, dp, sheets, sh, res, bo, sig_dpd, ud) in out.items():
    rep = planner.path_report(lat, dp)
    print(f"{name:<11} {len(sheets):>6} {res['n_knots']:>5}/{res['n_dense']:<6} "
          f"{rep['min_sigma']:>8.4f} {bo['min_sigma']:>8.4f} "
          f"{rep['min_margin']:>8.3f} {bo['min_margin']:>9.3f} "
          f"{bo['tip_err']:>9.1e} {bo['fallbacks']:>6}")
