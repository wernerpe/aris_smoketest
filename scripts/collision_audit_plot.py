#!/usr/bin/env python3
"""The collision audit in one page — out/collision_audit.png.

Reads `out/collision_audit.json` (and `out/collision_audit_height.npz` if it is
there) and draws the four things the audit is FOR:

  1  how far each link's real surface reaches OUTSIDE the capsule that is
     supposed to envelope it — the pose-independent optimism budget;
  2  the distribution of (capsule clearance - mesh clearance) over near pose
     pairs, split at zero into CONSERVATIVE and OPTIMISTIC;
  3  the base column: the measured radial profile of link0 + swept link1
     against the r = 0.09 capsule and the r = 0.12 box that stand for it;
  4  the height verdict: union coverage at h = 0.850 and h = 0.922 under
     empty air, the published column, and the measured one.

    python3 scripts/collision_audit_plot.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

# The reference data-viz palette, used at its documented slots: categorical
# 1/2/3 (blue, orange, aqua — the three that validate all-pairs), the
# blue<->red diverging pair with a gray midpoint for the signed error, and the
# documented chrome inks.  Slots are never cycled and never re-ordered.
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"
CRIT = "#d03b3b"

LINK_ORDER = ["link0_cable", "link0", "link1", "link2", "link3", "link4",
              "link5", "link6", "link7", "hand", "leftfinger", "rightfinger",
              "housing", "cap", "lead"]
PRETTY = {"link0_cable": "link0 connector + cable"}


def _tidy(ax, xlabel=None, ylabel=None, title=None, sub=None):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(labelsize=7.4, colors=MUTED, length=3, width=0.8)
    for lb in ax.get_xticklabels() + ax.get_yticklabels():
        lb.set_color(INK2)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=7.8, color=INK2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=7.8, color=INK2)
    if title:
        ax.text(0, 1.085, title, transform=ax.transAxes, fontsize=10.5,
                color=INK, va="bottom", fontweight="bold")
    if sub:
        ax.text(0, 1.028, sub, transform=ax.transAxes, fontsize=7.6,
                color=MUTED, va="bottom")


def panel_excess(ax, rec):
    """1 — per-part capsule containment excess."""
    fid = rec.get("fidelity", {})
    key = "h0.850" if "h0.850" in fid else next(iter(fid))
    exc = fid[key]["excess_mm"]
    # one bar per physical part: the collision and visual meshes of a link are
    # the same object measured twice, so keep the larger and say so.
    merged = {}
    for name, v in exc.items():
        k = name.split(":", 1)[-1].replace("tool:", "")
        merged[k] = max(merged.get(k, -1e9), v)
    names = [n for n in LINK_ORDER if n in merged]
    names += [n for n in merged if n not in names]
    vals = [merged[n] for n in names]
    y = np.arange(len(names))[::-1]
    cols = [RED if v > 0.5 else BLUE for v in vals]
    ax.barh(y, vals, height=0.62, color=cols, linewidth=0)
    ax.axvline(0, color=BASE, lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([PRETTY.get(n, n) for n in names], fontsize=7.4)
    for yy, v in zip(y, vals):
        ax.text(v + (2.5 if v > 0 else -2.5), yy, f"{v:+.0f}",
                va="center", ha="left" if v > 0 else "right", fontsize=7,
                color=INK2)
    ax.set_xlim(min(vals) - 20, max(vals) + 30)
    ax.set_ylim(-0.9, len(names) - 0.1)
    _tidy(ax, xlabel="mm outside the capsule union   (+ = geometry the model "
                     "does not contain)",
          title="1  Every link of the arm sticks out of its own capsule",
          sub="worst over the certified drawing poses, the baked parks and "
              "gated random poses; mesh = collision shell ∪ visual shell")
    ax.text(0.99, 0.985, "OPTIMISTIC — the danger class", va="top",
            transform=ax.transAxes, ha="right", fontsize=7.8, color=RED,
            fontweight="bold")
    ax.text(0.012, 0.985, "contained", transform=ax.transAxes, ha="left",
            va="top", fontsize=7.8, color=BLUE, fontweight="bold")


def panel_hist(ax, rec):
    """2 — the signed error distribution over near pose pairs."""
    fid = rec.get("fidelity", {})
    allsamp = []
    for k, v in fid.items():
        if isinstance(v, dict) and "samples" in v:
            allsamp += v["samples"]
    err = 1000 * np.array([s["d_caps"] - s["d_mesh"] for s in allsamp])
    dc = 1000 * np.array([s["d_caps"] for s in allsamp])
    near = dc < 160
    lo, hi = np.percentile(err, 0.2), np.percentile(err, 99.8)
    bins = np.linspace(min(lo, -5), max(hi, 5), 61)
    for mask, col, lab, a in ((err <= 0, BLUE, "conservative (capsule under-"
                                                "states clearance)", 0.95),
                              (err > 0, RED, "OPTIMISTIC (capsule over-states "
                                             "clearance)", 0.95)):
        ax.hist(err[mask], bins=bins, color=col, alpha=a, linewidth=0)
    ax.axvline(0, color=BASE, lw=1.0)
    n_opt = int((err > 0.5).sum())
    ax.set_yscale("log")
    ax.set_ylim(top=ax.get_ylim()[1] * 14)
    fc = sum(v.get("n_false_clear", 0) for v in fid.values()
             if isinstance(v, dict))
    so = sum(v.get("n_signed_off", 0) for v in fid.values()
             if isinstance(v, dict))
    _tidy(ax, xlabel="capsule clearance − mesh clearance  (mm)",
          ylabel="pose pairs",
          title="2  ... so the pairwise error is two-sided",
          sub=f"{len(err)} near pose pairs (capsule clearance < 300 mm) "
              f"across both heights; {n_opt} of them optimistic")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE),
                       plt.Rectangle((0, 0), 1, 1, color=RED)],
              labels=["conservative — capsule ≤ mesh",
                      "OPTIMISTIC — capsule > mesh"],
              fontsize=7.4, frameon=False, loc="upper left",
              labelcolor=[INK2, INK2])
    if near.any():
        ax.text(0.99, 0.97,
                f"in the band that decides a schedule\n"
                f"(capsule < 160 mm, n = {int(near.sum())}):\n"
                f"median {np.median(err[near]):+.0f} mm   "
                f"worst {err[near].max():+.0f} mm\n\n"
                f"FALSE CLEAR — capsule ≥ 80 mm, mesh < 80 mm:\n"
                f"{fc} of {so} pairs the conductor would sign off",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.4,
                color=INK2, linespacing=1.55)


def panel_column(ax, rec):
    """3 — the measured base column against the two models that stand for it."""
    col = rec.get("column", {})
    prof = col.get("profiles", {})
    d1 = col.get("d1", 0.333)
    z = np.array(prof["column_all"]["z"])
    r = np.array(prof["column_all"]["r"])
    s = r > 0
    # one bar per z bin, not a filled polygon: the profile has bins with no
    # geometry in them at all, and a polygon would draw a fill straight
    # through those instead of leaving them empty
    dz = float(np.median(np.diff(z))) if len(z) > 1 else 0.005
    ax.barh(z[s], 1000 * r[s], height=dz * 1.02, color=BLUE, alpha=0.14,
            linewidth=0, zorder=1, align="center")
    ax.step(1000 * r[s], z[s], where="mid", color=BLUE, lw=2.2, zorder=4,
            label="MEASURED envelope: link0 ∪ link1 swept over q1")
    for key, c, lab, ls in (
            ("link0_all", ORANGE, "link0 alone, incl. connector + cable",
             (0, (5, 2))),
            ("link1_swept", AQUA, "link1 alone, swept over q1", (0, (2, 2)))):
        if key not in prof:
            continue
        zz = np.array(prof[key]["z"])
        rr = np.array(prof[key]["r"])
        m = rr > 0
        ax.step(1000 * rr[m], zz[m], where="mid", color=c, lw=1.5, label=lab,
                ls=ls, zorder=6)
    rc, rb = col.get("claimed_capsule_r", 0.09), col.get("claimed_box_r", 0.12)
    ax.plot([1000 * rc, 1000 * rc], [0, d1], color=RED, lw=2.0, ls="--",
            zorder=5,
            label=f"claimed capsule  r = {rc:.2f} m over base z [0, {d1:.3f}]")
    ax.plot([1000 * rb, 1000 * rb], [-rb, d1 + rb], color=CRIT, lw=1.3,
            ls=":", zorder=5,
            label=f"claimed box  r = {rb:.2f} m (AABB, ±r caps)")
    ax.axhline(0, color=BASE, lw=1.1, zorder=5)
    ax.text(198, -0.006, "mount plate  (base z = 0, world z = h)",
            fontsize=7.0, color=MUTED, va="bottom", ha="right")
    ax.set_ylim(0.62, -0.27)          # a blank strip at the bottom for the key
    _tidy(ax, xlabel="radius from the base z axis  (mm)",
          ylabel="base-frame z  (m)     ↓ = down, away from the plate",
          title="3  The base column is twice the radius it is modelled at",
          sub="a hanging arm's own body swept over the whole q1 range; "
              "positive z hangs below the plate, into the room")
    ax.legend(fontsize=7.0, frameon=False, loc="lower left",
              labelcolor=INK2, borderaxespad=0.9)
    ax.set_xlim(0, 200)


def panel_height(ax, rec):
    """4 — the decision: union coverage against mount height, per model."""
    hh = rec.get("height", {}).get("rows", {})
    if not hh:
        ax.text(0.5, 0.5, "part C not run yet", ha="center", va="center",
                transform=ax.transAxes, color=MUTED, fontsize=9)
        ax.axis("off")
        return
    hs = sorted(float(k[1:]) for k in hh)
    keys = [f"h{h:.3f}" for h in hs]
    models = [("union_air", BLUE, "empty air"),
              ("union_mounts", ORANGE, "published column, r = 0.09"),
              ("union_mesh_col", AQUA, "MEASURED column"),
              ("union_mesh", RED, "MEASURED column + grown capsules")]
    for key, c, lab in models:
        v = [hh[k][key] for k in keys]
        ax.plot(hs, v, "-o", color=c, lw=2.0, ms=4.5, mew=0, label=lab,
                zorder=3, clip_on=False)
    for h, tag in ((0.850, "adopted"), (0.922, "proposed")):
        if h in hs:
            ax.axvline(h, color=BASE, lw=1.0, ls="--", zorder=1)
            ax.text(h + 0.002, 100.72, f"{tag}\nh = {h:.3f}", fontsize=7.2,
                    color=MUTED, va="top")
    best = max(hs, key=lambda h: hh[f"h{h:.3f}"]["union_mesh"])
    ax.annotate(f"corrected best\nh = {best:.3f}   "
                f"{hh[f'h{best:.3f}']['union_mesh']:.2f} %",
                (best, hh[f"h{best:.3f}"]["union_mesh"]),
                textcoords="offset points", xytext=(6, -30), fontsize=7.4,
                color=RED, ha="left",
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.9))
    ax.set_xlim(0.845, 1.005)
    ax.set_ylim(92, 100.8)
    ax.set_xticks([0.85, 0.875, 0.90, 0.925, 0.95, 0.975, 1.00])
    _tidy(ax, xlabel="ceiling mount height h  (m)",
          ylabel="union strict-GO coverage  (% of the 2 cm canvas)",
          title="4  The height verdict survives — and the number moves up",
          sub="same fiber, same gates, same 80 mm margin as "
              "scripts/layout_rescore.py; only the occupancy model changes")
    ax.legend(fontsize=7.4, frameon=False, loc="lower right", labelcolor=INK2,
              borderaxespad=1.1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(ROOT / "out/collision_audit.json"))
    ap.add_argument("--png", default=str(ROOT / "out/collision_audit.png"))
    a = ap.parse_args()
    rec = json.loads(Path(a.json).read_text())

    fig, axes = plt.subplots(2, 2, figsize=(15.2, 11.8), facecolor=SURF)
    for ax in axes.flat:
        ax.set_facecolor(SURF)
        ax.grid(True, color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
    panel_excess(axes[0, 0], rec)
    panel_hist(axes[0, 1], rec)
    panel_column(axes[1, 0], rec)
    panel_height(axes[1, 1], rec)
    axes[0, 0].grid(axis="y", visible=False)
    axes[1, 1].grid(axis="x", visible=False)

    fig.suptitle("Collision audit — the schematic models against the meshes",
                 fontsize=13.5, x=0.008, y=0.988, ha="left", color=INK,
                 fontweight="bold")
    m = rec.get("meta", {})
    fig.text(0.008, 0.9635,
             "exact FCL BVH mesh distance on this repo's own FK (validated "
             "against drake to 6e-10 m); ground truth = franka's collision "
             "shells ∪ the full-res visual shells, "
             f"pen holder from raw CAD    ·    {m.get('when', '')}",
             fontsize=7.8, color=MUTED, ha="left")
    fig.tight_layout(rect=(0.004, 0.005, 0.996, 0.945), h_pad=5.2, w_pad=4.0)
    fig.savefig(a.png, dpi=125, facecolor=SURF)
    print("wrote", a.png)


if __name__ == "__main__":
    main()
