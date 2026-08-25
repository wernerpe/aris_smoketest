#!/usr/bin/env python3
"""Draw ANY picture with the six-arm rig: trace -> place -> allocate -> conduct.

    ARIS_RIG=final6_opt python3 scripts/draw.py PICTURE [--inks auto|N]
        [--placement auto|off|FILE.json] [--out NAME]

    # the Trollface, end to end, at the shipped settings
    ARIS_RIG=final6_opt python3 scripts/draw.py \
        assets/artworks/trollface/trollface.png --inks auto --placement auto \
        --out trollface --title "the Trollface" --select-profile --program \
        --fps 12 --substeps 4 --draw-speed 0.15 --transit-speed 0.30

ONE FRONT DOOR OVER THE PIPELINE THAT ALREADY EXISTS.  Nothing here plans,
allocates, sequences, conducts or checks anything of its own: every stage is
the one the CSAIL scripts use, called on a picture instead of on that logo.

  1. TRACE      `trace.trace_any` — a raster is colour-clustered to the ink
                count (`--inks`), thinned, and walked THROUGH its junctions so
                a crossing leaves as two strokes rather than four fragments;
                solid regions are traced as their boundary instead
                (`trace.trace_art`).  An SVG is flattened instead of thinned
                (`trace.trace_svg`).  Either way what comes out is the pixel
                stroke set `trace.to_sheet` expects.
  2. PLACE      `artwork.search_placement` — the atlas proxy ranks the
                rotation x scale x translation grid, the top few of every cell
                are ALLOCATED FOR REAL, and the largest placement within
                `--slack` of the best real coverage wins.  `--placement off`
                uses `--target-width/--offset/--rotate` as given instead.
  3. ALLOCATE   `csail_allocate.run_allocation` — probe, cover, repair, balance,
                split, merge, sequence.  ONE ink means one fixed colour map and
                no pen swap; two or more can be drawn in one pass (the colour
                partition is enumerated) or in two with `--two-pass`.
  4. CONDUCT    `csail_schedule.build` — freeze the per-arm timelines, conduct
                them under the idle policy, and (with `--select-profile`) do it
                for four execution profiles and ship the fastest that certifies.
  5. CHECK      `scene_check` has the veto, inside `build_phase`.  Nothing is
                written unless the paper gate, the frame gate and every
                inter-arm pair pass.
  6. RENDER     `csail_drawing_demo.py` under the station venv, because it needs
                pydrake and the rest of this needs the system python's batch IK.

Everything is named by `--out`: out/NAME_trace.png, _strokes.json,
_placement.{json,png}, _program.json, _schedule.{npz,json}, _allocation.png,
_final.png, NAME.html and NAME.zip.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from aris_sixarm import allocate, artwork, trace              # noqa: E402
from aris_sixarm.fleet import FLEET, SHEET                    # noqa: E402
from csail_allocate import (add_args, allocation_png, final_png,   # noqa: E402
                            program_json, totals)
from csail_schedule import (build, payload, profile_json,     # noqa: E402
                            schedule_args, summary_json)
from csail_trace import sheet_axes                            # noqa: E402

VENV = "/home/franka/git/franka_manipulation_station/.venv/bin/python"


# ---------------------------------------------------------------------------
def trace_png(px, dbg, path, name, source):
    """What the tracer saw and what it decided. -> writes a 4-panel figure.

    LOOK AT THIS BEFORE ANYTHING ELSE.  Every later stage is expensive and none
    of them can tell you that the picture came out as confetti; this can, and
    the two middle panels say WHY — which pixels were called a line and which a
    fill, and what the skeleton of the line half looks like.
    """
    pal = artwork.palette_of(dbg)
    names = list(dbg.get("names") or [])
    H, W = dbg.get("shape", (1, 1))
    fig, ax = plt.subplots(2, 2, figsize=(15.5, 12.5))
    if dbg.get("rgb") is not None:
        ax[0, 0].imshow(dbg["rgb"])
        ax[0, 0].set_title(f"source at the {W} x {H} working resolution "
                           f"(background {dbg.get('bg')})", fontsize=10)
    else:
        ax[0, 0].text(0.5, 0.5, "vector source — nothing was rasterised",
                      ha="center", va="center")
        ax[0, 0].set_title("source", fontsize=10)

    masks = dbg.get("masks")
    if masks is not None and len(masks):
        rgb = np.ones((H, W, 3))
        for i, nm in enumerate(names):
            thick = dbg["per_ink"][nm].get("thick")
            c = np.array([int(pal[nm][j:j + 2], 16) / 255.0 for j in (1, 3, 5)])
            rgb[np.asarray(masks[i], bool)] = 0.45 + 0.55 * c
            if thick is not None:
                rgb[np.asarray(thick, bool)] = c * 0.8
        ax[0, 1].imshow(rgb)
        ax[0, 1].set_title("unmixed ink masks; DARK = traced as a fill "
                           "(boundary), PALE = as a line (centreline)",
                           fontsize=10)
        sk = np.ones((H, W, 3))
        any_sk = False
        for nm in names:
            s = dbg["per_ink"][nm].get("skel")
            if s is not None:
                sk[np.asarray(s, bool)] = [0.15, 0.15, 0.18]
                any_sk = True
        ax[1, 0].imshow(sk)
        ax[1, 0].set_title("Zhang-Suen skeletons of the line masks"
                           if any_sk else "no line mask (everything is a fill)",
                           fontsize=10)
    else:
        for k in (ax[0, 1], ax[1, 0]):
            k.text(0.5, 0.5, "vector input: no masks, no skeleton",
                   ha="center", va="center")

    for k, s in enumerate(px):
        p = np.asarray(s["pts"], float)
        ax[1, 1].plot(p[:, 0], p[:, 1], "-", lw=1.15,
                      color=pal.get(s["color"], "#222222"))
        ax[1, 1].plot(p[0, 0], p[0, 1], ".", ms=3.2, color="#d62728")
    ax[1, 1].set_xlim(0, W)
    ax[1, 1].set_ylim(H, 0)
    ax[1, 1].set_aspect("equal")
    L = sum(trace.plen(s["pts"]) for s in px)
    ax[1, 1].set_title(f"{len(px)} traced strokes, {L:.0f} px of path "
                       f"(red dots = stroke starts)", fontsize=10)
    for k in ax.ravel():
        k.set_xticks([])
        k.set_yticks([])
    fig.suptitle(f"{name} — traced from {Path(source).name}: "
                 + ", ".join(f"{nm} {pal[nm]}" for nm in names), fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def sheet_png(strokes, info, path, name, palette):
    """The traced strokes at PAPER scale, on the canvas, with the fleet."""
    h = 9.4 * SHEET[1] / SHEET[0]
    fig, ax = plt.subplots(figsize=(max(6.4, 9.4 if h <= 12.5 else
                                        12.5 * SHEET[0] / SHEET[1]),
                                    min(12.5, max(4.2, h))))
    sheet_axes(ax)
    for s in strokes:
        p = np.asarray(s["pts"], float)
        ax.plot(p[:, 0], p[:, 1], "-", lw=2.2,
                color=artwork.hex_of(s["color"], palette),
                solid_capstyle="round", solid_joinstyle="round", zorder=3)
    for aid, spec in FLEET.items():
        ax.plot(*spec.xy, marker="o" if spec.mount == "floor" else "s", ms=12,
                mfc=spec.color if spec.active else "white", mec=spec.color,
                mew=2.0, zorder=6, clip_on=False)
        ax.annotate(str(aid), spec.xy, color="white" if spec.active else spec.color,
                    fontsize=7, fontweight="bold", ha="center", va="center",
                    zorder=7, clip_on=False)
    L = trace.total_length(strokes)
    ax.set_title(f"{name} — {len(strokes)} pen strokes, {L:.2f} m of path\n"
                 f"{info['logo_w']:.3f} x {info['logo_h']:.3f} m"
                 + (f", turned {info['rotate_deg']:.0f} deg"
                    if info.get("rotate_deg") else "")
                 + f" on the {SHEET[0]:.3f} x {SHEET[1]:.3f} m canvas",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def placement_png(doc, path, name, arms, slack, radius):
    """Coverage against SIZE IN SQUARE METRES, one pair of curves per rotation.

    The same figure `scripts/csail_place.py` writes, and the same reason for
    the x axis: the identical "scale" is a different drawing in each rotation,
    so a scale axis would compare unlike things.
    """
    per_cell = {(round(r["rot"], 6), round(r["f"], 6)): r
                for r in doc["per_scale"].values()}
    pick, grid = doc["chosen"], doc["proxy"]
    best = max(r["cov"] for r in doc["real"])
    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    cols = {r: c for r, c in zip(doc["rotations"],
                                 ["#cb6608", "#1f77b4", "#2ca02c", "#9467bd"])}
    # MATCH ON THE CELL'S OWN `f`, NEVER ON THE ROUNDED KEY.  The key is
    # `round(f, 6)` and the grid rows carry the raw `np.linspace` value, so a
    # scale that is not exact to six decimals — which every `--scales` whose
    # step is not a round number produces — matches nothing and `max()` is
    # handed an empty iterable.
    def proxy_at(cell, rot):
        hits = [g["proxy"] for g in grid
                if abs(g["f"] - cell["f"]) < 1e-9 and abs(g["rot"] - rot) < 1e-9]
        return 100 * max(hits) if hits else float("nan")

    for rot in doc["rotations"]:
        ks = sorted([k for k in per_cell if abs(k[0] - rot) < 1e-9])
        ar = [per_cell[k]["logo_w"] * per_cell[k]["logo_h"] for k in ks]
        ax.plot(ar, [proxy_at(per_cell[k], rot) for k in ks],
                "o--", ms=4, color=cols[rot], alpha=0.45,
                label=f"{rot:.0f}$\\degree$ atlas proxy (upper bound, r={radius} m)")
        ax.plot(ar, [100 * per_cell[k]["cov"] for k in ks], "o-", lw=2.2,
                color=cols[rot],
                label=f"{rot:.0f}$\\degree$ real allocation (certified)")
    ax.axhline(100 * (best - slack), color="#666665", lw=1.0, ls=":",
               label=f"best - {100 * slack:.0f} pp")
    ax.plot([pick["logo_w"] * pick["logo_h"]], [100 * pick["coverage"]], "*",
            ms=18, color=cols[pick["rotate_deg"]], mec="black", zorder=5,
            label="chosen")
    ax.set_xlabel("drawing area on the paper  [m$^2$]")
    ax.set_ylabel("% of traced length drawn")
    ax.set_title(f"{name}: coverage vs size and rotation, best translation per "
                 f"cell\n{len(arms)} arms, canvas {SHEET[0]:.3f} x "
                 f"{SHEET[1]:.3f} m", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def strokes_json(strokes, info, path, palette):
    with open(path, "w") as f:
        json.dump(dict(sheet=[float(SHEET[0]), float(SHEET[1])],
                       palette=palette,
                       info={k: (list(v) if isinstance(v, (tuple, list))
                                 else bool(v) if isinstance(v, bool) else float(v))
                             for k, v in info.items()},
                       n_strokes=len(strokes),
                       total_length=trace.total_length(strokes),
                       strokes=[dict(id=s["id"], color=s["color"], kind=s["kind"],
                                     length=trace.plen(s["pts"]),
                                     pts=np.round(s["pts"], 5).tolist())
                                for s in strokes]), f)


# ---------------------------------------------------------------------------
def parse_args(argv=None):
    # `conflict_handler="resolve"` so that the two shared arg sets can be
    # inherited whole and the three names this front door means differently
    # (--image becomes a positional, --out becomes a NAME, --rotate becomes a
    # LIST of rotations to search) can be redefined without forking either list.
    ap = argparse.ArgumentParser(
        conflict_handler="resolve",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(ap)
    schedule_args(ap)
    ap.add_argument("source", help="a raster (png/jpg/gif/...) or an .svg")
    ap.add_argument("--out", default=None,
                    help="output BASENAME under --outdir (default: the source "
                         "file's stem)")
    ap.add_argument("--outdir", default=str(ROOT / "out"))
    ap.add_argument("--title", default=None,
                    help="what the figures and the animation legend call it")
    # ---- trace ----------------------------------------------------------
    ap.add_argument("--inks", default="auto",
                    help="'auto' (default) measures the ink count off the "
                         "picture's own interior pixels, or give an integer.  "
                         "One ink means one pen for the whole fleet and no swap")
    ap.add_argument("--work-px", type=int, default=trace.WORK_PX,
                    help="long side of the working raster.  Every tracer "
                         "constant is a length in THESE pixels, so this is the "
                         "knob that decides how much detail survives")
    ap.add_argument("--fill-erode", type=int, default=None,
                    help="erosions a LINE must not survive; a region that does "
                         "is traced as its boundary instead of its centreline.  "
                         "Default: measured (trace.auto_fill_erode)")
    ap.add_argument("--fill-frac", type=float, default=trace.FILL_FRAC)
    ap.add_argument("--rdp", type=float, default=trace.RDP_TOL,
                    help="polyline simplification tolerance, working pixels")
    ap.add_argument("--min-px", type=float, default=trace.ART_MIN_PX,
                    help="working pixels; a shorter polyline is a speck")
    ap.add_argument("--min-len", type=float, default=0.025,
                    help="metres; a shorter stroke ON THE PAPER is dropped by "
                         "`trace.to_sheet` — the confetti gate that matters, "
                         "because it is the one measured at drawing scale")
    ap.add_argument("--no-bridge", action="store_true",
                    help="do not rejoin strokes whose ends face each other "
                         "across a small gap")
    ap.add_argument("--trace-only", action="store_true",
                    help="write the trace figure and stop.  This is the loop "
                         "to iterate the tracer knobs in; everything after it "
                         "costs minutes")
    # ---- placement ------------------------------------------------------
    ap.add_argument("--placement", default="auto",
                    help="'auto' (default) runs the two-stage search, 'off' "
                         "uses --target-width/--offset/--rotate as given, or a "
                         "path to a placement JSON to reuse one")
    ap.add_argument("--rotate", default="0,90",
                    help="rotations searched, in degrees.  With --placement off "
                         "the FIRST one is used as the fixed rotation")
    ap.add_argument("--scales", type=float, nargs=3, default=(0.4, 1.0, 13),
                    metavar=("LO", "HI", "N"))
    ap.add_argument("--search-offset", type=float, default=0.30,
                    help="+-metres of translation searched")
    ap.add_argument("--offset-step", type=float, default=0.10)
    ap.add_argument("--top", type=int, default=3,
                    help="translations allocated for real per rotation x scale")
    ap.add_argument("--slack", type=float, default=0.01,
                    help="coverage a bigger placement may give up (fraction)")
    ap.add_argument("--radius", type=float, default=0.03,
                    help="atlas proxy reach radius, m")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--place-only", action="store_true",
                    help="write the placement JSON and stop.  The search is "
                         "minutes and the conduct after it is tens of minutes, "
                         "so this is where to look at the coverage-vs-size "
                         "curve before paying for the rest")
    # ---- animation ------------------------------------------------------
    ap.add_argument("--no-anim", action="store_true",
                    help="stop after the schedule; do not render the animation")
    ap.add_argument("--venv", default=VENV,
                    help="python that has pydrake, for the animation only")
    ap.add_argument("--budget", type=float, default=28.0,
                    help="MiB, the zipped animation's size cap")
    a = ap.parse_args(argv)
    a.image = a.source
    a.name = a.title or Path(a.source).stem.replace("_", " ")
    a.stem = a.out or Path(a.source).stem
    a.out = str(Path(a.outdir))       # every shared stage reads `a.out` as a DIR
    if a.atlas is None:
        a.atlas = a.out
    return a


# ---------------------------------------------------------------------------
def main(argv=None):
    a = parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    P = lambda ext: out / f"{a.stem}{ext}"                    # noqa: E731
    t_all = time.time()

    # ---- 1. trace ------------------------------------------------------
    t0 = time.time()
    n_inks = None if str(a.inks).lower() in ("auto", "", "none") else int(a.inks)
    px, dbg = trace.trace_any(
        a.source, n_inks=n_inks, work_px=a.work_px, rdp_tol=a.rdp,
        min_px=a.min_px, fill_erode=a.fill_erode, fill_frac=a.fill_frac,
        bridge=not a.no_bridge)
    if not px:
        raise SystemExit(f"{a.source}: nothing traced — is it blank, or is the "
                         "background not what --work-px sees at the border?")
    palette = artwork.palette_of(dbg)
    a.palette = palette
    names = list(dbg.get("names") or [])
    print(f"traced {len(px)} strokes from {a.source} in {time.time() - t0:.1f} s")
    print(f"  {len(names)} ink(s): " + ", ".join(
        f"{nm} {palette[nm]}"
        + (f" (fill-erode {dbg['per_ink'][nm]['fill_erode']}, line half-width "
           f"{dbg['per_ink'][nm].get('halfwidth', 0) or 0:.1f} px)"
           if nm in dbg.get("per_ink", {}) else "") for nm in names))
    per_kind = {}
    for s in px:
        per_kind[s["kind"]] = per_kind.get(s["kind"], 0) + 1
    print(f"  {per_kind.get('outline', 0)} centreline + {per_kind.get('fill', 0)} "
          f"boundary strokes, {sum(trace.plen(s['pts']) for s in px):.0f} px of path")
    trace_png(px, dbg, P("_trace.png"), a.name, a.source)
    print(f"  wrote {P('_trace.png')}  <- LOOK AT THIS")
    if a.trace_only:
        return dict(px=px, dbg=dbg)

    # ---- 2. placement --------------------------------------------------
    arms = allocate.active_arms(_arms(a.arms))
    rots = [float(x) for x in str(a.rotate).split(",") if x.strip()]
    inks = artwork.inks_of(px)
    if a.placement == "auto":
        print(f"\n=== placement search ({len(rots)} rotations, "
              f"{int(a.scales[2])} scales) ===")
        t0 = time.time()
        doc = artwork.search_placement(
            px, arms, a.atlas, margin=a.margin, radius=a.radius,
            scales=tuple(a.scales), offset=a.search_offset,
            offset_step=a.offset_step, rotations=rots, top=a.top,
            slack=a.slack, jobs=a.jobs, min_len=a.min_len,
            single_ink=(inks[0] if len(inks) == 1 else None),
            # THE SEARCH IS A RANKING, so it pays for the one thing it ranks on
            # and nothing else.  `balance` only moves spans a second arm already
            # certified at the same endpoints, and `split` re-measures coverage
            # and is invariant by construction: neither can change `cov`, and on
            # this picture the two of them are 677 s of a 724 s allocation.  The
            # probe budget and the SEQUENCER are both kept at the run's own — the
            # budget because it is the one knob that does move coverage, the
            # sequencer because `prune_unflyable` bans the spans whose tour is
            # infeasible and a worse tour therefore bans more, and because it
            # costs 0.9 s of the 724.
            alloc_kw=dict(balance=False, split=False,
                          sequencer=getattr(a, "sequencer", "opt"),
                          max_probes=a.max_probes,
                          merge=not a.no_merge))
        print(f"  searched in {time.time() - t0:.1f} s")
        P("_placement.json").write_text(json.dumps(doc))
        placement_png(doc, P("_placement.png"), a.name, arms, a.slack, a.radius)
        a.placement = str(P("_placement.json"))
        print(f"  wrote {P('_placement.json')}, {P('_placement.png')}")
        if a.place_only:
            return doc
    elif a.placement in ("off", "none", "fixed"):
        a.placement = None
        a.rotate = rots[0]
    else:
        doc = json.loads(Path(a.placement).read_text())
        print(f"\nreusing the placement in {a.placement}: {doc['chosen']}")
    if a.placement:
        a.rotate = None            # let run_allocation take it from the file

    # ---- 3+4+5. allocate, conduct, check -------------------------------
    a.traced_px = px
    print("\n=== allocate, conduct, check ===")
    phases, strokes, info, built, dt, pens, sel = build(a)
    prof = profile_json(sel) if sel else None
    if prof:
        a.qd_frac, a.cluster = prof["qd_frac"], prof["cluster"]
        a.band_objective = prof["band_objective"]

    nF, nInk, M_tot, n_pause = payload(built, dt, pens, a, P("_schedule.npz"))
    doc = program_json(phases, strokes, info, P("_program.json"),
                       palette=palette, name=a.name, source=a.source)
    if prof:
        doc["profile"] = prof
        P("_program.json").write_text(json.dumps(doc))
    summary = summary_json(a, phases, strokes, info, built, dt, pens, prof,
                           nF, nInk, n_pause)
    P("_schedule.json").write_text(json.dumps(summary, indent=1))
    sheet_png(strokes, info, P("_sheet.png"), a.name, palette)
    strokes_json(strokes, info, P("_strokes.json"), palette)
    allocation_png(phases, strokes, P("_allocation.png"), name=a.name)
    final_png(phases, strokes, P("_final.png"), palette=palette, name=a.name)

    T = totals(phases)
    print(f"\n=== {a.name} ===")
    print(f"  {info['logo_w']:.3f} x {info['logo_h']:.3f} m"
          + (f", turned {info['rotate_deg']:.0f} deg"
             if info.get("rotate_deg") else "")
          + f" centred at ({info['center'][0]:.4f}, {info['center'][1]:.4f})")
    print(f"  {T['traced']:.4f} m traced, {T['drawn']:.4f} m drawn, "
          f"{T['dropped']:.4f} m in {T['n_dropped']} spans left empty "
          f"-> COVERAGE {100 * T['covered']:.4f} % over {T['n_segments']} "
          "certified segments")
    print(f"  makespan {summary['makespan_s']:.3f} s, "
          f"{summary['pause_total']:.1f} s of conducted pause, min inter-arm "
          f"clearance {1000 * summary['min_clearance']:.1f} mm "
          f"(margin {1000 * summary['margin']:.0f} mm)")
    for ph in summary["phases"]:
        print(f"    {ph['name']}: {ph['duration_s']:.3f} s, scene_check "
              f"{'PASS' if ph['scene_check_ok'] else 'FAIL'}, frame-failed "
              f"{ph['frame_failed'] or 'none'}, paper-failed "
              f"{ph['paper_failed'] or 'none'}")
        for x, m in sorted(ph["arm_metres"].items(), key=lambda kv: -kv[1]):
            if ph["arm_segments"][x]:
                print(f"      arm {x:>2} ({FLEET[int(x)].mount:>5}): {m:6.3f} m "
                      f"in {ph['arm_segments'][x]:>2} segments, "
                      f"{ph['arm_draw_s'][x]:5.1f} s drawing + "
                      f"{ph['arm_transit_s'][x]:5.1f} s pen-up")
    print(f"  wrote {P('_program.json')}, {P('_schedule.json')}, "
          f"{P('_schedule.npz')}, {P('_allocation.png')}, {P('_final.png')}, "
          f"{P('_sheet.png')}, {P('_strokes.json')}")

    # ---- 6. animation --------------------------------------------------
    if not a.no_anim:
        cmd = [a.venv, str(ROOT / "scripts/csail_drawing_demo.py"),
               "--schedule", str(P("_schedule.npz")),
               "--summary", str(P("_schedule.json")),
               "--out", str(P(".html")), "--zip", str(P(".zip")),
               "--budget", str(a.budget), "--title", a.name]
        print(f"\n=== animation ===\n  {' '.join(cmd)}")
        if not Path(a.venv).exists():
            print(f"  !! {a.venv} does not exist; skipping the render.  Run the "
                  "command above under a python that has pydrake.")
        else:
            subprocess.run(cmd, check=True, cwd=str(ROOT))
    print(f"\nDRAW WALL CLOCK {time.time() - t_all:.1f} s")
    return summary


def _arms(arg):
    if arg is None:
        return None
    return arg if arg == "all" else [int(x) for x in str(arg).split(",")]


if __name__ == "__main__":
    main()
