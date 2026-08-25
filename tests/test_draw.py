"""The generic front door's plumbing: ink count, fill/line split, end to end.

Run: pytest tests/test_draw.py    (fast: synthetic images, two arms, no drake)

`tests/test_csail.py` pins the CSAIL tracer on the CSAIL mark.  These pin the
three things `scripts/draw.py` adds on top of it, each on a picture built here
so the test says what it means rather than depending on a file:

  1. how many pens is this drawn with, and does saying so override the answer;
  2. is a stroke traced down its CENTRE and a filled shape round its EDGE —
     which is the tracer's only real judgement and the one that decides whether
     a picture reads or comes out as a stick figure;
  3. does a picture go all the way through trace -> place -> allocate ->
     conduct -> scene_check and out the other side as a certified programme.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from aris_sixarm import artwork, trace                          # noqa: E402


# ---------------------------------------------------------------------------
def _img(tmp, name, draw, size=(240, 240), bg=(255, 255, 255)):
    """Build a small RGB picture with `draw(arr)` and write it. -> path str."""
    a = np.zeros((size[1], size[0], 3), np.uint8)
    a[:] = bg
    draw(a)
    p = Path(tmp) / name
    Image.fromarray(a).save(p)
    return str(p)


def _one_ink(a):
    a[40:46, 20:220] = 0                     # a 6 px horizontal rule
    a[150:156, 20:220] = 0
    a[20:220, 110:116] = 0                   # and a vertical one through them


def _two_ink(a):
    _one_ink(a)
    a[90:96, 20:220] = (203, 102, 8)         # a second pen, unmistakably orange


# ---------------------------------------------------------------------------
def test_ink_count_is_measured_and_can_be_overridden(tmp_path):
    """`--inks auto` counts the pens; `--inks N` is obeyed but never invents one.

    The count is taken off INTERIOR pixels (`trace.ink_core`).  Off the raw ink
    mask the antialiased rim of every black line supplies a continuum of greys
    and k-means happily reports three inks in a one-pen drawing — which then
    cuts every stroke into interleaved fragments, because `ink_masks` gives each
    pixel to whichever of the three it is nearest.  That is the failure this
    test exists to keep out.
    """
    one = _img(tmp_path, "one.png", _one_ink)
    two = _img(tmp_path, "two.png", _two_ink)

    rgb, bg = trace.load_art(one)
    assert bg == (255, 255, 255)
    assert len(trace.detect_inks(rgb, bg)) == 1, "one pen, one ink"

    rgb2, bg2 = trace.load_art(two)
    pal2 = trace.detect_inks(rgb2, bg2)
    assert len(pal2) == 2, f"two pens, {len(pal2)} inks: {pal2}"
    assert sum(pal2[0]) < sum(pal2[1]), "palette is darkest first"
    assert max(abs(c - r) for c, r in zip(pal2[1], (203, 102, 8))) < 30, \
        f"the orange pen was not recovered: {pal2[1]}"

    # asking for MORE inks than the picture has must not shred the one it has
    assert len(trace.detect_inks(rgb, bg, n=3)) == 1, \
        "--inks 3 on a one-pen drawing must collapse back to one"
    # and asking for fewer is obeyed
    assert len(trace.detect_inks(rgb2, bg2, n=1)) == 1

    px, dbg = trace.trace_art(two)
    assert dbg["names"] == ["black", "orange"]
    assert {s["color"] for s in px} == {"black", "orange"}
    assert artwork.inks_of(px) == ["black", "orange"]


def test_a_line_is_traced_down_its_middle_and_a_fill_round_its_edge(tmp_path):
    """The tracer's one judgement, on a picture that is half of each.

    A pen has one width, so a uniform-width stroke wants its CENTRELINE: one
    polyline about as long as the stroke.  A solid shape has no centreline worth
    drawing — its skeleton is a stick figure — so it wants its BOUNDARY: a
    closed loop about as long as the perimeter.  Getting the two the wrong way
    round is silent and ruins the drawing, so both directions are asserted.
    """
    def draw(a):
        a[30:36, 20:220] = 0                 # 6 px line, 200 px long
        a[120:200, 60:180] = 0               # 120 x 80 solid block

    p = _img(tmp_path, "mixed.png", draw, size=(240, 240))
    px, dbg = trace.trace_art(p, n_inks=1, work_px=240)
    d = dbg["per_ink"]["black"]
    assert d["fill_erode"] >= 3
    # the block is a fill and the line is not
    assert d["thick"][160, 120], "the solid block was not called a fill"
    assert not d["thick"][33, 120], "the line was called a fill"

    lines = [s for s in px if s["kind"] == "outline"]
    fills = [s for s in px if s["kind"] == "fill"]
    assert len(lines) == 1, f"{len(lines)} centrelines, want 1"
    assert len(fills) == 1, f"{len(fills)} boundaries, want 1"

    L = trace.plen(lines[0]["pts"])
    assert 180 < L < 215, f"centreline is {L:.0f} px, want ~200 (not ~410 = twice)"
    F = trace.plen(fills[0]["pts"])
    assert 350 < F < 440, f"boundary is {F:.0f} px, want ~400 = the perimeter"
    assert np.allclose(fills[0]["pts"][0], fills[0]["pts"][-1]), \
        "a fill's boundary must close"


def test_a_vector_source_arrives_in_the_same_shape_as_a_raster(tmp_path):
    """An SVG is FLATTENED, not thinned, and comes out as the same stroke set.

    The point of the vector branch is that everything downstream — `to_sheet`,
    the placement search, the allocator, the payload — cannot tell which door
    the strokes came in through.  So the assertions are about that shape: pixel
    (x, y) polylines, one ink name per distinct paint, long side `work_px`.
    """
    svg = tmp_path / "v.svg"
    svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200" '
        'viewBox="0 0 200 200"><g transform="translate(10,10) scale(0.9)">'
        '<path d="M 10 10 L 190 10 L 190 190 L 10 190 Z" fill="none" '
        'stroke="#000000"/>'
        '<path d="M 20 100 C 60 20, 140 20, 180 100" fill="none" stroke="red"/>'
        '<circle cx="100" cy="100" r="30" fill="none" stroke="#0000ff"/>'
        '<path d="M 40 60 a 25 25 0 1 1 50 0" fill="none" stroke="black"/>'
        "</g></svg>")

    px, dbg = trace.trace_any(str(svg), work_px=900)
    assert dbg["names"] == ["black", "red", "blue"], dbg["names"]
    assert {s["color"] for s in px} == {"black", "red", "blue"}
    assert len(px) == 4, f"{len(px)} strokes, want one per element"

    P = np.vstack([s["pts"] for s in px])
    assert P.shape[1] == 2 and P.min() >= -1e-6
    assert abs(max(P[:, 0].max(), P[:, 1].max()) - 900) < 1.0, \
        "the long side must be scaled to work_px, like the raster branch"

    # the curves are really flattened, not left as two-point chords
    curve = next(s for s in px if s["color"] == "red")
    assert len(curve["pts"]) > 8, "a cubic came through as a straight line"
    circ = next(s for s in px if s["color"] == "blue")
    assert len(circ["pts"]) > 12 and trace.plen(circ["pts"]) > 700, \
        "the circle was not flattened"

    # and it maps onto the paper exactly as a raster trace does
    from aris_sixarm.fleet import SHEET
    st, info = trace.to_sheet(px, SHEET, margin=0.06)
    assert len(st) == 4 and info["fits"]
    assert artwork.inks_of(st) == ["black", "red", "blue"]


def test_a_picture_goes_end_to_end_and_comes_out_certified(tmp_path):
    """Trace -> place -> allocate -> conduct -> scene_check, on a tiny drawing.

    Two arms and four short strokes, so it is seconds rather than the minutes a
    real picture costs — but it is the SAME `run_allocation` and the same
    `build_phases`, so it fails if the front door stops handing the pipeline
    what the pipeline expects: pre-traced pixel strokes, a fixed one-ink colour
    map, a palette for the payload, and an `a.out` that is still a directory.
    """
    import argparse
    from csail_allocate import run_allocation, totals
    from csail_schedule import build_phases, payload, summary_json
    from aris_sixarm import fleet as fleet_mod

    def draw(a):
        for y in (60, 110, 160):
            a[y:y + 5, 60:180] = 0
        a[60:165, 115:120] = 0

    src = _img(tmp_path, "smoke.png", draw, size=(240, 240))
    px, dbg = trace.trace_art(src, n_inks=1, work_px=240)
    assert px and artwork.inks_of(px) == ["black"]

    ap = argparse.ArgumentParser()
    import csail_allocate as ca
    import csail_schedule as cs
    cs.schedule_args(ca.add_args(ap))
    a = ap.parse_args(["--arms", "13,31", "--no-prefilter", "--no-balance",
                       "--no-verify", "--target-width", "0.30",
                       "--offset", "0.0", "-0.30", "--fps", "8",
                       "--substeps", "2", "--max-probes", "2"])
    a.out = str(tmp_path)
    a.image = src
    a.name = "smoke"
    a.palette = artwork.palette_of(dbg)
    assert a.palette["black"].startswith("#")

    phases, strokes, info = run_allocation(a, verbose=False, px=px)
    assert len(phases) == 1, "one ink must be one pass"
    assert phases[0]["ink"] == "black", "the ink map must be fixed, not partitioned"
    assert all(v == "black" for v in phases[0]["colors"].values())
    T = totals(phases)
    assert T["drawn"] > 0.5 * T["traced"], \
        f"only {100 * T['covered']:.0f} % allocated on a tiny central drawing"

    dt = 1.0 / (a.fps * a.substeps)
    pens = {aid: next((p["pens"][aid] for p in phases if aid in p["pens"]), 0.110)
            for aid in fleet_mod.FLEET}
    built = build_phases(a, phases, dt, pens)
    assert len(built) == 1
    rep = built[0]["rep"]
    assert rep["ok"], f"scene_check refused the smoke run: {rep}"
    assert rep["segments_failed"] == 0 and rep["monotone"]

    npz = tmp_path / "smoke_schedule.npz"
    nF, nInk, M_tot, n_pause = payload(built, dt, pens, a, npz)
    assert npz.exists() and nF > 1 and nInk > 0
    d = np.load(npz, allow_pickle=False)
    assert [str(x) for x in d["ink_names"]] == ["black"], \
        "the payload must carry the picture's own ink names for the animation"
    assert str(d["ink_palette"][0]).startswith("#")
    assert set(str(x) for x in d["ink_hex"]) == {a.palette["black"]}

    s = summary_json(a, phases, strokes, info, built, dt, pens, None, nF, nInk,
                     n_pause)
    assert s["inks"] == ["black"] and s["name"] == "smoke"
    assert s["coverage"] == pytest.approx(T["covered"])
    assert s["makespan_s"] > 0 and s["min_clearance"] >= s["margin"] - 1e-12


def test_allocating_the_profiles_in_parallel_changes_only_the_clock(tmp_path):
    """`--profile-jobs` is a wall-clock knob, and this is what makes that true.

    The four execution profiles are independent and each is a pure function of
    `(args, profile)`, so mapping them over processes must return the same grid
    — the same floors, the same prune decisions, the same shipped makespan —
    as running them one after another.  It is worth having because on a dense
    picture the premise of `docs/BENCH.md` inverts: `balance_loads`' split
    search is 677 s of a 724 s allocation on the Trollface's 63 segments, so
    the four cells are the run and they were strictly serial.
    """
    import argparse
    import csail_allocate as ca
    import csail_schedule as cs

    def draw(a):
        for y in (60, 110, 160):
            a[y:y + 5, 60:180] = 0
        a[60:165, 115:120] = 0

    src = _img(tmp_path, "grid.png", draw, size=(240, 240))
    px, dbg = trace.trace_art(src, n_inks=1, work_px=240)

    def args(jobs):
        ap = argparse.ArgumentParser()
        cs.schedule_args(ca.add_args(ap))
        a = ap.parse_args(["--arms", "13,31", "--no-prefilter", "--no-verify",
                           "--target-width", "0.30", "--offset", "0.0", "-0.30",
                           "--fps", "8", "--substeps", "2", "--max-probes", "2",
                           "--select-profile"])
        a.out, a.image, a.name = str(tmp_path), src, "grid"
        a.palette, a.traced_px, a.profile_jobs = artwork.palette_of(dbg), px, jobs
        return a

    def summary(sel):
        return (sel["chosen"]["profile"], round(sel["chosen"]["makespan_s"], 9),
                tuple((r["profile"], r["status"],
                       None if r["floor_s"] is None else round(r["floor_s"], 9))
                      for r in sel["grid"]))

    serial = summary(cs.build(args(1))[6])
    parallel = summary(cs.build(args(4))[6])
    assert serial == parallel, f"\nserial   {serial}\nparallel {parallel}"
    # and the grid really did prune on floors rather than conducting all four
    assert any(s == "pruned" for _, s, _ in serial[2])
    assert any(s == "certified" for _, s, _ in serial[2])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
