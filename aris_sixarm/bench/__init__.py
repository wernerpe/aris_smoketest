"""Five generated drawings the whole pipeline is measured on, none of them CSAIL.

WHY THIS EXISTS.  Every number this repository has published — 153.5 s, 108.6 s,
105.4 s, 99.2 % — is one picture at one placement.  A load balancer that learns
to cut the CSAIL logo in exactly the two places the CSAIL logo needs cutting is
indistinguishable, from inside that measurement, from one that has understood
anything at all.  So these five drawings are written to stress the pipeline's
STRUCTURE and are deliberately not tuned to the logo: the placement is fixed and
logo-independent, the geometry is analytic, and the two that use randomness use
a fixed seed and nothing else.

  hatch      ~30 m of dense parallel hatching packed into ONE arm's territory.
             The worst case for a whole-segment balancer: everything is
             reachable by the arm it sits on top of, almost nothing is reachable
             end to end by anybody else, and the phase floors at one arm's ink
             unless the allocator can cut.
  scatter    ~7 m of short strokes thrown across the WHOLE sheet.  The opposite
             regime — nothing is worth splitting (every stroke is near the 5 cm
             floor) and the clock is almost all pen-up, so this is the one that
             catches a splitter that cuts because it can rather than because it
             helps.  It also runs into the corners, where coverage is a real
             question.
  starburst  24 rays out of the middle of the sheet.  The middle is the WAIST
             between the four inverted bases, the place `docs/DEAD_SPANS.md`
             found the logo's only hole; every ray starts in the hardest region
             on the paper and ends in an easy one, so every ray is splittable
             and the cut position is what decides the balance.
  spiral     ONE continuous 15 m spiral.  A single stroke cannot be balanced by
             any whole-segment move — there is nothing to move — so this is the
             pure test of cutting, and of whether the pieces still come out as
             corridors an arm can walk rather than as confetti.
  duotone    Ten interleaved bands, alternating grey and orange, each spanning
             most of the sheet.  A pen is one colour per arm per PHASE, so this
             is the two-pass problem with the phases forced to interleave in
             space: each pass has to be covered by the whole fleet and neither
             pass can be given a tidy half of the paper.

Every generator is a pure function of (sheet, seed) and returns the same stroke
dicts `trace.to_sheet` produces — `pts` in sheet metres, `color`, `kind`, `id` —
so `scripts/bench.py` can hand them to `allocate.allocate` with no adapter and
the corpus stays comparable with the logo run.
"""
import numpy as np

from ..fleet import FLEET, SHEET

# The one placement decision the corpus makes, and it is made HERE rather than
# per drawing: the hatching patch sits on arm 71, which is an inverted arm with
# a 200 mm pen and three neighbours — the same shape of neighbourhood the logo's
# orange pass has, without being the logo's geometry.
HATCH_ARM = 71


def _poly(p0, p1, n=17):
    """A straight stroke as `n` points (the tracer's output is never 2 points)."""
    return np.column_stack([np.linspace(p0[0], p1[0], n),
                            np.linspace(p0[1], p1[1], n)])


def _stroke(pts, color, kind, sid):
    return dict(pts=np.asarray(pts, float), color=color, kind=kind, id=int(sid))


def hatch(sheet=SHEET, seed=1, arm=HATCH_ARM, n_lines=43, width=0.70,
          height=0.55):
    """Dense parallel hatching packed into one arm's territory. -> (strokes, meta).

    Centred on the arm's own base so the patch is squarely inside the region only
    that arm reaches comfortably, and sized so the total is about 30 m.  `seed`
    is accepted and unused: there is nothing random about a hatch, and taking the
    argument keeps every generator's signature the same.
    """
    x0, y0 = FLEET[arm].xy
    xs = (x0 - 0.5 * width, x0 + 0.5 * width)
    ys = np.linspace(y0 - 0.5 * height, y0 + 0.5 * height, int(n_lines))
    ys = ys[(ys > 0.02) & (ys < sheet[1] - 0.02)]
    strokes = [_stroke(_poly((xs[0], y), (xs[1], y)), "grey", "hatch", i)
               for i, y in enumerate(ys)]
    return strokes, dict(name="hatch", regime="dense parallel hatching in one "
                         f"arm's territory (arm {arm})", seed=int(seed),
                         n_lines=len(strokes), spacing_mm=1000 * height /
                         max(len(strokes) - 1, 1))


def scatter(sheet=SHEET, seed=2, n=70, lo=0.06, hi=0.20):
    """Short strokes thrown across the whole sheet. -> (strokes, meta).

    Uniform over the sheet inside a 25 cm border, uniform in angle, uniform in
    length between `lo` and `hi` metres — the regime where the clock is pen-up
    rather than ink and where a splitter should decline to cut, because every
    stroke is already near the 5 cm floor.
    """
    rng = np.random.default_rng(int(seed))
    mx, my = 0.25, 0.25
    strokes = []
    for i in range(int(n)):
        c = np.array([rng.uniform(mx, sheet[0] - mx), rng.uniform(my, sheet[1] - my)])
        th = rng.uniform(0, np.pi)
        h = 0.5 * rng.uniform(lo, hi) * np.array([np.cos(th), np.sin(th)])
        strokes.append(_stroke(_poly(c - h, c + h, 9), "grey", "dash", i))
    return strokes, dict(name="scatter", regime="sparse short strokes over the "
                         "whole sheet", seed=int(seed), n_strokes=len(strokes),
                         length_m=(lo, hi))


def starburst(sheet=SHEET, seed=3, n_rays=24, r_in=0.06, r_out=0.90):
    """Rays out of the middle of the sheet. -> (strokes, meta).

    The middle is the waist between the four inverted bases — the hardest place
    on this paper — so every ray runs from the hardest region to an easy one and
    every ray is splittable somewhere along its length.  Rays are laid down in
    angle order, which is deliberately the WORST order for pen-up travel: it
    gives the sequencer something to fix.
    """
    cx, cy = 0.5 * sheet[0], 0.5 * sheet[1]
    strokes = []
    for i in range(int(n_rays)):
        th = 2.0 * np.pi * i / int(n_rays)
        d = np.array([np.cos(th), np.sin(th)])
        strokes.append(_stroke(_poly((cx + r_in * d[0], cy + r_in * d[1]),
                                     (cx + r_out * d[0], cy + r_out * d[1]), 21),
                               "grey", "ray", i))
    return strokes, dict(name="starburst", regime="radial rays from the sheet "
                         "centre (the waist between the inverted bases)",
                         seed=int(seed), n_rays=int(n_rays),
                         centre=(float(cx), float(cy)))


def spiral(sheet=SHEET, seed=4, turns=4.75, r_in=0.08, r_out=0.88, ds=0.01):
    """ONE continuous Archimedean spiral, just under 15 m. -> (strokes, meta).

    A single stroke has no whole-segment move available to it at all — there is
    nothing to relocate and nothing to swap — so whatever balance this drawing
    reaches is entirely the cutting move's doing.  Sampled at `ds` along the
    curve rather than uniformly in angle, so the inner turns are not
    over-resolved and the outer ones under-resolved.

    JUST UNDER 15 m IS DELIBERATE, and the corpus is how it was found.  At the
    default 10 mm lattice step `stroke_api.DEFAULTS["max_steps"]` = 1500 refuses
    a single stroke of more than 15.00 m outright, as "too_long" rather than as
    a split — so a 15.11 m spiral is not a hard allocation problem, it is an
    allocation problem nobody is allowed to attempt, and every arm drops the
    whole drawing.  `turns` is set to land at 14.3 m so this measures the
    allocator rather than that ceiling.  The ceiling itself is real and stays
    reported in `docs/BENCH.md`; raising it is a planner question, not this one.
    """
    cx, cy = 0.5 * sheet[0], 0.5 * sheet[1]
    th_max = 2.0 * np.pi * float(turns)
    b = (r_out - r_in) / th_max
    th = np.linspace(0.0, th_max, 20000)
    r = r_in + b * th
    p = np.column_stack([cx + r * np.cos(th), cy + r * np.sin(th)])
    t = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    u = np.arange(0.0, t[-1], float(ds))
    pts = np.column_stack([np.interp(u, t, p[:, 0]), np.interp(u, t, p[:, 1])])
    pts = np.vstack([pts, p[-1]])
    return ([_stroke(pts, "grey", "spiral", 0)],
            dict(name="spiral", regime="one continuous spiral", seed=int(seed),
                 turns=float(turns), length_m=float(t[-1]),
                 centre=(float(cx), float(cy))))


def duotone(sheet=SHEET, seed=5, n_bands=10, amp=0.06, waves=2.5):
    """Interleaved grey/orange bands across the sheet. -> (strokes, meta).

    Each band spans most of the width, so no single arm can hold one and every
    band is a handoff problem; the colours alternate band by band, so neither
    PASS gets a tidy half of the paper and both have to be covered by the whole
    fleet.  The waviness is there so the bands are not all the same stroke
    translated, which would make the second one free.
    """
    x0, x1 = 0.35, sheet[0] - 0.35
    ys = np.linspace(0.30, sheet[1] - 0.30, int(n_bands))
    strokes = []
    for i, y in enumerate(ys):
        x = np.linspace(x0, x1, 240)
        ph = np.pi * i / max(int(n_bands) - 1, 1)
        yy = y + amp * np.sin(2.0 * np.pi * waves * (x - x0) / (x1 - x0) + ph)
        strokes.append(_stroke(np.column_stack([x, yy]),
                               "grey" if i % 2 == 0 else "orange", "band", i))
    return strokes, dict(name="duotone", regime="two-colour interleaved bands",
                         seed=int(seed), n_bands=len(strokes),
                         amp_m=float(amp))


# name -> (generator, the seed the corpus is pinned to).  The seeds are part of
# the corpus, not a parameter of it: a regression table whose inputs move is not
# a regression table.
BENCH = {
    "hatch": (hatch, 1),
    "scatter": (scatter, 2),
    "starburst": (starburst, 3),
    "spiral": (spiral, 4),
    "duotone": (duotone, 5),
}
ORDER = ("hatch", "scatter", "starburst", "spiral", "duotone")


def make(name, sheet=SHEET, seed=None):
    """One drawing by name. -> (strokes, meta) with `meta` carrying its regime."""
    if name not in BENCH:
        raise KeyError(f"no bench drawing {name!r}; have {sorted(BENCH)}")
    fn, s = BENCH[name]
    strokes, meta = fn(sheet, s if seed is None else seed)
    meta.update(total_m=float(sum(total_length([x]) for x in strokes)),
                n_strokes=len(strokes),
                colors=sorted({x["color"] for x in strokes}))
    return strokes, meta


def total_length(strokes):
    """Metres of ink in a drawing (the same measure `trace.total_length` uses)."""
    tot = 0.0
    for s in strokes:
        p = np.asarray(s["pts"], float)
        if len(p) > 1:
            tot += float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    return tot
