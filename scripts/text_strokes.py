#!/usr/bin/env python3
"""A word -> single-stroke polylines on the installation's sheet.

WHY THIS EXISTS.  Every drawing this planner has conducted came out of
`trace.py`: a raster or an SVG, thinned to centrelines, fitted to the paper.
Hardware day 1 (2026-09-16) asks for something the tracer cannot give — the
word "unknown" written at a KNOWN place, at a KNOWN size, so that the hand-over
between arm 31 and arm 71 falls in the contested middle of the paper on
purpose and not by accident of where a logo happened to land.

WHY HERSHEY AND NOT A REAL FONT.  A pen draws CENTRELINES.  An outline font
(TrueType, and therefore anything matplotlib's `TextPath` would give) describes
the BOUNDARY of a letter, so drawing it lays down two lines per stem and the
planner then has to thin them back — which is exactly the tracer's job and
exactly the accuracy we do not want to spend on a legibility test.  Dr A. V.
Hershey's 1967 vector fonts are single-stroke by construction: each glyph IS
the pen path.  `futural.jhf` (Hershey Sans 1-stroke, the set plotter people
call "simplex") is vendored below as `_JHF` — the 95 printable ASCII glyphs,
raw `.jhf` payloads with the redundant glyph-number and vertex-count fields
stripped.  The Hershey fonts are public domain.

The pip package `HersheyFonts` was tried first, as asked, and does not exist on
PyPI (2026-09-15: "Could not find a version that satisfies the requirement
HersheyFonts (from versions: none)").  Hence the vendored table.

THE `.jhf` FORMAT, because it is unreadable otherwise.  After the two header
fields each glyph is a string of character PAIRS.  A pair `(a, b)` is the point
`(ord(a) - ord('R'), ord(b) - ord('R'))`; the literal pair `" R"` is a PEN UP
and starts a new polyline.  The FIRST pair of every glyph is not a point: it is
`(left, right)`, the glyph's side bearings, and `right - left` is its advance
width.  **y increases DOWNWARD** in Hershey coordinates and the BASELINE is at
y = +9, so the conversion into a normal y-up font frame is `(x - left, 9 - y)`.
Measured on this table: lowercase x-height = 14 units (y from -5 to +9),
ascender height = 21 units (y from -12 to +9).

WHAT COMES OUT.  `out/<stem>_strokes.json`, in the case-file schema the
conductor's own writers already use (`out/csail_schedule_h097_v19_strokes.json`):
`sheet`, `palette`, `info`, `n_strokes`, `total_length`, and `strokes[]` with
`id / color / kind / length / pts`.  `pts` are METRES in the canvas datum of
`docs/SYSTEM_MODEL.md` §1 — origin at the marked paper corner, z = 0 the paper
surface, x across the short side (0 .. 1.8034), y along the long one
(0 .. 3.63064).  ONE ink, so the allocation is single-pass by construction and
there is no pen swap to synchronise on a day with no fleet clock.

...plus a `placement` block, which is the part that makes the file USEFUL
rather than merely descriptive.  `csail_allocate.run_allocation` does not take
paper strokes; it takes PIXEL strokes and runs them through `trace.to_sheet`,
which centres the picture and scales it to fit.  A word placed to the
millimetre has to survive that.  `to_sheet` is affine and therefore invertible,
so `placement` carries the exact `(target_width, offset, rotate_deg)` triple at
which `to_sheet` is the IDENTITY on these points, together with the pixel
strokes to feed it.  `scripts/csail_schedule.py --strokes FILE` reads it.
`--verify` (on by default) re-runs the real `trace.to_sheet` and asserts the
round trip to 1e-9 m, so the claim is checked rather than asserted.

USAGE

    python3 scripts/text_strokes.py unknown \\
        --height 0.13 --x0 0.35 --x1 1.45 --y 1.815 \\
        --out out/unknown_strokes.json --png out/unknown_strokes.png

Defaults are hardware day 1's: the word "unknown" on the middle row, baseline
on the seam plane y = 1.8153 where arms 31 and 71 both stand, running from
x = 0.35 (outboard of arm 31, whose base is at x = 0.5967) to x = 1.45
(outboard of arm 71, at x = 1.2067).  The hand-over therefore falls near
x = 0.90, which is the middle of the contested band and the whole point of the
exercise.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ===========================================================================
# 1. The vendored font

BASELINE = 9          # raw Hershey y of the baseline (y grows DOWNWARD)
X_HEIGHT = 14         # font units, baseline to the top of 'x'      (y = -5)
ASCENDER = 21         # font units, baseline to the top of 'k'      (y = -12)
DESCENDER = 7         # font units below the baseline, 'g' 'p' 'q'  (y = +16)

FIRST_CHAR = 32       # _JHF[0] is ' '

_JHF = r"""JZ
MWRFRT RRYQZR[SZRY
JZNFNM RVFVM
H]SBLb RYBRb RLOZO RKUYU
H\PBP_ RTBT_ RYIWGTFPFMGKIKKLMMNOOUQWRXSYUYXWZT[P[MZKX
F^[FI[ RNFPHPJOLMMKMIKIIJGLFNFPGSHVHYG[F RWTUUTWTYV[X[ZZ[X[VYTWT
E_\O\N[MZMYNXPVUTXRZP[L[JZIYHWHUISJRQNRMSKSIRGPFNGMIMKNNPQUXWZY[[[\Z\Y
MWRHQGRFSGSIRKQL
KYVBTDRGPKOPOTPYR]T`Vb
KYNBPDRGTKUPUTTYR]P`Nb
JZRLRX RMOWU RWOMU
E_RIR[ RIR[R
NVSWRXQWRVSWSYQ[
E_IR[R
NVRVQWRXSWRV
G][BIb
H\QFNGLJKOKRLWNZQ[S[VZXWYRYOXJVGSFQF
H\NJPISFS[
H\LKLJMHNGPFTFVGWHXJXLWNUQK[Y[
H\MFXFRNUNWOXPYSYUXXVZS[P[MZLYKW
H\UFKTZT RUFU[
H\WFMFLOMNPMSMVNXPYSYUXXVZS[P[MZLYKW
H\XIWGTFRFOGMJLOLTMXOZR[S[VZXXYUYTXQVOSNRNOOMQLT
H\YFO[ RKFYF
H\PFMGLILKMMONSOVPXRYTYWXYWZT[P[MZLYKWKTLRNPQOUNWMXKXIWGTFPF
H\XMWPURRSQSNRLPKMKLLINGQFRFUGWIXMXRWWUZR[P[MZLX
NVROQPRQSPRO RRVQWRXSWRV
NVROQPRQSPRO RSWRXQWRVSWSYQ[
F^ZIJRZ[
E_IO[O RIU[U
F^JIZRJ[
I[LKLJMHNGPFTFVGWHXJXLWNVORQRT RRYQZR[SZRY
E`WNVLTKQKOLNMMPMSNUPVSVUUVS RQKOMNPNSOUPV RWKVSVUXVZV\T]Q]O\L[JYHWGTFQFNGLHJJILHOHRIUJWLYNZQ[T[WZYYZX RXKWSWUXV
I[RFJ[ RRFZ[ RMTWT
G\KFK[ RKFTFWGXHYJYLXNWOTP RKPTPWQXRYTYWXYWZT[K[
H]ZKYIWGUFQFOGMILKKNKSLVMXOZQ[U[WZYXZV
G\KFK[ RKFRFUGWIXKYNYSXVWXUZR[K[
H[LFL[ RLFYF RLPTP RL[Y[
HZLFL[ RLFYF RLPTP
H]ZKYIWGUFQFOGMILKKNKSLVMXOZQ[U[WZYXZVZS RUSZS
G]KFK[ RYFY[ RKPYP
NVRFR[
JZVFVVUYTZR[P[NZMYLVLT
G\KFK[ RYFKT RPOY[
HYLFL[ RL[X[
F^JFJ[ RJFR[ RZFR[ RZFZ[
G]KFK[ RKFY[ RYFY[
G]PFNGLIKKJNJSKVLXNZP[T[VZXXYVZSZNYKXIVGTFPF
G\KFK[ RKFTFWGXHYJYMXOWPTQKQ
G]PFNGLIKKJNJSKVLXNZP[T[VZXXYVZSZNYKXIVGTFPF RSWY]
G\KFK[ RKFTFWGXHYJYLXNWOTPKP RRPY[
H\YIWGTFPFMGKIKKLMMNOOUQWRXSYUYXWZT[P[MZKX
JZRFR[ RKFYF
G]KFKULXNZQ[S[VZXXYUYF
I[JFR[ RZFR[
F^HFM[ RRFM[ RRFW[ R\FW[
H\KFY[ RYFK[
I[JFRPR[ RZFRP
H\YFK[ RKFYF RK[Y[
KYOBOb RPBPb ROBVB RObVb
KYKFY^
KYTBTb RUBUb RNBUB RNbUb
JZRDJR RRDZR
I[Ib[b
NVSKQMQORPSORNQO
I\XMX[ RXPVNTMQMONMPLSLUMXOZQ[T[VZXX
H[LFL[ RLPNNPMSMUNWPXSXUWXUZS[P[NZLX
I[XPVNTMQMONMPLSLUMXOZQ[T[VZXX
I\XFX[ RXPVNTMQMONMPLSLUMXOZQ[T[VZXX
I[LSXSXQWOVNTMQMONMPLSLUMXOZQ[T[VZXX
MYWFUFSGRJR[ ROMVM
I\XMX]W`VaTbQbOa RXPVNTMQMONMPLSLUMXOZQ[T[VZXX
I\MFM[ RMQPNRMUMWNXQX[
NVQFRGSFREQF RRMR[
MWRFSGTFSERF RSMS^RaPbNb
IZMFM[ RWMMW RQSX[
NVRFR[
CaGMG[ RGQJNLMOMQNRQR[ RRQUNWMZM\N]Q][
I\MMM[ RMQPNRMUMWNXQX[
I\QMONMPLSLUMXOZQ[T[VZXXYUYSXPVNTMQM
H[LMLb RLPNNPMSMUNWPXSXUWXUZS[P[NZLX
I\XMXb RXPVNTMQMONMPLSLUMXOZQ[T[VZXX
KXOMO[ ROSPPRNTMWM
J[XPWNTMQMNNMPNRPSUTWUXWXXWZT[Q[NZMX
MYRFRWSZU[W[ ROMVM
I\MMMWNZP[S[UZXW RXMX[
JZLMR[ RXMR[
G]JMN[ RRMN[ RRMV[ RZMV[
J[MMX[ RXMM[
JZLMR[ RXMR[P_NaLbKb
J[XMM[ RMMXM RM[X[
KYTBRCQDPFPHQJRKSMSOQQ RRCQEQGRISJTLTNSPORSTTVTXSZR[Q]Q_Ra RQSSUSWRYQZP\P^Q`RaTb
NVRBRb
KYPBRCSDTFTHSJRKQMQOSQ RRCSESGRIQJPLPNQPURQTPVPXQZR[S]S_Ra RSSQUQWRYSZT\T^S`RaPb
F^IUISJPLONOPPTSVTXTZS[Q RISJQLPNPPQTTVUXUZT[Q[O""".split("\n")


def glyph(ch, chain=True):
    """One character -> (advance, [polyline, ...]) in FONT units, y UP.

    Polylines are lists of (x, y) ints with x measured from the glyph's own
    left bearing and y from the baseline.  An unknown character raises, on
    purpose: silently dropping a letter would produce a word that is not the
    word that was asked for, and the failure would only be visible in ink.
    """
    i = ord(ch) - FIRST_CHAR
    if not (0 <= i < len(_JHF)):
        raise KeyError(f"{ch!r} is not in the vendored Hershey table "
                       f"(printable ASCII {FIRST_CHAR}..{FIRST_CHAR + len(_JHF) - 1})")
    payload = _JHF[i]
    left = ord(payload[0]) - ord("R")
    right = ord(payload[1]) - ord("R")
    rest = payload[2:]
    if len(rest) % 2:
        raise ValueError(f"odd-length glyph payload for {ch!r}: {payload!r}")
    polys, cur = [], []
    for k in range(0, len(rest), 2):
        a, b = rest[k], rest[k + 1]
        if a == " " and b == "R":                       # pen up
            if len(cur) > 1:
                polys.append(cur)
            cur = []
            continue
        cur.append((ord(a) - ord("R") - left, BASELINE - (ord(b) - ord("R"))))
    if len(cur) > 1:
        polys.append(cur)
    return right - left, (chain_polys(polys) if chain else polys)


def chain_polys(polys):
    """Join polylines that share an endpoint. -> [polyline, ...].

    NOT cosmetic.  Hershey stores 'w' as FOUR separate two-point segments that
    happen to meet at their ends, and 'm', 'v', 'y' and 'z' likewise.  Drawn
    literally that is four strokes and three pen-ups per 'w', and on hardware a
    pen-up is a lift, a transit, a re-approach and a fresh touchdown — four
    chances to leave a blob and four places for the 1 mm of height error the
    ladder has not yet measured to show up.  Chained, 'w' is one stroke.

    Lossless: the joined polyline visits exactly the same points in the same
    order, and a segment is reversed only where that is what makes the ends
    meet.  Exact integer comparison, because these are integers.
    """
    todo = [list(p) for p in polys]
    out = []
    while todo:
        cur = todo.pop(0)
        joined = True
        while joined:
            joined = False
            for k, other in enumerate(todo):
                if cur[-1] == other[0]:
                    cur = cur + other[1:]
                elif cur[-1] == other[-1]:
                    cur = cur + other[::-1][1:]
                elif cur[0] == other[-1]:
                    cur = other + cur[1:]
                elif cur[0] == other[0]:
                    cur = other[::-1] + cur[1:]
                else:
                    continue
                todo.pop(k)
                joined = True
                break
        out.append(cur)
    return out


# ===========================================================================
# 2. A word on the paper

def layout(word, height, x0, x1, y, chain=True, tracking=None):
    """A word -> polylines in METRES in the canvas datum. -> (polys, info).

    `height` is the x-HEIGHT: the height of 'u', 'n', 'o', 'w' — the letters
    you actually see.  An ascender ('k' in "unknown") reaches 1.5x that, and a
    descender drops half that below the baseline; `info` reports the real ink
    bounding box so nobody has to guess.  x-height and not cap height because
    the word is lowercase and "letters 0.12-0.15 m tall" is a statement about
    what a person standing at the table will measure with a ruler.

    `x0`, `x1` are the INK bounds, not origins: the leftmost mark of the first
    letter lands on x0 and the rightmost mark of the last on x1, exactly.  The
    word is stretched to fit by opening the GAPS between letters (the glyph
    shapes are never scaled non-uniformly and never sheared) — so the letters
    stay the size `height` asked for and only the tracking changes.  Pass
    `tracking` (metres per gap) to fix the tracking instead and let the word
    end where it ends; `x1` is then ignored.

    `y` is the BASELINE.
    """
    if not word:
        raise ValueError("nothing to write")
    scale = float(height) / X_HEIGHT
    placed, pen = [], 0.0
    for ch in word:
        adv, polys = glyph(ch, chain=chain)
        placed.append(dict(ch=ch, origin=pen, adv=adv * scale,
                           polys=[np.asarray(p, float) * scale for p in polys]))
        pen += adv * scale
    inked = [g for g in placed if g["polys"]]
    if not inked:
        raise ValueError(f"{word!r} has no ink in it")

    def ink_span(gs):
        lo = min(float((g["origin"] + np.vstack(g["polys"])[:, 0]).min()) for g in gs)
        hi = max(float((g["origin"] + np.vstack(g["polys"])[:, 0]).max()) for g in gs)
        return lo, hi

    lo, hi = ink_span(inked)
    n_gap = len(placed) - 1
    if tracking is not None:
        extra = float(tracking)
    elif n_gap > 0:
        extra = ((float(x1) - float(x0)) - (hi - lo)) / n_gap
    else:
        extra = 0.0
    # THE GAP IS OPENED, THE GLYPH IS NOT TOUCHED.  Shifting the k-th origin by
    # k*extra adds exactly n_gap*extra to the ink span and nothing to any
    # letter's width, which is what keeps `height` honest.
    for k, g in enumerate(placed):
        g["origin"] += k * extra
    lo2, hi2 = ink_span(inked)
    dx = float(x0) - lo2
    tight = extra < -1e-12 and n_gap > 0

    out, ys = [], []
    for g in placed:
        for p in g["polys"]:
            xy = np.column_stack([p[:, 0] + g["origin"] + dx, p[:, 1] + float(y)])
            out.append(xy)
            ys.append(xy[:, 1])
    Y = np.concatenate(ys)
    P = np.vstack(out)
    info = dict(
        word=word, height_x=float(height), scale_m_per_unit=scale,
        baseline_y=float(y), tracking_m=float(extra), tight=bool(tight),
        natural_width=float(hi - lo), n_glyphs=len(placed), n_strokes=len(out),
        bbox=[float(P[:, 0].min()), float(Y.min()),
              float(P[:, 0].max()), float(Y.max())],
        ascender_m=float(height) * ASCENDER / X_HEIGHT,
        chained=bool(chain))
    return out, info


# ===========================================================================
# 3. The case file, and the placement that reproduces it exactly

def plen(pts):
    p = np.asarray(pts, float)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0


def placement_for(polys, sheet):
    """The (px strokes, target_width, offset, rotate) that `to_sheet` undoes.

    `trace.to_sheet` maps pixel points to metres by

        x_m = cx + (px - mid_x) * s,   y_m = cy - (py - mid_y) * s
        s   = min(avail_w / w, avail_h / h, target_width / w)
        cx  = sheet[0]/2 + off_x,      cy = sheet[1]/2 + off_y

    -- note the y FLIP.  Feed it `px = (x_m, -y_m)` and the mid-points become
    `(mid_x, -mid_y)` of the metre geometry, so both rows collapse to a pure
    translation once `s == 1`.  `s == 1` is bought by setting
    `target_width = w` (the metre width), which makes the third term exactly 1
    and is a MINIMUM, so it only binds if the word already fits the sheet's
    margin -- which `info["fits"]` is there to say.  The offset is then just
    the word's centre minus the sheet's.

    This is why the case file is directly conductable and why `--verify`
    can assert it: the identity is a claim about a function in this repository,
    not about a convention.
    """
    P = np.vstack(polys)
    xlo, xhi = float(P[:, 0].min()), float(P[:, 0].max())
    ylo, yhi = float(P[:, 1].min()), float(P[:, 1].max())
    w = xhi - xlo
    cx, cy = 0.5 * (xlo + xhi), 0.5 * (ylo + yhi)
    px = [np.column_stack([p[:, 0], -p[:, 1]]) for p in polys]
    return dict(target_width=w, rotate_deg=0.0,
                offset=[cx - sheet[0] / 2.0, cy - sheet[1] / 2.0],
                px=px)


def case_file(polys, info, sheet, ink="black", hexcolor="#111111",
              kind="outline", name="text"):
    """Polylines in metres -> the conductor's stroke case-file dict."""
    strokes = [dict(id=i, color=ink, kind=kind, length=plen(p),
                    pts=np.round(np.asarray(p, float), 6).tolist())
               for i, p in enumerate(polys)]
    pl = placement_for(polys, sheet)
    P = np.vstack(polys)
    return dict(
        sheet=[float(sheet[0]), float(sheet[1])],
        palette={ink: hexcolor},
        info=dict(info, name=name,
                  center=[float(0.5 * (P[:, 0].min() + P[:, 0].max())),
                          float(0.5 * (P[:, 1].min() + P[:, 1].max()))],
                  logo_w=float(P[:, 0].max() - P[:, 0].min()),
                  logo_h=float(P[:, 1].max() - P[:, 1].min()),
                  rotate_deg=0.0, offset=pl["offset"], fits=True),
        placement=dict(target_width=pl["target_width"],
                       offset=pl["offset"], rotate_deg=pl["rotate_deg"],
                       px=[np.round(q, 6).tolist() for q in pl["px"]],
                       note="feed `px` to trace.to_sheet with these three "
                            "knobs and it is the identity; "
                            "scripts/csail_schedule.py --strokes does that"),
        n_strokes=len(strokes),
        total_length=float(sum(s["length"] for s in strokes)),
        strokes=strokes)


def load_case(path):
    """A case file -> (px strokes ready for `run_allocation`, knobs). -> dict.

    The one reader.  `scripts/csail_schedule.py --strokes` calls this, and so
    does `--verify`, so the file can only mean one thing.
    """
    doc = json.loads(Path(path).read_text())
    pl = doc["placement"]
    px = [dict(id=i, color=s["color"], kind=s["kind"],
               pts=np.asarray(q, float))
          for i, (s, q) in enumerate(zip(doc["strokes"], pl["px"]))]
    return dict(px=px, target_width=float(pl["target_width"]),
                offset=tuple(float(v) for v in pl["offset"]),
                rotate_deg=float(pl["rotate_deg"]),
                sheet=tuple(float(v) for v in doc["sheet"]),
                palette=dict(doc["palette"]),
                name=doc["info"].get("name", "text"),
                doc=doc)


def verify(doc, margin=0.06, min_len=0.0):
    """Round-trip the placement through the REAL `trace.to_sheet`. -> max err.

    Raises if the identity does not hold to 1e-9 m or if `to_sheet` drops or
    reorders a stroke.  This is the only thing standing between "the word is at
    x = 0.35" and "the word is wherever the placement search felt like".
    """
    from aris_sixarm import trace
    px = [dict(id=i, color=s["color"], kind=s["kind"],
               pts=np.asarray(q, float))
          for i, (s, q) in enumerate(zip(doc["strokes"], doc["placement"]["px"]))]
    got, got_info = trace.to_sheet(
        px, tuple(doc["sheet"]), margin=margin, min_len=min_len,
        target_width=doc["placement"]["target_width"],
        offset=tuple(doc["placement"]["offset"]),
        rotate_deg=doc["placement"]["rotate_deg"])
    if len(got) != len(doc["strokes"]):
        raise AssertionError(f"to_sheet returned {len(got)} of "
                             f"{len(doc['strokes'])} strokes (min_len dropped "
                             "some; lower --min-len)")
    if not got_info["fits"]:
        raise AssertionError(f"the word does not fit inside the {margin:g} m "
                             f"margin: {got_info}")
    err = max(float(np.abs(np.asarray(g["pts"], float)
                           - np.asarray(s["pts"], float)).max())
              for g, s in zip(got, doc["strokes"]))
    if err > 1e-9:
        raise AssertionError(f"to_sheet round trip is off by {err:.3e} m; the "
                             "placement block does not reproduce this file")
    return err


# ===========================================================================
# 4. The picture

def plot(doc, path, arms=None, title=None, reach=0.855):
    """The word on the sheet, with the arm bases marked. -> None.

    TWO panels, because the two things a person needs from this picture are at
    different scales: where the word sits on a 1.8 x 3.6 m sheet (left), and
    whether the hand-over really falls between the two arms (right).  One panel
    showing both makes the letters four pixels tall.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sheet = doc["sheet"]
    info = doc["info"]
    bx = info["bbox"]
    hexc = list(doc["palette"].values())[0]
    arms = arms or {}
    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(11.0, 5.6), dpi=150,
        gridspec_kw=dict(width_ratios=[1.0, 1.9]))

    for ax, zoom in ((axa, False), (axb, True)):
        ax.add_patch(plt.Rectangle((0, 0), sheet[0], sheet[1], fc="#fbfaf7",
                                   ec="#bdb9b0", lw=1.0, zorder=0))
        for s in doc["strokes"]:
            p = np.asarray(s["pts"], float)
            ax.plot(p[:, 0], p[:, 1], "-", color=hexc,
                    lw=(1.2 if not zoom else 2.4), solid_capstyle="round",
                    zorder=3)
            ax.plot(p[0, 0], p[0, 1], "o", color="#1f77b4",
                    ms=(2.0 if not zoom else 4.5), zorder=4)
        ax.plot([bx[0], bx[2], bx[2], bx[0], bx[0]],
                [bx[1], bx[1], bx[3], bx[3], bx[1]], "--", color="#999",
                lw=0.8, zorder=2)
        ax.axhline(info["baseline_y"], color="#c33", lw=0.8, ls=":", zorder=1)
        for aid, (ax_, ay_) in sorted(arms.items()):
            ax.plot([ax_], [ay_], "x", color="#111", ms=11, mew=2.2, zorder=6)
            ax.add_patch(plt.Circle((ax_, ay_), reach, fc="none", ec="#7a9ec2",
                                    ls="--", lw=0.8, zorder=1))
        ax.set_aspect("equal")
        ax.grid(True, lw=0.3, color="#e6e4e0")

    axa.set_xlim(-0.08, sheet[0] + 0.08)
    axa.set_ylim(-0.08, sheet[1] + 0.08)
    axa.set_title("the whole sheet", fontsize=9)
    axa.set_xlabel("x  [m]")
    axa.set_ylabel("y  [m]   (canvas datum: origin at the marked paper corner)")

    pad = 0.16
    axb.set_xlim(bx[0] - pad, bx[2] + pad)
    axb.set_ylim(min(bx[1], *(v[1] for v in arms.values()) or (bx[1],)) - 0.22,
                 bx[3] + 0.16)
    for aid, (ax_, ay_) in sorted(arms.items()):
        axb.annotate(f"arm {aid}  ({ax_:.4f}, {ay_:.4f})", (ax_, ay_),
                     textcoords="offset points", xytext=(-52, -22), fontsize=8,
                     color="#111", zorder=7)
    mid = 0.5 * (bx[0] + bx[2])
    axb.axvline(mid, color="#2a8", lw=0.9, ls="-.", zorder=1)
    axb.text(mid, bx[3] + 0.02, f" hand-over band, x ≈ {mid:.3f}",
             color="#2a8", fontsize=7.5, va="bottom")
    axb.text(bx[0], info["baseline_y"] - 0.035,
             f"baseline y = {info['baseline_y']:.4f}  (the seam plane)",
             color="#c33", fontsize=7.5, va="top")
    axb.plot([], [], "o", color="#1f77b4", ms=4, label="stroke start (pen down)")
    axb.plot([], [], "-", color=hexc, lw=2, label="ink (single-line Hershey)")
    axb.plot([], [], "x", color="#111", ms=8, mew=1.6, label="arm J1 axis")
    axb.plot([], [], "--", color="#7a9ec2", lw=0.8,
             label=f"{reach:.3f} m planar reach")
    axb.legend(loc="lower right", fontsize=7, framealpha=0.95)
    axb.set_xlabel("x  [m]")
    axb.set_title(f"the contested middle — {doc['n_strokes']} strokes, "
                  f"{doc['total_length']:.3f} m of ink", fontsize=9)

    fig.suptitle(title or (f"{info['word']!r}   x-height "
                           f"{1000 * info['height_x']:.0f} mm, ascender "
                           f"{1000 * info['ascender_m']:.0f} mm, "
                           f"x {bx[0]:.3f} .. {bx[2]:.3f} m, tracking "
                           f"{1000 * info['tracking_m']:+.1f} mm"), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path)
    plt.close(fig)


# ===========================================================================

def _default_sheet():
    """The active rig's sheet, without forcing an import when one is given."""
    try:
        from aris_sixarm import fleet
        return tuple(float(v) for v in fleet.SHEET)
    except Exception:
        return (1.8034, 3.63064)


def _default_arms():
    """The two middle-row arms, from the layout if it will load."""
    fallback = {31: (0.5967, 1.8153), 71: (1.2067, 1.8153)}
    try:
        from aris_sixarm import fleet
        out = {}
        for aid in (31, 71):
            if aid in fleet.FLEET:
                T = fleet.FLEET[aid].T_world_base()
                out[aid] = (float(T[0, 3]), float(T[1, 3]))
        return out or fallback
    except Exception:
        return fallback


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="defaults are hardware day 1: 'unknown' across the middle row")
    ap.add_argument("word", nargs="?", default="unknown")
    ap.add_argument("--height", type=float, default=0.13, metavar="M",
                    help="x-HEIGHT in metres: how tall 'u', 'n', 'o', 'w' come "
                         "out.  'k' reaches 1.5x this (default 0.13)")
    ap.add_argument("--x0", type=float, default=0.35, metavar="M",
                    help="x of the LEFTMOST mark (default 0.35, just outboard "
                         "of arm 31 at x = 0.5967)")
    ap.add_argument("--x1", type=float, default=1.45, metavar="M",
                    help="x of the RIGHTMOST mark (default 1.45, just outboard "
                         "of arm 71 at x = 1.2067)")
    ap.add_argument("--y", type=float, default=1.8153, metavar="M",
                    help="the BASELINE (default 1.8153, the seam plane both "
                         "middle-row arms stand on)")
    ap.add_argument("--tracking", type=float, default=None, metavar="M",
                    help="fix the inter-letter gap instead of fitting x0..x1")
    ap.add_argument("--no-chain", action="store_true",
                    help="keep Hershey's own stroke breaks ('w' becomes four "
                         "strokes and three pen-ups)")
    ap.add_argument("--ink", default="black")
    ap.add_argument("--hex", default="#111111")
    ap.add_argument("--sheet", type=float, nargs=2, default=None,
                    metavar=("W", "H"))
    ap.add_argument("--margin", type=float, default=0.06,
                    help="the margin --verify holds the word inside (matches "
                         "csail_allocate's default)")
    ap.add_argument("--out", default=None, help="default out/<word>_strokes.json")
    ap.add_argument("--png", default=None, help="default beside --out")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the to_sheet round-trip assertion")
    ap.add_argument("--no-png", action="store_true")
    a = ap.parse_args(argv)

    sheet = tuple(a.sheet) if a.sheet else _default_sheet()
    polys, info = layout(a.word, a.height, a.x0, a.x1, a.y,
                         chain=not a.no_chain, tracking=a.tracking)
    doc = case_file(polys, info, sheet, ink=a.ink, hexcolor=a.hex,
                    name=a.word)

    out = Path(a.out) if a.out else ROOT / "out" / f"{a.word}_strokes.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    bx = info["bbox"]
    print(f"{a.word!r}: {doc['n_strokes']} single-line strokes, "
          f"{doc['total_length']:.4f} m of ink")
    print(f"  x-height {1000 * info['height_x']:.1f} mm, ascender "
          f"{1000 * info['ascender_m']:.1f} mm, baseline y = {info['baseline_y']:.4f}")
    print(f"  ink bbox x [{bx[0]:.4f}, {bx[2]:.4f}] = {bx[2] - bx[0]:.4f} m, "
          f"y [{bx[1]:.4f}, {bx[3]:.4f}] = {bx[3] - bx[1]:.4f} m")
    print(f"  tracking {1000 * info['tracking_m']:+.1f} mm per gap "
          f"(natural width would be {info['natural_width']:.4f} m)")
    if info["tight"]:
        print("  !! NEGATIVE tracking: the letters are being pushed into each "
              "other.  Raise --x1, lower --x0, or lower --height.")
    shortest = min(s["length"] for s in doc["strokes"])
    print(f"  shortest stroke {1000 * shortest:.1f} mm "
          f"(csail_allocate's --min-len default is 25 mm; pass --min-len "
          f"{max(0.001, 0.9 * shortest):.3f} or lower when conducting)")

    if not a.no_verify:
        err = verify(doc, margin=a.margin, min_len=0.0)
        print(f"  placement round-trip through trace.to_sheet: max error "
              f"{err:.2e} m  OK")
        doc["placement"]["verified_m"] = float(err)

    out.write_text(json.dumps(doc, indent=1))
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} kB)")

    if not a.no_png:
        png = Path(a.png) if a.png else out.with_suffix(".png")
        plot(doc, png, arms=_default_arms())
        print(f"wrote {png}  <- LOOK AT THIS")
    return doc


if __name__ == "__main__":
    main()
