# Trollface — where these two files came from

| file | size | what it is |
|---|---|---|
| `trollface.png` | 1499 x 1216, RGBA | the working source. Black line art on a transparent surround, cropped to the ink with an 8 px pad. Downloaded 2026-08-24 from `pngimg.com/uploads/trollface/trollface_PNG2.png` and cropped; no other edit. |
| `trollface_wp.png` | 222 x 180, RGBA | the Wikipedia article's own file, `https://upload.wikimedia.org/wikipedia/en/7/73/Trollface.png`, via the `File:Trollface.png` page. Kept because it is the citable copy, and because it traces correctly too — `--inks auto` finds one ink and the tracer returns 49 strokes against 46, at less detail. |

**Why the larger one is the working source.** The Wikipedia file is 222 px on
its long side, which is below the 400 px this pipeline wants: the mouth is
~90 px wide and carries ten teeth, so a tooth is 9 px and its dividing strokes
are 1–2 px. They survive tracing, but as ragged polylines. At 1499 px the same
dividers are 10–14 px and come out as clean closed loops.

**Rights.** The Trollface drawing is a 2008 work by Carlos Ramirez ("Whynne")
and is **not** public domain; English Wikipedia hosts its copy under a
non-free-content rationale. These files are here as INPUT to a tracing and
motion-planning experiment and are not redistributed as artwork. Nothing in
this repository licenses them. If that matters for your use, substitute your
own picture — `scripts/draw.py` takes any raster or SVG:

```
ARIS_RIG=final6_opt python3 scripts/draw.py MY_PICTURE.png --out mypicture
```
