# Prompt: build a reusable meshcat-Line-to-Blender tool

Paste everything below this line to the Claude on the target machine.

---

I need a single reusable script, `meshcat_lines_to_blender.py`, that fixes a known defect in
our meshcat→Blender import pipeline. Reference material is in `~/handoff_aris_render/`
(see `SETUP_NOTES.md` §1, and `scripts/extract_ink.py` + `scripts/fix_ink.py`, which are a
known-working but one-off, hardcoded-path implementation you should generalize — read them
before writing code).

## Background (verified facts, don't rediscover them)

- Drake "static meshcat" HTML recordings embed their command stream as base64 msgpack inside
  `fetch("data:application/octet-binary;base64,...")` calls. Extract with a regex over the file,
  decode each blob with `msgpack.Unpacker(raw=False, strict_map_key=False,
  unicode_errors="surrogateescape")`, and keep only decoded dicts that have a `"type"` key
  (blobs can carry trailing junk). Typed arrays (e.g. vertex positions) arrive as msgpack
  **ExtType**: struct-unpack `ext.data` as little-endian per the attribute's `"type"` field
  (Float32Array→f, Float64Array→d, Uint32Array→I, ...).
- Pen/marker/trajectory strokes are `set_object` commands whose inner `object.object.type` is
  `Line` (also handle `LineSegments` and `LineLoop`) with `BufferGeometry` positions, a material
  carrying `color` (hex int) and `linewidth`, and possibly a non-identity `matrix`.
- Their appear/disappear animation is in the single `set_animation` command: per-path clips with
  a `.visible` track (`keys: [{time, value}, ...]`; times are already frame numbers, and clip
  `fps` matches the scene). There may also be initial `set_property visible=false` commands.
- The Blender addon `meshcat_html_importer` imports these Line objects as garbage (it triangulates
  every BufferGeometry: "every 3 vertices form a triangle") and silently drops `.visible` tracks.
  Everything else (meshes, arm animation, collections) imports fine. Drake world coords map 1:1
  to Blender; consecutive stroke chunks share endpoints exactly.

## What the script must do

Run inside Blender against an already-imported scene:

```
blender -b scene.blend --python meshcat_lines_to_blender.py -- \
    --html recording.html [--radius 0.0035] [--color-override HEX] \
    [--path-prefix /ink] [--save] [--out /path/new.blend]
```

1. Parse the HTML, find every Line-family `set_object` (all paths by default; `--path-prefix`
   filters). Decode points, color, linewidth, object matrix, initial visibility, and the full
   `.visible` track for that path.
2. Remove any previously-broken import of those objects AND any objects this script created on
   an earlier run (make reruns idempotent — tag everything you create with a custom property,
   e.g. `obj["meshcat_line_rebuild"] = True`, and delete by tag + by matching the importer's
   derived names for those paths). Never touch unrelated objects.
3. Rebuild each stroke as a POLY curve through its points: `bevel_depth = --radius`
   (default 0.0035 m), `bevel_resolution=4`, `use_fill_caps=True`, apply the object matrix if
   non-identity. Materials: Principled BSDF from the meshcat sRGB hex color (convert to linear!),
   roughness ~0.65, deduplicated — one shared material per distinct color.
4. Place rebuilt objects in collections mirroring the meshcat path (e.g. `/ink/...` → `ink`).
5. Replay the FULL `.visible` track as `hide_viewport` + `hide_render` keyframes — don't assume
   it's a single false→true toggle; a path may blink multiple times. Insert a hidden key one
   frame before each becomes-visible key so constant interpolation lands exactly. Honor initial
   `set_property visible=false`.
6. Print a verification summary: strokes rebuilt, per-prefix counts, distinct colors, world-space
   bounds, min/max reveal frames, and keyframe count. Exit nonzero (and change nothing where
   practical) if the HTML contains no Line objects or decoding fails.
7. `--save` overwrites the .blend in place; `--out` saves a copy instead.

## Constraints / gotchas

- Blender 5.x: read fcurves via `action.layers → strips → channelbags → fcurves` (no
  `action.fcurves`); `keyframe_insert` on the object works normally. Boolean hide fcurves are
  constant-interpolated by default — keep it that way.
- Blender's bundled Python may lack `msgpack`. Prefer importing the vendored copy that ships
  inside the importer addon (`meshcat_html_importer/_msgpack/`), fall back to pip-installed
  msgpack, and error clearly if neither exists.
- Pure `bpy` + stdlib otherwise; no GUI assumptions (must work with `blender -b`).

## Acceptance test

Run it on `aris_writing.html` over a fresh import of that file. Expected: 134 strokes under
`/ink`, all points on the z=0.0015 plane, one color (0x0c0c0c), linewidth 2.0, reveal frames
from 60 to 1723, scene range 0–1788 @30fps. Then render a frame around 900 and around 1788
(Cycles, any quality) and visually confirm partial vs. complete black "ARIS" strokes on the
paper. Re-run the script and confirm object counts don't grow (idempotency).
