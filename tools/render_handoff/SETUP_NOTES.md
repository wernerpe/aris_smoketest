# Rendering Drake meshcat "writing" recordings in Blender — full setup notes

Written for handoff to another machine/Claude. This documents the complete pipeline used to
turn `aris_writing.html` (a Drake static-meshcat recording of 6 panda arms writing "ARIS")
into a rendered video, including the ink-marker fix, the house render style, the camera rig,
and the render/encode workflow. All referenced scripts are in `scripts/`.

## 0. Environment used

- Blender **5.2.0 LTS** (snap). API gotchas specific to 4.x/5.x are flagged below.
- 2× NVIDIA RTX 3090, Cycles on **OPTIX**. ~2 s/frame at the settings below (1080p, 25 samples).
- Python 3 with `msgpack` (pip) for decoding the meshcat HTML.
- ffmpeg for final encode.
- Optional: the Blender GUI ran a BlenderMCP-style addon listening on `127.0.0.1:9876`;
  `scripts/blender_bridge.py` sends `{"type": "execute_code", "params": {"code": ...}}` JSON
  over that TCP socket to drive the *live GUI session*. If there's no such addon on the target
  machine, run every scene-editing script headless instead:
  `blender -b scene.blend --python <script.py>` — the scripts only use `bpy`.

## 1. Import: meshcat HTML → Blender

Import was done with the `meshcat_html_importer` Blender extension
(`bl_ext.user_default.meshcat_html_importer`, lives under
`~/.config/blender/5.2/extensions/user_default/meshcat_html_importer/`):

```python
bpy.ops.preferences.addon_enable(module="bl_ext.user_default.meshcat_html_importer")
bpy.ops.import_scene.meshcat_html(filepath=..., hierarchical_collections=True)
```

This correctly brings in mesh geometry, the collection hierarchy mirroring meshcat paths,
and the arm joint animation (scene frame range comes from the recording; here 0–1788 @ 30 fps,
meshcat track times are already frame numbers). Drake world coordinates map 1:1 to Blender.

### 1a. KNOWN BUG — ink/marker strokes (three.js `Line` objects) import broken

The recording's writing lives at meshcat paths like `/ink/L0_S0_c000` (letter L*, stroke S*,
chunk c*): three.js `Line` objects with `BufferGeometry` (3–4 point polylines, ~1 cm segments),
each with a `.visible` animation track (hidden at frame 0, revealed at its own frame — that's
the progressive writing effect).

The importer (`blender_impl/mesh_builder.py::_create_from_mesh_geometry`) treats every
non-indexed BufferGeometry as a triangle soup ("every 3 vertices form a triangle"), so each
polyline becomes one degenerate sliver triangle — effectively invisible. Worse, the `.visible`
tracks are dropped entirely (0 visibility keyframes after import). Symptom: objects named
`ink_L0_S0_c000` etc. exist in an `ink` collection but nothing renders and nothing animates.

### 1b. The fix (two scripts)

**Step 1 — `scripts/extract_ink.py`** (plain python, edit `SRC`/`OUT` paths at top):
decodes the meshcat HTML directly and writes `ink_data.json`. Details that matter:

- Drake static HTML embeds commands as base64 msgpack in
  `fetch("data:application/octet-binary;base64,...")` calls — regex them out.
- Decode with `msgpack.Unpacker(raw=False, strict_map_key=False, unicode_errors="surrogateescape")`
  and keep only dicts with a `"type"` key (some blobs carry trailing junk).
- Typed arrays (positions) arrive as msgpack **ExtType** — struct-unpack `ext.data` as
  little-endian floats per the attribute's `"type"` (Float32Array etc.).
- Collect per chunk: `points` (Nx3, already in Drake/Blender world coords), material `color`
  (here 0x0c0c0c), and from the `set_animation` command the `.visible` track's last key time
  = the chunk's **reveal frame**.

**Step 2 — `scripts/fix_ink.py`** (runs inside Blender; edit `INK_JSON` path):
- deletes the broken `ink_*` mesh objects,
- rebuilds each chunk as a **POLY curve** through its points with `bevel_depth = 0.0035`
  (3.5 mm tube radius — scale to your scene; strokes here sit at z=0.0015 on a 4 mm-thick paper
  so the tube bottom hides inside the paper, no z-fighting), `bevel_resolution=4`,
  `use_fill_caps=True`,
- one shared `ink_material` (Principled, base color ~(0.004, 0.004, 0.004), roughness 0.65),
- keyframes `hide_viewport` + `hide_render`: hidden at frame 0 and at (reveal−1), visible at
  the reveal frame. Boolean fcurves interpolate CONSTANT so this reproduces meshcat exactly.
- Consecutive chunks share endpoints, so the tubes join into seamless strokes.

Sanity check: `scripts/check_ink_in_blender.py` dumps ink object counts, world transforms of
reference objects, and visibility-keyframe counts.

## 2. House render style ("blendervids" config)

Canonical source: `/home/peter/Documents/manipulation_station/blendervids/blender_scene_settings.md`
on the original machine. `scripts/apply_config.py` applies all of it (edit paths at top):

- **Cycles GPU**: OPTIX compute, enable all GPU devices, `samples=25`, `preview_samples=2`,
  OIDN denoising with `scene.cycles.denoising_use_gpu = True` (big speedup),
  `scene.render.use_persistent_data = True` (big speedup for animations).
- **Output**: 1920×1080 @ 100%, 30 fps, `film_transparent=False`.
  For direct video out (Blender 5.x API): `render.image_settings.media_type = 'VIDEO'`,
  ffmpeg MPEG4/H264, CRF MEDIUM, GOP 18, no audio. (But see §4 — prefer PNG frames.)
- **Lighting / "backlighting" trick** — the signature look (subjects lit realistically but on
  a pure-white background): World nodes =
  `TexCoord(Generated) → Mapping → EnvironmentTexture(brown_photostudio_02.hdr) → Background_HDRI`,
  plus a second pure-white `Background` (color 1,1,1, strength 1.0), mixed by
  `LightPath.Is Camera Ray → MixShader.Fac` (HDRI on shader-1/non-camera rays for lighting,
  white on shader-2/camera rays) → World Output. The HDRI (`brown_photostudio_02.hdr`, PolyHaven
  2k) is included in this handoff folder.
- **Shadow catcher**: a big plane with `is_shadow_catcher=True` + white diffuse material,
  placed *just under* the lowest scene surface so objects ground with soft shadows on the white.
  Here: 20×20 m at z = −0.0545, centered under the table (1.8035, 0.98).
- Domain-specific bits from the original doc (bin colors, gripper fades, trajectory materials)
  only apply to manipulation-station scenes — skip them for writing scenes.

## 3. Camera

The meshcat recording carries the viewpoint: `set_property` on
`/Cameras/default/rotated/<object>` gives the camera `position`, and a `set_target` command the
look-at, both in meshcat's rotated (y-up) frame → Drake world is `[x, −z_prop... ]` — in practice
here: target **(1.8035, 0.98, 0.35)** = table center + a bit up, camera direction (0, 3.43, −2.2)
normalized. Blender camera: 40 mm lens, 36 mm sensor, pulled back to `target − dir·5.2` so all
six arms fit at 16:9.

**Slow orbit rig** (`scripts/orbit_camera.py`): an Empty `CameraOrbit` at the target, camera
parented to it with **identity `matrix_parent_inverse`** and its LOCAL transform set explicitly
(local loc = world − pivot origin while pivot is unrotated). Do NOT rely on
`matrix_parent_inverse = pivot.matrix_world.inverted()` right after moving the pivot — the
depsgraph hasn't updated and you'll orbit around the wrong origin (this bit us). Keyframe the
empty's Z rotation 0 → 60° LINEAR over frames 0–1788 (= 1°/s). Because the look-at point lies
on the rotation axis, the camera stays aimed at center with no constraint needed.

**Overhead ending** (`scripts/overhead_end.py`): keyframes the camera's LOCAL loc/rot —
orbit pose held to frame 1590, Bezier swoop to straight-down overhead by 1740
(local loc (0, 0, 5.35) → world z = 5.7, world rotation identity so the text reads upright),
then to 1788 the local Z-rot linearly *counters* the still-rotating pivot
(−angle(1740) → −angle(1788)) so the overhead hold is perfectly still in world space.
Writing completes at frame 1723, so the word is finished during the hold.

## 4. Rendering + encode (crash-safe)

**Don't render straight to .mp4 for long jobs** — a killed ffmpeg-muxed render has no moov atom
and is unrecoverable (lost 27 min to this). Render PNG frames with overwrite off (resumable —
a rerun of the same command skips finished frames):

```bash
blender -b scene.blend --python-expr "
import bpy
s = bpy.context.scene
s.render.image_settings.media_type = 'IMAGE'
s.render.image_settings.file_format = 'PNG'
s.render.filepath = '/path/to/frames/'
s.render.use_overwrite = False
s.render.use_placeholder = True
" -a
```

Then encode:

```bash
# 1x (30 fps, 59.6 s)
ffmpeg -framerate 30 -i frames/%04d.png -c:v libx264 -preset medium -crf 18 \
       -pix_fmt yuv420p -g 18 -movflags +faststart aris_writing.mp4
# 2x (same frames played at 60 fps — smooth, no dropped frames)
ffmpeg -framerate 60 -i frames/%04d.png -c:v libx264 -preset medium -crf 18 \
       -pix_fmt yuv420p -movflags +faststart aris_writing_2x.mp4
```

## 5. Blender 5.x API gotchas hit along the way

- `action.fcurves` is gone (layered/slotted actions):
  iterate `action.layers → strips → channelbags → fcurves`.
- Video output is selected via `render.image_settings.media_type = 'VIDEO'`, not `file_format`.
- `scene.cycles.denoising_use_gpu` exists and matters.
- Addon module names for extensions: `bl_ext.user_default.<name>`.

## 6. Order of operations (fresh machine, headless-capable)

1. `pip install msgpack`; put the HDRI somewhere and fix the path in `apply_config.py`.
2. Import the meshcat HTML (GUI or `setup_scene.py`-style headless import).
3. `python3 extract_ink.py` (edit SRC/OUT) → `ink_data.json`.
4. Run in Blender: `fix_ink.py`, then `apply_config.py`, then `orbit_camera.py`,
   then `overhead_end.py` (each: `blender -b scene.blend --python script.py` + save, or via
   the socket bridge if the GUI addon is present).
5. Save .blend, render PNGs per §4, ffmpeg encode.
