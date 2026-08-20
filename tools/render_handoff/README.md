# render_handoff

Pete's Blender pipeline for rendering a Drake static-meshcat recording (an `aris_writing.html`
export of the six-arm drawing plan) into a video. From `~/handoff_aris_render.tar.gz`, 2026-08-19;
tarball still in `/home/franka/`. Read `SETUP_NOTES.md` first — it is the whole workflow;
`PROMPT_build_reusable_line_tool.md` specs a reusable rewrite of the `scripts/` ink fix.

Needs Blender 5.x (have: snap 5.2.0), `msgpack`, the `meshcat_html_importer` addon (missing),
and ffmpeg for the encode (missing). `brown_photostudio_02.hdr` (lighting, 6.2 MB) is here but
gitignored. Every script has hardcoded `/home/peter/...` paths to edit before use.
