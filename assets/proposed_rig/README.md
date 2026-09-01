# `assets/proposed_rig/` — still current, but not the model of the installation

This directory is **not deprecated** and nothing in it has changed. It is the
asset that `scripts/gen_proposed_rig_urdf.py` writes, that
`scripts/check_proposed_rig_urdf.py` and `tests/test_proposed_rig_urdf.py`
hold to the package, and that `scripts/collision_audit.py` reads. Every
certified number in this repo was earned against the obstacle model it
encodes.

Since 2026-09-01 there is also **`assets/system_model/`**, which is a
different thing and supersedes this one *as a description of the physical
installation*. Reach for it when the question is about the real build.

## Which one to open

| you want to… | use |
|---|---|
| reproduce or re-check a certified number | `proposed_rig` — it is the model the number was earned against |
| ask drake about arm-vs-arm clearance | **`system_model`** — this file's arm collision is the vendored Panda's 66 spheres per arm, which nothing has ever audited |
| show somebody what the rig looks like | **`system_model`** — this file's glTFs reference 65 texture files that are not present, so everything renders flat white |
| hand steel to a fabricator | **`system_model`** — this file models each mount as a Ø200 column to a 2.34 m "ceiling"; the real cage is a 2 × 2 post cluster and 2.34 m is a datum bug |
| plan against the certified keep-out envelope | `proposed_rig` — the schematic envelope is deliberately conservative and that is the point |

## The three differences, in one line each

1. **Arm collision.** Here: 66 unaudited spheres per arm. There: the
   manufacturer's own collision shells, *or* the audited capsule set — two
   files, no spheres.
2. **Textures.** Here: 65 dangling references and a warning per image per
   mesh. There: the real maps, vendored byte-identically.
3. **Structure.** Here: a schematic keep-out (Ø200 column, 2.34 m). There:
   the 80/20 cage off the original drawing, at the corrected datum — the
   drawing's 233,7 cm is floor-to-top-of-cage, not paper-to-ceiling.

`docs/SYSTEM_MODEL.md` explains all three, and lists what still has to be
re-certified before the two models can be reconciled.
