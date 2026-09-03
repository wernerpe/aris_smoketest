# `assets/proposed_rig/` — still current, but not the model of the installation

This directory is **not deprecated**. It is the asset that
`scripts/gen_proposed_rig_urdf.py` writes, that
`scripts/check_proposed_rig_urdf.py` and `tests/test_proposed_rig_urdf.py`
hold to the package, and that `scripts/collision_audit.py` reads. Every
certified number in this repo was earned against the obstacle model it
encodes.

**Two things in it have changed, and they are named here so nobody has to
diff for them.**

**(1) The pen holder's housing was mounted end-for-end.** On **2026-09-03**
`penholder22_T_hand` was found to point the housing's +X away from the pen
tip, so the model had the pen leaving by the tail land instead of through the
cap. It was corrected: the two hand-frame holder meshes were re-baked, the
holder's three collision cylinders moved 30 mm along the bore, a
fourth was added for the pencil tail, and `pen_lead` grew from 100.5 mm to
125.6 mm. **Nothing else moved** — no capsule radius, no arm collision
geometry, no base pose, no layout number — and the **pen tip is unchanged to
the picometre**, which is what makes it a correction to a body rather than to
a certified number. `docs/SYSTEM_MODEL.md` §7c has the evidence, the numbers
and the measured cost (the r = 0.050 lateral tool capsules contained the old
placement and do not contain this one).

**(2) The pen TIP moved, on the same day and for a different reason.** A photo
of the real gripper put the tip about 5 cm below the bottom edge of the Fat
Franka Finger blades' contact plates, not 15 cm below the hand:
`frames.PEN_LAT_HOLDER` and the new `frames.PEN_EXT_HOLDER` are **0.0588421 m**
each where the pair used to be 0.110 / 0.110. In this file that moves
`pen_bracket`'s and `pen_body`'s lengths, the `pen_tip` weld and `pen_lead`
(53.2 mm of graphite past the cap now, not 125.6) — **and nothing else**: the
holder's own placement depends on the ray's *direction* and on the grip centre,
both unchanged at 45°, so the two hand-frame holder meshes here are
**byte-identical** across the change. The old pair was never gate-validated;
only the INLINE pen's axial 0.110 is. `docs/SYSTEM_MODEL.md` §7e.

**What that costs the certified numbers, said plainly:** every programme and
atlas in `out/` was planned at the old tip and is now **stale**. Re-planning
them is a queued re-certification, not part of this change.

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
