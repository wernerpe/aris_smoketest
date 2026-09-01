# The system model — `assets/system_model/`

A dimensioned, provenance-annotated 3D model of the whole ARIS installation:
the 80/20 cage, the table and paper it stands over, six FR3 arms at
`FLEET_PROPOSED`, and a pen holder on every hand. Visual **and** collision.

Built 2026-09-01. Generator `scripts/gen_system_model.py`, truth
`aris_sixarm/system_model.py`, validator `scripts/check_system_model.py`,
tests `tests/test_system_model.py`.

```
python3 scripts/gen_system_model.py all           # meshes + URDFs + manifest
<station venv>/bin/python scripts/check_system_model.py
<station venv>/bin/python scripts/render_system_model.py
```

`<station venv>` = `/home/franka/git/franka_manipulation_station/.venv`.

---

## 1. The headline: the ceiling datum was wrong, and this model says so

**`mounts.MOUNTS.ceiling_z = 2.34 m` is a provenance bug.** It has not been
changed — see §6 — but the model carries the truth.

The original drawing has one overall height dimension, **233,7 cm**, and it is
**floor to top-of-construction**. The drawing describes a self-supporting cage
standing on the floor; it defines no room ceiling anywhere. The paper sits
**636.68 mm above that floor**. So above the *paper*, the same steel reaches:

| datum | mm above the paper | how |
|---|---:|---|
| top of construction | **1699.82** | `top_slab` hi, 233.65 cm − 63.668 cm |
| beam underside (what a post hangs from) | **1623.62** | `top_slab` lo |
| cage total height above the **floor** | **2336.50** | = the drawing's 233,7 cm |

`ceiling_z = 2.34 m` is that 233,7 re-datumed to the paper — a floor-
referenced number used as a paper-referenced one. It puts the top of the cage
**2976.68 mm** above the paper, **716.4 mm** higher than the drawing.

Being too tall is **conservative for collision** — a longer obstacle never
certifies a pose a shorter one refuses — which is why nothing has broken and
why no certified number is in question. It is **not** conservative for a
fabricator:

| | drop post length at h = 940 |
|---|---:|
| the buggy datum | 1434.98 mm |
| **the corrected datum** | **718.60 mm** |
| the original rig's own | 736.90 mm |

The corrected post is within 18 mm of the one that was actually built. The
sheet's own UNKNOWN 3 spotted the same thing from the other end ("a grid at
the ORIGINAL's own beam height gives a 718.6 post").

**Nothing here is a survey. The real room has never been measured.**

## 2. The z ladder in force

All mm, canvas frame, `z = 0` at the **top surface of the paper**.

```
  1699.82   grid top  = top of construction  (2336.50 above the floor)
  1623.62   grid underside — the runway beams' soffit, where posts hang
  1496.62   gusset bottom
  1048.40   clamp stack top
   952.70   plate top
   940.00   MOUNT PLANE — the arm bolts to the plate's underside
   905.02   post bottom (34.98 of post over-runs past the plate)
     0.00   paper top
    -2.00   table top / paper underside
   -27.38   cage leg bottom
  -636.68   floor
```

## 3. What is modelled

**77 static bodies**, every one an axis-aligned box with a named source.

| | count | what |
|---|---:|---|
| cage | 68 | 4 perimeter rails, 4 corner legs, 6 runway beams (3 rows × 2), 24 drop posts (6 arms × 4), 24 gussets, 6 clamp stacks |
| mount | 6 | the robot base plates, 225.82 × 190 × 12.7 |
| table | 2 | the table, and the floor plane |
| canvas | 1 | the paper, 1803.4 × 3630.64 × 2.0 |

Plus, per arm × 6:

- **visual** — the manufacturer's glTF meshes, **with their real textures**
  (§4)
- **collision** — the manufacturer's own collision shells (`installation.urdf`)
  or the audited capsule set (`installation_capsules.urdf`). §5.
- **pen holder** — housing + cap, re-decimated from the raw CAD, with the
  proven 3-cylinder envelope as collision. §7.
- **cable dress** — two translucent conduit loops, **ESTIMATED**, visual only.

### Cage details worth knowing

- **3″ × 3″ (76.2 mm) T-slot throughout**, the drawing's own MTEXT.
- Perimeter frame **2184.4 × 4011.64** outside — the 2184.4 is exactly
  86.00 in, the original frame's own cut. Margin 190.5 mm (7.5 in) clear of
  the canvas on all four sides.
- **Three double-beam runways**, 2 × 76.2 side by side (152.4 overall),
  span 2032.0 = 80.00 in, **seam on the row line** so each arm's J1 axis lies
  on it.
- **Per arm a 2 × 2 drop-post cluster**, pitch 317.6 (x) × 76.2 (y), plan
  footprint 393.8 × 152.4.
- The **plate nests between the post pairs**: 240.5 mm slot, 225.82 mm plate,
  **7.34 mm clear each side**. Its centre sits 25.15 mm in canvas **+x** from
  the J1 axis — and that direction is an inference. See §8.
- **Gussets rotated onto the runway's outboard y faces.** In the drawing's
  own orientation two 203.2 mm gussets face each other across a transverse
  pair and need 406.4 mm where only **216.20 mm** exists. Rotated, each plate
  is centred on its own post and the pair clears by **89.20 mm**.

## 4. The textures — solved by vendoring, not by stripping

The vendored glTFs referenced **65 image files that were never copied
across**. Every renderer fell back to flat white and printed a warning per
image per mesh (`Meshcat could not get data for the named uri 'hand_color.png'`).

**All 54 that the arm needs were found**, in the directory the meshes
themselves came from —
`~/git/franka_manipulation_station/assets/franka_description/meshes/visual`.
The 27 this model uses (3 maps × link0–7 + hand: colour, normal,
occlusion-roughness-metallic, 2048² each) are vendored **byte-identically**,
with a sha256 per file in the manifest. No resampling: a resized texture is a
texture whose provenance is "the tool resized it".

The `.ktx2` twins are **not** copied. VTK says so itself — *"glTF extension
KHR_texture_basisu is used in this model, but not supported by this loader"* —
so they were 0.8 MB of dead weight and one warning per mesh. The extension
declaration and the `.ktx2` image entries are removed and the texture table
re-indexed onto the PNGs that remain.

The `.bin` geometry buffers come too (5.95 MB, the ten meshes this model
mounts, not the 10 MB `cobot_pump` nothing here uses). Pointing the buffer at
`../../../franka_description` works in Drake and VTK but **trimesh refuses
it** — its resolver will not follow a URI out of the glTF's own directory. An
asset package only some loaders can open is not an asset package.

Result: `assets/system_model/` is **self-contained, 18.1 MB**, and opens in
Drake, VTK, trimesh, or anything else that reads glTF 2.0, with zero dangling
references. `scripts/check_system_model.py` §8 and
`test_no_gltf_asks_for_a_texture_that_is_not_there` both pin it.

## 5. Collision geometry — the unaudited spheres are gone

`assets/proposed_rig/installation.urdf` carries the vendored Panda's **66
collision spheres per arm**. Nothing in this repo has ever audited them, and
they are not the model any certified number was earned against — the
generator's own header says so. **They are not in this model.**

Two geometries replace them, and both are backed by measurement:

**`installation.urdf` — the manufacturer's collision shells.** Ten OBJs from
`vamp/resources/panda/meshes/collision`, vendored byte-identically.
`scripts/collision_audit.py --validate` compares each shell's AABB with the
FR3's own collision box in the station's `fr3_franka_hand.urdf`: eight of nine
agree to **≤ 0.5 µm** and link6 to **0.344 mm**, i.e. the Panda and FR3
collision shells are the same object. Drake reports their convex hulls at the
same volume to 0.1 % on nine of ten (link6 4.2 %) — they are **already convex
shells**, so nothing is lost by using them as URDF collision, and Drake's
signed-distance queries work on them directly.

**`installation_capsules.urdf` — the audited capsule set.**
`selfcoll.BODY_CAPSULES`, 31 per arm, emitted as `drake:capsule`. Every radius
was fitted to those same manufacturer meshes by
`scripts/self_collision_audit.py`, and each row in the source records the
exact mesh maximum it rounds up from.

Two opinions on the same question, from the same measurement, in the same
frame. Read a disagreement between them as a real result.

The tool is a cylinder set in both (3 envelope + 1 graphite). The cage is
boxes. **No sphere carries a collision role anywhere in either file.**

One shell needed care. `finger.obj` is a single mesh used for both fingers and
it lies **entirely on one side** — y ∈ [−0.0001, 0.0264] — so the right finger
has to carry the same `Rz(180)` mirror its *visual* carries, or its collision
geometry sits up to **52.8 mm** from the finger. The generator derives it from
the vendored URDF's own visual origin rather than hardcoding it, and
`test_every_collision_shell_lands_on_its_own_link` holds every shell's AABB to
its link's visual (link0 exempted, because its disagreement *is* the base-cable
finding of §8.2).

## 6. Reconciliation with `mounts.py` — the re-cert queue

`aris_sixarm/mounts.py` is **not changed by this work**. It carries the
schematic keep-out envelope every certified number was earned against.
`system_model.reconciliation()` lists every place the two disagree, machine-
readably, in `model_manifest.json`:

| item | Δ | which way it cuts | action |
|---|---:|---|---|
| **ceiling datum** | 716.38 mm | code is **conservative** (booms ~700 mm too long) | survey the room; the fabricator's post is 718.6, not 1435.0 |
| **drop cluster vs boom column** | 193.80 mm | **neither contains the other** — 393.8 × 152.4 real vs a Ø200 column | **RE-CERT REQUIRED before fabrication** |
| mount plate thickness | 37.30 mm | code is conservative (50 modelled vs 12.7 real) | none; known conservatism |
| steel below the mount plane | 34.98 mm | model has steel the code does not; 60 mm of the 95 mm chain gap still spare | confirm at re-certification |
| **base cable pass-through** | 230.70 mm | **neither model has it, and the steel to be cut is not drawn** | §8 |
| mount height | 90.00 mm | `docs/BUILD_SHEET.md` still publishes 850.0 against the 940.0 in force | re-issue the build sheet |
| cage legs | 1727.20 mm | model has steel the code does not; stands 190 mm clear of the canvas | confirm once count and position are decided |

**The two that block fabrication are the drop cluster and the cable
pass-through.**

## 7. The pen holder — RED FLAG

The placement is **inferred, not read**. The 2026.08.19 CAD delivery is eight
printed parts as STL + SLDPRT pairs with **no assembly file** and **no
fingertip cradle geometry**. `rig_final.penholder22_T_hand` puts the grip
centre on the hand TCP and aims the bore along the **planner's** TCP-to-tip
ray.

The housing's own machined flats clock at **23.0°** (measured off the STL).
The planner's ray leans **45.0°**. Those are different numbers and only one of
them can be right about the real part. This model uses the planner's ray,
which is the assumption that makes the geometry consistent with the
gate-validated tool transform — and the 3-cylinder envelope is only proven
conservative *for that placement* (under the 23° hypothesis 211 696 housing
vertices escape it, worst +19.43 mm).

**Pending Pete's caliper measurement.** There is an unexploited lead:
`raw_slack_file_dump/Pen holder cad(1).zip` contains a nested
`Natural hold assembly - closed.zip` which **does** hold `.SLDASM` files and a
`Franka_Finger_FR3 Fingertip only.SLDPRT` — for the older 10° build, but it is
ground truth about how a holder sits in a hand.

Meshes are re-decimated from the raw STLs, weld-then-decimate, 8000 faces per
part:

| part | raw faces | out | extent lost |
|---|---:|---:|---:|
| housing | 174 772 | 8 000 | **0.1122 mm** |
| cap | 30 770 | 8 000 | **0.0048 mm** |

The 3-cylinder envelope is re-proved against the result on every run: worst
escape **+0.109 µm**, against a 10 µm bar. (Not zero: the envelope's radii
were fitted to a coarser 5000-face decimation where every vertex sat inside;
a finer mesh follows the true barrel more closely and pokes a tenth of a
printer layer proud of one band. It is reported, not hidden —
`meshes.holder_decimation.envelope_worst_escape_m` in the manifest.)

## 8. Open physical questions — for Pete

All seven are in `system_model.OPEN_QUESTIONS` and in the manifest, each with
what rides on it and what would answer it.

1. **Plate offset direction.** Does the plate's 25.15 mm offset run in canvas
   +x or −x? The drawing gives it for an arm whose front faces +X; this rig
   clocks every arm the other way, so it flips — and *which edge of a Franka
   base plate is its front* is itself an inference. **7.34 mm rides on it:**
   offset the wrong way the plate does not fit between the posts at all. Fix
   either by measuring the real plate, or by opening the post gap to 304.8 mm
   (posts at axis ± 190.5), which fits either way with 14.4 mm each side.
   *Blocks: fabrication of the drop clusters.*

2. **Base cable pass-through.** *Found by this model, by nothing else.* The
   manufacturer's link0 visual carries the connector and cable stub **230.7 mm
   past the base flange** (677 of 40 142 vertices). Every arm here is
   inverted, so that 230.7 mm points straight **up**: through the 12.7 mm
   plate, through the 95.7 mm clamp stack, and 122 mm on into the drop
   cluster. The collision shell stops dead at the flange, so no clearance
   check in this repo has ever seen it. Vertically there is room (the posts
   run 905.0 → 1623.6, the cable tops out at 1170.7) but **the plate and the
   clamp stack are solid across it today**. *Blocks: fabrication of the plate
   and the clamp stack.*

3. **Ceiling survey.** How high is the real room above the paper? The drawing
   defines no room ceiling; every ceiling number in this repo descends from a
   floor-referenced dimension. 716.4 mm of modelled boom rides on it, and so
   does whether the cage fits the room at all. *Blocks: cutting the grid.*

4. **Pen holder cradle.** §7. *Blocks: pen-tip calibration.*

5. **Cage legs.** Correcting the datum turns the cage from something hanging
   off a room ceiling into something standing on the floor — and the re-issued
   cut list has no legs in it, because it assumed the former. Four corner legs
   are modelled, mirroring the original's exactly, but a 4.01 m frame on four
   legs has **no precedent**: the original spans 2.08 m. The unsupported
   perimeter span between runways is 1210.2 mm. *Blocks: fabrication.*

6. **Gusset attachment.** The rotation onto the y faces resolves a hard
   interference and buys 89.2 mm, but the drawing has no gusset part number
   and the attachment is an envelope, not a detail. *Blocks: fabrication.*

7. **Cable dress.** Nothing about it has been measured. The 30 mm loops here
   are the middle of the collision audit's 20–40 mm estimate, **visual only
   and off in collision** — turning them on is a re-certification, not a
   switch. *Blocks: nothing yet.*

## 9. Provenance classes

Every body names exactly one:

| class | meaning | static bodies |
|---|---|---:|
| **DRAWING** | lifted from the original drawing, through `rig_final.FRAME_BOXES_W_CM` / `ARM_MOUNTS_W`. The entity is named. | 47 (61.0 %) |
| **CODE** | a constant this repo plans against | 1 (1.3 %) |
| **AUDIT** | measured by a script here, off CAD or off the manufacturer's meshes | 0 (0.0 %) |
| **ASSUMED** | chosen here; every one is also an open question | 29 (37.7 %) |

That mix covers the **static** bodies only — the cage, table and canvas. The
29 ASSUMED are the 24 rotated gussets, the 4 cage legs and the table
footprint. The arms, their collision geometry and the tool are not boxes and
carry their own class: **arm visual, collision shells and capsules are all
AUDIT**; the tool **geometry** is AUDIT and its **placement** is ASSUMED.

## 10. Validation

| check | result |
|---|---|
| pydrake parse, all three URDFs | clean; 186 bodies / 42 DOF, environment 78 bodies / 0 DOF |
| pen tip vs `frames.fk`, 25 configs × 6 arms | **worst 5.26 pm** (bar 10 pm) |
| pen tip orientation | worst 1.33e-11 on a matrix entry (bar 1e-10) |
| every cage body vs `system_model` | exact |
| collision spheres in either file | **0** |
| arm collision bodies | 66 manufacturer shells / 186 audited capsules |
| dangling glTF references | **0** |
| URDF regeneration | byte-stable, twice, equal to what is committed |

The generator writes every number as an **exact float64 round trip**
(`repr`), so `float(text) == value` and there is no truncation error to pick a
tolerance for. The proposed rig's 1e-9 writer costs 1.06 nm at the far row,
where `y = 3.0255333333333336` truncates to `3.025533333`; that is why the bar
here is picometres rather than nanometres.

## 11. Renders

`scripts/render_system_model.py` → `out/` (gitignored):
`system_model_three_quarter.png`, `_elevation.png`, `_plan.png`,
`_drop_cluster.png`, `_holder.png` (2400 px, PBR, shadows) and
`system_model.html` (static meshcat, 32.6 MB). Arms are posed at
`Q_PARK_PROPOSED`.

## 12. Relationship to `assets/proposed_rig/`

`assets/proposed_rig/` is **untouched and still current** for what it is: the
asset every certified number was checked against, and the one
`scripts/collision_audit.py` reads. This is an addition. See
`assets/proposed_rig/README.md` for which to reach for.
