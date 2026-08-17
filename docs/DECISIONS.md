# Decisions — numbers the upstream repos disagree on

The `Aris_Kindt` branches carry conflicting values. Every pick below is revisitable;
change it in ONE place (`fleet.py` / `frames.py`) and rerun `scripts/run_atlas.py`
plus `tests/`.

| Quantity | Values in the wild | We use | Why |
|---|---|---|---|
| Inverted base → paper | 0.92 (verbal) · 0.924 (cm sheet, "authoritative") · 0.97 (GUI SCENE_ARGS) · 1.8542 world (booth URDF) · 1.00 (IKA answer) | **1.00** (`H_INV_DEFAULT`) | IKA 07-12: h=100cm + ≤15° tilt ⇒ solid ⌀154 cm, no center hole; "do NOT lower the mount". Pass `--h-inv 0.924` to model the rig as measured 2026-07-07. |
| Arm base XY | booth URDF (3 seats only) · presets (several) · registry `base_world_m` (only arm 13 surveyed) | **six_arm_display_ADJUSTED_h80.json** | Only complete 6-arm layout on record. |
| Floor base z | 0.6477 world (surveyed, arm 13) vs table top 0.6096 (URDF) / 0.635 (revised scene) | **+0.0127 above paper** | 0.6477 − 0.635 = exactly one base plate (12.7 mm). |
| Pen tip offset | 0.110 below TCP (IKA gate) · 0.209 from flange (ika_workspace) · 0.15 (variable_pressure yaml) · 0.015 (Isaac) | **0.110 below TCP** (`PEN_EXT`) | Gate-B validated against the real touchdown at MZ=0.924. |
| Arm 2 mount | registry: wall plate (pitch π/2, standoff 0.70) · presets/GUI: topdown boom | **topdown** | Matches every preset; the wall experiment still draws on the same paper. Revisit if arm 2 goes active as a wall arm. |
| Sheet size | 218×170 cm (1-arm) · 90×150 (raster) · 360.7×196.1 (6-arm preset) | **360.7×196.1 cm** | The 6-arm planner preset sheet. |
| σ_min gate | 0.08 (ika_workspace default) vs 0.14 (ika_plan strict mask) | **0.14** | Field-validated dual-mask value. |
| Joint margin gate | normalized 0..0.5 (workspace_map) vs absolute rad ≥0.30 (IKA) | **0.30 rad absolute** | IKA binding plan; comparable across joints. |

Open issues inherited from upstream (not resolved here):
- Base yaw of the inverted arms vs the table axes was never surveyed → per-arm atlases
  are yaw-invariant disks; table-edge clipping per arm is NOT modelled.
- 5 of 6 `base_world_m` entries in `arm_registry.py` are unsurveyed (None).
- Collision checks are the IKA proxies (paper clearance 2 cm, boom cylinder r 0.12);
  real rig geometry lives in `aris_planning_scene.py` upstream and should replace them
  before trusting near-boom / near-edge cells.
- Kinematics use the Panda visual/DH model (identical geometry to FR3; limits are FR3).
