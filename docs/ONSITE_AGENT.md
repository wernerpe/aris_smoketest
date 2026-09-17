# On-site agent runbook — read this first, you have zero context

You are on a machine at the installation. Six Franka FR3 arms hang upside down
over a paper table; today only the two middle-row positions are used. **WHICH ARM
IDS ARE MOUNTED IN THEM IS NOT KNOWN — identify the arms first** (2026-09-17:
"may be 97 and 71, position unknown"). The PLAN is named after the POSITION:
slot 31 is the left-middle position, slot 71 the right-middle, and `send
--as-arm <id>` dispatches a slot's file to whatever arm is actually bolted
there. The job today: **write the word "unknown" under each arm, one arm at a time,
first floating 30 mm above the paper, then on the paper.** Nothing moves two arms
at once. Pete (the PI) is with you; he holds the physical e-stop, which is the
only abort that counts.

This repo is the PLANNER. It produces a CSV of waypoints per arm. A separate,
already-deployed executor on the "operator PC" turns that CSV into motion. You
never talk to a robot from this repo directly; you copy a file and start a script
over ssh.

---

## 0. Facts you must not get wrong

**Everything site-specific is `config/site.json`** — operator host, per-slot
physical arm id, control-box IP, DDS domain, measured paper-z, whether the
mounting is confirmed, and the `RTFF_*` environment the dispatch line carries.
Run `day1.py site` before anything else, edit it with `day1.py site --set
KEY=VALUE`, and never hand-edit an address into a script: there is none in
there to edit (`tests/test_day1.py` enforces that).


| fact | value |
|---|---|
| mounting height h | **0.970 m** from the paper top to the underside of the mounting plate |
| slot 31 / 71 base, canvas frame | x 0.5967 / 1.2067 m, y 1.8153 m (the seam line), both inverted, same clocking. These are POSITIONS; the CSV's poses are in the position's base frame and are right for whichever arm is mounted there. |
| which arm ids are mounted | **UNKNOWN. Identify first**, then record it: `day1.py site --set slot31.arm=97 --set slot31.mounted=true`. Nothing downstream checks this for you. |
| tool tip in the hand frame | (0.0860, 0, 0.1494) m — from a photograph, **not a touchdown**; the floating pass exists to check it |
| **everything site-specific** | **`config/site.json`** — operator host, each slot's physical arm id / control-box IP / DDS domain / measured paper-z / mounting confirmed, and the `RTFF_*` environment the dispatch line carries. `day1.py site` prints it, `day1.py site --set KEY=VALUE` edits it, every run prints it, and the GUI reads and writes the same file. **No address is hard-coded in any script** (a test enforces that). |
| operator PC | `diemut@192.168.50.2`, key-only ssh (`ssh host 'cmd'`, never `bash -lc`) — from `config/site.json` |
| control boxes | arm 31 → 192.168.50.12, arm 71 → 192.168.50.14, arm 97 → 192.168.50.15. The id is the DDS domain AND the `ARM_ID` the supervisor reads. |
| executor | `~/RTff/draw_rtff_supervised.sh` on the operator PC; it **ignores the joint columns and timestamps** in our CSV, paces by arc length at 0.02 m/s, and keeps its own arm configuration. Today's runs therefore test the tool, the plane and the pose path — not the planner's redundancy resolution. Say so if asked. |
| first move | the executor ramps in a straight line from wherever the arm is to row 0 of the file, uncertified. **MEASURED: row 0 is NOT the park.** The certified PROGRAMME starts and ends at the park, but the exporter begins the CSV at the first frame whose tip is 50 mm clear of the paper — about **0.6 m and 4 rad** from the park on the day-1 word files. Put the arm at the park (operator's `go_start_pos.py`, position control), then **look at the gap `send` prints** before starting. |
| end of file | the executor lifts 80 mm along the pen axis unless `RTFF_DEPART_LIFT=0`; our dispatch line sets it, because the certified programme already ends at the park. |
| paper plane | no automatic gate for inverted arms, and `RTFF_CONTACT_DESCEND=0` means it does not feel for it either: the executor goes to the height baked into the file and the impedance spring holds contact. Measure it by hand first (operator tools `jog_descend.py` or `probe_surface.sh`). **`--paper-z Z` is how far the REAL paper sits ABOVE the modelled plane**, and it lifts the whole plan by Z. After a floating pass that asked for 30 mm and measured H, pass `--paper-z (0.030 - H)`; record it with `day1.py site --set slot31.paper_z=<Z>` and it becomes the default for that slot. |
| hover sign | `fr3_link0`'s +z points **down** on an inverted arm, so "30 mm above the paper" is `z_paper - 0.030` in the base frame. Verified in every file we write (`rows.tip_above_paper_base_m` in the json). |
| slot 31 quirk | it refuses a line exactly on the seam line y = 1.815 (go-home leg fails the paper gate); the word is placed at `--dy -0.10` by default. |
| the CSV | 19 columns: the v2 contract's 18, plus **`t_s` last** (the planner's pacing, same clock as the npz). A positional reader of the first 18 is unaffected. The executor ignores `q1..q7` and `t_s`. |
| one arm at a time | always. The other arm is at its park, which is what every certificate assumes. |

Do not edit gate constants, `frames.py`, `layout.py`, `mounts.py`, `scene_check.py`.
Do not run pytest with `ARIS_RIG`/`ARIS_TOOL` exported. Do not push to any remote.

---

## 1. Install and verify (10 minutes)

```bash
sudo apt install -y build-essential libeigen3-dev python3-venv
git clone -b aris2 git@github.com:wernerpe/aris_smoketest.git && cd aris_smoketest
python3 -m venv .venv && . .venv/bin/activate && pip install -U pip
pip install -r requirements-site.txt
pip install ./third_party/franka_analytical_ik      # ~5 s compile, vendored
pip install -e '.[gui]'
python -c "import aris_sixarm.ik as ik; print('batch:', ik._IK.has_batch)"   # True
python -m pytest tests/test_gates.py tests/test_hardware_prep.py -q          # 20 passed (own process!)
python -m pytest tests/test_day1.py -q
export ARIS_RIG=proposed ARIS_TOOL=lateral
```
If any of that fails: `docs/SITE_SETUP.md` §5 lists every failure seen so far.

Network check: `ssh diemut@192.168.50.2 'echo ok'` must print `ok`. If it asks
for a password, the key is not installed; stop and tell Pete.

---

## 2. The commands (all in `scripts/day1.py`, `--help` on each)

| command | what it does | output |
|---|---|---|
| `day1.py site` | prints `config/site.json` — operator, slots, arm ids, IPs, paper-z, dispatch env. **Read it first.** `--set KEY=VALUE` edits it | text |
| `day1.py park --arm 31` | prints the park joint vector and park tip pose — where the certified programme begins, and where to drive the arm before streaming | text |
| `day1.py park --arm 31 --from-q q1,…,q7` | certified joint path from the MEASURED joints to the park (RRT) — a check that the straight ramp is clear; the deployed executor cannot follow joint paths | `out/day1/park_31.*` |
| `day1.py word --arm 31 --hover` | the word under arm 31, floating 30 mm up; plans in ~20–50 s | `out/day1/unknown_hover_31.{csv,npz,json}` |
| `day1.py word --arm 31 [--paper-z Z]` | the word on the paper, Z = measured tip height of the paper if it differs from 0 | `out/day1/unknown_31.*` |
| `day1.py line --arm 31 --from x,y --to x,y --name N [--hover 0.03]` | one straight line, same pipeline, seconds | `out/day1/N_31.*` |
| `day1.py send --arm 31 --file out/day1/<file>.csv --dry-run` | PRINTS the scp and the run line, copies nothing | — |
| `day1.py send --arm 31 --file …` | copies the CSV to the operator PC and prints the run line | — |
| `day1.py send --arm 31 --as-arm 97 --file …` | the same, but dispatched to the PHYSICAL arm 97 standing in slot 31's position | — |
| `day1.py send … --live` | the same, and runs it over ssh | the arm moves |
| `python -m aris_sixarm.gui` | http://localhost:8765 — Day 1 panel does the above with buttons, a 3D viewer, and a "Run on arm" group with a typed confirmation | — |

Every planning command prints ONE line `PASS …` or `FAIL …` with the gate numbers
(inter-arm, frame, self, paper, joint speed as a fraction of the FR3 limit). A
FAIL writes **no CSV**. Never hand-edit a CSV.

---

## 3. What to check before an arm moves — every time

1. **Stack health:** `ssh diemut@192.168.50.2 'bash ~/RTff/aris_hold.sh stack 31'` → must contain `STACK HEALTHY`. This script was written for arms 13/17; if it rejects 31, show Pete the raw output.
2. **The arm is at its park pose** (compare the robot's joints to what `day1.py park --arm 31` prints; ±0.02 rad). If not, the operator's `go_start_pos.py` moves it there under position control. `day1.py park --arm 31 --from-q <measured joints>` plans and certifies a collision-free joint path from where it actually is to the park — that is a CHECK that the way is clear (the deployed executor follows tip poses only and cannot execute a joint path); the position-control move is still how you get there.
3. **The paper height is measured** and baked (`--paper-z`, or `site --set slot31.paper_z=Z`) for on-paper runs; for the floating run the baked plane is the nominal one and the 30 mm is the check.
4. **The arm id is identified and recorded** (`site --set slot31.arm=<id> --set slot31.mounted=true`), and `send` is given `--as-arm <id>` if it differs from the slot.
4. **The other arm is at its park** and nobody is under the arm.
5. **Pete has the e-stop in hand.**
6. **Ask Pete to confirm which way the fingers and the pen holder are mounted.**
   This is still ambiguous: the holder can sit in the jaw either way round, and the
   model's tool tip (86 mm to one side of the hand) depends on it. Use the GUI's
   "Show current pose" (reads the arm's live joints from the operator PC and poses
   the model with the tool drawn) and hold it next to the real hand: the pen must
   jut out on the same side and lean the same way. If it does not, STOP and tell
   Pete — every plan assumes the modelled side. Take a photo either way.

---

## 4. The run ladder, in order

For arm 31, then repeat everything for arm 71 (`--arm 71`, control box .14, domain 71):

```bash
day1.py word --arm 31 --hover                       # PASS line, tip ≈ +27..30 mm
day1.py send --arm 31 --file out/day1/unknown_hover_31.csv --dry-run   # read the line it prints
day1.py send --arm 31 --file out/day1/unknown_hover_31.csv --live      # floating pass
```
Watch: `ssh diemut@192.168.50.2 'tail -f /tmp/rtff_draw_arm31.log'`.
**Pass:** the pen traces the word in the air, about 27–30 mm above the paper by
eye/ruler everywhere, no reflex, no stop, the arm returns to its park. If the
height is wrong by more than 5 mm the tool transform is off: record the observed
height and tell Pete before anything touches paper.

Then on paper:
```bash
day1.py word --arm 31 --paper-z <measured>          # only if measured ≠ 0
day1.py send --arm 31 --file out/day1/unknown_31.csv --live
```
**Pass:** a continuous "unknown", no gaps inside strokes, no gouging, arm returns
to park. The first passes run in `RTFF_MODE=observe` (open-loop depth); closed-
loop force is a later step.

Stop / hold at any time (besides the e-stop):
```bash
ssh diemut@192.168.50.2 'bash ~/RTff/aris_hold.sh hold 31'      # keeps the checkpoint
ssh diemut@192.168.50.2 'ARM_ID=31 source ~/impedance_helpers/arm_env.sh && arm_pkill rtff_pathway_exec'
```

---

## 5. What to record and report back

- The PASS lines of every planning command (copy them verbatim).
- The observed floating height per arm, and the measured paper height you baked.
- Whether `aris_hold.sh stack 31/71` accepted the arm, and the raw output if not.
- The operator log `/tmp/rtff_draw_arm<N>.log` and the force logs in
  `~/RTff/force_logs/` (copy them back: `scp diemut@192.168.50.2:/tmp/rtff_draw_arm31.log out/day1/`).
- A photo of each drawn word.
- Anything the arm did that the 3D viewer did not show.

---

## 6. If something is off

| symptom | what it is | do |
|---|---|---|
| `FAIL … inter-arm` or `frame` | the plan cannot be certified with the other arm parked / against the steel | move the word (`--dy`, `--width`), never lower a gate |
| `FAIL … joint speed` | the pacing exceeds an FR3 joint limit | report it; do not run |
| word planned but 0 strokes for one arm | reach/height problem; at 0.940 arm 31 draws nothing | check h really is 0.970 |
| executor stops during a pen-up move | the deployed controller reconfiguring the arm under its own nullspace | hold, note the row from the log, tell Pete |
| floating height off by > 5 mm | tool transform wrong | stop; Pete decides on a touchdown calibration |
| `aris_hold.sh` rejects arm 31 | the health script only knows 13/17 | show Pete; the operator's own bring-up for inverted arms is Diemut's |

More detail, with every number and its provenance: `docs/HARDWARE_DAY1.md`,
`docs/SITE_SETUP.md`, `docs/SYSTEM_MODEL.md`.
