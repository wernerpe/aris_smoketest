# Vendored-tree patches — `~/ros2_ws/src/franka_ros2`

That tree is upstream `frankaemika/franka_ros2`, branch `jazzy`. It is NOT ours: there is no
branch of `Aris_Kindt` to push it to, so anything changed there exists on one disk only and is
lost by a workspace rebuild, a reclone, or an upstream pull.

**These patch files are the record.** Captured 2026-08-26.

**BASE: upstream `d812aab` (2026-05-04)** -- NOT today's upstream. The operator tree sits on
`d812aab` and is 96 commits behind `origin/jazzy` (`ee1b79b`, 2026-08-17). Patch `0001` was made
against `d812aab` and does NOT apply to today's upstream: 2 of its 4 hunks fail. Verified
2026-08-26 by rebuilding the file in a scratch directory -- `d812aab` + `0001` + `0002`
reproduces the running file byte for byte.

If franka_ros2 is ever pulled forward, `0001` must be re-made by hand against the new base.
`apply_vendor_patches.sh` reports CONFLICT rather than damaging anything.

| Patch | What it does |
|---|---|
| `0001-...-2026-05-31.patch` | local commit `1265246` — namespaced RViz + inverted world-mount (arm 31). Had lived only on the operator disk since 31 May. |
| `0002-...-2026-08-26.patch` | the gripper gets its own namespace, `fr3_gripper`. Was an uncommitted working-tree edit. |
| `0003-...-2026-09-08.patch` | local commit `3e48835` -- franka_gripper_node respawn=True/10 s: the node aborts on any libfranka exception (Desk self-test UDP timeout, connect while the robot is busy) and stayed dead until the next relaunch. Same node, same launch, never a substitute. |

## Why 0002 exists

The gripper launch was handed the stack-wide `namespace`, which the 2026-06-25 ns-clear set to
``. That was right for every node DDS domain keeps apart — and wrong for this one. Domain
isolation separates two ARMS; it cannot separate the same arm launched twice. At root under the
fixed name `franka_gripper`, a relaunch binds a second node to the same `robot_ip` and silently
CLEARS the gripper calibration, after which every command reports SUCCEEDED while holding
nothing. Arm 13 needed homing after every selftest from 25 June until this was found.

`fr3_gripper` is the path the arbiter already dispatches to (`/fr3_gripper/franka_gripper/...`),
so the launch now agrees with its caller. Everything else stays at ROOT deliberately.

## These patches are an ORDERED STACK

`0001` and `0002` modify the same file and the same import block. Applying or checking them
independently is meaningless -- `--check` will report a false `CONFLICT`. Apply in numeric order,
and use `--verify`, which rebuilds the files from `d812aab` in a scratch directory and diffs the
result against the live tree. That is the check that proved them on 2026-08-26.

## Use

    bash ~/impedance_helpers/patches/apply_vendor_patches.sh --verify  # the real check
    bash ~/impedance_helpers/patches/apply_vendor_patches.sh --check   # advisory only
    bash ~/impedance_helpers/patches/apply_vendor_patches.sh           # restore

Run it after ANY rebuild or upstream pull of franka_ros2. Idempotent — an already-present patch
is skipped, never applied twice. A CONFLICT line means upstream moved under the patch and it must
be re-made by hand; do not force it.
