# `aris/jazzy-dev` — offline ROS 2 Jazzy harness for the operator stack

A self-contained Docker image that reproduces enough of the remote **operator
PC**'s ROS 2 Jazzy environment to *compile* and *smoke-test*, on a machine with
no ROS installed:

* the C++ `ros2_control` effort controller **`cartesian_impedance_controller`**
* the rclpy executor **`rtff_pathway_exec.py`**

Nothing in here talks to a robot, and nothing is installed on the host.

---

## Safety properties (this is a shared lab machine with live robots)

| Property | How |
|---|---|
| No host installs | everything is inside the image; the host only runs `docker` |
| No host networking | `compose.yaml` / `test_smoke.sh` use the **default bridge**; there is no `network_mode: host` |
| No privilege | no `--privileged`, no `cap_add`, no `/dev` mounts, no `--ipc=host` |
| No ROS traffic leaves the container | `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST` confines DDS discovery to the container's own network namespace — it cannot see, or be seen by, any node on the lab networks |
| Nothing dials a robot | no `robot_ip` is ever set; `franka_hardware` is compiled but never instantiated against real hardware; T4 uses `mock_components/GenericSystem` |
| Host sources safe | every host bind mount is `:ro` except `./out` |
| Bounded parallelism | `MAKEFLAGS=-j8`, `colcon --parallel-workers 2` in both the image build and the overlay build |

`/home/franka/git` and `/home/franka/franka-net` are never referenced.

**Be precise about what the bridge does and does not do.** A default/user-defined
bridge network still has outbound NAT, so the container's *IP layer* can route to
128.30.16.50/.55 exactly as the host can — DNS inside the container resolves them.
What makes this harness safe is not the bridge: it is that (a) DDS discovery is
pinned to `LOCALHOST` so no ROS node here will ever peer with a node on a robot,
and (b) nothing in the harness is ever given a robot address. If you add anything
that takes an IP, that reasoning has to be re-done.

---

## Quick start

```bash
cd /home/franka/aris_project/aris_sixarm/docker/jazzy

./build.sh          # build aris/jazzy-dev:d812aab   (~3.5 min cold, ~1 s warm)
./test_smoke.sh     # run T1..T6 inside the container, PASS/FAIL per item

# interactive shell with everything sourced
docker compose run --rm dev
```

Logs from a smoke run land in `./out/logs/`.

---

## Swapping the controller source

The stale 350-line copy in the worktree is the default. When the live 378-line
operator file arrives, point one variable at its package directory — the one
that **contains `package.xml`** — and nothing else changes:

```bash
ARIS_CONTROLLER_SRC=/path/to/cartesian_impedance_controller ./test_smoke.sh
# or
ARIS_CONTROLLER_SRC=/path/to/cartesian_impedance_controller docker compose run --rm dev
```

It is bind-mounted read-only at `/src/controller` and built **out-of-source**
into the separate overlay workspace `/aris_ws`, so the source tree is never
written to and the heavy `franka_ros2` image layer is never invalidated.

Other knobs:

| Variable | Default |
|---|---|
| `ARIS_RTFF_SRC` | `/home/franka/aris_project/worktrees/aris2-rtff` → `/src/rtff:ro` |
| `ARIS_CONTROLLER_SRC` | `${ARIS_RTFF_SRC}/aris_kindt_dwatkins_ztouch_control/ros2_ws/src/cartesian_impedance_controller` |
| `ARIS_OUT_DIR` | `./out` → `/out` (the only writable mount) |
| `ARIS_IMAGE` / `ARIS_TAG` | `aris/jazzy-dev` / `d812aab` |
| `ARIS_COLCON_WORKERS` | `2` |
| `ROS_DOMAIN_ID` | `13` |
| `ARIS_T6_DURATION` | `20` (seconds) |

---

## What is in the image

| Component | Pin | Source |
|---|---|---|
| Base image | `ros:jazzy-ros-base` (amd64 digest `sha256:bab7e640bf79cd84957e4e18fcba7d87efc3385b4e3f36a32eeca01638e43206`), Ubuntu 24.04.4 LTS, `ros-jazzy-ros-base 0.11.0-1noble.20260616.084325` | Docker Hub |
| `franka_ros2` | **`d812aabfab939317bd16515a392ad2cf1928fa83`** ("update dependency repositories", 2026-05-04), branch `jazzy` | `github.com/frankarobotics/franka_ros2` |
| Aris patches | `0001`, `0002`, `0003` applied as an ordered stack on top of `d812aab` | `patches/` |
| `libfranka` | **`0.20.5`** (`3cc708cafa1cdba49ceaab536a14216dfb7bf48a`), with its `common` submodule | source, built in-tree by colcon |
| `franka_description` | **`2.7.0`** (`ec8789e18df38c1d08a694ee61ebc9b9495e22dc`) | source |
| `ros2_control` | `4.45.2-1noble.20260615.*` (`controller-manager`, `controller-interface`, `hardware-interface`) | **apt binary** |
| `realtime_tools` | `3.11.0-1noble.20260615.154205` | apt binary |
| `joint_state_broadcaster` | `4.40.1-1noble.20260615.171040` | apt binary |
| `pinocchio` | `ros-jazzy-pinocchio 4.0.0-2noble.20260604.104112` | apt binary |
| `xacro` | `2.1.1-1noble.20260519.011123` | apt binary |

`franka_ros2` packages actually built: `franka_msgs`, `franka_hardware`,
`franka_semantic_components`, `franka_robot_state_broadcaster`, plus
`franka_description`. Everything else in the repo is **deleted from the source
tree** during the build so neither colcon nor rosdep ever sees it.

### Layout inside the container

```
/opt/ros/jazzy          ROS 2 Jazzy binary install
/opt/franka_ws          franka_ros2 subset + libfranka + franka_description (built at image build)
  └── src/franka_ros2   the git checkout at d812aab, patches applied — inspectable
/aris_ws                overlay workspace; our controller is built here at RUN time
/src/controller  (ro)   bind mount: controller package source
/src/rtff        (ro)   bind mount: the rtff worktree (rtff_pathway_exec.py etc.)
/src/aris_sixarm (ro)   bind mount: the planner repo; on PYTHONPATH, never pip-installed
/src/franka_analytical_ik (ro)  bind mount: the compiled _franka_ik*.so aris_sixarm.ik loads
/opt/sil-venv           venv with pydrake (--system-site-packages: rclpy/franka_msgs/numpy stay)
/out                    writable: logs and test artifacts
/opt/aris_test          the smoke tests (baked in, and bind-mounted over for live editing)
/opt/patch_report.txt   per-patch `git apply` result recorded at build time (read by T2)
```

`entrypoint.sh` sources `/opt/ros/jazzy`, `/opt/franka_ws/install`, and
`/aris_ws/install` (if built), in that order.

### uid

The image builds a non-root user `user` with **uid/gid 1000** — matching the
host user `franka`, so bind-mounted sources are readable and anything written
to `/out` lands with sane ownership. Ubuntu 24.04 ships a stock `ubuntu` user
already holding uid 1000; the Dockerfile deletes it first. Upstream
`franka_ros2` uses 1001 instead; to match upstream:

```bash
docker build --build-arg USER_UID=1001 --build-arg USER_GID=1001 ...
```

`build.sh` defaults to `$(id -u)`/`$(id -g)`.

---

## Reproduced vs NOT reproduced (versus the operator PC)

### Reproduced
* ROS 2 Jazzy on Ubuntu 24.04, `/opt/ros/jazzy` binary install.
* `franka_ros2` at the operator's exact commit `d812aab`, **with** the three
  Aris-local patches applied in order.
* `libfranka` 0.20.5 and `franka_description` 2.7.0 built from source, exactly
  the versions `dependency.repos` pins at `d812aab`.
* `find_package(Franka 0.19.0 REQUIRED)` resolves to the source-built
  libfranka 0.20.5, as on the operator.
* The controller's compile-time dependency surface: `controller_interface`,
  `hardware_interface`, `rclcpp`, `rclcpp_lifecycle`, `pluginlib`,
  `realtime_tools`, `geometry_msgs`, Eigen3, `franka_semantic_components`,
  `Franka`.
* The executor's runtime imports: `rclpy`, `geometry_msgs`, `franka_msgs`,
  `numpy`.

### NOT reproduced — deliberate
* **`ros2_control` comes from apt, not source.** The operator's
  `dependency.repos` vcs-imports `ros-controls/ros2_control` @ `jazzy` and
  builds it. Here it is the Debian binary `4.45.2-1noble.20260615.*`. The apt
  build is a release off the same `jazzy` branch, but it is a *different
  commit* than whatever the operator's source checkout resolved to on the day
  it was built. If a controller-manager behaviour ever differs between here and
  the operator, this is the first thing to suspect.
* **No Gazebo / `gz_ros2_control`, no MoveIt, no RViz2, no
  `joint_state_publisher_gui`.** Skipped entirely.
* **No `zed_description`, `olvx_descriptions_module`, `ros2_robotiq_gripper`,
  `serial`.** These are in `dependency.repos` but unrelated to our stack.
* **`franka_gripper` is not built** — and it is precisely what patches `0002`
  and `0003` modify (see below).
* **`franka_bringup`, `franka_example_controllers`, `franka_fr3_moveit_config`,
  `franka_gazebo_bringup`, `franka_mobile*`, `franka_selfcollision`,
  `franka_spine`, `franka_vision_and_manipulation_kit`,
  `mobile_fr3_duo_trajectory_controller`, and the vendored `realtime_tools`
  are deleted** from the source tree. The vendored `realtime_tools` is replaced
  by the apt `ros-jazzy-realtime-tools 3.11.0`.
* **No real-time kernel, no `limits.conf` / `SCHED_FIFO`, no RT pinning.** The
  operator pins the control loop; this container does not and cannot. Timing
  behaviour here is meaningless.
* **No robot network.** No `robot_ip`, no FCI, no Desk. `franka_hardware`
  compiles and its plugin is installed, but it can never connect.
* **`robot_type`.** The operator's deployed config uses `robot_type: fr3v2`
  (joints `fr3v2_joint1..7`). The T4 mock URDF/params use `fr3` so the joint
  names line up with a plain `fr3_joint1..7` URDF. This is a *test* difference
  only; the package's own `config/cartesian_impedance_controller.yaml` is
  unchanged.

---

## The patches

`patches/` is a verbatim copy of the `patches/` directory from the
`origin/diemut-operator-helpers` branch of `Aris_Kindt`, including its README —
read `patches/README.md`, it is the authoritative record.

Two things matter for this harness:

1. They are an **ordered stack**: `0001` and `0002` touch the same file and the
   same import block. Checking them independently produces a false CONFLICT.
   The Dockerfile applies them strictly in numeric order with `git apply`, and
   records the result of each `git apply --check` to `/opt/patch_report.txt`.
2. They only touch
   `franka_fr3_moveit_config/launch/moveit.launch.py` and
   `franka_gripper/launch/gripper.launch.py` — **neither of which is in the
   subset this image builds**. So the patches are verified to apply here but
   have no effect on anything compiled. They are applied anyway so that
   `/opt/franka_ws/src/franka_ros2` is a faithful copy of the operator tree.

A patch failure does **not** fail the image build; it is recorded and reported
by T2.

---

## Smoke tests

`./test_smoke.sh` → `docker run` → `/opt/aris_test/run_tests.sh`. Exit code is
the number of failed tests.

| Test | What it checks |
|---|---|
| **T1** | the franka subset is present and built against libfranka 0.20.5 |
| **T2** | patches 0001–0003 apply cleanly, in order, onto `d812aab` |
| **T3** | `cartesian_impedance_controller` compiles in `/aris_ws` from the mounted source; the pluginlib index exports `cartesian_impedance_controller/CartesianImpedanceController` |
| **T4** | load / configure / activate under a `ros2_control_node` on `mock_components/GenericSystem` — **activation is expected to FAIL** (see below) |
| **T5** | `python3 /src/rtff/rtff_pathway_exec.py --help` (imports resolve) |
| **T6** | the executor, driven by a fake `franka_robot_state_broadcaster`, publishes `PoseStamped` on `/cartesian_impedance/equilibrium_pose` |
| **T8** | the executor, driven by the **Drake SIL plant** (`aris_sixarm.sil`) presenting contract §3, with a pencil 1 cm shorter than the robot believes — see below |

### Why T4 is expected to fail at `activate`

`CartesianImpedanceController::state_interface_configuration()` asks for, on
top of the 7 position and 7 velocity interfaces:

```cpp
franka_robot_model_->get_state_interface_names()   // fr3/robot_model, fr3/robot_state
```

Those are exported by **`franka_hardware`'s `FrankaHardwareInterface`**, from a
live libfranka connection. `mock_components/GenericSystem` exports only
per-joint `position` / `velocity` / `effort`. So the controller loads (the
plugin is real and discoverable) and configures (parameters only), but the
controller manager cannot satisfy its state-interface claim at activation.
T4 scores a *clean, correctly-reasoned refusal* as PASS and reaching `active`
as FAIL — reaching `active` would mean the interface claim was not what we
think it is.

Observed, and what T4 asserts:

```
### STEP load_controller     Successfully loaded controller cartesian_impedance_controller      rc=0
### STEP configure           Successfully configured cartesian_impedance_controller             rc=0
### STEP activate            Error activating controller, check controller_manager logs         rc=1

[WARN] [controller_manager]: Unable to activate controller 'cartesian_impedance_controller'
                             since the state interface 'fr3/robot_model' is not available.
final state: inactive
```

The controller's own `on_configure` log confirms it really configured:
`cartesian_impedance_controller configured: arm_id=fr3,
topic=/cartesian_impedance/equilibrium_pose, K=[2200.0 2200.0 1500.0 70.0 70.0
20.0]`.

The test harness also stands in for `robot_state_publisher`: Jazzy's
`controller_manager` takes the URDF from a latched topic, so
`test/robot_description_pub.py` publishes it as a `TRANSIENT_LOCAL`
`std_msgs/String` on both `/robot_description` and
`/controller_manager/robot_description`.

### T6 harness

`test/fake_robot_state.py` publishes, at 100 Hz with the executor's expected
QoS (`BEST_EFFORT` / `KEEP_LAST` / depth 10):

* `/franka_robot_state_broadcaster/robot_state` (`franka_msgs/FrankaRobotState`)
  with `o_t_ee` = identity rotation at `[0.5, 0.0, 0.3]` in `fr3_link0`, and
  `o_f_ext_hat_k` = zero wrench
* `/franka_robot_state_broadcaster/current_pose` (`PoseStamped`)
* `/franka_robot_state_broadcaster/external_wrench_in_base_frame` (`WrenchStamped`)

The executor is run in its most open-loop configuration:
`--mode observe` ("open-loop like BASE, just logs the force"), **no**
`--closed-loop`, **no** `--contact-descend`, **no** `--lag-latch`, **no**
`--resume`, on the 3-waypoint `test/wp3.csv` (`travel` → `draw` → `lift`).
`test/eq_pose_listener.py` counts what comes out. A healthy run looks like:

```
[INFO] [rtff_pathway_exec]: RTff: 3 waypoints, mode=observe
[INFO] [rtff_pathway_exec]: pathway 0.1 m; est 3 s; press 4 mm flat
[INFO] [rtff_pathway_exec]: EE REORIENT: slow turn at low lift (10 s, 2 cm hover)
[INFO] [rtff_pathway_exec]: [draw] f_raw=-0.00N f_meas= 0.00N base=-0.00N press=4.0mm
[INFO] [rtff_pathway_exec]: RTff pathway complete
[INFO] [rtff_pathway_exec]: DEPART LIFT: 80 mm off the surface (-pen axis)

EQ_POSE_SUMMARY {"count": 937, "first": {"x": 0.5, "y": 0.0, "z": 0.298394, ...},
                 "last": {"x": 0.52, "y": 0.01, "z": 0.132932, ...},
                 "span_s": 19.112, "rate_hz": 48.97}
```

The executor does **not** block on the controller: it never waits for
`/cartesian_impedance_controller/set_parameters` unless `--seeds` is given, and
it never waits on a control file. The only thing it waits for is robot state.

Note the executor's own startup gate: `run()` spins for up to **30 s** waiting
for the first robot-state sample and returns `EXIT_OTHER` with
`"no robot state -- is the stack up?"` if none arrives. In `--mode assist` it
additionally waits ~5 s for a wrench and errors with
`"no external wrench (o_f_ext_hat_k) -- cannot close the force loop"`. Both
gates are satisfied by the fake broadcaster.


### T8 harness — the SIL closed loop

T8 is the only test in which the executor is **driven by physics** instead of a
constant. `aris_sixarm/sil/ros_node.py` wraps the package's Drake plant (one
FR3 welded at its rig mount, a pen-tip sphere, a sheet of paper, and a numpy
replica of the deployed Cartesian impedance law) and presents contract §3:

```
publishes 100 Hz   /franka_robot_state_broadcaster/robot_state        FrankaRobotState
                   /franka_robot_state_broadcaster/current_pose       PoseStamped
                   /franka_robot_state_broadcaster/external_wrench_in_base_frame
subscribes         /cartesian_impedance/equilibrium_pose              PoseStamped
                   /cartesian_impedance/joint_reference               JointState (contract §2)
```

`o_t_ee` is the **NOMINAL** tip — what the robot believes — while the plant's
collision sphere sits at nominal + `--tip-error`. That gap is the experiment.

Run it by hand:

```bash
docker compose run --rm dev bash -lc '
  bash /opt/aris_test/t8_sil_run.sh closed_tip-10mm -0.010 closed 60
  python3 /opt/aris_test/t8_report.py /out/t8'
```

or as part of the suite (`ARIS_T8_SKIP=1` skips it, `ARIS_T8_DURATION`
time-boxes each run, `ARIS_T8_RT_FACTOR` slows the simulated clock if the
plant cannot keep up, `ARIS_T8_LATCH=0` drops the two extra latch probes):

```bash
./test_smoke.sh
```

Six runs on `aris_sixarm/sil/examples/line10cm_arm31.csv` (arm 31, `proposed`
rig, lateral holder, real `q1..q7` columns — the manifest's rig/arm/tool are
what the sim is built with, and the script aborts if the CSV's paper plane and
the sim's disagree). The executor is run the way `draw_rtff_supervised.sh`
runs it for an **inverted** arm (GUI `_rtff_inverted_env` + briefing §8.5/§9):
`ARM_ID=31 ARM_INVERSE=1 RTFF_CONTACT_DESCEND=0 RTFF_FORCE_SIGN=1
RTFF_TRAVEL_SPEED=0.02 RTFF_QREF=1`, hover 0.030, landing rate 0.010, air-trim
cap 0.012, `--draw-speed 0.02`.

| run | executor configuration | tip error |
|---|---|---|
| `observe_tip0` / `observe_tip-10mm` | `--mode observe --press 0.010` (open-loop depth) | 0 / −10 mm |
| `closed_tip0` / `closed_tip-10mm` | `--mode assist --closed-loop`, graphite band `0.7–1.0 N ×9`, `--f-max 3.5 --d-max 0.012 --min-press 0.009 --draw-press-floor 0.009 --kp 0.001 --slew 0.0012 --air-trim --air-trim-max 0.012` | 0 / −10 mm |
| `latch_tip0` / `latch_tip-10mm` | the same, **plus** `--contact-descend --lag-latch --lag-gap 0.0010 --lag-ticks 3` | 0 / −10 mm |

The last pair is an **extra**, not the live configuration:
`draw_rtff_supervised.sh` builds `--contact-descend` and `--lag-latch` into the
same `CONTACT_DESCEND_ARG` string, so `RTFF_CONTACT_DESCEND=0` — what the GUI
exports for every inverted arm — makes the lag latch unreachable on arm 31 no
matter what `RTFF_LAG_LATCH` is set to. Without these two runs T8 could say
nothing about the latch.

`t8_report.py` tabulates each run: what the PLANT did (touch fraction, contact
force, where the physical tip really was) against what the EXECUTOR believed
(the press it settled at, its `f_meas`, air-trim, guard trips, exit code).
Two touch fractions are reported and they differ on purpose:

* **draw** — the node's own phase rule. It cannot see the executor's phases, so
  a tick counts as draw when the commanded tip is at or below the paper plane
  (`--draw-tol` 2 mm). This CSV's `lift_start`/`lift_end` rows sit **at** the
  plane, so the executor's 5 s touchdown dwell lands inside this window and
  drags the number down.
* **pressed** — ticks whose commanded depth is ≥ 2 mm below the plane, i.e.
  the executor actually asked for a press. This is the number to compare with
  the host-side open-loop walker
  (`python -m aris_sixarm.sil --csv <the same file> --tip-error 0 --press 0.010`
  → touch 1.000, 7.87 N).

Timing: the node is wall-clock paced, because the executor is — it treats
robot_state older than 0.5 s as stale and 2 s as a reflex. One 100 Hz timer
advances the 1 kHz physics toward `(wall − t0) × --rt-factor`, capped at
`--physics-substep` ticks per slot so the state stream keeps its rate even if
the plant falls behind, and the achieved factor is written into
`summary.json`. On this machine the plant runs ~5× real time with contact, so
`--rt-factor 1.0` is never the binding constraint.

---

## Gotchas

* **`set -u` and ROS.** `/opt/ros/jazzy/setup.bash` dereferences
  `AMENT_TRACE_SETUP_FILES` unguarded. Any script that sources it must not use
  `set -u`. `run_tests.sh` says so in a comment; keep it that way.
* **rosdep `--skip-keys`.** Three keys are skipped:
  `rviz2` and `joint_state_publisher_gui` (`franka_description` `exec_depend`s
  on them; both are GUI-only and would drag in most of the desktop stack), and
  `libfranka` (satisfied from source in `src/`, not from apt).
* **The `franka_ros2` metapackage must be deleted, not just skipped.** Its
  `package.xml` depends on `franka_gripper`, which we remove; leaving it in the
  tree makes `rosdep install` fail with
  `Cannot locate rosdep definition for [franka_gripper]`.
* **libfranka needs its submodule.** `git clone --recursive` — the `common`
  submodule (`libfranka-common`) is required or the build fails.
* **libfranka cmake flags are applied in a separate `colcon build`.**
  `--cmake-args -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF` would otherwise leak
  into every ament package.
* **`ros:jazzy-ros-base` has no `robot_state_publisher`.** Hence
  `test/robot_description_pub.py` rather than an extra apt package.
* **uid 1000 collides with Ubuntu 24.04's stock `ubuntu` user.** The Dockerfile
  `userdel`s it before creating `user`.
* **The overlay build tree lives in a named volume** (`aris_ws`) under
  `docker compose`, so `colcon` is incremental between runs. `test_smoke.sh`
  uses a plain `docker run` with no such volume, so every smoke run rebuilds
  the controller from scratch (~1 min). To reset the compose volume:
  `docker compose down -v`.
* **Editing the tests needs no rebuild.** `./test` is bind-mounted over
  `/opt/aris_test`. A copy is baked into the image so it stands alone.
* **pydrake lives in its own venv, on purpose.** `/opt/sil-venv` is created
  with `--system-site-packages` and drake is installed `--no-deps`, so ROS's
  own numpy 1.26 stays in place under `rclpy`/`franka_msgs` and drake does not
  drag matplotlib, pydot and Mosek into the image. Run the SIL node as
  `/opt/sil-venv/bin/python -m aris_sixarm.sil.ros_node`; plain `python3` has
  ROS but no pydrake.
* **`aris_sixarm` is never installed.** It is bind-mounted read-only and
  reached over `PYTHONPATH`, so the container always runs the working tree.
  `aris_sixarm.ik` loads a compiled `_franka_ik*.so` from **outside** the repo;
  that directory is bind-mounted too and `ARIS_FRANKA_IK_PATH` points at it.
  Host and image are both Ubuntu 24.04 / CPython 3.12, so the cp312 extension
  loads as built — if that ever stops being true, `--start-q` lets the node
  start without IK.

---

## Last verified run — 2026-09-09

`aris/jazzy-dev:d812aab`, image id `c01ee2b0f2d3`, 2 786 928 231 bytes (2 657 MiB).
Cold layer build ≈ 3 min 8 s (83 s base+sources, 105 s workspaces) on top of the
`ros:jazzy-ros-base` pull; warm rebuild ≈ 1 s. `./test_smoke.sh` → **0 failures**.

```
PASS  T1  franka_msgs / franka_semantic_components / franka_hardware /
          franka_robot_state_broadcaster present; libfranka 0.20.5
PASS  T2  all 3 patches applied cleanly, in order, onto d812aab
PASS  T3  libcartesian_impedance_controller.so built from the 350-line worktree
          copy; pluginlib index exports
          cartesian_impedance_controller/CartesianImpedanceController
PASS  T4  type discovered, load OK, configure OK, activate REFUSED, state stays
          'inactive'.  Reason: "Unable to activate controller
          'cartesian_impedance_controller' since the state interface
          'fr3/robot_model' is not available."
PASS  T5  rtff_pathway_exec.py --help exits 0
PASS  T6  924 PoseStamped on /cartesian_impedance/equilibrium_pose over 18.8 s
          (~49 Hz); first z=0.298 (identity quat), last [0.52, 0.01, 0.125];
          executor logged "RTff pathway complete"
```
