# ---------------------------------------------------------------------------
# Aris — ROS 2 Jazzy operator-lookalike dev harness
#
# Reproduces enough of the remote operator PC's ROS 2 Jazzy environment to
# COMPILE and SMOKE-TEST:
#   * the C++ ros2_control effort controller `cartesian_impedance_controller`
#   * the rclpy executor `rtff_pathway_exec.py`
# ...on a machine that has no ROS installed.  Nothing in here talks to a robot.
#
# Layering (each FROM is a cache boundary; the heavy franka_ros2 build is in
# `frankaws` and is NOT invalidated by edits to our own sources, which are
# bind-mounted at RUN time into the separate overlay workspace /aris_ws):
#
#   base      ros:jazzy-ros-base + apt deps + non-root user
#   sources   franka_ros2@d812aab + patches + libfranka 0.20.5 + franka_description 2.7.0
#   frankaws  colcon build of the franka subset  (the expensive layer)
#   final     entrypoint, empty /aris_ws overlay
#   sil       pydrake in an isolated venv, for `aris_sixarm.sil` (T8)
#
# The `sil` stage is LAST and is what an untargeted `docker build` produces; it
# adds only pip layers on top of `final`, so the franka build cache above is
# untouched by anything the SIL needs.
# ---------------------------------------------------------------------------

# =========================== stage: base ===================================
FROM ros:jazzy-ros-base AS base

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    ROS_DISTRO=jazzy \
    RCUTILS_COLORIZED_OUTPUT=1 \
    MAKEFLAGS=-j8

# uid/gid: default 1000 to match the host user `franka` so bind-mounted files
# are readable and anything written back lands with sane ownership.  The
# ubuntu:24.04 base ships a stock `ubuntu` user already holding uid 1000, so it
# is removed first.  Upstream franka_ros2 uses 1001; override with
#   --build-arg USER_UID=1001 --build-arg USER_GID=1001
ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=1000

RUN apt-get update && apt-get install -y --no-install-recommends \
        bash-completion \
        build-essential \
        ca-certificates \
        curl \
        git \
        gdb \
        nano \
        python3-colcon-common-extensions \
        python3-vcstool \
        sudo \
        vim \
        libgtest-dev \
        libgmock-dev \
    && rm -rf /var/lib/apt/lists/*

# ros2_control and friends: BINARY from apt (the operator builds ros2_control
# from source via dependency.repos — see README "Deviations").
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-ament-cmake-clang-format \
        ros-jazzy-ament-cmake-clang-tidy \
        ros-jazzy-ament-cmake-ros \
        ros-jazzy-ament-cmake-test \
        ros-jazzy-backward-ros \
        ros-jazzy-control-msgs \
        ros-jazzy-controller-interface \
        ros-jazzy-controller-manager \
        ros-jazzy-example-interfaces \
        ros-jazzy-generate-parameter-library \
        ros-jazzy-hardware-interface \
        ros-jazzy-hardware-interface-testing \
        ros-jazzy-joint-state-broadcaster \
        ros-jazzy-joint-trajectory-controller \
        ros-jazzy-realtime-tools \
        ros-jazzy-ros2-control-cmake \
        ros-jazzy-ros2-control-test-assets \
        ros-jazzy-ros2controlcli \
        ros-jazzy-xacro \
    && rm -rf /var/lib/apt/lists/*

# libfranka 0.20.5 build deps (its package.xml keys: eigen fmt libpoco-dev
# pinocchio tinyxml2; its CMakeLists also wants console_bridge).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libeigen3-dev \
        libfmt-dev \
        libpoco-dev \
        libtinyxml2-dev \
        libconsole-bridge-dev \
        ros-jazzy-pinocchio \
    && rm -rf /var/lib/apt/lists/*

RUN if getent passwd ${USER_UID} >/dev/null; then \
        userdel -r "$(getent passwd ${USER_UID} | cut -d: -f1)" || true; \
    fi \
    && if getent group ${USER_GID} >/dev/null; then \
        groupdel "$(getent group ${USER_GID} | cut -d: -f1)" || true; \
    fi \
    && groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
    && echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /home/${USERNAME}/.bashrc

# ========================= stage: sources ==================================
FROM base AS sources

ARG USERNAME=user
# Operator pins.  Change these ONLY together with the patches/ stack.
ARG FRANKA_ROS2_URL=https://github.com/frankarobotics/franka_ros2.git
ARG FRANKA_ROS2_COMMIT=d812aab
ARG LIBFRANKA_URL=https://github.com/frankarobotics/libfranka.git
ARG LIBFRANKA_VERSION=0.20.5
ARG FRANKA_DESCRIPTION_URL=https://github.com/frankarobotics/franka_description.git
ARG FRANKA_DESCRIPTION_VERSION=2.7.0

ENV FRANKA_WS=/opt/franka_ws
RUN mkdir -p ${FRANKA_WS}/src && chown -R ${USERNAME}:${USERNAME} ${FRANKA_WS}
USER ${USERNAME}
WORKDIR ${FRANKA_WS}

# franka_ros2 at the operator's exact commit.  Full history (not --depth 1) so
# `git apply` / `git log` in the patch check below have a real base to work on.
RUN git clone ${FRANKA_ROS2_URL} src/franka_ros2 \
    && git -C src/franka_ros2 checkout ${FRANKA_ROS2_COMMIT} \
    && git -C src/franka_ros2 log --oneline -1

# The three Aris-local patches, applied as an ORDERED STACK on top of d812aab.
# Result of each `git apply --check` is recorded to /opt/patch_report.txt, which
# test_smoke.sh reads for T2.  A failure here does NOT fail the build — the
# patched files (moveit / gripper launch) are not in our built subset — but it
# is reported honestly.
COPY --chown=${USERNAME}:${USERNAME} patches/ /opt/patches/
RUN set -x; cd src/franka_ros2 \
    && : > /tmp/patch_report.txt \
    && for p in /opt/patches/000*.patch; do \
         if git apply --check "$p" 2>/tmp/err; then \
           git apply "$p" && echo "APPLIED_CLEAN $(basename $p)" >> /tmp/patch_report.txt; \
         else \
           echo "CONFLICT $(basename $p) :: $(tr '\n' ' ' < /tmp/err)" >> /tmp/patch_report.txt; \
         fi; \
       done \
    && sudo cp /tmp/patch_report.txt /opt/patch_report.txt \
    && cat /opt/patch_report.txt \
    && git -C . --no-pager diff --stat

# ONLY libfranka + franka_description from dependency.repos.  gz_ros2_control,
# ros2_control(source), zed, olive, robotiq and serial are deliberately skipped
# (see README).  libfranka needs its `common` submodule.
RUN git clone --recursive --depth 1 --branch ${LIBFRANKA_VERSION} ${LIBFRANKA_URL} src/libfranka \
    && git clone --depth 1 --branch ${FRANKA_DESCRIPTION_VERSION} ${FRANKA_DESCRIPTION_URL} src/franka_description \
    && git -C src/libfranka log --oneline -1 \
    && git -C src/franka_description log --oneline -1

# Packages we will never build: drop them so colcon/rosdep never see them.
RUN cd src/franka_ros2 && rm -rf \
        franka_bringup \
        franka_example_controllers \
        franka_fr3_moveit_config \
        franka_gazebo_bringup \
        franka_gripper \
        franka_mobile \
        franka_mobile_fr3_duo_moveit_config \
        franka_mobile_sensors \
        franka_ros2 \
        franka_selfcollision \
        franka_spine \
        franka_vision_and_manipulation_kit \
        mobile_fr3_duo_trajectory_controller \
        realtime_tools \
    && echo "--- franka_ros2 packages kept ---" && ls

# ======================== stage: frankaws ==================================
FROM sources AS frankaws

ARG USERNAME=user
ARG COLCON_WORKERS=2
ENV FRANKA_WS=/opt/franka_ws
WORKDIR ${FRANKA_WS}
USER ${USERNAME}

# rosdep.  --skip-keys:
#   rviz2, joint_state_publisher_gui : franka_description exec_depends (GUI, unused)
#   libfranka                        : satisfied from source in src/, not apt
RUN sudo apt-get update \
    && rosdep update --rosdistro ${ROS_DISTRO} \
    && rosdep install --from-paths src --ignore-src --rosdistro ${ROS_DISTRO} -y \
        --skip-keys "rviz2 joint_state_publisher_gui libfranka" \
    && sudo rm -rf /var/lib/apt/lists/* \
    && rm -rf /home/${USERNAME}/.ros

# libfranka first, on its own, so its cmake flags do not leak to the ament pkgs.
RUN . /opt/ros/${ROS_DISTRO}/setup.sh \
    && MAKEFLAGS=-j8 colcon build \
        --parallel-workers ${COLCON_WORKERS} \
        --packages-select libfranka \
        --event-handlers console_direct+ \
        --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF

# The franka_ros2 subset the operator's controller links against.
RUN . /opt/ros/${ROS_DISTRO}/setup.sh \
    && . ${FRANKA_WS}/install/setup.sh \
    && MAKEFLAGS=-j8 colcon build \
        --parallel-workers ${COLCON_WORKERS} \
        --packages-up-to franka_hardware franka_robot_state_broadcaster franka_description \
        --packages-skip libfranka \
        --event-handlers console_direct+ \
        --cmake-args -DCMAKE_BUILD_TYPE=Release

# =========================== stage: final ==================================
FROM frankaws AS final

ARG USERNAME=user
ENV FRANKA_WS=/opt/franka_ws \
    ARIS_WS=/aris_ws \
    ARIS_CONTROLLER_SRC=/src/controller

USER root
RUN mkdir -p ${ARIS_WS} /src/rtff /src/controller \
    && chown -R ${USERNAME}:${USERNAME} ${ARIS_WS} /src

COPY entrypoint.sh /entrypoint.sh
COPY test/ /opt/aris_test/
RUN chmod +x /entrypoint.sh && chmod -R a+rX /opt/aris_test \
    && echo "source /entrypoint_env.sh" >> /home/${USERNAME}/.bashrc \
    && printf '%s\n' \
        'source /opt/ros/jazzy/setup.bash' \
        '[ -f /opt/franka_ws/install/setup.bash ] && source /opt/franka_ws/install/setup.bash' \
        '[ -f /aris_ws/install/setup.bash ] && source /aris_ws/install/setup.bash' \
        > /entrypoint_env.sh

USER ${USERNAME}
WORKDIR ${ARIS_WS}
SHELL ["/bin/bash", "-c"]
ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]

# ============================ stage: sil ===================================
# The software-in-the-loop plant (`aris_sixarm.sil`, contract §6) needs
# pydrake; the ROS side needs the distro's own numpy.  Both in one process is
# the whole point of T8 -- the node publishes franka_msgs while Drake steps the
# contact -- so pydrake goes into a venv created WITH --system-site-packages:
# rclpy, franka_msgs and numpy 1.26 stay visible, and `pip --no-deps` keeps
# drake from dragging matplotlib/pydot/Mosek in and from upgrading numpy under
# the ROS packages.  Verified: `pydrake.multibody.plant` imports against numpy
# 1.26.4 and `import rclpy` still works in the same interpreter.
#
# The repo itself is NEVER pip-installed: `aris_sixarm` is bind-mounted
# read-only at /src/aris_sixarm and reached over PYTHONPATH, so the container
# always runs the working tree (contract §0: no ROS on the host, no host
# install of the container's deps).  `aris_sixarm.ik` loads a compiled
# `_franka_ik*.so` that lives OUTSIDE the repo; it is bind-mounted too and
# ARIS_FRANKA_IK_PATH points at it (host and image are both Ubuntu 24.04 /
# CPython 3.12, so the cp312 extension loads as built).
FROM final AS sil

ARG USERNAME=user
ARG DRAKE_VERSION=1.56.0
ENV SIL_VENV=/opt/sil-venv \
    ARIS_SIXARM_SRC=/src/aris_sixarm \
    ARIS_FRANKA_IK_PATH=/src/franka_analytical_ik/franka_analytical_ik

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv --system-site-packages ${SIL_VENV} \
    && ${SIL_VENV}/bin/pip install --no-cache-dir --no-deps \
        "drake==${DRAKE_VERSION}" \
    && ${SIL_VENV}/bin/python -c "import pydrake.multibody.plant, numpy; \
print('pydrake', __import__('pydrake').getDrakePath(), 'numpy', numpy.__version__)"

# Mount points for the two read-only source trees the SIL node needs.
RUN mkdir -p ${ARIS_SIXARM_SRC} /src/franka_analytical_ik \
    && chown -R ${USERNAME}:${USERNAME} /src

# Interactive shells and the entrypoint see the repo on PYTHONPATH; the SIL
# node itself is run as ${SIL_VENV}/bin/python -m aris_sixarm.sil.ros_node.
RUN printf '%s\n' \
        'export PYTHONPATH=/src/aris_sixarm${PYTHONPATH:+:$PYTHONPATH}' \
        'export ARIS_FRANKA_IK_PATH=/src/franka_analytical_ik/franka_analytical_ik' \
        'export SIL_VENV=/opt/sil-venv' \
        >> /entrypoint_env.sh

USER ${USERNAME}
WORKDIR ${ARIS_WS}
