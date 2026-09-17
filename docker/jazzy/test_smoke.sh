#!/usr/bin/env bash
# Host-side wrapper: runs the T1..T8 smoke tests INSIDE the container and
# prints a PASS/FAIL line per test.
#
# SAFETY: default bridge network, no privileged, no cap_add, no /dev, every
# host bind mount read-only except ./out.  Nothing here can reach a lab robot.
#
#   ./test_smoke.sh
#   ARIS_CONTROLLER_SRC=/path/to/other/cartesian_impedance_controller ./test_smoke.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE="${ARIS_IMAGE:-aris/jazzy-dev}"
TAG="${ARIS_TAG:-d812aab}"
RTFF_SRC="${ARIS_RTFF_SRC:-/home/franka/aris_project/worktrees/aris2-rtff}"
CONTROLLER_SRC="${ARIS_CONTROLLER_SRC:-${RTFF_SRC}/aris_kindt_dwatkins_ztouch_control/ros2_ws/src/cartesian_impedance_controller}"
# T8 (SIL closed loop): the planner repo and the analytic-IK extension it
# loads, both read-only.  Nothing here is installed into the image.
SIXARM_SRC="${ARIS_SIXARM_SRC:-$(cd "${HERE}/../.." && pwd)}"
IK_SRC="${ARIS_IK_SRC:-/home/franka/aris_project/franka_analytical_ik}"
OUT_DIR="${ARIS_OUT_DIR:-${HERE}/out}"

mkdir -p "${OUT_DIR}"

echo "image           : ${IMAGE}:${TAG}"
echo "rtff source     : ${RTFF_SRC}      -> /src/rtff       (ro)"
echo "controller src  : ${CONTROLLER_SRC} -> /src/controller (ro)"
echo "aris_sixarm     : ${SIXARM_SRC}      -> /src/aris_sixarm (ro)"
echo "franka ik .so   : ${IK_SRC}          -> /src/franka_analytical_ik (ro)"
echo "output dir      : ${OUT_DIR}        -> /out"
echo

# --rm, default bridge net, no privileged / cap-add / device / host-net.
exec docker run --rm -i \
    --init \
    --name "aris-jazzy-smoke-$$" \
    -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-13}" \
    -e ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
    -e ARIS_COLCON_WORKERS="${ARIS_COLCON_WORKERS:-2}" \
    -e ARIS_T6_DURATION="${ARIS_T6_DURATION:-20}" \
    -e ARIS_T7_DURATION="${ARIS_T7_DURATION:-35}" \
    -e ARIS_T7_CLI_DURATION="${ARIS_T7_CLI_DURATION:-20}" \
    -e ARIS_T7B_DURATION="${ARIS_T7B_DURATION:-35}" \
    -e ARIS_T8_DURATION="${ARIS_T8_DURATION:-60}" \
    -e ARIS_T8_SKIP="${ARIS_T8_SKIP:-0}" \
    -e ARIS_T8_RT_FACTOR="${ARIS_T8_RT_FACTOR:-1.0}" \
    -e ARIS_T8_LATCH="${ARIS_T8_LATCH:-1}" \
    -e MAKEFLAGS=-j8 \
    -v "${RTFF_SRC}:/src/rtff:ro" \
    -v "${CONTROLLER_SRC}:/src/controller:ro" \
    -v "${SIXARM_SRC}:/src/aris_sixarm:ro" \
    -v "${IK_SRC}:/src/franka_analytical_ik:ro" \
    -v "${HERE}/test:/opt/aris_test:ro" \
    -v "${OUT_DIR}:/out" \
    "${IMAGE}:${TAG}" \
    bash /opt/aris_test/run_tests.sh
