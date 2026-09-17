#!/usr/bin/env bash
# Build the Aris ROS 2 Jazzy dev-harness image.
#
# Shared lab machine: parallelism is capped (MAKEFLAGS=-j8, colcon
# --parallel-workers 2).  Nothing is installed on the host.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE="${ARIS_IMAGE:-aris/jazzy-dev}"
TAG="${ARIS_TAG:-d812aab}"
USER_UID="${ARIS_USER_UID:-$(id -u)}"
USER_GID="${ARIS_USER_GID:-$(id -g)}"
COLCON_WORKERS="${ARIS_COLCON_WORKERS:-2}"

echo "==> building ${IMAGE}:${TAG}  (uid=${USER_UID} gid=${USER_GID}, colcon -j${COLCON_WORKERS}, MAKEFLAGS=-j8)"
START=$(date +%s)

DOCKER_BUILDKIT=1 docker build \
    --progress=plain \
    --build-arg USER_UID="${USER_UID}" \
    --build-arg USER_GID="${USER_GID}" \
    --build-arg COLCON_WORKERS="${COLCON_WORKERS}" \
    -t "${IMAGE}:${TAG}" \
    -t "${IMAGE}:latest" \
    -f "${HERE}/Dockerfile" \
    "${HERE}"

END=$(date +%s)
echo "==> built ${IMAGE}:${TAG} in $(( END - START ))s"
SIZE_BYTES=$(docker image inspect "${IMAGE}:${TAG}" --format '{{.Size}}')
echo "==> image size: ${SIZE_BYTES} bytes ($(( SIZE_BYTES / 1048576 )) MiB)" 
