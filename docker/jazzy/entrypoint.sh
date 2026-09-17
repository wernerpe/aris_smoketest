#!/usr/bin/env bash
# Sources ROS 2 Jazzy, the pinned franka workspace, and (if built) the /aris_ws
# overlay, then execs whatever it was given.
set -e

source /opt/ros/jazzy/setup.bash
[ -f /opt/franka_ws/install/setup.bash ] && source /opt/franka_ws/install/setup.bash
[ -f /aris_ws/install/setup.bash ] && source /aris_ws/install/setup.bash

# Keep the container off the lab networks by construction: the compose file uses
# the default bridge, and DDS is confined to a private domain.
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-13}"
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-LOCALHOST}"

exec "$@"
