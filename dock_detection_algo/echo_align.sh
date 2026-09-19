#!/usr/bin/env bash
# One-command: echo DockAlign guidance topic.
set -eo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
WS="$(cd "${ROOT}/../control_code/ws" && pwd)"

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-${ROOT}/fastrtps_no_shm.xml}"
ALIGN_TOPIC="${ALIGN_TOPIC:-/Mako_01/dock_align}"

set +u
source /opt/ros/humble/setup.bash
source "${WS}/install/setup.bash"
set -u

echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
echo "Echoing ${ALIGN_TOPIC}  (Ctrl-C to stop)"
exec ros2 topic echo "${ALIGN_TOPIC}" "$@"
