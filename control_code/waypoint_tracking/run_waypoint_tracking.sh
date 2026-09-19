#!/usr/bin/env bash
# Source ROS 2 + local interfaces, then run the waypoint tracker.
set -eo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
CONTROL="$(cd "${ROOT}/.." && pwd)"

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"

set +u
source /opt/ros/humble/setup.bash
source "${CONTROL}/ws/install/setup.bash"
set -u

echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
exec python3 "${ROOT}/waypoint_tracking.py" --config "${ROOT}/waypoint_tracking.yaml" "$@"
