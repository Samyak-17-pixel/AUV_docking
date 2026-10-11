#!/usr/bin/env bash
# Source ROS 2 + local interfaces, then run the mission controller. Extra arguments go to mission.py (e.g. --config my_mission.yaml).
set -eo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
CONTROL="$(cd "${ROOT}/.." && pwd)"

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"

set +u
source /opt/ros/humble/setup.bash
source "${CONTROL}/ws/install/setup.bash"
set -u

echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
CONFIG_ARGS=(--config "${ROOT}/mission.yaml")
for a in "$@"; do [ "$a" = "--config" ] && CONFIG_ARGS=(); done
exec python3 "${ROOT}/mission.py" "${CONFIG_ARGS[@]}" "$@"
