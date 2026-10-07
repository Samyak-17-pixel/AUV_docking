#!/usr/bin/env bash
# Source ROS 2 + local interfaces, then run the dock standoff test controller.
# Run dock_detection_algo/run_live.sh in another terminal. Stop teleop first.
set -eo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
CONTROL="$(cd "${ROOT}/.." && pwd)"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
set +u
source /opt/ros/humble/setup.bash
source "${CONTROL}/ws/install/setup.bash"
set -u
echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
echo "Stop mavsim teleop before this node. It publishes zeros on actuator_cmd."
exec python3 "${ROOT}/dock_test.py" --config "${ROOT}/dock_test.yaml" "$@"
