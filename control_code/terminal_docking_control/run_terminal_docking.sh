#!/usr/bin/env bash
# Source ROS 2 + local interfaces, then run the terminal docking controller (alignment + entry into the funnel).
# Needs the dock detector (dock_detection_algo/run_live.sh) and odometry. Stop teleop first.
set -eo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
CONTROL="$(cd "${ROOT}/.." && pwd)"
if [[ "$1" == "-h" || "$1" == "--help" ]]; then sed -n '2,3p' "$0" | sed 's/^# \{0,1\}//'; echo "Options: --config FILE"; exit 0; fi
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
set +u
source /opt/ros/humble/setup.bash
source "${CONTROL}/ws/install/setup.bash"
set -u
echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
echo "Stop mavsim teleop before this node. It publishes zeros on actuator_cmd."
exec python3 "${ROOT}/terminal_docking.py" --config "${ROOT}/terminal_docking.yaml" "$@"
