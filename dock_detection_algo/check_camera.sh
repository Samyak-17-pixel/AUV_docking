#!/usr/bin/env bash
# Optional: check that the camera topic is alive before opening the GUI.
set -eo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
TOPIC="${TOPIC:-/dock_02/camera_02/image/compressed}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-${ROOT}/fastrtps_no_shm.xml}"

set +u
source /opt/ros/humble/setup.bash
set -u

echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
echo "FASTRTPS_DEFAULT_PROFILES_FILE=${FASTRTPS_DEFAULT_PROFILES_FILE}"
echo "Checking ${TOPIC} ..."
ros2 topic list | grep -F "${TOPIC}" || {
  echo "Topic not found. Available cameras:" >&2
  ros2 topic list | grep -i camera || true
  exit 1
}
echo "OK — topic exists. Measuring rate (needs UDP profile for Docker→host)..."
timeout 4 ros2 topic hz "${TOPIC}" || {
  echo "No data received. Is the sim camera streaming?" >&2
  exit 1
}
