#!/usr/bin/env bash
# One-command live dock-light viewer + DockAlign publisher.
# Prerequisite: mavsim running, camera publishing, display available.
set -eo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
WS="$(cd "${ROOT}/../control_code/ws" && pwd)"

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
TOPIC="${TOPIC:-/Mako_01/camera_03/image/compressed}"

# Docker bridge ↔ host: disable Fast-DDS shared memory or images never arrive.
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_fastrtps_cpp}"
export FASTRTPS_DEFAULT_PROFILES_FILE="${FASTRTPS_DEFAULT_PROFILES_FILE:-${ROOT}/fastrtps_no_shm.xml}"

set +u
source /opt/ros/humble/setup.bash
if [[ ! -f "${WS}/install/setup.bash" ]]; then
  echo "Building interfaces (DockAlign) in ${WS} ..."
  (cd "${WS}" && colcon build --packages-select interfaces)
fi
source "${WS}/install/setup.bash"
set -u

ALIGN_TOPIC="${ALIGN_TOPIC:-}"
echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
echo "TOPIC=${TOPIC}"
echo "DISPLAY=${DISPLAY:-<unset>}"
echo "FASTRTPS_DEFAULT_PROFILES_FILE=${FASTRTPS_DEFAULT_PROFILES_FILE}"
if [[ -z "${DISPLAY:-}" ]]; then
  echo "WARNING: DISPLAY is unset — GUI windows will not appear." >&2
  echo "Run this on the desktop session (or export DISPLAY=:0)." >&2
fi
echo "Tips: echo DockAlign with: ./echo_align.sh"

cd "${ROOT}"
if [[ -n "${ALIGN_TOPIC}" ]]; then
  exec python3 "${ROOT}/live_dock_lights.py" --topic "${TOPIC}" --align-topic "${ALIGN_TOPIC}" "$@"
else
  exec python3 "${ROOT}/live_dock_lights.py" --topic "${TOPIC}" "$@"
fi
