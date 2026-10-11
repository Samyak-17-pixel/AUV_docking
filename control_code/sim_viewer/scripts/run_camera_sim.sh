#!/usr/bin/env bash
# Synthetic dock camera for the OFFLINE simulator (fake_vehicle). Publishes camera_03 images + IMU from odometry.
# Run in a private ROS domain so it cannot clash with the real mavsim bridge:
#   export ROS_DOMAIN_ID=77 ROS_LOCALHOST_ONLY=1
#   python3 ../sim_offline/fake_vehicle.py --x 0 --y 0.8 --z 3 &
#   ./run_camera_sim.sh &
#   ../../dock_detection_algo/run_live.sh --no-gui &      # the REAL detector
#   ../dock_test/run_dock_test.sh                         # the controller
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONTROL="$(cd "${ROOT}/.." && pwd)"
if [[ "$1" == "-h" || "$1" == "--help" ]]; then sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0; fi
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
if [[ "${ROS_DOMAIN_ID}" == "42" && -z "${ALLOW_DOMAIN_42}" ]]; then
  echo "WARNING: ROS_DOMAIN_ID=42 is the real mavsim bridge. This node would publish a FAKE camera there." >&2
  echo "Use a private domain (export ROS_DOMAIN_ID=77 ROS_LOCALHOST_ONLY=1) or set ALLOW_DOMAIN_42=1 to override." >&2
  exit 1
fi
set +u
source /opt/ros/humble/setup.bash
source "${CONTROL}/ws/install/setup.bash"
set -u
exec python3 "${ROOT}/camera/camera_node.py" --config "${ROOT}/sim_viewer.yaml" "$@"
