#!/usr/bin/env bash
# Simulated Omniscan 450 SS side-scan sonars for the OFFLINE simulator (fake_vehicle). Publishes /Mako_01/sonar_01/ping (port), sonar_02/ping (starboard), altimeter/range, sonar/status
# and listens to /Mako_01/sonar/cmd (JSON: {"range_m": 40, "gain": 5}). Private ROS domain only:
#   export ROS_DOMAIN_ID=77 ROS_LOCALHOST_ONLY=1
#   python3 ../sim_offline/fake_vehicle.py --realistic --z 3 &
#   ./run_sidescan_sim.sh &
#   ./run_sim_viewer.sh --bottom-tab Sonar
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONTROL="$(cd "${ROOT}/.." && pwd)"
if [[ "$1" == "-h" || "$1" == "--help" ]]; then sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'; exit 0; fi
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-77}"
if [[ "${ROS_DOMAIN_ID}" == "42" && -z "${ALLOW_DOMAIN_42}" ]]; then
  echo "ROS_DOMAIN_ID=42 is the real mavsim bridge: refusing to publish FAKE sonar topics there (ALLOW_DOMAIN_42=1 overrides)." >&2
  exit 1
fi
set +u
source /opt/ros/humble/setup.bash
source "${CONTROL}/ws/install/setup.bash"
set -u
exec python3 "${ROOT}/sonar/sidescan_node.py" --config "${ROOT}/sim_viewer.yaml" "$@"
