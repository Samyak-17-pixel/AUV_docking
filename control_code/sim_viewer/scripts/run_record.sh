#!/usr/bin/env bash
# Record a run from the ROS topics to a CSV (read-only; it only subscribes). See record_run.py --help.
#   ROS_DOMAIN_ID=42 ./run_record.sh --label heave_step --frames 10     # the real mavsim
#   ROS_DOMAIN_ID=77 ROS_LOCALHOST_ONLY=1 ./run_record.sh --duration 60 # the offline fake vehicle
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONTROL="$(cd "${ROOT}/.." && pwd)"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
set +u
source /opt/ros/humble/setup.bash
source "${CONTROL}/ws/install/setup.bash"
set -u
exec python3 "${ROOT}/data/record_run.py" "$@"
