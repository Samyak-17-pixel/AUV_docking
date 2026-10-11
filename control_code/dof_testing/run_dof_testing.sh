#!/usr/bin/env bash
# Source ROS 2 + local interfaces, then run the per-DOF test harness.
#   ./run_dof_testing.sh --dof heave --mode step
# Run with no arguments (or -h/--help) to see how to use it.
set -eo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
CONTROL="$(cd "${ROOT}/.." && pwd)"

usage() {
cat <<'EOF'
dof_testing: test ONE degree of freedom of the Mako_01 at a time.

USAGE
  ./run_dof_testing.sh --dof <surge|heave|pitch|yaw|roll|sway> --mode <step|hold>

MODES
  step   open loop: a fixed push on that DOF only; prints PASS/FAIL on the DIRECTION of the response.
         Use it first, to find wrong signs / wiring.
  hold   closed loop: a PID moves that DOF to a setpoint; prints PASS/FAIL on the final error.
         Use it to check and tune the gains.

RECOMMENDED ORDER (stop at the first FAIL)
  ./run_dof_testing.sh --dof heave --mode step     # must dive
  ./run_dof_testing.sh --dof surge --mode step     # must move forward
  ./run_dof_testing.sh --dof pitch --mode step     # must pitch nose up (short pulse)
  ./run_dof_testing.sh --dof yaw   --mode step     # must turn to starboard (spins up to 1 m/s first)
  ./run_dof_testing.sh --dof roll  --mode step     # must roll starboard-down (short pulse)
  ./run_dof_testing.sh --dof sway  --mode step     # must drift to the right
  then the same six with --mode hold

BEFORE YOU RUN (the vehicle will move!)
  1. Simulator running and playing, vehicle at a safe depth (not at the surface, not near the dock).
  2. Stop the bridge's teleop node, it publishes zero commands on /Mako_01/actuator_cmd:
       ros2 topic info /Mako_01/actuator_cmd -v | grep "Node name"     # should NOT list mavsim_teleop
       docker exec <bridge-container> pkill -f teleop_node.py
  3. Only one controller at a time. Ctrl-C stops the test and sends zero commands.

Settings (step sizes, setpoints, gains, safety limits): dof_testing.yaml   Logs: outputs/logs/dof_testing/
More detail: ../../execution.md section 7.
EOF
}

if [[ $# -eq 0 || "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 0
fi

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
set +u
source /opt/ros/humble/setup.bash
source "${CONTROL}/ws/install/setup.bash"
set -u
echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
exec python3 "${ROOT}/dof_testing.py" --config "${ROOT}/dof_testing.yaml" "$@"
