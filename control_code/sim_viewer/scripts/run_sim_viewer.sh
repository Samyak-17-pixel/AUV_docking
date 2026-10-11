#!/usr/bin/env bash
# Desktop sim viewer: 3D scene with the real vehicle and dock meshes, live plots, the nose camera image, status panel.
#   ./run_sim_viewer.sh --demo                  # in-process demo, no ROS and no other terminals needed
#   ./run_sim_viewer.sh                         # OUR offline sim: private ROS domain 77 (ROS_LOCALHOST_ONLY=1); use the Controls tab -> "Start offline stack" to get the
#                                               #   vehicle, the camera feed and the detector. (Changed 2026-10-11: it used to default to domain 42 = the real mavsim, so the
#                                               #   camera pane stayed blank and the stack button was disabled.)
#   ./run_sim_viewer.sh --real                  # the REAL mavsim bridge: domain 42, read-only look at its topics
#   ROS_DOMAIN_ID=78 ./run_sim_viewer.sh        # any domain you export is respected (the offline stack must use the same one)
# Options: --view follow|free|top|side|dock|onboard   --config FILE   --time-scale N (demo)
# Mouse: left drag orbit, middle drag / Shift+left pan, right drag / wheel zoom.
set -eo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONTROL="$(cd "${ROOT}/.." && pwd)"
if [[ "$1" == "-h" || "$1" == "--help" ]]; then sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0; fi
ARGS=()
REAL=0
for a in "$@"; do if [[ "$a" == "--real" ]]; then REAL=1; else ARGS+=("$a"); fi; done
if [[ "${REAL}" == "1" ]]; then
  export ROS_DOMAIN_ID=42
elif [[ -z "${ROS_DOMAIN_ID:-}" ]]; then
  export ROS_DOMAIN_ID=77
  export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-1}"
fi
echo "[run_sim_viewer] ROS_DOMAIN_ID=${ROS_DOMAIN_ID}$([[ ${ROS_DOMAIN_ID} == 42 ]] && echo '  (the REAL mavsim bridge)' || echo '  (private offline domain)')"
if [[ -z "${DISPLAY}" ]]; then echo "ERROR: DISPLAY is not set: the viewer needs a desktop session." >&2; exit 1; fi
set +u
source /opt/ros/humble/setup.bash
source "${CONTROL}/ws/install/setup.bash"
set -u
exec python3 "${ROOT}/app/viewer_app.py" --config "${ROOT}/sim_viewer.yaml" "${ARGS[@]}"
