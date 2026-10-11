#!/usr/bin/env bash
# First run on the REAL mavsim: the docs/project_notes.md section 10 checklist as one command. It stops at the FIRST problem and says what to look at.
#
#   ./scripts/run_real_sim_checklist.sh                 # stages 0-6 (everything up to station keeping), domain 42
#   ./scripts/run_real_sim_checklist.sh --from 3        # resume at stage 3
#   ./scripts/run_real_sim_checklist.sh --only 5        # one stage
#   ./scripts/run_real_sim_checklist.sh --dock          # also stage 7 (detector check + dock_test), only after 0-6 passed
#   ROS_DOMAIN_ID=77 ./scripts/run_real_sim_checklist.sh --offline-self-test   # point it at the offline fake vehicle to see the script itself work
#
# BEFORE YOU RUN (the vehicle MOVES; nothing here has ever run on the real sim):
#   - sim running and PLAYING, all six DOF active (the vessel file has surge+heave only), vehicle at about 3 m depth, away from the dock;
#   - mavsim_teleop stopped (stage 0 checks);
#   - you are at the keyboard: Ctrl-C stops the running test and every controller sends neutral on exit.
#
# STAGES   0 preflight (odometry alive, not frozen, no other actuator_cmd publisher)       4 yaw / roll step (fin sign + moment arm)
#          1 IMU vs odometry attitude convention (needs a tilt: skipped if --no-imu)       5 hold for each DOF (gains)
#          2 heave step (must dive)                                                        6 station_keeping: depth -> +pitch -> +surge -> +heading
#          3 surge step, pitch step                                                        7 (--dock) detector on camera_03, then dock_test
# Logs of every stage: outputs/logs/real_sim_checklist_<date>/ ; CSVs of the tests: outputs/logs/dof_testing/, outputs/logs/station_keeping/
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "${ROOT}/.." && pwd)"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
set +u
source /opt/ros/humble/setup.bash
source "${ROOT}/ws/install/setup.bash"
set -u

FROM=0; ONLY=""; DOCK=0; NO_IMU=0; SELF=0; KEEP_S=40; ASSUME_RESET=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from) FROM="$2"; shift 2;;
    --only) ONLY="$2"; shift 2;;
    --dock) DOCK=1; shift;;
    --no-imu) NO_IMU=1; shift;;
    --offline-self-test) SELF=1; shift;;
    --hold-seconds) KEEP_S="$2"; shift 2;;
    --no-prompt) ASSUME_RESET=1; shift;;       # do not wait for you to reset the vehicle between tests (only if something else resets it)
    -h|--help) sed -n 2,22p "$0"; exit 0;;
    *) echo "unknown option $1"; exit 64;;
  esac
done
if [[ "${ROS_DOMAIN_ID}" == "42" && "${SELF}" == "1" ]]; then echo "refusing --offline-self-test on domain 42 (the real bridge)"; exit 64; fi

LOGDIR="${REPO}/outputs/logs/real_sim_checklist_$(date +%Y%m%d_%H%M%S)"; mkdir -p "${LOGDIR}"
echo "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}   logs: ${LOGDIR}"

want() { [[ -n "${ONLY}" ]] && { [[ "${ONLY}" == "$1" ]]; return; }; [[ "$1" -ge "${FROM}" ]]; }
stop_here() { echo; echo "STOP at stage $1: $2"; echo "Fix that first, then: $0 --from $1"; exit 1; }

# The tests leave the vehicle moved, rolled and yawed (the roll/pitch/yaw steps have no righting moment: seen offline, a roll step left it at roll 60 deg,
# yaw 90 deg and the next test ABORTed on 'roll 60 deg'). So before every test the vehicle must be level, at depth, and still. On the real sim YOU reset it.
reset_vehicle() {
  if [[ "${SELF}" == "1" ]]; then
    ros2 topic pub --once /Mako_01/sim/cmd std_msgs/msg/String "{data: '{\"cmd\": \"reset\"}'}" > /dev/null 2>&1; sleep 2
  elif [[ "${ASSUME_RESET}" == "0" ]]; then
    read -r -p ">> Reset the vehicle in the sim UI (level, about 3 m depth, away from the dock, not moving), then press Enter: " _
  fi
  python3 "${ROOT}/tuning/real_sim_preflight.py" --seconds 4 > "${LOGDIR}/reset_check.log" 2>&1
  if [[ $? -ne 0 ]]; then cat "${LOGDIR}/reset_check.log"; echo "vehicle not ready (level / depth / not frozen)"; return 1; fi
  return 0
}

verdict_of() { grep -o "'verdict': '[A-Z]*'" "$1" | tail -1 | sed "s/.*: '//; s/'//"; }

run_dof() {   # dof mode -> logs and checks the verdict
  local dof="$1" mode="$2" log="${LOGDIR}/dof_${1}_${2}.log"
  reset_vehicle || return 1
  echo "--- dof_testing ${dof} ${mode}"
  "${ROOT}/dof_testing/run_dof_testing.sh" --dof "${dof}" --mode "${mode}" 2>&1 | tee "${log}" | grep -E "verdict|\[|ABORT" | tail -3
  local v; v="$(verdict_of "${log}")"
  echo "    verdict: ${v:-NONE}"
  case "${v}" in
    PASS) return 0;;
    INVALID) echo "    odometry froze during the test: no verdict. Run it again (docs/project_notes.md 2c item 3)."; return 1;;
    FAIL) echo "    see the reason line in ${log}; a FAIL with a LOCKED hint = that DOF is locked in the session."; return 1;;
    *) echo "    no verdict (aborted/interrupted): ${log}"; return 1;;
  esac
}

if want 0; then
  echo "=== stage 0: preflight"
  python3 "${ROOT}/tuning/real_sim_preflight.py" 2>&1 | tee "${LOGDIR}/stage0.log"
  [[ "${PIPESTATUS[0]}" == "0" ]] || stop_here 0 "the preflight is not READY (see above)"
fi

if want 1 && [[ "${NO_IMU}" == "0" ]]; then
  echo "=== stage 1: IMU vs odometry attitude (tilt the vehicle by a few degrees while this runs, e.g. a pitch step in the sim UI)"
  python3 "${REPO}/dock_detection_algo/check_imu_convention.py" --seconds 20 2>&1 | tee "${LOGDIR}/stage1.log"
  grep -q "FLIPPED" "${LOGDIR}/stage1.log" && echo "NOTE: set camera.imu_pitch_sign / imu_roll_sign to -1 in dock_detection.yaml before using the detector (not a blocker for the control stages)."
  grep -q "UNDECIDED" "${LOGDIR}/stage1.log" && echo "NOTE: not decided (no tilt). The control stages do not depend on it; repeat it before stage 7."
fi

if want 2; then echo "=== stage 2: heave step"; run_dof heave step || stop_here 2 "heave step"; fi
if want 3; then
  echo "=== stage 3: surge and pitch step"
  run_dof surge step || stop_here 3 "surge step"
  run_dof pitch step || stop_here 3 "pitch step"
fi
if want 4; then
  echo "=== stage 4: yaw and roll step (this confirms or breaks the fin assumptions)"
  run_dof yaw step || stop_here 4 "yaw step: wrong sign or moment arm, or the speed spin-up failed"
  run_dof roll step || stop_here 4 "roll step"
fi
if want 5; then
  echo "=== stage 5: holds (retune gains: dof_testing.yaml; copy good ones to station_keeping.yaml)"
  for d in heave pitch surge yaw roll; do run_dof "${d}" hold || stop_here 5 "${d} hold"; done
fi

if want 6; then
  echo "=== stage 6: station keeping, loops added one by one (${KEEP_S} s each; Ctrl-C to stop early)"
  TMPD="$(mktemp -d)"
  for stage in "depth" "depth pitch" "depth pitch surge" "depth pitch surge heading"; do
    python3 - "${ROOT}/station_keeping/station_keeping.yaml" "${TMPD}/sk.yaml" "${stage}" <<'PY'
import sys, yaml
src, dst, loops = sys.argv[1], sys.argv[2], sys.argv[3].split()
cfg = yaml.safe_load(open(src))
cfg["enable"] = {k: (k in loops) for k in cfg["enable"]}
yaml.safe_dump(cfg, open(dst, "w"))
PY
    reset_vehicle || stop_here 6 "vehicle not ready before station keeping (${stage})"
    echo "--- station_keeping with: ${stage}"
    log="${LOGDIR}/sk_${stage// /_}.log"
    timeout -s INT "${KEEP_S}" python3 "${ROOT}/station_keeping/station_keeping.py" --config "${TMPD}/sk.yaml" 2>&1 | tee "${log}" | tail -4
    grep -qiE "SAFETY TRIP|FROZEN|frozen" "${log}" && stop_here 6 "station keeping (${stage}): safety trip or frozen odometry, see ${log}"
  done
  rm -rf "${TMPD}"
fi

if [[ "${DOCK}" == "1" ]] && want 7; then
  echo "=== stage 7: detector on camera_03, then dock_test"
  echo "In ANOTHER terminal run:  ${REPO}/dock_detection_algo/run_live.sh   and check: cores = 4, valid, elevation plausible (docs/project_notes.md 7)."
  read -r -p "Detector shows 4 cores and valid? [y/N] " a
  [[ "${a}" == "y" ]] || stop_here 7 "detector not confirmed"
  "${ROOT}/dock_test/run_dock_test.sh" 2>&1 | tee "${LOGDIR}/dock_test.log"
fi

echo; echo "ALL REQUESTED STAGES PASSED. Logs: ${LOGDIR}"
echo "Record in docs/project_notes.md section 13: what PASSED/FAILED, the retuned gains, and which assumptions (section 8) the runs confirmed or broke."
