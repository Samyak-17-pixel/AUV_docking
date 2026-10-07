# Execution guide — AUV docking (Mako_01)

Step-by-step instructions for every task. Each task says **which terminal**, **what to type**, **what you should see**, and
**what to do if you do not**. Last updated 2026-10-08.

Paths are written from the home directory. `AUV` below means `~/Research/MAVSIM/AUV_docking`.

---

## 0. What do you want to do? (quick map)

| I want to... | Go to |
|---|---|
| Set up a terminal so `ros2` commands work | [1](#1-set-up-every-terminal-do-this-first) |
| Install things (first time only) | [2](#2-one-time-setup) |
| Start the simulator and check it works | [3](#3-start-the-simulator-and-check-it) |
| Hold a depth | [5](#5-task-depth-hold) |
| Follow waypoints | [6](#6-task-waypoint-tracking) |
| Test each degree of freedom / check actuator signs | [7](#7-task-per-dof-tests-actuator-allocation) |
| Hold position (hover) | [8](#8-task-station-keeping-hover) |
| Detect the dock lights, including a pitched view | [9](#9-task-dock-light-detection) |
| Hold a standoff on the dock (match depth, square up, do not enter) | [15](#15-task-dock-standoff-controller) |
| Run everything with no simulator | [10](#10-task-offline-mode-no-simulator) |
| Run the unit tests | [11](#11-task-unit-tests) |
| Do a full docking run | [12](#12-full-docking-run-order) |
| Stop the vehicle / something went wrong | [13](#13-stop-everything) and [14](#14-troubleshooting) |

**Rules that apply to every task**
- Commands to the vehicle are thruster **RPM** (`th_XX`) and fin **degrees** (`cs_XX`), 0 = stop. Not PWM.
- Both heave thrusters point **UP**: positive RPM = up, so **diving = negative RPM** (already handled, `heave_sign: -1`).
- Depth is NED metres, **positive = deeper**. The vehicle starts at (x 0, y 0, depth 3 m); the dock is at (10, 0, 3).
- Run **only one controller at a time** (depth, waypoint, DOF test, station keeping, dock standoff). They all publish on the same topic. The dock detector does **not** publish actuator commands; it can run next to a controller.
- Edit the **YAML** next to a script to change its behaviour. Do not edit the Python.

---

## 1. Set up every terminal (do this first)

Every new terminal that runs `ros2` or any script in this guide needs these three lines:

```bash
export ROS_DOMAIN_ID=42
source /opt/ros/humble/setup.bash
source ~/Research/MAVSIM/AUV_docking/control_code/ws/install/setup.bash
```

**Why:** without the last line `ros2 topic echo /Mako_01/actuator_cmd` fails with
`The message type 'interfaces/msg/Actuator' is invalid` (the `interfaces` package is not on the path).

**Make it permanent** (so new terminals do it for you): add those three lines to the end of `~/.bashrc`, then open a new terminal.

**Check it worked:**
```bash
ros2 interface show interfaces/msg/Actuator
```
You should see `float64[] actuator_values`, `string[] actuator_names`, `float64[] covariance`.
If you get `Unknown package 'interfaces'`, the third line is missing or the workspace is not built (see 2.1).

> The `run_*.sh` scripts source ROS and the workspace for you. This section is for the plain `ros2 ...` commands you type by hand.

---

## 2. One-time setup

### 2.1 Build the ROS messages (only if `control_code/ws/install` does not exist, or the `.msg` files changed)

`DockAlign.msg` gained the pitched-view fields (`lateral_px`, `elevation_rad`, `search_*`, and the rest listed in section 9.6). If the detector or the standoff controller fails with a missing field, rebuild:
```bash
cd ~/Research/MAVSIM/AUV_docking/control_code/ws
source /opt/ros/humble/setup.bash
colcon build --packages-select interfaces
source install/setup.bash
```
Expected: `Summary: 1 package finished`. Afterwards `ros2 interface show interfaces/msg/Actuator` works (section 1).

### 2.2 Python packages for dock detection
```bash
cd ~/Research/MAVSIM/AUV_docking/dock_detection_algo
pip install -r requirements.txt        # opencv-python, numpy, pyyaml
```
Use `opencv-python` (not `-headless`): the detector opens GUI windows. Check: `python3 -c "import cv2; print(cv2.__version__)"`.

---

## 3. Start the simulator and check it

### 3.1 Start the bridge — Terminal 1
```bash
cd ~/Research/MAVSIM/mavsim-controller
./start.sh                      # web mode, no arguments needed
```
Then open **http://localhost:8888**, and start/play a session with the **Mako_01** vessel (`~/Research/MAVSIM/Mako_01.mavsim`).
Other pages the bridge starts: visualizer http://localhost:8899, keyboard teleop http://localhost:8900.

Options: `./start.sh <controller-code> --vessel-name Mako_01`, or `--frontend-url http://<host>:5173` if the MAVSim frontend
is on another machine (camera images need the frontend to be reachable).

### 3.2 Check the topics — Terminal 2 (after section 1)
```bash
ros2 topic list | grep Mako_01
```
Expected (at least): `/Mako_01/actuator_cmd`, `/Mako_01/odometry_sim`, `/Mako_01/camera_03/image/compressed`,
`/Mako_01/imu_01/data`, `/Mako_01/dvl_03/data`, `/Mako_01/vessel_state`.
Nothing listed: the session is not playing, or `ROS_DOMAIN_ID` is not 42, or the terminal is not set up (section 1).

### 3.3 Check the vehicle state
```bash
ros2 topic echo /Mako_01/odometry_sim --once
```
Expected: `frame_id: NED`, `child_frame_id: BODY`, position about `x: 0, y: 0, z: 3`, orientation `w: 1`.
(Odometry arrives at only ~4 Hz. That is normal.)

### 3.4 STOP teleop before running any controller (important)
The bridge starts a keyboard-teleop node that publishes **all-zero commands at 20 Hz** on `/Mako_01/actuator_cmd`.
Our controllers publish on the same topic, so zeros from teleop would be mixed in with our commands.

1. See who publishes:
   ```bash
   ros2 topic info /Mako_01/actuator_cmd -v | grep "Node name"
   ```
   If you see `mavsim_teleop`, stop it (steps 2-3).
2. Find the bridge container name (it changes each run):
   ```bash
   docker ps --format '{{.Names}}'          # e.g. mavsim-bridge-312798
   ```
3. Stop the teleop process inside it:
   ```bash
   docker exec <container-name> pkill -f teleop_node.py
   ```
   (If `pkill` is missing there: `docker exec <container-name> ps -eo pid,cmd | grep teleop`, then `docker exec <container-name> kill <PID>`.
   The teleop process is `/app/teleop_node.py --http-port 8900 --ws-port 8901`.)
4. Re-run step 1. Only `mavsim_bridge` should remain as a subscriber; no `mavsim_teleop` publisher.
   Restarting `./start.sh` brings teleop back; repeat steps 2-3.

---

## 4. How to read results (applies to all tasks)

- **Controller log lines** print every ~2 s in the terminal that runs the controller.
- **Ctrl-C** stops any controller and makes it publish zero commands on the way out.
- **CSV logs** are saved in your home directory: `~/dof_testing_logs/` and `~/station_keeping_logs/`. Open them in a spreadsheet
  or plot them to see the response.
- Watch the commands being sent: `ros2 topic echo /Mako_01/actuator_cmd`.
- Watch the vehicle: `ros2 topic echo /Mako_01/odometry_sim --once` (repeat) or the visualizer at http://localhost:8899.

---

## 5. Task: depth hold

Holds an absolute depth with a PID on the two heave thrusters, then stops itself.

1. Do sections 1 and 3 (simulator running, teleop stopped).
2. Optional: set the target depth. Edit `AUV/control_code/depth_control/depth_control.yaml`:
   `depth.setpoint_m: 5.0` (metres, positive = deeper). Gains are under `pid:`.
3. Run — Terminal 2:
   ```bash
   cd ~/Research/MAVSIM/AUV_docking/control_code/depth_control
   ./run_depth_control.sh
   ```
4. Expected log (every 2 s): `z=3.02 m  z_d=5.00  err=1.98  heave=-600 RPM  settle=0.0/30 s`.
   Heave RPM is **negative** while diving. `err` should shrink towards 0.
5. When `|err| < 0.2 m` for 30 s it prints `Settled ... Zeroing actuators and exiting.` and ends. Ctrl-C ends it sooner.
6. Use another file: `./run_depth_control.sh --config /path/to/other.yaml`.

If the vehicle goes **up** instead of down, or barely moves: see troubleshooting ("rises when it should dive" / "cannot reach depth").
`rpm_cap: 1000` in this YAML is marginal for this vehicle (it floats up with 7.25 N net buoyancy); 1400 or more is safer.
This controller has not yet been run on the real sim since the PWM -> RPM change: watch it the first time.

---

## 6. Task: waypoint tracking

Steers through a list of NED waypoints (heading by the fins, speed by the axial thruster, depth by the heave thrusters).

1. Do sections 1 and 3.
2. Edit `AUV/control_code/waypoint_tracking/waypoint_tracking.yaml`:
   - `waypoints:` list of `{x, y, z}` (North, East, Down in metres). Default is a 5 m x 3 m rectangle at depth 3 m.
     The dock is at (10, 0, 3): keep the last waypoint short of it.
   - `mission.acceptance_radius_m` and `depth_acceptance_m` decide when a waypoint counts as reached.
3. Run — Terminal 2:
   ```bash
   cd ~/Research/MAVSIM/AUV_docking/control_code/waypoint_tracking
   ./run_waypoint_tracking.sh
   ```
4. Expected log: `WP0/3 pos=(1.2,0.0,3.0) r_xy=3.8 hd_err=2.1° surge=700 RPM heave=-50 RPM fin=1.4°`, then
   `Reached WP0 ...` for each waypoint, and finally `Mission complete — actuators at neutral.`
5. Heading control uses the fins, which only work with forward speed. The vehicle only turns once it is moving.

---

## 7. Task: per-DOF tests (actuator allocation)

Checks, one degree of freedom at a time, that each command moves the vehicle the right way, and lets you tune the PID of each.
Do this **before** relying on station keeping or the controllers above.

### 7.1 What a test does
- `--mode step` (open loop): applies a fixed push on only that DOF and prints PASS/FAIL on the **direction** of the response.
  Use it to find wrong signs / wiring.
- `--mode hold` (closed loop): a PID moves that DOF to a setpoint and prints PASS/FAIL on the final error. Use it to tune gains.
- Yaw, roll and sway need water flowing over the fins, so the test first drives the axial thruster up to about 1 m/s.
  Sway has no actuator: it is produced by yawing while moving.
- Settings (step size, setpoints, gains, tolerances, safety limits): `AUV/control_code/dof_testing/dof_testing.yaml`.

> The script **needs two arguments**, `--dof` and `--mode`. Running `./run_dof_testing.sh` with none (or with `-h`) just prints a
> help text (usage, the run order, the pre-flight checklist) and does nothing else. Giving only one of them shows
> `error: the following arguments are required: --mode` (or `--dof`): add the missing one.

### 7.2 Run order — do these in order and stop at the first FAIL
Terminal 2, after sections 1 and 3 (teleop stopped, vehicle at a safe depth, not near the surface or the dock):
```bash
cd ~/Research/MAVSIM/AUV_docking/control_code/dof_testing
./run_dof_testing.sh --dof heave --mode step      # 1. must dive (depth increases)
./run_dof_testing.sh --dof surge --mode step      # 2. must move forward
./run_dof_testing.sh --dof pitch --mode step      # 3. must pitch nose up (short pulse)
./run_dof_testing.sh --dof yaw   --mode step      # 4. must turn to starboard (spins up to 1 m/s first)
./run_dof_testing.sh --dof roll  --mode step      # 5. must roll starboard-down (short pulse)
./run_dof_testing.sh --dof sway  --mode step      # 6. must drift to the right
```
Then the same six with `--mode hold` to check and tune the PID gains.

### 7.3 What you see
- Every ~2 s: `[run] t=... z=... pitch=... yaw=... u=... | X=... Z=... K=... M=... N=...` (the commanded forces/moments).
- At the end:
  ```
  === RESULT ===
  {'verdict': 'PASS', 'mode': 'step', 'dof': 'heave', 'response_rate': 0.42, 'required': 0.03, ...}
  log: /home/vallabh/dof_testing_logs/dof_heave_step.csv
  ```
- `verdict` is `PASS`, `FAIL`, or `ABORT` (a safety limit tripped, with a reason; the vehicle is sent to neutral).

### 7.4 What a FAIL means
| Test | A FAIL points to |
|---|---|
| heave step | wrong heave sign, thrust-unit assumption (`rpm_to_rps`), or `rpm_cap` too low to beat the 7.25 N buoyancy |
| surge step | axial thruster sign / too weak |
| pitch step | heave thrusters' differential sign or moment arm |
| yaw / roll step | **fin sign or fin moment arm** (these are unverified assumptions in `common/mako_geometry.yaml`) |
| sway step | yaw sign, or the vehicle did not reach flow speed |
| `ABORT: speed spin-up timeout` | the vehicle could not reach `fin_dofs.cruise_speed_mps` (default 1.0 m/s) in 20 s |

Fix wrong signs in `AUV/control_code/common/mako_geometry.yaml`, not in the controller, then repeat the step test.
Pitch and roll steps are short pulses on purpose: this vehicle has no righting moment, so a long push would spin it up.

### 7.5 Tuning the PID of one DOF (`--mode hold`)
1. Run `./run_dof_testing.sh --dof heave --mode hold`.
2. Open the CSV it prints. Look at overshoot and settling time.
3. Edit `gains:` for that DOF in `dof_testing.yaml` (`kp` stiffer, `kd` more damping, `ki` removes the steady offset).
4. Run again. Change one number at a time.
5. When it passes comfortably, copy the gains for the DOFs you use into `station_keeping.yaml` (section 8).

---

## 8. Task: station keeping (hover)

Captures the current pose and holds depth, pitch, surge position and heading. Sideways drift is **not** held (no sideways thruster),
and heading is only held while water flows over the fins (above ~0.3 m/s).

1. Do sections 1 and 3 and pass at least the heave and pitch tests in section 7.
2. Edit `AUV/control_code/station_keeping/station_keeping.yaml` if needed. Commissioning order: first set the `enable:` block to
   depth only, then add pitch, then surge, then heading.
3. Put the vehicle where you want it to stay (use depth hold, section 5, or just start where it is).
4. Run — Terminal 2:
   ```bash
   cd ~/Research/MAVSIM/AUV_docking/control_code/station_keeping
   ./run_station_keeping.sh
   ```
5. For the first `capture.average_s` seconds (default 2) it measures the pose, then starts holding it.
6. Expected log every 2 s:
   `depth_err=+0.001 m  pitch_err=+0.00 deg  ahead_err=+0.000 m  lateral_drift=+0.00 m  hdg_err=+0.0 deg  u=0.00 m/s`.
   The line `[heading NOT controllable: no flow over fins]` is normal when the vehicle is still.
7. Stop with Ctrl-C (sends zero commands, saves `~/station_keeping_logs/station_keeping_<time>.csv`).
8. It stops by itself with `SAFETY TRIP: ...` if the depth, pitch or roll limit is exceeded, if it drifts further than
   `safety.max_drift_m` (5 m) from the hold point, or if odometry stops arriving.

---

## 9. Task: dock light detection

Finds the four dock lights in a camera image and publishes `/Mako_01/dock_align` (where the dock is relative to the image center).
All settings: `AUV/dock_detection_algo/dock_detection.yaml` (every parameter explained there).
You need a desktop session (the viewer opens windows). `echo $DISPLAY` must not be empty.

### 9.1 Check the camera streams — Terminal 3 (optional)
```bash
cd ~/Research/MAVSIM/AUV_docking/dock_detection_algo
TOPIC=/Mako_01/camera_03/image/compressed ./check_camera.sh
```
Expected: `OK — topic exists` and an `average rate: ...` line. `camera_03`, `camera_04` and `camera_05` all exist on the live sim.
Pick the camera that actually sees the dock and put it in `dock_detection.yaml` under `camera.topic`
(or pass it with `TOPIC=...` below). `check_camera.sh` defaults to `/dock_02/camera_02/...` if you do not set `TOPIC`.

### 9.2 Run the detector — Terminal 3
```bash
cd ~/Research/MAVSIM/AUV_docking/dock_detection_algo
./run_live.sh
```
Two windows open: **Dock camera** (detected lights T/B/L/R, center, errors, `cores=N`) and **Bloom mask** (the mask and the tuning sliders).
Goal: `cores=4` with T/B/L/R on the right lights. Keys: **p** prints the slider values as YAML, **q** or **Esc** quits.

Options:
```bash
TOPIC=/Mako_01/camera_04/image/compressed ./run_live.sh      # another camera
CONFIG=/path/to/my_tuning.yaml ./run_live.sh                 # another tuning file
ALIGN_TOPIC=/Mako_01/dock_align ./run_live.sh                # another output topic
```

### 9.3 Read the output — Terminal 4
```bash
cd ~/Research/MAVSIM/AUV_docking/dock_detection_algo
./echo_align.sh              # stream /Mako_01/dock_align (Ctrl-C to stop)
./echo_align.sh --once       # one message
```
- `valid: true` needs all 4 lights labelled, with both side lights between top and bottom.
- `error_x_px > 0`: dock center is to the **right** of the image center.
- `error_y_px > 0`: dock center is **below** the image center. That is the pitch cue (keep it in frame), not the dive cue.
- `aligned: true` and `spread_px` near 0: the camera is **level** and the dock is seen square-on. A pitched camera turns the light circle into an ellipse, so `spread_px` grows even when the heading is already right. Do not treat that as a failed heading.
- `confidence` is 0 to 1.

The detector also subscribes to `/Mako_01/imu_01/data`. It removes roll before labelling, and it uses pitch to compute `elevation_rad`. The two windows show the roll-compensated image, so the T/B/L/R markers sit on the lights. The new fields are listed in section 9.6.

### 9.4 Tune the detector
1. Run `./run_live.sh` with the dock in view.
2. Move the sliders in **Bloom mask** one at a time until `cores=4`.
3. Press **p** in the terminal; copy the printed values into `dock_detection.yaml`.
   Parameters without sliders (`tight_floor`, `dog_sigma_*`, ...) are edited directly in the YAML.
4. Restart `./run_live.sh` and confirm the same result.

| Symptom | Try |
|---|---|
| `cores` < 4, lights missing | lower `mask.v_thresh`; lower `peaks.core_pct`; lower `peaks.peak_sep` |
| `cores` < 4, lights merged | raise `peaks.core_pct`; lower `peaks.peak_sep`; lower `mask.close_k` |
| `cores` > 4, spurious blobs | raise `mask.v_thresh`; raise `peaks.response_min`; raise `peaks.peak_sep` |
| Centers jitter | raise `peaks.dog_sigma_small`; raise `peaks.refine_half_window` |
| Never `aligned` while the camera is level and square | raise `alignment.spread_align_frac` or `spread_align_min_px` |
| Never `aligned` while pitched up or down at the dock | Expected. `spread_px` is the level, square-on check. See section 9.6 |

### 9.5 Optional: via ROS launch
```bash
cd ~/Research/MAVSIM/AUV_docking/dock_detection_algo
ros2 launch dock_detection_algo_launch.py topic:=/Mako_01/camera_03/image/compressed config:=$PWD/dock_detection.yaml
```

### 9.6 Fields added for a pitched view and a partial detection

The four lights sit on a 1 m circle. Top and bottom are a vertical diameter. The two side lights are both above the center, at 45° from the top. They are the same color, so a missing light is not identified by color.

| Field | What it means | What a controller should do |
|---|---|---|
| `center_exact` | true when the side-light midpoint lies on the top–bottom line, so `center` is the cross-ratio dock center | If false, `center` is only the top–bottom midpoint |
| `lateral_px` | side midpoint to the **right** of the top–bottom line, in pixels. Near 0 when the heading is perpendicular to the dock, even if the vehicle is shifted sideways | Yaw. A pure sideways shift with a perpendicular heading shows up in `error_x_px` instead |
| `elevation_rad` | dock center **below the horizon**, radians. Positive = dock is deeper | Heave down. This is the depth command |
| `elevation_valid` | an IMU pitch sample has arrived | If false, do not heave from elevation |
| `obliqueness` | how foreshortened the diameter is. Zero when the camera is level, any depth | A check only. Not a heave or pitch command |
| `search_yaw_norm` | +1 = yaw toward image right. Published only while `valid` is false | Yaw. Needs forward speed so the fins work |
| `search_pitch_norm` | +1 = pitch down. Published only while `valid` is false | Pitch with the heave thrusters |
| `search_surge_norm` | about +0.35 when a yaw needs fin flow; −1 when the dock is too close to fit in the frame | Axial thruster. There is no sway thruster |

While fewer than four lights are trusted, `valid` is false and the detector does not publish a depth command. It first retries peak finding at half `peak_sep` (merged bloom). Then it yaws or pitches toward an edge light, nods about ±20° if the lights are inside the frame, or asks for reverse thrust if the 2 m dock cannot fit in the camera (inside about 2.3 m).

### 9.7 Check the geometry without the simulator

No ROS:
```bash
cd ~/Research/MAVSIM/AUV_docking/dock_detection_algo
python3 test_pitched_geometry.py
```
Expected: `10 tests passed`. These project the real 1 m light circle through a pinhole camera (level, deep, pitched, offset, yawed, rolled) and check the search commands.

---

## 10. Task: offline mode (no simulator)

Runs the controllers against a simple fake vehicle. It catches sign and gain-structure mistakes; it is **not** the real sim
(drag, thrust units and fin behaviour are assumptions). It uses its own ROS domain so it cannot touch the real bridge.

1. Terminal A — fake vehicle:
   ```bash
   cd ~/Research/MAVSIM/AUV_docking/control_code
   export ROS_DOMAIN_ID=77 ROS_LOCALHOST_ONLY=1
   source /opt/ros/humble/setup.bash && source ws/install/setup.bash
   python3 sim_offline/fake_vehicle.py --z 3
   ```
   Options: `--x --y --z --roll --pitch --yaw` (start pose), `--rate`, `--cmd-timeout`. Expected: `fake_vehicle up @ 100 Hz`.
2. Terminal B — same three environment lines as Terminal A (domain **77**, `ROS_LOCALHOST_ONLY=1`), then any task from sections 5-8, for example:
   ```bash
   cd ~/Research/MAVSIM/AUV_docking/control_code/station_keeping && ./run_station_keeping.sh
   ```
   (The scripts keep a `ROS_DOMAIN_ID` that is already set.)
3. Stop: Ctrl-C in Terminal B, then Ctrl-C in Terminal A.
4. **Always stop the fake vehicle afterwards.** A forgotten one publishes a second odometry stream and gives false results.
   Check with `ps -eo pid,cmd | grep fake_vehicle`; stop it with `kill <PID>`.

---

## 11. Task: unit tests

No ROS and no simulator needed:
```bash
cd ~/Research/MAVSIM/AUV_docking/control_code
python3 -m pytest tests -q
```
Expected: `37 passed` in about 3 s. They check allocation signs (heave thrusters point up), the PID, all DOF tests against the
offline vehicle, a wrong-sign actuator being caught, and station keeping (still water, vertical push, current, low `rpm_cap`,
no-flow heading, safety trips).

The dock work has its own scripts, also with no ROS:
```bash
cd ~/Research/MAVSIM/AUV_docking/dock_detection_algo && python3 test_pitched_geometry.py
cd ~/Research/MAVSIM/AUV_docking/control_code/dock_test && python3 test_dock_test_core.py
```
Expected: `10 tests passed`, then `8 tests passed`.

---

## 12. Full docking run (order)

1. Terminal 1: `./start.sh` in `mavsim-controller`, start the Mako_01 session (section 3.1).
2. Any terminal: stop teleop (3.4).
3. Terminal 2: pass the DOF tests (section 7).
4. Terminal 3: `./run_live.sh` (detector, section 9.2). Leave it running.
5. Terminal 4: `./echo_align.sh` and confirm messages are arriving. `valid: true` means four lights. `valid: false` is the search (section 9.6).
6. Optional: depth hold (5) or waypoint tracking (6) to bring the vehicle in front of the dock. Stop that controller before the next step. Only one program may publish `/Mako_01/actuator_cmd`.
7. Terminal 2: the standoff controller (section 15). It matches depth, squares the heading, keeps the dock in frame, then holds. It does **not** drive into the funnel.

---

## 13. Stop everything

- A controller: **Ctrl-C** in its terminal (it publishes zeros on exit).
- The detector: press **q** in a window, or Ctrl-C in its terminal.
- Force the actuators to zero from any terminal (needs section 1; press Ctrl-C to stop it):
  ```bash
  ros2 topic pub -r 10 /Mako_01/actuator_cmd interfaces/msg/Actuator \
    "{actuator_names: [th_01, th_02, th_03, cs_04, cs_06, cs_07, cs_08], actuator_values: [0, 0, 0, 0, 0, 0, 0]}"
  ```
  The bridge also zeroes commands by itself if none arrive for 1 s.
- Pause or stop the simulation from the web UI (http://localhost:8888).

---

## 14. Troubleshooting

| Problem | Fix |
|---|---|
| `The message type 'interfaces/msg/Actuator' is invalid` | Section 1: source `control_code/ws/install/setup.bash` in that terminal. If it still fails: `ros2 pkg prefix interfaces`; if empty, build it (2.1). |
| `ros2 topic list` shows no `/Mako_01/...` | Session not playing, `ROS_DOMAIN_ID` is not 42 (or you are in an offline terminal on 77), or the bridge is not running. |
| Vehicle does not move / moves erratically | `mavsim_teleop` is still publishing zeros: do 3.4. Also check the session is playing. |
| Vehicle rises when it should dive | Dive = NEGATIVE RPM on both heave thrusters. Run `--dof heave --mode step`. If it FAILs, fix the heave sign / `rpm_to_rps` in `common/mako_geometry.yaml`. |
| Depth hold cannot reach depth | Net buoyancy is +7.25 N. `rpm_cap: 1000` is marginal; use 1400+ (station keeping and DOF tests use 1800). |
| DOF yaw/roll step FAILs | Fin sign or fin moment arm (unverified) in `common/mako_geometry.yaml`. |
| DOF test `ABORT speed spin-up timeout` | Vehicle did not reach 1 m/s: lower `fin_dofs.cruise_speed_mps` in `dof_testing.yaml` or raise the surge limits. |
| Station keeping `SAFETY TRIP ... drifted` | Sideways drift cannot be corrected: raise `safety.max_drift_m`, or restart. |
| Station keeping `SAFETY TRIP ... too shallow` at start | A forgotten fake vehicle or a second odometry source is running, or depth < `safety.min_depth_m`. `ps -eo pid,cmd \| grep fake_vehicle`. |
| Station keeping heading drifts | Expected with no flow over the fins (below ~0.3 m/s). |
| Controller exits immediately | The depth or waypoints are already satisfied: change the setpoint in its YAML. |
| Detector: topic listed but no image | Fast-DDS shared memory: the scripts set `FASTRTPS_DEFAULT_PROFILES_FILE=dock_detection_algo/fastrtps_no_shm.xml`. Set the same in any terminal where you run `ros2 topic hz/echo` on a camera. |
| Detector: `interfaces.msg.DockAlign not found` | Section 1 / 2.1, or use `./run_live.sh` (it sources the workspace). |
| Detector: no GUI windows | `echo $DISPLAY` must be set (e.g. `export DISPLAY=:0`); install `opencv-python`, not the headless build. |
| `ModuleNotFoundError: yaml` | `pip install pyyaml`. |
| Bridge has no camera images | The MAVSim frontend is not reachable: `./start.sh --frontend-url http://<host>:5173`. |
| `DockAlign` has no `elevation_rad` or `search_yaw_norm` | The message file changed. Rebuild `interfaces` (section 2.1) and open a new terminal so it sources `install/setup.bash`. |
| Detector windows show lights but `elevation_valid: false` | `/Mako_01/imu_01/data` is not arriving. Check `ros2 topic echo /Mako_01/imu_01/data --once`. Do not heave until it is true. |
| `spread_px` stays large while looking up or down at the dock | Expected. A pitched camera sees an ellipse. Use `elevation_rad` for depth and `lateral_px` / `error_x_px` for heading. `aligned` becomes true again after the vehicle is level and square. |
| Dock test `dock_align stale -> neutral` | The detector is not running, or this terminal is not on `ROS_DOMAIN_ID=42`. Start section 9.2 first. |
| Dock test vehicle yaws but does not turn | The fins need forward speed. `mode=creep` or `mode=search` should show `X` around `+4 N`. If `u` stays near 0, the axial thruster is not producing flow (teleop still publishing zeros, or the session is paused). |
| Dock test dives or climbs the wrong way | Positive `elevation_rad` must dive. Watch one log line: `Z` positive is a downward force. If the vehicle goes the other way, stop it (section 13) and check the heave sign with section 7 before trying again. |
| Dock test `SAFETY TRIP` on pitch | Pitch passed 35°. The node exits and publishes zeros. Lower `gains.pitch_nm_per_px` in `dock_test.yaml` if it pitches too hard. |

---

## 15. Task: dock standoff controller

Reads `/Mako_01/dock_align` and publishes real thruster RPM and fin degrees. It tries to end **stopped**, at the dock's depth, with the heading square and the dock in the middle of the image. It does not keep driving into the funnel.

Gains and limits: `AUV/control_code/dock_test/dock_test.yaml`. They are starting values, not yet tuned on the simulator. Watch the first run.

### 15.1 Before you start
1. Sections 1 and 3: simulator playing, teleop stopped (3.4).
2. Section 2.1 if you have not rebuilt `interfaces` since `DockAlign.msg` changed.
3. Terminal 3: detector running (`./run_live.sh`, section 9.2). Leave it running. It does not publish actuator commands.
4. Stop depth hold, waypoint tracking, and station keeping. This node uses the same actuator topic.

### 15.2 Run it — Terminal 2
```bash
cd ~/Research/MAVSIM/AUV_docking/control_code/dock_test
./run_dock_test.sh
```
The script sources ROS and the workspace, sets `ROS_DOMAIN_ID` to 42 if you have not set it, and prints a reminder to stop teleop.

### 15.3 What you should see
A status line every 2 seconds:
```text
mode=creep  X=+4.0 N  Z=+11.2 N  M=-1.20  N=+0.80  u=0.35 m/s
```
`X` is axial force (positive = forward). `Z` is heave force (positive = down). `M` is pitch moment (positive = nose up). `N` is yaw moment (positive = bow to starboard). `u` is forward speed.

| `mode` | The vehicle is doing this |
|---|---|
| `search` | Fewer than four lights. Yaw and pitch follow the detector search. Heave is only the 7.25 N buoyancy trim, so depth does not change from a partial view. `X` is about `+4 N` when the fins must yaw, or about `−4 N` when the dock is too close to fit in the camera. |
| `creep` | Four lights, but the dock is still left/right in the image or the heading is not square. A small forward push (`X` about `+4 N`) gives the fins flow. Heave follows `elevation_rad`. Pitch follows `error_y_px` only to keep the dock in frame. |
| `standoff` | Depth, heading, and the image are inside the deadbands (about 1° of elevation, 10 px left/right, 10 px up/down, 8 px on the side-light cue). `X` is 0. The vehicle holds. It does not enter the funnel. |
| `backup` | Four lights and the top–bottom radius is above 200 px (dock inside roughly 2.3 m). `X` is about `−4 N`. |

Ctrl-C publishes zeros and exits.

### 15.4 What it will not do
- It will not strafe. There is no sway thruster. A sideways offset is corrected by yawing and moving forward (`mode=creep`), then stopping.
- It will not heave while `valid` is false or `elevation_valid` is false.
- It will not keep a closing speed after the standoff deadbands are met.
- Yaw does nothing useful if `u` stays near 0. The fins need flow.

### 15.5 Check the commands without moving the vehicle
No ROS:
```bash
cd ~/Research/MAVSIM/AUV_docking/control_code/dock_test
python3 test_dock_test_core.py
```
Expected: `8 tests passed`.

### 15.6 If it misbehaves
Stop it with Ctrl-C, or force zeros (section 13). Then see the dock-test rows in section 14. To make the first run gentler, lower `gains.elevation_n_per_rad`, `gains.pitch_nm_per_px`, and `gains.yaw_nm_per_px` in `dock_test.yaml`, and restart. Change one at a time.
