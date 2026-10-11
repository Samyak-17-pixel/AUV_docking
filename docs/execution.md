# Execution guide — AUV docking (Mako_01)

> **Folder layout (2026-10-11):** this guide lives in `docs/`. The viewer code is in `control_code/sim_viewer/{app,data,view3d,plots,sonar,camera}/` (launchers in `scripts/`, with wrappers `./run_sim_viewer.sh`, `./run_camera_sim.sh`, `./run_sidescan_sim.sh`, `./run_record.sh` left in `sim_viewer/`, so those commands are unchanged); single tools are run by their sub-folder path, e.g. `python3 plots/compare.py`, `python3 app/viewer_app.py`. The checklist script is `control_code/scripts/run_real_sim_checklist.sh`. Outputs: `outputs/logs/{dof_testing,station_keeping}/`, `outputs/docking_sweeps/`, `outputs/detection_reports/`, `outputs/sim_viewer_runs/`, scratch `outputs/tmp/`. Tests: `cd control_code && python3 -m pytest ...` as before (a root `pytest.ini` keeps the cache in `outputs/tmp/`).

Step-by-step instructions for every task. Each task says **which terminal**, **what to type**, **what you should see**, and
**what to do if you do not**. Last updated 2026-10-10.

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
| Fly a lawnmower survey, an orbit, a spiral, a yo-yo, or a chained mission | [19](#19-task-missions-lawnmower-orbit-spiral-yo-yo-chained-legs) |
| Edit waypoints / parameters of a controller from the viewer (map, table, tree) | [10.2](#102-desktop-viewer-3d-scene-plots-camera-image) (Config tab) |
| Test each degree of freedom / check actuator signs | [7](#7-task-per-dof-tests-actuator-allocation) |
| Hold position (hover) | [8](#8-task-station-keeping-hover) |
| Simulate the side-scan sonars, map the rugged sea floor, see the waterfall and mosaic, replay a saved survey | [20](#20-task-side-scan-sonars-and-the-rugged-sea-floor-added-2026-10-11) |
| Detect the dock lights, including a pitched view | [9](#9-task-dock-light-detection) |
| See the detector pop-ups / the separate `dock_align` window, or measure how reliable the detector is | [9.8](#98-pop-up-windows-the-dock-align-window-and-the-reliability-study-added-2026-10-11) |
| Check docking from every start pose and look at the saved good runs | [16](#16-task-align-with-the-dock-and-dock-terminal_docking) (sweep, showcase) |
| The camera pane in the viewer is blank | [10.2](#102-desktop-viewer-3d-scene-plots-camera-image) and [14](#14-troubleshooting) |
| Hold a standoff on the dock (match depth, square up, do not enter) | [15](#15-task-dock-standoff-controller) |
| Align with the dock and dock (enter the funnel), from any angle in the camera's view | [16](#16-task-align-with-the-dock-and-dock-terminal_docking) |
| Re-tune gains / the detector thresholds, or check them over ROS | [17](#17-tuning-tools-and-how-to-use-them) |
| Run everything with no simulator | [10](#10-task-offline-mode-no-simulator) |
| Run the unit tests | [11](#11-task-unit-tests) |
| Do a full docking run | [12](#12-full-docking-run-order) |
| Stop the vehicle / something went wrong | [13](#13-stop-everything) and [14](#14-troubleshooting) |

**Rules that apply to every task**
- **Every file the tools write goes to `AUV_docking/outputs/`** (runs, logs, screenshots, checklist logs, temporary configs in `outputs/tmp/`); nothing is written to your Home directory.
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
Then open **http://localhost:8888**, and start/play a session with the **Mako_01** vessel (`~/Research/MAVSIM/Mako (1).mavsim`, with **all six DOFs active**: its `active_dof` is `[1,0,1,0,0,0]` = surge and heave only, which locks yaw, pitch and roll).
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
- **CSV logs** are saved in your home directory: `outputs/logs/dof_testing/` and `outputs/logs/station_keeping/`. Open them in a spreadsheet
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
`rpm_cap: 1000` gives about 7 N of heave thrust: enough for gentle depth changes on this neutrally buoyant vehicle, but 1400 or more leaves margin for disturbances.
This controller has not yet been run on the real sim since the PWM -> RPM change: watch it the first time.

---

## 6. Task: waypoint tracking

Steers through a list of NED waypoints. **Rebuilt on `common/` on 2026-10-10**: heading is a yaw MOMENT loop turned into fin degrees by the allocator (speed-scheduled, sign mix from the vessel geometry), speed is a speed loop that keeps water flowing over the fins and slows towards each waypoint, depth is the heave loop through the allocator. (The first version spun after the first waypoint offline: its hard-coded fin signs were the opposite of the allocator's yaw mix, its fin gain was not scheduled with speed, and nothing braked the coasting vehicle.)

1. Do sections 1 and 3.
2. Edit `AUV/control_code/waypoint_tracking/waypoint_tracking.yaml`:
   - `waypoints:` list of `{x, y, z}` (North, East, Down in metres). Default is a 5 m (north) x 8 m (east) rectangle at depth 3 m. The dock is at (10, 0, 3): keep the last waypoint short of it and remember the 2.5 m turning circle swings the path about 2.5 m beyond a corner.
   - **The vehicle cannot strafe or pivot: its turning circle is about 2.5 m** (`speed.turn_radius_m`, measured offline). A waypoint that lies inside that circle after a corner is unreachable and the vehicle orbits it for ever. The node prints a warning for every such corner at start-up (the old 5 m x 3 m default had two).
   - `speed:` (cruise, flow speed, deceleration), `gains:` (same loops and numbers as station keeping), `mission.acceptance_radius_m` and `depth_acceptance_m` decide when a waypoint counts as reached.
3. Run — Terminal 2:
   ```bash
   cd ~/Research/MAVSIM/AUV_docking/control_code/waypoint_tracking
   ./run_waypoint_tracking.sh
   ```
4. Expected log every 2 s: `WP0/3 pos=(2.1,0.0,3.0) r_xy=2.9 hd_err=+0.0deg u=0.86/1.00 m/s depth_err=+0.00 fin_authority=1.00` (u = measured / target speed), then the next waypoint number as each is reached, and finally `Mission complete - actuators at neutral.` Offline (fake vehicle with realistic odometry) the shipped rectangle finishes in about 70 s.
5. Heading control uses the fins, which only work with forward speed: the target speed never drops below `speed.flow_mps` (0.6 m/s) between waypoints. Frozen or stale odometry makes it publish neutral (and forget its integrals); a depth, pitch or roll limit trips a safety stop.
6. Tests: `python3 -m pytest .` in that folder (closed loop against the offline vehicle, the corner check, line following under a push, a negative control with the old fin signs). The old file is `git show HEAD:control_code/waypoint_tracking/waypoint_tracking.py`.
7. **Edit the waypoints in the viewer instead of the YAML:** open the viewer (section 10.2), *Config* tab, pick `waypoint_tracking`: click on the map to add waypoints, drag to move them, right-click to delete; the red circles are corners the vehicle cannot make. *Apply to next Start* is on by default, so *Start* in the Controls tab uses the edited list; *Save to file* writes it into this YAML with the comments kept.
8. For surveys (lanes, circles, depth swings) use the mission controller (section 19).

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
  log: outputs/logs/dof_testing/dof_heave_step.csv
  ```
- `verdict` is `PASS`, `FAIL`, or `ABORT` (a safety limit tripped, with a reason; the vehicle is sent to neutral).

### 7.4 What a FAIL means
| Test | A FAIL points to |
|---|---|
| heave step | wrong heave sign, thrust-unit assumption (`rpm_to_rps`), or `rpm_cap` too low; `INVALID` means the sim state did not change at all (paused or frozen odometry) and the run proves nothing; a `hint` about a LOCKED DOF means no response at all |
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
7. Stop with Ctrl-C (sends zero commands, saves `outputs/logs/station_keeping/station_keeping_<time>.csv`).
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
Expected: `11 tests passed`. These project the real 1 m light circle through a pinhole camera (level, deep, pitched, offset, yawed, rolled) and check the search commands.

---

### 9.8 Pop-up windows, the `dock_align` window and the reliability study (added 2026-10-11)
```bash
cd ~/Research/MAVSIM/AUV_docking/dock_detection_algo
./run_live.sh                       # the two detector windows: "Dock camera" (banner, labelled lights, range/bearing/elevation/view-angle gauges, top-down mini map, history strip) and "Bloom mask"
./run_live.sh --align-window        # ... plus "Dock align": steering lamps (YAW L/R, UP/DOWN), the dock's offset on a target, the command in words, every DockAlign number, rolling plots
python3 dock_align_view.py          # the align window on its own (start the detector elsewhere); --save-png out.png writes one picture after 3 s and exits
python3 render_hud_samples.py /tmp/hud     # PNGs of all windows for a good lock, an oblique view, a close view and a hidden dock (no display needed)
```
Read the *Dock align* window like this: a lit **YAW R >** lamp = the dock is more than 10 px right of the picture centre, so yaw right (the fins need forward flow); lit **UP** = the dock is above the centre (pitch / heave up); both lamps dark and green = centred. While there is no lock the lower card shows the detector's search command (yaw / pitch / surge hints). `message age` turns the whole window into `no message for N s` if the detector stops.
**How reliable is the detector?** (about 2 minutes, no ROS nodes needed, workspace sourced for the message)
```bash
python3 reliability_study.py --list                         # the axes: range, view_deg, head, lat, pitch, roll, noise, jpeg, gain, side, water, background, blur, motion, bubbles, snow, backscatter, glint, distractors, reflections, occlude
python3 reliability_study.py --n 40 --tag mine              # every axis, 40 frames per value -> ../outputs/detection_reports/mine/{summary.txt,results.csv,reliability_*.png}
python3 reliability_study.py --axes distractors,background --config other.yaml --tag try      # only some axes, another detector config
python3 compare_reports.py ../outputs/detection_reports/before ../outputs/detection_reports/after     # before / after table (ships in outputs/detection_reports/comparison.txt)
python3 -m pytest test_dock_ring.py test_robustness.py test_dock_hud.py test_reliability_study.py -q      # the tests (about 1 minute)
```
`summary.txt` lists, per value: good %, bad-lock %, miss %, out-of-view %, pixel error, centre error, range error of `f / radius_px`, and ms per frame. Results are in README section 5.10b. All of it is on the SYNTHETIC camera (`sim_viewer.yaml`, `look:` and `stress:`): it says how the detector copes with those guesses, not how it will do on the real camera.

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

### 10.1 Closed loop with a synthetic dock camera (detector + `dock_test`, no mavsim)
The fake vehicle has no camera. `sim_viewer/camera/camera_node.py` adds one: it draws the four dock lights from the vehicle pose and publishes
`/Mako_01/camera_03/image/compressed` and `/Mako_01/imu_01/data`, so the **real** detector and `dock_test` run end to end.
Add `--realistic` to the fake vehicle to make its odometry behave like the real sim's (about 4.5 Hz with jitter, 0.25 s late, noisy, and occasional 5-25 s FREEZES: `--odom-freeze-interval 0` turns the freezes off). The viewer's *Start offline stack* uses it by default.
Use a private ROS domain; the launcher refuses domain 42 (the real bridge). Each line below goes in its own terminal, after
`export ROS_DOMAIN_ID=77 ROS_LOCALHOST_ONLY=1`, `source /opt/ros/humble/setup.bash` and `source ~/Research/MAVSIM/AUV_docking/control_code/ws/install/setup.bash`:
```bash
cd ~/Research/MAVSIM/AUV_docking/control_code
python3 sim_offline/fake_vehicle.py --realistic --x 0 --y 0.8 --z 3   # 1. vehicle 9.4 m from the dock, 0.8 m off the centreline (realistic odometry)
./sim_viewer/run_camera_sim.sh                                  # 2. synthetic nose camera
../dock_detection_algo/run_live.sh                              # 3. the real detector (opens its two windows; add --no-gui for none)
./dock_test/run_dock_test.sh                                    # 4. the standoff controller
```
Watch `ros2 topic echo /Mako_01/dock_align --once` for `valid: true`, `num_lights: 4`. Expected with the shipped settings: the detector locks on
around 8 m, the vehicle closes at up to ~0.45 m/s and stops at about 2.3-2.6 m from the dock plane, and never closer. The heading then
drifts (the fins have no authority at zero speed); that is a known limitation, not a simulator bug.
Stop in reverse order with Ctrl-C, then check `ps -eo pid,cmd | grep -E "fake_vehicle|camera_node|live_dock|dock_test"`.
How the lights LOOK is an assumption (`sim_viewer/sim_viewer.yaml`, `look:`); compare against a few real camera_03 frames before trusting range limits.

### 10.2 Desktop viewer (3D scene, plots, camera image)
A PyQt5 window that shows the vehicle and the dock in 3D (the real meshes from the vessel file), live plots, the nose-camera image and a status panel.

**Quick look, no ROS and no other terminals** (scripted manoeuvres in-process):
```bash
cd ~/Research/MAVSIM/AUV_docking/control_code/sim_viewer
./run_sim_viewer.sh --demo
```
**With the offline closed loop** (2026-10-11: `./run_sim_viewer.sh` now uses the PRIVATE domain 77 by default, so the camera feed shows up with no exports; before, it defaulted to the real-mavsim domain 42, where the camera pane stayed blank and the stack button was disabled):
```bash
cd ~/Research/MAVSIM/AUV_docking/control_code/sim_viewer
./run_sim_viewer.sh                      # opens on domain 77; Controls tab -> "Start offline stack" starts the fake vehicle, the synthetic camera (the picture pane) and the detector
./run_sim_viewer.sh --view top           # add --view top|side|dock|onboard|free|follow to pick the camera
ROS_DOMAIN_ID=78 ./run_sim_viewer.sh     # a domain you export yourself is respected (the stack then uses the same one)
```
The camera pane says why it is empty (domain, frames received, what to start). Tick *Open the detector pop-up windows* in the Controls tab before *Start offline stack* if you want the detector's windows (needs a display). The path vehicle -> camera node -> viewer is tested over ROS (`sim_viewer/tests/test_camera_feed_ros.py`).
**With the real mavsim:** `./run_sim_viewer.sh --real` (= `ROS_DOMAIN_ID=42`, read-only: it only subscribes). The 3D view interpolates between the ~4 Hz odometry samples.

**Controls tab** (next to Status): pause/resume, step 0.5 s, time scale, reset the vehicle to a pose, push it (presets or custom force/moment), start/stop a controller with edited gains, or start the whole offline stack (fake vehicle + synthetic camera + real detector) with one click. These simulation controls work only with the offline fake vehicle or `--demo`; against the real mavsim they stay disabled. Starting a controller while on ROS domain 42 asks you to confirm, and the offline stack refuses to start there. Edited gains go to a temporary copy; the YAML file is never changed. Controller setpoints appear as dashed lines in the plots.

**Stopping and switching controllers (Controls tab).** *Stop running controller* stops WHATEVER is running (also a controller you started in a terminal: it is found by its process and sent Ctrl-C, which makes it publish neutral). Starting another controller while one runs asks *Stop X and start Y?* and does both in order. The red **EMERGENCY STOP** button stops everything with a 1 s grace and publishes a neutral actuator command itself. The text under the buttons always says what is running.
**Live gains.** Under *Live gains* pick a loop (heave, pitch, yaw ...) and a number (kp, kd, ki, lpf_tau_s ...), change it with the box or the x0.8 / x1.25 buttons: the running controller applies it at once (it logs `live gain heave.kp: 7.5 -> 9`). *Save as new default...* writes the value into the controller's YAML in place (comments kept; asks first). Works for terminal_docking, dof_testing and station_keeping (they listen on `/Mako_01/ctrl_gains`); not in `--demo` or replay.
**Config tab (2026-10-10): edit a controller's YAML from the window.** Pick the file (`waypoint_tracking`, `mission`, `dock_test`, `station_keeping`, ...).
- *Map* (top view, north up): the path, the dock, the geofence and dock keep-out circle, the vehicle, its trail, and the planned / covered sensor swath with a coverage percentage. For `waypoint_tracking`: **click** adds a waypoint, **drag** moves it, **right click** deletes it; for `mission`: tick *click sets selected leg position* and click to place the selected leg's origin / centre / start. Mouse wheel zooms, middle drag (or Shift + left drag) pans, *Fit map* frames everything. Red circles mark corners the vehicle cannot make.
- *Table*: waypoints (x, y, z; edit a cell) or mission legs (select a row, edit its numbers / switches / lists in the form, *Add* with the leg type box, *Remove*, *Up*, *Down*).
- *Parameters*: every other value as a tree (numbers, true/false and TEXT) with the file value next to it and a filter box; changed values are highlighted.
- *Checks* box: the controller's own checks as you edit (turning circle, geofence, dock keep-out, missing leg parameters, depth limits).
- *Apply to next Start* (ticked by default): *Start* in the Controls tab then runs the controller with a temporary copy of the edited values and the Controls tab says `edited (N values)`; the YAML file is NOT changed. *Save to file* asks first, copies the old file to `<file>.yaml.bak` and writes only the changed text, so every comment stays (if that is not possible it refuses and writes nothing). *Save as new file...* writes a copy and leaves the original alone (start it with `--config <file>`). *Revert* returns to the file values, *Preview file text* shows what would be written.
- While a `mission` runs, the tab shows the leg, a progress bar, the ETA, the cross-track error and the swath coverage; the **Mission** plot tab (under the 3D view) shows cross-track error, heading error, speed vs setpoint, depth error, progress and the path against the plan.
- **Run summary:** when a `mission` or `waypoint_tracking` run ends, the viewer saves `outputs/sim_viewer_runs/auto_<controller>_<time>_summary.png` and lists the numbers in the Status tab; for any recorded run: `python3 sim_viewer/plots/run_summary.py RUN.csv --mission mission/mission.yaml --swath 3`.
**Scenarios tab.** *Live*: choose range, lateral offset and heading offset, press *Run scenario*: the fake vehicle is put there, `terminal_docking` starts and a judge shows PASS/FAIL live (wall contact, bad entry, overshoot, retry, depth/attitude limits, pinned thruster, timeout; judged on the fake vehicle's ground truth). Needs the offline stack. *Batch*: runs `terminal_docking_eval.py` for the chosen grid/plants/vision in a separate process and fills a scoreboard (also saved to `outputs/sim_viewer_runs/scoreboard_<time>.csv`).
**History tab.** Every controller run seen by the viewer is recorded to `outputs/sim_viewer_runs/auto_<controller>_<time>.csv` (switch off with `viewer.auto_record: false`). Select a run: *Replay* (opens a second viewer window), *Replay + model* (ghost vehicle), select two (Ctrl+click) and *Compare 2 selected* for an overlay plot and a table of differences (e.g. before/after a gain change), *Delete*.
**Look and HUD.** The 3D view shows a procedural sea floor (sand, ripples, rocks), a rippling surface seen from below, sun shafts, drifting particles, cones of light from the four dock lights and the vehicle's shadow; a heads-up display (depth, heading, speed, dock range/bearing, controller phase), a top-view minimap and the detected lights marked on the camera pane. All of it is decoration, configured in `sim_viewer.yaml` under `viewer.look3d` / `viewer.hud` (`enabled: false` = the old plain scene).

Mouse in the 3D view: left drag = orbit, middle drag or Shift+left = pan, right drag or wheel = zoom. Buttons: reset the trail, overview camera, pause the display, save a screenshot (`outputs/sim_viewer_shots/`).
Status panel: `STALE` appears if odometry is older than 2 s; `dock_align` shows `valid`, lights found, `err_x`, `err_y`, elevation and radius.
Settings (refresh rates, plot window, trail length, mesh reduction, colours): `sim_viewer.yaml`, section `viewer`.
Needs a desktop session (`echo $DISPLAY`). If the window opens but the 3D area stays empty, check the terminal for a VTK/GL error.
Tests: `cd ~/Research/MAVSIM/AUV_docking/control_code && python3 -m pytest sim_viewer/tests -q`; the ROS tests (`test_ros_link.py`, `test_fake_vehicle_sim_control.py`, `test_ctrl_debug_ros.py`) need the workspace sourced and plugin autoload off:
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest sim_viewer/tests -q` (after sourcing ROS and `ws/install/setup.bash`).

**Detection tab (2026-10-11).** In the bottom-left tab bar (Vehicle / Dock detector / Mission / **Detection**): the detector at work next to the 3D view, so you can check a trajectory and what the detector saw together. It shows the camera picture with the labelled lights, a banner (LOCKED-ALIGNED / LOCKED-off centre / SEARCHING n of 4 / NO LIGHTS), range, bearing, elevation and view-angle gauges, the steering lamps (YAW L / YAW R / UP / DOWN light when the dock is more than 10 px off the picture centre) and a top-down map of the dock. It works live (the offline stack and the real sim) and in a replay; *Detach (big windows)* opens the full camera view and the dock_align view side by side, *Save picture* writes `outputs/sim_viewer_shots/detection_<time>.png`. Open it first with `./run_sim_viewer.sh --replay FILE --bottom-tab Detection --view top`. The numbers behind it are the same `DockAlign` fields as the pop-up windows (section 9.8).

**Replaying a recorded run (2026-10-11): the camera pane now works in a replay.** A recording holds no pictures, so the replay re-draws the nose-camera picture from the recorded pose (the same synthetic camera) and runs the real detector on it: the lights overlay, `lights 4 VALID r 90px` and the Dock detector plots are live, as in the offline stack (`ReplaySource(..., camera=False)` switches it off). The saved docking runs: `./run_sim_viewer.sh --replay ../../outputs/sim_viewer_runs/showcase_odometry_straight_in_r8_lp0_hp0.csv --view dock` (or the History tab). Test: `sim_viewer/tests/test_replay_camera.py`.

### 10.3 Record a run, replay it, compare it with the model, calibrate the model
All of this works on CSV files, so a run recorded on the REAL sim can be examined later with no simulator running.

**1. Record** (read-only; it only subscribes). Any terminal, same ROS setup as the controllers:
```bash
cd ~/Research/MAVSIM/AUV_docking/control_code/sim_viewer
ROS_DOMAIN_ID=42 ./run_record.sh --label heave_step                 # real sim; Ctrl-C (or --duration 60) to stop
ROS_DOMAIN_ID=77 ROS_LOCALHOST_ONLY=1 ./run_record.sh --duration 60 # the offline fake vehicle
./run_record.sh --label look_check --frames 10 --frames-every 5      # also saves 10 camera_03 JPEGs (+ frames.csv with the pose) in run_..._frames/
```
Start it BEFORE the controller, run the test, then stop it. File: `outputs/sim_viewer_runs/run_<time>.csv` (change with `--out`). The CSVs written by `dof_testing`
(`outputs/logs/dof_testing/`) and `station_keeping` (`outputs/logs/station_keeping/`) can be used directly too (station_keeping logs from before 2026-10-08 lack the pose columns and are refused).

**2. Replay in the viewer** (no ROS needed):
```bash
./run_sim_viewer.sh --replay outputs/sim_viewer_runs/run_XXXX.csv            # Replay tab: Play/Pause, slider, speed, loop
./run_sim_viewer.sh --replay FILE.csv --overlay                         # + ghost vehicle and dashed lines = the MODEL driven by the same commands
./run_sim_viewer.sh --replay FILE.csv --overlay --overlay-mode free     # model from the first sample only (shows accumulated drift)
```
`--overlay-mode segments` (default, `--segment-s 5`) restarts the model from the recorded state every 5 s, so the ghost stays near the vehicle and only short-term differences show.

**3. Numbers instead of pictures:**
```bash
python3 plots/compare.py FILE.csv --mode segments --segment-s 5 --plot overlay.png
python3 plots/compare.py FILE.csv --set rpm_to_rps=0.0233                      # try a parameter value before editing any YAML
```
Read the table: `RMS err` per channel in its unit, and `RMS / range` (error relative to how much that channel moved). Small = the numbers in `common/mako_geometry.yaml` describe the vehicle.
Large, or a **bias** that keeps one sign = a wrong assumption (thrust scale, drag, net buoyancy ...). The recorded command is held between rows; the real sim's command delay is not modelled.

**4. Fit the model to the recording:**
```bash
python3 plots/calibrate.py FILE.csv [FILE2.csv ...]                            # default parameters: rpm_to_rps, thruster_tau_s, drag_quad_X/Z/M, net_up_n, added_mass_frac_w
python3 plots/calibrate.py FILE.csv --params net_up_n,drag_quad_Z --json fit.json
```
Look at the `status` column first: `ok` = measurable from this run, `weak` = entangled with another parameter (for example thrust scale and surge drag cannot be separated by a plain surge run: only their ratio is known) or noisy,
`NOT IDENTIFIABLE` = this run does not depend on it (fin lift slope needs forward speed AND fin commands). The standard errors are optimistic (they ignore model-structure error). It prints the lines to change in `mako_geometry.yaml` and never edits anything: check the change with `compare.py --set` first.
Good recordings: a heave step from rest (net buoyancy, heave drag, added mass), a surge step (thrust scale, lag), a pitch pulse (pitch drag). If a fit stays poor (the report says so), the model STRUCTURE is missing something, such as the pitch pendulum seen on the live sim on 2026-10-01.
Verified offline: the fit recovers parameters that were deliberately changed in a synthetic run (lag 0.35 s, heave drag 200, net buoyancy +1 N ...) to within a few per cent; it has NOT been run on a real-sim recording yet.

---

## 11. Task: unit tests

Everything (control, dock_test, viewer):
```bash
cd ~/Research/MAVSIM/AUV_docking/control_code
python3 -m pytest tests dock_test waypoint_tracking mission sim_viewer/tests terminal_docking_control tuning ../dock_detection_algo/test_tune_detection.py ../dock_detection_algo/test_imu_convention.py ../dock_detection_algo/test_dock_ring.py ../dock_detection_algo/test_robustness.py ../dock_detection_algo/test_dock_hud.py ../dock_detection_algo/test_reliability_study.py -q      # no ROS needed: 400 passed + 21 skipped (the ROS tests)
# with ROS: also runs the tests that start real fake_vehicle / camera / recorder processes (about 80 s)
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1; source /opt/ros/humble/setup.bash; source ws/install/setup.bash
python3 -m pytest tests dock_test waypoint_tracking mission sim_viewer/tests terminal_docking_control tuning ../dock_detection_algo/test_tune_detection.py ../dock_detection_algo/test_imu_convention.py ../dock_detection_algo/test_dock_ring.py ../dock_detection_algo/test_robustness.py ../dock_detection_algo/test_dock_hud.py ../dock_detection_algo/test_reliability_study.py -q      # 447 passed (+ 11 more in ../dock_detection_algo/test_pitched_geometry.py, which needs ROS)
```
They check allocation signs (heave thrusters point up), the PID, all DOF tests and station keeping against the offline vehicle, the dock_test loops, the synthetic camera and the real
detector on its frames, the viewer (3D scene, plots, Controls tab, Replay tab, whole window headless), the recorder against a live fake vehicle, replay, compare and calibration recovery.
Qt/VTK tests use the `offscreen` Qt platform and still need `DISPLAY` for VTK (they skip without it).

The dock detection geometry has its own script (needs ROS sourced):
```bash
cd ~/Research/MAVSIM/AUV_docking/dock_detection_algo && python3 test_pitched_geometry.py     # 11 tests passed
```

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
| Depth hold cannot reach depth | `rpm_cap: 1000` gives only ~7 N of heave thrust; use 1400+ (station keeping and DOF tests use 1800). Also check that the session has heave active. |
| DOF yaw/roll step FAILs | Fin sign or fin moment arm (unverified) in `common/mako_geometry.yaml`. |
| DOF test `ABORT speed spin-up timeout` | Vehicle did not reach 1 m/s: lower `fin_dofs.cruise_speed_mps` in `dof_testing.yaml` or raise the surge limits. |
| Station keeping `SAFETY TRIP ... drifted` | Sideways drift cannot be corrected: raise `safety.max_drift_m`, or restart. |
| Station keeping `SAFETY TRIP ... too shallow` at start | A forgotten fake vehicle or a second odometry source is running, or depth < `safety.min_depth_m`. `ps -eo pid,cmd \| grep fake_vehicle`. |
| Station keeping heading drifts | Expected with no flow over the fins (below ~0.3 m/s). |
| Controller exits immediately | The depth or waypoints are already satisfied: change the setpoint in its YAML. |
| Detector: topic listed but no image | Fast-DDS shared memory: the scripts set `FASTRTPS_DEFAULT_PROFILES_FILE=dock_detection_algo/fastrtps_no_shm.xml`. Set the same in any terminal where you run `ros2 topic hz/echo` on a camera. |
| Detector: `interfaces.msg.DockAlign not found` | Section 1 / 2.1, or use `./run_live.sh` (it sources the workspace). |
| Viewer camera pane says `no camera image` in a REPLAY | Fixed 2026-10-11: the replay re-renders the picture from the recorded pose. If it still shows nothing the run's CSV has no usable poses, or the window was started from a folder where `synthetic_camera.py` cannot be imported (use `./run_sim_viewer.sh`). |
| Viewer camera pane says `no camera image` | Read the text in the pane: it shows the ROS domain, the camera/odometry frame counts and what to start. The launcher now defaults to the private domain 77; in the Controls tab press *Start offline stack* (fake vehicle + synthetic camera + detector). A viewer started with `--real` or `ROS_DOMAIN_ID=42` only shows the real bridge's camera while a mavsim session is running and its controller is up. |
| Detector pop-up windows are empty / do not open from the viewer | The viewer starts the detector with `--no-gui` unless *Open the detector pop-up windows* is ticked (Controls tab) before *Start offline stack*; from a terminal use `./run_live.sh`. They need `DISPLAY` and `opencv-python` (not the headless build). |
| Detector: no GUI windows | `echo $DISPLAY` must be set (e.g. `export DISPLAY=:0`); install `opencv-python`, not the headless build. |
| `ModuleNotFoundError: yaml` | `pip install pyyaml`. |
| Bridge has no camera images | The MAVSim frontend is not reachable: `./start.sh --frontend-url http://<host>:5173`. |
| `DockAlign` has no `elevation_rad` or `search_yaw_norm` | The message file changed. Rebuild `interfaces` (section 2.1) and open a new terminal so it sources `install/setup.bash`. |
| Detector windows show lights but `elevation_valid: false` | `/Mako_01/imu_01/data` is not arriving. Check `ros2 topic echo /Mako_01/imu_01/data --once`. Do not heave until it is true. |
| `spread_px` stays large while looking up or down at the dock | Expected. A pitched camera sees an ellipse. Use `elevation_rad` for depth and `lateral_px` / `error_x_px` for heading. `aligned` becomes true again after the vehicle is level and square. |
| Dock test `dock_align stale -> neutral` | The detector is not running, or this terminal is not on `ROS_DOMAIN_ID=42`. Start section 9.2 first. |
| Dock test vehicle yaws but does not turn | The fins need forward speed. `mode=creep` or `mode=search` should show `X` around `+4 N`. If `u` stays near 0, the axial thruster is not producing flow (teleop still publishing zeros, or the session is paused). At the hold distance with a leftover heading error the speed target used to be 0 there (no fin flow); since 2026-10-10 it backs away (`mode=realign`, section 15) to get flow. |
| Dock test `mode=realign` keeps repeating, or the vehicle swings in heading | The yaw loop is too stiff for the delays: lower `gains.yaw_nm_per_px`, raise `gains.yaw_kd` (section 15.5b has the tool), or turn the behaviour off with `speed.flow_min_mps: -1`. It also needs about 2 m of clear water behind the vehicle. |
| Dock test backs away in `mode=search_back` and never settles | The dock is partly out of frame and the node has no distance memory: after `speed.blind_max_forward_m` (0.25 m) of forward search it reverses until the whole dock is seen (this replaced driving into the dock from a close start). Start farther out, or yaw towards the dock by hand first; set `blind_max_forward_m: -1` only if you want the old blind surge (it can hit the dock). |
| Docking: `SEARCH` for ever, circling | The dock cannot be seen from where the vehicle is (from the side or behind the dock the lights are dark), or it started 2-4 m from the dock with the nose away. Give it a rough position: `recover.dock_hint_ned` (section 16); with no hint start 6 m or more in front of the mouth. |
| Docking: it backs out although it looks lined up | `recover.turn_radius_m` (2.4 m, assumed) is too big for the real vehicle, or `margin` is too high: lower them after the yaw-step test measured the real turning circle. `recover.enabled: false` switches the behaviour off. |
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
| `search` | Fewer than four lights. Yaw and pitch follow the detector search. Heave is only the buoyancy trim (0 N for the current neutral vessel), so depth does not change from a partial view. `X` regulates the speed: about `speed.search_mps` (0.5 m/s) forward when the fins must yaw, or a slow reverse when the dock is too close to fit in the camera. |
| `creep` | Four lights, but the dock is still left/right in the image or the heading is not square. The speed loop holds `speed.creep_mps` (0.45 m/s) so the fins have flow, slowing along a stopping-distance envelope as the dock gets closer (it reaches 0 m/s at about 2.4 m). Heave follows `elevation_rad`. Pitch follows `error_y_px` only to keep the dock in frame. |
| `realign` | NEW 2026-10-10. The heading is wrong but the stopping envelope allows no forward speed (the vehicle is at the standoff), so the fins would have no flow. It BACKS AWAY at `speed.realign_mps` (0.55 m/s) while the yaw loop turns it square (the allocator flips the fin sign for reverse flow). It stops backing when the heading is square and still, or when it is `speed.realign_max_back_m` (2 m) beyond the standoff. Then it creeps forward again if still needed. Needs clear water behind the vehicle. `speed.flow_min_mps: -1` switches this off (the old behaviour). |
| `standoff` | Depth, heading, and the image are inside the bands (about 1° of elevation, 20 px left/right for the yaw moment, 61 px for the decision to move, 10 px up/down, 8 px on the side-light cue) and the vehicle is not rotating faster than `speed.settle_yaw_rate_rad_s`. `X` is 0. The vehicle holds. It does not enter the funnel. |
| `backup` | Four lights and the top–bottom radius is above 200 px (dock inside roughly 2.3 m). The speed target is `−speed.backup_mps` (0.3 m/s). |

Ctrl-C publishes zeros and exits.

### 15.4 What it will not do
- It will not strafe. There is no sway thruster. A sideways offset is corrected by yawing and moving (`mode=creep` or `realign`), then stopping.
- It will not turn square from a very close start with the dock partly out of view. With fewer than four lights and no distance memory the detector's `search` asks for a forward surge; the old node followed it into the dock (3 m start with 24° or more of heading error: 3/3 ROS runs, closest 0.14 m). Since 2026-10-10 a blind search is limited to `speed.blind_max_forward_m` (0.25 m) of forward travel, then `mode=search_back` reverses until the whole dock is seen. Over ROS with the real detector (3 m, +30° and -30°): at rest, closest 2.28 m and 2.18 m; with `--set speed.blind_max_forward_m=-1` (the old behaviour) +30° ends 0.75 m from the dock and fails. Still: start 6 m or farther out when you can, or use `terminal_docking` (section 16), which remembers the dock.
- It will not heave while `valid` is false or `elevation_valid` is false.
- It will not keep a closing speed after the standoff deadbands are met.
- Yaw does nothing useful if `u` stays near 0. The fins need flow.

### 15.5 Check the commands without moving the vehicle
No ROS:
```bash
cd ~/Research/MAVSIM/AUV_docking/control_code/dock_test
python3 test_dock_test_core.py
```
Expected: `pytest -q` in that folder passes (33 tests: sign checks, speed loop, the heading-error realign, the close-start blind-search guard with a negative control, and closed-loop runs of the standoff harness).

### 15.5b Heading error at the standoff: harness, tuning, ROS check (added 2026-10-10)
```bash
cd ~/Research/MAVSIM/AUV_docking/control_code/dock_test
python3 standoff_sim.py                    # table: 3 plants x yaw errors, real core + allocator + vehicle + odometry sensor model (no ROS)
python3 tune_standoff.py --set gains.yaw_kd=8 --set deadband.hold_extra_px=30     # score one candidate (runs at rest / closest approach)
python3 tune_standoff.py --random 100      # random search over RANGES (edit them in the file); about 6 min on 32 cores
# real ROS on a PRIVATE domain: fake vehicle + synthetic camera + the real detector + the dock_test node
source /opt/ros/humble/setup.bash; source ../ws/install/setup.bash
python3 ros_standoff_confirm.py --domain 71 --yaws 15,-20 --seconds 90 [--trace]
python3 ros_standoff_confirm.py --standoff 3 --yaws 30,-30 --seconds 80                        # the close start with the dock partly out of frame (the blind-search guard)
python3 ros_standoff_confirm.py --standoff 3 --yaws 30 --seconds 60 --set speed.blind_max_forward_m=-1       # the same with the guard OFF = the old behaviour (expect FAIL)
```
Offline-model numbers (assumed plants): 90 fresh runs (3 plants, yaw starts -35..28°, lateral 0/±0.5 m): with the old behaviour 7/90 end at rest, with the new 38/90 (mean 75% of the last 20 s at rest, closest approach 1.96 m, one run under 2.0 m). Over ROS with the real detector: starts at 3 m with yaw 8, -12, 15, -20° came to rest (closest 2.2-2.4 m); the close starts of +30° and -30° (dock partly out of frame) came to rest too with the blind-search guard (closest 2.28 m and 2.18 m); with the guard off (`--set speed.blind_max_forward_m=-1`) +30° FAILS (closest 0.75 m, 0 % at rest). Offline (`standoff_sim.py ... blind_surge=1`): closest approach 2.2-2.5 m on the nominal and light plants and 1.7 m on the heavy+late one, versus through the dock without the guard.

### 15.6 If it misbehaves
Stop it with Ctrl-C, or force zeros (section 13). Then see the dock-test rows in section 14. If it swings in heading, lower `gains.yaw_nm_per_px` or raise `gains.yaw_kd`, or switch the realign off with `speed.flow_min_mps: -1`; the blind-search budget is `speed.blind_max_forward_m` (-1 = off). To make the first run gentler, lower `gains.elevation_n_per_rad`, `gains.pitch_nm_per_px`, and `gains.yaw_nm_per_px` in `dock_test.yaml`, and restart. Change one at a time.

---

## 16. Task: align with the dock and dock (`terminal_docking`)

`control_code/terminal_docking_control/` is the controller that takes the vehicle from "the dock is in the camera's view" to "nose inside the funnel, stopped". It uses ONLY `/Mako_01/dock_align` (the four light pixels) and `/Mako_01/odometry_sim`; the dock's world position is estimated, not hard-coded. Phases (shown in the viewer and in `/Mako_01/ctrl_debug`): `WAIT` (the first 3 s) -> `SEARCH` (2026-10-11: the dock has never been seen: a slow circle, or a route to a staging point if you give it a rough dock position, see below) -> `APPROACH` (steer onto the dock axis while moving) -> gate check -> `TERMINAL` (slow, stop the nose 0.5 m inside) -> `DOCKED`; `RETRY` (back off and try again: after a failed gate (max 2) or as a PLANNED back-out (max 3, `recover.*`) when the remembered dock pose says the vehicle cannot get onto the axis in the room it has; the viewer shows `back-outs N`), `SAFE_STOP` (odometry frozen or stale: brake open loop and wait).

**Offline, one command per terminal** (each after `export ROS_DOMAIN_ID=77 ROS_LOCALHOST_ONLY=1`, `source /opt/ros/humble/setup.bash`, `source ~/Research/MAVSIM/AUV_docking/control_code/ws/install/setup.bash`):
```bash
cd ~/Research/MAVSIM/AUV_docking/control_code
python3 sim_offline/fake_vehicle.py --realistic --x 2 --y 0.8 --z 3 --yaw 8      # 1. 8 m out, 0.8 m east of the axis, heading 8 deg crooked
./sim_viewer/run_camera_sim.sh                                                   # 2. synthetic nose camera
../dock_detection_algo/run_live.sh --no-gui                                      # 3. the real detector (no windows)
./terminal_docking_control/run_terminal_docking.sh                               # 4. the docking controller
./sim_viewer/run_sim_viewer.sh                                                   # 5. (optional) watch it: HUD shows the phase
```
Or do all of it from the viewer: *Controls -> Start offline stack*, then *Scenarios -> Run scenario* (pick the start pose there).
Expected: `APPROACH` for ~15 s while the detector locks (valid, 4 lights) and the vehicle steers onto the axis, `TERMINAL` for the last ~1.5 m, then `DOCKED` with the nose about 0.5 m inside the funnel; Ctrl-C writes `outputs/terminal_docking_logs/terminal_docking_<time>.csv`.
**Start it only when the dock is in the camera's view** (the dock must have been seen once: with no detection yet the controller sends nothing). Stop teleop and every other controller first. On the real sim (domain 42) start with the vehicle far from anything else and be ready to press Ctrl-C: nothing here has run on the real simulator.

**Check it without ROS or a window** (in-process closed loop: vehicle + realistic odometry + camera + the REAL detector + the controller, judged on ground truth):
```bash
cd ~/Research/MAVSIM/AUV_docking/control_code/terminal_docking_control
python3 terminal_docking_eval.py --grid wide --vision perfect          # 240 runs in ~10 s (4 plants x 60 starts); --vision detector renders frames (minutes)
python3 terminal_docking_eval.py --grid moderate --plants nominal --vision detector --show-pass
python3 terminal_docking_eval.py --grid wide --set speed.cruise_mps=0.6 --set guidance.gate_s_m=2.0     # try a changed value without editing the yaml
```
It prints the pass count, the failure reasons, and every failing start with its reason. A run is a **first-try pass** only if there was no back-off and none of the failures happened: wall contact, bad entry (speed > 0.5 m/s, heading > 5 deg, lateral/vertical > 15 cm at the mouth), overshoot or stopping short, lost dock (> 10 s invalid inside 4 m), timeout (180 s), more than 2 back-offs, depth outside 0.5-8 m, tilt > 30 deg, a thruster at its cap > 2 s, heave-command chatter > 300 RPM p-p, oscillation, acting on stale odometry. "Reachable" in the output is a screening rule (can a vehicle with this turning radius line up in the room it has), not a proof.
**Over ROS** (the whole stack on a private domain, 10 starts, live judge): `python3 ../tuning/ros_dock_confirm.py` (about 8 minutes). Add `--starts 7,0.5,10:9,-1,0` (range,lateral,heading) for your own, `--freeze 15` to include odometry freezes.

**Search and back-out (2026-10-11), `recover:` in `terminal_docking.yaml`.** The vehicle cannot sway, so it needs room to line up, and the lights are only visible from inside the dock's 82 degree cone:
- *Too close / too far off the axis to line up in time* -> it backs out straight along the axis (heading held), then approaches again. The test is `room_needed_m` (Dubins-style: two arcs of the turning circle `recover.turn_radius_m` = 2.4 m, ASSUMED, plus a straight piece; turning away from the axis costs one more arc). Errors below `dead_e_m` 0.2 m / `dead_chi_deg` 4 deg are ignored. Judged only when the dock estimate is good (`min_axis_sigma_deg` 8). `recover.enabled: false` returns to the old behaviour.
- *Dock never seen* -> `SEARCH`. Blind (default): a slow circle (`search_speed_mps`) that turns the camera round; partial lights steer it with the detector's hints, and it backs away when the ring is bigger than the picture. **With a rough dock position** (`recover.dock_hint_ned: [10.0, 0.0, 3.0]`, `dock_axis_deg: 0`) it is much safer: it first leaves the dock's swing zone (straight away, or reverse), goes round to a staging point `hint_stage_m` = 7 m in front of the mouth, turns to the axis heading, then drives at the dock until the ring appears (backing off if it comes closer than `hint_see_m` 4.5 m). Edit the hint in the viewer's Config tab, or `--set recover.dock_hint_ned=[10.6,-0.5,3.1]` in the sweep tools.

**Vehicle position from odometry or from dead reckoning (2026-10-11).** `estimator.position_source: dead_reckoning` in `terminal_docking.yaml` (or `--set estimator.position_source=dead_reckoning` in the sweep tools): the vehicle x, y are integrated from the filtered body speed and attitude since the start instead of read from the odometry position; the dock is triangulated from the lights either way. Use it when the odometry position drifts or jumps (the sweeps below compare both). With it, `recover.dock_hint_ned` is relative to the START pose, and it still needs the speed/heading from the odometry twist and IMU (without any speed it cannot do the last 2 m). Prove it offline: `python3 -m pytest terminal_docking_control/test_position_source.py -q`; inject faults in the sim with `Scenario(odom_faults={'pos_drift_mps': (0.05, 0.03)})` (also `pos_offset_m`, `pos_jump_m` + `pos_jump_t_s`, `drop_twist`).

**Try every start, then look at the good runs** (all offline, private ROS domain not needed, workspace sourced):
```bash
cd ~/Research/MAVSIM/AUV_docking/control_code/terminal_docking_control
python3 docking_sweep.py --grid full --hint near --T 260 --tag mine --workers 24     # 330 starts, ~30 min on 32 cores: range 2-10 m x lateral -3..+3 m x heading -75..+75 deg
python3 docking_sweep.py --grid quick --hint near                                     # 45 starts, ~2 min (also: --grid close for starts at 1.5-4 m)
python3 docking_sweep.py --rerun ../../outputs/docking_sweeps/mine/results.csv --hint near --T 260 --tag again     # only the failures of an earlier sweep (test a change fast)
python3 docking_sweep.py --grid full --set recover.enabled=false --tag old_controller          # the same grid with the old controller (negative control)
python3 showcase.py --from-csv ../../outputs/docking_sweeps/mine/results.csv --hint near --tag odometry --out ../../outputs/sim_viewer_runs/docking_showcase/odometry      # -> .../odometry/index.html (about 12 good runs + 3 that failed, labelled) (about 12 good runs + 3 that failed, labelled)
python3 showcase.py --starts "7,0,0;4,1.5,30;3,-1.5,45" --hint near                    # your own starts
```
Open `outputs/sim_viewer_runs/docking_showcase/odometry/index.html` in a browser: per run a top view (path coloured by phase: SEARCH blue, APPROACH green, TERMINAL dark green, RETRY/back-out orange, DOCKED red), a timeline (range, cross-track, heading, speed, thrust), four camera pictures with the detector pop-up drawing, and `trajectory.csv` (all seven actuator commands: three thrusters and ALL FOUR fins; the timeline has a fin panel); each CSV is also copied to `outputs/sim_viewer_runs/showcase_<tag>_<name>.csv` so the viewer's **History tab** can replay it. `outputs/docking_sweeps/<tag>/feasibility_map.png` is the map of outcomes by start pose. Outcomes: DOCKED, RECOVERED (docked after a planned back-out), NOT_SEEN, COLLISION, FAIL (timeout etc.).
**All saved trajectories in one page: `outputs/sim_viewer_runs/index.html`** (docking with odometry position, docking with dead-reckoned position, the missions, the missions with odometry freezes, the earlier showcase). Both docking showcases are built from the sweeps below: `python3 docking_sweep.py --grid full --hint near --T 260 --tag after_odometry`, and the same with `--set estimator.position_source=dead_reckoning --tag after_deadreckoning`, then `showcase.py --from-csv outputs/docking_sweeps/<tag>/results.csv --hint near --tag <odometry|dead_reckoning> [--set estimator.position_source=dead_reckoning] --out ...`.
**Odometry vs dead reckoning (2026-10-11), same 330 starts:** odometry position 311/330 dock, 2 collisions; dead-reckoned position 300/330 dock, 14 collisions (7 marginal lateral entries of 0.15-0.18 m, 4 entry headings of 6-16 deg, 3 wall contacts from close starts). Under odometry position faults dead reckoning holds up: a 0.7 m jump 43/45 against 39/45 on the quick grid, drifts of 0.17 / 0.35 m/s docked first try / 2 of 3 against back-outs only / failure (README section 9 item 23). Compare the two feasibility maps in `outputs/docking_sweeps/after_odometry/` and `outputs/docking_sweeps/after_deadreckoning/`; fault runs: `docking_sweep.py --grid quick --hint near --faults '{"pos_drift_mps":[0.05,0.03]}'`.
**Results (2026-10-11, nominal plant, real detector, synthetic camera, from the final_hint2 sweep before the position-source change):** **311/330 dock with a rough dock position (52 first try, 259 after a planned back-out, 1 collision, 18 unsolved), 224/330 blind with 101 collisions; all 68 starts with the dock in view dock.** Not solved (the 18 timeouts and the 1 collision): 2 m ahead and 3 m to the east with headings -75..+30 (8 starts, one of them the collision), 2-3 m ahead with the nose pointing 60-75 deg away (5), and 6 m ahead with the nose 60-75 deg away (6). Files: `outputs/docking_sweeps/final_hint2/` and `outputs/docking_sweeps/final_blind/`.

**What to expect (measured offline on the assumed model, 2026-10-10).** Wide grid (60 starts: range 5/7/9 m x lateral -2..+2 m x heading -30..+30 deg, dock inside the field of view, plus random combinations) x 4 plants, real detector in the loop: **202/240 first-try passes**. From 8-10 m every start passes (84/84); 6.5-8 m 80/88; 5-6.5 m 38/68 (not enough room to line up from rest: there is no sway thruster, the vehicle steers by yawing while moving). Among passing runs: entry speed 0.29-0.31 m/s (limit 0.5), entry lateral/vertical error at most 12/5 cm (limit 15), entry heading at most 5.0 deg (limit 5), minimum wall clearance 4 cm (mean 20 cm), 25 s to dock on average. Over ROS with realistic odometry: 10/10 from 6-9 m. With odometry freezes during the final approach 2 of 4 runs failed. The numbers are for the ASSUMED vehicle model and camera look; the real simulator may differ.

**What to change when it misbehaves** (all in `terminal_docking.yaml`, each number is documented there with measured effect sizes): arrives crooked -> `guidance.gate_heading_deg`, `gains.yaw.kd`; wall contact on offset starts -> `guidance.lookahead_base_m` (smaller = steers onto the axis earlier), `speed.align_cruise_mps`; too fast at the mouth -> `speed.terminal_mps`, `speed.decel_mps2`; stops short or overshoots -> `guidance.stop_nose_s_m`, `speed.stop_gain_per_s`; keeps backing off -> `guidance.gate_s_m`, `guidance.gate_lateral_m`. Change live with the viewer's *Live gains* for the loop gains.

---

## 17. Tuning tools and how to use them

All of these are offline (no ROS needed unless said) and print results; none of them overwrites a shipped file (except `write_*` which are one-off dev tools). Run from `~/Research/MAVSIM/AUV_docking/control_code/tuning` (ROS sourced where the message types are needed).

| Tool | What it does |
|---|---|
| `python3 loop_tuner.py show` / `tune heave pitch` / `yaw` / `hover` / `sens` | Scores the hold-loop gains (heave, pitch, yaw, roll, speed) on four plants (nominal, heavy+late, light+fast, stress) with realistic odometry through the pose filter: overshoot, settling, thruster/fin chatter, disturbance rejection. `tune`/`yaw`/`hover` search better gains; `sens` measures the effect of changing each number (the numbers quoted in the yaml comments). |
| `python3 tune_docking.py --grid wide --vision perfect --rounds 4 [--also-detector-moderate] [--params a.b,c.d]` | Coordinate search of `terminal_docking.yaml` values against the docking grid. Prints the best values; apply them by hand. Takes 10-40 minutes. |
| `python3 sensitivity.py --grid wide --out sens.json` | Each docking number changed alone by x0.5 ... x1.5: how many of the 240 runs still pass. |
| `python3 ros_confirm.py [--only heave:hold,yaw:step] [--freeze 15]` | Over ROS (domain 71): all 12 `dof_testing` tests + 45 s of `station_keeping` on the fake vehicle with realistic odometry (about 10 minutes). |
| `python3 ros_dock_confirm.py` | Over ROS (domain 70): terminal_docking from 10 starts with the real detector, live judge (about 8 minutes). |
| `../../dock_detection_algo/tune_detection.py report` / `tune --n 900 --rounds 6` | Scores / tunes the detector thresholds on synthetic frames (clear and murky water, many looks of the lights): GOOD (4 lights all within 6 px), BAD LOCK (valid but wrong), MISS, by range band, on a training split, a hold-out and a fresh set. The result is only printed: copy the values into `dock_detection.yaml` (the shipped file documents the measured sensitivity of each). Needs ROS sourced. |

**Honest limits.** Everything is tuned on the ASSUMED offline vehicle model (`common/mako_geometry.yaml`: drag, thrust units, fin lift, lag) and an ASSUMED camera look: nothing was validated on the real simulator. The numbers in the yaml comments are for direction and relative size. After the first real runs: record them (`sim_viewer/run_record.sh`), compare with the model (`compare.py`), calibrate (`calibrate.py`), re-run these tools.

---

## 18. First run on the real sim: one command, plus the diagnostic tools (added 2026-10-10)

None of this has run on the real mavsim. The sim was not running when it was written. The scripts below only listen, except the checklist, which commands the vehicle.

### 18.1 The checklist as one script
```bash
cd ~/Research/MAVSIM/AUV_docking/control_code
./scripts/run_real_sim_checklist.sh                # stages 0-6, domain 42, stops at the FIRST problem
./scripts/run_real_sim_checklist.sh --from 3       # resume
./scripts/run_real_sim_checklist.sh --only 5       # one stage
./scripts/run_real_sim_checklist.sh --dock         # also stage 7 (detector check, then dock_test) after 0-6 passed
./scripts/run_real_sim_checklist.sh --no-imu       # skip stage 1 (needs a tilt)
./scripts/run_real_sim_checklist.sh --help
```
Before: sim playing, all six DOF active, vehicle at about 3 m away from the dock, `mavsim_teleop` stopped. Stages: 0 preflight (odometry alive and not frozen, who publishes `actuator_cmd`), 1 IMU vs odometry attitude sign, 2 heave step, 3 surge and pitch step, 4 yaw and roll step (confirms or breaks the fin assumptions), 5 holds, 6 station keeping loop by loop. The script asks you to **reset the vehicle in the sim UI** before every test (the roll/pitch/yaw steps leave it tilted and turned; offline the next test aborted on `roll 60 deg`). Logs go to `outputs/logs/real_sim_checklist_<time>/`.
Self-test against the offline fake vehicle (never domain 42): `ROS_DOMAIN_ID=77 ROS_LOCALHOST_ONLY=1 ./scripts/run_real_sim_checklist.sh --offline-self-test --no-imu` (with `python3 sim_offline/fake_vehicle.py --z 3 --realistic --odom-freeze-interval 0` running). Result 2026-10-10: stages 2-6 passed (all steps and holds PASS, station keeping ran in each loop configuration).

### 18.2 Single tools
```bash
python3 tuning/real_sim_preflight.py                      # is the sim up, odometry not frozen, rate, depth, who publishes actuator_cmd (exit 0 = READY)
python3 ../dock_detection_algo/check_imu_convention.py    # tilt the vehicle meanwhile; says SAME / FLIPPED / UNDECIDED for IMU pitch and roll vs odometry
python3 sim_viewer/plots/odom_freeze_monitor.py --minutes 10 --label "A baseline"   # logs each odometry freeze with load, top CPU and camera rates to outputs/odom_freeze_log.csv
python3 sim_offline/pendulum_analysis.py                  # hypothesis for the live pitch pendulum (CG-CB offset), prints the live experiment to run
```
If `check_imu_convention.py` says FLIPPED set `camera.imu_pitch_sign: -1` (and/or `imu_roll_sign`) in `dock_detection.yaml`: a wrong pitch sign makes the dock elevation wrong by twice the pitch.
Freeze experiment: run `odom_freeze_monitor.py` in four conditions and compare the freeze fraction: A nothing else, B plus the detector (`run_live.sh`), C plus a controller publishing at 20 Hz, D both. Freezes that follow B or C point at the bridge/camera path, equal freezes in all four at the simulator.

---

## 19. Task: missions (lawnmower, orbit, spiral, yo-yo, chained legs) (added 2026-10-10)

`control_code/mission/` flies a list of **legs** from one YAML. **Offline only so far**: the fake vehicle with realistic odometry; the 2.5 m turning circle, fin signs and thrust law are assumptions until the real-sim checklist (section 18) has run. Do sections 1 and 3 first for the real sim (or section 10 for the offline stack).

### 19.1 Run the shipped mission
```bash
cd ~/Research/MAVSIM/AUV_docking/control_code/mission
./run_mission.sh                           # reads mission.yaml; ROS_DOMAIN_ID defaults to 42 (the real sim): set a private one for the offline stack
./run_mission.sh --config my_mission.yaml  # another file (for example saved from the viewer's Config tab with "Save as new file")
```
The shipped mission (vehicle starts at 0, 0, 3 heading north): a **lawnmower** of 3 lanes (10 m long, 8 m apart, east-west, starting at x = 0, y = 5), an **orbit** (radius 4 m) after the last lane, a **yo-yo** (depth 2 <-> 4 m over 9 m), then **return home** (the start position) and hold. It needs about 40 m x 35 m of room and stays more than 6 m from the dock.
Log every 2 s: `lawnmower z=3.0 wp 3/16 12% eta 210s pos=(0.0,9.8,3.0) hd_err=+1.0deg ct=+0.12m u=0.98/1.00 m/s depth_err=+0.00`. At the start it prints the plan (segments, metres, ETA) and every warning of `validate_mission` (a corner inside the turning circle, a point outside the geofence or inside the dock keep-out, a missing leg parameter).

### 19.2 Legs (`legs:` in `mission.yaml`, flown in order)
| Leg | Parameters |
|---|---|
| `goto` | `x, y, z` |
| `lawnmower` | `origin [x, y]`, `heading_deg` (0 north, 90 east), `length_m`, `width_m`, `spacing_m`, `z` or `depths: [3, 4]` (stepped survey: the lanes are flown at each depth in turn), optional `leadin_m` (run-in, default 5 m), `arc_points` |
| `orbit` | `centre [x, y]`, `radius_m` (at least 2.75 m), `revs`, `clockwise`, `start_deg` (0 = start north of the centre, 180 = south), `z`, `points_per_rev` |
| `spiral` | `centre`, `r_start_m`, `r_end_m`, `pitch_m` (radius change per revolution = the gap between turns), `z` |
| `yoyo` | `from [x, y]`, `to [x, y]`, `z_top`, `z_bottom`, `cycles` |
| `hold` | `seconds` (thrust off, depth held, NO heading hold: the fins need flow) |
| `return_home` | none; target `mission.home` or the start position |
Any leg may add `acceptance_m`, `depth_acceptance_m`, `speed_mps`.

**Rules of thumb (the vehicle cannot strafe or pivot):**
- Lane spacing: at least **5 m** (two turning radii) for a plain U-turn; closer lanes are flown with a skip pattern (lanes 0, 2, 4, then 1, 3 ...) that needs 5 or more lanes, and the node says so. The U-turn bulges half a spacing PAST the lane ends: leave that room.
- **Chain legs the way the vehicle is heading.** Each leg should start roughly where, and in the direction, the previous one ended; a reversal needs a whole turning circle of room (the checks print "a 170 deg turn sweeps a 2.5 m circle that ..." when it would leave the geofence or reach the dock).
- `safety.geofence` (box) and `safety.dock_keepout_m` (circle around the dock at 10, 0): leaving the box or entering the circle **aborts the mission and the vehicle returns home** (it logs `MISSION ABORTED: ...`, the viewer shows `aborted`); an odometry gap longer than `safety.failsafe.odom_loss_s` does the same when the odometry returns. Set the keep-out to the closest approach you accept PLUS about 3 m, because the vehicle needs ~5 m to turn back (offline, 3 m let it reach the dock).
- `cross_track.enabled: true` flies straight legs as lines (lookahead `speed.lookahead_m` plus a trim from `gains.cross_track`); `false` aims at the next point only (drifts off the lane in a current).

### 19.3 Edit and preview in the viewer
Viewer (section 10.2) -> *Config* tab -> `mission`: the map shows the whole expanded path with the dock, geofence and keep-out; edit the legs in the table and form; the checks box lists problems; *Apply to next Start*, then start `mission` in the Controls tab. During the run the map shows the vehicle, trail and covered swath, the tab a progress bar and ETA, and the *Mission* plot tab the cross-track error and speed. A summary figure is saved when the run ends (`outputs/sim_viewer_runs/..._summary.png`).

### 19.4 Check it offline
```bash
cd ~/Research/MAVSIM/AUV_docking/control_code
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest mission -q                   # no ROS: paths, every leg type closed loop, cross-track, failsafes, negative controls
# over ROS on a PRIVATE domain: fake vehicle with realistic odometry + the mission node, judged from ground truth (PASS/FAIL)
source /opt/ros/humble/setup.bash; source ws/install/setup.bash; export ROS_DOMAIN_ID=77
python3 mission/ros_mission_confirm.py --preset quick                              # lawnmower + home, ~75 s
python3 mission/ros_mission_confirm.py --preset full --seconds 400 [--freezes]     # the shipped mission, ~4 min (--freezes lets the fake odometry freeze like the real sim)
```
Offline result 2026-10-10 (assumed model): shipped mission completes in ~240 s, ends 0.4 m from home, closest dock approach 6.8 m (ROS: `quick` 75 s PASS, `full` 234 s PASS without freezes, 287 s PASS with `--freezes`).

### 19.4b Fly every leg and save the trajectories (2026-10-11)
```bash
cd ~/Research/MAVSIM/AUV_docking/control_code/mission
python3 mission_showcase.py                       # lawnmower, orbit, spiral, yo-yo, go-to + hold, the chained shipped mission -> outputs/sim_viewer_runs/missions_showcase/index.html
python3 mission_showcase.py --only lawnmower --freezes 70 --out ../../outputs/tmp/try      # with odometry freezes (mean 70 s apart), another folder
python3 -m pytest test_mission_showcase.py -q    # the harness and the judge (a bad mission is judged FAILED)
```
Each mission flies closed loop with REALISTIC odometry (4.5 Hz, 0.25 s late, noisy, through the pose filter) and is judged on the simulated true path: finished, not aborted, never within the 6 m dock keep-out, inside the geofence, ends within 1.5 m of home, lane error (95th percentile of the straight lane parts) under 1.5 m. Per mission: `trajectory.csv` (all seven actuator commands; replay it in the viewer), `top_view.png`, `timeline.png` (cross-track, speed, depth, progress, fins), `summary.json`. Failing runs are saved too (`FAILED_*`). Finding: with realistic odometry the default run-in of 2 turning radii left the first lawnmower lane 1.9 m off; `leadin_m: 8` (now in `mission.yaml`) gives 0.65 m. Results: **without freezes 6/6 pass** (lawnmower 133 s, orbit 118 s, spiral 116 s, yo-yo 106 s, go-to + hold 66 s, the chained mission 243 s; closest to the dock 6.7 m, every run ends within 0.53 m of home, lane error p95 0.65 m). **With odometry freezes (5-25 s, about every 70 s) 3/6 pass** (orbit, spiral, yo-yo); the lawnmower, the go-to + hold and the chained mission drift outside the geofence during a freeze, abort, return home and hold there (1.5-1.7 m from home): the safety behaviour works but the mission is not completed; they are saved as `FAILED_*` in `outputs/sim_viewer_runs/missions_showcase_with_odometry_freezes/`.

### 19.5 If it misbehaves
| Symptom | Fix |
|---|---|
| `MISSION ABORTED: outside the geofence` / `inside the ... dock keep-out` | The vehicle swung wider than planned (a turn bigger than the room): widen the geofence, move the leg, or chain the legs so the turn is smaller. The vehicle goes home and holds. |
| It circles a point and never moves on | That point is inside the turning circle after the turn before it (`validate_mission` prints it at start-up). Move it, or give the leg a larger `acceptance_m`. |
| Lanes come out wavy | Lower `gains.cross_track.kp` / `ki`, or raise `speed.lookahead_m`. |
| It drifts off the lane in a current | Check `cross_track.enabled: true`; raise `gains.cross_track.ki` (the integral takes ~20 s to cancel a steady push). |
| Yo-yo does not reach its depths | Lengthen the line (the vehicle changes depth with its heave thrusters only) or raise `depth_acceptance_m` less / lower `speed_mps`. |

## 20. Task: side-scan sonars and the rugged sea floor (added 2026-10-11)

A simulated pair of **Omniscan 450 SS** side-scan sonars (30 deg off the nadir, 50 deg x 0.5 deg beams, 450 kHz) over a 300 x 300 m rugged sea floor, OFFLINE only (mavsim has no sonar). Physics and files: README section 5.11. All echo levels are assumptions.

**20.1 Watch it live** (private domain, three terminals after `export ROS_DOMAIN_ID=77 ROS_LOCALHOST_ONLY=1`, `source /opt/ros/humble/setup.bash`, `source ~/Research/MAVSIM/AUV_docking/control_code/ws/install/setup.bash`):
```bash
cd ~/Research/MAVSIM/AUV_docking/control_code
python3 sim_offline/fake_vehicle.py --realistic --x 0 --y 0 --z 3      # 1. the vehicle (use the viewer's Controls tab to push it, or start a controller)
./sim_viewer/run_sidescan_sim.sh                                          # 2. the sonars (pings, altimeter, status; listens for /Mako_01/sonar/cmd)
./sim_viewer/run_sim_viewer.sh --bottom-tab Sonar                         # 3. the viewer: the rugged floor in 3D, and the Sonar tab (Waterfall | Mosaic)
```
Start a mission (`mission/run_mission.sh`, or the Controls tab) and watch the waterfall scroll (port left, starboard right, newest on top) and the mosaic grow. **In the Sonar tab: Range [m] (2-150), Gain (auto or 0-7), Bins (200-1200) and *Apply to sonar*** change the sonars at once (same as `ros2 topic pub --once /Mako_01/sonar/cmd std_msgs/msg/String "{data: '{\"range_m\": 15, \"gain\": 5}'}"`); *freeze*, *Clear mosaic*, *Save pictures* (to `outputs/sim_viewer_shots/`). The status line shows altitude, ping rate and the mosaic coverage.
The viewer records the pings of every live run next to its CSV (`outputs/sim_viewer_runs/auto_<controller>_<time>_sidescan.npz`), so any live run can be replayed with its sonar data.

**20.2 Fly a lawnmower survey and save the data (done in one go, no ROS needed):**
```bash
cd ~/Research/MAVSIM/AUV_docking/control_code/mission
python3 sonar_survey.py                       # both modes: constant_depth and terrain_following, 4 lanes x 90 m, 14 m apart, 30 m range -> outputs/sim_viewer_runs/sonar_showcase/index.html (about 4 minutes)
python3 sonar_survey.py --only terrain_following --lanes 6 --spacing 12 --range 20 --gain 4 --altitude 5
```
Per mode: `trajectory.csv` + `sidescan.npz` (the data), `waterfall.png`, `mosaic.png`, `mosaic_vs_truth.png` (the sonar mosaic next to the TRUE backscatter map and the TRUE depth with the path), `summary.json` (coverage, gaps, min altitude, shadow fraction, objects). A run that does not finish, aborts, covers less than 80% of the area or comes closer than 1 m to the floor is saved as `FAILED_<mode>`. **Look at the data again:**
```bash
cd ~/Research/MAVSIM/AUV_docking/control_code/sim_viewer
./run_sim_viewer.sh --replay ../../outputs/sim_viewer_runs/sonar_showcase/terrain_following/trajectory.csv --bottom-tab Sonar --view top --speed 4
```
(or the History tab: `sonar_terrain_following.csv` and `sonar_constant_depth.csv` are listed there; the matching `_sidescan.npz` is found by name). The stored pings play back in step with the vehicle; the waterfall and the mosaic fill as it moves. The replay's 3D view shows the same rugged floor. Results (README 5.11): constant depth 100% coverage at 8-20 m altitude, terrain following 98% at 6 m, closest 1.9 m.

**20.3 Terrain following in a mission:** `mission.yaml` -> `terrain_follow: {enabled: true, altitude_m: 6.0, rate_mps: 0.8, max_depth_m: 20.0}` (the depth of the legs is then ignored; the node needs `/Mako_01/altimeter/range` from the sonar node). It reacts to the altimeter only: raise `rate_mps` for steep floors, keep `altitude_m` above the tallest boulder you expect. Without the sonar node (no altimeter) it does nothing and the legs' own depth is used.

**20.4 Change the terrain, the mount or the physics** (`sim_viewer.yaml`): `terrain:` (seed, size, depth, relief, boulders, trenches, objects), `sonar.sensor:` (`mount_from_nadir_deg` 30, `location_m`, beam widths, range, bins, gain, sound speed, absorption, looks). Another mount angle is a one-line change (60 deg = looking much more sideways: swath 0.0 to 12 H, shallower echoes). `look3d.use_terrain: false` returns the old small decorative floor in the 3D view (the sonar always uses `terrain:`).

**20.5 Tests:** `python3 -m pytest sim_viewer/tests/test_terrain.py sim_viewer/tests/test_synthetic_sidescan.py sim_viewer/tests/test_sidescan_mosaic_and_recording.py sim_viewer/tests/test_sonar_panel.py mission/test_sonar_survey.py -q` (about 1.5 min, no ROS); the node over ROS: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest sim_viewer/tests/test_sidescan_ros.py -q` (needs the workspace sourced; `interfaces` must contain `SideScan.msg`: after pulling this change run `cd ws && colcon build --packages-select interfaces` once).

**20.6 If it misbehaves:** Sonar tab empty and `no side-scan pings yet` -> the sonar node is not running on this domain (step 2 above), or the replay has no `sidescan.npz` next to its CSV. `ImportError: SideScan` -> rebuild `interfaces` (20.5). Waterfall mostly black -> the vehicle is too high for the range setting (swath is ~1.4 x altitude per side; lower the range or fly lower), or gain is 0. A numba crash on import (`coverage.types`) -> already worked around in `terrain.py`; without numba everything still runs but ~100x slower. Pings are slow with 1200 bins and 150 m range: lower the range.

**Viewer layout and redesign (2026-10-11, session 7).** One dark theme (`sim_viewer/app/theme.py`: Qt stylesheet + matplotlib style + shared colours). Window = 3D view (top left), a tab area (bottom left), a side panel (right, resizable splitter):
- bottom-left tabs: **Plots** (sub-tabs Vehicle, Dock detector, Mission, **Pose**, **Actuators** (thrusters, 4 fins with the 25 deg cap, fins-at-cap %, thruster effort), **Tracking** (depth / heading / speed / cross-track error, distance to waypoint, altitude), **Overview** (top view coloured by speed, depth profile, speed, odometry interval); the corner dropdown sets the history 15 s - 5 min), **Detection** (the ONLY place the camera feed is shown: camera + gauges), **Sonar** (sub-tabs *Overview* = narrow waterfall + BIG mosaic, *Mosaic*, *Waterfall*; buttons *Enlarge panel* (gives the tab 80% of the height) and *Pop out mosaic* (own big window)), **3D Trajectory** (`traj3d.py` + `traj3d_panel.py`, VTK offscreen like the main 3D view: flown path coloured by speed / depth / time / altitude, planned path with waypoints from the Config tab, the seabed from `terrain.py`, the side-scan swath footprint, *Load run...* a saved CSV to compare, views iso/top/side, Fit, Save picture; mouse: left = orbit, right/wheel = zoom, middle or Shift+left = pan).
- side panel: **Status** (colour-coded tiles: depth, speed, heading, odometry age, lights, mode + text; optional *Small camera preview* 320x240), **Controls** (collapsible sections: Offline closed-loop stack, Controllers, Live gains, Simulation), Config, Scenarios, History.
- window size = the yaml size but never more than 97% x 95% of the screen; `--size WxH`, `--side-tab`, `--bottom-tab` (any bottom tab or plot sub-tab name) for scripts/screenshots.
- *Start offline stack* also starts the side-scan node (checkbox *Side-scan sonars*); the Sonar tab is selected automatically when the first pings arrive (unless you clicked a bottom tab or passed `--bottom-tab`).
- Cause of the earlier "zoomed in, Controls cut off" window: after the stack started the sonar node, one long status text (the Sonar tab's note) set the window's minimum width above the screen width (a 1872 px window on a 1920 px screen); long labels now use an *Ignored* size policy, explicit minimum sizes pin the panels, and every side tab is in a scroll area.
- Waypoint spin (session 7): the shipped 8-point list (4-8 m depth steps, cruise 10 m/s) made the AUV orbit waypoint 3 for ever: it passed 0.5-2 m beside the point, could not turn back inside its ~2.5 m turning circle (no strafing), and the old rule also required the depth to be within 0.25 m. Now `mission.depth_gate: false` (reached on the horizontal distance; depth keeps settling) and `mission.pass_radius_m: 3.0` (also reached when it came within 3 m and is now moving away), default `cruise_mps` 1.5. Offline: the shipped list completes in ~250 s; the old rule + cruise 10 m/s never finishes (test with a negative control). NOT run on the real sim.
