# AUV Docking

ROS 2 software for **autonomous underwater vehicle (AUV) docking** in the **mavsim** simulator.

The vehicle (**Mako_01**, a torpedo-shaped AUV) has to find and approach a **funnel dock** marked by **four lights**. This
repository contains everything we have built towards that goal:

| Area | What it does | Folder |
|------|--------------|--------|
| **Actuator allocation** | Turns a wanted force/moment into thruster RPM and fin angles for Mako_01 | `control_code/common/` |
| **Per-DOF testing** | Tests and tunes surge, heave, pitch, yaw, roll and sway one at a time | `control_code/dof_testing/` |
| **Station keeping** | Hovers: holds depth, pitch, surge position and heading | `control_code/station_keeping/` |
| **Dock test controller** | Reads `DockAlign` and holds a standoff: match depth, square the heading, keep the dock in frame. Does not enter the funnel | `control_code/dock_test/` |
| **Basic controllers** | Depth hold and waypoint following | `control_code/depth_control/`, `control_code/waypoint_tracking/` |
| **Offline test vehicle** | A simple fake Mako_01 so the controllers can run without the simulator | `control_code/sim_offline/` |
| **Synthetic dock camera and desktop viewer** | Draws what the nose camera would see of the dock lights (so the real detector and `dock_test` run closed-loop without mavsim), and a PyQt5 window that shows the vehicle and dock in 3D with live plots | `control_code/sim_viewer/` |
| **ROS messages** | The `interfaces` package (`Actuator`, `DockAlign`, `DVL`, `WaveProbe`) | `control_code/ws/` |
| **Dock perception** | Finds the four lights in a camera image and publishes how far off-center the dock is | `dock_detection_algo/` |

**What changed on 2026-10-11:** the dock detector was made robust against false lights, glint, murky water and a hidden light and now has clean pop-up windows plus a separate `dock_align` window; `terminal_docking` searches for a dock it cannot see and backs out when there is no room to line up; every start pose from 2 to 10 m, -3..+3 m sideways and -75..+75 degrees is swept and the good runs are saved as trajectories (sections 5.4c, 5.10b, 9 items 20-22).

**What is still missing:** a final run into the funnel on the real simulator. `control_code/dock_test/` already reads `/Mako_01/dock_align` and
publishes actuator commands, but it stops at a standoff once depth and heading are matched. Perception and that controller
are still separate programs (run the detector, then the controller).

Primary stack: **ROS 2 Humble**, **Python 3.10**, **OpenCV**, **NumPy**, talking to a running **mavsim-bridge** Docker
container (`ROS_DOMAIN_ID=42`).

> **How to run things:** this README explains *what each file is*. The step-by-step run instructions for every task are in
> [`docs/execution.md`](docs/execution.md).

---

## Table of contents

1. [Quick start](#1-quick-start)
2. [Repository tree (every file)](#2-repository-tree-every-file)
3. [Conventions you must know](#3-conventions-you-must-know)
4. [The vehicle: Mako_01](#4-the-vehicle-mako_01)
5. [File-by-file reference](#5-file-by-file-reference)
   - [5.1 Root files](#51-root-files)
   - [5.2 `control_code/common/`](#52-control_codecommon--shared-building-blocks)
   - [5.3 `control_code/dof_testing/`](#53-control_codedof_testing--per-dof-test-and-tuning-harness)
   - [5.4 `control_code/station_keeping/`](#54-control_codestation_keeping--hover-hold)
   - [5.4b `control_code/dock_test/`](#54b-control_codedock_test--standoff-from-the-dock-lights)
   - [5.5 `control_code/sim_offline/`](#55-control_codesim_offline--fake-vehicle)
   - [5.6 `control_code/tests/`](#56-control_codetests--pytest-suite)
   - [5.7 `control_code/depth_control/`](#57-control_codedepth_control--depth-pid)
   - [5.8 `control_code/waypoint_tracking/`](#58-control_codewaypoint_tracking--waypoint-follower)
   - [5.8b `control_code/mission/`](#58b-control_codemission--lawnmower-orbit-spiral-yo-yo-and-chained-missions)
   - [5.9 `control_code/ws/`](#59-control_codews--ros-2-interfaces-package)
   - [5.10 `dock_detection_algo/`](#510-dock_detection_algo--dock-light-perception)
   - [5.10b Detector reliability, robustness and the pop-ups](#510b-detector-reliability-robustness-and-the-pop-ups-added-2026-10-11)
   - [5.11 Side-scan sonars and a rugged sea floor](#511-side-scan-sonars-omniscan-450-ss-and-a-rugged-sea-floor-added-2026-10-11)
6. [ROS topics and messages](#6-ros-topics-and-messages)
7. [How the pieces fit together (data flow)](#7-how-the-pieces-fit-together-data-flow)
8. [Testing](#8-testing)
9. [Known limits and unverified assumptions](#9-known-limits-and-unverified-assumptions)
10. [Docker / DDS note](#10-docker--dds-note)
11. [Suggested end-to-end docking flow](#11-suggested-end-to-end-docking-flow)
12. [License, authors, related pieces](#12-license-authors-related-pieces)
13. [Pitched-view alignment](#13-pitched-view-alignment)

---

## 1. Quick start

**Folder map:** `control_code/` (controllers and the offline simulator: `common/`, `dof_testing/`, `station_keeping/`, `dock_test/`, `terminal_docking_control/`, `waypoint_tracking/`, `mission/`, `depth_control/`, `tuning/`, `sim_offline/`, `sim_viewer/` (sub-folders `app/ data/ view3d/ plots/ sonar/ camera/ scripts/ tests/`), `scripts/`, `tests/`, `ws/`), `dock_detection_algo/` (the dock detector), `docs/` (`execution.md` = every command, `project_notes.md` = local project memory), `outputs/` (everything the tools write: `logs/`, `docking_sweeps/`, `detection_reports/`, `sim_viewer_runs/`, `tmp/`).

1. **Every terminal** that uses ROS needs the workspace sourced, otherwise `ros2 topic echo /Mako_01/actuator_cmd` fails with
   `The message type 'interfaces/msg/Actuator' is invalid`:
   ```bash
   export ROS_DOMAIN_ID=42
   source /opt/ros/humble/setup.bash
   source ~/Research/MAVSIM/AUV_docking/control_code/ws/install/setup.bash
   ```
2. Build the messages once if `control_code/ws/install/` does not exist:
   ```bash
   cd control_code/ws && colcon build --packages-select interfaces
   ```
3. Start the mavsim bridge and a Mako_01 session, then **stop the bridge's teleop node** (it publishes zero commands on
   `/Mako_01/actuator_cmd` and would fight our controllers). See `docs/execution.md` section 3.
4. Check the actuator directions before trusting any controller:
   ```bash
   cd control_code/dof_testing
   ./run_dof_testing.sh --dof heave --mode step
   ```
5. No simulator? Run everything against the fake vehicle: `docs/execution.md` section 10.
6. Run the unit tests (no ROS needed): `cd control_code && python3 -m pytest tests -q`.
7. The offline simulator window (no mavsim needed): `cd control_code/sim_viewer && ./run_sim_viewer.sh --demo`, or without `--demo` and press *Start offline stack* in the Controls tab (details: section 5.5b and `docs/execution.md`).

---

## 2. Repository tree (every file)

```
AUV_docking/
├── README.md                          this file
├── LICENSE                            MIT license (root project)
├── .gitignore                         files git must not track (build output, caches, local-only folders)
├── pytest.ini                         keeps the pytest cache inside outputs/tmp/ (no .pytest_cache at the root)
├── CLAUDE.md                          short local-only pointer file for the AI assistant (git-ignored)
├── docs/
│   ├── execution.md                   step-by-step run guide for every task (commands, expected output, troubleshooting)
│   └── project_notes.md               local-only project memory: state, verified vs assumed, session log, TODOs (git-ignored)
│
├── control_code/
│   ├── .gitignore                     ignores colcon output, caches inside control_code
│   │
│   ├── common/                        shared building blocks (used by dof_testing, station_keeping, sim_offline, tests)
│   │   ├── allocation.py              wrench -> thruster RPM + fin degrees (and the inverse)
│   │   ├── pid.py                     PID with integral clamp and derivative-on-measurement
│   │   ├── loops.py                   single-axis hold loops (heave, pitch, surge, speed, yaw, roll, cross-track)
│   │   ├── state.py                   vehicle state from odometry (position, Euler angles, body velocities)
│   │   ├── pose_filter.py             Kalman smoothing of the slow, late, noisy odometry (PoseFilter) + PoseTracker (stale/frozen detection)
│   │   ├── live_gains.py              change a controller's gains while it runs (/Mako_01/ctrl_gains)
│   │   └── mako_geometry.yaml         Mako_01 data (thrusters, fins, mass, buoyancy, drag) with assumptions flagged
│   │
│   ├── dof_testing/                   test and tune one degree of freedom at a time
│   │   ├── dof_testing.py             test engine + ROS node
│   │   ├── dof_testing.yaml           step sizes, setpoints, gains, tolerances, safety limits
│   │   └── run_dof_testing.sh         launcher (prints help when run with no arguments)
│   │
│   ├── station_keeping/               hover hold
│   │   ├── station_keeping_core.py    controller engine (pure Python, no ROS)
│   │   ├── station_keeping.py         ROS node around the engine, CSV logging
│   │   ├── station_keeping.yaml       every tunable, documented with effect and analogy
│   │   └── run_station_keeping.sh     launcher
│   │
│   ├── terminal_docking_control/      alignment with the dock and entry into the funnel (vision + odometry only)
│   │   ├── terminal_docking_core.py   dock EKF (DockEstimator), guidance state machine (TerminalDockingCore), no ROS
│   │   ├── terminal_docking.py        ROS node
│   │   ├── terminal_docking.yaml      every tunable, documented with measured effect sizes
│   │   ├── run_terminal_docking.sh    launcher
│   │   ├── docking_sim.py             in-process closed loop (vehicle + sensors + camera + REAL detector + controller) with the failure judge
│   │   ├── terminal_docking_eval.py   runs the grid of start poses x plants in parallel, prints a scoreboard
│   │   ├── docking_sweep.py           every start (range x lateral x heading, 330 starts): outcome map, per-run table (NEW 2026-10-11)
│   │   ├── showcase.py                saves good runs as trajectory.csv + top view + timeline + camera frames + index.html (NEW 2026-10-11)
│   │   ├── test_position_source.py    odometry vs dead-reckoned vehicle position: offset, drift, jump, no speed (NEW 2026-10-11)
│   │   ├── test_terminal_docking_core.py   unit tests + closed-loop docking
│   │   ├── test_recovery.py           search + back-out behaviour, closed loop, with negative controls (NEW 2026-10-11)
│   │   └── test_terminal_docking_ros.py    the whole stack over ROS (fake vehicle, camera, real detector, node)
│   │
│   ├── tuning/                        offline tuning tools (no ROS, no GUI)
│   │   ├── loop_tuner.py              heave/pitch/yaw/speed gains against four plants with realistic sensing
│   │   ├── tune_docking.py            coordinate search of terminal_docking.yaml against the docking grid
│   │   ├── sensitivity.py             one-at-a-time sensitivity (the numbers quoted in the yaml comments)
│   │   └── real_sim_preflight.py      READY / NOT READY check before the first real-sim run (odometry alive, not frozen, who publishes actuator_cmd)
│   │
│   ├── sim_offline/                   fake vehicle for testing without mavsim
│   │   ├── vehicle_model.py           6-DOF rigid-body model of Mako_01 (pure Python)
│   │   ├── sensors.py                 odometry sensor model: ~4.5 Hz, jitter, 0.25 s latency, noise, freezes (like the real sim)
│   │   ├── pendulum_analysis.py       hypothesis for the live pitch pendulum (CG-CB offset) and the experiment that would confirm it
│   │   └── fake_vehicle.py            ROS node: actuator_cmd in, odometry_sim out (--realistic = the sensor model above)
│   │
│   ├── sim_viewer/                    synthetic dock camera + desktop viewer (re-grouped into sub-folders 2026-10-11; modules still import each other by plain name)
│   │   ├── sim_viewer.yaml            camera/dock data from the vessel file, the assumed 'look' of the lights, terrain, sonar, viewer settings
│   │   ├── run_sim_viewer.sh, run_camera_sim.sh, run_sidescan_sim.sh, run_record.sh   thin wrappers (the real launchers are in scripts/)
│   │   ├── scripts/                   the four launchers (the window refuses nothing, the camera and sonar nodes refuse ROS domain 42)
│   │   ├── app/                       the window and its tabs
│   │   │   ├── viewer_app.py          the desktop window: 3D scene, bottom tabs (Plots, Detection, Sonar, 3D Trajectory), side tabs (Status, Controls, Config, Scenarios, History)
│   │   │   ├── theme.py               ONE dark colour theme: Qt stylesheet + matplotlib style + shared colours (NEW 2026-10-11)
│   │   │   ├── widgets.py             coloured KPI tiles of the Status tab (NEW)
│   │   │   ├── collapsible.py         fold-away sections of the Controls tab (NEW)
│   │   │   ├── controls_panel.py      'Controls' tab: simulation control, controller launcher, live gains, offline stack (fake vehicle, camera, detector, sonar)
│   │   │   ├── process_manager.py     starts/stops the controller and stack processes (SIGINT first)
│   │   │   ├── config_model.py, config_panel.py, waypoint_map.py, yaml_edit.py, gains_editor.py   'Config' tab: comment-preserving YAML editor with a map
│   │   │   ├── scenario_judge.py, scenario_panel.py   'Scenarios' tab: live docking pass/fail judge + batch scoreboard
│   │   │   ├── history.py, replay_panel.py   'History' tab (auto-recorded runs, replay, overlay, compare, delete) and the Replay tab
│   │   ├── data/                      where the numbers come from
│   │   │   ├── telemetry.py           thread-safe store the window reads (odometry, commands, DockAlign, image, sonar pings)
│   │   │   ├── ros_link.py            ROS source: topics -> telemetry (+ sim/sonar commands)
│   │   │   ├── demo_source.py         in-process source: scripted manoeuvres + sonar pings, no ROS needed
│   │   │   ├── recording.py, record_run.py   CSV format of a recorded run (+ SonarRecorder) and the ROS recorder
│   │   │   └── replay.py              loads recorder / dof_testing / station_keeping CSVs; ReplaySource re-renders the camera and plays stored sonar pings
│   │   ├── view3d/                    3D
│   │   │   ├── scene3d.py, environment.py, assets.py, sim_viewer_math.py   main 3D scene (VTK offscreen): vehicle and dock meshes, sea floor from terrain.py, trail
│   │   │   ├── traj3d.py, traj3d_panel.py   '3D Trajectory' tab: path coloured by speed/depth/time/altitude, planned vs flown, saved runs, sonar swath (NEW 2026-10-11)
│   │   │   └── hud.py                 heads-up display, minimap, detection overlay
│   │   ├── plots/                     plots and analysis
│   │   │   ├── plots.py, plot_pages.py   live plots: Vehicle, Dock detector, Mission, Pose, Actuators, Tracking, Overview (+ history dropdown)
│   │   │   ├── coverage.py, run_summary.py   swath coverage and the per-run summary figure
│   │   │   ├── compare.py, compare_runs.py, calibrate.py   real-vs-model replay, two-run overlay, least-squares model fit
│   │   │   └── odom_freeze_monitor.py logs live odometry freezes
│   │   ├── sonar/                     side-scan sonars and the sea floor
│   │   │   ├── terrain.py             the rugged 300 m sea floor shared by the 3D view, the sonar and the tests
│   │   │   ├── synthetic_sidescan.py  physics of the Omniscan 450 SS pair + altimeter
│   │   │   ├── sidescan_node.py       ROS node: SideScan pings, altimeter, run-time range/gain commands
│   │   │   └── sidescan_mosaic.py, sidescan_view.py, sonar_panel.py   mosaic, drawing and the Sonar tab (Overview / Mosaic / Waterfall, Enlarge, Pop out)
│   │   ├── camera/                    the synthetic nose camera and the detector tab
│   │   │   ├── synthetic_camera.py    pose -> 640x480 image of the four dock lights (pure Python)
│   │   │   ├── camera_node.py         ROS node: odometry_sim in, camera_03 image + IMU out
│   │   │   └── detection_panel.py     'Detection' tab: camera feed with the labelled lights, gauges, steering lamps (the ONLY place the feed is shown)
│   │   └── tests/                     camera geometry, detector on the frames, telemetry, 3D scene, demo, whole window headless, replay, sonar, redesign, ROS tests; yaml_fixture.py keeps the tests independent of your waypoint yaml
│   │
│   ├── tests/                         pytest suite
│   │   ├── conftest.py                adds the code folders to the import path
│   │   ├── test_allocation.py         allocation signs and fin mixing
│   │   ├── test_pid.py                PID behaviour
│   │   └── test_offline_loops.py      DOF tests and station keeping against the offline vehicle
│   │
│   ├── depth_control/                 closed-loop depth hold
│   │   ├── depth_control.py           ROS node
│   │   ├── depth_control.yaml         depth setpoint, PID gains, limits
│   │   └── run_depth_control.sh       launcher
│   │
│   ├── waypoint_tracking/             multi-waypoint following
│   │   ├── waypoint_tracking_core.py  WaypointTracker (no ROS): yaw moment + speed + heave loops from common/, corner feasibility check
│   │   ├── waypoint_tracking.py       ROS node
│   │   ├── waypoint_tracking.yaml     waypoints, speed, gains, limits (rebuilt 2026-10-10)
│   │   ├── test_waypoint_tracking_core.py  closed loop against the offline vehicle + negative control
│   │   └── run_waypoint_tracking.sh   launcher
│   │
│   ├── mission/                       chained missions: lawnmower, orbit, spiral, yo-yo, go-to, hold, return-home
│   │   ├── paths.py                   path generators + turning-circle / geofence / dock keep-out checks (pure Python)
│   │   ├── mission_core.py            MissionRunner: legs -> segments flown by WaypointTracker, cross-track lanes, geofence + return-home failsafe, progress/ETA
│   │   ├── mission.py                 ROS node
│   │   ├── mission.yaml               legs, speed, gains, safety (geofence, dock keep-out, failsafe), survey swath
│   │   ├── run_mission.sh             launcher
│   │   ├── ros_mission_confirm.py     fly a mission over ROS on a private domain against the fake vehicle (PASS/FAIL)
│   │   ├── test_paths.py              generators and checks, no ROS
│   │   └── test_mission_core.py       every leg type closed loop, lanes with and without cross-track, failsafes, negative controls
│   │
│   ├── dock_test/                     standoff controller from DockAlign
│   │   ├── dock_test_core.py          wrench from dock cues (pure Python, no ROS)
│   │   ├── dock_test.py               ROS node: odometry + DockAlign in, actuator_cmd out
│   │   ├── dock_test.yaml             gains, deadbands, flow surge, blind-search budget, safety limits
│   │   ├── run_dock_test.sh           launcher
│   │   ├── loop_sim.py                closed-loop pitch + heave harness
│   │   ├── standoff_sim.py            closed-loop heading-error / close-start harness (odometry sensor model, image noise)
│   │   ├── tune_standoff.py           scoring, grid and random search for the standoff parameters
│   │   ├── ros_standoff_confirm.py    the standoff behaviour over ROS on a private domain (PASS/FAIL from ground truth)
│   │   └── test_dock_test_core.py, test_standoff_sim.py     sign tests and closed-loop regressions, no ROS
│   │
│   └── ws/                            colcon workspace
│       └── src/interfaces/            ROS 2 message package
│           ├── package.xml            package manifest
│           ├── CMakeLists.txt         message generation rules
│           ├── LICENSE                Apache-2.0 (vendored from mavsim)
│           ├── README.md              notes on keeping the messages in sync with mavsim
│           └── msg/
│               ├── Actuator.msg       actuator command
│               ├── DockAlign.msg      dock alignment guidance (ours)
│               ├── DVL.msg            Doppler velocity log (mavsim)
│               └── WaveProbe.msg      wave elevation probe (mavsim)
│
└── dock_detection_algo/               dock-light perception
    ├── live_dock_lights.py            ROS node + OpenCV windows + DockAlign publisher
    ├── dock_light_mask.py             bloom mask + light-core finding
    ├── dock_geometry.py               labels, cross-ratio center, cross-track, spread
    ├── dock_acquire.py                yaw/pitch/surge search when fewer than 4 lights are trusted
    ├── dock_align_msg.py              builds the DockAlign message
    ├── dock_ring.py                   picks the 4 lights that FORM the dock ring out of up to 8 bright spots (NEW 2026-10-11)
    ├── dock_hud.py                    the pop-up drawings (camera window, mask window, align window), pure OpenCV (NEW 2026-10-11)
    ├── dock_align_view.py             separate window for the /dock_align topic: steering arrows, numbers, plots (NEW 2026-10-11)
    ├── reliability_study.py           one-axis-at-a-time reliability study of the detector, CSV + PNG report (NEW 2026-10-11)
    ├── compare_reports.py             before/after table of two studies
    ├── render_hud_samples.py          renders the pop-ups to PNG without a display
    ├── test_dock_ring.py, test_robustness.py, test_dock_hud.py, test_reliability_study.py   tests of the above
    ├── test_pitched_geometry.py       synthetic camera test of the pitched-view cues
    ├── dock_detection_config.py       loads dock_detection.yaml
    ├── dock_detection.yaml            all detection tunables, documented
    ├── dock_detection_algo_launch.py  optional ros2 launch wrapper
    ├── run_live.sh                    launcher for the live detector
    ├── echo_align.sh                  prints the /dock_align topic
    ├── check_camera.sh                checks that a camera topic is streaming
    ├── fastrtps_no_shm.xml            Fast-DDS UDP-only profile (Docker to host)
    └── requirements.txt               Python packages for the detector
```

Not tracked by git (ignored): `control_code/ws/build|install|log/`, `__pycache__/`, `.pytest_cache/`, `custom_auv/`, `Papers/`,
`Gate-Detection/`, all of `outputs/` (except `.gitkeep`), and the local notes files `CLAUDE.md` and `docs/project_notes.md`.
Runtime logs are written to `outputs/logs/dof_testing/` and `outputs/logs/station_keeping/` (CSV files).

---

### Where every file the tools write goes (2026-10-11)

**Everything lands in `AUV_docking/outputs/`; nothing is written to the Home directory.** `outputs/sim_viewer_runs/` (recorded runs, summaries, scoreboards, `docking_showcase/`), `outputs/sim_viewer_shots/`, `outputs/logs/dof_testing/`, `outputs/logs/station_keeping/`, `outputs/terminal_docking_logs/`, `outputs/logs/real_sim_checklist_<date>/`, `outputs/odom_freeze_log.csv`, `outputs/tmp/` (temporary configs and scratch). The helper is `control_code/common/outdirs.py` (`out_dir()`, `tmp_dir()`); a `~/...` path in an old config is mapped INTO `outputs/`, not to Home. Docking sweep results are in `outputs/docking_sweeps/` (and `docking_results.txt`), detector studies in `outputs/detection_reports/`. Curated galleries (docking, missions, sonar survey) are in `outputs/sim_viewer_runs/` with one `index.html` listing them; the History tab shows the CSVs in that folder. Scratch is safe to delete: `outputs/tmp/` is recreated on demand.

## 3. Conventions you must know

### Frames and signs
- **World frame: NED** (x North, y East, **z Down**). Depth is the z coordinate, **positive = deeper**.
- **Body frame:** x forward, y starboard, **z down**. Euler angles are intrinsic **ZYX** (roll, pitch, yaw), quaternion order x, y, z, w.
- **Wrench order everywhere: `[X, Y, Z, K, M, N]`**

  | Symbol | Meaning | Positive means |
  |---|---|---|
  | X | surge force | forward |
  | Y | sway force | to starboard |
  | Z | heave force | **down** |
  | K | roll moment | starboard side down |
  | M | pitch moment | **nose up** |
  | N | yaw moment | bow to starboard |

### Actuator units (what is sent on `/Mako_01/actuator_cmd`)
- Thrusters `th_XX` are commanded in **RPM**, fins `cs_XX` in **degrees**. **0 = stopped / centred.** (This is not PWM.)
- Hardware limits: thrusters ±2668 RPM, fins ±35°. Each controller applies its own softer caps (`rpm_cap`, `fin_deg_cap`).
- Names must match the vessel: `th_01` axial thruster, `th_02`/`th_03` heave thrusters, `cs_04`/`cs_06`/`cs_07`/`cs_08` fins.
  The bridge matches commands **by name**, so the order inside the message does not matter.

### The most important sign: the heave thrusters point UP
The vessel file (`Mako (1).mavsim`, formerly `Mako_01.mavsim`) gives both heave thrusters the orientation pitch = +90°, which points their thrust axis along body −z, i.e. **up**.
So **positive RPM = up, and diving needs NEGATIVE RPM**. The allocator handles this (a positive Z/dive demand becomes negative
heave RPM) and the older controllers use `heave_sign: -1.0` in their YAML.

### Image convention for `DockAlign`
Image x points right and y points down. `error_x_px > 0`: the dock is to the **right** of the image center.
`error_y_px > 0`: the dock is **below** the image center.

---

## 4. The vehicle: Mako_01

Values come from `Mako (1).mavsim` (the current vessel file, a zip archive; the data is in `config.json`, `agents[0]`). Items marked *assumption* are not in
that file and are collected in `control_code/common/mako_geometry.yaml`.

| Property | Value |
|---|---|
| Mass / buoyancy mass | 20.0 kg / 20.0 kg, so the vehicle is **neutrally buoyant** (the old vessel file had 10 kg / 10.74 kg = +7.25 N, so heave trims of 7.25 N are now wrong) |
| Centre of gravity vs buoyancy | identical in the file, so no righting moment is expected; the live sim nevertheless swung in pitch like a pendulum when all six DOFs were active (see section 9) |
| Damping (config) | only surge, sway, yaw (0.01); **none in heave, roll, pitch**, so the PID derivative terms must provide damping |
| Axial thruster `th_01` | x = −0.688 m, thrust along body x, propeller diameter 0.10 m |
| Heave thrusters `th_02`, `th_03` | x = +0.349 m and −0.349 m, thrust along body −z (up), diameter 0.06 m |
| Fins `cs_04`, `cs_06`, `cs_07`, `cs_08` | X configuration at the aft: `cs_04` at 45°, `cs_08` at 135°, `cs_07` at 225°, `cs_06` at −45° (**`cs_06`/`cs_08` are swapped compared with the old vessel file**; getting this wrong turns a yaw command into a pitch moment), ±35° travel, area 0.00473 m² |
| Thrust curve | symmetric forward and reverse |
| Start pose | NED (0, 0, 3) |
| Dock (`dock_02`) | NED (10, 0, 3), heading 180° |
| Cameras published | `camera_03` (the nose camera, 640×480, **60° vertical** field of view, so f = 415.7 px and a 75° horizontal FOV), `camera_04`, `camera_05` |

### Which degrees of freedom can be controlled?

| DOF | Actuators | Works at zero speed? |
|---|---|---|
| Surge | axial thruster (forward and reverse) | yes |
| Heave | both heave thrusters together | yes |
| Pitch | heave thrusters pushing against each other (arm ±0.349 m) | yes |
| Yaw | X-fins | **no**: fin force grows with speed², so it needs flow |
| Roll | X-fins | **no** (same reason) |
| Sway | none | only indirectly: yaw while moving forward |

---

## 5. File-by-file reference

### 5.1 Root files

| File | Role |
|------|------|
| `README.md` | This document. |
| `docs/execution.md` | The run guide: terminal setup, starting the simulator, and numbered steps with expected output for every task, plus troubleshooting. |
| `LICENSE` | MIT license for the root project (Copyright 2026 Samyak). |
| `.gitignore` | Tells git to skip `custom_auv/`, `Papers/`, `Gate-Detection/`, colcon output (`control_code/ws/build|install|log`), Python and pytest caches, **everything in `outputs/`** (only `outputs/.gitkeep` is tracked; regenerate galleries and sweeps with the scripts), `CLAUDE.md`, `docs/project_notes.md`, and editor/OS junk. |

### 5.2 `control_code/common/` — shared building blocks

The code that decides *which actuator does what*. Used by the test harness, station keeping, the offline vehicle and the tests.

| File | Role |
|------|------|
| `allocation.py` | (2026-10-10: `small_force_n` makes the force-to-RPM map linear below that thrust, which cuts thruster chatter from noise by 3-4x; fins give the same moment in REVERSE flow by reversing the deflection.) **`Allocator`** converts a wrench `[X,Y,Z,K,M,N]` into actuator commands, and back. Thrusters realise X, Z and M: it builds the thruster matrix from position and direction, inverts it (pseudo-inverse) and converts each force to RPM with `T = KT·ρ·D⁴·n²`. Fins realise K and N: it builds a 3×4 mixing matrix from each fin's lift direction and moment arm, divides by speed² (**gain scheduling**, because fin force grows with speed²), fades the fins out below `u_fin_off_mps` where they have no authority, and caps the result. Key methods: `allocate(wrench, u)`, `thrusters_from_wrench`, `fins_from_moments`, `force_to_rpm`, `thrust_n`, `fin_authority`, and the inverses `wrench_from_rpm` / `wrench_from_fins` (used by the offline vehicle). Also `eul_to_rotm` (ZYX rotation matrix) and `load_geometry`. |
| `pid.py` | **`Pid`**: proportional-integral-derivative controller with an integral clamp expressed in output units, a low-pass filter on the derivative, and **derivative-on-measurement** (pass the measured rate and no kick appears when the setpoint jumps). Also `clamp`. |
| `loops.py` | **`HoldLoops`** (since 2026-10-10 also: an output low-pass `lpf_tau_s` and a setpoint filter `sp_tau_s` per loop, and `set_gain()` for live tuning): the single-axis hold loops built on `Pid`: `heave` (depth → Z force, includes the buoyancy feed-forward), `pitch` (→ M moment), `surge` (position along an axis → X force), `speed` (forward speed → X force), `yaw` (heading → N moment), `roll` (→ K moment), `cross_track` (lateral error → heading offset, used to produce sway). Error is always `setpoint − measurement`. |
| `pose_filter.py` | **`PoseFilter`**: one small Kalman filter per axis (value + rate) turns the slow (~4.5 Hz), late (~0.25 s), noisy odometry into a smooth state at the control rate (predicts between samples, compensates the latency; measured: depth RMS 0.8 cm). **`PoseTracker`** wraps it and also detects stale or FROZEN odometry (the same sample repeating). Used by `terminal_docking`, `station_keeping`, `dof_testing` (`estimator:` in their yaml). |
| `live_gains.py` | Change a controller's gains while it runs: message format and `attach()` for `/Mako_01/ctrl_gains` (JSON `{controller, loop, key, value}`); the viewer's *Live gains* group sends them. |
| `state.py` | **`State`**: position (NED), Euler angles, body velocities `[u,v,w,p,q,r]`, built from an `Odometry` message (`from_odom`) or from the offline model (`from_model`). Also `quat_to_euler` and `wrap_pi`. |
| `mako_geometry.yaml` | Mako_01 data for the allocator and the offline model. Sections: `vehicle` (mass, buoyancy, gyration radii, linear and quadratic drag, added mass), `thrusters` (`th_01`, `th_02`, `th_03` position, orientation, diameter, RPM limits; `kt_fwd`, `kt_rev`; `rpm_to_rps`), `fins` (`cs_04`…`cs_08` position and orientation; travel, area, lift slope). Every value that is **not** in the vessel file is marked `ASSUMPTION` in a comment. |

### 5.3 `control_code/dof_testing/` — per-DOF test and tuning harness

Tests **one** degree of freedom at a time. Run with both arguments, or with none to see the help text.

```bash
./run_dof_testing.sh --dof <surge|heave|pitch|yaw|roll|sway> --mode <step|hold>
```

| File | Role |
|------|------|
| `dof_testing.py` | **`DofTest`** (pure Python engine) with phases *baseline* (2 s of zero command to measure drift) → *spinup* (yaw, roll and sway only: drive forward to ~1 m/s so the fins have flow) → *run* → done. `--mode step` is **open loop**: a fixed push on that DOF only, PASS/FAIL on whether the response goes in the expected direction (checks wiring and signs). `--mode hold` is **closed loop**: a PID drives the DOF to a setpoint, PASS/FAIL on the final error (checks and tunes gains). While one DOF is tested, depth and pitch (and heading for roll) are held still by background loops. A safety check aborts to neutral on depth, pitch or roll limits. `run_offline(...)` drives the same engine against the offline vehicle without ROS (used by the tests). The ROS node reads `/Mako_01/odometry_sim`, publishes `/Mako_01/actuator_cmd`, prints the result and writes a CSV. |
| `dof_testing.yaml` | Settings. Sections: `node`, `topics`, `limits` (RPM and fin caps, fin speed scheduling), `hold` (setpoints per DOF), `step` (push size and duration per DOF, minimum response for PASS), `fin_dofs` (cruise speed, spin-up timeout), `timing`, `tolerance` (PASS limits), `background` (keep depth/pitch, buoyancy feed-forward), `safety`, `gains` (PID gains for heave, pitch, surge, speed, yaw, roll, cross_track), `logging`. |
| `run_dof_testing.sh` | Sources ROS and the workspace and runs `dof_testing.py` with `dof_testing.yaml`. With no arguments (or `-h`) it prints the usage, the recommended order and a pre-flight checklist. |

Results: `PASS`, `FAIL` or `ABORT` (a safety limit tripped), plus a CSV in `outputs/logs/dof_testing/`.

### 5.4 `control_code/station_keeping/` — hover hold

Captures the pose when it starts and holds it: **depth** and **pitch** with the heave thrusters, **surge position** along the start
heading with the axial thruster (forward and reverse), **heading** with the fins. Sideways (sway) drift is **not** held (no sway
actuator), and heading is only held while there is flow over the fins (above about 0.3 m/s).

| File | Role |
|------|------|
| `station_keeping_core.py` | **`StationKeeper`** (pure Python, no ROS): `feed_capture` averages the first seconds of state and latches the hold point, `capture_command` holds the buoyancy trim while that happens, `update` runs the loops with dead-bands and slew limits and returns actuator commands plus a status dict, `safety_reason` checks depth, attitude, leash distance. `shrink` implements the dead-band. |
| `station_keeping.py` | ROS node around the engine: subscribes to odometry, publishes actuator commands, stops to neutral on a safety trip or stale odometry, logs a status line every 2 s and writes a CSV to `outputs/logs/station_keeping/` on exit (errors, commands, and since 2026-10-08 the full state `x y z roll_deg pitch_deg yaw_deg u v w p q r`, so the viewer can replay it). |
| `station_keeping.yaml` | **Every tunable**, each with a comment on what changing it does, a quantified example and an analogy. Sections: `node`, `topics`, `capture`, `enable` (switch loops on/off while commissioning), `gains`, `feedforward`, `deadband`, `slew`, `limits`, `safety`, `logging`. The overshoot and settling numbers in the comments were measured on the offline vehicle, so treat them as direction and relative size, not absolute values. |
| `run_station_keeping.sh` | Sources ROS and the workspace and runs `station_keeping.py` with the YAML. |

### 5.4b `control_code/dock_test/` — standoff from the dock lights

Reads `/Mako_01/dock_align` and publishes `/Mako_01/actuator_cmd`. It matches depth, squares the heading, and keeps the dock in frame, then holds. It does not drive into the funnel. Run `dock_detection_algo/run_live.sh` in another terminal, and stop teleop first.

| File | Role |
|------|------|
| `dock_test_core.py` | **`DockTestCore`** (no ROS). A fresh invalid detection follows `search_yaw_norm`, `search_pitch_norm` and `search_surge_norm`, with heave held at the buoyancy trim (0 N for the current neutral vessel). A valid detection heaves on `elevation_rad`, pitches on `error_y_px`, and yaws on `error_x_px` plus `lateral_px`. **Surge is a speed loop, not a fixed force:** it creeps at `speed.creep_mps` while the yaw error is outside the deadband, the allowed speed falls with the estimated distance (a stopping-distance envelope from `radius_px`) to zero at the hold distance, the axial thruster brakes actively (feed-forward plus P), it backs off if `radius_px` is above `too_close_radius_px`, and the target is zero at the standoff. A stale message returns a zero wrench. **Dropout safety** (found by the closed-loop sim): it remembers the last trusted dock distance and dead-reckons it with the measured speed; when the lights are lost it never surges forward faster than the stopping envelope for that distance, backs away if they vanish inside the too-close distance (`lost_close`), and ignores a detection whose distance disagrees with the remembered one (spurious blobs when the glow swamps the image). **Heading error at the hold distance (2026-10-10):** the envelope allows no speed there, so the fins have no flow and a leftover heading error could never be corrected. Now it BACKS AWAY at `speed.realign_mps` while the yaw loop turns it square (`mode=realign`; the allocator flips the fin sign for reverse flow), stops backing when square and still or `realign_max_back_m` beyond the standoff, and creeps forward again if still needed. Yaw gains were re-tuned for it (`yaw_nm_per_px` 0.03 -> 0.006, `yaw_kd` 2 -> 10: the old pair had a damping ratio of ~0.15), the speed decision uses a wider band than the yaw moment (`hold_extra_px`), and the vehicle is not 'square' while it still rotates (`settle_yaw_rate_rad_s`). Offline (assumed plants, 90 fresh runs): 38/90 end at rest vs 7/90 before. Over ROS with the real detector: 4/4 starts at 3 m with a 8-20 deg yaw error came to rest, closest 2.2-2.4 m. **Close start with the dock partly out of frame (fixed 2026-10-10, offline and over ROS only):** with the dock never seen with four lights (24 deg or more of heading error at 3 m) there is no distance memory, the detector's `search` asks for a forward surge, and the old node drove into the dock (3/3 ROS runs, closest 0.14 m). Now `speed.blind_max_forward_m` (0.25 m) caps the net forward travel in a search without a known distance; after that `mode=search_back` reverses at `backup_mps` until the whole dock is seen (`blind_back_m` of hysteresis). Offline with a forward search hint (`standoff_sim.py ... blind_surge=+1`, 3 plants x yaw 26/35/-30): closest approach 2.2-2.5 m on the nominal and light plants and 1.7 m on the heavy+late one; with `blind_max_forward_m: -1` (the old behaviour) every run goes through the dock plane. The vehicle does not turn square in that mode (the harness gives no yaw hint; the real detector does), it only stays out of the dock. Switch the heading recovery off with `speed.flow_min_mps: -1` and the blind budget off with `speed.blind_max_forward_m: -1`. **Loops retuned 2026-10-08** with `loop_sim.py` after the viewer showed a sustained pitch oscillation: a velocity-damping term on the heave loop (`elevation_kd_n_per_mps`), softer pitch gains (`pitch_nm_per_px` 0.008, `pitch_kd` 8) and **continuous deadbands** (only the part beyond the band counts, so a measurement flickering across the edge no longer makes steps). Result: no oscillation on four assumed plants, settled in 6-8 s. |
| `dock_test.py` | ROS node. Subscribes to odometry and `DockAlign`, allocates the wrench with `common/allocation.py`, publishes actuator RPM and fin degrees. Neutral on stale odometry, stale `DockAlign`, or a depth / pitch / roll trip. |
| `dock_test.yaml` | Gains, deadbands, the `speed` block (creep speed, brake limits, planned deceleration, stop margin), camera `vfov_deg` and dock radius for the distance estimate, the 200 px too-close radius, buoyancy feed-forward (0), and safety limits. The pitch and heave gains are documented with the measured effect of each change. |
| `run_dock_test.sh` | Sources ROS and the workspace and runs the node. |
| `loop_sim.py` | **Closed-loop pitch + heave harness** (no ROS, no GUI): the real `DockTestCore` and allocator against the full vehicle model with thruster lag, ~4.5 Hz odometry for the rates and image delay, the dock 2.4 m ahead seen through the synthetic camera. `python3 loop_sim.py` compares the old and shipped gains on four plants (nominal, heavy+late, light+fast, stress). Use it before changing `pitch_*` or `elevation_*` gains. The plant numbers are assumptions. |
| `standoff_sim.py`, `tune_standoff.py` | **Heading-error harness and search** (no ROS): the real core + allocator + vehicle model + the odometry sensor model (4.5 Hz, 0.25 s late, noise) + projected light centres with 1 px noise. `python3 standoff_sim.py` prints a table of yaw starts x plants (final yaw, closest approach, % of the last 15 s at rest); `tune_standoff.py` scores one candidate (`--set a.b=v`), a grid or a seeded random search over `RANGES`. |
| `ros_standoff_confirm.py` | The same behaviour over real ROS on a private domain (fake vehicle with realistic odometry + camera node + the real detector node + the dock_test node), judged from the fake vehicle's ground truth: at rest >= 50% of the last 20 s and closest approach > 2.0 m. |
| `test_dock_test_core.py`, `test_standoff_sim.py` | 33 tests with both files, no ROS: signs, speed loop and stopping envelope, closed-loop approaches, distance memory, deadband continuity, heave damping, regression tests for the pitch retune, the realign behaviour (backs away, latches, ends when square, can be switched off) and closed-loop regression with a negative control (`python3 -m pytest .`). |

### 5.4c `control_code/terminal_docking_control/` — align with the dock and dock

Takes the vehicle from "the dock is in the camera's view" to "nose 0.5 m inside the funnel, stopped". Inputs: `DockAlign` (the four light pixels) and `odometry_sim` only; the dock's world position is estimated and remembered. Details and commands: `docs/execution.md` section 16.

| File | Role |
|------|------|
| `terminal_docking_core.py` | No ROS. **`DockEstimator`**: extended Kalman filter for the dock pose in the world (ring-centre position + axis heading) from the four light pixels, with the vehicle pose at image time, outlier gating and re-initialisation; the estimate is remembered when the lights leave the image. **`TerminalDockingCore`**: **Vehicle position source (2026-10-11): `estimator.position_source: odometry | dead_reckoning`.** The dock position is always triangulated from the light pixels; `s, e, chi` are the vehicle position minus the dock position in one frame, so a constant odometry offset never mattered. `dead_reckoning` replaces the vehicle x, y by the integral of the filtered body speed and attitude since the start (no absolute position is used at all; depth still from the depth sensor; `recover.dock_hint_ned` is then relative to the start pose), which removes the dependence on odometry drift and jumps in the last ~2 m where the lights are out of view. It still needs speed and heading. The state machine `WAIT -> SEARCH -> APPROACH -> (gate) -> TERMINAL -> DOCKED`, `RETRY` (failed gate or a PLANNED back-out), `SAFE_STOP`; **new 2026-10-11:** `SEARCH` when the dock has never been seen (a slow circle, or with `recover.dock_hint_ned` a route to a staging point on the dock axis), and a feasibility check (`room_needed_m`: can a vehicle with a 2.4 m turning circle still get onto the axis, pointing along it, before the gate?) that backs the vehicle out along the axis BEFORE it commits to a start it cannot make; line-of-sight guidance onto the axis (no sway thruster: it yaws while moving), a speed profile with a stopping law, and open-loop braking with a dead-reckoned speed when the odometry is frozen. Uses `common/loops.py`, `common/pose_filter.py`. |
| `terminal_docking.py` | ROS node (odometry + `DockAlign` in, `actuator_cmd` + `ctrl_debug` out, CSV log in `outputs/terminal_docking_logs/`, live gains on `/ctrl_gains`, neutral on Ctrl-C and on every safety trip). |
| `terminal_docking.yaml` | Every tunable with its meaning, an analogy and the MEASURED effect (first-try passes on the 240-run grid when the value is changed alone by x0.5 ... x1.5). |
| `run_terminal_docking.sh` | Launcher. |
| `docking_sim.py` | In-process closed loop (offline vehicle + realistic odometry + synthetic camera + the real detector or projected light centres + the controller) and the **judge** with the agreed failure list (wall contact from the dock STL's funnel profile, bad entry, overshoot, retries, lost dock, timeout, depth/attitude, pinned thrusters, chatter, oscillation, stale data). |
| `terminal_docking_eval.py` | Runs the grid of starts x plants in parallel and prints the scoreboard (`--grid wide|moderate|smoke`, `--vision perfect|detector`, `--set a.b=value`). |
| `docking_sweep.py` | **Every start, not just easy ones** (2026-10-11): 6 ranges (2-10 m) x 5 lateral offsets (-3..+3 m) x 11 heading errors (-75..+75 deg) = 330 starts with the real detector on rendered frames. Each start is tagged by how much of the dock is in view at t=0 (IN_VIEW / PARTIAL / HIDDEN) and each run by outcome: DOCKED, RECOVERED (after a planned back-out), NOT_SEEN, COLLISION, FAIL. Writes `outputs/docking_sweeps/<tag>/results.csv`, `summary.txt` and `feasibility_map.png`. `--hint near` gives the search a rough dock position (see below), `--rerun results.csv` re-runs only the failures of an earlier sweep, `--set a.b=v` overrides a controller value. |
| `showcase.py` | Re-runs chosen starts and SAVES the runs for you to look at: `trajectory.csv` (replayable in the viewer's History tab), `top_view.png`, `timeline.png`, `frames.png` (camera pictures with the detector pop-up drawing), `summary.json`, plus `index.html`. |
| `test_terminal_docking_core.py`, `test_terminal_docking_ros.py` | 22 unit/closed-loop tests; the whole stack over ROS (fake vehicle, camera, real detector node, controller node). |
| `test_recovery.py` | 5 closed-loop tests of the search and the back-out, each with a negative control (`recover.enabled: false`). About 3 minutes. |
| `test_position_source.py` | 4 tests of `estimator.position_source` (2026-10-11): the dead reckoner integrates a known path; a constant odometry position offset (3 m, -2 m) changes nothing in either mode; a 5 cm/s odometry drift breaks the odometry mode (negative control) but not dead reckoning; with no speed information neither can finish. |

### 5.4d `control_code/tuning/` — offline tuning tools

| File | Role |
|------|------|
| `loop_tuner.py` | Single-loop tuning of heave, pitch, yaw, roll and speed against four plants (nominal / heavy+late / light+fast / stress) with realistic odometry through the pose filter; step, disturbance and chatter scores; `sens` measures the effect of each gain. |
| `tune_docking.py`, `sensitivity.py` | Coordinate search of `terminal_docking.yaml` against the docking grid (optionally also the real-detector grid); one-at-a-time sensitivity. |
| `ros_confirm.py`, `ros_dock_confirm.py` | The same controllers confirmed over real ROS on a private domain (12 dof tests + station keeping; docking from 10 starts with a live judge). |
| `write_gain_docs.py`, `write_docking_yaml.py` | Dev tools that wrote the measured numbers into the yaml comments. |
| `real_sim_preflight.py` (+ test) | First-real-sim preflight (2026-10-10): listens for 8 s and says READY / NOT READY: odometry present, rate, not frozen, depth, tilt, and who else publishes `actuator_cmd` (teleop). Never commands anything. |
| `make_docking_results.py`, `write_docking_yaml.py`, `write_gain_docs.py` | Run the wide docking grid and write the compact RESULTS summary (`outputs/docking_sweeps/docking_results.txt`), and regenerate the documented numbers in `terminal_docking.yaml` and the hold-gain yaml headers from the measurements (so the yaml comments never drift from the data). `test_loop_tuner.py`, `test_real_sim_preflight.py` test the tuners. |

Also new on 2026-10-10 for the first real-sim run: `control_code/scripts/run_real_sim_checklist.sh` (the whole checklist as one command, stops at the first problem, asks you to reset the vehicle between tests; `docs/execution.md` section 18), `dock_detection_algo/check_imu_convention.py` (IMU vs odometry roll/pitch sign) with `camera.imu_pitch_sign` / `imu_roll_sign` in `dock_detection.yaml`, `sim_viewer/plots/odom_freeze_monitor.py` (logs the odometry freezes with load and camera rates), `sim_offline/pendulum_analysis.py` (hypothesis for the live pitch pendulum).

### 5.5 `control_code/sim_offline/` — fake vehicle

A small physics model so controllers can be exercised without mavsim. It catches sign mistakes and gain-structure problems; it is
**not** the real simulator (see section 9).

| File | Role |
|------|------|
| `vehicle_model.py` | **`VehicleModel`** (pure Python): 6-DOF rigid body with mass and inertia, net buoyancy from the vessel data (0 N for the current vessel), linear and quadratic drag, simple added mass, thruster thrust ∝ RPM² with a 0.2 s lag, fin force ∝ speed² with a rate limit, and an external-wrench input `ext` for pushes and currents. Methods: `set_command`, `step`, `quaternion`. |
| `pendulum_analysis.py` | Reads CG/CB from the `.mavsim`, fits the CG-CB offset that reproduces the live pitch swing (7..85 deg, 11 s: about 6.5 mm) and prints the live experiment that would confirm it. `VehicleModel` now applies `cg_minus_cb_m` as a real righting moment (it was ignored before). |
| `sensors.py` | **`OdometrySensor`** (pure Python): makes the fake odometry look like the real sim's: ~4.5 Hz with jitter, 0.25 s latency, noise, and FREEZES (all values repeated for 5-25 s, as seen on 2026-10-01). `fake_vehicle.py --realistic` uses it; the status topic also carries the ground truth pose for the viewer's judge. |
| `sim_control.py` | **`SimControl`** (pure Python): the pause / step / reset / push / time-scale commands (JSON) used by `fake_vehicle.py` and by the viewer's demo. |
| `fake_vehicle.py` | ROS node around the model; also listens on `/Mako_01/sim/cmd` and publishes `/Mako_01/sim/status` (see `sim_control.py`). Subscribes to `/Mako_01/actuator_cmd`, steps the model at 100 Hz and publishes `/Mako_01/odometry_sim`. A command older than 1 s counts as zero (like the real bridge). Options: `--x --y --z --roll --pitch --yaw --rate --cmd-timeout --vessel`. Run it in its own ROS domain (77) so it cannot touch the real bridge. |

### 5.5b `control_code/sim_viewer/` — synthetic dock camera (closed loop without mavsim)

Lets the **real** detector (`dock_detection_algo/`) and `dock_test` run against the fake vehicle: `camera_node.py` reads `/Mako_01/odometry_sim`,
draws the four dock lights as the nose camera would see them, and publishes `/Mako_01/camera_03/image/compressed` and `/Mako_01/imu_01/data`.
The camera mounting, 640×480, 60° **vertical** field of view, dock pose, light positions, lumens and beam angle are copied from the vessel file.
**How a lamp looks is assumed** (glow size, brightness versus distance, water tint, noise): the real renderer is not available here, so tune
`look:` in `sim_viewer.yaml` against a few recorded real frames before trusting range limits. The two side lights are 10× dimmer than top and
bottom in the vessel file, so with the shipped look the detector finds all four from 2.5 to 8 m and only the bright pair beyond ~10 m.

| File | Role |
|------|------|
| `detection_panel.py` | **Detection tab** (2026-10-11): in the bottom-left tab bar next to Vehicle / Dock detector / Mission. Shows, live or in a replay, what the detector sees: the camera picture with labelled lights, banner, range / bearing / elevation / view-angle gauges, the steering lamps (YAW L/R, UP/DOWN) and a top-down map; *Detach* opens the big camera + dock_align windows, *Save picture* writes to `outputs/sim_viewer_shots/`. Drawing code: `dock_detection_algo/dock_hud.py` (`draw_compact_view`, `info_from_align_dict`). `viewer_app.py --bottom-tab Detection` opens it first. Tests: `tests/test_detection_panel.py`. |
| `synthetic_camera.py` | **`SyntheticCamera`** (no ROS): `project` (sim convention: `R = Rz·Ry·Rx`, camera looks down −Z, `v` flipped), `brightness` (lumens/d², water attenuation, beam pattern), `render` (core disk + bloom halo + background + noise), `truth` (ground-truth pixels for tests), `encode_jpeg`. **Image degradations** (2026-10-11, all OFF by default, `sim_viewer.yaml` `stress:`): Gaussian and motion blur, backscatter veil, bubbles, marine snow, false bright lights, surface glint, reflections, one light hidden (`_apply_stress`). |
| `camera_node.py` | ROS node at `node.rate_hz` (15): odometry → image + IMU (orientation and angular rate from the same odometry). |
| `sim_viewer.yaml` | `node`, `topics`, `camera` (exact), `dock` (exact), `look` (assumed, documented with effect sizes). |
| `run_camera_sim.sh` | Launcher; refuses ROS domain 42 (the real bridge) unless `ALLOW_DOMAIN_42=1`. |
| `app/viewer_app.py` | **The desktop window** (PyQt5, dark theme). Left: the 3D scene over a bottom tab area: **Plots** (Vehicle, Dock detector, Mission, Pose, Actuators, Tracking, Overview; history dropdown 15 s - 5 min), **Detection** (the only place the camera feed is shown), **Sonar**, **3D Trajectory**. Right (resizable): **Status** (colour-coded tiles for depth, speed, heading, odometry age, lights, mode; view selector `follow`, `free`, `top`, `side`, `dock`, `onboard`; buttons; text status; optional 320x240 camera preview), **Controls** (collapsible sections), Config, Scenarios, History. The window never opens larger than 97% x 95% of the screen (`--size WxH` to override; `--side-tab`, `--bottom-tab` choose the first tabs). Mouse in the 3D scenes: left drag orbit, middle drag / Shift+left pan, right drag or wheel zoom. `--demo` runs an in-process scripted demo (with sonar pings) and needs no ROS. |
| `app/theme.py`, `app/widgets.py`, `app/collapsible.py` | **The look (2026-10-11)**: one dark stylesheet + matplotlib style and shared accent colours (`theme.py`), the coloured KPI tiles of the Status tab (`widgets.py`), the fold-away sections of the Controls tab (`collapsible.py`). Long labels use an *Ignored* size policy and explicit minimum sizes pin the panels, so a long status text can no longer widen the window beyond the screen (the cause of the earlier 'zoomed-in, Controls cut off' window). |
| `plots/plot_pages.py` | The extra live-plot pages: **Pose** (N, E, depth, roll/pitch, heading, body velocities), **Actuators** (thrusters, the four fins with the 25 deg cap, fins-at-cap %, thruster effort), **Tracking** (depth / heading / speed / cross-track error, distance to the waypoint, altitude above the seabed), **Overview** (top view coloured by speed, depth profile, speed, odometry interval). Each derives its curves from the telemetry window, so it works live and in a replay. |
| `view3d/traj3d.py`, `view3d/traj3d_panel.py` | **3D Trajectory tab** (VTK offscreen, like the main 3D view; matplotlib's `Axes3D` does not import in this environment): the flown path coloured by speed / depth / time / altitude above the seabed, the planned path with waypoint spheres (from the Config tab), the sea floor from `terrain.py`, the side-scan swath footprint, a second saved run loaded with *Load run...*, views iso / top / side, Fit, Save picture. |
| `scene3d.py` | **`Scene3D`** (pure VTK, no Qt): the real Mako and dock STL meshes (vehicle decimated 90%), the four ring lights, the water surface, a reference grid, a trail, one arrow per thruster (the thrust force direction, scaled by RPM) and the nose camera's field-of-view pyramid. World frame NED, `up = (0,0,-1)`. Renders offscreen to a numpy image; it needs a `DISPLAY` but opens no window. |
| `plots/plots.py` | **`LivePlots`**: matplotlib in Qt. Vehicle tab: depth, attitude, speed and rates, thruster commands, fin commands, top-view path. Dock tab: `error_x/y`, elevation and side cue, radius, lights found and `valid`. **Mission tab** (2026-10-10): cross-track error, heading error and yaw rate, speed vs setpoint, depth error, progress and ETA, and the path against the plan (the plan comes from the Config tab, dashed; it is also drawn on the Vehicle tab's path plot). |
| `telemetry.py` | **`Telemetry`**: thread-safe store. Uses arrival time (the real odometry stamps are 0). `pose_now` moves the pose forward with its own velocity (at most `extrapolate_s`) so the view is smooth at the real sim's ~4 Hz. |
| `ros_link.py` | **`RosLink`** (a plain thread): odometry, `actuator_cmd`, `dock_align`, camera image -> `Telemetry`. |
| `demo_source.py` | **`DemoSource`**: the offline vehicle flies scripted manoeuvres towards the dock with the synthetic camera rendering its view; restarts at 2.4 m. Not a controller test. |
| `assets.py` | Loads the STL meshes from the vessel archive and decimates them; falls back to simple shapes if the archive is missing. |
| `sim_viewer_math.py` | `body_to_ned`, `pose_matrix`, `rotation_between`. |
| `controls_panel.py` | **The Controls tab.** Simulation: pause/resume, step 0.5 s, time scale, reset pose, push presets (sideways, down, up, forward, pitch/yaw kicks) and custom wrenches. Controllers: pick `dof_testing` (dof and mode), `station_keeping`, `terminal_docking`, `dock_test`, `depth_control`, `waypoint_tracking` or `mission`, edit their gains, Start/Stop (a Start uses the values edited in the Config tab, if any). One-click **offline stack** (fake vehicle at the reset pose, synthetic camera, real detector without windows). Simulation controls are enabled only when a fake vehicle answers; starting a controller on ROS domain 42 (the real mavsim) asks for confirmation, and the offline stack refuses to start there. |
| `process_manager.py` | **`ProcessManager`** (QProcess): starts helpers, shows their output, stops with SIGINT first (controllers publish zeros on Ctrl-C) then kills after a grace period. |
| `gains_editor.py` | Edit a controller's YAML numbers in a tree dialog. Edits go to a temporary copy passed with `--config`; the original file is never changed. |
| `environment.py` | Underwater look: procedural sea floor (sand, ripples, rock crests; seeded), rippling surface, sun shafts, drifting particles, cones of light from the dock lights, vehicle shadow. Decoration only; `viewer.look3d` in `sim_viewer.yaml`. |
| `hud.py` | Heads-up display, top-view minimap and the detected-light overlay on the camera pane. |
| `scenario_judge.py`, `scenario_panel.py` | The *Scenarios* tab: a live docking scenario with a PASS/FAIL judge (ground truth from the fake vehicle) and a batch scoreboard. |
| `history.py`, `compare_runs.py` | The *History* tab: every controller run is auto-recorded; replay, overlay with the model, compare two runs, delete. |
| `yaml_edit.py` | Edit one gain in a controller yaml in place keeping every comment (the viewer's *Save as new default*). |
| `config_panel.py`, `config_model.py`, `waypoint_map.py` | **The Config tab** (2026-10-10): edit any controller YAML from the viewer. A top-view **map** (north up) shows the waypoints or the whole expanded mission with the dock, geofence, dock keep-out, the vehicle and its trail, the sensor swath (planned and covered, with a coverage percentage) and the corners the vehicle cannot make (red, with their turning circle): click to add a waypoint, drag to move it, right click to delete it; with *click sets selected leg position* a click sets the origin / centre / from point of the selected mission leg. A **table** lists the waypoints (x, y, z) or the mission legs (type, parameters) with add / remove / up / down and a form for the selected leg (numbers, switches, lists, text). A **tree** edits every other parameter (numbers, switches AND text) next to its file value, with a filter. The controller's own **checks** (turning circle, geofence, dock keep-out, missing leg parameters) are shown as you edit. *Apply to next Start* (on by default) hands the edited copy to the Controls tab, which starts the controller with a temporary file (the YAML is untouched); *Save to file* asks, copies the old file to `<file>.bak`, and writes only the changed text so **every comment stays** (the result is parsed again and must equal the edited values, otherwise nothing is written); *Save as new file* leaves the original alone; *Revert* and *Preview file text*. While a mission runs the tab shows the leg, progress bar, ETA and cross-track error. `config_model.ConfigDoc` is plain Python (tested without Qt). |
| `coverage.py`, `run_summary.py` | Swath coverage on a grid (share of the planned swath that the trail covered) and the **per-run summary**: `python3 plots/run_summary.py RUN.csv --mission mission.yaml --swath 3` prints path error (RMS, 95 %, max), depth error, speed, coverage and actuator use and saves a figure (path coloured by its error, error / speed / depth / thrusters / fins over time). The viewer writes `<run>_summary.png` next to every finished `mission` or `waypoint_tracking` recording and shows the numbers in the Status tab. |
| `recording.py` | **`CsvRecorder`** (no ROS) and the column list of a recorded run: `t, x, y, z(depth), roll_deg, pitch_deg, yaw_deg, u, v, w, p, q, r`, the seven actuator commands (`th_01 th_02 th_03` RPM, `cs_04 cs_06 cs_07 cs_08` deg, held until the next command), `cmd_age`, and the last `dock_align` values. `# key: value` header lines carry the label, vessel and date. |
| `record_run.py` | ROS recorder (only subscribes, so it is safe on the real sim): one row per odometry sample. `--frames N --frames-every S` also saves camera_03 JPEGs with the pose at that moment (to tune `look:` against the real renderer). Default file `outputs/sim_viewer_runs/run_<time>.csv`. |
| `replay.py` | **`load_run`** reads the recorder CSV and the CSVs of `dof_testing` and `station_keeping` (the latter needs the pose columns added 2026-10-08; older logs are refused with an explanation). Unwraps yaw, drops NaN and duplicate-time rows. **`Run.at(t)`** interpolates the state and holds the commands. **`ReplaySource`** plays a run into the viewer (same interface as the live sources): play, pause, seek, speed, loop. |
| `replay_panel.py` | The viewer's **Replay** tab. |
| `compare.py` | Replays a recording's actuator commands through `VehicleModel` and prints RMS, bias and final error per channel. `--mode free` (one run from the first sample: shows accumulated drift) or `segments` (restart from the recorded state every N s: tests the short-term dynamics). `--set NAME=VALUE` overrides a model parameter; `--out` writes the model prediction as a CSV; `--plot` writes an overlay PNG. The viewer's `--overlay` draws the same prediction as a translucent ghost vehicle and dashed plot lines. |
| `calibrate.py` | Least-squares fit (scipy) of `rpm_to_rps`, thruster lag, quadratic drags, net buoyancy, fin lift slope and added mass to one or more recordings, segment by segment. Reports a standard error and a status per parameter (`ok`, `weak` = entangled or noisy, `NOT IDENTIFIABLE`) and prints the `mako_geometry.yaml` lines it would change. **It never edits any file.** |
| `tests/` | `conftest.py`, `test_synthetic_camera.py` (22), `test_synthetic_stress.py` (11), `test_camera_feed_ros.py` (camera feed + detector over ROS on domain 78), `test_replay_camera.py` (3), `test_detection_panel.py` (4), `test_telemetry.py`, `test_plots.py`, `test_scene3d.py`, `test_gains_editor.py`, `test_config_model.py`, `test_config_panel.py`, `test_coverage.py`, `test_run_summary.py`, `test_process_manager.py`, `test_controls_panel.py`, `test_demo_and_viewer.py` (includes a headless run of the whole window), `test_replay_compare.py` (loaders, replay source, compare, calibration recovery), `test_replay_viewer.py` (model lines, ghost, Replay tab, whole window in replay mode), and ROS tests that need the workspace sourced: `test_ros_link.py`, `test_fake_vehicle_sim_control.py` (a real `fake_vehicle` process driven over ROS), `test_ctrl_debug_ros.py`, `test_record_run_ros.py` (recorder against a real fake vehicle and camera node, then compare), `test_mission_ui_ros.py` (edit a mission in the Config tab, start it from the Controls tab, watch it fly over ROS on domain 79). |

Planned for this folder: log replay and a real-versus-model overlay. The plots draw controller setpoints as dashed lines when the controller publishes `ctrl_debug`.

### 5.6 `control_code/tests/` — pytest suite

The whole suite (command in section 8) gives **400 passed + 21 skipped** without ROS and **447 passed** with ROS (2026-10-11; 369 before the detector, sweep and showcase tests) and the workspace sourced (`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`; the skipped ones need ROS). Run it in the foreground: started as a background job (SIGINT ignored) two process-manager tests fail. A failed ROS test can leave a `fake_vehicle` / `camera_node` on its private domain (76 for the recorder test): stop strays with `kill -INT <pid>`, otherwise the next run sees two odometry streams.

| File | Role |
|------|------|
| `conftest.py` | Adds `common/`, `sim_offline/`, `dof_testing/` and `station_keeping/` to the import path. |
| `test_allocation.py` | Heave thruster axis points up; dive needs negative RPM on both; nose-up moment drives the forward heave thruster up; surge uses only the axial thruster and can reverse; unreachable DOFs are ignored by the thrusters; RPM ↔ force round trip and cap; fin roll/pitch/yaw modes are orthogonal; fin moment round trip; fin force scales with speed² and fades out without flow; fin cap. |
| `test_pid.py` | Proportional term and output clamp; integral clamp; derivative-on-measurement (no setpoint kick); reset. |
| `test_offline_loops.py` | Offline vehicle floats up without thrust and dives with negative heave RPM; **every DOF passes in both modes** (12 cases); a deliberately wrong-sign actuator is caught by the step test; station keeping holds still water, rejects a vertical push and a current, shows that a low RPM cap cannot hold depth, shows heading is not held without flow, dead-band shrink, safety leash and depth limits, capture-window buoyancy trim. |
| `test_outdirs.py`, `test_pose_filter.py`, `test_righting_moment.py`, `test_sensors.py`, `test_sim_control.py` | The outputs rule (nothing lands in Home or /tmp; a `~/...` path maps into `outputs/`), the Kalman pose filter, the CG-CB righting moment of the offline model, the odometry sensor model (rate, latency, noise, freezes) and the pause/step/reset/push control of the fake vehicle. |

### 5.7 `control_code/depth_control/` — depth PID

Holds an absolute depth with a PID on the heave thrusters, then zeroes the actuators and exits after the depth has stayed within the
tolerance for the settle time. *Older controller: it has not yet been run on the real simulator since the switch to RPM.*

| File | Role |
|------|------|
| `depth_control.py` | ROS node: subscribes to `/Mako_01/odometry_sim`, PID on `setpoint − z`, publishes heave RPM (`heave_sign = −1` so a dive is negative RPM) on `/Mako_01/actuator_cmd`. Surge and fins stay at zero. Uses the shared `common/pid.py` (its private copy was removed 2026-10-10). |
| `depth_control.yaml` | Sections: `node`, `topics`, `actuators` (names and roles), `depth` (`setpoint_m`), `pid` (gains, `i_max`, derivative filter), `limits` (`rpm_cap`, `fin_deg_cap`, `heave_sign`, hold values), `behaviour` (settle tolerance and time, wait for odometry, zero on exit), `logging`. |
| `run_depth_control.sh` | Sources ROS and the workspace and runs the node with the YAML. |

### 5.8 `control_code/waypoint_tracking/` — waypoint follower

Steers through a list of NED waypoints. **Rebuilt on `common/` on 2026-10-10** after the first version spun the vehicle after its first waypoint offline (see below). *Not yet run on the real simulator.*

| File | Role |
|------|------|
| `waypoint_tracking_core.py` | **`WaypointTracker`** (no ROS): for the current waypoint a yaw-moment loop (`common/loops.py`) steers; the allocator turns the moment into fin degrees (1/speed^2 schedule, sign mix from the vessel geometry); a speed loop keeps the fins in flow (never below `speed.flow_mps` between waypoints, turning at the flow speed when the heading is far off) and slows with a stopping law towards the waypoint (stops only on the last one); depth through the heave loop. `unreachable_corners()` finds corners inside the vehicle's 2.5 m turning circle (it would orbit them; `tol_m` forgives the acceptance radius, so dense arcs pass). **Since 2026-10-10 (for the mission controller):** `set_path()` swaps the waypoint list without touching the loops (no jump in the filters at a leg change), a path can be flown as a LINE (aim at a carrot `speed.lookahead_m` ahead on the previous-point -> waypoint line, plus the optional `gains.cross_track` trim), `pass_by` counts a waypoint as reached when the vehicle passes its perpendicular close by, and per-leg acceptance / depth acceptance / cruise speed overrides. The default behaviour (aim at the next waypoint) is unchanged. |
| `waypoint_tracking.py` | ROS node around the core: odometry through the pose filter, neutral on frozen/stale odometry, safety trips on depth/pitch/roll, warns at start-up about unreachable corners, publishes `ctrl_debug` for the viewer. |
| `waypoint_tracking.yaml` | `node`, `topics`, `waypoints` (the shipped list is the user's 8-point route with depth steps of 4-8 m; edit it in the Config tab), `mission` (acceptance radius, **`depth_gate: false`** = a waypoint is reached on the horizontal distance and the depth keeps settling, **`pass_radius_m: 3.0`** = also reached when it is within 3 m and the waypoint is behind the vehicle), `speed` (`cruise_mps` 1.5), `speed`, `gains` (same loops and numbers as station keeping), `feedforward`, `limits`, `estimator`, `safety`, `logging`. |
| `test_waypoint_tracking_core.py` | 13 tests, no ROS (own 5x8 m rectangle for the unit tests; the shipped 8-point list is tested separately, with a negative control that reproduces the orbit of the old rule): the shipped mission completes closed-loop against the offline vehicle, waypoints visited in order, the corner check (and its tolerance), a vehicle really orbits an unreachable corner, the target-speed rules, a negative control showing the OLD fin signs turn the vehicle the wrong way, `set_path` keeps the loops, line mode holds the lane under a sideways push where bearing mode drifts off, and the pass-by rule. |
| `run_waypoint_tracking.sh` | Sources ROS and the workspace and runs the node with the YAML. |

What was wrong with the first version (found by running it against the fake vehicle on 2026-10-10; the PID numbers themselves were unchanged by the Pid migration): its `fin_yaw_signs: [1, 1, -1, -1]` are the opposite of the yaw mix the allocator derives from the vessel file for (cs_04, cs_06, cs_07, cs_08) = (-, -, +, +), so the heading loop was positive feedback; the fin loop gain was not scheduled with speed (fin force ~ speed^2); and the surge command was a bare RPM with no speed loop, so on this almost drag-free vehicle it coasted away at 1.7 m/s and, misaligned, had no flow to turn with. **Which fin sign is physically right is still the unverified assumption of section 9 item 2**: it is now decided in one place (the allocator), which `dof_testing --dof yaw --mode step` checks on the real sim.

### 5.8b `control_code/mission/` — lawnmower, orbit, spiral, yo-yo and chained missions

One controller for every survey-like job. A mission is a list of `legs:` in `mission.yaml`; each leg is expanded into waypoints by a path generator and flown by the verified `WaypointTracker` (so the yaw, speed and heave loops, the allocator and the pose filter are the same as for waypoint tracking). **Offline only (fake vehicle, assumed 2.5 m turning circle); nothing ran on the real sim.**

| Leg | Parameters | What it flies |
|-----|------------|---------------|
| `goto` | `x, y, z` | to a point |
| `lawnmower` | `origin, heading_deg, length_m, width_m, spacing_m, z` (or `depths: [..]`) | parallel lanes with semicircular U-turns that bulge half a step past the lane ends; a straight run-in (`leadin_m`, default 2 turning radii) lines the vehicle up. A spacing under 5 m (2 turning radii) is flown with a SKIP pattern (lanes 0, 2, 4, then 1, 3, ...) and needs at least 5 lanes; the checks say when a step is still too small. `depths` flies the lanes at each depth in turn (stepped survey) |
| `orbit` | `centre, radius_m, revs, clockwise, start_deg, z` | a circle (polygon, `points_per_rev` points, acceptance 1 m); radius at least 1.1 x the turning circle |
| `spiral` | `centre, r_start_m, r_end_m, pitch_m, z` | an orbit whose radius changes by `pitch_m` per revolution |
| `yoyo` | `from, to, z_top, z_bottom, cycles` | a straight line with the depth swinging between the two depths (depth acceptance 0.6 m) |
| `hold` | `seconds` | thrust stopped, depth held, no heading hold (the fins need flow) |
| `return_home` | | to `mission.home` (default: where the mission started), then hold |

Because the vehicle cannot turn on the spot, **legs are chained geometrically**: each leg should start roughly where and the way the previous one ended (the shipped example does: lawnmower from the start, orbit entered tangentially after the last lane, yo-yo along the exit heading, a wide U-turn home). `validate_mission()` checks the WHOLE chained path (turning circle, geofence, dock keep-out, and the circle swept by every turn of more than 45 degrees), and the Config tab shows the warnings as you edit. Straight legs are flown as lines (`cross_track.enabled`): offline, with a steady 1.5 N sideways push the lane error stays below 1 m with it and grows past 3 m without it.

Safety: `safety.geofence` (a box) and `safety.dock_keepout_m` (a circle around the dock, applied to every path point too). Leaving the box or entering the circle ABORTS the mission and the vehicle returns home once and holds (`aborted` is shown in the viewer and in `ctrl_debug`); an odometry gap longer than `safety.failsafe.odom_loss_s` does the same when the odometry comes back; a depth or attitude limit trip is a hard stop. The vehicle needs about two turning radii to turn back, so the keep-out must be that much bigger than the closest approach you accept (offline, a 3 m circle did not stop the vehicle reaching the dock; 6 m does).

| File | Role |
|------|------|
| `paths.py` | `lawnmower`, `orbit`, `spiral`, `yoyo`, `lane_order`, `check_path` (turning circle with a tolerance, geofence, dock keep-out, `swing_warnings`). Pure Python. |
| `mission_core.py` | `expand_legs`, `validate_mission`, `plan_polyline`, **`MissionRunner`** (segments, per-leg overrides, hold, return-home, failsafe, progress, ETA, `debug()` for the viewer). |
| `mission.py`, `run_mission.sh` | ROS node (odometry through the pose filter, neutral on frozen or stale odometry, `ctrl_debug` with leg, progress, ETA, cross-track error, target) and launcher; `./run_mission.sh --config my_mission.yaml`. |
| `mission.yaml` | The example mission (lawnmower -> orbit -> yo-yo -> return home), speed, gains (incl. `cross_track`), limits, estimator, safety, `survey.swath_width_m`. Every parameter has its quantified effect in a comment. |
| `ros_mission_confirm.py` | Flies the `quick` (lawnmower + home) or `full` preset over ROS on a private domain against the fake vehicle (realistic odometry, `--freezes` optional) and judges from ground truth. |
| `test_paths.py`, `test_mission_core.py` | 13 + 17 tests, no ROS: geometry, skip pattern, reachable turns, keep-out and geofence checks; every leg type closed loop; the lane with and without cross-track (negative control); geofence, keep-out and odometry-loss aborts (and the same missions without them flying on); progress and ETA; validation messages. |
| `mission_showcase.py` (+ `test_mission_showcase.py`) | Flies every mission type (lawnmower, orbit, spiral, yo-yo, go-to + hold, the chained shipped mission) in the offline model with realistic odometry, optionally with odometry freezes, and saves per mission `trajectory.csv` (replayable), top view, plots and an `index.html` gallery in `outputs/sim_viewer_runs/missions_showcase/`; copies the CSVs to `outputs/sim_viewer_runs/mission_<name>.csv` for the History tab. `python3 mission_showcase.py` (about 10 minutes). |
| `sonar_survey.py` (+ `test_sonar_survey.py`) | The side-scan lawnmower survey over the rugged terrain (constant depth and terrain following) with the judge (coverage >= 0.8, minimum altitude >= 1 m); saves trajectory, `sidescan.npz`, waterfall, mosaic and mosaic-vs-truth to `outputs/sim_viewer_runs/sonar_showcase/`. See section 5.11. |

Offline results: the shipped mission completes in ~240 s of simulated time and ends 0.4 m from home, closest to the dock 6.8 m. Over ROS (fake vehicle with realistic odometry, no freezes): `quick` 75 s, `full` 234 s, both PASS; `full --freezes` (the fake odometry freezes about every 60 s like the real sim) 287 s, PASS (a freeze longer than `safety.failsafe.odom_loss_s` = 15 s would abort to return-home).

### 5.9 `control_code/ws/` — ROS 2 interfaces package

A colcon workspace holding one package, `interfaces`, so Python code can `from interfaces.msg import Actuator, DockAlign`.
Build output (`build/`, `install/`, `log/`) is ignored by git.

| File | Role |
|------|------|
| `ws/src/interfaces/package.xml` | Package manifest: depends on `std_msgs` and `geometry_msgs`, uses `rosidl_default_generators`. |
| `ws/src/interfaces/CMakeLists.txt` | Generates the four message types listed below. |
| `ws/src/interfaces/LICENSE` | Apache-2.0 text, kept because the package is vendored from mavsim. |
| `ws/src/interfaces/README.md` | Notes that `Actuator`, `DVL` and `WaveProbe` must stay in sync with the mavsim repo, while `DockAlign` exists only here. |
| `msg/Actuator.msg` | `header`, `actuator_values[]`, `actuator_names[]`, `covariance[]`. The command message: names like `th_02`, `cs_04` with RPM or degrees. |
| `msg/DockAlign.msg` | Dock guidance published by the detector (full field list in section 6). Ours only. |
| `msg/DVL.msg` | `header`, body-frame `velocity` (m/s), `covariance[9]`. From mavsim; not used by our code. |
| `msg/WaveProbe.msg` | `header`, wave `elevation` (m) and `location`. From mavsim; not used by our code. |

Build: `cd control_code/ws && source /opt/ros/humble/setup.bash && colcon build --packages-select interfaces && source install/setup.bash`.

### 5.10 `dock_detection_algo/` — dock-light perception

Subscribes to a mavsim **compressed camera** topic and to **`/Mako_01/imu_01/data`**, builds a **bloom mask**, finds up to **four
light cores**, removes **camera roll**, then does one of two things. With four trusted lights it publishes the dock center, the
cross-track cue, the top–bottom tilt, and the dock center’s angle below the horizon. With fewer than four it does not guess which
lamps are missing; it publishes a yaw / pitch / surge **search** that tries to bring the others into frame. Two OpenCV windows show
the roll-compensated image. The detector itself does not move the vehicle; `control_code/dock_test` (standoff) and
`control_code/terminal_docking_control` (alignment and docking, which also uses the four light pixels) consume `DockAlign`. Section 13 is the full explanation.

Pipeline: `BGR image → HSV → bloom mask → cores (peaks, and a second pass at half peak_sep if fewer than four) → unrotate by IMU
roll → four lights: cross-ratio center, lateral cue, spread, elevation; otherwise: acquire search → DockAlign`.

| File | Role |
|------|------|
| `live_dock_lights.py` | Main program. Node **`DockLightsLive`** subscribes to the camera and the IMU, publishes `DockAlign`. The main thread runs two OpenCV windows on the **roll-compensated** frame (**Dock camera**: banner, labelled lights, range / bearing / elevation / view-angle gauges, a top-down mini map, a history strip; **Bloom mask**: the mask in colour with the chosen lights and the tuning trackbars), plus an optional third window (`--align-window`). If the first peak pass finds fewer than four cores, it immediately retries at half `peak_sep`. Keys: **p** prints the current trackbar values as YAML, **q** or **Esc** quits. Options: `--topic`, `--align-topic`, `--config`, `--no-gui` (publish `DockAlign` without the OpenCv windows; trackbars take the YAML values), `--align-window` (also open the **Dock align** window). |
| `dock_hud.py` | **The pop-up drawings** (2026-10-11), pure OpenCV + numpy: `draw_camera_view`, `draw_mask_view`, `draw_align_view`, `History` (rolling buffer), `info_from_msg` / `info_from_result` (the plain dict they take), `range_m`, `bearing_deg`, `view_angle_deg`. No ROS, so they are tested and rendered to PNG without a display (`render_hud_samples.py`). |
| `dock_align_view.py` | **Separate window for the `dock_align` topic** (2026-10-11): big YAW L / YAW R and UP / DOWN lamps that light when the error is outside 10 px, a target with the dock's offset from the picture centre, the command in words (`yaw LEFT + pitch/heave UP`, or the search hints while there is no lock), all numbers (range, bearing, elevation, errors, ring radius, spread, lateral cue, obliqueness, view angle, lights/confidence, message age) and rolling plots. Runs on its own (`python3 dock_align_view.py`, `--save-png`) or inside the live node. Shows `no message for N s` instead of stale numbers. |
| `dock_ring.py` | **Which bright spots are the dock?** (2026-10-11) `select_ring` takes up to 8 bright spots and keeps the 4 that form the dock ring: an affine image of the 1 m circle (top and bottom on a diameter, the two side lights above the centre), checked by a least-squares fit whose leftover (`residual`) must be below `ring.fit_tol` (0.08). A bubble, glint, reflection or false light then cannot take the place of a real light, and a ring with one light hidden (3 real + 1 false) is rejected instead of being steered on. |
| `reliability_study.py`, `compare_reports.py`, `render_hud_samples.py` | The reliability study (one thing harder at a time: range, view angle, heading, lateral offset, pitch, roll, noise, JPEG quality, brightness, side-light dimness, water clarity, background, blur, motion blur, bubbles, marine snow, backscatter, glint, false lights, reflections, one light hidden), the before/after table, and the pop-up PNG renderer. Reports are in `outputs/detection_reports/`. |
| `dock_detector.py` | **`DockDetector`**: the whole per-frame pipeline (mask, cores, roll removal, geometry, search hints, `DockAlign` message) as a plain object with no ROS node and no windows. `live_dock_lights.py`, the closed-loop docking sim and the threshold tuner all call it, so they run the very same code. |
| `tune_detection.py` | Scores and tunes the thresholds on synthetic frames (random poses x random looks incl. murky water); GOOD / BAD LOCK / MISS by range band with train, hold-out and fresh sets. See the header of `dock_detection.yaml` for the 2026-10-10 result. |
| `dock_light_mask.py` | Core vision. `bloom_mask_and_cores(...)` builds the mask (`_build_bloom_mask`; since 2026-10-11 after subtracting the water's own brightness above `mask.bg_ref_v` per image row, `background_excess`) and finds cores either as **peaks** (`_cores_from_peaks`: difference of Gaussians + distance transform + `_nms_peaks`, needed when the lights merge) or as **blob centroids** (`_cores_from_contours`); a saturated (clipped) light is centred on its flat plateau (`_plateau_centre`, 0.40 -> 0.11-0.23 px error) and up to `ring.candidates` peaks are returned for the ring check. `draw_cores` overlays markers. All constants are parameters. |
| `dock_geometry.py` | The guidance centre is a smooth blend between the cross-ratio centre and the diameter midpoint across the lateral gate (a hard switch made it jump by ~10 px). `unrotate_cores` removes roll. `label_dock_lights` assigns Top (smallest y), Bottom (largest y), then Left/Right by x, and rejects the frame if a side light does not fall between top and bottom along the diameter. `evaluate_dock_geometry` returns spread from the top–bottom **midpoint** (the old square-on test) and, when the side midpoint lies on that diameter, the **cross-ratio center** of the dock. `lateral_px` and `obliqueness` are the pitched-view cues. `draw_dock_geometry` and `draw_mask_debug` draw the overlays. |
| `dock_acquire.py` | `PartialAcquire` turns 0–3 cores into `search_yaw_norm`, `search_pitch_norm` and `search_surge_norm`. `track_surge_norm` asks for a little forward thrust whenever four lights are visible but the vehicle still needs the fins to yaw. |
| `dock_align_msg.py` | `build_dock_align_msg` fills every `DockAlign` field. `quat_to_roll_pitch` matches `control_code/common/state.py`. `elevation_down_rad` is the heave cue. `align_topic_from_camera` maps `/Mako_01/camera_03/...` to `/Mako_01/dock_align`. |
| `dock_detection_config.py` | `load_config` reads `dock_detection.yaml`; `detector_kwargs` turns the mask and peak sections into the keyword arguments of `bloom_mask_and_cores`. |
| `dock_detection.yaml` | **All detection tunables**, each commented. Sections: `camera` (topics, `vfov_deg`, the vertical field of view), `mask`, `peaks`, `alignment` (spread gate plus `lateral_exact_px`), `acquire` (edge margin, nod, backup), `hud`. |
| `test_pitched_geometry.py` | Projects the real 1 m light circle through a pinhole camera and checks the cues. No ROS. `python3 test_pitched_geometry.py` from this folder. |
| `dock_detection_algo_launch.py` | Optional `ros2 launch` wrapper with `topic` and `config` arguments. |
| `run_live.sh` | One-command launcher: sets the domain and the Fast-DDS UDP profile, sources ROS and the workspace (builds `interfaces` if missing), runs `live_dock_lights.py`. Overrides: `TOPIC`, `CONFIG`, `ALIGN_TOPIC`. |
| `echo_align.sh` | Prints `/Mako_01/dock_align` (`ros2 topic echo`); extra arguments are passed through (for example `--once`). Override the topic with `ALIGN_TOPIC`. |
| `check_camera.sh` | Checks that a camera topic exists and receives data (`ros2 topic hz`). Set `TOPIC`; without it, it defaults to `/dock_02/camera_02/image/compressed`. |
| `fastrtps_no_shm.xml` | Fast-DDS profile that forces **UDP** and disables shared memory (see section 10). |
| `requirements.txt` | `opencv-python`, `numpy`, `pyyaml`. |

### 5.10b Detector reliability, robustness and the pop-ups (added 2026-10-11)

**Measured, not guessed.** `dock_detection_algo/reliability_study.py` renders frames of the synthetic camera and changes ONE thing at a time (40 frames per value, same frames for the before and after runs; `reports/before` = the old settings in `reports/dock_detection_before.yaml`, `reports/after` = the shipped ones, `reports/comparison.txt` = both side by side). A frame is GOOD if the detector is valid with four lights each within 6 px of the truth, BAD LOCK if it is valid but a light is wrong (the dangerous case: the controller would steer on it), MISS if it is not valid; frames where the dock is not fully inside the picture are counted apart (OUT), not as misses.

| Thing made harder | GOOD before -> after | BAD LOCK before -> after |
|---|---|---|
| one false bright light per frame | 12.8% -> 100% | 87.2% -> 0% |
| two / four false lights | 0% -> 97.5% | 95% / 87.5% -> 2.5% |
| surface glint 0.3 / 0.6 / 1.0 | 82% / 7.5% / 2.5% -> 100% / 97.5% / 97.5% | 18% / 92.5% / 90% -> 0% / 2.5% / 2.5% |
| murky water, background x3 / x4 / x5 | 97.5% / 10% / 10% -> 100% / 95% / 55% | 2.5% / 47.5% / 45% -> 0% / 5% / 42.5% |
| JPEG quality 25 / 12 | 95% / 66.7% -> 100% / 100% | 5% / 33.3% -> 0% / 0% |
| backscatter veil 0.8 | 41% -> 87.2% | 59% -> 12.8% |
| bubbles 30 per frame | 100% -> 95% | 0% -> 5% (one frame of 40: the one place where it got worse) |
| one ring light hidden (any of the four) | never GOOD (correct) | 0% -> 0% (a hidden light is a MISS, never a wrong lock) |
| blur sigma 6 px, noise 0.09, motion blur 25 px, snow 400, reflections 4 | 100% -> 100% | 0% -> 0% |
| pixel error of a good frame | 0.40 -> 0.11-0.23 px | |

What changed in the detector (each step was measured alone; `dock_detection.yaml` comments carry the numbers): (1) **ring check** (`dock_ring.py`, `ring:`) instead of "the four strongest peaks"; (2) **plateau centres** (`peaks.plateau_min_px`): a light brighter than the sensor range clips to a flat top and the old peak landed anywhere on it; (3) **water subtraction** (`mask.bg_subtract`, `bg_ref_v: 110`): the water's own brightness above 110/255 is removed per image row, so murky water and a bright veil no longer flood the mask (it does nothing in clear water: the old frames are unchanged). On the 400-frame tuning set (`tune_detection.py report`) the old settings gave GOOD 89.2% / BAD LOCK 4.5% (pixel error 0.70 px), the new ones 91.5% / 3.0% (0.33 px); the coordinate-search tuner finds nothing worth changing on a fresh set (89.0% -> 89.3%), so the thresholds were NOT retuned.

**Limits that stay (physical, not a bug):** seen from more than about 45-50 degrees off the dock axis the ring is invisible (the lights shine in an 82 degree cone: `view_deg` +-60 -> 0% for old and new alike); closer than about 2.3 m the 2 m ring no longer fits the 60 degree picture (range 1.5 and 2 m are OUT); ring plus 8 false lights at once is more than the 8 candidates; murky water x5 and a veil of 0.8 still lose frames.

**The pop-up windows** (`dock_hud.py`, run with `dock_detection_algo/run_live.sh`; `--align-window` adds the third; `python3 render_hud_samples.py OUT_DIR` writes PNGs of all of them): *Dock camera* = banner (LOCKED-ALIGNED / LOCKED-off centre / SEARCHING n of 4 / NO LIGHTS) with lights and a confidence bar, the picture with T/B/L/R-labelled lights, the ring, the picture centre and an arrow to the dock centre, gauges for range, bearing, elevation and view angle, a top-down mini map (vehicle, field-of-view wedge, 2/5/10 m rings, the dock at the measured range and bearing) and a rolling history strip; *Bloom mask* = the mask in colour with the chosen lights; *Dock align* = steering lamps, target, the command in words, every DockAlign number, the search hints and rolling plots. I could only look at these as saved PNGs (no display here): please open them once on your desktop.

### 5.11 Side-scan sonars (Omniscan 450 SS) and a rugged sea floor (added 2026-10-11)

The real AUV carries a pair of **Cerulean Omniscan 450 SS** side-scan sonars. mavsim has no sonar and no terrain (its bridge only reserves a sonar port), so both are built in the OFFLINE simulator. Mounting (confirmed): each transducer's axis is **30 deg off straight down** toward its side (60 deg below the horizontal), at the longitudinal centre on the belly; with the 50 deg across-track beam the swath covers 5-55 deg from the nadir: on a flat floor ground range 0.09 H to 1.43 H and slant range up to 1.74 H per side (H = altitude), a blind gap straight below. Datasheet numbers used: 450 kHz, 0.5 deg x 50 deg beam, up to 150 m, 200-1200 bins, ping rate up to 20 Hz, gain 0-7 or auto.

| File (`control_code/sim_viewer/` unless noted) | Role |
|---|---|
| `terrain.py` | The sea floor, shared by the 3D view, the sonar and the tests: 300 x 300 m at 0.25 m cells, depth 6-26 m, median slope ~24 deg: fractal relief at several scales, sharp ridges, trenches, 320 boulders, sand ripples, rock / gravel / sand / mud patches (Lambert backscatter per class in dB), a container, a block, a slab and a pipeline standing on the bottom, and a calm patch round the dock (dock floor ~11 m). Seeded, deterministic, bilinear `height / normal / reflectivity_db` (numba). Config block `terrain:` in `sim_viewer.yaml`. The 3D floor is built from it (`environment.py`, `look3d.use_terrain`). |
| `synthetic_sidescan.py` | The PHYSICS (`SideScanSim.ping`): a fan of 1100 rays x 5 along-track lines is marched through the terrain; the first surface hit gives slant range, the normal and the backscatter; shadows behind ridges and boulders appear by themselves, layover adds up, the nadir is blind. Echo power per ray = beam pattern x Lambert backscatter x |n.r| x solid angle / R^2 x absorption (0.10 dB/m, sea water 15 deg C, c = 1500 m/s), summed into range bins, speckle (gamma, 3 looks), noise floor, time-varied gain (compensates spreading and absorption), gain 0-7 (6 dB steps) or auto, uint16. Ping rate min(20 Hz, 0.9 c / 2R). Also the simulated altimeter (straight-down range, 1% + 2 cm noise) and `flat_swath_geometry`. 3.4 ms per ping. |
| `sidescan_node.py`, `run_sidescan_sim.sh` | ROS node: pings on `/Mako_01/sonar_01/ping` (port) and `sonar_02/ping` (starboard) as new `interfaces/SideScan` messages (header, side, ping number, start/end range, sound speed, gain, ping rate, altitude, vehicle pose at transmit, `uint16[] intensity`), `/Mako_01/altimeter/range` (`sensor_msgs/Range`), `/Mako_01/sonar/status`; **range, gain and bins can be changed at run time** with a JSON on `/Mako_01/sonar/cmd` (`{"range_m": 40, "gain": 5}`; gain -1 = auto; `{"enabled": false}`). Refuses domain 42. |
| `sonar_panel.py`, `sidescan_view.py`, `sidescan_mosaic.py` | The viewer's **Sonar tab** (bottom-left tab bar): scrolling WATERFALL (port left, starboard right, newest on top), a georeferenced MOSAIC (slant-range corrected with the altimeter, north up, only bins inside the beam), boxes for range / gain / bins that command the sonar, freeze, clear, save pictures. |
| `recording.py` (`SonarRecorder`, `SonarLog`) | Saved pings: `<run>_sidescan.npz` next to a run CSV (or `sidescan.npz` in a survey folder). `replay.py` plays them back in step with the trajectory (so a saved survey can be looked at again later); the viewer's auto-recording (`history.py`) saves them for live runs. |
| `mission/sonar_survey.py` | The lawnmower SURVEY: flies the mission closed loop (realistic odometry) with the sonars pinging at their physical rate, in two modes (constant depth, terrain following), judges it (finished, coverage >= 80%, altitude >= 1 m) and saves trajectory.csv + sidescan.npz + waterfall / mosaic / mosaic-vs-truth pictures + summary.json + index.html. |
| `mission/mission_core.py`, `mission.yaml` `terrain_follow:` | **Terrain following**: hold a fixed altitude (default 6 m) by moving the depth target with the altimeter (rate limited); `mission.py` subscribes `/Mako_01/altimeter/range`. |

**Results (offline, assumed levels):** the lawnmower survey (4 lanes of 90 m, 14 m apart, 30 m range, auto gain, 502 s, 20 102 pings): constant depth 2 m: coverage 100%, altitude 8-20 m (min 8.0 m), no shadows (too high); terrain following at 6 m: coverage 98% (90 m2 of gaps), min altitude 1.9 m, depth 2-15 m, 3% of the insonified cells in shadow. Tests prove: the swath edges against the analytic geometry, the nadir gap, a boulder's shadow length against `g*h/(H-h)` (and none without the boulder), port/starboard mirror symmetry, gain and absorption effects, the mount angle, roll, the ping rate, the mosaic georeferencing (objects and their shadows land on the correct side), recording / replay, the viewer tab, the node over ROS with run-time commands, terrain following over a ramp and a survey that is judged FAILED when its lanes are too far apart. **Assumed, not verified:** the echo levels (source level, noise floor, gain steps, backscatter dB per class), the speckle, the sea floor itself, and the mount angle's effect on real data; no real Omniscan recording was available. The mosaic-vs-truth correlation of the backscatter is only 0.03-0.06 on this terrain because geometry (grazing angle, shadows, ranges distorted by the flat-bottom assumption) dominates the picture, as in real surveys. Details and commands: `docs/execution.md` section 20.

---

## 6. ROS topics and messages

### Topics used (vessel `Mako_01`)

| Topic | Type | Direction | Used by |
|-------|------|-----------|---------|
| `/Mako_01/odometry_sim` | `nav_msgs/Odometry` (`frame_id NED`, `child_frame_id BODY`, ~4 Hz on the live sim) | read | all controllers, `fake_vehicle` writes it offline |
| `/Mako_01/actuator_cmd` | `interfaces/Actuator` | write | all controllers, read by the bridge |
| `/Mako_01/camera_03/image/compressed` | `sensor_msgs/CompressedImage` | read | dock detector (default camera; `camera_04` and `camera_05` also exist) |
| `/Mako_01/dock_align` | `interfaces/DockAlign` | write | dock detector |
| `/Mako_01/ctrl_debug` | `std_msgs/String` (JSON: controller, mode, setpoints) | write | `station_keeping`, `dof_testing`, `dock_test` (for the viewer's plots) |
| `/Mako_01/ctrl_gains` | `std_msgs/String` (JSON `{controller, loop, key, value}`) | write (viewer) / read | `terminal_docking`, `dof_testing`, `station_keeping` apply it to the running loops |
| `/Mako_01/sim/cmd`, `/Mako_01/sim/status` | `std_msgs/String` (JSON) | cmd read / status write | `fake_vehicle` only (pause, step, reset, push, time scale; the status also carries the ground-truth pose `truth`); the real mavsim has neither |
| `/Mako_01/vessel_state` | `std_msgs/Float64MultiArray` | read | not used by the current controllers (`waypoint_tracking` used it optionally before its 2026-10-10 rebuild) |
| `/Mako_01/imu_01/data` | `sensor_msgs/Imu` | read | dock detector (roll compensation and pitch for elevation). DVL is still unused. |
| `/Mako_01/dvl_03/data` | DVL | not used by our code | |

Always match `ROS_DOMAIN_ID` to the bridge (default **42**). Only one program should publish `/Mako_01/actuator_cmd` at a time; the
bridge's own **`mavsim_teleop` node publishes zeros on it at 20 Hz** and must be stopped first (`docs/execution.md` section 3.4).

### `interfaces/Actuator`
| Field | Meaning |
|-------|---------|
| `header` | timestamp and frame |
| `actuator_names[]` | e.g. `th_01`, `th_02`, `th_03`, `cs_04`, `cs_06`, `cs_07`, `cs_08` |
| `actuator_values[]` | same order as the names: RPM for `th_*`, degrees for `cs_*` |
| `covariance[]` | unused by our code (zeros) |

### `interfaces/DockAlign`
| Field | Meaning |
|-------|---------|
| `header` | timestamp, frame `camera` |
| `valid` | four lights were labelled and the side lights lie between top and bottom. False during the search |
| `num_lights` | number of cores found |
| `error_x_px`, `error_y_px` | guidance center minus image center (`+x` = dock right, `+y` = dock below). `error_y` is the **pitch** cue that keeps the dock in frame, not the heave cue |
| `error_x_norm`, `error_y_norm` | the same, divided by half the image width / height |
| `image_width`, `image_height` | frame size |
| `radius_px` | half the top–bottom distance |
| `spread_px` | largest minus smallest of the four distances from the **top–bottom midpoint**. Near 0 only when the camera is level and the dock is seen square-on. A pitched view makes this large even with a perfect heading, so it is ignored until the vehicle is level again |
| `err_left_px`, `err_right_px` | left/right light distance from that midpoint, minus the top–bottom radius |
| `aligned` | `spread_px` below `max(spread_align_min_px, spread_align_frac × radius_px)`. This is the final square-on check, after depth has been matched and pitch has returned to zero |
| `diameter_angle_deg` | angle of the top→bottom line against image vertical (0 = upright). On a level camera a pure yaw leaves this near 0; the line only tilts from roll or from pitch combined with yaw. Roll is removed before this angle is computed |
| `center_exact` | true when the side-light midpoint lies on the top–bottom line, so `center` is the cross-ratio dock center. False means `center` fell back to the top–bottom midpoint |
| `lateral_px` | signed pixels of the side-light midpoint to the **right** of the top→bottom line. Near 0 when the heading is perpendicular to the dock, **including** when the vehicle is shifted sideways. A yaw pulls it off the line, and the pull is much larger at close range than at 8 m |
| `obliqueness` | where the side midpoint sits along the diameter, minus 0.1464. Zero when the camera is level (any depth). Negative when the camera is below the dock and pitched up to look at it. Not the heave command |
| `elevation_rad` | angle of the dock center **below the horizon**, radians. Positive means the dock is deeper than the camera, so heave should go down. `atan((center_y − image_center_y) / fy) − pitch`, with pitch positive nose-up |
| `elevation_valid` | an IMU pitch sample has been received. Until then elevation is 0 and must not be used for heave |
| `search_yaw_norm` | partial-light yaw hint, +1 = yaw toward image right. 0 while `valid` |
| `search_pitch_norm` | partial-light pitch hint, +1 = pitch down. 0 while `valid` |
| `search_surge_norm` | axial hint. About +0.35 when the fins need flow to yaw. −1 when the dock is too close to fit in the frame. 0 when four lights are visible and already square |
| `center`, `top`, `bottom`, `left`, `right` | pixel positions in the roll-compensated image (`z` unused). `center` is the cross-ratio center when `center_exact`, otherwise the top–bottom midpoint |
| `d_top`, `d_bottom`, `d_left`, `d_right` | distances from the **top–bottom midpoint** (the spread reference), not necessarily from `center` |
| `status` | `ok`, `ok_not_aligned`, or a search status: `search_edge`, `search_nod`, `search_no_lights`, `backup_taller_than_frame`, `backup_close` |
| `confidence` | 0 to 1: grows with the number of lights found and falls with spread |

---

## 7. How the pieces fit together (data flow)

```
                      ┌───────────────────────────── mavsim (simulator + bridge, Docker) ─────────────────────────────┐
                      │                                                                                                │
  /Mako_01/camera_03  │  /Mako_01/odometry_sim (NED pose, BODY twist)                    /Mako_01/actuator_cmd (RPM, deg)│
          │           │            │                                                                  ▲                 │
          ▼           └────────────┼──────────────────────────────────────────────────────────────────┼─────────────────┘
 ┌────────────────────┐            │                                                                  │
 │ dock_detection_algo│            ▼                                                                  │
 │ live_dock_lights.py│   ┌──────────────────────────────── control_code ────────────────────────────────────┐
 └─────────┬──────────┘   │ depth_control / waypoint_tracking / dof_testing / station_keeping                  │
           │              │            state ──► PID loops (common/loops.py) ──► wrench [X Y Z K M N]          │
           ▼              │                                         │                                          │
 /Mako_01/dock_align      │                       common/allocation.py: wrench ──► thruster RPM + fin degrees  │
        │                 └────────────────────────────────────────────────────────────────────────────────────┘
        └──────────────► control_code/dock_test  (standoff only; does not enter the funnel)

 The detector also reads `/Mako_01/imu_01/data` (roll and pitch). It does not publish actuator commands.

 Offline: sim_offline/fake_vehicle.py replaces the simulator box (it reads actuator_cmd and publishes odometry_sim).
```

---

## 8. Testing

| Level | How | What it proves |
|-------|-----|----------------|
| Unit and offline system tests | `cd control_code && python3 -m pytest tests -q` | Allocation signs, PID behaviour, every DOF test passes offline, wrong-sign actuators are caught, station keeping holds against pushes |
| Detector robustness, ring check, pop-ups, reliability harness (2026-10-11) | `cd dock_detection_algo && python3 -m pytest test_dock_ring.py test_robustness.py test_dock_hud.py test_reliability_study.py -q` (needs ROS sourced; about 1 minute) | The ring picks the real lights among false ones; every degradation (false lights, glint, hidden light, murky water, veil, blur, JPEG, bubbles) is handled, each with a NEGATIVE CONTROL on the old behaviour; saturated lights are centred on their plateau; the drawings have the right size and colours and the direction lamps light on the right side; the study harness writes its report |
| Search and back-out (2026-10-11) | `cd control_code/terminal_docking_control && python3 -m pytest test_recovery.py -q` (about 3 minutes) | A close, angled start backs out instead of driving into the dock (the old settings hit it), a start with room does not back out, a hidden dock is found with a rough position (the old controller waits for ever), a close sideways start is not swung into the dock |
| Dock geometry | `cd dock_detection_algo && python3 test_pitched_geometry.py` (11 tests; needs ROS sourced) | Level, deep, pitched, offset, yawed and rolled views of the real 1 m light circle; edge, nod and backup search commands |
| Side-scan sonar and terrain (2026-10-11) | `cd control_code && python3 -m pytest sim_viewer/tests/test_terrain.py sim_viewer/tests/test_synthetic_sidescan.py sim_viewer/tests/test_sidescan_mosaic_and_recording.py sim_viewer/tests/test_sonar_panel.py mission/test_sonar_survey.py -q` (no ROS, about 1.5 min); `sim_viewer/tests/test_sidescan_ros.py` (ROS, domain 75) | See section 5.11: analytic swath, shadows with negative controls, gain / absorption / mount / roll / ping rate, mosaic placement, recording and replay, the Sonar tab, the node over ROS with run-time commands, terrain following, the survey harness |
| Docking, tuning, viewer (2026-10-10) | `cd control_code && python3 -m pytest tests dock_test waypoint_tracking mission sim_viewer/tests terminal_docking_control tuning ../dock_detection_algo/test_tune_detection.py ../dock_detection_algo/test_imu_convention.py ../dock_detection_algo/test_dock_ring.py ../dock_detection_algo/test_robustness.py ../dock_detection_algo/test_dock_hud.py ../dock_detection_algo/test_reliability_study.py -q` | The whole suite (456 passed with ROS sourced and `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, 467 with `test_pitched_geometry.py`; 409 passed + 21 skipped without ROS; about 8 minutes with ROS): docking core and closed loop, the full stack over ROS, pose filter, sensor model, tuning tools, missions, viewer panels/tabs/HUD/environment/Config tab, the recorder, replay, compare, calibrate |
| Missions | `cd control_code && python3 -m pytest mission -q` (no ROS), `python3 mission/ros_mission_confirm.py --preset quick` (ROS, private domain) | Path geometry and checks; every leg type flown closed loop against the offline vehicle; lane keeping with and without cross-track (negative control); geofence, dock keep-out and odometry-loss aborts; the node end to end over ROS against the fake vehicle with realistic odometry |
| Viewer Config tab | `cd control_code && python3 -m pytest sim_viewer/tests/test_config_model.py sim_viewer/tests/test_config_panel.py -q` (no ROS); `sim_viewer/tests/test_mission_ui_ros.py` (ROS) | Comment-preserving YAML edits of the real controller files, waypoint / leg editing, map events, saving with a backup, refusal of edits that cannot be written exactly, and the full chain edit -> Start -> the edited values are flown |
| Dock standoff wrench, heading recovery and the close-start guard | `cd control_code/dock_test && python3 -m pytest . -q` (33 tests, no ROS) | Elevation dives, dock-below pitches nose down, dock-right yaws right, a squared pose commands zero surge, a partial detection does not heave, the heading error at the standoff is worked off by backing away, a blind search stops after 0.25 m and backs away instead of driving into the dock (negative control: without it every run goes through the dock plane), closed-loop runs with a negative control |
| IMU / elevation convention | `cd dock_detection_algo && python3 -m pytest test_imu_convention.py -q` | The IMU and odometry quaternion conversions are the same function; the elevation equals the geometric angle below the horizon at several pitches; a flipped pitch sign is detectable. Whether the REAL IMU uses this convention is only decided by `check_imu_convention.py` on the live sim |
| Offline end-to-end over ROS | `sim_offline/fake_vehicle.py` plus any controller, in ROS domain 77 (`docs/execution.md` section 10) | The ROS nodes and topics work together |
| Real simulator | `dof_testing` (`docs/execution.md` section 7), then station keeping | The only test that confirms signs, thrust and fin behaviour on the real vehicle |

---

## 9. Known limits and unverified assumptions

These are the things to check on the first real-simulator run.

1. **Control code has so far been tested only against the offline model**, not the real simulator. `depth_control`,
   `waypoint_tracking` (rebuilt 2026-10-10 on `common/`) and the new `mission` controller have not been run on the simulator.
2. **Fin deflection sign** is assumed (positive deflection gives force along the fin's lift direction). Run `dof_testing` for yaw and roll
   in `step` mode: a wrong sign shows up as FAIL.
3. **Fin moment arm:** the vessel file gives fin x = 0, but the fin geometry lies about 0.42–0.49 m aft; the code uses **−0.454 m**. The lift slope
   (`cl_alpha_per_rad: 2.0`) is a guess.
4. **Thrust units:** the allocator assumes `T = KT·ρ·D⁴·n²` with `n = RPM / 60`. If the real thrust is far off, change the single value
   `rpm_to_rps` in `mako_geometry.yaml`.
5. **Odometry twist frame** is assumed to be BODY (the message says `child_frame_id: BODY`, but it has not been proven with a motion
   test). Odometry arrives at only ~4 Hz. The IMU orientation convention used for the dock elevation (roll/pitch signs) is also unverified.
6. **`rpm_cap: 1000`** in `depth_control.yaml` (`waypoint_tracking.yaml` now uses 1800 like station keeping): with the thrust assumption above, the two heave thrusters give
   about 7 N at 1000 RPM. That is enough for gentle moves on the neutral vessel but leaves little margin for disturbances. `dof_testing` and
   `station_keeping` use 1800.
7. **Offline model:** drag is assumed (quadratic values chosen to give plausible speeds). The overshoot and settling numbers quoted in the YAML
   comments come from this model: trust the direction and relative size, not the absolute values.
8. **Yaw and roll authority needs flow:** station keeping cannot hold heading in still water, and sideways drift cannot be corrected at all.
9. **Pitch and roll step tests are short pulses** on purpose: the vehicle has no righting moment and no damping in those axes.
10. **Dock detection** is checked by projecting the sim’s 1 m light circle (`test_pitched_geometry.py`) and its thresholds were tuned (2026-10-10) on SYNTHETIC frames only (`dock_detection_algo/tune_detection.py`; the shipped yaml header lists the measured result and its sensitivity). The look of the lights, the water colour and the noise are assumed; several values sit at the edge of what those frames allow. It has not been tuned on a live camera stream.
11. **`dock_test` has not been run on the simulator.** Its wrench signs are checked offline. Gains in `dock_test.yaml` are starting values, not measurements. It holds a standoff and will not enter the funnel. Yaw still needs flow, and there is still no sway thruster: a heading error at the standoff is now worked off by backing away (`mode=realign`, 2026-10-10; offline 38/90 runs end at rest vs 7/90 before, ROS 4/4 from in-view starts), and a close start with the dock partly out of frame no longer drives into it: a blind search is limited to 0.25 m of forward travel (`speed.blind_max_forward_m`) and then backs away (offline closest approach 1.7-2.5 m, over ROS with the real detector 2.28 m and 2.18 m from +30 and -30 deg starts at 3 m, where with the guard off the same +30 deg start ends 0.75 m from the dock and fails; the vehicle is only kept out of the dock in that mode, it does not turn square without a yaw hint).
12. `depth_control` now uses the shared `common/pid.py` (identical numbers to its old private copy: `tests/test_legacy_pid_migration.py`) but is otherwise unchanged and has not run on the real sim since the RPM migration. `waypoint_tracking` was REBUILT on `common/` (section 5.8): the old one was broken offline.
    Its turning circle (~2.5 m, measured offline, independent of speed) means a waypoint closer than that sideways after a corner can never be reached.
13. **`terminal_docking` (alignment + docking) has only run against the offline model** with an ASSUMED camera look. Offline results (wide grid, real detector in the loop, 4 plants x 60 starts): 202/240 first-try passes; from 8-10 m every start passes (84/84, bearings up to +-25 deg, lateral +-2 m, heading +-30 deg); from 6.5-8 m 80/88; from 5-6.5 m only 38/68, because there is no room to line up (the vehicle has no sway thruster and steers by yawing while moving, from rest, with fins that need flow). Over ROS (fake vehicle with realistic odometry, camera node, real detector node): 10/10 first-try passes from 6-9 m starts. **Odometry freezes (5-25 s on the real sim) during the final approach can still ruin an entry**: the controller brakes open loop on a dead-reckoned speed and waits, but 2 of 4 ROS runs with freezes every ~15 s failed (a 32 cm lateral error at the mouth, a 5 cm overshoot of the 1.2 m limit). The thinnest margins among passing runs: wall clearance 4 cm, entry heading 5.0 deg. Nothing about the funnel's real collision behaviour is known: the wall check uses the funnel profile of the STL.
14. **Hold gains were re-tuned for realistic sensing** and are softer than before: a steady 5 N push moves the depth 0.57 m before it is pushed back (the old 60 N/m gains were stiffer but chattered the thrusters 2000+ RPM peak-to-peak once the odometry was 4.5 Hz, 0.25 s late and noisy), a 0.4 m depth step overshoots ~25% (10 cm), and pitch settles in ~25 s. The old ROLL gains oscillated (fins +-20 deg); the new ones pass. The noise levels, latency, freezes and plant variants are assumptions.
15. **Frozen odometry** (the same sample repeating, as seen on the real sim): `station_keeping` and `dof_testing` now publish neutral and forget their integrals (`dof_testing` ends INVALID after 3 s of freeze); before this the pitch ran away to 77 deg in a test.
16. **The live pitch pendulum is explained only as a hypothesis** (`sim_offline/pendulum_analysis.py`): the vessel file has CG and CB equal to 1e-10 m, yet the live vehicle swung between 7 and 85 deg with an 11 s period. A CG-CB mismatch of about 6.5 mm (about 4.7 mm aft and 4.5 mm below, i.e. slightly tail-heavy so it hangs nose-up at about 46 deg, in the model's convention) reproduces exactly that swing in the offline model, and the shipped pitch hold would damp it (closed-loop damping ratio ~0.9 on the ideal rates). It could equally be a different added mass, damping or buoyancy in the live session: only the tilt-and-release experiment printed by the script decides it.
17. **The odometry freezes are not explained.** `sim_viewer/plots/odom_freeze_monitor.py` is the tool to find out which condition (detector subscribed, controller publishing, camera count) changes them; it has not been run on the live sim.
18. **Missions are offline only.** `mission` flies lawnmower, orbit, spiral, yo-yo, go-to, hold and return-home legs through the verified `WaypointTracker`. Everything depends on the assumed 2.5 m turning circle (lane spacing, U-turn arcs, orbit radius, the keep-out margin), fin signs and thrust law. Arc turns need a radius of about 1.2 x the turning circle to be tracked well (offline: a 3 m radius semicircle at 1 m/s derailed the lanes, 4 m did not), so the shipped lanes are 8 m apart. Legs must be chained the way the vehicle is heading (a reversal sweeps a whole turning circle: offline the first example swung out to the dock side and the geofence caught it). The dock keep-out must exceed the closest approach you accept by about 3 m, because the vehicle needs ~5 m to turn back after an abort. Cross-track following is a carrot (lookahead) plus a small PI trim: offline it holds a lane to < 1 m under a steady 1.5 N sideways push (3.4 m without); the push test is a body force, not a real current. The yo-yo's depth extremes are targets (depth acceptance 0.6 m), not settled values. `hold` legs hold depth only (no heading without flow).
19. **The Config tab saves with comments kept only when the edit has a place in the text.** Scalars, short number lists and waypoint / leg lists are rewritten in place; a brand-new section or key, or a list emptied completely, is refused (nothing is written, the temporary copy still works). The result is parsed again and must equal the edited values. The live map, trail and coverage need odometry; the coverage percentage is computed on a 0.5 m grid from the odometry trail and assumes a flat swath of `survey.swath_width_m` centred on the vehicle (the real sensor footprint is not modelled).

20. **Detector robustness and the docking sweep are SYNTHETIC (2026-10-11).** The image degradations (false lights, glint, bubbles, veil, blur, murky water, a hidden light) are my guesses at what the real renderer or water could do; the real mavsim look is not known, so the reliability numbers in section 5.10b say how the detector copes with those guesses, not how it will do on camera_03. The pop-up windows were checked as saved PNGs only.
21. **Docking from every start (2026-10-11), offline, nominal plant, 330 starts, real detector + controller:** **with a rough dock position (`recover.dock_hint_ned` 0.8 m off the truth) 311/330 dock (94%): 52 first try, 259 after one or more planned back-outs; 1 collision; 18 never get into position within 260 s** (median 51 s; by start range 2 m 45/55, 3 m 52/55, 4 m 55/55, 6 m 49/55, 8 m 55/55, 10 m 55/55); every one of the 68 starts with the whole dock in view at t=0 docks (68/68, no collision). **Blind (no position hint): 224/330 dock and 101 COLLIDE** (2 m: 35 of 55, 3 m: 29, 4 m: 20, 6 m: 11, 8 m: 4, 10 m: 2): starting close with the dock out of view, a blind circle swings into it. Old controller on the same kind of grid: the first sweep of this session (before any change) docked 11 of 45 quick-grid starts and hit the dock in 4 of the 15 it could see. Maps: `outputs/docking_sweeps/final_hint2/feasibility_map.png` and `final_blind/`.
    The failures that remain are starts within about 4 m of the dock and 3 m to its side with the nose pointing at or past it; they do not collide but never get into position within 260 s. Starts that need the vehicle to swing round the dock are limited by the ASSUMED 2.4 m turning circle (`recover.turn_radius_m`): measure it on the real sim (yaw step) and put it in `terminal_docking.yaml` and `mission.yaml` before trusting any of the back-out distances. The dock body is assumed to have an outer radius of 1.15 m (`DOCK_OUTER_R` in `docking_sim.py`, only used by the judge).
22. **The search needs a rough dock position to be safe.** Without `recover.dock_hint_ned` the search is blind: it drives a slow circle until the ring appears. From 6-10 m that works, but starting 2-4 m from the dock with the nose pointing away it can swing into it (see the blind numbers in item 21). With a hint (the mission plan knows roughly where the dock is; the sweep uses a hint 0.8 m off) the vehicle first goes to a staging point 7 m in front of the dock mouth, because the lights only shine out of the mouth in an 82 degree cone and cannot be seen from the side.

23. **Vehicle position from odometry or dead reckoning (2026-10-11), offline.** Full 330-start sweep, hinted search, 260 s limit: odometry mode **311/330 dock, 2 collisions**; dead-reckoning mode **300/330 dock, 14 collisions**. Dead reckoning is slightly WORSE in clean conditions: in one traced run its x, y error reached 14 cm during the first turns and was 4 cm at the end, and of its 14 collisions 7 are marginal entries (lateral 0.15-0.18 m against the 0.15 m limit), 4 are entry headings of 6-16 deg (limit 5) and 3 are wall contacts from close starts; tightening `guidance.gate_lateral_m` to 0.06 turned only 4 of the 14 into docks, so the default was left alone. It is BETTER when the odometry position is wrong, but only when the error is large: on the 45-start quick grid a 5 cm/s drift changes nothing (odometry 43/45 with 0 collisions, dead reckoning 43/45 with 1), a 0.7 m jump at 20 s costs the odometry mode 4 docks (39/45 against 43/45); on three single starts with the published position drifting 0.06 / 0.17 / 0.35 m/s the odometry mode docks only after back-outs up to 0.17 m/s (never a first-try pass) and fails at 0.35 m/s, while dead reckoning passes first try up to 0.17 m/s and docks 2 of 3 at 0.35 m/s. A constant odometry offset (3 m, -2 m) changes nothing in either mode (the dock is triangulated in the same frame); with the odometry speed removed neither mode can do the last 2 m (negative control, `test_position_source.py`). So: use `odometry` while the odometry position is trustworthy, `dead_reckoning` if it drifts or jumps; the real odometry's drift, speed and heading quality are unknown, so which one is better on the real sim is unverified. Run-to-run variation: 3 of 330 starts flip between two identical sweeps (the camera noise generator's state depends on the order of runs in a worker), so differences of a few starts are noise.
24. **The Detection tab and the replay camera were checked as offscreen screenshots only** (live offline stack and replay): please look at them once on a real desktop. The replay camera is RE-RENDERED from the recorded pose by the synthetic camera, so it shows what the assumed camera would have seen, not a recorded picture.
25. **Missions with realistic odometry (2026-10-11):** all six (lawnmower, orbit, spiral, yo-yo, go-to + hold, the chained shipped mission) pass without freezes (README 5.8b / `docs/execution.md` 19.4b); with 5-25 s odometry freezes about every 70 s only 3 of 6 pass (the others drift out of the geofence during a freeze, abort, return home and hold). The default lawnmower run-in (2 turning radii) was too short with realistic odometry (first lane 1.9 m off); the shipped mission now uses `leadin_m: 8` (0.65 m).

26. **The side-scan sonar and the rugged sea floor are an ASSUMED model (2026-10-11).** Datasheet geometry (450 kHz, 0.5 x 50 deg, 150 m, bins, gain 0-7) is used, but the echo levels, noise floor, gain steps, speckle, the backscatter per sediment class and the terrain are my assumptions (no real Omniscan data; mavsim itself has no sonar). With the 30 deg-off-nadir mount the swath is only ~1.4 x altitude per side (a 30 m range setting is useful only above ~17 m altitude): lane spacing follows from the altitude. The mosaic assumes a flat bottom for slant-range correction and uses the true vehicle pose (survey-grade navigation), so real mosaics will be worse. Terrain following reacts to the altimeter only (no look-ahead): 1.9 m was the closest to the floor over the rugged terrain.
27. **Sonar replay shows stored pings** (`sidescan.npz`); a replay of a run without that file shows an empty Sonar tab. Pings are not re-rendered from the pose (that would need the same terrain config).

---

## 10. Docker / DDS note

The mavsim **bridge runs in Docker**. On the host, `ros2 topic list` may show camera topics while `ros2 topic hz` or your node receive
nothing. That is usually **Fast-DDS shared memory** failing across the Docker boundary.

`run_live.sh`, `echo_align.sh` and `check_camera.sh` export:
- `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`
- `FASTRTPS_DEFAULT_PROFILES_FILE=…/dock_detection_algo/fastrtps_no_shm.xml`

Use the same profile in any other host terminal that must receive large camera or `DockAlign` traffic from the bridge.
The simulation session must also be **playing**, or odometry stays frozen and the actuators appear to do nothing.

---

## 11. Suggested end-to-end docking flow

1. Start mavsim and the bridge; confirm the cameras and `/Mako_01/odometry_sim` update; stop the teleop node.
2. Pass the `dof_testing` step tests (heave, surge, pitch, yaw, roll, sway) so the signs are verified.
3. Run `./run_live.sh` and tune the mask until T/B/L/R are stable; confirm with `./echo_align.sh`.
4. Use depth hold or waypoint tracking to bring the vehicle in front of the dock; `station_keeping` can hold it there.
5. To go all the way in: with the detector running and the dock in view, start `control_code/terminal_docking_control/run_terminal_docking.sh` (`docs/execution.md` section 16). `control_code/dock_test/run_dock_test.sh` only holds a standoff: with the detector already running, start it. It subscribes to `/Mako_01/dock_align` and publishes `/Mako_01/actuator_cmd` through `common/allocation.py`. Heave follows `elevation_rad`. Pitch follows `error_y` only to keep the lights in frame. Yaw follows `error_x_px` and `lateral_px`, with a small forward surge so the fins have flow. While `valid` is false it follows the `search_*` fields and does not change depth. When the errors are inside the deadbands, surge goes to zero and the vehicle holds that standoff. It does not drive into the funnel.

---

## 12. License, authors, related pieces

- Root project license: **MIT** (`LICENSE`), Copyright (c) 2026 Samyak.
- `control_code/ws/src/interfaces` carries an **Apache-2.0** `LICENSE`, consistent with the mavsim interfaces package it is copied from.
- A classical gate-detection project (SAUVC gates) may exist as a local `Gate-Detection/` folder on a developer machine. It is **gitignored
  and not part of this repository**; the dock-light detector here borrows its idea of a classical mask but uses brightness peaks.

Not in this repo:
- **mavsim** (web simulator) and the **mavsim-controller / mavsim-bridge** Docker stack;
- the vessel files `Mako_01.mavsim` and `sookshma_multi.mavsim` (kept next to this folder);
- optional sibling folders `custom_auv/` and `Papers/` (ignored by git).

If you add or remove a file, update the tree in section 2 and the matching table in section 5 in the same change.

---

## 13. Pitched-view alignment

This section is how the detector behaves when the dock and the vehicle are not at the same depth. The same-depth test is unchanged and is still the last check before the final approach. It is not used while the camera is pitched.

### The four lights

In the simulator the four lamps sit on a circle of radius 1 m in the dock face. Body z points down.

| Light | Dock position [x, y, z] m | What it is |
|-------|---------------------------|------------|
| Top | [0, 0, −1] | upper end of the vertical diameter, bright |
| Bottom | [0, 0, +1] | lower end of that diameter, bright |
| Left | [0, −0.707, −0.707] | 45° up from the center, toward port, dimmer |
| Right | [0, +0.707, −0.707] | 45° up from the center, toward starboard, dimmer |

Top and bottom are 2 m apart. The two side lights are both **above** the dock center. They are not a left–right pair at the center height. Their 3D midpoint lies on the diameter, 14.64% of the way from top to bottom (`FRONTAL_FRACTION` in `dock_geometry.py`).

All four are white, so the detector cannot tell them apart by color. It labels the highest core Top, the lowest core Bottom, and the other two Left and Right. A frame is rejected if either side light does not fall along the segment between top and bottom, because a combined yaw and pitch can make a side light the highest blob.

### Why a pitched view is an ellipse

Seen straight on, with the camera level, the circle projects to a circle. The midpoint of top and bottom is the dock center, and the four distances from that midpoint are equal. `spread_px` near 0 and `aligned` true mean exactly that. This stays true at a **different depth** as long as pitch stays zero and the heading stays perpendicular: the dock is simply higher or lower in the image (`error_y_px`).

Pitching the camera to look up or down at the dock turns the circle into an ellipse. Two things then go wrong if you keep using the old test:

- The midpoint of top and bottom is pulled toward whichever light is closer. It is no longer the dock center.
- The side lights leave the top–bottom radius, so `spread_px` grows even when the heading is already right. At 2 m in front and 2 m below, looking straight at the dock center, that error is large enough to fail the align threshold.

`spread_px` is therefore only trusted again after the depths match and pitch has fallen back to zero.

### What the vehicle can actually do

From `control_code/common/allocation.py`:

- The **axial thruster** produces surge, including in reverse, at zero speed.
- The **two heave thrusters** produce heave (same RPM) and pitch (opposite RPM), at zero speed.
- The **four X-fins** produce yaw and roll, but only while water is flowing over them. Authority fades out below about 0.15 m/s.
- **Nothing produces sway.** A sideways offset is corrected by yawing toward the dock centerline and driving forward. `cross_track` in `control_code/common/loops.py` is already this idea: a lateral error becomes a heading offset, not a sideways force.

### Four lights, camera pitched

Roll from the IMU is removed first, about the image center. The windows show that leveled image, so the markers sit on the lights.

1. **Side midpoint against the top–bottom line (`lateral_px`).** The two side lights are mirrors across the vertical diameter. When the heading is perpendicular to the dock, their image midpoint lies on the top–bottom line and `lateral_px` is about 0. That remains true if the vehicle is shifted sideways but still pointing straight at the dock face: a pure sideways shift does **not** show up in `lateral_px`. It shows up as `error_x_px`, the whole pattern sliding left or right, while `spread_px` stays small. A yaw does the opposite near the dock: the top–bottom line stays vertical (`diameter_angle_deg` stays near 0 on a level camera), and the nearer side light pulls the midpoint off the line. At 8 m a 15° yaw is only about 1 px. At 3 m the same yaw is about 9 px, and `center_exact` goes false once it passes the gate (`max(8 px, 6% of the radius)`).
2. **Dock center.** When `center_exact` is true, `center` is recovered from the cross-ratio of top, the side midpoint, and bottom, using the known 14.64% fraction. `error_x_px` and `error_y_px` use that point. When `lateral_px` is large, the side midpoint is not the 3D midpoint, the cross-ratio is rejected, and `center` falls back to the top–bottom midpoint (`center_exact` false).
3. **Keep the lights in frame with pitch, not with heave.** `error_y_px > 0` means the center is below the middle of the image. Differential heave pitches the nose down just enough to hold it inside the frame. If it is already well inside, hold pitch at 0. This works while stopped.
4. **Match depth with heave, from elevation, not from `error_y_px`.** Once pitch is holding the dock in the middle of the image, `error_y_px` is near zero even if the dock is still 2 m shallower. The heave command is `elevation_rad`: the center’s angle below the horizon. It is the pixel angle of the center below the optical axis, minus the IMU pitch (positive pitch = nose up). Positive elevation means the dock is deeper, so heave down. Negative elevation means heave up. `fy` comes from `hfov_deg` (60°) and the image width; for 640×480 that focal length is about 554 px. `elevation_valid` stays false until the first IMU message, and heave must wait for it.
5. **`obliqueness` is a check, not a command.** It is zero whenever the camera is level, at any depth. It goes negative when the vehicle is below the dock and pitched up to look at it, because the nearer bottom light stretches the lower half of the diameter. Heave uses elevation, not this number.
6. **Get onto the centerline while moving.** There is no sway thruster, so this cannot be done in a hover.
   - `lateral_px` near 0 and `error_x_px` large, with `spread_px` still small: the heading is already perpendicular and the dock is off to one side. Yaw toward `error_x_px` and drive forward on the axial thruster. The fins need that forward speed. While you yaw, you are briefly not perpendicular, so `lateral_px` will grow. Keep surging.
   - `error_x_px` near 0 but `lateral_px` not: you are looking at the dock from the side. A positive `lateral_px` means the side midpoint is to the right of the top–bottom line. Continue forward and yaw to bring `lateral_px` back through zero so the last part of the approach is square.
   - `search_surge_norm` is about +0.35 while either cue is outside its deadband, and 0 once both the center and the heading are square. That field is a hint for the controller. It is not an RPM.
7. **Hand back to the old test.** As heave drives elevation to zero, the pitch needed to see the dock falls back to zero by itself. The ellipse closes into a circle. `spread_px` near 0 then means square-on, and `aligned` is the confirmation for the final approach.

`diameter_angle_deg` is still published. After roll is removed it stays near zero for a pure yaw on a level camera. Do not use it as the main yaw loop. It grows when pitch and yaw happen together, which is a useful sanity check, not the steering command.

### One, two, three, or zero lights

The ellipse sequence runs only with four labelled lights. A missing light cannot be identified: they are the same color, and the side pair is not on the horizontal. `valid` is false, `aligned` is false, elevation is not published, and **heave is not commanded**. Guessing top and bottom from two blobs can drive the vehicle the wrong way in depth.

What is commanded instead is a search that keeps the lights already seen and looks for the others. The other lamps are on the same 1 m circle, so they are near the ones in frame. Pitch uses the heave thrusters and works while stopped. Any yaw keeps `search_surge_norm` positive so the fins have flow.

The order in `PartialAcquire` is:

1. **Split a merged bloom first**, before any motion. If the first pass finds fewer than four cores, `live_dock_lights.py` runs peak finding again with `peak_sep` cut in half (never below 5 px) and keeps the result when it finds more cores. A light hiding inside one glow is a detector miss. Moving the vehicle would throw away a light you already have.
2. **Lights pressed against both the top and the bottom edge** (`backup_taller_than_frame`): the 2 m diameter does not fit. `search_surge_norm = -1` (reverse the axial thruster). This camera’s vertical field of view is about 47° at 60° horizontal and 640×480, so inside about 2.3 m the top and bottom cannot be in the frame together. Pitching only swaps one light for the other.
3. **A light against one edge** (`search_edge`): yaw or pitch toward that edge. A light on the left sets `search_yaw_norm` to −1. A light on the top sets `search_pitch_norm` to −1 (pitch up). Yaw also sets a positive surge hint. Pitch alone does not.
4. **Lights inside the frame but still short of four** (`search_nod`): center their centroid, then nod pitch by ±20° and wiggle yaw by ±8° over an 8 s period. Twenty degrees covers the 2 m pattern beyond roughly 3 m. The nod stops being published as soon as four lights label successfully.
5. **The nod fails and the cluster is large** (`backup_close`): if that interior search has already run for `backup_after_s` (8 s) and the cluster radius is more than 22% of the image height, the dock is too close. Reverse.
6. **No lights at all** (`search_no_lights`): a slow yaw sweep with forward thrust, so the fins can turn the vehicle. Still no heave.

`search_*` is 0 on every frame where `valid` is true, except `search_surge_norm`, which stays at the small forward hint while yaw correction is still needed.

**Viewer layout and redesign (2026-10-11, session 7).** One dark theme (`sim_viewer/app/theme.py`: Qt stylesheet + matplotlib style + shared colours). Window = 3D view (top left), a tab area (bottom left), a side panel (right, resizable splitter):
- bottom-left tabs: **Plots** (sub-tabs Vehicle, Dock detector, Mission, **Pose**, **Actuators** (thrusters, 4 fins with the 25 deg cap, fins-at-cap %, thruster effort), **Tracking** (depth / heading / speed / cross-track error, distance to waypoint, altitude), **Overview** (top view coloured by speed, depth profile, speed, odometry interval); the corner dropdown sets the history 15 s - 5 min), **Detection** (the ONLY place the camera feed is shown: camera + gauges), **Sonar** (sub-tabs *Overview* = narrow waterfall + BIG mosaic, *Mosaic*, *Waterfall*; buttons *Enlarge panel* (gives the tab 80% of the height) and *Pop out mosaic* (own big window)), **3D Trajectory** (`traj3d.py` + `traj3d_panel.py`, VTK offscreen like the main 3D view: flown path coloured by speed / depth / time / altitude, planned path with waypoints from the Config tab, the seabed from `terrain.py`, the side-scan swath footprint, *Load run...* a saved CSV to compare, views iso/top/side, Fit, Save picture; mouse: left = orbit, right/wheel = zoom, middle or Shift+left = pan).
- side panel: **Status** (colour-coded tiles: depth, speed, heading, odometry age, lights, mode + text; optional *Small camera preview* 320x240), **Controls** (collapsible sections: Offline closed-loop stack, Controllers, Live gains, Simulation), Config, Scenarios, History.
- window size = the yaml size but never more than 97% x 95% of the screen; `--size WxH`, `--side-tab`, `--bottom-tab` (any bottom tab or plot sub-tab name) for scripts/screenshots.
- *Start offline stack* also starts the side-scan node (checkbox *Side-scan sonars*); the Sonar tab is selected automatically when the first pings arrive (unless you clicked a bottom tab or passed `--bottom-tab`).
- Cause of the earlier "zoomed in, Controls cut off" window: after the stack started the sonar node, one long status text (the Sonar tab's note) set the window's minimum width above the screen width (a 1872 px window on a 1920 px screen); long labels now use an *Ignored* size policy, explicit minimum sizes pin the panels, and every side tab is in a scroll area.
- Waypoint spin (session 7): the shipped 8-point list (4-8 m depth steps, cruise 10 m/s) made the AUV orbit waypoint 3 for ever: it passed 0.5-2 m beside the point, could not turn back inside its ~2.5 m turning circle (no strafing), and the old rule also required the depth to be within 0.25 m. Now `mission.depth_gate: false` (reached on the horizontal distance; depth keeps settling) and `mission.pass_radius_m: 3.0` (also reached when it came within 3 m and is now moving away), default `cruise_mps` 1.5. Offline: the shipped list completes in ~250 s; the old rule + cruise 10 m/s never finishes (test with a negative control). NOT run on the real sim.
