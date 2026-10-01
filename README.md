# AUV Docking

ROS 2 software for **autonomous underwater vehicle (AUV) docking** in the **mavsim** simulator.

The vehicle (**Mako_01**, a torpedo-shaped AUV) has to find and approach a **funnel dock** marked by **four lights**. This
repository contains everything we have built towards that goal:

| Area | What it does | Folder |
|------|--------------|--------|
| **Actuator allocation** | Turns a wanted force/moment into thruster RPM and fin angles for Mako_01 | `control_code/common/` |
| **Per-DOF testing** | Tests and tunes surge, heave, pitch, yaw, roll and sway one at a time | `control_code/dof_testing/` |
| **Station keeping** | Hovers: holds depth, pitch, surge position and heading | `control_code/station_keeping/` |
| **Basic controllers** | Depth hold and waypoint following | `control_code/depth_control/`, `control_code/waypoint_tracking/` |
| **Offline test vehicle** | A simple fake Mako_01 so the controllers can run without the simulator | `control_code/sim_offline/` |
| **ROS messages** | The `interfaces` package (`Actuator`, `DockAlign`, `DVL`, `WaveProbe`) | `control_code/ws/` |
| **Dock perception** | Finds the four lights in a camera image and publishes how far off-center the dock is | `dock_detection_algo/` |

**What is still missing:** a docking controller that reads `/Mako_01/dock_align` and drives the actuators. Perception and
control are separate programs today (see section 9).

Primary stack: **ROS 2 Humble**, **Python 3.10**, **OpenCV**, **NumPy**, talking to a running **mavsim-bridge** Docker
container (`ROS_DOMAIN_ID=42`).

> **How to run things:** this README explains *what each file is*. The step-by-step run instructions for every task are in
> [`execution.md`](execution.md).

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
   - [5.5 `control_code/sim_offline/`](#55-control_codesim_offline--fake-vehicle)
   - [5.6 `control_code/tests/`](#56-control_codetests--pytest-suite)
   - [5.7 `control_code/depth_control/`](#57-control_codedepth_control--depth-pid)
   - [5.8 `control_code/waypoint_tracking/`](#58-control_codewaypoint_tracking--waypoint-follower)
   - [5.9 `control_code/ws/`](#59-control_codews--ros-2-interfaces-package)
   - [5.10 `dock_detection_algo/`](#510-dock_detection_algo--dock-light-perception)
6. [ROS topics and messages](#6-ros-topics-and-messages)
7. [How the pieces fit together (data flow)](#7-how-the-pieces-fit-together-data-flow)
8. [Testing](#8-testing)
9. [Known limits and unverified assumptions](#9-known-limits-and-unverified-assumptions)
10. [Docker / DDS note](#10-docker--dds-note)
11. [Suggested end-to-end docking flow](#11-suggested-end-to-end-docking-flow)
12. [License, authors, related pieces](#12-license-authors-related-pieces)

---

## 1. Quick start

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
   `/Mako_01/actuator_cmd` and would fight our controllers). See `execution.md` section 3.
4. Check the actuator directions before trusting any controller:
   ```bash
   cd control_code/dof_testing
   ./run_dof_testing.sh --dof heave --mode step
   ```
5. No simulator? Run everything against the fake vehicle: `execution.md` section 10.
6. Run the unit tests (no ROS needed): `cd control_code && python3 -m pytest tests -q` (37 tests).

---

## 2. Repository tree (every file)

```
AUV_docking/
├── README.md                          this file
├── execution.md                       step-by-step run guide for every task
├── LICENSE                            MIT license (root project)
├── .gitignore                         files git must not track (build output, caches, local-only folders)
│
├── control_code/
│   ├── .gitignore                     ignores colcon output, caches inside control_code
│   │
│   ├── common/                        shared building blocks (used by dof_testing, station_keeping, sim_offline, tests)
│   │   ├── allocation.py              wrench -> thruster RPM + fin degrees (and the inverse)
│   │   ├── pid.py                     PID with integral clamp and derivative-on-measurement
│   │   ├── loops.py                   single-axis hold loops (heave, pitch, surge, speed, yaw, roll, cross-track)
│   │   ├── state.py                   vehicle state from odometry (position, Euler angles, body velocities)
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
│   ├── sim_offline/                   fake vehicle for testing without mavsim
│   │   ├── vehicle_model.py           6-DOF rigid-body model of Mako_01 (pure Python)
│   │   └── fake_vehicle.py            ROS node: actuator_cmd in, odometry_sim out
│   │
│   ├── tests/                         pytest suite (37 tests)
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
│   │   ├── waypoint_tracking.py       ROS node
│   │   ├── waypoint_tracking.yaml     waypoints, heading/surge/depth gains, limits
│   │   └── run_waypoint_tracking.sh   launcher
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
    ├── dock_geometry.py               top/bottom/left/right labelling, center, distances, drawing
    ├── dock_align_msg.py              builds the DockAlign message
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
`Gate-Detection/`, and a local `CLAUDE.md` notes file.
Runtime logs are written outside the repo: `~/dof_testing_logs/` and `~/station_keeping_logs/` (CSV files).

---

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
`Mako_01.mavsim` gives both heave thrusters the orientation pitch = +90°, which points their thrust axis along body −z, i.e. **up**.
So **positive RPM = up, and diving needs NEGATIVE RPM**. The allocator handles this (a positive Z/dive demand becomes negative
heave RPM) and the older controllers use `heave_sign: -1.0` in their YAML.

### Image convention for `DockAlign`
Image x points right and y points down. `error_x_px > 0`: the dock is to the **right** of the image center.
`error_y_px > 0`: the dock is **below** the image center.

---

## 4. The vehicle: Mako_01

Values come from `Mako_01.mavsim` (a zip archive; the data is in `config.json`, `agents[0]`). Items marked *assumption* are not in
that file and are collected in `control_code/common/mako_geometry.yaml`.

| Property | Value |
|---|---|
| Mass / buoyancy mass | 10.0 kg / 10.74 kg, so **+7.25 N net buoyancy: it floats up by itself** |
| Centre of gravity vs buoyancy | identical, so there is **no passive righting moment** in pitch or roll |
| Damping (config) | only surge, sway, yaw (0.01); **none in heave, roll, pitch**, so the PID derivative terms must provide damping |
| Axial thruster `th_01` | x = −0.688 m, thrust along body x, propeller diameter 0.10 m |
| Heave thrusters `th_02`, `th_03` | x = +0.349 m and −0.349 m, thrust along body −z (up), diameter 0.06 m |
| Fins `cs_04`, `cs_06`, `cs_07`, `cs_08` | X configuration at the aft, roll angles 45°, 135°, 225°, −45°, ±35° travel, area 0.00473 m² |
| Thrust curve | symmetric forward and reverse |
| Start pose | NED (0, 0, 3) |
| Dock (`dock_02`) | NED (10, 0, 3), heading 180° |
| Cameras published | `camera_03`, `camera_04`, `camera_05` (on the live sim) |

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
| `execution.md` | The run guide: terminal setup, starting the simulator, and numbered steps with expected output for every task, plus troubleshooting. |
| `LICENSE` | MIT license for the root project (Copyright 2026 Samyak). |
| `.gitignore` | Tells git to skip `custom_auv/`, `Papers/`, `Gate-Detection/`, colcon output (`control_code/ws/build|install|log`), `__pycache__`, `*.pyc`, `.DS_Store` and `CLAUDE.md`. |

### 5.2 `control_code/common/` — shared building blocks

The code that decides *which actuator does what*. Used by the test harness, station keeping, the offline vehicle and the tests.

| File | Role |
|------|------|
| `allocation.py` | **`Allocator`** converts a wrench `[X,Y,Z,K,M,N]` into actuator commands, and back. Thrusters realise X, Z and M: it builds the thruster matrix from position and direction, inverts it (pseudo-inverse) and converts each force to RPM with `T = KT·ρ·D⁴·n²`. Fins realise K and N: it builds a 3×4 mixing matrix from each fin's lift direction and moment arm, divides by speed² (**gain scheduling**, because fin force grows with speed²), fades the fins out below `u_fin_off_mps` where they have no authority, and caps the result. Key methods: `allocate(wrench, u)`, `thrusters_from_wrench`, `fins_from_moments`, `force_to_rpm`, `thrust_n`, `fin_authority`, and the inverses `wrench_from_rpm` / `wrench_from_fins` (used by the offline vehicle). Also `eul_to_rotm` (ZYX rotation matrix) and `load_geometry`. |
| `pid.py` | **`Pid`**: proportional-integral-derivative controller with an integral clamp expressed in output units, a low-pass filter on the derivative, and **derivative-on-measurement** (pass the measured rate and no kick appears when the setpoint jumps). Also `clamp`. |
| `loops.py` | **`HoldLoops`**: the single-axis hold loops built on `Pid`: `heave` (depth → Z force, includes the buoyancy feed-forward), `pitch` (→ M moment), `surge` (position along an axis → X force), `speed` (forward speed → X force), `yaw` (heading → N moment), `roll` (→ K moment), `cross_track` (lateral error → heading offset, used to produce sway). Error is always `setpoint − measurement`. |
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

Results: `PASS`, `FAIL` or `ABORT` (a safety limit tripped), plus a CSV in `~/dof_testing_logs/`.

### 5.4 `control_code/station_keeping/` — hover hold

Captures the pose when it starts and holds it: **depth** and **pitch** with the heave thrusters, **surge position** along the start
heading with the axial thruster (forward and reverse), **heading** with the fins. Sideways (sway) drift is **not** held (no sway
actuator), and heading is only held while there is flow over the fins (above about 0.3 m/s).

| File | Role |
|------|------|
| `station_keeping_core.py` | **`StationKeeper`** (pure Python, no ROS): `feed_capture` averages the first seconds of state and latches the hold point, `capture_command` holds the buoyancy trim while that happens, `update` runs the loops with dead-bands and slew limits and returns actuator commands plus a status dict, `safety_reason` checks depth, attitude, leash distance. `shrink` implements the dead-band. |
| `station_keeping.py` | ROS node around the engine: subscribes to odometry, publishes actuator commands, stops to neutral on a safety trip or stale odometry, logs a status line every 2 s and writes a CSV to `~/station_keeping_logs/` on exit. |
| `station_keeping.yaml` | **Every tunable**, each with a comment on what changing it does, a quantified example and an analogy. Sections: `node`, `topics`, `capture`, `enable` (switch loops on/off while commissioning), `gains`, `feedforward`, `deadband`, `slew`, `limits`, `safety`, `logging`. The overshoot and settling numbers in the comments were measured on the offline vehicle, so treat them as direction and relative size, not absolute values. |
| `run_station_keeping.sh` | Sources ROS and the workspace and runs `station_keeping.py` with the YAML. |

### 5.5 `control_code/sim_offline/` — fake vehicle

A small physics model so controllers can be exercised without mavsim. It catches sign mistakes and gain-structure problems; it is
**not** the real simulator (see section 9).

| File | Role |
|------|------|
| `vehicle_model.py` | **`VehicleModel`** (pure Python): 6-DOF rigid body with mass and inertia, +7.25 N net buoyancy, linear and quadratic drag, simple added mass, thruster thrust ∝ RPM² with a 0.2 s lag, fin force ∝ speed² with a rate limit, and an external-wrench input `ext` for pushes and currents. Methods: `set_command`, `step`, `quaternion`. |
| `fake_vehicle.py` | ROS node around the model: subscribes to `/Mako_01/actuator_cmd`, steps the model at 100 Hz and publishes `/Mako_01/odometry_sim`. A command older than 1 s counts as zero (like the real bridge). Options: `--x --y --z --roll --pitch --yaw --rate --cmd-timeout --vessel`. Run it in its own ROS domain (77) so it cannot touch the real bridge. |

### 5.6 `control_code/tests/` — pytest suite

No ROS and no simulator needed. `cd control_code && python3 -m pytest tests -q` gives **37 passed** in about 3 s.

| File | Role |
|------|------|
| `conftest.py` | Adds `common/`, `sim_offline/`, `dof_testing/` and `station_keeping/` to the import path. |
| `test_allocation.py` | Heave thruster axis points up; dive needs negative RPM on both; nose-up moment drives the forward heave thruster up; surge uses only the axial thruster and can reverse; unreachable DOFs are ignored by the thrusters; RPM ↔ force round trip and cap; fin roll/pitch/yaw modes are orthogonal; fin moment round trip; fin force scales with speed² and fades out without flow; fin cap. |
| `test_pid.py` | Proportional term and output clamp; integral clamp; derivative-on-measurement (no setpoint kick); reset. |
| `test_offline_loops.py` | Offline vehicle floats up without thrust and dives with negative heave RPM; **every DOF passes in both modes** (12 cases); a deliberately wrong-sign actuator is caught by the step test; station keeping holds still water, rejects a vertical push and a current, shows that a low RPM cap cannot hold depth, shows heading is not held without flow, dead-band shrink, safety leash and depth limits, capture-window buoyancy trim. |

### 5.7 `control_code/depth_control/` — depth PID

Holds an absolute depth with a PID on the heave thrusters, then zeroes the actuators and exits after the depth has stayed within the
tolerance for the settle time. *Older controller: it has not yet been run on the real simulator since the switch to RPM.*

| File | Role |
|------|------|
| `depth_control.py` | ROS node: subscribes to `/Mako_01/odometry_sim`, PID on `setpoint − z`, publishes heave RPM (`heave_sign = −1` so a dive is negative RPM) on `/Mako_01/actuator_cmd`. Surge and fins stay at zero. Has its own copy of the PID class. |
| `depth_control.yaml` | Sections: `node`, `topics`, `actuators` (names and roles), `depth` (`setpoint_m`), `pid` (gains, `i_max`, derivative filter), `limits` (`rpm_cap`, `fin_deg_cap`, `heave_sign`, hold values), `behaviour` (settle tolerance and time, wait for odometry, zero on exit), `logging`. |
| `run_depth_control.sh` | Sources ROS and the workspace and runs the node with the YAML. |

### 5.8 `control_code/waypoint_tracking/` — waypoint follower

Steers through a list of NED waypoints: heading error → fins (yaw mix), surge RPM when roughly aligned, depth PID → heave thrusters.
*Older controller: not yet run on the real simulator since the switch to RPM.*

| File | Role |
|------|------|
| `waypoint_tracking.py` | ROS node. For the current waypoint it computes the heading error, a fin yaw command from a heading PID, a surge RPM that depends on how well it is aligned, and a depth PID for the heave thrusters. A waypoint is reached when inside `acceptance_radius_m` and `depth_acceptance_m`. Optionally reads `vessel_state` instead of odometry. Ends with neutral actuators when the mission is complete. |
| `waypoint_tracking.yaml` | Sections: `node`, `topics`, `actuators` (including `fin_yaw_signs`), `waypoints` (a 5 m × 3 m rectangle at depth 3 m by default, stopping short of the dock at (10, 0, 3)), `mission`, `heading`, `surge`, `depth_pid`, `limits`, `logging`. |
| `run_waypoint_tracking.sh` | Sources ROS and the workspace and runs the node with the YAML. |

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

Subscribes to a mavsim **compressed camera** topic, builds a **bloom mask** (lights white, background black), finds **four light
cores** (even when the glow merges into one blob at close range), labels them **Top / Bottom / Left / Right**, takes the **dock
center** as the midpoint of top and bottom, measures distances and **alignment errors** against the image center, shows two
OpenCV windows and **publishes `interfaces/DockAlign`**.

Pipeline: `BGR image → HSV → bloom mask (brightness + optional cyan assist + morphology) → cores = local peaks of a difference of
Gaussians fused with a distance transform, with non-maximum suppression → T/B/L/R labels → center, radius, spread → DockAlign`.

| File | Role |
|------|------|
| `live_dock_lights.py` | Main program. Node **`DockLightsLive`** subscribes to the camera, publishes `DockAlign`; the main thread runs the two OpenCV windows (**Dock camera**: cores, labels, center, errors, align hint; **Bloom mask**: the mask and tuning trackbars). Trackbars start from `dock_detection.yaml`. Keys: **p** prints the current trackbar values as YAML, **q** or **Esc** quits. Options: `--topic`, `--align-topic`, `--config`. |
| `dock_light_mask.py` | Core vision. `bloom_mask_and_cores(...)` builds the mask (`_build_bloom_mask`) and finds cores either as **peaks** (`_cores_from_peaks`: difference of Gaussians + distance transform + `_nms_peaks`, needed when the lights merge) or as **blob centroids** (`_cores_from_contours`). `draw_cores` overlays markers. All constants are parameters. |
| `dock_geometry.py` | `label_dock_lights` assigns Top (smallest y), Bottom (largest y), then Left/Right by x. `evaluate_dock_geometry` returns a `DockGeometry` (center = midpoint of top/bottom, distances to the four lights, radius, spread = largest minus smallest distance, side errors). `draw_dock_geometry` and `draw_mask_debug` draw the overlays. |
| `dock_align_msg.py` | `build_dock_align_msg` fills every `DockAlign` field (errors in pixels and normalised, radius, spread, `aligned`, diameter angle, points, confidence). `align_topic_from_camera` maps `/Mako_01/camera_03/...` to `/Mako_01/dock_align`. |
| `dock_detection_config.py` | `load_config` reads `dock_detection.yaml`; `detector_kwargs` turns it into the keyword arguments of `bloom_mask_and_cores`. |
| `dock_detection.yaml` | **All detection tunables**, each commented with what it does and roughly how much to change it. Sections: `camera` (topic names), `mask` (`v_thresh`, cyan band, `open_k`, `close_k`, `min_area`, `max_blobs`), `peaks` (`peak_mode`, `peak_sep`, `core_pct`, DoG sigmas, thresholds, refinement window…), `alignment` (`spread_align_frac`, `spread_align_min_px`, `confidence_base`), `hud` (display only). |
| `dock_detection_algo_launch.py` | Optional `ros2 launch` wrapper with `topic` and `config` arguments. |
| `run_live.sh` | One-command launcher: sets the domain and the Fast-DDS UDP profile, sources ROS and the workspace (builds `interfaces` if missing), runs `live_dock_lights.py`. Overrides: `TOPIC`, `CONFIG`, `ALIGN_TOPIC`. |
| `echo_align.sh` | Prints `/Mako_01/dock_align` (`ros2 topic echo`); extra arguments are passed through (for example `--once`). Override the topic with `ALIGN_TOPIC`. |
| `check_camera.sh` | Checks that a camera topic exists and receives data (`ros2 topic hz`). Set `TOPIC`; without it, it defaults to `/dock_02/camera_02/image/compressed`. |
| `fastrtps_no_shm.xml` | Fast-DDS profile that forces **UDP** and disables shared memory (see section 10). |
| `requirements.txt` | `opencv-python`, `numpy`, `pyyaml`. |

---

## 6. ROS topics and messages

### Topics used (vessel `Mako_01`)

| Topic | Type | Direction | Used by |
|-------|------|-----------|---------|
| `/Mako_01/odometry_sim` | `nav_msgs/Odometry` (`frame_id NED`, `child_frame_id BODY`, ~4 Hz on the live sim) | read | all controllers, `fake_vehicle` writes it offline |
| `/Mako_01/actuator_cmd` | `interfaces/Actuator` | write | all controllers, read by the bridge |
| `/Mako_01/camera_03/image/compressed` | `sensor_msgs/CompressedImage` | read | dock detector (default camera; `camera_04` and `camera_05` also exist) |
| `/Mako_01/dock_align` | `interfaces/DockAlign` | write | dock detector |
| `/Mako_01/vessel_state` | `std_msgs/Float64MultiArray` | read | optional input of `waypoint_tracking` |
| `/Mako_01/imu_01/data`, `/Mako_01/dvl_03/data` | sensor data | not used by our code | |

Always match `ROS_DOMAIN_ID` to the bridge (default **42**). Only one program should publish `/Mako_01/actuator_cmd` at a time; the
bridge's own **`mavsim_teleop` node publishes zeros on it at 20 Hz** and must be stopped first (`execution.md` section 3.4).

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
| `valid` | geometry is usable (all four lights found and labelled) |
| `num_lights` | number of cores found |
| `error_x_px`, `error_y_px` | dock center minus image center, in pixels (`+x` = dock right, `+y` = dock below) |
| `error_x_norm`, `error_y_norm` | the same, divided by half the image width / height |
| `image_width`, `image_height` | frame size |
| `radius_px` | half the top–bottom distance |
| `spread_px` | largest minus smallest of the four center-to-light distances (0 when seen square-on) |
| `err_left_px`, `err_right_px` | left/right light distance minus the top–bottom radius |
| `aligned` | `spread_px` below `max(spread_align_min_px, spread_align_frac × radius_px)` |
| `diameter_angle_deg` | angle of the top→bottom line against image vertical (0 = upright) |
| `center`, `top`, `bottom`, `left`, `right` | pixel positions (`z` unused) |
| `d_top`, `d_bottom`, `d_left`, `d_right` | distance from the center to each light, pixels |
| `status` | `ok`, `ok_not_aligned`, or the reason it is invalid |
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
 (no consumer yet)        └────────────────────────────────────────────────────────────────────────────────────┘

 Offline: sim_offline/fake_vehicle.py replaces the simulator box (it reads actuator_cmd and publishes odometry_sim).
```

---

## 8. Testing

| Level | How | What it proves |
|-------|-----|----------------|
| Unit and offline system tests | `cd control_code && python3 -m pytest tests -q` (37 tests) | Allocation signs, PID behaviour, every DOF test passes offline, wrong-sign actuators are caught, station keeping holds against pushes |
| Offline end-to-end over ROS | `sim_offline/fake_vehicle.py` plus any controller, in ROS domain 77 (`execution.md` section 10) | The ROS nodes and topics work together |
| Real simulator | `dof_testing` (`execution.md` section 7), then station keeping | The only test that confirms signs, thrust and fin behaviour on the real vehicle |

---

## 9. Known limits and unverified assumptions

These are the things to check on the first real-simulator run.

1. **Control code has so far been tested only against the offline model**, not the real simulator. `depth_control` and
   `waypoint_tracking` have not been run on the simulator since the switch to RPM and degrees.
2. **Fin deflection sign** is assumed (positive deflection gives force along the fin's lift direction). Run `dof_testing` for yaw and roll
   in `step` mode: a wrong sign shows up as FAIL.
3. **Fin moment arm:** the vessel file gives fin x = 0, but the fin geometry lies about 0.42–0.49 m aft; the code uses **−0.454 m**. The lift slope
   (`cl_alpha_per_rad: 2.0`) is a guess.
4. **Thrust units:** the allocator assumes `T = KT·ρ·D⁴·n²` with `n = RPM / 60`. If the real thrust is far off, change the single value
   `rpm_to_rps` in `mako_geometry.yaml`.
5. **Odometry twist frame** is assumed to be BODY (the message says `child_frame_id: BODY`, but it has not been proven with a motion
   test). Odometry arrives at only ~4 Hz.
6. **`rpm_cap: 1000`** in `depth_control.yaml` and `waypoint_tracking.yaml` is marginal: with the thrust assumption above, the two heave
   thrusters give only about 7 N at 1000 RPM against a net buoyancy of 7.25 N. `dof_testing` and `station_keeping` use 1800.
7. **Offline model:** drag is assumed (quadratic values chosen to give plausible speeds). The overshoot and settling numbers quoted in the YAML
   comments come from this model: trust the direction and relative size, not the absolute values.
8. **Yaw and roll authority needs flow:** station keeping cannot hold heading in still water, and sideways drift cannot be corrected at all.
9. **Pitch and roll step tests are short pulses** on purpose: the vehicle has no righting moment and no damping in those axes.
10. **Dock detection** was checked on a synthetic four-light image and must still be tuned on the real camera stream.
11. **No docking controller yet:** nothing consumes `DockAlign`.
12. The older controllers each keep their own copy of the PID class (the shared one is in `common/pid.py`).

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
5. A future **docking controller** (not in this repo yet) would:
   - subscribe to `/Mako_01/dock_align`;
   - if `valid` is false, search or hold;
   - steer from `error_x_*` (yaw) and `error_y_*` (depth/pitch);
   - check `aligned` and `spread_px` before closing the distance;
   - publish actuator commands on `/Mako_01/actuator_cmd`, reusing `common/allocation.py` and `common/loops.py`.

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
