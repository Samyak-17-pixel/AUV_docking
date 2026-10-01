# Docking

ROS 2 software for **autonomous underwater vehicle (AUV) docking** in the **mavsim** simulator.

The vehicle (**Mako_01**) approaches a **funnel dock** marked by **four lights**. This repository covers:

1. **Motion control** — depth hold, waypoint tracking, per-DOF testing and station keeping via actuator commands (thruster RPM / fin degrees) on `/Mako_01/actuator_cmd`
2. **Dock perception** — detect the four lights, estimate dock center, publish alignment errors
3. **Gate detection (reference, local only)** — classical CV for SAUVC-style gates may exist as a local `Gate-Detection/` folder on disk; it is **gitignored and not published** to this GitHub repo. Dock light perception in `dock_detection_algo/` is inspired by that classical-mask approach but uses brightness peaks instead of vertical poles.

Primary stack: **ROS 2 Humble**, **OpenCV**, **Python**, talking to a running **mavsim-bridge** Docker container (`ROS_DOMAIN_ID=42` by default).

---

## Table of contents

1. [What this project is doing](#1-what-this-project-is-doing)
2. [Repository layout](#2-repository-layout)
3. [Root files](#3-root-files)
4. [`control_code/` — vehicle controllers](#4-control_code--vehicle-controllers)
5. [`dock_detection_algo/` — funnel-dock perception](#5-dock_detection_algo--funnel-dock-perception)
6. [`Gate-Detection/` — SAUVC gate CV reference](#6-gate-detection--sauvc-gate-cv-reference)
7. [ROS topics and messages](#7-ros-topics-and-messages)
8. [Conventions (NED, units, alignment signs)](#8-conventions-ned-units-alignment-signs)
9. [Quick start recipes](#9-quick-start-recipes)
10. [Docker / DDS note (important)](#10-docker--dds-note-important)
11. [Suggested end-to-end docking flow](#11-suggested-end-to-end-docking-flow)

---

## 1. What this project is doing

### The docking problem

- The dock is a **funnel** with **four lights** arranged so that:
  - **Top** and **bottom** lights are the ends of a **vertical diameter**
  - The other two sit roughly **±45°** from the top (left / right)
- The AUV camera must **see the lights**, estimate the **dock center** in the image, and **align the camera center** with that dock center (lateral + vertical), while also checking that the four light–center distances are similar when facing the dock squarely.
- Separately, low-level controllers command **heave / surge / fins** so the vehicle can hold depth and track waypoints in the sim.

### How the folders split the work

| Folder | Role |
|--------|------|
| `control_code/` | Controllers that **subscribe to state** and **publish actuator commands (thruster RPM, fin degrees)** |
| `dock_detection_algo/` | Vision that **subscribes to camera**, shows masks/geometry, **publishes `/…/dock_align`** |
| `Gate-Detection/` | Optional **local-only** gate-pole detector (gitignored; not on GitHub) |

Perception (`dock_align`) and control (`actuator_cmd`) are currently **separate nodes**. A future docking controller would subscribe to `DockAlign` and write actuator commands.

---

## 2. Repository layout

```
Docking/
├── README.md                 ← this file
├── LICENSE                   ← MIT (Samyak, 2026)
├── .gitignore
├── control_code/             ← depth + waypoint control + ROS interfaces package
│   ├── common/               ← shared allocation (thrusters + X-fins), PID, hold loops, vehicle geometry
│   ├── dof_testing/          ← per-DOF open-loop step + closed-loop hold tests
│   ├── station_keeping/      ← hover hold (depth, pitch, surge, heading)
│   ├── sim_offline/          ← simple 6-DOF fake vehicle (no mavsim needed)
│   ├── tests/                ← pytest suite
│   ├── depth_control/
│   ├── waypoint_tracking/
│   └── ws/                   ← colcon workspace (interfaces msgs)
├── dock_detection_algo/      ← live dock-light perception + DockAlign publisher
└── (optional local) Gate-Detection/   ← SAUVC gate CV; gitignored, not pushed
```

> **`Gate-Detection/` is not part of the GitHub repository.** If present on your machine it is ignored by git. Section 6 below documents that local tree when you have it checked out separately.

---

## 3. Root files

| File | Purpose |
|------|---------|
| **`README.md`** | Project documentation for the whole Docking tree (this document). |
| **`LICENSE`** | MIT License, copyright Samyak (2026). |
| **`.gitignore`** | Ignores `Gate-Detection/`, `custom_auv/`, `Papers/`, colcon artifacts under `control_code/ws/`, Python caches, `.DS_Store`. |

---

## 4. `control_code/` — vehicle controllers

**What we do here:** run ROS 2 nodes that read vehicle odometry (and for waypoints, heading) and publish `interfaces/Actuator` commands to the mavsim bridge so thrusters and fins move.

**Layout:**

```
control_code/
├── .gitignore
├── depth_control/
│   ├── depth_control.py
│   ├── depth_control.yaml
│   └── run_depth_control.sh
├── waypoint_tracking/
│   ├── waypoint_tracking.py
│   ├── waypoint_tracking.yaml
│   └── run_waypoint_tracking.sh
└── ws/                         # ROS 2 colcon workspace
    └── src/interfaces/
        ├── CMakeLists.txt
        ├── package.xml
        ├── LICENSE
        ├── README.md
        └── msg/
            ├── Actuator.msg
            ├── DVL.msg
            ├── WaveProbe.msg
            └── DockAlign.msg
```

### 4.1 `control_code/.gitignore`

Ignores local colcon `ws/build`, `ws/install`, `ws/log`, and Python bytecode so build products are not committed.

### 4.2 `depth_control/` — closed-loop depth PID

**Goal:** Drive the AUV to an **absolute NED depth** from YAML, hold within a tolerance for a configured settle time using a PID on the **heave thrusters**, then command **zero** and exit.

| File | Purpose |
|------|---------|
| **`depth_control.py`** | ROS 2 node. Subscribes to `/Mako_01/odometry_sim` (`nav_msgs/Odometry`), runs a PID on depth error `setpoint − z`, maps signed effort to a heave thruster RPM (soft-capped; `heave_sign = -1` because the heave thrusters point UP), publishes `interfaces/Actuator` on `/Mako_01/actuator_cmd`. Surge and fins stay at neutral. On settle or shutdown, publishes neutral and exits. |
| **`depth_control.yaml`** | All tunables: rate, topics, actuator names (`th_02`/`th_03` heave, `th_01` surge, `cs_*` fins), RPM/fin-degree limits, depth `setpoint_m`, PID gains, settle tolerance/time, logging. |
| **`run_depth_control.sh`** | One-shot launcher: sets `ROS_DOMAIN_ID` (default 42), sources Humble + `control_code/ws/install`, runs `depth_control.py` with the YAML beside it. |

**Typical run:**

```bash
cd ~/Docking/control_code/depth_control
./run_depth_control.sh
```

**Prerequisite:** build `interfaces` once (see §4.4). mavsim session must be **playing** (physics advancing) or the vehicle will not move even if commands publish.

### 4.3 `waypoint_tracking/` — multi-waypoint guidance

**Goal:** Follow a list of NED waypoints with combined **heading (fins)**, **surge**, and **depth (heave)** control, all as thruster RPM / fin degrees.

| File | Purpose |
|------|---------|
| **`waypoint_tracking.py`** | ROS 2 node. Subscribes to odometry; for the current waypoint computes heading error → fin yaw mix (deg), surge RPM when aligned, depth PID → heave RPM. Advances when inside XY radius and depth band. Publishes `Actuator` on the configured topic. Zeros/neutral on mission complete or exit. |
| **`waypoint_tracking.yaml`** | Waypoint list, acceptance radii, RPM/degree limits, heading/surge/depth PID gains, fin mix signs, actuator IDs, logging. |
| **`run_waypoint_tracking.sh`** | Sources ROS + interfaces workspace and runs the tracker with its YAML. |

**Typical run:**

```bash
cd ~/Docking/control_code/waypoint_tracking
./run_waypoint_tracking.sh
```

### 4.4 `ws/` — ROS 2 interfaces package (colcon)

**Goal:** Define and build the custom message package **`interfaces`** so Python nodes can `from interfaces.msg import Actuator, DockAlign, …`.

After build, artifacts live in `ws/install/` (gitignored). Source with:

```bash
source /opt/ros/humble/setup.bash
source ~/Docking/control_code/ws/install/setup.bash
```

#### `ws/src/interfaces/`

| File | Purpose |
|------|---------|
| **`package.xml`** | ament package manifest; depends on `std_msgs`, `geometry_msgs`, rosidl generators. |
| **`CMakeLists.txt`** | Generates ROS interfaces from the `.msg` files listed below. |
| **`LICENSE`** | Apache-2.0 (mavsim-style interfaces package license). |
| **`README.md`** | How to build/source this package; notes that **Actuator / DVL / WaveProbe** are meant to stay in sync with mavsim’s wire format, while **`DockAlign` is Docking-repo-only** (not a mavsim bridge message). |

#### `ws/src/interfaces/msg/`

| Message | Purpose |
|---------|---------|
| **`Actuator.msg`** | Actuator command / state: `header`, `actuator_values[]`, `actuator_names[]`, `covariance[]`. Controllers fill names like `th_02`, `cs_04` with RPM (`th_XX`) or degrees (`cs_XX`). Bridge maps by **name**. |
| **`DVL.msg`** | DVL body-frame velocity + covariance (for consumers of DVL; not used by the current depth/waypoint nodes). |
| **`WaveProbe.msg`** | Wave surface elevation at a world point (mavsim sensor; not used by current controllers). |
| **`DockAlign.msg`** | Dock-centering guidance published by `dock_detection_algo`: validity, pixel/normalized errors, radius/spread, T/B/L/R points, confidence, status. See §7. |

**Build:**

```bash
cd ~/Docking/control_code/ws
source /opt/ros/humble/setup.bash
colcon build --packages-select interfaces
source install/setup.bash
```

### 4.5 `common/`, `dof_testing/`, `station_keeping/`, `sim_offline/`

See `CLAUDE.md` for the full description and `execution.md` for the commands.

| Folder | Purpose |
|--------|---------|
| `common/` | `allocation.py` (wrench → thruster RPM + X-fin degrees; fins gain-scheduled on speed²), `pid.py`, `loops.py` (single-axis hold loops), `state.py`, `mako_geometry.yaml` (vehicle data extracted from `Mako_01.mavsim`, with every assumption flagged). |
| `dof_testing/` | `./run_dof_testing.sh --dof surge\|heave\|pitch\|yaw\|roll\|sway --mode step\|hold`. `step` verifies allocation/signs (PASS/FAIL), `hold` verifies/tunes the PID. Tunables: `dof_testing.yaml`. |
| `station_keeping/` | `./run_station_keeping.sh`. Hover: holds depth + pitch (heave thrusters), surge position (axial thruster) and heading (fins, only with flow). Sway cannot be held (no actuator). Tunables with effect/analogy notes: `station_keeping.yaml`. |
| `sim_offline/` | `fake_vehicle.py` publishes `/Mako_01/odometry_sim` from `/Mako_01/actuator_cmd` using a simple model built from the vessel config. Catches sign and gain-structure bugs only; not the real sim. |

---

## 5. `dock_detection_algo/` — funnel-dock perception

**What we do here:** Subscribe to a mavsim **compressed camera** topic, build a **bloom mask** (lights white / background black), find **four light cores** (even when bloom merges at close range), label **Top / Bottom / Left / Right**, estimate **dock center** as the midpoint of the top–bottom diameter, compute **distances** from center to each light and **alignment errors** vs the image center, show two OpenCV windows, and **publish `interfaces/DockAlign`**.

```
dock_detection_algo/
├── dock_light_mask.py              # bloom mask + peak-based cores
├── dock_geometry.py                # T/B/L/R labeling + radius metrics + drawing
├── dock_align_msg.py               # fill DockAlign + topic naming helper
├── live_dock_lights.py             # ROS node + GUI loop + publisher
├── run_live.sh                     # one-command live viewer
├── echo_align.sh                   # one-command ros2 topic echo of DockAlign
├── check_camera.sh                 # verify camera topic is publishing
├── dock_detection_algo_launch.py   # optional ros2 launch wrapper
├── fastrtps_no_shm.xml             # Fast-DDS UDP profile (Docker↔host)
└── requirements.txt                # opencv-python, numpy
```

### File-by-file

| File | Purpose |
|------|---------|
| **`dock_light_mask.py`** | Core vision. Builds HSV **bloom mask** (brightness + optional cyan assist, morphology). Finds cores via **peak mode** (Difference-of-Gaussians + distance-transform peaks + NMS) so four lights still resolve when bloom merges into one blob near the dock; optional legacy **blob-centroid** mode. Returns `(mask, cores)`. |
| **`dock_geometry.py`** | Takes four cores; labels **T** (min y), **B** (max y), **L/R** (remaining by x). Center = midpoint(T,B). Computes `d_top/bottom/left/right`, `radius_tb`, `spread`, side errors. Drawing helpers for camera overlay and mask debug markers. |
| **`dock_align_msg.py`** | `align_topic_from_camera()` maps `/Mako_01/camera_03/...` → `/Mako_01/dock_align`. `build_dock_align_msg()` fills every `DockAlign` field (errors, norms, points, confidence, status). |
| **`live_dock_lights.py`** | Main entry. Opens **Dock camera** and **Bloom mask** windows on the **main thread** (required by OpenCV). Subscribes to `CompressedImage`, runs mask→geometry→HUD, publishes `DockAlign`. Trackbars for online tuning (`V_thresh`, morphology, `PeakMode`, `PeakSep`, `CorePct`, …). Quit with **q** / **Esc**. |
| **`run_live.sh`** | Sets `ROS_DOMAIN_ID`, Fast-DDS no-SHM profile, sources Humble + `control_code/ws/install` (builds interfaces if needed), runs `live_dock_lights.py`. Default camera: `/Mako_01/camera_03/image/compressed`. Override with `TOPIC=...`. |
| **`echo_align.sh`** | Same env/DDS/setup as live; runs `ros2 topic echo` on `/Mako_01/dock_align` by default. Override with `ALIGN_TOPIC=...`. |
| **`check_camera.sh`** | Confirms the camera topic exists and prints `ros2 topic hz` (uses the same DDS profile so Docker→host images are visible). |
| **`dock_detection_algo_launch.py`** | Minimal `ros2 launch` file that `ExecuteProcess`es `live_dock_lights.py` with a `topic` launch argument (no ament package required). Prefer `./run_live.sh` for day-to-day use. |
| **`fastrtps_no_shm.xml`** | Forces Fast-DDS to use **UDP only** (disables shared memory). Without this, the host often **sees** camera topics from the mavsim Docker bridge but receives **no image data**. |
| **`requirements.txt`** | `opencv-python`, `numpy` pins/floors for the vision scripts. |

### Windows and trackbars (live viewer)

| Window | Content |
|--------|---------|
| **Dock camera** | Camera image, T/B/L/R cores, diameter line, center, radii, ALIGN OK/OFF, pixel/norm align errors |
| **Bloom mask** | Binary mask (+ optional T/B/L/R markers) and all tuning trackbars |

| Trackbar | Meaning |
|----------|---------|
| `V_thresh` | HSV Value floor for bloom mask |
| `MinArea` | Min contour area (blob mode / fallback) |
| `Open` / `Close` | Morphology cleanup / fill |
| `CyanAssist` | Include cyan/teal bloom |
| `MaxBlobs` | Cap number of cores (default 4) |
| `PeakMode` | 1 = peak splitting (near-dock); 0 = blob centroids |
| `PeakSep` | Minimum pixel spacing between peaks |
| `CorePct` | Percentile for tight core mask inside bloom |

### Typical run

```bash
# Terminal A — viewer + publisher
cd ~/Docking/dock_detection_algo
./run_live.sh

# Terminal B — guidance topic
cd ~/Docking/dock_detection_algo
./echo_align.sh
```

---

## 6. `Gate-Detection/` — SAUVC gate CV reference

**What we do here:** Classical (non-ML) detection of **two vertical gate poles** for the Singapore AUV Challenge style tasks. This is **not** the funnel-dock light detector; it is included as the prior pipeline that motivated “mask-based / classical” perception. Dock lights reuse the idea of a binary support mask and live viewers, but replace pole histograms with **brightness peaks**.

Full algorithmic documentation already lives in **`Gate-Detection/README.md`** (very detailed). Below is the file map and what each piece does in this Docking monorepo context.

```
Gate-Detection/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── data/README.md
├── gate_detection/          # importable library
│   ├── __init__.py
│   ├── detection_core.py
│   ├── pipeline.py
│   ├── pose_pnp.py
│   ├── temporal.py
│   └── draw.py
└── scripts/                 # CLI entry points
    ├── __init__.py
    ├── _repo.py
    ├── batch_viewer.py
    ├── navigation_gate_detector.py
    ├── qualification_gate_detector.py
    ├── live_gate_detector.py
    └── evaluate_gates.py
```

### Root of `Gate-Detection/`

| File | Purpose |
|------|---------|
| **`README.md`** | Exhaustive docs: algorithm, PnP, CLI flags, tuning, vehicle integration. |
| **`requirements.txt`** | Pinned `opencv-python` and `numpy` for the gate stack. |
| **`pyproject.toml`** | Package metadata for `pip install -e .` (`gate-detection`). |
| **`.gitignore`** | Ignores local datasets (`images/`, videos), venvs, caches. |
| **`data/README.md`** | Where to put untracked image/video datasets; reminds that `--folder` is relative to the Gate-Detection repo root. |

### `gate_detection/` library

| File | Purpose |
|------|---------|
| **`__init__.py`** | Public API exports (`process_frame`, `PipelineConfig`, detectors, pose helpers, temporal filter). |
| **`detection_core.py`** | 2D pole finding: CLAHE, Sobel-x, Canny, optional HSV red/orange boost, morphology, column histogram, peak pairing → states `none` / `one` / `two`. |
| **`pipeline.py`** | One-frame orchestration: detect → optional EMA → horizontal bars → PnP → draw. Defines `PipelineConfig`. |
| **`pose_pnp.py`** | Camera matrix from FOV, rectangle object points, `solvePnP`, plane normal, yaw-error hint, reprojection error. |
| **`temporal.py`** | EMA smoother on left/right pole `x` with jump reset for live video. |
| **`draw.py`** | Overlays: poles, center, status text, optional pose HUD, column-strength debug strip. |

### `scripts/` CLIs

| File | Purpose |
|------|---------|
| **`_repo.py`** | Resolves repo root and puts it on `sys.path` so scripts import `gate_detection` without install. |
| **`__init__.py`** | Makes `scripts` a package. |
| **`batch_viewer.py`** | Shared PNG batch loop + GUI used by navigation/qualification CLIs. |
| **`navigation_gate_detector.py`** | Batch CLI with **no** HSV color boost (structure-only poles). |
| **`qualification_gate_detector.py`** | Batch CLI **with** red/orange HSV boost. |
| **`live_gate_detector.py`** | Webcam / V4L2 live loop with temporal smoothing defaults. |
| **`evaluate_gates.py`** | Offline metrics (state counts, PnP reproj, yaw stats); optional viz. |

**Note:** Put PNG sequences under `Gate-Detection/images/` locally (gitignored). This folder does not talk to mavsim by itself unless you adapt the live script to a ROS image topic.

---

## 7. ROS topics and messages

### Common mavsim topics (examples)

| Topic | Type | Used by |
|-------|------|---------|
| `/Mako_01/odometry_sim` | `nav_msgs/Odometry` | depth / waypoint control |
| `/Mako_01/actuator_cmd` | `interfaces/Actuator` | depth / waypoint publish |
| `/Mako_01/camera_03/image/compressed` | `sensor_msgs/CompressedImage` | dock perception (default) |
| `/Mako_01/dock_align` | `interfaces/DockAlign` | dock perception publish |
| `/dock_02/...` | various | dock vessel / cameras in multi-vessel sessions |

Always match **`ROS_DOMAIN_ID`** to the bridge (default **42**).

### `DockAlign` fields (summary)

| Field | Meaning |
|-------|---------|
| `valid` | Geometry OK (typically 4 labeled lights) |
| `num_lights` | How many cores were found |
| `error_x_px` / `error_y_px` | Dock center minus image center (pixels) |
| `error_x_norm` / `error_y_norm` | Same, scaled by half-width / half-height |
| `radius_px` | Half of top–bottom diameter |
| `spread_px` | Max−min of the four center→light distances |
| `aligned` | Spread small enough vs radius (square-on cue) |
| `diameter_angle_deg` | Top→bottom vs image vertical |
| `center`, `top`, `bottom`, `left`, `right` | Pixel points (`z` unused) |
| `d_*` | Distances center→each light |
| `status`, `confidence` | Human/debug status and 0–1 score |

---

## 8. Conventions (NED, units, alignment signs)

### NED (mavsim / odometry)

- **z positive down**
- Depth setpoint in YAML is absolute NED depth in metres

### Actuator units and signs (this repo's controllers)

| | |
|--|--|
| Thrusters `th_XX` | **RPM**, hardware ±2668, soft cap in each YAML (`rpm_cap`) |
| Fins `cs_XX` | **degrees**, hardware ±35, soft cap `fin_deg_cap` |
| Stop / centred | **0** |
| Heave thrusters (`th_02`, `th_03`) | point **UP** (config orientation pitch = +90°): **positive RPM = up, dive = NEGATIVE RPM** (`heave_sign: -1`) |
| Wrench convention | `[X, Y, Z, K, M, N]`, body frame, **Z positive down**, M positive nose-up, N positive bow-to-starboard |

Actuator **names** must match the vessel config (`th_01`, `th_02`, `th_03`, `cs_04`, …). Empty `actuator_names` are ignored by the bridge.

### DockAlign signs (image: x right, y down)

- **`error_x_px > 0`** → dock is to the **right** of the image center → move / yaw **right** to center it  
- **`error_y_px > 0`** → dock is **below** the image center → dive / pitch accordingly  

---

## 9. Quick start recipes

### One-time: build messages

```bash
cd ~/Docking/control_code/ws
source /opt/ros/humble/setup.bash
colcon build --packages-select interfaces
source install/setup.bash
```

### Depth hold

```bash
cd ~/Docking/control_code/depth_control
./run_depth_control.sh
```

### Waypoints

```bash
cd ~/Docking/control_code/waypoint_tracking
./run_waypoint_tracking.sh
```

### Dock lights + alignment topic

```bash
cd ~/Docking/dock_detection_algo
./run_live.sh          # GUI + publish /Mako_01/dock_align
./echo_align.sh        # another terminal
```

Change camera:

```bash
TOPIC=/Mako_01/camera_04/image/compressed ./run_live.sh
```

### Gate detection (offline PNGs)

```bash
cd ~/Docking/Gate-Detection
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/navigation_gate_detector.py --folder images/your_folder
```

---

## 10. Docker / DDS note (important)

The mavsim **bridge runs in Docker**. On the host, ROS 2 **topic list** may show camera topics while **`ros2 topic hz` / your node receive nothing**. That is usually **Fast-DDS shared memory** failing across the Docker boundary.

`dock_detection_algo/run_live.sh`, `echo_align.sh`, and `check_camera.sh` export:

- `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`
- `FASTRTPS_DEFAULT_PROFILES_FILE=…/fastrtps_no_shm.xml`

Use the same profile in any other host terminal that must receive large camera (or `DockAlign`) traffic from the bridge.

Also: the simulation session must be **running / not paused**, or odometry stays frozen and actuators appear to “do nothing.”

---

## 11. Suggested end-to-end docking flow

1. Start mavsim + bridge; confirm cameras and `/Mako_01/odometry_sim` update.  
2. Run **`./run_live.sh`** and tune mask/peaks until T/B/L/R are stable; confirm **`./echo_align.sh`**.  
3. Use **depth control** (and/or teleop) to bring the vehicle to a working depth where the dock lights are visible.  
4. A future **docking controller** (not yet in-repo) would:
   - subscribe to `/Mako_01/dock_align`
   - if `!valid` → search / hold
   - drive lateral/vertical from `error_x_*` / `error_y_*`
   - use `aligned` / `spread_px` before closing distance
   - publish actuator commands on `/Mako_01/actuator_cmd`
5. **Waypoint tracking** remains available for transit legs before the visual docking phase.

---

## Authors / license

- Root project license: **MIT** (`LICENSE`), Copyright (c) 2026 Samyak.  
- `control_code/ws/src/interfaces` carries an **Apache-2.0** file consistent with the mavsim interfaces package.  
- Gate detection originated as part of **Team Aritra** SAUVC work (see `Gate-Detection/README.md`).

---

## Related external pieces (not in this repo)

- **mavsim** web sim + **mavsim-controller / mavsim-bridge** Docker stack  
- Vessel configs (e.g. Mako thruster IDs) living in the sim / bridge handshake  
- Optional sibling folders on the machine (`custom_auv/`, `Papers/`) ignored by git here  

If something is missing from this map after you add new files, update this README in the same PR so the monorepo stays navigable.
