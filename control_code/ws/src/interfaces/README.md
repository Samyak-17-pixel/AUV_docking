# `interfaces` ROS2 package

> **Keep in sync with mavsim.** These `.msg` files are the wire format between
> the simulator and this bridge, so they are a vendored copy of
> `ros2_ws/src/interfaces` in the mavsim repo, not an independent definition.
> A CI job there fails if the two diverge. Change them upstream, then re-vendor
> — editing only this copy is what previously left `WaveProbe.msg` missing here
> while the bridge imported it, silently disabling wave-probe publishing for
> anyone who built from this directory.

This is the message package your controller needs to **send actuator commands**
to the mavsim bridge. It defines:

| Message | Used for |
|---------|----------|
| `interfaces/Actuator` | actuator commands (`/<vessel>/actuator_cmd`) and, with `--observe-others`, observed actuator state (`/<vessel>/actuator_state`) |
| `interfaces/DVL` | the DVL sensor message (only relevant if you consume DVL data) |
| `interfaces/WaveProbe` | the wave-probe sensor message (wave surface elevation at a world-frame point) |
| `interfaces/DockAlign` | **Docking-repo only** (not mavsim wire format): dock-centering errors from `dock_detection_algo` on `/<vessel>/dock_align` |

`interfaces/Actuator`:

```
std_msgs/Header header
float64[] actuator_values   # values, e.g. [10.0, 1500.0]
string[]  actuator_names    # names,  e.g. ['cs_01', 'th_01']
float64[] covariance        # row-major; send zeros if unused
```

- Control surfaces: `cs_<id>` (zero-padded, e.g. `cs_01`) in **degrees** (-180..180)
- Thrusters: `th_<id>` (e.g. `th_01`) in **RPM** (0..10000)

> The bridge ignores an `actuator_cmd` that has no `actuator_names`, because the
> names are required to map values to the right actuators. Always fill both
> `actuator_names` and `actuator_values`.

## Building it in your ROS2 workspace

You only need this if your controller runs **outside** the bridge container
(i.e. native on the host or in your own container) and uses ROS2 directly.

```bash
# 1. Put this package in a colcon workspace
mkdir -p ~/mavsim_ws/src
cp -r interfaces ~/mavsim_ws/src/

# 2. Build it
cd ~/mavsim_ws
source /opt/ros/humble/setup.bash      # or your ROS2 distro
colcon build --packages-select interfaces

# 3. Source it (do this in every shell that runs your controller)
source ~/mavsim_ws/install/setup.bash
```

Now `from interfaces.msg import Actuator` (Python) or
`#include "interfaces/msg/actuator.hpp"` (C++) works.

## Remember: ROS_DOMAIN_ID

Your controller and the bridge must share the same `ROS_DOMAIN_ID` (and be on
the same machine / Docker network) to discover each other:

```bash
export ROS_DOMAIN_ID=0   # must match the bridge's --ros-domain-id
```
