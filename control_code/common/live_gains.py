"""Change controller gains WHILE they run: a ROS helper for the controller nodes and the message format the viewer's sliders send. Needs rclpy only in `attach`.

Topic /<vessel>/ctrl_gains, std_msgs/String, JSON, one change per message:
  {"controller": "terminal_docking", "loop": "heave", "key": "kp", "value": 12.0}
`controller` (optional) must match the receiving node's name when given, so a slider aimed at one controller does not retune another.
The node applies it to its HoldLoops (common/loops.py) and answers on /<vessel>/ctrl_debug-independent logging only (no reply topic).
"""

from __future__ import annotations

import json
from typing import Callable, Optional


def make_message(controller: str, loop: str, key: str, value: float) -> str:
    return json.dumps({"controller": controller, "loop": loop, "key": key, "value": float(value)})


def apply_message(data: str, controller: str, loops) -> Optional[str]:
    """Apply a ctrl_gains message to `loops` (a HoldLoops). Returns a human-readable line when something changed, else None."""
    try:
        d = json.loads(data)
        if not isinstance(d, dict):
            return None
        who = d.get("controller")
        if who not in (None, "", controller):
            return None
        loop, key, value = str(d["loop"]), str(d["key"]), float(d["value"])
    except (ValueError, KeyError, TypeError):
        return None
    old = loops.gains.get(loop, {}).get(key)
    if not loops.set_gain(loop, key, value):
        return f"rejected live gain {loop}.{key} = {value!r}"
    return f"live gain {loop}.{key}: {old} -> {value:g}"


def attach(node, vessel: str, controller: str, loops, log: Optional[Callable[[str], None]] = None) -> None:
    """Subscribe `node` to /<vessel>/ctrl_gains and route messages into `loops`."""
    from std_msgs.msg import String

    def on_msg(m: String) -> None:
        line = apply_message(m.data, controller, loops)
        if line and log:
            log(line)

    node.create_subscription(String, f"/{vessel}/ctrl_gains", on_msg, 10)
