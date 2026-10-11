"""Copy a controller yaml into a test folder. The shipped waypoint_tracking.yaml is the USER's file (they edit its waypoints and speeds), so the tests must not depend on it:
for that one file the waypoint list is replaced by the 5x8 m rectangle at 3 m depth and the cruise speed by 1.0 m/s (text surgery: the comments stay)."""

import re
import shutil
from pathlib import Path

RECT = ["  - {x: 5.0, y: 0.0, z: 3.0}", "  - {x: 5.0, y: 8.0, z: 3.0}", "  - {x: 0.0, y: 8.0, z: 3.0}", "  - {x: 0.0, y: 0.0, z: 3.0}"]


def copy_config(src: Path, dst: Path) -> None:
    if src.name != "waypoint_tracking.yaml":
        shutil.copy(src, dst)
        return
    out, in_block = [], False
    for line in src.read_text().splitlines():
        if re.match(r"^  - \{x:", line):
            if not in_block:
                out += RECT
                in_block = True
            continue
        in_block = False
        out.append(re.sub(r"cruise_mps: [0-9.]+", "cruise_mps: 1.0", line) if line.lstrip().startswith("cruise_mps:") else line)
    dst.write_text("\n".join(out) + "\n")
