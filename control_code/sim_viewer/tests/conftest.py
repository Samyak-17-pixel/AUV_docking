import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                       # control_code/
for p in (HERE.parent, *sorted(d for d in HERE.parent.iterdir() if d.is_dir() and d.name not in ("tests", "scripts", "__pycache__")), ROOT / "common", ROOT / "sim_offline", ROOT.parent / "dock_detection_algo"):
    sys.path.insert(0, str(p))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")      # Qt tests run without a window system
