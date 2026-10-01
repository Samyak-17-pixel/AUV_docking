import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("common", "sim_offline", "dof_testing", "station_keeping"):
    sys.path.insert(0, str(ROOT / sub))
