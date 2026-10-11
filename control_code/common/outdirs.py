"""Where every tool writes its files: ONE folder inside the repository, `AUV_docking/outputs/`. Nothing is ever written to the Home directory.

  out_dir("sim_viewer_runs")        -> <repo>/outputs/sim_viewer_runs   (created on demand by the caller)
  out_dir("~/dof_testing_logs")     -> <repo>/outputs/logs/dof_testing  (a leading '~/' from an old config is mapped INTO outputs/, not to Home)
  out_dir("/some/absolute/path")    -> that path (only if you spell out an absolute path yourself)
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

OUT_ROOT = Path(__file__).resolve().parents[2] / "outputs"


def out_dir(name: Union[str, Path]) -> Path:
    s = str(name).strip()
    if s.startswith("~"):
        s = s.lstrip("~").lstrip("/\\")
    if s.startswith("outputs/"):
        s = s[len("outputs/"):]
    p = Path(s)
    return p if p.is_absolute() else OUT_ROOT / p


def tmp_dir() -> Path:
    """Scratch space for temporary files (edited-config copies, confirm-run configs): `outputs/tmp/`, never /tmp or Home. Created on demand."""
    d = OUT_ROOT / "tmp"
    d.mkdir(parents=True, exist_ok=True)
    return d
