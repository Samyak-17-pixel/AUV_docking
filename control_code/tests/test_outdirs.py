"""Nothing the tools write may go to the Home directory: everything under AUV_docking/outputs/ (user rule 2026-10-11)."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from outdirs import OUT_ROOT, out_dir, tmp_dir  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def test_paths_map_into_outputs_never_home():
    assert OUT_ROOT == REPO / "outputs"
    assert out_dir("sim_viewer_runs") == OUT_ROOT / "sim_viewer_runs"
    assert out_dir("~/dof_testing_logs") == OUT_ROOT / "dof_testing_logs"             # an old '~/' config path is mapped INTO outputs
    assert out_dir("outputs/logs/station_keeping") == OUT_ROOT / "logs" / "station_keeping"             # a leading outputs/ is not doubled
    assert out_dir(Path.home().name + "/x").is_relative_to(OUT_ROOT)
    assert tmp_dir() == OUT_ROOT / "tmp" and tmp_dir().is_dir()


def test_no_tool_writes_to_home_or_tmp():
    """No source file builds a path from Path.home(), expanduser or $HOME for output, and the yaml log dirs are plain names."""
    bad = []
    for f in list((REPO / "control_code").rglob("*.py")) + list((REPO / "dock_detection_algo").rglob("*.py")):
        if any(p in f.parts for p in ("ws", "tests", "__pycache__")) or f.name.startswith("test_") or f.name == "outdirs.py":
            continue
        text = f.read_text(errors="ignore")
        if re.search(r"Path\.home\(\)|os\.path\.expanduser|\$HOME|\$\{HOME\}", text):
            bad.append(f.name)
        if re.search(r"gettempdir\(\)|mkdtemp\((?![^)]*dir=)|NamedTemporaryFile\((?![^)]*dir=)", text):
            bad.append(f.name + " (tmp)")
    assert not bad, bad
    for y in ("dof_testing/dof_testing.yaml", "station_keeping/station_keeping.yaml", "terminal_docking_control/terminal_docking.yaml"):
        assert not re.search(r"^\s*dir:\s*~", (REPO / "control_code" / y).read_text(), re.M), y
