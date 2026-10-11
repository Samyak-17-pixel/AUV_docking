"""Edit the numbers of a controller's YAML without touching the file.

The helpers are plain Python (tested without Qt). The dialog shows every numeric / boolean / list-of-numbers leaf of the YAML in a tree; the edited
copy is written to a temporary file (comments are not kept there) and passed to the controller with --config. The original file is never changed.
"""

from __future__ import annotations

import copy
import tempfile
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "common"))
from outdirs import tmp_dir  # noqa: E402
from pathlib import Path
from typing import Any, List, Optional, Tuple

import yaml

Path_ = Tuple[Any, ...]


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def is_editable(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return True
    return isinstance(value, list) and len(value) > 0 and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)


def leaves(cfg: Any, prefix: Path_ = ()) -> List[Tuple[Path_, Any]]:
    """Every editable leaf as (path, value). Dict keys and list indices of dicts are part of the path."""
    out: List[Tuple[Path_, Any]] = []
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            out += leaves(v, prefix + (k,))
    elif isinstance(cfg, list) and not is_editable(cfg):
        for i, v in enumerate(cfg):
            out += leaves(v, prefix + (i,))
    elif is_editable(cfg):
        out.append((prefix, cfg))
    return out


def get_path(cfg: Any, path: Path_) -> Any:
    for k in path:
        cfg = cfg[k]
    return cfg


def set_path(cfg: Any, path: Path_, value: Any) -> None:
    parent = get_path(cfg, path[:-1])
    parent[path[-1]] = value


def format_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, list):
        return ", ".join(format_value(x) for x in v)
    if isinstance(v, float):
        return repr(v)
    return str(v)


def parse_value(text: str, original: Any) -> Any:
    """Text -> a value of the same kind as `original`. Raises ValueError if it does not fit."""
    t = text.strip()
    if isinstance(original, bool):
        if t.lower() in ("true", "1", "yes", "on"):
            return True
        if t.lower() in ("false", "0", "no", "off"):
            return False
        raise ValueError(f"{text!r} is not true/false")
    if isinstance(original, int):
        return int(float(t)) if float(t).is_integer() else float(t)
    if isinstance(original, float):
        return float(t)
    if isinstance(original, list):
        parts = [p for p in t.replace(";", ",").split(",") if p.strip()]
        if len(parts) != len(original):
            raise ValueError(f"expected {len(original)} numbers, got {len(parts)}")
        return [parse_value(p, o) for p, o in zip(parts, original)]
    raise ValueError("this value cannot be edited")


def apply_edits(cfg: dict, edits: dict) -> dict:
    """Return a deep copy of cfg with {path: text} applied (texts are parsed against the original value)."""
    out = copy.deepcopy(cfg)
    for path, text in edits.items():
        set_path(out, path, parse_value(text, get_path(cfg, path)))
    return out


_MISSING = object()


def changed_leaves(original: dict, edited: dict) -> List[Tuple[Path_, Any, Any]]:
    """Leaves of `original` whose value differs in `edited` (a leaf that no longer exists there counts as changed, new value None). Structural edits made in the
    Config tab (a waypoint added, a mission leg replaced) are therefore counted too, instead of raising."""
    out = []
    for p, v in leaves(original):
        try:
            new = get_path(edited, p)
        except (KeyError, IndexError, TypeError):
            new = _MISSING
        if new is _MISSING:
            out.append((p, v, None))
        elif new != v:
            out.append((p, v, new))
    return out


def write_temp_yaml(cfg: dict, name: str, directory: Optional[Path] = None) -> Path:
    d = Path(directory) if directory else tmp_dir() / "auv_sim_viewer"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}_edited.yaml"
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=None)
    return p


# ---------------------------------------------------------------------------- dialog
def make_dialog(cfg: dict, title: str, parent=None):
    """QDialog listing the leaves of cfg. After exec_() == Accepted use dialog.edited_config(). Imported lazily so the helpers above need no Qt."""
    from PyQt5 import QtCore, QtGui, QtWidgets

    class GainsDialog(QtWidgets.QDialog):
        def __init__(self) -> None:
            super().__init__(parent)
            self.setWindowTitle(title)
            self.resize(560, 640)
            self._cfg = cfg
            self._items: dict = {}
            lay = QtWidgets.QVBoxLayout(self)
            self.filter = QtWidgets.QLineEdit()
            self.filter.setPlaceholderText("filter by name (for example pitch, kd, deadband)")
            self.filter.textChanged.connect(self._apply_filter)
            lay.addWidget(self.filter)
            self.tree = QtWidgets.QTreeWidget()
            self.tree.setColumnCount(3)
            self.tree.setHeaderLabels(["parameter", "value", "file value"])
            self.tree.setColumnWidth(0, 300)
            lay.addWidget(self.tree, 1)
            self._build()
            self.tree.itemChanged.connect(self._on_changed)
            self.note = QtWidgets.QLabel("Edits go to a temporary copy; the YAML file is not changed. Double-click a value to edit it.")
            self.note.setWordWrap(True)
            lay.addWidget(self.note)
            row = QtWidgets.QHBoxLayout()
            reset = QtWidgets.QPushButton("Reset all to the file values")
            reset.clicked.connect(self._reset)
            row.addWidget(reset)
            row.addStretch(1)
            bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
            bb.accepted.connect(self._accept)
            bb.rejected.connect(self.reject)
            row.addWidget(bb)
            lay.addLayout(row)

        def _build(self) -> None:
            sections: dict = {}
            for path, val in leaves(self._cfg):
                parent_item = self.tree.invisibleRootItem()
                for depth, key in enumerate(path[:-1]):
                    k = path[: depth + 1]
                    if k not in sections:
                        it = QtWidgets.QTreeWidgetItem(parent_item, [str(key)])
                        it.setFirstColumnSpanned(False)
                        sections[k] = it
                    parent_item = sections[k]
                it = QtWidgets.QTreeWidgetItem(parent_item, [str(path[-1]), format_value(val), format_value(val)])
                it.setFlags(it.flags() | QtCore.Qt.ItemIsEditable)
                self._items[path] = it
            self.tree.expandToDepth(0)

        def _on_changed(self, item, col) -> None:
            if col != 1:
                return
            f = item.font(1)
            f.setBold(item.text(1) != item.text(2))
            item.setFont(1, f)

        def _apply_filter(self, text: str) -> None:
            t = text.strip().lower()
            for path, it in self._items.items():
                hit = (not t) or t in ".".join(str(p) for p in path).lower()
                it.setHidden(not hit)
                p = it.parent()
                while p is not None:
                    p.setHidden(False if hit else p.isHidden())
                    p = p.parent()

        def _reset(self) -> None:
            for it in self._items.values():
                it.setText(1, it.text(2))

        def _accept(self) -> None:
            try:
                self.edited_config()
            except ValueError as exc:
                QtWidgets.QMessageBox.warning(self, "Invalid value", str(exc))
                return
            self.accept()

        def edited_config(self) -> dict:
            edits = {}
            for path, it in self._items.items():
                if it.text(1) != it.text(2):
                    try:
                        parse_value(it.text(1), get_path(self._cfg, path))
                    except ValueError as exc:
                        raise ValueError(f"{'.'.join(str(p) for p in path)}: {exc}") from exc
                    edits[path] = it.text(1)
            return apply_edits(self._cfg, edits)

        def set_value(self, path: Path_, text: str) -> None:
            """Programmatic edit (used by the tests)."""
            self._items[path].setText(1, text)

    return GainsDialog()
