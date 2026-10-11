"""Edit a controller YAML from the viewer (pure Python, no Qt, tested without it).

`ConfigDoc` keeps the file text, the data as loaded and the data as edited. Edits are made on the data (set a value, add / remove / move a waypoint or a mission leg);
`save()` writes them back into the ORIGINAL TEXT so every comment stays: scalars (numbers, bools, strings, short lists of numbers) are replaced in place using the
node positions of yaml.compose, and a list whose length changed (waypoints, legs) is re-rendered from its first to its last item. The result is parsed again and must equal the
edited data, otherwise nothing is written. A `.bak` copy of the old file is made first. `to_temp()` writes a plain copy (comments not kept) for "run with these values
without touching the file". `validate()` returns the warnings a controller's own checks give (turning circle, geofence, dock keep-out, missing leg parameters).
"""

from __future__ import annotations

import copy
import io
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

HERE = Path(__file__).resolve().parents[1]
CTRL = HERE.parent
sys.path[:0] = [str(HERE)] + [str(_d) for _d in sorted(HERE.iterdir()) if _d.is_dir() and _d.name not in ("tests", "scripts", "__pycache__")]   # the viewer's sub-folders stay flat-importable
for _p in (HERE, CTRL / "common", CTRL / "waypoint_tracking", CTRL / "mission"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import gains_editor  # noqa: E402

Path_ = Tuple[Any, ...]

# Which key holds the list that the table editor edits, per controller (None = only the parameter tree).
LIST_KEYS = {"waypoint_tracking": "waypoints", "mission": "legs"}

LEG_TEMPLATES: Dict[str, dict] = {
    "goto": {"type": "goto", "x": 3.0, "y": 0.0, "z": 3.0},
    "lawnmower": {"type": "lawnmower", "origin": [0.0, 5.0], "heading_deg": 90.0, "length_m": 10.0, "width_m": 16.0, "spacing_m": 8.0, "z": 3.0},
    "orbit": {"type": "orbit", "centre": [-8.0, 8.0], "radius_m": 4.0, "revs": 1.0, "clockwise": True, "start_deg": 180.0, "z": 3.0},
    "spiral": {"type": "spiral", "centre": [-8.0, 8.0], "r_start_m": 4.0, "r_end_m": 10.0, "pitch_m": 3.0, "start_deg": 180.0, "z": 3.0},
    "yoyo": {"type": "yoyo", "from": [0.0, 0.0], "to": [0.0, 12.0], "z_top": 2.0, "z_bottom": 4.0, "cycles": 2},
    "hold": {"type": "hold", "seconds": 10.0},
    "return_home": {"type": "return_home"},
}
# Which keys of a leg say WHERE it is (set from a click on the map): (point key, ...) or ("x", "y") for goto.
LEG_POSITION_KEYS = {"goto": ("x", "y"), "lawnmower": ("origin",), "orbit": ("centre",), "spiral": ("centre",), "yoyo": ("from",)}


def _flow(value: Any) -> str:
    """One value or mapping as short YAML flow text, e.g. {x: 5.0, y: 0.0, z: 3.0}."""
    s = yaml.safe_dump(value, default_flow_style=True, width=10 ** 6, sort_keys=False)
    s = s.strip()
    if s.endswith("..."):
        s = s[:-3].strip()
    return s


class ConfigDoc:
    def __init__(self, path: Path, name: str = "") -> None:
        self.path = Path(path)
        self.name = name or self.path.stem
        self.text = self.path.read_text(encoding="utf-8")
        self.orig: dict = yaml.safe_load(self.text) or {}
        self.data: dict = copy.deepcopy(self.orig)

    # ------------------------------------------------------------------ access
    @property
    def list_key(self) -> Optional[str]:
        return LIST_KEYS.get(self.name)

    def items(self) -> List[Any]:
        k = self.list_key
        return self.data.get(k, []) if k else []

    def get(self, path: Path_) -> Any:
        return gains_editor.get_path(self.data, path)

    def set(self, path: Path_, value: Any) -> None:
        gains_editor.set_path(self.data, path, value)

    # ------------------------------------------------------------------ list editing (waypoints / legs)
    def add_item(self, item: Any, index: Optional[int] = None) -> int:
        k = self.list_key
        if not k:
            raise ValueError(f"{self.name} has no editable list")
        lst = self.data.setdefault(k, [])
        i = len(lst) if index is None else max(0, min(index, len(lst)))
        lst.insert(i, copy.deepcopy(item))
        return i

    def remove_item(self, index: int) -> None:
        del self.data[self.list_key][index]

    def move_item(self, index: int, delta: int) -> int:
        lst = self.data[self.list_key]
        j = index + delta
        if not (0 <= index < len(lst) and 0 <= j < len(lst)):
            return index
        lst[index], lst[j] = lst[j], lst[index]
        return j

    def waypoint(self, x: float, y: float, z: Optional[float] = None) -> dict:
        """A new waypoint entry in the file's own style (z defaults to the previous waypoint's, else mission.hold_depth_m, else 3)."""
        if z is None:
            lst = self.items()
            z = float(lst[-1].get("z", 3.0)) if lst and isinstance(lst[-1], dict) else float(self.data.get("mission", {}).get("hold_depth_m", 3.0))
        return {"x": round(float(x), 2), "y": round(float(y), 2), "z": round(float(z), 2)}

    # ------------------------------------------------------------------ change tracking
    def changes(self) -> List[Tuple[Path_, Any, Any]]:
        """(path, old, new) for every scalar that differs, plus one ((list_key,), old list, new list) when the list's length or structure changed."""
        out: List[Tuple[Path_, Any, Any]] = []
        k = self.list_key
        skip_list = False
        if k and self._list_structure_changed(k):
            out.append(((k,), self.orig.get(k), self.data.get(k)))
            skip_list = True
        for p, v, new in self._diff(self.orig, self.data, ()):
            if skip_list and p and p[0] == k:
                continue
            out.append((p, v, new))
        return out

    def _list_structure_changed(self, k: str) -> bool:
        a, b = self.orig.get(k), self.data.get(k)
        if not isinstance(a, list) or not isinstance(b, list):
            return a != b
        if len(a) != len(b):
            return True
        return any(isinstance(x, dict) and isinstance(y, dict) and set(x) != set(y) for x, y in zip(a, b))

    def _diff(self, a: Any, b: Any, prefix: Path_) -> List[Tuple[Path_, Any, Any]]:
        out: List[Tuple[Path_, Any, Any]] = []
        if isinstance(a, dict) and isinstance(b, dict):
            for key in a:
                if key in b:
                    out += self._diff(a[key], b[key], prefix + (key,))
        elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b) and not gains_editor.is_editable(a):
            for i, (x, y) in enumerate(zip(a, b)):
                out += self._diff(x, y, prefix + (i,))
        elif a != b:
            out.append((prefix, a, b))
        return out

    @property
    def dirty(self) -> bool:
        return self.data != self.orig

    def revert(self) -> None:
        self.data = copy.deepcopy(self.orig)

    # ------------------------------------------------------------------ writing
    def to_temp(self) -> Path:
        return gains_editor.write_temp_yaml(self.data, self.name)

    def render(self) -> str:
        """The file text with the edits applied and every comment kept. Raises ValueError if that cannot be done exactly."""
        text = self.text
        root = yaml.compose(text)
        repl: List[Tuple[int, int, str]] = []
        k = self.list_key
        structural = bool(k) and self._list_structure_changed(k)
        if structural:
            node = self._node(root, (k,))
            new_list = self.data[k]
            if node is None:
                raise ValueError(f"'{k}:' was not found in the file text")
            if not new_list:
                raise ValueError(f"'{k}' would be empty: add at least one entry")
            old_items = node.value
            col = node.start_mark.column
            if node.flow_style:
                repl.append((node.start_mark.index, node.end_mark.index, _flow(new_list)))
            else:
                body = ("\n" + " " * col).join("- " + _flow(it) for it in new_list)
                end = old_items[-1].end_mark.index if old_items else node.end_mark.index
                repl.append((node.start_mark.index, end, body))
        for p, old, new in self._diff(self.orig, self.data, ()):
            if structural and p and p[0] == k:
                continue
            node = self._node(root, p)
            if node is None:
                raise ValueError(f"cannot find {'.'.join(str(x) for x in p)} in the file text")
            if isinstance(node, yaml.SequenceNode):
                new_text = _flow(new)
            elif isinstance(node, yaml.ScalarNode):
                new_text = self._scalar_text(new, node)
            else:
                raise ValueError(f"cannot edit {'.'.join(str(x) for x in p)} in place")
            repl.append((node.start_mark.index, node.end_mark.index, new_text))
        for a, b, s in sorted(repl, reverse=True):
            text = text[:a] + s + text[b:]
        try:
            check = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ValueError(f"the edited text is not valid YAML: {exc}") from None
        if check != self.data:
            raise ValueError("the edited text does not match the edited values (a comment-preserving save is not possible for this change)")
        return text

    @staticmethod
    def _scalar_text(value: Any, node) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return repr(value)
        if value is None:
            return "null"
        s = _flow(value)
        return s

    @staticmethod
    def _node(root, path: Path_):
        node = root
        for key in path:
            if isinstance(node, yaml.MappingNode):
                hit = None
                for k, v in node.value:
                    if k.value == str(key):
                        hit = v
                        break
                if hit is None:
                    return None
                node = hit
            elif isinstance(node, yaml.SequenceNode) and isinstance(key, int) and 0 <= key < len(node.value):
                node = node.value[key]
            else:
                return None
        return node

    def save(self, path: Optional[Path] = None, backup: bool = True) -> Path:
        """Write the edited text to `path` (default: the original file; a .bak copy of the old file is made first). Raises ValueError and writes nothing on a problem."""
        text = self.render()
        dest = Path(path) if path else self.path
        if backup and dest.exists():
            shutil.copy2(dest, dest.with_suffix(dest.suffix + ".bak"))
        dest.write_text(text, encoding="utf-8")
        if dest == self.path:
            self.text = text
            self.orig = yaml.safe_load(text) or {}
            self.data = copy.deepcopy(self.orig)
        return dest

    # ------------------------------------------------------------------ checks
    def validate(self, start_xy: Tuple[float, float] = (0.0, 0.0)) -> List[str]:
        """Warnings for the edited data (an empty list = nothing found)."""
        msgs: List[str] = []
        try:
            if self.name == "waypoint_tracking":
                from waypoint_tracking_core import parse_waypoints, unreachable_corners
                wps = parse_waypoints(self.data.get("waypoints", []), float(self.data.get("mission", {}).get("hold_depth_m", 5.0)))
                if not wps:
                    return ["the waypoint list is empty"]
                R = float(self.data.get("speed", {}).get("turn_radius_m", 2.5))
                msgs += unreachable_corners(wps, R, bool(self.data.get("mission", {}).get("loop", False)), start_xy,
                                            float(self.data.get("mission", {}).get("acceptance_radius_m", 0.5)))
                lim = self.data.get("safety", {})
                for i, w in enumerate(wps):
                    if not (float(lim.get("min_depth_m", 0.0)) <= w[2] <= float(lim.get("max_depth_m", 1e9))):
                        msgs.append(f"waypoint {i}: depth {w[2]:.1f} m is outside the safety limits")
                    if abs(w[0] - 10.0) < 1.5 and abs(w[1]) < 1.5:
                        msgs.append(f"waypoint {i} ({w[0]:.1f}, {w[1]:.1f}) is at the dock (10, 0)")
            elif self.name == "mission":
                from mission_core import validate_mission
                msgs += validate_mission(self.data, start_xy)
        except (KeyError, ValueError, TypeError) as exc:
            msgs.append(f"{type(exc).__name__}: {exc}")
        return msgs

    def path_points(self, start_xy: Tuple[float, float] = (0.0, 0.0)) -> List[Tuple[float, float, float, str]]:
        """The path as (x, y, z, label) for the map: the waypoints, or the whole expanded mission. Never raises (an unbuildable mission gives [])."""
        try:
            if self.name == "waypoint_tracking":
                from waypoint_tracking_core import parse_waypoints
                hz = float(self.data.get("mission", {}).get("hold_depth_m", 5.0))
                return [(w[0], w[1], w[2], f"wp{i}") for i, w in enumerate(parse_waypoints(self.data.get("waypoints", []), hz))]
            if self.name == "mission":
                from mission_core import plan_polyline
                return plan_polyline(self.data, start_xy)
        except (KeyError, ValueError, TypeError):
            return []
        return []
