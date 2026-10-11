"""The 'Config' tab: edit a controller's YAML from the viewer.

  * a top-view MAP of the path (waypoints, or a whole mission expanded from its legs) with the dock, geofence, dock keep-out, the vehicle, its trail and the sensor swath;
    click to add a waypoint, drag to move it, right click to delete it. Corners the vehicle cannot make (inside its turning circle) are drawn red.
  * a TABLE of the waypoints (waypoint_tracking) or of the mission legs (mission) with add / remove / up / down and a form for the selected leg.
  * a TREE of every other parameter (numbers, switches and text), with the file value next to it.
  * the checks of the controller itself (turning circle, geofence, dock keep-out, missing leg parameters) as you type.
"Apply to next Start" (on by default) hands the edited values to the Controls tab, which starts the controller with a temporary copy; "Save to file" writes them into the real
YAML keeping every comment (a .bak copy is made first) after asking; "Save as new file" keeps the original untouched.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets

import gains_editor
from config_model import LEG_POSITION_KEYS, LEG_TEMPLATES, ConfigDoc
from controls_panel import CONTROLLERS
from waypoint_map import WaypointMap

TABLE_CONTROLLERS = ("waypoint_tracking", "mission")


def _summary(leg: dict) -> str:
    t = leg.get("type", "?")
    keys = [k for k in leg if k != "type"]
    return ", ".join(f"{k}={leg[k]}" for k in keys[:3]) + (" ..." if len(keys) > 3 else "")


class ConfigPanel(QtWidgets.QWidget):
    def __init__(self, controls, tel=None, parent=None, confirm: Optional[Callable[[str, str], bool]] = None,
                 ask_path: Optional[Callable[[str], Optional[str]]] = None) -> None:
        super().__init__(parent)
        self.controls = controls
        self.tel = tel
        self._confirm = confirm or self._ask
        self._ask_path = ask_path or self._ask_save_path
        self.docs: dict = {}
        self.doc: Optional[ConfigDoc] = None
        self._busy = False
        self._leg_widgets: dict = {}
        self._set_from_map = False
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("File:"))
        self.ctrl = QtWidgets.QComboBox()
        self.ctrl.addItems(list(CONTROLLERS))
        self.ctrl.currentTextChanged.connect(self._on_controller)
        row.addWidget(self.ctrl, 1)
        self.btn_reload = QtWidgets.QPushButton("Reload from file")
        self.btn_reload.clicked.connect(self._reload)
        row.addWidget(self.btn_reload)
        lay.addLayout(row)
        self.path_label = QtWidgets.QLabel("")
        self.path_label.setWordWrap(True)
        self.path_label.setStyleSheet("color:#78909c")
        lay.addWidget(self.path_label)

        self.map = WaypointMap()
        self.map.pointAdded.connect(self._on_map_add)
        self.map.pointMoved.connect(self._on_map_move)
        self.map.pointRemoved.connect(self._on_map_remove)
        self.map.clicked.connect(self._on_map_click)
        lay.addWidget(self.map, 3)
        mrow = QtWidgets.QHBoxLayout()
        self.btn_fit = QtWidgets.QPushButton("Fit map")
        self.btn_fit.clicked.connect(self.map.fit)
        self.btn_clear_trail = QtWidgets.QPushButton("Clear trail")
        self.btn_clear_trail.clicked.connect(self._clear_trail)
        self.chk_place = QtWidgets.QCheckBox("click sets selected leg position")
        self.chk_place.toggled.connect(lambda v: setattr(self, "_set_from_map", bool(v)))
        mrow.addWidget(self.btn_fit)
        mrow.addWidget(self.btn_clear_trail)
        mrow.addWidget(self.chk_place)
        mrow.addStretch(1)
        lay.addLayout(mrow)
        self.live_label = QtWidgets.QLabel("")
        self.live_label.setWordWrap(True)
        lay.addWidget(self.live_label)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(True)
        self.progress.setVisible(False)
        lay.addWidget(self.progress)

        # ---- list table
        self.list_box = QtWidgets.QGroupBox("Waypoints")
        ll = QtWidgets.QVBoxLayout(self.list_box)
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setMinimumHeight(130)
        self.table.itemChanged.connect(self._on_cell)
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        ll.addWidget(self.table)
        br = QtWidgets.QHBoxLayout()
        self.leg_type = QtWidgets.QComboBox()
        self.leg_type.addItems(list(LEG_TEMPLATES))
        self.btn_add = QtWidgets.QPushButton("Add")
        self.btn_del = QtWidgets.QPushButton("Remove")
        self.btn_up = QtWidgets.QPushButton("Up")
        self.btn_dn = QtWidgets.QPushButton("Down")
        for b, f in ((self.btn_add, self._add), (self.btn_del, self._remove), (self.btn_up, lambda: self._move(-1)), (self.btn_dn, lambda: self._move(1))):
            b.clicked.connect(f)
        br.addWidget(self.leg_type)
        for b in (self.btn_add, self.btn_del, self.btn_up, self.btn_dn):
            br.addWidget(b)
        ll.addLayout(br)
        self.leg_form_box = QtWidgets.QGroupBox("Selected leg")
        self.leg_form = QtWidgets.QFormLayout(self.leg_form_box)
        ll.addWidget(self.leg_form_box)
        lay.addWidget(self.list_box)

        # ---- parameter tree
        pb = QtWidgets.QGroupBox("Parameters")
        pl = QtWidgets.QVBoxLayout(pb)
        self.filter = QtWidgets.QLineEdit()
        self.filter.setPlaceholderText("filter by name (for example cruise, kd, geofence)")
        self.filter.textChanged.connect(self._apply_filter)
        pl.addWidget(self.filter)
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(["parameter", "value", "file value"])
        self.tree.setMinimumHeight(220)
        self.tree.setColumnWidth(0, 190)
        self.tree.itemChanged.connect(self._on_tree_item)
        pl.addWidget(self.tree)
        lay.addWidget(pb)

        self.warn = QtWidgets.QPlainTextEdit()
        self.warn.setReadOnly(True)
        self.warn.setMaximumHeight(110)
        self.warn.setPlaceholderText("checks")
        lay.addWidget(self.warn)
        self.info = QtWidgets.QLabel("")
        self.info.setWordWrap(True)
        lay.addWidget(self.info)

        self.chk_apply = QtWidgets.QCheckBox("Apply to next Start")
        self.chk_apply.setToolTip("The Controls tab starts the controller with these values (a temporary copy; the file is untouched until Save)")
        self.chk_apply.setChecked(True)
        self.chk_apply.toggled.connect(lambda _v: self._push_to_controls())
        lay.addWidget(self.chk_apply)
        r1 = QtWidgets.QHBoxLayout()
        self.btn_save = QtWidgets.QPushButton("Save to file")
        self.btn_save_as = QtWidgets.QPushButton("Save as new file...")
        self.btn_revert = QtWidgets.QPushButton("Revert")
        self.btn_preview = QtWidgets.QPushButton("Preview file text")
        for b, f in ((self.btn_save, self._save), (self.btn_save_as, self._save_as), (self.btn_revert, self._revert), (self.btn_preview, self._preview)):
            b.clicked.connect(f)
            r1.addWidget(b)
        lay.addLayout(r1)
        lay.addStretch(1)
        self._on_controller(self.ctrl.currentText())

    # ------------------------------------------------------------------ helpers
    def _ask(self, title: str, text: str) -> bool:
        return QtWidgets.QMessageBox.question(self, title, text) == QtWidgets.QMessageBox.Yes

    def _ask_save_path(self, suggested: str) -> Optional[str]:
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save as new file", suggested, "YAML (*.yaml)")
        return p or None

    def _say(self, text: str) -> None:
        self.info.setText(text)

    # ------------------------------------------------------------------ loading
    def _load(self, name: str, force: bool = False) -> ConfigDoc:
        if force or name not in self.docs:
            doc = ConfigDoc(Path(CONTROLLERS[name]["config"]), name)
            edited = self.controls._edited.get(name)
            if edited is not None and not force:
                doc.data = copy.deepcopy(edited)             # edits already made in the Controls tab (Edit gains...) carry over
            self.docs[name] = doc
        return self.docs[name]

    def _on_controller(self, name: str) -> None:
        if not name:
            return
        self.doc = self._load(name)
        self.path_label.setText(str(self.doc.path))
        self._refresh_all()

    def _reload(self) -> None:
        name = self.ctrl.currentText()
        if self.doc is not None and self.doc.dirty and not self._confirm("Reload", "Discard the edits and read the file again?"):
            return
        self.controls._edited.pop(name, None)
        self.doc = self._load(name, force=True)
        self._refresh_all()
        self._say("reloaded from file")

    def _revert(self) -> None:
        if self.doc is None:
            return
        self.doc.revert()
        self.controls._edited.pop(self.doc.name, None)
        self._refresh_all()
        self._say("reverted to the file values")

    def refresh_from_controls(self) -> None:
        """Pick up edits made in the Controls tab (the Edit gains dialog) for the selected controller."""
        name = self.ctrl.currentText()
        ed = self.controls._edited.get(name)
        if self.doc is not None and ed is not None and ed != self.doc.data:
            self.doc.data = copy.deepcopy(ed)
            self._refresh_all()

    # ------------------------------------------------------------------ refresh
    def _refresh_all(self) -> None:
        self._busy = True
        try:
            doc = self.doc
            has_list = doc.name in TABLE_CONTROLLERS
            self.list_box.setVisible(has_list)
            self.map.setVisible(has_list)
            self.btn_fit.setVisible(has_list)
            self.btn_clear_trail.setVisible(has_list)
            self.chk_place.setVisible(doc.name == "mission")
            self.map.edit_mode = doc.name == "waypoint_tracking"
            self.leg_type.setVisible(doc.name == "mission")
            self.leg_form_box.setVisible(doc.name == "mission")
            self._fill_table()
            self._fill_tree()
        finally:
            self._busy = False
        self._after_change(push=False)

    def _fill_table(self) -> None:
        doc = self.doc
        t = self.table
        t.blockSignals(True)
        t.setRowCount(0)
        if doc.name == "waypoint_tracking":
            self.list_box.setTitle("Waypoints (NED: x north, y east, z depth down, metres)")
            t.setColumnCount(4)
            t.setHorizontalHeaderLabels(["#", "x [m]", "y [m]", "z [m]"])
            for i, w in enumerate(doc.items()):
                t.insertRow(i)
                it = QtWidgets.QTableWidgetItem(str(i))
                it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
                t.setItem(i, 0, it)
                for j, k in enumerate(("x", "y", "z"), start=1):
                    t.setItem(i, j, QtWidgets.QTableWidgetItem(f"{float(w.get(k, 0.0)):g}"))
        elif doc.name == "mission":
            self.list_box.setTitle("Mission legs (flown in order)")
            t.setColumnCount(3)
            t.setHorizontalHeaderLabels(["#", "type", "parameters"])
            for i, leg in enumerate(doc.items()):
                t.insertRow(i)
                for j, txt in enumerate((str(i), str(leg.get("type", "?")), _summary(leg))):
                    it = QtWidgets.QTableWidgetItem(txt)
                    it.setFlags(it.flags() & ~QtCore.Qt.ItemIsEditable)
                    t.setItem(i, j, it)
            t.resizeColumnToContents(0)
            t.resizeColumnToContents(1)
        t.blockSignals(False)
        self._fill_leg_form()

    def _selected_row(self) -> int:
        r = self.table.currentRow()
        return r if 0 <= r < self.table.rowCount() else -1

    def _fill_leg_form(self) -> None:
        while self.leg_form.rowCount():
            self.leg_form.removeRow(0)
        self._leg_widgets = {}
        if self.doc is None or self.doc.name != "mission":
            return
        r = self._selected_row()
        items = self.doc.items()
        if r < 0 or r >= len(items):
            self.leg_form_box.setTitle("Selected leg (select a row)")
            return
        leg = items[r]
        self.leg_form_box.setTitle(f"Leg {r}: {leg.get('type')}")
        for k, v in leg.items():
            if k == "type":
                continue
            if isinstance(v, bool):
                w = QtWidgets.QCheckBox()
                w.setChecked(v)
                w.toggled.connect(lambda val, k=k: self._on_leg_value(k, bool(val)))
            elif isinstance(v, (int, float)):
                w = QtWidgets.QDoubleSpinBox()
                w.setRange(-1e5, 1e5)
                w.setDecimals(3)
                w.setSingleStep(0.5)
                w.setValue(float(v))
                w.valueChanged.connect(lambda val, k=k, orig=v: self._on_leg_value(k, int(val) if isinstance(orig, int) and not isinstance(orig, bool) else float(val)))
            elif isinstance(v, list):
                w = QtWidgets.QLineEdit(", ".join(f"{x:g}" if isinstance(x, (int, float)) else str(x) for x in v))
                w.editingFinished.connect(lambda k=k, w=w, n=len(v): self._on_leg_list(k, w, n))
            else:
                w = QtWidgets.QLineEdit(str(v))
                w.editingFinished.connect(lambda k=k, w=w: self._on_leg_value(k, w.text()))
            self.leg_form.addRow(k, w)
            self._leg_widgets[k] = w

    def _on_leg_value(self, key: str, value: Any) -> None:
        r = self._selected_row()
        if self._busy or r < 0:
            return
        self.doc.items()[r][key] = value
        self._update_row_text(r)
        self._after_change()

    def _on_leg_list(self, key: str, w, n: int) -> None:
        r = self._selected_row()
        if self._busy or r < 0:
            return
        try:
            vals = [float(p) for p in w.text().replace(";", ",").split(",") if p.strip()]
            if len(vals) != n:
                raise ValueError(f"expected {n} numbers")
        except ValueError as exc:
            self._say(f"{key}: {exc}")
            return
        self.doc.items()[r][key] = vals
        self._update_row_text(r)
        self._after_change()

    def _update_row_text(self, r: int) -> None:
        if self.doc.name != "mission":
            return
        self.table.blockSignals(True)
        it = self.table.item(r, 2)
        if it is not None:
            it.setText(_summary(self.doc.items()[r]))
        self.table.blockSignals(False)

    # ------------------------------------------------------------------ tree
    @staticmethod
    def _tree_leaves(data: Any, skip: Optional[str], prefix: Tuple = ()) -> List[Tuple[Tuple, Any]]:
        out: List[Tuple[Tuple, Any]] = []
        if isinstance(data, dict):
            for k, v in data.items():
                if not prefix and k == skip:
                    continue
                out += ConfigPanel._tree_leaves(v, skip, prefix + (k,))
        elif isinstance(data, list) and not gains_editor.is_editable(data):
            for i, v in enumerate(data):
                out += ConfigPanel._tree_leaves(v, skip, prefix + (i,))
        elif gains_editor.is_editable(data) or isinstance(data, str):
            out.append((prefix, data))
        return out

    def _fill_tree(self) -> None:
        self.tree.blockSignals(True)
        self.tree.clear()
        nodes: dict = {}
        doc = self.doc
        for path, val in self._tree_leaves(doc.data, doc.list_key):
            parent = self.tree.invisibleRootItem()
            for depth in range(len(path) - 1):
                key = path[: depth + 1]
                if key not in nodes:
                    n = QtWidgets.QTreeWidgetItem(parent, [str(path[depth]), "", ""])
                    nodes[key] = n
                parent = nodes[key]
            try:
                orig = gains_editor.get_path(doc.orig, path)
            except (KeyError, IndexError, TypeError):
                orig = None
            it = QtWidgets.QTreeWidgetItem(parent, [str(path[-1]), gains_editor.format_value(val) if not isinstance(val, str) else val,
                                                    "" if orig is None else (gains_editor.format_value(orig) if not isinstance(orig, str) else orig)])
            it.setFlags(it.flags() | QtCore.Qt.ItemIsEditable)
            it.setData(0, QtCore.Qt.UserRole, path)
            if orig != val:
                it.setBackground(1, QtGui.QColor("#4e3b00"))
        self.tree.expandToDepth(0)
        self.tree.blockSignals(False)
        self._apply_filter(self.filter.text())

    def _apply_filter(self, text: str) -> None:
        t = text.strip().lower()

        def visit(item) -> bool:
            if item.childCount() == 0:
                full = self._item_full_name(item).lower()
                vis = (not t) or (t in full)
                item.setHidden(not vis)
                return vis
            vis = False
            for i in range(item.childCount()):
                vis |= visit(item.child(i))
            item.setHidden(not vis)
            if t and vis:
                item.setExpanded(True)
            return vis

        for i in range(self.tree.topLevelItemCount()):
            visit(self.tree.topLevelItem(i))

    @staticmethod
    def _item_full_name(item) -> str:
        parts = []
        while item is not None:
            parts.append(item.text(0))
            item = item.parent()
        return ".".join(reversed(parts))

    def _on_tree_item(self, item, col: int) -> None:
        if self._busy or col != 1:
            return
        path = item.data(0, QtCore.Qt.UserRole)
        if path is None:
            return
        old = self.doc.get(path)
        try:
            new = item.text(1) if isinstance(old, str) else gains_editor.parse_value(item.text(1), old)
        except ValueError as exc:
            self._say(f"{'.'.join(map(str, path))}: {exc}")
            self.tree.blockSignals(True)
            item.setText(1, old if isinstance(old, str) else gains_editor.format_value(old))
            self.tree.blockSignals(False)
            return
        self.doc.set(path, new)
        self.tree.blockSignals(True)
        item.setBackground(1, QtGui.QColor("#4e3b00") if gains_editor.get_path(self.doc.orig, path) != new else QtGui.QBrush())
        self.tree.blockSignals(False)
        self._after_change()

    # ------------------------------------------------------------------ table edits
    def _on_cell(self, item) -> None:
        if self._busy or self.doc is None or self.doc.name != "waypoint_tracking":
            return
        r, c = item.row(), item.column()
        if c == 0:
            return
        try:
            v = float(item.text())
        except ValueError:
            self._say(f"'{item.text()}' is not a number")
            self.table.blockSignals(True)
            item.setText(f"{float(self.doc.items()[r].get(('x', 'y', 'z')[c - 1], 0.0)):g}")
            self.table.blockSignals(False)
            return
        self.doc.items()[r][("x", "y", "z")[c - 1]] = v
        self._after_change()

    def _on_row_selected(self) -> None:
        if self._busy:
            return
        self._fill_leg_form()

    def _add(self) -> None:
        d = self.doc
        if d.name == "waypoint_tracking":
            last = d.items()[-1] if d.items() else {"x": 0.0, "y": 0.0, "z": 3.0}
            i = d.add_item(d.waypoint(float(last["x"]) + 2.0, float(last["y"])))
        elif d.name == "mission":
            r = self._selected_row()
            i = d.add_item(copy.deepcopy(LEG_TEMPLATES[self.leg_type.currentText()]), None if r < 0 else r + 1)
        else:
            return
        self._refresh_all()
        self.table.selectRow(i)
        self._after_change()

    def _remove(self) -> None:
        r = self._selected_row()
        if r < 0 or self.doc is None or not self.doc.list_key:
            return
        if len(self.doc.items()) <= 1:
            self._say("the list needs at least one entry")
            return
        self.doc.remove_item(r)
        self._refresh_all()
        self.table.selectRow(min(r, self.table.rowCount() - 1))
        self._after_change()

    def _move(self, delta: int) -> None:
        r = self._selected_row()
        if r < 0 or self.doc is None or not self.doc.list_key:
            return
        j = self.doc.move_item(r, delta)
        self._refresh_all()
        self.table.selectRow(j)
        self._after_change()

    # ------------------------------------------------------------------ map events
    def _on_map_add(self, x: float, y: float) -> None:
        if self.doc.name != "waypoint_tracking":
            return
        i = self.doc.add_item(self.doc.waypoint(x, y))
        self._refresh_all()
        self.table.selectRow(i)
        self._after_change()

    def _on_map_move(self, i: int, x: float, y: float) -> None:
        if self.doc.name != "waypoint_tracking" or not (0 <= i < len(self.doc.items())):
            return
        w = self.doc.items()[i]
        w["x"], w["y"] = x, y
        self._fill_table()
        self._after_change()

    def _on_map_remove(self, i: int) -> None:
        if self.doc.name != "waypoint_tracking" or len(self.doc.items()) <= 1 or not (0 <= i < len(self.doc.items())):
            return
        self.doc.remove_item(i)
        self._refresh_all()
        self._after_change()

    def _on_map_click(self, x: float, y: float) -> None:
        if self.doc.name != "mission" or not self._set_from_map:
            return
        r = self._selected_row()
        if r < 0:
            self._say("select a leg first")
            return
        leg = self.doc.items()[r]
        keys = LEG_POSITION_KEYS.get(str(leg.get("type")))
        if not keys:
            self._say(f"a {leg.get('type')} leg has no position")
            return
        if keys == ("x", "y"):
            leg["x"], leg["y"] = x, y
        else:
            k = keys[0]
            old = leg.get(k)
            leg[k] = [x, y]
            if str(leg.get("type")) == "yoyo" and isinstance(leg.get("to"), list) and isinstance(old, list):      # the yo-yo keeps its length and direction
                leg["to"] = [leg["to"][0] + x - old[0], leg["to"][1] + y - old[1]]
        self._fill_table()
        self.table.selectRow(r)
        self._fill_leg_form()
        self._after_change()

    # ------------------------------------------------------------------ the common tail of every edit
    def _after_change(self, push: bool = True) -> None:
        doc = self.doc
        if doc is None:
            return
        msgs = doc.validate()
        self.warn.setPlainText("\n".join(msgs) if msgs else "no problems found")
        pts = doc.path_points()
        bad: set = set()
        if doc.name == "waypoint_tracking":
            for m in msgs:
                if m.startswith("waypoint ") and "->" in m:
                    try:
                        bad.add(int(m.split()[1]) + 1)           # the target that sits inside the turning circle is the NEXT waypoint
                    except ValueError:
                        pass
        self.map.turn_radius_m = float(doc.data.get("speed", {}).get("turn_radius_m", 2.5))
        sf = doc.data.get("safety", {})
        self.map.geofence = dict(sf.get("geofence") or {})
        self.map.keepout_m = float(sf.get("dock_keepout_m", 0.0) or 0.0)
        self.map.swath_m = float(doc.data.get("survey", {}).get("swath_width_m", 0.0) or 0.0)
        self.map.set_points(pts, bad)
        if not self._busy and pts:
            if not getattr(self, "_fitted", None) == doc.name:
                self.map.fit()
                self._fitted = doc.name
        n = len(doc.changes())
        self._say(f"edited: {n} value(s) differ from the file" if n else "no edits (the file values)")
        if push:
            self._push_to_controls()

    def _push_to_controls(self) -> None:
        if self.doc is None:
            return
        name = self.doc.name
        if self.chk_apply.isChecked() and self.doc.dirty:
            self.controls._edited[name] = copy.deepcopy(self.doc.data)
        else:
            self.controls._edited.pop(name, None)
        if self.controls.ctrl.currentText() == name:
            self.controls._on_ctrl_changed(name)

    # ------------------------------------------------------------------ saving
    def _save(self) -> None:
        doc = self.doc
        if not doc.dirty:
            self._say("nothing to save")
            return
        try:
            doc.render()
        except ValueError as exc:
            self._say(f"not saved: {exc}")
            return
        if not self._confirm("Save to file", f"Write {len(doc.changes())} change(s) into\n{doc.path}\n\nEvery comment is kept; the old file is copied to {doc.path.name}.bak. Continue?"):
            self._say("not saved")
            return
        try:
            doc.save()
        except (ValueError, OSError) as exc:
            self._say(f"not saved: {exc}")
            return
        self.controls._edited.pop(doc.name, None)
        self._refresh_all()
        self._push_to_controls()
        self._say(f"saved to {doc.path.name} (old file: {doc.path.name}.bak)")

    def _save_as(self) -> None:
        doc = self.doc
        suggested = str(doc.path.with_name(f"{doc.path.stem}_{time.strftime('%Y%m%d_%H%M')}.yaml"))
        dest = self._ask_path(suggested)
        if not dest:
            self._say("not saved")
            return
        try:
            out = doc.save(Path(dest), backup=True) if not doc.dirty else self._save_copy(Path(dest))
        except (ValueError, OSError) as exc:
            self._say(f"not saved: {exc}")
            return
        self._say(f"saved a copy to {out} (the original {doc.path.name} is unchanged; start it with --config {out})")

    def _save_copy(self, dest: Path) -> Path:
        text = self.doc.render()
        dest.write_text(text, encoding="utf-8")
        return dest

    def _preview(self) -> None:
        try:
            text = self.doc.render()
        except ValueError as exc:
            self._say(f"cannot preview: {exc}")
            return
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"{self.doc.path.name} after saving")
        dlg.resize(760, 640)
        v = QtWidgets.QVBoxLayout(dlg)
        ed = QtWidgets.QPlainTextEdit(text)
        ed.setReadOnly(True)
        ed.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        v.addWidget(ed)
        dlg.show()
        self._preview_dialog = dlg

    # ------------------------------------------------------------------ live view
    def _clear_trail(self) -> None:
        self.map.trail = []
        self.map.update()

    def update_live(self) -> None:
        """Called by the viewer a few times a second: vehicle, trail, mission progress."""
        if self.tel is None or self.doc is None:
            return
        s = self.tel.latest()
        if s is not None:
            tr = self.map.trail
            xy = (float(s.pos[0]), float(s.pos[1]))
            if not tr or (xy[0] - tr[-1][0]) ** 2 + (xy[1] - tr[-1][1]) ** 2 > 0.09:
                if tr and ((xy[0] - tr[-1][0]) ** 2 + (xy[1] - tr[-1][1]) ** 2) > 25.0:
                    tr = []                                           # the vehicle was reset: no line across the map
                tr = (tr + [xy])[-4000:]
            self.map.set_vehicle((xy[0], xy[1], float(s.eul[2])), tr)
            self.map.start_xy = tr[0] if tr else self.map.start_xy
        _, ctl = self.tel.latest_ctrl()
        if ctl and ctl.get("ctrl") == "mission":
            pr = float(ctl.get("progress", 0.0) or 0.0)
            self.progress.setVisible(True)
            self.progress.setValue(int(round(100 * pr)))
            ab = str(ctl.get("aborted") or "")
            txt = f"leg {ctl.get('leg_index')}: {ctl.get('leg')} | {ctl.get('mode')} | ETA {float(ctl.get('eta_s', 0.0)):.0f} s | cross-track {float(ctl.get('cross_track_m', 0.0)):+.2f} m"
            cvr = self.map.coverage_now()
            if cvr is not None:
                txt += f" | swath covered {cvr:.0f} %"
            if ab:
                txt += f" | ABORTED: {ab}"
            self.live_label.setText(txt)
            if "target_x" in ctl:
                self.map.target_xy = (float(ctl["target_x"]), float(ctl["target_y"]))
        else:
            self.progress.setVisible(False)
            self.live_label.setText("")
            self.map.target_xy = None
