"""Edit single numbers in a controller YAML IN PLACE, keeping every comment and the layout. Used by the viewer's 'Save as defaults'. Pure Python.

Only changes `<key>: <number>` for one loop inside the top-level `gains:` section, in either style:
    gains:                                   gains:
      heave: {kp: 7.5, ki: 0.1}                heave:
                                                 kp: 7.5        # comment survives
Anything else (a missing loop or key, a value that is not a plain number) raises KeyError and the file is NOT touched.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def _fmt(v: float) -> str:
    return f"{v:.6g}"


def set_gain_in_text(text: str, loop: str, key: str, value: float) -> str:
    lines = text.split("\n")
    # --- find the top-level `gains:` section
    start = next((i for i, l in enumerate(lines) if re.match(r"^gains:\s*(#.*)?$", l)), None)
    if start is None:
        raise KeyError("no top-level 'gains:' section")
    end = next((i for i in range(start + 1, len(lines)) if lines[i].strip() and not lines[i].startswith((" ", "\t", "#"))), len(lines))
    # --- find the loop inside it
    loop_re = re.compile(rf"^(\s+){re.escape(loop)}\s*:(.*)$")
    li = next((i for i in range(start + 1, end) if loop_re.match(lines[i])), None)
    if li is None:
        raise KeyError(f"loop '{loop}' not found under gains")
    m = loop_re.match(lines[li])
    indent, rest = len(m.group(1)), m.group(2)
    key_re = re.compile(rf"(\b{re.escape(key)}\s*:\s*)({NUM})(?![\w.])")
    if rest.strip().startswith("{"):                                          # flow style on one line
        new, n = key_re.subn(lambda mm: mm.group(1) + _fmt(value), lines[li], count=1)
        if n == 0:
            raise KeyError(f"key '{key}' not found in {loop}")
        lines[li] = new
        return "\n".join(lines)
    # block style: the keys are the following, more indented lines
    for i in range(li + 1, end):
        l = lines[i]
        if l.strip() and (len(l) - len(l.lstrip())) <= indent and not l.lstrip().startswith("#"):
            break
        if re.match(rf"^\s+{re.escape(key)}\s*:", l):
            new, n = key_re.subn(lambda mm: mm.group(1) + _fmt(value), l, count=1)
            if n == 0:
                raise KeyError(f"'{key}' of {loop} is not a plain number")
            lines[i] = new
            return "\n".join(lines)
    raise KeyError(f"key '{key}' not found in {loop}")


def save_gain(path: Path, loop: str, key: str, value: float) -> None:
    p = Path(path)
    new = set_gain_in_text(p.read_text(encoding="utf-8"), loop, key, value)
    p.write_text(new, encoding="utf-8")


def save_gains(path: Path, changes: List[Tuple[str, str, float]]) -> int:
    """Apply several (loop, key, value) changes; all or nothing. Returns how many were written."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    for loop, key, value in changes:
        text = set_gain_in_text(text, loop, key, value)
    p.write_text(text, encoding="utf-8")
    return len(changes)
