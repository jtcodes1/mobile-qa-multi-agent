"""
Android-specific UI helpers shared across the execution engine.

These routines stay centralized so they can be reused outside main.py
without duplicating the somewhat-finicky coordinate math.
"""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from typing import Optional, Tuple

from adb.device import AndroidDevice

__all__ = [
    "_safe_parse_xml",
    "_find_switch_node",
    "_find_navigate_up_node",
    "_parse_bounds",
    "tap_bounds_center",
    "_wm_size_xy",
    "tap_sidebar_button",
    "tap_settings_gear",
    "tap_new_note_button",
    "tap_note_title_area",
    "tap_note_body_area",
    "tap_text_bounds",
]


def _safe_parse_xml(xml_text: str) -> Optional[ET.Element]:
    """Defensive XML parsing; returns None for blank dumps or parse errors."""
    if not xml_text:
        return None
    try:
        return ET.fromstring(xml_text)
    except Exception:
        return None


def _find_switch_node(root: ET.Element) -> Optional[ET.Element]:
    """Locate the settings toggle, preferring nodes that expose 'checked'."""
    if root is None:
        return None

    # Prefer explicit Switch/CompoundButton nodes with a "checked" attribute.
    for node in root.iter("node"):
        cls = (node.attrib.get("class") or "")
        if cls in ("android.widget.Switch", "android.widget.CompoundButton") and "checked" in node.attrib:
            return node

    # Fallback: any checkable node that exposes "checked"
    for node in root.iter("node"):
        if node.attrib.get("checkable") == "true" and "checked" in node.attrib:
            return node

    return None


def _find_navigate_up_node(root: ET.Element) -> Optional[ET.Element]:
    """Top-left back arrow for Android Settings screens."""
    if root is None:
        return None
    for node in root.iter("node"):
        desc = (node.attrib.get("content-desc") or "").lower()
        rid = (node.attrib.get("resource-id") or "")
        if "navigate up" in desc or rid.endswith(":id/up"):
            return node
    return None


def _parse_bounds(bounds: str) -> Optional[Tuple[int, int, int, int]]:
    """Convert Android bounds string into (left, top, right, bottom)."""
    if not bounds:
        return None
    try:
        left, top, right, bottom = map(int, re.findall(r"\d+", bounds))
        return left, top, right, bottom
    except Exception:
        return None


def tap_bounds_center(device: AndroidDevice, bounds: str) -> bool:
    """Tap the center of the supplied bounds rectangle."""
    rect = _parse_bounds(bounds)
    if not rect:
        return False
    left, top, right, bottom = rect
    x = (left + right) // 2
    y = (top + bottom) // 2
    device.tap(x, y)
    return True


def _wm_size_xy(device: AndroidDevice) -> Tuple[int, int]:
    """Best-effort window size helper (falls back to 1080x2400)."""
    out = ""
    try:
        out = device.wm_size() or ""
    except Exception:
        pass
    m = re.search(r"(\d+)\s*x\s*(\d+)", out)
    if m:
        return int(m.group(1)), int(m.group(2))
    return (1080, 2400)


def _wait_for_ui_change(device: AndroidDevice, prev_xml: str, timeout: float = 0.8, poll: float = 0.2) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        xml = device.ui_dump() or ""
        if xml and xml != prev_xml:
            return True
        time.sleep(poll)
    return False


def tap_sidebar_button(device: AndroidDevice):
    # Known good: adb shell input tap 100 140
    w, h = _wm_size_xy(device)
    x = int(w * (100 / 1080))
    y = int(h * (140 / 2400))
    prev_xml = device.ui_dump() or ""
    for attempt in range(2):
        device.tap(x, y)
        if _wait_for_ui_change(device, prev_xml):
            break
        prev_xml = device.ui_dump() or prev_xml
    device.sleep_ms(400)


def tap_settings_gear(device: AndroidDevice):
    # Known good for 1080x2400: (900,140)
    # Use ratios so it still works if size changes a bit
    w, h = _wm_size_xy(device)
    x = int(w * (900 / 1080))
    y = int(h * (140 / 2400))
    prev_xml = device.ui_dump() or ""
    for attempt in range(2):
        device.tap(x, y)
        if _wait_for_ui_change(device, prev_xml):
            break
        prev_xml = device.ui_dump() or prev_xml
    device.sleep_ms(350)


def tap_new_note_button(device: AndroidDevice):
    # Bottom toolbar "+" button (roughly centered).
    w, h = _wm_size_xy(device)
    x = int(w * 0.5)
    y = int(h * 0.93)
    device.tap(x, y)
    device.sleep_ms(300)


def tap_note_title_area(device: AndroidDevice):
    # Focus the explicit title field near the top.
    w, h = _wm_size_xy(device)
    x = int(w * 0.5)
    y = int(h * 0.18)
    device.tap(x, y)
    device.sleep_ms(120)


def tap_note_body_area(device: AndroidDevice):
    # Focus the editor body below the title field.
    w, h = _wm_size_xy(device)
    x = int(w * 0.5)
    y = int(h * 0.36)
    device.tap(x, y)
    device.sleep_ms(120)


def tap_text_bounds(device: AndroidDevice, needle: str) -> bool:
    """Dump once and tap whichever node contains the supplied text."""
    xml_raw = device.ui_dump() or ""
    root = _safe_parse_xml(xml_raw)
    if not root:
        return False
    n = needle.lower()
    for node in root.iter("node"):
        text = (node.attrib.get("text") or "")
        desc = (node.attrib.get("content-desc") or "")
        hay = (text + " " + desc).lower()
        bounds = node.attrib.get("bounds") or ""
        if n in hay and bounds:
            tap_bounds_center(device, bounds)
            device.sleep_ms(300)
            return True
    return False
