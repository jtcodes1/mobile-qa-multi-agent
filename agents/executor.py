import re
import time
import xml.etree.ElementTree as ET
import logging
from difflib import SequenceMatcher
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from adb.device import AndroidDevice

ENABLE_VISION_FALLBACK = False


@dataclass
class UiNode:
    text: str
    desc: str
    res_id: str
    clickable: bool
    focusable: bool
    bounds: str


def is_xml_unusable(xml: str) -> bool:
    """Very small guard for empty/garbled uiautomator dumps."""
    if not xml:
        return True
    xml = xml.strip()
    if len(xml) < 50:
        return True
    if "<node" not in xml:
        return True
    return False


def _parse_bounds(bounds: str) -> Tuple[int, int, int, int]:
    # Example: "[0,123][456,789]"
    m = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", bounds)
    if not m:
        raise ValueError(f"Bad bounds format: {bounds}")
    x1, y1, x2, y2 = map(int, m.groups())
    return x1, y1, x2, y2


def _center(bounds: str) -> Tuple[int, int]:
    x1, y1, x2, y2 = _parse_bounds(bounds)
    return (x1 + x2) // 2, (y1 + y2) // 2


class InvalidActionForUI(Exception):
    """Raised when the requested action is not valid for the current UI."""


class NoProgressError(Exception):
    """Raised when repeated actions fail to change the UI."""


class ExecutorAgent:
    """Takes a planned action and executes it via adb.

    Important: The Executor does NOT decide what to do next.
    It only tries to do what it was told.
    """

    def __init__(self, device: AndroidDevice, deterministic_actions: Optional[Dict[str, Callable[["ExecutorAgent", dict], Any]]] = None):
        # Deterministic action hooks let main.run_test intercept planner actions without changing flow.
        self.device = device
        self._deterministic_actions = deterministic_actions or {}
        self._deterministic_events: list[dict] = []
        self.vision_locator = None
        self._last_ui_hash: Optional[int] = None
        self._no_progress_count = 0
        self._ready_guard = None
        self._menu_open = False
        self._failed_reference_sig: Optional[str] = None
        self._consecutive_failure_count = 0
        self._last_action_signature: Optional[str] = None

    # UI dump parsing

    def _dump_tree(self) -> ET.Element:
        xml_text = self.device.ui_dump()
        if not xml_text:
            raise RuntimeError("UI dump was empty (uiautomator dump failed)")
        return ET.fromstring(xml_text)

    def _iter_nodes(self, root: ET.Element):
        for el in root.iter():
            if el.tag != "node":
                continue
            yield UiNode(
                text=(el.attrib.get("text") or "").strip(),
                desc=(el.attrib.get("content-desc") or "").strip(),
                res_id=(el.attrib.get("resource-id") or "").strip(),
                clickable=(el.attrib.get("clickable") == "true"),
                focusable=(el.attrib.get("focusable") == "true"),
                bounds=(el.attrib.get("bounds") or "").strip(),
            )

    def _find_by_text(self, query: str) -> Optional[UiNode]:
        """Find a UI node by label, *not brittle*.

        Why this exists:
        - LLMs will sometimes say “Create new vault” even if the button says “Create a vault”.
        - Exact substring matching makes the whole system fall apart for a single extra word.

        So we:
        1) try exact substring match (fast)
        2) if that fails, use a fuzzy score based on token overlap + sequence similarity
        """

        q_raw = (query or "").strip()
        if not q_raw:
            return None

        def norm(s: str) -> str:
            s = (s or "").lower()
            # keep letters/numbers/spaces, drop punctuation
            s = re.sub(r"[^a-z0-9\s]", " ", s)
            s = re.sub(r"\s+", " ", s).strip()
            return s

        # words we don't want to over-weight
        STOP = {
            "a",
            "an",
            "the",
            "new",
            "my",
            "your",
            "to",
            "on",
            "this",
            "device",
            "existing",
            "vault",
        }

        def tokens(s: str) -> set[str]:
            t = {w for w in norm(s).split() if w and w not in STOP}
            return t

        q = norm(q_raw)
        q_toks = tokens(q_raw)

        root = self._dump_tree()
        nodes = list(self._iter_nodes(root))

        # 1) exact substring match, prefer clickable nodes
        exact_clickable = []
        exact_any = []
        for n in nodes:
            hay = norm(f"{n.text} {n.desc}")
            if q and q in hay:
                (exact_clickable if n.clickable else exact_any).append(n)
        if exact_clickable:
            return exact_clickable[0]
        if exact_any:
            return exact_any[0]

        # 2) fuzzy best match
        best: Optional[UiNode] = None
        best_score = 0.0

        for n in nodes:
            label = f"{n.text} {n.desc}".strip()
            lab_n = norm(label)
            if not lab_n:
                continue

            lab_toks = tokens(label)
            # token overlap (handles “create vault” vs “create a vault” etc)
            overlap = 0.0
            if q_toks:
                overlap = len(q_toks & lab_toks) / max(1, len(q_toks))

            # sequence similarity (handles minor phrasing differences)
            seq = SequenceMatcher(None, q, lab_n).ratio()

            score = (0.65 * overlap) + (0.35 * seq)

            # tiny boost if it's clickable, because that's usually what we want
            if n.clickable:
                score += 0.05

            if score > best_score:
                best_score = score
                best = n

        # be conservative: don't click something random
        if best is not None and best_score >= 0.50:
            return best
        return None

    def _find_by_res_id(self, res_id: str) -> Optional[UiNode]:
        rid = res_id.strip()
        root = self._dump_tree()
        for n in self._iter_nodes(root):
            if n.res_id == rid:
                return n
        return None

    def _current_xml(self) -> str:
        return self.device.ui_dump() or ""

    def _state_signature(self, xml: str) -> int:
        focus = self.device.current_focus() or ""
        snippet = (xml or "")[:400]
        return hash(f"{focus}|{len(xml)}|{snippet}")

    def _match_node_in_xml(self, xml: str, query: str) -> bool:
        if is_xml_unusable(xml):
            return False
        q_raw = (query or "").strip()
        if not q_raw:
            return False
        try:
            root = ET.fromstring(xml)
        except Exception:
            return False

        def norm(s: str) -> str:
            return re.sub(r"\s+", " ", (s or "").lower()).strip()

        q = norm(q_raw)
        for node in root.iter("node"):
            text = norm(node.attrib.get("text") or "")
            desc = norm(node.attrib.get("content-desc") or "")
            if q and (q in text or q in desc):
                return True
        return False

    def _has_focusable_edit_field(self, xml: str) -> bool:
        if is_xml_unusable(xml):
            return True
        x = xml.lower()
        return ("edittext" in x) or ('class="android.widget.edittext"' in x) or ('focusable="true"' in x)

    def _vision_query_allowed(self, query: str) -> bool:
        q = (query or "").strip().lower()
        forbidden = [
            "menu",
            "sidebar",
            "navigate",
            "navigation",
            "new note",
            "create",
            "untitled",
            "settings",
            "more options",
            "back",
            "drawer",
        ]
        return all(f not in q for f in forbidden)

    def _validate_action_preconditions(self, action: dict, xml: str) -> tuple[bool, str]:
        kind = action.get("action")
        query = (action.get("query") or action.get("text") or "").strip()
        xml_unusable = is_xml_unusable(xml)

        if kind == "tap_text":
            if xml_unusable:
                return True, ""
            if self._match_node_in_xml(xml, query):
                return True, ""
            return False, f"tap_text target not present in current UI dump: {query}"

        if kind in {"clear_and_type", "type", "type_text"}:
            if xml_unusable or self._has_focusable_edit_field(xml):
                return True, ""
            return False, f"{kind} attempted but no focusable edit field detected in UI dump"

        return True, ""

    def _meeting_note_goal_met(self) -> bool:
        xml = (self.device.ui_dump() or "").lower()
        return ("meeting notes" in xml) and ("daily standup" in xml)

    def _action_signature(self, kind: str, action: dict) -> str:
        target = (
            action.get("query")
            or action.get("res_id")
            or action.get("text")
            or action.get("bounds")
            or f"{action.get('x')}:{action.get('y')}"
        )
        return f"{kind}|{target}"

    def _reset_failure_tracking(self):
        self._failed_reference_sig = None
        self._consecutive_failure_count = 0

    def _mark_action_success(self, sig: str):
        self._reset_failure_tracking()
        self._last_action_signature = sig

    def _note_action_failure(self, sig: str, kind: str, action: dict, reason: str):
        logging.debug(f"[NO_PROGRESS] {reason}")
        self.record_deterministic_event(
            {
                "type": "no_progress",
                "action": kind,
                "target": action.get("query") or action.get("text"),
                "reason": reason,
            }
        )
        if self._failed_reference_sig == sig:
            self._consecutive_failure_count += 1
        else:
            self._failed_reference_sig = sig
            self._consecutive_failure_count = 1
        self._last_action_signature = sig

    def _should_monitor_success(self, kind: str) -> bool:
        return kind in {"tap", "tap_bounds", "tap_text", "tap_id", "type", "clear_and_type"}

    def _looks_like_input_target(self, query: str) -> bool:
        q = (query or "").strip().lower()
        if not q:
            return False
        hints = ("name", "title", "note", "vault", "body", "search", "internvault")
        return any(h in q for h in hints)

    def _drawer_is_open(self, root: Optional[ET.Element]) -> bool:
        if root is None:
            return False
        for node in root.iter("node"):
            desc = (node.attrib.get("content-desc") or "").lower()
            rid = (node.attrib.get("resource-id") or "").lower()
            cls = (node.attrib.get("class") or "").lower()
            if "close drawer" in desc:
                return True
            if "drawer" in rid or "drawer" in cls:
                return True
        return False

    def _menu_is_open(self, xml: str) -> bool:
        if is_xml_unusable(xml):
            return False
        try:
            root = ET.fromstring(xml)
        except Exception:
            root = None
        if self._drawer_is_open(root):
            return True
        if root is None:
            return False
        for node in root.iter("node"):
            rid = (node.attrib.get("resource-id") or "").lower()
            cls = (node.attrib.get("class") or "").lower()
            if rid.endswith("/id/title") or "menuitem" in cls or "listmenuitemview" in cls:
                return True
        return False

    def _update_menu_state(self, xml: str):
        self._menu_open = self._menu_is_open(xml)

    def _input_is_focused(self, xml: str) -> bool:
        if is_xml_unusable(xml):
            return False
        try:
            root = ET.fromstring(xml)
        except Exception:
            return False
        for node in root.iter("node"):
            focused = node.attrib.get("focused") == "true"
            cls = (node.attrib.get("class") or "")
            if focused and ("EditText" in cls or "edittext" in cls.lower()):
                return True
        return False

    def _normalize_typed_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", (text or "").replace("%s", " ")).strip().lower()

    def _build_success_checker(self, kind: str, action: dict, prev_hash: int):
        def state_changed(xml: str) -> bool:
            if not xml:
                return False
            return self._state_signature(xml) != prev_hash

        query = (action.get("query") or "").strip().lower()
        if kind in {"type", "clear_and_type"}:
            expected = self._normalize_typed_text(action.get("text") or "")
            if expected:
                return (
                    lambda xml: expected in (xml or "").lower(),
                    f"text '{expected}' not detected",
                )
        if kind == "tap_text":
            menu_labels = {"more options", "more actions"}
            if query in menu_labels:
                return (
                    lambda xml: self._menu_is_open(xml),
                    "menu overlay not detected",
                )
            if self._looks_like_input_target(query):
                return (
                    lambda xml: self._input_is_focused(xml),
                    "input focus not detected",
                )
        if kind == "tap_bounds":
            return (
                lambda xml: self._input_is_focused(xml) or state_changed(xml),
                "tap did not focus input or change UI",
            )
        return state_changed, "UI did not change after action"

    def _await_action_success(self, kind: str, action: dict, prev_xml: str, prev_hash: int) -> tuple[bool, str]:
        checker, reason = self._build_success_checker(kind, action, prev_hash)
        delays = (0.2, 0.2, 0.2)
        for delay in delays:
            time.sleep(delay)
            xml = self._current_xml()
            self._update_menu_state(xml)
            if checker(xml):
                self._last_ui_hash = self._state_signature(xml)
                self._no_progress_count = 0
                return True, ""
        return False, reason

    def _after_action(self, kind: str, action: dict, prev_xml: str, prev_hash: int):
        if not self._should_monitor_success(kind):
            return
        sig = self._action_signature(kind, action)
        ok, reason = self._await_action_success(kind, action, prev_xml, prev_hash)
        if ok:
            self._mark_action_success(sig)
            return
        self._note_action_failure(sig, kind, action, reason)

    # Action helpers

    def _tap_bounds(self, bounds: str):
        x, y = _center(bounds)
        self.device.tap(x, y)

    def _clear_focused_field(self):
        # Ctrl+A then Backspace. Challenge doc explicitly mentions this.
        # Ctrl = 113, A = 29, Backspace = 67
        self.device.keycombination(113, 29)
        self.device.key(67)

    # Public entrypoint

    def set_deterministic_actions(self, mapping: Optional[Dict[str, Callable[["ExecutorAgent", dict], Any]]]):
        self._deterministic_actions = mapping or {}

    def set_ready_guard(self, guard):
        self._ready_guard = guard

    def get_ready_guard(self):
        return self._ready_guard

    def record_deterministic_event(self, event: dict):
        # Deterministic helpers push assertion results/events back to run_test.
        self._deterministic_events.append(event)

    def consume_deterministic_events(self) -> list[dict]:
        events = self._deterministic_events[:]
        self._deterministic_events.clear()
        return events

    def execute(self, action: dict):
        """Execute exactly ONE action dict.

        Supported actions:
        - tap (x,y)
        - tap_bounds (bounds)
        - tap_text (query)
        - tap_id (res_id)
        - type (text)
        - clear_and_type (text)
        - key (keycode)
        - swipe (x1,y1,x2,y2,duration_ms)
        - wait (ms)
        - done (no-op)
        """
        if not isinstance(action, dict):
            raise ValueError(f"Action must be a dict, got: {type(action)}")

        kind = action.get("action")
        if not kind:
            raise ValueError(f"Missing 'action' key: {action}")

        handler = self._deterministic_actions.get(kind)
        if handler:
            return handler(self, action)

        guard = self._ready_guard
        if guard:
            try:
                guard_result = guard()
            except Exception as guard_err:
                guard_result = (False, f"guard_error:{guard_err}")
            if isinstance(guard_result, tuple):
                allowed, state = guard_result
            else:
                allowed = bool(guard_result)
                state = ""
            if not allowed:
                state_msg = state or "not in ready state"
                logging.debug(f"[EXEC_GUARD] blocked action: not in ready state ({state_msg})")
                self.record_deterministic_event({
                    "type": "ready_guard",
                    "state": state_msg,
                    "action": kind,
                })
                raise InvalidActionForUI(f"not in ready state: {state_msg}")

        sig = self._action_signature(kind, action)
        if self._failed_reference_sig and sig != self._failed_reference_sig:
            self._reset_failure_tracking()
        if self._failed_reference_sig == sig and self._consecutive_failure_count >= 2:
            raise InvalidActionForUI("repeated action failed twice; blocking duplicate attempt")

        xml_snapshot = self._current_xml()
        self._update_menu_state(xml_snapshot)
        prev_hash = self._state_signature(xml_snapshot)
        valid, reason = self._validate_action_preconditions(action, xml_snapshot)
        if not valid:
            logging.debug(f"[EXEC_GUARD] blocked: {reason}")
            self.record_deterministic_event({
                "type": "blocked_action",
                "action": kind,
                "target": action.get("query") or action.get("text"),
                "reason": reason,
            })
            raise InvalidActionForUI(reason)

        if kind == "tap":
            if not ENABLE_VISION_FALLBACK:
                raise InvalidActionForUI("raw tap disabled while vision fallback is off")
            source = action.get("source")
            if source not in {"xml", "vision", "deterministic"}:
                x = action.get("x")
                y = action.get("y")
                msg = f"blocked raw tap: ({x},{y}) not XML-derived"
                logging.debug(f"[EXEC_GUARD] {msg}")
                self.record_deterministic_event({
                    "type": "blocked_raw_tap",
                    "x": x,
                    "y": y,
                })
                raise InvalidActionForUI("raw tap blocked; use tap_text or xml node")
            self.device.tap(int(action["x"]), int(action["y"]))
            self._after_action(kind, action, xml_snapshot, prev_hash)
            return

        if kind == "tap_bounds":
            self._tap_bounds(action["bounds"])
            self._after_action(kind, action, xml_snapshot, prev_hash)
            return

        if kind == "tap_text":
            query = action["query"]
            q_norm = str(query or "").strip().lower()
            if q_norm == "new note" and self._meeting_note_goal_met():
                self.record_deterministic_event({
                    "type": "noop",
                    "reason": "note already exists",
                    "query": query,
                })
                return
            if q_norm in {"more options", "more actions"} and self._menu_open:
                raise InvalidActionForUI("menu already open")
            # Small stabilization delay to ensure UI has settled before parsing XML.
            time.sleep(0.2)
            if self._tap_text_via_xml(query):
                self._after_action(kind, action, xml_snapshot, prev_hash)
                return
            node = self._find_by_text(query)
            if node and node.bounds:
                label = node.text or node.desc or query
                logging.debug(f'[XML_PATH] using node bounds for "{label}"')
                self._tap_bounds(node.bounds)
                self._after_action(kind, action, xml_snapshot, prev_hash)
                return

            if self.vision_locator:
                xml_snapshot = ""
                try:
                    xml_snapshot = self.device.ui_dump() or ""
                except Exception:
                    xml_snapshot = ""
                if is_xml_unusable(xml_snapshot):
                    logging.debug("[VISION_FALLBACK] XML unusable")
                    if not ENABLE_VISION_FALLBACK:
                        raise InvalidActionForUI("XML unusable and vision fallback disabled.")
                    if not self._vision_query_allowed(query):
                        raise InvalidActionForUI("Vision fallback disallowed for this query when XML is unusable.")
                    logging.debug("[VISION_FALLBACK] XML unusable, using vision-based locator")
                    try:
                        screenshot_path = self.device.screenshot("_vision_fallback.png")
                    except Exception:
                        screenshot_path = None
                    coords = None
                    if screenshot_path:
                        try:
                            coords = self.vision_locator.locate(str(screenshot_path), query)
                        except Exception:
                            coords = None
                    if coords:
                        self.device.tap(int(coords[0]), int(coords[1]))
                        try:
                            self.record_deterministic_event({
                                "type": "vision_fallback",
                                "target": query,
                                "coords": [int(coords[0]), int(coords[1])],
                            })
                        except Exception:
                            pass
                        self._after_action(kind, action, xml_snapshot, prev_hash)
                        return

            raise InvalidActionForUI(f"tap_text could not find node: {action['query']}")


        if kind == "tap_id":
            rid = action.get("res_id") or action.get("query")
            if not rid:
                raise ValueError(f"tap_id missing res_id/query: {action}")
            node = self._find_by_res_id(rid)
            if not node or not node.bounds:
                raise ValueError(f"tap_id could not find node: {rid}")
            self._tap_bounds(node.bounds)
            self._after_action(kind, action, xml_snapshot, prev_hash)
            return

        if kind == "try_tap_text":
            try:
                node = self._find_by_text(action["query"])
                if node and node.bounds:
                    self._tap_bounds(node.bounds)
            except Exception:
                pass
            return

        if kind == "type":
            self.device.type_text(str(action["text"]))
            self._after_action(kind, action, xml_snapshot, prev_hash)
            return

        if kind == "clear_and_type":
            # Assumes the correct input field is already focused.
            self._clear_focused_field()
            self.device.type_text(str(action["text"]))
            self._after_action(kind, action, xml_snapshot, prev_hash)
            return

        if kind == "key":
            self.device.key(int(action["keycode"]))
            return

        if kind == "swipe":
            self.device.swipe(
                int(action["x1"]), int(action["y1"]),
                int(action["x2"]), int(action["y2"]),
                int(action.get("duration_ms", 350)),
            )
            return

        if kind == "wait":
            self.device.sleep_ms(int(action.get("ms", 500)))
            return

        if kind == "done":
            return

        raise ValueError(f"Unknown action type: {kind}")

    def _tap_text_via_xml(self, query: str) -> bool:
        xml_text = self.device.ui_dump()
        if not xml_text:
            return False
        def norm(s: str) -> str:
            return re.sub(r"\s+", " ", (s or "").lower()).strip()

        query_lower = norm(query or "")
        if not query_lower:
            return False
        try:
            root = ET.fromstring(xml_text)
        except Exception:
            return False

        for node in root.iter("node"):
            text = norm(node.attrib.get("text") or "")
            desc = norm(node.attrib.get("content-desc") or "")
            clickable = node.attrib.get("clickable") == "true"
            enabled = node.attrib.get("enabled", "true") == "true"
            bounds = node.attrib.get("bounds") or ""
            if not clickable or not enabled or not bounds:
                continue
            if query_lower in text or query_lower in desc:
                try:
                    label = (node.attrib.get("text") or node.attrib.get("content-desc") or query or "").strip()
                    if label:
                        logging.debug(f'[XML_PATH] using node bounds for "{label}"')
                    self._tap_bounds(bounds)
                    return True
                except Exception:
                    continue
        return False
