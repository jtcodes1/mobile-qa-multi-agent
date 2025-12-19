"""
Execution engine for the QualGent Mobile QA challenge.

This module implements the concrete execution of QA test cases using a
Planner–Executor–Supervisor architecture. It combines LLM-driven planning
with deterministic guardrails and assertions to reduce flakiness in
mobile UI automation.

High-level agent orchestration and tool registration are handled by
Google ADK (see mobile_qa_adk/agent.py).
"""

import json
import os
import re
import logging
from datetime import datetime
from PIL import Image
import numpy as np

from adb.device import AndroidDevice
from agents.android_helpers import (
    _safe_parse_xml as safe_parse_xml,
    _find_switch_node as find_switch_node,
    _find_navigate_up_node as find_navigate_up_node,
    _parse_bounds as parse_bounds,
    tap_bounds_center,
    _wm_size_xy as wm_size_xy,
    tap_sidebar_button,
    tap_settings_gear,
    tap_new_note_button,
    tap_note_title_area,
    tap_note_body_area,
    tap_text_bounds,
)
from agents.gemini_llm import GeminiLLM
from agents.planner import PlannerAgent
from agents.executor import ExecutorAgent, InvalidActionForUI
from agents.supervisor import SupervisorAgent
from agents.screen_reader import ScreenReader
from agents.vision_locator import VisionLocator

OBSIDIAN_PACKAGE = "md.obsidian"

# Mix of PASS + FAIL, as required by the challenge.
TESTS = [
    {
        "name": "T1_create_vault",
        "expected": "PASS",
        "case": "Open Obsidian, create a new Vault named 'InternVault', and enter the vault.",
    },
    {
        "name": "T2_create_note",
        "expected": "PASS",
        "case": "If no vault is open, create or open the 'InternVault' vault first. Then create a new note titled 'Meeting Notes' and type the text 'Daily Standup' into the body.",
    },
    {
        "name": "T3_assert_red_icon",
        "expected": "FAIL",
        "case": "Go to Settings and verify that the 'Appearance' tab icon is the color Red.",
    },
    {
        "name": "T4_print_to_pdf",
        "expected": "FAIL",
        "case": "Find and click the 'Print to PDF' button in the main file menu.",
    },
]

ACTION_BUDGETS = {
    "T2_create_note": 15,
    "T3_assert_red_icon": 10,
    "T4_print_to_pdf": 10,
}

AGENT_INTENTS = {
    "T1_create_vault": "Preparing InternVault via onboarding and permissions.",
    "T2_create_note": "Ensuring the Meeting Notes document exists with correct content.",
    "T3_assert_red_icon": "Verifying the Appearance accent color icon state.",
    "T4_print_to_pdf": "Checking whether Print to PDF is available in the mobile UI.",
}

logging.basicConfig(level=logging.INFO, format="%(message)s")


def _bool_env(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _int_env(name: str, default: int) -> int:
    v = os.environ.get(name)
    try:
        return int(v) if v is not None else default
    except Exception:
        return default

if _bool_env("VERBOSE_LOGS", False):
    logging.getLogger().setLevel(logging.DEBUG)


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def handle_system_picker(device: AndroidDevice, executor: ExecutorAgent, vault_name: str = "InternVault"):
    focus = (device.current_focus() or "").lower()
    if "com.android.documentsui" not in focus:
        return

    xml_raw = (device.ui_dump() or "").lower()
    xml = _normalize_whitespace(xml_raw)
    if not xml:
        return

    if "use this folder" in xml:
        try:
            executor.execute({"action": "try_tap_text", "query": vault_name})
        except Exception:
            pass
        for query in ("USE THIS FOLDER", "Use this folder"):
            try:
                executor.execute({"action": "tap_text", "query": query})
                break
            except Exception:
                continue

        xml2 = _normalize_whitespace((device.ui_dump() or "").lower())
        if "allow" in xml2:
            for query in ("Allow", "ALLOW"):
                try:
                    executor.execute({"action": "tap_text", "query": query})
                    break
                except Exception:
                    continue


def _is_all_files_access_screen(device: AndroidDevice, xml_lower: str) -> bool:
    markers = [
        "allow access to manage all files",
        "manage all files",
        "all files access",
    ]
    return any(m in (xml_lower or "") for m in markers)

def handle_all_files_access_screen(device: AndroidDevice, executor: ExecutorAgent) -> bool:
    """
    Deterministic fix for the Android Settings 'All files access' screen:
    - Toggle ON exactly once (only if currently OFF)
    - Tap top-left Navigate up
    """
    xml = device.ui_dump() or ""
    lower = xml.lower()
    if not _is_all_files_access_screen(device, lower):
        return False
    
    logging.debug("[DEBUG] Detected All files access screen. focus=%s", device.current_focus())

    root = safe_parse_xml(xml)
    if root is None:
        # If we can't parse, do NOT thrash taps. Just back out once.
        device.key(4)
        return True

    sw = find_switch_node(root)
    if sw is not None:
        checked = (sw.attrib.get("checked") or "").lower() == "true"
        logging.debug("[DEBUG] All files access switch checked=%s", checked)
        if not checked:
            # Tap ON once
            tap_bounds_center(device, sw.attrib.get("bounds", ""))
            device.sleep_ms(450)

            # Verify it is ON if still OFF, allow ONE more tap
            xml2 = device.ui_dump() or ""
            root2 = safe_parse_xml(xml2)
            if root2 is not None:
                sw2 = find_switch_node(root2)
                checked2 = sw2 is not None and (sw2.attrib.get("checked") or "").lower() == "true"
                if not checked2 and sw2 is not None:
                    tap_bounds_center(device, sw2.attrib.get("bounds", ""))
                    device.sleep_ms(450)

    # Must hit top-left back ("Navigate up") on this screen
    xml3 = device.ui_dump() or ""
    root3 = safe_parse_xml(xml3)
    if root3 is not None:
        up = find_navigate_up_node(root3)
        if up is not None and tap_bounds_center(device, up.attrib.get("bounds", "")):
            device.sleep_ms(400)
            return True

    # Fallback if Navigate up not found
    device.key(4)
    device.sleep_ms(400)
    return True


def _is_permission_dialog(xml_lower: str) -> bool:
    if not xml_lower:
        return False
    if "allow file access" in xml_lower:
        return True
    if "obsidian needs permission" in xml_lower:
        return True
    return False


def _has_obsidian_ready_markers(xml_lower: str) -> bool:
    if not xml_lower:
        return False
    onboarding_markers = [
        "configure your new vault",
        "create a vault",
        "continue without sync",
        "where is your vault located",
        "use this folder",
        "allow access",
    ]
    if any(m in xml_lower for m in onboarding_markers):
        return False
    ready_markers = [
        "internvault",
        "file explorer",
        "files",
        "new note",
        "create new note",
        "search",
        "untitled",
    ]
    return any(m in xml_lower for m in ready_markers)


def is_onboarding_screen(xml_text: str) -> bool:
    if not xml_text:
        return False
    lower = xml_text.lower()
    return ("create a vault" in lower) or ("use my existing vault" in lower)


def _make_executor_ready_guard(device: AndroidDevice):
    def guard() -> tuple[bool, str]:
        xml_lower = (device.ui_dump() or "").lower()
        if _is_permission_dialog(xml_lower):
            return False, "permission_dialog"
        return True, "ready"

    return guard


def ensure_ready_state(device: AndroidDevice, executor: ExecutorAgent, max_attempts: int = 8) -> tuple[bool, str]:
    prev_guard = None
    guard_supported = hasattr(executor, "get_ready_guard") and hasattr(executor, "set_ready_guard")
    if guard_supported:
        prev_guard = executor.get_ready_guard()
        executor.set_ready_guard(None)

    relaunch_used = False
    try:
        for _ in range(max_attempts):
            xml = device.ui_dump() or ""
            lower = xml.lower()

            if is_onboarding_screen(xml):
                logging.debug("[READY_GATE] onboarding")
                return True, "onboarding"

            if handle_all_files_access_screen(device, executor):
                device.sleep_ms(300)
                continue

            if _is_permission_dialog(lower):
                logging.debug("[READY_GATE] permission_dialog")
                tapped = False
                for label in ("Allow file access", "ALLOW FILE ACCESS"):
                    try:
                        executor.execute({"action": "tap_text", "query": label})
                        tapped = True
                        break
                    except Exception:
                        continue
                if not tapped:
                    device.sleep_ms(250)
                device.sleep_ms(300)
                handle_all_files_access_screen(device, executor)
                if not relaunch_used:
                    device.launch_app(OBSIDIAN_PACKAGE)
                    relaunch_used = True
                    device.sleep_ms(600)
                continue

            focus = (device.current_focus() or "").lower()
            if OBSIDIAN_PACKAGE not in focus:
                logging.debug("[READY_GATE] android_settings")
                device.key(4)
                device.sleep_ms(250)
                focus = (device.current_focus() or "").lower()
                if OBSIDIAN_PACKAGE not in focus:
                    device.key(4)
                    device.sleep_ms(250)
                if OBSIDIAN_PACKAGE not in focus and not relaunch_used:
                    device.launch_app(OBSIDIAN_PACKAGE)
                    relaunch_used = True
                    device.sleep_ms(600)
                continue
            else:
                logging.debug("[READY_GATE] ready_focus")
                return True, "ready"

            if _has_obsidian_ready_markers(lower):
                logging.debug("[READY_GATE] ready")
                return True, "ready"

            device.sleep_ms(350)

        return False, "not_ready"
    finally:
        if guard_supported:
            executor.set_ready_guard(prev_guard)



def _find_node_for_vault_input(xml_text: str):
    root = safe_parse_xml(xml_text)
    if root is None:
        return None

    for node in root.iter("node"):
        rid = (node.attrib.get("resource-id") or "")
        cls = (node.attrib.get("class") or "")
        text = (node.attrib.get("text") or "").strip()
        if (
            rid.endswith("vault_name_input")
            or cls.endswith("EditText")
            or text.lower() in {"my vault", "vault name"}
        ):
            return node
    return None

def is_past_obsidian_onboarding(device) -> bool:
    focus = (device.current_focus() or "").lower()
    xml = (device.ui_dump() or "").lower()

    if "md.obsidian" not in focus:
        return False

    # If we're still on onboarding / permission / vault creation screens, NOT done.
    onboarding_markers = [
        "configure your new vault",
        "create a vault",
        "all files access",
        "use this folder",
        "allow access",
    ]
    if any(m in xml for m in onboarding_markers):
        return False

    # If we're in the editor/UI (common markers), we've entered the vault.
    editor_markers = [
        "untitled",          # note title
        "md.obsidian",       # app resource ids often include this
        "file explorer",
        "search",
    ]
    return any(m in xml for m in editor_markers)

def is_meeting_notes_done(device: AndroidDevice) -> bool:
    xml = (device.ui_dump() or "").lower()
    # Requires both the title and the body text to be present somewhere in the UI tree
    return ("meeting notes" in xml) and ("daily standup" in xml)


def _find_node_by_res_id(xml_text: str, res_id: str):
    root = safe_parse_xml(xml_text)
    if root is None:
        return None
    for node in root.iter("node"):
        rid = (node.attrib.get("resource-id") or "")
        if rid == res_id or rid.endswith(res_id):
            return node
    return None


def _find_first_class(xml_text: str, class_name: str):
    root = safe_parse_xml(xml_text)
    if root is None:
        return None
    for node in root.iter("node"):
        cls = (node.attrib.get("class") or "")
        if cls == class_name or cls.endswith(f".{class_name}"):
            return node
    return None


def _find_clickable_text_node(xml_text: str, query: str):
    root = safe_parse_xml(xml_text)
    if root is None:
        return None
    q = (query or "").lower()
    for node in root.iter("node"):
        clickable = node.attrib.get("clickable") == "true"
        enabled = node.attrib.get("enabled", "true") == "true"
        if not clickable or not enabled:
            continue
        text = (node.attrib.get("text") or "").lower()
        desc = (node.attrib.get("content-desc") or "").lower()
        if q in text or q in desc:
            return node
    return None


def _bounds_top(bounds: str) -> int:
    rect = parse_bounds(bounds)
    if not rect:
        return 10**9
    return rect[1]

def _find_topmost_clickable_text_node(xml_text: str, query: str):
    root = safe_parse_xml(xml_text)
    if root is None:
        return None

    q = (query or "").lower()
    best = None
    best_top = 10**9

    for node in root.iter("node"):
        if node.attrib.get("clickable") != "true":
            continue
        text = (node.attrib.get("text") or "").lower()
        desc = (node.attrib.get("content-desc") or "").lower()
        if q not in (text + " " + desc):
            continue

        top = _bounds_top(node.attrib.get("bounds", ""))
        if top < best_top:
            best_top = top
            best = node

    return best

def _find_first_edit_text_node(xml_text: str):
    root = safe_parse_xml(xml_text)
    if root is None:
        return None
    for node in root.iter("node"):
        cls = (node.attrib.get("class") or "")
        if cls.endswith("EditText"):
            return node
    return None

def rename_note_to(device: AndroidDevice, executor: ExecutorAgent, new_name: str) -> bool:
    """
    Try 1: tap the header 'Untitled' (topmost clickable) -> rename dialog -> type name
    Try 2 fallback: More options -> Rename -> type name
    """
    xml1 = device.ui_dump() or ""
    node = _find_topmost_clickable_text_node(xml1, "untitled")
    if node is not None:
        tap_bounds_center(device, node.attrib.get("bounds", ""))
        device.sleep_ms(250)

    xml2 = device.ui_dump() or ""
    edit = _find_first_edit_text_node(xml2)

    if edit is None:
        # Fallback: open 3-dot menu and look for Rename
        try:
            executor.execute({"action": "tap_text", "query": "More options"})
        except Exception:
            pass
        device.sleep_ms(220)

        for q in ("Rename", "RENAME"):
            try:
                executor.execute({"action": "tap_text", "query": q})
                break
            except Exception:
                continue
        device.sleep_ms(250)

        xml2 = device.ui_dump() or ""
        edit = _find_first_edit_text_node(xml2)

    if edit is None:
        return False

    tap_bounds_center(device, edit.attrib.get("bounds", ""))
    device.sleep_ms(80)
    executor.execute({"action": "clear_and_type", "text": new_name})
    device.sleep_ms(100)

    # Confirm if a button exists otherwise Enter
    xml3 = (device.ui_dump() or "").lower()
    for btn in ("OK", "Ok", "SAVE", "Save", "DONE", "Done", "RENAME", "Rename"):
        if btn.lower() in xml3:
            try:
                executor.execute({"action": "tap_text", "query": btn})
                device.sleep_ms(140)
                break
            except Exception:
                pass
    else:
        device.key(66)  # Enter
        device.sleep_ms(140)

    return (new_name.lower() in (device.ui_dump() or "").lower())


def enforce_vault_name(
    device: AndroidDevice,
    executor: ExecutorAgent,
    vault_name: str = "InternVault",
) -> bool:
    """Ensure the vault name field is focused and contains the desired value."""
    xml = device.ui_dump() or ""
    lower = xml.lower()
    if "configure your new vault" not in lower:
        return False

    node = _find_node_for_vault_input(xml)
    if node is None:
        return False

    current_text = (node.attrib.get("text") or "").strip()
    if current_text.lower() == vault_name.lower():
        return True

    bounds = node.attrib.get("bounds")
    if bounds:
        try:
            executor.execute({"action": "tap_bounds", "bounds": bounds})
        except Exception:
            pass

    executor.execute({"action": "clear_and_type", "text": vault_name})
    device.sleep_ms(300)
    confirm_xml = device.ui_dump() or ""
    return vault_name.lower() in confirm_xml.lower()


def maybe_press_create_button(device: AndroidDevice, executor: ExecutorAgent, name_ready: bool) -> bool:
    if not name_ready:
        return False
    xml = (device.ui_dump() or "").lower()
    if "configure your new vault" not in xml:
        return False
    for query in ("Create a vault", "CREATE A VAULT"):
        try:
            executor.execute({"action": "tap_text", "query": query})
            return True
        except Exception:
            continue
    return False


def handle_configure_new_vault_screen(device: AndroidDevice, executor: ExecutorAgent, vault_name: str = "InternVault") -> bool:
    xml = device.ui_dump() or ""
    lower = xml.lower()
    if "configure your new vault" not in lower:
        return False

    attempt_info = []
    node = _find_node_by_res_id(xml, "md.obsidian:id/vault_name_input") or _find_first_class(xml, "EditText")
    focused = False
    if node is not None and tap_bounds_center(device, node.attrib.get("bounds", "")):
        focused = True
        attempt_info.append("[DEBUG] Focused vault input via node bounds.")
    else:
        attempt_info.append("[DEBUG] Focus fallback to relative coordinate.")
        device.tap_relative(0.5, 0.285)
        focused = True

    for attempt in range(2):
        device.keycombination(113, 29)
        device.key(67)
        device.type_text(vault_name)
        device.sleep_ms(200)
        check_xml = device.ui_dump() or ""
        if vault_name.lower() in check_xml.lower():
            attempt_info.append("[DEBUG] InternVault successfully typed.")
            break
        attempt_info.append("[DEBUG] InternVault not detected after typing.")
    else:
        logging.debug("\n".join(attempt_info))
        return False

    device.key(4)
    xml_after = device.ui_dump() or ""
    create_node = _find_clickable_text_node(xml_after, "Create a vault")
    if create_node is not None and tap_bounds_center(device, create_node.attrib.get("bounds", "")):
        attempt_info.append("[DEBUG] Tapped Create a vault via UI node.")
    else:
        attempt_info.append("[DEBUG] Create button fallback tap.")
        device.tap_relative(0.5, 0.73)

    logging.debug("\n".join(attempt_info))
    return True


def bootstrap_vault_if_needed(device: AndroidDevice, executor: ExecutorAgent, vault_name: str = "InternVault"):
    """Deterministically complete onboarding so we start from inside a vault."""
    for _ in range(20):
       
        if is_past_obsidian_onboarding(device):
            break

        xml = (device.ui_dump() or "").lower()
        if not xml:
            device.sleep_ms(250)
            continue
        
        # handle Android "All files access" screen deterministically
        if handle_all_files_access_screen(device, executor):
            device.sleep_ms(250)
            continue


        if "your thoughts are yours" in xml and "create a vault" in xml:
            try:
                executor.execute({"action": "tap_text", "query": "Create a vault"})
            except Exception:
                pass
            device.sleep_ms(250)
            xml2 = (device.ui_dump() or "").lower()
            if "your thoughts are yours" not in xml2 and "create a vault" not in xml2:
                break
            continue

        if "continue without sync" in xml:
            try:
                executor.execute({"action": "tap_text", "query": "Continue without sync"})
            except Exception:
                pass
            device.sleep_ms(350)
             # If we successfully moved on, stop looping
            xml2 = (device.ui_dump() or "").lower()
            if "continue without sync" not in xml2:
                break
            continue

        if "where is your vault located" in xml or "device storage" in xml:
            for option in ("On this device", "Device storage"):
                try:
                    executor.execute({"action": "tap_text", "query": option})
                    break
                except Exception:
                    continue
            for button in ("Connect to Obsidian Sync", "Continue", "Next"):
                if button.lower() in xml:
                    try:
                        executor.execute({"action": "tap_text", "query": button})
                        break
                    except Exception:
                        continue
            device.sleep_ms(350)
            continue

        if "configure your new vault" in xml:
            handle_configure_new_vault_screen(device, executor, vault_name=vault_name)
            handle_system_picker(device, executor, vault_name=vault_name)
            device.sleep_ms(350)
            continue

        if "use this folder" in xml or "documents" in xml:
            handle_system_picker(device, executor, vault_name=vault_name)
            device.sleep_ms(350)
            continue

        # handle "Create new note(s)" screen by tapping node bounds (avoids slow tap_text retries)
        if ("create new note" in xml):
            # Parse the SAME dump we already used (don’t ui_dump again)
            xml_raw = device.ui_dump() or ""
            root = safe_parse_xml(xml_raw)

            tapped = False
            if root:
                targets = {"create new note", "create new notes", "create a new note", "new note"}
                for node in root.iter("node"):
                    if node.attrib.get("clickable") != "true":
                        continue

                    t = (node.attrib.get("text") or "").strip().lower()
                    d = (node.attrib.get("content-desc") or "").strip().lower()

                    if t in targets or d in targets:
                        b = node.attrib.get("bounds") or ""
                        if b and tap_bounds_center(device, b):
                            tapped = True
                            break

            # Fallback ONLY if bounds-tap didn’t find a node
            if not tapped:
                try:
                    executor.execute({"action": "tap_text", "query": "Create new note"})
                except Exception:
                    pass

            device.sleep_ms(120)

            # If we moved on, stop bootstrapping
            if is_past_obsidian_onboarding(device):
                break

            continue

        # Reached a screen that isn't part of onboarding.
        break


def tap_settings_icon_if_present(device: AndroidDevice) -> bool:
    xml = device.ui_dump() or ""
    root = safe_parse_xml(xml)
    if not root:
        return False

    # Look for a clickable node whose content-desc/text hints "Settings"
    for node in root.iter("node"):
        if node.attrib.get("clickable") != "true":
            continue
        desc = (node.attrib.get("content-desc") or "").lower()
        text = (node.attrib.get("text") or "").lower()
        if "setting" in desc or "setting" in text or "preferences" in desc or "preferences" in text:
            bounds = node.attrib.get("bounds")
            if bounds:
                return tap_bounds_center(device, bounds)
    return False

def ensure_note_title_and_body(
    device: AndroidDevice,
    executor: ExecutorAgent,
    title: str = "Meeting Notes",
    body: str = "Daily Standup",
) -> bool:
    """Deterministically correct the note title/body if the planner missed."""
    lower = (device.ui_dump() or "").lower()
    title_ok = title.lower() in lower
    body_ok = body.lower() in lower
    tap_note_title_area(device)
    executor.execute({"action": "clear_and_type", "text": title})
    device.sleep_ms(150)

    tap_note_body_area(device)
    executor.execute({"action": "clear_and_type", "text": body})
    device.sleep_ms(150)

    final = (device.ui_dump() or "").lower()
    return title.lower() in final and body.lower() in final

def is_accent_swatch_red(screenshot_path: str, device: AndroidDevice) -> dict:
    """
    On Appearance screen, the accent color swatch is the colored circle near top-right.
    We crop around that region and check if average color is 'red-ish'.
    """
    w, h = wm_size_xy(device)

    # Tuned to your screenshot: swatch is near top-right around 22-26% height.
    cx = int(w * 0.90)
    cy = int(h * 0.235)
    r  = int(min(w, h) * 0.035)  # 35-45 px on 1080x2400

    x1, y1 = max(0, cx - r), max(0, cy - r)
    x2, y2 = min(w, cx + r), min(h, cy + r)

    img = Image.open(screenshot_path).convert("RGB")
    crop = img.crop((x1, y1, x2, y2))
    arr = np.asarray(crop, dtype=np.float32)

    avg = arr.mean(axis=(0, 1))  # [R,G,B]
    R, G, B = float(avg[0]), float(avg[1]), float(avg[2])

    redish = (R > 150) and (R > G + 40) and (R > B + 40)
    return {"ok": redish, "avg_rgb": [R, G, B], "crop": [x1, y1, x2, y2]}

def build_verdict(
    test_name: str,
    expected: str,
    steps: int,
    start_shot: str,
    end_shot: str,
    outcome: str,
    failure_type: str,
    verdict: str,
    notes: str,
):
    return {
        "outcome": outcome,
        "failure_type": failure_type,
        "verdict": verdict,
        "notes": notes,
        "name": test_name,
        "expected": expected,
        "steps": steps,
        "artifacts": {"start": start_shot, "end": end_shot},
    }


def _build_deterministic_executor_actions(
    device: AndroidDevice,
    run_dir: str,
    test_name: str,
):
    def open_settings(executor: ExecutorAgent, action: dict):
        tap_sidebar_button(device)
        device.sleep_ms(200)
        tap_settings_gear(device)

    def tap_appearance(executor: ExecutorAgent, action: dict):
        if tap_text_bounds(device, "Appearance"):
            return
        w, h = wm_size_xy(device)
        device.swipe(int(w * 0.5), int(h * 0.75), int(w * 0.5), int(h * 0.35), 350)
        device.sleep_ms(300)
        tap_text_bounds(device, "Appearance")

    def assert_accent_red(executor: ExecutorAgent, action: dict):
        shot = os.path.join(run_dir, f"{test_name}_det_accent.png")
        device.screenshot(shot)
        info = is_accent_swatch_red(shot, device)
        executor.record_deterministic_event(
            {
                "type": "assert_accent_red",
                "screenshot": shot,
                "info": info,
                "passed": info.get("ok") is True,
            }
        )

    def check_print_to_pdf(executor: ExecutorAgent, action: dict):
        tap_sidebar_button(device)
        device.sleep_ms(200)
        shot = os.path.join(run_dir, f"{test_name}_det_print.png")
        device.screenshot(shot)
        xml = (device.ui_dump() or "").lower()
        found = "print to pdf" in xml
        executor.record_deterministic_event(
            {
                "type": "check_print_to_pdf",
                "screenshot": shot,
                "found": found,
            }
        )

    return {
        "open_settings": open_settings,
        "tap_appearance": tap_appearance,
        "assert_accent_red": assert_accent_red,
        "check_print_to_pdf": check_print_to_pdf,
    }


def _consume_deterministic_events(
    executor: ExecutorAgent,
    test: dict,
    start_shot: str,
    expected: str,
    history: list[str],
    steps_executed: int,
):
    events = executor.consume_deterministic_events()
    verdict = None
    for evt in events:
        history.append(f"[DETERMINISTIC_EVENT/{evt.get('type','unknown')}] {evt}")
        etype = evt.get("type")
        if verdict is not None:
            continue
        if etype == "assert_accent_red":
            end_shot = evt.get("screenshot") or start_shot
            if evt.get("passed"):
                verdict = build_verdict(
                    test_name=test["name"],
                    expected=expected,
                    steps=steps_executed,
                    start_shot=start_shot,
                    end_shot=end_shot,
                    outcome="PASS",
                    failure_type="NONE",
                    verdict="NO_BUG",
                    notes="Accent swatch appears red (unexpected).",
                )
            else:
                verdict = build_verdict(
                    test_name=test["name"],
                    expected=expected,
                    steps=steps_executed,
                    start_shot=start_shot,
                    end_shot=end_shot,
                    outcome="FAIL",
                    failure_type="FAILED_ASSERTION",
                    verdict="NO_BUG",
                    notes="Accent swatch is not red (expected mismatch).",
                )
        elif etype == "check_print_to_pdf":
            end_shot = evt.get("screenshot") or start_shot
            if evt.get("found"):
                verdict = build_verdict(
                    test_name=test["name"],
                    expected=expected,
                    steps=steps_executed,
                    start_shot=start_shot,
                    end_shot=end_shot,
                    outcome="PASS",
                    failure_type="NONE",
                    verdict="NO_BUG",
                    notes="Unexpectedly found 'Print to PDF' in sidebar/menu.",
                )
            else:
                verdict = build_verdict(
                    test_name=test["name"],
                    expected=expected,
                    steps=steps_executed,
                    start_shot=start_shot,
                    end_shot=end_shot,
                    outcome="FAIL",
                    failure_type="FAILED_ASSERTION",
                    verdict="NO_BUG",
                    notes="'Print to PDF' not found in sidebar/menu (expected on mobile).",
                )
    return verdict, events


def handle_t3_assert_red_icon(
    device: AndroidDevice,
    executor: ExecutorAgent,
    screen_reader: ScreenReader,
    run_dir: str,
    test_name: str,
    start_shot: str,
    expected: str,
):
    history = []

    tap_sidebar_button(device)
    device.sleep_ms(200)

    tap_settings_gear(device)
    device.sleep_ms(200)

    tap_text_bounds(device, "Appearance")


    # 3) Tap Appearance (bounds-based scroll once if needed)
    if not tap_text_bounds(device, "Appearance"):
        w, h = wm_size_xy(device)
        device.swipe(int(w*0.5), int(h*0.75), int(w*0.5), int(h*0.35), 350)
        device.sleep_ms(450)
        if not tap_text_bounds(device, "Appearance"):
            end_shot = os.path.join(run_dir, f"{test_name}_end.png")
            device.screenshot(end_shot)
            return build_verdict(
                test_name=test_name,
                expected=expected,
                steps=0,
                start_shot=start_shot,
                end_shot=end_shot,
                outcome="FAIL",
                failure_type="FAILED_STEP",
                verdict="NO_BUG",
                notes="Could not tap 'Appearance' from Settings.",
            )

    # 4) Now you are on Appearance screen -> screenshot + swatch color check
    end_shot = os.path.join(run_dir, f"{test_name}_appearance.png")
    device.screenshot(end_shot)
    screen_reader.describe(end_shot)

    info = is_accent_swatch_red(end_shot, device)
    history.append(f"[ACCENT_SWATCH] {info}")

    if info.get("ok") is True:
        return build_verdict(
            test_name=test_name,
            expected=expected,
            steps=0,
            start_shot=start_shot,
            end_shot=end_shot,
            outcome="PASS",
            failure_type="NONE",
            verdict="NO_BUG",
            notes="Accent swatch appears red (unexpected).",
        )

    return build_verdict(
        test_name=test_name,
        expected=expected,
        steps=0,
        start_shot=start_shot,
        end_shot=end_shot,
        outcome="FAIL",  # expected FAIL
        failure_type="FAILED_ASSERTION",
        verdict="NO_BUG",
        notes="Accent swatch is not red (expected mismatch).",
    )



def handle_t4_print_to_pdf(
    device: AndroidDevice,
    executor: ExecutorAgent,
    screen_reader: ScreenReader,
    run_dir: str,
    test_name: str,
    start_shot: str,
    expected: str,
):
    history = []

    # Open the sidebar/menu (NOT 3 dots)
    tap_sidebar_button(device)

    # Evidence screenshot of where we looked
    end_shot = os.path.join(run_dir, f"{test_name}_sidebar.png")
    device.screenshot(end_shot)
    screen_reader.describe(end_shot)

    xml = (device.ui_dump() or "").lower()

    if "print to pdf" in xml:
        # Unexpectedly found it: try tapping it
        try:
            tap_text_bounds(device, "Print to PDF")
        except Exception:
            pass
        return build_verdict(
            test_name=test_name,
            expected=expected,
            steps=0,
            start_shot=start_shot,
            end_shot=end_shot,
            outcome="PASS",
            failure_type="NONE",
            verdict="NO_BUG",
            notes="Unexpectedly found 'Print to PDF' in sidebar/menu.",
        )

    # Expected: not found -> FAIL but NO_BUG
    return build_verdict(
        test_name=test_name,
        expected=expected,
        steps=0,
        start_shot=start_shot,
        end_shot=end_shot,
        outcome="FAIL",
        failure_type="FAILED_STEP",
        verdict="NO_BUG",
        notes="'Print to PDF' not found in sidebar/menu (expected on mobile).",
    )


def _perform_meeting_note_flow(
    device: AndroidDevice,
    executor: ExecutorAgent,
    title: str = "Meeting Notes",
    body: str = "Daily Standup",
) -> tuple[bool, bool, bool]:
    if is_meeting_notes_done(device):
        return True, True, True

    xml_state_full = device.ui_dump() or ""
    xml_state = xml_state_full.lower()
    note_present = ("untitled" in xml_state) or (title.lower() in xml_state)
    if not note_present and _find_first_edit_text_node(xml_state_full) is not None:
        # Any visible EditText implies the editor is already open, so do not spawn another note.
        note_present = True
    if not note_present:
        tap_new_note_button(device)
        device.sleep_ms(350)

    renamed = rename_note_to(device, executor, title)
    device.sleep_ms(150)

    ensured = ensure_note_title_and_body(device, executor, title=title, body=body)
    ok = is_meeting_notes_done(device)
    return renamed, ensured, ok


def handle_t2_create_note(
    device: AndroidDevice,
    executor: ExecutorAgent,
    screen_reader: ScreenReader,
    run_dir: str,
    test_name: str,
    start_shot: str,
    expected: str,
):
    renamed, ensured, ok = _perform_meeting_note_flow(device, executor)

    end_shot = os.path.join(run_dir, f"{test_name}_end.png")
    device.screenshot(end_shot)
    end_desc = screen_reader.describe(end_shot)

    ok = is_meeting_notes_done(device)

    verdict = build_verdict(
        test_name=test_name,
        expected=expected,
        steps=0,
        start_shot=start_shot,
        end_shot=end_shot,
        outcome="PASS" if ok else "FAIL",
        failure_type="NONE" if ok else "FAILED_ASSERTION",
        verdict="NO_BUG" if ok else "BUG",
        notes=(
            "Created note and confirmed title 'Meeting Notes' + body 'Daily Standup'."
            if ok
            else "Could not confirm title 'Meeting Notes' + body 'Daily Standup' in UI dump."
        ),
    )
    verdict["history"] = [
        f"[RENAME] renamed={renamed}",
        f"[NOTE_FIX] ensured_title_body={ensured}",
        f"[ASSERT] meeting_notes_done={ok}",
    ]
    verdict["start_screen_desc"] = ""
    verdict["end_screen_desc"] = end_desc
    return verdict



def run_test(
    test: dict,
    device: AndroidDevice,
    planner: PlannerAgent,
    executor: ExecutorAgent,
    supervisor: SupervisorAgent,
    screen_reader: ScreenReader,
    run_dir: str,
) -> dict:
    """
    Fast + robust flow:

      1) Snapshot start state (screenshot + UI dump summary).
      2) Planner makes ONE short plan (sequence of actions).
      3) Executor runs deterministically.
         - Stops early on {"action":"done"}
         - If an action fails, we re-plan at most once (cheap but helps flakiness)
      4) Snapshot end state and let Supervisor classify:
         - PASS/FAIL
         - FAILED_STEP vs FAILED_ASSERTION
         - BUG vs NO_BUG
    """
    history: list[str] = []
    failed_step = None
    steps_executed = 0
    create_button_pressed = False

    needs_vault_ready = test["name"] in {
        "T2_create_note",
        "T3_assert_red_icon",
        "T4_print_to_pdf",
    }

    intent_msg = AGENT_INTENTS.get(test["name"])
    if intent_msg:
        logging.info(f"[AGENT] {intent_msg}")

    def perform_bootstrap(reason: str):
        bootstrap_vault_if_needed(device, executor, vault_name="InternVault")
        history.append(f"[BOOTSTRAP] {reason}")

    def run_ready_gate(tag: str = "initial") -> tuple[bool, str]:
        ok, state = ensure_ready_state(device, executor)
        if ok and state == "onboarding" and needs_vault_ready:
            perform_bootstrap(f"Vault onboarding required ({tag}).")
            ok, state = ensure_ready_state(device, executor)
            if not ok:
                return False, state
            if state == "onboarding":
                return True, "onboarding"
        return ok, state

    # Config knobs (speed / stability)
    pace_ms = _int_env("PACE_MS", 250)                 # lower = faster, higher = less flaky
    max_actions = _int_env("MAX_ACTIONS", 18)          # planner cap
    budget = ACTION_BUDGETS.get(test["name"])
    if budget is not None:
        max_actions = min(max_actions, budget)
    max_replans = _int_env("MAX_REPLANS", 1)           # keep LLM calls low

    # Always launch fresh into Obsidian.
    device.launch_app(OBSIDIAN_PACKAGE)

    if needs_vault_ready:
        initial_xml = device.ui_dump() or ""
        if is_onboarding_screen(initial_xml):
            perform_bootstrap("Detected onboarding immediately after launch.")

    ready_state_ok, ready_state_state = run_ready_gate("initial")
    start_shot = os.path.join(run_dir, f"{test['name']}_start.png")
    device.screenshot(start_shot)
    start_desc = screen_reader.describe(start_shot)
    cur_desc = start_desc

    if not ready_state_ok:
        history.append(f"[READY_GATE] Failed to reach ready state before planning (state={ready_state_state}).")
        fail_verdict = build_verdict(
            test_name=test["name"],
            expected=test["expected"],
            steps=0,
            start_shot=start_shot,
            end_shot=start_shot,
            outcome="FAIL",
            failure_type="FAILED_STEP",
            verdict="BUG",
            notes=f"ensure_ready_state: could not reach ready UI before planning (state={ready_state_state}).",
        )
        fail_verdict["history"] = history[:]
        fail_verdict["start_screen_desc"] = start_desc
        fail_verdict["end_screen_desc"] = start_desc
        return fail_verdict

    ready_guard_fn = _make_executor_ready_guard(device)
    executor.set_ready_guard(ready_guard_fn)
    plan_calls = 0
    max_plan_calls = max_actions * max(1, max_replans)
    last_ui_hash = None
    stagnant_count = 0

    executor.set_deterministic_actions(
        _build_deterministic_executor_actions(
            device=device,
            run_dir=run_dir,
            test_name=test["name"],
        )
    )

    accent_assertion_done = False
    print_check_done = False

    goal_reached = False
    last_action_key = None
    repeat_action_count = 0
    while True:
        if steps_executed >= max_actions:
            history.append("[BUDGET_EXCEEDED] stopping planner loop")
            break
        if plan_calls >= max_plan_calls:
            break
        if test["name"] == "T2_create_note" and is_meeting_notes_done(device):
            history.append("[GOAL_REACHED] Meeting Notes + Daily Standup confirmed.")
            goal_reached = True
            break

        # stop toggle thrashing on Android Settings "All files access"
        if handle_all_files_access_screen(device, executor):
            auto_shot = os.path.join(run_dir, f"{test['name']}_all_files_access_fix_{plan_calls}.png")
            device.screenshot(auto_shot)
            cur_desc = screen_reader.describe(auto_shot)
            continue

        if handle_configure_new_vault_screen(device, executor, vault_name="InternVault"):
            handle_system_picker(device, executor, vault_name="InternVault")
            auto_shot = os.path.join(run_dir, f"{test['name']}_autofix_{plan_calls}.png")
            device.screenshot(auto_shot)
            cur_desc = screen_reader.describe(auto_shot)
            continue

        initial_ready = enforce_vault_name(device, executor, vault_name="InternVault")
        if not create_button_pressed:
            create_button_pressed = maybe_press_create_button(
                device, executor, initial_ready
            )
        handle_system_picker(device, executor, vault_name="InternVault")

        try:
            plan_actions = planner.plan_once(
                test_case=test["case"],
                history="\n".join(history),
                screen_desc=cur_desc,
                max_actions=max_actions,
            )
        except RuntimeError as e:
            if "quota" in str(e).lower() or "resourceexhausted" in str(e):
                history.append(f"[PLAN_ERROR] {e!r}")
                return quota_failure_result(
                    device=device,
                    test=test,
                    start_shot=start_shot,
                    run_dir=run_dir,
                    steps_executed=steps_executed,
                    history=history,
                    error=str(e),
                )
            raise
        except ValueError as e:
            history.append(f"[PLAN_ERROR] {e!r}")
            plan_calls += 1
            if plan_calls >= max_plan_calls:
                history.append("[STOP] Plan/action budget exhausted due to invalid planner output.")
                break
            continue
        history.append(f"[PLAN_{plan_calls}] {plan_actions}")
        plan_calls += 1

        actions = plan_actions if isinstance(plan_actions, list) else []
        if not actions:
            history.append("[PLAN_EMPTY] Planner returned no actions; continuing.")
            continue

        action = actions[0]
        action_name = str(action.get("action", "")).lower()
        if action_name == "done":
            history.append(f"[DONE] at step={steps_executed + 1} action={action}")
            break

        try:
            action_key = (
                action_name,
                action.get("query") or action.get("res_id") or action.get("text") or action.get("bounds") or f"{action.get('x')}:{action.get('y')}",
            )
            if action_key == last_action_key:
                repeat_action_count += 1
            else:
                repeat_action_count = 1
                last_action_key = action_key
            if repeat_action_count > 2:
                history.append(f"[DEDUP] Blocking repeated action: {action}")
                raise InvalidActionForUI("Planner repeated the same action too many times.")
            executor.execute(action)
            name_ready = enforce_vault_name(device, executor, vault_name="InternVault")
            if not create_button_pressed:
                create_button_pressed = maybe_press_create_button(
                    device, executor, name_ready
                )
            handle_system_picker(device, executor, vault_name="InternVault")
            if handle_configure_new_vault_screen(device, executor, vault_name="InternVault"):
                auto_shot = os.path.join(run_dir, f"{test['name']}_autofix_post_{steps_executed}.png")
                device.screenshot(auto_shot)
                cur_desc = screen_reader.describe(auto_shot)
                continue
            steps_executed += 1
            history.append(f"[OK] step {steps_executed}: {action}")
            device.sleep_ms(pace_ms)

            det_verdict, det_events = _consume_deterministic_events(
                executor=executor,
                test=test,
                start_shot=start_shot,
                expected=test["expected"],
                history=history,
                steps_executed=steps_executed,
            )
            for evt in det_events:
                etype = evt.get("type")
                if etype == "assert_accent_red":
                    accent_assertion_done = True
                elif etype == "check_print_to_pdf":
                    print_check_done = True
            if det_verdict:
                return det_verdict

        except InvalidActionForUI as e:
            failed_step = {"step": steps_executed + 1, "action": action, "error": repr(e)}
            history.append(f"[EXECUTOR_ERROR] step {steps_executed + 1}: {action} -> {e!r}")

            err_shot = os.path.join(run_dir, f"{test['name']}_error_{plan_calls}.png")
            device.screenshot(err_shot)
            cur_desc = screen_reader.describe(err_shot)
            stagnant_count = 0

            lowered = str(e).lower()
            if "not in ready state" in lowered:
                history.append("[READY_GATE] Guard blocked action; re-running ensure_ready_state().")
                recovered, recovered_state = run_ready_gate(f"recovery_step_{steps_executed + 1}")
                recover_shot = os.path.join(run_dir, f"{test['name']}_ready_recover_{plan_calls}.png")
                device.screenshot(recover_shot)
                cur_desc = screen_reader.describe(recover_shot)
                if recovered:
                    executor.set_ready_guard(ready_guard_fn)
                    history.append("[READY_GATE] Ready state re-established.")
                    failed_step = None
                    continue

                history.append(f"[READY_GATE] Could not re-establish ready state mid-test (state={recovered_state}).")
                fail_verdict = build_verdict(
                    test_name=test["name"],
                    expected=test["expected"],
                    steps=steps_executed,
                    start_shot=start_shot,
                    end_shot=recover_shot,
                    outcome="FAIL",
                    failure_type="FAILED_STEP",
                    verdict="BUG",
                    notes=f"Lost ready state mid-test and recovery gate failed (state={recovered_state}).",
                )
                fail_verdict["history"] = history[:]
                fail_verdict["start_screen_desc"] = start_desc
                fail_verdict["end_screen_desc"] = cur_desc
                return fail_verdict

            if plan_calls >= max_plan_calls:
                break

            failed_step = None
            continue

        except Exception as e:
            failed_step = {"step": steps_executed + 1, "action": action, "error": repr(e)}
            history.append(f"[EXECUTOR_ERROR] step {steps_executed + 1}: {action} -> {e!r}")

            err_shot = os.path.join(run_dir, f"{test['name']}_error_{plan_calls}.png")
            device.screenshot(err_shot)
            cur_desc = screen_reader.describe(err_shot)
            stagnant_count = 0

            if plan_calls >= max_plan_calls:
                break

            failed_step = None
            continue

        xml_after = device.ui_dump() or ""
        current_hash = hash(xml_after)
        if current_hash == last_ui_hash:
            stagnant_count += 1
            if stagnant_count >= 2:
                history.append("[STAGNANT] UI hash unchanged twice; sending BACK and replanning.")
                device.key(4)
                stagnant_shot = os.path.join(run_dir, f"{test['name']}_stagnant_{steps_executed}.png")
                device.screenshot(stagnant_shot)
                cur_desc = screen_reader.describe(stagnant_shot)
                last_ui_hash = current_hash
                continue
        else:
            stagnant_count = 0
        last_ui_hash = current_hash

        step_shot = os.path.join(run_dir, f"{test['name']}_step_{steps_executed}.png")
        device.screenshot(step_shot)
        cur_desc = screen_reader.describe(step_shot)

    if goal_reached:
        end_shot = os.path.join(run_dir, f"{test['name']}_end.png")
        device.screenshot(end_shot)
        end_desc = screen_reader.describe(end_shot)
        verdict = build_verdict(
            test_name=test["name"],
            expected=test["expected"],
            steps=steps_executed,
            start_shot=start_shot,
            end_shot=end_shot,
            outcome="PASS",
            failure_type="NONE",
            verdict="NO_BUG",
            notes="Deterministic goal satisfied: Meeting Notes + Daily Standup present.",
        )
        verdict["history"] = history[:]
        verdict["start_screen_desc"] = start_desc
        verdict["end_screen_desc"] = end_desc
        return verdict

    if plan_calls >= max_plan_calls and steps_executed < max_actions:
        history.append("[STOP] Plan/action budget exhausted before DONE.")

    det_verdict, det_events = _consume_deterministic_events(
        executor=executor,
        test=test,
        start_shot=start_shot,
        expected=test["expected"],
        history=history,
        steps_executed=steps_executed,
    )
    for evt in det_events:
        etype = evt.get("type")
        if etype == "assert_accent_red":
            accent_assertion_done = True
        elif etype == "check_print_to_pdf":
            print_check_done = True
    if det_verdict:
        return det_verdict

    if test["name"] == "T3_assert_red_icon" and not accent_assertion_done:
        return handle_t3_assert_red_icon(
            device=device,
            executor=executor,
            screen_reader=screen_reader,
            run_dir=run_dir,
            test_name=test["name"],
            start_shot=start_shot,
            expected=test["expected"],
        )

    if test["name"] == "T4_print_to_pdf" and not print_check_done:
        return handle_t4_print_to_pdf(
            device=device,
            executor=executor,
            screen_reader=screen_reader,
            run_dir=run_dir,
            test_name=test["name"],
            start_shot=start_shot,
            expected=test["expected"],
        )

    # Final check for lingering picker dialogs before capturing artifacts.
    handle_system_picker(device, executor, vault_name="InternVault")

    if test["name"] == "T2_create_note" and not is_meeting_notes_done(device):
        return handle_t2_create_note(
            device=device,
            executor=executor,
            screen_reader=screen_reader,
            run_dir=run_dir,
            test_name=test["name"],
            start_shot=start_shot,
            expected=test["expected"],
        )

    if test["name"] == "T2_create_note":
        fixed = ensure_note_title_and_body(device, executor, title="Meeting Notes", body="Daily Standup")
        history.append(f"[NOTE_FIX] ensured_title_body={fixed}")

    # End artifacts
    end_shot = os.path.join(run_dir, f"{test['name']}_end.png")
    device.screenshot(end_shot)
    end_desc = screen_reader.describe(end_shot)

    # Deterministic pass for T1/T2 (prevents supervisor hallucinating FAIL)
    try:
        if test["name"] == "T1_create_vault":
            ok = is_past_obsidian_onboarding(device)
            history.append(f"[DETERMINISTIC_ASSERT] past_onboarding={ok}")
            if ok:
                verdict = {
                    "outcome": "PASS",
                    "failure_type": "NONE",
                    "verdict": "NO_BUG",
                    "notes": "Deterministic check: Obsidian focused and not on onboarding/permission screens.",
                }
            else:
                verdict = supervisor.judge(
                    test_case=test["case"],
                    expected=test["expected"],
                    history="\n".join(history),
                    start_screen=start_desc,
                    end_screen=end_desc,
                    failed_step=failed_step,
                )

        elif test["name"] == "T2_create_note":
            ok = is_meeting_notes_done(device)
            history.append(f"[DETERMINISTIC_ASSERT] meeting_notes_done={ok}")
            if ok:
                verdict = {
                    "outcome": "PASS",
                    "failure_type": "NONE",
                    "verdict": "NO_BUG",
                    "notes": "Deterministic check: 'Meeting Notes' with 'Daily Standup' present on screen.",
                }
            else:
                verdict = supervisor.judge(
                    test_case=test["case"],
                    expected=test["expected"],
                    history="\n".join(history),
                    start_screen=start_desc,
                    end_screen=end_desc,
                    failed_step=failed_step,
                )
        else:
            verdict = supervisor.judge(
                test_case=test["case"],
                expected=test["expected"],
                history="\n".join(history),
                start_screen=start_desc,
                end_screen=end_desc,
                failed_step=failed_step,
            )
    except RuntimeError as e:
        if "quota" in str(e).lower() or "resourceexhausted" in str(e).lower():
            history.append(f"[SUPERVISOR_ERROR] {e!r}")
            return quota_failure_result(
                device=device,
                test=test,
                start_shot=start_shot,
                run_dir=run_dir,
                steps_executed=steps_executed,
                history=history,
                error=str(e),
            )
        raise
    except Exception:
        verdict = supervisor.judge(
            test_case=test["case"],
            expected=test["expected"],
            history="\n".join(history),
            start_screen=start_desc,
            end_screen=end_desc,
            failed_step=failed_step,
        )


    verdict["name"] = test["name"]
    verdict["expected"] = test["expected"]
    verdict["steps"] = steps_executed
    verdict["artifacts"] = {"start": start_shot, "end": end_shot}
    return verdict

def quota_failure_result(
    device: AndroidDevice,
    test: dict,
    start_shot: str,
    run_dir: str,
    steps_executed: int,
    history: list[str],
    error: str,
):
    end_shot = os.path.join(run_dir, f"{test['name']}_quota_end.png")
    try:
        device.screenshot(end_shot)
    except Exception:
        end_shot = start_shot
    return {
        "outcome": "FAIL",
        "failure_type": "FAILED_STEP",
        "verdict": "UNKNOWN",
        "notes": f"LLM quota exceeded: {error}",
        "name": test["name"],
        "expected": test["expected"],
        "steps": steps_executed,
        "artifacts": {"start": start_shot, "end": end_shot},
        "history": history,
    }

def run_suite(tests=None, clear_each=None):
    """Run the suite via CLI/ADK wrapper and return (run_dir, results)."""
    logging.info("=== Mobile QA Agent Run ===")
    logging.info(f"Time: {datetime.now().isoformat(timespec='seconds')}")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise KeyError(
            'GEMINI_API_KEY is not set. Example:\n  export GEMINI_API_KEY="YOUR_KEY_HERE"\n'
        )

    # Model swapping is part of the rubric, so keep this configurable.
    model = os.environ.get("GEMINI_MODEL", "gemma-3-27b-it")
    rpm = int(os.environ.get("GEMINI_RPM", "8"))
    llm = GeminiLLM(api_key=api_key, model_name=model, rpm_limit=rpm)
    logging.info(f"[LLM] Gemini model={model} rpm_limit={rpm}")

    device = AndroidDevice()
    screen_reader = ScreenReader(device)
    planner = PlannerAgent(llm)
    vision = VisionLocator(llm)
    executor = ExecutorAgent(device)
    executor.vision_locator = vision
    supervisor = SupervisorAgent(llm)

    run_dir = os.path.join("runs", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    # Optional: fully clear Obsidian data before each test (fresh runs).
    clear_each_flag = clear_each if clear_each is not None else _bool_env("CLEAR_APP_EACH_TEST", False)
    tests_to_run = list(tests or TESTS)

    # Ensure one fresh start per run.
    device.force_stop(OBSIDIAN_PACKAGE)
    device.pm_clear(OBSIDIAN_PACKAGE)

    results = []
    for t in tests_to_run:
        logging.info(f"\n--- Running {t['name']} (expected {t['expected']}) ---")
        if clear_each_flag:
            device.force_stop(OBSIDIAN_PACKAGE)
            device.pm_clear(OBSIDIAN_PACKAGE)

        r = run_test(t, device, planner, executor, supervisor, screen_reader, run_dir)
        logging.info(f"Result: {r}")
        results.append(r)

    # Write a simple JSONL log for your report + debugging.
    log_path = os.path.join(run_dir, "results.jsonl")
    with open(log_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logging.info(f"\nDone. Artifacts + results saved under: {run_dir}")
    return run_dir, results


def run_one(test_name: str):
    """Run a single test case and return (run_dir, verdict)."""
    tests = [t for t in TESTS if t["name"] == test_name]
    if not tests:
        raise ValueError(f"Unknown test name: {test_name}")
    run_dir, results = run_suite(tests=tests)
    return run_dir, results[0] if results else None


def main():
    run_suite()




if __name__ == "__main__":
    main()
