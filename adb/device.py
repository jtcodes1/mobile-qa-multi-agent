import re
import subprocess
import time
from pathlib import Path

# Name of the adb executable (assumes adb is on PATH)
ADB = "adb"

ADB_TEXT_KW = dict(text=True, encoding="utf-8", errors="ignore")


class AndroidDevice:
    """Very small wrapper around adb.

    I'm keeping this class dumb on purpose:
    - It *only* does actions (tap, type, swipe, etc.)
    - It does NOT do any reasoning / decision making
    The Planner + Supervisor agents are where the "thinking" happens.
    """

    def __init__(self):
        pass

    def _run(self, cmd: list[str], check: bool = True):
        # Pretty-print the command so your screen recording looks clean.
        print(f"[ADB] {' '.join(cmd)}")
        return subprocess.run(cmd, check=check)

    def _run_capture(self, cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run a command and capture stdout/stderr.

        We use this for commands where we need to parse output (e.g., resolve
        launcher activity). For normal ADB actions, `_run` is cleaner.
        """
        print(f"[ADB] {' '.join(cmd)}")
        return subprocess.run(cmd, check=check, capture_output=True, **ADB_TEXT_KW)

    # App lifecycle helpers

    def launch_app(self, package: str):
        """Launch an app reliably.

        `adb shell monkey` is convenient, but on some builds it can fail with:
        "No activities found to run" even when the package is installed.

        Strategy:
        1) Try monkey (fast).
        2) If it fails, resolve the LAUNCHER activity via `cmd package resolve-activity`
           and start it explicitly with `am start`.
        """

        try:
            proc = self._run_capture([
                ADB, "shell", "monkey",
                "-p", package,
                "-c", "android.intent.category.LAUNCHER",
                "1",
            ], check=False)
            out = (proc.stdout or "") + (proc.stderr or "")
            if proc.returncode == 0 and "No activities found" not in out:
                time.sleep(2)
                return
        except Exception:
            # fall through to explicit start
            pass

        # Fallback: resolve and start the launcher activity explicitly.
        resolved = self._run_capture([
            ADB, "shell", "cmd", "package", "resolve-activity", "--brief",
            "-c", "android.intent.category.LAUNCHER",
            package,
        ], check=False)

        # Typical output contains a line like: md.obsidian/.MainActivity
        lines = ((resolved.stdout or "") + "\n" + (resolved.stderr or "")).splitlines()
        target = None
        for line in lines:
            line = line.strip()
            if "/" in line and package in line:
                target = line
                break

        if not target:
            raise RuntimeError(
                f"Failed to launch {package}. Could not resolve launcher activity. Output: {lines[-10:]}"
            )

        self._run([
            ADB, "shell", "am", "start", "-W",
            "-n", target,
            "-a", "android.intent.action.MAIN",
            "-c", "android.intent.category.LAUNCHER",
        ])
        time.sleep(2)

    def force_stop(self, package: str):
        self._run([ADB, "shell", "am", "force-stop", package])
        time.sleep(0.5)

    def pm_clear(self, package: str):
        # This wipes app data. Good for "fresh start" test runs.
        self._run([ADB, "shell", "pm", "clear", package])
        time.sleep(1.0)


    # Basic input

    def tap(self, x: int, y: int):
        self._run([ADB, "shell", "input", "tap", str(x), str(y)])
        time.sleep(0.4)

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 350):
        self._run([ADB, "shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(duration_ms)])
        time.sleep(0.5)

    def type_text(self, text: str):
        # adb input text treats spaces weirdly unless you escape them as %s.
        safe = text.replace(" ", "%s")
        self._run([ADB, "shell", "input", "text", safe])
        time.sleep(0.4)

    def key(self, keycode: int):
        self._run([ADB, "shell", "input", "keyevent", str(keycode)])
        time.sleep(0.3)

    def keycombination(self, *keycodes: int):
        # Android 13+ supports this. API 34 is Android 14, so we're good.
        self._run([ADB, "shell", "input", "keycombination", *[str(k) for k in keycodes]])
        time.sleep(0.3)

    def sleep_ms(self, ms: int):
        time.sleep(ms / 1000.0)


    # Screens + UI hierarchy

    def screenshot(self, path_or_name: str) -> Path:
        """Take a screenshot.

        If you pass:
        - "something.png" => it saves exactly there (relative path ok)
        - "something"     => it saves to runs/screenshots/something.png
        """
        p = Path(path_or_name)
        if p.suffix.lower() != ".png":
            p = Path("runs") / "screenshots" / f"{path_or_name}.png"

        p.parent.mkdir(parents=True, exist_ok=True)

        # exec-out avoids line ending corruption
        with open(p, "wb") as f:
            subprocess.run([ADB, "exec-out", "screencap", "-p"], stdout=f, check=True)

        return p

    def wm_size(self) -> str:
        p = subprocess.run([ADB, "shell", "wm", "size"], capture_output=True, **ADB_TEXT_KW)
        return (p.stdout or p.stderr or "").strip()

    def ui_dump(self) -> str:
        remote = "/sdcard/window_dump.xml"
        try:
            subprocess.run(
                [ADB, "shell", "uiautomator", "dump", remote],
                capture_output=True,
                timeout=5,
                **ADB_TEXT_KW,
            )
            p = subprocess.run(
                [ADB, "shell", "cat", remote],
                capture_output=True,
                timeout=5,
                **ADB_TEXT_KW,
            )

            return (p.stdout or "").strip()
        except subprocess.TimeoutExpired:
            return ""
        except Exception:
            return ""

    def current_focus(self) -> str:
        try:
            proc = subprocess.run(
                [ADB, "shell", "dumpsys", "window"],
                capture_output=True,
                timeout=5,
                **ADB_TEXT_KW,
            )

            text = (proc.stdout or "") + (proc.stderr or "")
            for line in text.splitlines():
                if "mCurrentFocus" in line:
                    return line.strip()
        except subprocess.TimeoutExpired:
            return ""
        except Exception:
            return ""
        return ""

    def tap_relative(self, rel_x: float, rel_y: float):
        size = self.wm_size()
        m = re.search(r"(\d+)\s*x\s*(\d+)", size)
        if not m:
            return
        width = int(m.group(1))
        height = int(m.group(2))
        x = int(width * rel_x)
        y = int(height * rel_y)
        self.tap(x, y)
