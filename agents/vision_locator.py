import re
from pathlib import Path
from typing import Optional, Tuple

from agents.gemini_llm import GeminiLLM


class VisionLocator:
    """Very small Gemini Vision helper that returns approximate tap coords."""

    def __init__(self, llm: GeminiLLM):
        self.llm = llm

    def locate(self, screenshot_path: str, target_text: str) -> Optional[Tuple[int, int]]:
        if not self.llm:
            return None
        if not screenshot_path or not target_text:
            return None

        p = Path(screenshot_path)
        if not p.exists():
            return None

        try:
            image_bytes = p.read_bytes()
        except Exception:
            return None

        instructions = (
            "You are assisting an automated mobile QA agent. "
            "Given the screenshot, find the UI element whose label, text, or icon best matches "
            f"the target text: '{target_text}'. "
            "Respond ONLY with two integers 'x,y' for the approximate center pixel coordinates. "
            "If you truly cannot find it, reply with 'NONE'."
        )

        try:
            throttle = getattr(self.llm, "_throttle", None)
            if callable(throttle):
                throttle()
            resp = self.llm.model.generate_content(
                [
                    {
                        "role": "user",
                        "parts": [
                            {"text": instructions},
                            {
                                "inline_data": {
                                    "mime_type": "image/png",
                                    "data": image_bytes,
                                }
                            },
                        ],
                    }
                ],
                request_options={"timeout": getattr(self.llm, "timeout_s", 45)},
            )
        except Exception:
            return None

        text = (getattr(resp, "text", None) or "").strip()
        if not text or text.lower().startswith("none"):
            return None

        # Accept "123,456" or "x=123 y=456" style outputs.
        match = re.search(r"(-?\d+)\s*,\s*(-?\d+)", text)
        if not match:
            match = re.search(r"x\s*=?\s*(-?\d+).+?y\s*=?\s*(-?\d+)", text, re.IGNORECASE | re.DOTALL)
        if not match:
            return None
        try:
            x = int(match.group(1))
            y = int(match.group(2))
        except Exception:
            return None

        if x < 0 or y < 0:
            return None
        return x, y
