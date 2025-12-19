import json

from agents.prompts import SUPERVISOR_SYSTEM_PROMPT, supervisor_user_prompt


class SupervisorAgent:
    """Final verifier / reporter.

    The challenge rubric cares about:
    - FAIL STEP vs FAIL ASSERTION
    - Bug vs No Bug

    We force the model to return those fields directly as JSON.
    """

    def __init__(self, llm):
        self.llm = llm

    def _extract_json(self, text: str) -> dict:
        s = (text or "").strip()
        if s.startswith("{") and s.endswith("}"):
            return json.loads(s)

        start = s.find("{")
        end = s.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"Supervisor did not return JSON. Raw: {text[:400]}")
        return json.loads(s[start:end + 1])

    def judge(self, test_case: str, expected: str, history: str, start_screen: str, end_screen: str, failed_step):
        user = supervisor_user_prompt(
            test_case=test_case,
            expected=expected,
            history=history,
            start_screen=start_screen,
            end_screen=end_screen,
            failed_step=failed_step,
        )

        raw = self.llm.generate(system_prompt=SUPERVISOR_SYSTEM_PROMPT, user_prompt=user)
        obj = self._extract_json(raw)

        # Fill missing keys so main.py doesn't crash if the model is slightly off.
        obj.setdefault("outcome", "UNKNOWN")
        obj.setdefault("verdict", "UNKNOWN")
        obj.setdefault("failure_type", "UNKNOWN")
        obj.setdefault("notes", "")
        return obj
