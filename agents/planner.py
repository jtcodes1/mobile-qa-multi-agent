import json
import re
from typing import Any, Dict, List

from agents.prompts import PLANNER_SYSTEM_PROMPT, planner_user_prompt


class PlannerAgent:
    """Planner agent (LLM).

    Goal:
    - Keep LLM calls LOW (speed + quota).
    - Return a SHORT sequence of actions executable by ExecutorAgent.

    The caller can re-plan if needed, but by default we prefer:
      one plan call per test.
    """

    def __init__(self, llm):
        self.llm = llm

    def _extract_json(self, text: str) -> Any:
        s = (text or "").strip()
        if not s:
            raise ValueError("Planner returned empty response.")

        # Fast-path: try to parse entire string.
        try:
            return json.loads(s)
        except Exception:
            pass

        decoder = json.JSONDecoder()
        for idx, ch in enumerate(s):
            if ch not in "{[":
                continue
            try:
                obj, _ = decoder.raw_decode(s[idx:])
                return obj
            except json.JSONDecodeError:
                continue

        raise ValueError(f"Planner did not return JSON. Raw: {s[:400]}")

    def plan_once(
        self,
        test_case: str,
        history: str,
        screen_desc: str,
        max_actions: int = 18,
    ) -> List[Dict[str, Any]]:
        prompt = planner_user_prompt(
            test_case=test_case,
            history=history,
            screen_desc=screen_desc,
            max_actions=max_actions,
        )

        raw = self.llm.generate(
            system_prompt=PLANNER_SYSTEM_PROMPT,
            user_prompt=prompt,
        )

        obj = self._extract_json(raw)

        # Accept:
        #   {"actions": [...]}
        #   {"action": "...", ...}
        #   [ {...}, {...} ]
        actions = None
        if isinstance(obj, dict) and "actions" in obj:
            actions = obj["actions"]
        elif isinstance(obj, dict) and "action" in obj:
            actions = [obj]
        elif isinstance(obj, list):
            actions = obj
        else:
            raise ValueError(f"Planner JSON missing 'actions' or 'action': {obj}")

        if not isinstance(actions, list):
            raise ValueError(f"Planner actions must be a list. Got: {type(actions)}")

        cleaned: List[Dict[str, Any]] = []
        for a in actions[:max_actions]:
            if isinstance(a, dict) and a.get("action"):
                cleaned.append(a)

        if not cleaned:
            raise ValueError(f"Planner returned no usable actions: {actions}")

        return cleaned
