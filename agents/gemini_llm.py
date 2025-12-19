import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, DeadlineExceeded, NotFound


class GeminiLLM:
    """Thin LLM wrapper with:
    - RPM throttling (avoid 429s)
    - small on-disk cache (so reruns don't burn quota)
    - nicer error messages for daily caps

    NOTE: This is still a "real" model call (no fake / stub).
    """

    def __init__(self, api_key: str, model_name: str, rpm_limit: int = 2, timeout_s: int = 45, cache_path: str = "runs/_llm_cache.jsonl"):
        genai.configure(api_key=api_key)

        # The SDK accepts "models/<name>"
        if not model_name.startswith("models/"):
            model_name = f"models/{model_name}"

        self.model_name = model_name
        self.rpm_limit = max(1, int(rpm_limit))
        self.timeout_s = timeout_s

        self.model = genai.GenerativeModel(model_name)

        self._last_call_ts = 0.0

        # Very small cache (key -> response text)
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache = {}
        if self.cache_path.exists():
            try:
                for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    self._cache[obj["k"]] = obj["v"]
            except Exception:
                # If cache is corrupt, just ignore it.
                self._cache = {}

    def _cache_key(self, system_prompt: str, user_prompt: str) -> str:
        h = hashlib.sha256()
        h.update(self.model_name.encode("utf-8"))
        h.update(b"\n---\n")
        h.update(system_prompt.encode("utf-8"))
        h.update(b"\n---\n")
        h.update(user_prompt.encode("utf-8"))
        return h.hexdigest()

    def _throttle(self):
        # Simple RPM throttle (requests per minute)
        min_interval = 60.0 / float(self.rpm_limit)
        now = time.time()
        dt = now - self._last_call_ts
        if dt < min_interval:
            time.sleep(min_interval - dt)
        self._last_call_ts = time.time()

    @staticmethod
    def _retry_after_seconds(msg: str) -> Optional[float]:
        # Gemini error text often includes:
        # "Please retry in 39.264860182s."
        m = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)s", msg, re.IGNORECASE)
        if not m:
            return None
        try:
            return float(m.group(1))
        except Exception:
            return None

    @staticmethod
    def _is_daily_cap(msg: str) -> bool:
        # Daily cap errors usually mention "GenerateRequestsPerDay".
        return "GenerateRequestsPerDay" in msg or "PerDay" in msg

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        k = self._cache_key(system_prompt, user_prompt)
        if k in self._cache:
            return self._cache[k]

        self._throttle()

        prompt = system_prompt.strip() + "\n\n" + user_prompt.strip()

        # Keep retries small. If we hit a daily cap, retries won't help.
        last_err = None
        for attempt in range(1, 7):
            try:
                resp = self.model.generate_content(
                    prompt,
                    request_options={"timeout": self.timeout_s},
                )
                text = (resp.text or "").strip()
                if not text:
                    raise RuntimeError("Empty response text from Gemini")

                # Save to cache
                self._cache[k] = text
                with open(self.cache_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"k": k, "v": text}, ensure_ascii=False) + "\n")

                return text

            except ResourceExhausted as e:
                msg = str(e)
                last_err = e

                # If this is a daily cap, stop fast with a helpful message.
                if self._is_daily_cap(msg):
                    raise RuntimeError(
                        "Gemini free-tier DAILY quota hit for this model/key.\n"

                        "Fix: switch to a fresh API key OR switch model (ex: gemini-2.5-pro) OR enable billing.\n"

                        f"Raw error: {msg}"
                    ) from e

                # Otherwise it's usually RPM/token rate. Obey retry_after if present.
                wait_s = self._retry_after_seconds(msg)
                if wait_s is None:
                    wait_s = min(8 * attempt, 30)
                time.sleep(wait_s + 0.5)

            except (DeadlineExceeded,) as e:
                last_err = e
                time.sleep(min(2 * attempt, 10))

            except NotFound as e:
                # Model name mismatch. Fail fast.
                raise RuntimeError(f"Model not found: {self.model_name}. Raw: {e}") from e

            except Exception as e:
                last_err = e
                time.sleep(min(2 * attempt, 10))

        raise RuntimeError(f"GeminiLLM.generate failed after retries. Last error: {last_err!r}")
