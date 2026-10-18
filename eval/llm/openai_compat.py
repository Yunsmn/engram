"""One OpenAI-compatible chat backend, configured per role from env vars:
<ROLE>_BASE_URL, <ROLE>_API_KEY, <ROLE>_MODEL, <ROLE>_RPM, <ROLE>_TPM (0 = unlimited)."""

from __future__ import annotations

import json
import os

from eval.llm.base import Completion
from eval.llm.http import RateLimiter, post_json_with_retry

CHARS_PER_TOKEN = 4  # rough estimate used only for client-side TPM pacing


class OpenAICompatLLM:
    def __init__(self, base_url: str, api_key: str, model: str, rpm: int = 0, tpm: int = 0, extra: dict | None = None):
        if not api_key:
            raise RuntimeError(f"missing API key for {base_url}")
        self._extra = extra or {}
        suffix = f"+{json.dumps(self._extra, sort_keys=True)}" if self._extra else ""
        self.name = f"{base_url.split('//')[-1].split('/')[0]}/{model}{suffix}"  # cache key includes extras
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self._model = model
        self._limiter = RateLimiter(rpm, tpm)

    @classmethod
    def from_env(cls, role: str) -> "OpenAICompatLLM":
        prefix = role.upper()

        def env(name: str, default: str = "") -> str:
            return os.environ.get(f"{prefix}_{name}", default).strip()

        extra = {"reasoning_effort": "low"} if "gpt-oss" in env("MODEL") else {}
        if env("EXTRA_BODY"):  # e.g. {"chat_template_kwargs": {"enable_thinking": false}}
            extra = {**extra, **json.loads(env("EXTRA_BODY"))}
        return cls(env("BASE_URL"), env("API_KEY"), env("MODEL"), int(env("RPM", "0") or 0),
                   int(env("TPM", "0") or 0), extra)

    def complete(self, prompt: str, system: str = "", max_tokens: int = 512) -> Completion:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        body = {"model": self._model, "messages": messages, "temperature": 0, "max_tokens": max_tokens, **self._extra}
        est = (len(prompt) + len(system)) // CHARS_PER_TOKEN + max_tokens
        data = post_json_with_retry(self._url, body, self._headers, limiter=self._limiter, est_tokens=est)
        usage = data.get("usage") or {}
        content = data["choices"][0]["message"].get("content") or ""
        return Completion(content.strip(), int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)))
