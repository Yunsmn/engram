"""LLM interface: text in, text out. Used only for summaries, topic names, merges, and
(in evaluation) answering, judging and the baseline agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from engram.cache import Cache
from engram.metrics import Metrics


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int
    output_tokens: int


class ContextOverflow(RuntimeError):
    """The prompt plus max_tokens did not fit the model's context window (counts in the model's own tokens)."""

    def __init__(self, n_prompt: int, n_ctx: int) -> None:
        super().__init__(f"prompt of {n_prompt} tokens exceeds context of {n_ctx}")
        self.n_prompt = n_prompt
        self.n_ctx = n_ctx


class LLM(Protocol):
    name: str

    def complete(self, prompt: str, system: str = "", max_tokens: int = 512) -> Completion: ...


class CachedLLM:
    """Content-hash cache + token/call accounting around any LLM. Temperature is always 0."""

    def __init__(self, inner: LLM, cache: Cache | None, metrics: Metrics | None = None, role: str = "llm"):
        self.inner = inner
        self.name = inner.name
        self.role = role
        self._cache = cache
        self._metrics = metrics or Metrics()

    def complete(self, prompt: str, system: str = "", max_tokens: int = 512) -> Completion:
        key = {"p": prompt, "s": system, "m": max_tokens}
        namespace = f"llm:{self.name}"
        hit = self._cache.get(namespace, key) if self._cache is not None else None
        if hit is not None:
            self._metrics.incr(f"{self.role}.cached")
            result = Completion(**hit)
        else:
            with self._metrics.timed(f"{self.role}.uncached"):
                result = self.inner.complete(prompt, system, max_tokens)
            if self._cache is not None:
                self._cache.put(namespace, key, result.__dict__)
        # logical accounting: a cache hit still counts as the call the system made
        self._metrics.incr(f"{self.role}.calls")
        self._metrics.incr(f"{self.role}.input_tokens", result.input_tokens)
        self._metrics.incr(f"{self.role}.output_tokens", result.output_tokens)
        return result


class EchoLLM:
    """Test double: returns a deterministic function of the prompt."""

    name = "echo"

    def __init__(self, fn=None):
        self._fn = fn or (lambda prompt, system: prompt.strip().splitlines()[-1][:200])

    def complete(self, prompt: str, system: str = "", max_tokens: int = 512) -> Completion:
        text = self._fn(prompt, system)
        return Completion(text, len(prompt) // 4, len(text) // 4)
