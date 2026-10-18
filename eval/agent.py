"""The agent's side of a write, as used in the evaluation: the protocol's model plays the agent and writes the
summary and the dated facts of each stored conversation. (In real use, your agent writes them.)

Prompts are kept verbatim from the evaluated runs, so cached responses match them exactly. `offline=True` reads
summaries and facts from the cache only and never calls the API (e.g. a Kaggle run with a pre-generated cache).
"""

from __future__ import annotations

import json
import re

from engram.cache import Cache
from eval.llm.base import CachedLLM
from eval.llm.http import HTTPError

SUMMARY_MAX_WORDS = 60
SUMMARY_MAX_TOKENS = 200
EMPTY_SUMMARY_FALLBACK_CHARS = 400
SUMMARY_SYSTEM = 'You write compact, factual memory summaries. Never invent facts.'
SUMMARY_PROMPT = 'Summarize the conversation below in at most {max_words} words.\nKeep every concrete fact the participants state about themselves or their lives (names, numbers, places, dates,\npreferences, plans, purchases, changes), attributing each fact to the right person. Output only the summary.\n\nCONVERSATION:\n{text}'
FACTS_SYSTEM = 'You extract memory facts from conversations. Output JSON only. Never invent facts.'
FACTS_PROMPT = 'This conversation took place on {date}.\nList the facts worth remembering about the people in it (preferences, events, plans, possessions, relationships,\nopinions, changes). Each fact is ONE self-contained sentence: name the person (or "the user"), keep exact names and\nnumbers, and rewrite relative times ("yesterday", "last Friday", "next month") as absolute dates or periods computed\nfrom the conversation date. Output {{"facts": ["...", "..."]}} (empty list if none).\n\nCONVERSATION:\n{text}'
FACTS_MAX_TOKENS = 1500
SUMMARY_INPUT_CHARS = 9000
ASSISTANT_TURN_CHARS = 300
_TURN = re.compile('^(user|assistant):', 40)


def summary_input(text: str, limit: int = SUMMARY_INPUT_CHARS) -> str:
    """Shrink chat transcripts for the summarizer: keep user turns, clip assistant turns.

    The full text is always stored and chunked losslessly; this only bounds summarizer cost.
    """
    if _TURN.search(text):
        parts = re.split(r"(?=^(?:user|assistant):)", text, flags=re.M)
        clipped = [p if not p.startswith("assistant:") or len(p) <= ASSISTANT_TURN_CHARS
                   else p[:ASSISTANT_TURN_CHARS] + " [...]\n\n" for p in parts]
        text = "".join(clipped)
    return text if len(text) <= limit else text[:limit] + " [...]"


def parse_facts(raw: str) -> list[str]:
    m = re.search(r"\{.*\}", raw, re.S)
    facts = json.loads(m.group(0))["facts"] if m else []
    return [f.strip() for f in facts if isinstance(f, str) and f.strip()]


class Agent:
    """Writes the summary and the dated facts of a conversation with the given LLM (through the shared cache)."""

    def __init__(self, llm: CachedLLM, cache: Cache, offline: bool = False):
        self.llm, self.cache, self.offline = llm, cache, offline
        self.fallbacks = 0

    def _complete(self, prompt: str, system: str, max_tokens: int) -> str | None:
        if self.offline:
            hit = self.cache.get(f"llm:{self.llm.name}", {"p": prompt, "s": system, "m": max_tokens})
            return hit["text"] if hit is not None else None
        return self.llm.complete(prompt, system=system, max_tokens=max_tokens).text

    def summary(self, text: str, date: str | None) -> str:
        prompt = SUMMARY_PROMPT.format(date=date or "an unknown date", max_words=SUMMARY_MAX_WORDS, text=text)
        try:
            out = self._complete(prompt, SUMMARY_SYSTEM, SUMMARY_MAX_TOKENS)
        except HTTPError as exc:  # the provider refused (content filter): the session's opening text, clipped
            if exc.status != 400:
                raise
            out = None
        if out is None:
            self.fallbacks += 1
            return summary_input(text, EMPTY_SUMMARY_FALLBACK_CHARS)
        return out

    def facts(self, text: str, date: str | None) -> list[str]:
        prompt = FACTS_PROMPT.format(date=date or "an unknown date", text=text)
        try:
            out = self._complete(prompt, FACTS_SYSTEM, FACTS_MAX_TOKENS)
            return parse_facts(out) if out is not None else []
        except Exception:  # noqa: BLE001 -- content filter / unparsable output: no facts for this conversation
            return []
