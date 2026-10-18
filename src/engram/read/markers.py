"""Recall markers (H19): text the engine hands back is wrapped so it can never be stored again as new evidence.

An agent that recalls memory, talks about it, and stores the conversation would otherwise re-store old content
under a new date. `mark()` wraps recalled text; `strip_marked()` removes every marked span before a write.
Deterministic, no model call. Text without markers is returned unchanged (byte for byte), so caches keyed by the
session text keep matching.
"""

from __future__ import annotations

import re

OPEN, CLOSE = "<memory>", "</memory>"
_SPAN = re.compile(re.escape(OPEN) + r".*?" + re.escape(CLOSE), re.S)


def mark(text: str) -> str:
    return f"{OPEN}\n{text}\n{CLOSE}" if text else text


def strip_marked(text: str) -> tuple[str, int]:
    """Remove closed marked spans. An unclosed OPEN loses only the marker itself (the text after it is the
    user's or the agent's and is kept). Returns (text, number of unclosed markers)."""
    if OPEN not in text and CLOSE not in text:
        return text, 0
    cleaned = _SPAN.sub("", text)
    unclosed = cleaned.count(OPEN)
    cleaned = cleaned.replace(OPEN, "").replace(CLOSE, "")
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip(), unclosed
