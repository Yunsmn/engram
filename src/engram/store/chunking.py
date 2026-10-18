"""Split text into retrieval chunks of about a paragraph or a few lines."""

from __future__ import annotations

import re

_PARAGRAPH = re.compile(r"\n\s*\n")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text: str, max_chars: int = 800, min_chars: int = 120) -> list[str]:
    """Paragraphs first; oversized paragraphs split by line then sentence; tiny ones merged."""
    pieces: list[str] = []
    for para in _PARAGRAPH.split(text):
        para = para.strip()
        if para:
            pieces.extend(_split_long(para, max_chars))
    return _merge_short(pieces, max_chars, min_chars)


def _split_long(para: str, max_chars: int) -> list[str]:
    if len(para) <= max_chars:
        return [para]
    units = [u for u in para.split("\n") if u.strip()]
    if len(units) == 1:
        units = _SENTENCE.split(para)
    out: list[str] = []
    current = ""
    for unit in units:
        unit = unit.strip()
        if len(unit) > max_chars:
            if current:
                out.append(current)
                current = ""
            out.extend(unit[i : i + max_chars] for i in range(0, len(unit), max_chars))
            continue
        candidate = f"{current}\n{unit}" if current else unit
        if len(candidate) > max_chars:
            out.append(current)
            current = unit
        else:
            current = candidate
    if current:
        out.append(current)
    return out


def _merge_short(pieces: list[str], max_chars: int, min_chars: int) -> list[str]:
    merged: list[str] = []
    for piece in pieces:
        if merged and (len(merged[-1]) < min_chars or len(piece) < min_chars):
            candidate = f"{merged[-1]}\n\n{piece}"
            if len(candidate) <= max_chars:
                merged[-1] = candidate
                continue
        merged.append(piece)
    return merged
