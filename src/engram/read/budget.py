"""Token counting and truncation. One tokenizer (tiktoken cl100k_base) for every system, so the
1,300-token recall budget means the same thing for the engine and every baseline."""

from __future__ import annotations

from functools import lru_cache

ENCODING = "cl100k_base"


@lru_cache(maxsize=1)
def _encoder():
    import tiktoken

    return tiktoken.get_encoding(ENCODING)


def count_tokens(text: str) -> int:
    return len(_encoder().encode(text, disallowed_special=()))


def truncate(text: str, budget: int) -> str:
    tokens = _encoder().encode(text, disallowed_special=())
    if len(tokens) <= budget:
        return text
    return _encoder().decode(tokens[:budget])


def pack(blocks: list[str], budget: int, separator: str = "\n\n") -> str:
    """Greedily add whole blocks in order; the first block that does not fit is truncated, then stop."""
    out: list[str] = []
    used = 0
    sep_cost = count_tokens(separator)
    for block in blocks:
        cost = count_tokens(block) + (sep_cost if out else 0)
        if used + cost <= budget:
            out.append(block)
            used += cost
            continue
        remaining = budget - used - (sep_cost if out else 0)
        if remaining > 20:
            out.append(truncate(block, remaining))
        break
    return separator.join(out)
