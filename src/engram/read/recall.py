"""recall(): rank memories, choose a layout from the question's shape, pack it into a token budget.

Memories are ranked by their best chunk's cosine similarity to the query; recency and usage only break near-ties.
Layouts (the `plan` default picks one, no model call):
- time questions:      the best-matching dated facts first (FACT_SHARE of the budget), then the layout below
- counting / listing:  `full`   -- dated summaries of the top-k memories, then their original conversations
- everything else:     `hybrid` -- dated summaries of the top-k (HYBRID_SUMMARY_SHARE), then the best-matching
                       dated chunks over all memories
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from engram.read.budget import count_tokens, pack
from engram.read.markers import mark
from engram.read.plan import is_aggregation, is_time_shaped
from engram.store.vectors import top_k

if TYPE_CHECKING:
    from engram.engine import Engram

MODES = ("plan", "hybrid", "full", "summary")
HYBRID_SUMMARY_SHARE = 0.4
FACT_SHARE = 0.35
RECENCY_WEIGHT = 0.02
USAGE_WEIGHT = 0.005
MIN_DETAIL_TOKENS = 40
FACTS_LABEL = "FACTS:\n"


@dataclass(frozen=True)
class RecallResult:
    text: str
    mode: str
    tokens: int
    memory_ids: tuple[int, ...]

    @property
    def marked(self) -> str:
        """The text wrapped in <memory> markers: what an agent should paste into a conversation, so that
        remember() never stores recalled content again as new evidence."""
        return mark(self.text)


def rank_memories(engine: "Engram", query_vec: np.ndarray, pool: int) -> list[tuple[int, float]]:
    store = engine.store
    chunks = store.chunks_for(store.all_memory_ids())
    if not chunks:
        return []
    matrix = np.vstack([c.embedding for c in chunks])
    best: dict[int, float] = {}
    for idx, score in top_k(query_vec, matrix, len(chunks)):
        best.setdefault(chunks[idx].memory_id, score)
    shortlist = sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:pool]
    memories = {m.id: m for m in store.get_memories([m for m, _ in shortlist])}
    newest = sorted(memories.values(), key=lambda m: m.created_at)
    recency = {m.id: (i + 1) / len(newest) for i, m in enumerate(newest)}
    scored = [(mid, s + RECENCY_WEIGHT * recency[mid] + USAGE_WEIGHT * np.log1p(memories[mid].access_count))
              for mid, s in shortlist if mid in memories]
    return sorted(scored, key=lambda kv: kv[1], reverse=True)


def _when(m) -> str:
    first, last = m.created_at[:16], m.updated_at[:16]
    return first if first == last else f"{first} .. {last}"


def _summary_blocks(engine: "Engram", ids: list[int]) -> list[str]:
    return [f"[{_when(m)}] {m.summary}" for m in engine.store.get_memories(ids)]


def _full(engine: "Engram", ids: list[int], budget: int) -> str:
    head = pack(_summary_blocks(engine, ids), budget)
    detail = [f"[{_when(m)}]\n{engine.store.full_text(m.id)}" for m in engine.store.get_memories(ids)]
    remaining = budget - count_tokens(head) - 2
    tail = pack(detail, remaining) if remaining > MIN_DETAIL_TOKENS else ""
    return f"{head}\n\n{tail}" if tail else head


def _hybrid(engine: "Engram", ids: list[int], query_vec: np.ndarray, budget: int) -> str:
    head = pack(_summary_blocks(engine, ids), int(budget * HYBRID_SUMMARY_SHARE))
    chunks = engine.store.chunks_for(engine.store.all_memory_ids())
    if not chunks:
        return head
    dates = {m.id: m.created_at[:16] for m in engine.store.get_memories(sorted({c.memory_id for c in chunks}))}
    sims = np.vstack([c.embedding for c in chunks]) @ query_vec
    order = [int(i) for i in np.argsort(-sims, kind="stable")]
    tail = pack([f"[{dates[chunks[i].memory_id]}] {chunks[i].text}" for i in order], budget - count_tokens(head) - 2)
    return f"{head}\n\n{tail}" if tail else head


def recall(engine: "Engram", query: str, mode: str, k: int, token_budget: int) -> RecallResult:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    query_vec = engine.embedder.embed_query([query])[0]
    ranked = rank_memories(engine, query_vec, engine.config.candidate_pool)
    ids = [mid for mid, _ in ranked[:k]]
    if not ids:
        return RecallResult("", mode, 0, ())
    facts = engine.store.facts_for(None) if mode == "plan" and is_time_shaped(query) else []
    facts_head, budget = "", token_budget
    if facts:  # the rest of the budget is reserved as soon as facts exist (identical to the evaluated version)
        sims = np.vstack([f.embedding for f in facts]) @ query_vec
        facts_head = pack([f"[{facts[i].created_at[:16]}] {facts[i].text}" for i in np.argsort(-sims)],
                          int(token_budget * FACT_SHARE))
        budget = token_budget - count_tokens(facts_head) - 2 - count_tokens(FACTS_LABEL)
    layout = mode if mode != "plan" else ("full" if is_aggregation(query) else "hybrid")
    if engine.config.track_access:
        engine.store.bump_access(ids)
    if layout == "hybrid":
        text = _hybrid(engine, ids, query_vec, budget)
    elif layout == "full":
        text = _full(engine, ids, budget)
    else:
        text = pack(_summary_blocks(engine, ids), budget)
    if facts:
        text = f"{FACTS_LABEL}{facts_head}\n\n{text}" if facts_head else text
        layout = f"facts+{layout}"
    return RecallResult(text, layout, count_tokens(text), tuple(ids))
