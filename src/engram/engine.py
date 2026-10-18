"""Engram: the agent says what to remember and what it wants back; the engine applies fixed rules.

remember(): drop recalled text, drop exact duplicates, store the original (compressed), the agent's summary and
dated facts, and embedded chunks. Every write is a new memory; nothing is overwritten or deleted automatically.
recall(): see engram.read.recall. forget()/restore()/purge(): explicit deletion only.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from engram.cache import Cache, content_hash
from engram.config import Config
from engram.metrics import Metrics
from engram.read.markers import strip_marked
from engram.read.recall import RecallResult, recall as run_recall
from engram.store.chunking import chunk_text
from engram.store.db import Store
from engram.store.embed import Embedder, SentenceTransformerEmbedder


@dataclass(frozen=True)
class WriteResult:
    action: str  # "new" | "exact duplicate" | "nothing new"
    memory_id: int | None


class Engram:
    def __init__(self, store: Store, embedder: Embedder, config: Config | None = None,
                 metrics: Metrics | None = None):
        self.store = store
        self.embedder = embedder
        self.config = config or Config()
        self.metrics = metrics or Metrics()
        self._lock = threading.RLock()
        self.store.bind_embedder(getattr(embedder, "model_name", type(embedder).__name__))

    @classmethod
    def open(cls, db_path: Path | str, config: Config | None = None, metrics: Metrics | None = None) -> "Engram":
        """A store at `db_path` with the default embedder (e5-large-v2, local) and a shared embedding cache."""
        config = config or Config.from_env()
        return cls(Store(db_path), SentenceTransformerEmbedder.from_config(config, Cache(config.cache_path)),
                   config, metrics)

    # ---- write ----------------------------------------------------------------------------

    def clean_incoming(self, text: str) -> str:
        """Text the engine handed out (marked by recall) is never stored again as new evidence."""
        if self.config.strip_recalled:
            text, unclosed = strip_marked(text)
            if unclosed:
                self.metrics.incr("writes.unclosed_marker", unclosed)
        return text

    def remember(self, text: str, summary: str, source: str = "", timestamp: str | None = None,
                 facts: Sequence[str] | None = None) -> WriteResult:
        """Store `text` with the summary (and optional dated facts) written by the calling agent."""
        if not text.strip():
            raise ValueError("remember() needs non-empty text")
        if not summary or not summary.strip():
            raise ValueError("remember() needs the agent's summary of the text")
        text = self.clean_incoming(text)
        if not text.strip():  # everything was recalled content: nothing new to store
            self.metrics.incr("writes.nothing_new")
            return WriteResult("nothing new", None)
        with self._lock:
            self.metrics.incr("writes")
            digest = content_hash(text)
            existing = self.store.find_by_hash(digest)
            if existing is not None:
                self.metrics.incr("writes.exact_duplicate")
                return WriteResult("exact duplicate", existing)
            chunks = chunk_text(text, self.config.chunk_max_chars, self.config.chunk_min_chars)
            embeddings = self.embedder.embed(chunks)
            with self.store.transaction():
                memory_id = self.store.insert_memory(text, summary.strip(), source, digest, timestamp)
                self.store.replace_chunks(memory_id, chunks, embeddings)
            facts = [f.strip() for f in (facts or []) if f and f.strip()]
            if facts:
                self.store.add_facts(memory_id, facts, self.embedder.embed(facts),
                                     timestamp or self.store.get_memory(memory_id).created_at)
                self.metrics.incr("writes.facts", len(facts))
            self.metrics.incr("writes.new")
            return WriteResult("new", memory_id)

    # ---- read -----------------------------------------------------------------------------

    def recall(self, query: str, mode: str = "plan", k: int | None = None,
               token_budget: int | None = None) -> RecallResult:
        with self._lock:
            result = run_recall(self, query, mode=mode, k=k or self.config.recall_k,
                                token_budget=token_budget or self.config.recall_token_budget)
            self.metrics.incr("recalls")
            self.metrics.incr("recall.tokens", result.tokens)
            return result

    # ---- delete (explicit only: the engine never deletes on its own) ----------------------

    def forget(self, memory_ids: Sequence[int]) -> int:
        """Logical delete: the memories leave recall but stay in the store; restore() undoes it."""
        with self._lock:
            return self.store.forget(list(memory_ids))

    def restore(self, memory_ids: Sequence[int]) -> int:
        with self._lock:
            return self.store.restore(list(memory_ids))

    def purge(self, memory_ids: Sequence[int]) -> int:
        """Hard delete (not undoable). The embedding cache is a separate file the operator owns."""
        with self._lock:
            return self.store.purge(list(memory_ids))
