"""Flat vector RAG over raw text: chunk, embed, top-k by cosine. No organization, no LLM."""

from __future__ import annotations

import numpy as np

from engram.config import Config
from engram.metrics import Metrics
from engram.read.budget import pack
from engram.store.chunking import chunk_text
from engram.store.embed import Embedder
from engram.store.vectors import top_k

TOP_K = 40


class FlatRAG:
    name = "flat_rag"

    def __init__(self, embedder: Embedder, config: Config | None = None, metrics: Metrics | None = None):
        self.config = config or Config()
        self.metrics = metrics or Metrics()
        self._embedder = embedder
        self._texts: list[str] = []
        self._sources: list[str] = []
        self._vectors: list[np.ndarray] = []

    def remember(self, text: str, source: str = "", timestamp: str | None = None) -> None:
        chunks = chunk_text(text, self.config.chunk_max_chars, self.config.chunk_min_chars)
        if not chunks:
            return
        vectors = self._embedder.embed(chunks)
        prefix = f"[{timestamp}] " if timestamp else ""
        self._texts.extend(prefix + c for c in chunks)
        self._sources.extend([source] * len(chunks))
        self._vectors.extend(vectors)
        self.metrics.incr("writes")

    def finish_ingest(self) -> None:
        return None

    def recall(self, query: str, token_budget: int) -> str:
        if not self._texts:
            return ""
        matrix = np.vstack(self._vectors)
        hits = top_k(self._embedder.embed_query([query])[0], matrix, TOP_K)
        return pack([self._texts[i] for i, _ in hits], token_budget)
