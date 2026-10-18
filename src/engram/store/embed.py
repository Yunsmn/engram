"""Sentence embeddings with a content-hash cache. Vectors are float32 and L2-normalised."""

from __future__ import annotations

import hashlib
from typing import Protocol, Sequence

import numpy as np

from engram.cache import Cache

EMBED_DIM = 384  # placeholder until the model loads (then its real dimension)


class Embedder(Protocol):
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Sentence-transformers embedder (the engine default is e5-large-v2, see Config). Loaded lazily.
    Asymmetric models (e5: "query: " / "passage: ") get their prefixes; stored text is embedded with
    `passage_prefix`, recall queries with `query_prefix`. Both empty for symmetric models like MiniLM."""

    def __init__(self, model_name: str, cache: Cache | None = None, device: str | None = None,
                 query_prefix: str = "", passage_prefix: str = ""):
        self.model_name = model_name
        self.dim = EMBED_DIM
        self._cache = cache
        self._device = device
        self._model = None
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix

    @classmethod
    def from_config(cls, config, cache: Cache | None = None) -> "SentenceTransformerEmbedder":
        return cls(config.embed_model, cache, device=config.embed_device,
                   query_prefix=config.embed_query_prefix, passage_prefix=config.embed_passage_prefix)

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self._device)
            self.dim = self._model.get_embedding_dimension()
        return self._model

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode([self._passage_prefix + t for t in texts])

    def embed_query(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode([self._query_prefix + t for t in texts])

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out: list[np.ndarray | None] = [None] * len(texts)
        missing: list[int] = []
        for i, text in enumerate(texts):
            hit = self._cache.get(f"embed:{self.model_name}", text) if self._cache else None
            if hit is not None:
                out[i] = np.frombuffer(hit, dtype=np.float32)
            else:
                missing.append(i)
        if missing:
            vectors = self._load().encode(
                [texts[i] for i in missing], normalize_embeddings=True, convert_to_numpy=True
            ).astype(np.float32)
            for i, vec in zip(missing, vectors):
                out[i] = vec
                if self._cache:
                    self._cache.put(f"embed:{self.model_name}", texts[i], vec.tobytes())
        return np.vstack(out)


class HashEmbedder:
    """Deterministic bag-of-words embedder for tests: shared words give positive cosine."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in text.lower().split():
                digest = hashlib.md5(word.strip(".,!?").encode()).digest()
                matrix[row, int.from_bytes(digest[:4], "little") % self.dim] += 1.0
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.where(norms == 0, 1.0, norms)

    def embed_query(self, texts: Sequence[str]) -> np.ndarray:
        return self.embed(texts)
