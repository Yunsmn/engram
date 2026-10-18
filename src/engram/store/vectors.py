"""Brute-force cosine search. Fine for the 10^4-10^5 chunk scale this prototype targets."""

from __future__ import annotations

import numpy as np


def to_blob(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def top_k(query: np.ndarray, matrix: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Return (row_index, cosine) for the k best rows. Inputs must be L2-normalised."""
    if matrix.shape[0] == 0 or k <= 0:
        return []
    scores = matrix @ query
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [(int(i), float(scores[i])) for i in idx]
