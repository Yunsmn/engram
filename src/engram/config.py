"""Engine configuration. Every setting has a default; ENGRAM_<NAME> environment variables override it."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace
from pathlib import Path


@dataclass(frozen=True)
class Config:
    cache_path: Path = Path("data/cache.sqlite")  # embedding cache, shared by stores
    # chunking
    chunk_max_chars: int = 800
    chunk_min_chars: int = 120
    # embeddings (e5 is asymmetric: queries and stored text get different prefixes)
    embed_model: str = "intfloat/e5-large-v2"
    embed_device: str = "cpu"
    embed_query_prefix: str = "query: "
    embed_passage_prefix: str = "passage: "
    # recall
    recall_k: int = 5  # memories whose summaries are shown
    recall_token_budget: int = 1300
    candidate_pool: int = 50
    track_access: int = 1  # count how often each memory is recalled (a ranking tie-breaker)
    # writes
    strip_recalled: int = 1  # remember() drops <memory>...</memory> spans, i.e. text recall() handed out

    @classmethod
    def from_env(cls, **overrides: object) -> "Config":
        """Defaults, then ENGRAM_<FIELD> environment variables, then explicit overrides."""
        values: dict[str, object] = {}
        for f in fields(cls):
            raw = os.environ.get(f"ENGRAM_{f.name.upper()}")
            if raw is not None:
                values[f.name] = _coerce(raw, f.type)
        return replace(cls(), **{**values, **overrides})


def _coerce(raw: str, type_name: object) -> object:
    name = str(type_name)
    if "int" in name:
        return int(raw)
    if "float" in name:
        return float(raw)
    if "Path" in name:
        return Path(raw)
    return raw
