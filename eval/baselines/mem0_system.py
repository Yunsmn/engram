"""Mem0 (open-source `mem0ai`) behind the harness's MemorySystem interface — for the normalized baseline run.

Normalization: Mem0's internal LLM is the protocol's model (OpenAI-compatible endpoint, e.g. gpt-4.1-mini on
Azure), embeddings are local sentence-transformers (all-MiniLM-L6-v2, as engram uses), the vector store is an
on-disk Qdrant per question. Sessions are added in date order with their date, as Mem0's own LoCoMo
evaluation does; recall returns Mem0's top-k memories, dated, packed into the same token budget as every system.
Runs where `mem0ai` can be installed (Kaggle CPU notebook); not imported by the local harness.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
import tempfile

from engram.read.budget import pack

TOP_K = int(os.environ.get("MEM0_TOP_K", "30"))  # Mem0 retrieves k memories; the token budget cuts the rest


class Mem0System:
    name = "mem0"

    def __init__(self, user_id: str, base_url: str, api_key: str, model: str, store_dir: str | None = None):
        from mem0 import Memory  # imported lazily: only available where mem0ai is installed

        self._user = user_id
        self._dir = store_dir or tempfile.mkdtemp(prefix="mem0-")
        self.memory = Memory.from_config({
            "llm": {"provider": "openai", "config": {"model": model, "api_key": api_key,
                                                    "openai_base_url": base_url, "temperature": 0}},
            "embedder": {"provider": "huggingface", "config": {"model": "sentence-transformers/all-MiniLM-L6-v2"}},
            "vector_store": {"provider": "qdrant", "config": {"path": os.path.join(self._dir, "qdrant"),
                                                              "collection_name": user_id[:60],
                                                              "embedding_model_dims": 384, "on_disk": True}},
            "history_db_path": os.path.join(self._dir, "history.db"),
        })
        self.metrics = None
        self.content_filter_fallbacks = 0

    def remember(self, text: str, source: str = "", timestamp: str | None = None,
                 turns: Sequence[tuple[str, str]] | None = None) -> None:
        """`turns` = (role or speaker name, content) per turn — preferred: a turn may contain blank lines, so
        splitting `text` on them would cut multi-paragraph turns into separate, mis-attributed messages."""
        messages = [{"role": "user", "content": f"[This conversation took place on {timestamp}]"}] if timestamp else []
        if turns is None:  # fallback for callers without turns: one paragraph per message
            turns = [tuple(block.partition(": ")[::2]) for block in text.split("\n\n")]
        for role, content in turns:
            if role in ("user", "assistant"):
                messages.append({"role": role, "content": content})
            else:  # LoCoMo: two named people; both speak as "user", the name is kept in the text
                messages.append({"role": "user", "content": f"[{role}]: {content}"})
        metadata = {"date": timestamp or "", "source": source}
        try:
            self.memory.add(messages, user_id=self._user, metadata=metadata)
        except Exception as exc:  # noqa: BLE001 — mem0 wraps provider errors in its own LLMError
            if "content_filter" not in str(exc):
                raise
            # The provider refused Mem0's extraction prompt for this session. Like engram's clipped-text fallback,
            # keep the session anyway with Mem0's own no-LLM mode (raw messages stored as memories), and count it.
            self.content_filter_fallbacks += 1
            self.memory.add(messages, user_id=self._user, metadata=metadata, infer=False)

    def finish_ingest(self) -> None:
        pass

    def recall(self, query: str, token_budget: int) -> str:
        try:  # mem0ai >= 1.x scopes searches through filters; older versions took user_id directly
            hits = self.memory.search(query, filters={"user_id": self._user}, limit=TOP_K)
        except TypeError:
            hits = self.memory.search(query, user_id=self._user, limit=TOP_K)
        results = hits.get("results", hits) if isinstance(hits, dict) else hits
        blocks = [f"[{(h.get('metadata') or {}).get('date', '')}] {h['memory']}" for h in results]
        return pack(blocks, token_budget)
