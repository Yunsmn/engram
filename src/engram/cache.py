"""Content-hash cache for expensive deterministic calls (summaries, embeddings, routing, API).

Keys are sha256 of a canonical JSON encoding of (namespace, payload), so identical inputs
across runs and across systems hit the same entry.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def content_hash(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class Cache:
    """A tiny persistent KV store. Values are JSON (or raw bytes for embeddings)."""

    def __init__(self, path: Path | str | None) -> None:
        self._lock = threading.Lock()
        target = ":memory:" if path is None else str(path)
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(target, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, ns TEXT, value BLOB)"
        )
        self.hits = 0
        self.misses = 0

    def get(self, namespace: str, payload: Any) -> Any | None:
        key = content_hash([namespace, payload])
        with self._lock:
            row = self._conn.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        value = row[0]
        return value if isinstance(value, bytes) else json.loads(value)

    def put(self, namespace: str, payload: Any, value: Any) -> None:
        key = content_hash([namespace, payload])
        stored = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO cache (key, ns, value) VALUES (?, ?, ?)",
                (key, namespace, stored),
            )
            self._conn.commit()

    def memoize(self, namespace: str, payload: Any, compute: Callable[[], T]) -> T:
        cached = self.get(namespace, payload)
        if cached is not None:
            return cached
        value = compute()
        self.put(namespace, payload, value)
        return value

    def close(self) -> None:
        with self._lock:
            self._conn.close()
