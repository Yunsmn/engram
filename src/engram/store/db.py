"""SQLite storage. All SQL lives here."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np

from engram.store.compress import compress, decompress
from engram.store.vectors import from_blob, to_blob

ACTIVE = "forgotten_at IS NULL"  # forgotten memories stay in the table but out of every read path
_MEMORY_COLS = "id, summary, created_at, updated_at, source, content_hash, access_count, forgotten_at"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Memory:
    id: int
    summary: str
    created_at: str
    updated_at: str
    source: str
    content_hash: str
    access_count: int
    forgotten_at: str | None = None


@dataclass(frozen=True)
class Chunk:
    id: int
    memory_id: int
    seq: int
    text: str
    embedding: np.ndarray


@dataclass(frozen=True)
class Fact:
    id: int
    memory_id: int
    text: str
    created_at: str
    embedding: np.ndarray


class Store:
    def __init__(self, path: Path | str = ":memory:") -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)  # guarded by Engram's lock
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA secure_delete=ON")  # purge() overwrites freed pages instead of leaving the text
        self.conn.executescript(resources.files("engram.store").joinpath("schema.sql").read_text())
        self._depth = 0

    def _commit(self) -> None:
        if self._depth == 0:
            self.conn.commit()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Group several writes atomically (nested calls join the outer transaction)."""
        self._depth += 1
        try:
            yield
        except BaseException:
            self._depth -= 1
            if self._depth == 0:
                self.conn.rollback()
            raise
        self._depth -= 1
        self._commit()

    def close(self) -> None:
        self.conn.close()

    def bind_embedder(self, model_name: str) -> None:
        """Vectors from different embedders are not comparable: a store keeps the model that built it."""
        row = self.conn.execute("SELECT value FROM meta WHERE key='embed_model'").fetchone()
        if row is None:
            if self.conn.execute("SELECT 1 FROM chunks LIMIT 1").fetchone():
                return  # built before stores recorded their embedder: unknown model, so leave it unbound
            self.conn.execute("INSERT INTO meta (key, value) VALUES ('embed_model', ?)", (model_name,))
            self._commit()
        elif row[0] != model_name:
            raise ValueError(f"this store was built with embedder {row[0]!r}, not {model_name!r}; "
                             f"set ENGRAM_EMBED_MODEL={row[0]} (and its prefixes) or rebuild the store")

    # ---- memories -------------------------------------------------------------------------

    def insert_memory(self, full_text: str, summary: str, source: str, content_hash: str,
                      created_at: str | None) -> int:
        ts = created_at or now_iso()
        cur = self.conn.execute(
            "INSERT INTO memories (summary, compressed_full, created_at, updated_at, source, content_hash)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (summary, compress(full_text), ts, ts, source, content_hash),
        )
        self._commit()
        return int(cur.lastrowid)

    def get_memory(self, memory_id: int) -> Memory | None:
        row = self.conn.execute(f"SELECT {_MEMORY_COLS} FROM memories WHERE id=?", (memory_id,)).fetchone()
        return Memory(*row) if row else None

    def get_memories(self, ids: Sequence[int]) -> list[Memory]:
        if not ids:
            return []
        marks = ",".join("?" * len(ids))
        rows = self.conn.execute(f"SELECT {_MEMORY_COLS} FROM memories WHERE id IN ({marks})", list(ids))
        by_id = {r[0]: Memory(*r) for r in rows}
        return [by_id[i] for i in ids if i in by_id]

    def full_text(self, memory_id: int) -> str:
        row = self.conn.execute("SELECT compressed_full FROM memories WHERE id=?", (memory_id,)).fetchone()
        if row is None:
            raise KeyError(memory_id)
        return decompress(row[0])

    def find_by_hash(self, content_hash: str) -> int | None:
        row = self.conn.execute(f"SELECT id FROM memories WHERE content_hash=? AND {ACTIVE} ORDER BY id LIMIT 1",
                                (content_hash,)).fetchone()
        return int(row[0]) if row else None

    def bump_access(self, ids: Iterable[int]) -> None:
        self.conn.executemany("UPDATE memories SET access_count=access_count+1 WHERE id=?", [(i,) for i in ids])
        self._commit()

    def all_memory_ids(self) -> list[int]:
        return [r[0] for r in self.conn.execute(f"SELECT id FROM memories WHERE {ACTIVE} ORDER BY id")]

    def count_memories(self) -> int:
        return len(self.all_memory_ids())

    # ---- explicit deletion ----------------------------------------------------------------

    def forget(self, ids: Sequence[int], at: str | None = None) -> int:
        """Logical delete: out of recall and duplicate checks, kept in the table (restore() undoes it)."""
        return self._set_forgotten(ids, at or now_iso(), ACTIVE)

    def restore(self, ids: Sequence[int]) -> int:
        return self._set_forgotten(ids, None, "forgotten_at IS NOT NULL")

    def _set_forgotten(self, ids: Sequence[int], value: str | None, only: str) -> int:
        if not ids:
            return 0
        marks = ",".join("?" * len(ids))
        cur = self.conn.execute(f"UPDATE memories SET forgotten_at=? WHERE id IN ({marks}) AND {only}",
                                [value, *ids])
        self._commit()
        return cur.rowcount

    def purge(self, ids: Sequence[int]) -> int:
        """Hard delete: the memory, its chunks and facts. Freed pages are overwritten (secure_delete) and the WAL
        is checkpointed, so the text does not linger in the database files."""
        if not ids:
            return 0
        marks = ",".join("?" * len(ids))
        with self.transaction():
            for table in ("facts", "chunks"):
                self.conn.execute(f"DELETE FROM {table} WHERE memory_id IN ({marks})", list(ids))
            removed = self.conn.execute(f"DELETE FROM memories WHERE id IN ({marks})", list(ids)).rowcount
        self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return removed

    # ---- chunks and facts -----------------------------------------------------------------

    def replace_chunks(self, memory_id: int, texts: Sequence[str], embeddings: np.ndarray) -> None:
        self.conn.execute("DELETE FROM chunks WHERE memory_id=?", (memory_id,))
        self.conn.executemany(
            "INSERT INTO chunks (memory_id, seq, text, embedding) VALUES (?, ?, ?, ?)",
            [(memory_id, i, t, to_blob(e)) for i, (t, e) in enumerate(zip(texts, embeddings))],
        )
        self._commit()

    def add_facts(self, memory_id: int, texts: Sequence[str], embeddings: np.ndarray, created_at: str) -> None:
        self.conn.executemany(
            "INSERT INTO facts (memory_id, text, created_at, embedding) VALUES (?, ?, ?, ?)",
            [(memory_id, t, created_at, to_blob(e)) for t, e in zip(texts, embeddings)],
        )
        self._commit()

    def facts_for(self, memory_ids: Sequence[int] | None = None) -> list[Fact]:
        cols = "SELECT id, memory_id, text, created_at, embedding FROM facts"
        if memory_ids is None:
            rows = self.conn.execute(f"{cols} WHERE memory_id IN (SELECT id FROM memories WHERE {ACTIVE})")
        elif not memory_ids:
            return []
        else:
            marks = ",".join("?" * len(memory_ids))
            rows = self.conn.execute(f"{cols} WHERE memory_id IN ({marks})", list(memory_ids))
        return [Fact(r[0], r[1], r[2], r[3], from_blob(r[4])) for r in rows]

    def chunks_for(self, memory_ids: Sequence[int] | None = None) -> list[Chunk]:
        if memory_ids is None:
            rows = self.conn.execute("SELECT id, memory_id, seq, text, embedding FROM chunks"
                                     f" WHERE memory_id IN (SELECT id FROM memories WHERE {ACTIVE})")
        elif not memory_ids:
            return []
        else:
            marks = ",".join("?" * len(memory_ids))
            rows = self.conn.execute(
                f"SELECT id, memory_id, seq, text, embedding FROM chunks WHERE memory_id IN ({marks})",
                list(memory_ids),
            )
        return [Chunk(r[0], r[1], r[2], r[3], from_blob(r[4])) for r in rows]
