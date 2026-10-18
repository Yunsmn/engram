-- Memories are flat rows. The original text is kept compressed; summaries, facts and chunks are views on it.
-- The engine never deletes on its own: forget() sets forgotten_at (undoable), purge() removes rows for good.

CREATE TABLE IF NOT EXISTS memories (
    id              INTEGER PRIMARY KEY,
    summary         TEXT    NOT NULL,       -- written by the agent
    compressed_full BLOB    NOT NULL,       -- zstd-compressed original
    created_at      TEXT    NOT NULL,       -- ISO-8601 time of the conversation
    updated_at      TEXT    NOT NULL,
    source          TEXT    NOT NULL DEFAULT '',
    content_hash    TEXT    NOT NULL,
    access_count    INTEGER NOT NULL DEFAULT 0,  -- ranking tie-breaker only
    forgotten_at    TEXT                         -- set by forget(); NULL = active
);
CREATE INDEX IF NOT EXISTS idx_memories_hash ON memories(content_hash);

CREATE TABLE IF NOT EXISTS chunks (
    id        INTEGER PRIMARY KEY,
    memory_id INTEGER NOT NULL REFERENCES memories(id),
    seq       INTEGER NOT NULL,
    text      TEXT    NOT NULL,
    embedding BLOB    NOT NULL     -- float32, L2-normalised
);
CREATE INDEX IF NOT EXISTS idx_chunks_memory ON chunks(memory_id);

-- Dated, self-contained facts written by the agent alongside a memory.
CREATE TABLE IF NOT EXISTS facts (
    id         INTEGER PRIMARY KEY,
    memory_id  INTEGER NOT NULL REFERENCES memories(id),
    text       TEXT    NOT NULL,
    created_at TEXT    NOT NULL,
    embedding  BLOB    NOT NULL     -- float32, L2-normalised
);
CREATE INDEX IF NOT EXISTS idx_facts_memory ON facts(memory_id);

-- Store-level settings that must not change after the first write (the embedding model).
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
