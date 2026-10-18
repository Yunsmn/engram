"""Per-operation counters (calls, tokens, latency) and a JSONL event log.

Counters are the one deliberately mutable object in the engine: they are an accumulator,
owned by an Engine/baseline instance, and read out as an immutable snapshot.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


class Metrics:
    def __init__(self, log_path: Path | None = None) -> None:
        self.counts: Counter[str] = Counter()
        self.seconds: Counter[str] = Counter()
        self._log_path = log_path
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)

    def incr(self, key: str, amount: float = 1) -> None:
        self.counts[key] += amount

    @contextmanager
    def timed(self, key: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[key] += time.perf_counter() - start
            self.counts[f"{key}.calls"] += 1

    def log(self, event: str, **fields: Any) -> None:
        if self._log_path is None:
            return
        record = {"ts": time.time(), "event": event, **fields}
        with self._log_path.open("a") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def snapshot(self) -> dict[str, float]:
        out = {k: float(v) for k, v in self.counts.items()}
        out.update({f"{k}.seconds": round(v, 4) for k, v in self.seconds.items()})
        return out

    def delta(self, before: dict[str, float]) -> dict[str, float]:
        now = self.snapshot()
        return {k: round(v - before.get(k, 0.0), 4) for k, v in now.items() if v != before.get(k, 0.0)}
