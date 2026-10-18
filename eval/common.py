"""Shared helpers for evaluation scripts: run ids and append-only JSONL output."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def new_run_id(kind: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{kind}-{stamp}-{git_rev()}"


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        for record in records:
            fh.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    """Tolerates a truncated final line (process killed mid-write); any other bad line is an error."""
    # split on "\n" only: str.splitlines also breaks on U+2028, U+0085, ... which JSON strings may contain
    lines = [line for line in path.read_text().split("\n") if line.strip()]
    records = []
    for i, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
    return records
