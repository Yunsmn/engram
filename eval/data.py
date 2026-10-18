"""Benchmark data in one format: a question file holds the question, its answer, and the dated conversation
sessions the memory system ingests first. EVAL_DATA_DIR selects the dataset (default data/beam)."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(os.environ.get("EVAL_DATA_DIR", "data/beam"))
QUESTIONS_DIR = DATA_DIR / "questions"
INDEX = DATA_DIR / "index.json"


@dataclass(frozen=True)
class Turn:
    role: str
    content: str
    has_answer: bool


@dataclass(frozen=True)
class Session:
    session_id: str
    date: str
    turns: tuple[Turn, ...]

    def as_text(self) -> str:
        return "\n\n".join(f"{t.role}: {t.content}" for t in self.turns)


@dataclass(frozen=True)
class Question:
    question_id: str
    question_type: str
    is_abstention: bool
    question: str
    question_date: str
    answer: str
    answer_session_ids: tuple[str, ...]
    sessions: tuple[Session, ...]


def load_question(qid: str, root: Path = QUESTIONS_DIR) -> Question:
    raw = json.loads((root / f"{qid}.json").read_text())
    sessions = tuple(
        Session(s["session_id"], s["date"], tuple(Turn(**t) for t in s["turns"])) for s in raw["sessions"]
    )
    return Question(raw["question_id"], raw["question_type"], raw["is_abstention"], raw["question"],
                    raw["question_date"], raw["answer"], tuple(raw["answer_session_ids"]), sessions)


def iso(date: str) -> str:
    """'2023/05/20 (Sat) 02:21' -> '2023-05-20T02:21'."""
    cleaned = re.sub(r"\s*\(\w+\)\s*", " ", date).strip()
    return datetime.strptime(cleaned, "%Y/%m/%d %H:%M").strftime("%Y-%m-%dT%H:%M")
