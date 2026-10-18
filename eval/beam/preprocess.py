"""Convert BEAM (Mohammadta/BEAM, ICLR 2026; CC BY-SA 4.0), split "100K" (the 128K chats: 20 conversations,
400 probing questions, 10 memory abilities) into the harness's question-file format.

BEAM chats have 3 sessions of ~50K tokens each. Every system receives the same write units: consecutive groups of
UNIT_MESSAGES messages (2 user/assistant exchanges, ~3K tokens), dated with their session's time anchor (minutes
increase within a session to keep order). Generation artefacts (" ->-> 1,1" suffixes on user messages) are removed.
Each question file keeps BEAM's rubric (the items a correct answer must satisfy) for the official-style judge.
Input: data/beam/raw/rows_*.json (rows API of the Hugging Face datasets server).
Usage: python -m eval.beam.download && python -m eval.beam.preprocess  ->  data/beam/questions/*.json, data/beam/index.json
       python -m eval.beam.preprocess --raw data/beam500/raw --out data/beam500 --prefix b500c --expected 10  (dev)
"""

from __future__ import annotations

import ast
import json
import re
import argparse
from datetime import datetime, timedelta
from pathlib import Path

RAW = Path("data/beam/raw")
OUT = Path("data/beam")
UNIT_MESSAGES = 4
ARTEFACT = re.compile(r"\s*->->\s*\d+\s*,\s*\d+\s*$")
FMT = "%Y/%m/%d (%a) %H:%M"


def _anchor(text: str) -> datetime:
    return datetime.strptime(text.strip(), "%B-%d-%Y").replace(hour=9)


def convert(row: dict, prefix: str = "beam") -> list[dict]:
    cid = f"{prefix}{row['conversation_id']}"
    sessions = []
    for b, batch in enumerate(row["chat"], 1):
        start = _anchor(batch[0]["time_anchor"])
        for u in range(0, len(batch), UNIT_MESSAGES):
            msgs = batch[u:u + UNIT_MESSAGES]
            turns = [{"role": m["role"], "content": ARTEFACT.sub("", m["content"]), "has_answer": False} for m in msgs]
            sessions.append({"session_id": f"{cid}_s{b}_{u // UNIT_MESSAGES:02d}",
                             "date": (start + timedelta(minutes=u // UNIT_MESSAGES)).strftime(FMT), "turns": turns})
    last = max(datetime.strptime(s["date"], FMT) for s in sessions)
    question_date = last.replace(hour=23, minute=59).strftime(FMT)
    out = []
    for ability, items in ast.literal_eval(row["probing_questions"]).items():
        for i, q in enumerate(items):
            gold = q.get("ideal_response") or q.get("answer") or ""
            out.append({"question_id": f"{cid}_{ability}_{i}", "question_type": ability,
                        "is_abstention": ability == "abstention", "question": q["question"],
                        "question_date": question_date, "answer": gold if isinstance(gold, str) else json.dumps(gold),
                        "answer_session_ids": [], "rubric": list(q["rubric"]), "sessions": sessions})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=str(RAW))
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--prefix", default="beam")
    parser.add_argument("--expected", type=int, default=20)
    args = parser.parse_args()
    raw, out_dir = Path(args.raw), Path(args.out)
    rows = [r["row"] for f in sorted(raw.glob("rows_*.json")) for r in json.loads(f.read_text())["rows"]]
    assert len({r["conversation_id"] for r in rows}) == args.expected, f"expected {args.expected} conversations"
    qdir = out_dir / "questions"
    qdir.mkdir(parents=True, exist_ok=True)
    index = []
    for row in sorted(rows, key=lambda r: int(r["conversation_id"])):
        for q in convert(row, args.prefix):
            (qdir / f"{q['question_id']}.json").write_text(json.dumps(q, ensure_ascii=False))
            index.append({k: q[k] for k in ("question_id", "question_type", "is_abstention")}
                         | {"n_sessions": len(q["sessions"])})
    (out_dir / "index.json").write_text(json.dumps(index, indent=0))
    print(f"wrote {len(index)} questions; write units per conversation: "
          f"{sorted({r['n_sessions'] for r in index})}")


if __name__ == "__main__":
    main()
