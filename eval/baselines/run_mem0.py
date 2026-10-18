"""Mem0 baseline: ingest each conversation into its own Mem0 store, then recall every question of it. Records go to
results/<run-prefix>/mem0.jsonl in the same format as eval.run. Mem0's LLM is READER_* (as in the reported runs,
gpt-4.1-mini). MEM0_TOP_K (default 30; the reported BEAM run used 100) memories are retrieved, then cut to the
token budget. One process per conversation is safest (each Mem0 store locks a process-wide Qdrant store).
Usage: pip install mem0ai==2.2.1
       python -m eval.baselines.run_mem0 --conversations beam1 --run-prefix beam-test   (repeat per conversation)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

from engram.read.budget import count_tokens
from eval.baselines.mem0_system import Mem0System
from eval.common import read_jsonl, write_jsonl
from eval.data import INDEX, iso, load_question


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversations", nargs="*", default=[])
    parser.add_argument("--questions", nargs="*", default=[],
                        help="LongMemEval-style: each question has its own history, one store per question")
    parser.add_argument("--budgets", nargs="+", type=int, default=[1300])
    parser.add_argument("--run-prefix", default="locomo-test")
    args = parser.parse_args()
    by_conv: dict[str, list[str]] = defaultdict(list)
    for r in json.loads(INDEX.read_text()):
        conv = r["question_id"].split("_")[0]
        if conv in args.conversations:
            by_conv[conv].append(r["question_id"])
    for qid in args.questions:  # one "conversation" per question
        by_conv[qid] = [qid]
    for conv, qids in by_conv.items():
        outs = {b: Path("results") / (args.run_prefix + ("" if b == 1300 else f"-{b // 1000}k")) / "mem0.jsonl"
                for b in args.budgets}
        done = {b: {r["question_id"] for r in read_jsonl(p)} if p.exists() else set() for b, p in outs.items()}
        if all(set(qids) <= done[b] for b in args.budgets):
            continue
        system = Mem0System(user_id=conv.replace("_", "-"), base_url=os.environ["READER_BASE_URL"], api_key=os.environ["READER_API_KEY"],
                            model=os.environ["READER_MODEL"])
        sessions = sorted(load_question(qids[0]).sessions, key=lambda s: s.date)
        start = time.perf_counter()
        for s in sessions:
            system.remember(s.as_text(), source=s.session_id, timestamp=iso(s.date),
                            turns=[(t.role, t.content) for t in s.turns])
        ingest_s = time.perf_counter() - start
        print(f"[mem0] {conv}: {len(sessions)} sessions ingested in {ingest_s:.0f}s", flush=True)
        for qid in qids:
            q = load_question(qid)
            for b in args.budgets:
                if qid in done[b]:
                    continue
                t = time.perf_counter()
                context = system.recall(q.question, b)
                write_jsonl(outs[b], [{"kind": "recall", "question_id": qid, "question_type": q.question_type,
                                       "is_abstention": q.is_abstention, "system": "mem0", "n_sessions": len(sessions),
                                       "context": context, "tokens": count_tokens(context),
                                       "ingest_seconds": round(ingest_s, 2),
                                       "content_filter_fallbacks": system.content_filter_fallbacks,
                                       "recall_seconds": round(time.perf_counter() - t, 3)}])
        print(f"[mem0] {conv}: {len(qids)} questions recalled", flush=True)


if __name__ == "__main__":
    main()
