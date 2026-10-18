"""Recall stage of the evaluation: every system ingests each conversation's history, then answers nothing yet --
it only returns its context for every question. Records go to results/<run-id>/<system>.jsonl.

engram: the agent (READER model, or the cache with --offline) writes each session's summary and dated facts.
flat_rag: plain chunk retrieval with all-MiniLM-L6-v2, as in the reported runs.
Usage: python -m eval.run --run-id beam-test --systems engram flat_rag [--offline] [--conversations beam1 beam2]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from collections import defaultdict
from pathlib import Path

from engram.cache import Cache
from engram.config import Config
from engram.engine import Engram
from engram.read.budget import count_tokens
from engram.store.db import Store
from engram.store.embed import SentenceTransformerEmbedder
from eval.agent import Agent
from eval.baselines.flat_rag import FlatRAG
from eval.common import read_jsonl, write_jsonl
from eval.data import INDEX, iso, load_question
from eval.llm import make_remote_llm

RESULTS = Path("results")
FLAT_EMBEDDER = {"embed_model": "sentence-transformers/all-MiniLM-L6-v2", "embed_query_prefix": "",
                 "embed_passage_prefix": ""}


def conversations(selected: list[str]) -> dict[str, list[str]]:
    by_conv: dict[str, list[str]] = defaultdict(list)
    for r in json.loads(INDEX.read_text()):
        conv = r["question_id"].split("_")[0]
        if not selected or conv in selected:
            by_conv[conv].append(r["question_id"])
    return by_conv


def build(system: str, config: Config, cache: Cache, agent: Agent | None):
    if system == "flat_rag":
        flat = dataclasses.replace(config, **FLAT_EMBEDDER)
        return FlatRAG(SentenceTransformerEmbedder.from_config(flat, cache), flat)
    # one store per conversation; access counts off, so every question sees the same store as on a fresh build
    engine = Engram(Store(":memory:"), SentenceTransformerEmbedder.from_config(config, cache),
                    dataclasses.replace(config, track_access=0))

    class EngramWithAgent:
        def remember(self, text: str, source: str, timestamp: str) -> None:
            seen = engine.clean_incoming(text)
            if not seen.strip():
                return
            engine.remember(text, summary=agent.summary(seen, timestamp), source=source, timestamp=timestamp,
                            facts=agent.facts(seen, timestamp))

        def recall(self, query: str, token_budget: int):
            return engine.recall(query, token_budget=token_budget)

    return EngramWithAgent()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--systems", nargs="+", default=["engram", "flat_rag"], choices=["engram", "flat_rag"])
    parser.add_argument("--conversations", nargs="*", default=[])
    parser.add_argument("--budget", type=int, default=1300)
    parser.add_argument("--offline", action="store_true", help="agent summaries/facts from the cache only")
    args = parser.parse_args()
    config = Config.from_env()
    cache = Cache(config.cache_path)
    agent = Agent(make_remote_llm("READER", cache), cache, offline=args.offline) if "engram" in args.systems else None
    out = RESULTS / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    for system in args.systems:
        path = out / f"{system}.jsonl"
        done = {r["question_id"] for r in read_jsonl(path)} if path.exists() else set()
        for conv, qids in conversations(args.conversations).items():
            if set(qids) <= done:
                continue
            start = time.perf_counter()
            memory = build(system, config, cache, agent)
            sessions = sorted(load_question(qids[0]).sessions, key=lambda s: s.date)
            for s in sessions:
                memory.remember(s.as_text(), source=s.session_id, timestamp=iso(s.date))
            ingest_s = time.perf_counter() - start
            rows = []
            for qid in qids:
                q = load_question(qid)
                res = memory.recall(q.question, args.budget)
                text = res if isinstance(res, str) else res.text
                rows.append({"kind": "recall", "question_id": qid, "question_type": q.question_type,
                             "is_abstention": q.is_abstention, "system": system, "context": text,
                             "tokens": count_tokens(text), "recall_mode": getattr(res, "mode", system),
                             "n_sessions": len(sessions), "ingest_seconds": round(ingest_s, 1)})
            write_jsonl(path, rows)
            print(f"[{system}] {conv}: {len(sessions)} sessions, {len(qids)} questions, {ingest_s:.0f}s", flush=True)
    if agent and agent.fallbacks:
        print(f"agent summary fallbacks (provider refused or not cached): {agent.fallbacks}")


if __name__ == "__main__":
    main()
