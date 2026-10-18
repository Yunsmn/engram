"""BEAM scoring, following the benchmark's official evaluation: for every rubric item of a question, an LLM judge
(gpt-4.1-mini, BEAM's default) scores the response 1.0 / 0.5 / 0.0 with BEAM's unified judge prompt
(eval/beam/judge_prompt.txt, copied verbatim from github.com/mohammadtavakoli78/BEAM src/prompts.py, MIT License);
the question's score is the mean over its rubric items. Deviation, stated: for event ordering BEAM also adds a
sequence-alignment score; only the rubric judge is used here, for every ability.

The response comes from the same reader and prompt as every other harness run (eval.qa.answer) given
the system's recalled context. Answer and judge: READER (gpt-4.1-mini). Resumable.
Usage: python -m eval.beam.judge --run-id beam-test --systems engram-v5 engram-v531 mem0 flat_rag
"""

from __future__ import annotations

import argparse
import json
import random
import re
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from engram.cache import Cache
from engram.config import Config
from eval.llm import make_remote_llm
from eval.common import read_jsonl, write_jsonl
from eval import qa
from eval.data import QUESTIONS_DIR, load_question

PROMPT = (Path(__file__).parent / "judge_prompt.txt").read_text()
RESULTS = Path("results")


def parse_score(text: str) -> float:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    match = re.search(r"\{.*\}", text, re.S)
    try:
        value = json.loads(match.group(0))["score"] if match else None
    except (json.JSONDecodeError, KeyError):  # e.g. an invalid backslash escape inside "reason"
        found = re.search(r'"score"\s*:\s*"?([0-9.]+)', text)
        value = found.group(1) if found else None
    score = float(value)
    if score not in (0.0, 0.5, 1.0):
        raise ValueError(f"unexpected score {value!r}")
    return score


def rubric_of(qid: str) -> list[str]:
    return json.loads((QUESTIONS_DIR / f"{qid}.json").read_text())["rubric"]


def score_one(llm, system: str, record: dict) -> dict:
    q = load_question(record["question_id"])
    response = qa.answer(llm, q.question, q.question_date, record["context"])
    items = []
    for item in rubric_of(q.question_id):
        prompt = PROMPT.replace("<rubric_item>", item).replace("<llm_response>", response)
        items.append(parse_score(llm.complete(prompt, max_tokens=400).text))
    return {"kind": "beam", "question_id": q.question_id, "question_type": q.question_type, "system": system,
            "response": response, "item_scores": items, "score": sum(items) / len(items),
            "tokens": record.get("tokens")}


def report(rows: list[dict], systems: list[str]) -> None:
    by = defaultdict(dict)
    for r in rows:
        by[r["question_id"]][r["system"]] = r
    common = [q for q, v in by.items() if all(s in v for s in systems)]
    types = sorted({by[q][systems[0]]["question_type"] for q in common})
    print(f"n = {len(common)} questions scored by every system")
    print("| ability | " + " | ".join(systems) + " |")
    for t in types + ["ALL"]:
        sel = [q for q in common if t == "ALL" or by[q][systems[0]]["question_type"] == t]
        cells = [f"{100 * sum(by[q][s]['score'] for q in sel) / len(sel):.1f}" for s in systems]
        print(f"| {t} ({len(sel)}) | " + " | ".join(cells) + " |")
    ref = systems[-1]
    for s in systems[:-1]:
        d = [by[q][s]["score"] - by[q][ref]["score"] for q in common]
        rng = random.Random(0)
        boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(4000))
        print(f"{s} - {ref}: {100 * sum(d) / len(d):+.1f} [{100 * boots[100]:+.1f}, {100 * boots[3899]:+.1f}] "
              "(paired bootstrap 95% CI, score points)")


def rejudge(run_id: str, systems: list[str], role: str, workers: int) -> None:
    """R2: re-score existing responses (same answers) with another judge; writes beam-qa-<role>.jsonl."""
    src = RESULTS / run_id / "beam-qa.jsonl"
    out = RESULTS / run_id / f"beam-qa-{role.lower()}.jsonl"
    done = {(r["question_id"], r["system"]) for r in read_jsonl(out)} if out.exists() else set()
    judge = make_remote_llm(role, Cache(Config.from_env().cache_path))
    lock = threading.Lock()
    rows = [r for r in read_jsonl(src) if r["system"] in systems and (r["question_id"], r["system"]) not in done]

    def job(r):
        try:
            items = [parse_score(judge.complete(PROMPT.replace("<rubric_item>", it).replace("<llm_response>", r["response"]),
                                                max_tokens=400).text) for it in rubric_of(r["question_id"])]
        except Exception as exc:  # noqa: BLE001
            print(f"unscored {r['system']} {r['question_id']}: {exc!r}"[:160], flush=True)
            return
        with lock:
            write_jsonl(out, [{**r, "item_scores": items, "score": sum(items) / len(items), "judge": role}])

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(job, rows))
    report(list(read_jsonl(out)), systems)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--systems", nargs="+", required=True, help="recall files <system>.jsonl; last = reference")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--rejudge-role", help="R2: re-score existing responses with this env role (e.g. OFFICIAL_JUDGE)")
    args = parser.parse_args()
    if args.rejudge_role:
        rejudge(args.run_id, args.systems, args.rejudge_role, args.workers)
        return
    out = RESULTS / args.run_id / "beam-qa.jsonl"
    done = {(r["question_id"], r["system"]) for r in read_jsonl(out)} if out.exists() else set()
    if not args.report_only:
        llm = make_remote_llm("READER", Cache(Config.from_env().cache_path))
        lock = threading.Lock()
        jobs = [(s, r) for s in args.systems for r in read_jsonl(RESULTS / args.run_id / f"{s}.jsonl")
                if r.get("kind", "recall") == "recall" and (r["question_id"], s) not in done]

        def job(item):
            s, r = item
            try:
                row = score_one(llm, s, r)
            except Exception as exc:  # noqa: BLE001 — content filter etc.: left unscored, reported
                print(f"unscored {s} {r['question_id']}: {exc!r}"[:160], flush=True)
                return
            with lock:
                write_jsonl(out, [row])

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(job, jobs))
    report(list(read_jsonl(out)), args.systems)


if __name__ == "__main__":
    main()
