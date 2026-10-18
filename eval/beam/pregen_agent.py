"""Write the agent's summary and dated facts for every BEAM write unit once, into the cache (ENGRAM_CACHE_PATH),
so that recall runs (`python -m eval.run --offline`) read them without calling the API.
Usage: python -m eval.beam.pregen_agent   (needs READER_* for the agent's model)
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

from engram.cache import Cache
from engram.config import Config
from eval.agent import Agent
from eval.data import INDEX, iso, load_question
from eval.llm import make_remote_llm

WORKERS = 8


def main() -> None:
    cache = Cache(Config.from_env().cache_path)
    agent = Agent(make_remote_llm("READER", cache), cache)
    units = {}
    for r in json.loads(INDEX.read_text()):
        for s in load_question(r["question_id"]).sessions:
            units.setdefault((s.as_text(), iso(s.date)), s.session_id)
    print(f"{len(units)} write units", flush=True)

    def job(key):
        text, date = key
        agent.summary(text, date)
        return len(agent.facts(text, date))

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        counts = list(pool.map(job, units))
    print(f"done: {sum(counts)} facts, {agent.fallbacks} summary fallbacks", flush=True)


if __name__ == "__main__":
    main()
