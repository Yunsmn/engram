"""OpenAI-compatible LLM client for the evaluation (answering model, judge, the agent's model). The engine itself
never calls an LLM. Each role reads <ROLE>_BASE_URL, <ROLE>_API_KEY, <ROLE>_MODEL from the environment or .env."""

from __future__ import annotations

from pathlib import Path

from engram.cache import Cache
from engram.metrics import Metrics
from eval.llm.base import CachedLLM


def load_env() -> None:
    from dotenv import load_dotenv

    here = Path.cwd()
    for candidate in [here / ".env", *(p / ".env" for p in here.parents if (p / ".git").is_dir())]:
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            return


def make_remote_llm(role: str, cache: Cache | None, metrics: Metrics | None = None) -> CachedLLM:
    from eval.llm.openai_compat import OpenAICompatLLM

    load_env()
    return CachedLLM(OpenAICompatLLM.from_env(role), cache, metrics, role=role.lower())
