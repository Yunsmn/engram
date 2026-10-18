from engram.cache import Cache
from eval.agent import FACTS_MAX_TOKENS, FACTS_PROMPT, FACTS_SYSTEM, Agent
from eval.beam.judge import parse_score


class FakeLLM:
    name = "host/model"

    def complete(self, *a, **k):
        raise AssertionError("offline mode must not call the model")


def test_offline_agent_reads_the_cache_and_never_calls_the_model(tmp_path):
    cache = Cache(tmp_path / "c.sqlite")
    key = {"p": FACTS_PROMPT.format(date="2024-05-01T10:00", text="hi"), "s": FACTS_SYSTEM, "m": FACTS_MAX_TOKENS}
    cache.put("llm:host/model", key, {"text": '{"facts": ["Ann moved on 2024-05-01.", " "]}'})
    agent = Agent(FakeLLM(), cache, offline=True)
    assert agent.facts("hi", "2024-05-01T10:00") == ["Ann moved on 2024-05-01."]
    assert agent.facts("not cached", "2024-05-01T10:00") == []
    assert agent.summary("user: hello", "2024-05-01T10:00") == "user: hello" and agent.fallbacks == 1


def test_judge_score_parsing_tolerates_bad_escapes():
    assert parse_score('{"score": 0.5, "reason": "ok"}') == 0.5
    assert parse_score('```json\n{"score": 1.0, "reason": "costs \\$5 in C:\\path"}\n```') == 1.0
