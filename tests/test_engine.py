import pytest

from engram.engine import Engram
from engram.read.markers import mark, strip_marked
from engram.store.db import Store
from engram.store.embed import HashEmbedder


def engine(path=":memory:") -> Engram:
    return Engram(Store(path), HashEmbedder())


def test_remember_stores_original_summary_chunks_and_facts():
    e = engine()
    r = e.remember("user: I adopted a grey cat named Miso.", summary="The user adopted Miso, a grey cat.",
                   timestamp="2024-03-02T10:00", facts=["The user adopted a grey cat named Miso in March 2024."])
    assert r.action == "new"
    assert e.store.full_text(r.memory_id) == "user: I adopted a grey cat named Miso."
    assert e.store.get_memory(r.memory_id).summary == "The user adopted Miso, a grey cat."
    assert e.store.chunks_for([r.memory_id]) and len(e.store.facts_for([r.memory_id])) == 1


def test_summary_is_required_and_exact_duplicates_are_dropped():
    e = engine()
    with pytest.raises(ValueError):
        e.remember("user: hello", summary="")
    first = e.remember("user: hello there", summary="Greeting.")
    again = e.remember("user: hello there", summary="Greeting.")
    assert again.action == "exact duplicate" and again.memory_id == first.memory_id
    assert e.store.count_memories() == 1


def test_recalled_text_is_never_stored_again():
    e = engine()
    e.remember("user: My cat Miso is grey.", summary="The user's cat Miso is grey.", timestamp="2024-01-01T00:00")
    recalled = e.recall("cat")
    r = e.remember(f"assistant: {recalled.marked}\n\nuser: I also bought a bike.", summary="The user bought a bike.",
                   timestamp="2024-02-01T00:00")
    assert "Miso" not in e.store.full_text(r.memory_id)
    assert e.remember(mark("Earlier: the user has a cat."), summary="x").action == "nothing new"


def test_text_without_markers_is_returned_byte_for_byte():
    text = "user: hello   \n\n\n\nassistant: hi\n"
    assert strip_marked(text) == (text, 0)
    assert strip_marked("user: a <memory> own tail") == ("user: a  own tail", 1)


def test_time_questions_get_dated_facts_first_and_counting_questions_the_full_layout():
    e = engine()
    e.remember("user: I moved to Berlin.", summary="The user moved to Berlin.", timestamp="2024-05-01T10:00",
               facts=["The user moved to Berlin in May 2024."])
    when = e.recall("When did I move to Berlin?")
    assert when.mode == "facts+hybrid" and when.text.startswith("FACTS:\n[2024-05-01T10:00] The user moved")
    many = e.recall("How many cities have I lived in?")
    assert many.mode == "full" and "user: I moved to Berlin." in many.text
    assert e.recall("Where do I live?").mode == "hybrid"


def test_recall_respects_the_token_budget():
    e = engine()
    for i in range(30):
        e.remember(f"user: note number {i} " + "about the garden " * 40, summary=f"Garden note {i}.",
                   timestamp=f"2024-01-{i % 28 + 1:02d}T10:00")
    assert e.recall("garden", token_budget=300).tokens <= 300


def test_forget_restore_and_purge():
    e = engine()
    a = e.remember("user: I live in Paris.", summary="The user lives in Paris.").memory_id
    b = e.remember("user: I like jazz.", summary="The user likes jazz.").memory_id
    assert e.forget([a]) == 1 and "Paris" not in e.recall("where do I live").text
    assert e.restore([a]) == 1 and "Paris" in e.recall("where do I live").text
    assert e.purge([a]) == 1 and e.store.get_memory(a) is None and e.store.all_memory_ids() == [b]


def test_purge_leaves_no_trace_in_the_database_files(tmp_path):
    e = engine(tmp_path / "m.sqlite")
    secret = e.remember("user: My passport number is ZX9912345.", summary="Passport ZX9912345.",
                        facts=["The user's passport number is ZX9912345."]).memory_id
    e.remember("user: I bought a road bike.", summary="The user bought a road bike.")
    e.purge([secret])
    e.store.close()
    for path in tmp_path.iterdir():
        assert b"ZX9912345" not in path.read_bytes(), path.name


def test_a_store_refuses_a_different_embedder(tmp_path):
    db = tmp_path / "m.sqlite"
    Engram(Store(db), HashEmbedder()).store.close()

    class Other(HashEmbedder):
        model_name = "other-model"

    with pytest.raises(ValueError, match="HashEmbedder"):
        Engram(Store(db), Other())
