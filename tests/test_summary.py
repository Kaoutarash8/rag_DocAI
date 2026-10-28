import json

import pytest

from app.core.config import get_settings
from app.services import summary


def _segments(n=8):
    return [{"page": i, "section": "1 Intro" if i <= 3 else "2 Méthode > 2.1", "kind": "text",
             "text": " ".join(["mot"] * 150)} for i in range(1, n + 1)]


@pytest.fixture()
def small_blocks(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "summary_block_words", 200)
    monkeypatch.setattr(s, "summary_max_llm_calls", 10)


def test_hierarchical_summary_follows_sections(small_blocks):
    calls = []

    def fake(system, user):
        calls.append(user)
        return f"R{len(calls)}"
    r = summary.summarize_segments(_segments(), "Doc", fake)
    assert [(p["title"], p["pages"]) for p in r["parts"]] == [("1 Intro", "1-3"), ("2 Méthode", "4-8")]
    assert r["llm_calls"] == len(calls) and not r["truncated"]
    assert "Fusionne" in calls[-1]  # le dernier appel fusionne les résumés de sections
    assert not any("Intro" in c and "Méthode" in c for c in calls[:-1])  # un bloc ne mélange pas 2 sections


def test_call_budget_is_respected_and_truncation_flagged(small_blocks):
    calls = []
    r = summary.summarize_segments(_segments(120), "Doc", lambda s, u: calls.append(1) or "ok")
    assert r["truncated"] is True and len(calls) <= 10


def test_summary_is_cached(client, small_blocks):
    doc_id = 987
    with summary.segments_path(doc_id).open("w", encoding="utf-8") as f:
        f.write("\n".join(json.dumps(s) for s in _segments(3)))
    n = []
    import app.services.llm as llm
    orig, llm.complete = llm.complete, lambda s, u: n.append(1) or "résumé"
    try:
        a = summary.summarize_document(doc_id, "Doc")
        b = summary.summarize_document(doc_id, "Doc")
    finally:
        llm.complete = orig
        summary.cleanup(doc_id)
    assert a == b and len(n) == a["llm_calls"]
