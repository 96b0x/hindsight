"""Tests for Hindsight's current-turn memory context compiler."""

import json
import sys
from types import SimpleNamespace

from hindsight_hermes.compiler import (
    MemoryCandidate,
    build_recall_query,
    compile_memory_context,
    render_ranked_memory_context,
)


def _candidate(candidate_id: str, text: str, *, document_id: str = "") -> MemoryCandidate:
    return MemoryCandidate(
        candidate_id=candidate_id,
        memory_id=f"memory-{candidate_id}",
        text=text,
        memory_type="observation",
        context="canonical memory corpus",
        occurred_start="2026-08-29T00:00:00Z",
        mentioned_at="2026-08-30T00:00:00Z",
        document_id=document_id,
        score="0.97",
        metadata={"source": "hermes-memory"},
    )


def test_build_recall_query_keeps_current_request_first_and_adds_recent_dialogue():
    query = build_recall_query(
        "Why is it designed that way?",
        [
            {"role": "user", "content": "Explain the Hindsight recall pipeline."},
            {
                "role": "assistant",
                "content": "It currently prefetches after each turn.",
            },
            {"role": "tool", "content": "must not be included"},
        ],
        max_chars=1000,
    )

    assert query.startswith("CURRENT USER REQUEST:\nWhy is it designed that way?")
    assert "RECENT DIALOGUE" in query
    assert "Explain the Hindsight recall pipeline." in query
    assert "It currently prefetches after each turn." in query
    assert "must not be included" not in query


def test_build_recall_query_truncation_preserves_current_request():
    current = "CURRENT-NEEDLE " + ("x" * 120)
    query = build_recall_query(
        current,
        [{"role": "assistant", "content": "old " * 200}],
        max_chars=180,
    )

    assert "CURRENT-NEEDLE" in query
    assert len(query) <= 180


def test_compile_selects_valid_ids_and_renders_exact_evidence(monkeypatch):
    candidates = [
        _candidate("M0001", "The old design used next-turn prefetch."),
        _candidate(
            "M0002",
            "The current decision is synchronous current-turn recall.",
            document_id="hermes-memory:topics/openclaw-memory-systems.md",
        ),
    ]
    payload = {
        "current_state_ids": ["M0002", "MADE_UP"],
        "relevant_context_ids": ["M0001"],
        "prior_decision_ids": [],
        "conflicts": [
            {
                "ids": ["M0001", "M0002", "UNKNOWN"],
                "note": "The newer decision supersedes the lagged design.",
            }
        ],
        "deep_source_ids": ["M0002"],
    }

    def fake_call_llm(*, task, messages, temperature, max_tokens):
        kwargs = dict(task=task, messages=messages, temperature=temperature, max_tokens=max_tokens)
        assert kwargs["task"] == "memory_context"
        assert "Why is memory stale?" in kwargs["messages"][1]["content"]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))])

    monkeypatch.setattr(sys.modules["agent.memory_provider"], "call_auxiliary_task", fake_call_llm, raising=False)

    compiled = compile_memory_context(
        current_query="Why is memory stale?",
        recent_messages=[{"role": "assistant", "content": "We found one-turn lag."}],
        candidates=candidates,
    )

    assert compiled is not None
    assert compiled.count == 2
    assert "## Current state" in compiled.text
    assert "## Relevant memory" in compiled.text
    assert "## Conflicts and staleness" in compiled.text
    assert "The current decision is synchronous current-turn recall." in compiled.text
    assert "The old design used next-turn prefetch." in compiled.text
    assert "MADE_UP" not in compiled.text
    assert "UNKNOWN" not in compiled.text
    assert "hermes-memory:topics/openclaw-memory-systems.md" in compiled.text


def test_compile_returns_none_on_invalid_json(monkeypatch):
    monkeypatch.setattr(
        sys.modules["agent.memory_provider"],
        "call_auxiliary_task",
        lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="not-json"))]),
        raising=False,
    )

    compiled = compile_memory_context(
        current_query="query",
        recent_messages=[],
        candidates=[_candidate("M0001", "evidence")],
    )

    assert compiled is None


def test_ranked_fallback_renders_exact_provenance():
    compiled = render_ranked_memory_context([_candidate("M0001", "exact evidence", document_id="topics/memory.md")])

    assert compiled is not None
    assert "exact evidence" in compiled.text
    assert "memory_id=memory-M0001" in compiled.text
    assert "observation" in compiled.text
    assert "occurred=2026-08-29T00:00:00Z" in compiled.text
    assert "mentioned=2026-08-30T00:00:00Z" in compiled.text
    assert "document=topics/memory.md" in compiled.text
    assert "score=0.97" in compiled.text
    assert "source=hermes-memory" in compiled.text


def test_compile_caps_selected_evidence_to_24(monkeypatch):
    candidates = [_candidate(f"M{index:04d}", f"evidence {index}") for index in range(1, 33)]
    payload = {
        "current_state_ids": [candidate.candidate_id for candidate in candidates],
        "relevant_context_ids": [],
        "prior_decision_ids": [],
        "conflicts": [],
        "deep_source_ids": [],
    }
    monkeypatch.setattr(
        sys.modules["agent.memory_provider"],
        "call_auxiliary_task",
        lambda **kwargs: SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
        ),
        raising=False,
    )

    compiled = compile_memory_context(current_query="query", recent_messages=[], candidates=candidates)

    assert compiled is not None
    assert compiled.count == 24
    assert len(compiled.selected_ids) == 24


def test_conflict_and_deep_source_ids_must_be_primary_selections(monkeypatch):
    candidates = [
        _candidate("M0001", "selected evidence", document_id="topics/selected.md"),
        _candidate("M0002", "unselected evidence", document_id="topics/unselected.md"),
    ]
    payload = {
        "current_state_ids": ["M0001"],
        "relevant_context_ids": [],
        "prior_decision_ids": [],
        "conflicts": [{"ids": ["M0001", "M0002"], "note": "must be dropped"}],
        "deep_source_ids": ["M0002", "M0001"],
    }
    monkeypatch.setattr(
        sys.modules["agent.memory_provider"],
        "call_auxiliary_task",
        lambda **kwargs: SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
        ),
        raising=False,
    )

    compiled = compile_memory_context(current_query="query", recent_messages=[], candidates=candidates)

    assert compiled is not None
    assert compiled.selected_ids == ("M0001",)
    assert "must be dropped" not in compiled.text
    assert "topics/selected.md" in compiled.text
    assert "topics/unselected.md" not in compiled.text


def test_tiny_retrieval_budget_keeps_intent_instead_of_only_a_heading():
    assert build_recall_query("current intent", [], max_chars=7) == "current"
