"""Public provider lifecycle regressions; Hermes loader coverage is separate.

The shared upstream conftest simulates Hermes. These tests use the actual plugin
and, for evidence, actual SDK models; no live credentials/services are needed.
"""

import asyncio
import json
import sys
import threading
from types import SimpleNamespace

import pytest
from conftest import FakeClient, FakeRecallResponse
from hindsight_client_api.models.recall_result import RecallResult
from hindsight_client_api.models.recall_scores import RecallScores

import hindsight_hermes as plugin
from hindsight_hermes.compiler import build_recall_query


def install_aux(monkeypatch, callback):
    monkeypatch.setattr(sys.modules["agent.memory_provider"], "call_auxiliary_task", callback, raising=False)


def completion(payload):
    content = payload if isinstance(payload, str) else json.dumps(payload)
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_current_request_dialogue_and_default_sync_do_not_queue(provider, monkeypatch):
    calls = []

    def auxiliary(*, task, messages, temperature, max_tokens):
        calls.append(json.loads(messages[1]["content"]))
        assert task == "memory_context"
        assert temperature == 0 and max_tokens == 1200
        return completion({"current_state_ids": ["M0001"]})

    install_aux(monkeypatch, auxiliary)
    instance, fake = provider({"context_compiler": True}, client=FakeClient(recall_texts=["exact evidence"]))
    history = [
        {"role": "system", "content": "discard system"},
        {
            "role": " User ",
            "content": [{"type": "text", "text": "earlier request"}, {"type": "image_url", "image_url": "opaque"}],
        },
        {"role": "assistant", "content": [{"type": "output_text", "text": "earlier answer"}]},
        {"role": "tool", "content": "discard tool"},
    ]
    snapshot = json.dumps(history)
    instance.on_turn_start(1, "  explain this decision  ")
    block = instance.prefetch("  explain this decision  ", session_id="session-1", messages=history)
    instance.queue_prefetch("previous turn")
    assert len(fake.recalls) == 1 and instance._prefetch_thread is None
    query = fake.recalls[0]["query"]
    assert query.startswith("CURRENT USER REQUEST:\nexplain this decision")
    assert "USER: earlier request" in query and "ASSISTANT: earlier answer" in query
    assert "opaque" not in query and "discard" not in query
    assert calls[0]["current_user_request"] == "explain this decision"
    assert calls[0]["recent_dialogue"] == [
        {"role": "user", "content": "earlier request"},
        {"role": "assistant", "content": "earlier answer"},
    ]
    assert json.dumps(history) == snapshot
    assert "exact evidence" in block and instance.recall_status().count == 1
    instance.shutdown()


@pytest.mark.parametrize(
    "query", [None, "", " \n\t", [], [{"type": "image_url", "image_url": "opaque"}], {"text": "opaque"}]
)
@pytest.mark.parametrize("sync", [True, False])
def test_no_usable_content_never_retrieves_compiles_or_joins(provider, monkeypatch, query, sync):
    instance, fake = provider({"recall_sync": sync, "context_compiler": True})
    install_aux(monkeypatch, lambda **kw: pytest.fail("compiler must not run"))
    monkeypatch.setattr(instance, "_join_prefetch", lambda *a, **kw: pytest.fail("blank input must not join"))
    instance.on_turn_start(1, "")
    assert instance.prefetch(query, messages=[{"role": "user", "content": "old useful request"}]) == ""
    instance.queue_prefetch(query)
    assert not fake.recalls and instance.recall_status() is None
    # Undo the join assertion before the normal shutdown lifecycle.
    monkeypatch.undo()
    instance.shutdown()


@pytest.mark.parametrize(
    "response",
    [
        "not-json",
        "null",
        "[]",
        '{"current_state_ids": "M0001"}',
        '{"current_state_ids": [null]}',
        '{"current_state_ids": [{}]}',
        '{"current_state_ids": ["UNKNOWN"]}',
        '{"current_state_ids": ["M0001"], "conflicts": [{"ids": 7}]}',
    ],
)
def test_bad_compiler_output_falls_back_to_original_evidence(provider, monkeypatch, response):
    install_aux(monkeypatch, lambda **kw: completion(response))
    instance, _ = provider({"context_compiler": True}, client=FakeClient(recall_texts=["first fact", "second fact"]))
    instance.on_turn_start(1, "request")
    text = instance.prefetch("request")
    assert "ranked for this request" in text
    assert text.index("first fact") < text.index("second fact")
    assert instance.recall_status().count == 2
    instance.shutdown()


@pytest.mark.parametrize("error", [TimeoutError, RuntimeError, ValueError])
def test_compiler_error_is_ranked_fallback(provider, monkeypatch, error):
    def fail(**kw):
        raise error("private provider detail")

    install_aux(monkeypatch, fail)
    instance, _ = provider({"context_compiler": True}, client=FakeClient(recall_texts=["original"]))
    assert "original" in instance.prefetch("request")
    assert instance.recall_status().count == 1
    instance.shutdown()


def test_no_data_and_retrieval_failure_skip_compiler(provider, monkeypatch, caplog):
    install_aux(monkeypatch, lambda **kw: pytest.fail("no candidates"))
    instance, _ = provider({"context_compiler": True})
    assert instance.prefetch("private query") == ""

    class Broken(FakeClient):
        async def arecall(self, **kw):
            raise TimeoutError("private query and credential must not be logged")

    broken, _ = provider({"context_compiler": True}, client=Broken())
    assert broken.prefetch("private query") == ""
    assert broken.recall_status() is None
    assert "private query" not in caplog.text
    instance.shutdown()
    broken.shutdown()


def test_exact_source_provenance_survives_and_compiler_cannot_add_it(provider, monkeypatch):
    fact = RecallResult(
        id="fact-id",
        text="  EXACT fact\nwith newline.  ",
        type="observation",
        context="source context",
        occurred_start="2026-08-29T03:00:00+03:00",
        occurred_end="2026-08-30T00:00:00Z",
        mentioned_at="2026-08-31T00:00:00Z",
        document_id="document-id",
        chunk_id="chunk-id",
        metadata={"source": "original-source", "source_path": "original/path", "notes": "source-owned note"},
        tags=["original-tag"],
        source_fact_ids=["original-fact-id"],
        scores=RecallScores(final=0.0, reranker=0.9),
    )

    class Client(FakeClient):
        async def arecall(self, **kw):
            return SimpleNamespace(results=[fact, RecallResult(id="second", text="second original")])

    def auxiliary(**kw):
        request = json.loads(kw["messages"][1]["content"])
        assert request["candidates"][0]["text"] == fact.text
        assert request["candidates"][0]["tags"] == fact.tags
        return completion(
            {
                "current_state_ids": ["M0001", "M0002", "forged-id"],
                "conflicts": [{"ids": ["M0001", "M0002", "forged-id"], "note": "invented claim"}],
                "deep_source_ids": ["M0001", "forged-id"],
                "tags": ["invented-tag"],
                "metadata": {"source": "invented-source"},
                "text": "invented fact",
            }
        )

    install_aux(monkeypatch, auxiliary)
    instance, _ = provider({"context_compiler": True}, client=Client())
    text = instance.prefetch("current request")
    for value in [
        fact.text,
        "fact-id",
        "source context",
        fact.occurred_start,
        fact.occurred_end,
        fact.mentioned_at,
        "document-id",
        "chunk-id",
        "original-source",
        "original/path",
        "source-owned note",
        "original-tag",
        "original-fact-id",
        '"final": 0.0',
        '"reranker": 0.9',
    ]:
        assert value in text
    assert "invented" not in text and "forged-id" not in text
    assert "Potential conflict" in text and instance.recall_status().count == 2
    assert fact.text == "  EXACT fact\nwith newline.  "
    instance.shutdown()


@pytest.mark.parametrize("boundary", ["new_prefetch", "new_turn", "switch", "rewind", "shutdown", "deadline"])
@pytest.mark.parametrize("blocked_phase", ["recall", "compiler"])
def test_late_completion_cannot_publish_across_boundary(provider, monkeypatch, boundary, blocked_phase):
    entered, release = threading.Event(), threading.Event()
    returned = []
    clock = [100.0]
    monkeypatch.setattr(plugin, "time", SimpleNamespace(monotonic=lambda: clock[0]))

    class Client(FakeClient):
        async def arecall(self, **kw):
            old = "old request" in kw["query"]
            if old and blocked_phase == "recall":
                entered.set()
                assert await asyncio.to_thread(release.wait, 5)
            return FakeRecallResponse(["old evidence" if old else "fresh evidence"])

    def auxiliary(**kw):
        request = json.loads(kw["messages"][1]["content"])
        if request["current_user_request"] == "old request" and blocked_phase == "compiler":
            entered.set()
            assert release.wait(5)
        return completion({"current_state_ids": ["M0001"]})

    install_aux(monkeypatch, auxiliary)
    instance, _ = provider({"context_compiler": True}, client=Client())
    instance.on_turn_start(1, "old request")
    worker = threading.Thread(target=lambda: returned.append(instance.prefetch("old request")))
    worker.start()
    try:
        assert entered.wait(5)
        if boundary == "new_prefetch":
            assert "fresh evidence" in instance.prefetch("new request")
        elif boundary == "new_turn":
            instance.on_turn_start(2, "")  # host skips recall on this turn
        elif boundary in {"switch", "rewind"}:
            instance.on_session_switch("session-2" if boundary == "switch" else "session-1", rewound=True)
        elif boundary == "shutdown":
            instance.shutdown()
        elif boundary == "deadline":
            clock[0] += instance.prefetch_timeout_seconds + 1
        status_before = instance.recall_status()
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive() and returned == [""]
    assert instance.recall_status() == status_before
    if boundary != "shutdown":
        instance.shutdown()


def test_empty_new_legacy_result_replaces_prior_result(provider):
    class Client(FakeClient):
        async def arecall(self, **kw):
            return FakeRecallResponse(["stale"] if kw["query"] == "old" else [])

    instance, _ = provider({"recall_sync": False, "prefetch_waits_for_retain": False}, client=Client())
    instance.queue_prefetch("old")
    instance._prefetch_thread.join(5)
    instance.queue_prefetch("new")
    instance._prefetch_thread.join(5)
    assert instance.prefetch("next") == "" and instance.recall_status() is None
    instance.shutdown()


def test_budget_covers_request_and_compiler_and_legacy_uses_host_budget(provider):
    instance, _ = provider({"timeout": 300})
    assert instance.prefetch_timeout_seconds == 390
    legacy, _ = provider({"recall_sync": False})
    assert legacy.prefetch_timeout_seconds is None
    instance.shutdown()
    legacy.shutdown()


def test_dialogue_bounds_preserve_roles_and_current_request():
    for budget in range(1, 150):
        text = build_recall_query("current", [{"role": "assistant", "content": "a" * 1000}], max_chars=budget)
        assert len(text) <= budget
        if "RECENT DIALOGUE" in text:
            assert "ASSISTANT: " in text
    text = build_recall_query("current", [{"role": "assistant", "content": "x" * 10000}], max_chars=0)
    assert len(text) < 2200


def test_current_turn_keeps_multibank_partial_failure_filtering_and_budget(provider, caplog):
    calls = []

    class Client(FakeClient):
        async def arecall(self, **kw):
            calls.append(kw)
            bank = kw["bank_id"]
            if bank == "down":
                raise RuntimeError("private query must not appear in diagnostics")
            facts = ["shared evidence", f"{bank} evidence"]
            return SimpleNamespace(
                results=[
                    RecallResult(id=f"{bank}-{i}", text=text, scores=RecallScores(final=0.9, reranker=0.9))
                    for i, text in enumerate(facts)
                ]
                + [RecallResult(id="low", text="below floor", scores=RecallScores(final=0.1, reranker=0.1))]
            )

    instance, _ = provider(
        {
            "bank_id": "primary",
            "additional_banks": ["down"],
            "recall_additional_banks": ["vault"],
            "recall_max_tokens": 123,
            "recall_min_scores": {"reranker": 0.5},
        },
        client=Client(),
    )
    text = instance.prefetch("private query")
    assert text.count("shared evidence") == 1
    assert text.index("primary evidence") < text.index("vault evidence")
    assert "below floor" not in text and instance.recall_status().count == 3
    assert {call["bank_id"] for call in calls} == {"primary", "down", "vault"}
    assert all(call["max_tokens"] == 123 for call in calls)
    assert "private query" not in caplog.text
    instance.shutdown()


def test_malformed_content_blocks_do_not_stringify_unknown_data(provider):
    instance, fake = provider()
    malformed = [{"type": {}, "text": "not text"}, {"type": [], "text": "not text"}]
    assert instance.prefetch(malformed) == "" and not fake.recalls
    instance.shutdown()


def test_dialogue_snapshot_is_shared_by_retrieval_and_compiler(provider, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    history = [{"role": "assistant", "content": [{"type": "text", "text": "original dialogue"}]}]
    observed = []

    class Client(FakeClient):
        async def arecall(self, **kw):
            observed.append(kw["query"])
            entered.set()
            assert await asyncio.to_thread(release.wait, 5)
            return FakeRecallResponse(["fact"])

    def auxiliary(**kw):
        observed.append(json.loads(kw["messages"][1]["content"])["recent_dialogue"])
        return completion({"current_state_ids": ["M0001"]})

    install_aux(monkeypatch, auxiliary)
    instance, _ = provider({"context_compiler": True}, client=Client())
    result = []
    worker = threading.Thread(target=lambda: result.append(instance.prefetch("request", messages=history)))
    worker.start()
    try:
        assert entered.wait(5)
        history[0]["content"][0]["text"] = "changed in another turn"
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive() and result and "fact" in result[0]
    assert "ASSISTANT: original dialogue" in observed[0]
    assert observed[1] == [{"role": "assistant", "content": "original dialogue"}]
    instance.shutdown()


def test_empty_legacy_queue_invalidates_a_pending_result(provider):
    instance, _ = provider(
        {"recall_sync": False, "prefetch_waits_for_retain": False}, client=FakeClient(recall_texts=["old"])
    )
    instance.queue_prefetch("old request")
    instance._prefetch_thread.join(5)
    instance.queue_prefetch("")
    assert instance.prefetch("later request") == ""
    instance.shutdown()


def test_late_bank_success_cannot_clear_newer_turns_failure_cooldown(provider):
    entered, release = threading.Event(), threading.Event()
    vault_queries = []
    result = []

    class Client(FakeClient):
        async def arecall(self, **kw):
            if kw["bank_id"] == "vault":
                vault_queries.append(kw["query"])
                if "old request" in kw["query"]:
                    entered.set()
                    assert await asyncio.to_thread(release.wait, 5)
                    return FakeRecallResponse(["late vault success"])
                raise RuntimeError("vault unavailable")
            return FakeRecallResponse(["primary evidence"])

    instance, _ = provider({"recall_additional_banks": ["vault"]}, client=Client())
    worker = threading.Thread(target=lambda: result.append(instance.prefetch("old request")))
    worker.start()
    try:
        assert entered.wait(5)
        assert "primary evidence" in instance.prefetch("new request")
    finally:
        release.set()
        worker.join(5)
    assert result == [""] and not worker.is_alive()
    assert "primary evidence" in instance.prefetch("third request")
    assert len(vault_queries) == 2  # cooldown from the NEW request still applies
    instance.shutdown()
