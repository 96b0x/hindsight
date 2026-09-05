"""OpenRouter needs explicit float output; SDK-default base64 can return no vectors."""

from types import SimpleNamespace

import pytest

from hindsight_api.engine.embeddings import OpenAIEmbeddings


@pytest.mark.asyncio
async def test_dimension_probe_requests_float(monkeypatch):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2], index=0)])

    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: SimpleNamespace(embeddings=SimpleNamespace(create=create)))
    embeddings = OpenAIEmbeddings(api_key="unit-test", model="custom-model", base_url="https://openrouter.ai/api/v1")
    await embeddings.initialize()
    assert calls[0]["encoding_format"] == "float"
    assert embeddings.dimension == 2


@pytest.mark.asyncio
async def test_every_query_and_document_batch_requests_float(monkeypatch):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[float(i), 0.2], index=i) for i in reversed(range(len(kwargs["input"])))]
        )

    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: SimpleNamespace(embeddings=SimpleNamespace(create=create)))
    embeddings = OpenAIEmbeddings(api_key="unit-test", dimensions=2, batch_size=2)
    await embeddings.initialize()
    for method in (embeddings.encode_query, embeddings.encode_documents):
        vectors = method(["one", "two", "three"])
        assert len(vectors) == 3
        assert vectors[0] == [0.0, 0.2]
        assert vectors[1] == [1.0, 0.2]
    assert len(calls) == 4
    assert all(call["encoding_format"] == "float" and call["dimensions"] == 2 for call in calls)
