"""OpenRouter needs explicit float output; SDK-default base64 can return no vectors."""

import json
from types import SimpleNamespace

import httpx
import pytest

from hindsight_api.engine.embeddings import OpenAIEmbeddings


@pytest.mark.asyncio
async def test_dimension_probe_requests_float(monkeypatch):
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2], index=0)])

    monkeypatch.setattr(
        "openai.AsyncOpenAI",
        lambda **kwargs: SimpleNamespace(api_key=kwargs["api_key"], embeddings=SimpleNamespace(create=create)),
    )
    embeddings = OpenAIEmbeddings(api_key="unit-test", model="custom-model", base_url="https://openrouter.ai/api/v1")
    await embeddings.initialize()
    assert calls[0]["encoding_format"] == "float"
    assert embeddings.dimension == 2


@pytest.mark.asyncio
async def test_every_query_and_document_batch_requests_float(monkeypatch):
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[float(i), 0.2], index=i) for i in reversed(range(len(kwargs["input"])))]
        )

    monkeypatch.setattr(
        "openai.AsyncOpenAI",
        lambda **kwargs: SimpleNamespace(api_key=kwargs["api_key"], embeddings=SimpleNamespace(create=create)),
    )
    embeddings = OpenAIEmbeddings(api_key="unit-test", dimensions=2, batch_size=2)
    await embeddings.initialize()
    for method in (embeddings.encode_query, embeddings.encode_documents):
        vectors = await method(["one", "two", "three"])
        assert len(vectors) == 3
        assert vectors[0] == [0.0, 0.2]
        assert vectors[1] == [1.0, 0.2]
    assert len(calls) == 4
    assert all(call["encoding_format"] == "float" and call["dimensions"] == 2 for call in calls)


@pytest.mark.asyncio
async def test_probe_and_batches_request_float_on_real_sdk_wire(monkeypatch):
    """Exercise SDK serialization, not only a fake embeddings.create signature."""
    from openai import AsyncOpenAI

    requests = []

    def handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": body["model"],
                "data": [
                    {"object": "embedding", "index": i, "embedding": [float(i), 0.2]}
                    for i in reversed(range(len(body["input"])))
                ],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )

    client = AsyncOpenAI(
        api_key="unit-test",
        base_url="https://embedding.test/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )
    monkeypatch.setattr("openai.AsyncOpenAI", lambda **kwargs: client)
    embeddings = OpenAIEmbeddings(api_key="unit-test", model="custom-model", batch_size=2)
    try:
        await embeddings.initialize()
        assert embeddings.dimension == 2
        for method in (embeddings.encode_query, embeddings.encode_documents):
            assert await method(["one", "two", "three"]) == [[0.0, 0.2], [1.0, 0.2], [0.0, 0.2]]
    finally:
        await client.close()

    assert len(requests) == 5
    assert requests[0]["input"] == ["test"]
    assert all(request["encoding_format"] == "float" for request in requests)
