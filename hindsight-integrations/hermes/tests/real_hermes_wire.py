"""Executed in a clean subprocess: real Hermes + real OpenAI SDK, HTTP stub only.

Set HERMES_SOURCE_PATH to a checkout with the public call_auxiliary_task facade.
This is separate from conftest's reduced Hermes modules so API drift is visible.
"""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import sys
from pathlib import Path
from unittest.mock import patch

import httpx


def main():
    root = Path(__file__).resolve().parents[1]
    home = Path(os.environ["HERMES_HOME"])
    home.mkdir(parents=True, exist_ok=True)
    # This is a disposable test profile, never a user's provider config.
    (home / "config.yaml").write_text(
        json.dumps(
            {
                "auxiliary": {
                    "transient_retries": 0,
                    "memory_context": {
                        "provider": "custom",
                        "model": "fixture-memory-model",
                        "base_url": "https://memory.invalid/v1",
                        "api_key": "test-only",
                        "api_mode": "chat_completions",
                        "timeout": 17,
                        "max_concurrency": 1,
                    },
                },
            }
        )
    )
    import agent.auxiliary_client as auxiliary
    from agent.memory_provider import call_auxiliary_task
    from pm.plugin_declarations import native_manifest_file, read_native_manifest, read_python_declaration

    manifest = read_native_manifest(native_manifest_file(root))
    declaration = read_python_declaration(root)
    assert manifest["name"] == "hindsight"
    assert declaration.is_member and declaration.pyproject == root / "pyproject.toml"
    assert len(declaration.install_requirements) == 2

    spec = importlib.util.spec_from_file_location("real_hindsight_compiler", root / "compiler.py")
    compiler = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = compiler
    spec.loader.exec_module(compiler)
    requests = []
    timeout_seen = []
    response_content = '{"current_state_ids":["M0001"],"tags":["fabricated"]}'

    def http_handler(request):
        assert request.url == "https://memory.invalid/v1/chat/completions"
        requests.append(json.loads(request.content))
        timeout_seen.append(request.extensions["timeout"])
        return httpx.Response(
            200,
            json={
                "id": "stub-completion",
                "object": "chat.completion",
                "created": 1,
                "model": "fixture-memory-model",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": response_content}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            },
        )

    def no_socket(*args, **kwargs):
        raise AssertionError("Real network is forbidden in this regression")

    transport = httpx.MockTransport(http_handler)
    with (
        httpx.Client(transport=transport) as http_client,
        patch.object(auxiliary, "_openai_http_client_kwargs", return_value={"http_client": http_client}),
        patch.object(socket.socket, "connect", no_socket),
    ):
        compiled = compiler.compile_memory_context(
            current_query="why this decision?",
            recent_messages=[{"role": "assistant", "content": "previous answer"}],
            candidates=[
                compiler.MemoryCandidate(candidate_id="M0001", memory_id="fact", text="EXACT original evidence")
            ],
        )
        assert compiled is not None, "Real auxiliary facade failed; do not hide drift with a call_llm mock"
        assert compiled.count == 1 and "EXACT original evidence" in compiled.text
        assert "fabricated" not in compiled.text
        assert len(requests) == 1
        wire = requests[0]
        assert wire["model"] == "fixture-memory-model"
        assert wire["temperature"] == 0
        # Current Hermes deliberately omits token caps on generic custom
        # routes. The plugin supplies 1200 to the facade, which owns this policy.
        assert "max_tokens" not in wire and "max_completion_tokens" not in wire
        assert [m["role"] for m in wire["messages"]] == ["system", "user"]
        payload = json.loads(wire["messages"][1]["content"])
        assert payload["current_user_request"] == "why this decision?"
        assert payload["recent_dialogue"] == [{"role": "assistant", "content": "previous answer"}]
        assert payload["candidates"][0]["id"] == "M0001"
        assert timeout_seen[0]["read"] == 17
        # The helper itself must preserve the SDK completion envelope.
        result = call_auxiliary_task("memory_context", messages=[{"role": "user", "content": "probe"}])
        assert result.choices[0].message.content == response_content
    auxiliary.shutdown_cached_clients()
    print("Real Hermes facade, task config/model routing, OpenAI SDK HTTP envelope and native PM declaration: passed")


if __name__ == "__main__":
    main()
