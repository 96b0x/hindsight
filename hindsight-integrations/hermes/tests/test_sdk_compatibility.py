"""Real installed client/embed API compatibility; local HTTP fixture only."""

import inspect
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import version

import pytest
from hindsight_client import Hindsight

import hindsight_hermes as plugin


@pytest.fixture
def api_stub():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.path, body, dict(self.headers)))
            if self.path.endswith("/memories/recall"):
                response = {
                    "results": [
                        {
                            "id": "sdk-fact",
                            "text": "SDK original evidence",
                            "type": "observation",
                            "document_id": "sdk-document",
                            "occurred_start": "2026-09-05T00:00:00Z",
                            "mentioned_at": "2026-09-06T00:00:00+00:00",
                            "tags": ["sdk-tag"],
                            "metadata": {"source": "sdk-source"},
                            "scores": {"reranker": 0.9, "final": 0.8},
                        }
                    ]
                }
            elif self.path.endswith("/reflect"):
                response = {"text": "SDK synthesis"}
            elif self.path.endswith("/memories"):
                response = {
                    "success": True,
                    "bank_id": "hermes",
                    "items_count": 1,
                    "async": True,
                    "operation_id": "stub-operation",
                }
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        def do_GET(self):
            requests.append((self.path, None, dict(self.headers)))
            response = {"operation_id": "stub-operation", "status": "completed"}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        worker.join(5)


def test_real_sdk_through_public_provider_lifecycle(provider, api_stub):
    url, requests = api_stub
    client = Hindsight(base_url=url, api_key="test-only", timeout=2, user_agent="hermes-compat-test")
    instance, _ = provider({"recall_tags": ["tag"], "recall_min_scores": {"reranker": 0.5}}, client=client)
    try:
        instance.on_turn_start(1, "current SDK request")
        text = instance.prefetch("current SDK request", messages=[{"role": "assistant", "content": "prior SDK answer"}])
        for fragment in [
            "SDK original evidence",
            "sdk-fact",
            "sdk-document",
            "sdk-source",
            "sdk-tag",
            "2026-09-05T00:00:00Z",
        ]:
            assert fragment in text
        assert instance.recall_status().count == 1
        recall = next(body for path, body, _ in requests if path.endswith("/memories/recall"))
        assert recall["query"].startswith("CURRENT USER REQUEST:\ncurrent SDK request")
        assert "ASSISTANT: prior SDK answer" in recall["query"]
        assert recall["budget"] == "mid" and recall["max_tokens"] == 4096
        assert recall["types"] == ["observation"] and recall["tags"] == ["tag"]
        assert recall["min_scores"]["reranker"] == 0.5
        assert "SDK synthesis" in instance.handle_tool_call("hindsight_reflect", {"query": "synthesize"})
        instance.sync_turn("user SDK turn", "assistant SDK turn", session_id="session-1")
        instance._retain_queue.join()
        assert instance._wait_for_retains_drained(2)
        retained = next(body for path, body, _ in requests if path.endswith("/memories"))
        item = retained["items"][0]
        assert "user SDK turn" in item["content"] and "assistant SDK turn" in item["content"]
        assert item["metadata"]["session_id"] == "session-1"
        assert item["timestamp"].endswith("+00:00") or item["timestamp"].endswith("Z")
        assert any("/operations/stub-operation" in path for path, _, _ in requests)
    finally:
        instance.shutdown()


def test_installed_embed_surface_without_starting_a_daemon(hermes_env):
    import hindsight_embed
    from hindsight_embed.daemon_embed_manager import DaemonEmbedManager

    assert callable(hindsight_embed.get_embed_manager)
    assert plugin._check_local_runtime().available
    # Inspect signatures on the real class, not a fake manager that accepts anything.
    signature = inspect.signature(DaemonEmbedManager.ensure_running)
    signature.bind(None, {"HINDSIGHT_API_PORT": "9999"}, "isolated-test")
    for name in ("get_url", "is_running", "stop"):
        inspect.signature(getattr(DaemonEmbedManager, name)).bind(None, "isolated-test")
    assert version("hindsight-client") and version("hindsight-embed")
