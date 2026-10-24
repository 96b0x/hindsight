"""Real Hermes provider discovery and plugin-host RPC, with disposable state."""

from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path


def main():
    source = Path(__file__).resolve().parents[1]
    home = Path(os.environ["HERMES_HOME"])
    isolation = sys.argv[1]
    (home / "plugins").mkdir(parents=True)
    (home / "plugins/hindsight").symlink_to(source, target_is_directory=True)
    (home / "config.yaml").write_text(json.dumps({"plugins": {"isolation": isolation}}))
    (home / "hindsight").mkdir()
    (home / "hindsight/config.json").write_text(
        json.dumps(
            {"mode": "local_external", "api_url": "http://127.0.0.1:1", "auto_recall": False, "auto_retain": False}
        )
    )

    from agent.memory_provider import MemoryProvider
    from hermes_cli.plugin_isolation import user_plugin_host
    from plugins.memory import find_provider_dir, load_memory_provider

    host = None
    try:
        path = find_provider_dir("hindsight")
        assert path and path.resolve() == source.resolve()
        provider = load_memory_provider("hindsight")
        host = user_plugin_host()
        assert isinstance(provider, MemoryProvider), repr(provider)
        assert provider.name == "hindsight"
        assert "messages" in inspect.signature(provider.prefetch).parameters
        assert provider.prefetch_timeout_seconds >= 180
        if isolation == "host":
            assert host is not None
        else:
            assert host is None
            module = sys.modules[type(provider).__module__]
            assert Path(module.__file__).resolve() == (source / "__init__.py").resolve()
        # Initializing applies the explicit no-recall/no-retain test config. An
        # uninitialized provider has cloud defaults and may lazily make a request.
        provider.initialize("real-loader-probe", platform="cli", agent_context="primary")
        # Exercise the real host signature/RPC and lifecycle without any service.
        provider.on_turn_start(1, "current question")
        assert provider.prefetch("current question", messages=[{"role": "assistant", "content": "prior answer"}]) == ""
        assert provider.recall_status() is None
        provider.shutdown()
        print(
            json.dumps(
                {
                    "real_provider_loader": True,
                    "isolation": isolation,
                    "messages_rpc": True,
                    "live_budget_property": True,
                }
            )
        )
    finally:
        if host is not None:
            host.shutdown()


if __name__ == "__main__":
    main()
