"""Opt-in cross-check against a real Hermes checkout, without conftest stubs."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_current_hermes_auxiliary_wire(tmp_path):
    source = os.environ.get("HERMES_SOURCE_PATH")
    if not source:
        pytest.skip("Set HERMES_SOURCE_PATH to run the real Hermes auxiliary/PM contract")
    env = {
        "PATH": os.defpath,
        "HOME": str(tmp_path),
        "HERMES_HOME": str(tmp_path / "hermes"),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
        "PYTHONPATH": source,
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    result = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("real_hermes_wire.py"))],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "native PM declaration: passed" in result.stdout


@pytest.mark.parametrize("isolation", ["in_process", "host"])
def test_current_hermes_provider_loader(tmp_path, isolation):
    source = os.environ.get("HERMES_SOURCE_PATH")
    if not source:
        pytest.skip("Set HERMES_SOURCE_PATH to run the real Hermes loader/host contract")
    env = {
        "PATH": os.defpath,
        "HOME": str(tmp_path),
        "HERMES_HOME": str(tmp_path / "hermes"),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
        "PYTHONPATH": source,
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    result = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("real_hermes_loader.py")), isolation],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"real_provider_loader": true' in result.stdout
