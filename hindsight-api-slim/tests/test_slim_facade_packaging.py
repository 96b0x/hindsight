"""The custom importer facade must not opt remote-provider deployments into local ML."""

from pathlib import Path
import tomllib

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[2]


def test_facade_keeps_embedded_db_without_local_ml() -> None:
    project = tomllib.loads((ROOT / "hindsight-all/pyproject.toml").read_text())["project"]
    dependencies = {
        requirement.name: requirement for value in project["dependencies"] if (requirement := Requirement(value))
    }
    api = dependencies["hindsight-api-slim"]
    assert api.extras == {"embedded-db"}
    assert str(api.specifier) == f"=={project['version']}"
    assert "hindsight-client" in dependencies
    assert "hindsight-embed" in dependencies


def test_facade_lock_does_not_restore_all_extra() -> None:
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    facade = next(package for package in lock["package"] if package["name"] == "hindsight-all")
    dependency = next(item for item in facade["dependencies"] if item["name"] == "hindsight-api-slim")
    assert dependency["extra"] == ["embedded-db"]
    metadata = next(
        item
        for item in facade["metadata"]["requires-dist"]
        if item["name"] == "hindsight-api-slim" and "marker" not in item
    )
    assert metadata["extras"] == ["embedded-db"]


def test_plugin_dependencies_stay_client_only() -> None:
    project = tomllib.loads((ROOT / "hindsight-integrations/hermes/pyproject.toml").read_text())["project"]
    dependencies = [Requirement(value) for value in project["dependencies"]]
    assert {requirement.name for requirement in dependencies} == {"hindsight-client", "hindsight-embed"}
    assert all(not requirement.extras for requirement in dependencies)
