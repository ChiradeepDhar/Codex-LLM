from __future__ import annotations

from llm_agent_starter.reporting import (
    format_artifact_table,
    list_artifacts,
    write_json_artifact,
    write_markdown_report,
)


def test_reporting_writes_and_lists_artifacts(settings_factory, workspace_factory) -> None:
    workspace = workspace_factory()
    settings = settings_factory(workspace)

    markdown_path = write_markdown_report(settings, "Test Report", "# Hello")
    json_path = write_json_artifact(settings, "Trace", {"ok": True})

    artifacts = list_artifacts(settings, limit=10)
    artifact_names = {artifact["name"] for artifact in artifacts}

    assert markdown_path.name in artifact_names
    assert json_path.name in artifact_names


def test_format_artifact_table_has_headers(settings_factory, workspace_factory) -> None:
    workspace = workspace_factory()
    settings = settings_factory(workspace)
    write_markdown_report(settings, "Test Report", "# Hello")

    table = format_artifact_table(list_artifacts(settings, limit=10))

    assert "Name | Type | Bytes | Modified | Path" in table
