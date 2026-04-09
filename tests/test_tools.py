from __future__ import annotations

from llm_agent_starter.tools import build_tool_registry, execute_tool


def test_read_text_file_blocks_env(settings_factory, workspace_factory) -> None:
    workspace = workspace_factory()
    settings = settings_factory(workspace)
    (workspace / ".env").write_text("OPENAI_API_KEY=secret", encoding="utf-8")

    result = execute_tool(settings, build_tool_registry(), "read_text_file", '{"path": ".env"}')

    assert "not allowed" in result.lower()


def test_read_text_file_rejects_path_traversal(settings_factory, workspace_factory) -> None:
    workspace = workspace_factory()
    settings = settings_factory(workspace)

    result = execute_tool(settings, build_tool_registry(), "read_text_file", '{"path": "../outside.txt"}')

    assert "Path must stay inside the current workspace." in result


def test_read_text_file_rejects_large_files(settings_factory, workspace_factory) -> None:
    workspace = workspace_factory()
    settings = settings_factory(workspace, max_file_bytes=4)
    (workspace / "big.txt").write_text("12345", encoding="utf-8")

    result = execute_tool(settings, build_tool_registry(), "read_text_file", '{"path": "big.txt"}')

    assert "too large" in result.lower()


def test_execute_tool_rejects_invalid_json(settings_factory, workspace_factory) -> None:
    workspace = workspace_factory()
    settings = settings_factory(workspace)

    result = execute_tool(settings, build_tool_registry(), "read_text_file", '{"path": ')

    assert result.startswith("Invalid tool arguments:")
