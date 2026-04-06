from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from llm_agent_starter.config import Settings


@dataclass(frozen=True, slots=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[Settings, dict[str, Any]], str]

    def as_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _safe_eval(expr: str) -> Any:
    allowed_nodes = {
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Mod,
        ast.Pow,
        ast.FloorDiv,
        ast.USub,
        ast.UAdd,
        ast.Constant,
        ast.Load,
    }

    parsed = ast.parse(expr, mode="eval")
    for node in ast.walk(parsed):
        if type(node) not in allowed_nodes:
            raise ValueError(f"Unsupported expression element: {type(node).__name__}")
    return eval(compile(parsed, "<expression>", "eval"), {"__builtins__": {}}, {})


def resolve_in_workspace(settings: Settings, relative_path: str) -> Path:
    root = Path(settings.workspace_root).resolve()
    target = (root / relative_path).resolve()
    if root != target and root not in target.parents:
        raise ValueError("Path must stay inside the current workspace.")
    return target


def _is_allowed_text_file(settings: Settings, path: Path) -> bool:
    if path.name == ".env":
        return False
    if path.name == ".env.example":
        return ".env.example" in settings.allowed_extensions
    return path.suffix.lower() in settings.allowed_extensions


def _should_skip_path(settings: Settings, path: Path) -> bool:
    ignored_names = {".git", ".venv", "__pycache__", ".pytest_cache"}
    if any(part in ignored_names for part in path.parts):
        return True

    try:
        output_dir = Path(settings.output_dir).resolve()
        resolved = path.resolve()
        if resolved == output_dir or output_dir in resolved.parents:
            return True
    except OSError:
        return True

    return False


def _list_files(settings: Settings, tool_input: dict[str, Any]) -> str:
    target = resolve_in_workspace(settings, str(tool_input.get("path", ".")))
    limit = int(tool_input.get("limit", 100))

    if not target.exists():
        return f"Path does not exist: {target}"
    if not target.is_dir():
        return f"Path is not a directory: {target}"

    root = Path(settings.workspace_root)
    items: list[str] = []
    for child in sorted(target.iterdir(), key=lambda item: item.name.lower()):
        if _should_skip_path(settings, child):
            continue
        if child.name.startswith(".env") and child.name != ".env.example":
            continue
        suffix = "/" if child.is_dir() else ""
        items.append(child.relative_to(root).as_posix() + suffix)

    return "\n".join(items[:limit]) if items else "(empty directory)"


def _read_text_file(settings: Settings, tool_input: dict[str, Any]) -> str:
    path = str(tool_input["path"])
    target = resolve_in_workspace(settings, path)

    if not target.exists():
        return f"File does not exist: {path}"
    if not target.is_file():
        return f"Path is not a file: {path}"
    if not _is_allowed_text_file(settings, target):
        return "This file type is not allowed, or it may contain local secrets."
    if target.stat().st_size > settings.max_file_bytes:
        return f"File is too large to read safely ({target.stat().st_size} bytes)."

    return target.read_text(encoding="utf-8", errors="replace")


def _search_text(settings: Settings, tool_input: dict[str, Any]) -> str:
    query = str(tool_input["query"])
    path = str(tool_input.get("path", "."))
    limit = int(tool_input.get("limit", 30))
    target = resolve_in_workspace(settings, path)

    if not target.exists():
        return f"Path does not exist: {path}"

    files = [target] if target.is_file() else sorted(target.rglob("*"))
    matches: list[str] = []
    root = Path(settings.workspace_root)

    for file_path in files:
        if len(matches) >= limit:
            break
        if _should_skip_path(settings, file_path):
            continue
        if not file_path.is_file() or not _is_allowed_text_file(settings, file_path):
            continue
        if file_path.stat().st_size > settings.max_file_bytes:
            continue

        content = file_path.read_text(encoding="utf-8", errors="replace")
        for line_number, line in enumerate(content.splitlines(), start=1):
            if query.lower() in line.lower():
                rel = file_path.relative_to(root).as_posix()
                matches.append(f"{rel}:{line_number}: {line.strip()[:240]}")
                if len(matches) >= limit:
                    break

    return "\n".join(matches) if matches else "No matches found."


def _workspace_summary(settings: Settings, tool_input: dict[str, Any]) -> str:
    path = str(tool_input.get("path", "."))
    target = resolve_in_workspace(settings, path)
    limit = int(tool_input.get("limit", 120))

    if not target.exists():
        return f"Path does not exist: {path}"
    if not target.is_dir():
        return f"Path is not a directory: {path}"

    root = Path(settings.workspace_root)
    lines: list[str] = []
    for file_path in sorted(target.rglob("*")):
        if len(lines) >= limit:
            break
        if _should_skip_path(settings, file_path):
            continue
        if not file_path.is_file():
            continue
        if not _is_allowed_text_file(settings, file_path):
            continue
        rel = file_path.relative_to(root).as_posix()
        lines.append(f"{rel} ({file_path.stat().st_size} bytes)")

    return "\n".join(lines) if lines else "No readable project files found."


def _calculate(settings: Settings, tool_input: dict[str, Any]) -> str:
    del settings
    return str(_safe_eval(str(tool_input["expression"])))


def _current_time(settings: Settings, tool_input: dict[str, Any]) -> str:
    del settings, tool_input
    return datetime.now().isoformat(timespec="seconds")


def build_tool_registry() -> dict[str, Tool]:
    tools = [
        Tool(
            name="list_files",
            description="List files and directories inside the local workspace.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Workspace-relative directory path."},
                    "limit": {"type": "integer", "description": "Maximum number of entries to return."},
                },
                "required": [],
                "additionalProperties": False,
            },
            handler=_list_files,
        ),
        Tool(
            name="read_text_file",
            description="Read a small, safe text file inside the workspace. The local .env file is blocked.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Workspace-relative file path."},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            handler=_read_text_file,
        ),
        Tool(
            name="search_text",
            description="Search readable workspace text files for a case-insensitive string.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Text to search for."},
                    "path": {"type": "string", "description": "Workspace-relative path to search."},
                    "limit": {"type": "integer", "description": "Maximum number of matches to return."},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            handler=_search_text,
        ),
        Tool(
            name="workspace_summary",
            description="Return a compact list of readable files in the workspace.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Workspace-relative directory path."},
                    "limit": {"type": "integer", "description": "Maximum number of files to return."},
                },
                "required": [],
                "additionalProperties": False,
            },
            handler=_workspace_summary,
        ),
        Tool(
            name="calculate",
            description="Evaluate a simple arithmetic expression.",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Arithmetic expression to evaluate."},
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
            handler=_calculate,
        ),
        Tool(
            name="current_time",
            description="Return the current local time.",
            parameters={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
            handler=_current_time,
        ),
    ]
    return {tool.name: tool for tool in tools}


def execute_tool(settings: Settings, registry: dict[str, Tool], name: str, arguments: str) -> str:
    tool = registry.get(name)
    if tool is None:
        return f"Unknown tool requested: {name}"

    try:
        payload = json.loads(arguments or "{}")
    except json.JSONDecodeError as exc:
        return f"Invalid tool arguments: {exc}"

    try:
        return tool.handler(settings, payload)
    except Exception as exc:
        return f"Tool error: {exc}"
