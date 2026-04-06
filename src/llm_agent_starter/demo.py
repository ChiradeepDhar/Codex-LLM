from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from llm_agent_starter.config import Settings
from llm_agent_starter.reporting import write_markdown_report
from llm_agent_starter.tools import build_tool_registry, execute_tool


@dataclass(slots=True)
class DemoResult:
    report: str
    report_path: str


def build_offline_demo(settings: Settings, title: str = "Offline Demo") -> DemoResult:
    registry = build_tool_registry()
    file_list = execute_tool(settings, registry, "list_files", '{"path": ".", "limit": 20}')
    workspace = execute_tool(settings, registry, "workspace_summary", '{"path": ".", "limit": 50}')
    readme_preview = execute_tool(settings, registry, "read_text_file", '{"path": "README.md"}')[:1600]
    env_guardrail = execute_tool(settings, registry, "read_text_file", '{"path": ".env"}')

    report = f"""
# {title}: LLM + Agentic AI Project Assistant

## What this demonstrates

This no-cost demo validates the local agent runtime without calling a model. It shows the same read-only tools that the live model-backed agent can use: file listing, workspace summarization, safe text-file reads, and secret guardrails.

## Where this report is stored

Reports are saved under the configured output directory:

```text
{settings.output_dir}
```

## Runtime Configuration

- Provider: `{settings.provider}`
- Model: `{settings.model_name}`
- Workspace: `{settings.workspace_root}`
- Output directory: `{settings.output_dir}`
- Generated at: `{datetime.now().isoformat(timespec="seconds")}`
- Max file bytes: `{settings.max_file_bytes}`
- Request timeout seconds: `{settings.request_timeout_seconds}`
- Max retries: `{settings.max_retries}`

## Workspace File Listing

```text
{file_list}
```

## Workspace Summary

```text
{workspace}
```

## README Preview

```text
{readme_preview}
```

## Secret Guardrail Check

Attempting to read `.env` returns:

```text
{env_guardrail}
```

## Live Demo Commands

Once your OpenAI quota is active, run:

```powershell
python -m llm_agent_starter.main basic --prompt "In two sentences, explain what this local LLM project assistant is useful for."
python -m llm_agent_starter.main ask "What are the important files in this project?"
python -m llm_agent_starter.main project-brief
```
""".strip()

    report_path = write_markdown_report(settings, title, report)
    return DemoResult(report=report, report_path=str(report_path))
