from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from llm_agent_starter.config import Settings
from llm_agent_starter.reporting import write_json_artifact
from llm_agent_starter.tools import build_tool_registry, execute_tool


SYSTEM_PROMPT = """
You are a production-minded local project assistant.

Your job:
- Help the user understand, improve, and operate the current workspace.
- Use tools to inspect files before making claims about the project.
- Keep answers practical, specific, and concise.
- Never ask to read the local .env file and never repeat secrets.
- If a requested action is risky or destructive, explain the risk and suggest a safer next step.
- When you produce recommendations, include assumptions and next actions.
""".strip()

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AgentResult:
    answer: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    artifact_path: str | None = None


def _safe_json_loads(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value or "{}")
    except json.JSONDecodeError:
        return {"_raw": value}
    return parsed if isinstance(parsed, dict) else {"_value": parsed}


def run_agent_task(
    client: OpenAI,
    settings: Settings,
    task: str,
    max_steps: int = 6,
    save_trace: bool = False,
) -> AgentResult:
    registry = build_tool_registry()
    tools = [tool.as_openai_tool() for tool in registry.values()]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    steps: list[dict[str, Any]] = []

    for step_number in range(1, max_steps + 1):
        logger.info("Running agent step %s", step_number)
        response = client.chat.completions.create(
            model=settings.model_name,
            temperature=0,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        message = response.choices[0].message
        tool_calls = message.tool_calls or []

        if not tool_calls:
            answer = (message.content or "").strip() or "No final answer returned."
            result = AgentResult(answer=answer, steps=steps)
            if save_trace:
                result.artifact_path = str(
                    write_json_artifact(
                        settings,
                        "agent-trace",
                        {"task": task, "answer": result.answer, "steps": result.steps},
                    )
                )
            return result

        messages.append(
            {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                    for tool_call in tool_calls
                ],
            }
        )

        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_arguments = tool_call.function.arguments or "{}"
            tool_output = execute_tool(settings, registry, tool_name, tool_arguments)

            steps.append(
                {
                    "step": step_number,
                    "tool": tool_name,
                    "arguments": _safe_json_loads(tool_arguments),
                    "output_preview": tool_output[:1000],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_output,
                }
            )

    result = AgentResult(
        answer="The agent reached the maximum number of steps before producing a final answer.",
        steps=steps,
    )
    if save_trace:
        result.artifact_path = str(
            write_json_artifact(settings, "agent-trace", {"task": task, "answer": result.answer, "steps": steps})
        )
    return result


def build_project_brief_task() -> str:
    return """
Create a production-oriented project brief for the current workspace.

Use tools to inspect the file tree and the key source/configuration files. Return a Markdown report with:
- Executive summary
- Current capabilities
- Architecture and important files
- Configuration and secrets handling notes
- Operational risks
- Recommended next improvements
- Suggested commands to run locally

Do not read the local .env file. You may read .env.example.
""".strip()
