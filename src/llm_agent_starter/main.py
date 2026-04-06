from __future__ import annotations

import argparse
import json
import sys

from openai import APIConnectionError, APIStatusError, RateLimitError

from llm_agent_starter.agentic import build_project_brief_task, run_agent_task
from llm_agent_starter.client import build_client
from llm_agent_starter.config import load_settings
from llm_agent_starter.demo import build_offline_demo
from llm_agent_starter.llm_basic import run_basic_prompt
from llm_agent_starter.logging_config import configure_logging
from llm_agent_starter.reporting import write_markdown_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-starter",
        description="Starter project for LLM and agentic AI concepts.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    show_config = subparsers.add_parser("show-config", help="Print the active model configuration.")
    show_config.set_defaults(command="show-config")

    doctor = subparsers.add_parser("doctor", help="Check local configuration without printing secrets.")
    doctor.set_defaults(command="doctor")

    demo = subparsers.add_parser("demo", help="Run a no-cost offline demo and save a report.")
    demo.set_defaults(command="demo")

    basic = subparsers.add_parser("basic", help="Run a simple LLM prompt.")
    basic.add_argument("--prompt", required=True, help="Prompt to send to the model.")

    agent = subparsers.add_parser("agent", help="Run a tool-using local project assistant.")
    agent.add_argument("--task", required=True, help="Task for the agent.")
    agent.add_argument("--max-steps", type=int, default=6, help="Maximum number of reasoning steps.")
    agent.add_argument("--save-trace", action="store_true", help="Save a JSON trace of tool usage.")

    ask = subparsers.add_parser("ask", help="Ask a question about this workspace.")
    ask.add_argument("question", help="Question to answer using local project context.")
    ask.add_argument("--max-steps", type=int, default=6, help="Maximum number of reasoning steps.")

    brief = subparsers.add_parser("project-brief", help="Generate a saved Markdown project brief.")
    brief.add_argument("--max-steps", type=int, default=8, help="Maximum number of reasoning steps.")

    return parser


def _print_api_error(exc: Exception) -> None:
    if isinstance(exc, RateLimitError):
        print(
            "OpenAI request failed: the configured key hit a rate limit or quota limit. "
            "If the error says insufficient_quota, check billing/quota and try again."
        )
        return
    if isinstance(exc, APIConnectionError):
        print(
            "OpenAI request failed: the API could not be reached. "
            "Check your network, proxy, and OPENAI_BASE_URL."
        )
        return
    if isinstance(exc, APIStatusError):
        print(f"OpenAI request failed with status {exc.status_code}: {exc.message}")
        return
    raise exc


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        settings = load_settings()
    except ValueError as exc:
        parser.error(str(exc))
        return

    configure_logging(settings)

    if args.command == "show-config":
        printable = {
            "provider": settings.provider,
            "model_name": settings.model_name,
            "base_url": settings.base_url,
            "workspace_root": settings.workspace_root,
            "output_dir": settings.output_dir,
            "max_file_bytes": settings.max_file_bytes,
            "request_timeout_seconds": settings.request_timeout_seconds,
            "max_retries": settings.max_retries,
            "allowed_extensions": sorted(settings.allowed_extensions),
        }
        print(json.dumps(printable, indent=2))
        return

    if args.command == "doctor":
        has_real_key = bool(settings.api_key and settings.api_key != "ollama")
        checks = {
            "provider": settings.provider,
            "model_name": settings.model_name,
            "base_url": settings.base_url,
            "api_key_configured": has_real_key,
            "workspace_root": settings.workspace_root,
            "output_dir": settings.output_dir,
        }
        print(json.dumps(checks, indent=2))
        if settings.provider == "openai" and not has_real_key:
            sys.exit(1)
        return

    if args.command == "demo":
        result = build_offline_demo(settings)
        print(f"Offline demo report saved to: {result.report_path}")
        print("\nPreview:")
        print(result.report[:1200])
        return

    client = build_client(settings)

    if args.command == "basic":
        try:
            print(run_basic_prompt(client, settings, args.prompt))
        except (RateLimitError, APIConnectionError, APIStatusError) as exc:
            _print_api_error(exc)
            sys.exit(1)
        return

    if args.command == "agent":
        try:
            result = run_agent_task(client, settings, args.task, max_steps=args.max_steps, save_trace=args.save_trace)
        except (RateLimitError, APIConnectionError, APIStatusError) as exc:
            _print_api_error(exc)
            sys.exit(1)
        print(result.answer)
        if result.artifact_path:
            print(f"\nTrace saved to: {result.artifact_path}")
        print("\nAgent steps:")
        for step in result.steps:
            print(json.dumps(step, indent=2))
        return

    if args.command == "ask":
        task = f"Answer this question about the current workspace: {args.question}"
        try:
            result = run_agent_task(client, settings, task, max_steps=args.max_steps)
        except (RateLimitError, APIConnectionError, APIStatusError) as exc:
            _print_api_error(exc)
            sys.exit(1)
        print(result.answer)
        return

    if args.command == "project-brief":
        try:
            result = run_agent_task(client, settings, build_project_brief_task(), max_steps=args.max_steps, save_trace=True)
        except (RateLimitError, APIConnectionError, APIStatusError) as exc:
            _print_api_error(exc)
            sys.exit(1)
        report_path = write_markdown_report(settings, "project-brief", result.answer)
        print(f"Project brief saved to: {report_path}")
        if result.artifact_path:
            print(f"Agent trace saved to: {result.artifact_path}")
        return


if __name__ == "__main__":
    main()
