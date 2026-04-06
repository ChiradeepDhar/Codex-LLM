from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from llm_agent_starter.config import Settings


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return slug[:64] or "run"


def ensure_output_dir(settings: Settings) -> Path:
    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_markdown_report(settings: Settings, title: str, content: str) -> Path:
    output_dir = ensure_output_dir(settings)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = output_dir / f"{timestamp}-{_slugify(title)}.md"
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def write_json_artifact(settings: Settings, title: str, payload: dict[str, Any]) -> Path:
    output_dir = ensure_output_dir(settings)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = output_dir / f"{timestamp}-{_slugify(title)}.json"

    def default(value: Any) -> Any:
        if is_dataclass(value):
            return asdict(value)
        return str(value)

    path.write_text(json.dumps(payload, indent=2, default=default), encoding="utf-8")
    return path


def list_artifacts(settings: Settings, limit: int = 10) -> list[dict[str, Any]]:
    output_dir = ensure_output_dir(settings)
    artifacts = [path for path in output_dir.iterdir() if path.is_file() and path.suffix in {".md", ".json"}]
    artifacts.sort(key=lambda path: path.stat().st_mtime, reverse=True)

    results: list[dict[str, Any]] = []
    for path in artifacts[:limit]:
        stats = path.stat()
        results.append(
            {
                "name": path.name,
                "path": str(path),
                "type": path.suffix.lstrip("."),
                "bytes": stats.st_size,
                "modified": datetime.fromtimestamp(stats.st_mtime).isoformat(timespec="seconds"),
            }
        )
    return results


def format_artifact_table(artifacts: list[dict[str, Any]]) -> str:
    if not artifacts:
        return "No reports or artifacts found yet."

    lines = ["Name | Type | Bytes | Modified | Path", "--- | --- | ---: | --- | ---"]
    for artifact in artifacts:
        lines.append(
            f"{artifact['name']} | {artifact['type']} | {artifact['bytes']} | "
            f"{artifact['modified']} | {artifact['path']}"
        )
    return "\n".join(lines)
