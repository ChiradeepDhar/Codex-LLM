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
