from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(slots=True)
class Settings:
    provider: str
    model_name: str
    base_url: str
    api_key: str
    workspace_root: str
    app_env: str
    log_level: str
    output_dir: str
    max_file_bytes: int
    request_timeout_seconds: float
    max_retries: int
    allowed_extensions: set[str]


def _parse_allowed_extensions(value: str) -> set[str]:
    extensions: set[str] = set()
    for item in value.split(","):
        cleaned = item.strip().lower()
        if cleaned:
            extensions.add(cleaned if cleaned.startswith(".") else f".{cleaned}")
    return extensions


def _common_settings() -> dict[str, object]:
    workspace_root = os.getcwd()
    output_dir = os.getenv("OUTPUT_DIR", "runs").strip() or "runs"
    allowed_extensions = os.getenv(
        "ALLOWED_EXTENSIONS",
        ".py,.md,.txt,.toml,.json,.yaml,.yml,.csv,.env.example",
    )

    return {
        "workspace_root": workspace_root,
        "app_env": os.getenv("APP_ENV", "local").strip() or "local",
        "log_level": os.getenv("LOG_LEVEL", "INFO").strip().upper() or "INFO",
        "output_dir": str((Path(workspace_root) / output_dir).resolve()),
        "max_file_bytes": int(os.getenv("MAX_FILE_BYTES", "20000")),
        "request_timeout_seconds": float(os.getenv("REQUEST_TIMEOUT_SECONDS", "60")),
        "max_retries": int(os.getenv("OPENAI_MAX_RETRIES", "2")),
        "allowed_extensions": _parse_allowed_extensions(allowed_extensions),
    }


def load_settings() -> Settings:
    load_dotenv()

    provider = os.getenv("MODEL_PROVIDER", "ollama").strip().lower()
    common = _common_settings()

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is required when MODEL_PROVIDER=openai. "
                "Create a .env file from .env.example first."
            )

        return Settings(
            provider=provider,
            model_name=os.getenv("MODEL_NAME", "gpt-4o-mini").strip(),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip(),
            api_key=api_key,
            **common,
        )

    return Settings(
        provider="ollama",
        model_name=os.getenv("MODEL_NAME", "llama3.2:3b").strip(),
        base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1").strip(),
        api_key=os.getenv("OPENAI_API_KEY", "ollama").strip() or "ollama",
        **common,
    )
