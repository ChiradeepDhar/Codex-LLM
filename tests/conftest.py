from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def workspace_factory():
    base_dir = ROOT / "runs" / "test-workspaces"
    base_dir.mkdir(parents=True, exist_ok=True)

    def _build() -> Path:
        workspace = base_dir / f"workspace-{uuid4().hex}"
        workspace.mkdir(parents=True, exist_ok=False)
        return workspace

    return _build


@pytest.fixture
def settings_factory():
    from llm_agent_starter.config import Settings

    def _build(root: Path, **overrides: object) -> Settings:
        values: dict[str, object] = {
            "provider": "ollama",
            "model_name": "llama3.2:3b",
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama",
            "workspace_root": str(root),
            "app_env": "test",
            "log_level": "INFO",
            "output_dir": str((root / "runs").resolve()),
            "max_file_bytes": 20_000,
            "request_timeout_seconds": 60.0,
            "max_retries": 2,
            "allowed_extensions": {
                ".py",
                ".md",
                ".txt",
                ".toml",
                ".json",
                ".yaml",
                ".yml",
                ".csv",
                ".env.example",
            },
        }
        values.update(overrides)
        return Settings(**values)

    return _build
