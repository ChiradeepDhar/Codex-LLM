from __future__ import annotations

import pytest

from llm_agent_starter import config as config_module
from llm_agent_starter.config import load_settings


def test_load_settings_defaults_to_ollama(monkeypatch, workspace_factory) -> None:
    workspace = workspace_factory()
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(config_module, "load_dotenv", lambda: None)
    for key in (
        "MODEL_PROVIDER",
        "MODEL_NAME",
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "APP_ENV",
        "LOG_LEVEL",
        "OUTPUT_DIR",
        "MAX_FILE_BYTES",
        "REQUEST_TIMEOUT_SECONDS",
        "OPENAI_MAX_RETRIES",
        "ALLOWED_EXTENSIONS",
    ):
        monkeypatch.delenv(key, raising=False)

    settings = load_settings()

    assert settings.provider == "ollama"
    assert settings.model_name == "llama3.2:3b"
    assert settings.base_url == "http://localhost:11434/v1"
    assert settings.workspace_root == str(workspace)
    assert settings.output_dir == str((workspace / "runs").resolve())
    assert ".md" in settings.allowed_extensions


def test_load_settings_requires_openai_key(monkeypatch, workspace_factory) -> None:
    workspace = workspace_factory()
    monkeypatch.chdir(workspace)
    monkeypatch.setattr(config_module, "load_dotenv", lambda: None)
    monkeypatch.setenv("MODEL_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
        load_settings()
