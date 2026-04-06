from __future__ import annotations

from openai import OpenAI

from llm_agent_starter.config import Settings


def build_client(settings: Settings) -> OpenAI:
    return OpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url,
        max_retries=settings.max_retries,
        timeout=settings.request_timeout_seconds,
    )
