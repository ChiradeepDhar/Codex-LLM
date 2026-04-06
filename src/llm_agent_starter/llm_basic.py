from __future__ import annotations

from openai import OpenAI

from llm_agent_starter.config import Settings


def run_basic_prompt(client: OpenAI, settings: Settings, prompt: str) -> str:
    response = client.chat.completions.create(
        model=settings.model_name,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful AI tutor. Explain concepts clearly and keep answers practical."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )

    message = response.choices[0].message.content
    return message.strip() if message else "No response returned."
