from __future__ import annotations

from dataclasses import dataclass

from llm_agent_starter import agentic
from llm_agent_starter.tools import Tool


@dataclass
class FakeFunction:
    name: str
    arguments: str


@dataclass
class FakeToolCall:
    id: str
    function: FakeFunction


@dataclass
class FakeMessage:
    content: str | None
    tool_calls: list[FakeToolCall] | None = None


@dataclass
class FakeChoice:
    message: FakeMessage


@dataclass
class FakeResponse:
    choices: list[FakeChoice]


class FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)

    def create(self, **kwargs):
        del kwargs
        return self._responses.pop(0)


class FakeChat:
    def __init__(self, responses):
        self.completions = FakeCompletions(responses)


class FakeClient:
    def __init__(self, responses):
        self.chat = FakeChat(responses)


def test_run_agent_task_returns_final_answer_without_tool_calls(
    settings_factory, workspace_factory, monkeypatch
) -> None:
    workspace = workspace_factory()
    settings = settings_factory(workspace)
    client = FakeClient([FakeResponse([FakeChoice(FakeMessage(content="All done.", tool_calls=[]))])])

    monkeypatch.setattr(agentic, "build_tool_registry", lambda: {})

    result = agentic.run_agent_task(client, settings, "Say hello", max_steps=1)

    assert result.answer == "All done."
    assert result.steps == []


def test_run_agent_task_stops_at_max_steps(settings_factory, workspace_factory, monkeypatch) -> None:
    workspace = workspace_factory()
    settings = settings_factory(workspace)
    registry = {
        "fake_tool": Tool(
            name="fake_tool",
            description="Fake tool used in tests.",
            parameters={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
            handler=lambda settings, payload: "ok",
        )
    }
    responses = [
        FakeResponse(
            [
                FakeChoice(
                    FakeMessage(
                        content="",
                        tool_calls=[FakeToolCall(id="call-1", function=FakeFunction(name="fake_tool", arguments="{}"))],
                    )
                )
            ]
        ),
        FakeResponse(
            [
                FakeChoice(
                    FakeMessage(
                        content="",
                        tool_calls=[FakeToolCall(id="call-2", function=FakeFunction(name="fake_tool", arguments="{}"))],
                    )
                )
            ]
        ),
    ]
    client = FakeClient(responses)

    monkeypatch.setattr(agentic, "build_tool_registry", lambda: registry)

    result = agentic.run_agent_task(client, settings, "Loop forever", max_steps=2)

    assert "maximum number of steps" in result.answer.lower()
    assert len(result.steps) == 2
    assert result.steps[0]["tool"] == "fake_tool"
