import os

from app.services.llm_service import ClaudeProvider


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def test_claude_tool_calling_round_trip(monkeypatch):
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    os.environ["CLAUDE_MODEL"] = "test-model"

    responses = [
        FakeResponse(
            {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_test_123",
                        "name": "read_file",
                        "input": {
                            "path": "app/agent/loop.py",
                            "line_start": 1,
                            "line_end": 5,
                        },
                    }
                ]
            }
        ),
        FakeResponse(
            {
                "content": [
                    {
                        "type": "text",
                        "text": "The agent loop is implemented in app/agent/loop.py.",
                    }
                ]
            }
        ),
    ]

    captured_requests = []

    def fake_post(*args, **kwargs):
        captured_requests.append(kwargs)
        return responses.pop(0)

    monkeypatch.setattr(
        "app.services.llm_service.requests.post",
        fake_post,
    )

    executed_tools = []

    def fake_executor(tool_name, arguments):
        executed_tools.append(
            {
                "tool": tool_name,
                "arguments": arguments,
            }
        )

        return "import logging\nimport re\n\nfrom typing import List, Dict, Any, Optional"

    provider = ClaudeProvider()

    result = provider.generate_with_tools(
        question="Where is the agent loop implemented?",
        context="",
        tools=[
            {
                "name": "read_file",
                "description": "Read a repository file.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "line_start": {"type": "integer"},
                        "line_end": {"type": "integer"},
                    },
                    "required": ["path", "line_start"],
                },
            }
        ],
        tool_executor=fake_executor,
    )

    assert result["answer"] == (
        "The agent loop is implemented in app/agent/loop.py."
    )

    assert len(result["tool_calls"]) == 1

    assert result["tool_calls"][0]["tool"] == "read_file"

    assert executed_tools == [
        {
            "tool": "read_file",
            "arguments": {
                "path": "app/agent/loop.py",
                "line_start": 1,
                "line_end": 5,
            },
        }
    ]

    assert len(captured_requests) == 2

    first_messages = captured_requests[0]["json"]["messages"]

    assert first_messages[0]["role"] == "user"

    second_messages = captured_requests[1]["json"]["messages"]

    assert second_messages[0]["role"] == "user"
    assert second_messages[1]["role"] == "assistant"
    assert second_messages[2]["role"] == "user"

    tool_result = second_messages[2]["content"][0]

    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_use_id"] == "toolu_test_123"
    assert "import logging" in tool_result["content"]


def test_claude_tool_calling_handles_tool_error(monkeypatch):
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    os.environ["CLAUDE_MODEL"] = "test-model"

    responses = [
        FakeResponse(
            {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_error_123",
                        "name": "read_file",
                        "input": {
                            "path": "../../.env",
                            "line_start": 1,
                            "line_end": 2,
                        },
                    }
                ]
            }
        ),
        FakeResponse(
            {
                "content": [
                    {
                        "type": "text",
                        "text": "The requested file could not be accessed.",
                    }
                ]
            }
        ),
    ]

    def fake_post(*args, **kwargs):
        return responses.pop(0)

    monkeypatch.setattr(
        "app.services.llm_service.requests.post",
        fake_post,
    )

    def failing_executor(tool_name, arguments):
        raise ValueError(
            "Path must remain inside the repository."
        )

    provider = ClaudeProvider()

    result = provider.generate_with_tools(
        question="Read the requested file.",
        context="",
        tools=[],
        tool_executor=failing_executor,
    )

    assert result["answer"] == (
        "The requested file could not be accessed."
    )

    assert result["tool_calls"] == []
