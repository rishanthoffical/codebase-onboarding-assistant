from app.services.llm_service import (
    MAX_TOOL_OUTPUT_CHARS,
    OLLAMA_NUM_CTX,
    OllamaProvider,
    _sanitize_tool_output,
)


FAKE_SECRET = "AKIAIOSFODNN7EXAMPLE"


def test_tool_output_secret_is_removed_before_llm_context():
    raw_tool_output = (
        "Repository configuration:\n"
        f"AWS_ACCESS_KEY_ID={FAKE_SECRET}\n"
        "AWS_SECRET_ACCESS_KEY=EXAMPLE_SECRET_VALUE_123456789"
    )

    sanitized = _sanitize_tool_output(raw_tool_output)

    assert FAKE_SECRET not in sanitized
    assert "AKIA" not in sanitized
    assert "[REDACTED]" in sanitized


def test_normal_tool_output_reaches_llm_context_unchanged():
    raw_tool_output = (
        "def authenticate_user(username, password):\n"
        "    return find_user(username)"
    )

    sanitized = _sanitize_tool_output(raw_tool_output)

    assert sanitized == raw_tool_output


def test_insecure_tool_output_gets_security_warning():
    raw_tool_output = (
        'query = "SELECT * FROM users WHERE id=" + user_id'
    )

    sanitized = _sanitize_tool_output(raw_tool_output)

    assert "SECURITY WARNING" in sanitized


def test_ollama_tool_calls_bound_output_and_set_context(
    monkeypatch,
):
    class FakeResponse:
        status_code = 200
        text = ""

        def __init__(self, message):
            self.message = message

        def raise_for_status(self):
            return None

        def json(self):
            return {"message": self.message}

    tool_calls = [
        {
            "function": {
                "name": "read_file",
                "arguments": {},
            }
        }
        for _ in range(5)
    ]
    responses = [
        FakeResponse(
            {
                "content": "",
                "tool_calls": tool_calls,
            }
        ),
        FakeResponse(
            {
                "content": "Done.",
                "tool_calls": [],
            }
        ),
    ]
    requests = []

    def post(url, **kwargs):
        assert url.endswith("/api/chat")
        requests.append(kwargs["json"])
        return responses.pop(0)

    monkeypatch.setattr(
        "app.services.llm_service.requests.post",
        post,
    )

    provider = OllamaProvider()
    provider._supports_tools_cache = True
    executed_tools = []

    result = provider.generate_with_tools(
        question="Summarize this repository.",
        context="repository context",
        tools=[
            {
                "name": "read_file",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                },
            }
        ],
        tool_executor=lambda name, arguments: (
            executed_tools.append((name, arguments))
            or "x" * 10000
        ),
        max_iterations=2,
    )

    assert result["answer"] == "Done."
    assert len(executed_tools) == 3
    assert all(
        request["options"]["num_ctx"] == OLLAMA_NUM_CTX
        for request in requests
    )
    tool_output = [
        message["content"]
        for message in requests[1]["messages"]
        if message["role"] == "tool"
    ]
    assert sum(map(len, tool_output)) <= MAX_TOOL_OUTPUT_CHARS + 256
