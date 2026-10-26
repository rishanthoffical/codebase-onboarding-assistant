from __future__ import annotations

import ast
import json
import logging
from app.agent.tool_registry import TOOL_DEFINITIONS, execute_tool


def test_tool_definitions_exist():
    names = [tool["name"] for tool in TOOL_DEFINITIONS]

    assert names == [
        "grep_repo",
        "read_file",
        "git_blame",
    ]


def test_read_file_through_registry():
    result = execute_tool(
        "read_file",
        {
            "path": "app/agent/loop.py",
            "line_start": 1,
            "line_end": 10,
        },
    )

    assert isinstance(result, str)
    assert "import logging" in result


def test_grep_repo_through_registry():
    result = execute_tool(
        "grep_repo",
        {
            "pattern": "def agent_loop",
            "path_glob": "app/**/*.py",
        },
    )

    assert isinstance(result, list)
    assert any(
        item["file_path"] == "app/agent/loop.py"
        for item in result
    )


def test_git_blame_through_registry():
    result = execute_tool(
        "git_blame",
        {
            "path": "app/agent/loop.py",
            "line": 1,
        },
    )

    assert isinstance(result, dict)


def test_path_traversal_is_blocked():
    try:
        execute_tool(
            "read_file",
            {
                "path": "../../.env",
                "line_start": 1,
                "line_end": 2,
            },
        )
    except ValueError as exc:
        assert "Path must remain inside the repository" in str(exc)
    else:
        raise AssertionError(
            "Path traversal should have been rejected"
        )


def test_unknown_tool_is_rejected():
    try:
        execute_tool(
            "unknown_tool",
            {},
        )
    except ValueError as exc:
        assert "Unknown tool" in str(exc)
    else:
        raise AssertionError(
            "Unknown tool should have been rejected"
        )
