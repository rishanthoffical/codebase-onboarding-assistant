from __future__ import annotations

from typing import Any, Dict

from app.agent.tools import (
    REPO_ROOT,
    grep_repo,
    read_file,
    git_blame,
)


TOOL_DEFINITIONS = [
    {
        "name": "grep_repo",
        "description": (
            "Search repository source files for a text or regular "
            "expression pattern. Use this to locate functions, "
            "classes, implementations, configuration, or references."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": (
                        "Text or regular expression to search for."
                    ),
                },
                "path_glob": {
                    "type": "string",
                    "description": (
                        "Optional repository-relative glob such as "
                        "'app/**/*.py'."
                    ),
                    "default": "**/*",
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "read_file",
        "description": (
            "Read a specific line range from a repository file. "
            "Paths must be repository-relative."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Repository-relative file path.",
                },
                "line_start": {
                    "type": "integer",
                    "description": "First line to read, starting at 1.",
                    "minimum": 1,
                },
                "line_end": {
                    "type": "integer",
                    "description": "Last line to read, inclusive.",
                    "minimum": 1,
                },
            },
            "required": [
                "path",
                "line_start",
            ],
        },
    },
    {
        "name": "git_blame",
        "description": (
            "Get Git blame information for one specific line "
            "of a repository file."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Repository-relative file path.",
                },
                "line": {
                    "type": "integer",
                    "description": "Line number.",
                    "minimum": 1,
                },
            },
            "required": [
                "path",
                "line",
            ],
        },
    },
]


def execute_tool(
    tool_name: str,
    arguments: Dict[str, Any],
    repo_root: str = str(REPO_ROOT),
) -> Any:

    if tool_name == "grep_repo":
        return grep_repo(
            pattern=arguments["pattern"],
            path_glob=arguments.get(
                "path_glob",
                "**/*",
            ),
            repo_root=repo_root,
        )

    if tool_name == "read_file":
        return read_file(
            path=arguments["path"],
            line_start=int(
                arguments.get(
                    "line_start",
                    1,
                )
            ),
            line_end=(
                int(arguments["line_end"])
                if arguments.get("line_end") is not None
                else None
            ),
            repo_root=repo_root,
        )

    if tool_name == "git_blame":
        return git_blame(
            path=arguments["path"],
            line=int(arguments["line"]),
            repo_root=repo_root,
        )

    raise ValueError(
        f"Unknown tool: {tool_name}"
    )
