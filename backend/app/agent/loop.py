from __future__ import annotations

import asyncio
import ast
import json
import logging
import re

from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, Iterator, List, Optional, Tuple

from app.agent.tools import git_blame, grep_repo, read_file
from app.agent.tool_registry import TOOL_DEFINITIONS, execute_tool
from app.guardrails.pipeline import (
    guardrail_pipeline,
    scan_for_secrets,
    scope_guardrail,
)
from app.indexer.chunker import language_for_path
from app.models.chat import ChatResponse
from app.services.chat_cache import ChatResponseCache
from app.services.llm_service import LLMService
from app.services.project_scanner import build_project_overview
from app.services.vector_store import VectorStore


logger = logging.getLogger(__name__)

chat_response_cache = ChatResponseCache()

SESSION_MEMORY: Dict[str, List[Dict[str, Any]]] = {}

MAX_RETRIEVAL_DISTANCE = 1.5
OVERVIEW_RETRIEVAL_DISTANCE = 2.0


# ============================================================
# QUESTION HELPERS
# ============================================================

def is_project_overview_question(question: str) -> bool:
    question_lower = question.lower()

    matches_known_phrase = any(
        phrase in question_lower
        for phrase in (
            "overview",
            "what does this project do",
            "what does this codebase do",
            "what does this repository do",
            "what does this repo do",
            "what is this project",
            "what is this repository",
            "what is the purpose of this project",
            "what is the purpose of this codebase",
            "explain its architecture",
            "explain the architecture",
            "important files",
            "main execution flow",
            "entry points",
            "about this codebase",
            "about this repository",
            "summarize this project",
            "summarize the project",
            "summarize this codebase",
            "summarize the codebase",
            "summarize this repository",
            "summarize the repository",
        )
    )

    summarizes_project = re.search(
        r"\b(?:summarize|describe)\b"
        r"(?:\W+\w+){0,4}\W+"
        r"(?:project|codebase|repository|repo)\b",
        question_lower,
    )

    return (
        matches_known_phrase
        or summarizes_project is not None
    )


def _extract_file_references(question: str) -> List[str]:
    """
    Extract repository-relative file references.

    Examples:
        tool_registry.py
        agent/tools.py
        app/agent/loop.py
        ./agent/tools.py
        agent\\tools.py

    Absolute paths and parent-directory traversal are rejected.
    """

    pattern = (
        r"(?<![\w])"
        r"(?:\.[/\\])?"
        r"(?:[\w.@+-]+[/\\])*"
        r"[\w@+-]+\.[A-Za-z0-9]{1,12}"
        r"(?![\w])"
    )

    references: List[str] = []

    for match in re.findall(pattern, question):
        normalized = match.replace("\\", "/")

        # Ignore dotted Python symbols such as OllamaProvider.generate.
        # Preserve real filenames, including CamelCase.py.
        if "/" not in normalized and "." in normalized:
            prefix, suffix = normalized.rsplit(".", 1)
            known_extensions = {
                "py", "pyi", "js", "jsx", "ts", "tsx", "mjs", "cjs",
                "json", "yaml", "yml", "toml", "ini", "cfg", "conf",
                "md", "txt", "rst", "html", "css", "scss", "sql", "sh",
                "ps1", "bat", "cmd", "env", "xml", "csv", "ipynb",
                "java", "kt", "go", "rs", "c", "h", "cpp", "hpp",
            }
            if (
                suffix.casefold() not in known_extensions
                and re.fullmatch(r"[A-Z][A-Za-z0-9]*", prefix)
                and re.search(r"[a-z][A-Z]", prefix)
            ):
                continue

        while normalized.startswith("./"):
            normalized = normalized[2:]

        relative_path = PurePosixPath(normalized)
        windows_path = PureWindowsPath(match)

        if (
            relative_path.is_absolute()
            or windows_path.drive
            or ".." in relative_path.parts
        ):
            continue

        normalized = relative_path.as_posix()

        if normalized not in references:
            references.append(normalized)

    return references


FOLLOW_UP_PHRASES = [
    "that function",
    "this function",
    "that method",
    "this method",
    "that file",
    "this file",
    "what does it do",
    "explain it",
    "explain that",
    "explain this",
    "tell me more about it",
    "the above",
    "the previous answer",
    "the previous function",
    "the same function",
    "the same method",
    "the same file",
]


def is_follow_up_question(question: str) -> bool:
    question_lower = question.strip().lower()

    return any(
        re.search(
            rf"\b{re.escape(phrase)}\b",
            question_lower,
        )
        is not None
        for phrase in FOLLOW_UP_PHRASES
    )


def build_contextual_question(
    question: str,
    session_id: Optional[str],
) -> str:
    if not session_id:
        return question

    history = SESSION_MEMORY.get(session_id, [])

    if not history or not is_follow_up_question(question):
        return question

    previous = history[-1]

    return (
        f"Previous developer question: "
        f"{previous.get('question', '')}\n\n"
        f"Previous assistant answer: "
        f"{previous.get('answer', '')}\n\n"
        f"Current developer follow-up: "
        f"{question}"
    )


def _save_session(
    session_id: Optional[str],
    question: str,
    answer: str,
    citations: List[str],
) -> None:
    if not session_id:
        return

    SESSION_MEMORY.setdefault(
        session_id,
        [],
    ).append(
        {
            "question": question,
            "answer": answer,
            "citations": citations,
        }
    )

    SESSION_MEMORY[session_id] = (
        SESSION_MEMORY[session_id][-10:]
    )


# ============================================================
# SERVICES
# ============================================================

vector_store = VectorStore()
llm_service = LLMService()


# ============================================================
# PATH HELPERS
# ============================================================

IGNORED_PATH_PARTS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    "chroma_db",
    "dist",
    "build",
    "coverage",
}

IGNORED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".pdf",
    ".zip",
    ".exe",
    ".dll",
    ".pyc",
    ".class",
}


def _normalize_path(path: str) -> str:
    normalized = str(path).replace("\\", "/").strip()

    while normalized.startswith("./"):
        normalized = normalized[2:]

    normalized = normalized.lstrip("/")

    return normalized


def _is_relevant_source_file(path: str) -> bool:
    normalized = _normalize_path(path)

    if not normalized:
        return False

    parts = normalized.split("/")

    if any(
        part in IGNORED_PATH_PARTS
        for part in parts
    ):
        return False

    filename = parts[-1]

    if "." not in filename:
        return True

    extension = (
        "." + filename.rsplit(".", 1)[-1].lower()
    )

    return extension not in IGNORED_EXTENSIONS

def _path_matches_reference(
    indexed_path: str,
    reference: str,
) -> bool:
    """
    Match an indexed repository path against a user reference.

    Supports:
        agent/tool_registry.py
        tool_registry.py
        backend/app/agent/tool_registry.py

    The requested path may match the suffix of the indexed path.
    """

    candidate = _normalize_path(
        indexed_path
    ).casefold()

    requested = _normalize_path(
        reference
    ).casefold()

    if not candidate or not requested:
        return False

    # Exact match.
    if candidate == requested:
        return True

    requested_path = PurePosixPath(
        requested
    )

    candidate_path = PurePosixPath(
        candidate
    )

    # If the user supplied only the filename,
    # match by basename.
    if len(requested_path.parts) == 1:
        return (
            candidate_path.name.casefold()
            == requested_path.name.casefold()
        )

    # Allow a user path to match the suffix of the
    # repository-indexed path.
    #
    # Example:
    # backend/app/agent/tool_registry.py
    #                 agent/tool_registry.py
    candidate_parts = candidate_path.parts
    requested_parts = requested_path.parts

    if len(requested_parts) <= len(candidate_parts):
        return candidate_parts[
            -len(requested_parts):
        ] == requested_parts

    return False
def _metadata_file_path(
    metadata: Dict[str, Any],
) -> str:
    return _normalize_path(
        str(
            metadata.get(
                "file_path",
                metadata.get(
                    "source",
                    metadata.get(
                        "file",
                        "",
                    ),
                ),
            )
        )
    )


# ============================================================
# KNOWN FUNCTIONS
# ============================================================

KNOWN_FUNCTIONS = [
    "scan_for_secrets",
    "scan_for_insecure_patterns",
    "scope_guardrail",
    "guardrail_pipeline",
    "grep_repo",
    "read_file",
    "git_blame",
    "agent_loop",
    "agent_stream",
    "build_context",
    "build_contextual_question",
    "find_function_sources",
    "read_function_implementation",
    "find_source_line_range",
    "build_function_verification_context",
    "extract_citations",
    "prioritize_citations",
    "verify_repository_sources",
    "detect_tool_calls",
    "retrieve_information",
    "incremental_index",
    "get_incremental_changes",
]


def extract_function_names(
    question: str,
) -> List[str]:
    question_lower = question.lower()

    found: List[str] = []

    for function_name in KNOWN_FUNCTIONS:
        if re.search(
            rf"\b{re.escape(function_name.lower())}\b",
            question_lower,
        ):
            found.append(function_name)

    return found


def get_target_functions(
    question: str,
) -> List[str]:
    functions = extract_function_names(question)

    if functions:
        return functions

    question_lower = question.lower()

    aliases = {
        "secret scanning guardrail": "scan_for_secrets",
        "secret scanning": "scan_for_secrets",
        "secret scanner": "scan_for_secrets",
        "secrets scanner": "scan_for_secrets",
        "insecure pattern scanner": (
            "scan_for_insecure_patterns"
        ),
        "insecure pattern guardrail": (
            "scan_for_insecure_patterns"
        ),
        "scope guardrail": "scope_guardrail",
        "guardrail pipeline": "guardrail_pipeline",
        "guardrail function": "guardrail_pipeline",
        "vector search": "retrieve_information",
    }

    for phrase, function_name in aliases.items():
        if phrase in question_lower:
            return [function_name]

    return []


FUNCTION_SOURCE_MAP = {
    "scan_for_secrets": "app/guardrails/pipeline.py",
    "scan_for_insecure_patterns": (
        "app/guardrails/insecure_patterns.py"
    ),
    "scope_guardrail": "app/guardrails/pipeline.py",
    "guardrail_pipeline": "app/guardrails/pipeline.py",
    "grep_repo": "app/agent/tools.py",
    "read_file": "app/agent/tools.py",
    "git_blame": "app/agent/tools.py",
    "agent_loop": "app/agent/loop.py",
    "agent_stream": "app/agent/loop.py",
    "build_context": "app/agent/loop.py",
    "build_contextual_question": "app/agent/loop.py",
    "find_function_sources": "app/agent/loop.py",
    "read_function_implementation": (
        "app/agent/loop.py"
    ),
    "find_source_line_range": (
        "app/agent/loop.py"
    ),
    "build_function_verification_context": (
        "app/agent/loop.py"
    ),
    "extract_citations": "app/agent/loop.py",
    "prioritize_citations": "app/agent/loop.py",
    "verify_repository_sources": (
        "app/agent/loop.py"
    ),
    "detect_tool_calls": "app/agent/loop.py",
    "retrieve_information": "app/agent/loop.py",
    "incremental_index": "app/indexer/indexer.py",
    "get_incremental_changes": "app/indexer/diff.py",
}


def find_function_sources(
    function_names: List[str],
    repo_id: str = "local",
) -> List[str]:
    if repo_id != "local":
        return []

    sources: List[str] = []

    for function_name in function_names:
        source = FUNCTION_SOURCE_MAP.get(
            function_name
        )

        if source and source not in sources:
            sources.append(source)

    return sources


# ============================================================
# FUNCTION QUESTIONS
# ============================================================

def is_function_location_question(
    question: str,
) -> bool:
    question_lower = question.strip().lower()

    location_phrases = [
        "where is",
        "which file",
        "where does",
        "implemented",
        "implementation",
        "located",
        "location",
    ]

    return bool(
        get_target_functions(question)
        and any(
            phrase in question_lower
            for phrase in location_phrases
        )
    )


def is_function_explanation_question(
    question: str,
) -> bool:
    question_lower = question.lower().strip()

    explanation_phrases = [
        "what does",
        "what do",
        "what is the purpose of",
        "what is the function of",
        "explain",
        "describe",
        "how does",
        "how do",
        "what does it do",
        "what does that do",
        "what does this do",
        "what does that function do",
        "what does this function do",
        "what does that method do",
        "what does this method do",
        "explain that function",
        "explain this function",
        "explain that method",
        "explain this method",
    ]

    return any(
        phrase in question_lower
        for phrase in explanation_phrases
    )


def is_ollama_generation_request_question(
    question: str,
) -> bool:
    question_lower = question.lower()

    # INSERT THE NEW CODE HERE

    names_provider_method = bool(
        re.search(
            r"\bollamaprovider\s*\.\s*"
            r"(?:generate_with_tools|generate)\b",
            question_lower,
        )
    )

    names_provider_method = bool(
        re.search(
            r"\bollamaprovider\s*\.\s*"
            r"(?:generate_with_tools|generate)\b",
            question_lower,
        )
    )
    asks_about_method = bool(
        re.search(
            r"\b(?:definition|defined|implementation|source|"
            r"method|function|code)\b",
            question_lower,
        )
    )

    if names_provider_method and asks_about_method:
        return True

    asks_about_request = bool(
        re.search(
            r"\b(?:request|requests|post|send|sends|call|calls|"
            r"invoke|invokes)\b",
            question_lower,
        )
    )
    names_ollama_endpoint = bool(
        re.search(
            r"/api/(?:generate|chat)\b",
            question_lower,
        )
    )

    if asks_about_request and names_ollama_endpoint:
        return True

    return bool(
        "ollama" in question_lower
        and asks_about_request
        and re.search(
            r"\b(?:generate|generation|answer|response)\b",
            question_lower,
        )
    )


def is_docker_backend_startup_question(question: str) -> bool:
    question_lower = question.lower()
    asks_how_started = bool(re.search(
        r"\b(?:start|starts|started|run|runs|launch|entry\s*point|command)\b",
        question_lower,
    ))
    asks_about_backend = bool(re.search(
        r"\b(?:fastapi|backend)\b",
        question_lower,
    ))
    asks_about_docker = bool(re.search(
        r"\b(?:docker|compose)\b",
        question_lower,
    ))
    return asks_how_started and asks_about_backend and asks_about_docker


def build_docker_backend_startup_answer(
    repository_root: str,
) -> Optional[Tuple[str, List[str]]]:
    """Trace the best-supported Compose backend command to its Python app."""
    root = Path(repository_root).resolve()
    compose_names = ("docker-compose.yml", "compose.yaml", "compose.yml")
    ignored_directories = {
        ".git", "node_modules", "__pycache__", "venv", ".venv",
        "uploaded_repositories", "chroma_db", "chroma_cache",
    }
    compose_candidates: List[Path] = []
    pending_directories = [(root, 0)]
    while pending_directories:
        parent, depth = pending_directories.pop(0)
        for name in compose_names:
            candidate = parent / name
            if candidate.is_file():
                compose_candidates.append(candidate)
        if depth >= 4:
            continue
        try:
            children = list(parent.iterdir())
        except OSError:
            continue
        pending_directories.extend(
            (child, depth + 1)
            for child in children
            if child.is_dir()
            and child.name.casefold() not in ignored_directories
            and "backup" not in child.name.casefold()
        )
    compose_path = min(
        compose_candidates,
        key=lambda path: (len(path.relative_to(root).parts), path.as_posix()),
        default=None,
    )
    if compose_path is None:
        return None

    try:
        compose_lines = compose_path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
    except OSError:
        return None

    service_start = next((
        index for index, line in enumerate(compose_lines)
        if re.match(r"^  backend:\s*$", line)
    ), None)
    if service_start is None:
        return None

    service_end = next((
        index for index in range(service_start + 1, len(compose_lines))
        if re.match(r"^  [\w-]+:\s*$", compose_lines[index])
    ), len(compose_lines))
    service_lines = compose_lines[service_start:service_end]

    context_match = next((
        (offset, match.group(1).strip().strip("\"'"))
        for offset, line in enumerate(service_lines)
        if (match := re.match(r"^\s{6}context:\s*(.+?)\s*$", line))
    ), None)
    dockerfile_match = next((
        (offset, match.group(1).strip().strip("\"'"))
        for offset, line in enumerate(service_lines)
        if (match := re.match(r"^\s{6}dockerfile:\s*(.+?)\s*$", line))
    ), None)
    command_start = next((
        offset for offset, line in enumerate(service_lines)
        if re.match(r"^\s{4}command:\s*", line)
    ), None)
    if context_match is None or command_start is None:
        return None

    command_lines = [service_lines[command_start]]
    for line in service_lines[command_start + 1:]:
        if not line.strip() or len(line) - len(line.lstrip()) <= 4:
            break
        command_lines.append(line)
    command_text = " ".join(command_lines)
    command_match = re.search(
        r"\buvicorn\s+([A-Za-z_][\w.]*)\s*:\s*([A-Za-z_]\w*)\b",
        command_text,
    )
    if command_match is None:
        return None

    context = context_match[1]
    dockerfile_name = (
        dockerfile_match[1]
        if dockerfile_match is not None
        else "Dockerfile"
    )
    module_name, app_name = command_match.groups()
    entry_relative_path = module_name.replace(".", "/") + ".py"
    candidate_roots = [compose_path.parent, *compose_path.parent.parents]
    candidate_roots = [
        candidate_root for candidate_root in candidate_roots
        if candidate_root == root or root in candidate_root.parents
    ]
    project_root: Optional[Path] = None
    entry_path: Optional[Path] = None
    app_line: Optional[int] = None
    for candidate_root in candidate_roots:
        candidate_entry = (
            candidate_root / context / entry_relative_path
        ).resolve()
        try:
            candidate_entry.relative_to(root)
            candidate_lines = candidate_entry.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        except (OSError, ValueError):
            continue
        candidate_app_line = next((
            index for index, line in enumerate(candidate_lines, start=1)
            if re.search(
                rf"^\s*{re.escape(app_name)}\s*=\s*FastAPI\s*\(",
                line,
            )
        ), None)
        if candidate_app_line is not None:
            project_root = candidate_root
            entry_path = candidate_entry
            app_line = candidate_app_line
            break

    if project_root is None or entry_path is None or app_line is None:
        return None

    compose_relative = compose_path.relative_to(root).as_posix()
    entry_relative = entry_path.relative_to(root).as_posix()
    context_line = service_start + context_match[0] + 1
    dockerfile_citation = ""
    citations = [
        f"{compose_relative}:{context_line}-{context_line}",
        f"{compose_relative}:{service_start + command_start + 1}-"
        f"{service_start + command_start + len(command_lines)}",
    ]
    if dockerfile_match is not None:
        dockerfile_config_line = service_start + dockerfile_match[0] + 1
        citations.append(
            f"{compose_relative}:{dockerfile_config_line}-"
            f"{dockerfile_config_line}"
        )
        dockerfile_path = (
            project_root / context / dockerfile_match[1]
        ).resolve()
        try:
            dockerfile_path.relative_to(root)
        except ValueError:
            return None
        if dockerfile_path.is_file():
            dockerfile_relative = dockerfile_path.relative_to(root).as_posix()
            dockerfile_lines = dockerfile_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
            dockerfile_cmd = next((
                index for index, line in enumerate(dockerfile_lines, start=1)
                if re.match(r"^\s*CMD\s+", line)
            ), None)
            if dockerfile_cmd is not None:
                dockerfile_citation = (
                    f" The image also defines a default command in "
                    f"`{dockerfile_relative}`, but Compose's `command` "
                    "above is the command used for this service."
                )
                citations.append(
                    f"{dockerfile_relative}:{dockerfile_cmd}-{dockerfile_cmd}"
                )

    citations.append(f"{entry_relative}:{app_line}-{app_line}")
    compose_scope = (
        "The root Compose file"
        if compose_path.parent == project_root
        else "The shallowest Compose file present in the upload"
    )
    answer = (
        f"{compose_scope} `{compose_relative}` defines the "
        f"`backend` service. It builds using `{context}/{dockerfile_name}` "
        f"and runs `uvicorn {module_name}:{app_name}` from its Compose "
        f"`command`. That module maps to `{entry_relative}`, where "
        f"`{app_name}` is initialized as "
        f"the FastAPI app.{dockerfile_citation} I verified the service "
        "against its referenced entry file. If this is only a nested "
        "Compose copy, the upload does not prove it is the runtime config; "
        "deeper nested and backup copies were not used."
    )
    return answer, citations


def context_shows_ollama_generation_call(
    context: str,
) -> bool:
    return bool(
        re.search(
            r"/api/(?:generate|chat)\b"
            r"|\bollama\s*\.\s*(?:generate|chat)\s*\(",
            context,
            re.IGNORECASE,
        )
    )


def find_ollama_request_sources(
    repository_root: str,
) -> List[Dict[str, Any]]:
    """Find Python methods that directly POST to Ollama endpoints.

    Searching the selected repository source avoids asking the model to
    infer which method owns an API call from semantically similar chunks.
    """
    root = Path(repository_root).resolve()
    matches = grep_repo(
        r"/api/(?:generate|chat)\b",
        path_glob="**/*.py",
        repo_root=root,
    )

    matched_files = sorted({
        str(match.get("file_path", ""))
        for match in matches
        if match.get("file_path")
    })
    results: List[Dict[str, Any]] = []

    for relative_path in matched_files:
        source_path = (root / relative_path).resolve()
        try:
            source_path.relative_to(root)
            source = source_path.read_text(
                encoding="utf-8",
                errors="replace",
            )
            tree = ast.parse(source)
        except (OSError, RuntimeError, SyntaxError, ValueError):
            continue

        def visit(node: ast.AST, class_name: Optional[str] = None) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.ClassDef):
                    visit(child, child.name)
                    continue

                if not isinstance(
                    child,
                    (ast.FunctionDef, ast.AsyncFunctionDef),
                ):
                    visit(child, class_name)
                    continue

                function_source = ast.get_source_segment(source, child) or ""
                for endpoint in ("generate", "chat"):
                    endpoint_path = f"/api/{endpoint}"
                    if endpoint_path not in function_source:
                        continue

                    for call in ast.walk(child):
                        if not isinstance(call, ast.Call):
                            continue
                        if not isinstance(call.func, ast.Attribute):
                            continue
                        if call.func.attr != "post":
                            continue

                        call_source = ast.get_source_segment(source, call) or ""
                        if endpoint_path not in call_source:
                            continue

                        results.append({
                            "file_path": relative_path,
                            "class_name": class_name,
                            "function_name": child.name,
                            "function_line": int(child.lineno),
                            "request_start": int(call.lineno),
                            "request_end": int(
                                getattr(call, "end_lineno", call.lineno)
                            ),
                            "endpoint": endpoint_path,
                        })

        visit(tree)

    # Uploaded archives can contain nested backup copies. For the same
    # method/endpoint, prefer the least-nested path and avoid duplicate claims.
    results.sort(key=lambda item: (
        item["function_name"],
        item["endpoint"],
        item["file_path"].count("/"),
        item["file_path"],
    ))
    unique: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for result in results:
        unique.setdefault(
            (result["function_name"], result["endpoint"]),
            result,
        )
    return list(unique.values())



def build_ollama_request_answer(
    repository_root: str,
    question: Optional[str] = None,
) -> Optional[Tuple[str, List[str]]]:
    """Build a source-grounded answer for Ollama request questions."""
    root = Path(repository_root).resolve()
    sources = find_ollama_request_sources(repository_root)
    if not sources:
        return None

    # Combine results only when both methods are positively requested.
    if question:
        asks_both = (
            re.search(r"(?i)\bgenerate_with_tools\b", question)
            and re.search(r"(?i)\bgenerate\b", question)
            and not re.search(
                r"(?i)\b(?:do not use|don't use|exclude|except|not)\s+"
                r"(?:`?OllamaProvider\.)?`?generate_with_tools\b",
                question,
            )
        )
        if asks_both:
            combined_answers = []
            combined_citations = []
            for method_name in ("generate", "generate_with_tools"):
                result = build_ollama_request_answer(
                    repository_root,
                    f"OllamaProvider.{method_name} in services/llm_service.py",
                )
                if result:
                    combined_answers.append(result[0])
                    combined_citations.extend(result[1])
            if combined_answers:
                return (
                    " ".join(combined_answers),
                    list(dict.fromkeys(combined_citations)),
                )

    # Identify the exact provider method requested.
    named_method = None
    if question:
        match = re.search(
            r"(?i)\b(?:([A-Za-z_]\w*)\s*\.\s*)?"
            r"(generate_with_tools|generate)\b",
            question,
        )
        if match:
            named_method = (match.group(1), match.group(2))

    if named_method:
        requested_class, requested_function = named_method

        candidates = [
            item for item in sources
            if item["function_name"] == requested_function
            and (
                not requested_class
                or item["class_name"].casefold()
                == requested_class.casefold()
            )
        ]

        # Prefer the path explicitly named by the user.
        requested_path_match = re.search(
            r"(?i)(?:^|[\\/])((?:services|app[\\/]services)"
            r"[\\/]llm_service\.py)\b",
            question or "",
        )
        if requested_path_match and candidates:
            requested_path = requested_path_match.group(1).replace("\\", "/")
            matching_path = [
                item for item in candidates
                if item["file_path"].replace("\\", "/").endswith(
                    requested_path
                )
            ]
            if matching_path:
                candidates = matching_path

        # Prefer the least-nested copy in an archive containing duplicates.
        candidates.sort(
            key=lambda item: (
                item["file_path"].count("/"),
                item["file_path"],
            )
        )

        for source_info in candidates:
            file_path = source_info["file_path"]
            source_path = (root / file_path).resolve()

            try:
                source_path.relative_to(root)
                source_text = source_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
                tree = ast.parse(source_text)
            except (OSError, RuntimeError, SyntaxError, ValueError):
                continue

            target = None

            # Find the exact method directly inside the requested class.
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue

                if (
                    requested_class
                    and node.name.casefold() != requested_class.casefold()
                ):
                    continue

                for child in node.body:
                    if (
                        isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and child.name == requested_function
                    ):
                        target = child
                        break

                if target is not None:
                    break

            if target is None:
                continue

            endpoint = source_info["endpoint"]

            # Convert the archive path to a readable citation path.
            normalized_path = file_path.replace("\\", "/")
            if normalized_path.endswith("app/services/llm_service.py"):
                logical_path = "app/services/llm_service.py"
            elif normalized_path.endswith("services/llm_service.py"):
                logical_path = "services/llm_service.py"
            else:
                logical_path = Path(file_path).name

            summary = [
                f"`{source_info['class_name']}.{requested_function}()` "
                f"is defined in `{logical_path}`.",
                f"It sends an HTTP POST request to `{endpoint}`.",
            ]

            if requested_function == "generate":
                summary.extend([
                    "The method builds a JSON payload for normal text "
                    "generation and sends it to Ollama.",
                    "It checks the HTTP response, parses the JSON result, "
                    "extracts and strips the generated response text, and "
                    "returns a fallback message if the result is empty.",
                ])
            else:
                summary.append(
                    "See the cited method implementation for its tool-calling "
                    "and fallback behavior."
                )

            # Return immediately; do not append citations for other methods.
            return (
                " ".join(summary),
                [f"{logical_path}:{target.lineno}-{target.end_lineno}"],
            )

        return None

    # No exact method was named: summarize the discovered endpoints.
    descriptions = []
    citations = []
    endpoint_labels = {
        "/api/generate": "normal text generation",
        "/api/chat": "tool-calling chat generation",
    }

    for source in sources:
        method = source["function_name"]
        class_name = source["class_name"]
        qualified_name = (
            f"{class_name}.{method}" if class_name else method
        )
        file_path = source["file_path"]
        endpoint = source["endpoint"]
        endpoint_label = endpoint_labels.get(endpoint, endpoint)

        if (
            method == "generate_with_tools"
            and endpoint == "/api/generate"
        ):
            endpoint_label = "fallback normal text generation"

        descriptions.append(
            f"`{qualified_name}()` sends the {endpoint_label} "
            f"request to Ollama at `{endpoint}`."
        )
        citations.extend((
            f"{file_path}:{source['function_line']}-"
            f"{source['function_line']}",
            f"{file_path}:{source['request_start']}-"
            f"{source['request_end']}",
        ))

    return " ".join(descriptions), citations


   
def read_function_implementation(
    source_file: str,
    function_name: str,
) -> str:
    """Read a function implementation from a source file."""
    try:
        pattern = (
            rf"^\s*(?:async\s+)?def\s+"
            rf"{re.escape(function_name)}\s*\("
        )

        matches = grep_repo(
            pattern,
            source_file,
        )

        if not matches:
            return ""

        start_line = matches[0].get(
            "line",
            matches[0].get(
                "line_number",
                1,
            ),
        )

        try:
            start_line = int(start_line)
        except (TypeError, ValueError):
            start_line = 1

        return read_file(
            source_file,
            start_line,
            start_line + 80,
        )

    except (OSError, ValueError, TypeError):
        return ""


def find_source_line_range(
    source_file: str,
    function_name: Optional[str] = None,
) -> Optional[Tuple[int, int]]:
    normalized = _normalize_path(
        source_file
    )

    if not normalized:
        return None

    try:
        path = Path(normalized)

        if (
            path.is_absolute()
            or not path.exists()
            or not path.is_file()
        ):
            return None

        source = path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        if not source.strip():
            return None

        if (
            path.suffix.lower() == ".py"
            and function_name
        ):
            tree = ast.parse(source)

            for node in ast.walk(tree):
                if isinstance(
                    node,
                    (
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                    ),
                ) and node.name == function_name:
                    start_line = int(
                        getattr(
                            node,
                            "lineno",
                            0,
                        )
                    )

                    end_line = int(
                        getattr(
                            node,
                            "end_lineno",
                            start_line,
                        )
                    )

                    if (
                        start_line >= 1
                        and end_line >= start_line
                    ):
                        return (
                            start_line,
                            end_line,
                        )

        lines = source.splitlines()

        if lines:
            return 1, len(lines)

        return None

    except (
        OSError,
        UnicodeError,
        SyntaxError,
        ValueError,
        TypeError,
    ):
        return None


def build_function_verification_context(
    question: str,
    function_sources: List[str],
) -> str:
    context_parts: List[str] = []

    target_functions = get_target_functions(
        question
    )

    if not target_functions:
        return ""

    for source_file in function_sources:
        for function_name in target_functions:
            implementation = (
                read_function_implementation(
                    source_file,
                    function_name,
                )
            )

            if not implementation:
                continue

            source_range = find_source_line_range(
                source_file,
                function_name,
            )

            range_text = ""

            if source_range:
                range_text = (
                    f":{source_range[0]}-"
                    f"{source_range[1]}"
                )

            context_parts.append(
                f"FILE: {source_file}{range_text}\n"
                f"{implementation}"
            )

            break

    return "\n\n".join(context_parts)


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_information(
    question: str,
    repo_id: str = "local",
    repository_root: Optional[str] = None,
    overview_request: Optional[bool] = None,
) -> Dict[str, Any]:

    file_references = _extract_file_references(
        question
    )
    is_file_inventory = bool(
        re.search(
            r"\b(?:what|which)\s+files?\b"
            r"|\bfiles?\s+(?:are|exist|available)\b"
            r"|\blist\s+(?:all\s+)?files?\b"
            r"|\bfile\s+list\b"
            r"|\b(?:show|list|find)\s+(?:all\s+)?repository\s+files?\b"
            r"|\b(?:show|list|find)\s+(?:all\s+)?codebase\s+files?\b",
            question.lower(),
        )
    )

    result: Dict[str, Any] = {
        "documents": [],
        "metadatas": [],
        "tool_calls": [],
        "is_file_inventory": is_file_inventory,
        "requested_file_references": file_references,
        "resolved_file_references": [],
        "missing_file_references": [],
        "file_reference_lookup_complete": (
            not file_references
        ),
    }

    normal_entries: List[
        Tuple[str, Dict[str, Any]]
    ] = []

    explicit_entries: List[
        Tuple[str, Dict[str, Any]]
    ] = []

    try:
        is_overview = (
            overview_request
            if overview_request is not None
            else is_project_overview_question(question)
        )

        search_query = question
        max_distance = MAX_RETRIEVAL_DISTANCE

        if is_overview:
            search_query = (
                f"{question} "
                "README purpose languages frameworks "
                "entry points important files dependencies "
                "execution flow"
            )

            max_distance = (
                OVERVIEW_RETRIEVAL_DISTANCE
            )

        search_result = vector_store.search(
            search_query,
            n_results=10 if is_overview else 5,
            repo_id=repo_id,
        )

        documents = search_result.get(
            "documents",
            [[]],
        )

        metadatas = search_result.get(
            "metadatas",
            [[]],
        )

        distances = search_result.get(
            "distances",
            [[]],
        )

        if documents:
            result_documents = list(
                documents[0]
            )

            result_metadatas = (
                list(metadatas[0])
                if metadatas
                else [{}] * len(result_documents)
            )

            result_distances = (
                list(distances[0])
                if distances
                else []
            )

            for index, document in enumerate(
                result_documents
            ):
                metadata = (
                    result_metadatas[index]
                    if index < len(result_metadatas)
                    else {}
                )

                if (
                    index < len(result_distances)
                    and result_distances[index]
                    > max_distance
                ):
                    continue

                normal_entries.append(
                    (
                        document,
                        metadata,
                    )
                )

        if "vector_search" not in result["tool_calls"]:
            result["tool_calls"].append(
                "vector_search"
            )

    except Exception as exc:
        logger.debug(
            "Vector search failed: %s",
            exc,
        )

    # Framework initialization questions need a source-level check: short
    # assignment lines such as ``app = FastAPI(...)`` can rank below package
    # __init__.py chunks in semantic search. Scan only the selected repo's
    # indexed chunks and put exact initialization evidence first.
    asks_app_initialization = (
        re.search(r"\b(?:fastapi|flask|starlette)\b", question, re.IGNORECASE)
        and re.search(
            r"\b(?:where|which\s+file|initialized|initialised|created|defined|instantiated)\b",
            question,
            re.IGNORECASE,
        )
    )
    if asks_app_initialization:
        try:
            initialization_pattern = re.compile(
                r"\b(?:app|application)\s*=\s*(?:FastAPI|Flask|Starlette)\s*\(",
                re.IGNORECASE,
            )
            initialization_entries = []

            if repository_root:
                source_matches = grep_repo(
                    r"\b(?:app|application)\s*=\s*(?:FastAPI|Flask|Starlette)\s*\(",
                    path_glob="**/*.py",
                    repo_root=repository_root,
                )

                def initialization_path_priority(match: Dict[str, Any]):
                    path = PurePosixPath(
                        _normalize_path(str(match.get("file_path", ""))).casefold()
                    )
                    parts = path.parts
                    is_backend_app_main = parts[-3:] == (
                        "backend",
                        "app",
                        "main.py",
                    )
                    return (
                        0 if is_backend_app_main else 1,
                        0 if path.name == "main.py" else 1,
                        len(parts),
                    )

                if source_matches:
                    match = min(
                        source_matches,
                        key=initialization_path_priority,
                    )
                    file_path = str(match["file_path"])
                    line = int(match.get("line_number", 1))
                    first_line = max(1, line - 3)
                    last_line = line + 3
                    source = read_file(
                        file_path,
                        first_line,
                        last_line,
                        repo_root=repository_root,
                    )
                    initialization_entries.append((
                        f"FastAPI/framework initialization at "
                        f"{file_path}:{line}\n{source}",
                        {
                            "repo_id": repo_id,
                            "file_path": file_path,
                            "line_start": first_line,
                            "line_end": min(last_line, line + 3),
                        },
                    ))

            if not initialization_entries:
                repository_chunks = vector_store.get_by_repo(repo_id)
                repository_documents = repository_chunks.get("documents", [])
                repository_metadatas = repository_chunks.get("metadatas", [])
                for document, metadata in zip(
                    repository_documents,
                    repository_metadatas,
                ):
                    if (
                        isinstance(metadata, dict)
                        and metadata.get("repo_id") == repo_id
                        and initialization_pattern.search(str(document))
                    ):
                        initialization_entries.append((str(document), metadata))

            if initialization_entries:
                # This is a verified source-level answer. Keep unrelated
                # semantic hits out of its context so duplicate package
                # initializers cannot be reported as additional app entrypoints.
                normal_entries = initialization_entries
                if "get_by_repo" not in result["tool_calls"]:
                    result["tool_calls"].append("get_by_repo")
        except Exception as exc:
            logger.debug(
                "Framework initialization lookup failed: %s",
                exc,
            )

    # --------------------------------------------------------
    # EXPLICIT FILE RETRIEVAL
    # --------------------------------------------------------

    if is_file_inventory:
        indexed_paths = vector_store.get_file_paths(repo_id)
        if indexed_paths:
            inventory_document = (
                "REPOSITORY FILE INVENTORY:\n"
                + "\n".join(f"- {path}" for path in indexed_paths)
            )
            result["documents"].append(inventory_document)
            result["metadatas"].append({
                "repo_id": repo_id,
                "file_path": "__repository_file_inventory__",
            })
            result["tool_calls"].append("get_file_paths")
            return result

    if file_references:
        # existing code continues here...
        try:
            indexed_paths = (
                vector_store.get_file_paths(
                    repo_id
                )
            )

            resolved_references = set()

            for reference in file_references:
                matching_paths = [
                    indexed_path
                    for indexed_path in indexed_paths
                    if _path_matches_reference(
                        indexed_path,
                        reference,
                    )
                ]

                # If suffix matching fails, accept a basename match only
                # when exactly one indexed path has that basename.
                if not matching_paths:
                    requested_basename = (
                        PurePosixPath(
                            _normalize_path(reference)
                        ).name.casefold()
                    )
                    basename_matches = [
                        indexed_path
                        for indexed_path in indexed_paths
                        if PurePosixPath(
                            _normalize_path(indexed_path)
                        ).name.casefold() == requested_basename
                    ]
                    if len(basename_matches) == 1:
                        matching_paths = basename_matches

                # A fully qualified repository path should win over a
                # duplicate nested path that only shares the same suffix.
                exact_paths = [
                    indexed_path
                    for indexed_path in matching_paths
                    if _normalize_path(indexed_path).casefold()
                    == _normalize_path(reference).casefold()
                ]
                if exact_paths:
                    matching_paths = exact_paths

                for indexed_path in matching_paths:
                    # Prioritize named class definitions when a
                    # question explicitly requests a source file.
                    symbol_names = re.findall(
                        r"\b(?:[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*|[A-Z]{2,}[A-Z][a-z][A-Za-z0-9]*)\b",
                        question,
                    )
                    symbol_names = list(dict.fromkeys(symbol_names))

                    file_query = (
                        " ".join(
                            f"class {name}"
                            for name in symbol_names
                        )
                        if symbol_names
                        else question
                    )

                    if hasattr(vector_store, "get_by_file"):
                        # Explicit file requests need every chunk, not
                        # only the chunks ranked highest by semantic search.
                        raw_file_result = vector_store.get_by_file(
                            file_path=indexed_path,
                            repo_id=repo_id,
                        )
                        file_result = {
                            "documents": [
                                raw_file_result.get("documents", [])
                            ],
                            "metadatas": [
                                raw_file_result.get("metadatas", [])
                            ],
                        }
                    else:
                        # Compatibility for lightweight test doubles.
                        file_result = vector_store.search(
                            file_query,
                            n_results=50,
                            repo_id=repo_id,
                            file_path=indexed_path,
                        )

                    if (
                        "vector_search"
                        not in result["tool_calls"]
                    ):
                        result["tool_calls"].append(
                            "vector_search"
                        )

                    file_documents = (
                        file_result.get(
                            "documents",
                            [[]],
                        )
                    )

                    file_metadatas = (
                        file_result.get(
                            "metadatas",
                            [[]],
                        )
                    )

                    if (
                        not file_documents
                        or not file_documents[0]
                    ):
                        continue

                    resolved_references.add(
                        reference
                    )

                    document_list = list(
                        file_documents[0]
                    )

                    metadata_list = (
                        list(file_metadatas[0])
                        if file_metadatas
                        else [{}] * len(document_list)
                    )

                    file_entries = []
                    for index, document in enumerate(
                        document_list
                    ):
                        metadata = (
                            metadata_list[index]
                            if index < len(metadata_list)
                            else {}
                        )

                        file_entries.append(
                            (
                                document,
                                metadata,
                            )
                        )

                    # Move only chunks that declare a requested class to
                    # the front. Preserve vector order for every other chunk.
                    for symbol_name in symbol_names:
                        declaration = re.compile(
                            rf"\bclass\s+{re.escape(symbol_name)}\b"
                        )
                        matching_entries = [
                            entry
                            for entry in file_entries
                            if declaration.search(entry[0])
                        ]
                        if matching_entries:
                            file_entries = matching_entries + [
                                entry
                                for entry in file_entries
                                if entry not in matching_entries
                            ]

                    explicit_entries.extend(file_entries)

            result[
                "resolved_file_references"
            ] = sorted(
                resolved_references
            )

            result[
                "missing_file_references"
            ] = [
                reference
                for reference in file_references
                if reference
                not in resolved_references
            ]

            result[
                "file_reference_lookup_complete"
            ] = True

        except Exception:
            logger.exception(
                "Explicit repository file retrieval "
                "failed for repo_id=%s",
                repo_id,
            )

    # --------------------------------------------------------
    # COMBINE RETRIEVAL RESULTS
    #
    # Explicit requested files come FIRST.
    # Normal semantic results come afterwards.
    # --------------------------------------------------------

    seen_entries = set()

    if file_references and explicit_entries:
        entries_to_use = (
            explicit_entries
            + normal_entries
        )
    else:
        entries_to_use = normal_entries

    
    for document, metadata in entries_to_use:
        source_path = (
            metadata.get("file_path")
            or metadata.get("source")
            or metadata.get("file")
            or ""
        )

        source_name = PurePosixPath(
            _normalize_path(str(source_path))
        ).name.casefold()

        entry_key = (
            source_name or str(source_path),
            str(document).strip(),
        )


        if entry_key in seen_entries:
            continue

        seen_entries.add(entry_key)

        result["documents"].append(
            document
        )

        result["metadatas"].append(
            metadata
        )

    # --------------------------------------------------------
    # KNOWN FUNCTION VERIFICATION
    # --------------------------------------------------------

    target_functions = get_target_functions(
        question
    )

    function_sources = find_function_sources(
        target_functions,
        repo_id=repo_id,
    )

    if function_sources:
        for source_file in function_sources:
            for function_name in target_functions:
                try:
                    implementation = (
                        read_function_implementation(
                            source_file,
                            function_name,
                        )
                    )

                    if not implementation:
                        continue

                    source_range = (
                        find_source_line_range(
                            source_file,
                            function_name,
                        )
                    )

                    metadata: Dict[str, Any] = {
                        "source": source_file,
                        "file_path": source_file,
                        "repo_id": repo_id,
                    }

                    if source_range:
                        metadata["line_start"] = (
                            source_range[0]
                        )

                        metadata["line_end"] = (
                            source_range[1]
                        )

                    result["documents"].append(
                        implementation
                    )

                    result["metadatas"].append(
                        metadata
                    )

                    if (
                        "read_file"
                        not in result["tool_calls"]
                    ):
                        result["tool_calls"].append(
                            "read_file"
                        )

                    break

                except Exception as exc:
                    logger.debug(
                        "Repository retrieval failed: %s",
                        exc,
                    )

    return result


# ============================================================
# CONTEXT BUILDING
# ============================================================

def find_python_symbol_line_range(
    source: str,
    symbol_name: str,
    line_offset: int = 0,
) -> Optional[Tuple[int, int]]:
    """Find a Python class or method range in source text.

    line_offset is the number of original-file lines before this
    source chunk, allowing indexed chunks to retain absolute lines.
    """
    if not source.strip() or not symbol_name.strip():
        return None

    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, TypeError):
        return None

    matches: List[Tuple[int, int]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == symbol_name:
            matches.append((int(node.lineno), int(node.end_lineno or node.lineno)))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == symbol_name:
            matches.append((int(node.lineno), int(node.end_lineno or node.lineno)))

    if len(matches) != 1:
        return None

    start, end = matches[0]
    return start + line_offset, end + line_offset



def find_answer_python_class_citations(answer: str, context: str) -> List[str]:
    """Resolve mentioned Python classes and explicitly qualified methods."""
    header_pattern = re.compile(r"(?m)^FILE:\s*(.+):(\d+)-(\d+)\s*$")
    headers = list(header_pattern.finditer(context))
    resolved: List[str] = []

    def add(path: str, start_line: int, end_line: int) -> None:
        citation = f"{path}:{start_line}-{end_line}"
        if citation not in resolved:
            resolved.append(citation)

    for index, header in enumerate(headers):
        path = header.group(1).strip()
        chunk_start = int(header.group(2))
        chunk_end = int(header.group(3))
        content_end = (
            headers[index + 1].start()
            if index + 1 < len(headers)
            else len(context)
        )
        chunk_text = context[header.end():content_end].strip()

        try:
            tree = ast.parse(chunk_text)
        except (SyntaxError, ValueError, TypeError):
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue

            class_pattern = (
                rf"(?<![A-Za-z0-9_]){re.escape(node.name)}"
                rf"(?![A-Za-z0-9_])"
            )
            if not re.search(class_pattern, answer):
                continue

            class_start = chunk_start + node.lineno - 1
            class_end = chunk_start + (node.end_lineno or node.lineno) - 1

            if class_start < chunk_start or class_end > chunk_end:
                continue

            matched_methods = []
            for member in node.body:
                if not isinstance(
                    member, (ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    continue

                qualified_pattern = (
                    rf"(?<![A-Za-z0-9_]){re.escape(node.name)}"
                    rf"\s*\.\s*{re.escape(member.name)}"
                    rf"(?![A-Za-z0-9_])"
                )
                if not re.search(qualified_pattern, answer):
                    continue

                method_start = chunk_start + member.lineno - 1
                method_end = (
                    chunk_start + (member.end_lineno or member.lineno) - 1
                )

                if (
                    method_start >= class_start
                    and method_end <= class_end
                    and method_start <= method_end
                    and method_end <= chunk_end
                ):
                    matched_methods.append((method_start, method_end))

            if matched_methods:
                for method_start, method_end in matched_methods:
                    add(path, method_start, method_end)
            else:
                add(path, class_start, class_end)

    return resolved


def correct_python_class_citations(
    citations: List[str], answer: str, context: str
) -> List[str]:
    """Replace unsupported same-file ranges with resolved Python symbol ranges."""
    resolved = find_answer_python_class_citations(answer, context)
    if not resolved:
        return citations

    resolved_paths = {
        _normalize_path(re.sub(r":\d+-\d+$", "", item)).casefold()
        for item in resolved
    }

    result = [
        item for item in citations
        if _normalize_path(re.sub(r":\d+-\d+$", "", item)).casefold()
        not in resolved_paths
    ]

    result.extend(item for item in resolved if item not in result)
    return result
def build_context(
    information: Dict[str, Any],
    priority_files: Optional[List[str]] = None,
) -> str:
    """
    Build LLM context.

    Explicitly requested files are placed first so the LLM
    sees the requested source before unrelated semantic
    retrieval results.
    """

    documents = information.get(
        "documents",
        [],
    )

    metadatas = information.get(
        "metadatas",
        [],
    )

    priority_files = priority_files or []

    normalized_priority_files = [
        _normalize_path(str(path)).casefold()
        for path in priority_files
        if _normalize_path(str(path))
    ]

    entries: List[
        Tuple[str, Dict[str, Any]]
    ] = []

    for index, document in enumerate(
        documents
    ):
        metadata: Dict[str, Any] = {}

        if index < len(metadatas):
            raw_metadata = metadatas[index]

            if isinstance(
                raw_metadata,
                dict,
            ):
                metadata = raw_metadata

        entries.append(
            (
                str(document),
                metadata,
            )
        )

    priority_entries: List[
        Tuple[str, Dict[str, Any]]
    ] = []

    normal_entries: List[
        Tuple[str, Dict[str, Any]]
    ] = []

    for document, metadata in entries:
        source = _metadata_file_path(
            metadata
        )

        is_priority = False

        if source:
            for requested_file in (
                normalized_priority_files
            ):
                if _path_matches_reference(
                    source,
                    requested_file,
                ):
                    is_priority = True
                    break

        if is_priority:
            priority_entries.append(
                (
                    document,
                    metadata,
                )
            )
        else:
            normal_entries.append(
                (
                    document,
                    metadata,
                )
            )

    if normalized_priority_files and priority_entries:
        ordered_entries = priority_entries
    else:
        ordered_entries = priority_entries + normal_entries

    context_parts: List[str] = []

    for document, metadata in ordered_entries:
        secret_result = scan_for_secrets(
            str(document)
        )

        safe_document = (
            secret_result.redacted
            if (
                secret_result.found
                and secret_result.redacted
                is not None
            )
            else str(document)
        )

        source = metadata.get(
            "file_path",
            metadata.get(
                "source",
                metadata.get(
                    "file",
                    "",
                ),
            ),
        )

        line_start = metadata.get(
            "line_start"
        )

        line_end = metadata.get(
            "line_end"
        )

        language = metadata.get(
            "language"
        )

        if not language and source:
            language = language_for_path(
                Path(str(source))
            )

        location = str(source)

        if (
            source
            and line_start is not None
            and line_end is not None
        ):
            location = (
                f"{source}:"
                f"{line_start}-"
                f"{line_end}"
            )

        if location:
            language_label = (
                f"LANGUAGE: {language}\n"
                if language
                else ""
            )

            priority_label = ""

            normalized_source = (
                _normalize_path(
                    str(source)
                ).casefold()
            )

            if any(
                _path_matches_reference(
                    normalized_source,
                    requested_file,
                )
                for requested_file
                in normalized_priority_files
            ):
                priority_label = (
                    "EXPLICITLY REQUESTED FILE: YES\n"
                )

            context_parts.append(
                f"{priority_label}"
                f"{language_label}"
                f"FILE: {location}\n"
                f"{safe_document}"
            )
        else:
            context_parts.append(
                safe_document
            )

    return "\n\n".join(context_parts)


# ============================================================
# CITATIONS
# ============================================================

def extract_citations(
    answer: str,
    context: str,
) -> List[str]:
    citations: List[str] = []

    # Accept only recognized file extensions. This prevents expressions
    # such as self.base, self.model, and os.getenv from being citations.
    extensions = (
        r"py|pyi|js|jsx|ts|tsx|java|go|rs|cpp|c|h|hpp|cs|rb|php|"
        r"swift|kt|kts|sql|yaml|yml|toml|json|jsonl|md|mdx|sh|bash|"
        r"html|htm|css|scss|xml|ini|cfg|conf|txt|ipynb|vue|svelte|"
        r"gradle|properties|dockerfile"
    )
    path_pattern = (
        r"(?<![\w])"
        r"(?:[\w.@+-]+[/\\])+"
        r"[\w@+-]+\.(?:" + extensions + r")"
        r"(?:\:\d+(?:-\d+)?)?"
        r"|(?<![\w])"
        r"[\w@+-]+\.(?:" + extensions + r")"
        r"(?:\:\d+(?:-\d+)?)?"
    )

    def add_path(path: str) -> None:
        normalized = _normalize_path(path)

        if (
            normalized
            and normalized not in citations
            and _is_relevant_source_file(normalized)
        ):
            citations.append(normalized)

    # Extract citations from the generated answer first.
    for path in re.findall(
        path_pattern,
        answer,
    ):
        add_path(path)

    # If the answer already contains precise line-range citations,
    # do not add bare file references from the context.
    has_ranged_citation = any(
        re.search(r":\d+-\d+$", citation)
        for citation in citations
    )

    if not has_ranged_citation:
        for path in re.findall(
            path_pattern,
            context,
        ):
            add_path(path)

    # Prefer precise line-range citations over bare file citations.
    ranged_files = {
        re.sub(r":\d+-\d+$", "", citation)
        for citation in citations
        if re.search(r":\d+-\d+$", citation)
    }

    if ranged_files:
        citations = [
            citation
            for citation in citations
            if (
                re.search(r":\d+-\d+$", citation)
                or citation not in ranged_files
            )
        ]

    return citations

def prioritize_citations(
    citations: List[str],
    function_sources: List[str],
) -> List[str]:
    prioritized: List[str] = []

    for source in function_sources:
        normalized = _normalize_path(
            source
        )

        if (
            normalized
            and normalized not in prioritized
        ):
            prioritized.append(
                normalized
            )

    for citation in citations:
        normalized = _normalize_path(
            citation
        )

        if (
            normalized
            and normalized not in prioritized
        ):
            prioritized.append(
                normalized
            )

    return prioritized


def verify_repository_sources(
    citations: List[str],
    function_sources: Optional[List[str]] = None,
    target_functions: Optional[List[str]] = None,
    metadatas: Optional[
        List[Dict[str, Any]]
    ] = None,
    repository_root: Optional[str] = None,
) -> List[str]:
    """
    Verify repository citations.

    Priority:
    1. Exact AST range for known Python functions.
    2. Indexed chunk metadata.
    3. Complete current-file range.

    LLM-generated line numbers are never trusted.
    """

    verified: List[str] = []

    function_sources = (
        function_sources or []
    )

    target_functions = (
        target_functions or []
    )

    metadatas = metadatas or []

    normalized_function_sources = {
        _normalize_path(source)
        for source in function_sources
        if _normalize_path(source)
    }

    metadata_ranges: Dict[
        str,
        List[Tuple[int, int]],
    ] = {}

    for metadata in metadatas:
        if not isinstance(
            metadata,
            dict,
        ):
            continue

        file_path = (
            metadata.get("file_path")
            or metadata.get("source")
            or metadata.get("file")
        )

        if not file_path:
            continue

        normalized_file = _normalize_path(
            str(file_path)
        )

        try:
            line_start = int(
                metadata.get(
                    "line_start",
                    0,
                )
            )

            line_end = int(
                metadata.get(
                    "line_end",
                    0,
                )
            )

        except (
            TypeError,
            ValueError,
        ):
            continue

        if (
            line_start < 1
            or line_end < line_start
        ):
            continue

        metadata_ranges.setdefault(
            normalized_file,
            [],
        ).append(
            (
                line_start,
                line_end,
            )
        )

    for citation in citations:
        normalized = _normalize_path(
            citation
        )

        if not normalized:
            continue

        # Preserve an LLM-provided line range before normalizing
        # the path for repository verification.
        range_match = re.search(
            r":(\d+)-(\d+)$",
            normalized,
        )

        requested_range = None

        if range_match:
            requested_range = (
                int(range_match.group(1)),
                int(range_match.group(2)),
            )

        normalized = re.sub(
            r":\d+-\d+$",
            "",
            normalized,
        )

        if not _is_relevant_source_file(
            normalized
        ):
            continue

        try:
            relative_path = Path(normalized)

            if relative_path.is_absolute():
                continue

            citation_path = normalized
            path_exists = False

            if repository_root:
                root = Path(repository_root).resolve()
                path = (root / relative_path).resolve()

                # Some repositories expose backend files under
                # repository_root/backend/.
                if (
                    not path.exists()
                    and not normalized.startswith("backend/")
                ):
                    backend_path = (
                        root / "backend" / relative_path
                    ).resolve()

                    if backend_path.exists():
                        path = backend_path
                        citation_path = (
                            Path("backend") / relative_path
                        ).as_posix()

                # Prevent citations from escaping the repository.
                path.relative_to(root)
                path_exists = path.exists() and path.is_file()

            else:
                path = relative_path.resolve()
                path_exists = path.exists() and path.is_file()

        except (OSError, ValueError):
            continue

        # Indexed repositories may not exist as physical files.
        # In that case, Chroma metadata is sufficient to verify
        # the citation.
        has_indexed_metadata = bool(
            metadata_ranges.get(normalized)
        )

        if not path_exists and not has_indexed_metadata:
            continue

        # -----------------------------------------------------
        # Exact AST function citation
        # -----------------------------------------------------

        if (
            normalized
            in normalized_function_sources
            and target_functions
        ):
            source_range = None

            for function_name in target_functions:
                source_range = (
                    find_source_line_range(
                        normalized,
                        function_name,
                    )
                )

                if source_range:
                    break

            if source_range:
                start_line, end_line = (
                    source_range
                )

                citation_with_range = (
                    f"{citation_path}:"
                    f"{start_line}-"
                    f"{end_line}"
                )

                if (
                    citation_with_range
                    not in verified
                ):
                    verified.append(
                        citation_with_range
                    )

                continue

        # -----------------------------------------------------
        # Indexed chunk citation
        # -----------------------------------------------------

        ranges = metadata_ranges.get(
            normalized,
            [],
        )

        if ranges:
            # If the LLM supplied a specific line range, keep only
            # the verified indexed chunk that contains that range.
            if requested_range:
                requested_start, requested_end = (
                    requested_range
                )

                # Reject malformed explicit line ranges.
                if requested_start < 1 or requested_end < requested_start:
                    continue

                matching_ranges = [
                    (start_line, end_line)
                    for start_line, end_line in ranges
                    if (
                        start_line <= requested_start
                        and requested_end <= end_line
                    )
                ]

                if matching_ranges:
                    for start_line, end_line in sorted(
                        set(matching_ranges)
                    ):
                        citation_with_range = (
                            f"{citation_path}:"
                            f"{start_line}-"
                            f"{end_line}"
                        )

                        if (
                            citation_with_range
                            not in verified
                        ):
                            verified.append(
                                citation_with_range
                            )

                    continue

                # Never replace an unverified explicit range with
                # broad citations covering unrelated chunks.
                continue

            # No usable requested range: preserve the existing
            # behavior and return all verified indexed chunks.
            for start_line, end_line in sorted(
                set(ranges)
            ):
                citation_with_range = (
                    f"{citation_path}:"
                    f"{start_line}-"
                    f"{end_line}"
                )

                if (
                    citation_with_range
                    not in verified
                ):
                    verified.append(
                        citation_with_range
                    )

            continue

        # -----------------------------------------------------
        # Complete file fallback
        # -----------------------------------------------------

        file_range = find_source_line_range(
            normalized
        )

        if file_range:
            start_line, end_line = (
                file_range
            )

            citation_with_range = (
                f"{citation_path}:"
                f"{start_line}-"
                f"{end_line}"
            )

            if (
                citation_with_range
                not in verified
            ):
                verified.append(
                    citation_with_range
                )

    return verified


def has_unverified_function_implementation_claim(
    answer: str,
    repository_root: Optional[str],
) -> bool:
    """Return whether an answer cites lines that don't define its named function."""
    if not answer or not repository_root:
        return False

    claim_pattern = re.compile(
        r"\b(?:function|method)\s+[`*]*"
        r"([A-Za-z_]\w*)[`*]*\s+"
        r"(?:is\s+)?(?:implemented|defined)\s+"
        r"(?:in|at)\s+[`*]*"
        r"([^`*\s:]+\.py):(\d+)-(\d+)",
        re.IGNORECASE,
    )

    try:
        root = Path(repository_root).resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return False

    for match in claim_pattern.finditer(answer):
        function_name, cited_path, start_text, end_text = (
            match.groups()
        )

        try:
            start_line = int(start_text)
            end_line = int(end_text)
            relative_path = Path(
                _normalize_path(cited_path)
            )
            source_path = (root / relative_path).resolve()
            source_path.relative_to(root)

            if not source_path.is_file():
                backend_path = (root / "backend" / relative_path).resolve()
                backend_path.relative_to(root)
                source_path = backend_path

            source = source_path.read_text(
                encoding="utf-8",
                errors="replace",
            )
            tree = ast.parse(source)
        except (OSError, RuntimeError, SyntaxError, TypeError, ValueError):
            # The normal citation verifier handles unavailable or invalid files.
            continue

        definition_is_cited = any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
            and start_line <= int(node.lineno) <= end_line
            for node in ast.walk(tree)
        )

        if not definition_is_cited:
            return True

    return False


# ============================================================
# TOOL DETECTION
# ============================================================

def detect_tool_calls(
    question: str,
) -> List[str]:
    question_lower = question.lower()

    calls: List[str] = []

    if any(
        keyword in question_lower
        for keyword in [
            "where",
            "which file",
            "implemented",
            "implementation",
            "function",
            "method",
            "located",
        ]
    ):
        calls.extend(
            [
                "grep_repo",
                "read_file",
            ]
        )

    if any(
        keyword in question_lower
        for keyword in [
            "who wrote",
            "author",
            "blame",
            "commit",
        ]
    ):
        calls.append(
            "git_blame"
        )

    if any(
        keyword in question_lower
        for keyword in [
            "secret",
            "secrets",
            "password",
            "token",
            "api key",
        ]
    ):
        calls.append(
            "secret_scanner"
        )

    if any(
        keyword in question_lower
        for keyword in [
            "insecure",
            "security",
            "eval",
            "exec",
            "sql",
        ]
    ):
        calls.append(
            "insecure_pattern_scanner"
        )

    return calls


# ============================================================
# DETERMINISTIC FUNCTION ANSWERS
# ============================================================

def build_function_location_answer(
    question: str,
    function_sources: List[str],
) -> Optional[str]:

    if not is_function_location_question(
        question
    ):
        return None

    if not function_sources:
        return None

    target_functions = get_target_functions(
        question
    )

    if not target_functions:
        return None

    function_name = target_functions[0]
    source_file = function_sources[0]

    return (
        f"The function `{function_name}` "
        f"is implemented in "
        f"`{source_file}`."
    )


def build_function_explanation_answer(
    question: str,
    function_sources: List[str],
) -> Optional[str]:

    if not is_function_explanation_question(
        question
    ):
        return None

    if not function_sources:
        return None

    target_functions = get_target_functions(
        question
    )

    if not target_functions:
        return None

    # ---------------------------------------------------------
    # incremental_index
    # ---------------------------------------------------------

    if "incremental_index" in target_functions:
        source_file = FUNCTION_SOURCE_MAP.get(
            "incremental_index"
        )

        if source_file:
            source_range = find_source_line_range(
                source_file,
                "incremental_index",
            )

            if source_range:
                return (
                    "`incremental_index` updates the "
                    "repository index incrementally "
                    "instead of rebuilding the entire "
                    "repository. It compares the base "
                    "commit with the current HEAD, "
                    "identifies added, modified, and "
                    "deleted files, removes stale vectors "
                    "for affected content, re-chunks and "
                    "embeds changed files, and upserts the "
                    "new chunks into the vector store. "
                    "Unchanged files do not need to be "
                    "re-indexed. The implementation is in "
                    f"`{source_file}`."
                )

    # ---------------------------------------------------------
    # Other known functions
    # ---------------------------------------------------------

    for function_name in target_functions:
        for source_file in function_sources:
            try:
                pattern = (
                    rf"^\s*(?:async\s+)?def\s+"
                    rf"{re.escape(function_name)}\s*\("
                )

                matches = grep_repo(
                    pattern,
                    source_file,
                )

                if not matches:
                    continue

                implementation = (
                    read_function_implementation(
                        source_file,
                        function_name,
                    )
                )

                if not implementation:
                    continue

                # -------------------------------------------------
                # retrieve_information
                # -------------------------------------------------

                if function_name == (
                    "retrieve_information"
                ):
                    return (
                        "`retrieve_information` retrieves "
                        "relevant code from the vector store "
                        "and directly reads repository "
                        "functions when the question references "
                        "known functions. It also preserves "
                        "source metadata such as file paths "
                        "and line ranges for citation "
                        "verification. It is implemented in "
                        f"`{source_file}`."
                    )

                # -------------------------------------------------
                # verify_repository_sources
                # -------------------------------------------------

                if function_name == (
                    "verify_repository_sources"
                ):
                    return (
                        "`verify_repository_sources` validates "
                        "repository citations against files "
                        "that actually exist. For known Python "
                        "functions it uses AST line ranges, "
                        "and for retrieved chunks it can use "
                        "indexed line metadata. It does not "
                        "trust line numbers generated by the "
                        "LLM. It is implemented in "
                        f"`{source_file}`."
                    )

                # -------------------------------------------------
                # Generic known-function explanation
                # -------------------------------------------------

                return (
                    f"`{function_name}` is implemented in "
                    f"`{source_file}`. Its implementation "
                    "was verified directly from the "
                    "repository."
                )

            except Exception as exc:
                logger.debug(
                    "Could not explain function %s: %s",
                    function_name,
                    exc,
                )

    return None


# ============================================================
# MAIN AGENT LOOP
# ============================================================

async def agent_loop(
    question: str,
    repo_id: str = "local",
    session_id: Optional[str] = None,
) -> ChatResponse:

    overview_request = is_project_overview_question(
        question
    )

    contextual_question = (
        build_contextual_question(
            question,
            session_id,
        )
    )

    indexed_commit: Optional[str] = None
    repository_path: Optional[str] = None

    # ---------------------------------------------------------
    # Repository information
    # ---------------------------------------------------------

    try:
        from app.services.repo_service import RepoService

        repo_service = RepoService()

        repo_info = repo_service.get_repo_info(
            repo_id
        )

        if repo_info is not None:
            repository_path = getattr(
                repo_info,
                "path",
                None,
            )

            indexed_commit = getattr(
                repo_info,
                "last_indexed_commit",
                None,
            )

    except Exception:
        logger.debug(
            "Could not determine indexed commit "
            "for chat cache",
            exc_info=True,
        )

    # ---------------------------------------------------------
    # Cache
    # ---------------------------------------------------------

    cached_response = chat_response_cache.get(
        repo_id=repo_id,
        indexed_commit=indexed_commit,
        question=contextual_question,
    )

    if (
        cached_response is not None
        and not (
            repository_path
            and overview_request
        )
    ):
        logger.info(
            "Chat response cache hit: repo_id=%s",
            repo_id,
        )

        cached_response = cached_response.copy(
            update={
                "tool_calls_made": [
                    "response_cache"
                ]
            }
        )

        _save_session(
            session_id=session_id,
            question=question,
            answer=cached_response.answer,
            citations=cached_response.citations,
        )

        return cached_response

    logger.info(
        "Chat response cache miss: repo_id=%s",
        repo_id,
    )

    # ---------------------------------------------------------
    # Target functions
    # ---------------------------------------------------------

    target_functions = get_target_functions(
        contextual_question
    )

    function_sources = find_function_sources(
        target_functions,
        repo_id=repo_id,
    )

    # ---------------------------------------------------------
    # Retrieval
    # ---------------------------------------------------------

    information = retrieve_information(
        contextual_question,
        repo_id,
        repository_root=repository_path,
        overview_request=overview_request,
    )

    function_verification_context = (
        build_function_verification_context(
            contextual_question,
            function_sources,
        )
    )

    # ---------------------------------------------------------
    # Explicit requested files get priority
    # ---------------------------------------------------------

    requested_file_references = information.get(
        "requested_file_references",
        [],
    )

    context = build_context(
        information,
        priority_files=requested_file_references,
    )
    if requested_file_references and len(context) > 50000:
        context = context[:50000]

    # ---------------------------------------------------------
    # IMPORTANT:
    # Put direct function verification BEFORE normal context.
    #
    # This protects deterministic function evidence from being
    # pushed beyond an LLM context truncation limit.
    # ---------------------------------------------------------

    if function_verification_context:
        verification_header = (
            "DIRECT FUNCTION VERIFICATION:\n"
        )

        if context:
            context = (
                verification_header
                + function_verification_context
                + "\n\n"
                + context
            )
        else:
            context = (
                verification_header
                + function_verification_context
            )

    tool_calls = list(
        information.get(
            "tool_calls",
            [],
        )
    )

    resolved_file_references = information.get(
        "resolved_file_references",
        [],
    )

    # ---------------------------------------------------------
    # Requested file was not found
    # ---------------------------------------------------------

    if (
        requested_file_references
        and information.get(
            "file_reference_lookup_complete",
            False,
        )
        and not resolved_file_references
    ):
        requested_files = ", ".join(
            f"`{reference}`"
            for reference in requested_file_references
        )

        response = ChatResponse(
            answer=(
                f"I could not find the requested file "
                f"{requested_files} in the current repository."
            ),
            citations=[],
            tool_calls_made=tool_calls,
            confidence=0.0,
            next_steps_suggestion=(
                "Check the file path and ensure the file was indexed."
            ),
        )

        _save_session(
            session_id=session_id,
            question=question,
            answer=response.answer,
            citations=response.citations,
        )

        return response

    # ---------------------------------------------------------
    # Scope guardrail
    # ---------------------------------------------------------

    if (
        not context
        and not scope_guardrail(
            contextual_question
        )
    ):
        response = ChatResponse(
            answer=(
                "This question is off-topic for "
                "the Codebase Inspector."
            ),
            citations=[],
            tool_calls_made=tool_calls,
            confidence=0.0,
            next_steps_suggestion=(
                "Ask about software concepts or "
                "the selected repository."
            ),
        )

        _save_session(
            session_id=session_id,
            question=question,
            answer=response.answer,
            citations=response.citations,
        )

        return response

    # ---------------------------------------------------------
    # Project overview
    # ---------------------------------------------------------

    if (
        overview_request
        and repository_path
    ):
        overview = build_project_overview(
            repo_path=repository_path,
            repo_id=repo_id,
            vector_store=vector_store,
        )

        response = ChatResponse(
            answer=overview.answer,
            citations=overview.citations,
            tool_calls_made=[
                *tool_calls,
                "repository_scan",
            ],
            confidence=overview.confidence,
            next_steps_suggestion=(
                "Ask about a cited file or route to "
                "trace its implementation in more detail."
            ),
        )

        _save_session(
            session_id=session_id,
            question=question,
            answer=response.answer,
            citations=response.citations,
        )

        try:
            chat_response_cache.set(
                repo_id=repo_id,
                indexed_commit=indexed_commit,
                question=contextual_question,
                response=response,
            )
        except Exception:
            logger.debug(
                "Could not cache repository overview",
                exc_info=True,
            )

        return response

    # ---------------------------------------------------------
    # Deterministic Ollama request attribution
    # ---------------------------------------------------------

    if is_ollama_generation_request_question(contextual_question):
        ollama_request_answer = (
            build_ollama_request_answer(repository_path, contextual_question)
            if repository_path
            else None
        )

        if ollama_request_answer:
            answer, citations = ollama_request_answer
            response = ChatResponse(
                answer=answer,
                citations=citations,
                tool_calls_made=tool_calls,
                confidence=0.98,
                next_steps_suggestion=(
                    "Ask about either endpoint to trace its callers."
                ),
            )
        else:
            response = ChatResponse(
                answer=(
                    "I could not verify an Ollama request method in the "
                    "selected repository source, so I won't guess."
                ),
                citations=[],
                tool_calls_made=tool_calls,
                confidence=0.1,
                next_steps_suggestion=(
                    "Check that the selected repository includes its LLM service file."
                ),
            )

        _save_session(
            session_id=session_id,
            question=question,
            answer=response.answer,
            citations=response.citations,
        )
        return response

    # ---------------------------------------------------------
    # Deterministic Docker-to-FastAPI startup trace
    # ---------------------------------------------------------

    if is_docker_backend_startup_question(contextual_question):
        startup_answer = (
            build_docker_backend_startup_answer(repository_path)
            if repository_path
            else None
        )
        if startup_answer:
            answer, citations = startup_answer
            response = ChatResponse(
                answer=answer,
                citations=citations,
                tool_calls_made=[*tool_calls, "repository_scan"],
                confidence=(
                    0.99
                    if answer.startswith("The root Compose file")
                    else 0.6
                ),
                next_steps_suggestion=(
                    "Ask about the FastAPI startup code or another service."
                ),
            )
        else:
            response = ChatResponse(
                answer=(
                    "I could not verify the backend startup command from "
                    "the selected repository's root Compose file, so I "
                    "won't guess."
                ),
                citations=[],
                tool_calls_made=tool_calls,
                confidence=0.1,
                next_steps_suggestion=(
                    "Check that the selected repository includes its root "
                    "Compose file and backend entry point."
                ),
            )
        _save_session(
            session_id=session_id,
            question=question,
            answer=response.answer,
            citations=response.citations,
        )
        return response

    # ---------------------------------------------------------
    # No repository context
    # ---------------------------------------------------------

    if not context:
        try:
            answer = llm_service.generate(
                question=contextual_question,
                context=(
                    "No relevant repository evidence "
                    "was found for this question."
                ),
            ).strip()

            if answer == "OUT_OF_SCOPE":
                response = ChatResponse(
                    answer=(
                        "This question is off-topic for "
                        "the Codebase Inspector."
                    ),
                    citations=[],
                    tool_calls_made=tool_calls,
                    confidence=0.0,
                    next_steps_suggestion=(
                        "Ask about software concepts or "
                        "the selected repository."
                    ),
                )

                _save_session(
                    session_id=session_id,
                    question=question,
                    answer=response.answer,
                    citations=response.citations,
                )

                return response

            if (
                "answer_generation"
                not in tool_calls
            ):
                tool_calls.append(
                    "answer_generation"
                )

        except Exception:
            logger.exception(
                "General response generation failed "
                "without repository evidence"
            )

            answer = (
                "I could not determine this confidently "
                "from the uploaded files. The repository "
                "search did not return evidence relevant "
                "to this question."
            )

        response = ChatResponse(
            answer=answer,
            citations=[],
            tool_calls_made=tool_calls,
            confidence=(
                0.45
                if "answer_generation"
                in tool_calls
                else 0.1
            ),
            next_steps_suggestion=(
                "This is a general explanation, not a "
                "description of the selected repository's "
                "implementation."
            ),
        )

        _save_session(
            session_id=session_id,
            question=question,
            answer=response.answer,
            citations=response.citations,
        )

        return response

    # =========================================================
    # REPOSITORY TOOL EXECUTOR
    # =========================================================

    tool_source_metadata: List[
        Dict[str, Any]
    ] = []

    def repository_tool_executor(
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Any:

        if (
            not repository_path
            and repo_id != "local"
        ):
            raise ValueError(
                "The selected repository path "
                "could not be resolved."
            )

        if repository_path:
            result = execute_tool(
                tool_name,
                arguments,
                repo_root=repository_path,
            )
        else:
            result = execute_tool(
                tool_name,
                arguments,
            )

        # -----------------------------------------------------
        # read_file metadata
        # -----------------------------------------------------

        if (
            tool_name == "read_file"
            and result
        ):
            file_path = str(
                arguments.get(
                    "path",
                    "",
                )
            ).replace("\\", "/")

            try:
                line_start = int(
                    arguments.get(
                        "line_start",
                        1,
                    )
                )
            except (
                TypeError,
                ValueError,
            ):
                line_start = 1

            line_end = arguments.get(
                "line_end"
            )

            if line_end is None:
                line_end = (
                    line_start
                    + len(
                        str(result).splitlines()
                    )
                    - 1
                )

            if (
                file_path
                and line_end >= line_start
            ):
                tool_source_metadata.append(
                    {
                        "file_path": file_path,
                        "line_start": line_start,
                        "line_end": int(
                            line_end
                        ),
                    }
                )

        # -----------------------------------------------------
        # grep_repo metadata
        # -----------------------------------------------------

        elif (
            tool_name == "grep_repo"
            and isinstance(result, list)
        ):
            for match in result:
                if not isinstance(
                    match,
                    dict,
                ):
                    continue

                file_path = match.get(
                    "file_path"
                )

                line_number = match.get(
                    "line_number",
                    match.get("line"),
                )

                if (
                    file_path
                    and line_number
                ):
                    try:
                        line_number = int(
                            line_number
                        )
                    except (
                        TypeError,
                        ValueError,
                    ):
                        continue

                    tool_source_metadata.append(
                        {
                            "file_path": str(
                                file_path
                            ),
                            "line_start": line_number,
                            "line_end": line_number,
                        }
                    )

        # -----------------------------------------------------
        # git_blame metadata
        # -----------------------------------------------------

        elif (
            tool_name == "git_blame"
            and isinstance(result, dict)
        ):
            file_path = arguments.get(
                "path"
            )

            line_number = result.get(
                "line"
            )

            if (
                file_path
                and line_number
            ):
                try:
                    line_number = int(
                        line_number
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    return result

                tool_source_metadata.append(
                    {
                        "file_path": str(
                            file_path
                        ),
                        "line_start": line_number,
                        "line_end": line_number,
                    }
                )

        return result

    # =========================================================
    # DETERMINISTIC ANSWERS
    # =========================================================

    deterministic_explanation = (
        build_function_explanation_answer(
            contextual_question,
            function_sources,
        )
    )

    deterministic_location = None

    if deterministic_explanation:
        final_answer = (
            deterministic_explanation
        )

    else:
        deterministic_location = (
            build_function_location_answer(
                contextual_question,
                function_sources,
            )
        )

        if deterministic_location:
            final_answer = (
                deterministic_location
            )

        else:
            # =================================================
            # LLM / TOOL DECISION
            # =================================================

            question_lower = (
                contextual_question.lower()
            )

            tool_required_keywords = [
                "grep",
                "search the repository",
                "search repository",
                "find all usages",
                "find usages",
                "find usage",
                "who changed",
                "git blame",
                "commit",
                "commits",
                "where is this function called",
                "where is this method called",
                "where is this class used",
                "where is this function used",
                "trace the code",
                "trace this function",
                "trace this method",
                "show me all references",
                "find references",
                "search for",
            ]

            requires_repository_tool = any(
                keyword in question_lower
                for keyword in tool_required_keywords
            )

            # =================================================
            # NORMAL RAG QUESTION
            # =================================================

            if not requires_repository_tool:
                try:
                    final_answer = (
                        llm_service.generate(
                            question=contextual_question,
                            context=context,
                        )
                    )

                    if (
                        "answer_generation"
                        not in tool_calls
                    ):
                        tool_calls.append(
                            "answer_generation"
                        )

                except Exception:
                    logger.exception(
                        "Normal LLM generation failed"
                    )

                    final_answer = (
                        "I could not generate an AI "
                        "response from the available "
                        "repository context."
                    )

            # =================================================
            # REPOSITORY TOOL QUESTION
            # =================================================

            elif (
                llm_service.provider_name
                in {
                    "claude",
                    "ollama",
                }
            ):
                try:
                    tool_result = (
                        llm_service.generate_with_tools(
                            question=contextual_question,
                            context=context,
                            tools=TOOL_DEFINITIONS,
                            tool_executor=(
                                repository_tool_executor
                            ),
                            max_iterations=2,
                        )
                    )

                    if (
                        "answer_generation"
                        not in tool_calls
                    ):
                        tool_calls.append(
                            "answer_generation"
                        )

                    final_answer = (
                        tool_result.get(
                            "answer",
                            "",
                        )
                    )

                    for tool_call in (
                        tool_result.get(
                            "tool_calls",
                            [],
                        )
                    ):
                        if isinstance(
                            tool_call,
                            dict,
                        ):
                            tool_name = (
                                tool_call.get(
                                    "name"
                                )
                                or tool_call.get(
                                    "tool"
                                )
                            )
                        else:
                            tool_name = str(
                                tool_call
                            )

                        if (
                            tool_name
                            and tool_name
                            not in tool_calls
                        ):
                            tool_calls.append(
                                tool_name
                            )

                except Exception:
                    logger.exception(
                        "LLM tool-calling generation failed"
                    )

                    # -----------------------------------------
                    # SAFE FALLBACK
                    # -----------------------------------------

                    try:
                        final_answer = (
                            llm_service.generate(
                                question=(
                                    contextual_question
                                ),
                                context=context,
                            )
                        )

                        if (
                            "answer_generation"
                            not in tool_calls
                        ):
                            tool_calls.append(
                                "answer_generation"
                            )

                    except Exception:
                        logger.exception(
                            "Fallback LLM generation failed"
                        )

                        final_answer = (
                            "I could not generate an AI "
                            "response from the available "
                            "repository context."
                        )

            else:
                # -------------------------------------------------
                # Unsupported provider fallback
                # -------------------------------------------------

                try:
                    final_answer = (
                        llm_service.generate(
                            question=contextual_question,
                            context=context,
                        )
                    )

                    if (
                        "answer_generation"
                        not in tool_calls
                    ):
                        tool_calls.append(
                            "answer_generation"
                        )

                except Exception:
                    logger.exception(
                        "Unsupported-provider fallback "
                        "generation failed"
                    )

                    final_answer = (
                        "I could not generate an AI response "
                        "from the available repository context."
                    )

    # =========================================================
    # GUARDRAILS
    # =========================================================

    unverified_function_claim = (
        has_unverified_function_implementation_claim(
            final_answer,
            repository_path,
        )
    )

    if unverified_function_claim:
        logger.warning(
            "Discarding function answer whose cited range does not "
            "contain its definition: repo_id=%s",
            repo_id,
        )
        final_answer = (
            "The available repository evidence does not verify which "
            "function makes this request. I can't answer confidently "
            "from the retrieved source."
        )

    guardrail_result = guardrail_pipeline(
        agent_output=final_answer,
        question=contextual_question,
    )

    final_answer = guardrail_result.get(
        "output",
        final_answer,
    )

    # =========================================================
    # CITATIONS
    # =========================================================

    citations = extract_citations(
        final_answer,
        context,
    )
    citations = correct_python_class_citations(
        citations, final_answer, context
    )

    citations = prioritize_citations(
        citations,
        function_sources,
    )

    source_metadata = [
        *information.get(
            "metadatas",
            [],
        ),
        *tool_source_metadata,
    ]

    used_source_paths = {
        _normalize_path(
            str(
                metadata.get(
                    "file_path",
                    metadata.get(
                        "source",
                        "",
                    ),
                )
            )
        )
        for metadata in source_metadata
        if (
            isinstance(
                metadata,
                dict,
            )
            and metadata.get(
                "file_path",
                metadata.get(
                    "source"
                ),
            )
        )
    }

    used_source_paths.update(
        _normalize_path(source)
        for source in function_sources
    )

    citations = [
        citation
        for citation in citations
        if (
            re.sub(
                r":\d+-\d+$",
                "",
                _normalize_path(citation),
            )
            in used_source_paths
        )
    ]

    citations = verify_repository_sources(
        citations,
        function_sources=function_sources,
        target_functions=target_functions,
        metadatas=source_metadata,
        repository_root=repository_path,
    )

    # Restore AST-resolved Python class/method ranges after indexed-chunk verification.
    citations = correct_python_class_citations(
        citations, final_answer, context
    )

    # =========================================================
    # DETERMINISTIC CITATIONS
    # =========================================================

    if (
        deterministic_explanation
        or deterministic_location
    ):
        accurate_citations: List[str] = []

        for function_name in target_functions:
            source_file = FUNCTION_SOURCE_MAP.get(
                function_name
            )

            if not source_file:
                continue

            source_range = (
                find_source_line_range(
                    source_file,
                    function_name,
                )
            )

            if source_range:
                start_line, end_line = (
                    source_range
                )

                citation = (
                    f"{source_file}:"
                    f"{start_line}-"
                    f"{end_line}"
                )

                if (
                    citation
                    not in accurate_citations
                ):
                    accurate_citations.append(
                        citation
                    )

        if accurate_citations:
            citations = accurate_citations

    # Prefer AST-verified symbols explicitly requested by the user.
    # This prevents unrelated methods mentioned by the generated answer
    # from replacing the citation for the requested method.
    question_symbol_citations = find_answer_python_class_citations(
        contextual_question,
        context,
    )
    if question_symbol_citations:
        # Honor an explicitly requested relative file path when supplied.
        requested_path_match = re.search(
            r"(?i)((?:[A-Za-z0-9_.-]+[\\/])+[A-Za-z0-9_.-]+\.py)\b",
            contextual_question,
        )

        selected_symbol_citations = question_symbol_citations

        if requested_path_match:
            requested_path = requested_path_match.group(1).replace(
                "\\", "/"
            ).casefold()
            matching_paths = [
                citation
                for citation in question_symbol_citations
                if re.sub(r":\d+-\d+$", "", citation)
                .replace("\\", "/")
                .casefold()
                .endswith(requested_path)
            ]
            if matching_paths:
                selected_symbol_citations = matching_paths

        # Duplicate uploads can contain the same file under nested folders.
        # If no full path was specified, prefer the shallowest matching copy.
        if not requested_path_match or not matching_paths:
            def citation_path_depth(citation):
                path = re.sub(r":\d+-\d+$", "", citation)
                normalized = path.replace("\\", "/")
                return (normalized.count("/"), len(normalized))

            minimum_depth = min(
                citation_path_depth(citation)
                for citation in selected_symbol_citations
            )
            selected_symbol_citations = [
                citation
                for citation in selected_symbol_citations
                if citation_path_depth(citation) == minimum_depth
            ]

        citations = selected_symbol_citations

    # =========================================================
    # APPEND VERIFIED SOURCES
    # =========================================================

    if citations:
        final_answer = re.sub(
            r"\s*\[[^\]\r\n]+:\d+-\d+\]",
            "",
            final_answer,
        ).rstrip()

        final_answer += (
            "\n\nVerified repository sources: "
            + ", ".join(
                f"[{citation}]"
                for citation in citations
            )
        )

    # =========================================================
    # CONFIDENCE
    # =========================================================

    if unverified_function_claim:
        confidence = 0.1

    elif deterministic_explanation:
        confidence = 0.95

    elif deterministic_location:
        confidence = 0.95

    elif citations:
        confidence = 0.85

    elif context:
        confidence = 0.60

    else:
        confidence = 0.1

    next_steps_suggestion = (
        "You can ask me to trace the relevant "
        "code path, inspect a specific file, "
        "or explain the implementation in "
        "more detail."
    )

    # =========================================================
    # RESPONSE
    # =========================================================

    response = ChatResponse(
        answer=final_answer,
        citations=citations,
        tool_calls_made=tool_calls,
        confidence=confidence,
        next_steps_suggestion=(
            next_steps_suggestion
        ),
    )

    _save_session(
        session_id=session_id,
        question=question,
        answer=response.answer,
        citations=response.citations,
    )

    try:
        chat_response_cache.set(
            repo_id=repo_id,
            indexed_commit=indexed_commit,
            question=contextual_question,
            response=response,
        )

    except Exception:
        logger.debug(
            "Could not cache chat response",
            exc_info=True,
        )

    return response


# ============================================================
# STREAMING
# ============================================================

def agent_stream(
    question: str,
    repo_id: str = "local",
    session_id: Optional[str] = None,
) -> Iterator[str]:
    """
    Server-Sent Events wrapper around agent_loop().
    """

    # ---------------------------------------------------------
    # START EVENT
    # ---------------------------------------------------------

    yield "event: start\ndata: {}\n\n"

    try:
        response = asyncio.run(
            agent_loop(
                question=question,
                repo_id=repo_id,
                session_id=session_id,
            )
        )

        # -----------------------------------------------------
        # EXTRACT RESPONSE DATA
        # -----------------------------------------------------

        if hasattr(response, "answer"):
            answer = response.answer
            citations = response.citations
            tool_calls = response.tool_calls_made
            confidence = response.confidence
            next_steps = response.next_steps_suggestion

        else:
            answer = response.get(
                "answer",
                "",
            )

            citations = response.get(
                "citations",
                [],
            )

            tool_calls = response.get(
                "tool_calls_made",
                [],
            )

            confidence = response.get(
                "confidence",
                0.0,
            )

            next_steps = response.get(
                "next_steps_suggestion",
                None,
            )

        # -----------------------------------------------------
        # MESSAGE EVENT
        # -----------------------------------------------------

        message_data = json.dumps(
            {
                "content": answer,
            }
        )

        yield (
            "event: message\n"
            f"data: {message_data}\n\n"
        )

        # -----------------------------------------------------
        # TOOL EVENTS
        # -----------------------------------------------------

        for tool_name in tool_calls:
            tool_data = json.dumps(
                {
                    "name": tool_name,
                }
            )

            yield (
                "event: tool\n"
                f"data: {tool_data}\n\n"
            )

        # -----------------------------------------------------
        # CITATIONS EVENT
        # -----------------------------------------------------

        citations_data = json.dumps(
            {
                "citations": citations,
            }
        )

        yield (
            "event: citations\n"
            f"data: {citations_data}\n\n"
        )

        # -----------------------------------------------------
        # COMPLETE EVENT
        # -----------------------------------------------------

        complete_data = json.dumps(
            {
                "confidence": confidence,
                "next_steps_suggestion": next_steps,
            }
        )

        yield (
            "event: complete\n"
            f"data: {complete_data}\n\n"
        )

    except Exception:
        logger.exception(
            "Streaming agent execution failed"
        )

        error_data = json.dumps(
            {
                "error": (
                    "Chat processing failed. "
                    "Check the backend logs for details."
                )
            }
        )

        yield (
            "event: error\n"
            f"data: {error_data}\n\n"
        )
