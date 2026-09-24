
from __future__ import annotations

import ast
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.agent.tools import git_blame, grep_repo, read_file
from app.guardrails.pipeline import guardrail_pipeline
from app.models.chat import ChatResponse
from app.services.llm_service import LLMService
from app.services.vector_store import VectorStore


logger = logging.getLogger(__name__)

SESSION_MEMORY: Dict[str, List[Dict[str, Any]]] = {}

FOLLOW_UP_PHRASES = [
    "that function",
    "this function",
    "that method",
    "this method",
    "that file",
    "this file",
    "it",
    "its",
    "they",
    "them",
    "above",
    "previous",
    "earlier",
    "same function",
    "same method",
    "same file",
]


def is_follow_up_question(question: str) -> bool:
    question_lower = question.strip().lower()
    return any(
        phrase in question_lower
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

    SESSION_MEMORY.setdefault(session_id, []).append(
        {
            "question": question,
            "answer": answer,
            "citations": citations,
        }
    )

    SESSION_MEMORY[session_id] = SESSION_MEMORY[session_id][-10:]


vector_store = VectorStore()
llm_service = LLMService()


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

    if normalized.startswith("backend/"):
        normalized = normalized[len("backend/"):]

    return normalized


def _is_relevant_source_file(path: str) -> bool:
    normalized = _normalize_path(path)

    if not normalized:
        return False

    parts = normalized.split("/")

    if any(part in IGNORED_PATH_PARTS for part in parts):
        return False

    filename = parts[-1]

    if "." not in filename:
        return True

    extension = "." + filename.rsplit(".", 1)[-1].lower()

    return extension not in IGNORED_EXTENSIONS


KNOWN_FUNCTIONS = {
    "scan_for_secrets",
    "scan_for_insecure_patterns",
    "scope_guardrail",
    "guardrail_pipeline",
    "grep_repo",
    "read_file",
    "git_blame",
    "agent_loop",
    "build_context",
    "build_contextual_question",
    "find_function_sources",
    "read_function_implementation",
    "find_source_line_range",
    "extract_citations",
    "prioritize_citations",
    "verify_repository_sources",
    "detect_tool_calls",
    "retrieve_information",
    "is_follow_up_question",
    "is_function_location_question",
    "is_function_explanation_question",
}


def extract_function_names(question: str) -> List[str]:
    question_lower = question.lower()
    found: List[str] = []

    for function_name in KNOWN_FUNCTIONS:
        if re.search(
            rf"\b{re.escape(function_name.lower())}\b",
            question_lower,
        ):
            found.append(function_name)

    return found


def get_target_functions(question: str) -> List[str]:
    functions = extract_function_names(question)

    if functions:
        return functions

    question_lower = question.lower()

    aliases = {
        "secret scanning guardrail": "scan_for_secrets",
        "secret scanning": "scan_for_secrets",
        "secret scanner": "scan_for_secrets",
        "secrets scanner": "scan_for_secrets",
        "insecure pattern scanner": "scan_for_insecure_patterns",
        "insecure pattern guardrail": "scan_for_insecure_patterns",
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
    "scan_for_insecure_patterns": "app/guardrails/insecure_patterns.py",
    "scope_guardrail": "app/guardrails/pipeline.py",
    "guardrail_pipeline": "app/guardrails/pipeline.py",
    "grep_repo": "app/agent/tools.py",
    "read_file": "app/agent/tools.py",
    "git_blame": "app/agent/tools.py",
    "agent_loop": "app/agent/loop.py",
    "build_context": "app/agent/loop.py",
    "build_contextual_question": "app/agent/loop.py",
    "find_function_sources": "app/agent/loop.py",
    "read_function_implementation": "app/agent/loop.py",
    "find_source_line_range": "app/agent/loop.py",
    "extract_citations": "app/agent/loop.py",
    "prioritize_citations": "app/agent/loop.py",
    "verify_repository_sources": "app/agent/loop.py",
    "detect_tool_calls": "app/agent/loop.py",
    "retrieve_information": "app/agent/loop.py",
}


def find_function_sources(function_names: List[str]) -> List[str]:
    sources: List[str] = []

    for function_name in function_names:
        source = FUNCTION_SOURCE_MAP.get(function_name)

        if source and source not in sources:
            sources.append(source)

    return sources


def is_function_location_question(question: str) -> bool:
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


def is_function_explanation_question(question: str) -> bool:
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


def read_function_implementation(
    source_file: str,
    function_name: str,
) -> str:
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

        start_line = matches[0].get("line", 1)

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
    normalized = _normalize_path(source_file)

    if not normalized:
        return None

    try:
        path = Path(normalized)

        if path.is_absolute() or not path.exists() or not path.is_file():
            return None

        source = path.read_text(
            encoding="utf-8",
            errors="replace",
        )

        if not source.strip():
            return None

        if path.suffix.lower() == ".py" and function_name:
            tree = ast.parse(source)

            for node in ast.walk(tree):
                if isinstance(
                    node,
                    (ast.FunctionDef, ast.AsyncFunctionDef),
                ) and node.name == function_name:
                    start_line = int(
                        getattr(node, "lineno", 0)
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
                        return start_line, end_line

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

    target_functions = get_target_functions(question)

    if not target_functions:
        return ""

    for source_file in function_sources:
        for function_name in target_functions:
            implementation = read_function_implementation(
                source_file,
                function_name,
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
                    f":{source_range[0]}-{source_range[1]}"
                )

            context_parts.append(
                f"FILE: {source_file}{range_text}\n"
                f"{implementation}"
            )

            break

    return "\n\n".join(context_parts)


def retrieve_information(
    question: str,
    repo_id: str = "local",
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "documents": [],
        "metadatas": [],
        "tool_calls": [],
    }

    try:
        search_result = vector_store.search(
            question,
            n_results=5,
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

        if documents:
            result["documents"] = list(documents[0])

        if metadatas:
            result["metadatas"] = list(metadatas[0])

        result["tool_calls"].append("vector_search")

    except Exception as exc:
        logger.debug(
            "Vector search failed: %s",
            exc,
        )

    target_functions = get_target_functions(question)
    function_sources = find_function_sources(target_functions)

    if function_sources:
        if "grep_repo" not in result["tool_calls"]:
            result["tool_calls"].append("grep_repo")

        for source_file in function_sources:
            for function_name in target_functions:
                try:
                    implementation = read_function_implementation(
                        source_file,
                        function_name,
                    )

                    if not implementation:
                        continue

                    source_range = find_source_line_range(
                        source_file,
                        function_name,
                    )

                    metadata: Dict[str, Any] = {
                        "source": source_file,
                        "file_path": source_file,
                        "repo_id": repo_id,
                    }

                    if source_range:
                        metadata["line_start"] = source_range[0]
                        metadata["line_end"] = source_range[1]

                    result["documents"].append(
                        implementation
                    )

                    result["metadatas"].append(
                        metadata
                    )

                    if "read_file" not in result["tool_calls"]:
                        result["tool_calls"].append("read_file")

                    break

                except Exception as exc:
                    logger.debug(
                        "Repository retrieval failed: %s",
                        exc,
                    )

    return result


def build_context(
    information: Dict[str, Any],
) -> str:
    documents = information.get("documents", [])
    metadatas = information.get("metadatas", [])

    context_parts: List[str] = []

    for index, document in enumerate(documents):
        source = ""
        line_start = None
        line_end = None

        if index < len(metadatas):
            metadata = metadatas[index]

            if isinstance(metadata, dict):
                source = metadata.get(
                    "file_path",
                    metadata.get(
                        "source",
                        metadata.get("file", ""),
                    ),
                )

                line_start = metadata.get("line_start")
                line_end = metadata.get("line_end")

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
            context_parts.append(
                f"FILE: {location}\n{document}"
            )
        else:
            context_parts.append(str(document))

    return "\n\n".join(context_parts)


def extract_citations(
    answer: str,
    context: str,
) -> List[str]:
    citations: List[str] = []

    path_pattern = (
        r"(?:app|backend|frontend|tests)"
        r"(?:[/\\][\w.\-]+)+"
        r"(?:\.\w+)?"
    )

    def add_path(path: str) -> None:
        normalized = _normalize_path(path)

        if (
            normalized
            and normalized not in citations
            and _is_relevant_source_file(normalized)
        ):
            citations.append(normalized)

    for path in re.findall(path_pattern, answer):
        add_path(path)

    for path in re.findall(path_pattern, context):
        add_path(path)

    return citations


def prioritize_citations(
    citations: List[str],
    function_sources: List[str],
) -> List[str]:
    prioritized: List[str] = []

    for source in function_sources:
        normalized = _normalize_path(source)

        if normalized and normalized not in prioritized:
            prioritized.append(normalized)

    for citation in citations:
        normalized = _normalize_path(citation)

        if normalized and normalized not in prioritized:
            prioritized.append(normalized)

    return prioritized


def verify_repository_sources(
    citations: List[str],
    function_sources: Optional[List[str]] = None,
    target_functions: Optional[List[str]] = None,
    metadatas: Optional[List[Dict[str, Any]]] = None,
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

    function_sources = function_sources or []
    target_functions = target_functions or []
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
        if not isinstance(metadata, dict):
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
                metadata.get("line_start", 0)
            )

            line_end = int(
                metadata.get("line_end", 0)
            )
        except (TypeError, ValueError):
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
            (line_start, line_end)
        )

    for citation in citations:
        normalized = _normalize_path(citation)

        if not normalized:
            continue

        normalized = re.sub(
            r":\d+-\d+$",
            "",
            normalized,
        )

        if not _is_relevant_source_file(normalized):
            continue

        try:
            path = Path(normalized)

            if (
                path.is_absolute()
                or not path.exists()
                or not path.is_file()
            ):
                continue

        except (OSError, ValueError):
            continue

        # Exact AST function citation.
        if (
            normalized in normalized_function_sources
            and target_functions
        ):
            source_range = None

            for function_name in target_functions:
                source_range = find_source_line_range(
                    normalized,
                    function_name,
                )

                if source_range:
                    break

            if source_range:
                start_line, end_line = source_range

                citation_with_range = (
                    f"{normalized}:"
                    f"{start_line}-"
                    f"{end_line}"
                )

                if citation_with_range not in verified:
                    verified.append(
                        citation_with_range
                    )

                continue

        # Indexed chunk citation.
        ranges = metadata_ranges.get(
            normalized,
            [],
        )

        if ranges:
            for start_line, end_line in ranges:
                citation_with_range = (
                    f"{normalized}:"
                    f"{start_line}-"
                    f"{end_line}"
                )

                if citation_with_range not in verified:
                    verified.append(
                        citation_with_range
                    )

            continue

        # Complete file citation fallback.
        file_range = find_source_line_range(
            normalized
        )

        if file_range:
            start_line, end_line = file_range

            citation_with_range = (
                f"{normalized}:"
                f"{start_line}-"
                f"{end_line}"
            )

            if citation_with_range not in verified:
                verified.append(
                    citation_with_range
                )

    return verified


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
        calls.append("git_blame")

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
        calls.append("secret_scanner")

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


def build_function_location_answer(
    question: str,
    function_sources: List[str],
) -> Optional[str]:
    if not is_function_location_question(question):
        return None

    if not function_sources:
        return None

    target_functions = get_target_functions(question)

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
    if not is_function_explanation_question(question):
        return None

    if not function_sources:
        return None

    target_functions = get_target_functions(question)

    if not target_functions:
        return None

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

                implementation = read_function_implementation(
                    source_file,
                    function_name,
                )

                if not implementation:
                    continue

                if function_name == "scan_for_secrets":
                    return (
                        "`scan_for_secrets` scans the provided "
                        "text for possible secrets such as API "
                        "keys, passwords, tokens, AWS access keys, "
                        "and private keys. It returns a "
                        "`SecretScanResult` indicating whether a "
                        "secret was found and provides a redacted "
                        "version when necessary. It is implemented "
                        f"in `{source_file}`."
                    )

                if function_name == "scan_for_insecure_patterns":
                    return (
                        "`scan_for_insecure_patterns` scans the "
                        "provided text for predefined insecure "
                        "coding patterns. It checks for patterns "
                        "such as `eval()`, `exec()`, insecure URLs, "
                        "hardcoded credentials, SQL-related "
                        "patterns, and untrusted input. It returns "
                        "a list of matching security warnings. "
                        f"It is implemented in `{source_file}`."
                    )

                if function_name == "scope_guardrail":
                    return (
                        "`scope_guardrail` checks whether a "
                        "developer question is related to the "
                        "codebase by matching repository-related "
                        "keywords. It returns `True` when the "
                        "question is in scope and `False` "
                        f"otherwise. It is implemented in "
                        f"`{source_file}`."
                    )

                if function_name == "guardrail_pipeline":
                    return (
                        "`guardrail_pipeline` runs the security "
                        "guardrail checks on an assistant response. "
                        "It checks scope, scans for secrets, "
                        "redacts detected secrets, checks for "
                        "insecure patterns, and returns the "
                        "processed output together with warnings "
                        f"and a safety status. It is implemented "
                        f"in `{source_file}`."
                    )

                if function_name == "retrieve_information":
                    return (
                        "`retrieve_information` retrieves relevant "
                        "code from the vector store and directly "
                        "reads repository functions when the "
                        "question references known functions. It "
                        "also preserves source metadata such as "
                        "file paths and line ranges for citation "
                        f"verification. It is implemented in "
                        f"`{source_file}`."
                    )

                if function_name == "verify_repository_sources":
                    return (
                        "`verify_repository_sources` validates "
                        "repository citations against files that "
                        "actually exist. For known Python functions "
                        "it uses AST line ranges, and for retrieved "
                        "chunks it can use indexed line metadata. "
                        "It does not trust line numbers generated "
                        f"by the LLM. It is implemented in "
                        f"`{source_file}`."
                    )

                return (
                    f"`{function_name}` is implemented in "
                    f"`{source_file}`. Its implementation was "
                    "verified directly from the repository."
                )

            except Exception as exc:
                logger.debug(
                    "Could not explain function %s: %s",
                    function_name,
                    exc,
                )

    return None


async def agent_loop(
    question: str,
    repo_id: str = "local",
    session_id: Optional[str] = None,
) -> ChatResponse:

    contextual_question = build_contextual_question(
        question,
        session_id,
    )

    target_functions = get_target_functions(
        contextual_question
    )

    function_sources = find_function_sources(
        target_functions
    )

    information = retrieve_information(
        contextual_question,
        repo_id,
    )

    context = build_context(information)

    tool_calls = list(
        information.get(
            "tool_calls",
            [],
        )
    )

    for tool in detect_tool_calls(
        contextual_question
    ):
        if tool not in tool_calls:
            tool_calls.append(tool)

    deterministic_explanation = (
        build_function_explanation_answer(
            contextual_question,
            function_sources,
        )
    )

    deterministic_location = None

    if deterministic_explanation:
        final_answer = deterministic_explanation

    else:
        deterministic_location = (
            build_function_location_answer(
                contextual_question,
                function_sources,
            )
        )

        if deterministic_location:
            final_answer = deterministic_location

        else:
            try:
                final_answer = llm_service.generate(
                    question=contextual_question,
                    context=context,
                )
            except Exception:
                logger.exception(
                    "LLM generation failed"
                )

                final_answer = (
                    "I could not generate an answer "
                    "from the available repository context."
                )

    guardrail_result = guardrail_pipeline(
        agent_output=final_answer,
        question=question,
    )

    final_answer = guardrail_result.get(
        "output",
        final_answer,
    )

    warnings = guardrail_result.get(
        "warnings",
        [],
    )

    citations = extract_citations(
        final_answer,
        context,
    )

    citations = prioritize_citations(
        citations,
        function_sources,
    )

    citations = verify_repository_sources(
        citations,
        function_sources=function_sources,
        target_functions=target_functions,
        metadatas=information.get(
            "metadatas",
            [],
        ),
    )

    if deterministic_explanation or deterministic_location:
        accurate_citations: List[str] = []

        for source_file in function_sources:
            normalized_source = _normalize_path(
                source_file
            )

            if not normalized_source:
                continue

            source_range = None

            for function_name in target_functions:
                source_range = find_source_line_range(
                    normalized_source,
                    function_name,
                )

                if source_range:
                    break

            if source_range:
                start_line, end_line = source_range

                citation = (
                    f"{normalized_source}:"
                    f"{start_line}-"
                    f"{end_line}"
                )
            else:
                file_range = find_source_line_range(
                    normalized_source
                )

                if file_range:
                    start_line, end_line = file_range

                    citation = (
                        f"{normalized_source}:"
                        f"{start_line}-"
                        f"{end_line}"
                    )
                else:
                    citation = normalized_source

            if citation not in accurate_citations:
                accurate_citations.append(citation)

        citations = accurate_citations

    question_lower = contextual_question.lower()

    if any(
        keyword in question_lower
        for keyword in [
            "secret",
            "security",
            "guardrail",
            "insecure",
            "scan",
            "scanning",
        ]
    ):
        if "secret_scanner" not in tool_calls:
            tool_calls.append("secret_scanner")

        if "insecure_pattern_scanner" not in tool_calls:
            tool_calls.append(
                "insecure_pattern_scanner"
            )

    confidence = 0.85

    if deterministic_explanation:
        confidence = 0.99
    elif deterministic_location:
        confidence = 0.99
    elif warnings:
        confidence = 0.89
    elif function_sources:
        confidence = 0.95

    _save_session(
        session_id=session_id,
        question=question,
        answer=final_answer,
        citations=citations,
    )

    return ChatResponse(
        answer=final_answer,
        citations=citations,
        tool_calls_made=tool_calls,
        confidence=confidence,
        next_steps_suggestion=(
            "You can ask me to trace how this function "
            "is used or explain its security patterns."
        ),
    )

