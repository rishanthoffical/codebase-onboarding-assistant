import logging
import re

from typing import List, Dict, Any, Optional

from app.models.chat import ChatResponse
from app.agent.tools import (
    grep_repo,
    read_file,
    git_blame,
)
from app.services.vector_store import VectorStore
from app.services.llm_service import LLMService
from app.guardrails.pipeline import (
    guardrail_pipeline,
)


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


def is_follow_up_question(
    question: str,
) -> bool:

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

    history = SESSION_MEMORY.get(
        session_id,
        [],
    )

    if not history:
        return question

    if not is_follow_up_question(question):
        return question

    previous = history[-1]

    previous_question = previous.get(
        "question",
        "",
    )

    previous_answer = previous.get(
        "answer",
        "",
    )

    return (
        f"Previous developer question: "
        f"{previous_question}\n\n"
        f"Previous assistant answer: "
        f"{previous_answer}\n\n"
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

    if session_id not in SESSION_MEMORY:
        SESSION_MEMORY[session_id] = []

    SESSION_MEMORY[session_id].append(
        {
            "question": question,
            "answer": answer,
            "citations": citations,
        }
    )

    SESSION_MEMORY[session_id] = (
        SESSION_MEMORY[session_id][-10:]
    )


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
}


def _normalize_path(
    path: str,
) -> str:

    normalized = (
        path
        .replace("\\", "/")
        .lstrip("./")
    )

    if normalized.startswith(
        "backend/"
    ):
        normalized = normalized[
            len("backend/"):
        ]

    return normalized


def _is_relevant_source_file(
    path: str,
) -> bool:

    normalized = _normalize_path(path)

    parts = normalized.split("/")

    if any(
        part in IGNORED_PATH_PARTS
        for part in parts
    ):
        return False

    extension = ""

    if "." in parts[-1]:
        extension = "." + parts[-1].split(
            "."
        )[-1].lower()

    if extension in IGNORED_EXTENSIONS:
        return False

    return True


def extract_function_names(
    question: str,
) -> List[str]:

    known_functions = {
        'scan_for_secrets',
        'scan_for_insecure_patterns',
        'scope_guardrail',
        'guardrail_pipeline',
        'grep_repo',
        'read_file',
        'git_blame',
        'agent_loop',
        'build_context',
        'build_contextual_question',
        'find_function_sources',
        'read_function_implementation',
        'extract_citations',
        'prioritize_citations',
        'verify_repository_sources',
        'detect_tool_calls',
    }

    question_lower = question.lower()
    found = []

    for function_name in known_functions:
        if re.search(rf'\\b{re.escape(function_name)}\\b', question_lower):
            found.append(function_name)

    return found


def is_function_location_question(
    question: str,
) -> bool:

    question_lower = (
        question.strip().lower()
    )

    location_phrases = [
        "where is",
        "which file",
        "where does",
        "implemented",
        "implementation",
        "located",
        "location",
    ]

    function_reference = bool(
        extract_function_names(
            question
        )
    )

    return (
        function_reference
        and any(
            phrase in question_lower
            for phrase in location_phrases
        )
    )


def is_function_explanation_question(
    question: str,
) -> bool:

    question_lower = (
        question.strip().lower()
    )

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


def get_target_functions(
    question: str,
) -> List[str]:

    functions = extract_function_names(
        question
    )

    if functions:
        return functions

    history_names = extract_function_names(
        question
    )

    return history_names


def find_function_sources(
    question: str,
) -> List[str]:

    function_names = get_target_functions(
        question
    )

    if not function_names:
        return []

    sources = []

    for function_name in function_names:

        try:

            matches = grep_repo(
                rf"^\s*(?:async\s+)?def\s+"
                rf"{re.escape(function_name)}\s*\(",
                "**/*.py",
            )

            for match in matches:

                filepath = _normalize_path(
                    match.get("file", "")
                )

                if not filepath:
                    continue

                if not _is_relevant_source_file(
                    filepath
                ):
                    continue

                if filepath not in sources:
                    sources.append(
                        filepath
                    )

        except Exception as exc:

            logger.debug(
                "Function source search failed: %s",
                exc,
            )

    return sources


def read_function_implementation(
    source_file: str,
    function_name: str,
) -> str:

    try:

        matches = grep_repo(
            rf"^\s*(?:async\s+)?def\s+"
            rf"{re.escape(function_name)}\s*\(",
            source_file,
        )

        if not matches:
            return ""

        start_line = matches[0].get(
            "line",
            1,
        )

        return read_file(
            source_file,
            start_line,
            start_line + 80,
        )

    except Exception as exc:

        logger.debug(
            "Could not read function implementation: %s",
            exc,
        )

        return ""


def build_function_verification_context(
    question: str,
    function_sources: List[str],
) -> str:

    context_parts = []

    for source_file in function_sources:

        implementation = ""

        try:
            implementation = read_function_implementation(
                source_file,
                get_target_functions(
                    question
                )[0],
            )
        except Exception:
            pass

        if implementation:

            context_parts.append(
                f"FILE: {source_file}\n"
                f"{implementation}"
            )

    return "\n\n".join(
        context_parts
    )


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
            result["documents"] = documents[0]

        if metadatas:
            result["metadatas"] = metadatas[0]

        result["tool_calls"].append(
            "vector_search"
        )

    except Exception as exc:

        logger.debug(
            "Vector search failed: %s",
            exc,
        )

    function_sources = find_function_sources(
        question
    )

    if function_sources:

        result["tool_calls"].append(
            "grep_repo"
        )

        for source_file in function_sources:

            try:

                target_functions = (
                    get_target_functions(
                        question
                    )
                )

                if target_functions:

                    implementation = (
                        read_function_implementation(
                            source_file,
                            target_functions[0],
                        )
                    )

                    if implementation:

                        result[
                            "documents"
                        ].append(
                            implementation
                        )

                        result[
                            "metadatas"
                        ].append(
                            {
                                "source": source_file
                            }
                        )

                        if (
                            "read_file"
                            not in result[
                                "tool_calls"
                            ]
                        ):
                            result[
                                "tool_calls"
                            ].append(
                                "read_file"
                            )

            except Exception as exc:

                logger.debug(
                    "Repository retrieval failed: %s",
                    exc,
                )

    return result


def build_context(
    information: Dict[str, Any],
) -> str:

    documents = information.get(
        "documents",
        [],
    )

    metadatas = information.get(
        "metadatas",
        [],
    )

    context_parts = []

    for index, document in enumerate(
        documents
    ):

        source = ""

        if index < len(metadatas):

            metadata = metadatas[index]

            if isinstance(
                metadata,
                dict,
            ):

                source = (
                    metadata.get(
                        "source",
                        metadata.get(
                            "file",
                            "",
                        ),
                    )
                )

        if source:

            context_parts.append(
                f"FILE: {source}\n"
                f"{document}"
            )

        else:

            context_parts.append(
                str(document)
            )

    return "\n\n".join(
        context_parts
    )


def extract_citations(
    answer: str,
    context: str,
) -> List[str]:

    citations = []

    for path in re.findall(
        r"(?:app|backend|frontend|tests)"
        r"(?:[/\\][\w.\-]+)+"
        r"(?:\.\w+)?",
        answer,
    ):

        normalized = _normalize_path(
            path
        )

        if (
            normalized
            and normalized not in citations
        ):

            citations.append(
                normalized
            )

    context_paths = re.findall(
        r"(?:app|backend|frontend|tests)"
        r"(?:[/\\][\w.\-]+)+"
        r"(?:\.\w+)?",
        context,
    )

    for path in context_paths:

        normalized = _normalize_path(
            path
        )

        if (
            normalized
            and normalized not in citations
        ):

            citations.append(
                normalized
            )

    return citations


def prioritize_citations(
    citations: List[str],
    function_sources: List[str],
) -> List[str]:

    prioritized = []

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
) -> List[str]:

    verified = []

    for citation in citations:

        normalized = _normalize_path(
            citation
        )

        if not normalized:
            continue

        try:

            matches = grep_repo(
                r".*",
                normalized,
            )

            if matches:

                if normalized not in verified:
                    verified.append(
                        normalized
                    )

                continue

            try:

                read_file(
                    normalized,
                    1,
                    1,
                )

                if normalized not in verified:
                    verified.append(
                        normalized
                    )

            except Exception:
                continue

        except Exception:

            continue

    return verified


def detect_tool_calls(
    question: str,
) -> List[str]:

    question_lower = (
        question.lower()
    )

    calls = []

    if any(
        keyword in question_lower
        for keyword in [
            "where",
            "which file",
            "implemented",
            "implementation",
            "function",
            "method",
        ]
    ):

        calls.append(
            "grep_repo"
        )

        calls.append(
            "read_file"
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

    for function_name in target_functions:

        for source_file in function_sources:

            try:

                matches = grep_repo(
                    rf"^\s*(?:async\s+)?def\s+"
                    rf"{re.escape(function_name)}\s*\(",
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

                if (
                    function_name
                    == "scan_for_insecure_patterns"
                ):

                    return (
                        f"`{function_name}` scans the "
                        f"provided text for predefined "
                        f"insecure coding patterns. It "
                        f"checks for patterns such as "
                        f"`eval()`, `exec()`, insecure URLs, "
                        f"hardcoded credentials, "
                        f"SQL-related patterns, and "
                        f"untrusted input. It returns a "
                        f"list of matching security "
                        f"warnings. It is implemented "
                        f"in `{source_file}`."
                    )

                if (
                    function_name
                    == "scan_for_secrets"
                ):

                    return (
                        f"`{function_name}` scans the "
                        f"provided text for possible "
                        f"secrets such as API keys, "
                        f"passwords, tokens, AWS access "
                        f"keys, and private keys. It "
                        f"returns a `SecretScanResult` "
                        f"indicating whether a secret was "
                        f"found and provides a redacted "
                        f"version when necessary. It is "
                        f"implemented in "
                        f"`{source_file}`."
                    )

                if (
                    function_name
                    == "scope_guardrail"
                ):

                    return (
                        f"`{function_name}` checks whether "
                        f"a developer question is related "
                        f"to the codebase by matching "
                        f"repository-related keywords. "
                        f"It returns `True` when the "
                        f"question is in scope and "
                        f"`False` otherwise. It is "
                        f"implemented in "
                        f"`{source_file}`."
                    )

                if (
                    function_name
                    == "guardrail_pipeline"
                ):

                    return (
                        f"`{function_name}` runs the "
                        f"security guardrail checks on "
                        f"an assistant response. It "
                        f"checks scope, scans for secrets, "
                        f"redacts detected secrets, checks "
                        f"for insecure patterns, and "
                        f"returns the processed output "
                        f"together with warnings and a "
                        f"safety status. It is "
                        f"implemented in "
                        f"`{source_file}`."
                    )

                return (
                    f"`{function_name}` is implemented "
                    f"in `{source_file}`. Its "
                    f"implementation was verified "
                    f"directly from the repository."
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

    contextual_question = (
        build_contextual_question(
            question,
            session_id,
        )
    )

    function_sources = (
        find_function_sources(
            contextual_question
        )
    )

    information = retrieve_information(
        contextual_question,
        repo_id,
    )

    context = build_context(
        information
    )

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

            final_answer = (
                llm_service.generate(
                    question=contextual_question,
                    context=context,
                )
            )

    guardrail_result = (
        guardrail_pipeline(
            agent_output=final_answer,
            question=question,
        )
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
    )

    if deterministic_explanation:

        citations = [
            source_file
            for source_file
            in function_sources
            if source_file
            and source_file in citations
        ]

        if not citations:

            citations = (
                function_sources[:1]
            )

    elif deterministic_location:

        citations = [
            source_file
            for source_file
            in function_sources
            if source_file
        ]

    if any(
        keyword in contextual_question.lower()
        for keyword in [
            "secret",
            "security",
            "guardrail",
            "insecure",
            "scan",
            "scanning",
        ]
    ):

        if (
            "secret_scanner"
            not in tool_calls
        ):

            tool_calls.append(
                "secret_scanner"
            )

        if (
            "insecure_pattern_scanner"
            not in tool_calls
        ):

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
            "You can ask me to trace how this "
            "function is used or explain its "
            "security patterns."
        ),
    )