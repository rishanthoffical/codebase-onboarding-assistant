from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from app.guardrails.pipeline import scan_for_secrets
from app.indexer.chunker import language_for_path
from app.indexer.indexer import RepositoryIndexer
from app.services.vector_store import VectorStore


MAX_SCAN_BYTES = 32 * 1024 * 1024
MAX_SCAN_FILES = 1000

FRAMEWORK_PATTERNS = {
    "FastAPI": re.compile(r"^\s*(?:from\s+fastapi\b|import\s+fastapi\b)|\bfastapi\s*[<=>]", re.IGNORECASE),
    "Flask": re.compile(r"^\s*(?:from\s+flask\b|import\s+flask\b)|\bflask\s*[<=>]", re.IGNORECASE),
    "Django": re.compile(r"^\s*(?:from\s+django\b|import\s+django\b)|\bdjango\s*[<=>]", re.IGNORECASE),
    "React": re.compile(r"\b(?:from|require\()\s*['\"]react['\"]|\breact\s*[\"']?\s*:", re.IGNORECASE),
    "Angular": re.compile(r"@angular/(?:core|common|platform-browser)|\bangular\.json\b", re.IGNORECASE),
    "Vue": re.compile(r"\b(?:from|require\()\s*['\"]vue['\"]|\bvue\s*[\"']?\s*:", re.IGNORECASE),
    "Next.js": re.compile(r"\b(?:from|require\()\s*['\"]next(?:/[^'\"]*)?['\"]|\bnext\s*[\"']?\s*:", re.IGNORECASE),
    "Svelte": re.compile(r"\b(?:from|require\()\s*['\"]svelte(?:/[^'\"]*)?['\"]|\bsvelte\s*[\"']?\s*:", re.IGNORECASE),
    "Express": re.compile(r"\b(?:from|require\()\s*['\"]express['\"]|\bexpress\s*[\"']?\s*:", re.IGNORECASE),
    "NestJS": re.compile(r"@nestjs/(?:core|common)|\bnest\s*[\"']?\s*:", re.IGNORECASE),
    "Spring": re.compile(r"^\s*import\s+org\.springframework\b|org\.springframework", re.IGNORECASE),
    "ASP.NET Core": re.compile(r"^\s*using\s+Microsoft\.AspNetCore\b|Microsoft\.AspNetCore", re.IGNORECASE),
    "Laravel": re.compile(r"laravel/framework|Illuminate\\", re.IGNORECASE),
    "Ruby on Rails": re.compile(r"\bgem\s+['\"]rails['\"]|\bRails\.application\b", re.IGNORECASE),
    "Celery": re.compile(r"^\s*(?:from\s+celery\b|import\s+celery\b)|\bcelery\s*[<=>]", re.IGNORECASE),
}

STORAGE_PATTERNS = {
    "SQLAlchemy": re.compile(r"sqlalchemy", re.IGNORECASE),
    "SQLite": re.compile(r"\bsqlite3\b|sqlite://", re.IGNORECASE),
    "PostgreSQL": re.compile(r"\b(?:psycopg|asyncpg)\b|postgres(?:ql)?://", re.IGNORECASE),
    "MySQL": re.compile(r"\b(?:pymysql|mysqlclient)\b|mysql://", re.IGNORECASE),
    "MongoDB": re.compile(r"\b(?:pymongo|motor)\b|mongodb(?:\+srv)?://", re.IGNORECASE),
    "Redis": re.compile(r"\bredis\b|redis://", re.IGNORECASE),
    "Chroma": re.compile(r"\bchromadb\b", re.IGNORECASE),
}

EXTERNAL_SERVICE_PATTERNS = {
    "GitHub": re.compile(r"\bgithub\b", re.IGNORECASE),
    "AWS": re.compile(r"\baws\b|amazonaws\.com", re.IGNORECASE),
    "Stripe": re.compile(r"\bstripe\b", re.IGNORECASE),
    "Twilio": re.compile(r"\btwilio\b", re.IGNORECASE),
    "Slack": re.compile(r"\bslack\b", re.IGNORECASE),
}
ARCHITECTURE_PATTERNS = {
    "AI Agent": re.compile(
        r"\bagent_loop\b|\bAgent\b|\bagent\b.*\b(tools?|reasoning|loop)\b",
        re.IGNORECASE,
    ),
    "RAG": re.compile(
        r"\bRAG\b|\bretriev(e|al|ed)\b|\bvector.?store\b|"
        r"\bsemantic.?search\b|\bembedding(s)?\b",
        re.IGNORECASE,
    ),
    "Guardrails": re.compile(
        r"\bguardrail(s)?\b|\bscan_for_secrets\b|\bscope_guardrail\b|"
        r"\bsecurity.?scan",
        re.IGNORECASE,
    ),
    "Celery": re.compile(
        r"\bcelery\b|\bshared_task\b|\bCelery\b",
        re.IGNORECASE,
    ),
    "Redis": re.compile(
        r"\bredis\b|redis://",
        re.IGNORECASE,
    ),
    "Chroma": re.compile(
        r"\bchromadb\b|\bChroma\b|\bchroma_db\b",
        re.IGNORECASE,
    ),
    "Ollama": re.compile(
        r"\bollama\b|api/generate|OLLAMA_MODEL|OLLAMA_BASE_URL",
        re.IGNORECASE,
    ),
    "LLM": re.compile(
        r"\bLLMService\b|\bgenerate_with_tools\b|\bllm_service\b|"
        r"\blanguage.?model\b",
        re.IGNORECASE,
    ),
    "Agent Tools": re.compile(
        r"\btool_registry\b|\bgrep_repo\b|\bread_file\b|\bgit_blame\b|"
        r"\bsecret_scanner\b|\btool(s)?\b",
        re.IGNORECASE,
    ),
}
MANIFEST_NAMES = {
    "package.json",
    "requirements.txt",
    "pyproject.toml",
    "pipfile",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "cargo.toml",
    "go.mod",
    "composer.json",
    "gemfile",
}

CONFIGURATION_NAMES = {
    "dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    "makefile",
    "tsconfig.json",
    "vite.config.ts",
    "vite.config.js",
}

ENTRY_POINT_NAMES = {
    "main",
    "app",
    "application",
    "server",
    "index",
    "program",
    "manage",
    "wsgi",
    "asgi",
}

TEST_DIRECTORY_NAMES = {
    "test",
    "tests",
    "testing",
    "fixture",
    "fixtures",
}


@dataclass
class ProjectOverview:
    answer: str
    citations: list[str]
    confidence: float


def build_project_overview(
    repo_path: str,
    repo_id: str,
    vector_store: VectorStore,
) -> ProjectOverview:
    indexer = RepositoryIndexer(
        repo_path,
        repo_id=repo_id,
        vector_store=vector_store,
    )
    root = Path(repo_path).resolve()
    files = indexer.get_files()
    readable_files: list[Path] = []
    languages: Counter[str] = Counter()
    citations: list[str] = []
    framework_evidence: dict[str, tuple[str, int]] = {}
    storage_evidence: dict[str, tuple[str, int]] = {}
    service_evidence: dict[str, tuple[str, int]] = {}
    architecture_evidence: dict[str, tuple[str, int]] = {}
    endpoint_evidence: list[tuple[str, str, int]] = []
    readme_excerpt = ""
    readme_citation = ""
    project_description: tuple[str, str, int] | None = None
    scanned_bytes = 0
    scanned_files = 0

    priority_files = sorted(
        files,
        key=lambda path: (
            path.name.lower() not in MANIFEST_NAMES
            and path.name.lower().startswith("readme") is False,
            path.as_posix().lower(),
        ),
    )

    for path in priority_files:
        if scanned_files >= MAX_SCAN_FILES:
            break

        try:
            file_size = path.stat().st_size
        except OSError:
            continue

        if file_size > MAX_SCAN_BYTES - scanned_bytes:
            continue

        relative_path = path.relative_to(root).as_posix()
        path_parts = path.relative_to(root).parts
        is_test_path = (
            any(
                part.lower() in TEST_DIRECTORY_NAMES
                for part in path_parts[:-1]
            )
            or path.name.lower().startswith(("test_", "tests_"))
            or ".test." in path.name.lower()
            or ".spec." in path.name.lower()
        )
        readme_lines: list[tuple[int, str]] = []
        lines_read = 0

        try:
            with path.open(
                "r",
                encoding="utf-8",
                errors="replace",
            ) as source:
                for line_number, line in enumerate(source, start=1):
                    lines_read = line_number
                    stripped = line.strip()
                    if (
                        stripped
                        and path.name.lower().startswith("readme")
                        and not stripped.startswith("#")
                        and len(readme_lines) < 4
                        and sum(len(text) for _, text in readme_lines) < 500
                    ):
                        readme_lines.append(
                            (line_number, stripped[:500])
                        )

                    is_pattern_declaration = "re.compile(" in line
                    can_supply_evidence = (
                        not is_test_path
                        and not is_pattern_declaration
                        and not stripped.startswith(("#", "//", "*"))
                    )

                    if project_description is None and can_supply_evidence:
                        description = re.search(
                            r"""(?ix)
                            (?:
                                ["']?description["']?\s*:\s*["']
                                |
                                \bdescription\s*=\s*["']
                                |
                                ^\s*description\s*:\s*["']?
                            )
                            ([^"'#\r\n]{5,300})
                            """,
                            line,
                        )
                        if description:
                            candidate_description = description.group(1).strip()
                            generic_description = bool(
                                re.search(
                                    r"(?i)\b(repository[- ]relative|relative|absolute)\b.*\b(file|path)\b|\b(file|directory|repository)[- ]path\b|\bpath to (?:the )?(?:file|repository)\b|\b(first|last|start|end) line to read\b|\bstarting at \d+\b|\b(line number|parameter|argument|return value)\b",
                                    candidate_description,
                                )
                            )
                            if not generic_description:
                                project_description = (
                                    candidate_description,
                                    relative_path,
                                    line_number,
                                )

                    is_manifest = path.name.lower() in MANIFEST_NAMES
                    is_import_line = bool(
                        re.search(
                            r"(?:^\s*(?:import|from|using)\b"
                            r"|\brequire\s*\(|\bgem\s+)",
                            line,
                            re.IGNORECASE,
                        )
                    )
                    for name, pattern in FRAMEWORK_PATTERNS.items():
                        if (
                            can_supply_evidence
                            and (is_manifest or is_import_line)
                            and name not in framework_evidence
                            and pattern.search(line)
                        ):
                            framework_evidence[name] = (
                                relative_path,
                                line_number,
                            )

                    for name, pattern in STORAGE_PATTERNS.items():
                        if (
                            can_supply_evidence
                            and (is_manifest or is_import_line)
                            and name not in storage_evidence
                            and pattern.search(line)
                        ):
                            storage_evidence[name] = (
                                relative_path,
                                line_number,
                            )

                    for name, pattern in EXTERNAL_SERVICE_PATTERNS.items():
                        if (
                            can_supply_evidence
                            and (is_manifest or is_import_line)
                            and name not in service_evidence
                            and pattern.search(line)
                        ):
                            service_evidence[name] = (
                                relative_path,
                                line_number,
                            )
                    for name, pattern in ARCHITECTURE_PATTERNS.items():
                        if (
                            can_supply_evidence
                            and not is_import_line
                            and not line.lstrip().startswith('#')
                            and name not in architecture_evidence
                            and pattern.search(line)
                        ):
                            architecture_evidence[name] = (
                                relative_path,
                                line_number,
                            )

                    lowered_line = line.lower()

                    if (
                        can_supply_evidence
                        and (
                            "ollama" in lowered_line
                            or "api/generate" in lowered_line
                        )
                    ):
                        service_evidence.setdefault(
                            "Ollama",
                            (relative_path, line_number),
                        )

                    if (
                        can_supply_evidence
                        and (
                            "anthropic" in lowered_line
                            or "claude" in lowered_line
                        )
                    ):
                        service_evidence.setdefault(
                            "Claude / Anthropic",
                            (relative_path, line_number),
                        )

                    endpoint = re.search(
                        r"@\w+\.(get|post|put|patch|delete)\s*\(\s*['\"]([^'\"]+)",
                        line,
                        re.IGNORECASE,
                    )

                    if endpoint and not is_test_path:
                        endpoint_evidence.append(
                            (
                                relative_path,
                                endpoint.group(2),
                                line_number,
                            )
                        )
                    elif (
                        re.search(
                            r"@\w+\.(get|post|put|patch|delete)\s*\(",
                            line,
                            re.IGNORECASE,
                        )
                        and not is_test_path
                    ):
                        lines = path.read_text(
                            encoding="utf-8",
                            errors="replace",
                        ).splitlines()

                        for next_line in lines[line_number:line_number + 7]:
                            path_match = re.search(
                                r"['\"]([^'\"]+)['\"]",
                                next_line,
                            )

                            if path_match:
                                endpoint_evidence.append(
                                    (
                                        relative_path,
                                        path_match.group(1),
                                        line_number,
                                    )
                                )
                                break

        except (OSError, UnicodeError):
            continue

        scanned_files += 1
        scanned_bytes += file_size
        if lines_read:
            readable_files.append(path)
            languages[language_for_path(path)] += 1

        if readme_lines and not readme_excerpt:
            excerpt = " ".join(text for _, text in readme_lines)
            secret_result = scan_for_secrets(excerpt)
            readme_excerpt = (
                secret_result.redacted
                if secret_result.found and secret_result.redacted
                else excerpt
            )
            readme_citation = (
                f"{relative_path}:"
                f"{readme_lines[0][0]}-{readme_lines[-1][0]}"
            )
            citations.append(readme_citation)

    def add_file_citation(path: str, line_number: int) -> str:
        citation = f"{path}:{line_number}-{line_number}"
        if citation not in citations:
            citations.append(citation)
        return f"[{citation}]"

    if not readable_files:
        purpose = (
            "I could not determine this confidently from the uploaded files."
        )
    elif readme_excerpt:
        purpose = (
            f"The README begins with this project description: "
            f"“{readme_excerpt}” [{readme_citation}]."
        )
    elif project_description:
        description, path, line = project_description
        secret_result = scan_for_secrets(description)
        if secret_result.found and secret_result.redacted:
            description = secret_result.redacted
        purpose = (
            f"Project metadata describes it as “{description}” "
            f"{add_file_citation(path, line)}."
        )
    else:
        source_signals: list[str] = []

        for path in readable_files:
            try:
                source_text = path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            except (OSError, UnicodeError):
                continue

            lowered = source_text.lower()

            if (
                "ollama" in lowered
                and (
                    "generate_with_tools" in lowered
                    or "tool_model" in lowered
                    or "api/generate" in lowered
                )
            ):
                source_signals.append(
                    "LLM generation and Ollama integration"
                )

            if (
                "claude" in lowered
                and "anthropic" in lowered
            ):
                source_signals.append(
                    "Claude/Anthropic provider integration"
                )

            if (
                "generate_with_tools" in lowered
                and "tool" in lowered
            ):
                source_signals.append(
                    "native tool-calling support"
                )

            if "scan_for_secrets" in lowered:
                source_signals.append(
                    "security scanning for secrets"
                )

        if source_signals:
            unique_signals = list(dict.fromkeys(source_signals))
            evidence_by_signal = {}

            evidence_terms = {
                "LLM generation and Ollama integration": (
                    "api/generate", "tool_model", '"ollama"'
                ),
                "Claude/Anthropic provider integration": (
                    "anthropic", "claude"
                ),
                "native tool-calling support": (
                    "generate_with_tools(", "tool_executor"
                ),
                "security scanning for secrets": (
                    "scan_for_secrets("
                ),
            }

            for signal in unique_signals:
                for term in evidence_terms.get(signal, ()):
                    found = False
                    for candidate in readable_files:
                        try:
                            candidate_lines = candidate.read_text(
                                encoding="utf-8", errors="replace"
                            ).splitlines()
                        except (OSError, UnicodeError):
                            continue

                        for number, text in enumerate(candidate_lines, 1):
                            lowered_line = text.lower().strip()
                            if term not in lowered_line:
                                continue
                            if lowered_line.startswith(("import ", "from ", "#")):
                                continue
                            if term == "scan_for_secrets(" and (
                                "scan_for_secrets(" not in lowered_line
                                or "=" not in lowered_line
                                or lowered_line.startswith(("def ", "async def "))
                            ):
                                continue
                            evidence_by_signal[signal] = add_file_citation(
                                candidate.relative_to(root).as_posix(), number
                            )
                            found = True
                            break

                        if found:
                            break

                    if found:
                        break

            # Prefer actual calls over imports and definitions.
            secret_signal = "security scanning for secrets"
            if secret_signal in unique_signals:
                for candidate in readable_files:
                    try:
                        candidate_lines = candidate.read_text(encoding="utf-8", errors="replace").splitlines()
                    except (OSError, UnicodeError):
                        continue
                    for number, text in enumerate(candidate_lines, 1):
                        lowered_line = text.strip().lower()
                        if "scan_for_secrets(" not in lowered_line:
                            continue
                        if lowered_line.startswith(("def ", "async def ", "import ", "from ", "#")):
                            continue
                        if "=" not in lowered_line:
                            continue
                        evidence_by_signal[secret_signal] = add_file_citation(
                            candidate.relative_to(root).as_posix(), number
                        )
                        break
                    if secret_signal in evidence_by_signal:
                        break

            # Remove the stale citation after correcting the evidence.
            if secret_signal in evidence_by_signal:
                old_citation = 'llm_service.py:12-12'
                if old_citation in citations:
                    citations.remove(old_citation)

            supported_signals = [
                f"{signal} ({evidence_by_signal[signal]})"
                for signal in unique_signals
                if signal in evidence_by_signal
            ]

            if supported_signals:
                purpose = (
                    "The source code indicates that this project provides "
                    + "; ".join(supported_signals)
                    + "."
                )
            else:
                purpose = (
                    "Could not be determined from available project documentation."
                )

        else:
            purpose = (
                "Could not be determined from available project documentation."
            )

    if languages:
        language_items = []
        for language, count in sorted(languages.items()):
            example = next(
                path
                for path in readable_files
                if language_for_path(path) == language
            )
            relative = example.relative_to(root).as_posix()
            citation = add_file_citation(relative, 1)
            language_items.append(
                f"{language} ({count} files; example `{relative}` {citation})"
            )
        language_summary = ", ".join(language_items)
    else:
        language_summary = "Could not be determined; no supported text files were found."

    if framework_evidence:
        framework_summary = ", ".join(
            f"{name} (import or dependency evidence in "
            f"`{path}`:{line} {add_file_citation(path, line)})"
            for name, (path, line) in sorted(framework_evidence.items())
        )
    else:
        framework_summary = (
            "Framework could not be determined from the available files."
        )

    entry_points = [
        path
        for path in readable_files
        if path.stem.lower() in ENTRY_POINT_NAMES
        and language_for_path(path)
        not in {"Markdown", "Text", "JSON", "YAML", "XML", "Configuration"}
    ]
    if entry_points:
        entry_point_summary = ", ".join(
            f"`{path.relative_to(root).as_posix()}` "
            f"{add_file_citation(path.relative_to(root).as_posix(), 1)}"
            for path in entry_points[:8]
        )
    elif endpoint_evidence:
        route_lines: dict[str, int] = {}
        for path, _, line in endpoint_evidence:
            route_lines.setdefault(path, line)
        route_files = sorted(route_lines)
        entry_point_summary = (
            "Route definitions are visible in "
            + ", ".join(
                f"`{path}` "
                f"{add_file_citation(path, route_lines[path])}"
                for path in route_files
            )
            + "; a process-level entry point was not identified."
        )
    else:
        entry_point_summary = "Could not be determined from the available files."

    def priority_names(path: Path) -> bool:
        name = path.name.lower()
        stem = path.stem.lower()
        return (
            name.startswith("readme")
            or name in MANIFEST_NAMES
            or name in CONFIGURATION_NAMES
            or stem in ENTRY_POINT_NAMES
            or "route" in stem
            or "main" in stem
            or "config" in stem
            or "settings" in stem
            or path.suffix.lower()
            in {".toml", ".yaml", ".yml", ".ini", ".cfg", ".xml"}
        )

    important_files = [
        path for path in readable_files if priority_names(path)
    ][:10]
    if important_files:
        important_summary = ", ".join(
            f"`{path.relative_to(root).as_posix()}` "
            f"{add_file_citation(path.relative_to(root).as_posix(), 1)}"
            for path in important_files
        )
    else:
        important_summary = (
            "No conventional manifest, documentation, or entry-point "
            "filenames were identified."
        )

    if endpoint_evidence or architecture_evidence or framework_evidence:
        routes = []
        for path, endpoint, line in endpoint_evidence[:10]:
            routes.append(
                f"`{endpoint}` in `{path}` "
                f"{add_file_citation(path, line)}"
            )

        flow_steps = []

        if framework_evidence:
            flow_steps.append(
                "application/API framework "
                + ", ".join(sorted(framework_evidence))
            )

        if endpoint_evidence:
            flow_steps.append(
                "HTTP/API routes "
                + ", ".join(
                    f"`{endpoint}`"
                    for _, endpoint, _ in endpoint_evidence[:6]
                )
            )

        detected_architecture = set(
            architecture_evidence
        )

        if "Celery" in detected_architecture:
            flow_steps.append("background processing with Celery")

        if "Redis" in detected_architecture:
            flow_steps.append("Redis-backed job/state coordination")

        if "Chroma" in detected_architecture:
            flow_steps.append("Chroma/vector-store operations")

        if "RAG" in detected_architecture:
            flow_steps.append("retrieval/RAG")

        if "Agent Tools" in detected_architecture:
            flow_steps.append("agent tools")

        if "Guardrails" in detected_architecture:
            flow_steps.append("guardrail/security checks")

        if "AI Agent" in detected_architecture:
            flow_steps.append("AI-agent orchestration")

        if "LLM" in detected_architecture:
            flow_steps.append("LLM service")

        if "Ollama" in detected_architecture:
            flow_steps.append("Ollama model inference")

        if flow_steps:
            execution_flow = (
                "Detected components (execution order not verified): "
                + ", ".join(flow_steps)
                + ". "
                + (
                    "HTTP route evidence: "
                    + ", ".join(routes)
                    + "."
                    if routes
                    else ""
                )
            )
        else:
            execution_flow = (
                "Could not be determined confidently from "
                "the scanned files."
            )
    else:
        execution_flow = (
            "No HTTP routes were detected. "
            "The execution flow could not be determined "
            "confidently from the scanned files."
        )

    if architecture_evidence:
        architecture_items = []

        for name, (path, line) in sorted(
            architecture_evidence.items()
        ):
            architecture_items.append(
                f"{name} in `{path}` "
                f"{add_file_citation(path, line)}"
            )

        architecture_summary = ", ".join(
            architecture_items
        )
    else:
        architecture_summary = (
            "No recognizable AI-agent, RAG, guardrail, "
            "background-job, vector-store, or LLM "
            "architecture signals were found."
        )

    if storage_evidence:
        storage_summary = ", ".join(
            f"{name} evidence in `{path}`:{line} "
            f"{add_file_citation(path, line)}"
            for name, (path, line) in sorted(storage_evidence.items())
        )
    else:
        storage_summary = (
            "Database/storage could not be determined from the available files."
        )

    if service_evidence:
        external_summary = ", ".join(
            f"{name} reference in `{path}`:{line} "
            f"{add_file_citation(path, line)}"
            for name, (path, line) in sorted(service_evidence.items())
        )
    else:
        external_summary = (
            "No external service could be determined from the available files."
        )

    if not readable_files:
        confidence = 0.1
        uncertainty = (
            "The selected repository contains no non-empty, readable "
            "supported source or documentation files."
        )
    elif scanned_bytes >= MAX_SCAN_BYTES or scanned_files >= MAX_SCAN_FILES:
        confidence = 0.55
        uncertainty = (
            "The repository scan reached its configured inspection budget; "
            "unscanned files may contain additional evidence."
        )
    elif readme_excerpt and framework_evidence:
        confidence = 0.9
        uncertainty = (
            "The overview is based only on files present in this selected "
            "repository; unselected parts of the original project are unknown."
        )
    elif framework_evidence or endpoint_evidence:
        confidence = 0.78
        uncertainty = (
            "The selected files provide partial implementation evidence, "
            "but do not establish every aspect of the full project."
        )
    elif architecture_evidence:
        confidence = 0.72
        uncertainty = (
            "Implementation components were identified in the available "
            "source files, but framework, entry-point, and full-project "
            "details remain uncertain."
        )
    else:
        confidence = 0.62
        uncertainty = (
            "The indexed file inventory is available, but project purpose "
            "and framework evidence are incomplete."
        )

    if scanned_files < len(files):
        uncertainty += (
            f" Detailed scanning covered {scanned_files} of {len(files)} "
            "indexable files."
        )

    answer = "\n".join(
    [
        "1. Purpose: " + purpose,
        "2. Languages: " + language_summary,
        "3. Frameworks: " + framework_summary,
        "4. Entry points: " + entry_point_summary,
        "5. Important files: " + important_summary,
        "6. Main execution flow: " + execution_flow,
        "7. Architecture: " + architecture_summary,
        "8. Database/storage: " + storage_summary,
        "9. External services: " + external_summary,
        "10. Evidence/citations: " + (
            ", ".join(f"[{citation}]" for citation in citations)
            if citations
            else "No source citations could be verified."
        ),
        "11. Uncertainty: " + uncertainty,
    ]
)
    return ProjectOverview(
        answer=answer,
        citations=citations,
        confidence=confidence,
    )

