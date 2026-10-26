import re

from dataclasses import dataclass
from typing import Dict, Any, Optional

from app.guardrails.insecure_patterns import (
    scan_for_insecure_patterns,
)


@dataclass
class SecretScanResult:
    found: bool
    redacted: Optional[str]


SECRET_PATTERNS = [
    # AWS-style access keys
    r'\bAKIA[0-9A-Z]{16}\b',

    # API keys / secrets / passwords / tokens
    r'(?i)(api[_-]?key\s*[=:]\s*)["\']?[A-Za-z0-9_\-]{16,}["\']?',
    r'(?i)(secret\s*[=:]\s*)["\']?[A-Za-z0-9_\-]{8,}["\']?',
    r'(?i)(password\s*[=:]\s*)["\']?(?!safe_password\b)[^\s"\']{6,}["\']?',
    r'(?i)(token\s*[=:]\s*)["\']?[A-Za-z0-9_\-\.]{16,}["\']?',

    # Private keys
    r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
]


def scan_for_secrets(text: str) -> SecretScanResult:
    redacted = text
    found = False

    for pattern in SECRET_PATTERNS:
        try:
            redacted, count = re.subn(
                pattern,
                lambda m: (
                    "[REDACTED]"
                    if m.lastindex is None
                    else m.group(1) + "[REDACTED]"
                ),
                redacted,
            )

            if count > 0:
                found = True

        except re.error:
            continue

    return SecretScanResult(
        found=found,
        redacted=redacted if found else None,
    )


SCOPE_KEYWORDS = {
    "code",
    "codebase",
    "repo",
    "repository",
    "project",
    "file",
    "files",
    "function",
    "functions",
    "class",
    "classes",
    "method",
    "methods",
    "variable",
    "variables",
    "insecure",
    "pattern",
    "patterns",
    "scan",
    "scanning",
    "guardrail",
    "guardrails",
    "security",
    "secure",
    "implementation",
    "architecture",
    "api",
    "apis",
    "backend",
    "frontend",
    "database",
    "db",
    "sql",
    "search",
    "vector",
    "rag",
    "retrieval",
    "index",
    "indexing",
    "embedding",
    "embeddings",
    "llm",
    "model",
    "models",
    "gemma",
    "ollama",
    "fastapi",
    "celery",
    "redis",
    "task",
    "tasks",
    "worker",
    "queue",
    "test",
    "testing",
    "tests",
    "bug",
    "bugs",
    "debug",
    "error",
    "errors",
    "exception",
    "exceptions",
    "config",
    "configuration",
    "service",
    "services",
    "module",
    "modules",
    "endpoint",
    "endpoints",
    "route",
    "routes",
    "auth",
    "authentication",
    "authorization",
    "credential",
    "credentials",
    "password",
    "secret",
    "secrets",
    "token",
    "apikey",
    "api_key",
    "application",
    "app",
    "postgres",
    "postgresql",
    "mysql",
    "mongodb",
    "chromadb",
    "chroma",
    "uvicorn",
}


def scope_guardrail(question: str) -> bool:
    """
    Determine whether a user question is related to the repository.

    The guardrail accepts explicit code identifiers, repository terminology,
    and questions that clearly describe code implementation or behavior.
    """

    question_lower = question.lower().strip()

    if not question_lower:
        return False

    # ---------------------------------------------------------
    # 1. Explicit code identifiers
    # ---------------------------------------------------------
    identifiers = re.findall(
        r"\b[A-Za-z_][A-Za-z0-9_]*\b",
        question,
    )

    # snake_case identifiers:
    #   incremental_test_function
    #   scan_for_secrets
    #   repo_update
    if any(
        "_" in identifier and len(identifier) >= 3
        for identifier in identifiers
    ):
        return True

    # CamelCase/class-style identifiers:
    #   RepositoryIndexer
    #   ChatResponse
    #   VectorStore
    if any(
        re.search(r"[a-z][A-Z]", identifier)
        for identifier in identifiers
    ):
        return True

    # ---------------------------------------------------------
    # 2. Explicit function/method calls
    # ---------------------------------------------------------
    if re.search(
        r"\b[A-Za-z_][A-Za-z0-9_]*\s*\(",
        question,
    ):
        return True

    # ---------------------------------------------------------
    # 3. Existing repository scope keywords
    # ---------------------------------------------------------
    words = re.findall(
        r"\b[a-zA-Z_]+\b",
        question_lower,
    )

    if any(
        word in SCOPE_KEYWORDS
        for word in words
    ):
        return True

    # ---------------------------------------------------------
    # 4. Explicit repository/code terminology
    # ---------------------------------------------------------
    repository_patterns = [
        r"\bcodebase\b",
        r"\brepository\b",
        r"\brepo\b",
        r"\bsource\s+code\b",
        r"\bsource\s+file\b",
        r"\bcode\s+file\b",
        r"\bproject\s+code\b",
        r"\bbackend\b",
        r"\bfrontend\b",
        r"\bapi\b",
        r"\bendpoint\b",
        r"\broute\b",
        r"\brouter\b",
        r"\bservice\b",
        r"\bmodule\b",
        r"\bpackage\b",
        r"\bcomponent\b",
        r"\bclass\b",
        r"\bfunction\b",
        r"\bmethod\b",
        r"\bfile\b",
        r"\bimplementation\b",
        r"\bimplemented\b",
        r"\bcode\b",
        r"\btest\b",
        r"\btests\b",
        r"\bbug\b",
        r"\berror\b",
        r"\bexception\b",
        r"\bstack\s*trace\b",
        r"\bdependency\b",
        r"\bdependencies\b",
    ]

    if any(
        re.search(pattern, question_lower)
        for pattern in repository_patterns
    ):
        return True

    # ---------------------------------------------------------
    # 5. Code-specific question patterns
    #
    # These require a code-related noun so that questions such as
    # "What is the weather today?" remain out of scope.
    # ---------------------------------------------------------
    code_question_patterns = [
        r"\bwhere\s+is\b.*\b(?:function|method|class|file|module|code|implemented|defined)\b",
        r"\bwhere\s+are\b.*\b(?:functions|methods|classes|files|modules)\b",
        r"\bwhich\s+file\b",
        r"\bwhich\s+class\b",
        r"\bwhich\s+function\b",
        r"\bwhich\s+method\b",
        r"\bwhat\s+file\b",
        r"\bwhat\s+class\b",
        r"\bwhat\s+function\b",
        r"\bwhat\s+method\b",
        r"\bhow\s+does\b.*\b(?:code|function|method|class|api|service|system|repository|repo)\b",
        r"\bhow\s+do\b.*\b(?:code|functions|methods|classes|apis|services)\b",
        r"\bwhat\s+does\b.*\b(?:function|method|class|api|service|module|code)\b",
        r"\bexplain\b.*\b(?:code|function|method|class|implementation|repository|repo|api|service)\b",
        r"\bdescribe\b.*\b(?:code|function|method|class|implementation|repository|repo|api|service)\b",
    ]

    if any(
        re.search(pattern, question_lower)
        for pattern in code_question_patterns
    ):
        return True

    # Accept common technical inquiry forms without requiring users to name
    # a repository file or use a predefined technical term.
    technical_inquiry_patterns = [
        r"\bwhat\s+(?:is|are)\s+(?:(?:a|an)\s+)?"
        r"[\w.-]+(?:\s+[\w.-]+){0,5}\??$",
        r"\bwhat\s+does\b.+\bdo\??$",
        r"\bwhat\b.+\bis\s+used\??$",
        r"\bhow\s+does\b.+\bwork\??$",
        r"\b(?:explain|describe|trace)\b.+",
        r"\b[\w.-]+\.(?:py|pyi|js|jsx|ts|tsx|java|go|rs|cs|php|rb|"
        r"kt|swift|cpp|c|h|sql|html|css|json|ya?ml|toml|md)\b",
    ]

    if any(
        re.search(pattern, question_lower)
        for pattern in technical_inquiry_patterns
    ):
        return True

    return False

def guardrail_pipeline(
    agent_output: str,
    question: str,
) -> Dict[str, Any]:
    # Scan generated output for secrets.
    secret_result = scan_for_secrets(agent_output)

    output = (
        secret_result.redacted
        if secret_result.found
        and secret_result.redacted is not None
        else agent_output
    )

    # Scan generated output for insecure patterns.
    warnings = scan_for_insecure_patterns(output)

    return {
        "output": output,
        "warnings": warnings,
        "is_safe": not warnings,
    }
