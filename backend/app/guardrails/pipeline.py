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
    question_lower = question.lower().strip()

    # Explicit function/method/code references are always in scope.
    if re.search(
        r"\b(?:what does|what is|explain|describe|where is|which file)"
        r"\b.*\b[a-zA-Z_][a-zA-Z0-9_]*\s*\(",
        question_lower,
    ):
        return True

    if re.search(
        r"\b[a-zA-Z_][a-zA-Z0-9_]*\s+(?:do|does|mean|refer)\b",
        question_lower,
    ):
        return True

    words = re.findall(
        r"\b[a-zA-Z_]+\b",
        question_lower,
    )

    return any(
        word in SCOPE_KEYWORDS
        for word in words
    )


def guardrail_pipeline(
    agent_output: str,
    question: str,
) -> Dict[str, Any]:

    if not scope_guardrail(question):
        return {
            "output": "This question is off-topic.",
            "warnings": [],
            "is_safe": True,
        }

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

