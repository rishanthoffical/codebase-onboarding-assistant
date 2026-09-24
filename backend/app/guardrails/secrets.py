import re


SECRET_PATTERNS = [
    r'\bAKIA[0-9A-Z]{16}\b',
    r'(?i)(api[_-]?key\s*[=:]\s*)["\']?[A-Za-z0-9_\-]{16,}["\']?',
    r'(?i)(secret\s*[=:]\s*)["\']?[A-Za-z0-9_\-]{8,}["\']?',
    r'(?i)(password\s*[=:]\s*)["\']?(?!safe_password\b)[^\s"\']{6,}["\']?',
    r'(?i)(token\s*[=:]\s*)["\']?[A-Za-z0-9_\-\.]{16,}["\']?',
    r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
]


def scan_for_secrets(text: str):
    matches = []

    for pattern in SECRET_PATTERNS:
        try:
            matches.extend(re.findall(pattern, text))
        except re.error:
            continue

    return matches


def redact_secrets(text: str) -> str:
    redacted = text

    for pattern in SECRET_PATTERNS:
        try:
            redacted = re.sub(
                pattern,
                "[REDACTED]",
                redacted,
            )
        except re.error:
            continue

    return redacted
