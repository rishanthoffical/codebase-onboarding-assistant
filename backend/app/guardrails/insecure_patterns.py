import re


INSECURE_PATTERNS = [
    r'\beval\s*\(',
    r'\bexec\s*\(',
    r'(?i)\b(?:http|ftp|ws)://',
    r'(?i)(?:password|secret|apikey|api_key|token|credential)\s*=\s*["\'](?!safe_password["\'])[^"\']+["\']',
    r'(?i)\b(?:select|insert|update|delete)\b.*\bwhere\b',
    r'(?i)string[-_ ]concatenated\s+sql',
    r'(?i)\buntrusted\b',
]


def scan_for_insecure_patterns(text: str) -> list:
    warnings = []

    for pattern in INSECURE_PATTERNS:
        try:
            if re.search(pattern, text):
                warnings.append(pattern)
        except re.error:
            continue

    return warnings
