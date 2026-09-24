from __future__ import annotations

import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


# The tools operate against the repository containing this backend.
# backend/app/agent/tools.py -> backend -> repository root.
REPO_ROOT = Path(__file__).resolve().parents[2]


IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    "build",
    "dist",
    "coverage",
    "htmlcov",
    "chroma_db",
    ".idea",
    ".vscode",
}


IGNORED_FILES = {
    ".env",
    ".env.local",
    ".env.development",
    ".env.production",
    ".env.test",
}


SENSITIVE_SUFFIXES = {
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".crt",
    ".cer",
    ".der",
    ".keystore",
}


def _normalize_repo_path(path: str) -> str:
    """
    Normalize repository-relative paths.

    Examples:
        ./app/main.py -> app/main.py
        backend/app/main.py -> app/main.py
        app\\main.py -> app/main.py
    """

    if not isinstance(path, str):
        raise ValueError("Path must be a string.")

    normalized = path.strip().replace("\\", "/")

    while normalized.startswith("./"):
        normalized = normalized[2:]

    # Existing project code sometimes refers to paths from backend/.
    if normalized.startswith("backend/"):
        normalized = normalized[len("backend/"):]

    return normalized


def _is_sensitive_relative_path(relative_path: Path) -> bool:
    """
    Reject environment files and private-key/certificate files.
    """

    if relative_path.name.lower() in IGNORED_FILES:
        return True

    if relative_path.suffix.lower() in SENSITIVE_SUFFIXES:
        return True

    return any(
        part.lower() in {
            ".git",
            ".venv",
            "venv",
            "node_modules",
            "chroma_db",
        }
        for part in relative_path.parts
    )


def _resolve_repo_path(path: str) -> Path:
    """
    Resolve a user-supplied path and ensure it remains inside REPO_ROOT.

    Blocks:
    - absolute Windows paths
    - absolute POSIX paths
    - ../ traversal
    - sensitive files
    """

    normalized = _normalize_repo_path(path)

    if not normalized:
        raise ValueError("Path cannot be empty.")

    candidate = Path(normalized)

    if candidate.is_absolute():
        raise ValueError("Absolute paths are not allowed.")

    try:
        resolved = (REPO_ROOT / candidate).resolve()
        resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError(
            "Path must remain inside the repository."
        ) from exc

    relative = resolved.relative_to(REPO_ROOT)

    if _is_sensitive_relative_path(relative):
        raise ValueError(
            "Access to sensitive repository files is not allowed."
        )

    return resolved


def _is_ignored_file(path: Path) -> bool:
    """
    Check whether a repository file should be excluded from search.
    """

    try:
        relative = path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return True

    if _is_sensitive_relative_path(relative):
        return True

    return any(
        part.lower() in IGNORED_DIRS
        for part in relative.parts
    )


def _iter_repository_files(path_glob: str) -> List[Path]:
    """
    Return safe files matching the requested repository glob.
    """

    if not isinstance(path_glob, str):
        raise ValueError("path_glob must be a string.")

    # Prevent a glob from escaping the repository.
    if Path(path_glob).is_absolute():
        raise ValueError("Absolute glob paths are not allowed.")

    normalized_glob = path_glob.replace("\\", "/")

    if ".." in Path(normalized_glob).parts:
        raise ValueError(
            "Parent-directory traversal is not allowed."
        )

    files = []

    for filepath in REPO_ROOT.glob(normalized_glob):
        if not filepath.is_file():
            continue

        try:
            filepath.resolve().relative_to(REPO_ROOT)
        except ValueError:
            continue

        if _is_ignored_file(filepath):
            continue

        files.append(filepath)

    return files


def grep_repo(
    pattern: str,
    path_glob: str = "**/*",
) -> List[Dict]:
    """
    Search current repository files for a regular-expression pattern.

    Returns structured matches containing:
        file_path
        line_number
        matched_text

    Backward-compatible keys:
        file
        line
        content
    """

    if not isinstance(pattern, str):
        raise ValueError("pattern must be a string.")

    try:
        regex = re.compile(
            pattern,
            re.IGNORECASE,
        )
    except re.error:
        regex = re.compile(
            re.escape(pattern),
            re.IGNORECASE,
        )

    matches: List[Dict] = []

    for filepath in _iter_repository_files(path_glob):

        try:
            with filepath.open(
                "r",
                encoding="utf-8",
                errors="ignore",
            ) as file:

                for line_number, line in enumerate(
                    file,
                    start=1,
                ):

                    if not regex.search(line):
                        continue

                    relative_path = filepath.relative_to(
                        REPO_ROOT
                    ).as_posix()

                    matched_text = line.rstrip("\r\n")

                    matches.append(
                        {
                            "file_path": relative_path,
                            "line_number": line_number,
                            "matched_text": matched_text,

                            # Backward compatibility.
                            "file": relative_path,
                            "line": line_number,
                            "content": matched_text,
                        }
                    )

        except (OSError, UnicodeError):
            continue

    return matches


def read_file(
    path: str,
    line_start: int = 1,
    line_end: int | None = None,
) -> str:
    """
    Read exact lines from the current repository.

    line numbers are 1-based and inclusive.
    """

    if not isinstance(line_start, int):
        raise ValueError("line_start must be an integer.")

    if line_start < 1:
        raise ValueError("line_start must be >= 1.")

    if line_end is not None:

        if not isinstance(line_end, int):
            raise ValueError("line_end must be an integer.")

        if line_end < line_start:
            raise ValueError(
                "line_end must be >= line_start."
            )

    filepath = _resolve_repo_path(path)

    try:
        with filepath.open(
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as file:

            lines = file.readlines()

    except OSError as exc:
        raise ValueError(
            f"Unable to read repository file: {path}"
        ) from exc

    if line_end is None:
        line_end = len(lines)

    return "".join(
        lines[line_start - 1:line_end]
    )


def git_blame(
    path: str,
    line: int,
) -> dict:
    """
    Return Git blame information for one repository line.

    Returns:
        commit
        author
        date
        summary
        line

    Unavailable/uncommitted lines return an empty dictionary.
    """

    if not isinstance(line, int):
        raise ValueError("line must be an integer.")

    if line < 1:
        raise ValueError("line must be >= 1.")

    filepath = _resolve_repo_path(path)

    relative_path = filepath.relative_to(
        REPO_ROOT
    ).as_posix()

    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "blame",
                "-L",
                f"{line},{line}",
                "--",
                relative_path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )

    except (
        subprocess.SubprocessError,
        OSError,
    ):
        return {}

    output = result.stdout.strip()

    if not output:
        return {}

    # Git output normally resembles:
    #
    # <commit> (<author> <date> <line>) <summary>
    #
    # We preserve the complete raw line while extracting useful
    # structured fields where possible.

    first_space = output.find(" ")

    if first_space <= 0:
        return {
            "commit": output,
            "line": line,
            "blame_info": output,
        }

    commit = output[:first_space]

    remainder = output[first_space:].strip()

    author = ""
    date = ""
    summary = ""

    if remainder.startswith("("):

        closing = remainder.find(")")

        if closing != -1:

            metadata = remainder[
                1:closing
            ].strip()

            summary = remainder[
                closing + 1:
            ].strip()

            metadata_parts = metadata.split()

            if metadata_parts:

                # Git blame's author may contain spaces.
                # Date/time is normally near the end.
                date_index = -1

                for index, value in enumerate(
                    metadata_parts
                ):

                    if re.fullmatch(
                        r"\d{4}-\d{2}-\d{2}",
                        value,
                    ):
                        date_index = index
                        break

                if date_index > 0:
                    author = " ".join(
                        metadata_parts[:date_index]
                    )

                    date = " ".join(
                        metadata_parts[
                            date_index:
                        ]
                    )

                else:
                    author = " ".join(
                        metadata_parts
                    )

    return {
        "commit": commit,
        "author": author,
        "date": date,
        "summary": summary,
        "line": line,
        "blame_info": output,
    }