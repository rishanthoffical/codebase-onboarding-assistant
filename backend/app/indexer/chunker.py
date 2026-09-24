
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Callable


SUPPORTED_EXTENSIONS = {
    ".py",
    ".java",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".go",
    ".rs",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".html",
    ".css",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".sql",
}


CODE_EXTENSIONS = {
    ".py",
    ".java",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".go",
    ".rs",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".sql",
}


DOC_EXTENSIONS = {
    ".md",
    ".txt",
}


def _line_chunks(
    lines: list[str],
    chunk_size: int = 100,
    overlap: int = 10,
) -> list[tuple[str, int, int]]:
    """
    Generic sliding-window chunking.

    Line numbers are 1-based and refer to the original file.
    """
    if not lines:
        return []

    chunk_size = max(1, chunk_size)
    overlap = max(0, min(overlap, chunk_size - 1))

    chunks = []
    start = 0

    while start < len(lines):
        end = min(start + chunk_size, len(lines))
        content = "\n".join(lines[start:end])

        if content.strip():
            chunks.append(
                (
                    content,
                    start + 1,
                    end,
                )
            )

        if end >= len(lines):
            break

        start = end - overlap

    return chunks


def _python_ast_chunks(code: str) -> list[tuple[str, int, int]]:
    """
    Extract Python functions/classes using the AST.

    Falls back to line chunks when the source cannot be parsed.
    """
    if not code:
        return []

    lines = code.splitlines()

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return _line_chunks(lines)

    chunks: list[tuple[str, int, int]] = []

    nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            ),
        )
    ]

    # Keep top-level structural units where possible.
    nodes.sort(
        key=lambda node: (
            getattr(node, "lineno", 0),
            -getattr(node, "end_lineno", getattr(node, "lineno", 0)),
        )
    )

    covered_ranges: list[tuple[int, int]] = []

    for node in nodes:
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None)

        if not start or not end:
            continue

        # Skip nested nodes when their parent already contains them.
        if any(
            existing_start <= start
            and end <= existing_end
            for existing_start, existing_end in covered_ranges
        ):
            continue

        content = "\n".join(lines[start - 1:end])

        if content.strip():
            chunks.append((content, start, end))
            covered_ranges.append((start, end))

    if not chunks:
        return _line_chunks(lines)

    # Include meaningful module-level code outside structural blocks.
    covered_lines = set()

    for start, end in covered_ranges:
        covered_lines.update(range(start, end + 1))

    remaining: list[tuple[str, int, int]] = []
    current_start = None
    current_lines: list[str] = []

    for number, line in enumerate(lines, start=1):
        if number in covered_lines:
            if current_lines and current_start is not None:
                remaining.append(
                    (
                        "\n".join(current_lines),
                        current_start,
                        number - 1,
                    )
                )
                current_lines = []
                current_start = None
            continue

        if line.strip():
            if current_start is None:
                current_start = number
            current_lines.append(line)
        elif current_lines:
            current_lines.append(line)

    if current_lines and current_start is not None:
        remaining.append(
            (
                "\n".join(current_lines).rstrip(),
                current_start,
                len(lines),
            )
        )

    chunks.extend(
        chunk
        for chunk in remaining
        if chunk[0].strip() and chunk[1] <= chunk[2]
    )

    chunks.sort(key=lambda item: item[1])

    return chunks


def _structural_code_chunks(
    code: str,
    extension: str,
) -> list[tuple[str, int, int]]:
    """
    Lightweight structural chunking for JavaScript/TypeScript and
    other C-style languages.

    This intentionally avoids requiring a heavy parser dependency.
    """
    lines = code.splitlines()

    if not lines:
        return []

    pattern = re.compile(
        r"^\s*(?:"
        r"(?:export\s+)?(?:async\s+)?function\s+\w+"
        r"|(?:export\s+)?class\s+\w+"
        r"|(?:public|private|protected)?\s*(?:static\s+)?class\s+\w+"
        r"|(?:public|private|protected)?\s*(?:static\s+)?(?:async\s+)?[\w<>\[\]]+\s+\w+\s*\("
        r"|(?:fn|func)\s+\w+"
        r"|(?:pub\s+)?fn\s+\w+"
        r")"
    )

    starts = [
        index
        for index, line in enumerate(lines)
        if pattern.search(line)
    ]

    if not starts:
        return _line_chunks(lines)

    chunks = []

    for position, start in enumerate(starts):
        end = (
            starts[position + 1] - 1
            if position + 1 < len(starts)
            else len(lines) - 1
        )

        content = "\n".join(lines[start:end + 1]).strip()

        if content:
            chunks.append(
                (
                    content,
                    start + 1,
                    end + 1,
                )
            )

    return chunks or _line_chunks(lines)


def _markdown_chunks(
    docs: str,
) -> list[tuple[str, int, int]]:
    """
    Chunk Markdown by headings.

    Each heading starts a section that continues until the next
    heading of the same or higher level.
    """
    lines = docs.splitlines()

    if not lines:
        return []

    heading_pattern = re.compile(r"^\s{0,3}(#{1,6})\s+.+$")

    headings = [
        index
        for index, line in enumerate(lines)
        if heading_pattern.match(line)
    ]

    if not headings:
        return _line_chunks(lines, chunk_size=80, overlap=10)

    chunks = []

    # Content before the first heading.
    if headings[0] > 0:
        prefix = "\n".join(lines[:headings[0]]).strip()

        if prefix:
            chunks.append(
                (
                    prefix,
                    1,
                    headings[0],
                )
            )

    for position, start in enumerate(headings):
        end = (
            headings[position + 1] - 1
            if position + 1 < len(headings)
            else len(lines) - 1
        )

        content = "\n".join(lines[start:end + 1]).strip()

        if content:
            chunks.append(
                (
                    content,
                    start + 1,
                    end + 1,
                )
            )

    return chunks


def chunk_code(code: str):
    """
    Preserve the existing line-based API used by the project's tests.

    Empty lines are preserved.
    Leading/trailing newline characters are ignored.
    """
    if not code:
        return []

    lines = code.strip("\n").splitlines()

    return [
        (
            line,
            index + 1,
            index + 1,
        )
        for index, line in enumerate(lines)
    ]


def chunk_docs(docs: str):
    """
    Preserve the existing documentation API used by tests.

    Blank lines after Markdown headings are ignored for compatibility
    with the existing project behavior.
    """
    if not docs:
        return []

    lines = docs.strip().splitlines()

    chunks = []

    line_number = 0
    previous_content = None

    for line in lines:
        if line.strip():
            line_number += 1

            chunks.append(
                (
                    line,
                    line_number,
                    line_number,
                )
            )

            previous_content = line.strip()

        else:
            if (
                previous_content
                and not previous_content.startswith("#")
            ):
                line_number += 1

    return chunks


def chunk_file(
    file_path: str,
    chunk_size: int = 100,
    overlap: int = 10,
):
    """
    Read a supported file and return chunk dictionaries.

    Existing fields are preserved:
        content
        file
        start_line
        end_line

    Additional fields:
        extension
        chunk_type
    """
    path = Path(file_path)

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        return []

    try:
        content = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    except (OSError, UnicodeError):
        return []

    if not content:
        return []

    if extension == ".py":
        raw_chunks = _python_ast_chunks(content)
        chunk_type = "ast"

    elif extension == ".md":
        raw_chunks = _markdown_chunks(content)
        chunk_type = "markdown"

    elif extension in {
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".java",
        ".go",
        ".rs",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
    }:
        raw_chunks = _structural_code_chunks(
            content,
            extension,
        )
        chunk_type = "structural"

    elif extension in DOC_EXTENSIONS:
        raw_chunks = _line_chunks(
            content.splitlines(),
            chunk_size=chunk_size,
            overlap=overlap,
        )
        chunk_type = "document"

    else:
        raw_chunks = _line_chunks(
            content.splitlines(),
            chunk_size=chunk_size,
            overlap=overlap,
        )
        chunk_type = "fallback"

    return [
        {
            "content": chunk,
            "file": str(path),
            "start_line": start_line,
            "end_line": end_line,
            "extension": extension,
            "chunk_type": chunk_type,
        }
        for chunk, start_line, end_line in raw_chunks
        if chunk.strip()
    ]


def get_chunker(extension: str) -> Callable[[str], list]:
    """
    Return a chunking function for a file extension.

    This provides a simple chunker registry without adding
    another dependency.
    """
    extension = extension.lower()

    if extension == ".py":
        return _python_ast_chunks

    if extension == ".md":
        return _markdown_chunks

    if extension in CODE_EXTENSIONS:
        return _structural_code_chunks

    return lambda content: _line_chunks(
        content.splitlines(),
        chunk_size=100,
        overlap=10,
    )
