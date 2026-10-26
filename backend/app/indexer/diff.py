from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class FileChange:
    """
    Represents the change status of one repository file.
    """

    file_path: str
    status: str
    old_path: Optional[str] = None


VALID_STATUSES = {
    "NEW",
    "MODIFIED",
    "DELETED",
    "RENAMED",
    "COPIED",
}


def _repository_path() -> str:
    """
    Return the Git repository working directory.

    Docker uses /workspace/repository as the default mounted
    repository. REPOSITORY_PATH can override it.
    """
    return os.getenv(
        "REPOSITORY_PATH",
        "/workspace/repository",
    )


def _run_git_diff(
    base_commit: str,
    head_commit: str,
) -> str:
    """
    Return git name-status information between two commits.
    """
    result = subprocess.run(
        [
            "git",
            "diff",
            "--name-status",
            "--find-renames",
            "--find-copies",
            base_commit,
            head_commit,
        ],
        cwd=_repository_path(),
        capture_output=True,
        text=True,
        check=True,
    )

    return result.stdout


def get_file_changes(
    base_commit: str,
    head_commit: str,
) -> list[FileChange]:
    """
    Return structured file changes between two commits.

    Git status codes are normalized to:
        A -> NEW
        M -> MODIFIED
        D -> DELETED
        R -> RENAMED
        C -> COPIED
    """
    output = _run_git_diff(
        base_commit,
        head_commit,
    )

    changes: list[FileChange] = []

    for raw_line in output.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        parts = line.split("\t")

        if len(parts) < 2:
            continue

        status_code = parts[0].strip()
        status = status_code[:1]

        if status == "A":
            changes.append(
                FileChange(
                    file_path=parts[-1],
                    status="NEW",
                )
            )

        elif status == "M":
            changes.append(
                FileChange(
                    file_path=parts[-1],
                    status="MODIFIED",
                )
            )

        elif status == "D":
            changes.append(
                FileChange(
                    file_path=parts[-1],
                    status="DELETED",
                )
            )

        elif status == "R":
            old_path = parts[1] if len(parts) >= 3 else None
            new_path = parts[-1]

            changes.append(
                FileChange(
                    file_path=new_path,
                    status="RENAMED",
                    old_path=old_path,
                )
            )

        elif status == "C":
            old_path = parts[1] if len(parts) >= 3 else None
            new_path = parts[-1]

            changes.append(
                FileChange(
                    file_path=new_path,
                    status="COPIED",
                    old_path=old_path,
                )
            )

    return changes


def diff_index(
    base_commit: str,
    head_commit: str,
) -> list[str]:
    """
    Backward-compatible API.

    Returns changed file paths exactly like the previous implementation.
    """
    return [
        change.file_path
        for change in get_file_changes(
            base_commit,
            head_commit,
        )
    ]


def get_file_diff(
    file_path: str,
    base_commit: str,
    head_commit: str,
) -> str:
    """
    Return the complete git diff for one file.
    """
    result = subprocess.run(
        [
            "git",
            "diff",
            base_commit,
            head_commit,
            "--",
            file_path,
        ],
        cwd=_repository_path(),
        capture_output=True,
        text=True,
        check=True,
    )

    return result.stdout


def diff_index_with_details(
    base_commit: str,
    head_commit: str,
) -> dict[str, str]:
    """
    Backward-compatible API returning raw diffs.
    """
    changed_files = diff_index(
        base_commit,
        head_commit,
    )

    file_diffs: dict[str, str] = {}

    for file_path in changed_files:
        file_diffs[file_path] = get_file_diff(
            file_path,
            base_commit,
            head_commit,
        )

    return file_diffs


def get_incremental_changes(
    base_commit: str,
    head_commit: str,
) -> dict[str, list[str]]:
    """
    Return files grouped by incremental indexing operation.
    """
    changes = get_file_changes(
        base_commit,
        head_commit,
    )

    result: dict[str, list[str]] = {
        "NEW": [],
        "MODIFIED": [],
        "DELETED": [],
        "RENAMED": [],
        "COPIED": [],
        "UNCHANGED": [],
    }

    for change in changes:
        if change.status in result:
            result[change.status].append(change.file_path)

    return result
