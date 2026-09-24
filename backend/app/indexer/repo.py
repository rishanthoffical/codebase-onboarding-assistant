from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import git


IGNORED_DIRECTORIES = {
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


IGNORED_FILE_NAMES = {
    ".env",
    ".env.local",
    ".env.development",
    ".env.production",
    ".env.test",
}


IGNORED_FILE_SUFFIXES = {
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".crt",
    ".cer",
    ".der",
    ".keystore",
}


BINARY_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".bmp",
    ".ico",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".7z",
    ".rar",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".bin",
    ".class",
    ".jar",
    ".pyc",
    ".whl",
    ".mp3",
    ".mp4",
    ".mov",
    ".avi",
    ".wav",
}


class RepoManager:
    """
    Safe repository manager used by the indexing layer.
    """

    def __init__(
        self,
        repo_path: str,
    ) -> None:
        self.repo_path = Path(
            repo_path
        ).resolve()

        self.repo = git.Repo(
            self.repo_path
        )

    # --------------------------------------------------------------
    # File filtering
    # --------------------------------------------------------------

    def _is_ignored_path(
        self,
        path: Path,
    ) -> bool:
        try:
            relative = path.relative_to(
                self.repo_path
            )
        except ValueError:
            return True

        return any(
            part in IGNORED_DIRECTORIES
            for part in relative.parts
        )

    def _is_ignored_file(
        self,
        path: Path,
    ) -> bool:
        if path.name in IGNORED_FILE_NAMES:
            return True

        if path.name.startswith(".env"):
            return True

        if (
            path.suffix.lower()
            in IGNORED_FILE_SUFFIXES
        ):
            return True

        if (
            path.suffix.lower()
            in BINARY_EXTENSIONS
        ):
            return True

        return False

    def _is_binary(
        self,
        path: Path,
    ) -> bool:
        if (
            path.suffix.lower()
            in BINARY_EXTENSIONS
        ):
            return True

        try:
            with path.open("rb") as file:
                sample = file.read(4096)

            return b"\x00" in sample

        except (
            OSError,
            PermissionError,
        ):
            return True

    def _load_gitignore(self) -> list[str]:
        gitignore = (
            self.repo_path
            / ".gitignore"
        )

        if not gitignore.exists():
            return []

        try:
            return [
                line.strip()
                for line in gitignore.read_text(
                    encoding="utf-8",
                    errors="ignore",
                ).splitlines()
                if line.strip()
                and not line.strip().startswith("#")
                and not line.strip().startswith("!")
            ]
        except OSError:
            return []

    @staticmethod
    def _matches_pattern(
        value: str,
        pattern: str,
    ) -> bool:
        from fnmatch import fnmatch

        return fnmatch(
            value,
            pattern,
        )

    def _matches_gitignore(
        self,
        relative_path: Path,
        patterns: list[str],
    ) -> bool:
        relative = (
            relative_path.as_posix()
        )
        name = relative_path.name

        for pattern in patterns:
            directory_pattern = (
                pattern.endswith("/")
            )

            pattern = pattern.strip()

            if directory_pattern:
                pattern = pattern.rstrip("/")

            pattern = pattern.lstrip("/")

            if directory_pattern:
                if (
                    relative == pattern
                    or relative.startswith(
                        pattern + "/"
                    )
                ):
                    return True

            elif "/" in pattern:
                if self._matches_pattern(
                    relative,
                    pattern,
                ):
                    return True

            elif self._matches_pattern(
                name,
                pattern,
            ):
                return True

        return False

    # --------------------------------------------------------------
    # Repository walking
    # --------------------------------------------------------------

    def walk_repo(
        self,
    ) -> Iterator[Path]:
        """
        Yield safe, indexable repository files.
        """

        if not self.repo_path.exists():
            return

        gitignore_patterns = (
            self._load_gitignore()
        )

        for root, dirs, files in os.walk(
            self.repo_path
        ):
            root_path = Path(root)

            # Prevent traversal into ignored directories.
            dirs[:] = [
                directory
                for directory in dirs
                if directory
                not in IGNORED_DIRECTORIES
            ]

            for filename in files:
                path = (
                    root_path
                    / filename
                )

                if self._is_ignored_path(
                    path
                ):
                    continue

                if self._is_ignored_file(
                    path
                ):
                    continue

                try:
                    relative = path.relative_to(
                        self.repo_path
                    )
                except ValueError:
                    continue

                if self._matches_gitignore(
                    relative,
                    gitignore_patterns,
                ):
                    continue

                if self._is_binary(
                    path
                ):
                    continue

                yield path

    # --------------------------------------------------------------
    # File access
    # --------------------------------------------------------------

    def get_file_content(
        self,
        file_path: str | Path,
    ) -> str:
        path = self._resolve_file_path(
            file_path
        )

        return path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

    def _resolve_file_path(
        self,
        file_path: str | Path,
    ) -> Path:
        path = Path(
            file_path
        )

        if not path.is_absolute():
            path = (
                self.repo_path
                / path
            )

        path = path.resolve()

        try:
            path.relative_to(
                self.repo_path
            )
        except ValueError as exc:
            raise ValueError(
                "File path escapes repository root."
            ) from exc

        if self._is_ignored_path(
            path
        ):
            raise ValueError(
                "Access to ignored repository paths "
                "is not allowed."
            )

        if self._is_ignored_file(
            path
        ):
            raise ValueError(
                "Access to sensitive files "
                "is not allowed."
            )

        if not path.is_file():
            raise FileNotFoundError(
                f"Repository file not found: {file_path}"
            )

        return path

    # --------------------------------------------------------------
    # Git metadata
    # --------------------------------------------------------------

    def get_git_blame(
        self,
        file_path: str | Path,
        line_number: int,
    ) -> str:
        if line_number < 1:
            raise ValueError(
                "line_number must be >= 1"
            )

        path = self._resolve_file_path(
            file_path
        )

        relative = path.relative_to(
            self.repo_path
        ).as_posix()

        return self.repo.git.blame(
            relative,
            "-L",
            f"{line_number},{line_number}",
        )

    def get_last_modified_commit(
        self,
        file_path: str | Path,
    ) -> str | None:
        path = self._resolve_file_path(
            file_path
        )

        relative = path.relative_to(
            self.repo_path
        ).as_posix()

        commit_info = self.repo.git.log(
            "-1",
            "--format=%H",
            "--",
            relative,
        )

        return (
            commit_info.strip()
            if commit_info
            else None
        )

    def is_file_modified(
        self,
        file_path: str | Path,
        commit_sha: str,
    ) -> bool:
        path = self._resolve_file_path(
            file_path
        )

        relative = path.relative_to(
            self.repo_path
        ).as_posix()

        modified_files = (
            self.repo.git.diff(
                commit_sha,
                "--name-only",
            )
        )

        return (
            relative
            in modified_files.splitlines()
            if modified_files
            else False
        )

    # --------------------------------------------------------------
    # Repository information
    # --------------------------------------------------------------

    def get_current_commit(
        self,
    ) -> str | None:
        try:
            return (
                self.repo.head.commit.hexsha
            )
        except (
            ValueError,
            AttributeError,
        ):
            return None

    def get_repo_info(
        self,
    ) -> dict:
        return {
            "repo_path": str(
                self.repo_path
            ),
            "current_commit": (
                self.get_current_commit()
            ),
            "branches": [
                branch.name
                for branch in self.repo.branches
            ],
            "tags": [
                tag.name
                for tag in self.repo.tags
            ],
        }