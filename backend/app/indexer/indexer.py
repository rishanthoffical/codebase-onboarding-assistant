from __future__ import annotations

import hashlib
import mimetypes
import subprocess
from pathlib import Path
from typing import Iterable

from app.indexer.chunker import chunk_file
from app.indexer.diff import get_incremental_changes
from app.services.vector_store import VectorStore


IGNORED_DIRECTORIES = {
    ".git",
    ".venv",
    "venv",
    "env",
    ".env",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    "build",
    "dist",
    "coverage",
    ".coverage",
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


class RepositoryIndexer:
    """
    Repository indexer responsible for discovering files,
    chunking them, generating stable metadata and storing them
    in the active vector store.

    Public API:

        RepositoryIndexer(repo_path)
        indexer.get_files()
        indexer.index()
        indexer.incremental_index()
    """

    def __init__(
        self,
        repo_path: str,
        repo_id: str = "local",
        vector_store: VectorStore | None = None,
    ):
        self.repo_path = Path(repo_path).resolve()
        self.repo_id = repo_id
        self.vector_store = vector_store or VectorStore()

    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------

    def _is_ignored_path(self, path: Path) -> bool:
        """
        Check whether a path contains one of the directories that should
        never be indexed.
        """
        try:
            relative = path.relative_to(self.repo_path)
        except ValueError:
            return True

        return any(
            part in IGNORED_DIRECTORIES
            for part in relative.parts
        )

    def _is_ignored_file(self, path: Path) -> bool:
        """
        Exclude known secret/configuration and binary files.
        """
        if path.name in IGNORED_FILE_NAMES:
            return True

        if path.suffix.lower() in IGNORED_FILE_SUFFIXES:
            return True

        if path.name.startswith(".env"):
            return True

        return False

    def _is_binary(self, path: Path) -> bool:
        """
        Detect obvious binary files without reading the complete file.
        """
        if path.suffix.lower() in BINARY_EXTENSIONS:
            return True

        try:
            with path.open("rb") as file:
                sample = file.read(4096)

            if b"\x00" in sample:
                return True

            return False

        except (OSError, PermissionError):
            return True

    def _load_gitignore(self) -> list[str]:
        """
        Load simple .gitignore patterns.

        This intentionally avoids introducing a new dependency while still
        supporting common repository ignore patterns.
        """
        gitignore = self.repo_path / ".gitignore"

        if not gitignore.exists():
            return []

        try:
            lines = gitignore.read_text(
                encoding="utf-8",
                errors="ignore",
            ).splitlines()
        except OSError:
            return []

        patterns = []

        for line in lines:
            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            if line.startswith("!"):
                # Negated gitignore rules require a full gitignore parser.
                # We skip them safely rather than accidentally indexing
                # ignored content.
                continue

            patterns.append(line)

        return patterns

    def _matches_gitignore(
        self,
        relative_path: Path,
        patterns: list[str],
    ) -> bool:
        """
        Lightweight .gitignore matching for common patterns.
        """
        relative = relative_path.as_posix()
        name = relative_path.name

        for pattern in patterns:
            pattern = pattern.strip()

            if not pattern:
                continue

            directory_pattern = pattern.endswith("/")

            if directory_pattern:
                pattern = pattern.rstrip("/")

            pattern = pattern.lstrip("/")

            if directory_pattern:
                if (
                    relative == pattern
                    or relative.startswith(pattern + "/")
                ):
                    return True

            elif "/" in pattern:
                if self._fnmatch(relative, pattern):
                    return True

            else:
                if self._fnmatch(name, pattern):
                    return True

        return False

    @staticmethod
    def _fnmatch(value: str, pattern: str) -> bool:
        """
        Match using pathlib-style glob semantics.
        """
        from fnmatch import fnmatch

        return fnmatch(value, pattern)

    def get_files(self) -> list[Path]:
        """
        Return indexable repository files.
        """
        if not self.repo_path.exists():
            return []

        gitignore_patterns = self._load_gitignore()
        files: list[Path] = []

        for path in self.repo_path.rglob("*"):
            if not path.is_file():
                continue

            if self._is_ignored_path(path):
                continue

            if self._is_ignored_file(path):
                continue

            try:
                relative_path = path.relative_to(self.repo_path)
            except ValueError:
                continue

            if self._matches_gitignore(
                relative_path,
                gitignore_patterns,
            ):
                continue

            if self._is_binary(path):
                continue

            files.append(path)

        return sorted(files)

    # ------------------------------------------------------------------
    # Git metadata
    # ------------------------------------------------------------------

    def _run_git(
        self,
        *args: str,
    ) -> str | None:
        """
        Execute a git command safely inside the repository.
        """
        try:
            result = subprocess.run(
                [
                    "git",
                    "-C",
                    str(self.repo_path),
                    *args,
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            return result.stdout.strip()

        except (
            OSError,
            subprocess.CalledProcessError,
        ):
            return None

    def get_current_commit(self) -> str | None:
        """
        Return the current repository HEAD SHA.
        """
        return self._run_git(
            "rev-parse",
            "HEAD",
        )

    def get_last_modified_commit(
        self,
        file_path: Path,
    ) -> str | None:
        """
        Return the latest commit SHA affecting a file.
        """
        try:
            relative_path = file_path.relative_to(
                self.repo_path
            ).as_posix()
        except ValueError:
            return None

        return self._run_git(
            "log",
            "-1",
            "--format=%H",
            "--",
            relative_path,
        )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @staticmethod
    def _get_doc_type(
        file_path: Path,
        content: str,
    ) -> str:
        """
        Classify indexed content.
        """
        suffix = file_path.suffix.lower()
        name = file_path.name.lower()

        if name in {
            "readme",
            "readme.md",
            "readme.txt",
        }:
            return "readme"

        if (
            "docs" in {
                part.lower()
                for part in file_path.parts
            }
            or "documentation" in {
                part.lower()
                for part in file_path.parts
            }
        ):
            return "arch_doc"

        if suffix in {
            ".md",
            ".txt",
            ".rst",
        }:
            return "readme"

        if suffix in {
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
            ".sql",
        }:
            stripped = content.strip()

            if (
                stripped.startswith('"""')
                or stripped.startswith("'''")
            ):
                return "docstring"

            return "code"

        return "code"

    def _make_chunk_id(
        self,
        file_path: str,
        line_start: int,
        line_end: int,
        content: str,
    ) -> str:
        """
        Create a deterministic chunk identifier.

        Including the content prevents two different chunks occupying the
        same line range from accidentally sharing an ID.
        """
        raw = (
            f"{self.repo_id}:"
            f"{file_path}:"
            f"{line_start}:"
            f"{line_end}:"
            f"{content}"
        )

        digest = hashlib.sha256(
            raw.encode("utf-8")
        ).hexdigest()[:24]

        return f"{self.repo_id}:{digest}"

    # ------------------------------------------------------------------
    # Indexing helpers
    # ------------------------------------------------------------------

    def _read_file(
        self,
        file_path: Path,
    ) -> str | None:
        try:
            return file_path.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        except (
            OSError,
            UnicodeError,
        ):
            return None

    def _build_metadata(
        self,
        file_path: Path,
        chunk: dict,
        chunk_id: str,
        last_modified_commit: str | None,
    ) -> dict:
        relative_path = file_path.relative_to(
            self.repo_path
        ).as_posix()

        content = chunk["content"]

        doc_type = self._get_doc_type(
            file_path,
            content,
        )

        return {
            # New canonical metadata
            "repo_id": self.repo_id,
            "file_path": relative_path,
            "line_start": int(chunk["start_line"]),
            "line_end": int(chunk["end_line"]),
            "last_modified_commit": (
                last_modified_commit or ""
            ),
            "doc_type": doc_type,
            "chunk_id": chunk_id,

            # Backward-compatible metadata
            "file": relative_path,
            "start_line": int(chunk["start_line"]),
            "end_line": int(chunk["end_line"]),
        }

    # ------------------------------------------------------------------
    # Full indexing
    # ------------------------------------------------------------------

    def index(self) -> dict:
        """
        Index all eligible repository files.

        Returns the basic result structure while adding repository metadata.
        """
        documents: list[str] = []
        ids: list[str] = []
        metadatas: list[dict] = []

        files = self.get_files()
        current_commit = self.get_current_commit()

        for file_path in files:
            chunks = chunk_file(
                str(file_path)
            )

            last_modified_commit = (
                self.get_last_modified_commit(
                    file_path
                )
            )

            for chunk in chunks:
                content = chunk.get(
                    "content",
                    "",
                )

                if not content.strip():
                    continue

                relative_path = file_path.relative_to(
                    self.repo_path
                ).as_posix()

                chunk_id = self._make_chunk_id(
                    file_path=relative_path,
                    line_start=int(
                        chunk["start_line"]
                    ),
                    line_end=int(
                        chunk["end_line"]
                    ),
                    content=content,
                )

                documents.append(content)
                ids.append(chunk_id)

                metadatas.append(
                    self._build_metadata(
                        file_path=file_path,
                        chunk=chunk,
                        chunk_id=chunk_id,
                        last_modified_commit=(
                            last_modified_commit
                            or current_commit
                        ),
                    )
                )

        if documents:
            self.vector_store.add_documents(
                documents=documents,
                ids=ids,
                metadatas=metadatas,
            )

        return {
            "repo_id": self.repo_id,
            "files": len(files),
            "chunks": len(documents),
            "commit": current_commit,
        }

    # ------------------------------------------------------------------
    # Incremental indexing
    # ------------------------------------------------------------------

    def incremental_index(
        self,
        base_commit: str,
        head_commit: str | None = None,
    ) -> dict:
        """
        Incrementally update the vector store using Git changes.

        NEW files are indexed.
        MODIFIED files have their old vectors removed and are re-indexed.
        DELETED files have their vectors removed.
        RENAMED files are handled by deleting the old path and indexing
        the new path.
        COPIED files are indexed as new files.
        """
        if head_commit is None:
            head_commit = self.get_current_commit()

        if not head_commit:
            return {
                "repo_id": self.repo_id,
                "base_commit": base_commit,
                "head_commit": None,
                "new_files": 0,
                "modified_files": 0,
                "deleted_files": 0,
                "files_processed": 0,
                "chunks_added": 0,
                "vectors_deleted": 0,
            }

        changes = get_incremental_changes(
            base_commit=base_commit,
            head_commit=head_commit,
        )

        new_files = changes.get(
            "NEW",
            [],
        )

        modified_files = changes.get(
            "MODIFIED",
            [],
        )

        deleted_files = changes.get(
            "DELETED",
            [],
        )

        files_to_index = [
            *new_files,
            *modified_files,
        ]

        files_to_delete = [
            *deleted_files,
            *modified_files,
        ]

        deleted_vectors = 0

        # --------------------------------------------------------------
        # Remove old vectors
        # --------------------------------------------------------------

        for relative_path in files_to_delete:
            deleted_vectors += (
                self.vector_store.delete_by_file(
                    file_path=relative_path,
                    repo_id=self.repo_id,
                )
            )

        # --------------------------------------------------------------
        # Index new and modified files
        # --------------------------------------------------------------

        chunks_added = 0
        files_processed = 0

        for relative_path in files_to_index:
            file_path = (
                self.repo_path / relative_path
            ).resolve()

            # Prevent paths from escaping the repository.
            try:
                file_path.relative_to(
                    self.repo_path
                )
            except ValueError:
                continue

            # The file may no longer exist.
            if not file_path.exists():
                continue

            if not file_path.is_file():
                continue

            # Apply the same safety filters used by full indexing.
            if self._is_ignored_path(file_path):
                continue

            if self._is_ignored_file(file_path):
                continue

            if self._is_binary(file_path):
                continue

            chunks = chunk_file(
                str(file_path)
            )

            last_modified_commit = (
                self.get_last_modified_commit(
                    file_path
                )
            )

            documents: list[str] = []
            ids: list[str] = []
            metadatas: list[dict] = []

            for chunk in chunks:
                content = chunk.get(
                    "content",
                    "",
                )

                if not content.strip():
                    continue

                chunk_id = self._make_chunk_id(
                    file_path=relative_path,
                    line_start=int(
                        chunk["start_line"]
                    ),
                    line_end=int(
                        chunk["end_line"]
                    ),
                    content=content,
                )

                documents.append(content)
                ids.append(chunk_id)

                metadatas.append(
                    self._build_metadata(
                        file_path=file_path,
                        chunk=chunk,
                        chunk_id=chunk_id,
                        last_modified_commit=(
                            last_modified_commit
                            or head_commit
                        ),
                    )
                )

            if documents:
                self.vector_store.add_documents(
                    documents=documents,
                    ids=ids,
                    metadatas=metadatas,
                )

                chunks_added += len(documents)

            files_processed += 1

        return {
            "repo_id": self.repo_id,
            "base_commit": base_commit,
            "head_commit": head_commit,
            "new_files": len(new_files),
            "modified_files": len(modified_files),
            "deleted_files": len(deleted_files),
            "files_processed": files_processed,
            "chunks_added": chunks_added,
            "vectors_deleted": deleted_vectors,
        }