from __future__ import annotations

import hashlib
import mimetypes
import subprocess
from pathlib import Path
from typing import Iterable

from app.guardrails.pipeline import scan_for_secrets
from app.indexer.chunker import (
    SUPPORTED_EXTENSIONS,
    SUPPORTED_FILENAMES,
    chunk_file,
    language_for_path,
)
from app.indexer.diff import get_file_changes
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
    "chroma_cache",
    "uploaded_repositories",
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

INDEX_BATCH_SIZE = 100


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
            part.lower() in IGNORED_DIRECTORIES
            or part.lower().startswith("chroma_db_backup")
            for part in relative.parts
        )

    def _is_ignored_file(self, path: Path) -> bool:
        """
        Exclude known secret/configuration and binary files.
        """
        file_name = path.name.lower()
        if file_name in IGNORED_FILE_NAMES:
            return True

        if path.suffix.lower() in IGNORED_FILE_SUFFIXES:
            return True

        if file_name.startswith(".env"):
            return True

        if (
            ".backup" in file_name
            or "-backup" in file_name
            or "_backup" in file_name
            or ".before-" in file_name
            or file_name.endswith(".tmp")
        ):
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

            if path.is_symlink():
                continue

            if self._is_ignored_path(path):
                continue

            if (
                path.suffix.lower() not in SUPPORTED_EXTENSIONS
                and path.name.lower() not in SUPPORTED_FILENAMES
            ):
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

    @staticmethod
    def _redact_secrets(content: str) -> str:
        result = scan_for_secrets(content)
        if result.found and result.redacted is not None:
            return result.redacted
        return content

    def _build_metadata(
        self,
        file_path: Path,
        chunk: dict,
        chunk_id: str,
        last_modified_commit: str | None,
        chunk_index: int,
    ) -> dict:
        relative_path = file_path.relative_to(
            self.repo_path
        ).as_posix()

        content = self._redact_secrets(chunk["content"])
        chunk = {
            **chunk,
            "content": content,
        }

        doc_type = self._get_doc_type(
            file_path,
            content,
        )

        return {
            # New canonical metadata
            "repo_id": self.repo_id,
            "file_path": relative_path,
            "language": language_for_path(file_path),
            "chunk_index": chunk_index,
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
        chunks_created = 0

        def flush_batch() -> None:
            nonlocal documents, ids, metadatas
            if not documents:
                return
            self.vector_store.add_documents(
                documents=documents,
                ids=ids,
                metadatas=metadatas,
            )
            documents = []
            ids = []
            metadatas = []

        discovered_files = [
            path
            for path in self.repo_path.rglob("*")
            if path.is_file()
            and not path.is_symlink()
            and not self._is_ignored_path(path)
        ] if self.repo_path.exists() else []
        files = self.get_files()
        current_commit = self.get_current_commit()
        indexed_files: set[Path] = set()

        for file_path in files:
            chunks = chunk_file(
                str(file_path)
            )

            last_modified_commit = (
                self.get_last_modified_commit(
                    file_path
                )
            )

            for chunk_index, chunk in enumerate(chunks):
                content = self._redact_secrets(
                    chunk.get("content", "")
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
                indexed_files.add(file_path)
                chunks_created += 1

                metadatas.append(
                    self._build_metadata(
                        file_path=file_path,
                        chunk=chunk,
                        chunk_id=chunk_id,
                        last_modified_commit=(
                            last_modified_commit
                            or current_commit
                        ),
                        chunk_index=chunk_index,
                    )
                )

                if len(documents) >= INDEX_BATCH_SIZE:
                    flush_batch()

        flush_batch()

        return {
            "repo_id": self.repo_id,
            "files": len(files),
            "files_discovered": len(discovered_files),
            "files_indexed": len(indexed_files),
            "files_skipped": max(
                0,
                len(discovered_files) - len(indexed_files),
            ),
            "chunks": chunks_created,
            "chunks_created": chunks_created,
            "vectors_created": chunks_created,
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

        NEW:
            Index the new file.

        MODIFIED:
            Delete existing vectors and re-index the file.

        DELETED:
            Delete existing vectors.

        RENAMED:
            Delete vectors for the old path and index the new path.

        COPIED:
            Keep the original vectors and index the new path.
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
                "renamed_files": 0,
                "copied_files": 0,
                "files_processed": 0,
                "chunks_added": 0,
                "vectors_deleted": 0,
            }

        changes = get_file_changes(
            base_commit=base_commit,
            head_commit=head_commit,
        )

        new_changes = [
            change
            for change in changes
            if change.status == "NEW"
        ]

        modified_changes = [
            change
            for change in changes
            if change.status == "MODIFIED"
        ]

        deleted_changes = [
            change
            for change in changes
            if change.status == "DELETED"
        ]

        renamed_changes = [
            change
            for change in changes
            if change.status == "RENAMED"
        ]

        copied_changes = [
            change
            for change in changes
            if change.status == "COPIED"
        ]

        deleted_vectors = 0

        # ---------------------------------------------------------
        # DELETE
        # ---------------------------------------------------------

        for change in deleted_changes:
            deleted_vectors += (
                self.vector_store.delete_by_file(
                    file_path=change.file_path,
                    repo_id=self.repo_id,
                )
            )

        # ---------------------------------------------------------
        # MODIFY
        # ---------------------------------------------------------

        for change in modified_changes:
            deleted_vectors += (
                self.vector_store.delete_by_file(
                    file_path=change.file_path,
                    repo_id=self.repo_id,
                )
            )

        # ---------------------------------------------------------
        # RENAME
        #
        # Delete vectors belonging to the old path.
        # The new path will be indexed below.
        # ---------------------------------------------------------

        for change in renamed_changes:
            if change.old_path:
                deleted_vectors += (
                    self.vector_store.delete_by_file(
                        file_path=change.old_path,
                        repo_id=self.repo_id,
                    )
                )

            # Defensive cleanup if the new path already exists
            # in the vector store.
            deleted_vectors += (
                self.vector_store.delete_by_file(
                    file_path=change.file_path,
                    repo_id=self.repo_id,
                )
            )

        # ---------------------------------------------------------
        # COPY
        #
        # Do NOT delete the original path.
        # Only index the new path.
        # ---------------------------------------------------------

        files_to_index = [
            *new_changes,
            *modified_changes,
            *renamed_changes,
            *copied_changes,
        ]

        chunks_added = 0
        files_processed = 0

        for change in files_to_index:
            relative_path = change.file_path

            file_path = (
                self.repo_path / relative_path
            ).resolve()

            # Prevent path traversal.
            try:
                file_path.relative_to(
                    self.repo_path
                )
            except ValueError:
                continue

            if not file_path.exists():
                continue

            if not file_path.is_file():
                continue

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

            for chunk_index, chunk in enumerate(chunks):
                content = self._redact_secrets(
                    chunk.get("content", "")
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
                        chunk_index=chunk_index,
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
            "new_files": len(new_changes),
            "modified_files": len(modified_changes),
            "deleted_files": len(deleted_changes),
            "renamed_files": len(renamed_changes),
            "copied_files": len(copied_changes),
            "files_processed": files_processed,
            "chunks_added": chunks_added,
            "vectors_deleted": deleted_vectors,
        }