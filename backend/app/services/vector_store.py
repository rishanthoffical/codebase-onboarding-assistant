from __future__ import annotations

from functools import wraps
import logging
from pathlib import Path
import time
from typing import Any
from collections.abc import Callable

import chromadb
from filelock import FileLock

from app.config import settings

logger = logging.getLogger(__name__)


def _serialized(operation: Callable) -> Callable:
    """Serialize access to a persistent Chroma directory across processes."""
    @wraps(operation)
    def wrapper(self, *args, **kwargs):
        lock = getattr(self, "_operation_lock", None)
        if lock is None:  # Supports lightweight VectorStore test doubles.
            return operation(self, *args, **kwargs)
        with lock:
            return operation(self, *args, **kwargs)

    return wrapper


class VectorStore:
    """
    Chroma-backed vector store.

    Existing APIs are preserved:
        add_documents()
        search()
        count()

    Additional APIs support incremental indexing:
        delete()
        delete_by_file()
        delete_by_repo()
        get_by_file()
        get_by_repo()
        get_file_paths()
    """

    def __init__(
        self,
        persist_directory: str | None = None,
        collection_name: str = "codebase",
    ):
        self.persist_directory = (
            persist_directory or settings.chroma_persist_directory
        )
        self.collection_name = collection_name

        persist_path = Path(self.persist_directory).resolve()
        persist_path.mkdir(parents=True, exist_ok=True)
        self._operation_lock = FileLock(
            str(persist_path / ".codebase-vector-store.lock")
        )

        with self._operation_lock:
            if settings.chroma_host and persist_directory is None:
                # In Docker, backend and Celery must share one Chroma server.
                # PersistentClient opens a local database in each process and
                # is not a supported multi-process shared-store deployment.
                self.client = chromadb.HttpClient(
                    host=settings.chroma_host,
                    port=settings.chroma_port,
                )
            else:
                self.client = chromadb.PersistentClient(
                    path=self.persist_directory
                )
            self.collection = self.client.get_or_create_collection(
                name=collection_name
            )

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    @_serialized
    def add_documents(
        self,
        documents: list[str],
        ids: list[str],
        metadatas: list[dict],
    ) -> None:
        """
        Insert or update documents.

        Chroma upsert makes this operation idempotent.
        """
        if not documents:
            return

        if not (
            len(documents)
            == len(ids)
            == len(metadatas)
        ):
            raise ValueError(
                "documents, ids and metadatas must have "
                "the same length"
            )

        self.collection.upsert(
            documents=documents,
            ids=ids,
            metadatas=metadatas,
        )

    @_serialized
    def delete(
        self,
        ids: list[str],
    ) -> None:
        """
        Delete specific vector IDs.
        """
        if not ids:
            return

        self.collection.delete(
            ids=ids
        )

    @_serialized
    def delete_by_file(
        self,
        file_path: str,
        repo_id: str = "local",
    ) -> int:
        """
        Delete all chunks belonging to one repository file.

        Returns the number of deleted vectors.
        """
        result = self.collection.get(
            where={
                "$and": [
                    {
                        "repo_id": repo_id
                    },
                    {
                        "file_path": file_path
                    },
                ]
            },
            include=["metadatas"],
        )

        ids = result.get("ids", [])

        if ids:
            self.collection.delete(
                ids=ids
            )

        return len(ids)

    @_serialized
    def delete_by_repo(
        self,
        repo_id: str,
    ) -> int:
        """
        Delete every vector belonging to a repository.
        """
        result = self.collection.get(
            where={
                "repo_id": repo_id
            },
            include=["metadatas"],
        )

        ids = result.get("ids", [])

        if ids:
            self.collection.delete(
                ids=ids
            )

        return len(ids)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def _read_with_id_error_retry(self, operation: Callable, **kwargs: Any) -> Any:
        """Retry one transient Chroma ID lookup failure while holding the store lock."""
        for attempt in range(2):
            try:
                return operation(**kwargs)
            except Exception as exc:
                if attempt or "error finding id" not in str(exc).casefold():
                    raise
                logger.warning(
                    "Retrying Chroma read after transient 'Error finding id'"
                )
                time.sleep(0.05)

        raise RuntimeError("Unreachable Chroma retry state")

    @staticmethod
    def _empty_search_result() -> dict[str, Any]:
        return {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

    @staticmethod
    def _filter_search_result(
        result: dict[str, Any],
        n_results: int,
        repo_id: str | None = None,
        file_path: str | None = None,
    ) -> dict[str, Any]:
        """Apply repository/file isolation to Chroma query results in Python."""
        ids = (result.get("ids") or [[]])[0] or []
        documents = (result.get("documents") or [[]])[0] or []
        metadatas = (result.get("metadatas") or [[]])[0] or []
        distances = (result.get("distances") or [[]])[0] or []

        filtered_ids: list[Any] = []
        filtered_documents: list[Any] = []
        filtered_metadatas: list[Any] = []
        filtered_distances: list[Any] = []

        for index, metadata in enumerate(metadatas):
            if not isinstance(metadata, dict):
                continue
            if repo_id and metadata.get("repo_id") != repo_id:
                continue
            if (
                file_path
                and str(metadata.get("file_path", "")) != file_path
            ):
                continue
            if index >= len(ids) or index >= len(documents):
                continue

            filtered_ids.append(ids[index])
            filtered_documents.append(documents[index])
            filtered_metadatas.append(metadata)
            if index < len(distances):
                filtered_distances.append(distances[index])
            if len(filtered_ids) >= n_results:
                break

        return {
            "ids": [filtered_ids],
            "documents": [filtered_documents],
            "metadatas": [filtered_metadatas],
            "distances": [filtered_distances],
        }

    def _search_without_metadata_filter(
        self,
        query: str,
        n_results: int,
        repo_id: str | None,
        file_path: str | None,
    ) -> dict[str, Any]:
        """Work around Chroma filtered-query ID lookup failures safely.

        Chroma 1.5.x can raise ``Error finding id`` while resolving a
        metadata-filtered HNSW query. Retry without ``where`` and expand the
        nearest-neighbor window until enough matching repository results are
        found or the collection is exhausted. Filtering remains mandatory in
        Python, so another repository's documents are never returned.
        """
        total = int(self.collection.count())
        if total <= 0:
            return self._empty_search_result()

        candidate_count = min(
            total,
            max(n_results * 10, 50),
        )
        while True:
            result = self._read_with_id_error_retry(
                self.collection.query,
                query_texts=[query],
                n_results=candidate_count,
            )
            filtered = self._filter_search_result(
                result,
                n_results=n_results,
                repo_id=repo_id,
                file_path=file_path,
            )

            if (
                len(filtered["ids"][0]) >= n_results
                or candidate_count >= total
            ):
                return filtered

            candidate_count = min(
                total,
                candidate_count * 2,
            )

    @_serialized
    def search(
        self,
        query: str,
        n_results: int = 5,
        repo_id: str | None = None,
        file_path: str | None = None,
    ) -> dict[str, Any]:
        """
        Search the vector store.

        repo_id filters the repository at the Chroma level.

        When file_path is supplied, Chroma narrows the search to that path.
        Repository isolation is always retained. If Chroma's filtered query
        hits its intermittent ID lookup failure, the query is retried without
        a metadata filter and the repository/file constraints are applied in
        Python instead.
        """
        kwargs: dict[str, Any] = {
            "query_texts": [query],
        }

        if file_path:
            file_where: dict[str, Any] = {"file_path": file_path}
            if repo_id:
                file_where = {
                    "$and": [
                        {"repo_id": repo_id},
                        {"file_path": file_path},
                    ]
                }
            try:
                matching = self._read_with_id_error_retry(
                    self.collection.get,
                    where=file_where,
                    include=["metadatas"],
                )
            except Exception as exc:
                if "error finding id" not in str(exc).casefold():
                    raise
                logger.warning(
                    "Chroma file lookup failed with 'Error finding id'; "
                    "retrying with Python-side repository/file filtering"
                )
                return self._search_without_metadata_filter(
                    query,
                    n_results,
                    repo_id,
                    file_path,
                )

            candidate_count = len(matching.get("ids", []))
            if candidate_count == 0:
                return self._empty_search_result()
            kwargs["n_results"] = candidate_count
            kwargs["where"] = file_where
        else:
            kwargs["n_results"] = n_results

        if repo_id and not file_path:
            kwargs["where"] = {
                "repo_id": repo_id
            }

        try:
            result = self._read_with_id_error_retry(
                self.collection.query,
                **kwargs
            )
        except Exception as exc:
            if (
                "error finding id" not in str(exc).casefold()
                or not (repo_id or file_path)
            ):
                raise
            logger.warning(
                "Chroma metadata-filtered query failed with "
                "'Error finding id'; retrying with Python-side filtering"
            )
            return self._search_without_metadata_filter(
                query,
                n_results,
                repo_id,
                file_path,
            )

        if not file_path:
            return result

        return self._filter_search_result(
            result,
            n_results=n_results,
            repo_id=repo_id,
            file_path=file_path,
        )

    @_serialized
    def count(
        self,
        repo_id: str | None = None,
    ) -> int:
        """
        Return total vector count.

        If repo_id is supplied, return only that repository's count.
        """
        if not repo_id:
            return self.collection.count()

        result = self.collection.get(
            where={
                "repo_id": repo_id
            },
            include=[],
        )

        return len(
            result.get(
                "ids",
                [],
            )
        )

    @_serialized
    def get_by_file(
        self,
        file_path: str,
        repo_id: str = "local",
    ) -> dict[str, Any]:
        """
        Retrieve all chunks belonging to one file.
        """
        return self.collection.get(
            where={
                "$and": [
                    {
                        "repo_id": repo_id
                    },
                    {
                        "file_path": file_path
                    },
                ]
            },
            include=[
                "documents",
                "metadatas",
            ],
        )

    @_serialized
    def get_file_paths(
        self,
        repo_id: str,
    ) -> list[str]:
        """
        Return indexed file paths for one repository
        without loading chunks.
        """
        result = self.collection.get(
            where={
                "repo_id": repo_id
            },
            include=[
                "metadatas",
            ],
        )

        return sorted(
            {
                str(metadata["file_path"])
                for metadata in result.get(
                    "metadatas",
                    [],
                )
                if isinstance(metadata, dict)
                and metadata.get("file_path")
            }
        )

    @_serialized
    def get_by_repo(
        self,
        repo_id: str,
    ) -> dict[str, Any]:
        """
        Retrieve all chunks belonging to one repository.
        """
        return self.collection.get(
            where={
                "repo_id": repo_id
            },
            include=[
                "documents",
                "metadatas",
            ],
        )
