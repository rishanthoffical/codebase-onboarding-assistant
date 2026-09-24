from __future__ import annotations

from typing import Any

import chromadb


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
    """

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "codebase",
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

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

    def search(
        self,
        query: str,
        n_results: int = 5,
        repo_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Search the vector store.

        repo_id is optional so the existing agent code continues to work.
        """
        kwargs: dict[str, Any] = {
            "query_texts": [query],
            "n_results": n_results,
        }

        if repo_id:
            kwargs["where"] = {
                "repo_id": repo_id
            }

        return self.collection.query(
            **kwargs
        )

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
            result.get("ids", [])
        )

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