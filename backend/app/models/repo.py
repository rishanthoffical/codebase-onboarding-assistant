from __future__ import annotations

from typing import Optional


class Repo:
    """
    Repository registry model.

    Represents a repository configured for the assistant.
    """

    def __init__(
        self,
        repo_id: str,
        name: str,
        path: str,
        current_commit: Optional[str] = None,
        last_indexed_commit: Optional[str] = None,
        status: str = "registered",
    ):
        self.repo_id = repo_id
        self.name = name
        self.path = path
        self.current_commit = current_commit
        self.last_indexed_commit = last_indexed_commit
        self.status = status

    def dict(self) -> dict:
        return {
            "repo_id": self.repo_id,
            "name": self.name,
            "path": self.path,
            "current_commit": self.current_commit,
            "last_indexed_commit": self.last_indexed_commit,
            "status": self.status,
        }

    def model_dump(self) -> dict:
        return self.dict()

    def __repr__(self) -> str:
        return (
            f"<Repo("
            f"repo_id={self.repo_id!r}, "
            f"name={self.name!r}, "
            f"path={self.path!r}, "
            f"current_commit={self.current_commit!r}, "
            f"last_indexed_commit={self.last_indexed_commit!r}, "
            f"status={self.status!r}"
            f")>"
        )
