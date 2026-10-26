from __future__ import annotations

from pathlib import Path
from typing import Any

from app.tasks.celery_app import celery_app
from app.indexer.indexer import RepositoryIndexer
from app.services.repo_service import RepoService


def _progress(
    task,
    state: str,
    status: str,
    **extra: Any,
) -> None:
    task.update_state(
        state=state,
        meta={
            "status": status,
            **extra,
        },
    )


@celery_app.task(
    bind=True,
    name="reindex_repository",
)
def reindex_task(
    self,
    repo_path: str,
    repo_id: str = "local",
    base_commit: str | None = None,
    head_commit: str | None = None,
) -> dict[str, Any]:
    """
    Run repository indexing as a background Celery task.

    Supports:
    - full indexing
    - incremental indexing
    - multi-repository identifiers
    - Celery progress reporting
    - repository index-state persistence
    """

    path = Path(repo_path).resolve()

    if not path.exists():
        raise ValueError(
            f"Repository path does not exist: {path}"
        )

    if not path.is_dir():
        raise ValueError(
            f"Repository path is not a directory: {path}"
        )

    _progress(
        self,
        "REINDEXING",
        "Starting repository indexing...",
        repo_id=repo_id,
        repo_path=str(path),
    )

    try:
        indexer = RepositoryIndexer(
            str(path),
            repo_id=repo_id,
        )

        if base_commit and head_commit:
            _progress(
                self,
                "REINDEXING",
                "Running incremental indexing...",
                repo_id=repo_id,
                base_commit=base_commit,
                head_commit=head_commit,
            )

            result = indexer.incremental_index(
                base_commit=base_commit,
                head_commit=head_commit,
            )

            indexed_commit = head_commit

        else:
            _progress(
                self,
                "REINDEXING",
                "Running full repository indexing...",
                repo_id=repo_id,
            )

            result = indexer.index()

            indexed_commit = None

            try:
                indexed_commit = (
                    indexer.repo_manager.get_current_commit()
                )
            except Exception:
                try:
                    from app.indexer.repo import RepoManager

                    indexed_commit = (
                        RepoManager(
                            str(path)
                        ).get_current_commit()
                    )
                except Exception:
                    indexed_commit = None

        repo_service = RepoService()

        repository = repo_service.mark_indexed(
            repo_id=repo_id,
            commit_sha=indexed_commit,
        )

        _progress(
            self,
            "REINDEX_COMPLETE",
            "Repository indexing complete.",
            repo_id=repo_id,
            indexed_commit=indexed_commit,
            result=result,
        )

        return {
            "status": "completed",
            "repo_id": repo_id,
            "repo_path": str(path),
            "indexed_commit": indexed_commit,
            "repository_status": repository.status,
            "result": result,
        }

    except Exception as exc:
        _progress(
            self,
            "REINDEX_FAILED",
            "Repository indexing failed.",
            repo_id=repo_id,
            error=type(exc).__name__,
        )

        raise