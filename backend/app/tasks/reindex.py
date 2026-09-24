from celery import shared_task
from app.tasks.celery_app import celery_app

from app.indexer.indexer import RepositoryIndexer


@celery_app.task(bind=True, name="reindex_repository")
def reindex_task(self, repo_path: str):
    """
    Run repository indexing as a background Celery task.
    """

    self.update_state(
        state="REINDEXING",
        meta={
            "status": "Starting repository indexing..."
        },
    )

    try:
        self.update_state(
            state="REINDEXING",
            meta={
                "status": "Walking and indexing repository..."
            },
        )

        indexer = RepositoryIndexer(repo_path)

        result = indexer.index()

        self.update_state(
            state="REINDEX_COMPLETE",
            meta={
                "status": "Repository indexing complete.",
                "result": result,
            },
        )

        return result

    except Exception as exc:
        self.update_state(
            state="REINDEX_FAILED",
            meta={
                "status": "Repository indexing failed.",
                "error": str(exc),
            },
        )

        raise