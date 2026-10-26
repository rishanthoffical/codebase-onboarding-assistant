import os
from pathlib import Path

from dotenv import load_dotenv
from celery import Celery
from celery.schedules import crontab

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

celery_app = Celery(
    "codebase_onboarding",
    broker=os.getenv(
        "CELERY_BROKER_URL",
        os.getenv(
            "REDIS_URL",
            "redis://localhost:6379/0",
        ),
    ),
    backend=os.getenv(
        "CELERY_RESULT_BACKEND",
        os.getenv(
            "REDIS_URL",
            "redis://localhost:6379/0",
        ),
    ),
    include=[
        "app.tasks.reindex",
    ],
)
# Run the local repository reindex once every night at 02:00 UTC.
celery_app.conf.beat_schedule = {
    "nightly-reindex-local-repository": {
        "task": "reindex_repository",
        "schedule": crontab(hour=2, minute=0),
        "args": (
            os.getenv(
                "REPOSITORY_PATH",
                str(Path(__file__).resolve().parents[3]),
            ),
            "local",
        ),
    },
}
