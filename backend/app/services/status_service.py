
import os

import redis
from fastapi import HTTPException

from app.services.vector_store import VectorStore


class StatusService:

    @staticmethod
    def get_readiness_status():
        # Check if Redis and Chroma are reachable
        redis_status = StatusService.check_redis_connection()
        chroma_status = StatusService.check_chroma_connection()

        if not redis_status or not chroma_status:
            unavailable = [
                name
                for name, available in (
                    ("Redis", redis_status),
                    ("Chroma", chroma_status),
                )
                if not available
            ]

            raise HTTPException(
                status_code=503,
                detail=(
                    "Service is not ready: "
                    + ", ".join(unavailable)
                    + " unavailable."
                ),
            )

        return {"status": "ready"}

    @staticmethod
    def check_redis_connection():
        try:
            redis_url = os.getenv(
                "REDIS_URL",
                "redis://localhost:6379/0"
            )

            print(
                "REDIS DEBUG: URL =",
                repr(redis_url),
                flush=True,
            )

            client = redis.from_url(
                redis_url,
                socket_connect_timeout=2,
                socket_timeout=2,
            )

            result = client.ping()

            print(
                "REDIS DEBUG: PING =",
                result,
                flush=True,
            )

            return bool(result)

        except Exception as exc:
            print(
                "REDIS DEBUG: ERROR =",
                repr(exc),
                type(exc).__name__,
                flush=True,
            )

            return False

    @staticmethod
    def check_chroma_connection():
        try:
            VectorStore().count()
            return True

        except Exception:
            return False

    @staticmethod
    def get_index_status(repo_id):
        vector_count = VectorStore().count(repo_id=repo_id)

        return {
            "repo_id": repo_id,
            "vector_count": vector_count,
            "index_freshness": "indexed" if vector_count else "empty",
        }

