import os

import redis
from fastapi import HTTPException

from app.services.vector_store import VectorStore

class StatusService:
    @staticmethod
    def get_health_status():
        return {"status": "healthy"}

    @staticmethod
    def get_readiness_status():
        # Check if Redis and Chroma are reachable
        redis_status = StatusService.check_redis_connection()
        chroma_status = StatusService.check_chroma_connection()
        
        if not redis_status or not chroma_status:
            raise HTTPException(status_code=503, detail="Service is not ready")
        
        return {"status": "ready"}

    @staticmethod
    def check_redis_connection():
        try:
            client = redis.from_url(
                os.getenv(
                    "REDIS_URL",
                    "redis://localhost:6379/0"
                )
            )
            return bool(client.ping())
        except redis.RedisError:
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
        document_count = 0

        try:
            document_count = VectorStore().count()
        except Exception:
            pass

        return {
            "repo_id": repo_id,
            "last_indexed": None,
            "index_freshness": (
                "indexed"
                if document_count > 0
                else "not indexed"
            ),
        }