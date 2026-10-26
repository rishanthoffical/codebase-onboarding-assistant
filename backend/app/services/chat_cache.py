from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Optional

from app.models.chat import ChatResponse
from app.services.cache import Cache


logger = logging.getLogger(__name__)


class ChatResponseCache:
    """
    Redis-backed cache for fully processed ChatResponse objects.

    Cache entries are versioned by:
      - repository ID
      - last indexed commit
      - contextual question
      - LLM/provider configuration

    Redis failures are intentionally non-fatal. The assistant continues
    normally when Redis is unavailable.
    """

    PREFIX = "codebase_assistant:chat_cache"
    DEFAULT_TTL = 3600
    ANSWER_PIPELINE_VERSION = "repository-evidence-v5"

    def __init__(
        self,
        cache_service: Optional[Cache] = None,
    ) -> None:
        self.cache_service = cache_service

        if self.cache_service is None:
            try:
                redis_url = os.getenv(
                    "REDIS_URL",
                    "redis://localhost:6379/0",
                ).strip()

                if redis_url:
                    self.cache_service = Cache(
                        redis_url
                    )
            except Exception:
                self.cache_service = None

    # ------------------------------------------------------------------
    # Configuration fingerprint
    # ------------------------------------------------------------------

    @staticmethod
    def _model_config() -> dict[str, str]:
        """
        Include configuration that can materially change an answer.

        Values themselves are safe configuration identifiers; secrets
        such as API keys are deliberately excluded.
        """

        return {
            "answer_pipeline_version": (
                ChatResponseCache.ANSWER_PIPELINE_VERSION
            ),

            "provider": os.getenv(
                "LLM_PROVIDER",
                "ollama",
            ).strip(),

            "model": os.getenv(
                "OLLAMA_MODEL",
                "gemma4:26b",
            ).strip(),

            "ollama_base_url": os.getenv(
                "OLLAMA_BASE_URL",
                "",
            ).strip(),

            "claude_model": os.getenv(
                "CLAUDE_MODEL",
                "",
            ).strip(),

            "agent_max_steps": os.getenv(
                "AGENT_MAX_STEPS",
                "4",
            ).strip(),
        }

    # ------------------------------------------------------------------
    # Hash helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sha256(value: str) -> str:
        return hashlib.sha256(
            value.encode(
                "utf-8",
                errors="replace",
            )
        ).hexdigest()

    @classmethod
    def _config_hash(cls) -> str:
        serialized = json.dumps(
            cls._model_config(),
            sort_keys=True,
            separators=(",", ":"),
        )

        return cls._sha256(
            serialized
        )[:32]

    @classmethod
    def _question_hash(
        cls,
        question: str,
    ) -> str:
        normalized = " ".join(
            question.strip().split()
        )

        return cls._sha256(
            normalized
        )[:32]

    # ------------------------------------------------------------------
    # Key
    # ------------------------------------------------------------------

    @classmethod
    def build_key(
        cls,
        repo_id: str,
        indexed_commit: str,
        question: str,
    ) -> str:
        """
        Build a deterministic cache key.

        The actual question is hashed so raw user questions are not
        exposed in Redis keys.
        """

        safe_repo_id = (
            repo_id.strip()
            .replace(":", "_")
            .replace(" ", "_")
        )

        safe_commit = (
            indexed_commit.strip()
            or "unindexed"
        )

        question_hash = cls._question_hash(
            question
        )

        config_hash = cls._config_hash()

        return (
            f"{cls.PREFIX}:"
            f"{safe_repo_id}:"
            f"{safe_commit}:"
            f"{question_hash}:"
            f"{config_hash}"
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(
        self,
        repo_id: str,
        indexed_commit: Optional[str],
        question: str,
    ) -> Optional[ChatResponse]:
        """
        Return a cached response when Redis is available.

        Unindexed repositories are not cached because there is no
        stable repository version to bind the response to.
        """

        if self.cache_service is None:
            return None

        if not indexed_commit:
            return None

        key = self.build_key(
            repo_id=repo_id,
            indexed_commit=indexed_commit,
            question=question,
        )

        try:
            value = self.cache_service.get(
                key
            )

            if not isinstance(
                value,
                dict,
            ):
                return None

            return ChatResponse(
                **value
            )

        except Exception:
            logger.debug(
                "Chat response cache read failed",
                exc_info=True,
            )

            return None

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def set(
        self,
        repo_id: str,
        indexed_commit: Optional[str],
        question: str,
        response: ChatResponse,
        expiration: Optional[int] = None,
    ) -> bool:
        """
        Cache a fully processed and guardrail-checked response.

        No response is cached unless the repository has an indexed
        commit.
        """

        if self.cache_service is None:
            return False

        if not indexed_commit:
            return False

        if expiration is None:
            expiration = int(
                os.getenv(
                    "CHAT_CACHE_TTL",
                    str(self.DEFAULT_TTL),
                )
            )

        if expiration <= 0:
            return False

        key = self.build_key(
            repo_id=repo_id,
            indexed_commit=indexed_commit,
            question=question,
        )

        try:
            payload = response.dict()

            # Never cache anything that accidentally contains a secret.
            # The response should already have passed guardrails, but this
            # extra check prevents accidental unsafe cache writes.
            from app.guardrails.pipeline import (
                scan_for_secrets,
            )

            secret_result = scan_for_secrets(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                )
            )

            if getattr(
                secret_result,
                "found",
                False,
            ):
                logger.warning(
                    "Unsafe response rejected by chat cache"
                )
                return False

            self.cache_service.set(
                key,
                payload,
                expiration=expiration,
            )

            return True

        except Exception:
            logger.debug(
                "Chat response cache write failed",
                exc_info=True,
            )

            return False

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete(
        self,
        repo_id: str,
        indexed_commit: Optional[str],
        question: str,
    ) -> bool:
        if self.cache_service is None:
            return False

        if not indexed_commit:
            return False

        key = self.build_key(
            repo_id=repo_id,
            indexed_commit=indexed_commit,
            question=question,
        )

        try:
            self.cache_service.delete(
                key
            )

            return True

        except Exception:
            logger.debug(
                "Chat response cache delete failed",
                exc_info=True,
            )

            return False
