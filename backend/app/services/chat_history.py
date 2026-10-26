from __future__ import annotations

import json
import os
from typing import Any

import redis


class ChatHistoryService:
    """
    Persistent chat history backed by Redis.

    History is isolated by:
        repo_id
        session_id

    No LLM secrets or sensitive tool output should be stored here.
    """

    def __init__(
        self,
        redis_url: str | None = None,
        expiration: int = 86400 * 30,
    ):
        self.redis_url = (
            redis_url
            or os.getenv(
                "REDIS_URL",
                "redis://localhost:6379/0",
            )
        )

        self.expiration = expiration
        self.client = redis.from_url(
            self.redis_url,
            decode_responses=True,
        )

    def _key(
        self,
        session_id: str,
        repo_id: str,
    ) -> str:
        return (
            f"chat_history:"
            f"{repo_id}:"
            f"{session_id}"
        )

    @staticmethod
    def _sanitize_message(
        message: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Store only safe conversational fields.

        Tool traces, raw tool output, prompts, and secrets
        are intentionally excluded.
        """

        role = str(
            message.get(
                "role",
                "",
            )
        )

        content = str(
            message.get(
                "content",
                "",
            )
        )

        if role not in {
            "user",
            "assistant",
        }:
            raise ValueError(
                "Invalid message role"
            )

        return {
            "role": role,
            "content": content,
        }

    def add_message(
        self,
        session_id: str,
        repo_id: str,
        role: str,
        content: str,
    ) -> bool:
        if not session_id:
            return False

        if not repo_id:
            repo_id = "local"

        safe_message = self._sanitize_message(
            {
                "role": role,
                "content": content,
            }
        )

        key = self._key(
            session_id=session_id,
            repo_id=repo_id,
        )

        try:
            self.client.rpush(
                key,
                json.dumps(
                    safe_message,
                    ensure_ascii=False,
                ),
            )

            self.client.expire(
                key,
                self.expiration,
            )

            return True

        except redis.RedisError:
            return False

    def add_exchange(
        self,
        session_id: str,
        repo_id: str,
        user_message: str,
        assistant_message: str,
    ) -> bool:
        if not user_message or not assistant_message:
            return False

        user_saved = self.add_message(
            session_id=session_id,
            repo_id=repo_id,
            role="user",
            content=user_message,
        )

        assistant_saved = self.add_message(
            session_id=session_id,
            repo_id=repo_id,
            role="assistant",
            content=assistant_message,
        )

        return user_saved and assistant_saved

    def get_history(
        self,
        session_id: str,
        repo_id: str = "local",
    ) -> list[dict[str, str]]:
        if not session_id:
            return []

        key = self._key(
            session_id=session_id,
            repo_id=repo_id,
        )

        try:
            raw_messages = self.client.lrange(
                key,
                0,
                -1,
            )
        except redis.RedisError:
            return []

        history: list[dict[str, str]] = []

        for raw_message in raw_messages:
            try:
                parsed = json.loads(raw_message)

                safe_message = self._sanitize_message(
                    parsed
                )

                history.append(
                    {
                        "role": safe_message["role"],
                        "content": safe_message["content"],
                    }
                )

            except (
                ValueError,
                TypeError,
                json.JSONDecodeError,
            ):
                continue

        return history

    def clear_history(
        self,
        session_id: str,
        repo_id: str = "local",
    ) -> bool:
        if not session_id:
            return False

        try:
            self.client.delete(
                self._key(
                    session_id=session_id,
                    repo_id=repo_id,
                )
            )

            return True

        except redis.RedisError:
            return False

    def is_available(self) -> bool:
        try:
            return bool(
                self.client.ping()
            )
        except redis.RedisError:
            return False