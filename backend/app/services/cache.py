from typing import Any
import json
import redis


class Cache:
    def __init__(self, redis_url: str):
        self.redis_client = redis.from_url(redis_url)

    def set(
        self,
        key: str,
        value: Any,
        expiration: int = 3600
    ) -> None:
        self.redis_client.set(
            key,
            json.dumps(value),
            ex=expiration
        )

    def get(self, key: str) -> Any:
        value = self.redis_client.get(key)

        if value:
            return json.loads(value)

        return None

    def delete(self, key: str) -> None:
        self.redis_client.delete(key)

    def exists(self, key: str) -> bool:
        return self.redis_client.exists(key) > 0

    def clear(self) -> None:
        self.redis_client.flushall()