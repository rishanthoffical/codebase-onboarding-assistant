from __future__ import annotations

import hmac
import os
import secrets
import time
from typing import Dict, Tuple

import redis
from fastapi import Header, HTTPException, Request
from passlib.context import CryptContext


pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
)


# ============================================================
# Password helpers
# ============================================================

def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(
    plain_password: str,
    hashed_password: str,
) -> bool:
    return pwd_context.verify(
        plain_password,
        hashed_password,
    )


def get_password_hash(password: str) -> str:
    return hash_password(password)


# ============================================================
# Authentication
# ============================================================

def is_auth_enabled() -> bool:
    value = os.getenv(
        "AUTH_ENABLED",
        "false",
    )

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def get_auth_secret() -> str:
    return os.getenv(
        "AUTH_SECRET",
        "",
    ).strip()


def generate_api_token() -> str:
    return secrets.token_urlsafe(32)


def verify_api_token(
    token: str | None,
) -> bool:
    if not token:
        return False

    expected = get_auth_secret()

    if not expected:
        return False

    return hmac.compare_digest(
        token,
        expected,
    )


def require_api_auth(
    authorization: str | None = Header(
        default=None,
        alias="Authorization",
    ),
) -> None:
    """
    Require Bearer authentication when AUTH_ENABLED=true.
    """

    if not is_auth_enabled():
        return

    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authentication required.",
            headers={
                "WWW-Authenticate": "Bearer"
            },
        )

    scheme, _, token = authorization.partition(" ")

    if (
        scheme.lower() != "bearer"
        or not token
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication scheme.",
            headers={
                "WWW-Authenticate": "Bearer"
            },
        )

    if not verify_api_token(token):
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token.",
            headers={
                "WWW-Authenticate": "Bearer"
            },
        )


# ============================================================
# Redis connection
# ============================================================

_redis_client: redis.Redis | None = None


def get_redis_client() -> redis.Redis | None:
    """
    Return a Redis client.

    Redis failures are handled gracefully by returning None.
    No secret values are logged or exposed.
    """

    global _redis_client

    if _redis_client is not None:
        return _redis_client

    redis_url = (
        os.getenv(
            "REDIS_URL",
            "redis://localhost:6379/0",
        )
        .strip()
    )

    if not redis_url:
        return None

    try:
        client = redis.Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=1,
            socket_timeout=1,
        )

        # Verify connection once.
        client.ping()

        _redis_client = client

        return _redis_client

    except Exception:
        return None


# ============================================================
# Local fallback rate limiter
# ============================================================

class InMemoryRateLimiter:
    """
    Local fallback rate limiter.

    Redis is preferred because it works across multiple
    backend workers.

    This fallback keeps local development functional when
    Redis is unavailable.
    """

    def __init__(self) -> None:
        self._requests: Dict[
            Tuple[str, str],
            list[float],
        ] = {}

    def check(
        self,
        client_id: str,
        bucket: str,
        limit: int,
        window_seconds: int,
    ) -> None:
        now = time.monotonic()

        key = (
            client_id,
            bucket,
        )

        timestamps = self._requests.get(
            key,
            [],
        )

        cutoff = now - window_seconds

        timestamps = [
            timestamp
            for timestamp in timestamps
            if timestamp > cutoff
        ]

        if len(timestamps) >= limit:
            raise HTTPException(
                status_code=429,
                detail=(
                    "Rate limit exceeded. "
                    "Please try again later."
                ),
                headers={
                    "Retry-After": str(
                        window_seconds
                    )
                },
            )

        timestamps.append(now)

        self._requests[key] = timestamps


rate_limiter = InMemoryRateLimiter()


# ============================================================
# Rate-limit configuration
# ============================================================

def parse_rate_limit(
    value: str,
) -> tuple[int, int]:
    try:
        amount_text, period = value.split(
            "/",
            1,
        )

        amount = int(
            amount_text.strip()
        )

        period = period.strip().lower()

    except (
        ValueError,
        AttributeError,
    ):
        raise ValueError(
            f"Invalid rate limit: {value}"
        )

    if amount <= 0:
        raise ValueError(
            "Rate limit amount must be positive."
        )

    windows = {
        "second": 1,
        "minute": 60,
        "hour": 3600,
    }

    if period not in windows:
        raise ValueError(
            "Rate limit period must be "
            "second, minute, or hour."
        )

    return (
        amount,
        windows[period],
    )


# ============================================================
# Redis rate limiter
# ============================================================

def check_redis_rate_limit(
    client_id: str,
    bucket: str,
    limit: int,
    window_seconds: int,
) -> bool:
    """
    Check a rate limit using Redis.

    Returns:
        True  -> Redis was used successfully.
        False -> Redis was unavailable.

    Redis INCR is atomic, which makes this suitable for
    multiple backend workers.
    """

    client = get_redis_client()

    if client is None:
        return False

    key = (
        f"codebase_assistant:"
        f"rate_limit:"
        f"{bucket}:"
        f"{client_id}"
    )

    try:
        count = client.incr(key)

        if count == 1:
            client.expire(
                key,
                window_seconds,
            )

        if count > limit:
            raise HTTPException(
                status_code=429,
                detail=(
                    "Rate limit exceeded. "
                    "Please try again later."
                ),
                headers={
                    "Retry-After": str(
                        window_seconds
                    )
                },
            )

        return True

    except HTTPException:
        raise

    except Exception:
        # Redis is unavailable.
        # Caller will use the local fallback.
        return False


def check_rate_limit(
    client_id: str,
    bucket: str,
    limit_config: str,
) -> None:
    """
    Redis-backed rate limiting with a safe local fallback.

    Example:
        30/minute
        10/minute
    """

    limit, window = parse_rate_limit(
        limit_config
    )

    redis_used = check_redis_rate_limit(
        client_id=client_id,
        bucket=bucket,
        limit=limit,
        window_seconds=window,
    )

    if redis_used:
        return

    # Redis unavailable -> local fallback.
    rate_limiter.check(
        client_id=client_id,
        bucket=bucket,
        limit=limit,
        window_seconds=window,
    )


# ============================================================
# Client identification
# ============================================================

def get_client_id(
    request: Request,
) -> str:
    if (
        request.client
        and request.client.host
    ):
        return request.client.host

    return "unknown-client"


# ============================================================
# API security
# ============================================================

def require_chat_security(
    request: Request,
    authorization: str | None = Header(
        default=None,
        alias="Authorization",
    ),
) -> None:
    require_api_auth(
        authorization=authorization
    )

    client_id = get_client_id(
        request
    )

    chat_rate_limit = os.getenv(
        "CHAT_RATE_LIMIT",
        "30/minute",
    )

    check_rate_limit(
        client_id=client_id,
        bucket="chat",
        limit_config=chat_rate_limit,
    )


def require_api_security(
    request: Request,
    authorization: str | None = Header(
        default=None,
        alias="Authorization",
    ),
) -> None:
    require_api_auth(
        authorization=authorization
    )