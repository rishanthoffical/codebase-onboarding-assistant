import os

import pytest
from fastapi import HTTPException

from app.core.security import (
    generate_api_token,
    hash_password,
    verify_password,
    verify_api_token,
    parse_rate_limit,
    InMemoryRateLimiter,
)


def test_password_hash_and_verify():
    password = "test-password"

    hashed = hash_password(password)

    assert hashed != password
    assert verify_password(
        password,
        hashed,
    )
    assert not verify_password(
        "wrong-password",
        hashed,
    )


def test_generate_api_token():
    token = generate_api_token()

    assert isinstance(token, str)
    assert len(token) >= 32


def test_verify_api_token(monkeypatch):
    monkeypatch.setenv(
        "AUTH_SECRET",
        "test-secret",
    )

    assert verify_api_token(
        "test-secret"
    )

    assert not verify_api_token(
        "wrong-secret"
    )

    assert not verify_api_token(
        None
    )


def test_parse_rate_limit():
    assert parse_rate_limit(
        "30/minute"
    ) == (30, 60)

    assert parse_rate_limit(
        "10/second"
    ) == (10, 1)

    assert parse_rate_limit(
        "100/hour"
    ) == (100, 3600)


def test_parse_invalid_rate_limit():
    with pytest.raises(ValueError):
        parse_rate_limit(
            "invalid"
        )

    with pytest.raises(ValueError):
        parse_rate_limit(
            "10/day"
        )


def test_rate_limiter_blocks_after_limit():
    limiter = InMemoryRateLimiter()

    limiter.check(
        client_id="test-client",
        bucket="chat",
        limit=2,
        window_seconds=60,
    )

    limiter.check(
        client_id="test-client",
        bucket="chat",
        limit=2,
        window_seconds=60,
    )

    with pytest.raises(HTTPException) as exc:
        limiter.check(
            client_id="test-client",
            bucket="chat",
            limit=2,
            window_seconds=60,
        )

    assert exc.value.status_code == 429