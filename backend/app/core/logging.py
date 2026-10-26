from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any


SENSITIVE_KEYS = {
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "access_token",
    "refresh_token",
    "private_key",
}


def _redact(value: Any) -> Any:
    """
    Recursively redact sensitive values before they reach logs.
    """

    if isinstance(value, dict):
        result: dict[str, Any] = {}

        for key, item in value.items():
            normalized_key = str(key).lower().replace("-", "_")

            if (
                normalized_key in SENSITIVE_KEYS
                or any(
                    sensitive in normalized_key
                    for sensitive in (
                        "password",
                        "token",
                        "secret",
                        "api_key",
                        "authorization",
                    )
                )
            ):
                result[str(key)] = "[REDACTED]"
            else:
                result[str(key)] = _redact(item)

        return result

    if isinstance(value, list):
        return [_redact(item) for item in value]

    if isinstance(value, tuple):
        return [_redact(item) for item in value]

    return value


class SafeJsonFormatter(logging.Formatter):
    """
    JSON formatter for structured application logs.

    Sensitive fields are recursively redacted.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),

            "level": record.levelname,

            "logger": record.name,

            "message": record.getMessage(),
        }

        if record.exc_info:
            payload["exception"] = self.formatException(
                record.exc_info
            )

        if record.stack_info:
            payload["stack"] = record.stack_info

        extra_fields = getattr(
            record,
            "structured",
            None,
        )

        if isinstance(extra_fields, dict):
            payload.update(
                _redact(extra_fields)
            )

        return json.dumps(
            _redact(payload),
            ensure_ascii=False,
            default=str,
        )


class SafeLoggerAdapter(logging.LoggerAdapter):
    """
    LoggerAdapter that automatically stores structured fields
    in the `structured` LogRecord attribute.
    """

    def process(
        self,
        msg: Any,
        kwargs: dict[str, Any],
    ):
        extra = kwargs.setdefault(
            "extra",
            {},
        )

        structured = extra.setdefault(
            "structured",
            {},
        )

        if self.extra:
            structured.update(
                _redact(self.extra)
            )

        return msg, kwargs


def configure_logging() -> None:
    """
    Configure application-wide structured JSON logging.

    LOG_LEVEL defaults to INFO.
    """

    level_name = os.getenv(
        "LOG_LEVEL",
        "INFO",
    ).upper()

    level = getattr(
        logging,
        level_name,
        logging.INFO,
    )

    root_logger = logging.getLogger()

    root_logger.setLevel(level)

    # Avoid duplicate handlers when called more than once.
    for handler in list(
        root_logger.handlers
    ):
        if getattr(
            handler,
            "_codebase_assistant_handler",
            False,
        ):
            root_logger.removeHandler(
                handler
            )

    handler = logging.StreamHandler(
        sys.stdout
    )

    handler._codebase_assistant_handler = True

    handler.setLevel(level)

    handler.setFormatter(
        SafeJsonFormatter()
    )

    root_logger.addHandler(handler)


def get_safe_logger(
    name: str,
) -> SafeLoggerAdapter:
    """
    Return a logger adapter suitable for structured logging.
    """

    return SafeLoggerAdapter(
        logging.getLogger(name),
        {},
    )
