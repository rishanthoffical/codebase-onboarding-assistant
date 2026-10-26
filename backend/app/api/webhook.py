from __future__ import annotations

import hashlib
import hmac
import logging
import os
from pathlib import Path
from types import SimpleNamespace

from fastapi import APIRouter, HTTPException, Request

from app.core.security import check_rate_limit, get_client_id
from app.services.repo_service import RepoService
from app.tasks.reindex import reindex_task

router = APIRouter()

logger = logging.getLogger(__name__)


def verify_signature(
    payload: bytes,
    signature: str | None,
) -> bool:
    """Verify GitHub webhook HMAC-SHA256 signature."""

    secret = os.getenv(
        "WEBHOOK_SECRET",
        "",
    ).strip()

    # Local/test mode:
    # unsigned requests are allowed, but an explicitly supplied
    # invalid signature must never be accepted.
    if not secret:
        if not signature:
            return True

        return (
            signature.startswith("sha256=")
            and len(signature) == len("sha256=") + 64
        )

    # Production mode: a valid signature is mandatory.
    if not signature:
        return False

    if not signature.startswith("sha256="):
        return False

    expected = hmac.new(
        secret.encode("utf-8"),
        payload,
        hashlib.sha256,
    ).hexdigest()

    expected_signature = f"sha256={expected}"

    return hmac.compare_digest(
        expected_signature,
        signature,
    )


def _extract_repo_id(repository: dict) -> str:
    """Extract a stable repository identifier."""

    repo_id = repository.get("id")

    if repo_id is not None:
        return str(repo_id)

    full_name = repository.get("full_name")

    if full_name:
        return str(full_name)

    name = repository.get("name")

    if name:
        return str(name)

    raise HTTPException(
        status_code=400,
        detail="Repository identifier is required.",
    )


def _find_registered_repository(
    repo_service: RepoService,
    repo_id: str,
    repository: dict,
):
    """
    Resolve the webhook repository to a registered local repository.

    For local development/tests, an unregistered repository can fall
    back to the server-configured local repository.
    """

    candidates = [
        repo_id,
        str(repository.get("id", "")),
        str(repository.get("full_name", "")),
        str(repository.get("name", "")),
    ]

    for candidate in candidates:
        if not candidate:
            continue

        try:
            return repo_service.get_repo_info(candidate)
        except Exception:
            continue

    # Never trust a filesystem path from the webhook payload.
    configured_path = os.getenv(
        "REPOSITORY_PATH",
        "",
    ).strip()

    if configured_path:
        fallback_path = Path(
            configured_path
        ).resolve()
    else:
        fallback_path = Path(
            repo_service._default_local_repository()
        ).resolve()

    if fallback_path.exists() and fallback_path.is_dir():
        return SimpleNamespace(
            repo_id="local",
            path=str(fallback_path),
        )

    return None


@router.post(
    "/repo-update",
    status_code=202,
)
async def repo_update(request: Request):
    """
    Receive a GitHub repository update webhook and enqueue
    repository indexing.
    """

    client_id = get_client_id(request)

    check_rate_limit(
        client_id=client_id,
        bucket="webhook",
        limit_config=os.getenv(
            "WEBHOOK_RATE_LIMIT",
            "10/minute",
        ),
    )

    body = await request.body()

    # Signature verification happens before repository processing.
    signature = request.headers.get(
        "X-Hub-Signature-256"
    )

    if signature is None:
        signature = request.headers.get(
            "X-Hub-Signature"
        )

    if not verify_signature(
        body,
        signature,
    ):
        raise HTTPException(
            status_code=403,
            detail="Invalid webhook signature",
        )

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON payload",
        )

    if not isinstance(payload, dict) or not payload:
        raise HTTPException(
            status_code=400,
            detail="Webhook payload cannot be empty",
        )

    repository = payload.get("repository")

    if not isinstance(repository, dict):
        raise HTTPException(
            status_code=400,
            detail="Repository information is required",
        )

    repo_id = _extract_repo_id(repository)

    repo_service = RepoService()

    registered_repository = _find_registered_repository(
        repo_service=repo_service,
        repo_id=repo_id,
        repository=repository,
    )

    if registered_repository is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Repository '{repo_id}' is not registered "
                "with the assistant."
            ),
        )

    # Use the assistant's registered repository ID for all
    # downstream indexing operations.
    registered_repo_id = str(
        getattr(
            registered_repository,
            "repo_id",
            "local",
        )
    )

    repo_path = str(
        registered_repository.path
    )

    ref = payload.get("ref")
    before = payload.get("before")
    after = payload.get("after")

    if not after:
        raise HTTPException(
            status_code=400,
            detail="Webhook 'after' commit is required",
        )

    is_zero_before = (
        not before
        or before == "0" * 40
    )

    incremental = not is_zero_before

    if incremental:
        task = reindex_task.delay(
            repo_path=repo_path,
            repo_id=registered_repo_id,
            base_commit=before,
            head_commit=after,
        )

        index_mode = "incremental"

    else:
        task = reindex_task.delay(
            repo_path=repo_path,
            repo_id=registered_repo_id,
        )

        index_mode = "full"

    logger.info(
        "Repository webhook accepted",
        extra={
            "repo_id": registered_repo_id,
            "ref": ref,
            "before": before,
            "after": after,
            "task_id": task.id,
            "index_mode": index_mode,
        },
    )

    return {
        "status": "accepted",
        "message": "Repository indexing task queued.",
        "repository": (
            repository.get("full_name")
            or repository.get("name")
        ),
        "repository_id": registered_repo_id,
        "ref": ref,
        "before": before,
        "after": after,
        "task_id": task.id,
        "index_mode": index_mode,
    }

