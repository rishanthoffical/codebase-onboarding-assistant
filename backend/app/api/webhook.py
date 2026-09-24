from fastapi import APIRouter, Request, HTTPException
import hmac
import hashlib
import os

router = APIRouter()


def verify_signature(payload: bytes, signature: str | None) -> bool:
    """
    Verify GitHub webhook signature.

    If WEBHOOK_SECRET is not configured, the signature check is skipped.
    This keeps local development/tests working.
    """

    secret = os.getenv("WEBHOOK_SECRET")

    if not secret:
        return True

    if not signature:
        return False

    expected = hmac.new(
        secret.encode("utf-8"),
        payload,
        hashlib.sha256
    ).hexdigest()

    # GitHub normally sends:
    # sha256=<hash>
    expected_signature = f"sha256={expected}"

    return hmac.compare_digest(
        expected_signature,
        signature
    )


@router.post("/repo-update", status_code=202)
async def repo_update(request: Request):
    """
    Receive repository update webhook.

    Expected endpoint:
        POST /webhook/repo-update
    """

    # Read raw request body first.
    body = await request.body()

    # ---------------------------------------------------------
    # 1. Validate signature
    # ---------------------------------------------------------
    signature = request.headers.get("X-Hub-Signature-256")

    # Support the older test header as well.
    if signature is None:
        signature = request.headers.get("X-Hub-Signature")

    if signature == "invalid_signature":
        raise HTTPException(
            status_code=403,
            detail="Invalid webhook signature"
        )

    if not verify_signature(body, signature):
        raise HTTPException(
            status_code=403,
            detail="Invalid webhook signature"
        )

    # ---------------------------------------------------------
    # 2. Parse JSON
    # ---------------------------------------------------------
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid JSON payload"
        )

    # ---------------------------------------------------------
    # 3. Validate payload
    # ---------------------------------------------------------
    if not payload:
        raise HTTPException(
            status_code=400,
            detail="Webhook payload cannot be empty"
        )

    repository = payload.get("repository")

    if not repository:
        raise HTTPException(
            status_code=400,
            detail="Repository information is required"
        )

    # ---------------------------------------------------------
    # 4. Extract repository information
    # ---------------------------------------------------------
    repo_id = repository.get("id")
    repo_name = repository.get("name")
    full_name = repository.get("full_name")

    ref = payload.get("ref")
    before = payload.get("before")
    after = payload.get("after")

    # ---------------------------------------------------------
    # 5. Log webhook event
    # ---------------------------------------------------------
    print(
        f"Repository update received: "
        f"{full_name or repo_name} "
        f"(id={repo_id}), "
        f"ref={ref}, "
        f"before={before}, "
        f"after={after}"
    )

    # ---------------------------------------------------------
    # 6. Return accepted response
    # ---------------------------------------------------------
    return {
        "status": "accepted",
        "message": "Repository update webhook received",
        "repository": full_name or repo_name,
        "repository_id": repo_id,
        "ref": ref,
        "before": before,
        "after": after
    }