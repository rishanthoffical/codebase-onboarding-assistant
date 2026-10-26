import hashlib
import hmac
import json
import os
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


WEBHOOK_SECRET = os.getenv(
    "WEBHOOK_SECRET",
    "webhook-test-secret",
)


def make_signature(payload: dict) -> str:
    body = json.dumps(
        payload,
        separators=(",", ":"),
    ).encode("utf-8")

    digest = hmac.new(
        WEBHOOK_SECRET.encode("utf-8"),
        body,
        hashlib.sha256,
    ).hexdigest()

    return f"sha256={digest}"


def test_webhook_update():
    payload = {
        "ref": "refs/heads/main",
        "before": "0000000000000000000000000000000000000000",
        "after": "1234567890abcdef1234567890abcdef12345678",
        "repository": {
            "id": "local",
            "name": "Local Repository",
            "full_name": "local",
            "owner": {
                "name": "local"
            }
        }
    }

    body = json.dumps(
        payload,
        separators=(",", ":"),
    )

    signature = make_signature(payload)

    fake_task = type(
        "FakeTask",
        (),
        {"id": "test-task-id"},
    )()

    with patch(
        "app.api.webhook.reindex_task.delay",
        return_value=fake_task,
    ):
        response = client.post(
            "/webhook/repo-update",
            content=body,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": signature,
            },
        )

    assert response.status_code == 202

    data = response.json()

    assert data["status"] == "accepted"
    assert data["task_id"] == "test-task-id"
    assert data["repository_id"] == "local"
    assert data["index_mode"] == "full"


def test_webhook_invalid_signature():
    payload = {
        "ref": "refs/heads/main",
        "before": "0000000000000000000000000000000000000000",
        "after": "1234567890abcdef1234567890abcdef12345678",
        "repository": {
            "id": "local",
            "name": "Local Repository",
            "full_name": "local",
            "owner": {
                "name": "local"
            }
        }
    }

    response = client.post(
        "/webhook/repo-update",
        json=payload,
        headers={
            "X-Hub-Signature-256": "sha256=invalid_signature",
        },
    )

    assert response.status_code == 403


def test_webhook_no_payload():
    payload = {}

    body = json.dumps(
        payload,
        separators=(",", ":"),
    )

    signature = make_signature(payload)

    response = client.post(
        "/webhook/repo-update",
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-Hub-Signature-256": signature,
        },
    )

    assert response.status_code == 400
