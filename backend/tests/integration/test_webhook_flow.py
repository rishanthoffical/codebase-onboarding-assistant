import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_webhook_update():
    # Simulate a webhook payload
    payload = {
        "ref": "refs/heads/main",
        "before": "0000000000000000000000000000000000000000",
        "after": "1234567890abcdef1234567890abcdef12345678",
        "repository": {
            "id": 1,
            "name": "test-repo",
            "full_name": "user/test-repo",
            "owner": {
                "name": "user"
            }
        }
    }

    response = client.post("/webhook/repo-update", json=payload)
    assert response.status_code == 202

def test_webhook_invalid_signature():
    # Simulate a webhook payload with an invalid signature
    payload = {
        "ref": "refs/heads/main",
        "before": "0000000000000000000000000000000000000000",
        "after": "1234567890abcdef1234567890abcdef12345678",
        "repository": {
            "id": 1,
            "name": "test-repo",
            "full_name": "user/test-repo",
            "owner": {
                "name": "user"
            }
        }
    }

    # Here we would normally set an invalid signature in the headers
    headers = {
        "X-Hub-Signature": "invalid_signature"
    }

    response = client.post("/webhook/repo-update", json=payload, headers=headers)
    assert response.status_code == 403

def test_webhook_no_payload():
    # Test the webhook with no payload
    response = client.post("/webhook/repo-update", json={})
    assert response.status_code == 400