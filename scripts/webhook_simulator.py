import json
import requests

def simulate_webhook_event(repo_id, event_type, payload):
    url = f"http://localhost:8000/webhook/repo-update"
    headers = {
        "Content-Type": "application/json",
        "X-Hub-Signature": "sha1=YOUR_SIGNATURE"  # Replace with actual signature logic
    }
    
    event_data = {
        "repo_id": repo_id,
        "event_type": event_type,
        "payload": payload
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(event_data))
    
    if response.status_code == 202:
        print("Webhook event simulated successfully.")
    else:
        print(f"Failed to simulate webhook event: {response.status_code} - {response.text}")

if __name__ == "__main__":
    # Example usage
    repo_id = "example_repo"
    event_type = "push"
    payload = {
        "ref": "refs/heads/main",
        "commits": [
            {
                "id": "abc123",
                "message": "Example commit message",
                "timestamp": "2023-10-01T12:00:00Z",
                "url": "http://example.com/commit/abc123"
            }
        ]
    }
    
    simulate_webhook_event(repo_id, event_type, payload)