from __future__ import annotations

import os

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_local_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_local_reset_and_tasks_contract() -> None:
    reset_response = client.post("/reset", json={})
    assert reset_response.status_code == 200

    tasks_response = client.get("/tasks")
    assert tasks_response.status_code == 200

    tasks = tasks_response.json()
    assert isinstance(tasks, list)
    assert len(tasks) >= 3

    task_ids = {item.get("id") for item in tasks if isinstance(item, dict)}
    assert {"easy", "medium", "hard"}.issubset(task_ids)


def test_local_step_and_score_contract() -> None:
    reset_response = client.post("/reset/easy")
    assert reset_response.status_code == 200
    assert reset_response.json()["task_id"] == "easy"

    step_response = client.post("/step", json={})
    assert step_response.status_code == 200
    step_payload = step_response.json()
    assert "state" in step_payload
    assert "reward" in step_payload

    score_response = client.get("/score")
    assert score_response.status_code == 200
    score_payload = score_response.json()
    assert score_payload["task_id"] == "easy"
    assert 0.0 <= score_payload["grade"]["score"] <= 1.0


@pytest.mark.external
def test_remote_hf_space_endpoints_if_configured() -> None:
    ping_url = os.getenv("PING_URL", "").strip().rstrip("/")
    if not ping_url:
        pytest.skip("Set PING_URL to enable remote HF Space endpoint checks")

    with httpx.Client(timeout=30.0) as http:
        reset_response = http.post(f"{ping_url}/reset", json={})
        assert reset_response.status_code == 200

        tasks_response = http.get(f"{ping_url}/tasks")
        assert tasks_response.status_code == 200

        tasks = tasks_response.json()
        assert isinstance(tasks, list)
        assert len(tasks) >= 3

