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
    assert reset_response.json()["observation"]["task_id"] == "easy"

    step_response = client.post("/step", json={})
    assert step_response.status_code == 200
    step_payload = step_response.json()
    assert "observation" in step_payload
    assert "reward" in step_payload

    score_response = client.get("/grader")
    assert score_response.status_code == 200
    score_payload = score_response.json()
    assert score_payload["task_id"] == "easy"
    assert 0.0 <= score_payload["grade"]["score"] <= 1.0


def test_demo_run_endpoint_reaches_final_score() -> None:
    response = client.post("/demo/run/easy")
    assert response.status_code == 200

    payload = response.json()
    assert payload["task_id"] == "easy"
    assert payload["done"] is True
    assert payload["steps"] >= 1
    assert 0.0 <= payload["score"] <= 1.0

    grader_response = client.get("/grader")
    assert grader_response.status_code == 200
    grader_payload = grader_response.json()
    assert grader_payload["done"] is True
    assert grader_payload["grade"]["score"] == payload["score"]


def test_demo_benchmark_endpoint_returns_all_tasks() -> None:
    response = client.get("/demo/benchmark")
    assert response.status_code == 200

    payload = response.json()
    assert payload["policy"] == "inference.heuristic_policy"

    results = payload["results"]
    assert {"easy", "medium", "hard"}.issubset(set(results.keys()))

    for task_id in ["easy", "medium", "hard"]:
        row = results[task_id]
        assert 0.0 <= row["score"] <= 1.0
        assert row["steps"] >= 1


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

