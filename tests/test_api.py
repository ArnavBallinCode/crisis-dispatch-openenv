from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_tasks_endpoint_exposes_core_tasks() -> None:
    response = client.get("/tasks")

    assert response.status_code == 200
    tasks = response.json()
    task_ids = {task["id"] for task in tasks}

    assert {"easy", "medium", "hard"}.issubset(task_ids)


def test_reset_step_and_score_roundtrip() -> None:
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
