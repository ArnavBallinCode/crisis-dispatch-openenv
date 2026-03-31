import pytest

from inference import heuristic_policy, run_episode


@pytest.mark.parametrize("task_id", ["easy", "medium", "hard"])
def test_heuristic_policy_is_deterministic(task_id: str) -> None:
    first = run_episode(task_id=task_id, policy=heuristic_policy)
    second = run_episode(task_id=task_id, policy=heuristic_policy)

    assert first.score == second.score
    assert first.cumulative_reward == second.cumulative_reward


def test_baseline_outputs_task_rows_and_determinism_passes(project_root) -> None:
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "inference.py",
        "--mode",
        "heuristic",
        "--task",
        "all",
        "--check-determinism",
    ]
    completed = subprocess.run(
        cmd,
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=1200,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    output = completed.stdout

    lines = [line.strip() for line in output.splitlines() if line.strip()]
    score_rows = [line for line in lines if line.startswith(("easy,", "medium,", "hard,"))]
    assert len(score_rows) >= 3

    for row in score_rows:
        score = float(row.split(",")[1])
        assert 0.0 <= score <= 1.0

    assert "easy: PASS" in output
    assert "medium: PASS" in output
    assert "hard: PASS" in output
