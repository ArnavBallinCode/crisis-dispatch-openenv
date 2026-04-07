import subprocess
import sys

import pytest


@pytest.mark.parametrize("task_id", ["easy", "medium", "hard"])
def test_heuristic_produces_valid_logs(task_id: str, project_root) -> None:
    """Verify that inference.py emits the mandatory [START]/[STEP]/[END] format."""
    cmd = [
        sys.executable,
        "inference.py",
        "--mode",
        "heuristic",
        "--task",
        task_id,
    ]
    completed = subprocess.run(
        cmd,
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    output = completed.stdout

    lines = [line.strip() for line in output.splitlines() if line.strip()]
    allowed_prefixes = ("[START]", "[STEP]", "[END]")
    assert all(line.startswith(allowed_prefixes) for line in lines)

    start_lines = [l for l in lines if l.startswith("[START]")]
    step_lines = [l for l in lines if l.startswith("[STEP]")]
    end_lines = [l for l in lines if l.startswith("[END]")]

    assert len(start_lines) == 1, f"Expected 1 [START] line, got {len(start_lines)}"
    assert len(end_lines) == 1, f"Expected 1 [END] line, got {len(end_lines)}"
    assert len(step_lines) >= 1, "Expected at least 1 [STEP] line"

    # Verify [START] format
    assert f"task={task_id}" in start_lines[0]
    assert "env=crisis-dispatch" in start_lines[0]

    # Verify [END] format (aligned with mandatory min.md sample script)
    end_line = end_lines[0]
    assert "success=" in end_line
    assert "steps=" in end_line
    assert "rewards=" in end_line
    assert "score=" in end_line


def test_all_tasks_baseline_runs(project_root) -> None:
    """Verify that running --task all produces [START]/[END] for each task."""
    cmd = [
        sys.executable,
        "inference.py",
        "--mode",
        "heuristic",
        "--task",
        "all",
    ]
    completed = subprocess.run(
        cmd,
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    output = completed.stdout

    lines = [line.strip() for line in output.splitlines() if line.strip()]
    allowed_prefixes = ("[START]", "[STEP]", "[END]")
    assert all(line.startswith(allowed_prefixes) for line in lines)

    start_lines = [l for l in lines if l.startswith("[START]")]
    end_lines = [l for l in lines if l.startswith("[END]")]

    assert len(start_lines) == 3, f"Expected 3 [START] lines, got {len(start_lines)}"
    assert len(end_lines) == 3, f"Expected 3 [END] lines, got {len(end_lines)}"
