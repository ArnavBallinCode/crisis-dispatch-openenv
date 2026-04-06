import pytest

from app.baseline import heuristic_policy
from app.environment import CrisisDispatchEnvironment
from app.models import Action
from app.tasks import TASKS


def test_task_catalog_has_minimum_coverage() -> None:
    assert len(TASKS) >= 3
    assert {"easy", "medium", "hard"}.issubset(set(TASKS.keys()))


@pytest.mark.parametrize("task_id", ["easy", "medium", "hard"])
def test_environment_episode_finishes_with_valid_score(task_id: str) -> None:
    env = CrisisDispatchEnvironment(default_task_id=task_id)
    state = env.reset(task_id=task_id)

    assert state.task_id == task_id

    while not state.done:
        state = env.step(Action()).observation

    grade = env.grade()
    assert 0.0 <= grade.score <= 1.0


@pytest.mark.parametrize("task_id", ["easy", "medium", "hard"])
def test_reset_returns_requested_task(task_id: str) -> None:
    env = CrisisDispatchEnvironment(default_task_id="easy")
    state = env.reset(task_id=task_id)
    assert state.task_id == task_id


def test_grader_progress_changes_with_episode_progress() -> None:
    env = CrisisDispatchEnvironment(default_task_id="easy")
    state = env.reset(task_id="easy")

    start_score = env.grade().score
    state = env.step(Action(unit_id="A1", incident_id="E-MED-1")).observation
    mid_score = env.grade().score

    while not state.done:
        state = env.step(Action()).observation

    final_score = env.grade().score

    assert mid_score != start_score
    assert final_score > mid_score


@pytest.mark.parametrize("task_id", ["easy", "medium", "hard"])
def test_grader_distinguishes_wait_vs_heuristic(task_id: str) -> None:
    wait_env = CrisisDispatchEnvironment(default_task_id=task_id)
    wait_state = wait_env.reset(task_id=task_id)
    while not wait_state.done:
        wait_state = wait_env.step(Action()).observation
    wait_score = wait_env.grade().score

    heuristic_env = CrisisDispatchEnvironment(default_task_id=task_id)
    heuristic_state = heuristic_env.reset(task_id=task_id)
    while not heuristic_state.done:
        heuristic_state = heuristic_env.step(heuristic_policy(heuristic_state)).observation
    heuristic_score = heuristic_env.grade().score

    assert heuristic_score > wait_score
