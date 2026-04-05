import pytest

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
