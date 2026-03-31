import pytest

from app.environment import CrisisDispatchEnvironment
from app.models import DispatchAction


@pytest.mark.parametrize("task_id", ["easy", "medium", "hard"])
def test_environment_episode_finishes_with_valid_score(task_id: str) -> None:
    env = CrisisDispatchEnvironment(default_task_id=task_id)
    state = env.reset(task_id=task_id)

    assert state.task_id == task_id

    while not state.done:
        state = env.step(DispatchAction()).state

    grade = env.grade()
    assert 0.0 <= grade.score <= 1.0
