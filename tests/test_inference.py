import pytest

from inference import heuristic_policy, run_episode


@pytest.mark.parametrize("task_id", ["easy", "medium", "hard"])
def test_heuristic_policy_is_deterministic(task_id: str) -> None:
    first = run_episode(task_id=task_id, policy=heuristic_policy)
    second = run_episode(task_id=task_id, policy=heuristic_policy)

    assert first.score == second.score
    assert first.cumulative_reward == second.cumulative_reward
