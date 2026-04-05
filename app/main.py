from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

from app.environment import CrisisDispatchEnvironment
from app.models import Action, Observation, ResetRequest, ScoreResponse, StepResult, TaskSummary
from app.tasks import list_task_summaries


if load_dotenv is not None:
    # Load local development variables from project root if present.
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=False)


app = FastAPI(
    title="Crisis Resource Dispatch Environment",
    version="1.0.0",
    description=(
        "OpenEnv-compatible emergency dispatch simulator with deterministic grading and "
        "dense step-wise rewards."
    ),
)

environment = CrisisDispatchEnvironment(default_task_id="easy")


@app.get("/")
def root() -> dict:
    return {
        "name": "crisis-dispatch-env",
        "message": "Crisis dispatch environment is ready",
        "current_task": environment.state().task_id,
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/tasks", response_model=List[TaskSummary])
def tasks() -> List[TaskSummary]:
    return list_task_summaries()


@app.post("/reset", response_model=StepResult)
def reset(request: Optional[ResetRequest] = None) -> StepResult:
    task_id = request.task_id if request and request.task_id else "easy"
    try:
        obs = environment.reset(task_id=task_id)
        return StepResult(observation=obs, reward=0.0, done=False)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/reset/{task_id}", response_model=StepResult)
def reset_by_path(task_id: str) -> StepResult:
    try:
        obs = environment.reset(task_id=task_id)
        return StepResult(observation=obs, reward=0.0, done=False)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/step", response_model=StepResult)
def step(action: Action) -> StepResult:
    try:
        return environment.step(action)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/state", response_model=Observation)
def state() -> Observation:
    return environment.state()


@app.get("/grader", response_model=ScoreResponse)
def grader() -> ScoreResponse:
    grade = environment.grade()
    current_state = environment.state()
    return ScoreResponse(grade=grade, task_id=current_state.task_id, done=current_state.done)


@app.get("/baseline")
def baseline() -> dict:
    """Run heuristic baseline on the easy task and return scores."""
    env = CrisisDispatchEnvironment(default_task_id="easy")
    obs = env.reset(task_id="easy")
    rewards = []

    while not obs.done:
        # Simple nearest-available-correct-unit heuristic
        action = _baseline_heuristic(obs)
        result = env.step(action)
        rewards.append(result.reward)
        obs = result.observation

    grade = env.grade()
    return {
        "task_id": "easy",
        "score": grade.score,
        "cumulative_reward": obs.cumulative_reward,
        "steps": obs.step_count,
        "rewards": rewards,
    }


def _baseline_heuristic(obs: Observation) -> Action:
    """Minimal correct-type nearest dispatch for baseline endpoint."""
    active = [i for i in obs.incidents if not i.resolved and not i.failed]
    avail = [u for u in obs.units if u.status.value == "available"]

    if not active or not avail:
        return Action()

    best_score = float("-inf")
    best_action = Action()

    for incident in active:
        required = set(incident.required_units)
        responding = set(incident.responding_units)
        missing = required - responding

        for unit in avail:
            if unit.unit_type not in missing:
                continue
            distance = abs(unit.position.x - incident.position.x) + abs(unit.position.y - incident.position.y)
            slack = incident.max_wait - incident.elapsed
            if distance > slack:
                continue
            score = 3.0 * (3.5 if incident.severity.value == "critical" else 1.0) - distance
            if score > best_score:
                best_score = score
                best_action = Action(unit_id=unit.id, incident_id=incident.id)

    return best_action
