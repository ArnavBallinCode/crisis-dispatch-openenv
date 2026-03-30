from __future__ import annotations

from typing import List, Optional

from fastapi import FastAPI, HTTPException

from app.environment import CrisisDispatchEnvironment
from app.models import DispatchAction, EnvironmentState, ResetRequest, ScoreResponse, StepResult, TaskSummary
from app.tasks import list_task_summaries


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


@app.post("/reset", response_model=EnvironmentState)
def reset(request: Optional[ResetRequest] = None) -> EnvironmentState:
    task_id = request.task_id if request and request.task_id else "easy"
    try:
        return environment.reset(task_id=task_id)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/reset/{task_id}", response_model=EnvironmentState)
def reset_by_path(task_id: str) -> EnvironmentState:
    try:
        return environment.reset(task_id=task_id)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/step", response_model=StepResult)
def step(action: DispatchAction) -> StepResult:
    try:
        return environment.step(action)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/state", response_model=EnvironmentState)
def state() -> EnvironmentState:
    return environment.state()


@app.get("/score", response_model=ScoreResponse)
def score() -> ScoreResponse:
    grade = environment.grade()
    current_state = environment.state()
    return ScoreResponse(grade=grade, task_id=current_state.task_id, done=current_state.done)
