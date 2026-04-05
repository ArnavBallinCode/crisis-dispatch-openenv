from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

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


@app.get("/", response_class=HTMLResponse)
def root() -> str:
        return """<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Crisis Dispatch OpenEnv Demo</title>
    <style>
        :root {
            --bg: #f7f3ed;
            --panel: #fffdf9;
            --ink: #1c2329;
            --muted: #5b6671;
            --accent: #0f6a7a;
            --accent-2: #e16f3d;
            --line: #d6d9dd;
            --ok: #1f8d57;
            --warn: #b0552d;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            color: var(--ink);
            background:
                radial-gradient(circle at 0% 0%, #f0d8bc 0%, rgba(240, 216, 188, 0) 40%),
                radial-gradient(circle at 100% 100%, #cddfea 0%, rgba(205, 223, 234, 0) 45%),
                var(--bg);
            font-family: "Avenir Next", "Segoe UI", sans-serif;
            min-height: 100vh;
            padding: 20px;
        }
        .wrap {
            max-width: 1100px;
            margin: 0 auto;
            display: grid;
            gap: 16px;
            grid-template-columns: 1fr;
        }
        .hero {
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 20px;
            background: linear-gradient(125deg, #fff7ee 0%, #f6fbff 100%);
            box-shadow: 0 8px 30px rgba(17, 24, 39, 0.07);
        }
        h1 {
            margin: 0 0 8px;
            font-size: 1.6rem;
            letter-spacing: 0.02em;
        }
        .subtitle {
            margin: 0;
            color: var(--muted);
            line-height: 1.45;
        }
        .grid {
            display: grid;
            gap: 16px;
            grid-template-columns: 1fr;
        }
        .card {
            border: 1px solid var(--line);
            border-radius: 16px;
            background: var(--panel);
            box-shadow: 0 8px 20px rgba(17, 24, 39, 0.06);
            overflow: hidden;
        }
        .card > .head {
            padding: 12px 14px;
            font-weight: 700;
            border-bottom: 1px solid var(--line);
            background: #f8fafc;
        }
        .card > .body {
            padding: 14px;
        }
        .row {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            align-items: center;
            margin-bottom: 10px;
        }
        label {
            font-size: 0.9rem;
            color: var(--muted);
        }
        select, input {
            border: 1px solid var(--line);
            border-radius: 10px;
            padding: 9px 10px;
            font-size: 0.95rem;
            min-width: 180px;
            background: #fff;
            color: var(--ink);
        }
        button {
            border: none;
            border-radius: 10px;
            padding: 9px 12px;
            font-weight: 700;
            cursor: pointer;
            transition: transform .06s ease, filter .2s ease;
        }
        button:active { transform: translateY(1px); }
        .primary { background: var(--accent); color: #fff; }
        .secondary { background: #2f3840; color: #fff; }
        .warm { background: var(--accent-2); color: #fff; }
        .ghost { background: #eef1f4; color: var(--ink); }
        pre {
            margin: 0;
            border-radius: 10px;
            background: #0f1820;
            color: #dde6ee;
            padding: 12px;
            overflow: auto;
            min-height: 140px;
            font-size: 0.83rem;
            line-height: 1.4;
        }
        .status {
            font-size: 0.92rem;
            margin-top: 8px;
            color: var(--muted);
        }
        .status.ok { color: var(--ok); }
        .status.error { color: #b42318; }
        .pill {
            display: inline-block;
            border: 1px solid var(--line);
            background: #fff;
            border-radius: 999px;
            padding: 3px 8px;
            font-size: 0.78rem;
            color: var(--muted);
            margin-right: 6px;
        }
        .endpoints a {
            color: var(--accent);
            text-decoration: none;
            margin-right: 10px;
            font-weight: 600;
        }
        .endpoints a:hover { text-decoration: underline; }
        @media (min-width: 900px) {
            .grid { grid-template-columns: 1fr 1fr; }
        }
    </style>
</head>
<body>
    <div class="wrap">
        <section class="hero">
            <h1>Crisis Dispatch OpenEnv Demo</h1>
            <p class="subtitle">
                Interact with the environment directly from this Space: reset a task, dispatch units, inspect state, and verify grading behavior.
            </p>
            <div class="row" style="margin-top:10px;">
                <span class="pill">OpenEnv: reset / step / state</span>
                <span class="pill">Deterministic grader</span>
                <span class="pill">Dense rewards</span>
            </div>
            <div class="row endpoints">
                <a href="/docs" target="_blank" rel="noopener">API Docs</a>
                <a href="/tasks" target="_blank" rel="noopener">Tasks JSON</a>
                <a href="/health" target="_blank" rel="noopener">Health</a>
                <a href="/meta" target="_blank" rel="noopener">Meta JSON</a>
            </div>
        </section>

        <section class="grid">
            <article class="card">
                <div class="head">Interactive Controls</div>
                <div class="body">
                    <div class="row">
                        <label for="taskSelect">Task</label>
                        <select id="taskSelect"></select>
                        <button class="primary" id="resetBtn">Reset Task</button>
                        <button class="ghost" id="stateBtn">Refresh State</button>
                        <button class="ghost" id="scoreBtn">Get Score</button>
                    </div>
                    <div class="row">
                        <label for="unitInput">unit_id</label>
                        <input id="unitInput" placeholder="A1">
                        <label for="incidentInput">incident_id</label>
                        <input id="incidentInput" placeholder="E-MED-1">
                    </div>
                    <div class="row">
                        <button class="secondary" id="dispatchBtn">Step Dispatch</button>
                        <button class="warm" id="waitBtn">Step Wait</button>
                        <button class="ghost" id="baselineBtn">Run Baseline (easy)</button>
                    </div>
                    <div id="status" class="status">Ready.</div>
                </div>
            </article>

            <article class="card">
                <div class="head">Live Output</div>
                <div class="body">
                    <pre id="output">Loading tasks...</pre>
                </div>
            </article>
        </section>
    </div>

    <script>
        const outputEl = document.getElementById("output");
        const statusEl = document.getElementById("status");
        const taskSelect = document.getElementById("taskSelect");

        function showStatus(text, level) {
            statusEl.textContent = text;
            statusEl.classList.remove("ok", "error");
            if (level) statusEl.classList.add(level);
        }

        function pretty(obj) {
            return JSON.stringify(obj, null, 2);
        }

        async function api(path, options) {
            const response = await fetch(path, {
                headers: { "Content-Type": "application/json" },
                ...(options || {}),
            });
            const text = await response.text();
            let payload;
            try {
                payload = text ? JSON.parse(text) : {};
            } catch {
                payload = { raw: text };
            }
            if (!response.ok) {
                const msg = payload && payload.detail ? payload.detail : response.status + " " + response.statusText;
                throw new Error(msg);
            }
            return payload;
        }

        async function loadTasks() {
            const tasks = await api("/tasks");
            taskSelect.innerHTML = "";
            tasks.forEach((t) => {
                const opt = document.createElement("option");
                opt.value = t.id;
                opt.textContent = t.id + " - " + (t.description || "");
                taskSelect.appendChild(opt);
            });
            if (taskSelect.options.length) taskSelect.value = "easy";
            return tasks;
        }

        async function refreshState() {
            const payload = await api("/state");
            outputEl.textContent = pretty({ endpoint: "/state", payload });
            showStatus("State refreshed.", "ok");
        }

        async function refreshScore() {
            const payload = await api("/grader");
            outputEl.textContent = pretty({ endpoint: "/grader", payload });
            showStatus("Score refreshed.", "ok");
        }

        async function resetTask() {
            const taskId = taskSelect.value;
            const payload = await api("/reset/" + encodeURIComponent(taskId), { method: "POST" });
            outputEl.textContent = pretty({ endpoint: "/reset/" + taskId, payload });
            showStatus("Task reset to " + taskId + ".", "ok");
        }

        async function stepDispatch() {
            const unitId = document.getElementById("unitInput").value.trim();
            const incidentId = document.getElementById("incidentInput").value.trim();
            const body = {};
            if (unitId) body.unit_id = unitId;
            if (incidentId) body.incident_id = incidentId;
            const payload = await api("/step", { method: "POST", body: JSON.stringify(body) });
            outputEl.textContent = pretty({ endpoint: "/step", action: body, payload });
            showStatus("Step submitted.", "ok");
        }

        async function stepWait() {
            const payload = await api("/step", { method: "POST", body: "{}" });
            outputEl.textContent = pretty({ endpoint: "/step", action: {}, payload });
            showStatus("Wait step submitted.", "ok");
        }

        async function runBaseline() {
            const payload = await api("/baseline");
            outputEl.textContent = pretty({ endpoint: "/baseline", payload });
            showStatus("Baseline completed.", "ok");
        }

        async function guarded(runFn) {
            try {
                showStatus("Running...", null);
                await runFn();
            } catch (err) {
                showStatus("Error: " + err.message, "error");
            }
        }

        document.getElementById("resetBtn").addEventListener("click", () => guarded(resetTask));
        document.getElementById("stateBtn").addEventListener("click", () => guarded(refreshState));
        document.getElementById("scoreBtn").addEventListener("click", () => guarded(refreshScore));
        document.getElementById("dispatchBtn").addEventListener("click", () => guarded(stepDispatch));
        document.getElementById("waitBtn").addEventListener("click", () => guarded(stepWait));
        document.getElementById("baselineBtn").addEventListener("click", () => guarded(runBaseline));

        (async () => {
            try {
                await loadTasks();
                await resetTask();
            } catch (err) {
                showStatus("Startup error: " + err.message, "error");
            }
        })();
    </script>
</body>
</html>
"""


@app.get("/meta")
def meta() -> dict:
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
