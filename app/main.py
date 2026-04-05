from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

from app.environment import CrisisDispatchEnvironment
from app.models import Action, Observation, ResetRequest, ScoreResponse, StepResult, TaskSummary
from app.tasks import list_task_summaries
from inference import heuristic_policy


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
            .card.full { grid-column: 1 / -1; }
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
                <a href="/demo/benchmark" target="_blank" rel="noopener">Benchmark JSON</a>
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
                        <label for="unitSelect">unit_id</label>
                        <select id="unitSelect">
                            <option value="">Choose available unit</option>
                        </select>
                        <label for="incidentSelect">incident_id</label>
                        <select id="incidentSelect">
                            <option value="">Choose active incident</option>
                        </select>
                    </div>
                    <div class="row">
                        <button class="secondary" id="dispatchBtn">Step Dispatch</button>
                        <button class="warm" id="waitBtn">Step Wait</button>
                        <button class="primary" id="autoBtn">Run Auto Demo (selected task)</button>
                        <button class="secondary" id="benchmarkBtn">Run Benchmark (all tasks)</button>
                        <button class="ghost" id="baselineBtn">Run Baseline (easy)</button>
                    </div>
                    <div id="snapshot" class="status">Snapshot pending...</div>
                    <div id="status" class="status">
                        Ready. Use dropdowns to pick valid IDs; they auto-refresh from current state.
                    </div>
                </div>
            </article>

            <article class="card">
                <div class="head">Live Output</div>
                <div class="body">
                    <pre id="output">Loading tasks...</pre>
                </div>
            </article>

            <article class="card full">
                <div class="head">How This Demo Works</div>
                <div class="body">
                    <p class="subtitle" style="margin-bottom:10px;">
                        <strong>unit_id</strong> is the emergency vehicle you dispatch (for example <code>A1</code>, <code>F1</code>, <code>P1</code>).
                        <strong>incident_id</strong> is the active emergency call (for example <code>E-MED-1</code>, <code>M-FIRE-1</code>).
                    </p>
                    <p class="subtitle" style="margin-bottom:10px;">
                        Score behavior: <code>/grader</code> reports the score for the current episode snapshot. It can remain low while incidents are unresolved.
                        Use <strong>Run Auto Demo</strong> to complete an episode and view final score.
                    </p>
                    <p class="subtitle" style="margin-bottom:6px;"><strong>Recommended flow</strong></p>
                    <ol style="margin: 0 0 12px 18px; color: var(--muted);">
                        <li>Choose task and click <strong>Reset Task</strong>.</li>
                        <li>Select unit and incident from dropdowns, then click <strong>Step Dispatch</strong>.</li>
                        <li>Use <strong>Step Wait</strong> when no dispatch should be made.</li>
                        <li>Click <strong>Get Score</strong> to inspect the current snapshot.</li>
                        <li>Click <strong>Run Auto Demo</strong> for a full deterministic episode score.</li>
                        <li>Click <strong>Run Benchmark</strong> to compare easy/medium/hard against local heuristic scores.</li>
                    </ol>
                    <div id="idGuide" class="status">IDs will appear after reset/state refresh.</div>
                </div>
            </article>
        </section>
    </div>

    <script>
        const outputEl = document.getElementById("output");
        const statusEl = document.getElementById("status");
        const snapshotEl = document.getElementById("snapshot");
        const idGuideEl = document.getElementById("idGuide");
        const taskSelect = document.getElementById("taskSelect");
        const unitSelect = document.getElementById("unitSelect");
        const incidentSelect = document.getElementById("incidentSelect");

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

        function summarizeState(obs) {
            snapshotEl.textContent =
                "task=" + obs.task_id +
                " | step=" + obs.step_count + "/" + obs.max_steps +
                " | done=" + obs.done +
                " | cumulative_reward=" + Number(obs.cumulative_reward).toFixed(2);
        }

        function refreshActionOptions(obs) {
            const availableUnits = obs.units.filter((u) => u.status === "available");
            const activeIncidents = obs.incidents.filter((i) => !i.resolved && !i.failed);

            const prevUnit = unitSelect.value;
            const prevIncident = incidentSelect.value;

            unitSelect.innerHTML = '<option value="">Choose available unit</option>';
            availableUnits.forEach((unit) => {
                const opt = document.createElement("option");
                opt.value = unit.id;
                opt.textContent = unit.id + " (" + unit.unit_type + " @ " + unit.position.x + "," + unit.position.y + ")";
                unitSelect.appendChild(opt);
            });

            incidentSelect.innerHTML = '<option value="">Choose active incident</option>';
            activeIncidents.forEach((incident) => {
                const opt = document.createElement("option");
                opt.value = incident.id;
                opt.textContent =
                    incident.id +
                    " (" + incident.severity + ", needs " + incident.required_units.join("+") + ")";
                incidentSelect.appendChild(opt);
            });

            if (prevUnit && availableUnits.some((u) => u.id === prevUnit)) unitSelect.value = prevUnit;
            if (prevIncident && activeIncidents.some((i) => i.id === prevIncident)) incidentSelect.value = prevIncident;

            const unitsLabel = availableUnits.length
                ? availableUnits.map((u) => u.id + "=" + u.unit_type).join(", ")
                : "none";
            const incidentsLabel = activeIncidents.length
                ? activeIncidents.map((i) => i.id + "=" + i.incident_type + "/" + i.severity).join(", ")
                : "none";

            idGuideEl.textContent =
                "Available units: " + unitsLabel + " | Active incidents: " + incidentsLabel;
        }

        function syncFromObservation(obs) {
            summarizeState(obs);
            refreshActionOptions(obs);
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

        async function refreshState(options) {
            const payload = await api("/state");
            if (!options || !options.preserveOutput) {
                outputEl.textContent = pretty({ endpoint: "/state", payload });
            }
            syncFromObservation(payload);
            showStatus("State refreshed.", "ok");
        }

        async function refreshScore() {
            const payload = await api("/grader");
            outputEl.textContent = pretty({ endpoint: "/grader", payload });
            if (payload.done) {
                showStatus("Final score: " + payload.grade.score.toFixed(4), "ok");
            } else {
                showStatus(
                    "Snapshot score " + payload.grade.score.toFixed(4) +
                    " (episode still running). Keep stepping or use auto demo.",
                    null
                );
            }
        }

        async function resetTask() {
            const taskId = taskSelect.value;
            const payload = await api("/reset/" + encodeURIComponent(taskId), { method: "POST" });
            outputEl.textContent = pretty({ endpoint: "/reset/" + taskId, payload });
            syncFromObservation(payload.observation);
            showStatus("Task reset to " + taskId + ".", "ok");
        }

        async function stepDispatch() {
            const unitId = unitSelect.value;
            const incidentId = incidentSelect.value;
            if (!unitId || !incidentId) {
                throw new Error("Select both unit_id and incident_id from the dropdowns, or use Step Wait.");
            }
            const body = {};
            if (unitId) body.unit_id = unitId;
            if (incidentId) body.incident_id = incidentId;
            const payload = await api("/step", { method: "POST", body: JSON.stringify(body) });
            outputEl.textContent = pretty({ endpoint: "/step", action: body, payload });
            syncFromObservation(payload.observation);
            showStatus("Step submitted.", "ok");
        }

        async function stepWait() {
            const payload = await api("/step", { method: "POST", body: "{}" });
            outputEl.textContent = pretty({ endpoint: "/step", action: {}, payload });
            syncFromObservation(payload.observation);
            showStatus("Wait step submitted.", "ok");
        }

        async function runBaseline() {
            const payload = await api("/baseline");
            outputEl.textContent = pretty({ endpoint: "/baseline", payload });
            showStatus("Baseline completed.", "ok");
            await refreshState({ preserveOutput: true });
        }

        async function runAutoDemo() {
            const taskId = taskSelect.value;
            const payload = await api("/demo/run/" + encodeURIComponent(taskId), { method: "POST" });
            outputEl.textContent = pretty({ endpoint: "/demo/run/" + taskId, payload });
            showStatus("Auto demo finished with score " + payload.score.toFixed(4) + ".", "ok");
            await refreshState({ preserveOutput: true });
        }

        async function runBenchmark() {
            const payload = await api("/demo/benchmark");
            outputEl.textContent = pretty({ endpoint: "/demo/benchmark", payload });
            showStatus("Benchmark refreshed for easy/medium/hard.", "ok");
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
        document.getElementById("autoBtn").addEventListener("click", () => guarded(runAutoDemo));
        document.getElementById("benchmarkBtn").addEventListener("click", () => guarded(runBenchmark));
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


@app.post("/demo/run/{task_id}")
def run_demo_episode(task_id: str) -> dict:
    """Run a full heuristic episode on the shared environment for live demo purposes."""
    try:
        return _run_heuristic_episode(environment, task_id)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/demo/benchmark")
def demo_benchmark() -> dict:
    """Run the same deterministic heuristic as inference.py across all tasks."""
    summaries = list_task_summaries()
    results: Dict[str, dict] = {}

    for summary in summaries:
        env = CrisisDispatchEnvironment(default_task_id=summary.id)
        episode = _run_heuristic_episode(env, summary.id)
        results[summary.id] = {
            "score": episode["score"],
            "steps": episode["steps"],
            "cumulative_reward": episode["cumulative_reward"],
        }

    return {
        "policy": "inference.heuristic_policy",
        "note": "These scores should match local `python inference.py --mode heuristic --task all --check-determinism`.",
        "results": results,
    }


@app.get("/baseline")
def baseline() -> dict:
    """Run heuristic baseline on the easy task and return scores."""
    env = CrisisDispatchEnvironment(default_task_id="easy")
    return _run_heuristic_episode(env, "easy")


def _run_heuristic_episode(env: CrisisDispatchEnvironment, task_id: str) -> dict:
    obs = env.reset(task_id=task_id)
    rewards = []
    actions = []

    while not obs.done:
        action = heuristic_policy(obs)
        actions.append({"unit_id": action.unit_id, "incident_id": action.incident_id})
        result = env.step(action)
        rewards.append(result.reward)
        obs = result.observation

    grade = env.grade()
    return {
        "task_id": task_id,
        "score": grade.score,
        "cumulative_reward": obs.cumulative_reward,
        "steps": obs.step_count,
        "done": obs.done,
        "rewards": rewards,
        "actions": actions,
        "grade": grade.model_dump(),
    }
