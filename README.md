---
title: Crisis Dispatch OpenEnv
emoji: 🚑
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
tags:
  - openenv
pinned: false
---

# Crisis Resource Dispatch Environment

A deterministic OpenEnv environment simulating real-world emergency dispatch operations.

An agent must manage a limited fleet of emergency units (ambulances, fire trucks, police) across a city grid, responding to incidents with varying severity, time constraints, and resource requirements.

## Environment Description

**Domain**: Emergency response dispatch planning  
**Task type**: Sequential decision making under resource constraints  
**Reward type**: Dense (step-level signals throughout the episode)

The environment models the real challenge dispatchers face: prioritizing which incidents to respond to first, routing the right unit types, managing travel time, and handling incident escalation when responses are delayed.

## Action Space

Each step the agent submits one of:

| Action | Description |
|--------|-------------|
| `Action(unit_id="A1", incident_id="E-MED-1")` | Dispatch unit to incident |
| `Action()` (both null) | Wait — no dispatch this step |

## Observation Space

The `Observation` model returned by `reset()` and `step()` includes:

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | `str` | Active task name |
| `step_count` | `int` | Steps taken so far |
| `max_steps` | `int` | Episode step limit |
| `units` | `List[UnitState]` | Fleet status (position, availability, target) |
| `incidents` | `List[IncidentState]` | Active incidents (severity, elapsed, responding units) |
| `metrics` | `EnvironmentMetrics` | Running dispatch accuracy, resolution counts |
| `done` | `bool` | Whether the episode has ended |
| `cumulative_reward` | `float` | Total reward so far |

## Tasks

| Task ID | Difficulty | Description | Max Steps |
|---------|-----------|-------------|-----------|
| `easy` | Easy | One critical medical incident, 3 units, 6×6 grid | 18 |
| `medium` | Medium | 4 incidents, 4 units, 8×8 grid, resource contention | 30 |
| `hard` | Hard | 5 incidents, 6 units, 10×10 grid, multi-agency coordination required | 40 |

## Reward Function

Step-level rewards provide dense signal throughout the episode:

- **+dispatch bonus** (`+0.35`): Correct unit type dispatched to valid incident
- **+arrival stabilization** (`+0.40`): Unit arrives and contributes required response type
- **+resolution reward** (`+1.0–3.0 × timeliness factor`): Incident resolved by responding unit; scaled by `initial_severity` (not current, to avoid escalation farming)
- **-wait penalty** (`-0.02`): Each step with no dispatch
- **-wrong dispatch** (`-0.70`): Wrong unit type for incident
- **-unit unavailable** (`-0.35`): Dispatching busy unit
- **-invalid target** (`-0.50`): Dispatch references unknown unit or incident
- **-active incident wait cost** (`-0.02/-0.05/-0.09`): Applied each step for low/medium/critical unresolved incidents
- **-escalation penalty** (`-0.20`): Applied when an incident severity escalates
- **-failure penalty** (`-1.1–3.6`): Incident exceeds `max_wait` without resolution
- **-timeout close penalty** (`-0.75 × failure penalty`): Applied when max episode steps are reached with open incidents

The deterministic grader combines weighted success, timeliness, dispatch accuracy, first-response progress, and unit-coverage progress, with penalties for failed critical incidents and unresolved critical pressure near deadline.

## Baseline Scores (Heuristic Policy)

| Task | Score | Steps |
|------|-------|-------|
| easy | 0.950 | 3 |
| medium | 0.946 | 4 |
| hard | 0.778 | 8 |

Scores are deterministic and reproducible. Run:

```bash
uv run inference.py --mode heuristic --task all --emit-summary --check-determinism
```

By default, `inference.py` emits only strict structured logs (`[START]`, `[STEP]`, `[END]`) for evaluator compatibility.

## Setup & Usage

### Install

```bash
uv venv --python 3.12 .venv
uv pip install -r requirements.txt
```

### Run API Server

```bash
uv run -m uvicorn app.main:app --host 0.0.0.0 --port 7860
```

### Run Baseline Inference

```bash
uv run inference.py --mode heuristic --task all
```

### Run Tests

```bash
uv run -m pytest tests/ -v
```

### Validate OpenEnv Compliance

```bash
uvx --from "openenv-core[cli]" openenv validate
```

### Docker Build & Run

```bash
docker build -t crisis-dispatch-env .
docker run --rm -p 7860:7860 crisis-dispatch-env
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | `GET` | Interactive demo UI for reset/step/state/grader testing |
| `/meta` | `GET` | Lightweight JSON status metadata |
| `/demo/run/{task_id}` | `POST` | Auto-runs a full heuristic episode on selected task |
| `/demo/benchmark` | `GET` | Runs deterministic heuristic on easy/medium/hard |
| `/reset` | `POST` | Start new episode |
| `/step` | `POST` | Submit an action |
| `/state` | `GET` | Current environment state |
| `/tasks` | `GET` | List all available tasks |
| `/grader` | `GET` | Score the current episode |
| `/baseline` | `GET` | Run heuristic baseline on easy task |
| `/health` | `GET` | Health check |

The Hugging Face Space homepage is intentionally interactive so reviewers can validate the OpenEnv interface behavior directly from the Space URL without external tools.

Note: `/grader` reports score for the current in-progress episode state. For a complete non-zero demo run in one call, use `/demo/run/{task_id}` or the homepage auto-demo button.

## Using The Space Demo (Step By Step)

When you open the Space URL, you can simulate an episode directly in the browser.

1. Choose a task (`easy`, `medium`, `hard`) and click **Reset Task**.
2. Select a `unit_id` and an `incident_id` from dropdowns populated from the latest state.
3. Click **Step Dispatch** to send that unit to that incident.
4. Click **Step Wait** when no dispatch should be made this turn.
5. Click **Get Score** to inspect the current grading snapshot.
6. Click **Run Auto Demo (selected task)** to execute the full deterministic heuristic episode.
7. Click **Run Benchmark (all tasks)** to compare easy/medium/hard scores in one shot.

### What `unit_id` and `incident_id` mean

- `unit_id`: emergency resource to dispatch (examples: `A1`, `F1`, `P1`)
- `incident_id`: active emergency call (examples: `E-MED-1`, `M-FIRE-1`, `H-TRAFFIC-1`)

The demo UI only shows currently valid options to reduce invalid actions.

## Score Consistency: Local vs Hugging Face Space

The Space demo endpoints use the **same deterministic heuristic policy** as `inference.py`.

- Local command:

```bash
python inference.py --mode heuristic --task all --emit-summary --check-determinism
```

- Space benchmark endpoint:

```bash
curl -s https://blackmamba2408-crisis-dispatch-openenv.hf.space/demo/benchmark
```

If the Space has finished rebuilding from the latest commit, these should align closely task-by-task.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | Yes (for LLM mode) | Hugging Face API token |
| `API_BASE_URL` | No | LLM endpoint (default: HF router) |
| `MODEL_NAME` | No | Model identifier (default: Qwen/Qwen2.5-72B-Instruct) |

## Project Layout

```text
crisis-dispatch-openenv/
├── Dockerfile
├── README.md
├── openenv.yaml
├── requirements.txt
├── inference.py
└── app/
    ├── environment.py   ← Core simulation (step/reset/state/close)
    ├── main.py          ← FastAPI server + all endpoints
    ├── models.py        ← Typed models: Action, Observation, Reward
    └── tasks.py         ← Task definitions + deterministic grader
```

## HF Space

- Space: `https://huggingface.co/spaces/blackmamba2408/Crisis-Dispatch-OpenEnv`
- Host: `https://blackmamba2408-crisis-dispatch-openenv.hf.space`
