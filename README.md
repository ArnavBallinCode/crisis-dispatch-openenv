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

- **+dispatch bonus** (`+0.05–0.14`): Correct unit type dispatched to valid incident
- **-wait penalty** (`-0.02`): Each step with no dispatch
- **+resolution reward** (`+1.0–3.0 × timeliness factor`): Incident resolved by responding unit; scaled by `initial_severity` (not current — prevents escalation farming)
- **-wrong dispatch** (`-0.35`): Wrong unit type for incident
- **-unit unavailable** (`-0.35`): Dispatching busy unit
- **-escalation penalty** (`-0.09 per step`): Active critical incident unresolved
- **-failure penalty** (`-1.1–3.6`): Incident exceeds `max_wait` without resolution
- **-critical failure** (`-5.0`): Critical incident expires

## Baseline Scores (Heuristic Policy)

| Task | Score | Steps |
|------|-------|-------|
| easy | 0.932 | 3 |
| medium | 0.927 | 4 |
| hard | 0.752 | 8 |

Scores are deterministic and reproducible. Run:

```bash
uv run inference.py --mode heuristic --task all
```

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
| `/reset` | `POST` | Start new episode |
| `/step` | `POST` | Submit an action |
| `/state` | `GET` | Current environment state |
| `/tasks` | `GET` | List all available tasks |
| `/grader` | `GET` | Score the current episode |
| `/baseline` | `GET` | Run heuristic baseline on easy task |
| `/health` | `GET` | Health check |

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
