---
title: Crisis Dispatch OpenEnv
emoji: "🚑"
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Crisis Resource Dispatch Environment

A production-style OpenEnv environment that simulates emergency response dispatch across a grid city.

This environment is designed for hackathon judging criteria:
- Real-world utility: constrained, multi-agency emergency operations.
- Task depth: easy, medium, hard scenarios with escalating coordination difficulty.
- Deterministic, interpretable grading in the range 0.0-1.0.
- Dense reward shaping for policy learning and rapid feedback.
- Docker-ready deployment with HuggingFace Spaces compatibility.

## Project Layout

```text
crisis-dispatch-env/
├── Dockerfile
├── README.md
├── inference.py
├── openenv.yaml
├── requirements.txt
└── app/
    ├── __init__.py
    ├── environment.py
    ├── main.py
    ├── models.py
    └── tasks.py
```

## OpenEnv Interface

The simulator implements:
- `reset(task_id=None)`
- `step(action)`
- `state()`

Typed Pydantic models are used for all environment contracts:
- action model: `DispatchAction`
- observation model: `EnvironmentState`
- step output model: `StepResult`

## Simulation Mechanics

### City and travel
- City is represented as a grid.
- Units and incidents have `(x, y)` coordinates.
- Travel time uses Manhattan distance.

### Unit types
- `ambulance`
- `fire_truck`
- `police`

### Incident model
Each incident includes:
- incident type: medical, fire, traffic
- severity: low, medium, critical
- max wait time before becoming unrecoverable
- escalation interval (severity increases over time)
- required unit types (single- or multi-agency)

### Constraints and consequences
- Wrong dispatches apply immediate penalty and consume time.
- Units are unavailable while en route.
- Unresolved incidents incur per-step delay penalties.
- Incident escalation increases downstream penalties.
- Timeout closes unresolved incidents with extra penalty.

## Reward Function (Dense)

Per-step reward includes:
- Positive reward for correct dispatch.
- Larger positive reward for resolving incidents (scaled by severity and timeliness).
- Penalty for incorrect dispatch.
- Penalty for each unresolved incident per step.
- Penalty when incidents escalate.
- Penalty for incident failure or timeout closure.

This creates a continuous optimization signal, not sparse terminal-only reward.

## Deterministic Grader

Final score is in `[0.0, 1.0]` and combines:
- weighted success (critical incidents weighted highest)
- weighted timeliness of first response
- dispatch accuracy
- critical failure penalty

The grader is deterministic and implemented in `app/tasks.py` (`grade_episode`).

## Tasks

- `easy`: single critical medical call with one correct responder type.
- `medium`: multiple concurrent incidents with limited units and prioritization pressure.
- `hard`: multi-agency incidents with conflicting priorities and distance/severity tradeoffs.

## API Endpoints

- `GET /health`
- `GET /tasks`
- `POST /reset`
- `POST /reset/{task_id}`
- `POST /step`
- `GET /state`
- `GET /score`

## Local Run

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Start server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 7860
```

### 3) Example interaction

```bash
curl -s http://localhost:7860/reset/easy
curl -s http://localhost:7860/step \
  -H 'Content-Type: application/json' \
  -d '{"unit_id":"A1","incident_id":"E-MED-1"}'
curl -s http://localhost:7860/score
```

## Inference Baseline

`inference.py` includes:
- `heuristic` policy (deterministic baseline)
- `random` policy (seeded baseline)
- `openai` policy using the OpenAI client

Run baseline:

```bash
python inference.py --mode heuristic --task all --check-determinism
```

### Reproducible baseline scores

Measured on 30 March 2026 with deterministic `heuristic` policy:

| Task | Score | Cumulative Reward | Steps | Resolved | Failed |
|---|---:|---:|---:|---:|---:|
| easy | 0.8325 | 0.6700 | 3 | 1 | 0 |
| medium | 0.7683 | 1.8843 | 10 | 3 | 1 |
| hard | 0.4877 | -10.3100 | 12 | 2 | 3 |

Seeded `random` baseline (`--seed 2026`) for comparison:

| Task | Score |
|---|---:|
| easy | 0.7650 |
| medium | 0.0000 |
| hard | 0.0000 |

Determinism check status for heuristic policy: PASS across all tasks.

Run OpenAI-driven policy (requires API key):

```bash
OPENAI_API_KEY=... python inference.py --mode openai --task hard --model gpt-4.1-mini
```

## Docker

Build:

```bash
docker build -t crisis-dispatch-env .
```

Run:

```bash
docker run --rm -p 7860:7860 crisis-dispatch-env
```

## HuggingFace Spaces Deployment

This repository is Docker-compatible for Spaces:
- `Dockerfile` launches `uvicorn app.main:app` on `${PORT:-7860}`.
- `openenv.yaml` includes Spaces runtime hints.

In a Docker Space:
1. Push this folder contents.
2. Ensure Space SDK is set to Docker.
3. Launch and use the exposed app URL.

### Exact commands for your Space

Your Space repo is:

```bash
https://huggingface.co/spaces/blackmamba2408/Crisis-Dispatch-OpenEnv
```

Clone the Space repo:

```bash
git clone https://huggingface.co/spaces/blackmamba2408/Crisis-Dispatch-OpenEnv
```

Install the Hugging Face CLI:

```bash
curl -LsSf https://hf.co/cli/install.sh | bash
```

Optional download check:

```bash
hf download blackmamba2408/Crisis-Dispatch-OpenEnv --repo-type=space
```

Push this project to your Space directly from this repository:

```bash
git remote add hf https://huggingface.co/spaces/blackmamba2408/Crisis-Dispatch-OpenEnv
git push hf main
```

When prompted for password/credential, use a Hugging Face access token with write permission.

## Reproducibility Notes

- Environment transitions are deterministic.
- Task definitions are static.
- Grader has no stochastic components.
- Random baseline mode is explicitly seeded.
- Heuristic baseline scores are reproducible across repeated runs.
