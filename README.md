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

This project is designed for hackathon-quality evaluation:
- Real-world utility: multi-agency response with limited resources.
- Three difficulties: easy, medium, hard.
- Deterministic grading in [0.0, 1.0].
- Dense reward shaping for training and evaluation.
- Docker and Hugging Face Spaces readiness.

## Documentation Map

- Start here for full beginner explanation:
  - `BEGINNER_DEPLOYMENT_GUIDE.md`
- OpenEnv and project summary:
  - `README.md` (this file)
- Environment contract details:
  - `openenv.yaml`

## Quick Start (Local)

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Prepare environment file:

```bash
cp .env.example .env
```

3. Start API server:

```bash
uvicorn app.main:app --app-dir . --host 0.0.0.0 --port 7860
```

4. Test basic endpoint:

```bash
curl -s http://127.0.0.1:7860/health
```

5. Run deterministic baseline:

```bash
python inference.py --mode heuristic --task all --check-determinism
```

## Project Layout

```text
crisis-dispatch-env/
├── .dockerignore
├── .env.example
├── Dockerfile
├── BEGINNER_DEPLOYMENT_GUIDE.md
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

## Task Difficulty Design

- easy
  - One critical medical incident.
  - One correct unit type is enough.
  - Teaches action format and travel-time effect.

- medium
  - Multiple incidents at once.
  - Limited unit pool.
  - Requires prioritization and avoiding wrong dispatches.

- hard
  - Multi-agency incidents with conflicting priorities.
  - Tradeoff pressure: severity vs distance vs resource lock-up.
  - Includes cascade risk if critical events are delayed.

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

## API Endpoints

- `GET /health`
- `GET /tasks`
- `POST /reset`
- `POST /reset/{task_id}`
- `POST /step`
- `GET /state`
- `GET /score`

## Example API Interaction

```bash
curl -s http://localhost:7860/reset/easy
curl -s http://localhost:7860/step \
  -H 'Content-Type: application/json' \
  -d '{"unit_id":"A1","incident_id":"E-MED-1"}'
curl -s http://localhost:7860/state
curl -s http://localhost:7860/score
```

## Inference Baseline

`inference.py` includes:
- `heuristic` policy (deterministic baseline)
- `random` policy (seeded baseline)
- `openai` policy using the OpenAI client

Environment notes:
- If `OPENAI_API_KEY` is present in `.env`, openai mode can run without exporting each time.
- `OPENAI_MODEL` in `.env` can be used as your default model choice in your own wrappers.

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

If you see daemon/socket errors, start Docker Desktop or OrbStack first.

## HuggingFace Spaces Deployment

This repository is Docker-compatible for Spaces:
- `Dockerfile` launches `uvicorn app.main:app` on `${PORT:-7860}`.
- `openenv.yaml` includes Spaces runtime hints.

### Your Space

- Space repo: `https://huggingface.co/spaces/blackmamba2408/Crisis-Dispatch-OpenEnv`
- Public app URL pattern: `https://blackmamba2408-Crisis-Dispatch-OpenEnv.hf.space`

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

If already logged in with `hf auth login`, git push should reuse stored credentials.

## Reproducibility Notes

- Environment transitions are deterministic.
- Task definitions are static.
- Grader has no stochastic components.
- Random baseline mode is explicitly seeded.
- Heuristic baseline scores are reproducible across repeated runs.
