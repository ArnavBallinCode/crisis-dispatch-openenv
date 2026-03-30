# Beginner Guide: Crisis Dispatch OpenEnv Project

This guide explains the project as if you are new to environments and deployment.
It is intentionally plain-English and step-by-step.

## 1) What this project is, in simple words

You built a small "world" where emergency incidents happen and an agent chooses who to dispatch.

Resources:
- ambulance
- fire truck
- police

Incident examples:
- medical
- fire
- traffic

Why this is realistic:
- units are limited
- units have travel time
- sending the wrong unit wastes time
- incidents can worsen if you delay

So the objective is not "one perfect action". It is a sequence of good tradeoffs.

## 2) Core concepts you should know

- state
  - current snapshot of city, units, incidents, and metrics.

- action
  - one dispatch decision per step.
  - either send one unit to one incident, or wait.

- reward
  - immediate feedback each step.
  - positive when you make progress, negative when you delay/mis-dispatch/fail incidents.

- episode
  - one full run of a task from reset until done.

- score
  - final deterministic quality score between 0.0 and 1.0.

## 3) What each task actually means

### easy task: Single Medical Priority

What you see:
- one critical medical incident
- a few units available

What good behavior looks like:
- dispatch an ambulance quickly
- avoid useless actions

What this teaches:
- action format
- travel time impact
- basic reward mechanics

### medium task: Limited Fleet Prioritization

What you see:
- multiple incidents at once
- incidents with different severities
- too few units to do everything instantly

What good behavior looks like:
- respond to critical incidents first
- avoid wrong unit dispatch
- choose where each unit gives most value

What this teaches:
- prioritization under constraints
- managing opportunity cost

### hard task: Multi-Agency Incident Cascade

What you see:
- several incidents with conflicting urgency
- some incidents require multiple unit types
- if you delay, problems cascade

What good behavior looks like:
- coordinate unit types, not just nearest unit
- resolve critical incidents before they fail
- avoid locking key units in low-value routes

What this teaches:
- real tradeoff reasoning
- non-greedy multi-step planning

## 4) How scoring works (plain explanation)

Final score combines:
- weighted success
  - resolving critical incidents matters more than low severity incidents.
- weighted timeliness
  - faster response earns more.
- dispatch accuracy
  - fewer wrong dispatches improves score.
- critical failure penalty
  - unresolved critical incidents heavily reduce score.

This is deterministic, so same policy + same task => same final score.

## 5) Project files and why they exist

```text
crisis-dispatch-env/
├── .env.example
├── .gitignore
├── Dockerfile
├── BEGINNER_DEPLOYMENT_GUIDE.md
├── README.md
├── inference.py
├── openenv.yaml
├── requirements.txt
└── app/
    ├── environment.py
    ├── main.py
    ├── models.py
    └── tasks.py
```

Key files:
- app/models.py
  - data contracts (actions, state, score output)
- app/tasks.py
  - easy/medium/hard task definitions and deterministic grader
- app/environment.py
  - reset, step, state mechanics
- app/main.py
  - FastAPI routes for interacting with environment
- inference.py
  - baseline policies and reproducibility checks
- openenv.yaml
  - OpenEnv metadata and contract declaration
- .env.example
  - safe template of local environment variables

## 6) Environment variables (.env) explained

This project supports loading variables from a local `.env` file.

Create your local file from template:

```bash
cp .env.example .env
```

Variables used:
- PORT
  - server port, default 7860 in Docker context.
- OPENAI_API_KEY
  - only needed for `inference.py --mode openai`.
- OPENAI_MODEL
  - optional model name for your own wrappers/scripts.
- HF_SPACE_ID
  - helper metadata value for documentation/workflow.

Important safety rule:
- never commit real tokens into git.

## 7) Local run: exact sequence

From project root:

```bash
cd /Users/arnavangarkar/Desktop/crisis-dispatch-openenv/crisis-dispatch-env
```

Install dependencies:

```bash
/usr/bin/python3 -m pip install -r requirements.txt
```

Start API (this command avoids module path errors):

```bash
/usr/bin/python3 -m uvicorn app.main:app --app-dir /Users/arnavangarkar/Desktop/crisis-dispatch-openenv/crisis-dispatch-env --host 127.0.0.1 --port 8011
```

In another terminal, smoke test:

```bash
curl -s http://127.0.0.1:8011/health
curl -s http://127.0.0.1:8011/tasks
curl -s -X POST http://127.0.0.1:8011/reset/hard
curl -s -X POST http://127.0.0.1:8011/step -H 'Content-Type: application/json' -d '{"unit_id":"A1","incident_id":"H-MED-1"}'
curl -s http://127.0.0.1:8011/state
curl -s http://127.0.0.1:8011/score
```

## 8) Baseline inference runs

Deterministic heuristic baseline:

```bash
/usr/bin/python3 inference.py --mode heuristic --task all --check-determinism
```

Seeded random baseline:

```bash
/usr/bin/python3 inference.py --mode random --task all --episodes 1 --seed 2026
```

OpenAI mode (needs OPENAI_API_KEY in env):

```bash
/usr/bin/python3 inference.py --mode openai --task hard --model gpt-4.1-mini
```

## 9) Docker: what to do and what to expect

Build image:

```bash
docker build -t crisis-dispatch-env .
```

Run container:

```bash
docker run --rm -p 7860:7860 crisis-dispatch-env
```

Check it is live:

```bash
curl -s http://127.0.0.1:7860/health
```

If Docker fails with daemon/socket errors, start Docker Desktop or OrbStack and rerun.

## 10) Hugging Face Spaces: exact flow for your Space

Your Space:
- blackmamba2408/Crisis-Dispatch-OpenEnv

Your current remote is already configured as `hf`.

Optional (clone/check):

```bash
git clone https://huggingface.co/spaces/blackmamba2408/Crisis-Dispatch-OpenEnv
curl -LsSf https://hf.co/cli/install.sh | bash
hf download blackmamba2408/Crisis-Dispatch-OpenEnv --repo-type=space
```

Push current project to Space:

```bash
git remote -v
git push hf main
```

After push:
1. Open the Space page.
2. Watch build logs.
3. Wait until build is green.
4. Test endpoint:

```bash
curl -s https://blackmamba2408-Crisis-Dispatch-OpenEnv.hf.space/health
```

## 11) Common errors and direct fixes

### Error: ModuleNotFoundError: No module named app

Reason:
- you started uvicorn from outside project root without app-dir.

Fix:
- use `--app-dir /absolute/path/to/crisis-dispatch-env`.

### Error: push rejected to hf

Reason:
- auth token missing/invalid, or wrong account permissions.

Fix:
1. Run `hf auth login`.
2. Confirm token has write scope.
3. Retry `git push hf main`.

### Error: openai mode fails

Reason:
- `OPENAI_API_KEY` is not set.

Fix:
- add it to `.env` or export in shell.

## 12) Final checklist before submission

- [ ] local API responds on health/tasks/reset/step/state/score
- [ ] heuristic determinism check passes
- [ ] Docker image builds and runs
- [ ] Space build succeeds on Hugging Face
- [ ] score and task explanation included in README
- [ ] no secrets committed

If you complete this checklist, your environment is in strong hackathon submission shape.
