# Beginner Guide: Crisis Dispatch OpenEnv

This guide is for someone who knows basic Python but is new to environment design and deployment.

## 1) What this project does

Think of this as a training simulator for emergency operations.

At every step, an agent decides:
- which unit to send
- to which incident
- or whether to wait

The world then updates:
- units travel
- incidents get older
- some incidents escalate
- rewards/penalties are applied

Goal:
- maximize final outcome with limited resources.

## 2) Why this is useful for the challenge

The challenge asks for realistic decision-making under constraints. This project has exactly that:
- limited emergency units
- distance and travel delay
- severity and urgency
- wrong dispatch penalties
- multi-agency coordination in hard mode

## 3) Understand tasks clearly

### easy
- one critical incident
- one correct responder type
- perfect for understanding action format and environment loop

### medium
- many incidents concurrently
- not enough units to do everything immediately
- you must prioritize and avoid wasteful dispatches

### hard
- multi-agency incidents need combinations of responders
- conflicting priorities and tighter failure windows
- requires planning beyond nearest-unit greedy behavior

## 4) Setup once (local)

From project root:

```bash
cd /Users/arnavangarkar/Desktop/crisis-dispatch-openenv/crisis-dispatch-env
/usr/bin/python3 -m pip install -r requirements.txt
cp .env.example .env
```

## 5) .env explained

Your local `.env` can contain:
- `PORT`
- `OPENAI_API_KEY` / `API_KEY` / `GROQ_API_KEY`
- `OPENAI_BASE_URL` / `API_BASE_URL` / `GROQ_BASE_URL`
- `OPENAI_MODEL` / `MODEL_NAME` / `GROQ_MODEL`

For Groq, commonly:
- `API_BASE_URL=https://api.groq.com/openai/v1`
- `MODEL_NAME=llama-3.1-8b-instant`

Never commit real keys.

## 6) Start API and test

Run server:

```bash
/usr/bin/python3 -m uvicorn app.main:app --app-dir /Users/arnavangarkar/Desktop/crisis-dispatch-openenv/crisis-dispatch-env --host 127.0.0.1 --port 8011
```

In another terminal:

```bash
curl -s http://127.0.0.1:8011/health
curl -s http://127.0.0.1:8011/tasks
curl -s -X POST http://127.0.0.1:8011/reset/hard
curl -s -X POST http://127.0.0.1:8011/step -H 'Content-Type: application/json' -d '{"unit_id":"A1","incident_id":"H-MED-1"}'
curl -s http://127.0.0.1:8011/state
curl -s http://127.0.0.1:8011/score
```

## 7) Baseline runs

Deterministic baseline:

```bash
/usr/bin/python3 inference.py --mode heuristic --task all --check-determinism
```

Random baseline:

```bash
/usr/bin/python3 inference.py --mode random --task all --episodes 1 --seed 2026
```

OpenAI-compatible provider mode (OpenAI or Groq):

```bash
/usr/bin/python3 inference.py --mode openai --task easy --episodes 1
```

## 8) Docker validation

Build:

```bash
docker build -t crisis-dispatch-env .
```

Run:

```bash
docker run --rm -p 7860:7860 crisis-dispatch-env
```

Check:

```bash
curl -s http://127.0.0.1:7860/health
```

## 9) Hugging Face Spaces

Your Space:
- `blackmamba2408/Crisis-Dispatch-OpenEnv`

Push path:

```bash
git push hf main
```

If git auth fails despite login, use HF upload:

```bash
HF_TOKEN="$(cat ~/.cache/huggingface/token)" hf upload blackmamba2408/Crisis-Dispatch-OpenEnv /Users/arnavangarkar/Desktop/crisis-dispatch-openenv/crisis-dispatch-env . --repo-type=space --exclude ".git/*" --commit-message "Sync update"
```

Live URL:
- `https://blackmamba2408-crisis-dispatch-openenv.hf.space`

## 10) If something breaks

`ModuleNotFoundError: No module named app`
- run uvicorn with `--app-dir` absolute path.

Space stuck building
- check Space API runtime stage and build logs.
- ensure Dockerfile listens on port `7860`.
- ensure dependencies install cleanly.

OpenAI/Groq inference fails
- verify key variable exists in `.env`
- verify base URL is correct for provider
- verify model id is valid for your provider
