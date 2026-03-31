---
title: Crisis Dispatch OpenEnv
emoji: 🚑
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Crisis Resource Dispatch Environment

A deterministic OpenEnv environment for crisis resource dispatch.

This project simulates real emergency decision-making with limited responders, travel-time constraints, severity escalation, and multi-agency coordination.

## Start Here

- Overview and quick commands: `README.md`
- Detailed docs index: `docs/README.md`
- Challenge mapping and deep environment mechanics:
  - `docs/CHALLENGE_ALIGNMENT_AND_ENVIRONMENT.md`
- Beginner setup and usage:
  - `docs/BEGINNER_GUIDE.md`
- Hugging Face deployment and stuck-build troubleshooting:
  - `docs/DEPLOYMENT_HF_SPACE.md`

## Quick Commands

Install:

```bash
pip install -r requirements.txt
cp .env.example .env
```

Run API:

```bash
uvicorn app.main:app --app-dir . --host 0.0.0.0 --port 7860
```

Run deterministic baseline:

```bash
python inference.py --mode heuristic --task all --check-determinism
```

Run tests:

```bash
pip install pytest
pytest -q
```

Run OpenAI-compatible provider baseline (OpenAI or Groq endpoint):

```bash
python inference.py --mode openai --task easy --episodes 1
```

Docker build/run:

```bash
docker build -t crisis-dispatch-env .
docker run --rm -p 7860:7860 crisis-dispatch-env
```

## Current Space

- Repo: `https://huggingface.co/spaces/blackmamba2408/Crisis-Dispatch-OpenEnv`
- Host: `https://blackmamba2408-crisis-dispatch-openenv.hf.space`

## CI + Auto Sync Pipeline

This repository now includes a GitHub Actions workflow at
`.github/workflows/ci-hf-sync.yml`.

Pipeline behavior:

1. Runs on pull requests to `main`, pushes to `main`, and manual `workflow_dispatch`.
2. Installs dependencies and runs `pytest -q`.
3. Runs a deterministic baseline check:
   `python inference.py --mode heuristic --task all --check-determinism`.
4. If tests pass on `main`, pushes the latest commit to your Hugging Face Space.

Required GitHub repository secrets:

- `HF_TOKEN`: Hugging Face token with write access to the target Space.
- `HF_SPACE_ID`: Space id in `owner/space-name` format.
  Example: `blackmamba2408/Crisis-Dispatch-OpenEnv`.

Notes:

- The sync job is blocked unless test checks pass.
- Pull requests run validation only (no Space sync).
- Deployment uses git push to Hugging Face so the Space reflects the latest
  repository state.

## Project Layout

```text
crisis-dispatch-env/
├── .dockerignore
├── .env.example
├── .gitignore
├── Dockerfile
├── README.md
├── docs/
│   ├── README.md
│   ├── BEGINNER_GUIDE.md
│   ├── CHALLENGE_ALIGNMENT_AND_ENVIRONMENT.md
│   └── DEPLOYMENT_HF_SPACE.md
├── inference.py
├── openenv.yaml
├── requirements.txt
└── app/
    ├── environment.py
    ├── main.py
    ├── models.py
    └── tasks.py
```
