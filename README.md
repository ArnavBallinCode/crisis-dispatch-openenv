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
