# Beginner Guide: Crisis Dispatch OpenEnv Project

This guide is written for someone who knows basic Python but is new to environment projects, deployment, and hackathon packaging.

## 1) What this project is

You built a simulation called **Crisis Resource Dispatch Environment**.

Think of it as a training world where an AI agent learns to dispatch emergency units:
- Ambulance
- Fire truck
- Police

The agent must make decisions under constraints:
- Limited units
- Travel time (distance matters)
- Different incident types need different units
- Incidents can get worse over time

The environment gives:
- A **state** (what is happening now)
- A way to take an **action** (dispatch a unit)
- A **reward** signal (good or bad decision quality)
- A final **score** (0.0 to 1.0)

---

## 2) Project structure and what each file does

```text
crisis-dispatch-env/
├── BEGINNER_DEPLOYMENT_GUIDE.md
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

### Root files

- `requirements.txt`
  - Python dependencies.
  - Install these before running the project.

- `Dockerfile`
  - Defines how to build a container image.
  - Needed for consistent runtime and Hugging Face Docker Spaces.

- `openenv.yaml`
  - Metadata/config for OpenEnv.
  - Declares task ids, models, interface methods, grader function, and runtime hints.

- `inference.py`
  - Runs baseline policies (heuristic/random/openai).
  - Produces reproducible scores for easy/medium/hard tasks.

- `README.md`
  - Technical project documentation for judges and collaborators.

- `BEGINNER_DEPLOYMENT_GUIDE.md` (this file)
  - Beginner-first setup and deployment checklist.

### app folder

- `app/models.py`
  - Typed Pydantic models for actions, units, incidents, environment state, and score output.

- `app/tasks.py`
  - Three tasks (easy/medium/hard).
  - Severity weights and deterministic grading logic.

- `app/environment.py`
  - Core simulator logic:
    - `reset()`
    - `step(action)`
    - `state()`
  - Handles travel, penalties, escalation, resolution, and episode termination.

- `app/main.py`
  - FastAPI server and endpoints:
    - `GET /health`
    - `GET /tasks`
    - `POST /reset`
    - `POST /reset/{task_id}`
    - `POST /step`
    - `GET /state`
    - `GET /score`

---

## 3) How the environment works (simple mental model)

### Step-by-step loop

1. You call `reset(task_id)` to start a scenario.
2. You get a state containing units + incidents.
3. You pick an action (dispatch unit X to incident Y, or wait).
4. `step(action)` updates the world by one tick:
   - unit moves closer
   - incident waits/escalates
   - rewards/penalties are applied
5. Repeat until done.
6. Read final score from `GET /score` or grader output.

### Why score can be low

- Wrong unit sent
- High-severity incident delayed
- Critical incident unresolved
- Too many steps with waiting/escalation

---

## 4) Local setup (exact commands)

Run from the project root:

```bash
cd /Users/arnavangarkar/Desktop/crisis-dispatch-openenv/crisis-dispatch-env
```

Install dependencies:

```bash
/usr/bin/python3 -m pip install -r requirements.txt
```

Start API server:

```bash
/usr/bin/python3 -m uvicorn app.main:app --app-dir /Users/arnavangarkar/Desktop/crisis-dispatch-openenv/crisis-dispatch-env --host 127.0.0.1 --port 8011
```

In a second terminal, smoke test:

```bash
curl -s http://127.0.0.1:8011/health
curl -s http://127.0.0.1:8011/tasks
curl -s -X POST http://127.0.0.1:8011/reset/hard
curl -s -X POST http://127.0.0.1:8011/step -H 'Content-Type: application/json' -d '{"unit_id":"A1","incident_id":"H-MED-1"}'
curl -s http://127.0.0.1:8011/state
curl -s http://127.0.0.1:8011/score
```

---

## 5) Baseline inference (reproducible)

Heuristic baseline + determinism check:

```bash
/usr/bin/python3 /Users/arnavangarkar/Desktop/crisis-dispatch-openenv/crisis-dispatch-env/inference.py --mode heuristic --task all --check-determinism
```

Random baseline (seeded):

```bash
/usr/bin/python3 /Users/arnavangarkar/Desktop/crisis-dispatch-openenv/crisis-dispatch-env/inference.py --mode random --task all --episodes 1 --seed 2026
```

OpenAI policy:

```bash
OPENAI_API_KEY=your_key_here /usr/bin/python3 /Users/arnavangarkar/Desktop/crisis-dispatch-openenv/crisis-dispatch-env/inference.py --mode openai --task hard --model gpt-4.1-mini
```

---

## 6) Docker workflow

Build image:

```bash
docker build -t crisis-dispatch-env .
```

Run container:

```bash
docker run --rm -p 7860:7860 crisis-dispatch-env
```

Test container API:

```bash
curl -s http://127.0.0.1:7860/health
```

If Docker command fails with daemon/socket error, start Docker Desktop or OrbStack first, then retry.

---

## 7) Hugging Face Spaces deployment (exact beginner flow)

### Step A: Prepare repository

Make sure these files are present at repo root:
- `Dockerfile`
- `requirements.txt`
- `openenv.yaml`
- `app/` folder

### Step B: Create Space

1. Go to Hugging Face.
2. Click **New Space**.
3. Choose:
   - SDK: **Docker**
   - Visibility: public/private (your choice)
4. Create the Space.

### Step C: Push code to the Space repo

In your local project:

```bash
git remote add hf https://huggingface.co/spaces/<username>/<space-name>
git push hf main
```

If your default branch is `master`, use that instead of `main`.

### Step D: Wait for build logs

Hugging Face will build the Docker image automatically.
- If build fails, open build logs and fix dependency/runtime issues.
- If build succeeds, app URL becomes live.

### Step E: Verify live endpoints

Use your Space URL:

```bash
curl -s https://<username>-<space-name>.hf.space/health
curl -s https://<username>-<space-name>.hf.space/tasks
```

### Your exact Space commands

Use these exact values for your Space:

```bash
git clone https://huggingface.co/spaces/blackmamba2408/Crisis-Dispatch-OpenEnv
curl -LsSf https://hf.co/cli/install.sh | bash
hf download blackmamba2408/Crisis-Dispatch-OpenEnv --repo-type=space
```

From this project folder, connect and push:

```bash
git remote add hf https://huggingface.co/spaces/blackmamba2408/Crisis-Dispatch-OpenEnv
git push hf main
```

When asked for password, use your Hugging Face access token (write permissions required):
- https://huggingface.co/settings/tokens

---

## 8) Hackathon submission checklist

- [ ] `reset()`, `step(action)`, `state()` implemented
- [ ] typed Pydantic models implemented
- [ ] `openenv.yaml` included
- [ ] easy/medium/hard tasks included
- [ ] deterministic grader outputs 0.0-1.0
- [ ] dense reward function included
- [ ] `inference.py` with OpenAI client included
- [ ] baseline scores documented and reproducible
- [ ] Docker build works
- [ ] Hugging Face Space deployed and reachable

---

## 9) Git workflow to finalize and push

Check status:

```bash
git status
```

Stage everything:

```bash
git add .
```

Commit:

```bash
git commit -m "Add beginner deployment guide and validate crisis dispatch environment"
```

Push:

```bash
git push origin main
```

If your branch is `master`, use:

```bash
git push origin master
```

---

## 10) Common problems and fixes

### Problem: `ModuleNotFoundError: No module named app`
Use `--app-dir` with absolute path in uvicorn command.

### Problem: Docker daemon connection error
Start Docker Desktop or OrbStack and rerun `docker build`.

### Problem: OpenAI mode fails
Set `OPENAI_API_KEY` in your shell before running inference.

### Problem: zero score on hard task
This can happen with weak policies; compare heuristic vs random and inspect dispatch choices.

---

## 11) What to do next (recommended order)

1. Run local API checks.
2. Run heuristic and random baselines.
3. Build Docker image locally.
4. Push to GitHub.
5. Create Hugging Face Docker Space.
6. Push code to Space repo.
7. Validate Space endpoints.
8. Submit with links + scores + explanation of design choices.

You now have a complete, practical environment pipeline from local dev to cloud deployment.
