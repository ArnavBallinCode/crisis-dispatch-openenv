# Challenge Alignment And Environment Deep Dive

This document explains exactly how the Crisis Dispatch environment satisfies the challenge requirements and what each part of the system does.

## 1) What the challenge is asking for

The hackathon asks for a realistic environment with:
- OpenEnv interface (`reset`, `step`, `state`)
- typed models
- multiple task difficulties
- deterministic grading
- dense reward shaping
- inference/baseline script
- Docker + Hugging Face deployability

## 2) How this project maps to those requirements

### Requirement: Real-world task, not a game
Implemented as emergency response resource allocation:
- multiple agencies (ambulance/fire/police)
- geographic response constraints (grid + travel time)
- severity escalation under delay
- wrong dispatch penalties and finite unit availability

### Requirement: OpenEnv interface
Implemented in `app/environment.py`:
- `reset(task_id)` initializes one scenario
- `step(action)` applies one dispatch decision and advances simulation time
- `state()` returns the full typed snapshot

### Requirement: Typed Pydantic models
Implemented in `app/models.py`:
- `DispatchAction` (action contract)
- `EnvironmentState` (observation contract)
- `StepResult` and `GradeResult` (evaluation contract)

### Requirement: minimum 3 tasks (easy, medium, hard)
Implemented in `app/tasks.py`:
- `easy`: one critical incident
- `medium`: multiple incidents + constrained units
- `hard`: multi-agency incidents with conflicting priorities

### Requirement: deterministic grading (0.0 to 1.0)
Implemented in `app/tasks.py::grade_episode`:
- deterministic weighted aggregation
- no random graders
- final clamped score in [0.0, 1.0]

### Requirement: dense reward function
Implemented in `app/environment.py`:
- positive for correct dispatch and resolution
- penalties for delay, escalation, wrong dispatch, and failures
- per-step signal exists throughout episode

### Requirement: inference.py with reproducible baselines
Implemented in `inference.py`:
- deterministic heuristic baseline
- seeded random baseline
- openai-compatible provider mode (OpenAI or Groq endpoint)

### Requirement: Docker + Hugging Face Spaces
Implemented with:
- `Dockerfile`
- `.dockerignore`
- runtime port wiring (`7860`)
- tested local container run path

## 3) Core simulation model

### Entities
- Units:
  - `ambulance`, `fire_truck`, `police`
  - each has location, availability, and travel state
- Incidents:
  - type (`medical`, `fire`, `traffic`)
  - severity (`low`, `medium`, `critical`)
  - max wait and escalation interval
  - required responder set

### Time model
Each `step(action)` is one time tick:
1. apply dispatch action
2. move en-route units
3. process arrivals and potential resolutions
4. increment incident elapsed timers
5. apply wait/escalation/failure effects
6. compute reward and done flag

### Dispatch constraints
- one dispatch pair per step (`unit_id`, `incident_id`) or wait
- unit must be available
- incident must be active
- wrong unit dispatches incur immediate penalty

### Travel and resolution
- travel uses Manhattan distance
- units become unavailable while en-route
- incident can require one or multiple unit types
- incident resolves only when required unit set is satisfied

## 4) Why hard task requires reasoning

Hard is intentionally not solved by "nearest first":
- multi-agency incidents need composition, not just proximity
- critical incidents have short failure windows
- dispatching one unit affects future coverage
- early wrong choices create cascade failures

So the policy must reason about:
- severity urgency
- travel-time slack
- missing required responder types
- opportunity cost of committing scarce units

## 5) Grading details

Final score combines:
- weighted success
- weighted timeliness
- dispatch accuracy
- critical failure penalty

Key behavior:
- critical incidents have the highest importance weights
- late first response reduces timeliness
- unresolved critical incidents sharply reduce score

## 6) Reward shaping details

Dense reward exists at each step:
- `+` valid dispatch contribution
- `+` stabilization/resolution contribution
- `-` waiting cost per unresolved incident
- `-` escalation cost
- `-` wrong dispatch cost
- `-` failure/timeout penalty

This supports both RL-style optimization and policy debugging via reward traces.

## 7) API surface

Served by `app/main.py`:
- `GET /health`
- `GET /tasks`
- `POST /reset`
- `POST /reset/{task_id}`
- `POST /step`
- `GET /state`
- `GET /score`

## 8) LLM provider compatibility (OpenAI + Groq)

`inference.py` supports OpenAI-compatible APIs using these env aliases:
- API key: `OPENAI_API_KEY` or `API_KEY` or `GROQ_API_KEY`
- base URL: `OPENAI_BASE_URL` or `API_BASE_URL` or `GROQ_BASE_URL`
- model: `OPENAI_MODEL` or `MODEL_NAME` or `GROQ_MODEL`

This allows Groq endpoint usage while keeping one inference implementation.

## 9) Deployment reality check

Local runtime validation completed:
- deterministic baseline runs
- Docker image builds
- container serves API endpoints

Space status should be verified from the Hugging Face API and live endpoint after each push/upload.
