#!/usr/bin/env bash
#
# validate-submission.sh — OpenEnv Submission Validator
#
# Checks that your HF Space is live, Docker image builds, OpenEnv validate passes,
# baseline inference reproduces, and task/grader sanity checks pass.
#
# Prerequisites:
#   - Docker:       https://docs.docker.com/get-docker/
#   - OpenEnv CLI:  openenv OR uvx (script can use uvx fallback)
#   - curl (usually pre-installed)
#
# Run:
#   curl -fsSL https://raw.githubusercontent.com/<owner>/<repo>/main/scripts/validate-submission.sh | bash -s -- <ping_url> [repo_dir]
#
#   Or download and run locally:
#     chmod +x validate-submission.sh
#     ./validate-submission.sh <ping_url> [repo_dir]
#
# Arguments:
#   ping_url   Your HuggingFace Space URL (e.g. https://your-space.hf.space)
#   repo_dir   Path to your repo (default: current directory)
#
# Examples:
#   ./validate-submission.sh https://my-team.hf.space
#   ./validate-submission.sh https://my-team.hf.space ./my-repo
#

set -uo pipefail

DOCKER_BUILD_TIMEOUT=600
BASELINE_TIMEOUT=1200
if [ -t 1 ]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BOLD='\033[1m'
  NC='\033[0m'
else
  RED='' GREEN='' YELLOW='' BOLD='' NC=''
fi

run_with_timeout() {
  local secs="$1"; shift
  if command -v timeout &>/dev/null; then
    timeout "$secs" "$@"
  elif command -v gtimeout &>/dev/null; then
    gtimeout "$secs" "$@"
  else
    "$@" &
    local pid=$!
    ( sleep "$secs" && kill "$pid" 2>/dev/null ) &
    local watcher=$!
    wait "$pid" 2>/dev/null
    local rc=$?
    kill "$watcher" 2>/dev/null
    wait "$watcher" 2>/dev/null
    return $rc
  fi
}

portable_mktemp() {
  local prefix="${1:-validate}"
  mktemp "${TMPDIR:-/tmp}/${prefix}-XXXXXX" 2>/dev/null || mktemp
}

CLEANUP_FILES=()
cleanup() { rm -f "${CLEANUP_FILES[@]+"${CLEANUP_FILES[@]}"}"; }
trap cleanup EXIT

PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if [ -z "$PING_URL" ]; then
  printf "Usage: %s <ping_url> [repo_dir]\n" "$0"
  printf "\n"
  printf "  ping_url   Your HuggingFace Space URL (e.g. https://your-space.hf.space)\n"
  printf "  repo_dir   Path to your repo (default: current directory)\n"
  exit 1
fi

if ! REPO_DIR="$(cd "$REPO_DIR" 2>/dev/null && pwd)"; then
  printf "Error: directory '%s' not found\n" "${2:-.}"
  exit 1
fi
PING_URL="${PING_URL%/}"
export PING_URL
PASS=0

log()  { printf "[%s] %b\n" "$(date -u +%H:%M:%S)" "$*"; }
pass() { log "${GREEN}PASSED${NC} -- $1"; PASS=$((PASS + 1)); }
fail() { log "${RED}FAILED${NC} -- $1"; }
hint() { printf "  ${YELLOW}Hint:${NC} %b\n" "$1"; }
stop_at() {
  printf "\n"
  printf "${RED}${BOLD}Validation stopped at %s.${NC} Fix the above before continuing.\n" "$1"
  exit 1
}

pick_python_cmd() {
  if [ -x "$REPO_DIR/.venv/bin/python" ]; then
    printf "%s" "$REPO_DIR/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    printf "%s" "python3"
  else
    printf "%s" ""
  fi
}

pick_openenv_cmd() {
  if command -v openenv >/dev/null 2>&1; then
    printf "%s" "openenv"
  elif command -v uvx >/dev/null 2>&1; then
    printf "%s" "uvx --from openenv-core>=0.2.0 openenv"
  else
    printf "%s" ""
  fi
}

ensure_hf_token() {
  if [ -n "${HF_TOKEN:-}" ]; then
    return 0
  fi
  local token_file="${HF_HOME:-$HOME/.cache/huggingface}/token"
  if [ -f "$token_file" ]; then
    HF_TOKEN="$(tr -d '\r\n' < "$token_file")"
    export HF_TOKEN
  fi
}

printf "\n"
printf "${BOLD}========================================${NC}\n"
printf "${BOLD}  OpenEnv Submission Validator${NC}\n"
printf "${BOLD}========================================${NC}\n"
log "Repo:     $REPO_DIR"
log "Ping URL: $PING_URL"
printf "\n"

ensure_hf_token

PY_CMD="$(pick_python_cmd)"
if [ -z "$PY_CMD" ]; then
  fail "No Python interpreter found"
  hint "Install Python 3 and retry."
  stop_at "Setup"
fi

OPENENV_CMD_RAW="$(pick_openenv_cmd)"
if [ -z "$OPENENV_CMD_RAW" ]; then
  fail "No OpenEnv runner found (openenv or uvx)"
  hint "Install one of: pip install openenv-core OR brew/pip install uv"
  stop_at "Setup"
fi

IFS=' ' read -r -a OPENENV_CMD <<< "$OPENENV_CMD_RAW"

log "${BOLD}Step 1/5: Pinging HF Space${NC} ($PING_URL/reset + $PING_URL/tasks) ..."

CURL_OUTPUT=$(portable_mktemp "validate-curl")
CLEANUP_FILES+=("$CURL_OUTPUT")
HTTP_CODE=$(curl -s -o "$CURL_OUTPUT" -w "%{http_code}" -X POST \
  -H "Content-Type: application/json" -d '{}' \
  "$PING_URL/reset" --max-time 30 2>"$CURL_OUTPUT" || printf "000")

if [ "$HTTP_CODE" = "200" ]; then
  pass "HF Space is live and responds to /reset"
elif [ "$HTTP_CODE" = "000" ]; then
  fail "HF Space not reachable (connection failed or timed out)"
  hint "Check your network connection and that the Space is running."
  hint "Try: curl -s -o /dev/null -w '%%{http_code}' -X POST $PING_URL/reset"
  stop_at "Step 1"
else
  fail "HF Space /reset returned HTTP $HTTP_CODE (expected 200)"
  hint "Make sure your Space is running and the URL is correct."
  hint "Try opening $PING_URL in your browser first."
  stop_at "Step 1"
fi

TASKS_OUTPUT=$(portable_mktemp "validate-tasks")
CLEANUP_FILES+=("$TASKS_OUTPUT")
TASKS_CODE=$(curl -s -o "$TASKS_OUTPUT" -w "%{http_code}" "$PING_URL/tasks" --max-time 30 || printf "000")
if [ "$TASKS_CODE" != "200" ]; then
  fail "HF Space /tasks returned HTTP $TASKS_CODE (expected 200)"
  stop_at "Step 1"
fi

TASK_PARSE_OK=false
TASK_PARSE_OUTPUT=$("$PY_CMD" - "$TASKS_OUTPUT" 2>&1 <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text())
if not isinstance(data, list):
    raise SystemExit("/tasks did not return a list")
if len(data) < 3:
    raise SystemExit("/tasks returned fewer than 3 tasks")
ids = [item.get("id") for item in data if isinstance(item, dict)]
print("task_ids:", ids)
PY
) && TASK_PARSE_OK=true

if [ "$TASK_PARSE_OK" = true ]; then
  pass "HF Space /tasks returns >=3 tasks"
else
  fail "Could not validate /tasks content"
  printf "%s\n" "$TASK_PARSE_OUTPUT"
  stop_at "Step 1"
fi

log "${BOLD}Step 2/5: Running docker build${NC} ..."

if ! command -v docker &>/dev/null; then
  fail "docker command not found"
  hint "Install Docker: https://docs.docker.com/get-docker/"
  stop_at "Step 2"
fi

if [ -f "$REPO_DIR/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR"
elif [ -f "$REPO_DIR/server/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR/server"
else
  fail "No Dockerfile found in repo root or server/ directory"
  stop_at "Step 2"
fi

log "  Found Dockerfile in $DOCKER_CONTEXT"

BUILD_OK=false
BUILD_OUTPUT=$(run_with_timeout "$DOCKER_BUILD_TIMEOUT" docker build "$DOCKER_CONTEXT" 2>&1) && BUILD_OK=true

if [ "$BUILD_OK" = true ]; then
  pass "Docker build succeeded"
else
  fail "Docker build failed (timeout=${DOCKER_BUILD_TIMEOUT}s)"
  printf "%s\n" "$BUILD_OUTPUT" | tail -20
  stop_at "Step 2"
fi

log "${BOLD}Step 3/5: Running openenv validate${NC} ..."

VALIDATE_OK=false
VALIDATE_OUTPUT=$(cd "$REPO_DIR" && HF_TOKEN="${HF_TOKEN:-}" "${OPENENV_CMD[@]}" validate 2>&1) && VALIDATE_OK=true

if [ "$VALIDATE_OK" = true ]; then
  pass "openenv validate passed"
  [ -n "$VALIDATE_OUTPUT" ] && log "  $VALIDATE_OUTPUT"
else
  fail "openenv validate failed"
  printf "%s\n" "$VALIDATE_OUTPUT"
  stop_at "Step 3"
fi

log "${BOLD}Step 4/5: Running baseline inference${NC} ..."

if [ ! -f "$REPO_DIR/inference.py" ]; then
  fail "Missing inference.py in repo root"
  stop_at "Step 4"
fi

BASELINE_OK=false
BASELINE_OUTPUT=$(cd "$REPO_DIR" && run_with_timeout "$BASELINE_TIMEOUT" "$PY_CMD" inference.py --mode heuristic --task all --check-determinism 2>&1) && BASELINE_OK=true

if [ "$BASELINE_OK" != true ]; then
  fail "Baseline inference failed or timed out"
  printf "%s\n" "$BASELINE_OUTPUT" | tail -40
  stop_at "Step 4"
fi

TASK_LINES=$(printf "%s\n" "$BASELINE_OUTPUT" | grep -E '^(easy|medium|hard),')
TASK_LINE_COUNT=$(printf "%s\n" "$TASK_LINES" | sed '/^$/d' | wc -l | tr -d ' ')

if [ "${TASK_LINE_COUNT:-0}" -lt 3 ]; then
  fail "Baseline output did not include scores for easy/medium/hard"
  printf "%s\n" "$BASELINE_OUTPUT"
  stop_at "Step 4"
fi

SCORES_OK=true
SCORE_CHECK=$(printf "%s\n" "$TASK_LINES" | awk -F',' '{ if ($2 < 0 || $2 > 1) print $0 }')
if [ -n "$SCORE_CHECK" ]; then
  SCORES_OK=false
fi

if [ "$SCORES_OK" != true ]; then
  fail "Baseline produced score(s) outside [0,1]"
  printf "%s\n" "$SCORE_CHECK"
  stop_at "Step 4"
fi

for t in easy medium hard; do
  if ! printf "%s\n" "$BASELINE_OUTPUT" | grep -q "^$t: PASS"; then
    fail "Determinism check missing PASS for task '$t'"
    printf "%s\n" "$BASELINE_OUTPUT"
    stop_at "Step 4"
  fi
done

pass "Baseline reproduces deterministic scores across tasks"

log "${BOLD}Step 5/5: Task and grader sanity checks${NC} ..."

SANITY_OK=false
SANITY_OUTPUT=$(cd "$REPO_DIR" && "$PY_CMD" - 2>&1 <<'PY'
from app.environment import CrisisDispatchEnvironment
from app.models import DispatchAction
from app.tasks import TASKS

if len(TASKS) < 3:
    raise SystemExit("Need at least 3 tasks")

for task_id in TASKS:
    env = CrisisDispatchEnvironment(default_task_id=task_id)
    state = env.reset(task_id=task_id)
    if state.task_id != task_id:
        raise SystemExit(f"reset returned wrong task id for {task_id}")

    while not state.done:
        state = env.step(DispatchAction()).state

    score = env.grade().score
    if score < 0.0 or score > 1.0:
        raise SystemExit(f"score out of range for {task_id}: {score}")

print(f"task_count={len(TASKS)}")
print("grader_score_range_ok=true")
PY
) && SANITY_OK=true

if [ "$SANITY_OK" = true ]; then
  pass "Task count and grader score range sanity checks passed"
  [ -n "$SANITY_OUTPUT" ] && log "  $SANITY_OUTPUT"
else
  fail "Task/grader sanity checks failed"
  printf "%s\n" "$SANITY_OUTPUT"
  stop_at "Step 5"
fi

printf "\n"
printf "${BOLD}========================================${NC}\n"
printf "${GREEN}${BOLD}  All 5/5 checks passed!${NC}\n"
printf "${GREEN}${BOLD}  Your submission is ready to submit.${NC}\n"
printf "${BOLD}========================================${NC}\n"
printf "\n"

exit 0