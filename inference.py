from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from openai import OpenAI

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

from app.environment import CrisisDispatchEnvironment
from app.models import Action, Observation, Severity
from app.baseline import heuristic_policy, random_policy


if load_dotenv is not None:
    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)


# ---------------------------------------------------------------------------
# Environment variable resolution (strict hackathon rules)
# ---------------------------------------------------------------------------

# Required/optional envs expected by the submission checklist.
HF_TOKEN = os.getenv("HF_TOKEN")
API_BASE_URL = (
    os.getenv("API_BASE_URL")
    or os.getenv("OPENAI_BASE_URL")
    or "https://router.huggingface.co/v1"
)
MODEL_NAME = (
    os.getenv("MODEL_NAME")
    or os.getenv("OPENAI_MODEL")
    or "Qwen/Qwen2.5-72B-Instruct"
)
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE") or "0.0")

# Backward-compatible key alias for local/provider flexibility.
API_KEY = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or HF_TOKEN
ENV_NAME = "crisis-dispatch"


# ---------------------------------------------------------------------------
# Structured stdout logging (MANDATORY format)
# ---------------------------------------------------------------------------


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Heuristic helpers
# ---------------------------------------------------------------------------

SEVERITY_PRIORITY = {
    Severity.LOW: 1.0,
    Severity.MEDIUM: 2.0,
    Severity.CRITICAL: 3.5,
}


def active_incidents(state: Observation):
    return [incident for incident in state.incidents if not incident.resolved and not incident.failed]


def available_units(state: Observation):
    return [unit for unit in state.units if unit.status.value == "available"]


def travel_distance(unit_x: int, unit_y: int, incident_x: int, incident_y: int) -> int:
    return abs(unit_x - incident_x) + abs(unit_y - incident_y)


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


def llm_policy(
    state: Observation,
    client: OpenAI,
    model: str,
    fallback_enabled: bool = False,
    verbose: bool = False,
) -> Action:
    incidents = active_incidents(state)
    units = available_units(state)
    if not incidents or not units:
        return Action()

    # Lean State for Maximum Performance
    dist_matrix = {}
    for unit in state.units:
        if unit.status.value == "available":
            dist_matrix[unit.id] = {
                inc.id: travel_distance(unit.position.x, unit.position.y, inc.position.x, inc.position.y)
                for inc in incidents
            }

    prompt_payload = {
        "step": state.step_count,
        "distances": dist_matrix,
        "units": [
            {"id": u.id, "type": u.unit_type.value, "status": u.status.value, "x": u.position.x, "y": u.position.y, "eta": u.travel_remaining}
            for u in state.units
        ],
        "incidents": [
            {
                "id": inc.id, "type": inc.incident_type.value, "severity": inc.severity.value,
                "missing": [r.value for r in inc.required_units if r not in inc.responding_units],
                "time_left": max(0, inc.max_wait - inc.elapsed)
            }
            for inc in incidents
        ],
    }

    system_prompt = (
        "You are an emergency dispatcher. Return strict JSON only. "
        "Goal: Resolve incidents before time_left hits 0. Priority: Critical > Medium > Low. \n"
        "Format: {\"dispatches\": [{\"unit_id\": \"U1\", \"incident_id\": \"I1\"}, ...]}\n"
        "Use {} if no moves are possible."
    )
    user_prompt = json.dumps(prompt_payload)

    extra_kwargs = {}
    if API_BASE_URL and "nvidia.com" in API_BASE_URL.lower():
        extra_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True, "clear_thinking": True}}

    try:
        completion = client.chat.completions.create(
            model=model,
            temperature=LLM_TEMPERATURE,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1000,
            **extra_kwargs,
        )
        text = (
            completion.choices[0].message.content.strip()
            if completion.choices and completion.choices[0].message.content
            else ""
        )
        if verbose:
            # Windows-safe print: ignore non-ascii for terminal display.
            safe_text = text.encode("ascii", "ignore").decode("ascii")
            print(f"    [LLM DEBUG] Raw output: {safe_text}", flush=True)
    except Exception as e:
        if verbose:
            print(f"    [LLM ERROR] Exception during call: {e}", flush=True)
        if fallback_enabled:
            if verbose:
                print(f"    [LLM FALLBACK] Falling back to heuristic.", flush=True)
            return [heuristic_policy(state)]
        return [Action()]

    if not text:
        if fallback_enabled:
            if verbose:
                print(f"    [LLM FALLBACK] Empty response, falling back to heuristic.", flush=True)
            return [heuristic_policy(state)]
        return [Action()]
        
    # Strip non-JSON content gracefully (CoT might be present)
    clean_text = text.strip()
    if "JSON:" in clean_text:
        clean_text = clean_text.split("JSON:")[-1].strip()
    
    # Standard markdown stripping
    if clean_text.startswith("```json"):
        clean_text = clean_text[7:].strip()
    elif clean_text.startswith("```"):
        clean_text = clean_text[3:].strip()
    if clean_text.endswith("```"):
        clean_text = clean_text[:-3].strip()

    # Final attempt to find the outer {} if the model added prose after JSON
    start = clean_text.find("{")
    end = clean_text.rfind("}")
    if start != -1 and end != -1:
        clean_text = clean_text[start : end + 1]

    try:
        payload = json.loads(clean_text)
        # Support both {"unit_id": "...", "incident_id": "..."} and {"dispatches": [...]}
        dispatches = payload.get("dispatches", [])
        if not dispatches and "unit_id" in payload:
            dispatches = [payload]
            
        actions = []
        for d in dispatches:
            uid = d.get("unit_id")
            iid = d.get("incident_id")
            if uid and iid:
                actions.append(Action(unit_id=uid, incident_id=iid))
        
        return actions if actions else [Action()]
    except Exception as e:
        if verbose:
            print(f"    [LLM ERROR] JSON parse failure: {e}", flush=True)
        if fallback_enabled:
            if verbose:
                print(f"    [LLM FALLBACK] Falling back to heuristic.", flush=True)
            return [heuristic_policy(state)]
        return [Action()]


# ---------------------------------------------------------------------------
# Episode runner with mandatory logging
# ---------------------------------------------------------------------------


@dataclass
class EpisodeResult:
    task_id: str
    score: float
    cumulative_reward: float
    steps: int
    resolved: int
    failed: int


def run_episode(
    task_id: str,
    policy: Callable[[Observation], Action],
    model_name: str,
    silent: bool = False,
) -> EpisodeResult:
    env = CrisisDispatchEnvironment(default_task_id=task_id)
    obs = env.reset(task_id=task_id)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    if not silent:
        log_start(task=task_id, env=ENV_NAME, model=model_name)

    action_buffer: List[Action] = []
    known_incident_ids = {inc.id for inc in obs.incidents}
    
    try:
        while not obs.done:
            # If buffer is empty, re-prompt the fleet policy
            if not action_buffer:
                result_actions = policy(obs)
                # handle if policy returns a single action vs list
                if isinstance(result_actions, list):
                    action_buffer.extend(result_actions)
                else:
                    action_buffer.append(result_actions)
            
            # Pop next action (safely default to wait)
            next_action = action_buffer.pop(0) if action_buffer else Action()
            
            # Final safety: Ensure LLM didn't pick a busy unit (common error)
            # If unit is not available, we convert to wait() to avoid environment errors
            busy_units = {u.id for u in obs.units if u.status.value != "available"}
            if next_action.unit_id in busy_units:
                next_action = Action()

            # Build action string for logging
            if next_action.unit_id and next_action.incident_id:
                action_str = f"dispatch({next_action.unit_id},{next_action.incident_id})"
            else:
                action_str = "wait()"

            result = env.step(next_action)
            reward = result.reward
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken += 1
            obs = result.observation
            
            # If a NEW incident appears, clear buffer to allow re-planning (Dynamic)
            current_ids = {inc.id for inc in obs.incidents}
            if not current_ids.issubset(known_incident_ids):
                 action_buffer = []
                 known_incident_ids = current_ids

            if not silent:
                log_step(step=steps_taken, action=action_str, reward=reward, done=done, error=error)

            if done:
                break

        grade = env.grade()
        score = grade.score
        success = score >= 0.5

        resolved = sum(1 for inc in obs.incidents if inc.resolved and not inc.failed)
        failed = sum(1 for inc in obs.incidents if inc.failed)

    finally:
        try:
            env.close()
        except Exception:
            pass
        if not silent:
            log_end(success=success, steps=steps_taken, rewards=rewards)

    return EpisodeResult(
        task_id=task_id,
        score=round(score, 4),
        cumulative_reward=round(sum(rewards), 4),
        steps=steps_taken,
        resolved=resolved,
        failed=failed,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run crisis-dispatch baseline inference")
    # Default to openai when an API key is present (spec requirement),
    # fall back to heuristic for CI/local runs without credentials.
    default_mode = "openai" if (HF_TOKEN or os.getenv("API_KEY")) else "heuristic"
    parser.add_argument(
        "--mode",
        choices=["heuristic", "random", "openai"],
        default=default_mode,
        help="Policy mode for dispatch decisions (default: openai if HF_TOKEN set, else heuristic)",
    )
    parser.add_argument(
        "--task",
        choices=["easy", "medium", "hard", "all"],
        default="all",
        help="Task to evaluate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Seed used for random policy mode",
    )
    parser.add_argument(
        "--check-determinism",
        action="store_true",
        help="Run heuristic twice per task and verify scores are identical (CI gate)",
    )
    parser.add_argument(
        "--enable-heuristic-fallback",
        action="store_true",
        help="Silently revert to the heuristic if the LLM fails to output valid JSON (disabled by default)",
    )
    parser.add_argument(
        "--emit-summary",
        action="store_true",
        help="Emit non-structured CSV and determinism diagnostics (off by default for strict evaluator logs)",
    )
    parser.add_argument(
        "--verbose-llm",
        action="store_true",
        help="Print raw LLM debug/fallback output (off by default for strict evaluator logs)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = [args.task] if args.task != "all" else ["easy", "medium", "hard"]

    resolved_model = MODEL_NAME

    results: List[EpisodeResult] = []

    for task_id in tasks:
        if args.mode == "heuristic":
            policy: Callable[[Observation], Action] = heuristic_policy
        elif args.mode == "random":
            rng = random.Random(args.seed)
            policy = lambda state, _rng=rng: random_policy(state, _rng)
        elif args.mode == "openai":
            if not API_KEY:
                raise RuntimeError(
                    "Missing API key. Set HF_TOKEN or API_KEY environment variable."
                )
            client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
            policy = (
                lambda state,
                _client=client,
                _model=resolved_model,
                _fallback=args.enable_heuristic_fallback,
                _verbose=args.verbose_llm: llm_policy(
                    state,
                    _client,
                    _model,
                    _fallback,
                    _verbose,
                )
            )
        else:
            raise ValueError(f"Unknown mode: {args.mode}")

        result = run_episode(task_id=task_id, policy=policy, model_name=resolved_model)
        results.append(result)

    if args.emit_summary:
        # Optional CSV summary for local analysis.
        print("task_id,score,cumulative_reward,steps,resolved,failed", flush=True)
        for r in results:
            print(
                f"{r.task_id},{r.score:.4f},{r.cumulative_reward:.4f},{r.steps},{r.resolved},{r.failed}",
                flush=True,
            )

    # Determinism check: run heuristic a second time silently and compare
    if getattr(args, "check_determinism", False):
        if args.mode != "heuristic":
            if args.emit_summary:
                print("[WARN] --check-determinism only applies to --mode heuristic", flush=True)
        else:
            if args.emit_summary:
                print(f"\nDeterminism check (heuristic, two runs):", flush=True)
            all_pass = True
            for r in results:
                r2 = run_episode(
                    task_id=r.task_id,
                    policy=heuristic_policy,
                    model_name=resolved_model,
                    silent=True,
                )
                match_score = abs(r.score - r2.score) < 1e-6
                match_reward = abs(r.cumulative_reward - r2.cumulative_reward) < 1e-4
                status = "PASS" if (match_score and match_reward) else "FAIL"
                if status == "FAIL":
                    all_pass = False
                if args.emit_summary:
                    print(
                        f"{r.task_id}: {status} "
                        f"(score {r.score:.4f}/{r2.score:.4f}, "
                        f"reward {r.cumulative_reward:.4f}/{r2.cumulative_reward:.4f})",
                        flush=True,
                    )
            if not all_pass:
                raise SystemExit(1)


if __name__ == "__main__":
    main()