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


if load_dotenv is not None:
    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)


# ---------------------------------------------------------------------------
# Environment variable resolution (strict hackathon rules)
# ---------------------------------------------------------------------------

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
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


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
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


def heuristic_policy(state: Observation) -> Action:
    incidents = active_incidents(state)
    units = available_units(state)

    if not incidents or not units:
        return Action()

    incident_lookup = {incident.id: incident for incident in state.incidents}

    def min_eta_for_type(unit_type, incident) -> int:
        best = 10**9
        for unit in state.units:
            if unit.unit_type != unit_type:
                continue

            if unit.status.value == "available":
                eta = travel_distance(
                    unit.position.x,
                    unit.position.y,
                    incident.position.x,
                    incident.position.y,
                )
            else:
                target = incident_lookup.get(unit.target_incident_id or "")
                if target is None:
                    continue
                eta = unit.travel_remaining + 1 + travel_distance(
                    target.position.x,
                    target.position.y,
                    incident.position.x,
                    incident.position.y,
                )

            if eta < best:
                best = eta
        return best

    incident_info = {}
    for incident in incidents:
        required = set(incident.required_units)
        responding = set(incident.responding_units)
        planned = set(responding)
        for unit in state.units:
            if unit.target_incident_id == incident.id:
                planned.add(unit.unit_type)

        missing = required - planned
        slack = incident.max_wait - incident.elapsed
        doomed = any(min_eta_for_type(needed_type, incident) > slack for needed_type in missing)

        incident_info[incident.id] = {
            "incident": incident,
            "missing": missing,
            "slack": slack,
            "doomed": doomed,
        }

    forced_candidates = []
    for incident in incidents:
        info = incident_info[incident.id]
        missing = info["missing"]
        slack = info["slack"]

        if incident.initial_severity != Severity.CRITICAL:
            continue
        if not missing:
            continue

        for needed_type in missing:
            feasible_units = [
                unit
                for unit in units
                if unit.unit_type == needed_type
                and travel_distance(
                    unit.position.x,
                    unit.position.y,
                    incident.position.x,
                    incident.position.y,
                )
                <= slack
            ]

            if len(feasible_units) == 1:
                chosen = feasible_units[0]
                distance = travel_distance(
                    chosen.position.x,
                    chosen.position.y,
                    incident.position.x,
                    incident.position.y,
                )
                margin = slack - distance
                forced_candidates.append((margin, distance, chosen.id, incident.id))

    if forced_candidates:
        _, _, unit_id, incident_id = min(forced_candidates)
        return Action(unit_id=unit_id, incident_id=incident_id)

    severity_weight = {
        Severity.LOW: 1.0,
        Severity.MEDIUM: 2.5,
        Severity.CRITICAL: 5.0,
    }

    def reservation_penalty(unit, chosen_incident_id: str) -> float:
        penalty = 0.0
        for info in incident_info.values():
            incident = info["incident"]
            if incident.id == chosen_incident_id or info["doomed"]:
                continue
            if unit.unit_type not in info["missing"]:
                continue

            unit_distance = travel_distance(
                unit.position.x,
                unit.position.y,
                incident.position.x,
                incident.position.y,
            )
            if unit_distance > info["slack"]:
                continue

            feasible_units = [
                candidate
                for candidate in units
                if candidate.unit_type == unit.unit_type
                and travel_distance(
                    candidate.position.x,
                    candidate.position.y,
                    incident.position.x,
                    incident.position.y,
                )
                <= info["slack"]
            ]
            if len(feasible_units) == 1 and feasible_units[0].id == unit.id:
                if incident.initial_severity == Severity.CRITICAL:
                    penalty += 8.0
                elif incident.initial_severity == Severity.MEDIUM:
                    penalty += 3.0
                else:
                    penalty += 1.2

        return penalty

    best_score = float("-inf")
    best_action = Action()

    for info in incident_info.values():
        incident = info["incident"]
        missing = info["missing"]
        slack = info["slack"]

        if not missing or info["doomed"]:
            continue

        urgency = incident.elapsed / max(1, incident.max_wait)
        deadline_pressure = 4.0 / max(1, slack)
        completion_bonus = 2.0 if len(missing) == 1 else 0.0
        first_response_bonus = 1.5 if incident.first_response_step is None else 0.0

        incident_priority = (
            5.0 * severity_weight[incident.initial_severity]
            + 2.5 * urgency
            + deadline_pressure
            + completion_bonus
            + first_response_bonus
        )

        for unit in units:
            if unit.unit_type not in missing:
                continue

            distance = travel_distance(
                unit.position.x,
                unit.position.y,
                incident.position.x,
                incident.position.y,
            )
            if distance > slack:
                continue

            score = (
                incident_priority
                - 1.3 * distance
                - reservation_penalty(unit, incident.id)
            )

            if score > best_score:
                best_score = score
                best_action = Action(unit_id=unit.id, incident_id=incident.id)

    return best_action


def random_policy(state: Observation, rng: random.Random) -> Action:
    incidents = active_incidents(state)
    units = available_units(state)

    if not incidents or not units:
        return Action()

    if rng.random() < 0.15:
        return Action()

    unit = rng.choice(units)
    incident = rng.choice(incidents)
    return Action(unit_id=unit.id, incident_id=incident.id)


def llm_policy(state: Observation, client: OpenAI, model: str) -> Action:
    incidents = active_incidents(state)
    units = available_units(state)
    if not incidents or not units:
        return Action()

    prompt_payload = {
        "step": state.step_count,
        "available_units": [
            {
                "id": unit.id,
                "unit_type": unit.unit_type.value,
                "x": unit.position.x,
                "y": unit.position.y,
            }
            for unit in units
        ],
        "active_incidents": [
            {
                "id": incident.id,
                "incident_type": incident.incident_type.value,
                "severity": incident.severity.value,
                "required_units": [u.value for u in incident.required_units],
                "x": incident.position.x,
                "y": incident.position.y,
                "elapsed": incident.elapsed,
                "max_wait": incident.max_wait,
            }
            for incident in incidents
        ],
    }

    system_prompt = (
        "You dispatch emergency units. Return only JSON with keys unit_id and "
        "incident_id. Use null for both keys to wait."
    )
    user_prompt = json.dumps(prompt_payload)

    try:
        completion = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = (
            completion.choices[0].message.content.strip()
            if completion.choices and completion.choices[0].message.content
            else ""
        )
    except Exception:
        return heuristic_policy(state)

    if not text:
        return heuristic_policy(state)
    try:
        payload = json.loads(text)
        return Action(
            unit_id=payload.get("unit_id"),
            incident_id=payload.get("incident_id"),
        )
    except Exception:
        return heuristic_policy(state)


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

    try:
        while not obs.done:
            action = policy(obs)

            # Build action string for logging
            if action.unit_id and action.incident_id:
                action_str = f"dispatch({action.unit_id},{action.incident_id})"
            else:
                action_str = "wait()"

            result = env.step(action)
            reward = result.reward
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken += 1
            obs = result.observation

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
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

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
    default_mode = "openai" if (os.getenv("HF_TOKEN") or os.getenv("API_KEY")) else "heuristic"
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
            policy = lambda state, _client=client, _model=resolved_model: llm_policy(
                state, _client, _model
            )
        else:
            raise ValueError(f"Unknown mode: {args.mode}")

        result = run_episode(task_id=task_id, policy=policy, model_name=resolved_model)
        results.append(result)

    # CSV summary (matches original --check-determinism output format)
    print("task_id,score,cumulative_reward,steps,resolved,failed", flush=True)
    for r in results:
        print(f"{r.task_id},{r.score:.4f},{r.cumulative_reward:.4f},{r.steps},{r.resolved},{r.failed}", flush=True)

    # Determinism check: run heuristic a second time silently and compare
    if getattr(args, "check_determinism", False):
        if args.mode != "heuristic":
            print("[WARN] --check-determinism only applies to --mode heuristic", flush=True)
        else:
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
