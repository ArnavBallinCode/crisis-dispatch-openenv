from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from openai import OpenAI

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

from app.environment import CrisisDispatchEnvironment
from app.models import DispatchAction, EnvironmentState, Severity


if load_dotenv is not None:
    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)


SEVERITY_PRIORITY = {
    Severity.LOW: 1.0,
    Severity.MEDIUM: 2.0,
    Severity.CRITICAL: 3.5,
}


def _env_first(*keys: str) -> Optional[str]:
    for key in keys:
        value = os.getenv(key)
        if value is None:
            continue
        cleaned = value.strip().strip('"').strip("'")
        if cleaned:
            return cleaned
    return None


def resolve_llm_config(model_arg: Optional[str]) -> Tuple[str, str, Optional[str]]:
    """Resolve API key/model/base URL for OpenAI-compatible providers.

    Supported env var aliases:
    - API key: OPENAI_API_KEY, API_KEY, GROQ_API_KEY
    - Base URL: OPENAI_BASE_URL, API_BASE_URL, GROQ_BASE_URL
    - Model: OPENAI_MODEL, MODEL_NAME, GROQ_MODEL
    """
    api_key = _env_first("OPENAI_API_KEY", "API_KEY", "GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing API key for --mode openai. Set OPENAI_API_KEY or API_KEY (or GROQ_API_KEY)."
        )

    resolved_model = (
        model_arg.strip()
        if model_arg and model_arg.strip()
        else _env_first("OPENAI_MODEL", "MODEL_NAME", "GROQ_MODEL")
        or "gpt-4.1-mini"
    )
    base_url = _env_first("OPENAI_BASE_URL", "API_BASE_URL", "GROQ_BASE_URL")
    return api_key, resolved_model, base_url


@dataclass
class EpisodeResult:
    task_id: str
    score: float
    cumulative_reward: float
    steps: int
    resolved: int
    failed: int


def active_incidents(state: EnvironmentState):
    return [incident for incident in state.incidents if not incident.resolved and not incident.failed]


def available_units(state: EnvironmentState):
    return [unit for unit in state.units if unit.status.value == "available"]


def travel_distance(unit_x: int, unit_y: int, incident_x: int, incident_y: int) -> int:
    return abs(unit_x - incident_x) + abs(unit_y - incident_y)


# def heuristic_policy(state: EnvironmentState) -> DispatchAction:
#     incidents = active_incidents(state)
#     units = available_units(state)
#     if not incidents or not units:
#         return DispatchAction()

#     best_score = float("-inf")
#     best_action = DispatchAction()

#     for incident in incidents:
#         required = set(incident.required_units)
#         responding = set(incident.responding_units)
#         missing = required - responding
#         urgency = incident.elapsed / max(1, incident.max_wait)
#         severity_score = SEVERITY_PRIORITY[incident.severity]
#         first_response_bonus = 1.0 if incident.first_response_step is None else 0.0

#         for unit in units:
#             distance = travel_distance(
#                 unit.position.x,
#                 unit.position.y,
#                 incident.position.x,
#                 incident.position.y,
#             )

#             if unit.unit_type in missing:
#                 suitability = 1.6
#             elif unit.unit_type in required:
#                 suitability = -0.8
#             else:
#                 suitability = -2.0

#             slack = incident.max_wait - incident.elapsed - distance
#             if slack <= 0:
#                 deadline_pressure = 1.6
#             else:
#                 deadline_pressure = 1.0 / (1.0 + slack)

#             pair_score = (
#                 3.4 * severity_score
#                 + 2.2 * suitability
#                 + 1.8 * deadline_pressure
#                 + 1.4 * urgency
#                 + 1.8 * first_response_bonus
#                 - 0.25 * distance
#             )

#             if pair_score > best_score:
#                 best_score = pair_score
#                 best_action = DispatchAction(unit_id=unit.id, incident_id=incident.id)

#     return best_action

def heuristic_policy(state: EnvironmentState) -> DispatchAction:
    incidents = active_incidents(state)
    units = available_units(state)

    if not incidents or not units:
        return DispatchAction()

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

    forced_candidates = []
    for incident in incidents:
        if incident.severity != Severity.CRITICAL:
            continue

        required = set(incident.required_units)
        responding = set(incident.responding_units)
        missing = required - responding
        if not missing:
            continue

        time_left = incident.max_wait - incident.elapsed
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
                <= time_left
            ]

            if len(feasible_units) == 1:
                chosen = feasible_units[0]
                distance = travel_distance(
                    chosen.position.x,
                    chosen.position.y,
                    incident.position.x,
                    incident.position.y,
                )
                margin = time_left - distance
                forced_candidates.append(
                    (margin, -1 if len(missing) > 1 else 0, distance, chosen.id, incident.id)
                )

    if forced_candidates:
        _, _, _, unit_id, incident_id = min(forced_candidates)
        return DispatchAction(unit_id=unit_id, incident_id=incident_id)

    best_score = float("-inf")
    best_action = DispatchAction()

    for incident in incidents:
        required = set(incident.required_units)
        responding = set(incident.responding_units)
        planned = set(responding)
        for unit in state.units:
            if unit.target_incident_id == incident.id:
                planned.add(unit.unit_type)

        missing = required - responding
        missing_effective = required - planned

        severity_weight = {
            Severity.LOW: 1.0,
            Severity.MEDIUM: 2.5,
            Severity.CRITICAL: 6.0,
        }[incident.severity]

        urgency = incident.elapsed / max(1, incident.max_wait)

        slack = incident.max_wait - incident.elapsed
        deadline_pressure = 2.5 if slack <= 2 else 1 / (1 + slack)

        multi_agent_bonus = 2.5 if len(missing) > 1 else 0.0

        first_response_bonus = 2.0 if incident.first_response_step is None else 0.0

        incident_priority = (
            4.0 * severity_weight
            + 3.0 * urgency
            + 3.0 * deadline_pressure
            + multi_agent_bonus
            + first_response_bonus
        )

        doomed = False
        for needed_type in missing_effective:
            if min_eta_for_type(needed_type, incident) > slack:
                doomed = True
                break

        for unit in units:
            distance = travel_distance(
                unit.position.x,
                unit.position.y,
                incident.position.x,
                incident.position.y,
            )

            if unit.unit_type in missing:
                suitability = 3.5
            elif unit.unit_type in required:
                suitability = -2.5
            else:
                suitability = -6.0

            time_left = incident.max_wait - incident.elapsed
            feasibility_penalty = -8.0 if distance > time_left else 0.0
            distance_penalty = -1.2 * distance
            overcommit_penalty = -3.0 if len(responding) >= len(required) else 0.0
            critical_completion_bonus = (
                7.0
                if incident.severity == Severity.CRITICAL
                and len(responding) > 0
                and unit.unit_type in missing
                else 0.0
            )
            doomed_penalty = -10.0 if doomed and incident.severity != Severity.CRITICAL else 0.0

            score = (
                incident_priority
                + 2.5 * suitability
                + feasibility_penalty
                + distance_penalty
                + overcommit_penalty
                + critical_completion_bonus
                + doomed_penalty
            )

            if score > best_score:
                best_score = score
                best_action = DispatchAction(unit_id=unit.id, incident_id=incident.id)

    return best_action


def random_policy(state: EnvironmentState, rng: random.Random) -> DispatchAction:
    incidents = active_incidents(state)
    units = available_units(state)

    if not incidents or not units:
        return DispatchAction()

    if rng.random() < 0.15:
        return DispatchAction()

    unit = rng.choice(units)
    incident = rng.choice(incidents)
    return DispatchAction(unit_id=unit.id, incident_id=incident.id)


def llm_policy(state: EnvironmentState, client: OpenAI, model: str) -> DispatchAction:
    incidents = active_incidents(state)
    units = available_units(state)
    if not incidents or not units:
        return DispatchAction()

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

    if hasattr(client, "responses"):
        response = client.responses.create(
            model=model,
            temperature=0,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = (getattr(response, "output_text", None) or "").strip()
    else:
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

    if not text:
        return heuristic_policy(state)
    try:
        payload = json.loads(text)
        return DispatchAction(
            unit_id=payload.get("unit_id"),
            incident_id=payload.get("incident_id"),
        )
    except Exception:
        return heuristic_policy(state)


def run_episode(
    task_id: str,
    policy: Callable[[EnvironmentState], DispatchAction],
) -> EpisodeResult:
    env = CrisisDispatchEnvironment(default_task_id=task_id)
    state = env.reset(task_id=task_id)

    while not state.done:
        action = policy(state)
        result = env.step(action)
        state = result.state

    grade = env.grade()
    return EpisodeResult(
        task_id=task_id,
        score=grade.score,
        cumulative_reward=state.cumulative_reward,
        steps=state.step_count,
        resolved=state.metrics.incidents_resolved,
        failed=state.metrics.incidents_failed,
    )


def run_baseline(
    mode: str,
    tasks: List[str],
    episodes: int,
    seed: int,
    model: Optional[str],
) -> List[EpisodeResult]:
    results: List[EpisodeResult] = []

    for task_id in tasks:
        for episode_idx in range(episodes):
            if mode == "heuristic":
                policy = heuristic_policy
            elif mode == "random":
                rng = random.Random(seed + episode_idx)
                policy = lambda state, _rng=rng: random_policy(state, _rng)
            elif mode == "openai":
                api_key, resolved_model, base_url = resolve_llm_config(model)
                client_kwargs = {"api_key": api_key}
                if base_url:
                    client_kwargs["base_url"] = base_url
                client = OpenAI(**client_kwargs)
                policy = lambda state, _client=client, _model=resolved_model: llm_policy(
                    state, _client, _model
                )
            else:
                raise ValueError(f"Unknown mode: {mode}")

            results.append(run_episode(task_id=task_id, policy=policy))

    return results


def print_results(results: List[EpisodeResult]) -> None:
    print("task_id,score,cumulative_reward,steps,resolved,failed")
    for result in results:
        print(
            f"{result.task_id},{result.score:.4f},{result.cumulative_reward:.4f},"
            f"{result.steps},{result.resolved},{result.failed}"
        )


def run_determinism_check(tasks: List[str]) -> None:
    print("\nDeterminism check (heuristic, two runs):")
    for task_id in tasks:
        first = run_episode(task_id=task_id, policy=heuristic_policy)
        second = run_episode(task_id=task_id, policy=heuristic_policy)
        same = first.score == second.score and first.cumulative_reward == second.cumulative_reward
        status = "PASS" if same else "FAIL"
        print(
            f"{task_id}: {status} (score {first.score:.4f}/{second.score:.4f}, "
            f"reward {first.cumulative_reward:.4f}/{second.cumulative_reward:.4f})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run crisis-dispatch baseline inference")
    parser.add_argument(
        "--mode",
        choices=["heuristic", "random", "openai"],
        default="heuristic",
        help="Policy mode for dispatch decisions",
    )
    parser.add_argument(
        "--task",
        choices=["easy", "medium", "hard", "all"],
        default="all",
        help="Task to evaluate",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Episodes per task",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Seed used for random policy mode",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model used in --mode openai. If omitted, uses OPENAI_MODEL/MODEL_NAME/GROQ_MODEL "
            "or falls back to gpt-4.1-mini."
        ),
    )
    parser.add_argument(
        "--check-determinism",
        action="store_true",
        help="Run deterministic consistency check for heuristic policy",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = [args.task] if args.task != "all" else ["easy", "medium", "hard"]

    results = run_baseline(
        mode=args.mode,
        tasks=tasks,
        episodes=max(1, args.episodes),
        seed=args.seed,
        model=args.model,
    )
    print_results(results)

    if args.check_determinism:
        run_determinism_check(tasks)


if __name__ == "__main__":
    main()
