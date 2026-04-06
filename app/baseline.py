import random
from typing import List, Callable, Tuple

from app.models import Action, Observation, UnitState as Unit, IncidentState as Incident, Severity


def travel_distance(x1: int, y1: int, x2: int, y2: int) -> int:
    return abs(x1 - x2) + abs(y1 - y2)


def active_incidents(state: Observation) -> List[Incident]:
    return [i for i in state.incidents if not i.resolved and not i.failed]


def available_units(state: Observation) -> List[Unit]:
    return [u for u in state.units if u.status.value == "available"]


def heuristic_policy(state: Observation) -> Action:
    incidents = active_incidents(state)
    units = available_units(state)

    if not incidents or not units:
        return Action()

    incident_lookup = {inc.id: inc for inc in state.incidents}

    def min_eta_for_type(needed_type: str, incident: Incident) -> int:
        best = float("inf")
        for unit in state.units:
            if unit.unit_type != needed_type:
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
