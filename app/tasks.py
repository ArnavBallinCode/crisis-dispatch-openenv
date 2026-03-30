from __future__ import annotations

from typing import Dict, List

from app.models import (
    EnvironmentState,
    GradeResult,
    IncidentType,
    Position,
    Severity,
    TaskDefinition,
    TaskSummary,
    UnitType,
)


SEVERITY_WEIGHTS = {
    Severity.LOW: 1.0,
    Severity.MEDIUM: 2.0,
    Severity.CRITICAL: 3.5,
}


WAIT_PENALTIES = {
    Severity.LOW: 0.02,
    Severity.MEDIUM: 0.05,
    Severity.CRITICAL: 0.09,
}


RESOLUTION_REWARDS = {
    Severity.LOW: 1.0,
    Severity.MEDIUM: 1.8,
    Severity.CRITICAL: 3.0,
}


FAILURE_PENALTIES = {
    Severity.LOW: 1.1,
    Severity.MEDIUM: 2.0,
    Severity.CRITICAL: 3.6,
}


TASKS: Dict[str, TaskDefinition] = {
    "easy": TaskDefinition(
        id="easy",
        name="Single Medical Priority",
        description=(
            "One high-severity medical incident. Correctly dispatching the nearest ambulance "
            "is enough to succeed."
        ),
        difficulty="easy",
        city_width=6,
        city_height=6,
        max_steps=18,
        units=[
            {"id": "A1", "unit_type": UnitType.AMBULANCE, "position": Position(x=0, y=0)},
            {"id": "F1", "unit_type": UnitType.FIRE_TRUCK, "position": Position(x=5, y=5)},
            {"id": "P1", "unit_type": UnitType.POLICE, "position": Position(x=3, y=4)},
        ],
        incidents=[
            {
                "id": "E-MED-1",
                "incident_type": IncidentType.MEDICAL,
                "severity": Severity.CRITICAL,
                "position": Position(x=1, y=2),
                "max_wait": 8,
                "escalation_interval": 2,
                "required_units": [UnitType.AMBULANCE],
            }
        ],
    ),
    "medium": TaskDefinition(
        id="medium",
        name="Limited Fleet Prioritization",
        description=(
            "Multiple incidents appear with constrained coverage. Agent must prioritize critical "
            "calls while avoiding mismatched dispatches."
        ),
        difficulty="medium",
        city_width=8,
        city_height=8,
        max_steps=30,
        units=[
            {"id": "A1", "unit_type": UnitType.AMBULANCE, "position": Position(x=0, y=0)},
            {"id": "A2", "unit_type": UnitType.AMBULANCE, "position": Position(x=7, y=0)},
            {"id": "F1", "unit_type": UnitType.FIRE_TRUCK, "position": Position(x=7, y=7)},
            {"id": "P1", "unit_type": UnitType.POLICE, "position": Position(x=2, y=6)},
        ],
        incidents=[
            {
                "id": "M-MED-1",
                "incident_type": IncidentType.MEDICAL,
                "severity": Severity.CRITICAL,
                "position": Position(x=1, y=2),
                "max_wait": 7,
                "escalation_interval": 2,
                "required_units": [UnitType.AMBULANCE],
            },
            {
                "id": "M-FIRE-1",
                "incident_type": IncidentType.FIRE,
                "severity": Severity.MEDIUM,
                "position": Position(x=6, y=6),
                "max_wait": 10,
                "escalation_interval": 3,
                "required_units": [UnitType.FIRE_TRUCK],
            },
            {
                "id": "M-TRAFFIC-1",
                "incident_type": IncidentType.TRAFFIC,
                "severity": Severity.MEDIUM,
                "position": Position(x=3, y=5),
                "max_wait": 9,
                "escalation_interval": 3,
                "required_units": [UnitType.POLICE],
            },
            {
                "id": "M-MED-2",
                "incident_type": IncidentType.MEDICAL,
                "severity": Severity.LOW,
                "position": Position(x=7, y=1),
                "max_wait": 12,
                "escalation_interval": 4,
                "required_units": [UnitType.AMBULANCE],
            },
        ],
    ),
    "hard": TaskDefinition(
        id="hard",
        name="Multi-Agency Incident Cascade",
        description=(
            "Critical and medium incidents compete for shared units. Some incidents require "
            "multi-agency response, forcing tradeoffs between distance and severity."
        ),
        difficulty="hard",
        city_width=10,
        city_height=10,
        max_steps=40,
        units=[
            {"id": "A1", "unit_type": UnitType.AMBULANCE, "position": Position(x=0, y=0)},
            {"id": "A2", "unit_type": UnitType.AMBULANCE, "position": Position(x=9, y=1)},
            {"id": "F1", "unit_type": UnitType.FIRE_TRUCK, "position": Position(x=9, y=9)},
            {"id": "F2", "unit_type": UnitType.FIRE_TRUCK, "position": Position(x=4, y=8)},
            {"id": "P1", "unit_type": UnitType.POLICE, "position": Position(x=0, y=9)},
            {"id": "P2", "unit_type": UnitType.POLICE, "position": Position(x=5, y=4)},
        ],
        incidents=[
            {
                "id": "H-FIRE-1",
                "incident_type": IncidentType.FIRE,
                "severity": Severity.CRITICAL,
                "position": Position(x=8, y=8),
                "max_wait": 7,
                "escalation_interval": 2,
                "required_units": [UnitType.FIRE_TRUCK, UnitType.POLICE],
            },
            {
                "id": "H-MED-1",
                "incident_type": IncidentType.MEDICAL,
                "severity": Severity.CRITICAL,
                "position": Position(x=2, y=2),
                "max_wait": 6,
                "escalation_interval": 2,
                "required_units": [UnitType.AMBULANCE],
            },
            {
                "id": "H-TRAFFIC-1",
                "incident_type": IncidentType.TRAFFIC,
                "severity": Severity.MEDIUM,
                "position": Position(x=5, y=2),
                "max_wait": 8,
                "escalation_interval": 3,
                "required_units": [UnitType.POLICE, UnitType.AMBULANCE],
            },
            {
                "id": "H-FIRE-2",
                "incident_type": IncidentType.FIRE,
                "severity": Severity.MEDIUM,
                "position": Position(x=3, y=8),
                "max_wait": 11,
                "escalation_interval": 3,
                "required_units": [UnitType.FIRE_TRUCK],
            },
            {
                "id": "H-MED-2",
                "incident_type": IncidentType.MEDICAL,
                "severity": Severity.LOW,
                "position": Position(x=9, y=0),
                "max_wait": 12,
                "escalation_interval": 4,
                "required_units": [UnitType.AMBULANCE],
            },
        ],
    ),
}


def list_task_summaries() -> List[TaskSummary]:
    return [
        TaskSummary(
            id=task.id,
            name=task.name,
            description=task.description,
            difficulty=task.difficulty,
            max_steps=task.max_steps,
        )
        for task in TASKS.values()
    ]


def get_task(task_id: str) -> TaskDefinition:
    if task_id not in TASKS:
        raise KeyError(f"Unknown task_id: {task_id}")
    return TASKS[task_id].model_copy(deep=True)


def severity_weight(severity: Severity) -> float:
    return SEVERITY_WEIGHTS[severity]


def wait_penalty(severity: Severity) -> float:
    return WAIT_PENALTIES[severity]


def resolution_reward(severity: Severity) -> float:
    return RESOLUTION_REWARDS[severity]


def failure_penalty(severity: Severity) -> float:
    return FAILURE_PENALTIES[severity]


def escalate_severity(severity: Severity) -> Severity:
    if severity == Severity.LOW:
        return Severity.MEDIUM
    if severity == Severity.MEDIUM:
        return Severity.CRITICAL
    return Severity.CRITICAL


def grade_episode(state: EnvironmentState) -> GradeResult:
    incidents = state.incidents
    if not incidents:
        return GradeResult(
            score=0.0,
            weighted_success=0.0,
            weighted_timeliness=0.0,
            dispatch_accuracy=0.0,
            critical_failure_penalty=0.0,
        )

    total_weight = sum(severity_weight(incident.initial_severity) for incident in incidents)
    weighted_success = (
        sum(
            severity_weight(incident.initial_severity)
            for incident in incidents
            if incident.resolved and not incident.failed
        )
        / total_weight
    )

    weighted_timeliness_raw = 0.0
    for incident in incidents:
        if incident.resolved and incident.first_response_step is not None:
            timeliness = max(0.0, 1.0 - (incident.first_response_step - 1) / incident.max_wait)
            weighted_timeliness_raw += severity_weight(incident.initial_severity) * timeliness
    weighted_timeliness = weighted_timeliness_raw / total_weight

    dispatch_accuracy = (
        state.metrics.correct_dispatches / max(1, state.metrics.total_dispatches)
    )

    critical_incidents = [i for i in incidents if i.initial_severity == Severity.CRITICAL]
    critical_failed = [i for i in critical_incidents if not i.resolved]
    critical_failure_penalty = len(critical_failed) / max(1, len(critical_incidents))

    base_score = (
        0.58 * weighted_success
        + 0.27 * weighted_timeliness
        + 0.15 * dispatch_accuracy
    )
    final_score = max(0.0, min(1.0, base_score - 0.25 * critical_failure_penalty))

    return GradeResult(
        score=round(final_score, 4),
        weighted_success=round(weighted_success, 4),
        weighted_timeliness=round(weighted_timeliness, 4),
        dispatch_accuracy=round(dispatch_accuracy, 4),
        critical_failure_penalty=round(critical_failure_penalty, 4),
    )
