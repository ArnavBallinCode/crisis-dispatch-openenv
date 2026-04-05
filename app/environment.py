from __future__ import annotations

from typing import Dict, Optional, Tuple

from app.models import (
    Action,
    EnvironmentMetrics,
    Observation,
    GradeResult,
    IncidentState,
    Severity,
    StepResult,
    UnitState,
    UnitStatus,
)
from app.tasks import (
    escalate_severity,
    failure_penalty,
    get_task,
    grade_episode,
    resolution_reward,
    wait_penalty,
)


class CrisisDispatchEnvironment:
    """Deterministic crisis dispatch simulator with dense rewards and typed state."""

    def __init__(self, default_task_id: str = "easy") -> None:
        self.default_task_id = default_task_id
        self._state: Optional[Observation] = None
        self.reset(task_id=default_task_id)

    def reset(self, task_id: Optional[str] = None) -> Observation:
        selected_task_id = task_id or self.default_task_id
        task = get_task(selected_task_id)

        units = [
            UnitState(
                id=unit.id,
                unit_type=unit.unit_type,
                position=unit.position.model_copy(deep=True),
                status=UnitStatus.AVAILABLE,
                target_incident_id=None,
                travel_remaining=0,
            )
            for unit in task.units
        ]

        incidents = [
            IncidentState(
                id=incident.id,
                incident_type=incident.incident_type,
                severity=incident.severity,
                initial_severity=incident.severity,
                position=incident.position.model_copy(deep=True),
                max_wait=incident.max_wait,
                escalation_interval=incident.escalation_interval,
                required_units=list(incident.required_units),
                elapsed=0,
                resolved=False,
                failed=False,
                responding_units=[],
                first_response_step=None,
                resolved_step=None,
            )
            for incident in task.incidents
        ]

        self._state = Observation(
            task_id=task.id,
            task_name=task.name,
            difficulty=task.difficulty,
            city_width=task.city_width,
            city_height=task.city_height,
            step_count=0,
            max_steps=task.max_steps,
            cumulative_reward=0.0,
            done=False,
            units=units,
            incidents=incidents,
            metrics=EnvironmentMetrics(),
        )
        return self.state()

    def state(self) -> Observation:
        if self._state is None:
            raise RuntimeError("Environment has not been initialized")
        return self._state.model_copy(deep=True)

    def step(self, action: Action) -> StepResult:
        if self._state is None:
            raise RuntimeError("Environment has not been initialized")

        if self._state.done:
            final_score = self.grade().score
            return StepResult(
                observation=self.state(),
                reward=0.0,
                done=True,
                score=final_score,
                message="Episode already complete. Call reset() to start a new run.",
            )

        self._state.step_count += 1
        reward = 0.0
        messages = []

        dispatch_reward, dispatch_message = self._apply_dispatch(action)
        reward += dispatch_reward
        if dispatch_message:
            messages.append(dispatch_message)

        reward += self._advance_simulation(messages)

        timed_out = self._state.step_count >= self._state.max_steps
        all_closed = self._all_incidents_closed()
        if timed_out and not all_closed:
            reward += self._fail_remaining_incidents(messages)
            all_closed = True

        self._state.done = all_closed or timed_out
        self._state.cumulative_reward = round(self._state.cumulative_reward + reward, 4)

        score = self.grade().score if self._state.done else None
        return StepResult(
            observation=self.state(),
            reward=round(reward, 4),
            done=self._state.done,
            score=score,
            message=" | ".join(messages),
        )

    def grade(self) -> GradeResult:
        if self._state is None:
            raise RuntimeError("Environment has not been initialized")
        return grade_episode(self._state)

    def _apply_dispatch(self, action: Action) -> Tuple[float, str]:
        if self._state is None:
            raise RuntimeError("Environment has not been initialized")

        if action.unit_id is None and action.incident_id is None:
            return -0.02, "No dispatch this step"

        unit_lookup: Dict[str, UnitState] = {unit.id: unit for unit in self._state.units}
        incident_lookup: Dict[str, IncidentState] = {
            incident.id: incident for incident in self._state.incidents
        }

        unit = unit_lookup.get(action.unit_id or "")
        incident = incident_lookup.get(action.incident_id or "")

        if unit is None or incident is None:
            return -0.5, "Invalid dispatch target"
        if unit.status != UnitStatus.AVAILABLE:
            return -0.35, f"Unit {unit.id} is not available"
        if incident.resolved or incident.failed:
            return -0.3, f"Incident {incident.id} already closed"

        distance = self._manhattan_distance(unit.position.x, unit.position.y, incident.position.x, incident.position.y)
        unit.status = UnitStatus.EN_ROUTE
        unit.target_incident_id = incident.id
        unit.travel_remaining = max(1, distance)

        self._state.metrics.total_dispatches += 1

        if unit.unit_type in incident.required_units:
            self._state.metrics.correct_dispatches += 1
            return 0.35, f"Dispatched {unit.id} to {incident.id} ({distance} step travel)"

        self._state.metrics.wrong_dispatches += 1
        incident.elapsed += 1
        return -0.7, f"Incorrect unit {unit.id} dispatched to {incident.id}"

    def _advance_simulation(self, messages: list[str]) -> float:
        if self._state is None:
            raise RuntimeError("Environment has not been initialized")

        reward = 0.0

        for unit in self._state.units:
            if unit.status == UnitStatus.EN_ROUTE:
                unit.travel_remaining = max(0, unit.travel_remaining - 1)
                if unit.travel_remaining == 0:
                    arrival_reward, arrival_message = self._resolve_arrival(unit)
                    reward += arrival_reward
                    if arrival_message:
                        messages.append(arrival_message)

        for incident in self._state.incidents:
            if incident.resolved or incident.failed:
                continue

            incident.elapsed += 1
            reward -= wait_penalty(incident.severity)

            if incident.elapsed % incident.escalation_interval == 0:
                new_severity = escalate_severity(incident.severity)
                if new_severity != incident.severity:
                    incident.severity = new_severity
                    reward -= 0.2
                    messages.append(f"{incident.id} escalated to {incident.severity.value}")

            if incident.elapsed >= incident.max_wait and not incident.resolved:
                incident.failed = True
                self._state.metrics.incidents_failed += 1
                if incident.severity == Severity.CRITICAL:
                    self._state.metrics.unresolved_critical += 1
                reward -= failure_penalty(incident.severity)
                messages.append(f"{incident.id} became unrecoverable")

        return reward

    def _resolve_arrival(self, unit: UnitState) -> Tuple[float, str]:
        if self._state is None:
            raise RuntimeError("Environment has not been initialized")

        if not unit.target_incident_id:
            unit.status = UnitStatus.AVAILABLE
            return -0.05, f"{unit.id} has no valid destination"

        incident_lookup: Dict[str, IncidentState] = {
            incident.id: incident for incident in self._state.incidents
        }
        incident = incident_lookup.get(unit.target_incident_id)

        if incident is None:
            unit.status = UnitStatus.AVAILABLE
            unit.target_incident_id = None
            return -0.05, f"{unit.id} destination missing"

        unit.position = incident.position.model_copy(deep=True)
        unit.status = UnitStatus.AVAILABLE
        unit.target_incident_id = None

        if incident.resolved or incident.failed:
            return -0.05, f"{unit.id} arrived after {incident.id} was already closed"

        if incident.first_response_step is None:
            incident.first_response_step = self._state.step_count
            self._state.metrics.response_times[incident.id] = self._state.step_count

        if unit.unit_type not in incident.required_units:
            return -0.2, f"{unit.id} reached {incident.id} but cannot resolve it"

        if unit.unit_type not in incident.responding_units:
            incident.responding_units.append(unit.unit_type)

        reward = 0.4
        if self._incident_requirements_met(incident):
            incident.resolved = True
            incident.resolved_step = self._state.step_count
            self._state.metrics.incidents_resolved += 1

            timeliness_factor = max(0.2, 1.0 - incident.elapsed / incident.max_wait)
            reward += resolution_reward(incident.initial_severity) * timeliness_factor
            return reward, f"{incident.id} resolved by {unit.id}"

        return reward, f"{unit.id} partially stabilized {incident.id}"

    def _fail_remaining_incidents(self, messages: list[str]) -> float:
        if self._state is None:
            raise RuntimeError("Environment has not been initialized")

        reward = 0.0
        for incident in self._state.incidents:
            if incident.resolved or incident.failed:
                continue

            incident.failed = True
            self._state.metrics.incidents_failed += 1
            if incident.severity == Severity.CRITICAL:
                self._state.metrics.unresolved_critical += 1

            penalty = 0.75 * failure_penalty(incident.severity)
            reward -= penalty
            messages.append(f"Timeout closed unresolved incident {incident.id}")

        return reward

    def _all_incidents_closed(self) -> bool:
        if self._state is None:
            raise RuntimeError("Environment has not been initialized")
        return all(incident.resolved or incident.failed for incident in self._state.incidents)

    @staticmethod
    def _incident_requirements_met(incident: IncidentState) -> bool:
        return set(incident.required_units).issubset(set(incident.responding_units))

    @staticmethod
    def _manhattan_distance(x1: int, y1: int, x2: int, y2: int) -> int:
        return abs(x1 - x2) + abs(y1 - y2)

    def close(self) -> None:
        """Clean up (no-op for this deterministic environment)."""
        self._state = None
