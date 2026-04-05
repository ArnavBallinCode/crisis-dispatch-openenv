from __future__ import annotations

from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class UnitType(str, Enum):
    AMBULANCE = "ambulance"
    FIRE_TRUCK = "fire_truck"
    POLICE = "police"


class IncidentType(str, Enum):
    MEDICAL = "medical"
    FIRE = "fire"
    TRAFFIC = "traffic"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    CRITICAL = "critical"


class UnitStatus(str, Enum):
    AVAILABLE = "available"
    EN_ROUTE = "en_route"


class Position(BaseModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)


class UnitTemplate(BaseModel):
    id: str
    unit_type: UnitType
    position: Position


class IncidentTemplate(BaseModel):
    id: str
    incident_type: IncidentType
    severity: Severity
    position: Position
    max_wait: int = Field(ge=1)
    escalation_interval: int = Field(default=3, ge=1)
    required_units: List[UnitType] = Field(default_factory=list)


class TaskDefinition(BaseModel):
    id: str
    name: str
    description: str
    difficulty: Literal["easy", "medium", "hard"]
    city_width: int = Field(ge=3)
    city_height: int = Field(ge=3)
    max_steps: int = Field(ge=1)
    units: List[UnitTemplate] = Field(default_factory=list)
    incidents: List[IncidentTemplate] = Field(default_factory=list)


class UnitState(BaseModel):
    id: str
    unit_type: UnitType
    position: Position
    status: UnitStatus = UnitStatus.AVAILABLE
    target_incident_id: Optional[str] = None
    travel_remaining: int = Field(default=0, ge=0)


class IncidentState(BaseModel):
    id: str
    incident_type: IncidentType
    severity: Severity
    initial_severity: Severity
    position: Position
    max_wait: int = Field(ge=1)
    escalation_interval: int = Field(ge=1)
    required_units: List[UnitType] = Field(default_factory=list)
    elapsed: int = Field(default=0, ge=0)
    resolved: bool = False
    failed: bool = False
    responding_units: List[UnitType] = Field(default_factory=list)
    first_response_step: Optional[int] = None
    resolved_step: Optional[int] = None


class EnvironmentMetrics(BaseModel):
    total_dispatches: int = 0
    correct_dispatches: int = 0
    wrong_dispatches: int = 0
    incidents_resolved: int = 0
    incidents_failed: int = 0
    unresolved_critical: int = 0
    response_times: Dict[str, int] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# OpenEnv-mandated model names: Observation, Action, Reward
# ---------------------------------------------------------------------------


class Observation(BaseModel):
    """The typed Observation returned from reset()/step()."""
    task_id: str
    task_name: str
    difficulty: Literal["easy", "medium", "hard"]
    city_width: int
    city_height: int
    step_count: int
    max_steps: int
    cumulative_reward: float
    done: bool
    units: List[UnitState]
    incidents: List[IncidentState]
    metrics: EnvironmentMetrics


# Backward compatibility alias used internally
EnvironmentState = Observation


class Action(BaseModel):
    """The typed Action accepted by step()."""
    unit_id: Optional[str] = None
    incident_id: Optional[str] = None

    @model_validator(mode="after")
    def validate_pair(self) -> "Action":
        if (self.unit_id is None) != (self.incident_id is None):
            raise ValueError("unit_id and incident_id must be both provided or both omitted for wait")
        return self


# Backward compatibility alias
DispatchAction = Action


class Reward(BaseModel):
    """Typed Reward model as required by OpenEnv spec."""
    value: float = 0.0
    message: str = ""


class GradeResult(BaseModel):
    score: float
    weighted_success: float
    weighted_timeliness: float
    dispatch_accuracy: float
    critical_failure_penalty: float


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    score: Optional[float] = None
    message: str = ""
    info: Dict = Field(default_factory=dict)


class TaskSummary(BaseModel):
    id: str
    name: str
    description: str
    difficulty: Literal["easy", "medium", "hard"]
    max_steps: int


class ResetRequest(BaseModel):
    task_id: Optional[str] = None


class ScoreResponse(BaseModel):
    grade: GradeResult
    task_id: str
    done: bool
