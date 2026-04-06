"""Typed models for the ChargebackOps OpenEnv environment."""

from __future__ import annotations

from typing import Any, Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field

SystemName = Literal["orders", "payment", "shipping", "support", "refunds", "risk"]
StrategyName = Literal["contest", "accept_chargeback", "issue_refund"]
ActionType = Literal[
    "select_case",
    "inspect_case",
    "query_system",
    "retrieve_policy",
    "add_evidence",
    "remove_evidence",
    "set_strategy",
    "submit_representment",
    "resolve_case",
]


class CaseQueueItem(BaseModel):
    """Queue-level summary of a chargeback case."""

    case_id: str
    transaction_id: str
    transaction_timestamp: str
    dispute_opened_at: str
    merchant_name: str
    merchant_mcc: str
    masked_card: str
    card_network: str
    network_reason_code: str
    response_window_days: int
    amount: float
    currency: str
    reason_code: str
    status: str
    summary: str
    deadline_step: int
    steps_until_deadline: int


class EvidenceCard(BaseModel):
    """Evidence snippet visible to the agent."""

    evidence_id: str
    source_system: SystemName
    title: str
    summary: str
    attached: bool = False


class PolicyView(BaseModel):
    """Visible reason-code policy guidance."""

    reason_code: str
    guidance: str
    required_evidence: list[str] = Field(default_factory=list)


class VisibleCase(BaseModel):
    """Current workspace for the selected case."""

    case_id: str
    transaction_id: str
    transaction_timestamp: str
    dispute_opened_at: str
    order_id: str
    customer_id: str
    merchant_name: str
    merchant_mcc: str
    masked_card: str
    card_network: str
    network_reason_code: str
    response_window_days: int
    compelling_evidence_category: str
    amount: float
    currency: str
    reason_code: str
    status: str
    current_strategy: StrategyName | None = None
    summary: str
    inspection_notes: str | None = None
    systems_revealed: list[SystemName] = Field(default_factory=list)
    retrieved_evidence: list[EvidenceCard] = Field(default_factory=list)
    attached_evidence: list[EvidenceCard] = Field(default_factory=list)
    policy: PolicyView | None = None
    submission_status: str | None = None


class TaskSummary(BaseModel):
    """Metadata for a built-in task."""

    task_id: str
    title: str
    difficulty: Literal["easy", "medium", "hard", "nightmare"]
    objective: str
    description: str
    max_steps: int
    case_count: int


class ActionTraceItem(BaseModel):
    """Compact action history row."""

    step_index: int
    action_type: str
    case_id: str | None = None
    outcome: str
    reward: float


class CaseResolutionState(BaseModel):
    """Public case state in the current episode."""

    case_id: str
    status: str
    current_strategy: StrategyName | None = None
    resolved: bool = False
    steps_until_deadline: int


class CaseScoreBreakdown(BaseModel):
    """Per-case grading breakdown."""

    case_id: str
    strategy_correctness: float
    evidence_quality: float
    packet_validity: float
    deadline_compliance: float
    efficiency: float
    outcome_quality: float
    note_quality: float = 0.0
    weighted_score: float
    final_resolution: str
    notes: str


class GraderReport(BaseModel):
    """Episode-level deterministic grade report."""

    episode_id: str
    task_id: str
    total_score: float
    normalized_score: float
    completed: bool
    case_reports: list[CaseScoreBreakdown] = Field(default_factory=list)
    summary: str


class BaselineTaskResult(BaseModel):
    """Baseline score for one task."""

    task_id: str
    title: str
    score: float
    steps_used: int
    final_status: str


class BaselineRunResult(BaseModel):
    """Aggregate baseline result payload."""

    provider: str
    model_name: str
    mode: str
    provider_calls_attempted: int = 0
    provider_calls_succeeded: int = 0
    provider_errors: dict[str, int] = Field(default_factory=dict)
    task_results: list[BaselineTaskResult]
    average_score: float


class TasksResponse(BaseModel):
    """Payload returned by /tasks."""

    tasks: list[TaskSummary]
    action_schema: dict[str, Any]


class ChargebackOpsAction(Action):
    """Action schema for ChargebackOps."""

    action_type: ActionType
    case_id: str | None = Field(default=None, max_length=64, description="Target case id when applicable")
    system_name: SystemName | None = Field(
        default=None,
        description="System to query when action_type is query_system",
    )
    evidence_ids: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Evidence ids to attach or remove",
    )
    strategy: StrategyName | None = Field(
        default=None,
        description="Strategy to set or use when resolving a case",
    )
    note: str | None = Field(
        default=None,
        max_length=500,
        description="Optional short rationale for the action",
    )


class ChargebackOpsObservation(Observation):
    """Observation returned by reset() and step()."""

    task_id: str
    task_title: str
    difficulty: Literal["easy", "medium", "hard", "nightmare"]
    objective: str
    selected_case_id: str | None = None
    queue: list[CaseQueueItem] = Field(default_factory=list)
    visible_case: VisibleCase | None = None
    last_action_result: str = ""
    available_actions: list[str] = Field(default_factory=list)
    steps_remaining: int
    progress_score: float = 0.0
    info: dict[str, Any] = Field(default_factory=dict)
    grader_report: GraderReport | None = None


class ChargebackOpsState(State):
    """Extended environment state returned by state()."""

    task_id: str
    task_title: str
    difficulty: Literal["easy", "medium", "hard", "nightmare"]
    objective: str
    selected_case_id: str | None = None
    queue_state: list[CaseResolutionState] = Field(default_factory=list)
    action_history: list[ActionTraceItem] = Field(default_factory=list)
    metrics: dict[str, float] = Field(default_factory=dict)
    latest_grade: float | None = None
    grader_report: GraderReport | None = None
    completed: bool = False
