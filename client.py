"""WebSocket client for ChargebackOps."""

from __future__ import annotations

from typing import Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import (
    ActionTraceItem,
    BaselineRunResult,
    CaseQueueItem,
    CaseResolutionState,
    ChargebackOpsAction,
    ChargebackOpsObservation,
    ChargebackOpsState,
    EvidenceCard,
    GraderReport,
    PolicyView,
    VisibleCase,
)


def _parse_evidence(payload: dict[str, Any]) -> EvidenceCard:
    return EvidenceCard(**payload)


def _parse_policy(payload: dict[str, Any] | None) -> PolicyView | None:
    if payload is None:
        return None
    return PolicyView(**payload)


def _parse_visible_case(payload: dict[str, Any] | None) -> VisibleCase | None:
    if payload is None:
        return None
    data = dict(payload)
    data["retrieved_evidence"] = [
        _parse_evidence(item) for item in data.get("retrieved_evidence", [])
    ]
    data["attached_evidence"] = [
        _parse_evidence(item) for item in data.get("attached_evidence", [])
    ]
    data["policy"] = _parse_policy(data.get("policy"))
    return VisibleCase(**data)


def _parse_grader(payload: dict[str, Any] | None) -> GraderReport | None:
    if payload is None:
        return None
    return GraderReport(**payload)


class ChargebackOpsEnv(
    EnvClient[ChargebackOpsAction, ChargebackOpsObservation, ChargebackOpsState]
):
    """Typed client for the ChargebackOps environment."""

    def _step_payload(self, action: ChargebackOpsAction) -> dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[ChargebackOpsObservation]:
        obs_data = dict(payload.get("observation", {}))
        obs_data["queue"] = [CaseQueueItem(**item) for item in obs_data.get("queue", [])]
        obs_data["visible_case"] = _parse_visible_case(obs_data.get("visible_case"))
        obs_data["grader_report"] = _parse_grader(obs_data.get("grader_report"))
        observation = ChargebackOpsObservation(
            **obs_data,
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict[str, Any]) -> ChargebackOpsState:
        data = dict(payload)
        data["queue_state"] = [
            CaseResolutionState(**item) for item in data.get("queue_state", [])
        ]
        data["action_history"] = [
            ActionTraceItem(**item) for item in data.get("action_history", [])
        ]
        data["grader_report"] = _parse_grader(data.get("grader_report"))
        return ChargebackOpsState(**data)
