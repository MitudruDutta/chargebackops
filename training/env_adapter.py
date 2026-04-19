"""Text prompt / completion adapter for the merchant policy.

Serialize an observation into a compact prompt the model can condition
on, and parse a JSON completion back into a typed
``ChargebackOpsAction``. Both helpers are pure — no provider calls, no
side effects — so they are cheap to unit-test.
"""

from __future__ import annotations

import json
from typing import Any

try:
    from ..core.models import ChargebackOpsAction
except ImportError:  # pragma: no cover
    from core.models import ChargebackOpsAction


_SYSTEM_INSTRUCTION = (
    "You play the merchant-side agent in a chargeback dispute. "
    "Look at the observation and choose the single best next action. "
    "Return JSON only: "
    '{"action_type": "...", "case_id": "...", "strategy": "...", '
    '"evidence_ids": [...], "note": "..."} '
    "Use only action_types listed in available_actions. Omit fields you "
    "do not need."
)


_ALLOWED_ACTION_FIELDS: frozenset[str] = frozenset(
    {
        "action_type",
        "case_id",
        "system_name",
        "evidence_ids",
        "compelling_evidence_ids",
        "strategy",
        "note",
    }
)


def _compact_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """Drop fields that add tokens without signal for the merchant policy."""

    visible_case = observation.get("visible_case")
    compact_case: dict[str, Any] | None = None
    if visible_case is not None:
        compact_case = {
            "case_id": visible_case["case_id"],
            "status": visible_case["status"],
            "reason_code": visible_case["reason_code"],
            "amount": visible_case["amount"],
            "currency": visible_case["currency"],
            "current_strategy": visible_case.get("current_strategy"),
            "systems_revealed": visible_case.get("systems_revealed", []),
            "retrieved_evidence": [
                {
                    "evidence_id": item["evidence_id"],
                    "source_system": item["source_system"],
                    "title": item["title"],
                }
                for item in visible_case.get("retrieved_evidence", [])
            ],
            "attached_evidence": [
                item["evidence_id"]
                for item in visible_case.get("attached_evidence", [])
            ],
            "policy": visible_case.get("policy"),
        }

    return {
        "objective": observation.get("objective", ""),
        "selected_case_id": observation.get("selected_case_id"),
        "available_actions": observation.get("available_actions", []),
        "steps_remaining": observation.get("steps_remaining", 0),
        "queue": [
            {
                "case_id": item["case_id"],
                "status": item["status"],
                "reason_code": item["reason_code"],
                "amount": item["amount"],
                "steps_until_deadline": item["steps_until_deadline"],
            }
            for item in observation.get("queue", [])
        ],
        "visible_case": compact_case,
        "last_action_result": observation.get("last_action_result", ""),
    }


def build_prompt(observation: dict[str, Any]) -> str:
    """Return a deterministic prompt for the merchant policy."""

    compact = _compact_observation(observation)
    body = json.dumps(compact, separators=(",", ":"), sort_keys=True)
    return f"{_SYSTEM_INSTRUCTION}\nOBSERVATION:\n{body}\nACTION:"


def parse_completion(text: str) -> dict[str, Any] | None:
    """Parse a model completion into a raw action dict, or return None."""

    if not text:
        return None
    cleaned = text.strip()
    # Strip common code-fence patterns.
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].lstrip()
    # Find the first {...} block so prose before JSON is tolerated.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        data = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return {k: v for k, v in data.items() if k in _ALLOWED_ACTION_FIELDS}


def action_from_completion(text: str) -> ChargebackOpsAction | None:
    """Parse a completion and build a validated :class:`ChargebackOpsAction`."""

    parsed = parse_completion(text)
    if parsed is None or "action_type" not in parsed:
        return None
    try:
        return ChargebackOpsAction(**parsed)
    except Exception:
        return None


__all__ = [
    "action_from_completion",
    "build_prompt",
    "parse_completion",
]
