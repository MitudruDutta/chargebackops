"""LLM softening helper for the scripted Issuer agent.

The deterministic Issuer resolves the ambiguity band (0.40 < score < 0.70)
with a midpoint fallback. When LLM softening is enabled, the agent instead
asks a provider to adjudicate the band and only falls back to the midpoint
rule if every configured provider errors. This keeps benchmarks reproducible
for anyone without API keys while letting keyed runs exercise a realistic
adversarial Issuer.

The module exposes a single factory: :func:`build_default_llm_decider`. It
returns a pure callable ``decider(case, progress, score) -> IssuerDecision |
None`` that tries each provider in turn. Injecting a test double is a drop-in
replacement — the IssuerAgent only calls the callable.
"""

from __future__ import annotations

import json
import os
from typing import Callable

try:
    from .issuer_model import IssuerDecision
    from .simulation import CaseProgress, InternalCase
except ImportError:  # pragma: no cover
    from scenarios.issuer_model import IssuerDecision
    from scenarios.simulation import CaseProgress, InternalCase


IssuerDeciderFn = Callable[
    [InternalCase, CaseProgress, float], "IssuerDecision | None"
]


# Provider name, base URL, API-key env var, default model slug.
_PROVIDER_CHAIN: tuple[tuple[str, str, str, str], ...] = (
    (
        "openrouter",
        "https://openrouter.ai/api/v1",
        "OPENROUTER_API_KEY",
        "openai/gpt-oss-120b",
    ),
    (
        "google",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "GOOGLE_API_KEY",
        "gemini-2.5-flash",
    ),
    (
        "groq",
        "https://api.groq.com/openai/v1",
        "GROQ_API_KEY",
        "llama-3.3-70b-versatile",
    ),
)


_SYSTEM_PROMPT = (
    "You role-play as the fraud-operations reviewer at a card-issuing bank. "
    "A merchant has rebutted a chargeback and you must either ACCEPT their "
    "representment (reverse the chargeback) or REQUEST_MORE_EVIDENCE (keep "
    "the chargeback open and demand compelling evidence). "
    "You are adversarial: protect the cardholder's funds, but do not be "
    "unreasonable when the packet clearly supports the merchant. "
    'Return JSON only: {"decision": "accept" | "request_more_evidence", '
    '"rationale": "one short sentence"}'
)


def _build_user_prompt(
    case: InternalCase, progress: CaseProgress, score: float
) -> str:
    attached = list(progress.attached_evidence_ids)
    required = list(case.required_evidence_ids)
    harmful = list(case.harmful_evidence_ids)
    note = progress.representment_note or ""
    policy = "; ".join(case.policy_requirements) or "none documented"
    return json.dumps(
        {
            "reason_code": case.reason_code,
            "dispute_amount": case.amount,
            "currency": case.currency,
            "network_reason_code": case.network_reason_code,
            "policy_requirements": policy,
            "required_evidence_ids": required,
            "harmful_evidence_ids": harmful,
            "attached_evidence_ids": attached,
            "representment_note": note,
            "deterministic_score": round(score, 3),
        }
    )


def _parse_decision(text: str) -> IssuerDecision | None:
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    raw = str(data.get("decision", "")).strip().lower()
    if raw == "accept":
        return IssuerDecision.ACCEPT
    if raw == "request_more_evidence":
        return IssuerDecision.REQUEST_MORE_EVIDENCE
    return None


def _try_provider(
    base_url: str,
    api_key: str,
    model: str,
    case: InternalCase,
    progress: CaseProgress,
    score: float,
) -> IssuerDecision | None:
    try:
        from openai import OpenAI
    except ImportError:  # pragma: no cover
        return None

    try:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=float(os.getenv("ISSUER_LLM_TIMEOUT_SECONDS", "8")),
            max_retries=0,
        )
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=120,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(case, progress, score)},
            ],
        )
    except Exception:
        return None

    content = ""
    try:
        content = response.choices[0].message.content or ""
    except (AttributeError, IndexError):
        return None
    return _parse_decision(content)


def build_default_llm_decider(
    *, model_override: str | None = None
) -> IssuerDeciderFn:
    """Return a decider that walks the configured provider chain.

    The returned callable is pure in its inputs but impure overall — it reads
    env vars and makes network calls. Tests should inject a stub instead.
    """

    def _decide(
        case: InternalCase, progress: CaseProgress, score: float
    ) -> IssuerDecision | None:
        for _name, base_url, env_var, default_model in _PROVIDER_CHAIN:
            api_key = os.getenv(env_var)
            if not api_key:
                continue
            model = model_override or default_model
            decision = _try_provider(
                base_url=base_url,
                api_key=api_key,
                model=model,
                case=case,
                progress=progress,
                score=score,
            )
            if decision is not None:
                return decision
        return None

    return _decide


__all__ = [
    "IssuerDeciderFn",
    "build_default_llm_decider",
]
