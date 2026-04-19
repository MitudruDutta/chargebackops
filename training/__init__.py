"""Training helpers for ChargebackOps.

Lightweight pure-Python wrappers that convert the environment into a
prompt/completion/reward interface compatible with TRL's GRPO trainer.
The module is import-safe without ``trl`` / ``torch`` installed so unit
tests stay fast and offline.
"""

from __future__ import annotations

from .env_adapter import (
    action_from_completion,
    build_prompt,
    parse_completion,
)
from .reward_adapter import (
    EpisodeResult,
    compute_reward,
    run_episode_with_text_policy,
)

__all__ = [
    "EpisodeResult",
    "action_from_completion",
    "build_prompt",
    "compute_reward",
    "parse_completion",
    "run_episode_with_text_policy",
]
