"""Training helpers for ChargebackOps.

Lightweight pure-Python wrappers that convert the environment into a
prompt/completion/reward interface compatible with TRL's GRPO trainer.
The module is import-safe without ``trl`` / ``torch`` installed so unit
tests stay fast and offline.
"""

from __future__ import annotations

from .curve import (
    CheckpointEval,
    TaskOutcome,
    evaluate_checkpoint,
    evaluate_policy_across_tasks,
    plot_training_curve,
)
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
from .sft_dataset import (
    SFTSample,
    action_to_completion,
    build_sft_dataset,
)

__all__ = [
    "CheckpointEval",
    "EpisodeResult",
    "SFTSample",
    "TaskOutcome",
    "action_from_completion",
    "action_to_completion",
    "build_prompt",
    "build_sft_dataset",
    "compute_reward",
    "evaluate_checkpoint",
    "evaluate_policy_across_tasks",
    "parse_completion",
    "plot_training_curve",
    "run_episode_with_text_policy",
]
