"""Brutal local audit for ChargebackOps agent quality.

This script is intentionally harsher than the standard unit tests:

- profiles any datasets placed under ``data/``
- derives deterministic seeds from dataset rows
- runs the heuristic agent across generated easy/medium/hard tasks
- compares it against a deliberately weak control policy
- reports score gaps, failure counts, and difficulty behavior

It does not require external APIs and is safe to run offline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from runners.baseline_runner import _heuristic_pick, _obvious_next_action, candidate_actions
from evaluation.grading import grade_episode
from core.models import ChargebackOpsAction
from server.chargeback_ops_environment import ChargebackOpsEnvironment
from scenarios.simulation import get_task

DATA_DIR = Path("data")

AMOUNT_COLUMNS = (
    "transaction_amount",
    "amt",
    "amount",
)
FRAUD_COLUMNS = (
    "isFraud",
    "is_fraud",
)


def _stable_seed(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _detect_amount_column(fieldnames: list[str]) -> str | None:
    for name in AMOUNT_COLUMNS:
        if name in fieldnames:
            return name
    return None


def _detect_fraud_column(fieldnames: list[str]) -> str | None:
    for name in FRAUD_COLUMNS:
        if name in fieldnames:
            return name
    return None


def _quantile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return ordered[index]


def _map_iso_reason(code: str, description: str) -> str | None:
    text = f"{code} {description}".lower()
    if "unauthorized" in text or "fraud" in text or code.upper().startswith("FR"):
        return "fraud_cnp"
    if "not received" in text or "goods or services not received" in text or code.upper().startswith("NR"):
        return "goods_not_received"
    if "credit not processed" in text or "refund" in text:
        return "credit_not_processed"
    if "duplicate" in text:
        return "duplicate_processing"
    if "not as described" in text or "defective" in text:
        return "product_not_as_described"
    if "service not provided" in text or "service not rendered" in text:
        return "service_not_provided"
    return None


def profile_dataset(path: Path, row_limit: int = 5000) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        amount_col = _detect_amount_column(fieldnames)
        fraud_col = _detect_fraud_column(fieldnames)

        sampled_rows = 0
        amounts: list[float] = []
        fraud_values: list[int] = []
        mapped_reason_counts: dict[str, int] = {}

        for row in reader:
            sampled_rows += 1
            if amount_col and row.get(amount_col):
                try:
                    amounts.append(float(row[amount_col]))
                except ValueError:
                    pass
            if fraud_col and row.get(fraud_col) is not None:
                try:
                    fraud_values.append(int(float(row[fraud_col])))
                except ValueError:
                    pass
            if "chargeback_reason_code" in row and "chargeback_reason_description" in row:
                mapped = _map_iso_reason(
                    row.get("chargeback_reason_code", ""),
                    row.get("chargeback_reason_description", ""),
                )
                if mapped is not None:
                    mapped_reason_counts[mapped] = mapped_reason_counts.get(mapped, 0) + 1
            if sampled_rows >= row_limit:
                break

    fraud_rate = None
    if fraud_values:
        fraud_rate = round(sum(fraud_values) / len(fraud_values), 4)

    return {
        "file": path.name,
        "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
        "sampled_rows": sampled_rows,
        "amount_column": amount_col,
        "fraud_column": fraud_col,
        "amount_stats": {
            "min": min(amounts) if amounts else None,
            "p50": _quantile(amounts, 0.50),
            "p90": _quantile(amounts, 0.90),
            "max": max(amounts) if amounts else None,
        },
        "fraud_rate": fraud_rate,
        "mapped_reason_counts": mapped_reason_counts or None,
    }


def derive_dataset_seeds(data_dir: Path, seeds_needed: int) -> list[int]:
    seeds: list[int] = []
    if not data_dir.exists():
        return list(range(100, 100 + seeds_needed))

    for path in sorted(data_dir.glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                token = "|".join(
                    f"{key}={row[key]}"
                    for key in sorted(row.keys())[:8]
                )
                seeds.append(_stable_seed(f"{path.name}|{token}"))
                if len(seeds) >= seeds_needed:
                    return seeds

    if len(seeds) < seeds_needed:
        start = len(seeds) + 100
        seeds.extend(range(start, start + seeds_needed - len(seeds)))
    return seeds[:seeds_needed]


def _bad_policy_action(observation) -> ChargebackOpsAction:
    if observation.selected_case_id is None:
        open_case = next(case for case in observation.queue if case.status == "open")
        return ChargebackOpsAction(action_type="select_case", case_id=open_case.case_id)

    case_id = observation.selected_case_id
    visible_case = observation.visible_case
    if visible_case and visible_case.current_strategy is None:
        return ChargebackOpsAction(
            action_type="set_strategy",
            case_id=case_id,
            strategy="accept_chargeback",
        )
    if visible_case and visible_case.current_strategy == "accept_chargeback":
        return ChargebackOpsAction(
            action_type="resolve_case",
            case_id=case_id,
            strategy="accept_chargeback",
        )
    return ChargebackOpsAction(
        action_type="query_system",
        case_id=case_id,
        system_name="payment",
    )


def run_episode(task_id: str, policy: str = "heuristic") -> dict[str, Any]:
    env = ChargebackOpsEnvironment()
    observation = env.reset(task_id=task_id)
    total_reward = 0.0
    stalled = False

    while not observation.done:
        if policy == "heuristic":
            payload = observation.model_dump()
            candidates = candidate_actions(payload)
            if not candidates:
                stalled = True
                break
            candidate = _obvious_next_action(payload, candidates) or _heuristic_pick(candidates)
            action = candidate.action
        elif policy == "bad":
            action = _bad_policy_action(observation)
        else:
            raise ValueError(f"Unknown policy '{policy}'.")
        observation = env.step(action)
        total_reward += observation.reward or 0.0

    report = observation.grader_report
    if report is None:
        report = grade_episode(
            get_task(task_id),
            env._progress_by_case,  # type: ignore[attr-defined]
            env.state.step_count,
            env.state.episode_id or "",
            completed=env.state.completed,
        )

    return {
        "task_id": task_id,
        "policy": policy,
        "score": round(report.normalized_score, 4),
        "reward": round(total_reward, 4),
        "steps": env.state.step_count,
        "summary": report.summary,
        "stalled": stalled,
    }


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [item["score"] for item in results]
    rewards = [item["reward"] for item in results]
    steps = [item["steps"] for item in results]
    return {
        "episodes": len(results),
        "avg_score": round(statistics.mean(scores), 4) if scores else None,
        "min_score": min(scores) if scores else None,
        "max_score": max(scores) if scores else None,
        "avg_reward": round(statistics.mean(rewards), 4) if rewards else None,
        "avg_steps": round(statistics.mean(steps), 2) if steps else None,
        "stall_count": sum(1 for item in results if item.get("stalled")),
    }


def evaluate_generated_suite(
    seeds: list[int],
    episodes_per_difficulty: int,
    control_episodes: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    heuristic_overall: list[dict[str, Any]] = []
    bad_overall: list[dict[str, Any]] = []

    for difficulty in ("easy", "medium", "hard"):
        heuristic_results: list[dict[str, Any]] = []
        bad_results: list[dict[str, Any]] = []

        for seed in seeds[:episodes_per_difficulty]:
            heuristic_results.append(
                run_episode(f"generated_{difficulty}_s{seed}", policy="heuristic")
            )

        for seed in seeds[:control_episodes]:
            bad_results.append(
                run_episode(f"generated_{difficulty}_s{seed}", policy="bad")
            )

        heuristic_overall.extend(heuristic_results)
        bad_overall.extend(bad_results)
        report[difficulty] = {
            "heuristic": aggregate_results(heuristic_results),
            "bad_control": aggregate_results(bad_results),
            "heuristic_beats_bad": (
                aggregate_results(heuristic_results)["avg_score"]
                > aggregate_results(bad_results)["avg_score"]
            ),
            "sample_tasks": [item["task_id"] for item in heuristic_results[:3]],
        }

    report["overall"] = {
        "heuristic": aggregate_results(heuristic_overall),
        "bad_control": aggregate_results(bad_overall),
        "heuristic_beats_bad": (
            aggregate_results(heuristic_overall)["avg_score"]
            > aggregate_results(bad_overall)["avg_score"]
        ),
    }
    easy_avg = report["easy"]["heuristic"]["avg_score"]
    hard_avg = report["hard"]["heuristic"]["avg_score"]
    report["difficulty_signal"] = {
        "easy_avg_score": easy_avg,
        "hard_avg_score": hard_avg,
        "easy_harder_gap_expected": round((easy_avg or 0.0) - (hard_avg or 0.0), 4),
    }
    return report


def evaluate_fixed_tasks() -> dict[str, Any]:
    task_ids = (
        "goods_not_received_easy",
        "fraud_signal_ambiguity",
        "queue_optimization_hard",
    )
    results = [run_episode(task_id, policy="heuristic") for task_id in task_ids]
    return {
        "tasks": results,
        "aggregate": aggregate_results(results),
    }


def build_report(
    data_dir: Path,
    episodes_per_difficulty: int,
    control_episodes: int,
    dataset_profile_rows: int,
) -> dict[str, Any]:
    datasets = []
    if data_dir.exists():
        for path in sorted(data_dir.glob("*.csv")):
            datasets.append(profile_dataset(path, row_limit=dataset_profile_rows))

    seeds = derive_dataset_seeds(
        data_dir,
        seeds_needed=max(episodes_per_difficulty, control_episodes),
    )
    generated = evaluate_generated_suite(
        seeds,
        episodes_per_difficulty=episodes_per_difficulty,
        control_episodes=control_episodes,
    )
    fixed = evaluate_fixed_tasks()

    overall_gap = (
        generated["overall"]["heuristic"]["avg_score"]
        - generated["overall"]["bad_control"]["avg_score"]
    )
    verdict = {
        "score_bounds_ok": all(
            0.0 <= item["score"] <= 1.0 for item in fixed["tasks"]
        ),
        "heuristic_beats_bad_overall": generated["overall"]["heuristic_beats_bad"],
        "difficulty_signal_present": generated["difficulty_signal"]["easy_avg_score"]
        >= generated["difficulty_signal"]["hard_avg_score"],
        "overall_gap": round(overall_gap, 4),
    }
    verdict["status"] = (
        "strong"
        if verdict["heuristic_beats_bad_overall"]
        and verdict["difficulty_signal_present"]
        and overall_gap >= 0.15
        else "warning"
    )

    return {
        "datasets": datasets,
        "dataset_seed_sample": seeds[: min(10, len(seeds))],
        "fixed_tasks": fixed,
        "generated_suite": generated,
        "verdict": verdict,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a brutal local audit of ChargebackOps.")
    parser.add_argument("--episodes-per-difficulty", type=int, default=12)
    parser.add_argument("--control-episodes", type=int, default=6)
    parser.add_argument("--dataset-profile-rows", type=int, default=5000)
    args = parser.parse_args()

    report = build_report(
        DATA_DIR,
        episodes_per_difficulty=max(1, args.episodes_per_difficulty),
        control_episodes=max(1, args.control_episodes),
        dataset_profile_rows=max(100, args.dataset_profile_rows),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
