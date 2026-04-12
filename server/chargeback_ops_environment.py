"""Core environment implementation for ChargebackOps."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..core.episode_store import record_report
    from ..evaluation.grading import grade_episode
    from ..evaluation.rubrics import ChargebackOpsEpisodeRubric
    from ..core.models import (
        ActionTraceItem,
        CaseQueueItem,
        CaseResolutionState,
        ChargebackOpsAction,
        ChargebackOpsObservation,
        ChargebackOpsState,
        EvidenceCard,
        PolicyView,
        VisibleCase,
    )
    from ..scenarios.simulation import (
        ActionRecord,
        CaseProgress,
        InternalCase,
        get_task,
    )
except ImportError:  # pragma: no cover
    from core.episode_store import record_report
    from evaluation.grading import grade_episode
    from evaluation.rubrics import ChargebackOpsEpisodeRubric
    from core.models import (
        ActionTraceItem,
        CaseQueueItem,
        CaseResolutionState,
        ChargebackOpsAction,
        ChargebackOpsObservation,
        ChargebackOpsState,
        EvidenceCard,
        PolicyView,
        VisibleCase,
    )
    from scenarios.simulation import ActionRecord, CaseProgress, InternalCase, get_task


class ChargebackOpsEnvironment(
    Environment[ChargebackOpsAction, ChargebackOpsObservation, ChargebackOpsState]
):
    """Synthetic merchant chargeback representment environment."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        super().__init__(rubric=ChargebackOpsEpisodeRubric())
        self._task = get_task("goods_not_received_easy")
        self._selected_case_id: str | None = None
        self._last_action_result = "Environment initialized."
        self._action_history: list[ActionRecord] = []
        self._progress_by_case: dict[str, CaseProgress] = {}
        self._state = ChargebackOpsState(
            episode_id=str(uuid4()),
            step_count=0,
            task_id=self._task.task_id,
            task_title=self._task.title,
            difficulty=self._task.difficulty,
            objective=self._task.objective,
        )
        self._done = False
        self._latest_report = None
        self._reset_task_state()

    _MERCHANT_PROFILES = {
        "goods_not_received": ("Northwind Home", "5712"),
        "fraud_cnp": ("TechForge Direct", "5732"),
        "credit_not_processed": ("StreamClub Media", "4899"),
        "duplicate_processing": ("Meridian Retail", "5311"),
        "product_not_as_described": ("Aurora Commerce", "5942"),
        "service_not_provided": ("Metro Service Group", "7299"),
    }

    def _reset_task_state(self) -> None:
        self._progress_by_case = {
            case.case_id: CaseProgress() for case in self._task.cases
        }
        self._selected_case_id = None
        self._action_history = []
        self._last_action_result = f"{self._task.title} ready."
        self._done = False
        self._latest_report = None

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs,
    ) -> ChargebackOpsObservation:
        task_id = kwargs.get("task_id")
        difficulty = kwargs.get("difficulty")
        _VALID_DIFFICULTIES = {"easy", "medium", "hard", "nightmare"}
        if difficulty is not None and difficulty not in _VALID_DIFFICULTIES:
            raise ValueError(
                f"Invalid difficulty {difficulty!r}. "
                f"Must be one of: {', '.join(sorted(_VALID_DIFFICULTIES))}"
            )
        if task_id is None and difficulty in _VALID_DIFFICULTIES:
            resolved_seed = (
                seed if seed is not None else int(kwargs.get("generated_seed", 42))
            )
            task_id = f"generated_{difficulty}_s{resolved_seed}"
        if task_id is None:
            task_id = "goods_not_received_easy"
        self._task = get_task(task_id)
        self._state = ChargebackOpsState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=self._task.task_id,
            task_title=self._task.title,
            difficulty=self._task.difficulty,
            objective=self._task.objective,
        )
        self._reset_task_state()
        self._reset_rubric()
        return self._build_observation(reward=0.0, done=False)

    def step(
        self,
        action: ChargebackOpsAction,
        timeout_s: float | None = None,
        **kwargs,
    ) -> ChargebackOpsObservation:
        del timeout_s, kwargs
        if self._done:
            return self._build_observation(
                reward=-0.1,
                done=True,
                result="Episode already completed. Reset to start another task.",
            )

        self._state.step_count += 1
        reward = 0.0
        result = ""

        try:
            reward, result = self._apply_action(action)
        except ValueError as exc:
            reward = -0.12
            result = str(exc)
            case_id = action.case_id or self._selected_case_id
            if case_id and case_id in self._progress_by_case:
                self._progress_by_case[case_id].invalid_actions += 1

        reward += self._apply_deadline_penalties()
        done = self._check_done()
        if done:
            report = grade_episode(
                self._task,
                self._progress_by_case,
                self._state.step_count,
                self._state.episode_id or "",
                completed=self._all_cases_resolved(),
            )
            self._latest_report = report
            self._state.latest_grade = report.normalized_score
            self._state.grader_report = report
            self._state.completed = True
            record_report(report)
            reward += 0.5 * report.normalized_score
        else:
            self._state.latest_grade = self._estimated_progress_score()
            self._state.completed = False

        self._last_action_result = result
        self._action_history.append(
            ActionRecord(
                step_index=self._state.step_count,
                action_type=action.action_type,
                case_id=action.case_id or self._selected_case_id,
                outcome=result,
                reward=round(reward, 4),
            )
        )
        return self._build_observation(
            reward=round(reward, 4),
            done=done,
            result=result,
        )

    def _apply_action(self, action: ChargebackOpsAction) -> tuple[float, str]:
        if action.action_type == "select_case":
            return self._select_case(action.case_id)
        case = self._require_case(action.case_id)

        if action.action_type == "inspect_case":
            return self._inspect_case(case)
        if action.action_type == "query_system":
            return self._query_system(case, action.system_name)
        if action.action_type == "retrieve_policy":
            return self._retrieve_policy(case)
        if action.action_type == "add_evidence":
            return self._add_evidence(case, action.evidence_ids)
        if action.action_type == "remove_evidence":
            return self._remove_evidence(case, action.evidence_ids)
        if action.action_type == "set_strategy":
            return self._set_strategy(case, action.strategy)
        if action.action_type == "submit_representment":
            return self._submit_representment(case, note=action.note)
        if action.action_type == "resolve_case":
            return self._resolve_case(case, action.strategy)
        raise ValueError(f"Unsupported action_type '{action.action_type}'.")

    def _select_case(self, case_id: str | None) -> tuple[float, str]:
        if not case_id:
            raise ValueError("select_case requires case_id.")
        case = self._lookup_case(case_id)
        self._selected_case_id = case.case_id
        progress = self._progress_by_case[case.case_id]
        if progress.resolution_status != "open":
            return -0.02, f"Case {case.case_id} is already resolved."
        return 0.02, f"Selected case {case.case_id}."

    def _inspect_case(self, case: InternalCase) -> tuple[float, str]:
        progress = self._progress_by_case[case.case_id]
        if progress.inspected:
            return -0.01, f"Case {case.case_id} was already inspected."
        progress.inspected = True
        return 0.04, f"Inspected case {case.case_id}."

    def _query_system(
        self,
        case: InternalCase,
        system_name: str | None,
    ) -> tuple[float, str]:
        if system_name is None:
            raise ValueError("query_system requires system_name.")
        progress = self._progress_by_case[case.case_id]
        if system_name in progress.revealed_systems:
            progress.duplicate_queries += 1
            return (
                -0.03,
                f"System '{system_name}' was already queried for case {case.case_id}.",
            )

        progress.revealed_systems.add(system_name)
        new_evidence = case.evidence_by_system.get(system_name, ())
        progress.retrieved_evidence_ids.update(
            item.evidence_id for item in new_evidence
        )
        helpful = sum(1 for item in new_evidence if item.helpful)
        if helpful > 0:
            return 0.06 + 0.01 * helpful, (
                f"Queried {system_name} for case {case.case_id}; found {len(new_evidence)} evidence items, "
                f"including {helpful} useful ones."
            )
        return -0.01 if len(new_evidence) == 0 else 0.01, (
            f"Queried {system_name} for case {case.case_id}; found {len(new_evidence)} evidence items."
        )

    def _retrieve_policy(self, case: InternalCase) -> tuple[float, str]:
        progress = self._progress_by_case[case.case_id]
        if progress.policy_retrieved:
            return -0.01, f"Policy already retrieved for case {case.case_id}."
        progress.policy_retrieved = True
        return 0.05, f"Retrieved policy guidance for case {case.case_id}."

    def _add_evidence(
        self,
        case: InternalCase,
        evidence_ids: list[str],
    ) -> tuple[float, str]:
        if not evidence_ids:
            raise ValueError("add_evidence requires at least one evidence id.")
        progress = self._progress_by_case[case.case_id]
        all_evidence = self._evidence_map(case)
        reward = 0.0
        added = []
        for evidence_id in evidence_ids:
            if evidence_id not in progress.retrieved_evidence_ids:
                reward -= 0.04
                continue
            if evidence_id in progress.attached_evidence_ids:
                reward -= 0.02
                continue
            progress.attached_evidence_ids.append(evidence_id)
            added.append(evidence_id)
            evidence = all_evidence[evidence_id]
            if evidence.helpful:
                reward += 0.08
            elif evidence.harmful:
                reward -= 0.08
            else:
                reward += 0.01
        return reward if added else -0.05, (
            f"Attached evidence {', '.join(added)} to case {case.case_id}."
            if added
            else f"No evidence was attached to case {case.case_id}."
        )

    def _remove_evidence(
        self,
        case: InternalCase,
        evidence_ids: list[str],
    ) -> tuple[float, str]:
        if not evidence_ids:
            raise ValueError("remove_evidence requires at least one evidence id.")
        progress = self._progress_by_case[case.case_id]
        evidence_map = self._evidence_map(case)
        reward = 0.0
        removed = []
        for evidence_id in evidence_ids:
            if evidence_id not in progress.attached_evidence_ids:
                reward -= 0.02
                continue
            progress.attached_evidence_ids.remove(evidence_id)
            removed.append(evidence_id)
            if evidence_map[evidence_id].harmful:
                reward += 0.05
            elif evidence_map[evidence_id].helpful:
                reward -= 0.03
            else:
                reward += 0.01
        return reward if removed else -0.04, (
            f"Removed evidence {', '.join(removed)} from case {case.case_id}."
            if removed
            else f"No evidence was removed from case {case.case_id}."
        )

    def _set_strategy(
        self,
        case: InternalCase,
        strategy: str | None,
    ) -> tuple[float, str]:
        if strategy is None:
            raise ValueError("set_strategy requires strategy.")
        progress = self._progress_by_case[case.case_id]
        progress.current_strategy = strategy
        if strategy == case.optimal_strategy:
            return (
                0.1,
                f"Set the optimal strategy '{strategy}' for case {case.case_id}.",
            )
        if strategy in case.acceptable_strategies:
            return (
                0.03,
                f"Set an acceptable fallback strategy '{strategy}' for case {case.case_id}.",
            )
        return -0.08, f"Set a weak strategy '{strategy}' for case {case.case_id}."

    def _submit_representment(
        self, case: InternalCase, *, note: str | None = None
    ) -> tuple[float, str]:
        progress = self._progress_by_case[case.case_id]
        progress.submit_attempts += 1
        if note:
            progress.representment_note = note
        if progress.current_strategy != "contest":
            raise ValueError(
                "submit_representment requires current strategy to be 'contest'."
            )
        if progress.resolution_status != "open":
            return -0.05, f"Case {case.case_id} is already resolved."

        attached = set(progress.attached_evidence_ids)
        missing = set(case.required_evidence_ids).difference(attached)
        harmful = set(case.harmful_evidence_ids).intersection(attached)
        if self._state.step_count > case.deadline_step:
            progress.final_resolution = "contest"
            progress.resolution_status = "lost_late"
            progress.resolved_at_step = self._state.step_count
            return (
                -0.2,
                f"Representment for case {case.case_id} was submitted after the deadline.",
            )
        if missing:
            progress.final_resolution = "contest"
            progress.resolution_status = "lost_incomplete"
            progress.resolved_at_step = self._state.step_count
            return -0.18, (
                f"Representment for case {case.case_id} is incomplete; missing {', '.join(sorted(missing))}."
            )
        if harmful:
            progress.final_resolution = "contest"
            progress.resolution_status = "lost_harmful_evidence"
            progress.resolved_at_step = self._state.step_count
            return -0.15, (
                f"Representment for case {case.case_id} included harmful evidence {', '.join(sorted(harmful))}."
            )

        progress.final_resolution = "contest"
        progress.resolved_at_step = self._state.step_count
        if case.optimal_strategy == "contest":
            progress.resolution_status = "won"
            return (
                0.2,
                f"Submitted a strong representment package for case {case.case_id}.",
            )
        progress.resolution_status = "lost_contest"
        return (
            -0.12,
            f"Contested case {case.case_id}, but the case was not supportable.",
        )

    def _resolve_case(
        self,
        case: InternalCase,
        strategy: str | None,
    ) -> tuple[float, str]:
        progress = self._progress_by_case[case.case_id]
        resolution = strategy or progress.current_strategy
        if resolution not in {"accept_chargeback", "issue_refund"}:
            raise ValueError(
                "resolve_case requires strategy accept_chargeback or issue_refund."
            )
        if progress.resolution_status != "open":
            return -0.04, f"Case {case.case_id} is already resolved."
        progress.final_resolution = resolution
        progress.current_strategy = resolution
        progress.resolved_at_step = self._state.step_count
        progress.resolution_status = (
            "refunded" if resolution == "issue_refund" else "accepted_chargeback"
        )
        if self._state.step_count > case.deadline_step:
            return -0.15, f"Resolved case {case.case_id} after the response deadline."
        if resolution == case.optimal_strategy:
            return (
                0.16,
                f"Resolved case {case.case_id} with the optimal non-contest strategy.",
            )
        if resolution in case.acceptable_strategies:
            return (
                0.06,
                f"Resolved case {case.case_id} with an acceptable fallback strategy.",
            )
        return -0.12, f"Resolved case {case.case_id} with the wrong strategy."

    def _apply_deadline_penalties(self) -> float:
        penalty = 0.0
        for case in self._task.cases:
            progress = self._progress_by_case[case.case_id]
            if (
                progress.resolution_status == "open"
                and self._state.step_count > case.deadline_step
            ):
                if not progress.deadline_penalized:
                    progress.deadline_penalized = True
                    penalty -= 0.15
        return penalty

    def _check_done(self) -> bool:
        if self._all_cases_resolved():
            self._done = True
        elif self._state.step_count >= self._task.max_steps:
            self._done = True
        return self._done

    def _all_cases_resolved(self) -> bool:
        return all(
            progress.resolution_status != "open"
            for progress in self._progress_by_case.values()
        )

    def _lookup_case(self, case_id: str) -> InternalCase:
        for case in self._task.cases:
            if case.case_id == case_id:
                return case
        raise ValueError(f"Unknown case_id '{case_id}'.")

    def _require_case(self, case_id: str | None) -> InternalCase:
        target_case_id = case_id or self._selected_case_id
        if target_case_id is None:
            raise ValueError("Select a case before taking this action.")
        return self._lookup_case(target_case_id)

    def _evidence_map(self, case: InternalCase):
        return {
            item.evidence_id: item
            for items in case.evidence_by_system.values()
            for item in items
        }

    def _case_fingerprint(self, case: InternalCase) -> str:
        return hashlib.sha1(
            f"{case.case_id}|{case.order_id}|{case.customer_id}|{case.reason_code}".encode(
                "utf-8"
            )
        ).hexdigest()

    def _case_display_metadata(self, case: InternalCase) -> dict[str, str]:
        fingerprint = self._case_fingerprint(case)
        merchant_name, merchant_mcc = self._MERCHANT_PROFILES.get(
            case.reason_code,
            ("Atlas Commerce", "5999"),
        )
        last4 = str(int(fingerprint[:8], 16))[-4:].zfill(4)
        base_time = datetime(2025, 1, 1, 9, 0, tzinfo=timezone.utc)
        txn_minutes = int(fingerprint[8:14], 16) % (180 * 24 * 60)
        dispute_lag_hours = 24 + (int(fingerprint[14:18], 16) % 96)
        transaction_dt = base_time + timedelta(minutes=txn_minutes)
        dispute_dt = transaction_dt + timedelta(hours=dispute_lag_hours)
        return {
            "transaction_id": f"txn_{fingerprint[:14]}",
            "transaction_timestamp": transaction_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "dispute_opened_at": dispute_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "merchant_name": merchant_name,
            "merchant_mcc": merchant_mcc,
            "masked_card": f"**** **** **** {last4}",
        }

    def _episode_metrics(self) -> dict[str, float]:
        """User-observable episode metrics.  Never exposes grader-internal
        labels such as required/helpful/harmful evidence IDs or coverage
        against the hidden answer key."""
        open_cases = 0
        urgent_cases = 0
        resolved_cases = 0
        total_attached = 0
        total_retrieved = 0

        for case in self._task.cases:
            progress = self._progress_by_case[case.case_id]
            total_attached += len(progress.attached_evidence_ids)
            total_retrieved += len(progress.retrieved_evidence_ids)
            steps_until_deadline = case.deadline_step - self._state.step_count
            if progress.resolution_status == "open":
                open_cases += 1
                if steps_until_deadline <= 2:
                    urgent_cases += 1
            else:
                resolved_cases += 1

        deadline_pressure = (
            0.0 if len(self._task.cases) == 0 else urgent_cases / len(self._task.cases)
        )
        triage_efficiency = resolved_cases / max(1, self._state.step_count)
        return {
            "open_case_count": float(open_cases),
            "resolved_case_count": float(resolved_cases),
            "deadline_pressure_index": round(deadline_pressure, 4),
            "triage_efficiency": round(triage_efficiency, 4),
            "total_evidence_attached": float(total_attached),
            "total_evidence_retrieved": float(total_retrieved),
        }

    def _selected_case_info(self) -> dict[str, object]:
        """Per-case diagnostic info visible to agents.  Only exposes
        signals an analyst could observe (deadline proximity, which
        systems haven't been queried, counts).  Does NOT expose which
        evidence IDs are required, helpful, or harmful."""
        if self._selected_case_id is None:
            return {
                "deadline_warning": False,
                "unqueried_systems": [],
            }

        case = self._lookup_case(self._selected_case_id)
        progress = self._progress_by_case[case.case_id]
        all_systems = {"orders", "payment", "shipping", "support", "refunds", "risk"}
        return {
            "deadline_warning": (case.deadline_step - self._state.step_count) <= 2,
            "unqueried_systems": sorted(
                all_systems.difference(progress.revealed_systems)
            ),
            "attached_evidence_count": len(progress.attached_evidence_ids),
            "retrieved_evidence_count": len(progress.retrieved_evidence_ids),
            "steps_until_deadline": case.deadline_step - self._state.step_count,
        }

    def _build_queue(self) -> list[CaseQueueItem]:
        queue = []
        for case in self._task.cases:
            progress = self._progress_by_case[case.case_id]
            display = self._case_display_metadata(case)
            queue.append(
                CaseQueueItem(
                    case_id=case.case_id,
                    transaction_id=display["transaction_id"],
                    transaction_timestamp=display["transaction_timestamp"],
                    dispute_opened_at=display["dispute_opened_at"],
                    merchant_name=display["merchant_name"],
                    merchant_mcc=display["merchant_mcc"],
                    masked_card=display["masked_card"],
                    card_network=case.card_network,
                    network_reason_code=case.network_reason_code,
                    response_window_days=case.response_window_days,
                    amount=case.amount,
                    currency=case.currency,
                    reason_code=case.reason_code,
                    status=progress.resolution_status,
                    summary=case.summary,
                    deadline_step=case.deadline_step,
                    steps_until_deadline=case.deadline_step - self._state.step_count,
                )
            )
        return queue

    def _build_visible_case(self) -> VisibleCase | None:
        if self._selected_case_id is None:
            return None
        case = self._lookup_case(self._selected_case_id)
        progress = self._progress_by_case[case.case_id]
        display = self._case_display_metadata(case)
        evidence_map = self._evidence_map(case)
        retrieved = [
            EvidenceCard(
                evidence_id=evidence_id,
                source_system=evidence_map[evidence_id].source_system,
                title=evidence_map[evidence_id].title,
                summary=evidence_map[evidence_id].summary,
                attached=evidence_id in progress.attached_evidence_ids,
            )
            for evidence_id in sorted(progress.retrieved_evidence_ids)
        ]
        attached = [
            EvidenceCard(
                evidence_id=evidence_id,
                source_system=evidence_map[evidence_id].source_system,
                title=evidence_map[evidence_id].title,
                summary=evidence_map[evidence_id].summary,
                attached=True,
            )
            for evidence_id in progress.attached_evidence_ids
        ]
        policy = None
        if progress.policy_retrieved:
            policy = PolicyView(
                reason_code=case.reason_code,
                guidance=case.policy_guidance,
                required_evidence=list(case.policy_requirements),
            )
        return VisibleCase(
            case_id=case.case_id,
            transaction_id=display["transaction_id"],
            transaction_timestamp=display["transaction_timestamp"],
            dispute_opened_at=display["dispute_opened_at"],
            order_id=case.order_id,
            customer_id=case.customer_id,
            merchant_name=display["merchant_name"],
            merchant_mcc=display["merchant_mcc"],
            masked_card=display["masked_card"],
            card_network=case.card_network,
            network_reason_code=case.network_reason_code,
            response_window_days=case.response_window_days,
            compelling_evidence_category=case.compelling_evidence_category,
            amount=case.amount,
            currency=case.currency,
            reason_code=case.reason_code,
            status=progress.resolution_status,
            current_strategy=progress.current_strategy,
            summary=case.summary,
            inspection_notes=case.inspection_notes if progress.inspected else None,
            systems_revealed=sorted(progress.revealed_systems),
            retrieved_evidence=retrieved,
            attached_evidence=attached,
            policy=policy,
            submission_status=progress.resolution_status
            if progress.resolution_status != "open"
            else None,
        )

    def _build_available_actions(self) -> list[str]:
        if self._done:
            return []
        base = ["select_case"]
        if self._selected_case_id is None:
            return base
        case_progress = self._progress_by_case[self._selected_case_id]
        if case_progress.resolution_status != "open":
            return ["select_case"]
        return base + [
            "inspect_case",
            "query_system",
            "retrieve_policy",
            "add_evidence",
            "remove_evidence",
            "set_strategy",
            "submit_representment",
            "resolve_case",
        ]

    def _estimated_progress_score(self) -> float:
        report = grade_episode(
            self._task,
            self._progress_by_case,
            self._state.step_count,
            self._state.episode_id or "",
            completed=False,
        )
        return report.normalized_score

    def _build_observation(
        self,
        reward: float,
        done: bool,
        result: str | None = None,
    ) -> ChargebackOpsObservation:
        progress_score = (
            self._latest_report.normalized_score
            if self._latest_report is not None
            else self._estimated_progress_score()
        )
        self._state.queue_state = [
            CaseResolutionState(
                case_id=case.case_id,
                status=self._progress_by_case[case.case_id].resolution_status,
                current_strategy=self._progress_by_case[case.case_id].current_strategy,
                resolved=self._progress_by_case[case.case_id].resolution_status
                != "open",
                steps_until_deadline=case.deadline_step - self._state.step_count,
            )
            for case in self._task.cases
        ]
        self._state.action_history = [
            ActionTraceItem(
                step_index=record.step_index,
                action_type=record.action_type,
                case_id=record.case_id,
                outcome=record.outcome,
                reward=record.reward,
            )
            for record in self._action_history
        ]
        self._state.selected_case_id = self._selected_case_id
        self._state.metrics = self._episode_metrics()
        observation_info = {
            **self._selected_case_info(),
            "episode_metrics": self._state.metrics,
            "current_task_max_steps": self._task.max_steps,
        }

        return ChargebackOpsObservation(
            task_id=self._task.task_id,
            task_title=self._task.title,
            difficulty=self._task.difficulty,
            objective=self._task.objective,
            selected_case_id=self._selected_case_id,
            queue=self._build_queue(),
            visible_case=self._build_visible_case(),
            last_action_result=result or self._last_action_result,
            available_actions=self._build_available_actions(),
            steps_remaining=max(0, self._task.max_steps - self._state.step_count),
            progress_score=round(progress_score, 4),
            info=observation_info,
            grader_report=self._latest_report,
            done=done,
            reward=reward,
            metadata={
                "reward_components": {
                    "step_reward": reward,
                    "progress_score": round(progress_score, 4),
                },
                "info": observation_info,
            },
        )

    @property
    def state(self) -> ChargebackOpsState:
        return self._state
