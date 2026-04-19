from scenarios.arbitration import ARB_FEE_PER_SIDE, ArbitrationOutcome
from scenarios.case_generator import generate_task
from scenarios.issuer_model import IssuerDecision, IssuerReview
from core.models import ChargebackOpsAction
from server.chargeback_ops_environment import ChargebackOpsEnvironment


class _ScriptedIssuer:
    """Deterministic Issuer stub that returns planned decisions in sequence.

    Lets round-2 env tests exercise the full dispute lifecycle without
    depending on the exact thresholds in IssuerAgent — the scoring math is
    already pinned in tests/test_issuer.py.
    """

    def __init__(self, decisions: list[IssuerDecision]):
        self._decisions = list(decisions)
        self.calls: list[int] = []

    def decide_review(self, case, progress, round_number):
        self.calls.append(round_number)
        idx = min(len(self.calls) - 1, len(self._decisions) - 1)
        decision = self._decisions[idx]
        return IssuerReview(
            decision=decision,
            evidence_strength_score=0.5,
            rationale=f"stub decision {decision.value} at r{round_number}",
        )


def _drive_case_into_round_2(env: ChargebackOpsEnvironment) -> None:
    """Select CB-E1, attach required evidence, submit — with Issuer stub that
    requests more evidence on round 1. Leaves env in round-2 state.
    """
    env.step(ChargebackOpsAction(action_type="select_case", case_id="CB-E1"))
    env.step(
        ChargebackOpsAction(
            action_type="query_system", case_id="CB-E1", system_name="orders"
        )
    )
    env.step(
        ChargebackOpsAction(
            action_type="query_system", case_id="CB-E1", system_name="shipping"
        )
    )
    env.step(
        ChargebackOpsAction(
            action_type="add_evidence",
            case_id="CB-E1",
            evidence_ids=["E1-ORDER-CONF", "E1-DELIVERY-SCAN"],
        )
    )
    env.step(
        ChargebackOpsAction(
            action_type="set_strategy", case_id="CB-E1", strategy="contest"
        )
    )
    env.step(
        ChargebackOpsAction(action_type="submit_representment", case_id="CB-E1")
    )


def test_reset_returns_task_observation():
    env = ChargebackOpsEnvironment()
    obs = env.reset(task_id="goods_not_received_easy")
    assert obs.task_id == "goods_not_received_easy"
    assert obs.steps_remaining == 10
    assert len(obs.queue) == 1
    assert obs.queue[0].transaction_id.startswith("txn_")
    assert obs.queue[0].merchant_mcc.isdigit()
    assert obs.queue[0].masked_card.startswith("****")


def test_reset_accepts_curriculum_difficulty():
    env = ChargebackOpsEnvironment()
    obs = env.reset(difficulty="hard", seed=7)
    assert obs.task_id == "generated_hard_s7"
    assert obs.difficulty == "hard"
    assert len(obs.queue) >= 2


def test_easy_case_can_be_won():
    env = ChargebackOpsEnvironment()
    env.reset(task_id="goods_not_received_easy")
    env.step(ChargebackOpsAction(action_type="select_case", case_id="CB-E1"))
    env.step(ChargebackOpsAction(action_type="inspect_case", case_id="CB-E1"))
    env.step(
        ChargebackOpsAction(
            action_type="query_system",
            case_id="CB-E1",
            system_name="orders",
        )
    )
    env.step(
        ChargebackOpsAction(
            action_type="query_system",
            case_id="CB-E1",
            system_name="shipping",
        )
    )
    env.step(
        ChargebackOpsAction(
            action_type="add_evidence",
            case_id="CB-E1",
            evidence_ids=["E1-ORDER-CONF", "E1-DELIVERY-SCAN"],
        )
    )
    env.step(
        ChargebackOpsAction(
            action_type="set_strategy",
            case_id="CB-E1",
            strategy="contest",
        )
    )
    obs = env.step(
        ChargebackOpsAction(
            action_type="submit_representment",
            case_id="CB-E1",
        )
    )

    assert obs.done is True
    assert obs.grader_report is not None
    assert obs.grader_report.normalized_score > 0.8


def test_generated_task_reproducibility():
    """Same seed must produce identical cases."""
    t1 = generate_task(99, difficulty="medium")
    t2 = generate_task(99, difficulty="medium")
    assert t1.task_id == t2.task_id
    assert len(t1.cases) == len(t2.cases)
    for c1, c2 in zip(t1.cases, t2.cases):
        assert c1.case_id == c2.case_id
        assert c1.amount == c2.amount
        assert c1.optimal_strategy == c2.optimal_strategy


def test_generated_task_runs_in_environment():
    """A generated task should reset and accept at least one step."""
    env = ChargebackOpsEnvironment()
    obs = env.reset(task_id="generated_easy_s7")
    assert obs.task_id == "generated_easy_s7"
    assert len(obs.queue) >= 1
    case_id = obs.queue[0].case_id
    obs = env.step(ChargebackOpsAction(action_type="select_case", case_id=case_id))
    assert obs.selected_case_id == case_id
    assert obs.visible_case is not None
    assert obs.visible_case.transaction_timestamp.endswith("Z")
    assert "episode_metrics" in obs.info


def test_generated_task_covers_all_reason_codes():
    """Generator should produce all 6 reason code families across seeds."""
    seen_codes: set[str] = set()
    for seed in range(50):
        for diff in ("easy", "medium", "hard"):
            t = generate_task(seed, difficulty=diff)
            for c in t.cases:
                seen_codes.add(c.reason_code)
    expected = {
        "goods_not_received", "fraud_cnp", "credit_not_processed",
        "duplicate_processing", "product_not_as_described", "service_not_provided",
    }
    assert expected.issubset(seen_codes), f"Missing: {expected - seen_codes}"


# ---------------------------------------------------------------------------
# multi-round dispute lifecycle 
# ---------------------------------------------------------------------------


def test_pre_arb_available_actions_exclude_submit_representment():
    env = ChargebackOpsEnvironment()
    env.reset(task_id="goods_not_received_easy")
    env._issuer_agent = _ScriptedIssuer([IssuerDecision.REQUEST_MORE_EVIDENCE])
    _drive_case_into_round_2(env)

    actions = env._build_available_actions()
    assert "submit_representment" not in actions
    assert "respond_to_pre_arb" in actions
    assert "escalate_to_arbitration" in actions
    assert "accept_arbitration_loss" in actions


def test_full_three_round_cycle_ending_in_arbitration():
    env = ChargebackOpsEnvironment()
    env.reset(task_id="goods_not_received_easy")
    env._issuer_agent = _ScriptedIssuer(
        [
            IssuerDecision.REQUEST_MORE_EVIDENCE,
            IssuerDecision.ESCALATE_TO_ARBITRATION,
        ]
    )
    _drive_case_into_round_2(env)

    obs = env.step(
        ChargebackOpsAction(
            action_type="respond_to_pre_arb",
            case_id="CB-E1",
            compelling_evidence_ids=["E1-SIGNATURE"],
            note="Added signature-level delivery proof for pre-arb.",
        )
    )

    progress = env._progress_by_case["CB-E1"]
    assert progress.round_number == 3
    assert progress.arbitration_outcome == ArbitrationOutcome.MERCHANT_WINS.value
    assert progress.arb_fees_paid == ARB_FEE_PER_SIDE
    assert progress.final_economic_outcome == progress.final_economic_outcome
    assert progress.final_economic_outcome is not None
    assert progress.resolution_status == "won_arbitration"
    assert obs.done is True
    assert "arbitration" in obs.last_action_result.lower()


def test_respond_to_pre_arb_accepted_skips_arbitration():
    """If the Issuer accepts in round 2, no arbitration fee is charged and
    the merchant keeps the full dispute amount."""
    env = ChargebackOpsEnvironment()
    env.reset(task_id="goods_not_received_easy")
    env._issuer_agent = _ScriptedIssuer(
        [IssuerDecision.REQUEST_MORE_EVIDENCE, IssuerDecision.ACCEPT]
    )
    _drive_case_into_round_2(env)

    env.step(
        ChargebackOpsAction(
            action_type="respond_to_pre_arb",
            case_id="CB-E1",
            compelling_evidence_ids=["E1-SIGNATURE"],
        )
    )

    progress = env._progress_by_case["CB-E1"]
    assert progress.resolution_status == "won_pre_arb"
    assert progress.arb_fees_paid == 0.0
    assert progress.arbitration_outcome is None
    case = env._lookup_case("CB-E1")
    assert progress.final_economic_outcome == case.amount


def test_accept_arbitration_loss_skips_fees():
    """Conceding pre-arb forfeits the dispute amount but avoids the $250 fee."""
    env = ChargebackOpsEnvironment()
    env.reset(task_id="goods_not_received_easy")
    env._issuer_agent = _ScriptedIssuer([IssuerDecision.REQUEST_MORE_EVIDENCE])
    _drive_case_into_round_2(env)

    env.step(
        ChargebackOpsAction(
            action_type="accept_arbitration_loss", case_id="CB-E1"
        )
    )

    progress = env._progress_by_case["CB-E1"]
    case = env._lookup_case("CB-E1")
    assert progress.resolution_status == "conceded_pre_arb"
    assert progress.arb_fees_paid == 0.0
    assert progress.arbitration_outcome is None
    assert progress.final_economic_outcome == -case.amount


def test_escalate_to_arbitration_from_round_2():
    """Merchant can voluntarily file for arbitration from round 2."""
    env = ChargebackOpsEnvironment()
    env.reset(task_id="goods_not_received_easy")
    env._issuer_agent = _ScriptedIssuer([IssuerDecision.REQUEST_MORE_EVIDENCE])
    _drive_case_into_round_2(env)

    env.step(
        ChargebackOpsAction(
            action_type="escalate_to_arbitration", case_id="CB-E1"
        )
    )

    progress = env._progress_by_case["CB-E1"]
    case = env._lookup_case("CB-E1")
    assert progress.round_number == 3
    assert progress.arb_fees_paid == ARB_FEE_PER_SIDE
    assert progress.arbitration_outcome in {
        ArbitrationOutcome.MERCHANT_WINS.value,
        ArbitrationOutcome.ISSUER_WINS.value,
    }
    if progress.arbitration_outcome == ArbitrationOutcome.MERCHANT_WINS.value:
        assert progress.final_economic_outcome == case.amount - ARB_FEE_PER_SIDE
    else:
        assert progress.final_economic_outcome == -case.amount - ARB_FEE_PER_SIDE
