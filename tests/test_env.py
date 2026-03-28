from case_generator import generate_task
from models import ChargebackOpsAction
from server.chargeback_ops_environment import ChargebackOpsEnvironment


def test_reset_returns_task_observation():
    env = ChargebackOpsEnvironment()
    obs = env.reset(task_id="goods_not_received_easy")
    assert obs.task_id == "goods_not_received_easy"
    assert obs.steps_remaining == 10
    assert len(obs.queue) == 1


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
