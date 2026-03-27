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
