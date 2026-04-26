from runners.inference import run_inference
from core.models import ChargebackOpsAction
from server.app import baseline, grader, root, tasks
from server.chargeback_ops_environment import ChargebackOpsEnvironment


def test_tasks_endpoint_payload():
    payload = tasks()
    assert len(payload.tasks) >= 3
    assert "properties" in payload.action_schema


def test_root_endpoint_payload():
    response = root()
    assert response.status_code == 200
    assert b"ChargebackOps" in response.body
    assert b"tasks_url" in response.body
    assert b"demo_url" in response.body
    assert b"huggingface.co/spaces" in response.body
    assert b"interactive_demo_url" in response.body


def test_baseline_endpoint_works_without_api_key(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("API_BASE_URL", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    payload = baseline()
    assert payload.mode == "heuristic_fallback"
    assert len(payload.task_results) >= 3


def test_inference_script_falls_back_without_hf_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("API_BASE_URL", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)
    payload = run_inference()
    assert payload.mode == "heuristic_fallback"
    assert len(payload.task_results) >= 3


def test_grader_endpoint_after_completed_episode():
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
            action_type="query_system",
            case_id="CB-E1",
            system_name="support",
        )
    )
    env.step(
        ChargebackOpsAction(
            action_type="add_evidence",
            case_id="CB-E1",
            evidence_ids=[
                "E1-ORDER-CONF",
                "E1-DELIVERY-SCAN",
                "E1-SIGNATURE",
                "E1-SUPPORT-ACK",
            ],
        )
    )
    env.step(
        ChargebackOpsAction(
            action_type="set_strategy",
            case_id="CB-E1",
            strategy="contest",
        )
    )
    final_obs = env.step(
        ChargebackOpsAction(
            action_type="submit_representment",
            case_id="CB-E1",
        )
    )

    assert final_obs.grader_report is not None
    payload = grader(final_obs.grader_report.episode_id)
    assert payload["episode_id"] == final_obs.grader_report.episode_id
    assert 0.0 <= payload["normalized_score"] <= 1.0
