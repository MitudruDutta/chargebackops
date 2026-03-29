from evaluation.agent_brutal_audit import aggregate_results, run_episode


def test_heuristic_beats_bad_on_generated_suite():
    heuristic_results = []
    bad_results = []

    for difficulty in ("easy", "medium", "hard"):
        for seed in (11, 12):
            heuristic_results.append(
                run_episode(f"generated_{difficulty}_s{seed}", policy="heuristic")
            )
            bad_results.append(
                run_episode(f"generated_{difficulty}_s{seed}", policy="bad")
            )

    heuristic_avg = aggregate_results(heuristic_results)["avg_score"]
    bad_avg = aggregate_results(bad_results)["avg_score"]

    assert heuristic_avg is not None
    assert bad_avg is not None
    assert heuristic_avg > bad_avg
    assert all(0.0 <= result["score"] <= 1.0 for result in heuristic_results + bad_results)


def test_data_directory_is_ignored():
    with open(".gitignore", encoding="utf-8") as handle:
        gitignore = handle.read()
    with open(".dockerignore", encoding="utf-8") as handle:
        dockerignore = handle.read()

    assert "data/" in gitignore
    assert "data/" in dockerignore
