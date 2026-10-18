import json

from flapp.strategies.logging_utils import log_round


def test_log_round_creates_file_with_schema(tmp_path):
    results_path = tmp_path / "shapley.json"
    log_round(
        results_path, round_num=1, metrics={"accuracy": 0.81}, num_clients=10, num_free_riders=3,
        strategy_name="shapley",
        extra={"client_scores": {"0": 0.03}, "detected_free_riders": [2], "metadata": {}},
    )
    data = json.loads(results_path.read_text())
    assert data["strategy"] == "shapley"
    assert data["num_clients"] == 10
    assert data["num_free_riders"] == 3
    assert len(data["rounds"]) == 1
    assert data["rounds"][0]["round"] == 1
    assert data["rounds"][0]["accuracy"] == 0.81
    assert data["rounds"][0]["detected_free_riders"] == [2]


def test_log_round_appends_across_calls(tmp_path):
    results_path = tmp_path / "fedavg.json"
    log_round(results_path, 1, {"accuracy": 0.7}, 5, 0, "fedavg", extra={})
    log_round(results_path, 2, {"accuracy": 0.75}, 5, 0, "fedavg", extra={})
    data = json.loads(results_path.read_text())
    assert [r["round"] for r in data["rounds"]] == [1, 2]
