import pytest

from flapp.aggregate import (
    aggregate_cumulative_reward_distance,
    aggregate_detection_rate,
    aggregate_metric,
    mean_std_by_round,
    results_filename,
    reward_shares,
)


def test_results_filename_embeds_strategy_and_seed():
    assert results_filename("fedavg", 42) == "fedavg_seed42.json"


def test_mean_std_by_round_averages_across_runs():
    results_list = [
        {"rounds": [{"round": 1, "accuracy": 0.2}, {"round": 2, "accuracy": 0.4}]},
        {"rounds": [{"round": 1, "accuracy": 0.4}, {"round": 2, "accuracy": 0.6}]},
    ]
    rounds, means, stds = mean_std_by_round(results_list, lambda row, run: row["accuracy"])
    assert rounds == [1, 2]
    assert means == pytest.approx([0.3, 0.5])
    assert stds[0] > 0.0


def test_mean_std_by_round_drops_rounds_missing_from_any_run():
    results_list = [
        {"rounds": [{"round": 1, "accuracy": 0.2}, {"round": 2, "accuracy": 0.4}]},
        {"rounds": [{"round": 1, "accuracy": 0.4}]},
    ]
    rounds, means, stds = mean_std_by_round(results_list, lambda row, run: row["accuracy"])
    assert rounds == [1]
    assert means == pytest.approx([0.3])


def test_aggregate_metric_reads_named_metric():
    results_list = [
        {"rounds": [{"round": 1, "f1": 0.5}]},
        {"rounds": [{"round": 1, "f1": 0.7}]},
    ]
    rounds, means, _ = aggregate_metric(results_list, "f1")
    assert rounds == [1]
    assert means == [0.6]


def test_aggregate_detection_rate_divides_detected_by_free_riders():
    results_list = [
        {
            "num_free_riders": 2,
            "rounds": [{"round": 1, "detected_free_riders": [0, 1]}],
        },
        {
            "num_free_riders": 2,
            "rounds": [{"round": 1, "detected_free_riders": [0]}],
        },
    ]
    rounds, means, _ = aggregate_detection_rate(results_list)
    assert rounds == [1]
    assert means == [0.75]


def test_aggregate_detection_rate_ignores_honest_clients_flagged_by_mistake():
    results_list = [
        {
            "num_free_riders": 2,
            "rounds": [{"round": 1, "detected_free_riders": [0, 5]}],
        },
    ]
    rounds, means, _ = aggregate_detection_rate(results_list)
    assert rounds == [1]
    assert means == [0.5]


def test_aggregate_detection_rate_skips_runs_with_no_free_riders():
    results_list = [{"num_free_riders": 0, "rounds": [{"round": 1, "detected_free_riders": []}]}]
    rounds, means, _ = aggregate_detection_rate(results_list)
    assert rounds == []
    assert means == []


def test_mean_std_by_round_ignores_run_without_values():
    results_list = [
        {"rounds": [{"round": 1, "accuracy": 0.2}, {"round": 2, "accuracy": 0.4}]},
        {"rounds": [{"round": 1, "accuracy": 0.4}, {"round": 2, "accuracy": 0.6}]},
        {"rounds": [{"round": 1}, {"round": 2}]},
    ]
    with pytest.warns(UserWarning):
        rounds, means, stds = mean_std_by_round(results_list, lambda row, run: row.get("accuracy"))
    assert rounds == [1, 2]
    assert means == pytest.approx([0.3, 0.5])


def test_reward_shares_normalize_the_logged_rewards():
    row = {"metadata": {"rewards": {"0": 0.0, "1": 1.0, "2": 3.0}}}
    assert reward_shares(row) == {0: 0.0, 1: 0.25, 2: 0.75}


def test_reward_shares_are_zero_when_nobody_was_paid():
    row = {"metadata": {"rewards": {"0": 0.0, "1": 0.0}}}
    assert reward_shares(row) == {0: 0.0, 1: 0.0}


def test_cumulative_reward_distance_is_zero_when_rewards_follow_the_reference():
    rounds = [{"round": 0, "accuracy": 0.1}]
    for round_number in range(1, 4):
        rounds.append({"round": round_number, "metadata": {"rewards": {"0": 0.0, "1": 1.0, "2": 3.0}}})
    run = {"num_free_riders": 1, "rounds": rounds}
    reference = {0: 0.0, 1: 0.25, 2: 0.75}
    result_rounds, means, stds = aggregate_cumulative_reward_distance([run, run], [reference, reference])
    assert result_rounds == [1, 2, 3]
    assert means[-1] == pytest.approx(0.0)


def test_load_price_schedule_reads_the_price_of_each_round(tmp_path):
    import json

    from flapp.aggregate import load_price_schedule

    run = {
        "rounds": [
            {"round": 0, "accuracy": 0.1},
            {"round": 1, "metadata": {"price": 1.2}},
            {"round": 2, "metadata": {"price": 1.3}},
        ]
    }
    path = tmp_path / "stackelberg_seed42.json"
    path.write_text(json.dumps(run))
    assert load_price_schedule(path) == {1: 1.2, 2: 1.3}
