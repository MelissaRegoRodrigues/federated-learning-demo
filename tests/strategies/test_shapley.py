import numpy as np
import pytest
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays

from flapp.strategies.shapley import (
    ShapleyWeightedStrategy,
    coalition_model,
    contribution_weights,
    lowest_contributions,
    merged_contributions,
)


class FakeClientProxy:
    def __init__(self, cid):
        self.cid = cid


def fit_res(ndarrays, partition_id, num_examples=10):
    return FitRes(
        status=Status(code=Code.OK, message=""),
        parameters=ndarrays_to_parameters(ndarrays),
        num_examples=num_examples,
        metrics={"partition_id": partition_id},
    )


def distance_utility(server_round, ndarrays, config):
    target = np.array([1.0, 1.0])
    distance = float(np.linalg.norm(ndarrays[0] - target))
    return 0.0, {"accuracy": 1.0 / (1.0 + distance)}


def make_strategy(voting_rate=0.2):
    return ShapleyWeightedStrategy(
        evaluate_fn=distance_utility,
        shapley_eval_fn=distance_utility,
        results_path=None,
        num_clients=5,
        num_free_riders=1,
        voting_rate=voting_rate,
    )


def five_clients_with_one_free_rider():
    results = [(FakeClientProxy("0"), fit_res([np.array([0.0, 0.0])], 0))]
    for partition_id in range(1, 5):
        results.append((FakeClientProxy(str(partition_id)), fit_res([np.array([1.0, 1.0])], partition_id)))
    return results


def test_coalition_model_is_weighted_by_data_size():
    models = {0: [np.array([0.0])], 1: [np.array([10.0])]}
    sizes = {0: 30, 1: 10}
    result = coalition_model(models, sizes, [0, 1])
    np.testing.assert_allclose(result[0], [2.5])


def test_lowest_contributions_flags_twenty_percent():
    values = {0: 0.5, 1: -0.1, 2: 0.3, 3: 0.0, 4: 0.9, 5: 0.2, 6: 0.4, 7: 0.6, 8: 0.7, 9: 0.8}
    assert lowest_contributions(values, 0.2) == [1, 3]


def test_merged_contributions_follow_majority_voting_with_one_validator():
    contributions = merged_contributions([0, 1, 2], flagged=[1])
    assert contributions == {0: 1, 1: 0, 2: 1}
    assert contribution_weights(contributions) == {0: 0.5, 1: 0.0, 2: 0.5}


def test_free_rider_gets_the_lowest_contribution_and_is_flagged():
    strategy = make_strategy()
    strategy.aggregate_fit(1, five_clients_with_one_free_rider(), [])
    extra = strategy._round_extra()
    scores = {int(k): v for k, v in extra["client_scores"].items()}
    assert min(scores, key=scores.get) == 0
    assert extra["detected_free_riders"] == [0]
    assert extra["metadata"]["rewards"]["0"] == 0.0


def test_aggregation_averages_only_the_clients_not_flagged():
    strategy = make_strategy()
    parameters, _ = strategy.aggregate_fit(1, five_clients_with_one_free_rider(), [])
    np.testing.assert_allclose(parameters_to_ndarrays(parameters)[0], [1.0, 1.0])


def test_empty_results_returns_none():
    strategy = make_strategy()
    assert strategy.aggregate_fit(1, [], []) == (None, {})


def test_aggregate_fit_does_not_depend_on_the_order_clients_answer():
    rng = np.random.default_rng(0)
    updates = []
    for partition_id in range(5):
        updates.append([rng.normal(size=(2,))])

    outputs = []
    for order in [[0, 1, 2, 3, 4], [4, 2, 0, 3, 1]]:
        strategy = make_strategy()
        results = []
        for partition_id in order:
            results.append((FakeClientProxy(str(partition_id)), fit_res(updates[partition_id], partition_id)))
        parameters, _ = strategy.aggregate_fit(1, results, [])
        outputs.append((parameters_to_ndarrays(parameters), strategy._round_extra()["detected_free_riders"]))

    np.testing.assert_array_equal(outputs[0][0][0], outputs[1][0][0])
    assert outputs[0][1] == outputs[1][1]


def test_missing_eval_fn_warns():
    with pytest.warns(UserWarning):
        ShapleyWeightedStrategy(evaluate_fn=distance_utility, results_path=None, num_clients=2, num_free_riders=0)
