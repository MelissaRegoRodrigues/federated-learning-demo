import math

import numpy as np
import pytest
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays

from flapp.strategies.stackelberg import StackelbergStrategy, compatibility, entro


class FakeClientProxy:
    def __init__(self, cid):
        self.cid = cid


def fit_res(ndarrays, partition_id, num_examples):
    return FitRes(
        status=Status(code=Code.OK, message=""),
        parameters=ndarrays_to_parameters(ndarrays),
        num_examples=num_examples,
        metrics={"partition_id": partition_id},
    )


def make_strategy():
    strategy = StackelbergStrategy(
        evaluate_fn=lambda *a: (0.0, {"accuracy": 0.5}),
        results_path=None,
        num_clients=4,
        num_free_riders=1,
        seed=42,
        dataset_sizes={0: 1000, 1: 1000, 2: 1000, 3: 1000},
        num_rounds=5,
    )
    strategy.current_global = [np.zeros(3)]
    strategy.current_price = 2.0
    return strategy


def results_with_one_free_rider(order=(0, 1, 2, 3)):
    models = {
        0: [np.zeros(3)],
        1: [np.array([1.0, 1.0, 0.0])],
        2: [np.array([1.0, 0.9, 0.1])],
        3: [np.array([0.9, 1.0, 0.0])],
    }
    sizes = {0: 500, 1: 100, 2: 300, 3: 600}
    results = []
    for partition_id in order:
        results.append((FakeClientProxy(str(partition_id)), fit_res(models[partition_id], partition_id, sizes[partition_id])))
    return results


def test_entro_of_identical_gradients_is_maximal():
    gradient = np.array([1.0, 2.0])
    expected = 0.6 * math.log(2) + 0.4 * 1.0
    assert entro(gradient, gradient) == pytest.approx(expected)


def test_entro_with_a_zero_gradient_is_zero():
    assert entro(np.zeros(2), np.array([1.0, 2.0])) == 0.0
    assert entro(np.zeros(2), np.zeros(2)) == 0.0


def test_compatibility_is_the_mean_entro_with_the_others():
    gradients = {0: np.array([1.0, 0.0]), 1: np.array([1.0, 0.0]), 2: np.zeros(2)}
    values = compatibility(gradients)
    same = 0.6 * math.log(2) + 0.4
    assert values[0] == pytest.approx(same / 2)
    assert values[2] == 0.0


def test_free_rider_is_barred_and_not_paid():
    strategy = make_strategy()
    strategy.aggregate_fit(1, results_with_one_free_rider(), [])
    extra = strategy._round_extra()
    assert extra["detected_free_riders"] == [0]
    assert extra["metadata"]["rewards"]["0"] == 0.0
    assert extra["metadata"]["rewards"]["1"] == pytest.approx(2.0 * 100)


def test_aggregation_weights_included_clients_by_their_data():
    strategy = make_strategy()
    parameters, _ = strategy.aggregate_fit(1, results_with_one_free_rider(), [])
    expected = (100 * np.array([1.0, 1.0, 0.0]) + 300 * np.array([1.0, 0.9, 0.1]) + 600 * np.array([0.9, 1.0, 0.0])) / 1000
    np.testing.assert_allclose(parameters_to_ndarrays(parameters)[0], expected)


def test_barred_client_stays_out_in_later_rounds():
    strategy = make_strategy()
    strategy.aggregate_fit(1, results_with_one_free_rider(), [])
    strategy.aggregate_fit(2, results_with_one_free_rider(), [])
    assert 0 in strategy.barred
    assert 0 not in strategy.active_client_params()


def test_aggregate_fit_does_not_depend_on_the_order_clients_answer():
    first = make_strategy()
    second = make_strategy()
    a, _ = first.aggregate_fit(1, results_with_one_free_rider((0, 1, 2, 3)), [])
    b, _ = second.aggregate_fit(1, results_with_one_free_rider((3, 1, 0, 2)), [])
    np.testing.assert_array_equal(parameters_to_ndarrays(a)[0], parameters_to_ndarrays(b)[0])


def test_strategy_calibrates_a_positive_time_budget():
    strategy = make_strategy()
    assert strategy.total_time > 0
    assert strategy.price_for_round(1) > 0
