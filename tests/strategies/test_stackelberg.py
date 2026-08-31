import numpy as np
import pytest
from flwr.common import FitRes, Status, Code, ndarrays_to_parameters

from flapp.strategies.stackelberg import (
    StackelbergStrategy,
    _coordinate_median,
    _cosine_similarity,
)


class _FakeClientProxy:
    def __init__(self, cid: str):
        self.cid = cid


def _fit_res(ndarrays, partition_id):
    return FitRes(
        status=Status(code=Code.OK, message=""),
        parameters=ndarrays_to_parameters(ndarrays),
        num_examples=10,
        metrics={"partition_id": partition_id},
    )


def _make_strategy():
    return StackelbergStrategy(
        evaluate_fn=lambda *a: (0.0, {"accuracy": 0.5}),
        results_path=None,
        num_clients=3,
        num_free_riders=1,
    )


def test_coordinate_median_is_elementwise_median():
    a = [np.array([1.0, 1.0])]
    b = [np.array([2.0, 2.0])]
    c = [np.array([9.0, 9.0])]
    median = _coordinate_median([a, b, c])
    np.testing.assert_allclose(median[0], [2.0, 2.0])


def test_cosine_similarity_identical_vectors_is_one():
    a = [np.array([1.0, 0.0])]
    assert _cosine_similarity(a, a) == pytest.approx(1.0)


def test_round_one_has_no_detections_but_seeds_reputation():
    strategy = _make_strategy()
    results = [
        (_FakeClientProxy("0"), _fit_res([np.array([1.0, 1.0])], 0)),
        (_FakeClientProxy("1"), _fit_res([np.array([1.0, 1.0])], 1)),
        (_FakeClientProxy("2"), _fit_res([np.array([50.0, 50.0])], 2)),
    ]
    strategy.aggregate_fit(1, results, [])
    extra = strategy._round_extra()
    assert extra["detected_free_riders"] == []
    assert set(strategy.reputation.keys()) == {0, 1, 2}


def test_free_rider_reputation_drops_below_honest_over_rounds():
    strategy = _make_strategy()
    for round_num in range(1, 4):
        results = [
            (_FakeClientProxy("0"), _fit_res([np.array([1.0, 1.0])], 0)),
            (_FakeClientProxy("1"), _fit_res([np.array([1.0, 1.0])], 1)),
            (_FakeClientProxy("2"), _fit_res([np.array([-50.0, 50.0])], 2)),  # consistently divergent direction
        ]
        strategy.aggregate_fit(round_num, results, [])
    assert strategy.reputation[2] < strategy.reputation[0]
    assert strategy.reputation[2] < strategy.reputation[1]


def test_aggregate_fit_returns_valid_parameters():
    strategy = _make_strategy()
    results = [
        (_FakeClientProxy("0"), _fit_res([np.array([1.0, 2.0])], 0)),
        (_FakeClientProxy("1"), _fit_res([np.array([3.0, 4.0])], 1)),
    ]
    parameters, metrics = strategy.aggregate_fit(1, results, [])
    assert parameters is not None
    assert metrics == {}
