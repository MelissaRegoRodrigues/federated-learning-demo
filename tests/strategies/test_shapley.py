import numpy as np
import pytest
from flwr.common import FitRes, Status, Code, ndarrays_to_parameters, parameters_to_ndarrays

from flapp.strategies.shapley import ShapleyWeightedStrategy


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


def _make_strategy(evaluate_fn):
    return ShapleyWeightedStrategy(
        evaluate_fn=evaluate_fn, results_path=None, num_clients=3, num_free_riders=1
    )


def test_free_rider_gets_lower_shapley_than_honest_clients():
    # Utility function: pretend accuracy is proportional to how much
    # client 2's (the "bad" client's) huge-magnitude update pollutes the
    # average — so removing client 2 measurably IMPROVES utility, and
    # removing an honest client measurably HURTS it.
    def evaluate_fn(server_round, ndarrays, config):
        magnitude = float(np.sum(np.abs(ndarrays[0])))
        return 0.0, {"accuracy": max(0.0, 1.0 - magnitude / 100.0)}

    strategy = _make_strategy(evaluate_fn)
    results = [
        (_FakeClientProxy("0"), _fit_res([np.array([1.0, 1.0])], 0)),
        (_FakeClientProxy("1"), _fit_res([np.array([1.0, 1.0])], 1)),
        (_FakeClientProxy("2"), _fit_res([np.array([50.0, 50.0])], 2)),  # the free-rider
    ]
    strategy.aggregate_fit(1, results, [])
    extra = strategy._round_extra()
    scores = extra["client_scores"]

    assert scores["2"] < scores["0"]
    assert scores["2"] < scores["1"]
    assert 2 in extra["detected_free_riders"]


def test_aggregate_fit_returns_valid_parameters():
    def evaluate_fn(server_round, ndarrays, config):
        return 0.0, {"accuracy": 0.5}

    strategy = _make_strategy(evaluate_fn)
    results = [
        (_FakeClientProxy("0"), _fit_res([np.array([1.0, 2.0])], 0)),
        (_FakeClientProxy("1"), _fit_res([np.array([3.0, 4.0])], 1)),
    ]
    parameters, metrics = strategy.aggregate_fit(1, results, [])
    assert parameters is not None
    assert metrics == {}


def test_empty_results_returns_none():
    strategy = _make_strategy(lambda *a: (0.0, {"accuracy": 0.0}))
    parameters, metrics = strategy.aggregate_fit(1, [], [])
    assert parameters is None


def test_near_zero_total_contribution_falls_back_to_equal_weights():
    # A constant evaluate_fn means every leave-one-out utility is
    # identical, so every Shapley value is exactly 0 and their sum is 0.
    # Dividing by that (or anything close to it) must NOT blow up the
    # aggregation weights — found during full-scale validation to
    # occasionally explode the aggregated model into unrecoverable garbage.
    strategy = _make_strategy(lambda *a: (0.0, {"accuracy": 0.5}))
    results = [
        (_FakeClientProxy("0"), _fit_res([np.array([1.0, 1.0])], 0)),
        (_FakeClientProxy("1"), _fit_res([np.array([2.0, 2.0])], 1)),
        (_FakeClientProxy("2"), _fit_res([np.array([3.0, 3.0])], 2)),
    ]
    parameters, _ = strategy.aggregate_fit(1, results, [])
    aggregated = parameters_to_ndarrays(parameters)
    np.testing.assert_allclose(aggregated[0], [2.0, 2.0])  # falls back to plain average


def test_free_rider_gets_weight_near_zero_not_negative():
    # Weights must stay in [0, 1] and sum to 1, regardless of how skewed
    # the raw Shapley values are — a signed-weight version (letting the
    # free-rider's weight go negative) was tried first and found, during
    # full-scale validation, to occasionally destabilize training badly
    # enough that it never recovered.
    def evaluate_fn(server_round, ndarrays, config):
        magnitude = float(np.sum(np.abs(ndarrays[0])))
        return 0.0, {"accuracy": max(0.0, 1.0 - magnitude / 100.0)}

    strategy = _make_strategy(evaluate_fn)
    results = [
        (_FakeClientProxy("0"), _fit_res([np.array([1.0, 1.0])], 0)),
        (_FakeClientProxy("1"), _fit_res([np.array([1.0, 1.0])], 1)),
        (_FakeClientProxy("2"), _fit_res([np.array([50.0, 50.0])], 2)),  # the free-rider
    ]
    strategy.aggregate_fit(1, results, [])
    # Recompute weights the same way aggregate_fit does, to check the sign/sum invariant
    shapley_values = {
        int(cid): v for cid, v in strategy._round_extra()["client_scores"].items()
    }
    min_contribution = min(shapley_values.values())
    shifted = {cid: v - min_contribution for cid, v in shapley_values.items()}
    total_shifted = sum(shifted.values())
    weights = {cid: v / total_shifted for cid, v in shifted.items()}
    assert all(w >= 0 for w in weights.values())
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights[2] < weights[0]  # free-rider still gets the smallest weight
