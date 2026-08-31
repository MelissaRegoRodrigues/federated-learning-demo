import json

import numpy as np
from flwr.common import FitRes, Status, Code, ndarrays_to_parameters, parameters_to_ndarrays

from flapp.strategies.base import CentralEvalStrategy, average_ndarrays, weighted_average_ndarrays


class _FakeClientProxy:
    def __init__(self, cid: str):
        self.cid = cid


def _fit_res(ndarrays) -> FitRes:
    return FitRes(
        status=Status(code=Code.OK, message=""),
        parameters=ndarrays_to_parameters(ndarrays),
        num_examples=10,
        metrics={},
    )


def test_average_ndarrays_elementwise_mean():
    a = [np.array([1.0, 1.0])]
    b = [np.array([3.0, 3.0])]
    result = average_ndarrays([a, b])
    np.testing.assert_allclose(result[0], [2.0, 2.0])


def test_weighted_average_ndarrays():
    updates = {0: [np.array([1.0])], 1: [np.array([3.0])]}
    weights = {0: 0.25, 1: 0.75}
    result = weighted_average_ndarrays(updates, weights)
    np.testing.assert_allclose(result[0], [2.5])


class _ConcreteStrategy(CentralEvalStrategy):
    """Minimal concrete subclass just to exercise the shared base."""

    def aggregate_fit(self, server_round, results, failures):
        updates = {
            fit_res.metrics["partition_id"]: parameters_to_ndarrays(fit_res.parameters)
            for client, fit_res in results
        }
        aggregated = average_ndarrays(list(updates.values()))
        self._extra = {"client_scores": {}, "detected_free_riders": [], "metadata": {}}
        return ndarrays_to_parameters(aggregated), {}

    def _round_extra(self):
        return getattr(self, "_extra", {})


def test_evaluate_calls_evaluate_fn_and_logs(tmp_path):
    calls = []

    def fake_evaluate_fn(server_round, ndarrays, config):
        calls.append(server_round)
        return 0.1, {"accuracy": 0.42}

    results_path = tmp_path / "test.json"
    strategy = _ConcreteStrategy(
        strategy_name="test", evaluate_fn=fake_evaluate_fn, results_path=results_path,
        num_clients=2, num_free_riders=0,
    )
    params = strategy.initialize_parameters(client_manager=None)
    loss, metrics = strategy.evaluate(1, params)

    assert calls == [1]
    assert loss == 0.1
    assert metrics["accuracy"] == 0.42
    data = json.loads(results_path.read_text())
    assert data["rounds"][0]["accuracy"] == 0.42


def test_configure_evaluate_returns_empty_and_aggregate_evaluate_is_noop():
    strategy = _ConcreteStrategy(
        strategy_name="test", evaluate_fn=lambda *a: None, results_path=None,
        num_clients=2, num_free_riders=0,
    )
    assert strategy.configure_evaluate(1, None, client_manager=None) == []
    assert strategy.aggregate_evaluate(1, [], []) == (None, {})
