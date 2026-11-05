import numpy as np
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays

from flapp.strategies.fedavg import LoggingFedAvg


class FakeClientProxy:
    def __init__(self, cid):
        self.cid = cid


def fit_res(value, partition_id, num_examples):
    return FitRes(
        status=Status(code=Code.OK, message=""),
        parameters=ndarrays_to_parameters([np.array([value])]),
        num_examples=num_examples,
        metrics={"partition_id": partition_id},
    )


def make_results():
    return [
        (FakeClientProxy("0"), fit_res(0.0, 0, 30)),
        (FakeClientProxy("1"), fit_res(10.0, 1, 10)),
        (FakeClientProxy("2"), fit_res(20.0, 2, 10)),
    ]


def make_strategy(exclude_clients=()):
    return LoggingFedAvg(
        results_path=None,
        num_clients=3,
        num_free_riders=1,
        exclude_clients=exclude_clients,
    )


def test_fedavg_weights_every_client_by_data_size_by_default():
    parameters, _ = make_strategy().aggregate_fit(1, make_results(), [])
    np.testing.assert_allclose(parameters_to_ndarrays(parameters)[0], [6.0])


def test_fedavg_ignores_the_excluded_clients():
    parameters, _ = make_strategy(exclude_clients=(0,)).aggregate_fit(1, make_results(), [])
    np.testing.assert_allclose(parameters_to_ndarrays(parameters)[0], [15.0])
