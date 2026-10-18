import numpy as np
from flwr.common import FitIns, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import Strategy

from flapp.strategies.logging_utils import log_round
from flapp.task import SimpleCNN, get_params


def average_ndarrays(param_lists):
    result = []
    num_layers = len(param_lists[0])
    for layer_index in range(num_layers):
        layers = []
        for params in param_lists:
            layers.append(params[layer_index])
        result.append(np.mean(np.stack(layers), axis=0))
    return result


def weighted_average_ndarrays(updates, weights):
    first_client = next(iter(updates))
    num_layers = len(updates[first_client])
    result = []
    for layer_index in range(num_layers):
        weighted_sum = 0
        for client_id in updates:
            weighted_sum = weighted_sum + weights[client_id] * updates[client_id][layer_index]
        result.append(np.asarray(weighted_sum))
    return result


def partition_id_of(result):
    client, fit_res = result
    return fit_res.metrics["partition_id"]


def sort_by_partition_id(results):
    return sorted(results, key=partition_id_of)


def updates_by_partition(results):
    updates = {}
    for client, fit_res in sort_by_partition_id(results):
        updates[fit_res.metrics["partition_id"]] = parameters_to_ndarrays(fit_res.parameters)
    return updates


def sample_sizes_by_partition(results):
    sizes = {}
    for client, fit_res in sort_by_partition_id(results):
        sizes[fit_res.metrics["partition_id"]] = fit_res.num_examples
    return sizes


def scores_as_strings(scores):
    result = {}
    for client_id, value in scores.items():
        result[str(client_id)] = float(value)
    return result


def fit_config(server_round):
    return {"server_round": server_round}


def fail_if_clients_missing(server_round, failures):
    if failures:
        raise RuntimeError(
            f"Round {server_round}: {len(failures)} client(s) failed "
            f"(often Ray killing a worker for low memory). Aborting run."
        )


class CentralEvalStrategy(Strategy):
    def __init__(
        self,
        *,
        strategy_name,
        evaluate_fn,
        results_path,
        num_clients,
        num_free_riders,
        initial_parameters=None,
    ):
        self.strategy_name = strategy_name
        self.evaluate_fn = evaluate_fn
        self.results_path = results_path
        self.num_clients = num_clients
        self.num_free_riders = num_free_riders
        self.initial_parameters = initial_parameters
        self.current_global = None

    def initialize_parameters(self, client_manager):
        if self.initial_parameters is not None:
            return self.initial_parameters
        return ndarrays_to_parameters(get_params(SimpleCNN()))

    def configure_fit(self, server_round, parameters, client_manager):
        self.current_global = parameters_to_ndarrays(parameters)
        clients = client_manager.sample(num_clients=self.num_clients, min_num_clients=self.num_clients)
        instructions = []
        for client in clients:
            instructions.append((client, FitIns(parameters, fit_config(server_round))))
        return instructions

    def configure_evaluate(self, server_round, parameters, client_manager):
        return []

    def aggregate_evaluate(self, server_round, results, failures):
        return None, {}

    def evaluate(self, server_round, parameters):
        result = self.evaluate_fn(server_round, parameters_to_ndarrays(parameters), {})
        if result is None:
            return None

        loss, metrics = result
        if self.results_path is not None:
            log_round(
                self.results_path,
                server_round,
                metrics,
                self.num_clients,
                self.num_free_riders,
                self.strategy_name,
                extra=self._round_extra(),
            )
        return loss, metrics

    def _round_extra(self):
        return {}
