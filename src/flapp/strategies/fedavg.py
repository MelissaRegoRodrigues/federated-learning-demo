import time

from flwr.server.strategy import FedAvg

from flapp.strategies.base import fail_if_clients_missing, partition_id_of, sort_by_partition_id
from flapp.strategies.logging_utils import log_round


class LoggingFedAvg(FedAvg):
    def __init__(self, *, results_path, num_clients, num_free_riders, exclude_clients=(), **kwargs):
        super().__init__(**kwargs)
        # Clients whose update is ignored. Used for the control with perfect detection.
        self.exclude_clients = set(exclude_clients)
        self.results_path = results_path
        self.num_clients = num_clients
        self.num_free_riders = num_free_riders
        self.last_aggregate_seconds = 0.0

    def aggregate_fit(self, server_round, results, failures):
        fail_if_clients_missing(server_round, failures)
        start_time = time.perf_counter()
        results = sort_by_partition_id(results)
        results = [result for result in results if partition_id_of(result) not in self.exclude_clients]
        result = super().aggregate_fit(server_round, results, failures)
        self.last_aggregate_seconds = time.perf_counter() - start_time
        return result

    def evaluate(self, server_round, parameters):
        result = super().evaluate(server_round, parameters)
        if result is None:
            return None

        loss, metrics = result
        extra = {
            "metadata": {
                "aggregate_seconds": self.last_aggregate_seconds,
                "server_seconds": self.last_aggregate_seconds,
            }
        }
        log_round(
            self.results_path,
            server_round,
            metrics,
            self.num_clients,
            self.num_free_riders,
            "fedavg",
            extra=extra,
        )
        return result
