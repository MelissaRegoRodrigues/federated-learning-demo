"""FedAvg baseline — Flower's real, unmodified aggregation math. Only
evaluate() gets a thin wrapper so its results land in the same JSON
schema as Shapley/Stackelberg."""
from pathlib import Path

from flwr.server.strategy import FedAvg

from flapp.strategies.logging_utils import log_round


class LoggingFedAvg(FedAvg):
    def __init__(self, *, results_path: Path, num_clients: int, num_free_riders: int, **kwargs):
        super().__init__(**kwargs)
        self.results_path = results_path
        self.num_clients = num_clients
        self.num_free_riders = num_free_riders

    def evaluate(self, server_round, parameters):
        result = super().evaluate(server_round, parameters)
        if result is not None:
            loss, metrics = result
            log_round(
                self.results_path,
                server_round,
                metrics.get("accuracy"),
                self.num_clients,
                self.num_free_riders,
                "fedavg",
                extra={},
            )
        return result
