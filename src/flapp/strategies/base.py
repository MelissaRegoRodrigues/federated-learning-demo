"""Scaffolding shared by ShapleyWeightedStrategy and StackelbergStrategy:
both need a fresh SimpleCNN's initial parameters, both invite ALL clients
every round, both are evaluated only via a centralized test set (never
Flower's federated/client-side evaluate), and both need a place to log
their round's results. Only aggregate_fit() differs between them, so
that's the only method left abstract here.

FedAvg does NOT use this base — flwr.server.strategy.FedAvg already
implements all of this correctly and is used unmodified; only its
evaluate() gets a thin logging wrapper (see strategies/fedavg.py)."""
from abc import abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from flwr.common import FitIns, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import Strategy

from flapp.strategies.logging_utils import log_round
from flapp.task import SimpleCNN, get_params


def average_ndarrays(param_lists: list[list]) -> list:
    """Element-wise mean across clients — the same math FedAvg uses,
    exposed here as a plain function since Shapley needs it repeatedly
    (once per leave-one-out coalition) and Stackelberg needs it once
    (over whichever clients pass the reputation/price gates)."""
    return [np.mean(np.stack(layer), axis=0) for layer in zip(*param_lists)]


def weighted_average_ndarrays(updates: dict, weights: dict) -> list:
    """Shapley-only: unlike FedAvg's fixed 1/n, each client's weight here
    reflects its (shifted, normalized) Shapley value — callers are
    responsible for ensuring weights sum to 1 (see shapley.py)."""
    num_layers = len(next(iter(updates.values())))
    result = []
    for layer_idx in range(num_layers):
        weighted_sum = sum(weights[cid] * updates[cid][layer_idx] for cid in updates)
        result.append(np.asarray(weighted_sum))
    return result


class CentralEvalStrategy(Strategy):
    def __init__(
        self,
        *,
        strategy_name: str,
        evaluate_fn,
        results_path: Path | None,
        num_clients: int,
        num_free_riders: int,
    ) -> None:
        self.strategy_name = strategy_name
        self.evaluate_fn = evaluate_fn
        self.results_path = results_path
        self.num_clients = num_clients
        self.num_free_riders = num_free_riders

    def initialize_parameters(self, client_manager: ClientManager) -> Parameters | None:
        return ndarrays_to_parameters(get_params(SimpleCNN()))

    def configure_fit(self, server_round, parameters, client_manager: ClientManager):
        clients = client_manager.sample(
            num_clients=self.num_clients, min_num_clients=self.num_clients
        )
        return [(client, FitIns(parameters, {})) for client in clients]

    def configure_evaluate(self, server_round, parameters, client_manager: ClientManager):
        # Centralized evaluation only (see evaluate() below) — no
        # client-side federated evaluate round is needed for this thesis.
        return []

    def aggregate_evaluate(self, server_round, results, failures):
        return None, {}

    def evaluate(self, server_round: int, parameters: Parameters):
        result = self.evaluate_fn(server_round, parameters_to_ndarrays(parameters), {})
        if result is None:
            return None
        loss, metrics = result
        if self.results_path is not None:
            log_round(
                self.results_path,
                server_round,
                metrics.get("accuracy"),
                self.num_clients,
                self.num_free_riders,
                self.strategy_name,
                extra=self._round_extra(),
            )
        return loss, metrics

    def _round_extra(self) -> dict[str, Any]:
        """Subclasses override: aggregate_fit() runs before evaluate() each
        round, so it can stash this round's client_scores/
        detected_free_riders/metadata on self for evaluate() to log."""
        return {}

    @abstractmethod
    def aggregate_fit(self, server_round, results, failures):
        ...
