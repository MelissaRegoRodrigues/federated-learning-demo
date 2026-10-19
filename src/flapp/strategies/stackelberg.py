import math
import time

import numpy as np
from flwr.common import FitIns, ndarrays_to_parameters, parameters_to_ndarrays

from flapp import economics
from flapp.strategies.base import (
    CentralEvalStrategy,
    fail_if_clients_missing,
    sample_sizes_by_partition,
    scores_as_strings,
    updates_by_partition,
    weighted_average_ndarrays,
)

ENTROPY_WEIGHT = 0.6
COMPATIBILITY_THRESHOLD = 0.01
MODEL_SIZE = 18406


def flatten(params):
    pieces = []
    for layer in params:
        pieces.append(np.asarray(layer, dtype=np.float64).ravel())
    return np.concatenate(pieces)


def gradient_of(global_weights, client_weights):
    return flatten(global_weights) - flatten(client_weights)


def magnitude_weight(gradient_i, gradient_j):
    energy_i = float(np.dot(gradient_i, gradient_i))
    energy_j = float(np.dot(gradient_j, gradient_j))
    return energy_i / (energy_i + energy_j)


def plogp(value):
    if value == 0:
        return 0.0
    return value * math.log(value)


def cosine(gradient_i, gradient_j):
    norms = np.linalg.norm(gradient_i) * np.linalg.norm(gradient_j)
    if norms == 0:
        return 0.0
    return float(np.dot(gradient_i, gradient_j) / norms)


def entro(gradient_i, gradient_j, alpha=ENTROPY_WEIGHT):
    if not np.any(gradient_i) and not np.any(gradient_j):
        return 0.0
    p_i = magnitude_weight(gradient_i, gradient_j)
    p_j = magnitude_weight(gradient_j, gradient_i)
    entropy = -(plogp(p_i) + plogp(p_j))
    return alpha * entropy + (1 - alpha) * cosine(gradient_i, gradient_j)


def compatibility(gradients):
    client_ids = list(gradients.keys())
    values = {}
    for client_id in client_ids:
        others = [other for other in client_ids if other != client_id]
        if len(others) == 0:
            values[client_id] = 0.0
            continue
        total = 0.0
        for other in others:
            total += entro(gradients[client_id], gradients[other])
        values[client_id] = total / len(others)
    return values


class StackelbergStrategy(CentralEvalStrategy):
    def __init__(
        self,
        *,
        seed=0,
        dataset_sizes=None,
        num_rounds=50,
        compatibility_threshold=COMPATIBILITY_THRESHOLD,
        model_size=MODEL_SIZE,
        **kwargs,
    ):
        super().__init__(strategy_name="stackelberg", **kwargs)
        self.seed = seed
        self.compatibility_threshold = compatibility_threshold
        self.model_size = model_size
        self.barred = set()
        self.last_extra = {}
        self.current_price = 0.0

        if dataset_sizes is None:
            dataset_sizes = {}
        self.dataset_sizes = dataset_sizes
        self.client_params = {}
        for client_id in dataset_sizes:
            self.client_params[client_id] = economics.client_parameters(seed, client_id)
        self.total_time = 0.0
        self.calibration_seconds = 0.0
        self.price_seconds = 0.0
        if self.client_params:
            start_time = time.perf_counter()
            self.total_time = economics.calibrate_total_time(
                num_rounds, self.client_params, self.dataset_sizes, self.model_size
            )
            self.calibration_seconds = time.perf_counter() - start_time

    def active_client_params(self):
        active = {}
        for client_id, params in self.client_params.items():
            if client_id not in self.barred:
                active[client_id] = params
        return active

    def price_for_round(self, server_round):
        active = self.active_client_params()
        if not active:
            return 0.0
        return economics.server_price(server_round, self.total_time, active, self.dataset_sizes, self.model_size)

    def configure_fit(self, server_round, parameters, client_manager):
        self.current_global = parameters_to_ndarrays(parameters)
        start_time = time.perf_counter()
        self.current_price = self.price_for_round(server_round)
        self.price_seconds = time.perf_counter() - start_time
        barred_text = ",".join(str(client_id) for client_id in sorted(self.barred))
        config = {
            "server_round": server_round,
            "price": self.current_price,
            "model_size": self.model_size,
            "barred": barred_text,
        }
        clients = client_manager.sample(num_clients=self.num_clients, min_num_clients=self.num_clients)
        instructions = []
        for client in clients:
            instructions.append((client, FitIns(parameters, config)))
        return instructions

    def aggregate_fit(self, server_round, results, failures):
        fail_if_clients_missing(server_round, failures)
        if not results:
            return None, {}
        start_time = time.perf_counter()

        models = updates_by_partition(results)
        datasizes = sample_sizes_by_partition(results)

        participants = []
        for client_id, datasize in datasizes.items():
            if datasize > 0 and client_id not in self.barred:
                participants.append(client_id)

        gradients = {}
        for client_id in participants:
            gradients[client_id] = gradient_of(self.current_global, models[client_id])
        compatibilities = compatibility(gradients)

        for client_id, value in compatibilities.items():
            if value < self.compatibility_threshold:
                self.barred.add(client_id)

        included = [client_id for client_id in participants if client_id not in self.barred]
        payments = {}
        for client_id in models:
            if client_id in included:
                payments[client_id] = economics.DISCOUNT ** (server_round - 1) * self.current_price * datasizes[client_id]
            else:
                payments[client_id] = 0.0

        if included:
            total_data = sum(datasizes[client_id] for client_id in included)
            weights = {client_id: datasizes[client_id] / total_data for client_id in included}
            included_models = {client_id: models[client_id] for client_id in included}
            aggregated = weighted_average_ndarrays(included_models, weights)
        else:
            aggregated = self.current_global

        aggregate_seconds = time.perf_counter() - start_time
        self.last_extra = {
            "client_scores": scores_as_strings(compatibilities),
            "detected_free_riders": sorted(self.barred),
            "metadata": {
                "price": self.current_price,
                "total_time": self.total_time,
                "datasizes": scores_as_strings(datasizes),
                "num_participated": len(included),
                "rewards": scores_as_strings(payments),
                "payment_cost": sum(payments.values()),
                "aggregate_seconds": aggregate_seconds,
                "price_seconds": self.price_seconds,
                "calibration_seconds": self.calibration_seconds,
                "server_seconds": aggregate_seconds + self.price_seconds,
            },
        }
        return ndarrays_to_parameters(aggregated), {}

    def _round_extra(self):
        return self.last_extra
