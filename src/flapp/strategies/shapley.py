import time
import warnings

from flwr.common import ndarrays_to_parameters

from flapp.strategies.base import (
    CentralEvalStrategy,
    fail_if_clients_missing,
    sample_sizes_by_partition,
    scores_as_strings,
    updates_by_partition,
    weighted_average_ndarrays,
)

VOTING_RATE = 0.2


def data_size_weights(sizes, members):
    total = 0
    for client_id in members:
        total += sizes[client_id]
    weights = {}
    for client_id in members:
        weights[client_id] = sizes[client_id] / total
    return weights


def coalition_model(models, sizes, members):
    member_models = {}
    for client_id in members:
        member_models[client_id] = models[client_id]
    return weighted_average_ndarrays(member_models, data_size_weights(sizes, members))


def lowest_contributions(values, voting_rate):
    ranked = sorted(values, key=lambda client_id: (values[client_id], client_id))
    number_to_flag = int(voting_rate * len(values))
    return ranked[:number_to_flag]


def merged_contributions(client_ids, flagged):
    contributions = {}
    for client_id in client_ids:
        if client_id in flagged:
            contributions[client_id] = 0
        else:
            contributions[client_id] = 1
    return contributions


def contribution_weights(contributions):
    total = sum(contributions.values())
    weights = {}
    for client_id, value in contributions.items():
        weights[client_id] = value / total
    return weights


class ShapleyWeightedStrategy(CentralEvalStrategy):
    def __init__(self, *, voting_rate=VOTING_RATE, shapley_eval_fn=None, **kwargs):
        super().__init__(strategy_name="shapley", **kwargs)
        self.voting_rate = voting_rate
        if shapley_eval_fn is None:
            warnings.warn(
                "ShapleyWeightedStrategy: shapley_eval_fn not provided, using the test set "
                "to compute contributions. Pass shapley_eval_fn=make_shapley_utility_fn(seed).",
                UserWarning,
                stacklevel=2,
            )
        self.shapley_eval_fn = shapley_eval_fn
        self.last_extra = {}

    def evaluate_utility(self, ndarrays):
        if self.shapley_eval_fn is not None:
            evaluate = self.shapley_eval_fn
        else:
            evaluate = self.evaluate_fn
        result = evaluate(0, ndarrays, {})
        if result is None:
            return 0.0
        return result[1]["accuracy"]

    def leave_one_out_values(self, models, sizes):
        everyone = list(models.keys())
        full_utility = self.evaluate_utility(coalition_model(models, sizes, everyone))

        values = {}
        for client_id in everyone:
            others = [other for other in everyone if other != client_id]
            utility_without_client = self.evaluate_utility(coalition_model(models, sizes, others))
            values[client_id] = full_utility - utility_without_client
        return values

    def aggregate_fit(self, server_round, results, failures):
        fail_if_clients_missing(server_round, failures)
        if not results:
            return None, {}
        start_time = time.perf_counter()

        models = updates_by_partition(results)
        sizes = sample_sizes_by_partition(results)
        values = self.leave_one_out_values(models, sizes)

        flagged = lowest_contributions(values, self.voting_rate)
        contributions = merged_contributions(models.keys(), flagged)
        weights = contribution_weights(contributions)
        aggregated = weighted_average_ndarrays(models, weights)

        self.last_extra = {
            "client_scores": scores_as_strings(values),
            "detected_free_riders": flagged,
            "metadata": {
                "rewards": scores_as_strings(weights),
                "aggregate_seconds": time.perf_counter() - start_time,
            },
        }
        return ndarrays_to_parameters(aggregated), {}

    def _round_extra(self):
        return self.last_extra
