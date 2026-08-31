"""SHAPLEY-WEIGHTED AGGREGATION (leave-one-out)

Shapley Value asks: "how much did removing client i change the model's
quality?" Leave-one-out computes this exactly (not Monte Carlo — exact is
affordable at this experiment's client counts, e.g. n=10; see the spec's
non-goals for why Monte Carlo is out of scope):

  1. Aggregate ALL clients' updates -> evaluate -> full_utility
  2. For each client i: aggregate everyone EXCEPT i -> evaluate -> partial_utility
  3. shapley[i] = full_utility - partial_utility
     shapley[i] > 0 => removing client i made the model WORSE (client i helps)
     shapley[i] < 0 => removing client i made the model BETTER (client i is a free-rider!)
  4. Turn shapley values into aggregation weights by SHIFTING them to be
     non-negative (subtract the minimum) and then normalizing to sum to 1
     — the same technique this repo's original (pre-Flower) code used.
     A signed-weight version (dividing raw shapley values by their sum,
     letting free-riders go negative) was tried first and found, during
     full-scale validation, to occasionally amplify a weight enough to
     push the aggregated model somewhere training never recovered from —
     bounding weights to [0, 1] summing to 1 makes this aggregation
     exactly as stable as FedAvg's plain averaging, just re-weighted.
     A free-rider's weight approaches (but never goes below) 0 rather
     than actively reversing its update.

Cost: n+1 model evaluations per round (1 full + n leave-one-out) — this is
the concrete price of Shapley's exactness, worth stating explicitly next
to Stackelberg's O(1)-per-client reputation update (see stackelberg.py).
"""
import numpy as np
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from flapp.strategies.base import CentralEvalStrategy, average_ndarrays, weighted_average_ndarrays

DETECTION_PERCENTILE = 25  # bottom 25% of Shapley values are flagged as free-riders


class ShapleyWeightedStrategy(CentralEvalStrategy):
    def __init__(self, **kwargs):
        super().__init__(strategy_name="shapley", **kwargs)
        self._last_extra: dict = {}

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        # Keyed by the client-reported partition_id, NOT client.cid — Flower's
        # own ClientProxy.cid is an internal simulation-engine id unrelated
        # to the small 0..num_clients-1 ids free-rider behavior was assigned
        # from (see client_app.py's FlowerClient.fit()).
        updates = {
            fit_res.metrics["partition_id"]: parameters_to_ndarrays(fit_res.parameters)
            for client, fit_res in results
        }

        full_params = average_ndarrays(list(updates.values()))
        full_utility = self._evaluate(full_params)

        shapley_values: dict[int, float] = {}
        for cid in updates:
            remaining = {c: u for c, u in updates.items() if c != cid}
            partial_params = average_ndarrays(list(remaining.values())) if remaining else full_params
            partial_utility = self._evaluate(partial_params)
            shapley_values[cid] = full_utility - partial_utility

        # Shift to non-negative, then normalize to sum to 1 — guarantees
        # weights in [0, 1] with no possibility of blow-up, regardless of
        # how the raw shapley values happen to be distributed this round.
        min_contribution = min(shapley_values.values())
        shifted = {cid: v - min_contribution for cid, v in shapley_values.items()}
        total_shifted = sum(shifted.values())
        if total_shifted > 0:
            weights = {cid: v / total_shifted for cid, v in shifted.items()}
        else:
            # All shapley values identical this round (degenerate case,
            # e.g. a single client, or ties) — equal weighting.
            weights = {cid: 1.0 / len(shapley_values) for cid in shapley_values}
        aggregated = weighted_average_ndarrays(updates, weights)

        threshold = np.percentile(list(shapley_values.values()), DETECTION_PERCENTILE)
        detected = [cid for cid, v in shapley_values.items() if v < threshold]

        self._last_extra = {
            "client_scores": {str(cid): float(v) for cid, v in shapley_values.items()},
            "detected_free_riders": detected,
            "metadata": {},
        }
        return ndarrays_to_parameters(aggregated), {}

    def _round_extra(self):
        return self._last_extra

    def _evaluate(self, ndarrays) -> float:
        result = self.evaluate_fn(0, ndarrays, {})
        return result[1]["accuracy"] if result is not None else 0.0
