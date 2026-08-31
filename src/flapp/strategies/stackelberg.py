"""STACKELBERG-STYLE REPUTATION-GATED AGGREGATION

Unlike Shapley (which re-evaluates the model n+1 times every single
round — exact, but expensive), Stackelberg here builds a cheap, CUMULATIVE
reputation signal with zero extra model evaluations, then runs a
price/participation game only among clients that reputation says are
still trustworthy. This is the concrete practical trade-off to state next
to Shapley in the thesis: Shapley = expensive + exact + no memory across
rounds; Stackelberg = cheap + approximate + builds a track record.

Per round:
  1. Compute the coordinate-wise MEDIAN update across all clients — a
     robust "consensus direction" a minority of bad updates can't drag
     around the way a mean can.
  2. For each client, cosine-similarity(its update, the median) is this
     round's quality signal (no model evaluation needed — just linear
     algebra on the updates already in hand).
  3. Update a running reputation (EMA) per client from that signal.
  4. Clients in the bottom 25th percentile of reputation (round 2+ only —
     round 1 has no history to threshold) are excluded from this round's
     aggregation: the server "won't offer them a price."
  5. Among the remaining eligible clients, run a simple price-equilibrium
     search (server picks the price that minimizes its own cost; clients
     "accept" if the price covers their simulated participation cost) —
     this is the actual Stackelberg leader/follower game, layered on top
     of the reputation gate from step 4.
  6. Aggregate (plain mean) over whichever clients are both
     reputation-eligible and price-participating.
"""
import numpy as np
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from flapp.strategies.base import CentralEvalStrategy, average_ndarrays

DETECTION_PERCENTILE = 25
REPUTATION_HISTORY_WEIGHT = 0.7  # how much of the OLD reputation carries forward each round


def _coordinate_median(param_lists: list[list]) -> list:
    return [np.median(np.stack(layer), axis=0) for layer in zip(*param_lists)]


def _flatten(params: list) -> np.ndarray:
    return np.concatenate([p.ravel() for p in params])


def _cosine_similarity(a: list, b: list) -> float:
    flat_a, flat_b = _flatten(a), _flatten(b)
    denom = np.linalg.norm(flat_a) * np.linalg.norm(flat_b)
    if denom == 0:
        return 0.0
    return float(np.dot(flat_a, flat_b) / denom)


def estimate_client_costs(client_ids, rng: np.random.Generator) -> dict:
    """Simulated per-client participation cost (stand-in for real-world
    signals like battery/bandwidth/compute cost — out of scope to model
    precisely here; see spec's non-goals)."""
    return {cid: 0.5 + 1.5 * rng.random() for cid in client_ids}


def calculate_participation_rate(price: float, client_costs: dict):
    participating = [cid for cid, cost in client_costs.items() if price > 0.8 * cost]
    rate = len(participating) / len(client_costs) if client_costs else 0.0
    return rate, participating


def calculate_server_cost(price: float, participation_rate: float, num_clients: int):
    num_participated = max(1, int(participation_rate * num_clients)) if num_clients else 0
    payment_cost = price * num_participated
    compute_cost = num_participated * 0.1
    return payment_cost + compute_cost, payment_cost, compute_cost


def find_stackelberg_equilibrium(client_costs: dict, max_price: float = 3.0) -> float:
    best_price, best_cost = 0.1, float("inf")
    num_clients = len(client_costs)
    for price in np.arange(0.1, max_price, 0.1):
        rate, _ = calculate_participation_rate(price, client_costs)
        cost, _, _ = calculate_server_cost(price, rate, num_clients)
        if cost < best_cost:
            best_cost, best_price = cost, price
    return float(best_price)


class StackelbergStrategy(CentralEvalStrategy):
    def __init__(self, **kwargs):
        super().__init__(strategy_name="stackelberg", **kwargs)
        self.reputation: dict[int, float] = {}
        self._last_extra: dict = {}
        self._rng = np.random.default_rng()

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        # Keyed by the client-reported partition_id, NOT client.cid — see
        # the same note in strategies/shapley.py.
        updates = {
            fit_res.metrics["partition_id"]: parameters_to_ndarrays(fit_res.parameters)
            for client, fit_res in results
        }
        for cid in updates:
            self.reputation.setdefault(cid, 1.0)

        median_update = _coordinate_median(list(updates.values()))
        for cid, update in updates.items():
            similarity = _cosine_similarity(update, median_update)
            self.reputation[cid] = (
                REPUTATION_HISTORY_WEIGHT * self.reputation[cid]
                + (1 - REPUTATION_HISTORY_WEIGHT) * similarity
            )

        if server_round == 1:
            # No history yet to threshold meaningfully — seed reputation
            # this round, detect nothing.
            detected: list[int] = []
            eligible_ids = set(updates.keys())
        else:
            threshold = np.percentile(list(self.reputation.values()), DETECTION_PERCENTILE)
            detected = [cid for cid in updates if self.reputation[cid] < threshold]
            eligible_ids = set(updates.keys()) - set(detected)

        costs = estimate_client_costs(eligible_ids, self._rng)
        price = find_stackelberg_equilibrium(costs)
        participation_rate, participating = calculate_participation_rate(price, costs)
        server_cost, payment_cost, compute_cost = calculate_server_cost(
            price, participation_rate, len(eligible_ids)
        )

        selected_ids = participating if participating else list(eligible_ids)
        aggregated = average_ndarrays([updates[cid] for cid in selected_ids])

        self._last_extra = {
            "client_scores": {str(cid): float(r) for cid, r in self.reputation.items()},
            "detected_free_riders": detected,
            "metadata": {
                "price": price,
                "participation_rate": participation_rate,
                "num_participated": len(selected_ids),
                "server_cost": server_cost,
                "payment_cost": payment_cost,
                "compute_cost": compute_cost,
            },
        }
        return ndarrays_to_parameters(aggregated), {}

    def _round_extra(self):
        return self._last_extra
