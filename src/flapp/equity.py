"""Reward equity: does a strategy's per-round client score (Shapley value
or Stackelberg reputation) actually rank honest clients above free-riders?

This is deliberately a POST-HOC analysis, not something the strategies
compute themselves — a strategy being handed the ground-truth honest/
free-rider label would defeat the point of *detecting* free-riders from
behavior alone. Free-rider identity is reconstructed here only from the
same deterministic rule client_app.py already uses (the first
`num_free_riders` partition ids are free-riders — see free_rider.py's
`assign_behavior`), using nothing the strategy itself had access to.
"""
import numpy as np


def reward_equity_index(client_scores: dict[str, float], num_free_riders: int) -> float | None:
    """Point-biserial correlation between each client's score this round
    and whether it is honest (1) or a free-rider (0).

    +1.0 => scores perfectly rank every honest client above every
            free-rider (ideal equity: reward tracks true contribution)
     0.0 => scores carry no information about who is honest
    -1.0 => scores are inverted (free-riders rewarded more than honest
            clients — the mechanism is actively counterproductive)

    Returns None when there is nothing to correlate against: no
    free-riders in this run (every client is honest, so the ground-truth
    label has no variation), or every client received an identical score.
    """
    if not client_scores:
        return None

    client_ids = sorted(int(cid) for cid in client_scores)
    labels = np.array([0.0 if cid < num_free_riders else 1.0 for cid in client_ids])
    scores = np.array([client_scores[str(cid)] for cid in client_ids])

    if np.std(labels) == 0 or np.std(scores) == 0:
        return None

    return float(np.corrcoef(scores, labels)[0, 1])
