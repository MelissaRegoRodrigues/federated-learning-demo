"""Dirichlet-distribution non-IID partitioning. Lower alpha => each client's
label distribution is more skewed (more heterogeneous / more realistic FL);
alpha=0.5 is the value used across this experiment's literature review.

Takes an explicit seed (np.random.default_rng) instead of relying on
global numpy random state, so independent client processes in the
simulation each derive the SAME partitioning without needing to
coordinate with each other.
"""
import numpy as np


def partition_dirichlet(
    labels: np.ndarray, num_clients: int, alpha: float, seed: int
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    num_classes = len(np.unique(labels))
    # label_distribution[class_k, client_i] = share of class k given to client i
    label_distribution = rng.dirichlet([alpha] * num_clients, num_classes)

    client_indices: list[list[np.ndarray]] = [[] for _ in range(num_clients)]
    for class_k in range(num_classes):
        idx_k = np.where(labels == class_k)[0]
        rng.shuffle(idx_k)

        proportions = label_distribution[class_k]
        split_points = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        split_indices = np.split(idx_k, split_points)

        for client_i, indices in enumerate(split_indices):
            client_indices[client_i].append(indices)

    return [np.concatenate(idcs) for idcs in client_indices]
