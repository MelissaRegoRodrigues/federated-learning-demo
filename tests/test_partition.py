import numpy as np

from flapp.partition import partition_dirichlet


def _labels(n=1000, num_classes=10, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, num_classes, size=n)


def test_returns_one_index_array_per_client():
    result = partition_dirichlet(_labels(), num_clients=5, alpha=0.5, seed=1)
    assert len(result) == 5


def test_every_sample_assigned_to_exactly_one_client():
    labels = _labels(n=500)
    result = partition_dirichlet(labels, num_clients=4, alpha=0.5, seed=1)
    all_indices = np.concatenate(result)
    assert len(all_indices) == len(labels)
    assert len(set(all_indices.tolist())) == len(labels)  # no duplicates across clients


def test_same_seed_is_deterministic():
    labels = _labels()
    result_a = partition_dirichlet(labels, num_clients=5, alpha=0.5, seed=42)
    result_b = partition_dirichlet(labels, num_clients=5, alpha=0.5, seed=42)
    for a, b in zip(result_a, result_b):
        np.testing.assert_array_equal(a, b)


def test_different_seeds_produce_different_partitions():
    labels = _labels()
    result_a = partition_dirichlet(labels, num_clients=5, alpha=0.5, seed=1)
    result_b = partition_dirichlet(labels, num_clients=5, alpha=0.5, seed=2)
    assert any(
        not np.array_equal(np.sort(a), np.sort(b)) for a, b in zip(result_a, result_b)
    )
