import numpy as np
import torch

from flapp.task import SimpleCNN, evaluate_metrics, get_params, set_params, train


def test_get_params_returns_list_of_ndarrays():
    model = SimpleCNN()
    params = get_params(model)
    assert isinstance(params, list)
    assert all(isinstance(p, np.ndarray) for p in params)
    assert len(params) == len(list(model.state_dict().keys()))


def test_set_params_round_trip():
    model_a = SimpleCNN()
    model_b = SimpleCNN()
    params_a = get_params(model_a)
    set_params(model_b, params_a)
    for p_a, p_b in zip(get_params(model_a), get_params(model_b)):
        np.testing.assert_array_equal(p_a, p_b)


def test_train_changes_weights():
    model = SimpleCNN()
    before = [p.copy() for p in get_params(model)]
    images = torch.randn(16, 1, 28, 28)
    labels = torch.randint(0, 10, (16,))
    loader = [(images, labels)]
    train(model, loader, epochs=1)
    after = get_params(model)
    assert any(not np.array_equal(b, a) for b, a in zip(before, after))


def test_evaluate_metrics_returns_accuracy_precision_recall_f1_and_loss():
    model = SimpleCNN()
    images = torch.randn(16, 1, 28, 28)
    labels = torch.randint(0, 10, (16,))
    loader = [(images, labels)]
    metrics = evaluate_metrics(model, loader)
    assert set(metrics) == {"loss", "accuracy", "precision", "recall", "f1"}
    assert metrics["loss"] > 0.0
    bounded_metrics = {k: v for k, v in metrics.items() if k != "loss"}
    assert all(0.0 <= v <= 1.0 for v in bounded_metrics.values())


def test_validation_never_overlaps_with_any_client_partition():
    from flapp.task import load_partition_dataset, validation_indices

    seed = 42
    val = set(validation_indices(seed).tolist())
    seen_by_clients = set()
    for partition_id in range(10):
        dataset = load_partition_dataset(partition_id, 10, 0.5, seed)
        client = set(dataset.indices.tolist())
        assert not client & val
        assert not client & seen_by_clients
        seen_by_clients |= client

    assert len(seen_by_clients) + len(val) == 60000


def test_get_params_returns_copies_not_views_of_the_model():
    model = SimpleCNN()
    params = get_params(model)
    before = params[0].copy()
    with torch.no_grad():
        for tensor in model.parameters():
            tensor.add_(1.0)
    np.testing.assert_array_equal(params[0], before)
