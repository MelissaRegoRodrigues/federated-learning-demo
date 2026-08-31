import numpy as np
import torch

from flapp.task import SimpleCNN, get_params, set_params, train
from flapp.task import test as evaluate


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
    loader = [(images, labels)]  # a plain iterable of (data, target) batches works for train()
    train(model, loader, epochs=1)
    after = get_params(model)
    assert any(not np.array_equal(b, a) for b, a in zip(before, after))


def test_test_returns_accuracy_in_unit_interval():
    model = SimpleCNN()
    images = torch.randn(16, 1, 28, 28)
    labels = torch.randint(0, 10, (16,))
    loader = [(images, labels)]
    accuracy = evaluate(model, loader)
    assert 0.0 <= accuracy <= 1.0
