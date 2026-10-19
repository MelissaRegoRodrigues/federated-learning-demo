import torch
from torch.utils.data import DataLoader, TensorDataset

from flapp import server_app
from flapp.task import SimpleCNN, get_params


def test_make_centralized_eval_fn_returns_real_loss(monkeypatch):
    images = torch.randn(8, 1, 28, 28)
    labels = torch.randint(0, 10, (8,))
    fake_test_loader = DataLoader(TensorDataset(images, labels), batch_size=8)
    monkeypatch.setattr(server_app, "load_test_set", lambda dataset="mnist": fake_test_loader)

    evaluate_fn = server_app.make_centralized_eval_fn()
    model = SimpleCNN()

    loss, metrics = evaluate_fn(1, get_params(model), {})

    assert loss > 0.0
    assert loss == metrics["loss"]
