"""Single source of truth for 'what is the model and what is the data' —
every client and the server's centralized evaluator use exactly this
module, so there is never an ambiguity about which CNN or which MNIST
copy is in play."""
import functools
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from flapp.partition import partition_dirichlet

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_MNIST_TRANSFORM = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)


class SimpleCNN(nn.Module):
    """Same architecture as the repo's original SimpleCNN: 2 conv + 2 fc,
    sized for MNIST's 1x28x28 input."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def get_params(model: nn.Module) -> list[np.ndarray]:
    """Flower's NumPyClient works with plain numpy arrays, not state_dicts."""
    return [value.detach().cpu().numpy() for value in model.state_dict().values()]


def set_params(model: nn.Module, params: list[np.ndarray]) -> None:
    keys = list(model.state_dict().keys())
    state_dict = {k: torch.tensor(v) for k, v in zip(keys, params)}
    model.load_state_dict(state_dict, strict=True)


def train(model: nn.Module, loader, epochs: int, device: str = "cpu") -> None:
    model.to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(epochs):
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()


def test(model: nn.Module, loader, device: str = "cpu") -> float:
    model.to(device)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            pred = model(data).argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    return correct / total if total > 0 else 0.0


@functools.lru_cache(maxsize=1)
def _load_mnist_train():
    return datasets.MNIST(str(DATA_DIR), train=True, download=True, transform=_MNIST_TRANSFORM)


@functools.lru_cache(maxsize=1)
def _load_mnist_test():
    return datasets.MNIST(str(DATA_DIR), train=False, download=True, transform=_MNIST_TRANSFORM)


@functools.lru_cache(maxsize=None)
def load_mnist_partition(
    partition_id: int, num_partitions: int, alpha: float, seed: int, batch_size: int = 32
) -> DataLoader:
    """Each simulated client process calls this independently and gets the
    SAME partitioning (same seed => same np.random.default_rng draws), so
    no cross-process coordination is needed — see partition.py."""
    train_set = _load_mnist_train()
    indices_per_client = partition_dirichlet(
        train_set.targets.numpy(), num_partitions, alpha, seed
    )
    return DataLoader(
        Subset(train_set, indices_per_client[partition_id]), batch_size=batch_size, shuffle=True
    )


def load_test_set(batch_size: int = 64) -> DataLoader:
    return DataLoader(_load_mnist_test(), batch_size=batch_size)
