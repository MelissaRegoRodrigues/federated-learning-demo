import functools
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from flapp.partition import partition_dirichlet

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
VALIDATION_FRACTION = 0.1
DATASETS = {
    "mnist": (datasets.MNIST, 0.1307, 0.3081),
    "fashion-mnist": (datasets.FashionMNIST, 0.2860, 0.3530),
}


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def get_params(model):
    params = []
    for value in model.state_dict().values():
        params.append(value.detach().cpu().numpy().copy())
    return params


def set_params(model, params):
    keys = list(model.state_dict().keys())
    state_dict = {}
    for key, value in zip(keys, params):
        state_dict[key] = torch.tensor(value)
    model.load_state_dict(state_dict, strict=True)


def subtract_weights(new_weights, old_weights):
    result = []
    for new_layer, old_layer in zip(new_weights, old_weights):
        result.append(new_layer - old_layer)
    return result


def train(model, loader, epochs, device="cpu", lr=0.01):
    model.to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()


def predict_all(model, loader, device="cpu"):
    model.to(device)
    model.eval()
    all_predictions = []
    all_targets = []
    total_loss = 0.0
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            total_loss += F.cross_entropy(output, target, reduction="sum").item()
            all_predictions.append(output.argmax(dim=1).cpu())
            all_targets.append(target.cpu())
    return all_predictions, all_targets, total_loss


def evaluate_metrics(model, loader, device="cpu"):
    all_predictions, all_targets, total_loss = predict_all(model, loader, device)
    if len(all_predictions) == 0:
        return {"loss": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    predictions = torch.cat(all_predictions).numpy()
    targets = torch.cat(all_targets).numpy()
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, average="macro", zero_division=0
    )
    return {
        "loss": total_loss / len(targets),
        "accuracy": float((predictions == targets).mean()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def make_transform(dataset):
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}. Choose one of {list(DATASETS)}.")
    dataset_class, mean, std = DATASETS[dataset]
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean,), (std,))])


@functools.lru_cache(maxsize=None)
def load_train_set(dataset="mnist"):
    dataset_class = DATASETS[dataset][0]
    return dataset_class(str(DATA_DIR), train=True, download=True, transform=make_transform(dataset))


@functools.lru_cache(maxsize=None)
def load_full_test_set(dataset="mnist"):
    dataset_class = DATASETS[dataset][0]
    return dataset_class(str(DATA_DIR), train=False, download=True, transform=make_transform(dataset))


def validation_indices(seed, fraction=VALIDATION_FRACTION, dataset="mnist"):
    num_train = len(load_train_set(dataset))
    num_validation = int(num_train * fraction)
    rng = np.random.default_rng(seed)
    return rng.choice(num_train, num_validation, replace=False)


def training_indices(seed, dataset="mnist"):
    num_train = len(load_train_set(dataset))
    is_validation = np.zeros(num_train, dtype=bool)
    is_validation[validation_indices(seed, dataset=dataset)] = True
    return np.where(~is_validation)[0]


@functools.lru_cache(maxsize=None)
def partition_all_clients(num_partitions, alpha, seed, dataset="mnist"):
    train_set = load_train_set(dataset)
    available = training_indices(seed, dataset)
    labels = train_set.targets.numpy()[available]
    positions_per_client = partition_dirichlet(labels, num_partitions, alpha, seed)
    indices_per_client = []
    for positions in positions_per_client:
        indices_per_client.append(available[positions])
    return indices_per_client


def load_partition_dataset(partition_id, num_partitions, alpha, seed, dataset="mnist"):
    train_set = load_train_set(dataset)
    indices_per_client = partition_all_clients(num_partitions, alpha, seed, dataset)
    return Subset(train_set, indices_per_client[partition_id])


def make_train_loader(dataset, batch_size, shuffle_seed):
    generator = torch.Generator().manual_seed(shuffle_seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)


def load_test_set(batch_size=512, dataset="mnist"):
    return DataLoader(load_full_test_set(dataset), batch_size=batch_size)


def load_validation_set(fraction=VALIDATION_FRACTION, seed=0, batch_size=512, dataset="mnist"):
    train_set = load_train_set(dataset)
    indices = validation_indices(seed, fraction, dataset)
    return DataLoader(Subset(train_set, indices), batch_size=batch_size)
