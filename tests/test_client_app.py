import numpy as np
import torch
from torch.utils.data import TensorDataset

from flapp.client_app import FlowerClient, make_round_seed
from flapp.task import get_params


def make_client(free_rider=False):
    torch.manual_seed(0)
    images = torch.randn(16, 1, 28, 28)
    labels = torch.randint(0, 10, (16,))
    dataset = TensorDataset(images, labels)
    return FlowerClient(dataset, free_rider=free_rider, local_epochs=1, partition_id=1, seed=42, batch_size=4)


def fit_once(server_round):
    client = make_client()
    torch.manual_seed(1)
    start = get_params(client.model)
    params, num_examples, metrics = client.fit(start, {"server_round": server_round})
    return params, num_examples, metrics


def test_fit_reports_partition_id_and_dataset_size():
    params, num_examples, metrics = fit_once(1)
    assert num_examples == 16
    assert metrics == {"partition_id": 1}


def test_fit_is_reproducible_for_the_same_round():
    first, _, _ = fit_once(3)
    second, _, _ = fit_once(3)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)


def test_fit_changes_between_rounds():
    round_one, _, _ = fit_once(1)
    round_two, _, _ = fit_once(2)
    assert any(not np.array_equal(a, b) for a, b in zip(round_one, round_two))


def test_round_seed_is_different_for_each_client_and_round():
    seeds = set()
    for partition_id in range(10):
        for server_round in range(1, 51):
            seeds.add(make_round_seed(42, partition_id, server_round))
    assert len(seeds) == 10 * 50


def test_free_rider_sends_back_the_global_model_without_training():
    client = make_client(free_rider=True)
    global_weights = get_params(client.model)
    sent, num_examples, _ = client.fit(global_weights, {"server_round": 1})
    assert num_examples == 16
    for sent_layer, global_layer in zip(sent, global_weights):
        np.testing.assert_array_equal(sent_layer, global_layer)


def test_honest_client_changes_the_model():
    client = make_client(free_rider=False)
    global_weights = get_params(client.model)
    sent, _, _ = client.fit(global_weights, {"server_round": 1})
    assert any(not np.array_equal(a, b) for a, b in zip(sent, global_weights))


def stackelberg_config(price, barred=""):
    return {"server_round": 1, "price": price, "model_size": 18406, "barred": barred}


def test_with_a_price_the_client_trains_on_the_datasize_it_chooses():
    client = make_client()
    global_weights = get_params(client.model)
    _, datasize, _ = client.fit(global_weights, stackelberg_config(50.0))
    assert datasize == 16
    _, datasize, _ = client.fit(global_weights, stackelberg_config(0.2))
    assert datasize == 0


def test_free_rider_claims_a_datasize_but_sends_the_global_model():
    client = make_client(free_rider=True)
    global_weights = get_params(client.model)
    sent, datasize, _ = client.fit(global_weights, stackelberg_config(50.0))
    assert datasize == 16
    for sent_layer, global_layer in zip(sent, global_weights):
        np.testing.assert_array_equal(sent_layer, global_layer)


def test_barred_client_does_not_participate():
    client = make_client()
    global_weights = get_params(client.model)
    _, datasize, _ = client.fit(global_weights, stackelberg_config(50.0, barred="0,1"))
    assert datasize == 0
