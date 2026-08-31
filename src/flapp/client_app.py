"""Flower ClientApp: each simulated client loads its own MNIST shard,
trains locally, and — if it's a free-rider — corrupts its own update
before returning it. There is no `flwr run`/pyproject run_config here (see
spec's environment findings), so this file exposes a factory that closes
over one experiment's concrete settings instead of reading them from a
Context.run_config that doesn't exist in this launch model."""
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context

from flapp.free_rider import apply_behavior, assign_behavior
from flapp.task import SimpleCNN, get_params, load_mnist_partition, set_params, test, train


class FlowerClient(NumPyClient):
    def __init__(self, loader, behavior: str, local_epochs: int, partition_id: int) -> None:
        self.loader = loader
        self.behavior = behavior
        self.local_epochs = local_epochs
        self.partition_id = partition_id
        self.model = SimpleCNN()

    def get_parameters(self, config):
        return get_params(self.model)

    def fit(self, parameters, config):
        set_params(self.model, parameters)
        train(self.model, self.loader, epochs=self.local_epochs)
        updated = get_params(self.model)
        # Corruption happens here — after real training, right before the
        # update would be sent to the server — matching where real
        # free-riding would occur.
        updated = apply_behavior(updated, self.behavior)
        # Flower's own ClientProxy.cid is an internal simulation-engine id,
        # NOT the small 0..num_clients-1 partition id free-rider behavior
        # was assigned from — reporting partition_id explicitly here is
        # what lets the server-side strategies key client_scores/
        # detected_free_riders by the same id used for free-rider
        # assignment (see strategies/shapley.py, strategies/stackelberg.py).
        return updated, len(self.loader.dataset), {"partition_id": self.partition_id}

    def evaluate(self, parameters, config):
        set_params(self.model, parameters)
        accuracy = test(self.model, self.loader)
        # Local accuracy is reported for curiosity/debugging only — the
        # metric that matters for the thesis is the server's centralized
        # evaluate_fn (see server_app.py), which uses the full MNIST test
        # set rather than one client's small, non-IID shard.
        return 0.0, len(self.loader.dataset), {"accuracy": accuracy}


def build_client_app(alpha: float, seed: int, num_free_riders: int, local_epochs: int) -> ClientApp:
    def client_fn(context: Context) -> Client:
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        loader = load_mnist_partition(partition_id, num_partitions, alpha, seed)
        behavior = assign_behavior(partition_id, num_free_riders)
        return FlowerClient(loader, behavior, local_epochs, partition_id).to_client()

    return ClientApp(client_fn=client_fn)
