import numpy as np
from flwr.client import ClientApp, NumPyClient
from torch.utils.data import Subset

from flapp import economics
from flapp.free_rider import is_free_rider
from flapp.task import SimpleCNN, get_params, load_partition_dataset, make_train_loader, set_params, train


def make_round_seed(seed, partition_id, server_round):
    return seed * 1_000_000 + server_round * 1_000 + partition_id


class FlowerClient(NumPyClient):
    def __init__(
        self,
        dataset,
        free_rider,
        local_epochs,
        partition_id,
        seed=0,
        batch_size=32,
        learning_rate=0.01,
        price_schedule=None,
        model_size=18406,
        free_rider_epochs=None,
    ):
        self.dataset = dataset
        self.free_rider_epochs = free_rider_epochs
        self.price_schedule = price_schedule
        self.model_size = model_size
        self.free_rider = free_rider
        self.local_epochs = local_epochs
        self.partition_id = partition_id
        self.seed = seed
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = SimpleCNN()

    def get_parameters(self, config):
        return get_params(self.model)

    def is_barred(self, config):
        barred_text = config.get("barred", "")
        if barred_text == "":
            return False
        barred_ids = [int(text) for text in barred_text.split(",")]
        return self.partition_id in barred_ids

    def chosen_datasize(self, config):
        if self.price_schedule is not None:
            params = economics.client_parameters(self.seed, self.partition_id)
            price = self.price_schedule[config["server_round"]]
            return economics.best_datasize(price, config["server_round"], params, self.model_size, len(self.dataset))
        if "price" not in config:
            return len(self.dataset)
        params = economics.client_parameters(self.seed, self.partition_id)
        return economics.best_datasize(
            config["price"], config["server_round"], params, config["model_size"], len(self.dataset)
        )

    def training_data(self, datasize, round_seed):
        if datasize == len(self.dataset):
            return self.dataset
        rng = np.random.default_rng(round_seed)
        indices = rng.choice(len(self.dataset), datasize, replace=False)
        return Subset(self.dataset, indices.tolist())

    def fit(self, parameters, config):
        metrics = {"partition_id": self.partition_id}
        if self.is_barred(config):
            return parameters, 0, metrics

        datasize = self.chosen_datasize(config)
        if datasize == 0:
            return parameters, datasize, metrics
        if self.free_rider and self.free_rider_epochs is None:
            return parameters, datasize, metrics

        epochs = self.local_epochs
        if self.free_rider:
            epochs = self.free_rider_epochs

        server_round = config.get("server_round", 0)
        round_seed = make_round_seed(self.seed, self.partition_id, server_round)
        set_params(self.model, parameters)
        loader = make_train_loader(self.training_data(datasize, round_seed), self.batch_size, round_seed)
        train(self.model, loader, epochs=epochs, lr=self.learning_rate)
        return get_params(self.model), datasize, metrics


def build_client_app(
    alpha,
    seed,
    num_free_riders,
    local_epochs,
    batch_size=32,
    learning_rate=0.01,
    price_schedule=None,
    model_size=18406,
    dataset_name="mnist",
    free_rider_epochs=None,
):
    def client_fn(context):
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        dataset = load_partition_dataset(partition_id, num_partitions, alpha, seed, dataset_name)
        client = FlowerClient(
            dataset,
            is_free_rider(partition_id, num_free_riders),
            local_epochs,
            partition_id,
            seed=seed,
            batch_size=batch_size,
            learning_rate=learning_rate,
            price_schedule=price_schedule,
            model_size=model_size,
            free_rider_epochs=free_rider_epochs,
        )
        return client.to_client()

    return ClientApp(client_fn=client_fn)
