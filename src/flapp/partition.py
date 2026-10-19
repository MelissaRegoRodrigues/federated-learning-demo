import numpy as np
from datasets import Dataset
from flwr_datasets.partitioner import DirichletPartitioner

MIN_PARTITION_SIZE = 10


def partition_dirichlet(labels, num_clients, alpha, seed, min_partition_size=MIN_PARTITION_SIZE):
    partitioner = DirichletPartitioner(
        num_partitions=num_clients,
        partition_by="label",
        alpha=alpha,
        min_partition_size=min_partition_size,
        self_balancing=False,
        shuffle=True,
        seed=seed,
    )
    partitioner.dataset = Dataset.from_dict({"label": labels.tolist(), "index": list(range(len(labels)))})

    indices_per_client = []
    for client_id in range(num_clients):
        indices_per_client.append(np.array(partitioner.load_partition(client_id)["index"]))
    return indices_per_client
