import numpy as np

MIN_PARTITION_SIZE = 10


def split_classes_once(labels, num_clients, alpha, rng):
    num_classes = len(np.unique(labels))
    proportions_per_class = rng.dirichlet([alpha] * num_clients, num_classes)

    indices_per_client = []
    for client_id in range(num_clients):
        indices_per_client.append([])

    for k in range(num_classes):
        indices_of_class_k = np.where(labels == k)[0]
        rng.shuffle(indices_of_class_k)
        proportions = proportions_per_class[k]
        split_points = (np.cumsum(proportions) * len(indices_of_class_k)).astype(int)[:-1]
        pieces = np.split(indices_of_class_k, split_points)
        for client_id in range(num_clients):
            indices_per_client[client_id].append(pieces[client_id])

    result = []
    for pieces in indices_per_client:
        result.append(np.concatenate(pieces))
    return result


MAX_ATTEMPTS = 1000


def partition_dirichlet(labels, num_clients, alpha, seed, min_partition_size=MIN_PARTITION_SIZE):
    rng = np.random.default_rng(seed)
    for attempt in range(MAX_ATTEMPTS):
        partitions = split_classes_once(labels, num_clients, alpha, rng)
        smallest = min(len(partition) for partition in partitions)
        if smallest >= min_partition_size:
            return partitions
    raise ValueError(
        f"Could not give every client at least {min_partition_size} samples "
        f"after {MAX_ATTEMPTS} attempts. Use fewer clients or a larger alpha."
    )
