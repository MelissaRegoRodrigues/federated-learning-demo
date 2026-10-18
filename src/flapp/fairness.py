def reference_shares(dataset_sizes, num_free_riders):
    honest_total = 0
    for client_id, size in dataset_sizes.items():
        if client_id >= num_free_riders:
            honest_total += size
    reference = {}
    for client_id, size in dataset_sizes.items():
        if client_id < num_free_riders:
            reference[client_id] = 0.0
        else:
            reference[client_id] = size / honest_total
    return reference


def normalized_shares(values):
    total = sum(values.values())
    shares = {}
    for client_id, value in values.items():
        if total > 0:
            shares[client_id] = value / total
        else:
            shares[client_id] = 0.0
    return shares


def squared_distance(shares, reference):
    total = 0.0
    for client_id, target in reference.items():
        difference = shares.get(client_id, 0.0) - target
        total += difference * difference
    return total
