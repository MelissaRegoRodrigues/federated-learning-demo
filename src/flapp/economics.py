import math

import numpy as np

ENERGY_PRICE = 0.06
LOSS_WEIGHT = 1.0e11
BANDWIDTH = 150000
CHANNEL_GAIN = 50
DISCOUNT = 0.998
COMPUTING_POWER_RANGE = (3000, 5000)
TRANSFER_LIMIT_RANGE = (30, 60)
ENERGY_EFFICIENCY_RANGE = (0.3, 0.5)
PREPROCESSING_RANGE = (0.05, 0.07)


def client_parameters(seed, partition_id):
    rng = np.random.default_rng([seed, partition_id])
    return {
        "computing_power": rng.uniform(*COMPUTING_POWER_RANGE),
        "transfer_limit": rng.uniform(*TRANSFER_LIMIT_RANGE),
        "energy_efficiency": rng.uniform(*ENERGY_EFFICIENCY_RANGE),
        "preprocessing": rng.uniform(*PREPROCESSING_RANGE),
    }


def computing_cost_per_sample(params, model_size):
    return ENERGY_PRICE * params["energy_efficiency"] * model_size / params["computing_power"]


def transmission_cost(params, model_size):
    limit = params["transfer_limit"]
    return BANDWIDTH * (2 ** (limit / BANDWIDTH) - 1) * model_size * ENERGY_PRICE / (limit * CHANNEL_GAIN)


def client_cost(datasize, params, model_size):
    local_cost = params["preprocessing"] * datasize ** 1.5
    computing_cost = computing_cost_per_sample(params, model_size) * datasize
    return local_cost + computing_cost + transmission_cost(params, model_size)


def client_utility(datasize, price, server_round, params, model_size):
    payment = DISCOUNT ** (server_round - 1) * datasize * price
    return payment - client_cost(datasize, params, model_size)


def unconstrained_datasize(price, server_round, params, model_size):
    margin = DISCOUNT ** (server_round - 1) * price - computing_cost_per_sample(params, model_size)
    return 4 / (9 * params["preprocessing"] ** 2) * margin ** 2


def best_datasize(price, server_round, params, model_size, max_datasize):
    candidate = unconstrained_datasize(price, server_round, params, model_size)
    if client_utility(candidate, price, server_round, params, model_size) <= 0:
        return 0
    return int(min(candidate, max_datasize))


def iteration_time(datasize, params, model_size):
    computing_time = datasize * model_size / params["computing_power"]
    transmission_time = model_size / params["transfer_limit"]
    return computing_time + transmission_time


def average_parameters(all_params):
    averages = {}
    for key in all_params[0]:
        values = [params[key] for params in all_params]
        averages[key] = sum(values) / len(values)
    return averages


def price_condition(price, num_clients, total_time, average, slowest, model_size):
    first = (
        -3 * LOSS_WEIGHT * average["preprocessing"] / (2 * math.sqrt(num_clients * total_time))
        * math.sqrt(model_size * slowest["transfer_limit"])
        * price ** -1.5
    )
    second = (
        8 * LOSS_WEIGHT * model_size * (price - computing_cost_per_sample(slowest, model_size))
        / (9 * total_time * slowest["preprocessing"] ** 2 * slowest["computing_power"])
    )
    third = (
        (price - computing_cost_per_sample(average, model_size))
        * 2 * num_clients * total_time * slowest["preprocessing"] ** 2 * slowest["computing_power"]
        / (model_size * average["preprocessing"])
    )
    return first + second + third


def solve_price(num_clients, total_time, average, slowest, model_size):
    low = 1e-9
    high = 1.0
    while price_condition(high, num_clients, total_time, average, slowest, model_size) < 0:
        high = high * 2
    for step in range(200):
        middle = (low + high) / 2
        if price_condition(middle, num_clients, total_time, average, slowest, model_size) < 0:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def slowest_client(price, server_round, params_by_client, max_datasizes, model_size):
    slowest_id = None
    slowest_time = -1.0
    for client_id, params in params_by_client.items():
        datasize = best_datasize(price, server_round, params, model_size, max_datasizes[client_id])
        time = iteration_time(datasize, params, model_size)
        if time > slowest_time:
            slowest_time = time
            slowest_id = client_id
    return slowest_id


def server_price(server_round, total_time, params_by_client, max_datasizes, model_size, max_iterations=50):
    average = average_parameters(list(params_by_client.values()))
    num_clients = len(params_by_client)
    slowest_id = next(iter(params_by_client))
    price = solve_price(num_clients, total_time, average, params_by_client[slowest_id], model_size)
    for iteration in range(max_iterations):
        new_slowest_id = slowest_client(price, server_round, params_by_client, max_datasizes, model_size)
        if new_slowest_id == slowest_id:
            break
        slowest_id = new_slowest_id
        price = solve_price(num_clients, total_time, average, params_by_client[slowest_id], model_size)
    return price


def round_duration(price, server_round, params_by_client, max_datasizes, model_size):
    longest = 0.0
    for client_id, params in params_by_client.items():
        datasize = best_datasize(price, server_round, params, model_size, max_datasizes[client_id])
        longest = max(longest, iteration_time(datasize, params, model_size))
    return longest


def time_for_rounds(total_time, num_rounds, params_by_client, max_datasizes, model_size):
    elapsed = 0.0
    for server_round in range(1, num_rounds + 1):
        price = server_price(server_round, total_time, params_by_client, max_datasizes, model_size)
        elapsed += round_duration(price, server_round, params_by_client, max_datasizes, model_size)
    return elapsed


def calibrate_total_time(num_rounds, params_by_client, max_datasizes, model_size):
    low = 1.0
    high = 1.0
    while time_for_rounds(high, num_rounds, params_by_client, max_datasizes, model_size) > high:
        high = high * 2
    for step in range(100):
        middle = (low + high) / 2
        if time_for_rounds(middle, num_rounds, params_by_client, max_datasizes, model_size) > middle:
            low = middle
        else:
            high = middle
    return high
