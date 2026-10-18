import torch
from flwr.common import ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from flapp.strategies import STRATEGIES
from flapp.strategies.base import fit_config
from flapp.task import (
    SimpleCNN,
    evaluate_metrics,
    get_params,
    load_partition_dataset,
    load_test_set,
    load_validation_set,
    set_params,
)


def make_centralized_eval_fn():
    test_loader = load_test_set()

    def evaluate_fn(server_round, ndarrays, config):
        model = SimpleCNN()
        set_params(model, ndarrays)
        metrics = evaluate_metrics(model, test_loader)
        return metrics["loss"], metrics

    return evaluate_fn


def make_shapley_utility_fn(seed):
    validation_loader = load_validation_set(seed=seed)

    def utility_fn(server_round, ndarrays, config):
        model = SimpleCNN()
        set_params(model, ndarrays)
        metrics = evaluate_metrics(model, validation_loader)
        return metrics["loss"], metrics

    return utility_fn


def partition_sizes(num_clients, alpha, seed):
    sizes = {}
    for partition_id in range(num_clients):
        sizes[partition_id] = len(load_partition_dataset(partition_id, num_clients, alpha, seed))
    return sizes


def make_strategy(strategy_name, common_arguments, settings):
    strategy_class = STRATEGIES[strategy_name]
    num_clients = settings["num_clients"]
    seed = settings["seed"]
    if strategy_name == "fedavg":
        return strategy_class(
            **common_arguments,
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            on_fit_config_fn=fit_config,
            min_fit_clients=num_clients,
            min_available_clients=num_clients,
        )
    if strategy_name == "shapley":
        return strategy_class(
            **common_arguments,
            voting_rate=settings["voting_rate"],
            shapley_eval_fn=make_shapley_utility_fn(seed),
        )
    return strategy_class(
        **common_arguments,
        seed=seed,
        dataset_sizes=partition_sizes(num_clients, settings["alpha"], seed),
        num_rounds=settings["num_rounds"],
        compatibility_threshold=settings["compatibility_threshold"],
        model_size=settings["model_size"],
    )


def build_server_app(
    strategy_name,
    num_rounds,
    results_path,
    num_clients,
    num_free_riders,
    seed,
    alpha=0.5,
    voting_rate=0.2,
    compatibility_threshold=0.01,
    model_size=18406,
):
    settings = {
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "seed": seed,
        "alpha": alpha,
        "voting_rate": voting_rate,
        "compatibility_threshold": compatibility_threshold,
        "model_size": model_size,
    }

    def server_fn(context):
        torch.manual_seed(seed)
        initial_parameters = ndarrays_to_parameters(get_params(SimpleCNN()))
        common_arguments = {
            "evaluate_fn": make_centralized_eval_fn(),
            "results_path": results_path,
            "num_clients": num_clients,
            "num_free_riders": num_free_riders,
            "initial_parameters": initial_parameters,
        }
        strategy = make_strategy(strategy_name, common_arguments, settings)
        config = ServerConfig(num_rounds=num_rounds)
        return ServerAppComponents(strategy=strategy, config=config)

    return ServerApp(server_fn=server_fn)
